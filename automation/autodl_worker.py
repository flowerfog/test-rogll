from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automation.common import (  # noqa: E402
    AutomationError,
    JobSpec,
    ProcessLock,
    StateStore,
    TERMINAL_RUN_STATES,
    build_codex_args,
    git,
    load_json,
    load_toml,
    log,
    run_checked,
    sleep_until_next_poll,
    utc_now,
    write_json,
)


BRANCH_RE = re.compile(r"^(?!/)(?!.*\.\.)(?!.*//)[A-Za-z0-9._/-]+$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REVIEW_TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class GpuBusyError(AutomationError):
    """Raised when a requested GPU is present but not currently available."""


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise AutomationError(f"missing [{name}] section")
    return value


def _string(section: Mapping[str, Any], key: str, context: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AutomationError(f"{context}.{key} must be a non-empty string")
    return value


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


class AutoDLWorker:
    def __init__(self, config_path: Path, *, dry_run: bool = False) -> None:
        self.config_path = config_path.resolve()
        self.config = load_toml(self.config_path)
        self.base_dir = self.config_path.parent
        self.dry_run = dry_run

        worker = _section(self.config, "worker")
        source = _section(self.config, "source")
        results = _section(self.config, "results")
        execution = _section(self.config, "execution")
        codex = _section(self.config, "codex")

        self.worker_id = str(worker.get("id", "autodl-worker"))
        self.poll_seconds = int(worker.get("poll_seconds", 60))
        if self.poll_seconds < 1:
            raise AutomationError("worker.poll_seconds must be positive")
        self.state_dir = _path(
            _string(worker, "state_dir", "worker"), self.base_dir
        )
        self.work_dir = _path(_string(worker, "work_dir", "worker"), self.base_dir)
        self.manifest_path = str(worker.get("manifest_path", "autodl/job.json")).replace(
            "\\", "/"
        )
        if self.manifest_path.startswith("/") or ".." in Path(self.manifest_path).parts:
            raise AutomationError("worker.manifest_path must be repository-relative")

        self.source_url = _string(source, "url", "source")
        self.source_branch = _string(source, "branch", "source")
        self._validate_branch(self.source_branch)
        self.source_mirror = _path(
            _string(source, "mirror_dir", "source"), self.base_dir
        )

        self.results_url = _string(results, "url", "results")
        self.results_branch = _string(results, "branch", "results")
        self._validate_branch(self.results_branch)
        self.results_checkout = _path(
            _string(results, "checkout_dir", "results"), self.base_dir
        )
        self.git_author_name = str(results.get("author_name", "AutoDL Worker"))
        self.git_author_email = str(
            results.get("author_email", "autodl-worker@example.invalid")
        )

        programs = execution.get("allowed_programs", ["python", "python3"])
        if not isinstance(programs, list) or not all(isinstance(x, str) for x in programs):
            raise AutomationError("execution.allowed_programs must be an array of strings")
        self.allowed_programs = tuple(programs)
        allowed_env = execution.get("allowed_job_env", ["PYTHONUNBUFFERED"])
        if not isinstance(allowed_env, list) or not all(
            isinstance(x, str) for x in allowed_env
        ):
            raise AutomationError("execution.allowed_job_env must be an array of strings")
        self.allowed_job_env = tuple(allowed_env)
        self.max_timeout_seconds = int(execution.get("max_timeout_seconds", 86400))
        self.max_artifact_bytes = int(
            execution.get("max_artifact_bytes", 100 * 1024 * 1024)
        )
        self.max_log_bytes = int(execution.get("max_log_bytes", 100 * 1024 * 1024))
        self.min_free_disk_bytes = int(execution.get("min_free_disk_bytes", 0))
        if self.max_log_bytes < 1024:
            raise AutomationError("execution.max_log_bytes must be at least 1024")
        if self.min_free_disk_bytes < 0:
            raise AutomationError("execution.min_free_disk_bytes cannot be negative")
        self.require_gpu = bool(execution.get("require_gpu", True))
        self.max_gpu_memory_used_mb = int(
            execution.get("max_gpu_memory_used_mb", 1024)
        )
        if self.max_gpu_memory_used_mb < 0:
            raise AutomationError("execution.max_gpu_memory_used_mb cannot be negative")
        inherit_env = execution.get(
            "inherit_env",
            [
                "PATH",
                "HOME",
                "USER",
                "LANG",
                "LC_ALL",
                "TMPDIR",
                "LD_LIBRARY_PATH",
                "CUDA_HOME",
                "CONDA_PREFIX",
                "VIRTUAL_ENV",
                "PYTHONPATH",
            ],
        )
        if not isinstance(inherit_env, list) or not all(
            isinstance(x, str) for x in inherit_env
        ):
            raise AutomationError("execution.inherit_env must be an array of strings")
        self.inherit_env = tuple(inherit_env)
        static_env = execution.get("env", {})
        if not isinstance(static_env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in static_env.items()
        ):
            raise AutomationError("execution.env must be a string-to-string table")
        self.static_env = dict(static_env)

        self.codex_enabled = bool(codex.get("enabled", True))
        command = codex.get("command", ["codex", "exec"])
        if not isinstance(command, list) or not command or not all(
            isinstance(x, str) and x for x in command
        ):
            raise AutomationError("codex.command must be a non-empty array of strings")
        self.codex_command = tuple(command)
        self.codex_model = str(codex.get("model", ""))
        self.codex_timeout_seconds = int(codex.get("timeout_seconds", 1800))
        self.codex_ignore_user_config = bool(codex.get("ignore_user_config", False))
        self.max_review_context_bytes = int(
            codex.get("max_review_context_bytes", 750_000)
        )
        if not 10_000 <= self.max_review_context_bytes <= 1_500_000:
            raise AutomationError(
                "codex.max_review_context_bytes must be between 10000 and 1500000"
            )
        self.retry_sha: str | None = None
        prompt_value = str(
            codex.get(
                "review_prompt",
                str(Path(__file__).parent / "prompts" / "review.txt"),
            )
        )
        self.review_prompt_path = _path(prompt_value, self.base_dir)

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.store = StateStore(self.state_dir / "state.sqlite3")

    @staticmethod
    def _validate_branch(branch: str) -> None:
        if not BRANCH_RE.fullmatch(branch) or branch.endswith("/"):
            raise AutomationError(f"unsafe Git branch name: {branch!r}")

    def close(self) -> None:
        self.store.close()

    def ensure_source_mirror(self) -> None:
        if not (self.source_mirror / "HEAD").exists():
            self.source_mirror.parent.mkdir(parents=True, exist_ok=True)
            log("cloning source mirror")
            git(["clone", "--mirror", self.source_url, str(self.source_mirror)], timeout=900)
        else:
            git(
                [
                    "--git-dir",
                    str(self.source_mirror),
                    "remote",
                    "set-url",
                    "origin",
                    self.source_url,
                ]
            )
        git(
            [
                "--git-dir",
                str(self.source_mirror),
                "fetch",
                "--prune",
                "origin",
                f"+refs/heads/{self.source_branch}:refs/heads/{self.source_branch}",
            ],
            timeout=900,
        )

    def latest_source_sha(self) -> str:
        return git(
            [
                "--git-dir",
                str(self.source_mirror),
                "rev-parse",
                f"refs/heads/{self.source_branch}",
            ]
        )

    def create_worktree(self, sha: str, run_dir: Path) -> Path:
        workspace = run_dir / "repo"
        git(
            [
                "--git-dir",
                str(self.source_mirror),
                "worktree",
                "add",
                "--detach",
                str(workspace),
                sha,
            ],
            timeout=900,
        )
        return workspace

    def remove_worktree(self, workspace: Path) -> None:
        try:
            git(
                [
                    "--git-dir",
                    str(self.source_mirror),
                    "worktree",
                    "remove",
                    "--force",
                    str(workspace),
                ],
                timeout=300,
            )
        except AutomationError as exc:
            log(f"worktree cleanup warning: {exc}")
        try:
            git(
                ["--git-dir", str(self.source_mirror), "worktree", "prune"],
                timeout=120,
            )
        except AutomationError as exc:
            log(f"worktree prune warning: {exc}")

    def read_job(self, workspace: Path) -> JobSpec | None:
        manifest = workspace / Path(self.manifest_path)
        if not manifest.is_file():
            return None
        raw = load_json(manifest)
        if not isinstance(raw, dict):
            raise AutomationError("job manifest root must be an object")
        return JobSpec.from_mapping(
            raw,
            allowed_programs=self.allowed_programs,
            allowed_env=self.allowed_job_env,
            max_timeout_seconds=self.max_timeout_seconds,
        )

    def static_review_context(self, workspace: Path, sha: str) -> str:
        """Create a bounded, non-executing snapshot for container-safe review."""
        tracked = [
            line
            for line in git(["ls-files"], cwd=workspace).splitlines()
            if line
        ]
        changed = git(
            ["diff-tree", "--root", "--no-commit-id", "--name-status", "-r", sha],
            cwd=workspace,
        )
        sections = [
            f"COMMIT: {sha}",
            "CHANGED PATHS:",
            changed or "(none)",
            "TRACKED FILE SNAPSHOT:",
        ]
        used = sum(len(item.encode("utf-8")) for item in sections)
        omitted: list[str] = []
        workspace_root = workspace.resolve()

        for relative_name in tracked:
            path = workspace / relative_name
            if path.is_symlink() or not path.is_file():
                omitted.append(f"{relative_name} (not a regular file)")
                continue
            if path.suffix.lower() not in REVIEW_TEXT_SUFFIXES:
                omitted.append(f"{relative_name} (non-text suffix)")
                continue
            try:
                path.resolve().relative_to(workspace_root)
            except ValueError:
                omitted.append(f"{relative_name} (escapes workspace)")
                continue
            data = path.read_bytes()
            if b"\x00" in data:
                omitted.append(f"{relative_name} (binary content)")
                continue
            content = data.decode("utf-8", errors="replace")
            numbered = "\n".join(
                f"{line_number:6}: {line}"
                for line_number, line in enumerate(content.splitlines(), start=1)
            )
            section = f"\n===== FILE {relative_name} =====\n{numbered}\n"
            section_bytes = len(section.encode("utf-8"))
            if used + section_bytes > self.max_review_context_bytes:
                omitted.append(f"{relative_name} (context size limit)")
                continue
            sections.append(section)
            used += section_bytes

        if omitted:
            sections.extend(["OMITTED PATHS:", *omitted])
        return "\n".join(sections)

    def review_commit(
        self, workspace: Path, run_dir: Path, sha: str, job: JobSpec
    ) -> dict[str, Any]:
        if not self.codex_enabled:
            review = {
                "decision": "approved",
                "summary": "Codex review is disabled by worker configuration.",
                "risk": "high",
                "findings": [],
            }
            write_json(run_dir / "review.json", review)
            return review

        prompt_template = self.review_prompt_path.read_text(encoding="utf-8")
        review_context = self.static_review_context(workspace, sha)
        prompt = prompt_template.format(
            sha=sha,
            manifest_path=self.manifest_path,
            job_json=json.dumps(job.as_dict(), ensure_ascii=False, indent=2),
            review_context=review_context,
        )
        output_path = run_dir / "review.json"
        schema_path = Path(__file__).parent / "schemas" / "review.schema.json"
        args = build_codex_args(
            self.codex_command,
            ignore_user_config=self.codex_ignore_user_config,
            model=self.codex_model,
        )
        args.extend([
            "-C",
            str(workspace),
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
        ])
        args.append(prompt)
        try:
            result = run_checked(
                args,
                cwd=workspace,
                timeout=self.codex_timeout_seconds,
                env=os.environ.copy(),
            )
            (run_dir / "codex.log").write_text(
                result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else ""),
                encoding="utf-8",
            )
        except AutomationError as exc:
            (run_dir / "codex.log").write_text(str(exc) + "\n", encoding="utf-8")
            raise

        review = load_json(output_path)
        if not isinstance(review, dict) or review.get("decision") not in {
            "approved",
            "rejected",
        }:
            raise AutomationError("Codex returned an invalid review result")
        findings = review.get("findings")
        if not isinstance(findings, list):
            raise AutomationError("Codex review findings must be an array")
        if any(item.get("blocking") is True for item in findings if isinstance(item, dict)):
            review["decision"] = "rejected"
        write_json(output_path, review)
        return review

    def preflight_gpu(self, requested_gpu: str) -> str:
        if not self.require_gpu:
            return "GPU preflight disabled"
        result = run_checked(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=30,
        )
        if not result.stdout.strip():
            raise AutomationError("nvidia-smi returned no GPUs")
        requested = set(requested_gpu.split(","))
        found: set[str] = set()
        busy: list[str] = []
        for line in result.stdout.splitlines():
            columns = [column.strip() for column in line.split(",")]
            if len(columns) != 4:
                raise AutomationError(f"unexpected nvidia-smi output: {line!r}")
            index, _, memory_used, _ = columns
            if index not in requested:
                continue
            found.add(index)
            try:
                used_mb = int(memory_used)
            except ValueError as exc:
                raise AutomationError(
                    f"unexpected GPU memory value: {memory_used!r}"
                ) from exc
            if used_mb > self.max_gpu_memory_used_mb:
                busy.append(f"GPU {index} uses {used_mb} MiB")
        missing = requested - found
        if missing:
            raise AutomationError(
                "requested GPU indexes are absent: " + ", ".join(sorted(missing))
            )
        if busy:
            raise GpuBusyError(
                "; ".join(busy)
                + f" (limit {self.max_gpu_memory_used_mb} MiB); retrying next poll"
            )
        return result.stdout.strip()

    def execution_env(self, job: JobSpec) -> dict[str, str]:
        env = {key: os.environ[key] for key in self.inherit_env if key in os.environ}
        env.update(job.env)
        # Worker-owned values are deployment policy and must win over repository jobs.
        env.update(self.static_env)
        env["CUDA_VISIBLE_DEVICES"] = job.gpu
        env["AUTOLAB_JOB_ID"] = job.job_id
        return env

    def run_job(self, workspace: Path, run_dir: Path, job: JobSpec) -> tuple[int, str]:
        gpu_info = self.preflight_gpu(job.gpu)
        (run_dir / "gpu.txt").write_text(gpu_info + "\n", encoding="utf-8")
        log_path = run_dir / "run.log"
        env = self.execution_env(job)
        creation_flags = 0
        start_new_session = os.name != "nt"
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            handle.write(f"started_at={utc_now()}\n")
            handle.write(f"command={json.dumps(list(job.command), ensure_ascii=False)}\n")
            handle.write(f"CUDA_VISIBLE_DEVICES={job.gpu}\n\n")
            handle.flush()
            try:
                process = subprocess.Popen(
                    list(job.command),
                    cwd=workspace,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=start_new_session,
                    creationflags=creation_flags,
                )
            except FileNotFoundError as exc:
                raise AutomationError(f"job executable not found: {job.command[0]}") from exc
            deadline = time.monotonic() + job.timeout_seconds
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    self.terminate_process(process)
                    return 124, f"timed out after {job.timeout_seconds}s"
                handle.flush()
                if log_path.stat().st_size > self.max_log_bytes:
                    self.terminate_process(process)
                    return 125, f"run log exceeded {self.max_log_bytes} bytes"
                if (
                    self.min_free_disk_bytes
                    and shutil.disk_usage(run_dir).free < self.min_free_disk_bytes
                ):
                    self.terminate_process(process)
                    return 126, (
                        "free disk space fell below "
                        f"{self.min_free_disk_bytes} bytes"
                    )
                time.sleep(1)
            return int(process.returncode), "completed"

    @staticmethod
    def terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            process.kill()
            process.wait(timeout=20)
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=20)

    def collect_artifacts(self, workspace: Path, job: JobSpec) -> list[Path]:
        workspace_resolved = workspace.resolve()
        collected: dict[str, Path] = {}
        for pattern in job.artifacts:
            for matched in workspace.glob(pattern):
                candidates = [matched]
                if matched.is_dir():
                    candidates = [path for path in matched.rglob("*") if path.is_file()]
                for candidate in candidates:
                    if candidate.is_symlink() or not candidate.is_file():
                        continue
                    resolved = candidate.resolve()
                    try:
                        relative = resolved.relative_to(workspace_resolved)
                    except ValueError as exc:
                        raise AutomationError(
                            f"artifact escapes workspace: {candidate}"
                        ) from exc
                    collected[relative.as_posix()] = resolved
        total = sum(path.stat().st_size for path in collected.values())
        if total > self.max_artifact_bytes:
            raise AutomationError(
                f"artifacts total {total} bytes exceeds limit {self.max_artifact_bytes}"
            )
        return [collected[key] for key in sorted(collected)]

    def ensure_results_checkout(self) -> None:
        if not (self.results_checkout / ".git").exists():
            self.results_checkout.parent.mkdir(parents=True, exist_ok=True)
            git(
                [
                    "clone",
                    "--branch",
                    self.results_branch,
                    "--single-branch",
                    self.results_url,
                    str(self.results_checkout),
                ],
                timeout=900,
            )
        else:
            git(["remote", "set-url", "origin", self.results_url], cwd=self.results_checkout)
        dirty = git(["status", "--porcelain"], cwd=self.results_checkout)
        if dirty:
            raise AutomationError(
                f"results checkout is dirty; refusing to overwrite it:\n{dirty[-2000:]}"
            )
        git(["fetch", "origin", self.results_branch], cwd=self.results_checkout, timeout=900)
        git(["checkout", self.results_branch], cwd=self.results_checkout)
        git(["pull", "--ff-only", "origin", self.results_branch], cwd=self.results_checkout, timeout=900)

    def prepare_retry(self, sha: str) -> None:
        if not SHA_RE.fullmatch(sha):
            raise AutomationError("--retry-sha must be a full lowercase 40-character SHA")
        self.ensure_source_mirror()
        latest = self.latest_source_sha()
        if sha != latest:
            raise AutomationError(
                f"--retry-sha {sha} is not the latest source SHA {latest}"
            )
        state = self.store.get_run(sha)
        if state is None or state["status"] not in {"failed", "rejected"}:
            status = None if state is None else state["status"]
            raise AutomationError(
                f"--retry-sha requires failed or rejected state, got {status!r}"
            )
        self.retry_sha = sha
        self.store.update_run(
            sha,
            "discovered",
            message=f"explicit retry requested after {state['status']}",
        )
        log(f"explicitly retrying source {sha[:12]} after {state['status']}")

    def publish_result(
        self,
        run_dir: Path,
        workspace: Path,
        record: Mapping[str, Any],
        job: JobSpec | None,
        artifacts: Sequence[Path],
    ) -> str:
        self.ensure_results_checkout()
        sha = str(record["source_sha"])
        job_id = str(record.get("job_id") or "invalid-job")
        relative_destination = Path("runs") / job_id / sha[:12]
        destination = self.results_checkout / relative_destination
        existing_record = destination / "run.json"
        if existing_record.exists():
            existing = load_json(existing_record)
            if isinstance(existing, dict) and existing.get("source_sha") == sha:
                if self.retry_sha != sha:
                    return git(["rev-parse", "HEAD"], cwd=self.results_checkout)
                resolved_destination = destination.resolve()
                resolved_destination.relative_to(self.results_checkout.resolve())
                shutil.rmtree(resolved_destination)
                log(f"replacing prior result for explicit retry {sha[:12]}")
            else:
                raise AutomationError(f"result destination collision: {destination}")
        destination.mkdir(parents=True, exist_ok=False)

        write_json(destination / "run.json", dict(record))
        for name in (
            "review.json",
            "codex.log",
            "run.log",
            "gpu.txt",
            "worker-error.log",
        ):
            source = run_dir / name
            if source.is_file():
                shutil.copy2(source, destination / name)

        workspace_root = workspace.resolve()
        for artifact in artifacts:
            relative = artifact.resolve().relative_to(workspace_root)
            target = destination / "artifacts" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact, target)

        metrics: Any = None
        if job and job.metrics_file:
            metrics_path = workspace / job.metrics_file
            if metrics_path.is_file() and metrics_path.stat().st_size <= 2_000_000:
                try:
                    metrics = load_json(metrics_path)
                except (json.JSONDecodeError, OSError):
                    metrics = None
        self.write_run_readme(destination, record, metrics)

        git(["add", "--", relative_destination.as_posix()], cwd=self.results_checkout)
        commit_message = (
            f"[autodl] {record['status']} {job_id} {sha[:12]}"
        )
        git(
            [
                "-c",
                f"user.name={self.git_author_name}",
                "-c",
                f"user.email={self.git_author_email}",
                "commit",
                "-m",
                commit_message,
            ],
            cwd=self.results_checkout,
        )
        try:
            git(
                ["push", "origin", self.results_branch],
                cwd=self.results_checkout,
                timeout=900,
            )
        except AutomationError:
            git(
                ["pull", "--rebase", "origin", self.results_branch],
                cwd=self.results_checkout,
                timeout=900,
            )
            git(
                ["push", "origin", self.results_branch],
                cwd=self.results_checkout,
                timeout=900,
            )
        return git(["rev-parse", "HEAD"], cwd=self.results_checkout)

    @staticmethod
    def write_run_readme(destination: Path, record: Mapping[str, Any], metrics: Any) -> None:
        lines = [
            f"# {record.get('job_id') or 'Invalid job'}",
            "",
            f"- Status: `{record['status']}`",
            f"- Source SHA: `{record['source_sha']}`",
            f"- Worker: `{record['worker_id']}`",
            f"- Started: `{record['started_at']}`",
            f"- Finished: `{record['finished_at']}`",
            f"- Duration: `{record['duration_seconds']:.1f}s`",
        ]
        if record.get("return_code") is not None:
            lines.append(f"- Return code: `{record['return_code']}`")
        if record.get("message"):
            lines.extend(["", "## Message", "", str(record["message"])])
        if metrics is not None:
            lines.extend(
                [
                    "",
                    "## Metrics",
                    "",
                    "```json",
                    json.dumps(metrics, ensure_ascii=False, indent=2),
                    "```",
                ]
            )
        lines.extend(
            [
                "",
                "## Files",
                "",
                "See `run.json`, `review.json`, `run.log`, and `artifacts/` in this directory.",
                "",
            ]
        )
        (destination / "README.md").write_text("\n".join(lines), encoding="utf-8")

    def process_latest(self) -> str:
        self.ensure_source_mirror()
        sha = self.latest_source_sha()
        state = self.store.discover_run(sha)
        if state["status"] in TERMINAL_RUN_STATES:
            log(f"source {sha[:12]} already finished as {state['status']}")
            return state["status"]
        if state["status"] != "discovered":
            log(f"recovering interrupted source {sha[:12]} from {state['status']}")
            self.store.update_run(sha, "discovered", message="recovered after interruption")

        run_dir = Path(tempfile.mkdtemp(prefix=f"{sha[:12]}-", dir=self.work_dir))
        workspace: Path | None = None
        job: JobSpec | None = None
        started_wall = utc_now()
        started = time.monotonic()
        status = "failed"
        message = ""
        return_code: int | None = None
        review: dict[str, Any] | None = None
        artifacts: list[Path] = []
        published = False
        try:
            workspace = self.create_worktree(sha, run_dir)
            job = self.read_job(workspace)
            if job is None:
                status = "ignored"
                message = f"{self.manifest_path} is absent"
                log(f"ignoring {sha[:12]}: {message}")
            elif not job.enabled:
                status = "ignored"
                message = "job manifest is disabled"
                log(f"ignoring disabled job {job.job_id} at {sha[:12]}")
            elif self.dry_run:
                status = "dry_run"
                message = "manifest validated; review and execution skipped"
            else:
                gpu_info = self.preflight_gpu(job.gpu)
                (run_dir / "gpu.txt").write_text(gpu_info + "\n", encoding="utf-8")
                self.store.update_run(
                    sha,
                    "reviewing",
                    job_id=job.job_id,
                    increment_attempt=True,
                )
                review = self.review_commit(workspace, run_dir, sha, job)
                if review["decision"] != "approved":
                    status = "rejected"
                    message = str(review.get("summary", "Codex rejected the commit"))
                else:
                    self.store.update_run(sha, "running", job_id=job.job_id)
                    return_code, execution_message = self.run_job(workspace, run_dir, job)
                    if return_code == 0:
                        status = "succeeded"
                        message = execution_message
                        artifacts = self.collect_artifacts(workspace, job)
                    else:
                        status = "failed"
                        message = (
                            f"job exited with {return_code}"
                            if execution_message == "completed"
                            else execution_message
                        )
        except GpuBusyError as exc:
            status = "waiting_gpu"
            message = str(exc)
            log(f"deferring {sha[:12]}: {message}")
        except Exception as exc:
            status = "failed"
            message = f"{type(exc).__name__}: {exc}"
            (run_dir / "worker-error.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            log(f"processing {sha[:12]} failed: {message}")

        finished_wall = utc_now()
        record = {
            "schema_version": 1,
            "job_id": job.job_id if job else None,
            "status": status,
            "source_sha": sha,
            "source_branch": self.source_branch,
            "worker_id": self.worker_id,
            "started_at": started_wall,
            "finished_at": finished_wall,
            "duration_seconds": round(time.monotonic() - started, 3),
            "return_code": return_code,
            "message": message,
            "command": list(job.command) if job else None,
            "gpu": job.gpu if job else None,
            "artifact_count": len(artifacts),
            "review_decision": review.get("decision") if review else None,
        }
        write_json(run_dir / "run.json", record)

        should_publish = (
            workspace is not None
            and not self.dry_run
            and status not in {"ignored", "waiting_gpu"}
        )
        if should_publish:
            try:
                self.store.update_run(
                    sha,
                    "publishing",
                    job_id=job.job_id if job else None,
                    message=message,
                )
                result_commit = self.publish_result(
                    run_dir, workspace, record, job, artifacts
                )
                published = True
                self.store.update_run(
                    sha,
                    status,
                    job_id=job.job_id if job else None,
                    message=message,
                    result_commit=result_commit,
                )
                log(
                    f"published {record['job_id'] or 'invalid-job'} {sha[:12]} "
                    f"as {status} at {result_commit[:12]}"
                )
            except Exception as exc:
                publish_message = f"result publish failed: {type(exc).__name__}: {exc}"
                self.store.update_run(
                    sha,
                    "failed",
                    job_id=job.job_id if job else None,
                    message=f"{message}; {publish_message}",
                )
                log(publish_message)
                status = "failed"
        else:
            self.store.update_run(sha, status, message=message)

        if workspace is not None:
            self.remove_worktree(workspace)
        if published or self.dry_run or status in {"ignored", "waiting_gpu"}:
            shutil.rmtree(run_dir, ignore_errors=True)
        else:
            log(f"preserved local run directory {run_dir}")
        return status

    def run_forever(self, *, once: bool = False) -> int:
        with ProcessLock(self.state_dir / "worker.lock"):
            while True:
                started = time.monotonic()
                try:
                    self.process_latest()
                except KeyboardInterrupt:
                    return 130
                except Exception as exc:
                    log(f"poll failed: {type(exc).__name__}: {exc}")
                    if once:
                        return 1
                if once:
                    return 0
                sleep_until_next_poll(started, self.poll_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll a source repository, review commits with Codex, and run AutoDL jobs."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--once", action="store_true", help="process one poll and exit")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the latest manifest without Codex, GPU execution, or publishing",
    )
    parser.add_argument(
        "--retry-sha",
        help="explicitly retry the latest source SHA when its state is failed or rejected",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    worker: AutoDLWorker | None = None
    try:
        worker = AutoDLWorker(args.config, dry_run=args.dry_run)
        if args.retry_sha:
            worker.prepare_retry(args.retry_sha)
        return worker.run_forever(once=args.once)
    except AutomationError as exc:
        log(f"configuration error: {exc}")
        return 2
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
