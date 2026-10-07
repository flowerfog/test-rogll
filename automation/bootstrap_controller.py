from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automation.common import (  # noqa: E402
    AutomationError,
    JobSpec,
    ProcessLock,
    build_codex_args,
    git,
    load_json,
    load_toml,
    log,
    run_checked,
    write_json,
)


BRANCH_RE = re.compile(r"^(?!/)(?!.*\.\.)(?!.*//)[A-Za-z0-9._/-]+$")
DATA_SUFFIXES = {
    ".csv",
    ".feather",
    ".npy",
    ".npz",
    ".parquet",
    ".png",
    ".pt",
    ".pth",
    ".zip",
}


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


class BootstrapController:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = load_toml(self.config_path)
        self.base_dir = self.config_path.parent
        bootstrap = _section(self.config, "bootstrap")
        source = _section(self.config, "source")
        codex = _section(self.config, "codex")
        execution = _section(self.config, "execution")

        self.state_dir = _path(
            _string(bootstrap, "state_dir", "bootstrap"), self.base_dir
        )
        self.checkout = _path(
            _string(bootstrap, "checkout_dir", "bootstrap"), self.base_dir
        )
        self.local_data_root = _path(
            _string(bootstrap, "local_data_root", "bootstrap"), self.base_dir
        )
        self.autodl_data_root = _string(
            bootstrap, "autodl_data_root", "bootstrap"
        )
        self.manifest_path = str(
            bootstrap.get("manifest_path", "autodl/job.json")
        ).replace("\\", "/")
        self.max_generated_bytes = int(
            bootstrap.get("max_generated_bytes", 10 * 1024 * 1024)
        )
        self.author_name = str(bootstrap.get("author_name", "Codex Bootstrap"))
        self.author_email = str(
            bootstrap.get("author_email", "codex-bootstrap@example.invalid")
        )
        prompt_value = str(
            bootstrap.get(
                "prompt",
                str(Path(__file__).parent / "prompts" / "bootstrap.txt"),
            )
        )
        self.prompt_path = _path(prompt_value, self.base_dir)

        task_files = bootstrap.get("task_files", [])
        if not isinstance(task_files, list) or not all(
            isinstance(item, str) for item in task_files
        ):
            raise AutomationError("bootstrap.task_files must be an array of strings")
        self.task_files = tuple(_path(item, self.base_dir) for item in task_files)

        self.source_url = _string(source, "url", "source")
        self.source_branch = _string(source, "branch", "source")
        if not BRANCH_RE.fullmatch(self.source_branch) or self.source_branch.endswith("/"):
            raise AutomationError(f"unsafe source branch: {self.source_branch!r}")

        command = codex.get("command", ["codex", "exec"])
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise AutomationError("codex.command must be a non-empty array of strings")
        self.codex_command = tuple(command)
        self.codex_model = str(codex.get("model", ""))
        self.codex_timeout = int(codex.get("timeout_seconds", 3600))
        self.codex_ignore_user_config = bool(codex.get("ignore_user_config", False))

        programs = execution.get("allowed_programs", ["python", "python3"])
        env_names = execution.get("allowed_job_env", ["PYTHONUNBUFFERED"])
        if not isinstance(programs, list) or not all(isinstance(x, str) for x in programs):
            raise AutomationError("execution.allowed_programs must be an array")
        if not isinstance(env_names, list) or not all(
            isinstance(x, str) for x in env_names
        ):
            raise AutomationError("execution.allowed_job_env must be an array")
        self.allowed_programs = tuple(programs)
        self.allowed_job_env = tuple(env_names)
        self.max_timeout_seconds = int(execution.get("max_timeout_seconds", 86400))

        self.state_dir.mkdir(parents=True, exist_ok=True)

    def recovery_proposal(self) -> dict[str, Any] | None:
        proposal_path = self.state_dir / "bootstrap-proposal.json"
        if not proposal_path.is_file():
            return None
        proposal = load_json(proposal_path)
        if not isinstance(proposal, dict):
            return None
        reported = proposal.get("changed_files")
        if not isinstance(reported, list) or not all(
            isinstance(item, str) for item in reported
        ):
            return None
        if set(reported) != set(self.changed_files()):
            return None
        return proposal

    def ensure_empty_source_checkout(self) -> bool:
        if not (self.checkout / ".git").exists():
            self.checkout.parent.mkdir(parents=True, exist_ok=True)
            log("cloning empty source repository")
            git(["clone", self.source_url, str(self.checkout)], timeout=900)
        else:
            git(["remote", "set-url", "origin", self.source_url], cwd=self.checkout)

        remote_heads = git(["ls-remote", "--heads", "origin"], cwd=self.checkout)
        if remote_heads:
            raise AutomationError(
                "source repository is not empty; bootstrap is intentionally one-shot"
            )
        try:
            git(["rev-parse", "--verify", "HEAD"], cwd=self.checkout)
        except AutomationError:
            current_branch = git(
                ["symbolic-ref", "--short", "HEAD"], cwd=self.checkout
            )
            if current_branch != self.source_branch:
                git(["switch", "--orphan", self.source_branch], cwd=self.checkout)
        else:
            raise AutomationError("bootstrap checkout unexpectedly has a commit")

        dirty = git(["status", "--porcelain"], cwd=self.checkout)
        if dirty:
            proposal = self.recovery_proposal()
            if proposal is None:
                raise AutomationError(
                    "bootstrap checkout is dirty and does not exactly match the "
                    f"saved recovery proposal:\n{dirty[-2000:]}"
                )
            log("recovering Codex-generated files from an interrupted bootstrap")
            return True
        return False

    @staticmethod
    def csv_profile(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = 0
            null_counts = {column: 0 for column in columns}
            examples: list[dict[str, str]] = []
            for row in reader:
                rows += 1
                if len(examples) < 2:
                    examples.append({key: str(value)[:120] for key, value in row.items()})
                for column in columns:
                    value = row.get(column)
                    if value is None or value.strip().lower() in {"", "nan", "null", "none"}:
                        null_counts[column] += 1
        return {
            "path": str(path),
            "columns": columns,
            "rows": rows,
            "null_counts": null_counts,
            "examples": examples,
        }

    def build_data_inventory(self) -> dict[str, Any]:
        if not self.local_data_root.is_dir():
            raise AutomationError(
                f"local_data_root does not exist: {self.local_data_root}"
            )
        def split_inventory(name: str) -> dict[str, Any]:
            split_root = self.local_data_root / name
            files = [path for path in split_root.rglob("*") if path.is_file()]
            horizontal = sorted(
                path for path in files if path.name.endswith("__horizontal_well.csv")
            )
            typewells = sorted(
                path for path in files if path.name.endswith("__typewell.csv")
            )
            profiles = [
                self.csv_profile(path)
                for path in [
                    horizontal[0] if horizontal else None,
                    typewells[0] if typewells else None,
                ]
                if path is not None
            ]
            return {
                "path": str(split_root),
                "exists": split_root.is_dir(),
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
                "horizontal_well_files": len(horizontal),
                "typewell_files": len(typewells),
                "sample_profiles": profiles,
            }

        sample_submission = self.local_data_root / "sample_submission.csv"
        return {
            "local_data_root": str(self.local_data_root),
            "autodl_data_root": self.autodl_data_root,
            "train": split_inventory("train"),
            "test": split_inventory("test"),
            "sample_submission": (
                self.csv_profile(sample_submission)
                if sample_submission.is_file()
                else None
            ),
            "task_files": [
                {
                    "path": str(path),
                    "exists": path.is_file(),
                    "bytes": path.stat().st_size if path.is_file() else None,
                }
                for path in self.task_files
            ],
        }

    def invoke_codex(
        self,
        inventory: Mapping[str, Any],
        *,
        recovery_error: str = "",
    ) -> dict[str, Any]:
        prompt_template = self.prompt_path.read_text(encoding="utf-8")
        prompt = prompt_template.format(
            branch=self.source_branch,
            manifest_path=self.manifest_path,
            autodl_data_root=self.autodl_data_root,
            inventory_json=json.dumps(inventory, ensure_ascii=False, indent=2),
        )
        if recovery_error:
            prompt += (
                "\n\nRECOVERY MODE\n"
                "A previous bootstrap attempt generated the untracked files in this "
                "checkout but failed validation. Inspect and repair those files in place. "
                "Do not discard a sound baseline. Re-run the required tests and return a "
                "complete proposal whose changed_files exactly matches every untracked "
                "repository file. The previous validation error was:\n"
                + recovery_error
            )
        output_path = self.state_dir / "bootstrap-proposal.json"
        schema_path = Path(__file__).parent / "schemas" / "proposal.schema.json"
        args = build_codex_args(
            self.codex_command,
            ignore_user_config=self.codex_ignore_user_config,
            model=self.codex_model,
        )
        args.extend([
            "-C",
            str(self.checkout),
            "--sandbox",
            "workspace-write",
            "--ephemeral",
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
        ])
        args.append(prompt)
        result = run_checked(
            args,
            cwd=self.checkout,
            timeout=self.codex_timeout,
            env=os.environ.copy(),
        )
        (self.state_dir / "bootstrap-codex.log").write_text(
            result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else ""),
            encoding="utf-8",
        )
        proposal = load_json(output_path)
        if not isinstance(proposal, dict) or proposal.get("decision") != "proposed":
            decision = proposal.get("decision") if isinstance(proposal, dict) else None
            summary = proposal.get("summary") if isinstance(proposal, dict) else None
            raise AutomationError(
                "bootstrap Codex must return decision=proposed; "
                f"received decision={decision!r}, summary={summary!r}"
            )
        return proposal

    def changed_files(self) -> list[str]:
        output = git(
            ["ls-files", "--others", "--exclude-standard"], cwd=self.checkout
        )
        return sorted(line for line in output.splitlines() if line)

    def upgrade_minimal_job_manifest(self) -> bool:
        """Upgrade the three-field manifest emitted by an interrupted early bootstrap."""
        manifest = self.checkout / self.manifest_path
        if not manifest.is_file():
            return False
        raw = load_json(manifest)
        if not isinstance(raw, dict) or set(raw) != {"job_id", "enabled", "command"}:
            return False
        command_value = raw.get("command")
        if not isinstance(command_value, str):
            return False
        try:
            command = shlex.split(command_value, posix=True)
        except ValueError as exc:
            raise AutomationError(
                f"cannot parse minimal job command: {command_value!r}"
            ) from exc
        job_env: dict[str, str] = {}
        if "PYTHONUNBUFFERED" in self.allowed_job_env:
            job_env["PYTHONUNBUFFERED"] = "1"
        if "ROGII_DATA_ROOT" in self.allowed_job_env:
            job_env["ROGII_DATA_ROOT"] = self.autodl_data_root
        upgraded = {
            "version": 1,
            "job_id": raw.get("job_id"),
            "enabled": raw.get("enabled"),
            "command": command,
            "timeout_seconds": min(21600, self.max_timeout_seconds),
            "gpu": "0",
            "artifacts": [],
            "metrics_file": None,
            "notes": "Initial Codex-generated grouped OOF baseline.",
            "env": job_env,
        }
        JobSpec.from_mapping(
            upgraded,
            allowed_programs=self.allowed_programs,
            allowed_env=self.allowed_job_env,
            max_timeout_seconds=self.max_timeout_seconds,
        )
        write_json(manifest, upgraded)
        log("upgraded interrupted three-field job manifest to schema version 1")
        return True

    def validate_generated_files(self, proposal: Mapping[str, Any]) -> JobSpec:
        changed = self.changed_files()
        if not changed:
            raise AutomationError("Codex generated no files")
        reported = proposal.get("changed_files")
        if not isinstance(reported, list) or set(reported) != set(changed):
            raise AutomationError(
                "proposal.changed_files does not match generated repository files"
            )
        protected = [
            path
            for path in changed
            if fnmatch.fnmatch(path, ".github/workflows/**")
            or fnmatch.fnmatch(path, "**/*.key")
            or fnmatch.fnmatch(path, "**/*.pem")
            or fnmatch.fnmatch(path, "**/.env")
        ]
        if protected:
            raise AutomationError("bootstrap generated protected paths: " + ", ".join(protected))
        data_files = [path for path in changed if Path(path).suffix.lower() in DATA_SUFFIXES]
        if data_files:
            raise AutomationError(
                "bootstrap must not commit datasets or generated artifacts: "
                + ", ".join(data_files)
            )
        total = sum((self.checkout / path).stat().st_size for path in changed)
        if total > self.max_generated_bytes:
            raise AutomationError(
                f"generated source is {total} bytes; limit is {self.max_generated_bytes}"
            )
        if "README.md" not in changed:
            raise AutomationError("bootstrap must generate README.md")
        if "requirements.txt" not in changed:
            raise AutomationError("bootstrap must generate requirements.txt")
        if not any(Path(path).suffix == ".py" for path in changed):
            raise AutomationError("bootstrap must generate Python source")
        manifest = self.checkout / self.manifest_path
        if not manifest.is_file():
            raise AutomationError(f"bootstrap did not create {self.manifest_path}")
        raw = load_json(manifest)
        if not isinstance(raw, dict):
            raise AutomationError("bootstrap job manifest must be an object")
        job = JobSpec.from_mapping(
            raw,
            allowed_programs=self.allowed_programs,
            allowed_env=self.allowed_job_env,
            max_timeout_seconds=self.max_timeout_seconds,
        )
        if not job.enabled:
            raise AutomationError("bootstrap job must be enabled")
        if proposal.get("next_job_id") != job.job_id:
            raise AutomationError(
                "proposal.next_job_id must match the generated job manifest"
            )
        if (
            len(job.command) >= 2
            and job.command[0] in {"python", "python3"}
            and job.command[1].endswith(".py")
            and not (self.checkout / job.command[1]).is_file()
        ):
            raise AutomationError(
                f"job command references a missing script: {job.command[1]}"
            )
        return job

    def commit_and_push(self, proposal: Mapping[str, Any], job: JobSpec) -> str:
        git(["add", "-A"], cwd=self.checkout)
        summary = str(proposal.get("summary", "initial baseline")).splitlines()[0][:120]
        git(
            [
                "-c",
                f"user.name={self.author_name}",
                "-c",
                f"user.email={self.author_email}",
                "commit",
                "-m",
                f"[bootstrap] {summary}",
                "-m",
                f"Initial-AutoDL-Job: {job.job_id}",
            ],
            cwd=self.checkout,
        )
        git(
            ["push", "-u", "origin", self.source_branch],
            cwd=self.checkout,
            timeout=900,
        )
        return git(["rev-parse", "HEAD"], cwd=self.checkout)

    def run(self) -> str:
        marker = self.state_dir / "BOOTSTRAP_COMPLETE.json"
        if marker.exists():
            raise AutomationError(f"bootstrap already completed: {marker}")
        with ProcessLock(self.state_dir / "bootstrap.lock"):
            recovering = self.ensure_empty_source_checkout()
            inventory = self.build_data_inventory()
            write_json(self.state_dir / "data-inventory.json", inventory)
            proposal = self.recovery_proposal() if recovering else None
            if proposal is not None:
                if self.upgrade_minimal_job_manifest():
                    proposal = dict(proposal)
                    proposal["decision"] = "proposed"
                    proposal["summary"] = (
                        str(proposal.get("summary", "")).rstrip()
                        + " The bootstrap controller upgraded the interrupted minimal "
                        "job manifest to the validated version 1 schema."
                    ).strip()
                    validation = proposal.get("validation", [])
                    if not isinstance(validation, list):
                        validation = []
                    proposal["validation"] = [
                        *validation,
                        "Validated the upgraded autodl/job.json with JobSpec.",
                    ]
                    write_json(
                        self.state_dir / "bootstrap-proposal.json", proposal
                    )
                try:
                    job = self.validate_generated_files(proposal)
                    log("saved bootstrap proposal is valid; resuming commit and push")
                except AutomationError as exc:
                    log(f"saved bootstrap proposal needs repair: {exc}")
                    proposal = self.invoke_codex(
                        inventory, recovery_error=str(exc)
                    )
                    job = self.validate_generated_files(proposal)
            else:
                proposal = self.invoke_codex(inventory)
                job = self.validate_generated_files(proposal)
            sha = self.commit_and_push(proposal, job)
            write_json(
                marker,
                {
                    "source_sha": sha,
                    "source_branch": self.source_branch,
                    "job_id": job.job_id,
                    "proposal": proposal,
                },
            )
            log(f"bootstrap pushed {sha[:12]} with job {job.job_id}")
            return sha


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and push the first Codex-generated experiment to an empty source repo."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        BootstrapController(args.config).run()
        return 0
    except AutomationError as exc:
        log(f"bootstrap failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
