from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automation.common import (  # noqa: E402
    AutomationError,
    ProcessLock,
    StateStore,
    bounded_text,
    build_codex_args,
    git,
    load_json,
    load_toml,
    log,
    run_checked,
    sleep_until_next_poll,
    write_json,
)


BRANCH_RE = re.compile(r"^(?!/)(?!.*\.\.)(?!.*//)[A-Za-z0-9._/-]+$")
FINAL_EVENT_STATES = {"ignored", "analyzed", "proposed", "pushed", "no_change"}


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


class LocalWatcher:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = load_toml(self.config_path)
        self.base_dir = self.config_path.parent

        watcher = _section(self.config, "watcher")
        source = _section(self.config, "source")
        results = _section(self.config, "results")
        codex = _section(self.config, "codex")
        proposal = _section(self.config, "proposal")

        self.poll_seconds = int(watcher.get("poll_seconds", 60))
        if self.poll_seconds < 1:
            raise AutomationError("watcher.poll_seconds must be positive")
        self.state_dir = _path(
            _string(watcher, "state_dir", "watcher"), self.base_dir
        )
        self.ignore_existing = bool(watcher.get("ignore_existing_on_first_run", True))
        self.stop_file = self.state_dir / str(watcher.get("stop_file", "STOP"))

        self.source_url = _string(source, "url", "source")
        self.source_branch = _string(source, "branch", "source")
        self.source_checkout = _path(
            _string(source, "checkout_dir", "source"), self.base_dir
        )
        self._validate_branch(self.source_branch)

        self.results_url = _string(results, "url", "results")
        self.results_branch = _string(results, "branch", "results")
        self.results_checkout = _path(
            _string(results, "checkout_dir", "results"), self.base_dir
        )
        self._validate_branch(self.results_branch)

        command = codex.get("command", ["codex", "exec"])
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise AutomationError("codex.command must be a non-empty array of strings")
        self.codex_command = tuple(command)
        self.codex_model = str(codex.get("model", ""))
        self.codex_timeout = int(codex.get("timeout_seconds", 3600))
        self.codex_ignore_user_config = bool(codex.get("ignore_user_config", False))

        self.mode = str(proposal.get("mode", "analysis"))
        if self.mode not in {"analysis", "edit", "autopush"}:
            raise AutomationError("proposal.mode must be analysis, edit, or autopush")
        self.max_rounds = int(proposal.get("max_rounds", 20))
        statuses = proposal.get("result_statuses", ["succeeded", "failed", "rejected"])
        if not isinstance(statuses, list) or not all(isinstance(x, str) for x in statuses):
            raise AutomationError("proposal.result_statuses must be an array of strings")
        self.result_statuses = set(statuses)
        globs = proposal.get(
            "allowed_change_globs",
            [
                "*.py",
                "**/*.py",
                "*.json",
                "**/*.json",
                "*.md",
                "**/*.md",
                "*.toml",
                "**/*.toml",
                "*.yaml",
                "**/*.yaml",
                "*.yml",
                "**/*.yml",
            ],
        )
        if not isinstance(globs, list) or not all(isinstance(x, str) for x in globs):
            raise AutomationError("proposal.allowed_change_globs must be an array of strings")
        self.allowed_change_globs = tuple(globs)
        denied_globs = proposal.get(
            "denied_change_globs",
            [
                ".git*",
                ".github/workflows/**",
                "automation/autodl_worker.py",
                "automation/local_watcher.py",
                "automation/services/**",
                "automation/gitea/**",
                "**/*.key",
                "**/*.pem",
                "**/.env",
            ],
        )
        if not isinstance(denied_globs, list) or not all(
            isinstance(x, str) for x in denied_globs
        ):
            raise AutomationError("proposal.denied_change_globs must be an array of strings")
        self.denied_change_globs = tuple(denied_globs)
        prompt_value = str(
            proposal.get(
                "prompt",
                str(Path(__file__).parent / "prompts" / "next_experiment.txt"),
            )
        )
        self.prompt_path = _path(prompt_value, self.base_dir)
        self.manifest_path = str(proposal.get("manifest_path", "autodl/job.json"))
        self.author_name = str(proposal.get("author_name", "Local Codex Controller"))
        self.author_email = str(
            proposal.get("author_email", "local-codex@example.invalid")
        )

        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "proposals").mkdir(parents=True, exist_ok=True)
        self.store = StateStore(self.state_dir / "state.sqlite3")

    @staticmethod
    def _validate_branch(branch: str) -> None:
        if not BRANCH_RE.fullmatch(branch) or branch.endswith("/"):
            raise AutomationError(f"unsafe Git branch name: {branch!r}")

    def close(self) -> None:
        self.store.close()

    @staticmethod
    def _ensure_clean(checkout: Path, label: str) -> None:
        dirty = git(["status", "--porcelain"], cwd=checkout)
        if dirty:
            raise AutomationError(
                f"{label} checkout is dirty; use a dedicated clone:\n{dirty[-2000:]}"
            )

    def ensure_checkout(self, *, url: str, branch: str, checkout: Path, label: str) -> str:
        if not (checkout / ".git").exists():
            checkout.parent.mkdir(parents=True, exist_ok=True)
            log(f"cloning dedicated {label} checkout")
            git(
                ["clone", "--branch", branch, "--single-branch", url, str(checkout)],
                timeout=900,
            )
        else:
            self._ensure_clean(checkout, label)
            git(["remote", "set-url", "origin", url], cwd=checkout)
        git(["fetch", "origin", branch], cwd=checkout, timeout=900)
        git(["checkout", branch], cwd=checkout)
        git(["pull", "--ff-only", "origin", branch], cwd=checkout, timeout=900)
        return git(["rev-parse", "HEAD"], cwd=checkout)

    def changed_run_records(self, old_sha: str | None, new_sha: str) -> list[str]:
        if old_sha is None:
            output = git(["ls-tree", "-r", "--name-only", new_sha, "--", "runs"], cwd=self.results_checkout)
        else:
            try:
                git(["cat-file", "-e", f"{old_sha}^{{commit}}"], cwd=self.results_checkout)
                output = git(
                    ["diff", "--name-only", old_sha, new_sha, "--", "runs"],
                    cwd=self.results_checkout,
                )
            except AutomationError:
                output = git(
                    ["ls-tree", "-r", "--name-only", new_sha, "--", "runs"],
                    cwd=self.results_checkout,
                )
        return sorted(
            {
                line.strip()
                for line in output.splitlines()
                if line.strip().endswith("/run.json")
            }
        )

    def result_context(self, run_path: Path) -> str:
        run_dir = run_path.parent
        parts = [
            "RUN.JSON\n" + bounded_text(run_path, 50_000),
            "README\n" + bounded_text(run_dir / "README.md", 50_000),
            "CODE REVIEW\n" + bounded_text(run_dir / "review.json", 50_000),
            "TRAIN LOG\n" + bounded_text(run_dir / "run.log", 120_000),
        ]
        return "\n\n=====\n\n".join(parts)

    def invoke_codex(
        self,
        *,
        event_key: str,
        result_commit: str,
        result_path: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        source_head = self.ensure_checkout(
            url=self.source_url,
            branch=self.source_branch,
            checkout=self.source_checkout,
            label="source",
        )
        prompt_template = self.prompt_path.read_text(encoding="utf-8")
        context = self.result_context(self.results_checkout / result_path)
        prompt = prompt_template.format(
            mode=self.mode,
            source_head=source_head,
            result_commit=result_commit,
            result_path=result_path,
            manifest_path=self.manifest_path,
            result_json=json.dumps(result, ensure_ascii=False, indent=2),
            result_context=context,
        )
        output_path = self.state_dir / "proposals" / f"{event_key}.json"
        schema_path = Path(__file__).parent / "schemas" / "proposal.schema.json"
        sandbox = "read-only" if self.mode == "analysis" else "workspace-write"
        args = build_codex_args(
            self.codex_command,
            ignore_user_config=self.codex_ignore_user_config,
            model=self.codex_model,
        )
        args.extend([
            "-C",
            str(self.source_checkout),
            "--sandbox",
            sandbox,
            "--ephemeral",
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
        ])
        args.append(prompt)
        log(f"invoking local Codex for result {event_key}")
        execution = run_checked(
            args,
            cwd=self.source_checkout,
            timeout=self.codex_timeout,
            env=os.environ.copy(),
        )
        (self.state_dir / "proposals" / f"{event_key}.log").write_text(
            execution.stdout
            + ("\nSTDERR:\n" + execution.stderr if execution.stderr else ""),
            encoding="utf-8",
        )
        proposal = load_json(output_path)
        if not isinstance(proposal, dict) or proposal.get("decision") not in {
            "no_change",
            "proposed",
            "blocked",
        }:
            raise AutomationError("Codex returned an invalid proposal")
        write_json(output_path, proposal)
        return proposal

    def changed_source_files(self) -> list[str]:
        tracked = git(["diff", "--name-only", "HEAD"], cwd=self.source_checkout)
        untracked = git(
            ["ls-files", "--others", "--exclude-standard"], cwd=self.source_checkout
        )
        return sorted({line for line in (tracked + "\n" + untracked).splitlines() if line})

    def validate_changes(
        self,
        changed: Sequence[str],
        proposal: Mapping[str, Any] | None = None,
    ) -> None:
        if not changed:
            raise AutomationError("Codex reported a proposal but made no changes")
        if proposal is not None:
            reported = proposal.get("changed_files")
            if not isinstance(reported, list) or not all(
                isinstance(path, str) for path in reported
            ):
                raise AutomationError("proposal.changed_files must be an array of strings")
            if set(reported) != set(changed):
                raise AutomationError(
                    "proposal.changed_files does not match the actual Git diff"
                )
        denied = [
            path
            for path in changed
            if not any(fnmatch.fnmatch(path, pattern) for pattern in self.allowed_change_globs)
        ]
        if denied:
            raise AutomationError(
                "Codex changed paths outside proposal.allowed_change_globs: "
                + ", ".join(denied)
            )
        protected = [
            path
            for path in changed
            if any(fnmatch.fnmatch(path, pattern) for pattern in self.denied_change_globs)
        ]
        if protected:
            raise AutomationError(
                "Codex changed protected paths: " + ", ".join(protected)
            )
        manifest = self.source_checkout / self.manifest_path
        if not manifest.is_file():
            raise AutomationError(f"proposal must leave a manifest at {self.manifest_path}")
        raw_manifest = load_json(manifest)
        if not isinstance(raw_manifest, dict) or raw_manifest.get("enabled") is not True:
            raise AutomationError("next job manifest must be an enabled JSON object")

    def commit_and_push(self, event_key: str, proposal: Mapping[str, Any]) -> str:
        changed = self.changed_source_files()
        self.validate_changes(changed, proposal)
        remote_head = git(
            ["rev-parse", f"origin/{self.source_branch}"], cwd=self.source_checkout
        )
        local_head = git(["rev-parse", "HEAD"], cwd=self.source_checkout)
        if remote_head != local_head:
            raise AutomationError(
                "source branch advanced during proposal generation; refusing automatic merge"
            )
        git(["add", "-A"], cwd=self.source_checkout)
        summary = str(proposal.get("summary", "next experiment")).splitlines()[0][:120]
        git(
            [
                "-c",
                f"user.name={self.author_name}",
                "-c",
                f"user.email={self.author_email}",
                "commit",
                "-m",
                f"[autolab] {summary}",
                "-m",
                f"Triggered-by-result: {event_key}",
            ],
            cwd=self.source_checkout,
        )
        git(
            ["push", "origin", self.source_branch],
            cwd=self.source_checkout,
            timeout=900,
        )
        return git(["rev-parse", "HEAD"], cwd=self.source_checkout)

    def process_event(self, result_commit: str, result_path: str) -> bool:
        result_file = self.results_checkout / result_path
        result = load_json(result_file)
        if not isinstance(result, dict):
            raise AutomationError(f"result is not an object: {result_path}")
        source_sha = str(result.get("source_sha", ""))
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", source_sha):
            raise AutomationError(f"invalid source_sha in {result_path}")
        event_key = source_sha
        existing = self.store.get_event(event_key)
        if existing and existing["status"] in FINAL_EVENT_STATES:
            return True
        status = str(result.get("status", ""))
        if status not in self.result_statuses:
            self.store.update_event(
                event_key,
                result_commit=result_commit,
                result_path=result_path,
                status="ignored",
                message=f"result status {status!r} is not configured",
            )
            return True

        round_count = int(self.store.get_metadata("round_count") or "0")
        if round_count >= self.max_rounds:
            self.stop_file.write_text(
                f"max_rounds={self.max_rounds} reached at {result_commit}\n",
                encoding="utf-8",
            )
            raise AutomationError(f"maximum autonomous rounds reached: {self.max_rounds}")

        self.store.update_event(
            event_key,
            result_commit=result_commit,
            result_path=result_path,
            status="processing",
        )
        try:
            proposal = self.invoke_codex(
                event_key=event_key,
                result_commit=result_commit,
                result_path=result_path,
                result=result,
            )
            decision = str(proposal["decision"])
            if decision == "blocked":
                raise AutomationError(str(proposal.get("summary", "Codex was blocked")))
            if decision == "no_change":
                changed = self.changed_source_files()
                if changed:
                    raise AutomationError(
                        "Codex returned no_change but modified: " + ", ".join(changed)
                    )
                event_status = "no_change"
                message = str(proposal.get("summary", "no change"))
            elif self.mode == "analysis":
                event_status = "analyzed"
                message = str(proposal.get("summary", "analysis complete"))
            elif self.mode == "edit":
                changed = self.changed_source_files()
                self.validate_changes(changed, proposal)
                event_status = "proposed"
                message = "changes left in dedicated source checkout: " + ", ".join(changed)
            else:
                pushed_sha = self.commit_and_push(event_key, proposal)
                event_status = "pushed"
                message = f"pushed source commit {pushed_sha}"
            self.store.update_event(
                event_key,
                result_commit=result_commit,
                result_path=result_path,
                status=event_status,
                message=message,
            )
            self.store.set_metadata("round_count", str(round_count + 1))
            log(f"local result {event_key} finished as {event_status}")
            return True
        except Exception as exc:
            self.store.update_event(
                event_key,
                result_commit=result_commit,
                result_path=result_path,
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
            )
            log(f"local result {event_key} failed: {type(exc).__name__}: {exc}")
            return False

    def poll_once(self) -> bool:
        if self.stop_file.exists():
            log(f"stop file present: {self.stop_file}")
            return True
        new_sha = self.ensure_checkout(
            url=self.results_url,
            branch=self.results_branch,
            checkout=self.results_checkout,
            label="results",
        )
        old_sha = self.store.get_metadata("results_head")
        if old_sha == new_sha:
            return True
        if old_sha is None and self.ignore_existing:
            self.store.set_metadata("results_head", new_sha)
            log(f"initialized results cursor at {new_sha[:12]}")
            return True

        paths = self.changed_run_records(old_sha, new_sha)
        all_complete = True
        for path in paths:
            if not self.process_event(new_sha, path):
                all_complete = False
        if all_complete:
            self.store.set_metadata("results_head", new_sha)
        return all_complete

    def run_forever(self, *, once: bool = False) -> int:
        with ProcessLock(self.state_dir / "watcher.lock"):
            while True:
                if self.stop_file.exists():
                    log(f"stop file present; watcher exiting: {self.stop_file}")
                    return 0
                started = time.monotonic()
                try:
                    complete = self.poll_once()
                except KeyboardInterrupt:
                    return 130
                except Exception as exc:
                    complete = False
                    log(f"local poll failed: {type(exc).__name__}: {exc}")
                if once:
                    return 0 if complete else 1
                sleep_until_next_poll(started, self.poll_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll experiment results and ask local Codex for the next experiment."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    watcher: LocalWatcher | None = None
    try:
        watcher = LocalWatcher(args.config)
        return watcher.run_forever(once=args.once)
    except AutomationError as exc:
        log(f"configuration error: {exc}")
        return 2
    finally:
        if watcher is not None:
            watcher.close()


if __name__ == "__main__":
    raise SystemExit(main())
