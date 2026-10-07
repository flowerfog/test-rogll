from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover - Python 3.11+ is required.
    raise RuntimeError("Python 3.11 or newer is required") from exc


JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
TERMINAL_RUN_STATES = {"ignored", "rejected", "failed", "succeeded"}
CODEX_PROVIDER_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class AutomationError(RuntimeError):
    """Raised when an automation invariant is violated."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def ensure_relative_path(value: str, field: str, *, allow_glob: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AutomationError(f"{field} must be a non-empty string")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or path.parts[0] in {".", ""}:
        raise AutomationError(f"{field} must stay inside the job workspace: {value!r}")
    if not allow_glob and any(token in normalized for token in "*?[]"):
        raise AutomationError(f"{field} must not contain glob characters: {value!r}")
    return normalized


def require_keys(mapping: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise AutomationError(f"unknown {context} keys: {', '.join(sorted(unknown))}")


@dataclass(frozen=True)
class JobSpec:
    version: int
    job_id: str
    enabled: bool
    command: tuple[str, ...]
    timeout_seconds: int
    gpu: str
    artifacts: tuple[str, ...]
    metrics_file: str | None
    notes: str
    env: Mapping[str, str]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        allowed_programs: Sequence[str],
        allowed_env: Sequence[str],
        max_timeout_seconds: int,
    ) -> "JobSpec":
        require_keys(
            raw,
            {
                "version",
                "job_id",
                "enabled",
                "command",
                "timeout_seconds",
                "gpu",
                "artifacts",
                "metrics_file",
                "notes",
                "env",
            },
            "job",
        )
        version = raw.get("version")
        if version != 1:
            raise AutomationError(f"job version must be 1, got {version!r}")

        job_id = raw.get("job_id")
        if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
            raise AutomationError(
                "job_id must match ^[a-z0-9][a-z0-9._-]{0,79}$"
            )

        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise AutomationError("enabled must be a boolean")

        command_value = raw.get("command")
        if (
            not isinstance(command_value, list)
            or not command_value
            or not all(isinstance(item, str) and item for item in command_value)
        ):
            raise AutomationError("command must be a non-empty array of strings")
        command = tuple(command_value)
        program = Path(command[0]).name
        allowed = {Path(item).name for item in allowed_programs}
        if program not in allowed:
            raise AutomationError(
                f"command program {program!r} is not in allowed_programs"
            )

        timeout_seconds = raw.get("timeout_seconds", 3600)
        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or timeout_seconds < 1
            or timeout_seconds > max_timeout_seconds
        ):
            raise AutomationError(
                f"timeout_seconds must be between 1 and {max_timeout_seconds}"
            )

        gpu = raw.get("gpu", "0")
        if not isinstance(gpu, str) or not re.fullmatch(r"\d+(,\d+)*", gpu):
            raise AutomationError("gpu must look like '0' or '0,1'")

        artifact_values = raw.get("artifacts", [])
        if not isinstance(artifact_values, list) or not all(
            isinstance(item, str) for item in artifact_values
        ):
            raise AutomationError("artifacts must be an array of relative path globs")
        artifacts = tuple(
            ensure_relative_path(item, "artifacts[]", allow_glob=True)
            for item in artifact_values
        )

        metrics_value = raw.get("metrics_file")
        metrics_file = (
            None
            if metrics_value in (None, "")
            else ensure_relative_path(metrics_value, "metrics_file")
        )

        notes = raw.get("notes", "")
        if not isinstance(notes, str):
            raise AutomationError("notes must be a string")

        env_value = raw.get("env", {})
        if not isinstance(env_value, dict):
            raise AutomationError("env must be an object")
        env: dict[str, str] = {}
        allowed_env_set = set(allowed_env)
        for key, value in env_value.items():
            if (
                not isinstance(key, str)
                or not ENV_NAME_RE.fullmatch(key)
                or key not in allowed_env_set
            ):
                raise AutomationError(f"job env key {key!r} is not allowed")
            if not isinstance(value, str):
                raise AutomationError(f"job env value for {key!r} must be a string")
            env[key] = value

        return cls(
            version=version,
            job_id=job_id,
            enabled=enabled,
            command=command,
            timeout_seconds=timeout_seconds,
            gpu=gpu,
            artifacts=artifacts,
            metrics_file=metrics_file,
            notes=notes,
            env=env,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "job_id": self.job_id,
            "enabled": self.enabled,
            "command": list(self.command),
            "timeout_seconds": self.timeout_seconds,
            "gpu": self.gpu,
            "artifacts": list(self.artifacts),
            "metrics_file": self.metrics_file,
            "notes": self.notes,
            "env": dict(self.env),
        }


def run_checked(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int = 120,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    resolved_args = list(args)
    if os.name == "nt" and resolved_args:
        executable = resolved_args[0]
        if Path(executable).name.lower() == "codex":
            npm_wrapper = shutil.which("codex.cmd")
            node = shutil.which("node.exe") or shutil.which("node")
            if npm_wrapper and node:
                codex_js = (
                    Path(npm_wrapper).parent
                    / "node_modules"
                    / "@openai"
                    / "codex"
                    / "bin"
                    / "codex.js"
                )
                if codex_js.is_file():
                    resolved_args = [node, str(codex_js), *resolved_args[1:]]
        elif not Path(executable).suffix:
            # CreateProcess does not reliably resolve npm/PowerShell wrappers.
            # Prefer a native executable over a .cmd or PowerShell wrapper.
            native = shutil.which(executable + ".exe")
            if native:
                resolved_args[0] = native
    try:
        return subprocess.run(
            resolved_args,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise AutomationError(f"executable not found: {resolved_args[0]}") from exc
    except PermissionError as exc:
        raise AutomationError(
            f"executable cannot be started: {resolved_args[0]} ({exc})"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AutomationError(
            f"command timed out after {timeout}s: {resolved_args!r}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()[-4000:]
        raise AutomationError(
            f"command failed ({exc.returncode}): {resolved_args!r}\n{detail}"
        ) from exc


def build_codex_args(
    command: Sequence[str],
    *,
    ignore_user_config: bool = False,
    model: str = "",
) -> list[str]:
    """Build a Codex command, optionally using an environment-only provider."""
    args = list(command)
    base_url = os.environ.get("CODEX_PROVIDER_BASE_URL", "").strip()
    effective_model = model.strip()

    if base_url:
        api_key = os.environ.get("CODEX_PROVIDER_API_KEY", "")
        if not api_key.strip():
            raise AutomationError(
                "CODEX_PROVIDER_API_KEY is required when "
                "CODEX_PROVIDER_BASE_URL is set"
            )
        if not re.fullmatch(r"https?://[^\s]+", base_url):
            raise AutomationError(
                "CODEX_PROVIDER_BASE_URL must be an http(s) URL"
            )

        provider_id = os.environ.get("CODEX_PROVIDER_ID", "custom").strip()
        if not CODEX_PROVIDER_ID_RE.fullmatch(provider_id):
            raise AutomationError(
                "CODEX_PROVIDER_ID may contain only letters, numbers, '_' and '-'"
            )
        provider_name = os.environ.get(
            "CODEX_PROVIDER_NAME", "Environment provider"
        ).strip()
        if not provider_name:
            raise AutomationError("CODEX_PROVIDER_NAME must not be empty")

        effective_model = effective_model or os.environ.get(
            "CODEX_PROVIDER_MODEL", ""
        ).strip()
        if not effective_model:
            raise AutomationError(
                "set codex.model or CODEX_PROVIDER_MODEL for the custom provider"
            )

        provider_prefix = f"model_providers.{provider_id}"
        overrides = {
            "model_provider": provider_id,
            f"{provider_prefix}.name": provider_name,
            f"{provider_prefix}.base_url": base_url,
            f"{provider_prefix}.env_key": "CODEX_PROVIDER_API_KEY",
            f"{provider_prefix}.wire_api": "responses",
        }
        for key, value in overrides.items():
            args.extend(["-c", f"{key}={json.dumps(value)}"])
        args.extend(["-c", f"{provider_prefix}.requires_openai_auth=false"])

        reasoning_effort = os.environ.get(
            "CODEX_PROVIDER_REASONING_EFFORT", ""
        ).strip()
        if reasoning_effort:
            args.extend(
                [
                    "-c",
                    "model_reasoning_effort=" + json.dumps(reasoning_effort),
                ]
            )

    if ignore_user_config:
        args.append("--ignore-user-config")
    if effective_model:
        args.extend(["--model", effective_model])
    return args


def git(args: Sequence[str], *, cwd: Path | None = None, timeout: int = 120) -> str:
    result = run_checked(["git", *args], cwd=cwd, timeout=timeout)
    return result.stdout.strip()


class StateStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                sha TEXT PRIMARY KEY,
                job_id TEXT,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                result_commit TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS watcher_events (
                event_key TEXT PRIMARY KEY,
                result_commit TEXT NOT NULL,
                result_path TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get_run(self, sha: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM runs WHERE sha = ?", (sha,)
        ).fetchone()
        return dict(row) if row else None

    def discover_run(self, sha: str) -> dict[str, Any]:
        now = utc_now()
        self.connection.execute(
            """
            INSERT OR IGNORE INTO runs (sha, status, created_at, updated_at)
            VALUES (?, 'discovered', ?, ?)
            """,
            (sha, now, now),
        )
        self.connection.commit()
        result = self.get_run(sha)
        assert result is not None
        return result

    def update_run(
        self,
        sha: str,
        status: str,
        *,
        job_id: str | None = None,
        message: str = "",
        result_commit: str | None = None,
        increment_attempt: bool = False,
    ) -> None:
        now = utc_now()
        self.connection.execute(
            """
            UPDATE runs
            SET status = ?,
                job_id = COALESCE(?, job_id),
                message = ?,
                result_commit = COALESCE(?, result_commit),
                attempt = attempt + ?,
                updated_at = ?
            WHERE sha = ?
            """,
            (
                status,
                job_id,
                message[-8000:],
                result_commit,
                1 if increment_attempt else 0,
                now,
                sha,
            ),
        )
        self.connection.commit()

    def get_event(self, event_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM watcher_events WHERE event_key = ?", (event_key,)
        ).fetchone()
        return dict(row) if row else None

    def update_event(
        self,
        event_key: str,
        *,
        result_commit: str,
        result_path: str,
        status: str,
        message: str = "",
    ) -> None:
        now = utc_now()
        self.connection.execute(
            """
            INSERT INTO watcher_events
                (event_key, result_commit, result_path, status, message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_key) DO UPDATE SET
                status = excluded.status,
                message = excluded.message,
                updated_at = excluded.updated_at
            """,
            (
                event_key,
                result_commit,
                result_path,
                status,
                message[-8000:],
                now,
                now,
            ),
        )
        self.connection.commit()

    def get_metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return str(row[0]) if row else None

    def set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            """
            INSERT INTO metadata (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self.connection.commit()


class ProcessLock:
    """A cross-platform advisory lock for one daemon instance per state directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "ProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        self.handle.seek(0)
        self.handle.write(f"{os.getpid()}\n")
        self.handle.flush()
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise AutomationError(f"another daemon holds {self.path}") from exc
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is None:
            return
        with contextlib.suppress(OSError):
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def bounded_text(path: Path, max_bytes: int = 100_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    data = path.read_bytes()
    if len(data) <= max_bytes:
        return data.decode("utf-8", errors="replace")
    head_size = max_bytes // 3
    tail_size = max_bytes - head_size
    return (
        data[:head_size].decode("utf-8", errors="replace")
        + "\n\n[... content truncated ...]\n\n"
        + data[-tail_size:].decode("utf-8", errors="replace")
    )


def sleep_until_next_poll(started: float, poll_seconds: int) -> None:
    remaining = poll_seconds - (time.monotonic() - started)
    if remaining > 0:
        time.sleep(remaining)


def log(message: str) -> None:
    print(f"{utc_now()} {message}", file=sys.stderr, flush=True)
