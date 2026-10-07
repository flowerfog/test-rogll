"""Build a compact evidence matrix from all local experiment verdicts.

This is intentionally read-only with respect to experiment artifacts. It
extracts every RMSE-like scalar together with experiment lineage and verdict,
then highlights sub-6 oracle ceilings whose corresponding experiment failed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


VERDICT_NAMES = {"verdict.json", "result.json", "summary.json"}
RMSE_PATTERN = re.compile(r"(?:^|_)(?:rmse|.*_rmse)$", re.IGNORECASE)


def _scalar(value) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, complex)


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(item, name)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _flatten(item, f"{prefix}[{index}]")
    elif _scalar(value):
        yield prefix, value


def _is_rmse(path: str) -> bool:
    leaf = re.sub(r"\[\d+\]", "", path.rsplit(".", 1)[-1])
    return bool(RMSE_PATTERN.fullmatch(leaf))


def _metric_scope(path: str) -> str:
    lower = path.lower()
    diagnostic_tokens = (
        "near_",
        "mid_",
        "far_",
        "le1000",
        "gt2000",
        "median_well",
        "mean_well",
        "p75_well",
        "p95_well",
        "max_well",
        "parity",
        "gap",
        "improves_",
    )
    if any(token in lower for token in diagnostic_tokens):
        return "diagnostic"
    return "primary"


def _role(path: str) -> str:
    lower = path.lower()
    if "oracle" in lower:
        return "oracle"
    if any(token in lower for token in ("base", "parent", "reference", "incumbent")):
        return "base"
    if any(token in lower for token in ("control", "shuffle", "opposite", "no_gr")):
        return "control"
    return "candidate"


def _oracle_kind(experiment: str, metric_path: str) -> str:
    text = f"{experiment} {metric_path}".lower()
    if any(
        token in text
        for token in (
            "representation",
            "decomposition",
            "error_budget",
            "grid",
            "increment_oracle",
            "capacity",
        )
    ):
        return "representation_ceiling"
    if any(
        token in text
        for token in (
            "perwell",
            "top4",
            "top8",
            "top16",
            "top32",
            "top64",
            "block_oracle",
            "candidate_oracle",
            "label_oracle",
        )
    ):
        return "truth_selected_ceiling"
    return "oracle_ceiling"


def _verdict_state(strict_pass, next_action: str) -> str:
    if strict_pass is True:
        return "passed"
    lower = str(next_action).strip().lower()
    if lower.startswith("pass"):
        return "continuation_requested"
    if any(token in lower for token in ("fail", "close", "stop", "retain")):
        return "explicitly_closed"
    return "needs_manual_lineage_check"


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args()

    root = args.root.resolve()
    output = (args.output or root / "output" / "experiment_evidence_matrix").resolve()
    output.mkdir(parents=True, exist_ok=True)

    artifacts = []
    for directory in (root / "oof_exports", root / "output"):
        if directory.is_dir():
            artifacts.extend(
                path for path in directory.rglob("*.json") if path.name in VERDICT_NAMES
            )

    metric_rows = []
    experiment_rows = []
    seen_experiments = set()
    for path in sorted(set(artifacts)):
        payload = _load(path)
        if not isinstance(payload, dict):
            continue
        experiment = str(payload.get("experiment") or path.parent.name)
        strict_pass = payload.get("strict_pass", payload.get("investment_pass"))
        next_action = payload.get("next", "")
        stage = payload.get("stage", "")
        flattened = list(_flatten(payload))
        rmse_values = []
        for metric_path, value in flattened:
            if not _is_rmse(metric_path):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            rmse_values.append((_role(metric_path), metric_path, numeric))
            metric_rows.append(
                {
                    "experiment": experiment,
                    "stage": stage,
                    "strict_pass": strict_pass,
                    "role": _role(metric_path),
                    "metric_scope": _metric_scope(metric_path),
                    "metric_path": metric_path,
                    "rmse": numeric,
                    "artifact": str(path),
                    "next": next_action,
                }
            )
        if experiment in seen_experiments:
            continue
        seen_experiments.add(experiment)
        oracle = [(value, metric_path) for role, metric_path, value in rmse_values if role == "oracle"]
        primary_oracle = [
            (value, metric_path)
            for role, metric_path, value in rmse_values
            if role == "oracle" and _metric_scope(metric_path) == "primary"
        ]
        candidates = [(value, metric_path) for role, metric_path, value in rmse_values if role == "candidate"]
        bases = [(value, metric_path) for role, metric_path, value in rmse_values if role == "base"]
        best_oracle = min(oracle, default=(None, ""))
        best_primary_oracle = min(primary_oracle, default=(None, ""))
        best_candidate = min(candidates, default=(None, ""))
        best_base = min(bases, default=(None, ""))
        experiment_rows.append(
            {
                "experiment": experiment,
                "stage": stage,
                "strict_pass": strict_pass,
                "verdict_state": _verdict_state(strict_pass, next_action),
                "best_oracle_rmse": best_oracle[0],
                "best_oracle_metric_path": best_oracle[1],
                "oracle_kind": (
                    _oracle_kind(experiment, best_oracle[1]) if best_oracle[0] is not None else ""
                ),
                "best_primary_oracle_rmse": best_primary_oracle[0],
                "best_primary_oracle_metric_path": best_primary_oracle[1],
                "best_candidate_rmse": best_candidate[0],
                "best_candidate_metric_path": best_candidate[1],
                "best_base_rmse": best_base[0],
                "best_base_metric_path": best_base[1],
                "sub6_oracle": bool(best_oracle[0] is not None and best_oracle[0] < 6.0),
                "sub6_primary_oracle": bool(
                    best_primary_oracle[0] is not None and best_primary_oracle[0] < 6.0
                ),
                "artifact": str(path),
                "next": next_action,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    experiments = pd.DataFrame(experiment_rows)
    if metrics.empty or experiments.empty:
        raise RuntimeError(f"No experiment verdict metrics found below {root}")

    metrics = metrics.sort_values(["role", "rmse", "experiment"], kind="stable")
    experiments = experiments.sort_values(
        ["sub6_oracle", "best_oracle_rmse", "best_candidate_rmse"],
        ascending=[False, True, True],
        na_position="last",
        kind="stable",
    )
    metrics.to_csv(output / "all_rmse_metrics.csv", index=False)
    experiments.to_csv(output / "experiment_summary.csv", index=False)

    unresolved = experiments[
        experiments["sub6_primary_oracle"]
        & ~experiments["strict_pass"].fillna(False).astype(bool)
    ].copy()
    unresolved.to_csv(output / "failed_with_sub6_oracle.csv", index=False)
    continuation = unresolved[
        unresolved["verdict_state"].eq("continuation_requested")
    ].copy()
    continuation.to_csv(output / "sub6_oracle_continuation_requested.csv", index=False)

    print(
        "EVIDENCE MATRIX",
        {
            "artifacts": len(set(artifacts)),
            "experiments": len(experiments),
            "rmse_metrics": len(metrics),
            "failed_with_sub6_primary_oracle": len(unresolved),
            "continuation_requested": len(continuation),
            "output": str(output),
        },
    )
    columns = [
        "experiment",
        "best_oracle_rmse",
        "oracle_kind",
        "best_oracle_metric_path",
        "best_primary_oracle_rmse",
        "best_primary_oracle_metric_path",
        "best_candidate_rmse",
        "best_candidate_metric_path",
        "best_base_rmse",
        "verdict_state",
        "next",
    ]
    print("\nFAILED EXPERIMENTS WITH SUB-6 ORACLE")
    print(unresolved[columns].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
