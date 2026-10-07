"""V137c: read-only representation audit for the V137b capacity failure."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "absolute_rate_jump_trajectory_v137_cell.py"
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "absolute_rate_jump_representation_audit_v137c"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
source = SOURCE.read_text(encoding="utf-8")
marker = "# Capacity gate: train and evaluate on the same 24 physical wells."
if marker not in source:
    raise RuntimeError(f"V137 capacity marker missing from {SOURCE}")
module = types.ModuleType("absolute_v137c_v137b_definitions")
module.__file__ = str(SOURCE)
sys.modules[module.__name__] = module
exec(compile(source.split(marker, 1)[0] + "\n", str(SOURCE), "exec"), module.__dict__)

indices = module.train_index[:module.CAPACITY_WELLS]
target_grid = module.targets[indices].astype(np.float64)

# The exact 256-node target projected back to every original evaluation row.
grid_oracle_metric, matched, grid_prediction = module._evaluate(
    indices, target_grid, "grid256_oracle"
)

# Maximum representable increment is smooth 5 ft plus jump 24 ft.
increment = np.diff(target_grid, axis=1, prepend=np.zeros((len(target_grid), 1)))
clipped_increment = np.clip(increment, -29.0, 29.0)
clipped_grid = np.cumsum(clipped_increment, axis=1)
clipped_metric, _, clipped_prediction = module._evaluate(
    indices, clipped_grid, "rate_jump_increment_oracle"
)

# A denser grid quantifies whether 256 nodes are the actual bottleneck.
def _dense_oracle(nodes):
    dense_grid = np.linspace(0.0, 1.0, int(nodes))
    parts = []
    predictions = []
    for index in indices:
        well = module.well_order[int(index)]
        local = module._well_frame(well)
        record = module.Q._load_record_relaxed(module.V, well)
        distance = local["md_since"].to_numpy(np.float64)
        progress = distance / max(float(distance[-1]), 1.0)
        z = module.Q._fill(record["z"], record["md"])[record["eval_rows"]]
        reference = float(record["anchor_surface"]) - z
        residual = local[module.TRUTH].to_numpy(np.float64) - reference
        progress_extended = np.concatenate(([0.0], progress))
        residual_extended = np.concatenate(([0.0], residual))
        values = np.interp(dense_grid, progress_extended, residual_extended)
        prediction = reference + np.interp(progress, dense_grid, values)
        parts.append(local)
        predictions.append(prediction)
    local_frame = pd.concat(parts, ignore_index=True)
    prediction = np.concatenate(predictions)
    return module.R._metric(local_frame, prediction, f"grid{nodes}_oracle")


dense_metrics = [_dense_oracle(nodes) for nodes in (128, 256, 384, 512, 768, 1024)]
summary = pd.DataFrame([grid_oracle_metric, clipped_metric, *dense_metrics])
summary = summary.drop_duplicates(subset=["endpoint"], keep="first")
summary.to_csv(OUT_DIR / "representation_summary.csv", index=False)

increment_abs = np.abs(increment[:, 1:])
verdict = {
    "experiment": "absolute_rate_jump_representation_audit_v137c",
    "stage": "read_only_24well_grid_and_increment_oracle",
    "training_count": 0,
    "wells": len(indices),
    "grid256_oracle_rmse": grid_oracle_metric["rmse"],
    "rate_jump_increment_oracle_rmse": clipped_metric["rmse"],
    "increment_abs_p95_ft": float(np.quantile(increment_abs, 0.95)),
    "increment_abs_p99_ft": float(np.quantile(increment_abs, 0.99)),
    "increment_over_29_fraction": float(np.mean(increment_abs > 29.0)),
    "dense_grid_rmse": {
        str(row["endpoint"]): float(row["rmse"]) for row in dense_metrics
    },
    "interpretation": (
        "representation_limited"
        if grid_oracle_metric["rmse"] > 2.0
        else "optimization_or_observability_limited"
    ),
    "next": (
        "Increase native resolution before retraining"
        if grid_oracle_metric["rmse"] > 2.0
        else "Stop architecture scaling; the network cannot infer its representable target"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)
print("\nV137c REPRESENTATION SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV137c FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
