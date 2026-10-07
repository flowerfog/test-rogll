"""V201b: extend the V201 anchor-relative state-axis coverage audit.

V201 exhausted its radius grid at +/-256 ft without satisfying the registered
coverage gate.  This read-only continuation uses the saved truth blocks,
selects a radius on Folds 4/5 only, and reports Fold-3 confirmation coverage.
No model is trained and Folds 1/2 remain untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "truth_block_axis_extension_v201b"
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
SOURCE = OOF / "truth_block_transition_audit_v201" / "truth_blocks.parquet"
OUT = OOF / EXPERIMENT
OUT.mkdir(parents=True, exist_ok=True)
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

BLOCK_SIZE = 128.0
RADII = np.arange(256.0, 1280.0 + 1e-6, 64.0)
SPACINGS = (1.0, 2.0, 4.0, 8.0)
ROW_COVERAGE_GATE = 0.995
P05_WELL_COVERAGE_GATE = 0.95


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


blocks = pd.read_parquet(SOURCE)
required = {"well", "fold", "block_size", "state_residual"}
missing = sorted(required.difference(blocks.columns))
if missing:
    raise RuntimeError(f"V201 truth blocks missing columns: {missing}")
blocks["well"] = blocks["well"].astype(str)
blocks = blocks.loc[blocks["block_size"].eq(BLOCK_SIZE) & blocks["fold"].isin([3, 4, 5])].copy()
if blocks["well"].nunique() != 463:
    raise RuntimeError(f"Expected 463 Fold345 wells, found {blocks['well'].nunique()}")

rows = []
for radius in RADII:
    row = {"axis_radius_ft": radius}
    for label, fold_mask in (
        ("discovery", blocks["fold"].isin([4, 5])),
        ("confirmation", blocks["fold"].eq(3)),
    ):
        local = blocks.loc[fold_mask]
        covered = np.abs(local["state_residual"].to_numpy(np.float64)) <= radius
        per_well = pd.Series(covered).groupby(local["well"].to_numpy()).mean()
        row[f"{label}_row_coverage"] = float(np.mean(covered))
        row[f"{label}_minimum_well_coverage"] = float(per_well.min())
        row[f"{label}_p05_well_coverage"] = float(per_well.quantile(0.05))
        row[f"{label}_complete_well_fraction"] = float(np.mean(per_well >= 1.0 - 1e-12))
    rows.append(row)

coverage = pd.DataFrame(rows)
eligible = coverage.loc[
    coverage["discovery_row_coverage"].ge(ROW_COVERAGE_GATE)
    & coverage["discovery_p05_well_coverage"].ge(P05_WELL_COVERAGE_GATE)
]
selected_radius = float(eligible.iloc[0]["axis_radius_ft"]) if len(eligible) else None

complexity_rows = []
if selected_radius is not None:
    for spacing in SPACINGS:
        states = int(round(2.0 * selected_radius / spacing)) + 1
        complexity_rows.append({
            "axis_radius_ft": selected_radius,
            "state_spacing_ft": spacing,
            "states": states,
            "unary_values_per_48block_well": 48 * states,
            "float32_mb_per_48block_well": 48 * states * 4 / 1024**2,
        })
complexity = pd.DataFrame(complexity_rows)

discovery = blocks.loc[blocks["fold"].isin([4, 5]), "state_residual"].abs().to_numpy(np.float64)
confirmation = blocks.loc[blocks["fold"].eq(3), "state_residual"].abs().to_numpy(np.float64)
quantiles = pd.DataFrame([
    {
        "split": label,
        **{f"abs_residual_q{str(q).replace('.', 'p')}": float(np.quantile(values, q)) for q in (0.95, 0.99, 0.995, 0.999, 1.0)},
    }
    for label, values in (("discovery", discovery), ("confirmation", confirmation))
])

coverage.to_csv(OUT / "axis_coverage_extended.csv", index=False)
complexity.to_csv(OUT / "state_complexity.csv", index=False)
quantiles.to_csv(OUT / "residual_quantiles.csv", index=False)

confirmation_row_coverage = None
confirmation_p05 = None
if selected_radius is not None:
    selected_row = coverage.loc[coverage["axis_radius_ft"].eq(selected_radius)].iloc[0]
    confirmation_row_coverage = float(selected_row["confirmation_row_coverage"])
    confirmation_p05 = float(selected_row["confirmation_p05_well_coverage"])

verdict = {
    "experiment": EXPERIMENT,
    "stage": "V201_128ft_state_axis_extension_Fold45_selection_Fold3_confirmation",
    "source": SOURCE,
    "block_size_ft": BLOCK_SIZE,
    "selected_radius_ft": selected_radius,
    "confirmation_row_coverage": confirmation_row_coverage,
    "confirmation_p05_well_coverage": confirmation_p05,
    "checks": {
        "discovery_axis_found": selected_radius is not None,
        "confirmation_row_coverage_ge_0p995": bool(confirmation_row_coverage is not None and confirmation_row_coverage >= 0.995),
        "confirmation_p05_well_coverage_ge_0p95": bool(confirmation_p05 is not None and confirmation_p05 >= 0.95),
    },
    "next": (
        "Freeze the selected radius; use 4-ft states for the V202 gateway and continuous local refinement after decoding."
        if selected_radius is not None and confirmation_row_coverage >= 0.995 and confirmation_p05 >= 0.95
        else "Do not build a fixed-axis CRF; replace the linear tangent centre with a legal adaptive centre first."
    ),
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8")

print("V201b EXTENDED AXIS COVERAGE")
print(coverage.to_string(index=False), flush=True)
print("\nV201b RESIDUAL QUANTILES")
print(quantiles.to_string(index=False), flush=True)
print("\nV201b STATE COMPLEXITY")
print(complexity.to_string(index=False), flush=True)
print("\nV201b FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT, flush=True)
