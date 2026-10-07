"""Apply the frozen V113 consensus center after V69 All773 plus V77 adaptive dose.

Run order:
  1. the complete 6.377 parent pipeline
  2. rawdata_lattice_coarse_all773_kaggle_v69_cell.py (V116-patched)
  3. rawdata_lattice_coarse_adaptive_kaggle_v77_cell.py
  4. this cell

The V69 canonical345 ranker defines the deduplicated top-33 candidate set.
The V77 All773 adaptive endpoint remains the base and supplies the per-well
dose. Rows through 1000 ft are exactly frozen.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


V116_STARTED = time.time()
V116_WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
V116_OUTPUT = V116_WORK / "submission.csv"
V116_NAMED_OUTPUT = V116_WORK / "submission_v116_consensus_center.csv"
V116_BASE_OUTPUT = V116_WORK / "submission_v116_v76_base.csv"
V116_CONTROL_DIR = V116_WORK / "v116_control"
V116_CANDIDATE_DIR = V116_WORK / "v116_candidate"
V116_CONTROL_DIR.mkdir(parents=True, exist_ok=True)
V116_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
V116_SUMMARY = V116_WORK / "v116_consensus_center_summary.json"

TOPK = 33
FREEZE_FT = 1000.0
RAMP_FT = 250.0
CAP_FT = 6.0
ALPHA = 1.0

required_live = (
    "sample", "sample_value_column", "consensus_by_model", "coarse_by_model",
    "v77_output", "v77_output_values", "v77_coarse", "distance", "well_rows",
    "weight_by_well", "bundle",
)
missing_live = [name for name in required_live if name not in globals()]
if missing_live:
    raise RuntimeError(
        "Run the patched V69 cell and then V77 immediately before V116. Missing: "
        + repr(missing_live)
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


ids = sample["id"].astype(str)
base = np.asarray(v77_output_values, np.float64).copy()
coarse = np.asarray(v77_coarse, np.float64)
consensus = ids.map(consensus_by_model["canonical345"]).to_numpy(np.float64)
available = np.isfinite(coarse) & np.isfinite(consensus) & np.isfinite(distance)
if float(available.mean()) < 0.98:
    raise RuntimeError(f"V116 consensus coverage too low: {available.mean():.3%}")

final = base.copy()
active_rows = 0
fallback_wells = []
move_parts = []
for well, positions in well_rows.items():
    selected = positions[available[positions]]
    if not len(selected):
        fallback_wells.append(str(well))
        continue
    dose = float(weight_by_well.get(well, bundle["fixed_weight"]))
    taper = _smoothstep((distance[selected] - FREEZE_FT) / RAMP_FT)
    raw_move = ALPHA * dose * (consensus[selected] - coarse[selected])
    move = taper * np.clip(raw_move, -CAP_FT, CAP_FT)
    final[selected] += move
    active_rows += int(np.sum(np.abs(move) > 1e-12))
    move_parts.append(move)

if not np.isfinite(final).all():
    raise RuntimeError("V116 produced non-finite predictions")
prefix = available & (distance <= FREEZE_FT)
prefix_gap = float(np.max(np.abs(final[prefix] - base[prefix]))) if prefix.any() else 0.0
if prefix_gap != 0.0:
    raise RuntimeError(f"V116 changed the frozen prefix: {prefix_gap}")

base_output = pd.DataFrame({"id": ids, sample_value_column: base})
base_output.to_csv(V116_BASE_OUTPUT, index=False)
base_output.to_csv(V116_CONTROL_DIR / "submission.csv", index=False)
output = sample.copy()
output[sample_value_column] = final
output.to_csv(V116_OUTPUT, index=False)
output.to_csv(V116_NAMED_OUTPUT, index=False)
output.to_csv(V116_CANDIDATE_DIR / "submission.csv", index=False)

all_moves = np.concatenate(move_parts) if move_parts else np.empty(0, np.float64)
summary = {
    "experiment": "rawdata_lattice_consensus_center_kaggle_v116",
    "base": "V69 All773 coarse plus V77 adaptive dose",
    "recipe": {
        "ranking_model": "V69 canonical345",
        "topk": TOPK,
        "aggregator": "uniform_mean",
        "alpha": ALPHA,
        "freeze_ft": FREEZE_FT,
        "ramp_ft": RAMP_FT,
        "cap_ft": CAP_FT,
    },
    "rows": int(len(output)),
    "test_wells": int(len(well_rows)),
    "coverage": float(available.mean()),
    "fallback_wells": fallback_wells,
    "active_rows": active_rows,
    "frozen_prefix_max_abs_difference": prefix_gap,
    "move": {
        "mean_abs_ft": float(np.mean(np.abs(all_moves))) if len(all_moves) else 0.0,
        "p95_abs_ft": float(np.quantile(np.abs(all_moves), 0.95)) if len(all_moves) else 0.0,
        "max_abs_ft": float(np.max(np.abs(all_moves))) if len(all_moves) else 0.0,
    },
    "outputs": {
        "submission": str(V116_OUTPUT),
        "named_copy": str(V116_NAMED_OUTPUT),
        "v76_base": str(V116_BASE_OUTPUT),
        "paired_control": str(V116_CONTROL_DIR / "submission.csv"),
        "paired_candidate": str(V116_CANDIDATE_DIR / "submission.csv"),
    },
    "sha256": _sha256(V116_OUTPUT),
    "elapsed_minutes": (time.time() - V116_STARTED) / 60.0,
    "instruction": "Submit /kaggle/working/submission.csv; no later cell may rewrite it.",
}
V116_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nV116 FINAL SUMMARY")
print(json.dumps(summary, indent=2), flush=True)
