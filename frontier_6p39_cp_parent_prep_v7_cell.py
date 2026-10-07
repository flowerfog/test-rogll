"""Deprecated: this rebased historical CP12 onto the wrong parent."""

raise RuntimeError(
    "Do not run frontier_6p39_cp_parent_prep_v7_cell.py. Use "
    "frontier_6p39_cp12_sidecurve_prep_v8_cell.py after notebook cell 39."
)

from pathlib import Path
import shutil

import pandas as pd


work = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
weight_sp45 = float(globals().get("SP45_BLEND_WEIGHT", 0.60))
named_core = work / f"submission_sp45_learned_w{weight_sp45:.2f}.csv"
if not named_core.exists() and "_final_name" in globals():
    candidate = work / str(globals()["_final_name"])
    if candidate.exists():
        named_core = candidate
if not named_core.exists():
    named_core = work / "submission.csv"
if not named_core.exists():
    raise FileNotFoundError("Run the 6.39 core blend cell first")

core = pd.read_csv(named_core)
if not {"id", "tvt"}.issubset(core.columns) or core["id"].duplicated().any():
    raise RuntimeError(f"Malformed 6.39 core: {named_core}")
core[["id", "tvt"]].to_csv(work / "frontier_6p39_core_before_cp.csv", index=False)
core[["id", "tvt"]].to_csv(work / "submission_before_gold.csv", index=False)

for global_name, filename in (
    ("sub_1", "ridge_pp_submission.csv"),
    ("sub_2", "selector_submission.csv"),
):
    frame = globals().get(global_name)
    if isinstance(frame, pd.DataFrame) and {"id", "tvt"}.issubset(frame.columns):
        frame[["id", "tvt"]].to_csv(work / filename, index=False)

print("frozen 6.39 core for CP12:", named_core, flush=True)
print("next: run the existing CP12 cell, but stop before Gold/guarded overlays", flush=True)
