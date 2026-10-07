"""Export the frozen V168 metadata-only analog selector for Kaggle.

Run after V166 and V167 in the same AutoDL kernel, and after V170 has passed
for the parent endpoint that will be used on Kaggle. The sequence features are
deliberately omitted: V167b proved that all five rankers assigned them zero
gain and zero splits. A single all-773 metadata refit is packaged together
with the historical trajectory library.
"""

from pathlib import Path
import json
import shutil
import time

import joblib
import lightgbm as lgb
import numpy as np


STARTED_BUNDLE_V170 = time.time()
ROOT_BUNDLE_V170 = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OUT_BUNDLE_V170 = ROOT_BUNDLE_V170 / "historical_analog_deployment_bundle_v170"
OUT_BUNDLE_V170.mkdir(parents=True, exist_ok=True)

required_bundle_v170 = (
    "records", "wells", "arrays", "candidate_v167", "params_v167",
    "FEATURE_NAMES_V167", "RELEVANCE_LEVELS", "FAMILIES",
)
missing_bundle_v170 = [name for name in required_bundle_v170 if name not in globals()]
if missing_bundle_v170:
    raise RuntimeError("Run V166 and V167 first. Missing: " + repr(missing_bundle_v170))

verdict_path_v170 = (
    ROOT_BUNDLE_V170 / "oof_exports" / "historical_analog_parent_lineage_v170"
    / "verdict.json"
)
if not verdict_path_v170.is_file():
    raise FileNotFoundError("Run historical_analog_parent_lineage_v170_cell.py first")
verdict_parent_v170 = json.loads(verdict_path_v170.read_text(encoding="utf-8"))
if not verdict_parent_v170.get("strict_pass"):
    raise RuntimeError("V170 parent-lineage audit did not pass; deployment is blocked")
deployable_parents_v170 = verdict_parent_v170.get("deployable_parents", [])
if "nested_sequence_coefficient" not in deployable_parents_v170:
    raise RuntimeError(
        "V170 did not validate the frozen analog move on the exact V144 "
        f"nested_sequence_coefficient parent. Passing parents: {deployable_parents_v170}. "
        "Do not export or deploy this addon."
    )

# The last 12 columns are the metadata block proven by V167b to carry 100% of
# model gain. Refit the same ranking objective on all labelled historical wells.
metadata_names_v170 = list(FEATURE_NAMES_V167[-12:])
ordered_v170 = candidate_v167.sort_values(
    ["well", "reference_well"], kind="stable"
).reset_index(drop=True)
x_v170 = np.stack(ordered_v170["features"].to_numpy()).astype(np.float32)[:, -12:]
groups_v170 = ordered_v170.groupby("well", sort=False).size().to_numpy(np.int32)
params_refit_v170 = dict(params_v167)
params_refit_v170["n_estimators"] = 500
model_v170 = lgb.LGBMRanker(**params_refit_v170, random_state=20260804)
model_v170.fit(
    x_v170,
    ordered_v170["relevance"].to_numpy(np.int32),
    group=groups_v170,
)
model_path_v170 = OUT_BUNDLE_V170 / "metadata_ranker_all773.txt"
model_v170.booster_.save_model(str(model_path_v170))

reference_payload_v170 = {
    "wells": list(wells),
    "arrays": {name: np.asarray(value) for name, value in arrays.items()},
    "records": {
        str(well): {
            "anchor_xy": np.asarray(records[well]["anchor_xy"], np.float32),
            "direction": np.asarray(records[well]["direction"], np.float32),
            "length": float(records[well]["length"]),
            "slopes": np.asarray(records[well]["slopes"], np.float32),
            "post_progress": np.asarray(records[well]["post_progress"], np.float32),
            "post_surface_delta": np.asarray(records[well]["post_surface_delta"], np.float32),
        }
        for well in wells
    },
}
library_path_v170 = OUT_BUNDLE_V170 / "reference_library.joblib"
joblib.dump(reference_payload_v170, library_path_v170, compress=3)

manifest_v170 = {
    "experiment": "historical_analog_deployment_bundle_v170",
    "source_validation": "historical_analog_metadata_fullrow_v168",
    "parent_lineage_audit": verdict_parent_v170,
    "model": model_path_v170.name,
    "reference_library": library_path_v170.name,
    "metadata_features": metadata_names_v170,
    "families": list(FAMILIES),
    "menu_per_family": 64,
    "max_candidates": 256,
    "blend_weight": 0.05,
    "training_wells": int(ordered_v170["well"].nunique()),
    "training_candidates": int(len(ordered_v170)),
    "elapsed_minutes": (time.time() - STARTED_BUNDLE_V170) / 60.0,
}
(OUT_BUNDLE_V170 / "manifest.json").write_text(
    json.dumps(manifest_v170, indent=2, default=str), encoding="utf-8"
)
archive_v170 = shutil.make_archive(
    str(ROOT_BUNDLE_V170 / "historical_analog_deployment_bundle_v170"),
    "zip", ROOT_BUNDLE_V170, OUT_BUNDLE_V170.name,
)
print("V170 DEPLOYMENT BUNDLE")
print(json.dumps(manifest_v170, indent=2, default=str), flush=True)
print("upload:", archive_v170, flush=True)
