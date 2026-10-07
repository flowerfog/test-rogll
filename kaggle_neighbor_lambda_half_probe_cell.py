"""Final Kaggle cell for the frozen Neighbor lambda=0.5 score probe.

Required order in the same notebook run:
1. CP12 -> Gold writes submission.csv.
2. Save that file as submission_before_neighbor.csv, then run the exact
   LB-confirmed Neighbor v0 cell. Its resulting submission.csv is lambda=1.
3. This cell halves that exact frozen Neighbor increment and overwrites only
   the final submission.csv.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


WORK = Path("/kaggle/working")
LAMBDA_NEIGHBOR = 0.50
BEFORE_PATH = WORK / "submission_before_neighbor.csv"
AFTER_PATH = WORK / "submission.csv"
ORIGINAL_AUDIT_PATH = WORK / "neighbor_structure_submit_audit.json"
OUTPUT_PATH = WORK / "submission_neighbor_lambda_0p5.csv"
FINAL_PATH = WORK / "submission.csv"
LAMBDA_ONE_SNAPSHOT = WORK / "submission_neighbor_lambda_1_snapshot.csv"


def _normalise(path):
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "id" not in frame.columns:
        raise RuntimeError(f"missing id column: {path}")
    prediction_columns = [column for column in frame.columns if column != "id"]
    if "tvt" in prediction_columns:
        prediction_column = "tvt"
    elif len(prediction_columns) == 1:
        prediction_column = prediction_columns[0]
    else:
        raise RuntimeError(f"cannot identify prediction column: {path}")
    output = frame[["id", prediction_column]].rename(
        columns={prediction_column: "tvt"}
    )
    output["id"] = output["id"].astype(str)
    output["tvt"] = pd.to_numeric(output["tvt"], errors="coerce")
    if output["id"].duplicated().any() or not np.isfinite(output["tvt"]).all():
        raise RuntimeError(f"invalid submission: {path}")
    return output


before = _normalise(BEFORE_PATH)
after = _normalise(AFTER_PATH)
if set(before["id"]) != set(after["id"]):
    raise RuntimeError("Neighbor before/after IDs differ")

after_lookup = after.set_index("id")["tvt"]
after_values = before["id"].map(after_lookup).to_numpy(np.float64)
before_values = before["tvt"].to_numpy(np.float64)
if not np.isfinite(after_values).all():
    raise RuntimeError("failed to align Neighbor after values")

# Prevent accidental use of the experimental geometry candidate.
if (WORK / "submission_neighbor_full_progress_geom.csv").exists():
    geometry = _normalise(WORK / "submission_neighbor_full_progress_geom.csv")
    geometry_values = before["id"].map(geometry.set_index("id")["tvt"]).to_numpy(float)
    if np.allclose(after_values, geometry_values, rtol=0.0, atol=1e-10):
        raise RuntimeError("AFTER_PATH is the geometry experiment, not Neighbor v0")

if ORIGINAL_AUDIT_PATH.exists():
    original_audit = json.loads(ORIGINAL_AUDIT_PATH.read_text(encoding="utf-8"))
    method = str(original_audit.get("config", {}).get("method", ""))
    if method not in {"structure_progress", "progress_v0"}:
        raise RuntimeError(f"unexpected Neighbor source method: {method!r}")
    if original_audit.get("guard_status") == "fallback_to_strong":
        raise RuntimeError("original Neighbor guard fell back; no valid increment")
else:
    original_audit = None

original_move = after_values - before_values
original_abs = np.abs(original_move)
if float(np.max(original_abs)) > 6.000001:
    raise RuntimeError("original Neighbor move exceeds the frozen 6 ft cap")
if float(np.mean(original_abs)) <= 1e-8:
    raise RuntimeError("original Neighbor increment is empty")

probe_move = LAMBDA_NEIGHBOR * original_move
probe_values = before_values + probe_move
if not np.isfinite(probe_values).all():
    raise RuntimeError("non-finite lambda probe")
if float(np.max(np.abs(probe_move))) > 3.000001:
    raise RuntimeError("lambda=0.5 move exceeds the expected 3 ft cap")

output = before[["id"]].copy()
output["tvt"] = probe_values
# Preserve the exact lambda=1 vector inside the running kernel before the
# final submission.csv is replaced by the half-dose probe.
after.to_csv(LAMBDA_ONE_SNAPSHOT, index=False)
output.to_csv(OUTPUT_PATH, index=False)
output.to_csv(FINAL_PATH, index=False)

report = {
    "experiment": "neighbor_frozen_lambda_half_probe",
    "lambda_neighbor": LAMBDA_NEIGHBOR,
    "before": str(BEFORE_PATH),
    "after": str(AFTER_PATH),
    "lambda_one_snapshot": str(LAMBDA_ONE_SNAPSHOT),
    "rows": int(len(output)),
    "wells": int(output["id"].str.rsplit("_", n=1).str[0].nunique()),
    "original_move": {
        "mean_abs": float(np.mean(original_abs)),
        "rms": float(np.sqrt(np.mean(original_move**2))),
        "p95_abs": float(np.quantile(original_abs, 0.95)),
        "max_abs": float(np.max(original_abs)),
    },
    "probe_move": {
        "mean_abs": float(np.mean(np.abs(probe_move))),
        "rms": float(np.sqrt(np.mean(probe_move**2))),
        "p95_abs": float(np.quantile(np.abs(probe_move), 0.95)),
        "max_abs": float(np.max(np.abs(probe_move))),
    },
    "output": str(OUTPUT_PATH),
    "final": str(FINAL_PATH),
}
(WORK / "neighbor_lambda_half_probe_audit.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
)

print("NEIGHBOR LAMBDA=0.5 PROBE COMPLETE")
print(json.dumps(report, indent=2))
print("final Kaggle file:", FINAL_PATH)
