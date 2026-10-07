"""Final Kaggle audit for the clean 6.39 + CP8-noCP12 submission.

Run after the original notebook's guarded same-well, balanced visible-prefix,
model-package, and midpoint cells. Do not run the old Q0522 cell first.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


WORK = Path("/kaggle/working")
DATA = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
if "CFG" in globals():
    DATA = Path(getattr(CFG, "DATA", getattr(CFG, "dataset_path", DATA)))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned(path, value_name, sample):
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "tvt"]:
        raise RuntimeError(f"{path}: expected exactly id,tvt")
    frame["id"] = frame["id"].astype(str)
    if frame["id"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate ids")
    merged = sample[["id"]].merge(
        frame.rename(columns={"tvt": value_name}),
        on="id",
        how="left",
        validate="one_to_one",
    )
    values = pd.to_numeric(merged[value_name], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path}: missing or non-finite predictions")
    return values


sample_path = DATA / "sample_submission.csv"
final_path = WORK / "submission.csv"
parent_path = WORK / "submission_sp45_learned_w0.60.csv"
cp_path = WORK / "submission_frontier_6p39_cp8_nocp12_before_overlays.csv"
deployment_path = WORK / "frontier_6p39_cp8_nocp12_deployment.json"
q0522_path = WORK / "q0522_cp8_rebased_summary.json"

for required in (sample_path, final_path, parent_path, cp_path, deployment_path):
    if not required.exists():
        raise FileNotFoundError(required)

sample = pd.read_csv(sample_path)
if "id" not in sample:
    raise RuntimeError("sample_submission.csv has no id column")
sample = sample[["id"]].copy()
sample["id"] = sample["id"].astype(str)
if sample["id"].duplicated().any():
    raise RuntimeError("sample_submission.csv contains duplicate ids")

parent = aligned(parent_path, "parent", sample)
cp = aligned(cp_path, "cp", sample)
final = aligned(final_path, "final", sample)

cp_move = cp - parent
overlay_move = final - cp
if np.max(np.abs(cp_move)) > 12.0001:
    raise RuntimeError(f"CP cap violated: {np.max(np.abs(cp_move))}")
if np.allclose(cp_move, 0.0):
    raise RuntimeError("CP8 made no change")

deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
if deployment.get("cp_before_overlays_sha256") != sha256(cp_path):
    raise RuntimeError("CP checkpoint hash no longer matches deployment metadata")

report = {
    "experiment": "frontier_6p39_cp8_nocp12_clean_lb_v10",
    "rows": int(len(sample)),
    "wells": int(sample["id"].str.rsplit("_", n=1).str[0].nunique()),
    "parent_sha256": sha256(parent_path),
    "cp_before_overlays_sha256": sha256(cp_path),
    "final_sha256": sha256(final_path),
    "cp_move": {
        "mean_abs": float(np.mean(np.abs(cp_move))),
        "p95_abs": float(np.quantile(np.abs(cp_move), 0.95)),
        "max_abs": float(np.max(np.abs(cp_move))),
    },
    "post_cp_overlay_move": {
        "changed_rows": int(np.count_nonzero(np.abs(overlay_move) > 1e-9)),
        "mean_abs": float(np.mean(np.abs(overlay_move))),
        "p95_abs": float(np.quantile(np.abs(overlay_move), 0.95)),
        "max_abs": float(np.max(np.abs(overlay_move))),
    },
    "q0522_rebased": (
        json.loads(q0522_path.read_text(encoding="utf-8"))
        if q0522_path.exists()
        else None
    ),
    "excluded": [
        "historical CP12 parent/sidecurve",
        "Neighbor",
        "old final audit bound to the pre-CP hash",
    ],
    "submission": str(final_path),
    "ready": True,
}
(WORK / "frontier_6p39_cp8_nocp12_final_audit_v10.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print("CP8 CLEAN FINAL AUDIT", json.dumps(report, indent=2), flush=True)
print("SUBMIT:", final_path, flush=True)
