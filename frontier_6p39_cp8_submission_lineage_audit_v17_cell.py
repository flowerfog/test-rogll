"""Read-only lineage audit for the Frontier 6.39 + CP8 Kaggle submission.

Run this as the final Kaggle cell. It never writes submission.csv. The audit
measures whether each later notebook stage retained or cancelled the CP8 move.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

STAGES = [
    ("parent_6p39", "submission_sp45_learned_w0.60.csv"),
    ("cp8_checkpoint", "submission_frontier_6p39_cp8_nocp12_before_overlays.csv"),
    # Cell 44 creates this from the current submission after guarded overlap.
    ("after_guard_before_visible_prefix", "submission_self_verified_anchor.csv"),
    # Cell 45 snapshots the selected visible-prefix output.
    ("after_visible_prefix", "submission_before_model_package.csv"),
    # Cell 46 snapshots the model-package output.
    ("after_model_package", "submission_before_branch_hedge.csv"),
    # The rebased Q0522 cell snapshots the midpoint output.
    ("after_midpoint_before_q0522", "submission_frontier_6p39_cp8_nocp12_no_q0522.csv"),
    ("after_q0522_named", "submission_frontier_6p39_cp8_nocp12_q0522.csv"),
    ("final_submission", "submission.csv"),
]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    frame = pd.read_csv(path, dtype={"id": "string"})
    if list(frame.columns) != ["id", "tvt"]:
        raise RuntimeError(f"{path}: expected exactly id,tvt")
    if frame["id"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate ids")
    values = pd.to_numeric(frame["tvt"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path}: non-finite predictions")
    return frame["id"].astype(str).to_numpy(), values


available = [(label, WORK / name) for label, name in STAGES if (WORK / name).exists()]
paths = dict(available)
for required in ("parent_6p39", "cp8_checkpoint", "final_submission"):
    if required not in paths:
        raise FileNotFoundError(f"Missing required lineage artifact: {required}")

base_ids, parent = _read(paths["parent_6p39"])
cp_ids, cp8 = _read(paths["cp8_checkpoint"])
if not np.array_equal(base_ids, cp_ids):
    raise RuntimeError("Parent and CP8 ids/order differ")

cp_move = cp8 - parent
cp_energy = float(np.dot(cp_move, cp_move))
if cp_energy <= 1e-12:
    raise RuntimeError("CP8 checkpoint is identical to the 6.39 parent")

active = np.abs(cp_move) >= 0.05
rows = []
previous = None
first_cancellation = None

for label, path in available:
    ids, values = _read(path)
    if not np.array_equal(base_ids, ids):
        raise RuntimeError(f"{path}: ids/order differ from parent")

    stage_move = values - parent
    beta = float(np.dot(stage_move, cp_move) / cp_energy)
    residual = stage_move - beta * cp_move
    cp_distance = values - cp8
    cancelled = active & (np.abs(stage_move) <= 0.10 * np.abs(cp_move))
    changed_from_previous = (
        0 if previous is None else int(np.count_nonzero(np.abs(values - previous) > 1e-9))
    )
    row = {
        "stage": label,
        "file": path.name,
        "sha256": _sha256(path),
        "projection_on_cp8_move": beta,
        "rmse_from_cp8_checkpoint": float(np.sqrt(np.mean(cp_distance ** 2))),
        "orthogonal_overlay_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mean_abs_move_from_parent": float(np.mean(np.abs(stage_move))),
        "changed_rows_from_previous": changed_from_previous,
        "cancelled_active_cp8_rows": int(cancelled.sum()),
        "cancelled_active_cp8_fraction": float(cancelled.sum() / max(1, active.sum())),
        "exactly_cp8_checkpoint": bool(np.array_equal(values, cp8)),
        "exactly_parent": bool(np.array_equal(values, parent)),
    }
    rows.append(row)
    if (
        label not in {"parent_6p39", "cp8_checkpoint"}
        and first_cancellation is None
        and (beta < 0.80 or row["cancelled_active_cp8_fraction"] > 0.20)
    ):
        first_cancellation = label
    previous = values

report = pd.DataFrame(rows)
final = report.loc[report["stage"].eq("final_submission")].iloc[0]
verdict = {
    "experiment": "frontier_6p39_cp8_submission_lineage_audit_v17",
    "read_only": True,
    "cp8_move": {
        "rows": int(len(cp_move)),
        "active_rows_ge_0p05": int(active.sum()),
        "mean_abs_ft": float(np.mean(np.abs(cp_move))),
        "p95_abs_ft": float(np.quantile(np.abs(cp_move), 0.95)),
        "max_abs_ft": float(np.max(np.abs(cp_move))),
    },
    "final_projection_on_cp8_move": float(final["projection_on_cp8_move"]),
    "final_rmse_from_cp8_checkpoint": float(final["rmse_from_cp8_checkpoint"]),
    "final_cancelled_active_cp8_fraction": float(final["cancelled_active_cp8_fraction"]),
    "first_suspected_cancellation_stage": first_cancellation,
    "cp8_survived": bool(
        final["projection_on_cp8_move"] >= 0.80
        and final["cancelled_active_cp8_fraction"] <= 0.20
    ),
    "interpretation": (
        "CP8 is present in the final submission; the LB result is not explained by a global overwrite."
        if final["projection_on_cp8_move"] >= 0.80
        and final["cancelled_active_cp8_fraction"] <= 0.20
        else "CP8 was substantially cancelled; inspect the first cancellation stage above."
    ),
}

report.to_csv(WORK / "frontier_6p39_cp8_submission_lineage_v17.csv", index=False)
(WORK / "frontier_6p39_cp8_submission_lineage_v17.json").write_text(
    json.dumps(verdict, indent=2), encoding="utf-8"
)

print("CP8 SUBMISSION LINEAGE")
print(
    report[
        [
            "stage",
            "projection_on_cp8_move",
            "rmse_from_cp8_checkpoint",
            "orthogonal_overlay_rmse",
            "changed_rows_from_previous",
            "cancelled_active_cp8_fraction",
            "exactly_cp8_checkpoint",
            "exactly_parent",
        ]
    ].to_string(index=False)
)
print("FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
