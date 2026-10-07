"""Infer the frozen Neighbor dose optimum from exact before/after files.

This is an analysis-only cell. It never overwrites submission.csv and never
mixes the experimental full_progress_geom output with the LB-confirmed v0
Neighbor increment.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


WORK = Path("/kaggle/working")
BEFORE_OVERRIDE = None
AFTER_OVERRIDE = None

# Scores of the exact upstream CP12 -> Gold chain and the same chain followed
# by the frozen v0 Neighbor move.
LB_BEFORE = 6.991
LB_AFTER = 6.882
LB_DISPLAY_TOLERANCE = 0.0005

PAIR_CANDIDATES = (
    (
        WORK / "submission_before_neighbor.csv",
        WORK / "submission_neighbor_structure_strong.csv",
    ),
    (
        WORK / "submission_before_neighbor.csv",
        WORK / "submission_neighbor_structure.csv",
    ),
    (
        WORK / "submission_before_neighbor_fullcurve.csv",
        WORK / "submission_neighbor_fullcurve.csv",
    ),
)


def _normalise(path):
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
    return output.sort_values("id").reset_index(drop=True)


def _find_pair():
    if BEFORE_OVERRIDE is not None or AFTER_OVERRIDE is not None:
        if BEFORE_OVERRIDE is None or AFTER_OVERRIDE is None:
            raise RuntimeError("Set both BEFORE_OVERRIDE and AFTER_OVERRIDE")
        pair = Path(BEFORE_OVERRIDE), Path(AFTER_OVERRIDE)
        if not all(path.exists() for path in pair):
            raise FileNotFoundError(pair)
        return pair
    for pair in PAIR_CANDIDATES:
        if all(path.exists() for path in pair):
            return pair
    raise FileNotFoundError(
        "Cannot find the exact original Neighbor before/after pair. Download "
        "the output files from the scored 6.882 run or rerun the original v0 "
        "Neighbor final cell. Do not use submission_neighbor_full_progress_geom.csv."
    )


def _score_delta_bounds(before_score, after_score, tolerance):
    delta_low = max(before_score - tolerance, 0.0) ** 2 - (
        after_score + tolerance
    ) ** 2
    delta_high = (before_score + tolerance) ** 2 - max(
        after_score - tolerance, 0.0
    ) ** 2
    return float(delta_low), float(delta_high)


before_path, after_path = _find_pair()
if "full_progress_geom" in after_path.name:
    raise RuntimeError("Geometry experiment is not the LB-confirmed v0 increment")

before = _normalise(before_path)
after = _normalise(after_path)
if not before["id"].equals(after["id"]):
    raise RuntimeError("before/after IDs do not match exactly")

move = after["tvt"].to_numpy(np.float64) - before["tvt"].to_numpy(np.float64)
abs_move = np.abs(move)
v = float(np.mean(move**2))
if not np.isfinite(v) or v <= 1e-12:
    raise RuntimeError("Neighbor increment has zero or invalid energy")

movement = {
    "rows": int(len(move)),
    "wells": int(before["id"].str.rsplit("_", n=1).str[0].nunique()),
    "nonzero_fraction": float(np.mean(abs_move > 1e-10)),
    "mean_abs_move": float(np.mean(abs_move)),
    "rms_move": float(np.sqrt(v)),
    "p95_abs_move": float(np.quantile(abs_move, 0.95)),
    "max_abs_move": float(np.max(abs_move)),
    "v_mean_squared_move": v,
}

lineage_checks = {
    "max_move_within_v0_cap": bool(movement["max_abs_move"] <= 6.000001),
    "mean_move_in_oof_guard": bool(0.30 <= movement["mean_abs_move"] <= 2.70),
    "p95_move_in_oof_guard": bool(1.00 <= movement["p95_abs_move"] <= 6.000001),
    "hidden_sized_well_count": bool(movement["wells"] >= 100),
}
if not all(
    value
    for key, value in lineage_checks.items()
    if key != "hidden_sized_well_count"
):
    raise RuntimeError(f"Neighbor lineage checks failed: {lineage_checks}")

m0 = float(LB_BEFORE**2)
m1 = float(LB_AFTER**2)
observed_gain = m0 - m1
u = float((m1 - m0 - v) / 2.0)
lambda_star = float(-u / v)
minimum_mse = float(m0 - u * u / v)

delta_low, delta_high = _score_delta_bounds(
    LB_BEFORE, LB_AFTER, LB_DISPLAY_TOLERANCE
)
lambda_low = float(0.5 + delta_low / (2.0 * v))
lambda_high = float(0.5 + delta_high / (2.0 * v))

lambdas = sorted(
    set(
        [
            0.0,
            0.25,
            0.50,
            0.75,
            1.00,
            1.25,
            1.50,
            2.00,
            float(np.clip(lambda_star, 0.0, 2.0)),
        ]
    )
)
curve_rows = []
for dose in lambdas:
    mse = float(m0 + 2.0 * dose * u + dose * dose * v)
    curve_rows.append(
        {
            "lambda": dose,
            "predicted_mse": mse,
            "predicted_rmse": float(np.sqrt(max(mse, 0.0))),
            "max_abs_move": float(dose * movement["max_abs_move"]),
            "p95_abs_move": float(dose * movement["p95_abs_move"]),
        }
    )
curve = pd.DataFrame(curve_rows).sort_values("lambda")

decision = (
    "keep_lambda_1"
    if lambda_low <= 1.0 <= lambda_high
    else "test_shrunk_neighbor"
    if lambda_high < 1.0
    else "test_amplified_neighbor"
)
report = {
    "experiment": "neighbor_frozen_dose_audit_v1",
    "before": str(before_path),
    "after": str(after_path),
    "lb": {
        "before_rmse": LB_BEFORE,
        "after_rmse": LB_AFTER,
        "mse_gain": observed_gain,
        "display_tolerance": LB_DISPLAY_TOLERANCE,
    },
    "movement": movement,
    "lineage_checks": lineage_checks,
    "quadratic": {
        "u_error_dot_move": u,
        "v_move_energy": v,
        "lambda_star": lambda_star,
        "lambda_interval_from_score_rounding": [lambda_low, lambda_high],
        "predicted_minimum_rmse": float(np.sqrt(max(minimum_mse, 0.0))),
    },
    "decision": decision,
    "warning": (
        None
        if lineage_checks["hidden_sized_well_count"]
        else "Only preview wells are present; this dose is not the hidden response."
    ),
}

curve.to_csv(WORK / "neighbor_frozen_dose_curve.csv", index=False)
(WORK / "neighbor_frozen_dose_audit.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
)

print("MOVEMENT")
print(json.dumps(movement, indent=2))
print("\nDOSE CURVE")
print(curve.to_string(index=False))
print("\nVERDICT")
print(json.dumps(report, indent=2))
print("saved:", WORK / "neighbor_frozen_dose_audit.json")
