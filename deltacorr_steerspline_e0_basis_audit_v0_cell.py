"""E0 fold-clean functional-basis audit for DeltaCorr-SteerSpline.

Run in the live native-resolution kernel.  PCA and the fold-clean mean curve
are fitted using only the 618 outer-train wells.  Validation truth is used only
to measure each representation's oracle projection on the 155 untouched Fold-1
wells.  All scores are exact row-pooled RMSE on native evaluation rows.
"""

from pathlib import Path
import json
import shutil
import time

import numpy as np
import pandas as pd


REQUIRED_E0 = {
    "records",
    "train_wells",
    "valid_wells",
    "z_arrays_s",
    "_bspline_basis_s",
    "_rmse",
    "OOF_ROOT",
}
missing_e0 = sorted(REQUIRED_E0.difference(globals()))
if missing_e0:
    raise RuntimeError(
        "Run this after the native-resolution/S-head definition cells; missing "
        + repr(missing_e0)
    )


EXPERIMENT_E0 = "deltacorr_steerspline_e0_basis_audit_v0"
OUT_DIR_E0 = OOF_ROOT / EXPERIMENT_E0
OUT_DIR_E0.mkdir(parents=True, exist_ok=True)

GRID_POINTS_E0 = 257
GRID_E0 = np.linspace(0.0, 1.0, GRID_POINTS_E0, dtype=np.float64)
PCA_DIMS_E0 = (4, 8, 16, 32)
DCT_DIMS_E0 = (4, 8, 16, 32)
BSPLINE_CONTROLS_E0 = (8, 12, 16, 24, 32)
RIDGE_E0 = 1e-5
POOLED_GATE_E0 = 1.5
ANCHOR_TOLERANCE_E0 = 1e-7


def _json_e0(path, payload):
    def default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


def _native_curve_e0(well):
    record = records[well]
    eval_index = record.eval_index
    denominator = max(
        float(record.md[-1] - record.md[record.anchor_index]), 1.0
    )
    progress = (
        record.md[eval_index].astype(np.float64)
        - float(record.md[record.anchor_index])
    ) / denominator
    z = z_arrays_s[well]
    z_delta = (
        z[eval_index].astype(np.float64) - float(z[record.anchor_index])
    )
    delta_s = (
        record.truth[eval_index].astype(np.float64)
        - float(record.anchor_tvt)
        + z_delta
    )
    return record, np.clip(progress, 0.0, 1.0), delta_s


def _grid_curve_e0(well):
    _, progress, delta_s = _native_curve_e0(well)
    x = np.r_[0.0, progress]
    y = np.r_[0.0, delta_s]
    keep = np.r_[True, np.diff(x) > 1e-10]
    x, y = x[keep], y[keep]
    if x[-1] < 1.0:
        x = np.r_[x, 1.0]
        y = np.r_[y, y[-1]]
    result = np.interp(GRID_E0, x, y)
    result[0] = 0.0
    return result.astype(np.float64)


def _interpolate_grid_family_e0(values, progress):
    values = np.asarray(values, np.float64)
    if values.ndim == 1:
        return np.interp(progress, GRID_E0, values)
    return np.stack(
        [np.interp(progress, GRID_E0, row) for row in values], axis=1
    )


def _fit_projection_e0(target, offset, design):
    target = np.asarray(target, np.float64)
    offset = np.asarray(offset, np.float64)
    design = np.asarray(design, np.float64)
    residual = target - offset
    gram = design.T @ design
    rhs = design.T @ residual
    coefficients = np.linalg.solve(
        gram + RIDGE_E0 * np.eye(gram.shape[0]), rhs
    )
    return offset + design @ coefficients, coefficients


def _evaluate_family_e0(method, dimension, mean_grid=None, components=None):
    truths, predictions, bywell = [], [], []
    for well in valid_wells:
        _, progress, target = _native_curve_e0(well)
        if method == "pca":
            offset = _interpolate_grid_family_e0(mean_grid, progress)
            design = _interpolate_grid_family_e0(
                components[:dimension], progress
            )
        elif method == "dct":
            offset = _interpolate_grid_family_e0(mean_grid, progress)
            frequencies = np.arange(1, dimension + 1, dtype=np.float64)
            design = np.cos(np.pi * progress[:, None] * frequencies[None]) - 1.0
        elif method == "bspline":
            offset = np.zeros_like(target)
            design = _bspline_basis_s(progress, int(dimension)).astype(np.float64)
        else:
            raise ValueError(method)
        prediction, coefficients = _fit_projection_e0(target, offset, design)
        truths.append(target)
        predictions.append(prediction)
        bywell.append(
            {
                "method": method,
                "dimension": int(dimension),
                "well": well,
                "rows": len(target),
                "rmse": _rmse(target, prediction),
                "mae": float(np.mean(np.abs(prediction - target))),
                "coefficient_l2": float(np.linalg.norm(coefficients)),
            }
        )
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    bywell = pd.DataFrame(bywell)
    return {
        "method": method,
        "dimension": int(dimension),
        "free_dimensions": (
            int(dimension - 1) if method == "bspline" else int(dimension)
        ),
        "rows": len(truth),
        "wells": len(valid_wells),
        "pooled_rmse": _rmse(truth, prediction),
        "pooled_mae": float(np.mean(np.abs(prediction - truth))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
        "anchor_abs_max": 0.0,
    }, bywell


started_e0 = time.time()
print(
    "E0 CONFIG",
    {
        "train_wells": len(train_wells),
        "valid_wells": len(valid_wells),
        "grid_points": GRID_POINTS_E0,
        "basis_source": "outer-train truth only",
        "validation": "native Fold-1 evaluation rows",
    },
    flush=True,
)

train_grid_e0 = np.stack([_grid_curve_e0(well) for well in train_wells])
mean_grid_e0 = train_grid_e0.mean(axis=0)
mean_grid_e0[0] = 0.0
centered_train_e0 = train_grid_e0 - mean_grid_e0[None]

# SVD is deterministic and sufficient for the 618 x 257 fold-clean matrix.
_, singular_values_e0, components_e0 = np.linalg.svd(
    centered_train_e0, full_matrices=False
)
components_e0[:, 0] = 0.0

anchor_guard_e0 = {
    "train_grid_anchor_abs_max": float(np.max(np.abs(train_grid_e0[:, 0]))),
    "mean_anchor_abs": float(abs(mean_grid_e0[0])),
    "pca_component_anchor_abs_max": float(
        np.max(np.abs(components_e0[: max(PCA_DIMS_E0), 0]))
    ),
    "dct_anchor_abs_max": float(
        np.max(
            np.abs(
                np.cos(
                    np.pi
                    * GRID_E0[0]
                    * np.arange(1, max(DCT_DIMS_E0) + 1)
                )
                - 1.0
            )
        )
    ),
}
if max(anchor_guard_e0.values()) > ANCHOR_TOLERANCE_E0:
    raise RuntimeError(f"E0 hard-anchor guard failed: {anchor_guard_e0}")

summary_rows_e0, bywell_frames_e0 = [], []
for dimension in PCA_DIMS_E0:
    summary, bywell = _evaluate_family_e0(
        "pca", dimension, mean_grid=mean_grid_e0, components=components_e0
    )
    summary_rows_e0.append(summary)
    bywell_frames_e0.append(bywell)
for dimension in DCT_DIMS_E0:
    summary, bywell = _evaluate_family_e0(
        "dct", dimension, mean_grid=mean_grid_e0
    )
    summary_rows_e0.append(summary)
    bywell_frames_e0.append(bywell)
for controls in BSPLINE_CONTROLS_E0:
    summary, bywell = _evaluate_family_e0("bspline", controls)
    summary_rows_e0.append(summary)
    bywell_frames_e0.append(bywell)

summary_e0 = pd.DataFrame(summary_rows_e0).sort_values(
    ["pooled_rmse", "free_dimensions", "method"]
).reset_index(drop=True)
bywell_e0 = pd.concat(bywell_frames_e0, ignore_index=True)
summary_e0.to_csv(OUT_DIR_E0 / "basis_summary.csv", index=False)
bywell_e0.to_csv(OUT_DIR_E0 / "basis_bywell.csv", index=False)

passing_e0 = summary_e0[summary_e0["pooled_rmse"] <= POOLED_GATE_E0].copy()
if len(passing_e0):
    winner_e0 = passing_e0.sort_values(
        ["free_dimensions", "pooled_rmse", "p95_well_rmse"]
    ).iloc[0]
else:
    winner_e0 = summary_e0.iloc[0]

basis_artifact_e0 = OUT_DIR_E0 / "fold1_basis_bank.npz"
np.savez_compressed(
    basis_artifact_e0,
    progress_grid=GRID_E0.astype(np.float32),
    train_mean=mean_grid_e0.astype(np.float32),
    pca_components=components_e0[: max(PCA_DIMS_E0)].astype(np.float32),
    singular_values=singular_values_e0[: max(PCA_DIMS_E0)].astype(np.float32),
    train_wells=np.asarray(train_wells, dtype="U16"),
    valid_wells=np.asarray(valid_wells, dtype="U16"),
)

strict_pass_e0 = bool(len(passing_e0) > 0)
winner_payload_e0 = {
    key: (
        int(value)
        if key in {"dimension", "free_dimensions", "rows", "wells"}
        else float(value)
        if isinstance(value, (float, np.floating))
        else value
    )
    for key, value in winner_e0.to_dict().items()
}
verdict_e0 = {
    "experiment": EXPERIMENT_E0,
    "stage": "E0_foldclean_basis_oracle",
    "train_wells": len(train_wells),
    "valid_wells": len(valid_wells),
    "grid_points": GRID_POINTS_E0,
    "basis_fit_scope": "outer_train_only",
    "validation_truth_usage": "oracle_projection_only",
    "anchor_guard": anchor_guard_e0,
    "gate_pooled_rmse_le": POOLED_GATE_E0,
    "passing_configurations": int(len(passing_e0)),
    "winner": winner_payload_e0,
    "strict_pass": strict_pass_e0,
    "basis_artifact": str(basis_artifact_e0),
    "elapsed_minutes": (time.time() - started_e0) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> run 1/4/24-well E1 capacity with the frozen winning basis"
        if strict_pass_e0
        else "FAIL -> stop DeltaCorr-SteerSpline deterministic functional head"
    ),
}
_json_e0(OUT_DIR_E0 / "verdict.json", verdict_e0)

print("\nE0 BASIS SUMMARY")
print(summary_e0.to_string(index=False), flush=True)
print("\nE0 FINAL VERDICT")
print(json.dumps(verdict_e0, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_E0, flush=True)
