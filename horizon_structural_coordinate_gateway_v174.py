"""V174: fivefold horizon-relative structural-coordinate gateway.

The six mapped formation surfaces in each horizontal-well file are available
throughout the prediction interval. This experiment tests whether their spatial
movement explains the target geological surface S = TVT + Z. It uses no future
GR or target truth at inference time.

The model is deliberately closed form:

* one equal-well sampled ridge fit per held-out fold;
* one analytic blend dose fitted on the other four folds;
* a within-well rolled-surface control;
* a per-well truth oracle used only as a representation ceiling.

Run as a script or one notebook cell. In Jupyter, use `%run path/to/file.py`.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd


EXPERIMENT = "horizon_structural_coordinate_gateway_v174"
CWD = Path.cwd()
ROOT = (
    CWD
    if (CWD / "train").is_dir()
    else Path("/root/autodl-tmp")
    if Path("/root/autodl-tmp").is_dir()
    else CWD
)
DATA_ROOT = next(
    (path for path in (ROOT / "rogii-wellbore-geology-prediction", ROOT) if (path / "train").is_dir()),
    ROOT,
)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").is_dir() else ROOT / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
SAMPLE_NODES = 192
RIDGE = 0.05
ROLL_FRACTION = 0.20


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


def _read_base() -> tuple[pd.DataFrame, str, str]:
    candidates = (
        (
            ROOT / "oof_exports" / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet",
            "nested_sequence_coefficient",
        ),
        (
            ROOT / "oof_exports" / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
            "incumbent_6p377_replay",
        ),
        (
            DATA_ROOT / "output" / "tail_only_reanchored_neighbor_gateway_v0" / "tail_only_oof.csv.gz",
            "base",
        ),
    )
    for path, prediction in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(
                path,
                usecols=["well", "fold", "row_idx", "truth", "distance", prediction],
                dtype={"well": str, "fold": np.int8, "row_idx": np.int32},
            )
        truth = next((name for name in ("tvt_true", "truth") if name in frame), None)
        distance = next((name for name in ("md_since", "distance") if name in frame), None)
        if truth is None or distance is None or prediction not in frame:
            continue
        output = frame[["well", "fold", "row_idx", truth, distance, prediction]].copy()
        output.columns = ["well", "fold", "row_idx", "truth", "distance", "base"]
        output["well"] = output["well"].astype(str)
        output["fold"] = output["fold"].astype(np.int8)
        output["row_idx"] = output["row_idx"].astype(np.int32)
        return output.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True), str(path), prediction
    raise FileNotFoundError("No supported complete fivefold OOF endpoint found")


def _feature_matrix(horizontal: pd.DataFrame, anchor: int, rows: np.ndarray) -> np.ndarray:
    horizon = horizontal[list(HORIZONS)].to_numpy(np.float64)
    md = horizontal["MD"].to_numpy(np.float64)
    for column in range(horizon.shape[1]):
        valid = np.isfinite(md) & np.isfinite(horizon[:, column])
        if int(valid.sum()) >= 2:
            horizon[:, column] = np.interp(
                md, md[valid], horizon[valid, column]
            )
        elif int(valid.sum()) == 1:
            horizon[:, column] = horizon[valid, column][0]
        else:
            horizon[:, column] = 0.0
    delta = horizon[rows] - horizon[anchor][None, :]
    xy = horizontal[["X", "Y"]].to_numpy(np.float64)
    distance = md[rows] - md[anchor]
    displacement = xy[rows] - xy[anchor][None, :]
    # Horizon deltas are the primary physical coordinates. MD and XY terms let
    # the ridge represent a shared residual regional trend without an intercept,
    # preserving an exact zero correction at prediction start.
    output = np.column_stack(
        (
            delta,
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
            distance / 1000.0,
            displacement[:, 0] / 1000.0,
            displacement[:, 1] / 1000.0,
        )
    )
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite structural feature matrix")
    return output


def _sample_indices(length: int) -> np.ndarray:
    if length <= SAMPLE_NODES:
        return np.arange(length, dtype=np.int32)
    return np.unique(np.round(np.linspace(0, length - 1, SAMPLE_NODES)).astype(np.int32))


def _fit_ridge(x: np.ndarray, y: np.ndarray, well: np.ndarray) -> np.ndarray:
    counts = pd.Series(well).value_counts().to_dict()
    weight = np.asarray([1.0 / counts[item] for item in well], np.float64)
    weight *= len(weight) / np.sum(weight)
    scale = np.sqrt(np.average(x * x, axis=0, weights=weight))
    scale = np.where(scale > 1e-6, scale, 1.0)
    xs = x / scale
    lhs = xs.T @ (weight[:, None] * xs) + RIDGE * np.eye(xs.shape[1])
    rhs = xs.T @ (weight * y)
    return np.linalg.solve(lhs, rhs) / scale


def _metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame["truth"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    if not np.isfinite(error).all():
        raise RuntimeError(f"{endpoint}: non-finite predictions")
    by_well = (
        pd.DataFrame({"well": frame["well"], "e2": error * error})
        .groupby("well", sort=False)["e2"]
        .mean()
    )
    far = frame["distance"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(by_well))),
        "p95_well_rmse": float(np.quantile(np.sqrt(by_well), 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


started = time.time()
base_frame, base_source, base_column = _read_base()
test_horizontal = sorted((DATA_ROOT / "test").glob("*__horizontal_well.csv"))
test_schema_has_horizons = bool(test_horizontal) and all(
    set(HORIZONS).issubset(pd.read_csv(path, nrows=0).columns)
    for path in test_horizontal
)
if len(base_frame) != EXPECTED_ROWS or base_frame["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS}/{EXPECTED_WELLS} rows/wells, found "
        f"{len(base_frame)}/{base_frame['well'].nunique()}"
    )

records = {}
sample_x = []
sample_y = []
sample_well = []
sample_fold = []
for number, (well, local) in enumerate(base_frame.groupby("well", sort=True), 1):
    path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    horizontal = pd.read_csv(
        path,
        usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input", *HORIZONS],
    )
    known = np.flatnonzero(np.isfinite(horizontal["TVT_input"].to_numpy(np.float64)))
    rows = local["row_idx"].to_numpy(np.int64)
    official = np.flatnonzero(
        ~np.isfinite(horizontal["TVT_input"].to_numpy(np.float64))
        & np.isfinite(horizontal["TVT"].to_numpy(np.float64))
    )
    if not len(known) or not np.array_equal(rows, official):
        raise RuntimeError(f"{well}: OOF/native evaluation-mask mismatch")
    anchor = int(known[-1])
    x = _feature_matrix(horizontal, anchor, rows)
    truth = horizontal["TVT"].to_numpy(np.float64)
    z = horizontal["Z"].to_numpy(np.float64)
    target_surface_delta = (truth[rows] + z[rows]) - (truth[anchor] + z[anchor])
    anchor_projection = truth[anchor] - (z[rows] - z[anchor])
    if np.max(np.abs(truth[rows] - local["truth"].to_numpy(np.float64))) > 0.01:
        raise RuntimeError(f"{well}: truth parity failed")
    take = _sample_indices(len(rows))
    sample_x.append(x[take])
    sample_y.append(target_surface_delta[take])
    sample_well.extend([well] * len(take))
    sample_fold.extend([int(local["fold"].iloc[0])] * len(take))
    records[well] = {
        "positions": local.index.to_numpy(np.int64),
        "x": x,
        "surface_target": target_surface_delta,
        "anchor_projection": anchor_projection,
    }
    if number % 100 == 0 or number == EXPECTED_WELLS:
        print(f"V174 records {number}/{EXPECTED_WELLS}", flush=True)

sample_x = np.concatenate(sample_x)
sample_y = np.concatenate(sample_y)
sample_well = np.asarray(sample_well, str)
sample_fold = np.asarray(sample_fold, np.int8)

structural = np.full(len(base_frame), np.nan, np.float64)
rolled = np.full(len(base_frame), np.nan, np.float64)
perwell_oracle = np.full(len(base_frame), np.nan, np.float64)
model_rows = []
fold_array = base_frame["fold"].to_numpy(np.int8)
for held_fold in range(1, 6):
    train = sample_fold != held_fold
    coefficient = _fit_ridge(sample_x[train], sample_y[train], sample_well[train])
    for well, record in records.items():
        positions = record["positions"]
        if int(fold_array[positions[0]]) != held_fold:
            continue
        x = record["x"]
        structural[positions] = record["anchor_projection"] + x @ coefficient
        shift = max(1, int(round(len(x) * ROLL_FRACTION)))
        rolled[positions] = record["anchor_projection"] + np.roll(x, shift, axis=0) @ coefficient
        oracle_coefficient = np.linalg.lstsq(x, record["surface_target"], rcond=1e-6)[0]
        perwell_oracle[positions] = record["anchor_projection"] + x @ oracle_coefficient
    model_rows.append(
        {
            "heldout_fold": held_fold,
            **{f"coef_{name}": float(value) for name, value in zip(
                (*HORIZONS, "horizon_mean", "horizon_std", "md", "x", "y"), coefficient
            )},
        }
    )

base = base_frame["base"].to_numpy(np.float64)
truth = base_frame["truth"].to_numpy(np.float64)
nested = np.full(len(base_frame), np.nan, np.float64)
blend_rows = []
for held_fold in range(1, 6):
    train = fold_array != held_fold
    valid = ~train
    move = structural[train] - base[train]
    alpha = float(np.dot(truth[train] - base[train], move) / max(np.dot(move, move), 1e-12))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    nested[valid] = base[valid] + alpha * (structural[valid] - base[valid])
    blend_rows.append({"heldout_fold": held_fold, "selected_alpha": alpha})

summary = pd.DataFrame(
    [
        _metric(base_frame, base, "base_current"),
        _metric(base_frame, structural, "horizon_ridge_crossfit"),
        _metric(base_frame, nested, "horizon_nested_blend"),
        _metric(base_frame, rolled, "rolled_horizon_control"),
        _metric(base_frame, perwell_oracle, "perwell_horizon_oracle"),
    ]
)
fold_summary = []
for fold in range(1, 6):
    mask = fold_array == fold
    local = base_frame.loc[mask].reset_index(drop=True)
    for endpoint, prediction in (
        ("base_current", base[mask]),
        ("horizon_ridge_crossfit", structural[mask]),
        ("horizon_nested_blend", nested[mask]),
    ):
        fold_summary.append({"fold": fold, **_metric(local, prediction, endpoint)})
fold_summary = pd.DataFrame(fold_summary)

summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
fold_summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
pd.DataFrame(model_rows).to_csv(OUT_DIR / "coefficients.csv", index=False)
pd.DataFrame(blend_rows).to_csv(OUT_DIR / "blend_weights.csv", index=False)

output = base_frame[["well", "fold", "row_idx", "truth", "distance"]].copy()
output["base_current"] = base.astype(np.float32)
output["horizon_ridge_crossfit"] = structural.astype(np.float32)
output["horizon_nested_blend"] = nested.astype(np.float32)
output.to_csv(OUT_DIR / "oof.csv.gz", index=False, compression="gzip")

lookup = summary.set_index("endpoint")
base_rmse = float(lookup.loc["base_current", "rmse"])
structural_rmse = float(lookup.loc["horizon_ridge_crossfit", "rmse"])
blend_rmse = float(lookup.loc["horizon_nested_blend", "rmse"])
rolled_rmse = float(lookup.loc["rolled_horizon_control", "rmse"])
oracle_rmse = float(lookup.loc["perwell_horizon_oracle", "rmse"])
fold_pivot = fold_summary.pivot(index="fold", columns="endpoint", values="rmse")
fold_gains = (
    fold_pivot["base_current"] - fold_pivot["horizon_nested_blend"]
).to_dict()
verdict = {
    "experiment": EXPERIMENT,
    "stage": "complete_fivefold_horizon_relative_structural_coordinate_gateway",
    "base_source": base_source,
    "base_column": base_column,
    "features": [*HORIZONS, "horizon_mean", "horizon_std", "md", "x", "y"],
    "coverage": {"rows": len(base_frame), "wells": base_frame["well"].nunique()},
    "test_schema_has_horizons": test_schema_has_horizons,
    "values": {
        "base_rmse": base_rmse,
        "structural_rmse": structural_rmse,
        "nested_blend_rmse": blend_rmse,
        "nested_gain": base_rmse - blend_rmse,
        "rolled_increment": rolled_rmse - structural_rmse,
        "perwell_oracle_rmse": oracle_rmse,
        "fold_gains": fold_gains,
    },
    "checks": {
        "features_available_in_test": test_schema_has_horizons,
        "standalone_rmse_le_8": structural_rmse <= 8.0,
        "nested_gain_ge_0p10": base_rmse - blend_rmse >= 0.10,
        "positive_folds_ge_4": sum(value > 0.0 for value in fold_gains.values()) >= 4,
        "aligned_beats_rolled_ge_0p20": rolled_rmse - structural_rmse >= 0.20,
        "perwell_oracle_rmse_le_4": oracle_rmse <= 4.0,
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["checks"].values()))
verdict["next"] = (
    "PASS -> direct horizon-coordinate inference is legal"
    if verdict["strict_pass"]
    else "FAIL DIRECT USE -> distill train-only horizon labels from legal X/Y inputs"
)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, ensure_ascii=True), encoding="utf-8"
)

print("\nV174 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV174 FOLD SUMMARY")
print(fold_summary.to_string(index=False), flush=True)
print("\nV174 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
