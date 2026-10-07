"""Nested Track8 + STRIDE fusion against fold-clean strong endpoints.

This is a zero-training deployment-value gate.  It does not pretend that a
candidate-pool endpoint is the exact public-LB 6.781 submission.  Instead it
answers the narrower question that must be settled first: after a strong,
fold-clean endpoint is present, do Track8 and STRIDE retain useful independent
error, and does a three-way non-negative blend improve every held-out fold?

The three-way optimizer is exact for the simplex

    prediction = w_base * base + w_track8 * track8 + w_stride * stride
    w >= 0, sum(w) = 1.

Weights are fitted on four folds and applied unchanged to the held-out fold.
Truth is never used to choose a row- or well-specific deployment action.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "track8_stride_strong_nested_fusion_v4"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRACK8_PATH = OOF_ROOT / "track8_sp45_foldclean_fivefold_v2" / "track8_oof.parquet"
STRIDE_DIR = OOF_ROOT / "stride_sp45_fivefold_oof_v9" / "folds"
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
NEIGHBOR_V0_DIR = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
NEIGHBOR_V1_DIR = OOF_ROOT / "neighbor_fullcurve_projection_direction_v1"
NEIGHBOR_V1_CANDIDATES = (
    ROOT / "primary_candidate_oof.npz",
    NEIGHBOR_V1_DIR / "primary_candidate_oof.npz",
)

BASE_PREFERENCE = ("strong_bias", "cp12", "strong", "fleongg")
NEIGHBOR_WEIGHT = 0.25
NEIGHBOR_RAW_CAP_FT = 24.0
MIN_ABS_DIRECTION_DOT = 0.70


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    return float(np.sqrt(np.mean(error * error)))


def _metrics(frame: pd.DataFrame, column: str) -> dict:
    return _metrics_values(frame, frame[column].to_numpy(np.float64))


def _metrics_values(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    truth = frame["truth"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    temp = pd.DataFrame({"well": frame["well"].to_numpy(), "error2": error * error})
    well_rmse = np.sqrt(temp.groupby("well", sort=False)["error2"].mean())
    output = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(well_rmse.mean()),
        "median_well_rmse": float(well_rmse.median()),
        "p75_well_rmse": float(well_rmse.quantile(0.75)),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    for name, mask in (
        ("near_0_500_rmse", distance <= 500.0),
        ("mid_500_1000_rmse", (distance > 500.0) & (distance <= 1000.0)),
        ("mid_1000_2000_rmse", (distance > 1000.0) & (distance <= 2000.0)),
        ("far_gt2000_rmse", distance > 2000.0),
    ):
        output[name] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else float("nan")
    return output


def _pair_weight(base: np.ndarray, candidate: np.ndarray, truth: np.ndarray) -> float:
    direction = np.asarray(candidate, np.float64) - np.asarray(base, np.float64)
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return 0.0
    return float(
        np.clip(
            np.dot(np.asarray(truth, np.float64) - np.asarray(base, np.float64), direction)
            / denominator,
            0.0,
            1.0,
        )
    )


def _simplex_three_weights(
    base: np.ndarray,
    track8: np.ndarray,
    stride: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """Exact constrained least-squares solution on the 3-weight simplex."""
    base = np.asarray(base, np.float64)
    track8 = np.asarray(track8, np.float64)
    stride = np.asarray(stride, np.float64)
    truth = np.asarray(truth, np.float64)
    d1 = track8 - base
    d2 = stride - base
    target = truth - base

    candidates: list[np.ndarray] = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    # Interior stationary point.
    gram = np.array(
        [[np.dot(d1, d1), np.dot(d1, d2)], [np.dot(d1, d2), np.dot(d2, d2)]],
        dtype=np.float64,
    )
    rhs = np.array([np.dot(d1, target), np.dot(d2, target)], dtype=np.float64)
    if np.linalg.cond(gram) < 1e12:
        inner = np.linalg.solve(gram, rhs)
        if inner[0] >= 0.0 and inner[1] >= 0.0 and inner.sum() <= 1.0:
            candidates.append(np.array([1.0 - inner.sum(), inner[0], inner[1]]))

    # Three edges of the triangle.
    w_track8 = _pair_weight(base, track8, truth)
    candidates.append(np.array([1.0 - w_track8, w_track8, 0.0]))
    w_stride = _pair_weight(base, stride, truth)
    candidates.append(np.array([1.0 - w_stride, 0.0, w_stride]))
    w_track8_on_edge = _pair_weight(stride, track8, truth)
    candidates.append(np.array([0.0, w_track8_on_edge, 1.0 - w_track8_on_edge]))

    best_weights = candidates[0]
    best_sse = float("inf")
    for weights in candidates:
        prediction = weights[0] * base + weights[1] * track8 + weights[2] * stride
        error = prediction - truth
        sse = float(np.dot(error, error))
        if sse < best_sse:
            best_sse = sse
            best_weights = weights
    return best_weights


def _nested_prediction(
    frame: pd.DataFrame,
    base_column: str,
    mode: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    output = np.full(len(frame), np.nan, dtype=np.float64)
    rows = []
    folds = frame["fold"].to_numpy(np.int8)
    for held_fold in range(1, 6):
        train = folds != held_fold
        valid = ~train
        truth_train = frame.loc[train, "truth"].to_numpy(np.float64)
        base_train = frame.loc[train, base_column].to_numpy(np.float64)
        track8_train = frame.loc[train, "track8"].to_numpy(np.float64)
        stride_train = frame.loc[train, "stride"].to_numpy(np.float64)

        if mode == "base_track8":
            wt = _pair_weight(base_train, track8_train, truth_train)
            weights = np.array([1.0 - wt, wt, 0.0])
        elif mode == "base_stride":
            ws = _pair_weight(base_train, stride_train, truth_train)
            weights = np.array([1.0 - ws, 0.0, ws])
        elif mode == "track8_stride":
            wt = _pair_weight(stride_train, track8_train, truth_train)
            weights = np.array([0.0, wt, 1.0 - wt])
        elif mode == "three_way":
            weights = _simplex_three_weights(
                base_train, track8_train, stride_train, truth_train
            )
        else:
            raise ValueError(mode)

        valid_base = frame.loc[valid, base_column].to_numpy(np.float64)
        valid_track8 = frame.loc[valid, "track8"].to_numpy(np.float64)
        valid_stride = frame.loc[valid, "stride"].to_numpy(np.float64)
        output[valid] = (
            weights[0] * valid_base
            + weights[1] * valid_track8
            + weights[2] * valid_stride
        )
        valid_truth = frame.loc[valid, "truth"].to_numpy(np.float64)
        base_rmse = _rmse(valid_truth, valid_base)
        blend_rmse = _rmse(valid_truth, output[valid])
        rows.append(
            {
                "base": base_column,
                "mode": mode,
                "fold": held_fold,
                "weight_base": float(weights[0]),
                "weight_track8": float(weights[1]),
                "weight_stride": float(weights[2]),
                "base_rmse": base_rmse,
                "blend_rmse": blend_rmse,
                "gain_vs_base": base_rmse - blend_rmse,
            }
        )
    if not np.isfinite(output).all():
        raise RuntimeError(f"Incomplete nested prediction: {base_column}/{mode}")
    return output, pd.DataFrame(rows)


def _per_well_oracle(frame: pd.DataFrame, base_column: str) -> tuple[dict, pd.DataFrame]:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    rows = []
    for well, indices in frame.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        part = frame.iloc[indices]
        weights = _simplex_three_weights(
            part[base_column].to_numpy(np.float64),
            part["track8"].to_numpy(np.float64),
            part["stride"].to_numpy(np.float64),
            part["truth"].to_numpy(np.float64),
        )
        prediction[indices] = (
            weights[0] * part[base_column].to_numpy(np.float64)
            + weights[1] * part["track8"].to_numpy(np.float64)
            + weights[2] * part["stride"].to_numpy(np.float64)
        )
        rows.append(
            {
                "well": well,
                "weight_base": float(weights[0]),
                "weight_track8": float(weights[1]),
                "weight_stride": float(weights[2]),
            }
        )
    return _metrics_values(frame, prediction), pd.DataFrame(rows)


def _load_stride() -> pd.DataFrame:
    parts = []
    for fold in range(1, 6):
        path = STRIDE_DIR / f"fold{fold}_stride_raw.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            wells = data["wells"].astype(str)
            well_code = data["well_code"].astype(np.int64)
            row_idx = data["row_idx"].astype(np.int64)
            prediction = data["stride_raw"].astype(np.float32)
        well = wells[well_code]
        parts.append(
            pd.DataFrame(
                {
                    "id": [f"{item}_{int(row)}" for item, row in zip(well, row_idx)],
                    "stride_fold": np.full(len(row_idx), fold, np.int8),
                    "stride": prediction,
                }
            )
        )
    output = pd.concat(parts, ignore_index=True)
    output["id"] = output["id"].astype(str).str.lower()
    if output["id"].duplicated().any():
        raise RuntimeError("Duplicate STRIDE ids")
    return output


for required in (TRACK8_PATH,):
    if not required.exists():
        raise FileNotFoundError(required)
pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError(f"No candidate pool found: {POOL_CANDIDATES}")

track8 = pd.read_parquet(
    TRACK8_PATH,
    columns=["id", "well", "row_idx", "fold", "truth", "md_since", "track8"],
)
track8["id"] = track8["id"].astype(str).str.lower()
track8["well"] = track8["well"].astype(str).str.lower()
track8["row_idx"] = track8["row_idx"].astype(np.int32)
if len(track8) < 3_760_000 or track8["well"].nunique() != 773 or track8["id"].duplicated().any():
    raise RuntimeError("Invalid Track8 OOF coverage")

stride = _load_stride()
data = track8.merge(stride, on="id", how="inner", validate="one_to_one")
if len(data) != len(track8):
    raise RuntimeError(f"Track8/STRIDE alignment lost {len(track8) - len(data)} rows")
if not np.array_equal(data["fold"].to_numpy(np.int8), data["stride_fold"].to_numpy(np.int8)):
    raise RuntimeError("Track8 and STRIDE folds disagree")
data = data.drop(columns="stride_fold")

pool_header = pd.read_csv(pool_path, nrows=0).columns.tolist()
base_columns = [column for column in BASE_PREFERENCE if column in pool_header]
if not base_columns:
    raise RuntimeError(f"No supported strong base in pool: {pool_header}")
pool_usecols = [
    column
    for column in ("id", "well", "row_idx", "tvt_true", *base_columns)
    if column in pool_header
]
pool = pd.read_csv(pool_path, usecols=pool_usecols, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str).str.lower()
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
pool["well"] = pool["well"].astype(str).str.lower()
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["row_idx"] = pool["row_idx"].astype(np.int32)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
if pool["id"].duplicated().any():
    raise RuntimeError("Candidate pool contains duplicate ids")

# Reconstruct frozen neighbor endpoints.  v1 is preferred when its exact raw
# candidate survived; otherwise v0 remains an explicitly named proxy.
neighbor_meta_path = NEIGHBOR_V0_DIR / "selected_neighbors.csv"
neighbor_sources: dict[str, np.ndarray] = {}
neighbor_source_paths: dict[str, str] = {}
if neighbor_meta_path.exists():
    v1_path = next((path for path in NEIGHBOR_V1_CANDIDATES if path.exists()), None)
    if v1_path is not None:
        with np.load(v1_path, allow_pickle=False) as payload:
            candidate = payload["candidate"].astype(np.float64)
        if len(candidate) == len(pool):
            neighbor_sources["neighbor_v1_full_progress_geom"] = candidate
            neighbor_source_paths["neighbor_v1_full_progress_geom"] = str(v1_path)
        else:
            raise RuntimeError(
                f"Uploaded primary_candidate_oof length {len(candidate)} != pool rows {len(pool)}"
            )
    v0_path = NEIGHBOR_V0_DIR / "neighbor_transfer_oof.npz"
    if v0_path.exists():
        with np.load(v0_path, allow_pickle=False) as payload:
            candidate = payload["structure_progress"].astype(np.float64)
        if len(candidate) == len(pool):
            neighbor_sources["neighbor_v0_progress"] = candidate
            neighbor_source_paths["neighbor_v0_progress"] = str(v0_path)

if neighbor_sources:
    meta = pd.read_csv(neighbor_meta_path)
    meta["well"] = meta["well"].astype(str).str.lower()
    meta = meta.drop_duplicates("well").set_index("well")
    median_distance = pool["well"].map(meta["median_distance"]).to_numpy(np.float64)
    p90_distance = pool["well"].map(meta["p90_distance"]).to_numpy(np.float64)
    anchor_distance = pool["well"].map(meta["anchor_distance_full"]).to_numpy(np.float64)
    direction = pool["well"].map(meta["abs_direction_dot"]).to_numpy(np.float64)
    neighbor_eligible = (
        (median_distance <= 600.0)
        & (p90_distance <= 1200.0)
        & (anchor_distance <= 1200.0)
        & (direction >= MIN_ABS_DIRECTION_DOT)
    )
    original_bases = list(base_columns)
    for source_name, candidate in neighbor_sources.items():
        for base_name in original_bases:
            base = pool[base_name].to_numpy(np.float64)
            valid = neighbor_eligible & np.isfinite(candidate) & np.isfinite(base)
            prediction = base.copy()
            prediction[valid] += NEIGHBOR_WEIGHT * np.clip(
                candidate[valid] - base[valid], -NEIGHBOR_RAW_CAP_FT, NEIGHBOR_RAW_CAP_FT
            )
            endpoint_name = f"{base_name}__{source_name}"
            pool[endpoint_name] = prediction.astype(np.float32)
            base_columns.append(endpoint_name)

pool_keep = pool[["id", "tvt_true", *base_columns]].copy()
data = data.merge(pool_keep, on="id", how="inner", validate="one_to_one")
if len(data) != len(track8):
    raise RuntimeError(f"Strong endpoint alignment lost {len(track8) - len(data)} rows")
truth_gap = np.abs(data["truth"].to_numpy(np.float64) - data["tvt_true"].to_numpy(np.float64))
if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 1e-3:
    raise RuntimeError(f"Track8/pool truth mismatch: max={float(np.nanmax(truth_gap))}")
data = data.drop(columns="tvt_true")
numeric = ["truth", "md_since", "track8", "stride", *base_columns]
if not np.isfinite(data[numeric].to_numpy(np.float64)).all():
    raise RuntimeError("Fusion inputs contain non-finite values")

print(
    "V4 FUSION INPUTS",
    {
        "rows": len(data),
        "wells": data["well"].nunique(),
        "pool": str(pool_path),
        "bases": base_columns,
        "neighbor_sources": neighbor_source_paths,
        "exact_6p781_oof_available": False,
    },
    flush=True,
)

summary_rows = []
fold_parts = []
base_reports = []
best_proxy_prediction = None
best_proxy_name = None
best_proxy_rmse = float("inf")

for base_name in base_columns:
    base_metrics = _metrics(data, base_name)
    mode_metrics = {}
    mode_folds = {}
    for mode in ("base_track8", "base_stride", "track8_stride", "three_way"):
        prediction, fold_report = _nested_prediction(data, base_name, mode)
        metrics = _metrics_values(data, prediction)
        mode_metrics[mode] = metrics
        mode_folds[mode] = fold_report
        fold_parts.append(fold_report)
        summary_rows.append(
            {
                "base": base_name,
                "mode": mode,
                "base_rmse": base_metrics["rmse"],
                "rmse": metrics["rmse"],
                "gain_vs_base": base_metrics["rmse"] - metrics["rmse"],
                "mean_well_rmse": metrics["mean_well_rmse"],
                "p95_well_rmse": metrics["p95_well_rmse"],
                "far_gt2000_rmse": metrics["far_gt2000_rmse"],
                "minimum_fold_gain": float(fold_report["gain_vs_base"].min()),
                "positive_folds": int((fold_report["gain_vs_base"] > 0.0).sum()),
                "median_weight_base": float(fold_report["weight_base"].median()),
                "median_weight_track8": float(fold_report["weight_track8"].median()),
                "median_weight_stride": float(fold_report["weight_stride"].median()),
            }
        )
        if mode == "three_way" and metrics["rmse"] < best_proxy_rmse:
            best_proxy_rmse = metrics["rmse"]
            best_proxy_name = base_name
            best_proxy_prediction = prediction.astype(np.float32)

    oracle_metrics, oracle_weights = _per_well_oracle(data, base_name)
    oracle_weights.to_csv(OUT_DIR / f"{base_name}_threeway_perwell_oracle.csv", index=False)
    best_pair_rmse = min(
        mode_metrics["base_track8"]["rmse"],
        mode_metrics["base_stride"]["rmse"],
        mode_metrics["track8_stride"]["rmse"],
    )
    three_folds = mode_folds["three_way"]
    full_weights = _simplex_three_weights(
        data[base_name].to_numpy(np.float64),
        data["track8"].to_numpy(np.float64),
        data["stride"].to_numpy(np.float64),
        data["truth"].to_numpy(np.float64),
    )
    base_reports.append(
        {
            "base": base_name,
            "is_exact_6p781_oof": False,
            "base_metrics": base_metrics,
            "nested_base_track8": mode_metrics["base_track8"],
            "nested_base_stride": mode_metrics["base_stride"],
            "nested_track8_stride": mode_metrics["track8_stride"],
            "nested_three_way": mode_metrics["three_way"],
            "three_way_gain_vs_base": base_metrics["rmse"] - mode_metrics["three_way"]["rmse"],
            "three_way_increment_vs_best_pair": best_pair_rmse - mode_metrics["three_way"]["rmse"],
            "three_way_minimum_fold_gain": float(three_folds["gain_vs_base"].min()),
            "three_way_positive_folds": int((three_folds["gain_vs_base"] > 0.0).sum()),
            "three_way_fold_weights": three_folds[
                ["fold", "weight_base", "weight_track8", "weight_stride", "gain_vs_base"]
            ].to_dict(orient="records"),
            "full_oof_candidate_weights_not_validation": {
                "weight_base": float(full_weights[0]),
                "weight_track8": float(full_weights[1]),
                "weight_stride": float(full_weights[2]),
            },
            "perwell_threeway_oracle": oracle_metrics,
            "mean_perwell_oracle_weights": {
                "base": float(oracle_weights["weight_base"].mean()),
                "track8": float(oracle_weights["weight_track8"].mean()),
                "stride": float(oracle_weights["weight_stride"].mean()),
            },
        }
    )

summary = pd.DataFrame(summary_rows).sort_values(["rmse", "base", "mode"], ignore_index=True)
fold_report = pd.concat(fold_parts, ignore_index=True)
best_report = min(base_reports, key=lambda item: item["nested_three_way"]["rmse"])
checks = {
    "best_threeway_gain_vs_base_ge_0p10": best_report["three_way_gain_vs_base"] >= 0.10,
    "best_threeway_increment_vs_best_pair_ge_0p03": best_report[
        "three_way_increment_vs_best_pair"
    ] >= 0.03,
    "best_threeway_positive_folds_ge_4": best_report["three_way_positive_folds"] >= 4,
    "best_threeway_minimum_fold_gain_ge_minus0p05": best_report[
        "three_way_minimum_fold_gain"
    ] >= -0.05,
}

summary.to_csv(OUT_DIR / "summary.csv", index=False)
fold_report.to_csv(OUT_DIR / "nested_fold_weights.csv", index=False)
if best_proxy_prediction is None or best_proxy_name is None:
    raise RuntimeError("No three-way proxy prediction was produced")
prediction_export = data[
    ["id", "well", "row_idx", "fold", "truth", "md_since", "track8", "stride", best_proxy_name]
].copy()
prediction_export = prediction_export.rename(columns={best_proxy_name: "strong_proxy"})
prediction_export["nested_three_way"] = best_proxy_prediction
prediction_export.to_parquet(
    OUT_DIR / "best_nested_predictions.parquet", index=False, compression="zstd"
)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_nested_track8_stride_strong_endpoint_fusion_gate",
    "sample_unit": "complete physical well; pooled row RMSE",
    "protocol": {
        "weights_fit_on": "four outer folds",
        "weights_applied_to": "untouched held outer fold",
        "constraint": "nonnegative simplex weights summing to one",
        "row_or_well_specific_truth_selection": False,
    },
    "inputs": {
        "track8": str(TRACK8_PATH),
        "stride": str(STRIDE_DIR),
        "candidate_pool": str(pool_path),
        "neighbor_sources": neighbor_source_paths,
    },
    "lineage_warning": (
        "These are fold-clean strong/neighbor proxies, not the exact OOF of the public-LB 6.781 "
        "submission. Promotion to Kaggle still requires regenerating deployment-equivalent mainline OOF."
    ),
    "summary": summary.to_dict(orient="records"),
    "base_reports": base_reports,
    "best_proxy": best_report,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "next": (
        "PASS -> regenerate exact deployment-equivalent mainline OOF, then repeat this frozen blend"
        if all(checks.values())
        else "FAIL -> Track8/STRIDE diversity is consumed by strong endpoints; do not package deployment"
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV4 NESTED FUSION SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV4 BEST PROXY")
print(json.dumps(best_report, indent=2, default=_json_default), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
