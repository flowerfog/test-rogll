"""Candidate-conditioned TVT-bin projection consistency gateway.

This is an investment audit for a signal that earlier experiments did not
measure: project hidden horizontal-well GR onto each *candidate* TVT path,
aggregate repeated observations in TVT bins, and compare that denoised curve
with the legally prefix-calibrated typewell reference.

The experiment is whole-well five-fold OOF.  Truth is used only for candidate
MSE labels and final metrics.  Projection features use horizontal GR, candidate
paths, typewell GR, and visible-prefix TVT_input/GR.  A no-projection model and
a well-shuffled projection-feature model are matched controls.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.stats import binned_statistic, spearmanr
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


warnings.filterwarnings("ignore")

ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "candidate_tvtbin_projection_gateway_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V18_PATH = (
    OOF_ROOT / "frontier_6p39_full_oof_cp_increment_v18"
    / "full_oof_cp_endpoints.parquet"
)
DATA_ROOT_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    ROOT / "dataset",
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
)
DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if (p / "train").is_dir()), None)
if DATA_ROOT is None:
    raise FileNotFoundError(f"Cannot locate raw data under {DATA_ROOT_CANDIDATES}")
if not V18_PATH.exists():
    raise FileNotFoundError(V18_PATH)

PARENT_COLUMN = "full_parent_matched_vp_replay"
CP_COLUMN = "full_parent_plus_raw_cp_crossfit"
SOURCE_COLUMNS = (PARENT_COLUMN, CP_COLUMN)
SOURCE_NAMES = ("parent", "cp8")
OFFSETS_FT = np.asarray([-16, -12, -8, -4, 0, 4, 8, 12, 16], np.float64)
BIN_WIDTHS_FT = (0.5, 1.0, 2.0, 4.0)
OFFSET_RAMP_FT = 1000.0
MIN_BIN_ROWS = 3
N_JOBS = max(1, int(os.environ.get("ROGII_TVTBIN_JOBS", "6")))
MAX_WELLS = int(os.environ.get("ROGII_TVTBIN_MAX_WELLS", "0"))
SEED = 20260725
DISTANCE_EDGES = (-np.inf, 500.0, 1000.0, 2000.0, np.inf)
DISTANCE_NAMES = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")
started = time.time()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
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


def _corr(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4:
        return 0.0
    x = x[valid] - np.mean(x[valid])
    y = y[valid] - np.mean(y[valid])
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else 0.0


def _robust_affine(expected, observed):
    expected = np.asarray(expected, np.float64)
    observed = np.asarray(observed, np.float64)
    valid = np.isfinite(expected) & np.isfinite(observed)
    x = expected[valid]
    y = observed[valid]
    if len(x) < 32:
        return 1.0, 0.0, 12.0
    keep = np.ones(len(x), bool)
    gain, intercept = 1.0, float(np.median(y - x))
    for _ in range(4):
        design = np.column_stack([x[keep], np.ones(keep.sum())])
        gain, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        gain = float(np.clip(gain, 0.50, 1.50))
        residual = y - (gain * x + intercept)
        centre = float(np.median(residual))
        scale = float(1.4826 * np.median(np.abs(residual - centre)))
        scale = max(scale, 1.0)
        new_keep = np.abs(residual - centre) <= 3.5 * scale
        if new_keep.sum() < 24 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    residual = y - (gain * x + intercept)
    gamma = float(max(np.median(np.abs(residual - np.median(residual))), 2.0))
    return gain, float(intercept), gamma


def _prepare_typewell(frame):
    work = pd.DataFrame(
        {
            "tvt": pd.to_numeric(frame["TVT"], errors="coerce"),
            "gr": pd.to_numeric(frame["GR"], errors="coerce"),
        }
    ).dropna()
    work = work.groupby("tvt", as_index=False)["gr"].mean().sort_values("tvt")
    if len(work) < 32:
        raise RuntimeError("too few usable typewell rows")
    return work["tvt"].to_numpy(np.float64), work["gr"].to_numpy(np.float64)


def _projection_features(path, gr, tw_tvt, calibrated_tw_gr, gamma, width):
    valid = (
        np.isfinite(path)
        & np.isfinite(gr)
        & (path >= tw_tvt[0])
        & (path <= tw_tvt[-1])
    )
    output = {}
    output["row_coverage"] = float(valid.mean())
    if valid.sum() < 32:
        for name in (
            "occupied_bins", "occupied_span_fraction", "mean_bin_count",
            "p90_bin_count", "repeat_row_fraction", "within_bin_std",
            "mae", "rmse", "cauchy", "bias", "ncc", "d1_ncc",
        ):
            output[name] = 0.0
        return output

    values = path[valid]
    observed = gr[valid]
    lower = np.floor(values.min() / width) * width - 0.5 * width
    upper = np.ceil(values.max() / width) * width + 0.5 * width
    edges = np.arange(lower, upper + 0.51 * width, width, dtype=np.float64)
    if len(edges) < 3:
        edges = np.asarray([lower, lower + width, lower + 2 * width])
    median, _, bin_number = binned_statistic(
        values, observed, statistic="median", bins=edges
    )
    bin_count = len(edges) - 1
    count = np.bincount(bin_number, minlength=bin_count + 2)[1 : bin_count + 1].astype(np.float64)
    total = np.bincount(
        bin_number, weights=observed, minlength=bin_count + 2
    )[1 : bin_count + 1]
    total2 = np.bincount(
        bin_number, weights=observed * observed, minlength=bin_count + 2
    )[1 : bin_count + 1]
    mean = total / np.maximum(count, 1.0)
    mean2 = total2 / np.maximum(count, 1.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    usable = (count >= MIN_BIN_ROWS) & np.isfinite(median)
    if usable.sum() < 12:
        usable = (count >= 1) & np.isfinite(median)
    reference = np.interp(centres, tw_tvt, calibrated_tw_gr)
    usable &= np.isfinite(reference)
    if usable.sum() < 8:
        for name in (
            "occupied_bins", "occupied_span_fraction", "mean_bin_count",
            "p90_bin_count", "repeat_row_fraction", "within_bin_std",
            "mae", "rmse", "cauchy", "bias", "ncc", "d1_ncc",
        ):
            output[name] = 0.0
        return output

    obs = median[usable]
    ref = reference[usable]
    counts = count[usable]
    residual = obs - ref
    within_std = np.sqrt(np.maximum(mean2[usable] - mean[usable] ** 2, 0.0))
    occupied = int(usable.sum())
    span_bins = max(1, int(np.ceil((values.max() - values.min()) / width)))
    output.update(
        {
            "occupied_bins": float(occupied),
            "occupied_span_fraction": float(occupied / span_bins),
            "mean_bin_count": float(np.mean(counts)),
            "p90_bin_count": float(np.quantile(counts, 0.90)),
            "repeat_row_fraction": float(np.sum(counts[counts > 1]) / max(valid.sum(), 1)),
            "within_bin_std": float(np.mean(within_std)),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "cauchy": float(np.mean(np.log1p((residual / gamma) ** 2))),
            "bias": float(np.median(residual)),
            "ncc": _corr(obs, ref),
            "d1_ncc": _corr(np.diff(obs), np.diff(ref)),
        }
    )
    return output


header = pd.read_parquet(V18_PATH).columns.tolist()
required = {
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    PARENT_COLUMN, CP_COLUMN,
}
missing = sorted(required.difference(header))
if missing:
    raise RuntimeError(f"v18 OOF missing columns: {missing}")
oof = pd.read_parquet(V18_PATH, columns=sorted(required))
oof["well"] = oof["well"].astype(str)
oof = oof.sort_values(["well", "row_idx"]).reset_index(drop=True)
if oof["id"].duplicated().any():
    raise RuntimeError("duplicate v18 ids")

wells = list(oof["well"].drop_duplicates())
if MAX_WELLS > 0:
    wells = wells[:MAX_WELLS]
    oof = oof[oof["well"].isin(wells)].copy().reset_index(drop=True)
groups = {well: np.asarray(index, np.int64) for well, index in oof.groupby("well", sort=False).indices.items()}


def _build_well(well):
    indices = groups[well]
    frame = oof.iloc[indices]
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    try:
        hw = pd.read_csv(horizontal_path, usecols=["GR", "TVT_input"])
        tw = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        row_idx = frame["row_idx"].to_numpy(np.int64)
        if row_idx.min() < 0 or row_idx.max() >= len(hw):
            raise RuntimeError("row_idx outside horizontal file")
        eval_gr = pd.to_numeric(hw["GR"], errors="coerce").to_numpy(np.float64)[row_idx]
        tvt_input = pd.to_numeric(hw["TVT_input"], errors="coerce").to_numpy(np.float64)
        full_gr = pd.to_numeric(hw["GR"], errors="coerce").to_numpy(np.float64)
        tw_tvt, tw_gr = _prepare_typewell(tw)
        known = np.isfinite(tvt_input) & np.isfinite(full_gr)
        expected_known = np.interp(tvt_input[known], tw_tvt, tw_gr)
        gain, intercept, gamma = _robust_affine(expected_known, full_gr[known])
        calibrated_tw = gain * tw_gr + intercept

        truth = frame["tvt_true"].to_numpy(np.float64)
        distance = frame["md_since"].to_numpy(np.float64)
        ramp = np.clip(distance / OFFSET_RAMP_FT, 0.0, 1.0)
        records = []
        for source_name, source_column in zip(SOURCE_NAMES, SOURCE_COLUMNS):
            source = frame[source_column].to_numpy(np.float64)
            for offset in OFFSETS_FT:
                candidate = source + float(offset) * ramp
                error = candidate - truth
                record = {
                    "well": well,
                    "fold": int(frame["fold"].iloc[0]),
                    "source": source_name,
                    "source_code": float(source_name == "cp8"),
                    "offset_ft": float(offset),
                    "abs_offset_ft": float(abs(offset)),
                    "candidate_rows": int(len(frame)),
                    "candidate_sse": float(np.dot(error, error)),
                    "candidate_mse": float(np.mean(error ** 2)),
                    "candidate_mae": float(np.mean(np.abs(error))),
                    "path_end_minus_parent": float(candidate[-1] - frame[PARENT_COLUMN].iloc[-1]),
                    "path_mean_minus_parent": float(np.mean(candidate - frame[PARENT_COLUMN].to_numpy(np.float64))),
                    "path_tv": float(np.mean(np.abs(np.diff(candidate)))) if len(candidate) > 1 else 0.0,
                    "prefix_gain": gain,
                    "prefix_intercept": intercept,
                    "prefix_gamma": gamma,
                    "eval_gr_valid_fraction": float(np.isfinite(eval_gr).mean()),
                }
                for lower, upper, name in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:], DISTANCE_NAMES):
                    mask = (distance > lower) & (distance <= upper)
                    record[f"{name}_rows"] = int(mask.sum())
                    record[f"{name}_sse"] = float(np.dot(error[mask], error[mask])) if mask.any() else 0.0
                for width in BIN_WIDTHS_FT:
                    features = _projection_features(
                        candidate, eval_gr, tw_tvt, calibrated_tw, gamma, width
                    )
                    tag = str(width).replace(".", "p")
                    for name, value in features.items():
                        record[f"proj_b{tag}_{name}"] = float(value)
                records.append(record)
        return records, None
    except Exception as exc:
        return [], {"well": well, "reason": repr(exc)}


records = []
failures = []
with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
    futures = {executor.submit(_build_well, well): well for well in wells}
    for future in tqdm(as_completed(futures), total=len(futures), desc="TVT-bin projection wells"):
        rows, failure = future.result()
        records.extend(rows)
        if failure is not None:
            failures.append(failure)

pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)
if failures:
    raise RuntimeError(f"{len(failures)} well builds failed: {failures[:5]}")

candidates = pd.DataFrame(records).sort_values(["well", "source_code", "offset_ft"]).reset_index(drop=True)
candidates.to_parquet(OUT_DIR / "candidate_features.parquet", index=False, compression="zstd")
print(
    "TVTBIN MATRICES",
    {
        "candidate_rows": len(candidates),
        "wells": candidates["well"].nunique(),
        "features": int(sum(c.startswith("proj_") for c in candidates.columns)),
        "elapsed_minutes": (time.time() - started) / 60.0,
    },
    flush=True,
)

label_columns = {
    "well", "fold", "source", "candidate_rows", "candidate_sse", "candidate_mse", "candidate_mae",
    *[f"{name}_{suffix}" for name in DISTANCE_NAMES for suffix in ("rows", "sse")],
}
feature_columns = [
    column for column in candidates.columns
    if column not in label_columns and pd.api.types.is_numeric_dtype(candidates[column])
]
projection_columns = [column for column in feature_columns if column.startswith("proj_")]
control_columns = [column for column in feature_columns if column not in projection_columns]


def _shuffle_projection(frame, rng):
    output = frame.copy()
    well_order = output["well"].drop_duplicates().to_numpy()
    shuffled = rng.permutation(well_order)
    blocks = {well: output.index[output["well"].eq(well)].to_numpy() for well in well_order}
    source = output[projection_columns].copy()
    for destination_well, source_well in zip(well_order, shuffled):
        destination = blocks[destination_well]
        origin = blocks[source_well]
        if len(destination) != len(origin):
            raise RuntimeError("candidate block sizes differ")
        output.loc[destination, projection_columns] = source.loc[origin, projection_columns].to_numpy()
    return output


def _model_parameters(seed, n_estimators):
    return dict(
        objective="huber",
        alpha=0.90,
        n_estimators=int(n_estimators),
        learning_rate=0.025,
        num_leaves=23,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=8.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _matrix(frame, columns):
    return (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _fit_predict(train, apply, columns, seed, inner_fold):
    inner_train = train[train["fold"].ne(inner_fold)]
    inner_valid = train[train["fold"].eq(inner_fold)]
    if not len(inner_train) or not len(inner_valid):
        raise RuntimeError(f"empty inner split for Fold-{inner_fold}")
    x_inner_train = _matrix(inner_train, columns)
    x_inner_valid = _matrix(inner_valid, columns)
    y_inner_train = np.log1p(inner_train["candidate_mse"].to_numpy(np.float64))
    y_inner_valid = np.log1p(inner_valid["candidate_mse"].to_numpy(np.float64))
    inner_weights = inner_train["candidate_rows"].to_numpy(np.float64)
    inner_weights /= np.mean(inner_weights)
    selector = LGBMRegressor(**_model_parameters(seed, 1600))
    selector.fit(
        x_inner_train, y_inner_train, sample_weight=inner_weights,
        eval_set=[(x_inner_valid, y_inner_valid)], eval_metric="rmse",
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )
    iteration = max(25, int(selector.best_iteration_ or 400))

    x_train = _matrix(train, columns)
    x_apply = _matrix(apply, columns)
    y_train = np.log1p(train["candidate_mse"].to_numpy(np.float64))
    weights = train["candidate_rows"].to_numpy(np.float64)
    weights /= np.mean(weights)
    model = LGBMRegressor(**_model_parameters(seed + 5000, iteration))
    model.fit(x_train, y_train, sample_weight=weights)
    return model.predict(x_apply), iteration, model


score_columns = {
    "full": np.full(len(candidates), np.nan, np.float64),
    "no_projection": np.full(len(candidates), np.nan, np.float64),
    "shuffled_projection": np.full(len(candidates), np.nan, np.float64),
}
fold_reports = []
importance_rows = []
for fold in sorted(candidates["fold"].unique()):
    train = candidates[candidates["fold"].ne(fold)].copy()
    valid = candidates[candidates["fold"].eq(fold)].copy()
    valid_indices = valid.index.to_numpy()
    inner_folds = sorted(train["fold"].unique())
    inner_fold = int(inner_folds[(int(fold) - 1) % len(inner_folds)])

    full_score, iteration, model = _fit_predict(
        train, valid, feature_columns, SEED + int(fold), inner_fold
    )
    score_columns["full"][valid_indices] = full_score
    for name, importance in zip(feature_columns, model.feature_importances_):
        importance_rows.append({"fold": int(fold), "feature": name, "importance": int(importance)})

    control_score, control_iteration, _ = _fit_predict(
        train, valid, control_columns, SEED + 100 + int(fold), inner_fold
    )
    score_columns["no_projection"][valid_indices] = control_score

    rng = np.random.default_rng(SEED + 1000 + int(fold))
    shuffled_train = _shuffle_projection(train, rng)
    shuffled_valid = _shuffle_projection(valid, rng)
    shuffled_score, shuffled_iteration, _ = _fit_predict(
        shuffled_train, shuffled_valid, feature_columns,
        SEED + 200 + int(fold), inner_fold
    )
    score_columns["shuffled_projection"][valid_indices] = shuffled_score
    fold_reports.append(
        {
            "fold": int(fold),
            "train_wells": int(train["well"].nunique()),
            "valid_wells": int(valid["well"].nunique()),
            "inner_early_stopping_fold": inner_fold,
            "full_best_iteration": iteration,
            "control_best_iteration": control_iteration,
            "shuffled_best_iteration": shuffled_iteration,
        }
    )

for name, values in score_columns.items():
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite crossfit scores: {name}")
    candidates[f"score_{name}"] = values


def _chosen_by_score(column):
    return candidates.loc[candidates.groupby("well")[column].idxmin()].copy()


def _aggregate(chosen, variant):
    rows = int(chosen["candidate_rows"].sum())
    sse = float(chosen["candidate_sse"].sum())
    well_rmse = np.sqrt(chosen["candidate_mse"].to_numpy(np.float64))
    result = {
        "variant": variant,
        "rows": rows,
        "wells": int(chosen["well"].nunique()),
        "exact_sse": sse,
        "rmse": float(np.sqrt(sse / rows)),
        "mae": float(np.average(chosen["candidate_mae"], weights=chosen["candidate_rows"])),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }
    for name in DISTANCE_NAMES:
        count = int(chosen[f"{name}_rows"].sum())
        band_sse = float(chosen[f"{name}_sse"].sum())
        result[f"{name}_rmse"] = float(np.sqrt(band_sse / count)) if count else np.nan
    return result


zero = candidates[np.isclose(candidates["offset_ft"], 0.0)]
parent = zero[zero["source"].eq("parent")]
cp8 = zero[zero["source"].eq("cp8")]
oracle = candidates.loc[candidates.groupby("well")["candidate_mse"].idxmin()]

# A target-free heuristic: robust mismatch ranks at 1/2/4 ft plus derivative agreement.
heuristic_parts = []
for _, group in candidates.groupby("well", sort=False):
    score = np.zeros(len(group), np.float64)
    for width in (1.0, 2.0, 4.0):
        tag = str(width).replace(".", "p")
        for column, ascending in (
            (f"proj_b{tag}_cauchy", True),
            (f"proj_b{tag}_ncc", False),
            (f"proj_b{tag}_d1_ncc", False),
        ):
            score += group[column].rank(method="average", ascending=ascending).to_numpy(np.float64)
    heuristic_parts.append(group.iloc[[int(np.argmin(score))]])
heuristic = pd.concat(heuristic_parts, ignore_index=True)

chosen = {
    "parent": parent,
    "cp8": cp8,
    "bank_oracle": oracle,
    "projection_heuristic": heuristic,
    "no_projection_crossfit": _chosen_by_score("score_no_projection"),
    "shuffled_projection_crossfit": _chosen_by_score("score_shuffled_projection"),
    "projection_crossfit": _chosen_by_score("score_full"),
}
summary = pd.DataFrame([_aggregate(frame, name) for name, frame in chosen.items()])
summary.to_csv(OUT_DIR / "summary.csv", index=False)
pd.DataFrame(fold_reports).to_csv(OUT_DIR / "fold_reports.csv", index=False)
pd.DataFrame(importance_rows).groupby("feature", as_index=False)["importance"].sum().sort_values(
    "importance", ascending=False
).to_csv(OUT_DIR / "feature_importance.csv", index=False)

# Direct parent-vs-CP8 diagnostic, independent of the expanded offset bank.
pair = parent[["well", "candidate_mse"]].rename(columns={"candidate_mse": "parent_mse"}).merge(
    cp8[["well", "candidate_mse", *projection_columns]], on="well", suffixes=("", "_cp8")
)
parent_proj = parent[["well", *projection_columns]].copy()
parent_proj = parent_proj.rename(columns={c: f"{c}_parent" for c in projection_columns})
pair = pair.merge(parent_proj, on="well")
pair["cp8_benefit_mse"] = pair["parent_mse"] - pair["candidate_mse"]
pair["cp8_wins"] = pair["cp8_benefit_mse"] > 0
pair_rows = []
for width in BIN_WIDTHS_FT:
    tag = str(width).replace(".", "p")
    for metric, lower_better in (("cauchy", True), ("rmse", True), ("ncc", False), ("d1_ncc", False)):
        column = f"proj_b{tag}_{metric}"
        delta = pair[column] - pair[f"{column}_parent"]
        preference = -delta if lower_better else delta
        rho = float(spearmanr(preference, pair["cp8_benefit_mse"]).statistic)
        auc = float(roc_auc_score(pair["cp8_wins"], preference)) if pair["cp8_wins"].nunique() == 2 else np.nan
        pair_rows.append({"bin_width_ft": width, "metric": metric, "spearman_benefit": rho, "auc_cp8_wins": auc})
pair_report = pd.DataFrame(pair_rows).sort_values("spearman_benefit", ascending=False)
pair_report.to_csv(OUT_DIR / "parent_cp8_signal.csv", index=False)

metrics = {row["variant"]: row for row in summary.to_dict(orient="records")}
projection_gain_vs_cp8 = float(metrics["cp8"]["rmse"] - metrics["projection_crossfit"]["rmse"])
projection_gain_vs_control = float(metrics["no_projection_crossfit"]["rmse"] - metrics["projection_crossfit"]["rmse"])
shuffle_degradation = float(metrics["shuffled_projection_crossfit"]["rmse"] - metrics["projection_crossfit"]["rmse"])
oracle_gain = float(metrics["cp8"]["rmse"] - metrics["bank_oracle"]["rmse"])

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_candidate_conditioned_hidden_GR_TVT_bin_projection_gateway",
    "sample_unit": "one complete physical well with candidate paths",
    "legal_inputs_only": True,
    "truth_usage": "candidate MSE labels and final metrics only",
    "source": str(V18_PATH),
    "parent_column": PARENT_COLUMN,
    "cp_column": CP_COLUMN,
    "candidate_bank": {
        "sources": list(SOURCE_NAMES),
        "offsets_ft": OFFSETS_FT.tolist(),
        "offset_ramp_ft": OFFSET_RAMP_FT,
        "candidates_per_well": int(len(SOURCE_NAMES) * len(OFFSETS_FT)),
        "bin_widths_ft": list(BIN_WIDTHS_FT),
    },
    "summary": metrics,
    "direct_parent_cp8_signal": pair_report.to_dict(orient="records"),
    "gains": {
        "bank_oracle_vs_cp8": oracle_gain,
        "projection_crossfit_vs_cp8": projection_gain_vs_cp8,
        "projection_crossfit_vs_no_projection": projection_gain_vs_control,
        "shuffled_projection_degradation": shuffle_degradation,
    },
    "gates": {
        "bank_oracle_gain_ge_0p30": bool(oracle_gain >= 0.30),
        "projection_gain_vs_cp8_ge_0p05": bool(projection_gain_vs_cp8 >= 0.05),
        "projection_gain_vs_control_ge_0p03": bool(projection_gain_vs_control >= 0.03),
        "shuffle_degradation_ge_0p03": bool(shuffle_degradation >= 0.03),
    },
    "strict_pass": bool(
        oracle_gain >= 0.30
        and projection_gain_vs_cp8 >= 0.05
        and projection_gain_vs_control >= 0.03
        and shuffle_degradation >= 0.03
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["next"] = (
    "PASS -> move TVT-bin projection features into the CP8 control-point residual head"
    if verdict["strict_pass"]
    else "FAIL -> aggregation does not select deployable paths beyond geometry/control; do not add it to CP8"
)
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nTVT-BIN PROJECTION SUMMARY", flush=True)
print(summary.to_string(index=False), flush=True)
print("\nDIRECT PARENT/CP8 SIGNAL", flush=True)
print(pair_report.to_string(index=False), flush=True)
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
