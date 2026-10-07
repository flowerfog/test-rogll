"""Strict one-fold, full-trajectory GR matcher gateway.

Folds 2-4 build an anchored residual-shape menu, fold 5 trains a listwise GR
path ranker, and fold 1 is untouched validation.  No validation well informs
the candidate bank or the ranker.  This deliberately tests whole-path GR
matching rather than local row emissions.
"""

from pathlib import Path
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
FOLD_SOURCE = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
OUT_DIR = OOF_ROOT / "gr_path_matcher_onefold_v0"

BANK_FOLDS = (2, 3, 4)
RANKER_TRAIN_FOLD = 5
VALID_FOLD = 1

PROGRESS_POINTS = 96
PROGRESS_GRID = np.linspace(0.0, 1.0, PROGRESS_POINTS, dtype=np.float32)
KMEANS_CLUSTERS = 32
KMEANS_SCALES = (0.75, 1.0, 1.25)
PCA_COMPONENTS = 6
PCA_LEVELS = (-2.0, -1.0, 1.0, 2.0)
LINE_QUANTILES = np.linspace(0.02, 0.98, 17)
LENGTH_ALPHA = 0.5
FEATURE_MD_STEP_FT = 8.0
SMOOTH_SCALES_FT = (32, 64, 128, 256, 512, 1024)
SEGMENTS = 8
RANK_LEVELS = 16

PASS_GAIN_VS_STRONG = 0.30
PASS_POSITIVE_WELL_FRACTION = 0.55
PASS_MEDIAN_WELL_GAIN = 0.10
PASS_MEAN_REGRET = 1.50


def _rmse(truth, prediction):
    truth = np.asarray(truth, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _smooth_shape(progress, residual):
    frame = pd.DataFrame(
        {"progress": np.asarray(progress, float), "residual": np.asarray(residual, float)}
    )
    frame = frame[np.isfinite(frame).all(axis=1)]
    frame = frame.groupby("progress", as_index=False)["residual"].median().sort_values("progress")
    x = np.concatenate([[0.0], frame["progress"].to_numpy(float)])
    y = np.concatenate([[0.0], frame["residual"].to_numpy(float)])
    x, indices = np.unique(x, return_index=True)
    y = y[indices]
    shape = np.interp(PROGRESS_GRID, x, y)
    shape = savgol_filter(shape, 11, 2, mode="interp")
    return (shape - shape[0]).astype(np.float32)


def _deduplicate(templates, alphas, names):
    output_templates, output_alphas, output_names = [], [], []
    seen = set()
    for template, alpha, name in zip(templates, alphas, names):
        template = np.asarray(template, np.float32)
        template -= template[0]
        key = (round(float(alpha), 3), np.round(template, 2).tobytes())
        if key in seen:
            continue
        seen.add(key)
        output_templates.append(template)
        output_alphas.append(float(alpha))
        output_names.append(str(name))
    return (
        np.stack(output_templates).astype(np.float32),
        np.asarray(output_alphas, np.float32),
        np.asarray(output_names, dtype=str),
    )


def _build_bank(wells):
    shapes = np.stack([WELL_DATA[well]["shape"] for well in wells]).astype(np.float32)
    lengths = np.asarray([WELL_DATA[well]["eval_span"] for well in wells], float)
    median_length = float(np.median(lengths))
    ratios = np.maximum(lengths / max(median_length, 1.0), 0.1)
    normalised = shapes / ratios[:, None] ** LENGTH_ALPHA
    cap = float(max(64.0, min(192.0, np.quantile(np.abs(shapes), 0.997))))

    templates = [np.zeros(PROGRESS_POINTS, np.float32)]
    alphas = [0.0]
    names = ["zero"]
    for index, amplitude in enumerate(np.quantile(shapes[:, -1], LINE_QUANTILES)):
        templates.append((amplitude * PROGRESS_GRID).astype(np.float32))
        alphas.append(0.0)
        names.append(f"line_q{index:02d}")
    for family, values, alpha in (("raw", shapes, 0.0), ("lenhalf", normalised, LENGTH_ALPHA)):
        model = KMeans(
            n_clusters=min(KMEANS_CLUSTERS, max(2, len(values) // 8)),
            random_state=20260716,
            n_init=10,
        ).fit(values)
        for cluster, centre in enumerate(model.cluster_centers_):
            for scale in KMEANS_SCALES:
                templates.append((centre * scale).astype(np.float32))
                alphas.append(alpha)
                names.append(f"{family}_k{cluster:02d}_s{scale:.2f}")
    pca = PCA(n_components=min(PCA_COMPONENTS, len(normalised) - 1), random_state=20260716).fit(normalised)
    templates.append(pca.mean_.astype(np.float32))
    alphas.append(LENGTH_ALPHA)
    names.append("pca_mean")
    for component in range(len(pca.components_)):
        sigma = float(np.sqrt(max(pca.explained_variance_[component], 0.0)))
        for level in PCA_LEVELS:
            templates.append((pca.mean_ + level * sigma * pca.components_[component]).astype(np.float32))
            alphas.append(LENGTH_ALPHA)
            names.append(f"pca_c{component:02d}_l{level:+.0f}")
    templates, alphas, names = _deduplicate(templates, alphas, names)
    return {
        "templates": templates,
        "alphas": alphas,
        "names": names,
        "median_length": median_length,
        "cap": cap,
    }


def _fit_affine(horizontal_gr, typewell_gr):
    x = np.asarray(horizontal_gr, float)
    y = np.asarray(typewell_gr, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 32 or np.std(x) < 1e-6:
        return 1.0, 0.0
    keep = np.ones(len(x), bool)
    slope, intercept = 1.0, float(np.median(y) - np.median(x))
    for _ in range(4):
        design = np.column_stack([x[keep], np.ones(keep.sum())])
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        centre = float(np.median(residual))
        scale = float(1.4826 * np.median(np.abs(residual - centre)) + 1e-6)
        keep = np.abs(residual - centre) <= 3.5 * scale
        if keep.sum() < 24:
            break
    return float(np.clip(slope, -5.0, 5.0)), float(intercept)


def _masked_metrics(reference, observed, valid, prefix):
    reference = np.asarray(reference, np.float32)
    observed = np.asarray(observed, np.float32)
    valid = np.asarray(valid, bool)
    weight = valid.astype(np.float32)
    count = np.maximum(weight.sum(axis=1), 1.0)
    difference = (reference - observed) * weight
    mean_x = (reference * weight).sum(axis=1) / count
    mean_y = (observed * weight).sum(axis=1) / count
    centred_x = (reference - mean_x[:, None]) * weight
    centred_y = (observed - mean_y[:, None]) * weight
    covariance = (centred_x * centred_y).sum(axis=1)
    denominator = np.sqrt(
        np.maximum((centred_x * centred_x).sum(axis=1) * (centred_y * centred_y).sum(axis=1), 1e-8)
    )
    absolute = np.where(valid, np.abs(reference - observed), np.nan)
    return {
        f"{prefix}_coverage": valid.mean(axis=1),
        f"{prefix}_bias": difference.sum(axis=1) / count,
        f"{prefix}_mae": np.nansum(absolute, axis=1) / count,
        f"{prefix}_rmse": np.sqrt((difference * difference).sum(axis=1) / count),
        f"{prefix}_p90": np.nanquantile(absolute, 0.90, axis=1),
        f"{prefix}_ncc": covariance / denominator,
    }


def _moving_average(values, valid, size):
    weight = valid.astype(np.float32)
    count = uniform_filter1d(weight, size=size, axis=1, mode="constant", cval=0.0) * size
    total = uniform_filter1d(values * weight, size=size, axis=1, mode="constant", cval=0.0) * size
    output = total / np.maximum(count, 1.0)
    output_valid = count >= max(4, int(np.ceil(0.5 * size)))
    return output.astype(np.float32), output_valid


def _candidate_corrections(data, progress):
    ratio = max(data["eval_span"] / max(BANK["median_length"], 1.0), 0.1)
    scale = ratio ** BANK["alphas"]
    corrections = np.stack(
        [np.interp(progress, PROGRESS_GRID, template) for template in BANK["templates"]]
    )
    corrections *= scale[:, None]
    return np.clip(corrections, -BANK["cap"], BANK["cap"]).astype(np.float32)


def _path_features(well):
    data = WELL_DATA[well]
    horizontal = data["horizontal"]
    typewell = data["typewell"]
    md = horizontal["MD"].to_numpy(float)
    z = horizontal["Z"].to_numpy(float)
    gr_raw = horizontal["GR"].to_numpy(float)
    tvt_input = horizontal["TVT_input"].to_numpy(float)
    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr_raw = typewell["GR"].to_numpy(float)
    rows = data["row_idx"]
    anchor = int(rows.min() - 1)
    anchor_md = float(md[anchor])
    end_md = float(md[rows.max()])
    sample_md = np.arange(float(md[rows.min()]), end_md + 0.5 * FEATURE_MD_STEP_FT, FEATURE_MD_STEP_FT)
    if sample_md[-1] < end_md:
        sample_md = np.append(sample_md, end_md)
    progress = np.clip((sample_md - anchor_md) / max(end_md - anchor_md, 1.0), 0.0, 1.0)
    strong = np.interp(sample_md, md[rows], data["strong"])
    z_sample = np.interp(sample_md, md, z)

    known = np.isfinite(tvt_input) & np.isfinite(gr_raw)
    known_tw_valid = known & (tvt_input >= tw_tvt[0]) & (tvt_input <= tw_tvt[-1])
    known_tw = np.interp(tvt_input[known_tw_valid], tw_tvt, tw_gr_raw)
    slope, intercept = _fit_affine(gr_raw[known_tw_valid], known_tw)
    calibrated_gr = slope * gr_raw + intercept
    observed_valid_rows = np.isfinite(calibrated_gr)
    if observed_valid_rows.sum() < 2:
        observed_fill = np.zeros(len(md), float)
    else:
        observed_fill = np.interp(md, md[observed_valid_rows], calibrated_gr[observed_valid_rows])
    observed = np.interp(sample_md, md, observed_fill).astype(np.float32)
    observed_valid = np.interp(sample_md, md, observed_valid_rows.astype(float)) >= 0.5

    calibration_values = known_tw if len(known_tw) >= 16 else tw_gr_raw[np.isfinite(tw_gr_raw)]
    location = float(np.median(calibration_values))
    scale = float(1.4826 * np.median(np.abs(calibration_values - location)) + 1e-3)
    observed = np.clip((observed - location) / scale, -8.0, 8.0).astype(np.float32)
    tw_gr = np.clip((tw_gr_raw - location) / scale, -8.0, 8.0).astype(np.float32)

    corrections = _candidate_corrections(data, progress)
    candidate_tvt = strong[None, :] + corrections
    candidate_valid = (
        (candidate_tvt >= tw_tvt[0])
        & (candidate_tvt <= tw_tvt[-1])
        & observed_valid[None, :]
    )
    candidate_gr = np.interp(candidate_tvt.ravel(), tw_tvt, tw_gr).reshape(candidate_tvt.shape).astype(np.float32)
    observed_grid = np.broadcast_to(observed[None, :], candidate_gr.shape).astype(np.float32)

    features = {}
    features.update(_masked_metrics(candidate_gr, observed_grid, candidate_valid, "raw"))
    smoothed = {}
    for scale_ft in SMOOTH_SCALES_FT:
        size = max(3, int(round(scale_ft / FEATURE_MD_STEP_FT)))
        if size % 2 == 0:
            size += 1
        reference_smooth, reference_valid = _moving_average(candidate_gr, candidate_valid, size)
        observed_smooth, observed_smooth_valid = _moving_average(observed_grid, candidate_valid, size)
        smooth_valid = reference_valid & observed_smooth_valid
        prefix = f"s{scale_ft}"
        features.update(_masked_metrics(reference_smooth, observed_smooth, smooth_valid, prefix))
        smoothed[scale_ft] = (reference_smooth, observed_smooth, smooth_valid)

    reference_128, observed_128, valid_128 = smoothed[128]
    segment_ncc = []
    segment_mae = []
    for segment in range(SEGMENTS):
        left = segment / SEGMENTS
        right = (segment + 1) / SEGMENTS
        mask = (progress >= left) & (progress <= right if segment == SEGMENTS - 1 else progress < right)
        metrics = _masked_metrics(
            reference_128[:, mask], observed_128[:, mask], valid_128[:, mask], "segment"
        )
        segment_ncc.append(metrics["segment_ncc"])
        segment_mae.append(metrics["segment_mae"])
    segment_ncc = np.stack(segment_ncc, axis=1)
    segment_mae = np.stack(segment_mae, axis=1)
    features["segment_ncc_mean"] = np.nanmean(segment_ncc, axis=1)
    features["segment_ncc_min"] = np.nanmin(segment_ncc, axis=1)
    features["segment_ncc_std"] = np.nanstd(segment_ncc, axis=1)
    features["segment_mae_mean"] = np.nanmean(segment_mae, axis=1)
    features["segment_mae_max"] = np.nanmax(segment_mae, axis=1)
    features["segment_mae_std"] = np.nanstd(segment_mae, axis=1)

    for scale_ft in (64, 256, 512):
        reference_smooth, observed_smooth, smooth_valid = smoothed[scale_ft]
        derivative_valid = smooth_valid[:, 1:] & smooth_valid[:, :-1]
        derivative_metrics = _masked_metrics(
            np.diff(reference_smooth, axis=1),
            np.diff(observed_smooth, axis=1),
            derivative_valid,
            f"d{scale_ft}",
        )
        features.update(derivative_metrics)

    features["correction_end"] = corrections[:, -1]
    features["correction_mean"] = corrections.mean(axis=1)
    features["correction_std"] = corrections.std(axis=1)
    features["correction_max_abs"] = np.max(np.abs(corrections), axis=1)
    features["correction_tv"] = np.mean(np.abs(np.diff(corrections, axis=1)), axis=1)
    features["correction_curvature"] = np.mean(np.abs(np.diff(corrections, n=2, axis=1)), axis=1)
    features["calibration_slope"] = np.full(len(corrections), slope)
    features["calibration_intercept_scaled"] = np.full(len(corrections), intercept / scale)
    features["known_frac"] = np.full(len(corrections), data["known_frac"])
    features["gr_missing_frac"] = np.full(len(corrections), data["gr_missing_frac"])
    features["eval_span_kft"] = np.full(len(corrections), data["eval_span"] / 1000.0)

    full_corrections = _candidate_corrections(data, data["progress"])
    candidate_rmse = np.sqrt(np.mean((full_corrections - data["residual"][None, :]) ** 2, axis=1))
    family = np.asarray([name.split("_")[0] for name in BANK["names"]])
    frame = pd.DataFrame(features)
    frame.insert(0, "candidate_index", np.arange(len(BANK["names"]), dtype=np.int16))
    frame.insert(1, "candidate_name", BANK["names"])
    frame.insert(2, "family", family)
    frame.insert(3, "well", well)
    frame["fold"] = data["fold"]
    frame["candidate_rmse"] = candidate_rmse
    rank = np.argsort(np.argsort(candidate_rmse))
    frame["relevance"] = np.clip(
        RANK_LEVELS - 1 - np.floor(rank * RANK_LEVELS / len(rank)).astype(int),
        0,
        RANK_LEVELS - 1,
    )
    for family_name in ("zero", "line", "raw", "lenhalf", "pca"):
        frame[f"family_{family_name}"] = (family == family_name).astype(np.int8)
    return frame


def _add_group_ranks(frame):
    frame = frame.copy()
    ascending_costs = [
        "raw_mae", "raw_rmse", "s32_rmse", "s128_rmse", "s256_rmse",
        "s512_rmse", "s1024_rmse", "segment_mae_mean", "segment_mae_max",
        "d64_rmse", "d256_rmse", "d512_rmse",
    ]
    descending_scores = [
        "raw_ncc", "s32_ncc", "s128_ncc", "s256_ncc", "s512_ncc",
        "s1024_ncc", "segment_ncc_mean", "segment_ncc_min",
        "d64_ncc", "d256_ncc", "d512_ncc",
    ]
    for column in ascending_costs:
        frame[f"rank_{column}"] = frame.groupby("well")[column].rank(pct=True, ascending=True)
    for column in descending_scores:
        frame[f"rank_{column}"] = frame.groupby("well")[column].rank(pct=True, ascending=False)
    rank_columns = [column for column in frame.columns if column.startswith("rank_")]
    frame["heuristic_rank_mean"] = frame[rank_columns].mean(axis=1)
    return frame


def _selected_prediction(well, candidate_index):
    data = WELL_DATA[well]
    corrections = _candidate_corrections(data, data["progress"])
    return data["strong"] + corrections[int(candidate_index)]


def _evaluate_selector(valid_features, selector_name, selected_indices):
    parts = []
    bywell = []
    for well, candidate_index in selected_indices.items():
        data = WELL_DATA[well]
        prediction = _selected_prediction(well, candidate_index)
        parts.append(
            pd.DataFrame(
                {
                    "well": well,
                    "truth": data["truth"],
                    "strong": data["strong"],
                    "prediction": prediction,
                }
            )
        )
        strong_rmse = _rmse(data["truth"], data["strong"])
        prediction_rmse = _rmse(data["truth"], prediction)
        oracle_rmse = float(valid_features.loc[
            (valid_features["well"] == well), "candidate_rmse"
        ].min())
        bywell.append(
            {
                "selector": selector_name,
                "well": well,
                "rows": len(data["truth"]),
                "strong_rmse": strong_rmse,
                "prediction_rmse": prediction_rmse,
                "gain": strong_rmse - prediction_rmse,
                "oracle_rmse": oracle_rmse,
                "regret": prediction_rmse - oracle_rmse,
                "candidate_index": int(candidate_index),
                "candidate_name": str(BANK["names"][int(candidate_index)]),
            }
        )
    rows = pd.concat(parts, ignore_index=True)
    bywell = pd.DataFrame(bywell)
    truth = rows["truth"].to_numpy(float)
    strong = rows["strong"].to_numpy(float)
    prediction = rows["prediction"].to_numpy(float)
    report = {
        "selector": selector_name,
        "rows": len(rows),
        "wells": len(bywell),
        "strong_rmse": _rmse(truth, strong),
        "rmse": _rmse(truth, prediction),
        "gain_vs_strong": _rmse(truth, strong) - _rmse(truth, prediction),
        "mean_well_rmse": float(bywell["prediction_rmse"].mean()),
        "mean_well_gain": float(bywell["gain"].mean()),
        "median_well_gain": float(bywell["gain"].median()),
        "positive_well_fraction": float((bywell["gain"] > 0).mean()),
        "mean_regret": float(bywell["regret"].mean()),
        "median_regret": float(bywell["regret"].median()),
    }
    return report, rows, bywell


for path in (DATA_ROOT / "train", FOLD_SOURCE):
    if not path.exists():
        raise FileNotFoundError(path)
pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
OUT_DIR.mkdir(parents=True, exist_ok=True)

header = pd.read_csv(pool_path, nrows=0).columns.tolist()
usecols = [column for column in ("id", "well", "row_idx", "tvt_true", "strong") if column in header]
pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pool["row_idx"].astype(np.int32)
fold_map = pd.read_csv(FOLD_SOURCE, dtype={"well": str}).set_index("well")["fold"].astype(int).to_dict()
pool["fold"] = pool["well"].map(fold_map).astype(int)

started = time.time()
WELL_DATA = {}
failures = []
needed_folds = set(BANK_FOLDS) | {RANKER_TRAIN_FOLD, VALID_FOLD}
for well, frame in tqdm(pool.groupby("well", sort=False), desc="load GR path wells"):
    if int(frame["fold"].iloc[0]) not in needed_folds:
        continue
    try:
        frame = frame.sort_values("row_idx")
        horizontal = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
            usecols=["MD", "Z", "GR", "TVT_input"],
        )
        typewell = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__typewell.csv",
            usecols=["TVT", "GR"],
        )
        for column in ("MD", "Z", "GR", "TVT_input"):
            horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
        typewell = typewell.apply(pd.to_numeric, errors="coerce").dropna(subset=["TVT"])
        typewell = typewell.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
        typewell["GR"] = typewell["GR"].interpolate(limit_direction="both").fillna(0.0)
        rows = frame["row_idx"].to_numpy(int)
        md = horizontal["MD"].to_numpy(float)
        anchor = int(rows.min() - 1)
        eval_span = float(max(md[rows.max()] - md[anchor], 1.0))
        progress = np.clip((md[rows] - md[anchor]) / eval_span, 0.0, 1.0)
        truth = frame["tvt_true"].to_numpy(float)
        strong = frame["strong"].to_numpy(float)
        residual = truth - strong
        WELL_DATA[str(well)] = {
            "fold": int(frame["fold"].iloc[0]),
            "row_idx": rows,
            "progress": progress.astype(np.float32),
            "truth": truth.astype(np.float32),
            "strong": strong.astype(np.float32),
            "residual": residual.astype(np.float32),
            "shape": _smooth_shape(progress, residual),
            "eval_span": eval_span,
            "known_frac": float(horizontal["TVT_input"].notna().mean()),
            "gr_missing_frac": float(horizontal["GR"].isna().mean()),
            "horizontal": horizontal,
            "typewell": typewell,
        }
    except Exception as exc:
        failures.append({"well": str(well), "reason": repr(exc)})

pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)
if failures:
    raise RuntimeError(f"failed wells: {len(failures)}; see failures.csv")

bank_wells = [well for well, data in WELL_DATA.items() if data["fold"] in BANK_FOLDS]
train_wells = [well for well, data in WELL_DATA.items() if data["fold"] == RANKER_TRAIN_FOLD]
valid_wells = [well for well, data in WELL_DATA.items() if data["fold"] == VALID_FOLD]
BANK = _build_bank(bank_wells)
np.savez_compressed(
    OUT_DIR / "restricted_path_bank.npz",
    templates=BANK["templates"],
    length_alphas=BANK["alphas"],
    names=BANK["names"],
    progress_grid=PROGRESS_GRID,
    median_length=np.asarray([BANK["median_length"]], np.float32),
    cap=np.asarray([BANK["cap"]], np.float32),
    bank_wells=np.asarray(bank_wells, dtype=str),
)
print(
    {
        "experiment": "gr_path_matcher_onefold_v0",
        "bank_folds": BANK_FOLDS,
        "ranker_train_fold": RANKER_TRAIN_FOLD,
        "valid_fold": VALID_FOLD,
        "bank_wells": len(bank_wells),
        "train_wells": len(train_wells),
        "valid_wells": len(valid_wells),
        "candidates": len(BANK["names"]),
    }
)

feature_parts = []
for well in tqdm(train_wells + valid_wells, desc="whole-path GR features"):
    feature_parts.append(_path_features(well))
features = pd.concat(feature_parts, ignore_index=True)
features = _add_group_ranks(features)
features.to_pickle(OUT_DIR / "path_features.pkl")

train = features[features["fold"] == RANKER_TRAIN_FOLD].copy()
valid = features[features["fold"] == VALID_FOLD].copy()
train = train.sort_values(["well", "candidate_index"]).reset_index(drop=True)
valid = valid.sort_values(["well", "candidate_index"]).reset_index(drop=True)
excluded = {
    "well", "candidate_name", "family", "fold", "candidate_rmse", "relevance",
}
feature_columns = [column for column in features.columns if column not in excluded]
for frame in (train, valid):
    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
train_group = train.groupby("well", sort=False).size().to_list()
valid_group = valid.groupby("well", sort=False).size().to_list()

ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=2500,
    learning_rate=0.025,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.85,
    colsample_bytree=0.80,
    reg_lambda=2.0,
    random_state=20260716,
    n_jobs=-1,
    verbosity=-1,
)
ranker.fit(
    train[feature_columns],
    train["relevance"].astype(int),
    group=train_group,
    eval_set=[(valid[feature_columns], valid["relevance"].astype(int))],
    eval_group=[valid_group],
    eval_at=[1, 3, 5, 10],
    callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(100)],
)
joblib.dump(
    {"model": ranker, "feature_columns": feature_columns, "bank": BANK},
    OUT_DIR / "ranker_bundle.joblib",
    compress=3,
)
valid["ranker_score"] = ranker.predict(valid[feature_columns], num_iteration=ranker.best_iteration_)

selectors = {}
selectors["oracle"] = valid.loc[valid.groupby("well")["candidate_rmse"].idxmin()].set_index("well")["candidate_index"].to_dict()
selectors["heuristic"] = valid.loc[valid.groupby("well")["heuristic_rank_mean"].idxmin()].set_index("well")["candidate_index"].to_dict()
selectors["ranker"] = valid.loc[valid.groupby("well")["ranker_score"].idxmax()].set_index("well")["candidate_index"].to_dict()

reports = []
bywell_parts = []
selected_rows = None
for selector, selected in selectors.items():
    report, rows, bywell = _evaluate_selector(valid, selector, selected)
    reports.append(report)
    bywell_parts.append(bywell)
    if selector == "ranker":
        selected_rows = rows
reports = pd.DataFrame(reports).sort_values("rmse")
bywell_report = pd.concat(bywell_parts, ignore_index=True)
reports.to_csv(OUT_DIR / "selector_summary.csv", index=False)
bywell_report.to_csv(OUT_DIR / "selector_bywell.csv", index=False)
valid.to_pickle(OUT_DIR / "valid_candidate_scores.pkl")
selected_rows.to_pickle(OUT_DIR / "ranker_selected_oof.pkl")

ranker_report = reports.set_index("selector").loc["ranker"].to_dict()
strict_pass = bool(
    ranker_report["gain_vs_strong"] >= PASS_GAIN_VS_STRONG
    and ranker_report["positive_well_fraction"] >= PASS_POSITIVE_WELL_FRACTION
    and ranker_report["median_well_gain"] >= PASS_MEDIAN_WELL_GAIN
    and ranker_report["mean_regret"] <= PASS_MEAN_REGRET
)
importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance_gain": ranker.booster_.feature_importance(importance_type="gain"),
        "importance_split": ranker.booster_.feature_importance(importance_type="split"),
    }
).sort_values("importance_gain", ascending=False)
importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)
verdict = {
    "experiment": "gr_path_matcher_onefold_v0",
    "bank_folds": list(BANK_FOLDS),
    "ranker_train_fold": RANKER_TRAIN_FOLD,
    "valid_fold": VALID_FOLD,
    "bank_wells": len(bank_wells),
    "train_wells": len(train_wells),
    "valid_wells": len(valid_wells),
    "candidates": len(BANK["names"]),
    "features": len(feature_columns),
    "best_iteration": int(ranker.best_iteration_),
    "ranker": ranker_report,
    "gates": {
        "gain_vs_strong_min": PASS_GAIN_VS_STRONG,
        "positive_well_fraction_min": PASS_POSITIVE_WELL_FRACTION,
        "median_well_gain_min": PASS_MEDIAN_WELL_GAIN,
        "mean_regret_max": PASS_MEAN_REGRET,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> extend to nested five-fold pseudo-PS GR path ranking"
        if strict_pass
        else "FAIL -> whole-path GR features cannot select the high-ceiling menu"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nSELECTORS")
print(reports.to_string(index=False))
print("\nTOP IMPORTANCE")
print(importance.head(40).to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
