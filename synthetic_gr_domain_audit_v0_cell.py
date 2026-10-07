"""Fold-clean domain audit for a synthetic GR pretraining generator.

Generator templates are built only from outer-train folds 2-5.  Untouched
fold 1 supplies the real-domain audit segments.  Two synthetic arms are
compared:

* naive: truth-aligned typewell GR plus white noise and a borrowed gap mask;
* residual_block: typewell GR plus a correlated, calibrated residual block and
  the exact missingness pattern borrowed from another outer-train segment.

The cell does not train CorrNet.  It decides whether a 50k-100k synthetic
pretraining run is defensible before spending GPU time and disk.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm


ROOT = Path("/root/autodl-tmp")
DATA_ROOT_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    Path.cwd(),
)
DATA_ROOT = next(
    (path for path in DATA_ROOT_CANDIDATES if (path / "train").exists()),
    DATA_ROOT_CANDIDATES[0],
)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").exists() else DATA_ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "synthetic_gr_domain_audit_v0"
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)

VALID_FOLD = 1
POINTS = 128
SEGMENTS_PER_TRAIN_WELL = 4
SEGMENTS_PER_VALID_WELL = 8
SEGMENT_SPANS_FT = (300.0, 450.0, 600.0)
SHIFT_OFFSETS_FT = np.arange(-48.0, 48.01, 4.0)
RANDOM_STATE = 20260717
N_DOMAIN_TREES = 400


def _first_existing(paths):
    return next((path for path in paths if path.exists()), None)


def _robust_parameters(values):
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    if int(valid.sum()) < 4:
        return 0.0, 1.0
    centre = float(np.median(values[valid]))
    scale = float(1.4826 * np.median(np.abs(values[valid] - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values[valid]), 1.0))
    return centre, scale


def _normalise(values, centre, scale):
    values = np.asarray(values, np.float64)
    return np.clip((values - centre) / max(scale, 1e-6), -8.0, 8.0)


def _interp_finite(x, y, query, left=np.nan, right=np.nan):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    query = np.asarray(query, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return np.full(query.shape, np.nan, np.float64)
    order = np.argsort(x[valid], kind="mergesort")
    xv = x[valid][order]
    yv = y[valid][order]
    unique, inverse = np.unique(xv, return_inverse=True)
    if len(unique) != len(xv):
        yv = np.bincount(inverse, weights=yv) / np.maximum(np.bincount(inverse), 1)
        xv = unique
    return np.interp(query, xv, yv, left=left, right=right)


def _nearest_distance(x, query):
    x = np.sort(np.asarray(x, np.float64)[np.isfinite(x)])
    query = np.asarray(query, np.float64)
    if not len(x):
        return np.full(query.shape, np.inf)
    position = np.searchsorted(x, query)
    left = np.abs(query - x[np.clip(position - 1, 0, len(x) - 1)])
    right = np.abs(query - x[np.clip(position, 0, len(x) - 1)])
    return np.minimum(left, right)


def _correlation(left, right, valid):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 12:
        return 0.0
    x = left[valid] - np.mean(left[valid])
    y = right[valid] - np.mean(right[valid])
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 1e-8 else 0.0


def _autocorrelation(values, valid, lag):
    values = np.asarray(values, np.float64)
    valid = np.asarray(valid, bool)
    if lag >= len(values):
        return 0.0
    return _correlation(values[:-lag], values[lag:], valid[:-lag] & valid[lag:])


def _longest_missing(mask):
    missing = ~np.asarray(mask, bool)
    longest = current = 0
    for value in missing:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _spectral_features(values, valid, prefix):
    values = np.asarray(values, np.float64)
    valid = np.asarray(valid, bool)
    if int(valid.sum()) < 8:
        return {
            f"{prefix}_spectral_low": 0.0,
            f"{prefix}_spectral_mid": 0.0,
            f"{prefix}_spectral_high": 0.0,
        }
    indices = np.arange(len(values))
    filled = np.interp(indices, indices[valid], values[valid])
    filled = filled - np.mean(filled)
    power = np.abs(np.fft.rfft(filled)) ** 2
    total = float(np.sum(power[1:]) + 1e-8)
    n = len(power)
    low = float(np.sum(power[1 : max(2, n // 8)]) / total)
    mid = float(np.sum(power[max(2, n // 8) : max(3, n // 3)]) / total)
    high = float(np.sum(power[max(3, n // 3) :]) / total)
    return {
        f"{prefix}_spectral_low": low,
        f"{prefix}_spectral_mid": mid,
        f"{prefix}_spectral_high": high,
    }


def _sample_positions(start_md, end_md, count, well_index):
    rows = []
    available = float(end_md - start_md)
    if available <= 100.0:
        return rows
    for slot in range(count):
        span = min(SEGMENT_SPANS_FT[(slot + well_index) % len(SEGMENT_SPANS_FT)], available)
        unit = (slot + 0.5) / count
        left = start_md + unit * max(available - span, 0.0)
        rows.append((float(left), float(min(left + span, end_md))))
    return rows


def _load_well(well):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    required = {"MD", "GR", "TVT", "TVT_input"}
    if not required.issubset(horizontal.columns):
        raise RuntimeError(f"{well}: horizontal misses {sorted(required-set(horizontal.columns))}")
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
    truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    tw = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
    tw = tw.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr_raw = tw["GR"].to_numpy(float)
    if len(tw_tvt) < 32:
        raise RuntimeError(f"{well}: typewell too short")
    h_centre, h_scale = _robust_parameters(gr)
    tw_centre, tw_scale = _robust_parameters(tw_gr_raw)
    gr_norm = _normalise(gr, h_centre, h_scale)
    tw_norm = _normalise(tw_gr_raw, tw_centre, tw_scale)
    known = np.flatnonzero(np.isfinite(tvt_input))
    hidden = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
    if len(known) < 64 or len(hidden) < POINTS:
        raise RuntimeError(f"{well}: invalid official prefix/eval")
    return {
        "well": well,
        "md": md,
        "gr": gr_norm,
        "gr_finite": np.isfinite(gr),
        "truth": truth,
        "tvt_input": tvt_input,
        "known": known,
        "hidden": hidden,
        "tw_tvt": tw_tvt,
        "tw_gr": tw_norm,
        "md_step": float(np.nanmedian(np.diff(md[np.isfinite(md)]))),
    }


def _extract_segment(record, left_md, right_md, segment_id):
    query_md = np.linspace(left_md, right_md, POINTS)
    finite_gr_md = record["md"][record["gr_finite"]]
    horizontal = _interp_finite(
        finite_gr_md,
        record["gr"][record["gr_finite"]],
        query_md,
        left=np.nan,
        right=np.nan,
    )
    horizontal_valid = (
        _nearest_distance(finite_gr_md, query_md)
        <= max(6.0, 2.5 * record["md_step"])
    )
    horizontal = np.nan_to_num(horizontal, nan=0.0)
    truth_tvt = _interp_finite(record["md"], record["truth"], query_md)
    aligned_typewell = _interp_finite(
        record["tw_tvt"], record["tw_gr"], truth_tvt, left=np.nan, right=np.nan
    )
    typewell_valid = np.isfinite(aligned_typewell)
    aligned_typewell = np.nan_to_num(aligned_typewell, nan=0.0)
    valid = horizontal_valid & typewell_valid
    residual = horizontal - aligned_typewell
    residual[~valid] = 0.0
    return {
        "well": record["well"],
        "segment_id": segment_id,
        "query_md": query_md.astype(np.float32),
        "truth_tvt": truth_tvt.astype(np.float32),
        "horizontal": horizontal.astype(np.float32),
        "horizontal_valid": horizontal_valid.astype(bool),
        "aligned_typewell": aligned_typewell.astype(np.float32),
        "typewell_valid": typewell_valid.astype(bool),
        "residual": residual.astype(np.float32),
        "tw_tvt": record["tw_tvt"].astype(np.float32),
        "tw_gr": record["tw_gr"].astype(np.float32),
        "span_ft": float(right_md - left_md),
    }


def _segment_features(sample, domain, method):
    horizontal = np.asarray(sample["horizontal"], np.float64)
    horizontal_valid = np.asarray(sample["horizontal_valid"], bool)
    reference = np.asarray(sample["aligned_typewell"], np.float64)
    typewell_valid = np.asarray(sample["typewell_valid"], bool)
    valid = horizontal_valid & typewell_valid
    residual = horizontal - reference
    finite_residual = residual[valid]
    if not len(finite_residual):
        finite_residual = np.asarray([0.0])

    shift_scores = []
    for offset in SHIFT_OFFSETS_FT:
        shifted = _interp_finite(
            sample["tw_tvt"],
            sample["tw_gr"],
            sample["truth_tvt"] + offset,
            left=np.nan,
            right=np.nan,
        )
        shifted_valid = np.isfinite(shifted)
        shift_scores.append(
            _correlation(horizontal, np.nan_to_num(shifted, nan=0.0), horizontal_valid & shifted_valid)
        )
    shift_scores = np.asarray(shift_scores, np.float64)
    order = np.argsort(shift_scores)[::-1]
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    positive = np.exp(4.0 * (shift_scores - np.max(shift_scores)))
    probability = positive / max(float(np.sum(positive)), 1e-8)
    entropy = float(-np.sum(probability * np.log(probability + 1e-12)))
    zero_index = int(np.argmin(np.abs(SHIFT_OFFSETS_FT)))

    row = {
        "well": sample["well"],
        "segment_id": sample["segment_id"],
        "domain": domain,
        "method": method,
        "span_ft": sample["span_ft"],
        "missing_fraction": float(1.0 - np.mean(horizontal_valid)),
        "longest_missing_fraction": _longest_missing(horizontal_valid) / POINTS,
        "typewell_coverage": float(np.mean(typewell_valid)),
        "truth_ncc": float(shift_scores[zero_index]),
        "best_ncc": float(shift_scores[best]),
        "best_offset_abs": float(abs(SHIFT_OFFSETS_FT[best])),
        "peak_margin": float(shift_scores[best] - shift_scores[second]),
        "profile_std": float(np.std(shift_scores)),
        "profile_entropy": entropy,
        "residual_mean": float(np.mean(finite_residual)),
        "residual_std": float(np.std(finite_residual)),
        "residual_mae": float(np.mean(np.abs(finite_residual))),
        "residual_p90": float(np.quantile(np.abs(finite_residual), 0.90)),
        "horizontal_d1_std": float(np.std(np.diff(horizontal)[horizontal_valid[1:] & horizontal_valid[:-1]]))
        if np.any(horizontal_valid[1:] & horizontal_valid[:-1]) else 0.0,
        "typewell_d1_std": float(np.std(np.diff(reference)[typewell_valid[1:] & typewell_valid[:-1]]))
        if np.any(typewell_valid[1:] & typewell_valid[:-1]) else 0.0,
        "residual_d1_std": float(np.std(np.diff(residual)[valid[1:] & valid[:-1]]))
        if np.any(valid[1:] & valid[:-1]) else 0.0,
    }
    for lag in (1, 2, 4, 8, 16):
        row[f"horizontal_ac{lag}"] = _autocorrelation(horizontal, horizontal_valid, lag)
        row[f"residual_ac{lag}"] = _autocorrelation(residual, valid, lag)
    row.update(_spectral_features(horizontal, horizontal_valid, "horizontal"))
    row.update(_spectral_features(residual, valid, "residual"))
    return row


def _make_synthetic(target, donor, method, rng):
    reference = np.asarray(target["aligned_typewell"], np.float64)
    typewell_valid = np.asarray(target["typewell_valid"], bool)
    donor_mask = np.asarray(donor["horizontal_valid"], bool)
    if method == "naive":
        donor_residual = donor["residual"][
            donor["horizontal_valid"] & donor["typewell_valid"]
        ]
        sigma = float(np.std(donor_residual)) if len(donor_residual) else 1.0
        synthetic = reference + rng.normal(0.0, max(sigma, 0.05), POINTS)
    elif method == "residual_block":
        residual = np.asarray(donor["residual"], np.float64).copy()
        valid_residual = donor["horizontal_valid"] & donor["typewell_valid"]
        if int(valid_residual.sum()) >= 2:
            positions = np.arange(POINTS)
            residual = np.interp(positions, positions[valid_residual], residual[valid_residual])
        else:
            residual[:] = 0.0
        scale = float(rng.uniform(0.85, 1.15))
        gain = float(rng.normal(1.0, 0.06))
        offset = float(rng.normal(0.0, 0.08))
        synthetic = gain * reference + offset + scale * residual
    else:
        raise ValueError(method)
    synthetic_valid = donor_mask & typewell_valid
    synthetic = np.clip(synthetic, -8.0, 8.0)
    synthetic[~synthetic_valid] = 0.0
    output = dict(target)
    output["horizontal"] = synthetic.astype(np.float32)
    output["horizontal_valid"] = synthetic_valid.astype(bool)
    return output


fold_path = _first_existing(FOLD_CANDIDATES)
if fold_path is None:
    raise FileNotFoundError("fold metadata not found")
OUT_DIR.mkdir(parents=True, exist_ok=True)
fold = pd.read_csv(fold_path, dtype={"well": str})
fold = fold.groupby("well", as_index=False)["fold"].first()
fold["well"] = fold["well"].astype(str)
fold["fold"] = fold["fold"].astype(int)
train_wells = fold.loc[fold["fold"] != VALID_FOLD, "well"].tolist()
valid_wells = fold.loc[fold["fold"] == VALID_FOLD, "well"].tolist()

print(
    {
        "experiment": "synthetic_gr_domain_audit_v0",
        "fold_source": str(fold_path),
        "train_wells": len(train_wells),
        "valid_wells": len(valid_wells),
        "points": POINTS,
        "shift_offsets": [float(SHIFT_OFFSETS_FT[0]), float(SHIFT_OFFSETS_FT[-1])],
    },
    flush=True,
)

started = time.time()
train_segments = []
valid_segments = []
failures = []
for split, wells, count in (
    ("train", train_wells, SEGMENTS_PER_TRAIN_WELL),
    ("valid", valid_wells, SEGMENTS_PER_VALID_WELL),
):
    destination = train_segments if split == "train" else valid_segments
    for well_index, well in enumerate(tqdm(wells, desc=f"build {split} GR segments")):
        try:
            record = _load_well(well)
            start_row = int(record["hidden"][0])
            start_md = float(record["md"][start_row])
            end_md = float(record["md"][record["hidden"][-1]])
            positions = _sample_positions(start_md, end_md, count, well_index)
            for segment_id, (left, right) in enumerate(positions):
                destination.append(_extract_segment(record, left, right, segment_id))
        except Exception as exc:
            failures.append({"split": split, "well": well, "reason": repr(exc)})

failure_frame = pd.DataFrame(failures)
failure_frame.to_csv(OUT_DIR / "failures.csv", index=False)
if len(valid_segments) < 800 or len(train_segments) < 1800:
    print(failure_frame.head(20).to_string(index=False))
    raise RuntimeError(
        f"too few segments: train={len(train_segments)}, valid={len(valid_segments)}"
    )

rng = np.random.default_rng(RANDOM_STATE)
feature_rows = []
for sample in tqdm(valid_segments, desc="real-domain features"):
    feature_rows.append(_segment_features(sample, "real", "real"))

for method in ("naive", "residual_block"):
    for index, real_sample in enumerate(tqdm(valid_segments, desc=f"{method} synthetic features")):
        target = train_segments[int(rng.integers(0, len(train_segments)))]
        donor = train_segments[int(rng.integers(0, len(train_segments)))]
        synthetic = _make_synthetic(target, donor, method, rng)
        synthetic["segment_id"] = index
        feature_rows.append(_segment_features(synthetic, "synthetic", method))

features = pd.DataFrame(feature_rows)
features.to_csv(OUT_DIR / "segment_features.csv", index=False)
metadata_columns = {"well", "segment_id", "domain", "method"}
feature_columns = [column for column in features.columns if column not in metadata_columns]

domain_rows = []
importance_rows = []
distance_rows = []
real = features[features["domain"] == "real"].copy()
for method in ("naive", "residual_block"):
    synthetic = features[features["method"] == method].copy()
    pair = pd.concat([real, synthetic], ignore_index=True)
    X = pair[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    y = (pair["domain"] == "synthetic").astype(int).to_numpy()
    groups = pair["domain"].astype(str) + ":" + pair["well"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    prediction = np.full(len(pair), np.nan, np.float64)
    fold_auc = []
    importances = []
    for fold_index, (train_index, valid_index) in enumerate(splitter.split(X, y, groups)):
        classifier = ExtraTreesClassifier(
            n_estimators=N_DOMAIN_TREES,
            min_samples_leaf=8,
            max_features=0.7,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE + fold_index,
        )
        classifier.fit(X[train_index], y[train_index])
        prediction[valid_index] = classifier.predict_proba(X[valid_index])[:, 1]
        fold_auc.append(float(roc_auc_score(y[valid_index], prediction[valid_index])))
        importances.append(classifier.feature_importances_)
    auc = float(roc_auc_score(y, prediction))
    domain_rows.append(
        {
            "method": method,
            "segments_per_domain": len(real),
            "domain_auc": auc,
            "mean_fold_auc": float(np.mean(fold_auc)),
            "min_fold_auc": float(np.min(fold_auc)),
            "max_fold_auc": float(np.max(fold_auc)),
        }
    )
    mean_importance = np.mean(importances, axis=0)
    for column, importance in zip(feature_columns, mean_importance):
        importance_rows.append(
            {"method": method, "feature": column, "importance": float(importance)}
        )
    for column in feature_columns:
        real_values = real[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        synthetic_values = synthetic[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        pooled_scale = float(
            np.sqrt(0.5 * (np.var(real_values) + np.var(synthetic_values))) + 1e-6
        )
        distance_rows.append(
            {
                "method": method,
                "feature": column,
                "real_mean": float(np.mean(real_values)),
                "synthetic_mean": float(np.mean(synthetic_values)),
                "standardised_mean_difference": float(
                    abs(np.mean(real_values) - np.mean(synthetic_values)) / pooled_scale
                ),
                "ks_statistic": float(ks_2samp(real_values, synthetic_values).statistic),
            }
        )

domain_report = pd.DataFrame(domain_rows).sort_values("domain_auc")
importance_report = pd.DataFrame(importance_rows).sort_values(
    ["method", "importance"], ascending=[True, False]
)
distance_report = pd.DataFrame(distance_rows)
domain_report.to_csv(OUT_DIR / "domain_classifier.csv", index=False)
importance_report.to_csv(OUT_DIR / "domain_feature_importance.csv", index=False)
distance_report.to_csv(OUT_DIR / "feature_distances.csv", index=False)

summary_rows = []
for method in ("naive", "residual_block"):
    local = distance_report[distance_report["method"] == method]
    synthetic = features[features["method"] == method]
    summary_rows.append(
        {
            "method": method,
            "domain_auc": float(
                domain_report.loc[domain_report["method"] == method, "domain_auc"].iloc[0]
            ),
            "median_standardised_mean_difference": float(
                local["standardised_mean_difference"].median()
            ),
            "p90_standardised_mean_difference": float(
                local["standardised_mean_difference"].quantile(0.90)
            ),
            "median_ks": float(local["ks_statistic"].median()),
            "real_truth_ncc_median": float(real["truth_ncc"].median()),
            "synthetic_truth_ncc_median": float(synthetic["truth_ncc"].median()),
            "truth_ncc_median_gap": float(
                abs(real["truth_ncc"].median() - synthetic["truth_ncc"].median())
            ),
            "real_peak_margin_median": float(real["peak_margin"].median()),
            "synthetic_peak_margin_median": float(synthetic["peak_margin"].median()),
            "real_missing_fraction": float(real["missing_fraction"].mean()),
            "synthetic_missing_fraction": float(synthetic["missing_fraction"].mean()),
            "missing_fraction_gap": float(
                abs(real["missing_fraction"].mean() - synthetic["missing_fraction"].mean())
            ),
        }
    )
summary = pd.DataFrame(summary_rows).sort_values("domain_auc")
summary.to_csv(OUT_DIR / "summary.csv", index=False)
calibrated = summary[summary["method"] == "residual_block"].iloc[0]
strict_pass = bool(
    float(calibrated["domain_auc"]) <= 0.75
    and float(calibrated["median_standardised_mean_difference"]) <= 0.25
    and float(calibrated["truth_ncc_median_gap"]) <= 0.10
    and float(calibrated["missing_fraction_gap"]) <= 0.05
)
verdict = {
    "experiment": "synthetic_gr_domain_audit_v0",
    "fold_source": str(fold_path),
    "generator_source_folds": [2, 3, 4, 5],
    "real_audit_fold": VALID_FOLD,
    "train_segments": len(train_segments),
    "real_segments": len(valid_segments),
    "feature_count": len(feature_columns),
    "calibrated_domain_auc": float(calibrated["domain_auc"]),
    "calibrated_truth_ncc_gap": float(calibrated["truth_ncc_median_gap"]),
    "calibrated_missing_gap": float(calibrated["missing_fraction_gap"]),
    "calibrated_median_standardised_difference": float(
        calibrated["median_standardised_mean_difference"]
    ),
    "strict_pass": strict_pass,
    "next": (
        "PASS -> build 50k residual-block synthetic bank and pretrain one-fold matcher"
        if strict_pass
        else "FAIL -> calibrate the generator using the reported dominant domain features before GPU pretraining"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nDOMAIN SUMMARY")
print(summary.to_string(index=False))
print("\nTOP DOMAIN FEATURES")
print(importance_report.groupby("method", group_keys=False).head(15).to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)

