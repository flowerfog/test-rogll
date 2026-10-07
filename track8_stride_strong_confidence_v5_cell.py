"""Nested legal per-well confidence gate for strong + Track8 + STRIDE.

V4 proved that Track8/STRIDE are complementary to each other, but fixed
weights do not survive once the fold-clean strong-neighbour endpoint is
present.  The remaining author-like hypothesis is per-well confidence.

This cell tests that hypothesis with deployable evidence only:
* pairwise endpoint disagreement, including distance strata;
* standalone SP45 PF/beam/selector uncertainty;
* frozen neighbour geometry diagnostics.

The target is used only on outer-training wells to fit confidence weights.
Every reported row is predicted by a gate trained without that physical well.
A matched within-fold shuffle of the learned weights is mandatory.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "track8_stride_strong_confidence_v5"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V4_DIR = OOF_ROOT / "track8_stride_strong_nested_fusion_v4"
ROW_PATH = V4_DIR / "best_nested_predictions.parquet"
V4_VERDICT_PATH = V4_DIR / "verdict.json"
SP45_PATH = OOF_ROOT / "sp45_foldclean_feature_export_v1" / "sp45_feature_oof.parquet"
NEIGHBOR_META_PATH = OOF_ROOT / "neighbor_fullcurve_transfer_v0" / "selected_neighbors.csv"

for required in (ROW_PATH, V4_VERDICT_PATH, SP45_PATH, NEIGHBOR_META_PATH):
    if not required.exists():
        raise FileNotFoundError(required)

SP45_DIRECT = [
    "pf_std",
    "beam_std",
    "pf_scale_range",
    "beam_range",
    "pf_beam_disagreement",
    "likelihood_max",
    "likelihood_mean",
    "likelihood_std",
    "likelihood_top_gap",
    "selector_code",
    "z_span",
    "gr_scale",
    "initial_rate",
]
SP45_PATHS = [
    "pf_mean",
    "beam_mean",
    "sp45_selector_raw",
    "sp45_selector_projected",
]

MODEL_CONFIGS = [
    {"name": "ridge_a10_d025", "family": "ridge", "alpha": 10.0, "dose": 0.25},
    {"name": "ridge_a10_d050", "family": "ridge", "alpha": 10.0, "dose": 0.50},
    {"name": "ridge_a100_d025", "family": "ridge", "alpha": 100.0, "dose": 0.25},
    {"name": "ridge_a100_d050", "family": "ridge", "alpha": 100.0, "dose": 0.50},
    {"name": "rf_d1_l48_d025", "family": "rf", "depth": 1, "leaf": 48, "dose": 0.25},
    {"name": "rf_d2_l48_d025", "family": "rf", "depth": 2, "leaf": 48, "dose": 0.25},
    {"name": "rf_d2_l48_d050", "family": "rf", "depth": 2, "leaf": 48, "dose": 0.50},
    {"name": "rf_d3_l64_d025", "family": "rf", "depth": 3, "leaf": 64, "dose": 0.25},
]


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


def _normalise_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    weights = np.clip(weights, 0.0, 1.0)
    denominator = weights.sum(axis=1, keepdims=True)
    fallback = denominator[:, 0] <= 1e-12
    weights = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 1e-12)
    weights[fallback, 0] = 1.0
    return weights


def _simplex_from_quadratic(indices: np.ndarray, terms: dict[str, np.ndarray]) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    a = float(np.sum(terms["a"][indices]))
    b1 = float(np.sum(terms["b1"][indices]))
    b2 = float(np.sum(terms["b2"][indices]))
    c11 = float(np.sum(terms["c11"][indices]))
    c12 = float(np.sum(terms["c12"][indices]))
    c22 = float(np.sum(terms["c22"][indices]))

    def objective(weights):
        wt, ws = float(weights[1]), float(weights[2])
        return a + 2.0 * wt * b1 + 2.0 * ws * b2 + wt * wt * c11 + 2.0 * wt * ws * c12 + ws * ws * c22

    candidates = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    gram = np.array([[c11, c12], [c12, c22]], np.float64)
    rhs = -np.array([b1, b2], np.float64)
    if np.linalg.cond(gram) < 1e12:
        inner = np.linalg.solve(gram, rhs)
        if inner[0] >= 0.0 and inner[1] >= 0.0 and inner.sum() <= 1.0:
            candidates.append(np.array([1.0 - inner.sum(), inner[0], inner[1]]))
    wt = float(np.clip(-b1 / max(c11, 1e-12), 0.0, 1.0))
    candidates.append(np.array([1.0 - wt, wt, 0.0]))
    ws = float(np.clip(-b2 / max(c22, 1e-12), 0.0, 1.0))
    candidates.append(np.array([1.0 - ws, 0.0, ws]))
    denominator = c11 + c22 - 2.0 * c12
    cross = b1 - b2 + c12 - c22
    wt_edge = float(np.clip(-cross / max(denominator, 1e-12), 0.0, 1.0))
    candidates.append(np.array([0.0, wt_edge, 1.0 - wt_edge]))
    return min(candidates, key=objective)


def _well_sse(weights: np.ndarray, terms: dict[str, np.ndarray]) -> np.ndarray:
    weights = _normalise_weights(weights)
    wt = weights[:, 1]
    ws = weights[:, 2]
    return (
        terms["a"]
        + 2.0 * wt * terms["b1"]
        + 2.0 * ws * terms["b2"]
        + wt * wt * terms["c11"]
        + 2.0 * wt * ws * terms["c12"]
        + ws * ws * terms["c22"]
    )


def _fit_model(config: dict, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, seed: int):
    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x)
    if config["family"] == "ridge":
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_imputed)
        model = Ridge(alpha=float(config["alpha"]))
        model.fit(x_scaled, y, sample_weight=sample_weight)
        return ("ridge", imputer, scaler, model)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=int(config["depth"]),
        min_samples_leaf=int(config["leaf"]),
        max_features=0.70,
        n_jobs=-1,
        random_state=int(seed),
    )
    model.fit(x_imputed, y, sample_weight=sample_weight)
    return ("rf", imputer, None, model)


def _predict_model(bundle, x: np.ndarray) -> np.ndarray:
    family, imputer, scaler, model = bundle
    x = imputer.transform(x)
    if family == "ridge":
        x = scaler.transform(x)
    return _normalise_weights(model.predict(x))


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 20 or np.std(left[valid]) <= 1e-12:
        return float("nan")
    return float(spearmanr(left[valid], right[valid]).statistic)


rows = pd.read_parquet(ROW_PATH)
rows["id"] = rows["id"].astype(str).str.lower()
rows["well"] = rows["well"].astype(str).str.lower()
if len(rows) < 3_760_000 or rows["well"].nunique() != 773 or rows["id"].duplicated().any():
    raise RuntimeError("Invalid V4 row coverage")

v4_verdict = json.loads(V4_VERDICT_PATH.read_text(encoding="utf-8"))
base_lineage = str(v4_verdict["best_proxy"]["base"])

# The OOF is already ordered by complete well; factorization is used rather
# than relying on contiguous order for all exact quadratic statistics.
codes, well_names = pd.factorize(rows["well"], sort=False)
groups = len(well_names)
counts = np.bincount(codes, minlength=groups).astype(np.float64)
first_position = np.full(groups, len(rows), dtype=np.int64)
last_position = np.zeros(groups, dtype=np.int64)
np.minimum.at(first_position, codes, np.arange(len(rows), dtype=np.int64))
np.maximum.at(last_position, codes, np.arange(len(rows), dtype=np.int64))
fold_by_well = rows["fold"].to_numpy(np.int8)[first_position]
distance = rows["md_since"].to_numpy(np.float64)
truth = rows["truth"].to_numpy(np.float64)
base = rows["strong_proxy"].to_numpy(np.float64)
track8 = rows["track8"].to_numpy(np.float64)
stride = rows["stride"].to_numpy(np.float64)
base_error = base - truth
d1 = track8 - base
d2 = stride - base
terms = {
    "a": np.bincount(codes, weights=base_error * base_error, minlength=groups),
    "b1": np.bincount(codes, weights=base_error * d1, minlength=groups),
    "b2": np.bincount(codes, weights=base_error * d2, minlength=groups),
    "c11": np.bincount(codes, weights=d1 * d1, minlength=groups),
    "c12": np.bincount(codes, weights=d1 * d2, minlength=groups),
    "c22": np.bincount(codes, weights=d2 * d2, minlength=groups),
}

oracle_weights = np.zeros((groups, 3), np.float64)
for code in range(groups):
    oracle_weights[code] = _simplex_from_quadratic(np.array([code]), terms)
oracle_sse = _well_sse(oracle_weights, terms)
base_sse = terms["a"]
oracle_gain = np.sqrt(base_sse / counts) - np.sqrt(np.maximum(oracle_sse, 0.0) / counts)


def _group_mean(values: np.ndarray) -> np.ndarray:
    return np.bincount(codes, weights=np.asarray(values, np.float64), minlength=groups) / counts


def _group_rms(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    return np.sqrt(np.bincount(codes, weights=values * values, minlength=groups) / counts)


feature_frame = pd.DataFrame(
    {
        "well": well_names,
        "fold": fold_by_well,
        "rows": counts,
        "log_rows": np.log1p(counts),
        "max_md_since": np.zeros(groups, np.float64),
    }
)
max_md_since = np.zeros(groups, np.float64)
np.maximum.at(max_md_since, codes, distance)
feature_frame["max_md_since"] = max_md_since

endpoint_arrays = {"base": base, "track8": track8, "stride": stride}
for endpoint_name, values in endpoint_arrays.items():
    feature_frame[f"{endpoint_name}_span"] = values[last_position] - values[first_position]
    delta = np.zeros(len(values), np.float64)
    same_well = codes[1:] == codes[:-1]
    delta_positions = np.flatnonzero(same_well) + 1
    delta[delta_positions] = np.diff(values)[same_well]
    feature_frame[f"{endpoint_name}_d1_rms"] = _group_rms(delta)

for pair_name, difference in (
    ("track8_minus_base", d1),
    ("stride_minus_base", d2),
    ("track8_minus_stride", track8 - stride),
):
    absolute = np.abs(difference)
    mean = _group_mean(difference)
    feature_frame[f"{pair_name}_mean"] = mean
    feature_frame[f"{pair_name}_rms"] = _group_rms(difference)
    feature_frame[f"{pair_name}_std"] = np.sqrt(
        np.maximum(_group_mean(difference * difference) - mean * mean, 0.0)
    )
    feature_frame[f"{pair_name}_mean_abs"] = _group_mean(absolute)
    feature_frame[f"{pair_name}_endpoint"] = difference[last_position]
    for lower, upper, label in (
        (0.0, 500.0, "near"),
        (500.0, 1000.0, "mid1"),
        (1000.0, 2000.0, "mid2"),
        (2000.0, np.inf, "far"),
    ):
        selected = (distance > lower) & (distance <= upper)
        selected_counts = np.bincount(codes[selected], minlength=groups).astype(np.float64)
        sums = np.bincount(codes[selected], weights=absolute[selected], minlength=groups)
        squares = np.bincount(codes[selected], weights=difference[selected] ** 2, minlength=groups)
        feature_frame[f"{pair_name}_{label}_mean_abs"] = np.divide(
            sums, selected_counts, out=np.zeros(groups), where=selected_counts > 0
        )
        feature_frame[f"{pair_name}_{label}_rms"] = np.sqrt(
            np.divide(squares, selected_counts, out=np.zeros(groups), where=selected_counts > 0)
        )

# SP45 internals are external evidence relative to the three endpoint paths.
sp45_columns = ["id", "well", *SP45_DIRECT, *SP45_PATHS]
sp45 = pd.read_parquet(SP45_PATH, columns=sp45_columns)
sp45["id"] = sp45["id"].astype(str).str.lower()
sp45["well"] = sp45["well"].astype(str).str.lower()
if len(sp45) != len(rows) or sp45["id"].duplicated().any():
    raise RuntimeError("Invalid SP45 confidence feature coverage")
if np.array_equal(sp45["id"].to_numpy(), rows["id"].to_numpy()):
    sp45_distance = distance
else:
    distance_by_id = rows.set_index("id", verify_integrity=True)["md_since"]
    aligned_distance = distance_by_id.reindex(sp45["id"].to_numpy()).to_numpy(np.float64)
    if not np.isfinite(aligned_distance).all():
        raise RuntimeError("SP45/V4 row alignment failed")
    sp45_distance = aligned_distance

sp45_codes = pd.Categorical(sp45["well"], categories=well_names).codes
if (sp45_codes < 0).any():
    raise RuntimeError("SP45 contains unknown wells")
sp45_counts = np.bincount(sp45_codes, minlength=groups).astype(np.float64)
if not np.array_equal(sp45_counts, counts):
    raise RuntimeError("SP45/V4 per-well row counts differ")

for column in SP45_DIRECT:
    values = pd.to_numeric(sp45[column], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(values)
    if not finite.all():
        median = float(np.nanmedian(values))
        values[~finite] = median
    mean = np.bincount(sp45_codes, weights=values, minlength=groups) / counts
    square_mean = np.bincount(sp45_codes, weights=values * values, minlength=groups) / counts
    feature_frame[f"sp45_{column}_mean"] = mean
    feature_frame[f"sp45_{column}_std"] = np.sqrt(np.maximum(square_mean - mean * mean, 0.0))
    far = sp45_distance > 2000.0
    far_counts = np.bincount(sp45_codes[far], minlength=groups).astype(np.float64)
    far_sum = np.bincount(sp45_codes[far], weights=values[far], minlength=groups)
    feature_frame[f"sp45_{column}_far_mean"] = np.divide(
        far_sum, far_counts, out=mean.copy(), where=far_counts > 0
    )

for name, left, right in (
    ("selector_pf", "sp45_selector_projected", "pf_mean"),
    ("selector_beam", "sp45_selector_projected", "beam_mean"),
    ("pf_beam", "pf_mean", "beam_mean"),
):
    difference = sp45[left].to_numpy(np.float64) - sp45[right].to_numpy(np.float64)
    feature_frame[f"sp45_{name}_mean"] = np.bincount(
        sp45_codes, weights=difference, minlength=groups
    ) / counts
    feature_frame[f"sp45_{name}_rms"] = np.sqrt(
        np.bincount(sp45_codes, weights=difference * difference, minlength=groups) / counts
    )

neighbor = pd.read_csv(NEIGHBOR_META_PATH)
neighbor["well"] = neighbor["well"].astype(str).str.lower()
neighbor = neighbor.drop_duplicates("well").set_index("well")
for column in (
    "min_distance",
    "median_distance",
    "p90_distance",
    "anchor_distance_full",
    "abs_direction_dot",
    "direction_dot",
):
    if column in neighbor:
        feature_frame[f"neighbor_{column}"] = feature_frame["well"].map(neighbor[column])

target_frame = feature_frame[["well", "fold"]].copy()
target_frame["oracle_gain_rmse"] = oracle_gain
target_frame["oracle_weight_base"] = oracle_weights[:, 0]
target_frame["oracle_weight_track8"] = oracle_weights[:, 1]
target_frame["oracle_weight_stride"] = oracle_weights[:, 2]
target_frame.to_csv(OUT_DIR / "perwell_oracle_targets.csv", index=False)

# Everything below operates on 773 well-level rows and the exact quadratic
# sufficient statistics. Release the multi-million-row frames before fitting.
del rows, sp45, truth, base, track8, stride, base_error, d1, d2, distance, codes
for _large_name in (
    "endpoint_arrays",
    "difference",
    "absolute",
    "values",
    "sp45_distance",
    "sp45_codes",
    "aligned_distance",
    "distance_by_id",
):
    globals().pop(_large_name, None)
gc.collect()

feature_names = [column for column in feature_frame.columns if column not in {"well", "fold"}]
x_frame = feature_frame[feature_names].replace([np.inf, -np.inf], np.nan)
usable = [
    column
    for column in feature_names
    if x_frame[column].notna().mean() >= 0.80 and float(x_frame[column].std(skipna=True)) > 1e-10
]
x = x_frame[usable].to_numpy(np.float64)
y = oracle_weights

association_rows = []
for index, feature in enumerate(usable):
    association_rows.append(
        {
            "feature": feature,
            "spearman_oracle_gain": _safe_spearman(x[:, index], oracle_gain),
            "spearman_weight_track8": _safe_spearman(x[:, index], y[:, 1]),
            "spearman_weight_stride": _safe_spearman(x[:, index], y[:, 2]),
        }
    )
associations = pd.DataFrame(association_rows)
associations["max_abs_association"] = associations[
    ["spearman_oracle_gain", "spearman_weight_track8", "spearman_weight_stride"]
].abs().max(axis=1)
associations = associations.sort_values("max_abs_association", ascending=False, ignore_index=True)
associations.to_csv(OUT_DIR / "feature_associations.csv", index=False)

# Full nested selection: each outer fold chooses its model from inner folds
# contained entirely in the other four outer folds.
outer_weights = np.zeros((groups, 3), np.float64)
constant_weights = np.zeros((groups, 3), np.float64)
shuffle_weights = np.zeros((groups, 3), np.float64)
outer_rows = []
inner_rows = []
rng = np.random.default_rng(20260722)

for held_fold in range(1, 6):
    outer_train = np.flatnonzero(fold_by_well != held_fold)
    outer_valid = np.flatnonzero(fold_by_well == held_fold)
    outer_constant = _simplex_from_quadratic(outer_train, terms)
    constant_weights[outer_valid] = outer_constant

    scans = []
    for config in MODEL_CONFIGS:
        inner_prediction = np.zeros((groups, 3), np.float64)
        inner_constant_prediction = np.zeros((groups, 3), np.float64)
        inner_valid_all = []
        for inner_fold in sorted(np.unique(fold_by_well[outer_train])):
            inner_train = outer_train[fold_by_well[outer_train] != inner_fold]
            inner_valid = outer_train[fold_by_well[outer_train] == inner_fold]
            inner_constant = _simplex_from_quadratic(inner_train, terms)
            bundle = _fit_model(
                config,
                x[inner_train],
                y[inner_train],
                counts[inner_train],
                20260722 + held_fold * 100 + int(inner_fold),
            )
            predicted = _predict_model(bundle, x[inner_valid])
            dose = float(config["dose"])
            predicted = _normalise_weights(
                (1.0 - dose) * inner_constant[None, :] + dose * predicted
            )
            inner_prediction[inner_valid] = predicted
            inner_constant_prediction[inner_valid] = inner_constant
            inner_valid_all.extend(inner_valid.tolist())
        inner_valid_all = np.asarray(inner_valid_all, dtype=np.int64)
        adaptive_sse = float(np.sum(_well_sse(inner_prediction, terms)[inner_valid_all]))
        constant_sse = float(
            np.sum(_well_sse(inner_constant_prediction, terms)[inner_valid_all])
        )
        row_count = float(np.sum(counts[inner_valid_all]))
        scans.append(
            {
                "held_fold": held_fold,
                "config": config["name"],
                "inner_rmse": float(np.sqrt(adaptive_sse / row_count)),
                "inner_constant_rmse": float(np.sqrt(constant_sse / row_count)),
            }
        )
    scan = pd.DataFrame(scans).sort_values(["inner_rmse", "config"], ignore_index=True)
    inner_rows.extend(scan.to_dict(orient="records"))
    selected_name = str(scan.iloc[0]["config"])
    selected = next(config for config in MODEL_CONFIGS if config["name"] == selected_name)
    use_adaptive = float(scan.iloc[0]["inner_rmse"]) < float(scan.iloc[0]["inner_constant_rmse"])

    if use_adaptive:
        bundle = _fit_model(
            selected,
            x[outer_train],
            y[outer_train],
            counts[outer_train],
            20260722 + held_fold,
        )
        predicted = _predict_model(bundle, x[outer_valid])
        dose = float(selected["dose"])
        predicted = _normalise_weights(
            (1.0 - dose) * outer_constant[None, :] + dose * predicted
        )
    else:
        predicted = np.tile(outer_constant, (len(outer_valid), 1))
    outer_weights[outer_valid] = predicted
    shuffle_weights[outer_valid] = predicted[rng.permutation(len(predicted))]

    base_fold_sse = float(np.sum(base_sse[outer_valid]))
    constant_fold_sse = float(np.sum(_well_sse(constant_weights, terms)[outer_valid]))
    adaptive_fold_sse = float(np.sum(_well_sse(outer_weights, terms)[outer_valid]))
    fold_rows_count = float(np.sum(counts[outer_valid]))
    outer_rows.append(
        {
            "fold": held_fold,
            "wells": len(outer_valid),
            "selected_config": selected_name,
            "use_adaptive": bool(use_adaptive),
            "inner_rmse": float(scan.iloc[0]["inner_rmse"]),
            "inner_constant_rmse": float(scan.iloc[0]["inner_constant_rmse"]),
            "base_rmse": float(np.sqrt(base_fold_sse / fold_rows_count)),
            "constant_rmse": float(np.sqrt(constant_fold_sse / fold_rows_count)),
            "adaptive_rmse": float(np.sqrt(adaptive_fold_sse / fold_rows_count)),
            "adaptive_gain_vs_base": float(
                np.sqrt(base_fold_sse / fold_rows_count) - np.sqrt(adaptive_fold_sse / fold_rows_count)
            ),
            "adaptive_gain_vs_constant": float(
                np.sqrt(constant_fold_sse / fold_rows_count) - np.sqrt(adaptive_fold_sse / fold_rows_count)
            ),
        }
    )

base_weights = np.zeros((groups, 3), np.float64)
base_weights[:, 0] = 1.0
base_well_sse = _well_sse(base_weights, terms)
constant_well_sse = _well_sse(constant_weights, terms)
adaptive_well_sse = _well_sse(outer_weights, terms)
shuffle_well_sse = _well_sse(shuffle_weights, terms)
total_rows = float(np.sum(counts))


def _summary(name: str, well_sse: np.ndarray) -> dict:
    well_rmse = np.sqrt(np.maximum(well_sse, 0.0) / counts)
    return {
        "variant": name,
        "exact_sse": float(np.sum(well_sse)),
        "rmse": float(np.sqrt(np.sum(well_sse) / total_rows)),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }


summary = pd.DataFrame(
    [
        _summary("base", base_well_sse),
        _summary("nested_constant", constant_well_sse),
        _summary("nested_adaptive", adaptive_well_sse),
        _summary("matched_shuffle", shuffle_well_sse),
        _summary("perwell_oracle", oracle_sse),
    ]
)
metric = summary.set_index("variant")["rmse"]
adaptive_gain = float(metric["base"] - metric["nested_adaptive"])
increment = float(metric["nested_constant"] - metric["nested_adaptive"])
shuffle_loss = float(metric["matched_shuffle"] - metric["nested_adaptive"])
outer_report = pd.DataFrame(outer_rows)
checks = {
    "oracle_gain_ge_1p0": float(metric["base"] - metric["perwell_oracle"]) >= 1.0,
    "adaptive_gain_vs_base_ge_0p05": adaptive_gain >= 0.05,
    "increment_vs_constant_ge_0p03": increment >= 0.03,
    "minimum_fold_gain_vs_constant_ge_minus0p05": float(
        outer_report["adaptive_gain_vs_constant"].min()
    ) >= -0.05,
    "adaptive_selected_in_at_least_3_folds": int(outer_report["use_adaptive"].sum()) >= 3,
    "shuffle_worse_by_ge_0p02": shuffle_loss >= 0.02,
}

well_weights = feature_frame[["well", "fold"]].copy()
for prefix, weights in (
    ("constant", constant_weights),
    ("adaptive", outer_weights),
    ("shuffle", shuffle_weights),
    ("oracle", oracle_weights),
):
    well_weights[f"{prefix}_weight_base"] = weights[:, 0]
    well_weights[f"{prefix}_weight_track8"] = weights[:, 1]
    well_weights[f"{prefix}_weight_stride"] = weights[:, 2]
well_weights["oracle_gain_rmse"] = oracle_gain

summary.to_csv(OUT_DIR / "summary.csv", index=False)
outer_report.to_csv(OUT_DIR / "outer_fold_report.csv", index=False)
pd.DataFrame(inner_rows).to_csv(OUT_DIR / "inner_config_scan.csv", index=False)
well_weights.to_csv(OUT_DIR / "crossfit_well_weights.csv", index=False)
feature_frame.to_parquet(OUT_DIR / "legal_perwell_features.parquet", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_nested_three_endpoint_perwell_confidence_gate",
    "base_lineage": base_lineage,
    "sample_unit": "complete physical well",
    "features": {
        "count": len(usable),
        "sources": [
            "three endpoint disagreement and path shape",
            "standalone SP45 PF/beam/selector uncertainty",
            "frozen neighbour geometry diagnostics",
        ],
        "top_associations": associations.head(20).to_dict(orient="records"),
    },
    "models": MODEL_CONFIGS,
    "summary": summary.to_dict(orient="records"),
    "outer_folds": outer_rows,
    "adaptive_gain_vs_base": adaptive_gain,
    "increment_vs_constant": increment,
    "matched_shuffle_loss": shuffle_loss,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "next": (
        "PASS -> legal per-well confidence is learnable; refit and package deployment lineage"
        if all(checks.values())
        else "FAIL -> per-well oracle is not reachable from current legal confidence features; close this fusion path"
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV5 CONFIDENCE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV5 OUTER FOLDS")
print(outer_report.to_string(index=False), flush=True)
print("\nV5 TOP ASSOCIATIONS")
print(associations.head(30).to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
