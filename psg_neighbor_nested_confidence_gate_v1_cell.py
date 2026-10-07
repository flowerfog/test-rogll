"""Nested well-level confidence gate for the PSG-SP45 branch.

The gate is deliberately small because the zero-training audit found a high
per-well oracle but weak label-free associations. For every held outer fold,
the remaining four folds run an inner fold selection over:

* one global constant weight;
* a decreasing isotonic gate on base-vs-branch disagreement;
* strongly regularised Ridge gates.

All selection and reporting use exact row-pooled SSE. A matched shuffled-fit
control must lose the adaptive increment. Oracle targets never enter held-fold
features or predictions.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge


EXPERIMENT = "psg_neighbor_nested_confidence_gate_v1"
ROOT = Path("/root/autodl-tmp/oof_exports")
ROW_PATH = ROOT / "psg_sp45_branch_survival_audit_v0" / "neighbor_branch_crossfit_oof.parquet"
FEATURE_PATH = ROOT / "psg_neighbor_perwell_oracle_audit_v0" / "per_well_oracle_and_features.csv"
OUT_DIR = ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260722
MIN_TOTAL_GAIN_FT = 0.10
MIN_INCREMENT_VS_CONSTANT_FT = 0.03
MIN_FOLD_GAIN_FT = -0.05
MIN_SHUFFLE_LOSS_FT = 0.02

EXCLUDED_FEATURES = {
    "well",
    "fold",
    "base_rmse",
    "candidate_rmse",
    "candidate_benefit_mse",
    "candidate_wins",
    "oracle_weight",
}


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


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _optimal_weight(b, c, indices):
    denominator = float(np.sum(c[indices]))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(-np.sum(b[indices]) / denominator, 0.0, 1.0))


def _sse(a, b, c, indices, weight):
    weight = np.asarray(weight, np.float64)
    return float(np.sum(a[indices] + 2.0 * b[indices] * weight + c[indices] * weight**2))


def _rmse(sse, counts, indices):
    return float(np.sqrt(max(float(sse), 0.0) / np.sum(counts[indices])))


def _feature_matrix(frame, columns):
    return frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)


def _prepare_linear(x_train, x_test):
    median = np.nanmedian(x_train, axis=0)
    median[~np.isfinite(median)] = 0.0
    x_train = np.where(np.isfinite(x_train), x_train, median)
    x_test = np.where(np.isfinite(x_test), x_test, median)
    centre = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale[scale < 1e-8] = 1.0
    return (x_train - centre) / scale, (x_test - centre) / scale


def _config_name(config):
    if config["family"] == "constant":
        return "constant"
    if config["family"] == "isotonic":
        return f"isotonic_b{config['bins']}_d{config['dose']:.2f}"
    return f"ridge_a{config['alpha']:g}_d{config['dose']:.2f}"


def _fit_predict(config, train_idx, test_idx, features, a, b, c, shuffle_seed=None):
    global_weight = _optimal_weight(b, c, train_idx)
    if config["family"] == "constant":
        return np.full(len(test_idx), global_weight, np.float64)

    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    target = np.zeros(len(train_idx), np.float64)
    valid = c[train_idx] > 1e-12
    target[valid] = np.clip(-b[train_idx][valid] / c[train_idx][valid], 0.0, 1.0)
    impact = np.maximum(c[train_idx], 1e-6)

    if config["family"] == "isotonic":
        column = feature_names.index("mean_abs_base_branch_disagreement")
        x_train = features[train_idx, column].copy()
        x_test = features[test_idx, column].copy()
        fill = float(np.nanmedian(x_train))
        x_train[~np.isfinite(x_train)] = fill
        x_test[~np.isfinite(x_test)] = fill
        fit_x = x_train.copy()
        if rng is not None:
            fit_x = fit_x[rng.permutation(len(fit_x))]

        quantiles = np.linspace(0.0, 1.0, int(config["bins"]) + 1)
        edges = np.unique(np.quantile(fit_x, quantiles))
        if len(edges) < 3:
            return np.full(len(test_idx), global_weight, np.float64)
        train_bin = np.clip(np.searchsorted(edges[1:-1], fit_x, side="right"), 0, len(edges) - 2)
        centres = []
        raw_weights = []
        bin_impact = []
        for group in range(len(edges) - 1):
            selected = train_bin == group
            if not int(selected.sum()):
                continue
            denominator = float(np.sum(impact[selected]))
            centres.append(float(np.average(fit_x[selected], weights=impact[selected])))
            raw_weights.append(float(np.average(target[selected], weights=impact[selected])))
            bin_impact.append(denominator)
        if len(centres) < 2:
            prediction = np.full(len(test_idx), global_weight, np.float64)
        else:
            order = np.argsort(centres)
            model = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(
                np.asarray(centres)[order],
                np.asarray(raw_weights)[order],
                sample_weight=np.asarray(bin_impact)[order],
            )
            prediction = model.predict(x_test)
    else:
        x_train = features[train_idx].copy()
        x_test = features[test_idx].copy()
        if rng is not None:
            x_train = x_train[rng.permutation(len(x_train))]
        x_train, x_test = _prepare_linear(x_train, x_test)
        model = Ridge(alpha=float(config["alpha"]))
        model.fit(x_train, target, sample_weight=impact / np.median(impact))
        prediction = np.clip(model.predict(x_test), 0.0, 1.0)

    dose = float(config["dose"])
    return np.clip(global_weight + dose * (prediction - global_weight), 0.0, 1.0)


started = time.time()
for path in (ROW_PATH, FEATURE_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

rows = pd.read_parquet(ROW_PATH)
required_rows = {
    "well",
    "fold",
    "truth",
    "cp12_neighbor_full_progress_geom",
    "psg_sp45_branch",
}
if not required_rows.issubset(rows.columns):
    raise RuntimeError(f"Row artifact missing {sorted(required_rows - set(rows.columns))}")
rows["well"] = rows["well"].astype(str).str.lower()

codes, well_names = pd.factorize(rows["well"], sort=False)
groups = len(well_names)
counts = np.bincount(codes).astype(np.float64)
base = rows["cp12_neighbor_full_progress_geom"].to_numpy(np.float64)
candidate = rows["psg_sp45_branch"].to_numpy(np.float64)
truth = rows["truth"].to_numpy(np.float64)
base_error = base - truth
direction = candidate - base
a = np.bincount(codes, weights=base_error**2, minlength=groups)
b = np.bincount(codes, weights=base_error * direction, minlength=groups)
c = np.bincount(codes, weights=direction**2, minlength=groups)
first_position = np.full(groups, len(rows), np.int64)
np.minimum.at(first_position, codes, np.arange(len(rows), dtype=np.int64))
folds = rows["fold"].to_numpy(np.int8)[first_position]

feature_frame = pd.read_csv(FEATURE_PATH)
feature_frame["well"] = feature_frame["well"].astype(str).str.lower()
if feature_frame["well"].duplicated().any():
    raise RuntimeError("Feature file has duplicate wells")
missing_feature_wells = sorted(set(well_names).difference(feature_frame["well"]))
if missing_feature_wells:
    raise RuntimeError(f"Feature alignment missed wells: {missing_feature_wells[:20]}")
feature_frame = (
    feature_frame.set_index("well")
    .reindex(pd.Index(well_names, name="well"))
    .reset_index()
)
feature_names = [
    column
    for column in feature_frame.columns
    if column not in EXCLUDED_FEATURES
    and pd.to_numeric(feature_frame[column], errors="coerce").notna().mean() >= 0.80
]
if "mean_abs_base_branch_disagreement" not in feature_names:
    raise RuntimeError(f"Primary disagreement feature missing: {feature_names}")
features = _feature_matrix(feature_frame, feature_names)

configs = [{"family": "constant"}]
for bins in (3, 4, 5):
    for dose in (0.25, 0.50, 0.75, 1.00):
        configs.append({"family": "isotonic", "bins": bins, "dose": dose})
for alpha in (1.0, 10.0, 100.0):
    for dose in (0.25, 0.50, 0.75, 1.00):
        configs.append({"family": "ridge", "alpha": alpha, "dose": dose})

adaptive_weight = np.full(groups, np.nan, np.float64)
constant_weight = np.full(groups, np.nan, np.float64)
shuffle_weight = np.full(groups, np.nan, np.float64)
selection_rows = []
outer_rows = []

for outer_fold in sorted(np.unique(folds)):
    outer_train = folds != outer_fold
    outer_valid = folds == outer_fold
    train_indices = np.flatnonzero(outer_train)
    valid_indices = np.flatnonzero(outer_valid)
    inner_folds = sorted(np.unique(folds[outer_train]))

    config_scores = []
    for config in configs:
        total_sse = 0.0
        total_rows = 0.0
        for inner_fold in inner_folds:
            inner_train = np.flatnonzero(outer_train & (folds != inner_fold))
            inner_valid = np.flatnonzero(outer_train & (folds == inner_fold))
            weight = _fit_predict(config, inner_train, inner_valid, features, a, b, c)
            total_sse += _sse(a, b, c, inner_valid, weight)
            total_rows += float(np.sum(counts[inner_valid]))
        config_scores.append(
            {
                "outer_fold": int(outer_fold),
                "config": _config_name(config),
                "family": config["family"],
                "inner_rmse": float(np.sqrt(total_sse / total_rows)),
            }
        )
    score_frame = pd.DataFrame(config_scores).sort_values(
        ["inner_rmse", "config"], ignore_index=True
    )
    selected_name = str(score_frame.iloc[0]["config"])
    selected_config = next(config for config in configs if _config_name(config) == selected_name)
    selection_rows.extend(config_scores)

    adaptive = _fit_predict(selected_config, train_indices, valid_indices, features, a, b, c)
    constant = _fit_predict({"family": "constant"}, train_indices, valid_indices, features, a, b, c)
    shuffled = _fit_predict(
        selected_config,
        train_indices,
        valid_indices,
        features,
        a,
        b,
        c,
        shuffle_seed=SEED + int(outer_fold),
    )
    adaptive_weight[valid_indices] = adaptive
    constant_weight[valid_indices] = constant
    shuffle_weight[valid_indices] = shuffled

    base_sse = float(np.sum(a[valid_indices]))
    adaptive_sse = _sse(a, b, c, valid_indices, adaptive)
    constant_sse = _sse(a, b, c, valid_indices, constant)
    shuffle_sse = _sse(a, b, c, valid_indices, shuffled)
    base_rmse = _rmse(base_sse, counts, valid_indices)
    adaptive_rmse = _rmse(adaptive_sse, counts, valid_indices)
    constant_rmse = _rmse(constant_sse, counts, valid_indices)
    shuffle_rmse = _rmse(shuffle_sse, counts, valid_indices)
    outer_rows.append(
        {
            "outer_fold": int(outer_fold),
            "selected_config": selected_name,
            "selected_family": selected_config["family"],
            "held_wells": int(len(valid_indices)),
            "base_rmse": base_rmse,
            "constant_rmse": constant_rmse,
            "adaptive_rmse": adaptive_rmse,
            "shuffle_rmse": shuffle_rmse,
            "constant_gain": base_rmse - constant_rmse,
            "adaptive_gain": base_rmse - adaptive_rmse,
            "increment_vs_constant": constant_rmse - adaptive_rmse,
            "shuffle_increment_vs_constant": constant_rmse - shuffle_rmse,
            "mean_adaptive_weight": float(np.mean(adaptive)),
            "mean_constant_weight": float(np.mean(constant)),
        }
    )

for name, values in (
    ("adaptive", adaptive_weight),
    ("constant", constant_weight),
    ("shuffle", shuffle_weight),
):
    if not np.isfinite(values).all():
        raise RuntimeError(f"Missing {name} gate weights")

base_sse = float(np.sum(a))
candidate_sse = float(np.sum(a + 2.0 * b + c))
constant_sse = _sse(a, b, c, np.arange(groups), constant_weight)
adaptive_sse = _sse(a, b, c, np.arange(groups), adaptive_weight)
shuffle_sse = _sse(a, b, c, np.arange(groups), shuffle_weight)
total_rows = float(np.sum(counts))
base_rmse = float(np.sqrt(base_sse / total_rows))
candidate_rmse = float(np.sqrt(candidate_sse / total_rows))
constant_rmse = float(np.sqrt(constant_sse / total_rows))
adaptive_rmse = float(np.sqrt(adaptive_sse / total_rows))
shuffle_rmse = float(np.sqrt(shuffle_sse / total_rows))

outer_report = pd.DataFrame(outer_rows)
selection_report = pd.DataFrame(selection_rows)
summary = pd.DataFrame(
    [
        {"variant": "base", "rmse": base_rmse, "gain": 0.0},
        {"variant": "candidate", "rmse": candidate_rmse, "gain": base_rmse - candidate_rmse},
        {"variant": "nested_constant", "rmse": constant_rmse, "gain": base_rmse - constant_rmse},
        {"variant": "nested_adaptive", "rmse": adaptive_rmse, "gain": base_rmse - adaptive_rmse},
        {"variant": "matched_shuffle", "rmse": shuffle_rmse, "gain": base_rmse - shuffle_rmse},
    ]
)

adaptive_gain = base_rmse - adaptive_rmse
increment = constant_rmse - adaptive_rmse
shuffle_loss = shuffle_rmse - adaptive_rmse
checks = {
    "adaptive_gain_ge_0p10": adaptive_gain >= MIN_TOTAL_GAIN_FT,
    "increment_vs_constant_ge_0p03": increment >= MIN_INCREMENT_VS_CONSTANT_FT,
    "minimum_fold_gain_ge_minus0p05": float(outer_report["adaptive_gain"].min()) >= MIN_FOLD_GAIN_FT,
    "shuffle_worse_by_ge_0p02": shuffle_loss >= MIN_SHUFFLE_LOSS_FT,
    "adaptive_selected_in_at_least_3_folds": int((outer_report["selected_family"] != "constant").sum()) >= 3,
}
strict_pass = bool(all(checks.values()))

well_report = feature_frame[["well", "fold"]].copy()
well_report["rows"] = counts.astype(int)
well_report["base_sse"] = a
well_report["direction_cross"] = b
well_report["direction_sse"] = c
well_report["constant_weight"] = constant_weight
well_report["adaptive_weight"] = adaptive_weight
well_report["shuffle_weight"] = shuffle_weight
well_report.to_csv(OUT_DIR / "crossfit_well_weights.csv", index=False)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
outer_report.to_csv(OUT_DIR / "outer_fold_report.csv", index=False)
selection_report.to_csv(OUT_DIR / "inner_config_scan.csv", index=False)

row_adaptive_weight = adaptive_weight[codes]
export = rows[["id", "well", "row_idx", "fold", "truth", "cp12_neighbor_full_progress_geom", "psg_sp45_branch"]].copy()
export["adaptive_weight"] = row_adaptive_weight.astype(np.float32)
export["adaptive_prediction"] = (
    base + row_adaptive_weight * (candidate - base)
).astype(np.float32)
export.to_parquet(OUT_DIR / "nested_adaptive_oof.parquet", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_nested_well_confidence_gate",
    "sample_unit": "complete physical well",
    "features": feature_names,
    "configurations": len(configs),
    "base_rmse": base_rmse,
    "candidate_rmse": candidate_rmse,
    "nested_constant_rmse": constant_rmse,
    "nested_adaptive_rmse": adaptive_rmse,
    "matched_shuffle_rmse": shuffle_rmse,
    "adaptive_gain": adaptive_gain,
    "increment_vs_constant": increment,
    "shuffle_loss": shuffle_loss,
    "selected_configs": outer_report["selected_config"].tolist(),
    "fold_adaptive_gains": outer_report["adaptive_gain"].tolist(),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> confidence gate is real OOF signal; rebuild against exact deployment lineage before LB"
        if strict_pass
        else "FAIL -> oracle is not reachable from legal confidence features; close PSG deployment branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nPSG CONFIDENCE GATE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nOUTER FOLDS")
print(outer_report.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
