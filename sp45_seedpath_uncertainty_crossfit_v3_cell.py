"""Cross-fit PF-posterior trust gate on the 72 cached Fold3 wells.

One row is one complete well.  Features are label-free summaries of the
128-seed posterior and disagreement with fold-clean SP45.  The target is the
per-well oracle weight on the frozen Cauchy+persistence endpoint.  Six outer
folds compare dynamic weights with a train-only fixed weight and a matched
shuffled-target control.  No additional PF runs are performed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "sp45_seedpath_uncertainty_crossfit_v3"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V0_CANDIDATES = (
    ROOT / "sp45_seedpath_stride_rerank_gateway_v0_cell.py",
    Path.cwd() / "sp45_seedpath_stride_rerank_gateway_v0_cell.py",
)
V0_PATH = next((path for path in V0_CANDIDATES if path.exists()), None)
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
if V0_PATH is None:
    raise FileNotFoundError("Upload sp45_seedpath_stride_rerank_gateway_v0_cell.py")
if POOL_PATH is None:
    raise FileNotFoundError("Missing candidate_pool_full_merged_oof_honest_fle.csv")

WELL_COUNT = 72
OUTER_FOLDS = 6
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
SEED = 20260723


def _load_v0(path):
    source = path.read_text(encoding="utf-8")
    marker = "reports = Parallel(n_jobs=N_JOBS, backend=\"threading\")"
    if marker not in source:
        raise RuntimeError("Uploaded v0 source lacks the threading repair")
    prefix = source.split(marker, 1)[0]
    name = "sp45_seedpath_uncertainty_v3_bootstrap"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(prefix, str(path), "exec"), module.__dict__)
    tree = ast.parse(source, filename=str(path))
    keep = {"_robust_z", "_soft_path", "_candidate_costs"}
    body = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in keep
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
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


def _stable_fold(well):
    digest = hashlib.sha256(f"seedpath-trust-v3:{well}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "little") % OUTER_FOLDS


def _weights_from_cost(cost, temperature=0.5):
    logits = -np.asarray(cost, np.float64) / float(temperature)
    logits -= np.max(logits)
    weight = np.exp(np.clip(logits, -80.0, 0.0))
    return weight / np.sum(weight)


def _distribution_features(prefix, values):
    values = np.asarray(values, np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_max": float(np.max(values)),
    }


def _rmse(a, b, mask=None):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if mask is not None:
        a = a[mask]
        b = b[mask]
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else np.nan


def _metric_from_arrays(predictions, truths, wells):
    prediction = np.concatenate(predictions).astype(np.float64)
    truth = np.concatenate(truths).astype(np.float64)
    well = np.concatenate(wells)
    error = prediction - truth
    bywell = np.sqrt(
        pd.DataFrame({"well": well, "error2": error * error})
        .groupby("well", sort=False)["error2"]
        .mean()
    )
    return {
        "rows": int(len(error)),
        "wells": int(len(bywell)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(bywell.mean()),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
    }


B = _load_v0(V0_PATH)
wells = list(B.fold3_wells[:WELL_COUNT])
if len(wells) != WELL_COUNT:
    raise RuntimeError(f"Expected {WELL_COUNT} wells, got {len(wells)}")
for well in wells:
    if not B._bank_valid(B._bank_path(well)):
        raise FileNotFoundError(f"Missing cached bank: {B._bank_path(well)}")

pool = pd.read_csv(
    POOL_PATH,
    usecols=["id", "well", "tvt_true", "sp45"],
    dtype={"id": str, "well": str},
)
pool = pool.loc[pool["well"].isin(wells)].copy()
pool_by_well = {well: frame.set_index("id") for well, frame in pool.groupby("well", sort=False)}

feature_rows = []
payloads = {}
for number, well in enumerate(wells, start=1):
    with np.load(B._bank_path(well), allow_pickle=False) as data:
        members = data["members"].astype(np.float64)
        truth = data["truth"].astype(np.float64)
        md = data["md"].astype(np.float64)
        distance = md - float(data["last_md"][0])
        row_idx = data["row_idx"].astype(np.int64)
        ids = [f"{well}_{int(row)}" for row in row_idx]
        aligned = pool_by_well[well].reindex(ids)
        if aligned["sp45"].isna().any():
            raise RuntimeError(f"SP45 alignment failure for {well}")
        sp45 = aligned["sp45"].to_numpy(np.float64)
        truth_gap = np.abs(truth - aligned["tvt_true"].to_numpy(np.float64))
        if float(np.max(truth_gap)) > 0.01:
            raise RuntimeError(f"Truth mismatch for {well}: {float(np.max(truth_gap))}")

        cauchy, persistence = B._candidate_costs(data)
        cauchy_z = B._robust_z(cauchy)
        persistence_z = B._robust_z(persistence)
        joint = cauchy_z + 0.3 * persistence_z
        posterior_weight = _weights_from_cost(joint, 0.5)
        endpoint = np.sum(members * posterior_weight[:, None], axis=0)
        pf_mean = members.mean(axis=0)
        direction = endpoint - sp45
        denominator = float(np.dot(direction, direction))
        oracle_weight = (
            float(np.clip(-np.dot(sp45 - truth, direction) / denominator, 0.0, 1.0))
            if denominator > 1e-12 else 0.0
        )
        oracle = sp45 + oracle_weight * direction

        seed_std = np.std(members, axis=0)
        near = (distance >= 0.0) & (distance < 500.0)
        middle = (distance >= 1000.0) & (distance < 2000.0)
        far = distance >= 2000.0
        entropy = float(-np.sum(posterior_weight * np.log(posterior_weight + 1e-12)) / np.log(len(posterior_weight)))
        ess = float(1.0 / np.sum(posterior_weight ** 2) / len(posterior_weight))
        order = np.sort(posterior_weight)[::-1]
        features = {
            "well": well,
            "outer_fold": _stable_fold(well),
            "rows": int(len(truth)),
            "length_ft": float(distance[-1] - distance[0]),
            "posterior_entropy": entropy,
            "posterior_ess_fraction": ess,
            "posterior_top1_mass": float(order[0]),
            "posterior_top2_mass": float(np.sum(order[:2])),
            "posterior_top8_mass": float(np.sum(order[:8])),
            "loglik_std": float(np.std(data["log_likelihood"].astype(np.float64))),
            "cauchy_std": float(np.std(cauchy_z)),
            "persistence_std": float(np.std(persistence_z)),
            "rankcorr_loglik_cauchy": float(
                pd.Series(data["log_likelihood"].astype(np.float64)).corr(
                    pd.Series(-cauchy), method="spearman"
                )
            ),
            "rankcorr_cauchy_persistence": float(
                pd.Series(cauchy).corr(pd.Series(persistence), method="spearman")
            ),
            "endpoint_sp45_rmse": _rmse(endpoint, sp45),
            "endpoint_sp45_bias": float(np.mean(endpoint - sp45)),
            "pfmean_sp45_rmse": _rmse(pf_mean, sp45),
            "endpoint_pfmean_rmse": _rmse(endpoint, pf_mean),
            "endpoint_sp45_near_rmse": _rmse(endpoint, sp45, near),
            "endpoint_sp45_mid_rmse": _rmse(endpoint, sp45, middle),
            "endpoint_sp45_far_rmse": _rmse(endpoint, sp45, far),
            "seed_std_near_mean": float(np.mean(seed_std[near])) if near.any() else 0.0,
            "seed_std_mid_mean": float(np.mean(seed_std[middle])) if middle.any() else 0.0,
            "seed_std_far_mean": float(np.mean(seed_std[far])) if far.any() else 0.0,
            "seed_std_growth_far_near": (
                float(np.mean(seed_std[far]) - np.mean(seed_std[near]))
                if far.any() and near.any() else 0.0
            ),
            "oracle_weight": oracle_weight,
            "benefit_mse": float(np.mean((sp45 - truth) ** 2) - np.mean((oracle - truth) ** 2)),
            "sp45_rmse_label": _rmse(sp45, truth),
            "endpoint_rmse_label": _rmse(endpoint, truth),
        }
        features.update(_distribution_features("seed_std", seed_std))
        feature_rows.append(features)
        payloads[well] = {
            "truth": truth.astype(np.float32),
            "sp45": sp45.astype(np.float32),
            "endpoint": endpoint.astype(np.float32),
        }
    if number % 12 == 0 or number == len(wells):
        print(f"uncertainty features {number}/{len(wells)}", flush=True)

feature_frame = pd.DataFrame(feature_rows)
label_columns = {
    "well", "outer_fold", "oracle_weight", "benefit_mse",
    "sp45_rmse_label", "endpoint_rmse_label",
}
feature_columns = [column for column in feature_frame.columns if column not in label_columns]
feature_frame[feature_columns] = (
    feature_frame[feature_columns]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(feature_frame[feature_columns].median())
    .fillna(0.0)
)

associations = []
for column in feature_columns:
    associations.append(
        {
            "feature": column,
            "spearman_oracle_weight": float(feature_frame[column].corr(feature_frame["oracle_weight"], method="spearman")),
            "spearman_benefit_mse": float(feature_frame[column].corr(feature_frame["benefit_mse"], method="spearman")),
        }
    )
association_frame = pd.DataFrame(associations).sort_values(
    "spearman_benefit_mse", key=lambda series: series.abs(), ascending=False
)
association_frame.to_csv(OUT_DIR / "feature_associations.csv", index=False)

oof_dynamic = {}
oof_fixed = {}
oof_shuffle = {}
fold_rows = []
coefficient_rows = []
rng = np.random.default_rng(SEED)

for heldout in range(OUTER_FOLDS):
    train = feature_frame.loc[feature_frame["outer_fold"].ne(heldout)].copy()
    valid = feature_frame.loc[feature_frame["outer_fold"].eq(heldout)].copy()
    if len(valid) < 5:
        raise RuntimeError(f"Fold {heldout} has only {len(valid)} wells")
    x_train = train[feature_columns].to_numpy(np.float64)
    x_valid = valid[feature_columns].to_numpy(np.float64)
    y_train = train["oracle_weight"].to_numpy(np.float64)

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    model.fit(x_train, y_train)
    dynamic_weight = np.clip(model.predict(x_valid), 0.0, 1.0)

    shuffled_model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    shuffled_model.fit(x_train, rng.permutation(y_train))
    shuffled_weight = np.clip(shuffled_model.predict(x_valid), 0.0, 1.0)

    train_truth = np.concatenate([payloads[well]["truth"] for well in train["well"]]).astype(np.float64)
    train_sp45 = np.concatenate([payloads[well]["sp45"] for well in train["well"]]).astype(np.float64)
    train_endpoint = np.concatenate([payloads[well]["endpoint"] for well in train["well"]]).astype(np.float64)
    train_direction = train_endpoint - train_sp45
    denominator = float(np.dot(train_direction, train_direction))
    fixed_weight = float(np.clip(-np.dot(train_sp45 - train_truth, train_direction) / denominator, 0.0, 1.0))

    fold_truths = []
    fold_sp45 = []
    fold_dynamic = []
    fold_fixed = []
    fold_shuffle = []
    fold_wells = []
    for index, well in enumerate(valid["well"]):
        payload = payloads[well]
        direction = payload["endpoint"] - payload["sp45"]
        fold_truths.append(payload["truth"])
        fold_sp45.append(payload["sp45"])
        fold_dynamic.append(payload["sp45"] + dynamic_weight[index] * direction)
        fold_fixed.append(payload["sp45"] + fixed_weight * direction)
        fold_shuffle.append(payload["sp45"] + shuffled_weight[index] * direction)
        fold_wells.append(np.full(len(payload["truth"]), well))
        oof_dynamic[well] = fold_dynamic[-1]
        oof_fixed[well] = fold_fixed[-1]
        oof_shuffle[well] = fold_shuffle[-1]

    sp45_metric = _metric_from_arrays(fold_sp45, fold_truths, fold_wells)
    dynamic_metric = _metric_from_arrays(fold_dynamic, fold_truths, fold_wells)
    fixed_metric = _metric_from_arrays(fold_fixed, fold_truths, fold_wells)
    shuffle_metric = _metric_from_arrays(fold_shuffle, fold_truths, fold_wells)
    ridge = model.named_steps["ridgecv"]
    scaler = model.named_steps["standardscaler"]
    coefficient = ridge.coef_ / np.maximum(scaler.scale_, 1e-12)
    for feature, value in zip(feature_columns, coefficient):
        coefficient_rows.append({"heldout_fold": heldout, "feature": feature, "coefficient": float(value)})
    fold_rows.append(
        {
            "heldout_fold": heldout,
            "train_wells": len(train),
            "valid_wells": len(valid),
            "ridge_alpha": float(ridge.alpha_),
            "fixed_weight": fixed_weight,
            "mean_dynamic_weight": float(np.mean(dynamic_weight)),
            "mean_shuffle_weight": float(np.mean(shuffled_weight)),
            "sp45_rmse": sp45_metric["rmse"],
            "fixed_rmse": fixed_metric["rmse"],
            "dynamic_rmse": dynamic_metric["rmse"],
            "shuffle_rmse": shuffle_metric["rmse"],
            "dynamic_gain_vs_sp45": sp45_metric["rmse"] - dynamic_metric["rmse"],
            "dynamic_gain_vs_fixed": fixed_metric["rmse"] - dynamic_metric["rmse"],
        }
    )

fold_frame = pd.DataFrame(fold_rows)
fold_frame.to_csv(OUT_DIR / "crossfit_folds.csv", index=False)
pd.DataFrame(coefficient_rows).to_csv(OUT_DIR / "ridge_coefficients.csv", index=False)

ordered_wells = feature_frame["well"].tolist()
truths = [payloads[well]["truth"] for well in ordered_wells]
well_arrays = [np.full(len(payloads[well]["truth"]), well) for well in ordered_wells]
sp45_arrays = [payloads[well]["sp45"] for well in ordered_wells]
dynamic_arrays = [oof_dynamic[well] for well in ordered_wells]
fixed_arrays = [oof_fixed[well] for well in ordered_wells]
shuffle_arrays = [oof_shuffle[well] for well in ordered_wells]

metrics = {
    "sp45": _metric_from_arrays(sp45_arrays, truths, well_arrays),
    "crossfit_fixed": _metric_from_arrays(fixed_arrays, truths, well_arrays),
    "crossfit_dynamic": _metric_from_arrays(dynamic_arrays, truths, well_arrays),
    "shuffled_target_control": _metric_from_arrays(shuffle_arrays, truths, well_arrays),
}
gain_vs_sp45 = float(metrics["sp45"]["rmse"] - metrics["crossfit_dynamic"]["rmse"])
gain_vs_fixed = float(metrics["crossfit_fixed"]["rmse"] - metrics["crossfit_dynamic"]["rmse"])
shuffle_excess_vs_fixed = float(metrics["crossfit_fixed"]["rmse"] - metrics["shuffled_target_control"]["rmse"])
positive_folds = int((fold_frame["dynamic_gain_vs_sp45"] > 0.0).sum())

feature_frame.to_csv(OUT_DIR / "well_features_and_labels.csv", index=False)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "72well_sixfold_posterior_uncertainty_trust_gate",
    "sample_unit": "one complete well",
    "features": feature_columns,
    "feature_count": len(feature_columns),
    "protocol": {
        "wells": WELL_COUNT,
        "outer_folds": OUTER_FOLDS,
        "target": "per-well oracle weight between fold-clean SP45 and frozen seed-path endpoint",
        "ridge_alpha_selection": "training wells only",
        "remaining_fold3_wells_touched": False,
    },
    "metrics": metrics,
    "comparison": {
        "gain_vs_sp45": gain_vs_sp45,
        "gain_vs_crossfit_fixed": gain_vs_fixed,
        "shuffle_excess_gain_vs_fixed": shuffle_excess_vs_fixed,
        "positive_outer_folds": positive_folds,
    },
    "folds": fold_frame.to_dict(orient="records"),
    "top_feature_associations": association_frame.head(15).to_dict(orient="records"),
    "gates": {
        "dynamic_gain_vs_sp45_ge_0p10": bool(gain_vs_sp45 >= 0.10),
        "dynamic_gain_vs_fixed_ge_0p05": bool(gain_vs_fixed >= 0.05),
        "positive_outer_folds_ge_4": bool(positive_folds >= 4),
        "shuffle_does_not_beat_dynamic": bool(
            metrics["crossfit_dynamic"]["rmse"] < metrics["shuffled_target_control"]["rmse"]
        ),
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["gates"].values()))
verdict["next"] = (
    "PASS -> generate the remaining Fold3 seed banks and apply the frozen uncertainty feature/model family"
    if verdict["strict_pass"]
    else "FAIL -> posterior uncertainty does not predict transferable SP45 trust; close this branch"
)
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nTOP FEATURE ASSOCIATIONS")
print(association_frame.head(20).to_string(index=False), flush=True)
print("\nCROSSFIT FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nUNCERTAINTY CROSSFIT METRICS")
print(pd.DataFrame(metrics).T.to_string(), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
