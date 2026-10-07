"""HN19: same-well pseudo-cut replay adaptation gateway.

This is not another multi-cut training experiment.  The frozen matched-K5
model is replayed from several earlier pseudo prediction starts.  Only replay
rows before the well's real prediction start are scored, so every diagnostic
is observable for the corresponding test well.  A small cross-fitted ridge
model maps those diagnostics to a smooth correction of the complete V66
suffix.  Official suffix truth is used only for fitting/evaluation in outer
folds and never enters a held well's features.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF = ROOT / "oof_exports"
EXPERIMENT = "samewell_pseudocut_adaptation_hn19"
OUT_DIR = OOF / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V66_PATH = (
    OOF / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
K3_DIR = OOF / "multicut_fleongg_matched_k3_gateway_v1"
K5_DIR = OOF / "multicut_fleongg_matched_k5_extension_v1"
FEATURE_DIR = K3_DIR / "features"
META_PATH = K3_DIR / "well_metadata.csv"
K3_MANIFEST = K3_DIR / "cut_manifest.csv"
K5_MANIFEST = K5_DIR / "new_cut_manifest.csv"
FEATURE_SCHEMA = K3_DIR / "matched_k3_gateway_features.csv"
MODEL_DIR = K5_DIR / "models"
K3_MODEL_DIR = K3_DIR / "models"

TAGS = ("pseudo1", "pseudo2", "pseudo3", "pseudo4")
PROGRESS_EDGES = np.linspace(0.0, 1.0, 6)
PROGRESS_CENTRES = (PROGRESS_EDGES[:-1] + PROGRESS_EDGES[1:]) / 2.0
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DOSES = (0.0, 0.25, 0.50, 0.75, 1.0)
CORRECTION_CAP_FT = 8.0
MIN_VISIBLE_REPLAY_ROWS = 24
SEED = 20260731
REBUILD_ROWS_PER_STATE = 320
REBUILD_ESTIMATORS = 800

for required in (
    V66_PATH, META_PATH, K3_MANIFEST, FEATURE_SCHEMA, FEATURE_DIR
):
    if not required.exists():
        raise FileNotFoundError(required)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _metrics(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    wells = frame["well"].astype(str).to_numpy()
    well_mse = pd.Series(error * error).groupby(wells, sort=False).mean()
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(well_mse))),
        "p95_well_rmse": float(np.quantile(np.sqrt(well_mse), 0.95)),
        "max_well_rmse": float(np.max(np.sqrt(well_mse))),
        "far_gt2000_rmse": (
            float(np.sqrt(np.mean(error[far] ** 2))) if np.any(far) else None
        ),
    }


def _prediction_column(frame):
    preferred = (
        "incumbent_6p377_replay", "current_dose_0p20", "prediction", "pred"
    )
    for column in preferred:
        if column in frame:
            return column
    excluded = {"id", "well", "row_idx", "fold", "tvt_true", "md_since"}
    numeric = [
        column for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(numeric) != 1:
        raise RuntimeError(f"Cannot identify V66 prediction column: {numeric}")
    return numeric[0]


def _load_manifest():
    parts = [pd.read_csv(K3_MANIFEST, dtype={"well": str})]
    if K5_MANIFEST.is_file():
        parts.append(pd.read_csv(K5_MANIFEST, dtype={"well": str}))
    manifest = pd.concat(parts, ignore_index=True, sort=False)
    return manifest.drop_duplicates(["well", "tag"], keep="last")


def _sample_path(well, tag):
    return FEATURE_DIR / f"{well}__{tag}__sample.pkl"


def _rebuild_k3_models(meta, feature_names):
    output_dir = OUT_DIR / "rebuilt_replay_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        fold: output_dir / f"replay_k3_fold{fold}.joblib"
        for fold in range(1, 6)
    }
    if all(path.is_file() for path in paths.values()):
        return paths

    rng = np.random.default_rng(SEED)
    tags = ("official", "pseudo1", "pseudo2")
    for held_fold in range(1, 6):
        output_path = paths[held_fold]
        if output_path.is_file():
            continue
        matrices = []
        targets = []
        weights = []
        train_wells = meta.loc[meta["fold"].astype(int) != held_fold, "well"].astype(str)
        loaded_states = 0
        for number, well in enumerate(train_wells, 1):
            for tag in tags:
                path = _sample_path(well, tag)
                if not path.is_file():
                    continue
                frame = pd.read_pickle(path)
                if "target" not in frame or len(frame) < 8:
                    continue
                if len(frame) > REBUILD_ROWS_PER_STATE:
                    positions = np.linspace(
                        0, len(frame) - 1, REBUILD_ROWS_PER_STATE
                    ).round().astype(np.int64)
                    # A tiny random phase prevents every well using identical MD bins.
                    shift = int(rng.integers(0, max(len(frame) // REBUILD_ROWS_PER_STATE, 1)))
                    positions = np.clip(positions + shift, 0, len(frame) - 1)
                    positions = np.unique(positions)
                    frame = frame.iloc[positions]
                matrix = frame.reindex(columns=feature_names).to_numpy(np.float32)
                matrix[np.isinf(matrix)] = np.nan
                target = frame["target"].to_numpy(np.float32)
                finite_target = np.isfinite(target)
                if finite_target.sum() < 8:
                    continue
                matrix = matrix[finite_target]
                target = target[finite_target]
                matrices.append(matrix)
                targets.append(target)
                # Equal state weight, hence approximately equal well/cut influence.
                weights.append(np.full(len(target), 1.0 / len(target), np.float32))
                loaded_states += 1
            if number % 100 == 0:
                print(
                    f"HN19 rebuild fold={held_fold} wells={number}/{len(train_wells)} states={loaded_states}",
                    flush=True,
                )
        if loaded_states < 1200:
            raise RuntimeError(
                f"Fold {held_fold}: only {loaded_states} cached replay training states"
            )
        x = np.concatenate(matrices)
        y = np.concatenate(targets)
        sample_weight = np.concatenate(weights).astype(np.float64)
        sample_weight *= len(sample_weight) / sample_weight.sum()
        params = dict(
            objective="regression",
            n_estimators=REBUILD_ESTIMATORS,
            learning_rate=0.035,
            num_leaves=64,
            min_child_samples=80,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_alpha=0.10,
            reg_lambda=10.0,
            max_bin=255,
            n_jobs=-1,
            random_state=SEED + held_fold,
            verbosity=-1,
        )
        model = lgb.LGBMRegressor(**params)
        model.fit(x, y, sample_weight=sample_weight)
        joblib.dump(model, output_path)
        print(
            "HN19 rebuilt replay model",
            {"fold": held_fold, "rows": len(y), "states": loaded_states, "path": str(output_path)},
            flush=True,
        )
        del x, y, sample_weight, matrices, targets, weights, model
    return paths


def _resolve_model_family(meta, feature_names):
    families = (
        (
            "matched_k5",
            TAGS,
            [MODEL_DIR / f"k5_fold{fold}.joblib" for fold in range(1, 6)],
        ),
        (
            "matched_k3",
            ("pseudo1", "pseudo2"),
            [K3_MODEL_DIR / f"matched_k3_fold{fold}.joblib" for fold in range(1, 6)],
        ),
    )
    for name, tags, paths in families:
        if all(path.is_file() for path in paths):
            return name, tuple(tags), dict(zip(range(1, 6), paths))
    inventory = {
        "k5": sorted(path.name for path in MODEL_DIR.glob("*.joblib"))
        if MODEL_DIR.is_dir() else [],
        "k3": sorted(path.name for path in K3_MODEL_DIR.glob("*.joblib"))
        if K3_MODEL_DIR.is_dir() else [],
    }
    print(
        "HN19 frozen replay models unavailable; rebuilding from cached samples",
        inventory,
        flush=True,
    )
    rebuilt = _rebuild_k3_models(meta, feature_names)
    return "rebuilt_matched_k3_fixed800", ("pseudo1", "pseudo2"), rebuilt


def _postprocess_sparse(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(np.float64)
    md_since = frame["md_since"].to_numpy(np.float64)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / 85.0)
    lik = frame.get("likpf_scale_5", frame["pf_ancc"])
    lik = pd.to_numeric(lik, errors="coerce").to_numpy(np.float64)
    lik = np.where(np.isfinite(lik), lik, last)
    return last + 0.60 * warm * np.asarray(model_delta) + 0.40 * (lik - last)


def _slope(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 8 or np.ptp(x[valid]) < 1e-6:
        return np.nan
    x = x[valid] - np.mean(x[valid])
    return float(np.dot(x, y[valid] - np.mean(y[valid])) / np.dot(x, x))


def _summarise_error(prefix, error, tag):
    prefix = np.asarray(prefix, np.float64)
    error = np.asarray(error, np.float64)
    valid = np.isfinite(prefix) & np.isfinite(error)
    prefix, error = prefix[valid], error[valid]
    output = {f"{tag}_rows": int(len(error))}
    if len(error) < MIN_VISIBLE_REPLAY_ROWS:
        for name in (
            "bias", "median", "rmse", "mae", "p90_abs", "slope",
            "late_bias", "late_rmse", "endpoint", "sign_consistency",
        ):
            output[f"{tag}_{name}"] = np.nan
        return output
    order = np.argsort(prefix)
    prefix, error = prefix[order], error[order]
    late = np.arange(len(error)) >= max(0, int(np.floor(0.67 * len(error))))
    endpoint_n = min(16, len(error))
    output.update({
        f"{tag}_bias": float(np.mean(error)),
        f"{tag}_median": float(np.median(error)),
        f"{tag}_rmse": float(np.sqrt(np.mean(error ** 2))),
        f"{tag}_mae": float(np.mean(np.abs(error))),
        f"{tag}_p90_abs": float(np.quantile(np.abs(error), 0.90)),
        f"{tag}_slope": _slope(prefix / 1000.0, error),
        f"{tag}_late_bias": float(np.mean(error[late])),
        f"{tag}_late_rmse": float(np.sqrt(np.mean(error[late] ** 2))),
        f"{tag}_endpoint": float(np.mean(error[-endpoint_n:])),
        f"{tag}_sign_consistency": float(abs(np.mean(np.sign(error)))),
    })
    return output


def _well_targets(frame):
    rows = []
    for well, group in frame.groupby("well", sort=False):
        group = group.sort_values("md_since")
        distance = group["md_since"].to_numpy(np.float64)
        progress = distance / max(float(np.max(distance)), 1.0)
        error = group["base"].to_numpy(np.float64) - group["tvt_true"].to_numpy(np.float64)
        row = {
            "well": str(well),
            "fold": int(group["fold"].iloc[0]),
            "official_rows": int(len(group)),
            "max_md_since": float(np.max(distance)),
        }
        for index in range(len(PROGRESS_CENTRES)):
            mask = (progress >= PROGRESS_EDGES[index]) & (
                progress <= PROGRESS_EDGES[index + 1]
                if index == len(PROGRESS_CENTRES) - 1
                else progress < PROGRESS_EDGES[index + 1]
            )
            row[f"target_bin{index}"] = float(np.mean(error[mask])) if np.any(mask) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _pipeline(alpha):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


def _fit_predict(train, valid, feature_names, alpha):
    targets = [f"target_bin{i}" for i in range(len(PROGRESS_CENTRES))]
    model = _pipeline(alpha)
    weight = train["official_rows"].to_numpy(np.float64).copy()
    weight /= max(float(np.mean(weight)), 1.0)
    model.fit(
        train[feature_names], train[targets], ridge__sample_weight=weight
    )
    return model, model.predict(valid[feature_names])


def _apply(frame, well_predictions, dose):
    output = frame["base"].to_numpy(np.float64).copy()
    lookup = {str(well): values for well, values in well_predictions.items()}
    for well, positions in frame.groupby("well", sort=False).indices.items():
        positions = np.asarray(positions, np.int64)
        distance = frame.iloc[positions]["md_since"].to_numpy(np.float64)
        progress = distance / max(float(np.max(distance)), 1.0)
        knot_error = np.clip(lookup[str(well)], -CORRECTION_CAP_FT, CORRECTION_CAP_FT)
        correction = np.interp(progress, PROGRESS_CENTRES, knot_error)
        output[positions] -= float(dose) * correction
    return output


meta = pd.read_csv(META_PATH, dtype={"well": str})[["well", "fold"]]
meta["fold"] = meta["fold"].astype(int)
manifest = _load_manifest()
feature_names = pd.read_csv(FEATURE_SCHEMA)["feature"].astype(str).tolist()
MODEL_FAMILY, TAGS, MODEL_PATHS = _resolve_model_family(meta, feature_names)
print(
    "HN19 REPLAY FAMILY",
    {"family": MODEL_FAMILY, "tags": TAGS, "models": [str(MODEL_PATHS[i]) for i in range(1, 6)]},
    flush=True,
)

base = pd.read_parquet(V66_PATH)
base["well"] = base["well"].astype(str)
base_column = _prediction_column(base)
base = base.rename(columns={base_column: "base"})
needed = ["id", "well", "row_idx", "fold", "tvt_true", "md_since", "base"]
missing = [column for column in needed if column not in base]
if missing:
    raise RuntimeError(f"V66 missing {missing}")
base = base[needed].copy()

fold_lookup = dict(zip(meta["well"], meta["fold"]))
manifest_lookup = {
    (str(row.well), str(row.tag)): row
    for row in manifest.itertuples(index=False)
}
models = {}
for fold in range(1, 6):
    path = MODEL_PATHS[fold]
    models[fold] = joblib.load(path)

diagnostics = []
failures = []
for number, well in enumerate(meta["well"].astype(str), 1):
    try:
        fold = int(fold_lookup[well])
        official = manifest_lookup[(well, "official")]
        official_cut = int(official.cut_position)
        row = {"well": well, "replay_fold": fold}
        available_errors = []
        for tag in TAGS:
            path = _sample_path(well, tag)
            if not path.is_file() or (well, tag) not in manifest_lookup:
                row.update(_summarise_error([], [], tag))
                continue
            sample = pd.read_pickle(path)
            matrix = sample.reindex(columns=feature_names).to_numpy(np.float32)
            matrix[np.isinf(matrix)] = np.nan
            delta = models[fold].predict(matrix)
            prediction = _postprocess_sparse(sample, delta)
            truth = (
                sample["target"].to_numpy(np.float64)
                + sample["last_known_tvt"].to_numpy(np.float64)
            )
            source_rows = sample["_row_idx"].to_numpy(np.int64)
            legal = source_rows < official_cut
            error = prediction[legal] - truth[legal]
            replay_distance = sample.loc[legal, "md_since"].to_numpy(np.float64)
            row.update(_summarise_error(replay_distance, error, tag))
            if len(error) >= MIN_VISIBLE_REPLAY_ROWS:
                available_errors.append(error)
        row["usable_tags"] = int(sum(row[f"{tag}_rows"] >= MIN_VISIBLE_REPLAY_ROWS for tag in TAGS))
        diagnostics.append(row)
    except Exception as error:
        failures.append({"well": well, "reason": repr(error)})
    if number % 50 == 0 or number == len(meta):
        print(
            f"HN19 replay {number}/{len(meta)} success={len(diagnostics)} failures={len(failures)}",
            flush=True,
        )

diagnostics = pd.DataFrame(diagnostics)
pd.DataFrame(failures).to_csv(OUT_DIR / "replay_failures.csv", index=False)
if diagnostics["well"].nunique() < 700:
    raise RuntimeError(f"Only {diagnostics['well'].nunique()} replay wells: {failures[:5]}")

targets = _well_targets(base)
well_frame = targets.merge(diagnostics, on="well", how="inner", validate="one_to_one")
replay_features = [
    column for column in diagnostics.columns
    if column not in {"well", "replay_fold"} and column != "usable_tags"
]
control_features = ["official_rows", "max_md_since"]
all_features = control_features + ["usable_tags"] + replay_features

fold_reports = []
oof_prediction = np.full(len(base), np.nan, np.float64)
control_prediction = np.full(len(base), np.nan, np.float64)
shuffle_prediction = np.full(len(base), np.nan, np.float64)
rng = np.random.default_rng(SEED)

for held_fold in range(1, 6):
    train_wells = well_frame[well_frame["fold"] != held_fold].copy()
    held_wells = well_frame[well_frame["fold"] == held_fold].copy()
    if held_wells["well"].nunique() < 145:
        raise RuntimeError(f"Fold {held_fold} only has {held_wells['well'].nunique()} wells")

    # Select alpha and dose only by cross-predicting folds inside outer training.
    scan = []
    for alpha in ALPHAS:
        inner_well_predictions = {}
        for inner_fold in sorted(train_wells["fold"].unique()):
            inner_train = train_wells[train_wells["fold"] != inner_fold]
            inner_valid = train_wells[train_wells["fold"] == inner_fold]
            _, prediction = _fit_predict(inner_train, inner_valid, all_features, alpha)
            inner_well_predictions.update(dict(zip(inner_valid["well"], prediction)))
        inner_rows = base[base["well"].isin(train_wells["well"])].copy()
        base_score = _metrics(inner_rows, inner_rows["base"], "base")["rmse"]
        for dose in DOSES:
            candidate = _apply(inner_rows, inner_well_predictions, dose)
            score = _metrics(inner_rows, candidate, "candidate")["rmse"]
            scan.append({"alpha": alpha, "dose": dose, "rmse": score, "gain": base_score - score})
    scan = pd.DataFrame(scan).sort_values(["rmse", "alpha", "dose"]).reset_index(drop=True)
    selected = scan.iloc[0]
    alpha, dose = float(selected.alpha), float(selected.dose)

    _, held_knots = _fit_predict(train_wells, held_wells, all_features, alpha)
    _, control_knots = _fit_predict(train_wells, held_wells, control_features, alpha)
    shuffled = held_wells.copy()
    shuffled_values = shuffled[replay_features + ["usable_tags"]].to_numpy(copy=True)
    shuffled_values = shuffled_values[rng.permutation(len(shuffled_values))]
    shuffled.loc[:, replay_features + ["usable_tags"]] = shuffled_values
    _, shuffled_knots = _fit_predict(train_wells, shuffled, all_features, alpha)

    held_mask = base["fold"].to_numpy(int) == held_fold
    held_rows = base.loc[held_mask].copy()
    full_lookup = dict(zip(held_wells["well"], held_knots))
    control_lookup = dict(zip(held_wells["well"], control_knots))
    shuffle_lookup = dict(zip(held_wells["well"], shuffled_knots))
    candidate = _apply(held_rows, full_lookup, dose)
    control = _apply(held_rows, control_lookup, dose)
    shuffled_candidate = _apply(held_rows, shuffle_lookup, dose)
    oof_prediction[held_mask] = candidate
    control_prediction[held_mask] = control
    shuffle_prediction[held_mask] = shuffled_candidate

    base_m = _metrics(held_rows, held_rows["base"], "base")
    full_m = _metrics(held_rows, candidate, "replay")
    control_m = _metrics(held_rows, control, "control")
    shuffle_m = _metrics(held_rows, shuffled_candidate, "shuffle")
    fold_reports.append({
        "fold": held_fold,
        "selected_alpha": alpha,
        "selected_dose": dose,
        "inner_gain": float(selected.gain),
        "rows": int(len(held_rows)),
        "wells": int(held_rows["well"].nunique()),
        "base_rmse": base_m["rmse"],
        "control_rmse": control_m["rmse"],
        "rmse": full_m["rmse"],
        "gain": base_m["rmse"] - full_m["rmse"],
        "increment_vs_control": control_m["rmse"] - full_m["rmse"],
        "far_gain": base_m["far_gt2000_rmse"] - full_m["far_gt2000_rmse"],
        "shuffle_loss": shuffle_m["rmse"] - full_m["rmse"],
        "p95_change": full_m["p95_well_rmse"] - base_m["p95_well_rmse"],
    })
    print("HN19 FOLD", fold_reports[-1], flush=True)

if not np.isfinite(oof_prediction).all():
    raise RuntimeError("Incomplete HN19 crossfit prediction")

folds = pd.DataFrame(fold_reports)
base_metrics = _metrics(base, base["base"], "base")
control_metrics = _metrics(base, control_prediction, "population_control")
replay_metrics = _metrics(base, oof_prediction, "samewell_replay")
shuffle_metrics = _metrics(base, shuffle_prediction, "replay_well_shuffle")
checks = {
    "overall_gain_ge_0p05": bool(base_metrics["rmse"] - replay_metrics["rmse"] >= 0.05),
    "increment_vs_control_ge_0p03": bool(control_metrics["rmse"] - replay_metrics["rmse"] >= 0.03),
    "positive_folds_ge_4": bool((folds["gain"] > 0).sum() >= 4),
    "minimum_fold_gain_ge_minus0p03": bool(folds["gain"].min() >= -0.03),
    "shuffle_loses_ge_0p03": bool(shuffle_metrics["rmse"] - replay_metrics["rmse"] >= 0.03),
    "replay_usable_wells_ge_700": bool((well_frame["usable_tags"] > 0).sum() >= 700),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_target_well_pseudocut_error_adaptation_gateway",
    "sample_unit": "complete physical well",
    "legal_feature_boundary": "only pseudo-cut replay rows before official PS",
    "sources": {
        "v66": str(V66_PATH),
        "k5_features": str(FEATURE_DIR),
        "replay_model_family": MODEL_FAMILY,
        "replay_models": [str(MODEL_PATHS[i]) for i in range(1, 6)],
    },
    "coverage": {
        "rows": int(len(base)),
        "wells": int(base["well"].nunique()),
        "diagnostic_wells": int(len(well_frame)),
        "usable_replay_wells": int((well_frame["usable_tags"] > 0).sum()),
        "failures": int(len(failures)),
    },
    "features": {"control": len(control_features), "replay": len(all_features)},
    "base": base_metrics,
    "population_control": control_metrics,
    "samewell_replay": replay_metrics,
    "replay_well_shuffle": shuffle_metrics,
    "gain": base_metrics["rmse"] - replay_metrics["rmse"],
    "increment_vs_control": control_metrics["rmse"] - replay_metrics["rmse"],
    "shuffle_loss": shuffle_metrics["rmse"] - replay_metrics["rmse"],
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> package target-well pseudo-cut replay and frozen meta-corrector"
        if strict_pass
        else "FAIL -> pseudo-cut errors do not extrapolate; close test-time adaptation"
    ),
}

output = base[["id", "well", "row_idx", "fold", "tvt_true", "md_since", "base"]].copy()
output["population_control"] = control_prediction
output["samewell_replay"] = oof_prediction
output["replay_well_shuffle"] = shuffle_prediction
output.to_parquet(OUT_DIR / "hn19_oof.parquet", index=False)
well_frame.to_parquet(OUT_DIR / "replay_well_features.parquet", index=False)
folds.to_csv(OUT_DIR / "folds.csv", index=False)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8"
)
print("\nHN19 FOLDS")
print(folds.to_string(index=False))
print("\nHN19 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR)
