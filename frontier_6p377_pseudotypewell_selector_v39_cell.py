"""Nested selector for the frozen V38 pseudo-typewell SGM candidate bank.

The model never predicts TVT directly.  Its sample is a complete-well
pseudo-reference candidate and its target is whether that frozen candidate
reduces complete-well MSE relative to the assigned-typewell V34 path.

For each held fold, the other two folds cross-predict one another to choose a
probability threshold and a conservative path dose.  The final held-fold model
is then refit on both outer-training folds.  Candidate-aligned feature shuffling
is evaluated with the same frozen models/configurations.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import time
import types

import lightgbm as lgb
import numpy as np
import pandas as pd


EXPERIMENT = "frontier_6p377_pseudotypewell_selector_v39"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V38_SOURCE = ROOT / "frontier_6p377_pseudotypewell_sgm_oracle_v38_cell.py"
V38_DIR = OOF_ROOT / "frontier_6p377_pseudotypewell_sgm_oracle_v38"
PATHS_PATH = V38_DIR / "pseudotypewell_sgm_oof.parquet"
MANIFEST_PATH = V38_DIR / "candidate_manifest.csv"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PATH_COLUMNS = ["candidate_0", "candidate_1", "candidate_2", "candidate_3"]
THRESHOLDS = tuple(np.round(np.arange(0.35, 0.851, 0.05), 2))
DOSES = (0.25, 0.50, 0.75, 1.00)
SEED = 20260727
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


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _load_v38_definitions():
    if not V38_SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing {V38_SOURCE}. Upload the V38 cell beside this file."
        )
    source = V38_SOURCE.read_text(encoding="utf-8")
    marker = "\nfor required_path in (PARENT_PATH, V34_PATH, FOLD_PATH):"
    if marker not in source:
        raise RuntimeError("Cannot locate V38 definition marker")
    name = "frontier_v38_definitions_for_v39"
    module = types.ModuleType(name)
    module.__file__ = str(V38_SOURCE)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0], str(V38_SOURCE), "exec"), module.__dict__)
    return module


R = _load_v38_definitions()
DATA_ROOT = R.DATA_ROOT


def _metrics(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    well_mse = pd.Series(error**2).groupby(frame["well"].to_numpy(), sort=False).mean()
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(well_mse))),
        "p95_well_rmse": float(np.quantile(np.sqrt(well_mse), 0.95)),
        "max_well_rmse": float(np.max(np.sqrt(well_mse))),
    }
    for name, mask in {
        "near_le1000": distance <= 1000.0,
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else np.nan
    return result


def _slope(values, distance):
    values = np.asarray(values, np.float64)
    distance = np.asarray(distance, np.float64)
    if len(values) < 2 or np.ptp(distance) < 1e-6:
        return 0.0
    x = (distance - np.mean(distance)) / max(np.std(distance), 1e-6)
    return float(np.mean(x * (values - np.mean(values))) / max(np.std(x) ** 2, 1e-6))


def _difference_features(prefix, difference, distance):
    output = {}
    difference = np.asarray(difference, np.float64)
    distance = np.asarray(distance, np.float64)
    masks = {
        "tail": distance > 1000.0,
        "mid": (distance > 1000.0) & (distance <= 2000.0),
        "far": distance > 2000.0,
    }
    for label, mask in masks.items():
        values = difference[mask]
        local_distance = distance[mask]
        if not len(values):
            values = np.array([0.0])
            local_distance = np.array([0.0])
        output[f"{prefix}_{label}_mean"] = float(np.mean(values))
        output[f"{prefix}_{label}_rms"] = float(np.sqrt(np.mean(values**2)))
        output[f"{prefix}_{label}_std"] = float(np.std(values))
        output[f"{prefix}_{label}_p95abs"] = float(np.quantile(np.abs(values), 0.95))
        output[f"{prefix}_{label}_endpoint"] = float(values[-1])
        output[f"{prefix}_{label}_slope"] = _slope(values, local_distance)
    return output


def _fit_model(train, feature_names):
    target = train["benefit_positive"].to_numpy(np.int8)
    magnitude = np.clip(np.abs(train["gain_mse"].to_numpy(np.float64)), 1.0, 225.0)
    rows = np.maximum(train["well_rows"].to_numpy(np.float64), 1.0)
    weight = np.sqrt(rows / np.median(rows)) * np.sqrt(magnitude)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=450,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        min_child_samples=24,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=8.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train[feature_names], target, sample_weight=weight)
    return model


def _predict_probability(model, candidates, feature_names, shuffled=False, seed=SEED):
    matrix = candidates[feature_names].copy()
    if shuffled:
        rng = np.random.default_rng(seed)
        # Pandas may expose an Arrow-backed read-only NumPy view here.
        values = np.array(matrix.to_numpy(dtype=np.float64), copy=True)
        wells = candidates["well"].to_numpy()
        for well in np.unique(wells):
            positions = np.flatnonzero(wells == well)
            if len(positions) > 1:
                values[positions] = values[rng.permutation(positions)]
        matrix = pd.DataFrame(values, columns=feature_names, index=candidates.index)
    return model.predict_proba(matrix)[:, 1]


for path in (PATHS_PATH, MANIFEST_PATH):
    if not path.is_file():
        raise FileNotFoundError(path)

paths = pd.read_parquet(
    PATHS_PATH,
    columns=[
        "id", "well", "row_idx", "fold", "tvt_true", "md_since",
        "v33_base", "assigned_typewell_v34", "candidate_family_oracle",
    ] + PATH_COLUMNS,
).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
manifest = pd.read_csv(MANIFEST_PATH)
manifest["well"] = manifest["well"].astype(str)
manifest["rank"] = manifest["rank"].astype(int)
manifest = manifest[manifest["rank"] > 0].copy()
if paths["id"].duplicated().any():
    raise RuntimeError("Duplicate V38 path ids")

groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in paths.groupby("well", sort=False).indices.items()
}
manifest_groups = {str(well): local.copy() for well, local in manifest.groupby("well", sort=False)}
feature_rows = []
feature_failures = []

for number, (well, indices) in enumerate(groups.items(), 1):
    if well not in manifest_groups:
        continue
    try:
        local = paths.iloc[indices]
        distance = local["md_since"].to_numpy(np.float64)
        truth = local["tvt_true"].to_numpy(np.float64)
        own = local["candidate_0"].to_numpy(np.float64)
        base = local["v33_base"].to_numpy(np.float64)
        candidate_matrix = local[PATH_COLUMNS].to_numpy(np.float64).T
        own_mse = float(np.mean((own - truth) ** 2))
        spread = np.std(candidate_matrix, axis=0)

        horizontal = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
            usecols=lambda c: c in {"GR", "TVT_input", "TVT", "MD"},
        ).apply(pd.to_numeric, errors="coerce")
        row_idx = local["row_idx"].to_numpy(np.int64)
        eval_gr = horizontal["GR"].to_numpy(np.float64)[row_idx]
        own_typewell = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__typewell.csv",
            usecols=lambda c: c in {"TVT", "GR"},
        ).apply(pd.to_numeric, errors="coerce")
        own_axis, own_values = R._reference_curve(
            own_typewell["TVT"].to_numpy(np.float64), own_typewell["GR"].to_numpy(np.float64)
        )
        own_fit = R._prefix_fit(horizontal, own_axis, own_values)
        own_prefix_score = float(own_fit["score"]) if own_fit is not None else np.nan

        target_features = {
            "log_well_rows": float(np.log1p(len(local))),
            "max_md_since": float(np.max(distance)),
            "eval_gr_coverage": float(np.mean(np.isfinite(eval_gr))),
            "far_gr_coverage": float(np.mean(np.isfinite(eval_gr[distance > 2000.0]))) if np.any(distance > 2000.0) else 0.0,
            "candidate_spread_tail_rms": float(np.sqrt(np.mean(spread[distance > 1000.0] ** 2))) if np.any(distance > 1000.0) else 0.0,
            "candidate_spread_far_rms": float(np.sqrt(np.mean(spread[distance > 2000.0] ** 2))) if np.any(distance > 2000.0) else 0.0,
            "candidate_spread_endpoint": float(spread[-1]),
            "own_prefix_score": own_prefix_score,
        }
        target_features.update(_difference_features("own_vs_base", own - base, distance))

        local_manifest = manifest_groups[well].set_index("rank")
        for rank in sorted(set(local_manifest.index).intersection({1, 2, 3})):
            row = local_manifest.loc[rank]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            candidate = local[f"candidate_{rank}"].to_numpy(np.float64)
            candidate_mse = float(np.mean((candidate - truth) ** 2))
            gain_mse = own_mse - candidate_mse
            features = {
                "well": well,
                "fold": int(local["fold"].iloc[0]),
                "rank": int(rank),
                "well_rows": int(len(local)),
                "gain_mse": float(gain_mse),
                "benefit_positive": int(gain_mse > 0.0),
                "prefix_score": float(row.get("prefix_score", np.nan)),
                "prefix_score_delta_vs_own": float(row.get("prefix_score", np.nan) - own_prefix_score),
                "prefix_coverage": float(row.get("prefix_coverage", np.nan)),
                "prefix_gain": float(row.get("prefix_gain", np.nan)),
                "prefix_bias": float(row.get("prefix_bias", np.nan)),
                "prefix_gamma": float(row.get("prefix_gamma", np.nan)),
                "geometry_distance_log": float(np.log1p(max(float(row.get("distance", 0.0)), 0.0))),
                "direction_dot": float(row.get("direction_dot", np.nan)),
            }
            features.update(target_features)
            features.update(_difference_features("candidate_vs_own", candidate - own, distance))
            features.update(_difference_features("candidate_vs_base", candidate - base, distance))
            features.update(_difference_features("candidate_vs_mean", candidate - np.mean(candidate_matrix, axis=0), distance))
            feature_rows.append(features)
    except Exception as exc:
        feature_failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(groups):
        print(f"V39 features {number}/{len(groups)} rows={len(feature_rows)} failures={len(feature_failures)}", flush=True)

candidates = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], np.nan)
pd.DataFrame(feature_failures).to_csv(OUT_DIR / "feature_failures.csv", index=False)
if candidates["well"].nunique() < 450:
    raise RuntimeError(f"Only {candidates['well'].nunique()} selector wells")

excluded = {"well", "fold", "rank", "well_rows", "gain_mse", "benefit_positive"}
feature_names = [column for column in candidates.columns if column not in excluded]
for column in feature_names:
    values = pd.to_numeric(candidates[column], errors="coerce")
    median = float(values.median()) if values.notna().any() else 0.0
    candidates[column] = values.fillna(median).astype(np.float32)

path_by_well = {}
for well, indices in groups.items():
    local = paths.iloc[indices]
    path_by_well[well] = {
        "indices": indices,
        "truth": local["tvt_true"].to_numpy(np.float64),
        "own": local["candidate_0"].to_numpy(np.float64),
        "paths": {rank: local[f"candidate_{rank}"].to_numpy(np.float64) for rank in (1, 2, 3)},
    }


def _assemble(candidate_scores, threshold, dose, wells):
    prediction_parts = []
    truth_parts = []
    chosen = []
    score_table = candidate_scores.sort_values(["well", "probability"], ascending=[True, False]).groupby("well", sort=False).first()
    for well in wells:
        payload = path_by_well[well]
        own = payload["own"]
        rank = 0
        probability = 0.0
        if well in score_table.index:
            row = score_table.loc[well]
            probability = float(row["probability"])
            if probability >= threshold:
                rank = int(row["rank"])
        candidate = payload["paths"].get(rank, own)
        prediction_parts.append(own + float(dose) * (candidate - own))
        truth_parts.append(payload["truth"])
        chosen.append({"well": well, "rank": rank, "probability": probability})
    prediction = np.concatenate(prediction_parts)
    truth = np.concatenate(truth_parts)
    return float(np.sqrt(np.mean((prediction - truth) ** 2))), prediction, pd.DataFrame(chosen)


selector_prediction = paths["candidate_0"].to_numpy(np.float64).copy()
shuffle_prediction = selector_prediction.copy()
fold_rows = []
config_rows = []
choice_rows = []

for held_fold in (1, 2, 3):
    training_folds = [fold for fold in (1, 2, 3) if fold != held_fold]
    fold_a, fold_b = training_folds
    train_a = candidates[candidates["fold"] == fold_a].copy()
    train_b = candidates[candidates["fold"] == fold_b].copy()
    model_a = _fit_model(train_a, feature_names)
    model_b = _fit_model(train_b, feature_names)
    inner_a = train_a[["well", "rank"]].copy()
    inner_a["probability"] = _predict_probability(model_b, train_a, feature_names)
    inner_b = train_b[["well", "rank"]].copy()
    inner_b["probability"] = _predict_probability(model_a, train_b, feature_names)
    inner_scores = pd.concat([inner_a, inner_b], ignore_index=True)
    inner_wells = sorted(set(inner_scores["well"]))

    scan = []
    for threshold in THRESHOLDS:
        for dose in DOSES:
            rmse, _, choices = _assemble(inner_scores, threshold, dose, inner_wells)
            scan.append({
                "held_fold": held_fold,
                "threshold": threshold,
                "dose": dose,
                "inner_rmse": rmse,
                "inner_switch_rate": float(np.mean(choices["rank"] > 0)),
            })
    scan = pd.DataFrame(scan).sort_values(["inner_rmse", "dose", "threshold"], kind="stable")
    selected = scan.iloc[0].to_dict()
    config_rows.extend(scan.to_dict(orient="records"))

    outer_train = candidates[candidates["fold"].isin(training_folds)].copy()
    held = candidates[candidates["fold"] == held_fold].copy()
    model = _fit_model(outer_train, feature_names)
    model.booster_.save_model(str(MODEL_DIR / f"selector_fold{held_fold}.txt"))
    held_scores = held[["well", "rank"]].copy()
    held_scores["probability"] = _predict_probability(model, held, feature_names)
    shuffled_scores = held[["well", "rank"]].copy()
    shuffled_scores["probability"] = _predict_probability(
        model, held, feature_names, shuffled=True, seed=SEED + held_fold
    )
    held_wells = sorted(set(paths.loc[paths["fold"] == held_fold, "well"].astype(str)))
    _, held_prediction, held_choices = _assemble(
        held_scores, selected["threshold"], selected["dose"], held_wells
    )
    _, held_shuffle, shuffle_choices = _assemble(
        shuffled_scores, selected["threshold"], selected["dose"], held_wells
    )

    held_indices = np.concatenate([path_by_well[well]["indices"] for well in held_wells])
    # _assemble concatenates sorted wells; align those same chunks explicitly.
    selector_prediction[held_indices] = held_prediction
    shuffle_prediction[held_indices] = held_shuffle
    choice = held_choices.merge(shuffle_choices, on="well", suffixes=("", "_shuffle"))
    choice["fold"] = held_fold
    choice["threshold"] = selected["threshold"]
    choice["dose"] = selected["dose"]
    choice_rows.extend(choice.to_dict(orient="records"))

    fold_mask = paths["fold"].to_numpy(np.int64) == held_fold
    local = paths.loc[fold_mask].reset_index(drop=True)
    own_metrics = _metrics(local, paths.loc[fold_mask, "candidate_0"].to_numpy(np.float64), "assigned_typewell")
    selector_metrics = _metrics(local, selector_prediction[fold_mask], "nested_selector")
    shuffle_metrics = _metrics(local, shuffle_prediction[fold_mask], "candidate_feature_shuffle")
    fold_rows.append({
        "fold": held_fold,
        "threshold": selected["threshold"],
        "dose": selected["dose"],
        "inner_rmse": selected["inner_rmse"],
        "switch_rate": float(np.mean(held_choices["rank"] > 0)),
        "base_rmse": own_metrics["rmse"],
        "rmse": selector_metrics["rmse"],
        "gain": own_metrics["rmse"] - selector_metrics["rmse"],
        "far_gain": own_metrics["far_gt2000_rmse"] - selector_metrics["far_gt2000_rmse"],
        "p95_change": selector_metrics["p95_well_rmse"] - own_metrics["p95_well_rmse"],
        "shuffle_rmse": shuffle_metrics["rmse"],
        "shuffle_loss": shuffle_metrics["rmse"] - selector_metrics["rmse"],
    })

fold_report = pd.DataFrame(fold_rows)
pd.DataFrame(config_rows).to_csv(OUT_DIR / "inner_config_scan.csv", index=False)
pd.DataFrame(choice_rows).to_csv(OUT_DIR / "nested_choices.csv", index=False)
fold_report.to_csv(OUT_DIR / "folds.csv", index=False)

summary = []
for endpoint, prediction in {
    "assigned_typewell_v34": paths["candidate_0"].to_numpy(np.float64),
    "nested_selector": selector_prediction,
    "candidate_feature_shuffle": shuffle_prediction,
    "candidate_family_oracle": paths["candidate_family_oracle"].to_numpy(np.float64),
}.items():
    summary.append(_metrics(paths, prediction, endpoint))
summary = pd.DataFrame(summary)
summary.to_csv(OUT_DIR / "summary.csv", index=False)

output = paths[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["assigned_typewell_v34"] = paths["candidate_0"].astype(np.float32)
output["nested_selector"] = selector_prediction.astype(np.float32)
output["candidate_feature_shuffle"] = shuffle_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "selector_oof.parquet", index=False, compression="zstd")
candidates.to_parquet(OUT_DIR / "candidate_features.parquet", index=False, compression="zstd")

def _row(endpoint):
    return summary[summary["endpoint"] == endpoint].iloc[0].to_dict()

base = _row("assigned_typewell_v34")
selected = _row("nested_selector")
shuffled = _row("candidate_feature_shuffle")
oracle = _row("candidate_family_oracle")
gain = float(base["rmse"] - selected["rmse"])
oracle_gain = float(base["rmse"] - oracle["rmse"])
capture = gain / max(oracle_gain, 1e-12)
checks = {
    "selector_gain_ge_0p05": gain >= 0.05,
    "minimum_fold_gain_ge_minus0p02": float(fold_report["gain"].min()) >= -0.02,
    "positive_folds_ge_2": int((fold_report["gain"] > 0).sum()) >= 2,
    "oracle_capture_ge_0p25": capture >= 0.25,
    "candidate_shuffle_worse_by_ge_0p02": float(shuffled["rmse"] - selected["rmse"]) >= 0.02,
    "maximum_fold_p95_change_le_0p15": float(fold_report["p95_change"].max()) <= 0.15,
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_complete_well_pseudotypewell_selector",
    "sample_unit": "complete-well pseudo-reference candidate",
    "source": str(V38_DIR),
    "features": feature_names,
    "feature_count": len(feature_names),
    "candidate_wells": int(candidates["well"].nunique()),
    "candidate_rows": int(len(candidates)),
    "protocol": {
        "outer_folds": [1, 2, 3],
        "inner_selection": "two-fold cross-prediction within each outer training set",
        "thresholds": THRESHOLDS,
        "doses": DOSES,
        "target": "candidate complete-well MSE improvement over assigned typewell",
        "near_le1000_policy": "inherited bitwise freeze from V38",
    },
    "base": base,
    "nested_selector": selected,
    "candidate_feature_shuffle": shuffled,
    "candidate_family_oracle": oracle,
    "gain": gain,
    "oracle_gain": oracle_gain,
    "oracle_capture_fraction": capture,
    "folds": fold_report.to_dict(orient="records"),
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> refit selector and package pseudo-typewell candidate generation for Kaggle"
        if all(checks.values())
        else "FAIL -> legal candidate features cannot recover the V38 oracle; keep V35 and close selector"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV39 SUMMARY")
print(summary.to_string(index=False))
print("\nV39 FOLDS")
print(fold_report.to_string(index=False))
print("\nV39 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR)
