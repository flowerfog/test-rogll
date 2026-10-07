"""Exact HN7 deployment-equivalence audit on HN's fully unseen Fold 1.

All four HN3 models excluded stable Fold 1 from training. This cell reproduces
Kaggle's mean-pair-probability tournament on every eligible Fold-1 well and
compares the frozen 0.275 blend against the complete V66 parent OOF.
"""

from pathlib import Path
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V4_SOURCE = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
HN3_DIR = OOF_ROOT / "realistic_pairwise_crossfit_hn3"
V66_PATH = (
    OOF_ROOT
    / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
OUT_DIR = OOF_ROOT / "realistic_pairwise_deployment_equivalence_hn10"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUT_DIR / "hn_fold1_deployment_endpoints.parquet"

HN_FOLD = 1
PRIOR_FOLDS = (4, 5)
MODEL_FOLDS = (2, 3, 4, 5)
MENU_SIZE = 128
PAIR_BATCH = 65536
FROZEN_WEIGHT = 0.275


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _bootstrap_v4():
    source = V4_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V4 execution marker missing from {V4_SOURCE}")
    namespace = {"__name__": "hn10_v4_definitions", "__file__": str(V4_SOURCE)}
    exec(compile(source.split(marker, 1)[0], str(V4_SOURCE), "exec"), namespace)
    return namespace


def _tournament_scores(models, values):
    count = len(values)
    if count == 1:
        return np.ones((len(models), 1), np.float64), np.ones(1, np.float64)
    sums = np.zeros((len(models), count), np.float64)
    seen = np.zeros(count, np.int64)
    left_all, right_all = np.triu_indices(count, 1)
    for start in range(0, len(left_all), PAIR_BATCH):
        left = left_all[start : start + PAIR_BATCH]
        right = right_all[start : start + PAIR_BATCH]
        delta = values[left] - values[right]
        probabilities = np.vstack([model.predict(delta) for model in models])
        for model_index, probability in enumerate(probabilities):
            sums[model_index, left] += probability
            sums[model_index, right] += 1.0 - probability
        seen[left] += 1
        seen[right] += 1
    single = sums / np.maximum(seen[None, :], 1)
    # This is algebraically identical to averaging pair probabilities first.
    ensemble = np.mean(single, axis=0)
    return single, ensemble


def _metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    far = distance > 2000.0
    well_rmse = (
        pd.DataFrame({"well": frame["well"].astype(str), "error2": error * error})
        .groupby("well", sort=False)["error2"]
        .mean()
        .pow(0.5)
    )
    return {
        "rows": int(len(frame)),
        "wells": int(well_rmse.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


if not V4_SOURCE.is_file() or not V66_PATH.is_file():
    raise FileNotFoundError(f"Missing V4/V66 source: {V4_SOURCE} / {V66_PATH}")
Q = _bootstrap_v4()
V = Q["V"]
B = Q["B"]

selected_features = json.loads(
    (HN3_DIR / "selected_features.json").read_text(encoding="utf-8")
)
models = [
    lgb.Booster(model_file=str(HN3_DIR / f"pairwise_hold{fold}.txt"))
    for fold in MODEL_FOLDS
]
if any(model.num_feature() != len(selected_features) for model in models):
    raise RuntimeError("HN model/feature schema mismatch")

data_root = Path(Q["DATA_ROOT"])
all_wells = [
    path.name.split("__", 1)[0]
    for path in sorted((data_root / "train").glob("*__horizontal_well.csv"))
]
if len(all_wells) != 773:
    raise RuntimeError(f"Expected 773 train wells, got {len(all_wells)}")
stable_folds = V["_stable_folds"](all_wells)
stable_fold_by_well = dict(
    zip(stable_folds["well"].astype(str), stable_folds["fold"].astype(int))
)
requested_wells = sorted(
    well for well in all_wells if stable_fold_by_well[well] == HN_FOLD
)

if CACHE_PATH.is_file():
    endpoints = pd.read_parquet(CACHE_PATH)
    failures = []
    print(
        "HN10 CACHE",
        {"rows": len(endpoints), "wells": int(endpoints["well"].nunique())},
        flush=True,
    )
else:
    prior_records = []
    prior_failures = []
    prior_wells = [well for well in all_wells if stable_fold_by_well[well] in PRIOR_FOLDS]
    for number, well in enumerate(prior_wells, 1):
        try:
            prior_records.append(V["_load_record"](well))
        except Exception as exc:
            prior_failures.append({"well": well, "reason": repr(exc)})
        if number % 50 == 0 or number == len(prior_wells):
            print(
                f"HN10 prior {number}/{len(prior_wells)} failures={len(prior_failures)}",
                flush=True,
            )
    if len(prior_records) < 307:
        raise RuntimeError(f"Only {len(prior_records)} prior records")
    prior = V["_fit_prior"](prior_records, Q["SEGMENT_LENGTH_FT"])

    blocks = []
    failures = []
    for number, well in enumerate(requested_wells, 1):
        try:
            record = V["_load_record"](well)
            paths, source_mask, arm_rank, arm_score_z, segments, _ = Q["_candidate_union"](
                record, prior
            )
            features, feature_names = Q["_candidate_evidence"](
                record, prior, paths, segments, arm_rank, arm_score_z, source_mask
            )
            feature_index = {name: index for index, name in enumerate(feature_names)}
            missing = [name for name in selected_features if name not in feature_index]
            if missing:
                raise RuntimeError(f"missing selected features: {missing[:10]}")
            columns = np.asarray([feature_index[name] for name in selected_features], np.int64)
            legal_rank = np.min(np.asarray(arm_rank, np.float64), axis=1)
            menu = np.argsort(legal_rank, kind="stable")[: min(MENU_SIZE, len(paths))]
            values = np.asarray(features[menu][:, columns], np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=12.0, neginf=-12.0)
            single_scores, ensemble_score = _tournament_scores(models, values)
            selected = {
                f"hn_model_hold{fold}": int(menu[int(np.argmax(single_scores[index]))])
                for index, fold in enumerate(MODEL_FOLDS)
            }
            selected["hn_ensemble"] = int(menu[int(np.argmax(ensemble_score))])
            eval_rows = np.asarray(record["eval_rows"], np.int64)
            block = pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in eval_rows],
                    "well": well,
                    "row_idx": eval_rows,
                }
            )
            for endpoint, path_index in selected.items():
                prediction = B["_path_prediction"](
                    record, prior, paths[path_index], segments
                )
                block[endpoint] = np.asarray(prediction[eval_rows], np.float32)
            blocks.append(block)
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 20 == 0 or number == len(requested_wells):
            print(
                f"HN10 inference {number}/{len(requested_wells)} "
                f"success={len(blocks)} failures={len(failures)}",
                flush=True,
            )
    pd.DataFrame(prior_failures).to_csv(
        OUT_DIR / "prior_record_failures.csv", index=False
    )
    pd.DataFrame(failures).to_csv(OUT_DIR / "inference_failures.csv", index=False)
    if len(blocks) < 145:
        raise RuntimeError(
            f"Only {len(blocks)}/{len(requested_wells)} unseen Fold1 wells succeeded: "
            f"{failures[:5]}"
        )
    endpoints = pd.concat(blocks, ignore_index=True)
    endpoints.to_parquet(CACHE_PATH, index=False, compression="zstd")

v66 = pd.read_parquet(
    V66_PATH,
    columns=[
        "id", "well", "row_idx", "fold", "tvt_true", "md_since",
        "incumbent_6p377_replay",
    ],
)
v66["id"] = v66["id"].astype(str)
v66["well"] = v66["well"].astype(str)
endpoints["id"] = endpoints["id"].astype(str)
endpoints["well"] = endpoints["well"].astype(str)
aligned = v66.merge(
    endpoints,
    on=["id", "well", "row_idx"],
    how="inner",
    validate="one_to_one",
)
if aligned["well"].nunique() < 145:
    raise RuntimeError(f"Only {aligned['well'].nunique()} HN Fold1 wells align to V66")
if any(stable_fold_by_well[well] != HN_FOLD for well in aligned["well"].unique()):
    raise RuntimeError("Aligned endpoint includes an HN training fold")

base = aligned["incumbent_6p377_replay"].to_numpy(np.float64)
base_metrics = _metrics(aligned, base)
endpoint_columns = [f"hn_model_hold{fold}" for fold in MODEL_FOLDS] + ["hn_ensemble"]
rows = []
for endpoint in endpoint_columns:
    candidate = aligned[endpoint].to_numpy(np.float64)
    candidate_metrics = _metrics(aligned, candidate)
    blend = base + FROZEN_WEIGHT * (candidate - base)
    blend_metrics = _metrics(aligned, blend)
    rows.append(
        {
            "endpoint": endpoint,
            "candidate_rmse": candidate_metrics["rmse"],
            "blend_rmse": blend_metrics["rmse"],
            "gain": base_metrics["rmse"] - blend_metrics["rmse"],
            "far_gain": base_metrics["far_gt2000_rmse"]
            - blend_metrics["far_gt2000_rmse"],
            "p95_change": blend_metrics["p95_well_rmse"]
            - base_metrics["p95_well_rmse"],
            "median_change": blend_metrics["median_well_rmse"]
            - base_metrics["median_well_rmse"],
        }
    )
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "endpoint_summary.csv", index=False)
ensemble_row = summary.loc[summary["endpoint"].eq("hn_ensemble")].iloc[0]
checks = {
    "unseen_fold_wells_ge_145": bool(aligned["well"].nunique() >= 145),
    "deployment_ensemble_gain_ge_0p10": bool(ensemble_row["gain"] >= 0.10),
    "deployment_ensemble_far_gain_ge_0p10": bool(ensemble_row["far_gain"] >= 0.10),
    "deployment_ensemble_p95_change_le_0p20": bool(ensemble_row["p95_change"] <= 0.20),
    "ensemble_not_worse_than_median_single": bool(
        ensemble_row["blend_rmse"]
        <= summary.loc[summary["endpoint"].ne("hn_ensemble"), "blend_rmse"].median()
    ),
}
verdict = {
    "experiment": "realistic_pairwise_deployment_equivalence_hn10",
    "stage": "exact_hn7_four_model_ensemble_on_fully_unseen_hn_fold1",
    "selection_or_tuning_count": 0,
    "sources": {"v66": V66_PATH, "hn3_models": HN3_DIR},
    "protocol": {
        "hn_holdout_fold": HN_FOLD,
        "models": list(MODEL_FOLDS),
        "all_models_unseen_on_every_evaluation_well": True,
        "aggregation": "mean_pair_probability_then_round_robin_argmax",
        "frozen_weight": FROZEN_WEIGHT,
        "menu_size": MENU_SIZE,
    },
    "coverage": {
        "requested_wells": len(requested_wells),
        "successful_wells": int(endpoints["well"].nunique()),
        "aligned_wells": int(aligned["well"].nunique()),
        "aligned_rows": int(len(aligned)),
        "canonical_v66_fold_wells": (
            aligned[["well", "fold"]]
            .drop_duplicates()
            .groupby("fold")["well"]
            .nunique()
            .astype(int)
            .to_dict()
        ),
    },
    "base": base_metrics,
    "endpoints": rows,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> deployment endpoint is locally equivalent; LB null indicates hidden shift"
        if all(checks.values())
        else "FAIL -> HN7 four-model deployment aggregation does not reproduce crossfit gain"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)
print("\nHN10 ENDPOINT SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nHN10 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
