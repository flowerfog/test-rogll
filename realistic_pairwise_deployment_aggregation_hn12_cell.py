"""Decompose HN7's deployment aggregation on HN-unseen Fold 1.

This is a read-only attribution experiment. All HN models excluded stable
Fold 1 from training. It compares deterministic score-level and path-level
aggregation rules with the frozen 0.275 mainline blend. No rule is tuned on
the evaluation labels.
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
OUT_DIR = OOF_ROOT / "realistic_pairwise_deployment_aggregation_hn12"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUT_DIR / "fold1_aggregation_endpoints.parquet"

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
    namespace = {"__name__": "hn12_v4_definitions", "__file__": str(V4_SOURCE)}
    exec(compile(source.split(marker, 1)[0], str(V4_SOURCE), "exec"), namespace)
    return namespace


def _tournament_scores(models, values):
    count = len(values)
    if count == 1:
        return np.ones((len(models), 1), np.float64)
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
    return sums / np.maximum(seen[None, :], 1)


def _rank_scores(single_scores):
    ranks = np.empty_like(single_scores, dtype=np.float64)
    for index, score in enumerate(single_scores):
        order = np.argsort(score, kind="stable")
        ranks[index, order] = np.arange(len(order), dtype=np.float64)
    return np.mean(ranks, axis=0)


def _aggregate_score_rules(single_scores):
    mean_score = np.mean(single_scores, axis=0)
    median_score = np.median(single_scores, axis=0)
    minimum_score = np.min(single_scores, axis=0)
    rank_borda = _rank_scores(single_scores)

    margins = []
    for score in single_scores:
        top = np.partition(score, -2)[-2:] if len(score) > 1 else np.array([0.0, 1.0])
        margins.append(max(float(top[-1] - top[-2]), 1e-6))
    margins = np.asarray(margins, np.float64)
    margin_weighted = np.average(single_scores, axis=0, weights=margins)

    winner_indices = np.argmax(single_scores, axis=1)
    votes = np.bincount(winner_indices, minlength=single_scores.shape[1]).astype(np.float64)
    # Mean score only breaks exact vote ties; it cannot overrule a vote.
    mean_scaled = (mean_score - mean_score.min()) / max(np.ptp(mean_score), 1e-12)
    majority_vote = votes + 1e-3 * mean_scaled
    return {
        "score_mean": mean_score,
        "score_median": median_score,
        "score_borda": rank_borda,
        "score_maximin": minimum_score,
        "score_margin_weighted": margin_weighted,
        "score_majority_vote": majority_vote,
    }


def _medoid_index(predictions, rows):
    values = np.asarray([prediction[rows] for prediction in predictions], np.float64)
    distance = np.zeros((len(values), len(values)), np.float64)
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            rms = float(np.sqrt(np.mean((values[left] - values[right]) ** 2)))
            distance[left, right] = rms
            distance[right, left] = rms
    return int(np.argmin(distance.mean(axis=1))), float(
        distance[np.triu_indices(len(values), 1)].mean()
    )


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


for required in (V4_SOURCE, V66_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

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
        "HN12 CACHE",
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
                f"HN12 prior {number}/{len(prior_wells)} failures={len(prior_failures)}",
                flush=True,
            )
    if len(prior_records) < 307:
        raise RuntimeError(f"Only {len(prior_records)} prior records")
    prior = V["_fit_prior"](prior_records, Q["SEGMENT_LENGTH_FT"])

    blocks = []
    failures = []
    well_diagnostics = []
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
            columns = np.asarray(
                [feature_index[name] for name in selected_features], np.int64
            )
            legal_rank = np.min(np.asarray(arm_rank, np.float64), axis=1)
            menu = np.argsort(legal_rank, kind="stable")[: min(MENU_SIZE, len(paths))]
            values = np.asarray(features[menu][:, columns], np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=12.0, neginf=-12.0)
            single_scores = _tournament_scores(models, values)
            score_rules = _aggregate_score_rules(single_scores)

            selected_global = {
                f"single_hold{fold}": int(menu[int(np.argmax(single_scores[index]))])
                for index, fold in enumerate(MODEL_FOLDS)
            }
            selected_global.update(
                {
                    name: int(menu[int(np.argmax(score))])
                    for name, score in score_rules.items()
                }
            )
            predictions = {
                name: B["_path_prediction"](record, prior, paths[index], segments)
                for name, index in selected_global.items()
            }
            single_predictions = [
                predictions[f"single_hold{fold}"] for fold in MODEL_FOLDS
            ]
            eval_rows = np.asarray(record["eval_rows"], np.int64)
            medoid, winner_disagreement = _medoid_index(single_predictions, eval_rows)
            predictions["path_mean"] = np.mean(single_predictions, axis=0)
            predictions["path_median"] = np.median(single_predictions, axis=0)
            predictions["path_medoid"] = single_predictions[medoid]

            block = pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in eval_rows],
                    "well": well,
                    "row_idx": eval_rows,
                }
            )
            for endpoint, prediction in predictions.items():
                block[endpoint] = np.asarray(prediction[eval_rows], np.float32)
            blocks.append(block)
            winner_local = np.argmax(single_scores, axis=1)
            well_diagnostics.append(
                {
                    "well": well,
                    "unique_model_winners": int(len(np.unique(winner_local))),
                    "winner_path_pairwise_rms": winner_disagreement,
                    "medoid_model_fold": int(MODEL_FOLDS[medoid]),
                }
            )
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 20 == 0 or number == len(requested_wells):
            print(
                f"HN12 inference {number}/{len(requested_wells)} "
                f"success={len(blocks)} failures={len(failures)}",
                flush=True,
            )
    pd.DataFrame(prior_failures).to_csv(
        OUT_DIR / "prior_record_failures.csv", index=False
    )
    pd.DataFrame(failures).to_csv(OUT_DIR / "inference_failures.csv", index=False)
    pd.DataFrame(well_diagnostics).to_csv(
        OUT_DIR / "well_aggregation_diagnostics.csv", index=False
    )
    if len(blocks) < 145:
        raise RuntimeError(
            f"Only {len(blocks)}/{len(requested_wells)} Fold1 wells succeeded: "
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
for frame in (v66, endpoints):
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
aligned = v66.merge(
    endpoints,
    on=["id", "well", "row_idx"],
    how="inner",
    validate="one_to_one",
)
if aligned["well"].nunique() < 145:
    raise RuntimeError(f"Only {aligned['well'].nunique()} Fold1 wells align to V66")
if any(stable_fold_by_well[well] != HN_FOLD for well in aligned["well"].unique()):
    raise RuntimeError("Aligned endpoint includes an HN training fold")

base = aligned["incumbent_6p377_replay"].to_numpy(np.float64)
base_metrics = _metrics(aligned, base)
endpoint_columns = [
    column
    for column in endpoints.columns
    if column not in {"id", "well", "row_idx"}
]
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
summary = pd.DataFrame(rows).sort_values("blend_rmse", kind="stable")
summary.to_csv(OUT_DIR / "aggregation_summary.csv", index=False)

current = summary.loc[summary["endpoint"].eq("score_mean")].iloc[0]
best = summary.iloc[0]
checks = {
    "unseen_fold_wells_ge_145": bool(aligned["well"].nunique() >= 145),
    "current_score_mean_matches_hn10_gain_ge_0p25": bool(current["gain"] >= 0.25),
    "some_rule_beats_score_mean_by_ge_0p05": bool(
        current["blend_rmse"] - best["blend_rmse"] >= 0.05
    ),
    "best_rule_gain_ge_0p35": bool(best["gain"] >= 0.35),
    "best_rule_far_gain_ge_0p35": bool(best["far_gain"] >= 0.35),
    "best_rule_p95_change_le_0p20": bool(best["p95_change"] <= 0.20),
}
verdict = {
    "experiment": "realistic_pairwise_deployment_aggregation_hn12",
    "stage": "exact_hn7_aggregation_decomposition_on_hn_unseen_fold1",
    "selection_or_tuning_count": 0,
    "validation_status": "retrospective attribution; Fold1 was already read by HN10",
    "sources": {"v66": V66_PATH, "hn3_models": HN3_DIR},
    "protocol": {
        "hn_holdout_fold": HN_FOLD,
        "models": list(MODEL_FOLDS),
        "all_models_unseen_on_every_evaluation_well": True,
        "frozen_weight": FROZEN_WEIGHT,
        "menu_size": MENU_SIZE,
        "truth_used_for_aggregation": False,
    },
    "coverage": {
        "requested_wells": len(requested_wells),
        "successful_wells": int(endpoints["well"].nunique()),
        "aligned_wells": int(aligned["well"].nunique()),
        "aligned_rows": int(len(aligned)),
    },
    "base": base_metrics,
    "current_rule": current.to_dict(),
    "best_diagnostic_rule": best.to_dict(),
    "rules": summary.to_dict("records"),
    "checks": checks,
    "mechanism_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze the winning aggregation logic, then require one new LB confirmation"
        if all(checks.values())
        else "FAIL -> aggregation is not the HN transfer bottleneck; close HN deployment"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8"
)
print("\nHN12 AGGREGATION SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nHN12 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
