"""Multi-seed variance reduction for the frozen V4 LambdaRank selector.

V82 showed that replacing the complete candidate ordering with top-K binary
targets is harmful.  This experiment therefore keeps the original labels,
features, candidates, tree capacity, 67 rounds, and 0.30 endpoint blend fixed.
Only the LightGBM seed changes.  Candidate evidence is built once per well and
the five model scores are combined by raw-score or within-well percentile-rank
averaging.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import types
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_multiseed_rank_ensemble_v83b"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V82_SOURCE = ROOT / "rawdata_lattice_topk_selector_v82_cell.py"
SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
ITERATIONS = 67
BLEND_WEIGHT = 0.30
BASELINE = "single_seed1"
ENSEMBLES = ("raw_mean3", "raw_mean5", "rank_mean3", "rank_mean5")
MEMBERS = tuple(f"member{number}" for number in range(1, len(SEEDS) + 1))
OUTPUTS = (BASELINE,) + MEMBERS[1:] + ENSEMBLES


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _bootstrap_v82():
    if not V82_SOURCE.is_file():
        raise FileNotFoundError(V82_SOURCE)
    source = V82_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V82 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"Bootstrap marker missing from {V82_SOURCE}; upload the updated V82 file")
    module = types.ModuleType("raw_lattice_v82_multiseed_definitions")
    module.__file__ = str(V82_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V82_SOURCE), "exec"), module.__dict__)
    return module


T = _bootstrap_v82()
N = T.N
Q = T.Q
incumbent = T.incumbent
fold_by_well = T.fold_by_well
available = set(T.available)
RANK_RECIPE = T.RECIPES[0]


def _model_params(seed):
    return dict(
        objective="lambdarank",
        metric="ndcg",
        eval_at=(1, 8, 32),
        n_estimators=ITERATIONS,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.2,
        reg_lambda=2.0,
        # Match V82 exactly: sklearn/lightgbm derives all component seeds from
        # random_state.  Setting component seeds explicitly changes seed-1 too.
        random_state=int(seed),
        n_jobs=max(1, min(16, os.cpu_count() or 1)),
        verbosity=-1,
    )


def _fit_models(wells, label):
    # All members see exactly the same rows, labels, groups, and features.
    x, y, _weights, groups, used = T._matrix(wells, RANK_RECIPE, shuffle=False)
    models = []
    for number, seed in enumerate(SEEDS, 1):
        model = lgb.LGBMRanker(**_model_params(seed))
        model.fit(x, y, group=groups)
        model.booster_.save_model(str(MODEL_DIR / f"{label}_seed{number}.txt"))
        models.append(model.booster_)
        print(
            "V83 MODEL",
            {"label": label, "member": number, "seed": seed, "rows": len(x), "wells": len(groups)},
            flush=True,
        )
    del x, y, _weights
    gc.collect()
    return models, used


def _percentile_ranks(values):
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), np.float64)
    if len(values) == 1:
        ranks[0] = 1.0
    else:
        ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    return ranks


def _aggregate_scores(score_matrix):
    score_matrix = np.asarray(score_matrix, np.float64)
    rank_matrix = np.vstack([_percentile_ranks(row) for row in score_matrix])
    output = {
        BASELINE: score_matrix[0],
        "raw_mean3": np.mean(score_matrix[:3], axis=0),
        "raw_mean5": np.mean(score_matrix, axis=0),
        "rank_mean3": np.mean(rank_matrix[:3], axis=0),
        "rank_mean5": np.mean(rank_matrix, axis=0),
    }
    for number in range(1, len(SEEDS)):
        output[f"member{number + 1}"] = score_matrix[number]
    return output


def _evaluate(wells, models, label):
    parts, reports = [], []
    for number, well in enumerate(wells, 1):
        parent = incumbent[incumbent["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
        parent_prediction = parent[N.INCUMBENT_COLUMN].to_numpy(np.float64)
        report = {"well": well, "stage": label, "status": "built", "reason": ""}
        try:
            record = N.V["_load_record"](well)
            paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(record, N.prior)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                features, names = Q._candidate_evidence(
                    record, N.prior, paths, segments, arm_rank, arm_score_z, source_mask
                )
            if list(names) != list(N.feature_names):
                raise RuntimeError("feature schema mismatch")
            eval_rows = record["eval_rows"]
            truth = record["truth"][eval_rows].astype(np.float64)
            distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
            if len(parent) != len(eval_rows):
                raise RuntimeError(f"row count {len(parent)}/{len(eval_rows)}")
            if np.max(np.abs(parent[N.TRUTH_COLUMN].to_numpy(np.float64) - truth)) > 0.01:
                raise RuntimeError("truth alignment")
            if np.max(np.abs(parent["md_since"].to_numpy(np.float64) - distance)) > 1.01:
                raise RuntimeError("distance alignment")

            score_matrix = np.vstack([model.predict(features) for model in models])
            aggregates = _aggregate_scores(score_matrix)
            chosen = {}
            for output_id in OUTPUTS:
                index = int(np.argmax(aggregates[output_id]))
                prediction = N.B["_path_prediction"](
                    record, N.prior, paths[index], segments
                )[eval_rows]
                if not np.isfinite(prediction).all():
                    raise RuntimeError(f"{output_id}: non-finite path")
                parent[f"coarse_{output_id}"] = np.asarray(prediction, np.float32)
                parent[f"blend_{output_id}"] = (
                    parent_prediction + BLEND_WEIGHT * (prediction - parent_prediction)
                ).astype(np.float32)
                chosen[output_id] = index
            report["candidate_count"] = int(len(paths))
            report["chosen"] = chosen
            report["member_unique_choices"] = int(len(set(chosen[name] for name in (BASELINE,) + MEMBERS[1:])))
        except Exception as exc:
            report["status"] = "parent_fallback"
            report["reason"] = repr(exc)
            for output_id in OUTPUTS:
                parent[f"coarse_{output_id}"] = parent_prediction.astype(np.float32)
                parent[f"blend_{output_id}"] = parent_prediction.astype(np.float32)
        parts.append(parent)
        reports.append(report)
        if number % 10 == 0 or number == len(wells):
            fallbacks = sum(row["status"] != "built" for row in reports)
            print(
                f"V83 evaluate {label} {number}/{len(wells)} fallbacks={fallbacks} "
                f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
                flush=True,
            )
    return pd.concat(parts, ignore_index=True), reports


def _summary(frame, stage):
    rows = [{"stage": stage, "model": "parent", **N._metric(frame, frame[N.INCUMBENT_COLUMN], "parent")}]
    for output_id in OUTPUTS:
        rows.append({"stage": stage, "model": output_id, **N._metric(frame, frame[f"coarse_{output_id}"], "coarse")})
        rows.append({"stage": stage, "model": output_id, **N._metric(frame, frame[f"blend_{output_id}"], "blend_w0p30")})
    return rows


def _blend_row(table, model):
    return table[table["model"].eq(model) & table["endpoint"].eq("blend_w0p30")].iloc[0]


def _gains(table, selected):
    base = _blend_row(table, BASELINE)
    candidate = _blend_row(table, selected)
    member_rmse = [float(_blend_row(table, name)["rmse"]) for name in (BASELINE,) + MEMBERS[1:]]
    return {
        "gain": float(base["rmse"] - candidate["rmse"]),
        "far_gain": float(base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"]),
        "near_degradation": float(candidate["near_0_500_rmse"] - base["near_0_500_rmse"]),
        "p95_gain": float(base["p95_well_rmse"] - candidate["p95_well_rmse"]),
        "member_rmse": member_rmse,
        "median_member_rmse": float(np.median(member_rmse)),
        "gain_vs_median_member": float(np.median(member_rmse) - candidate["rmse"]),
        "best_member_rmse": float(np.min(member_rmse)),
    }


# --- V83 EXECUTION ---

train45 = sorted(well for well, fold in fold_by_well.items() if fold in (4, 5) and well in available)
fold3 = sorted(well for well, fold in fold_by_well.items() if fold == 3)
fold2 = sorted(well for well, fold in fold_by_well.items() if fold == 2)
fold1 = sorted(well for well, fold in fold_by_well.items() if fold == 1)
print(
    "V83 CONFIG",
    {
        "train45": len(train45), "fold3": len(fold3), "fold2": len(fold2),
        "fold1": len(fold1), "seeds": SEEDS, "ensembles": ENSEMBLES,
        "candidate_generation_per_well": 1,
    },
    flush=True,
)

# Fold-3 selects only the score aggregation rule.  Labels and model capacity are frozen.
models3, _ = _fit_models(train45, "fold45")
frame3, reports3 = _evaluate(fold3, models3, "v83_fold3_selection")
table3 = pd.DataFrame(_summary(frame3, "fold3_selection"))

# Seed-1 is required to reproduce V82's LambdaRank endpoint exactly.  Refuse to
# interpret the ensemble if the matched baseline drifts for any reason.
v82_summary_path = OOF_ROOT / "rawdata_lattice_topk_selector_v82" / "matched_summary.csv"
baseline_parity = None
if v82_summary_path.is_file():
    v82_summary = pd.read_csv(v82_summary_path)
    expected_rows = v82_summary[
        v82_summary["stage"].eq("fold3_selection")
        & v82_summary["model"].eq("lambdarank")
        & v82_summary["endpoint"].eq("blend_w0p30")
    ]
    if len(expected_rows) != 1:
        raise RuntimeError("Cannot identify the V82 LambdaRank baseline")
    expected_rmse = float(expected_rows.iloc[0]["rmse"])
    observed_rmse = float(_blend_row(table3, BASELINE)["rmse"])
    baseline_parity = {
        "expected_v82_rmse": expected_rmse,
        "observed_seed1_rmse": observed_rmse,
        "absolute_difference": abs(expected_rmse - observed_rmse),
    }
    if baseline_parity["absolute_difference"] > 1e-6:
        raise RuntimeError(f"V82 seed-1 parity failed: {baseline_parity}")
eligible3 = table3[table3["endpoint"].eq("blend_w0p30") & table3["model"].isin(ENSEMBLES)]
selected_id = str(eligible3.sort_values(["rmse", "model"], kind="stable").iloc[0]["model"])
result3 = _gains(table3, selected_id)
selection_pass = bool(
    result3["gain"] >= 0.03
    and result3["far_gain"] >= 0.0
    and result3["near_degradation"] <= 0.03
    and result3["gain_vs_median_member"] >= 0.02
)

rows2, reports2, frame2 = [], [], None
confirmation_pass = False
if selection_pass:
    train345 = sorted(well for well, fold in fold_by_well.items() if fold in (3, 4, 5) and well in available)
    models2, _ = _fit_models(train345, "fold345")
    frame2, reports2 = _evaluate(fold2, models2, "v83_fold2_confirmation")
    rows2 = _summary(frame2, "fold2_confirmation")
    table2 = pd.DataFrame(rows2)
    result2 = _gains(table2, selected_id)
    confirmation_pass = bool(
        result2["gain"] >= 0.03
        and result2["far_gain"] >= 0.0
        and result2["near_degradation"] <= 0.03
        and result2["gain_vs_median_member"] >= 0.01
    )
    frame2.to_parquet(OUT_DIR / "fold2_matched_oof.parquet", index=False)
else:
    result2 = None

rows1, reports1, frame1 = [], [], None
supplement2 = []
final_pass = False
if confirmation_pass:
    missing2 = [well for well in fold2 if well not in available]
    for number, well in enumerate(missing2, 1):
        report = N._build_supplement_panel(well)
        supplement2.append(report)
        if report["status"] != "failed":
            available.add(well)
        if number % 10 == 0 or number == len(missing2):
            print(f"V83 supplement Fold2 {number}/{len(missing2)}", flush=True)
    train2345 = sorted(
        well for well, fold in fold_by_well.items() if fold in (2, 3, 4, 5) and well in available
    )
    models1, _ = _fit_models(train2345, "fold2345")
    frame1, reports1 = _evaluate(fold1, models1, "v83_fold1_final")
    rows1 = _summary(frame1, "fold1_final")
    table1 = pd.DataFrame(rows1)
    result1 = _gains(table1, selected_id)
    final_pass = bool(
        result1["gain"] >= 0.03
        and result1["far_gain"] >= 0.0
        and result1["near_degradation"] <= 0.03
        and result1["gain_vs_median_member"] >= 0.01
    )
    frame1.to_parquet(OUT_DIR / "fold1_matched_oof.parquet", index=False)
else:
    result1 = None

summary = pd.concat([table3, pd.DataFrame(rows2), pd.DataFrame(rows1)], ignore_index=True)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
pd.DataFrame(reports3 + reports2 + reports1).to_json(
    OUT_DIR / "evaluation_report.jsonl", orient="records", lines=True
)
pd.DataFrame(supplement2).to_csv(OUT_DIR / "fold2_supplement_report.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_multiseed_aggregation_selection_Fold2_confirmation_Fold1_final",
    "sample_unit": "one complete physical well",
    "matched_constants": {
        "candidate_bank": "V4 frozen union",
        "feature_count": len(N.feature_names),
        "target": "original V4 complete MSE ordering",
        "iterations": ITERATIONS,
        "blend_weight": BLEND_WEIGHT,
        "seeds": SEEDS,
        "candidate_generation_per_well": 1,
    },
    "selected_aggregation": selected_id,
    "v82_seed1_parity": baseline_parity,
    "fold3_selection": {**result3, "pass": selection_pass},
    "fold2_confirmation": None if result2 is None else {**result2, "pass": confirmation_pass},
    "fold1_final": None if result1 is None else {**result1, "pass": final_pass},
    "fold1_touched": bool(confirmation_pass),
    "strict_pass": bool(selection_pass and confirmation_pass and final_pass),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> rebuild V76 adaptive dose from the frozen multi-seed coarse endpoint"
        if selection_pass and confirmation_pass and final_pass
        else "FAIL -> retain V76; selector variance is not the remaining bottleneck"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV83 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV83 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
