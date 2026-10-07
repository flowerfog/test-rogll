"""V86: exact-V70 multi-seed disagreement as a V76 tail-dose guard.

Unlike V85, every ranker in this experiment uses the exact V70 cross-fit
training rows, labels, groups, features, candidates, and hyperparameters.
Seed 1 is loaded from V70 and must reproduce the persisted V70 coarse path.
Only seeds 2-5 are fitted.  Disagreement never replaces the selected path; it
may only shrink the frozen V76 correction after 1000 ft.
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
EXPERIMENT = "rawdata_lattice_v70_matched_disagreement_guard_v86"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V70_SOURCE = ROOT / "rawdata_lattice_coarse_all773_crossfit_v70_cell.py"
V85_SOURCE = ROOT / "rawdata_lattice_rank_disagreement_dose_guard_v85_cell.py"
V70_DIR = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70"
V70_PATH = V70_DIR / "all773_proxy_crossfit_oof.parquet"
V76_WEIGHTS = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
ITERATIONS = 67
TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
MEMBER_COLUMNS = (
    "coarse_single_seed1", "coarse_member2", "coarse_member3",
    "coarse_member4", "coarse_member5",
)


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


def _bootstrap_v70():
    if not V70_SOURCE.is_file():
        raise FileNotFoundError(V70_SOURCE)
    source = V70_SOURCE.read_text(encoding="utf-8")
    marker = "parts = []\nreports = []\nmodel_manifest = []\nfor heldout_fold in HONEST_FOLDS:"
    if marker not in source:
        raise RuntimeError(f"Cannot identify V70 execution boundary in {V70_SOURCE}")
    module = types.ModuleType("raw_lattice_v70_v86_definitions")
    module.__file__ = str(V70_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V70_SOURCE), "exec"), module.__dict__)
    return module


def _bootstrap_v85_helpers():
    if not V85_SOURCE.is_file():
        raise FileNotFoundError(V85_SOURCE)
    source = V85_SOURCE.read_text(encoding="utf-8")
    marker = "for required in (V83_SOURCE, V70_PATH, V76_WEIGHTS, V84_DIR / \"fold2_matched_oof.parquet\"):"
    if marker not in source:
        raise RuntimeError(f"Cannot identify V85 execution boundary in {V85_SOURCE}")
    module = types.ModuleType("raw_lattice_v85_v86_helpers")
    module.__file__ = str(V85_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V85_SOURCE), "exec"), module.__dict__)
    return module


def _model_params(seed):
    return dict(
        objective="lambdarank", metric="ndcg", eval_at=(1, 8, 32),
        n_estimators=ITERATIONS, learning_rate=0.025, num_leaves=31,
        max_depth=-1, min_child_samples=80, subsample=0.85,
        colsample_bytree=0.80, reg_alpha=0.2, reg_lambda=2.0,
        random_state=int(seed), n_jobs=max(1, min(16, os.cpu_count() or 1)),
        verbosity=-1,
    )


def _fit_or_load_models(heldout_fold):
    seed1_path = V70_DIR / "models" / f"coarse_ranker_leave_fold{heldout_fold}_out.txt"
    if not seed1_path.is_file():
        raise FileNotFoundError(seed1_path)
    models = [lgb.Booster(model_file=str(seed1_path))]
    missing = [number for number in range(2, 6) if not (MODEL_DIR / f"fold{heldout_fold}_seed{number}.txt").is_file()]
    if not missing:
        return models + [
            lgb.Booster(model_file=str(MODEL_DIR / f"fold{heldout_fold}_seed{number}.txt"))
            for number in range(2, 6)
        ]

    train_wells = sorted(
        well for well, fold in R.fold_by_well.items()
        if fold != heldout_fold and R.sample_paths[well] is not None
    )
    x_parts, y_parts, groups = [], [], []
    for number, well in enumerate(train_wells, 1):
        with np.load(R.sample_paths[well], allow_pickle=False) as data:
            x_parts.append(data["x"].astype(np.float32, copy=False))
            y_parts.append(data["y"].astype(np.int32, copy=False))
            groups.append(int(len(data["y"])))
        if number % 100 == 0 or number == len(train_wells):
            print(f"V86 Fold{heldout_fold} load panels {number}/{len(train_wells)}", flush=True)
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    del x_parts, y_parts
    for number, seed in enumerate(SEEDS[1:], 2):
        path = MODEL_DIR / f"fold{heldout_fold}_seed{number}.txt"
        if path.is_file():
            booster = lgb.Booster(model_file=str(path))
        else:
            model = lgb.LGBMRanker(**_model_params(seed))
            model.fit(x, y, group=groups)
            model.booster_.save_model(str(path))
            booster = model.booster_
        models.append(booster)
        print("V86 MODEL", {"fold": heldout_fold, "member": number, "seed": seed}, flush=True)
    del x, y
    gc.collect()
    return models


def _percentile_ranks(values):
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def _evaluate_fold(fold, models):
    wells = sorted(well for well, value in R.fold_by_well.items() if value == fold)
    parts, reports = [], []
    for number, well in enumerate(wells, 1):
        parent = R.incumbent[R.incumbent["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
        parent_prediction = parent[PARENT].to_numpy(np.float64)
        report = {"fold": int(fold), "well": str(well), "status": "built", "reason": ""}
        try:
            record = R.V["_load_record"](well)
            paths, source_mask, arm_rank, arm_score_z, segments, _ = R.Q._candidate_union(record, R.prior)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                features, names = R.Q._candidate_evidence(
                    record, R.prior, paths, segments, arm_rank, arm_score_z, source_mask
                )
            if list(names) != list(R.feature_names):
                raise RuntimeError("feature schema mismatch")
            eval_rows = record["eval_rows"]
            truth = record["truth"][eval_rows].astype(np.float64)
            distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
            if len(parent) != len(eval_rows):
                raise RuntimeError(f"row count {len(parent)}/{len(eval_rows)}")
            if np.max(np.abs(parent[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
                raise RuntimeError("truth alignment")
            if np.max(np.abs(parent["md_since"].to_numpy(np.float64) - distance)) > 1.01:
                raise RuntimeError("distance alignment")
            chosen = []
            score_vectors = []
            for member, model in enumerate(models, 1):
                scores = np.asarray(model.predict(features), np.float64)
                score_vectors.append(scores)
                index = int(np.argmax(scores))
                prediction = R.B["_path_prediction"](record, R.prior, paths[index], segments)[eval_rows]
                column = "coarse_single_seed1" if member == 1 else f"coarse_member{member}"
                parent[column] = np.asarray(prediction, np.float32)
                chosen.append(index)
            score_ranks = np.vstack([_percentile_ranks(scores) for scores in score_vectors[:3]])
            consensus_index = int(np.argmax(np.mean(score_ranks, axis=0)))
            parent["coarse_rank_mean3"] = np.asarray(
                R.B["_path_prediction"](record, R.prior, paths[consensus_index], segments)[eval_rows],
                np.float32,
            )
            report["candidate_count"] = int(len(paths))
            report["member_unique_choices"] = int(len(set(chosen)))
        except Exception as exc:
            report["status"] = "parent_fallback"
            report["reason"] = repr(exc)
            for column in MEMBER_COLUMNS + ("coarse_rank_mean3",):
                parent[column] = parent_prediction.astype(np.float32)
        parts.append(parent)
        reports.append(report)
        if number % 10 == 0 or number == len(wells):
            fallback = sum(item["status"] != "built" for item in reports)
            print(f"V86 Fold{fold} evaluate {number}/{len(wells)} fallbacks={fallback}", flush=True)
    return pd.concat(parts, ignore_index=True), reports


for required in (V70_SOURCE, V85_SOURCE, V70_PATH, V76_WEIGHTS):
    if not required.is_file():
        raise FileNotFoundError(required)

R = _bootstrap_v70()
H = _bootstrap_v85_helpers()

models3 = _fit_or_load_models(3)
multi3, reports3 = _evaluate_fold(3, models3)
frame3, lineage3 = H._base_frame(3, multi3)
if lineage3["v70_vs_v83_seed1_max_abs_gap"] > 0.01:
    raise RuntimeError(f"Exact V70 seed-1 parity failed on Fold3: {lineage3}")
risk3 = H._risk_features(frame3)
scan3, baseline3 = H._scan(frame3, risk3)
scan3["gain"] = baseline3["rmse"] - scan3["rmse"]
scan3["far_gain"] = baseline3["far_gt2000_rmse"] - scan3["far_gt2000_rmse"]
scan3["near_degradation"] = scan3["near_0_500_rmse"] - baseline3["near_0_500_rmse"]
scan3["p95_gain"] = baseline3["p95_well_rmse"] - scan3["p95_well_rmse"]
eligible3 = scan3[
    scan3["feature"].ne("none") & scan3["gain"].ge(0.03)
    & scan3["far_gain"].ge(0.03) & scan3["near_degradation"].le(1e-10)
].copy()
scan3.sort_values("rmse", kind="stable").to_csv(OUT_DIR / "fold3_scan.csv", index=False)
risk3.to_csv(OUT_DIR / "fold3_disagreement_features.csv", index=False)

selection_pass = not eligible3.empty
selected = None
result2 = None
confirmation_pass = False
summary_rows = [{"stage": "fold3_selection", **baseline3}]
reports2 = []
lineage2 = None

if selection_pass:
    selected = eligible3.sort_values(["rmse", "feature", "quantile", "retain"], kind="stable").iloc[0].to_dict()
    map3 = dict(zip(risk3["well"].astype(str), risk3[selected["feature"]]))
    selected3 = H._metric(
        frame3,
        H._guard_prediction(frame3, map3, selected["feature"], selected["threshold"], selected["retain"]),
        "selected_guard",
    )
    summary_rows.append({"stage": "fold3_selection", **selected3})

    models2 = _fit_or_load_models(2)
    multi2, reports2 = _evaluate_fold(2, models2)
    frame2, lineage2 = H._base_frame(2, multi2)
    if lineage2["v70_vs_v83_seed1_max_abs_gap"] > 0.01:
        raise RuntimeError(f"Exact V70 seed-1 parity failed on Fold2: {lineage2}")
    risk2 = H._risk_features(frame2)
    map2 = dict(zip(risk2["well"].astype(str), risk2[selected["feature"]]))
    prediction2 = H._guard_prediction(
        frame2, map2, selected["feature"], selected["threshold"], selected["retain"]
    )
    base2 = H._metric(frame2, frame2["v76"], "v76")
    candidate2 = H._metric(frame2, prediction2, "selected_guard")
    shuffled = risk2[selected["feature"]].to_numpy(np.float64).copy()
    np.random.default_rng(20260730).shuffle(shuffled)
    shuffle2 = H._metric(
        frame2,
        H._guard_prediction(
            frame2, dict(zip(risk2["well"].astype(str), shuffled)),
            selected["feature"], selected["threshold"], selected["retain"],
        ),
        "matched_risk_shuffle",
    )
    result2 = {
        "gain": base2["rmse"] - candidate2["rmse"],
        "far_gain": base2["far_gt2000_rmse"] - candidate2["far_gt2000_rmse"],
        "near_degradation": candidate2["near_0_500_rmse"] - base2["near_0_500_rmse"],
        "p95_gain": base2["p95_well_rmse"] - candidate2["p95_well_rmse"],
        "shuffle_gain": base2["rmse"] - shuffle2["rmse"],
    }
    confirmation_pass = bool(
        result2["gain"] >= 0.03 and result2["far_gain"] >= 0.03
        and result2["near_degradation"] <= 1e-10 and result2["p95_gain"] >= 0.0
        and result2["shuffle_gain"] <= 0.30 * result2["gain"]
    )
    summary_rows.extend([
        {"stage": "fold2_confirmation", **base2},
        {"stage": "fold2_confirmation", **candidate2},
        {"stage": "fold2_confirmation", **shuffle2},
    ])
    risk2.to_csv(OUT_DIR / "fold2_disagreement_features.csv", index=False)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
pd.DataFrame(reports3 + reports2).to_json(OUT_DIR / "evaluation_report.jsonl", orient="records", lines=True)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "exact_V70_multiseed_disagreement_Fold3_selection_Fold2_confirmation",
    "matched_constants": {
        "seed1": "exact persisted V70 fold model",
        "training_rows_labels_groups_features_candidates": "identical within heldout fold",
        "iterations": ITERATIONS,
        "seeds": SEEDS,
        "path_consumed": "V70 seed1 only",
        "base_dose": "V76 fixed Hist/Ridge ensemble",
    },
    "lineage_parity": {"fold3": lineage3, "fold2": lineage2},
    "selection_search_count": int(len(scan3) - 1),
    "selected_recipe": selected,
    "fold3_selection_pass": selection_pass,
    "fold2_confirmation": result2,
    "fold1_touched": False,
    "strict_pass": bool(selection_pass and confirmation_pass),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze rule and run untouched Fold1 final"
        if selection_pass and confirmation_pass
        else "FAIL -> close ranker-disagreement dosing; retain V76"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV86 FOLD3 TOP 20")
print(scan3.sort_values("rmse", kind="stable").head(20).to_string(index=False), flush=True)
print("\nV86 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV86 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
