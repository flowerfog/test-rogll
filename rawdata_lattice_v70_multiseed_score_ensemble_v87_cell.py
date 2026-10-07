"""V87: exact-V70 multi-seed score aggregation without uncertainty gating.

Five rankers share V70's rows, labels, groups, features, candidates, and
hyperparameters. Seed 1 is the persisted V70 model. Fold 3 chooses among a
small preregistered set of aggregation rules; Fold 2 is confirmation only.
The frozen V76 per-well dose is retained and Fold 1 remains untouched.
"""

from __future__ import annotations

import json
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_v70_multiseed_score_ensemble_v87"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V86_SOURCE = ROOT / "rawdata_lattice_v70_matched_disagreement_guard_v86_cell.py"
V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V76_WEIGHTS = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
RULES = ("borda_mean3", "borda_mean5", "borda_median5", "zmean5", "path_mean5")


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


def _bootstrap_v86():
    if not V86_SOURCE.is_file():
        raise FileNotFoundError(V86_SOURCE)
    source = V86_SOURCE.read_text(encoding="utf-8")
    marker = "for required in (V70_SOURCE, V85_SOURCE, V70_PATH, V76_WEIGHTS):"
    if marker not in source:
        raise RuntimeError(f"Cannot identify V86 execution boundary in {V86_SOURCE}")
    module = types.ModuleType("raw_lattice_v86_v87_definitions")
    module.__file__ = str(V86_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V86_SOURCE), "exec"), module.__dict__)
    module.R = module._bootstrap_v70()
    module.H = module._bootstrap_v85_helpers()
    return module


def _ranks(values):
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="stable")
    output = np.empty(len(values), dtype=np.float64)
    output[order] = np.arange(len(values), dtype=np.float64)
    return output / max(len(values) - 1, 1)


def _zscore(values):
    values = np.asarray(values, np.float64)
    scale = float(np.std(values))
    return (values - float(np.mean(values))) / max(scale, 1e-8)


def _base(fold):
    frame = pd.read_parquet(
        V70_PATH,
        columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
        filters=[("fold", "=", int(fold))],
    ).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    weights = pd.read_parquet(V76_WEIGHTS)
    weights = weights[weights["fold"].eq(fold)][["well", "ensemble_weight"]]
    weight_map = dict(zip(weights["well"].astype(str), weights["ensemble_weight"]))
    frame["dose"] = frame["well"].astype(str).map(weight_map)
    if frame["dose"].isna().any():
        raise RuntimeError(f"Fold{fold}: missing V76 dose")
    frame["v76"] = frame[PARENT].to_numpy(np.float64) + frame["dose"].to_numpy(np.float64) * (
        frame[COARSE].to_numpy(np.float64) - frame[PARENT].to_numpy(np.float64)
    )
    return frame


def _evaluate(fold, models):
    base = _base(fold)
    wells = sorted(base["well"].astype(str).unique())
    parts, reports = [], []
    for number, well in enumerate(wells, 1):
        parent = base[base["well"].astype(str).eq(well)].copy()
        report = {"fold": int(fold), "well": well, "status": "built", "reason": ""}
        try:
            record = M.R.V["_load_record"](well)
            paths, source_mask, arm_rank, arm_score_z, segments, _ = M.R.Q._candidate_union(record, M.R.prior)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                features, names = M.R.Q._candidate_evidence(
                    record, M.R.prior, paths, segments, arm_rank, arm_score_z, source_mask
                )
            if list(names) != list(M.R.feature_names):
                raise RuntimeError("feature schema mismatch")
            eval_rows = record["eval_rows"]
            truth = record["truth"][eval_rows].astype(np.float64)
            if len(parent) != len(eval_rows):
                raise RuntimeError(f"row count {len(parent)}/{len(eval_rows)}")
            if np.max(np.abs(parent[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
                raise RuntimeError("truth alignment")

            score = np.vstack([np.asarray(model.predict(features), np.float64) for model in models])
            selected_indices = np.argmax(score, axis=1)
            selected_paths = np.vstack([
                M.R.B["_path_prediction"](record, M.R.prior, paths[index], segments)[eval_rows]
                for index in selected_indices
            ])
            aggregates = {
                "borda_mean3": np.mean(np.vstack([_ranks(row) for row in score[:3]]), axis=0),
                "borda_mean5": np.mean(np.vstack([_ranks(row) for row in score]), axis=0),
                "borda_median5": np.median(np.vstack([_ranks(row) for row in score]), axis=0),
                "zmean5": np.mean(np.vstack([_zscore(row) for row in score]), axis=0),
            }
            for rule, aggregate in aggregates.items():
                index = int(np.argmax(aggregate))
                parent[f"coarse_{rule}"] = np.asarray(
                    M.R.B["_path_prediction"](record, M.R.prior, paths[index], segments)[eval_rows],
                    np.float32,
                )
            parent["coarse_path_mean5"] = np.mean(selected_paths, axis=0).astype(np.float32)
            seed1 = selected_paths[0]
            parity = seed1 - parent[COARSE].to_numpy(np.float64)
            report.update({
                "candidate_count": int(len(paths)),
                "unique_member_choices": int(len(set(selected_indices.tolist()))),
                "seed1_rmse_gap": float(np.sqrt(np.mean(parity ** 2))),
                "seed1_max_abs_gap": float(np.max(np.abs(parity))),
            })
            if report["seed1_max_abs_gap"] > 0.01:
                raise RuntimeError(f"seed1 parity {report['seed1_max_abs_gap']}")
        except Exception as exc:
            report["status"] = "seed1_fallback"
            report["reason"] = repr(exc)
            for rule in RULES:
                parent[f"coarse_{rule}"] = parent[COARSE].to_numpy(np.float32)
        parts.append(parent)
        reports.append(report)
        if number % 10 == 0 or number == len(wells):
            fallback = sum(item["status"] != "built" for item in reports)
            print(f"V87 Fold{fold} evaluate {number}/{len(wells)} fallbacks={fallback}", flush=True)
    frame = pd.concat(parts, ignore_index=True).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    return frame, reports


def _pred(frame, rule):
    return frame[PARENT].to_numpy(np.float64) + frame["dose"].to_numpy(np.float64) * (
        frame[f"coarse_{rule}"].to_numpy(np.float64) - frame[PARENT].to_numpy(np.float64)
    )


def _summaries(frame, stage):
    rows = [{"stage": stage, "rule": "seed1", **M.H._metric(frame, frame["v76"], "v76_seed1")}]
    baseline = rows[0]
    for rule in RULES:
        metric = M.H._metric(frame, _pred(frame, rule), rule)
        rows.append({
            "stage": stage,
            "rule": rule,
            **metric,
            "gain": baseline["rmse"] - metric["rmse"],
            "far_gain": baseline["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
            "near_degradation": metric["near_0_500_rmse"] - baseline["near_0_500_rmse"],
            "p95_gain": baseline["p95_well_rmse"] - metric["p95_well_rmse"],
        })
    return pd.DataFrame(rows)


for required in (V86_SOURCE, V70_PATH, V76_WEIGHTS):
    if not required.is_file():
        raise FileNotFoundError(required)

M = _bootstrap_v86()
models3 = M._fit_or_load_models(3)
frame3, reports3 = _evaluate(3, models3)
summary3 = _summaries(frame3, "fold3_selection")
eligible3 = summary3[
    summary3["rule"].ne("seed1")
    & summary3["gain"].ge(0.05)
    & summary3["far_gain"].ge(0.0)
    & summary3["near_degradation"].le(0.02)
    & summary3["p95_gain"].ge(0.0)
].copy()

selected = None
summary2 = pd.DataFrame()
reports2 = []
confirmation = None
strict_pass = False
if not eligible3.empty:
    selected = eligible3.sort_values(["rmse", "rule"], kind="stable").iloc[0].to_dict()
    models2 = M._fit_or_load_models(2)
    frame2, reports2 = _evaluate(2, models2)
    all2 = _summaries(frame2, "fold2_confirmation")
    summary2 = all2[all2["rule"].isin(("seed1", selected["rule"]))].copy()
    base2 = summary2[summary2["rule"].eq("seed1")].iloc[0]
    candidate2 = summary2[summary2["rule"].eq(selected["rule"])].iloc[0]
    confirmation = {
        "rule": selected["rule"],
        "gain": float(base2["rmse"] - candidate2["rmse"]),
        "far_gain": float(base2["far_gt2000_rmse"] - candidate2["far_gt2000_rmse"]),
        "near_degradation": float(candidate2["near_0_500_rmse"] - base2["near_0_500_rmse"]),
        "p95_gain": float(base2["p95_well_rmse"] - candidate2["p95_well_rmse"]),
    }
    strict_pass = bool(
        confirmation["gain"] >= 0.05
        and confirmation["far_gain"] >= 0.0
        and confirmation["near_degradation"] <= 0.02
        and confirmation["p95_gain"] >= 0.0
    )

summary = pd.concat([summary3, summary2], ignore_index=True)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
pd.DataFrame(reports3 + reports2).to_json(OUT_DIR / "evaluation_report.jsonl", orient="records", lines=True)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "exact_V70_multiseed_score_aggregation_Fold3_selection_Fold2_confirmation",
    "candidate_rules": list(RULES),
    "selection_search_count": len(RULES),
    "selected_rule": None if selected is None else selected["rule"],
    "fold3_selection": None if selected is None else {
        key: selected[key] for key in ("rule", "gain", "far_gain", "near_degradation", "p95_gain")
    },
    "fold2_confirmation": confirmation,
    "fold1_touched": False,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze score aggregation and run untouched Fold1"
        if strict_pass else "FAIL -> close multi-seed ranker branch; retain V76"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV87 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV87 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
