"""V106: cross-fitted blend-aware reranking inside the base top-256.

The first-stage standalone ranker retrieves a trustworthy shortlist. Training
shortlists are cross-fitted: Fold4 candidates are scored by a base model fit on
Fold5 and vice versa. Two matched second-stage rankers use the same 211 inputs
(131 base evidence + 78 parent-relative + base score/rank); only their labels
differ between standalone and blend-aware MSE. Fold3 is the sole evaluation
fold, split into 24 discovery and 130 confirmation wells. Fold2/Fold1 remain
untouched.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_blendaware_shortlist_reranker_v106"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

V81_SOURCE = ROOT / "rawdata_lattice_structural_risk_ranker_v81b_cell.py"
V103_PANELS = OOF_ROOT / "rawdata_lattice_blendaware_ranker_v103" / "matched_panels"
V104_RELATIVE = OOF_ROOT / "rawdata_lattice_parentrelative_ranker_v104" / "relative_features"
V104_BASE45 = OOF_ROOT / "rawdata_lattice_parentrelative_ranker_v104" / "models" / "fold45_base.txt"
V105_VERDICT = OOF_ROOT / "rawdata_lattice_blendaware_shortlist_oracle_v105" / "verdict.json"
V76_WEIGHTS = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
DISCOVERY_WELLS = 24
TOPK = 256
ITERATIONS = 67
RANDOM_SEED = 20260721
PARENT = "incumbent_6p377_replay"
TRUTH = "tvt_true"


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


def _bootstrap_v81():
    source = V81_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V81B EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V81b marker missing from {V81_SOURCE}")
    module = types.ModuleType("raw_lattice_v81_shortlist_reranker_definitions")
    module.__file__ = str(V81_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V81_SOURCE), "exec"), module.__dict__)
    return module


for required in (V81_SOURCE, V103_PANELS, V104_RELATIVE, V104_BASE45, V105_VERDICT, V76_WEIGHTS):
    if not required.exists():
        raise FileNotFoundError(required)
v105 = json.loads(V105_VERDICT.read_text(encoding="utf-8"))
if not v105.get("strict_pass") or int(v105.get("selected_topk", -1)) != TOPK:
    raise RuntimeError(f"Unexpected V105 verdict: pass={v105.get('strict_pass')} topk={v105.get('selected_topk')}")

N = _bootstrap_v81()
Q, V, B, prior = N.Q, N.V, N.B, N.prior
incumbent = N.incumbent
fold_by_well = N.fold_by_well
base_feature_names = list(N.feature_names)
base45 = lgb.Booster(model_file=str(V104_BASE45))

weights = pd.read_parquet(V76_WEIGHTS)
weights["well"] = weights["well"].astype(str)
weight_map = dict(zip(weights["well"], weights["ensemble_weight"].astype(float)))


def _load_panel(well):
    with np.load(V103_PANELS / f"{well}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_relative(well):
    with np.load(V104_RELATIVE / f"{well}.npz", allow_pickle=False) as data:
        values = data["features"].astype(np.float32)
        names = json.loads(str(data["feature_names_json"].item()))
    return values, names


sample_well = next(
    well for well, fold in fold_by_well.items()
    if fold in (3, 4, 5)
    and (V103_PANELS / f"{well}.npz").is_file()
    and (V104_RELATIVE / f"{well}.npz").is_file()
)
sample_panel = _load_panel(sample_well)
sample_relative, relative_feature_names = _load_relative(sample_well)
if len(sample_panel["features"]) != len(sample_relative):
    raise RuntimeError("Sample base/relative alignment failed")


def _base_features(panel):
    values = panel["features"].astype(np.float32, copy=False)
    if values.shape[1] != len(base_feature_names):
        raise RuntimeError("Base feature schema mismatch")
    return values


def _all_features(well, panel):
    base = _base_features(panel)
    relative, names = _load_relative(well)
    if names != relative_feature_names or len(relative) != len(base):
        raise RuntimeError(f"{well}: relative feature alignment")
    return np.concatenate([base, relative], axis=1)


def _score_prior(scores):
    scores = np.asarray(scores, np.float64)
    centre = float(np.median(scores))
    scale = max(float(1.4826 * np.median(np.abs(scores - centre))), 1e-6)
    zscore = np.clip((scores - centre) / scale, -12.0, 12.0)
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(scores), np.float64)
    rank[order] = np.arange(len(scores), dtype=np.float64) / max(len(scores) - 1, 1)
    return np.stack([zscore, rank], axis=1).astype(np.float32)


def _fit_ranker(x, y, groups, label):
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", eval_at=(1, 8, 32),
        n_estimators=ITERATIONS, learning_rate=0.025,
        num_leaves=31, max_depth=-1, min_child_samples=80,
        subsample=0.85, colsample_bytree=0.80,
        reg_alpha=0.2, reg_lambda=2.0, random_state=RANDOM_SEED,
        n_jobs=max(1, min(16, os.cpu_count() or 1)), verbosity=-1,
    )
    model.fit(x, y, group=groups)
    model.booster_.save_model(str(MODEL_DIR / f"{label}.txt"))
    print("V106 MODEL", {"label": label, "rows": len(x), "features": x.shape[1], "wells": len(groups)}, flush=True)
    return model.booster_


def _base_matrix(wells):
    x_parts, y_parts, groups = [], [], []
    for well in wells:
        panel = _load_panel(well)
        indices = Q._training_indices(panel, well)
        x_parts.append(_base_features(panel)[indices])
        y_parts.append(Q._rank_labels(panel["mse"].astype(np.float64)[indices]))
        groups.append(int(len(indices)))
    return np.concatenate(x_parts), np.concatenate(y_parts), groups


fold4 = sorted(w for w, f in fold_by_well.items() if f == 4 and (V103_PANELS / f"{w}.npz").is_file())
fold5 = sorted(w for w, f in fold_by_well.items() if f == 5 and (V103_PANELS / f"{w}.npz").is_file())
fold3 = sorted(w for w, f in fold_by_well.items() if f == 3 and (V103_PANELS / f"{w}.npz").is_file())
if len(fold4) < 153 or len(fold5) < 153 or len(fold3) < 153:
    raise RuntimeError(f"Coverage fold4={len(fold4)} fold5={len(fold5)} fold3={len(fold3)}")

print("V106 CONFIG", {
    "fold4": len(fold4), "fold5": len(fold5), "fold3": len(fold3),
    "topk": TOPK, "base_features": len(base_feature_names),
    "relative_features": len(relative_feature_names), "reranker_features": len(base_feature_names) + len(relative_feature_names) + 2,
}, flush=True)

x5, y5, g5 = _base_matrix(fold5)
base5 = _fit_ranker(x5, y5, g5, "base_fold5_for_fold4")
del x5, y5
gc.collect()
x4, y4, g4 = _base_matrix(fold4)
base4 = _fit_ranker(x4, y4, g4, "base_fold4_for_fold5")
del x4, y4
gc.collect()


def _rerank_matrix(wells, base_model, target):
    x_parts, y_parts, groups = [], [], []
    for number, well in enumerate(wells, 1):
        panel = _load_panel(well)
        base = _base_features(panel)
        scores = base_model.predict(base)
        retained = np.argsort(-scores, kind="stable")[: min(TOPK, len(scores))]
        features = np.concatenate([_all_features(well, panel), _score_prior(scores)], axis=1)
        labels = panel["blend_mse"] if target == "blendaware" else panel["mse"]
        x_parts.append(features[retained])
        y_parts.append(Q._rank_labels(labels.astype(np.float64)[retained]))
        groups.append(int(len(retained)))
        if number % 100 == 0 or number == len(wells):
            print(f"V106 shortlist matrix {target} {number}/{len(wells)}", flush=True)
    return np.concatenate(x_parts), np.concatenate(y_parts), groups


shortlist_sources = ((fold4, base5), (fold5, base4))
rerank_models = {}
for target in ("standalone", "blendaware"):
    x_parts, y_parts, groups = [], [], []
    for wells_part, source_model in shortlist_sources:
        x, y, local_groups = _rerank_matrix(wells_part, source_model, target)
        x_parts.append(x)
        y_parts.append(y)
        groups.extend(local_groups)
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    rerank_models[target] = _fit_ranker(x, y, groups, f"reranker_{target}")
    del x, y, x_parts, y_parts
    gc.collect()


parts, reports = [], []
for number, well in enumerate(fold3, 1):
    local = incumbent[incumbent["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
    parent = local[PARENT].to_numpy(np.float64)
    panel = _load_panel(well)
    record = V["_load_record"](well)
    paths = panel["paths"].astype(np.int16, copy=False)
    segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
    base = _base_features(panel)
    base_scores = base45.predict(base)
    retained = np.argsort(-base_scores, kind="stable")[: min(TOPK, len(base_scores))]
    rerank_features = np.concatenate([_all_features(well, panel), _score_prior(base_scores)], axis=1)
    chosen = {"base": int(np.argmax(base_scores))}
    for target, model in rerank_models.items():
        local_scores = model.predict(rerank_features[retained])
        chosen[f"rerank_{target}"] = int(retained[np.argmax(local_scores)])
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    if len(local) != len(eval_rows) or np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
        raise RuntimeError(f"{well}: Fold3 alignment")
    if well not in weight_map:
        raise RuntimeError(f"{well}: missing V76 dose")
    dose = float(weight_map[well])
    for endpoint, index in chosen.items():
        prediction = B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
        local[endpoint] = (parent + dose * (np.asarray(prediction, np.float64) - parent)).astype(np.float32)
    parts.append(local)
    reports.append({
        "well": well, "dose": dose, "candidates": len(paths), "retained": len(retained),
        **{f"index_{key}": value for key, value in chosen.items()},
    })
    if number % 10 == 0 or number == len(fold3):
        print(f"V106 Fold3 {number}/{len(fold3)} elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True)

frame = pd.concat(parts, ignore_index=True)
pd.DataFrame(reports).to_csv(OUT_DIR / "fold3_choices.csv", index=False)
discovery = set(fold3[:DISCOVERY_WELLS])
summary_rows = []
for stage, selected_wells in {
    "fold3_discovery": discovery,
    "fold3_confirmation": set(fold3).difference(discovery),
}.items():
    selected = frame[frame["well"].isin(selected_wells)]
    for endpoint in ("base", "rerank_standalone", "rerank_blendaware"):
        row = N._metric(selected, selected[endpoint], endpoint)
        row["stage"] = stage
        summary_rows.append(row)
summary = pd.DataFrame(summary_rows)


def _row(stage, endpoint):
    return summary[summary["stage"].eq(stage) & summary["endpoint"].eq(endpoint)].iloc[0]


disc_base = _row("fold3_discovery", "base")
disc_new = _row("fold3_discovery", "rerank_blendaware")
conf_base = _row("fold3_confirmation", "base")
conf_old = _row("fold3_confirmation", "rerank_standalone")
conf_new = _row("fold3_confirmation", "rerank_blendaware")
values = {
    "discovery_gain_vs_base": float(disc_base["rmse"] - disc_new["rmse"]),
    "confirmation_gain_vs_base": float(conf_base["rmse"] - conf_new["rmse"]),
    "confirmation_gain_vs_standalone_reranker": float(conf_old["rmse"] - conf_new["rmse"]),
    "confirmation_far_gain_vs_base": float(conf_base["far_gt2000_rmse"] - conf_new["far_gt2000_rmse"]),
    "confirmation_p95_gain_vs_base": float(conf_base["p95_well_rmse"] - conf_new["p95_well_rmse"]),
}
checks = {
    "discovery_gain_positive": values["discovery_gain_vs_base"] > 0.0,
    "confirmation_gain_ge_0p05": values["confirmation_gain_vs_base"] >= 0.05,
    "confirmation_beats_standalone_reranker": values["confirmation_gain_vs_standalone_reranker"] >= 0.0,
    "confirmation_far_gain_nonnegative": values["confirmation_far_gain_vs_base"] >= 0.0,
    "confirmation_p95_not_worse_by_gt_0p10": values["confirmation_p95_gain_vs_base"] >= -0.10,
}
strict_pass = bool(all(checks.values()))

summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
frame[[
    "id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT,
    "base", "rerank_standalone", "rerank_blendaware",
]].to_parquet(OUT_DIR / "fold3_reranker_oof.parquet", index=False, compression="zstd")
print("\nV106 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_crossfit_top256_reranker_Fold3_confirmation",
    "sample_unit": "one complete physical well",
    "topk": TOPK,
    "training_shortlists": "Fold4 scored by Fold5 base; Fold5 scored by Fold4 base",
    "features": {
        "base": len(base_feature_names),
        "parent_relative": len(relative_feature_names),
        "first_stage_score_rank": 2,
        "total": len(base_feature_names) + len(relative_feature_names) + 2,
    },
    "matched_rerankers": ["standalone", "blendaware"],
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery),
        "confirmation_wells": len(fold3) - len(discovery),
        "fold2_touched": False,
        "fold1_touched": False,
        "iterations": ITERATIONS,
    },
    "values": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> cross-fit Fold345 shortlists, train the frozen reranker, and apply once to Fold2"
        if strict_pass
        else "FAIL -> shortlist oracle is not learnable; close blend-aware branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV106 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
