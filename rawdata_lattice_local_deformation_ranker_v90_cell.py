"""V90: outer-fold candidate ranker for V88 local deformations.

The available V70/V76 lineage contains Folds 1/2/3 only. Train on complete
wells from Folds 1/2, use 24 deterministic Fold3 wells to choose one frozen
model/dose, and confirm on the remaining Fold3 wells. A pass is only a gateway
to reconstructing Folds 4/5; it is not a deployment verdict.
"""

from __future__ import annotations

import json
import re
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
EXPERIMENT = "rawdata_lattice_local_deformation_ranker_v90"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V89_SOURCE = ROOT / "rawdata_lattice_local_deformation_gr_scorer_v89_cell.py"
MODEL_CONFIGS = (
    {"id": "leaf15_child100", "num_leaves": 15, "min_child_samples": 100, "reg_lambda": 8.0},
    {"id": "leaf31_child100", "num_leaves": 31, "min_child_samples": 100, "reg_lambda": 12.0},
    {"id": "leaf31_child200", "num_leaves": 31, "min_child_samples": 200, "reg_lambda": 16.0},
)
DOSES = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75)
PROFILE_COLUMNS = (
    "raw_cauchy_mean",
    "raw_cauchy_blockmedian",
    "shift_cauchy_mean",
    "raw_shift_cauchy",
    "raw_clip_blockmedian",
)
EXCLUSIONS = []


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


def _bootstrap(path, marker, name):
    if not path.is_file():
        raise FileNotFoundError(path)
    source = path.read_text(encoding="utf-8")
    if marker not in source:
        raise RuntimeError(f"Cannot identify execution boundary in {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(path), "exec"), module.__dict__)
    return module


def _name_features(name):
    values = {
        "name_identity": float(name == "identity"),
        "name_trend": float(name.startswith("trend_")),
        "name_step": float(name.startswith("step_")),
        "name_middle": 0.0,
        "name_endpoint": 0.0,
        "name_onset": 0.0,
        "name_step_offset": 0.0,
    }
    trend = re.match(r"trend_m([+-]?[0-9.]+)_e([+-]?[0-9.]+)", name)
    step = re.match(r"step_s([0-9.]+)_o([+-]?[0-9.]+)", name)
    if trend:
        values["name_middle"] = float(trend.group(1)) / 24.0
        values["name_endpoint"] = float(trend.group(2)) / 24.0
    if step:
        values["name_onset"] = float(step.group(1)) / 4000.0
        values["name_step_offset"] = float(step.group(2)) / 24.0
    return values


def _candidate_rank(values):
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), np.float64)
    rank[order] = np.arange(len(values), dtype=np.float64)
    return rank / max(len(values) - 1, 1)


def _panel(frame, caches, labelled):
    rows = []
    groups = []
    for cache in caches:
        local = frame.iloc[cache["indices"]]
        baseline = local["v76"].to_numpy(np.float64)
        truth = local[N.TRUTH].to_numpy(np.float64)
        distance = local["md_since"].to_numpy(np.float64)
        candidates = cache["predictions"].astype(np.float64)
        count = len(cache["names"])
        groups.append(count)

        if labelled:
            mse = np.mean((candidates - truth[None, :]) ** 2, axis=1)
            ordinal = np.argsort(np.argsort(mse, kind="stable"), kind="stable")
            relevance = (count - 1 - ordinal).astype(np.int32)
        else:
            relevance = np.zeros(count, np.int32)

        profile_rank = {
            name: _candidate_rank(cache["aligned"][name]) for name in PROFILE_COLUMNS
        }
        score_spread = {
            name: float(np.std(cache["aligned"][name])) for name in PROFILE_COLUMNS
        }
        max_distance = max(float(np.max(distance)), 1.0)
        for index, name in enumerate(cache["names"]):
            move = candidates[index] - baseline
            item = {
                "well": cache["well"],
                "candidate_index": int(index),
                "candidate": name,
                "label": int(relevance[index]),
                "candidate_fraction": index / max(count - 1, 1),
                "penalty": float(cache["penalty"][index]),
                "move_mean": float(np.mean(move)) / 8.0,
                "move_std": float(np.std(move)) / 8.0,
                "move_rms": float(np.sqrt(np.mean(move * move))) / 8.0,
                "move_endpoint": float(move[-1]) / 8.0,
                "move_midpoint": float(move[len(move) // 2]) / 8.0,
                "move_max_abs": float(np.max(np.abs(move))) / 8.0,
                "move_slope": float((move[-1] - move[0]) / max_distance) * 1000.0,
                "well_rows": len(local) / 6000.0,
                "well_max_distance": max_distance / 6000.0,
                "well_dose": float(local["dose"].iloc[0]),
            }
            item.update(_name_features(name))
            for profile in PROFILE_COLUMNS:
                aligned = float(cache["aligned"][profile][index])
                rolled = float(cache["rolled"][profile][index])
                item[f"{profile}_score"] = aligned
                item[f"{profile}_rank"] = float(profile_rank[profile][index])
                item[f"{profile}_vs_rolled"] = aligned - rolled
                item[f"{profile}_well_spread"] = score_spread[profile]
            rows.append(item)
    panel = pd.DataFrame(rows)
    metadata = {"well", "candidate", "candidate_index", "label"}
    features = [column for column in panel.columns if column not in metadata]
    panel[features] = panel[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return panel, groups, features


def _fit_model(train_panel, groups, features, config):
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=list(range(48)),
        n_estimators=300,
        learning_rate=0.025,
        num_leaves=config["num_leaves"],
        min_child_samples=config["min_child_samples"],
        reg_alpha=2.0,
        reg_lambda=config["reg_lambda"],
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=20260730,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_panel[features], train_panel["label"], group=groups)
    return model


def _choices(panel, model, features):
    scored = panel[["well", "candidate_index", "candidate"]].copy()
    scored["score"] = model.predict(panel[features])
    return (
        scored.sort_values(["well", "score", "candidate_index"], ascending=[True, False, True], kind="stable")
        .groupby("well", sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def _apply(frame, caches, choices, dose):
    output = frame["v76"].to_numpy(np.float64).copy()
    choice_map = dict(zip(choices["well"].astype(str), choices["candidate_index"].astype(int)))
    for cache in caches:
        index = int(choice_map.get(str(cache["well"]), 0))
        rows = cache["indices"]
        baseline = frame.iloc[rows]["v76"].to_numpy(np.float64)
        candidate = cache["predictions"][index].astype(np.float64)
        prediction = baseline + float(dose) * (candidate - baseline)
        distance = frame.iloc[rows]["md_since"].to_numpy(np.float64)
        prediction[distance <= 1000.0] = baseline[distance <= 1000.0]
        output[rows] = prediction
    return output


def _subset_metric(frame, prediction, wells, endpoint):
    mask = frame["well"].astype(str).isin(set(wells)).to_numpy()
    local = frame.loc[mask].reset_index(drop=True)
    return N.O._metric(local, np.asarray(prediction)[mask], endpoint)


def _comparison(base, candidate):
    return {
        "gain": base["rmse"] - candidate["rmse"],
        "far_gain": base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"],
        "near_degradation": candidate["near_0_500_rmse"] - base["near_0_500_rmse"],
        "p95_gain": base["p95_well_rmse"] - candidate["p95_well_rmse"],
    }


def _eligible_frame_and_caches(frame, fold):
    """Apply the lattice loader's existing eligibility rule before panel build."""
    eligible = []
    for well in sorted(frame["well"].astype(str).unique()):
        try:
            N.R.V["_load_record"](well)
            eligible.append(well)
        except Exception as exc:
            EXCLUSIONS.append({"fold": int(fold), "well": well, "reason": repr(exc)})
    output = frame[frame["well"].astype(str).isin(set(eligible))].copy().reset_index(drop=True)
    if output.empty:
        raise RuntimeError(f"Fold{fold}: no lattice-eligible wells")
    print(
        f"V90 Fold{fold} eligibility {len(eligible)}/{frame['well'].nunique()} "
        f"excluded={frame['well'].nunique() - len(eligible)}",
        flush=True,
    )
    return output, N._well_cache(output, fold)


N = _bootstrap(V89_SOURCE, "for required in (V88_SOURCE, V86_SOURCE):", "v89_v90_defs")
N.O = N._bootstrap(N.V88_SOURCE, "for required in (V70_PATH, V76_WEIGHTS):", "v88_v90_defs")
N.M = N._bootstrap(
    N.V86_SOURCE,
    "for required in (V70_SOURCE, V85_SOURCE, V70_PATH, V76_WEIGHTS):",
    "v86_v90_defs",
)
N.M.R = N.M._bootstrap_v70()
N.M.H = N.M._bootstrap_v85_helpers()
N.R = N.M.R

train_frame = pd.concat([N.O._fold_frame(1), N.O._fold_frame(2)], ignore_index=True)
train_frame, train_caches = _eligible_frame_and_caches(train_frame, 12)
train_panel, train_groups, features = _panel(train_frame, train_caches, labelled=True)

fold3 = N.O._fold_frame(3)
fold3, fold3_caches = _eligible_frame_and_caches(fold3, 3)
fold3_panel, _, features3 = _panel(fold3, fold3_caches, labelled=False)
if features3 != features:
    raise RuntimeError("Fold3 feature schema mismatch")

fold3_wells = sorted(fold3["well"].astype(str).unique())
discovery_wells = fold3_wells[:24]
confirmation_wells = fold3_wells[24:]
base_discovery = _subset_metric(fold3, fold3["v76"], discovery_wells, "v76")
base_confirmation = _subset_metric(fold3, fold3["v76"], confirmation_wells, "v76")

models = {}
choices3 = {}
scan_rows = []
for config in MODEL_CONFIGS:
    model = _fit_model(train_panel, train_groups, features, config)
    models[config["id"]] = model
    choices = _choices(fold3_panel, model, features)
    choices3[config["id"]] = choices
    for dose in DOSES:
        prediction = _apply(fold3, fold3_caches, choices, dose)
        metric = _subset_metric(fold3, prediction, discovery_wells, "candidate")
        scan_rows.append({
            "model": config["id"],
            "dose": float(dose),
            "changed_wells": int(np.sum(choices["candidate_index"].to_numpy() != 0)),
            **metric,
            **_comparison(base_discovery, metric),
        })
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "model", "dose"], kind="stable")
scan.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)

eligible = scan[
    scan["gain"].ge(0.05)
    & scan["far_gain"].ge(0.05)
    & scan["near_degradation"].le(1e-8)
    & scan["p95_gain"].ge(-0.05)
].copy()

selected = None
confirmation = None
fold2_result = None
strict_pass = False
summary_rows = [{"stage": "fold3_discovery", **base_discovery}]
if not eligible.empty:
    selected = eligible.iloc[0].to_dict()
    model_id = selected["model"]
    dose = float(selected["dose"])
    pred3 = _apply(fold3, fold3_caches, choices3[model_id], dose)
    metric3 = _subset_metric(fold3, pred3, confirmation_wells, "ranker")
    confirmation = _comparison(base_confirmation, metric3)
    summary_rows.extend([
        {"stage": "fold3_confirmation", **base_confirmation},
        {"stage": "fold3_confirmation", **metric3},
    ])
    confirmation_pass = bool(
        confirmation["gain"] >= 0.05
        and confirmation["far_gain"] >= 0.05
        and confirmation["near_degradation"] <= 1e-8
        and confirmation["p95_gain"] >= -0.05
    )
    strict_pass = confirmation_pass

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold12_candidate_ranker_Fold3_24well_selection_130well_confirmation_gateway",
    "sample_unit": "one complete physical well",
    "candidate_family": "V88 frozen 47 local deformations",
    "feature_count": len(features),
    "train_wells": int(train_frame["well"].nunique()),
    "excluded_wells": EXCLUSIONS,
    "selection_search_count": len(MODEL_CONFIGS) * len(DOSES),
    "selected_recipe": None if selected is None else {
        "model": selected["model"], "dose": selected["dose"],
        "discovery_gain": selected["gain"], "discovery_far_gain": selected["far_gain"],
    },
    "fold3_confirmation": confirmation,
    "fold2_application": None,
    "fold1_touched": True,
    "coverage_limited": True,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS GATEWAY -> reconstruct V70/V76 candidate panels on Folds 4/5 for true outer confirmation"
        if strict_pass else "FAIL -> candidate identity is not predictable across wells; retain V76"
    ),
}
_write_json(OUT_DIR / "feature_schema.json", features)
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV90 DISCOVERY TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV90 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV90 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
