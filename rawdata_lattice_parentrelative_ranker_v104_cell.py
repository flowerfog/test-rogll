"""V104: parent-relative candidate evidence for the blend-aware rank target.

V103 proved that changing labels alone fails because the frozen 131 features do
not describe a candidate relative to the deployed parent. This matched test
adds one fixed parent-relative feature block and fits three 67-tree rankers:

  base:       original features, standalone labels
  rel_old:    original + parent-relative features, standalone labels
  rel_blend:  original + parent-relative features, blend-aware labels

Fold45 trains, Fold3 discovery/confirmation evaluates, and Fold2 is touched
once only if rel_blend passes the frozen confirmation gates. Fold1 is untouched.
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
EXPERIMENT = "rawdata_lattice_parentrelative_ranker_v104"
OUT_DIR = OOF_ROOT / EXPERIMENT
REL_DIR = OUT_DIR / "relative_features"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, REL_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V81_SOURCE = ROOT / "rawdata_lattice_structural_risk_ranker_v81b_cell.py"
V103_DIR = OOF_ROOT / "rawdata_lattice_blendaware_ranker_v103"
V103_PANELS = V103_DIR / "matched_panels"
V76_WEIGHTS = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
DISCOVERY_WELLS = 24
FROZEN_ITERATIONS = 67
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
    if not V81_SOURCE.is_file():
        raise FileNotFoundError(V81_SOURCE)
    source = V81_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V81B EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V81b marker missing from {V81_SOURCE}")
    module = types.ModuleType("raw_lattice_v81_parentrelative_definitions")
    module.__file__ = str(V81_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V81_SOURCE), "exec"), module.__dict__)
    return module


N = _bootstrap_v81()
Q, V, B, prior = N.Q, N.V, N.B, N.prior
incumbent = N.incumbent
fold_by_well = N.fold_by_well
base_feature_names = list(N.feature_names)

for required in (V103_PANELS, V76_WEIGHTS):
    if not required.exists():
        raise FileNotFoundError(required)

weights = pd.read_parquet(V76_WEIGHTS)
weights["well"] = weights["well"].astype(str)
weight_map = dict(zip(weights["well"], weights["ensemble_weight"].astype(float)))


def _panel_path(well):
    return V103_PANELS / f"{well}.npz"


def _load_panel(well):
    with np.load(_panel_path(well), allow_pickle=False) as data:
        panel = {key: data[key] for key in data.files}
    if panel["features"].shape[1] != len(base_feature_names):
        raise RuntimeError(f"{well}: V103 base feature schema mismatch")
    return panel


def _candidate_tvt(record, paths, segments):
    rates = prior["rate_grid"][paths]
    count = len(paths)
    parts = []
    start_surface = np.full(count, record["anchor_surface"], np.float64)
    start_md = float(record["anchor_md"])
    for segment_index, rows in enumerate(segments):
        rate = rates[:, segment_index]
        delta = record["md"][rows][None, :] - start_md
        surface = start_surface[:, None] + rate[:, None] * delta
        parts.append(surface - record["z"][rows][None, :])
        start_surface += rate * float(record["md"][rows][-1] - start_md)
        start_md = float(record["md"][rows][-1])
    return np.concatenate(parts, axis=1), np.concatenate(segments)


def _relative_features(record, paths, parent):
    segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
    candidate_tvt, sampled_rows = _candidate_tvt(record, paths, segments)
    eval_rows = record["eval_rows"]
    positions = np.searchsorted(eval_rows, sampled_rows)
    if (
        np.any(positions >= len(eval_rows))
        or not np.array_equal(eval_rows[positions], sampled_rows)
        or len(parent) != len(eval_rows)
    ):
        raise RuntimeError(f"{record['well']}: parent/sample alignment")
    move = candidate_tvt - np.asarray(parent, np.float64)[positions][None, :]
    md = record["md"][sampled_rows].astype(np.float64)
    delta_md = np.maximum(np.diff(md), 1e-6)
    slope = np.diff(move, axis=1) / delta_md[None, :]

    blocks = []
    names = []
    for values, prefix, scale in (
        (move, "parent_move", 24.0),
        (np.abs(move), "parent_abs_move", 24.0),
        (slope, "parent_move_slope", 0.05),
        (np.abs(slope), "parent_abs_slope", 0.05),
    ):
        block, block_names = Q._channel_features(values / scale, prefix)
        blocks.append(block)
        names.extend(block_names)

    endpoint = np.stack(
        [
            move[:, 0] / 24.0,
            move[:, -1] / 24.0,
            (move[:, -1] - move[:, 0]) / 24.0,
            np.sqrt(np.mean(move * move, axis=1)) / 24.0,
            np.max(move, axis=1) / 24.0,
            np.min(move, axis=1) / 24.0,
        ],
        axis=1,
    )
    blocks.append(endpoint)
    names.extend([
        "parent_move_first", "parent_move_last", "parent_move_span",
        "parent_move_rms", "parent_move_max", "parent_move_min",
    ])
    output = np.concatenate(blocks, axis=1)
    output = np.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
    return np.clip(output, -50.0, 50.0).astype(np.float32), names


def _relative_path(well):
    return REL_DIR / f"{well}.npz"


relative_feature_names = None


def _build_relative(well):
    global relative_feature_names
    destination = _relative_path(well)
    panel = _load_panel(well)
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as data:
            values = data["features"]
            names = json.loads(str(data["feature_names_json"].item()))
        if len(values) == len(panel["paths"]):
            if relative_feature_names is None:
                relative_feature_names = names
            elif names != relative_feature_names:
                raise RuntimeError(f"{well}: cached relative schema mismatch")
            return {"well": well, "status": "reused", "reason": ""}

    record = V["_load_record"](well)
    local = incumbent[incumbent["well"].eq(well)].sort_values("row_idx", kind="stable")
    parent = local[PARENT].to_numpy(np.float64)
    values, names = _relative_features(record, panel["paths"], parent)
    if relative_feature_names is None:
        relative_feature_names = names
    elif names != relative_feature_names:
        raise RuntimeError(f"{well}: relative schema mismatch")
    np.savez_compressed(
        destination,
        features=values,
        feature_names_json=np.asarray(json.dumps(names)),
    )
    return {"well": well, "status": "built", "reason": ""}


panel_wells = sorted(
    well for well, fold in fold_by_well.items()
    if fold in (3, 4, 5) and _panel_path(well).is_file()
)
reports = []
for number, well in enumerate(panel_wells, 1):
    try:
        reports.append(_build_relative(well))
    except Exception as exc:
        reports.append({"well": well, "status": "failed", "reason": repr(exc)})
    if number % 25 == 0 or number == len(panel_wells):
        failures = sum(row["status"] == "failed" for row in reports)
        print(
            f"V104 relative features {number}/{len(panel_wells)} failures={failures} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )
report_frame = pd.DataFrame(reports)
report_frame.to_csv(OUT_DIR / "relative_feature_report.csv", index=False)
failed = report_frame[report_frame["status"].eq("failed")]
if len(failed):
    raise RuntimeError(f"{len(failed)} relative feature failures: {failed.head().to_dict('records')}")
if relative_feature_names is None:
    raise RuntimeError("No relative feature schema")


def _features(panel, well, use_relative):
    base = panel["features"].astype(np.float32, copy=False)
    if not use_relative:
        return base
    with np.load(_relative_path(well), allow_pickle=False) as data:
        relative = data["features"].astype(np.float32)
    if len(relative) != len(base):
        raise RuntimeError(f"{well}: relative/base row mismatch")
    return np.concatenate([base, relative], axis=1)


def _target(panel, blendaware):
    key = "blend_mse" if blendaware else "mse"
    return panel[key].astype(np.float64)


def _matrix(wells, use_relative, blendaware):
    x_parts, y_parts, groups = [], [], []
    for number, well in enumerate(wells, 1):
        panel = _load_panel(well)
        indices = Q._training_indices(panel, well)
        x_parts.append(_features(panel, well, use_relative)[indices])
        y_parts.append(Q._rank_labels(_target(panel, blendaware)[indices]))
        groups.append(int(len(indices)))
        if number % 100 == 0 or number == len(wells):
            print(
                f"V104 matrix rel={use_relative} blend={blendaware} {number}/{len(wells)}",
                flush=True,
            )
    return np.concatenate(x_parts), np.concatenate(y_parts), groups


def _fit(wells, use_relative, blendaware, label):
    x, y, groups = _matrix(wells, use_relative, blendaware)
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", eval_at=(1, 8, 32),
        n_estimators=FROZEN_ITERATIONS, learning_rate=0.025,
        num_leaves=31, max_depth=-1, min_child_samples=80,
        subsample=0.85, colsample_bytree=0.80,
        reg_alpha=0.2, reg_lambda=2.0, random_state=RANDOM_SEED,
        n_jobs=max(1, min(16, os.cpu_count() or 1)), verbosity=-1,
    )
    model.fit(x, y, group=groups)
    model.booster_.save_model(str(MODEL_DIR / f"{label}.txt"))
    print("V104 MODEL", {"label": label, "rows": len(x), "features": x.shape[1]}, flush=True)
    del x, y
    gc.collect()
    return model.booster_


MODEL_SPECS = {
    "base": {"relative": False, "blendaware": False},
    "rel_old": {"relative": True, "blendaware": False},
    "rel_blend": {"relative": True, "blendaware": True},
}


def _evaluate(wells, models, label, panels):
    parts, reports = [], []
    for number, well in enumerate(wells, 1):
        local = incumbent[incumbent["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
        parent = local[PARENT].to_numpy(np.float64)
        report = {"well": well, "stage": label, "status": "built", "reason": ""}
        try:
            record = V["_load_record"](well)
            if panels:
                panel = _load_panel(well)
                paths = panel["paths"].astype(np.int16, copy=False)
                segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
                base = panel["features"].astype(np.float32, copy=False)
                relative = _features(panel, well, True)[:, len(base_feature_names):]
            else:
                paths, source_mask, arm_rank, arm_score_z, segments, _ = Q._candidate_union(record, prior)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                    base, names = Q._candidate_evidence(
                        record, prior, paths, segments, arm_rank, arm_score_z, source_mask
                    )
                if list(names) != base_feature_names:
                    raise RuntimeError("base feature schema mismatch")
                relative, names = _relative_features(record, paths, parent)
                if names != relative_feature_names:
                    raise RuntimeError("relative feature schema mismatch")
            if len(base) != len(paths) or len(relative) != len(paths):
                raise RuntimeError("candidate feature/path mismatch")
            eval_rows = record["eval_rows"]
            truth = record["truth"][eval_rows].astype(np.float64)
            distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
            if len(local) != len(eval_rows):
                raise RuntimeError(f"row count {len(local)}/{len(eval_rows)}")
            if np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
                raise RuntimeError("truth alignment")
            if np.max(np.abs(local["md_since"].to_numpy(np.float64) - distance)) > 1.01:
                raise RuntimeError("distance alignment")
            if well not in weight_map:
                raise RuntimeError("missing V76 dose")
            dose = float(weight_map[well])
            for model_id, model in models.items():
                spec = MODEL_SPECS[model_id]
                matrix = np.concatenate([base, relative], axis=1) if spec["relative"] else base
                chosen = int(np.argmax(model.predict(matrix)))
                prediction = B["_path_prediction"](record, prior, paths[chosen], segments)[eval_rows]
                local[f"blend_{model_id}"] = (
                    parent + dose * (np.asarray(prediction, np.float64) - parent)
                ).astype(np.float32)
                report[f"chosen_{model_id}"] = chosen
            report["dose"] = dose
            report["candidate_count"] = int(len(paths))
        except Exception as exc:
            report["status"] = "parent_fallback"
            report["reason"] = repr(exc)
            for model_id in models:
                local[f"blend_{model_id}"] = parent.astype(np.float32)
        parts.append(local)
        reports.append(report)
        if number % 10 == 0 or number == len(wells):
            fallbacks = sum(row["status"] != "built" for row in reports)
            print(
                f"V104 evaluate {label} {number}/{len(wells)} fallbacks={fallbacks} "
                f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
            )
    return pd.concat(parts, ignore_index=True), reports


def _summary(frame, models, stage):
    rows = [{"stage": stage, "model": "parent", **N._metric(frame, frame[PARENT], "parent")}]
    for model_id in models:
        rows.append({
            "stage": stage, "model": model_id,
            **N._metric(frame, frame[f"blend_{model_id}"], "blend_v76"),
        })
    return rows


train45 = sorted(w for w, f in fold_by_well.items() if f in TRAIN_FOLDS and _panel_path(w).is_file())
fold3 = sorted(w for w, f in fold_by_well.items() if f == SELECTION_FOLD and _panel_path(w).is_file())
fold2 = sorted(w for w, f in fold_by_well.items() if f == APPLICATION_FOLD)
if len(train45) < 307 or len(fold3) < 153 or len(fold2) < 154:
    raise RuntimeError(f"coverage train45={len(train45)} fold3={len(fold3)} fold2={len(fold2)}")
discovery = fold3[:DISCOVERY_WELLS]
confirmation = fold3[DISCOVERY_WELLS:]

print("V104 CONFIG", {
    "train45": len(train45), "fold3": len(fold3), "fold2": len(fold2),
    "base_features": len(base_feature_names),
    "relative_features": len(relative_feature_names),
}, flush=True)

models45 = {
    model_id: _fit(train45, spec["relative"], spec["blendaware"], f"fold45_{model_id}")
    for model_id, spec in MODEL_SPECS.items()
}
fold3_frame, fold3_reports = _evaluate(fold3, models45, "fold3", panels=True)
pd.DataFrame(fold3_reports).to_csv(OUT_DIR / "fold3_report.csv", index=False)
summary_rows = _summary(
    fold3_frame[fold3_frame["well"].isin(discovery)], models45, "fold3_discovery"
)
summary_rows += _summary(
    fold3_frame[fold3_frame["well"].isin(confirmation)], models45, "fold3_confirmation"
)
summary = pd.DataFrame(summary_rows)


def _row(stage, model):
    return summary[summary["stage"].eq(stage) & summary["model"].eq(model)].iloc[0]


disc_base, disc_new = _row("fold3_discovery", "base"), _row("fold3_discovery", "rel_blend")
conf_base, conf_old, conf_new = (
    _row("fold3_confirmation", "base"),
    _row("fold3_confirmation", "rel_old"),
    _row("fold3_confirmation", "rel_blend"),
)
fold3_values = {
    "discovery_gain_vs_base": float(disc_base["rmse"] - disc_new["rmse"]),
    "confirmation_gain_vs_base": float(conf_base["rmse"] - conf_new["rmse"]),
    "confirmation_gain_vs_relative_old_label": float(conf_old["rmse"] - conf_new["rmse"]),
    "confirmation_far_gain_vs_base": float(conf_base["far_gt2000_rmse"] - conf_new["far_gt2000_rmse"]),
    "confirmation_p95_gain_vs_base": float(conf_base["p95_well_rmse"] - conf_new["p95_well_rmse"]),
}
confirmation_pass = bool(
    fold3_values["discovery_gain_vs_base"] > 0.0
    and fold3_values["confirmation_gain_vs_base"] >= 0.05
    and fold3_values["confirmation_gain_vs_relative_old_label"] >= 0.0
    and fold3_values["confirmation_far_gain_vs_base"] >= 0.0
    and fold3_values["confirmation_p95_gain_vs_base"] >= -0.10
)

fold2_values = None
application_pass = False
if confirmation_pass:
    train345 = sorted(w for w, f in fold_by_well.items() if f in (3, 4, 5) and _panel_path(w).is_file())
    models345 = {
        model_id: _fit(train345, spec["relative"], spec["blendaware"], f"fold345_{model_id}")
        for model_id, spec in MODEL_SPECS.items()
    }
    fold2_frame, fold2_reports = _evaluate(fold2, models345, "fold2", panels=False)
    pd.DataFrame(fold2_reports).to_csv(OUT_DIR / "fold2_report.csv", index=False)
    summary_rows += _summary(fold2_frame, models345, "fold2_application")
    summary = pd.DataFrame(summary_rows)
    fold2_base = summary[summary["stage"].eq("fold2_application") & summary["model"].eq("base")].iloc[0]
    fold2_old = summary[summary["stage"].eq("fold2_application") & summary["model"].eq("rel_old")].iloc[0]
    fold2_new = summary[summary["stage"].eq("fold2_application") & summary["model"].eq("rel_blend")].iloc[0]
    fold2_values = {
        "gain_vs_base": float(fold2_base["rmse"] - fold2_new["rmse"]),
        "gain_vs_relative_old_label": float(fold2_old["rmse"] - fold2_new["rmse"]),
        "far_gain_vs_base": float(fold2_base["far_gt2000_rmse"] - fold2_new["far_gt2000_rmse"]),
        "p95_gain_vs_base": float(fold2_base["p95_well_rmse"] - fold2_new["p95_well_rmse"]),
    }
    application_pass = bool(
        fold2_values["gain_vs_base"] >= 0.05
        and fold2_values["gain_vs_relative_old_label"] >= 0.0
        and fold2_values["far_gain_vs_base"] >= 0.0
        and fold2_values["p95_gain_vs_base"] >= -0.10
    )
    keep = ["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, *[f"blend_{m}" for m in models345]]
    fold2_frame[keep].to_parquet(
        OUT_DIR / "fold2_parentrelative_oof.parquet", index=False, compression="zstd"
    )

summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
print("\nV104 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)

strict_pass = bool(confirmation_pass and application_pass)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_parentrelative_attribution_Fold3_confirmation_Fold2_application",
    "changed_information": "candidate trajectory relative to legal deployed parent",
    "models": MODEL_SPECS,
    "feature_counts": {
        "base": len(base_feature_names),
        "relative_added": len(relative_feature_names),
        "total": len(base_feature_names) + len(relative_feature_names),
    },
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery),
        "confirmation_wells": len(confirmation),
        "application_fold": APPLICATION_FOLD if confirmation_pass else None,
        "fold1_touched": False,
        "iterations": FROZEN_ITERATIONS,
    },
    "fold3": {**fold3_values, "pass": confirmation_pass},
    "fold2": None if fold2_values is None else {**fold2_values, "pass": application_pass},
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> fit all-773 parent-relative blend-aware ranker and export Kaggle bundle"
        if strict_pass
        else "FAIL -> parent-relative observability is insufficient; close blend-aware ranker branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV104 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
