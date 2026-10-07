"""M342-Q1: multi-cut invariant residual GBDT after frozen K2.

The same physical hidden row is presented under official, pseudo1 and pseudo2
visible-cut states whenever available. Its residual label remains truth-K2.
The model is evaluated only on official full rows. This tests whether cut-state
augmentation fixes P1's fold-dependent residual sign.
"""

from pathlib import Path
import gc
import json
import sys
import time
import types

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
P1_SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
OUT = ROOT / "oof_exports" / "main342_multicut_invariant_residual_q1"
MODEL_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
UNTOUCHED_FOLD = 1
TAGS = ("official", "pseudo1", "pseudo2")
MAX_P95_CHANGE = 0.20


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_p1():
    source = P1_SOURCE.read_text(encoding="utf-8")
    marker = "base, manifest, base_features, coverage = O[\"_prepare_inventory\"]()"
    if marker not in source:
        raise RuntimeError("P1 execution marker not found")
    name = "m342_q1_p1_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(P1_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(P1_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


P = _load_p1()
O = P["O"]
B = P["B"]
B["OUT_DIR"] = OUT
B["MODEL_DIR"] = MODEL_DIR
P["OUT"] = OUT
P["MODEL_DIR"] = MODEL_DIR


def _multicut_samples(wells, manifest, base_features, endpoints):
    wanted = set(map(str, wells))
    local = manifest[
        manifest["well"].isin(wanted) & manifest["tag"].isin(TAGS)
    ].copy()
    local["cut_bin"] = 0
    local["states_in_bin"] = local.groupby("well")["tag"].transform("size")
    identity = ["id", "_row_idx", "row_idx", "last_known_tvt", "_cut_tag"]
    requested = list(dict.fromkeys(list(base_features) + identity))
    frame = B["_load_samples"](wells, local, requested)
    id_missing = "id" not in frame or frame["id"].isna().all()
    if id_missing:
        row_column = next(
            (column for column in ("_row_idx", "row_idx") if column in frame and frame[column].notna().all()),
            None,
        )
        if row_column is None:
            raise RuntimeError("Multi-cut samples lack both id and row index")
        frame["id"] = frame["well"].astype(str) + "_" + frame[row_column].astype(np.int64).astype(str)
    elif frame["id"].isna().any():
        raise RuntimeError("Multi-cut samples contain partially missing ids")
    frame["id"] = frame["id"].astype(str)
    if "_cut_tag" not in frame or frame["_cut_tag"].isna().any():
        raise RuntimeError("Multi-cut sample tag was not preserved")
    # Duplicated physical ids across cut states are expected. Endpoint values
    # and labels are shared through a many-to-one merge.
    output = frame.merge(endpoints, on="id", how="inner", validate="many_to_one")
    retained = output.groupby("well")["_cut_tag"].nunique()
    if len(retained) != len(wanted) or int((retained >= 2).sum()) < int(0.90 * len(wanted)):
        raise RuntimeError(
            f"Insufficient multi-cut overlap: wells={len(retained)}/{len(wanted)}, "
            f"two_plus={(retained >= 2).sum()}"
        )
    output["residual_target"] = (
        output["tvt_true_endpoint"].to_numpy(np.float64)
        - output["k2_parent"].to_numpy(np.float64)
    )
    # Preserve B1's per-well total weighting while redistributing it over only
    # the rows that overlap the official hidden suffix.
    totals = output.groupby("well")["_global_weight"].transform("sum")
    output["_global_weight"] = output["_global_weight"] / np.maximum(totals, 1e-12)
    return output.reset_index(drop=True), retained


base, manifest, base_features, coverage = O["_prepare_inventory"]()
endpoints = P["_endpoint_table"]()
features = list(dict.fromkeys(base_features + list(P["ENGINEERED"])))
train_wells = sorted(base.loc[base["fold"].isin(TRAIN_FOLDS), "well"].unique())
selection_wells = sorted(base.loc[base["fold"].eq(SELECTION_FOLD), "well"].unique())
train, train_state_counts = _multicut_samples(train_wells, manifest, base_features, endpoints)
valid_sample, valid_state_counts = _multicut_samples(
    selection_wells, manifest, base_features, endpoints
)
official_manifest = manifest[manifest["tag"].eq("official")].copy()
official_manifest["cut_bin"] = 0
selection_full = B["_load_full"](selection_wells, official_manifest, base)
selection_full = selection_full.merge(endpoints, on="id", how="inner", validate="one_to_one")

print(
    "M342-Q1 CONFIG",
    {
        "coverage": coverage, "features": len(features), "train_rows": len(train),
        "selection_rows": len(selection_full),
        "train_three_state_wells": int((train_state_counts >= 3).sum()),
        "selection_three_state_wells": int((valid_state_counts >= 3).sum()),
    },
    flush=True,
)
model, best_iteration = P["_fit"](
    train, valid_sample, features, label="selection_multicut_residual"
)
correction = P["_correction"](selection_full, model, features)
parent_metric, scan = P["_evaluate"](selection_full, correction, "multicut_residual")
safe = scan.loc[scan["far_gain"].ge(0.0) & scan["p95_change"].le(MAX_P95_CHANGE)].copy()
selected = (safe.iloc[0] if len(safe) else scan.iloc[0]).to_dict()
dose = float(selected["dose"])
parent = selection_full["k2_parent"].to_numpy(np.float64)
prediction = parent + dose * correction
rolled = np.empty_like(correction)
for positions in selection_full.groupby("well", sort=False).indices.values():
    positions = np.asarray(positions, np.int64)
    rolled[positions] = np.roll(correction[positions], max(len(positions) // 4, 1))
roll_prediction = parent + dose * rolled
candidate_metric = B["_metrics"](selection_full, prediction, "multicut_residual")
roll_metric = B["_metrics"](selection_full, roll_prediction, "depth_roll_control")
roll_loss = roll_metric["rmse"] - candidate_metric["rmse"]
mechanism_pass = bool(
    len(safe) > 0 and dose > 0.0 and selected["gain"] >= 0.03
    and selected["far_gain"] >= 0.0 and selected["p95_change"] <= MAX_P95_CHANGE
    and roll_loss >= 0.015
)
print("\nM342-Q1 SELECTION SCAN")
print(scan.to_string(index=False), flush=True)
print("M342-Q1 SELECTION", {**selected, "roll_loss": roll_loss, "pass": mechanism_pass}, flush=True)

confirmation = None
checks = {"selection_mechanism_pass": mechanism_pass}
if mechanism_pass:
    del train, valid_sample, model
    gc.collect()
    train2_wells = sorted(base.loc[base["fold"].isin((3, 4, 5)), "well"].unique())
    fold2_wells = sorted(base.loc[base["fold"].eq(CONFIRMATION_FOLD), "well"].unique())
    train2, train2_state_counts = _multicut_samples(
        train2_wells, manifest, base_features, endpoints
    )
    held2 = B["_load_full"](fold2_wells, official_manifest, base)
    held2 = held2.merge(endpoints, on="id", how="inner", validate="one_to_one")
    model2, _ = P["_fit"](
        train2, None, features, iterations=best_iteration,
        label="confirmation_multicut_residual",
    )
    correction2 = P["_correction"](held2, model2, features)
    parent2 = held2["k2_parent"].to_numpy(np.float64)
    prediction2 = parent2 + dose * correction2
    reverse2 = parent2 - dose * correction2
    parent2_metric = B["_metrics"](held2, parent2, "k2_parent")
    candidate2_metric = B["_metrics"](held2, prediction2, "multicut_residual")
    reverse2_metric = B["_metrics"](held2, reverse2, "negative_direction_control")
    confirmation = {
        "parent": parent2_metric, "candidate": candidate2_metric, "reverse": reverse2_metric,
        "gain": parent2_metric["rmse"] - candidate2_metric["rmse"],
        "far_gain": parent2_metric["far_gt2000_rmse"] - candidate2_metric["far_gt2000_rmse"],
        "reverse_loss": reverse2_metric["rmse"] - candidate2_metric["rmse"],
        "p95_change": candidate2_metric["p95_well_rmse"] - parent2_metric["p95_well_rmse"],
        "train_three_state_wells": int((train2_state_counts >= 3).sum()),
    }
    checks.update({
        "confirmation_gain_ge_0p02": confirmation["gain"] >= 0.02,
        "confirmation_far_gain_ge_0": confirmation["far_gain"] >= 0.0,
        "reverse_loses_ge_0p03": confirmation["reverse_loss"] >= 0.03,
        "confirmation_p95_change_le_0p20": confirmation["p95_change"] <= MAX_P95_CHANGE,
    })

strict_pass = bool(mechanism_pass and confirmation and all(checks.values()))
verdict = {
    "experiment": "main342_multicut_invariant_residual_q1",
    "stage": "fold45_multicut_residual_fold3_selection_fold345_refit_fold2_confirmation",
    "legal_inputs_only": True,
    "sample_unit": "same physical row under up to three visible-cut states",
    "protocol": {
        "train_folds": TRAIN_FOLDS, "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD if mechanism_pass else None,
        "untouched_fold": UNTOUCHED_FOLD,
    },
    "tags": TAGS,
    "features": {"base": len(base_features), "engineered": P["ENGINEERED"], "total": len(features)},
    "model": {"best_iteration": best_iteration, "correction_cap_ft": P["CORRECTION_CAP_FT"],
              "smooth_rows": P["SMOOTH_ROWS"]},
    "coverage": coverage,
    "selection": {**selected, "safe_doses": safe["dose"].astype(float).tolist(), "roll_loss": roll_loss},
    "mechanism_pass": mechanism_pass,
    "confirmation": confirmation,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze multi-cut model/dose and confirm once on Fold1"
        if strict_pass else
        "FAIL -> multi-cut invariance does not repair residual sign transfer; close Q1"
    ),
}

scan.to_csv(OUT / "selection_scan.csv", index=False)
(OUT / "feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-Q1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
