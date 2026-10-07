"""V89: label-free GR scorer for V88's frozen local deformation family.

Fold 3 selects one of five preregistered robust GR profiles and five fixed
deformation penalties. Fold 2 is confirmation only. A circularly rolled
future-GR control must lose. Truth is used only for configuration selection
and metrics, never in candidate generation or candidate scoring.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_local_deformation_gr_scorer_v89"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V88_SOURCE = ROOT / "rawdata_lattice_local_deformation_oracle_v88_cell.py"
V86_SOURCE = ROOT / "rawdata_lattice_v70_matched_disagreement_guard_v86_cell.py"
PROFILES = (
    "raw_cauchy_mean",
    "raw_cauchy_blockmedian",
    "shift_cauchy_mean",
    "raw_shift_cauchy",
    "raw_clip_blockmedian",
)
PENALTIES = (0.0, 0.03, 0.10, 0.30, 1.0)
TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"


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


def _robust_standardize(values):
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 10.0, np.float64)
    fill = float(np.max(values[finite]) + 5.0)
    values = np.where(finite, values, fill)
    centre = float(np.median(values))
    scale = float(np.median(np.abs(values - centre))) * 1.4826
    if scale < 1e-8:
        scale = float(np.std(values))
    return (values - centre) / max(scale, 1e-8)


def _block_reduce(values, distance, block_ft=128.0):
    blocks = np.floor(np.asarray(distance, np.float64) / block_ft).astype(np.int32)
    output = []
    for block in np.unique(blocks):
        mask = blocks == block
        if mask.any():
            output.append(np.nanmean(values[:, mask], axis=1))
    return np.nanmedian(np.vstack(output), axis=0) if output else np.nanmean(values, axis=1)


def _profile_costs(residual, distance):
    residual = np.asarray(residual, np.float64)
    cauchy = np.log1p(residual * residual)
    clipped = np.minimum(residual * residual, 9.0)
    return {
        "cauchy_mean": np.nanmean(cauchy, axis=1),
        "cauchy_blockmedian": _block_reduce(cauchy, distance),
        "clip_blockmedian": _block_reduce(clipped, distance),
    }


def _well_cache(frame, fold):
    caches = []
    wells = sorted(frame["well"].astype(str).unique())
    for number, well in enumerate(wells, 1):
        local = frame[frame["well"].astype(str).eq(well)].copy()
        record = R.V["_load_record"](well)
        eval_rows = record["eval_rows"]
        truth = record["truth"][eval_rows].astype(np.float64)
        if len(local) != len(eval_rows):
            raise RuntimeError(f"{well}: row mismatch {len(local)}/{len(eval_rows)}")
        if np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
            raise RuntimeError(f"{well}: truth mismatch")
        distance = local["md_since"].to_numpy(np.float64)
        names, deformation = O._candidate_deformations(distance)
        parent = local[PARENT].to_numpy(np.float64)
        coarse = local[COARSE].to_numpy(np.float64)
        dose = float(local["dose"].iloc[0])
        candidate_coarse = coarse[None, :] + deformation
        predictions = parent[None, :] + dose * (candidate_coarse - parent[None, :])

        # Four-row thinning keeps the score cheap while retaining sub-formation detail.
        score_rows = np.arange(0, len(eval_rows), 4, dtype=np.int64)
        candidate_tvt = candidate_coarse[:, score_rows]
        observed = record["gr"][eval_rows][score_rows].astype(np.float64)
        rolled = np.roll(observed, max(1, len(observed) // 5))
        raw_reference = R.V["_interp_nan"](
            candidate_tvt.ravel(), record["type_tvt"], record["type_gr"]
        ).reshape(candidate_tvt.shape)
        shifted_gr = R.B["_reference_for_record"](record, "shift")
        shift_reference = R.V["_interp_nan"](
            candidate_tvt.ravel(), record["type_tvt"], shifted_gr
        ).reshape(candidate_tvt.shape)
        scale = max(float(record["noise_scale"]), 1e-6)
        raw = _profile_costs((raw_reference - observed[None, :]) / scale, distance[score_rows])
        shifted = _profile_costs((shift_reference - observed[None, :]) / scale, distance[score_rows])
        raw_rolled = _profile_costs((raw_reference - rolled[None, :]) / scale, distance[score_rows])
        shifted_rolled = _profile_costs((shift_reference - rolled[None, :]) / scale, distance[score_rows])

        coverage = np.mean(np.isfinite(raw_reference), axis=1)
        coverage_penalty = np.maximum(0.0, 0.90 - coverage) * 20.0
        rms = np.sqrt(np.mean(deformation * deformation, axis=1)) / 8.0
        rough = np.mean(np.abs(np.diff(deformation, axis=1)), axis=1) * 4.0
        penalty = rms + rough

        def profiles(raw_cost, shift_cost):
            return {
                "raw_cauchy_mean": _robust_standardize(raw_cost["cauchy_mean"]) + coverage_penalty,
                "raw_cauchy_blockmedian": _robust_standardize(raw_cost["cauchy_blockmedian"]) + coverage_penalty,
                "shift_cauchy_mean": _robust_standardize(shift_cost["cauchy_mean"]) + coverage_penalty,
                "raw_shift_cauchy": 0.5 * (
                    _robust_standardize(raw_cost["cauchy_mean"])
                    + _robust_standardize(shift_cost["cauchy_mean"])
                ) + coverage_penalty,
                "raw_clip_blockmedian": _robust_standardize(raw_cost["clip_blockmedian"]) + coverage_penalty,
            }

        caches.append({
            "well": well,
            "indices": local.index.to_numpy(np.int64),
            "names": names,
            "predictions": predictions.astype(np.float32),
            "aligned": profiles(raw, shifted),
            "rolled": profiles(raw_rolled, shifted_rolled),
            "penalty": penalty,
        })
        if number % 25 == 0 or number == len(wells):
            print(f"V89 Fold{fold} cache {number}/{len(wells)}", flush=True)
    return caches


def _prediction(frame, caches, profile, penalty_weight, control):
    output = frame["v76"].to_numpy(np.float64).copy()
    choices = []
    for cache in caches:
        scores = cache[control][profile] + float(penalty_weight) * cache["penalty"]
        choice = int(np.argmin(scores))
        output[cache["indices"]] = cache["predictions"][choice]
        choices.append({
            "well": cache["well"],
            "candidate": cache["names"][choice],
            "candidate_index": choice,
        })
    return output, pd.DataFrame(choices)


def _scan_fold(frame, caches, fold):
    baseline = O._metric(frame, frame["v76"], "v76")
    rows = []
    for profile in PROFILES:
        for penalty in PENALTIES:
            prediction, choices = _prediction(frame, caches, profile, penalty, "aligned")
            metric = O._metric(frame, prediction, "gr_selected")
            rows.append({
                "fold": int(fold), "profile": profile, "penalty": float(penalty),
                "changed_wells": int(np.sum(choices["candidate_index"].to_numpy() != 0)),
                **metric,
                "gain": baseline["rmse"] - metric["rmse"],
                "far_gain": baseline["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
                "near_degradation": metric["near_0_500_rmse"] - baseline["near_0_500_rmse"],
                "p95_gain": baseline["p95_well_rmse"] - metric["p95_well_rmse"],
            })
    return pd.DataFrame(rows), baseline


for required in (V88_SOURCE, V86_SOURCE):
    if not required.is_file():
        raise FileNotFoundError(required)

O = _bootstrap(V88_SOURCE, "for required in (V70_PATH, V76_WEIGHTS):", "v88_v89_defs")
M = _bootstrap(V86_SOURCE, "for required in (V70_SOURCE, V85_SOURCE, V70_PATH, V76_WEIGHTS):", "v86_v89_defs")
M.R = M._bootstrap_v70()
M.H = M._bootstrap_v85_helpers()
R = M.R

frame3 = O._fold_frame(3)
caches3 = _well_cache(frame3, 3)
scan3, baseline3 = _scan_fold(frame3, caches3, 3)
scan3.sort_values(["rmse", "profile", "penalty"], kind="stable").to_csv(OUT_DIR / "fold3_scan.csv", index=False)
eligible3 = scan3[
    scan3["gain"].ge(0.10)
    & scan3["far_gain"].ge(0.10)
    & scan3["near_degradation"].le(1e-10)
    & scan3["p95_gain"].ge(0.0)
].copy()

selected = None
confirmation = None
summary_rows = [{"stage": "fold3_selection", **baseline3}]
choices2 = pd.DataFrame()
strict_pass = False
if not eligible3.empty:
    selected = eligible3.sort_values(["rmse", "profile", "penalty"], kind="stable").iloc[0].to_dict()
    pred3, _ = _prediction(frame3, caches3, selected["profile"], selected["penalty"], "aligned")
    summary_rows.append({"stage": "fold3_selection", **O._metric(frame3, pred3, "selected")})

    frame2 = O._fold_frame(2)
    caches2 = _well_cache(frame2, 2)
    aligned2, choices2 = _prediction(frame2, caches2, selected["profile"], selected["penalty"], "aligned")
    rolled2, _ = _prediction(frame2, caches2, selected["profile"], selected["penalty"], "rolled")
    base2 = O._metric(frame2, frame2["v76"], "v76")
    candidate2 = O._metric(frame2, aligned2, "aligned_gr")
    control2 = O._metric(frame2, rolled2, "rolled_gr_control")
    summary_rows.extend([
        {"stage": "fold2_confirmation", **base2},
        {"stage": "fold2_confirmation", **candidate2},
        {"stage": "fold2_confirmation", **control2},
    ])
    confirmation = {
        "gain": base2["rmse"] - candidate2["rmse"],
        "far_gain": base2["far_gt2000_rmse"] - candidate2["far_gt2000_rmse"],
        "near_degradation": candidate2["near_0_500_rmse"] - base2["near_0_500_rmse"],
        "p95_gain": base2["p95_well_rmse"] - candidate2["p95_well_rmse"],
        "aligned_increment_vs_rolled": control2["rmse"] - candidate2["rmse"],
        "changed_wells": int(np.sum(choices2["candidate_index"].to_numpy() != 0)),
    }
    strict_pass = bool(
        confirmation["gain"] >= 0.10
        and confirmation["far_gain"] >= 0.10
        and confirmation["near_degradation"] <= 1e-10
        and confirmation["p95_gain"] >= 0.0
        and confirmation["aligned_increment_vs_rolled"] >= 0.05
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
if not choices2.empty:
    choices2.to_csv(OUT_DIR / "fold2_selected_candidates.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "V88_frozen_deformation_GR_selection_Fold3_selection_Fold2_confirmation",
    "deployable": True,
    "truth_usage": "configuration selection and metrics only",
    "legal_evidence": "future horizontal GR and assigned typewell GR",
    "selection_search_count": len(PROFILES) * len(PENALTIES),
    "selected_recipe": None if selected is None else {
        "profile": selected["profile"], "penalty": selected["penalty"],
        "fold3_gain": selected["gain"], "fold3_far_gain": selected["far_gain"],
    },
    "fold2_confirmation": confirmation,
    "fold1_touched": False,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze recipe and run untouched Fold1 final"
        if strict_pass else "FAIL -> GR cannot select the valuable local deformation; close branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV89 FOLD3 TOP 20")
print(scan3.sort_values("rmse", kind="stable").head(20).to_string(index=False), flush=True)
print("\nV89 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV89 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
