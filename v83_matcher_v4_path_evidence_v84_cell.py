"""Project the frozen V83 matcher posterior onto complete V4 lattice paths.

This is an attribution gateway, not a deployment model.  The V83 baseline
matcher has causal local correspondence signal but poor pointwise argmax.  We
therefore ask whether its state posterior can rank already-existing complete
paths.  Fold 2 is split deterministically into 24 discovery wells and a
confirmation remainder.  Fold 1 remains untouched.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("Install lightgbm before running V84") from exc


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "v83_matcher_v4_path_evidence_v84"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V4_SOURCE_CANDIDATES = (
    ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py",
    Path("rawdata_lattice_coarse_ranker_v4_cell.py"),
)
MATCHER_SOURCE_CANDIDATES = (
    ROOT / "real_correspondence_multiscale_matcher_gateway_v0_cell.py",
    Path("real_correspondence_multiscale_matcher_gateway_v0_cell.py"),
)
V4_SOURCE = next((path for path in V4_SOURCE_CANDIDATES if path.is_file()), None)
MATCHER_SOURCE = next((path for path in MATCHER_SOURCE_CANDIDATES if path.is_file()), None)
if V4_SOURCE is None or MATCHER_SOURCE is None:
    raise FileNotFoundError(
        "Upload rawdata_lattice_coarse_ranker_v4_cell.py and "
        "real_correspondence_multiscale_matcher_gateway_v0_cell.py to /root/autodl-tmp"
    )

V83_DIR = OOF_ROOT / "competition_nuisance_matcher_ab_v83"
V4_MODEL_CANDIDATES = (
    OOF_ROOT / "rawdata_lattice_coarse_canonical_gateway_v67" / "models"
    / "coarse_ranker_canonical345.txt",
    OOF_ROOT / "rawdata_lattice_coarse_canonical_final_bundle_v68" / "deployment"
    / "coarse_ranker_canonical345_validated.txt",
    OOF_ROOT / "rawdata_lattice_coarse_ranker_v4" / "coarse_ranker_outer345.txt",
)
V4_MODEL_PATH = next((path for path in V4_MODEL_CANDIDATES if path.is_file()), None)
V83_MODEL_PATH = V83_DIR / "models" / "baseline_outer345.pt"
if V4_MODEL_PATH is None:
    raise FileNotFoundError(
        "No fold-clean Fold345 coarse ranker found. Checked: "
        + repr([str(path) for path in V4_MODEL_CANDIDATES])
    )
if not V83_MODEL_PATH.is_file():
    raise FileNotFoundError(V83_MODEL_PATH)

SEED = 20260801
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 24
SEGMENT_LENGTH_FT = 128.0
WEIGHTS = np.asarray([0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0])
AGGREGATORS = ("mean", "deep_weighted", "lower_quartile")


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _load_v4_definitions(path):
    source = path.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V4 marker missing from {path}")
    namespace = {"__name__": "v4_path_evidence_definitions", "__file__": str(path)}
    exec(compile(source.split(marker, 1)[0], str(path), "exec"), namespace)
    return namespace


def _load_matcher_definitions(path):
    source = path.read_text(encoding="utf-8")
    marker = "started = time.time()"
    if marker not in source:
        raise RuntimeError(f"Matcher execution marker missing from {path}")
    name = "v84_matcher_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    return module.__dict__


V4 = _load_v4_definitions(V4_SOURCE)
M = _load_matcher_definitions(MATCHER_SOURCE)


def _stable_key(well):
    return int.from_bytes(hashlib.sha256(f"v84:{well}".encode("ascii")).digest()[:8], "little")


def _robust_z(values):
    values = np.asarray(values, np.float64)
    centre = float(np.nanmedian(values))
    scale = float(1.4826 * np.nanmedian(np.abs(values - centre)))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return np.clip((values - centre) / scale, -12.0, 12.0)


def _metrics(frame, prefix):
    rows = int(frame["rows"].sum())
    far_rows = int(frame["far_rows"].sum())
    sse = float(frame[f"{prefix}_sse"].sum())
    well_rmse = np.sqrt(frame[f"{prefix}_sse"] / frame["rows"])
    return {
        "rows": rows,
        "wells": int(frame["well"].nunique()),
        "exact_sse": sse,
        "rmse": math.sqrt(sse / rows),
        "mae": float(frame[f"{prefix}_sae"].sum() / rows),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": math.sqrt(float(frame[f"{prefix}_far_sse"].sum()) / far_rows),
    }


def _candidate_offsets(record, prior, paths, segments, sample_rows):
    output = np.empty((len(paths), len(sample_rows)), np.float32)
    for index, path in enumerate(paths):
        prediction = V4["B"]["_path_prediction"](record, prior, path, segments)
        output[index] = prediction[sample_rows] - float(record["anchor_tvt"])
    return output


def _project(log_probability, offsets, state_valid):
    state_half = int(M["STATE_HALF_WIDTH_FT"])
    index = np.rint(offsets + state_half).astype(np.int64)
    in_axis = (index >= 0) & (index < log_probability.shape[1])
    safe = np.clip(index, 0, log_probability.shape[1] - 1)
    row = np.arange(log_probability.shape[0], dtype=np.int64)[None, :]
    score = log_probability[row, safe]
    valid = in_axis & state_valid[safe]
    score = np.where(valid, score, -20.0)
    return score, valid


def _aggregate(score, valid, deep_weight):
    safe = np.where(valid, score, np.nan)
    mean = np.nanmean(safe, axis=1)
    weights = np.asarray(deep_weight, np.float64)
    weighted_num = np.nansum(safe * weights[None, :], axis=1)
    weighted_den = np.sum(np.isfinite(safe) * weights[None, :], axis=1)
    deep = weighted_num / np.maximum(weighted_den, 1e-8)
    lower = np.nanquantile(safe, 0.25, axis=1)
    coverage = np.mean(valid, axis=1)
    outputs = {
        "mean": np.nan_to_num(mean, nan=-20.0),
        "deep_weighted": np.nan_to_num(deep, nan=-20.0),
        "lower_quartile": np.nan_to_num(lower, nan=-20.0),
    }
    return outputs, coverage


@torch.no_grad()
def _matcher_log_probability(model, sample, device, state_roll=False):
    query_set = sample.evaluation
    query, reference, valid, _, _, _ = M["_to_tensors"](
        sample, query_set, device
    )
    model.eval()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(query, reference, valid)
    logp = torch.log_softmax(logits.float(), dim=1).cpu().numpy()
    if state_roll:
        rng = np.random.default_rng(SEED + _stable_key(sample.well) % 1000000)
        rolled = np.empty_like(logp)
        for row in range(len(logp)):
            shift = int(rng.integers(17, logp.shape[1] - 16))
            rolled[row] = np.roll(logp[row], shift)
        logp = rolled
    return logp


def _choice_rows(cache, wells, aggregator, weight, control):
    rows = []
    for well in wells:
        item = cache[well]
        matcher = item[f"{control}_{aggregator}"]
        combined = (1.0 - weight) * item["base_z"] + weight * _robust_z(matcher)
        chosen = int(np.argmax(combined))
        base = int(np.argmax(item["base_score"]))
        oracle = int(np.argmin(item["panel"]["mse"]))
        row = {
            "well": well,
            "rows": int(item["panel"]["rows"][0]),
            "far_rows": int(item["panel"]["far_rows"][0]),
            "matcher_coverage": float(np.mean(item["coverage"])),
        }
        for prefix, index in (("base", base), ("candidate", chosen), ("oracle", oracle)):
            row[f"{prefix}_sse"] = float(item["panel"]["sse"][index])
            row[f"{prefix}_sae"] = float(item["panel"]["sae"][index])
            row[f"{prefix}_far_sse"] = float(item["panel"]["far_sse"][index])
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame, {
        "base": _metrics(frame, "base"),
        "candidate": _metrics(frame, "candidate"),
        "oracle": _metrics(frame, "oracle"),
    }


started = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("V84 requires CUDA for the frozen V83 matcher")

wells_all = [
    path.name.split("__", 1)[0]
    for path in sorted((V4["DATA_ROOT"] / "train").glob("*__horizontal_well.csv"))
]
folds = V4["V"]["_stable_folds"](wells_all)
fold_by_well = dict(zip(folds["well"].astype(str), folds["fold"].astype(int)))
prior_wells = [well for well in wells_all if fold_by_well[well] in (4, 5)]
application_wells = [well for well in wells_all if fold_by_well[well] == APPLICATION_FOLD]

records = {}
failures = []
for number, well in enumerate(prior_wells + application_wells, 1):
    try:
        records[well] = V4["V"]["_load_record"](well)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(prior_wells) + len(application_wells):
        print(f"V84 lattice records {number}/{len(prior_wells)+len(application_wells)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)
prior_wells = [well for well in prior_wells if well in records]
application_wells = [well for well in application_wells if well in records]
if len(prior_wells) < 307 or len(application_wells) < 154:
    raise RuntimeError(f"Insufficient V84 coverage: prior={len(prior_wells)} application={len(application_wells)}")

prior = V4["V"]["_fit_prior"]([records[well] for well in prior_wells], SEGMENT_LENGTH_FT)
coarse = lgb.Booster(model_file=str(V4_MODEL_PATH))
checkpoint = torch.load(V83_MODEL_PATH, map_location="cpu", weights_only=False)
matcher = M["RealCorrespondenceMatcher"]().to(device)
matcher.load_state_dict(checkpoint["model"], strict=True)

# The source builder creates evaluation sets only for TEST_FOLD.  Rebind it to
# Fold 2; this changes sampling eligibility, not any model parameter.
M["TEST_FOLD"] = APPLICATION_FOLD
M["TUNE_FOLD"] = 3

cache = {}
build_failures = []
for number, well in enumerate(application_wells, 1):
    try:
        record = records[well]
        # Old V4 panels may have been removed during disk cleanup.  Their
        # generation is deterministic under the Fold4/5 prior fitted above.
        if not V4["_panel_path"](well).is_file():
            V4["_build_panel"](record, prior)
        panel = V4["_load_panel"](well)
        paths, _, arm_rank, _, segments, _ = V4["_candidate_union"](record, prior)
        if len(paths) != len(panel["mse"]):
            raise RuntimeError(f"panel/path count mismatch {len(panel['mse'])}/{len(paths)}")
        if not np.allclose(arm_rank, panel["arm_rank"], equal_nan=True, atol=1e-7):
            raise RuntimeError("candidate order does not match frozen panel")

        sample = M["_build_well_sample"](well, APPLICATION_FOLD)
        if sample.evaluation is None:
            raise RuntimeError("matcher evaluation sample missing")
        logp = _matcher_log_probability(matcher, sample, device, state_roll=False)
        shuffled_logp = _matcher_log_probability(matcher, sample, device, state_roll=True)
        offsets = _candidate_offsets(
            record, prior, paths, segments, sample.evaluation.sample_rows
        )
        aligned_score, aligned_valid = _project(logp, offsets, sample.state_valid)
        shuffled_score, shuffled_valid = _project(shuffled_logp, offsets, sample.state_valid)
        distance = (
            M["_load_raw_record"](well).md[sample.evaluation.sample_rows]
            - M["_load_raw_record"](well).md[M["_load_raw_record"](well).anchor_index]
        )
        deep_weight = np.clip(distance / 1500.0, 0.5, 2.0)
        aligned, coverage = _aggregate(aligned_score, aligned_valid, deep_weight)
        shuffled, _ = _aggregate(shuffled_score, shuffled_valid, deep_weight)
        base_score = coarse.predict(panel["features"])
        item = {
            "panel": panel,
            "base_score": np.asarray(base_score, np.float64),
            "base_z": _robust_z(base_score),
            "coverage": coverage,
        }
        for aggregator in AGGREGATORS:
            item[f"aligned_{aggregator}"] = aligned[aggregator]
            item[f"state_roll_{aggregator}"] = shuffled[aggregator]
        cache[well] = item
    except Exception as exc:
        build_failures.append({"well": well, "reason": repr(exc)})
    if number % 10 == 0 or number == len(application_wells):
        print(f"V84 project {number}/{len(application_wells)} success={len(cache)} failures={len(build_failures)}", flush=True)

pd.DataFrame(build_failures).to_csv(OUT_DIR / "projection_failures.csv", index=False)
application_wells = [well for well in application_wells if well in cache]
if len(application_wells) < 145:
    raise RuntimeError(f"Only {len(application_wells)} projected wells: {build_failures[:5]}")

ordered = sorted(application_wells, key=_stable_key)
discovery = ordered[: min(DISCOVERY_WELLS, len(ordered) // 4)]
confirmation = [well for well in ordered if well not in set(discovery)]

scan_rows = []
for aggregator in AGGREGATORS:
    for weight in WEIGHTS:
        _, report = _choice_rows(cache, discovery, aggregator, float(weight), "aligned")
        scan_rows.append({
            "aggregator": aggregator,
            "weight": float(weight),
            "base_rmse": report["base"]["rmse"],
            "rmse": report["candidate"]["rmse"],
            "gain": report["base"]["rmse"] - report["candidate"]["rmse"],
            "far_gain": report["base"]["far_gt2000_rmse"] - report["candidate"]["far_gt2000_rmse"],
            "p95_change": report["candidate"]["p95_well_rmse"] - report["base"]["p95_well_rmse"],
        })
scan = pd.DataFrame(scan_rows)
safe = scan[scan["p95_change"] <= 0.5]
if not len(safe):
    safe = scan
selected = safe.sort_values(["rmse", "weight", "aggregator"]).iloc[0]
aggregator = str(selected["aggregator"])
weight = float(selected["weight"])
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
print("\nV84 DISCOVERY SCAN")
print(scan.sort_values(["rmse", "weight"]).head(20).to_string(index=False), flush=True)
print("V84 SELECTED", {"aggregator": aggregator, "weight": weight}, flush=True)

aligned_frame, aligned = _choice_rows(cache, confirmation, aggregator, weight, "aligned")
shuffle_frame, shuffled = _choice_rows(cache, confirmation, aggregator, weight, "state_roll")
aligned_frame.to_csv(OUT_DIR / "confirmation_bywell.csv", index=False)
shuffle_frame.to_csv(OUT_DIR / "state_roll_bywell.csv", index=False)

gain = aligned["base"]["rmse"] - aligned["candidate"]["rmse"]
far_gain = aligned["base"]["far_gt2000_rmse"] - aligned["candidate"]["far_gt2000_rmse"]
shuffle_increment = shuffled["candidate"]["rmse"] - aligned["candidate"]["rmse"]
p95_change = aligned["candidate"]["p95_well_rmse"] - aligned["base"]["p95_well_rmse"]
checks = {
    "confirmation_wells_ge_115": len(confirmation) >= 115,
    "confirmation_gain_ge_0p20": gain >= 0.20,
    "confirmation_far_gain_ge_0p20": far_gain >= 0.20,
    "state_roll_loses_ge_0p15": shuffle_increment >= 0.15,
    "p95_change_le_0p20": p95_change <= 0.20,
    "bank_oracle_rmse_le_5p8": aligned["oracle"]["rmse"] <= 5.8,
}
strict_pass = all(checks.values())
verdict = {
    "experiment": EXPERIMENT,
    "stage": "frozen_v83_state_posterior_to_v4_complete_path_attribution",
    "sample_unit": "complete physical well and real V4 candidate paths",
    "protocol": {
        "prior_folds": [4, 5],
        "matcher_train_folds": checkpoint.get("train_folds", [3, 4, 5]),
        "application_fold": APPLICATION_FOLD,
        "discovery_wells": len(discovery),
        "confirmation_wells": len(confirmation),
        "fold1_touched": False,
    },
    "sources": {
        "v4_model": str(V4_MODEL_PATH),
        "v83_matcher": str(V83_MODEL_PATH),
    },
    "selected": {"aggregator": aggregator, "weight": weight},
    "confirmation": {
        "base": aligned["base"],
        "candidate": aligned["candidate"],
        "oracle": aligned["oracle"],
        "state_roll_candidate": shuffled["candidate"],
        "gain": gain,
        "far_gain": far_gain,
        "state_roll_increment": shuffle_increment,
        "p95_change": p95_change,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> add frozen matcher path summaries to HN pairwise features"
        if strict_pass
        else "FAIL -> local matcher signal does not survive complete-path aggregation; close learned correspondence"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV84 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)

del matcher
gc.collect()
torch.cuda.empty_cache()
