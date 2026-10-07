"""V162: consume frozen V161 emissions with a physical path decoder.

V161 established that real multiscale GR/typewell correspondence has causal
top-K recall but poor rowwise argmax.  This cell does not retrain the matcher.
It reconstructs Fold-1 logits from the frozen checkpoint, selects a first-order
physical DP on 24 development wells, and reports once on the remaining 131
wells.  Truth-dependent top-K projections are oracle diagnostics only.
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numba import njit


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "multiscale_emission_path_ceiling_v162"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_CANDIDATES = (
    ROOT / "target_specific_multiscale_emission_v161_cell.py",
    Path.cwd() / "target_specific_multiscale_emission_v161_cell.py",
)
SOURCE = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
CHECKPOINT = (
    OOF_ROOT / "target_specific_multiscale_emission_v161"
    / "models" / "fold1_gateway_matcher.pt"
)
if SOURCE is None:
    raise FileNotFoundError(
        "Upload target_specific_multiscale_emission_v161_cell.py to /root/autodl-tmp"
    )
if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = "# Execute"
    if marker not in source:
        raise RuntimeError(f"Cannot locate execution marker in {path}")
    name = "_v161_frozen_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    return module


V = _bootstrap(SOURCE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("V162 requires CUDA for frozen V161 inference")


@njit(cache=True)
def _viterbi(emission, states, md, anchor_md, rate_cap, penalty):
    rows, count = emission.shape
    previous = np.empty(count, np.float64)
    first_gap = max(md[0] - anchor_md, 1.0)
    for state in range(count):
        delta = states[state]
        if abs(delta) > rate_cap * first_gap + 4.0:
            previous[state] = 1e30
        else:
            previous[state] = emission[0, state] + penalty * delta * delta / first_gap
    back = np.empty((rows, count), np.int16)
    back[0, :] = -1
    for row in range(1, rows):
        gap = max(md[row] - md[row - 1], 1.0)
        limit = rate_cap * gap + 4.0
        radius = int(np.ceil(limit))
        current = np.full(count, 1e30, np.float64)
        for state in range(count):
            best = 1e30
            best_previous = 0
            lower = max(0, state - radius)
            upper = min(count, state + radius + 1)
            for prior in range(lower, upper):
                delta = states[state] - states[prior]
                if abs(delta) > limit:
                    continue
                value = previous[prior] + penalty * delta * delta / gap
                if value < best:
                    best = value
                    best_previous = prior
            current[state] = best + emission[row, state]
            back[row, state] = best_previous
        previous = current
    indices = np.empty(rows, np.int16)
    indices[-1] = np.argmin(previous)
    for row in range(rows - 1, 0, -1):
        indices[row - 1] = back[row, indices[row]]
    return indices


def _softmax_cost(logits):
    logits = np.asarray(logits, np.float64)
    maximum = np.max(logits, axis=1, keepdims=True)
    shifted = logits - maximum
    return -(shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True)))


def _interpolate(record, sample_rows, sampled_offset):
    anchor_md = float(record.md[record.anchor_index])
    node_md = np.r_[anchor_md, record.md[sample_rows].astype(np.float64)]
    node_offset = np.r_[0.0, np.asarray(sampled_offset, np.float64)]
    eval_md = record.md[record.eval_index].astype(np.float64)
    return np.interp(eval_md, node_md, node_offset)


def _metric(rows, endpoint):
    error = rows[endpoint].to_numpy(np.float64) - rows["truth"].to_numpy(np.float64)
    bywell = pd.Series(error * error).groupby(rows["well"].to_numpy()).mean().pow(0.5)
    return {
        "endpoint": endpoint,
        "rows": len(rows),
        "wells": int(rows["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
    }


checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = V.RealCorrespondenceMatcher().to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

wells = V._horizontal_wells()
folds, fold_path = V._load_folds(wells)
fold1_wells = sorted(folds.loc[folds["fold"].eq(1), "well"].astype(str))
if len(fold1_wells) != 155:
    raise RuntimeError(f"Expected 155 Fold1 wells, found {len(fold1_wells)}")

payload = {}
with torch.no_grad():
    for number, well in enumerate(fold1_wells, 1):
        sample = V._build_well_sample(well, 1)
        query_set = sample.evaluation
        query, reference, valid, _, _, _ = V._to_tensors(
            sample, query_set, DEVICE
        )
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(query, reference, valid).float().cpu().numpy()
        record = V._load_raw_record(well)
        payload[well] = {
            "record": record,
            "rows": query_set.sample_rows.astype(np.int32),
            "labels": query_set.labels.astype(np.float64),
            "logits": logits.astype(np.float32),
        }
        if number % 25 == 0 or number == len(fold1_wells):
            print(f"V162 frozen inference {number}/{len(fold1_wells)}", flush=True)

development = set(fold1_wells[:24])
confirmation = set(fold1_wells[24:])
CONFIGS = tuple(
    (rate_cap, penalty)
    for rate_cap in (0.10, 0.20, 0.35, 0.60)
    for penalty in (0.05, 0.20, 0.80, 3.20)
)


def _decode(well, rate_cap, penalty):
    item = payload[well]
    record = item["record"]
    costs = _softmax_cost(item["logits"])
    md = record.md[item["rows"]].astype(np.float64)
    indices = _viterbi(
        costs,
        V.STATE_OFFSETS.astype(np.float64),
        md,
        float(record.md[record.anchor_index]),
        float(rate_cap),
        float(penalty),
    )
    return _interpolate(record, item["rows"], V.STATE_OFFSETS[indices])


def _oracle(well, topk):
    item = payload[well]
    record = item["record"]
    logits = item["logits"]
    top = np.argpartition(logits, -topk, axis=1)[:, -topk:]
    candidates = V.STATE_OFFSETS[top].astype(np.float64)
    nearest = np.argmin(np.abs(candidates - item["labels"][:, None]), axis=1)
    selected = candidates[np.arange(len(candidates)), nearest]
    return _interpolate(record, item["rows"], selected)


def _frame(selected_wells, rate_cap, penalty, include_oracles=False):
    parts = []
    for well in sorted(selected_wells):
        item = payload[well]
        record = item["record"]
        truth = record.truth[record.eval_index].astype(np.float64)
        values = {
            "well": well,
            "row_idx": record.eval_index,
            "truth": truth,
            "anchor": np.full(len(truth), record.anchor_tvt),
            "dp": record.anchor_tvt + _decode(well, rate_cap, penalty),
        }
        if include_oracles:
            values["oracle_top8"] = record.anchor_tvt + _oracle(well, 8)
            values["oracle_top32"] = record.anchor_tvt + _oracle(well, 32)
        part = pd.DataFrame(values)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


scan_rows = []
for rate_cap, penalty in CONFIGS:
    frame = _frame(development, rate_cap, penalty)
    metric = _metric(frame, "dp")
    scan_rows.append({"rate_cap": rate_cap, "penalty": penalty, **metric})
    print("V162 scan", scan_rows[-1], flush=True)
scan = pd.DataFrame(scan_rows).sort_values(
    ["rmse", "p95_well_rmse"], kind="stable"
).reset_index(drop=True)
scan.to_csv(OUT_DIR / "development_scan.csv", index=False)
selected = scan.iloc[0]

confirmation_frame = _frame(
    confirmation,
    float(selected["rate_cap"]),
    float(selected["penalty"]),
    include_oracles=True,
)
confirmation_frame.to_parquet(OUT_DIR / "fold1_confirmation_paths.parquet", index=False)
summary = pd.DataFrame([
    _metric(confirmation_frame, "anchor"),
    _metric(confirmation_frame, "dp"),
    _metric(confirmation_frame, "oracle_top8"),
    _metric(confirmation_frame, "oracle_top32"),
])
summary.to_csv(OUT_DIR / "confirmation_summary.csv", index=False)
lookup = summary.set_index("endpoint")
dp_rmse = float(lookup.loc["dp", "rmse"])
oracle8 = float(lookup.loc["oracle_top8", "rmse"])
oracle32 = float(lookup.loc["oracle_top32", "rmse"])
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold1_24well_DP_selection_131well_confirmation",
    "source_checkpoint": str(CHECKPOINT),
    "fold_source": str(fold_path),
    "development_wells": 24,
    "confirmation_wells": 131,
    "search_count": len(CONFIGS),
    "selected_recipe": {
        "rate_cap": float(selected["rate_cap"]),
        "penalty": float(selected["penalty"]),
    },
    "summary": summary.to_dict(orient="records"),
    "checks": {
        "dp_rmse_le_10p5": dp_rmse <= 10.5,
        "top8_oracle_rmse_le_6": oracle8 <= 6.0,
        "top32_oracle_rmse_le_4p5": oracle32 <= 4.5,
    },
    "investment_pass": bool(dp_rmse <= 10.5 and oracle32 <= 4.5),
    "next": (
        "PASS -> build k-best second-order complete trajectories"
        if dp_rmse <= 10.5 and oracle32 <= 4.5
        else "FAIL -> frozen emissions lack the path-support ceiling required for CV<6"
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2), encoding="utf-8"
)
print("V162 DEVELOPMENT TOP")
print(scan.head(12).to_string(index=False), flush=True)
print("V162 CONFIRMATION SUMMARY")
print(summary.to_string(index=False), flush=True)
print("V162 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
