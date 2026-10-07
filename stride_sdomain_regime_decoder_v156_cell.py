"""V156: S-domain regime-switching STRIDE decoder.

The decoded state is S = TVT + Z.  Segment rates therefore describe layer
dip in the structural domain instead of directly accumulating TVT drift.
Transitions allow a flat regime, a persistent jump regime, and a short
recovery regime.  Priors are fitted from the other four folds only.
"""
from __future__ import annotations

import ast
import json
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
import pyarrow.parquet as pq

EXPERIMENT = "stride_sdomain_regime_decoder_v156"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
STARTED = time.time()

SOURCE_CANDIDATES = (
    ROOT / "stride_joint_segment_decoder_gateway_v1_cell.py",
    Path.cwd() / "stride_joint_segment_decoder_gateway_v1_cell.py",
)
SOURCE = next((p for p in SOURCE_CANDIDATES if p.exists()), None)
if SOURCE is None:
    raise FileNotFoundError("Upload stride_joint_segment_decoder_gateway_v1_cell.py")


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    cutoff = next((i for i, node in enumerate(tree.body)
                   if isinstance(node, ast.If)
                   and "FOLD_PATH is None" in (ast.unparse(node.test) if hasattr(ast, "unparse") else "")), None)
    if cutoff is None:
        raise RuntimeError("Cannot locate STRIDE definition boundary")
    class DisableCache(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "njit":
                    for kw in dec.keywords:
                        if kw.arg == "cache":
                            kw.value = ast.Constant(False)
                    if not any(kw.arg == "cache" for kw in dec.keywords):
                        dec.keywords.append(ast.keyword(arg="cache", value=ast.Constant(False)))
            return node
    tree = DisableCache().visit(tree)
    ast.fix_missing_locations(tree)
    module = types.ModuleType("stride_v156_defs")
    module.__file__ = str(path)
    exec(compile(ast.Module(body=tree.body[:cutoff], type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


V = _bootstrap(SOURCE)
FOLD_PATH = V.FOLD_PATH
_load_record = V._load_record
_node_rows = V._node_rows
DATA_ROOT = V.DATA_ROOT

BASE_CANDIDATES = (
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
    OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet",
)
BASE_PATH = next((p for p in BASE_CANDIDATES if p.exists()), None)
if BASE_PATH is None:
    raise FileNotFoundError(BASE_CANDIDATES)

NODE_STEP = 32.0
LENGTH_NODES = np.asarray((1, 2, 3, 5, 7, 10, 16), np.int64)
DELTA_MULTIPLIERS = np.asarray((-2.0, -1.0, 0.0, 1.0, 2.0), np.float64)
BEAM = 48
SAMPLES = 24
PERSISTENCE_WEIGHT = 0.8
LENGTH_WEIGHT = 0.20
JUMP_SCALE_MULTIPLIER = 1.0
RECOVERY_MULTIPLIER = 0.35


def _json_default(v):
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v) if np.isfinite(v) else None
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, Path): return str(v)
    raise TypeError(type(v).__name__)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _folds():
    frame = pd.read_csv(FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    counts = frame.groupby("fold")["well"].nunique().to_dict()
    if set(counts) != {1, 2, 3, 4, 5} or min(counts.values()) < 150:
        raise RuntimeError(f"Unexpected fold coverage {counts}")
    return frame


def _load_base():
    names = pq.ParquetFile(BASE_PATH).schema.names
    column = next((x for x in ("incumbent_6p377_replay", "current_dose_0p20", "full_cp8_ordered") if x in names), None)
    if column is None:
        raise RuntimeError("No deployed baseline column")
    frame = pd.read_parquet(BASE_PATH, columns=["well", "row_idx", "tvt_true", "md_since", column])
    frame["well"] = frame["well"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(int)
    return frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True), column


def _s_profile(record):
    eval_rows = record["evaluation"]
    index = np.r_[record["anchor"], eval_rows]
    md = record["md"][index]
    s = np.r_[record["anchor_tvt"] + record["z"][record["anchor"]], record["truth"][eval_rows] + record["z"][eval_rows]]
    targets = np.arange(md[0], md[-1] + 0.5 * NODE_STEP, NODE_STEP)
    if targets[-1] < md[-1] - 1e-6:
        targets = np.r_[targets, md[-1]]
    return targets, np.interp(targets, md, s)


def _fit_priors(records):
    slopes = []
    for record in records:
        md, s = _s_profile(record)
        local = np.diff(s) / np.maximum(np.diff(md), 1e-6)
        local = pd.Series(local).rolling(3, center=True, min_periods=1).median().to_numpy(float)
        slopes.append(local)
    values = np.concatenate(slopes)
    low, high = np.quantile(values, [0.005, 0.995])
    changes = np.concatenate([np.diff(x) for x in slopes if len(x) > 1])
    scale = float(max(1.4826 * np.median(np.abs(changes - np.median(changes))), 0.25 * np.std(changes), 0.0025))
    return {"rate_low": float(low - 0.25 * (high - low)), "rate_high": float(high + 0.25 * (high - low)), "change_scale": scale, "well_count": len(records), "slope_count": len(values)}


@njit(cache=False)
def _interp(axis, values, query):
    if query <= axis[0]: return values[0]
    if query >= axis[-1]: return values[-1]
    left, right = 0, len(axis) - 1
    while right - left > 1:
        mid = (left + right) // 2
        if axis[mid] <= query: left = mid
        else: right = mid
    frac = (query - axis[left]) / max(axis[right] - axis[left], 1e-9)
    return values[left] * (1.0 - frac) + values[right] * frac


@njit(cache=False)
def _decode_s(md, z, gr, nodes, anchor_s, initial_rate, tw_tvt, tw_gr, gr_scale, rate_low, rate_high, change_scale, mode):
    n = len(nodes)
    costs = np.full((n, BEAM), 1e30)
    state_s = np.zeros((n, BEAM))
    state_rate = np.zeros((n, BEAM))
    parent_node = np.full((n, BEAM), -1, np.int64)
    parent_slot = np.full((n, BEAM), -1, np.int64)
    costs[0, 0] = 0.0
    state_s[0, 0] = anchor_s
    state_rate[0, 0] = min(max(initial_rate, rate_low), rate_high)
    for target in range(1, n):
        for h in LENGTH_NODES:
            source = target - h
            if source < 0: continue
            length = md[nodes[target]] - md[nodes[source]]
            if length <= 0: continue
            length_cost = LENGTH_WEIGHT * (np.log(max(length, 1.0)) - np.log(160.0)) ** 2
            for slot0 in range(BEAM):
                if costs[source, slot0] >= 1e29: continue
                old_s = state_s[source, slot0]
                old_rate = state_rate[source, slot0]
                for jump in DELTA_MULTIPLIERS:
                    new_rate = old_rate + jump * change_scale * JUMP_SCALE_MULTIPLIER
                    new_rate = min(max(new_rate, rate_low), rate_high)
                    new_s = old_s + new_rate * length
                    emission = 0.0
                    count = nodes[target] - nodes[source]
                    samples = min(count, SAMPLES)
                    for k in range(samples):
                        row = nodes[source] + 1 + ((k + 1) * count - 1) // samples
                        frac = (md[row] - md[nodes[source]]) / max(length, 1e-9)
                        candidate_s = old_s + frac * (new_s - old_s)
                        candidate_tvt = candidate_s - z[row]
                        expected = _interp(tw_tvt, tw_gr, candidate_tvt)
                        residual = (gr[row] - expected) / max(gr_scale, 1e-6)
                        if mode == 1: emission += 0.5 * min(residual * residual, 400.0)
                        else: emission += np.log1p(min(residual * residual, 1e4))
                    emission *= count / max(samples, 1)
                    transition_weight = RECOVERY_MULTIPLIER if mode == 2 else PERSISTENCE_WEIGHT
                    transition = transition_weight * 0.5 * jump * jump
                    value = costs[source, slot0] + emission + transition + length_cost
                    worst = 0
                    for slot in range(1, BEAM):
                        if costs[target, slot] > costs[target, worst]: worst = slot
                    if value < costs[target, worst]:
                        costs[target, worst] = value
                        state_s[target, worst] = new_s
                        state_rate[target, worst] = new_rate
                        parent_node[target, worst] = source
                        parent_slot[target, worst] = slot0
    best = 0
    for slot in range(1, BEAM):
        if costs[n - 1, slot] < costs[n - 1, best]: best = slot
    path = np.full(n, np.nan)
    node, slot = n - 1, best
    while node >= 0 and slot >= 0:
        path[node] = state_s[node, slot]
        previous_node = parent_node[node, slot]
        previous_slot = parent_slot[node, slot]
        if previous_node < 0: break
        left_s, right_s = state_s[previous_node, previous_slot], state_s[node, slot]
        left_md, right_md = md[nodes[previous_node]], md[nodes[node]]
        for fill in range(previous_node + 1, node):
            frac = (md[nodes[fill]] - left_md) / max(right_md - left_md, 1e-9)
            path[fill] = left_s + frac * (right_s - left_s)
        node, slot = previous_node, previous_slot
    return path, costs[n - 1, best]


def _decode(record, priors, mode):
    nodes = _node_rows(record)
    anchor_s = record["anchor_tvt"] + record["z"][record["anchor"]]
    known = record["known"]
    recent = known[record["md"][known] >= record["md"][record["anchor"]] - 500.0]
    if len(recent) >= 8 and np.ptp(record["md"][recent]) >= 50.0:
        initial_rate = float(np.polyfit(
            record["md"][recent],
            record["tvt_input"][recent] + record["z"][recent],
            1,
        )[0])
    else:
        initial_rate = 0.5 * (priors["rate_low"] + priors["rate_high"])
    path_s, objective = _decode_s(record["md"], record["z"], record["gr"], nodes, anchor_s, initial_rate, record["tw_tvt"], record["tw_gr_calibrated"], record["gr_scale"], priors["rate_low"], priors["rate_high"], priors["change_scale"], mode)
    if not np.isfinite(path_s).all(): raise RuntimeError(f"{record['well']}: no terminal S path")
    rows = record["evaluation"]
    s = np.interp(record["md"][rows], record["md"][nodes], path_s)
    return s - record["z"][rows], float(objective)


def _metrics(frame, column):
    e = frame[column].to_numpy(float) - frame["tvt_true"].to_numpy(float)
    d = frame["md_since"].to_numpy(float)
    result = {"rows": len(e), "wells": frame["well"].nunique(), "exact_sse": float(np.dot(e, e)), "rmse": float(np.sqrt(np.mean(e * e))), "mae": float(np.mean(np.abs(e)))}
    wr = pd.Series(e * e).groupby(frame["well"].to_numpy()).mean().pow(0.5)
    result["median_well_rmse"], result["p95_well_rmse"] = float(wr.median()), float(wr.quantile(0.95))
    for name, lo, hi in (("near_0_500", 0, 500), ("mid_500_1000", 500, 1000), ("mid_1000_2000", 1000, 2000), ("far_gt2000", 2000, np.inf)):
        mask = (d >= lo) & (d < hi)
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(e[mask] ** 2))) if mask.any() else np.nan
    return result


folds = _folds()
base, base_column = _load_base()
rows = []
for heldout in (1, 2, 3, 4, 5):
    train_wells = folds.loc[folds.fold.ne(heldout), "well"].tolist()
    valid_wells = folds.loc[folds.fold.eq(heldout), "well"].tolist()
    train_records = [_load_record(well) for well in train_wells]
    priors = _fit_priors(train_records)
    for mode in (0, 1, 2):
        parts = []
        for number, well in enumerate(valid_wells, 1):
            record = _load_record(well)
            prediction, objective = _decode(record, priors, mode)
            evaluation = record["evaluation"]
            parts.append(pd.DataFrame({"well": well, "row_idx": evaluation.astype(int), "prediction": prediction, "fold": heldout, "mode": mode, "objective": objective}))
            if number % 25 == 0 or number == len(valid_wells): print(f"V156 fold{heldout} mode{mode} {number}/{len(valid_wells)}", flush=True)
        pred = pd.concat(parts, ignore_index=True)
        joined = pred.merge(base[["well", "row_idx", "tvt_true", "md_since"]], on=["well", "row_idx"], validate="one_to_one")
        rows.append({"fold": heldout, "mode": mode, **_metrics(joined, "prediction")})

component = pd.DataFrame(rows)
component["mode_name"] = component["mode"].map({0: "full_cauchy_persistent", 1: "gaussian_persistent", 2: "cauchy_spike_recovery"})
component.to_csv(OUT_DIR / "component_by_fold.csv", index=False)
full = component.loc[component["mode"].eq(0)]
summary = component.groupby(["mode", "mode_name"], as_index=False).sum(numeric_only=True)
summary["rmse"] = np.sqrt(summary["exact_sse"] / summary["rows"])
summary.to_csv(OUT_DIR / "summary.csv", index=False)
verdict = {"experiment": EXPERIMENT, "stage": "fivefold_S_domain_regime_decoder", "base": str(BASE_PATH), "decoder": {"state": "S=TVT+Z", "regimes": ["flat", "persistent_jump", "spike_recovery"], "likelihood": "Cauchy", "typewell_calibration": "robust_affine"}, "coverage": {str(k): int(v) for k, v in folds.groupby("fold").size().items()}, "summary": summary.to_dict(orient="records"), "gates": {"capacity_and_confirmation_required": True}, "next": "Review component_by_fold.csv; do not blend until standalone RMSE and fold stability are confirmed", "elapsed_minutes": (time.time() - STARTED) / 60.0}
_write_json(OUT_DIR / "verdict.json", verdict)
print("V156 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("V156 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
