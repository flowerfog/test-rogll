"""M342-X4: whole-well endpoint selection with frozen 256-row GR evidence.

X2 established that 256-row amplitude scores improve local candidate ranks,
while X3 showed that unconstrained residual-path decoding is unstable. X4 uses
the same evidence only to score the existing legal endpoint bank. The first 24
Fold3 wells select a bounded aggregation and dose; the remaining Fold3 wells
confirm with a matched-range typewell shuffle and reverse-direction control.
"""

from pathlib import Path
import json
import sys
import time
import types

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
P1_SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
X1_SOURCE = ROOT / "main342_multiscale_rank_consistency_x1.py"
OUT = ROOT / "oof_exports" / "main342_longwindow_endpoint_selector_x4"
OUT.mkdir(parents=True, exist_ok=True)

FOLD = 3
DISCOVERY_WELLS = 24
HALF_WINDOW = 256
STATIONS_PER_WELL = 64
FORMULAS = ("winner", "top2_equal", "soft_t0p5", "soft_t1")
DOSES = (0.0, 0.05, 0.10, 0.20, 0.30)
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
    return value


def _bootstrap(path, marker, name):
    source = path.read_text(encoding="utf-8")
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


P = _bootstrap(
    P1_SOURCE,
    'base, manifest, base_features, coverage = O["_prepare_inventory"]()',
    "m342_x4_p1_definitions",
)
X = _bootstrap(
    X1_SOURCE,
    "fold3_wells = sorted(",
    "m342_x4_x1_definitions",
)

G = P["K"]["_load_g1"]()
inventory = G["_load_inventory"]()
frame = inventory["base"].reset_index(drop=True).copy()
bank = np.asarray(inventory["matrix"], np.float64)
bank_names = list(map(str, inventory["names"]))
if any(any(token in name.lower() for token in ("oracle", "truth", "gold")) for name in bank_names):
    raise RuntimeError("Endpoint bank contains a forbidden diagnostic endpoint")

k2_table = P["_endpoint_table"]()[["id", "k2_parent"]]
k2 = frame[["id"]].merge(k2_table, on="id", how="left", validate="one_to_one")[
    "k2_parent"
].to_numpy(np.float64)
if not np.isfinite(k2).all():
    raise RuntimeError("K2 endpoint alignment is incomplete")
bank = np.vstack([bank, k2])
bank_names.append("k2_parent")

frame["well"] = frame["well"].astype(str).str.lower()
if "row_idx" not in frame:
    frame["row_idx"] = frame["id"].astype(str).str.rsplit("_", n=1).str[1].astype(int)
DATA_ROOT = X["DATA_ROOT"]
METRICS = P["B"]["_metrics"]


def _robust_scores(values):
    values = np.asarray(values, np.float64)
    centre = float(np.median(values))
    scale = float(np.median(np.abs(values - centre)) * 1.4826)
    if scale < 1e-6:
        scale = max(float(np.std(values)), 1e-6)
    return (values - centre) / scale


def _endpoint_from_scores(local_bank, scores, formula):
    order = np.argsort(-scores, kind="stable")
    if formula == "winner":
        return local_bank[order[0]]
    if formula == "top2_equal":
        return np.mean(local_bank[order[:2]], axis=0)
    if formula == "soft_t0p5":
        temperature = 0.5
    elif formula == "soft_t1":
        temperature = 1.0
    else:
        raise ValueError(formula)
    z = _robust_scores(scores)
    weight = np.exp(np.clip((z - np.max(z)) / temperature, -50.0, 0.0))
    weight /= np.sum(weight)
    return weight @ local_bank


def _score_well(well, local, local_bank, typewell_override=None):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    type_name = typewell_override or well
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{type_name}__typewell.csv")
    n_rows = len(horizontal)
    row_idx = local["row_idx"].to_numpy(np.int64)
    if row_idx.min() < 0 or row_idx.max() >= n_rows:
        raise RuntimeError(f"{well}: row index outside horizontal file")

    prefix = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    full_paths = np.repeat(prefix[None, :], len(local_bank), axis=0)
    full_paths[:, row_idx] = local_bank
    for index in range(len(full_paths)):
        filled = X["H"]["_fill"](full_paths[index])
        if filled is None:
            raise RuntimeError(f"{well}: cannot complete endpoint {index}")
        full_paths[index] = filled

    hgr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    tw_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64)
    tw_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64)
    good = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    if good.sum() < 64:
        raise RuntimeError(f"{type_name}: too few typewell rows")
    order = np.argsort(tw_tvt[good])
    tw_tvt, tw_gr = tw_tvt[good][order], tw_gr[good][order]

    stations = X["H"]["_sample_positions"](row_idx, STATIONS_PER_WELL)
    station_scores = []
    for centre in stations:
        left = max(0, int(centre) - HALF_WINDOW)
        right = min(n_rows, int(centre) + HALF_WINDOW + 1)
        observed = hgr[left:right]
        if len(observed) < HALF_WINDOW or np.isfinite(observed).mean() < 0.35:
            continue
        observed = X["H"]["_fill"](observed)
        if observed is None:
            continue
        candidate_gr = np.interp(
            full_paths[:, left:right].ravel(), tw_tvt, tw_gr, left=np.nan, right=np.nan
        ).reshape(len(local_bank), -1)
        valid = np.isfinite(candidate_gr).mean(axis=1) >= 0.95
        if valid.sum() != len(local_bank):
            continue
        medians = np.nanmedian(candidate_gr, axis=1, keepdims=True)
        candidate_gr = np.where(np.isfinite(candidate_gr), candidate_gr, medians)
        station_scores.append(
            X["H"]["_components"](observed, candidate_gr)["amp"]
        )
    if len(station_scores) < 12:
        raise RuntimeError(f"{well}: only {len(station_scores)} common endpoint stations")
    return np.mean(np.asarray(station_scores, np.float64), axis=0)


def _prepare(wells, shuffle_map=None, label="aligned"):
    output, failures = {}, []
    positions_by_well = frame.groupby("well", sort=False).indices
    for number, well in enumerate(wells, 1):
        positions = np.asarray(positions_by_well[well], np.int64)
        local = frame.iloc[positions].sort_values("row_idx").copy()
        reorder = local.index.to_numpy(np.int64)
        local_bank = bank[:, reorder]
        try:
            override = shuffle_map.get(well) if shuffle_map else None
            scores = _score_well(well, local, local_bank, override)
            output[well] = (local, local_bank, scores)
        except Exception as exc:
            failures.append({"well": well, "control": label, "reason": repr(exc)})
        if number % 25 == 0 or number == len(wells):
            print(
                f"M342-X4 {label} {number}/{len(wells)} success={len(output)} "
                f"failures={len(failures)}", flush=True
            )
    return output, failures


def _assemble(prepared, formula):
    tables, parents, endpoints = [], [], []
    for well in sorted(prepared):
        local, local_bank, scores = prepared[well]
        tables.append(local)
        parents.append(local_bank[-1])
        endpoints.append(_endpoint_from_scores(local_bank, scores, formula))
    if not tables:
        raise RuntimeError("No X4 wells available")
    return pd.concat(tables, ignore_index=True), np.concatenate(parents), np.concatenate(endpoints)


def _metric(local, prediction, endpoint):
    return METRICS(local, prediction, endpoint)


fold_wells = sorted(frame.loc[frame["fold"].eq(FOLD), "well"].unique())
discovery_wells = fold_wells[:DISCOVERY_WELLS]
confirmation_wells = fold_wells[DISCOVERY_WELLS:]
if len(discovery_wells) != DISCOVERY_WELLS or len(confirmation_wells) < 120:
    raise RuntimeError("Unexpected X4 Fold3 split")

discovery_data, failures_a = _prepare(discovery_wells, label="discovery")
scan_rows = []
for formula in FORMULAS:
    local, parent, endpoint = _assemble(discovery_data, formula)
    parent_metric = _metric(local, parent, "k2_parent")
    for dose in DOSES:
        metric = _metric(local, parent + dose * (endpoint - parent), "candidate")
        scan_rows.append({
            "formula": formula,
            "dose": dose,
            "rmse": metric["rmse"],
            "gain": parent_metric["rmse"] - metric["rmse"],
            "far_gain": parent_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
            "p95_change": metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
        })
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "formula", "dose"])
safe = scan[
    scan["dose"].gt(0.0)
    & scan["far_gain"].ge(0.0)
    & scan["p95_change"].le(MAX_P95_CHANGE)
]
selected = safe.iloc[0] if not safe.empty else scan.loc[scan["dose"].eq(0.0)].iloc[0]
formula, dose = str(selected["formula"]), float(selected["dose"])

aligned_data, failures_b = _prepare(confirmation_wells, label="aligned")
local, parent, endpoint = _assemble(aligned_data, formula)
parent_metric = _metric(local, parent, "k2_parent")
candidate = parent + dose * (endpoint - parent)
candidate_metric = _metric(local, candidate, "endpoint_selector")
reverse_metric = _metric(local, parent - dose * (endpoint - parent), "reverse_control")

centres = {}
for well in confirmation_wells:
    table = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT"])
    centres[well] = float(pd.to_numeric(table["TVT"], errors="coerce").median())
ordered = sorted(confirmation_wells, key=lambda well: (centres[well], well))
shuffle_map = dict(zip(ordered, ordered[1:] + ordered[:1]))
shuffle_data, failures_c = _prepare(confirmation_wells, shuffle_map, "typewell_shuffle")
common = sorted(set(aligned_data).intersection(shuffle_data))
aligned_common = {well: aligned_data[well] for well in common}
shuffle_common = {well: shuffle_data[well] for well in common}
common_local, common_parent, common_endpoint = _assemble(aligned_common, formula)
shuffle_local, shuffle_parent, shuffle_endpoint = _assemble(shuffle_common, formula)
if not common_local["id"].astype(str).equals(shuffle_local["id"].astype(str)):
    raise RuntimeError("X4 aligned/shuffle row mismatch")
aligned_common_metric = _metric(
    common_local, common_parent + dose * (common_endpoint - common_parent), "aligned_common"
)
shuffle_metric = _metric(
    shuffle_local, shuffle_parent + dose * (shuffle_endpoint - shuffle_parent), "shuffle"
)

gain = float(parent_metric["rmse"] - candidate_metric["rmse"])
far_gain = float(parent_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"])
p95_change = float(candidate_metric["p95_well_rmse"] - parent_metric["p95_well_rmse"])
shuffle_loss = float(shuffle_metric["rmse"] - aligned_common_metric["rmse"])
reverse_loss = float(reverse_metric["rmse"] - candidate_metric["rmse"])
checks = {
    "discovery_selected_positive_dose": dose > 0.0,
    "confirmation_wells_ge_120": int(candidate_metric["wells"]) >= 120,
    "confirmation_gain_ge_0p03": gain >= 0.03,
    "confirmation_far_gain_ge_0": far_gain >= 0.0,
    "shuffle_loses_ge_0p02": shuffle_loss >= 0.02,
    "reverse_loses_ge_0p03": reverse_loss >= 0.03,
    "p95_change_le_0p20": p95_change <= MAX_P95_CHANGE,
}
strict_pass = bool(all(checks.values()))

verdict = {
    "experiment": "main342_longwindow_endpoint_selector_x4",
    "stage": "fold3_24well_selection_remaining_fold3_confirmation",
    "legal_inputs_only": True,
    "endpoint_bank": bank_names,
    "protocol": {
        "selection_wells": len(discovery_wells),
        "confirmation_wells_requested": len(confirmation_wells),
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "selected": selected.to_dict(),
    "confirmation": {
        "parent": parent_metric,
        "candidate": candidate_metric,
        "aligned_common": aligned_common_metric,
        "typewell_shuffle": shuffle_metric,
        "reverse": reverse_metric,
        "gain": gain,
        "far_gain": far_gain,
        "p95_change": p95_change,
        "shuffle_loss": shuffle_loss,
        "reverse_loss": reverse_loss,
        "common_shuffle_wells": len(common),
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "failures": {
        "discovery": len(failures_a),
        "aligned": len(failures_b),
        "shuffle": len(failures_c),
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze X4 and apply once to Fold2"
        if strict_pass else
        "FAIL -> long-window evidence cannot select existing endpoints; close X1-X4"
    ),
}

scan.to_csv(OUT / "selection_scan.csv", index=False)
pd.DataFrame([parent_metric, candidate_metric, reverse_metric]).to_csv(
    OUT / "confirmation_summary.csv", index=False
)
pd.DataFrame(failures_a + failures_b + failures_c).to_csv(OUT / "failures.csv", index=False)
(OUT / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2), encoding="utf-8"
)
print("\nM342-X4 SELECTION SCAN")
print(scan.to_string(index=False), flush=True)
print("\nM342-X4 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
