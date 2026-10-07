"""M342-X3: decode frozen 256-row GR evidence as a residual path after K2.

The first 24 Fold3 wells select only a transition scale and conservative dose.
The remaining Fold3 wells confirm the frozen decoder with typewell-shuffle,
depth-roll, and reversed-direction controls. Fold1 and Fold2 are not evaluated.
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
OUT = ROOT / "oof_exports" / "main342_longwindow_path_residual_x3"
OUT.mkdir(parents=True, exist_ok=True)

FOLD = 3
DISCOVERY_WELLS = 24
HALF_WINDOW = 256
STATIONS_PER_WELL = 64
TRANSITION_SCALES_FT = (4.0, 8.0, 16.0, 32.0)
DOSES = (0.0, 0.025, 0.05, 0.10, 0.20)
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
    "m342_x3_p1_definitions",
)
X = _bootstrap(
    X1_SOURCE,
    "fold3_wells = sorted(",
    "m342_x3_x1_definitions",
)

base, _, _, coverage = P["O"]["_prepare_inventory"]()
endpoints = P["_endpoint_table"]()[["id", "k2_parent"]]
frame = base.merge(endpoints, on="id", how="inner", validate="one_to_one")
if len(frame) != len(base):
    raise RuntimeError(f"K2 alignment retained {len(frame)}/{len(base)} rows")
frame["well"] = frame["well"].astype(str).str.lower()
if "row_idx" not in frame:
    frame["row_idx"] = frame["id"].astype(str).str.rsplit("_", n=1).str[1].astype(int)

OFFSETS = np.asarray(X["OFFSETS"], np.float64)
ZERO_STATE = int(np.argmin(np.abs(OFFSETS)))
DATA_ROOT = X["DATA_ROOT"]
METRICS = P["B"]["_metrics"]


def _station_emissions(local, typewell_override=None):
    local_for_loader = local[["row_idx", "k2_parent"]].rename(
        columns={"k2_parent": "base"}
    )
    well = str(local["well"].iloc[0])
    path, hgr, _, tw_tvt, tw_gr = X["H"]["_load_well"](
        well, local_for_loader, typewell_override
    )
    hidden_rows = local["row_idx"].to_numpy(np.int64)
    positions = X["H"]["_sample_positions"](hidden_rows, STATIONS_PER_WELL)
    kept_positions, emissions = [], []
    for centre in positions:
        left = max(0, int(centre) - HALF_WINDOW)
        right = min(len(path), int(centre) + HALF_WINDOW + 1)
        horizontal = hgr[left:right]
        if len(horizontal) < HALF_WINDOW or np.isfinite(horizontal).mean() < 0.35:
            continue
        horizontal = X["H"]["_fill"](horizontal)
        if horizontal is None:
            continue
        candidate_tvt = path[left:right][None, :] + OFFSETS[:, None]
        candidate_gr = np.interp(
            candidate_tvt.ravel(), tw_tvt, tw_gr, left=np.nan, right=np.nan
        ).reshape(len(OFFSETS), -1)
        valid = np.isfinite(candidate_gr).mean(axis=1) >= 0.95
        if valid.sum() < 64:
            continue
        medians = np.nanmedian(candidate_gr, axis=1, keepdims=True)
        candidate_gr = np.where(np.isfinite(candidate_gr), candidate_gr, medians)
        amplitude = np.asarray(
            X["H"]["_components"](horizontal, candidate_gr)["amp"], np.float64
        )
        emissions.append(X["_robust_z"](amplitude, valid))
        kept_positions.append(int(centre))
    if len(kept_positions) < 12:
        raise RuntimeError(f"{well}: only {len(kept_positions)} long-window stations")
    return np.asarray(kept_positions, np.int64), np.asarray(emissions, np.float64)


def _decode(emission, transition_scale):
    count, states = emission.shape
    transition = np.square(
        (OFFSETS[:, None] - OFFSETS[None, :]) / float(transition_scale)
    )
    previous = emission[0] - np.square(OFFSETS / float(transition_scale))
    back = np.empty((count, states), np.int16)
    back[0] = ZERO_STATE
    for row in range(1, count):
        values = previous[:, None] - transition
        parent = np.argmax(values, axis=0)
        previous = values[parent, np.arange(states)] + emission[row]
        back[row] = parent.astype(np.int16)
    state = int(np.argmax(previous))
    path = np.empty(count, np.int16)
    path[-1] = state
    for row in range(count - 1, 0, -1):
        path[row - 1] = back[row, path[row]]
    return OFFSETS[path]


def _full_correction(local, positions, station_offsets):
    hidden = local["row_idx"].to_numpy(np.int64)
    anchor = max(int(hidden.min()) - 1, 0)
    x = np.r_[anchor, positions].astype(np.float64)
    y = np.r_[0.0, station_offsets].astype(np.float64)
    unique = np.r_[True, np.diff(x) > 0]
    return np.interp(hidden, x[unique], y[unique])


def _prepare(wells, shuffle_map=None, label="aligned"):
    output, failures = {}, []
    for number, well in enumerate(wells, 1):
        local = frame.loc[frame["well"].eq(well)].sort_values("row_idx").copy()
        try:
            override = shuffle_map.get(well) if shuffle_map else None
            output[well] = (local, *_station_emissions(local, override))
        except Exception as exc:
            failures.append({"well": well, "control": label, "reason": repr(exc)})
        if number % 25 == 0 or number == len(wells):
            print(
                f"M342-X3 {label} {number}/{len(wells)} success={len(output)} "
                f"failures={len(failures)}", flush=True
            )
    return output, failures


def _predictions(prepared, transition_scale):
    parent_parts, correction_parts, tables = [], [], []
    for well in sorted(prepared):
        local, positions, emission = prepared[well]
        station_offsets = _decode(emission, transition_scale)
        correction = _full_correction(local, positions, station_offsets)
        parent_parts.append(local["k2_parent"].to_numpy(np.float64))
        correction_parts.append(correction)
        tables.append(local)
    if not tables:
        raise RuntimeError("No X3 predictions")
    return (
        pd.concat(tables, ignore_index=True),
        np.concatenate(parent_parts),
        np.concatenate(correction_parts),
    )


def _metric(local, prediction, endpoint):
    return METRICS(local, prediction, endpoint)


fold_wells = sorted(frame.loc[frame["fold"].eq(FOLD), "well"].unique())
discovery_wells = fold_wells[:DISCOVERY_WELLS]
confirmation_wells = fold_wells[DISCOVERY_WELLS:]
if len(discovery_wells) != DISCOVERY_WELLS or len(confirmation_wells) < 120:
    raise RuntimeError("Unexpected Fold3 split")

discovery_data, failures_a = _prepare(discovery_wells, label="discovery")
scan_rows = []
for transition_scale in TRANSITION_SCALES_FT:
    local, parent, correction = _predictions(discovery_data, transition_scale)
    parent_metric = _metric(local, parent, "k2_parent")
    far_base = parent_metric["far_gt2000_rmse"]
    for dose in DOSES:
        metric = _metric(local, parent + dose * correction, "candidate")
        scan_rows.append({
            "transition_scale_ft": transition_scale,
            "dose": dose,
            "rmse": metric["rmse"],
            "gain": parent_metric["rmse"] - metric["rmse"],
            "far_gain": far_base - metric["far_gt2000_rmse"],
            "p95_change": metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
        })
scan = pd.DataFrame(scan_rows).sort_values(["rmse", "transition_scale_ft", "dose"])
safe = scan[
    scan["far_gain"].ge(0.0)
    & scan["p95_change"].le(MAX_P95_CHANGE)
    & scan["dose"].gt(0.0)
]
if safe.empty:
    selected = scan.loc[scan["dose"].eq(0.0)].iloc[0]
else:
    selected = safe.iloc[0]
selected_scale = float(selected["transition_scale_ft"])
selected_dose = float(selected["dose"])

confirmation_data, failures_b = _prepare(confirmation_wells, label="aligned")
local, parent, correction = _predictions(confirmation_data, selected_scale)
parent_metric = _metric(local, parent, "k2_parent")
candidate = parent + selected_dose * correction
candidate_metric = _metric(local, candidate, "longwindow_path")

centres = {}
for well in confirmation_wells:
    table = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT"])
    centres[well] = float(pd.to_numeric(table["TVT"], errors="coerce").median())
ordered = sorted(confirmation_wells, key=lambda well: (centres[well], well))
shuffle_map = dict(zip(ordered, ordered[1:] + ordered[:1]))
shuffle_data, failures_c = _prepare(
    confirmation_wells, shuffle_map, label="typewell_shuffle"
)
common_wells = sorted(set(confirmation_data).intersection(shuffle_data))
aligned_common = {well: confirmation_data[well] for well in common_wells}
shuffle_common = {well: shuffle_data[well] for well in common_wells}
common_local, common_parent, common_correction = _predictions(aligned_common, selected_scale)
shuffle_local, shuffle_parent, shuffle_correction = _predictions(shuffle_common, selected_scale)
if not common_local["id"].astype(str).equals(shuffle_local["id"].astype(str)):
    raise RuntimeError("Aligned/shuffle row order mismatch")
shuffle_metric = _metric(
    shuffle_local,
    shuffle_parent + selected_dose * shuffle_correction,
    "typewell_shuffle",
)
aligned_common_metric = _metric(
    common_local,
    common_parent + selected_dose * common_correction,
    "aligned_common",
)

rolled = np.empty_like(correction)
cursor = 0
for _, group in local.groupby("well", sort=False):
    count = len(group)
    shift = max(count // 3, 1)
    rolled[cursor:cursor + count] = np.roll(correction[cursor:cursor + count], shift)
    cursor += count
roll_metric = _metric(local, parent + selected_dose * rolled, "depth_roll_control")
reverse_metric = _metric(local, parent - selected_dose * correction, "reverse_control")

gain = float(parent_metric["rmse"] - candidate_metric["rmse"])
far_gain = float(parent_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"])
p95_change = float(candidate_metric["p95_well_rmse"] - parent_metric["p95_well_rmse"])
shuffle_loss = float(shuffle_metric["rmse"] - aligned_common_metric["rmse"])
roll_loss = float(roll_metric["rmse"] - candidate_metric["rmse"])
reverse_loss = float(reverse_metric["rmse"] - candidate_metric["rmse"])

checks = {
    "discovery_selected_positive_dose": selected_dose > 0.0,
    "confirmation_wells_ge_120": int(candidate_metric["wells"]) >= 120,
    "confirmation_gain_ge_0p03": gain >= 0.03,
    "confirmation_far_gain_ge_0": far_gain >= 0.0,
    "shuffle_loses_ge_0p02": shuffle_loss >= 0.02,
    "depth_roll_loses_ge_0p01": roll_loss >= 0.01,
    "reverse_loses_ge_0p03": reverse_loss >= 0.03,
    "p95_change_le_0p20": p95_change <= MAX_P95_CHANGE,
}
strict_pass = bool(all(checks.values()))

verdict = {
    "experiment": "main342_longwindow_path_residual_x3",
    "stage": "fold3_24well_decoder_selection_remaining_fold3_confirmation",
    "legal_inputs_only": True,
    "protocol": {
        "selection_fold": FOLD,
        "selection_wells": len(discovery_wells),
        "confirmation_fold": FOLD,
        "confirmation_wells_requested": len(confirmation_wells),
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "decoder": {
        "half_window": HALF_WINDOW,
        "stations_per_well": STATIONS_PER_WELL,
        "transition_scale_ft": selected_scale,
        "dose": selected_dose,
    },
    "selection": selected.to_dict(),
    "confirmation": {
        "parent": parent_metric,
        "candidate": candidate_metric,
        "aligned_common": aligned_common_metric,
        "typewell_shuffle": shuffle_metric,
        "depth_roll": roll_metric,
        "reverse": reverse_metric,
        "gain": gain,
        "far_gain": far_gain,
        "p95_change": p95_change,
        "shuffle_loss": shuffle_loss,
        "roll_loss": roll_loss,
        "reverse_loss": reverse_loss,
        "common_shuffle_wells": len(common_wells),
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
        "PASS -> freeze X3 and apply once to Fold2"
        if strict_pass else
        "FAIL -> long-window rank evidence does not survive path decoding; close X1-X3"
    ),
}

scan.to_csv(OUT / "selection_scan.csv", index=False)
pd.DataFrame([parent_metric, candidate_metric, roll_metric, reverse_metric]).to_csv(
    OUT / "confirmation_summary.csv", index=False
)
pd.DataFrame(failures_a + failures_b + failures_c).to_csv(
    OUT / "failures.csv", index=False
)
(OUT / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2), encoding="utf-8"
)
print("\nM342-X3 SELECTION SCAN")
print(scan.to_string(index=False), flush=True)
print("\nM342-X3 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
