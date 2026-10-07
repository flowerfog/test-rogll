"""SEGN-R9: typewell-source group prior oracle and confirmation audit.

The raw files contain many typewell excerpts that share long exact GR
subsequences even when their TVT ranges differ.  Before adding a typewell-ID
condition to SEGN, this zero-training experiment asks whether that identity has
label-free predictive value on held-out wells.

Groups are built from input-only GR shingles.  Outer-train wells fit global and
group-specific structural trajectories in S = TVT + Z coordinates.  Fold-3 is
split into 32 selection and 122 confirmation wells.  A shuffled-group control
and a per-well blend oracle distinguish real group transfer from smoothing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd


ROOT_R9 = Path("/root/autodl-tmp")
SOURCE_CANDIDATES_R9 = (
    ROOT_R9 / "segn_r6_densemd_evidence_fold3_v0_cell.py",
    Path.cwd() / "segn_r6_densemd_evidence_fold3_v0_cell.py",
)
SOURCE_PATH_R9 = next((path for path in SOURCE_CANDIDATES_R9 if path.exists()), None)
if SOURCE_PATH_R9 is None:
    raise FileNotFoundError(
        "Upload segn_r6_densemd_evidence_fold3_v0_cell.py to /root/autodl-tmp."
    )
source_r9 = SOURCE_PATH_R9.read_text(encoding="utf-8")
marker_r9 = "\nstarted_r6 = time.time()"
if marker_r9 not in source_r9:
    raise RuntimeError(f"Cannot find R6 execution marker in {SOURCE_PATH_R9}")
exec(compile(source_r9.split(marker_r9, 1)[0], str(SOURCE_PATH_R9), "exec"), globals())


EXPERIMENT_R9 = "segn_r9_typewell_group_prior_oracle_v0"
OUT_DIR_R9 = OOF_ROOT / EXPERIMENT_R9
OUT_DIR_R9.mkdir(parents=True, exist_ok=True)
OUT_DIR = OUT_DIR_R9

TRAIN_FOLDS_R9 = (1, 2, 4, 5)
VALID_FOLD_R9 = 3
SELECTION_WELLS_R9 = 32
SHINGLE_LENGTH_R9 = 12
SHINGLE_ROUND_DECIMALS_R9 = 4
MIN_SHARED_SHINGLES_R9 = 50
MIN_SMALLER_OVERLAP_R9 = 0.50
PROFILE_POINTS_R9 = 257
MIN_GROUP_TRAIN_WELLS_R9 = 2
ALPHAS_R9 = (0.0, 0.25, 0.50, 0.75, 1.0)
PROFILE_REDUCERS_R9 = ("mean", "median")


def _json_default_r9(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json_r9(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default_r9), encoding="utf-8"
    )


class _UnionFindR9:
    def __init__(self, count):
        self.parent = list(range(count))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left = self.find(left)
        right = self.find(right)
        if left != right:
            self.parent[right] = left


def _typewell_shingles_r9(well):
    path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    frame = pd.read_csv(path, usecols=["GR"])
    values = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(np.float64)
    values = np.round(
        np.nan_to_num(values, nan=9.87654321e8), SHINGLE_ROUND_DECIMALS_R9
    ).astype("<f8")
    if len(values) < SHINGLE_LENGTH_R9:
        return set()
    return {
        hashlib.blake2b(
            values[index : index + SHINGLE_LENGTH_R9].tobytes(), digest_size=8
        ).digest()
        for index in range(len(values) - SHINGLE_LENGTH_R9 + 1)
    }


def _build_typewell_groups_r9(wells):
    shingles = []
    inverted = defaultdict(list)
    for number, well in enumerate(wells, 1):
        values = _typewell_shingles_r9(well)
        shingles.append(values)
        for token in values:
            inverted[token].append(number - 1)
        if number % 100 == 0 or number == len(wells):
            print(f"R9 typewell shingles {number}/{len(wells)}", flush=True)

    intersections = Counter()
    for members in inverted.values():
        # Very common tokens are uninformative flat/no-data patterns.
        if not (1 < len(members) <= 30):
            continue
        for offset, left in enumerate(members):
            for right in members[offset + 1 :]:
                intersections[(left, right)] += 1

    union = _UnionFindR9(len(wells))
    accepted_edges = []
    for (left, right), shared in intersections.items():
        denominator = max(1, min(len(shingles[left]), len(shingles[right])))
        overlap = shared / denominator
        if shared >= MIN_SHARED_SHINGLES_R9 and overlap >= MIN_SMALLER_OVERLAP_R9:
            union.union(left, right)
            accepted_edges.append((left, right, shared, overlap))

    roots = [union.find(index) for index in range(len(wells))]
    root_members = defaultdict(list)
    for index, root in enumerate(roots):
        root_members[root].append(index)
    ordered_roots = sorted(
        root_members,
        key=lambda root: (-len(root_members[root]), min(root_members[root])),
    )
    group_for_root = {root: group for group, root in enumerate(ordered_roots)}
    mapping = {
        well: int(group_for_root[union.find(index)]) for index, well in enumerate(wells)
    }
    sizes = Counter(mapping.values())
    frame = pd.DataFrame(
        {
            "well": wells,
            "typewell_group": [mapping[well] for well in wells],
            "typewell_group_size": [sizes[mapping[well]] for well in wells],
            "shingles": [len(values) for values in shingles],
        }
    )
    summary = {
        "wells": len(wells),
        "groups": len(sizes),
        "duplicate_groups": int(sum(size > 1 for size in sizes.values())),
        "wells_in_duplicate_groups": int(
            sum(size for size in sizes.values() if size > 1)
        ),
        "maximum_group_size": int(max(sizes.values())),
        "accepted_edges": len(accepted_edges),
        "shingle_length": SHINGLE_LENGTH_R9,
        "minimum_shared_shingles": MIN_SHARED_SHINGLES_R9,
        "minimum_smaller_overlap": MIN_SMALLER_OVERLAP_R9,
    }
    return mapping, frame, summary


def _z_for_record_r9(record):
    path = DATA_ROOT / "train" / f"{record.well}__horizontal_well.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    md_column = next((name for name in ("MD", "md") if name in header), None)
    z_column = next((name for name in ("Z", "z") if name in header), None)
    if md_column is None or z_column is None:
        raise RuntimeError(f"{record.well}: cannot find MD/Z in {header}")
    frame = pd.read_csv(path, usecols=[md_column, z_column])
    md = pd.to_numeric(frame[md_column], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(frame[z_column], errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(md) & np.isfinite(z)
    if valid.sum() < 2:
        raise RuntimeError(f"{record.well}: insufficient finite MD/Z")
    if len(md) == len(record.md) and np.nanmax(np.abs(md - record.md)) <= 0.02:
        return z
    return np.interp(record.md.astype(np.float64), md[valid], z[valid])


def _profile_for_record_r9(record, z, grid):
    eval_rows = record.eval_index
    anchor = record.anchor_index
    anchor_md = float(record.md[anchor])
    end_md = float(record.md[eval_rows[-1]])
    span = max(end_md - anchor_md, 1.0)
    coordinate = (record.md[eval_rows].astype(np.float64) - anchor_md) / span
    anchor_structure = float(record.anchor_tvt) + float(z[anchor])
    structure_delta = (
        record.truth[eval_rows].astype(np.float64) + z[eval_rows] - anchor_structure
    )
    x = np.r_[0.0, coordinate]
    y = np.r_[0.0, structure_delta]
    keep = np.r_[True, np.diff(x) > 1.0e-9]
    return np.interp(grid, x[keep], y[keep])


def _fit_profiles_r9(records, group_map, z_by_well, grid, reducer):
    profiles = defaultdict(list)
    all_profiles = []
    for record in records:
        profile = _profile_for_record_r9(record, z_by_well[record.well], grid)
        profiles[group_map[record.well]].append(profile)
        all_profiles.append(profile)
    reduction = np.mean if reducer == "mean" else np.median
    global_profile = reduction(np.stack(all_profiles), axis=0)
    group_profiles = {
        group: reduction(np.stack(values), axis=0)
        for group, values in profiles.items()
        if len(values) >= MIN_GROUP_TRAIN_WELLS_R9
    }
    group_train_counts = {group: len(values) for group, values in profiles.items()}
    return global_profile, group_profiles, group_train_counts


def _prediction_r9(record, z, profile, grid):
    eval_rows = record.eval_index
    anchor = record.anchor_index
    anchor_md = float(record.md[anchor])
    end_md = float(record.md[eval_rows[-1]])
    span = max(end_md - anchor_md, 1.0)
    coordinate = np.clip(
        (record.md[eval_rows].astype(np.float64) - anchor_md) / span, 0.0, 1.0
    )
    structure_delta = np.interp(coordinate, grid, profile)
    anchor_structure = float(record.anchor_tvt) + float(z[anchor])
    return anchor_structure + structure_delta - z[eval_rows]


def _metrics_r9(rows):
    truth = np.concatenate([row["truth"] for row in rows])
    prediction = np.concatenate([row["prediction"] for row in rows])
    distance = np.concatenate([row["distance"] for row in rows])
    error = prediction - truth
    squared = error * error
    well_rmse = np.asarray(
        [
            np.sqrt(np.mean((row["prediction"] - row["truth"]) ** 2))
            for row in rows
        ],
        np.float64,
    )
    result = {
        "rows": int(len(truth)),
        "wells": int(len(rows)),
        "exact_sse": float(squared.sum()),
        "rmse": float(np.sqrt(squared.mean())),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }
    for name, mask in (
        ("near_0_500_rmse", distance <= 500.0),
        ("mid_500_1000_rmse", (distance > 500.0) & (distance <= 1000.0)),
        ("mid_1000_2000_rmse", (distance > 1000.0) & (distance <= 2000.0)),
        ("far_gt2000_rmse", distance > 2000.0),
    ):
        result[name] = (
            float(np.sqrt(np.mean(squared[mask]))) if int(mask.sum()) else math.nan
        )
    return result


def _evaluate_alpha_r9(
    records,
    group_map,
    z_by_well,
    grid,
    global_profile,
    group_profiles,
    alpha,
    shuffled_groups=None,
):
    rows = []
    covered_rows = []
    for index, record in enumerate(records):
        group = (
            shuffled_groups[index]
            if shuffled_groups is not None
            else group_map[record.well]
        )
        group_profile = group_profiles.get(group)
        covered = group_profile is not None
        profile = (
            global_profile
            if not covered
            else global_profile + alpha * (group_profile - global_profile)
        )
        prediction = _prediction_r9(record, z_by_well[record.well], profile, grid)
        eval_rows = record.eval_index
        item = {
            "well": record.well,
            "truth": record.truth[eval_rows].astype(np.float64),
            "prediction": prediction,
            "distance": record.md[eval_rows].astype(np.float64)
            - float(record.md[record.anchor_index]),
            "covered": covered,
        }
        rows.append(item)
        if covered:
            covered_rows.append(item)
    return _metrics_r9(rows), (_metrics_r9(covered_rows) if covered_rows else None), rows


def _perwell_oracle_r9(global_rows, group_rows):
    output = []
    weights = []
    for global_item, group_item in zip(global_rows, group_rows):
        truth = global_item["truth"]
        left = global_item["prediction"]
        right = group_item["prediction"]
        delta = right - left
        denominator = float(np.dot(delta, delta))
        weight = (
            float(np.clip(np.dot(truth - left, delta) / denominator, 0.0, 1.0))
            if denominator > 1.0e-12
            else 0.0
        )
        weights.append(weight)
        output.append({**global_item, "prediction": left + weight * delta})
    return _metrics_r9(output), weights


started_r9 = time.time()
folds_r9, fold_path_r9 = _load_folds()
all_wells_r9 = folds_r9["well"].astype(str).tolist()
train_wells_r9 = folds_r9.loc[
    folds_r9["fold"].isin(TRAIN_FOLDS_R9), "well"
].astype(str).tolist()
fold3_wells_r9 = folds_r9.loc[
    folds_r9["fold"].eq(VALID_FOLD_R9), "well"
].astype(str).tolist()
if len(all_wells_r9) != 773 or len(train_wells_r9) != 619 or len(fold3_wells_r9) != 154:
    raise RuntimeError(
        f"Unexpected all/train/Fold3 sizes: "
        f"{len(all_wells_r9)}/{len(train_wells_r9)}/{len(fold3_wells_r9)}"
    )

group_map_r9, group_frame_r9, group_summary_r9 = _build_typewell_groups_r9(
    all_wells_r9
)
group_frame_r9.to_csv(OUT_DIR_R9 / "well_to_typewell_group.csv", index=False)
print("SEGN-R9 GROUPS", json.dumps(group_summary_r9, indent=2), flush=True)

records_r9 = {
    record.well: record
    for record in _load_records(sorted(set(train_wells_r9 + fold3_wells_r9)), "r9_all")
}
train_records_r9 = [records_r9[well] for well in train_wells_r9]
fold3_records_r9 = [records_r9[well] for well in fold3_wells_r9]

order_r9 = np.argsort([record.eval_rows for record in fold3_records_r9])
selection_positions_r9 = np.unique(
    np.rint(np.linspace(0, len(order_r9) - 1, SELECTION_WELLS_R9)).astype(int)
)
selection_indices_r9 = set(order_r9[selection_positions_r9].tolist())
selection_records_r9 = [
    record for index, record in enumerate(fold3_records_r9) if index in selection_indices_r9
]
confirmation_records_r9 = [
    record for index, record in enumerate(fold3_records_r9) if index not in selection_indices_r9
]

z_by_well_r9 = {}
for number, record in enumerate(train_records_r9 + fold3_records_r9, 1):
    z_by_well_r9[record.well] = _z_for_record_r9(record)
    if number % 100 == 0 or number == len(train_records_r9) + len(fold3_records_r9):
        print(
            f"SEGN-R9 load Z {number}/{len(train_records_r9) + len(fold3_records_r9)}",
            flush=True,
        )

grid_r9 = np.linspace(0.0, 1.0, PROFILE_POINTS_R9, dtype=np.float64)
profile_banks_r9 = {}
scan_r9 = []
for reducer_r9 in PROFILE_REDUCERS_R9:
    bank_r9 = _fit_profiles_r9(
        train_records_r9, group_map_r9, z_by_well_r9, grid_r9, reducer_r9
    )
    profile_banks_r9[reducer_r9] = bank_r9
    global_profile_r9, group_profiles_r9, group_train_counts_r9 = bank_r9
    for alpha_r9 in ALPHAS_R9:
        metrics_r9, covered_r9, _ = _evaluate_alpha_r9(
            selection_records_r9,
            group_map_r9,
            z_by_well_r9,
            grid_r9,
            global_profile_r9,
            group_profiles_r9,
            alpha_r9,
        )
        scan_r9.append(
            {
                "reducer": reducer_r9,
                "alpha": alpha_r9,
                "covered_wells": 0 if covered_r9 is None else covered_r9["wells"],
                "covered_rmse": math.nan if covered_r9 is None else covered_r9["rmse"],
                **metrics_r9,
            }
        )

scan_frame_r9 = pd.DataFrame(scan_r9).sort_values(["rmse", "reducer", "alpha"])
scan_frame_r9.to_csv(OUT_DIR_R9 / "selection_scan.csv", index=False)
print("SEGN-R9 SELECTION")
print(scan_frame_r9.to_string(index=False), flush=True)
positive_scan_r9 = scan_frame_r9[scan_frame_r9["alpha"] > 0.0]
winner_r9 = positive_scan_r9.iloc[0].to_dict()
reducer_r9 = str(winner_r9["reducer"])
alpha_r9 = float(winner_r9["alpha"])
global_profile_r9, group_profiles_r9, group_train_counts_r9 = profile_banks_r9[
    reducer_r9
]

global_metrics_r9, global_covered_r9, global_rows_r9 = _evaluate_alpha_r9(
    confirmation_records_r9,
    group_map_r9,
    z_by_well_r9,
    grid_r9,
    global_profile_r9,
    group_profiles_r9,
    0.0,
)
group_metrics_r9, group_covered_r9, group_rows_r9 = _evaluate_alpha_r9(
    confirmation_records_r9,
    group_map_r9,
    z_by_well_r9,
    grid_r9,
    global_profile_r9,
    group_profiles_r9,
    alpha_r9,
)

rng_r9 = np.random.default_rng(SEED + 9001)
original_confirmation_groups_r9 = np.asarray(
    [group_map_r9[record.well] for record in confirmation_records_r9], np.int64
)
covered_positions_r9 = np.asarray(
    [
        index
        for index, group in enumerate(original_confirmation_groups_r9)
        if int(group) in group_profiles_r9
    ],
    np.int64,
)
shuffled_groups_r9 = np.full(len(confirmation_records_r9), -1, np.int64)
if len(covered_positions_r9):
    shuffled_groups_r9[covered_positions_r9] = rng_r9.permutation(
        original_confirmation_groups_r9[covered_positions_r9]
    )
shuffled_groups_r9 = shuffled_groups_r9.tolist()
shuffled_metrics_r9, shuffled_covered_r9, _ = _evaluate_alpha_r9(
    confirmation_records_r9,
    group_map_r9,
    z_by_well_r9,
    grid_r9,
    global_profile_r9,
    group_profiles_r9,
    alpha_r9,
    shuffled_groups=shuffled_groups_r9,
)
oracle_metrics_r9, oracle_weights_r9 = _perwell_oracle_r9(
    global_rows_r9, group_rows_r9
)

confirmation_gain_r9 = global_metrics_r9["rmse"] - group_metrics_r9["rmse"]
covered_gain_r9 = (
    global_covered_r9["rmse"] - group_covered_r9["rmse"]
    if global_covered_r9 is not None and group_covered_r9 is not None
    else math.nan
)
shuffle_degradation_r9 = shuffled_metrics_r9["rmse"] - group_metrics_r9["rmse"]
oracle_gain_r9 = global_metrics_r9["rmse"] - oracle_metrics_r9["rmse"]

checks_r9 = {
    "confirmation_gain_ge_0p30": confirmation_gain_r9 >= 0.30,
    "covered_subset_gain_ge_0p50": covered_gain_r9 >= 0.50,
    "shuffled_degradation_ge_0p30": shuffle_degradation_r9 >= 0.30,
    "perwell_oracle_gain_ge_0p75": oracle_gain_r9 >= 0.75,
    "confirmation_group_coverage_ge_0p50": (
        group_covered_r9 is not None
        and group_covered_r9["wells"] / len(confirmation_records_r9) >= 0.50
    ),
}
strict_pass_r9 = bool(all(checks_r9.values()))

comparison_r9 = pd.DataFrame(
    [
        {"variant": "global_prior", **global_metrics_r9},
        {"variant": "typewell_group_prior", **group_metrics_r9},
        {"variant": "shuffled_group_prior", **shuffled_metrics_r9},
        {"variant": "perwell_blend_oracle", **oracle_metrics_r9},
    ]
)
comparison_r9.to_csv(OUT_DIR_R9 / "confirmation_comparison.csv", index=False)
print("SEGN-R9 CONFIRMATION")
print(comparison_r9.to_string(index=False), flush=True)

verdict_r9 = {
    "experiment": EXPERIMENT_R9,
    "stage": "fold3_input_only_typewell_group_structural_prior_gateway",
    "optimizer_steps": 0,
    "uses_validation_truth_for_grouping": False,
    "uses_neighbor_coordinates": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS_R9),
        "validation_fold": VALID_FOLD_R9,
        "selection_wells": len(selection_records_r9),
        "confirmation_wells": len(confirmation_records_r9),
    },
    "grouping": group_summary_r9,
    "selected": winner_r9,
    "confirmation": {
        "global_prior": global_metrics_r9,
        "typewell_group_prior": group_metrics_r9,
        "covered_global_prior": global_covered_r9,
        "covered_typewell_group_prior": group_covered_r9,
        "shuffled_group_prior": shuffled_metrics_r9,
        "perwell_blend_oracle": oracle_metrics_r9,
        "mean_oracle_group_weight": float(np.mean(oracle_weights_r9)),
        "gain_vs_global": confirmation_gain_r9,
        "covered_subset_gain": covered_gain_r9,
        "shuffle_degradation": shuffle_degradation_r9,
        "oracle_gain_vs_global": oracle_gain_r9,
    },
    "checks": checks_r9,
    "strict_pass": strict_pass_r9,
    "elapsed_minutes": (time.time() - started_r9) / 60.0,
    "next": (
        "PASS -> add an input-derived typewell-group prior as a separately gated SEGN channel"
        if strict_pass_r9
        else "FAIL -> typewell source identity is not the missing SEGN condition; do not add an ID embedding"
    ),
}
_write_json_r9(OUT_DIR_R9 / "verdict.json", verdict_r9)
print("SEGN-R9 FINAL VERDICT", json.dumps(verdict_r9, indent=2), flush=True)
print("saved:", OUT_DIR_R9, flush=True)

if not strict_pass_r9:
    raise RuntimeError(json.dumps(verdict_r9, indent=2))
