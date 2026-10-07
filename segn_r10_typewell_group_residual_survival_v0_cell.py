"""SEGN-R10: typewell-group residual survival on fold-clean base paths.

R9 proved that input-derived typewell groups carry real structural information,
but its standalone priors were unusable (about 100 ft RMSE).  This experiment
tests the only safe consumption position: a low-dose residual correction around
fold-clean SP45/FLE paths.  Group labels are built without target values; only
outer-train residuals fit the correction profiles.

Fold 3 is split into 32 selection and 122 confirmation wells.  Matched prior,
shuffled-group, p95, and per-well oracle checks prevent a relative gain on a bad
baseline from passing again.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
import json
import math
import time

import numpy as np
import pandas as pd


ROOT_R10 = Path("/root/autodl-tmp")
SOURCE_CANDIDATES_R10 = (
    ROOT_R10 / "segn_r9_typewell_group_prior_oracle_v0_cell.py",
    Path.cwd() / "segn_r9_typewell_group_prior_oracle_v0_cell.py",
)
SOURCE_PATH_R10 = next((path for path in SOURCE_CANDIDATES_R10 if path.exists()), None)
if SOURCE_PATH_R10 is None:
    raise FileNotFoundError(
        "Upload segn_r9_typewell_group_prior_oracle_v0_cell.py to /root/autodl-tmp."
    )
source_r10 = SOURCE_PATH_R10.read_text(encoding="utf-8")
marker_r10 = "\nstarted_r9 = time.time()"
if marker_r10 not in source_r10:
    raise RuntimeError(f"Cannot find R9 execution marker in {SOURCE_PATH_R10}")
exec(compile(source_r10.split(marker_r10, 1)[0], str(SOURCE_PATH_R10), "exec"), globals())


EXPERIMENT_R10 = "segn_r10_typewell_group_residual_survival_v0"
OUT_DIR_R10 = OOF_ROOT / EXPERIMENT_R10
OUT_DIR_R10.mkdir(parents=True, exist_ok=True)
OUT_DIR = OUT_DIR_R10

POOL_CANDIDATES_R10 = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
POOL_PATH_R10 = next((path for path in POOL_CANDIDATES_R10 if path.exists()), None)
if POOL_PATH_R10 is None:
    raise FileNotFoundError("Cannot find the fold-clean candidate pool CSV")

GROUP_PATH_R10 = (
    OOF_ROOT / "segn_r9_typewell_group_prior_oracle_v0" / "well_to_typewell_group.csv"
)
TRAIN_FOLDS_R10 = (1, 2, 4, 5)
VALID_FOLD_R10 = 3
SELECTION_WELLS_R10 = 32
BASE_PREFERENCE_R10 = ("sp45", "fleongg")
PROFILE_POINTS_R10 = 257
MIN_GROUP_TRAIN_WELLS_R10 = 2
REDUCERS_R10 = ("mean", "median")
GLOBAL_DOSES_R10 = (0.0, 0.25, 0.50, 1.0)
GROUP_DOSES_R10 = (0.0, 0.25, 0.50, 1.0)
TRUTH_TOLERANCE_FT_R10 = 0.02


def _json_default_r10(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json_r10(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default_r10), encoding="utf-8"
    )


def _load_group_map_r10(wells):
    if GROUP_PATH_R10.exists():
        frame = pd.read_csv(GROUP_PATH_R10, dtype={"well": str})
        if set(frame["well"]) == set(wells) and frame["well"].nunique() == len(wells):
            mapping = dict(
                zip(frame["well"], frame["typewell_group"].astype(int))
            )
            sizes = Counter(mapping.values())
            return mapping, frame, {
                "source": str(GROUP_PATH_R10),
                "groups": len(sizes),
                "duplicate_groups": int(sum(size > 1 for size in sizes.values())),
                "wells_in_duplicate_groups": int(
                    sum(size for size in sizes.values() if size > 1)
                ),
                "maximum_group_size": int(max(sizes.values())),
            }
    mapping, frame, summary = _build_typewell_groups_r9(wells)
    summary["source"] = "recomputed_input_only_GR_shingles"
    return mapping, frame, summary


def _load_pool_r10(records):
    header = pd.read_csv(POOL_PATH_R10, nrows=0).columns.tolist()
    truth_column = next(
        (column for column in ("tvt_true", "TVT", "target") if column in header),
        None,
    )
    bases = [base for base in BASE_PREFERENCE_R10 if base in header]
    if truth_column is None or "well" not in header or not bases:
        raise RuntimeError(f"Unexpected candidate-pool columns: {header}")
    wanted = {record.well for record in records}
    chunks = []
    for chunk in pd.read_csv(
        POOL_PATH_R10,
        usecols=["well", truth_column, *bases],
        dtype={"well": str},
        chunksize=500_000,
    ):
        selected = chunk[chunk["well"].isin(wanted)]
        if len(selected):
            chunks.append(selected.copy())
    pool = pd.concat(chunks, ignore_index=True)
    grouped = {well: frame for well, frame in pool.groupby("well", sort=False)}
    output = {}
    maximum_gap = 0.0
    for record in records:
        frame = grouped.get(record.well)
        if frame is None or len(frame) != len(record.eval_index):
            raise RuntimeError(
                f"{record.well}: pool rows {0 if frame is None else len(frame)} "
                f"!= eval rows {len(record.eval_index)}"
            )
        pool_truth = pd.to_numeric(frame[truth_column], errors="raise").to_numpy(
            np.float64
        )
        record_truth = record.truth[record.eval_index].astype(np.float64)
        gap = np.abs(pool_truth - record_truth)
        if not np.isfinite(gap).all() or float(gap.max()) > TRUTH_TOLERANCE_FT_R10:
            raise RuntimeError(f"{record.well}: truth alignment gap={gap.max():.6f}")
        maximum_gap = max(maximum_gap, float(gap.max()))
        output[record.well] = {}
        for base in bases:
            values = pd.to_numeric(frame[base], errors="coerce").to_numpy(np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"{record.well}/{base}: non-finite base path")
            output[record.well][base] = values
    return output, bases, {
        "source": str(POOL_PATH_R10),
        "rows": int(len(pool)),
        "wells": int(len(grouped)),
        "maximum_truth_gap": maximum_gap,
    }


def _normalised_coordinate_r10(record):
    eval_rows = record.eval_index
    anchor_md = float(record.md[record.anchor_index])
    span = max(float(record.md[eval_rows[-1]]) - anchor_md, 1.0)
    return np.clip(
        (record.md[eval_rows].astype(np.float64) - anchor_md) / span, 0.0, 1.0
    )


def _residual_profile_r10(record, base, grid):
    coordinate = _normalised_coordinate_r10(record)
    truth = record.truth[record.eval_index].astype(np.float64)
    residual = truth - base
    x = np.r_[0.0, coordinate]
    y = np.r_[0.0, residual]
    keep = np.r_[True, np.diff(x) > 1.0e-9]
    return np.interp(grid, x[keep], y[keep])


def _fit_residual_bank_r10(
    records, base_paths, group_map, base_name, grid, reducer
):
    all_profiles = []
    grouped_profiles = defaultdict(list)
    for record in records:
        profile = _residual_profile_r10(
            record, base_paths[record.well][base_name], grid
        )
        all_profiles.append(profile)
        grouped_profiles[group_map[record.well]].append(profile)
    reduction = np.mean if reducer == "mean" else np.median
    global_profile = reduction(np.stack(all_profiles), axis=0)
    group_profiles = {
        group: reduction(np.stack(values), axis=0)
        for group, values in grouped_profiles.items()
        if len(values) >= MIN_GROUP_TRAIN_WELLS_R10
    }
    group_counts = {group: len(values) for group, values in grouped_profiles.items()}
    return global_profile, group_profiles, group_counts


def _evaluate_r10(
    records,
    base_paths,
    group_map,
    base_name,
    grid,
    global_profile,
    group_profiles,
    global_dose,
    group_dose,
    shuffled_groups=None,
):
    rows = []
    covered_rows = []
    for index, record in enumerate(records):
        coordinate = _normalised_coordinate_r10(record)
        base = base_paths[record.well][base_name]
        global_correction = np.interp(coordinate, grid, global_profile)
        group = (
            int(shuffled_groups[index])
            if shuffled_groups is not None
            else int(group_map[record.well])
        )
        group_profile = group_profiles.get(group)
        covered = group_profile is not None
        group_delta = (
            np.zeros_like(global_correction)
            if not covered
            else np.interp(coordinate, grid, group_profile - global_profile)
        )
        prediction = base + global_dose * global_correction + group_dose * group_delta
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


def _oracle_between_r10(left_rows, right_rows):
    output = []
    weights = []
    for left_item, right_item in zip(left_rows, right_rows):
        truth = left_item["truth"]
        left = left_item["prediction"]
        delta = right_item["prediction"] - left
        denominator = float(np.dot(delta, delta))
        weight = (
            float(np.clip(np.dot(truth - left, delta) / denominator, 0.0, 1.0))
            if denominator > 1.0e-12
            else 0.0
        )
        weights.append(weight)
        output.append({**left_item, "prediction": left + weight * delta})
    return _metrics_r9(output), weights


def _perwell_comparison_r10(
    records, group_map, group_counts, matched_rows, group_rows, oracle_weights
):
    output = []
    for record, matched, grouped, oracle_weight in zip(
        records, matched_rows, group_rows, oracle_weights
    ):
        truth = matched["truth"]
        matched_error = matched["prediction"] - truth
        group_error = grouped["prediction"] - truth
        far = matched["distance"] > 2000.0
        group = int(group_map[record.well])
        output.append(
            {
                "well": record.well,
                "typewell_group": group,
                "group_train_wells": int(group_counts.get(group, 0)),
                "covered": bool(grouped["covered"]),
                "rows": int(len(truth)),
                "matched_rmse": float(np.sqrt(np.mean(np.square(matched_error)))),
                "group_rmse": float(np.sqrt(np.mean(np.square(group_error)))),
                "rmse_gain": float(
                    np.sqrt(np.mean(np.square(matched_error)))
                    - np.sqrt(np.mean(np.square(group_error)))
                ),
                "matched_far_rmse": (
                    float(np.sqrt(np.mean(np.square(matched_error[far]))))
                    if np.any(far)
                    else math.nan
                ),
                "group_far_rmse": (
                    float(np.sqrt(np.mean(np.square(group_error[far]))))
                    if np.any(far)
                    else math.nan
                ),
                "oracle_group_weight": float(oracle_weight),
            }
        )
    return pd.DataFrame(output).sort_values("rmse_gain", ascending=False)


started_r10 = time.time()
folds_r10, fold_path_r10 = _load_folds()
all_wells_r10 = folds_r10["well"].astype(str).tolist()
train_wells_r10 = folds_r10.loc[
    folds_r10["fold"].isin(TRAIN_FOLDS_R10), "well"
].astype(str).tolist()
fold3_wells_r10 = folds_r10.loc[
    folds_r10["fold"].eq(VALID_FOLD_R10), "well"
].astype(str).tolist()
if len(all_wells_r10) != 773 or len(train_wells_r10) != 619 or len(fold3_wells_r10) != 154:
    raise RuntimeError(
        f"Unexpected all/train/Fold3 sizes: "
        f"{len(all_wells_r10)}/{len(train_wells_r10)}/{len(fold3_wells_r10)}"
    )

group_map_r10, group_frame_r10, group_summary_r10 = _load_group_map_r10(
    all_wells_r10
)
group_frame_r10.to_csv(OUT_DIR_R10 / "well_to_typewell_group.csv", index=False)
records_r10 = {
    record.well: record
    for record in _load_records(all_wells_r10, "r10_all")
}
train_records_r10 = [records_r10[well] for well in train_wells_r10]
fold3_records_r10 = [records_r10[well] for well in fold3_wells_r10]
base_paths_r10, base_names_r10, pool_alignment_r10 = _load_pool_r10(
    list(records_r10.values())
)

order_r10 = np.argsort([record.eval_rows for record in fold3_records_r10])
selection_positions_r10 = np.unique(
    np.rint(np.linspace(0, len(order_r10) - 1, SELECTION_WELLS_R10)).astype(int)
)
selection_indices_r10 = set(order_r10[selection_positions_r10].tolist())
selection_records_r10 = [
    record for index, record in enumerate(fold3_records_r10) if index in selection_indices_r10
]
confirmation_records_r10 = [
    record for index, record in enumerate(fold3_records_r10) if index not in selection_indices_r10
]

grid_r10 = np.linspace(0.0, 1.0, PROFILE_POINTS_R10, dtype=np.float64)
banks_r10 = {}
scan_r10 = []
for base_r10 in base_names_r10:
    for reducer_r10 in REDUCERS_R10:
        bank_r10 = _fit_residual_bank_r10(
            train_records_r10,
            base_paths_r10,
            group_map_r10,
            base_r10,
            grid_r10,
            reducer_r10,
        )
        banks_r10[(base_r10, reducer_r10)] = bank_r10
        global_profile_r10, group_profiles_r10, _ = bank_r10
        for global_dose_r10 in GLOBAL_DOSES_R10:
            matched_selection_r10, matched_selection_covered_r10, _ = _evaluate_r10(
                selection_records_r10,
                base_paths_r10,
                group_map_r10,
                base_r10,
                grid_r10,
                global_profile_r10,
                group_profiles_r10,
                global_dose_r10,
                0.0,
            )
            for group_dose_r10 in GROUP_DOSES_R10:
                metrics_r10, covered_r10, _ = _evaluate_r10(
                    selection_records_r10,
                    base_paths_r10,
                    group_map_r10,
                    base_r10,
                    grid_r10,
                    global_profile_r10,
                    group_profiles_r10,
                    global_dose_r10,
                    group_dose_r10,
                )
                selection_covered_gain_r10 = (
                    matched_selection_covered_r10["rmse"] - covered_r10["rmse"]
                    if matched_selection_covered_r10 is not None
                    and covered_r10 is not None
                    else math.nan
                )
                scan_r10.append(
                    {
                        "base": base_r10,
                        "reducer": reducer_r10,
                        "global_dose": global_dose_r10,
                        "group_dose": group_dose_r10,
                        "covered_wells": 0 if covered_r10 is None else covered_r10["wells"],
                        "covered_rmse": math.nan if covered_r10 is None else covered_r10["rmse"],
                        "matched_rmse": matched_selection_r10["rmse"],
                        "gain_vs_matched": matched_selection_r10["rmse"]
                        - metrics_r10["rmse"],
                        "covered_gain_vs_matched": selection_covered_gain_r10,
                        "p95_degradation_vs_matched": metrics_r10["p95_well_rmse"]
                        - matched_selection_r10["p95_well_rmse"],
                        **metrics_r10,
                    }
                )

scan_frame_r10 = pd.DataFrame(scan_r10).sort_values(
    ["rmse", "base", "reducer", "global_dose", "group_dose"]
)
scan_frame_r10.to_csv(OUT_DIR_R10 / "selection_scan.csv", index=False)
print("SEGN-R10 SELECTION TOP")
print(scan_frame_r10.head(30).to_string(index=False), flush=True)
positive_r10 = scan_frame_r10[scan_frame_r10["group_dose"] > 0.0]
credible_r10 = positive_r10[
    (positive_r10["gain_vs_matched"] > 0.0)
    & (positive_r10["covered_gain_vs_matched"] > 0.0)
]
winner_pool_r10 = credible_r10 if len(credible_r10) else positive_r10
winner_r10 = winner_pool_r10.sort_values(
    ["rmse", "p95_well_rmse", "base", "reducer", "global_dose", "group_dose"]
).iloc[0].to_dict()

base_r10 = str(winner_r10["base"])
reducer_r10 = str(winner_r10["reducer"])
global_dose_r10 = float(winner_r10["global_dose"])
group_dose_r10 = float(winner_r10["group_dose"])
global_profile_r10, group_profiles_r10, group_counts_r10 = banks_r10[
    (base_r10, reducer_r10)
]

raw_metrics_r10, _, raw_rows_r10 = _evaluate_r10(
    confirmation_records_r10,
    base_paths_r10,
    group_map_r10,
    base_r10,
    grid_r10,
    global_profile_r10,
    group_profiles_r10,
    0.0,
    0.0,
)
matched_metrics_r10, matched_covered_r10, matched_rows_r10 = _evaluate_r10(
    confirmation_records_r10,
    base_paths_r10,
    group_map_r10,
    base_r10,
    grid_r10,
    global_profile_r10,
    group_profiles_r10,
    global_dose_r10,
    0.0,
)
group_metrics_r10, group_covered_r10, group_rows_r10 = _evaluate_r10(
    confirmation_records_r10,
    base_paths_r10,
    group_map_r10,
    base_r10,
    grid_r10,
    global_profile_r10,
    group_profiles_r10,
    global_dose_r10,
    group_dose_r10,
)

rng_r10 = np.random.default_rng(SEED + 10001)
original_groups_r10 = np.asarray(
    [group_map_r10[record.well] for record in confirmation_records_r10], np.int64
)
covered_positions_r10 = np.asarray(
    [
        index
        for index, group in enumerate(original_groups_r10)
        if int(group) in group_profiles_r10
    ],
    np.int64,
)
shuffled_groups_r10 = np.full(len(confirmation_records_r10), -1, np.int64)
if len(covered_positions_r10):
    shuffled_groups_r10[covered_positions_r10] = rng_r10.permutation(
        original_groups_r10[covered_positions_r10]
    )
shuffled_metrics_r10, _, _ = _evaluate_r10(
    confirmation_records_r10,
    base_paths_r10,
    group_map_r10,
    base_r10,
    grid_r10,
    global_profile_r10,
    group_profiles_r10,
    global_dose_r10,
    group_dose_r10,
    shuffled_groups=shuffled_groups_r10.tolist(),
)
oracle_metrics_r10, oracle_weights_r10 = _oracle_between_r10(
    matched_rows_r10, group_rows_r10
)
perwell_r10 = _perwell_comparison_r10(
    confirmation_records_r10,
    group_map_r10,
    group_counts_r10,
    matched_rows_r10,
    group_rows_r10,
    oracle_weights_r10,
)
perwell_r10.to_csv(OUT_DIR_R10 / "confirmation_per_well.csv", index=False)

gain_r10 = matched_metrics_r10["rmse"] - group_metrics_r10["rmse"]
covered_gain_r10 = (
    matched_covered_r10["rmse"] - group_covered_r10["rmse"]
    if matched_covered_r10 is not None and group_covered_r10 is not None
    else math.nan
)
shuffle_degradation_r10 = shuffled_metrics_r10["rmse"] - group_metrics_r10["rmse"]
oracle_gain_r10 = matched_metrics_r10["rmse"] - oracle_metrics_r10["rmse"]
p95_degradation_r10 = (
    group_metrics_r10["p95_well_rmse"] - matched_metrics_r10["p95_well_rmse"]
)

checks_r10 = {
    "selection_positive_group_dose": group_dose_r10 > 0.0,
    "selection_gain_vs_matched_gt_0": winner_r10["gain_vs_matched"] > 0.0,
    "selection_covered_gain_vs_matched_gt_0": winner_r10[
        "covered_gain_vs_matched"
    ]
    > 0.0,
    "confirmation_gain_vs_matched_ge_0p10": gain_r10 >= 0.10,
    "covered_subset_gain_ge_0p20": covered_gain_r10 >= 0.20,
    "shuffled_degradation_ge_0p10": shuffle_degradation_r10 >= 0.10,
    "perwell_oracle_gain_ge_0p30": oracle_gain_r10 >= 0.30,
    "p95_degradation_le_0p20": p95_degradation_r10 <= 0.20,
    "absolute_rmse_not_worse_than_raw_base": group_metrics_r10["rmse"]
    <= raw_metrics_r10["rmse"],
}
strict_pass_r10 = bool(all(checks_r10.values()))

comparison_r10 = pd.DataFrame(
    [
        {"variant": "raw_base", **raw_metrics_r10},
        {"variant": "matched_global_residual", **matched_metrics_r10},
        {"variant": "typewell_group_residual", **group_metrics_r10},
        {"variant": "shuffled_group_residual", **shuffled_metrics_r10},
        {"variant": "perwell_group_dose_oracle", **oracle_metrics_r10},
    ]
)
comparison_r10.to_csv(OUT_DIR_R10 / "confirmation_comparison.csv", index=False)
print("SEGN-R10 CONFIRMATION")
print(comparison_r10.to_string(index=False), flush=True)

verdict_r10 = {
    "experiment": EXPERIMENT_R10,
    "stage": "fold3_typewell_group_residual_survival_gateway",
    "optimizer_steps": 0,
    "uses_neighbor_coordinates": False,
    "uses_input_only_typewell_grouping": True,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS_R10),
        "validation_fold": VALID_FOLD_R10,
        "selection_wells": len(selection_records_r10),
        "confirmation_wells": len(confirmation_records_r10),
    },
    "grouping": group_summary_r10,
    "candidate_pool": pool_alignment_r10,
    "selected": winner_r10,
    "confirmation": {
        "raw_base": raw_metrics_r10,
        "matched_global_residual": matched_metrics_r10,
        "typewell_group_residual": group_metrics_r10,
        "covered_matched": matched_covered_r10,
        "covered_group": group_covered_r10,
        "shuffled_group_residual": shuffled_metrics_r10,
        "perwell_group_dose_oracle": oracle_metrics_r10,
        "mean_oracle_group_weight": float(np.mean(oracle_weights_r10)),
        "gain_vs_matched": gain_r10,
        "covered_subset_gain": covered_gain_r10,
        "shuffle_degradation": shuffle_degradation_r10,
        "oracle_gain_vs_matched": oracle_gain_r10,
        "p95_degradation": p95_degradation_r10,
        "positive_well_fraction": float(np.mean(perwell_r10["rmse_gain"] > 0.0)),
        "covered_positive_well_fraction": float(
            np.mean(perwell_r10.loc[perwell_r10["covered"], "rmse_gain"] > 0.0)
        ),
    },
    "checks": checks_r10,
    "strict_pass": strict_pass_r10,
    "elapsed_minutes": (time.time() - started_r10) / 60.0,
    "next": (
        "PASS -> expose the group residual as a separately gated low-dose SEGN prior channel"
        if strict_pass_r10
        else "FAIL -> R9's group signal does not survive on a usable base; do not add typewell identity to SEGN"
    ),
}
_write_json_r10(OUT_DIR_R10 / "verdict.json", verdict_r10)
print("SEGN-R10 FINAL VERDICT", json.dumps(verdict_r10, indent=2), flush=True)
print("saved:", OUT_DIR_R10, flush=True)

if not strict_pass_r10:
    raise RuntimeError(json.dumps(verdict_r10, indent=2))
