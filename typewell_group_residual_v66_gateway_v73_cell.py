"""V73: typewell-group residual transfer on the complete V66 incumbent OOF.

R9 established that input-derived typewell groups contain real structural
information, but consumed it as an unusably weak standalone prior.  V73 tests
the safer position: a small, tail-only correction to the exact V66 mainline.

Every held fold is predicted by profiles fitted without that fold.  Candidate
settings are selected by four inner leave-one-fold evaluations.  A matched
group-label shuffle checks that any gain comes from the group identity rather
than generic residual smoothing.
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


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "typewell_group_residual_v66_gateway_v73"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V66_PATH = (
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
GROUP_CANDIDATES = (
    OOF_ROOT / "segn_r9_typewell_group_prior_oracle_v0" / "well_to_typewell_group.csv",
    ROOT / "well_to_typewell_group.csv",
)
TRAIN_DIR_CANDIDATES = (
    ROOT / "replay_data" / "train",
    ROOT / "data" / "train",
    ROOT / "train",
)

EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
FOLDS = (1, 2, 3, 4, 5)
BASE_COLUMN = "incumbent_6p377_replay"
PROFILE_POINTS = {"normalised": 129, "distance": 81}
MAX_DISTANCE_FT = 10_240.0
REDUCERS = ("mean", "median")
SHRINK_K = (1.0, 4.0, 8.0)
DOSES = (0.05, 0.10, 0.20, 0.30)
STARTS_FT = (1000.0, 2000.0)
TRANSITION_FT = 500.0
CAP_FT = 24.0
MIN_GROUP_TRAIN_WELLS = 2
MAX_INNER_FOLD_LOSS = 0.03
MAX_INNER_P95_CHANGE = 0.20
SEED = 20260729


def metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    grouped = pd.DataFrame({"well": frame["well"], "sse": error * error})
    well_rmse = np.sqrt(grouped.groupby("well", sort=False)["sse"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
        "near_le1000_rmse": float(np.sqrt(np.mean(error[distance <= 1000.0] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def smoothstep(distance: np.ndarray, start_ft: float) -> np.ndarray:
    value = np.clip((distance - start_ft) / TRANSITION_FT, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _typewell_shingles(well: str, train_dir: Path) -> set[bytes]:
    path = train_dir / f"{well}__typewell.csv"
    values = pd.to_numeric(pd.read_csv(path, usecols=["GR"])["GR"], errors="coerce")
    values = np.round(np.nan_to_num(values.to_numpy(np.float64), nan=9.87654321e8), 4)
    width = 12
    return {
        hashlib.blake2b(values[i:i + width].astype("<f8").tobytes(), digest_size=8).digest()
        for i in range(max(0, len(values) - width + 1))
    }


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def recompute_groups(wells: list[str]) -> pd.DataFrame:
    train_dir = next((path for path in TRAIN_DIR_CANDIDATES if path.is_dir()), None)
    if train_dir is None:
        raise FileNotFoundError(
            "Missing R9 group map and raw train directory; upload well_to_typewell_group.csv"
        )
    shingles, inverted = [], defaultdict(list)
    for number, well in enumerate(wells, 1):
        tokens = _typewell_shingles(well, train_dir)
        shingles.append(tokens)
        for token in tokens:
            inverted[token].append(number - 1)
        if number % 100 == 0 or number == len(wells):
            print(f"V73 typewell shingles {number}/{len(wells)}", flush=True)
    intersections = Counter()
    for members in inverted.values():
        if 1 < len(members) <= 30:
            for position, left in enumerate(members):
                for right in members[position + 1:]:
                    intersections[(left, right)] += 1
    union = UnionFind(len(wells))
    for (left, right), shared in intersections.items():
        overlap = shared / max(1, min(len(shingles[left]), len(shingles[right])))
        if shared >= 50 and overlap >= 0.50:
            union.union(left, right)
    roots = [union.find(i) for i in range(len(wells))]
    members = defaultdict(list)
    for index, root in enumerate(roots):
        members[root].append(index)
    ordered = sorted(members, key=lambda root: (-len(members[root]), min(members[root])))
    ids = {root: index for index, root in enumerate(ordered)}
    groups = [ids[union.find(i)] for i in range(len(wells))]
    sizes = Counter(groups)
    return pd.DataFrame({
        "well": wells,
        "typewell_group": groups,
        "typewell_group_size": [sizes[group] for group in groups],
    })


def load_groups(wells: list[str]) -> tuple[pd.DataFrame, str]:
    path = next((candidate for candidate in GROUP_CANDIDATES if candidate.is_file()), None)
    if path is None:
        frame = recompute_groups(wells)
        source = "recomputed_input_only_GR_shingles"
    else:
        frame = pd.read_csv(path, dtype={"well": str})
        source = str(path)
    frame = frame[["well", "typewell_group"]].drop_duplicates("well")
    frame["well"] = frame["well"].astype(str)
    if set(frame["well"]) != set(wells):
        raise RuntimeError(
            f"Typewell group coverage mismatch: groups={frame['well'].nunique()} OOF={len(wells)}"
        )
    return frame, source


def profile_grid(mode: str) -> np.ndarray:
    if mode == "normalised":
        return np.linspace(0.0, 1.0, PROFILE_POINTS[mode])
    return np.linspace(0.0, MAX_DISTANCE_FT, PROFILE_POINTS[mode])


def coordinate(frame: pd.DataFrame, mode: str) -> np.ndarray:
    distance = frame["md_since"].to_numpy(np.float64)
    if mode == "distance":
        return np.clip(distance, 0.0, MAX_DISTANCE_FT)
    span = max(float(distance[-1]), 1.0)
    return np.clip(distance / span, 0.0, 1.0)


def well_profile(frame: pd.DataFrame, mode: str, grid: np.ndarray) -> np.ndarray:
    x = coordinate(frame, mode)
    residual = frame["tvt_true"].to_numpy(np.float64) - frame[BASE_COLUMN].to_numpy(np.float64)
    x = np.r_[0.0, x]
    residual = np.r_[0.0, residual]
    keep = np.r_[True, np.diff(x) > 1e-9]
    return np.interp(grid, x[keep], residual[keep])


def fit_bank(
    train_wells: list[str], profiles: dict[tuple[str, str], np.ndarray],
    group_by_well: dict[str, int], mode: str, reducer: str,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    values = [profiles[(well, mode)] for well in train_wells]
    reduction = np.mean if reducer == "mean" else np.median
    global_profile = reduction(np.stack(values), axis=0)
    grouped = defaultdict(list)
    for well in train_wells:
        grouped[group_by_well[well]].append(profiles[(well, mode)])
    group_profiles = {
        group: reduction(np.stack(items), axis=0)
        for group, items in grouped.items() if len(items) >= MIN_GROUP_TRAIN_WELLS
    }
    counts = {group: len(items) for group, items in grouped.items()}
    return global_profile, group_profiles, counts


def predict_fold(
    frame: pd.DataFrame, wells: list[str], mode: str, reducer: str,
    shrink_k: float, dose: float, start_ft: float,
    profiles: dict[tuple[str, str], np.ndarray], group_by_well: dict[str, int],
    train_wells: list[str], shuffled: bool = False, seed: int = SEED,
) -> np.ndarray:
    grid = profile_grid(mode)
    global_profile, group_profiles, counts = fit_bank(
        train_wells, profiles, group_by_well, mode, reducer
    )
    prediction = frame[BASE_COLUMN].to_numpy(np.float64).copy()
    target_groups = np.asarray([group_by_well[well] for well in wells], np.int64)
    if shuffled:
        rng = np.random.default_rng(seed)
        target_groups = rng.permutation(target_groups)
    positions = frame.groupby("well", sort=False).indices
    for well, selected_group in zip(wells, target_groups):
        indices = np.asarray(positions[well], np.int64)
        group_profile = group_profiles.get(int(selected_group))
        if group_profile is None:
            continue
        count = counts[int(selected_group)]
        shrink = count / (count + shrink_k)
        local = frame.iloc[indices]
        delta = np.interp(coordinate(local, mode), grid, group_profile - global_profile)
        delta = np.clip(shrink * delta, -CAP_FT, CAP_FT)
        gate = smoothstep(local["md_since"].to_numpy(np.float64), start_ft)
        prediction[indices] += dose * gate * delta
    return prediction


def config_id(config: dict) -> str:
    return (
        f"{config['mode']}_{config['reducer']}_k{config['shrink_k']:g}_"
        f"d{config['dose']:g}_s{config['start_ft']:g}"
    )


if not V66_PATH.is_file():
    raise FileNotFoundError(V66_PATH)
columns = ["id", "well", "row_idx", "fold", "tvt_true", "md_since", BASE_COLUMN]
frame = pd.read_parquet(V66_PATH, columns=columns)
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if len(frame) != EXPECTED_ROWS or frame["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(f"Unexpected V66 coverage: {len(frame)}/{frame['well'].nunique()}")
if frame[columns[4:]].isna().any().any() or frame["id"].duplicated().any():
    raise RuntimeError("Invalid V66 alignment or non-finite values")

wells = frame.groupby("well", sort=False)["fold"].first().index.astype(str).tolist()
fold_by_well = frame.groupby("well", sort=False)["fold"].first().astype(int).to_dict()
fold_coverage = Counter(fold_by_well.values())
if set(fold_coverage) != set(FOLDS) or min(fold_coverage.values()) < 154:
    raise RuntimeError(f"Non-canonical fold coverage: {dict(fold_coverage)}")

group_frame, group_source = load_groups(wells)
group_frame.to_csv(OUT_DIR / "well_to_typewell_group.csv", index=False)
group_by_well = dict(zip(group_frame["well"], group_frame["typewell_group"].astype(int)))
group_sizes = Counter(group_by_well.values())

well_frames = {well: local for well, local in frame.groupby("well", sort=False)}
profiles = {}
for number, well in enumerate(wells, 1):
    for mode in PROFILE_POINTS:
        profiles[(well, mode)] = well_profile(well_frames[well], mode, profile_grid(mode))
    if number % 100 == 0 or number == len(wells):
        print(f"V73 profiles {number}/{len(wells)}", flush=True)

configs = [{
    "mode": "normalised", "reducer": "median", "shrink_k": 4.0,
    "dose": 0.0, "start_ft": 1000.0,
}] + [
    {"mode": mode, "reducer": reducer, "shrink_k": shrink_k,
     "dose": dose, "start_ft": start_ft}
    for mode in PROFILE_POINTS
    for reducer in REDUCERS
    for shrink_k in SHRINK_K
    for dose in DOSES
    for start_ft in STARTS_FT
]

base = frame[BASE_COLUMN].to_numpy(np.float64)
crossfit = base.copy()
shuffle_crossfit = base.copy()
fold_reports, selection_rows = [], []

for held_fold in FOLDS:
    outer_train_folds = [fold for fold in FOLDS if fold != held_fold]
    inner_reports = defaultdict(list)
    for selection_fold in outer_train_folds:
        train_wells = [
            well for well in wells
            if fold_by_well[well] not in (held_fold, selection_fold)
        ]
        valid_wells = [well for well in wells if fold_by_well[well] == selection_fold]
        valid_mask = frame["fold"].to_numpy() == selection_fold
        valid_frame = frame.loc[valid_mask].reset_index(drop=True)
        base_metrics = metric(valid_frame, valid_frame[BASE_COLUMN].to_numpy(), "base")
        for config in configs:
            prediction = predict_fold(
                valid_frame, valid_wells, **config, profiles=profiles,
                group_by_well=group_by_well, train_wells=train_wells,
            )
            result = metric(valid_frame, prediction, config_id(config))
            inner_reports[config_id(config)].append({
                "fold": selection_fold,
                "gain": base_metrics["rmse"] - result["rmse"],
                "far_gain": base_metrics["far_gt2000_rmse"] - result["far_gt2000_rmse"],
                "p95_change": result["p95_well_rmse"] - base_metrics["p95_well_rmse"],
                "sse": result["exact_sse"], "rows": result["rows"],
            })

    candidates = []
    by_id = {config_id(config): config for config in configs}
    for identifier, reports in inner_reports.items():
        pooled_rmse = math.sqrt(sum(item["sse"] for item in reports) / sum(item["rows"] for item in reports))
        gains = [item["gain"] for item in reports]
        p95_changes = [item["p95_change"] for item in reports]
        robust = min(gains) >= -MAX_INNER_FOLD_LOSS and max(p95_changes) <= MAX_INNER_P95_CHANGE
        row = {
            "held_fold": held_fold, "config": identifier, "inner_rmse": pooled_rmse,
            "minimum_inner_gain": min(gains), "mean_inner_gain": float(np.mean(gains)),
            "minimum_inner_far_gain": min(item["far_gain"] for item in reports),
            "maximum_inner_p95_change": max(p95_changes), "robust": robust,
        }
        candidates.append(row)
        selection_rows.append(row)
    robust = [item for item in candidates if item["robust"]]
    selected_row = min(robust or candidates, key=lambda item: item["inner_rmse"])
    selected = by_id[selected_row["config"]]

    held_wells = [well for well in wells if fold_by_well[well] == held_fold]
    held_mask = frame["fold"].to_numpy() == held_fold
    held_positions = np.flatnonzero(held_mask)
    held_frame = frame.loc[held_mask].reset_index(drop=True)
    outer_train_wells = [well for well in wells if fold_by_well[well] != held_fold]
    prediction = predict_fold(
        held_frame, held_wells, **selected, profiles=profiles,
        group_by_well=group_by_well, train_wells=outer_train_wells,
    )
    shuffled = predict_fold(
        held_frame, held_wells, **selected, profiles=profiles,
        group_by_well=group_by_well, train_wells=outer_train_wells,
        shuffled=True, seed=SEED + held_fold,
    )
    crossfit[held_positions] = prediction
    shuffle_crossfit[held_positions] = shuffled
    base_metrics = metric(held_frame, held_frame[BASE_COLUMN].to_numpy(), "base")
    result = metric(held_frame, prediction, "group_residual")
    shuffled_result = metric(held_frame, shuffled, "group_shuffle")
    fold_reports.append({
        "fold": held_fold, "selected": config_id(selected),
        "inner_rmse": selected_row["inner_rmse"],
        "minimum_inner_gain": selected_row["minimum_inner_gain"],
        "base_rmse": base_metrics["rmse"], "rmse": result["rmse"],
        "gain": base_metrics["rmse"] - result["rmse"],
        "base_far_rmse": base_metrics["far_gt2000_rmse"],
        "far_rmse": result["far_gt2000_rmse"],
        "far_gain": base_metrics["far_gt2000_rmse"] - result["far_gt2000_rmse"],
        "p95_change": result["p95_well_rmse"] - base_metrics["p95_well_rmse"],
        "shuffle_loss": shuffled_result["rmse"] - result["rmse"],
    })
    print("V73 FOLD", fold_reports[-1], flush=True)

selection = pd.DataFrame(selection_rows)
selection.to_csv(OUT_DIR / "nested_selection_scan.csv", index=False)
fold_table = pd.DataFrame(fold_reports)
fold_table.to_csv(OUT_DIR / "fold_results.csv", index=False)

base_metrics = metric(frame, base, "v66_base")
candidate_metrics = metric(frame, crossfit, "typewell_group_residual")
shuffle_metrics = metric(frame, shuffle_crossfit, "matched_group_shuffle")
gain = base_metrics["rmse"] - candidate_metrics["rmse"]
far_gain = base_metrics["far_gt2000_rmse"] - candidate_metrics["far_gt2000_rmse"]
shuffle_loss = shuffle_metrics["rmse"] - candidate_metrics["rmse"]

output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["v66_base"] = base.astype(np.float32)
output["typewell_group_residual"] = crossfit.astype(np.float32)
output["matched_group_shuffle"] = shuffle_crossfit.astype(np.float32)
output.to_parquet(OUT_DIR / "typewell_group_residual_oof.parquet", index=False, compression="zstd")

checks = {
    "overall_gain_ge_0p03": gain >= 0.03,
    "far_gain_ge_0p05": far_gain >= 0.05,
    "positive_folds_ge_4": int((fold_table["gain"] > 0).sum()) >= 4,
    "minimum_fold_gain_ge_minus0p03": float(fold_table["gain"].min()) >= -0.03,
    "maximum_p95_change_le_0p20": float(fold_table["p95_change"].max()) <= 0.20,
    "shuffle_loses_ge_0p02": shuffle_loss >= 0.02,
    "near_rows_bitwise_unchanged": bool(np.array_equal(base[frame["md_since"] <= 1000.0], crossfit[frame["md_since"] <= 1000.0])),
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "canonical_fivefold_nested_typewell_group_residual_transfer",
    "sample_unit": "complete physical well",
    "source": str(V66_PATH),
    "group_source": group_source,
    "grouping": {
        "groups": len(group_sizes),
        "duplicate_groups": int(sum(size > 1 for size in group_sizes.values())),
        "wells_in_duplicate_groups": int(sum(size for size in group_sizes.values() if size > 1)),
        "maximum_group_size": int(max(group_sizes.values())),
    },
    "protocol": {
        "outer_folds": list(FOLDS),
        "inner_selection": "four leave-one-inner-fold predictions per outer fold",
        "prefix_frozen_through_ft": 1000.0,
        "truth_usage": "outer-train residual profiles only",
    },
    "base": base_metrics,
    "candidate": candidate_metrics,
    "matched_group_shuffle": shuffle_metrics,
    "gain": gain,
    "far_gain": far_gain,
    "shuffle_loss": shuffle_loss,
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> fit full-data group residual profiles and package a frozen low-dose tail correction"
        if strict_pass else
        "FAIL -> group identity is real but does not survive on the V66 residual; close cohort transfer"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nV73 FOLDS")
print(fold_table.to_string(index=False), flush=True)
print("\nV73 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
