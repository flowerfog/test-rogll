"""Full-reference projection and drilling-direction audit.

Prerequisite: neighbor_fullcurve_transfer_v0_cell.py has completed.

The v0 neighbor, coherent600 gate, weight, and movement cap are frozen. This
cell only changes how the selected reference well is mapped onto the target:

* progress_v0 / xy_v0: exact controls from the v0 sidecar;
* full_progress_dot: full reference path, direction chosen by direction dot;
* full_progress_geom: full path, forward/reverse chosen by mapped XY error;
* full_xy_raw: nearest point on the complete reference trajectory;
* full_xy_monotonic: nearest-point progress made monotonic with isotonic fit;
* full_along_axis: target drilling-axis coordinate mapped to the reference;
* hybrid_geom_monotonic: mean of geometry-direction and monotonic mappings.

Target TVT truth is not read during mapping. It is used only after all curves
and direction decisions have been frozen.
"""

from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE_DIR = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
OUT_DIR = OOF_ROOT / "neighbor_fullcurve_projection_direction_v1"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)

N_FOLDS = 5
RANDOM_STATE = 20260716
MIN_ABS_DIRECTION_DOT = 0.70
SAFE_WEIGHT = 0.25
RAW_MOVE_CAP = 24.0

CANDIDATE_NAMES = (
    "progress_v0",
    "xy_v0",
    "full_progress_dot",
    "full_progress_geom",
    "full_xy_raw",
    "full_xy_monotonic",
    "full_along_axis",
    "hybrid_geom_monotonic",
)
PRIMARY_CANDIDATE = "full_progress_geom"

if POOL_PATH is None:
    raise FileNotFoundError("candidate pool not found")
for path in (
    SOURCE_DIR / "neighbor_transfer_oof.npz",
    SOURCE_DIR / "selected_neighbors.csv",
):
    if not path.exists():
        raise FileNotFoundError(path)
if not (DATA_ROOT / "train").exists():
    raise FileNotFoundError(DATA_ROOT / "train")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rmse(y, prediction):
    y = np.asarray(y, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _cumulative_xy(xy):
    step = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    step[~np.isfinite(step)] = 0.0
    return np.r_[0.0, np.cumsum(step)]


def _unique_progress(progress, *arrays):
    """Keep the last point of each repeated XY-progress run."""
    progress = np.asarray(progress, np.float64)
    keep = np.r_[np.diff(progress) > 1e-8, True]
    output = [progress[keep]]
    output.extend(np.asarray(array)[keep] for array in arrays)
    return output


def _geometry_score(target_xy, mapped_xy, valid):
    valid = np.asarray(valid, bool) & np.isfinite(mapped_xy).all(axis=1)
    if not valid.any():
        return np.inf, 0.0
    distance = np.linalg.norm(mapped_xy[valid] - target_xy[valid], axis=1)
    coverage = float(valid.mean())
    score = (
        float(np.median(distance))
        + 0.25 * float(np.quantile(distance, 0.90))
        + 1000.0 * (1.0 - coverage)
    )
    return score, coverage


def _fill_with_fallback(values, fallback):
    values = np.asarray(values, np.float64).copy()
    fallback = np.asarray(fallback, np.float64)
    bad = ~np.isfinite(values)
    values[bad] = fallback[bad]
    return values


print(
    {
        "experiment": "neighbor_fullcurve_projection_direction_v1",
        "source": str(SOURCE_DIR),
        "pool": str(POOL_PATH),
        "candidates": CANDIDATE_NAMES,
        "primary": PRIMARY_CANDIDATE,
        "frozen_apply": {
            "weight": SAFE_WEIGHT,
            "raw_cap": RAW_MOVE_CAP,
            "applied_cap": SAFE_WEIGHT * RAW_MOVE_CAP,
        },
    }
)


# ---------------------------------------------------------------------------
# Established pool order and v0 controls.
# ---------------------------------------------------------------------------
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
base_cols = [column for column in ("strong", "strong_bias", "cp12") if column in header]
if not {"id", "tvt_true", "strong"}.issubset(header):
    raise RuntimeError("candidate pool misses id/tvt_true/strong")
usecols = [
    column
    for column in ("id", "well", "row_idx", "tvt_true", *base_cols)
    if column in header
]
pool = pd.read_csv(
    POOL_PATH,
    usecols=usecols,
    dtype={"id": "string", "well": "string"},
)
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
pool["well"] = pool["well"].astype(str)
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)

wells = sorted(pool["well"].unique())
pool_indices = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in pool.groupby("well", sort=False).indices.items()
}
pool_row_idx = pool["row_idx"].to_numpy(np.int32)
truth = pool["tvt_true"].to_numpy(np.float64)

sidecar = np.load(SOURCE_DIR / "neighbor_transfer_oof.npz", allow_pickle=False)
progress_v0 = sidecar["structure_progress"].astype(np.float64)
xy_v0 = sidecar["structure_xy"].astype(np.float64)
if len(progress_v0) != len(pool) or len(xy_v0) != len(pool):
    raise RuntimeError("v0 sidecar row mismatch")

selected = pd.read_csv(SOURCE_DIR / "selected_neighbors.csv")
selected["well"] = selected["well"].astype(str)
selected["neighbor"] = selected["neighbor"].astype(str)
selected_meta = selected.set_index("well")

eligible_well = (
    (selected_meta["median_distance"] <= 600.0)
    & (selected_meta["p90_distance"] <= 1200.0)
    & (selected_meta["anchor_distance_full"] <= 1200.0)
    & (selected_meta["abs_direction_dot"] >= MIN_ABS_DIRECTION_DOT)
)
eligible_map = eligible_well.to_dict()


# ---------------------------------------------------------------------------
# Load complete labeled reference curves. Target mapping uses only geometry,
# Z, and its visible anchor TVT.
# ---------------------------------------------------------------------------
well_data = {}
load_failures = []
for well in tqdm(wells, desc="load complete reference paths"):
    try:
        path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        frame = pd.read_csv(path, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
        visible = frame["TVT_input"].notna().to_numpy()
        if not visible.any() or visible.all():
            raise ValueError("invalid TVT_input mask")
        anchor = int(np.flatnonzero(visible)[-1])
        xy = frame[["X", "Y"]].to_numpy(np.float64)
        z = frame["Z"].to_numpy(np.float64)
        tvt = frame["TVT"].to_numpy(np.float64)
        progress = _cumulative_xy(xy)
        direction = xy[-1] - xy[anchor]
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0:
            raise ValueError("degenerate direction")
        direction /= norm
        unique_progress, unique_structure, unique_x, unique_y = _unique_progress(
            progress,
            tvt + z,
            xy[:, 0],
            xy[:, 1],
        )
        well_data[well] = {
            "xy": xy,
            "z": z,
            "tvt": tvt,
            "structure": tvt + z,
            "anchor": anchor,
            "anchor_tvt": float(frame["TVT_input"].iloc[anchor]),
            "direction": direction,
            "progress": progress,
            "unique_progress": unique_progress,
            "unique_structure": unique_structure,
            "unique_x": unique_x,
            "unique_y": unique_y,
            "tree": cKDTree(xy, leafsize=64),
        }
    except Exception as exc:
        load_failures.append({"well": well, "reason": repr(exc)})

pd.DataFrame(load_failures).to_csv(OUT_DIR / "load_failures.csv", index=False)
if len(well_data) < len(wells) - 5:
    raise RuntimeError(f"loaded only {len(well_data)} / {len(wells)} wells")


def _structure_to_target_tvt(structure, target_z, target, ref_anchor_structure):
    target_anchor_z = float(target["z"][target["anchor"]])
    ref_anchor_tvt_on_target = float(ref_anchor_structure - target_anchor_z)
    return (
        np.asarray(structure, np.float64)
        - target_z
        + (target["anchor_tvt"] - ref_anchor_tvt_on_target)
    )


def _progress_mapping(target, reference, target_rows, sign):
    target_xy = target["xy"][target_rows]
    target_z = target["z"][target_rows]
    target_progress_all = _cumulative_xy(target["xy"][target["anchor"] :])
    target_progress = target_progress_all[target_rows - target["anchor"]]
    anchor_xy = target["xy"][target["anchor"]]
    _, ref_anchor_index = reference["tree"].query(anchor_xy, k=1)
    ref_anchor_index = int(ref_anchor_index)
    ref_anchor_progress = float(reference["progress"][ref_anchor_index])
    query = ref_anchor_progress + float(sign) * target_progress
    valid = (query >= reference["unique_progress"][0]) & (
        query <= reference["unique_progress"][-1]
    )
    mapped_structure = np.full(len(target_rows), np.nan, np.float64)
    mapped_xy = np.full((len(target_rows), 2), np.nan, np.float64)
    if valid.any():
        q = query[valid]
        mapped_structure[valid] = np.interp(
            q, reference["unique_progress"], reference["unique_structure"]
        )
        mapped_xy[valid, 0] = np.interp(
            q, reference["unique_progress"], reference["unique_x"]
        )
        mapped_xy[valid, 1] = np.interp(
            q, reference["unique_progress"], reference["unique_y"]
        )
    ref_anchor_structure = float(reference["structure"][ref_anchor_index])
    candidate = _structure_to_target_tvt(
        mapped_structure, target_z, target, ref_anchor_structure
    )
    score, coverage = _geometry_score(target_xy, mapped_xy, valid)
    return candidate, mapped_xy, valid, score, coverage, ref_anchor_index


def _raw_xy_mapping(target, reference, target_rows):
    target_xy = target["xy"][target_rows]
    target_z = target["z"][target_rows]
    _, nearest = reference["tree"].query(target_xy, k=1)
    _, anchor_nearest = reference["tree"].query(
        target["xy"][target["anchor"]], k=1
    )
    nearest = nearest.astype(np.int64)
    anchor_nearest = int(anchor_nearest)
    mapped_structure = reference["structure"][nearest]
    return _structure_to_target_tvt(
        mapped_structure,
        target_z,
        target,
        reference["structure"][anchor_nearest],
    ), nearest, anchor_nearest


def _monotonic_xy_mapping(target, reference, target_rows, sign, raw_nearest, anchor_nearest):
    target_z = target["z"][target_rows]
    target_progress_all = _cumulative_xy(target["xy"][target["anchor"] :])
    target_progress = target_progress_all[target_rows - target["anchor"]]
    raw_progress = reference["progress"][raw_nearest]
    anchor_progress = float(reference["progress"][anchor_nearest])
    x = np.r_[0.0, target_progress]
    y = np.r_[anchor_progress, raw_progress]
    sample_weight = np.r_[100.0, np.ones(len(target_progress))]
    model = IsotonicRegression(
        increasing=bool(sign > 0), out_of_bounds="clip"
    )
    fitted = model.fit_transform(x, y, sample_weight=sample_weight)[1:]
    fitted = np.clip(
        fitted,
        reference["unique_progress"][0],
        reference["unique_progress"][-1],
    )
    mapped_structure = np.interp(
        fitted, reference["unique_progress"], reference["unique_structure"]
    )
    return _structure_to_target_tvt(
        mapped_structure,
        target_z,
        target,
        reference["structure"][anchor_nearest],
    )


def _along_axis_mapping(target, reference, target_rows, anchor_nearest):
    target_xy = target["xy"][target_rows]
    target_z = target["z"][target_rows]
    axis = target["direction"]
    target_anchor_xy = target["xy"][target["anchor"]]
    reference_anchor_xy = reference["xy"][anchor_nearest]
    query = (target_xy - target_anchor_xy) @ axis
    reference_coordinate = (reference["xy"] - reference_anchor_xy) @ axis
    order = np.argsort(reference_coordinate, kind="mergesort")
    coordinate = reference_coordinate[order]
    structure = reference["structure"][order]
    keep = np.r_[np.diff(coordinate) > 1e-8, True]
    coordinate = coordinate[keep]
    structure = structure[keep]
    valid = (query >= coordinate[0]) & (query <= coordinate[-1])
    mapped_structure = np.full(len(target_rows), np.nan, np.float64)
    if valid.any():
        mapped_structure[valid] = np.interp(query[valid], coordinate, structure)
    return _structure_to_target_tvt(
        mapped_structure,
        target_z,
        target,
        reference["structure"][anchor_nearest],
    )


# ---------------------------------------------------------------------------
# Freeze all mappings well by well.
# ---------------------------------------------------------------------------
candidates = {
    "progress_v0": progress_v0.astype(np.float32).copy(),
    "xy_v0": xy_v0.astype(np.float32).copy(),
}
for name in CANDIDATE_NAMES:
    candidates.setdefault(name, progress_v0.astype(np.float32).copy())

direction_rows = []
mapping_failures = []
started = time.time()
for done, target_well in enumerate(tqdm(wells, desc="full-path mappings"), 1):
    try:
        if target_well not in well_data or target_well not in selected_meta.index:
            continue
        reference_well = str(selected_meta.loc[target_well, "neighbor"])
        if reference_well not in well_data:
            continue
        target = well_data[target_well]
        reference = well_data[reference_well]
        output_indices = pool_indices[target_well]
        target_rows = pool_row_idx[output_indices]
        fallback = progress_v0[output_indices]
        target_xy = target["xy"][target_rows]

        positive = _progress_mapping(target, reference, target_rows, +1.0)
        negative = _progress_mapping(target, reference, target_rows, -1.0)
        dot_sign = (
            +1.0
            if float(np.dot(target["direction"], reference["direction"])) >= 0
            else -1.0
        )
        dot_result = positive if dot_sign > 0 else negative
        geom_sign = +1.0 if positive[3] <= negative[3] else -1.0
        geom_result = positive if geom_sign > 0 else negative

        raw_xy, raw_nearest, anchor_nearest = _raw_xy_mapping(
            target, reference, target_rows
        )
        monotonic = _monotonic_xy_mapping(
            target,
            reference,
            target_rows,
            geom_sign,
            raw_nearest,
            anchor_nearest,
        )
        along_axis = _along_axis_mapping(
            target, reference, target_rows, anchor_nearest
        )
        hybrid = np.where(
            np.isfinite(geom_result[0]) & np.isfinite(monotonic),
            0.5 * geom_result[0] + 0.5 * monotonic,
            np.where(np.isfinite(geom_result[0]), geom_result[0], monotonic),
        )

        generated = {
            "full_progress_dot": dot_result[0],
            "full_progress_geom": geom_result[0],
            "full_xy_raw": raw_xy,
            "full_xy_monotonic": monotonic,
            "full_along_axis": along_axis,
            "hybrid_geom_monotonic": hybrid,
        }
        for name, values in generated.items():
            candidates[name][output_indices] = _fill_with_fallback(
                values, fallback
            ).astype(np.float32)

        direction_rows.append(
            {
                "well": target_well,
                "neighbor": reference_well,
                "direction_dot": float(
                    np.dot(target["direction"], reference["direction"])
                ),
                "dot_sign": int(dot_sign),
                "geometry_sign": int(geom_sign),
                "sign_changed": int(dot_sign != geom_sign),
                "positive_geometry_score": float(positive[3]),
                "negative_geometry_score": float(negative[3]),
                "dot_coverage": float(dot_result[4]),
                "geometry_coverage": float(geom_result[4]),
            }
        )
    except Exception as exc:
        mapping_failures.append({"well": target_well, "reason": repr(exc)})

    if done % 100 == 0:
        print(
            f"mappings {done}/{len(wells)} ok={len(direction_rows)} "
            f"elapsed_min={(time.time()-started)/60:.1f}",
            flush=True,
        )

direction_report = pd.DataFrame(direction_rows)
failure_report = pd.DataFrame(mapping_failures)
direction_report.to_csv(OUT_DIR / "direction_diagnostics.csv", index=False)
failure_report.to_csv(OUT_DIR / "mapping_failures.csv", index=False)
print(
    "direction summary:",
    {
        "mapped_wells": len(direction_report),
        "sign_changed_fraction": float(direction_report["sign_changed"].mean()),
        "dot_coverage_mean": float(direction_report["dot_coverage"].mean()),
        "geometry_coverage_mean": float(
            direction_report["geometry_coverage"].mean()
        ),
    },
)


# ---------------------------------------------------------------------------
# Stable folds and frozen coherent600 eligibility.
# ---------------------------------------------------------------------------
fold_candidates = [
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
]
fold_map = None
fold_source = None
for path in fold_candidates:
    if not path.exists():
        continue
    frame = pd.read_csv(path, dtype={"well": str})
    if {"well", "fold"}.issubset(frame.columns):
        current = frame.groupby("well")["fold"].first().astype(int).to_dict()
        if set(wells).issubset(current):
            fold_map = current
            fold_source = str(path)
            break
if fold_map is None:
    counts = pool.groupby("well", as_index=False).size().rename(columns={"size": "rows"})
    counts["stratum"] = pd.qcut(
        counts["rows"].rank(method="first"), 10, labels=False
    ).astype(int)
    counts["fold"] = -1
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    for fold, (_, valid) in enumerate(
        splitter.split(counts, counts["stratum"]), 1
    ):
        counts.loc[valid, "fold"] = fold
    fold_map = dict(zip(counts["well"].astype(str), counts["fold"].astype(int)))
    fold_source = "deterministic row-count StratifiedKFold"
fold_values = pool["well"].map(fold_map).to_numpy(np.int8)
eligible = pool["well"].map(eligible_map).fillna(False).to_numpy(bool)


# ---------------------------------------------------------------------------
# Fixed OOF scoring on each historical downstream stage.
# ---------------------------------------------------------------------------
score_rows = []
fold_rows = []
for base_col in base_cols:
    base = pool[base_col].to_numpy(np.float64)
    base_rmse = _rmse(truth, base)
    top40_sse = (
        pd.DataFrame({"well": pool["well"], "se": (base - truth) ** 2})
        .groupby("well", as_index=False)["se"]
        .sum()
    )
    top40_wells = set(top40_sse.nlargest(40, "se")["well"].astype(str))
    top40 = pool["well"].isin(top40_wells).to_numpy()

    for name in CANDIDATE_NAMES:
        candidate = candidates[name].astype(np.float64)
        valid = eligible & np.isfinite(candidate)
        move = np.zeros(len(pool), np.float64)
        move[valid] = SAFE_WEIGHT * np.clip(
            candidate[valid] - base[valid], -RAW_MOVE_CAP, RAW_MOVE_CAP
        )
        prediction = base + move
        gains = []
        for fold in range(1, N_FOLDS + 1):
            mask = fold_values == fold
            fold_base = _rmse(truth[mask], base[mask])
            fold_prediction = _rmse(truth[mask], prediction[mask])
            gain = fold_base - fold_prediction
            gains.append(gain)
            fold_rows.append(
                {
                    "base_col": base_col,
                    "candidate": name,
                    "fold": fold,
                    "rows": int(mask.sum()),
                    "base_rmse": fold_base,
                    "prediction_rmse": fold_prediction,
                    "gain": gain,
                }
            )
        prediction_rmse = _rmse(truth, prediction)
        score_rows.append(
            {
                "base_col": base_col,
                "candidate": name,
                "base_rmse": base_rmse,
                "prediction_rmse": prediction_rmse,
                "gain": base_rmse - prediction_rmse,
                "mean_fold_gain": float(np.mean(gains)),
                "min_fold_gain": float(np.min(gains)),
                "positive_folds": int(np.sum(np.asarray(gains) > 0)),
                "top40_gain": _rmse(truth[top40], base[top40])
                - _rmse(truth[top40], prediction[top40]),
                "mean_abs_move": float(np.mean(np.abs(move))),
                "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
                "max_abs_move": float(np.max(np.abs(move))),
            }
        )

scores = pd.DataFrame(score_rows)
fold_report = pd.DataFrame(fold_rows)


# Nested choice is diagnostic only; the fixed primary controls promotion.
nested_rows = []
for base_col in base_cols:
    for held_fold in range(1, N_FOLDS + 1):
        train = fold_report[
            (fold_report["base_col"] == base_col)
            & (fold_report["fold"] != held_fold)
        ]
        summary = (
            train.groupby("candidate", as_index=False)
            .agg(train_mean_gain=("gain", "mean"), train_min_gain=("gain", "min"))
            .sort_values(
                ["train_mean_gain", "train_min_gain", "candidate"],
                ascending=[False, False, True],
            )
        )
        selected_name = str(summary.iloc[0]["candidate"])
        valid = fold_report[
            (fold_report["base_col"] == base_col)
            & (fold_report["fold"] == held_fold)
            & (fold_report["candidate"] == selected_name)
        ].iloc[0]
        nested_rows.append(
            {
                "base_col": base_col,
                "fold": held_fold,
                "selected_candidate": selected_name,
                "train_mean_gain": float(summary.iloc[0]["train_mean_gain"]),
                "train_min_gain": float(summary.iloc[0]["train_min_gain"]),
                "valid_gain": float(valid["gain"]),
                "valid_base_rmse": float(valid["base_rmse"]),
                "valid_prediction_rmse": float(valid["prediction_rmse"]),
            }
        )
nested = pd.DataFrame(nested_rows)


def _score_row(base_col, candidate):
    return scores[
        (scores["base_col"] == base_col) & (scores["candidate"] == candidate)
    ].iloc[0]


primary_base = "cp12" if "cp12" in base_cols else "strong"
control = _score_row(primary_base, "progress_v0")
primary = _score_row(primary_base, PRIMARY_CANDIDATE)
control_folds = fold_report[
    (fold_report["base_col"] == primary_base)
    & (fold_report["candidate"] == "progress_v0")
].sort_values("fold")
primary_folds = fold_report[
    (fold_report["base_col"] == primary_base)
    & (fold_report["candidate"] == PRIMARY_CANDIDATE)
].sort_values("fold")
folds_better = int(
    (
        primary_folds["prediction_rmse"].to_numpy()
        < control_folds["prediction_rmse"].to_numpy()
    ).sum()
)

strict_pass = bool(
    float(primary["gain"]) >= float(control["gain"]) + 0.03
    and int(primary["positive_folds"]) == N_FOLDS
    and folds_better >= 4
    and float(primary["top40_gain"]) >= float(control["top40_gain"])
)

result = {
    "pool": str(POOL_PATH),
    "source": str(SOURCE_DIR),
    "fold_source": fold_source,
    "primary_base": primary_base,
    "primary_candidate": PRIMARY_CANDIDATE,
    "control_gain": float(control["gain"]),
    "primary_gain": float(primary["gain"]),
    "incremental_gain_vs_progress_v0": float(
        primary["gain"] - control["gain"]
    ),
    "control_top40_gain": float(control["top40_gain"]),
    "primary_top40_gain": float(primary["top40_gain"]),
    "primary_min_fold_gain": float(primary["min_fold_gain"]),
    "primary_positive_folds": int(primary["positive_folds"]),
    "folds_better_than_progress_v0": folds_better,
    "direction_sign_changed_fraction": float(
        direction_report["sign_changed"].mean()
    ),
    "dot_coverage_mean": float(direction_report["dot_coverage"].mean()),
    "geometry_coverage_mean": float(
        direction_report["geometry_coverage"].mean()
    ),
    "strict_pass": strict_pass,
    "next": (
        "PASS -> build deterministic Kaggle full-reference mapping"
        if strict_pass
        else "STOP projection/direction variants; continue to pad graph audit"
    ),
}

scores.to_csv(OUT_DIR / "scores.csv", index=False)
fold_report.to_csv(OUT_DIR / "folds.csv", index=False)
nested.to_csv(OUT_DIR / "nested_variant_selection.csv", index=False)
with open(OUT_DIR / "result.json", "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, ensure_ascii=True)
# Persist the frozen primary endpoint even when the historical promotion gate
# fails. Downstream artifact-first audits need the exact row vector; the flag
# prevents an audit sidecar from being mistaken for a promoted model.
np.savez_compressed(
    OUT_DIR / "primary_candidate_oof.npz",
    candidate=candidates[PRIMARY_CANDIDATE].astype(np.float32),
    strict_pass=np.asarray([strict_pass], np.uint8),
    primary_candidate=np.asarray([PRIMARY_CANDIDATE], str),
)

print("\nDIRECTION DIAGNOSTICS")
display(direction_report.describe(include="all"))
print("\nSCORES")
display(scores.sort_values(["base_col", "gain"], ascending=[True, False]))
print("\nFOLDS")
display(fold_report)
print("\nNESTED VARIANT SELECTION")
display(nested)
print("\nRESULT", result)
print("saved:", OUT_DIR)

gc.collect()
