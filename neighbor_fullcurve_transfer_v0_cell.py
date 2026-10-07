"""Leave-one-well-out full-curve neighbor-transfer kill-test.

The reference well is selected using only deployable trajectory geometry. Its
full labeled TVT curve is then transferred to the target and re-anchored at the
target's last visible TVT. This simulates deployment, where every hidden well
can use all 773 labeled training wells as references.

Direction is used only to orient progress-based transfer. It is not used to
split or select a supervised model.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / "neighbor_fullcurve_transfer_v0"

BASE_CANDIDATES = ("strong_bias", "cp12", "strong")
TRANSFER_METHODS = (
    "structure_xy",
    "tvt_xy",
    "structure_progress",
    "tvt_progress",
)

GEOMETRY_STRIDE = 16
GLOBAL_QUERY_K = 96
SHORTLIST_SIZE = 24
MIN_ABS_DIRECTION_DOT = 0.70

GATES = {
    # A coherent close offset: the full lateral remains near the reference.
    "coherent150": {"min": np.inf, "median": 150.0, "p90": 300.0, "anchor": 300.0},
    "coherent300": {"min": np.inf, "median": 300.0, "p90": 600.0, "anchor": 600.0},
    "coherent600": {"min": np.inf, "median": 600.0, "p90": 1200.0, "anchor": 1200.0},
    # Tests the forum's literal nearest-distance claim while rejecting a lone
    # crossing point whose remaining lateral is extremely far away.
    "touch150": {"min": 150.0, "median": 600.0, "p90": 1200.0, "anchor": 1200.0},
}
BLEND_WEIGHTS = (0.10, 0.25, 0.50, 1.00)
MOVE_CAPS_FT = (6.0, 12.0, 24.0, np.inf)

N_FOLDS = 5
RANDOM_STATE = 20260715
OVERALL_GAIN_GATE = 0.20
TOP40_GAIN_GATE = 0.30

if POOL_PATH is None:
    raise FileNotFoundError(f"candidate pool not found under {OOF_ROOT}")
if not (DATA_ROOT / "train").exists():
    raise FileNotFoundError(DATA_ROOT / "train")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rmse(y, prediction):
    y = np.asarray(y, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _rmse_from_sse(sse, rows):
    return float(np.sqrt(max(float(sse), 0.0) / max(int(rows), 1)))


def _cumulative_xy(xy):
    step = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    step[~np.isfinite(step)] = 0.0
    return np.r_[0.0, np.cumsum(step)]


print(
    {
        "experiment": "neighbor_fullcurve_transfer_v0",
        "pool": str(POOL_PATH),
        "geometry_stride": GEOMETRY_STRIDE,
        "global_query_k": GLOBAL_QUERY_K,
        "shortlist_size": SHORTLIST_SIZE,
        "direction_dot_floor": MIN_ABS_DIRECTION_DOT,
        "gates": GATES,
        "weights": BLEND_WEIGHTS,
        "caps": [None if not np.isfinite(value) else value for value in MOVE_CAPS_FT],
    }
)


# -----------------------------------------------------------------------------
# Load OOF predictions
# -----------------------------------------------------------------------------
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
base_cols = [column for column in BASE_CANDIDATES if column in header]
if not {"id", "tvt_true"}.issubset(header):
    raise RuntimeError(f"candidate pool missing id/tvt_true: {header}")
if not base_cols:
    raise RuntimeError(f"no supported bases in candidate pool: {header}")

usecols = [column for column in ("id", "well", "row_idx", "tvt_true", *base_cols) if column in header]
pool = pd.read_csv(
    POOL_PATH,
    usecols=usecols,
    dtype={"id": "string", "well": "string"},
)
pool["id"] = pool["id"].astype(str)
if "well" not in pool.columns:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
pool["well"] = pool["well"].astype(str)
if "row_idx" not in pool.columns:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)

truth = pool["tvt_true"].to_numpy(np.float64)
wells = sorted(pool["well"].unique())
well_to_code = {well: code for code, well in enumerate(wells)}
well_codes = pool["well"].map(well_to_code).to_numpy(np.int32)
pool_indices = {
    well: np.asarray(indices, np.int64)
    for well, indices in pool.groupby("well", sort=False).indices.items()
}
print("pool:", pool.shape, "wells:", len(wells), "bases:", base_cols)


# -----------------------------------------------------------------------------
# Load every training trajectory once and build a global geometry index
# -----------------------------------------------------------------------------
well_data = {}
sample_xy_parts = []
sample_code_parts = []
load_failures = []
started = time.time()

for well in tqdm(wells, desc="load reference trajectories"):
    try:
        path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        frame = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
        visible = frame["TVT_input"].notna().to_numpy()
        if not visible.any() or visible.all():
            raise ValueError("invalid TVT_input mask")
        anchor = int(np.flatnonzero(visible)[-1])
        xy = frame[["X", "Y"]].to_numpy(np.float64)
        if not np.isfinite(xy).all():
            raise ValueError("non-finite XY")
        direction = xy[-1] - xy[anchor]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0:
            raise ValueError("degenerate eval direction")
        direction = direction / direction_norm

        lateral_indices = np.arange(anchor, len(frame), dtype=np.int32)
        sample_indices = lateral_indices[::GEOMETRY_STRIDE]
        if sample_indices[-1] != lateral_indices[-1]:
            sample_indices = np.r_[sample_indices, lateral_indices[-1]]

        well_data[well] = {
            "md": frame["MD"].to_numpy(np.float64),
            "xy": xy,
            "z": frame["Z"].to_numpy(np.float64),
            "tvt": frame["TVT"].to_numpy(np.float64),
            "anchor": anchor,
            "anchor_tvt": float(frame["TVT_input"].iloc[anchor]),
            "direction": direction,
            "lateral_indices": lateral_indices,
            "sample_indices": sample_indices,
            "sample_xy": xy[sample_indices],
        }
        sample_xy_parts.append(xy[sample_indices])
        sample_code_parts.append(
            np.full(len(sample_indices), well_to_code[well], dtype=np.int32)
        )
    except Exception as exc:
        load_failures.append({"well": well, "stage": "load", "reason": repr(exc)})

if load_failures:
    pd.DataFrame(load_failures).to_csv(OUT_DIR / "failures.csv", index=False)
if len(well_data) != len(wells):
    raise RuntimeError(f"loaded {len(well_data)} / {len(wells)} trajectories")

global_sample_xy = np.concatenate(sample_xy_parts, axis=0)
global_sample_codes = np.concatenate(sample_code_parts, axis=0)
global_tree = cKDTree(global_sample_xy, leafsize=64)
print(
    "geometry index:",
    {
        "sample_points": len(global_sample_xy),
        "elapsed_sec": time.time() - started,
    },
)


def _shortlist_neighbors(target_well):
    target = well_data[target_well]
    query_xy = target["sample_xy"]
    target_code = well_to_code[target_well]
    best_contact = {}

    # A long isolated target can occupy all of its first K global neighbors.
    # Expand K until enough *other-well* candidates appear. Each query result is
    # discarded before the next target, so the larger fallback does not create
    # persistent memory pressure.
    query_sizes = []
    k = int(GLOBAL_QUERY_K)
    while True:
        query_sizes.append(min(k, len(global_sample_xy)))
        if query_sizes[-1] == len(global_sample_xy) or query_sizes[-1] >= 3072:
            break
        k *= 2

    for k in query_sizes:
        distances, indices = global_tree.query(query_xy, k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        candidate_codes = global_sample_codes[indices]
        flat_codes = candidate_codes.ravel()
        flat_distances = distances.ravel()
        valid = flat_codes != target_code
        flat_codes = flat_codes[valid]
        flat_distances = flat_distances[valid]

        for code, distance in zip(flat_codes, flat_distances):
            code = int(code)
            previous = best_contact.get(code)
            if previous is None or distance < previous:
                best_contact[code] = float(distance)
        if len(best_contact) >= SHORTLIST_SIZE:
            break

    if not best_contact:
        return []

    # Shortlist by the closest contact. Exact whole-lateral scoring below
    # rejects candidates that only intersect at one point.
    ranked = sorted(best_contact, key=best_contact.get)[:SHORTLIST_SIZE]
    return [wells[code] for code in ranked]


def _select_neighbor(target_well):
    target = well_data[target_well]
    query_xy = target["sample_xy"]
    anchor_xy = target["xy"][target["anchor"]]
    candidates = _shortlist_neighbors(target_well)
    if not candidates:
        raise RuntimeError("empty geometry shortlist")

    scored = []
    for reference_well in candidates:
        reference = well_data[reference_well]
        tree = cKDTree(reference["sample_xy"], leafsize=32)
        distances, _ = tree.query(query_xy, k=1)
        anchor_distance = float(tree.query(anchor_xy, k=1)[0])
        direction_dot = float(np.dot(target["direction"], reference["direction"]))
        scored.append(
            {
                "neighbor": reference_well,
                "min_distance": float(np.min(distances)),
                "median_distance": float(np.median(distances)),
                "p90_distance": float(np.quantile(distances, 0.90)),
                "max_distance": float(np.max(distances)),
                "anchor_distance": anchor_distance,
                "direction_dot": direction_dot,
                "abs_direction_dot": abs(direction_dot),
                "parallel": abs(direction_dot) >= MIN_ABS_DIRECTION_DOT,
                "geometry_score": float(
                    np.median(distances)
                    + 0.25 * np.quantile(distances, 0.90)
                    + 0.10 * anchor_distance
                ),
            }
        )

    report = pd.DataFrame(scored)
    parallel = report[report["parallel"]]
    chosen = (parallel if len(parallel) else report).sort_values("geometry_score").iloc[0]
    return chosen.to_dict(), report


def _transfer_from_neighbor(target_well, reference_well):
    target = well_data[target_well]
    reference = well_data[reference_well]
    output_indices = pool_indices[target_well]
    target_rows = pool["row_idx"].to_numpy(np.int32)[output_indices]
    if np.any(target_rows < 0) or np.any(target_rows >= len(target["xy"])):
        raise IndexError("OOF row index outside horizontal file")

    target_xy = target["xy"][target_rows]
    target_z = target["z"][target_rows]
    anchor_xy = target["xy"][target["anchor"]]
    anchor_z = float(target["z"][target["anchor"]])
    anchor_tvt = float(target["anchor_tvt"])

    ref_indices = reference["lateral_indices"]
    ref_xy = reference["xy"][ref_indices]
    ref_z = reference["z"][ref_indices]
    ref_tvt = reference["tvt"][ref_indices]
    ref_structure = ref_tvt + ref_z
    ref_tree = cKDTree(ref_xy, leafsize=64)

    row_distance, nearest = ref_tree.query(target_xy, k=1)
    anchor_distance, anchor_nearest = ref_tree.query(anchor_xy, k=1)
    nearest = nearest.astype(np.int64)
    anchor_nearest = int(anchor_nearest)

    mapped_tvt = ref_tvt[nearest]
    tvt_xy = mapped_tvt + (anchor_tvt - float(ref_tvt[anchor_nearest]))

    mapped_structure = ref_structure[nearest]
    structure_anchor_prediction = float(ref_structure[anchor_nearest] - anchor_z)
    structure_xy = (
        mapped_structure
        - target_z
        + (anchor_tvt - structure_anchor_prediction)
    )

    # Progress mapping copies a coherent curve rather than allowing the nearest
    # XY point to jump between distant positions on the reference lateral.
    ref_progress = _cumulative_xy(ref_xy)
    target_lateral_xy = target["xy"][target["anchor"] :]
    target_progress_all = _cumulative_xy(target_lateral_xy)
    target_progress = target_progress_all[target_rows - target["anchor"]]
    orientation = 1.0 if np.dot(target["direction"], reference["direction"]) >= 0 else -1.0
    reference_query = float(ref_progress[anchor_nearest]) + orientation * target_progress
    progress_valid = (reference_query >= ref_progress[0]) & (reference_query <= ref_progress[-1])

    tvt_progress = np.full(len(target_rows), np.nan, np.float64)
    structure_progress = np.full(len(target_rows), np.nan, np.float64)
    if progress_valid.any():
        q = reference_query[progress_valid]
        ref_tvt_at_anchor = float(ref_tvt[anchor_nearest])
        ref_structure_at_anchor = float(ref_structure[anchor_nearest])
        tvt_progress[progress_valid] = (
            np.interp(q, ref_progress, ref_tvt)
            + (anchor_tvt - ref_tvt_at_anchor)
        )
        structure_progress[progress_valid] = (
            np.interp(q, ref_progress, ref_structure)
            - target_z[progress_valid]
            + (anchor_tvt - (ref_structure_at_anchor - anchor_z))
        )

    return {
        "indices": output_indices,
        "row_distance": row_distance.astype(np.float32),
        "anchor_distance_full": float(anchor_distance),
        "progress_valid": progress_valid,
        "tvt_xy": tvt_xy.astype(np.float32),
        "structure_xy": structure_xy.astype(np.float32),
        "tvt_progress": tvt_progress.astype(np.float32),
        "structure_progress": structure_progress.astype(np.float32),
    }


# -----------------------------------------------------------------------------
# Select one coherent neighbor and build four target-free transfer curves
# -----------------------------------------------------------------------------
transfer = {
    method: np.full(len(pool), np.nan, np.float32) for method in TRANSFER_METHODS
}
row_neighbor_distance = np.full(len(pool), np.nan, np.float32)
neighbor_rows = []
candidate_parts = []
transfer_failures = []
started = time.time()

for done, target_well in enumerate(tqdm(wells, desc="neighbor full-curve transfer"), 1):
    try:
        chosen, candidates = _select_neighbor(target_well)
        reference_well = str(chosen["neighbor"])
        curves = _transfer_from_neighbor(target_well, reference_well)
        indices = curves["indices"]
        for method in TRANSFER_METHODS:
            transfer[method][indices] = curves[method]
        row_neighbor_distance[indices] = curves["row_distance"]

        chosen.update(
            {
                "well": target_well,
                "neighbor": reference_well,
                "rows": int(len(indices)),
                "anchor_distance_full": curves["anchor_distance_full"],
                "progress_coverage": float(curves["progress_valid"].mean()),
            }
        )
        neighbor_rows.append(chosen)
        candidates["well"] = target_well
        candidates["selected"] = candidates["neighbor"].eq(reference_well)
        candidate_parts.append(candidates)
    except Exception as exc:
        transfer_failures.append(
            {"well": target_well, "stage": "select_or_transfer", "reason": repr(exc)}
        )

    if done % 100 == 0:
        print(
            f"neighbors {done}/{len(wells)} ok={len(neighbor_rows)} "
            f"elapsed_min={(time.time() - started) / 60:.1f}",
            flush=True,
        )

neighbor_meta = pd.DataFrame(neighbor_rows)
candidate_report = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
failure_report = pd.DataFrame(load_failures + transfer_failures)
neighbor_meta.to_csv(OUT_DIR / "selected_neighbors.csv", index=False)
candidate_report.to_csv(OUT_DIR / "neighbor_shortlist_diagnostics.csv", index=False)
failure_report.to_csv(OUT_DIR / "failures.csv", index=False)

missing_transfer_wells = len(wells) - len(neighbor_meta)
if missing_transfer_wells > max(5, int(0.05 * len(wells))):
    raise RuntimeError(f"built only {len(neighbor_meta)} / {len(wells)} neighbor transfers")
if missing_transfer_wells:
    print(
        f"WARNING: {missing_transfer_wells} wells have no transfer; "
        "their rows fall back to each scanned base",
        flush=True,
    )

print(
    "neighbor selection:",
    {
        "wells": len(neighbor_meta),
        "parallel_fraction": float(neighbor_meta["parallel"].mean()),
        "opposite_direction_fraction": float((neighbor_meta["direction_dot"] < 0).mean()),
        "median_neighbor_distance": float(neighbor_meta["median_distance"].median()),
        "p90_neighbor_distance": float(neighbor_meta["median_distance"].quantile(0.90)),
        "coherent150_wells": int(
            (
                (neighbor_meta["median_distance"] <= 150)
                & (neighbor_meta["p90_distance"] <= 300)
            ).sum()
        ),
        "elapsed_min": (time.time() - started) / 60,
    },
)

# Compact sidecar for later feature experiments. Its row order is exactly the
# sorted candidate-pool order used in this run.
np.savez_compressed(
    OUT_DIR / "neighbor_transfer_oof.npz",
    structure_xy=transfer["structure_xy"],
    tvt_xy=transfer["tvt_xy"],
    structure_progress=transfer["structure_progress"],
    tvt_progress=transfer["tvt_progress"],
    row_neighbor_distance=row_neighbor_distance,
)


# -----------------------------------------------------------------------------
# Stable fold labels and row-level distance gates
# -----------------------------------------------------------------------------
fold_source_candidates = [
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75" / "well_metadata.csv",
]
fold_map = None
fold_source = None
for path in fold_source_candidates:
    if not path.exists():
        continue
    candidate = pd.read_csv(path)
    if {"well", "fold"}.issubset(candidate.columns):
        candidate["well"] = candidate["well"].astype(str)
        current_map = candidate.groupby("well")["fold"].first().astype(int).to_dict()
        if set(wells).issubset(current_map):
            fold_map = current_map
            fold_source = str(path)
            break

if fold_map is None:
    counts = pool.groupby("well", as_index=False).size().rename(columns={"size": "rows"})
    counts["stratum"] = pd.qcut(
        counts["rows"].rank(method="first"), q=10, labels=False
    ).astype(int)
    counts["fold"] = -1
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (_, valid) in enumerate(splitter.split(counts, counts["stratum"]), 1):
        counts.loc[valid, "fold"] = fold
    fold_map = dict(zip(counts["well"].astype(str), counts["fold"].astype(int)))
    fold_source = "deterministic row-count StratifiedKFold"

fold_values = pool["well"].map(fold_map).to_numpy(np.int8)
meta_index = neighbor_meta.set_index("well")
row_min_distance = pool["well"].map(meta_index["min_distance"]).to_numpy(np.float64)
row_median_distance = pool["well"].map(meta_index["median_distance"]).to_numpy(np.float64)
row_p90_distance = pool["well"].map(meta_index["p90_distance"]).to_numpy(np.float64)
row_anchor_distance = pool["well"].map(meta_index["anchor_distance_full"]).to_numpy(np.float64)
row_abs_direction_dot = pool["well"].map(meta_index["abs_direction_dot"]).to_numpy(np.float64)
print("fold source:", fold_source)

gate_masks = {}
for gate_name, limits in GATES.items():
    gate_masks[gate_name] = (
        (row_min_distance <= limits["min"])
        & (row_median_distance <= limits["median"])
        & (row_p90_distance <= limits["p90"])
        & (row_anchor_distance <= limits["anchor"])
        & (row_abs_direction_dot >= MIN_ABS_DIRECTION_DOT)
    )


# -----------------------------------------------------------------------------
# Standalone transfer diagnostics by neighbor-distance bin
# -----------------------------------------------------------------------------
distance_bin = pd.cut(
    row_median_distance,
    bins=[-np.inf, 150.0, 300.0, 600.0, np.inf],
    labels=["lt150", "150to300", "300to600", "gt600"],
    right=True,
).astype(str)

distance_rows = []
diagnostic_base = "cp12" if "cp12" in base_cols else base_cols[0]
diagnostic_prediction = pool[diagnostic_base].to_numpy(np.float64)
for bin_name in ("lt150", "150to300", "300to600", "gt600"):
    bin_mask = distance_bin == bin_name
    for method in TRANSFER_METHODS:
        valid = bin_mask & np.isfinite(transfer[method])
        if not valid.any():
            continue
        distance_rows.append(
            {
                "distance_measure": "median_whole_lateral",
                "distance_bin": bin_name,
                "method": method,
                "rows": int(valid.sum()),
                "wells": int(np.unique(well_codes[valid]).size),
                "base_col": diagnostic_base,
                "base_rmse": _rmse(truth[valid], diagnostic_prediction[valid]),
                "transfer_rmse": _rmse(truth[valid], transfer[method][valid]),
                "gain": _rmse(truth[valid], diagnostic_prediction[valid])
                - _rmse(truth[valid], transfer[method][valid]),
                "mean_row_neighbor_distance": float(np.mean(row_neighbor_distance[valid])),
            }
        )

distance_report = pd.DataFrame(distance_rows)
distance_report.to_csv(OUT_DIR / "distance_bin_standalone.csv", index=False)
print("\nSTANDALONE BY MEDIAN NEIGHBOR DISTANCE")
display(distance_report)


# -----------------------------------------------------------------------------
# Small fixed scan: blend/cap neighbor transfer into established OOF bases
# -----------------------------------------------------------------------------
scan_rows = []
fold_rows = []

for base_col in base_cols:
    base = pool[base_col].to_numpy(np.float64)
    error = base - truth
    squared_error = error * error
    base_sse = float(squared_error.sum())
    base_rmse = _rmse_from_sse(base_sse, len(pool))

    well_sse = np.bincount(well_codes, weights=squared_error, minlength=len(wells))
    top40_codes = np.argsort(well_sse)[-40:]
    top40_mask = np.isin(well_codes, top40_codes)
    top40_rows = int(top40_mask.sum())
    top40_base_sse = float(squared_error[top40_mask].sum())
    top40_base_rmse = _rmse_from_sse(top40_base_sse, top40_rows)

    fold_base = {}
    for fold in range(1, N_FOLDS + 1):
        mask = fold_values == fold
        fold_base[fold] = {
            "rows": int(mask.sum()),
            "sse": float(squared_error[mask].sum()),
        }

    for method in TRANSFER_METHODS:
        candidate = transfer[method].astype(np.float64)
        finite_candidate = np.isfinite(candidate)
        raw_delta = candidate - base

        for gate_name, gate_mask in gate_masks.items():
            eligible = gate_mask & finite_candidate
            eligible_indices = np.flatnonzero(eligible)
            if len(eligible_indices) == 0:
                continue
            local_delta = raw_delta[eligible_indices]

            for weight in BLEND_WEIGHTS:
                for cap in MOVE_CAPS_FT:
                    if np.isfinite(cap):
                        move = float(weight) * np.clip(local_delta, -float(cap), float(cap))
                        cap_label = float(cap)
                    else:
                        move = float(weight) * local_delta
                        cap_label = "full"

                    changed = np.abs(move) > 1e-9
                    changed_indices = eligible_indices[changed]
                    changed_move = move[changed]
                    old_error = error[changed_indices]
                    delta_sse = 2.0 * old_error * changed_move + changed_move * changed_move

                    candidate_sse = base_sse + float(delta_sse.sum())
                    score = _rmse_from_sse(candidate_sse, len(pool))
                    fold_gains = []
                    for fold in range(1, N_FOLDS + 1):
                        local_fold = fold_values[changed_indices] == fold
                        fold_delta = float(delta_sse[local_fold].sum())
                        fold_score = _rmse_from_sse(
                            fold_base[fold]["sse"] + fold_delta,
                            fold_base[fold]["rows"],
                        )
                        fold_base_rmse = _rmse_from_sse(
                            fold_base[fold]["sse"], fold_base[fold]["rows"]
                        )
                        gain = fold_base_rmse - fold_score
                        fold_gains.append(gain)
                        fold_rows.append(
                            {
                                "base_col": base_col,
                                "method": method,
                                "gate": gate_name,
                                "weight": weight,
                                "move_cap_ft": cap_label,
                                "fold": fold,
                                "rows": fold_base[fold]["rows"],
                                "base_rmse": fold_base_rmse,
                                "rmse": fold_score,
                                "gain": gain,
                            }
                        )

                    top_changed = top40_mask[changed_indices]
                    top40_sse = top40_base_sse + float(delta_sse[top_changed].sum())
                    top40_rmse = _rmse_from_sse(top40_sse, top40_rows)
                    abs_move = np.abs(changed_move)
                    scan_rows.append(
                        {
                            "base_col": base_col,
                            "method": method,
                            "gate": gate_name,
                            "weight": weight,
                            "move_cap_ft": cap_label,
                            "base_rmse": base_rmse,
                            "rmse": score,
                            "gain": base_rmse - score,
                            "mean_fold_gain": float(np.mean(fold_gains)),
                            "min_fold_gain": float(np.min(fold_gains)),
                            "positive_folds": int(np.sum(np.asarray(fold_gains) > 0)),
                            "top40_base_rmse": top40_base_rmse,
                            "top40_rmse": top40_rmse,
                            "top40_gain": top40_base_rmse - top40_rmse,
                            "eligible_wells": int(np.unique(well_codes[eligible]).size),
                            "eligible_rows": int(eligible.sum()),
                            "changed_wells": int(np.unique(well_codes[changed_indices]).size),
                            "changed_rows": int(len(changed_indices)),
                            "mean_abs_move_changed": float(abs_move.mean()) if len(abs_move) else 0.0,
                            "p95_abs_move_changed": float(np.quantile(abs_move, 0.95)) if len(abs_move) else 0.0,
                            "max_abs_move": float(abs_move.max()) if len(abs_move) else 0.0,
                        }
                    )

scan = pd.DataFrame(scan_rows)
fold_report = pd.DataFrame(fold_rows)
if len(scan) == 0:
    raise RuntimeError("empty neighbor-transfer scan")

scan["strict_gate"] = (
    (scan["gain"] >= OVERALL_GAIN_GATE)
    & (scan["positive_folds"] == N_FOLDS)
    & (scan["min_fold_gain"] > 0.0)
    & (scan["top40_gain"] >= TOP40_GAIN_GATE)
)
scan["robust_score"] = scan["gain"] + np.minimum(scan["min_fold_gain"], 0.0)
scan = scan.sort_values(["strict_gate", "robust_score", "gain"], ascending=False).reset_index(drop=True)
fold_report = fold_report.merge(
    scan[["base_col", "method", "gate", "weight", "move_cap_ft", "strict_gate"]],
    on=["base_col", "method", "gate", "weight", "move_cap_ft"],
    how="left",
)

scan.to_csv(OUT_DIR / "neighbor_transfer_scan.csv", index=False)
fold_report.to_csv(OUT_DIR / "neighbor_transfer_folds.csv", index=False)
best_by_base = (
    scan.sort_values(["strict_gate", "robust_score", "gain"], ascending=False)
    .groupby("base_col", as_index=False)
    .first()
    .sort_values("gain", ascending=False)
)
best_by_base.to_csv(OUT_DIR / "best_by_base.csv", index=False)

print("\nBEST BY BASE")
display(best_by_base)
print("\nTOP 30 CONFIGS")
display(scan.head(30))

passing = scan[scan["strict_gate"]]
close_rows = distance_report[distance_report["distance_bin"].eq("lt150")]
best_close_rmse = float(close_rows["transfer_rmse"].min()) if len(close_rows) else np.nan
verdict = {
    "pool": str(POOL_PATH),
    "rows": int(len(pool)),
    "wells": int(len(wells)),
    "transferred_wells": int(len(neighbor_meta)),
    "selected_parallel_fraction": float(neighbor_meta["parallel"].mean()),
    "selected_opposite_direction_fraction": float((neighbor_meta["direction_dot"] < 0).mean()),
    "median_neighbor_distance": float(neighbor_meta["median_distance"].median()),
    "coherent150_wells": int(
        (
            (neighbor_meta["median_distance"] <= 150)
            & (neighbor_meta["p90_distance"] <= 300)
            & (neighbor_meta["abs_direction_dot"] >= MIN_ABS_DIRECTION_DOT)
        ).sum()
    ),
    "best_lt150_standalone_rmse": best_close_rmse,
    "passing_configs": int(len(passing)),
    "best_gain": float(scan.iloc[0]["gain"]),
    "best_top40_gain": float(scan.iloc[0]["top40_gain"]),
    "strict_pass": bool(len(passing) > 0),
    "next": (
        "PASS -> add typewell-frame agreement and build deployment bundle"
        if len(passing) > 0
        else "FAIL -> stop full-curve neighbor-transfer route"
    ),
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, ensure_ascii=True)

print("\nVERDICT", verdict)
print("saved:", OUT_DIR)
