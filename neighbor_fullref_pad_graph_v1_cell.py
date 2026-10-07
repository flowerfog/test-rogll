"""Full-reference pad-graph consensus gateway.

Uses the v0 geometry shortlist, but maps each shortlisted reference through its
complete labeled trajectory. This fixes the coverage failure seen in the first
multi-neighbor experiment. No supervised selector is trained.

Frozen candidates:
* single_full: selected v0 neighbor with complete-path progress mapping;
* median3/5_geometry: robust consensus ordered by mapped XY fit;
* median3_anchorclean: geometry plus visible-anchor structure mismatch;
* trimmed3_anchorclean: clipped weighted mean of the anchor-clean top three;
* consensus3_anchorclean: use the median only where top-three spread <= 12 ft;
* plane5_anchorclean: ridge local plane from up to five mapped references.

Truth is used only for final scoring and an explicitly diagnostic oracle.
"""

from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE_DIR = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
OUT_DIR = OOF_ROOT / "neighbor_fullref_pad_graph_v1"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)

N_FOLDS = 5
RANDOM_STATE = 20260716
MAX_REFERENCES = 8
MIN_ABS_DIRECTION_DOT = 0.70
MIN_PROGRESS_COVERAGE = 0.80
SAFE_WEIGHT = 0.25
RAW_MOVE_CAP = 24.0
ANCHOR_MISMATCH_WEIGHT = 1.0
CONSENSUS_SPREAD_CAP = 12.0
TRIM_CAP = 12.0
PLANE_RIDGE = 5.0
PLANE_MEDIAN_CAP = 12.0

CANDIDATE_NAMES = (
    "single_full",
    "median3_geometry",
    "median5_geometry",
    "median3_anchorclean",
    "trimmed3_anchorclean",
    "consensus3_anchorclean",
    "plane5_anchorclean",
)
PRIMARY_CANDIDATE = "consensus3_anchorclean"

if POOL_PATH is None:
    raise FileNotFoundError("candidate pool not found")
for path in (
    SOURCE_DIR / "neighbor_transfer_oof.npz",
    SOURCE_DIR / "selected_neighbors.csv",
    SOURCE_DIR / "neighbor_shortlist_diagnostics.csv",
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
    progress = np.asarray(progress, np.float64)
    keep = np.r_[np.diff(progress) > 1e-8, True]
    output = [progress[keep]]
    output.extend(np.asarray(array)[keep] for array in arrays)
    return output


def _weighted_median(values, weights, fallback):
    values = np.asarray(values, np.float64).T
    weights = np.asarray(weights, np.float64)
    weights = np.broadcast_to(weights[None, :], values.shape).copy()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    work_values = np.where(valid, values, np.inf)
    work_weights = np.where(valid, weights, 0.0)
    order = np.argsort(work_values, axis=1)
    sorted_values = np.take_along_axis(work_values, order, axis=1)
    sorted_weights = np.take_along_axis(work_weights, order, axis=1)
    cumulative = np.cumsum(sorted_weights, axis=1)
    total = cumulative[:, -1]
    position = np.argmax(cumulative >= (0.5 * total)[:, None], axis=1)
    output = sorted_values[np.arange(len(values)), position]
    bad = (total <= 0) | ~np.isfinite(output)
    output[bad] = np.asarray(fallback, np.float64)[bad]
    return output


def _local_plane(values, mapped_xy, target_xy, weights, fallback):
    values = np.asarray(values, np.float64).T
    mapped_xy = np.asarray(mapped_xy, np.float64).transpose(1, 0, 2)
    target_xy = np.asarray(target_xy, np.float64)
    weights = np.asarray(weights, np.float64)
    weights = np.broadcast_to(weights[None, :], values.shape).copy()
    valid = np.isfinite(values) & np.isfinite(mapped_xy).all(axis=2)
    weights[~valid] = 0.0
    y = np.where(valid, values, 0.0)
    offset = (mapped_xy - target_xy[:, None, :]) / 1000.0
    offset[~np.isfinite(offset)] = 0.0
    x = np.concatenate(
        [np.ones((len(values), values.shape[1], 1)), offset], axis=2
    )
    a = np.einsum("nki,nk,nkj->nij", x, weights, x)
    b = np.einsum("nki,nk,nk->ni", x, weights, y)
    a[:, 0, 0] += 1e-6
    a[:, 1, 1] += PLANE_RIDGE
    a[:, 2, 2] += PLANE_RIDGE
    enough = (weights > 0).sum(axis=1) >= 3
    output = np.asarray(fallback, np.float64).copy()
    if enough.any():
        try:
            coefficients = np.linalg.solve(a[enough], b[enough])
            estimate = coefficients[:, 0]
            center = np.asarray(fallback, np.float64)[enough]
            output[enough] = np.clip(
                estimate,
                center - PLANE_MEDIAN_CAP,
                center + PLANE_MEDIAN_CAP,
            )
        except np.linalg.LinAlgError:
            pass
    return output


print(
    {
        "experiment": "neighbor_fullref_pad_graph_v1",
        "source": str(SOURCE_DIR),
        "pool": str(POOL_PATH),
        "max_references": MAX_REFERENCES,
        "min_coverage": MIN_PROGRESS_COVERAGE,
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
# Pool, controls, and graph edges.
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

v0_sidecar = np.load(SOURCE_DIR / "neighbor_transfer_oof.npz", allow_pickle=False)
v0_progress = v0_sidecar["structure_progress"].astype(np.float64)
if len(v0_progress) != len(pool):
    raise RuntimeError("v0 sidecar row mismatch")

selected = pd.read_csv(SOURCE_DIR / "selected_neighbors.csv")
selected["well"] = selected["well"].astype(str)
selected["neighbor"] = selected["neighbor"].astype(str)
selected_meta = selected.set_index("well")

shortlist = pd.read_csv(SOURCE_DIR / "neighbor_shortlist_diagnostics.csv")
shortlist["well"] = shortlist["well"].astype(str)
shortlist["neighbor"] = shortlist["neighbor"].astype(str)
shortlist = shortlist[
    shortlist["well"].isin(wells)
    & shortlist["neighbor"].isin(wells)
    & (shortlist["abs_direction_dot"].fillna(0.0) >= MIN_ABS_DIRECTION_DOT)
].copy()
shortlist_groups = {
    str(well): frame.sort_values(["geometry_score", "neighbor"])
    for well, frame in shortlist.groupby("well", sort=False)
}

eligible_well = (
    (selected_meta["median_distance"] <= 600.0)
    & (selected_meta["p90_distance"] <= 1200.0)
    & (selected_meta["anchor_distance_full"] <= 1200.0)
    & (selected_meta["abs_direction_dot"] >= MIN_ABS_DIRECTION_DOT)
)
eligible_map = eligible_well.to_dict()


# ---------------------------------------------------------------------------
# Full reference bank.
# ---------------------------------------------------------------------------
well_data = {}
load_failures = []
for well in tqdm(wells, desc="load pad graph curves"):
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


def _full_progress_transfer(target_well, reference_well, target_rows):
    target = well_data[target_well]
    reference = well_data[reference_well]
    target_rows = np.asarray(target_rows, np.int32)
    target_xy = target["xy"][target_rows]
    target_z = target["z"][target_rows]
    target_progress_all = _cumulative_xy(target["xy"][target["anchor"] :])
    target_progress = target_progress_all[target_rows - target["anchor"]]
    target_anchor_xy = target["xy"][target["anchor"]]
    _, ref_anchor_index = reference["tree"].query(target_anchor_xy, k=1)
    ref_anchor_index = int(ref_anchor_index)
    sign = (
        1.0
        if float(np.dot(target["direction"], reference["direction"])) >= 0
        else -1.0
    )
    query = float(reference["progress"][ref_anchor_index]) + sign * target_progress
    valid = (query >= reference["unique_progress"][0]) & (
        query <= reference["unique_progress"][-1]
    )
    candidate = np.full(len(target_rows), np.nan, np.float64)
    mapped_xy = np.full((len(target_rows), 2), np.nan, np.float64)
    if valid.any():
        q = query[valid]
        mapped_structure = np.interp(
            q, reference["unique_progress"], reference["unique_structure"]
        )
        mapped_xy[valid, 0] = np.interp(
            q, reference["unique_progress"], reference["unique_x"]
        )
        mapped_xy[valid, 1] = np.interp(
            q, reference["unique_progress"], reference["unique_y"]
        )
        target_anchor_z = float(target["z"][target["anchor"]])
        ref_anchor_structure = float(reference["structure"][ref_anchor_index])
        ref_anchor_tvt_on_target = ref_anchor_structure - target_anchor_z
        candidate[valid] = (
            mapped_structure
            - target_z[valid]
            + (target["anchor_tvt"] - ref_anchor_tvt_on_target)
        )

    distance = np.linalg.norm(mapped_xy - target_xy, axis=1)
    finite = valid & np.isfinite(distance)
    if finite.any():
        geometry_score = float(
            np.median(distance[finite])
            + 0.25 * np.quantile(distance[finite], 0.90)
            + 1000.0 * (1.0 - finite.mean())
        )
    else:
        geometry_score = np.inf
    target_anchor_structure = target["anchor_tvt"] + target["z"][target["anchor"]]
    anchor_mismatch = float(
        reference["structure"][ref_anchor_index] - target_anchor_structure
    )
    return {
        "candidate": candidate,
        "mapped_xy": mapped_xy,
        "coverage": float(finite.mean()),
        "geometry_score": geometry_score,
        "anchor_mismatch": anchor_mismatch,
    }


# ---------------------------------------------------------------------------
# Freeze graph candidates and diagnostic oracle.
# ---------------------------------------------------------------------------
candidates = {
    name: v0_progress.astype(np.float32).copy() for name in CANDIDATE_NAMES
}
graph_rows = []
edge_rows = []
failures = []
started = time.time()

for done, target_well in enumerate(tqdm(wells, desc="full-reference pad graph"), 1):
    try:
        if target_well not in well_data or target_well not in shortlist_groups:
            continue
        output_indices = pool_indices[target_well]
        target_rows = pool_row_idx[output_indices]
        target_truth = truth[output_indices]
        target_xy = well_data[target_well]["xy"][target_rows]
        fallback = v0_progress[output_indices]
        selected_neighbor = str(selected_meta.loc[target_well, "neighbor"])
        ordered = [selected_neighbor] + shortlist_groups[target_well][
            "neighbor"
        ].astype(str).tolist()
        reference_wells = []
        for name in ordered:
            if (
                name != target_well
                and name in well_data
                and name not in reference_wells
            ):
                reference_wells.append(name)
            if len(reference_wells) >= MAX_REFERENCES:
                break

        records = []
        for reference_well in reference_wells:
            record = _full_progress_transfer(
                target_well, reference_well, target_rows
            )
            if record["coverage"] < MIN_PROGRESS_COVERAGE:
                continue
            record["neighbor"] = reference_well
            record["is_selected_v0"] = int(reference_well == selected_neighbor)
            record["anchorclean_score"] = float(
                record["geometry_score"]
                + ANCHOR_MISMATCH_WEIGHT * abs(record["anchor_mismatch"])
            )
            records.append(record)

        if not records:
            continue
        selected_records = [row for row in records if row["is_selected_v0"]]
        single_record = selected_records[0] if selected_records else records[0]
        single = np.asarray(single_record["candidate"], np.float64)
        single = np.where(np.isfinite(single), single, fallback)

        # All deployable graph candidates share the same frozen fallback. This
        # also makes the diagnostic per-reference oracle comparable on exactly
        # the same target rows.
        for row in records:
            values = np.asarray(row["candidate"], np.float64)
            row["candidate"] = np.where(np.isfinite(values), values, single)

        geometry_order = sorted(records, key=lambda row: row["geometry_score"])
        anchor_order = sorted(records, key=lambda row: row["anchorclean_score"])

        def stack_records(rows):
            values = np.asarray([row["candidate"] for row in rows], np.float64)
            mapped = np.asarray([row["mapped_xy"] for row in rows], np.float64)
            return values, mapped

        def median_from(rows, score_key):
            values, _ = stack_records(rows)
            weights = 1.0 / np.maximum(
                np.asarray([row[score_key] for row in rows], np.float64) + 50.0,
                50.0,
            ) ** 2
            return _weighted_median(values, weights, single), weights

        geometry3 = geometry_order[: min(3, len(geometry_order))]
        geometry5 = geometry_order[: min(5, len(geometry_order))]
        anchor3 = anchor_order[: min(3, len(anchor_order))]
        anchor5 = anchor_order[: min(5, len(anchor_order))]
        median3_geometry, _ = median_from(geometry3, "geometry_score")
        median5_geometry, _ = median_from(geometry5, "geometry_score")
        median3_anchor, anchor3_weights = median_from(
            anchor3, "anchorclean_score"
        )

        anchor3_values, _ = stack_records(anchor3)
        clipped = np.clip(
            anchor3_values,
            median3_anchor[None, :] - TRIM_CAP,
            median3_anchor[None, :] + TRIM_CAP,
        )
        valid = np.isfinite(clipped)
        weighted = np.where(valid, clipped, 0.0) * anchor3_weights[:, None]
        denominator = np.sum(valid * anchor3_weights[:, None], axis=0)
        trimmed = np.divide(
            np.sum(weighted, axis=0),
            denominator,
            out=single.copy(),
            where=denominator > 0,
        )
        spread = np.nanmax(anchor3_values, axis=0) - np.nanmin(
            anchor3_values, axis=0
        )
        enough = np.isfinite(anchor3_values).sum(axis=0) >= 2
        consensus = np.where(
            enough & (spread <= CONSENSUS_SPREAD_CAP),
            median3_anchor,
            single,
        )

        anchor5_values, anchor5_mapped = stack_records(anchor5)
        anchor5_weights = 1.0 / np.maximum(
            np.asarray(
                [row["anchorclean_score"] for row in anchor5], np.float64
            )
            + 50.0,
            50.0,
        ) ** 2
        plane = _local_plane(
            anchor5_values,
            anchor5_mapped,
            target_xy,
            anchor5_weights,
            median3_anchor,
        )

        generated = {
            "single_full": single,
            "median3_geometry": median3_geometry,
            "median5_geometry": median5_geometry,
            "median3_anchorclean": median3_anchor,
            "trimmed3_anchorclean": trimmed,
            "consensus3_anchorclean": consensus,
            "plane5_anchorclean": plane,
        }
        for name, values in generated.items():
            values = np.asarray(values, np.float64)
            bad = ~np.isfinite(values)
            values[bad] = single[bad]
            candidates[name][output_indices] = values.astype(np.float32)

        reference_rmse = np.asarray(
            [_rmse(target_truth, row["candidate"]) for row in records],
            np.float64,
        )
        oracle_position = int(np.argmin(reference_rmse))
        predicted_position = int(
            np.argmin([row["anchorclean_score"] for row in records])
        )
        single_rmse = _rmse(target_truth, single)
        graph_rows.append(
            {
                "well": target_well,
                "references": len(records),
                "single_neighbor": single_record["neighbor"],
                "anchorclean_neighbor": records[predicted_position]["neighbor"],
                "oracle_neighbor": records[oracle_position]["neighbor"],
                "anchorclean_hit_oracle": int(predicted_position == oracle_position),
                "single_rmse": single_rmse,
                "oracle_rmse": float(reference_rmse[oracle_position]),
                "oracle_gain_vs_single": single_rmse
                - float(reference_rmse[oracle_position]),
                "median3_anchor_rmse": _rmse(target_truth, median3_anchor),
                "consensus_rmse": _rmse(target_truth, consensus),
                "spread_mean": float(np.nanmean(spread)),
                "spread_p90": float(np.nanquantile(spread, 0.90)),
            }
        )
        for position, row in enumerate(records):
            edge_rows.append(
                {
                    "well": target_well,
                    "neighbor": row["neighbor"],
                    "position": position,
                    "geometry_score": row["geometry_score"],
                    "anchor_mismatch": row["anchor_mismatch"],
                    "anchorclean_score": row["anchorclean_score"],
                    "coverage": row["coverage"],
                    "true_rmse": float(reference_rmse[position]),
                    "is_oracle": int(position == oracle_position),
                }
            )
    except Exception as exc:
        failures.append({"well": target_well, "reason": repr(exc)})

    if done % 100 == 0:
        print(
            f"graph {done}/{len(wells)} ok={len(graph_rows)} "
            f"elapsed_min={(time.time()-started)/60:.1f}",
            flush=True,
        )

graph_report = pd.DataFrame(graph_rows)
edge_report = pd.DataFrame(edge_rows)
failure_report = pd.DataFrame(failures)
graph_report.to_csv(OUT_DIR / "graph_bywell.csv", index=False)
edge_report.to_csv(OUT_DIR / "graph_edges.csv", index=False)
failure_report.to_csv(OUT_DIR / "failures.csv", index=False)

print(
    "graph coverage:",
    graph_report["references"].describe().to_dict(),
)
print(
    "graph oracle:",
    {
        "mean_oracle_gain_vs_single": float(
            graph_report["oracle_gain_vs_single"].mean()
        ),
        "positive_oracle_fraction": float(
            (graph_report["oracle_gain_vs_single"] > 0).mean()
        ),
        "anchorclean_top1_accuracy": float(
            graph_report["anchorclean_hit_oracle"].mean()
        ),
        "anchor_mismatch_true_rmse_spearman": float(
            edge_report[["anchor_mismatch", "true_rmse"]]
            .assign(anchor_abs=lambda frame: frame["anchor_mismatch"].abs())
            [["anchor_abs", "true_rmse"]]
            .corr(method="spearman")
            .iloc[0, 1]
        ),
    },
)


# ---------------------------------------------------------------------------
# Stable folds and frozen coherent600 rows.
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
# Score fixed graph candidates on historical downstream bases.
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


# Nested graph-rule selection is diagnostic only.
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
control = _score_row(primary_base, "single_full")
primary = _score_row(primary_base, PRIMARY_CANDIDATE)
control_folds = fold_report[
    (fold_report["base_col"] == primary_base)
    & (fold_report["candidate"] == "single_full")
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
    "incremental_gain_vs_single_full": float(
        primary["gain"] - control["gain"]
    ),
    "control_top40_gain": float(control["top40_gain"]),
    "primary_top40_gain": float(primary["top40_gain"]),
    "primary_min_fold_gain": float(primary["min_fold_gain"]),
    "primary_positive_folds": int(primary["positive_folds"]),
    "folds_better_than_single_full": folds_better,
    "mean_graph_oracle_gain_vs_single": float(
        graph_report["oracle_gain_vs_single"].mean()
    ),
    "graph_oracle_positive_fraction": float(
        (graph_report["oracle_gain_vs_single"] > 0).mean()
    ),
    "anchorclean_top1_accuracy": float(
        graph_report["anchorclean_hit_oracle"].mean()
    ),
    "strict_pass": strict_pass,
    "next": (
        "PASS -> promote pad consensus and test final-chain insertion"
        if strict_pass
        else "STOP fixed pad consensus; use oracle/rank diagnostics to decide graph model"
    ),
}

scores.to_csv(OUT_DIR / "scores.csv", index=False)
fold_report.to_csv(OUT_DIR / "folds.csv", index=False)
nested.to_csv(OUT_DIR / "nested_variant_selection.csv", index=False)
with open(OUT_DIR / "result.json", "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, ensure_ascii=True)
if strict_pass:
    np.savez_compressed(
        OUT_DIR / "primary_candidate_oof.npz",
        candidate=candidates[PRIMARY_CANDIDATE].astype(np.float32),
    )

print("\nGRAPH DIAGNOSTICS")
display(graph_report.describe(include="all"))
print("\nSCORES")
display(scores.sort_values(["base_col", "gain"], ascending=[True, False]))
print("\nFOLDS")
display(fold_report)
print("\nNESTED VARIANT SELECTION")
display(nested)
print("\nRESULT", result)
print("saved:", OUT_DIR)

gc.collect()
