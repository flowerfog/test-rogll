"""Multi-neighbor robust transfer and local-plane gateway.

Prerequisite: neighbor_fullcurve_transfer_v0_cell.py has completed.

This cell keeps the frozen v0 application rule unchanged:
    base + 0.25 * clip(neighbor_candidate - base, -24, 24)

Only candidate generation changes. The selected single-neighbor progress curve
is compared with fixed K=3/K=5 robust aggregates and a ridge local plane. All
references are selected from deployable trajectory geometry; target truth is
used only after every candidate curve has been frozen.
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
OUT_DIR = OOF_ROOT / "neighbor_multik_localplane_v1"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)

N_FOLDS = 5
RANDOM_STATE = 20260715
MIN_ABS_DIRECTION_DOT = 0.70
MAX_REFERENCES = 8
SAFE_WEIGHT = 0.25
RAW_MOVE_CAP = 24.0
PLANE_RIDGE = 5.0  # slopes are expressed per 1,000 coordinate units
PLANE_MEDIAN_CAP = 12.0

CANDIDATE_NAMES = (
    "single_v0",
    "median3_iso",
    "median5_iso",
    "median3_aniso",
    "median5_aniso",
    "plane5_aniso",
)
PRIMARY_CANDIDATE = "median5_aniso"

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


def _weighted_median(values, weights, fallback):
    """Row-wise weighted median for values shaped (K, rows)."""
    values = np.asarray(values, np.float64).T
    weights = np.asarray(weights, np.float64)
    if weights.ndim == 1:
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
    """Batched weighted ridge plane; returns the intercept at target XY."""
    values = np.asarray(values, np.float64).T  # rows, K
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
            estimate = np.clip(
                estimate,
                center - PLANE_MEDIAN_CAP,
                center + PLANE_MEDIAN_CAP,
            )
            output[enough] = estimate
        except np.linalg.LinAlgError:
            pass
    return output


print(
    {
        "experiment": "neighbor_multik_localplane_v1",
        "source": str(SOURCE_DIR),
        "pool": str(POOL_PATH),
        "max_references": MAX_REFERENCES,
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
# Load the established OOF row order and frozen v0 sidecars.
# ---------------------------------------------------------------------------
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
base_cols = [column for column in ("strong", "strong_bias", "cp12") if column in header]
required = {"id", "tvt_true", "strong"}
if not required.issubset(header):
    raise RuntimeError(f"candidate pool missing {sorted(required-set(header))}")
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
single_v0 = sidecar["structure_progress"].astype(np.float64)
if len(single_v0) != len(pool):
    raise RuntimeError("v0 sidecar row mismatch")

selected = pd.read_csv(SOURCE_DIR / "selected_neighbors.csv")
selected["well"] = selected["well"].astype(str)
selected["neighbor"] = selected["neighbor"].astype(str)
selected_meta = selected.set_index("well")

shortlist = pd.read_csv(SOURCE_DIR / "neighbor_shortlist_diagnostics.csv")
shortlist["well"] = shortlist["well"].astype(str)
shortlist["neighbor"] = shortlist["neighbor"].astype(str)
shortlist = shortlist[
    shortlist["well"].isin(wells) & shortlist["neighbor"].isin(wells)
].copy()
shortlist = shortlist[
    shortlist["abs_direction_dot"].fillna(0.0) >= MIN_ABS_DIRECTION_DOT
]

eligible_well = (
    (selected_meta["median_distance"] <= 600.0)
    & (selected_meta["p90_distance"] <= 1200.0)
    & (selected_meta["anchor_distance_full"] <= 1200.0)
    & (selected_meta["abs_direction_dot"] >= MIN_ABS_DIRECTION_DOT)
)
eligible_map = eligible_well.to_dict()


# ---------------------------------------------------------------------------
# Load complete train trajectories. OOF target is excluded only as a reference.
# ---------------------------------------------------------------------------
well_data = {}
load_failures = []
for well in tqdm(wells, desc="load full curves"):
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
        lateral = np.arange(anchor, len(frame), dtype=np.int32)
        lateral_xy = xy[lateral]
        direction = xy[-1] - xy[anchor]
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0:
            raise ValueError("degenerate direction")
        direction /= norm
        well_data[well] = {
            "xy": xy,
            "z": z,
            "tvt": tvt,
            "anchor": anchor,
            "anchor_tvt": float(frame["TVT_input"].iloc[anchor]),
            "direction": direction,
            "lateral": lateral,
            "lateral_xy": lateral_xy,
            "progress": _cumulative_xy(lateral_xy),
            "tree": cKDTree(lateral_xy, leafsize=64),
        }
    except Exception as exc:
        load_failures.append({"well": well, "reason": repr(exc)})

pd.DataFrame(load_failures).to_csv(OUT_DIR / "load_failures.csv", index=False)
if len(well_data) < len(wells) - 5:
    raise RuntimeError(f"loaded only {len(well_data)} / {len(wells)} wells")


def _progress_transfer(target_well, reference_well, target_rows):
    target = well_data[target_well]
    reference = well_data[reference_well]
    target_rows = np.asarray(target_rows, np.int32)
    if np.any(target_rows < target["anchor"]):
        raise ValueError("pool row lies before anchor")

    target_xy = target["xy"][target_rows]
    target_z = target["z"][target_rows]
    target_progress_all = _cumulative_xy(target["xy"][target["anchor"] :])
    target_progress = target_progress_all[target_rows - target["anchor"]]
    anchor_xy = target["xy"][target["anchor"]]
    _, anchor_position = reference["tree"].query(anchor_xy, k=1)
    anchor_position = int(anchor_position)
    orientation = (
        1.0
        if float(np.dot(target["direction"], reference["direction"])) >= 0
        else -1.0
    )
    query = reference["progress"][anchor_position] + orientation * target_progress
    valid = (query >= reference["progress"][0]) & (
        query <= reference["progress"][-1]
    )

    prediction = np.full(len(target_rows), np.nan, np.float64)
    mapped_xy = np.full((len(target_rows), 2), np.nan, np.float64)
    if valid.any():
        q = query[valid]
        lateral = reference["lateral"]
        ref_structure = reference["tvt"][lateral] + reference["z"][lateral]
        structure_q = np.interp(q, reference["progress"], ref_structure)
        x_q = np.interp(q, reference["progress"], reference["lateral_xy"][:, 0])
        y_q = np.interp(q, reference["progress"], reference["lateral_xy"][:, 1])
        anchor_structure = float(ref_structure[anchor_position])
        anchor_z = float(target["z"][target["anchor"]])
        anchor_prediction = anchor_structure - anchor_z
        prediction[valid] = (
            structure_q
            - target_z[valid]
            + (target["anchor_tvt"] - anchor_prediction)
        )
        mapped_xy[valid, 0] = x_q
        mapped_xy[valid, 1] = y_q

    delta_xy = mapped_xy - target_xy
    normal = np.array([-target["direction"][1], target["direction"][0]])
    along = np.abs(delta_xy @ target["direction"])
    cross = np.abs(delta_xy @ normal)
    finite = valid & np.isfinite(along) & np.isfinite(cross)
    if finite.any():
        iso_score = float(np.median(np.sqrt(along[finite] ** 2 + cross[finite] ** 2)))
        aniso_score = float(np.median(cross[finite]) + 0.25 * np.median(along[finite]))
        coverage = float(finite.mean())
    else:
        iso_score = np.inf
        aniso_score = np.inf
        coverage = 0.0
    return prediction, mapped_xy, iso_score, aniso_score, coverage


# ---------------------------------------------------------------------------
# Build frozen multi-reference curves. Missing rows fall back to single_v0.
# ---------------------------------------------------------------------------
candidates = {
    name: single_v0.astype(np.float32).copy() for name in CANDIDATE_NAMES
}
reference_rows = []
transfer_failures = []
started = time.time()

shortlist_groups = {
    str(well): frame.sort_values(["geometry_score", "neighbor"])
    for well, frame in shortlist.groupby("well", sort=False)
}

for done, target_well in enumerate(tqdm(wells, desc="multi-neighbor transfer"), 1):
    try:
        if target_well not in well_data or target_well not in shortlist_groups:
            continue
        output_indices = pool_indices[target_well]
        target_rows = pool_row_idx[output_indices]
        fallback = single_v0[output_indices]
        target_xy = well_data[target_well]["xy"][target_rows]
        group = shortlist_groups[target_well]
        reference_wells = [
            name
            for name in group["neighbor"].astype(str).tolist()
            if name != target_well and name in well_data
        ][:MAX_REFERENCES]

        transferred = []
        mapped = []
        iso_scores = []
        aniso_scores = []
        coverages = []
        kept_references = []
        for reference_well in reference_wells:
            curve, mapped_xy, iso_score, aniso_score, coverage = _progress_transfer(
                target_well, reference_well, target_rows
            )
            if coverage < 0.50:
                continue
            transferred.append(curve)
            mapped.append(mapped_xy)
            iso_scores.append(iso_score)
            aniso_scores.append(aniso_score)
            coverages.append(coverage)
            kept_references.append(reference_well)

        if not transferred:
            continue
        transferred = np.asarray(transferred, np.float64)
        mapped = np.asarray(mapped, np.float64)
        iso_scores = np.asarray(iso_scores, np.float64)
        aniso_scores = np.asarray(aniso_scores, np.float64)

        iso_order = np.argsort(iso_scores)
        aniso_order = np.argsort(aniso_scores)

        def aggregate(order, k, score_values):
            chosen = order[: min(k, len(order))]
            values = transferred[chosen]
            scores = score_values[chosen]
            weights = 1.0 / np.maximum(scores + 50.0, 50.0) ** 2
            return _weighted_median(values, weights, fallback), chosen, weights

        median3_iso, _, _ = aggregate(iso_order, 3, iso_scores)
        median5_iso, _, _ = aggregate(iso_order, 5, iso_scores)
        median3_aniso, _, _ = aggregate(aniso_order, 3, aniso_scores)
        median5_aniso, plane_indices, plane_weights = aggregate(
            aniso_order, 5, aniso_scores
        )
        plane5 = _local_plane(
            transferred[plane_indices],
            mapped[plane_indices],
            target_xy,
            plane_weights,
            median5_aniso,
        )

        generated = {
            "median3_iso": median3_iso,
            "median5_iso": median5_iso,
            "median3_aniso": median3_aniso,
            "median5_aniso": median5_aniso,
            "plane5_aniso": plane5,
        }
        for name, curve in generated.items():
            curve = np.asarray(curve, np.float64)
            bad = ~np.isfinite(curve)
            curve[bad] = fallback[bad]
            candidates[name][output_indices] = curve.astype(np.float32)

        reference_rows.append(
            {
                "well": target_well,
                "references_available": len(kept_references),
                "references": "|".join(kept_references),
                "iso_score_min": float(np.min(iso_scores)),
                "aniso_score_min": float(np.min(aniso_scores)),
                "coverage_min": float(np.min(coverages)),
                "coverage_median": float(np.median(coverages)),
            }
        )
    except Exception as exc:
        transfer_failures.append({"well": target_well, "reason": repr(exc)})

    if done % 100 == 0:
        print(
            f"multi-neighbor {done}/{len(wells)} ok={len(reference_rows)} "
            f"elapsed_min={(time.time()-started)/60:.1f}",
            flush=True,
        )

reference_report = pd.DataFrame(reference_rows)
failure_report = pd.DataFrame(transfer_failures)
reference_report.to_csv(OUT_DIR / "reference_coverage.csv", index=False)
failure_report.to_csv(OUT_DIR / "transfer_failures.csv", index=False)
print(
    "reference coverage:",
    reference_report["references_available"].describe().to_dict(),
)


# ---------------------------------------------------------------------------
# Stable folds.
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
eligible &= np.isfinite(single_v0)


# ---------------------------------------------------------------------------
# Score all fixed candidates on each historical downstream base.
# ---------------------------------------------------------------------------
score_rows = []
fold_rows = []
for base_col in base_cols:
    base = pool[base_col].to_numpy(np.float64)
    base_rmse = _rmse(truth, base)
    well_base_sse = (
        pd.DataFrame({"well": pool["well"], "se": (base - truth) ** 2})
        .groupby("well", as_index=False)["se"]
        .sum()
    )
    top40_wells = set(well_base_sse.nlargest(40, "se")["well"].astype(str))
    top40 = pool["well"].isin(top40_wells).to_numpy()

    for name in CANDIDATE_NAMES:
        candidate = candidates[name].astype(np.float64)
        move = np.zeros(len(pool), np.float64)
        valid = eligible & np.isfinite(candidate)
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
        score_rows.append(
            {
                "base_col": base_col,
                "candidate": name,
                "base_rmse": base_rmse,
                "prediction_rmse": _rmse(truth, prediction),
                "gain": base_rmse - _rmse(truth, prediction),
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


# Nested variant selection is a secondary anti-grid diagnostic. For each held
# fold, candidate choice uses only the other four folds.
nested_rows = []
for base_col in base_cols:
    for held_fold in range(1, N_FOLDS + 1):
        train_folds = [fold for fold in range(1, N_FOLDS + 1) if fold != held_fold]
        train_rows = fold_report[
            (fold_report["base_col"] == base_col)
            & fold_report["fold"].isin(train_folds)
        ]
        train_summary = (
            train_rows.groupby("candidate", as_index=False)
            .agg(train_mean_gain=("gain", "mean"), train_min_gain=("gain", "min"))
            .sort_values(
                ["train_mean_gain", "train_min_gain", "candidate"],
                ascending=[False, False, True],
            )
        )
        selected_name = str(train_summary.iloc[0]["candidate"])
        valid = fold_report[
            (fold_report["base_col"] == base_col)
            & (fold_report["candidate"] == selected_name)
            & (fold_report["fold"] == held_fold)
        ].iloc[0]
        nested_rows.append(
            {
                "base_col": base_col,
                "fold": held_fold,
                "selected_candidate": selected_name,
                "train_mean_gain": float(train_summary.iloc[0]["train_mean_gain"]),
                "train_min_gain": float(train_summary.iloc[0]["train_min_gain"]),
                "valid_gain": float(valid["gain"]),
                "valid_base_rmse": float(valid["base_rmse"]),
                "valid_prediction_rmse": float(valid["prediction_rmse"]),
            }
        )
nested = pd.DataFrame(nested_rows)


def _row(base_col, candidate):
    return scores[
        (scores["base_col"] == base_col) & (scores["candidate"] == candidate)
    ].iloc[0]


primary_base = "cp12" if "cp12" in base_cols else "strong"
primary = _row(primary_base, PRIMARY_CANDIDATE)
single = _row(primary_base, "single_v0")
primary_fold = fold_report[
    (fold_report["base_col"] == primary_base)
    & (fold_report["candidate"] == PRIMARY_CANDIDATE)
].sort_values("fold")
single_fold = fold_report[
    (fold_report["base_col"] == primary_base)
    & (fold_report["candidate"] == "single_v0")
].sort_values("fold")
fold_improvement = (
    primary_fold["prediction_rmse"].to_numpy()
    < single_fold["prediction_rmse"].to_numpy()
)

strict_pass = bool(
    float(primary["gain"]) >= float(single["gain"]) + 0.03
    and int(primary["positive_folds"]) == N_FOLDS
    and int(fold_improvement.sum()) >= 4
    and float(primary["top40_gain"]) >= float(single["top40_gain"])
)

result = {
    "pool": str(POOL_PATH),
    "source": str(SOURCE_DIR),
    "fold_source": fold_source,
    "primary_base": primary_base,
    "primary_candidate": PRIMARY_CANDIDATE,
    "single_gain": float(single["gain"]),
    "primary_gain": float(primary["gain"]),
    "incremental_gain_vs_single": float(primary["gain"] - single["gain"]),
    "primary_min_fold_gain": float(primary["min_fold_gain"]),
    "primary_positive_folds": int(primary["positive_folds"]),
    "folds_better_than_single": int(fold_improvement.sum()),
    "single_top40_gain": float(single["top40_gain"]),
    "primary_top40_gain": float(primary["top40_gain"]),
    "reference_coverage": {
        str(key): float(value)
        for key, value in reference_report["references_available"]
        .describe()
        .to_dict()
        .items()
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> freeze the winning multi-neighbor generator for Kaggle"
        if strict_pass
        else "STOP fixed multi-neighbor aggregation; continue to point-projection audit"
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

print("\nSCORES")
display(scores.sort_values(["base_col", "gain"], ascending=[True, False]))
print("\nFOLDS")
display(fold_report)
print("\nNESTED VARIANT SELECTION")
display(nested)
print("\nRESULT", result)
print("saved:", OUT_DIR)
gc.collect()
