"""Final-cell Kaggle inference for the selected v27 low-pass tail Neighbor.

Run this only after the complete 6.39 pipeline has written
``/kaggle/working/submission.csv``. The first 1000 ft after the visible anchor
is preserved exactly. The structure-progress Neighbor delta is first filtered
with a centred 256-ft MD box filter, then clipped, and finally consumed with a
0.30 dose after the 1000-1250 ft smoothstep.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Frozen configuration
# -----------------------------------------------------------------------------
BASE_OVERRIDE = None
DATA_ROOT_OVERRIDE = None
# Use "progress_v0" only to reconstruct the exact LB-confirmed 6.882
# before/after pair. Keep "full_progress_geom" for the experimental upgrade.
MAPPING_MODE = "progress_v0"
if MAPPING_MODE not in {"progress_v0", "full_progress_geom"}:
    raise ValueError(MAPPING_MODE)

WEIGHT = 0.30
RAW_MOVE_CAP = 24.0
APPLIED_MOVE_CAP = WEIGHT * RAW_MOVE_CAP
TAIL_START_FT = 1000.0
TAIL_TRANSITION_FT = 250.0
LOWPASS_WINDOW_FT = 256.0
GEOMETRY_STRIDE = 16
GLOBAL_QUERY_K = 96
SHORTLIST_SIZE = 24
MIN_ABS_DIRECTION_DOT = 0.70

# coherent600 gate, frozen from honest OOF
MAX_MEDIAN_DISTANCE = 600.0
MAX_P90_DISTANCE = 1200.0
MAX_ANCHOR_DISTANCE = 1200.0

# Honest v27 OOF evidence. These are reported for lineage only and are not
# hidden-distribution gates.
OOF_AUDIT = {
    "selection_recipe": "256ft_lowpass_cap24_flat_dose_0.30",
    "fold3_gain_vs_raw_frozen": 0.09954823168251536,
    "fold2_gain_vs_raw_frozen": 0.03407908263055326,
    "fold1_gain_vs_raw_frozen": 0.08418200300236656,
    "fold12_gain_vs_raw_frozen": 0.05976299703646326,
    "fold2_p95_change_vs_raw_frozen": -1.0722875525684827,
    "fold1_p95_change_vs_raw_frozen": -0.31502692521232945,
}

# Enforced only on the real hidden-sized test set, not the three-well preview.
HIDDEN_GUARD = {
    "eligible_well_fraction": (0.45, 0.94),
    "eligible_row_fraction": (0.45, 0.92),
    "median_neighbor_distance": (100.0, 1000.0),
    "opposite_direction_fraction": (0.01, 0.30),
    "mean_abs_move": (0.30, 2.70),
    "p95_abs_move": (1.00, 6.000001),
}
MIN_WELLS_FOR_GUARD = 100


def _tail_weight(distance_ft):
    t = np.clip(
        (np.asarray(distance_ft, np.float64) - TAIL_START_FT)
        / TAIL_TRANSITION_FT,
        0.0,
        1.0,
    )
    return t * t * (3.0 - 2.0 * t)


def _centred_lowpass(md, values, valid, window_ft):
    """NaN-aware centred box filter, matched to the v27 OOF implementation."""
    md = np.asarray(md, np.float64)
    values = np.asarray(values, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(values)
    if len(md) == 0 or np.any(np.diff(md) < 0.0):
        raise RuntimeError("Low-pass input MD must be sorted")
    half = 0.5 * float(window_ft)
    left = np.searchsorted(md, md - half, side="left")
    right = np.searchsorted(md, md + half, side="right")
    weighted = np.where(valid, values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(weighted, dtype=np.float64)))
    count = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    sums = csum[right] - csum[left]
    counts = count[right] - count[left]
    output_valid = counts > 0
    output = np.zeros(len(values), np.float64)
    output[output_valid] = sums[output_valid] / counts[output_valid]
    return output, output_valid


def _find_data_root():
    if DATA_ROOT_OVERRIDE is not None:
        path = Path(DATA_ROOT_OVERRIDE)
        if (path / "train").exists() and (path / "test").exists():
            return path
        raise FileNotFoundError(path)
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
    ]
    for path in candidates:
        if (path / "train").exists() and (path / "test").exists():
            return path
    for train_dir in Path("/kaggle/input").glob("**/train"):
        root = train_dir.parent
        if (root / "test").exists() and list(train_dir.glob("*__horizontal_well.csv")):
            return root
    raise FileNotFoundError("cannot locate competition train/test directories")


def _find_base():
    candidates = []
    if BASE_OVERRIDE is not None:
        candidates.append(Path(BASE_OVERRIDE))
    candidates.extend(
        [
            # This script is a final cell: preserve the exact upstream chain.
            Path("/kaggle/working/submission.csv"),
            Path("/kaggle/working/submission_after_gold.csv"),
            Path("/kaggle/working/submission_strong.csv"),
            Path("/kaggle/working/submission_sp45_fle_projected.csv"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing current upstream submission. Run the complete 6.39 pipeline first so "
        "/kaggle/working/submission.csv exists, or set BASE_OVERRIDE."
    )


def _normalise_submission(path):
    frame = pd.read_csv(path)
    if "id" not in frame.columns:
        raise RuntimeError(f"submission has no id column: {path}")
    prediction_columns = [column for column in frame.columns if column != "id"]
    if "tvt" in prediction_columns:
        prediction_column = "tvt"
    elif len(prediction_columns) == 1:
        prediction_column = prediction_columns[0]
    else:
        raise RuntimeError(f"cannot identify TVT prediction column: {frame.columns.tolist()}")
    output = frame[["id", prediction_column]].rename(columns={prediction_column: "tvt"})
    output["id"] = output["id"].astype(str)
    output["tvt"] = pd.to_numeric(output["tvt"], errors="coerce")
    if output["id"].duplicated().any() or not np.isfinite(output["tvt"]).all():
        raise RuntimeError("invalid strong-stage submission")
    return output


def _cumulative_xy(xy):
    step = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    step[~np.isfinite(step)] = 0.0
    return np.r_[0.0, np.cumsum(step)]


def _unique_progress(progress, *arrays):
    """Keep the last point from each repeated XY-progress run."""
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


DATA_ROOT = _find_data_root()
BASE_PATH = _find_base()
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

print("DATA_ROOT:", DATA_ROOT)
print("UPSTREAM COMPLETE 6.39 BASE:", BASE_PATH)
print(
    "FROZEN CONFIG:",
    {
        "weight": WEIGHT,
        "raw_move_cap": RAW_MOVE_CAP,
        "applied_move_cap": APPLIED_MOVE_CAP,
        "gate": "all_with_coverage_ge_0.80",
        "method": MAPPING_MODE,
        "tail_start_ft": TAIL_START_FT,
        "tail_transition_ft": TAIL_TRANSITION_FT,
        "lowpass_window_ft": LOWPASS_WINDOW_FT,
    },
)


# -----------------------------------------------------------------------------
# Load the exact complete 6.39 test predictions
# -----------------------------------------------------------------------------
base_submission = _normalise_submission(BASE_PATH)
if BASE_PATH.resolve() == (WORK / "submission.csv").resolve():
    current = base_submission.sort_values("id").reset_index(drop=True)
    for previous_candidate in (
        WORK / "submission_frontier_6p39_tail_lowpass_v27.csv",
        WORK / "submission_frontier_6p39_tail_neighbor.csv",
    ):
        if not previous_candidate.exists():
            continue
        previous = _normalise_submission(previous_candidate)
        previous = previous.sort_values("id").reset_index(drop=True)
        if current["id"].equals(previous["id"]) and np.allclose(
            current["tvt"], previous["tvt"], rtol=0.0, atol=1e-10
        ):
            raise RuntimeError(
                "submission.csv already contains a tail Neighbor result. Rerun "
                "the complete upstream 6.39 cells before this v27 final cell."
            )
# This script is also safe as the final cell of a complete submission notebook:
# the current submission is loaded fully before it is overwritten below.  Keep
# an automatic before-copy for lineage and movement audits.
base_submission[["id", "tvt"]].to_csv(
    WORK / "submission_before_frontier_6p39_tail_lowpass_v27.csv", index=False
)
base_submission["well"] = base_submission["id"].str.rsplit("_", n=1).str[0]
base_submission["row_idx"] = (
    base_submission["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
)
base_submission = base_submission.sort_values(["well", "row_idx"]).reset_index(drop=True)
test_wells = sorted(base_submission["well"].unique())
test_indices = {
    well: np.asarray(indices, np.int64)
    for well, indices in base_submission.groupby("well", sort=False).indices.items()
}

test_files = {
    path.name.split("__")[0]: path
    for path in (DATA_ROOT / "test").glob("*__horizontal_well.csv")
}
missing_test_files = sorted(set(test_wells) - set(test_files))
if missing_test_files:
    raise RuntimeError(f"missing test horizontal files: {missing_test_files[:20]}")


# -----------------------------------------------------------------------------
# Build the labeled train reference bank
# -----------------------------------------------------------------------------
reference_files = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
reference_wells = [path.name.split("__")[0] for path in reference_files]
reference_to_code = {well: code for code, well in enumerate(reference_wells)}
reference_data = {}
sample_xy_parts = []
sample_code_parts = []
started = time.time()

for path in tqdm(reference_files, desc="load labeled reference wells"):
    well = path.name.split("__")[0]
    frame = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
    visible = frame["TVT_input"].notna().to_numpy()
    if not visible.any() or visible.all():
        raise RuntimeError(f"invalid train TVT_input mask: {well}")
    anchor = int(np.flatnonzero(visible)[-1])
    xy = frame[["X", "Y"]].to_numpy(np.float64)
    if not np.isfinite(xy).all():
        raise RuntimeError(f"non-finite train XY: {well}")
    direction = xy[-1] - xy[anchor]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0:
        raise RuntimeError(f"degenerate train direction: {well}")
    direction /= norm

    lateral_indices = np.arange(anchor, len(frame), dtype=np.int32)
    sample_indices = lateral_indices[::GEOMETRY_STRIDE]
    if sample_indices[-1] != lateral_indices[-1]:
        sample_indices = np.r_[sample_indices, lateral_indices[-1]]

    structure = frame["TVT"].to_numpy(np.float64) + frame["Z"].to_numpy(np.float64)
    progress = _cumulative_xy(xy)
    unique_progress, unique_structure, unique_x, unique_y = _unique_progress(
        progress,
        structure,
        xy[:, 0],
        xy[:, 1],
    )
    reference_data[well] = {
        "xy": xy,
        "z": frame["Z"].to_numpy(np.float64),
        "tvt": frame["TVT"].to_numpy(np.float64),
        "structure": structure,
        "progress": progress,
        "unique_progress": unique_progress,
        "unique_structure": unique_structure,
        "unique_x": unique_x,
        "unique_y": unique_y,
        "tree": cKDTree(xy, leafsize=64),
        "anchor": anchor,
        "direction": direction,
        "lateral_indices": lateral_indices,
        "sample_xy": xy[sample_indices],
    }
    sample_xy_parts.append(xy[sample_indices])
    sample_code_parts.append(
        np.full(len(sample_indices), reference_to_code[well], dtype=np.int32)
    )

global_sample_xy = np.concatenate(sample_xy_parts, axis=0)
global_sample_codes = np.concatenate(sample_code_parts, axis=0)
global_tree = cKDTree(global_sample_xy, leafsize=64)
print(
    "REFERENCE BANK:",
    {
        "wells": len(reference_wells),
        "sample_points": len(global_sample_xy),
        "elapsed_sec": time.time() - started,
    },
)


def _build_test_target(well):
    frame = pd.read_csv(test_files[well], usecols=["MD", "X", "Y", "Z", "TVT_input"])
    visible = frame["TVT_input"].notna().to_numpy()
    if not visible.any() or visible.all():
        raise RuntimeError(f"invalid test TVT_input mask: {well}")
    anchor = int(np.flatnonzero(visible)[-1])
    xy = frame[["X", "Y"]].to_numpy(np.float64)
    if not np.isfinite(xy).all():
        raise RuntimeError(f"non-finite test XY: {well}")
    direction = xy[-1] - xy[anchor]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0:
        raise RuntimeError(f"degenerate test direction: {well}")
    direction /= norm
    lateral_indices = np.arange(anchor, len(frame), dtype=np.int32)
    sample_indices = lateral_indices[::GEOMETRY_STRIDE]
    if sample_indices[-1] != lateral_indices[-1]:
        sample_indices = np.r_[sample_indices, lateral_indices[-1]]
    return {
        "frame": frame,
        "xy": xy,
        "z": frame["Z"].to_numpy(np.float64),
        "anchor": anchor,
        "anchor_tvt": float(frame["TVT_input"].iloc[anchor]),
        "direction": direction,
        "sample_xy": xy[sample_indices],
    }


def _shortlist_neighbors(target_well, target):
    target_code = reference_to_code.get(target_well)
    best_contact = {}
    query_sizes = []
    k = int(GLOBAL_QUERY_K)
    while True:
        query_sizes.append(min(k, len(global_sample_xy)))
        if query_sizes[-1] == len(global_sample_xy) or query_sizes[-1] >= 3072:
            break
        k *= 2

    for k in query_sizes:
        distances, indices = global_tree.query(target["sample_xy"], k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        codes = global_sample_codes[indices].ravel()
        values = distances.ravel()
        if target_code is not None:
            keep = codes != target_code
            codes = codes[keep]
            values = values[keep]
        for code, distance in zip(codes, values):
            code = int(code)
            previous = best_contact.get(code)
            if previous is None or distance < previous:
                best_contact[code] = float(distance)
        if len(best_contact) >= SHORTLIST_SIZE:
            break

    if not best_contact:
        raise RuntimeError(f"no other-well reference candidates: {target_well}")
    ranked = sorted(best_contact, key=best_contact.get)[:SHORTLIST_SIZE]
    return [reference_wells[code] for code in ranked]


def _select_neighbor(target_well, target):
    anchor_xy = target["xy"][target["anchor"]]
    rows = []
    for reference_well in _shortlist_neighbors(target_well, target):
        reference = reference_data[reference_well]
        tree = cKDTree(reference["sample_xy"], leafsize=32)
        distances, _ = tree.query(target["sample_xy"], k=1)
        anchor_distance = float(tree.query(anchor_xy, k=1)[0])
        direction_dot = float(np.dot(target["direction"], reference["direction"]))
        rows.append(
            {
                "neighbor": reference_well,
                "min_distance": float(np.min(distances)),
                "median_distance": float(np.median(distances)),
                "p90_distance": float(np.quantile(distances, 0.90)),
                "max_distance": float(np.max(distances)),
                "anchor_distance_sampled": anchor_distance,
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
    report = pd.DataFrame(rows)
    parallel = report[report["parallel"]]
    chosen = (parallel if len(parallel) else report).sort_values("geometry_score").iloc[0]
    return chosen.to_dict()


def _structure_progress(target, reference_well, row_indices):
    reference = reference_data[reference_well]
    target_rows = row_indices.astype(np.int32)
    if np.any(target_rows <= target["anchor"]) or np.any(target_rows >= len(target["xy"])):
        raise RuntimeError("submission rows do not match the hidden eval suffix")

    ref_indices = reference["lateral_indices"]
    ref_xy = reference["xy"][ref_indices]
    ref_structure = reference["tvt"][ref_indices] + reference["z"][ref_indices]
    ref_tree = cKDTree(ref_xy, leafsize=64)
    anchor_xy = target["xy"][target["anchor"]]
    anchor_distance, anchor_nearest = ref_tree.query(anchor_xy, k=1)
    anchor_nearest = int(anchor_nearest)

    ref_progress = _cumulative_xy(ref_xy)
    target_progress_all = _cumulative_xy(target["xy"][target["anchor"] :])
    target_progress = target_progress_all[target_rows - target["anchor"]]
    orientation = 1.0 if np.dot(target["direction"], reference["direction"]) >= 0 else -1.0
    query = float(ref_progress[anchor_nearest]) + orientation * target_progress
    valid = (query >= ref_progress[0]) & (query <= ref_progress[-1])

    prediction = np.full(len(target_rows), np.nan, np.float64)
    if valid.any():
        anchor_z = float(target["z"][target["anchor"]])
        anchor_tvt = float(target["anchor_tvt"])
        offset = anchor_tvt - (float(ref_structure[anchor_nearest]) - anchor_z)
        prediction[valid] = (
            np.interp(query[valid], ref_progress, ref_structure)
            - target["z"][target_rows[valid]]
            + offset
        )
    return prediction, valid, float(anchor_distance)


def _full_progress_one_sign(target, reference, row_indices, sign):
    target_rows = row_indices.astype(np.int32)
    target_xy = target["xy"][target_rows]
    target_z = target["z"][target_rows]
    target_progress_all = _cumulative_xy(target["xy"][target["anchor"] :])
    target_progress = target_progress_all[target_rows - target["anchor"]]

    anchor_xy = target["xy"][target["anchor"]]
    anchor_distance, ref_anchor = reference["tree"].query(anchor_xy, k=1)
    ref_anchor = int(ref_anchor)
    query = float(reference["progress"][ref_anchor]) + float(sign) * target_progress
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

    anchor_z = float(target["z"][target["anchor"]])
    ref_anchor_structure = float(reference["structure"][ref_anchor])
    offset = float(target["anchor_tvt"]) - (ref_anchor_structure - anchor_z)
    candidate = mapped_structure - target_z + offset
    score, coverage = _geometry_score(target_xy, mapped_xy, valid)
    return {
        "candidate": candidate,
        "valid": valid,
        "score": float(score),
        "coverage": float(coverage),
        "anchor_distance": float(anchor_distance),
    }


def _full_progress_geometry(target, reference_well, row_indices):
    # Preserve v0 as the exact row-level fallback and as the frozen gate basis.
    fallback, fallback_valid, gate_anchor_distance = _structure_progress(
        target, reference_well, row_indices
    )
    reference = reference_data[reference_well]
    positive = _full_progress_one_sign(target, reference, row_indices, +1.0)
    negative = _full_progress_one_sign(target, reference, row_indices, -1.0)
    selected = positive if positive["score"] <= negative["score"] else negative
    candidate = np.asarray(selected["candidate"], np.float64).copy()
    use_fallback = ~np.isfinite(candidate)
    candidate[use_fallback] = fallback[use_fallback]
    # Full-reference coverage is intentionally wider than v0. The honest OOF
    # experiment applied every finite full candidate, including rows where the
    # lateral-only fallback was out of range.
    valid = np.isfinite(candidate)
    return candidate, valid, gate_anchor_distance, {
        "geometry_sign": 1 if selected is positive else -1,
        "positive_geometry_score": positive["score"],
        "negative_geometry_score": negative["score"],
        "geometry_coverage": selected["coverage"],
        "fallback_row_fraction": float(use_fallback.mean()),
        "full_anchor_distance": selected["anchor_distance"],
    }


# -----------------------------------------------------------------------------
# Apply the frozen correction
# -----------------------------------------------------------------------------
base_values = base_submission["tvt"].to_numpy(np.float64)
candidate_values = np.full(len(base_submission), np.nan, np.float64)
move = np.zeros(len(base_submission), np.float64)
eligible_rows = np.zeros(len(base_submission), bool)
frozen_rows = np.zeros(len(base_submission), bool)
audit_rows = []
failures = []
started = time.time()

for well in tqdm(test_wells, desc="hidden neighbor transfer"):
    indices = test_indices[well]
    try:
        target = _build_test_target(well)
        chosen = _select_neighbor(well, target)
        row_indices = base_submission["row_idx"].to_numpy(np.int32)[indices]
        if MAPPING_MODE == "progress_v0":
            candidate, progress_valid, anchor_distance_full = _structure_progress(
                target, str(chosen["neighbor"]), row_indices
            )
            geometry = {
                "geometry_sign": int(np.sign(chosen["direction_dot"])) or 1,
                "positive_geometry_score": np.nan,
                "negative_geometry_score": np.nan,
                "geometry_coverage": float(progress_valid.mean()),
                "fallback_row_fraction": 0.0,
                "full_anchor_distance": anchor_distance_full,
            }
        else:
            candidate, progress_valid, anchor_distance_full, geometry = (
                _full_progress_geometry(
                    target, str(chosen["neighbor"]), row_indices
                )
            )
        candidate_values[indices] = candidate

        # The selected v27 recipe uses no geometric gate. Coverage is the only
        # well-level eligibility condition used by the matched audit.
        eligible_well = bool(float(progress_valid.mean()) >= 0.80)
        local_eligible = progress_valid & eligible_well & np.isfinite(candidate)
        local_md = target["frame"]["MD"].to_numpy(np.float64)[row_indices]
        anchor_md = float(target["frame"]["MD"].iloc[target["anchor"]])
        md_since = np.maximum(local_md - anchor_md, 0.0)
        tail_weight = _tail_weight(md_since)
        frozen_rows[indices] = md_since <= TAIL_START_FT
        if local_eligible.any():
            local_base = base_values[indices]
            filtered_delta, filtered_valid = _centred_lowpass(
                md_since,
                candidate - local_base,
                local_eligible,
                LOWPASS_WINDOW_FT,
            )
            local_eligible = filtered_valid & eligible_well
            local_move = np.zeros(len(indices), np.float64)
            local_move[local_eligible] = WEIGHT * tail_weight[local_eligible] * np.clip(
                filtered_delta[local_eligible],
                -RAW_MOVE_CAP,
                RAW_MOVE_CAP,
            )
            move[indices] = local_move
        eligible_rows[indices] = local_eligible

        chosen.update(geometry)
        chosen.update(
            {
                "well": well,
                "rows": int(len(indices)),
                "anchor_distance_full": anchor_distance_full,
                "eligible_well": eligible_well,
                "eligible_rows": int(local_eligible.sum()),
                "progress_coverage": float(progress_valid.mean()),
                "frozen_rows": int(np.sum(md_since <= TAIL_START_FT)),
                "mean_abs_move": float(np.mean(np.abs(move[indices]))),
                "p95_abs_move": float(np.quantile(np.abs(move[indices]), 0.95)),
                "max_abs_move": float(np.max(np.abs(move[indices]))),
            }
        )
        audit_rows.append(chosen)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})

audit = pd.DataFrame(audit_rows)
failure_report = pd.DataFrame(failures)
audit.to_csv(WORK / "frontier_6p39_tail_lowpass_v27_test_audit.csv", index=False)
failure_report.to_csv(
    WORK / "frontier_6p39_tail_lowpass_v27_test_failures.csv", index=False
)

if len(failures) > max(2, int(0.03 * len(test_wells))):
    raise RuntimeError(f"too many hidden neighbor-transfer failures: {len(failures)}")
if not np.isfinite(base_values).all() or not np.isfinite(move).all():
    raise RuntimeError("non-finite base or movement")
if float(np.max(np.abs(move))) > APPLIED_MOVE_CAP + 1e-6:
    raise RuntimeError("movement exceeds the selected 7.2 ft applied cap")

raw_prediction = base_values + move
if not np.array_equal(raw_prediction[frozen_rows], base_values[frozen_rows]):
    raise RuntimeError("the first 1000 ft was not preserved exactly")
eligible_well_fraction = float(audit["eligible_well"].mean()) if len(audit) else 0.0
eligible_row_fraction = float(eligible_rows.mean())
opposite_fraction = float((audit["direction_dot"] < 0).mean()) if len(audit) else 0.0
median_distance = float(audit["median_distance"].median()) if len(audit) else np.inf
mean_abs_move = float(np.mean(np.abs(move)))
p95_abs_move = float(np.quantile(np.abs(move), 0.95))
max_abs_move = float(np.max(np.abs(move)))

observed = {
    "wells": int(len(test_wells)),
    "rows": int(len(base_submission)),
    "successful_wells": int(len(audit)),
    "failed_wells": int(len(failures)),
    "eligible_well_fraction": eligible_well_fraction,
    "eligible_row_fraction": eligible_row_fraction,
    "median_neighbor_distance": median_distance,
    "opposite_direction_fraction": opposite_fraction,
    "mean_abs_move": mean_abs_move,
    "p95_abs_move": p95_abs_move,
    "max_abs_move": max_abs_move,
}

guard_checks = {
    "first_1000ft_exactly_frozen": {
        "value": bool(np.array_equal(raw_prediction[frozen_rows], base_values[frozen_rows])),
        "pass": True,
    },
    "maximum_move_le_7p2ft": {
        "value": max_abs_move,
        "pass": bool(max_abs_move <= APPLIED_MOVE_CAP + 1e-6),
    },
}
guard_pass = bool(all(item["pass"] for item in guard_checks.values()))
guard_status = "pass" if guard_pass else "fallback_to_parent"

final_values = raw_prediction if guard_pass else base_values.copy()
output = base_submission[["id"]].copy()
output["tvt"] = final_values
output = output.sort_values("id").reset_index(drop=True)

sample_path = DATA_ROOT / "sample_submission.csv"
if sample_path.exists():
    sample = pd.read_csv(sample_path)
    if set(sample["id"].astype(str)) != set(output["id"]):
        raise RuntimeError("output IDs do not match sample_submission")
    sample_order = sample[["id"]].copy()
    sample_order["id"] = sample_order["id"].astype(str)
    output = sample_order.merge(output, on="id", how="left", validate="one_to_one")
    if not np.isfinite(output["tvt"]).all():
        raise RuntimeError("non-finite predictions after sample-order alignment")

candidate_path = WORK / "submission_frontier_6p39_tail_lowpass_v27.csv"
final_path = WORK / "submission.csv"
output.to_csv(candidate_path, index=False)
output.to_csv(final_path, index=False)

report = {
    "experiment": "frontier_6p39_tail_lowpass_structure_progress_submit_v27",
    "data_root": str(DATA_ROOT),
    "upstream_source": str(BASE_PATH),
    "config": {
        "method": MAPPING_MODE,
        "gate": "all_with_coverage_ge_0.80",
        "weight": WEIGHT,
        "raw_move_cap": RAW_MOVE_CAP,
        "applied_move_cap": APPLIED_MOVE_CAP,
        "tail_start_ft": TAIL_START_FT,
        "tail_transition_ft": TAIL_TRANSITION_FT,
        "lowpass_window_ft": LOWPASS_WINDOW_FT,
        "filter_order": "lowpass_delta_then_clip_then_tail_dose",
        "reanchor": "none",
    },
    "oof_audit": OOF_AUDIT,
    "observed": observed,
    "guard_checks": guard_checks,
    "guard_status": guard_status,
    "movement_applied": bool(guard_pass and np.any(np.abs(move) > 0)),
    "elapsed_minutes": (time.time() - started) / 60,
    "candidate": str(candidate_path),
    "final": str(final_path),
}
audit_path = WORK / "frontier_6p39_tail_lowpass_v27_submit_audit.json"
with open(audit_path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, ensure_ascii=True)

print("\n6.39 V27 LOW-PASS TAIL NEIGHBOR SUBMISSION COMPLETE")
print(json.dumps(report, indent=2))
if guard_status != "pass":
    print("WARNING: structural guard failed; submission.csv is unchanged upstream base")
print("saved:", candidate_path)
print("final Kaggle file:", final_path)
