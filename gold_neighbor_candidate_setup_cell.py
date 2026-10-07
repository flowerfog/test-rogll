"""Register full-reference neighbor candidates for Gold forward chaining.

Run after the original Gold helper-definition cell and before the patched
block_paired_gold_forward_chaining_cell.py. The factory is target-free: it uses
only target geometry/Z/visible TVT_input and the complete labeled reference
well selected by neighbor_fullcurve_transfer_v0.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.isotonic import IsotonicRegression


_NG_DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
_NG_SOURCE = Path("/root/autodl-tmp/oof_exports/neighbor_fullcurve_transfer_v0")
_NG_SELECTED_PATH = _NG_SOURCE / "selected_neighbors.csv"
if not _NG_SELECTED_PATH.exists():
    raise FileNotFoundError(_NG_SELECTED_PATH)

_ng_selected = pd.read_csv(_NG_SELECTED_PATH, dtype={"well": str, "neighbor": str})
_ng_selected["well"] = _ng_selected["well"].astype(str)
_ng_selected["neighbor"] = _ng_selected["neighbor"].astype(str)
_NG_NEIGHBOR_MAP = _ng_selected.set_index("well")["neighbor"].to_dict()
_NG_REFERENCE_CACHE = {}


def _ng_cumulative_xy(xy):
    step = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    step[~np.isfinite(step)] = 0.0
    return np.r_[0.0, np.cumsum(step)]


def _ng_unique_progress(progress, *arrays):
    progress = np.asarray(progress, float)
    keep = np.r_[np.diff(progress) > 1e-8, True]
    output = [progress[keep]]
    output.extend(np.asarray(array)[keep] for array in arrays)
    return output


def _ng_load_reference(wid):
    wid = str(wid)
    if wid in _NG_REFERENCE_CACHE:
        return _NG_REFERENCE_CACHE[wid]
    path = _NG_DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
    xy = frame[["X", "Y"]].to_numpy(float)
    z = frame["Z"].to_numpy(float)
    structure = frame["TVT"].to_numpy(float) + z
    progress = _ng_cumulative_xy(xy)
    visible = frame["TVT_input"].notna().to_numpy()
    anchor = int(np.flatnonzero(visible)[-1]) if visible.any() else 0
    direction = xy[-1] - xy[anchor]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0:
        direction = xy[-1] - xy[0]
        norm = max(float(np.linalg.norm(direction)), 1.0)
    direction = direction / norm
    unique_progress, unique_structure = _ng_unique_progress(progress, structure)
    record = {
        "xy": xy,
        "z": z,
        "structure": structure,
        "progress": progress,
        "unique_progress": unique_progress,
        "unique_structure": unique_structure,
        "tree": cKDTree(xy, leafsize=64),
        "direction": direction,
    }
    _NG_REFERENCE_CACHE[wid] = record
    return record


def _ng_reanchor(mapped_structure, target_z, anchor_tvt, anchor_z, ref_anchor_structure):
    return (
        np.asarray(mapped_structure, float)
        - np.asarray(target_z, float)
        + (float(anchor_tvt) - (float(ref_anchor_structure) - float(anchor_z)))
    )


def _ng_candidate_factory(wid, hw_masked):
    wid = str(wid)
    reference_wid = _NG_NEIGHBOR_MAP.get(wid)
    if reference_wid is None:
        return {}
    reference = _ng_load_reference(reference_wid)

    required = {"X", "Y", "Z", "TVT_input"}
    if not required.issubset(hw_masked.columns):
        return {}
    target_xy = hw_masked[["X", "Y"]].to_numpy(float)
    target_z = hw_masked["Z"].to_numpy(float)
    target_tvt = hw_masked["TVT_input"].to_numpy(float)
    known = np.flatnonzero(np.isfinite(target_tvt))
    if len(known) < 30:
        return {}
    anchor = int(known[-1])
    anchor_tvt = float(target_tvt[anchor])
    anchor_z = float(target_z[anchor])
    target_progress = _ng_cumulative_xy(target_xy)
    centered_progress = target_progress - target_progress[anchor]
    target_direction = target_xy[-1] - target_xy[anchor]
    norm = float(np.linalg.norm(target_direction))
    if norm <= 1.0:
        return {}
    target_direction /= norm

    _, ref_anchor = reference["tree"].query(target_xy[anchor], k=1)
    ref_anchor = int(ref_anchor)
    ref_anchor_progress = float(reference["progress"][ref_anchor])
    ref_anchor_structure = float(reference["structure"][ref_anchor])
    sign = 1.0 if float(np.dot(target_direction, reference["direction"])) >= 0 else -1.0

    query = ref_anchor_progress + sign * centered_progress
    progress_valid = (query >= reference["unique_progress"][0]) & (
        query <= reference["unique_progress"][-1]
    )
    progress_structure = np.full(len(hw_masked), np.nan, float)
    progress_structure[progress_valid] = np.interp(
        query[progress_valid],
        reference["unique_progress"],
        reference["unique_structure"],
    )
    progress_candidate = _ng_reanchor(
        progress_structure,
        target_z,
        anchor_tvt,
        anchor_z,
        ref_anchor_structure,
    )

    _, nearest = reference["tree"].query(target_xy, k=1)
    nearest = nearest.astype(np.int64)
    raw_structure = reference["structure"][nearest]
    raw_candidate = _ng_reanchor(
        raw_structure,
        target_z,
        anchor_tvt,
        anchor_z,
        ref_anchor_structure,
    )

    raw_progress = reference["progress"][nearest]
    isotonic = IsotonicRegression(
        increasing=bool(sign > 0), out_of_bounds="clip"
    )
    sample_weight = np.ones(len(hw_masked), float)
    sample_weight[anchor] = 100.0
    monotonic_query = isotonic.fit_transform(
        target_progress, raw_progress, sample_weight=sample_weight
    )
    monotonic_query = np.clip(
        monotonic_query,
        reference["unique_progress"][0],
        reference["unique_progress"][-1],
    )
    monotonic_structure = np.interp(
        monotonic_query,
        reference["unique_progress"],
        reference["unique_structure"],
    )
    monotonic_candidate = _ng_reanchor(
        monotonic_structure,
        target_z,
        anchor_tvt,
        anchor_z,
        ref_anchor_structure,
    )
    hybrid = np.where(
        np.isfinite(progress_candidate),
        0.5 * progress_candidate + 0.5 * monotonic_candidate,
        monotonic_candidate,
    )

    return {
        "neighbor_full_progress": progress_candidate.astype(float),
        "neighbor_full_xy": raw_candidate.astype(float),
        "neighbor_hybrid": hybrid.astype(float),
    }


_GOLD_EXTRA_CANDIDATE_FACTORY = _ng_candidate_factory
_GOLD_EXTRA_CANDIDATE_PREFIXES = ("neighbor_",)
_GOLD_BLOCK_CACHE_VERSION_OVERRIDE = "v3_neighbor_fullref_s24_p350_b128"

print(
    {
        "gold_extra_candidates": list(_ng_candidate_factory.__name__ for _ in [0]),
        "candidate_names": [
            "neighbor_full_progress",
            "neighbor_full_xy",
            "neighbor_hybrid",
        ],
        "neighbor_wells": len(_NG_NEIGHBOR_MAP),
        "cache_version": _GOLD_BLOCK_CACHE_VERSION_OVERRIDE,
    }
)
