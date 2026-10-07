"""Nested sparse-path gateway for strong-centred GR candidate recall.

Prerequisite experiment v1 established that GR raises top-8 recall around the
fold-clean strong path, especially beyond 2000 ft, while pointwise argmax is
worse than strong.  This cell therefore never applies argmax.  It builds a
sparse per-node union of prior states and GR-ranked states, then decodes one
continuous complete-well path.

Protocol:
* folds 3/4/5: fit only the residual transition scale;
* Fold 1: select fixed prior/transition/dose parameters;
* Fold 2: one frozen application and final gateway;
* truth never enters candidate generation or decoding.
"""

from __future__ import annotations

import gc
import json
import runpy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


REQUIRED_V1 = {
    "DATA_ROOT",
    "OOF_ROOT",
    "POOL_PATH",
    "FOLD_PATH",
    "STATE_OFFSETS",
    "WINDOWS_FT",
    "_deduplicate_axis",
    "_robust_affine",
    "_robust_location_scale",
    "_fill_1d",
    "_window_scores",
    "_row_standardise",
    "_finite_mean",
    "_global_ncc",
    "_distance_subset",
}


def _restore_v1_if_needed():
    missing = sorted(REQUIRED_V1.difference(globals()))
    if not missing:
        print("V2 bootstrap: reusing live v1 definitions", flush=True)
        return
    candidates = (
        Path("/root/autodl-tmp/strong_centered_alignment_sufficiency_audit_v1_cell.py"),
        Path.cwd() / "strong_centered_alignment_sufficiency_audit_v1_cell.py",
    )
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise RuntimeError(
            "Missing v1 definitions and source. Upload "
            "strong_centered_alignment_sufficiency_audit_v1_cell.py to "
            "/root/autodl-tmp, or run v1 in this kernel first. Missing: "
            + repr(missing)
        )
    print("V2 bootstrap: running v1 prerequisite", source, flush=True)
    namespace = runpy.run_path(str(source))
    globals().update(
        {key: value for key, value in namespace.items() if not key.startswith("__")}
    )


_restore_v1_if_needed()

EXPERIMENT_V2 = "strong_centered_topk_path_gateway_v2"
OUT_DIR_V2 = Path(OOF_ROOT) / EXPERIMENT_V2
OUT_DIR_V2.mkdir(parents=True, exist_ok=True)

CORE_FOLDS_V2 = (3, 4, 5)
TUNE_FOLD_V2 = 1
TEST_FOLD_V2 = 2
NODE_STRIDE_V2 = 16
PRIOR_STATES_V2 = 9
GR_STATES_V2 = 8
STATE_COUNT_V2 = len(STATE_OFFSETS)
ZERO_STATE_V2 = int(np.argmin(np.abs(STATE_OFFSETS)))
PRIOR_SCALE_FT_V2 = 12.0
PRIOR_DECAY_FT_V2 = 1600.0
TRANSITION_LAMBDAS_V2 = (0.02, 0.05, 0.10, 0.20, 0.40)
PRIOR_WEIGHTS_V2 = (0.05, 0.10, 0.25, 0.50)
DOSES_V2 = (0.25, 0.50, 0.75, 1.00)
MAX_STATES_PER_NODE_V2 = PRIOR_STATES_V2 + GR_STATES_V2
BUILD_WORKERS_V2 = 3


def _jsonable_v2(value):
    if isinstance(value, dict):
        return {str(key): _jsonable_v2(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_v2(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable_v2(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _metric_v2(truth, base, prediction, well, distance):
    truth = np.asarray(truth, np.float64)
    base = np.asarray(base, np.float64)
    prediction = np.asarray(prediction, np.float64)
    well = np.asarray(well, str)
    distance = np.asarray(distance, np.float64)
    error = prediction - truth
    base_error = base - truth
    result = {
        "rows": int(len(error)),
        "wells": int(len(np.unique(well))),
        "base_rmse": float(np.sqrt(np.mean(base_error * base_error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
    }
    result["gain_vs_base"] = result["base_rmse"] - result["rmse"]
    well_rows = []
    for name in np.unique(well):
        mask = well == name
        candidate_rmse = float(np.sqrt(np.mean(error[mask] ** 2)))
        base_rmse = float(np.sqrt(np.mean(base_error[mask] ** 2)))
        well_rows.append((name, base_rmse, candidate_rmse, base_rmse - candidate_rmse))
    well_frame = pd.DataFrame(
        well_rows, columns=["well", "base_rmse", "rmse", "gain"]
    )
    result.update(
        {
            "median_well_rmse": float(well_frame["rmse"].median()),
            "p95_well_rmse": float(well_frame["rmse"].quantile(0.95)),
            "positive_well_fraction": float((well_frame["gain"] > 0).mean()),
        }
    )
    strata = {
        "near_0_500": distance < 500.0,
        "mid_500_1000": (distance >= 500.0) & (distance < 1000.0),
        "mid_1000_2000": (distance >= 1000.0) & (distance < 2000.0),
        "far_gt2000": distance >= 2000.0,
    }
    for name, mask in strata.items():
        if mask.any():
            result[f"{name}_base_rmse"] = float(
                np.sqrt(np.mean(base_error[mask] ** 2))
            )
            result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2)))
            result[f"{name}_gain"] = (
                result[f"{name}_base_rmse"] - result[f"{name}_rmse"]
            )
    return result, well_frame


folds_v2 = pd.read_csv(FOLD_PATH, dtype={"well": str})
well_column_v2 = next(
    (column for column in ("well", "well_id", "WELL") if column in folds_v2), None
)
fold_column_v2 = next(
    (column for column in ("fold", "outer_fold", "Fold") if column in folds_v2), None
)
if well_column_v2 is None or fold_column_v2 is None:
    raise RuntimeError(f"invalid fold metadata: {folds_v2.columns.tolist()}")
folds_v2 = folds_v2[[well_column_v2, fold_column_v2]].copy()
folds_v2.columns = ["well", "fold"]
folds_v2["well"] = folds_v2["well"].astype(str)
folds_v2["fold"] = pd.to_numeric(folds_v2["fold"], errors="raise").astype(np.int8)
folds_v2 = folds_v2.groupby("well", as_index=False)["fold"].first()
fold_map_v2 = dict(zip(folds_v2["well"], folds_v2["fold"]))

needed_wells_v2 = set(
    folds_v2.loc[
        folds_v2["fold"].isin((*CORE_FOLDS_V2, TUNE_FOLD_V2, TEST_FOLD_V2)),
        "well",
    ]
)
pool_header_v2 = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
usecols_v2 = [
    column
    for column in ("id", "well", "row_idx", "strong", "tvt_true")
    if column in pool_header_v2
]
if not {"id", "strong", "tvt_true"}.issubset(usecols_v2):
    raise RuntimeError("candidate pool misses id/strong/tvt_true")
pool_parts_v2 = []
for chunk_v2 in pd.read_csv(
    POOL_PATH,
    usecols=usecols_v2,
    dtype={"id": str, "well": str},
    chunksize=500_000,
):
    if "well" not in chunk_v2:
        chunk_v2["well"] = (
            chunk_v2["id"].astype(str).str.rsplit("_", n=1).str[0]
        )
    chunk_v2["well"] = chunk_v2["well"].astype(str)
    selected_v2 = chunk_v2[chunk_v2["well"].isin(needed_wells_v2)].copy()
    if len(selected_v2):
        pool_parts_v2.append(selected_v2)
pool_v2 = pd.concat(pool_parts_v2, ignore_index=True)
if "row_idx" not in pool_v2:
    pool_v2["row_idx"] = (
        pool_v2["id"].astype(str).str.rsplit("_", n=1).str[-1]
    )
pool_v2["row_idx"] = pd.to_numeric(pool_v2["row_idx"], errors="raise").astype(np.int32)
pool_v2["strong"] = pd.to_numeric(pool_v2["strong"], errors="coerce").astype(np.float32)
pool_v2["tvt_true"] = pd.to_numeric(pool_v2["tvt_true"], errors="coerce").astype(np.float32)
pool_v2["fold"] = pool_v2["well"].map(fold_map_v2)
if pool_v2["fold"].isna().any():
    raise RuntimeError("pool rows missing folds")
pool_v2 = pool_v2.sort_values(["well", "row_idx"]).reset_index(drop=True)
pool_groups_v2 = {
    well: frame.copy() for well, frame in pool_v2.groupby("well", sort=True)
}


def _fit_transition_v2():
    slopes = []
    changes = []
    wells = folds_v2.loc[folds_v2["fold"].isin(CORE_FOLDS_V2), "well"].tolist()
    for well in wells:
        frame = pool_groups_v2[well]
        horizontal = pd.read_csv(
            Path(DATA_ROOT) / "train" / f"{well}__horizontal_well.csv",
            usecols=["MD"],
        )
        row_index = frame["row_idx"].to_numpy(np.int64)
        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)[
            row_index
        ]
        residual = (
            frame["tvt_true"].to_numpy(np.float64)
            - frame["strong"].to_numpy(np.float64)
        )
        delta_md = np.diff(md)
        delta = np.diff(residual)
        valid = np.isfinite(delta_md) & np.isfinite(delta) & (delta_md > 1e-6)
        if valid.any():
            slopes.append(np.abs(delta[valid] / delta_md[valid]))
            changes.append(np.abs(delta[valid]))
    slope = np.concatenate(slopes)
    change = np.concatenate(changes)
    return {
        "slope_p99": float(np.quantile(slope, 0.99)),
        "slope_p999": float(np.quantile(slope, 0.999)),
        "change_p99": float(np.quantile(change, 0.99)),
        "samples": int(len(slope)),
    }


transition_prior_v2 = _fit_transition_v2()


def _top_indices_v2(score, count):
    score = np.asarray(score, np.float32)
    valid = np.isfinite(score)
    safe = np.where(valid, score, -np.inf)
    count = min(int(count), score.shape[1])
    partition = np.argpartition(safe, -count, axis=1)[:, -count:]
    values = np.take_along_axis(safe, partition, axis=1)
    order = np.argsort(values, axis=1)[:, ::-1]
    return np.take_along_axis(partition, order, axis=1)


def _make_union_v2(emission):
    prior_indices = np.argsort(np.abs(STATE_OFFSETS))[:PRIOR_STATES_V2]
    gr_indices = _top_indices_v2(emission, GR_STATES_V2)
    n_nodes = len(emission)
    indices = np.full((n_nodes, MAX_STATES_PER_NODE_V2), -1, np.int16)
    scores = np.full((n_nodes, MAX_STATES_PER_NODE_V2), -np.inf, np.float32)
    for row in range(n_nodes):
        ordered = []
        seen = set()
        for index in prior_indices.tolist():
            index = int(index)
            if index not in seen:
                seen.add(index)
                ordered.append(index)
        for index in gr_indices[row].tolist():
            index = int(index)
            if index not in seen and np.isfinite(emission[row, index]):
                seen.add(index)
                ordered.append(index)
        if ZERO_STATE_V2 not in seen:
            ordered.insert(0, ZERO_STATE_V2)
        ordered = ordered[:MAX_STATES_PER_NODE_V2]
        indices[row, : len(ordered)] = ordered
        selected_scores = emission[row, ordered]
        scores[row, : len(ordered)] = np.where(
            np.isfinite(selected_scores), selected_scores, 0.0
        )
    return indices, scores


def _build_record_v2(well, control="full"):
    frame = pool_groups_v2[well]
    horizontal = pd.read_csv(Path(DATA_ROOT) / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(Path(DATA_ROOT) / "train" / f"{well}__typewell.csv")
    row_index = frame["row_idx"].to_numpy(np.int64)
    md_all = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    gr_all = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    visible = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.isfinite(visible)
    anchor = int(np.flatnonzero(known)[-1])
    anchor_md = float(md_all[anchor])
    strong = frame["strong"].to_numpy(np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)

    base_all = np.full(len(horizontal), np.nan, np.float64)
    base_all[known] = visible[known]
    base_all[row_index] = strong
    base_all = _fill_1d(md_all, base_all)
    tw_tvt, tw_gr = _deduplicate_axis(
        pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64),
        pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64),
    )
    tw_at_known = np.interp(visible[known], tw_tvt, tw_gr, left=np.nan, right=np.nan)
    gain, intercept, _, calibration_rmse = _robust_affine(tw_at_known, gr_all[known])
    centre, scale = _robust_location_scale(gr_all[known])
    observed = (_fill_1d(md_all, gr_all) - centre) / scale
    tw_calibrated = (gain * tw_gr + intercept - centre) / scale
    candidate_tvt = base_all[:, None] + STATE_OFFSETS[None, :]
    expected = np.interp(
        candidate_tvt.reshape(-1),
        tw_tvt,
        tw_calibrated,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape).astype(np.float32)
    valid = np.isfinite(expected) & np.isfinite(gr_all)[:, None]

    node_position = np.unique(
        np.r_[np.arange(0, len(row_index), NODE_STRIDE_V2), len(row_index) - 1]
    ).astype(np.int64)
    node_full_index = row_index[node_position]
    median_step = max(float(abs(np.nanmedian(np.diff(md_all[np.isfinite(md_all)])))), 0.25)
    window_rows = {
        float(window): max(9, int(round(float(window) / median_step)))
        for window in WINDOWS_FT
    }
    window_scores = _window_scores(
        observed, expected, valid, node_full_index, window_rows
    )
    ncc = [_row_standardise(window_scores[window][0]) for window in WINDOWS_FT]
    neg_rmse = [
        _row_standardise(-window_scores[window][1]) for window in WINDOWS_FT
    ]
    local = _finite_mean([_finite_mean(ncc), _finite_mean(neg_rmse)])
    global_profile = _global_ncc(observed[row_index], expected[row_index], valid[row_index])
    global_score = _row_standardise(global_profile[None, :])
    global_volume = np.broadcast_to(global_score, local.shape).copy()
    emission = _finite_mean([local, global_volume])
    if control == "shuffled":
        shift = 31 + (int(str(well)[:4], 16) % 61)
        emission = np.roll(emission, shift, axis=1)
    indices, scores = _make_union_v2(emission)
    del expected, valid, window_scores, ncc, neg_rmse, local, global_volume

    return {
        "well": str(well),
        "fold": int(frame["fold"].iloc[0]),
        "md": md_all[row_index].astype(np.float32),
        "distance": (md_all[row_index] - anchor_md).astype(np.float32),
        "strong": strong.astype(np.float32),
        "truth": truth.astype(np.float32),
        "node_position": node_position.astype(np.int32),
        "node_md": md_all[node_full_index].astype(np.float32),
        "node_truth_residual": (truth[node_position] - strong[node_position]).astype(
            np.float32
        ),
        "candidate_indices": indices,
        "candidate_scores": scores,
        "emission": emission.astype(np.float32),
        "calibration_rmse": calibration_rmse,
    }


def _decode_sparse_v2(record, transition_lambda, prior_weight):
    indices = record["candidate_indices"]
    emission = record["candidate_scores"].astype(np.float64)
    node_md = record["node_md"].astype(np.float64)
    node_distance = np.maximum(record["distance"][record["node_position"]], 0.0)
    n_nodes, width = indices.shape
    offsets = np.where(indices >= 0, STATE_OFFSETS[np.maximum(indices, 0)], np.nan)
    prior_decay = 1.0 / (1.0 + node_distance / PRIOR_DECAY_FT_V2)
    prior = -0.5 * (offsets / PRIOR_SCALE_FT_V2) ** 2
    unary = emission + float(prior_weight) * prior_decay[:, None] * prior
    unary[indices < 0] = -np.inf

    back = np.full((n_nodes, width), -1, np.int16)
    previous = unary[0].copy()
    previous -= np.where(
        np.isfinite(offsets[0]),
        2.0 * np.abs(offsets[0]) / PRIOR_SCALE_FT_V2,
        0.0,
    )
    previous[indices[0] < 0] = -np.inf
    for node in range(1, n_nodes):
        delta_md = max(float(node_md[node] - node_md[node - 1]), 1.0)
        previous_offset = offsets[node - 1][:, None]
        current_offset = offsets[node][None, :]
        delta = current_offset - previous_offset
        max_jump = max(
            4.0,
            transition_prior_v2["slope_p999"] * delta_md + 2.0,
        )
        transition = -float(transition_lambda) * delta * delta / delta_md
        transition[np.abs(delta) > max_jump] = -np.inf
        candidates = previous[:, None] + transition
        best_previous = np.argmax(candidates, axis=0)
        best_value = candidates[best_previous, np.arange(width)]
        current = unary[node] + best_value
        current[indices[node] < 0] = -np.inf
        if not np.isfinite(current).any():
            current = unary[node].copy()
            best_previous[:] = int(np.nanargmax(previous))
        back[node] = best_previous.astype(np.int16)
        previous = current

    state_slot = int(np.nanargmax(previous))
    path = np.empty(n_nodes, np.float64)
    for node in range(n_nodes - 1, -1, -1):
        path[node] = offsets[node, state_slot]
        if node:
            state_slot = int(back[node, state_slot])
    anchor_md = float(record["md"][0] - record["distance"][0])
    interpolation_md = np.r_[anchor_md, node_md]
    interpolation_path = np.r_[0.0, path]
    return np.interp(record["md"], interpolation_md, interpolation_path).astype(np.float32)


def _oracle_sparse_v2(record):
    indices = record["candidate_indices"]
    truth_residual = record["node_truth_residual"]
    node_offsets = np.where(
        indices >= 0, STATE_OFFSETS[np.maximum(indices, 0)], np.nan
    ).astype(np.float64)
    distance = np.abs(node_offsets - truth_residual[:, None])
    distance[indices < 0] = np.inf
    selected = node_offsets[np.arange(len(indices)), np.argmin(distance, axis=1)]
    anchor_md = float(record["md"][0] - record["distance"][0])
    residual = np.interp(
        record["md"],
        np.r_[anchor_md, record["node_md"]],
        np.r_[0.0, selected],
    )
    return (record["strong"] + residual).astype(np.float32)


def _evaluate_records_v2(records, transition_lambda, prior_weight, dose, control="full"):
    truths = []
    bases = []
    predictions = []
    wells = []
    distances = []
    for record in records:
        active = record
        if control == "shuffled":
            shift = 31 + (int(str(record["well"])[:4], 16) % 61)
            shuffled_emission = np.roll(record["emission"], shift, axis=1)
            indices, scores = _make_union_v2(shuffled_emission)
            active = {
                **record,
                "candidate_indices": indices,
                "candidate_scores": scores,
            }
        residual = _decode_sparse_v2(active, transition_lambda, prior_weight)
        prediction = active["strong"] + float(dose) * residual
        truths.append(active["truth"])
        bases.append(active["strong"])
        predictions.append(prediction)
        wells.append(np.repeat(active["well"], len(prediction)))
        distances.append(active["distance"])
    return _metric_v2(
        np.concatenate(truths),
        np.concatenate(bases),
        np.concatenate(predictions),
        np.concatenate(wells),
        np.concatenate(distances),
    )


def _evaluate_oracle_v2(records):
    truths = []
    bases = []
    predictions = []
    wells = []
    distances = []
    for record in records:
        prediction = _oracle_sparse_v2(record)
        truths.append(record["truth"])
        bases.append(record["strong"])
        predictions.append(prediction)
        wells.append(np.repeat(record["well"], len(prediction)))
        distances.append(record["distance"])
    return _metric_v2(
        np.concatenate(truths),
        np.concatenate(bases),
        np.concatenate(predictions),
        np.concatenate(wells),
        np.concatenate(distances),
    )


def _build_fold_records_v2(fold):
    wells = folds_v2.loc[folds_v2["fold"].eq(fold), "well"].tolist()
    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=BUILD_WORKERS_V2) as executor:
        futures = {executor.submit(_build_record_v2, well): well for well in wells}
        for index, future in enumerate(as_completed(futures), 1):
            well = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append({"well": well, "fold": fold, "reason": repr(exc)})
            if index % 25 == 0 or index == len(wells):
                print(
                    f"build Fold{fold} sparse records {index}/{len(wells)}",
                    flush=True,
                )
    if len(records) < int(0.98 * len(wells)):
        raise RuntimeError(f"Fold{fold}: only {len(records)} / {len(wells)} records")
    return records, failures


started_v2 = time.time()
print(
    "V2 CONFIG",
    {
        "core_folds": CORE_FOLDS_V2,
        "tune_fold": TUNE_FOLD_V2,
        "test_fold": TEST_FOLD_V2,
        "node_stride": NODE_STRIDE_V2,
        "prior_states": PRIOR_STATES_V2,
        "gr_states": GR_STATES_V2,
        "transition_prior": transition_prior_v2,
    },
    flush=True,
)

tune_records_v2, tune_failures_v2 = _build_fold_records_v2(TUNE_FOLD_V2)
tune_oracle_v2, _ = _evaluate_oracle_v2(tune_records_v2)
print("FOLD1 UNION ORACLE", tune_oracle_v2, flush=True)

scan_rows_v2 = []
best_v2 = None
for transition_lambda_v2 in TRANSITION_LAMBDAS_V2:
    for prior_weight_v2 in PRIOR_WEIGHTS_V2:
        residual_cache_v2 = []
        truths_v2 = []
        bases_v2 = []
        wells_v2 = []
        distances_v2 = []
        for record_v2 in tune_records_v2:
            residual_cache_v2.append(
                _decode_sparse_v2(record_v2, transition_lambda_v2, prior_weight_v2)
            )
            truths_v2.append(record_v2["truth"])
            bases_v2.append(record_v2["strong"])
            wells_v2.append(np.repeat(record_v2["well"], len(record_v2["truth"])))
            distances_v2.append(record_v2["distance"])
        residual_v2 = np.concatenate(residual_cache_v2)
        truth_v2 = np.concatenate(truths_v2)
        base_v2 = np.concatenate(bases_v2)
        well_v2 = np.concatenate(wells_v2)
        distance_v2 = np.concatenate(distances_v2)
        for dose_v2 in DOSES_V2:
            metrics_v2, _ = _metric_v2(
                truth_v2,
                base_v2,
                base_v2 + float(dose_v2) * residual_v2,
                well_v2,
                distance_v2,
            )
            row_v2 = {
                "transition_lambda": transition_lambda_v2,
                "prior_weight": prior_weight_v2,
                "dose": dose_v2,
                **metrics_v2,
            }
            scan_rows_v2.append(row_v2)
            if best_v2 is None or row_v2["rmse"] < best_v2["rmse"]:
                best_v2 = row_v2
        del residual_cache_v2, residual_v2

scan_v2 = pd.DataFrame(scan_rows_v2).sort_values("rmse")
scan_v2.to_csv(OUT_DIR_V2 / "fold1_config_scan.csv", index=False)
print("\nFOLD1 CONFIG SCAN TOP")
print(scan_v2.head(20).to_string(index=False))

selected_v2 = {
    "transition_lambda": float(best_v2["transition_lambda"]),
    "prior_weight": float(best_v2["prior_weight"]),
    "dose": float(best_v2["dose"]),
}
del tune_records_v2
gc.collect()

test_records_v2, test_failures_v2 = _build_fold_records_v2(TEST_FOLD_V2)
test_oracle_v2, _ = _evaluate_oracle_v2(test_records_v2)
test_metrics_v2, test_wells_v2 = _evaluate_records_v2(
    test_records_v2, **selected_v2
)
test_wells_v2.to_csv(OUT_DIR_V2 / "fold2_well_metrics.csv", index=False)

# The control is intentionally expensive only once, after all parameters freeze.
shuffled_metrics_v2, _ = _evaluate_records_v2(
    test_records_v2, control="shuffled", **selected_v2
)
pd.DataFrame(tune_failures_v2 + test_failures_v2).to_csv(
    OUT_DIR_V2 / "failures.csv", index=False
)

print("\nFOLD2 FROZEN")
print(json.dumps(_jsonable_v2(test_metrics_v2), indent=2))
print("\nFOLD2 UNION ORACLE")
print(json.dumps(_jsonable_v2(test_oracle_v2), indent=2))
print("\nFOLD2 SHUFFLED CONTROL")
print(json.dumps(_jsonable_v2(shuffled_metrics_v2), indent=2))

gates_v2 = {
    "union_oracle_rmse_le": 5.5,
    "union_oracle_far_rmse_le": 6.5,
    "frozen_gain_ge": 0.30,
    "frozen_far_gain_ge": 0.75,
    "positive_well_fraction_ge": 0.55,
    "shuffle_degradation_ge": 0.30,
}
checks_v2 = {
    "union_oracle": bool(test_oracle_v2["rmse"] <= gates_v2["union_oracle_rmse_le"]),
    "union_oracle_far": bool(
        test_oracle_v2.get("far_gt2000_rmse", np.inf)
        <= gates_v2["union_oracle_far_rmse_le"]
    ),
    "frozen_gain": bool(test_metrics_v2["gain_vs_base"] >= gates_v2["frozen_gain_ge"]),
    "frozen_far_gain": bool(
        test_metrics_v2.get("far_gt2000_gain", -np.inf)
        >= gates_v2["frozen_far_gain_ge"]
    ),
    "positive_wells": bool(
        test_metrics_v2["positive_well_fraction"]
        >= gates_v2["positive_well_fraction_ge"]
    ),
    "shuffle": bool(
        shuffled_metrics_v2["rmse"] - test_metrics_v2["rmse"]
        >= gates_v2["shuffle_degradation_ge"]
    ),
}
strict_pass_v2 = bool(all(checks_v2.values()))
verdict_v2 = _jsonable_v2(
    {
        "experiment": EXPERIMENT_V2,
        "stage": "nested_sparse_strong_union_path_gateway",
        "sample_unit": "one complete physical well",
        "uses_neighbor": False,
        "uses_cp12_gold": False,
        "protocol": {
            "transition_train_folds": list(CORE_FOLDS_V2),
            "config_selection_fold": TUNE_FOLD_V2,
            "frozen_test_fold": TEST_FOLD_V2,
            "selected": selected_v2,
        },
        "candidate_union": {
            "prior_states": PRIOR_STATES_V2,
            "gr_states": GR_STATES_V2,
            "max_states_per_node": MAX_STATES_PER_NODE_V2,
            "node_stride": NODE_STRIDE_V2,
        },
        "transition_prior": transition_prior_v2,
        "fold1_oracle": tune_oracle_v2,
        "fold1_selected": best_v2,
        "fold2_oracle": test_oracle_v2,
        "fold2_frozen": test_metrics_v2,
        "fold2_shuffled": shuffled_metrics_v2,
        "shuffle_degradation": shuffled_metrics_v2["rmse"] - test_metrics_v2["rmse"],
        "gates": gates_v2,
        "checks": checks_v2,
        "strict_pass": strict_pass_v2,
        "elapsed_minutes": (time.time() - started_v2) / 60.0,
        "next": (
            "PASS -> replace fixed energies with a learned complete-well sparse-state decoder"
            if strict_pass_v2
            else "FAIL -> inspect oracle versus frozen split before any learned decoder"
        ),
    }
)
(OUT_DIR_V2 / "verdict.json").write_text(
    json.dumps(verdict_v2, indent=2), encoding="utf-8"
)
print("\nFINAL VERDICT")
print(json.dumps(verdict_v2, indent=2))
print("saved:", OUT_DIR_V2)
