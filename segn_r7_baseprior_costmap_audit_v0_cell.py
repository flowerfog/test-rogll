"""SEGN-R7: frozen dense evidence inside a fold-clean base cost-map prior.

R6 established that dense bilinear/cosine evidence uses the assigned typewell,
but is not accurate enough to decode from a uniform state prior.  The SEGN work
note describes matching channels as one component of a fused spatial/base cost
map, with every component normalised by candidate column.  This zero-training
audit tests that consumption position directly:

* freeze the R6 dense evidence checkpoint;
* rebase each chunk on a fold-clean SP45/FLE path available at deployment;
* turn that path into a Gaussian candidate-state prior;
* column-normalise the R6 evidence and add a scanned positive dose;
* select on 32 deterministic Fold-3 wells and confirm on the other 122;
* require matched shuffled-typewell evidence to remove the confirmed gain.

An existing fold-clean SP45/FLE path is intentionally used as the base prior;
CP12/Gold/neighbor/strong paths are not used.  Fold 1 and Fold 2 are not loaded.
Only compact scan tables and a verdict are written.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch


ROOT_R7 = Path("/root/autodl-tmp")
SOURCE_CANDIDATES_R7 = (
    ROOT_R7 / "segn_r6_densemd_evidence_fold3_v0_cell.py",
    Path.cwd() / "segn_r6_densemd_evidence_fold3_v0_cell.py",
)
SOURCE_PATH_R7 = next((path for path in SOURCE_CANDIDATES_R7 if path.exists()), None)
if SOURCE_PATH_R7 is None:
    raise FileNotFoundError(
        "Upload segn_r6_densemd_evidence_fold3_v0_cell.py to /root/autodl-tmp."
    )

source_r7 = SOURCE_PATH_R7.read_text(encoding="utf-8")
marker_r7 = "\nstarted_r6 = time.time()"
if marker_r7 not in source_r7:
    raise RuntimeError(f"Cannot find R6 execution marker in {SOURCE_PATH_R7}")
exec(compile(source_r7.split(marker_r7, 1)[0], str(SOURCE_PATH_R7), "exec"), globals())


EXPERIMENT_R7 = "segn_r7_baseprior_costmap_audit_v0"
OUT_DIR_R7 = OOF_ROOT / EXPERIMENT_R7
OUT_DIR_R7.mkdir(parents=True, exist_ok=True)
OUT_DIR = OUT_DIR_R7

CHECKPOINT_CANDIDATES_R7 = (
    OOF_ROOT / "segn_r6_densemd_evidence_fold3_v0" / "models" / "best_densemd_evidence.pt",
    ROOT_R7 / "best_densemd_evidence.pt",
)
CHECKPOINT_PATH_R7 = next(
    (path for path in CHECKPOINT_CANDIDATES_R7 if path.exists()), None
)
if CHECKPOINT_PATH_R7 is None:
    raise FileNotFoundError(
        "Cannot find the frozen R6 best_densemd_evidence.pt checkpoint."
    )

POOL_CANDIDATES_R7 = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
POOL_PATH_R7 = next((path for path in POOL_CANDIDATES_R7 if path.exists()), None)
if POOL_PATH_R7 is None:
    raise FileNotFoundError("Cannot find the fold-clean candidate pool CSV.")

TRAIN_FOLDS_R7 = (4, 5)
VALID_FOLD_R7 = 3
SELECTION_WELLS_R7 = 32
BASE_PREFERENCE_R7 = ("sp45", "sp45_raw", "fleongg")
PRIOR_SIGMAS_FT_R7 = (2.0, 4.0, 8.0, 16.0, 32.0)
EVIDENCE_DOSES_R7 = (0.0, 0.001, 0.003, 0.01, 0.03, 0.10, 0.30, 1.0)
TRUTH_TOLERANCE_FT_R7 = 0.02
MIN_BASE_COVERAGE_R7 = 0.995


def _json_default_r7(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json_r7(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default_r7), encoding="utf-8"
    )


def _trusted_torch_load_r7(path):
    # This checkpoint was written locally by R6. PyTorch 2.6 otherwise rejects
    # the NumPy metadata in the payload under its new weights_only default.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_base_paths_r7(records):
    header = pd.read_csv(POOL_PATH_R7, nrows=0).columns.tolist()
    truth_column = next(
        (column for column in ("tvt_true", "TVT", "target") if column in header),
        None,
    )
    bases = [column for column in BASE_PREFERENCE_R7 if column in header]
    if truth_column is None or "well" not in header or not bases:
        raise RuntimeError(f"Unexpected candidate-pool columns: {header}")

    wanted = {record.well for record in records}
    chunks = []
    for chunk in pd.read_csv(
        POOL_PATH_R7,
        usecols=["well", truth_column, *bases],
        dtype={"well": str},
        chunksize=500_000,
    ):
        selected = chunk[chunk["well"].isin(wanted)]
        if len(selected):
            chunks.append(selected.copy())
    if not chunks:
        raise RuntimeError("Candidate pool contains none of the requested Fold-3 wells")
    pool = pd.concat(chunks, ignore_index=True)
    grouped = {well: frame for well, frame in pool.groupby("well", sort=False)}

    output = {}
    truth_gaps = []
    finite_counts = {base: 0 for base in bases}
    total_counts = {base: 0 for base in bases}
    repaired_counts = {base: 0 for base in bases}
    for record in records:
        frame = grouped.get(record.well)
        expected_rows = len(record.eval_index)
        if frame is None or len(frame) != expected_rows:
            raise RuntimeError(
                f"{record.well}: pool rows {0 if frame is None else len(frame)} "
                f"!= eval rows {expected_rows}"
            )
        pool_truth = pd.to_numeric(frame[truth_column], errors="raise").to_numpy(
            np.float64
        )
        record_truth = record.truth[record.eval_index].astype(np.float64)
        gap = np.abs(pool_truth - record_truth)
        if not np.isfinite(gap).all() or float(gap.max()) > TRUTH_TOLERANCE_FT_R7:
            raise RuntimeError(f"{record.well}: truth alignment max gap={gap.max():.6f}")
        truth_gaps.append(float(gap.max()))
        output[record.well] = {}
        for base in bases:
            values = pd.to_numeric(frame[base], errors="coerce").to_numpy(np.float64)
            valid = np.isfinite(values)
            finite_counts[base] += int(valid.sum())
            total_counts[base] += len(values)
            if valid.mean() < MIN_BASE_COVERAGE_R7 or valid.sum() < 2:
                raise RuntimeError(
                    f"{record.well}/{base}: finite coverage={valid.mean():.6f}"
                )
            if not valid.all():
                positions = np.arange(len(values), dtype=np.float64)
                values[~valid] = np.interp(positions[~valid], positions[valid], values[valid])
                repaired_counts[base] += int((~valid).sum())
            output[record.well][base] = values

    coverage = {
        base: float(finite_counts[base] / max(total_counts[base], 1)) for base in bases
    }
    return output, bases, {
        "pool": str(POOL_PATH_R7),
        "rows": int(len(pool)),
        "wells": int(len(grouped)),
        "maximum_truth_gap": float(max(truth_gaps)),
        "finite_coverage": coverage,
        "interpolated_values": repaired_counts,
    }


def _base_rebased_payload_r7(record, start, base_eval, control, shuffled_record):
    payload = _dense_payload_r6(
        record,
        int(start),
        control=control,
        shuffled_typewell=shuffled_record,
    )
    eval_md = record.md[record.eval_index].astype(np.float64)
    start_row = int(record.node_rows[int(start)])
    start_md = float(record.md[start_row])
    base_anchor = float(np.interp(start_md, eval_md, base_eval))
    target_rows = payload["target_rows"]
    target_md = record.md[target_rows].astype(np.float64)
    base_target = np.interp(target_md, eval_md, base_eval)

    payload["candidate_positions"] = (
        base_anchor - float(record.anchor_tvt) + STATE_OFFSETS
    ).astype(np.float32)
    payload["deployed_anchor"] = base_anchor
    payload["target_state"] = (
        record.truth[target_rows].astype(np.float64) - base_anchor
    ).astype(np.float32)
    return payload, base_target, base_anchor


@torch.no_grad()
def _normalised_evidence_r7(model, payload, device):
    tensors = _tensor_payload(payload, device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits, auxiliary = model(*tensors, return_aux=True)
    logits = logits.float().cpu().numpy().astype(np.float64)
    valid = auxiliary["candidate_valid"].cpu().numpy().astype(bool)
    if valid.ndim != 1 or valid.shape[0] != logits.shape[0]:
        raise RuntimeError(f"Unexpected candidate-valid shape: {valid.shape}")
    usable = logits[valid]
    centre = np.mean(usable, axis=0, keepdims=True)
    scale = np.std(usable, axis=0, keepdims=True)
    scale = np.maximum(scale, 1.0e-3)
    normalised = (logits - centre) / scale
    normalised[~valid] = -1.0e6
    return normalised, valid


def _posterior_expected_r7(evidence, valid, base_centre, sigma, dose):
    states = STATE_OFFSETS.astype(np.float64)
    prior = -0.5 * ((states[:, None] - base_centre[None]) / sigma) ** 2
    score = prior + dose * evidence
    score[~valid] = -1.0e6
    score -= np.max(score, axis=0, keepdims=True)
    probability = np.exp(np.clip(score, -80.0, 0.0))
    probability[~valid] = 0.0
    probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1.0e-12)
    return np.sum(probability * states[:, None], axis=0)


def _empty_accumulator_r7(configs):
    return {
        "configs": configs,
        "sse": np.zeros(len(configs), np.float64),
        "sae": np.zeros(len(configs), np.float64),
        "rows": np.zeros(len(configs), np.int64),
        "local_sse": np.zeros(len(configs), np.float64),
        "local_rows": np.zeros(len(configs), np.int64),
        "strata_sse": np.zeros((len(configs), 4), np.float64),
        "strata_rows": np.zeros((len(configs), 4), np.int64),
        "well_rmse": [[] for _ in configs],
        "axis_inside": np.zeros(len(configs), np.int64),
        "axis_total": np.zeros(len(configs), np.int64),
    }


def _strata_masks_r7(distance):
    return (
        distance <= 500.0,
        (distance > 500.0) & (distance <= 1000.0),
        (distance > 1000.0) & (distance <= 2000.0),
        distance > 2000.0,
    )


@torch.no_grad()
def _evaluate_configs_r7(
    model,
    records,
    base_paths,
    configs,
    device,
    control="full",
):
    model.eval()
    accumulator = _empty_accumulator_r7(configs)
    by_base = {}
    config_indices = {}
    for index, config in enumerate(configs):
        config_indices.setdefault(config[0], []).append(index)

    shuffled_records = (
        list(records[1:]) + list(records[:1]) if len(records) > 1 else list(records)
    )
    for number, (record, shuffled_record) in enumerate(
        zip(records, shuffled_records), 1
    ):
        row_count = len(record.md)
        prediction_sum = np.zeros((len(configs), row_count), np.float64)
        prediction_count = np.zeros(row_count, np.float64)
        prediction_sum[:, record.anchor_index] = float(record.anchor_tvt)
        prediction_count[record.anchor_index] = 1.0

        for base_name, indices in config_indices.items():
            base_eval = base_paths[record.well][base_name]
            for raw_start in record.chunk_starts:
                payload, base_target, base_anchor = _base_rebased_payload_r7(
                    record,
                    int(raw_start),
                    base_eval,
                    control,
                    shuffled_record,
                )
                evidence, valid = _normalised_evidence_r7(model, payload, device)
                rows = payload["target_rows"]
                truth_state = payload["target_state"].astype(np.float64)
                base_centre = base_target - base_anchor
                for index in indices:
                    _, sigma, dose = configs[index]
                    expected = _posterior_expected_r7(
                        evidence, valid, base_centre, sigma, dose
                    )
                    prediction_sum[index, rows] += base_anchor + expected
                    local_error = expected - truth_state
                    accumulator["local_sse"][index] += float(
                        np.dot(local_error, local_error)
                    )
                    accumulator["local_rows"][index] += len(local_error)
                    accumulator["axis_inside"][index] += int(
                        np.sum(np.abs(truth_state) <= STATE_HALF_WIDTH_FT - 0.5)
                    )
                    accumulator["axis_total"][index] += len(truth_state)
            # Every base uses the same dense target rows, so count once after
            # the first base. The overlap count is independent of config.
            if base_name == next(iter(config_indices)):
                for raw_start in record.chunk_starts:
                    rows = _dense_rows_r6(record, int(raw_start))
                    prediction_count[rows] += 1.0

        supported = prediction_count > 0
        support_md = record.md[supported].astype(np.float64)
        eval_rows = record.eval_index
        query_md = record.md[eval_rows].astype(np.float64)
        truth = record.truth[eval_rows].astype(np.float64)
        distance = query_md - float(record.md[record.anchor_index])
        masks = _strata_masks_r7(distance)
        for index in range(len(configs)):
            values = prediction_sum[index, supported] / prediction_count[supported]
            prediction = np.interp(query_md, support_md, values)
            error = prediction - truth
            squared = error * error
            accumulator["sse"][index] += float(squared.sum())
            accumulator["sae"][index] += float(np.abs(error).sum())
            accumulator["rows"][index] += len(error)
            accumulator["well_rmse"][index].append(float(np.sqrt(squared.mean())))
            for stratum, mask in enumerate(masks):
                if int(mask.sum()):
                    accumulator["strata_sse"][index, stratum] += float(
                        squared[mask].sum()
                    )
                    accumulator["strata_rows"][index, stratum] += int(mask.sum())

        for base_name in config_indices:
            if base_name not in by_base:
                by_base[base_name] = {
                    "sse": 0.0,
                    "sae": 0.0,
                    "rows": 0,
                    "well_rmse": [],
                    "strata_sse": np.zeros(4, np.float64),
                    "strata_rows": np.zeros(4, np.int64),
                }
            base = base_paths[record.well][base_name]
            error = base - truth
            squared = error * error
            item = by_base[base_name]
            item["sse"] += float(squared.sum())
            item["sae"] += float(np.abs(error).sum())
            item["rows"] += len(error)
            item["well_rmse"].append(float(np.sqrt(squared.mean())))
            for stratum, mask in enumerate(masks):
                if int(mask.sum()):
                    item["strata_sse"][stratum] += float(squared[mask].sum())
                    item["strata_rows"][stratum] += int(mask.sum())

        if number % 8 == 0 or number == len(records):
            print(
                f"SEGN-R7 evaluate {control} {number}/{len(records)} "
                f"configs={len(configs)}",
                flush=True,
            )

    return accumulator, by_base


def _metric_row_r7(accumulator, index):
    rows = int(accumulator["rows"][index])
    well_rmse = np.asarray(accumulator["well_rmse"][index], np.float64)
    result = {
        "base": accumulator["configs"][index][0],
        "prior_sigma_ft": float(accumulator["configs"][index][1]),
        "evidence_dose": float(accumulator["configs"][index][2]),
        "rows": rows,
        "wells": int(len(well_rmse)),
        "exact_sse": float(accumulator["sse"][index]),
        "rmse": float(np.sqrt(accumulator["sse"][index] / max(rows, 1))),
        "mae": float(accumulator["sae"][index] / max(rows, 1)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
        "local_expected_rmse": float(
            np.sqrt(
                accumulator["local_sse"][index]
                / max(accumulator["local_rows"][index], 1)
            )
        ),
        "true_state_axis_coverage": float(
            accumulator["axis_inside"][index]
            / max(accumulator["axis_total"][index], 1)
        ),
    }
    names = (
        "near_0_500_rmse",
        "mid_500_1000_rmse",
        "mid_1000_2000_rmse",
        "far_gt2000_rmse",
    )
    for stratum, name in enumerate(names):
        result[name] = float(
            np.sqrt(
                accumulator["strata_sse"][index, stratum]
                / max(accumulator["strata_rows"][index, stratum], 1)
            )
        )
    return result


def _base_metric_r7(item, name):
    rows = int(item["rows"])
    well_rmse = np.asarray(item["well_rmse"], np.float64)
    result = {
        "base": name,
        "rows": rows,
        "wells": int(len(well_rmse)),
        "exact_sse": float(item["sse"]),
        "rmse": float(np.sqrt(item["sse"] / max(rows, 1))),
        "mae": float(item["sae"] / max(rows, 1)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }
    names = (
        "near_0_500_rmse",
        "mid_500_1000_rmse",
        "mid_1000_2000_rmse",
        "far_gt2000_rmse",
    )
    for stratum, metric_name in enumerate(names):
        result[metric_name] = float(
            np.sqrt(item["strata_sse"][stratum] / max(item["strata_rows"][stratum], 1))
        )
    return result


started_r7 = time.time()
_seed_everything(SEED + 7001)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SEGN-R7")
device_r7 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds_r7, fold_path_r7 = _load_folds()
fold3_wells_r7 = folds_r7.loc[folds_r7["fold"].eq(VALID_FOLD_R7), "well"].tolist()
if len(fold3_wells_r7) != 154:
    raise RuntimeError(f"Unexpected Fold-3 size: {len(fold3_wells_r7)}")
fold3_records_r7 = _load_records(fold3_wells_r7, "r7_fold3")
if len(fold3_records_r7) != 154:
    raise RuntimeError(f"Expected 154 Fold-3 records, got {len(fold3_records_r7)}")

order_r7 = np.argsort([record.eval_rows for record in fold3_records_r7])
selection_positions_r7 = np.unique(
    np.rint(np.linspace(0, len(order_r7) - 1, SELECTION_WELLS_R7)).astype(int)
)
selection_indices_r7 = set(order_r7[selection_positions_r7].tolist())
selection_records_r7 = [
    record for index, record in enumerate(fold3_records_r7) if index in selection_indices_r7
]
confirmation_records_r7 = [
    record for index, record in enumerate(fold3_records_r7) if index not in selection_indices_r7
]

base_paths_r7, base_names_r7, alignment_r7 = _load_base_paths_r7(fold3_records_r7)
checkpoint_payload_r7 = _trusted_torch_load_r7(CHECKPOINT_PATH_R7)
state_r7 = (
    checkpoint_payload_r7["model"]
    if isinstance(checkpoint_payload_r7, dict) and "model" in checkpoint_payload_r7
    else checkpoint_payload_r7
)
model_r7 = AxisAwareEvidenceR5().to(device_r7)
model_r7.load_state_dict(state_r7)
model_r7.eval()

configs_r7 = [
    (base, sigma, dose)
    for base in base_names_r7
    for sigma in PRIOR_SIGMAS_FT_R7
    for dose in EVIDENCE_DOSES_R7
]
guards_r7 = {
    "source": str(SOURCE_PATH_R7),
    "checkpoint": str(CHECKPOINT_PATH_R7),
    "checkpoint_epoch": (
        int(checkpoint_payload_r7.get("epoch", -1))
        if isinstance(checkpoint_payload_r7, dict)
        else -1
    ),
    "fold_source": str(fold_path_r7),
    "selection_fold": VALID_FOLD_R7,
    "selection_wells": len(selection_records_r7),
    "confirmation_wells": len(confirmation_records_r7),
    "fold2_touched": False,
    "fold1_touched": False,
    "candidate_pool_alignment": alignment_r7,
    "bases": base_names_r7,
    "configs": len(configs_r7),
    "free_gb_before": float(
        __import__("shutil").disk_usage(ROOT_R7).free / (1024 ** 3)
    ),
}
print("SEGN-R7 GUARDS", json.dumps(guards_r7, indent=2), flush=True)

selection_accumulator_r7, selection_bases_r7 = _evaluate_configs_r7(
    model_r7,
    selection_records_r7,
    base_paths_r7,
    configs_r7,
    device_r7,
    control="full",
)
selection_rows_r7 = [
    _metric_row_r7(selection_accumulator_r7, index)
    for index in range(len(configs_r7))
]
selection_frame_r7 = pd.DataFrame(selection_rows_r7).sort_values(
    ["rmse", "base", "prior_sigma_ft", "evidence_dose"]
)
selection_frame_r7.to_csv(OUT_DIR_R7 / "selection_scan.csv", index=False)
print("SEGN-R7 SELECTION TOP")
print(selection_frame_r7.head(30).to_string(index=False), flush=True)

positive_frame_r7 = selection_frame_r7[selection_frame_r7["evidence_dose"] > 0.0]
if not len(positive_frame_r7):
    raise RuntimeError("SEGN-R7 has no positive-dose configuration")
winner_r7 = positive_frame_r7.iloc[0].to_dict()
winner_config_r7 = (
    str(winner_r7["base"]),
    float(winner_r7["prior_sigma_ft"]),
    float(winner_r7["evidence_dose"]),
)
prior_config_r7 = (winner_config_r7[0], winner_config_r7[1], 0.0)

confirmation_configs_r7 = [prior_config_r7, winner_config_r7]
confirmation_accumulator_r7, confirmation_bases_r7 = _evaluate_configs_r7(
    model_r7,
    confirmation_records_r7,
    base_paths_r7,
    confirmation_configs_r7,
    device_r7,
    control="full",
)
confirmation_prior_r7 = _metric_row_r7(confirmation_accumulator_r7, 0)
confirmation_evidence_r7 = _metric_row_r7(confirmation_accumulator_r7, 1)
confirmation_base_r7 = _base_metric_r7(
    confirmation_bases_r7[winner_config_r7[0]], winner_config_r7[0]
)

shuffled_accumulator_r7, _ = _evaluate_configs_r7(
    model_r7,
    confirmation_records_r7,
    base_paths_r7,
    [winner_config_r7],
    device_r7,
    control="shuffled_typewell",
)
confirmation_shuffled_r7 = _metric_row_r7(shuffled_accumulator_r7, 0)

selection_prior_match_r7 = selection_frame_r7[
    (selection_frame_r7["base"] == winner_config_r7[0])
    & (selection_frame_r7["prior_sigma_ft"] == winner_config_r7[1])
    & (selection_frame_r7["evidence_dose"] == 0.0)
].iloc[0].to_dict()

confirmation_gain_r7 = confirmation_prior_r7["rmse"] - confirmation_evidence_r7["rmse"]
confirmation_far_gain_r7 = (
    confirmation_prior_r7["far_gt2000_rmse"]
    - confirmation_evidence_r7["far_gt2000_rmse"]
)
shuffle_degradation_r7 = (
    confirmation_shuffled_r7["rmse"] - confirmation_evidence_r7["rmse"]
)
shuffled_gain_r7 = (
    confirmation_prior_r7["rmse"] - confirmation_shuffled_r7["rmse"]
)
gain_retained_r7 = (
    shuffled_gain_r7 / confirmation_gain_r7
    if confirmation_gain_r7 > 1.0e-9
    else math.inf
)

checks_r7 = {
    "positive_evidence_dose_selected": winner_config_r7[2] > 0.0,
    "selection_gain_vs_matched_prior_ge_0p10": (
        float(selection_prior_match_r7["rmse"]) - float(winner_r7["rmse"])
    )
    >= 0.10,
    "confirmation_gain_vs_matched_prior_ge_0p20": confirmation_gain_r7 >= 0.20,
    "confirmation_far_gain_ge_0p20": confirmation_far_gain_r7 >= 0.20,
    "shuffled_degradation_ge_0p15": shuffle_degradation_r7 >= 0.15,
    "shuffled_retains_le_30pct_gain": gain_retained_r7 <= 0.30,
    "true_state_axis_coverage_ge_0p99": confirmation_evidence_r7[
        "true_state_axis_coverage"
    ]
    >= 0.99,
}
strict_pass_r7 = bool(all(checks_r7.values()))

comparison_r7 = pd.DataFrame(
    [
        {"variant": "raw_base", **confirmation_base_r7},
        {"variant": "matched_prior_only", **confirmation_prior_r7},
        {"variant": "base_prior_plus_evidence", **confirmation_evidence_r7},
        {"variant": "shuffled_typewell_evidence", **confirmation_shuffled_r7},
    ]
)
comparison_r7.to_csv(OUT_DIR_R7 / "confirmation_comparison.csv", index=False)
print("SEGN-R7 CONFIRMATION")
print(comparison_r7.to_string(index=False), flush=True)

verdict_r7 = {
    "experiment": EXPERIMENT_R7,
    "stage": "fold3_frozen_dense_evidence_inside_foldclean_base_prior",
    "sample_unit": "base-rebased dense chunks nested within complete wells",
    "deployable_inputs": True,
    "optimizer_steps": 0,
    "uses_sp45_or_fleongg_base": True,
    "uses_cp12_gold_neighbor_strong": False,
    "worknote_mapping": {
        "matching_component": "frozen R6 bilinear+cosine logits",
        "base_component": winner_config_r7[0],
        "component_normalisation": "per-column candidate-state z-score",
        "fusion_location": "candidate cost map before posterior readout",
        "readout": "posterior mean",
    },
    "protocol": {
        "selection_fold": VALID_FOLD_R7,
        "selection_wells": len(selection_records_r7),
        "confirmation_wells": len(confirmation_records_r7),
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "guards": guards_r7,
    "selected": winner_r7,
    "selection_matched_prior": selection_prior_match_r7,
    "confirmation": {
        "raw_base": confirmation_base_r7,
        "matched_prior_only": confirmation_prior_r7,
        "base_prior_plus_evidence": confirmation_evidence_r7,
        "shuffled_typewell_evidence": confirmation_shuffled_r7,
        "gain_vs_matched_prior": confirmation_gain_r7,
        "far_gain_vs_matched_prior": confirmation_far_gain_r7,
        "shuffle_degradation": shuffle_degradation_r7,
        "shuffled_gain_retained_fraction": gain_retained_r7,
    },
    "checks": checks_r7,
    "strict_pass": strict_pass_r7,
    "elapsed_minutes": (time.time() - started_r7) / 60.0,
    "free_gb_after": float(
        __import__("shutil").disk_usage(ROOT_R7).free / (1024 ** 3)
    ),
    "next": (
        "PASS -> train only a soft source-weight gate on matched OOF cost-map features"
        if strict_pass_r7
        else "FAIL -> R6 matching evidence adds nothing inside the correct base-prior consumption position; close this evidence encoder"
    ),
}
_write_json_r7(OUT_DIR_R7 / "verdict.json", verdict_r7)
print("SEGN-R7 FINAL VERDICT", json.dumps(verdict_r7, indent=2), flush=True)
print("saved:", OUT_DIR_R7, flush=True)

if not strict_pass_r7:
    raise RuntimeError(json.dumps(verdict_r7, indent=2))
