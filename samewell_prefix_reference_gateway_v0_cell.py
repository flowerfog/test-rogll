"""Fold-honest audit of the visible heel GR as a same-well reference log.

This is a zero-learning observability gateway.  It compares absolute TVT
emissions from the assigned typewell with emissions obtained by matching the
hidden horizontal GR back to visible-prefix horizontal GR windows whose TVT is
known.  Folds 3/4/5 select one fusion weight; Fold 2 is evaluated once.  Fold 1
is never touched.

Prerequisite
------------
Run ``native_resolution_dualref_capacity_v0_cell.py`` once to create the legal
record cache.  No live-kernel globals are required.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT = "samewell_prefix_reference_gateway_v0"
ROOT = Path("/root/autodl-tmp")
DATA_ROOT_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    Path.cwd(),
)
DATA_ROOT = next(
    (path for path in DATA_ROOT_CANDIDATES if (path / "train").exists()),
    DATA_ROOT_CANDIDATES[0],
)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").exists() else DATA_ROOT / "oof_exports"
RECORD_DIR_CANDIDATES = (
    OOF_ROOT / "native_resolution_dualref_capacity_v0" / "records",
    OOF_ROOT / "native_resolution_dualref_onefold_v0" / "records",
)
RECORD_DIRS = tuple(path for path in RECORD_DIR_CANDIDATES if path.exists())
RECORD_DIR = (
    max(RECORD_DIRS, key=lambda path: sum(1 for _ in path.glob("*.npz")))
    if RECORD_DIRS
    else RECORD_DIR_CANDIDATES[0]
)
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (3, 4, 5)
VALID_FOLD = 2
SEED = 20260720

STATE_HALF_WIDTH_FT = 128
STATE_OFFSETS = np.arange(-STATE_HALF_WIDTH_FT, STATE_HALF_WIDTH_FT + 1, dtype=np.float32)
N_STATES = len(STATE_OFFSETS)

QUERY_HALF_WIDTHS_MD = np.asarray([64.0, 128.0, 256.0, 512.0, 1024.0], np.float32)
TYPEWELL_HALF_WIDTHS_TVT = np.asarray([4.0, 8.0, 16.0, 32.0, 64.0], np.float32)
HEEL_HALF_WIDTHS_MD = QUERY_HALF_WIDTHS_MD.copy()
WINDOW_POINTS = 65
WINDOW_UNIT = np.linspace(-1.0, 1.0, WINDOW_POINTS, dtype=np.float32)

TRAIN_QUERY_ROWS = 96
VALID_QUERY_ROWS = 192
HEEL_CENTRES = 192
HEEL_LOOKBACK_FT = 2400.0
HEEL_STATE_SIGMA_FT = 4.0
MIN_WINDOW_COVERAGE = 0.70
SOFTMAX_TEMPERATURE = 0.35
FUSION_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0, 1.50)

TOP8_GAIN_GATE = 0.05
FAR_RMSE_GAIN_GATE = 1.0
SHUFFLED_TOP8_DROP_GATE = 0.05


# ---------------------------------------------------------------------------
# Data and utilities
# ---------------------------------------------------------------------------

@dataclass
class Record:
    well: str
    horizontal: np.ndarray
    typewell: np.ndarray
    md: np.ndarray
    eval_index: np.ndarray
    truth: np.ndarray
    anchor_tvt: float
    anchor_index: int
    eval_rows: int


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _load_record(well):
    path = next(
        (directory / f"{well}.npz" for directory in RECORD_DIRS if (directory / f"{well}.npz").exists()),
        None,
    )
    if path is None:
        searched = [str(directory / f"{well}.npz") for directory in RECORD_DIR_CANDIDATES]
        raise FileNotFoundError(
            f"Missing record for {well}. Searched: {searched}. The capacity audit "
            "builds only 24 records; run build_native_dualref_all_records_v0_cell.py once."
        )
    with np.load(path, allow_pickle=False) as data:
        return Record(
            well=str(well),
            horizontal=data["horizontal"].astype(np.float32)[:2],
            typewell=data["typewell"].astype(np.float32)[:2],
            md=data["md"].astype(np.float32),
            eval_index=data["eval_index"].astype(np.int32),
            truth=data["truth"].astype(np.float32),
            anchor_tvt=float(data["anchor_tvt"][0]),
            anchor_index=int(data["anchor_index"][0]),
            eval_rows=int(data["eval_rows"][0]),
        )


def _load_folds():
    fold_path = next((path for path in FOLD_CANDIDATES if path.exists()), None)
    if fold_path is None:
        raise FileNotFoundError("Cannot find fold-clean well_metadata.csv")
    frame = pd.read_csv(fold_path, dtype=str)
    well_col = next((col for col in ("well", "well_id", "WELL") if col in frame), None)
    fold_col = next((col for col in ("fold", "outer_fold", "Fold") if col in frame), None)
    if well_col is None or fold_col is None:
        raise RuntimeError(f"Unexpected fold columns: {frame.columns.tolist()}")
    frame = frame[[well_col, fold_col]].rename(columns={well_col: "well", fold_col: "fold"})
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.drop_duplicates("well", keep="last")
    if sorted(frame["fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Unexpected folds: {sorted(frame['fold'].unique().tolist())}")
    return frame.sort_values(["fold", "well"]).reset_index(drop=True), fold_path


def _select_even(indices, count):
    indices = np.asarray(indices, np.int64)
    if len(indices) <= count:
        return indices
    positions = np.rint(np.linspace(0, len(indices) - 1, count)).astype(np.int64)
    return np.unique(indices[positions])


def _window_bank(axis, values, valid, centres, half_widths):
    axis = np.asarray(axis, np.float64)
    values = np.asarray(values, np.float64)
    valid = np.asarray(valid, np.float64)
    centres = np.asarray(centres, np.float64)
    half_widths = np.asarray(half_widths, np.float64)
    query = centres[:, None, None] + half_widths[None, :, None] * WINDOW_UNIT[None, None]
    flat = query.reshape(-1)
    sampled = np.interp(flat, axis, values).reshape(query.shape)
    sampled_valid = np.interp(flat, axis, valid).reshape(query.shape)
    mask = (query >= axis[0]) & (query <= axis[-1]) & (sampled_valid >= 0.5)
    return sampled.astype(np.float32), mask


def _normalised_embeddings(values, mask, device):
    values = torch.from_numpy(np.asarray(values, np.float32)).to(device)
    mask = torch.from_numpy(np.asarray(mask, bool)).to(device)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (values * mask).sum(dim=-1, keepdim=True) / count
    centred = (values - mean) * mask
    norm = torch.sqrt((centred * centred).sum(dim=-1, keepdim=True)).clamp_min(1e-5)
    embedding = centred / norm
    coverage = count[..., 0].float() / float(values.shape[-1])
    return embedding, coverage


def _row_standardise(score):
    median = score.median(dim=1, keepdim=True).values
    centred = score - median
    scale = torch.quantile(centred.abs(), 0.75, dim=1, keepdim=True).clamp_min(0.05)
    return centred / scale


def _heel_payload(record, device):
    anchor_md = float(record.md[record.anchor_index])
    prefix_rows = np.arange(record.anchor_index + 1, dtype=np.int64)
    prefix_rows = prefix_rows[record.md[prefix_rows] >= anchor_md - HEEL_LOOKBACK_FT]
    prefix_rows = prefix_rows[np.isfinite(record.truth[prefix_rows])]
    prefix_rows = _select_even(prefix_rows, HEEL_CENTRES)
    if len(prefix_rows) < 32:
        raise RuntimeError(f"{record.well}: only {len(prefix_rows)} legal heel centres")
    values, mask = _window_bank(
        record.md,
        record.horizontal[0],
        record.horizontal[1],
        record.md[prefix_rows],
        HEEL_HALF_WIDTHS_MD,
    )
    embedding, coverage = _normalised_embeddings(values, mask, device)
    offsets = torch.from_numpy(
        (record.truth[prefix_rows] - record.anchor_tvt).astype(np.float32)
    ).to(device)
    return embedding, coverage, offsets, prefix_rows


def _typewell_payload(record, device):
    length = record.typewell.shape[1]
    centre = length // 2
    axis = np.arange(length, dtype=np.float32) - float(centre)
    values, mask = _window_bank(
        axis,
        record.typewell[0],
        record.typewell[1],
        STATE_OFFSETS,
        TYPEWELL_HALF_WIDTHS_TVT,
    )
    return _normalised_embeddings(values, mask, device)


def _query_payload(record, rows, device):
    values, mask = _window_bank(
        record.md,
        record.horizontal[0],
        record.horizontal[1],
        record.md[rows],
        QUERY_HALF_WIDTHS_MD,
    )
    return _normalised_embeddings(values, mask, device)


@torch.no_grad()
def _emissions(record, rows, device, heel_source=None):
    query, query_coverage = _query_payload(record, rows, device)
    typewell, type_coverage = _typewell_payload(record, device)
    heel_record = record if heel_source is None else heel_source
    heel, heel_coverage, heel_offsets, _ = _heel_payload(heel_record, device)

    type_pair = torch.einsum("qsp,trp->qstr", query, typewell)
    type_valid = (
        (query_coverage[:, :, None, None] >= MIN_WINDOW_COVERAGE)
        & (type_coverage[None, None, :, :] >= MIN_WINDOW_COVERAGE)
    )
    type_pair = type_pair.masked_fill(~type_valid, -4.0)
    type_score = type_pair.amax(dim=(1, 3))

    heel_pair = torch.einsum("qsp,hrp->qshr", query, heel)
    heel_valid = (
        (query_coverage[:, :, None, None] >= MIN_WINDOW_COVERAGE)
        & (heel_coverage[None, None, :, :] >= MIN_WINDOW_COVERAGE)
    )
    heel_pair = heel_pair.masked_fill(~heel_valid, -4.0)
    heel_centre_score = heel_pair.amax(dim=(1, 3))
    state_offsets = torch.from_numpy(STATE_OFFSETS).to(device)
    state_penalty = 0.5 * (
        (state_offsets[None, :] - heel_offsets[:, None]) / HEEL_STATE_SIGMA_FT
    ) ** 2
    heel_score = (heel_centre_score[:, :, None] - state_penalty[None]).amax(dim=1)
    return (
        _row_standardise(type_score).cpu().numpy().astype(np.float32),
        _row_standardise(heel_score).cpu().numpy().astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _decode(score):
    score = np.asarray(score, np.float64)
    argmax = STATE_OFFSETS[np.argmax(score, axis=1)]
    shifted = (score - score.max(axis=1, keepdims=True)) / SOFTMAX_TEMPERATURE
    posterior = np.exp(np.clip(shifted, -60.0, 0.0))
    posterior /= posterior.sum(axis=1, keepdims=True).clip(1e-12)
    mean = posterior @ STATE_OFFSETS.astype(np.float64)
    return argmax.astype(np.float64), mean.astype(np.float64)


def _retrieval_rows(well, distance, truth_offset, score, variant):
    order = np.argsort(-score, axis=1, kind="mergesort")
    nearest = np.abs(STATE_OFFSETS[None] - truth_offset[:, None]).argmin(axis=1)
    rank = np.argmax(order == nearest[:, None], axis=1) + 1
    top1 = STATE_OFFSETS[order[:, 0]]
    top8 = STATE_OFFSETS[order[:, :8]]
    top8_hit = (np.abs(top8 - truth_offset[:, None]) <= 5.0).any(axis=1)
    truth_score = score[np.arange(len(score)), nearest]
    alias_scores = []
    for shift in (-24, 24):
        alias_index = np.clip(nearest + shift, 0, N_STATES - 1)
        alias_scores.append(score[np.arange(len(score)), alias_index])
    alias_win = truth_score > np.maximum(alias_scores[0], alias_scores[1])
    return pd.DataFrame(
        {
            "well": well,
            "variant": variant,
            "distance_ft": distance,
            "truth_offset": truth_offset,
            "top1_offset": top1,
            "top1_error": top1 - truth_offset,
            "top8_hit": top8_hit,
            "truth_rank": rank,
            "alias24_win": alias_win,
        }
    )


def _path_row(record, rows, score, variant):
    argmax, posterior = _decode(score)
    eval_rows = record.eval_index
    node_md = record.md[rows].astype(np.float64)
    eval_md = record.md[eval_rows].astype(np.float64)
    truth = record.truth[eval_rows].astype(np.float64)
    prediction = record.anchor_tvt + np.interp(eval_md, node_md, posterior)
    argmax_prediction = record.anchor_tvt + np.interp(eval_md, node_md, argmax)
    distance = eval_md - float(record.md[record.anchor_index])
    error = prediction - truth
    argmax_error = argmax_prediction - truth
    row = {
        "well": record.well,
        "variant": variant,
        "rows": len(eval_rows),
        "sse": float(np.sum(error * error)),
        "argmax_sse": float(np.sum(argmax_error * argmax_error)),
        "mae_sum": float(np.sum(np.abs(error))),
    }
    strata = (
        ("near_0_500", distance <= 500),
        ("mid_500_1000", (distance > 500) & (distance <= 1000)),
        ("mid_1000_2000", (distance > 1000) & (distance <= 2000)),
        ("far_gt2000", distance > 2000),
    )
    for name, mask in strata:
        row[f"{name}_rows"] = int(mask.sum())
        row[f"{name}_sse"] = float(np.sum(error[mask] ** 2))
    return row


def _summarise_path(frame):
    rows = int(frame["rows"].sum())
    report = {
        "rows": rows,
        "wells": int(frame["well"].nunique()),
        "rmse": float(math.sqrt(frame["sse"].sum() / rows)),
        "argmax_rmse": float(math.sqrt(frame["argmax_sse"].sum() / rows)),
        "mae": float(frame["mae_sum"].sum() / rows),
    }
    for name in ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000"):
        count = int(frame[f"{name}_rows"].sum())
        report[f"{name}_rmse"] = (
            float(math.sqrt(frame[f"{name}_sse"].sum() / count)) if count else float("nan")
        )
    return report


def _summarise_retrieval(frame):
    return {
        "sampled_rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "top1_within5": float((frame["top1_error"].abs() <= 5.0).mean()),
        "top8_within5": float(frame["top8_hit"].mean()),
        "median_truth_rank": float(frame["truth_rank"].median()),
        "p90_truth_rank": float(frame["truth_rank"].quantile(0.90)),
        "alias24_win_rate": float(frame["alias24_win"].mean()),
        "deep_top8_within5": float(frame.loc[frame["distance_ft"] > 2000, "top8_hit"].mean()),
    }


# ---------------------------------------------------------------------------
# Fold 3/4/5 weight selection and untouched Fold 2 evaluation
# ---------------------------------------------------------------------------

folds, fold_path = _load_folds()
train_wells = folds.loc[folds["fold"].isin(TRAIN_FOLDS), "well"].tolist()
valid_wells = folds.loc[folds["fold"].eq(VALID_FOLD), "well"].tolist()
if len(train_wells) not in (463, 464) or len(valid_wells) != 155:
    raise RuntimeError(f"Unexpected fold sizes: {len(train_wells)}/{len(valid_wells)}")
if not RECORD_DIR.exists():
    raise FileNotFoundError(RECORD_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA is required for the multiscale correlation audit")

started = time.time()
train_sse = {weight: 0.0 for weight in FUSION_WEIGHTS}
train_rows_total = 0
train_reports = []
for number, well in enumerate(train_wells, start=1):
    record = _load_record(well)
    rows = _select_even(record.eval_index, TRAIN_QUERY_ROWS)
    type_score, heel_score = _emissions(record, rows, device)
    truth_offset = record.truth[rows].astype(np.float64) - record.anchor_tvt
    for weight in FUSION_WEIGHTS:
        argmax, _ = _decode(type_score + weight * heel_score)
        train_sse[weight] += float(np.sum((argmax - truth_offset) ** 2))
    train_rows_total += len(rows)
    train_reports.append(
        {
            "well": well,
            "sampled_rows": len(rows),
            "heel_centres": len(_heel_payload(record, device)[3]),
        }
    )
    if number % 50 == 0 or number == len(train_wells):
        print(f"heel weight selection {number}/{len(train_wells)}", flush=True)

weight_scan = pd.DataFrame(
    [
        {
            "heel_weight": weight,
            "sampled_rows": train_rows_total,
            "argmax_rmse": math.sqrt(train_sse[weight] / train_rows_total),
        }
        for weight in FUSION_WEIGHTS
    ]
).sort_values(["argmax_rmse", "heel_weight"])
selected_weight = float(weight_scan.iloc[0]["heel_weight"])
weight_scan.to_csv(OUT_DIR / "fold345_weight_scan.csv", index=False)
pd.DataFrame(train_reports).to_csv(OUT_DIR / "fold345_build_report.csv", index=False)
print("FOLD345 WEIGHT SCAN")
print(weight_scan.to_string(index=False), flush=True)

retrieval_frames = []
path_rows = []
donor_wells = valid_wells[1:] + valid_wells[:1]
for number, (well, donor_well) in enumerate(zip(valid_wells, donor_wells), start=1):
    record = _load_record(well)
    donor = _load_record(donor_well)
    rows = _select_even(record.eval_index, VALID_QUERY_ROWS)
    type_score, heel_score = _emissions(record, rows, device)
    _, shuffled_heel_score = _emissions(record, rows, device, heel_source=donor)
    scores = {
        "typewell_only": type_score,
        "heel_only": heel_score,
        "combined": type_score + selected_weight * heel_score,
        "shuffled_heel": type_score + selected_weight * shuffled_heel_score,
    }
    distance = record.md[rows].astype(np.float64) - float(record.md[record.anchor_index])
    truth_offset = record.truth[rows].astype(np.float64) - record.anchor_tvt
    for variant, score in scores.items():
        retrieval_frames.append(
            _retrieval_rows(well, distance, truth_offset, score, variant)
        )
        path_rows.append(_path_row(record, rows, score, variant))
    if number % 25 == 0 or number == len(valid_wells):
        print(f"Fold2 same-well audit {number}/{len(valid_wells)}", flush=True)

retrieval = pd.concat(retrieval_frames, ignore_index=True)
paths = pd.DataFrame(path_rows)
retrieval.to_csv(OUT_DIR / "fold2_sampled_retrieval_rows.csv", index=False)
paths.to_csv(OUT_DIR / "fold2_path_bywell.csv", index=False)

reports = {}
for variant in ("typewell_only", "heel_only", "combined", "shuffled_heel"):
    retrieval_report = _summarise_retrieval(retrieval[retrieval["variant"].eq(variant)])
    path_report = _summarise_path(paths[paths["variant"].eq(variant)])
    reports[variant] = {**path_report, **retrieval_report}

comparison = pd.DataFrame(
    [{"variant": variant, **report} for variant, report in reports.items()]
)
comparison.to_csv(OUT_DIR / "fold2_comparison.csv", index=False)
print("FOLD2 COMPARISON")
print(comparison.to_string(index=False), flush=True)

top8_gain = reports["combined"]["top8_within5"] - reports["typewell_only"]["top8_within5"]
far_gain = reports["typewell_only"]["far_gt2000_rmse"] - reports["combined"]["far_gt2000_rmse"]
shuffle_drop = reports["combined"]["top8_within5"] - reports["shuffled_heel"]["top8_within5"]
checks = {
    "top8_gain": bool(top8_gain >= TOP8_GAIN_GATE),
    "far_rmse_gain": bool(far_gain >= FAR_RMSE_GAIN_GATE),
    "shuffled_control": bool(shuffle_drop >= SHUFFLED_TOP8_DROP_GATE),
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_samewell_visible_heel_reference_gateway",
    "sample_unit": "complete physical well; sampled stations nested within well",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_strong_cp12_gold": False,
    "protocol": {
        "weight_selection_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
        "selected_heel_weight": selected_weight,
        "heel_state_sigma_ft": HEEL_STATE_SIGMA_FT,
    },
    "fold_source": str(fold_path),
    "record_source": str(RECORD_DIR),
    "reports": reports,
    "diagnostics": {
        "combined_top8_gain_vs_typewell": top8_gain,
        "combined_far_rmse_gain_vs_typewell": far_gain,
        "combined_top8_drop_with_shuffled_heel": shuffle_drop,
    },
    "gates": {
        "top8_gain_ge": TOP8_GAIN_GATE,
        "far_rmse_gain_ge": FAR_RMSE_GAIN_GATE,
        "shuffled_top8_drop_ge": SHUFFLED_TOP8_DROP_GATE,
    },
    "checks": checks,
    "strict_pass": bool(all(checks.values()) and selected_weight > 0.0),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["next"] = (
    "PASS -> add same-well heel-reference emission to the complete-well model"
    if verdict["strict_pass"]
    else "FAIL -> the visible heel is not an independent absolute reference; keep it as an anchor only"
)
_write_json(OUT_DIR / "verdict.json", verdict)
print("FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR)
