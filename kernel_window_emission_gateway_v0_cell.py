"""Fold-honest kernel window-emission gateway.

The experiment isolates the GR observation model before any DP/Viterbi path
decoder.  Folds 3/4/5 choose one fixed emission recipe from pointwise, NCC,
exact RBF, and Random Fourier Feature approximations.  Fold 2 is evaluated
once with distance strata and shuffled-typewell/no-GR controls.  Fold 1 is not
touched.

Prerequisite: all 773 native dual-reference record files built by
``build_native_dualref_all_records_v0_cell.py``.
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
# Frozen configuration
# ---------------------------------------------------------------------------

EXPERIMENT = "kernel_window_emission_gateway_v0"
ROOT = Path("/root/autodl-tmp")
DATA_ROOT_CANDIDATES = (ROOT / "rogii-wellbore-geology-prediction", Path.cwd())
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
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (3, 4, 5)
VALID_FOLD = 2
SEED = 20260721

STATE_HALF_WIDTH_FT = 128
STATE_OFFSETS = np.arange(-STATE_HALF_WIDTH_FT, STATE_HALF_WIDTH_FT + 1, dtype=np.float32)
N_STATES = len(STATE_OFFSETS)
QUERY_HALF_WIDTHS_MD = np.asarray([64.0, 128.0, 256.0, 512.0, 1024.0], np.float32)
REFERENCE_HALF_WIDTHS_TVT = np.asarray([4.0, 8.0, 16.0, 32.0, 64.0], np.float32)
WINDOW_POINTS = 65
WINDOW_UNIT = np.linspace(-1.0, 1.0, WINDOW_POINTS, dtype=np.float32)
MIN_WINDOW_COVERAGE = 0.70

TRAIN_ROWS_PER_WELL = 96
VALID_ROWS_PER_WELL = 192
RFF_DIMENSIONS = 96
RFF_GAMMAS = (0.5, 1.0, 2.0)
SOFTMAX_TEMPERATURE = 0.35

TOP8_GATE = 0.75
TOP8_GAIN_VS_NCC_GATE = 0.05
MEDIAN_RANK_GATE = 5.0
ALIAS24_GATE = 0.75
DEEP_TOP8_GATE = 0.72
SHUFFLED_DROP_GATE = 0.15


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


def _record_path(well):
    return next(
        (directory / f"{well}.npz" for directory in RECORD_DIRS if (directory / f"{well}.npz").exists()),
        None,
    )


def _load_record(well):
    path = _record_path(well)
    if path is None:
        raise FileNotFoundError(
            f"Missing record for {well}; run build_native_dualref_all_records_v0_cell.py"
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
    path = next((candidate for candidate in FOLD_CANDIDATES if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("Cannot find fold-clean well_metadata.csv")
    frame = pd.read_csv(path, dtype=str)
    well_col = next((col for col in ("well", "well_id", "WELL") if col in frame), None)
    fold_col = next((col for col in ("fold", "outer_fold", "Fold") if col in frame), None)
    if well_col is None or fold_col is None:
        raise RuntimeError(f"Unexpected fold columns: {frame.columns.tolist()}")
    frame = frame[[well_col, fold_col]].rename(columns={well_col: "well", fold_col: "fold"})
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.drop_duplicates("well", keep="last")
    return frame.sort_values(["fold", "well"]).reset_index(drop=True), path


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


def _embeddings(values, mask, device):
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


def _rff_parameters(device):
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    parameters = {}
    for gamma in RFF_GAMMAS:
        weight = torch.randn(
            RFF_DIMENSIONS,
            WINDOW_POINTS,
            generator=generator,
            device=device,
        ) * math.sqrt(2.0 * gamma)
        phase = 2.0 * math.pi * torch.rand(
            RFF_DIMENSIONS,
            generator=generator,
            device=device,
        )
        parameters[gamma] = (weight, phase)
    return parameters


@torch.no_grad()
def _score_bundle(record, rows, device, rff_parameters, reference_record=None):
    reference_record = record if reference_record is None else reference_record
    query_values, query_mask = _window_bank(
        record.md,
        record.horizontal[0],
        record.horizontal[1],
        record.md[rows],
        QUERY_HALF_WIDTHS_MD,
    )
    query, query_coverage = _embeddings(query_values, query_mask, device)

    length = reference_record.typewell.shape[1]
    reference_axis = np.arange(length, dtype=np.float32) - float(length // 2)
    reference_values, reference_mask = _window_bank(
        reference_axis,
        reference_record.typewell[0],
        reference_record.typewell[1],
        STATE_OFFSETS,
        REFERENCE_HALF_WIDTHS_TVT,
    )
    reference, reference_coverage = _embeddings(reference_values, reference_mask, device)

    pair_valid = (
        (query_coverage[:, :, None, None] >= MIN_WINDOW_COVERAGE)
        & (reference_coverage[None, None, :, :] >= MIN_WINDOW_COVERAGE)
    )
    correlation = torch.einsum("qsp,trp->qstr", query, reference)
    correlation = correlation.masked_fill(~pair_valid, -1.0)
    flat_corr = correlation.permute(0, 2, 1, 3).reshape(len(rows), N_STATES, -1)
    top4_corr = flat_corr.topk(4, dim=-1).values

    query_centre = torch.from_numpy(record.horizontal[0, rows].astype(np.float32)).to(device)
    state_start = length // 2 - STATE_HALF_WIDTH_FT
    reference_centre = torch.from_numpy(
        reference_record.typewell[0, state_start : state_start + N_STATES].astype(np.float32)
    ).to(device)
    state_valid = torch.from_numpy(
        (reference_record.typewell[1, state_start : state_start + N_STATES] >= 0.5)
    ).to(device)
    pointwise = -(query_centre[:, None] - reference_centre[None, :]) ** 2

    scores = {
        "pointwise": _row_standardise(pointwise),
        "ncc_max": _row_standardise(flat_corr.amax(dim=-1)),
        "ncc_top4": _row_standardise(top4_corr.mean(dim=-1)),
    }
    distance2 = (2.0 - 2.0 * flat_corr).clamp(0.0, 4.0)
    for gamma in RFF_GAMMAS:
        exact = torch.exp(-gamma * distance2)
        scores[f"rbf_g{gamma:g}_top4"] = _row_standardise(
            exact.topk(4, dim=-1).values.mean(dim=-1)
        )

        weight, phase = rff_parameters[gamma]
        scale = math.sqrt(2.0 / RFF_DIMENSIONS)
        query_phi = scale * torch.cos(torch.einsum("qsp,dp->qsd", query, weight) + phase)
        reference_phi = scale * torch.cos(
            torch.einsum("trp,dp->trd", reference, weight) + phase
        )
        rff_pair = torch.einsum("qsd,trd->qstr", query_phi, reference_phi)
        rff_pair = rff_pair.masked_fill(~pair_valid, -1.0)
        flat_rff = rff_pair.permute(0, 2, 1, 3).reshape(len(rows), N_STATES, -1)
        scores[f"rff_g{gamma:g}_d{RFF_DIMENSIONS}_top4"] = _row_standardise(
            flat_rff.topk(4, dim=-1).values.mean(dim=-1)
        )
    for name in scores:
        scores[name] = scores[name].masked_fill(~state_valid[None], -1e6)
    return {name: value.cpu().numpy().astype(np.float32) for name, value in scores.items()}


def _row_metrics(record, rows, score, variant):
    truth_offset = record.truth[rows].astype(np.float64) - record.anchor_tvt
    distance = record.md[rows].astype(np.float64) - float(record.md[record.anchor_index])
    order = np.argsort(-score, axis=1, kind="mergesort")
    nearest = np.abs(STATE_OFFSETS[None] - truth_offset[:, None]).argmin(axis=1)
    rank = np.argmax(order == nearest[:, None], axis=1) + 1
    top1 = STATE_OFFSETS[order[:, 0]].astype(np.float64)
    top8 = STATE_OFFSETS[order[:, :8]].astype(np.float64)
    top8_hit = (np.abs(top8 - truth_offset[:, None]) <= 5.0).any(axis=1)
    truth_score = score[np.arange(len(rows)), nearest]
    minus = score[np.arange(len(rows)), np.clip(nearest - 24, 0, N_STATES - 1)]
    plus = score[np.arange(len(rows)), np.clip(nearest + 24, 0, N_STATES - 1)]
    return pd.DataFrame(
        {
            "well": record.well,
            "variant": variant,
            "distance_ft": distance,
            "truth_offset": truth_offset,
            "top1_offset": top1,
            "top1_error": top1 - truth_offset,
            "top8_hit": top8_hit,
            "truth_rank": rank,
            "alias24_win": truth_score > np.maximum(minus, plus),
        }
    )


def _path_row(record, rows, score, variant):
    shifted = (score.astype(np.float64) - score.max(axis=1, keepdims=True)) / SOFTMAX_TEMPERATURE
    probability = np.exp(np.clip(shifted, -60.0, 0.0))
    probability /= probability.sum(axis=1, keepdims=True).clip(1e-12)
    node_offset = probability @ STATE_OFFSETS.astype(np.float64)
    eval_rows = record.eval_index
    eval_md = record.md[eval_rows].astype(np.float64)
    prediction = record.anchor_tvt + np.interp(
        eval_md,
        record.md[rows].astype(np.float64),
        node_offset,
    )
    truth = record.truth[eval_rows].astype(np.float64)
    error = prediction - truth
    distance = eval_md - float(record.md[record.anchor_index])
    output = {
        "well": record.well,
        "variant": variant,
        "rows": len(eval_rows),
        "sse": float(np.sum(error * error)),
        "mae_sum": float(np.sum(np.abs(error))),
    }
    for name, mask in (
        ("near_0_500", distance <= 500),
        ("mid_500_1000", (distance > 500) & (distance <= 1000)),
        ("mid_1000_2000", (distance > 1000) & (distance <= 2000)),
        ("far_gt2000", distance > 2000),
    ):
        output[f"{name}_rows"] = int(mask.sum())
        output[f"{name}_sse"] = float(np.sum(error[mask] ** 2))
    return output


def _summarise_retrieval(frame):
    deep = frame[frame["distance_ft"] > 2000]
    return {
        "sampled_rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "top1_within5": float((frame["top1_error"].abs() <= 5.0).mean()),
        "top8_within5": float(frame["top8_hit"].mean()),
        "median_truth_rank": float(frame["truth_rank"].median()),
        "p90_truth_rank": float(frame["truth_rank"].quantile(0.90)),
        "alias24_win_rate": float(frame["alias24_win"].mean()),
        "deep_top8_within5": float(deep["top8_hit"].mean()),
        "sampled_argmax_rmse": float(np.sqrt(np.mean(frame["top1_error"] ** 2))),
    }


def _summarise_path(frame):
    rows = int(frame["rows"].sum())
    output = {
        "rows": rows,
        "wells": int(frame["well"].nunique()),
        "path_rmse": float(math.sqrt(frame["sse"].sum() / rows)),
        "path_mae": float(frame["mae_sum"].sum() / rows),
    }
    for name in ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000"):
        count = int(frame[f"{name}_rows"].sum())
        output[f"{name}_rmse"] = (
            float(math.sqrt(frame[f"{name}_sse"].sum() / count)) if count else float("nan")
        )
    return output


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

if not RECORD_DIRS:
    raise FileNotFoundError("No native dual-reference record directory exists")
folds, fold_path = _load_folds()
train_wells = folds.loc[folds["fold"].isin(TRAIN_FOLDS), "well"].tolist()
valid_wells = folds.loc[folds["fold"].eq(VALID_FOLD), "well"].tolist()
if len(train_wells) not in (463, 464) or len(valid_wells) != 155:
    raise RuntimeError(f"Unexpected fold sizes: {len(train_wells)}/{len(valid_wells)}")
missing = [well for well in train_wells + valid_wells if _record_path(well) is None]
if missing:
    raise FileNotFoundError(f"Missing {len(missing)} record files: {missing[:10]}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA is required")
rff_parameters = _rff_parameters(device)
started = time.time()

train_frames = []
for number, well in enumerate(train_wells, start=1):
    record = _load_record(well)
    rows = _select_even(record.eval_index, TRAIN_ROWS_PER_WELL)
    bundle = _score_bundle(record, rows, device, rff_parameters)
    for variant, score in bundle.items():
        train_frames.append(_row_metrics(record, rows, score, variant))
    if number % 50 == 0 or number == len(train_wells):
        print(f"kernel emission selection {number}/{len(train_wells)}", flush=True)

train_retrieval = pd.concat(train_frames, ignore_index=True)
train_summary = pd.DataFrame(
    [
        {"variant": variant, **_summarise_retrieval(group)}
        for variant, group in train_retrieval.groupby("variant", sort=True)
    ]
).sort_values(
    ["top8_within5", "alias24_win_rate", "median_truth_rank"],
    ascending=[False, False, True],
)
selected_variant = str(train_summary.iloc[0]["variant"])
train_summary.to_csv(OUT_DIR / "fold345_emission_selection.csv", index=False)
print("FOLD345 EMISSION SELECTION")
print(train_summary.to_string(index=False), flush=True)
del train_frames, train_retrieval

retrieval_frames = []
path_rows = []
donor_wells = valid_wells[1:] + valid_wells[:1]
for number, (well, donor_well) in enumerate(zip(valid_wells, donor_wells), start=1):
    record = _load_record(well)
    donor = _load_record(donor_well)
    rows = _select_even(record.eval_index, VALID_ROWS_PER_WELL)
    bundle = _score_bundle(record, rows, device, rff_parameters)
    shuffled = _score_bundle(
        record,
        rows,
        device,
        rff_parameters,
        reference_record=donor,
    )[selected_variant]
    bundle["selected_shuffled_typewell"] = shuffled
    bundle["selected_no_gr"] = np.where(
        bundle[selected_variant] > -1e5,
        0.0,
        -1e6,
    ).astype(np.float32)
    for variant, score in bundle.items():
        retrieval_frames.append(_row_metrics(record, rows, score, variant))
        path_rows.append(_path_row(record, rows, score, variant))
    if number % 25 == 0 or number == len(valid_wells):
        print(f"Fold2 kernel emission {number}/{len(valid_wells)}", flush=True)

retrieval = pd.concat(retrieval_frames, ignore_index=True)
paths = pd.DataFrame(path_rows)
retrieval.to_csv(OUT_DIR / "fold2_sampled_retrieval.csv", index=False)
paths.to_csv(OUT_DIR / "fold2_path_bywell.csv", index=False)
reports = {}
for variant, frame in retrieval.groupby("variant", sort=True):
    reports[variant] = {
        **_summarise_retrieval(frame),
        **_summarise_path(paths[paths["variant"].eq(variant)]),
    }
comparison = pd.DataFrame(
    [{"variant": variant, **report} for variant, report in reports.items()]
).sort_values(["top8_within5", "median_truth_rank"], ascending=[False, True])
comparison.to_csv(OUT_DIR / "fold2_comparison.csv", index=False)
print("FOLD2 COMPARISON")
print(comparison.to_string(index=False), flush=True)

selected = reports[selected_variant]
ncc = reports["ncc_top4"]
shuffled = reports["selected_shuffled_typewell"]
no_gr = reports["selected_no_gr"]
diagnostics = {
    "top8_gain_vs_ncc_top4": selected["top8_within5"] - ncc["top8_within5"],
    "shuffled_top8_drop": selected["top8_within5"] - shuffled["top8_within5"],
    "no_gr_top8_drop": selected["top8_within5"] - no_gr["top8_within5"],
}
checks = {
    "top8": bool(selected["top8_within5"] >= TOP8_GATE),
    "gain_vs_ncc": bool(diagnostics["top8_gain_vs_ncc_top4"] >= TOP8_GAIN_VS_NCC_GATE),
    "rank": bool(selected["median_truth_rank"] <= MEDIAN_RANK_GATE),
    "alias": bool(selected["alias24_win_rate"] >= ALIAS24_GATE),
    "deep": bool(selected["deep_top8_within5"] >= DEEP_TOP8_GATE),
    "shuffled": bool(diagnostics["shuffled_top8_drop"] >= SHUFFLED_DROP_GATE),
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_kernel_window_observation_gateway",
    "sample_unit": "sampled stations nested within complete physical wells",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_strong_cp12_gold": False,
    "protocol": {
        "selection_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
        "selected_variant": selected_variant,
        "rff_dimensions": RFF_DIMENSIONS,
        "window_input": "prefix-calibrated robust-normalised GR; no movement cost",
    },
    "fold_source": str(fold_path),
    "selected": selected,
    "ncc_reference": ncc,
    "controls": {
        "shuffled_typewell": shuffled,
        "no_gr": no_gr,
    },
    "diagnostics": diagnostics,
    "gates": {
        "top8_within5_ge": TOP8_GATE,
        "top8_gain_vs_ncc_ge": TOP8_GAIN_VS_NCC_GATE,
        "median_rank_le": MEDIAN_RANK_GATE,
        "alias24_win_ge": ALIAS24_GATE,
        "deep_top8_ge": DEEP_TOP8_GATE,
        "shuffled_top8_drop_ge": SHUFFLED_DROP_GATE,
    },
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["next"] = (
    "PASS -> use the frozen kernel emission inside the existing Viterbi decoder"
    if verdict["strict_pass"]
    else "FAIL -> kernelised window matching does not fix absolute alias observability; do not run DP"
)
_write_json(OUT_DIR / "verdict.json", verdict)
print("FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR)
