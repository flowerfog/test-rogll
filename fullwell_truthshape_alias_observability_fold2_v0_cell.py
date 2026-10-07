"""Complete-well GR alias observability gateway.

This experiment isolates the claim implied by the public discussion: a model
that receives one complete well can use full-lateral GR matching to resolve
the absolute TVT mode without neighbour-well information.

It is deliberately an observability gateway, not a deployable trajectory
model. For each well, the true path shape is frozen and translated by global
offsets from -96 to +96 ft. The network must identify the zero-offset path
from horizontal GR and that well's typewell GR. Candidate order is randomly
permuted on every training forward, no offset value is an input feature, and
the untouched validation fold uses only its official visible-prefix mask.

Protocol:
* folds 3/4/5 train one complete-well model;
* fold 2 selects the epoch and is reported with causal controls;
* fold 1 is never loaded;
* no neighbour, PF, strong, CP12, Gold, or submission signal is consumed.

A pass means complete-well GR contains enough absolute-mode information to
justify a direct path proposal model. It is not itself a CV or submission
score because validation truth defines the translated path family.
"""

from __future__ import annotations

from pathlib import Path
import gc
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path("/root/autodl-tmp")
DATA_ROOT_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    Path.cwd(),
)
DATA_ROOT = next(
    (path for path in DATA_ROOT_CANDIDATES if (path / "train").is_dir()),
    DATA_ROOT_CANDIDATES[0],
)
OOF_ROOT = ROOT / "oof_exports"
if not OOF_ROOT.exists():
    OOF_ROOT = DATA_ROOT / "oof_exports"

EXPERIMENT = "fullwell_truthshape_alias_observability_fold2_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "panels"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
if FOLD_PATH is None:
    raise FileNotFoundError("Cannot find well_metadata.csv under " + str(OOF_ROOT))

TRAIN_FOLDS = (3, 4, 5)
VALID_FOLD = 2
SEED = 20260720
SEQUENCE_LENGTH = 768
OFFSETS_FT = np.arange(-96.0, 96.0 + 0.1, 4.0, dtype=np.float32)
ZERO_INDEX = int(np.flatnonzero(OFFSETS_FT == 0.0)[0])
N_CANDIDATES = len(OFFSETS_FT)
MIN_CANDIDATE_COVERAGE = 0.50
MIN_TRUE_COVERAGE = 0.90
MIN_AVAILABLE_CANDIDATES = 9

MAX_EPOCHS = 18
MIN_EPOCHS = 5
PATIENCE = 5
LEARNING_RATE = 3.0e-4
MIN_LEARNING_RATE = 3.0e-5
WEIGHT_DECAY = 3.0e-5
GRAD_ACCUM_WELLS = 4
GRADIENT_CLIP = 4.0

EMBEDDING = 48
SET_DIMENSION = 96
LOSS_ALIAS_MARGIN = 0.20
ALIAS_MARGIN = 0.75

GATE_ARGMAX_RMSE = 5.5
GATE_TOP1_WITHIN5 = 0.70
GATE_ALIAS24_WIN = 0.80
GATE_NO_GR_DROP = 0.15
GATE_SHUFFLED_ALIGNMENT_DROP = 0.20


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _rmse(truth, prediction, weights=None):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    squared = (prediction - truth) ** 2
    if weights is None:
        return float(np.sqrt(np.mean(squared)))
    weights = np.asarray(weights, np.float64)
    return float(np.sqrt(np.sum(weights * squared) / np.sum(weights)))


def _numeric(frame, column):
    if column not in frame:
        return np.full(len(frame), np.nan, np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def _interp_finite(x, y, query):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    query = np.asarray(query, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return np.full(query.shape, np.nan, np.float64)
    xv = x[valid]
    yv = y[valid]
    order = np.argsort(xv, kind="mergesort")
    xv = xv[order]
    yv = yv[order]
    unique, inverse = np.unique(xv, return_inverse=True)
    if len(unique) != len(xv):
        sums = np.bincount(inverse, weights=yv)
        counts = np.bincount(inverse)
        xv = unique
        yv = sums / np.maximum(counts, 1)
    return np.interp(query, xv, yv, left=np.nan, right=np.nan)


def _robust_location_scale(values):
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    if int(valid.sum()) < 16:
        return 0.0, 1.0
    centre = float(np.median(values[valid]))
    scale = float(1.4826 * np.median(np.abs(values[valid] - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values[valid]), 1.0))
    return centre, scale


def _fit_affine(source, target):
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    valid = np.isfinite(source) & np.isfinite(target)
    x = source[valid]
    y = target[valid]
    if len(x) < 32 or np.std(x) < 1e-6:
        shift = float(np.nanmedian(y) - np.nanmedian(x)) if len(x) else 0.0
        return 1.0, shift, int(len(x)), np.nan
    keep = np.ones(len(x), bool)
    slope, intercept = 1.0, float(np.median(y) - np.median(x))
    for _ in range(4):
        design = np.c_[x[keep], np.ones(int(keep.sum()))]
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        centre, scale = _robust_location_scale(residual[keep])
        next_keep = np.abs(residual - centre) <= max(3.5 * scale, 8.0)
        if int(next_keep.sum()) < 32 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    fitted = slope * x[keep] + intercept
    fit_rmse = _rmse(y[keep], fitted) if int(keep.sum()) else np.nan
    return float(slope), float(intercept), int(keep.sum()), float(fit_rmse)


def _load_folds():
    frame = pd.read_csv(FOLD_PATH, dtype={"well": str})
    well_column = next(
        (column for column in ("well", "well_id", "WELL") if column in frame),
        None,
    )
    fold_column = next(
        (column for column in ("fold", "outer_fold", "Fold") if column in frame),
        None,
    )
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Invalid fold columns: {frame.columns.tolist()}")
    frame = frame[[well_column, fold_column]].rename(
        columns={well_column: "well", fold_column: "fold"}
    )
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.drop_duplicates("well", keep="last")
    return frame.sort_values(["fold", "well"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Complete-well translated-path panels
# ---------------------------------------------------------------------------


def _panel_path(well):
    return CACHE_DIR / f"{well}.npz"


def _read_well(well, typewell_well=None):
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_name = well if typewell_well is None else str(typewell_well)
    typewell_path = DATA_ROOT / "train" / f"{typewell_name}__typewell.csv"
    if not horizontal_path.exists():
        raise FileNotFoundError(horizontal_path)
    if not typewell_path.exists():
        raise FileNotFoundError(typewell_path)
    return pd.read_csv(horizontal_path), pd.read_csv(typewell_path)


def _build_arrays(well, typewell_well=None):
    horizontal, typewell = _read_well(well, typewell_well)
    md = _numeric(horizontal, "MD")
    horizontal_gr = _numeric(horizontal, "GR")
    truth_tvt = _numeric(horizontal, "TVT")
    tvt_input = _numeric(horizontal, "TVT_input")
    typewell_tvt = _numeric(typewell, "TVT")
    typewell_gr = _numeric(typewell, "GR")

    known = np.flatnonzero(np.isfinite(tvt_input))
    hidden = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth_tvt))
    if len(known) < 128 or len(hidden) < 128:
        raise RuntimeError(
            f"insufficient official rows known={len(known)} hidden={len(hidden)}"
        )
    if hidden[0] != len(known):
        raise RuntimeError("official TVT_input mask is not a prefix/suffix split")

    tw_prefix = _interp_finite(typewell_tvt, typewell_gr, tvt_input[known])
    slope, intercept, pairs, calibration_rmse = _fit_affine(
        tw_prefix, horizontal_gr[known]
    )
    calibrated_typewell = slope * typewell_gr + intercept
    centre, scale = _robust_location_scale(horizontal_gr[known])

    sequence_md = np.linspace(md[hidden[0]], md[hidden[-1]], SEQUENCE_LENGTH)
    query = _interp_finite(md, horizontal_gr, sequence_md)
    truth_sequence = _interp_finite(md, truth_tvt, sequence_md)
    query_valid = np.isfinite(query)
    query = np.clip((query - centre) / scale, -8.0, 8.0)

    candidate_tvt = truth_sequence[None, :] + OFFSETS_FT[:, None]
    reference = _interp_finite(
        typewell_tvt,
        calibrated_typewell,
        candidate_tvt.reshape(-1),
    ).reshape(N_CANDIDATES, SEQUENCE_LENGTH)
    valid = np.isfinite(reference) & query_valid[None, :]
    reference = np.clip((reference - centre) / scale, -8.0, 8.0)
    query = np.where(query_valid, query, 0.0)
    reference = np.where(valid, reference, 0.0)

    coverage = valid.mean(axis=1)
    available = coverage >= MIN_CANDIDATE_COVERAGE
    if float(coverage[ZERO_INDEX]) < MIN_TRUE_COVERAGE:
        raise RuntimeError(
            "true candidate coverage too low: "
            f"{float(coverage[ZERO_INDEX]):.3f}"
        )
    if int(available.sum()) < MIN_AVAILABLE_CANDIDATES:
        raise RuntimeError(
            f"only {int(available.sum())} sufficiently covered candidates"
        )
    return {
        "well": str(well),
        "query": query.astype(np.float32),
        "reference": reference.astype(np.float32),
        "valid": valid.astype(bool),
        "truth_tvt": truth_sequence.astype(np.float32),
        "eval_rows": int(len(hidden)),
        "coverage": coverage.astype(np.float32),
        "calibration_pairs": int(pairs),
        "calibration_rmse": float(calibration_rmse),
        "typewell_well": str(typewell_well or well),
    }


def _save_panel(panel):
    np.savez_compressed(
        _panel_path(panel["well"]),
        well=np.asarray(panel["well"]),
        query=panel["query"].astype(np.float16),
        reference=panel["reference"].astype(np.float16),
        valid=panel["valid"].astype(np.uint8),
        truth_tvt=panel["truth_tvt"].astype(np.float32),
        eval_rows=np.asarray([panel["eval_rows"]], np.int32),
        coverage=panel["coverage"].astype(np.float32),
        calibration_pairs=np.asarray([panel["calibration_pairs"]], np.int32),
        calibration_rmse=np.asarray([panel["calibration_rmse"]], np.float32),
        cache_version=np.asarray([1], np.int16),
    )


def _load_panel(well):
    with np.load(_panel_path(well), allow_pickle=False) as data:
        return {
            "well": str(data["well"].item()),
            "query": data["query"].astype(np.float32),
            "reference": data["reference"].astype(np.float32),
            "valid": data["valid"].astype(bool),
            "truth_tvt": data["truth_tvt"].astype(np.float32),
            "eval_rows": int(data["eval_rows"][0]),
            "coverage": data["coverage"].astype(np.float32),
        }


folds = _load_folds()
needed_folds = set(TRAIN_FOLDS) | {VALID_FOLD}
needed_wells = folds.loc[folds["fold"].isin(needed_folds), "well"].tolist()
train_wells = folds.loc[folds["fold"].isin(TRAIN_FOLDS), "well"].tolist()
valid_wells = folds.loc[folds["fold"].eq(VALID_FOLD), "well"].tolist()
if len(train_wells) < 450 or len(valid_wells) < 150:
    raise RuntimeError(
        f"Unexpected fold sizes train={len(train_wells)} valid={len(valid_wells)}"
    )

started = time.time()
build_rows = []
for index, well in enumerate(needed_wells, start=1):
    try:
        if not _panel_path(well).exists():
            _save_panel(_build_arrays(well))
        panel = _load_panel(well)
        build_rows.append(
            {
                "well": well,
                "status": "ok",
                "eval_rows": panel["eval_rows"],
                "min_coverage": float(panel["coverage"].min()),
                "mean_coverage": float(panel["coverage"].mean()),
                "true_coverage": float(panel["coverage"][ZERO_INDEX]),
                "available_candidates": int(
                    (panel["coverage"] >= MIN_CANDIDATE_COVERAGE).sum()
                ),
            }
        )
    except Exception as error:
        build_rows.append(
            {"well": well, "status": "failed", "reason": repr(error)}
        )
    if index % 50 == 0 or index == len(needed_wells):
        print(
            f"build alias panels {index}/{len(needed_wells)}", flush=True
        )

build_report = pd.DataFrame(build_rows)
build_report.to_csv(OUT_DIR / "panel_build_report.csv", index=False)
failed = build_report[build_report["status"].eq("failed")]
if len(failed):
    raise RuntimeError(
        f"{len(failed)} panel builds failed: "
        + repr(failed.head(10).to_dict(orient="records"))
    )

guards = {
    "train_wells": len(train_wells),
    "valid_wells": len(valid_wells),
    "candidates": N_CANDIDATES,
    "sequence_length": SEQUENCE_LENGTH,
    "minimum_coverage": float(build_report["min_coverage"].min()),
    "mean_coverage": float(build_report["mean_coverage"].mean()),
    "minimum_true_coverage": float(build_report["true_coverage"].min()),
    "minimum_available_candidates": int(
        build_report["available_candidates"].min()
    ),
    "fold1_touched": False,
}
print("PANEL GUARDS", guards, flush=True)


# ---------------------------------------------------------------------------
# Permutation-equivariant complete-well energy model
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(
            channels, channels, 5, padding=2 * dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )

    def forward(self, values):
        residual = F.gelu(self.norm1(values))
        residual = self.conv1(residual)
        residual = F.gelu(self.norm2(residual))
        residual = self.conv2(residual)
        return values + residual


class CompleteWellAliasEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(7, EMBEDDING, 7, padding=3)
        self.blocks = nn.ModuleList(
            [ResidualBlock(EMBEDDING, dilation) for dilation in (1, 2, 4, 8, 16, 32, 64)]
        )
        descriptor_size = EMBEDDING * 3 + 8
        self.descriptor = nn.Sequential(
            nn.Linear(descriptor_size, SET_DIMENSION),
            nn.GELU(),
            nn.LayerNorm(SET_DIMENSION),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=SET_DIMENSION,
            nhead=4,
            dim_feedforward=SET_DIMENSION * 3,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.energy = nn.Sequential(
            nn.LayerNorm(SET_DIMENSION * 2),
            nn.Linear(SET_DIMENSION * 2, SET_DIMENSION),
            nn.GELU(),
            nn.Linear(SET_DIMENSION, 1),
        )

    def forward(self, query, reference, valid):
        candidates, length = reference.shape
        available = valid.mean(dim=1) >= MIN_CANDIDATE_COVERAGE
        query = query[None, :].expand(candidates, -1)
        difference = query - reference
        absolute = difference.abs()
        smooth32 = F.avg_pool1d(
            difference[:, None, :], 33, stride=1, padding=16
        )[:, 0]
        smooth128 = F.avg_pool1d(
            difference[:, None, :], 129, stride=1, padding=64
        )[:, 0]
        progress = torch.linspace(
            0.0, 1.0, length, device=query.device, dtype=query.dtype
        )[None, :].expand(candidates, -1)
        inputs = torch.stack(
            [
                query,
                reference,
                difference,
                absolute,
                smooth32,
                smooth128,
                valid,
            ],
            dim=1,
        )
        hidden = self.stem(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        mask = valid[:, None, :]
        count = mask.sum(dim=2).clamp_min(1.0)
        mean = (hidden * mask).sum(dim=2) / count
        variance = ((hidden - mean[:, :, None]) ** 2 * mask).sum(dim=2) / count
        maximum = hidden.masked_fill(mask <= 0, -1e4).amax(dim=2)
        maximum = torch.where(available[:, None], maximum, torch.zeros_like(maximum))
        summaries = torch.stack(
            [
                (absolute * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0),
                torch.sqrt(
                    (difference.square() * valid).sum(dim=1)
                    / valid.sum(dim=1).clamp_min(1.0)
                    + 1e-8
                ),
                (difference * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0),
                valid.mean(dim=1),
                (smooth32.abs() * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0),
                (smooth128.abs() * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0),
                (absolute * valid * progress).sum(dim=1)
                / (valid * progress).sum(dim=1).clamp_min(1.0),
                (absolute * valid * (1.0 - progress)).sum(dim=1)
                / (valid * (1.0 - progress)).sum(dim=1).clamp_min(1.0),
            ],
            dim=1,
        )
        descriptor = self.descriptor(
            torch.cat([mean, torch.sqrt(variance + 1e-8), maximum, summaries], dim=1)
        )
        contextual = self.set_encoder(
            descriptor[None],
            src_key_padding_mask=(~available)[None],
        )[0]
        global_descriptor = (
            contextual * available[:, None]
        ).sum(dim=0, keepdim=True) / available.sum().clamp_min(1)
        global_context = global_descriptor.expand_as(contextual)
        logits = self.energy(torch.cat([contextual, global_context], dim=1))[:, 0]
        return logits.masked_fill(~available, -1e4)


def _tensor_panel(panel, device, control="full"):
    query = panel["query"].copy()
    reference = panel["reference"].copy()
    valid = panel["valid"].copy()
    if control == "no_gr":
        query.fill(0.0)
    elif control == "shuffled_alignment":
        # Preserve the same well, values, and coverage while assigning each
        # candidate the GR alignment from a different global mode.
        reference = np.roll(reference, shift=7, axis=0)
        valid = np.roll(valid, shift=7, axis=0)
    elif control != "full":
        raise ValueError(control)
    return (
        torch.from_numpy(query).to(device),
        torch.from_numpy(reference).to(device),
        torch.from_numpy(valid.astype(np.float32)).to(device),
    )


def _loss(model, panel, device):
    query, reference, valid = _tensor_panel(panel, device)
    permutation = torch.randperm(N_CANDIDATES, device=device)
    reference = reference[permutation]
    valid = valid[permutation]
    offsets = torch.from_numpy(OFFSETS_FT).to(device)[permutation]
    target = torch.nonzero(offsets == 0.0, as_tuple=False).flatten()[0]
    logits = model(query, reference, valid)
    ce = F.cross_entropy(logits[None], target[None])
    alias_mask = (offsets.abs() >= 20.0) & (offsets.abs() <= 28.0)
    usable_alias = alias_mask & (logits > -1e3)
    if usable_alias.any():
        alias_score = logits[usable_alias].amax()
        alias_margin = F.relu(ALIAS_MARGIN - logits[target] + alias_score)
    else:
        alias_margin = logits[target] * 0.0
    loss = ce + LOSS_ALIAS_MARGIN * alias_margin
    return loss, {
        "ce": float(ce.detach().cpu()),
        "alias_margin": float(alias_margin.detach().cpu()),
    }


@torch.no_grad()
def _predict_scores(model, panel, device, control="full"):
    model.eval()
    query, reference, valid = _tensor_panel(panel, device, control)
    return model(query, reference, valid).detach().cpu().numpy()


def _evaluate(model, wells, device, control="full"):
    rows = []
    for well in wells:
        panel = _load_panel(well)
        scores = _predict_scores(model, panel, device, control)
        probability = np.exp(scores - np.max(scores))
        probability /= np.sum(probability)
        selected = float(OFFSETS_FT[int(np.argmax(scores))])
        posterior_mean = float(np.sum(probability * OFFSETS_FT))
        ranked = np.argsort(-scores)
        rank_zero = int(np.flatnonzero(ranked == ZERO_INDEX)[0]) + 1
        alias_mask = (
            (np.abs(OFFSETS_FT) >= 20.0)
            & (np.abs(OFFSETS_FT) <= 28.0)
            & (scores > -1e3)
        )
        rows.append(
            {
                "well": well,
                "eval_rows": panel["eval_rows"],
                "selected_offset": selected,
                "posterior_mean_offset": posterior_mean,
                "rank_zero": rank_zero,
                "top1_within5": float(abs(selected) <= 5.0),
                "top5_contains_truth": float(rank_zero <= 5),
                "alias24_win": float(
                    not alias_mask.any()
                    or scores[ZERO_INDEX] > np.max(scores[alias_mask])
                ),
                "entropy": float(-np.sum(probability * np.log(probability + 1e-12))),
            }
        )
    frame = pd.DataFrame(rows)
    weights = frame["eval_rows"].to_numpy(np.float64)
    metrics = {
        "wells": len(frame),
        "rows": int(weights.sum()),
        "argmax_rmse": _rmse(
            np.zeros(len(frame)), frame["selected_offset"], weights
        ),
        "posterior_mean_rmse": _rmse(
            np.zeros(len(frame)), frame["posterior_mean_offset"], weights
        ),
        "top1_within5": float(
            np.average(frame["top1_within5"], weights=weights)
        ),
        "top5_contains_truth": float(
            np.average(frame["top5_contains_truth"], weights=weights)
        ),
        "median_truth_rank": float(frame["rank_zero"].median()),
        "p90_truth_rank": float(frame["rank_zero"].quantile(0.90)),
        "alias24_win_rate": float(
            np.average(frame["alias24_win"], weights=weights)
        ),
        "mean_entropy": float(np.average(frame["entropy"], weights=weights)),
    }
    return metrics, frame


def _train_epoch(model, wells, optimizer, scaler, device, epoch):
    model.train()
    order = list(wells)
    random.Random(SEED + epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    sums = {"loss": 0.0, "ce": 0.0, "alias_margin": 0.0}
    steps = 0
    for index, well in enumerate(order, start=1):
        panel = _load_panel(well)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            loss, parts = _loss(model, panel, device)
            scaled_loss = loss / GRAD_ACCUM_WELLS
        scaler.scale(scaled_loss).backward()
        if index % GRAD_ACCUM_WELLS == 0 or index == len(order):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        sums["loss"] += float(loss.detach().cpu())
        for key, value in parts.items():
            sums[key] += value
        steps += 1
        del panel, loss, scaled_loss
    return {key: value / max(steps, 1) for key, value in sums.items()}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA is required for this complete-well gateway")
_seed_everything(SEED)

model = CompleteWellAliasEnergy().to(device)
parameters = int(sum(parameter.numel() for parameter in model.parameters()))
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS, eta_min=MIN_LEARNING_RATE
)
scaler = torch.amp.GradScaler("cuda", enabled=True)

history = []
best = {"rmse": np.inf, "top1": -np.inf, "epoch": 0, "state": None}
bad_epochs = 0
for epoch in range(1, MAX_EPOCHS + 1):
    training = _train_epoch(model, train_wells, optimizer, scaler, device, epoch)
    metrics, _ = _evaluate(model, valid_wells, device, control="full")
    report = {
        "epoch": epoch,
        **{f"train_{key}": value for key, value in training.items()},
        **{f"valid_{key}": value for key, value in metrics.items()},
        "learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    history.append(report)
    print("ALIAS OBSERVABILITY EPOCH", report, flush=True)
    improved = (
        metrics["argmax_rmse"] < best["rmse"] - 1e-6
        or (
            abs(metrics["argmax_rmse"] - best["rmse"]) <= 1e-6
            and metrics["top1_within5"] > best["top1"]
        )
    )
    if improved:
        best = {
            "rmse": metrics["argmax_rmse"],
            "top1": metrics["top1_within5"],
            "epoch": epoch,
            "state": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
        }
        bad_epochs = 0
    else:
        bad_epochs += 1
    scheduler.step()
    if epoch >= MIN_EPOCHS and bad_epochs >= PATIENCE:
        print("early stopping", flush=True)
        break

if best["state"] is None:
    raise RuntimeError("Training did not produce a checkpoint")
model.load_state_dict(best["state"])
checkpoint_path = MODEL_DIR / "best_fold2_alias_observability.pt"
torch.save(
    {
        "experiment": EXPERIMENT,
        "epoch": best["epoch"],
        "model": best["state"],
        "offsets_ft": OFFSETS_FT,
        "parameters": parameters,
    },
    checkpoint_path,
)

history_frame = pd.DataFrame(history)
history_frame.to_csv(OUT_DIR / "training_history.csv", index=False)
full_metrics, full_wells = _evaluate(model, valid_wells, device, control="full")
no_gr_metrics, no_gr_wells = _evaluate(model, valid_wells, device, control="no_gr")
shuffle_metrics, shuffle_wells = _evaluate(
    model, valid_wells, device, control="shuffled_alignment"
)
full_wells.to_csv(OUT_DIR / "fold2_full_wells.csv", index=False)
no_gr_wells.to_csv(OUT_DIR / "fold2_no_gr_wells.csv", index=False)
shuffle_wells.to_csv(OUT_DIR / "fold2_shuffled_alignment_wells.csv", index=False)

controls = {
    "no_gr_top1_drop": full_metrics["top1_within5"] - no_gr_metrics["top1_within5"],
    "shuffled_alignment_top1_drop": (
        full_metrics["top1_within5"] - shuffle_metrics["top1_within5"]
    ),
    "no_gr_posterior_rmse_degradation": (
        no_gr_metrics["posterior_mean_rmse"] - full_metrics["posterior_mean_rmse"]
    ),
    "shuffled_alignment_posterior_rmse_degradation": (
        shuffle_metrics["posterior_mean_rmse"]
        - full_metrics["posterior_mean_rmse"]
    ),
}

gates = {
    "argmax_rmse_le": GATE_ARGMAX_RMSE,
    "top1_within5_ge": GATE_TOP1_WITHIN5,
    "alias24_win_rate_ge": GATE_ALIAS24_WIN,
    "no_gr_top1_drop_ge": GATE_NO_GR_DROP,
    "shuffled_alignment_top1_drop_ge": GATE_SHUFFLED_ALIGNMENT_DROP,
}
strict_pass = bool(
    full_metrics["argmax_rmse"] <= GATE_ARGMAX_RMSE
    and full_metrics["top1_within5"] >= GATE_TOP1_WITHIN5
    and full_metrics["alias24_win_rate"] >= GATE_ALIAS24_WIN
    and controls["no_gr_top1_drop"] >= GATE_NO_GR_DROP
    and controls["shuffled_alignment_top1_drop"] >= GATE_SHUFFLED_ALIGNMENT_DROP
)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_complete_well_absolute_mode_observability",
    "sample_unit": "one complete physical well",
    "deployable": False,
    "truth_usage": "validation truth defines translated path shape only",
    "legal_evidence": "horizontal GR plus the same well's typewell and visible prefix",
    "uses_neighbor": False,
    "uses_pf_strong_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
        "selected_epoch": best["epoch"],
        "candidate_order_randomised_during_training": True,
        "candidate_offset_is_model_input": False,
    },
    "parameters": parameters,
    "guards": guards,
    "full": full_metrics,
    "posterior_mean_warning": (
        "Diagnostic only: a symmetric uninformative posterior has mean zero, "
        "so posterior_mean_rmse is never used for selection or promotion."
    ),
    "no_gr": no_gr_metrics,
    "shuffled_alignment": shuffle_metrics,
    "controls": controls,
    "gates": gates,
    "strict_pass": strict_pass,
    "checkpoint": checkpoint_path,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": float(__import__("shutil").disk_usage(OOF_ROOT).free / 2**30),
    "next": (
        "PASS -> build direct complete-well path proposal on the same native evidence"
        if strict_pass
        else "FAIL -> absolute alias is not identifiable from this GR representation; stop decoder scaling"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("FULL", json.dumps(full_metrics, indent=2), flush=True)
print("CONTROLS", json.dumps(controls, indent=2), flush=True)
print("FINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)

del model
gc.collect()
torch.cuda.empty_cache()
