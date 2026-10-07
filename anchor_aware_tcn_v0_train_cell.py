"""Anchor-aware sequence residual decoder v0.

This is the first learned path decoder after closing the hand-written DP route.
It uses a strong honest CP12 OOF base and only deployment-available inputs:

* the complete eval GR/Z sequence and missingness mask;
* distance from the last known TVT anchor;
* the CP12 base path and its local increments;
* matcher-v1 score profiles at base +/-64 ft.

The TCN predicts an entire residual sequence. Its correction is multiplied by
a fixed anchor ramp, so movement is exactly zero at prediction start. Splits
are by original well and the matcher profile for fold f is produced by matcher
checkpoint f, which never trained on validation fold f.

This is a kill-test, not a deployment cell. A model is considered usable only
if pooled-row OOF gain is >=0.10 ft, all five folds are positive, the minimum
fold gain is >=0.10 ft, and top40 gain is positive.
"""

from pathlib import Path
import gc
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Frozen v0 configuration.
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
FOLD_OOF = (
    OOF_ROOT
    / "multicut_fleongg_v0_k3_nospatial_r768_lgb1"
    / "multicut_v0_oof.pkl"
)
MATCHER_DIR = OOF_ROOT / "learned_gr_matcher_v1_global_ncc_hardneg27"
MATCHER_MODEL_DIR = MATCHER_DIR / "models"
MATCHER_VERDICT = MATCHER_DIR / "fullgrid_verdict.json"

BASE_COLUMN = "cp12"
CACHE_VERSION = "v0_cp12_matcher33_stride8"
CACHE_DIR = OOF_ROOT / f"anchor_aware_tcn_{CACHE_VERSION}"
SEQUENCE_DIR = CACHE_DIR / "sequences"
MODEL_DIR = CACHE_DIR / "models"

WINDOW_LENGTH = 65
HALF_WINDOW = WINDOW_LENGTH // 2
ROW_STRIDE = 8
OFFSETS_FT = np.arange(-64.0, 64.0 + 0.1, 4.0, dtype=np.float32)
ANCHOR_RAMP_ROWS = 250.0
MAX_CORRECTION_FT = 32.0

N_FOLDS = 5
HIDDEN_CHANNELS = 96
DILATIONS = (1, 2, 4, 8, 16, 32)
DROPOUT = 0.10
BATCH_SIZE = 8
EPOCHS = 18
PATIENCE = 4
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
RANDOM_STATE = 20260713
APPLY_WEIGHTS = (0.25, 0.50, 0.75, 1.00)

PASS_GAIN = 0.10
PASS_MIN_FOLD_GAIN = 0.10

for directory in (CACHE_DIR, SEQUENCE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

required = [POOL_PATH, FOLD_OOF, MATCHER_VERDICT]
required += [MATCHER_MODEL_DIR / f"matcher_fold{fold}.pt" for fold in range(1, 6)]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Missing prerequisites: " + repr(missing))
matcher_verdict = json.loads(MATCHER_VERDICT.read_text())
if not bool(matcher_verdict.get("strict_pass", False)):
    raise RuntimeError("Matcher v1 did not pass its full-grid retrieval gate")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(
    {
        "experiment": "anchor_aware_tcn_v0",
        "device": str(DEVICE),
        "base": BASE_COLUMN,
        "row_stride": ROW_STRIDE,
        "score_offsets": OFFSETS_FT.tolist(),
        "max_correction_ft": MAX_CORRECTION_FT,
        "epochs": EPOCHS,
        "cache": str(CACHE_DIR),
    }
)


def _rmse(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _fill_1d(values):
    return (
        pd.Series(np.asarray(values, float))
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _robust_normalise(values, mask=None):
    values = np.asarray(values, float)
    if mask is None:
        mask = np.isfinite(values)
    else:
        mask = np.asarray(mask, bool) & np.isfinite(values)
    output = np.zeros(len(values), dtype=np.float32)
    finite = values[mask]
    if len(finite) < 4:
        return output
    median = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - median)))
    if scale < 1e-3:
        scale = float(np.std(finite) + 1e-3)
    output[mask] = np.clip((values[mask] - median) / scale, -6.0, 6.0)
    return output


def _normalise_window(values, mask):
    return _robust_normalise(np.asarray(values, float), np.asarray(mask, bool))


def _prepare_typewell(path):
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    frame = frame[np.isfinite(frame["TVT"])].sort_values("TVT")
    frame = frame.groupby("TVT", as_index=False)["GR"].mean()
    tvt = frame["TVT"].to_numpy(float)
    gr = _fill_1d(frame["GR"].to_numpy(float))
    if len(tvt) < WINDOW_LENGTH or np.ptp(tvt) < 200:
        raise RuntimeError("typewell coverage too short")
    return tvt, gr


def _horizontal_windows(horizontal_gr, positions):
    horizontal_gr = np.asarray(horizontal_gr, float)
    windows = np.zeros((len(positions), 2, WINDOW_LENGTH), dtype=np.float32)
    offsets = np.arange(-HALF_WINDOW, HALF_WINDOW + 1)
    for index, position in enumerate(positions):
        rows = int(position) + offsets
        inside = (rows >= 0) & (rows < len(horizontal_gr))
        clipped = np.clip(rows, 0, len(horizontal_gr) - 1)
        values = horizontal_gr[clipped]
        mask = inside & np.isfinite(values)
        windows[index, 0] = _normalise_window(values, mask)
        windows[index, 1] = mask.astype(np.float32)
    return windows


def _typewell_windows(tw_tvt, tw_gr, centres):
    relative = np.arange(-HALF_WINDOW, HALF_WINDOW + 1, dtype=np.float32)
    queries = np.asarray(centres, float)[:, None] + relative[None, :]
    masks = (queries >= tw_tvt[0]) & (queries <= tw_tvt[-1])
    values = np.interp(queries.reshape(-1), tw_tvt, tw_gr).reshape(
        len(centres), WINDOW_LENGTH
    )
    windows = np.zeros((len(centres), 2, WINDOW_LENGTH), dtype=np.float32)
    for index in range(len(centres)):
        windows[index, 0] = _normalise_window(values[index], masks[index])
    windows[:, 1] = masks.astype(np.float32)
    return windows


# ---------------------------------------------------------------------------
# Exact matcher-v1 architecture.
# ---------------------------------------------------------------------------
class MatcherResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
        )

    def forward(self, values):
        return F.silu(values + self.network(values))


class WindowEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 48, 7, padding=3),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            MatcherResidualBlock(48, 1),
            MatcherResidualBlock(48, 2),
            MatcherResidualBlock(48, 4),
        )
        self.output = nn.Sequential(
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Linear(96, embedding_dim),
        )

    def forward(self, values):
        hidden = self.blocks(self.stem(values))
        pooled = torch.cat([hidden.mean(-1), hidden.amax(-1)], dim=1)
        return F.normalize(self.output(pooled), dim=1)


class GRWindowMatcher(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.horizontal_encoder = WindowEncoder(embedding_dim)
        self.typewell_encoder = WindowEncoder(embedding_dim)
        self.scorer = nn.Sequential(
            nn.Linear(4 * embedding_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def _score_embeddings(self, horizontal, typewell):
        horizontal = horizontal[:, None].expand_as(typewell)
        pair = torch.cat(
            [horizontal, typewell, torch.abs(horizontal - typewell), horizontal * typewell],
            dim=-1,
        )
        return self.scorer(pair).squeeze(-1)


def _load_matcher(fold):
    model = GRWindowMatcher().to(DEVICE)
    path = MATCHER_MODEL_DIR / f"matcher_fold{int(fold)}.pt"
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _encode_windows(encoder, windows, batch_size=2048):
    parts = []
    for start in range(0, len(windows), batch_size):
        tensor = torch.from_numpy(
            np.asarray(windows[start : start + batch_size], np.float32)
        ).to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            parts.append(encoder(tensor).float().cpu())
    return torch.cat(parts).to(DEVICE, non_blocking=True)


@torch.no_grad()
def _score_grid_profiles(model, horizontal_windows, state_windows, row_chunk=48):
    """Score every sampled row against a typewell grid encoded once per well."""
    horizontal = _encode_windows(model.horizontal_encoder, horizontal_windows)
    forward = _encode_windows(model.typewell_encoder, state_windows)
    reverse = _encode_windows(
        model.typewell_encoder, state_windows[:, :, ::-1].copy()
    )
    parts = []
    for start in range(0, len(horizontal), row_chunk):
        h = horizontal[start : start + row_chunk]
        rows = len(h)
        tw_forward = forward[None].expand(rows, -1, -1)
        tw_reverse = reverse[None].expand(rows, -1, -1)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits = torch.maximum(
                model._score_embeddings(h, tw_forward),
                model._score_embeddings(h, tw_reverse),
            )
        parts.append(logits.float().cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Load the strong base and frozen well folds.
# ---------------------------------------------------------------------------
pool_columns = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
needed_columns = [column for column in ["id", "well", "row_idx", "tvt_true", BASE_COLUMN] if column in pool_columns]
if not {"id", "well", "tvt_true", BASE_COLUMN} <= set(needed_columns):
    raise RuntimeError(f"Pool is missing required columns; found {pool_columns}")
pool = pd.read_csv(POOL_PATH, usecols=needed_columns)
pool["well"] = pool["well"].astype(str)
pool["id"] = pool["id"].astype(str)
if "row_idx" not in pool.columns:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(int)

fold_source = pd.read_pickle(FOLD_OOF)[["well", "fold"]].drop_duplicates("well")
fold_source["well"] = fold_source["well"].astype(str)
fold_map = dict(zip(fold_source["well"], fold_source["fold"].astype(int)))
pool["fold"] = pool["well"].map(fold_map)
if pool["fold"].isna().any():
    missing_wells = sorted(pool.loc[pool["fold"].isna(), "well"].unique())
    raise RuntimeError(f"Missing fold labels for {len(missing_wells)} wells")
pool["fold"] = pool["fold"].astype(int)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
pool_group_indices = pool.groupby("well", sort=False).indices

print(
    "pool:",
    pool.shape,
    "wells:",
    pool["well"].nunique(),
    "base_rmse:",
    _rmse(pool["tvt_true"], pool[BASE_COLUMN]),
)


def _build_sequence(wid, fold, matcher, group):
    cache_path = SEQUENCE_DIR / f"{wid}.npz"
    if cache_path.exists():
        return cache_path, None
    try:
        group = group.sort_values("row_idx")
        horizontal_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
        typewell_path = DATA_ROOT / "train" / f"{wid}__typewell.csv"
        hw = pd.read_csv(horizontal_path, usecols=["Z", "GR", "TVT_input"])
        tw_tvt, tw_gr = _prepare_typewell(typewell_path)

        tvt_input = hw["TVT_input"].to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        evaluation = np.flatnonzero(~np.isfinite(tvt_input))
        if len(known) < 30 or len(evaluation) < 64:
            raise RuntimeError("insufficient known/eval rows")
        anchor_position = int(known[-1])
        if int(evaluation[0]) != anchor_position + 1:
            raise RuntimeError("TVT_input is not a contiguous prefix")
        if not np.array_equal(group["row_idx"].to_numpy(int), evaluation):
            raise RuntimeError("pool rows do not match official eval suffix")

        sampled = evaluation[::ROW_STRIDE]
        if sampled[-1] != evaluation[-1]:
            sampled = np.append(sampled, evaluation[-1])
        by_row = group.set_index("row_idx")
        base_full = by_row.loc[evaluation, BASE_COLUMN].to_numpy(float)
        truth_full = by_row.loc[evaluation, "tvt_true"].to_numpy(float)
        base_sampled = by_row.loc[sampled, BASE_COLUMN].to_numpy(float)
        truth_sampled = by_row.loc[sampled, "tvt_true"].to_numpy(float)

        horizontal = _horizontal_windows(hw["GR"].to_numpy(float), sampled)
        anchor_tvt = float(tvt_input[anchor_position])
        lower_step = int(np.floor((float(tw_tvt[0]) - anchor_tvt) / 4.0))
        upper_step = int(np.ceil((float(tw_tvt[-1]) - anchor_tvt) / 4.0))
        state_values = anchor_tvt + np.arange(
            lower_step, upper_step + 1, dtype=np.float32
        ) * 4.0
        state_windows = _typewell_windows(tw_tvt, tw_gr, state_values)
        grid_logits = _score_grid_profiles(matcher, horizontal, state_windows)
        base_state = np.rint((base_sampled - anchor_tvt) / 4.0).astype(int) - lower_step
        offset_steps = np.rint(OFFSETS_FT / 4.0).astype(int)
        candidate_indices = base_state[:, None] + offset_steps[None, :]
        candidate_indices = np.clip(candidate_indices, 0, len(state_values) - 1)
        logits = np.take_along_axis(grid_logits, candidate_indices, axis=1)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        clipped_logits = np.clip(logits, -12.0, 0.0)
        probabilities = np.exp(clipped_logits)
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-8)

        expected_offset = probabilities @ OFFSETS_FT
        offset_variance = np.sum(
            probabilities
            * (OFFSETS_FT[None, :] - expected_offset[:, None]) ** 2,
            axis=1,
        )
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        sorted_logits = np.sort(logits, axis=1)
        peak_margin = sorted_logits[:, -1] - sorted_logits[:, -2]

        z = hw["Z"].to_numpy(float)
        gr = hw["GR"].to_numpy(float)
        gr_mask = np.isfinite(gr)
        gr_normalised = _robust_normalise(gr, gr_mask)
        anchor_z = float(z[anchor_position])
        geometry_sampled = anchor_tvt - (z[sampled] - anchor_z)
        row_since = sampled.astype(float) - anchor_position
        frac = row_since / max(float(evaluation[-1] - anchor_position), 1.0)
        ramp = np.clip(row_since / ANCHOR_RAMP_ROWS, 0.0, 1.0)
        previous_base = np.concatenate(([anchor_tvt], base_sampled[:-1]))
        base_delta = base_sampled - previous_base
        previous_z = np.concatenate(([anchor_z], z[sampled[:-1]]))
        z_delta = z[sampled] - previous_z

        extra = np.column_stack(
            [
                expected_offset / 64.0,
                np.sqrt(np.maximum(offset_variance, 0.0)) / 64.0,
                entropy / np.log(len(OFFSETS_FT)),
                peak_margin / 4.0,
                (base_sampled - anchor_tvt) / 100.0,
                np.clip(base_delta / 8.0, -6.0, 6.0),
                np.clip((base_sampled - geometry_sampled) / 32.0, -6.0, 6.0),
                np.clip((z[sampled] - anchor_z) / 100.0, -8.0, 8.0),
                np.clip(z_delta / 8.0, -6.0, 6.0),
                gr_normalised[sampled],
                gr_mask[sampled].astype(float),
                frac,
                np.sqrt(np.clip(frac, 0.0, 1.0)),
                ramp,
            ]
        ).astype(np.float32)
        features = np.concatenate([clipped_logits / 4.0, extra], axis=1).astype(np.float32)

        np.savez_compressed(
            cache_path,
            features=features.astype(np.float16),
            target=(truth_sampled - base_sampled).astype(np.float32),
            ramp=ramp.astype(np.float32),
            sampled=sampled.astype(np.int32),
            evaluation=evaluation.astype(np.int32),
            base_full=base_full.astype(np.float32),
            truth_full=truth_full.astype(np.float32),
            anchor_position=np.int32(anchor_position),
            fold=np.int16(fold),
        )
        return cache_path, None
    except Exception as exc:
        return None, {"well": wid, "fold": fold, "reason": repr(exc)}


# Build fold-honest matcher profiles once.
build_started = time.time()
failures = []
sequence_paths = {}
for fold in range(1, N_FOLDS + 1):
    matcher = _load_matcher(fold)
    fold_wells = sorted(pool.loc[pool["fold"].eq(fold), "well"].unique())
    for wid in tqdm(fold_wells, desc=f"build TCN profiles fold {fold}"):
        group = pool.iloc[pool_group_indices[wid]]
        path, failure = _build_sequence(wid, fold, matcher, group)
        if path is not None:
            sequence_paths[wid] = path
        if failure is not None:
            failures.append(failure)
    del matcher
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

pd.DataFrame(failures).to_csv(CACHE_DIR / "sequence_build_failures.csv", index=False)
if failures:
    raise RuntimeError(f"Failed to build {len(failures)} wells; see sequence_build_failures.csv")
print(
    "sequence cache:",
    {"wells": len(sequence_paths), "elapsed_min": (time.time() - build_started) / 60.0},
)


def _load_record(wid):
    with np.load(sequence_paths[wid]) as data:
        return {key: data[key].copy() for key in data.files}


records = {wid: _load_record(wid) for wid in sorted(sequence_paths)}
N_FEATURES = int(next(iter(records.values()))["features"].shape[1])
print("sequence features:", N_FEATURES)


class WellSequenceDataset(Dataset):
    def __init__(self, wells, augment=False):
        self.wells = list(wells)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.wells)

    def __getitem__(self, index):
        wid = self.wells[index]
        record = records[wid]
        features = record["features"].astype(np.float32)
        target = record["target"].astype(np.float32)
        ramp = record["ramp"].astype(np.float32)
        if self.augment and len(target) > 192:
            minimum = max(128, len(target) // 2)
            stop = np.random.randint(minimum, len(target) + 1)
            features = features[:stop]
            target = target[:stop]
            ramp = ramp[:stop]
        return wid, features, target, ramp


def _collate(batch):
    lengths = [len(item[2]) for item in batch]
    maximum = max(lengths)
    features = np.zeros((len(batch), N_FEATURES, maximum), dtype=np.float32)
    target = np.zeros((len(batch), maximum), dtype=np.float32)
    ramp = np.zeros((len(batch), maximum), dtype=np.float32)
    mask = np.zeros((len(batch), maximum), dtype=np.float32)
    wells = []
    for index, (wid, x, y, r) in enumerate(batch):
        length = len(y)
        features[index, :, :length] = x.T
        target[index, :length] = y
        ramp[index, :length] = r
        mask[index, :length] = 1.0
        wells.append(wid)
    return (
        wells,
        torch.from_numpy(features),
        torch.from_numpy(target),
        torch.from_numpy(ramp),
        torch.from_numpy(mask),
    )


class TCNResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
        )

    def forward(self, values):
        return F.silu(values + self.network(values))


class AnchorAwareTCN(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(input_features, HIDDEN_CHANNELS, 1),
            nn.GroupNorm(8, HIDDEN_CHANNELS),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[
                TCNResidualBlock(HIDDEN_CHANNELS, dilation, DROPOUT)
                for dilation in DILATIONS
            ]
        )
        self.output = nn.Sequential(
            nn.Conv1d(HIDDEN_CHANNELS, 48, 1),
            nn.SiLU(),
            nn.Conv1d(48, 1, 1),
        )

    def forward(self, features, ramp):
        hidden = self.blocks(self.input(features))
        correction = MAX_CORRECTION_FT * torch.tanh(self.output(hidden).squeeze(1))
        return correction * ramp


def _make_loader(wells, augment, shuffle):
    return DataLoader(
        WellSequenceDataset(wells, augment=augment),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=_collate,
        persistent_workers=(NUM_WORKERS > 0),
    )


def _sequence_loss(prediction, target, mask):
    point = F.smooth_l1_loss(prediction, target, beta=4.0, reduction="none")
    point_loss = torch.sum(point * mask) / torch.clamp(mask.sum(), min=1.0)
    if prediction.shape[1] > 1:
        pred_delta = prediction[:, 1:] - prediction[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        delta_mask = mask[:, 1:] * mask[:, :-1]
        delta = F.smooth_l1_loss(pred_delta, target_delta, beta=1.0, reduction="none")
        delta_loss = torch.sum(delta * delta_mask) / torch.clamp(delta_mask.sum(), min=1.0)
    else:
        delta_loss = point_loss * 0.0
    shrink = torch.sum(torch.abs(prediction) * mask) / torch.clamp(mask.sum(), min=1.0)
    return point_loss + 0.15 * delta_loss + 0.002 * shrink


@torch.no_grad()
def _predict_loader(model, loader):
    model.eval()
    output = {}
    for wells, features, _, ramp, mask in loader:
        features = features.to(DEVICE, non_blocking=True)
        ramp = ramp.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            prediction = model(features, ramp).float().cpu().numpy()
        lengths = mask.sum(dim=1).numpy().astype(int)
        for wid, values, length in zip(wells, prediction, lengths):
            output[wid] = values[:length].copy()
    return output


def _full_row_prediction(wid, sampled_correction):
    record = records[wid]
    sampled = record["sampled"].astype(float)
    evaluation = record["evaluation"].astype(float)
    anchor_position = float(record["anchor_position"])
    correction = np.interp(
        evaluation,
        np.concatenate(([anchor_position], sampled)),
        np.concatenate(([0.0], np.asarray(sampled_correction, float))),
    )
    return record["base_full"].astype(float) + correction


# ---------------------------------------------------------------------------
# Five-fold sequence training and full-row OOF prediction.
# ---------------------------------------------------------------------------
fold_metrics = []
oof_parts = []
training_started = time.time()
all_wells = np.array(sorted(records))
fold_by_well = pool.groupby("well")["fold"].first().astype(int).to_dict()

for fold in range(1, N_FOLDS + 1):
    checkpoint = MODEL_DIR / f"tcn_fold{fold}.pt"
    prediction_path = CACHE_DIR / f"tcn_fold{fold}_predictions.pkl"
    metric_path = CACHE_DIR / f"tcn_fold{fold}_metric.json"
    if checkpoint.exists() and prediction_path.exists() and metric_path.exists():
        oof_parts.append(pd.read_pickle(prediction_path))
        fold_metrics.append(json.loads(metric_path.read_text()))
        print("loaded cached TCN fold", fold)
        continue

    train_wells = [wid for wid in all_wells if fold_by_well[wid] != fold]
    valid_wells = [wid for wid in all_wells if fold_by_well[wid] == fold]
    train_loader = _make_loader(train_wells, augment=True, shuffle=True)
    valid_loader = _make_loader(valid_wells, augment=False, shuffle=False)

    model = AnchorAwareTCN(N_FEATURES).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    best_rmse = np.inf
    best_epoch = 0
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        batches = 0
        progress = tqdm(train_loader, desc=f"TCN fold {fold} epoch {epoch}")
        for _, features, target, ramp, mask in progress:
            features = features.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            ramp = ramp.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                prediction = model(features, ramp)
                loss = _sequence_loss(prediction, target, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
            batches += 1
            progress.set_postfix(loss=running / batches)
        scheduler.step()

        sampled_prediction = _predict_loader(model, valid_loader)
        truth_parts = []
        base_parts = []
        pred_parts = []
        for wid in valid_wells:
            record = records[wid]
            truth_parts.append(record["truth_full"].astype(float))
            base_parts.append(record["base_full"].astype(float))
            pred_parts.append(_full_row_prediction(wid, sampled_prediction[wid]))
        truth = np.concatenate(truth_parts)
        base_values = np.concatenate(base_parts)
        pred_values = np.concatenate(pred_parts)
        valid_rmse = _rmse(truth, pred_values)
        base_rmse = _rmse(truth, base_values)
        print(
            f"fold={fold} epoch={epoch} train_loss={running/max(batches,1):.5f} "
            f"base={base_rmse:.5f} valid={valid_rmse:.5f} "
            f"gain={base_rmse-valid_rmse:.5f}",
            flush=True,
        )
        if valid_rmse < best_rmse - 1e-4:
            best_rmse = valid_rmse
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= PATIENCE:
                print("early stop fold", fold, "epoch", epoch)
                break

    try:
        state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state)
    sampled_prediction = _predict_loader(model, valid_loader)
    rows = []
    for wid in valid_wells:
        record = records[wid]
        prediction = _full_row_prediction(wid, sampled_prediction[wid])
        evaluation = record["evaluation"].astype(int)
        rows.append(
            pd.DataFrame(
                {
                    "id": [f"{wid}_{row}" for row in evaluation],
                    "well": wid,
                    "row_idx": evaluation,
                    "fold": fold,
                    "tvt_true": record["truth_full"].astype(float),
                    "base": record["base_full"].astype(float),
                    "tcn": prediction,
                }
            )
        )
    fold_prediction = pd.concat(rows, ignore_index=True)
    fold_prediction.to_pickle(prediction_path)
    metric = {
        "fold": fold,
        "wells": len(valid_wells),
        "rows": len(fold_prediction),
        "best_epoch": best_epoch,
        "base_rmse": _rmse(fold_prediction["tvt_true"], fold_prediction["base"]),
        "tcn_rmse": _rmse(fold_prediction["tvt_true"], fold_prediction["tcn"]),
    }
    metric["gain"] = metric["base_rmse"] - metric["tcn_rmse"]
    metric_path.write_text(json.dumps(metric, indent=2))
    fold_metrics.append(metric)
    oof_parts.append(fold_prediction)
    print("fold metric:", metric)

    del model, optimizer, scheduler, scaler, train_loader, valid_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Frozen global shrink scan. The model itself predicts a capped correction;
# these four weights measure whether additional shrink improves transfer.
# ---------------------------------------------------------------------------
oof = (
    pd.concat(oof_parts, ignore_index=True)
    .sort_values(["well", "row_idx"])
    .reset_index(drop=True)
)
y = oof["tvt_true"].to_numpy(float)
base_prediction = oof["base"].to_numpy(float)
tcn_prediction = oof["tcn"].to_numpy(float)
base_rmse = _rmse(y, base_prediction)

bywell = []
for wid, group in oof.groupby("well", sort=False):
    error = group["base"].to_numpy(float) - group["tvt_true"].to_numpy(float)
    bywell.append({"well": wid, "base_sse": float(np.sum(error * error))})
top40 = set(pd.DataFrame(bywell).nlargest(40, "base_sse")["well"].astype(str))
top40_mask = oof["well"].isin(top40).to_numpy()
top40_base = _rmse(y[top40_mask], base_prediction[top40_mask])

scan_rows = []
fold_scan_rows = []
for weight in APPLY_WEIGHTS:
    prediction = base_prediction + float(weight) * (tcn_prediction - base_prediction)
    fold_gains = []
    for fold, index in oof.groupby("fold", sort=True).groups.items():
        positions = oof.index.get_indexer(index)
        fold_base = _rmse(y[positions], base_prediction[positions])
        fold_score = _rmse(y[positions], prediction[positions])
        gain = fold_base - fold_score
        fold_gains.append(gain)
        fold_scan_rows.append(
            {
                "apply_weight": weight,
                "fold": int(fold),
                "rows": len(positions),
                "base_rmse": fold_base,
                "rmse": fold_score,
                "gain": gain,
            }
        )
    score = _rmse(y, prediction)
    top40_score = _rmse(y[top40_mask], prediction[top40_mask])
    move = prediction - base_prediction
    scan_rows.append(
        {
            "apply_weight": weight,
            "base_rmse": base_rmse,
            "rmse": score,
            "gain": base_rmse - score,
            "mean_fold_gain": float(np.mean(fold_gains)),
            "min_fold_gain": float(np.min(fold_gains)),
            "positive_folds": int(np.sum(np.asarray(fold_gains) > 0)),
            "top40_gain": top40_base - top40_score,
            "mean_abs_move": float(np.mean(np.abs(move))),
            "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
        }
    )

scan = pd.DataFrame(scan_rows)
scan["strict_gate"] = (
    (scan["gain"] >= PASS_GAIN)
    & (scan["positive_folds"] == 5)
    & (scan["min_fold_gain"] >= PASS_MIN_FOLD_GAIN)
    & (scan["top40_gain"] > 0.0)
)
scan = scan.sort_values(["strict_gate", "gain"], ascending=[False, False])
fold_scan = pd.DataFrame(fold_scan_rows)
strict_pass = bool(scan["strict_gate"].any())

print("\nTRAINING FOLDS")
display(pd.DataFrame(fold_metrics))
print("\nAPPLY SCAN")
display(scan)
print("\nAPPLY FOLDS")
display(fold_scan)
print(
    "\nVERDICT",
    {
        "base_rmse": base_rmse,
        "training_hours": (time.time() - training_started) / 3600.0,
        "strict_pass": strict_pass,
        "verdict": (
            "PASS -> retrain final sequence model and build deployment"
            if strict_pass
            else "FAIL -> stop TCN residual decoder"
        ),
    },
)

pd.DataFrame(fold_metrics).to_csv(CACHE_DIR / "training_fold_metrics.csv", index=False)
scan.to_csv(CACHE_DIR / "apply_scan.csv", index=False)
fold_scan.to_csv(CACHE_DIR / "apply_fold_metrics.csv", index=False)
oof.to_pickle(CACHE_DIR / "anchor_aware_tcn_oof.pkl")
(CACHE_DIR / "verdict.json").write_text(
    json.dumps(
        {
            "base_rmse": base_rmse,
            "training_hours": (time.time() - training_started) / 3600.0,
            "strict_pass": strict_pass,
        },
        indent=2,
    )
)
print("saved:", CACHE_DIR)
