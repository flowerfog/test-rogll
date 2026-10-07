"""Global-hard-negative GR matcher v1 with a full-grid OOF gate.

Run after ``learned_gr_matcher_v0_train_cell.py``.  The v0 matcher learned
local ordering but failed global retrieval because it never saw far-away
repeated-pattern negatives.  This cell fixes that exact train/deploy mismatch:

* positive window at true TVT;
* ten fixed local negatives at +/-4, 8, 16, 32, and 64 ft;
* eight deterministic random negatives from the full assigned typewell;
* eight far hard negatives mined by unsupervised full-grid NCC.

The v1 model is not sent to DP automatically.  Its final gate is five-fold
full-grid retrieval against orientation-invariant NCC on the same state grid.
Truth is used to construct supervised candidates and evaluate retrieval, never
to narrow the full-grid validation search.
"""

from pathlib import Path
import gc
import json
import random
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Frozen experiment configuration.
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
V0_DIR = OOF_ROOT / "learned_gr_matcher_v0_win65_rows128_offsets4to64"
V0_MODEL_DIR = V0_DIR / "models"
V1_DIR = OOF_ROOT / "learned_gr_matcher_v1_global_ncc_hardneg27"
V1_MODEL_DIR = V1_DIR / "models"

H_PATH = V0_DIR / "horizontal_windows.npy"
V0_META_PATH = V0_DIR / "sample_metadata.csv"
V0_BUILD_INFO_PATH = V0_DIR / "build_info.json"
V1_TW_PATH = V1_DIR / "typewell_candidate_windows.npy"
V1_CENTER_PATH = V1_DIR / "candidate_centres.npy"
V1_BUILD_INFO_PATH = V1_DIR / "build_info.json"
V1_FAILURE_PATH = V1_DIR / "build_failures.csv"

MULTICUT_DIR = OOF_ROOT / "multicut_fleongg_v0_k3_nospatial_r768_lgb1"
MULTICUT_BYWELL = MULTICUT_DIR / "multicut_v0_bywell.csv"
MULTICUT_OOF = MULTICUT_DIR / "multicut_v0_oof.pkl"

WINDOW_LENGTH = 65
HALF_WINDOW = WINDOW_LENGTH // 2
LOCAL_OFFSETS_FT = np.array(
    [-64.0, -32.0, -16.0, -8.0, -4.0, 4.0, 8.0, 16.0, 32.0, 64.0],
    dtype=np.float32,
)
N_MINED = 8
N_RANDOM = 8
N_CANDIDATES = 1 + len(LOCAL_OFFSETS_FT) + N_MINED + N_RANDOM
POSITIVE_INDEX = 0
GLOBAL_GRID_STEP_FT = 4.0
FAR_NEGATIVE_MIN_FT = 80.0
MINED_DIVERSITY_FT = 8.0

N_FOLDS = 5
BATCH_SIZE = 192
EPOCHS = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
ENCODER_BATCH = 2048
PAIR_ROW_CHUNK = 48
RANDOM_STATE = 20260713

# This gate is checked on full-grid OOF retrieval, not on the 27-way training
# candidate set.  Every fold must beat NCC in both tolerances.
PASS_WITHIN4_GAIN = 0.05
PASS_WITHIN8_GAIN = 0.08
PASS_TOP40_WITHIN4_GAIN = 0.03

for directory in (V1_DIR, V1_MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

required_paths = [H_PATH, V0_META_PATH, V0_BUILD_INFO_PATH]
required_paths += [V0_MODEL_DIR / f"matcher_fold{fold}.pt" for fold in range(1, 6)]
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Run learned_gr_matcher_v0_train_cell.py first. Missing: "
        + repr(missing_paths)
    )

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(
    {
        "experiment": "learned_gr_matcher_v1_global_hardneg",
        "device": str(DEVICE),
        "candidates": N_CANDIDATES,
        "local_negatives": len(LOCAL_OFFSETS_FT),
        "mined_negatives": N_MINED,
        "random_negatives": N_RANDOM,
        "full_grid_step_ft": GLOBAL_GRID_STEP_FT,
        "epochs": EPOCHS,
        "cache": str(V1_DIR),
    }
)


def _fill_1d(values):
    return (
        pd.Series(np.asarray(values, float))
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _normalise_window(values, mask):
    values = np.asarray(values, np.float32)
    mask = np.asarray(mask, bool)
    output = np.zeros_like(values, dtype=np.float32)
    finite_values = values[mask & np.isfinite(values)]
    if len(finite_values) < 4:
        return output
    median = float(np.median(finite_values))
    scale = float(1.4826 * np.median(np.abs(finite_values - median)))
    if scale < 1e-3:
        scale = float(np.std(finite_values) + 1e-3)
    output[mask] = np.clip((values[mask] - median) / scale, -6.0, 6.0)
    return output


def _prepare_typewell(path):
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    frame = frame[np.isfinite(frame["TVT"])].sort_values("TVT")
    frame = frame.groupby("TVT", as_index=False)["GR"].mean()
    tvt = frame["TVT"].to_numpy(float)
    gr = _fill_1d(frame["GR"].to_numpy(float))
    if len(tvt) < WINDOW_LENGTH or np.ptp(tvt) < 200:
        raise RuntimeError("typewell coverage is too short")
    return tvt, gr


def _make_typewell_windows(tw_tvt, tw_gr, centres):
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


def _full_state_grid(tw_tvt):
    lower = float(tw_tvt[0] + HALF_WINDOW)
    upper = float(tw_tvt[-1] - HALF_WINDOW)
    if upper <= lower:
        raise RuntimeError("no full-window typewell state range")
    count = int(np.floor((upper - lower) / GLOBAL_GRID_STEP_FT)) + 1
    return lower + np.arange(count, dtype=np.float32) * GLOBAL_GRID_STEP_FT


# ---------------------------------------------------------------------------
# Exact v0/v1 architecture.
# ---------------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        padding = 2 * dilation
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation),
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
            ResidualConvBlock(48, 1),
            ResidualConvBlock(48, 2),
            ResidualConvBlock(48, 4),
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
        horizontal = horizontal[:, None, :].expand_as(typewell)
        pair = torch.cat(
            [horizontal, typewell, torch.abs(horizontal - typewell), horizontal * typewell],
            dim=-1,
        )
        return self.scorer(pair).squeeze(-1)

    def forward(self, horizontal, typewell):
        batch, candidates, channels, length = typewell.shape
        horizontal_embedding = self.horizontal_encoder(horizontal)
        flat = typewell.reshape(batch * candidates, channels, length)
        forward = self.typewell_encoder(flat).reshape(batch, candidates, -1)
        reverse = torch.flip(typewell, dims=[-1]).reshape(
            batch * candidates, channels, length
        )
        reverse = self.typewell_encoder(reverse).reshape(batch, candidates, -1)
        return torch.maximum(
            self._score_embeddings(horizontal_embedding, forward),
            self._score_embeddings(horizontal_embedding, reverse),
        )


def _load_model(path):
    model = GRWindowMatcher().to(DEVICE)
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _encode(encoder, windows):
    parts = []
    for start in range(0, len(windows), ENCODER_BATCH):
        tensor = torch.from_numpy(
            np.asarray(windows[start : start + ENCODER_BATCH], np.float32)
        ).to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            parts.append(encoder(tensor).float().cpu())
    return torch.cat(parts).to(DEVICE, non_blocking=True)


def _score_pair_embeddings(model, horizontal, typewell):
    horizontal = horizontal[:, None, :].expand(-1, typewell.shape[1], -1)
    pair = torch.cat(
        [horizontal, typewell, torch.abs(horizontal - typewell), horizontal * typewell],
        dim=-1,
    )
    return model.scorer(pair).squeeze(-1)


@torch.no_grad()
def _score_grid(model, horizontal_windows, typewell_windows):
    horizontal_embedding = _encode(model.horizontal_encoder, horizontal_windows)
    tw_forward = _encode(model.typewell_encoder, typewell_windows)
    tw_reverse = _encode(
        model.typewell_encoder, typewell_windows[:, :, ::-1].copy()
    )
    parts = []
    for start in range(0, len(horizontal_embedding), PAIR_ROW_CHUNK):
        horizontal = horizontal_embedding[start : start + PAIR_ROW_CHUNK]
        rows = len(horizontal)
        forward = tw_forward[None].expand(rows, -1, -1)
        reverse = tw_reverse[None].expand(rows, -1, -1)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            score = torch.maximum(
                _score_pair_embeddings(model, horizontal, forward),
                _score_pair_embeddings(model, horizontal, reverse),
            )
        parts.append(score.float().cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


@torch.no_grad()
def _ncc_grid(horizontal_windows, typewell_windows):
    tw = torch.from_numpy(np.asarray(typewell_windows, np.float32)).to(DEVICE)
    parts = []
    for start in range(0, len(horizontal_windows), PAIR_ROW_CHUNK):
        horizontal = torch.from_numpy(
            np.asarray(horizontal_windows[start : start + PAIR_ROW_CHUNK], np.float32)
        ).to(DEVICE)
        h_value = horizontal[:, None, 0, :]
        h_mask = horizontal[:, None, 1, :]

        def score(candidate):
            tw_value = candidate[None, :, 0, :]
            tw_mask = candidate[None, :, 1, :]
            mask = h_mask * tw_mask
            numerator = torch.sum(mask * h_value * tw_value, dim=-1)
            denominator = torch.sqrt(
                torch.sum(mask * h_value * h_value, dim=-1)
                * torch.sum(mask * tw_value * tw_value, dim=-1)
                + 1e-8
            )
            return numerator / denominator

        forward = score(tw)
        reverse = score(torch.flip(tw, dims=[-1]))
        parts.append(torch.maximum(forward, reverse).float().cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


sample_meta = pd.read_csv(V0_META_PATH)
build_info = json.loads(V0_BUILD_INFO_PATH.read_text())
sample_count = int(build_info["samples"])
max_samples = int(build_info["max_samples"])
horizontal_windows = np.load(H_PATH, mmap_mode="r")
if len(sample_meta) != sample_count:
    raise RuntimeError("v0 sample metadata/cache mismatch")
if sample_meta["sample_index"].duplicated().any():
    raise RuntimeError("sample_index must be unique")


def _select_diverse_hard(states, scores, truth, count):
    order = np.argsort(scores)[::-1]
    selected = []
    for index in order:
        centre = float(states[index])
        if abs(centre - truth) < FAR_NEGATIVE_MIN_FT:
            continue
        if any(abs(centre - previous) < MINED_DIVERSITY_FT for previous in selected):
            continue
        selected.append(centre)
        if len(selected) == count:
            break
    return selected


def _select_random_global(states, truth, excluded, count, seed):
    eligible = np.asarray(
        [
            value
            for value in states
            if abs(float(value) - truth) >= FAR_NEGATIVE_MIN_FT
            and all(abs(float(value) - old) >= GLOBAL_GRID_STEP_FT for old in excluded)
        ],
        dtype=np.float32,
    )
    if not len(eligible):
        eligible = states[np.abs(states - truth) >= 2 * GLOBAL_GRID_STEP_FT]
    if not len(eligible):
        eligible = states
    rng = np.random.default_rng(int(seed))
    replace = len(eligible) < count
    return rng.choice(eligible, size=count, replace=replace).astype(float).tolist()


def _build_hard_negative_cache():
    typewell_store = open_memmap(
        V1_TW_PATH,
        mode="w+",
        dtype=np.float16,
        shape=(max_samples, N_CANDIDATES, 2, WINDOW_LENGTH),
    )
    centre_store = open_memmap(
        V1_CENTER_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(max_samples, N_CANDIDATES),
    )
    built = np.zeros(max_samples, dtype=bool)
    failures = []
    started = time.time()

    wells = sorted(sample_meta["well"].astype(str).unique())
    for wid in tqdm(wells, desc="mine full-grid NCC hard negatives"):
        group = sample_meta[sample_meta["well"].astype(str).eq(wid)].sort_values(
            "sample_index"
        )
        indices = group["sample_index"].to_numpy(int)
        truths = group["true_tvt"].to_numpy(float)
        fold = int(group["fold"].iloc[0])
        try:
            tw_tvt, tw_gr = _prepare_typewell(
                DATA_ROOT / "train" / f"{wid}__typewell.csv"
            )
            states = _full_state_grid(tw_tvt)
            state_windows = _make_typewell_windows(tw_tvt, tw_gr, states)
            scores = _ncc_grid(
                np.asarray(horizontal_windows[indices], np.float32),
                state_windows,
            )

            for row, (sample_index, truth) in enumerate(zip(indices, truths)):
                mined = _select_diverse_hard(
                    states, scores[row], float(truth), N_MINED
                )
                if len(mined) < N_MINED:
                    fill = _select_random_global(
                        states,
                        float(truth),
                        mined,
                        N_MINED - len(mined),
                        RANDOM_STATE + int(sample_index),
                    )
                    mined.extend(fill)
                seed = (
                    RANDOM_STATE
                    + zlib.crc32(wid.encode("ascii"))
                    + int(sample_index) * 17
                ) & 0xFFFFFFFF
                random_global = _select_random_global(
                    states, float(truth), mined, N_RANDOM, seed
                )
                centres = np.asarray(
                    [float(truth)]
                    + (float(truth) + LOCAL_OFFSETS_FT).tolist()
                    + mined
                    + random_global,
                    dtype=np.float32,
                )
                if len(centres) != N_CANDIDATES:
                    raise RuntimeError("candidate count mismatch")
                windows = _make_typewell_windows(tw_tvt, tw_gr, centres)
                typewell_store[sample_index] = windows.astype(np.float16)
                centre_store[sample_index] = centres
                built[sample_index] = True
        except Exception as exc:
            failures.append({"well": wid, "fold": fold, "reason": repr(exc)})

    typewell_store.flush()
    centre_store.flush()
    del typewell_store, centre_store

    required_indices = sample_meta["sample_index"].to_numpy(int)
    missing = required_indices[~built[required_indices]]
    pd.DataFrame(failures).to_csv(V1_FAILURE_PATH, index=False)
    if len(missing):
        raise RuntimeError(
            f"Failed to build {len(missing)} supervised rows; see {V1_FAILURE_PATH}"
        )
    V1_BUILD_INFO_PATH.write_text(
        json.dumps(
            {
                "samples": sample_count,
                "max_samples": max_samples,
                "candidates": N_CANDIDATES,
                "window_length": WINDOW_LENGTH,
                "elapsed_hours": (time.time() - started) / 3600.0,
            },
            indent=2,
        )
    )
    print(
        "hard-negative cache built:",
        {"samples": sample_count, "failures": len(failures)},
    )


if not all(path.exists() for path in [V1_TW_PATH, V1_CENTER_PATH, V1_BUILD_INFO_PATH]):
    _build_hard_negative_cache()

v1_info = json.loads(V1_BUILD_INFO_PATH.read_text())
if int(v1_info["candidates"]) != N_CANDIDATES:
    raise RuntimeError("stale v1 candidate cache; remove V1_DIR and rebuild")
typewell_candidates = np.load(V1_TW_PATH, mmap_mode="r")
candidate_centres = np.load(V1_CENTER_PATH, mmap_mode="r")
print("training cache:", typewell_candidates.shape, candidate_centres.shape)


class MatcherDataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        horizontal = np.asarray(horizontal_windows[index], dtype=np.float32)
        typewell = np.asarray(typewell_candidates[index], dtype=np.float32)
        return torch.from_numpy(horizontal), torch.from_numpy(typewell), index


def _make_loader(indices, shuffle):
    return DataLoader(
        MatcherDataset(indices),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=shuffle,
    )


@torch.no_grad()
def _evaluate_candidate_set(model, loader):
    model.eval()
    predictions = []
    indices = []
    loss_sum = 0.0
    for horizontal, typewell, sample_indices in loader:
        horizontal = horizontal.to(DEVICE, non_blocking=True)
        typewell = typewell.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits = model(horizontal, typewell)
            labels = torch.zeros(len(horizontal), dtype=torch.long, device=DEVICE)
            loss = F.cross_entropy(logits, labels)
        predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
        indices.append(sample_indices.numpy())
        loss_sum += float(loss.item()) * len(horizontal)
    return {
        "sample_index": np.concatenate(indices),
        "candidate": np.concatenate(predictions),
        "loss": loss_sum / max(1, len(loader.dataset)),
    }


# ---------------------------------------------------------------------------
# Fold-honest v1 training.  Fine-tuning starts from the corresponding v0 OOF
# checkpoint; fold f remains unseen by both the initializer and v1 training.
# ---------------------------------------------------------------------------
training_rows = []
training_started = time.time()
for fold in range(1, N_FOLDS + 1):
    checkpoint = V1_MODEL_DIR / f"matcher_fold{fold}.pt"
    metric_path = V1_DIR / f"candidate_set_fold{fold}.json"
    if checkpoint.exists() and metric_path.exists():
        metric = json.loads(metric_path.read_text())
        training_rows.append(metric)
        print("loaded trained fold", fold, metric)
        continue

    train_indices = sample_meta.loc[
        sample_meta["fold"].ne(fold), "sample_index"
    ].to_numpy(int)
    valid_indices = sample_meta.loc[
        sample_meta["fold"].eq(fold), "sample_index"
    ].to_numpy(int)
    train_loader = _make_loader(train_indices, True)
    valid_loader = _make_loader(valid_indices, False)
    model = _load_model(V0_MODEL_DIR / f"matcher_fold{fold}.pt")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"v1 fold {fold} epoch {epoch}")
        for horizontal, typewell, _ in progress:
            horizontal = horizontal.to(DEVICE, non_blocking=True)
            typewell = typewell.to(DEVICE, non_blocking=True)
            labels = torch.zeros(len(horizontal), dtype=torch.long, device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(horizontal, typewell)
                loss = F.cross_entropy(logits, labels, label_smoothing=0.02)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * len(horizontal)
            seen += len(horizontal)
            progress.set_postfix(loss=running / max(1, seen))

        validation = _evaluate_candidate_set(model, valid_loader)
        print(
            f"fold={fold} epoch={epoch} train_loss={running/max(1,seen):.5f} "
            f"valid_loss={validation['loss']:.5f}",
            flush=True,
        )

    torch.save(model.state_dict(), checkpoint)
    validation = _evaluate_candidate_set(model, valid_loader)
    chosen = candidate_centres[
        validation["sample_index"], validation["candidate"]
    ]
    truth = sample_meta.set_index("sample_index").loc[
        validation["sample_index"], "true_tvt"
    ].to_numpy(float)
    absolute = np.abs(chosen - truth)
    metric = {
        "fold": fold,
        "samples": len(valid_indices),
        "candidate_set_within4": float(np.mean(absolute <= 4.0)),
        "candidate_set_within8": float(np.mean(absolute <= 8.0)),
        "candidate_set_median_abs": float(np.median(absolute)),
        "valid_loss": float(validation["loss"]),
    }
    metric_path.write_text(json.dumps(metric, indent=2))
    training_rows.append(metric)
    print("candidate-set metric:", metric)

    del model, optimizer, scaler, train_loader, valid_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Decisive full-grid OOF retrieval.  The candidate set is the complete
# full-window typewell range at 4 ft spacing for every held-out sample.
# ---------------------------------------------------------------------------
fullgrid_records = []
fullgrid_started = time.time()
for fold in range(1, N_FOLDS + 1):
    prediction_path = V1_DIR / f"fullgrid_fold{fold}.pkl"
    if prediction_path.exists():
        fullgrid_records.append(pd.read_pickle(prediction_path))
        print("loaded full-grid fold", fold)
        continue

    model = _load_model(V1_MODEL_DIR / f"matcher_fold{fold}.pt")
    fold_meta = sample_meta[sample_meta["fold"].eq(fold)]
    parts = []
    for wid in tqdm(
        sorted(fold_meta["well"].astype(str).unique()),
        desc=f"full-grid retrieval fold {fold}",
    ):
        group = fold_meta[fold_meta["well"].astype(str).eq(wid)].sort_values(
            "sample_index"
        )
        indices = group["sample_index"].to_numpy(int)
        truths = group["true_tvt"].to_numpy(float)
        tw_tvt, tw_gr = _prepare_typewell(
            DATA_ROOT / "train" / f"{wid}__typewell.csv"
        )
        states = _full_state_grid(tw_tvt)
        state_windows = _make_typewell_windows(tw_tvt, tw_gr, states)
        horizontal = np.asarray(horizontal_windows[indices], np.float32)
        learned_score = _score_grid(model, horizontal, state_windows)
        ncc_score = _ncc_grid(horizontal, state_windows)
        learned_tvt = states[np.argmax(learned_score, axis=1)]
        ncc_tvt = states[np.argmax(ncc_score, axis=1)]
        parts.append(
            pd.DataFrame(
                {
                    "sample_index": indices,
                    "well": wid,
                    "fold": fold,
                    "true_tvt": truths,
                    "learned_tvt": learned_tvt,
                    "ncc_tvt": ncc_tvt,
                    "learned_abs": np.abs(learned_tvt - truths),
                    "ncc_abs": np.abs(ncc_tvt - truths),
                }
            )
        )
    fold_prediction = pd.concat(parts, ignore_index=True)
    fold_prediction.to_pickle(prediction_path)
    fullgrid_records.append(fold_prediction)
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

fullgrid = pd.concat(fullgrid_records, ignore_index=True)
if len(fullgrid) != sample_count:
    raise RuntimeError(f"full-grid rows mismatch: {len(fullgrid)} != {sample_count}")

fold_rows = []
for fold, group in fullgrid.groupby("fold", sort=True):
    fold_rows.append(
        {
            "fold": int(fold),
            "samples": len(group),
            "wells": group["well"].nunique(),
            "learned_within4": float(np.mean(group["learned_abs"] <= 4.0)),
            "ncc_within4": float(np.mean(group["ncc_abs"] <= 4.0)),
            "within4_gain": float(
                np.mean(group["learned_abs"] <= 4.0)
                - np.mean(group["ncc_abs"] <= 4.0)
            ),
            "learned_within8": float(np.mean(group["learned_abs"] <= 8.0)),
            "ncc_within8": float(np.mean(group["ncc_abs"] <= 8.0)),
            "within8_gain": float(
                np.mean(group["learned_abs"] <= 8.0)
                - np.mean(group["ncc_abs"] <= 8.0)
            ),
            "learned_median_abs": float(np.median(group["learned_abs"])),
            "ncc_median_abs": float(np.median(group["ncc_abs"])),
        }
    )
fold_report = pd.DataFrame(fold_rows)

top40 = set()
if MULTICUT_BYWELL.exists():
    bywell = pd.read_csv(MULTICUT_BYWELL)
    if {"well", "single_sse"} <= set(bywell.columns):
        top40 = set(bywell.nlargest(40, "single_sse")["well"].astype(str))
if not top40 and MULTICUT_OOF.exists():
    base_oof = pd.read_pickle(MULTICUT_OOF)
    if {"well", "tvt_true", "pred_single_cut"} <= set(base_oof.columns):
        base_oof = base_oof[["well", "tvt_true", "pred_single_cut"]].copy()
        base_oof["single_sqerr"] = (
            base_oof["pred_single_cut"].to_numpy(float)
            - base_oof["tvt_true"].to_numpy(float)
        ) ** 2
        bywell = base_oof.groupby("well", as_index=False)["single_sqerr"].sum()
        top40 = set(bywell.nlargest(40, "single_sqerr")["well"].astype(str))
top = fullgrid[fullgrid["well"].isin(top40)] if top40 else fullgrid.iloc[:0]

overall = {
    "samples": len(fullgrid),
    "wells": fullgrid["well"].nunique(),
    "learned_within4": float(np.mean(fullgrid["learned_abs"] <= 4.0)),
    "ncc_within4": float(np.mean(fullgrid["ncc_abs"] <= 4.0)),
    "within4_gain": float(
        np.mean(fullgrid["learned_abs"] <= 4.0)
        - np.mean(fullgrid["ncc_abs"] <= 4.0)
    ),
    "learned_within8": float(np.mean(fullgrid["learned_abs"] <= 8.0)),
    "ncc_within8": float(np.mean(fullgrid["ncc_abs"] <= 8.0)),
    "within8_gain": float(
        np.mean(fullgrid["learned_abs"] <= 8.0)
        - np.mean(fullgrid["ncc_abs"] <= 8.0)
    ),
    "learned_median_abs": float(np.median(fullgrid["learned_abs"])),
    "ncc_median_abs": float(np.median(fullgrid["ncc_abs"])),
    "top40_samples": len(top),
    "top40_learned_within4": (
        float(np.mean(top["learned_abs"] <= 4.0)) if len(top) else np.nan
    ),
    "top40_ncc_within4": (
        float(np.mean(top["ncc_abs"] <= 4.0)) if len(top) else np.nan
    ),
    "training_hours": (time.time() - training_started) / 3600.0,
    "fullgrid_hours": (time.time() - fullgrid_started) / 3600.0,
}
overall["top40_within4_gain"] = (
    overall["top40_learned_within4"] - overall["top40_ncc_within4"]
)
overall["strict_pass"] = bool(
    overall["within4_gain"] >= PASS_WITHIN4_GAIN
    and overall["within8_gain"] >= PASS_WITHIN8_GAIN
    and overall["top40_within4_gain"] >= PASS_TOP40_WITHIN4_GAIN
    and (fold_report["within4_gain"] > 0).all()
    and (fold_report["within8_gain"] > 0).all()
)
overall["verdict"] = (
    "PASS -> rerun learned-emission DP with v1"
    if overall["strict_pass"]
    else "FAIL -> stop matcher/DP; retain scores only as optional tabular features"
)

print("\nCANDIDATE-SET TRAINING")
display(pd.DataFrame(training_rows))
print("\nFULL-GRID FOLDS")
display(fold_report)
print("\nFULL-GRID OVERALL")
print(overall)
print("\nVERDICT:", overall["verdict"])

pd.DataFrame(training_rows).to_csv(V1_DIR / "candidate_set_training.csv", index=False)
fold_report.to_csv(V1_DIR / "fullgrid_fold_report.csv", index=False)
fullgrid.to_pickle(V1_DIR / "fullgrid_oof_predictions.pkl")
(V1_DIR / "fullgrid_verdict.json").write_text(json.dumps(overall, indent=2))
print("saved:", V1_DIR)
