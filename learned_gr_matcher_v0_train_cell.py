"""Five-fold supervised horizontal/typewell GR window matcher.

This standalone AutoDL cell learns an emission function, not TVT itself. For
each held-out-row example, the model must rank the true typewell TVT window
above ten hard negatives at +/-4, 8, 16, 32, and 64 ft. Splits are by original
well. Validation examples come only from each well's official eval suffix.

The strict gate compares the learned matcher against an orientation-invariant
normalised cross-correlation baseline on exactly the same candidate windows.
No DP path is decoded unless this matcher passes.
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
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Fixed v0 configuration.
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
CACHE_VERSION = "v0_win65_rows128_offsets4to64"
CACHE_DIR = OOF_ROOT / f"learned_gr_matcher_{CACHE_VERSION}"
MODEL_DIR = CACHE_DIR / "models"

MULTICUT_DIR = OOF_ROOT / "multicut_fleongg_v0_k3_nospatial_r768_lgb1"
MULTICUT_META = MULTICUT_DIR / "well_metadata.csv"
MULTICUT_BYWELL = MULTICUT_DIR / "multicut_v0_bywell.csv"
REPORT_BACKUP = OOF_ROOT / "keep_reports_20260713"

N_FOLDS = 5
ROWS_PER_WELL = 128
WINDOW_LENGTH = 65
HALF_WINDOW = WINDOW_LENGTH // 2
OFFSETS_FT = np.array(
    [-64.0, -32.0, -16.0, -8.0, -4.0, 0.0, 4.0, 8.0, 16.0, 32.0, 64.0],
    dtype=np.float32,
)
POSITIVE_INDEX = int(np.flatnonzero(OFFSETS_FT == 0.0)[0])
MIN_HORIZONTAL_VALID_FRAC = 0.35

BATCH_SIZE = 256
EPOCHS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
RANDOM_STATE = 20260713

PASS_WITHIN4_GAIN = 0.10
PASS_WITHIN8_GAIN = 0.08

for directory in (CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    {
        "experiment": "learned_gr_matcher_v0",
        "device": str(DEVICE),
        "window_length": WINDOW_LENGTH,
        "rows_per_well": ROWS_PER_WELL,
        "offsets_ft": OFFSETS_FT.tolist(),
        "epochs": EPOCHS,
        "cache": str(CACHE_DIR),
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


def _evenly_spaced(index, count):
    index = np.asarray(index, dtype=int)
    if len(index) <= count:
        return index
    positions = np.unique(
        np.round(np.linspace(0, len(index) - 1, int(count))).astype(int)
    )
    return index[positions]


def _load_fold_map(wells):
    candidates = [
        MULTICUT_META,
        REPORT_BACKUP / "multicut_v0__well_metadata.csv",
    ]
    for path in candidates:
        if path.exists():
            meta = pd.read_csv(path)
            if {"well", "fold"} <= set(meta.columns):
                mapping = dict(zip(meta["well"].astype(str), meta["fold"].astype(int)))
                if all(well in mapping for well in wells):
                    print("loaded fold map:", path)
                    return mapping
    # Deterministic fallback. Every well contributes at most ROWS_PER_WELL rows.
    return {well: 1 + index % N_FOLDS for index, well in enumerate(sorted(wells))}


def _find_top40_wells():
    candidates = [
        MULTICUT_BYWELL,
        REPORT_BACKUP / "multicut_v0__multicut_v0_bywell.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            if {"well", "single_sse"} <= set(frame.columns):
                return set(
                    frame.nlargest(40, "single_sse")["well"].astype(str)
                )
    return set()


H_PATH = CACHE_DIR / "horizontal_windows.npy"
TW_PATH = CACHE_DIR / "typewell_candidate_windows.npy"
SAMPLE_META_PATH = CACHE_DIR / "sample_metadata.csv"
BUILD_INFO_PATH = CACHE_DIR / "build_info.json"


def _build_window_cache():
    horizontal_files = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
    wells = [path.name.replace("__horizontal_well.csv", "") for path in horizontal_files]
    fold_map = _load_fold_map(wells)

    max_samples = len(wells) * ROWS_PER_WELL
    horizontal_store = open_memmap(
        H_PATH,
        mode="w+",
        dtype=np.float16,
        shape=(max_samples, 2, WINDOW_LENGTH),
    )
    typewell_store = open_memmap(
        TW_PATH,
        mode="w+",
        dtype=np.float16,
        shape=(max_samples, len(OFFSETS_FT), 2, WINDOW_LENGTH),
    )

    relative_window = np.arange(-HALF_WINDOW, HALF_WINDOW + 1, dtype=np.float32)
    metadata = []
    cursor = 0
    failures = []

    for path in tqdm(horizontal_files, desc="build GR matcher windows"):
        wid = path.name.replace("__horizontal_well.csv", "")
        typewell_path = DATA_ROOT / "train" / f"{wid}__typewell.csv"
        try:
            hw = pd.read_csv(path, usecols=["GR", "TVT", "TVT_input"])
            tw = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
            tw = tw[np.isfinite(tw["TVT"])].sort_values("TVT")
            tw = tw.groupby("TVT", as_index=False)["GR"].mean()

            horizontal_gr = hw["GR"].to_numpy(float)
            truth_tvt = hw["TVT"].to_numpy(float)
            tvt_input = hw["TVT_input"].to_numpy(float)
            tw_tvt = tw["TVT"].to_numpy(float)
            tw_gr = _fill_1d(tw["GR"].to_numpy(float))

            if len(tw_tvt) < WINDOW_LENGTH or np.ptp(tw_tvt) < 200:
                raise RuntimeError("typewell coverage too short")

            evaluation = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth_tvt))
            if len(evaluation) < 64:
                raise RuntimeError("too few official eval rows")

            max_extent = float(np.max(np.abs(OFFSETS_FT)) + HALF_WINDOW)
            candidate_rows = evaluation[
                (truth_tvt[evaluation] >= tw_tvt[0] + max_extent)
                & (truth_tvt[evaluation] <= tw_tvt[-1] - max_extent)
            ]

            valid_rows = []
            for row_index in candidate_rows:
                left = max(0, int(row_index) - HALF_WINDOW)
                right = min(len(horizontal_gr), int(row_index) + HALF_WINDOW + 1)
                valid_fraction = float(np.mean(np.isfinite(horizontal_gr[left:right])))
                if valid_fraction >= MIN_HORIZONTAL_VALID_FRAC:
                    valid_rows.append(int(row_index))
            selected = _evenly_spaced(valid_rows, ROWS_PER_WELL)
            if len(selected) < 32:
                raise RuntimeError("too few valid matcher centres")

            for row_index in selected:
                row_positions = row_index + np.arange(-HALF_WINDOW, HALF_WINDOW + 1)
                inside = (row_positions >= 0) & (row_positions < len(horizontal_gr))
                clipped = np.clip(row_positions, 0, len(horizontal_gr) - 1)
                h_values = horizontal_gr[clipped]
                h_mask = inside & np.isfinite(h_values)
                h_normalised = _normalise_window(h_values, h_mask)

                candidate_centres = truth_tvt[row_index] + OFFSETS_FT
                queries = candidate_centres[:, None] + relative_window[None, :]
                tw_mask = (queries >= tw_tvt[0]) & (queries <= tw_tvt[-1])
                tw_values = np.interp(queries.reshape(-1), tw_tvt, tw_gr).reshape(
                    len(OFFSETS_FT), WINDOW_LENGTH
                )
                tw_normalised = np.zeros_like(tw_values, dtype=np.float32)
                for candidate in range(len(OFFSETS_FT)):
                    tw_normalised[candidate] = _normalise_window(
                        tw_values[candidate], tw_mask[candidate]
                    )

                horizontal_store[cursor, 0] = h_normalised.astype(np.float16)
                horizontal_store[cursor, 1] = h_mask.astype(np.float16)
                typewell_store[cursor, :, 0] = tw_normalised.astype(np.float16)
                typewell_store[cursor, :, 1] = tw_mask.astype(np.float16)
                metadata.append(
                    {
                        "sample_index": cursor,
                        "well": wid,
                        "fold": int(fold_map[wid]),
                        "row_idx": int(row_index),
                        "true_tvt": float(truth_tvt[row_index]),
                    }
                )
                cursor += 1
        except Exception as exc:
            failures.append({"well": wid, "reason": repr(exc)})

    horizontal_store.flush()
    typewell_store.flush()
    del horizontal_store, typewell_store

    sample_meta = pd.DataFrame(metadata)
    sample_meta.to_csv(SAMPLE_META_PATH, index=False)
    pd.DataFrame(failures).to_csv(CACHE_DIR / "build_failures.csv", index=False)
    BUILD_INFO_PATH.write_text(
        json.dumps(
            {
                "samples": int(cursor),
                "max_samples": int(max_samples),
                "wells": int(sample_meta["well"].nunique()),
                "failures": int(len(failures)),
                "horizontal_shape": [max_samples, 2, WINDOW_LENGTH],
                "typewell_shape": [
                    max_samples,
                    len(OFFSETS_FT),
                    2,
                    WINDOW_LENGTH,
                ],
            },
            indent=2,
        )
    )
    print("window cache:", {"samples": cursor, "failures": len(failures)})


if not all(path.exists() for path in [H_PATH, TW_PATH, SAMPLE_META_PATH, BUILD_INFO_PATH]):
    _build_window_cache()

build_info = json.loads(BUILD_INFO_PATH.read_text())
sample_meta = pd.read_csv(SAMPLE_META_PATH)
sample_count = int(build_info["samples"])
max_samples = int(build_info["max_samples"])

horizontal_windows = np.load(H_PATH, mmap_mode="r")
typewell_windows = np.load(TW_PATH, mmap_mode="r")
if len(sample_meta) != sample_count:
    raise RuntimeError("window cache metadata mismatch")

print(
    "cached samples/wells:",
    sample_count,
    sample_meta["well"].nunique(),
    "disk arrays:",
    H_PATH,
    TW_PATH,
)


class MatcherDataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        horizontal = np.asarray(horizontal_windows[index], dtype=np.float32)
        typewell = np.asarray(typewell_windows[index], dtype=np.float32)
        return torch.from_numpy(horizontal), torch.from_numpy(typewell), index


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        padding = 2 * dilation
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
        )

    def forward(self, values):
        return F.silu(values + self.network(values))


class WindowEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 48, kernel_size=7, padding=3),
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
        pooled = torch.cat([hidden.mean(dim=-1), hidden.amax(dim=-1)], dim=1)
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
        forward_embedding = self.typewell_encoder(flat).reshape(batch, candidates, -1)
        reverse = torch.flip(typewell, dims=[-1]).reshape(batch * candidates, channels, length)
        reverse_embedding = self.typewell_encoder(reverse).reshape(batch, candidates, -1)
        forward_score = self._score_embeddings(horizontal_embedding, forward_embedding)
        reverse_score = self._score_embeddings(horizontal_embedding, reverse_embedding)
        return torch.maximum(forward_score, reverse_score)


def _ncc_scores(horizontal, typewell):
    h_value = horizontal[:, None, 0, :]
    h_mask = horizontal[:, None, 1, :]

    def score(candidate):
        tw_value = candidate[:, :, 0, :]
        tw_mask = candidate[:, :, 1, :]
        mask = h_mask * tw_mask
        numerator = torch.sum(mask * h_value * tw_value, dim=-1)
        denominator = torch.sqrt(
            torch.sum(mask * h_value * h_value, dim=-1)
            * torch.sum(mask * tw_value * tw_value, dim=-1)
            + 1e-8
        )
        return numerator / denominator

    return torch.maximum(score(typewell), score(torch.flip(typewell, dims=[-1])))


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
def _evaluate(model, loader):
    model.eval()
    learned_predictions = []
    ncc_predictions = []
    sample_indices = []
    losses = []
    for horizontal, typewell, indices in loader:
        horizontal = horizontal.to(DEVICE, non_blocking=True)
        typewell = typewell.to(DEVICE, non_blocking=True)
        logits = model(horizontal, typewell)
        labels = torch.full(
            (len(horizontal),), POSITIVE_INDEX, dtype=torch.long, device=DEVICE
        )
        losses.append(float(F.cross_entropy(logits, labels).item()) * len(horizontal))
        learned_predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
        ncc_predictions.append(
            torch.argmax(_ncc_scores(horizontal, typewell), dim=1).cpu().numpy()
        )
        sample_indices.append(indices.numpy())
    return {
        "sample_index": np.concatenate(sample_indices),
        "learned_candidate": np.concatenate(learned_predictions),
        "ncc_candidate": np.concatenate(ncc_predictions),
        "loss": float(np.sum(losses) / max(1, len(loader.dataset))),
    }


fold_records = []
oof_records = []
training_started = time.time()
top40_wells = _find_top40_wells()

for fold in range(1, N_FOLDS + 1):
    fold_prediction_path = CACHE_DIR / f"matcher_fold{fold}_predictions.pkl"
    fold_metric_path = CACHE_DIR / f"matcher_fold{fold}_metrics.json"
    if fold_prediction_path.exists() and fold_metric_path.exists():
        cached_prediction = pd.read_pickle(fold_prediction_path)
        cached_metric = json.loads(fold_metric_path.read_text())
        oof_records.append(cached_prediction)
        fold_records.append(cached_metric)
        print(f"loaded cached fold {fold}:", cached_metric, flush=True)
        continue

    train_indices = sample_meta.loc[sample_meta["fold"] != fold, "sample_index"].to_numpy(int)
    valid_indices = sample_meta.loc[sample_meta["fold"] == fold, "sample_index"].to_numpy(int)
    train_loader = _make_loader(train_indices, shuffle=True)
    valid_loader = _make_loader(valid_indices, shuffle=False)

    model = GRWindowMatcher().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    print(
        f"\nfold {fold}/{N_FOLDS} train={len(train_indices)} valid={len(valid_indices)}",
        flush=True,
    )
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"fold {fold} epoch {epoch}")
        for horizontal, typewell, _ in progress:
            horizontal = horizontal.to(DEVICE, non_blocking=True)
            typewell = typewell.to(DEVICE, non_blocking=True)
            labels = torch.full(
                (len(horizontal),), POSITIVE_INDEX, dtype=torch.long, device=DEVICE
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(horizontal, typewell)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * len(horizontal)
            seen += len(horizontal)
            progress.set_postfix(loss=running_loss / max(1, seen))

        validation = _evaluate(model, valid_loader)
        print(
            f"fold={fold} epoch={epoch} train_loss={running_loss/max(1,seen):.5f} "
            f"valid_loss={validation['loss']:.5f}",
            flush=True,
        )

    torch.save(model.state_dict(), MODEL_DIR / f"matcher_fold{fold}.pt")
    validation = _evaluate(model, valid_loader)
    learned_offset = OFFSETS_FT[validation["learned_candidate"]]
    ncc_offset = OFFSETS_FT[validation["ncc_candidate"]]
    learned_abs = np.abs(learned_offset)
    ncc_abs = np.abs(ncc_offset)

    valid_meta = sample_meta.set_index("sample_index").loc[validation["sample_index"]].reset_index()
    valid_meta["learned_offset"] = learned_offset
    valid_meta["ncc_offset"] = ncc_offset
    valid_meta["learned_abs_offset"] = learned_abs
    valid_meta["ncc_abs_offset"] = ncc_abs
    oof_records.append(valid_meta)

    row = {
        "fold": fold,
        "samples": len(valid_meta),
        "wells": valid_meta["well"].nunique(),
        "learned_within4": float(np.mean(learned_abs <= 4.0)),
        "ncc_within4": float(np.mean(ncc_abs <= 4.0)),
        "within4_gain": float(np.mean(learned_abs <= 4.0) - np.mean(ncc_abs <= 4.0)),
        "learned_within8": float(np.mean(learned_abs <= 8.0)),
        "ncc_within8": float(np.mean(ncc_abs <= 8.0)),
        "within8_gain": float(np.mean(learned_abs <= 8.0) - np.mean(ncc_abs <= 8.0)),
        "learned_median_abs": float(np.median(learned_abs)),
        "ncc_median_abs": float(np.median(ncc_abs)),
        "valid_loss": validation["loss"],
    }
    fold_records.append(row)
    print("fold metrics:", row, flush=True)
    valid_meta.to_pickle(fold_prediction_path)
    fold_metric_path.write_text(json.dumps(row, indent=2))

    del model, optimizer, scaler, train_loader, valid_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


fold_summary = pd.DataFrame(fold_records)
oof = pd.concat(oof_records, ignore_index=True)

learned_abs = oof["learned_abs_offset"].to_numpy(float)
ncc_abs = oof["ncc_abs_offset"].to_numpy(float)
overall = {
    "samples": len(oof),
    "wells": oof["well"].nunique(),
    "learned_within4": float(np.mean(learned_abs <= 4.0)),
    "ncc_within4": float(np.mean(ncc_abs <= 4.0)),
    "within4_gain": float(np.mean(learned_abs <= 4.0) - np.mean(ncc_abs <= 4.0)),
    "learned_within8": float(np.mean(learned_abs <= 8.0)),
    "ncc_within8": float(np.mean(ncc_abs <= 8.0)),
    "within8_gain": float(np.mean(learned_abs <= 8.0) - np.mean(ncc_abs <= 8.0)),
    "learned_median_abs": float(np.median(learned_abs)),
    "ncc_median_abs": float(np.median(ncc_abs)),
    "training_hours": (time.time() - training_started) / 3600.0,
}

if top40_wells:
    top_mask = oof["well"].astype(str).isin(top40_wells).to_numpy()
    overall["top40_samples"] = int(top_mask.sum())
    overall["top40_learned_within4"] = float(np.mean(learned_abs[top_mask] <= 4.0))
    overall["top40_ncc_within4"] = float(np.mean(ncc_abs[top_mask] <= 4.0))
    overall["top40_within4_gain"] = (
        overall["top40_learned_within4"] - overall["top40_ncc_within4"]
    )
else:
    overall["top40_samples"] = 0
    overall["top40_within4_gain"] = np.nan

strict_pass = bool(
    overall["within4_gain"] >= PASS_WITHIN4_GAIN
    and overall["within8_gain"] >= PASS_WITHIN8_GAIN
    and int((fold_summary["within4_gain"] > 0).sum()) == N_FOLDS
    and int((fold_summary["within8_gain"] > 0).sum()) == N_FOLDS
    and bool((fold_summary["learned_median_abs"] < fold_summary["ncc_median_abs"]).all())
    and (
        not top40_wells
        or float(overall["top40_within4_gain"]) >= 0.0
    )
)
overall["strict_pass"] = strict_pass
overall["verdict"] = (
    "PASS -> build learned-emission DP"
    if strict_pass
    else "FAIL / STOP supervised alignment"
)

print("\nFOLDS")
display(fold_summary)
print("\nOVERALL")
print(overall)
print("\nVERDICT:", overall["verdict"])

fold_summary.to_csv(CACHE_DIR / "matcher_v0_folds.csv", index=False)
oof.to_pickle(CACHE_DIR / "matcher_v0_oof.pkl")
pd.DataFrame([overall]).to_csv(CACHE_DIR / "matcher_v0_summary.csv", index=False)
print("saved:", CACHE_DIR)
