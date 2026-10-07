"""Learned full-sequence scorer for the strict one-fold GR path bank.

This is intentionally different from RankSDF and tabular GR summaries.  A
small shared 1D CNN reads each complete candidate-aligned GR sequence and is
trained listwise over the 235 trajectories for one well.  The bank, ranker
training fold, and validation fold remain mutually disjoint.
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
from scipy.ndimage import uniform_filter1d
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE_DIR = OOF_ROOT / "gr_path_matcher_onefold_v0"
BANK_PATH = SOURCE_DIR / "restricted_path_bank.npz"
FEATURE_PATH = SOURCE_DIR / "path_features.pkl"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
OUT_DIR = OOF_ROOT / "gr_path_sequence_scorer_onefold_v0"
CACHE_DIR = OUT_DIR / "sequence_cache"
MODEL_PATH = OUT_DIR / "best_sequence_scorer.pt"

TRAIN_FOLD = 5
VALID_FOLD = 1
SEQUENCE_LENGTH = 256
EPOCHS = 20
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
TARGET_TEMPERATURE_FT = 1.5
REGRESSION_SCALE_FT = 5.0
REGRESSION_WEIGHT = 0.20
NOISE_STD = 0.02
PATIENCE = 5
SEED = 20260716

PASS_GAIN_VS_STRONG = 0.20
PASS_POSITIVE_WELLS = 0.55
PASS_MEDIAN_WELL_GAIN = 0.10
PASS_MEAN_REGRET = 3.00

CHANNELS = [
    "observed_gr", "candidate_gr", "difference", "absolute_difference",
    "candidate_valid", "correction", "smooth64_difference",
    "smooth256_difference", "progress",
]


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _rmse(truth, prediction):
    return float(np.sqrt(np.mean((np.asarray(truth, float) - np.asarray(prediction, float)) ** 2)))


def _fit_affine(horizontal_gr, typewell_gr):
    x = np.asarray(horizontal_gr, float)
    y = np.asarray(typewell_gr, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 32 or np.std(x) < 1e-6:
        return 1.0, 0.0
    keep = np.ones(len(x), bool)
    slope, intercept = 1.0, float(np.median(y) - np.median(x))
    for _ in range(4):
        design = np.column_stack([x[keep], np.ones(keep.sum())])
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        centre = float(np.median(residual))
        scale = float(1.4826 * np.median(np.abs(residual - centre)) + 1e-6)
        keep = np.abs(residual - centre) <= 3.5 * scale
        if keep.sum() < 24:
            break
    return float(np.clip(slope, -5.0, 5.0)), float(intercept)


def _moving_average(values, valid, size):
    weight = valid.astype(np.float32)
    count = uniform_filter1d(weight, size=size, axis=1, mode="constant", cval=0.0) * size
    total = uniform_filter1d(values * weight, size=size, axis=1, mode="constant", cval=0.0) * size
    output = total / np.maximum(count, 1.0)
    output[count < max(4, int(np.ceil(0.5 * size)))] = 0.0
    return output.astype(np.float32)


def _cache_path(well):
    return CACHE_DIR / f"{well}.npz"


def _build_sequence_cache(well, group):
    output_path = _cache_path(well)
    if output_path.exists():
        return {"well": well, "status": "cached", "path": str(output_path)}
    try:
        group = group.sort_values("candidate_index")
        pool_well = POOL_GROUPS[well].sort_values("row_idx")
        horizontal = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
            usecols=["MD", "Z", "GR", "TVT_input"],
        )
        typewell = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__typewell.csv",
            usecols=["TVT", "GR"],
        )
        for column in ("MD", "Z", "GR", "TVT_input"):
            horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
        typewell = typewell.apply(pd.to_numeric, errors="coerce").dropna(subset=["TVT"])
        typewell = typewell.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
        typewell["GR"] = typewell["GR"].interpolate(limit_direction="both").fillna(0.0)

        md = horizontal["MD"].to_numpy(float)
        gr_raw = horizontal["GR"].to_numpy(float)
        tvt_input = horizontal["TVT_input"].to_numpy(float)
        rows = pool_well["row_idx"].to_numpy(int)
        anchor = int(rows.min() - 1)
        anchor_md = float(md[anchor])
        eval_md = md[rows]
        eval_span = float(max(eval_md[-1] - anchor_md, 1.0))
        sequence_md = np.linspace(float(eval_md[0]), float(eval_md[-1]), SEQUENCE_LENGTH)
        progress = np.clip((sequence_md - anchor_md) / eval_span, 0.0, 1.0).astype(np.float32)
        strong = np.interp(sequence_md, eval_md, pool_well["strong"].to_numpy(float))

        tw_tvt = typewell["TVT"].to_numpy(float)
        tw_gr_raw = typewell["GR"].to_numpy(float)
        known = np.isfinite(tvt_input) & np.isfinite(gr_raw)
        known &= (tvt_input >= tw_tvt[0]) & (tvt_input <= tw_tvt[-1])
        known_tw = np.interp(tvt_input[known], tw_tvt, tw_gr_raw)
        slope, intercept = _fit_affine(gr_raw[known], known_tw)
        calibrated = slope * gr_raw + intercept
        observed_valid_rows = np.isfinite(calibrated)
        if observed_valid_rows.sum() < 2:
            observed_fill = np.zeros(len(md), float)
        else:
            observed_fill = np.interp(md, md[observed_valid_rows], calibrated[observed_valid_rows])
        observed = np.interp(sequence_md, md, observed_fill)
        observed_valid = np.interp(sequence_md, md, observed_valid_rows.astype(float)) >= 0.5
        calibration_values = known_tw if len(known_tw) >= 16 else tw_gr_raw
        location = float(np.median(calibration_values))
        scale = float(1.4826 * np.median(np.abs(calibration_values - location)) + 1e-3)
        observed = np.clip((observed - location) / scale, -8.0, 8.0).astype(np.float32)
        tw_gr = np.clip((tw_gr_raw - location) / scale, -8.0, 8.0).astype(np.float32)

        length_ratio = max(eval_span / max(BANK_MEDIAN_LENGTH, 1.0), 0.1)
        template_scale = length_ratio ** BANK_ALPHAS
        corrections = np.stack(
            [np.interp(progress, BANK_PROGRESS, template) for template in BANK_TEMPLATES]
        )
        corrections *= template_scale[:, None]
        corrections = np.clip(corrections, -BANK_CAP, BANK_CAP).astype(np.float32)
        candidate_tvt = strong[None, :] + corrections
        candidate_valid = (
            (candidate_tvt >= tw_tvt[0])
            & (candidate_tvt <= tw_tvt[-1])
            & observed_valid[None, :]
        )
        candidate_gr = np.interp(candidate_tvt.ravel(), tw_tvt, tw_gr).reshape(candidate_tvt.shape)
        candidate_gr[~candidate_valid] = 0.0
        observed_grid = np.broadcast_to(observed[None, :], candidate_gr.shape).astype(np.float32)
        difference = (candidate_gr - observed_grid) * candidate_valid
        smooth64 = _moving_average(difference, candidate_valid, max(3, round(64 / (eval_span / SEQUENCE_LENGTH))))
        smooth256 = _moving_average(difference, candidate_valid, max(5, round(256 / (eval_span / SEQUENCE_LENGTH))))
        sequence = np.stack(
            [
                observed_grid,
                candidate_gr,
                difference,
                np.abs(difference),
                candidate_valid.astype(np.float32),
                corrections / max(BANK_CAP, 1.0),
                smooth64,
                smooth256,
                np.broadcast_to(progress[None, :], candidate_gr.shape),
            ],
            axis=1,
        ).astype(np.float16)
        candidate_rmse = group["candidate_rmse"].to_numpy(np.float32)
        if sequence.shape != (len(BANK_NAMES), len(CHANNELS), SEQUENCE_LENGTH):
            raise RuntimeError(f"sequence shape mismatch {sequence.shape}")
        if not np.array_equal(group["candidate_name"].to_numpy(str), BANK_NAMES):
            raise RuntimeError("candidate ordering mismatch")
        np.savez_compressed(
            output_path,
            sequence=sequence,
            candidate_rmse=candidate_rmse,
            candidate_names=BANK_NAMES,
            rows=np.asarray([len(pool_well)], np.int32),
            strong_rmse=np.asarray([candidate_rmse[ZERO_INDEX]], np.float32),
        )
        return {"well": well, "status": "built", "path": str(output_path)}
    except Exception as exc:
        return {"well": well, "status": "failed", "reason": repr(exc)}


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels),
        )
    def forward(self, values):
        return F.silu(values + self.net(values))


class PathSequenceScorer(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(inputs, 48, 7, padding=3, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(48, 1),
            ResidualBlock(48, 2),
            ResidualBlock(48, 4),
            ResidualBlock(48, 8),
        )
        self.head = nn.Sequential(
            nn.Linear(96, 64), nn.SiLU(), nn.Dropout(0.10), nn.Linear(64, 1)
        )
    def forward(self, values):
        hidden = self.blocks(self.stem(values))
        pooled = torch.cat([hidden.mean(-1), hidden.amax(-1)], dim=1)
        return self.head(pooled).squeeze(1)


def _load_group(well, device):
    with np.load(_cache_path(well), allow_pickle=False) as data:
        sequence = torch.from_numpy(data["sequence"].astype(np.float32)).to(device)
        rmse = torch.from_numpy(data["candidate_rmse"].astype(np.float32)).to(device)
        rows = int(data["rows"][0])
        strong_rmse = float(data["strong_rmse"][0])
    return sequence, rmse, rows, strong_rmse


@torch.no_grad()
def _evaluate(model, wells, device):
    model.eval()
    records = []
    total_sse = 0.0
    total_strong_sse = 0.0
    total_rows = 0
    for well in wells:
        sequence, rmse, rows, strong_rmse = _load_group(well, device)
        scores = model(sequence)
        selected = int(torch.argmax(scores).item())
        oracle = int(torch.argmin(rmse).item())
        selected_rmse = float(rmse[selected].item())
        oracle_rmse = float(rmse[oracle].item())
        total_sse += rows * selected_rmse ** 2
        total_strong_sse += rows * strong_rmse ** 2
        total_rows += rows
        records.append(
            {
                "well": well,
                "rows": rows,
                "strong_rmse": strong_rmse,
                "selected_rmse": selected_rmse,
                "oracle_rmse": oracle_rmse,
                "gain": strong_rmse - selected_rmse,
                "regret": selected_rmse - oracle_rmse,
                "selected_index": selected,
                "selected_name": str(BANK_NAMES[selected]),
            }
        )
    bywell = pd.DataFrame(records)
    strong_pooled = float(np.sqrt(total_strong_sse / total_rows))
    selected_pooled = float(np.sqrt(total_sse / total_rows))
    report = {
        "rows": total_rows,
        "wells": len(bywell),
        "strong_rmse": strong_pooled,
        "rmse": selected_pooled,
        "gain_vs_strong": strong_pooled - selected_pooled,
        "mean_well_rmse": float(bywell["selected_rmse"].mean()),
        "mean_well_gain": float(bywell["gain"].mean()),
        "median_well_gain": float(bywell["gain"].median()),
        "positive_well_fraction": float((bywell["gain"] > 0).mean()),
        "mean_regret": float(bywell["regret"].mean()),
        "median_regret": float(bywell["regret"].median()),
    }
    return report, bywell


for path in (BANK_PATH, FEATURE_PATH, DATA_ROOT / "train"):
    if not path.exists():
        raise FileNotFoundError(path)
pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

with np.load(BANK_PATH, allow_pickle=False) as bank:
    BANK_TEMPLATES = bank["templates"].astype(np.float32)
    BANK_ALPHAS = bank["length_alphas"].astype(np.float32)
    BANK_NAMES = bank["names"].astype(str)
    BANK_PROGRESS = bank["progress_grid"].astype(np.float32)
    BANK_MEDIAN_LENGTH = float(bank["median_length"][0])
    BANK_CAP = float(bank["cap"][0])
ZERO_INDEX = int(np.flatnonzero(BANK_NAMES == "zero")[0])

features = pd.read_pickle(FEATURE_PATH)
features = features[features["fold"].isin([TRAIN_FOLD, VALID_FOLD])].copy()
feature_groups = {
    str(well): group.copy()
    for well, group in features.groupby("well", sort=False)
}
train_wells = sorted(
    features.loc[features["fold"] == TRAIN_FOLD, "well"].astype(str).unique()
)
valid_wells = sorted(
    features.loc[features["fold"] == VALID_FOLD, "well"].astype(str).unique()
)

header = pd.read_csv(pool_path, nrows=0).columns.tolist()
usecols = [column for column in ("id", "well", "row_idx", "strong") if column in header]
pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pool["row_idx"].astype(np.int32)
POOL_GROUPS = {
    str(well): group.copy()
    for well, group in pool[pool["well"].isin(train_wells + valid_wells)].groupby("well", sort=False)
}

started = time.time()
results = []
for well in tqdm(train_wells + valid_wells, desc="build full-sequence GR cache"):
    results.append(_build_sequence_cache(well, feature_groups[well]))
build_report = pd.DataFrame(results)
build_report.to_csv(OUT_DIR / "cache_build_report.csv", index=False)
failures = build_report[build_report["status"] == "failed"]
if len(failures):
    print(failures.head(30).to_string(index=False))
    raise RuntimeError(f"sequence cache failures: {len(failures)}")

_seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PathSequenceScorer(len(CHANNELS)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
best_rmse = np.inf
best_epoch = 0
history = []

print(
    {
        "experiment": "gr_path_sequence_scorer_onefold_v0",
        "device": str(DEVICE),
        "train_wells": len(train_wells),
        "valid_wells": len(valid_wells),
        "candidates": len(BANK_NAMES),
        "channels": CHANNELS,
        "sequence_length": SEQUENCE_LENGTH,
    }
)

for epoch in range(1, EPOCHS + 1):
    model.train()
    shuffled = train_wells.copy()
    random.shuffle(shuffled)
    losses = []
    for well in tqdm(shuffled, desc=f"epoch {epoch}", leave=False):
        sequence, rmse, _, _ = _load_group(well, DEVICE)
        if NOISE_STD > 0:
            sequence = sequence.clone()
            sequence[:, :4] += torch.randn_like(sequence[:, :4]) * NOISE_STD
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            scores = model(sequence)
            target_probability = torch.softmax(
                -(rmse - torch.min(rmse)) / TARGET_TEMPERATURE_FT, dim=0
            )
            listwise = -(target_probability * F.log_softmax(scores, dim=0)).sum()
            relative_score = scores - scores[ZERO_INDEX]
            target_gain = (rmse[ZERO_INDEX] - rmse) / REGRESSION_SCALE_FT
            regression = F.smooth_l1_loss(relative_score, target_gain)
            loss = listwise + REGRESSION_WEIGHT * regression
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    report, bywell = _evaluate(model, valid_wells, DEVICE)
    record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **report}
    history.append(record)
    print("epoch", record, flush=True)
    if report["rmse"] < best_rmse:
        best_rmse = report["rmse"]
        best_epoch = epoch
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "report": report,
                "channels": CHANNELS,
                "bank_names": BANK_NAMES.tolist(),
            },
            MODEL_PATH,
        )
        bywell.to_csv(OUT_DIR / "best_bywell.csv", index=False)
    if epoch - best_epoch >= PATIENCE:
        print(f"early stopping at epoch {epoch}; best epoch {best_epoch}")
        break

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
best_report, best_bywell = _evaluate(model, valid_wells, DEVICE)
pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
best_bywell.to_csv(OUT_DIR / "selected_bywell.csv", index=False)

strict_pass = bool(
    best_report["gain_vs_strong"] >= PASS_GAIN_VS_STRONG
    and best_report["positive_well_fraction"] >= PASS_POSITIVE_WELLS
    and best_report["median_well_gain"] >= PASS_MEDIAN_WELL_GAIN
    and best_report["mean_regret"] <= PASS_MEAN_REGRET
)
verdict = {
    "experiment": "gr_path_sequence_scorer_onefold_v0",
    "source": str(SOURCE_DIR),
    "train_fold": TRAIN_FOLD,
    "valid_fold": VALID_FOLD,
    "train_wells": len(train_wells),
    "valid_wells": len(valid_wells),
    "candidates": len(BANK_NAMES),
    "sequence_length": SEQUENCE_LENGTH,
    "channels": CHANNELS,
    "best_epoch": int(checkpoint["epoch"]),
    "best": best_report,
    "gates": {
        "gain_vs_strong_min": PASS_GAIN_VS_STRONG,
        "positive_well_fraction_min": PASS_POSITIVE_WELLS,
        "median_well_gain_min": PASS_MEDIAN_WELL_GAIN,
        "mean_regret_max": PASS_MEAN_REGRET,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> extend learned sequence scoring to nested pseudo-PS folds"
        if strict_pass
        else "FAIL -> close own-typewell GR path matching despite the large path oracle"
    ),
    "cache_build_minutes": (time.time() - started) / 60.0,
}
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nBEST")
print(json.dumps(best_report, indent=2))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
