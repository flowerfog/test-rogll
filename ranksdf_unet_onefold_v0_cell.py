"""RankSDF Step 2: one honest outer-fold Rank-UNet baseline.

This is intentionally minimal: official prediction starts, prefix-only panel
centres, column-wise soft ranking loss, and posterior-mean decoding.  It does
not use CP12 as an input, synthetic data, SDF loss, DP, routing, or Gold.

One configured fold is held out by well using the established foldclean
metadata. The standalone model is diagnostic; downstream fixed-weight
ensembling is evaluated separately.
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
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
CACHE_DIR = OOF_ROOT / "ranksdf_panel_cache_v0_s257_c512"

POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
FOLD_CANDIDATES = [
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
]

# Independent confirmation configuration.  Hard-coded deliberately so a fresh
# notebook kernel cannot silently fall back to fold 1 or validation-selected
# epoch count.
VALID_FOLD = 3
FIXED_EPOCHS = 10
OUT_DIR = OOF_ROOT / f"ranksdf_unet_onefold_v0_fold{VALID_FOLD}_s257_c512"
MODEL_DIR = OUT_DIR / "models"
WIDTH_FT = 256.0
STATE_STEP_FT = 2.0
N_COLUMNS = 512
PREFIX_CONTEXT_FT = 1000.0
SOFT_LABEL_SIGMA_FT = 3.0

BASE_CHANNELS = 16
BATCH_SIZE = 2
EPOCHS = FIXED_EPOCHS if FIXED_EPOCHS > 0 else 12
PATIENCE = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 20260716

PASS_GAIN_VS_STRONG = 0.20
PASS_POSITIVE_WELL_FRACTION = 0.55

CHANNEL_NAMES = [
    "typewell_gr",
    "horizontal_gr",
    "gr_difference",
    "gr_abs_difference",
    "gr_difference_smooth5",
    "gr_difference_smooth21",
    "gr_difference_smooth51",
    "typewell_valid",
    "known_path_gaussian",
    "known_column_mask",
    "candidate_offset",
    "md_relative",
    "horizontal_gr_valid",
]
TYPEWELL_VALID_CHANNEL = CHANNEL_NAMES.index("typewell_valid")


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _fill_1d(values):
    return (
        pd.Series(np.asarray(values, np.float64))
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _robust_normalise(values, valid=None):
    values = np.asarray(values, np.float32)
    valid = np.isfinite(values) if valid is None else (np.asarray(valid, bool) & np.isfinite(values))
    output = np.zeros_like(values, np.float32)
    sample = values[valid]
    if len(sample) < 8:
        return output
    median = float(np.median(sample))
    scale = float(1.4826 * np.median(np.abs(sample - median)))
    if scale < 1e-3:
        scale = float(np.std(sample) + 1e-3)
    output[valid] = np.clip((values[valid] - median) / scale, -6.0, 6.0)
    return output


def _robust_slope(md, structure, anchor_md, anchor_structure):
    x = np.asarray(md, np.float64) - float(anchor_md)
    y = np.asarray(structure, np.float64) - float(anchor_structure)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 8 or np.ptp(x) < 20.0:
        return 0.0
    slope = float(np.sum(x * y) / max(np.sum(x * x), 1e-12))
    for _ in range(5):
        residual = y - slope * x
        scale = float(1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6)
        weight = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
        slope = float(np.sum(weight * x * y) / max(np.sum(weight * x * x), 1e-12))
    return float(np.clip(slope, -0.5, 0.5))


def _discover_files(wells, suffix):
    wells = set(map(str, wells))
    output = {}
    for root in (DATA_ROOT / "train", DATA_ROOT):
        if not root.exists():
            continue
        for path in root.rglob(f"*{suffix}"):
            if not path.name.endswith(suffix):
                continue
            well = path.name[: -len(suffix)]
            if well in wells and "test" not in {part.lower() for part in path.parts}:
                output.setdefault(well, path)
    missing = sorted(wells - set(output))
    if missing:
        raise FileNotFoundError(f"missing {suffix}: {missing[:20]}")
    return output


def _load_fold_map(wells):
    for path in FOLD_CANDIDATES:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if {"well", "fold"}.issubset(frame.columns):
            frame["well"] = frame["well"].astype(str)
            mapping = frame.groupby("well")["fold"].first().astype(int).to_dict()
            if set(wells).issubset(mapping):
                return mapping, str(path)
    raise FileNotFoundError("established well fold map not found")


def _build_panel(well, horizontal_path, typewell_path, pool_well):
    horizontal = pd.read_csv(horizontal_path)
    required = {"MD", "Z", "GR", "TVT", "TVT_input"}
    if not required.issubset(horizontal.columns):
        raise RuntimeError(f"{well} horizontal missing columns")
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    typewell = typewell[np.isfinite(typewell["TVT"])].sort_values("TVT")
    typewell = typewell.groupby("TVT", as_index=False)["GR"].mean()

    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md) & np.isfinite(z))
    anchor = int(known[-1])
    anchor_md = float(md[anchor])
    anchor_s = float(tvt_input[anchor] + z[anchor])
    recent = known[md[known] >= anchor_md - 500.0]
    if len(recent) < 16:
        recent = known[-min(64, len(known)):]
    slope = _robust_slope(md[recent], tvt_input[recent] + z[recent], anchor_md, anchor_s)

    sample_md = np.linspace(
        max(float(md[0]), anchor_md - PREFIX_CONTEXT_FT), float(md[-1]), N_COLUMNS
    )
    sample_z = np.interp(sample_md, md, z)
    sample_tvt = np.interp(sample_md, md, tvt)
    true_s = sample_tvt + sample_z
    center_s = anchor_s + slope * (sample_md - anchor_md)
    true_residual = true_s - center_s
    known_columns = sample_md <= anchor_md + 1e-6
    observed_known_s = np.interp(
        sample_md[known_columns], md[known], tvt_input[known] + z[known]
    )
    target_residual = true_residual.copy()
    target_residual[known_columns] = observed_known_s - center_s[known_columns]

    gr_valid = np.isfinite(gr_raw).astype(np.float32)
    sample_gr = np.interp(sample_md, md, _fill_1d(gr_raw)).astype(np.float32)
    sample_gr_valid = np.interp(sample_md, md, gr_valid) >= 0.5
    horizontal_gr = _robust_normalise(sample_gr, sample_gr_valid)

    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr_raw = typewell["GR"].to_numpy(float)
    tw_gr = _robust_normalise(_fill_1d(tw_gr_raw), np.isfinite(tw_gr_raw))
    offsets = np.arange(-WIDTH_FT, WIDTH_FT + 0.25 * STATE_STEP_FT, STATE_STEP_FT, dtype=np.float32)
    candidate_s = center_s[None] + offsets[:, None]
    candidate_tvt = candidate_s - sample_z[None]
    typewell_valid = (candidate_tvt >= tw_tvt[0]) & (candidate_tvt <= tw_tvt[-1])
    tw_grid = np.interp(candidate_tvt.reshape(-1), tw_tvt, tw_gr).reshape(len(offsets), N_COLUMNS).astype(np.float32)
    tw_grid[~typewell_valid] = 0.0
    horizontal_grid = np.broadcast_to(horizontal_gr[None], tw_grid.shape).astype(np.float32)
    difference = tw_grid - horizontal_grid

    known_path = np.zeros_like(tw_grid, np.float32)
    known_path[:, known_columns] = np.exp(
        -0.5 * ((offsets[:, None] - target_residual[None, known_columns]) / SOFT_LABEL_SIGMA_FT) ** 2
    ).astype(np.float32)
    known_mask = np.broadcast_to(known_columns[None], tw_grid.shape).astype(np.float32)
    offset_grid = np.broadcast_to((offsets[:, None] / WIDTH_FT), tw_grid.shape).astype(np.float32)
    md_relative = ((sample_md - anchor_md) / max(float(md[-1]) - anchor_md, 1.0)).astype(np.float32)

    panel = np.stack(
        [
            tw_grid,
            horizontal_grid,
            difference,
            np.abs(difference),
            uniform_filter1d(difference, 5, axis=1, mode="nearest"),
            uniform_filter1d(difference, 21, axis=1, mode="nearest"),
            uniform_filter1d(difference, 51, axis=1, mode="nearest"),
            typewell_valid.astype(np.float32),
            known_path,
            known_mask,
            offset_grid,
            np.broadcast_to(md_relative[None], tw_grid.shape).astype(np.float32),
            np.broadcast_to(sample_gr_valid[None], tw_grid.shape).astype(np.float32),
        ], axis=0
    ).astype(np.float16)

    pool_well = pool_well.sort_values("row_idx")
    eval_rows = pool_well["row_idx"].to_numpy(int)
    return {
        "panel": panel,
        "offsets": offsets,
        "sample_md": sample_md.astype(np.float32),
        "sample_z": sample_z.astype(np.float32),
        "center_s": center_s.astype(np.float32),
        "true_s": true_s.astype(np.float32),
        "target_residual": target_residual.astype(np.float32),
        "known_columns": known_columns.astype(np.uint8),
        "eval_md": md[eval_rows].astype(np.float32),
        "eval_tvt": pool_well["tvt_true"].to_numpy(np.float32),
        "eval_strong": pool_well["strong"].to_numpy(np.float32),
        "eval_cp12": pool_well["cp12"].to_numpy(np.float32) if "cp12" in pool_well else np.full(len(pool_well), np.nan, np.float32),
        "coverage": np.asarray([np.mean(np.abs(true_residual) <= WIDTH_FT)], np.float32),
    }


class PanelDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as data:
            return (
                torch.from_numpy(data["panel"].astype(np.float32)),
                torch.from_numpy(data["target_residual"].astype(np.float32)),
                torch.from_numpy(data["known_columns"].astype(bool)),
            )


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class TinyRankUNet(nn.Module):
    def __init__(self, in_channels, base=16):
        super().__init__()
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 6)
        self.mid = ConvBlock(base * 6, base * 8)
        self.d4 = ConvBlock(base * 14, base * 6)
        self.d3 = ConvBlock(base * 10, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2)
        self.d1 = ConvBlock(base * 3, base)
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def up(x, skip):
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        x = self.mid(F.max_pool2d(e4, 2))
        x = self.d4(torch.cat([self.up(x, e4), e4], 1))
        x = self.d3(torch.cat([self.up(x, e3), e3], 1))
        x = self.d2(torch.cat([self.up(x, e2), e2], 1))
        x = self.d1(torch.cat([self.up(x, e1), e1], 1))
        return self.out(x).squeeze(1)


def _loss(logits, panel, target_residual, known, offsets):
    valid = panel[:, TYPEWELL_VALID_CHANNEL] > 0.5
    logits = logits.masked_fill(~valid, -30.0)
    target = torch.exp(-0.5 * ((offsets[None, :, None] - target_residual[:, None]) / SOFT_LABEL_SIGMA_FT) ** 2)
    target = target * valid.float()
    target = target / target.sum(1, keepdim=True).clamp_min(1e-8)
    column_loss = -(target * F.log_softmax(logits, 1)).sum(1)
    weight = torch.where(known, torch.full_like(column_loss, 0.5), torch.ones_like(column_loss))
    return (column_loss * weight).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def _evaluate(model, paths, device):
    model.eval()
    well_rows, errors, strong_errors, cp12_errors = [], [], [], []
    for path in tqdm(paths, desc="validate RankSDF", leave=False):
        with np.load(path, allow_pickle=False) as data:
            panel = data["panel"].astype(np.float32)
            offsets = data["offsets"].astype(np.float32)
            known = data["known_columns"].astype(bool)
            target_residual = data["target_residual"].astype(float)
            sample_md = data["sample_md"].astype(float)
            center_s = data["center_s"].astype(float)
            sample_z = data["sample_z"].astype(float)
            eval_md = data["eval_md"].astype(float)
            truth = data["eval_tvt"].astype(float)
            strong = data["eval_strong"].astype(float)
            cp12 = data["eval_cp12"].astype(float)
            coverage = float(data["coverage"][0])
        tensor = torch.from_numpy(panel)[None].to(device)
        logits = model(tensor)
        valid = tensor[:, TYPEWELL_VALID_CHANNEL] > 0.5
        probability = torch.softmax(logits.masked_fill(~valid, -30.0), 1)
        offset_tensor = torch.from_numpy(offsets).to(device)
        predicted_offset = torch.sum(probability * offset_tensor[None, :, None], 1)[0].cpu().numpy()
        predicted_offset[known] = target_residual[known]
        sample_prediction = center_s + predicted_offset - sample_z
        prediction = np.interp(eval_md, sample_md, sample_prediction)
        error = prediction - truth
        strong_error = strong - truth
        cp12_error = cp12 - truth
        errors.append(error)
        strong_errors.append(strong_error)
        cp12_errors.append(cp12_error)
        well_rows.append({
            "well": path.stem,
            "rows": len(truth),
            "model_rmse": float(np.sqrt(np.mean(error**2))),
            "strong_rmse": float(np.sqrt(np.mean(strong_error**2))),
            "cp12_rmse": float(np.sqrt(np.mean(cp12_error**2))),
            "gain_vs_strong": float(np.sqrt(np.mean(strong_error**2)) - np.sqrt(np.mean(error**2))),
            "coverage": coverage,
        })
    error = np.concatenate(errors)
    strong_error = np.concatenate(strong_errors)
    cp12_error = np.concatenate(cp12_errors)
    cosine = float(np.dot(error, strong_error) / np.sqrt(np.dot(error, error) * np.dot(strong_error, strong_error)))
    return {
        "model_rmse": float(np.sqrt(np.mean(error**2))),
        "strong_rmse": float(np.sqrt(np.mean(strong_error**2))),
        "cp12_rmse": float(np.sqrt(np.mean(cp12_error**2))),
        "gain_vs_strong": float(np.sqrt(np.mean(strong_error**2)) - np.sqrt(np.mean(error**2))),
        "gain_vs_cp12": float(np.sqrt(np.mean(cp12_error**2)) - np.sqrt(np.mean(error**2))),
        "error_cosine_vs_strong": cosine,
        "rows": len(error),
    }, pd.DataFrame(well_rows), error, strong_error


_seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
started = time.time()
for directory in (OUT_DIR, CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
header = pd.read_csv(pool_path, nrows=0).columns.tolist()
required_pool = {"id", "tvt_true", "strong", "cp12"}
if not required_pool.issubset(header):
    raise RuntimeError(f"pool missing {sorted(required_pool - set(header))}")
usecols = [column for column in ("id", "well", "row_idx", "tvt_true", "strong", "cp12") if column in header]
pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
wells = sorted(pool["well"].unique())
fold_map, fold_source = _load_fold_map(wells)
horizontal_files = _discover_files(wells, "__horizontal_well.csv")
typewell_files = _discover_files(wells, "__typewell.csv")

panel_paths = {}
build_rows = []
for well, frame in tqdm(pool.groupby("well", sort=False), total=len(wells), desc="build RankSDF panels"):
    path = CACHE_DIR / f"{well}.npz"
    if not path.exists():
        record = _build_panel(well, horizontal_files[well], typewell_files[well], frame)
        np.savez_compressed(path, **record)
    with np.load(path, allow_pickle=False) as data:
        build_rows.append({"well": well, "fold": int(fold_map[well]), "coverage": float(data["coverage"][0]), "cache_mb": path.stat().st_size / 2**20})
    panel_paths[well] = path

build_report = pd.DataFrame(build_rows)
build_report.to_csv(OUT_DIR / "panel_build_report.csv", index=False)
train_paths = [panel_paths[well] for well in wells if fold_map[well] != VALID_FOLD]
valid_paths = [panel_paths[well] for well in wells if fold_map[well] == VALID_FOLD]
print({
    "experiment": "ranksdf_unet_onefold_v0",
    "device": str(DEVICE),
    "fold_source": fold_source,
    "valid_fold": VALID_FOLD,
    "train_wells": len(train_paths),
    "valid_wells": len(valid_paths),
    "cache_gb": sum(path.stat().st_size for path in panel_paths.values()) / 2**30,
    "mean_coverage": float(build_report["coverage"].mean()),
})

train_loader = DataLoader(PanelDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"), persistent_workers=NUM_WORKERS > 0)
model = TinyRankUNet(len(CHANNEL_NAMES), BASE_CHANNELS).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
offsets = torch.arange(-WIDTH_FT, WIDTH_FT + 0.25 * STATE_STEP_FT, STATE_STEP_FT, device=DEVICE)
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
best_rmse, best_epoch, stale = np.inf, 0, 0
best_path = MODEL_DIR / f"ranksdf_fold{VALID_FOLD}_best.pt"
last_path = MODEL_DIR / f"ranksdf_fold{VALID_FOLD}_last.pt"
selected_path = MODEL_DIR / f"ranksdf_fold{VALID_FOLD}_selected.pt"
history_rows = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    for panel, target_residual, known in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False):
        panel = panel.to(DEVICE, non_blocking=True)
        target_residual = target_residual.to(DEVICE, non_blocking=True)
        known = known.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(panel)
            loss = _loss(logits, panel, target_residual, known, offsets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    metrics, by_well, _, _ = _evaluate(model, valid_paths, DEVICE)
    row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
    history_rows.append(row)
    print("epoch", row, flush=True)
    current_checkpoint = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "channels": CHANNEL_NAMES,
        "selection_mode": "fixed_final_epoch" if FIXED_EPOCHS > 0 else "best_validation_epoch",
    }
    torch.save(current_checkpoint, last_path)
    if metrics["model_rmse"] < best_rmse:
        best_rmse, best_epoch, stale = metrics["model_rmse"], epoch, 0
        torch.save(current_checkpoint, best_path)
    else:
        stale += 1
    if FIXED_EPOCHS <= 0 and epoch >= 5 and stale >= PATIENCE:
        print("early stopping")
        break

source_path = last_path if FIXED_EPOCHS > 0 else best_path
checkpoint = torch.load(source_path, map_location=DEVICE, weights_only=False)
torch.save(checkpoint, selected_path)
model.load_state_dict(checkpoint["state_dict"])
metrics, by_well, model_error, strong_error = _evaluate(model, valid_paths, DEVICE)
by_well["improved_vs_strong"] = by_well["gain_vs_strong"] > 0
positive_fraction = float(by_well["improved_vs_strong"].mean())

# Top 20 validation wells by strong SSE; enough for a one-fold diagnostic.
by_well["strong_sse"] = (by_well["strong_rmse"] ** 2) * by_well["rows"]
top_wells = set(by_well.nlargest(20, "strong_sse")["well"])
top = by_well[by_well["well"].isin(top_wells)]
top_model_rmse = float(np.sqrt(np.sum((top["model_rmse"] ** 2) * top["rows"]) / top["rows"].sum()))
top_strong_rmse = float(np.sqrt(np.sum((top["strong_rmse"] ** 2) * top["rows"]) / top["rows"].sum()))

strict_pass = bool(
    metrics["gain_vs_strong"] >= PASS_GAIN_VS_STRONG
    and positive_fraction >= PASS_POSITIVE_WELL_FRACTION
)
verdict = {
    "experiment": "ranksdf_unet_onefold_v0",
    "valid_fold": VALID_FOLD,
    "best_epoch_diagnostic": best_epoch,
    "selected_epoch": int(checkpoint["epoch"]),
    "selection_mode": checkpoint["selection_mode"],
    **metrics,
    "positive_well_fraction": positive_fraction,
    "top20_gain_vs_strong": top_strong_rmse - top_model_rmse,
    "gates": {
        "gain_vs_strong_min": PASS_GAIN_VS_STRONG,
        "positive_well_fraction_min": PASS_POSITIVE_WELL_FRACTION,
    },
    "strict_pass": strict_pass,
    "next": "run remaining four OOF folds" if strict_pass else "stop or redesign RankSDF inputs before full OOF",
    "runtime_hours": (time.time() - started) / 3600.0,
    "selected_model": str(selected_path),
}
pd.DataFrame(history_rows).to_csv(OUT_DIR / "training_history.csv", index=False)
by_well.to_csv(OUT_DIR / "valid_by_well.csv", index=False)
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nVALID BY WELL")
print(by_well.sort_values("strong_sse", ascending=False).head(40).to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
