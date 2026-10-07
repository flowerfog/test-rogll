"""RankSDF Step 1: 12-well plumbing/overfit test for a 2D Rank-UNet.

The model sees only deployable inputs:

* horizontal MD/Z/GR and the visible TVT_input prefix;
* the assigned typewell TVT/GR curve;
* a prefix-only S = TVT + Z trend used to centre the candidate axis.

TVT is read only to build training labels and evaluate reconstruction.  This is
not an OOF score.  Its sole purpose is to prove that tensor orientation, masks,
soft labels, interpolation and decoding are internally capable of fitting the
training examples before spending compute on cross-validation.
"""

from pathlib import Path
import json
import random
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from torch.utils.data import Dataset, DataLoader


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
COVERAGE_DIR = OOF_ROOT / "ranksdf_panel_coverage_v0"
OUT_DIR = OOF_ROOT / "ranksdf_unet_overfit_v0_w12_s257_c512"
CACHE_DIR = OUT_DIR / "panels"

POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]

N_WELLS = 12
WIDTH_FT = 256.0
STATE_STEP_FT = 2.0
N_COLUMNS = 512
PREFIX_CONTEXT_FT = 1000.0
SOFT_LABEL_SIGMA_FT = 3.0

BASE_CHANNELS = 16
BATCH_SIZE = 2
EPOCHS = 80
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
SEED = 20260716

PASS_EVAL_RMSE = 1.50
PASS_KNOWN_RMSE = 0.75
PASS_PANEL_COVERAGE = 0.995

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
    values = np.asarray(values, np.float64)
    return (
        pd.Series(values)
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _robust_normalise(values, valid=None):
    values = np.asarray(values, np.float32)
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(valid, bool) & np.isfinite(values)
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
    md = np.asarray(md, np.float64)
    structure = np.asarray(structure, np.float64)
    x = md - float(anchor_md)
    y = structure - float(anchor_structure)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
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
    selected = {}
    for root in (DATA_ROOT / "train", DATA_ROOT):
        if not root.exists():
            continue
        for path in root.rglob(f"*{suffix}"):
            name = path.name
            if not name.endswith(suffix):
                continue
            well = name[: -len(suffix)]
            if well in wells and "test" not in {part.lower() for part in path.parts}:
                selected.setdefault(well, path)
    missing = sorted(wells - set(selected))
    if missing:
        raise FileNotFoundError(f"missing {suffix}: {missing[:20]}")
    return selected


def _stable_pick(frame):
    ranked = frame.copy()
    ranked["hash"] = ranked["well"].map(
        lambda value: zlib.crc32(f"{value}|{SEED}".encode())
    )
    return str(ranked.sort_values("hash").iloc[0]["well"])


def _select_wells(pool):
    if "strong" in pool:
        pool = pool.copy()
        pool["se"] = (
            pool["strong"].to_numpy(float) - pool["tvt_true"].to_numpy(float)
        ) ** 2
        meta = (
            pool.groupby("well", as_index=False)
            .agg(rows=("id", "size"), sse=("se", "sum"))
        )
        meta["rmse"] = np.sqrt(meta["sse"] / meta["rows"])
    else:
        meta = pool.groupby("well", as_index=False).agg(rows=("id", "size"))
        meta["rmse"] = 0.0
    meta["error_bin"] = pd.qcut(
        meta["rmse"].rank(method="first"), 4, labels=False
    ).astype(int)
    meta["length_bin"] = pd.qcut(
        meta["rows"].rank(method="first"), 3, labels=False
    ).astype(int)
    selected = []
    for error_bin in range(4):
        for length_bin in range(3):
            cell = meta[
                (meta["error_bin"] == error_bin)
                & (meta["length_bin"] == length_bin)
            ]
            if not cell.empty:
                selected.append(_stable_pick(cell))
    if len(selected) < N_WELLS:
        remaining = meta[~meta["well"].isin(selected)]
        remaining = remaining.assign(
            hash=remaining["well"].map(
                lambda value: zlib.crc32(f"extra|{value}|{SEED}".encode())
            )
        ).sort_values("hash")
        selected.extend(remaining["well"].astype(str).head(N_WELLS - len(selected)))
    return selected[:N_WELLS], meta


def _build_panel(well, horizontal_path, typewell_path):
    horizontal = pd.read_csv(horizontal_path)
    required = {"MD", "Z", "GR", "TVT", "TVT_input"}
    if not required.issubset(horizontal.columns):
        raise RuntimeError(
            f"{well} horizontal missing {sorted(required - set(horizontal.columns))}"
        )
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    typewell = typewell[np.isfinite(typewell["TVT"])].sort_values("TVT")
    typewell = typewell.groupby("TVT", as_index=False)["GR"].mean()

    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md) & np.isfinite(z))
    if len(known) < 32:
        raise RuntimeError(f"insufficient known prefix for {well}")
    anchor = int(known[-1])
    anchor_md = float(md[anchor])
    anchor_s = float(tvt_input[anchor] + z[anchor])
    recent = known[md[known] >= anchor_md - 500.0]
    if len(recent) < 16:
        recent = known[-min(64, len(known)):]
    slope = _robust_slope(
        md[recent], tvt_input[recent] + z[recent], anchor_md, anchor_s
    )

    start_md = max(float(md[0]), anchor_md - PREFIX_CONTEXT_FT)
    end_md = float(md[-1])
    sample_md = np.linspace(start_md, end_md, N_COLUMNS, dtype=np.float64)
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

    gr_valid_original = np.isfinite(gr_raw).astype(np.float32)
    gr_filled = _fill_1d(gr_raw)
    sample_gr = np.interp(sample_md, md, gr_filled).astype(np.float32)
    sample_gr_valid = np.interp(sample_md, md, gr_valid_original) >= 0.50
    horizontal_gr = _robust_normalise(sample_gr, sample_gr_valid)

    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr_raw = typewell["GR"].to_numpy(float)
    tw_gr_filled = _fill_1d(tw_gr_raw)
    tw_gr_norm = _robust_normalise(tw_gr_filled, np.isfinite(tw_gr_raw))

    offsets = np.arange(
        -WIDTH_FT, WIDTH_FT + 0.25 * STATE_STEP_FT, STATE_STEP_FT,
        dtype=np.float32,
    )
    candidate_s = center_s[None, :] + offsets[:, None]
    candidate_tvt = candidate_s - sample_z[None, :]
    typewell_valid = (candidate_tvt >= tw_tvt[0]) & (candidate_tvt <= tw_tvt[-1])
    candidate_tw_gr = np.interp(
        candidate_tvt.reshape(-1), tw_tvt, tw_gr_norm
    ).reshape(len(offsets), N_COLUMNS).astype(np.float32)
    candidate_tw_gr[~typewell_valid] = 0.0

    horizontal_grid = np.broadcast_to(
        horizontal_gr[None, :], candidate_tw_gr.shape
    ).astype(np.float32)
    difference = candidate_tw_gr - horizontal_grid
    abs_difference = np.abs(difference)
    smooth5 = uniform_filter1d(difference, size=5, axis=1, mode="nearest")
    smooth21 = uniform_filter1d(difference, size=21, axis=1, mode="nearest")
    smooth51 = uniform_filter1d(difference, size=51, axis=1, mode="nearest")

    known_path = np.zeros_like(candidate_tw_gr, np.float32)
    if known_columns.any():
        known_path[:, known_columns] = np.exp(
            -0.5
            * (
                (offsets[:, None] - target_residual[None, known_columns])
                / SOFT_LABEL_SIGMA_FT
            )
            ** 2
        ).astype(np.float32)
    known_mask_grid = np.broadcast_to(
        known_columns[None, :], candidate_tw_gr.shape
    ).astype(np.float32)
    offset_grid = np.broadcast_to(
        (offsets[:, None] / WIDTH_FT), candidate_tw_gr.shape
    ).astype(np.float32)
    md_relative = ((sample_md - anchor_md) / max(end_md - anchor_md, 1.0)).astype(
        np.float32
    )
    md_grid = np.broadcast_to(md_relative[None, :], candidate_tw_gr.shape).astype(
        np.float32
    )
    horizontal_valid_grid = np.broadcast_to(
        sample_gr_valid[None, :], candidate_tw_gr.shape
    ).astype(np.float32)

    panel = np.stack(
        [
            candidate_tw_gr,
            horizontal_grid,
            difference,
            abs_difference,
            smooth5,
            smooth21,
            smooth51,
            typewell_valid.astype(np.float32),
            known_path,
            known_mask_grid,
            offset_grid,
            md_grid,
            horizontal_valid_grid,
        ],
        axis=0,
    ).astype(np.float16)

    eval_rows = np.flatnonzero(~np.isfinite(tvt_input))
    if not len(eval_rows):
        raise RuntimeError(f"no eval rows for {well}")
    coverage = float(np.mean(np.abs(true_residual) <= WIDTH_FT + 1e-6))
    return {
        "panel": panel,
        "offsets": offsets,
        "sample_md": sample_md.astype(np.float32),
        "sample_z": sample_z.astype(np.float32),
        "center_s": center_s.astype(np.float32),
        "true_s": true_s.astype(np.float32),
        "true_residual": target_residual.astype(np.float32),
        "known_columns": known_columns.astype(np.uint8),
        "eval_md": md[eval_rows].astype(np.float32),
        "eval_tvt": tvt[eval_rows].astype(np.float32),
        "coverage": np.asarray([coverage], np.float32),
        "anchor_md": np.asarray([anchor_md], np.float32),
        "anchor_s": np.asarray([anchor_s], np.float32),
        "slope": np.asarray([slope], np.float32),
    }


class PanelDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as data:
            return {
                "well": self.paths[index].stem,
                "panel": torch.from_numpy(data["panel"].astype(np.float32)),
                "true_residual": torch.from_numpy(data["true_residual"].astype(np.float32)),
                "known_columns": torch.from_numpy(data["known_columns"].astype(bool)),
            }


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, values):
        return self.network(values)


class TinyRankUNet(nn.Module):
    def __init__(self, in_channels, base=16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 6)
        self.bottleneck = ConvBlock(base * 6, base * 8)
        self.dec4 = ConvBlock(base * 8 + base * 6, base * 6)
        self.dec3 = ConvBlock(base * 6 + base * 4, base * 4)
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2 + base, base)
        self.output = nn.Conv2d(base, 1, 1)

    @staticmethod
    def _up(values, skip):
        return F.interpolate(
            values, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, values):
        e1 = self.enc1(values)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        hidden = self.bottleneck(F.max_pool2d(e4, 2))
        hidden = self.dec4(torch.cat([self._up(hidden, e4), e4], dim=1))
        hidden = self.dec3(torch.cat([self._up(hidden, e3), e3], dim=1))
        hidden = self.dec2(torch.cat([self._up(hidden, e2), e2], dim=1))
        hidden = self.dec1(torch.cat([self._up(hidden, e1), e1], dim=1))
        return self.output(hidden).squeeze(1)


def _soft_ce_loss(logits, panel, true_residual, known_columns, offsets):
    valid_states = panel[:, TYPEWELL_VALID_CHANNEL] > 0.5
    logits = logits.masked_fill(~valid_states, -30.0)
    difference = offsets[None, :, None] - true_residual[:, None, :]
    target = torch.exp(-0.5 * (difference / SOFT_LABEL_SIGMA_FT) ** 2)
    target = target * valid_states.float()
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    column_loss = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    column_weight = torch.where(
        known_columns,
        torch.full_like(column_loss, 0.50),
        torch.ones_like(column_loss),
    )
    return (column_loss * column_weight).sum() / column_weight.sum().clamp_min(1.0)


@torch.no_grad()
def _predict_panel(model, panel, offsets, device):
    values = torch.from_numpy(panel.astype(np.float32))[None].to(device)
    logits = model(values)
    valid = values[:, TYPEWELL_VALID_CHANNEL] > 0.5
    logits = logits.masked_fill(~valid, -30.0)
    probability = torch.softmax(logits, dim=1)
    offset_tensor = torch.from_numpy(offsets.astype(np.float32)).to(device)
    prediction = torch.sum(probability * offset_tensor[None, :, None], dim=1)
    return prediction[0].cpu().numpy(), probability[0].cpu().numpy()


def _evaluate(model, paths, device):
    rows = []
    total_eval_sse = 0.0
    total_eval_rows = 0
    total_known_sse = 0.0
    total_known_rows = 0
    model.eval()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            panel = data["panel"].copy()
            offsets = data["offsets"].copy()
            center_s = data["center_s"].astype(float)
            sample_z = data["sample_z"].astype(float)
            eval_md = data["eval_md"].astype(float)
            sample_md = data["sample_md"].astype(float)
            eval_tvt = data["eval_tvt"].astype(float)
            known = data["known_columns"].astype(bool)
            true_s = data["true_s"].astype(float)
            target_residual = data["true_residual"].astype(float)
            coverage = float(data["coverage"][0])
        predicted_offset, probability = _predict_panel(
            model, panel, offsets, device
        )
        # Deployment knows these columns exactly.  They are a hard path anchor,
        # not values the network is asked to re-predict.
        predicted_offset[known] = target_residual[known]
        predicted_s = center_s + predicted_offset
        predicted_tvt_samples = predicted_s - sample_z
        eval_prediction = np.interp(eval_md, sample_md, predicted_tvt_samples)
        eval_error = eval_prediction - eval_tvt
        sample_true_tvt = true_s - sample_z
        known_error = predicted_tvt_samples[known] - sample_true_tvt[known]
        total_eval_sse += float(np.sum(eval_error**2))
        total_eval_rows += len(eval_error)
        total_known_sse += float(np.sum(known_error**2))
        total_known_rows += int(known.sum())
        entropy = -np.sum(
            probability * np.log(np.maximum(probability, 1e-12)), axis=0
        )
        rows.append(
            {
                "well": path.stem,
                "eval_rows": len(eval_error),
                "eval_rmse": float(np.sqrt(np.mean(eval_error**2))),
                "known_columns": int(known.sum()),
                "known_rmse": float(np.sqrt(np.mean(known_error**2))),
                "panel_coverage": coverage,
                "mean_entropy": float(np.mean(entropy)),
                "p95_abs_predicted_offset": float(
                    np.quantile(np.abs(predicted_offset), 0.95)
                ),
            }
        )
    return {
        "eval_rmse": float(np.sqrt(total_eval_sse / total_eval_rows)),
        "known_rmse": float(np.sqrt(total_known_sse / total_known_rows)),
        "eval_rows": total_eval_rows,
        "known_rows": total_known_rows,
    }, pd.DataFrame(rows)


_seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
started = time.time()
for directory in (OUT_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

coverage_path = COVERAGE_DIR / "selected_panel.json"
if not coverage_path.exists():
    raise FileNotFoundError(
        f"Run ranksdf_panel_coverage_v0_cell.py first; missing {coverage_path}"
    )
with coverage_path.open(encoding="utf-8") as handle:
    coverage_verdict = json.load(handle)
if not coverage_verdict.get("strict_pass"):
    raise RuntimeError("panel coverage gate did not pass")

pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
header = pd.read_csv(pool_path, nrows=0).columns.tolist()
usecols = [column for column in ("id", "well", "tvt_true", "strong") if column in header]
pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
pool["well"] = pool["well"].astype(str)
selected_wells, selection_meta = _select_wells(pool)
horizontal_files = _discover_files(selected_wells, "__horizontal_well.csv")
typewell_files = _discover_files(selected_wells, "__typewell.csv")
selection_meta[selection_meta["well"].isin(selected_wells)].to_csv(
    OUT_DIR / "selected_wells.csv", index=False
)

print(
    {
        "experiment": "ranksdf_unet_overfit_v0",
        "device": str(DEVICE),
        "wells": selected_wells,
        "panel": {
            "width_ft": WIDTH_FT,
            "state_step_ft": STATE_STEP_FT,
            "states": int(2 * WIDTH_FT / STATE_STEP_FT) + 1,
            "columns": N_COLUMNS,
            "channels": len(CHANNEL_NAMES),
            "centre": "prefix_last500",
        },
    }
)

panel_paths = []
build_rows = []
for well in selected_wells:
    path = CACHE_DIR / f"{well}.npz"
    if not path.exists():
        record = _build_panel(well, horizontal_files[well], typewell_files[well])
        np.savez_compressed(path, **record)
    with np.load(path, allow_pickle=False) as data:
        panel_shape = tuple(data["panel"].shape)
        panel_coverage = float(data["coverage"][0])
        anchor_md = float(data["anchor_md"][0])
        anchor_s = float(data["anchor_s"][0])
        prefix_slope = float(data["slope"][0])
    panel_paths.append(path)
    build_rows.append(
        {
            "well": well,
            "shape": str(panel_shape),
            "coverage": panel_coverage,
            "anchor_md": anchor_md,
            "anchor_s": anchor_s,
            "prefix_slope": prefix_slope,
            "cache_mb": path.stat().st_size / 2**20,
        }
    )
build_report = pd.DataFrame(build_rows)
build_report.to_csv(OUT_DIR / "panel_build_report.csv", index=False)
print("\nPANEL BUILD")
print(build_report.to_string(index=False))

dataset = PanelDataset(panel_paths)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)
model = TinyRankUNet(len(CHANNEL_NAMES), BASE_CHANNELS).to(DEVICE)
parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
offsets_t = torch.arange(
    -WIDTH_FT, WIDTH_FT + 0.25 * STATE_STEP_FT, STATE_STEP_FT,
    dtype=torch.float32,
    device=DEVICE,
)
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

history_rows = []
best_eval_rmse = np.inf
best_epoch = 0
best_path = OUT_DIR / "ranksdf_overfit_best.pt"
for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    for batch in loader:
        panel = batch["panel"].to(DEVICE, non_blocking=True)
        true_residual = batch["true_residual"].to(DEVICE, non_blocking=True)
        known_columns = batch["known_columns"].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(panel)
            loss = _soft_ce_loss(
                logits, panel, true_residual, known_columns, offsets_t
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))

    if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
        metrics, by_well = _evaluate(model, panel_paths, DEVICE)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            **metrics,
        }
        history_rows.append(row)
        print("epoch", row, flush=True)
        if metrics["eval_rmse"] < best_eval_rmse:
            best_eval_rmse = metrics["eval_rmse"]
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "channels": CHANNEL_NAMES,
                    "width_ft": WIDTH_FT,
                    "state_step_ft": STATE_STEP_FT,
                    "columns": N_COLUMNS,
                    "base_channels": BASE_CHANNELS,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                best_path,
            )
        if (
            metrics["eval_rmse"] <= PASS_EVAL_RMSE
            and metrics["known_rmse"] <= PASS_KNOWN_RMSE
            and epoch >= 10
        ):
            print("overfit gate reached; stopping early")
            break

checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
final_metrics, final_by_well = _evaluate(model, panel_paths, DEVICE)
history = pd.DataFrame(history_rows)
history.to_csv(OUT_DIR / "training_history.csv", index=False)
final_by_well.to_csv(OUT_DIR / "overfit_by_well.csv", index=False)

mean_coverage = float(build_report["coverage"].mean())
strict_pass = bool(
    final_metrics["eval_rmse"] <= PASS_EVAL_RMSE
    and final_metrics["known_rmse"] <= PASS_KNOWN_RMSE
    and mean_coverage >= PASS_PANEL_COVERAGE
)
verdict = {
    "experiment": "ranksdf_unet_overfit_v0",
    "device": str(DEVICE),
    "wells": selected_wells,
    "parameters": parameter_count,
    "best_epoch": best_epoch,
    "best_metrics": final_metrics,
    "mean_panel_coverage": mean_coverage,
    "gates": {
        "eval_rmse_max": PASS_EVAL_RMSE,
        "known_rmse_max": PASS_KNOWN_RMSE,
        "panel_coverage_min": PASS_PANEL_COVERAGE,
    },
    "strict_pass": strict_pass,
    "next": (
        "build one-fold honest Rank-UNet v0"
        if strict_pass
        else "debug representation/model before any OOF training"
    ),
    "runtime_minutes": (time.time() - started) / 60.0,
    "model": str(best_path),
}
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nOVERFIT BY WELL")
print(final_by_well.to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
