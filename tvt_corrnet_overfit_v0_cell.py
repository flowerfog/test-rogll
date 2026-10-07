"""TVT-CorrNet copy/overfit sanity on 24 real wells.

Unlike the failed absolute-S RankSDF, the state axis is relative TVT in
[-128,+128] ft. Horizontal and typewell GR patches are embedded by a shared
encoder, all-pairs correlation is decoded jointly by a small 2D U-Net, and the
final path is the posterior mean. No mixed precision is used before this test
passes.
"""

from pathlib import Path
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


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
OUT_DIR = OOF_ROOT / "tvt_corrnet_overfit_v0"
PANEL_DIR = OUT_DIR / "panels"
MODEL_PATH = OUT_DIR / "best_model.pt"

N_WELLS = 24
N_COLUMNS = 256
STATE_OFFSETS = np.arange(-128.0, 128.01, 2.0, dtype=np.float32)
PATCH_OFFSETS = np.arange(-64.0, 64.01, 2.0, dtype=np.float32)
EMBED_DIM = 32
BASE_CHANNELS = 16
BATCH_SIZE = 2
EPOCHS = 120
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
RANDOM_STATE = 20260717
REPORT_EVERY = 5

LOSS_CE = 1.0
LOSS_TVT = 0.20
LOSS_S_SLOPE = 0.05
LOSS_S_CURVATURE = 0.02
TARGET_SIGMA_FT = 2.0


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _robust_normalise(values):
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    output = np.zeros(values.shape, np.float32)
    if int(valid.sum()) < 4:
        return output
    center = float(np.median(values[valid]))
    scale = float(1.4826 * np.median(np.abs(values[valid] - center)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values[valid]), 1.0))
    output[valid] = np.clip((values[valid] - center) / scale, -8.0, 8.0)
    return output


def _interp_finite(x, y, query, left=np.nan, right=np.nan):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    query = np.asarray(query, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return np.full(query.shape, np.nan, np.float64)
    order = np.argsort(x[valid], kind="mergesort")
    xv = x[valid][order]
    yv = y[valid][order]
    unique, index = np.unique(xv, return_index=True)
    return np.interp(query, unique, yv[index], left=left, right=right)


def _nearest_distance(x, query):
    x = np.sort(np.asarray(x, np.float64)[np.isfinite(x)])
    query = np.asarray(query, np.float64)
    if not len(x):
        return np.full(query.shape, np.inf)
    position = np.searchsorted(x, query)
    left = np.abs(query - x[np.clip(position - 1, 0, len(x) - 1)])
    right = np.abs(query - x[np.clip(position, 0, len(x) - 1)])
    return np.minimum(left, right)


def _select_wells():
    rows = []
    for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv")):
        well = path.name.removesuffix("__horizontal_well.csv")
        frame = pd.read_csv(path, usecols=["TVT", "TVT_input", "GR"])
        tvt = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(float)
        tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        eval_rows = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(tvt))
        if len(known) < 30 or len(eval_rows) < 128:
            continue
        anchor_tvt = float(tvt_input[known[-1]])
        delta = tvt[eval_rows] - anchor_tvt
        rows.append(
            {
                "well": well,
                "delta_span": float(np.nanmax(delta) - np.nanmin(delta)),
                "delta_abs_max": float(np.nanmax(np.abs(delta))),
                "gr_missing": float(frame["GR"].isna().mean()),
                "eval_rows": len(eval_rows),
            }
        )
    audit = pd.DataFrame(rows)
    audit["difficulty"] = (
        audit["delta_span"].rank(pct=True)
        + audit["gr_missing"].rank(pct=True)
        + audit["eval_rows"].rank(pct=True)
    )
    audit["bin"] = pd.qcut(audit["difficulty"].rank(method="first"), 8, labels=False)
    selected = []
    for group in range(8):
        candidates = audit[audit["bin"] == group].sort_values(
            ["difficulty", "well"]
        )
        positions = np.unique(
            np.round(np.linspace(0, len(candidates) - 1, 3)).astype(int)
        )
        selected.extend(candidates.iloc[positions]["well"].astype(str).tolist())
    selected = selected[:N_WELLS]
    audit["selected"] = audit["well"].isin(selected)
    return selected, audit


def _build_panel(well):
    output = PANEL_DIR / f"{well}.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as cached:
            if "true_state_valid_fraction" in cached.files:
                return output
        output.unlink()
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    x = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    eval_rows = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
    anchor = int(known[-1])
    anchor_tvt = float(tvt_input[anchor])
    anchor_z = float(z[anchor])

    # Include the exact anchor as column zero, then the complete future span.
    column_md = np.linspace(md[anchor], md[eval_rows[-1]], N_COLUMNS)
    truth_tvt = _interp_finite(md, truth, column_md)
    truth_tvt[0] = anchor_tvt
    truth_delta = truth_tvt - anchor_tvt
    column_x = _interp_finite(md, x, column_md)
    column_y = _interp_finite(md, y, column_md)
    column_z = _interp_finite(md, z, column_md)

    horizontal_query = column_md[:, None] + PATCH_OFFSETS[None, :]
    horizontal_raw = _interp_finite(md, gr, horizontal_query, left=np.nan, right=np.nan)
    gr_norm_full = _robust_normalise(gr)
    finite_gr = np.isfinite(gr)
    horizontal_norm = _interp_finite(
        md[finite_gr],
        gr_norm_full[finite_gr],
        horizontal_query,
        left=np.nan,
        right=np.nan,
    )
    finite_gr_md = md[finite_gr]
    md_step = float(np.nanmedian(np.diff(md[np.isfinite(md)])))
    horizontal_mask = (
        _nearest_distance(finite_gr_md, horizontal_query) <= max(6.0, 2.5 * md_step)
    ).astype(np.float32)
    horizontal_norm = np.nan_to_num(horizontal_norm, nan=0.0).astype(np.float32)

    tw = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
    tw = tw.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(np.float64)
    tw_gr = tw["GR"].to_numpy(np.float64)
    tw_norm = _robust_normalise(tw_gr)
    state_tvt = anchor_tvt + STATE_OFFSETS
    typewell_query = state_tvt[:, None] + PATCH_OFFSETS[None, :]
    typewell_patch = _interp_finite(
        tw_tvt, tw_norm, typewell_query, left=np.nan, right=np.nan
    )
    typewell_mask = (
        (typewell_query >= tw_tvt.min()) & (typewell_query <= tw_tvt.max())
    ).astype(np.float32)
    typewell_patch = np.nan_to_num(typewell_patch, nan=0.0).astype(np.float32)
    state_valid = (typewell_mask.mean(axis=1) >= 0.50).astype(np.uint8)

    progress = (column_md - column_md[0]) / max(column_md[-1] - column_md[0], 1.0)
    dz = np.gradient(column_z, column_md, edge_order=1)
    dx = np.gradient(column_x, column_md, edge_order=1)
    dy = np.gradient(column_y, column_md, edge_order=1)
    geometry = np.stack(
        [
            progress,
            (column_x - column_x[0]) / 1000.0,
            (column_y - column_y[0]) / 1000.0,
            (column_z - column_z[0]) / 1000.0,
            dx,
            dy,
            dz,
            np.full(N_COLUMNS, column_x[0] / 100000.0),
            np.full(N_COLUMNS, column_y[0] / 100000.0),
            horizontal_mask.mean(axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    valid_offsets = STATE_OFFSETS[state_valid.astype(bool)]
    if len(valid_offsets) < 2:
        raise RuntimeError("too few typewell-valid states")
    teacher_index = np.argmin(
        np.abs(valid_offsets[:, None] - truth_delta[None, :]), axis=0
    )
    teacher_delta = valid_offsets[teacher_index]
    teacher_rmse = float(np.sqrt(np.mean((teacher_delta - truth_delta) ** 2)))
    label_coverage = float(np.mean(np.abs(truth_delta) <= np.max(np.abs(STATE_OFFSETS))))
    nearest_full_index = np.argmin(
        np.abs(STATE_OFFSETS[:, None] - truth_delta[None, :]), axis=0
    )
    true_state_valid_fraction = float(np.mean(state_valid[nearest_full_index]))

    np.savez_compressed(
        output,
        horizontal_patch=np.stack([horizontal_norm, horizontal_mask], axis=1).astype(np.float32),
        typewell_patch=np.stack([typewell_patch, typewell_mask], axis=1).astype(np.float32),
        geometry=geometry,
        state_valid=state_valid,
        state_offsets=STATE_OFFSETS,
        column_md=column_md.astype(np.float32),
        column_z=column_z.astype(np.float32),
        truth_delta=truth_delta.astype(np.float32),
        eval_rows=eval_rows.astype(np.int32),
        eval_md=md[eval_rows].astype(np.float32),
        eval_truth=truth[eval_rows].astype(np.float32),
        anchor_tvt=np.asarray([anchor_tvt], np.float32),
        anchor_z=np.asarray([anchor_z], np.float32),
        teacher_rmse=np.asarray([teacher_rmse], np.float32),
        label_coverage=np.asarray([label_coverage], np.float32),
        true_state_valid_fraction=np.asarray([true_state_valid_fraction], np.float32),
    )
    return output


class CorrPanelDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as data:
            return {
                "horizontal": torch.from_numpy(data["horizontal_patch"]),
                "typewell": torch.from_numpy(data["typewell_patch"]),
                "geometry": torch.from_numpy(data["geometry"]),
                "state_valid": torch.from_numpy(data["state_valid"].astype(bool)),
                "truth_delta": torch.from_numpy(data["truth_delta"]),
                "column_z": torch.from_numpy(data["column_z"]),
            }


class PatchEncoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 24, 5, padding=2),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv1d(24, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, embedding, 3, stride=2, padding=1),
            nn.GroupNorm(8, embedding),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, values):
        return self.network(values).squeeze(-1)


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        groups = 8 if output_channels >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, values):
        return self.block(values)


class CorrNet(nn.Module):
    def __init__(self, geometry_channels, embedding=EMBED_DIM, base=BASE_CHANNELS):
        super().__init__()
        self.patch_encoder = PatchEncoder(embedding)
        input_channels = 2 + geometry_channels
        self.enc1 = ConvBlock(input_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.bottleneck = ConvBlock(base * 2, base * 4)
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, horizontal, typewell, geometry, state_offsets):
        batch, columns, _, patch = horizontal.shape
        states = typewell.shape[1]
        h = self.patch_encoder(horizontal.reshape(batch * columns, 2, patch))
        t = self.patch_encoder(typewell.reshape(batch * states, 2, patch))
        h = F.normalize(h.reshape(batch, columns, -1), dim=-1)
        t = F.normalize(t.reshape(batch, states, -1), dim=-1)
        correlation = torch.einsum("bhd,bwd->bhw", t, h).unsqueeze(1)
        state_channel = state_offsets[None, :, None].expand(batch, states, columns)
        geometry_channels = geometry[:, :, None, :].expand(
            batch, geometry.shape[1], states, columns
        )
        values = torch.cat(
            [correlation, state_channel[:, None] / 128.0, geometry_channels], dim=1
        )
        e1 = self.enc1(values)
        e2 = self.enc2(F.avg_pool2d(e1, 2, ceil_mode=True))
        bottleneck = self.bottleneck(F.avg_pool2d(e2, 2, ceil_mode=True))
        up2 = F.interpolate(bottleneck, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        residual_logits = self.head(d1).squeeze(1)
        return 5.0 * correlation.squeeze(1) + residual_logits


def _loss(logits, truth_delta, column_z, state_valid, state_offsets):
    valid_logits = logits.masked_fill(~state_valid[:, :, None], -30.0)
    distance = state_offsets[None, :, None] - truth_delta[:, None, :]
    target = torch.softmax(-0.5 * (distance / TARGET_SIGMA_FT) ** 2, dim=1)
    target = target * state_valid[:, :, None]
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_probability = F.log_softmax(valid_logits, dim=1)
    column_weight = torch.ones_like(truth_delta)
    column_weight[:, 0] = 3.0
    ce_by_column = -(target * log_probability).sum(dim=1)
    ce = (ce_by_column * column_weight).sum() / column_weight.sum()

    probability = torch.softmax(valid_logits, dim=1)
    prediction = (probability * state_offsets[None, :, None]).sum(dim=1)
    tvt = F.smooth_l1_loss(prediction, truth_delta, beta=2.0)

    predicted_s = prediction + column_z - column_z[:, :1]
    true_s = truth_delta + column_z - column_z[:, :1]
    slope = F.smooth_l1_loss(
        predicted_s[:, 1:] - predicted_s[:, :-1],
        true_s[:, 1:] - true_s[:, :-1],
        beta=0.5,
    )
    curvature = F.smooth_l1_loss(
        predicted_s[:, 2:] - 2 * predicted_s[:, 1:-1] + predicted_s[:, :-2],
        true_s[:, 2:] - 2 * true_s[:, 1:-1] + true_s[:, :-2],
        beta=0.25,
    )
    total = LOSS_CE * ce + LOSS_TVT * tvt + LOSS_S_SLOPE * slope + LOSS_S_CURVATURE * curvature
    return total, {"ce": ce, "tvt": tvt, "slope": slope, "curvature": curvature}, prediction


@torch.no_grad()
def _predict(model, panel_path, device):
    with np.load(panel_path, allow_pickle=False) as data:
        horizontal = torch.from_numpy(data["horizontal_patch"])[None].to(device)
        typewell = torch.from_numpy(data["typewell_patch"])[None].to(device)
        geometry = torch.from_numpy(data["geometry"])[None].to(device)
        state_valid = torch.from_numpy(data["state_valid"].astype(bool))[None].to(device)
        offsets = torch.from_numpy(data["state_offsets"]).to(device)
        logits = model(horizontal, typewell, geometry, offsets)
        logits = logits.masked_fill(~state_valid[:, :, None], -30.0)
        probability = torch.softmax(logits, dim=1)
        delta = (probability * offsets[None, :, None]).sum(dim=1)[0].cpu().numpy()
        delta[0] = 0.0
        column_md = data["column_md"].astype(float)
        eval_md = data["eval_md"].astype(float)
        prediction = float(data["anchor_tvt"][0]) + np.interp(eval_md, column_md, delta)
        truth = data["eval_truth"].astype(float)
    return prediction, truth


@torch.no_grad()
def _evaluate(model, paths, device):
    rows = []
    all_error = []
    for path in paths:
        prediction, truth = _predict(model, path, device)
        error = prediction - truth
        all_error.append(error)
        rows.append(
            {
                "well": path.stem,
                "rows": len(error),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
            }
        )
    error = np.concatenate(all_error)
    return {
        "pooled_rmse": float(np.sqrt(np.mean(error**2))),
        "pooled_mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median([row["rmse"] for row in rows])),
        "p95_well_rmse": float(np.quantile([row["rmse"] for row in rows], 0.95)),
    }, pd.DataFrame(rows)


for directory in (OUT_DIR, PANEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)
_seed_everything(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")

selected_wells, selection_audit = _select_wells()
selection_audit.to_csv(OUT_DIR / "well_selection.csv", index=False)
panel_paths = [_build_panel(well) for well in tqdm(selected_wells, desc="build TVT-CorrNet panels")]

teacher_rmse = []
coverage = []
true_state_valid = []
for path in panel_paths:
    with np.load(path, allow_pickle=False) as data:
        teacher_rmse.append(float(data["teacher_rmse"][0]))
        coverage.append(float(data["label_coverage"][0]))
        true_state_valid.append(float(data["true_state_valid_fraction"][0]))
guards = {
    "wells": len(panel_paths),
    "teacher_decode_rmse": float(np.mean(teacher_rmse)),
    "mean_label_coverage": float(np.mean(coverage)),
    "min_label_coverage": float(np.min(coverage)),
    "mean_true_state_valid": float(np.mean(true_state_valid)),
    "min_true_state_valid": float(np.min(true_state_valid)),
}
if (
    guards["teacher_decode_rmse"] >= 1.0
    or guards["min_label_coverage"] < 0.995
    or guards["min_true_state_valid"] < 0.95
):
    raise RuntimeError(f"panel/axis guard failed: {guards}")

dataset = CorrPanelDataset(panel_paths)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)
with np.load(panel_paths[0], allow_pickle=False) as first:
    geometry_channels = first["geometry"].shape[0]
model = CorrNet(geometry_channels).to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
offset_tensor = torch.from_numpy(STATE_OFFSETS).to(DEVICE)

print(
    {
        "experiment": "tvt_corrnet_overfit_v0",
        "device": str(DEVICE),
        "wells": selected_wells,
        "states": len(STATE_OFFSETS),
        "columns": N_COLUMNS,
        "patch": len(PATCH_OFFSETS),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "guards": guards,
    }
)

started = time.time()
history = []
best_rmse = np.inf
for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_rows = []
    for batch in loader:
        horizontal = batch["horizontal"].to(DEVICE, non_blocking=True)
        typewell = batch["typewell"].to(DEVICE, non_blocking=True)
        geometry = batch["geometry"].to(DEVICE, non_blocking=True)
        state_valid = batch["state_valid"].to(DEVICE, non_blocking=True)
        truth_delta = batch["truth_delta"].to(DEVICE, non_blocking=True)
        column_z = batch["column_z"].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(horizontal, typewell, geometry, offset_tensor)
        loss, components, _ = _loss(
            logits, truth_delta, column_z, state_valid, offset_tensor
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_rows.append(
            {"loss": float(loss.detach().cpu()), **{k: float(v.detach().cpu()) for k, v in components.items()}}
        )

    if epoch == 1 or epoch % REPORT_EVERY == 0 or epoch == EPOCHS:
        model.eval()
        metrics, by_well = _evaluate(model, panel_paths, DEVICE)
        record = {
            "epoch": epoch,
            **{f"train_{key}": float(np.mean([row[key] for row in loss_rows])) for key in loss_rows[0]},
            **metrics,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(record)
        print("epoch", record, flush=True)
        if metrics["pooled_rmse"] < best_rmse:
            best_rmse = metrics["pooled_rmse"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "config": {
                        "state_offsets": STATE_OFFSETS.tolist(),
                        "patch_offsets": PATCH_OFFSETS.tolist(),
                        "columns": N_COLUMNS,
                    },
                },
                MODEL_PATH,
            )
            by_well.to_csv(OUT_DIR / "best_bywell.csv", index=False)
        if (
            epoch >= 20
            and metrics["pooled_rmse"] < 0.90
            and metrics["p95_well_rmse"] < 1.80
        ):
            print("overfit gate cleared; stopping early", flush=True)
            break

history_frame = pd.DataFrame(history)
history_frame.to_csv(OUT_DIR / "history.csv", index=False)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()
final_metrics, final_bywell = _evaluate(model, panel_paths, DEVICE)
final_bywell.to_csv(OUT_DIR / "final_bywell.csv", index=False)

strict_pass = bool(
    final_metrics["pooled_rmse"] < 1.0
    and final_metrics["p95_well_rmse"] < 2.0
)
verdict = {
    "experiment": "tvt_corrnet_overfit_v0",
    "device": str(DEVICE),
    "wells": len(panel_paths),
    "guards": guards,
    "best_epoch": int(checkpoint["epoch"]),
    "metrics": final_metrics,
    "training_hours": (time.time() - started) / 3600.0,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> run one held-out fold with negative controls"
        if strict_pass
        else "FAIL -> debug/capacity-tune CorrNet before any OOF run"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
