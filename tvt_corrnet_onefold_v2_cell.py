"""Held-out fold-1 TVT-CorrNet with trained negative controls.

The main arm uses only legal single-well inputs. Two equal-capacity controls
are trained from scratch:

* no_gr: horizontal/typewell GR values are zeroed while masks and geometry
  remain available;
* shuffled_typewell: each horizontal well receives another well's typewell.

The full model is also evaluated under both interventions after training. A
five-fold run is allowed only when predictive value and causal input use are
both demonstrated on fold 1.
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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
OUT_DIR = OOF_ROOT / "tvt_corrnet_onefold_v2"
PANEL_DIR = OUT_DIR / "panels"
MODEL_DIR = OUT_DIR / "models"

FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "gr_path_bank_oracle_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)

VALID_FOLD = 1
N_COLUMNS = 256
STATE_OFFSETS = np.arange(-128.0, 128.01, 2.0, dtype=np.float32)
PATCH_OFFSETS = np.arange(-64.0, 64.01, 2.0, dtype=np.float32)
EMBED_DIM = 64
BASE_CHANNELS = 48
BATCH_SIZE = 4
EPOCHS = 50
WARMUP_EPOCHS = 25
REPORT_EVERY = 2
EARLY_STOPPING_REPORTS = 7
LEARNING_RATE = 5e-4
MIN_LEARNING_RATE = 5e-5
RANDOM_STATE = 20260717
TARGET_SIGMA_FT = 2.0

LOSS_CE = 1.0
LOSS_TVT = 0.08
LOSS_S_SLOPE = 0.02
LOSS_S_CURVATURE = 0.01

ARMS = ("full", "no_gr", "shuffled_typewell")
BLEND_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def _first_existing(paths):
    return next((path for path in paths if path.exists()), None)


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


def _build_panel(well):
    output = PANEL_DIR / f"{well}.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as cached:
            if "true_state_valid_fraction" in cached.files:
                return {
                    "well": well,
                    "status": "cached",
                    "columns": int(cached["horizontal_patch"].shape[0]),
                }
        output.unlink()
    try:
        horizontal = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        )
        typewell = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__typewell.csv"
        )
        required = {"MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"}
        if not required.issubset(horizontal.columns):
            raise RuntimeError(
                f"missing horizontal columns {sorted(required-set(horizontal.columns))}"
            )
        if not {"TVT", "GR"}.issubset(typewell.columns):
            raise RuntimeError("typewell misses TVT/GR")

        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
        x = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(float)
        z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
        gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
        tvt_input = pd.to_numeric(
            horizontal["TVT_input"], errors="coerce"
        ).to_numpy(float)
        truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        eval_rows = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
        if len(known) < 30 or len(eval_rows) < 128:
            raise RuntimeError("too few prefix/eval rows")
        anchor = int(known[-1])
        anchor_tvt = float(tvt_input[anchor])

        column_md = np.linspace(md[anchor], md[eval_rows[-1]], N_COLUMNS)
        truth_tvt = _interp_finite(md, truth, column_md)
        truth_tvt[0] = anchor_tvt
        truth_delta = truth_tvt - anchor_tvt
        column_x = _interp_finite(md, x, column_md)
        column_y = _interp_finite(md, y, column_md)
        column_z = _interp_finite(md, z, column_md)

        horizontal_query = column_md[:, None] + PATCH_OFFSETS[None, :]
        gr_norm = _robust_normalise(gr)
        finite_gr = np.isfinite(gr)
        horizontal_patch = _interp_finite(
            md[finite_gr],
            gr_norm[finite_gr],
            horizontal_query,
            left=np.nan,
            right=np.nan,
        )
        finite_gr_md = md[finite_gr]
        md_step = float(np.nanmedian(np.diff(md[np.isfinite(md)])))
        horizontal_mask = (
            _nearest_distance(finite_gr_md, horizontal_query)
            <= max(6.0, 2.5 * md_step)
        ).astype(np.float32)
        horizontal_patch = np.nan_to_num(
            horizontal_patch, nan=0.0
        ).astype(np.float32)

        tw = typewell[["TVT", "GR"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        tw = tw.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
        tw_tvt = tw["TVT"].to_numpy(float)
        tw_gr = tw["GR"].to_numpy(float)
        if len(tw_tvt) < 32:
            raise RuntimeError("typewell too short")
        tw_norm = _robust_normalise(tw_gr)
        state_tvt = anchor_tvt + STATE_OFFSETS
        typewell_query = state_tvt[:, None] + PATCH_OFFSETS[None, :]
        typewell_patch = _interp_finite(
            tw_tvt, tw_norm, typewell_query, left=np.nan, right=np.nan
        )
        typewell_mask = (
            (typewell_query >= tw_tvt.min())
            & (typewell_query <= tw_tvt.max())
        ).astype(np.float32)
        typewell_patch = np.nan_to_num(
            typewell_patch, nan=0.0
        ).astype(np.float32)
        state_valid = (typewell_mask.mean(axis=1) >= 0.50).astype(np.uint8)

        progress = (column_md - column_md[0]) / max(
            column_md[-1] - column_md[0], 1.0
        )
        dx = np.gradient(column_x, column_md, edge_order=1)
        dy = np.gradient(column_y, column_md, edge_order=1)
        dz = np.gradient(column_z, column_md, edge_order=1)
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
            raise RuntimeError("too few valid TVT states")
        teacher_index = np.argmin(
            np.abs(valid_offsets[:, None] - truth_delta[None, :]), axis=0
        )
        teacher_delta = valid_offsets[teacher_index]
        teacher_rmse = float(
            np.sqrt(np.mean((teacher_delta - truth_delta) ** 2))
        )
        label_coverage = float(
            np.mean(np.abs(truth_delta) <= np.max(np.abs(STATE_OFFSETS)))
        )
        nearest_index = np.argmin(
            np.abs(STATE_OFFSETS[:, None] - truth_delta[None, :]), axis=0
        )
        true_state_valid_fraction = float(
            np.mean(state_valid[nearest_index])
        )

        np.savez_compressed(
            output,
            horizontal_patch=np.stack(
                [horizontal_patch, horizontal_mask], axis=1
            ).astype(np.float32),
            typewell_patch=np.stack(
                [typewell_patch, typewell_mask], axis=1
            ).astype(np.float32),
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
            teacher_rmse=np.asarray([teacher_rmse], np.float32),
            label_coverage=np.asarray([label_coverage], np.float32),
            true_state_valid_fraction=np.asarray(
                [true_state_valid_fraction], np.float32
            ),
        )
        return {"well": well, "status": "built", "columns": N_COLUMNS}
    except Exception as exc:
        return {"well": well, "status": "failed", "reason": repr(exc)}


class PanelStore:
    def __init__(self, wells):
        self.records = []
        for well in tqdm(wells, desc="load CorrNet panels into RAM"):
            with np.load(PANEL_DIR / f"{well}.npz", allow_pickle=False) as data:
                self.records.append(
                    {
                        "well": well,
                        "horizontal": data["horizontal_patch"].astype(np.float32),
                        "typewell": data["typewell_patch"].astype(np.float32),
                        "geometry": data["geometry"].astype(np.float32),
                        "state_valid": data["state_valid"].astype(bool),
                        "truth_delta": data["truth_delta"].astype(np.float32),
                        "column_z": data["column_z"].astype(np.float32),
                        "column_md": data["column_md"].astype(np.float32),
                        "eval_rows": data["eval_rows"].astype(np.int32),
                        "eval_md": data["eval_md"].astype(np.float32),
                        "eval_truth": data["eval_truth"].astype(np.float32),
                        "anchor_tvt": float(data["anchor_tvt"][0]),
                    }
                )

    def __len__(self):
        return len(self.records)


def _donor_index(index, length):
    if length < 2:
        return index
    shift = max(1, length // 3)
    return (index + shift) % length


def _model_inputs(store, index, mode):
    record = store.records[index]
    horizontal = torch.from_numpy(record["horizontal"])
    typewell = torch.from_numpy(record["typewell"])
    state_valid = torch.from_numpy(record["state_valid"])
    if mode == "no_gr":
        horizontal = horizontal.clone()
        typewell = typewell.clone()
        horizontal[:, 0, :] = 0.0
        typewell[:, 0, :] = 0.0
    elif mode == "shuffled_typewell":
        donor = store.records[_donor_index(index, len(store))]
        typewell = torch.from_numpy(donor["typewell"])
    elif mode != "full":
        raise ValueError(mode)
    return horizontal, typewell, state_valid


class PanelDataset(Dataset):
    def __init__(self, store, mode):
        self.store = store
        self.mode = mode

    def __len__(self):
        return len(self.store)

    def __getitem__(self, index):
        record = self.store.records[index]
        horizontal, typewell, state_valid = _model_inputs(
            self.store, index, self.mode
        )
        return {
            "horizontal": horizontal,
            "typewell": typewell,
            "state_valid": state_valid,
            "geometry": torch.from_numpy(record["geometry"]),
            "truth_delta": torch.from_numpy(record["truth_delta"]),
            "column_z": torch.from_numpy(record["column_z"]),
        }


class PatchEncoder(nn.Module):
    def __init__(self, embedding=EMBED_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 32, 5, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 48, 5, stride=2, padding=2),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv1d(48, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, embedding, 3, padding=1),
            nn.GroupNorm(8, embedding),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, values):
        return self.network(values).squeeze(-1)


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.project = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
        )

    def forward(self, values):
        return F.gelu(self.block(values) + self.project(values))


class CorrNetV2(nn.Module):
    def __init__(self, geometry_channels):
        super().__init__()
        self.patch_encoder = PatchEncoder()
        self.correlation_scale = nn.Parameter(torch.tensor(0.5))
        base = BASE_CHANNELS
        self.enc1 = ConvBlock(2 + geometry_channels, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            ConvBlock(base * 4, base * 4),
            ConvBlock(base * 4, base * 4),
        )
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2 + base, base)
        self.head = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, horizontal, typewell, geometry, state_offsets):
        batch, columns, _, patch = horizontal.shape
        states = typewell.shape[1]
        horizontal_embedding = self.patch_encoder(
            horizontal.reshape(batch * columns, 2, patch)
        )
        typewell_embedding = self.patch_encoder(
            typewell.reshape(batch * states, 2, patch)
        )
        horizontal_embedding = F.normalize(
            horizontal_embedding.reshape(batch, columns, -1), dim=-1
        )
        typewell_embedding = F.normalize(
            typewell_embedding.reshape(batch, states, -1), dim=-1
        )
        correlation = torch.einsum(
            "bhd,bwd->bhw", typewell_embedding, horizontal_embedding
        )
        state_channel = state_offsets[None, :, None].expand(
            batch, states, columns
        )
        geometry_channels = geometry[:, :, None, :].expand(
            batch, geometry.shape[1], states, columns
        )
        values = torch.cat(
            [
                correlation[:, None],
                state_channel[:, None] / 128.0,
                geometry_channels,
            ],
            dim=1,
        )
        enc1 = self.enc1(values)
        enc2 = self.enc2(self.down1(enc1))
        bottleneck = self.bottleneck(self.down2(enc2))
        up2 = F.interpolate(
            bottleneck,
            size=enc2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = F.interpolate(
            dec2,
            size=enc1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        return self.correlation_scale * correlation + self.head(dec1).squeeze(1)


def _loss(logits, truth_delta, column_z, state_valid, state_offsets, ce_only):
    valid_logits = logits.masked_fill(~state_valid[:, :, None], -30.0)
    distance = state_offsets[None, :, None] - truth_delta[:, None, :]
    target = torch.softmax(
        -0.5 * (distance / TARGET_SIGMA_FT) ** 2, dim=1
    )
    target = target * state_valid[:, :, None]
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_probability = F.log_softmax(valid_logits, dim=1)
    column_weight = torch.ones_like(truth_delta)
    column_weight[:, 0] = 4.0
    ce_columns = -(target * log_probability).sum(dim=1)
    ce = (ce_columns * column_weight).sum() / column_weight.sum()

    probability = torch.softmax(valid_logits, dim=1)
    prediction = (
        probability * state_offsets[None, :, None]
    ).sum(dim=1)
    tvt = F.smooth_l1_loss(prediction, truth_delta, beta=2.0)
    predicted_s = prediction + column_z - column_z[:, :1]
    true_s = truth_delta + column_z - column_z[:, :1]
    slope = F.smooth_l1_loss(
        predicted_s[:, 1:] - predicted_s[:, :-1],
        true_s[:, 1:] - true_s[:, :-1],
        beta=0.5,
    )
    curvature = F.smooth_l1_loss(
        predicted_s[:, 2:]
        - 2.0 * predicted_s[:, 1:-1]
        + predicted_s[:, :-2],
        true_s[:, 2:] - 2.0 * true_s[:, 1:-1] + true_s[:, :-2],
        beta=0.25,
    )
    total = ce if ce_only else (
        LOSS_CE * ce
        + LOSS_TVT * tvt
        + LOSS_S_SLOPE * slope
        + LOSS_S_CURVATURE * curvature
    )
    return total, {"ce": ce, "tvt": tvt, "slope": slope, "curvature": curvature}


@torch.no_grad()
def _predict_record(model, store, index, mode, device, state_offsets):
    record = store.records[index]
    horizontal, typewell, state_valid = _model_inputs(store, index, mode)
    horizontal = horizontal[None].to(device)
    typewell = typewell[None].to(device)
    state_valid = state_valid[None].to(device)
    geometry = torch.from_numpy(record["geometry"])[None].to(device)
    logits = model(horizontal, typewell, geometry, state_offsets)
    logits = logits.masked_fill(~state_valid[:, :, None], -30.0)
    probability = torch.softmax(logits, dim=1)
    delta = (
        probability * state_offsets[None, :, None]
    ).sum(dim=1)[0].cpu().numpy()
    delta[0] = 0.0
    prediction = record["anchor_tvt"] + np.interp(
        record["eval_md"], record["column_md"], delta
    )
    return prediction


@torch.no_grad()
def _evaluate(model, store, mode, device, state_offsets, return_rows=False):
    rows = []
    by_well = []
    pooled_errors = []
    for index, record in enumerate(store.records):
        prediction = _predict_record(
            model, store, index, mode, device, state_offsets
        )
        truth = record["eval_truth"].astype(float)
        error = prediction - truth
        pooled_errors.append(error)
        by_well.append(
            {
                "well": record["well"],
                "rows": len(error),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
            }
        )
        if return_rows:
            rows.append(
                pd.DataFrame(
                    {
                        "well": record["well"],
                        "row_idx": record["eval_rows"],
                        "truth": truth,
                        "prediction": prediction,
                    }
                )
            )
    pooled_errors = np.concatenate(pooled_errors)
    by_well = pd.DataFrame(by_well)
    metrics = {
        "rmse": float(np.sqrt(np.mean(pooled_errors**2))),
        "mae": float(np.mean(np.abs(pooled_errors))),
        "median_well_rmse": float(by_well["rmse"].median()),
        "p95_well_rmse": float(by_well["rmse"].quantile(0.95)),
    }
    row_frame = pd.concat(rows, ignore_index=True) if return_rows else None
    return metrics, by_well, row_frame


def _train_arm(arm, train_store, valid_store, geometry_channels, device, state_offsets):
    seed = RANDOM_STATE + ARMS.index(arm) * 1000
    _seed_everything(seed)
    model = CorrNetV2(geometry_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE
    )
    dataset = PanelDataset(train_store, arm)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    checkpoint_path = MODEL_DIR / f"{arm}.pt"
    history = []
    best_rmse = np.inf
    reports_without_improvement = 0
    started = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_rows = []
        ce_only = epoch <= WARMUP_EPOCHS
        for batch in loader:
            horizontal = batch["horizontal"].to(device, non_blocking=True)
            typewell = batch["typewell"].to(device, non_blocking=True)
            state_valid = batch["state_valid"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            truth_delta = batch["truth_delta"].to(device, non_blocking=True)
            column_z = batch["column_z"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(horizontal, typewell, geometry, state_offsets)
            loss, components = _loss(
                logits,
                truth_delta,
                column_z,
                state_valid,
                state_offsets,
                ce_only,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    **{
                        key: float(value.detach().cpu())
                        for key, value in components.items()
                    },
                }
            )
        scheduler.step()

        should_report = (
            epoch == 1
            or epoch % REPORT_EVERY == 0
            or epoch == WARMUP_EPOCHS
            or epoch == WARMUP_EPOCHS + 1
            or epoch == EPOCHS
        )
        if not should_report:
            continue
        model.eval()
        metrics, by_well, _ = _evaluate(
            model, valid_store, arm, device, state_offsets
        )
        record = {
            "arm": arm,
            "epoch": epoch,
            "phase": "ce_only" if ce_only else "joint",
            **{
                f"train_{key}": float(
                    np.mean([row[key] for row in loss_rows])
                )
                for key in loss_rows[0]
            },
            **{f"valid_{key}": value for key, value in metrics.items()},
            "correlation_scale": float(
                model.correlation_scale.detach().cpu()
            ),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(record)
        print("epoch", record, flush=True)

        if metrics["rmse"] < best_rmse - 1e-4:
            best_rmse = metrics["rmse"]
            reports_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "correlation_scale": float(
                        model.correlation_scale.detach().cpu()
                    ),
                    "geometry_channels": geometry_channels,
                    "arm": arm,
                },
                checkpoint_path,
            )
            by_well.to_csv(
                OUT_DIR / f"{arm}_best_bywell.csv", index=False
            )
        elif epoch > WARMUP_EPOCHS:
            reports_without_improvement += 1

        if (
            epoch >= WARMUP_EPOCHS + 6
            and reports_without_improvement >= EARLY_STOPPING_REPORTS
        ):
            print(f"{arm}: early stopping at epoch {epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(
        OUT_DIR / f"{arm}_history.csv", index=False
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics, by_well, rows = _evaluate(
        model, valid_store, arm, device, state_offsets, return_rows=True
    )
    by_well.to_csv(OUT_DIR / f"{arm}_final_bywell.csv", index=False)
    rows.to_pickle(OUT_DIR / f"{arm}_fold1_predictions.pkl")
    result = {
        "arm": arm,
        "best_epoch": int(checkpoint["epoch"]),
        **metrics,
        "correlation_scale": float(checkpoint["correlation_scale"]),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    del model, optimizer, scheduler, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, rows


def _load_model(path, geometry_channels, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CorrNetV2(geometry_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _rmse(truth, prediction):
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(prediction)) ** 2)))


def _cosine(left, right):
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else np.nan


for directory in (OUT_DIR, PANEL_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

fold_path = _first_existing(FOLD_CANDIDATES)
if fold_path is None:
    raise FileNotFoundError("fold metadata not found")
fold = pd.read_csv(fold_path, dtype={"well": str})
fold = fold.groupby("well", as_index=False)["fold"].first()
fold["well"] = fold["well"].astype(str)
fold["fold"] = fold["fold"].astype(int)
wells = fold["well"].tolist()
train_wells = fold.loc[fold["fold"] != VALID_FOLD, "well"].tolist()
valid_wells = fold.loc[fold["fold"] == VALID_FOLD, "well"].tolist()

print(
    {
        "experiment": "tvt_corrnet_onefold_v2",
        "fold_source": str(fold_path),
        "train_wells": len(train_wells),
        "valid_wells": len(valid_wells),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "arms": ARMS,
    }
)

build_started = time.time()
build_report = pd.DataFrame(
    [_build_panel(well) for well in tqdm(wells, desc="build CorrNet OOF panels")]
)
build_report.to_csv(OUT_DIR / "panel_build_report.csv", index=False)
failed = build_report[build_report["status"] == "failed"]
if len(failed):
    raise RuntimeError(f"failed to build {len(failed)} panels")
panel_build_minutes = (time.time() - build_started) / 60.0

guard_rows = []
for well in wells:
    with np.load(PANEL_DIR / f"{well}.npz", allow_pickle=False) as data:
        guard_rows.append(
            {
                "well": well,
                "teacher_rmse": float(data["teacher_rmse"][0]),
                "label_coverage": float(data["label_coverage"][0]),
                "true_state_valid_fraction": float(
                    data["true_state_valid_fraction"][0]
                ),
            }
        )
guard_frame = pd.DataFrame(guard_rows)
guard_frame.to_csv(OUT_DIR / "panel_guards.csv", index=False)
guards = {
    "teacher_rmse_mean": float(guard_frame["teacher_rmse"].mean()),
    "teacher_rmse_max": float(guard_frame["teacher_rmse"].max()),
    "label_coverage_mean": float(guard_frame["label_coverage"].mean()),
    "true_state_valid_mean": float(
        guard_frame["true_state_valid_fraction"].mean()
    ),
}
if guards["label_coverage_mean"] < 0.995 or guards["true_state_valid_mean"] < 0.98:
    raise RuntimeError(f"panel guard failed: {guards}")

train_store = PanelStore(train_wells)
valid_store = PanelStore(valid_wells)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")
with np.load(PANEL_DIR / f"{wells[0]}.npz", allow_pickle=False) as first:
    state_offsets = torch.from_numpy(first["state_offsets"]).to(DEVICE)
    geometry_channels = int(first["geometry"].shape[0])

arm_results = []
arm_predictions = {}
for arm in ARMS:
    print("\nTRAIN ARM", arm, flush=True)
    result, predictions = _train_arm(
        arm,
        train_store,
        valid_store,
        geometry_channels,
        DEVICE,
        state_offsets,
    )
    arm_results.append(result)
    arm_predictions[arm] = predictions
    print("ARM RESULT", result, flush=True)

arm_report = pd.DataFrame(arm_results).sort_values("rmse").reset_index(drop=True)
full_rows = arm_predictions["full"].copy()

# Intervene on the already-trained full model. This is separate from training
# the controls and catches a decoder that simply ignores its matching inputs.
full_model = _load_model(MODEL_DIR / "full.pt", geometry_channels, DEVICE)
intervention_rows = []
for mode in ("full", "no_gr", "shuffled_typewell"):
    metrics, _, predictions = _evaluate(
        full_model,
        valid_store,
        mode,
        DEVICE,
        state_offsets,
        return_rows=True,
    )
    predictions.to_pickle(OUT_DIR / f"full_intervention_{mode}.pkl")
    intervention_rows.append({"mode": mode, **metrics})
intervention_report = pd.DataFrame(intervention_rows)
full_intervention_rmse = float(
    intervention_report.loc[intervention_report["mode"] == "full", "rmse"].iloc[0]
)
intervention_report["degradation_vs_full"] = (
    intervention_report["rmse"] - full_intervention_rmse
)
del full_model
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

# Compare against the exact historical fold-1 rows and scan conservative
# fixed blends. This is triage, not a nested model-selection claim.
reference_report = pd.DataFrame()
blend_report = pd.DataFrame()
pool_path = _first_existing(POOL_CANDIDATES)
if pool_path is not None:
    header = pd.read_csv(pool_path, nrows=0).columns.tolist()
    reference_columns = [
        name for name in ("strong", "strong_bias", "cp12") if name in header
    ]
    usecols = [
        name
        for name in ("id", "well", "row_idx", "tvt_true", *reference_columns)
        if name in header
    ]
    pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": str, "well": str})
    if "well" not in pool.columns:
        split = pool["id"].str.rsplit("_", n=1, expand=True)
        pool["well"] = split[0]
    if "row_idx" not in pool.columns:
        pool["row_idx"] = (
            pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
        )
    pool["well"] = pool["well"].astype(str)
    pool = pool[pool["well"].isin(valid_wells)]
    merged = full_rows.merge(
        pool[["well", "row_idx", "tvt_true", *reference_columns]],
        on=["well", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(full_rows):
        raise RuntimeError(
            f"reference merge retained {len(merged)} / {len(full_rows)} rows"
        )
    reference_rows = []
    blend_rows = []
    model_error = merged["prediction"].to_numpy(float) - merged["truth"].to_numpy(float)
    for base in reference_columns:
        base_prediction = merged[base].to_numpy(float)
        truth = merged["truth"].to_numpy(float)
        base_rmse = _rmse(truth, base_prediction)
        base_error = base_prediction - truth
        reference_rows.append(
            {
                "model": base,
                "rmse": base_rmse,
                "error_cosine_vs_corrnet": _cosine(base_error, model_error),
            }
        )
        for weight in BLEND_WEIGHTS:
            blend = (1.0 - weight) * base_prediction + weight * merged[
                "prediction"
            ].to_numpy(float)
            score = _rmse(truth, blend)
            blend_rows.append(
                {
                    "base": base,
                    "weight_corrnet": weight,
                    "base_rmse": base_rmse,
                    "rmse": score,
                    "gain_vs_base": base_rmse - score,
                }
            )
    reference_report = pd.DataFrame(reference_rows).sort_values("rmse")
    blend_report = pd.DataFrame(blend_rows).sort_values("rmse")

full_rmse = float(arm_report.loc[arm_report["arm"] == "full", "rmse"].iloc[0])
no_gr_rmse = float(arm_report.loc[arm_report["arm"] == "no_gr", "rmse"].iloc[0])
shuffled_rmse = float(
    arm_report.loc[arm_report["arm"] == "shuffled_typewell", "rmse"].iloc[0]
)
intervention_no_gr = float(
    intervention_report.loc[intervention_report["mode"] == "no_gr", "rmse"].iloc[0]
)
intervention_shuffled = float(
    intervention_report.loc[
        intervention_report["mode"] == "shuffled_typewell", "rmse"
    ].iloc[0]
)
best_reference_rmse = (
    float(reference_report["rmse"].min()) if len(reference_report) else np.inf
)
best_blend_gain = (
    float(blend_report["gain_vs_base"].max()) if len(blend_report) else -np.inf
)
predictive_pass = bool(
    full_rmse < best_reference_rmse or best_blend_gain >= 0.10
)
causal_pass = bool(
    no_gr_rmse - full_rmse >= 0.10
    and shuffled_rmse - full_rmse >= 0.10
    and intervention_no_gr - full_rmse >= 0.25
    and intervention_shuffled - full_rmse >= 0.15
)
strict_pass = bool(predictive_pass and causal_pass)

verdict = {
    "experiment": "tvt_corrnet_onefold_v2",
    "fold_source": str(fold_path),
    "train_wells": len(train_wells),
    "valid_wells": len(valid_wells),
    "guards": guards,
    "panel_build_minutes": panel_build_minutes,
    "full_rmse": full_rmse,
    "best_reference_rmse": best_reference_rmse,
    "best_blend_gain": best_blend_gain,
    "trained_no_gr_degradation": no_gr_rmse - full_rmse,
    "trained_shuffled_typewell_degradation": shuffled_rmse - full_rmse,
    "intervention_no_gr_degradation": intervention_no_gr - full_rmse,
    "intervention_shuffled_typewell_degradation": intervention_shuffled - full_rmse,
    "predictive_pass": predictive_pass,
    "causal_pass": causal_pass,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> train five fold CorrNet and test OOF fusion"
        if strict_pass
        else "FAIL -> stop CorrNet OOF route; keep the capacity result as diagnostic only"
    ),
}

arm_report.to_csv(OUT_DIR / "arm_scores.csv", index=False)
intervention_report.to_csv(OUT_DIR / "intervention_scores.csv", index=False)
reference_report.to_csv(OUT_DIR / "reference_scores.csv", index=False)
blend_report.to_csv(OUT_DIR / "blend_scan.csv", index=False)
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2), encoding="utf-8"
)

print("\nARM SCORES")
print(arm_report.to_string(index=False))
print("\nFULL-MODEL INTERVENTIONS")
print(intervention_report.to_string(index=False))
print("\nHISTORICAL REFERENCES")
print(reference_report.to_string(index=False) if len(reference_report) else "<none>")
print("\nBLEND SCAN TOP 20")
print(blend_report.head(20).to_string(index=False) if len(blend_report) else "<none>")
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
