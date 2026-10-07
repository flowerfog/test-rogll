"""Faithful fold-1 standalone TVT-CorrNet with multi-pseudo-PS augmentation.

Each outer-train well contributes its official prediction start plus four
deterministic pseudo starts.  Outer-valid wells are evaluated only at their
untouched official prediction start.  The model consumes no strong/CP12/Gold/
Neighbor prediction; it is an independent single-well model.

Three seeds are trained.  Negative controls are interventions on the best
full model, not three additional training runs.  The five-fold promotion gate
is standalone pooled RMSE < 8 on fold 1 with causal GR/typewell degradation.
"""

from pathlib import Path
import gc
import hashlib
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


DATA_ROOT_CANDIDATES = (
    Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
    Path.cwd(),
)
DATA_ROOT = next(
    (path for path in DATA_ROOT_CANDIDATES if (path / "train").exists()),
    DATA_ROOT_CANDIDATES[0],
)
OOF_ROOT = (
    Path("/root/autodl-tmp/oof_exports")
    if Path("/root/autodl-tmp/oof_exports").exists()
    else DATA_ROOT / "oof_exports"
)
OUT_DIR = OOF_ROOT / "tvt_corrnet_multips_onefold_v3"
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
TRAIN_TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")
SEEDS = (20260717, 20260771, 20261707)
N_COLUMNS = 256
STATE_OFFSETS = np.arange(-128.0, 128.01, 2.0, dtype=np.float32)
PATCH_OFFSETS = np.arange(-64.0, 64.01, 2.0, dtype=np.float32)
EMBED_DIM = 64
BASE_CHANNELS = 48
BATCH_SIZE = 4
EPOCHS = 40
WARMUP_EPOCHS = 8
REPORT_EVERY = 2
EARLY_STOPPING_REPORTS = 6
LEARNING_RATE = 5e-4
MIN_LEARNING_RATE = 5e-5
RANDOM_STATE = 20260717
TARGET_SIGMA_FT = 2.0

LOSS_CE = 1.0
LOSS_TVT = 0.30
LOSS_S_SLOPE = 0.05
LOSS_S_CURVATURE = 0.02

CUT_MANIFEST_CANDIDATES = (
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "cut_manifest.csv",
)
EXTRA_CUT_MANIFEST_CANDIDATES = (
    OOF_ROOT / "multicut_fleongg_matched_k5_extension_v1" / "new_cut_manifest.csv",
)


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


def _panel_key(well, tag):
    return f"{well}__{tag}"


def _build_panel(task):
    well = str(task["well"])
    tag = str(task["tag"])
    cut_position = int(task["cut_position"])
    official_n_eval = int(task["official_n_eval"])
    state_mass = float(task["state_mass"])
    key = _panel_key(well, tag)
    output = PANEL_DIR / f"{key}.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as cached:
            if (
                "true_state_valid_fraction" in cached.files
                and "tag" in cached.files
                and str(cached["tag"][0]) == tag
                and int(cached["cut_position"][0]) == cut_position
            ):
                return {
                    "well": well,
                    "tag": tag,
                    "key": key,
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
        official_eval = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
        if len(known) < 30 or len(official_eval) < 128:
            raise RuntimeError("too few prefix/eval rows")
        official_cut = int(official_eval[0])
        if tag == "official":
            anchor = int(known[-1])
            eval_rows = official_eval
            if cut_position != official_cut:
                raise RuntimeError(
                    f"official cut mismatch {cut_position} != {official_cut}"
                )
            anchor_tvt = float(tvt_input[anchor])
        else:
            anchor = int(cut_position - 1)
            last_truth = int(np.flatnonzero(np.isfinite(truth))[-1])
            eval_rows = np.arange(cut_position, last_truth + 1, dtype=np.int32)
            anchor_tvt = float(truth[anchor])
        if anchor < 29 or len(eval_rows) < 128 or not np.isfinite(anchor_tvt):
            raise RuntimeError("invalid pseudo prediction start")

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
        typewell_step = float(np.median(np.diff(tw_tvt))) if len(tw_tvt) > 1 else 1.0
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
                np.full(N_COLUMNS, (anchor + 1) / len(md)),
                np.full(N_COLUMNS, np.mean(~np.isfinite(gr))),
                np.full(N_COLUMNS, typewell_step),
                np.full(N_COLUMNS, (tw_tvt[-1] - tw_tvt[0]) / 1000.0),
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
            eval_z=z[eval_rows].astype(np.float32),
            eval_truth=truth[eval_rows].astype(np.float32),
            anchor_tvt=np.asarray([anchor_tvt], np.float32),
            tag=np.asarray([tag], str),
            cut_position=np.asarray([cut_position], np.int32),
            panel_weight=np.asarray(
                [state_mass * max(official_n_eval, 1)], np.float32
            ),
            teacher_rmse=np.asarray([teacher_rmse], np.float32),
            label_coverage=np.asarray([label_coverage], np.float32),
            true_state_valid_fraction=np.asarray(
                [true_state_valid_fraction], np.float32
            ),
        )
        return {
            "well": well,
            "tag": tag,
            "key": key,
            "status": "built",
            "columns": N_COLUMNS,
        }
    except Exception as exc:
        return {
            "well": well,
            "tag": tag,
            "key": key,
            "status": "failed",
            "reason": repr(exc),
        }


class PanelStore:
    def __init__(self, keys):
        self.records = []
        for key in tqdm(keys, desc="load CorrNet panels into RAM"):
            with np.load(PANEL_DIR / f"{key}.npz", allow_pickle=False) as data:
                self.records.append(
                    {
                        "well": str(key).rsplit("__", 1)[0],
                        "tag": str(data["tag"][0]),
                        "horizontal": data["horizontal_patch"].astype(np.float32),
                        "typewell": data["typewell_patch"].astype(np.float32),
                        "geometry": data["geometry"].astype(np.float32),
                        "state_valid": data["state_valid"].astype(bool),
                        "truth_delta": data["truth_delta"].astype(np.float32),
                        "column_z": data["column_z"].astype(np.float32),
                        "column_md": data["column_md"].astype(np.float32),
                        "eval_rows": data["eval_rows"].astype(np.int32),
                        "eval_md": data["eval_md"].astype(np.float32),
                        "eval_z": data["eval_z"].astype(np.float32),
                        "eval_truth": data["eval_truth"].astype(np.float32),
                        "anchor_tvt": float(data["anchor_tvt"][0]),
                        "panel_weight": float(data["panel_weight"][0]),
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
            "panel_weight": torch.tensor(record["panel_weight"], dtype=torch.float32),
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


def _loss(
    logits,
    truth_delta,
    column_z,
    state_valid,
    state_offsets,
    panel_weight,
    ce_only,
):
    panel_weight = panel_weight.float().clamp_min(1e-6)

    def weighted_mean(values):
        return (values * panel_weight).sum() / panel_weight.sum()

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
    ce = weighted_mean(
        (ce_columns * column_weight).sum(dim=1) / column_weight.sum(dim=1)
    )

    probability = torch.softmax(valid_logits, dim=1)
    prediction = (
        probability * state_offsets[None, :, None]
    ).sum(dim=1)
    tvt = weighted_mean(
        F.smooth_l1_loss(
            prediction, truth_delta, beta=2.0, reduction="none"
        ).mean(dim=1)
    )
    predicted_s = prediction + column_z - column_z[:, :1]
    true_s = truth_delta + column_z - column_z[:, :1]
    slope = weighted_mean(
        F.smooth_l1_loss(
            predicted_s[:, 1:] - predicted_s[:, :-1],
            true_s[:, 1:] - true_s[:, :-1],
            beta=0.5,
            reduction="none",
        ).mean(dim=1)
    )
    curvature = weighted_mean(
        F.smooth_l1_loss(
            predicted_s[:, 2:]
            - 2.0 * predicted_s[:, 1:-1]
            + predicted_s[:, :-2],
            true_s[:, 2:] - 2.0 * true_s[:, 1:-1] + true_s[:, :-2],
            beta=0.25,
            reduction="none",
        ).mean(dim=1)
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
    column_s = record["anchor_tvt"] + delta + record["column_z"]
    eval_s = np.interp(record["eval_md"], record["column_md"], column_s)
    prediction = eval_s - record["eval_z"]
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


def _train_seed(seed, train_store, valid_store, geometry_channels, device, state_offsets):
    arm = "full"
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
    checkpoint_path = MODEL_DIR / f"seed_{seed}.pt"
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
            panel_weight = batch["panel_weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(horizontal, typewell, geometry, state_offsets)
            loss, components = _loss(
                logits,
                truth_delta,
                column_z,
                state_valid,
                state_offsets,
                panel_weight,
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
            "seed": seed,
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
                    "seed": seed,
                },
                checkpoint_path,
            )
            by_well.to_csv(
                OUT_DIR / f"seed_{seed}_best_bywell.csv", index=False
            )
        elif epoch > WARMUP_EPOCHS:
            reports_without_improvement += 1

        if (
            epoch >= WARMUP_EPOCHS + 6
            and reports_without_improvement >= EARLY_STOPPING_REPORTS
        ):
            print(f"seed {seed}: early stopping at epoch {epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(
        OUT_DIR / f"seed_{seed}_history.csv", index=False
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics, by_well, rows = _evaluate(
        model, valid_store, arm, device, state_offsets, return_rows=True
    )
    by_well.to_csv(OUT_DIR / f"seed_{seed}_final_bywell.csv", index=False)
    rows.to_pickle(OUT_DIR / f"seed_{seed}_fold1_predictions.pkl")
    result = {
        "arm": arm,
        "seed": seed,
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


def _stable_unit_interval(well, slot):
    digest = hashlib.blake2b(
        f"{well}:{slot}".encode("ascii"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") / float(2**64)


def _distinct_cut(base_cut, used, n_rows, unit):
    min_gap = max(32, int(round(0.02 * n_rows)))
    signs = (1, -1) if unit >= 0.5 else (-1, 1)
    offsets = [0]
    for multiplier in range(1, 7):
        offsets.extend(
            [signs[0] * multiplier * min_gap, signs[1] * multiplier * min_gap]
        )
    for offset in offsets:
        candidate = int(np.clip(base_cut + offset, 64, n_rows - 128))
        if all(abs(candidate - previous) >= min_gap for previous in used):
            return candidate
    grid = np.arange(64, max(65, n_rows - 127), max(1, min_gap // 2), dtype=int)
    distance = np.asarray(
        [min(abs(point - previous) for previous in used) for point in grid]
    )
    return int(grid[int(np.argmax(distance))])


def _build_cut_manifest(wells):
    metadata = []
    for well in tqdm(wells, desc="scan CorrNet cut metadata"):
        values = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
            usecols=["TVT_input", "TVT"],
        )
        tvt_input = pd.to_numeric(values["TVT_input"], errors="coerce").to_numpy(float)
        truth = pd.to_numeric(values["TVT"], errors="coerce").to_numpy(float)
        hidden = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
        known = np.flatnonzero(np.isfinite(tvt_input))
        if len(hidden) < 128 or len(known) < 64:
            raise RuntimeError(f"invalid official mask for {well}")
        official_cut = int(hidden[0])
        metadata.append(
            {
                "well": well,
                "n_rows": len(values),
                "official_cut": official_cut,
                "official_n_eval": len(hidden),
                "visible_frac": official_cut / len(values),
            }
        )
    metadata = pd.DataFrame(metadata)
    reference_fracs = metadata["visible_frac"].to_numpy(float)
    rows = []
    for record in metadata.itertuples(index=False):
        used = {int(record.official_cut)}
        rows.append(
            {
                "well": str(record.well),
                "tag": "official",
                "cut_position": int(record.official_cut),
                "official_n_eval": int(record.official_n_eval),
                "actual_frac": float(record.visible_frac),
            }
        )
        for slot, tag in enumerate(TRAIN_TAGS[1:]):
            unit = _stable_unit_interval(str(record.well), slot)
            half = slot % 2
            quantile = (half + unit) / 2.0
            requested_frac = float(np.quantile(reference_fracs, quantile))
            base_cut = int(
                np.clip(
                    round(requested_frac * record.n_rows),
                    64,
                    record.n_rows - 128,
                )
            )
            cut = _distinct_cut(base_cut, used, int(record.n_rows), unit)
            used.add(cut)
            rows.append(
                {
                    "well": str(record.well),
                    "tag": tag,
                    "cut_position": cut,
                    "official_n_eval": int(record.official_n_eval),
                    "actual_frac": cut / int(record.n_rows),
                }
            )
    manifest = pd.DataFrame(rows)
    state_mass = {
        "official": 0.40,
        "pseudo1": 0.15,
        "pseudo2": 0.15,
        "pseudo3": 0.15,
        "pseudo4": 0.15,
    }
    manifest["state_mass"] = manifest["tag"].map(state_mass).astype(float)
    return manifest


for directory in (OUT_DIR, PANEL_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

fold_path_v3 = _first_existing(FOLD_CANDIDATES)
if fold_path_v3 is None:
    raise FileNotFoundError("fold metadata not found")
fold_v3 = pd.read_csv(fold_path_v3, dtype={"well": str})
fold_v3 = fold_v3.groupby("well", as_index=False)["fold"].first()
fold_v3["well"] = fold_v3["well"].astype(str)
fold_v3["fold"] = fold_v3["fold"].astype(int)
wells_v3 = fold_v3["well"].tolist()
train_wells_v3 = fold_v3.loc[fold_v3["fold"] != VALID_FOLD, "well"].tolist()
valid_wells_v3 = fold_v3.loc[fold_v3["fold"] == VALID_FOLD, "well"].tolist()

manifest = _build_cut_manifest(wells_v3)
manifest.to_csv(OUT_DIR / "cut_manifest.csv", index=False)
print("CUT DISTRIBUTION")
print(
    manifest.groupby("tag")["actual_frac"].describe()[
        ["count", "mean", "min", "25%", "50%", "75%", "max"]
    ].to_string()
)

train_set = set(train_wells_v3)
valid_set = set(valid_wells_v3)
build_tasks = manifest[
    manifest["well"].isin(train_set)
    | (manifest["well"].isin(valid_set) & (manifest["tag"] == "official"))
].to_dict("records")
train_tasks = manifest[
    manifest["well"].isin(train_set) & manifest["tag"].isin(TRAIN_TAGS)
].copy()
valid_tasks = manifest[
    manifest["well"].isin(valid_set) & (manifest["tag"] == "official")
].copy()

print(
    {
        "experiment": "tvt_corrnet_multips_onefold_v3",
        "fold_source": str(fold_path_v3),
        "train_wells": len(train_wells_v3),
        "train_panels": len(train_tasks),
        "valid_wells": len(valid_wells_v3),
        "valid_panels": len(valid_tasks),
        "seeds": SEEDS,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "standalone": True,
    },
    flush=True,
)

build_started_v3 = time.time()
build_report_v3 = pd.DataFrame(
    [
        _build_panel(task)
        for task in tqdm(build_tasks, desc="build multi-PS CorrNet panels")
    ]
)
build_report_v3.to_csv(OUT_DIR / "panel_build_report.csv", index=False)
failed_v3 = build_report_v3[build_report_v3["status"] == "failed"]
if len(failed_v3):
    print(failed_v3.head(20).to_string(index=False))
    raise RuntimeError(f"failed to build {len(failed_v3)} panels")
panel_build_minutes_v3 = (time.time() - build_started_v3) / 60.0

guard_rows_v3 = []
for task in build_tasks:
    key = _panel_key(task["well"], task["tag"])
    with np.load(PANEL_DIR / f"{key}.npz", allow_pickle=False) as data:
        guard_rows_v3.append(
            {
                "well": str(task["well"]),
                "tag": str(task["tag"]),
                "teacher_rmse": float(data["teacher_rmse"][0]),
                "label_coverage": float(data["label_coverage"][0]),
                "true_state_valid_fraction": float(
                    data["true_state_valid_fraction"][0]
                ),
            }
        )
guard_frame_v3 = pd.DataFrame(guard_rows_v3)
guard_frame_v3.to_csv(OUT_DIR / "panel_guards.csv", index=False)
guards_v3 = {
    "teacher_rmse_mean": float(guard_frame_v3["teacher_rmse"].mean()),
    "teacher_rmse_max": float(guard_frame_v3["teacher_rmse"].max()),
    "label_coverage_mean": float(guard_frame_v3["label_coverage"].mean()),
    "label_coverage_min": float(guard_frame_v3["label_coverage"].min()),
    "true_state_valid_mean": float(
        guard_frame_v3["true_state_valid_fraction"].mean()
    ),
}

# A pseudo start can occasionally land before the lateral section, making its
# target impossible to represent on the deliberately deployment-sized +/-128
# ft state axis. Such augmentation is invalid training data, not evidence that
# the official axis should be widened. Official panels remain strict; pseudo
# panels are retained only when the target is representable and quantised well.
guard_frame_v3["key"] = guard_frame_v3.apply(
    lambda row: _panel_key(str(row["well"]), str(row["tag"])), axis=1
)
guard_frame_v3["usable"] = (
    (guard_frame_v3["label_coverage"] >= 0.98)
    & (guard_frame_v3["true_state_valid_fraction"] >= 0.98)
    & (guard_frame_v3["teacher_rmse"] <= 1.50)
)
guard_frame_v3.to_csv(OUT_DIR / "panel_guards.csv", index=False)
official_bad_v3 = guard_frame_v3[
    (guard_frame_v3["tag"] == "official") & ~guard_frame_v3["usable"]
]
if len(official_bad_v3):
    print("BAD OFFICIAL PANELS")
    print(official_bad_v3.sort_values("teacher_rmse", ascending=False).head(20))
    raise RuntimeError(
        f"{len(official_bad_v3)} official panels do not fit the state axis"
    )

train_tasks["key"] = train_tasks.apply(
    lambda row: _panel_key(str(row["well"]), str(row["tag"])), axis=1
)
valid_tasks["key"] = valid_tasks.apply(
    lambda row: _panel_key(str(row["well"]), str(row["tag"])), axis=1
)
usable_keys_v3 = set(guard_frame_v3.loc[guard_frame_v3["usable"], "key"])
train_panels_before_v3 = len(train_tasks)
train_tasks = train_tasks[train_tasks["key"].isin(usable_keys_v3)].copy()
dropped_train_panels_v3 = train_panels_before_v3 - len(train_tasks)
dropped_fraction_v3 = dropped_train_panels_v3 / max(train_panels_before_v3, 1)
if dropped_fraction_v3 > 0.10:
    raise RuntimeError(
        f"pseudo panel rejection too high: {dropped_train_panels_v3}/"
        f"{train_panels_before_v3} ({dropped_fraction_v3:.1%})"
    )

guard_summary_v3 = (
    guard_frame_v3.groupby("tag", as_index=False)
    .agg(
        panels=("well", "size"),
        usable_panels=("usable", "sum"),
        coverage_mean=("label_coverage", "mean"),
        coverage_min=("label_coverage", "min"),
        teacher_rmse_mean=("teacher_rmse", "mean"),
        teacher_rmse_max=("teacher_rmse", "max"),
    )
)
guard_summary_v3.to_csv(OUT_DIR / "panel_guard_by_tag.csv", index=False)
print("\nPANEL GUARD BY TAG")
print(guard_summary_v3.to_string(index=False))
print(
    "retained training panels:",
    {
        "before": train_panels_before_v3,
        "after": len(train_tasks),
        "dropped": dropped_train_panels_v3,
        "dropped_fraction": dropped_fraction_v3,
    },
)

train_keys_v3 = [
    str(row.key) for row in train_tasks.itertuples(index=False)
]
valid_keys_v3 = [
    str(row.key) for row in valid_tasks.itertuples(index=False)
]
train_store_v3 = PanelStore(train_keys_v3)
valid_store_v3 = PanelStore(valid_keys_v3)

# Renormalise retained state masses to one per well after rejecting invalid
# pseudo panels, preserving row-pooled well weighting.
retained_mass_v3 = train_tasks.groupby("well")["state_mass"].sum().to_dict()
for record_v3 in train_store_v3.records:
    record_v3["panel_weight"] /= max(
        float(retained_mass_v3[record_v3["well"]]), 1e-6
    )
DEVICE_V3 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE_V3.type == "cuda":
    torch.set_float32_matmul_precision("high")
with np.load(PANEL_DIR / f"{train_keys_v3[0]}.npz", allow_pickle=False) as first_v3:
    state_offsets_v3 = torch.from_numpy(first_v3["state_offsets"]).to(DEVICE_V3)
    geometry_channels_v3 = int(first_v3["geometry"].shape[0])

seed_results_v3 = []
seed_predictions_v3 = []
for seed_v3 in SEEDS:
    print("\nTRAIN SEED", seed_v3, flush=True)
    result_v3, prediction_v3 = _train_seed(
        seed_v3,
        train_store_v3,
        valid_store_v3,
        geometry_channels_v3,
        DEVICE_V3,
        state_offsets_v3,
    )
    seed_results_v3.append(result_v3)
    seed_predictions_v3.append(
        prediction_v3.rename(columns={"prediction": f"prediction_{seed_v3}"})
    )
    print("SEED RESULT", result_v3, flush=True)

seed_report_v3 = pd.DataFrame(seed_results_v3).sort_values("rmse")
seed_report_v3.to_csv(OUT_DIR / "seed_results.csv", index=False)
ensemble_v3 = seed_predictions_v3[0]
for frame_v3 in seed_predictions_v3[1:]:
    ensemble_v3 = ensemble_v3.merge(
        frame_v3.drop(columns=["truth"]),
        on=["well", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
prediction_columns_v3 = [f"prediction_{seed}" for seed in SEEDS]
ensemble_v3["prediction"] = ensemble_v3[prediction_columns_v3].mean(axis=1)
ensemble_rmse_v3 = _rmse(ensemble_v3["truth"], ensemble_v3["prediction"])
ensemble_mae_v3 = float(
    np.mean(np.abs(ensemble_v3["truth"] - ensemble_v3["prediction"]))
)
ensemble_v3.to_pickle(OUT_DIR / "ensemble_fold1_predictions.pkl")

best_seed_v3 = int(seed_report_v3.iloc[0]["seed"])
best_model_v3 = _load_model(
    MODEL_DIR / f"seed_{best_seed_v3}.pt", geometry_channels_v3, DEVICE_V3
)
intervention_rows_v3 = []
for mode_v3 in ("full", "no_gr", "shuffled_typewell"):
    metrics_v3, _, rows_v3 = _evaluate(
        best_model_v3,
        valid_store_v3,
        mode_v3,
        DEVICE_V3,
        state_offsets_v3,
        return_rows=True,
    )
    rows_v3.to_pickle(OUT_DIR / f"best_seed_intervention_{mode_v3}.pkl")
    intervention_rows_v3.append({"mode": mode_v3, **metrics_v3})
intervention_report_v3 = pd.DataFrame(intervention_rows_v3)
intervention_report_v3["degradation_vs_full"] = (
    intervention_report_v3["rmse"]
    - float(intervention_report_v3.loc[intervention_report_v3["mode"] == "full", "rmse"].iloc[0])
)
intervention_report_v3.to_csv(OUT_DIR / "interventions.csv", index=False)

no_gr_degradation_v3 = float(
    intervention_report_v3.loc[
        intervention_report_v3["mode"] == "no_gr", "degradation_vs_full"
    ].iloc[0]
)
shuffle_degradation_v3 = float(
    intervention_report_v3.loc[
        intervention_report_v3["mode"] == "shuffled_typewell",
        "degradation_vs_full",
    ].iloc[0]
)
predictive_pass_v3 = bool(
    ensemble_rmse_v3 < 8.0 and float(seed_report_v3["rmse"].max()) < 8.5
)
causal_pass_v3 = bool(
    no_gr_degradation_v3 >= 0.30 and shuffle_degradation_v3 >= 0.30
)
strict_pass_v3 = bool(predictive_pass_v3 and causal_pass_v3)
verdict_v3 = {
    "experiment": "tvt_corrnet_multips_onefold_v3",
    "standalone": True,
    "fold": VALID_FOLD,
    "fold_source": str(fold_path_v3),
    "train_wells": len(train_wells_v3),
    "train_panels": len(train_tasks),
    "dropped_invalid_pseudo_panels": dropped_train_panels_v3,
    "dropped_invalid_pseudo_fraction": dropped_fraction_v3,
    "valid_wells": len(valid_wells_v3),
    "valid_panels": len(valid_tasks),
    "seeds": list(SEEDS),
    "seed_rmse": seed_report_v3[["seed", "rmse"]].to_dict("records"),
    "ensemble_rmse": ensemble_rmse_v3,
    "ensemble_mae": ensemble_mae_v3,
    "best_seed": best_seed_v3,
    "no_gr_degradation": no_gr_degradation_v3,
    "shuffled_typewell_degradation": shuffle_degradation_v3,
    "guards": guards_v3,
    "panel_build_minutes": panel_build_minutes_v3,
    "predictive_pass": predictive_pass_v3,
    "causal_pass": causal_pass_v3,
    "strict_pass": strict_pass_v3,
    "next": (
        "PASS -> run five folds x three seeds; target standalone 5.4-5.8"
        if strict_pass_v3
        else "FAIL -> do not scale this CorrNet implementation to five folds"
    ),
    "elapsed_minutes": (time.time() - build_started_v3) / 60.0,
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle_v3:
    json.dump(verdict_v3, handle_v3, indent=2)

print("\nSEED RESULTS")
print(seed_report_v3.to_string(index=False))
print("\nENSEMBLE", {"rmse": ensemble_rmse_v3, "mae": ensemble_mae_v3})
print("\nINTERVENTIONS")
print(intervention_report_v3.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict_v3, indent=2))
print("saved:", OUT_DIR)

# Preserved only as inert source text to keep this mechanical v2-derived cell
# auditable. It is never executed by v3.
_LEGACY_V2_EXECUTION = r'''
for directory in (OUT_DIR, PANEL_DIR, MODEL_DIR):

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
'''
