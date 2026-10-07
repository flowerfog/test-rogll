"""Complete-well global TVT alignment U-Net, fold-1 standalone gateway.

One training sample is one complete well.  The horizontal axis is the entire
horizontal-well MD sequence.  The vertical axis is an absolute TVT grid around
the prediction-start anchor; it is not centred on PF or any learned baseline.

No neighbor well, strong, CP12, Gold, or Neighbor prediction is consumed.
Pseudo prediction starts only change the visible-path condition between
epochs.  Outer-valid wells always use their untouched official start.
"""

from pathlib import Path
import copy
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


ROOT = Path("/root/autodl-tmp")
DATA_ROOT_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    Path.cwd(),
)
DATA_ROOT = next(
    (path for path in DATA_ROOT_CANDIDATES if (path / "train").exists()),
    DATA_ROOT_CANDIDATES[0],
)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").exists() else DATA_ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "fullwell_global_tvt_unet_onefold_v0"
MODEL_DIR = OUT_DIR / "models"

FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
BANK_DIR = OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1"
EXTENSION_DIR = OOF_ROOT / "multicut_fleongg_matched_k5_extension_v1"
MANIFEST_CANDIDATES = (
    BANK_DIR / "cut_manifest.csv",
    BANK_DIR / "cut_manifest_v1.csv",
)
EXTRA_MANIFEST_CANDIDATES = (
    EXTENSION_DIR / "new_cut_manifest.csv",
    EXTENSION_DIR / "cut_manifest.csv",
)

VALID_FOLD = 1
TRAIN_TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")
N_COLUMNS = 768
STATE_OFFSETS = np.arange(-128.0, 128.01, 1.0, dtype=np.float32)
TARGET_SIGMA_FT = 1.5
BASE_CHANNELS = 16
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
EPOCHS = 24
WARMUP_EPOCHS = 2
EARLY_STOPPING_REPORTS = 6
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
SEED = 20260719

LOSS_CE = 0.35
LOSS_MSE = 1.0 / 64.0
LOSS_DIP = 0.02
LOSS_CURVATURE = 0.005
KNOWN_COLUMN_WEIGHT = 0.20

RAW_CHANNEL_NAMES = (
    "horizontal_gr",
    "horizontal_valid",
    "typewell_gr",
    "typewell_valid",
    "horizontal_x_typewell",
    "negative_abs_typewell_difference",
    "heel_reference_gr",
    "heel_reference_valid",
    "horizontal_x_heel",
    "negative_abs_heel_difference",
    "known_tvt_heatmap",
    "known_path_valid",
    "state_offset",
    "absolute_tvt_position",
    "typewell_edge_margin",
)
GR_VALUE_CHANNELS = (0, 2, 4, 5, 6, 8, 9)
TYPEWELL_CHANNELS = (2, 3, 4, 5)
EMISSION_CHANNEL = 5
KNOWN_HEAT_CHANNEL = 10


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


def _rmse(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _robust_normalise(values):
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    output = np.zeros(values.shape, np.float32)
    if int(valid.sum()) < 4:
        return output
    centre = float(np.median(values[valid]))
    scale = float(1.4826 * np.median(np.abs(values[valid] - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values[valid]), 1.0))
    output[valid] = np.clip((values[valid] - centre) / scale, -8.0, 8.0)
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
    unique, inverse = np.unique(xv, return_inverse=True)
    if len(unique) != len(xv):
        yv = np.bincount(inverse, weights=yv) / np.maximum(np.bincount(inverse), 1)
        xv = unique
    return np.interp(query, xv, yv, left=left, right=right)


def _nearest_distance(x, query):
    x = np.sort(np.asarray(x, np.float64)[np.isfinite(x)])
    query = np.asarray(query, np.float64)
    if not len(x):
        return np.full(query.shape, np.inf)
    position = np.searchsorted(x, query)
    left = np.abs(query - x[np.clip(position - 1, 0, len(x) - 1)])
    right = np.abs(query - x[np.clip(position, 0, len(x) - 1)])
    return np.minimum(left, right)


def _stable_slot(well):
    digest = hashlib.blake2b(str(well).encode("ascii"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def _load_manifest(wells):
    primary = _first_existing(MANIFEST_CANDIDATES)
    extra = _first_existing(EXTRA_MANIFEST_CANDIDATES)
    if primary is None or extra is None:
        raise FileNotFoundError("matched K5 cut manifests are required")
    frames = [pd.read_csv(primary, dtype={"well": str})]
    if extra != primary:
        frames.append(pd.read_csv(extra, dtype={"well": str}))
    manifest = pd.concat(frames, ignore_index=True)
    manifest["well"] = manifest["well"].astype(str)
    manifest["tag"] = manifest["tag"].astype(str)
    manifest = manifest[
        manifest["well"].isin(set(wells)) & manifest["tag"].isin(TRAIN_TAGS)
    ]
    manifest = manifest.drop_duplicates(["well", "tag"], keep="last")
    metadata = []
    for well in tqdm(wells, desc="scan global-TV T cut metadata"):
        frame = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
            usecols=["TVT_input", "TVT"],
        )
        tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(float)
        truth = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        hidden = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
        if len(known) < 64 or len(hidden) < 128:
            raise RuntimeError(f"invalid official mask for {well}")
        metadata.append(
            {
                "well": well,
                "n_rows": len(frame),
                "official_cut": int(hidden[0]),
                "official_n_eval": len(hidden),
            }
        )
    manifest = manifest.drop(
        columns=[
            column
            for column in ("n_rows", "official_cut", "official_n_eval")
            if column in manifest.columns
        ]
    )
    manifest = manifest.merge(
        pd.DataFrame(metadata), on="well", how="left", validate="many_to_one"
    )
    manifest["cut_position"] = manifest["cut_position"].astype(int)
    official = manifest["tag"] == "official"
    manifest.loc[official, "cut_position"] = manifest.loc[official, "official_cut"]
    manifest["actual_frac"] = manifest["cut_position"] / manifest["n_rows"]
    counts = manifest.groupby("well")["tag"].nunique()
    incomplete = sorted(set(wells) - set(counts[counts == len(TRAIN_TAGS)].index))
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} wells miss K5 cuts: {incomplete[:20]}")
    return manifest.sort_values(["well", "tag"]).reset_index(drop=True)


class CompleteWellStore:
    def __init__(self, wells, manifest):
        self.records = []
        cuts = {
            (str(row.well), str(row.tag)): int(row.cut_position)
            for row in manifest.itertuples(index=False)
        }
        official_rows = manifest[manifest["tag"] == "official"].set_index("well")
        for well in tqdm(wells, desc="load complete-well tensors"):
            horizontal = pd.read_csv(
                DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
            )
            typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
            required = {"MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"}
            if not required.issubset(horizontal.columns):
                raise RuntimeError(f"{well}: horizontal schema mismatch")
            md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
            x = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(float)
            z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
            gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
            truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(float)
            tvt_input = pd.to_numeric(
                horizontal["TVT_input"], errors="coerce"
            ).to_numpy(float)
            column_md = np.linspace(md[0], md[-1], N_COLUMNS)
            column_x = _interp_finite(md, x, column_md)
            column_y = _interp_finite(md, y, column_md)
            column_z = _interp_finite(md, z, column_md)
            column_truth = _interp_finite(md, truth, column_md)
            gr = _robust_normalise(gr_raw)
            finite_gr = np.isfinite(gr_raw)
            column_gr = _interp_finite(
                md[finite_gr], gr[finite_gr], column_md, left=np.nan, right=np.nan
            )
            md_step = float(np.nanmedian(np.diff(md[np.isfinite(md)])))
            column_gr_valid = (
                _nearest_distance(md[finite_gr], column_md)
                <= max(6.0, 2.5 * md_step)
            ).astype(np.float32)
            column_gr = np.nan_to_num(column_gr, nan=0.0).astype(np.float32)
            dx = np.gradient(column_x, column_md, edge_order=1)
            dy = np.gradient(column_y, column_md, edge_order=1)
            dz = np.gradient(column_z, column_md, edge_order=1)
            static_geometry = np.stack(
                [
                    (column_md - column_md[0]) / max(column_md[-1] - column_md[0], 1.0),
                    (column_x - column_x[0]) / 1000.0,
                    (column_y - column_y[0]) / 1000.0,
                    (column_z - column_z[0]) / 1000.0,
                    dx,
                    dy,
                    dz,
                    column_gr_valid,
                    np.full(N_COLUMNS, column_x[0] / 100000.0),
                    np.full(N_COLUMNS, column_y[0] / 100000.0),
                    np.full(N_COLUMNS, np.mean(~finite_gr)),
                ],
                axis=0,
            ).astype(np.float32)
            tw = typewell[["TVT", "GR"]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            tw = tw.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
            tw_tvt = tw["TVT"].to_numpy(float)
            tw_gr = _robust_normalise(tw["GR"].to_numpy(float))
            if len(tw_tvt) < 32:
                raise RuntimeError(f"{well}: typewell too short")
            official_cut = int(cuts[(well, "official")])
            eval_rows = np.arange(official_cut, len(md), dtype=np.int32)
            record = {
                "well": well,
                "md": md.astype(np.float32),
                "gr": gr.astype(np.float32),
                "gr_finite": finite_gr,
                "truth": truth.astype(np.float32),
                "tvt_input": tvt_input.astype(np.float32),
                "column_md": column_md.astype(np.float32),
                "column_x": column_x.astype(np.float32),
                "column_y": column_y.astype(np.float32),
                "column_z": column_z.astype(np.float32),
                "column_truth": column_truth.astype(np.float32),
                "column_gr": column_gr,
                "column_gr_valid": column_gr_valid,
                "static_geometry": static_geometry,
                "tw_tvt": tw_tvt.astype(np.float32),
                "tw_gr": tw_gr.astype(np.float32),
                "cuts": {tag: int(cuts[(well, tag)]) for tag in TRAIN_TAGS},
                "official_cut": official_cut,
                "official_n_eval": int(official_rows.loc[well, "official_n_eval"]),
                "eval_rows": eval_rows,
                "eval_md": md[eval_rows].astype(np.float32),
                "eval_z": z[eval_rows].astype(np.float32),
                "eval_truth": truth[eval_rows].astype(np.float32),
                "stable_slot": _stable_slot(well),
            }
            self.records.append(record)

    def __len__(self):
        return len(self.records)


def _compose_panel(record, tag="official", mode="full", donor=None):
    cut = int(record["cuts"][tag])
    anchor = cut - 1
    anchor_md = float(record["md"][anchor])
    anchor_tvt = float(record["truth"][anchor])
    state_tvt = anchor_tvt + STATE_OFFSETS
    horizontal = record["column_gr"]
    horizontal_valid = record["column_gr_valid"]
    if mode == "shuffled_typewell":
        if donor is None:
            raise RuntimeError("shuffled_typewell requires donor")
        tw_tvt = donor["tw_tvt"]
        tw_gr_source = donor["tw_gr"]
    else:
        tw_tvt = record["tw_tvt"]
        tw_gr_source = record["tw_gr"]
    typewell = _interp_finite(
        tw_tvt, tw_gr_source, state_tvt, left=np.nan, right=np.nan
    )
    typewell_valid = np.isfinite(typewell).astype(np.float32)
    typewell = np.nan_to_num(typewell, nan=0.0).astype(np.float32)

    visible = np.arange(cut, dtype=np.int32)
    visible_valid = record["gr_finite"][:cut] & np.isfinite(record["truth"][:cut])
    if int(visible_valid.sum()) >= 16:
        heel = _interp_finite(
            record["truth"][:cut][visible_valid],
            record["gr"][:cut][visible_valid],
            state_tvt,
            left=np.nan,
            right=np.nan,
        )
    else:
        heel = np.full(len(state_tvt), np.nan, np.float64)
    heel_valid = np.isfinite(heel).astype(np.float32)
    heel = np.nan_to_num(heel, nan=0.0).astype(np.float32)
    state_valid_vector = (typewell_valid > 0.5) | (heel_valid > 0.5)

    known_mask_1d = (record["column_md"] <= anchor_md).astype(np.float32)
    eval_mask_1d = 1.0 - known_mask_1d
    target_offset = record["column_truth"] - anchor_tvt
    target_in_axis = (
        np.isfinite(target_offset)
        & (np.abs(target_offset) <= float(np.max(np.abs(STATE_OFFSETS))))
    )
    nearest = np.argmin(
        np.abs(STATE_OFFSETS[:, None] - target_offset[None, :]), axis=0
    )
    target_state_valid = state_valid_vector[nearest]
    target_valid = target_in_axis & target_state_valid
    known_path_valid = known_mask_1d * target_valid.astype(np.float32)
    known_heat = np.exp(
        -0.5
        * ((STATE_OFFSETS[:, None] - target_offset[None, :]) / TARGET_SIGMA_FT) ** 2
    ) * known_path_valid[None, :]

    states = len(STATE_OFFSETS)
    h = np.broadcast_to(horizontal[None, :], (states, N_COLUMNS))
    h_valid = np.broadcast_to(horizontal_valid[None, :], (states, N_COLUMNS))
    tw_map = np.broadcast_to(typewell[:, None], (states, N_COLUMNS))
    tw_valid_map = np.broadcast_to(typewell_valid[:, None], (states, N_COLUMNS))
    heel_map = np.broadcast_to(heel[:, None], (states, N_COLUMNS))
    heel_valid_map = np.broadcast_to(heel_valid[:, None], (states, N_COLUMNS))
    state_offset_map = np.broadcast_to(
        (STATE_OFFSETS / 128.0)[:, None], (states, N_COLUMNS)
    )
    tw_span = max(float(tw_tvt[-1] - tw_tvt[0]), 1.0)
    absolute_position = np.broadcast_to(
        ((state_tvt - 0.5 * (tw_tvt[0] + tw_tvt[-1])) / tw_span)[:, None],
        (states, N_COLUMNS),
    )
    edge_margin = np.minimum(state_tvt - tw_tvt[0], tw_tvt[-1] - state_tvt)
    edge_margin = np.broadcast_to(
        np.clip(edge_margin / 128.0, -2.0, 2.0)[:, None],
        (states, N_COLUMNS),
    )
    raw = np.stack(
        [
            h,
            h_valid,
            tw_map,
            tw_valid_map,
            h * tw_map,
            -np.abs(h - tw_map),
            heel_map,
            heel_valid_map,
            h * heel_map,
            -np.abs(h - heel_map),
            known_heat,
            np.broadcast_to(known_path_valid[None, :], (states, N_COLUMNS)),
            state_offset_map,
            absolute_position,
            edge_margin,
        ],
        axis=0,
    ).astype(np.float32)
    if mode == "no_gr":
        raw[list(GR_VALUE_CHANNELS)] = 0.0
    elif mode not in ("full", "shuffled_typewell"):
        raise ValueError(mode)

    dynamic_geometry = np.stack(
        [
            known_mask_1d,
            eval_mask_1d,
            (record["column_md"] - anchor_md) / 1000.0,
            np.full(N_COLUMNS, cut / len(record["md"])),
            np.full(N_COLUMNS, anchor_tvt / 10000.0),
        ],
        axis=0,
    ).astype(np.float32)
    geometry_1d = np.concatenate([record["static_geometry"], dynamic_geometry], axis=0)
    geometry = np.broadcast_to(
        geometry_1d[:, None, :],
        (geometry_1d.shape[0], states, N_COLUMNS),
    )
    values = np.concatenate([raw, geometry], axis=0).astype(np.float32)
    state_valid = np.broadcast_to(
        state_valid_vector[:, None], (states, N_COLUMNS)
    ).copy()
    loss_weight = (
        KNOWN_COLUMN_WEIGHT * known_mask_1d + eval_mask_1d
    ) * target_valid.astype(np.float32)
    return {
        "input": values,
        "state_valid": state_valid,
        "target_offset": target_offset.astype(np.float32),
        "target_valid": target_valid,
        "loss_weight": loss_weight.astype(np.float32),
        "known_mask": known_mask_1d,
        "eval_mask": eval_mask_1d,
        "anchor_tvt": anchor_tvt,
        "anchor_md": anchor_md,
        "column_md": record["column_md"],
        "column_z": record["column_z"],
        "sample_weight": float(record["official_n_eval"]),
        "tag": tag,
    }


class CompleteWellDataset(Dataset):
    def __init__(self, store, training):
        self.store = store
        self.training = bool(training)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.store)

    def __getitem__(self, index):
        record = self.store.records[index]
        if self.training:
            tag_index = (record["stable_slot"] + self.epoch) % len(TRAIN_TAGS)
            tag = TRAIN_TAGS[tag_index]
        else:
            tag = "official"
        panel = _compose_panel(record, tag=tag, mode="full")
        return {
            "input": torch.from_numpy(panel["input"]),
            "state_valid": torch.from_numpy(panel["state_valid"]),
            "target_offset": torch.from_numpy(panel["target_offset"]),
            "loss_weight": torch.from_numpy(panel["loss_weight"]),
            "column_md": torch.from_numpy(panel["column_md"]),
            "column_z": torch.from_numpy(panel["column_z"]),
            "sample_weight": torch.tensor(panel["sample_weight"], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
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


class DilatedContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(3, 5),
                        padding=(1, 2 * dilation),
                        dilation=(1, dilation),
                        groups=channels,
                    ),
                    nn.Conv2d(channels, channels, 1),
                    nn.GroupNorm(8, channels),
                    nn.GELU(),
                )
                for dilation in (1, 2, 4, 8)
            ]
        )

    def forward(self, values):
        output = values
        for block in self.blocks:
            output = output + block(output)
        return output


class FullWellGlobalUNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        base = BASE_CHANNELS
        self.enc1 = ResidualBlock(input_channels, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.enc3 = ResidualBlock(base * 4, base * 4)
        self.down3 = nn.Conv2d(base * 4, base * 6, 3, stride=2, padding=1)
        self.enc4 = ResidualBlock(base * 6, base * 6)
        self.down4 = nn.Conv2d(base * 6, base * 8, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base * 8, base * 8),
            DilatedContext(base * 8),
            ResidualBlock(base * 8, base * 8),
        )
        self.dec4 = ResidualBlock(base * 8 + base * 6, base * 6)
        self.dec3 = ResidualBlock(base * 6 + base * 4, base * 4)
        self.dec2 = ResidualBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ResidualBlock(base * 2 + base, base)
        self.head = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.log_emission_scale = nn.Parameter(torch.tensor(0.0))
        self.log_prior_sigma = nn.Parameter(torch.tensor(np.log(48.0), dtype=torch.float32))

    @staticmethod
    def _up(values, skip):
        return F.interpolate(
            values, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, values, state_offsets):
        e1 = self.enc1(values)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        bottleneck = self.bottleneck(self.down4(e4))
        d4 = self.dec4(torch.cat([self._up(bottleneck, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        residual = self.head(d1).squeeze(1)
        emission_scale = torch.exp(self.log_emission_scale).clamp(0.05, 20.0)
        emission = emission_scale * values[:, EMISSION_CHANNEL]
        known_prior = 8.0 * values[:, KNOWN_HEAT_CHANNEL]
        sigma = torch.exp(self.log_prior_sigma).clamp(8.0, 128.0)
        prior = -0.5 * (state_offsets[:, None] / sigma) ** 2
        return residual + emission + known_prior + prior[None]


def _loss(
    logits,
    state_valid,
    target_offset,
    loss_weight,
    column_md,
    column_z,
    sample_weight,
    state_offsets,
    ce_only,
    mean_sample_weight,
):
    logits = logits.float().masked_fill(~state_valid, -30.0)
    distance = state_offsets[None, :, None] - target_offset[:, None, :]
    target = torch.softmax(-0.5 * (distance / TARGET_SIGMA_FT) ** 2, dim=1)
    target = target * state_valid
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    probability = torch.softmax(logits, dim=1)
    prediction = (probability * state_offsets[None, :, None]).sum(dim=1)
    denominator = loss_weight.sum(dim=1).clamp_min(1e-6)
    panel_scale = (sample_weight / mean_sample_weight).clamp(0.25, 4.0)

    ce_columns = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    ce_by_panel = (ce_columns * loss_weight).sum(dim=1) / denominator
    ce = (ce_by_panel * panel_scale).mean()
    squared = (prediction - target_offset) ** 2
    mse_by_panel = (squared * loss_weight).sum(dim=1) / denominator
    mse = (mse_by_panel * panel_scale).mean()

    true_s = target_offset + column_z - column_z[:, :1]
    predicted_s = prediction + column_z - column_z[:, :1]
    dmd = (column_md[:, 1:] - column_md[:, :-1]).clamp_min(1e-3)
    true_dip = (true_s[:, 1:] - true_s[:, :-1]) / dmd
    predicted_dip = (predicted_s[:, 1:] - predicted_s[:, :-1]) / dmd
    pair_weight = torch.minimum(loss_weight[:, 1:], loss_weight[:, :-1])
    pair_denominator = pair_weight.sum(dim=1).clamp_min(1e-6)
    dip_by_panel = (
        ((predicted_dip - true_dip) ** 2 * pair_weight).sum(dim=1)
        / pair_denominator
    )
    dip = (dip_by_panel * panel_scale).mean()
    curvature_weight = torch.minimum(pair_weight[:, 1:], pair_weight[:, :-1])
    curvature_by_panel = (
        (
            (
                (predicted_dip[:, 1:] - predicted_dip[:, :-1])
                - (true_dip[:, 1:] - true_dip[:, :-1])
            ).square()
            * curvature_weight
        ).sum(dim=1)
        / curvature_weight.sum(dim=1).clamp_min(1e-6)
    )
    curvature = (curvature_by_panel * panel_scale).mean()
    total = ce if ce_only else (
        LOSS_CE * ce
        + LOSS_MSE * mse
        + LOSS_DIP * dip
        + LOSS_CURVATURE * curvature
    )
    return total, {
        "ce": ce,
        "mse": mse,
        "dip": dip,
        "curvature": curvature,
    }


@torch.no_grad()
def _predict_record(model, store, index, mode, device, state_offsets):
    record = store.records[index]
    donor = store.records[(index + max(1, len(store) // 3)) % len(store)]
    panel = _compose_panel(
        record,
        tag="official",
        mode=mode,
        donor=donor if mode == "shuffled_typewell" else None,
    )
    values = torch.from_numpy(panel["input"])[None].to(device)
    state_valid = torch.from_numpy(panel["state_valid"])[None].to(device)
    logits = model(values, state_offsets).float().masked_fill(~state_valid, -30.0)
    probability = torch.softmax(logits, dim=1)
    offset = (
        probability * state_offsets[None, :, None]
    ).sum(dim=1)[0].cpu().numpy()
    # The visible path is an exact condition, not a prediction target at inference.
    known = panel["known_mask"] > 0.5
    offset[known] = panel["target_offset"][known]
    column_tvt = panel["anchor_tvt"] + offset
    column_s = column_tvt + record["column_z"]
    eval_s = np.interp(record["eval_md"], record["column_md"], column_s)
    prediction = eval_s - record["eval_z"]
    anchor_prediction = np.full(len(prediction), panel["anchor_tvt"], np.float64)
    return prediction, anchor_prediction


@torch.no_grad()
def _evaluate(model, store, mode, device, state_offsets, return_rows=False):
    errors = []
    anchor_errors = []
    by_well = []
    row_frames = []
    model.eval()
    for index, record in enumerate(store.records):
        prediction, anchor_prediction = _predict_record(
            model, store, index, mode, device, state_offsets
        )
        truth = record["eval_truth"].astype(float)
        error = prediction - truth
        anchor_error = anchor_prediction - truth
        errors.append(error)
        anchor_errors.append(anchor_error)
        by_well.append(
            {
                "well": record["well"],
                "rows": len(error),
                "anchor_rmse": float(np.sqrt(np.mean(anchor_error**2))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
            }
        )
        if return_rows:
            row_frames.append(
                pd.DataFrame(
                    {
                        "well": record["well"],
                        "row_idx": record["eval_rows"],
                        "truth": truth,
                        "prediction": prediction,
                    }
                )
            )
    errors = np.concatenate(errors)
    anchor_errors = np.concatenate(anchor_errors)
    by_well = pd.DataFrame(by_well)
    metrics = {
        "anchor_rmse": float(np.sqrt(np.mean(anchor_errors**2))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "gain_vs_anchor": float(
            np.sqrt(np.mean(anchor_errors**2)) - np.sqrt(np.mean(errors**2))
        ),
        "mae": float(np.mean(np.abs(errors))),
        "median_well_rmse": float(by_well["rmse"].median()),
        "p95_well_rmse": float(by_well["rmse"].quantile(0.95)),
    }
    rows = pd.concat(row_frames, ignore_index=True) if return_rows else None
    return metrics, by_well, rows


def _train(train_store, valid_store, input_channels, device, state_offsets):
    _seed_everything(SEED)
    model = FullWellGlobalUNet(input_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    dataset = CompleteWellDataset(train_store, training=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    mean_sample_weight = torch.tensor(
        np.mean([record["official_n_eval"] for record in train_store.records]),
        dtype=torch.float32,
        device=device,
    )
    checkpoint_path = MODEL_DIR / f"seed_{SEED}.pt"
    best_rmse = np.inf
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()

    for epoch in range(1, EPOCHS + 1):
        dataset.set_epoch(epoch - 1)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_rows = []
        ce_only = epoch <= WARMUP_EPOCHS
        for batch_index, batch in enumerate(loader, 1):
            values = batch["input"].to(device, non_blocking=True)
            state_valid = batch["state_valid"].to(device, non_blocking=True)
            target_offset = batch["target_offset"].to(device, non_blocking=True)
            loss_weight = batch["loss_weight"].to(device, non_blocking=True)
            column_md = batch["column_md"].to(device, non_blocking=True)
            column_z = batch["column_z"].to(device, non_blocking=True)
            sample_weight = batch["sample_weight"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(values, state_offsets)
                loss, components = _loss(
                    logits,
                    state_valid,
                    target_offset,
                    loss_weight,
                    column_md,
                    column_z,
                    sample_weight,
                    state_offsets,
                    ce_only,
                    mean_sample_weight,
                )
                scaled_loss = loss / GRADIENT_ACCUMULATION
            scaler.scale(scaled_loss).backward()
            should_step = (
                batch_index % GRADIENT_ACCUMULATION == 0
                or batch_index == len(loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    **{
                        name: float(value.detach().cpu())
                        for name, value in components.items()
                    },
                }
            )
        scheduler.step()
        metrics, by_well, _ = _evaluate(
            model, valid_store, "full", device, state_offsets
        )
        row = {
            "epoch": epoch,
            "phase": "ce_only" if ce_only else "joint",
            **{
                f"train_{name}": float(np.mean([item[name] for item in batch_rows]))
                for name in batch_rows[0]
            },
            **{f"valid_{name}": value for name, value in metrics.items()},
            "emission_scale": float(torch.exp(model.log_emission_scale).detach().cpu()),
            "prior_sigma": float(torch.exp(model.log_prior_sigma).detach().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("FULL-WELL EPOCH", row, flush=True)
        if metrics["rmse"] < best_rmse - 1e-4:
            best_rmse = float(metrics["rmse"])
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "input_channels": input_channels,
                    "seed": SEED,
                },
                checkpoint_path,
            )
            by_well.to_csv(OUT_DIR / "best_bywell.csv", index=False)
        elif epoch > WARMUP_EPOCHS:
            stale += 1
        if epoch >= WARMUP_EPOCHS + 5 and stale >= EARLY_STOPPING_REPORTS:
            print(f"full-well early stopping at epoch {epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics, by_well, rows = _evaluate(
        model, valid_store, "full", device, state_offsets, return_rows=True
    )
    by_well.to_csv(OUT_DIR / "final_bywell.csv", index=False)
    rows.to_pickle(OUT_DIR / "fold1_predictions.pkl")
    result = {
        "best_epoch": int(checkpoint["epoch"]),
        **metrics,
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    del optimizer, scheduler, scaler, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, result


for directory in (OUT_DIR, MODEL_DIR):
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
manifest = _load_manifest(wells)
manifest.to_csv(OUT_DIR / "cut_manifest.csv", index=False)
train_manifest = manifest[manifest["well"].isin(set(train_wells))].copy()
valid_manifest = manifest[manifest["well"].isin(set(valid_wells))].copy()

train_store = CompleteWellStore(train_wells, train_manifest)
valid_store = CompleteWellStore(valid_wells, valid_manifest)
sample_panel = _compose_panel(train_store.records[0], tag="official", mode="full")
input_channels = int(sample_panel["input"].shape[0])
eval_coverages = []
known_coverages = []
for record in train_store.records + valid_store.records:
    panel = _compose_panel(record, tag="official", mode="full")
    eval_mask = panel["eval_mask"] > 0.5
    known_mask = panel["known_mask"] > 0.5
    eval_coverages.append(float(np.mean(panel["target_valid"][eval_mask])))
    known_coverages.append(float(np.mean(panel["target_valid"][known_mask])))
guards = {
    "input_shape": list(sample_panel["input"].shape),
    "input_channels": input_channels,
    "eval_coverage_mean": float(np.mean(eval_coverages)),
    "eval_coverage_p10": float(np.quantile(eval_coverages, 0.10)),
    "known_coverage_mean": float(np.mean(known_coverages)),
    "finite_input": bool(np.isfinite(sample_panel["input"]).all()),
}
if (
    guards["eval_coverage_mean"] < 0.995
    or guards["eval_coverage_p10"] < 0.98
    or not guards["finite_input"]
):
    raise RuntimeError(f"global panel guard failed: {guards}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")
state_offsets_tensor = torch.from_numpy(STATE_OFFSETS).to(device)
parameter_probe = FullWellGlobalUNet(input_channels)
parameters = sum(parameter.numel() for parameter in parameter_probe.parameters())
del parameter_probe
print(
    {
        "experiment": "fullwell_global_tvt_unet_onefold_v0",
        "device": str(device),
        "fold": VALID_FOLD,
        "train_wells": len(train_store),
        "valid_wells": len(valid_store),
        "columns": N_COLUMNS,
        "states": len(STATE_OFFSETS),
        "state_half_width_ft": float(np.max(np.abs(STATE_OFFSETS))),
        "input_channels": input_channels,
        "parameters": parameters,
        "guards": guards,
        "sample_unit": "complete well",
        "standalone": True,
    },
    flush=True,
)

model, result = _train(
    train_store, valid_store, input_channels, device, state_offsets_tensor
)
intervention_rows = []
for mode in ("full", "no_gr", "shuffled_typewell"):
    metrics, _, rows = _evaluate(
        model, valid_store, mode, device, state_offsets_tensor, return_rows=True
    )
    rows.to_pickle(OUT_DIR / f"intervention_{mode}.pkl")
    intervention_rows.append({"mode": mode, **metrics})
interventions = pd.DataFrame(intervention_rows)
full_rmse = float(interventions.loc[interventions["mode"] == "full", "rmse"].iloc[0])
interventions["degradation_vs_full"] = interventions["rmse"] - full_rmse
interventions.to_csv(OUT_DIR / "interventions.csv", index=False)
no_gr_degradation = float(
    interventions.loc[interventions["mode"] == "no_gr", "degradation_vs_full"].iloc[0]
)
shuffle_degradation = float(
    interventions.loc[
        interventions["mode"] == "shuffled_typewell", "degradation_vs_full"
    ].iloc[0]
)
architecture_pass = bool(result["rmse"] < 7.0)
causal_pass = bool(no_gr_degradation >= 0.50 and shuffle_degradation >= 0.50)
strict_pass = bool(architecture_pass and causal_pass)
verdict = {
    "experiment": "fullwell_global_tvt_unet_onefold_v0",
    "sample_unit": "complete well",
    "standalone": True,
    "fold": VALID_FOLD,
    "fold_source": str(fold_path),
    "train_wells": len(train_store),
    "valid_wells": len(valid_store),
    "columns": N_COLUMNS,
    "states": len(STATE_OFFSETS),
    "input_channels": input_channels,
    "parameters": parameters,
    "guards": guards,
    "result": result,
    "no_gr_degradation": no_gr_degradation,
    "shuffled_typewell_degradation": shuffle_degradation,
    "architecture_pass": architecture_pass,
    "causal_pass": causal_pass,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> run all five folds; pooled target <=5.8 then <=5.5 ft"
        if strict_pass
        else "FAIL -> inspect complete-well representation/decoder before any ensemble or postprocess"
    ),
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nRESULT")
print(json.dumps(result, indent=2))
print("\nINTERVENTIONS")
print(interventions.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)

