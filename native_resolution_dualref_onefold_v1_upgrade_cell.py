"""V1 upgrade after the 43K native model failed to generalise on fold 1.

Run in the same notebook kernel after interrupting onefold v0.  The cell reuses
the 773 official native records already in memory/on disk and adds:

1. a ~million-parameter multi-scale GR encoder;
2. a temporal U-Net whose receptive field covers the complete well;
3. two deterministic fold-safe pseudo prediction starts per outer-train well;
4. a direct eval-row weighted squared-path loss;
5. a fixed outer-train subset metric at every report, separating optimisation
   failure from held-out generalisation failure.

Validation remains the untouched official mask for fold 1.  No neighbour well,
PF, tabular prediction, CP12, Gold, or submission signal is consumed.
"""

from pathlib import Path
import gc
import hashlib
import json
import math
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reuse the validated live-kernel data definitions and official records.
# ---------------------------------------------------------------------------


REQUIRED_LIVE = {
    "Record",
    "records",
    "folds",
    "train_wells",
    "valid_wells",
    "train_records",
    "valid_records",
    "DATA_ROOT",
    "OOF_ROOT",
    "STATE_OFFSETS",
    "N_STATES",
    "HORIZONTAL_STRIDE",
    "STATE_HALF_WIDTH_FT",
    "REFERENCE_OFFSETS",
    "REFERENCE_STATE_START",
    "_interp_finite",
    "_nearest_distance",
    "_robust_affine",
    "_robust_location_scale",
    "_normalise",
    "_fill_1d",
    "_gradient",
    "_reference_features",
    "_heel_reference",
    "_rmse",
    "_seed_everything",
}
missing_live = sorted(REQUIRED_LIVE.difference(globals()))
if missing_live:
    raise RuntimeError(
        "Run/interrupt native_resolution_dualref_onefold_v0_cell.py in this "
        f"kernel first; missing live definitions: {missing_live}"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


EXPERIMENT_V1 = "native_resolution_dualref_onefold_v1"
OUT_DIR_V1 = OOF_ROOT / EXPERIMENT_V1
MODEL_DIR_V1 = OUT_DIR_V1 / "models"
PSEUDO_CACHE_DIR_V1 = OUT_DIR_V1 / "pseudo_records"

SEED_V1 = 20260720
EPOCHS_V1 = 32
REPORT_EVERY_V1 = 2
MIN_EPOCHS_V1 = 10
EARLY_STOPPING_REPORTS_V1 = 5
LEARNING_RATE_V1 = 2.5e-4
MIN_LEARNING_RATE_V1 = 3e-5
WEIGHT_DECAY_V1 = 3e-5
GRADIENT_CLIP_V1 = 3.0
GRAD_SCALER_INITIAL_SCALE_V1 = 512.0

EMBEDDING_V1 = 48
SCALE_EMBEDDING_V1 = 16
CONDITION_CHANNELS_V1 = 32
BASE_CHANNELS_V1 = 48
TARGET_SIGMA_FT_V1 = 1.5

LOSS_CE_V1 = 0.35
LOSS_MSE_V1 = 1.00
LOSS_SLOPE_V1 = 0.08
LOSS_CURVATURE_V1 = 0.02
MSE_SCALE_FT_V1 = 8.0

OFFICIAL_STATE_PROBABILITY = 0.50
PSEUDO_LOW_RANGE = (0.17, 0.24)
PSEUDO_HIGH_RANGE = (0.28, 0.36)
PSEUDO_AXIS_COVERAGE_MIN = 0.995

ARCHITECTURE_GATE_FT_V1 = 7.0
TARGET_GATE_FT_V1 = 5.5
FAIL_FAST_EPOCH_V1 = 12
FAIL_FAST_VALID_RMSE_V1 = 11.5
FAIL_FAST_TRAIN_SUBSET_RMSE_V1 = 4.0

for directory in (OUT_DIR_V1, MODEL_DIR_V1, PSEUDO_CACHE_DIR_V1):
    directory.mkdir(parents=True, exist_ok=True)


def _write_json_v1(path, payload):
    def default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


def _stable_unit(well, tag):
    digest = hashlib.blake2b(
        f"{well}|{tag}|{SEED_V1}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") / float(2**64 - 1)


def _pseudo_fraction(well, tag):
    if tag == "pseudo_low":
        lo, hi = PSEUDO_LOW_RANGE
    elif tag == "pseudo_high":
        lo, hi = PSEUDO_HIGH_RANGE
    else:
        raise ValueError(tag)
    return lo + (hi - lo) * _stable_unit(well, tag)


def _pseudo_path_v1(well, tag):
    return PSEUDO_CACHE_DIR_V1 / f"{well}__{tag}.npz"


def _save_pseudo_v1(record, tag, fraction):
    np.savez_compressed(
        _pseudo_path_v1(record.well, tag),
        well=np.asarray(record.well),
        tag=np.asarray(tag),
        fraction=np.asarray([fraction], np.float32),
        horizontal=record.horizontal.astype(np.float16),
        condition=record.condition.astype(np.float16),
        typewell=record.typewell.astype(np.float16),
        heel=record.heel.astype(np.float16),
        latent_index=record.latent_index.astype(np.int32),
        latent_truth_offset=record.latent_truth_offset.astype(np.float32),
        latent_eval_mask=record.latent_eval_mask.astype(np.uint8),
        md=record.md.astype(np.float32),
        eval_index=record.eval_index.astype(np.int32),
        truth=record.truth.astype(np.float32),
        anchor_tvt=np.asarray([record.anchor_tvt], np.float32),
        anchor_index=np.asarray([record.anchor_index], np.int32),
        eval_rows=np.asarray([record.eval_rows], np.int32),
        calibration=np.asarray(
            [
                record.calibration_a,
                record.calibration_b,
                record.calibration_pairs,
                record.calibration_rmse,
            ],
            np.float32,
        ),
        teacher_rmse=np.asarray([record.teacher_rmse], np.float32),
        axis_coverage=np.asarray([record.axis_coverage], np.float32),
        cache_version=np.asarray([1], np.int16),
    )


def _load_pseudo_v1(well, tag):
    with np.load(_pseudo_path_v1(well, tag), allow_pickle=False) as data:
        calibration = data["calibration"].astype(np.float64)
        return Record(
            well=str(data["well"].item()),
            horizontal=data["horizontal"].astype(np.float32),
            condition=data["condition"].astype(np.float32),
            typewell=data["typewell"].astype(np.float32),
            heel=data["heel"].astype(np.float32),
            latent_index=data["latent_index"].astype(np.int32),
            latent_truth_offset=data["latent_truth_offset"].astype(np.float32),
            latent_eval_mask=data["latent_eval_mask"].astype(bool),
            md=data["md"].astype(np.float32),
            eval_index=data["eval_index"].astype(np.int32),
            truth=data["truth"].astype(np.float32),
            anchor_tvt=float(data["anchor_tvt"][0]),
            anchor_index=int(data["anchor_index"][0]),
            eval_rows=int(data["eval_rows"][0]),
            calibration_a=float(calibration[0]),
            calibration_b=float(calibration[1]),
            calibration_pairs=int(round(calibration[2])),
            calibration_rmse=float(calibration[3]),
            teacher_rmse=float(data["teacher_rmse"][0]),
            axis_coverage=float(data["axis_coverage"][0]),
        )


def _build_record_at_cut_v1(well, fraction):
    horizontal = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    )
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")

    def numeric(frame, column):
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)

    md = numeric(horizontal, "MD")
    x = numeric(horizontal, "X")
    y = numeric(horizontal, "Y")
    z = numeric(horizontal, "Z")
    gr = numeric(horizontal, "GR")
    truth = numeric(horizontal, "TVT")
    tw_tvt = numeric(typewell, "TVT")
    tw_gr = numeric(typewell, "GR")
    n_rows = len(horizontal)
    cut = int(np.clip(round(float(fraction) * n_rows), 128, n_rows - 128))
    known = np.arange(cut, dtype=np.int32)
    eval_index = np.flatnonzero(
        (np.arange(n_rows) >= cut) & np.isfinite(truth)
    ).astype(np.int32)
    anchor_index = cut - 1
    anchor_tvt = float(truth[anchor_index])
    visible_tvt = np.full(n_rows, np.nan, np.float64)
    visible_tvt[known] = truth[known]

    tw_at_prefix = _interp_finite(tw_tvt, tw_gr, visible_tvt[known])
    a, b, pairs, calibration_rmse = _robust_affine(tw_at_prefix, gr[known])
    tw_calibrated = a * tw_gr + b
    centre, scale = _robust_location_scale(gr)
    gr_normalised = _normalise(gr, centre, scale)
    tw_normalised = _normalise(tw_calibrated, centre, scale)

    gr_valid = np.isfinite(gr).astype(np.float64)
    gr_filled = _fill_1d(md, gr_normalised)
    gr_d1 = _gradient(gr_filled, md)
    gr_d2 = _gradient(gr_d1, md)
    horizontal_features = np.stack(
        [
            np.clip(gr_filled, -8.0, 8.0),
            gr_valid,
            np.clip(gr_d1, -4.0, 4.0),
            np.clip(gr_d2, -4.0, 4.0),
        ],
        axis=0,
    ).astype(np.float32)

    dx = _gradient(_fill_1d(md, x), md)
    dy = _gradient(_fill_1d(md, y), md)
    dz = _gradient(_fill_1d(md, z), md)
    d2z = _gradient(dz, md)
    progress = (md - md[0]) / max(md[-1] - md[0], 1.0)
    ps_progress = (md - md[anchor_index]) / max(md[-1] - md[anchor_index], 1.0)
    z_relative = (_fill_1d(md, z) - z[anchor_index]) / 512.0
    known_mask = np.zeros(n_rows, np.float64)
    known_mask[known] = 1.0
    known_offset = np.where(
        known_mask > 0,
        (visible_tvt - anchor_tvt) / STATE_HALF_WIDTH_FT,
        0.0,
    )
    condition = np.stack(
        [
            known_mask,
            np.clip(known_offset, -2.0, 2.0),
            np.clip(progress, 0.0, 1.0),
            np.clip(ps_progress, -1.0, 1.0),
            np.clip(z_relative, -4.0, 4.0),
            np.clip(dx, -2.0, 2.0),
            np.clip(dy, -2.0, 2.0),
            np.clip(dz, -2.0, 2.0),
            np.clip(d2z * 32.0, -2.0, 2.0),
            gr_valid,
        ],
        axis=0,
    ).astype(np.float32)

    reference_tvt = anchor_tvt + REFERENCE_OFFSETS
    typewell_values = _interp_finite(tw_tvt, tw_normalised, reference_tvt)
    typewell_valid = (
        np.isfinite(typewell_values)
        & (_nearest_distance(tw_tvt[np.isfinite(tw_gr)], reference_tvt) <= 1.25)
    ).astype(np.float64)
    typewell_features = _reference_features(typewell_values, typewell_valid)
    heel_values, heel_valid = _heel_reference(
        visible_tvt[known], gr_normalised[known], reference_tvt
    )
    heel_features = _reference_features(heel_values, heel_valid)

    latent_index = np.arange(0, n_rows, HORIZONTAL_STRIDE, dtype=np.int32)
    latent_truth = truth[latent_index] - anchor_tvt
    latent_eval_mask = (latent_index >= cut) & np.isfinite(latent_truth)
    eval_offset = truth[eval_index] - anchor_tvt
    axis_coverage = float(np.mean(np.abs(eval_offset) <= STATE_HALF_WIDTH_FT))
    latent_eval_index = latent_index[latent_eval_mask]
    support_md = np.r_[md[anchor_index], md[latent_eval_index]]
    support_tvt = np.r_[anchor_tvt, truth[latent_eval_index]]
    teacher = np.interp(md[eval_index], support_md, support_tvt)
    teacher_rmse = _rmse(truth[eval_index], teacher)

    return Record(
        well=well,
        horizontal=horizontal_features,
        condition=condition,
        typewell=typewell_features,
        heel=heel_features,
        latent_index=latent_index,
        latent_truth_offset=latent_truth.astype(np.float32),
        latent_eval_mask=latent_eval_mask,
        md=md.astype(np.float32),
        eval_index=eval_index,
        truth=truth.astype(np.float32),
        anchor_tvt=anchor_tvt,
        anchor_index=anchor_index,
        eval_rows=len(eval_index),
        calibration_a=a,
        calibration_b=b,
        calibration_pairs=pairs,
        calibration_rmse=calibration_rmse,
        teacher_rmse=teacher_rmse,
        axis_coverage=axis_coverage,
    )


def _prepare_pseudo_bank_v1():
    rows = []
    bank = {}
    for index, well in enumerate(train_wells, 1):
        states = {"official": records[well]}
        for tag in ("pseudo_low", "pseudo_high"):
            fraction = _pseudo_fraction(well, tag)
            path = _pseudo_path_v1(well, tag)
            try:
                if path.exists():
                    record = _load_pseudo_v1(well, tag)
                    source = "cached"
                else:
                    record = _build_record_at_cut_v1(well, fraction)
                    _save_pseudo_v1(record, tag, fraction)
                    source = "built"
                eligible = bool(
                    record.axis_coverage >= PSEUDO_AXIS_COVERAGE_MIN
                    and record.teacher_rmse <= 0.35
                )
                if eligible:
                    states[tag] = record
                rows.append(
                    {
                        "well": well,
                        "tag": tag,
                        "fraction": fraction,
                        "source": source,
                        "axis_coverage": record.axis_coverage,
                        "teacher_rmse": record.teacher_rmse,
                        "eval_rows": record.eval_rows,
                        "eligible": eligible,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "well": well,
                        "tag": tag,
                        "fraction": fraction,
                        "source": "failed",
                        "eligible": False,
                        "reason": repr(exc),
                    }
                )
        bank[well] = states
        if index % 50 == 0 or index == len(train_wells):
            print(f"build/reuse pseudo states {index}/{len(train_wells)}", flush=True)
    report = pd.DataFrame(rows)
    report.to_csv(OUT_DIR_V1 / "pseudo_state_audit.csv", index=False)
    eligible_fraction = float(report["eligible"].fillna(False).mean())
    print(
        "PSEUDO BANK",
        {
            "rows": len(report),
            "eligible_fraction": eligible_fraction,
            "axis_coverage_min_eligible": float(
                report.loc[report["eligible"], "axis_coverage"].min()
            ),
            "teacher_rmse_max_eligible": float(
                report.loc[report["eligible"], "teacher_rmse"].max()
            ),
        },
        flush=True,
    )
    if eligible_fraction < 0.90:
        raise RuntimeError(
            f"Only {eligible_fraction:.3f} pseudo states passed; repair the cut bank"
        )
    return bank, report


# ---------------------------------------------------------------------------
# V1 model: multi-scale shared GR encoder plus a global temporal U-Net.
# ---------------------------------------------------------------------------


class Residual1DV1(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
        )

    def forward(self, values):
        return F.gelu(values + self.block(values))


class MultiScaleGREncoderV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, EMBEDDING_V1, 9, padding=4),
            nn.GroupNorm(8, EMBEDDING_V1),
            nn.GELU(),
        )
        self.local = nn.Sequential(
            Residual1DV1(EMBEDDING_V1, 1),
            Residual1DV1(EMBEDDING_V1, 2),
        )
        self.middle = nn.Sequential(
            Residual1DV1(EMBEDDING_V1, 4),
            Residual1DV1(EMBEDDING_V1, 8),
        )
        self.long = nn.Sequential(
            Residual1DV1(EMBEDDING_V1, 16),
            Residual1DV1(EMBEDDING_V1, 32),
        )
        self.project = nn.ModuleList(
            [nn.Conv1d(EMBEDDING_V1, SCALE_EMBEDDING_V1, 1) for _ in range(3)]
        )

    def forward(self, values):
        base = self.stem(values)
        local = self.local(base)
        middle = self.middle(local)
        long = self.long(middle)
        return tuple(
            projection(feature)
            for projection, feature in zip(self.project, (local, middle, long))
        )


class DownsampleQueryV1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 5, stride=2, padding=2),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 5, stride=2, padding=2),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )

    def forward(self, values):
        return self.network(values)


class ConditionEncoderV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(10, 48, 9, padding=4),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv1d(48, 48, 5, stride=2, padding=2),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv1d(48, CONDITION_CHANNELS_V1, 5, stride=2, padding=2),
            nn.GroupNorm(8, CONDITION_CHANNELS_V1),
            nn.GELU(),
        )

    def forward(self, values):
        return self.network(values)


class AxialBlockV1(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.state = nn.Sequential(
            nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.time = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                (1, 9),
                padding=(0, 4 * dilation),
                dilation=(1, dilation),
                groups=channels,
            ),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, values):
        return F.gelu(values + self.time(self.state(values)))


class TemporalDownV1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, (3, 5), stride=(1, 2), padding=(1, 2)),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
        )

    def forward(self, values):
        return self.network(values)


class TemporalUpV1(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 1),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            AxialBlockV1(output_channels, 1),
        )

    def forward(self, values, skip):
        values = F.interpolate(
            values,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.project(torch.cat([values, skip], dim=1))


class GlobalCostUNetV1(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(input_channels, BASE_CHANNELS_V1, 1),
            nn.GroupNorm(8, BASE_CHANNELS_V1),
            nn.GELU(),
        )
        self.level0 = nn.Sequential(
            AxialBlockV1(48, 1),
            AxialBlockV1(48, 2),
        )
        self.down1 = TemporalDownV1(48, 64)
        self.level1 = nn.Sequential(AxialBlockV1(64, 1), AxialBlockV1(64, 2))
        self.down2 = TemporalDownV1(64, 96)
        self.level2 = nn.Sequential(AxialBlockV1(96, 1), AxialBlockV1(96, 2))
        self.down3 = TemporalDownV1(96, 128)
        self.bottleneck = nn.Sequential(
            AxialBlockV1(128, 1),
            AxialBlockV1(128, 2),
            AxialBlockV1(128, 4),
            AxialBlockV1(128, 8),
        )
        self.up2 = TemporalUpV1(128, 96, 96)
        self.up1 = TemporalUpV1(96, 64, 64)
        self.up0 = TemporalUpV1(64, 48, 48)
        self.output = nn.Conv2d(48, 1, 1)

    def forward(self, values):
        x0 = self.level0(self.input(values))
        x1 = self.level1(self.down1(x0))
        x2 = self.level2(self.down2(x1))
        x3 = self.bottleneck(self.down3(x2))
        y2 = self.up2(x3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)
        return self.output(y0).squeeze(1)


class NativeDualReferenceWarpNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.gr_encoder = MultiScaleGREncoderV1()
        self.query_downsample = DownsampleQueryV1(SCALE_EMBEDDING_V1)
        self.condition_encoder = ConditionEncoderV1()
        self.correlation_scale = nn.Parameter(torch.full((3,), 3.0))
        # 6 correlations + 2 raw differences + 3 validities + state/progress
        # + 32 condition channels.
        self.cost_channels = 6 + 2 + 3 + 2 + CONDITION_CHANNELS_V1
        self.global_unet = GlobalCostUNetV1(self.cost_channels)

    def forward(self, horizontal, condition, typewell, heel):
        horizontal_scales = self.gr_encoder(horizontal)
        type_scales = self.gr_encoder(typewell)
        heel_scales = self.gr_encoder(heel)
        queries = [self.query_downsample(feature) for feature in horizontal_scales]
        condition_embedding = self.condition_encoder(condition)
        columns = min(
            [condition_embedding.shape[-1]] + [query.shape[-1] for query in queries]
        )
        condition_embedding = condition_embedding[..., :columns]
        state_slice = slice(REFERENCE_STATE_START, REFERENCE_STATE_START + N_STATES)
        correlations = []
        scales = F.softplus(self.correlation_scale)
        for index, (query, type_feature, heel_feature) in enumerate(
            zip(queries, type_scales, heel_scales)
        ):
            query = F.normalize(query[..., :columns], dim=1, eps=1e-5)
            type_key = F.normalize(type_feature[..., state_slice], dim=1, eps=1e-5)
            heel_key = F.normalize(heel_feature[..., state_slice], dim=1, eps=1e-5)
            correlations.extend(
                [
                    scales[index]
                    * torch.einsum("bct,bcs->bst", query, type_key)[:, None],
                    scales[index]
                    * torch.einsum("bct,bcs->bst", query, heel_key)[:, None],
                ]
            )

        h_value = horizontal[:, 0:1, ::HORIZONTAL_STRIDE][..., :columns].unsqueeze(2)
        h_valid = horizontal[:, 1:2, ::HORIZONTAL_STRIDE][..., :columns].unsqueeze(2)
        type_value = typewell[:, 0:1, state_slice].unsqueeze(-1)
        heel_value = heel[:, 0:1, state_slice].unsqueeze(-1)
        type_valid = typewell[:, 1:2, state_slice].unsqueeze(-1)
        heel_valid = heel[:, 1:2, state_slice].unsqueeze(-1)
        states = N_STATES
        h_valid = h_valid.expand(-1, -1, states, -1)
        type_valid = type_valid.expand(-1, -1, -1, columns)
        heel_valid = heel_valid.expand(-1, -1, -1, columns)
        state_coordinate = torch.linspace(
            -1.0, 1.0, states, device=horizontal.device, dtype=horizontal.dtype
        )[None, None, :, None].expand(horizontal.shape[0], -1, -1, columns)
        progress = torch.linspace(
            0.0, 1.0, columns, device=horizontal.device, dtype=horizontal.dtype
        )[None, None, None, :].expand(horizontal.shape[0], -1, states, -1)
        condition_volume = condition_embedding.unsqueeze(2).expand(-1, -1, states, -1)
        cost = torch.cat(
            correlations
            + [
                -torch.abs(h_value - type_value),
                -torch.abs(h_value - heel_value),
                h_valid,
                type_valid,
                heel_valid,
                state_coordinate,
                progress,
                condition_volume,
            ],
            dim=1,
        )
        return self.global_unet(cost)


# ---------------------------------------------------------------------------
# V1 loss, decoding and evaluation.
# ---------------------------------------------------------------------------


def _inputs_v1(record, device, mode="full", donor=None):
    horizontal = record.horizontal.copy()
    condition = record.condition.copy()
    typewell = record.typewell.copy()
    heel = record.heel.copy()
    if mode == "no_gr":
        horizontal[[0, 2, 3]] = 0.0
        typewell[[0, 2, 3]] = 0.0
        heel[[0, 2, 3]] = 0.0
    elif mode == "shuffled_typewell":
        typewell = donor.typewell.copy()
    elif mode == "shuffled_heel":
        heel = donor.heel.copy()
    return tuple(
        torch.from_numpy(array)[None].to(device)
        for array in (horizontal, condition, typewell, heel)
    )


def _loss_v1(logits, record, device, row_weight):
    columns = logits.shape[-1]
    target = torch.from_numpy(record.latent_truth_offset[:columns]).to(device)
    mask = torch.from_numpy(record.latent_eval_mask[:columns]).to(device)
    state = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    target_eval = target[mask][None]
    distribution = torch.exp(
        -0.5 * ((state - target_eval) / TARGET_SIGMA_FT_V1) ** 2
    )
    distribution = distribution / distribution.sum(dim=0, keepdim=True).clamp_min(1e-8)
    log_probability = F.log_softmax(logits[0, :, mask], dim=0)
    ce = -(distribution * log_probability).sum(dim=0).mean()
    probability = F.softmax(logits[0, :, mask], dim=0)
    expected = (probability * state).sum(dim=0)
    error = expected - target[mask]
    mse = torch.mean((error / MSE_SCALE_FT_V1) ** 2)
    expected_anchor = torch.cat([torch.zeros(1, device=device), expected])
    target_anchor = torch.cat([torch.zeros(1, device=device), target[mask]])
    slope = F.smooth_l1_loss(
        torch.diff(expected_anchor) / 2.0,
        torch.diff(target_anchor) / 2.0,
    )
    curvature = F.smooth_l1_loss(
        torch.diff(expected_anchor, n=2) / 2.0,
        torch.diff(target_anchor, n=2) / 2.0,
    )
    loss = row_weight * (
        LOSS_CE_V1 * ce
        + LOSS_MSE_V1 * mse
        + LOSS_SLOPE_V1 * slope
        + LOSS_CURVATURE_V1 * curvature
    )
    return loss, ce.detach(), mse.detach()


@torch.no_grad()
def _predict_v1(model, record, device, mode="full", donor=None):
    model.eval()
    inputs = _inputs_v1(record, device, mode=mode, donor=donor)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(*inputs)
    probability = F.softmax(logits[0].float(), dim=0)
    state = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    expected = (probability * state).sum(dim=0).cpu().numpy()
    columns = min(len(expected), len(record.latent_index))
    latent_index = record.latent_index[:columns]
    latent_eval = record.latent_eval_mask[:columns]
    support_index = latent_index[latent_eval]
    support_md = np.r_[record.md[record.anchor_index], record.md[support_index]]
    support_tvt = np.r_[
        record.anchor_tvt,
        record.anchor_tvt + expected[:columns][latent_eval],
    ]
    prediction = np.interp(record.md[record.eval_index], support_md, support_tvt)
    truth = record.truth[record.eval_index].astype(np.float64)
    return truth, prediction.astype(np.float64)


@torch.no_grad()
def _evaluate_v1(model, evaluation_records, device, mode="full", save_rows=False):
    truths, predictions, labels, bywell = [], [], [], []
    for index, record in enumerate(evaluation_records):
        donor = None
        if mode in ("shuffled_typewell", "shuffled_heel"):
            donor = evaluation_records[(index + max(1, len(evaluation_records) // 2)) % len(evaluation_records)]
        truth, prediction = _predict_v1(model, record, device, mode, donor)
        truths.append(truth)
        predictions.append(prediction)
        if save_rows:
            labels.append(np.full(len(truth), record.well, dtype="U16"))
        bywell.append(
            {
                "well": record.well,
                "rows": len(truth),
                "rmse": _rmse(truth, prediction),
                "mae": float(np.mean(np.abs(prediction - truth))),
                "mean_error": float(np.mean(prediction - truth)),
            }
        )
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    bywell = pd.DataFrame(bywell)
    metrics = {
        "rows": len(truth),
        "wells": len(evaluation_records),
        "rmse": _rmse(truth, prediction),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
        "wells_above_12ft": int((bywell["rmse"] > 12).sum()),
    }
    payload = None
    if save_rows:
        payload = {
            "well": np.concatenate(labels),
            "truth": truth.astype(np.float32),
            "prediction": prediction.astype(np.float32),
        }
    return metrics, bywell, payload


def _train_v1(bank, device):
    model = NativeDualReferenceWarpNetV1().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_V1, weight_decay=WEIGHT_DECAY_V1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_V1, eta_min=MIN_LEARNING_RATE_V1
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=GRAD_SCALER_INITIAL_SCALE_V1,
        growth_interval=500,
    )
    official_mean_rows = float(np.mean([records[well].eval_rows for well in train_wells]))
    train_subset_wells = train_wells[:: max(1, len(train_wells) // 24)][:24]
    train_subset = [records[well] for well in train_subset_wells]
    rng = np.random.default_rng(SEED_V1)
    history, best = [], None
    reports_without_improvement = 0
    started = time.time()

    for epoch in range(1, EPOCHS_V1 + 1):
        model.train()
        order = rng.permutation(len(train_wells))
        sums = {"loss": 0.0, "ce": 0.0, "mse": 0.0, "skipped": 0}
        tag_counts = {"official": 0, "pseudo_low": 0, "pseudo_high": 0}
        for batch_number, index in enumerate(order, 1):
            well = train_wells[int(index)]
            available = bank[well]
            if epoch <= 2 or rng.random() < OFFICIAL_STATE_PROBABILITY:
                tag = "official"
            else:
                pseudo = [tag for tag in ("pseudo_low", "pseudo_high") if tag in available]
                tag = pseudo[int(rng.integers(len(pseudo)))] if pseudo else "official"
            record = available[tag]
            tag_counts[tag] += 1
            row_weight = record.eval_rows / max(official_mean_rows, 1.0)
            inputs = _inputs_v1(record, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(*inputs)
                loss, ce, mse = _loss_v1(logits, record, device, row_weight)
            before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_V1)
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) < before:
                sums["skipped"] += 1
            sums["loss"] += float(loss.detach().cpu())
            sums["ce"] += float(ce.cpu())
            sums["mse"] += float(mse.cpu())
            del inputs, logits, loss
            if batch_number % 100 == 0:
                print(
                    f"V1 epoch {epoch}/{EPOCHS_V1} wells {batch_number}/{len(train_wells)}",
                    flush=True,
                )
        scheduler.step()

        should_report = epoch <= 2 or epoch % REPORT_EVERY_V1 == 0 or epoch == EPOCHS_V1
        if not should_report:
            continue
        valid_metrics, valid_bywell, _ = _evaluate_v1(model, valid_records, device)
        train_metrics, _, _ = _evaluate_v1(model, train_subset, device)
        row = {
            "epoch": epoch,
            "train_loss": sums["loss"] / len(train_wells),
            "train_ce": sums["ce"] / len(train_wells),
            "train_scaled_mse": sums["mse"] / len(train_wells),
            "train_subset_rmse": train_metrics["rmse"],
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "official_states": tag_counts["official"],
            "pseudo_low_states": tag_counts["pseudo_low"],
            "pseudo_high_states": tag_counts["pseudo_high"],
            "skipped_steps": sums["skipped"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "correlation_scales": [
                float(value)
                for value in F.softplus(model.correlation_scale).detach().cpu()
            ],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("V1 ONEFOLD EPOCH", row, flush=True)
        pd.DataFrame(history).to_csv(OUT_DIR_V1 / "training_history.csv", index=False)

        if best is None or valid_metrics["rmse"] < best["metrics"]["rmse"] - 1e-5:
            best = {
                "epoch": epoch,
                "metrics": valid_metrics,
                "bywell": valid_bywell.copy(),
                "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            torch.save(
                {
                    "experiment": EXPERIMENT_V1,
                    "epoch": epoch,
                    "metrics": valid_metrics,
                    "model": best["state"],
                    "state_offsets": STATE_OFFSETS,
                },
                MODEL_DIR_V1 / "best_fold1.pt",
            )
            reports_without_improvement = 0
        else:
            reports_without_improvement += 1

        if (
            epoch >= FAIL_FAST_EPOCH_V1
            and valid_metrics["rmse"] > FAIL_FAST_VALID_RMSE_V1
            and train_metrics["rmse"] <= FAIL_FAST_TRAIN_SUBSET_RMSE_V1
        ):
            print("V1 FAIL-FAST: train fit exists but held-out transfer is absent", flush=True)
            break
        if (
            epoch >= MIN_EPOCHS_V1
            and reports_without_improvement >= EARLY_STOPPING_REPORTS_V1
        ):
            print(f"V1 early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history), train_subset, (time.time() - started) / 60.0


# ---------------------------------------------------------------------------
# Execute V1.
# ---------------------------------------------------------------------------


STARTED_V1 = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("V1 requires CUDA")
device_v1 = torch.device("cuda")
_seed_everything(SEED_V1)

pseudo_bank_v1, pseudo_report_v1 = _prepare_pseudo_bank_v1()
probe_v1 = NativeDualReferenceWarpNetV1().to(device_v1)
probe_record_v1 = records[train_wells[0]]
with torch.no_grad(), torch.amp.autocast("cuda"):
    probe_output_v1 = probe_v1(*_inputs_v1(probe_record_v1, device_v1))
parameters_v1 = int(sum(parameter.numel() for parameter in probe_v1.parameters()))
smoke_v1 = {
    "parameters": parameters_v1,
    "output": list(probe_output_v1.shape),
    "finite": bool(torch.isfinite(probe_output_v1).all()),
    "cost_channels": probe_v1.cost_channels,
}
print("V1 MODEL SMOKE", smoke_v1, flush=True)
del probe_v1, probe_output_v1
gc.collect()
torch.cuda.empty_cache()

model_v1, best_v1, history_v1, train_subset_v1, training_minutes_v1 = _train_v1(
    pseudo_bank_v1, device_v1
)

full_metrics_v1, bywell_v1, payload_v1 = _evaluate_v1(
    model_v1, valid_records, device_v1, save_rows=True
)
bywell_v1.to_csv(OUT_DIR_V1 / "fold1_bywell.csv", index=False)
np.savez_compressed(OUT_DIR_V1 / "fold1_oof.npz", **payload_v1)

control_rows_v1 = []
for mode in ("full", "no_gr", "shuffled_typewell", "shuffled_heel"):
    metrics, _, _ = _evaluate_v1(model_v1, valid_records, device_v1, mode=mode)
    control_rows_v1.append(
        {
            "mode": mode,
            **metrics,
            "degradation_vs_full": metrics["rmse"] - full_metrics_v1["rmse"],
        }
    )
controls_v1 = pd.DataFrame(control_rows_v1)
controls_v1.to_csv(OUT_DIR_V1 / "causal_controls.csv", index=False)

architecture_pass_v1 = bool(full_metrics_v1["rmse"] <= ARCHITECTURE_GATE_FT_V1)
target_pass_v1 = bool(full_metrics_v1["rmse"] <= TARGET_GATE_FT_V1)
no_gr_degradation_v1 = float(
    controls_v1.loc[controls_v1["mode"] == "no_gr", "degradation_vs_full"].iloc[0]
)
typewell_degradation_v1 = float(
    controls_v1.loc[
        controls_v1["mode"] == "shuffled_typewell", "degradation_vs_full"
    ].iloc[0]
)
heel_degradation_v1 = float(
    controls_v1.loc[
        controls_v1["mode"] == "shuffled_heel", "degradation_vs_full"
    ].iloc[0]
)
causal_pass_v1 = bool(no_gr_degradation_v1 >= 1.0 and typewell_degradation_v1 >= 0.20)

verdict_v1 = {
    "experiment": EXPERIMENT_V1,
    "standalone": True,
    "sample_unit": "one complete well state",
    "legal_inputs_only": True,
    "train_wells": len(train_wells),
    "valid_wells": len(valid_wells),
    "parameters": parameters_v1,
    "pseudo_states_eligible_fraction": float(
        pseudo_report_v1["eligible"].fillna(False).mean()
    ),
    "best_epoch": int(best_v1["epoch"]),
    "result": full_metrics_v1,
    "causal_controls": {
        "no_gr_degradation": no_gr_degradation_v1,
        "shuffled_typewell_degradation": typewell_degradation_v1,
        "shuffled_heel_degradation": heel_degradation_v1,
    },
    "architecture_pass": architecture_pass_v1,
    "target_5p5_pass": target_pass_v1,
    "causal_pass": causal_pass_v1,
    "strict_pass": bool(architecture_pass_v1 and causal_pass_v1),
    "training_minutes": training_minutes_v1,
    "elapsed_minutes": (time.time() - STARTED_V1) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "TARGET PASS -> freeze V1 and run five folds"
        if target_pass_v1 and causal_pass_v1
        else (
            "ARCHITECTURE PASS -> inspect residual buckets before five-fold expansion"
            if architecture_pass_v1 and causal_pass_v1
            else "FAIL -> close this GR cost-volume family; the missing mechanism is not width or pseudo-PS"
        )
    ),
}
_write_json_v1(OUT_DIR_V1 / "verdict.json", verdict_v1)

print("\nV1 TRAINING HISTORY")
print(history_v1.to_string(index=False), flush=True)
print("\nV1 CAUSAL CONTROLS")
print(controls_v1.to_string(index=False), flush=True)
print("\nV1 FINAL VERDICT")
print(json.dumps(verdict_v1, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_V1, flush=True)

