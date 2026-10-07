"""Nested Fold-1 gateway for an end-to-end full-well Correlation-CRF.

This is a new standalone architecture, not a continuation of the retired
PrefixRidge/matcher-posterior route.  Each sample is one complete physical
well.  The model builds a trainable native-resolution correlation volume from
horizontal GR, typewell GR and the visible heel reference, then trains its
unary path energy jointly with a fixed-form anchored linear-chain CRF.

Protocol
--------
* Folds 3/4/5 train the epoch-selection model.
* Fold 2 selects the epoch count and applies causal controls.
* Fold 1 is untouched unless the Fold-2 gateway passes.
* The final model is then trained from scratch on Folds 2/3/4/5 for the frozen
  epoch count and applied to Fold 1 exactly once.
* Inference is one neural forward plus one deterministic Viterbi decode.  It
  performs no per-well optimisation, candidate search or parameter tuning.

Prerequisite
------------
Upload ``native_resolution_dualref_capacity_v0_cell.py`` to
``/root/autodl-tmp``.  Its validated record builder/cache is reused, but its
network and pointwise loss are not reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import types

import numpy as np
import pandas as pd

# Must be set before the first CUDA/CuBLAS operation in a fresh kernel.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Import only the validated native record definitions.
# ---------------------------------------------------------------------------


def _bootstrap_native_records_v1():
    candidates = (
        Path("/root/autodl-tmp/native_resolution_dualref_capacity_v0_cell.py"),
        Path.cwd() / "native_resolution_dualref_capacity_v0_cell.py",
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        required = {
            "Record",
            "_build_record",
            "_horizontal_paths",
            "_load_record",
            "_record_path",
            "_save_record",
            "_seed_everything",
            "_rmse",
            "STATE_OFFSETS",
            "N_STATES",
            "HORIZONTAL_STRIDE",
            "REFERENCE_STATE_START",
            "OOF_ROOT",
            "DATA_ROOT",
            "CACHE_DIR",
        }
        missing = sorted(required.difference(globals()))
        if missing:
            raise FileNotFoundError(
                "Upload native_resolution_dualref_capacity_v0_cell.py or run "
                "that cell in this kernel; missing live definitions: "
                + repr(missing)
            )
        print("native bootstrap: using validated live-kernel records", flush=True)
        return globals(), Path("live_kernel_native_record_definitions")

    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STARTED"
            for target in node.targets
        ):
            break
        definitions.append(node)
    module_name = "native_resolution_record_definitions_for_corrcrf_v1"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    namespace = module.__dict__
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace, source_path


V, NATIVE_SOURCE = _bootstrap_native_records_v1()

Record = V["Record"]
_build_record = V["_build_record"]
_horizontal_paths = V["_horizontal_paths"]
_load_record = V["_load_record"]
_record_path = V["_record_path"]
_save_record = V["_save_record"]
_seed_native = V["_seed_everything"]
_rmse_native = V["_rmse"]
STATE_OFFSETS = np.asarray(V["STATE_OFFSETS"], np.float32)
N_STATES = int(V["N_STATES"])
HORIZONTAL_STRIDE = int(V["HORIZONTAL_STRIDE"])
REFERENCE_STATE_START = int(V["REFERENCE_STATE_START"])
OOF_ROOT = Path(V["OOF_ROOT"])
DATA_ROOT = Path(V["DATA_ROOT"])
RECORD_CACHE_DIR = Path(V["CACHE_DIR"])


# ---------------------------------------------------------------------------
# Frozen experiment configuration.
# ---------------------------------------------------------------------------


EXPERIMENT = "fullwell_correlation_crf_nested_gateway_v1"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260729
REQUIRE_CUDA = True
TUNE_FOLD = 2
TEST_FOLD = 1

TUNE_EPOCHS = 16
REPORT_EPOCHS = {1, 2, 4, 6, 8, 10, 12, 14, 16}
MIN_EPOCHS_BEFORE_STOP = 8
EARLY_STOPPING_REPORTS = 4

LEARNING_RATE = 2.5e-4
MIN_LEARNING_RATE = 2.5e-5
WEIGHT_DECAY = 2e-5
GRADIENT_CLIP = 3.0
GRAD_ACCUM_WELLS = 4

EMBEDDING = 32
CONDITION_CHANNELS = 24
BASE_CHANNELS = 32
TEMPORAL_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CORRELATION_SMOOTHING = (33, 129)
TRANSITION_DELTA_CAP_FT = 2.0

STRUCTURED_COLUMN_STRIDE = 4
TARGET_SIGMA_FT = 1.5
ALIAS_OFFSETS_FT = (-48.0, -24.0, 24.0, 48.0)
ALIAS_RAMP_FT = 500.0
ALIAS_MARGIN = 0.08

LOSS_STRUCTURED = 0.45
LOSS_POINT_CE = 0.20
LOSS_PATH = 0.30
LOSS_ALIAS = 0.05

FOLD2_RMSE_GATE = 10.0
FOLD1_ARCHITECTURE_GATE = 8.0
TARGET_GATE = 5.5
NO_GR_DEGRADATION_GATE = 0.75
SHUFFLED_TYPEWELL_GATE = 0.20
SHUFFLED_HEEL_GATE = 0.15

FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)


@dataclass
class TransitionPriorV1:
    slope_centre: float
    residual_scale: float
    residual_cap: int
    edges: int
    residual_p99: float
    residual_p999: float


def _json_default_v1(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json_v1(path, payload):
    Path(path).write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=_json_default_v1,
        ),
        encoding="utf-8",
    )


def _seed_everything_v1(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _load_folds_v1(wells):
    fold_path = next((path for path in FOLD_CANDIDATES if path.exists()), None)
    if fold_path is None:
        raise FileNotFoundError(
            "Cannot find fold-clean well_metadata.csv under " + str(OOF_ROOT)
        )
    frame = pd.read_csv(fold_path, dtype={"well": str})
    well_column = next(
        (column for column in ("well", "well_id", "WELL") if column in frame),
        None,
    )
    fold_column = next(
        (column for column in ("fold", "outer_fold", "Fold") if column in frame),
        None,
    )
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Invalid fold columns: {frame.columns.tolist()}")
    frame = frame[[well_column, fold_column]].rename(
        columns={well_column: "well", fold_column: "fold"}
    )
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.drop_duplicates("well", keep="last")
    frame = frame[frame["well"].isin(set(map(str, wells)))]
    if len(frame) != len(wells):
        missing = sorted(set(map(str, wells)) - set(frame["well"]))
        raise RuntimeError(
            f"Fold metadata covers {len(frame)}/{len(wells)} wells; "
            f"missing {missing[:10]}"
        )
    if sorted(frame["fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("Expected folds 1-5")
    return frame.sort_values(["fold", "well"]).reset_index(drop=True), fold_path


def _ensure_records_v1(wells):
    RECORD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, well in enumerate(wells, 1):
        path = _record_path(well)
        try:
            valid = False
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    valid = (
                        "cache_version" in data.files
                        and int(data["cache_version"][0]) == 1
                    )
            if valid:
                record = _load_record(well)
                source = "cached"
            else:
                record = _build_record(well)
                _save_record(record)
                source = "built"
            rows.append(
                {
                    "well": well,
                    "status": "ok",
                    "source": source,
                    "rows": len(record.md),
                    "eval_rows": record.eval_rows,
                    "teacher_rmse": record.teacher_rmse,
                    "axis_coverage": record.axis_coverage,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "well": well,
                    "status": "failed",
                    "reason": repr(exc),
                }
            )
        if index % 50 == 0 or index == len(wells):
            print(f"prepare native records {index}/{len(wells)}", flush=True)
    report = pd.DataFrame(rows)
    report.to_csv(OUT_DIR / "record_build_report.csv", index=False)
    failures = report[report["status"] != "ok"]
    if len(failures):
        raise RuntimeError(f"Failed to prepare {len(failures)} records")
    return report


def _eval_latent_positions_v1(record, columns=None):
    stop = len(record.latent_index) if columns is None else min(
        int(columns), len(record.latent_index)
    )
    return np.flatnonzero(record.latent_eval_mask[:stop]).astype(np.int64)


def _latent_geometry_v1(record, positions):
    native_index = record.latent_index[np.asarray(positions, np.int64)]
    md = record.md[native_index].astype(np.float64)
    z_relative = (
        record.condition[4, native_index].astype(np.float64) * 512.0
    )
    return md, z_relative


def _fit_transition_prior_v1(records):
    slopes = []
    edge_payload = []
    for record in records:
        positions = _eval_latent_positions_v1(record)
        positions = positions[::STRUCTURED_COLUMN_STRIDE]
        if len(positions) == 0:
            continue
        if positions[-1] != _eval_latent_positions_v1(record)[-1]:
            positions = np.r_[positions, _eval_latent_positions_v1(record)[-1]]
        md, z = _latent_geometry_v1(record, positions)
        offset = record.latent_truth_offset[positions].astype(np.float64)
        anchor_md = float(record.md[record.anchor_index])
        structure = np.r_[0.0, offset + z]
        md = np.r_[anchor_md, md]
        delta_md = np.diff(md)
        delta_s = np.diff(structure)
        valid = np.isfinite(delta_s) & np.isfinite(delta_md) & (delta_md > 0)
        slopes.append(delta_s[valid] / delta_md[valid])
        edge_payload.append((delta_md[valid], delta_s[valid]))
    slopes = np.concatenate(slopes)
    slope_centre = float(np.median(slopes))
    residual = np.concatenate(
        [delta_s - slope_centre * delta_md for delta_md, delta_s in edge_payload]
    )
    absolute = np.abs(residual)
    scale = float(max(np.quantile(absolute, 0.68), 0.75))
    p99 = float(np.quantile(absolute, 0.99))
    p999 = float(np.quantile(absolute, 0.999))
    cap = int(np.clip(math.ceil(p999 + 2.0), 16, 24))
    return TransitionPriorV1(
        slope_centre=slope_centre,
        residual_scale=scale,
        residual_cap=cap,
        edges=len(residual),
        residual_p99=p99,
        residual_p999=p999,
    )


def _transition_coverage_v1(records, prior):
    residuals = []
    for record in records:
        all_positions = _eval_latent_positions_v1(record)
        positions = all_positions[::STRUCTURED_COLUMN_STRIDE]
        if len(positions) == 0:
            continue
        if positions[-1] != all_positions[-1]:
            positions = np.r_[positions, all_positions[-1]]
        md, z = _latent_geometry_v1(record, positions)
        offset = record.latent_truth_offset[positions].astype(np.float64)
        structure = np.r_[0.0, offset + z]
        md = np.r_[float(record.md[record.anchor_index]), md]
        residuals.append(
            np.diff(structure) - float(prior.slope_centre) * np.diff(md)
        )
    residual = np.concatenate(residuals)
    absolute = np.abs(residual)
    return {
        "edges": len(residual),
        "coverage": float(np.mean(absolute <= float(prior.residual_cap))),
        "p99": float(np.quantile(absolute, 0.99)),
        "p999": float(np.quantile(absolute, 0.999)),
        "max": float(np.max(absolute)),
    }


# ---------------------------------------------------------------------------
# Full-well trainable correlation pyramid.
# ---------------------------------------------------------------------------


class Residual1DV1(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                5,
                padding=2 * dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv1d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(4, channels),
        )

    def forward(self, values):
        return F.gelu(values + self.network(values))


class SharedGREncoderV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(4, EMBEDDING, 11, padding=5),
            nn.GroupNorm(4, EMBEDDING),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            Residual1DV1(EMBEDDING, 1),
            Residual1DV1(EMBEDDING, 2),
            Residual1DV1(EMBEDDING, 4),
            Residual1DV1(EMBEDDING, 8),
            Residual1DV1(EMBEDDING, 16),
        )

    def forward(self, values):
        return self.blocks(self.input(values))


class DownsampleV1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 7, stride=2, padding=3),
            nn.GroupNorm(4, output_channels),
            nn.GELU(),
            nn.Conv1d(output_channels, output_channels, 7, stride=2, padding=3),
            nn.GroupNorm(4, output_channels),
            nn.GELU(),
        )

    def forward(self, values):
        return self.network(values)


class AxialEnergyBlockV1(nn.Module):
    def __init__(self, channels, temporal_dilation):
        super().__init__()
        self.state = nn.Sequential(
            nn.Conv2d(channels, channels, (9, 1), padding=(4, 0), groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )
        self.time = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                (1, 5),
                padding=(0, 2 * temporal_dilation),
                dilation=(1, temporal_dilation),
                groups=channels,
            ),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(4, channels),
        )

    def forward(self, values):
        return F.gelu(values + self.time(self.state(values)))


class FullWellCorrelationCRFV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.gr_encoder = SharedGREncoderV1()
        self.query_downsample = DownsampleV1(EMBEDDING, EMBEDDING)
        self.condition_downsample = DownsampleV1(10, CONDITION_CHANNELS)

        # learned correlations 2, raw differences 2, validity 3, coordinates 2,
        # smoothed correlation pyramid 4, cumulative evidence 2, condition 24.
        input_channels = 2 + 2 + 3 + 2 + 4 + 2 + CONDITION_CHANNELS
        self.cost_input = nn.Sequential(
            nn.Conv2d(input_channels, BASE_CHANNELS, 1),
            nn.GroupNorm(4, BASE_CHANNELS),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                AxialEnergyBlockV1(BASE_CHANNELS, dilation)
                for dilation in TEMPORAL_DILATIONS
            ]
        )
        self.unary_head = nn.Conv2d(BASE_CHANNELS, 1, 1)
        self.transition_head = nn.Sequential(
            nn.Conv1d(EMBEDDING + CONDITION_CHANNELS, 32, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, 1),
        )
        self.correlation_scale = nn.Parameter(torch.tensor(3.0))

    @staticmethod
    def _smooth(correlation, kernel):
        batch, states, columns = correlation.shape
        values = correlation.reshape(batch * states, 1, columns)
        values = F.avg_pool1d(values, kernel, stride=1, padding=kernel // 2)
        return values.reshape(batch, states, columns)

    def forward(self, horizontal, condition, typewell, heel):
        horizontal_embedding = self.gr_encoder(horizontal)
        query = self.query_downsample(horizontal_embedding)
        condition_embedding = self.condition_downsample(condition)
        columns = min(query.shape[-1], condition_embedding.shape[-1])
        query = query[..., :columns]
        condition_embedding = condition_embedding[..., :columns]

        type_embedding = self.gr_encoder(typewell)
        heel_embedding = self.gr_encoder(heel)
        state_slice = slice(REFERENCE_STATE_START, REFERENCE_STATE_START + N_STATES)
        type_key = type_embedding[..., state_slice]
        heel_key = heel_embedding[..., state_slice]

        query_norm = F.normalize(query, dim=1, eps=1e-5)
        type_norm = F.normalize(type_key, dim=1, eps=1e-5)
        heel_norm = F.normalize(heel_key, dim=1, eps=1e-5)
        scale = F.softplus(self.correlation_scale)
        type_corr = scale * torch.einsum("bct,bcs->bst", query_norm, type_norm)
        heel_corr = scale * torch.einsum("bct,bcs->bst", query_norm, heel_norm)

        horizontal_value = horizontal[:, 0:1, ::HORIZONTAL_STRIDE][..., :columns]
        horizontal_valid = horizontal[:, 1:2, ::HORIZONTAL_STRIDE][..., :columns]
        type_value = typewell[:, 0:1, state_slice]
        heel_value = heel[:, 0:1, state_slice]
        type_valid = typewell[:, 1:2, state_slice]
        heel_valid = heel[:, 1:2, state_slice]

        horizontal_value = horizontal_value.unsqueeze(2)
        type_difference = -torch.abs(horizontal_value - type_value.unsqueeze(-1))
        heel_difference = -torch.abs(horizontal_value - heel_value.unsqueeze(-1))
        states = N_STATES
        h_valid = horizontal_valid.unsqueeze(2).expand(-1, -1, states, -1)
        type_valid_volume = type_valid.unsqueeze(-1).expand(-1, -1, -1, columns)
        heel_valid_volume = heel_valid.unsqueeze(-1).expand(-1, -1, -1, columns)

        state_coordinate = torch.linspace(
            -1.0, 1.0, states, device=query.device, dtype=query.dtype
        )[None, None, :, None].expand(query.shape[0], -1, -1, columns)
        progress = torch.linspace(
            0.0, 1.0, columns, device=query.device, dtype=query.dtype
        )[None, None, None, :].expand(query.shape[0], -1, states, -1)
        condition_volume = condition_embedding.unsqueeze(2).expand(
            -1, -1, states, -1
        )

        smooth_channels = []
        for kernel in CORRELATION_SMOOTHING:
            smooth_channels.extend(
                [
                    self._smooth(type_corr, kernel)[:, None],
                    self._smooth(heel_corr, kernel)[:, None],
                ]
            )
        denominator = torch.arange(
            1, columns + 1, device=query.device, dtype=query.dtype
        )[None, None, :]
        cumulative_type = (torch.cumsum(type_corr, dim=-1) / denominator)[:, None]
        cumulative_heel = (torch.cumsum(heel_corr, dim=-1) / denominator)[:, None]

        cost = torch.cat(
            [
                type_corr[:, None],
                heel_corr[:, None],
                type_difference,
                heel_difference,
                h_valid,
                type_valid_volume,
                heel_valid_volume,
                state_coordinate,
                progress,
                *smooth_channels,
                cumulative_type,
                cumulative_heel,
                condition_volume,
            ],
            dim=1,
        )
        hidden = self.cost_input(cost)
        for block in self.blocks:
            if self.training and hidden.requires_grad:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        unary = self.unary_head(hidden).squeeze(1)
        transition_delta = TRANSITION_DELTA_CAP_FT * torch.tanh(
            self.transition_head(
                torch.cat([query, condition_embedding], dim=1)
            ).squeeze(1)
        )
        return unary, transition_delta


# ---------------------------------------------------------------------------
# Structured CRF loss and deterministic decode.
# ---------------------------------------------------------------------------


def _tensor_inputs_v1(record, device, mode="full", donor=None):
    horizontal = record.horizontal.copy()
    condition = record.condition.copy()
    typewell = record.typewell.copy()
    heel = record.heel.copy()
    if mode == "no_gr":
        horizontal[[0, 2, 3]] = 0.0
        typewell[[0, 2, 3]] = 0.0
        heel[[0, 2, 3]] = 0.0
    elif mode == "shuffled_typewell":
        if donor is None:
            raise ValueError("shuffled_typewell requires a donor")
        typewell = donor.typewell.copy()
    elif mode == "shuffled_heel":
        if donor is None:
            raise ValueError("shuffled_heel requires a donor")
        heel = donor.heel.copy()
    elif mode != "full":
        raise ValueError(mode)
    return (
        torch.from_numpy(horizontal)[None].to(device),
        torch.from_numpy(condition)[None].to(device),
        torch.from_numpy(typewell)[None].to(device),
        torch.from_numpy(heel)[None].to(device),
    )


def _path_score_v1(
    unary,
    transition_delta,
    state_index,
    node_md,
    node_z,
    prior,
):
    node = torch.arange(len(state_index), device=unary.device)
    score = unary[state_index, node].sum()
    previous_state = torch.cat(
        [
            torch.tensor(
                [int(np.argmin(np.abs(STATE_OFFSETS)))],
                device=unary.device,
                dtype=torch.long,
            ),
            state_index[:-1],
        ]
    )
    state_values = torch.from_numpy(STATE_OFFSETS).to(unary.device)
    shift = state_values[state_index] - state_values[previous_state]
    previous_md = torch.cat([node_md.new_zeros(1), node_md[:-1]])
    previous_z = torch.cat([node_z.new_zeros(1), node_z[:-1]])
    # node_md is already relative to the anchor.
    delta_md = node_md - previous_md
    delta_z = node_z - previous_z
    residual = (
        shift
        + delta_z
        - float(prior.slope_centre) * delta_md
        - transition_delta
    )
    score = score - ((residual / float(prior.residual_scale)) ** 2).sum()
    return score


def _log_partition_v1(
    unary,
    transition_delta,
    node_md_np,
    node_z_np,
    prior,
):
    states = unary.shape[0]
    zero_state = int(np.argmin(np.abs(STATE_OFFSETS)))
    alpha = unary.new_full((states,), -1e9)
    alpha[zero_state] = 0.0
    previous_md = 0.0
    previous_z = 0.0
    state_axis = torch.arange(states, device=unary.device)

    for column in range(unary.shape[1]):
        current_md = float(node_md_np[column])
        current_z = float(node_z_np[column])
        delta_md = current_md - previous_md
        delta_z = current_z - previous_z
        expected = float(prior.slope_centre) * delta_md - delta_z
        centre_shift = int(round(expected))
        candidates = []
        for shift in range(
            centre_shift - int(prior.residual_cap),
            centre_shift + int(prior.residual_cap) + 1,
        ):
            if abs(shift) >= states:
                continue
            candidate = unary.new_full((states,), -1e9)
            if shift >= 0:
                source = slice(0, states - shift)
                target = slice(shift, states)
            else:
                source = slice(-shift, states)
                target = slice(0, states + shift)
            residual = (
                float(shift)
                + delta_z
                - float(prior.slope_centre) * delta_md
                - transition_delta[column]
            )
            candidate[target] = alpha[source] - (
                residual / float(prior.residual_scale)
            ) ** 2
            candidates.append(candidate)
        stacked = torch.stack(candidates, dim=0)
        alpha = unary[:, column] + torch.logsumexp(stacked, dim=0)
        previous_md = current_md
        previous_z = current_z
    return torch.logsumexp(alpha, dim=0)


def _structured_nodes_v1(record, columns):
    all_positions = _eval_latent_positions_v1(record, columns)
    positions = all_positions[::STRUCTURED_COLUMN_STRIDE]
    if len(positions) == 0:
        raise RuntimeError(f"{record.well}: no structured columns")
    if positions[-1] != all_positions[-1]:
        positions = np.r_[positions, all_positions[-1]]
    md, z = _latent_geometry_v1(record, positions)
    md = md - float(record.md[record.anchor_index])
    target = record.latent_truth_offset[positions].astype(np.float32)
    return positions, md.astype(np.float32), z.astype(np.float32), target


def _loss_v1(unary, transition_delta, record, prior, row_weight):
    unary = unary[0].float()
    transition_delta = transition_delta[0].float()
    columns = min(unary.shape[-1], len(record.latent_index))
    unary = unary[:, :columns]
    transition_delta = transition_delta[:columns]
    eval_positions = _eval_latent_positions_v1(record, columns)
    target_all = torch.from_numpy(
        record.latent_truth_offset[eval_positions].astype(np.float32)
    ).to(unary.device)
    state = torch.from_numpy(STATE_OFFSETS).to(unary.device)[:, None]

    target_distribution = torch.exp(
        -0.5 * ((state - target_all[None, :]) / TARGET_SIGMA_FT) ** 2
    )
    target_distribution = target_distribution / target_distribution.sum(
        dim=0, keepdim=True
    ).clamp_min(1e-8)
    point_log_probability = F.log_softmax(unary[:, eval_positions], dim=0)
    point_ce = -(target_distribution * point_log_probability).sum(dim=0).mean()
    point_probability = F.softmax(unary[:, eval_positions], dim=0)
    expected = (point_probability * state).sum(dim=0)
    path_loss = torch.mean(((expected - target_all) / 8.0) ** 2)

    positions, node_md_np, node_z_np, target_np = _structured_nodes_v1(
        record, columns
    )
    structured_unary = unary[:, positions]
    structured_delta = transition_delta[positions]
    node_md = torch.from_numpy(node_md_np).to(unary.device)
    node_z = torch.from_numpy(node_z_np).to(unary.device)
    target_index = torch.from_numpy(
        np.clip(
            np.rint(target_np - float(STATE_OFFSETS[0])).astype(np.int64),
            0,
            N_STATES - 1,
        )
    ).to(unary.device)
    log_partition = _log_partition_v1(
        structured_unary,
        structured_delta,
        node_md_np,
        node_z_np,
        prior,
    )
    truth_score = _path_score_v1(
        structured_unary,
        structured_delta,
        target_index,
        node_md,
        node_z,
        prior,
    )
    structured_nll = (log_partition - truth_score) / len(positions)

    distance = node_md.clamp_min(0.0)
    ramp = torch.clamp(distance / ALIAS_RAMP_FT, 0.0, 1.0)
    alias_losses = []
    for alias_offset in ALIAS_OFFSETS_FT:
        alias_target = torch.from_numpy(target_np).to(unary.device) + (
            float(alias_offset) * ramp
        )
        alias_index = torch.clamp(
            torch.round(alias_target - float(STATE_OFFSETS[0])).long(),
            0,
            N_STATES - 1,
        )
        alias_score = _path_score_v1(
            structured_unary,
            structured_delta,
            alias_index,
            node_md,
            node_z,
            prior,
        )
        alias_losses.append(
            F.relu(ALIAS_MARGIN + alias_score / len(positions) - truth_score / len(positions))
        )
    alias_loss = torch.stack(alias_losses).mean()

    loss = float(row_weight) * (
        LOSS_STRUCTURED * structured_nll
        + LOSS_POINT_CE * point_ce
        + LOSS_PATH * path_loss
        + LOSS_ALIAS * alias_loss
    )
    diagnostics = {
        "loss": loss.detach(),
        "structured_nll": structured_nll.detach(),
        "point_ce": point_ce.detach(),
        "path_loss": path_loss.detach(),
        "alias_loss": alias_loss.detach(),
    }
    return loss, diagnostics


def _viterbi_numpy_v1(unary, transition_delta, record, prior):
    columns = min(unary.shape[-1], len(record.latent_index))
    eval_positions = _eval_latent_positions_v1(record, columns)
    emission = unary[:, eval_positions].astype(np.float64)
    delta_prediction = transition_delta[eval_positions].astype(np.float64)
    node_md, node_z = _latent_geometry_v1(record, eval_positions)
    node_md = node_md - float(record.md[record.anchor_index])

    states, sequence = emission.shape
    zero_state = int(np.argmin(np.abs(STATE_OFFSETS)))
    dp = np.full(states, -np.inf, np.float64)
    dp[zero_state] = 0.0
    back = np.full((sequence, states), -1, np.int16)
    previous_md = 0.0
    previous_z = 0.0
    state_index = np.arange(states)

    for column in range(sequence):
        delta_md = float(node_md[column] - previous_md)
        delta_z = float(node_z[column] - previous_z)
        expected = float(prior.slope_centre) * delta_md - delta_z
        centre_shift = int(round(expected))
        new = np.full(states, -np.inf, np.float64)
        new_back = np.full(states, -1, np.int16)
        for shift in range(
            centre_shift - int(prior.residual_cap),
            centre_shift + int(prior.residual_cap) + 1,
        ):
            if abs(shift) >= states:
                continue
            if shift >= 0:
                source = slice(0, states - shift)
                target = slice(shift, states)
            else:
                source = slice(-shift, states)
                target = slice(0, states + shift)
            residual = (
                float(shift)
                + delta_z
                - float(prior.slope_centre) * delta_md
                - float(delta_prediction[column])
            )
            candidate = dp[source] - (
                residual / float(prior.residual_scale)
            ) ** 2 + emission[target, column]
            better = candidate > new[target]
            if np.any(better):
                target_values = state_index[target][better]
                source_values = target_values - shift
                new[target_values] = candidate[better]
                new_back[target_values] = source_values.astype(np.int16)
        if not np.isfinite(new).any():
            return None
        dp = new
        back[column] = new_back
        previous_md = float(node_md[column])
        previous_z = float(node_z[column])

    path = np.empty(sequence, np.int16)
    path[-1] = int(np.argmax(dp))
    for column in range(sequence - 1, 0, -1):
        previous = int(back[column, path[column]])
        if previous < 0:
            return None
        path[column - 1] = previous
    return eval_positions, STATE_OFFSETS[path].astype(np.float64)


@torch.no_grad()
def _predict_v1(model, record, prior, device, mode="full", donor=None):
    model.eval()
    inputs = _tensor_inputs_v1(record, device, mode=mode, donor=donor)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        unary, transition_delta = model(*inputs)
    unary_np = unary[0].float().cpu().numpy()
    transition_np = transition_delta[0].float().cpu().numpy()
    decoded = _viterbi_numpy_v1(unary_np, transition_np, record, prior)
    if decoded is None:
        return None
    eval_positions, offset = decoded
    latent_native = record.latent_index[eval_positions]
    support_md = np.r_[record.md[record.anchor_index], record.md[latent_native]]
    support_tvt = np.r_[record.anchor_tvt, record.anchor_tvt + offset]
    prediction = np.interp(
        record.md[record.eval_index], support_md, support_tvt
    ).astype(np.float64)
    truth = record.truth[record.eval_index].astype(np.float64)
    distance = (
        record.md[record.eval_index].astype(np.float64)
        - float(record.md[record.anchor_index])
    )
    return truth, prediction, distance


def _subset_rmse_v1(truth, prediction, mask):
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((truth[mask] - prediction[mask]) ** 2)))


@torch.no_grad()
def _evaluate_v1(
    model,
    records,
    prior,
    device,
    mode="full",
    return_rows=False,
):
    truths = []
    predictions = []
    distances = []
    labels = []
    bywell_rows = []
    solve_failures = 0
    for index, record in enumerate(records):
        donor = None
        if mode in ("shuffled_typewell", "shuffled_heel"):
            donor = records[(index + max(1, len(records) // 2)) % len(records)]
        result = _predict_v1(
            model,
            record,
            prior,
            device,
            mode=mode,
            donor=donor,
        )
        if result is None:
            solve_failures += 1
            continue
        truth, prediction, distance = result
        error = truth - prediction
        truths.append(truth)
        predictions.append(prediction)
        distances.append(distance)
        labels.append(np.full(len(truth), record.well, dtype="U16"))
        bywell_rows.append(
            {
                "well": record.well,
                "rows": len(truth),
                "sse": float(np.sum(error**2)),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "near_rmse": _subset_rmse_v1(truth, prediction, distance <= 500),
                "far_rmse": _subset_rmse_v1(truth, prediction, distance > 2000),
            }
        )
    if solve_failures:
        raise RuntimeError(f"Viterbi solve failures: {solve_failures}")
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    distance = np.concatenate(distances)
    well = np.concatenate(labels)
    error = truth - prediction
    bywell = pd.DataFrame(bywell_rows)
    worst_count = max(1, int(math.ceil(0.20 * len(bywell))))
    worst = bywell.nlargest(worst_count, "rmse")
    worst_mask = np.isin(well, worst["well"].to_numpy())
    metrics = {
        "rows": len(truth),
        "wells": len(bywell),
        "exact_sse": float(np.sum(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
        "worst20_wells": worst_count,
        "worst20_rmse": _subset_rmse_v1(truth, prediction, worst_mask),
        "worst20_sse_share": float(np.sum(error[worst_mask] ** 2) / np.sum(error**2)),
        "near_0_500_rmse": _subset_rmse_v1(truth, prediction, distance <= 500),
        "mid_500_1000_rmse": _subset_rmse_v1(
            truth, prediction, (distance > 500) & (distance <= 1000)
        ),
        "mid_1000_2000_rmse": _subset_rmse_v1(
            truth, prediction, (distance > 1000) & (distance <= 2000)
        ),
        "far_gt2000_rmse": _subset_rmse_v1(truth, prediction, distance > 2000),
        "anchor_abs_error": 0.0,
        "solve_failures": solve_failures,
    }
    payload = None
    if return_rows:
        payload = {
            "well": well,
            "truth": truth.astype(np.float32),
            "prediction": prediction.astype(np.float32),
            "distance": distance.astype(np.float32),
        }
    return metrics, bywell, payload


def _anchor_metrics_v1(records):
    truth_parts = []
    prediction_parts = []
    distance_parts = []
    well_parts = []
    for record in records:
        truth = record.truth[record.eval_index].astype(np.float64)
        prediction = np.full(len(truth), record.anchor_tvt, np.float64)
        distance = (
            record.md[record.eval_index].astype(np.float64)
            - float(record.md[record.anchor_index])
        )
        truth_parts.append(truth)
        prediction_parts.append(prediction)
        distance_parts.append(distance)
        well_parts.append(np.full(len(truth), record.well, dtype="U16"))
    truth = np.concatenate(truth_parts)
    prediction = np.concatenate(prediction_parts)
    distance = np.concatenate(distance_parts)
    well = np.concatenate(well_parts)
    error = truth - prediction
    bywell = []
    for label in np.unique(well):
        mask = well == label
        bywell.append(float(np.sqrt(np.mean(error[mask] ** 2))))
    return {
        "rows": len(truth),
        "exact_sse": float(np.sum(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
        "near_0_500_rmse": _subset_rmse_v1(truth, prediction, distance <= 500),
        "far_gt2000_rmse": _subset_rmse_v1(truth, prediction, distance > 2000),
        "anchor_abs_error": 0.0,
    }


# ---------------------------------------------------------------------------
# Nested training.
# ---------------------------------------------------------------------------


def _train_phase_v1(
    train_records,
    prior,
    device,
    epochs,
    seed,
    label,
    valid_records=None,
):
    _seed_everything_v1(seed)
    model = FullWellCorrelationCRFV1().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=MIN_LEARNING_RATE,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=1024.0,
        growth_interval=500,
    )
    mean_rows = float(np.mean([record.eval_rows for record in train_records]))
    row_weight = {
        record.well: record.eval_rows / max(mean_rows, 1.0)
        for record in train_records
    }
    rng = np.random.default_rng(seed)
    history = []
    best = None
    stale_reports = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_records))
        optimizer.zero_grad(set_to_none=True)
        sums = {
            "loss": 0.0,
            "structured_nll": 0.0,
            "point_ce": 0.0,
            "path_loss": 0.0,
            "alias_loss": 0.0,
        }
        successful_steps = 0
        skipped_steps = 0
        for sequence_index, record_index in enumerate(order, 1):
            record = train_records[int(record_index)]
            inputs = _tensor_inputs_v1(record, device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                unary, transition_delta = model(*inputs)
            loss, diagnostics = _loss_v1(
                unary,
                transition_delta,
                record,
                prior,
                row_weight[record.well],
            )
            scaled_loss = loss / GRAD_ACCUM_WELLS
            scale_before = float(scaler.get_scale())
            scaler.scale(scaled_loss).backward()
            should_step = (
                sequence_index % GRAD_ACCUM_WELLS == 0
                or sequence_index == len(order)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if float(scaler.get_scale()) < scale_before:
                    skipped_steps += 1
                else:
                    successful_steps += 1
            for key in sums:
                sums[key] += float(diagnostics[key].cpu())
            del inputs, unary, transition_delta, loss, scaled_loss
            if sequence_index % 100 == 0:
                print(
                    f"{label} epoch {epoch}/{epochs} wells "
                    f"{sequence_index}/{len(train_records)}",
                    flush=True,
                )
        scheduler.step()

        row = {
            "phase": label,
            "epoch": epoch,
            **{f"train_{key}": value / len(train_records) for key, value in sums.items()},
            "optimizer_steps": successful_steps,
            "skipped_steps": skipped_steps,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "correlation_scale": float(
                F.softplus(model.correlation_scale).detach().cpu()
            ),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        should_report = valid_records is not None and (
            epoch in REPORT_EPOCHS or epoch == epochs
        )
        if should_report:
            metrics, bywell, _ = _evaluate_v1(
                model,
                valid_records,
                prior,
                device,
                mode="full",
                return_rows=False,
            )
            row.update({f"valid_{key}": value for key, value in metrics.items()})
            print("CORR-CRF EPOCH", row, flush=True)
            if best is None or metrics["rmse"] < best["metrics"]["rmse"] - 1e-5:
                best = {
                    "epoch": epoch,
                    "metrics": metrics,
                    "bywell": bywell.copy(),
                    "model": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                }
                stale_reports = 0
            else:
                stale_reports += 1
        history.append(row)
        pd.DataFrame(history).to_csv(
            OUT_DIR / f"{label}_training_history.csv", index=False
        )
        if (
            valid_records is not None
            and epoch >= MIN_EPOCHS_BEFORE_STOP
            and stale_reports >= EARLY_STOPPING_REPORTS
        ):
            print(f"{label} early stopping at epoch {epoch}", flush=True)
            break

    if valid_records is None:
        best = {
            "epoch": epochs,
            "metrics": None,
            "model": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
        }
    elif best is None:
        raise RuntimeError(f"{label}: no validation checkpoint")
    model.load_state_dict(best["model"])
    return model, best, pd.DataFrame(history), (time.time() - started) / 60.0


def _controls_v1(model, records, prior, device):
    rows = []
    full_metrics = None
    for mode in ("full", "no_gr", "shuffled_typewell", "shuffled_heel"):
        metrics, _, _ = _evaluate_v1(
            model,
            records,
            prior,
            device,
            mode=mode,
            return_rows=False,
        )
        if full_metrics is None:
            full_metrics = metrics
        rows.append(
            {
                "mode": mode,
                **metrics,
                "degradation_vs_full": metrics["rmse"] - full_metrics["rmse"],
            }
        )
    return pd.DataFrame(rows)


def _control_value_v1(frame, mode):
    return float(
        frame.loc[frame["mode"].eq(mode), "degradation_vs_full"].iloc[0]
    )


def _bitwise_replay_v1(model, records, prior, device):
    first_metrics, _, first = _evaluate_v1(
        model, records, prior, device, return_rows=True
    )
    second_metrics, _, second = _evaluate_v1(
        model, records, prior, device, return_rows=True
    )
    same_well = np.array_equal(first["well"], second["well"])
    same_truth = np.array_equal(first["truth"], second["truth"])
    same_prediction = np.array_equal(first["prediction"], second["prediction"])
    max_difference = float(
        np.max(
            np.abs(
                first["prediction"].astype(np.float64)
                - second["prediction"].astype(np.float64)
            )
        )
    )
    return {
        "bitwise_equal": bool(same_well and same_truth and same_prediction),
        "well_equal": bool(same_well),
        "truth_equal": bool(same_truth),
        "prediction_equal": bool(same_prediction),
        "max_abs_prediction_difference": max_difference,
        "rows": first_metrics["rows"],
    }, first_metrics, first


# ---------------------------------------------------------------------------
# Execute the nested gateway.
# ---------------------------------------------------------------------------


STARTED = time.time()
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("This experiment requires a CUDA GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_seed_everything_v1(SEED)

wells = [path.name.split("__", 1)[0] for path in _horizontal_paths()]
folds, fold_path = _load_folds_v1(wells)
build_report = _ensure_records_v1(wells)
records = {well: _load_record(well) for well in wells}

core_wells = folds.loc[folds["fold"].isin([3, 4, 5]), "well"].tolist()
tune_wells = folds.loc[folds["fold"].eq(TUNE_FOLD), "well"].tolist()
outer_wells = folds.loc[folds["fold"].isin([2, 3, 4, 5]), "well"].tolist()
test_wells = folds.loc[folds["fold"].eq(TEST_FOLD), "well"].tolist()
core_records = [records[well] for well in core_wells]
tune_records = [records[well] for well in tune_wells]
outer_records = [records[well] for well in outer_wells]
test_records = [records[well] for well in test_wells]
if (len(outer_records), len(test_records)) != (618, 155):
    raise RuntimeError(
        f"Unexpected outer/test sizes: {len(outer_records)}/{len(test_records)}"
    )

core_prior = _fit_transition_prior_v1(core_records)
outer_prior = _fit_transition_prior_v1(outer_records)
core_transition_coverage = _transition_coverage_v1(tune_records, core_prior)
outer_transition_coverage = _transition_coverage_v1(outer_records, outer_prior)
outer_build_report = build_report[build_report["well"].isin(set(outer_wells))]
guards = {
    "experiment": EXPERIMENT,
    "native_source": str(NATIVE_SOURCE),
    "record_cache": str(RECORD_CACHE_DIR),
    "fold_source": str(fold_path),
    "core_train_folds": [3, 4, 5],
    "epoch_selection_fold": 2,
    "outer_train_folds": [2, 3, 4, 5],
    "untouched_test_fold": 1,
    "core_wells": len(core_records),
    "tune_wells": len(tune_records),
    "outer_wells": len(outer_records),
    "test_wells": len(test_records),
    "test_rows": int(sum(record.eval_rows for record in test_records)),
    "teacher_rmse_max_outer_train": float(outer_build_report["teacher_rmse"].max()),
    "axis_coverage_min_outer_train": float(outer_build_report["axis_coverage"].min()),
    "fold2_transition_coverage": core_transition_coverage,
    "outer_train_transition_coverage": outer_transition_coverage,
    "sample_unit": "one complete physical well",
    "state_axis": "TVT - visible-prefix anchor TVT, +/-128 ft at 1-ft spacing",
    "normalizer": "per-well visible-prefix robust GR calibration from native record builder",
    "test_optimizer_steps": 0,
    "test_candidate_searches": 0,
    "decoder": "one deterministic anchored Viterbi",
    "condition_number": "not applicable; no linear solve",
}
print("CORR-CRF CONFIG", json.dumps(guards, indent=2), flush=True)
print("CORE TRANSITION PRIOR", core_prior, flush=True)
print("OUTER TRANSITION PRIOR", outer_prior, flush=True)
if (
    guards["teacher_rmse_max_outer_train"] > 0.35
    or guards["axis_coverage_min_outer_train"] < 0.999
    or core_transition_coverage["coverage"] < 0.999
    or outer_transition_coverage["coverage"] < 0.999
):
    raise RuntimeError(f"Representation guard failed: {guards}")

probe_model = FullWellCorrelationCRFV1().to(device)
probe_inputs = _tensor_inputs_v1(core_records[0], device)
with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
    probe_unary, probe_transition = probe_model(*probe_inputs)
smoke = {
    "parameters": int(sum(parameter.numel() for parameter in probe_model.parameters())),
    "unary_shape": list(probe_unary.shape),
    "transition_shape": list(probe_transition.shape),
    "finite": bool(
        torch.isfinite(probe_unary).all()
        and torch.isfinite(probe_transition).all()
    ),
}
del probe_model, probe_inputs, probe_unary, probe_transition
gc.collect()
torch.cuda.empty_cache()
print("MODEL SMOKE", smoke, flush=True)
if not smoke["finite"]:
    raise RuntimeError(f"Model smoke failed: {smoke}")

fold2_anchor = _anchor_metrics_v1(tune_records)
print("FOLD2 ANCHOR", fold2_anchor, flush=True)
core_model, core_best, core_history, core_minutes = _train_phase_v1(
    core_records,
    core_prior,
    device,
    TUNE_EPOCHS,
    SEED,
    "core345",
    valid_records=tune_records,
)
core_model.load_state_dict(core_best["model"])
fold2_controls = _controls_v1(
    core_model, tune_records, core_prior, device
)
fold2_controls.to_csv(OUT_DIR / "fold2_causal_controls.csv", index=False)
fold2_no_gr = _control_value_v1(fold2_controls, "no_gr")
fold2_typewell = _control_value_v1(fold2_controls, "shuffled_typewell")
fold2_heel = _control_value_v1(fold2_controls, "shuffled_heel")
fold2_pass = bool(
    core_best["metrics"]["rmse"] <= FOLD2_RMSE_GATE
    and fold2_no_gr >= NO_GR_DEGRADATION_GATE
    and fold2_typewell >= SHUFFLED_TYPEWELL_GATE
    and fold2_heel >= SHUFFLED_HEEL_GATE
)
selected_epoch = int(core_best["epoch"])

fold2_verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_fold2_gateway",
    "guards": guards,
    "smoke": smoke,
    "core_transition_prior": core_prior.__dict__,
    "fold2_anchor": fold2_anchor,
    "selected_epoch": selected_epoch,
    "fold2_result": core_best["metrics"],
    "fold2_controls": fold2_controls.to_dict(orient="records"),
    "gates": {
        "rmse_le": FOLD2_RMSE_GATE,
        "no_gr_degradation_ge": NO_GR_DEGRADATION_GATE,
        "shuffled_typewell_degradation_ge": SHUFFLED_TYPEWELL_GATE,
        "shuffled_heel_degradation_ge": SHUFFLED_HEEL_GATE,
    },
    "strict_pass": fold2_pass,
    "training_minutes": core_minutes,
    "next": (
        "PASS -> train Folds 2-5 for the frozen epoch count and apply Fold1 once"
        if fold2_pass
        else "FAIL -> stop before Fold1; close Correlation-CRF v1"
    ),
}
_write_json_v1(OUT_DIR / "fold2_gateway_verdict.json", fold2_verdict)
print("\nFOLD2 CONTROLS")
print(fold2_controls.to_string(index=False), flush=True)
print("\nFOLD2 VERDICT")
print(json.dumps(fold2_verdict, indent=2, default=_json_default_v1), flush=True)

if not fold2_pass:
    torch.save(
        {
            "experiment": EXPERIMENT,
            "stage": "fold2_failed",
            "epoch": selected_epoch,
            "metrics": core_best["metrics"],
            "model": core_best["model"],
            "prior": core_prior.__dict__,
            "state_offsets": STATE_OFFSETS,
        },
        MODEL_DIR / "core345_fold2_failed.pt",
    )
    raise RuntimeError(json.dumps(fold2_verdict, indent=2, default=_json_default_v1))

del core_model
gc.collect()
torch.cuda.empty_cache()

print(
    f"training outer2345 for frozen epoch count={selected_epoch}",
    flush=True,
)
outer_model, outer_best, outer_history, outer_minutes = _train_phase_v1(
    outer_records,
    outer_prior,
    device,
    selected_epoch,
    SEED + 101,
    "outer2345",
    valid_records=None,
)
torch.save(
    {
        "experiment": EXPERIMENT,
        "epoch": selected_epoch,
        "model": outer_best["model"],
        "prior": outer_prior.__dict__,
        "state_offsets": STATE_OFFSETS,
        "fold_source": str(fold_path),
        "native_source": str(NATIVE_SOURCE),
        "config": guards,
    },
    MODEL_DIR / "outer2345_fold1_model.pt",
)

fold1_anchor = _anchor_metrics_v1(test_records)
replay, fold1_metrics, fold1_payload = _bitwise_replay_v1(
    outer_model,
    test_records,
    outer_prior,
    device,
)
fold1_metrics_direct, fold1_bywell, _ = _evaluate_v1(
    outer_model,
    test_records,
    outer_prior,
    device,
    return_rows=False,
)
if abs(fold1_metrics_direct["rmse"] - fold1_metrics["rmse"]) > 1e-10:
    raise RuntimeError("Fold1 replay metric mismatch")
fold1_bywell.to_csv(OUT_DIR / "fold1_bywell.csv", index=False)
np.savez_compressed(OUT_DIR / "fold1_oof.npz", **fold1_payload)

fold1_controls = _controls_v1(
    outer_model, test_records, outer_prior, device
)
fold1_controls.to_csv(OUT_DIR / "fold1_causal_controls.csv", index=False)
fold1_no_gr = _control_value_v1(fold1_controls, "no_gr")
fold1_typewell = _control_value_v1(fold1_controls, "shuffled_typewell")
fold1_heel = _control_value_v1(fold1_controls, "shuffled_heel")
causal_pass = bool(
    fold1_no_gr >= NO_GR_DEGRADATION_GATE
    and fold1_typewell >= SHUFFLED_TYPEWELL_GATE
    and fold1_heel >= SHUFFLED_HEEL_GATE
)
architecture_pass = bool(fold1_metrics["rmse"] <= FOLD1_ARCHITECTURE_GATE)
target_pass = bool(fold1_metrics["rmse"] <= TARGET_GATE)
strict_pass = bool(architecture_pass and causal_pass and replay["bitwise_equal"])

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_untouched_fold1_complete_well_correlation_crf",
    "standalone": True,
    "legal_test_inputs_only": True,
    "sample_unit": "one complete physical well",
    "guards": guards,
    "smoke": smoke,
    "protocol": {
        "epoch_train_folds": [3, 4, 5],
        "epoch_validation_fold": 2,
        "selected_epoch": selected_epoch,
        "final_train_folds": [2, 3, 4, 5],
        "untouched_test_fold": 1,
    },
    "transition_priors": {
        "core345": core_prior.__dict__,
        "outer2345": outer_prior.__dict__,
    },
    "fold2_gateway": fold2_verdict,
    "fold1_anchor": fold1_anchor,
    "fold1_result": fold1_metrics,
    "fold1_controls": fold1_controls.to_dict(orient="records"),
    "bitwise_replay": replay,
    "inference_contract": {
        "optimizer_steps_per_test_well": 0,
        "searches_per_test_well": 0,
        "neural_forwards_per_prediction": 1,
        "deterministic_decodes_per_prediction": 1,
        "condition_number": "not applicable",
        "solve_failures": fold1_metrics["solve_failures"],
    },
    "gates": {
        "architecture_rmse_le": FOLD1_ARCHITECTURE_GATE,
        "target_rmse_le": TARGET_GATE,
        "no_gr_degradation_ge": NO_GR_DEGRADATION_GATE,
        "shuffled_typewell_degradation_ge": SHUFFLED_TYPEWELL_GATE,
        "shuffled_heel_degradation_ge": SHUFFLED_HEEL_GATE,
        "bitwise_replay": True,
    },
    "architecture_pass": architecture_pass,
    "target_5p5_pass": target_pass,
    "causal_pass": causal_pass,
    "strict_pass": strict_pass,
    "training_minutes": {
        "core345": core_minutes,
        "outer2345": outer_minutes,
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "TARGET PASS -> freeze recipe and run all five folds"
        if target_pass and causal_pass and replay["bitwise_equal"]
        else (
            "ARCHITECTURE PASS -> run the remaining outer folds without tuning"
            if strict_pass
            else "FAIL -> close Correlation-CRF v1; do not change only the decoder"
        )
    ),
}
_write_json_v1(OUT_DIR / "verdict.json", verdict)

print("\nFOLD1 CONTROLS")
print(fold1_controls.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default_v1), flush=True)
print("saved:", OUT_DIR, flush=True)
