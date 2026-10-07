"""Nested gateway for a pretrained local matcher plus a complete-well path net.

This cell is restart-safe.  Upload these three source files to
``/root/autodl-tmp`` and run only this file:

1. ``real_correspondence_multiscale_matcher_gateway_v0_cell.py``
2. ``prefix_contrastive_matcher_adaptation_gateway_v1_cell.py``
3. this file

The first two files are loaded for definitions only; their experiment tails
are not executed.  The validated v1 matcher checkpoint is restored from
``oof_exports/prefix_contrastive_matcher_adaptation_gateway_v1/models``.

The two prerequisite cells build a real-correspondence, multiscale local GR
matcher.  This experiment fills the missing architectural square: the local
matcher is pretrained on real row correspondences, then a trainable complete-
well head denoises its dense posterior volume.  The deployable artifact remains
one model pipeline and uses no neighbor wells, CP12, Gold, or test-time search.

Protocol
--------
* Folds 3/4/5 train a fresh local matcher and complete-well head.
* Fold 2 selects only the path-head epoch.
* If Fold 2 clears the gateway, the already outer-trained local matcher is
  paired with a fresh path head trained on Folds 2/3/4/5.
* Fold 1 is then evaluated once, with no-GR and shuffled-typewell controls.

The local matcher is frozen while the path head is trained.  This is deliberate:
it separates "does full-well context resolve local aliases?" from another
unstable end-to-end optimisation experiment.  A successful gateway would
authorise a later low-learning-rate joint fine-tune.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import gc
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Restart-safe bootstrap of the validated local matcher
# ---------------------------------------------------------------------------


def _source_path_v2(filename):
    candidates = (
        Path("/root/autodl-tmp") / filename,
        Path.cwd() / filename,
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Upload {filename} to /root/autodl-tmp before running this cell"
        )
    return path


def _exec_definitions_v2(path, stop_assignment):
    """Execute a source file only up to its top-level experiment tail."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    definitions = []
    found_stop = False
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {
                target.id for target in targets if isinstance(target, ast.Name)
            }
            if stop_assignment in names:
                found_stop = True
                break
        definitions.append(node)
    if not found_stop:
        raise RuntimeError(
            f"Cannot find top-level assignment {stop_assignment!r} in {path}"
        )
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(path) + "::<definitions_only>",
            "exec",
        ),
        globals(),
    )


def _bootstrap_restarted_kernel_v2():
    ready = {
        "samples",
        "folds",
        "core_wells",
        "tune_wells",
        "outer_train_wells",
        "test_wells",
        "final_model_v1",
        "prefix_queries_v1",
        "mixed_epochs_v1",
        "selected_lambda_v1",
        "selected_dose_v1",
        "_fit_mixed_v1",
        "_score_features_v1",
    }
    if ready.issubset(globals()):
        print("PATHNET BOOTSTRAP: reusing validated live-kernel globals", flush=True)
        return

    v0_path = _source_path_v2(
        "real_correspondence_multiscale_matcher_gateway_v0_cell.py"
    )
    _exec_definitions_v2(v0_path, "started")

    global device, wells, folds, fold_path, samples
    global core_wells, tune_wells, outer_train_wells, test_wells, selected_epoch
    wells = _horizontal_wells()
    folds, fold_path = _load_folds(wells)
    fold_by_well = dict(zip(folds["well"], folds["fold"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This gateway requires CUDA")

    missing_records = [
        well for well in wells if not (Path(RECORD_DIR) / f"{well}.npz").exists()
    ]
    if missing_records:
        raise FileNotFoundError(
            "The native legal-input record cache is incomplete. Run "
            "native_resolution_dualref_capacity_v0_cell.py once; first missing: "
            + repr(missing_records[:10])
        )

    samples = {}
    bootstrap_started = time.time()
    for index, well in enumerate(wells, 1):
        samples[well] = _build_well_sample(well, int(fold_by_well[well]))
        if index % 100 == 0 or index == len(wells):
            print(
                f"PATHNET BOOTSTRAP records {index}/{len(wells)} "
                f"elapsed_min={(time.time()-bootstrap_started)/60:.2f}",
                flush=True,
            )

    core_wells = folds.loc[folds["fold"].isin([3, 4, 5]), "well"].tolist()
    tune_wells = folds.loc[folds["fold"].eq(2), "well"].tolist()
    outer_train_wells = folds.loc[folds["fold"].ne(1), "well"].tolist()
    test_wells = folds.loc[folds["fold"].eq(1), "well"].tolist()
    if (
        len(outer_train_wells) != 618
        or len(test_wells) != 155
        or len(core_wells) + len(tune_wells) != len(outer_train_wells)
    ):
        raise RuntimeError(
            "Unexpected nested fold sizes: "
            + repr(
                (
                    len(core_wells),
                    len(tune_wells),
                    len(outer_train_wells),
                    len(test_wells),
                )
            )
        )

    v0_checkpoint = (
        Path(OOF_ROOT)
        / "real_correspondence_multiscale_matcher_gateway_v0"
        / "models"
        / "fold1_gateway_matcher.pt"
    )
    if v0_checkpoint.exists():
        v0_payload = torch.load(
            v0_checkpoint, map_location="cpu", weights_only=False
        )
        selected_epoch = int(v0_payload.get("selected_epoch", 7))
    else:
        selected_epoch = 7
        print(
            "PATHNET BOOTSTRAP: v0 checkpoint absent; using validated epoch=7",
            flush=True,
        )

    v1_path = _source_path_v2(
        "prefix_contrastive_matcher_adaptation_gateway_v1_cell.py"
    )
    _exec_definitions_v2(v1_path, "started_v1")

    global prefix_queries_v1, mixed_epochs_v1
    global selected_lambda_v1, selected_dose_v1, final_model_v1
    prefix_queries_v1 = {}
    prefix_minimum = math.inf
    for index, well in enumerate(sorted(samples), 1):
        prefix = _legal_prefix_query_v1(well, samples[well].state_valid)
        prefix_queries_v1[well] = prefix
        prefix_minimum = min(prefix_minimum, len(prefix.labels))
        if index % 100 == 0 or index == len(samples):
            print(f"PATHNET BOOTSTRAP prefix {index}/{len(samples)}", flush=True)
    if prefix_minimum < 48:
        raise RuntimeError(f"Only {prefix_minimum} legal prefix rows in one well")

    # Recreate exactly the mixed prefix/suffix training sets used by v1.
    for well in outer_train_wells:
        suffix = samples[well].train
        if suffix is None:
            raise RuntimeError(f"{well}: missing suffix training query")
        prefix = _global_prefix_subset_v1(prefix_queries_v1[well])
        samples[well].train = _concat_query_v1(suffix, prefix)

    v1_checkpoint_candidates = (
        Path(OOF_ROOT)
        / "prefix_contrastive_matcher_adaptation_gateway_v1"
        / "models"
        / "fold1_prefix_contrastive_matcher.pt",
        Path("/root/autodl-tmp") / "fold1_prefix_contrastive_matcher.pt",
    )
    v1_checkpoint = next(
        (path for path in v1_checkpoint_candidates if path.exists()), None
    )
    if v1_checkpoint is None:
        raise FileNotFoundError(
            "Cannot find the validated v1 matcher checkpoint. Expected: "
            + repr([str(path) for path in v1_checkpoint_candidates])
            + ". Restore the small checkpoint or rerun the v0 and v1 cells once."
        )
    checkpoint = torch.load(v1_checkpoint, map_location="cpu", weights_only=False)
    mixed_epochs_v1 = int(checkpoint.get("epochs", max(3, selected_epoch)))
    selected_lambda_v1 = float(checkpoint["ridge_lambda"])
    selected_dose_v1 = float(checkpoint["adapter_dose"])
    final_model_v1 = RealCorrespondenceMatcher().to(device)
    final_model_v1.load_state_dict(checkpoint["model"])
    final_model_v1.eval()
    print(
        "PATHNET BOOTSTRAP COMPLETE",
        {
            "v0_source": str(v0_path),
            "v1_source": str(v1_path),
            "v1_checkpoint": str(v1_checkpoint),
            "wells": len(samples),
            "mixed_epochs": mixed_epochs_v1,
            "ridge_lambda": selected_lambda_v1,
            "adapter_dose": selected_dose_v1,
        },
        flush=True,
    )


_bootstrap_restarted_kernel_v2()


# ---------------------------------------------------------------------------
# Bootstrapped contract
# ---------------------------------------------------------------------------


REQUIRED_V2 = {
    "OOF_ROOT",
    "RECORD_DIR",
    "samples",
    "folds",
    "fold_path",
    "core_wells",
    "tune_wells",
    "outer_train_wells",
    "test_wells",
    "device",
    "final_model_v1",
    "prefix_queries_v1",
    "mixed_epochs_v1",
    "selected_lambda_v1",
    "selected_dose_v1",
    "_fit_mixed_v1",
    "_load_raw_record",
    "_window_bank",
    "_score_features_v1",
    "_pair_stats_v1",
    "_adapt_scores_v1",
    "_write_json",
    "_json_default",
    "QuerySet",
    "RetrievalBlockV1",
    "QUERY_HALF_WIDTHS_MD",
    "STATE_OFFSETS",
    "N_STATES",
    "STATE_HALF_WIDTH_FT",
}
missing_v2 = sorted(REQUIRED_V2.difference(globals()))
if missing_v2:
    raise RuntimeError(
        "Restart-safe matcher bootstrap is incomplete; missing: " + repr(missing_v2)
    )


# ---------------------------------------------------------------------------
# Frozen experiment configuration
# ---------------------------------------------------------------------------


EXPERIMENT_V2 = "pretrained_matcher_pathnet_nested_gateway_v2"
OUT_DIR_V2 = Path(OOF_ROOT) / EXPERIMENT_V2
MODEL_DIR_V2 = OUT_DIR_V2 / "models"
for directory_v2 in (OUT_DIR_V2, MODEL_DIR_V2):
    directory_v2.mkdir(parents=True, exist_ok=True)

SEED_V2 = 20260809
NODE_STEP_FT_V2 = 32.0
MAX_TUNE_EPOCHS_V2 = 14
MIN_TUNE_EPOCHS_V2 = 5
EARLY_STOPPING_V2 = 4
LEARNING_RATE_V2 = 3.0e-4
MIN_LEARNING_RATE_V2 = 3.0e-5
WEIGHT_DECAY_V2 = 2.0e-5
GRAD_ACCUM_WELLS_V2 = 4
GRADIENT_CLIP_V2 = 4.0
BASE_CHANNELS_V2 = 32
TEMPORAL_DILATIONS_V2 = (1, 2, 4, 8, 16, 32, 64)
TARGET_SIGMA_FT_V2 = 1.5
LOSS_CE_V2 = 0.55
LOSS_PATH_V2 = 0.35
LOSS_EDGE_V2 = 0.10

FOLD2_GATE_RMSE_V2 = 10.0
FOLD1_ARCHITECTURE_GATE_V2 = 8.0
TARGET_GATE_V2 = 5.5
NO_GR_DEGRADATION_GATE_V2 = 0.30
SHUFFLED_TYPEWELL_GATE_V2 = 0.30

STATE_OFFSETS_V2 = np.asarray(STATE_OFFSETS, np.float32)
N_STATES_V2 = int(N_STATES)
ZERO_STATE_V2 = int(np.argmin(np.abs(STATE_OFFSETS_V2)))
INPUT_CHANNELS_V2 = 11


def _seed_v2(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _rmse_v2(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _normalise_states_v2(values, valid):
    values = np.asarray(values, np.float32)
    valid = np.asarray(valid, bool)
    output = np.zeros_like(values, np.float32)
    if not valid.any():
        return output
    selected = values[:, valid]
    centre = np.median(selected, axis=1, keepdims=True)
    scale = np.quantile(selected, 0.90, axis=1, keepdims=True) - np.quantile(
        selected, 0.10, axis=1, keepdims=True
    )
    scale = np.maximum(scale, 1e-3)
    output[:, valid] = np.clip((selected - centre) / scale, -6.0, 6.0)
    return output


def _robust_prefix_slope_v2(record, z_relative):
    rows = np.arange(record.anchor_index + 1, dtype=np.int64)
    if len(rows) < 16:
        return 0.0
    anchor_md = float(record.md[record.anchor_index])
    recent = rows[record.md[rows] >= anchor_md - 750.0]
    if len(recent) < 32:
        recent = rows[-min(len(rows), 128):]
    x = record.md[recent].astype(np.float64) - anchor_md
    s = (
        record.truth[recent].astype(np.float64)
        - float(record.anchor_tvt)
        + z_relative[recent]
    )
    valid = np.isfinite(x) & np.isfinite(s)
    x = x[valid]
    s = s[valid]
    if len(x) < 16 or np.ptp(x) < 10.0:
        return 0.0
    slopes = []
    step = max(1, len(x) // 32)
    for left in range(0, len(x) - step, step):
        right = min(len(x) - 1, left + max(step, len(x) // 4))
        dx = x[right] - x[left]
        if dx > 1e-6:
            slopes.append((s[right] - s[left]) / dx)
    if not slopes:
        return 0.0
    return float(np.clip(np.median(slopes), -0.08, 0.08))


def _native_z_relative_v2(well):
    with np.load(Path(RECORD_DIR) / f"{well}.npz", allow_pickle=False) as data:
        condition = data["condition"].astype(np.float32)
    if condition.shape[0] < 5:
        raise RuntimeError(f"{well}: missing native z-relative condition")
    return condition[4].astype(np.float64) * 512.0


@dataclass
class DensePathVolumeV2:
    well: str
    eval_rows: int
    node_rows: np.ndarray
    node_md: np.ndarray
    node_z_relative: np.ndarray
    truth_offset: np.ndarray
    state_valid: np.ndarray
    input: np.ndarray


def _dense_query_v2(well):
    record = _load_raw_record(well)
    anchor_md = float(record.md[record.anchor_index])
    final_md = float(record.md[-1])
    node_md = np.arange(
        anchor_md,
        final_md + 0.25 * NODE_STEP_FT_V2,
        NODE_STEP_FT_V2,
        dtype=np.float64,
    )
    node_md = node_md[node_md <= final_md + 1e-6]
    if final_md - node_md[-1] > 0.5 * NODE_STEP_FT_V2:
        node_md = np.r_[node_md, final_md]
    node_rows = np.searchsorted(record.md, node_md, side="left")
    node_rows = np.clip(node_rows, record.anchor_index, len(record.md) - 1)
    node_rows = np.unique(node_rows)
    if node_rows[0] != record.anchor_index:
        node_rows = np.r_[record.anchor_index, node_rows]
    node_md = record.md[node_rows].astype(np.float64)

    labels = record.truth[node_rows].astype(np.float64) - float(record.anchor_tvt)
    label_index = np.rint(labels + float(STATE_HALF_WIDTH_FT)).astype(np.int64)
    safe = np.clip(label_index, 0, N_STATES_V2 - 1)
    legal = (
        np.isfinite(labels)
        & (label_index >= 0)
        & (label_index < N_STATES_V2)
        & samples[well].state_valid[safe]
    )
    if float(np.mean(legal)) < 0.995:
        raise RuntimeError(f"{well}: dense label coverage={np.mean(legal):.5f}")

    patches = _window_bank(
        record.md,
        record.horizontal[0],
        record.horizontal[1],
        node_md,
        QUERY_HALF_WIDTHS_MD,
    )
    distance = node_md - anchor_md
    query = QuerySet(
        patches=patches,
        labels=labels.astype(np.float32),
        label_index=safe.astype(np.int64),
        row_weight=np.ones(len(node_rows), np.float32),
        deep_mask=(distance >= 1500.0),
        sample_rows=node_rows.astype(np.int32),
        requested_rows=int(len(node_rows)),
    )
    z_relative = _native_z_relative_v2(well)[node_rows]
    return record, query, node_rows, node_md, z_relative


def _build_path_volume_v2(model, well, mode="full", donor_well=None):
    record, query, node_rows, node_md, z_relative = _dense_query_v2(well)
    sample = samples[well]
    donor_sample = samples[donor_well] if donor_well is not None else None
    logits, pair_features, state_valid = _score_features_v1(
        model,
        sample,
        query,
        mode=mode,
        donor_sample=donor_sample,
    )

    prefix = prefix_queries_v1[well]
    prefix_logits, prefix_features, prefix_valid = _score_features_v1(
        model,
        sample,
        prefix,
        mode=mode,
        donor_sample=donor_sample,
    )
    if not np.array_equal(state_valid, prefix_valid):
        raise RuntimeError(f"{well}: prefix and suffix validity differ")
    stats = _pair_stats_v1(
        prefix_logits,
        prefix_features,
        prefix.label_index,
        state_valid,
    )
    block = RetrievalBlockV1(
        logits=logits,
        features=pair_features,
        labels=query.labels,
        label_index=query.label_index,
        deep_mask=query.deep_mask,
        requested_rows=query.requested_rows,
    )
    adapted = _adapt_scores_v1(
        block,
        stats,
        selected_lambda_v1,
        selected_dose_v1,
    )

    frozen_norm = _normalise_states_v2(logits, state_valid)
    adapted_norm = _normalise_states_v2(adapted, state_valid)
    pair_mean = _normalise_states_v2(pair_features.mean(axis=2), state_valid)
    pair_max = _normalise_states_v2(pair_features.max(axis=2), state_valid)
    pair_std = _normalise_states_v2(pair_features.std(axis=2), state_valid)

    progress = (node_md - node_md[0]) / max(node_md[-1] - node_md[0], 1.0)
    slope = _robust_prefix_slope_v2(record, _native_z_relative_v2(well))
    constant_s_offset = -z_relative
    slope_s_offset = slope * (node_md - node_md[0]) - z_relative
    state = STATE_OFFSETS_V2[:, None]
    constant_s_map = np.exp(
        -0.5 * ((state - constant_s_offset[None]) / 12.0) ** 2
    ).astype(np.float32)
    slope_s_map = np.exp(
        -0.5 * ((state - slope_s_offset[None]) / 12.0) ** 2
    ).astype(np.float32)
    anchor_map = np.exp(-0.5 * (state / 12.0) ** 2).astype(np.float32)
    anchor_map = np.broadcast_to(anchor_map, (N_STATES_V2, len(node_md)))
    state_coordinate = np.broadcast_to(
        (STATE_OFFSETS_V2 / float(STATE_HALF_WIDTH_FT))[:, None],
        (N_STATES_V2, len(node_md)),
    )
    progress_map = np.broadcast_to(progress[None], (N_STATES_V2, len(node_md)))
    valid_map = np.broadcast_to(state_valid[:, None], (N_STATES_V2, len(node_md)))

    values = np.stack(
        [
            frozen_norm.T,
            adapted_norm.T,
            pair_mean.T,
            pair_max.T,
            pair_std.T,
            constant_s_map,
            slope_s_map,
            anchor_map,
            state_coordinate,
            progress_map,
            valid_map.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float16)
    if values.shape != (INPUT_CHANNELS_V2, N_STATES_V2, len(node_md)):
        raise RuntimeError(f"{well}: unexpected input shape {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{well}: non-finite path volume")
    return DensePathVolumeV2(
        well=well,
        eval_rows=int(record.eval_rows),
        node_rows=node_rows.astype(np.int32),
        node_md=node_md.astype(np.float32),
        node_z_relative=z_relative.astype(np.float32),
        truth_offset=query.labels.astype(np.float32),
        state_valid=state_valid.astype(bool),
        input=values,
    )


def _build_volume_bank_v2(model, wells, label, mode="full"):
    output = {}
    donor_wells = list(wells[1:]) + [wells[0]] if len(wells) > 1 else list(wells)
    started = time.time()
    for index, (well, donor) in enumerate(zip(wells, donor_wells), 1):
        output[well] = _build_path_volume_v2(
            model,
            well,
            mode=mode,
            donor_well=donor if mode == "shuffled_typewell" else None,
        )
        if index % 50 == 0 or index == len(wells):
            print(
                f"PATH VOLUMES {label} {index}/{len(wells)} "
                f"elapsed_min={(time.time()-started)/60:.2f}",
                flush=True,
            )
    return output


# ---------------------------------------------------------------------------
# Complete-well posterior-volume head
# ---------------------------------------------------------------------------


class AxialPathBlockV2(nn.Module):
    def __init__(self, channels, dilation):
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
                padding=(0, 2 * dilation),
                dilation=(1, dilation),
                groups=channels,
            ),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(4, channels),
        )
        self.mix = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, 1),
            nn.GLU(dim=1),
        )

    def forward(self, values):
        hidden = values + self.time(self.state(values))
        return hidden + self.mix(hidden)


class PretrainedMatcherPathNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS_V2, BASE_CHANNELS_V2, 1),
            nn.GroupNorm(4, BASE_CHANNELS_V2),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                AxialPathBlockV2(BASE_CHANNELS_V2, dilation)
                for dilation in TEMPORAL_DILATIONS_V2
            ]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(4, BASE_CHANNELS_V2),
            nn.GELU(),
            nn.Conv2d(BASE_CHANNELS_V2, 1, 1),
        )
        self.base_scale = nn.Parameter(torch.tensor(0.75))

    def forward(self, values):
        hidden = self.input(values)
        for block in self.blocks:
            hidden = block(hidden)
        residual = self.output(hidden).squeeze(1)
        logits = residual + F.softplus(self.base_scale) * values[:, 0]
        valid = values[:, -1] > 0.5
        return logits.masked_fill(~valid, -30.0)


def _volume_tensors_v2(volume):
    values = torch.from_numpy(volume.input.astype(np.float32))[None].to(
        device, non_blocking=True
    )
    truth = torch.from_numpy(volume.truth_offset).to(device, non_blocking=True)
    z_relative = torch.from_numpy(volume.node_z_relative).to(device, non_blocking=True)
    return values, truth, z_relative


def _path_loss_v2(logits, truth, z_relative, state_offsets):
    target = torch.exp(
        -0.5
        * ((state_offsets[:, None] - truth[None]) / TARGET_SIGMA_FT_V2) ** 2
    )
    target = target / target.sum(dim=0, keepdim=True).clamp_min(1e-8)
    log_probability = F.log_softmax(logits, dim=0)
    ce = -(target * log_probability).sum(dim=0).mean()
    probability = torch.softmax(logits, dim=0)
    expected = (probability * state_offsets[:, None]).sum(dim=0)
    path = F.smooth_l1_loss(expected / 8.0, truth / 8.0)
    if len(truth) > 1:
        predicted_s = expected + z_relative
        truth_s = truth + z_relative
        edge = F.smooth_l1_loss(
            torch.diff(predicted_s) / 4.0,
            torch.diff(truth_s) / 4.0,
        )
    else:
        edge = expected.new_zeros(())
    loss = LOSS_CE_V2 * ce + LOSS_PATH_V2 * path + LOSS_EDGE_V2 * edge
    return loss, ce.detach(), path.detach(), edge.detach()


@torch.no_grad()
def _predict_nodes_v2(model, volume, temperature=1.0):
    model.eval()
    values, _, _ = _volume_tensors_v2(volume)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(values)[0] / float(temperature)
    probability = torch.softmax(logits.float(), dim=0)
    offsets = torch.from_numpy(STATE_OFFSETS_V2).to(device)
    prediction = (probability * offsets[:, None]).sum(dim=0).cpu().numpy()
    prediction[0] = 0.0
    return prediction.astype(np.float32)


def _path_payload_v2(volume, prediction_offset):
    record = _load_raw_record(volume.well)
    eval_rows = record.eval_index.astype(np.int64)
    prediction_offset = np.asarray(prediction_offset, np.float64)
    prediction = float(record.anchor_tvt) + np.interp(
        record.md[eval_rows],
        volume.node_md,
        prediction_offset,
    )
    truth = record.truth[eval_rows].astype(np.float64)
    distance = record.md[eval_rows].astype(np.float64) - float(
        record.md[record.anchor_index]
    )
    return {
        "well": np.full(len(eval_rows), volume.well, dtype="U16"),
        "truth": truth,
        "prediction": prediction,
        "distance": distance,
    }


def _summarise_payloads_v2(payloads):
    merged = {
        key: np.concatenate([payload[key] for payload in payloads])
        for key in payloads[0]
    }
    error = merged["prediction"] - merged["truth"]
    bywell = []
    for well in np.unique(merged["well"]):
        mask = merged["well"] == well
        bywell.append(
            {
                "well": well,
                "rows": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
                "mae": float(np.mean(np.abs(error[mask]))),
            }
        )
    bywell = pd.DataFrame(bywell)
    metrics = {
        "rows": int(len(error)),
        "wells": int(len(bywell)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
    }
    for label, low, high in (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    ):
        mask = (merged["distance"] >= low) & (merged["distance"] < high)
        metrics[f"{label}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return metrics, bywell, merged


def _evaluate_pathnet_v2(model, volumes, temperature=1.0):
    payloads = []
    for well in sorted(volumes):
        prediction = _predict_nodes_v2(model, volumes[well], temperature)
        payloads.append(_path_payload_v2(volumes[well], prediction))
    return _summarise_payloads_v2(payloads)


def _train_path_epoch_v2(model, volumes, optimizer, scaler, epoch, phase):
    model.train()
    rng = np.random.default_rng(SEED_V2 + epoch * 1009 + (0 if phase == "core" else 50000))
    wells = list(volumes)
    mean_eval_rows = float(np.mean([volumes[well].eval_rows for well in wells]))
    order = rng.permutation(len(wells))
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "ce": 0.0, "path": 0.0, "edge": 0.0, "skipped": 0}
    state_offsets = torch.from_numpy(STATE_OFFSETS_V2).to(device)
    for step, position in enumerate(order, 1):
        volume = volumes[wells[int(position)]]
        values, truth, z_relative = _volume_tensors_v2(volume)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(values)[0]
            loss, ce, path, edge = _path_loss_v2(
                logits, truth, z_relative, state_offsets
            )
            loss = loss * volume.eval_rows / max(mean_eval_rows, 1.0)
            loss = loss / GRAD_ACCUM_WELLS_V2
        scaler.scale(loss).backward()
        update = step % GRAD_ACCUM_WELLS_V2 == 0 or step == len(order)
        if update:
            before = float(scaler.get_scale())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_V2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            totals["skipped"] += int(float(scaler.get_scale()) < before)
        totals["loss"] += float(loss.detach().cpu()) * GRAD_ACCUM_WELLS_V2
        totals["ce"] += float(ce.cpu())
        totals["path"] += float(path.cpu())
        totals["edge"] += float(edge.cpu())
        if step % 100 == 0 or step == len(order):
            print(
                f"PATHNET {phase} epoch={epoch} wells={step}/{len(order)} "
                f"loss={totals['loss']/step:.4f}",
                flush=True,
            )
    return {
        "train_loss": totals["loss"] / len(order),
        "train_ce": totals["ce"] / len(order),
        "train_path": totals["path"] / len(order),
        "train_edge": totals["edge"] / len(order),
        "skipped_steps": totals["skipped"],
    }


def _new_pathnet_v2(epochs, seed):
    _seed_v2(seed)
    model = PretrainedMatcherPathNetV2().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_V2,
        weight_decay=WEIGHT_DECAY_V2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=MIN_LEARNING_RATE_V2,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=512.0
    )
    return model, optimizer, scheduler, scaler


def _select_path_epoch_v2(train_volumes, valid_volumes):
    model, optimizer, scheduler, scaler = _new_pathnet_v2(
        MAX_TUNE_EPOCHS_V2, SEED_V2
    )
    history = []
    best = None
    stale = 0
    started = time.time()
    for epoch in range(1, MAX_TUNE_EPOCHS_V2 + 1):
        train_metrics = _train_path_epoch_v2(
            model, train_volumes, optimizer, scaler, epoch, "core"
        )
        scheduler.step()
        valid_metrics, _, _ = _evaluate_pathnet_v2(model, valid_volumes)
        row = {
            "epoch": epoch,
            **train_metrics,
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("PATHNET FOLD2 EPOCH", row, flush=True)
        if best is None or valid_metrics["rmse"] < best["rmse"] - 1e-4:
            best = {
                "epoch": epoch,
                "rmse": valid_metrics["rmse"],
                "metrics": valid_metrics,
            }
            stale = 0
        else:
            stale += 1
        if epoch >= MIN_TUNE_EPOCHS_V2 and stale >= EARLY_STOPPING_V2:
            break
    del model, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best, pd.DataFrame(history)


def _fit_final_pathnet_v2(train_volumes, epochs):
    model, optimizer, scheduler, scaler = _new_pathnet_v2(epochs, SEED_V2 + 1000)
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = _train_path_epoch_v2(
            model, train_volumes, optimizer, scaler, epoch, "outer"
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            **train_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("PATHNET OUTER EPOCH", row, flush=True)
    del optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Execute nested gateway
# ---------------------------------------------------------------------------


started_v2 = time.time()
if device.type != "cuda":
    raise RuntimeError("This gateway requires CUDA")
_seed_v2(SEED_V2)

print(
    "PATHNET CONFIG",
    {
        "experiment": EXPERIMENT_V2,
        "sample_unit": "complete physical well",
        "node_step_ft": NODE_STEP_FT_V2,
        "states": N_STATES_V2,
        "input_channels": INPUT_CHANNELS_V2,
        "matcher_epochs": mixed_epochs_v1,
        "adapter_lambda": selected_lambda_v1,
        "adapter_dose": selected_dose_v1,
        "protocol": "folds345 -> fold2 epoch selection -> folds2345 -> fold1",
    },
    flush=True,
)

# Preserve the outer matcher while the core matcher is trained.
outer_matcher_state_v2 = {
    key: value.detach().cpu().clone()
    for key, value in final_model_v1.state_dict().items()
}
final_model_v1.to("cpu")
gc.collect()
torch.cuda.empty_cache()

core_matcher_v2, core_matcher_history_v2 = _fit_mixed_v1(
    core_wells,
    mixed_epochs_v1,
    SEED_V2 + 10,
    "pathnet_core_matcher",
)
core_matcher_history_v2.to_csv(
    OUT_DIR_V2 / "core_matcher_history.csv", index=False
)
core_volumes_v2 = _build_volume_bank_v2(
    core_matcher_v2, core_wells, "core345"
)
tune_volumes_v2 = _build_volume_bank_v2(
    core_matcher_v2, tune_wells, "fold2"
)
del core_matcher_v2
gc.collect()
torch.cuda.empty_cache()

fold2_best_v2, fold2_history_v2 = _select_path_epoch_v2(
    core_volumes_v2, tune_volumes_v2
)
fold2_history_v2.to_csv(OUT_DIR_V2 / "fold2_epoch_history.csv", index=False)
fold2_pass_v2 = bool(fold2_best_v2["rmse"] <= FOLD2_GATE_RMSE_V2)
print("PATHNET FOLD2 BEST", fold2_best_v2, flush=True)

if not fold2_pass_v2:
    verdict_v2 = {
        "experiment": EXPERIMENT_V2,
        "stage": "fold2_nested_gateway",
        "sample_unit": "complete physical well",
        "legal_inputs_only": True,
        "uses_neighbor": False,
        "uses_cp12_gold": False,
        "matcher_pretraining": "real multiscale prefix+suffix correspondence",
        "fold_source": str(fold_path),
        "selected_epoch": int(fold2_best_v2["epoch"]),
        "fold2": fold2_best_v2["metrics"],
        "gate_rmse_le": FOLD2_GATE_RMSE_V2,
        "strict_pass": False,
        "next": "FAIL -> local pretraining plus complete-well path context does not transfer",
        "elapsed_minutes": (time.time() - started_v2) / 60.0,
        "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    }
    _write_json(OUT_DIR_V2 / "verdict.json", verdict_v2)
    print("PATHNET FINAL VERDICT", json.dumps(verdict_v2, indent=2), flush=True)
    raise RuntimeError(json.dumps(verdict_v2, indent=2))

del core_volumes_v2, tune_volumes_v2
gc.collect()

# Restore the outer matcher without retraining it a third time.
final_model_v1.load_state_dict(outer_matcher_state_v2)
final_model_v1.to(device)
final_model_v1.eval()
del outer_matcher_state_v2
torch.cuda.empty_cache()

outer_volumes_v2 = _build_volume_bank_v2(
    final_model_v1, outer_train_wells, "outer2345"
)
fold1_volumes_v2 = _build_volume_bank_v2(
    final_model_v1, test_wells, "fold1"
)
pathnet_v2, outer_history_v2 = _fit_final_pathnet_v2(
    outer_volumes_v2,
    int(fold2_best_v2["epoch"]),
)
outer_history_v2.to_csv(OUT_DIR_V2 / "outer_pathnet_history.csv", index=False)
fold1_metrics_v2, fold1_bywell_v2, fold1_payload_v2 = _evaluate_pathnet_v2(
    pathnet_v2, fold1_volumes_v2
)
fold1_bywell_v2.to_csv(OUT_DIR_V2 / "fold1_bywell.csv", index=False)
np.savez_compressed(OUT_DIR_V2 / "fold1_predictions.npz", **fold1_payload_v2)

# Matched causal controls use the same frozen path head.
no_gr_volumes_v2 = _build_volume_bank_v2(
    final_model_v1, test_wells, "fold1_no_gr", mode="no_gr"
)
no_gr_metrics_v2, _, _ = _evaluate_pathnet_v2(pathnet_v2, no_gr_volumes_v2)
del no_gr_volumes_v2
gc.collect()

shuffle_volumes_v2 = _build_volume_bank_v2(
    final_model_v1,
    test_wells,
    "fold1_shuffled_typewell",
    mode="shuffled_typewell",
)
shuffle_metrics_v2, _, _ = _evaluate_pathnet_v2(pathnet_v2, shuffle_volumes_v2)
del shuffle_volumes_v2
gc.collect()

no_gr_degradation_v2 = no_gr_metrics_v2["rmse"] - fold1_metrics_v2["rmse"]
shuffle_degradation_v2 = shuffle_metrics_v2["rmse"] - fold1_metrics_v2["rmse"]
architecture_pass_v2 = bool(fold1_metrics_v2["rmse"] <= FOLD1_ARCHITECTURE_GATE_V2)
target_pass_v2 = bool(fold1_metrics_v2["rmse"] <= TARGET_GATE_V2)
causal_pass_v2 = bool(
    no_gr_degradation_v2 >= NO_GR_DEGRADATION_GATE_V2
    and shuffle_degradation_v2 >= SHUFFLED_TYPEWELL_GATE_V2
)

checkpoint_v2 = MODEL_DIR_V2 / "fold1_pretrained_matcher_pathnet.pt"
torch.save(
    {
        "matcher": {
            key: value.detach().cpu() for key, value in final_model_v1.state_dict().items()
        },
        "pathnet": {
            key: value.detach().cpu() for key, value in pathnet_v2.state_dict().items()
        },
        "path_epoch": int(fold2_best_v2["epoch"]),
        "matcher_epochs": int(mixed_epochs_v1),
        "state_offsets": STATE_OFFSETS_V2,
        "node_step_ft": NODE_STEP_FT_V2,
        "adapter_lambda": float(selected_lambda_v1),
        "adapter_dose": float(selected_dose_v1),
        "fold_source": str(fold_path),
    },
    checkpoint_v2,
)

interventions_v2 = pd.DataFrame(
    [
        {"mode": "full", **fold1_metrics_v2, "degradation_vs_full": 0.0},
        {
            "mode": "no_gr",
            **no_gr_metrics_v2,
            "degradation_vs_full": no_gr_degradation_v2,
        },
        {
            "mode": "shuffled_typewell",
            **shuffle_metrics_v2,
            "degradation_vs_full": shuffle_degradation_v2,
        },
    ]
)
interventions_v2.to_csv(OUT_DIR_V2 / "fold1_interventions.csv", index=False)

verdict_v2 = {
    "experiment": EXPERIMENT_V2,
    "stage": "nested_fold1_complete_well_gateway",
    "sample_unit": "complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_cp12_gold": False,
    "architecture": "pretrained real-correspondence matcher + frozen complete-well path head",
    "protocol": {
        "epoch_selection_train_folds": [3, 4, 5],
        "epoch_selection_valid_fold": 2,
        "final_train_folds": [2, 3, 4, 5],
        "untouched_test_fold": 1,
        "selected_path_epoch": int(fold2_best_v2["epoch"]),
        "matcher_epochs": int(mixed_epochs_v1),
    },
    "parameters": {
        "matcher": int(sum(parameter.numel() for parameter in final_model_v1.parameters())),
        "pathnet": int(sum(parameter.numel() for parameter in pathnet_v2.parameters())),
    },
    "fold2": fold2_best_v2["metrics"],
    "fold1": fold1_metrics_v2,
    "controls": {
        "no_gr": no_gr_metrics_v2,
        "shuffled_typewell": shuffle_metrics_v2,
        "no_gr_degradation": no_gr_degradation_v2,
        "shuffled_typewell_degradation": shuffle_degradation_v2,
    },
    "gates": {
        "fold2_rmse_le": FOLD2_GATE_RMSE_V2,
        "fold1_architecture_rmse_le": FOLD1_ARCHITECTURE_GATE_V2,
        "target_rmse_le": TARGET_GATE_V2,
        "each_control_degradation_ge": 0.30,
    },
    "architecture_pass": architecture_pass_v2,
    "target_5p5_pass": target_pass_v2,
    "causal_pass": causal_pass_v2,
    "strict_pass": bool(architecture_pass_v2 and causal_pass_v2),
    "checkpoint": str(checkpoint_v2),
    "elapsed_minutes": (time.time() - started_v2) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS 5.5 -> expand to the remaining four folds"
        if target_pass_v2 and causal_pass_v2
        else (
            "PASS architecture -> add low-rate joint matcher fine-tuning"
            if architecture_pass_v2 and causal_pass_v2
            else "FAIL -> pretrained local evidence still cannot support a transferable full-well path"
        )
    ),
}
_write_json(OUT_DIR_V2 / "verdict.json", verdict_v2)

print("\nPATHNET FOLD1 INTERVENTIONS")
print(interventions_v2.to_string(index=False), flush=True)
print("\nPATHNET FINAL VERDICT")
print(json.dumps(verdict_v2, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR_V2, flush=True)
