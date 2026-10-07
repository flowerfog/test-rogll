"""Learned complete-well decoder for the frozen strong/GR sparse union.

The candidate generator is frozen from strong_centered_topk_path_gateway_v2:
nine states nearest the fold-clean strong proposal plus eight states proposed
by legal GR/typewell alignment.  This model learns candidate scoring from the
decomposed evidence instead of the failed scalar hand score.

Protocol: folds 3/4/5 train, Fold 1 selects epoch/temperature/dose, and Fold 2
is evaluated once after all choices freeze.  One complete physical well is the
sample unit.  Gold, CP12, Neighbor, validation truth candidate generation, and
test-time fitting are forbidden.
"""

from __future__ import annotations

import gc
import json
import random
import runpy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REQUIRED_V2 = {
    "DATA_ROOT",
    "OOF_ROOT",
    "POOL_PATH",
    "STATE_OFFSETS",
    "WINDOWS_FT",
    "folds_v2",
    "fold_map_v2",
    "pool_groups_v2",
    "transition_prior_v2",
    "_deduplicate_axis",
    "_robust_affine",
    "_robust_location_scale",
    "_fill_1d",
    "_window_scores",
    "_row_standardise",
    "_finite_mean",
    "_global_ncc",
    "_make_union_v2",
    "_metric_v2",
}


def _restore_v2_if_needed():
    missing = sorted(REQUIRED_V2.difference(globals()))
    if not missing:
        print("V3 bootstrap: reusing live v2 definitions", flush=True)
        return
    candidates = (
        Path("/root/autodl-tmp/strong_centered_topk_path_gateway_v2_cell.py"),
        Path.cwd() / "strong_centered_topk_path_gateway_v2_cell.py",
    )
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise RuntimeError(
            "Upload strong_centered_topk_path_gateway_v2_cell.py to "
            "/root/autodl-tmp, or run v2 in this kernel first. Missing: "
            + repr(missing)
        )
    print("V3 bootstrap: running v2 prerequisite", source, flush=True)
    namespace = runpy.run_path(str(source))
    globals().update(
        {key: value for key, value in namespace.items() if not key.startswith("__")}
    )


_restore_v2_if_needed()

EXPERIMENT_V3 = "strong_centered_sparse_decoder_v3"
OUT_DIR_V3 = Path(OOF_ROOT) / EXPERIMENT_V3
MODEL_DIR_V3 = OUT_DIR_V3 / "models"
OUT_DIR_V3.mkdir(parents=True, exist_ok=True)
MODEL_DIR_V3.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS_V3 = (3, 4, 5)
VALID_FOLD_V3 = 1
TEST_FOLD_V3 = 2
NODE_STRIDE_V3 = 16
PRIOR_STATES_V3 = 9
GR_STATES_V3 = 8
MAX_STATES_V3 = PRIOR_STATES_V3 + GR_STATES_V3
ZERO_STATE_V3 = int(np.argmin(np.abs(STATE_OFFSETS)))
PRIOR_SCALE_FT_V3 = 12.0
TARGET_SIGMA_FT_V3 = 3.0
SEED_V3 = 20260720
MAX_EPOCHS_V3 = 20
MIN_EPOCHS_V3 = 5
PATIENCE_V3 = 5
GRAD_ACCUM_WELLS_V3 = 4
LEARNING_RATE_V3 = 3e-4
WEIGHT_DECAY_V3 = 1e-4
BUILD_WORKERS_V3 = 3
TEMPERATURES_V3 = (0.50, 0.75, 1.00, 1.50, 2.00)
DOSES_V3 = (0.00, 0.25, 0.50, 0.75, 1.00)


def _seed_v3(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable_v3(value):
    if isinstance(value, dict):
        return {str(key): _jsonable_v3(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_v3(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable_v3(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _safe_gradient_v3(values, axis):
    values = np.asarray(values, np.float64)
    axis = np.asarray(axis, np.float64).copy()
    spacing = np.diff(axis)
    positive = spacing[np.isfinite(spacing) & (spacing > 1e-6)]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    for index in range(1, len(axis)):
        if not np.isfinite(axis[index]) or axis[index] <= axis[index - 1]:
            axis[index] = axis[index - 1] + fallback
    return np.gradient(values, axis).astype(np.float32)


def _rolling_mean_v3(values, rows):
    values = np.asarray(values, np.float64)
    rows = max(int(rows), 3)
    return (
        pd.Series(values)
        .rolling(rows, center=True, min_periods=max(2, rows // 4))
        .mean()
        .interpolate(limit_direction="both")
        .to_numpy(np.float32)
    )


def _gather_states_v3(matrix, indices):
    matrix = np.asarray(matrix, np.float32)
    indices = np.asarray(indices, np.int64)
    safe = np.maximum(indices, 0)
    gathered = np.take_along_axis(matrix, safe, axis=1)
    gathered[indices < 0] = 0.0
    gathered[~np.isfinite(gathered)] = 0.0
    return gathered.astype(np.float32)


def _prior_only_union_v3(n_nodes):
    prior_indices = np.argsort(np.abs(STATE_OFFSETS))[:PRIOR_STATES_V3]
    indices = np.full((n_nodes, MAX_STATES_V3), -1, np.int16)
    indices[:, :PRIOR_STATES_V3] = prior_indices[None, :]
    return indices


# Candidate feature columns 0:5 are legal static state/prior descriptors.  All
# remaining columns contain GR/typewell alignment evidence and are zeroed in
# the matched no-alignment control.
STATIC_CANDIDATE_FEATURES_V3 = 5
GR_CONTEXT_START_V3 = 7


def _build_rich_record_v3(well, control="full"):
    frame = pool_groups_v2[str(well)]
    horizontal = pd.read_csv(Path(DATA_ROOT) / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(Path(DATA_ROOT) / "train" / f"{well}__typewell.csv")
    row_index = frame["row_idx"].to_numpy(np.int64)
    md_all = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    z_all = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
    gr_all = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    visible = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.isfinite(visible)
    if known.sum() < 64 or len(row_index) < 64:
        raise RuntimeError(f"insufficient known/eval rows {known.sum()}/{len(row_index)}")
    anchor = int(np.flatnonzero(known)[-1])
    anchor_md = float(md_all[anchor])
    anchor_tvt = float(visible[anchor])

    strong = frame["strong"].to_numpy(np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    if not np.isfinite(strong).all() or not np.isfinite(truth).all():
        raise RuntimeError("non-finite strong/truth")
    base_all = np.full(len(horizontal), np.nan, np.float64)
    base_all[known] = visible[known]
    base_all[row_index] = strong
    base_all = _fill_1d(md_all, base_all)

    tw_tvt, tw_gr = _deduplicate_axis(
        pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64),
        pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64),
    )
    tw_at_known = np.interp(visible[known], tw_tvt, tw_gr, left=np.nan, right=np.nan)
    gain, intercept, calibration_pairs, calibration_rmse = _robust_affine(
        tw_at_known, gr_all[known]
    )
    centre, scale = _robust_location_scale(gr_all[known])
    observed = (_fill_1d(md_all, gr_all) - centre) / scale
    tw_calibrated = (gain * tw_gr + intercept - centre) / scale
    candidate_tvt = base_all[:, None] + STATE_OFFSETS[None, :]
    expected = np.interp(
        candidate_tvt.reshape(-1),
        tw_tvt,
        tw_calibrated,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape).astype(np.float32)
    valid = np.isfinite(expected) & np.isfinite(gr_all)[:, None]

    node_position = np.unique(
        np.r_[np.arange(0, len(row_index), NODE_STRIDE_V3), len(row_index) - 1]
    ).astype(np.int64)
    node_full_index = row_index[node_position]
    node_md = md_all[node_full_index]
    distance = md_all[row_index] - anchor_md
    node_distance = distance[node_position]
    median_step = max(float(abs(np.nanmedian(np.diff(md_all[np.isfinite(md_all)])))), 0.25)
    window_rows = {
        float(window): max(9, int(round(float(window) / median_step)))
        for window in WINDOWS_FT
    }
    scores = _window_scores(observed, expected, valid, node_full_index, window_rows)
    raw_ncc = [scores[window][0].astype(np.float32) for window in WINDOWS_FT]
    raw_rmse = [scores[window][1].astype(np.float32) for window in WINDOWS_FT]
    coverage = [scores[window][2].astype(np.float32) for window in WINDOWS_FT]
    ncc_z = [_row_standardise(item) for item in raw_ncc]
    neg_rmse_z = [_row_standardise(-item) for item in raw_rmse]
    ncc_all = _finite_mean(ncc_z)
    neg_rmse_all = _finite_mean(neg_rmse_z)
    local = _finite_mean([ncc_all, neg_rmse_all])
    global_profile = _global_ncc(observed[row_index], expected[row_index], valid[row_index])
    global_z = _row_standardise(global_profile[None, :])
    global_volume = np.broadcast_to(global_z, local.shape).copy()
    emission = _finite_mean([local, global_volume])

    if control == "shuffled":
        shift = 31 + (int(str(well)[:4], 16) % 61)
        matrices = [
            *raw_ncc,
            *raw_rmse,
            *coverage,
            *ncc_z,
            *neg_rmse_z,
            ncc_all,
            neg_rmse_all,
            local,
            global_volume,
            emission,
        ]
        matrices = [np.roll(item, shift, axis=1) for item in matrices]
        cursor = 0
        raw_ncc = matrices[cursor : cursor + len(WINDOWS_FT)]
        cursor += len(WINDOWS_FT)
        raw_rmse = matrices[cursor : cursor + len(WINDOWS_FT)]
        cursor += len(WINDOWS_FT)
        coverage = matrices[cursor : cursor + len(WINDOWS_FT)]
        cursor += len(WINDOWS_FT)
        ncc_z = matrices[cursor : cursor + len(WINDOWS_FT)]
        cursor += len(WINDOWS_FT)
        neg_rmse_z = matrices[cursor : cursor + len(WINDOWS_FT)]
        cursor += len(WINDOWS_FT)
        ncc_all, neg_rmse_all, local, global_volume, emission = matrices[cursor:]

    if control == "no_alignment":
        indices = _prior_only_union_v3(len(node_position))
    else:
        indices, _ = _make_union_v2(emission)
    valid_candidate = indices >= 0
    safe_indices = np.maximum(indices, 0)
    offsets = STATE_OFFSETS[safe_indices].astype(np.float32)
    offsets[~valid_candidate] = 0.0
    prior = -0.5 * (offsets / PRIOR_SCALE_FT_V3) ** 2
    prior_member = np.isin(
        safe_indices, np.argsort(np.abs(STATE_OFFSETS))[:PRIOR_STATES_V3]
    ).astype(np.float32)

    candidate_parts = [
        offsets / 32.0,
        np.abs(offsets) / 32.0,
        np.square(offsets / 32.0),
        np.clip(prior / 8.0, -2.0, 0.0),
        prior_member,
        _gather_states_v3(emission, indices),
        _gather_states_v3(local, indices),
        _gather_states_v3(global_volume, indices),
        _gather_states_v3(ncc_all, indices),
        _gather_states_v3(neg_rmse_all, indices),
    ]
    for item in ncc_z:
        candidate_parts.append(_gather_states_v3(item, indices))
    for item in neg_rmse_z:
        candidate_parts.append(_gather_states_v3(item, indices))
    for item in raw_ncc:
        candidate_parts.append(np.clip(_gather_states_v3(item, indices), -1.0, 1.0))
    for item in raw_rmse:
        candidate_parts.append(
            np.log1p(np.clip(_gather_states_v3(item, indices), 0.0, 20.0)) / 3.0
        )
    for item in coverage:
        candidate_parts.append(np.clip(_gather_states_v3(item, indices), 0.0, 1.0))
    candidate_features = np.stack(candidate_parts, axis=-1).astype(np.float32)
    candidate_features[~valid_candidate] = 0.0
    if control == "no_alignment":
        candidate_features[..., STATIC_CANDIDATE_FEATURES_V3:] = 0.0

    strong_d1 = _safe_gradient_v3(strong, md_all[row_index])
    strong_d2 = _safe_gradient_v3(strong_d1, md_all[row_index])
    z_d1 = _safe_gradient_v3(z_all[row_index], md_all[row_index])
    gr_eval = observed[row_index].astype(np.float32)
    smooth64 = _rolling_mean_v3(gr_eval, max(5, int(round(64.0 / median_step))))
    smooth256 = _rolling_mean_v3(gr_eval, max(9, int(round(256.0 / median_step))))
    finite_emission = np.where(np.isfinite(emission), emission, -np.inf)
    sorted_emission = np.sort(finite_emission, axis=1)
    margin = sorted_emission[:, -1] - sorted_emission[:, -2]
    emission_valid = np.isfinite(emission)
    emission_mean = np.divide(
        np.where(emission_valid, emission, 0.0).sum(axis=1),
        np.maximum(emission_valid.sum(axis=1), 1),
    )
    emission_var = np.divide(
        np.where(emission_valid, (emission - emission_mean[:, None]) ** 2, 0.0).sum(axis=1),
        np.maximum(emission_valid.sum(axis=1), 1),
    )
    eval_span = max(float(distance[-1]), 1.0)
    context = np.stack(
        [
            np.clip(node_distance / eval_span, 0.0, 1.5),
            node_distance / 3000.0,
            (strong[node_position] - anchor_tvt) / 64.0,
            strong_d1[node_position] * 32.0,
            strong_d2[node_position] * 1024.0,
            z_d1[node_position],
            np.full(len(node_position), np.log1p(len(row_index)) / 10.0),
            gr_eval[node_position],
            smooth64[node_position],
            gr_eval[node_position] - smooth64[node_position],
            smooth256[node_position],
            gr_eval[node_position] - smooth256[node_position],
            np.clip(margin, 0.0, 8.0),
            np.sqrt(np.maximum(emission_var, 0.0)),
            np.full(len(node_position), gain / 3.0),
            np.full(len(node_position), intercept / 64.0),
            np.full(
                len(node_position),
                0.0 if not np.isfinite(calibration_rmse) else calibration_rmse / 32.0,
            ),
        ],
        axis=1,
    ).astype(np.float32)
    if control == "no_alignment":
        context[:, GR_CONTEXT_START_V3:] = 0.0

    node_weight = np.diff(np.r_[node_position, len(row_index)]).astype(np.float32)
    node_weight /= max(float(node_weight.mean()), 1e-6)
    del expected, valid, candidate_tvt, scores
    return {
        "well": str(well),
        "fold": int(frame["fold"].iloc[0]),
        "candidate_features": candidate_features,
        "context": context,
        "candidate_indices": indices.astype(np.int16),
        "candidate_valid": valid_candidate,
        "candidate_offsets": offsets,
        "target_residual": (truth[node_position] - strong[node_position]).astype(np.float32),
        "node_weight": node_weight,
        "node_position": node_position.astype(np.int32),
        "node_md": node_md.astype(np.float32),
        "md": md_all[row_index].astype(np.float32),
        "distance": distance.astype(np.float32),
        "strong": strong.astype(np.float32),
        "truth": truth.astype(np.float32),
        "calibration_pairs": int(calibration_pairs),
        "calibration_rmse": calibration_rmse,
    }


def _build_records_v3(folds, control="full"):
    folds = tuple(int(fold) for fold in folds)
    wells = folds_v2.loc[folds_v2["fold"].isin(folds), "well"].tolist()
    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=BUILD_WORKERS_V3) as executor:
        futures = {
            executor.submit(_build_rich_record_v3, well, control): well for well in wells
        }
        for index, future in enumerate(as_completed(futures), 1):
            well = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append(
                    {"well": well, "fold": int(fold_map_v2[well]), "reason": repr(exc)}
                )
            if index % 25 == 0 or index == len(wells):
                print(
                    f"build {control} folds{folds} {index}/{len(wells)}",
                    flush=True,
                )
    records.sort(key=lambda item: item["well"])
    if len(records) < int(0.98 * len(wells)):
        raise RuntimeError(f"only {len(records)} / {len(wells)} records for {folds}")
    return records, failures


class SparseCompleteWellDecoderV3(nn.Module):
    def __init__(self, candidate_features, context_features, hidden=64):
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.sequence = nn.GRU(
            input_size=hidden * 3,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.10,
        )
        self.score = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, candidate_features, context, valid):
        encoded = self.candidate_encoder(candidate_features)
        valid_float = valid[..., None].float()
        mean = (encoded * valid_float).sum(dim=2) / valid_float.sum(dim=2).clamp_min(1.0)
        masked = encoded.masked_fill(~valid[..., None], -1e4)
        maximum = masked.max(dim=2).values
        context_encoded = self.context_encoder(context)
        sequence_input = torch.cat([mean, maximum, context_encoded], dim=-1)
        sequence, _ = self.sequence(sequence_input)
        expanded = sequence[:, :, None, :].expand(-1, -1, encoded.shape[2], -1)
        logits = self.score(torch.cat([encoded, expanded], dim=-1)).squeeze(-1)
        return logits.masked_fill(~valid, -1e4)


def _tensor_record_v3(record, device):
    return {
        "candidate_features": torch.from_numpy(record["candidate_features"])[None].to(device),
        "context": torch.from_numpy(record["context"])[None].to(device),
        "valid": torch.from_numpy(record["candidate_valid"])[None].to(device),
        "offsets": torch.from_numpy(record["candidate_offsets"])[None].to(device),
        "target": torch.from_numpy(record["target_residual"])[None].to(device),
        "weight": torch.from_numpy(record["node_weight"])[None].to(device),
    }


def _loss_v3(model, record, device):
    batch = _tensor_record_v3(record, device)
    logits = model(batch["candidate_features"], batch["context"], batch["valid"])
    target_score = -0.5 * (
        (batch["offsets"] - batch["target"][:, :, None]) / TARGET_SIGMA_FT_V3
    ) ** 2
    target_score = target_score.masked_fill(~batch["valid"], -1e4)
    target_probability = torch.softmax(target_score, dim=-1)
    log_probability = torch.log_softmax(logits, dim=-1)
    ce_node = -(target_probability * log_probability).sum(dim=-1)
    probability = torch.softmax(logits, dim=-1)
    expected = (probability * batch["offsets"]).sum(dim=-1)
    path_mse_node = (expected - batch["target"]) ** 2
    weight = batch["weight"]
    ce = (ce_node * weight).sum() / weight.sum().clamp_min(1e-6)
    path_mse = (path_mse_node * weight).sum() / weight.sum().clamp_min(1e-6)
    if expected.shape[1] > 1:
        delta_error = torch.diff(expected, dim=1) - torch.diff(batch["target"], dim=1)
        delta = torch.mean(delta_error**2)
    else:
        delta = torch.zeros((), device=device)
    loss = ce + 0.20 * path_mse / 64.0 + 0.05 * delta / 16.0
    return loss, {
        "ce": float(ce.detach().cpu()),
        "path_mse": float(path_mse.detach().cpu()),
        "delta_mse": float(delta.detach().cpu()),
    }


@torch.no_grad()
def _record_logits_v3(model, record, device):
    batch = _tensor_record_v3(record, device)
    return (
        model(batch["candidate_features"], batch["context"], batch["valid"])[0]
        .float()
        .cpu()
        .numpy()
    )


def _decode_posterior_v3(record, logits, temperature, dose):
    valid = record["candidate_valid"]
    safe = np.where(valid, logits / float(temperature), -1e9)
    safe -= np.max(safe, axis=1, keepdims=True)
    probability = np.exp(safe) * valid
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
    node_residual = np.sum(probability * record["candidate_offsets"], axis=1)
    anchor_md = float(record["md"][0] - record["distance"][0])
    residual = np.interp(
        record["md"],
        np.r_[anchor_md, record["node_md"]],
        np.r_[0.0, node_residual],
    )
    return (record["strong"] + float(dose) * residual).astype(np.float32)


def _evaluate_logits_v3(records, logits_by_well, temperature, dose):
    truth = []
    base = []
    prediction = []
    wells = []
    distance = []
    for record in records:
        current = _decode_posterior_v3(
            record, logits_by_well[record["well"]], temperature, dose
        )
        truth.append(record["truth"])
        base.append(record["strong"])
        prediction.append(current)
        wells.append(np.repeat(record["well"], len(current)))
        distance.append(record["distance"])
    return _metric_v2(
        np.concatenate(truth),
        np.concatenate(base),
        np.concatenate(prediction),
        np.concatenate(wells),
        np.concatenate(distance),
    )


def _evaluate_oracle_v3(records):
    truth = []
    base = []
    prediction = []
    wells = []
    distance = []
    for record in records:
        candidate_distance = np.abs(
            record["candidate_offsets"] - record["target_residual"][:, None]
        )
        candidate_distance[~record["candidate_valid"]] = np.inf
        selected = record["candidate_offsets"][
            np.arange(len(candidate_distance)), np.argmin(candidate_distance, axis=1)
        ]
        anchor_md = float(record["md"][0] - record["distance"][0])
        residual = np.interp(
            record["md"],
            np.r_[anchor_md, record["node_md"]],
            np.r_[0.0, selected],
        )
        current = record["strong"] + residual
        truth.append(record["truth"])
        base.append(record["strong"])
        prediction.append(current)
        wells.append(np.repeat(record["well"], len(current)))
        distance.append(record["distance"])
    return _metric_v2(
        np.concatenate(truth),
        np.concatenate(base),
        np.concatenate(prediction),
        np.concatenate(wells),
        np.concatenate(distance),
    )


def _collect_logits_v3(model, records, device):
    model.eval()
    return {
        record["well"]: _record_logits_v3(model, record, device)
        for record in records
    }


started_v3 = time.time()
device_v3 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_seed_v3(SEED_V3)
print(
    "V3 CONFIG",
    {
        "device": str(device_v3),
        "train_folds": TRAIN_FOLDS_V3,
        "valid_fold": VALID_FOLD_V3,
        "test_fold": TEST_FOLD_V3,
        "candidate_union": [PRIOR_STATES_V3, GR_STATES_V3],
        "node_stride": NODE_STRIDE_V3,
    },
    flush=True,
)

train_records_v3, train_failures_v3 = _build_records_v3(TRAIN_FOLDS_V3)
valid_records_v3, valid_failures_v3 = _build_records_v3((VALID_FOLD_V3,))
feature_count_v3 = int(train_records_v3[0]["candidate_features"].shape[-1])
context_count_v3 = int(train_records_v3[0]["context"].shape[-1])
valid_oracle_v3, _ = _evaluate_oracle_v3(valid_records_v3)
print("V3 FOLD1 UNION ORACLE", valid_oracle_v3, flush=True)

# Fail quickly if the architecture/loss cannot consume even one fixed menu.
_seed_v3(SEED_V3 + 1)
capacity_record_v3 = train_records_v3[0]
capacity_model_v3 = SparseCompleteWellDecoderV3(
    feature_count_v3, context_count_v3
).to(device_v3)
capacity_optimizer_v3 = torch.optim.AdamW(
    capacity_model_v3.parameters(), lr=1e-3, weight_decay=0.0
)
for capacity_step_v3 in range(1, 251):
    capacity_model_v3.train()
    capacity_optimizer_v3.zero_grad(set_to_none=True)
    capacity_loss_v3, _ = _loss_v3(
        capacity_model_v3, capacity_record_v3, device_v3
    )
    capacity_loss_v3.backward()
    torch.nn.utils.clip_grad_norm_(capacity_model_v3.parameters(), 5.0)
    capacity_optimizer_v3.step()
capacity_logits_v3 = _collect_logits_v3(
    capacity_model_v3, [capacity_record_v3], device_v3
)
capacity_rows_v3 = []
for capacity_temperature_v3 in (0.25, 0.50, 1.00):
    capacity_metrics_v3, _ = _evaluate_logits_v3(
        [capacity_record_v3],
        capacity_logits_v3,
        temperature=capacity_temperature_v3,
        dose=1.0,
    )
    capacity_rows_v3.append(
        {"temperature": capacity_temperature_v3, **capacity_metrics_v3}
    )
capacity_best_v3 = min(capacity_rows_v3, key=lambda item: item["rmse"])
capacity_oracle_v3, _ = _evaluate_oracle_v3([capacity_record_v3])
capacity_pass_v3 = bool(
    capacity_best_v3["rmse"] <= capacity_oracle_v3["rmse"] + 1.0
)
capacity_verdict_v3 = _jsonable_v3(
    {
        "well": capacity_record_v3["well"],
        "updates": 250,
        "oracle": capacity_oracle_v3,
        "best": capacity_best_v3,
        "gate_oracle_plus_ft": 1.0,
        "pass": capacity_pass_v3,
    }
)
(OUT_DIR_V3 / "capacity_verdict.json").write_text(
    json.dumps(capacity_verdict_v3, indent=2), encoding="utf-8"
)
print("V3 CAPACITY", json.dumps(capacity_verdict_v3, indent=2), flush=True)
del capacity_model_v3, capacity_optimizer_v3, capacity_logits_v3
gc.collect()
if device_v3.type == "cuda":
    torch.cuda.empty_cache()
if not capacity_pass_v3:
    raise RuntimeError(
        "Sparse decoder capacity failed before population training: "
        + json.dumps(capacity_verdict_v3)
    )

_seed_v3(SEED_V3)
model_v3 = SparseCompleteWellDecoderV3(feature_count_v3, context_count_v3).to(device_v3)
parameters_v3 = int(sum(parameter.numel() for parameter in model_v3.parameters()))
optimizer_v3 = torch.optim.AdamW(
    model_v3.parameters(), lr=LEARNING_RATE_V3, weight_decay=WEIGHT_DECAY_V3
)
scheduler_v3 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_v3, T_max=MAX_EPOCHS_V3, eta_min=3e-5
)
scaler_v3 = torch.amp.GradScaler("cuda", enabled=device_v3.type == "cuda")
print(
    "V3 MODEL",
    {
        "candidate_features": feature_count_v3,
        "context_features": context_count_v3,
        "parameters": parameters_v3,
        "train_wells": len(train_records_v3),
        "valid_wells": len(valid_records_v3),
    },
    flush=True,
)

history_v3 = []
best_v3 = None
stale_v3 = 0
for epoch_v3 in range(1, MAX_EPOCHS_V3 + 1):
    model_v3.train()
    order_v3 = np.random.permutation(len(train_records_v3))
    optimizer_v3.zero_grad(set_to_none=True)
    totals_v3 = {"loss": 0.0, "ce": 0.0, "path_mse": 0.0, "delta_mse": 0.0}
    updates_v3 = 0
    for step_v3, record_index_v3 in enumerate(order_v3, 1):
        record_v3 = train_records_v3[int(record_index_v3)]
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device_v3.type == "cuda"
        ):
            loss_v3, parts_v3 = _loss_v3(model_v3, record_v3, device_v3)
            scaled_v3 = loss_v3 / GRAD_ACCUM_WELLS_V3
        scaler_v3.scale(scaled_v3).backward()
        if step_v3 % GRAD_ACCUM_WELLS_V3 == 0 or step_v3 == len(order_v3):
            scaler_v3.unscale_(optimizer_v3)
            torch.nn.utils.clip_grad_norm_(model_v3.parameters(), 5.0)
            scaler_v3.step(optimizer_v3)
            scaler_v3.update()
            optimizer_v3.zero_grad(set_to_none=True)
            updates_v3 += 1
        totals_v3["loss"] += float(loss_v3.detach().cpu())
        for key_v3, value_v3 in parts_v3.items():
            totals_v3[key_v3] += value_v3
        if step_v3 % 100 == 0 or step_v3 == len(order_v3):
            print(
                f"V3 epoch {epoch_v3} wells {step_v3}/{len(order_v3)} "
                f"loss={totals_v3['loss']/step_v3:.4f}",
                flush=True,
            )
    scheduler_v3.step()
    valid_logits_v3 = _collect_logits_v3(model_v3, valid_records_v3, device_v3)
    valid_raw_v3, _ = _evaluate_logits_v3(
        valid_records_v3, valid_logits_v3, temperature=1.0, dose=1.0
    )
    epoch_dose_rows_v3 = []
    for epoch_dose_v3 in DOSES_V3[1:]:
        epoch_metrics_v3, _ = _evaluate_logits_v3(
            valid_records_v3,
            valid_logits_v3,
            temperature=1.0,
            dose=epoch_dose_v3,
        )
        epoch_dose_rows_v3.append((epoch_dose_v3, epoch_metrics_v3))
    epoch_best_dose_v3, valid_metrics_v3 = min(
        epoch_dose_rows_v3, key=lambda item: item[1]["rmse"]
    )
    row_v3 = {
        "epoch": epoch_v3,
        "train_loss": totals_v3["loss"] / len(order_v3),
        "train_ce": totals_v3["ce"] / len(order_v3),
        "train_path_rmse": float(
            np.sqrt(max(totals_v3["path_mse"] / len(order_v3), 0.0))
        ),
        "train_delta_rmse": float(
            np.sqrt(max(totals_v3["delta_mse"] / len(order_v3), 0.0))
        ),
        "updates": updates_v3,
        "valid_raw_rmse": valid_raw_v3["rmse"],
        "valid_calibrated_dose": epoch_best_dose_v3,
        **{f"valid_{key}": value for key, value in valid_metrics_v3.items()},
        "learning_rate": optimizer_v3.param_groups[0]["lr"],
        "elapsed_minutes": (time.time() - started_v3) / 60.0,
    }
    history_v3.append(row_v3)
    print("V3 EPOCH", row_v3, flush=True)
    if best_v3 is None or valid_metrics_v3["rmse"] < best_v3["rmse"] - 1e-5:
        best_v3 = {
            "epoch": epoch_v3,
            "rmse": valid_metrics_v3["rmse"],
            "state": {
                key: value.detach().cpu().clone()
                for key, value in model_v3.state_dict().items()
            },
        }
        stale_v3 = 0
    else:
        stale_v3 += 1
    if epoch_v3 >= MIN_EPOCHS_V3 and stale_v3 >= PATIENCE_V3:
        print("V3 early stopping", flush=True)
        break

pd.DataFrame(history_v3).to_csv(OUT_DIR_V3 / "training_history.csv", index=False)
model_v3.load_state_dict(best_v3["state"])
checkpoint_v3 = MODEL_DIR_V3 / "best_sparse_decoder_fold1.pt"
torch.save(
    {
        "model": best_v3["state"],
        "epoch": int(best_v3["epoch"]),
        "candidate_features": feature_count_v3,
        "context_features": context_count_v3,
        "seed": SEED_V3,
    },
    checkpoint_v3,
)

valid_logits_v3 = _collect_logits_v3(model_v3, valid_records_v3, device_v3)
scan_rows_v3 = []
for temperature_v3 in TEMPERATURES_V3:
    for dose_v3 in DOSES_V3:
        metrics_v3, _ = _evaluate_logits_v3(
            valid_records_v3, valid_logits_v3, temperature_v3, dose_v3
        )
        scan_rows_v3.append(
            {"temperature": temperature_v3, "dose": dose_v3, **metrics_v3}
        )
scan_v3 = pd.DataFrame(scan_rows_v3).sort_values("rmse")
scan_v3.to_csv(OUT_DIR_V3 / "fold1_decoder_scan.csv", index=False)
print("\nV3 FOLD1 DECODER SCAN")
print(scan_v3.head(20).to_string(index=False))
selected_v3 = scan_v3.iloc[0]
nonzero_scan_v3 = scan_v3[scan_v3["dose"] > 0]
selected_nonzero_v3 = nonzero_scan_v3.iloc[0]

del train_records_v3, valid_logits_v3
gc.collect()
if device_v3.type == "cuda":
    torch.cuda.empty_cache()

test_records_v3, test_failures_v3 = _build_records_v3((TEST_FOLD_V3,))
test_oracle_v3, _ = _evaluate_oracle_v3(test_records_v3)
test_logits_v3 = _collect_logits_v3(model_v3, test_records_v3, device_v3)
test_metrics_v3, test_wells_v3 = _evaluate_logits_v3(
    test_records_v3,
    test_logits_v3,
    float(selected_v3["temperature"]),
    float(selected_v3["dose"]),
)
test_nonzero_v3, _ = _evaluate_logits_v3(
    test_records_v3,
    test_logits_v3,
    float(selected_nonzero_v3["temperature"]),
    float(selected_nonzero_v3["dose"]),
)
test_wells_v3.to_csv(OUT_DIR_V3 / "fold2_well_metrics.csv", index=False)

shuffled_records_v3, shuffled_failures_v3 = _build_records_v3(
    (TEST_FOLD_V3,), control="shuffled"
)
shuffled_logits_v3 = _collect_logits_v3(model_v3, shuffled_records_v3, device_v3)
shuffled_metrics_v3, _ = _evaluate_logits_v3(
    shuffled_records_v3,
    shuffled_logits_v3,
    float(selected_nonzero_v3["temperature"]),
    float(selected_nonzero_v3["dose"]),
)

no_alignment_records_v3 = []
for record_v3 in test_records_v3:
    control_v3 = {key: value for key, value in record_v3.items()}
    control_v3["candidate_indices"] = _prior_only_union_v3(
        len(record_v3["node_position"])
    )
    control_v3["candidate_valid"] = control_v3["candidate_indices"] >= 0
    safe_v3 = np.maximum(control_v3["candidate_indices"], 0)
    offsets_v3 = STATE_OFFSETS[safe_v3].astype(np.float32)
    offsets_v3[~control_v3["candidate_valid"]] = 0.0
    control_v3["candidate_offsets"] = offsets_v3
    features_v3 = np.zeros_like(record_v3["candidate_features"])
    features_v3[..., :STATIC_CANDIDATE_FEATURES_V3] = record_v3[
        "candidate_features"
    ][..., :STATIC_CANDIDATE_FEATURES_V3]
    features_v3[:, PRIOR_STATES_V3:, :] = 0.0
    control_v3["candidate_features"] = features_v3
    context_v3 = record_v3["context"].copy()
    context_v3[:, GR_CONTEXT_START_V3:] = 0.0
    control_v3["context"] = context_v3
    no_alignment_records_v3.append(control_v3)
no_alignment_logits_v3 = _collect_logits_v3(
    model_v3, no_alignment_records_v3, device_v3
)
no_alignment_metrics_v3, _ = _evaluate_logits_v3(
    no_alignment_records_v3,
    no_alignment_logits_v3,
    float(selected_nonzero_v3["temperature"]),
    float(selected_nonzero_v3["dose"]),
)

all_failures_v3 = train_failures_v3 + valid_failures_v3 + test_failures_v3 + shuffled_failures_v3
pd.DataFrame(all_failures_v3).to_csv(OUT_DIR_V3 / "failures.csv", index=False)

print("\nV3 FOLD2 DEPLOYMENT")
print(json.dumps(_jsonable_v3(test_metrics_v3), indent=2))
print("\nV3 FOLD2 BEST NONZERO")
print(json.dumps(_jsonable_v3(test_nonzero_v3), indent=2))
print("\nV3 FOLD2 UNION ORACLE")
print(json.dumps(_jsonable_v3(test_oracle_v3), indent=2))
print("\nV3 CONTROLS")
print(
    json.dumps(
        _jsonable_v3(
            {
                "shuffled_alignment": shuffled_metrics_v3,
                "no_alignment": no_alignment_metrics_v3,
            }
        ),
        indent=2,
    )
)

gates_v3 = {
    "fold2_union_oracle_le": 5.5,
    "fold2_rmse_le": 8.0,
    "fold2_gain_ge": 0.50,
    "far_gain_ge": 1.0,
    "positive_well_fraction_ge": 0.55,
    "shuffle_degradation_ge": 0.50,
    "no_alignment_degradation_ge": 0.30,
}
checks_v3 = {
    "union_oracle": bool(
        test_oracle_v3["rmse"] <= gates_v3["fold2_union_oracle_le"]
    ),
    "rmse": bool(test_nonzero_v3["rmse"] <= gates_v3["fold2_rmse_le"]),
    "gain": bool(test_nonzero_v3["gain_vs_base"] >= gates_v3["fold2_gain_ge"]),
    "far_gain": bool(
        test_nonzero_v3.get("far_gt2000_gain", -np.inf) >= gates_v3["far_gain_ge"]
    ),
    "positive_wells": bool(
        test_nonzero_v3["positive_well_fraction"]
        >= gates_v3["positive_well_fraction_ge"]
    ),
    "shuffle": bool(
        shuffled_metrics_v3["rmse"] - test_nonzero_v3["rmse"]
        >= gates_v3["shuffle_degradation_ge"]
    ),
    "no_alignment": bool(
        no_alignment_metrics_v3["rmse"] - test_nonzero_v3["rmse"]
        >= gates_v3["no_alignment_degradation_ge"]
    ),
}
strict_pass_v3 = bool(all(checks_v3.values()))
verdict_v3 = _jsonable_v3(
    {
        "experiment": EXPERIMENT_V3,
        "stage": "learned_completewell_sparse_candidate_gateway",
        "sample_unit": "one complete physical well",
        "centre": "fold-clean strong OOF",
        "uses_neighbor": False,
        "uses_cp12_gold": False,
        "candidate_generation_uses_truth": False,
        "protocol": {
            "train_folds": list(TRAIN_FOLDS_V3),
            "epoch_and_decoder_selection_fold": VALID_FOLD_V3,
            "frozen_test_fold": TEST_FOLD_V3,
            "best_epoch": int(best_v3["epoch"]),
            "deployment_decoder": {
                "temperature": float(selected_v3["temperature"]),
                "dose": float(selected_v3["dose"]),
            },
            "best_nonzero_decoder": {
                "temperature": float(selected_nonzero_v3["temperature"]),
                "dose": float(selected_nonzero_v3["dose"]),
            },
        },
        "model": {
            "parameters": parameters_v3,
            "candidate_features": feature_count_v3,
            "context_features": context_count_v3,
            "candidate_states": MAX_STATES_V3,
            "node_stride": NODE_STRIDE_V3,
        },
        "capacity": capacity_verdict_v3,
        "fold1_union_oracle": valid_oracle_v3,
        "fold2_union_oracle": test_oracle_v3,
        "fold2_deployment": test_metrics_v3,
        "fold2_best_nonzero": test_nonzero_v3,
        "controls": {
            "shuffled_alignment": shuffled_metrics_v3,
            "no_alignment": no_alignment_metrics_v3,
            "shuffle_degradation": shuffled_metrics_v3["rmse"] - test_nonzero_v3["rmse"],
            "no_alignment_degradation": no_alignment_metrics_v3["rmse"] - test_nonzero_v3["rmse"],
        },
        "gates": gates_v3,
        "checks": checks_v3,
        "strict_pass": strict_pass_v3,
        "checkpoint": str(checkpoint_v3),
        "elapsed_minutes": (time.time() - started_v3) / 60.0,
        "next": (
            "PASS -> confirm with a second outer fold before five-fold training"
            if strict_pass_v3
            else "FAIL -> learned ranking cannot consume the 3.71-ft candidate ceiling"
        ),
    }
)
(OUT_DIR_V3 / "verdict.json").write_text(
    json.dumps(verdict_v3, indent=2), encoding="utf-8"
)
print("\nFINAL VERDICT")
print(json.dumps(verdict_v3, indent=2))
print("saved:", OUT_DIR_V3)
