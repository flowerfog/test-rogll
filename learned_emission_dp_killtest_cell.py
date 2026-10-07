"""Fold-honest learned-emission base-band DP kill-test.

Run this after ``learned_gr_matcher_v0_train_cell.py``.  Each train well is
decoded with the matcher checkpoint that did not train on that well.  Truth is
not read until every path has been produced and cached.

The matcher v0 was trained only on negatives within +/-64 ft of truth.  Its
unbounded full-grid run exposed severe far-away false peaks.  This frozen
follow-up restricts each sampled row to the deployable base prediction +/-64
ft, matching the training support without reading truth.  A denser or wider
run is justified only if this path-level test passes.
"""

from pathlib import Path
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit
from scipy.signal import savgol_filter
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Frozen v0 configuration.
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
BASE_DIR = OOF_ROOT / "multicut_fleongg_v0_k3_nospatial_r768_lgb1"
BASE_OOF = BASE_DIR / "multicut_v0_oof.pkl"
MATCHER_DIR = OOF_ROOT / "learned_gr_matcher_v0_win65_rows128_offsets4to64"
MODEL_DIR = MATCHER_DIR / "models"

CACHE_VERSION = "v1_baseband64_step4_stride8"
CACHE_DIR = OOF_ROOT / f"learned_emission_dp_{CACHE_VERSION}"
WELL_CACHE = CACHE_DIR / "wells"

WINDOW_LENGTH = 65
HALF_WINDOW = WINDOW_LENGTH // 2
STATE_STEP_FT = 4.0
ROW_STRIDE = 8
MAX_RESIDUAL_JUMP_STATES = 3
BASE_BAND_FT = 64.0
ENCODER_BATCH = 2048
PAIR_ROW_CHUNK = 48

# Only transition/emission ratios differ.  This is a fixed three-point
# robustness check, not a post-hoc parameter search.
DP_CONFIGS = {
    "learned_loose": {"transition": 0.025, "emission_weight": 1.00},
    "learned_mid": {"transition": 0.100, "emission_weight": 1.00},
    "learned_strict": {"transition": 0.400, "emission_weight": 1.00},
}
BLEND_WEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.40)
RANDOM_STATE = 20260713

for directory in (CACHE_DIR, WELL_CACHE):
    directory.mkdir(parents=True, exist_ok=True)

if not BASE_OOF.exists():
    raise FileNotFoundError(f"Missing honest base OOF: {BASE_OOF}")
missing_checkpoints = [
    str(MODEL_DIR / f"matcher_fold{fold}.pt")
    for fold in range(1, 6)
    if not (MODEL_DIR / f"matcher_fold{fold}.pt").exists()
]
if missing_checkpoints:
    raise FileNotFoundError(
        "Run learned_gr_matcher_v0_train_cell.py first. Missing: "
        + repr(missing_checkpoints)
    )

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

print(
    {
        "experiment": "learned_emission_dp_killtest",
        "device": str(DEVICE),
        "base_oof": str(BASE_OOF),
        "matcher_dir": str(MATCHER_DIR),
        "state_step_ft": STATE_STEP_FT,
        "row_stride": ROW_STRIDE,
        "base_band_ft": BASE_BAND_FT,
        "dp_configs": DP_CONFIGS,
        "cache": str(CACHE_DIR),
    }
)


def _rmse(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _fill_1d(values):
    return (
        pd.Series(np.asarray(values, float))
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _normalise_window(values, mask):
    values = np.asarray(values, np.float32)
    mask = np.asarray(mask, bool)
    output = np.zeros_like(values, dtype=np.float32)
    finite_values = values[mask & np.isfinite(values)]
    if len(finite_values) < 4:
        return output
    median = float(np.median(finite_values))
    scale = float(1.4826 * np.median(np.abs(finite_values - median)))
    if scale < 1e-3:
        scale = float(np.std(finite_values) + 1e-3)
    output[mask] = np.clip((values[mask] - median) / scale, -6.0, 6.0)
    return output


def _prepare_typewell(typewell):
    frame = typewell[["TVT", "GR"]].copy()
    frame = frame[np.isfinite(frame["TVT"])].sort_values("TVT")
    frame = frame.groupby("TVT", as_index=False)["GR"].mean()
    tvt = frame["TVT"].to_numpy(float)
    gr = _fill_1d(frame["GR"].to_numpy(float))
    if len(tvt) < WINDOW_LENGTH or np.ptp(tvt) < 200:
        raise RuntimeError("typewell coverage is too short")
    return tvt, gr


# ---------------------------------------------------------------------------
# Exact architecture used by learned_gr_matcher_v0_train_cell.py.
# ---------------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        padding = 2 * dilation
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(8, channels),
        )

    def forward(self, values):
        return F.silu(values + self.network(values))


class WindowEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 48, kernel_size=7, padding=3),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            ResidualConvBlock(48, 1),
            ResidualConvBlock(48, 2),
            ResidualConvBlock(48, 4),
        )
        self.output = nn.Sequential(
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Linear(96, embedding_dim),
        )

    def forward(self, values):
        hidden = self.blocks(self.stem(values))
        pooled = torch.cat([hidden.mean(dim=-1), hidden.amax(dim=-1)], dim=1)
        return F.normalize(self.output(pooled), dim=1)


class GRWindowMatcher(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.horizontal_encoder = WindowEncoder(embedding_dim)
        self.typewell_encoder = WindowEncoder(embedding_dim)
        self.scorer = nn.Sequential(
            nn.Linear(4 * embedding_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )


def _load_model(fold):
    path = MODEL_DIR / f"matcher_fold{int(fold)}.pt"
    model = GRWindowMatcher().to(DEVICE)
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def _horizontal_windows(horizontal_gr, positions):
    horizontal_gr = np.asarray(horizontal_gr, float)
    windows = np.zeros((len(positions), 2, WINDOW_LENGTH), dtype=np.float32)
    valid_fraction = np.zeros(len(positions), dtype=np.float32)
    offsets = np.arange(-HALF_WINDOW, HALF_WINDOW + 1)
    for index, position in enumerate(positions):
        rows = int(position) + offsets
        inside = (rows >= 0) & (rows < len(horizontal_gr))
        clipped = np.clip(rows, 0, len(horizontal_gr) - 1)
        values = horizontal_gr[clipped]
        mask = inside & np.isfinite(values)
        windows[index, 0] = _normalise_window(values, mask)
        windows[index, 1] = mask.astype(np.float32)
        valid_fraction[index] = float(np.mean(mask))
    return windows, valid_fraction


def _typewell_windows(tw_tvt, tw_gr, state_values):
    relative = np.arange(-HALF_WINDOW, HALF_WINDOW + 1, dtype=np.float32)
    queries = state_values[:, None] + relative[None, :]
    masks = (queries >= tw_tvt[0]) & (queries <= tw_tvt[-1])
    values = np.interp(queries.reshape(-1), tw_tvt, tw_gr).reshape(
        len(state_values), WINDOW_LENGTH
    )
    windows = np.zeros((len(state_values), 2, WINDOW_LENGTH), dtype=np.float32)
    for state in range(len(state_values)):
        windows[state, 0] = _normalise_window(values[state], masks[state])
    windows[:, 1] = masks.astype(np.float32)
    return windows, masks.mean(axis=1).astype(np.float32)


@torch.no_grad()
def _encode_windows(encoder, windows):
    parts = []
    for start in range(0, len(windows), ENCODER_BATCH):
        batch = torch.from_numpy(windows[start : start + ENCODER_BATCH]).to(
            DEVICE, non_blocking=True
        )
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            parts.append(encoder(batch).float().cpu())
    return torch.cat(parts, dim=0).to(DEVICE, non_blocking=True)


def _pair_logits(model, horizontal, typewell):
    horizontal = horizontal[:, None, :].expand(-1, typewell.shape[1], -1)
    pair = torch.cat(
        [
            horizontal,
            typewell,
            torch.abs(horizontal - typewell),
            horizontal * typewell,
        ],
        dim=-1,
    )
    return model.scorer(pair).squeeze(-1)


@torch.no_grad()
def _learned_emission(model, horizontal_windows, typewell_windows):
    horizontal_embedding = _encode_windows(
        model.horizontal_encoder, horizontal_windows
    )
    typewell_forward = _encode_windows(model.typewell_encoder, typewell_windows)
    typewell_reverse = _encode_windows(
        model.typewell_encoder, typewell_windows[:, :, ::-1].copy()
    )

    score_parts = []
    for start in range(0, len(horizontal_embedding), PAIR_ROW_CHUNK):
        horizontal = horizontal_embedding[start : start + PAIR_ROW_CHUNK]
        rows = len(horizontal)
        forward = typewell_forward[None, :, :].expand(rows, -1, -1)
        reverse = typewell_reverse[None, :, :].expand(rows, -1, -1)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            score = torch.maximum(
                _pair_logits(model, horizontal, forward),
                _pair_logits(model, horizontal, reverse),
            )
        score_parts.append(score.float().cpu().numpy())

    scores = np.concatenate(score_parts, axis=0).astype(np.float32)
    # Additive row constants do not affect Viterbi.  Max-score produces a
    # non-negative cost while retaining the matcher's learned logit scale.
    emission = np.max(scores, axis=1, keepdims=True) - scores
    return emission.astype(np.float32), scores


@njit
def _viterbi_decode(
    emission,
    state_values,
    expected_changes,
    transition_penalty,
    max_residual_jump,
    anchor_state,
    state_step,
):
    n_rows, n_states = emission.shape
    large = np.float32(1e30)
    previous = np.full(n_states, large, dtype=np.float32)
    previous[anchor_state] = np.float32(0.0)
    backpointer = np.empty((n_rows, n_states), dtype=np.int16)

    for row in range(n_rows):
        current = np.full(n_states, large, dtype=np.float32)
        expected_change = expected_changes[row]
        expected_state_shift = int(np.round(expected_change / state_step))
        for state in range(n_states):
            expected_old_state = state - expected_state_shift
            start = max(0, expected_old_state - max_residual_jump)
            stop = min(n_states, expected_old_state + max_residual_jump + 1)
            best_cost = large
            best_previous = state
            for old_state in range(start, stop):
                actual_change = state_values[state] - state_values[old_state]
                residual_change = actual_change - expected_change
                candidate_cost = (
                    previous[old_state]
                    + transition_penalty * residual_change * residual_change
                )
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_previous = old_state
            current[state] = best_cost + emission[row, state]
            backpointer[row, state] = best_previous
        previous = current

    end_state = int(np.argmin(previous))
    states = np.empty(n_rows, dtype=np.int16)
    state = end_state
    for row in range(n_rows - 1, -1, -1):
        states[row] = state
        state = int(backpointer[row, state])
    return states, float(previous[end_state])


def _decode_well(wid, fold, model, well_base):
    cache_path = WELL_CACHE / f"{wid}.pkl"
    sample_cache_path = WELL_CACHE / f"{wid}_samples.pkl"
    if cache_path.exists() and sample_cache_path.exists():
        return pd.read_pickle(cache_path), pd.read_pickle(sample_cache_path), None

    horizontal_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{wid}__typewell.csv"
    try:
        hw = pd.read_csv(horizontal_path, usecols=["Z", "GR", "TVT_input"])
        tw = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        tvt_input = hw["TVT_input"].to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        evaluation = np.flatnonzero(~np.isfinite(tvt_input))
        if len(known) < 30 or len(evaluation) < 64:
            raise RuntimeError("insufficient known/eval rows")
        anchor_position = int(known[-1])
        if int(evaluation[0]) != anchor_position + 1:
            raise RuntimeError("TVT_input is not a contiguous prefix")

        base_by_row = well_base.set_index("row_idx")["pred_multi_cut"]
        if not np.all(np.isin(evaluation, base_by_row.index.to_numpy(int))):
            raise RuntimeError("base OOF does not cover every official eval row")

        sampled = evaluation[::ROW_STRIDE]
        if sampled[-1] != evaluation[-1]:
            sampled = np.append(sampled, evaluation[-1])

        tw_tvt, tw_gr = _prepare_typewell(tw)
        anchor_tvt = float(tvt_input[anchor_position])
        lower_step = int(np.floor((float(tw_tvt[0]) - anchor_tvt) / STATE_STEP_FT))
        upper_step = int(np.ceil((float(tw_tvt[-1]) - anchor_tvt) / STATE_STEP_FT))
        state_values = (
            anchor_tvt
            + np.arange(lower_step, upper_step + 1, dtype=np.float32) * STATE_STEP_FT
        ).astype(np.float32)
        anchor_state = int(-lower_step)
        if not (0 <= anchor_state < len(state_values)):
            raise RuntimeError("anchor is outside typewell TVT range")
        if len(state_values) >= np.iinfo(np.int16).max:
            raise RuntimeError("state grid is too large for int16 backpointers")

        h_windows, h_valid = _horizontal_windows(hw["GR"].to_numpy(float), sampled)
        tw_windows, tw_valid = _typewell_windows(tw_tvt, tw_gr, state_values)
        emission, scores = _learned_emission(model, h_windows, tw_windows)

        # The matcher was trained on complete typewell windows and horizontal
        # windows with >=35% observed GR.  Do not let out-of-domain windows
        # create confident path jumps.
        edge_penalty = 2.0 * np.maximum(0.0, 0.95 - tw_valid) ** 2
        emission += edge_penalty[None, :].astype(np.float32)
        weak_rows = h_valid < 0.20
        if np.any(weak_rows):
            emission[weak_rows] = edge_penalty[None, :]

        base_sampled = base_by_row.loc[sampled].to_numpy(float)
        within_base_band = (
            np.abs(state_values[None, :] - base_sampled[:, None]) <= BASE_BAND_FT
        )
        if np.any(np.sum(within_base_band, axis=1) == 0):
            raise RuntimeError("base band contains no state for at least one row")
        emission[~within_base_band] = np.float32(1e6)

        z = hw["Z"].to_numpy(float)
        previous_positions = np.concatenate(([anchor_position], sampled[:-1]))
        expected_changes = (-(z[sampled] - z[previous_positions])).astype(np.float32)
        geometry_center = anchor_tvt - (z[evaluation] - float(z[anchor_position]))

        frame = pd.DataFrame(
            {
                "id": [f"{wid}_{position}" for position in evaluation],
                "well": wid,
                "row_idx": evaluation.astype(int),
                "fold": int(fold),
                "geometry_center": geometry_center,
                "typewell_tvt_min": float(tw_tvt[0]),
                "typewell_tvt_max": float(tw_tvt[-1]),
                "base_band_ft": BASE_BAND_FT,
            }
        )
        masked_scores = np.where(within_base_band, scores, -np.inf)
        sample_frame = pd.DataFrame(
            {
                "id": [f"{wid}_{position}" for position in sampled],
                "well": wid,
                "row_idx": sampled.astype(int),
                "fold": int(fold),
                "horizontal_valid_frac": h_valid,
                "base_tvt": base_sampled,
                "local_best_tvt": state_values[np.argmax(masked_scores, axis=1)],
                "local_peak_margin": np.partition(masked_scores, -2, axis=1)[:, -1]
                - np.partition(masked_scores, -2, axis=1)[:, -2],
            }
        )

        x_anchor = np.concatenate(([anchor_position], sampled)).astype(float)
        for name, config in DP_CONFIGS.items():
            states, path_cost = _viterbi_decode(
                emission * np.float32(config["emission_weight"]),
                state_values,
                expected_changes,
                np.float32(config["transition"]),
                MAX_RESIDUAL_JUMP_STATES,
                anchor_state,
                np.float32(STATE_STEP_FT),
            )
            sampled_path = state_values[states].astype(float)
            path = np.interp(
                evaluation.astype(float),
                x_anchor,
                np.concatenate(([anchor_tvt], sampled_path)),
            )
            window = min(61, len(path))
            if window % 2 == 0:
                window -= 1
            if window >= 5:
                path = savgol_filter(path, window, 3)
            frame[name] = path
            frame[f"{name}_path_cost_per_sample"] = path_cost / max(1, len(sampled))
            sample_frame[f"{name}_sampled_tvt"] = sampled_path

        frame["mean_horizontal_valid_frac"] = float(np.mean(h_valid))
        frame.to_pickle(cache_path)
        sample_frame.to_pickle(sample_cache_path)
        return frame, sample_frame, None
    except Exception as exc:
        return None, None, {"well": wid, "fold": int(fold), "reason": repr(exc)}


# ---------------------------------------------------------------------------
# Decode fold by fold so every OOF well uses its held-out matcher checkpoint.
# ---------------------------------------------------------------------------
base = pd.read_pickle(BASE_OOF)
required = {
    "id",
    "well",
    "row_idx",
    "fold",
    "tvt_true",
    "pred_single_cut",
    "pred_multi_cut",
}
missing = sorted(required - set(base.columns))
if missing:
    raise RuntimeError(f"Missing base OOF columns: {missing}")
base["well"] = base["well"].astype(str)
base["id"] = base["id"].astype(str)

fold_by_well = base.groupby("well", sort=False)["fold"].first().astype(int)
if not fold_by_well.between(1, 5).all():
    raise RuntimeError("OOF fold labels must be 1..5")

started = time.time()
decoded_parts = []
sample_parts = []
failures = []
completed = 0
total_wells = len(fold_by_well)

for fold in range(1, 6):
    model = _load_model(fold)
    fold_wells = sorted(fold_by_well[fold_by_well == fold].index.astype(str))
    progress = tqdm(fold_wells, desc=f"learned DP fold {fold}")
    for wid in progress:
        well_base = base.loc[
            base["well"].eq(wid), ["row_idx", "pred_multi_cut"]
        ].copy()
        frame, sample_frame, failure = _decode_well(wid, fold, model, well_base)
        if frame is not None:
            decoded_parts.append(frame)
            sample_parts.append(sample_frame)
        if failure is not None:
            failures.append(failure)
        completed += 1
        if completed % 25 == 0:
            elapsed_minutes = (time.time() - started) / 60.0
            rate = completed / max(elapsed_minutes, 1e-9)
            eta = (total_wells - completed) / max(rate, 1e-9)
            print(
                f"learned DP {completed}/{total_wells} ok={len(decoded_parts)} "
                f"elapsed={elapsed_minutes:.1f}m eta={eta:.1f}m",
                flush=True,
            )
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

if not decoded_parts:
    raise RuntimeError("No learned-emission DP paths decoded")

decoded = pd.concat(decoded_parts, ignore_index=True)
sampled_diagnostics = pd.concat(sample_parts, ignore_index=True)
merged = base.merge(
    decoded.drop(columns=["fold"]),
    on=["id", "well", "row_idx"],
    how="inner",
    validate="one_to_one",
)
print("merged rows/wells:", merged.shape, merged["well"].nunique())
if len(merged) < 0.98 * len(base):
    warnings.warn(f"Only matched {len(merged)} / {len(base)} OOF rows")


# ---------------------------------------------------------------------------
# Path-level evaluation.  Truth enters for the first time below this line.
# ---------------------------------------------------------------------------
y = merged["tvt_true"].to_numpy(float)
base_prediction = merged["pred_multi_cut"].to_numpy(float)
base_error = base_prediction - y
base_rmse = _rmse(y, base_prediction)

summary_rows = []
fold_rows = []
for model_name in DP_CONFIGS:
    raw_prediction = merged[model_name].to_numpy(float)
    raw_error = raw_prediction - y
    model_rmse = _rmse(y, raw_prediction)
    cosine = float(
        np.dot(base_error, raw_error)
        / max(np.linalg.norm(base_error) * np.linalg.norm(raw_error), 1e-12)
    )
    for blend_weight in (1.0,) + BLEND_WEIGHTS:
        prediction = (
            raw_prediction
            if blend_weight == 1.0
            else (1.0 - blend_weight) * base_prediction
            + blend_weight * raw_prediction
        )
        row = {
            "model": model_name,
            "blend_weight": float(blend_weight),
            "base_rmse": base_rmse,
            "rmse": _rmse(y, prediction),
            "gain": base_rmse - _rmse(y, prediction),
            "error_cosine": cosine,
            "standalone_ratio": model_rmse / base_rmse,
        }
        fold_gains = []
        for fold, index in merged.groupby("fold", sort=True).groups.items():
            positions = merged.index.get_indexer(index)
            fold_base = _rmse(y[positions], base_prediction[positions])
            fold_score = _rmse(y[positions], prediction[positions])
            gain = fold_base - fold_score
            fold_gains.append(gain)
            fold_rows.append(
                {
                    "model": model_name,
                    "blend_weight": float(blend_weight),
                    "fold": int(fold),
                    "rows": len(positions),
                    "base_rmse": fold_base,
                    "rmse": fold_score,
                    "gain": gain,
                }
            )
        row["mean_fold_gain"] = float(np.mean(fold_gains))
        row["min_fold_gain"] = float(np.min(fold_gains))
        row["positive_folds"] = int(np.sum(np.asarray(fold_gains) > 0))
        summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
fold_summary = pd.DataFrame(fold_rows)

# Freeze top40 using base SSE only.
bywell_base = []
for wid, group in merged.groupby("well", sort=False):
    truth = group["tvt_true"].to_numpy(float)
    prediction = group["pred_multi_cut"].to_numpy(float)
    bywell_base.append(
        {"well": wid, "base_sse": float(np.sum((prediction - truth) ** 2))}
    )
top40_wells = set(
    pd.DataFrame(bywell_base).nlargest(40, "base_sse")["well"].astype(str)
)
top40_mask = merged["well"].isin(top40_wells).to_numpy()
top40_base_rmse = _rmse(y[top40_mask], base_prediction[top40_mask])
top40_gains = []
for row in summary.itertuples(index=False):
    raw = merged[row.model].to_numpy(float)
    prediction = (
        raw
        if row.blend_weight == 1.0
        else (1.0 - row.blend_weight) * base_prediction + row.blend_weight * raw
    )
    top40_gains.append(
        top40_base_rmse - _rmse(y[top40_mask], prediction[top40_mask])
    )
summary["top40_gain"] = top40_gains

state_coverage = float(
    np.mean(
        (y >= merged["typewell_tvt_min"].to_numpy(float))
        & (y <= merged["typewell_tvt_max"].to_numpy(float))
    )
)
base_band_coverage = float(
    np.mean(np.abs(y - base_prediction) <= BASE_BAND_FT)
)

# A path must be a plausible standalone model before a tiny blend is accepted.
blend_mask = summary["blend_weight"] < 1.0
summary["gate"] = False
summary.loc[blend_mask, "gate"] = (
    (summary.loc[blend_mask, "standalone_ratio"] <= 1.25)
    & (summary.loc[blend_mask, "error_cosine"] <= 0.90)
    & (summary.loc[blend_mask, "gain"] >= 0.05)
    & (summary.loc[blend_mask, "positive_folds"] >= 4)
    & (summary.loc[blend_mask, "min_fold_gain"] > -0.10)
    & (summary.loc[blend_mask, "top40_gain"] >= 0.0)
)
summary = summary.sort_values(["gate", "gain"], ascending=[False, False])
strict_pass = bool(
    summary["gate"].any()
    and state_coverage >= 0.99
    and base_band_coverage >= 0.99
)

# Local dense-grid diagnostic: this is secondary to path RMSE, but catches a
# model/grid mismatch before interpreting a failed DP transition prior.
local = sampled_diagnostics.merge(
    base[["id", "well", "tvt_true"]],
    on=["id", "well"],
    how="inner",
    validate="one_to_one",
)
local_abs = np.abs(local["local_best_tvt"] - local["tvt_true"])
local_metrics = {
    "samples": int(len(local)),
    "within4": float(np.mean(local_abs <= 4.0)),
    "within8": float(np.mean(local_abs <= 8.0)),
    "median_abs": float(np.median(local_abs)),
    "p90_abs": float(np.quantile(local_abs, 0.90)),
}

print("\nSUMMARY")
display(summary)
print("\nFOLDS")
display(fold_summary)
print("\nLOCAL MATCHER ON DENSE GRID", local_metrics)
print(
    "\nKILL-TEST",
    {
        "base_rmse": base_rmse,
        "state_coverage": state_coverage,
        "base_band_truth_coverage": base_band_coverage,
        "decoded_wells": int(merged["well"].nunique()),
        "failed_wells": int(len(failures)),
        "runtime_hours": (time.time() - started) / 3600.0,
        "strict_pass": strict_pass,
        "verdict": (
            "PASS -> refine grid and add learned path to fusion"
            if strict_pass
            else "FAIL -> keep matcher as feature; stop learned DP path"
        ),
    },
)

summary.to_csv(CACHE_DIR / "learned_emission_dp_summary.csv", index=False)
fold_summary.to_csv(CACHE_DIR / "learned_emission_dp_folds.csv", index=False)
merged.to_pickle(CACHE_DIR / "learned_emission_dp_oof.pkl")
sampled_diagnostics.to_pickle(CACHE_DIR / "learned_emission_dp_samples.pkl")
pd.DataFrame(failures).to_csv(CACHE_DIR / "learned_emission_dp_failures.csv", index=False)
(CACHE_DIR / "learned_emission_dp_verdict.json").write_text(
    json.dumps(
        {
            "base_rmse": base_rmse,
            "state_coverage": state_coverage,
            "base_band_truth_coverage": base_band_coverage,
            "decoded_wells": int(merged["well"].nunique()),
            "failed_wells": int(len(failures)),
            "runtime_hours": (time.time() - started) / 3600.0,
            "strict_pass": strict_pass,
            "local_metrics": local_metrics,
        },
        indent=2,
    )
)
print("saved:", CACHE_DIR)
