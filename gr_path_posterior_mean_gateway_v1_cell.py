"""E1 gateway: calibrated posterior mean over the existing path bank.

The sequence scorer and restricted 235-path bank already exist. Fold 5 is
used to calibrate top-M, softmax temperature, shrink, and cap. That frozen
recipe is then evaluated once on untouched fold 1. No Top-1 hard selection is
used in the primary candidate.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
MATCHER_DIR = OOF_ROOT / "gr_path_matcher_onefold_v0"
SCORER_DIR = OOF_ROOT / "gr_path_sequence_scorer_onefold_v0"
BANK_PATH = MATCHER_DIR / "restricted_path_bank.npz"
MODEL_PATH = SCORER_DIR / "best_sequence_scorer.pt"
SEQUENCE_DIR = SCORER_DIR / "sequence_cache"
PATH_FEATURES_PATH = MATCHER_DIR / "path_features.pkl"
OUT_DIR = OOF_ROOT / "gr_path_posterior_mean_gateway_v1"
SCORE_CACHE_DIR = OUT_DIR / "score_cache"

POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)

CALIBRATION_FOLD = 5
VALID_FOLD = 1
TOP_MS = (2, 4, 8, 16, 32, 64, 235)
TEMPERATURES = (0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0)
FORCE_ZERO_OPTIONS = (False, True)
ALPHAS = tuple(np.round(np.arange(0.0, 1.0001, 0.025), 3))
APPLIED_CAPS = (6.0, 12.0, 24.0, np.inf)

PASS_GAIN = 0.20
PASS_POSITIVE_WELLS = 0.55
PASS_MEDIAN_WELL_GAIN = 0.0


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                5,
                padding=2 * dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
        )

    def forward(self, values):
        return F.silu(values + self.net(values))


class PathSequenceScorer(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(inputs, 48, 7, padding=3, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(48, 1),
            ResidualBlock(48, 2),
            ResidualBlock(48, 4),
            ResidualBlock(48, 8),
        )
        self.head = nn.Sequential(
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, values):
        hidden = self.blocks(self.stem(values))
        pooled = torch.cat([hidden.mean(-1), hidden.amax(-1)], dim=1)
        return self.head(pooled).squeeze(1)


def _first_existing(paths):
    return next((path for path in paths if path.exists()), None)


def _rmse(truth, prediction):
    truth = np.asarray(truth, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _softmax(values, temperature):
    values = np.asarray(values, np.float64) / float(temperature)
    values -= np.max(values)
    weights = np.exp(np.clip(values, -60.0, 0.0))
    return weights / max(float(weights.sum()), 1e-12)


def _recipe_name(top_m, temperature, force_zero):
    return (
        f"m{int(top_m):03d}_t{temperature:.2f}_"
        f"{'forcezero' if force_zero else 'plain'}"
    )


def _posterior_move(payload, recipe, return_diagnostics=False):
    scores = payload["scores"]
    order = np.argsort(scores)[::-1]
    top_m = min(int(recipe["top_m"]), len(order))
    indices = order[:top_m].tolist()
    if recipe["force_zero"] and payload["zero_index"] not in indices:
        indices.append(payload["zero_index"])
    indices = np.asarray(indices, np.int32)
    weights = _softmax(scores[indices], float(recipe["temperature"]))
    combined_template = weights @ payload["scaled_templates"][indices]
    move = np.interp(
        payload["progress"],
        payload["progress_grid"],
        combined_template,
    )
    if not return_diagnostics:
        return move.astype(np.float32)
    zero_position = np.flatnonzero(indices == payload["zero_index"])
    zero_weight = (
        float(weights[zero_position[0]]) if len(zero_position) else 0.0
    )
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))))
    diagnostics = {
        "candidate_count": len(indices),
        "top_weight": float(np.max(weights)),
        "zero_weight": zero_weight,
        "entropy": entropy,
        "effective_candidates": float(1.0 / np.sum(weights**2)),
        "score_margin_top2": float(scores[order[0]] - scores[order[1]]),
        "mean_abs_move": float(np.mean(np.abs(move))),
        "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
    }
    return move.astype(np.float32), diagnostics


for path in (
    BANK_PATH,
    MODEL_PATH,
    SEQUENCE_DIR,
    PATH_FEATURES_PATH,
    DATA_ROOT / "train",
):
    if not path.exists():
        raise FileNotFoundError(
            f"Required E1 artifact is missing: {path}. Restore the old "
            "path-bank/sequence-scorer outputs before running this gateway."
        )

pool_path = _first_existing(POOL_CANDIDATES)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
for directory in (OUT_DIR, SCORE_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

with np.load(BANK_PATH, allow_pickle=False) as bank:
    templates = bank["templates"].astype(np.float32)
    length_alphas = bank["length_alphas"].astype(np.float32)
    candidate_names = bank["names"].astype(str)
    progress_grid = bank["progress_grid"].astype(np.float32)
    median_length = float(bank["median_length"][0])
    bank_cap = float(bank["cap"][0])
if len(candidate_names) != len(templates):
    raise RuntimeError("candidate bank name/template mismatch")
zero_matches = np.flatnonzero(candidate_names == "zero")
if len(zero_matches) != 1:
    raise RuntimeError("candidate bank must contain exactly one zero path")
zero_index = int(zero_matches[0])

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
channels = list(checkpoint["channels"])
model = PathSequenceScorer(len(channels))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

path_features = pd.read_pickle(PATH_FEATURES_PATH)
path_features["well"] = path_features["well"].astype(str)
fold_map = (
    path_features.groupby("well", sort=False)["fold"].first().astype(int).to_dict()
)

header = pd.read_csv(pool_path, nrows=0).columns.tolist()
usecols = [
    column
    for column in ("id", "well", "row_idx", "tvt_true", "strong")
    if column in header
]
pool = pd.read_csv(
    pool_path, usecols=usecols, dtype={"id": str, "well": str}
)
if not {"id", "tvt_true", "strong"}.issubset(pool.columns):
    raise RuntimeError(f"pool columns are incomplete: {pool.columns.tolist()}")
pool["id"] = pool["id"].astype(str)
if "well" not in pool.columns:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool.columns:
    pool["row_idx"] = (
        pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
    )
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pool["row_idx"].astype(np.int32)
pool_groups = {
    str(well): frame.sort_values("row_idx").reset_index(drop=True)
    for well, frame in pool.groupby("well", sort=False)
}

cache_wells = {
    path.stem for path in SEQUENCE_DIR.glob("*.npz") if path.stem in pool_groups
}
calibration_wells = sorted(
    well for well in cache_wells if fold_map.get(well) == CALIBRATION_FOLD
)
valid_wells = sorted(
    well for well in cache_wells if fold_map.get(well) == VALID_FOLD
)
if len(calibration_wells) < 100 or len(valid_wells) < 100:
    raise RuntimeError(
        f"insufficient scorer cache: calibration={len(calibration_wells)}, "
        f"valid={len(valid_wells)}"
    )

print(
    {
        "experiment": "gr_path_posterior_mean_gateway_v1",
        "pool": str(pool_path),
        "calibration_fold": CALIBRATION_FOLD,
        "valid_fold": VALID_FOLD,
        "calibration_wells": len(calibration_wells),
        "valid_wells": len(valid_wells),
        "candidates": len(candidate_names),
        "recipes": len(TOP_MS) * len(TEMPERATURES) * len(FORCE_ZERO_OPTIONS),
    }
)


def _load_payload(well):
    score_path = SCORE_CACHE_DIR / f"{well}.npy"
    if score_path.exists():
        scores = np.load(score_path).astype(np.float32)
    else:
        with np.load(SEQUENCE_DIR / f"{well}.npz", allow_pickle=False) as data:
            sequence = torch.from_numpy(data["sequence"].astype(np.float32))
        with torch.no_grad():
            scores = model(sequence).cpu().numpy().astype(np.float32)
        np.save(score_path, scores)
    if len(scores) != len(candidate_names):
        raise RuntimeError(f"{well}: score count mismatch")

    frame = pool_groups[well]
    horizontal = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__horizontal_well.csv", usecols=["MD"]
    )
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
    rows = frame["row_idx"].to_numpy(np.int32)
    anchor = int(rows.min() - 1)
    if anchor < 0:
        raise RuntimeError(f"{well}: invalid anchor")
    eval_span = float(max(md[rows.max()] - md[anchor], 1.0))
    progress = np.clip((md[rows] - md[anchor]) / eval_span, 0.0, 1.0)
    length_ratio = max(eval_span / max(median_length, 1.0), 0.1)
    scales = np.power(length_ratio, length_alphas).astype(np.float32)
    scaled_templates = np.clip(
        templates * scales[:, None], -bank_cap, bank_cap
    ).astype(np.float32)
    return {
        "well": well,
        "scores": scores,
        "progress": progress,
        "progress_grid": progress_grid,
        "scaled_templates": scaled_templates,
        "zero_index": zero_index,
        "truth": frame["tvt_true"].to_numpy(float),
        "strong": frame["strong"].to_numpy(float),
        "ids": frame["id"].astype(str).to_numpy(),
        "row_idx": rows,
    }


recipes = [
    {
        "name": _recipe_name(top_m, temperature, force_zero),
        "top_m": int(top_m),
        "temperature": float(temperature),
        "force_zero": bool(force_zero),
    }
    for top_m in TOP_MS
    for temperature in TEMPERATURES
    for force_zero in FORCE_ZERO_OPTIONS
]

# Stage 1: use analytic unbounded shrink to rank posterior recipes on fold 5.
calibration_stats = {
    recipe["name"]: {
        "rows": 0,
        "base_sse": 0.0,
        "move_sse": 0.0,
        "error_move": 0.0,
    }
    for recipe in recipes
}
started = time.time()
for well in tqdm(calibration_wells, desc="calibrate posterior recipes"):
    payload = _load_payload(well)
    error = payload["strong"] - payload["truth"]
    base_sse = float(np.dot(error, error))
    for recipe in recipes:
        move = _posterior_move(payload, recipe)
        stats = calibration_stats[recipe["name"]]
        stats["rows"] += len(error)
        stats["base_sse"] += base_sse
        stats["move_sse"] += float(np.dot(move, move))
        stats["error_move"] += float(np.dot(error, move))

recipe_rows = []
recipe_by_name = {recipe["name"]: recipe for recipe in recipes}
for name, stats in calibration_stats.items():
    analytic_alpha = float(
        np.clip(
            -stats["error_move"] / max(stats["move_sse"], 1e-12),
            0.0,
            1.0,
        )
    )
    base_rmse = math.sqrt(stats["base_sse"] / stats["rows"])
    posterior_sse = (
        stats["base_sse"]
        + 2.0 * analytic_alpha * stats["error_move"]
        + analytic_alpha**2 * stats["move_sse"]
    )
    posterior_rmse = math.sqrt(max(posterior_sse / stats["rows"], 0.0))
    recipe = recipe_by_name[name]
    recipe_rows.append(
        {
            **recipe,
            "rows": stats["rows"],
            "base_rmse": base_rmse,
            "analytic_alpha": analytic_alpha,
            "analytic_rmse": posterior_rmse,
            "analytic_gain": base_rmse - posterior_rmse,
        }
    )
recipe_report = pd.DataFrame(recipe_rows).sort_values(
    ["analytic_rmse", "top_m", "temperature"]
).reset_index(drop=True)
recipe_report.to_csv(OUT_DIR / "calibration_recipe_scan.csv", index=False)
selected_recipe = recipe_by_name[str(recipe_report.iloc[0]["name"])]


def _collect_moves(wells, recipe, description):
    parts = []
    diagnostics = []
    for well in tqdm(wells, desc=description):
        payload = _load_payload(well)
        move, diag = _posterior_move(payload, recipe, return_diagnostics=True)
        parts.append(
            pd.DataFrame(
                {
                    "id": payload["ids"],
                    "well": well,
                    "row_idx": payload["row_idx"],
                    "truth": payload["truth"],
                    "strong": payload["strong"],
                    "posterior_move": move,
                }
            )
        )
        diagnostics.append({"well": well, **diag})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(diagnostics)


# Stage 2: exact alpha/cap calibration for the selected recipe on fold 5.
calibration_rows, calibration_diagnostics = _collect_moves(
    calibration_wells, selected_recipe, "freeze posterior dose"
)
truth = calibration_rows["truth"].to_numpy(float)
strong = calibration_rows["strong"].to_numpy(float)
move = calibration_rows["posterior_move"].to_numpy(float)
calibration_dose_rows = []
for alpha in ALPHAS:
    for cap in APPLIED_CAPS:
        applied = alpha * move
        if np.isfinite(cap):
            applied = np.clip(applied, -cap, cap)
        prediction = strong + applied
        calibration_dose_rows.append(
            {
                "alpha": float(alpha),
                "cap_ft": float(cap),
                "base_rmse": _rmse(truth, strong),
                "rmse": _rmse(truth, prediction),
                "gain": _rmse(truth, strong) - _rmse(truth, prediction),
                "mean_abs_applied_move": float(np.mean(np.abs(applied))),
                "p95_abs_applied_move": float(
                    np.quantile(np.abs(applied), 0.95)
                ),
            }
        )
dose_report = pd.DataFrame(calibration_dose_rows).sort_values(
    ["rmse", "alpha", "cap_ft"]
).reset_index(drop=True)
dose_report.to_csv(OUT_DIR / "calibration_dose_scan.csv", index=False)
frozen_alpha = float(dose_report.iloc[0]["alpha"])
frozen_cap = float(dose_report.iloc[0]["cap_ft"])


def _score_frozen(rows, alpha, cap):
    truth = rows["truth"].to_numpy(float)
    strong = rows["strong"].to_numpy(float)
    applied = alpha * rows["posterior_move"].to_numpy(float)
    if np.isfinite(cap):
        applied = np.clip(applied, -cap, cap)
    prediction = strong + applied
    work = rows[["id", "well", "row_idx", "truth", "strong"]].copy()
    work["posterior_move"] = rows["posterior_move"].to_numpy(float)
    work["applied_move"] = applied
    work["prediction"] = prediction
    bywell_rows = []
    for well, frame in work.groupby("well", sort=False):
        base_rmse = _rmse(frame["truth"], frame["strong"])
        candidate_rmse = _rmse(frame["truth"], frame["prediction"])
        bywell_rows.append(
            {
                "well": well,
                "rows": len(frame),
                "strong_rmse": base_rmse,
                "rmse": candidate_rmse,
                "gain": base_rmse - candidate_rmse,
            }
        )
    bywell = pd.DataFrame(bywell_rows)
    report = {
        "rows": len(work),
        "wells": work["well"].nunique(),
        "strong_rmse": _rmse(truth, strong),
        "rmse": _rmse(truth, prediction),
        "gain_vs_strong": _rmse(truth, strong) - _rmse(truth, prediction),
        "mean_well_gain": float(bywell["gain"].mean()),
        "median_well_gain": float(bywell["gain"].median()),
        "positive_well_fraction": float((bywell["gain"] > 0).mean()),
        "mean_abs_applied_move": float(np.mean(np.abs(applied))),
        "p95_abs_applied_move": float(np.quantile(np.abs(applied), 0.95)),
        "max_abs_applied_move": float(np.max(np.abs(applied))),
    }
    return report, bywell, work


calibration_report, calibration_bywell, calibration_output = _score_frozen(
    calibration_rows, frozen_alpha, frozen_cap
)

# Stage 3: one untouched application to fold 1.
valid_rows, valid_diagnostics = _collect_moves(
    valid_wells, selected_recipe, "apply frozen posterior to fold 1"
)
valid_report, valid_bywell, valid_output = _score_frozen(
    valid_rows, frozen_alpha, frozen_cap
)

calibration_bywell.to_csv(OUT_DIR / "calibration_frozen_bywell.csv", index=False)
valid_bywell.to_csv(OUT_DIR / "valid_frozen_bywell.csv", index=False)
calibration_diagnostics.to_csv(
    OUT_DIR / "calibration_posterior_diagnostics.csv", index=False
)
valid_diagnostics.to_csv(
    OUT_DIR / "valid_posterior_diagnostics.csv", index=False
)
calibration_output.to_pickle(OUT_DIR / "calibration_frozen_rows.pkl")
valid_output.to_pickle(OUT_DIR / "valid_frozen_rows.pkl")

strict_pass = bool(
    frozen_alpha > 0
    and valid_report["gain_vs_strong"] >= PASS_GAIN
    and valid_report["positive_well_fraction"] >= PASS_POSITIVE_WELLS
    and valid_report["median_well_gain"] > PASS_MEDIAN_WELL_GAIN
)
verdict = {
    "experiment": "gr_path_posterior_mean_gateway_v1",
    "source_bank": str(BANK_PATH),
    "source_scorer": str(MODEL_PATH),
    "calibration_fold": CALIBRATION_FOLD,
    "valid_fold": VALID_FOLD,
    "selected_recipe": selected_recipe,
    "frozen_alpha": frozen_alpha,
    "frozen_cap_ft": (
        "full" if not np.isfinite(frozen_cap) else frozen_cap
    ),
    "calibration": calibration_report,
    "valid": valid_report,
    "valid_posterior": {
        "mean_top_weight": float(valid_diagnostics["top_weight"].mean()),
        "mean_zero_weight": float(valid_diagnostics["zero_weight"].mean()),
        "mean_entropy": float(valid_diagnostics["entropy"].mean()),
        "mean_effective_candidates": float(
            valid_diagnostics["effective_candidates"].mean()
        ),
    },
    "gates": {
        "gain_vs_strong_min": PASS_GAIN,
        "positive_well_fraction_min": PASS_POSITIVE_WELLS,
        "median_well_gain_min_exclusive": PASS_MEDIAN_WELL_GAIN,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> rebuild the posterior scorer across all five folds"
        if strict_pass
        else "FAIL -> close E1 posterior path reranking"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2), encoding="utf-8"
)

print("\nCALIBRATION RECIPE TOP 20")
print(recipe_report.head(20).to_string(index=False))
print("\nCALIBRATION DOSE TOP 20")
print(dose_report.head(20).to_string(index=False))
print("\nFROZEN CALIBRATION")
print(json.dumps(calibration_report, indent=2))
print("\nFROZEN VALID")
print(json.dumps(valid_report, indent=2))
print("\nVALID POSTERIOR DIAGNOSTICS")
print(valid_diagnostics.describe().to_string())
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
