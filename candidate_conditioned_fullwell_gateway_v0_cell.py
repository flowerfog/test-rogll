"""Fold-clean candidate-conditioned complete-well GR gateway.

This is the first deployable follow-up to the truth-shape observability result.
It tests the missing representation directly: every candidate path is rendered
as a complete horizontal-GR versus candidate-warped-typewell-GR alignment.

Protocol
--------
* Folds 3/4/5 fit a residual-shape proposal bank and train one energy model.
* Fold 2 is the only validation fold. Fold 1 is never read.
* Candidate generation uses only single-well PF/beam/scale paths plus shape
  templates fitted on outer-train truth. Validation truth is used only for
  oracle labels and final metrics.
* No neighbour, spatial strong, CP12, Gold, or leaderboard signal is consumed.
* The script stops before training unless both the full proposal bank and a
  label-free GR-pruned 192-path bank retain enough whole-path oracle ceiling.

Run as one AutoDL notebook cell (or upload and ``%run`` it). It is deliberately
self-contained so a restarted kernel has no live-global prerequisites.
"""

from __future__ import annotations

from pathlib import Path
import gc
import json
import math
import random
import time

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path("/root/autodl-tmp")
DATA_ROOT_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    Path.cwd(),
)
DATA_ROOT = next(
    (path for path in DATA_ROOT_CANDIDATES if (path / "train").is_dir()),
    DATA_ROOT_CANDIDATES[0],
)
OOF_ROOT = ROOT / "oof_exports"
if not OOF_ROOT.exists():
    OOF_ROOT = DATA_ROOT / "oof_exports"

EXPERIMENT = "candidate_conditioned_fullwell_gateway_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
PANEL_DIR = OUT_DIR / "panels"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, PANEL_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
if FOLD_PATH is None:
    raise FileNotFoundError("Cannot locate fold-clean well_metadata.csv")

BANK_CANDIDATES = (
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1",
    OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75",
)
BANK_ROOT = next(
    (
        path
        for path in BANK_CANDIDATES
        if (path / "features").is_dir()
        and len(list((path / "features").glob("*__official_full.pkl"))) >= 700
    ),
    None,
)
if BANK_ROOT is None:
    raise FileNotFoundError(
        "Cannot locate a complete multicut official_full feature bank"
    )

TRAIN_FOLDS = (3, 4, 5)
VALID_FOLD = 2
SEED = 20260725

TEMPLATE_POINTS = 129
KMEANS_TEMPLATES = 48
PCA_COMPONENTS = 10
PCA_LEVELS = (-2.0, -1.0, 1.0, 2.0)
TEMPLATE_CAP_FT = 128.0

SEQUENCE_LENGTH = 768
PRUNED_CANDIDATES = 192
MIN_REFERENCE_COVERAGE = 0.60
MIN_QUERY_ROWS = 128

MAX_EPOCHS = 14
MIN_EPOCHS = 4
PATIENCE = 4
LEARNING_RATE = 2.5e-4
MIN_LEARNING_RATE = 3.0e-5
WEIGHT_DECAY = 4.0e-5
GRAD_ACCUM_WELLS = 4
GRADIENT_CLIP = 4.0

EMBEDDING = 40
SET_DIMENSION = 80
SOFT_TARGET_TEMPERATURE_FT = 1.5
HARD_MARGIN_FT = 4.0
LOGIT_MARGIN = 0.75

# A 5.5-ft final model needs substantial selector error budget. These gates are
# intentionally tighter than simply asking whether the menu dips below 5.5.
GATE_FULL_ORACLE_RMSE = 4.25
GATE_PRUNED_ORACLE_RMSE = 4.50
GATE_PRUNED_P95 = 8.50
GATE_PRUNED_FAR = 5.75

GATE_MODEL_RMSE = 7.00
GATE_MODEL_MEDIAN = 5.50
GATE_MODEL_FAR = 8.50
GATE_CONTROL_DEGRADATION = 1.00

ABSOLUTE_CANDIDATES = (
    "pf_ancc",
    "pf_z",
    "likpf_scale_3",
    "likpf_scale_5",
    "likpf_scale_8",
    "likpf_scale_12",
    "likpf_mean",
)
DELTA_CANDIDATES = (
    "beam_cons_d",
    "beam_loose_d",
    "beam_vcons_d",
    "beam_sm5_d",
    "beam_vloose_d",
    "beam_mid_d",
    "beam_stiff_d",
    "beam_mean_d",
    "beam_med_d",
    "sc8_d",
    "sc15_d",
    "sc25_d",
    "sc_cons_d",
    "sc_ens_d",
    "hyb_d",
    "slp_b_d_all",
    "slp_b_d_50",
)
BASE_MENU = ABSOLUTE_CANDIDATES + DELTA_CANDIDATES
STRATA_NAMES = (
    "near_0_500",
    "mid_500_1000",
    "mid_1000_2000",
    "far_gt2000",
)


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _safe_array(values, fallback=0.0):
    values = np.asarray(values, np.float64).copy()
    values[~np.isfinite(values)] = np.nan
    if np.isfinite(values).any():
        return (
            pd.Series(values)
            .interpolate(limit_direction="both")
            .fillna(float(fallback))
            .to_numpy(np.float64)
        )
    return np.full(len(values), float(fallback), np.float64)


def _numeric(frame, column):
    if column not in frame:
        return np.full(len(frame), np.nan, np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def _interp_finite(x, y, query):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    query = np.asarray(query, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return np.full(query.shape, np.nan, np.float64)
    xv = x[valid]
    yv = y[valid]
    order = np.argsort(xv, kind="mergesort")
    xv = xv[order]
    yv = yv[order]
    unique, inverse = np.unique(xv, return_inverse=True)
    if len(unique) != len(xv):
        sums = np.bincount(inverse, weights=yv)
        counts = np.bincount(inverse)
        xv = unique
        yv = sums / np.maximum(counts, 1)
    return np.interp(query, xv, yv, left=np.nan, right=np.nan)


def _robust_location_scale(values):
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    if int(valid.sum()) < 16:
        return 0.0, 1.0
    centre = float(np.median(values[valid]))
    scale = float(1.4826 * np.median(np.abs(values[valid] - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values[valid]), 1.0))
    return centre, scale


def _fit_affine(source, target):
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    valid = np.isfinite(source) & np.isfinite(target)
    x = source[valid]
    y = target[valid]
    if len(x) < 32 or np.std(x) < 1e-6:
        shift = float(np.nanmedian(y) - np.nanmedian(x)) if len(x) else 0.0
        return 1.0, shift, int(len(x)), np.nan
    keep = np.ones(len(x), bool)
    slope, intercept = 1.0, float(np.median(y) - np.median(x))
    for _ in range(4):
        design = np.c_[x[keep], np.ones(int(keep.sum()))]
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        centre, scale = _robust_location_scale(residual[keep])
        next_keep = np.abs(residual - centre) <= max(3.5 * scale, 8.0)
        if int(next_keep.sum()) < 32 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    fitted = slope * x[keep] + intercept
    rmse = float(np.sqrt(np.mean((fitted - y[keep]) ** 2)))
    return float(slope), float(intercept), int(keep.sum()), rmse


def _load_folds():
    frame = pd.read_csv(FOLD_PATH, dtype={"well": str})
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
    return frame.drop_duplicates("well", keep="last").reset_index(drop=True)


def _feature_path(well):
    return BANK_ROOT / "features" / f"{well}__official_full.pkl"


def _read_raw_well(well):
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    if not horizontal_path.exists():
        raise FileNotFoundError(horizontal_path)
    if not typewell_path.exists():
        raise FileNotFoundError(typewell_path)
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    return horizontal, typewell


def _prepare_typewell(typewell):
    frame = pd.DataFrame(
        {
            "TVT": _numeric(typewell, "TVT"),
            "GR": _numeric(typewell, "GR"),
        }
    ).dropna(subset=["TVT"])
    if len(frame) < 32:
        raise RuntimeError(f"typewell has only {len(frame)} finite TVT rows")
    frame = frame.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    frame["GR"] = frame["GR"].interpolate(limit_direction="both")
    frame = frame.dropna(subset=["GR"])
    if len(frame) < 32 or not np.all(np.diff(frame["TVT"].to_numpy()) > 0):
        raise RuntimeError("typewell TVT axis is not a usable strict sequence")
    return (
        frame["TVT"].to_numpy(np.float64),
        frame["GR"].to_numpy(np.float64),
    )


def _candidate_curves(frame):
    frame = frame.copy()
    if "_row_idx" in frame:
        frame = frame.sort_values("_row_idx")
    frame = frame.reset_index(drop=True)
    missing = {"last_known_tvt", "target"}.difference(frame.columns)
    missing.update(column for column in BASE_MENU if column not in frame)
    if missing:
        raise RuntimeError(f"feature frame missing {sorted(missing)}")
    last = _safe_array(frame["last_known_tvt"], 0.0)
    fallback = float(np.median(last))
    curves = []
    for name in BASE_MENU:
        if name in ABSOLUTE_CANDIDATES:
            curves.append(_safe_array(frame[name], fallback))
        else:
            curves.append(last + _safe_array(frame[name], 0.0))
    truth = last + _safe_array(frame["target"], 0.0)
    return frame, last, np.stack(curves).astype(np.float32), truth.astype(np.float32)


def _well_payload(well):
    frame = pd.read_pickle(_feature_path(well))
    frame, last, bases, truth = _candidate_curves(frame)
    horizontal, typewell = _read_raw_well(well)
    md = _numeric(horizontal, "MD")
    gr = _numeric(horizontal, "GR")
    tvt_input = _numeric(horizontal, "TVT_input")
    raw_truth = _numeric(horizontal, "TVT")
    hidden = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(raw_truth))
    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(hidden) != len(frame):
        raise RuntimeError(
            f"feature/eval length mismatch {len(frame)}/{len(hidden)}"
        )
    if len(known) < 32 or len(hidden) < MIN_QUERY_ROWS:
        raise RuntimeError(
            f"insufficient official rows known={len(known)} hidden={len(hidden)}"
        )
    raw_difference = float(np.sqrt(np.mean((truth - raw_truth[hidden]) ** 2)))
    if raw_difference > 0.05:
        raise RuntimeError(f"truth alignment RMSE={raw_difference:.6f}")
    anchor = int(hidden[0] - 1)
    if anchor < 0:
        raise RuntimeError("no visible heel anchor")
    eval_md = md[hidden]
    md_since = eval_md - float(md[anchor])
    progress = np.clip(md_since / max(float(md_since[-1]), 1.0), 0.0, 1.0)
    return {
        "well": str(well),
        "horizontal": horizontal,
        "typewell": typewell,
        "md": md,
        "gr": gr,
        "tvt_input": tvt_input,
        "known": known,
        "hidden": hidden,
        "eval_md": eval_md,
        "md_since": md_since,
        "progress": progress,
        "last": last,
        "anchor_tvt": float(tvt_input[anchor]),
        "bases": bases,
        "truth": truth,
    }


# ---------------------------------------------------------------------------
# Outer-train residual shape bank
# ---------------------------------------------------------------------------


TEMPLATE_GRID = np.linspace(0.0, 1.0, TEMPLATE_POINTS, dtype=np.float32)


def _smooth_template(progress, residual):
    progress = np.asarray(progress, np.float64)
    residual = np.asarray(residual, np.float64)
    valid = np.isfinite(progress) & np.isfinite(residual)
    progress = progress[valid]
    residual = residual[valid]
    order = np.argsort(progress, kind="mergesort")
    progress = progress[order]
    residual = residual[order]
    unique, index = np.unique(progress, return_index=True)
    values = residual[index]
    shape = np.interp(TEMPLATE_GRID, unique, values)
    shape = savgol_filter(shape, 11, 2, mode="interp")
    shape -= shape[0]
    return np.clip(shape, -TEMPLATE_CAP_FT, TEMPLATE_CAP_FT).astype(np.float32)


def _analytic_templates():
    p = TEMPLATE_GRID.astype(np.float64)
    warm = 1.0 - np.exp(-8.0 * p)
    smooth_late = np.clip((p - 0.30) / 0.70, 0.0, 1.0)
    smooth_late = smooth_late * smooth_late * (3.0 - 2.0 * smooth_late)
    templates = [("zero", np.zeros_like(p))]
    for amplitude in (-64, -48, -32, -24, -16, -8, 8, 16, 24, 32, 48, 64):
        templates.append((f"warm_{amplitude:+d}", amplitude * warm))
    for amplitude in (-48, -32, -24, -16, -8, 8, 16, 24, 32, 48):
        templates.append((f"linear_{amplitude:+d}", amplitude * p))
    for amplitude in (-32, -24, -16, -8, 8, 16, 24, 32):
        templates.append((f"quadratic_{amplitude:+d}", amplitude * p * p))
    for amplitude in (-24, -16, -8, 8, 16, 24):
        templates.append((f"bow_{amplitude:+d}", amplitude * np.sin(np.pi * p)))
    for amplitude in (-32, -24, -16, -8, 8, 16, 24, 32):
        templates.append((f"late_{amplitude:+d}", amplitude * smooth_late))
    return templates


def _deduplicate_templates(names, templates):
    kept_names = []
    kept = []
    signatures = set()
    for name, template in zip(names, templates):
        template = np.asarray(template, np.float32).copy()
        template -= template[0]
        template = np.clip(template, -TEMPLATE_CAP_FT, TEMPLATE_CAP_FT)
        signature = np.round(template / 0.25).astype(np.int16).tobytes()
        if signature in signatures:
            continue
        signatures.add(signature)
        kept_names.append(str(name))
        kept.append(template)
    return np.asarray(kept_names), np.stack(kept).astype(np.float32)


def _fit_template_bank(train_wells):
    started = time.time()
    shapes = []
    source_rows = []
    for index, well in enumerate(train_wells, start=1):
        payload = _well_payload(well)
        errors = np.mean(
            (payload["bases"].astype(np.float64) - payload["truth"][None, :]) ** 2,
            axis=1,
        )
        best_base = int(np.argmin(errors))
        residual = payload["truth"] - payload["bases"][best_base]
        shape = _smooth_template(payload["progress"], residual)
        shapes.append(shape)
        source_rows.append(
            {
                "well": well,
                "best_base": BASE_MENU[best_base],
                "best_base_rmse": float(np.sqrt(errors[best_base])),
                "residual_end": float(shape[-1]),
                "residual_p95_abs": float(np.quantile(np.abs(shape), 0.95)),
            }
        )
        if index % 50 == 0 or index == len(train_wells):
            print(f"fit shape sources {index}/{len(train_wells)}", flush=True)
    shapes = np.stack(shapes).astype(np.float32)
    pd.DataFrame(source_rows).to_csv(
        OUT_DIR / "template_source_wells.csv", index=False
    )

    names = []
    templates = []
    for name, template in _analytic_templates():
        names.append(name)
        templates.append(template)

    kmeans = KMeans(
        n_clusters=min(KMEANS_TEMPLATES, len(shapes)),
        random_state=SEED,
        n_init=20,
    ).fit(shapes)
    for index, centre in enumerate(kmeans.cluster_centers_):
        names.append(f"kmeans_{index:02d}")
        templates.append(centre)

    components = min(PCA_COMPONENTS, len(shapes) - 1, shapes.shape[1])
    pca = PCA(n_components=components, random_state=SEED).fit(shapes)
    names.append("pca_mean")
    templates.append(pca.mean_)
    for component in range(components):
        scale = float(np.sqrt(max(pca.explained_variance_[component], 1e-8)))
        for level in PCA_LEVELS:
            names.append(f"pca_{component:02d}_{level:+.0f}")
            templates.append(pca.mean_ + level * scale * pca.components_[component])

    names, templates = _deduplicate_templates(names, templates)
    zero_candidates = np.flatnonzero(
        np.sqrt(np.mean(templates.astype(np.float64) ** 2, axis=1)) < 1e-8
    )
    if len(zero_candidates) != 1:
        raise RuntimeError(f"expected one zero template, found {zero_candidates}")
    zero = int(zero_candidates[0])
    if zero != 0:
        order = np.r_[zero, np.delete(np.arange(len(names)), zero)]
        names = names[order]
        templates = templates[order]

    np.savez_compressed(
        OUT_DIR / "outer_train_template_bank.npz",
        names=names,
        templates=templates.astype(np.float32),
        grid=TEMPLATE_GRID,
        source_folds=np.asarray(TRAIN_FOLDS, np.int16),
        cache_version=np.asarray([1], np.int16),
    )
    report = {
        "source_wells": len(train_wells),
        "raw_shapes": int(len(shapes)),
        "templates": int(len(templates)),
        "base_paths": int(len(BASE_MENU)),
        "full_candidates_per_well": int(len(templates) * len(BASE_MENU)),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    print("TEMPLATE BANK", json.dumps(report, indent=2), flush=True)
    return names, templates, report


# ---------------------------------------------------------------------------
# Candidate rendering, legal GR pruning, and panel cache
# ---------------------------------------------------------------------------


def _panel_path(well):
    return PANEL_DIR / f"{well}.npz"


def _masked_ncc(reference, query, valid):
    weight = valid.astype(np.float32)
    count = weight.sum(axis=1).clip(min=1.0)
    q = np.broadcast_to(query[None, :], reference.shape)
    mean_r = (reference * weight).sum(axis=1) / count
    mean_q = (q * weight).sum(axis=1) / count
    centred_r = (reference - mean_r[:, None]) * weight
    centred_q = (q - mean_q[:, None]) * weight
    numerator = (centred_r * centred_q).sum(axis=1)
    denominator = np.sqrt(
        (centred_r * centred_r).sum(axis=1)
        * (centred_q * centred_q).sum(axis=1)
        + 1e-8
    )
    return numerator / denominator


def _masked_smooth(values, valid, size):
    numerator = uniform_filter1d(
        values * valid, size=size, axis=1, mode="nearest"
    )
    denominator = uniform_filter1d(
        valid.astype(np.float32), size=size, axis=1, mode="nearest"
    )
    output = numerator / np.maximum(denominator, 1e-3)
    return output, denominator >= 0.80


def _percentile_rank(values):
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values)
    replacement = float(np.max(values[finite]) + 1.0) if finite.any() else 1.0
    safe = np.where(finite, values, replacement)
    order = np.argsort(safe, kind="mergesort")
    ranks = np.empty(len(safe), np.float64)
    ranks[order] = np.arange(len(safe), dtype=np.float64)
    return ranks / max(len(safe) - 1, 1)


def _heuristic_scores(query, reference, valid):
    weight = valid.astype(np.float32)
    count = weight.sum(axis=1).clip(min=1.0)
    difference = (reference - query[None, :]) * weight
    raw_mae = np.abs(difference).sum(axis=1) / count
    raw_rmse = np.sqrt((difference * difference).sum(axis=1) / count)
    raw_ncc_cost = 1.0 - _masked_ncc(reference, query, valid)

    smooth_costs = []
    query_matrix = np.broadcast_to(query[None, :], reference.shape).astype(np.float32)
    for size in (9, 33, 97):
        smooth_r, valid_r = _masked_smooth(reference, valid, size)
        smooth_q, valid_q = _masked_smooth(query_matrix, valid, size)
        smooth_valid = valid_r & valid_q
        smooth_weight = smooth_valid.astype(np.float32)
        smooth_count = smooth_weight.sum(axis=1).clip(min=1.0)
        smooth_costs.append(
            (np.abs(smooth_r - smooth_q) * smooth_weight).sum(axis=1)
            / smooth_count
        )

    derivative_r = np.diff(reference, axis=1, prepend=reference[:, :1])
    derivative_q = np.diff(query, prepend=query[:1])
    derivative_valid = valid & np.roll(valid, 1, axis=1)
    derivative_valid[:, 0] = False
    derivative_weight = derivative_valid.astype(np.float32)
    derivative_count = derivative_weight.sum(axis=1).clip(min=1.0)
    derivative_mae = (
        np.abs(derivative_r - derivative_q[None, :]) * derivative_weight
    ).sum(axis=1) / derivative_count

    segment_costs = []
    edges = np.linspace(0, reference.shape[1], 7).round().astype(int)
    for left, right in zip(edges[:-1], edges[1:]):
        segment_valid = valid[:, left:right]
        segment_count = segment_valid.sum(axis=1).clip(min=1)
        segment_costs.append(
            (
                np.abs(reference[:, left:right] - query[None, left:right])
                * segment_valid
            ).sum(axis=1)
            / segment_count
        )
    segment_p90 = np.quantile(np.stack(segment_costs, axis=1), 0.90, axis=1)
    coverage_cost = 1.0 - valid.mean(axis=1)

    costs = np.column_stack(
        [
            raw_mae,
            raw_rmse,
            raw_ncc_cost,
            *smooth_costs,
            derivative_mae,
            segment_p90,
            coverage_cost,
        ]
    )
    ranks = np.column_stack(
        [_percentile_rank(costs[:, column]) for column in range(costs.shape[1])]
    )
    score = ranks.mean(axis=1)
    score += 4.0 * (valid.mean(axis=1) < MIN_REFERENCE_COVERAGE)
    return score.astype(np.float32), costs.astype(np.float32)


def _strata_masks(md_since):
    return (
        md_since <= 500.0,
        (md_since > 500.0) & (md_since <= 1000.0),
        (md_since > 1000.0) & (md_since <= 2000.0),
        md_since > 2000.0,
    )


def _candidate_errors(payload, templates):
    template_native = np.stack(
        [np.interp(payload["progress"], TEMPLATE_GRID, row) for row in templates]
    ).astype(np.float32)
    base_count = len(BASE_MENU)
    template_count = len(templates)
    candidate_count = base_count * template_count
    mse = np.empty(candidate_count, np.float32)
    mae = np.empty(candidate_count, np.float32)
    strata_mse = np.full((candidate_count, len(STRATA_NAMES)), np.nan, np.float32)
    masks = _strata_masks(payload["md_since"])
    strata_counts = np.asarray([int(mask.sum()) for mask in masks], np.int32)
    for base_index in range(base_count):
        left = base_index * template_count
        right = left + template_count
        difference = (
            payload["bases"][base_index][None, :]
            + template_native
            - payload["truth"][None, :]
        ).astype(np.float64)
        mse[left:right] = np.mean(difference * difference, axis=1)
        mae[left:right] = np.mean(np.abs(difference), axis=1)
        for stratum, mask in enumerate(masks):
            if mask.any():
                strata_mse[left:right, stratum] = np.mean(
                    difference[:, mask] ** 2, axis=1
                )
    base_ids = np.repeat(np.arange(base_count, dtype=np.int16), template_count)
    template_ids = np.tile(
        np.arange(template_count, dtype=np.int16), base_count
    )
    return mse, mae, strata_mse, strata_counts, base_ids, template_ids


def _build_panel(well, templates):
    payload = _well_payload(well)
    typewell_tvt, typewell_gr = _prepare_typewell(payload["typewell"])
    known = payload["known"]
    tw_prefix = _interp_finite(
        typewell_tvt, typewell_gr, payload["tvt_input"][known]
    )
    slope, intercept, pairs, calibration_rmse = _fit_affine(
        tw_prefix, payload["gr"][known]
    )
    calibrated_typewell = slope * typewell_gr + intercept
    centre, scale = _robust_location_scale(payload["gr"][known])

    sequence_md = np.linspace(
        payload["eval_md"][0], payload["eval_md"][-1], SEQUENCE_LENGTH
    )
    # Keep the exact heel-anchor convention used by native-row oracle scoring.
    # The first hidden row is usually already a small positive distance from
    # the last visible station, so resetting it to progress zero would make
    # rendering and labels disagree near the heel.
    sequence_progress = np.interp(
        sequence_md, payload["eval_md"], payload["progress"]
    )
    query = _interp_finite(payload["md"], payload["gr"], sequence_md)
    query_valid = np.isfinite(query)
    query = np.clip((query - centre) / scale, -8.0, 8.0)
    query = np.where(query_valid, query, 0.0).astype(np.float32)

    base_sequence = np.stack(
        [
            np.interp(sequence_md, payload["eval_md"], base)
            for base in payload["bases"]
        ]
    ).astype(np.float32)
    template_sequence = np.stack(
        [np.interp(sequence_progress, TEMPLATE_GRID, row) for row in templates]
    ).astype(np.float32)
    all_paths = (
        base_sequence[:, None, :] + template_sequence[None, :, :]
    ).reshape(-1, SEQUENCE_LENGTH)
    reference_valid = (
        (all_paths >= float(np.nanmin(typewell_tvt)))
        & (all_paths <= float(np.nanmax(typewell_tvt)))
        & query_valid[None, :]
    )
    reference = np.interp(
        all_paths.reshape(-1), typewell_tvt, calibrated_typewell
    ).reshape(all_paths.shape)
    reference = np.clip((reference - centre) / scale, -8.0, 8.0)
    reference = np.where(reference_valid, reference, 0.0).astype(np.float32)

    heuristic, heuristic_components = _heuristic_scores(
        query, reference, reference_valid
    )
    mse, mae, strata_mse, strata_counts, base_ids, template_ids = _candidate_errors(
        payload, templates
    )
    coverage = reference_valid.mean(axis=1)
    deployable = coverage >= MIN_REFERENCE_COVERAGE
    deployable_mse = np.where(deployable, mse, np.inf)
    if not np.isfinite(deployable_mse).any():
        raise RuntimeError("no deployable candidates")

    # Preserve every unmodified base path, then each base's best GR match, then
    # fill globally by the same label-free full-well GR score.
    forced = []
    for base_index in range(len(BASE_MENU)):
        identity = base_index * len(templates)
        if deployable[identity]:
            forced.append(identity)
        members = np.flatnonzero((base_ids == base_index) & deployable)
        if len(members):
            forced.append(int(members[np.argmin(heuristic[members])]))
    selected = []
    seen = set()
    for candidate in forced + np.argsort(heuristic, kind="mergesort").tolist():
        candidate = int(candidate)
        if candidate in seen or not deployable[candidate]:
            continue
        seen.add(candidate)
        selected.append(candidate)
        if len(selected) == PRUNED_CANDIDATES:
            break
    if len(selected) != PRUNED_CANDIDATES:
        raise RuntimeError(
            f"only {len(selected)} deployable candidates after pruning"
        )
    selected = np.asarray(selected, np.int32)

    selected_heuristic = heuristic[selected]
    selected_heuristic = _percentile_rank(selected_heuristic).astype(np.float32)
    selected_paths = all_paths[selected]
    candidate_ids = np.column_stack([base_ids[selected], template_ids[selected]])
    panel = {
        "well": str(well),
        "query": query,
        "reference": reference[selected],
        "valid": reference_valid[selected],
        "path_delta": (selected_paths - payload["anchor_tvt"]).astype(np.float32),
        "heuristic": selected_heuristic,
        "candidate_mse": mse[selected],
        "candidate_mae": mae[selected],
        "strata_mse": strata_mse[selected],
        "strata_counts": strata_counts,
        "candidate_ids": candidate_ids,
        "eval_rows": int(len(payload["truth"])),
        "full_oracle_mse": float(np.min(deployable_mse)),
        "pruned_oracle_mse": float(np.min(mse[selected])),
        "heuristic_mse": float(mse[selected[int(np.argmin(selected_heuristic))]]),
        "full_candidates": int(len(mse)),
        "deployable_candidates": int(deployable.sum()),
        "minimum_selected_coverage": float(coverage[selected].min()),
        "calibration_pairs": int(pairs),
        "calibration_rmse": float(calibration_rmse),
        "heuristic_component_count": int(heuristic_components.shape[1]),
    }
    return panel


def _save_panel(panel):
    np.savez_compressed(
        _panel_path(panel["well"]),
        well=np.asarray(panel["well"]),
        query=panel["query"].astype(np.float16),
        reference=panel["reference"].astype(np.float16),
        valid=panel["valid"].astype(np.uint8),
        path_delta=panel["path_delta"].astype(np.float16),
        heuristic=panel["heuristic"].astype(np.float32),
        candidate_mse=panel["candidate_mse"].astype(np.float32),
        candidate_mae=panel["candidate_mae"].astype(np.float32),
        strata_mse=panel["strata_mse"].astype(np.float32),
        strata_counts=panel["strata_counts"].astype(np.int32),
        candidate_ids=panel["candidate_ids"].astype(np.int16),
        eval_rows=np.asarray([panel["eval_rows"]], np.int32),
        full_oracle_mse=np.asarray([panel["full_oracle_mse"]], np.float32),
        pruned_oracle_mse=np.asarray([panel["pruned_oracle_mse"]], np.float32),
        heuristic_mse=np.asarray([panel["heuristic_mse"]], np.float32),
        full_candidates=np.asarray([panel["full_candidates"]], np.int32),
        deployable_candidates=np.asarray(
            [panel["deployable_candidates"]], np.int32
        ),
        minimum_selected_coverage=np.asarray(
            [panel["minimum_selected_coverage"]], np.float32
        ),
        calibration_pairs=np.asarray([panel["calibration_pairs"]], np.int32),
        calibration_rmse=np.asarray([panel["calibration_rmse"]], np.float32),
        cache_version=np.asarray([1], np.int16),
    )


def _load_panel(well):
    with np.load(_panel_path(well), allow_pickle=False) as data:
        return {
            "well": str(data["well"].item()),
            "query": data["query"].astype(np.float32),
            "reference": data["reference"].astype(np.float32),
            "valid": data["valid"].astype(bool),
            "path_delta": data["path_delta"].astype(np.float32),
            "heuristic": data["heuristic"].astype(np.float32),
            "candidate_mse": data["candidate_mse"].astype(np.float32),
            "candidate_mae": data["candidate_mae"].astype(np.float32),
            "strata_mse": data["strata_mse"].astype(np.float32),
            "strata_counts": data["strata_counts"].astype(np.int32),
            "candidate_ids": data["candidate_ids"].astype(np.int16),
            "eval_rows": int(data["eval_rows"][0]),
            "full_oracle_mse": float(data["full_oracle_mse"][0]),
            "pruned_oracle_mse": float(data["pruned_oracle_mse"][0]),
            "heuristic_mse": float(data["heuristic_mse"][0]),
        }


def _build_panels(wells, templates, label):
    rows = []
    started = time.time()
    for index, well in enumerate(wells, start=1):
        try:
            if not _panel_path(well).exists():
                _save_panel(_build_panel(well, templates))
            panel = _load_panel(well)
            rows.append(
                {
                    "well": well,
                    "status": "ok",
                    "eval_rows": panel["eval_rows"],
                    "full_oracle_rmse": math.sqrt(panel["full_oracle_mse"]),
                    "pruned_oracle_rmse": math.sqrt(panel["pruned_oracle_mse"]),
                    "heuristic_rmse": math.sqrt(panel["heuristic_mse"]),
                }
            )
        except Exception as error:
            rows.append({"well": well, "status": "failed", "reason": repr(error)})
        if index % 25 == 0 or index == len(wells):
            print(
                f"build {label} panels {index}/{len(wells)} "
                f"elapsed={(time.time() - started) / 60.0:.1f}m",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / f"panel_build_{label}.csv", index=False)
    failed = frame[frame["status"].eq("failed")]
    if len(failed):
        raise RuntimeError(
            f"{len(failed)} {label} panels failed: "
            + repr(failed.head(10).to_dict(orient="records"))
        )
    return frame


def _aggregate_indices(wells, selector):
    rows = []
    total_sse = 0.0
    total_abs = 0.0
    total_rows = 0
    strata_sse = np.zeros(len(STRATA_NAMES), np.float64)
    strata_rows = np.zeros(len(STRATA_NAMES), np.int64)
    for well in wells:
        panel = _load_panel(well)
        if selector == "full_oracle":
            mse = panel["full_oracle_mse"]
            index = None
        elif selector == "pruned_oracle":
            index = int(np.argmin(panel["candidate_mse"]))
            mse = float(panel["candidate_mse"][index])
        elif selector == "heuristic":
            index = int(np.argmin(panel["heuristic"]))
            mse = float(panel["candidate_mse"][index])
        else:
            index = int(selector[well])
            mse = float(panel["candidate_mse"][index])
        count = panel["eval_rows"]
        total_sse += count * mse
        total_rows += count
        if index is not None:
            total_abs += count * float(panel["candidate_mae"][index])
            for stratum in range(len(STRATA_NAMES)):
                n_rows = int(panel["strata_counts"][stratum])
                value = float(panel["strata_mse"][index, stratum])
                if n_rows and np.isfinite(value):
                    strata_sse[stratum] += n_rows * value
                    strata_rows[stratum] += n_rows
        rows.append(
            {
                "well": well,
                "rows": count,
                "rmse": math.sqrt(mse),
                "selected_index": index,
            }
        )
    bywell = pd.DataFrame(rows)
    report = {
        "selector": selector if isinstance(selector, str) else "model",
        "rows": int(total_rows),
        "wells": int(len(bywell)),
        "exact_sse": float(total_sse),
        "rmse": float(math.sqrt(total_sse / total_rows)),
        "mae": float(total_abs / total_rows) if total_abs else np.nan,
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
    }
    for index, name in enumerate(STRATA_NAMES):
        report[f"{name}_rmse"] = (
            float(math.sqrt(strata_sse[index] / strata_rows[index]))
            if strata_rows[index]
            else np.nan
        )
    return report, bywell


# ---------------------------------------------------------------------------
# Complete-well candidate energy model
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 5
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(
            channels, channels, 5, padding=2 * dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )

    def forward(self, values):
        residual = self.conv1(F.gelu(self.norm1(values)))
        residual = self.conv2(F.gelu(self.norm2(residual)))
        return values + residual


class CandidateConditionedEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(11, EMBEDDING, 7, padding=3)
        self.blocks = nn.ModuleList(
            [ResidualBlock(EMBEDDING, dilation) for dilation in (1, 2, 4, 8, 16, 32)]
        )
        descriptor_size = EMBEDDING * 3 + 10
        self.descriptor = nn.Sequential(
            nn.Linear(descriptor_size, SET_DIMENSION),
            nn.GELU(),
            nn.LayerNorm(SET_DIMENSION),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=SET_DIMENSION,
            nhead=4,
            dim_feedforward=SET_DIMENSION * 3,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.energy = nn.Sequential(
            nn.LayerNorm(SET_DIMENSION * 2),
            nn.Linear(SET_DIMENSION * 2, SET_DIMENSION),
            nn.GELU(),
            nn.Linear(SET_DIMENSION, 1),
        )

    def forward(self, query, reference, valid, path_delta, heuristic):
        candidates, length = reference.shape
        available = valid.mean(dim=1) >= MIN_REFERENCE_COVERAGE
        q = query[None, :].expand(candidates, -1)
        difference = (q - reference) * valid
        absolute = difference.abs()
        q_derivative = torch.diff(q, dim=1, prepend=q[:, :1])
        r_derivative = torch.diff(reference, dim=1, prepend=reference[:, :1])
        path = torch.clamp(path_delta / 96.0, -8.0, 8.0)
        path_derivative = torch.diff(path, dim=1, prepend=path[:, :1]) * 16.0
        progress = torch.linspace(
            0.0, 1.0, length, device=query.device, dtype=query.dtype
        )[None, :].expand(candidates, -1)
        heuristic_sequence = heuristic[:, None].expand(-1, length)
        inputs = torch.stack(
            [
                q,
                reference,
                difference,
                absolute,
                q_derivative,
                r_derivative,
                path,
                path_derivative,
                valid,
                progress,
                heuristic_sequence,
            ],
            dim=1,
        )
        hidden = self.stem(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        mask = valid[:, None, :]
        count = mask.sum(dim=2).clamp_min(1.0)
        mean = (hidden * mask).sum(dim=2) / count
        variance = ((hidden - mean[:, :, None]) ** 2 * mask).sum(dim=2) / count
        maximum = hidden.masked_fill(mask <= 0, -1e4).amax(dim=2)
        maximum = torch.where(available[:, None], maximum, torch.zeros_like(maximum))
        valid_count = valid.sum(dim=1).clamp_min(1.0)
        early = progress <= 0.33
        late = progress >= 0.67
        summaries = torch.stack(
            [
                (absolute * valid).sum(dim=1) / valid_count,
                torch.sqrt((difference.square() * valid).sum(dim=1) / valid_count + 1e-8),
                (difference * valid).sum(dim=1) / valid_count,
                valid.mean(dim=1),
                (absolute * valid * early).sum(dim=1)
                / (valid * early).sum(dim=1).clamp_min(1.0),
                (absolute * valid * late).sum(dim=1)
                / (valid * late).sum(dim=1).clamp_min(1.0),
                path.abs().mean(dim=1),
                path_derivative.abs().mean(dim=1),
                heuristic,
                absolute.amax(dim=1),
            ],
            dim=1,
        )
        descriptor = self.descriptor(
            torch.cat([mean, torch.sqrt(variance + 1e-8), maximum, summaries], dim=1)
        )
        contextual = self.set_encoder(
            descriptor[None], src_key_padding_mask=(~available)[None]
        )[0]
        global_context = (
            (contextual * available[:, None]).sum(dim=0, keepdim=True)
            / available.sum().clamp_min(1)
        ).expand_as(contextual)
        logits = self.energy(torch.cat([contextual, global_context], dim=1))[:, 0]
        return logits.masked_fill(~available, -1e4)


def _tensor_panel(panel, device, control="full", permutation=None):
    query = panel["query"].copy()
    reference = panel["reference"].copy()
    valid = panel["valid"].copy()
    path_delta = panel["path_delta"].copy()
    heuristic = panel["heuristic"].copy()
    if control == "no_gr":
        query.fill(0.0)
    elif control == "shuffled_alignment":
        reference = np.roll(reference, 17, axis=0)
        valid = np.roll(valid, 17, axis=0)
    elif control != "full":
        raise ValueError(control)
    if permutation is not None:
        reference = reference[permutation]
        valid = valid[permutation]
        path_delta = path_delta[permutation]
        heuristic = heuristic[permutation]
    return (
        torch.from_numpy(query).to(device),
        torch.from_numpy(reference).to(device),
        torch.from_numpy(valid.astype(np.float32)).to(device),
        torch.from_numpy(path_delta).to(device),
        torch.from_numpy(heuristic).to(device),
    )


def _loss(model, panel, device):
    permutation = np.random.permutation(PRUNED_CANDIDATES)
    tensors = _tensor_panel(panel, device, permutation=permutation)
    mse = torch.from_numpy(panel["candidate_mse"][permutation]).to(device)
    rmse = torch.sqrt(mse.clamp_min(0.0) + 1e-8)
    target = torch.argmin(mse)
    logits = model(*tensors)
    hard_ce = F.cross_entropy(logits[None], target[None])
    soft_target = torch.softmax(
        -(rmse - rmse.min()) / SOFT_TARGET_TEMPERATURE_FT, dim=0
    )
    soft_ce = -(soft_target * F.log_softmax(logits, dim=0)).sum()
    hard_negative = (rmse >= rmse[target] + HARD_MARGIN_FT) & (logits > -1e3)
    if hard_negative.any():
        margin = F.relu(
            LOGIT_MARGIN - logits[target] + logits[hard_negative].amax()
        )
    else:
        margin = logits[target] * 0.0
    loss = 0.55 * hard_ce + 0.45 * soft_ce + 0.20 * margin
    return loss, {
        "hard_ce": float(hard_ce.detach().cpu()),
        "soft_ce": float(soft_ce.detach().cpu()),
        "margin": float(margin.detach().cpu()),
    }


@torch.no_grad()
def _predict(model, panel, device, control="full"):
    model.eval()
    return model(*_tensor_panel(panel, device, control=control)).cpu().numpy()


def _evaluate(model, wells, device, control="full"):
    selected = {}
    for well in wells:
        panel = _load_panel(well)
        selected[well] = int(np.argmax(_predict(model, panel, device, control)))
    return _aggregate_indices(wells, selected)


def _train_epoch(model, wells, optimizer, scaler, device, epoch):
    model.train()
    order = list(wells)
    random.Random(SEED + epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "hard_ce": 0.0, "soft_ce": 0.0, "margin": 0.0}
    steps = 0
    for index, well in enumerate(order, start=1):
        panel = _load_panel(well)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            loss, parts = _loss(model, panel, device)
            scaled_loss = loss / GRAD_ACCUM_WELLS
        scaler.scale(scaled_loss).backward()
        if index % GRAD_ACCUM_WELLS == 0 or index == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        totals["loss"] += float(loss.detach().cpu())
        for key, value in parts.items():
            totals[key] += value
        if index % 100 == 0 or index == len(order):
            print(
                f"ENERGY epoch={epoch} wells={index}/{len(order)} "
                f"loss={totals['loss'] / index:.4f}",
                flush=True,
            )
    output = {key: value / len(order) for key, value in totals.items()}
    output["optimizer_steps"] = steps
    return output


# ---------------------------------------------------------------------------
# Execute the nested gateway
# ---------------------------------------------------------------------------


_seed_everything(SEED)
folds = _load_folds()
train_wells = sorted(
    folds.loc[folds["fold"].isin(TRAIN_FOLDS), "well"].astype(str).tolist()
)
valid_wells = sorted(
    folds.loc[folds["fold"].eq(VALID_FOLD), "well"].astype(str).tolist()
)
if len(train_wells) < 450 or len(valid_wells) < 150:
    raise RuntimeError(
        f"Unexpected fold sizes train={len(train_wells)} valid={len(valid_wells)}"
    )
for well in train_wells + valid_wells:
    if not _feature_path(well).exists():
        raise FileNotFoundError(_feature_path(well))

started = time.time()
template_names, templates, template_report = _fit_template_bank(train_wells)

# Fold 2 is built first. No training cost is paid unless the deployable menu
# retains enough oracle ceiling after label-free full-well GR pruning.
valid_build = _build_panels(valid_wells, templates, "fold2")
full_oracle, full_oracle_wells = _aggregate_indices(valid_wells, "full_oracle")
pruned_oracle, pruned_oracle_wells = _aggregate_indices(valid_wells, "pruned_oracle")
heuristic, heuristic_wells = _aggregate_indices(valid_wells, "heuristic")
full_oracle_wells.to_csv(OUT_DIR / "fold2_full_oracle_bywell.csv", index=False)
pruned_oracle_wells.to_csv(OUT_DIR / "fold2_pruned_oracle_bywell.csv", index=False)
heuristic_wells.to_csv(OUT_DIR / "fold2_heuristic_bywell.csv", index=False)

oracle_gate = {
    "experiment": EXPERIMENT,
    "stage": "fold2_candidate_ceiling",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_spatial_strong_cp12_gold": False,
    "train_folds": list(TRAIN_FOLDS),
    "validation_fold": VALID_FOLD,
    "fold1_touched": False,
    "bank_root": str(BANK_ROOT),
    "base_menu": list(BASE_MENU),
    "template_bank": template_report,
    "full_oracle": full_oracle,
    "pruned_oracle": pruned_oracle,
    "label_free_heuristic": heuristic,
    "gates": {
        "full_oracle_rmse_le": GATE_FULL_ORACLE_RMSE,
        "pruned_oracle_rmse_le": GATE_PRUNED_ORACLE_RMSE,
        "pruned_p95_le": GATE_PRUNED_P95,
        "pruned_far_gt2000_rmse_le": GATE_PRUNED_FAR,
    },
}
oracle_gate["strict_pass"] = bool(
    full_oracle["rmse"] <= GATE_FULL_ORACLE_RMSE
    and pruned_oracle["rmse"] <= GATE_PRUNED_ORACLE_RMSE
    and pruned_oracle["p95_well_rmse"] <= GATE_PRUNED_P95
    and pruned_oracle["far_gt2000_rmse"] <= GATE_PRUNED_FAR
)
if oracle_gate["strict_pass"]:
    oracle_gate["next"] = (
        "PASS -> build outer-train panels and train complete-well energy model"
    )
elif full_oracle["rmse"] <= GATE_FULL_ORACLE_RMSE:
    oracle_gate["next"] = (
        "FULL BANK PASSES -> heuristic pruning is the bottleneck; train a "
        "full-bank unary retriever before any 192-path set reranker"
    )
else:
    oracle_gate["next"] = (
        "FAIL -> candidate coverage is still the bottleneck; expand proposals"
    )
_write_json(OUT_DIR / "oracle_gate.json", oracle_gate)
print("ORACLE GATE", json.dumps(oracle_gate, indent=2), flush=True)
if not oracle_gate["strict_pass"]:
    raise RuntimeError(json.dumps(oracle_gate, indent=2))

train_build = _build_panels(train_wells, templates, "train345")
guards = {
    "train_wells": len(train_wells),
    "valid_wells": len(valid_wells),
    "templates": len(templates),
    "base_paths": len(BASE_MENU),
    "full_candidates": int(len(templates) * len(BASE_MENU)),
    "pruned_candidates": PRUNED_CANDIDATES,
    "sequence_length": SEQUENCE_LENGTH,
    "minimum_reference_coverage": MIN_REFERENCE_COVERAGE,
    "fold1_touched": False,
}
print("PANEL GUARDS", guards, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CandidateConditionedEnergy().to(device)
parameters = int(sum(parameter.numel() for parameter in model.parameters()))
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_EPOCHS,
    eta_min=MIN_LEARNING_RATE,
)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
history = []
best = None
bad_epochs = 0
checkpoint_path = MODEL_DIR / "best_fold2_candidate_energy.pt"

for epoch in range(1, MAX_EPOCHS + 1):
    training = _train_epoch(model, train_wells, optimizer, scaler, device, epoch)
    validation, _ = _evaluate(model, valid_wells, device, control="full")
    report = {
        "epoch": epoch,
        **{f"train_{key}": value for key, value in training.items()},
        **{f"valid_{key}": value for key, value in validation.items() if key != "selector"},
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    history.append(report)
    print("ENERGY EPOCH", report, flush=True)
    if best is None or validation["rmse"] < best["rmse"] - 1e-4:
        best = {"epoch": epoch, **validation}
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": validation,
                "config": guards,
                "template_names": template_names,
                "base_menu": BASE_MENU,
            },
            checkpoint_path,
        )
        bad_epochs = 0
    else:
        bad_epochs += 1
    scheduler.step()
    if epoch >= MIN_EPOCHS and bad_epochs >= PATIENCE:
        print("ENERGY early stopping", flush=True)
        break

history_frame = pd.DataFrame(history)
history_frame.to_csv(OUT_DIR / "training_history.csv", index=False)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model"])
full, full_bywell = _evaluate(model, valid_wells, device, control="full")
no_gr, no_gr_bywell = _evaluate(model, valid_wells, device, control="no_gr")
shuffled, shuffled_bywell = _evaluate(
    model, valid_wells, device, control="shuffled_alignment"
)
full_bywell.to_csv(OUT_DIR / "fold2_model_bywell.csv", index=False)
no_gr_bywell.to_csv(OUT_DIR / "fold2_no_gr_bywell.csv", index=False)
shuffled_bywell.to_csv(OUT_DIR / "fold2_shuffled_bywell.csv", index=False)

controls = {
    "no_gr_rmse_degradation": float(no_gr["rmse"] - full["rmse"]),
    "shuffled_alignment_rmse_degradation": float(
        shuffled["rmse"] - full["rmse"]
    ),
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_candidate_conditioned_completewell_gateway",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_spatial_strong_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
        "selected_epoch": int(checkpoint["epoch"]),
    },
    "parameters": parameters,
    "guards": guards,
    "oracle_gate": oracle_gate,
    "model": full,
    "no_gr": no_gr,
    "shuffled_alignment": shuffled,
    "controls": controls,
    "gates": {
        "model_rmse_le": GATE_MODEL_RMSE,
        "model_median_well_rmse_le": GATE_MODEL_MEDIAN,
        "model_far_gt2000_rmse_le": GATE_MODEL_FAR,
        "each_control_degradation_ge": GATE_CONTROL_DEGRADATION,
        "target_5p5_rmse_le": 5.5,
    },
    "target_5p5_pass": bool(full["rmse"] <= 5.5),
    "strict_pass": bool(
        full["rmse"] <= GATE_MODEL_RMSE
        and full["median_well_rmse"] <= GATE_MODEL_MEDIAN
        and full["far_gt2000_rmse"] <= GATE_MODEL_FAR
        and controls["no_gr_rmse_degradation"] >= GATE_CONTROL_DEGRADATION
        and controls["shuffled_alignment_rmse_degradation"]
        >= GATE_CONTROL_DEGRADATION
    ),
    "checkpoint": str(checkpoint_path),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": float(
        __import__("shutil").disk_usage(OUT_DIR).free / (1024**3)
    ),
}
verdict["next"] = (
    "PASS -> repeat on untouched Fold1, then five-fold"
    if verdict["strict_pass"]
    else "FAIL -> complete-well evidence still cannot select the deployable bank"
)
_write_json(OUT_DIR / "verdict.json", verdict)
print("FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
