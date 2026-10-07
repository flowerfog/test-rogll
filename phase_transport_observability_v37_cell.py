"""Exact phase-envelope and positional-transport observability gate.

This cell intentionally does not train a model or consume historical prediction
layers.  Fold 3 selects a data term and dose; Fold 2 is touched once after the
recipe is frozen.  The experiment asks whether two literature-motivated data
terms add candidate-ranking information to the native pointwise GR emission.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


def _source_path() -> Path:
    configured = os.environ.get("ROGII_MISFIT_V1_SOURCE", "").strip()
    candidates = [
        Path("/root/autodl-tmp/misfit_family_point_fusion_fold2_v1_cell.py"),
        Path("misfit_family_point_fusion_fold2_v1_cell.py"),
    ]
    if configured:
        candidates.insert(0, Path(configured))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Upload misfit_family_point_fusion_fold2_v1_cell.py to /root/autodl-tmp"
    )


def _bootstrap() -> tuple[dict, Path]:
    path = _source_path()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STARTED"
            for target in node.targets
        ):
            break
        body.append(node)
    name = "phase_transport_source_v37"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.__dict__, path


V, SOURCE_PATH = _bootstrap()
OOF_ROOT = V["OOF_ROOT"]
STATE_OFFSETS = V["STATE_OFFSETS"].astype(np.float64)
REFERENCE_OFFSETS = V["V"]["REFERENCE_OFFSETS"].astype(np.float64)
N_STATES = len(STATE_OFFSETS)
ALIAS_FT = float(V["ALIAS_FT"])

_load_folds = V["_load_folds"]
_ensure_records = V["_ensure_records"]
_ensure_packs = V["_ensure_packs"]
_load_record = V["_load_record"]
_load_pack = V["_load_pack"]
_interp_matrix = V.get("_interp_matrix", V["V"].get("_interp_matrix"))
_stable_order = V["_stable_order"]
_truth_rank = V["_truth_rank"]
_score_at_offset = V["_score_at_offset"]
_interpolate_grid = V["_interpolate_grid"]
_metrics = V["_metrics"]
if _interp_matrix is None:
    raise RuntimeError("Cannot resolve _interp_matrix from v1 or its v0 source")

EXPERIMENT = "phase_transport_observability_v37"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260727
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 48
WINDOW_POINTS = 65
MIN_COVERAGE = 0.70
DOSES = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35)
QUANTILE_LEVELS = np.linspace(0.01, 0.99, 49, dtype=np.float64)
UNIT = np.linspace(-1.0, 1.0, WINDOW_POINTS, dtype=np.float64)

CONFIGS = (
    {"name": "hilbert_m512_t16", "family": "hilbert", "md_half": 512.0, "tvt_half": 16.0, "bidirectional": True},
    {"name": "hilbert_m1024_t32", "family": "hilbert", "md_half": 1024.0, "tvt_half": 32.0, "bidirectional": True},
    {"name": "posw2_b05_m512_t16", "family": "posw2", "beta": 0.5, "md_half": 512.0, "tvt_half": 16.0, "bidirectional": True},
    {"name": "posw2_b10_m512_t16", "family": "posw2", "beta": 1.0, "md_half": 512.0, "tvt_half": 16.0, "bidirectional": True},
    {"name": "posw2_b05_m1024_t32", "family": "posw2", "beta": 0.5, "md_half": 1024.0, "tvt_half": 32.0, "bidirectional": True},
    {"name": "hilbert_posw2_m512_t16", "family": "hybrid", "beta": 0.5, "md_half": 512.0, "tvt_half": 16.0, "bidirectional": True},
)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _window_arrays(record, pack, md_half, tvt_half, typewell_override=None):
    query_position = pack.grid_md.astype(np.float64)[:, None] + float(md_half) * UNIT[None]
    query = _interp_matrix(record.md, record.horizontal[0], query_position)
    query_valid = _interp_matrix(record.md, record.horizontal[1], query_position) > 0.50
    reference = record.typewell if typewell_override is None else typewell_override
    reference_position = STATE_OFFSETS[:, None] + float(tvt_half) * UNIT[None]
    ref = _interp_matrix(REFERENCE_OFFSETS, reference[0], reference_position)
    ref_valid = _interp_matrix(REFERENCE_OFFSETS, reference[1], reference_position) > 0.50
    return query, query_valid, ref, ref_valid


def _fill_detrend(values, valid):
    values = np.asarray(values, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(values)
    output = np.zeros_like(values)
    x = np.arange(values.shape[1], dtype=np.float64)
    for row in range(len(values)):
        mask = valid[row]
        if int(mask.sum()) < 3:
            continue
        filled = np.interp(x, x[mask], values[row, mask])
        design = np.column_stack((np.ones(len(x)), UNIT))
        coef = np.linalg.lstsq(design[mask], filled[mask], rcond=None)[0]
        output[row] = filled - design @ coef
    return output, valid


def _hilbert_envelope(values, valid):
    detrended, valid = _fill_detrend(values, valid)
    n = detrended.shape[1]
    multiplier = np.zeros(n, np.float64)
    multiplier[0] = 1.0
    if n % 2 == 0:
        multiplier[1 : n // 2] = 2.0
        multiplier[n // 2] = 1.0
    else:
        multiplier[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(np.fft.fft(detrended, axis=1) * multiplier[None], axis=1)
    return np.abs(analytic), valid


def _row_standardise(values, valid):
    valid = np.asarray(valid, bool) & np.isfinite(values)
    count = valid.sum(axis=1, keepdims=True)
    mean = np.sum(np.where(valid, values, 0.0), axis=1, keepdims=True) / np.maximum(count, 1)
    centred = np.where(valid, values - mean, 0.0)
    scale = np.sqrt(np.sum(centred * centred, axis=1, keepdims=True) / np.maximum(count, 1))
    return centred / np.maximum(scale, 1.0e-4), valid


def _pairwise_ncc(query, query_valid, reference, reference_valid):
    query, query_valid = _row_standardise(query, query_valid)
    reference, reference_valid = _row_standardise(reference, reference_valid)
    qv = query_valid.astype(np.float64)
    rv = reference_valid.astype(np.float64)
    numerator = reference @ query.T
    left = (reference * reference) @ qv.T
    right = rv @ (query * query).T
    joint = rv @ qv.T
    score = numerator / np.sqrt(np.maximum(left * right, 1.0e-12))
    return np.where(joint >= MIN_COVERAGE * WINDOW_POINTS, score, -40.0)


def _density(values, valid, beta):
    filled, valid = _fill_detrend(values, valid)
    z, valid = _row_standardise(filled, valid)
    mass = np.exp(float(beta) * np.clip(z, -4.0, 4.0)) * valid
    total = mass.sum(axis=1, keepdims=True)
    usable = (valid.mean(axis=1) >= MIN_COVERAGE) & (total[:, 0] > 0.0)
    mass = mass / np.maximum(total, 1.0e-12)
    return mass, usable


def _inverse_cdf(mass):
    cdf = np.cumsum(mass, axis=1)
    indices = np.sum(cdf[:, :, None] < QUANTILE_LEVELS[None, None, :], axis=1)
    indices = np.clip(indices, 0, mass.shape[1] - 1)
    previous = np.maximum(indices - 1, 0)
    c1 = np.take_along_axis(cdf, indices, axis=1)
    c0 = np.where(indices > 0, np.take_along_axis(cdf, previous, axis=1), 0.0)
    x1 = UNIT[indices]
    x0 = np.where(indices > 0, UNIT[previous], UNIT[0])
    fraction = (QUANTILE_LEVELS[None] - c0) / np.maximum(c1 - c0, 1.0e-12)
    return x0 + np.clip(fraction, 0.0, 1.0) * (x1 - x0)


def _positional_w2(query, query_valid, reference, reference_valid, beta):
    q_mass, q_usable = _density(query, query_valid, beta)
    r_mass, r_usable = _density(reference, reference_valid, beta)
    q_quantile = _inverse_cdf(q_mass)
    r_quantile = _inverse_cdf(r_mass)
    output = np.empty((len(reference), len(query)), np.float64)
    for start in range(0, len(query), 256):
        stop = min(start + 256, len(query))
        delta = r_quantile[:, None, :] - q_quantile[None, start:stop, :]
        output[:, start:stop] = -np.mean(delta * delta, axis=2)
    usable = r_usable[:, None] & q_usable[None]
    return np.where(usable, output, -40.0)


def _normalise_states(score):
    score = np.asarray(score, np.float64)
    valid = np.isfinite(score) & (score > -39.0)
    output = np.full(score.shape, -8.0, np.float64)
    usable = valid.sum(axis=0) >= 32
    if usable.any():
        local = np.where(valid[:, usable], score[:, usable], np.nan)
        centre = np.nanmedian(local, axis=0, keepdims=True)
        q25 = np.nanquantile(local, 0.25, axis=0, keepdims=True)
        q75 = np.nanquantile(local, 0.75, axis=0, keepdims=True)
        scale = np.maximum((q75 - q25) / 1.349, 1.0e-3)
        output[:, usable] = np.where(
            np.isfinite(local), np.clip((local - centre) / scale, -8.0, 8.0), -8.0
        )
    return output, usable


def _exact_profile(record, pack, config, typewell_override=None):
    query, q_valid, reference, r_valid = _window_arrays(
        record, pack, config["md_half"], config["tvt_half"], typewell_override
    )
    family = config["family"]
    if family in ("hilbert", "hybrid"):
        q_env, q_env_valid = _hilbert_envelope(query, q_valid)
        r_env, r_env_valid = _hilbert_envelope(reference, r_valid)
        hilbert = _pairwise_ncc(q_env, q_env_valid, r_env, r_env_valid)
        if config.get("bidirectional", False):
            hilbert = np.maximum(
                hilbert,
                _pairwise_ncc(q_env, q_env_valid, r_env[:, ::-1], r_env_valid[:, ::-1]),
            )
        if family == "hilbert":
            return hilbert
    w2 = _positional_w2(query, q_valid, reference, r_valid, config.get("beta", 0.5))
    if config.get("bidirectional", False):
        w2 = np.maximum(
            w2,
            _positional_w2(
                query, q_valid, reference[:, ::-1], r_valid[:, ::-1],
                config.get("beta", 0.5),
            ),
        )
    if family == "posw2":
        return w2
    hz, hu = _normalise_states(hilbert)
    wz, wu = _normalise_states(w2)
    output = np.full_like(hz, -40.0)
    usable = hu & wu
    output[:, usable] = 0.5 * hz[:, usable] + 0.5 * wz[:, usable]
    return output


_PROFILE_CACHE = {}


def _blend_scores(record, pack, config, dose, typewell_override=None, state_permutation=None):
    point = pack.emissions[0].astype(np.float64)
    cache_key = None
    if typewell_override is None:
        cache_key = (str(pack.well), config["name"])
    if cache_key is not None and cache_key in _PROFILE_CACHE:
        profile = _PROFILE_CACHE[cache_key]
    else:
        profile = _exact_profile(record, pack, config, typewell_override)
        if cache_key is not None:
            _PROFILE_CACHE[cache_key] = profile.astype(np.float32)
    if state_permutation is not None:
        profile = profile[state_permutation]
    point_z, _ = _normalise_states(point)
    profile_z, usable = _normalise_states(profile)
    blended = point_z.copy()
    blended[:, usable] += float(dose) * profile_z[:, usable]
    return point, profile, blended, usable


def _path_metrics(items):
    truths, predictions, distances, wells = [], [], [], []
    for pack, offset in items:
        truths.append(pack.truth.astype(np.float64))
        support = pack.grid_md > pack.anchor_md
        md = np.r_[pack.anchor_md, pack.grid_md[support]]
        values = np.r_[0.0, np.asarray(offset)[support]]
        predictions.append(pack.anchor_tvt + np.interp(pack.eval_md, md, values))
        distances.append(pack.eval_distance.astype(np.float64))
        wells.append(np.full(pack.eval_rows, pack.well, dtype="U16"))
    result, _, _ = _metrics(
        np.concatenate(truths), np.concatenate(predictions),
        np.concatenate(distances), np.concatenate(wells),
    )
    return result


def _evaluate(wells, records, packs, config, dose, control="full", donor_by_well=None):
    ranks, point_ranks, profile_ranks = [], [], []
    distances, targets, argmax_values, alias_wins = [], [], [], []
    path_items = []
    usable_rows = total_rows = 0
    permutation = None
    if control == "state_shuffle":
        permutation = np.random.default_rng(SEED + 991).permutation(N_STATES)
    for number, well in enumerate(wells, start=1):
        record, pack = records[well], packs[well]
        override = records[donor_by_well[well]].typewell if control == "donor_shuffle" else None
        point, profile, blended, usable = _blend_scores(
            record, pack, config, dose, override, permutation
        )
        order = _stable_order(blended)
        point_order = _stable_order(point)
        profile_order = _stable_order(profile)
        target = pack.target.astype(np.float64)
        mask = pack.supervised.astype(bool) & (
            np.abs(target) <= np.max(np.abs(STATE_OFFSETS)) - 5.0
        )
        rank = _truth_rank(order, target)
        point_rank = _truth_rank(point_order, target)
        profile_rank = _truth_rank(profile_order, target)
        argmax = STATE_OFFSETS[order[0]]
        argmax[: pack.anchor_grid + 1] = 0.0
        true_score = _score_at_offset(blended, target)
        plus = _score_at_offset(blended, target + ALIAS_FT)
        minus = _score_at_offset(blended, target - ALIAS_FT)
        alias_mask = mask & (
            np.abs(target) <= np.max(np.abs(STATE_OFFSETS)) - ALIAS_FT
        )
        ranks.append(rank[mask])
        point_ranks.append(point_rank[mask])
        profile_ranks.append(profile_rank[mask])
        distances.append((pack.grid_md - pack.anchor_md)[mask])
        targets.append(target[mask])
        argmax_values.append(argmax[mask])
        alias_wins.append((true_score > np.maximum(plus, minus))[alias_mask])
        path_items.append((pack, argmax))
        usable_rows += int((usable & mask).sum())
        total_rows += int(mask.sum())
        if number % 25 == 0 or number == len(wells):
            print(f"V37 {control} {config['name']} dose={dose:g} {number}/{len(wells)}", flush=True)
    rank = np.concatenate(ranks)
    point_rank = np.concatenate(point_ranks)
    profile_rank = np.concatenate(profile_ranks)
    distance = np.concatenate(distances)
    target = np.concatenate(targets)
    argmax = np.concatenate(argmax_values)
    alias = np.concatenate(alias_wins)
    deep = distance > 2000.0
    error = argmax - target
    path = _path_metrics(path_items)
    return {
        "profile": config["name"], "dose": float(dose), "control": control,
        "wells": int(len(wells)), "grid_rows": int(len(rank)),
        "usable_fraction": float(usable_rows / max(total_rows, 1)),
        "median_rank": float(np.median(rank)),
        "top8_recall": float(np.mean(rank <= 8)),
        "top32_recall": float(np.mean(rank <= 32)),
        "deep_top32_recall": float(np.mean(rank[deep] <= 32)),
        "alias24_win_rate": float(np.mean(alias)),
        "argmax_grid_rmse": float(np.sqrt(np.mean(error * error))),
        "path_rmse": float(path["rmse"]),
        "path_far_gt2000_rmse": float(path["far_gt2000_rmse"]),
        "union_top16_each_recall": float(np.mean((point_rank <= 16) | (profile_rank <= 16))),
        "profile_rescues_point_top32": float(np.mean((point_rank > 32) & (profile_rank <= 32))),
        "deep_profile_rescues_point_top32": float(
            np.mean((point_rank[deep] > 32) & (profile_rank[deep] <= 32))
        ),
    }


def _objective(row):
    return (
        1.25 * row["top32_recall"] + row["deep_top32_recall"]
        + 0.50 * row["alias24_win_rate"]
        + 0.35 * row["profile_rescues_point_top32"]
        - 0.001 * row["argmax_grid_rmse"]
    )


STARTED = time.time()
folds, fold_path = _load_folds()
folds["well"] = folds["well"].astype(str)
fold3_wells = folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].tolist()
fold2_wells = folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].tolist()
if len(fold3_wells) < 154 or len(fold2_wells) < 154:
    raise RuntimeError(f"Unexpected Fold3/Fold2 sizes: {len(fold3_wells)}/{len(fold2_wells)}")

rng = np.random.default_rng(SEED)
fold3_order = np.asarray(fold3_wells)[rng.permutation(len(fold3_wells))]
discovery_wells = fold3_order[:DISCOVERY_WELLS].tolist()
confirmation_wells = fold3_order[DISCOVERY_WELLS:].tolist()
needed = sorted(set(fold3_wells + fold2_wells))
_ensure_records(needed)
_ensure_packs(needed)
records = {well: _load_record(well) for well in needed}
packs = {well: _load_pack(well) for well in needed}

selection_rows = []
for config in CONFIGS:
    for dose in DOSES:
        row = _evaluate(discovery_wells, records, packs, config, dose)
        row["objective"] = _objective(row)
        selection_rows.append(row)
selection = pd.DataFrame(selection_rows).sort_values(
    ["objective", "top32_recall", "deep_top32_recall"], ascending=False
)
selection.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
selected_row = selection.iloc[0]
selected = next(config for config in CONFIGS if config["name"] == selected_row["profile"])
selected_dose = float(selected_row["dose"])

point_baseline = _evaluate(confirmation_wells, records, packs, selected, 0.0)
fold3_confirmation = _evaluate(
    confirmation_wells, records, packs, selected, selected_dose
)

# Freeze before Fold 2. Controls alter only the new branch; the point emission is unchanged.
donor_order = np.asarray(fold2_wells)[rng.permutation(len(fold2_wells))]
donor_by_well = dict(zip(fold2_wells, np.roll(donor_order, 1)))
fold2_baseline = _evaluate(fold2_wells, records, packs, selected, 0.0)
fold2_selected = _evaluate(fold2_wells, records, packs, selected, selected_dose)
fold2_state_shuffle = _evaluate(
    fold2_wells, records, packs, selected, selected_dose, control="state_shuffle"
)
fold2_donor_shuffle = _evaluate(
    fold2_wells, records, packs, selected, selected_dose,
    control="donor_shuffle", donor_by_well=donor_by_well,
)

summary = pd.DataFrame([
    {"stage": "fold3_confirmation_baseline", **point_baseline},
    {"stage": "fold3_confirmation_selected", **fold3_confirmation},
    {"stage": "fold2_baseline", **fold2_baseline},
    {"stage": "fold2_selected", **fold2_selected},
    {"stage": "fold2_state_shuffle", **fold2_state_shuffle},
    {"stage": "fold2_donor_shuffle", **fold2_donor_shuffle},
])
summary.to_csv(OUT_DIR / "confirmation_and_controls.csv", index=False)

fold3_gain = fold3_confirmation["top32_recall"] - point_baseline["top32_recall"]
fold3_deep_gain = fold3_confirmation["deep_top32_recall"] - point_baseline["deep_top32_recall"]
fold2_gain = fold2_selected["top32_recall"] - fold2_baseline["top32_recall"]
fold2_deep_gain = fold2_selected["deep_top32_recall"] - fold2_baseline["deep_top32_recall"]
shuffle_gain = fold2_state_shuffle["top32_recall"] - fold2_baseline["top32_recall"]
donor_gain = fold2_donor_shuffle["top32_recall"] - fold2_baseline["top32_recall"]

gates = {
    "fold3_confirmation_top32_gain_ge_0p02": bool(fold3_gain >= 0.02),
    "fold3_confirmation_deep_gain_ge_0p02": bool(fold3_deep_gain >= 0.02),
    "fold2_top32_gain_ge_0p02": bool(fold2_gain >= 0.02),
    "fold2_deep_gain_ge_0p02": bool(fold2_deep_gain >= 0.02),
    "fold2_alias_nonworse": bool(
        fold2_selected["alias24_win_rate"] >= fold2_baseline["alias24_win_rate"] - 0.005
    ),
    "state_shuffle_retains_le_30pct_gain": bool(
        shuffle_gain <= max(0.30 * fold2_gain, 0.002)
    ),
    "donor_shuffle_retains_le_30pct_gain": bool(
        donor_gain <= max(0.30 * fold2_gain, 0.002)
    ),
}
strict_pass = bool(all(gates.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_exact_data_term_selection_Fold2_once_only_confirmation",
    "source": str(SOURCE_PATH),
    "raw_inputs_only": True,
    "uses_cp_neighbor_gold_strong_mainline": False,
    "corrections_vs_old_misfit_test": {
        "hilbert": "analytic-signal amplitude after within-window detrending",
        "wasserstein": "inverse-CDF W2 over window position; GR amplitudes are not sorted",
    },
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery_wells),
        "internal_confirmation_wells": len(confirmation_wells),
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "selected_config": selected | {"dose": selected_dose},
    "fold3_confirmation": fold3_confirmation,
    "fold3_baseline": point_baseline,
    "fold2": fold2_selected,
    "fold2_baseline": fold2_baseline,
    "controls": {
        "state_shuffle": fold2_state_shuffle,
        "donor_typewell_shuffle": fold2_donor_shuffle,
    },
    "gains": {
        "fold3_top32": fold3_gain,
        "fold3_deep_top32": fold3_deep_gain,
        "fold2_top32": fold2_gain,
        "fold2_deep_top32": fold2_deep_gain,
        "fold2_state_shuffle_top32": shuffle_gain,
        "fold2_donor_shuffle_top32": donor_gain,
    },
    "gates": gates,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> add the frozen data term to a sequence decoder; do not tune it again"
        if strict_pass else
        "MIXED -> positional W2 transfers but donor specificity fails; decompose generic-depth and exact-reference evidence before decoder integration"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV37 DISCOVERY TOP 20")
print(selection.head(20).to_string(index=False))
print("\nV37 CONFIRMATION AND CONTROLS")
print(summary.to_string(index=False))
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
