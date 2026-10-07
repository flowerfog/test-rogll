"""Nested zero-training audit of windowed GR misfit families.

This experiment keeps the Fold-2 split, 257-state TVT axis, native records,
and path interpolation used by the existing cost-map experiments.  It changes
only the legal GR-to-typewell emission.  Forty-eight Fold-2 wells select one
pre-registered profile; the remaining 107 wells confirm it once.  Fold 1 is
untouched.

The goal is deliberately narrower than path modelling: determine whether a
windowed emission ranks the true state materially better than the current
pointwise typewell emission before spending more compute on another decoder.
"""

from __future__ import annotations

from pathlib import Path
import ast
import json
import shutil
import sys
import time
import types

import numpy as np
import pandas as pd


def _bootstrap_costmap():
    candidates = (
        Path("/root/autodl-tmp/inverted_costmap_transformer_fold2_v1_cell.py"),
        Path.cwd() / "inverted_costmap_transformer_fold2_v1_cell.py",
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Upload inverted_costmap_transformer_fold2_v1_cell.py, "
            "inverted_transformer_fullwell_fold2_v0_cell.py, and "
            "native_resolution_dualref_capacity_v0_cell.py to /root/autodl-tmp."
        )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STARTED"
            for target in node.targets
        ):
            break
        body.append(node)
    module_name = "misfit_observability_costmap_source_v0"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"),
        module.__dict__,
    )
    return module.__dict__, source_path


V, SOURCE_PATH = _bootstrap_costmap()
OOF_ROOT = V["OOF_ROOT"]
PACK_DIR = V["PACK_DIR"]
STATE_OFFSETS = V["STATE_OFFSETS"].astype(np.float64)
# cost-map module -> iTransformer module -> native record-builder module
REFERENCE_OFFSETS = V["V"]["V"]["REFERENCE_OFFSETS"].astype(np.float64)
N_STATES = len(STATE_OFFSETS)
_load_folds = V["_load_folds"]
_ensure_records = V["_ensure_records"]
_ensure_packs = V["_ensure_packs"]
_load_record = V["_load_record"]
_load_pack = V["_load_pack"]
_metrics = V["_metrics"]

EXPERIMENT = "misfit_family_observability_fold2_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260724
VALID_FOLD = 2
SELECTION_WELLS = 48
WINDOW_POINTS = 33
MIN_WINDOW_COVERAGE = 0.70
ALIAS_FT = 24.0
TOP_K = (1, 8, 32, 64, 128)

# All configurations are declared before labels are inspected.  Values are
# half-widths around a station, measured in their native coordinate systems.
CONFIGS = (
    {"name": "point_typewell", "family": "point", "md_half": 0.0, "tvt_half": 0.0},
    {"name": "ncc_m256_t8", "family": "ncc", "md_half": 256.0, "tvt_half": 8.0},
    {"name": "ncc_m512_t16", "family": "ncc", "md_half": 512.0, "tvt_half": 16.0},
    {"name": "ncc_m1024_t32", "family": "ncc", "md_half": 1024.0, "tvt_half": 32.0},
    {
        "name": "envelope_m512_t16",
        "family": "envelope",
        "md_half": 512.0,
        "tvt_half": 16.0,
    },
    {
        "name": "envelope_m1024_t32",
        "family": "envelope",
        "md_half": 1024.0,
        "tvt_half": 32.0,
    },
    {
        "name": "wasserstein_m512_t16",
        "family": "wasserstein",
        "md_half": 512.0,
        "tvt_half": 16.0,
    },
    {
        "name": "wasserstein_m1024_t32",
        "family": "wasserstein",
        "md_half": 1024.0,
        "tvt_half": 32.0,
    },
    {
        "name": "segment_consensus_m1024_t32",
        "family": "segment_consensus",
        "md_half": 1024.0,
        "tvt_half": 32.0,
    },
    {
        "name": "cow_proxy_m1024_t32",
        "family": "cow_proxy",
        "md_half": 1024.0,
        "tvt_half": 32.0,
    },
)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _interp_matrix(x, values, query):
    x = np.asarray(x, np.float64)
    values = np.asarray(values, np.float64)
    query = np.asarray(query, np.float64)
    finite = np.isfinite(x) & np.isfinite(values)
    if int(finite.sum()) < 2:
        return np.full(query.shape, np.nan, np.float64)
    xv = x[finite]
    yv = values[finite]
    order = np.argsort(xv, kind="mergesort")
    xv = xv[order]
    yv = yv[order]
    return np.interp(query.ravel(), xv, yv, left=np.nan, right=np.nan).reshape(query.shape)


def _window_arrays(record, pack, md_half, tvt_half, typewell_override=None):
    unit = np.linspace(-1.0, 1.0, WINDOW_POINTS, dtype=np.float64)
    query_position = pack.grid_md.astype(np.float64)[:, None] + float(md_half) * unit[None]
    query = _interp_matrix(record.md, record.horizontal[0], query_position)
    query_valid = _interp_matrix(record.md, record.horizontal[1], query_position) > 0.50

    reference = record.typewell if typewell_override is None else typewell_override
    reference_position = STATE_OFFSETS[:, None] + float(tvt_half) * unit[None]
    ref = _interp_matrix(REFERENCE_OFFSETS, reference[0], reference_position)
    ref_valid = _interp_matrix(REFERENCE_OFFSETS, reference[1], reference_position) > 0.50
    return query, query_valid, ref, ref_valid


def _row_standardise(values, valid):
    values = np.asarray(values, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(values)
    count = valid.sum(axis=1, keepdims=True)
    mean = np.sum(np.where(valid, values, 0.0), axis=1, keepdims=True) / np.maximum(count, 1)
    centred = np.where(valid, values - mean, 0.0)
    scale = np.sqrt(np.sum(centred * centred, axis=1, keepdims=True) / np.maximum(count, 1))
    return centred / np.maximum(scale, 1.0e-4), valid


def _pairwise_ncc(query, query_valid, reference, reference_valid):
    q, qv = _row_standardise(query, query_valid)
    r, rv = _row_standardise(reference, reference_valid)
    qv_f = qv.astype(np.float64)
    rv_f = rv.astype(np.float64)
    numerator = r @ q.T
    left_energy = (r * r) @ qv_f.T
    right_energy = rv_f @ (q * q).T
    denominator = np.sqrt(np.maximum(left_energy * right_energy, 1.0e-12))
    joint = rv_f @ qv_f.T
    score = numerator / denominator
    return np.where(joint >= MIN_WINDOW_COVERAGE * query.shape[1], score, -40.0)


def _smooth_rows(values, valid, width=7):
    kernel = np.ones(int(width), np.float64)
    output = np.zeros_like(values, np.float64)
    output_valid = np.zeros_like(valid, bool)
    for row in range(values.shape[0]):
        mask = valid[row] & np.isfinite(values[row])
        sums = np.convolve(np.where(mask, values[row], 0.0), kernel, mode="same")
        counts = np.convolve(mask.astype(np.float64), kernel, mode="same")
        output_valid[row] = counts >= max(2.0, 0.50 * width)
        output[row] = np.divide(sums, np.maximum(counts, 1.0))
    return output, output_valid


def _wasserstein_score(query, query_valid, reference, reference_valid):
    q, qv = _row_standardise(query, query_valid)
    r, rv = _row_standardise(reference, reference_valid)
    q_coverage = qv.mean(axis=1)
    r_coverage = rv.mean(axis=1)
    # Invalid entries are replaced by the row median (zero after centering).
    q_sorted = np.sort(np.where(qv, q, 0.0), axis=1)
    r_sorted = np.sort(np.where(rv, r, 0.0), axis=1)
    score = -np.mean(np.abs(r_sorted[:, None, :] - q_sorted[None, :, :]), axis=2)
    valid = (r_coverage[:, None] >= MIN_WINDOW_COVERAGE) & (
        q_coverage[None, :] >= MIN_WINDOW_COVERAGE
    )
    return np.where(valid, score, -40.0)


def _segment_consensus(query, query_valid, reference, reference_valid):
    segment_scores = []
    for indices in np.array_split(np.arange(WINDOW_POINTS), 4):
        segment_scores.append(
            _pairwise_ncc(
                query[:, indices],
                query_valid[:, indices],
                reference[:, indices],
                reference_valid[:, indices],
            )
        )
    stacked = np.stack(segment_scores, axis=0)
    invalid = np.any(stacked <= -39.0, axis=0)
    score = np.mean(stacked, axis=0) - 0.35 * np.std(stacked, axis=0)
    return np.where(invalid, -40.0, score)


def _cow_proxy(record, pack, md_half, tvt_half, typewell_override=None):
    # COW-style local stretch proxy: four window segments independently choose
    # among three reference dilation scales, with a penalty for scale changes.
    query, query_valid, _, _ = _window_arrays(
        record, pack, md_half, tvt_half, typewell_override=typewell_override
    )
    scale_scores_forward = []
    scale_scores_reverse = []
    for multiplier in (0.75, 1.0, 1.25):
        _, _, reference, reference_valid = _window_arrays(
            record,
            pack,
            md_half,
            tvt_half * multiplier,
            typewell_override=typewell_override,
        )
        per_segment_forward = []
        per_segment_reverse = []
        for indices in np.array_split(np.arange(WINDOW_POINTS), 4):
            per_segment_forward.append(
                _pairwise_ncc(
                    query[:, indices],
                    query_valid[:, indices],
                    reference[:, indices],
                    reference_valid[:, indices],
                )
            )
            reverse_indices = WINDOW_POINTS - 1 - indices
            per_segment_reverse.append(
                _pairwise_ncc(
                    query[:, indices],
                    query_valid[:, indices],
                    reference[:, reverse_indices],
                    reference_valid[:, reverse_indices],
                )
            )
        scale_scores_forward.append(np.stack(per_segment_forward, axis=0))
        scale_scores_reverse.append(np.stack(per_segment_reverse, axis=0))

    def collapse(scale_scores):
        scores = np.stack(scale_scores, axis=0)  # scale, segment, state, position
        best_scale = np.argmax(scores, axis=0)
        best = np.max(scores, axis=0)
        invalid = np.any(best <= -39.0, axis=0)
        changes = np.mean(np.abs(np.diff(best_scale, axis=0)), axis=0)
        score = np.mean(best, axis=0) - 0.10 * changes
        return np.where(invalid, -40.0, score)

    return np.maximum(collapse(scale_scores_forward), collapse(scale_scores_reverse))


def _profile(record, pack, config, typewell_override=None, no_gr=False):
    family = config["family"]
    if no_gr:
        return np.zeros((N_STATES, len(pack.grid_md)), np.float32)
    if family == "point":
        if typewell_override is None:
            return pack.emissions[0].astype(np.float32)
        return V["_build_pack"](record, typewell_override=typewell_override).emissions[0]
    if family == "cow_proxy":
        return _cow_proxy(
            record,
            pack,
            config["md_half"],
            config["tvt_half"],
            typewell_override=typewell_override,
        ).astype(np.float32)
    query, query_valid, reference, reference_valid = _window_arrays(
        record,
        pack,
        config["md_half"],
        config["tvt_half"],
        typewell_override=typewell_override,
    )
    if family == "ncc":
        score = np.maximum(
            _pairwise_ncc(query, query_valid, reference, reference_valid),
            _pairwise_ncc(
                query, query_valid, reference[:, ::-1], reference_valid[:, ::-1]
            ),
        )
    elif family == "envelope":
        query, query_valid = _smooth_rows(query, query_valid)
        reference, reference_valid = _smooth_rows(reference, reference_valid)
        score = np.maximum(
            _pairwise_ncc(query, query_valid, reference, reference_valid),
            _pairwise_ncc(
                query, query_valid, reference[:, ::-1], reference_valid[:, ::-1]
            ),
        )
    elif family == "wasserstein":
        score = _wasserstein_score(query, query_valid, reference, reference_valid)
    elif family == "segment_consensus":
        score = np.maximum(
            _segment_consensus(query, query_valid, reference, reference_valid),
            _segment_consensus(
                query, query_valid, reference[:, ::-1], reference_valid[:, ::-1]
            ),
        )
    else:
        raise ValueError(f"Unknown family: {family}")
    return score.astype(np.float32)


def _stable_order(score):
    return np.argsort(-score, axis=0, kind="stable")


def _truth_rank(order, target):
    inverse = np.empty_like(order)
    inverse[order, np.arange(order.shape[1])] = np.arange(N_STATES)[:, None]
    distance = np.abs(STATE_OFFSETS[:, None] - target[None])
    nearest = distance <= np.min(distance, axis=0, keepdims=True) + 1.0e-6
    return np.min(np.where(nearest, inverse, N_STATES), axis=0) + 1


def _score_at_offset(score, offset):
    state = np.argmin(np.abs(STATE_OFFSETS[:, None] - offset[None]), axis=0)
    return score[state, np.arange(score.shape[1])]


def _interpolate_grid(pack, grid_offset):
    support = pack.grid_md > pack.anchor_md
    support_md = np.r_[pack.anchor_md, pack.grid_md[support]]
    support_offset = np.r_[0.0, np.asarray(grid_offset)[support]]
    return pack.anchor_tvt + np.interp(pack.eval_md, support_md, support_offset)


def _path_metrics(items):
    truths, predictions, distances, wells = [], [], [], []
    for pack, offset in items:
        truths.append(pack.truth.astype(np.float64))
        predictions.append(_interpolate_grid(pack, offset))
        distances.append(pack.eval_distance.astype(np.float64))
        wells.append(np.full(pack.eval_rows, pack.well, dtype="U16"))
    metrics, _, _ = _metrics(
        np.concatenate(truths),
        np.concatenate(predictions),
        np.concatenate(distances),
        np.concatenate(wells),
    )
    return metrics


def _evaluate(wells, records, packs, config, controls=None, donor_by_well=None):
    controls = controls or "full"
    ranks, targets, distances, argmax_values, alias_wins = [], [], [], [], []
    path_items = []
    observable_rows = 0
    total_rows = 0
    finite_profile_rows = 0

    for well in wells:
        record = records[well]
        pack = packs[well]
        override = None
        if controls == "shuffled_typewell":
            override = records[donor_by_well[well]].typewell
        score = _profile(
            record,
            pack,
            config,
            typewell_override=override,
            no_gr=controls == "no_gr",
        )
        order = _stable_order(score)
        mask = pack.supervised.astype(bool)
        target = pack.target.astype(np.float64)
        observable = mask & (np.abs(target) <= np.max(np.abs(STATE_OFFSETS)) - 5.0)
        rank = _truth_rank(order, target)
        argmax = STATE_OFFSETS[order[0]]
        argmax[: pack.anchor_grid + 1] = 0.0

        true_score = _score_at_offset(score, target)
        alias_plus = _score_at_offset(score, target + ALIAS_FT)
        alias_minus = _score_at_offset(score, target - ALIAS_FT)
        alias_valid = observable & (np.abs(target) <= np.max(np.abs(STATE_OFFSETS)) - ALIAS_FT)

        ranks.append(rank[observable])
        targets.append(target[observable])
        distances.append((pack.grid_md - pack.anchor_md)[observable])
        argmax_values.append(argmax[observable])
        alias_wins.append((true_score > np.maximum(alias_plus, alias_minus))[alias_valid])
        path_items.append((pack, argmax))
        total_rows += int(mask.sum())
        observable_rows += int(observable.sum())
        finite_profile_rows += int(np.any(score > -39.0, axis=0)[mask].sum())

    rank = np.concatenate(ranks)
    target = np.concatenate(targets)
    distance = np.concatenate(distances)
    argmax = np.concatenate(argmax_values)
    error = argmax - target
    deep = distance > 2000.0
    alias = np.concatenate(alias_wins)
    path = _path_metrics(path_items)
    result = {
        "profile": config["name"],
        "control": controls,
        "wells": int(len(wells)),
        "grid_rows": int(len(rank)),
        "observable_fraction": float(observable_rows / max(total_rows, 1)),
        "finite_profile_fraction": float(finite_profile_rows / max(total_rows, 1)),
        "median_rank": float(np.median(rank)),
        "p90_rank": float(np.quantile(rank, 0.90)),
        "top1_recall": float(np.mean(rank <= 1)),
        "top8_recall": float(np.mean(rank <= 8)),
        "top32_recall": float(np.mean(rank <= 32)),
        "top64_recall": float(np.mean(rank <= 64)),
        "top128_recall": float(np.mean(rank <= 128)),
        "argmax_grid_rmse": float(np.sqrt(np.mean(error * error))),
        "argmax_within5": float(np.mean(np.abs(error) <= 5.0)),
        "argmax_edge16_fraction": float(
            np.mean(np.abs(argmax) >= np.max(np.abs(STATE_OFFSETS)) - 16.0)
        ),
        "alias24_win_rate": float(np.mean(alias)) if len(alias) else np.nan,
        "deep_rows": int(deep.sum()),
        "deep_median_rank": float(np.median(rank[deep])) if deep.any() else np.nan,
        "deep_top8_recall": float(np.mean(rank[deep] <= 8)) if deep.any() else np.nan,
        "deep_top32_recall": float(np.mean(rank[deep] <= 32)) if deep.any() else np.nan,
        "path_rmse": float(path["rmse"]),
        "path_far_gt2000_rmse": float(path["far_gt2000_rmse"]),
    }
    return result


def _selection_objective(row):
    # Ranking is the purpose of this gate.  Argmax is retained as a guard, not
    # allowed to dominate a profile whose useful state remains in top-K.
    return (
        1.50 * row["top8_recall"]
        + row["top32_recall"]
        + 0.50 * row["alias24_win_rate"]
        - 0.0025 * row["median_rank"]
        - 0.0010 * row["argmax_grid_rmse"]
    )


STARTED = time.time()
folds, fold_path = _load_folds()
folds["well"] = folds["well"].astype(str)
valid_wells = folds.loc[folds["fold"].eq(VALID_FOLD), "well"].tolist()
if len(valid_wells) != 155:
    raise RuntimeError(f"Expected 155 Fold2 wells, found {len(valid_wells)}")

rng = np.random.default_rng(SEED)
shuffled = np.asarray(valid_wells)[rng.permutation(len(valid_wells))]
selection_wells = shuffled[:SELECTION_WELLS].tolist()
confirmation_wells = shuffled[SELECTION_WELLS:].tolist()
_ensure_records(valid_wells)
_ensure_packs(valid_wells)
records = {well: _load_record(well) for well in valid_wells}
packs = {well: _load_pack(well) for well in valid_wells}

selection_rows = []
for number, config in enumerate(CONFIGS, 1):
    row = _evaluate(selection_wells, records, packs, config)
    row["selection_objective"] = _selection_objective(row)
    selection_rows.append(row)
    print(
        "MISFIT SELECTION",
        f"{number}/{len(CONFIGS)}",
        config["name"],
        {key: row[key] for key in ("median_rank", "top8_recall", "top32_recall", "alias24_win_rate", "argmax_grid_rmse")},
        flush=True,
    )

selection_frame = pd.DataFrame(selection_rows).sort_values(
    "selection_objective", ascending=False
)
selection_frame.to_csv(OUT_DIR / "selection_profiles.csv", index=False)
selected_name = str(selection_frame.iloc[0]["profile"])
selected_config = next(config for config in CONFIGS if config["name"] == selected_name)

# Confirmation reports every pre-registered family for diagnosis, but only the
# selection winner is eligible for the promotion decision.
confirmation_rows = []
for number, config in enumerate(CONFIGS, 1):
    row = _evaluate(confirmation_wells, records, packs, config)
    confirmation_rows.append(row)
    print(
        "MISFIT CONFIRMATION",
        f"{number}/{len(CONFIGS)}",
        config["name"],
        {key: row[key] for key in ("median_rank", "top8_recall", "top32_recall", "alias24_win_rate", "argmax_grid_rmse")},
        flush=True,
    )

confirmation_frame = pd.DataFrame(confirmation_rows)
confirmation_frame.to_csv(OUT_DIR / "confirmation_profiles.csv", index=False)
baseline = next(row for row in confirmation_rows if row["profile"] == "point_typewell")
selected = next(row for row in confirmation_rows if row["profile"] == selected_name)

donor_cycle = confirmation_wells.copy()
rng.shuffle(donor_cycle)
donor_by_well = {
    donor_cycle[index]: donor_cycle[(index + 1) % len(donor_cycle)]
    for index in range(len(donor_cycle))
}
no_gr = _evaluate(
    confirmation_wells, records, packs, selected_config, controls="no_gr"
)
shuffled_typewell = _evaluate(
    confirmation_wells,
    records,
    packs,
    selected_config,
    controls="shuffled_typewell",
    donor_by_well=donor_by_well,
)

gains = {
    "median_rank_reduction": float(baseline["median_rank"] - selected["median_rank"]),
    "top8_recall_gain": float(selected["top8_recall"] - baseline["top8_recall"]),
    "top32_recall_gain": float(selected["top32_recall"] - baseline["top32_recall"]),
    "deep_top32_recall_gain": float(
        selected["deep_top32_recall"] - baseline["deep_top32_recall"]
    ),
    "alias24_win_gain": float(selected["alias24_win_rate"] - baseline["alias24_win_rate"]),
    "argmax_grid_rmse_gain": float(
        baseline["argmax_grid_rmse"] - selected["argmax_grid_rmse"]
    ),
    "shuffled_top32_drop": float(
        selected["top32_recall"] - shuffled_typewell["top32_recall"]
    ),
    "no_gr_top32_drop": float(selected["top32_recall"] - no_gr["top32_recall"]),
}
checks = {
    "top32_recall_gain_ge_0p05": gains["top32_recall_gain"] >= 0.05,
    "median_rank_reduction_ge_8": gains["median_rank_reduction"] >= 8.0,
    "deep_top32_recall_gain_ge_0p05": gains["deep_top32_recall_gain"] >= 0.05,
    "alias24_not_worse": gains["alias24_win_gain"] >= -0.01,
    "shuffled_top32_drop_ge_0p05": gains["shuffled_top32_drop"] >= 0.05,
    "no_gr_top32_drop_ge_0p05": gains["no_gr_top32_drop"] >= 0.05,
}
strict_pass = bool(all(checks.values()))

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_nested_windowed_misfit_observability_gate",
    "sample_unit": "257-state grid stations nested within complete physical wells",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "validation_fold": VALID_FOLD,
        "selection_wells": len(selection_wells),
        "confirmation_wells": len(confirmation_wells),
        "fold1_touched": False,
        "fold_source": str(fold_path),
        "record_source": str(V["RECORD_DIR"]),
        "costmap_source": str(PACK_DIR),
        "selected_profile": selected_name,
    },
    "families": [dict(config) for config in CONFIGS],
    "selection": selection_rows,
    "confirmation": confirmation_rows,
    "baseline": baseline,
    "selected": selected,
    "controls": {"no_gr": no_gr, "shuffled_typewell": shuffled_typewell},
    "gains": gains,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> freeze the selected emission and test it inside the existing sequence decoder"
        if strict_pass
        else "FAIL -> these windowed misfit families do not repair state ranking; close emission-only work"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nMISFIT SELECTION TABLE")
print(selection_frame.to_string(index=False))
print("\nMISFIT CONFIRMATION TABLE")
print(confirmation_frame.to_string(index=False))
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
