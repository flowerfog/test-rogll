"""Fold-clean pseudo-typewell SGM reachability audit.

Run as one AutoDL notebook cell after uploading this file and the V34 source to
``/root/autodl-tmp``.  This is an oracle gate, not a deployable model:

* The target path is the exact V33 lineage used by V34/V35.
* Rows within 1000 ft of the anchor remain unchanged.
* Candidate pseudo-typewells are full GR(TVT) curves from outer-training wells.
  A target never consumes a reference from its own OOF fold.
* Geometry makes a shortlist; visible-prefix GR ranks that shortlist legally.
* The frozen V34 SGM recipe produces one complete-well path per reference.
* Truth is used only after all paths are frozen, to measure the candidate-family
  oracle and decide whether a later legal selector is worth building.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import time
import types

import numpy as np
import pandas as pd


EXPERIMENT = "frontier_6p377_pseudotypewell_sgm_oracle_v38"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V34_SOURCE = ROOT / "frontier_6p377_tail_sgm_residual_v34_cell.py"
PARENT_PATH = (
    OOF_ROOT
    / "frontier_6p377_cp8_neighbor_dose_replay_v33"
    / "cp8_neighbor_replay_oof.parquet"
)
V34_PATH = (
    OOF_ROOT
    / "frontier_6p377_tail_sgm_residual_v34"
    / "tail_sgm_oof.parquet"
)
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARENT_COLUMN = "current_dose_0p20"
REFERENCE_NEIGHBOURS = 24
PREFIX_CANDIDATES = 3
REFERENCE_BIN_FT = 0.5
MIN_REFERENCE_ROWS = 96
MIN_PREFIX_ROWS = 24
MIN_PREFIX_COVERAGE = 0.45
FROZEN_DOSE = 0.10
SEED = 20260727
started = time.time()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _load_v34_definitions():
    if not V34_SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing {V34_SOURCE}. Upload frontier_6p377_tail_sgm_residual_v34_cell.py first."
        )
    source = V34_SOURCE.read_text(encoding="utf-8")
    marker = "\nif not PARENT_PATH.is_file():"
    if marker not in source:
        raise RuntimeError("Cannot locate V34 definition marker")
    name = "frontier_v34_definitions_for_v38"
    module = types.ModuleType(name)
    module.__file__ = str(V34_SOURCE)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0], str(V34_SOURCE), "exec"), module.__dict__)
    return module


V = _load_v34_definitions()
DATA_ROOT = V.DATA_ROOT
OFFSETS_FT = V.OFFSETS_FT
ZERO_STATE = V.ZERO_STATE
START_FT = V.START_FT
NODE_SPACING_FT = V.NODE_SPACING_FT
WINDOW_HALF_FT = V.WINDOW_HALF_FT
MAX_WINDOW_ROWS = V.MAX_WINDOW_ROWS
MIN_WINDOW_ROWS = V.MIN_WINDOW_ROWS
WellVolume = V.WellVolume

FROZEN_RECIPE = {
    "id": "e200_s100_j3_p002",
    "temperature": 2.0,
    "smooth": 1.0,
    "jump": 3.0,
    "centre": 0.02,
}


def _reference_curve(tvt, gr):
    table = pd.DataFrame({"tvt": tvt, "gr": gr}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(table) < MIN_REFERENCE_ROWS:
        raise RuntimeError("too few finite reference GR rows")
    table["axis"] = np.round(table["tvt"] / REFERENCE_BIN_FT) * REFERENCE_BIN_FT
    table = table.groupby("axis", as_index=False)["gr"].median().sort_values("axis")
    if len(table) < MIN_REFERENCE_ROWS:
        raise RuntimeError("too few unique reference TVT bins")
    # A short robust smooth suppresses horizontal-log spikes without erasing beds.
    table["gr"] = table["gr"].rolling(3, center=True, min_periods=1).median()
    return table["axis"].to_numpy(np.float64), table["gr"].to_numpy(np.float64)


def _prefix_fit(horizontal, axis, values):
    observed = horizontal["GR"].to_numpy(np.float64)
    known_tvt = horizontal["TVT_input"].to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(observed) & np.isfinite(known_tvt))
    if len(known) > 768:
        known = known[np.linspace(0, len(known) - 1, 768).round().astype(int)]
    if len(known) < MIN_PREFIX_ROWS:
        return None
    expected = np.interp(known_tvt[known], axis, values, left=np.nan, right=np.nan)
    valid = np.isfinite(expected)
    coverage = float(np.mean(valid))
    if valid.sum() < MIN_PREFIX_ROWS or coverage < MIN_PREFIX_COVERAGE:
        return None
    expected = expected[valid]
    target = observed[known][valid]
    gain, bias, gamma = V._robust_affine(expected, target)
    calibrated = gain * expected + bias
    residual = (target - calibrated) / max(gamma, 1e-6)
    raw = float(np.mean(np.log1p(np.minimum(residual**2, 1e4))))
    x = (calibrated - np.median(calibrated)) / V._robust_scale(calibrated, 3.0)
    y = (target - np.median(target)) / V._robust_scale(target, 3.0)
    shape = float(np.mean(np.log1p(np.minimum((x - y) ** 2, 1e4))))
    score = 0.65 * raw + 0.35 * shape + 4.0 * max(0.85 - coverage, 0.0)
    return {
        "score": float(score),
        "coverage": coverage,
        "gain": float(gain),
        "bias": float(bias),
        "gamma": float(gamma),
    }


def _build_volume(well, indices, local, baseline, horizontal, axis, raw_values, fit):
    row_idx = local["row_idx"].to_numpy(np.int64)
    md_all = horizontal["MD"].to_numpy(np.float64)
    gr_all = horizontal["GR"].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    raw_distance = md_all[row_idx] - md_all[row_idx[0]] + float(distance[0])
    gap = np.abs(raw_distance - distance)
    if not np.isfinite(gap).all() or float(np.max(gap)) > 0.1:
        raise RuntimeError(f"{well}: OOF/raw MD mismatch {np.nanmax(gap)}")

    values = fit["gain"] * raw_values + fit["bias"]
    gamma = fit["gamma"]
    eval_gr = gr_all[row_idx]
    active_rows = np.flatnonzero(distance >= START_FT)
    if not len(active_rows):
        raise RuntimeError(f"{well}: no rows beyond {START_FT:g} ft")
    targets = np.arange(START_FT, float(distance[-1]) + 0.5, NODE_SPACING_FT)
    nodes = np.searchsorted(distance, targets, side="left")
    nodes = np.clip(nodes, active_rows[0], len(distance) - 1)
    nodes = np.unique(np.concatenate(([active_rows[0]], nodes, [len(distance) - 1]))).astype(np.int64)

    costs = np.zeros((len(nodes), len(OFFSETS_FT)), np.float64)
    usable = np.zeros(len(nodes), bool)
    for node_number, node in enumerate(nodes):
        if node_number == 0:
            costs[node_number] = 1e4
            costs[node_number, ZERO_STATE] = 0.0
            usable[node_number] = True
            continue
        rows = np.flatnonzero(np.abs(distance - distance[node]) <= WINDOW_HALF_FT)
        rows = rows[np.isfinite(eval_gr[rows])]
        if len(rows) > MAX_WINDOW_ROWS:
            rows = rows[np.linspace(0, len(rows) - 1, MAX_WINDOW_ROWS).round().astype(int)]
        if len(rows) < MIN_WINDOW_ROWS:
            continue
        candidate_tvt = baseline[rows, None] + OFFSETS_FT[None, :]
        expected = np.interp(candidate_tvt.ravel(), axis, values, left=np.nan, right=np.nan).reshape(candidate_tvt.shape)
        observed = eval_gr[rows, None]
        valid = np.isfinite(expected)
        coverage = valid.mean(axis=0)
        residual = (observed - expected) / max(gamma, 1e-6)
        raw_cost = np.nanmean(
            np.where(valid, np.log1p(np.minimum(residual**2, 1e4)), np.nan), axis=0
        )
        obs_centre = np.nanmedian(observed)
        obs_scale = V._robust_scale(observed[:, 0] - obs_centre, 3.0)
        exp_centre = np.nanmedian(expected, axis=0, keepdims=True)
        exp_scale = np.maximum(
            1.4826 * np.nanmedian(np.abs(expected - exp_centre), axis=0), 3.0
        )
        shape_residual = (observed - obs_centre) / obs_scale - (expected - exp_centre) / exp_scale
        shape_cost = np.nanmean(
            np.where(valid, np.log1p(np.minimum(shape_residual**2, 1e4)), np.nan), axis=0
        )
        combined = 0.65 * raw_cost + 0.35 * shape_cost + 4.0 * np.maximum(0.85 - coverage, 0.0)
        finite = np.isfinite(combined)
        if not np.any(finite):
            continue
        combined[~finite] = float(np.max(combined[finite]) + 10.0)
        costs[node_number] = combined - np.min(combined)
        usable[node_number] = True
    return WellVolume(
        well=str(well),
        indices=np.asarray(indices, np.int64),
        distance=distance,
        nodes=nodes,
        costs=costs,
        usable=usable,
        prefix_gain=fit["gain"],
        prefix_bias=fit["bias"],
        gamma=gamma,
    )


def _metrics(local, prediction, endpoint):
    truth = local["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = local["md_since"].to_numpy(np.float64)
    well_mse = pd.Series(error**2).groupby(local["well"].to_numpy(), sort=False).mean()
    out = {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(well_mse))),
        "p95_well_rmse": float(np.quantile(np.sqrt(well_mse), 0.95)),
        "max_well_rmse": float(np.max(np.sqrt(well_mse))),
    }
    for name, mask in {
        "near_le1000": distance <= 1000.0,
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else np.nan
    return out


for required_path in (PARENT_PATH, V34_PATH, FOLD_PATH):
    if not required_path.is_file():
        raise FileNotFoundError(required_path)

parent = pd.read_parquet(PARENT_PATH).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
required = {"id", "well", "row_idx", "fold", "tvt_true", "md_since", PARENT_COLUMN}
missing = sorted(required.difference(parent.columns))
if missing:
    raise RuntimeError(f"V33 parent missing columns: {missing}")
if sorted(parent["fold"].unique().tolist()) != [1, 2, 3]:
    raise RuntimeError("V38 expects the exact three-fold V33 lineage")

v34 = pd.read_parquet(V34_PATH)
if "v34_selected" not in v34.columns:
    raise RuntimeError("V34 artifact lacks v34_selected")
aligned = parent[["id"]].merge(v34[["id", "v34_selected"]], on="id", how="left", validate="one_to_one")
if aligned["v34_selected"].isna().mean() > 0.005:
    raise RuntimeError("V34/V33 row alignment failed")
baseline = parent[PARENT_COLUMN].to_numpy(np.float64)
own_prediction = aligned["v34_selected"].fillna(parent[PARENT_COLUMN]).to_numpy(np.float64)

fold_table = pd.read_csv(FOLD_PATH, usecols=lambda c: c in {"well", "fold"})
fold_by_well = fold_table.groupby("well")["fold"].first().astype(int).to_dict()
all_wells = sorted(fold_by_well)
target_wells = parent["well"].drop_duplicates().astype(str).tolist()
if len(target_wells) != 464:
    raise RuntimeError(f"Expected 464 V33 wells, got {len(target_wells)}")

# Build the fold-clean reference bank once.  Geometry is sampled because only
# anchor-to-trajectory distance is needed for shortlisting.
references = {}
reference_failures = []
for number, well in enumerate(all_wells, 1):
    try:
        path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        horizontal = pd.read_csv(
            path, usecols=lambda c: c in {"X", "Y", "TVT", "TVT_input", "GR"}
        ).apply(pd.to_numeric, errors="coerce")
        xy = horizontal[["X", "Y"]].to_numpy(np.float64)
        finite_xy = np.flatnonzero(np.isfinite(xy).all(axis=1))
        visible = np.flatnonzero(horizontal["TVT_input"].notna().to_numpy() & np.isfinite(xy).all(axis=1))
        if not len(finite_xy) or not len(visible):
            raise RuntimeError("missing geometry/anchor")
        anchor = int(visible[-1])
        direction = xy[finite_xy[-1]] - xy[anchor]
        norm = float(np.linalg.norm(direction))
        if norm < 1.0:
            raise RuntimeError("degenerate direction")
        direction /= norm
        sample = finite_xy[np.linspace(0, len(finite_xy) - 1, min(96, len(finite_xy))).round().astype(int)]
        axis, values = _reference_curve(
            horizontal["TVT"].to_numpy(np.float64), horizontal["GR"].to_numpy(np.float64)
        )
        references[well] = {
            "fold": int(fold_by_well[well]),
            "anchor_xy": xy[anchor],
            "xy": xy[sample],
            "direction": direction,
            "axis": axis,
            "values": values,
        }
    except Exception as exc:
        reference_failures.append({"well": well, "reason": repr(exc)})
    if number % 100 == 0 or number == len(all_wells):
        print(f"V38 reference bank {number}/{len(all_wells)} success={len(references)}", flush=True)
pd.DataFrame(reference_failures).to_csv(OUT_DIR / "reference_failures.csv", index=False)
if len(references) < 740:
    raise RuntimeError(f"Only {len(references)} usable pseudo-typewell references")

groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in parent.groupby("well", sort=False).indices.items()
}
legal_prediction = own_prediction.copy()
pseudo_top1_prediction = own_prediction.copy()
oracle_prediction = own_prediction.copy()
random_prediction = own_prediction.copy()
candidate_rows = []
well_rows = []
path_columns = {f"candidate_{rank}": own_prediction.copy() for rank in range(PREFIX_CANDIDATES + 1)}
rng = np.random.default_rng(SEED)
failures = []

for number, (well, indices) in enumerate(groups.items(), 1):
    try:
        local = parent.iloc[indices].reset_index(drop=True)
        target_fold = int(local["fold"].iloc[0])
        horizontal = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
            usecols=lambda c: c in {"MD", "X", "Y", "GR", "TVT_input"},
        ).apply(pd.to_numeric, errors="coerce")
        xy = horizontal[["X", "Y"]].to_numpy(np.float64)
        visible = np.flatnonzero(horizontal["TVT_input"].notna().to_numpy() & np.isfinite(xy).all(axis=1))
        if not len(visible):
            raise RuntimeError("target has no geometric anchor")
        anchor_xy = xy[int(visible[-1])]
        finite_xy = np.flatnonzero(np.isfinite(xy).all(axis=1))
        target_direction = xy[finite_xy[-1]] - anchor_xy
        target_direction /= max(float(np.linalg.norm(target_direction)), 1e-9)

        geometry = []
        for reference_well, reference in references.items():
            if reference_well == well or int(reference["fold"]) == target_fold:
                continue
            distance = float(np.min(np.linalg.norm(reference["xy"] - anchor_xy, axis=1)))
            direction_dot = abs(float(np.dot(target_direction, reference["direction"])))
            score = distance + 500.0 * max(0.75 - direction_dot, 0.0)
            geometry.append((score, distance, direction_dot, reference_well))
        geometry.sort()

        scored = []
        for geometry_score, distance, direction_dot, reference_well in geometry[:REFERENCE_NEIGHBOURS]:
            reference = references[reference_well]
            fit = _prefix_fit(horizontal, reference["axis"], reference["values"])
            if fit is None:
                continue
            scored.append((fit["score"], geometry_score, distance, direction_dot, reference_well, fit))
        scored.sort()
        selected = scored[:PREFIX_CANDIDATES]
        if not selected:
            raise RuntimeError("no prefix-compatible pseudo-typewell")

        candidates = [{
            "rank": 0,
            "reference": "assigned_typewell_v34",
            "prefix_score": np.nan,
            "prediction": own_prediction[indices].copy(),
            "distance": 0.0,
            "direction_dot": 1.0,
        }]
        for rank, (prefix_score, geometry_score, distance, direction_dot, reference_well, fit) in enumerate(selected, 1):
            reference = references[reference_well]
            volume = _build_volume(
                well, indices, local, baseline[indices], horizontal,
                reference["axis"], reference["values"], fit,
            )
            correction = V._row_correction(volume, V._decode(volume, FROZEN_RECIPE))
            prediction = baseline[indices] + FROZEN_DOSE * correction
            candidates.append({
                "rank": rank,
                "reference": reference_well,
                "prefix_score": prefix_score,
                "prediction": prediction,
                "distance": distance,
                "direction_dot": direction_dot,
                "prefix_coverage": fit["coverage"],
                "prefix_gain": fit["gain"],
                "prefix_bias": fit["bias"],
                "prefix_gamma": fit["gamma"],
            })

        truth = local["tvt_true"].to_numpy(np.float64)
        for candidate in candidates:
            candidate["rmse"] = float(np.sqrt(np.mean((candidate["prediction"] - truth) ** 2)))
            path_columns[f"candidate_{candidate['rank']}"][indices] = candidate["prediction"]
            candidate_rows.append({
                "well": well,
                "fold": target_fold,
                **{key: value for key, value in candidate.items() if key != "prediction"},
            })

        # Legal pseudo selection is rank 1.  A separate conservative selector
        # only replaces the assigned reference if its prefix score is finite and
        # materially better than the other pseudo candidates; no truth enters.
        pseudo_top1_prediction[indices] = candidates[1]["prediction"]
        pseudo_scores = np.asarray([item["prefix_score"] for item in candidates[1:]], np.float64)
        margin = float(np.partition(pseudo_scores, 1)[1] - pseudo_scores[0]) if len(pseudo_scores) > 1 else np.inf
        if margin >= 0.05:
            legal_prediction[indices] = candidates[1]["prediction"]
            legal_source = candidates[1]["reference"]
        else:
            legal_source = "assigned_typewell_v34"
        oracle = min(candidates, key=lambda item: item["rmse"])
        oracle_prediction[indices] = oracle["prediction"]
        random_choice = candidates[int(rng.integers(0, len(candidates)))]
        random_prediction[indices] = random_choice["prediction"]
        well_rows.append({
            "well": well,
            "fold": target_fold,
            "candidate_count": len(candidates),
            "legal_source": legal_source,
            "oracle_source": oracle["reference"],
            "oracle_rank": oracle["rank"],
            "own_rmse": candidates[0]["rmse"],
            "oracle_rmse": oracle["rmse"],
            "oracle_gain": candidates[0]["rmse"] - oracle["rmse"],
            "top1_prefix_score": candidates[1]["prefix_score"],
            "prefix_margin": margin,
        })
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 25 == 0 or number == len(groups):
        print(f"V38 targets {number}/{len(groups)} failures={len(failures)}", flush=True)

pd.DataFrame(candidate_rows).to_csv(OUT_DIR / "candidate_manifest.csv", index=False)
well_report = pd.DataFrame(well_rows)
well_report.to_csv(OUT_DIR / "per_well_oracle.csv", index=False)
pd.DataFrame(failures).to_csv(OUT_DIR / "target_failures.csv", index=False)
if len(failures) > 8:
    raise RuntimeError(f"Too many V38 failures: {failures[:10]}")

summary_rows = []
for endpoint, prediction in {
    "v33_base": baseline,
    "assigned_typewell_v34": own_prediction,
    "pseudo_prefix_top1": pseudo_top1_prediction,
    "legal_margin_gate": legal_prediction,
    "matched_random_menu": random_prediction,
    "candidate_family_oracle": oracle_prediction,
}.items():
    for fold in (0, 1, 2, 3):
        mask = np.ones(len(parent), bool) if fold == 0 else parent["fold"].to_numpy(np.int64) == fold
        summary_rows.append(_metrics(parent.loc[mask].reset_index(drop=True), prediction[mask], endpoint) | {"fold": fold})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)

output = parent[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["v33_base"] = baseline.astype(np.float32)
output["assigned_typewell_v34"] = own_prediction.astype(np.float32)
for name, values in path_columns.items():
    output[name] = values.astype(np.float32)
output["pseudo_prefix_top1"] = pseudo_top1_prediction.astype(np.float32)
output["legal_margin_gate"] = legal_prediction.astype(np.float32)
output["candidate_family_oracle"] = oracle_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "pseudotypewell_sgm_oof.parquet", index=False, compression="zstd")

def _summary(endpoint, fold=0):
    return summary[(summary["endpoint"] == endpoint) & (summary["fold"] == fold)].iloc[0].to_dict()

own = _summary("assigned_typewell_v34")
oracle = _summary("candidate_family_oracle")
legal = _summary("legal_margin_gate")
random_menu = _summary("matched_random_menu")
fold_gains = []
fold_far_gains = []
for fold in (1, 2, 3):
    own_fold = _summary("assigned_typewell_v34", fold)
    oracle_fold = _summary("candidate_family_oracle", fold)
    fold_gains.append(float(own_fold["rmse"] - oracle_fold["rmse"]))
    fold_far_gains.append(float(own_fold["far_gt2000_rmse"] - oracle_fold["far_gt2000_rmse"]))

pseudo_win_rate = float(np.mean(well_report["oracle_rank"] > 0)) if len(well_report) else 0.0
checks = {
    "oracle_gain_ge_0p15": float(own["rmse"] - oracle["rmse"]) >= 0.15,
    "oracle_far_gain_ge_0p20": float(own["far_gt2000_rmse"] - oracle["far_gt2000_rmse"]) >= 0.20,
    "minimum_fold_oracle_gain_ge_0p05": min(fold_gains) >= 0.05,
    "pseudo_reference_wins_ge_0p10": pseudo_win_rate >= 0.10,
    "oracle_beats_random_menu_ge_0p10": float(random_menu["rmse"] - oracle["rmse"]) >= 0.10,
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "foldclean_pseudotypewell_sgm_candidate_family_oracle_gate",
    "sample_unit": "one complete physical well",
    "truth_usage": "candidate-family reachability only; never used for legal reference ranking",
    "lineage": {
        "parent": str(PARENT_PATH),
        "parent_column": PARENT_COLUMN,
        "assigned_typewell_sgm": str(V34_PATH),
    },
    "protocol": {
        "target_folds": [1, 2, 3],
        "same_fold_reference_excluded": True,
        "geometry_shortlist": REFERENCE_NEIGHBOURS,
        "prefix_candidates": PREFIX_CANDIDATES,
        "frozen_recipe": FROZEN_RECIPE,
        "frozen_dose": FROZEN_DOSE,
        "near_le1000_policy": "bitwise frozen",
    },
    "coverage": {
        "target_wells": len(groups),
        "successful_wells": len(well_report),
        "reference_wells": len(references),
        "failures": len(failures),
    },
    "assigned_typewell": own,
    "legal_margin_gate": legal,
    "matched_random_menu": random_menu,
    "candidate_family_oracle": oracle,
    "oracle_gain": float(own["rmse"] - oracle["rmse"]),
    "oracle_far_gain": float(own["far_gt2000_rmse"] - oracle["far_gt2000_rmse"]),
    "fold_oracle_gains": fold_gains,
    "fold_far_gains": fold_far_gains,
    "pseudo_reference_win_rate": pseudo_win_rate,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> freeze candidate generator and train a nested complete-well reference selector"
        if all(checks.values())
        else "FAIL -> pseudo-typewell GR does not add enough reachable tail signal; close cross-well reference branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV38 SUMMARY")
print(summary[summary["fold"] == 0].to_string(index=False))
print("\nV38 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR)

