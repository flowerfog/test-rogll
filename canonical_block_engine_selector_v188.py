"""V188: fivefold legal blockwise selector for the V187 endpoint inventory.

Each sample is one candidate endpoint on one 128-ft target-well block.  The
model predicts block RMSE from legal candidate-conditioned GR/typewell match,
geometry, endpoint disagreement and trajectory-shape features.  Held-out
wells never contribute labels or hyperparameter choices.  A fixed Viterbi
decoder applies the V187 continuity cost.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


STARTED = time.time()
SEED = 20260803
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
DATA_ROOT = next((p for p in (ROOT / "rogii-wellbore-geology-prediction", ROOT) if (p / "train").is_dir()), ROOT)
OOF = ROOT / "oof_exports"
OUT = OOF / "canonical_block_engine_selector_v188"
OUT.mkdir(parents=True, exist_ok=True)

BASE_PATH = OOF / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
BASE_COLUMN = "metadata_analog_w0p05"
BASE_SHA = "43f23cb1f53212e8da77b38e873b24ae53ddd092a8f2fa16fa761e7a1536e8c6"
BLOCK_FT = 128.0
FREEZE_FT = 1000.0
CONTINUITY_ROWS = 32.0
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773

SOURCES = (
    ("v168", BASE_PATH, (BASE_COLUMN,)),
    ("v169", OOF / "historical_analog_ranked_consensus_v169" / "fullrow_oof.parquet", ("nested_consensus",)),
    ("v144", OOF / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet",
     ("consensus_center", "nested_sequence_coefficient", "exact_v76", "coarse_crossfit")),
    ("v66", OOF / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
     ("incumbent_6p377_replay", "full_cp8_ordered")),
    ("v96", OOF / "rawdata_lattice_v76_full773_training_v96" / "fivefold_coarse_oof.parquet", ("coarse_crossfit",)),
    ("psg", OOF / "psg_foldclean_fivefold_oof_v0" / "psg_oof.parquet", ("psg_tvt",)),
    ("track8", OOF / "track8_sp45_foldclean_fivefold_v2" / "track8_oof.parquet", ("track8", "common_only")),
    ("legacy", OOF / "frontier_6p39_existing_models_oof_v0" / "frontier_6p39_core_oof.parquet",
     ("sp45", "strong_bias", "strong", "cp12", "fleongg")),
    ("neighbor", OOF / "frontier_6p39_neighbor_parent_audit_v2" / "neighbor_parent_oof.parquet", ("fle_neighbor", "fle_parent")),
)


def _aligned(path: Path, columns: tuple[str, ...], canonical: pd.DataFrame) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    schema = set(pq.ParquetFile(path).schema.names)
    available = [column for column in columns if column in schema]
    if not available or not {"well", "row_idx"}.issubset(schema):
        return {}
    local = pd.read_parquet(path, columns=["well", "row_idx", *available])
    local["well"] = local["well"].astype(str)
    local["row_idx"] = local["row_idx"].astype(np.int64)
    local = local.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    if len(local) != len(canonical):
        return {}
    if not np.array_equal(local["well"].to_numpy(), canonical["well"].to_numpy()):
        return {}
    if not np.array_equal(local["row_idx"].to_numpy(), canonical["row_idx"].to_numpy()):
        return {}
    return {column: local[column].to_numpy(np.float32) for column in available if np.isfinite(local[column]).all()}


def _deduplicate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[valid], float), np.asarray(y[valid], float)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    unique, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=y)
    counts = np.bincount(inverse)
    return unique, sums / np.maximum(counts, 1)


def _affine(reference: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(reference) & np.isfinite(observed)
    x, y = reference[valid], observed[valid]
    if len(x) < 32 or np.std(x) < 1e-6:
        return 1.0, float(np.nanmedian(y) - np.nanmedian(x))
    keep = np.ones(len(x), bool)
    slope, intercept = 1.0, 0.0
    for _ in range(4):
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y - (slope * x + intercept)
        scale = 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep]))) + 1e-6
        keep = np.abs(residual) <= 3.5 * scale
        if keep.sum() < 32:
            break
    return float(np.clip(slope, 0.2, 5.0)), float(intercept)


def _fill(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    if valid.all():
        return values
    if not valid.any():
        return np.zeros_like(values)
    index = np.arange(len(values))
    return np.interp(index, index[valid], values[valid])


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 8:
        return 0.0
    x = left[valid] - np.mean(left[valid])
    y = right[valid] - np.mean(right[valid])
    scale = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / scale) if scale > 1e-9 else 0.0


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(values) < 3:
        return values
    width = min(width, len(values) if len(values) % 2 else len(values) - 1)
    if width < 3:
        return values
    return np.convolve(np.pad(values, (width // 2,), mode="edge"), np.ones(width) / width, mode="valid")


def _match_features(observed: np.ndarray, expected: np.ndarray) -> list[float]:
    valid = np.isfinite(observed) & np.isfinite(expected)
    coverage = float(np.mean(valid))
    if valid.sum() < 8:
        return [coverage] + [0.0] * 21
    observed = _fill(observed)
    expected = _fill(expected)
    output = [coverage]
    for width in (1, 7, 21):
        left, right = _smooth(observed, width), _smooth(expected, width)
        residual = np.clip(left - right, -8.0, 8.0)
        lag_corr = []
        for lag in (-8, -4, 0, 4, 8):
            if lag < 0:
                lag_corr.append(_corr(left[-lag:], right[:lag]))
            elif lag > 0:
                lag_corr.append(_corr(left[:-lag], right[lag:]))
            else:
                lag_corr.append(_corr(left, right))
        output.extend([
            _corr(left, right), _corr(np.gradient(left), np.gradient(right)),
            float(np.mean(residual)), float(np.mean(np.abs(residual))),
            float(np.sqrt(np.mean(residual * residual))),
            float(max(lag_corr)), float((-8, -4, 0, 4, 8)[int(np.argmax(lag_corr))] / 8.0),
        ])
    return output


def _metrics(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    error = np.asarray(prediction, float) - frame["tvt_true"].to_numpy(float)
    distance = frame["md_since"].to_numpy(float)
    codes, names = pd.factorize(frame["well"], sort=False)
    counts = np.bincount(codes)
    sse = np.bincount(codes, weights=error * error, minlength=len(names))
    wrmse = np.sqrt(sse / counts)
    far = distance > 2000.0
    return {
        "endpoint": endpoint, "rows": len(frame), "wells": len(names),
        "exact_sse": float(np.dot(error, error)), "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))), "median_well_rmse": float(np.median(wrmse)),
        "p95_well_rmse": float(np.quantile(wrmse, .95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


base = pd.read_parquet(BASE_PATH)
base["well"] = base["well"].astype(str)
base["row_idx"] = base["row_idx"].astype(np.int64)
base = base.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if len(base) != EXPECTED_ROWS or base["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError("Invalid V168 coverage")
if hashlib.sha256(base[BASE_COLUMN].to_numpy(np.float32).tobytes()).hexdigest() != BASE_SHA:
    raise RuntimeError("V168 SHA mismatch")

arrays: dict[str, np.ndarray] = {}
seen: set[str] = set()
for family, path, columns in SOURCES:
    for column, values in _aligned(path, columns, base).items():
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        if digest not in seen:
            arrays[f"{family}/{column}"] = values
            seen.add(digest)
names = list(arrays)
if "v168/metadata_analog_w0p05" not in names or len(names) < 8:
    raise RuntimeError(f"Incomplete V187 inventory: {names}")
base_index = names.index("v168/metadata_analog_w0p05")
paths = np.stack([arrays[name] for name in names]).astype(np.float64)
truth = base["tvt_true"].to_numpy(np.float64)
k = len(names)

feature_names = ["candidate_id", "block_progress", "block_rows", "block_z_slope", "block_z_range",
                 "candidate_mean", "candidate_slope", "candidate_curve_rms", "move_mean", "move_rms",
                 "move_slope", "ensemble_z_mean", "ensemble_z_rms", "ensemble_spread"]
for width in (1, 7, 21):
    feature_names += [f"match_w{width}_{name}" for name in ("corr", "gradient_corr", "residual_mean", "residual_absmean", "residual_rms", "best_lag_corr", "best_lag")]
feature_names.insert(14, "match_coverage")

features, labels_rmse, labels_sse, block_rows = [], [], [], []
well_slices: dict[str, list[tuple[int, int, np.ndarray]]] = {}
group_id = 0
for number, (well, positions) in enumerate(base.groupby("well", sort=False).indices.items(), 1):
    positions = np.asarray(positions, np.int64)
    local = base.iloc[positions]
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    row_idx = local["row_idx"].to_numpy(np.int64)
    md_all = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
    z_all = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
    gr_all = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
    visible = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    tw_tvt, tw_gr = _deduplicate(
        pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(float),
        pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(float),
    )
    known = np.isfinite(visible) & np.isfinite(gr_all)
    reference_known = np.interp(visible[known], tw_tvt, tw_gr, left=np.nan, right=np.nan)
    gain, intercept = _affine(reference_known, gr_all[known])
    calibrated_gr = gain * tw_gr + intercept
    centre = float(np.nanmedian(gr_all[known]))
    scale = float(1.4826 * np.nanmedian(np.abs(gr_all[known] - centre)) + 1e-6)
    observed = (gr_all[row_idx] - centre) / scale
    z = z_all[row_idx]
    md = md_all[row_idx]
    distance = local["md_since"].to_numpy(float)
    active = distance > FREEZE_FT
    labels = np.floor((distance[active] - FREEZE_FT) / BLOCK_FT).astype(np.int32)
    active_positions = np.flatnonzero(active)
    slices = []
    for block_label in np.unique(labels):
        local_pos = active_positions[labels == block_label]
        global_pos = positions[local_pos]
        start_group = group_id
        block_path = paths[:, global_pos]
        median_path = np.median(block_path, axis=0)
        spread = np.std(block_path, axis=0)
        block_md = md[local_pos]
        md_span = max(float(block_md[-1] - block_md[0]), 1e-6)
        block_z = z[local_pos]
        z_slope = float((block_z[-1] - block_z[0]) / md_span)
        for candidate in range(k):
            prediction = block_path[candidate]
            expected = np.interp(prediction, tw_tvt, calibrated_gr, left=np.nan, right=np.nan)
            expected = (expected - centre) / scale
            move = prediction - paths[base_index, global_pos]
            curve = np.diff(prediction, 2)
            common = [
                float(candidate), float(np.mean(distance[local_pos]) / 1000.0), float(len(local_pos) / 128.0),
                z_slope, float(np.ptp(block_z)), float(np.mean(prediction)),
                float((prediction[-1] - prediction[0]) / md_span),
                float(np.sqrt(np.mean(curve * curve))) if len(curve) else 0.0,
                float(np.mean(move)), float(np.sqrt(np.mean(move * move))),
                float((move[-1] - move[0]) / md_span),
                float(np.mean(prediction - median_path)),
                float(np.sqrt(np.mean((prediction - median_path) ** 2))), float(np.mean(spread)),
            ]
            features.append(common + _match_features(observed[local_pos], expected))
            error = prediction - truth[global_pos]
            sse = float(np.dot(error, error))
            labels_sse.append(sse)
            labels_rmse.append(float(np.sqrt(sse / len(local_pos))))
            block_rows.append({"group": group_id, "well": well, "fold": int(local["fold"].iloc[0]),
                               "candidate": candidate, "global_start": int(global_pos[0]), "rows": len(global_pos)})
        slices.append((start_group, group_id, global_pos))
        group_id += 1
    well_slices[str(well)] = slices
    if number % 50 == 0 or number == EXPECTED_WELLS:
        print(f"V188 features {number}/{EXPECTED_WELLS} blocks={group_id}", flush=True)

x = pd.DataFrame(np.asarray(features, np.float32), columns=feature_names)
x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
panel = pd.DataFrame(block_rows)
y = np.asarray(labels_rmse, np.float32)
sse = np.asarray(labels_sse, np.float64)
folds = panel["fold"].to_numpy(np.int8)
weights = panel["rows"].to_numpy(np.float32)
match_columns = [column for column in feature_names if column.startswith("match_")]
metadata_columns = [column for column in feature_names if column not in match_columns]

score = {name: np.full(len(panel), np.nan, np.float32) for name in ("full", "metadata", "rolled_match")}
reports = []
for held in range(1, 6):
    train = folds != held
    valid = folds == held
    models = {}
    for model_name, columns in (("full", feature_names), ("metadata", metadata_columns)):
        model = lgb.LGBMRegressor(
            objective="huber", alpha=.85, n_estimators=550, learning_rate=.025,
            num_leaves=31, min_child_samples=120, max_depth=-1,
            subsample=.8, subsample_freq=1, colsample_bytree=.75,
            reg_alpha=1.0, reg_lambda=20.0, random_state=SEED + held,
            n_jobs=-1, verbosity=-1, force_col_wise=True,
        )
        model.fit(x.loc[train, columns], y[train], sample_weight=weights[train])
        models[model_name] = model
        score[model_name][valid] = model.predict(x.loc[valid, columns]).astype(np.float32)
    rolled = x.loc[valid, feature_names].copy()
    for column in match_columns:
        values = rolled[column].to_numpy().reshape(-1, k)
        rolled[column] = np.roll(values, 1, axis=1).reshape(-1)
    score["rolled_match"][valid] = models["full"].predict(rolled).astype(np.float32)
    reports.append({"fold": held, "train_blocks": int(panel.loc[train, "group"].nunique()),
                    "valid_blocks": int(panel.loc[valid, "group"].nunique()), "features": len(feature_names)})
    print("V188 model", reports[-1], flush=True)

if any(not np.isfinite(values).all() for values in score.values()):
    raise RuntimeError("Incomplete V188 crossfit scores")


well_fold = base.groupby("well", sort=False)["fold"].first().astype(int).to_dict()
constant_choice = {}
candidate_index = panel["candidate"].to_numpy(np.int16)
for held in range(1, 6):
    train = folds != held
    totals = np.array([sse[train & (candidate_index == candidate)].sum() for candidate in range(k)])
    constant_choice[held] = int(np.argmin(totals))


def _decode(scores: np.ndarray, constant_by_fold: bool = False) -> tuple[np.ndarray, int]:
    output = paths[base_index].copy()
    switches = 0
    for well, slices in well_slices.items():
        if not slices:
            continue
        local_fold = well_fold[well]
        if constant_by_fold:
            chosen = np.full(len(slices), constant_choice[local_fold], np.int16)
        else:
            unary = []
            for group, _, positions in slices:
                rows = len(positions)
                indices = slice(group * k, (group + 1) * k)
                unary.append(np.square(np.clip(scores[indices], 0.0, 80.0)) * rows)
            dp = unary[0].astype(np.float64)
            back = np.zeros((len(slices), k), np.int16)
            for block in range(1, len(slices)):
                boundary = slices[block][2][0]
                values = paths[:, boundary]
                transition = CONTINUITY_ROWS * np.square(values[None, :] - values[:, None])
                total = dp[:, None] + transition
                back[block] = np.argmin(total, axis=0)
                dp = unary[block] + np.min(total, axis=0)
            chosen = np.empty(len(slices), np.int16)
            chosen[-1] = np.argmin(dp)
            for block in range(len(slices) - 1, 0, -1):
                chosen[block - 1] = back[block, chosen[block]]
        switches += int(np.count_nonzero(chosen[1:] != chosen[:-1]))
        for choice, (_, _, positions) in zip(chosen, slices):
            output[positions] = paths[int(choice), positions]
    return output, switches


predictions = {"v168": paths[base_index]}
switch_counts = {"v168": 0}
for name in score:
    predictions[name], switch_counts[name] = _decode(score[name])
predictions["constant_best"], switch_counts["constant_best"] = _decode(score["full"], constant_by_fold=True)

summary_rows, fold_rows = [], []
for name, prediction in predictions.items():
    metric = _metrics(base, prediction, name)
    metric["switches"] = switch_counts[name]
    summary_rows.append(metric)
    for fold in range(1, 6):
        mask = base["fold"].to_numpy(int) == fold
        local = _metrics(base.loc[mask].reset_index(drop=True), prediction[mask], name)
        fold_rows.append({"fold": fold, **local})
summary = pd.DataFrame(summary_rows).sort_values("rmse", ignore_index=True)
fold_summary = pd.DataFrame(fold_rows)
base_rmse = float(summary.loc[summary["endpoint"].eq("v168"), "rmse"].iloc[0])
full_rmse = float(summary.loc[summary["endpoint"].eq("full"), "rmse"].iloc[0])
metadata_rmse = float(summary.loc[summary["endpoint"].eq("metadata"), "rmse"].iloc[0])
rolled_rmse = float(summary.loc[summary["endpoint"].eq("rolled_match"), "rmse"].iloc[0])
full_folds = fold_summary[fold_summary["endpoint"].eq("full")].set_index("fold")["rmse"]
base_folds = fold_summary[fold_summary["endpoint"].eq("v168")].set_index("fold")["rmse"]
checks = {
    "full_rmse_lt_6": full_rmse < 6.0,
    "gain_ge_1": base_rmse - full_rmse >= 1.0,
    "positive_folds_5of5": bool((base_folds - full_folds > 0).all()),
    "maximum_fold_rmse_lt_6p5": float(full_folds.max()) < 6.5,
    "full_beats_metadata_ge_0p10": metadata_rmse - full_rmse >= .10,
    "full_beats_rolled_match_ge_0p10": rolled_rmse - full_rmse >= .10,
    "far_rmse_lt_7": float(summary.loc[summary["endpoint"].eq("full"), "far_gt2000_rmse"].iloc[0]) < 7.0,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "canonical_block_engine_selector_v188",
    "stage": "fivefold_candidate_conditioned_128ft_block_selector",
    "sample_unit": "candidate endpoint on one target-well block",
    "historical_prediction_inputs": names,
    "coverage": {"rows": len(base), "wells": base["well"].nunique(), "blocks": int(panel["group"].nunique()), "candidate_rows": len(panel)},
    "recipe": {"block_ft": BLOCK_FT, "freeze_ft": FREEZE_FT, "continuity_rows": CONTINUITY_ROWS},
    "features": {"total": len(feature_names), "candidate_conditioned_gr": len(match_columns)},
    "model_reports": reports, "summary": summary.to_dict("records"),
    "values": {"gain": base_rmse - full_rmse, "metadata_increment": metadata_rmse - full_rmse,
               "rolled_match_increment": rolled_rmse - full_rmse},
    "checks": checks, "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> freeze V188 and build deployment inference" if strict_pass else "FAIL -> local oracle is not observable from legal block evidence",
}
summary.to_csv(OUT / "summary.csv", index=False)
fold_summary.to_csv(OUT / "folds.csv", index=False)
pd.DataFrame(reports).to_csv(OUT / "model_reports.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
print("\nV188 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV188 FOLDS")
print(fold_summary.to_string(index=False), flush=True)
print("\nV188 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=str), flush=True)
print("saved:", OUT, flush=True)
