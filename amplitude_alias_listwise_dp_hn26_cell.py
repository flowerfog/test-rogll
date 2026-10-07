"""HN26: station-level real-alias listwise ranking plus continuity DP.

HN23 established that the legal amplitude top-32 menu has a strong oracle but
poor ordering.  This experiment learns only to reorder those real aliases.
Training uses folds 4/5, fold 3 freezes the DP strength and correction dose,
and fold 2 is the untouched confirmation.  Fold 1 is never loaded.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
SOURCE = ROOT / "multiscale_phase_event_observability_hn23_cell.py"
V66_PATH = OOF / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet"
EXPERIMENT = "amplitude_alias_listwise_dp_hn26"
OUT_DIR = OOF / EXPERIMENT
PANEL_DIR = OUT_DIR / "panels"
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, PANEL_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
ROWS_PER_WELL = 64
HALF_WINDOWS = (64, 128, 256)
MENU_SIZE = 32
MIN_WINDOW_COVERAGE = 0.35
SEED = 20260803
DP_LAMBDAS = (0.0, 0.01, 0.03, 0.10, 0.30, 1.0, 3.0)
DOSES = tuple(np.round(np.arange(0.0, 0.501, 0.05), 2))

for required in (SOURCE, V66_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)


def _load_hn23_definitions():
    source = SOURCE.read_text(encoding="utf-8")
    marker = "\nv66 = pd.read_parquet"
    if marker not in source:
        raise RuntimeError("Cannot locate HN23 execution marker")
    name = "hn23_definitions_for_hn26"
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0], str(SOURCE), "exec"), module.__dict__)
    return module.__dict__


H = _load_hn23_definitions()
OFFSETS = np.asarray(H["OFFSETS"], np.float64)
DATA_ROOT = Path(H["DATA_ROOT"])


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _prediction_column(frame):
    for column in ("incumbent_6p377_replay", "prediction", "tvt_pred", "pred"):
        if column in frame:
            return column
    raise RuntimeError(f"Cannot identify V66 prediction: {frame.columns.tolist()}")


def _rank_fraction(score, valid):
    score = np.asarray(score, np.float64).copy()
    score[~valid] = -np.inf
    order = np.argsort(-score, kind="stable")
    rank = np.empty(len(score), np.float64)
    rank[order] = np.arange(len(score), dtype=np.float64)
    return rank / max(valid.sum() - 1, 1)


def _standardise_score(score):
    score = np.asarray(score, np.float64)
    centre = np.median(score)
    scale = np.median(np.abs(score - centre)) * 1.4826
    if not np.isfinite(scale) or scale < 1e-5:
        scale = max(float(np.std(score)), 1e-5)
    return (score - centre) / scale


FEATURE_NAMES = [
    "offset", "abs_offset",
    "amp64", "amp128", "amp256",
    "amp64_z", "amp128_z", "amp256_z",
    "rank64", "rank128", "rank256",
    "gap64", "gap128", "gap256",
    "amp_cross_mean", "amp_cross_std", "amp_short_minus_long",
    "score_gradient", "score_curvature",
    "coverage64", "coverage128", "coverage256",
    "md_since_kft", "progress", "base_slope_128", "base_slope_512",
    "horizontal_gr_std", "horizontal_gr_diff_std",
]


def _slope(md, values, centre, half_window):
    left = max(0, int(centre) - int(half_window))
    right = min(len(md), int(centre) + int(half_window) + 1)
    x = np.asarray(md[left:right], np.float64)
    y = np.asarray(values[left:right], np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 8 or np.ptp(x[valid]) < 8.0:
        return 0.0
    return float(np.polyfit(x[valid] - x[valid][0], y[valid], 1)[0])


def _panel_path(well):
    return PANEL_DIR / f"{well}.npz"


def _build_panel(well, local):
    path = _panel_path(well)
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            if "feature_names" in data and data["feature_names"].astype(str).tolist() == FEATURE_NAMES:
                return {key: data[key] for key in data.files}

    base, horizontal_gr, truth, tw_tvt, tw_gr = H["_load_well"](well, local)
    horizontal = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
        usecols=["MD", "TVT_input"],
    )
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(known) < 32:
        raise RuntimeError(f"{well}: only {len(known)} visible rows")
    anchor = int(known[-1])
    eval_rows = local["row_idx"].to_numpy(np.int64)
    positions = H["_sample_positions"](eval_rows, ROWS_PER_WELL)
    max_md_since = max(float(md[eval_rows[-1]] - md[anchor]), 1.0)

    feature_blocks, label_blocks, offset_blocks = [], [], []
    station_rows, truth_offsets, amplitude_winners = [], [], []
    for centre in positions:
        truth_offset = float(truth[centre] - base[centre])
        if not np.isfinite(truth_offset) or truth_offset < OFFSETS[0] or truth_offset > OFFSETS[-1]:
            continue
        amp_scores, coverages = {}, {}
        valid_all = np.ones(len(OFFSETS), dtype=bool)
        horizontal_reference = None
        for half_window in HALF_WINDOWS:
            left = max(0, int(centre) - half_window)
            right = min(len(base), int(centre) + half_window + 1)
            if right - left < half_window:
                valid_all[:] = False
                break
            raw_h = horizontal_gr[left:right]
            coverage = float(np.isfinite(raw_h).mean())
            filled_h = H["_fill"](raw_h)
            if coverage < MIN_WINDOW_COVERAGE or filled_h is None:
                valid_all[:] = False
                break
            candidate_tvt = base[left:right][None, :] + OFFSETS[:, None]
            candidate_gr = np.interp(
                candidate_tvt.ravel(), tw_tvt, tw_gr, left=np.nan, right=np.nan
            ).reshape(len(OFFSETS), -1)
            valid = np.isfinite(candidate_gr).mean(axis=1) >= 0.95
            valid_all &= valid
            candidate_gr = np.where(
                np.isfinite(candidate_gr), candidate_gr,
                np.nanmedian(candidate_gr, axis=1, keepdims=True),
            )
            amp_scores[half_window] = H["_components"](filled_h, candidate_gr)["amp"]
            coverages[half_window] = coverage
            if half_window == 256:
                horizontal_reference = filled_h
        if valid_all.sum() < 64 or len(amp_scores) != len(HALF_WINDOWS):
            continue

        primary = np.asarray(amp_scores[256], np.float64).copy()
        primary[~valid_all] = -np.inf
        menu = np.argsort(-primary, kind="stable")[:MENU_SIZE]
        menu_offsets = OFFSETS[menu]
        ranks = {window: _rank_fraction(amp_scores[window], valid_all) for window in HALF_WINDOWS}
        zscores = {window: _standardise_score(amp_scores[window][valid_all]) for window in HALF_WINDOWS}
        zfull = {}
        for window in HALF_WINDOWS:
            values = np.full(len(OFFSETS), -8.0, np.float64)
            values[valid_all] = zscores[window]
            zfull[window] = values
        gradient = np.gradient(zfull[256])
        curvature = np.gradient(gradient)
        score_matrix = np.column_stack([amp_scores[w][menu] for w in HALF_WINDOWS])
        md_since = float(md[centre] - md[anchor])
        hstd = float(np.std(horizontal_reference))
        hdstd = float(np.std(np.diff(horizontal_reference)))
        rows = []
        for state in menu:
            cross = np.asarray([amp_scores[w][state] for w in HALF_WINDOWS], np.float64)
            row = [
                OFFSETS[state] / 128.0, abs(OFFSETS[state]) / 128.0,
                *cross.tolist(),
                zfull[64][state], zfull[128][state], zfull[256][state],
                ranks[64][state], ranks[128][state], ranks[256][state],
                np.nanmax(amp_scores[64][valid_all]) - amp_scores[64][state],
                np.nanmax(amp_scores[128][valid_all]) - amp_scores[128][state],
                np.nanmax(amp_scores[256][valid_all]) - amp_scores[256][state],
                float(np.mean(cross)), float(np.std(cross)), float(cross[0] - cross[-1]),
                float(gradient[state]), float(curvature[state]),
                coverages[64], coverages[128], coverages[256],
                md_since / 1000.0, md_since / max_md_since,
                _slope(md, base, centre, 128), _slope(md, base, centre, 512),
                hstd, hdstd,
            ]
            rows.append(row)
        matrix = np.asarray(rows, np.float32)
        if matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
            continue
        distance = np.abs(menu_offsets - truth_offset)
        distance_order = np.argsort(distance, kind="stable")
        distance_rank = np.empty(MENU_SIZE, np.int16)
        distance_rank[distance_order] = np.arange(MENU_SIZE, dtype=np.int16)
        # Every real-alias group remains informative, including stations where
        # the correct state falls outside top32: rank the least-wrong retained
        # candidate highest instead of producing an all-zero relevance group.
        labels = (15 - (distance_rank * 16 // MENU_SIZE)).astype(np.int16)
        feature_blocks.append(matrix)
        label_blocks.append(labels)
        offset_blocks.append(menu_offsets.astype(np.float32))
        station_rows.append(int(centre))
        truth_offsets.append(truth_offset)
        amplitude_winners.append(float(menu_offsets[0]))

    if len(feature_blocks) < 24:
        raise RuntimeError(f"{well}: only {len(feature_blocks)} usable stations")
    payload = {
        "features": np.stack(feature_blocks).astype(np.float32),
        "labels": np.stack(label_blocks).astype(np.int16),
        "offsets": np.stack(offset_blocks).astype(np.float32),
        "station_rows": np.asarray(station_rows, np.int32),
        "truth_offsets": np.asarray(truth_offsets, np.float32),
        "amplitude_winners": np.asarray(amplitude_winners, np.float32),
        "feature_names": np.asarray(FEATURE_NAMES),
    }
    np.savez_compressed(path, **payload)
    return payload


def _load_panel(well):
    with np.load(_panel_path(well), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _training_matrix(wells):
    x, y, groups = [], [], []
    for well in wells:
        panel = _load_panel(well)
        features = np.asarray(panel["features"], np.float32)
        labels = np.asarray(panel["labels"], np.int16)
        x.append(features.reshape(-1, features.shape[-1]))
        y.append(labels.reshape(-1))
        groups.extend([MENU_SIZE] * len(features))
    return np.concatenate(x), np.concatenate(y), np.asarray(groups, np.int32)


def _fit_ranker(train_wells, selection_wells):
    x_train, y_train, group_train = _training_matrix(train_wells)
    x_valid, y_valid, group_valid = _training_matrix(selection_wells)
    print("HN26 MATRICES", {
        "train": x_train.shape, "selection": x_valid.shape,
        "train_groups": len(group_train), "selection_groups": len(group_valid),
    }, flush=True)
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=list(range(16)),
        n_estimators=1200, learning_rate=0.025, num_leaves=31, max_depth=7,
        min_child_samples=120, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=4.0,
        random_state=SEED, n_jobs=max(1, min(16, os.cpu_count() or 1)), verbosity=-1,
    )
    model.fit(
        x_train, y_train, group=group_train,
        eval_set=[(x_valid, y_valid)], eval_group=[group_valid], eval_at=(1, 4, 8),
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(25)],
        feature_name=FEATURE_NAMES,
    )
    model.booster_.save_model(str(MODEL_DIR / "ranker_fold45.txt"))
    return model


def _dp_offsets(scores, offsets, station_md, transition_lambda):
    scores = np.asarray(scores, np.float64)
    offsets = np.asarray(offsets, np.float64)
    normalised = np.empty_like(scores)
    for index, row in enumerate(scores):
        normalised[index] = _standardise_score(row)
    if transition_lambda <= 0:
        return offsets[np.arange(len(offsets)), np.argmax(normalised, axis=1)]
    value = normalised[0] - transition_lambda * (offsets[0] / 16.0) ** 2
    back = np.zeros_like(scores, dtype=np.int16)
    for station in range(1, len(scores)):
        delta_md = max(float(station_md[station] - station_md[station - 1]), 1.0)
        allowance = 6.0 + 0.03 * delta_md
        transition = ((offsets[station][None, :] - offsets[station - 1][:, None]) / allowance) ** 2
        candidate = value[:, None] - transition_lambda * transition
        predecessor = np.argmax(candidate, axis=0)
        value = normalised[station] + candidate[predecessor, np.arange(MENU_SIZE)]
        back[station] = predecessor
    choice = np.empty(len(scores), np.int16)
    choice[-1] = int(np.argmax(value))
    for station in range(len(scores) - 1, 0, -1):
        choice[station - 1] = back[station, choice[station]]
    return offsets[np.arange(len(offsets)), choice]


def _metric(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    well_rmse = np.sqrt(
        pd.DataFrame({"well": frame["well"].astype(str), "e2": error * error})
        .groupby("well", sort=False)["e2"].mean().to_numpy(np.float64)
    )
    return {
        "endpoint": endpoint, "rows": len(frame), "wells": frame["well"].nunique(),
        "exact_sse": float(np.dot(error, error)), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))), "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
    }


_PREDICTION_CACHE = {}


def _prediction_payload(frame, well, model):
    if well in _PREDICTION_CACHE:
        return _PREDICTION_CACHE[well]
    local = frame.loc[frame["well"].eq(well)].sort_values("row_idx").copy()
    panel = _load_panel(well)
    features = np.asarray(panel["features"], np.float32)
    offsets = np.asarray(panel["offsets"], np.float64)
    station_rows = np.asarray(panel["station_rows"], np.int64)
    scores = model.predict(features.reshape(-1, features.shape[-1])).reshape(
        len(features), MENU_SIZE
    )
    horizontal = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
        usecols=["MD", "TVT_input"],
    )
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    anchor = int(np.flatnonzero(np.isfinite(tvt_input))[-1])
    payload = {
        "local": local, "panel": panel, "offsets": offsets,
        "station_rows": station_rows, "scores": scores, "md": md, "anchor": anchor,
    }
    _PREDICTION_CACHE[well] = payload
    return payload


def _predict_fold(frame, wells, model, transition_lambda, dose, shuffled=False):
    blocks, station_audits = [], []
    rng = np.random.default_rng(SEED + int(frame.loc[frame["well"].isin(wells), "fold"].iloc[0]) + int(shuffled))
    for well in wells:
        payload = _prediction_payload(frame, well, model)
        local = payload["local"].copy()
        panel = payload["panel"]
        offsets = payload["offsets"]
        station_rows = payload["station_rows"]
        scores = payload["scores"].copy()
        if shuffled:
            shuffled_scores = np.empty_like(scores)
            for station in range(len(scores)):
                shuffled_scores[station] = scores[station, rng.permutation(MENU_SIZE)]
            scores = shuffled_scores
        md = payload["md"]
        anchor = payload["anchor"]
        station_md = md[station_rows]
        selected = _dp_offsets(scores, offsets, station_md, transition_lambda)
        support_md = np.r_[md[anchor], station_md]
        support_offset = np.r_[0.0, selected]
        eval_md = md[local["row_idx"].to_numpy(np.int64)]
        correction = np.interp(eval_md, support_md, support_offset, left=0.0, right=support_offset[-1])
        prediction = local["base"].to_numpy(np.float64) + float(dose) * correction
        local["prediction"] = prediction.astype(np.float32)
        blocks.append(local)
        truth_offset = np.asarray(panel["truth_offsets"], np.float64)
        station_audits.append({
            "well": well,
            "station_rmse": float(np.sqrt(np.mean((selected - truth_offset) ** 2))),
            "station_within5": float(np.mean(np.abs(selected - truth_offset) <= 5.0)),
        })
    return pd.concat(blocks, ignore_index=True), pd.DataFrame(station_audits)


v66 = pd.read_parquet(V66_PATH, filters=[("fold", "in", [2, 3, 4, 5])])
prediction_column = _prediction_column(v66)
v66 = v66.rename(columns={prediction_column: "base"})
v66["well"] = v66["well"].astype(str)
needed_columns = {"well", "row_idx", "fold", "tvt_true", "md_since", "base"}
if not needed_columns.issubset(v66):
    raise RuntimeError(f"V66 missing {sorted(needed_columns.difference(v66.columns))}")
v66 = v66[list(needed_columns)].copy()
v66["fold"] = v66["fold"].astype(int)

fold_wells = {
    fold: sorted(v66.loc[v66["fold"].eq(fold), "well"].unique())
    for fold in (*TRAIN_FOLDS, SELECTION_FOLD, CONFIRMATION_FOLD)
}
failures = []
for fold in (*TRAIN_FOLDS, SELECTION_FOLD, CONFIRMATION_FOLD):
    for number, well in enumerate(fold_wells[fold], 1):
        try:
            local = v66.loc[v66["well"].eq(well)].sort_values("row_idx")
            _build_panel(well, local)
        except Exception as exc:
            failures.append({"fold": fold, "well": well, "reason": repr(exc)})
        if number % 40 == 0 or number == len(fold_wells[fold]):
            print(f"HN26 panels fold={fold} {number}/{len(fold_wells[fold])} failures={len(failures)}", flush=True)

pd.DataFrame(failures).to_csv(OUT_DIR / "panel_failures.csv", index=False)
usable = {path.stem for path in PANEL_DIR.glob("*.npz")}
fold_wells = {fold: [well for well in wells if well in usable] for fold, wells in fold_wells.items()}
coverage = {fold: len(wells) for fold, wells in fold_wells.items()}
if sum(coverage[fold] for fold in TRAIN_FOLDS) < 285 or coverage[SELECTION_FOLD] < 145 or coverage[CONFIRMATION_FOLD] < 145:
    raise RuntimeError(f"Insufficient HN26 coverage: {coverage}; failures={failures[:5]}")

model = _fit_ranker(
    fold_wells[4] + fold_wells[5],
    fold_wells[SELECTION_FOLD],
)
(OUT_DIR / "feature_names.json").write_text(
    json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8"
)

selection_scan = []
selection_base_frame = v66[v66["well"].isin(fold_wells[SELECTION_FOLD])].copy()
selection_base_metric = _metric(selection_base_frame, selection_base_frame["base"], "base")
for transition_lambda in DP_LAMBDAS:
    decoded, _ = _predict_fold(
        v66, fold_wells[SELECTION_FOLD], model, transition_lambda, 1.0
    )
    base_values = decoded["base"].to_numpy(np.float64)
    correction = decoded["prediction"].to_numpy(np.float64) - base_values
    for dose in DOSES:
        prediction = base_values + float(dose) * correction
        metrics = _metric(decoded, prediction, "candidate")
        selection_scan.append({
            "transition_lambda": transition_lambda, "dose": dose,
            "rmse": metrics["rmse"], "gain": selection_base_metric["rmse"] - metrics["rmse"],
            "far_gain": selection_base_metric["far_gt2000_rmse"] - metrics["far_gt2000_rmse"],
        })
selection_scan = pd.DataFrame(selection_scan).sort_values(
    ["rmse", "dose", "transition_lambda"], ascending=[True, True, True]
).reset_index(drop=True)
selected = selection_scan.iloc[0]
selection_scan.to_csv(OUT_DIR / "selection_scan.csv", index=False)

confirmation, station_audit = _predict_fold(
    v66, fold_wells[CONFIRMATION_FOLD], model,
    float(selected["transition_lambda"]), float(selected["dose"]),
)
shuffle, _ = _predict_fold(
    v66, fold_wells[CONFIRMATION_FOLD], model,
    float(selected["transition_lambda"]), float(selected["dose"]), shuffled=True,
)
base_metric = _metric(confirmation, confirmation["base"], "base")
candidate_metric = _metric(confirmation, confirmation["prediction"], "listwise_dp")
shuffle_metric = _metric(shuffle, shuffle["prediction"], "candidate_alignment_shuffle")

# Menu oracle is audited at stations only; it never enters reconstruction or tuning.
oracle_errors = []
for well in fold_wells[CONFIRMATION_FOLD]:
    panel = _load_panel(well)
    oracle = np.min(
        np.abs(np.asarray(panel["offsets"], np.float64) - np.asarray(panel["truth_offsets"], np.float64)[:, None]),
        axis=1,
    )
    oracle_errors.extend(oracle.tolist())
station_oracle_rmse = float(np.sqrt(np.mean(np.square(oracle_errors))))

gain = base_metric["rmse"] - candidate_metric["rmse"]
far_gain = base_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"]
shuffle_loss = shuffle_metric["rmse"] - candidate_metric["rmse"]
checks = {
    "train_wells_ge_285": sum(coverage[f] for f in TRAIN_FOLDS) >= 285,
    "selection_wells_ge_145": coverage[SELECTION_FOLD] >= 145,
    "confirmation_wells_ge_145": coverage[CONFIRMATION_FOLD] >= 145,
    "station_top32_oracle_rmse_le_6": station_oracle_rmse <= 6.0,
    "confirmation_gain_ge_0p10": gain >= 0.10,
    "confirmation_far_gain_ge_0p10": far_gain >= 0.10,
    "shuffle_loses_ge_0p10": shuffle_loss >= 0.10,
    "selected_dose_positive": float(selected["dose"]) > 0.0,
}
strict_pass = all(checks.values())
output = confirmation[["well", "row_idx", "fold", "tvt_true", "md_since", "base", "prediction"]].copy()
output.to_parquet(OUT_DIR / "hn26_fold2_oof.parquet", index=False, compression="zstd")
station_audit.to_csv(OUT_DIR / "fold2_station_audit.csv", index=False)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold45_amplitude_top32_listwise_fold3_dp_selection_fold2_confirmation",
    "sample_unit": "real amplitude aliases grouped within sampled stations and complete wells",
    "legal_inputs_only": True,
    "truth_usage": "LambdaRank labels and audit only; no prediction feature uses truth",
    "protocol": {
        "train_folds": list(TRAIN_FOLDS), "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD, "fold1_touched": False,
    },
    "coverage": coverage,
    "features": FEATURE_NAMES,
    "menu_size": MENU_SIZE,
    "best_iteration": int(model.best_iteration_),
    "selected": {
        "transition_lambda": float(selected["transition_lambda"]),
        "dose": float(selected["dose"]),
        "selection_gain": float(selected["gain"]),
        "selection_far_gain": float(selected["far_gain"]),
    },
    "confirmation": {
        "base": base_metric, "candidate": candidate_metric, "shuffle": shuffle_metric,
        "gain": gain, "far_gain": far_gain, "shuffle_loss": shuffle_loss,
        "mean_station_rmse": float(station_audit["station_rmse"].mean()),
        "mean_station_within5": float(station_audit["station_within5"].mean()),
        "station_top32_oracle_rmse": station_oracle_rmse,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> retrain with Fold3 added and test deployment-equivalent aggregation"
        if strict_pass else
        "FAIL -> top32 aliases are not learnably reorderable from legal station context; close branch"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nHN26 SELECTION TOP")
print(selection_scan.head(15).to_string(index=False))
print("\nHN26 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
