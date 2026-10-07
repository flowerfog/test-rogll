"""V91: conditional-mean consumption of V88's local deformation oracle.

Hard candidate selection failed in V90. For squared error, the Bayes action is
the conditional mean deformation. Fit a strongly regularized multi-output
ridge on Folds 1/2, select alpha/dose on 24 Fold3 wells, and confirm on the
remaining Fold3 wells. This is a coverage-limited gateway only.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_local_deformation_conditional_mean_v91"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V90_SOURCE = ROOT / "rawdata_lattice_local_deformation_ranker_v90_cell.py"
KNOTS = np.array([0.0, 0.12, 0.25, 0.40, 0.58, 0.76, 0.90, 1.0], np.float64)
ALPHAS = (10.0, 100.0, 1000.0, 10000.0)
DOSES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
MAX_MOVE_FT = 8.0


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _bootstrap(path, marker, name):
    if not path.is_file():
        raise FileNotFoundError(path)
    source = path.read_text(encoding="utf-8")
    if marker not in source:
        raise RuntimeError(f"Cannot identify execution boundary in {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(path), "exec"), module.__dict__)
    return module


def _rank(values):
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="stable")
    output = np.empty(len(values), np.float64)
    output[order] = np.arange(len(values), dtype=np.float64)
    return output / max(len(values) - 1, 1)


def _dataset(frame, caches, labelled):
    x_rows, y_rows, wells = [], [], []
    expected_names = None
    for cache in caches:
        names = tuple(cache["names"])
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise RuntimeError("Candidate order differs across wells")
        local = frame.iloc[cache["indices"]]
        baseline = local["v76"].to_numpy(np.float64)
        coarse = local[T.N.COARSE].to_numpy(np.float64)
        distance = local["md_since"].to_numpy(np.float64)
        candidates = cache["predictions"].astype(np.float64)

        feature = []
        for profile in T.PROFILE_COLUMNS:
            aligned = np.asarray(cache["aligned"][profile], np.float64)
            rolled = np.asarray(cache["rolled"][profile], np.float64)
            feature.extend(aligned.tolist())
            feature.extend(_rank(aligned).tolist())
            feature.extend((aligned - rolled).tolist())
        feature.extend(np.asarray(cache["penalty"], np.float64).tolist())
        move = coarse - baseline
        feature.extend([
            len(local) / 6000.0,
            float(np.max(distance)) / 6000.0,
            float(local["dose"].iloc[0]),
            float(np.mean(move)) / 8.0,
            float(np.std(move)) / 8.0,
            float(np.sqrt(np.mean(move * move))) / 8.0,
            float(move[-1]) / 8.0,
            float((move[-1] - move[0]) / max(np.max(distance), 1.0)) * 1000.0,
        ])
        x_rows.append(feature)
        wells.append(str(cache["well"]))

        if labelled:
            truth = local[T.N.TRUTH].to_numpy(np.float64)
            mse = np.mean((candidates - truth[None, :]) ** 2, axis=1)
            best = int(np.argmin(mse))
            oracle_move = candidates[best] - baseline
            tail_span = max(float(np.max(distance)) - 1000.0, 1.0)
            progress = np.clip((distance - 1000.0) / tail_span, 0.0, 1.0)
            knot_move = np.interp(KNOTS, progress, oracle_move)
            knot_move[0] = 0.0
            y_rows.append(knot_move)

    x = np.asarray(x_rows, np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y_rows, np.float64) if labelled else None
    return x, y, wells, expected_names


def _predict_frame(frame, caches, wells, knot_predictions, dose):
    output = frame["v76"].to_numpy(np.float64).copy()
    prediction_map = {well: knots for well, knots in zip(wells, knot_predictions)}
    for cache in caches:
        rows = cache["indices"]
        local = frame.iloc[rows]
        baseline = local["v76"].to_numpy(np.float64)
        distance = local["md_since"].to_numpy(np.float64)
        knots = np.asarray(prediction_map[str(cache["well"])], np.float64).copy()
        knots = np.clip(knots, -MAX_MOVE_FT, MAX_MOVE_FT)
        knots[0] = 0.0
        tail_span = max(float(np.max(distance)) - 1000.0, 1.0)
        progress = np.clip((distance - 1000.0) / tail_span, 0.0, 1.0)
        correction = np.interp(progress, KNOTS, knots)
        correction[distance <= 1000.0] = 0.0
        output[rows] = baseline + float(dose) * correction
    return output


def _subset_metric(frame, prediction, wells, endpoint):
    mask = frame["well"].astype(str).isin(set(wells)).to_numpy()
    local = frame.loc[mask].reset_index(drop=True)
    return T.N.O._metric(local, np.asarray(prediction)[mask], endpoint)


def _delta(base, candidate):
    return {
        "gain": base["rmse"] - candidate["rmse"],
        "far_gain": base["far_gt2000_rmse"] - candidate["far_gt2000_rmse"],
        "near_degradation": candidate["near_0_500_rmse"] - base["near_0_500_rmse"],
        "p95_gain": base["p95_well_rmse"] - candidate["p95_well_rmse"],
    }


T = _bootstrap(V90_SOURCE, "N = _bootstrap(V89_SOURCE", "v90_v91_defs")
T.N = T._bootstrap(T.V89_SOURCE, "for required in (V88_SOURCE, V86_SOURCE):", "v89_v91_defs")
T.N.O = T.N._bootstrap(T.N.V88_SOURCE, "for required in (V70_PATH, V76_WEIGHTS):", "v88_v91_defs")
T.N.M = T.N._bootstrap(
    T.N.V86_SOURCE,
    "for required in (V70_SOURCE, V85_SOURCE, V70_PATH, V76_WEIGHTS):",
    "v86_v91_defs",
)
T.N.M.R = T.N.M._bootstrap_v70()
T.N.M.H = T.N.M._bootstrap_v85_helpers()
T.N.R = T.N.M.R

train = pd.concat([T.N.O._fold_frame(1), T.N.O._fold_frame(2)], ignore_index=True)
train, train_caches = T._eligible_frame_and_caches(train, 12)
x_train, y_train, train_wells, candidate_names = _dataset(train, train_caches, True)

fold3 = T.N.O._fold_frame(3)
fold3, fold3_caches = T._eligible_frame_and_caches(fold3, 3)
x3, _, wells3, names3 = _dataset(fold3, fold3_caches, False)
if names3 != candidate_names:
    raise RuntimeError("Train/Fold3 candidate mismatch")

ordered_wells = sorted(wells3)
discovery_wells = ordered_wells[:24]
confirmation_wells = ordered_wells[24:]
base_discovery = _subset_metric(fold3, fold3["v76"], discovery_wells, "v76")
base_confirmation = _subset_metric(fold3, fold3["v76"], confirmation_wells, "v76")

models, knot_predictions = {}, {}
scan_rows = []
for alpha in ALPHAS:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x_train, y_train)
    knots = model.predict(x3)
    knots[:, 0] = 0.0
    models[alpha] = model
    knot_predictions[alpha] = knots
    for dose in DOSES:
        prediction = _predict_frame(fold3, fold3_caches, wells3, knots, dose)
        metric = _subset_metric(fold3, prediction, discovery_wells, "conditional_mean")
        scan_rows.append({"alpha": alpha, "dose": dose, **metric, **_delta(base_discovery, metric)})

scan = pd.DataFrame(scan_rows).sort_values(["rmse", "alpha", "dose"], kind="stable")
scan.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
eligible = scan[
    scan["gain"].ge(0.03)
    & scan["far_gain"].ge(0.03)
    & scan["near_degradation"].le(1e-8)
    & scan["p95_gain"].ge(-0.05)
]

selected = None
confirmation = None
strict_pass = False
summary_rows = [{"stage": "fold3_discovery", **base_discovery}]
if not eligible.empty:
    selected = eligible.iloc[0].to_dict()
    alpha, dose = float(selected["alpha"]), float(selected["dose"])
    knots = knot_predictions[alpha]
    prediction = _predict_frame(fold3, fold3_caches, wells3, knots, dose)
    metric = _subset_metric(fold3, prediction, confirmation_wells, "conditional_mean")
    confirmation = _delta(base_confirmation, metric)

    # Matched control: preserve the predicted curve distribution, assign it to the wrong wells.
    rolled_knots = np.roll(knots, max(1, len(knots) // 5), axis=0)
    rolled_prediction = _predict_frame(fold3, fold3_caches, wells3, rolled_knots, dose)
    control = _subset_metric(fold3, rolled_prediction, confirmation_wells, "rolled_well_control")
    confirmation["aligned_increment_vs_rolled"] = control["rmse"] - metric["rmse"]
    summary_rows.extend([
        {"stage": "fold3_confirmation", **base_confirmation},
        {"stage": "fold3_confirmation", **metric},
        {"stage": "fold3_confirmation", **control},
    ])
    strict_pass = bool(
        confirmation["gain"] >= 0.03
        and confirmation["far_gain"] >= 0.03
        and confirmation["near_degradation"] <= 1e-8
        and confirmation["p95_gain"] >= -0.05
        and confirmation["aligned_increment_vs_rolled"] >= 0.02
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold12_conditional_mean_Fold3_24well_selection_130well_confirmation_gateway",
    "candidate_family": "V88 frozen local deformations",
    "target": "per-well oracle deformation sampled at 8 normalized tail knots",
    "train_wells": len(train_wells),
    "feature_count": int(x_train.shape[1]),
    "selection_search_count": len(ALPHAS) * len(DOSES),
    "selected_recipe": None if selected is None else {
        "alpha": selected["alpha"], "dose": selected["dose"],
        "discovery_gain": selected["gain"], "discovery_far_gain": selected["far_gain"],
    },
    "fold3_confirmation": confirmation,
    "coverage_limited": True,
    "fold1_touched": True,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS GATEWAY -> reconstruct Folds 4/5 for true outer confirmation"
        if strict_pass else "FAIL -> V88 oracle deformation has no stable conditional mean; close branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV91 DISCOVERY TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV91 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV91 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
