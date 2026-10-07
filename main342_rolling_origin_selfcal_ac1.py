"""M342-AC1: legal rolling-origin self-calibration of direct endpoints.

For each Fold-3 well, pseudo-cut predictions are scored only on rows that are
visible by the official cut.  Their robust signed bias is then ramped onto the
official hidden suffix.  A cross-well bias shuffle tests whether any gain is
actually attributable to same-well prefix calibration.
"""

from pathlib import Path
import importlib.util
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
P1_SOURCE = ROOT / "main342_k2_residual_gbdt_p1.py"
AB1_DIR = ROOT / "oof_exports" / "main342_direct_catboost_endpoint_ab1"
OUT = ROOT / "oof_exports" / "main342_rolling_origin_selfcal_ac1"
FOLD = 3
DOSES = np.asarray([0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20], np.float64)
TAUS = (64.0, 128.0, 256.0, 512.0)
MIN_VISIBLE_ROWS = 12


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()'
    if marker not in source:
        raise RuntimeError(f"Cannot locate P1 runtime marker in {path}")
    name = "main342_p1_definitions_ac1"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _metric(frame, prediction, endpoint):
    return B["_metrics"](frame.reset_index(drop=True), prediction, endpoint)


def _load_raw_state(well, tag, actual_frac):
    path, _ = B["_feature_paths"](str(well), str(tag))
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_pickle(path).reset_index(drop=True)
    frame["well"] = str(well)
    frame["cut_visible_frac"] = float(actual_frac)
    if "_row_idx" not in frame:
        raise RuntimeError(f"{well}/{tag}: missing _row_idx")
    if "id" not in frame:
        frame["id"] = str(well) + "_" + frame["_row_idx"].astype(int).astype(str)
    required = {"target", "last_known_tvt", "md_since"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{well}/{tag}: missing {sorted(missing)}")
    return frame


def _state_bias(frame, official_start, model):
    row_idx = frame["_row_idx"].to_numpy(np.int64)
    known = np.flatnonzero(row_idx < int(official_start))
    if len(known) < MIN_VISIBLE_ROWS:
        raise RuntimeError(f"only {len(known)} visible pseudo-hidden rows")
    # The latest pseudo-hidden observations are the closest legal proxy for the
    # official hidden suffix and avoid allowing long early intervals to dominate.
    known = known[np.argsort(row_idx[known])]
    known = known[-min(96, len(known)):]
    raw = np.asarray(model.predict(B["_matrix"](frame, features)), np.float64)
    prediction = B["_postprocess"](frame, raw)
    truth = (
        frame["target"].to_numpy(np.float64)
        + frame["last_known_tvt"].to_numpy(np.float64)
    )
    error = prediction[known] - truth[known]
    error = error[np.isfinite(error)]
    if len(error) < MIN_VISIBLE_ROWS:
        raise RuntimeError(f"only {len(error)} finite calibration errors")
    median = float(np.median(error))
    mad = float(np.median(np.abs(error - median)))
    return median, mad, int(len(error))


def _scan(frame, parent, candidates, shuffled):
    base_metric = _metric(frame, parent, "k2_parent")
    records = []
    for name, endpoint in candidates.items():
        shuffled_endpoint = shuffled[name]
        for dose in DOSES:
            pred = parent + float(dose) * (endpoint - parent)
            control = parent + float(dose) * (shuffled_endpoint - parent)
            metric = _metric(frame, pred, name)
            control_metric = _metric(frame, control, name + "_shuffle")
            records.append({
                "candidate": name,
                "dose": float(dose),
                "rmse": metric["rmse"],
                "gain": base_metric["rmse"] - metric["rmse"],
                "far_gain": base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
                "p95_change": metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
                "shuffle_rmse": control_metric["rmse"],
                "shuffle_loss": control_metric["rmse"] - metric["rmse"],
            })
    return base_metric, pd.DataFrame(records)


started = time.time()
required = (
    P1_SOURCE,
    AB1_DIR / "fold3_endpoints.parquet",
    AB1_DIR / "models" / "direct_lgb_fold45.joblib",
    AB1_DIR / "models" / "direct_cat_fold45.cbm",
)
for path in required:
    if not path.is_file():
        raise FileNotFoundError(path)
OUT.mkdir(parents=True, exist_ok=True)

P = _bootstrap(P1_SOURCE)
B = P["B"]
O = P["O"]
base, manifest, features, coverage = O["_prepare_inventory"]()
features = list(features)

lgb_model = joblib.load(AB1_DIR / "models" / "direct_lgb_fold45.joblib")
try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise RuntimeError("CatBoost is unavailable in this environment") from exc
cat_model = CatBoostRegressor()
cat_model.load_model(str(AB1_DIR / "models" / "direct_cat_fold45.cbm"))
models = {"lgb": lgb_model, "cat": cat_model}

frame = pd.read_parquet(AB1_DIR / "fold3_endpoints.parquet")
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
if set(frame["fold"].astype(int).unique()) != {FOLD}:
    raise RuntimeError("AB1 endpoint artifact is not pure Fold 3")
manifest = manifest.copy()
manifest["well"] = manifest["well"].astype(str)
manifest["tag"] = manifest["tag"].astype(str)
held_wells = sorted(frame["well"].unique())
calibration_rows = []
failures = []
for number, well in enumerate(held_wells, 1):
    _, official_path = B["_feature_paths"](well, "official")
    if not official_path.is_file():
        failures.append({"well": well, "tag": "official", "reason": repr(FileNotFoundError(official_path))})
        continue
    official_frame = pd.read_pickle(official_path)
    if "_row_idx" not in official_frame or official_frame.empty:
        failures.append({"well": well, "tag": "official", "reason": "missing official _row_idx"})
        continue
    official_start = int(official_frame["_row_idx"].min())
    states = manifest[
        manifest["well"].eq(well) & ~manifest["tag"].eq("official")
    ].sort_values("actual_frac", ascending=False)
    for state in states.itertuples(index=False):
        try:
            raw_state = _load_raw_state(well, state.tag, state.actual_frac)
            for model_name, model in models.items():
                bias, mad, rows = _state_bias(raw_state, official_start, model)
                calibration_rows.append({
                    "well": well,
                    "tag": str(state.tag),
                    "actual_frac": float(state.actual_frac),
                    "model": model_name,
                    "bias": bias,
                    "mad": mad,
                    "rows": rows,
                })
        except Exception as exc:
            failures.append({"well": well, "tag": str(state.tag), "reason": repr(exc)})
    if number % 25 == 0 or number == len(held_wells):
        print(f"M342-AC1 calibrate {number}/{len(held_wells)} failures={len(failures)}", flush=True)

calibration = pd.DataFrame(calibration_rows)
pd.DataFrame(failures).to_csv(OUT / "calibration_failures.csv", index=False)
if calibration.empty:
    raise RuntimeError("No legal rolling-origin calibration rows")
calibration.to_csv(OUT / "well_state_biases.csv", index=False)

biases = {}
for model_name in models:
    local = calibration[calibration["model"].eq(model_name)].copy()
    latest = (
        local.sort_values(["well", "actual_frac"], ascending=[True, False])
        .drop_duplicates("well").set_index("well")["bias"]
    )
    pooled = local.groupby("well")["bias"].median()
    biases[(model_name, "latest")] = latest
    biases[(model_name, "pooled")] = pooled

usable = set(held_wells)
for model_name in models:
    available = set(calibration.loc[calibration["model"].eq(model_name), "well"].astype(str))
    usable.intersection_update(available)
usable = sorted(usable)
calibrated_wells = len(usable)
frame = frame.reset_index(drop=True)

parent = frame["k2_parent"].to_numpy(np.float64)
md_since = frame["md_since"].to_numpy(np.float64)
candidates = {}
shuffled = {}
well_order = sorted(frame["well"].unique())
shuffle_map = dict(zip(well_order, np.roll(well_order, 1)))
for model_name, source_column in (("lgb", "direct_lgb"), ("cat", "direct_cat")):
    endpoint = frame[source_column].to_numpy(np.float64)
    for mode in ("latest", "pooled"):
        series = biases[(model_name, mode)]
        fallback = biases[(model_name, "pooled" if mode == "latest" else "latest")]
        global_bias = float(calibration.loc[calibration["model"].eq(model_name), "bias"].median())
        bias_by_well = {
            well: float(series.get(well, fallback.get(well, global_bias)))
            for well in well_order
        }
        bias = frame["well"].map(bias_by_well).to_numpy(np.float64)
        shuffled_bias = frame["well"].map(
            {well: bias_by_well.get(shuffle_map[well], global_bias) for well in well_order}
        ).to_numpy(np.float64)
        for tau in TAUS:
            ramp = 1.0 - np.exp(-np.maximum(md_since, 0.0) / tau)
            name = f"{model_name}_{mode}_t{int(tau)}"
            candidates[name] = endpoint - ramp * bias
            shuffled[name] = endpoint - ramp * shuffled_bias

base_metric, scan = _scan(frame, parent, candidates, shuffled)
safe = scan[
    (scan["far_gain"] >= 0.0)
    & (scan["p95_change"] <= 0.20)
    & (scan["shuffle_loss"] >= 0.0)
].copy()
selected = (
    safe.sort_values(["rmse", "dose"]).iloc[0]
    if not safe.empty else scan.sort_values(["rmse", "dose"]).iloc[0]
)
selected_name = str(selected["candidate"])
selected_dose = float(selected["dose"])
prediction = parent + selected_dose * (candidates[selected_name] - parent)
selected_metric = _metric(frame, prediction, "rolling_origin_selfcal")

scan = scan.sort_values(["rmse", "candidate", "dose"], ignore_index=True)
scan.to_csv(OUT / "dose_scan.csv", index=False)
pd.DataFrame({
    "id": frame["id"],
    "well": frame["well"],
    "fold": frame["fold"].astype(int),
    "tvt_true": frame["tvt_true"],
    "md_since": frame["md_since"],
    "k2_parent": parent,
    "prediction": prediction,
}).to_parquet(OUT / "fold3_prediction.parquet", index=False)

checks = {
    "calibrated_wells_ge_135": calibrated_wells >= 135,
    "selected_positive_dose": selected_dose > 0.0,
    "gain_ge_0p03": float(selected["gain"]) >= 0.03,
    "far_gain_ge_0": float(selected["far_gain"]) >= 0.0,
    "shuffle_loses_ge_0p01": float(selected["shuffle_loss"]) >= 0.01,
    "p95_change_le_0p20": float(selected["p95_change"]) <= 0.20,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_rolling_origin_selfcal_ac1",
    "stage": "fold3_legal_pseudocut_to_officialcut_direct_endpoint_selfcalibration",
    "training_count": 0,
    "legal_inputs_only": 1,
    "known_truth_usage": "only rows hidden at pseudo cut but visible at official cut",
    "protocol": {"evaluation_fold": 3, "fold2_touched": 0, "fold1_touched": 0},
    "coverage": {
        "requested_wells": len(held_wells),
        "calibrated_wells": int(calibrated_wells),
        "fallback_wells": int(frame["well"].nunique() - calibrated_wells),
        "rows": len(frame),
        "state_model_calibrations": len(calibration),
        "failures": len(failures),
    },
    "bias_summary": calibration.groupby("model")["bias"].agg(["mean", "median", "std"]).to_dict("index"),
    "parent": base_metric,
    "selected": selected.to_dict(),
    "candidate": selected_metric,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> freeze calibration recipe and run one Fold2 confirmation"
        if strict_pass else
        "FAIL -> pseudo-cut bias does not forecast official hidden bias; close AC1"
    ),
}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")

print("\nM342-AC1 DOSE SCAN", flush=True)
print(scan.head(24).to_string(index=False), flush=True)
print("\nM342-AC1 FINAL VERDICT", flush=True)
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
