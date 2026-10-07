"""V184: T2 full-773 catastrophic-well robust consensus rescue."""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


STARTED = time.time()
SEED = 20260803
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").is_dir() else ROOT / "output"
V168_PATH = OOF_ROOT / "historical_analog_metadata_fullrow_v168/fullrow_oof.parquet"
V144_PATH = OOF_ROOT / "final_consensus_sequence_coefficient_full5_v144/crossfit_oof.parquet"
OUT_DIR = OOF_ROOT / "v168_catastrophic_consensus_rescue_v184"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARENT = "metadata_analog_w0p05"
EXPECTED_PARENT_SHA256 = "43f23cb1f53212e8da77b38e873b24ae53ddd092a8f2fa16fa761e7a1536e8c6"
ARMS = ("consensus_center", "frozen_v126_recipe", "exact_v76", "nested_sequence_coefficient", "robust_median")
FRACTIONS = (0.03, 0.05, 0.10, 0.15)


def _clean(v):
    if isinstance(v, dict): return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_clean(x) for x in v]
    if isinstance(v, Path): return str(v)
    if isinstance(v, (np.integer, int)): return int(v)
    if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.bool_, bool)): return bool(v)
    return v


def _summary(values, prefix):
    values = np.asarray(values, np.float64)
    return {f"{prefix}_mean": float(np.mean(values)), f"{prefix}_std": float(np.std(values)),
            f"{prefix}_rms": float(np.sqrt(np.mean(values * values))),
            f"{prefix}_p50": float(np.quantile(values, .5)), f"{prefix}_p95": float(np.quantile(values, .95)),
            f"{prefix}_maxabs": float(np.max(np.abs(values)))}


def _metric(local, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - local["tvt_true"].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    bywell = pd.DataFrame({"well": local["well"], "e2": error * error}).groupby("well", sort=False)["e2"].mean()
    result = {"endpoint": endpoint, "rows": len(local), "wells": local["well"].nunique(),
              "exact_sse": float(error @ error), "rmse": float(np.sqrt(np.mean(error * error))),
              "mae": float(np.mean(np.abs(error))), "median_well_rmse": float(np.median(np.sqrt(bywell))),
              "p95_well_rmse": float(np.quantile(np.sqrt(bywell), .95))}
    for name, mask in (("near_0_500_rmse", distance < 500), ("far_gt2000_rmse", distance >= 2000)):
        result[name] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


def _well_features(local):
    base = local[PARENT].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    result = {"well": str(local["well"].iloc[0]), "fold": int(local["fold"].iloc[0]), "rows": len(local),
              "maximum_distance": float(distance.max()), "base_range": float(np.ptp(base)),
              "base_diff_rms": float(np.sqrt(np.mean(np.diff(base) ** 2))),
              "base_curvature_rms": float(np.sqrt(np.mean(np.diff(base, 2) ** 2))) if len(base) > 2 else 0.0}
    candidates = []
    for arm in ARMS[:-1]:
        candidate = local[arm].to_numpy(np.float64); candidates.append(candidate)
        move = candidate - base
        result.update(_summary(move, f"{arm}_move"))
        far = distance >= 2000
        if far.any(): result.update(_summary(move[far], f"{arm}_far_move"))
    stack = np.vstack(candidates)
    spread = np.std(stack, axis=0)
    result.update(_summary(spread, "endpoint_spread"))
    truth = local["tvt_true"].to_numpy(np.float64)
    result["target_mse"] = float(np.mean((base - truth) ** 2))
    return result


for path in (V168_PATH, V144_PATH):
    if not path.is_file(): raise FileNotFoundError(path)
v168 = pd.read_parquet(V168_PATH)
v144_columns = ["id", "consensus_center", "frozen_v126_recipe", "exact_v76", "nested_sequence_coefficient"]
v144 = pd.read_parquet(V144_PATH, columns=v144_columns)
frame = v168.merge(v144, on="id", how="left", validate="one_to_one")
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if len(frame) != 3_783_989 or frame["well"].nunique() != 773 or frame[v144_columns[1:]].isna().any().any():
    raise RuntimeError("Incomplete V168/V144 alignment")
base = frame[PARENT].to_numpy(np.float64)
parent_hash = hashlib.sha256(base.astype(np.float32).tobytes()).hexdigest()
if parent_hash != EXPECTED_PARENT_SHA256: raise RuntimeError(f"V168 hash mismatch: {parent_hash}")
frame["robust_median"] = np.median(frame[["consensus_center", "frozen_v126_recipe", "exact_v76", "nested_sequence_coefficient"]].to_numpy(np.float64), axis=1).astype(np.float32)

panel = pd.DataFrame([_well_features(local) for _, local in frame.groupby("well", sort=True)])
features = [c for c in panel if c not in {"well", "fold", "target_mse"}]
x = panel[features].replace([np.inf, -np.inf], np.nan)
y = np.log1p(panel["target_mse"].to_numpy(np.float64))
folds = panel["fold"].to_numpy(np.int8)
scores = {"hist": np.full(len(panel), np.nan), "extra": np.full(len(panel), np.nan)}
models = {
    "hist": make_pipeline(SimpleImputer(), HistGradientBoostingRegressor(max_iter=220, learning_rate=.035, max_leaf_nodes=7, min_samples_leaf=18, l2_regularization=10, random_state=SEED)),
    "extra": make_pipeline(SimpleImputer(), ExtraTreesRegressor(n_estimators=600, min_samples_leaf=12, max_features=.7, n_jobs=-1, random_state=SEED)),
}
for held in (1, 2, 3, 4, 5):
    train, valid = np.flatnonzero(folds != held), np.flatnonzero(folds == held)
    for name, model in models.items():
        model.fit(x.iloc[train], y[train]); scores[name][valid] = model.predict(x.iloc[valid])
if any(not np.isfinite(v).all() for v in scores.values()): raise RuntimeError("Incomplete risk scores")
scores["ensemble"] = .5 * (scores["hist"] + scores["extra"])

well_index = {well: i for i, well in enumerate(panel["well"])}
row_well = frame["well"].map(well_index).to_numpy(np.int64)
truth = frame["tvt_true"].to_numpy(np.float64)
well_sse = {"base": np.zeros(len(panel), np.float64)}
for arm in ARMS:
    well_sse[arm] = np.zeros(len(panel), np.float64)
for well, positions in frame.groupby("well", sort=False).indices.items():
    position = well_index[str(well)]
    positions = np.asarray(positions, np.int64)
    well_sse["base"][position] = float(np.sum((base[positions] - truth[positions]) ** 2))
    for arm in ARMS:
        candidate = frame.iloc[positions][arm].to_numpy(np.float64)
        well_sse[arm][position] = float(np.sum((candidate - truth[positions]) ** 2))

def _selected_indices(model_name, fraction, held_fold, rolled=False):
    eligible = np.flatnonzero(folds == held_fold)
    local_scores = scores[model_name][eligible].copy()
    if rolled and len(local_scores): local_scores = np.roll(local_scores, max(1, len(local_scores) // 5))
    count = max(1, int(round(fraction * len(eligible))))
    return eligible[np.argsort(-local_scores, kind="stable")[:count]]

recipes = [(model, fraction, arm) for model in scores for fraction in FRACTIONS for arm in ARMS]
nested = base.copy(); rolled = base.copy(); selections = []
for held in (1, 2, 3, 4, 5):
    train_folds = [f for f in (1, 2, 3, 4, 5) if f != held]
    choices = []
    for recipe in recipes:
        model_name, fraction, arm = recipe
        train_wells = np.flatnonzero(np.isin(folds, train_folds))
        exact_sse = float(well_sse["base"][train_wells].sum())
        for fold in train_folds:
            selected_indices = _selected_indices(model_name, fraction, fold)
            exact_sse += float((well_sse[arm][selected_indices] - well_sse["base"][selected_indices]).sum())
        choices.append((exact_sse, recipe))
    _, selected_recipe = min(choices, key=lambda item: item[0])
    model_name, fraction, arm = selected_recipe
    selected_indices = _selected_indices(model_name, fraction, held)
    rolled_indices = _selected_indices(model_name, fraction, held, rolled=True)
    selected_set, rolled_set = set(selected_indices), set(rolled_indices)
    held_mask = frame["fold"].to_numpy(int) == held
    nested_mask = held_mask & np.fromiter((i in selected_set for i in row_well), bool, len(frame))
    rolled_mask = held_mask & np.fromiter((i in rolled_set for i in row_well), bool, len(frame))
    nested[nested_mask] = frame.loc[nested_mask, arm].to_numpy(np.float64)
    rolled[rolled_mask] = frame.loc[rolled_mask, arm].to_numpy(np.float64)
    selections.append({"fold": held, "model": model_name, "fraction": fraction, "arm": arm, "selected_wells": len(selected_indices)})

# Per-well arm oracle is diagnostic only and measures whether this menu can rescue catastrophes.
oracle = base.copy(); oracle_choices = []
for well, positions in frame.groupby("well", sort=False).indices.items():
    positions = np.asarray(positions, np.int64)
    options = [(float(np.sum((base[positions] - truth[positions]) ** 2)), "base", base[positions])]
    for arm in ARMS:
        candidate = frame.iloc[positions][arm].to_numpy(np.float64)
        options.append((float(np.sum((candidate - truth[positions]) ** 2)), arm, candidate))
    loss, arm, prediction = min(options, key=lambda item: item[0]); oracle[positions] = prediction
    oracle_choices.append({"well": well, "arm": arm, "sse": loss})

metrics = {name: _metric(frame, prediction, name) for name, prediction in (("v168", base), ("nested_rescue", nested), ("rolled_risk_control", rolled), ("perwell_arm_oracle", oracle))}
fold_results = []
for fold in (1, 2, 3, 4, 5):
    mask = frame["fold"].to_numpy(int) == fold; local = frame.loc[mask].reset_index(drop=True)
    b, n = _metric(local, base[mask], "base"), _metric(local, nested[mask], "nested")
    fold_results.append({"fold": fold, "gain": b["rmse"] - n["rmse"], "far_gain": b["far_gt2000_rmse"] - n["far_gt2000_rmse"], "p95_gain": b["p95_well_rmse"] - n["p95_well_rmse"]})
gain = metrics["v168"]["rmse"] - metrics["nested_rescue"]["rmse"]
far_gain = metrics["v168"]["far_gt2000_rmse"] - metrics["nested_rescue"]["far_gt2000_rmse"]
p95_gain = metrics["v168"]["p95_well_rmse"] - metrics["nested_rescue"]["p95_well_rmse"]
rolled_increment = metrics["rolled_risk_control"]["rmse"] - metrics["nested_rescue"]["rmse"]
oracle_gain = metrics["v168"]["rmse"] - metrics["perwell_arm_oracle"]["rmse"]
checks = {"arm_oracle_gain_ge_0p30": oracle_gain >= .30, "nested_gain_ge_0p10": gain >= .10,
          "positive_folds_ge_4": sum(r["gain"] > 0 for r in fold_results) >= 4,
          "minimum_fold_gain_ge_minus0p03": min(r["gain"] for r in fold_results) >= -.03,
          "far_gain_ge_0p10": far_gain >= .10, "p95_not_worse": p95_gain >= 0,
          "aligned_beats_rolled_ge_0p05": rolled_increment >= .05}
strict_pass = bool(all(checks.values()))

pd.DataFrame(selections).to_csv(OUT_DIR / "nested_selection.csv", index=False)
pd.DataFrame(fold_results).to_csv(OUT_DIR / "folds.csv", index=False)
pd.DataFrame(oracle_choices).to_csv(OUT_DIR / "oracle_choices.csv", index=False)
risk = panel[["well", "fold", "target_mse"]].copy()
for name, value in scores.items(): risk[f"risk_{name}"] = value
risk.to_csv(OUT_DIR / "risk_oof.csv", index=False)
verdict = {"experiment": "v168_catastrophic_consensus_rescue_v184", "stage": "fivefold_nested_T2_fullrow_gateway",
           "parent": {"path": V168_PATH, "column": PARENT, "prediction_sha256": parent_hash},
           "candidate_arms": ARMS, "fractions": FRACTIONS, "features": len(features), "selection": selections,
           "metrics": metrics, "values": {"gain": gain, "far_gain": far_gain, "p95_gain": p95_gain,
           "rolled_increment": rolled_increment, "arm_oracle_gain": oracle_gain}, "folds": fold_results,
           "checks": checks, "strict_pass": strict_pass, "elapsed_minutes": (time.time() - STARTED) / 60,
           "next": "PASS -> package high-risk rescue after override" if strict_pass else "FAIL -> existing robust endpoints cannot rescue catastrophic wells; retain V168"}
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nV184 SUMMARY"); print(pd.DataFrame(metrics.values()).to_string(index=False), flush=True)
print("\nV184 FOLDS"); print(pd.DataFrame(fold_results).to_string(index=False), flush=True)
print("\nV184 FINAL VERDICT"); print(json.dumps(_clean(verdict), indent=2), flush=True); print("saved:", OUT_DIR, flush=True)
