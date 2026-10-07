"""V185: T2 raw-GR/momentum arm-gain router on frozen V168."""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


STARTED = time.time(); SEED = 20260803
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
DATA_ROOT = next((p for p in (ROOT / "rogii-wellbore-geology-prediction", ROOT) if (p / "train").is_dir()), ROOT)
OOF = ROOT / "oof_exports" if (ROOT / "oof_exports").is_dir() else ROOT / "output"
V168 = OOF / "historical_analog_metadata_fullrow_v168/fullrow_oof.parquet"
V144 = OOF / "final_consensus_sequence_coefficient_full5_v144/crossfit_oof.parquet"
V66 = OOF / "frontier_6p377_full_fivefold_lineage_v66/incumbent_6p377_full_fivefold_oof.parquet"
OUT = OOF / "v168_gr_momentum_arm_router_v185"; OUT.mkdir(parents=True, exist_ok=True)
PARENT = "metadata_analog_w0p05"
PARENT_SHA = "43f23cb1f53212e8da77b38e873b24ae53ddd092a8f2fa16fa761e7a1536e8c6"
ARMS = ("consensus_center", "frozen_v126_recipe", "exact_v76", "nested_sequence_coefficient", "robust_median")
FRACTIONS = (.03, .05, .10, .15); DOSES = (.25, .50, 1.0)


def _clean(v):
    if isinstance(v, dict): return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_clean(x) for x in v]
    if isinstance(v, Path): return str(v)
    if isinstance(v, (np.integer, int)): return int(v)
    if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.bool_, bool)): return bool(v)
    return v


def _stats(values, prefix):
    values = np.asarray(values, np.float64); finite = values[np.isfinite(values)]
    if not len(finite): return {f"{prefix}_{name}": 0.0 for name in ("mean", "std", "rms", "p05", "p50", "p95", "range")}
    return {f"{prefix}_mean": float(finite.mean()), f"{prefix}_std": float(finite.std()),
            f"{prefix}_rms": float(np.sqrt(np.mean(finite * finite))), f"{prefix}_p05": float(np.quantile(finite, .05)),
            f"{prefix}_p50": float(np.quantile(finite, .5)), f"{prefix}_p95": float(np.quantile(finite, .95)),
            f"{prefix}_range": float(np.ptp(finite))}


def _metric(local, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - local["tvt_true"].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    bywell = pd.DataFrame({"well": local["well"], "e2": error * error}).groupby("well", sort=False)["e2"].mean()
    result = {"endpoint": endpoint, "rows": len(local), "wells": local["well"].nunique(), "exact_sse": float(error @ error),
              "rmse": float(np.sqrt(np.mean(error * error))), "mae": float(np.mean(np.abs(error))),
              "median_well_rmse": float(np.median(np.sqrt(bywell))), "p95_well_rmse": float(np.quantile(np.sqrt(bywell), .95))}
    for name, mask in (("near_0_500_rmse", distance < 500), ("far_gt2000_rmse", distance >= 2000)):
        result[name] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


def _raw_features(well, rows):
    raw = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv", usecols=["MD", "Z", "GR", "TVT"])
    md = raw["MD"].to_numpy(np.float64); z = raw["Z"].to_numpy(np.float64); gr = raw["GR"].to_numpy(np.float64)
    anchor = int(rows.min()) - 1; hidden_gr = gr[rows]; hidden_z = z[rows]; hidden_md = md[rows]
    prefix = np.arange(max(0, anchor - 1023), anchor + 1); prefix_gr = gr[prefix]
    result = {"gr_hidden_missing_fraction": float(np.mean(~np.isfinite(hidden_gr))),
              "gr_prefix_missing_fraction": float(np.mean(~np.isfinite(prefix_gr))),
              "hidden_md_range": float(np.ptp(hidden_md)), "hidden_z_range": float(np.ptp(hidden_z)),
              "hidden_z_slope": float((hidden_z[-1] - hidden_z[0]) / max(1e-6, hidden_md[-1] - hidden_md[0])),
              "hidden_z_diff_rms": float(np.sqrt(np.mean(np.diff(hidden_z) ** 2))) if len(hidden_z) > 1 else 0.0,
              "hidden_z_curvature_rms": float(np.sqrt(np.mean(np.diff(hidden_z, 2) ** 2))) if len(hidden_z) > 2 else 0.0}
    result.update(_stats(hidden_gr, "gr_hidden")); result.update(_stats(prefix_gr, "gr_prefix"))
    finite = np.isfinite(prefix_gr)
    if finite.sum() >= 32:
        x = md[prefix][finite] - md[prefix][finite][-1]; y = prefix_gr[finite] - np.mean(prefix_gr[finite]); den = float(x @ x)
        result["gr_prefix_slope"] = float((x @ y) / den) if den > 0 else 0.0
    else: result["gr_prefix_slope"] = 0.0
    return result


for path in (V168, V144, V66):
    if not path.is_file(): raise FileNotFoundError(path)
frame = pd.read_parquet(V168)
columns144 = ["id", "consensus_center", "frozen_v126_recipe", "exact_v76", "nested_sequence_coefficient"]
frame = frame.merge(pd.read_parquet(V144, columns=columns144), on="id", how="left", validate="one_to_one")
available66 = pd.read_parquet(V66)
columns66 = [c for c in ("id", "full_cp8_ordered", "neighbor_absolute", "incumbent_6p377_replay") if c in available66]
if not {"id", "full_cp8_ordered", "neighbor_absolute"}.issubset(columns66): raise RuntimeError(f"V66 missing PF/Neighbor columns: {available66.columns.tolist()}")
frame = frame.merge(available66[columns66], on="id", how="left", validate="one_to_one")
frame["well"] = frame["well"].astype(str); frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
required_finite = columns144[1:] + ["full_cp8_ordered"]
if "incumbent_6p377_replay" in columns66: required_finite.append("incumbent_6p377_replay")
if len(frame) != 3_783_989 or frame[required_finite].isna().any().any(): raise RuntimeError("Incomplete endpoint alignment")
# V66 deliberately leaves Neighbor unavailable outside its coverage gate.
# Preserve that information as a legal feature and make its neutral move zero.
frame["neighbor_available"] = np.isfinite(frame["neighbor_absolute"].to_numpy(np.float64))
frame["neighbor_absolute"] = frame["neighbor_absolute"].fillna(frame["full_cp8_ordered"])
base = frame[PARENT].to_numpy(np.float64); parent_hash = hashlib.sha256(base.astype(np.float32).tobytes()).hexdigest()
if parent_hash != PARENT_SHA: raise RuntimeError(f"V168 hash mismatch: {parent_hash}")
frame["robust_median"] = np.median(frame[["consensus_center", "frozen_v126_recipe", "exact_v76", "nested_sequence_coefficient"]], axis=1).astype(np.float32)
truth = frame["tvt_true"].to_numpy(np.float64)

panel_rows = []; well_positions = {}
for number, (well, positions) in enumerate(frame.groupby("well", sort=True).indices.items(), 1):
    positions = np.asarray(positions, np.int64); well_positions[str(well)] = positions; local = frame.iloc[positions]
    local_base = base[positions]; distance = local["md_since"].to_numpy(np.float64); far = distance >= 2000
    row = {"well": str(well), "fold": int(local["fold"].iloc[0]), "rows": len(local), "max_distance": float(distance.max()),
           "base_range": float(np.ptp(local_base)), "base_diff_rms": float(np.sqrt(np.mean(np.diff(local_base) ** 2)))}
    row.update(_raw_features(str(well), local["row_idx"].to_numpy(np.int64)))
    row["neighbor_available_fraction"] = float(local["neighbor_available"].mean())
    row["neighbor_far_available_fraction"] = float(local.loc[far, "neighbor_available"].mean()) if far.any() else 0.0
    neighbor_move = local["neighbor_absolute"].to_numpy(np.float64) - local["full_cp8_ordered"].to_numpy(np.float64)
    row.update(_stats(neighbor_move, "neighbor_move")); row.update(_stats(neighbor_move[far], "neighbor_far_move"))
    row["neighbor_rate_rms"] = float(np.sqrt(np.mean(np.diff(neighbor_move) ** 2))) if len(neighbor_move) > 1 else 0.0
    candidates = []
    base_sse = float(np.sum((local_base - truth[positions]) ** 2)); row["base_sse"] = base_sse
    for arm in ARMS[:-1]:
        candidate = local[arm].to_numpy(np.float64); candidates.append(candidate); move = candidate - local_base
        row.update(_stats(move, f"{arm}_move")); row.update(_stats(move[far], f"{arm}_far_move"))
    candidates.append(local["robust_median"].to_numpy(np.float64))
    for arm, candidate in zip(ARMS, candidates): row[f"gain_{arm}"] = base_sse - float(np.sum((candidate - truth[positions]) ** 2))
    row.update(_stats(np.std(np.vstack(candidates), axis=0), "endpoint_spread")); panel_rows.append(row)
    if number % 100 == 0 or number == 773: print(f"V185 features {number}/773", flush=True)

panel = pd.DataFrame(panel_rows); folds = panel["fold"].to_numpy(np.int8)
targets = np.column_stack([panel[f"gain_{arm}"].to_numpy(np.float64) / panel["rows"].to_numpy(np.float64) for arm in ARMS])
excluded = {"well", "fold", "base_sse", *(f"gain_{arm}" for arm in ARMS)}
features = [c for c in panel if c not in excluded]; x = panel[features].replace([np.inf, -np.inf], np.nan)
predictions = {name: np.full_like(targets, np.nan) for name in ("hist", "extra")}
for held in (1, 2, 3, 4, 5):
    train, valid = np.flatnonzero(folds != held), np.flatnonzero(folds == held)
    for arm_index in range(len(ARMS)):
        models = {
            "hist": make_pipeline(SimpleImputer(), HistGradientBoostingRegressor(max_iter=240, learning_rate=.035, max_leaf_nodes=7, min_samples_leaf=18, l2_regularization=12, random_state=SEED + arm_index)),
            "extra": make_pipeline(SimpleImputer(), ExtraTreesRegressor(n_estimators=700, min_samples_leaf=12, max_features=.7, n_jobs=-1, random_state=SEED + arm_index)),
        }
        for name, model in models.items(): model.fit(x.iloc[train], targets[train, arm_index]); predictions[name][valid, arm_index] = model.predict(x.iloc[valid])
if any(not np.isfinite(v).all() for v in predictions.values()): raise RuntimeError("Incomplete V185 scores")
predictions["ensemble"] = .5 * (predictions["hist"] + predictions["extra"])

well_order = panel["well"].tolist(); row_well = frame["well"].map({w: i for i, w in enumerate(well_order)}).to_numpy(np.int32)
recipes = [(name, fraction, dose) for name in predictions for fraction in FRACTIONS for dose in DOSES]
base_sse_by_well = panel["base_sse"].to_numpy(np.float64)
dose_sse = {dose: {arm: np.zeros(len(panel), np.float64) for arm in ARMS} for dose in DOSES}
for index, well in enumerate(well_order):
    positions = well_positions[well]
    for arm in ARMS:
        candidate = frame.iloc[positions][arm].to_numpy(np.float64)
        for dose in DOSES:
            prediction = base[positions] + dose * (candidate - base[positions])
            dose_sse[dose][arm][index] = float(np.sum((prediction - truth[positions]) ** 2))

def _selection(name, fraction, fold, rolled=False):
    eligible = np.flatnonzero(folds == fold); values = predictions[name][eligible].copy()
    if rolled: values = np.roll(values, max(1, len(values) // 5), axis=0)
    best_arm = np.argmax(values, axis=1); best_gain = values[np.arange(len(values)), best_arm]
    count = max(1, int(round(fraction * len(eligible)))); order = np.argsort(-best_gain, kind="stable")[:count]
    return eligible[order], best_arm[order]


def _recipe_sse(name, fraction, dose, train_folds):
    train_indices = np.flatnonzero(np.isin(folds, train_folds))
    total = float(base_sse_by_well[train_indices].sum())
    for fold in train_folds:
        selected, arm_indices = _selection(name, fraction, fold)
        for index, arm_index in zip(selected, arm_indices):
            arm = ARMS[int(arm_index)]
            total += dose_sse[dose][arm][index] - base_sse_by_well[index]
    return total

nested = base.copy(); rolled = base.copy(); selections = []
for held in (1, 2, 3, 4, 5):
    train_folds = [f for f in (1, 2, 3, 4, 5) if f != held]
    selected_recipe = min(recipes, key=lambda recipe: _recipe_sse(*recipe, train_folds))
    name, fraction, dose = selected_recipe
    for is_rolled, output in ((False, nested), (True, rolled)):
        selected, arm_indices = _selection(name, fraction, held, rolled=is_rolled)
        for index, arm_index in zip(selected, arm_indices):
            positions = well_positions[well_order[index]]; candidate = frame.iloc[positions][ARMS[int(arm_index)]].to_numpy(np.float64)
            output[positions] = base[positions] + dose * (candidate - base[positions])
    selections.append({"fold": held, "model": name, "fraction": fraction, "dose": dose, "selected_wells": len(_selection(name, fraction, held)[0])})

oracle = base.copy(); oracle_arms = []
for index, well in enumerate(well_order):
    positions = well_positions[well]; choices = [(float(np.sum((base[positions] - truth[positions]) ** 2)), "base", base[positions])]
    for arm in ARMS:
        candidate = frame.iloc[positions][arm].to_numpy(np.float64); choices.append((float(np.sum((candidate - truth[positions]) ** 2)), arm, candidate))
    loss, arm, prediction = min(choices, key=lambda value: value[0]); oracle[positions] = prediction; oracle_arms.append({"well": well, "arm": arm, "sse": loss})

metrics = {name: _metric(frame, prediction, name) for name, prediction in (("v168", base), ("gr_momentum_router", nested), ("rolled_feature_control", rolled), ("arm_oracle", oracle))}
fold_rows = []
for fold in (1, 2, 3, 4, 5):
    mask = frame["fold"].to_numpy(int) == fold; local = frame.loc[mask].reset_index(drop=True)
    b, n = _metric(local, base[mask], "base"), _metric(local, nested[mask], "nested")
    fold_rows.append({"fold": fold, "gain": b["rmse"] - n["rmse"], "far_gain": b["far_gt2000_rmse"] - n["far_gt2000_rmse"], "p95_gain": b["p95_well_rmse"] - n["p95_well_rmse"]})
gain = metrics["v168"]["rmse"] - metrics["gr_momentum_router"]["rmse"]
far_gain = metrics["v168"]["far_gt2000_rmse"] - metrics["gr_momentum_router"]["far_gt2000_rmse"]
p95_gain = metrics["v168"]["p95_well_rmse"] - metrics["gr_momentum_router"]["p95_well_rmse"]
rolled_increment = metrics["rolled_feature_control"]["rmse"] - metrics["gr_momentum_router"]["rmse"]
checks = {"gain_ge_0p10": gain >= .10, "positive_folds_ge_4": sum(r["gain"] > 0 for r in fold_rows) >= 4,
          "minimum_fold_gain_ge_minus0p03": min(r["gain"] for r in fold_rows) >= -.03, "far_gain_ge_0p10": far_gain >= .10,
          "p95_not_worse": p95_gain >= 0, "aligned_beats_rolled_ge_0p05": rolled_increment >= .05}
strict_pass = bool(all(checks.values()))
panel.to_csv(OUT / "well_panel.csv", index=False); pd.DataFrame(selections).to_csv(OUT / "nested_selection.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT / "folds.csv", index=False); pd.DataFrame(oracle_arms).to_csv(OUT / "oracle_arms.csv", index=False)
verdict = {"experiment": "v168_gr_momentum_arm_router_v185", "stage": "fivefold_nested_T2_raw_observability_gateway",
           "parent": {"path": V168, "column": PARENT, "sha256": parent_hash}, "features": len(features), "arms": ARMS,
           "selection": selections, "metrics": metrics, "values": {"gain": gain, "far_gain": far_gain, "p95_gain": p95_gain, "rolled_increment": rolled_increment},
           "folds": fold_rows, "checks": checks, "strict_pass": strict_pass, "elapsed_minutes": (time.time() - STARTED) / 60,
           "next": "PASS -> package T2 after override" if strict_pass else "FAIL -> raw GR/geometry/momentum cannot route existing arms; close T2"}
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nV185 SUMMARY"); print(pd.DataFrame(metrics.values()).to_string(index=False), flush=True)
print("\nV185 FOLDS"); print(pd.DataFrame(fold_rows).to_string(index=False), flush=True)
print("\nV185 FINAL VERDICT"); print(json.dumps(_clean(verdict), indent=2), flush=True); print("saved:", OUT, flush=True)
