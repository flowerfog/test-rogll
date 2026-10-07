"""V179: fivefold sparse formation-event observability gateway.

The target is represented as persistent 256-ft changes in the residual level
relative to a complete-well base.  Models use only deployment-time geometry,
horizontal GR, typewell GR evaluated around the base path, and the base path.
Predicted events are integrated and scored on every official hidden row.
"""

from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import balanced_accuracy_score


EXPERIMENT = "sparse_formation_event_gateway_v179"
CWD = Path.cwd()
ROOT = CWD if (CWD / "train").is_dir() else Path("/root/autodl-tmp")
DATA_ROOT = next((p for p in (ROOT / "rogii-wellbore-geology-prediction", ROOT) if (p / "train").is_dir()), ROOT)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").is_dir() else ROOT / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
BLOCK_FT = 256.0
EVENT_THRESHOLD = 1.0
SHIFTS = np.asarray([-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0])
SEED = 179


def _clean(value):
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, (np.integer, int)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)): return bool(value)
    return value


def _read_base():
    candidates = (
        (ROOT / "oof_exports" / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet", "nested_sequence_coefficient"),
        (ROOT / "oof_exports" / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet", "incumbent_6p377_replay"),
        (DATA_ROOT / "output" / "tail_only_reanchored_neighbor_gateway_v0" / "tail_only_oof.csv.gz", "base"),
    )
    for path, prediction in candidates:
        if not path.is_file(): continue
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        truth = next((c for c in ("tvt_true", "truth") if c in frame), None)
        distance = next((c for c in ("md_since", "distance") if c in frame), None)
        if truth is None or distance is None or prediction not in frame: continue
        out = frame[["well", "fold", "row_idx", truth, distance, prediction]].copy()
        out.columns = ["well", "fold", "row_idx", "truth", "distance", "base"]
        out["well"] = out["well"].astype(str); out["fold"] = out["fold"].astype(np.int8); out["row_idx"] = out["row_idx"].astype(np.int32)
        return out.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True), str(path), prediction
    raise FileNotFoundError("No supported complete OOF base")


def _fill(axis, values):
    axis = np.asarray(axis, float); values = np.asarray(values, float)
    valid = np.isfinite(axis) & np.isfinite(values)
    return np.interp(axis, axis[valid], values[valid]) if valid.sum() >= 2 else np.zeros(len(axis))


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - np.mean(a); b = b - np.mean(b)
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-8 else 0.0


def _summary(values):
    values = np.asarray(values, float)
    return [float(np.mean(values)), float(np.std(values)), float(np.quantile(values, .10)), float(np.median(values)), float(np.quantile(values, .90)), float(values[-1] - values[0])]


def _metric(frame, prediction, endpoint):
    error = np.asarray(prediction, float) - frame["truth"].to_numpy(float)
    by_well = pd.DataFrame({"well": frame["well"], "e2": error * error}).groupby("well", sort=False)["e2"].mean()
    far = frame["distance"].to_numpy(float) > 2000.0
    return {"endpoint": endpoint, "rows": int(len(frame)), "wells": int(frame["well"].nunique()),
            "exact_sse": float(np.dot(error, error)), "rmse": float(np.sqrt(np.mean(error * error))),
            "mae": float(np.mean(np.abs(error))), "median_well_rmse": float(np.median(np.sqrt(by_well))),
            "p95_well_rmse": float(np.quantile(np.sqrt(by_well), .95)),
            "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2)))}


started = time.time()
frame, base_source, base_column = _read_base()
if len(frame) != EXPECTED_ROWS or frame["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError("Incomplete base OOF")
base = frame["base"].to_numpy(float); truth = frame["truth"].to_numpy(float); folds = frame["fold"].to_numpy(np.int8)

feature_rows, target_delta, target_class, block_wells, block_folds = [], [], [], [], []
records = {}
oracle = np.full(len(frame), np.nan, float)
for number, (well, local) in enumerate(frame.groupby("well", sort=True), 1):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv", usecols=["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"])
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT", "GR"]).dropna()
    md = horizontal["MD"].to_numpy(float); x = _fill(md, horizontal["X"]); y = _fill(md, horizontal["Y"]); z = _fill(md, horizontal["Z"]); gr = _fill(md, horizontal["GR"])
    ttvt = typewell["TVT"].to_numpy(float); tgr = typewell["GR"].to_numpy(float)
    order = np.argsort(ttvt, kind="stable"); ttvt, tgr = ttvt[order], tgr[order]
    ttvt, unique_index = np.unique(ttvt, return_index=True); tgr = tgr[unique_index]
    positions = local.index.to_numpy(np.int64); eval_rows = local["row_idx"].to_numpy(np.int64)
    if np.max(np.abs(horizontal["TVT"].to_numpy(float)[eval_rows] - local["truth"].to_numpy(float))) > .01: raise RuntimeError(f"{well}: truth parity")
    visible = np.flatnonzero(np.isfinite(horizontal["TVT_input"].to_numpy(float))); anchor = int(visible[-1])
    distance = md[eval_rows] - md[anchor]
    block_id = np.floor(np.maximum(distance, 0.0) / BLOCK_FT).astype(int)
    centres, levels, local_features = [], [], []
    for block in np.unique(block_id):
        take_local = np.flatnonzero(block_id == block); rows = eval_rows[take_local]
        h = gr[rows]; b = local["base"].to_numpy(float)[take_local]
        residual = local["truth"].to_numpy(float)[take_local] - b
        centre = int(rows[len(rows) // 2]); centres.append(centre); levels.append(float(np.median(residual)))
        costs, correlations = [], []
        for shift in SHIFTS:
            reference = np.interp(b + shift, ttvt, tgr)
            scale = max(float(np.std(h)), float(np.std(reference)), 1.0)
            costs.append(float(np.mean(np.square((h - reference) / scale))))
            correlations.append(_corr(h, reference))
        best_cost = int(np.argmin(costs)); best_corr = int(np.argmax(correlations))
        feature = []
        feature += [float(np.mean(distance[take_local]) / 1000.0), float(len(rows) / BLOCK_FT), float(block)]
        feature += _summary(h) + _summary(b)
        feature += _summary(x[rows]) + _summary(y[rows]) + _summary(z[rows])
        feature += costs + correlations
        feature += [float(SHIFTS[best_cost]), float(SHIFTS[best_corr]), float(np.ptp(costs)), float(np.ptp(correlations))]
        feature += [float((x[rows[-1]] - x[rows[0]]) / max(md[rows[-1]] - md[rows[0]], 1.0)),
                    float((y[rows[-1]] - y[rows[0]]) / max(md[rows[-1]] - md[rows[0]], 1.0)),
                    float((z[rows[-1]] - z[rows[0]]) / max(md[rows[-1]] - md[rows[0]], 1.0))]
        local_features.append(feature)
    levels = np.asarray(levels, float); deltas = np.diff(np.r_[0.0, levels])
    classes = np.where(deltas > EVENT_THRESHOLD, 2, np.where(deltas < -EVENT_THRESHOLD, 0, 1)).astype(np.int8)
    feature_rows.extend(local_features); target_delta.extend(deltas); target_class.extend(classes)
    block_wells.extend([well] * len(levels)); block_folds.extend([int(local["fold"].iloc[0])] * len(levels))
    nodes = np.r_[anchor, centres]; oracle_level = np.r_[0.0, levels]
    oracle[positions] = base[positions] + np.interp(eval_rows, nodes, oracle_level)
    records[well] = {"positions": positions, "eval_rows": eval_rows, "nodes": nodes, "slice": None, "fold": int(local["fold"].iloc[0])}
    if number % 100 == 0 or number == EXPECTED_WELLS: print(f"V179 blocks {number}/{EXPECTED_WELLS}", flush=True)

x_block = np.nan_to_num(np.asarray(feature_rows, np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
y_delta = np.asarray(target_delta, float); y_class = np.asarray(target_class, np.int8); block_wells = np.asarray(block_wells, str); block_folds = np.asarray(block_folds, np.int8)
offset = 0
for well in sorted(records):
    count = len(records[well]["nodes"]) - 1; records[well]["slice"] = slice(offset, offset + count); offset += count

predicted_delta = np.full(len(x_block), np.nan, float); rolled_delta = np.full(len(x_block), np.nan, float)
predicted_class = np.full(len(x_block), -1, np.int8); rolled_class = np.full(len(x_block), -1, np.int8)
model_reports = []
for held_fold in range(1, 6):
    train = block_folds != held_fold; valid = ~train
    counts = pd.Series(block_wells[train]).value_counts().to_dict()
    equal_well = np.asarray([1.0 / counts[w] for w in block_wells[train]], float); equal_well *= len(equal_well) / equal_well.sum()
    event_weight = equal_well * np.where(y_class[train] == 1, 1.0, 4.0)
    classifier = ExtraTreesClassifier(n_estimators=256, max_depth=14, min_samples_leaf=8, max_features=.8, class_weight="balanced", n_jobs=-1, random_state=SEED + held_fold)
    regressor = ExtraTreesRegressor(n_estimators=256, max_depth=14, min_samples_leaf=8, max_features=.8, n_jobs=-1, random_state=SEED + 10 + held_fold)
    classifier.fit(x_block[train], y_class[train], sample_weight=equal_well)
    regressor.fit(x_block[train], y_delta[train], sample_weight=event_weight)
    predicted_class[valid] = classifier.predict(x_block[valid]); predicted_delta[valid] = np.clip(regressor.predict(x_block[valid]), -6.0, 6.0)
    for well in sorted(records):
        record = records[well]
        if record["fold"] != held_fold: continue
        sl = record["slice"]; rolled_x = np.roll(x_block[sl], max(1, int(round((sl.stop - sl.start) * .20))), axis=0)
        rolled_class[sl] = classifier.predict(rolled_x); rolled_delta[sl] = np.clip(regressor.predict(rolled_x), -6.0, 6.0)
    model_reports.append({"heldout_fold": held_fold, "train_blocks": int(train.sum()), "valid_blocks": int(valid.sum()), "features": int(x_block.shape[1])})
    del classifier, regressor; gc.collect()

candidate = np.full(len(frame), np.nan, float); rolled_candidate = np.full(len(frame), np.nan, float)
for well in sorted(records):
    record = records[well]; sl = record["slice"]; positions = record["positions"]; eval_rows = record["eval_rows"]; nodes = record["nodes"]
    level = np.r_[0.0, np.cumsum(predicted_delta[sl])]; rolled_level = np.r_[0.0, np.cumsum(rolled_delta[sl])]
    candidate[positions] = base[positions] + np.interp(eval_rows, nodes, level)
    rolled_candidate[positions] = base[positions] + np.interp(eval_rows, nodes, rolled_level)

nested = np.full(len(frame), np.nan, float); nested_rolled = np.full(len(frame), np.nan, float); blend_rows = []
for held_fold in range(1, 6):
    train = folds != held_fold; valid = ~train; move = candidate[train] - base[train]
    alpha = float(np.dot(truth[train] - base[train], move) / max(np.dot(move, move), 1e-12)); alpha = float(np.clip(alpha, 0.0, 1.0))
    nested[valid] = base[valid] + alpha * (candidate[valid] - base[valid]); nested_rolled[valid] = base[valid] + alpha * (rolled_candidate[valid] - base[valid])
    blend_rows.append({"heldout_fold": held_fold, "selected_alpha": alpha})

summary = pd.DataFrame([_metric(frame, base, "base_current"), _metric(frame, candidate, "event_candidate"), _metric(frame, nested, "nested_blend"), _metric(frame, nested_rolled, "rolled_feature_control"), _metric(frame, oracle, "block_level_oracle")])
fold_rows = []
for fold in range(1, 6):
    mask = folds == fold; local = frame.loc[mask].reset_index(drop=True)
    for endpoint, pred in (("base_current", base[mask]), ("nested_blend", nested[mask]), ("rolled_feature_control", nested_rolled[mask])): fold_rows.append({"fold": fold, **_metric(local, pred, endpoint)})
fold_summary = pd.DataFrame(fold_rows)

event_mask = y_class != 1
event_balanced = float(balanced_accuracy_score(y_class, predicted_class))
event_sign = float(np.mean(predicted_class[event_mask] == y_class[event_mask])) if event_mask.any() else 0.0
rolled_sign = float(np.mean(rolled_class[event_mask] == y_class[event_mask])) if event_mask.any() else 0.0
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False); fold_summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
pd.DataFrame(model_reports).to_csv(OUT_DIR / "model_reports.csv", index=False); pd.DataFrame(blend_rows).to_csv(OUT_DIR / "blend_weights.csv", index=False)
pd.DataFrame({"well": block_wells, "fold": block_folds, "target_delta": y_delta, "target_class": y_class, "predicted_delta": predicted_delta, "predicted_class": predicted_class}).to_csv(OUT_DIR / "block_oof.csv.gz", index=False, compression="gzip")

lookup = summary.set_index("endpoint"); base_rmse = float(lookup.loc["base_current", "rmse"]); blend_rmse = float(lookup.loc["nested_blend", "rmse"]); rolled_rmse = float(lookup.loc["rolled_feature_control", "rmse"]); oracle_rmse = float(lookup.loc["block_level_oracle", "rmse"])
pivot = fold_summary.pivot(index="fold", columns="endpoint", values="rmse"); fold_gains = (pivot["base_current"] - pivot["nested_blend"]).to_dict()
verdict = {"experiment": EXPERIMENT, "stage": "complete_fivefold_sparse_formation_event_gateway", "base_source": base_source, "base_column": base_column,
           "legal_inputs": ["base path", "MD/X/Y/Z/GR", "typewell TVT/GR"], "target": "256-ft persistent residual-level change", "blocks": int(len(x_block)),
           "class_counts": pd.Series(y_class).value_counts().sort_index().to_dict(), "values": {"base_rmse": base_rmse, "nested_blend_rmse": blend_rmse, "nested_gain": base_rmse - blend_rmse,
           "block_oracle_rmse": oracle_rmse, "event_balanced_accuracy": event_balanced, "event_sign_accuracy": event_sign, "rolled_event_sign_accuracy": rolled_sign,
           "aligned_increment_vs_rolled": rolled_rmse - blend_rmse, "fold_gains": fold_gains},
           "checks": {"block_oracle_rmse_lt_6": oracle_rmse < 6.0, "event_balanced_accuracy_ge_0p55": event_balanced >= .55, "event_sign_accuracy_ge_0p60": event_sign >= .60,
           "aligned_event_sign_beats_rolled_ge_0p05": event_sign - rolled_sign >= .05, "nested_gain_ge_0p20": base_rmse - blend_rmse >= .20, "positive_folds_5of5": all(v > 0 for v in fold_gains.values())},
           "elapsed_minutes": (time.time() - started) / 60.0}
verdict["strict_pass"] = bool(all(verdict["checks"].values())); verdict["next"] = "PASS -> build event-magnitude sequence decoder" if verdict["strict_pass"] else "FAIL -> event direction is not observable; close sparse-event correction"
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nV179 MATCHED SUMMARY"); print(summary.sort_values("rmse").to_string(index=False), flush=True)
print("\nV179 FOLD SUMMARY"); print(fold_summary.to_string(index=False), flush=True)
print("\nV179 FINAL VERDICT"); print(json.dumps(_clean(verdict), indent=2), flush=True); print("saved:", OUT_DIR, flush=True)
