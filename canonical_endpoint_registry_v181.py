"""V181: canonical full-row endpoint and metric-lineage registry.

Known artifacts are aligned on (well, row_idx), checked against native truth,
and scored under pooled full-row, equal-well, and fixed 192-node metrics.  Each
endpoint is explicitly tagged for deployment legality and parent lineage.
"""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd


EXPERIMENT = "canonical_endpoint_registry_v181"
CWD = Path.cwd()
ROOT = CWD if (CWD / "train").is_dir() else Path("/root/autodl-tmp")
DATA_ROOT = next((p for p in (ROOT / "rogii-wellbore-geology-prediction", ROOT) if (p / "train").is_dir()), ROOT)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").is_dir() else ROOT / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXPECTED_ROWS, EXPECTED_WELLS = 3_783_989, 773

SPECS = (
    ("v66_incumbent", ROOT / "oof_exports/frontier_6p377_full_fivefold_lineage_v66/incumbent_6p377_full_fivefold_oof.parquet", "incumbent_6p377_replay", True, "v66"),
    ("v144_sequence", ROOT / "oof_exports/final_consensus_sequence_coefficient_full5_v144/crossfit_oof.parquet", "nested_sequence_coefficient", True, "v144"),
    ("v168_metadata_analog", ROOT / "oof_exports/historical_analog_metadata_fullrow_v168/fullrow_oof.parquet", "metadata_analog_w0p05", True, "v144"),
    ("v169_nested_consensus", ROOT / "oof_exports/historical_analog_ranked_consensus_v169/fullrow_oof.parquet", "nested_consensus", True, "v144"),
    ("local_old_cp8", DATA_ROOT / "output/tail_only_reanchored_neighbor_gateway_v0/tail_only_oof.csv.gz", "base", True, "old_local"),
    ("v174_horizon_oracle", ROOT / "oof_exports/horizon_structural_coordinate_gateway_v174/oof.csv.gz", "horizon_ridge_crossfit", False, "train_only_horizon"),
    ("v177_gradient_student", ROOT / "oof_exports/horizon_gradient_teacher_v177/oof.csv.gz", "nested_blend", True, "old_local"),
    ("v178_residual_crf", ROOT / "oof_exports/residual_centered_correlation_crf_v178/oof.csv.gz", "nested_blend", True, "old_local"),
)


def _clean(v):
    if isinstance(v, dict): return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_clean(x) for x in v]
    if isinstance(v, Path): return str(v)
    if isinstance(v, (np.integer, int)): return int(v)
    if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.bool_, bool)): return bool(v)
    return v


def _read(path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _columns(frame):
    well = next((c for c in ("well", "well_id") if c in frame), None)
    row = next((c for c in ("row_idx", "row_index") if c in frame), None)
    truth = next((c for c in ("tvt_true", "truth", "target_tvt") if c in frame), None)
    fold = next((c for c in ("fold", "outer_fold") if c in frame), None)
    distance = next((c for c in ("md_since", "distance") if c in frame), None)
    return well, row, truth, fold, distance


_NATIVE_TRUTH_CACHE = {}


def _native_truth(well, rows):
    if well not in _NATIVE_TRUTH_CACHE:
        horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv", usecols=["TVT"])
        _NATIVE_TRUTH_CACHE[well] = horizontal["TVT"].to_numpy(np.float64)
    return _NATIVE_TRUTH_CACHE[well][rows]


started = time.time(); registry = []; canonical_keys = None; canonical_truth = None
for name, path, prediction, deployable, lineage in SPECS:
    if not path.is_file():
        registry.append({"endpoint": name, "status": "missing", "path": str(path), "prediction_column": prediction, "deployable": deployable, "lineage": lineage})
        continue
    try:
        frame = _read(path); well_col, row_col, truth_col, fold_col, distance_col = _columns(frame)
        required = [well_col, row_col, truth_col]
        if any(c is None for c in required) or prediction not in frame:
            raise RuntimeError(f"columns={frame.columns.tolist()[:30]}")
        local = frame[[well_col, row_col, truth_col, prediction] + ([fold_col] if fold_col else []) + ([distance_col] if distance_col else [])].copy()
        rename = {well_col: "well", row_col: "row_idx", truth_col: "truth", prediction: "prediction"}
        if fold_col: rename[fold_col] = "fold"
        if distance_col: rename[distance_col] = "distance"
        local = local.rename(columns=rename); local["well"] = local["well"].astype(str); local["row_idx"] = local["row_idx"].astype(np.int32)
        duplicate_keys = int(local.duplicated(["well", "row_idx"]).sum()); local = local.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
        keys = pd.util.hash_pandas_object(local[["well", "row_idx"]], index=False).to_numpy(np.uint64)
        key_hash = hashlib.sha256(keys.tobytes()).hexdigest()
        if canonical_keys is None and len(local) == EXPECTED_ROWS and local["well"].nunique() == EXPECTED_WELLS:
            canonical_keys, canonical_truth = keys.copy(), local["truth"].to_numpy(np.float64).copy()
        key_match = bool(canonical_keys is not None and len(keys) == len(canonical_keys) and np.array_equal(keys, canonical_keys))
        truth_gap = float(np.max(np.abs(local["truth"].to_numpy(np.float64) - canonical_truth))) if key_match else np.nan
        native_gaps = []
        for well, sample in local.groupby("well", sort=False):
            native_gaps.append(float(np.max(np.abs(sample["truth"].to_numpy(np.float64) - _native_truth(well, sample["row_idx"].to_numpy(np.int64))))))
        pred = local["prediction"].to_numpy(np.float64); truth = local["truth"].to_numpy(np.float64); error = pred - truth
        by_well = pd.DataFrame({"well": local["well"], "e2": error * error}).groupby("well", sort=False)["e2"].mean()
        sampled = []
        for _, positions in local.groupby("well", sort=False).indices.items():
            positions = np.asarray(positions, np.int64); take = np.unique(np.round(np.linspace(0, len(positions) - 1, min(192, len(positions)))).astype(int)); sampled.extend(positions[take])
        sampled = np.asarray(sampled, np.int64)
        fold_counts = local.groupby("fold")["well"].nunique().to_dict() if "fold" in local else {}
        registry.append({"endpoint": name, "status": "complete" if len(local) == EXPECTED_ROWS and local["well"].nunique() == EXPECTED_WELLS else "partial",
                         "path": str(path), "prediction_column": prediction, "deployable": deployable, "lineage": lineage, "rows": len(local), "wells": local["well"].nunique(),
                         "duplicate_keys": duplicate_keys, "key_hash": key_hash, "canonical_key_match": key_match, "canonical_truth_max_gap": truth_gap,
                         "native_truth_max_gap": max(native_gaps), "pooled_fullrow_rmse": float(np.sqrt(np.mean(error * error))),
                         "equal_well_rmse": float(np.sqrt(np.mean(by_well))), "median_well_rmse": float(np.median(np.sqrt(by_well))),
                         "sample192_rmse": float(np.sqrt(np.mean(error[sampled] ** 2))), "fold_wells": fold_counts,
                         "prediction_sha256": hashlib.sha256(pred.astype(np.float32).tobytes()).hexdigest()})
        print(f"V181 {name}: rows={len(local)} rmse={np.sqrt(np.mean(error * error)):.6f}", flush=True)
    except Exception as exc:
        registry.append({"endpoint": name, "status": "error", "path": str(path), "prediction_column": prediction, "deployable": deployable, "lineage": lineage, "error": repr(exc)})

table = pd.DataFrame(registry); table.to_csv(OUT_DIR / "endpoint_registry.csv", index=False)
eligible = table[(table["status"].eq("complete")) & (table["deployable"].eq(True)) & (table["canonical_key_match"].eq(True)) & (table["native_truth_max_gap"].le(.01))].copy() if "canonical_key_match" in table else pd.DataFrame()
best = None if eligible.empty else eligible.sort_values("pooled_fullrow_rmse").iloc[0].to_dict()
metric_spread = float((eligible["sample192_rmse"] - eligible["pooled_fullrow_rmse"]).abs().max()) if not eligible.empty else np.nan
required_strong = {"v144_sequence", "v168_metadata_analog"}
complete_names = set(table.loc[table["status"].eq("complete"), "endpoint"].astype(str))
strong_lineage_present = required_strong.issubset(complete_names)
verdict = {"experiment": EXPERIMENT, "stage": "canonical_completewell_metric_and_lineage_registry", "files_checked": len(SPECS),
           "complete_endpoints": int(table["status"].eq("complete").sum()), "eligible_deployable_endpoints": int(len(eligible)), "best_deployable": best,
           "metric_definition": {"primary": "sqrt(sum row squared error / all official hidden rows)", "target": "absolute TVT; equivalent to anchored dTVT RMSE when the same last-known TVT is subtracted", "sample192": "diagnostic only", "equal_well": "diagnostic only"},
           "maximum_sample192_vs_fullrow_gap": metric_spread,
           "checks": {"canonical_complete_endpoint_exists": not eligible.empty, "all_eligible_native_truth_parity": bool(not eligible.empty and eligible["native_truth_max_gap"].le(.01).all()),
                      "all_eligible_unique_keys": bool(not eligible.empty and eligible["duplicate_keys"].eq(0).all()), "best_endpoint_identified": best is not None,
                      "v144_and_v168_artifacts_present": strong_lineage_present},
           "elapsed_minutes": (time.time() - started) / 60.0}
verdict["strict_pass"] = bool(all(verdict["checks"].values())); verdict["next"] = "Freeze best_deployable as the sole parent for all future full-row experiments" if verdict["strict_pass"] else "Run this registry on AutoDL where V144/V168 artifacts exist"
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nV181 ENDPOINT REGISTRY"); print(table[[c for c in ("endpoint", "status", "deployable", "lineage", "rows", "wells", "pooled_fullrow_rmse", "sample192_rmse", "native_truth_max_gap") if c in table]].to_string(index=False), flush=True)
print("\nV181 FINAL VERDICT"); print(json.dumps(_clean(verdict), indent=2), flush=True); print("saved:", OUT_DIR, flush=True)
