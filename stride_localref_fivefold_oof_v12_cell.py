"""Five-fold STRIDE replay with the V11-selected local typewell reference.

V11 selected local_050 using only held-out visible-prefix GR and confirmed it
on the Fold-5 hidden suffix.  This cell freezes that legal recipe, reuses each
V9 outer-train prior, regenerates all five STRIDE endpoint folds, and cross-fits
the scalar SP45/STRIDE blend weight using only the other four OOF folds.
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "stride_localref_fivefold_oof_v12"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
FOLD_DIR = OUT_DIR / "folds"
FOLD_DIR.mkdir(parents=True, exist_ok=True)

V9_DIR = OOF_ROOT / "stride_sp45_fivefold_oof_v9"
V11_VERDICT = OOF_ROOT / "stride_typewell_reference_audit_v11" / "verdict.json"
V3_VERDICT = OOF_ROOT / "stride_figure1_posterior_parity_v3" / "verdict.json"
V5_VERDICT = OOF_ROOT / "stride_sequence_scale_repair_v5" / "verdict.json"
SOURCE_CANDIDATES = (
    ROOT / "stride_fullsequence_beam_v4_cell.py",
    Path.cwd() / "stride_fullsequence_beam_v4_cell.py",
)
POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
POOL_PATH = next((path for path in POOL_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload the updated stride_fullsequence_beam_v4_cell.py")
if POOL_PATH is None:
    raise FileNotFoundError("Missing candidate_pool_full_merged_oof_honest_fle.csv")
for required in (V11_VERDICT, V3_VERDICT, V5_VERDICT):
    if not required.exists():
        raise FileNotFoundError(required)
for fold in range(1, 6):
    for required in (
        V9_DIR / "folds" / f"fold{fold}_prior.json",
        V9_DIR / "folds" / f"fold{fold}_stride_raw.npz",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

HELDOUT_FOLDS = (1, 2, 3, 4, 5)
BEAM_SIZE = 48
REFERENCE_KIND = "local"
REFERENCE_DOSE = 0.50
REFERENCE_NAME = "local_050"


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "v3" in names:
                break
        body.append(node)
    name = "stride_v4_repaired_definitions_for_v12"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
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


def _local_reference(record, dose=REFERENCE_DOSE):
    raw = record["tw_gr_raw"].astype(np.float64)
    known_rows = record["known"]
    tvt = record["tvt_input"][known_rows]
    observed = record["gr"][known_rows]
    expected = np.interp(tvt, record["tw_tvt"], raw)
    valid = np.isfinite(tvt) & np.isfinite(observed) & np.isfinite(expected)
    tvt = tvt[valid]
    residual = (observed - expected)[valid]
    if len(tvt) < 32:
        return raw.copy()

    order = np.argsort(tvt, kind="stable")
    work = pd.DataFrame({"tvt": tvt[order], "residual": residual[order]})
    work["bin"] = np.round(work["tvt"] * 2.0) / 2.0
    work = work.groupby("bin", as_index=False)["residual"].median().sort_values("bin")
    window = min(31, len(work))
    if window % 2 == 0:
        window -= 1
    if window >= 5:
        work["residual"] = (
            work["residual"]
            .rolling(window, center=True, min_periods=max(3, window // 3))
            .median()
            .interpolate(limit_direction="both")
        )
    correction = np.interp(
        record["tw_tvt"],
        work["bin"].to_numpy(np.float64),
        work["residual"].to_numpy(np.float64),
    )
    return raw + float(dose) * correction


def _metrics(frame, prediction_column):
    error = frame[prediction_column].to_numpy(np.float64) - frame["truth"].to_numpy(np.float64)
    well_rmse = np.sqrt(pd.Series(error * error).groupby(frame["well"].to_numpy()).mean())
    output = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(well_rmse.mean()),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
    }
    distance = frame["distance_ft"].to_numpy(np.float64)
    for name, low, high in (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    ):
        mask = (distance >= low) & (distance < high)
        output[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return output


def _save_fold(path, wells, well_code, row_idx, prediction):
    np.savez_compressed(
        path,
        wells=np.asarray(wells, dtype="U8"),
        well_code=np.asarray(well_code, dtype=np.int16),
        row_idx=np.asarray(row_idx, dtype=np.int32),
        stride_local050=np.asarray(prediction, dtype=np.float32),
    )


def _load_fold_frame(path, records, heldout_fold, prediction_key="stride_local050"):
    with np.load(path, allow_pickle=False) as data:
        wells = data["wells"].astype(str)
        well_code = data["well_code"].astype(np.int64)
        row_idx = data["row_idx"].astype(np.int64)
        prediction = data[prediction_key].astype(np.float32)
    well = wells[well_code]
    truth = np.empty(len(row_idx), np.float32)
    distance = np.empty(len(row_idx), np.float32)
    for code, name in enumerate(wells):
        mask = well_code == code
        record = records[name]
        local_rows = row_idx[mask]
        truth[mask] = record["truth"][local_rows]
        distance[mask] = record["md"][local_rows] - record["anchor_md"]
    return pd.DataFrame(
        {
            "id": [f"{name}_{int(row)}" for name, row in zip(well, row_idx)],
            "well": well,
            "row_idx": row_idx,
            "fold": heldout_fold,
            "truth": truth,
            prediction_key: prediction,
            "distance_ft": distance,
        }
    )


V = _bootstrap(SOURCE_PATH)
v3 = json.loads(V3_VERDICT.read_text(encoding="utf-8"))
v5 = json.loads(V5_VERDICT.read_text(encoding="utf-8"))
v11 = json.loads(V11_VERDICT.read_text(encoding="utf-8"))
if not v11.get("strict_pass", False):
    raise RuntimeError("V11 did not pass")
if v11["selection"]["selected_recipe"] != REFERENCE_NAME:
    raise RuntimeError(f"Unexpected V11 recipe: {v11['selection']['selected_recipe']}")

recipe = v5["selected_recipe"]
temperature = float(v3["selected"]["emission_temperature"])
tolerance = float(v3["selected"]["segmentation_tolerance_ft"])
folds = pd.read_csv(V.V.FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
if len(folds) != 773 or set(folds["fold"].unique()) != set(HELDOUT_FOLDS):
    raise RuntimeError(f"Unexpected fold metadata: {len(folds)} / {sorted(folds['fold'].unique())}")

records = {}
failures = []
for number, well in enumerate(sorted(folds["well"]), start=1):
    try:
        records[well] = V.load_record(well)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(folds):
        print(f"STRIDE v12 load records {number}/{len(folds)} failures={len(failures)}", flush=True)
if failures:
    pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)
    raise RuntimeError(f"{len(failures)} record failures: {failures[:5]}")

started = time.time()
V.BEAM_SIZE = BEAM_SIZE
fold_frames = []
raw_fold_frames = []
for heldout_fold in HELDOUT_FOLDS:
    artifact = FOLD_DIR / f"fold{heldout_fold}_stride_local050.npz"
    raw_artifact = V9_DIR / "folds" / f"fold{heldout_fold}_stride_raw.npz"
    prior_path = V9_DIR / "folds" / f"fold{heldout_fold}_prior.json"
    heldout_wells = sorted(folds.loc[folds["fold"].eq(heldout_fold), "well"])
    prior = json.loads(prior_path.read_text(encoding="utf-8"))

    if not artifact.exists():
        fold_well_code = []
        fold_row_idx = []
        fold_prediction = []
        for code, well in enumerate(heldout_wells):
            record = records[well]
            calibrated = dict(record)
            calibrated["tw_gr_raw"] = _local_reference(record)
            prediction, _, _ = V._decode(
                calibrated,
                prior,
                temperature,
                "raw_reference",
                effective_spacing_ft=float(recipe["effective_spacing_ft"]),
                persistence_weight=float(recipe["persistence_weight"]),
                lambda_override=float(recipe["lambda_previous_dip"]),
            )
            evaluation = record["evaluation"]
            fold_well_code.append(np.full(len(evaluation), code, np.int16))
            fold_row_idx.append(evaluation.astype(np.int32))
            fold_prediction.append(prediction.astype(np.float32))
            number = code + 1
            if number % 25 == 0 or number == len(heldout_wells):
                print(f"STRIDE v12 Fold{heldout_fold} decode {number}/{len(heldout_wells)}", flush=True)
        _save_fold(
            artifact,
            heldout_wells,
            np.concatenate(fold_well_code),
            np.concatenate(fold_row_idx),
            np.concatenate(fold_prediction),
        )
        print(f"STRIDE v12 saved Fold{heldout_fold}: {artifact}", flush=True)
    else:
        print(f"STRIDE v12 resume Fold{heldout_fold}: {artifact}", flush=True)

    fold_frames.append(_load_fold_frame(artifact, records, heldout_fold))
    raw_fold_frames.append(
        _load_fold_frame(raw_artifact, records, heldout_fold, prediction_key="stride_raw")
    )

frame = pd.concat(fold_frames, ignore_index=True)
raw_frame = pd.concat(raw_fold_frames, ignore_index=True)[
    ["id", "well", "fold", "stride_raw"]
]
frame = frame.merge(raw_frame, on=["id", "well", "fold"], how="inner", validate="one_to_one")
if frame["well"].nunique() != 773:
    raise RuntimeError(f"OOF lost wells: {frame['well'].nunique()}")

pool = pd.read_csv(
    POOL_PATH,
    usecols=["id", "well", "tvt_true", "sp45"],
    dtype={"id": str, "well": str},
)
frame = frame.merge(pool, on=["id", "well"], how="inner", validate="one_to_one")
if frame["well"].nunique() != 773:
    raise RuntimeError(f"Pool alignment lost wells: {frame['well'].nunique()}")
truth_gap = np.abs(frame["truth"].to_numpy(np.float64) - frame["tvt_true"].to_numpy(np.float64))
if not np.isfinite(truth_gap).all() or float(np.max(truth_gap)) > 0.01:
    raise RuntimeError(f"Truth mismatch: max={float(np.max(truth_gap))}")
frame.drop(columns="tvt_true", inplace=True)

truth = frame["truth"].to_numpy(np.float64)
sp45 = frame["sp45"].to_numpy(np.float64)
stride = frame["stride_local050"].to_numpy(np.float64)
direction = stride - sp45
sp45_error = sp45 - truth
frame["crossfit_blend"] = np.nan
weight_rows = []
for heldout_fold in HELDOUT_FOLDS:
    train_mask = frame["fold"].to_numpy() != heldout_fold
    valid_mask = ~train_mask
    local_direction = direction[train_mask]
    denominator = float(np.dot(local_direction, local_direction))
    weight = float(np.clip(-np.dot(sp45_error[train_mask], local_direction) / denominator, 0.0, 1.0))
    frame.loc[valid_mask, "crossfit_blend"] = sp45[valid_mask] + weight * direction[valid_mask]
    local = frame.loc[valid_mask]
    sp45_rmse = _metrics(local, "sp45")["rmse"]
    blend_rmse = _metrics(local, "crossfit_blend")["rmse"]
    raw_rmse = _metrics(local, "stride_raw")["rmse"]
    local_rmse = _metrics(local, "stride_local050")["rmse"]
    weight_rows.append(
        {
            "heldout_fold": heldout_fold,
            "train_wells": int(frame.loc[train_mask, "well"].nunique()),
            "valid_wells": int(local["well"].nunique()),
            "weight_stride": weight,
            "sp45_rmse": sp45_rmse,
            "raw_stride_rmse": raw_rmse,
            "localref_stride_rmse": local_rmse,
            "reference_gain": raw_rmse - local_rmse,
            "blend_rmse": blend_rmse,
            "blend_gain": sp45_rmse - blend_rmse,
        }
    )
if not np.isfinite(frame["crossfit_blend"]).all():
    raise RuntimeError("Cross-fit blend has missing rows")

weight_frame = pd.DataFrame(weight_rows)
weight_frame.to_csv(OUT_DIR / "crossfit_weights.csv", index=False)
overall = {
    "sp45": _metrics(frame, "sp45"),
    "stride_raw_v9": _metrics(frame, "stride_raw"),
    "stride_local050": _metrics(frame, "stride_local050"),
    "crossfit_blend": _metrics(frame, "crossfit_blend"),
}
fold_metrics = []
for heldout_fold in HELDOUT_FOLDS:
    local = frame.loc[frame["fold"].eq(heldout_fold)]
    for endpoint in ("sp45", "stride_raw", "stride_local050", "crossfit_blend"):
        fold_metrics.append({"fold": heldout_fold, "endpoint": endpoint, **_metrics(local, endpoint)})
pd.DataFrame(fold_metrics).to_csv(OUT_DIR / "fold_metrics.csv", index=False)

stride_error = stride - truth
raw_correlation = float(np.corrcoef(sp45_error, stride_error)[0, 1])
distance_bin = pd.cut(
    frame["distance_ft"], bins=[-np.inf, 500.0, 1000.0, 2000.0, np.inf], labels=False
).to_numpy()
sp45_detrended = sp45_error.copy()
stride_detrended = stride_error.copy()
within_strata = {}
for index, name in enumerate(("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")):
    mask = distance_bin == index
    within_strata[name] = float(np.corrcoef(sp45_error[mask], stride_error[mask])[0, 1])
    sp45_detrended[mask] -= np.mean(sp45_detrended[mask])
    stride_detrended[mask] -= np.mean(stride_detrended[mask])
detrended_correlation = float(np.corrcoef(sp45_detrended, stride_detrended)[0, 1])

reference_gain = float(overall["stride_raw_v9"]["rmse"] - overall["stride_local050"]["rmse"])
blend_gain = float(overall["sp45"]["rmse"] - overall["crossfit_blend"]["rmse"])
positive_reference_folds = int((weight_frame["reference_gain"] > 0.0).sum())
positive_blend_folds = int((weight_frame["blend_gain"] > 0.0).sum())
verdict = {
    "experiment": EXPERIMENT,
    "stage": "773_well_fivefold_local_reference_STRIDE_and_SP45_crossfit_blend",
    "protocol": {
        "folds": list(HELDOUT_FOLDS),
        "prior_fit": "V9 other-four-fold prior reused for every held-out fold",
        "reference_selection": "V11 visible-prefix-only selection; local_050 frozen",
        "blend_weight_fit": "other four OOF folds only",
        "hidden_truth_used_for_reference": False,
    },
    "frozen_stride_recipe": {
        **recipe,
        "reference": REFERENCE_NAME,
        "reference_kind": REFERENCE_KIND,
        "reference_dose": REFERENCE_DOSE,
        "likelihood": "cauchy",
        "beam_size": BEAM_SIZE,
        "segmentation_tolerance_ft": tolerance,
        "emission_temperature": temperature,
    },
    "overall": overall,
    "correlation": {
        "signed_error_raw": raw_correlation,
        "signed_error_distance_strata_demeaned": detrended_correlation,
        "within_distance_strata": within_strata,
    },
    "crossfit": {
        "weights": weight_rows,
        "reference_gain_vs_v9_raw": reference_gain,
        "positive_reference_folds": positive_reference_folds,
        "pooled_blend_gain_vs_sp45": blend_gain,
        "positive_blend_folds": positive_blend_folds,
    },
    "references_not_gates": {
        "author_backbone_773_pooled": 9.99,
        "v11_fold5_gain": v11["fold5"]["gain"],
    },
    "gates": {
        "reference_gain_ge_0p20": reference_gain >= 0.20,
        "positive_reference_folds_ge_4": positive_reference_folds >= 4,
        "crossfit_blend_gain_ge_0p20": blend_gain >= 0.20,
        "positive_blend_folds_ge_4": positive_blend_folds >= 4,
        "all_weights_ge_0p10": bool((weight_frame["weight_stride"] >= 0.10).all()),
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "artifacts": {
        "fold_dir": str(FOLD_DIR),
        "weights": str(OUT_DIR / "crossfit_weights.csv"),
        "fold_metrics": str(OUT_DIR / "fold_metrics.csv"),
    },
}
verdict["strict_pass"] = bool(all(verdict["gates"].values()))
verdict["next"] = (
    "PASS -> fit the local-reference STRIDE deployment prior on all 773 wells and generate the test endpoint"
    if verdict["strict_pass"]
    else "FAIL -> local reference helps Fold5 but does not produce a stable five-fold complement"
)
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV12 CROSSFIT WEIGHTS")
print(weight_frame.to_string(index=False), flush=True)
print("\nV12 OVERALL")
print(pd.DataFrame(overall).T.to_string(), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
