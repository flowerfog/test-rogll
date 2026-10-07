"""Kaggle tail cell for the frozen PSG feature correction on exact 6.377.

Run this only after the complete CP8 + Neighbor notebook has written its final
parent submission. The cell rebuilds the same full-stack features, computes the
PSG feature-model delta, applies dose 0.20, and writes submission.csv.
"""

from pathlib import Path
import glob
import hashlib
import json
import os
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


DOSE = 0.20
BUNDLE_OVERRIDE = None
PARENT_OVERRIDE = None
RAW_ONLY_FOR_G5 = True
EXPECTED_TEST_WELLS = 3
EXPECTED_ROWS = 14151
started = time.time()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_bundle():
    if BUNDLE_OVERRIDE:
        path = Path(BUNDLE_OVERRIDE)
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    hits = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if root.exists():
            hits.extend(root.rglob("psg_v66_delta_deployment_v5/deployment_config.json"))
            hits.extend(root.rglob("deployment_config.json"))
    for config_path in sorted(set(hits)):
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("experiment") == "psg_v66_delta_deployment_v5":
            return config_path.parent
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if not root.exists():
            continue
        for archive in root.rglob("psg_v66_delta_deployment_v5.zip"):
            extracted = (
                Path("/kaggle/working")
                if Path("/kaggle/working").is_dir()
                else Path.cwd()
            ) / "psg_v66_bundle_extracted"
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(extracted)
            for config_path in extracted.rglob("deployment_config.json"):
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                if payload.get("experiment") == "psg_v66_delta_deployment_v5":
                    return config_path.parent
    raise FileNotFoundError(
        "Cannot locate the PSG-V5 deployment bundle. Upload the AutoDL "
        "psg_v66_delta_deployment_v5 directory or its ZIP as a Kaggle dataset."
    )


def _find_data_root():
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ]
    cfg = globals().get("CFG")
    if cfg is not None:
        for attr in ("DATA", "dataset_path"):
            value = getattr(cfg, attr, None)
            if value:
                candidates.insert(0, Path(value))
    for root in candidates:
        if (root / "train").is_dir() and (root / "test").is_dir():
            return root
    raise FileNotFoundError("Cannot locate competition train/test directories")


def _normalise_submission(path, sample):
    frame = pd.read_csv(path)
    value = "tvt" if "tvt" in frame else next(
        (column for column in frame if column != "id"), None
    )
    if value is None or "id" not in frame or frame["id"].duplicated().any():
        raise RuntimeError(f"Invalid submission: {path}")
    frame = frame[["id", value]].rename(columns={value: "tvt"})
    frame["id"] = frame["id"].astype(str)
    if not frame["id"].equals(sample["id"]):
        raise RuntimeError(f"Submission ID order differs from sample: {path}")
    if not np.isfinite(frame["tvt"].to_numpy(np.float64)).all():
        raise RuntimeError(f"Non-finite parent predictions: {path}")
    return frame


def _find_parent(work, sample):
    candidates = []
    if PARENT_OVERRIDE:
        candidates.append(Path(PARENT_OVERRIDE))
    candidates.extend(
        [
            work / "submission_frontier_6p39_tail_neighbor.csv",
            work / "submission.csv",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path, _normalise_submission(path, sample)
    raise FileNotFoundError("Run the exact complete 6.377 CP8 + Neighbor pipeline first")


def _matrix(frame, columns):
    missing = [column for column in columns if column not in frame]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} model features: {missing[:30]}")
    return (
        frame.loc[:, columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _postprocess(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(np.float64)
    md_since = frame["md_since"].to_numpy(np.float64)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / 85.0)
    lik = frame["likpf_scale_5"].to_numpy(np.float64) - last
    prediction = last + 0.60 * warm * np.asarray(model_delta, np.float64) + 0.40 * lik
    output = prediction.copy()
    for _, positions in frame.groupby("well", sort=False).indices.items():
        positions = np.asarray(positions, np.int64)
        window = min(61, len(positions) if len(positions) % 2 else len(positions) - 1)
        if window >= 5:
            output[positions] = savgol_filter(
                prediction[positions], window_length=window, polyorder=3, mode="interp"
            )
    return output


required_globals = {
    "CFG", "init_imputers", "build_likpf", "build_features", "add_likpf_features"
}
missing_globals = sorted(required_globals.difference(globals()))
if missing_globals:
    raise RuntimeError(
        "Run the original mainline definition cells first; missing: "
        + repr(missing_globals)
    )

bundle = _find_bundle()
data_root = _find_data_root()
work = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
sample = pd.read_csv(data_root / "sample_submission.csv")
sample["id"] = sample["id"].astype(str)
parent_path, parent = _find_parent(work, sample)
parent_snapshot = work / "submission_6p377_before_psg_v5.csv"
parent.to_csv(parent_snapshot, index=False)

config = json.loads((bundle / "deployment_config.json").read_text(encoding="utf-8"))
if float(config.get("dose", -1.0)) != DOSE:
    raise RuntimeError(f"Bundle dose mismatch: {config}")

# Run only the raw-PSG inference prefix. The old script's direct PSG blend is
# deliberately excluded because V4 validated a downstream feature delta.
runtime_path = bundle / "runtime" / "psg_infer_cell.py"
runtime_source = runtime_path.read_text(encoding="utf-8")
checkpoint_paths = [
    str(bundle / "psg_models" / f"psg_fold{fold}.pt") for fold in range(1, 6)
]
checkpoint_line = 'CKPTS = _find("psg_oof_fold*.pt")'
device_line = 'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"'
cut_marker = 'WORK="/kaggle/working" if os.path.isdir("/kaggle/working") else "."'
if (
    runtime_source.count(checkpoint_line) != 1
    or runtime_source.count(device_line) != 1
    or runtime_source.count(cut_marker) != 1
):
    raise RuntimeError("Unknown psg_infer_cell.py layout; refusing an unsafe partial exec")
runtime_source = runtime_source.replace(checkpoint_line, f"CKPTS = {checkpoint_paths!r}")
# Kaggle's current torch wheel can report CUDA availability even when it has
# no kernel image for the attached GPU architecture. Three-well inference is
# small enough that CPU is the reproducible deployment target.
runtime_source = runtime_source.replace(device_line, 'DEVICE = "cpu"')
# The archived runtime defines predict(model, w, ps, F), but its inference
# loop came from an older three-argument version. Repair that exact known call.
predict_signature = "def predict(model,w,ps,F):"
stale_predict_call = "p,rows=predict(m,w,ps)"
if predict_signature in runtime_source:
    if runtime_source.count(stale_predict_call) != 1:
        raise RuntimeError("Unknown PSG predict-call layout; refusing an unsafe patch")
    runtime_source = runtime_source.replace(
        stale_predict_call, "p,rows=predict(m,w,ps,F)"
    )
runtime_prefix = runtime_source.split(cut_marker, 1)[0]
psg_namespace = {"__name__": "psg_v5_raw_runtime", "__file__": str(runtime_path)}
exec(compile(runtime_prefix, str(runtime_path), "exec"), psg_namespace)
psg_predictions = psg_namespace["PSG_PRED"]

train_wids = sorted(
    path.name.split("__", 1)[0]
    for path in (data_root / "train").glob("*__horizontal_well.csv")
)
test_wids = sorted(
    path.name.split("__", 1)[0]
    for path in (data_root / "test").glob("*__horizontal_well.csv")
)
if len(test_wids) != EXPECTED_TEST_WELLS:
    raise RuntimeError(f"Expected {EXPECTED_TEST_WELLS} test wells, got {len(test_wids)}")

CFG.DATA = data_root
try:
    CFG.dataset_path = data_root
except Exception:
    pass
init_imputers(train_wids)


def _likpf_grid_v5(tw_tvt, tw_gr, step=0.2):
    tw_tvt = np.asarray(tw_tvt, np.float64)
    tw_gr = np.asarray(tw_gr, np.float64)
    tmin = float(tw_tvt.min())
    tmax = float(tw_tvt.max())
    tvt_grid = np.arange(tmin, tmax + float(step), float(step))
    return (
        np.interp(tvt_grid, tw_tvt, tw_gr).astype(np.float64),
        tmin,
        float(step),
    )


# Later notebook cells use `_grid` for Matplotlib GridSpec and overwrite the
# helper resolved dynamically by lik_pf. Restore it only while lik-PF runs.
likpf_globals_v5 = build_likpf.__globals__
missing_grid_v5 = object()
previous_grid_v5 = likpf_globals_v5.get("_grid", missing_grid_v5)
likpf_globals_v5["_grid"] = _likpf_grid_v5
try:
    likpf_test = build_likpf(test_wids, "test")
    test = add_likpf_features(
        build_features(test_wids, "test", is_train=False), likpf_test
    ).reset_index(drop=True)
finally:
    if previous_grid_v5 is missing_grid_v5:
        likpf_globals_v5.pop("_grid", None)
    else:
        likpf_globals_v5["_grid"] = previous_grid_v5
if test["id"].duplicated().any():
    raise RuntimeError("Duplicate test feature IDs")

row_idx = test["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
psg_tvt = np.empty(len(test), np.float64)
for position, (well, row) in enumerate(zip(test["well"].astype(str), row_idx)):
    psg_tvt[position] = psg_predictions.get(well, {}).get(int(row), np.nan)
if not np.isfinite(psg_tvt).all():
    raise RuntimeError(f"Raw PSG coverage={np.isfinite(psg_tvt).mean():.6%}")

last = test["last_known_tvt"].to_numpy(np.float32)
pf = test["pf_ancc"].to_numpy(np.float32)
progress = np.sqrt(np.maximum(test["md_since"].to_numpy(np.float32), 0.0) + 1.0)
test["psgx_psg_delta_last"] = psg_tvt - last
test["psgx_psg_minus_pf"] = psg_tvt - pf
test["psgx_abs_psg_minus_pf"] = np.abs(psg_tvt - pf)
for suffix, column in (
    ("lik3", "likpf_scale_3"),
    ("lik5", "likpf_scale_5"),
    ("lik8", "likpf_scale_8"),
    ("lik12", "likpf_scale_12"),
):
    difference = psg_tvt - test[column].to_numpy(np.float32)
    test[f"psgx_psg_minus_{suffix}"] = difference
    test[f"psgx_abs_psg_minus_{suffix}"] = np.abs(difference)
test["psgx_psg_delta_x_sqrt_progress"] = (psg_tvt - last) * progress / 100.0

if RAW_ONLY_FOR_G5:
    audit = {
        "experiment": "psg_v66_raw_endpoint_for_g5",
        "mode": "raw_only_for_main342_g5",
        "bundle": str(bundle),
        "rows": int(len(test)),
        "wells": int(len(test_wids)),
        "finite_fraction": float(np.isfinite(psg_tvt).mean()),
        "submission_modified": False,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    (work / "psg_v66_raw_for_g5_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print("PSG-V5 RAW ENDPOINT READY FOR G5")
    print(json.dumps(audit, indent=2), flush=True)
else:
    base_features = pd.read_csv(bundle / "metadata" / "base_features.csv")["feature"].astype(str).tolist()
    psg_features = pd.read_csv(bundle / "metadata" / "psg_features.csv")["feature"].astype(str).tolist()
    all_features = base_features + psg_features
    base_x = _matrix(test, base_features)
    psg_x = _matrix(test, all_features)

    base_deltas = []
    psg_deltas = []
    for fold in range(1, 6):
        base_model = joblib.load(bundle / "lgb_models" / f"baseline_fold{fold}.joblib")
        psg_model = joblib.load(bundle / "lgb_models" / f"psg_features_fold{fold}.joblib")
        base_deltas.append(np.asarray(base_model.predict(base_x), np.float64))
        psg_deltas.append(np.asarray(psg_model.predict(psg_x), np.float64))

    baseline_endpoint = _postprocess(test, np.mean(base_deltas, axis=0))
    psg_endpoint = _postprocess(test, np.mean(psg_deltas, axis=0))
    correction = psg_endpoint - baseline_endpoint
    correction_by_id = dict(zip(test["id"].astype(str), correction))
    aligned_correction = sample["id"].map(correction_by_id).to_numpy(np.float64)
    if not np.isfinite(aligned_correction).all() or len(aligned_correction) != EXPECTED_ROWS:
        raise RuntimeError("PSG correction does not fully align to sample_submission")

    final = parent.copy()
    final["tvt"] = parent["tvt"].to_numpy(np.float64) + DOSE * aligned_correction
    candidate_path = work / "submission_6p377_psg_v5_w020.csv"
    final.to_csv(candidate_path, index=False)
    final.to_csv(work / "submission.csv", index=False)

    move = final["tvt"].to_numpy(np.float64) - parent["tvt"].to_numpy(np.float64)
    audit = {
        "experiment": "psg_v66_delta_kaggle_v5",
        "parent": str(parent_path),
        "parent_snapshot": str(parent_snapshot),
        "parent_sha256": _sha256(parent_snapshot),
        "bundle": str(bundle),
        "dose": DOSE,
        "rows": len(final),
        "wells": len(test_wids),
        "base_features": len(base_features),
        "psg_features": len(psg_features),
        "mean_abs_move_ft": float(np.mean(np.abs(move))),
        "rms_move_ft": float(np.sqrt(np.mean(move * move))),
        "p95_abs_move_ft": float(np.quantile(np.abs(move), 0.95)),
        "maximum_abs_move_ft": float(np.max(np.abs(move))),
        "candidate": str(candidate_path),
        "final": str(work / "submission.csv"),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    (work / "psg_v66_delta_kaggle_v5_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print("PSG-V5 KAGGLE TAIL COMPLETE")
    print(json.dumps(audit, indent=2), flush=True)
