"""Apply the frozen CP12-free CP8 bundle to the notebook's 6.39 lineage."""

from pathlib import Path
import hashlib
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


WORK = Path("/kaggle/working")
DATA = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
if "CFG" in globals():
    DATA = Path(getattr(CFG, "DATA", getattr(CFG, "dataset_path", DATA)))


def aligned_csv(path, name, sample):
    path = Path(path)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if not {"id", "tvt"}.issubset(frame.columns):
        raise RuntimeError(f"{name}: malformed {path}")
    frame = frame[["id", "tvt"]].rename(columns={"tvt": name})
    frame["id"] = frame["id"].astype(str)
    out = sample[["id"]].merge(frame, on="id", how="left", validate="one_to_one")
    values = pd.to_numeric(out[name], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name}: incomplete alignment from {path}")
    out[name] = values
    print("loaded", name, path, flush=True)
    return out


def aligned_global(obj, name, sample):
    if not isinstance(obj, pd.DataFrame) or not {"id", "tvt"}.issubset(obj.columns):
        return None
    frame = obj[["id", "tvt"]].rename(columns={"tvt": name}).copy()
    frame["id"] = frame["id"].astype(str)
    out = sample[["id"]].merge(frame, on="id", how="left", validate="one_to_one")
    values = pd.to_numeric(out[name], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        return None
    out[name] = values
    return out


def first(name, sample, globals_to_try=(), paths=()):
    for key in globals_to_try:
        frame = aligned_global(globals().get(key), name, sample)
        if frame is not None:
            print("loaded", name, "global", key, flush=True)
            return frame
    for path in paths:
        frame = aligned_csv(path, name, sample)
        if frame is not None:
            return frame
    raise FileNotFoundError(f"Cannot locate {name}; tried {list(paths)}")


def safe_stats(values, prefix):
    values = np.asarray(values, float)
    x = values[np.isfinite(values)]
    if not len(x):
        return {f"{prefix}_{s}": 0.0 for s in ("mean", "std", "p10", "p50", "p90", "span")}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_p10": float(np.quantile(x, 0.10)),
        f"{prefix}_p50": float(np.quantile(x, 0.50)),
        f"{prefix}_p90": float(np.quantile(x, 0.90)),
        f"{prefix}_span": float(np.max(x) - np.min(x)),
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hw(well):
    path = DATA / "test" / f"{well}__horizontal_well.csv"
    hw = pd.read_csv(path)
    for column in ("MD", "X", "Y", "Z", "GR", "TVT_input"):
        if column not in hw:
            hw[column] = np.nan
    md = hw["MD"].to_numpy(float)
    x = hw["X"].to_numpy(float)
    y = hw["Y"].to_numpy(float)
    z = hw["Z"].to_numpy(float)
    gr = hw["GR"].to_numpy(float)
    tvt = hw["TVT_input"].to_numpy(float)
    dmd = np.where(np.abs(np.gradient(md)) < 1e-6, np.nan, np.gradient(md))
    build = np.gradient(z) / dmd
    dx, dy = np.gradient(x) / dmd, np.gradient(y) / dmd
    dogleg = np.sqrt(
        np.nan_to_num(np.gradient(dx), nan=0.0) ** 2
        + np.nan_to_num(np.gradient(dy), nan=0.0) ** 2
        + np.nan_to_num(np.gradient(build), nan=0.0) ** 2
    )
    gr_fill = pd.Series(gr).interpolate(limit_direction="both").fillna(0).to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt))
    last_known = int(known[-1]) if len(known) else 0
    n = len(hw)
    return {
        "n_total": n,
        "MD": md, "X": x, "Y": y, "Z": z, "GR": gr,
        "build": build, "dogleg": dogleg,
        "gr_d1": np.gradient(gr_fill), "gr_d2": np.gradient(np.gradient(gr_fill)),
        "pos": np.arange(n, dtype=float) / max(n - 1, 1),
        "since_anchor_norm": (np.arange(n, dtype=float) - last_known) / max(n - last_known - 1, 1),
        "prefix_n": int(len(known)),
    }


started = time.time()
placement = os.environ.get("ROGII_CP8_PLACEMENT", "before_overlays").strip().lower()
if placement not in {"before_overlays", "last_increment"}:
    raise ValueError(
        "ROGII_CP8_PLACEMENT must be 'before_overlays' or 'last_increment', "
        f"got {placement!r}"
    )
bundle_candidates = list(Path("/kaggle/input").rglob("frontier_6p39_cp8_nocp12_bundle_v9.joblib"))
if not bundle_candidates:
    raise FileNotFoundError("Upload the no-CP12 v9 deployment bundle as a Kaggle dataset")
bundle_path = bundle_candidates[0]
bundle = joblib.load(bundle_path)
if bundle.get("version") != "frontier_6p39_cp8_nocp12_w192_huber_v9":
    raise RuntimeError(f"Wrong bundle version: {bundle.get('version')}")
if bundle.get("reference_col") is not None or "cp12" in bundle.get("aux_cols", []):
    raise RuntimeError("This bundle is not CP12-free")

sample = pd.read_csv(DATA / "sample_submission.csv")
sample["id"] = sample["id"].astype(str)
sample["well"] = sample["id"].str.rsplit("_", n=1).str[0]
sample["row_idx"] = sample["id"].str.rsplit("_", n=1).str[1].astype(np.int32)

# The core is always required to construct CP8 features and its additive move.
weight = float(globals().get("SP45_BLEND_WEIGHT", 0.60))
named_parent = WORK / f"submission_sp45_learned_w{weight:.2f}.csv"
if not named_parent.exists():
    raise FileNotFoundError(
        f"Missing {named_parent}. Run the original notebook through the 6.39 core "
        "blend cell before applying CP8."
    )
parent_source = named_parent
sources = {
    "frontier_6p39_core": aligned_csv(parent_source, "frontier_6p39_core", sample),
    "ridge": first("ridge", sample, globals_to_try=("sub_1",), paths=(WORK / "ridge_pp_submission.csv",)),
    "selector": first("selector", sample, globals_to_try=("sub_2",), paths=(WORK / "selector_submission.csv",)),
    "fleongg": first(
        "fleongg", sample,
        paths=(WORK / "learned_trajectory_submission.csv", WORK / "fleongg_pretrained_submission.csv"),
    ),
    "sp45": first("sp45", sample, paths=(WORK / "sp45_projection_submission.csv",)),
}
if placement == "last_increment":
    full_submission_path = WORK / "submission.csv"
    sources["full_before_cp8"] = aligned_csv(
        full_submission_path, "full_before_cp8", sample
    )
    if sources["full_before_cp8"] is None:
        raise FileNotFoundError(
            "last_increment requires /kaggle/working/submission.csv from the fully "
            "completed original notebook"
        )
if sources["frontier_6p39_core"] is None:
    raise RuntimeError("6.39 parent is unavailable")

test = sample[["id", "well", "row_idx"]].copy()
for name, frame in sources.items():
    test = test.merge(frame[["id", name]], on="id", how="left", validate="one_to_one")
test = test.sort_values(["well", "row_idx"]).reset_index(drop=True)

models = list(bundle["models"])
feature_cols = list(bundle["feature_cols"])
aux_cols = list(bundle["aux_cols"])
if aux_cols != ["ridge", "selector", "fleongg", "sp45"]:
    raise RuntimeError(f"Unexpected no-CP12 candidates: {aux_cols}")
base = test["frontier_6p39_core"].to_numpy(np.float32)
candidates = {name: test[name].to_numpy(np.float32) for name in aux_cols}
records, caches = [], {}

groups = list(test.groupby("well", sort=False).indices.items())
for number, (well, index) in enumerate(groups, start=1):
    index = np.asarray(index, np.int64)
    n = len(index)
    if n < 32:
        raise RuntimeError(f"{well}: only {n} scored rows")
    hwf = load_hw(well)
    ridx = test["row_idx"].to_numpy(np.int32)[index]
    if ridx.min() < 0 or ridx.max() >= hwf["n_total"]:
        raise RuntimeError(f"{well}: row index outside source CSV")
    centers = np.unique(np.round(np.linspace(0, n - 1, int(bundle["n_cp"]))).astype(int))
    caches[well] = (index, n)
    for order, center_local in enumerate(centers):
        lo = max(0, center_local - int(bundle["win_rows"]))
        hi = min(n, center_local + int(bundle["win_rows"]) + 1)
        local = np.arange(lo, hi)
        loc, ridx_win = index[local], ridx[local]
        center_global, center_ridx = int(index[center_local]), int(ridx[center_local])
        base_window = base[loc]
        feat = {
            "well": well,
            "center_global": center_global,
            "center_local": int(center_local),
            "n_well": int(n), "n_window": int(len(loc)), "cp_order": int(order),
            "cp_order_norm": float(order / max(len(centers) - 1, 1)),
            "row_norm": float(center_local / max(n - 1, 1)),
            "prefix_n": hwf["prefix_n"],
            "center_GR": float(np.nan_to_num(hwf["GR"][center_ridx], nan=0.0)),
            "center_Z": float(np.nan_to_num(hwf["Z"][center_ridx], nan=0.0)),
            "center_build": float(np.nan_to_num(hwf["build"][center_ridx], nan=0.0)),
            "center_since_anchor_norm": float(np.nan_to_num(hwf["since_anchor_norm"][center_ridx], nan=0.0)),
            "base_center": float(base[center_global]),
            "base_slope_win": float((base_window[-1] - base_window[0]) / max(len(base_window) - 1, 1)),
        }
        for column in ("MD", "X", "Y", "Z", "GR", "build", "dogleg", "gr_d1", "gr_d2", "pos", "since_anchor_norm"):
            feat.update(safe_stats(hwf[column][ridx_win], column))
        feat.update(safe_stats(base_window, "base"))
        for name in aux_cols:
            diff = candidates[name][loc] - base_window
            feat.update(safe_stats(diff, f"{name}_minus_base"))
            feat[f"{name}_minus_base_center"] = float(candidates[name][center_global] - base[center_global])
            feat[f"{name}_minus_base_slope"] = float((diff[-1] - diff[0]) / max(len(diff) - 1, 1))
        records.append(feat)
    if number % 25 == 0 or number == len(groups):
        print(f"NOCP12 features {number}/{len(groups)}", flush=True)

control = pd.DataFrame(records)
missing = [column for column in feature_cols if column not in control]
if missing:
    raise RuntimeError(f"Feature mismatch: {missing[:20]}")
X_test = control[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
control["pred_resid"] = np.mean(
    np.column_stack([model.predict(X_test) for model in models]), axis=1
).astype(np.float32)

raw_move = np.zeros(len(test), np.float32)
for well, (index, n) in caches.items():
    part = control[control["well"].eq(well)].sort_values("center_local")
    xp = part["center_local"].to_numpy(float)
    fp = part["pred_resid"].to_numpy(float)
    keep = np.r_[True, np.diff(xp) > 0]
    xp, fp = xp[keep], fp[keep]
    move = np.full(n, fp[0]) if len(xp) == 1 else np.interp(np.arange(n), xp, fp)
    raw_move[index] = gaussian_filter1d(
        move, sigma=max(4.0, n / int(bundle["n_cp"]) / 8.0), mode="nearest"
    ).astype(np.float32)

move = np.clip(float(bundle["cp_alpha"]) * raw_move, -float(bundle["cp_cap"]), float(bundle["cp_cap"]))
cp8_prediction = base + move
if not np.isfinite(cp8_prediction).all():
    raise RuntimeError("Non-finite no-CP12 predictions")

checkpoint_path = WORK / "submission_frontier_6p39_cp8_nocp12_before_overlays.csv"
checkpoint = test[["id"]].copy()
checkpoint["tvt"] = cp8_prediction.astype(float)
checkpoint.to_csv(checkpoint_path, index=False)

full_before_path = None
if placement == "last_increment":
    full_before = test["full_before_cp8"].to_numpy(float)
    if not np.isfinite(full_before).all():
        raise RuntimeError("Non-finite completed-notebook predictions")
    prediction = full_before + move
    full_before_path = WORK / "submission_frontier_6p39_full_before_cp8_last.csv"
    full_before_frame = test[["id"]].copy()
    full_before_frame["tvt"] = full_before
    full_before_frame.to_csv(full_before_path, index=False)
    decomposition_error = float(np.max(np.abs((prediction - full_before) - move)))
    if decomposition_error > 1e-9:
        raise RuntimeError(f"CP-last additive decomposition failed: {decomposition_error}")
    named_output = WORK / "submission_frontier_6p39_cp8_nocp12_last_increment.csv"
else:
    prediction = cp8_prediction
    decomposition_error = 0.0
    named_output = checkpoint_path

if not np.isfinite(prediction).all():
    raise RuntimeError("Non-finite final predictions")
submission = test[["id"]].copy()
submission["tvt"] = prediction.astype(float)
submission.to_csv(named_output, index=False)
submission.to_csv(WORK / "submission.csv", index=False)

summary = {
    "bundle": str(bundle_path),
    "bundle_sha256": file_sha256(bundle_path),
    "placement": placement,
    "parent_source": str(parent_source),
    "parent_sha256": file_sha256(parent_source),
    "cp_checkpoint": str(checkpoint_path),
    "cp_checkpoint_sha256": file_sha256(checkpoint_path),
    "full_before_cp8": None if full_before_path is None else str(full_before_path),
    "full_before_cp8_sha256": None if full_before_path is None else file_sha256(full_before_path),
    "output": str(named_output),
    "output_sha256": file_sha256(named_output),
    "rows": len(submission),
    "wells": test["well"].nunique(),
    "mean_abs_move": float(np.mean(np.abs(move))),
    "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
    "max_abs_move": float(np.max(np.abs(move))),
    "additive_decomposition_max_error": decomposition_error,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "FINAL WRITER: submit submission.csv and do not run any later submission "
        "writer"
        if placement == "last_increment"
        else "continue the original notebook overlays, then use last_increment mode"
    ),
}
(WORK / "frontier_6p39_cp8_nocp12_deployment.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("NOCP12 SUBMISSION READY", json.dumps(summary, indent=2), flush=True)
