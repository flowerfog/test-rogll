"""Deploy the frozen CP8 residual head on the 6.39 notebook core.

Placement is strict: run after the notebook's cell that writes the 0.60 SP45
+ 0.40 learned/FLE core (cell 39 in the supplied notebook), after producing
the CP12 side curve, and before guarded same-well / visible-prefix overlays.
The immutable weighted-blend file from cell 39 is preferred over the mutable
``submission.csv``, because a CP12 helper may overwrite the latter.
"""

from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
DATA = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
if "CFG" in globals():
    DATA = Path(getattr(CFG, "DATA", getattr(CFG, "dataset_path", DATA)))


def _find_bundle():
    direct = [WORK / "frontier_6p39_cp8_bundle_v7.joblib"]
    found = direct + list(Path("/kaggle/input").rglob("frontier_6p39_cp8_bundle_v7.joblib"))
    for path in found:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Upload the AutoDL deployment directory containing "
        "frontier_6p39_cp8_bundle_v7.joblib"
    )


def _aligned_csv(path, name, sample):
    path = Path(path)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if not {"id", "tvt"}.issubset(frame.columns):
        raise RuntimeError(f"{name}: malformed submission file {path}")
    frame = frame[["id", "tvt"]].rename(columns={"tvt": name})
    frame["id"] = frame["id"].astype(str)
    if frame["id"].duplicated().any():
        raise RuntimeError(f"{name}: duplicate ids in {path}")
    out = sample[["id"]].merge(frame, on="id", how="left", validate="one_to_one")
    values = pd.to_numeric(out[name], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name}: incomplete/non-finite alignment from {path}")
    out[name] = values
    print(f"loaded {name}: {path}", flush=True)
    return out


def _aligned_global(obj, name, sample):
    if not isinstance(obj, pd.DataFrame) or not {"id", "tvt"}.issubset(obj.columns):
        return None
    frame = obj[["id", "tvt"]].rename(columns={"tvt": name}).copy()
    frame["id"] = frame["id"].astype(str)
    out = sample[["id"]].merge(frame, on="id", how="left", validate="one_to_one")
    values = pd.to_numeric(out[name], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        return None
    out[name] = values
    print(f"loaded {name}: notebook global", flush=True)
    return out


def _first(name, sample, globals_to_try=(), paths=()):
    for global_name in globals_to_try:
        frame = _aligned_global(globals().get(global_name), name, sample)
        if frame is not None:
            return frame
    for path in paths:
        frame = _aligned_csv(path, name, sample)
        if frame is not None:
            return frame
    raise FileNotFoundError(f"Cannot locate deployment candidate {name!r}; tried {list(paths)}")


def _safe_stats(values, prefix):
    values = np.asarray(values, float)
    x = values[np.isfinite(values)]
    if not len(x):
        return {f"{prefix}_{suffix}": 0.0 for suffix in ("mean", "std", "p10", "p50", "p90", "span")}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_p10": float(np.quantile(x, 0.10)),
        f"{prefix}_p50": float(np.quantile(x, 0.50)),
        f"{prefix}_p90": float(np.quantile(x, 0.90)),
        f"{prefix}_span": float(np.max(x) - np.min(x)),
    }


def _load_hw_features(well):
    path = DATA / "test" / f"{well}__horizontal_well.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    hw = pd.read_csv(path)
    for column in ("MD", "X", "Y", "Z", "GR", "TVT_input"):
        if column not in hw:
            hw[column] = np.nan
    md = hw["MD"].to_numpy(float)
    x = hw["X"].to_numpy(float)
    y = hw["Y"].to_numpy(float)
    z = hw["Z"].to_numpy(float)
    gr = hw["GR"].to_numpy(float)
    tvt_input = hw["TVT_input"].to_numpy(float)
    dmd = np.gradient(md)
    dmd = np.where(np.abs(dmd) < 1e-6, np.nan, dmd)
    build = np.gradient(z) / dmd
    dx = np.gradient(x) / dmd
    dy = np.gradient(y) / dmd
    dogleg = np.sqrt(
        np.nan_to_num(np.gradient(dx), nan=0.0) ** 2
        + np.nan_to_num(np.gradient(dy), nan=0.0) ** 2
        + np.nan_to_num(np.gradient(build), nan=0.0) ** 2
    )
    gr_fill = pd.Series(gr).interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
    gr_d1 = np.gradient(gr_fill)
    gr_d2 = np.gradient(gr_d1)
    known = np.flatnonzero(np.isfinite(tvt_input))
    last_known = int(known[-1]) if len(known) else 0
    n = len(hw)
    return {
        "n_total": n,
        "MD": md, "X": x, "Y": y, "Z": z, "GR": gr,
        "build": build, "dogleg": dogleg, "gr_d1": gr_d1, "gr_d2": gr_d2,
        "pos": np.arange(n, dtype=float) / max(n - 1, 1),
        "since_anchor_norm": (np.arange(n, dtype=float) - last_known) / max(n - last_known - 1, 1),
        "prefix_n": int(len(known)),
    }


started = time.time()
bundle_path = _find_bundle()
bundle = joblib.load(bundle_path)
if bundle.get("version") != "frontier_6p39_cp8_w192_huber_v7":
    raise RuntimeError(f"Wrong CP8 bundle version: {bundle.get('version')!r}")
if bundle.get("base_col") != "frontier_6p39_core" or bundle.get("reference_col") != "cp12":
    raise RuntimeError("CP8 bundle has the wrong parent/reference bloodline")

models = list(bundle["models"])
feature_cols = list(bundle["feature_cols"])
aux_cols = list(bundle["aux_cols"])
n_cp = int(bundle["n_cp"])
win_rows = int(bundle["win_rows"])
cp_alpha = float(bundle["cp_alpha"])
cp_cap = float(bundle["cp_cap"])

sample = pd.read_csv(DATA / "sample_submission.csv")
sample["id"] = sample["id"].astype(str)
sample["well"] = sample["id"].str.rsplit("_", n=1).str[0]
sample["row_idx"] = sample["id"].str.rsplit("_", n=1).str[1].astype(np.int32)

# Cell 39's output is the exact parent. Recover its immutable named output even
# if a CP12 helper has since overwritten submission.csv.
parent_path = WORK / "frontier_6p39_core_before_cp.csv"
if not parent_path.exists():
    selected_weight = float(globals().get("SP45_BLEND_WEIGHT", 0.60))
    named_parent = WORK / f"submission_sp45_learned_w{selected_weight:.2f}.csv"
    if named_parent.exists():
        source_parent = named_parent
    elif "_final_name" in globals() and (WORK / str(globals()["_final_name"])).exists():
        source_parent = WORK / str(globals()["_final_name"])
    else:
        source_parent = WORK / "submission.csv"
        print(
            "WARNING: immutable 6.39 blend file not found; using current submission.csv. "
            "This is valid only if CP12 has not overwritten it.",
            flush=True,
        )
    current = _aligned_csv(source_parent, "frontier_6p39_core", sample)
    current.rename(columns={"frontier_6p39_core": "tvt"}).to_csv(parent_path, index=False)
    print("snapshotted 6.39 core:", source_parent, "->", parent_path, flush=True)

sources = {
    "frontier_6p39_core": _aligned_csv(parent_path, "frontier_6p39_core", sample),
    "cp12": _first(
        "cp12", sample,
        paths=(
            WORK / "submission_controlpoint_before_gold.csv",
            WORK / "submission_cp12_before_gold.csv",
        ),
    ),
    "ridge": _first("ridge", sample, globals_to_try=("sub_1",), paths=(WORK / "ridge_pp_submission.csv",)),
    "selector": _first("selector", sample, globals_to_try=("sub_2",), paths=(WORK / "selector_submission.csv",)),
    "fleongg": _first(
        "fleongg", sample,
        paths=(WORK / "learned_trajectory_submission.csv", WORK / "fleongg_pretrained_submission.csv"),
    ),
    "sp45": _first("sp45", sample, paths=(WORK / "sp45_projection_submission.csv",)),
}

missing_aux = [column for column in aux_cols if column not in sources]
if missing_aux:
    raise RuntimeError(f"Bundle requires unavailable candidate curves: {missing_aux}")

test = sample[["id", "well", "row_idx"]].copy()
for name, frame in sources.items():
    test = test.merge(frame[["id", name]], on="id", how="left", validate="one_to_one")
test = test.sort_values(["well", "row_idx"]).reset_index(drop=True)

base = test["frontier_6p39_core"].to_numpy(np.float32)
candidates = {column: test[column].to_numpy(np.float32) for column in aux_cols}
row_indices = test["row_idx"].to_numpy(np.int32)
records = []
caches = {}
groups = list(test.groupby("well", sort=False).indices.items())

for number, (well, idx) in enumerate(groups, start=1):
    idx = np.asarray(idx, dtype=np.int64)
    n = len(idx)
    if n < 32:
        raise RuntimeError(f"{well}: only {n} scored rows")
    hwf = _load_hw_features(well)
    ridx = row_indices[idx]
    if ridx.min() < 0 or ridx.max() >= hwf["n_total"]:
        raise RuntimeError(f"{well}: scored row index outside horizontal-well CSV")
    centers = np.unique(np.round(np.linspace(0, n - 1, n_cp)).astype(int))
    row_norm = np.arange(n, dtype=float) / max(n - 1, 1)
    caches[well] = {"idx": idx, "n": n}

    for order, center_local in enumerate(centers):
        lo = max(0, center_local - win_rows)
        hi = min(n, center_local + win_rows + 1)
        local_positions = np.arange(lo, hi)
        loc = idx[local_positions]
        ridx_win = ridx[local_positions]
        center_global = int(idx[center_local])
        center_ridx = int(ridx[center_local])
        base_window = base[loc]
        feat = {
            "well": well,
            "center_global": center_global,
            "center_local": int(center_local),
            "n_well": int(n),
            "n_window": int(len(loc)),
            "cp_order": int(order),
            "cp_order_norm": float(order / max(len(centers) - 1, 1)),
            "row_norm": float(row_norm[center_local]),
            "prefix_n": hwf["prefix_n"],
            "center_GR": float(np.nan_to_num(hwf["GR"][center_ridx], nan=0.0)),
            "center_Z": float(np.nan_to_num(hwf["Z"][center_ridx], nan=0.0)),
            "center_build": float(np.nan_to_num(hwf["build"][center_ridx], nan=0.0)),
            "center_since_anchor_norm": float(np.nan_to_num(hwf["since_anchor_norm"][center_ridx], nan=0.0)),
            "base_center": float(base[center_global]),
            "base_slope_win": float((base_window[-1] - base_window[0]) / max(len(base_window) - 1, 1)),
        }
        for column in ("MD", "X", "Y", "Z", "GR", "build", "dogleg", "gr_d1", "gr_d2", "pos", "since_anchor_norm"):
            feat.update(_safe_stats(hwf[column][ridx_win], column))
        feat.update(_safe_stats(base_window, "base"))
        stack = np.column_stack([candidates[column][loc] for column in aux_cols])
        feat["pred_std_mean"] = float(np.mean(np.std(stack, axis=1)))
        feat["pred_range_mean"] = float(np.mean(np.max(stack, axis=1) - np.min(stack, axis=1)))
        for column in aux_cols:
            diff = candidates[column][loc] - base_window
            feat.update(_safe_stats(diff, f"{column}_minus_base"))
            feat[f"{column}_minus_base_center"] = float(candidates[column][center_global] - base[center_global])
            feat[f"{column}_minus_base_slope"] = float((diff[-1] - diff[0]) / max(len(diff) - 1, 1))
        records.append(feat)

    if number % 25 == 0 or number == len(groups):
        print(f"CP8 features {number}/{len(groups)}", flush=True)

control = pd.DataFrame(records)
missing_features = [column for column in feature_cols if column not in control]
if missing_features:
    raise RuntimeError(f"Deployment feature schema mismatch: {missing_features[:20]}")
x_test = control[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
member_predictions = np.column_stack([model.predict(x_test) for model in models])
control["pred_resid"] = np.mean(member_predictions, axis=1).astype(np.float32)

raw_move = np.zeros(len(test), dtype=np.float32)
for well, cache in caches.items():
    idx = cache["idx"]
    n = cache["n"]
    part = control[control["well"].eq(well)].sort_values("center_local")
    xp = part["center_local"].to_numpy(float)
    fp = part["pred_resid"].to_numpy(float)
    keep = np.r_[True, np.diff(xp) > 0]
    xp, fp = xp[keep], fp[keep]
    move = np.full(n, fp[0], dtype=float) if len(xp) == 1 else np.interp(np.arange(n, dtype=float), xp, fp)
    sigma = max(4.0, n / n_cp / 8.0)
    raw_move[idx] = gaussian_filter1d(move, sigma=sigma, mode="nearest").astype(np.float32)

move = np.clip(cp_alpha * raw_move, -cp_cap, cp_cap)
prediction = base + move
if not np.isfinite(prediction).all():
    raise RuntimeError("CP8 produced non-finite predictions")

out = test[["id"]].copy()
out["tvt"] = prediction.astype(float)
out.to_csv(WORK / "submission_frontier_6p39_cp8_before_overlays.csv", index=False)
out.to_csv(WORK / "submission.csv", index=False)

diagnostic = test[["id", "well", "row_idx", "frontier_6p39_core", "cp12"]].copy()
diagnostic["cp8_move"] = move
diagnostic["cp8_tvt"] = prediction
diagnostic.to_csv(WORK / "frontier_6p39_cp8_deployment_diagnostic.csv", index=False)

summary = {
    "bundle": str(bundle_path),
    "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
    "rows": int(len(out)),
    "wells": int(test["well"].nunique()),
    "mean_abs_move": float(np.mean(np.abs(move))),
    "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
    "max_abs_move": float(np.max(np.abs(move))),
    "cap_ft": cp_cap,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": "continue with the notebook guarded overlap and visible-prefix cells",
}
(WORK / "frontier_6p39_cp8_deployment.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("CP8 core submission ready:", json.dumps(summary, indent=2), flush=True)
