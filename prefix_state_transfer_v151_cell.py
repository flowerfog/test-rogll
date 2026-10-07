"""V151: prefix-state transfer for the deployed V76 endpoint.

This branch does not use Neighbor, CP, candidate ranking, or future truth as
features. It predicts a low-dimensional tail correction from the observed
prefix state and applies it only after 1000 ft.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor

STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "prefix_state_transfer_v151"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V150 = ROOT / "prefix_state_residual_refiner_v150_cell.py"
if not V150.is_file():
    V150 = Path.cwd() / "prefix_state_residual_refiner_v150_cell.py"
if not V150.is_file():
    raise FileNotFoundError(V150)

# Load only V150 definitions and data paths. Its training loop is below the
# first frame construction and is intentionally not executed.
source = V150.read_text(encoding="utf-8")
marker = "frame = pd.read_parquet(BASE_PATH)"
if marker not in source:
    raise RuntimeError("V150 bootstrap marker missing")
module = types.ModuleType("v151_v150_defs")
module.__file__ = str(V150)
sys.modules[module.__name__] = module
exec(compile(source.split(marker, 1)[0] + "\n", str(V150), "exec"), module.__dict__)

BASE_PATH = module._first_existing(module.BASE_PATH_CANDIDATES)
FOLD_PATH = module._first_existing(module.FOLD_PATH_CANDIDATES)
frame = pd.read_parquet(BASE_PATH)
base_col = module.BASE_COLUMN if module.BASE_COLUMN in frame.columns else module.AUX_COLUMN
if base_col not in frame.columns:
    raise RuntimeError(f"Missing V76 baseline in {BASE_PATH}")
frame = frame[["well", "row_idx", "fold", module.TRUTH, "md_since", base_col]].copy()
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
folds = pd.read_csv(FOLD_PATH, dtype={"well": str})
folds["well"] = folds["well"].astype(str)
fold_map = dict(zip(folds["well"], folds["fold"].astype(int)))
well_order = sorted(frame["well"].unique())
if len(well_order) != 773:
    raise RuntimeError(f"Expected 773 wells, got {len(well_order)}")

FREEZE = 1000.0
RAMP = 500.0
CAP = 6.0


def taper(distance):
    x = np.clip((distance - FREEZE) / RAMP, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def basis(distance):
    d = np.asarray(distance, float)
    w = taper(d)
    z = (d - 1500.0) / 1000.0
    return np.column_stack([w, w * z, w * np.maximum(z, 0.0), w * np.maximum(z - 0.75, 0.0)])


def coeff_target(local):
    distance = local["md_since"].to_numpy(float)
    residual = local[module.TRUTH].to_numpy(float) - local[base_col].to_numpy(float)
    design = basis(distance)
    keep = np.isfinite(distance) & np.isfinite(residual)
    design = design[keep]
    residual = residual[keep]
    gram = design.T @ design + 0.05 * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ residual)


def _window_stats(md, values, anchor_md, width):
    keep = np.isfinite(md) & np.isfinite(values) & (md >= anchor_md - width) & (md <= anchor_md)
    x = md[keep]
    y = values[keep]
    if len(x) < 8 or np.ptp(x) < 16.0:
        return [0.0] * 6
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    centered = x - x[-1]
    linear = np.polyfit(centered, y, 1)
    quadratic = np.polyfit(centered / max(width, 1.0), y, 2)
    residual = y - np.polyval(linear, centered)
    return [
        float(linear[0]),
        float(quadratic[0]),
        float(np.std(y)),
        float(np.std(residual)),
        float(np.mean(np.abs(np.diff(y)))),
        float(y[-1] - y[0]),
    ]


def prefix_features(well, local):
    horizontal_path = module.DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = module.DATA_ROOT / "train" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    required = {"MD", "X", "Y", "Z", "GR", "TVT_input"}
    if not required.issubset(horizontal.columns):
        raise RuntimeError(f"{well}: missing raw prefix columns")
    raw = horizontal[list(required)].apply(pd.to_numeric, errors="coerce")
    md = raw["MD"].to_numpy(float)
    tvt = raw["TVT_input"].to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt) & np.isfinite(md))
    if len(known) < 30:
        raise RuntimeError(f"{well}: too few visible TVT rows")
    anchor = int(known[-1])
    anchor_md = float(md[anchor])

    values = []
    for width in (128.0, 256.0, 512.0):
        values.extend(_window_stats(md, tvt, anchor_md, width))
        values.extend(_window_stats(md, raw["GR"].to_numpy(float), anchor_md, width))
        values.extend(_window_stats(md, raw["Z"].to_numpy(float), anchor_md, width))

    # The typewell comparison is evaluated only at visible TVT_input values.
    typewell = pd.read_csv(typewell_path)[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
    typewell = typewell.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    visible = known[max(0, len(known) - 512):]
    observed_gr = raw["GR"].to_numpy(float)[visible]
    expected_gr = np.interp(
        tvt[visible],
        typewell["TVT"].to_numpy(float),
        typewell["GR"].to_numpy(float),
        left=np.nan,
        right=np.nan,
    )
    valid = np.isfinite(observed_gr) & np.isfinite(expected_gr)
    if valid.sum() >= 16 and np.std(observed_gr[valid]) > 1e-6 and np.std(expected_gr[valid]) > 1e-6:
        correlation = float(np.corrcoef(observed_gr[valid], expected_gr[valid])[0, 1])
        mismatch = observed_gr[valid] - expected_gr[valid]
        values.extend([correlation, float(np.mean(mismatch)), float(np.std(mismatch))])
    else:
        values.extend([0.0, 0.0, 0.0])

    distance = local["md_since"].to_numpy(float)
    baseline = local[base_col].to_numpy(float)
    values.extend([
        float(tvt[anchor]) / 10000.0,
        float(raw["Z"].iloc[anchor]) / 10000.0,
        float(raw["X"].iloc[anchor]) / 1e6,
        float(raw["Y"].iloc[anchor]) / 1e6,
        float(np.nanmean(baseline)) / 10000.0,
        float(np.nanstd(baseline)) / 1000.0,
        float(np.nanmax(distance)) / 5000.0,
        float(np.nanmean(distance > 2000.0)),
    ])
    result = np.nan_to_num(np.asarray(values, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -100.0, 100.0)


X, Y = [], []
for well, local in frame.groupby("well", sort=True):
    X.append(prefix_features(well, local))
    Y.append(coeff_target(local))
X = np.stack(X).astype(np.float32)
Y = np.stack(Y).astype(np.float32)


def reconstruct(wells, coeffs, label):
    parts = []
    for well, coeff in zip(wells, coeffs):
        local = frame[frame["well"].eq(well)].copy()
        d = local["md_since"].to_numpy(float)
        base = local[base_col].to_numpy(float)
        truth = local[module.TRUTH].to_numpy(float)
        move = np.clip(basis(d) @ np.asarray(coeff, float), -CAP, CAP)
        pred = base + move
        parts.append(pd.DataFrame({"well": well, "distance": d, "truth": truth, "prediction": pred, "endpoint": label}))
    return pd.concat(parts, ignore_index=True)


def metrics(out):
    err = out["prediction"].to_numpy(float) - out["truth"].to_numpy(float)
    d = out["distance"].to_numpy(float)
    return {
        "endpoint": str(out["endpoint"].iloc[0]),
        "rows": int(len(out)),
        "wells": int(out["well"].nunique()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(err[d > 2000.0] ** 2))),
        "near_degradation": float(np.sqrt(np.mean(err[d <= 1000.0] ** 2))),
    }


all_parts = {name: [] for name in ("baseline", "learned", "oracle", "shuffled", "opposite")}
fold_rows = []
for heldout in range(1, 6):
    train_mask = np.asarray([fold_map[w] != heldout for w in well_order])
    valid_mask = ~train_mask
    model = MultiOutputRegressor(
        ExtraTreesRegressor(n_estimators=256, min_samples_leaf=10, max_features=0.8, random_state=module.SEED + heldout, n_jobs=-1)
    )
    model.fit(X[train_mask], Y[train_mask])
    pred_coeff = model.predict(X[valid_mask])
    valid_wells = [w for w, keep in zip(well_order, valid_mask) if keep]
    all_parts["learned"].append(reconstruct(valid_wells, pred_coeff, "learned"))
    all_parts["oracle"].append(reconstruct(valid_wells, Y[valid_mask], "oracle"))
    all_parts["baseline"].append(reconstruct(valid_wells, np.zeros_like(pred_coeff), "baseline"))
    rng = np.random.default_rng(module.SEED + heldout)
    all_parts["shuffled"].append(reconstruct(valid_wells, pred_coeff[rng.permutation(len(pred_coeff))], "shuffled"))
    all_parts["opposite"].append(reconstruct(valid_wells, -pred_coeff, "opposite"))
    fold_rows.append({"fold": heldout, **{name: metrics(all_parts[name][-1])["rmse"] for name in all_parts}})

outputs = {name: pd.concat(parts, ignore_index=True) for name, parts in all_parts.items()}
summary = pd.DataFrame([metrics(out) for out in outputs.values()])
summary.to_csv(OUT_DIR / "summary.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT_DIR / "fold_summary.csv", index=False)
for name, out in outputs.items():
    out.to_parquet(OUT_DIR / f"{name}_oof.parquet", index=False)

base = summary.set_index("endpoint").loc["baseline"]
learned = summary.set_index("endpoint").loc["learned"]
oracle = summary.set_index("endpoint").loc["oracle"]
shuffled = summary.set_index("endpoint").loc["shuffled"]
verdict = {
    "experiment": "prefix_state_transfer_v151",
    "hypothesis": "prefix-derived local dip/curvature state predicts the sign and size of V76 tail residual",
    "features": int(X.shape[1]),
    "folds": {str(f): int((folds["fold"] == f).sum()) for f in range(1, 6)},
    "summary": summary.to_dict(orient="records"),
    "learned_gain": float(base["rmse"] - learned["rmse"]),
    "oracle_gain": float(base["rmse"] - oracle["rmse"]),
    "shuffled_gap": float(shuffled["rmse"] - learned["rmse"]),
    "far_gain": float(base["far_gt2000_rmse"] - learned["far_gt2000_rmse"]),
    "checks": {
        "oracle_gain_ge_0p20": bool(base["rmse"] - oracle["rmse"] >= 0.20),
        "learned_gain_ge_0p05": bool(base["rmse"] - learned["rmse"] >= 0.05),
        "shuffled_gap_ge_0p02": bool(shuffled["rmse"] - learned["rmse"] >= 0.02),
        "far_gain_nonnegative": bool(base["far_gt2000_rmse"] - learned["far_gt2000_rmse"] >= 0.0),
    },
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print(summary.to_string(index=False))
print(json.dumps(verdict, indent=2))
