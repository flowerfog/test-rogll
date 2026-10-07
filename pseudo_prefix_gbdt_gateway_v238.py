"""V238: legal pseudo-prefix training gateway.

Run after the notebook cells defining build_well and lik_pf.  It moves the
training anchor 512 ft earlier, exposes only the artificial prefix as
TVT_input, rebuilds features, and evaluates on untouched held-out wells using
the official V215 cache.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping


ROOT = Path("/root/autodl-tmp")
DATA = ROOT / "rogii-wellbore-geology-prediction" / "train"
V215 = ROOT / "oof_exports" / "likpf_residual_gbdt_full5_v215"
OUT = ROOT / "oof_exports" / "pseudo_prefix_gbdt_gateway_v238"
HELD_FOLD = 1
HELD_WELLS = int(os.environ.get("V238_HELD_WELLS", "48"))
PSEUDO_TRAIN_WELLS = int(os.environ.get("V238_PSEUDO_WELLS", "96"))
SHIFT_FT = 512.0
ROWS_PER_WELL = int(os.environ.get("V238_ROWS_PER_WELL", "512"))
SEED = 238


if "build_well" not in globals() or "lik_pf" not in globals():
    raise RuntimeError("Run the notebook function-definition cells first: missing build_well/lik_pf")


class _NeutralFormationImputer:
    def impute(self, xy_q, self_wid=None):
        n = len(np.atleast_2d(xy_q))
        count = len(globals().get("FORMATIONS", [0, 1, 2, 3, 4, 5]))
        values = np.broadcast_to(np.arange(count, dtype=np.float32) * 100.0, (n, count)).copy()
        return values, np.full(n, 1e6, np.float32)


class _NeutralDenseImputer:
    def impute(self, xy_q, self_wid=None, k=None, nfetch=None):
        n = len(np.atleast_2d(xy_q))
        return np.zeros(n, np.float32), np.zeros(n, np.float32), np.full(n, 1e6, np.float32)


def _rmse(y, p):
    return float(np.sqrt(np.mean(np.square(np.asarray(y, float) - np.asarray(p, float)))))


def _add_likpf(frame, horizontal, typewell):
    last = frame["last_known_tvt"].to_numpy(np.float32)
    defaults = ("likpf_scale_3", "likpf_scale_5", "likpf_scale_8", "likpf_scale_12", "likpf_mean")
    for column in defaults:
        frame[column] = last
    output, row_index, _ = lik_pf(horizontal.copy(), typewell.copy())
    if len(row_index):
        table = pd.DataFrame({"_row_idx": np.asarray(row_index, int)})
        generated = []
        for key, values in output.items():
            column = "likpf_" + key.replace("pf_scale_", "scale_").replace("pf_mean", "mean")
            table[column] = np.asarray(values, np.float32)
            generated.append(column)
        frame = frame.drop(columns=[c for c in generated if c in frame.columns])
        frame = frame.merge(table, on="_row_idx", how="left", validate="one_to_one")
    for column in defaults:
        frame[column] = frame.get(column, frame["last_known_tvt"]).fillna(frame["last_known_tvt"]).astype(np.float32)
        frame[column + "_d"] = frame[column] - frame["last_known_tvt"]
    return frame


def _pseudo_well(well, shift_ft, feature_names):
    path = DATA / f"{well}__horizontal_well.csv"
    source = pd.read_csv(path)
    known = np.flatnonzero(np.isfinite(source["TVT_input"].to_numpy(dtype=float, copy=True)))
    anchor = int(known[-1])
    md = source["MD"].to_numpy(dtype=float, copy=True)
    target_md = md[anchor] - shift_ft
    cutoff = int(np.argmin(np.abs(md - target_md)))
    if cutoff < 128 or cutoff >= anchor - 64:
        raise RuntimeError(f"{well}: insufficient pseudo-prefix room")
    masked = source[["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"]].copy(deep=True)
    # Pandas may expose a read-only view for homogeneous columns. The mask
    # construction below mutates these arrays, so request independent storage.
    tvt = masked["TVT"].to_numpy(dtype=float, copy=True)
    input_values = masked["TVT_input"].to_numpy(dtype=float, copy=True)
    input_values[: cutoff + 1] = tvt[: cutoff + 1]
    input_values[cutoff + 1 :] = np.nan
    masked["TVT_input"] = input_values
    # The public builder derives well identity from the canonical filename.
    temp = OUT / "tmp" / f"{well}__horizontal_well.csv"
    temp.parent.mkdir(parents=True, exist_ok=True)
    masked.to_csv(temp, index=False)
    typewell_path = DATA / f"{well}__typewell.csv"
    typewell = pd.read_csv(typewell_path).sort_values("TVT")
    try:
        frame = build_well(str(temp), str(typewell_path), is_train=True)
        if frame is None or len(frame) < 64:
            raise RuntimeError("build_well returned too few rows")
        frame = frame.reset_index(drop=True)
        frame["_row_idx"] = frame["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
        frame = _add_likpf(frame, masked, typewell)
        last = frame["last_known_tvt"].to_numpy(float)
        lik = frame["likpf_scale_5"].to_numpy(float)
        warm = np.maximum(1.0 - np.exp(-np.maximum(frame["md_since"].to_numpy(float), 0.0) / 85.0), 0.15)
        frame["residual_target"] = np.clip((frame["target"].to_numpy(float) - (lik - last)) / warm, -80.0, 80.0)
        available = [c for c in feature_names if c in frame.columns]
        if len(available) != len(feature_names):
            raise RuntimeError(f"{well}: missing {len(feature_names) - len(available)} V215 features")
        positions = np.unique(np.round(np.linspace(0, len(frame) - 1, min(ROWS_PER_WELL, len(frame)))).astype(int))
        return frame.iloc[positions].copy(), len(frame), cutoff
    finally:
        if temp.is_file():
            temp.unlink()


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    feature_names = pd.read_csv(V215 / "features.csv")["feature"].astype(str).tolist()
    meta = pd.read_csv(V215 / "well_metadata.csv", dtype={"well": str})
    held = meta.loc[meta.fold.eq(HELD_FOLD), "well"].astype(str).tolist()[:HELD_WELLS]
    train_pool = meta.loc[~meta.fold.eq(HELD_FOLD), "well"].astype(str).tolist()
    rng = np.random.default_rng(SEED)
    train_wells = sorted(rng.choice(train_pool, size=PSEUDO_TRAIN_WELLS, replace=False).tolist())
    print("V238 CONFIG", {"held_wells": len(held), "pseudo_train_wells": len(train_wells), "shift_ft": SHIFT_FT}, flush=True)

    pseudo_parts = []
    build_reports = []
    builder_globals = build_well.__globals__
    old_fi, old_di = builder_globals.get("_FI"), builder_globals.get("_DI")
    builder_globals["_FI"], builder_globals["_DI"] = _NeutralFormationImputer(), _NeutralDenseImputer()
    try:
        for number, well in enumerate(train_wells, 1):
            try:
                frame, rows, cutoff = _pseudo_well(well, SHIFT_FT, feature_names)
                frame["well"] = well
                frame["sample_weight"] = 1.0
                pseudo_parts.append(frame)
                build_reports.append({"well": well, "status": "ok", "rows": rows, "cutoff": cutoff})
            except Exception as error:
                build_reports.append({
                    "well": well,
                    "status": "failed",
                    "reason": repr(error),
                    "traceback": traceback.format_exc(limit=8),
                })
            if number % 8 == 0 or number == len(train_wells):
                print(f"V238 pseudo features {number}/{len(train_wells)} failures={sum(r['status'] != 'ok' for r in build_reports)}", flush=True)
    finally:
        builder_globals["_FI"], builder_globals["_DI"] = old_fi, old_di
    if not pseudo_parts:
        failures = [row for row in build_reports if row["status"] != "ok"]
        raise RuntimeError({"message": "all pseudo wells failed", "failures": failures[:10]})
    pseudo = pd.concat(pseudo_parts, ignore_index=True)
    if pseudo.well.nunique() < int(PSEUDO_TRAIN_WELLS * 0.95):
        raise RuntimeError(f"Too many pseudo-prefix failures: {len(build_reports) - pseudo.well.nunique()}")

    valid_parts = []
    for well in held:
        frame = pd.read_pickle(V215 / "features" / f"{well}__full.pkl")
        valid_parts.append(frame)
    valid = pd.concat(valid_parts, ignore_index=True, sort=False)
    features = lambda frame: frame.reindex(columns=feature_names).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    x_train = features(pseudo)
    y_train = pseudo["residual_target"].to_numpy(np.float32)
    x_valid = features(valid)
    y_valid = valid["residual_target"].to_numpy(np.float32)
    params = dict(objective="regression", n_estimators=int(os.environ.get("V238_ESTIMATORS", "2200")), learning_rate=0.025, num_leaves=64, min_child_samples=80, subsample=.8, subsample_freq=1, colsample_bytree=.7, reg_alpha=.1, reg_lambda=10., max_bin=255, n_jobs=-1, random_state=SEED, verbosity=-1)
    if os.environ.get("V238_CPU", "0") != "1":
        params["device_type"] = "gpu"
    try:
        model = LGBMRegressor(**params)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], callbacks=[early_stopping(150, verbose=False)])
    except Exception as error:
        if "gpu" not in str(error).lower() and "opencl" not in str(error).lower():
            raise
        os.environ["V238_CPU"] = "1"
        model = LGBMRegressor(**params)
        params.pop("device_type", None)
        model = LGBMRegressor(**params)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], callbacks=[early_stopping(150, verbose=False)])
    correction = model.predict(x_valid, num_iteration=model.best_iteration_)
    last = valid["last_known_tvt"].to_numpy(float)
    lik = valid["likpf_scale_5"].to_numpy(float)
    truth = last + valid["target"].to_numpy(float)
    warm = np.maximum(1.0 - np.exp(-np.maximum(valid["md_since"].to_numpy(float), 0.0) / 85.0), 0.15)
    candidate = lik + warm * correction
    parent_rmse = _rmse(truth, lik)
    candidate_rmse = _rmse(truth, candidate)
    verdict = {
        "experiment": "pseudo_prefix_gbdt_gateway_v238",
        "stage": "artificial_cutoff_rebuilt_features_gateway",
        "held_fold": HELD_FOLD, "held_wells": len(held), "pseudo_train_wells": int(pseudo.well.nunique()),
        "pseudo_shift_ft": SHIFT_FT, "features": len(feature_names), "best_iteration": int(model.best_iteration_),
        "parent_rmse": parent_rmse, "candidate_rmse": candidate_rmse, "gain": parent_rmse - candidate_rmse,
        "build_failures": [r for r in build_reports if r["status"] != "ok"],
        "checks": {"complete_pseudo_wells_ge_0p95": pseudo.well.nunique() >= int(PSEUDO_TRAIN_WELLS * .95), "gain_ge_0p20": parent_rmse - candidate_rmse >= .20, "candidate_rmse_le_9": candidate_rmse <= 9.0},
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    verdict["strict_pass"] = bool(all(verdict["checks"].values()))
    verdict["next"] = "PASS -> run full fivefold pseudo-prefix training" if verdict["strict_pass"] else "FAIL -> pseudo-prefix training does not unlock the GBDT gap"
    pd.DataFrame(build_reports).to_csv(OUT / "build_report.csv", index=False)
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x), encoding="utf-8")
    print("\nV238 FINAL VERDICT\n", json.dumps(verdict, indent=2), flush=True)
    print("saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
