"""V236: 300-ft local bedding-slope prior audit.

Tests the specific hypothesis from the discussion comment:
S = TVT + Z has a locally stable slope in cumulative XY distance, and
neighboring wells can provide a prior for that slope.  This is an oracle
ceiling/low-dimensional prior audit, not an HMM implementation.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
DATA = ROOT / "rogii-wellbore-geology-prediction" / "train"
V168 = ROOT / "oof_exports" / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
FOLDS = ROOT / "oof_exports" / "likpf_residual_gbdt_full5_v215" / "well_metadata.csv"
OUT = ROOT / "oof_exports" / "local_bedding_slope_prior_audit_v236"
WINDOW = 300.0
K_VALUES = (4, 8, 16, 32, 64)


def rmse(y, p):
    return float(np.sqrt(np.mean(np.square(np.asarray(y, float) - np.asarray(p, float)))))


def slope(x, y, lo=None, hi=None):
    keep = np.isfinite(x) & np.isfinite(y)
    if lo is not None:
        keep &= x >= lo
    if hi is not None:
        keep &= x <= hi
    x, y = x[keep], y[keep]
    if len(x) < 8 or np.ptp(x) < 1.0:
        return 0.0
    x0 = x - x.mean()
    return float(np.dot(x0, y - y.mean()) / max(np.dot(x0, x0), 1e-9))


def read_well(well):
    frame = pd.read_csv(DATA / f"{well}__horizontal_well.csv", usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
    md = frame.MD.to_numpy(float)
    x = frame.X.to_numpy(float)
    y = frame.Y.to_numpy(float)
    z = frame.Z.to_numpy(float)
    tvt = frame.TVT.to_numpy(float)
    tvt_input = frame.TVT_input.to_numpy(float)
    anchor = int(np.flatnonzero(np.isfinite(tvt_input))[-1])
    l = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    s_true = tvt + z
    s_visible = tvt_input + z
    prefix = np.isfinite(tvt_input)
    prefix_slope = slope(l[prefix], s_visible[prefix], max(l[anchor] - WINDOW, l[prefix][0]), l[anchor])
    train_slope = slope(l, s_true, l[anchor], l[anchor] + WINDOW)
    hidden = ~np.isfinite(tvt_input) & np.isfinite(tvt)
    hidden_rows = np.flatnonzero(hidden)
    return {
        "well": str(well), "fold": None, "anchor_x": float(x[anchor]), "anchor_y": float(y[anchor]),
        "anchor_l": float(l[anchor]), "md_since": md[hidden_rows] - md[anchor],
        "l_since": l[hidden_rows] - l[anchor], "dz": z[hidden_rows] - z[anchor],
        "truth": tvt[hidden_rows], "last_tvt": float(tvt_input[anchor]),
        "prefix_slope": prefix_slope, "train_slope": train_slope,
        "hidden_rows": hidden_rows,
    }


def main():
    started = time.time()
    if not DATA.is_dir() or not V168.is_file() or not FOLDS.is_file():
        raise FileNotFoundError({"data": str(DATA), "v168": str(V168), "folds": str(FOLDS)})
    folds = pd.read_csv(FOLDS, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
    if len(folds) != 773:
        raise RuntimeError(f"expected 773 folds, found {len(folds)}")
    records = []
    for number, well in enumerate(folds.well.astype(str), 1):
        record = read_well(well)
        record["fold"] = int(folds.loc[folds.well.eq(well), "fold"].iloc[0])
        records.append(record)
        if number % 100 == 0 or number == len(folds):
            print(f"V236 wells {number}/{len(folds)}", flush=True)

    rows = []
    slope_bank_rows = []
    for record in records:
        slope_bank_rows.append({
            "well": record["well"], "fold": record["fold"],
            "x": record["anchor_x"], "y": record["anchor_y"],
            "slope": record["train_slope"],
        })
        for index in range(len(record["truth"])):
            rows.append({
                "well": record["well"], "fold": record["fold"],
                "truth": record["truth"][index], "last_tvt": record["last_tvt"],
                "dz": record["dz"][index], "l_since": record["l_since"][index],
                "md_since": record["md_since"][index],
                "prefix_slope": record["prefix_slope"], "oracle_slope": record["train_slope"],
                "id": f"{record['well']}_{int(record['hidden_rows'][index])}",
            })
    data = pd.DataFrame(rows)
    bank = pd.DataFrame(slope_bank_rows)
    data["geom_z"] = data["last_tvt"] - data["dz"]
    data["prefix_prediction"] = data["geom_z"] + data["prefix_slope"] * data["l_since"]
    data["oracle_prediction"] = data["geom_z"] + data["oracle_slope"] * data["l_since"]

    v168 = pd.read_parquet(V168)[["id", "metadata_analog_w0p05"]].rename(columns={"metadata_analog_w0p05": "v168"})
    data = data.merge(v168, on="id", how="left", validate="one_to_one")
    if data.v168.isna().any():
        raise RuntimeError("V168 alignment incomplete")

    prediction_columns = {
        "v168": data.v168.to_numpy(float),
        "prefix_slope": data.prefix_prediction.to_numpy(float),
        "oracle_local_slope": data.oracle_prediction.to_numpy(float),
    }
    fold_values = data.fold.to_numpy(np.int8)
    truth = data.truth.to_numpy(float)
    for k in K_VALUES:
        neighbor_slope = np.empty(len(data), float)
        for record in records:
            mask = data.well.eq(record["well"]).to_numpy()
            candidates = bank[bank.fold.ne(record["fold"])].copy()
            distance = np.hypot(candidates.x.to_numpy(float) - record["anchor_x"], candidates.y.to_numpy(float) - record["anchor_y"])
            selected = candidates.iloc[np.argsort(distance)[:k]]
            prior = float(np.median(selected.slope.to_numpy(float)))
            neighbor_slope[mask] = prior
        prediction_columns[f"neighbor_k{k}"] = data["geom_z"].to_numpy(float) + neighbor_slope * data["l_since"].to_numpy(float)

    summaries = []
    for name, prediction in prediction_columns.items():
        error = prediction - truth
        well_rmse = data.assign(_se=error * error).groupby("well", sort=False)["_se"].mean().pow(0.5)
        far = data.md_since.to_numpy(float) >= 2000.0
        summaries.append({
            "endpoint": name, "rows": len(data), "wells": int(data.well.nunique()),
            "rmse": rmse(truth, prediction), "mae": float(np.mean(np.abs(error))),
            "median_well_rmse": float(well_rmse.median()), "p95_well_rmse": float(well_rmse.quantile(.95)),
            "far_gt2000_rmse": rmse(truth[far], prediction[far]),
        })
    summary = pd.DataFrame(summaries).sort_values("rmse")

    nested = []
    for held in range(1, 6):
        donor = fold_values != held
        held_mask = fold_values == held
        candidates = [name for name in prediction_columns if name != "oracle_local_slope"]
        selected = min(candidates, key=lambda name: rmse(truth[donor], prediction_columns[name][donor]))
        nested.append({
            "fold": held, "selected": selected,
            "base_rmse": rmse(truth[held_mask], prediction_columns["v168"][held_mask]),
            "candidate_rmse": rmse(truth[held_mask], prediction_columns[selected][held_mask]),
            "gain": rmse(truth[held_mask], prediction_columns["v168"][held_mask]) - rmse(truth[held_mask], prediction_columns[selected][held_mask]),
        })
    nested_frame = pd.DataFrame(nested)
    oracle_rmse = float(summary.loc[summary.endpoint.eq("oracle_local_slope"), "rmse"].iloc[0])
    best_deployable = summary.loc[~summary.endpoint.eq("oracle_local_slope")].iloc[0]
    verdict = {
        "experiment": "local_bedding_slope_prior_audit_v236",
        "stage": "300ft_XY_local_bedding_slope_neighbor_prior_gateway",
        "window_ft": WINDOW, "coverage": {"rows": len(data), "wells": int(data.well.nunique())},
        "summary": summary.to_dict("records"), "nested": nested,
        "oracle_local_slope_rmse": oracle_rmse,
        "best_deployable": best_deployable.to_dict(),
        "checks": {
            "oracle_rmse_le_6p5": oracle_rmse <= 6.5,
            "deployable_gain_vs_v168_ge_0p10": float(summary.loc[summary.endpoint.eq("v168"), "rmse"].iloc[0] - best_deployable.rmse) >= 0.10,
            "nested_positive_folds_ge_4": int((nested_frame.gain > 0).sum()) >= 4,
        },
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    verdict["strict_pass"] = bool(all(verdict["checks"].values()))
    verdict["next"] = "PASS -> implement the 300ft slope HMM transition" if verdict["strict_pass"] else "FAIL -> local bedding slope prior lacks enough deployable ceiling; retain V116"
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    nested_frame.to_csv(OUT / "nested.csv", index=False)
    data.to_parquet(OUT / "oof.parquet", index=False)
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x), encoding="utf-8")
    print("\nV236 SUMMARY\n", summary.to_string(index=False))
    print("\nV236 NESTED\n", nested_frame.to_string(index=False))
    print("\nV236 FINAL VERDICT\n", json.dumps(verdict, indent=2))
    print("saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
