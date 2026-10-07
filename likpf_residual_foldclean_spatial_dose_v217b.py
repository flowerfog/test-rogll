"""V217b: read-only fixed and nested dose audit for the V217 spatial move."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
V215 = OOF / "likpf_residual_gbdt_full5_v215" / "oof.parquet"
V217 = OOF / "likpf_residual_foldclean_spatial_v217" / "oof.parquet"
OUT = OOF / "likpf_residual_foldclean_spatial_dose_v217b"
OUT.mkdir(parents=True, exist_ok=True)
for path in (V215, V217):
    if not path.is_file():
        raise FileNotFoundError(path)

base = pd.read_parquet(V215)[["id", "candidate"]].rename(columns={"candidate": "base"})
spatial = pd.read_parquet(V217)[["id", "well", "fold", "md_since", "tvt_true", "candidate"]].rename(
    columns={"candidate": "spatial"}
)
frame = spatial.merge(base, on="id", how="left", validate="one_to_one")
if len(frame) != 3_783_989 or frame["base"].isna().any() or frame["id"].duplicated().any():
    raise RuntimeError("Incomplete V215/V217 alignment")

truth = frame["tvt_true"].to_numpy(float)
base_pred = frame["base"].to_numpy(float)
move = frame["spatial"].to_numpy(float) - base_pred
fold = frame["fold"].to_numpy(int)


def _rmse(mask, prediction):
    error = prediction[mask] - truth[mask]
    return float(np.sqrt(np.mean(error * error)))


def _p95(mask, prediction):
    local = frame.loc[mask, ["well"]].copy()
    local["se"] = (prediction[mask] - truth[mask]) ** 2
    values = np.sqrt(local.groupby("well")["se"].mean())
    return float(values.quantile(0.95))


all_mask = np.ones(len(frame), dtype=bool)
far_mask = frame["md_since"].to_numpy(float) > 2000.0
base_rmse = _rmse(all_mask, base_pred)
base_far = _rmse(far_mask, base_pred)
base_p95 = _p95(all_mask, base_pred)

rows = []
for alpha in np.round(np.arange(0.0, 1.2501, 0.025), 6):
    prediction = base_pred + alpha * move
    row = {
        "alpha": float(alpha),
        "rmse": _rmse(all_mask, prediction),
        "far_gt2000_rmse": _rmse(far_mask, prediction),
        "p95_well_rmse": _p95(all_mask, prediction),
    }
    row["gain"] = base_rmse - row["rmse"]
    row["far_gain"] = base_far - row["far_gt2000_rmse"]
    row["p95_gain"] = base_p95 - row["p95_well_rmse"]
    fold_gains = []
    for held in range(1, 6):
        mask = fold == held
        gain = _rmse(mask, base_pred) - _rmse(mask, prediction)
        row[f"fold{held}_gain"] = gain
        fold_gains.append(gain)
    row["positive_folds"] = int(sum(value > 0 for value in fold_gains))
    row["minimum_fold_gain"] = float(min(fold_gains))
    rows.append(row)
scan = pd.DataFrame(rows).sort_values("rmse", kind="stable").reset_index(drop=True)

# Nested one-dimensional selection: choose alpha on the other four OOF folds.
nested_pred = base_pred.copy()
nested_rows = []
for held in range(1, 6):
    train_mask = fold != held
    valid_mask = fold == held
    candidates = []
    for alpha in np.round(np.arange(0.0, 1.0001, 0.025), 6):
        prediction = base_pred + alpha * move
        candidates.append((float(alpha), _rmse(train_mask, prediction)))
    selected_alpha, inner_rmse = min(candidates, key=lambda item: (item[1], item[0]))
    nested_pred[valid_mask] = base_pred[valid_mask] + selected_alpha * move[valid_mask]
    nested_rows.append({
        "fold": held,
        "selected_alpha": selected_alpha,
        "inner_rmse": inner_rmse,
        "base_rmse": _rmse(valid_mask, base_pred),
        "rmse": _rmse(valid_mask, nested_pred),
    })
for row in nested_rows:
    row["gain"] = row["base_rmse"] - row["rmse"]

nested_rmse = _rmse(all_mask, nested_pred)
best = scan.iloc[0].to_dict()
stable = scan.loc[(scan["positive_folds"] >= 4) & (scan["minimum_fold_gain"] >= -0.05)]
best_stable = stable.iloc[0].to_dict() if len(stable) else None
verdict = {
    "experiment": "likpf_residual_foldclean_spatial_dose_v217b",
    "stage": "read_only_full773_fixed_and_nested_spatial_dose_audit",
    "selection_or_training_count": 0,
    "base_rmse": base_rmse,
    "full_spatial_rmse": float(scan.loc[np.isclose(scan["alpha"], 1.0), "rmse"].iloc[0]),
    "best_fixed": best,
    "best_stable_fixed": best_stable,
    "nested": {
        "rmse": nested_rmse,
        "gain": base_rmse - nested_rmse,
        "folds": nested_rows,
    },
    "checks": {
        "stable_fixed_exists": best_stable is not None,
        "stable_fixed_gain_ge_0p30": bool(best_stable is not None and best_stable["gain"] >= 0.30),
        "nested_gain_ge_0p30": bool(base_rmse - nested_rmse >= 0.30),
        "nested_positive_folds_ge_4": sum(row["gain"] > 0 for row in nested_rows) >= 4,
    },
}

scan.to_csv(OUT / "dose_scan.csv", index=False)
pd.DataFrame(nested_rows).to_csv(OUT / "nested_folds.csv", index=False)
pd.DataFrame({
    "id": frame["id"], "well": frame["well"], "fold": frame["fold"],
    "md_since": frame["md_since"], "tvt_true": truth,
    "v215": base_pred, "v217": frame["spatial"], "nested": nested_pred,
}).to_parquet(OUT / "oof.parquet", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("V217b FIXED TOP 20")
print(scan.head(20).to_string(index=False))
print("\nV217b NESTED FOLDS")
print(pd.DataFrame(nested_rows).to_string(index=False))
print("\nV217b FINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT)
