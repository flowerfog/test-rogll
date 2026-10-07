"""V231b: read-only nested shrinkage audit of the frozen V231 move."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "oof_exports/canonical_polynomial_registration_full773_v231/fullrow_oof.parquet"
OUT = ROOT / "oof_exports/polynomial_move_risk_shrinkage_v231b"
OUT.mkdir(parents=True, exist_ok=True)

DOSES = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25)
CAPS = (2.0, 4.0, 6.0, 8.0, 12.0, 20.0, np.inf)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

data = pd.read_parquet(SOURCE)
required = {"id", "well", "fold", "truth", "v168", "polynomial_move", "v231"}
if not required.issubset(data.columns):
    raise RuntimeError({"missing": sorted(required - set(data.columns))})
if len(data) != 3_783_989 or data["well"].nunique() != 773:
    raise RuntimeError({"rows": len(data), "wells": data["well"].nunique()})

truth = data["truth"].to_numpy(np.float64)
base = data["v168"].to_numpy(np.float64)
raw_move = data["polynomial_move"].to_numpy(np.float64)
fold = data["fold"].to_numpy(np.int8)
base_error = base - truth
base_sse = float(np.dot(base_error, base_error))
base_rmse = float(np.sqrt(base_sse / len(data)))
base_fold_rmse = {
    held: float(np.sqrt(np.mean(base_error[fold == held] ** 2))) for held in range(1, 6)
}

configs = [(float(cap), float(dose)) for cap in CAPS for dose in DOSES]
fold_sse = {held: {} for held in range(1, 6)}
fold_rows = {held: int(np.sum(fold == held)) for held in range(1, 6)}
report = []

for cap, dose in configs:
    shaped = raw_move if np.isinf(cap) else cap * np.tanh(raw_move / cap)
    error = base_error + dose * shaped
    row = {"cap": cap, "dose": dose, "rmse": float(np.sqrt(np.mean(error ** 2)))}
    for held in range(1, 6):
        mask = fold == held
        sse = float(np.dot(error[mask], error[mask]))
        fold_sse[held][(cap, dose)] = sse
        row[f"fold{held}_rmse"] = float(np.sqrt(sse / mask.sum()))
        row[f"fold{held}_gain"] = base_fold_rmse[held] - row[f"fold{held}_rmse"]
    row["positive_folds"] = sum(row[f"fold{held}_gain"] > 0 for held in range(1, 6))
    row["minimum_fold_gain"] = min(row[f"fold{held}_gain"] for held in range(1, 6))
    report.append(row)

report = pd.DataFrame(report).sort_values("rmse").reset_index(drop=True)
stable_fixed = report.loc[
    (report["positive_folds"] >= 4) & (report["minimum_fold_gain"] >= -0.10)
]
best_stable_fixed = None if stable_fixed.empty else stable_fixed.iloc[0].to_dict()

nested_rows = []
nested_prediction = np.empty(len(data), np.float64)
for held in range(1, 6):
    train_folds = [value for value in range(1, 6) if value != held]
    key = min(
        configs,
        key=lambda candidate: (
            np.sqrt(
                sum(fold_sse[value][candidate] for value in train_folds)
                / sum(fold_rows[value] for value in train_folds)
            ),
            candidate,
        ),
    )
    cap, dose = key
    mask = fold == held
    shaped = raw_move[mask] if np.isinf(cap) else cap * np.tanh(raw_move[mask] / cap)
    nested_prediction[mask] = base[mask] + dose * shaped
    candidate_rmse = float(np.sqrt(np.mean((nested_prediction[mask] - truth[mask]) ** 2)))
    parent_rmse = float(np.sqrt(np.mean(base_error[mask] ** 2)))
    nested_rows.append({
        "fold": held, "cap": cap, "dose": dose,
        "base_rmse": parent_rmse, "candidate_rmse": candidate_rmse,
        "gain": parent_rmse - candidate_rmse,
    })

nested_error = nested_prediction - truth
nested_rmse = float(np.sqrt(np.mean(nested_error ** 2)))
perwell = pd.DataFrame({
    "well": data["well"].astype(str), "base_sq": base_error ** 2,
    "candidate_sq": nested_error ** 2,
}).groupby("well", as_index=False).agg(
    rows=("base_sq", "size"), base_sse=("base_sq", "sum"), candidate_sse=("candidate_sq", "sum")
)

rng = np.random.default_rng(231231)
gain = np.empty(5000, float)
rows = perwell["rows"].to_numpy(np.int64)
left = perwell["base_sse"].to_numpy(float)
right = perwell["candidate_sse"].to_numpy(float)
for draw in range(len(gain)):
    index = rng.integers(0, len(perwell), len(perwell))
    denominator = rows[index].sum()
    gain[draw] = np.sqrt(left[index].sum() / denominator) - np.sqrt(right[index].sum() / denominator)

bootstrap = {
    "draws": len(gain),
    "gain_p05": float(np.quantile(gain, 0.05)),
    "gain_median": float(np.median(gain)),
    "gain_p95": float(np.quantile(gain, 0.95)),
    "probability_gain_positive": float(np.mean(gain > 0)),
}
checks = {
    "nested_gain_ge_0p03": base_rmse - nested_rmse >= 0.03,
    "positive_folds_ge_4": sum(row["gain"] > 0 for row in nested_rows) >= 4,
    "minimum_fold_gain_ge_minus0p10": min(row["gain"] for row in nested_rows) >= -0.10,
    "bootstrap_probability_ge_0p90": bootstrap["probability_gain_positive"] >= 0.90,
}
verdict = {
    "experiment": "polynomial_move_risk_shrinkage_v231b",
    "stage": "read_only_nested_dose_and_smooth_cap_audit",
    "selection_or_training_count": 0,
    "source": str(SOURCE),
    "base_rmse": base_rmse,
    "fixed_v231_rmse": float(np.sqrt(np.mean((data["v231"].to_numpy(float) - truth) ** 2))),
    "best_fixed": report.iloc[0].to_dict(),
    "best_stable_fixed": best_stable_fixed,
    "nested_rmse": nested_rmse,
    "nested_gain": base_rmse - nested_rmse,
    "folds": nested_rows,
    "bootstrap": bootstrap,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> freeze nested-stable global cap/dose for Kaggle deployment" if all(checks.values())
            else "FAIL -> do not deploy the polynomial move",
}

report.to_csv(OUT / "scan.csv", index=False)
pd.DataFrame(nested_rows).to_csv(OUT / "folds.csv", index=False)
pd.DataFrame({
    "id": data["id"], "well": data["well"], "fold": data["fold"],
    "truth": truth, "v168": base, "nested_prediction": nested_prediction,
}).to_parquet(OUT / "nested_oof.parquet", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")

print("V231b TOP 20")
print(report.head(20).to_string(index=False))
print("\nV231b FOLDS")
print(pd.DataFrame(nested_rows).to_string(index=False))
print("\nV231b FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT)
