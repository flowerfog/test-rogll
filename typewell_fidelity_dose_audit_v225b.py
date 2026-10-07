"""Read-only nested dose audit for frozen V225 fidelity alpha predictions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
SOURCE = OOF / "typewell_fidelity_observability_v225"
PARENT_PATH = OOF / "historical_analog_metadata_fullrow_v168/fullrow_oof.parquet"
ALT_PATH = OOF / "likpf_residual_foldclean_spatial_v217/oof.parquet"
OUT = OOF / "typewell_fidelity_dose_audit_v225b"
OUT.mkdir(parents=True, exist_ok=True)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _rmse(truth, prediction):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    return float(np.sqrt(np.mean(error * error)))


def _endpoint(path, candidates, name):
    frame = pd.read_parquet(path)
    if "id" not in frame:
        well_col = next(column for column in ("well", "well_id") if column in frame)
        row_col = next(column for column in ("row_idx", "row_index", "_row_idx") if column in frame)
        frame["id"] = frame[well_col].astype(str) + "_" + frame[row_col].astype(int).astype(str)
    prediction = next(column for column in candidates if column in frame)
    truth = next((column for column in ("tvt_true", "truth", "TVT") if column in frame), None)
    columns = ["id", prediction] + ([truth] if truth else [])
    result = frame[columns].copy().rename(columns={prediction: name})
    if truth:
        result = result.rename(columns={truth: "tvt_true"})
    return result


for required in (SOURCE / "well_fidelity.csv", PARENT_PATH, ALT_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

well = pd.read_csv(SOURCE / "well_fidelity.csv", dtype={"well": str})
required_columns = {"well", "fold", "predicted_alpha", "shuffled_alpha"}
if not required_columns.issubset(well):
    raise RuntimeError(f"Missing V225 columns: {sorted(required_columns - set(well))}")

parent = _endpoint(PARENT_PATH, ("metadata_analog_w0p05",), "v168")
alternative = _endpoint(ALT_PATH, ("candidate", "v217"), "v217")
frame = parent.merge(alternative.drop(columns=["tvt_true"], errors="ignore"), on="id", validate="one_to_one")
if "tvt_true" not in frame:
    alt_truth = _endpoint(ALT_PATH, ("candidate", "v217"), "v217")
    frame = frame.merge(alt_truth[["id", "tvt_true"]], on="id", validate="one_to_one")
frame["well"] = frame["id"].astype(str).str.rsplit("_", n=1).str[0]
lookup = well.set_index("well")
frame["fold"] = frame["well"].map(lookup["fold"]).astype(int)
frame["alpha"] = frame["well"].map(lookup["predicted_alpha"]).to_numpy(float)
frame["shuffle_alpha"] = frame["well"].map(lookup["shuffled_alpha"]).to_numpy(float)
if frame[["fold", "alpha", "shuffle_alpha"]].isna().any().any():
    raise RuntimeError("Incomplete V225 well alignment")

truth = frame["tvt_true"].to_numpy(float)
parent_values = frame["v168"].to_numpy(float)
move = frame["v217"].to_numpy(float) - parent_values
fold = frame["fold"].to_numpy(int)
doses = np.round(np.arange(0.0, 1.0001, 0.025), 3)
scan_rows = []
for dose in doses:
    prediction = parent_values + dose * frame["alpha"].to_numpy(float) * move
    control = parent_values + dose * frame["shuffle_alpha"].to_numpy(float) * move
    row = {"dose": float(dose), "rmse": _rmse(truth, prediction),
           "shuffle_rmse": _rmse(truth, control)}
    for held in range(1, 6):
        mask = fold == held
        row[f"fold{held}_rmse"] = _rmse(truth[mask], prediction[mask])
    scan_rows.append(row)
scan = pd.DataFrame(scan_rows).sort_values("rmse").reset_index(drop=True)

nested = parent_values.copy()
nested_rows = []
alpha = frame["alpha"].to_numpy(float)
for held in range(1, 6):
    train = fold != held
    valid = ~train
    selected = min(doses, key=lambda dose: (_rmse(truth[train], parent_values[train] + dose * alpha[train] * move[train]), dose))
    nested[valid] = parent_values[valid] + selected * alpha[valid] * move[valid]
    base_rmse = _rmse(truth[valid], parent_values[valid])
    candidate_rmse = _rmse(truth[valid], nested[valid])
    nested_rows.append({"fold": held, "selected_dose": float(selected),
                        "base_rmse": base_rmse, "candidate_rmse": candidate_rmse,
                        "gain": base_rmse - candidate_rmse})

base_rmse = _rmse(truth, parent_values)
nested_rmse = _rmse(truth, nested)
best = scan.iloc[0]
checks = {
    "best_fixed_gain_ge_0p01": base_rmse - float(best["rmse"]) >= 0.01,
    "best_fixed_beats_shuffle_ge_0p01": float(best["shuffle_rmse"] - best["rmse"]) >= 0.01,
    "nested_gain_positive": base_rmse - nested_rmse > 0.0,
    "positive_folds_ge_4": sum(row["gain"] > 0 for row in nested_rows) >= 4,
    "minimum_fold_gain_ge_minus0p01": min(row["gain"] for row in nested_rows) >= -0.01,
}
verdict = {
    "experiment": "typewell_fidelity_dose_audit_v225b",
    "stage": "read_only_nested_shrinkage_of_frozen_V225_router",
    "selection_or_training_count": 0,
    "base_rmse": base_rmse,
    "best_fixed": best.to_dict(),
    "nested_rmse": nested_rmse,
    "nested_gain": base_rmse - nested_rmse,
    "folds": nested_rows,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> retain only the shrunken fidelity dose" if all(checks.values())
            else "FAIL -> fidelity is descriptive but not actionable; close this family",
}

scan.to_csv(OUT / "dose_scan.csv", index=False)
pd.DataFrame(nested_rows).to_csv(OUT / "folds.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")
print("\nV225b DOSE TOP 20")
print(scan.head(20).to_string(index=False))
print("\nV225b FOLDS")
print(pd.DataFrame(nested_rows).to_string(index=False))
print("\nV225b FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT)
