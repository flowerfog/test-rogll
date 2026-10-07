"""V234: read-only incremental audit of the frozen V233 geometry delta.

No model is fitted.  The V233 correction is added to the stronger V217 OOF
endpoint, with its dose selected on the other four folds for every held fold.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF = ROOT / "oof_exports"
V217 = OOF / "likpf_residual_foldclean_spatial_v217" / "oof.parquet"
V233 = OOF / "geometry_conditioned_likpf_residual_v233" / "oof.parquet"
OUT = OOF / "geometry_delta_incremental_audit_v234"
DOSES = np.round(np.arange(-0.25, 1.5001, 0.025), 3)


def rmse(truth, prediction):
    return float(np.sqrt(np.mean(np.square(np.asarray(truth) - np.asarray(prediction)))))


def metrics(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(float)
    error = prediction - truth
    well_rmse = frame.assign(_se=error * error).groupby("well", sort=False)["_se"].mean().pow(0.5)
    far = frame["md_since"].to_numpy(float) >= 2000.0
    return {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": rmse(truth, prediction),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "far_gt2000_rmse": rmse(truth[far], prediction[far]),
    }


def main():
    started = time.time()
    for path in (V217, V233):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)

    left = pd.read_parquet(V217)[["id", "well", "fold", "md_since", "tvt_true", "likpf", "candidate"]]
    left = left.rename(columns={"candidate": "v217"})
    right = pd.read_parquet(V233)[
        ["id", "likpf", "geometry_conditioned_nested", "geometry_residual_effective_raw"]
    ].rename(columns={"likpf": "likpf_v233"})
    frame = left.merge(right, on="id", how="left", validate="one_to_one")
    if len(frame) != 3_783_989 or frame["well"].nunique() != 773 or frame.isna().any().any():
        raise RuntimeError({"rows": len(frame), "wells": frame.well.nunique(), "na": frame.isna().sum().to_dict()})
    if np.max(np.abs(frame["likpf"] - frame["likpf_v233"])) > 1e-5:
        raise RuntimeError("V217 and V233 likPF parents differ")

    truth = frame["tvt_true"].to_numpy(float)
    parent = frame["v217"].to_numpy(float)
    delta = frame["geometry_conditioned_nested"].to_numpy(float) - frame["likpf"].to_numpy(float)
    folds = frame["fold"].to_numpy(np.int8)

    scan_rows = []
    predictions = {}
    for dose in DOSES:
        prediction = parent + dose * delta
        predictions[float(dose)] = prediction
        row = {"dose": float(dose), "rmse": rmse(truth, prediction)}
        for fold in range(1, 6):
            mask = folds == fold
            row[f"fold{fold}_rmse"] = rmse(truth[mask], prediction[mask])
        scan_rows.append(row)
    scan = pd.DataFrame(scan_rows).sort_values("rmse").reset_index(drop=True)

    nested = np.empty(len(frame), float)
    fold_rows = []
    for held_fold in range(1, 6):
        donor = folds != held_fold
        donor_truth = truth[donor]
        donor_parent = parent[donor]
        donor_delta = delta[donor]
        donor_rmse = np.asarray(
            [rmse(donor_truth, donor_parent + dose * donor_delta) for dose in DOSES]
        )
        dose = float(DOSES[int(np.argmin(donor_rmse))])
        held = folds == held_fold
        nested[held] = predictions[dose][held]
        row = {
            "fold": held_fold,
            "selected_dose": dose,
            "base_rmse": rmse(truth[held], parent[held]),
            "candidate_rmse": rmse(truth[held], nested[held]),
        }
        row["gain"] = row["base_rmse"] - row["candidate_rmse"]
        fold_rows.append(row)

    summaries = [metrics(frame, parent, "v217"), metrics(frame, nested, "v217_plus_nested_v233_delta")]
    gain = summaries[0]["rmse"] - summaries[1]["rmse"]
    fixed = scan.iloc[0].to_dict()
    verdict = {
        "experiment": "geometry_delta_incremental_audit_v234",
        "stage": "read_only_nested_incremental_value_of_V233_over_V217",
        "selection_or_training_count": 0,
        "base": summaries[0],
        "candidate": summaries[1],
        "nested_gain": gain,
        "best_fixed": fixed,
        "folds": fold_rows,
        "checks": {
            "nested_gain_ge_0p05": gain >= 0.05,
            "positive_folds_ge_4": sum(row["gain"] > 0 for row in fold_rows) >= 4,
            "minimum_fold_gain_ge_minus0p03": min(row["gain"] for row in fold_rows) >= -0.03,
            "median_selected_dose_ge_0p10": float(np.median([row["selected_dose"] for row in fold_rows])) >= 0.10,
        },
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    verdict["strict_pass"] = bool(all(verdict["checks"].values()))
    verdict["next"] = (
        "PASS -> merge the 33 V233 columns into the V217 training representation"
        if verdict["strict_pass"]
        else "FAIL -> V233 geometry gain is redundant with V217; do not retrain the merged model"
    )

    pd.DataFrame(fold_rows).to_csv(OUT / "folds.csv", index=False)
    scan.to_csv(OUT / "dose_scan.csv", index=False)
    pd.DataFrame({
        "id": frame["id"], "well": frame["well"], "fold": folds,
        "tvt_true": truth, "v217": parent, "v233_delta": delta, "candidate": nested,
    }).to_parquet(OUT / "oof.parquet", index=False)
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("V234 TOP 20")
    print(scan.head(20).to_string(index=False))
    print("\nV234 FOLDS")
    print(pd.DataFrame(fold_rows).to_string(index=False))
    print("\nV234 FINAL VERDICT")
    print(json.dumps(verdict, indent=2))
    print("saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
