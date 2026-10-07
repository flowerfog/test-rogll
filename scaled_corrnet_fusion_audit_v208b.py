"""Read-only Fold2 fusion audit for V208 scale profiles against V168.

Fold2 was the preregistered V208 gateway fold.  Truth is used only for this
investment diagnostic.  No Fold3/Fold1 artifact is loaded and no deployable
weight is produced here.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("/root/autodl-tmp")
OOF = ROOT / "oof_exports"
V208 = OOF / "tvt_corrnet_scaled_fullfold_gateway_v208"
V168 = OOF / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
OUT = OOF / "scaled_corrnet_fusion_audit_v208b"
OUT.mkdir(parents=True, exist_ok=True)
PROFILES = ("small", "medium", "large")
BASE_COLUMN = "metadata_analog_w0p05"


def metrics(truth, prediction, well, distance, endpoint):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    frame = pd.DataFrame({"well": np.asarray(well), "e2": error * error})
    by_well = np.sqrt(frame.groupby("well")["e2"].mean())
    result = {
        "endpoint": endpoint,
        "rows": int(len(error)),
        "wells": int(by_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(by_well.median()),
        "p95_well_rmse": float(by_well.quantile(0.95)),
    }
    bands = (
        ("near_0_500_rmse", (distance >= 0) & (distance < 500)),
        ("mid_500_1000_rmse", (distance >= 500) & (distance < 1000)),
        ("mid_1000_2000_rmse", (distance >= 1000) & (distance < 2000)),
        ("far_gt2000_rmse", distance >= 2000),
    )
    for name, mask in bands:
        result[name] = float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else None
    return result


def load_v168(keys):
    if not V168.is_file():
        raise FileNotFoundError(V168)
    header = pq.ParquetFile(V168).schema.names
    if BASE_COLUMN not in header:
        raise RuntimeError(f"V168 missing {BASE_COLUMN}: {header}")
    columns = [name for name in ("id", "well", "row_idx", "truth", "tvt_true", BASE_COLUMN) if name in header]
    selected_wells = sorted(keys["well"].astype(str).unique().tolist())
    base = pd.read_parquet(
        V168, columns=columns, filters=[("well", "in", selected_wells)]
    )
    base["well"] = base["well"].astype(str)
    if "row_idx" not in base:
        if "id" not in base:
            raise RuntimeError("V168 has neither row_idx nor id")
        base["row_idx"] = pd.to_numeric(base["id"].str.rsplit("_", n=1).str[-1], errors="raise").astype(np.int64)
    base = base[["well", "row_idx", BASE_COLUMN, *[name for name in ("truth", "tvt_true") if name in base]]]
    return keys.merge(base, on=["well", "row_idx"], how="left", validate="one_to_one")


def detrend(values, distance):
    values = np.asarray(values, np.float64)
    distance = np.asarray(distance, np.float64)
    output = values.copy()
    edges = np.unique(np.r_[0.0, np.quantile(distance, np.linspace(0, 1, 33)), np.inf])
    code = np.clip(np.searchsorted(edges, distance, side="right") - 1, 0, len(edges) - 2)
    for index in np.unique(code):
        mask = code == index
        output[mask] -= np.mean(output[mask])
    return output


first_path = V208 / "medium_fold2_fullrow.parquet"
if not first_path.is_file():
    raise FileNotFoundError(first_path)
first = pd.read_parquet(first_path)
first["well"] = first["well"].astype(str)
keys = first[["well", "row_idx"]].copy()
frame = load_v168(keys)
if frame[BASE_COLUMN].isna().any():
    raise RuntimeError(f"Incomplete V168 alignment: {int(frame[BASE_COLUMN].isna().sum())} rows")

truth_reference = first["truth"].to_numpy(np.float64)
truth_column = next((name for name in ("truth", "tvt_true") if name in frame), None)
if truth_column is not None:
    gap = float(np.max(np.abs(frame[truth_column].to_numpy(np.float64) - truth_reference)))
    if gap > 0.002:
        raise RuntimeError(f"Truth lineage gap: {gap}")
else:
    gap = None
frame["truth_v208"] = truth_reference
frame["distance"] = first["distance"].to_numpy(np.float64)

base = frame[BASE_COLUMN].to_numpy(np.float64)
truth = frame["truth_v208"].to_numpy(np.float64)
distance = frame["distance"].to_numpy(np.float64)
well = frame["well"].to_numpy()
base_error = base - truth
summary_rows = [metrics(truth, base, well, distance, "v168")]
pairwise_rows = []
scan_rows = []

for profile in PROFILES:
    path = V208 / f"{profile}_fold2_fullrow.parquet"
    if not path.is_file():
        continue
    candidate = pd.read_parquet(path)
    candidate["well"] = candidate["well"].astype(str)
    aligned = keys.merge(
        candidate[["well", "row_idx", "truth", "prediction", "distance"]],
        on=["well", "row_idx"], how="left", validate="one_to_one",
    )
    if aligned[["truth", "prediction", "distance"]].isna().any().any():
        raise RuntimeError(f"{profile}: incomplete V208 alignment")
    candidate_truth = aligned["truth"].to_numpy(np.float64)
    profile_gap = float(np.max(np.abs(candidate_truth - truth)))
    if profile_gap > 0.002:
        raise RuntimeError(f"{profile}: truth gap {profile_gap}")
    prediction = aligned["prediction"].to_numpy(np.float64)
    candidate_error = prediction - truth
    cross = float(np.dot(base_error, candidate_error) / len(truth))
    base_mse = float(np.mean(base_error ** 2))
    candidate_mse = float(np.mean(candidate_error ** 2))
    denominator = base_mse + candidate_mse - 2.0 * cross
    alpha = float(np.clip((base_mse - cross) / denominator, 0.0, 1.0)) if denominator > 0 else 0.0
    blend = (1.0 - alpha) * base + alpha * prediction
    profile_metrics = metrics(truth, prediction, well, distance, f"corrnet_{profile}")
    blend_metrics = metrics(truth, blend, well, distance, f"blend_{profile}_oracle_alpha")
    summary_rows.extend((profile_metrics, blend_metrics))
    raw_pearson = float(np.corrcoef(base_error, candidate_error)[0, 1])
    cosine = float(np.dot(base_error, candidate_error) / np.sqrt(np.dot(base_error, base_error) * np.dot(candidate_error, candidate_error)))
    detrended_pearson = float(np.corrcoef(detrend(base_error, distance), detrend(candidate_error, distance))[0, 1])
    row = {
        "profile": profile,
        "candidate_rmse": profile_metrics["rmse"],
        "raw_error_pearson": raw_pearson,
        "uncentred_error_cosine": cosine,
        "distance_detrended_pearson": detrended_pearson,
        "oracle_alpha": alpha,
        "oracle_blend_rmse": blend_metrics["rmse"],
        "oracle_gain_vs_v168": summary_rows[0]["rmse"] - blend_metrics["rmse"],
    }
    for name, mask in (
        ("near", (distance >= 0) & (distance < 500)),
        ("mid1", (distance >= 500) & (distance < 1000)),
        ("mid2", (distance >= 1000) & (distance < 2000)),
        ("far", distance >= 2000),
    ):
        row[f"{name}_error_cosine"] = float(
            np.dot(base_error[mask], candidate_error[mask])
            / np.sqrt(np.dot(base_error[mask], base_error[mask]) * np.dot(candidate_error[mask], candidate_error[mask]))
        )
    pairwise_rows.append(row)
    for fixed_alpha in np.arange(0.0, 0.5001, 0.025):
        fixed = (1.0 - fixed_alpha) * base + fixed_alpha * prediction
        fixed_metrics = metrics(truth, fixed, well, distance, "fixed")
        scan_rows.append({
            "profile": profile, "alpha": float(fixed_alpha),
            "rmse": fixed_metrics["rmse"],
            "gain": summary_rows[0]["rmse"] - fixed_metrics["rmse"],
            "p95_well_rmse": fixed_metrics["p95_well_rmse"],
            "far_gt2000_rmse": fixed_metrics["far_gt2000_rmse"],
        })

summary = pd.DataFrame(summary_rows)
pairwise = pd.DataFrame(pairwise_rows).sort_values("oracle_blend_rmse")
scan = pd.DataFrame(scan_rows).sort_values(["profile", "rmse"])
summary.to_csv(OUT / "summary.csv", index=False)
pairwise.to_csv(OUT / "pairwise.csv", index=False)
scan.to_csv(OUT / "fixed_weight_scan.csv", index=False)
best = pairwise.iloc[0].to_dict()
investment_pass = bool(
    best["oracle_blend_rmse"] <= 6.873
    and best["oracle_gain_vs_v168"] >= 0.30
    and best["uncentred_error_cosine"] <= 0.45
)
verdict = {
    "experiment": "scaled_corrnet_fusion_audit_v208b",
    "stage": "read_only_Fold2_V208_V168_fusion_investment_audit",
    "selection_or_training_count": 0,
    "fold3_touched": 0,
    "fold1_touched": 0,
    "truth_max_abs_gap": gap,
    "base_rmse": float(summary_rows[0]["rmse"]),
    "pairwise": pairwise.to_dict("records"),
    "best": best,
    "checks": {
        "oracle_blend_rmse_le_6p873": int(best["oracle_blend_rmse"] <= 6.873),
        "oracle_gain_ge_0p30": int(best["oracle_gain_vs_v168"] >= 0.30),
        "error_cosine_le_0p45": int(best["uncentred_error_cosine"] <= 0.45),
    },
    "investment_pass": int(investment_pass),
    "next": (
        "PASS -> freeze Fold2 alpha and evaluate medium once on untouched Fold3"
        if investment_pass else
        "FAIL -> CorrNet is too inaccurate/correlated for fusion; close V208"
    ),
}
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("V208b PAIRWISE")
print(pairwise.to_string(index=False), flush=True)
print("\nV208b FIXED WEIGHT TOP")
print(scan.groupby("profile", group_keys=False).head(6).to_string(index=False), flush=True)
print("\nV208b SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV208b FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT, flush=True)
