"""PSG-V3: transplant the frozen PSG feature delta onto exact V66/6.377 OOF.

No training and no dose selection occur here. The 0.40 dose was frozen by the
earlier PSG feature-delta survival audit. Nested dose scans are intentionally
excluded so this remains a lineage confirmation rather than another search.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/oof_exports")
V66_PATH = (
    ROOT / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
PSG_PATH = (
    ROOT / "psg_foldclean_fullstack_lgb_gateway_v1"
    / "psg_feature_lgb_oof.parquet"
)
OUT_DIR = ROOT / "psg_v66_exact_delta_gateway_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_DOSE = 0.40
DISTANCE_BINS = (
    ("near_0_500", 0.0, 500.0),
    ("mid_500_1000", 500.0, 1000.0),
    ("mid_1000_2000", 1000.0, 2000.0),
    ("far_gt2000", 2000.0, np.inf),
)
started = time.time()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    well_codes, wells = pd.factorize(frame["well"].astype(str), sort=False)
    counts = np.bincount(well_codes).astype(np.float64)
    sse = np.bincount(well_codes, weights=error * error, minlength=len(wells))
    well_rmse = np.sqrt(np.maximum(sse, 0.0) / counts)
    result = {
        "rows": int(len(frame)),
        "wells": int(len(wells)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    for name, lower, upper in DISTANCE_BINS:
        selected = (distance >= lower) & (distance < upper)
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[selected] ** 2))) if selected.any() else None
        )
    return result


def _normalise_keys(frame):
    frame = frame.copy()
    if "well" not in frame and "id" in frame:
        frame["well"] = frame["id"].astype(str).str.rsplit("_", n=1).str[0]
    if "row_idx" not in frame and "id" in frame:
        frame["row_idx"] = (
            frame["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(np.int32)
        )
    frame["well"] = frame["well"].astype(str).str.lower()
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int32)
    if frame.duplicated(["well", "row_idx"]).any():
        raise RuntimeError("Duplicate well/row_idx keys")
    return frame


for required in (V66_PATH, PSG_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

v66 = _normalise_keys(pd.read_parquet(V66_PATH))
psg = _normalise_keys(pd.read_parquet(PSG_PATH))

base_column = next(
    (
        column for column in (
            "incumbent_6p377_replay", "prediction", "tvt_pred", "pred"
        )
        if column in v66
    ),
    None,
)
if base_column is None:
    raise RuntimeError(f"Cannot identify V66 prediction column: {v66.columns.tolist()}")

required_v66 = {"well", "row_idx", "fold", "tvt_true", "md_since", base_column}
required_psg = {"well", "row_idx", "fold", "tvt_true", "baseline", "psg_features"}
if missing := sorted(required_v66.difference(v66.columns)):
    raise RuntimeError(f"V66 missing columns: {missing}")
if missing := sorted(required_psg.difference(psg.columns)):
    raise RuntimeError(f"PSG OOF missing columns: {missing}")

v66 = v66[list(required_v66)].rename(
    columns={base_column: "base", "fold": "base_fold", "tvt_true": "base_truth"}
)
psg = psg[list(required_psg)].rename(
    columns={"fold": "psg_fold", "tvt_true": "psg_truth"}
)
data = v66.merge(psg, on=["well", "row_idx"], how="inner", validate="one_to_one")
coverage = len(data) / len(v66)
if coverage < 0.995 or data["well"].nunique() < 770:
    raise RuntimeError(
        f"V66/PSG alignment insufficient: rows={coverage:.6%}, wells={data['well'].nunique()}"
    )
if not np.array_equal(
    data["base_fold"].to_numpy(np.int8), data["psg_fold"].to_numpy(np.int8)
):
    raise RuntimeError("V66 and PSG fold assignments differ")
truth_gap = np.abs(
    data["base_truth"].to_numpy(np.float64)
    - data["psg_truth"].to_numpy(np.float64)
)
if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.02:
    raise RuntimeError(f"Truth alignment gap: {float(np.nanmax(truth_gap))}")

data = data.rename(columns={"base_truth": "tvt_true", "base_fold": "fold"})
numeric = ["tvt_true", "md_since", "base", "baseline", "psg_features"]
if not np.isfinite(data[numeric].to_numpy(np.float64)).all():
    raise RuntimeError("Aligned data contains non-finite values")

base = data["base"].to_numpy(np.float64)
correction = (
    data["psg_features"].to_numpy(np.float64)
    - data["baseline"].to_numpy(np.float64)
)
candidate = base + FROZEN_DOSE * correction
reverse_control = base - FROZEN_DOSE * correction

base_metrics = _metrics(data, base)
candidate_metrics = _metrics(data, candidate)
reverse_metrics = _metrics(data, reverse_control)

fold_rows = []
for fold in sorted(data["fold"].unique()):
    held = data["fold"].to_numpy(np.int8) == int(fold)
    local = data.loc[held].reset_index(drop=True)
    base_local = base[held]
    candidate_local = candidate[held]
    reverse_local = reverse_control[held]
    bm = _metrics(local, base_local)
    cm = _metrics(local, candidate_local)
    rm = _metrics(local, reverse_local)
    fold_rows.append(
        {
            "fold": int(fold),
            "rows": cm["rows"],
            "wells": cm["wells"],
            "base_rmse": bm["rmse"],
            "rmse": cm["rmse"],
            "gain": bm["rmse"] - cm["rmse"],
            "base_far_rmse": bm["far_gt2000_rmse"],
            "far_rmse": cm["far_gt2000_rmse"],
            "far_gain": bm["far_gt2000_rmse"] - cm["far_gt2000_rmse"],
            "p95_change": cm["p95_well_rmse"] - bm["p95_well_rmse"],
            "reverse_rmse": rm["rmse"],
            "reverse_loss": rm["rmse"] - cm["rmse"],
        }
    )

fold_report = pd.DataFrame(fold_rows)
gain = base_metrics["rmse"] - candidate_metrics["rmse"]
far_gain = base_metrics["far_gt2000_rmse"] - candidate_metrics["far_gt2000_rmse"]
checks = {
    "alignment_ge_0p995": coverage >= 0.995,
    "aligned_wells_ge_770": int(data["well"].nunique()) >= 770,
    "fixed_gain_ge_0p03": gain >= 0.03,
    "fixed_far_gain_ge_0p03": far_gain >= 0.03,
    "positive_folds_ge_4": int((fold_report["gain"] > 0.0).sum()) >= 4,
    "minimum_fold_gain_ge_minus0p03": float(fold_report["gain"].min()) >= -0.03,
    "reverse_loses_in_ge_4_folds": int((fold_report["reverse_loss"] > 0.0).sum()) >= 4,
    "maximum_p95_change_le_0p20": float(fold_report["p95_change"].max()) <= 0.20,
}
strict_pass = bool(all(checks.values()))

fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
pd.DataFrame(
    {
        "well": data["well"],
        "row_idx": data["row_idx"],
        "fold": data["fold"],
        "tvt_true": data["tvt_true"],
        "md_since": data["md_since"],
        "base": base,
        "psg_feature_correction": correction,
        "candidate": candidate,
    }
).to_parquet(OUT_DIR / "psg_v66_delta_oof.parquet", index=False)

verdict = {
    "experiment": "psg_v66_exact_delta_gateway_v3",
    "stage": "frozen_psg_feature_delta_on_exact_complete_v66_lineage",
    "selection_or_tuning_count": 0,
    "dose": FROZEN_DOSE,
    "sources": {"v66": str(V66_PATH), "psg": str(PSG_PATH)},
    "alignment": {
        "rows": len(data),
        "wells": int(data["well"].nunique()),
        "coverage": coverage,
        "truth_gap_max": float(truth_gap.max()),
    },
    "base": base_metrics,
    "candidate": candidate_metrics,
    "reverse_control": reverse_metrics,
    "gain": gain,
    "far_gain": far_gain,
    "folds": fold_rows,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> build exact PSG feature-inference tail for the 6.377 notebook"
        if strict_pass
        else "FAIL -> close PSG; retain untouched 6.377 mainline"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8"
)

print("PSG-V3 FOLDS")
print(fold_report.to_string(index=False), flush=True)
print("\nPSG-V3 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
