"""AutoDL audit for PSG as a complementary endpoint to fold-clean SP45.

This cell does not submit and does not treat apply_pp, strong, CP12, or Gold as
the final mainline. It aligns the five-fold PSG OOF export with the historical
fold-clean SP45 endpoint, then cross-fits one scalar blend weight: four PSG OOF
folds select the weight and the held fold receives it.

Primary metrics use exact pooled SSE, matching the competition metric. Distance
detrending is diagnostic only and is never used to select the blend weight.
"""

from pathlib import Path
import json
import math
import os
import time

import numpy as np
import pandas as pd


EXPERIMENT = "psg_sp45_crossfit_audit_v0"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

PSG_OOF_OVERRIDE = os.environ.get("PSG_OOF_PATH", "").strip()
POOL_OVERRIDE = os.environ.get("CANDIDATE_POOL_PATH", "").strip()

FUSE_CLIP_FT = 60.0
LEGACY_WEIGHT = 0.4369776249231724
DISTANCE_EDGES = np.asarray([0.0, 500.0, 1000.0, 2000.0, np.inf])
DISTANCE_LABELS = ("0_500", "500_1000", "1000_2000", "gt2000")
DETREND_BIN_FT = 250.0
BOOTSTRAPS = 2000
SEED = 20260722

MIN_ALIGNMENT = 0.995
MAX_TRUTH_GAP_FT = 0.02
MIN_CROSSFIT_WEIGHT = 0.02
MIN_NESTED_GAIN_FT = 0.05
MIN_WORST_FOLD_GAIN_FT = -0.10


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


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _first_existing(candidates, label):
    for path in candidates:
        if path and Path(path).exists():
            return Path(path)
    raise FileNotFoundError(f"Cannot find {label}. Checked: {[str(p) for p in candidates if p]}")


def _discover_psg_oof():
    candidates = []
    if PSG_OOF_OVERRIDE:
        candidates.append(Path(PSG_OOF_OVERRIDE))
    candidates.extend(
        [
            OOF_ROOT / "psg_fivefold_oof_export_v1" / "psg_oof.parquet",
            OOF_ROOT / "psg_fivefold_oof_export_v0" / "psg_oof.parquet",
            ROOT / "psg_oof.parquet",
            ROOT / "psg_kaggle_pkg" / "psg_oof.parquet",
            OOF_ROOT / "psg_fivefold_oof_export_v1" / "psg_oof.csv",
            ROOT / "psg_oof.csv",
        ]
    )
    direct = next((path for path in candidates if path.exists()), None)
    if direct is not None:
        return direct
    recursive = sorted(ROOT.glob("**/psg_oof.parquet"))
    if len(recursive) == 1:
        return recursive[0]
    if not recursive:
        recursive = sorted(ROOT.glob("**/psg_oof.csv"))
    if len(recursive) != 1:
        raise RuntimeError(
            "Set PSG_OOF_PATH explicitly; recursive PSG OOF candidates="
            + repr([str(path) for path in recursive])
        )
    return recursive[0]


def _discover_pool():
    candidates = []
    if POOL_OVERRIDE:
        candidates.append(Path(POOL_OVERRIDE))
    candidates.extend(
        [
            OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
            ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
        ]
    )
    return _first_existing(candidates, "fold-clean candidate pool")


def _read_table(path, **kwargs):
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, **kwargs)


def _rho(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return math.nan
    left = left[valid] - np.mean(left[valid])
    right = right[valid] - np.mean(right[valid])
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else math.nan


def _cosine(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else math.nan


def _optimal_weight(base_error, candidate_error):
    base_error = np.asarray(base_error, np.float64)
    direction = np.asarray(candidate_error, np.float64) - base_error
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return 0.0, 0.0
    unconstrained = float(-np.dot(base_error, direction) / denominator)
    return unconstrained, float(np.clip(unconstrained, 0.0, 1.0))


def _metrics(truth, prediction, wells):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    codes, names = pd.factorize(np.asarray(wells).astype(str), sort=False)
    counts = np.bincount(codes).astype(np.float64)
    sse = np.bincount(codes, weights=error * error, minlength=len(names))
    well_rmse = np.sqrt(np.maximum(sse, 0.0) / counts)
    return {
        "rows": int(len(error)),
        "wells": int(len(names)),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }


def _crossfit_distance_detrend(error, distance, folds):
    error = np.asarray(error, np.float64)
    distance = np.maximum(np.asarray(distance, np.float64), 0.0)
    folds = np.asarray(folds, np.int16)
    maximum = float(np.nanmax(distance))
    bins = np.floor(distance / DETREND_BIN_FT).astype(np.int32)
    bin_count = int(np.floor(maximum / DETREND_BIN_FT)) + 1
    output = np.full(len(error), np.nan, np.float64)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = folds == fold
        sums = np.bincount(bins[train], weights=error[train], minlength=bin_count)
        counts = np.bincount(bins[train], minlength=bin_count)
        means = np.full(bin_count, np.nan, np.float64)
        populated = counts > 0
        means[populated] = sums[populated] / counts[populated]
        locations = np.arange(bin_count)
        if int(populated.sum()) == 0:
            means[:] = float(np.mean(error[train]))
        elif int(populated.sum()) == 1:
            means[:] = float(means[populated][0])
        else:
            means = np.interp(locations, locations[populated], means[populated])
        if len(means) >= 3:
            padded = np.pad(means, (1, 1), mode="edge")
            means = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
        output[valid] = error[valid] - means[bins[valid]]
    return output


def _distance_report(frame, base_error, psg_error, blend_error):
    distance = frame["md_since"].to_numpy(np.float64)
    rows = []
    for index, label in enumerate(DISTANCE_LABELS):
        selected = (distance >= DISTANCE_EDGES[index]) & (distance < DISTANCE_EDGES[index + 1])
        if not int(selected.sum()):
            continue
        rows.append(
            {
                "subset": label,
                "rows": int(selected.sum()),
                "wells": int(frame.loc[selected, "well"].nunique()),
                "sp45_rmse": float(np.sqrt(np.mean(base_error[selected] ** 2))),
                "psg_rmse": float(np.sqrt(np.mean(psg_error[selected] ** 2))),
                "blend_rmse": float(np.sqrt(np.mean(blend_error[selected] ** 2))),
                "blend_gain": float(
                    np.sqrt(np.mean(base_error[selected] ** 2))
                    - np.sqrt(np.mean(blend_error[selected] ** 2))
                ),
                "raw_error_rho": _rho(base_error[selected], psg_error[selected]),
                "raw_error_cosine": _cosine(base_error[selected], psg_error[selected]),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_gain(frame, base_error, blend_error):
    codes, names = pd.factorize(frame["well"].astype(str), sort=False)
    counts = np.bincount(codes).astype(np.float64)
    base_sse = np.bincount(codes, weights=base_error * base_error, minlength=len(names))
    blend_sse = np.bincount(codes, weights=blend_error * blend_error, minlength=len(names))
    rng = np.random.default_rng(SEED)
    gains = np.empty(BOOTSTRAPS, np.float64)
    for iteration in range(BOOTSTRAPS):
        sample = rng.integers(0, len(names), len(names))
        denominator = float(np.sum(counts[sample]))
        gains[iteration] = (
            np.sqrt(np.sum(base_sse[sample]) / denominator)
            - np.sqrt(np.sum(blend_sse[sample]) / denominator)
        )
    return {
        "samples": BOOTSTRAPS,
        "gain_p05": float(np.quantile(gains, 0.05)),
        "gain_p50": float(np.quantile(gains, 0.50)),
        "gain_p95": float(np.quantile(gains, 0.95)),
        "positive_gain_probability": float(np.mean(gains > 0.0)),
    }


started = time.time()
psg_path = _discover_psg_oof()
pool_path = _discover_pool()
print("PSG-SP45 INPUTS", {"psg": str(psg_path), "pool": str(pool_path)}, flush=True)

psg = _read_table(psg_path).copy()
required_psg = {"well", "row_idx", "md_since", "psg_tvt", "true_tvt", "fold"}
missing = sorted(required_psg.difference(psg.columns))
if missing:
    raise RuntimeError(f"PSG OOF missing columns: {missing}; columns={psg.columns.tolist()}")
psg = psg[list(required_psg)].copy()
psg["well"] = psg["well"].astype(str).str.lower()
psg["row_idx"] = pd.to_numeric(psg["row_idx"], errors="raise").astype(np.int32)
psg["fold"] = pd.to_numeric(psg["fold"], errors="raise").astype(np.int8)
for column in ("md_since", "psg_tvt", "true_tvt"):
    psg[column] = pd.to_numeric(psg[column], errors="raise").astype(np.float64)
if psg.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("PSG OOF has duplicate well/row_idx keys")
fold_values = sorted(psg["fold"].unique().tolist())
if fold_values != [1, 2, 3, 4, 5]:
    raise RuntimeError(f"Expected PSG folds [1,2,3,4,5], got {fold_values}")
psg["id"] = psg["well"] + "_" + psg["row_idx"].astype(str)

header = pd.read_csv(pool_path, nrows=0).columns.tolist()
truth_column = next((name for name in ("tvt_true", "true_tvt", "TVT", "target") if name in header), None)
sp45_column = next((name for name in ("sp45", "sp45_like") if name in header), None)
if truth_column is None or sp45_column is None or "id" not in header:
    raise RuntimeError(
        f"Candidate pool needs id/truth/SP45; truth={truth_column}, sp45={sp45_column}, columns={header}"
    )
pool = pd.read_csv(
    pool_path,
    usecols=["id", truth_column, sp45_column],
    dtype={"id": str, truth_column: np.float32, sp45_column: np.float32},
)
pool["id"] = pool["id"].astype(str).str.lower()
if pool["id"].duplicated().any():
    raise RuntimeError("Candidate pool has duplicate ids")
pool = pool.rename(columns={truth_column: "pool_truth", sp45_column: "sp45"})

data = psg.merge(pool, on="id", how="left", validate="one_to_one", indicator=True)
alignment = float(data["_merge"].eq("both").mean())
if alignment < MIN_ALIGNMENT:
    examples = data.loc[data["_merge"].ne("both"), ["well", "row_idx", "id"]].head(20)
    raise RuntimeError(f"PSG/SP45 alignment {alignment:.6%} < {MIN_ALIGNMENT:.3%}\n{examples}")
data = data[data["_merge"].eq("both")].drop(columns=["_merge"]).reset_index(drop=True)
finite = np.isfinite(data[["true_tvt", "pool_truth", "sp45", "psg_tvt", "md_since"]].to_numpy())
if not bool(finite.all()):
    raise RuntimeError(f"Aligned data contains {int((~finite).sum())} non-finite values")
truth_gap = np.abs(data["true_tvt"].to_numpy() - data["pool_truth"].to_numpy())
maximum_truth_gap = float(np.max(truth_gap))
if maximum_truth_gap > MAX_TRUTH_GAP_FT:
    raise RuntimeError(
        f"Truth mismatch max={maximum_truth_gap:.6f} ft > {MAX_TRUTH_GAP_FT:.3f} ft"
    )

truth = data["true_tvt"].to_numpy(np.float64)
sp45 = data["sp45"].to_numpy(np.float64)
psg_prediction = data["psg_tvt"].to_numpy(np.float64)
wells = data["well"].to_numpy()
folds = data["fold"].to_numpy(np.int8)
sp45_error = sp45 - truth
psg_error = psg_prediction - truth
raw_direction = psg_prediction - sp45
eligible = np.abs(raw_direction) < FUSE_CLIP_FT
guarded_psg = np.where(eligible, psg_prediction, sp45)
guarded_error = guarded_psg - truth

full_unconstrained, full_weight = _optimal_weight(sp45_error, guarded_error)
legacy_prediction = sp45 + LEGACY_WEIGHT * (guarded_psg - sp45)

nested_prediction = np.full(len(data), np.nan, np.float64)
nested_weight = np.full(len(data), np.nan, np.float32)
fold_rows = []
for fold in fold_values:
    train = folds != fold
    valid = folds == fold
    unconstrained, weight = _optimal_weight(sp45_error[train], guarded_error[train])
    candidate = sp45[valid] + weight * (guarded_psg[valid] - sp45[valid])
    nested_prediction[valid] = candidate
    nested_weight[valid] = weight
    base_metrics = _metrics(truth[valid], sp45[valid], wells[valid])
    candidate_metrics = _metrics(truth[valid], psg_prediction[valid], wells[valid])
    blend_metrics = _metrics(truth[valid], candidate, wells[valid])
    fold_rows.append(
        {
            "held_fold": int(fold),
            "fit_rows": int(train.sum()),
            "held_rows": int(valid.sum()),
            "held_wells": int(pd.Series(wells[valid]).nunique()),
            "unconstrained_weight": unconstrained,
            "weight": weight,
            "sp45_rmse": base_metrics["rmse"],
            "psg_rmse": candidate_metrics["rmse"],
            "blend_rmse": blend_metrics["rmse"],
            "gain": base_metrics["rmse"] - blend_metrics["rmse"],
            "fit_error_rho": _rho(sp45_error[train], psg_error[train]),
            "held_error_rho": _rho(sp45_error[valid], psg_error[valid]),
            "held_eligible_fraction": float(np.mean(eligible[valid])),
        }
    )

if not np.isfinite(nested_prediction).all() or not np.isfinite(nested_weight).all():
    raise RuntimeError("Cross-fitted blend has missing predictions or weights")

nested_error = nested_prediction - truth
detrended_sp45 = _crossfit_distance_detrend(sp45_error, data["md_since"], folds)
detrended_psg = _crossfit_distance_detrend(psg_error, data["md_since"], folds)

summary_rows = []
for name, prediction in (
    ("sp45", sp45),
    ("psg", psg_prediction),
    ("psg_guarded60", guarded_psg),
    ("legacy_guarded_w0p436978", legacy_prediction),
    ("crossfit_guarded", nested_prediction),
):
    summary_rows.append({"variant": name, **_metrics(truth, prediction, wells)})
summary = pd.DataFrame(summary_rows)
fold_report = pd.DataFrame(fold_rows)
distance_report = _distance_report(data, sp45_error, psg_error, nested_error)
bootstrap = _bootstrap_gain(data, sp45_error, nested_error)

codes, well_names = pd.factorize(data["well"].astype(str), sort=False)
counts = np.bincount(codes).astype(np.int64)
first_position = np.full(len(well_names), len(data), np.int64)
np.minimum.at(first_position, codes, np.arange(len(data), dtype=np.int64))
bywell = pd.DataFrame(
    {
        "well": well_names,
        "fold": folds[first_position],
        "rows": counts,
        "sp45_rmse": np.sqrt(
            np.bincount(codes, weights=sp45_error**2, minlength=len(well_names)) / counts
        ),
        "psg_rmse": np.sqrt(
            np.bincount(codes, weights=psg_error**2, minlength=len(well_names)) / counts
        ),
        "blend_rmse": np.sqrt(
            np.bincount(codes, weights=nested_error**2, minlength=len(well_names)) / counts
        ),
        "mean_abs_disagreement": np.bincount(
            codes, weights=np.abs(raw_direction), minlength=len(well_names)
        )
        / counts,
    }
)
bywell["gain"] = bywell["sp45_rmse"] - bywell["blend_rmse"]
bywell = bywell.sort_values("gain", ascending=False, ignore_index=True)

sp45_metrics = summary.loc[summary["variant"].eq("sp45")].iloc[0].to_dict()
nested_metrics = summary.loc[summary["variant"].eq("crossfit_guarded")].iloc[0].to_dict()
nested_gain = float(sp45_metrics["rmse"] - nested_metrics["rmse"])

checks = {
    "alignment_ge_0p995": alignment >= MIN_ALIGNMENT,
    "truth_gap_le_0p02": maximum_truth_gap <= MAX_TRUTH_GAP_FT,
    "full_weight_positive": full_weight >= MIN_CROSSFIT_WEIGHT,
    "each_crossfit_weight_ge_0p02": bool((fold_report["weight"] >= MIN_CROSSFIT_WEIGHT).all()),
    "nested_gain_ge_0p05": nested_gain >= MIN_NESTED_GAIN_FT,
    "worst_fold_gain_ge_minus0p10": float(fold_report["gain"].min()) >= MIN_WORST_FOLD_GAIN_FT,
    "bootstrap_p05_positive": bootstrap["gain_p05"] > 0.0,
}
strict_pass = bool(all(checks.values()))

summary.to_csv(OUT_DIR / "variant_summary.csv", index=False)
fold_report.to_csv(OUT_DIR / "crossfit_fold_report.csv", index=False)
distance_report.to_csv(OUT_DIR / "distance_strata.csv", index=False)
bywell.to_csv(OUT_DIR / "per_well.csv", index=False)

export = data[["id", "well", "row_idx", "fold", "md_since", "true_tvt", "sp45", "psg_tvt"]].copy()
export["psg_guarded"] = guarded_psg.astype(np.float32)
export["crossfit_weight"] = nested_weight
export["crossfit_blend"] = nested_prediction.astype(np.float32)
export_path = OUT_DIR / "psg_sp45_crossfit_oof.parquet"
try:
    export.to_parquet(export_path, index=False)
except Exception as exc:
    export_path = OUT_DIR / "psg_sp45_crossfit_oof.pkl"
    export.to_pickle(export_path)
    print(f"Parquet export unavailable; wrote pickle instead: {exc!r}", flush=True)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_crossfit_psg_into_foldclean_sp45",
    "inputs": {"psg_oof": str(psg_path), "candidate_pool": str(pool_path)},
    "protocol": {
        "weight_fit": "four PSG OOF folds",
        "weight_application": "held PSG OOF fold",
        "objective": "exact pooled SSE",
        "fuse_clip_ft": FUSE_CLIP_FT,
        "legacy_weight_diagnostic": LEGACY_WEIGHT,
        "uses_apply_pp_as_mainline": False,
        "claims_final_6p781_oof": False,
    },
    "alignment": {
        "requested_rows": int(len(psg)),
        "aligned_rows": int(len(data)),
        "coverage": alignment,
        "wells": int(data["well"].nunique()),
        "maximum_truth_gap_ft": maximum_truth_gap,
        "guarded_row_fraction": float(np.mean(eligible)),
    },
    "diagnostics": {
        "raw_signed_error_rho": _rho(sp45_error, psg_error),
        "raw_error_cosine": _cosine(sp45_error, psg_error),
        "distance_detrended_error_rho": _rho(detrended_sp45, detrended_psg),
        "full_data_unconstrained_weight": full_unconstrained,
        "full_data_clipped_weight": full_weight,
    },
    "sp45": sp45_metrics,
    "psg": summary.loc[summary["variant"].eq("psg")].iloc[0].to_dict(),
    "legacy_blend": summary.loc[
        summary["variant"].eq("legacy_guarded_w0p436978")
    ].iloc[0].to_dict(),
    "crossfit_blend": nested_metrics,
    "crossfit_gain": nested_gain,
    "fold_weights": fold_report["weight"].tolist(),
    "fold_gains": fold_report["gain"].tolist(),
    "bootstrap": bootstrap,
    "checks": checks,
    "strict_pass": strict_pass,
    "artifact": str(export_path),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> keep PSG-SP45 as a candidate branch; next build the exact final-mainline OOF audit"
        if strict_pass
        else "FAIL -> PSG does not add stable fold-clean value to SP45; do not blend it into 6.781"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nPSG-SP45 VARIANT SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nPSG-SP45 CROSSFIT FOLDS")
print(fold_report.to_string(index=False), flush=True)
print("\nPSG-SP45 DISTANCE STRATA")
print(distance_report.to_string(index=False), flush=True)
print("\nPSG-SP45 FINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
