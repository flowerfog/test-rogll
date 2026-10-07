"""V225: test whether legal typewell fidelity predicts future model reliability.

The future TVT is used only for audit labels and endpoint metrics.  Every model
feature is available at inference: visible-prefix TVT/GR, complete horizontal
GR, typewell GR, geometry, and the frozen V168 prediction.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF = ROOT / "oof_exports"
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
PARENT_PATH = OOF / "historical_analog_metadata_fullrow_v168/fullrow_oof.parquet"
ALT_PATH = OOF / "likpf_residual_foldclean_spatial_v217/oof.parquet"
FOLD_PATH = OOF / "likpf_residual_gbdt_full5_v215/well_metadata.csv"
OUT = OOF / "typewell_fidelity_observability_v225"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 225
PREFIX_WINDOWS = (128.0, 256.0, 512.0, 1024.0)


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


def _corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 12 or np.std(a[valid]) < 1e-8 or np.std(b[valid]) < 1e-8:
        return 0.0
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _autocorr(values, lag):
    values = np.asarray(values, float)
    if len(values) <= lag + 8:
        return 0.0
    return _corr(values[:-lag], values[lag:])


def _robust_affine(reference, observed):
    x = np.asarray(reference, float)
    y = np.asarray(observed, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 16:
        return 1.0, 0.0
    for _ in range(4):
        design = np.column_stack((x[keep], np.ones(keep.sum())))
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        centre = np.nanmedian(residual[keep])
        scale = 1.4826 * np.nanmedian(np.abs(residual[keep] - centre)) + 1e-6
        updated = np.isfinite(x) & np.isfinite(y) & (np.abs(residual - centre) <= 3.5 * scale)
        if updated.sum() < 16 or np.array_equal(updated, keep):
            break
        keep = updated
    return float(slope), float(intercept)


def _local_variance(typewell_tvt, typewell_gr, positions, radius=32.0):
    positions = np.asarray(positions, float)
    output = np.full(len(positions), np.nan, float)
    for index, position in enumerate(positions):
        left = np.searchsorted(typewell_tvt, position - radius, side="left")
        right = np.searchsorted(typewell_tvt, position + radius, side="right")
        if right - left >= 8:
            output[index] = np.std(typewell_gr[left:right])
    return output


for required in (DATA_ROOT / "train", PARENT_PATH, ALT_PATH, FOLD_PATH):
    if not required.exists():
        raise FileNotFoundError(required)

folds = pd.read_csv(FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
if len(folds) != 773:
    raise RuntimeError(f"Expected 773 wells, found {len(folds)}")


def _canonical_endpoint(path, prediction_candidates, output_name):
    frame = pd.read_parquet(path)
    if "id" not in frame:
        well_column = next(name for name in ("well", "well_id") if name in frame)
        row_column = next(name for name in ("row_idx", "row_index", "_row_idx") if name in frame)
        frame["id"] = frame[well_column].astype(str) + "_" + frame[row_column].astype(int).astype(str)
    prediction = next(name for name in prediction_candidates if name in frame)
    truth = next((name for name in ("tvt_true", "TVT", "truth") if name in frame), None)
    columns = ["id", prediction] + ([truth] if truth else [])
    result = frame[columns].copy().rename(columns={prediction: output_name})
    if truth:
        result = result.rename(columns={truth: "endpoint_truth"})
    if result["id"].duplicated().any():
        raise RuntimeError(f"Duplicate ids in {path}")
    return result


parent = _canonical_endpoint(PARENT_PATH, ("metadata_analog_w0p05",), "v168")
alternative = _canonical_endpoint(ALT_PATH, ("candidate", "v217"), "v217")
endpoint = parent.merge(alternative.drop(columns=["endpoint_truth"], errors="ignore"), on="id", validate="one_to_one")
endpoint_lookup = endpoint.set_index("id")


def _one_well(well, fold):
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path, usecols=["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"])
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"]).sort_values("TVT")
    md = horizontal["MD"].to_numpy(float)
    gr = horizontal["GR"].to_numpy(float)
    tvt = horizontal["TVT"].to_numpy(float)
    tvt_input = horizontal["TVT_input"].to_numpy(float)
    known = np.isfinite(tvt_input) & np.isfinite(gr)
    hidden = ~np.isfinite(tvt_input) & np.isfinite(gr) & np.isfinite(tvt)
    if known.sum() < 64 or hidden.sum() < 128:
        raise RuntimeError(f"{well}: invalid prefix/eval coverage")
    anchor = int(np.flatnonzero(known)[-1])
    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr = typewell["GR"].to_numpy(float)
    tw_prefix = np.interp(tvt_input[known], tw_tvt, tw_gr)
    slope, intercept = _robust_affine(tw_prefix, gr[known])
    fitted_prefix = slope * tw_prefix + intercept
    prefix_scale = max(float(np.std(gr[known])), 1.0)
    prefix_residual = gr[known] - fitted_prefix

    row = {"well": str(well), "fold": int(fold), "rows": int(hidden.sum())}
    row["prefix_corr_all"] = _corr(gr[known], fitted_prefix)
    row["prefix_nrmse_all"] = _rmse(gr[known], fitted_prefix) / prefix_scale
    row["prefix_resid_ac16"] = _autocorr(prefix_residual, 16)
    row["prefix_resid_ac64"] = _autocorr(prefix_residual, 64)
    row["prefix_scale_ratio"] = float(slope)

    distance_to_anchor = md[anchor] - md
    bin_corr = []
    bin_centres = []
    for window in PREFIX_WINDOWS:
        selected = known & (distance_to_anchor >= 0.0) & (distance_to_anchor <= window)
        reference = np.interp(tvt_input[selected], tw_tvt, tw_gr)
        local_slope, local_intercept = _robust_affine(reference, gr[selected])
        fitted = local_slope * reference + local_intercept
        scale = max(float(np.std(gr[selected])), 1.0)
        row[f"prefix_corr_{int(window)}"] = _corr(gr[selected], fitted)
        row[f"prefix_nrmse_{int(window)}"] = _rmse(gr[selected], fitted) / scale
        row[f"prefix_variance_ratio_{int(window)}"] = float(np.std(reference) / scale)
    edges = np.linspace(0.0, 1024.0, 5)
    for left, right in zip(edges[:-1], edges[1:]):
        selected = known & (distance_to_anchor >= left) & (distance_to_anchor < right)
        if selected.sum() >= 24:
            reference = np.interp(tvt_input[selected], tw_tvt, tw_gr)
            bin_corr.append(_corr(gr[selected], reference))
            bin_centres.append((left + right) / 2.0)
    row["prefix_corr_slope"] = float(np.polyfit(bin_centres, bin_corr, 1)[0]) if len(bin_corr) >= 3 else 0.0
    row["prefix_corr_range"] = float(np.ptp(bin_corr)) if bin_corr else 0.0

    hidden_rows = np.flatnonzero(hidden)
    ids = [f"{well}_{index}" for index in hidden_rows]
    aligned = endpoint_lookup.reindex(ids)
    if aligned[["v168", "v217"]].isna().any().any():
        raise RuntimeError(f"{well}: endpoint alignment failed")
    v168 = aligned["v168"].to_numpy(float)
    v217 = aligned["v217"].to_numpy(float)
    truth = tvt[hidden_rows]
    eval_gr = gr[hidden_rows]

    # Legal candidate-conditioned fidelity: query the typewell at frozen V168.
    tw_candidate = np.interp(v168, tw_tvt, tw_gr)
    fitted_candidate = slope * tw_candidate + intercept
    candidate_residual = eval_gr - fitted_candidate
    eval_scale = max(float(np.std(eval_gr)), 1.0)
    row["candidate_corr"] = _corr(eval_gr, fitted_candidate)
    row["candidate_nrmse"] = _rmse(eval_gr, fitted_candidate) / eval_scale
    row["candidate_resid_ac16"] = _autocorr(candidate_residual, 16)
    row["candidate_resid_ac64"] = _autocorr(candidate_residual, 64)
    local_std = _local_variance(tw_tvt, tw_gr, v168)
    row["candidate_tw_std_median"] = float(np.nanmedian(local_std))
    row["candidate_tw_std_p10"] = float(np.nanquantile(local_std, 0.10))
    row["candidate_tw_flat_fraction"] = float(np.nanmean(local_std <= 5.0))
    row["eval_horizontal_std"] = float(np.std(eval_gr))

    # Audit-only labels below this line.
    tw_truth = np.interp(truth, tw_tvt, tw_gr)
    fitted_truth = slope * tw_truth + intercept
    row["audit_eval_corr_truth"] = _corr(eval_gr, fitted_truth)
    row["audit_eval_nrmse_truth"] = _rmse(eval_gr, fitted_truth) / eval_scale
    parent_error = v168 - truth
    move = v217 - v168
    denominator = float(np.dot(move, move))
    alpha = -float(np.dot(parent_error, move)) / denominator if denominator > 1e-9 else 0.0
    row["audit_alpha_oracle"] = float(np.clip(alpha, 0.0, 1.0))
    row["audit_v168_sse"] = float(np.dot(parent_error, parent_error))
    row["audit_v217_sse"] = float(np.dot(v217 - truth, v217 - truth))
    return row


print("V225 fidelity extraction", {"wells": len(folds)}, flush=True)
reports = Parallel(n_jobs=8, prefer="threads")(
    delayed(_one_well)(str(row.well), int(row.fold)) for row in folds.itertuples(index=False)
)
well_table = pd.DataFrame(reports).sort_values("well").reset_index(drop=True)

feature_columns = [name for name in well_table if not name.startswith("audit_") and name not in {"well", "fold", "rows"}]
targets = ("audit_eval_corr_truth", "audit_eval_nrmse_truth", "audit_alpha_oracle")
associations = []
for target in targets:
    for feature in feature_columns:
        x = well_table[feature].to_numpy(float)
        y = well_table[target].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        associations.append({
            "target": target, "feature": feature, "n": int(valid.sum()),
            "pearson": float(pearsonr(x[valid], y[valid])[0]) if valid.sum() >= 20 else np.nan,
            "spearman": float(spearmanr(x[valid], y[valid])[0]) if valid.sum() >= 20 else np.nan,
        })
association_table = pd.DataFrame(associations)
association_table["abs_spearman"] = association_table["spearman"].abs()

# Frozen fivefold well-level soft router.  This tests observability, not a tuned endpoint.
predicted_alpha = np.zeros(len(well_table), float)
shuffled_alpha = np.zeros(len(well_table), float)
x_all = well_table[feature_columns].replace([np.inf, -np.inf], np.nan)
y_all = well_table["audit_alpha_oracle"].to_numpy(float)
rng = np.random.default_rng(SEED)
for held_fold in range(1, 6):
    train = ~well_table["fold"].eq(held_fold).to_numpy()
    valid = ~train
    medians = x_all.loc[train].median(axis=0)
    x_train = x_all.loc[train].fillna(medians).fillna(0.0).to_numpy(np.float32)
    x_valid = x_all.loc[valid].fillna(medians).fillna(0.0).to_numpy(np.float32)
    params = dict(n_estimators=600, min_samples_leaf=12, max_features=0.75,
                  n_jobs=-1, random_state=SEED + held_fold)
    model = ExtraTreesRegressor(**params)
    model.fit(x_train, y_all[train], sample_weight=well_table.loc[train, "rows"].to_numpy(float))
    predicted_alpha[valid] = np.clip(model.predict(x_valid), 0.0, 1.0)
    control = ExtraTreesRegressor(**params)
    control.fit(x_train, rng.permutation(y_all[train]), sample_weight=well_table.loc[train, "rows"].to_numpy(float))
    shuffled_alpha[valid] = np.clip(control.predict(x_valid), 0.0, 1.0)

alpha_by_well = dict(zip(well_table["well"], predicted_alpha))
shuffle_by_well = dict(zip(well_table["well"], shuffled_alpha))
fold_by_well = dict(zip(well_table["well"], well_table["fold"]))
row_parts = []
for row in reports:
    well = str(row["well"])
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv", usecols=["TVT", "TVT_input"])
    hidden_rows = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    ids = [f"{well}_{index}" for index in hidden_rows]
    aligned = endpoint_lookup.reindex(ids)
    truth = horizontal["TVT"].to_numpy(float)[hidden_rows]
    v168 = aligned["v168"].to_numpy(float)
    v217 = aligned["v217"].to_numpy(float)
    move = v217 - v168
    row_parts.append(pd.DataFrame({
        "well": well, "fold": int(fold_by_well[well]), "tvt_true": truth,
        "v168": v168, "v217": v217,
        "fidelity_router": v168 + alpha_by_well[well] * move,
        "shuffled_router": v168 + shuffle_by_well[well] * move,
        "oracle_router": v168 + float(row["audit_alpha_oracle"]) * move,
    }))
oof = pd.concat(row_parts, ignore_index=True)

summary = []
for name in ("v168", "v217", "fidelity_router", "shuffled_router", "oracle_router"):
    summary.append({"endpoint": name, "rmse": _rmse(oof["tvt_true"], oof[name])})
summary = pd.DataFrame(summary)
fold_rows = []
for fold, group in oof.groupby("fold"):
    base_rmse = _rmse(group["tvt_true"], group["v168"])
    candidate_rmse = _rmse(group["tvt_true"], group["fidelity_router"])
    fold_rows.append({"fold": int(fold), "base_rmse": base_rmse,
                      "candidate_rmse": candidate_rmse, "gain": base_rmse - candidate_rmse})

base_rmse = float(summary.loc[summary["endpoint"].eq("v168"), "rmse"].iloc[0])
router_rmse = float(summary.loc[summary["endpoint"].eq("fidelity_router"), "rmse"].iloc[0])
shuffle_rmse = float(summary.loc[summary["endpoint"].eq("shuffled_router"), "rmse"].iloc[0])
alpha_spearman = float(spearmanr(predicted_alpha, y_all)[0])
best_future_fidelity = association_table.loc[
    association_table["target"].isin(["audit_eval_corr_truth", "audit_eval_nrmse_truth"])
].sort_values("abs_spearman", ascending=False).iloc[0]
checks = {
    "prefix_predicts_future_fidelity_abs_spearman_ge_0p20": float(best_future_fidelity["abs_spearman"]) >= 0.20,
    "alpha_spearman_ge_0p15": alpha_spearman >= 0.15,
    "router_gain_ge_0p03": base_rmse - router_rmse >= 0.03,
    "aligned_beats_shuffle_ge_0p02": shuffle_rmse - router_rmse >= 0.02,
    "positive_folds_ge_4": sum(row["gain"] > 0 for row in fold_rows) >= 4,
}
verdict = {
    "experiment": "typewell_fidelity_observability_v225",
    "stage": "fivefold_legal_prefix_to_future_fidelity_and_soft_route_gate",
    "features": feature_columns,
    "best_future_fidelity_association": best_future_fidelity.to_dict(),
    "alpha_spearman": alpha_spearman,
    "mean_predicted_alpha": float(np.mean(predicted_alpha)),
    "summary": summary.to_dict("records"),
    "folds": fold_rows,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> add fidelity profile to candidate-conditioned model" if all(checks.values())
            else "FAIL -> future typewell reliability is not legally observable; close fidelity routing",
}

well_table["predicted_alpha"] = predicted_alpha
well_table["shuffled_alpha"] = shuffled_alpha
well_table.to_csv(OUT / "well_fidelity.csv", index=False)
association_table.sort_values(["target", "abs_spearman"], ascending=[True, False]).to_csv(OUT / "associations.csv", index=False)
summary.to_csv(OUT / "summary.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT / "folds.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")

print("\nV225 ASSOCIATIONS TOP")
print(association_table.sort_values("abs_spearman", ascending=False).head(25).to_string(index=False))
print("\nV225 SUMMARY")
print(summary.to_string(index=False))
print("\nV225 FOLDS")
print(pd.DataFrame(fold_rows).to_string(index=False))
print("\nV225 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default))
print("saved:", OUT)
