"""Nested observability audit for far-tail risk on the exact V33 lineage.

V43 found a valuable coherent endpoint oracle, while V44-V46 showed that its
identity cannot be recovered by candidate scoring or fixed averaging.  Before
attempting a hard-well-only endpoint, this experiment asks a narrower question:
can legal well-level evidence identify wells carrying high far-tail error mass?

This is an audit, not a deployment model.  Targets are used only for training
and held-fold evaluation.  No candidate endpoint is applied.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EXPERIMENT = "frontier_6p377_far_risk_observability_v47"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V33_PATH = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet"
V44_PATH = OOF_ROOT / "frontier_6p377_far_wholewell_selector_v44" / "well_candidate_features.parquet"
V45_PATH = OOF_ROOT / "frontier_6p377_neighbor_identity_audit_v45" / "neighbor_candidate_features.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMN = "current_dose_0p20"
FOLDS = (1, 2, 3)
FAR_START_FT = 2000.0
TOP_FRACTION = 0.20
SEED = 20260727
started = time.time()


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
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


def _spearman(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 8 or np.unique(a[valid]).size < 2 or np.unique(b[valid]).size < 2:
        return 0.0
    value = spearmanr(a[valid], b[valid]).statistic
    return float(value) if np.isfinite(value) else 0.0


def _risk_metrics(local, score, label):
    local = local.copy()
    local["score"] = np.asarray(score, np.float64)
    count = max(1, int(np.ceil(TOP_FRACTION * len(local))))
    selected = local.nlargest(count, "score")
    actual_hard = set(local.nlargest(count, "far_mse")["well"])
    selected_wells = set(selected["well"])
    total_sse = float(local["far_sse"].sum())
    selected_sse = float(selected["far_sse"].sum())
    return {
        "label": label,
        "wells": int(len(local)),
        "spearman_log_far_mse": _spearman(local["score"], local["log_far_mse"]),
        "spearman_far_rmse": _spearman(local["score"], local["far_rmse"]),
        "top20_far_sse_capture": selected_sse / max(total_sse, 1e-12),
        "top20_true_hard_recall": len(selected_wells & actual_hard) / max(len(actual_hard), 1),
        "selected_far_rmse": float(np.sqrt(selected_sse / max(selected["far_rows"].sum(), 1.0))),
        "unselected_far_rmse": float(
            np.sqrt(
                (total_sse - selected_sse)
                / max(local["far_rows"].sum() - selected["far_rows"].sum(), 1.0)
            )
        ),
    }


def _fit(train, feature_names, seed):
    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=300,
        learning_rate=0.02,
        num_leaves=7,
        max_depth=3,
        min_child_samples=24,
        subsample=0.85,
        colsample_bytree=0.65,
        reg_alpha=3.0,
        reg_lambda=20.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    weights = np.sqrt(np.maximum(train["far_rows"].to_numpy(np.float64), 1.0))
    model.fit(train[feature_names], train["log_far_mse"], sample_weight=weights)
    return model


for required in (V33_PATH, V44_PATH, V45_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

rows = pd.read_parquet(
    V33_PATH,
    columns=["well", "fold", "tvt_true", "md_since", BASE_COLUMN],
)
rows["well"] = rows["well"].astype(str)
rows = rows[rows["fold"].isin(FOLDS)].copy()
rows["error2"] = (
    rows[BASE_COLUMN].to_numpy(np.float64) - rows["tvt_true"].to_numpy(np.float64)
) ** 2

target_records = []
for well, local in rows.groupby("well", sort=False):
    far = local["md_since"].to_numpy(np.float64) > FAR_START_FT
    if far.sum() < 128:
        continue
    far_error2 = local.loc[far, "error2"].to_numpy(np.float64)
    all_error2 = local["error2"].to_numpy(np.float64)
    target_records.append({
        "well": str(well),
        "fold": int(local["fold"].iloc[0]),
        "rows": int(len(local)),
        "far_rows": int(far.sum()),
        "max_md_since_target": float(local["md_since"].max()),
        "base_sse": float(all_error2.sum()),
        "base_mse": float(all_error2.mean()),
        "far_sse": float(far_error2.sum()),
        "far_mse": float(far_error2.mean()),
        "base_rmse": float(np.sqrt(all_error2.mean())),
        "far_rmse": float(np.sqrt(far_error2.mean())),
        "far_bias": float(
            (local.loc[far, BASE_COLUMN].to_numpy(np.float64)
             - local.loc[far, "tvt_true"].to_numpy(np.float64)).mean()
        ),
    })
targets = pd.DataFrame(target_records)
targets["log_far_mse"] = np.log1p(targets["far_mse"])
if targets["well"].nunique() < 450:
    raise RuntimeError(f"Only {targets['well'].nunique()} wells have a far target")

# State zero is the deployed base.  Its V44 record contains legal complete-well
# GR/path summaries without exposing which alternative candidate wins.
v44 = pd.read_parquet(V44_PATH)
v44["well"] = v44["well"].astype(str)
v44 = v44[v44["state"].eq(0)].copy()
drop44 = {
    "fold", "state", "target_gain", "raw_gain_mse", "is_base",
    *[column for column in v44.columns if column.startswith("is_state_")],
}
rename44 = {
    column: f"v44_{column}"
    for column in v44.columns
    if column not in drop44 and column != "well"
}
v44 = v44[["well"] + list(rename44)].rename(columns=rename44)

# V45 contributes only legal Neighbor/geometry summaries for the deployed base.
v45 = pd.read_parquet(V45_PATH)
v45["well"] = v45["well"].astype(str)
v45 = v45[v45["state"].eq(0)].copy()
allowed45 = [
    column for column in v45.columns
    if column in {"geometry_available", "geometry_confidence"}
    or column.startswith("geometry_")
    or column.startswith("neighbor_")
    or column.startswith("conf_x_")
]
rename45 = {column: f"v45_{column}" for column in allowed45}
v45 = v45[["well"] + allowed45].rename(columns=rename45)

data = targets.merge(v44, on="well", how="inner", validate="one_to_one")
data = data.merge(v45, on="well", how="left", validate="one_to_one")
if data["well"].nunique() != targets["well"].nunique():
    raise RuntimeError("Feature alignment lost target wells")

excluded = {
    "well", "fold", "rows", "far_rows", "base_sse", "base_mse", "far_sse",
    "far_mse", "base_rmse", "far_rmse", "far_bias", "log_far_mse",
}
candidate_features = [column for column in data.columns if column not in excluded]
for column in candidate_features:
    data[column] = pd.to_numeric(data[column], errors="coerce")

oof_parts = []
fold_reports = []
importance_parts = []
for held_fold in FOLDS:
    train = data[~data["fold"].eq(held_fold)].copy()
    held = data[data["fold"].eq(held_fold)].copy()

    feature_names = []
    medians = {}
    for column in candidate_features:
        values = train[column].to_numpy(np.float64)
        finite = np.isfinite(values)
        if finite.mean() < 0.80:
            continue
        median = float(np.nanmedian(values[finite]))
        filled = np.where(finite, values, median)
        if np.std(filled) < 1e-9:
            continue
        medians[column] = median
        feature_names.append(column)
    if len(feature_names) < 20:
        raise RuntimeError(f"Fold {held_fold}: only {len(feature_names)} usable features")
    for column in feature_names:
        train[column] = train[column].fillna(medians[column])
        held[column] = held[column].fillna(medians[column])

    model = _fit(train, feature_names, SEED + held_fold)
    score = model.predict(held[feature_names])

    length_names = [
        column for column in feature_names
        if column in {"max_md_since_target", "v44_max_md_since", "v44_far_rows", "v44_segments"}
    ]
    if not length_names:
        length_names = ["max_md_since_target"]
    length_model = _fit(train, length_names, SEED + 100 + held_fold)
    length_score = length_model.predict(held[length_names])

    rng = np.random.default_rng(SEED + 1000 + held_fold)
    shuffled_matrix = np.array(held[feature_names].to_numpy(np.float64), copy=True)
    shuffled_matrix = shuffled_matrix[rng.permutation(len(shuffled_matrix))]
    shuffle_score = model.predict(shuffled_matrix)

    full_metrics = _risk_metrics(held, score, "legal_features")
    length_metrics = _risk_metrics(held, length_score, "length_only")
    shuffle_metrics = _risk_metrics(held, shuffle_score, "well_alignment_shuffle")
    report = {
        "fold": int(held_fold),
        "features": int(len(feature_names)),
        **{f"full_{key}": value for key, value in full_metrics.items() if key not in {"label", "wells"}},
        **{f"length_{key}": value for key, value in length_metrics.items() if key not in {"label", "wells"}},
        **{f"shuffle_{key}": value for key, value in shuffle_metrics.items() if key not in {"label", "wells"}},
        "incremental_rho_vs_length": (
            full_metrics["spearman_log_far_mse"] - length_metrics["spearman_log_far_mse"]
        ),
        "shuffle_rho_drop": (
            full_metrics["spearman_log_far_mse"] - shuffle_metrics["spearman_log_far_mse"]
        ),
    }
    fold_reports.append(report)
    print("V47 FOLD", report, flush=True)

    held_out = held[[
        "well", "fold", "far_rows", "far_sse", "far_mse", "far_rmse", "log_far_mse"
    ]].copy()
    held_out["risk_score"] = score
    held_out["length_score"] = length_score
    held_out["shuffle_score"] = shuffle_score
    oof_parts.append(held_out)

    importance_parts.append(pd.DataFrame({
        "fold": int(held_fold),
        "feature": feature_names,
        "gain": model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
    }))
    model.booster_.save_model(str(MODEL_DIR / f"risk_fold{held_fold}.txt"))

oof = pd.concat(oof_parts, ignore_index=True)
fold_table = pd.DataFrame(fold_reports)
importance = pd.concat(importance_parts, ignore_index=True)
importance_summary = (
    importance.groupby("feature", as_index=False)[["gain", "split"]].mean()
    .sort_values("gain", ascending=False)
)

overall = {
    "legal_features": _risk_metrics(oof, oof["risk_score"], "legal_features"),
    "length_only": _risk_metrics(oof, oof["length_score"], "length_only"),
    "well_alignment_shuffle": _risk_metrics(oof, oof["shuffle_score"], "well_alignment_shuffle"),
}
checks = {
    "mean_fold_rho_ge_0p25": bool(fold_table["full_spearman_log_far_mse"].mean() >= 0.25),
    "minimum_fold_rho_ge_0p10": bool(fold_table["full_spearman_log_far_mse"].min() >= 0.10),
    "mean_top20_sse_capture_ge_0p35": bool(fold_table["full_top20_far_sse_capture"].mean() >= 0.35),
    "mean_incremental_rho_vs_length_ge_0p05": bool(fold_table["incremental_rho_vs_length"].mean() >= 0.05),
    "mean_shuffle_rho_drop_ge_0p10": bool(fold_table["shuffle_rho_drop"].mean() >= 0.10),
}
strict_pass = bool(all(checks.values()))

oof.to_parquet(OUT_DIR / "far_risk_oof.parquet", index=False, compression="zstd")
fold_table.to_csv(OUT_DIR / "folds.csv", index=False)
importance.to_csv(OUT_DIR / "feature_importance_folds.csv", index=False)
importance_summary.to_csv(OUT_DIR / "feature_importance.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_far_tail_risk_observability",
    "sample_unit": "complete physical well",
    "lineage": {"base": str(V33_PATH), "base_column": BASE_COLUMN},
    "evidence": {"v44": str(V44_PATH), "v45": str(V45_PATH)},
    "wells": int(oof["well"].nunique()),
    "target": "log1p(per-well far>2000ft MSE)",
    "top_fraction": TOP_FRACTION,
    "overall": overall,
    "folds": fold_reports,
    "top_features": importance_summary.head(20).to_dict(orient="records"),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> fit a nested hard-well-only endpoint on predicted-risk wells"
        if strict_pass
        else "FAIL -> legal evidence cannot identify high-risk wells beyond length; close conditional endpoint gating"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV47 FOLDS")
print(fold_table.to_string(index=False))
print("\nV47 OVERALL")
print(pd.DataFrame(overall.values()).to_string(index=False))
print("\nV47 TOP FEATURES")
print(importance_summary.head(20).to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default))
print("saved:", OUT_DIR)
