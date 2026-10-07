"""Nested complete-well ranker for candidate-conditioned TVT-bin evidence."""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import time

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    try:
        _thread_count = int(os.environ.get(_thread_variable, "8"))
        if _thread_count <= 0:
            raise ValueError
    except (TypeError, ValueError):
        os.environ[_thread_variable] = "8"

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "candidate_tvtbin_emission_rank_audit_v1.py"
OUTPUT_ROOT = Path(os.environ.get("TVTBIN_OUTPUT_ROOT", ROOT / "output"))
OUT_DIR = OUTPUT_ROOT / "candidate_tvtbin_wholewell_ranker_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260725
DOSES = (0.0, 0.25, 0.50, 0.75, 1.0)
SCORE_PREFIXES = ("raw_cauchy", "bin0.5_", "bin1_", "bin2_", "bin4_")
BANDS = ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000")


def load_candidates() -> pd.DataFrame:
    source = SOURCE.read_text(encoding="utf-8")
    marker = 'score_columns = ["raw_cauchy"]'
    if marker not in source:
        raise RuntimeError("Candidate builder marker not found")
    namespace = {
        "__file__": str(SOURCE),
        "__name__": "candidate_tvtbin_builder_v2",
    }
    exec(compile(source.split(marker, 1)[0], str(SOURCE), "exec"), namespace)
    frame = namespace["candidates"].copy()
    if frame["well"].nunique() < 760:
        raise RuntimeError("Candidate builder returned incomplete well coverage")
    return frame


def add_features(frame: pd.DataFrame):
    frame = frame.sort_values(["well", "offset"]).reset_index(drop=True)
    excluded = {
        "fold", "rows", "sse", "mse",
        *[f"{band}_rows" for band in BANDS],
        *[f"{band}_sse" for band in BANDS],
    }
    raw_features = [
        column for column in frame.columns
        if column not in {"well", *excluded}
        and not column.startswith("shuffle_")
    ]
    evidence = [
        column for column in raw_features
        if column != "offset" and any(column.startswith(prefix) for prefix in SCORE_PREFIXES)
    ]
    features = ["offset"]
    frame["abs_offset"] = frame["offset"].abs()
    frame["offset_squared"] = frame["offset"] ** 2
    features += ["abs_offset", "offset_squared"]

    grouped = frame.groupby("well", sort=False)
    for column in evidence:
        values = frame[column].replace([np.inf, -np.inf], np.nan)
        fallback = values.groupby(frame["well"]).transform("median")
        values = values.fillna(fallback).fillna(values.median()).fillna(0.0)
        frame[column] = values
        centre = values.groupby(frame["well"]).transform("median")
        scale = values.groupby(frame["well"]).transform(
            lambda x: max(float(np.median(np.abs(x - np.median(x)))), 1e-6)
        )
        z_name = f"z__{column}"
        rank_name = f"rank__{column}"
        delta_name = f"delta0__{column}"
        frame[z_name] = np.clip((values - centre) / scale, -20.0, 20.0)
        ascending = not column.endswith("bins") and not column.endswith("coverage")
        frame[rank_name] = grouped[column].rank(method="average", pct=True, ascending=ascending)
        zero_map = frame.loc[np.isclose(frame["offset"], 0.0)].set_index("well")[column]
        frame[delta_name] = values - frame["well"].map(zero_map).to_numpy(float)
        features.extend([column, z_name, rank_name, delta_name])

    rank_columns = [name for name in features if name.startswith("rank__")]
    frame["rank_consensus_mean"] = frame[rank_columns].mean(axis=1)
    frame["rank_consensus_std"] = frame[rank_columns].std(axis=1)
    frame["rank_top_quarter_votes"] = (frame[rank_columns] <= 0.25).sum(axis=1)
    features += ["rank_consensus_mean", "rank_consensus_std", "rank_top_quarter_votes"]

    # Local profile shape tells the ranker whether a candidate is a stable basin
    # or a single-bin spike. It is computed strictly within each physical well.
    for column in [c for c in evidence if c.endswith(("mse", "cauchy", "ncc"))]:
        left = grouped[column].shift(1)
        right = grouped[column].shift(-1)
        curvature = f"curvature__{column}"
        symmetry = f"symmetry__{column}"
        frame[curvature] = (left + right - 2.0 * frame[column]).fillna(0.0)
        frame[symmetry] = (right - left).fillna(0.0)
        features += [curvature, symmetry]

    matrix = frame[features].replace([np.inf, -np.inf], np.nan)
    medians = matrix.median(axis=0).fillna(0.0)
    frame[features] = matrix.fillna(medians).astype(np.float32)
    return frame, features, evidence


def relevance(mse: np.ndarray) -> np.ndarray:
    order = np.argsort(mse, kind="mergesort")
    rank = np.empty(len(mse), np.int32)
    rank[order] = np.arange(len(mse), dtype=np.int32)
    return np.clip(30 - rank, 0, 30)


def matrix(frame, wells, features):
    subset = frame[frame["well"].isin(wells)].copy()
    subset = subset.sort_values(["well", "offset"])
    labels, groups, weights = [], [], []
    median_rows = float(subset.groupby("well")["rows"].first().median())
    for _, group in subset.groupby("well", sort=False):
        labels.append(relevance(group["mse"].to_numpy(float)))
        groups.append(len(group))
        query_weight = float(np.clip(group["rows"].iloc[0] / median_rows, 0.5, 2.0))
        weights.append(np.full(len(group), query_weight, np.float32))
    return (
        subset,
        subset[features].to_numpy(np.float32),
        np.concatenate(labels),
        np.concatenate(weights),
        groups,
    )


def model():
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=(1, 3, 5),
        label_gain=list(range(31)),
        n_estimators=1000,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=4.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def quadratic_payload(frame):
    payload = {}
    for well, group in frame.groupby("well", sort=False):
        offset = group["offset"].to_numpy(float)
        item = {"rows": int(group["rows"].iloc[0])}
        for key in ("", *BANDS):
            sse_column = "sse" if not key else f"{key}_sse"
            rows_column = "rows" if not key else f"{key}_rows"
            count = int(group[rows_column].iloc[0])
            if count:
                coefficients = np.polyfit(offset, group[sse_column].to_numpy(float), 2)
            else:
                coefficients = np.zeros(3, float)
            item[key or "all"] = (count, coefficients)
        payload[well] = item
    return payload


def exact_sse(payload, offsets, dose, key="all"):
    total_sse = 0.0
    total_rows = 0
    per_well = []
    for well, offset in offsets.items():
        rows, coefficients = payload[well][key]
        value = float(np.polyval(coefficients, dose * float(offset)))
        value = max(value, 0.0)
        total_sse += value
        total_rows += rows
        if key == "all" and rows:
            per_well.append(math.sqrt(value / rows))
    result = {
        "rows": int(total_rows),
        "exact_sse": float(total_sse),
        "rmse": math.sqrt(total_sse / total_rows) if total_rows else float("nan"),
    }
    if key == "all":
        result.update(
            median_well_rmse=float(np.median(per_well)),
            p75_well_rmse=float(np.quantile(per_well, 0.75)),
            p95_well_rmse=float(np.quantile(per_well, 0.95)),
            max_well_rmse=float(np.max(per_well)),
        )
    return result


def predict_offsets(fitted, subset, features, control=False):
    values = subset[features].copy()
    if control:
        evidence_columns = [
            column for column in features
            if column != "offset" and (
                any(token in column for token in ("bin0.5_", "bin1_", "bin2_", "bin4_", "raw_cauchy"))
                or column.startswith("rank_consensus")
                or column == "rank_top_quarter_votes"
            )
        ]
        # Shift all candidate-conditioned evidence by seven states. Offset priors
        # and complete-well composition remain unchanged.
        for _, indices in subset.groupby("well", sort=False).groups.items():
            positions = subset.index.get_indexer(indices)
            values.iloc[positions, values.columns.get_indexer(evidence_columns)] = np.roll(
                values.iloc[positions][evidence_columns].to_numpy(), 7, axis=0
            )
    scores = fitted.predict(values.to_numpy(np.float32), num_iteration=fitted.best_iteration_)
    scored = subset[["well", "offset", "mse"]].copy()
    scored["score"] = scores
    chosen = scored.loc[scored.groupby("well")["score"].idxmax()]
    return dict(zip(chosen["well"], chosen["offset"])), scored


started = time.time()
candidates, feature_names, evidence_names = add_features(load_candidates())
payload = quadratic_payload(candidates)
fold_by_well = candidates.groupby("well")["fold"].first().astype(int).to_dict()
all_wells = sorted(fold_by_well)

oof_offsets = {}
oof_control_offsets = {}
fold_reports = []
feature_importance = []
for outer_fold in range(1, 6):
    inner_fold = outer_fold % 5 + 1
    train_wells = [w for w in all_wells if fold_by_well[w] not in (outer_fold, inner_fold)]
    inner_wells = [w for w in all_wells if fold_by_well[w] == inner_fold]
    outer_wells = [w for w in all_wells if fold_by_well[w] == outer_fold]
    train_frame, x_train, y_train, w_train, g_train = matrix(candidates, train_wells, feature_names)
    inner_frame, x_inner, y_inner, w_inner, g_inner = matrix(candidates, inner_wells, feature_names)
    fitted = model()
    fitted.fit(
        x_train, y_train, sample_weight=w_train, group=g_train,
        eval_set=[(x_inner, y_inner)], eval_group=[g_inner],
        eval_sample_weight=[w_inner],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    inner_offsets, _ = predict_offsets(fitted, inner_frame, feature_names)
    dose_rows = []
    for dose in DOSES:
        metrics = exact_sse(payload, inner_offsets, dose)
        dose_rows.append({"dose": dose, "rmse": metrics["rmse"]})
    selected_dose = min(dose_rows, key=lambda row: (row["rmse"], row["dose"]))["dose"]

    outer_frame = candidates[candidates["well"].isin(outer_wells)].sort_values(["well", "offset"])
    offsets, scored = predict_offsets(fitted, outer_frame, feature_names)
    control_offsets, _ = predict_offsets(fitted, outer_frame, feature_names, control=True)
    oof_offsets.update({well: selected_dose * value for well, value in offsets.items()})
    oof_control_offsets.update({well: selected_dose * value for well, value in control_offsets.items()})
    report = exact_sse(payload, {w: offsets[w] for w in outer_wells}, selected_dose)
    report.update(
        fold=outer_fold,
        inner_fold=inner_fold,
        selected_dose=selected_dose,
        best_iteration=int(fitted.best_iteration_),
        wells=len(outer_wells),
    )
    fold_reports.append(report)
    feature_importance.append(pd.DataFrame({
        "fold": outer_fold,
        "feature": feature_names,
        "gain": fitted.booster_.feature_importance(importance_type="gain"),
    }))
    print("TVTBIN RANKER FOLD", report, flush=True)

base_offsets = {well: 0.0 for well in all_wells}
base = exact_sse(payload, base_offsets, 1.0)
full = exact_sse(payload, oof_offsets, 1.0)
control = exact_sse(payload, oof_control_offsets, 1.0)
for band in BANDS:
    full[f"{band}_rmse"] = exact_sse(payload, oof_offsets, 1.0, band)["rmse"]
    base[f"{band}_rmse"] = exact_sse(payload, base_offsets, 1.0, band)["rmse"]
    control[f"{band}_rmse"] = exact_sse(payload, oof_control_offsets, 1.0, band)["rmse"]

selection = pd.DataFrame({
    "well": all_wells,
    "fold": [fold_by_well[w] for w in all_wells],
    "selected_offset_ft": [oof_offsets[w] for w in all_wells],
    "control_offset_ft": [oof_control_offsets[w] for w in all_wells],
})
selection.to_csv(OUT_DIR / "oof_selected_offsets.csv", index=False)
pd.DataFrame(fold_reports).to_csv(OUT_DIR / "fold_reports.csv", index=False)
importance = pd.concat(feature_importance, ignore_index=True)
importance.groupby("feature", as_index=False)["gain"].mean().sort_values("gain", ascending=False).to_csv(
    OUT_DIR / "feature_importance.csv", index=False
)

fold_positive = sum(report["rmse"] < exact_sse(
    payload, {w: 0.0 for w in all_wells if fold_by_well[w] == report["fold"]}, 1.0
)["rmse"] for report in fold_reports)
verdict = {
    "experiment": "candidate_tvtbin_wholewell_ranker_v2",
    "stage": "fivefold_nested_complete_well_candidate_lambdarank",
    "sample_unit": "one complete physical well",
    "features": len(feature_names),
    "wells": len(all_wells),
    "base": base,
    "ranker": full,
    "candidate_alignment_shuffle": control,
    "gain": base["rmse"] - full["rmse"],
    "shuffle_loss": control["rmse"] - full["rmse"],
    "positive_folds": int(fold_positive),
    "folds": fold_reports,
    "checks": {
        "gain_ge_0p30": bool(base["rmse"] - full["rmse"] >= 0.30),
        "positive_folds_ge_4": bool(fold_positive >= 4),
        "shuffle_loses_70pct_gain": bool(control["rmse"] - full["rmse"] >= 0.70 * max(base["rmse"] - full["rmse"], 0.0)),
        "far_improves": bool(full["far_gt2000_rmse"] < base["far_gt2000_rmse"]),
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["checks"].values()))
verdict["next"] = (
    "PASS -> export the nested scorer recipe and test against the strongest endpoint"
    if verdict["strict_pass"]
    else "FAIL -> close static TVT-bin candidate ranking and move to sequence evidence"
)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR)
