"""Risk-gated far endpoint on the exact V33 / LB 6.377 lineage.

V47 established that legal evidence predicts far-tail risk.  This experiment
tests whether that risk signal can safely gate one robust alternative endpoint.
For each outer fold, the alternative endpoint and dose are selected only on
the other folds' truly hard wells.  The held fold is gated exclusively by its
V47 out-of-fold risk score.  Candidate identity is never predicted per well.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPERIMENT = "frontier_6p377_risk_gated_endpoint_v48"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V33_PATH = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet"
V28_PATH = OOF_ROOT / "frontier_6p39_tail_endpoint_oracle_v28" / "tail_endpoint_oof.parquet"
V47_PATH = OOF_ROOT / "frontier_6p377_far_risk_observability_v47" / "far_risk_oof.parquet"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMN = "current_dose_0p20"
FOLDS = (1, 2, 3)
FAR_START_FT = 2000.0
FADE_FT = 250.0
CAP_FT = 24.0
SMOOTH_FT = 128.0
RISK_FRACTION = 0.20
DOSES = (0.25, 0.50, 0.75, 1.00)
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


def _smoothstep(values):
    t = np.clip((np.asarray(values, np.float64) - FAR_START_FT) / FADE_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _centred_mean(x, values, width):
    x = np.asarray(x, np.float64)
    values = np.asarray(values, np.float64)
    left = np.searchsorted(x, x - width / 2.0, side="left")
    right = np.searchsorted(x, x + width / 2.0, side="right")
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    return (prefix[right] - prefix[left]) / np.maximum(right - left, 1)


def _metrics(local, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - local["tvt_true"].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    wells = local["well"].astype(str).to_numpy()
    per_well = pd.Series(error**2).groupby(wells, sort=False).mean().pow(0.5)
    far = distance > FAR_START_FT
    near = distance <= 1000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(per_well.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(per_well.median()),
        "p75_well_rmse": float(per_well.quantile(0.75)),
        "p95_well_rmse": float(per_well.quantile(0.95)),
        "max_well_rmse": float(per_well.max()),
        "near_le1000_rmse": float(np.sqrt(np.mean(error[near] ** 2))),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def _top_wells(table, score_column, fraction):
    count = max(1, int(np.ceil(float(fraction) * len(table))))
    return set(table.nlargest(count, score_column)["well"].astype(str))


for required in (V33_PATH, V28_PATH, V47_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

columns = ["id", "well", "row_idx", "fold", "tvt_true", "md_since", "pre_neighbor_cp8", BASE_COLUMN]
frame = pd.read_parquet(V33_PATH, columns=columns)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
frame = frame[frame["fold"].isin(FOLDS)].sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

risk = pd.read_parquet(V47_PATH)
risk["well"] = risk["well"].astype(str)
risk = risk[risk["fold"].isin(FOLDS)].drop_duplicates("well", keep="last")
required_risk = {"well", "fold", "far_mse", "far_sse", "far_rows", "risk_score"}
if not required_risk.issubset(risk.columns):
    raise RuntimeError(f"V47 columns missing: {sorted(required_risk.difference(risk.columns))}")

# Match V47's risk-observable well population exactly.
frame = frame[frame["well"].isin(set(risk["well"]))].reset_index(drop=True)
if frame["well"].nunique() != risk["well"].nunique():
    raise RuntimeError("V33/V47 well alignment failed")

tail_columns = sorted(column for column in pq.ParquetFile(V28_PATH).schema.names if column.startswith("tail_"))
tail = pd.read_parquet(V28_PATH, columns=["id"] + tail_columns)
tail["id"] = tail["id"].astype(str)
frame = frame.merge(tail, on="id", how="left", validate="one_to_one")

base = frame[BASE_COLUMN].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fold_values = frame["fold"].to_numpy(np.int8)
fade = _smoothstep(distance)
far = distance > FAR_START_FT
near = distance <= 1000.0

raw_paths = [base]
candidate_names = ["base_6p377"]
for name in ["pre_neighbor_cp8"] + tail_columns:
    absolute = frame[name].to_numpy(np.float64)
    finite = np.isfinite(absolute)
    if finite.mean() < 0.95:
        continue
    correction = np.where(finite, np.clip(absolute - base, -CAP_FT, CAP_FT), 0.0)
    candidate = base + fade * correction
    candidate[~far] = base[~far]
    raw_paths.append(candidate)
    candidate_names.append(name)
candidates = np.column_stack(raw_paths)

groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}
for indices in groups.values():
    local_distance = distance[indices]
    for state in range(1, candidates.shape[1]):
        correction = _centred_mean(
            local_distance,
            candidates[indices, state] - base[indices],
            SMOOTH_FT,
        )
        correction = np.clip(correction, -CAP_FT, CAP_FT)
        candidate = base[indices] + fade[indices] * correction
        candidate[local_distance <= FAR_START_FT] = base[indices][local_distance <= FAR_START_FT]
        candidates[indices, state] = candidate
if not np.isfinite(candidates).all():
    raise RuntimeError("Non-finite candidate bank")

oof_prediction = base.copy()
shuffle_prediction = base.copy()
oracle_gate_prediction = base.copy()
fold_reports = []
selection_rows = []

for held_fold in FOLDS:
    train_risk = risk[~risk["fold"].eq(held_fold)].copy()
    held_risk = risk[risk["fold"].eq(held_fold)].copy()
    train_hard = _top_wells(train_risk, "far_mse", RISK_FRACTION)
    held_predicted_hard = _top_wells(held_risk, "risk_score", RISK_FRACTION)
    held_oracle_hard = _top_wells(held_risk, "far_mse", RISK_FRACTION)

    rng = np.random.default_rng(SEED + held_fold)
    shuffled_scores = held_risk["risk_score"].to_numpy(np.float64)[rng.permutation(len(held_risk))]
    held_risk_shuffle = held_risk.assign(shuffle_score=shuffled_scores)
    held_shuffled_hard = _top_wells(held_risk_shuffle, "shuffle_score", RISK_FRACTION)

    train_well_mask = frame["well"].isin(train_hard).to_numpy()
    train_row_mask = (~np.equal(fold_values, held_fold)) & train_well_mask & far
    if train_row_mask.sum() < 1000:
        raise RuntimeError(f"Fold {held_fold}: too few hard-well training rows")

    scan = []
    base_sse = float(np.sum((base[train_row_mask] - truth[train_row_mask]) ** 2))
    for state in range(candidates.shape[1]):
        for dose in ((0.0,) if state == 0 else DOSES):
            prediction = base[train_row_mask] + float(dose) * (
                candidates[train_row_mask, state] - base[train_row_mask]
            )
            sse = float(np.sum((prediction - truth[train_row_mask]) ** 2))
            scan.append({
                "held_fold": int(held_fold),
                "state": int(state),
                "candidate": candidate_names[state],
                "dose": float(dose),
                "train_hard_wells": int(len(train_hard)),
                "train_hard_rows": int(train_row_mask.sum()),
                "train_gain_sse": base_sse - sse,
                "train_rmse": float(np.sqrt(sse / train_row_mask.sum())),
            })
    scan_table = pd.DataFrame(scan).sort_values(
        ["train_gain_sse", "dose"], ascending=[False, True]
    ).reset_index(drop=True)
    selected = scan_table.iloc[0]
    selection_rows.append(scan_table)
    state = int(selected["state"])
    dose = float(selected["dose"])

    held = frame["fold"].eq(held_fold).to_numpy()
    predicted_gate = held & frame["well"].isin(held_predicted_hard).to_numpy() & far
    shuffled_gate = held & frame["well"].isin(held_shuffled_hard).to_numpy() & far
    oracle_gate = held & frame["well"].isin(held_oracle_hard).to_numpy() & far
    candidate_delta = candidates[:, state] - base
    oof_prediction[predicted_gate] = base[predicted_gate] + dose * candidate_delta[predicted_gate]
    shuffle_prediction[shuffled_gate] = base[shuffled_gate] + dose * candidate_delta[shuffled_gate]
    oracle_gate_prediction[oracle_gate] = base[oracle_gate] + dose * candidate_delta[oracle_gate]

    held_frame = frame.loc[held].reset_index(drop=True)
    held_base = base[held]
    held_pred = oof_prediction[held]
    held_shuffle = shuffle_prediction[held]
    held_oracle = oracle_gate_prediction[held]
    base_metric = _metrics(held_frame, held_base, "base")
    metric = _metrics(held_frame, held_pred, "risk_gate")
    shuffle_metric = _metrics(held_frame, held_shuffle, "risk_shuffle")
    oracle_metric = _metrics(held_frame, held_oracle, "oracle_risk_gate")
    report = {
        "fold": int(held_fold),
        "selected_candidate": candidate_names[state],
        "selected_dose": dose,
        "train_hard_wells": int(len(train_hard)),
        "held_predicted_hard_wells": int(len(held_predicted_hard)),
        "risk_gate_overlap_oracle": float(
            len(held_predicted_hard & held_oracle_hard) / max(len(held_oracle_hard), 1)
        ),
        "base_rmse": base_metric["rmse"],
        "rmse": metric["rmse"],
        "gain": base_metric["rmse"] - metric["rmse"],
        "base_far_rmse": base_metric["far_gt2000_rmse"],
        "far_rmse": metric["far_gt2000_rmse"],
        "far_gain": base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
        "p95_change": metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
        "shuffle_rmse": shuffle_metric["rmse"],
        "shuffle_loss": shuffle_metric["rmse"] - metric["rmse"],
        "oracle_gate_rmse": oracle_metric["rmse"],
        "oracle_gate_gain": base_metric["rmse"] - oracle_metric["rmse"],
    }
    fold_reports.append(report)
    print("V48 FOLD", report, flush=True)

base_result = _metrics(frame, base, "base_6p377")
result = _metrics(frame, oof_prediction, "nested_risk_gated_endpoint")
shuffle_result = _metrics(frame, shuffle_prediction, "risk_alignment_shuffle")
oracle_result = _metrics(frame, oracle_gate_prediction, "oracle_hardness_gate")
gain = base_result["rmse"] - result["rmse"]
far_gain = base_result["far_gt2000_rmse"] - result["far_gt2000_rmse"]
shuffle_loss = shuffle_result["rmse"] - result["rmse"]
fold_table = pd.DataFrame(fold_reports)
checks = {
    "gain_ge_0p05": bool(gain >= 0.05),
    "far_gain_ge_0p08": bool(far_gain >= 0.08),
    "positive_folds_ge_2": bool((fold_table["gain"] > 0).sum() >= 2),
    "minimum_fold_gain_ge_minus0p03": bool(fold_table["gain"].min() >= -0.03),
    "shuffle_loses_70pct_gain": bool(shuffle_loss >= 0.70 * max(gain, 0.0)),
    "maximum_fold_p95_change_le_0p15": bool(fold_table["p95_change"].max() <= 0.15),
    "near_is_bitwise_frozen": bool(
        np.array_equal(oof_prediction[near], base[near])
        and np.array_equal(shuffle_prediction[near], base[near])
    ),
}
strict_pass = bool(all(checks.values()))

pd.concat(selection_rows, ignore_index=True).to_csv(OUT_DIR / "training_scans.csv", index=False)
fold_table.to_csv(OUT_DIR / "folds.csv", index=False)
frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].assign(
    base_6p377=base.astype(np.float32),
    nested_risk_gated_endpoint=oof_prediction.astype(np.float32),
    risk_alignment_shuffle=shuffle_prediction.astype(np.float32),
    oracle_hardness_gate=oracle_gate_prediction.astype(np.float32),
).to_parquet(OUT_DIR / "risk_gated_endpoint_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "threefold_nested_risk_gated_robust_far_endpoint",
    "sample_unit": "complete physical well",
    "lineage": {"base": str(V33_PATH), "base_column": BASE_COLUMN},
    "risk_source": str(V47_PATH),
    "candidate_source": str(V28_PATH),
    "risk_fraction": RISK_FRACTION,
    "candidate_names": candidate_names,
    "base": base_result,
    "nested_risk_gated_endpoint": result,
    "risk_alignment_shuffle": shuffle_result,
    "oracle_hardness_gate": oracle_result,
    "gain": gain,
    "far_gain": far_gain,
    "shuffle_loss": shuffle_loss,
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> refit the risk model and robust endpoint for a conservative hidden-well candidate"
        if strict_pass
        else "FAIL -> risk is observable but existing endpoint bank cannot exploit it; build a new hard-well endpoint"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV48 SUMMARY")
print(pd.DataFrame([base_result, result, shuffle_result, oracle_result]).to_string(index=False))
print("\nV48 FOLDS")
print(fold_table.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default))
print("saved:", OUT_DIR)
