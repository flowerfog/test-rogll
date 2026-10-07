"""Audit independent tail correction sources beyond v27 Neighbor.

The first 1000 ft is exactly frozen. Every endpoint delta is filtered with the
same centred 256-ft filter and clipped to 24 ft. Fold 3 chooses only endpoint
dose and fixed blend weight; Fold 2 confirms and Fold 1 is the final check.
Per-well choice/continuous oracles are diagnostic investment gates only.
"""

from pathlib import Path
import json
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V27_PATH = (
    OOF_ROOT / "frontier_6p39_tail_lowpass_profile_v27" / "tail_lowpass_oof.parquet"
)
CORE_PATH = (
    OOF_ROOT / "frontier_6p39_existing_models_oof_v0" / "frontier_6p39_core_oof.parquet"
)
CP_PATH = (
    OOF_ROOT / "frontier_6p39_full_oof_cp_increment_v18" / "full_oof_cp_endpoints.parquet"
)
OUT_DIR = OOF_ROOT / "frontier_6p39_tail_endpoint_oracle_v28"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
FINAL_FOLD = 1
START_FT = 1000.0
TRANSITION_FT = 250.0
LOWPASS_WINDOW_FT = 256.0
CAP_FT = 24.0
DOSES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
BLEND_WEIGHTS = tuple(np.round(np.linspace(0.0, 1.0, 21), 2))
started = time.time()


def _smoothstep(distance):
    t = np.clip((np.asarray(distance, np.float64) - START_FT) / TRANSITION_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _centred_lowpass(md, values, valid):
    md = np.asarray(md, np.float64)
    values = np.asarray(values, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(values)
    if len(md) == 0 or np.any(np.diff(md) < 0.0):
        raise RuntimeError("Low-pass input MD must be sorted")
    half = 0.5 * LOWPASS_WINDOW_FT
    left = np.searchsorted(md, md - half, side="left")
    right = np.searchsorted(md, md + half, side="right")
    csum = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    sums = csum[right] - csum[left]
    counts = count[right] - count[left]
    output_valid = counts > 0
    output = np.zeros(len(values), np.float64)
    output[output_valid] = sums[output_valid] / counts[output_valid]
    return output, output_valid


def _metric(local, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    truth = local["tvt_true"].to_numpy(np.float64)
    error = prediction - truth
    if len(error) != len(local) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid prediction: {endpoint}")
    well_mse = pd.Series(error * error).groupby(local["well"], sort=False).mean()
    result = {
        "endpoint": endpoint,
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(well_mse))),
        "p95_well_rmse": float(np.quantile(np.sqrt(well_mse), 0.95)),
    }
    distance = local["md_since"].to_numpy(np.float64)
    for name, mask in {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }.items():
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return result


def _perwell_oracles(local, first, second):
    truth = local["tvt_true"].to_numpy(np.float64)
    first = np.asarray(first, np.float64)
    second = np.asarray(second, np.float64)
    choice = first.copy()
    continuous = first.copy()
    weights = []
    for _, indices in local.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        e1 = first[indices] - truth[indices]
        e2 = second[indices] - truth[indices]
        if float(np.dot(e2, e2)) < float(np.dot(e1, e1)):
            choice[indices] = second[indices]
        direction = second[indices] - first[indices]
        denominator = float(np.dot(direction, direction))
        weight = (
            float(np.clip(-np.dot(e1, direction) / denominator, 0.0, 1.0))
            if denominator > 1e-12 else 0.0
        )
        continuous[indices] = first[indices] + weight * direction
        weights.append(weight)
    return choice, continuous, float(np.mean(weights))


for required in (V27_PATH, CORE_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

frame = pd.read_parquet(
    V27_PATH,
    columns=[
        "id", "well", "row_idx", "fold", "tvt_true", "md_since", "parent", "selected"
    ],
)
frame["id"] = frame["id"].astype(str)
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate v27 ids")

core_columns = ["id", "truth", "ridge_pp", "sp45", "fle", "frontier_6p39_core"]
core = pd.read_parquet(CORE_PATH, columns=core_columns)
core["id"] = core["id"].astype(str)
core = core.rename(columns={
    "truth": "core_truth",
    "frontier_6p39_core": "core_endpoint",
})
frame = frame.merge(core, on="id", how="left", validate="one_to_one")

endpoint_columns = ["ridge_pp", "sp45", "fle", "core_endpoint"]
if CP_PATH.is_file():
    cp_header = pq.ParquetFile(CP_PATH).schema.names
    cp_column = next(
        (name for name in ("full_cp8_ordered", "cp8_before_tail", "full_parent_plus_raw_cp_crossfit")
         if name in cp_header),
        None,
    )
    if cp_column is not None:
        cp = pd.read_parquet(CP_PATH, columns=["id", cp_column])
        cp["id"] = cp["id"].astype(str)
        cp = cp.rename(columns={cp_column: "cp8_endpoint"})
        frame = frame.merge(cp, on="id", how="left", validate="one_to_one")
        endpoint_columns.append("cp8_endpoint")

truth_gap = np.abs(
    frame["tvt_true"].to_numpy(np.float64) - frame["core_truth"].to_numpy(np.float64)
)
if not np.isfinite(truth_gap).all() or float(np.max(truth_gap)) > 0.01:
    raise RuntimeError(f"Core alignment mismatch: max truth gap={np.nanmax(truth_gap)}")
for column in endpoint_columns:
    if not np.isfinite(frame[column].to_numpy(np.float64)).all():
        raise RuntimeError(f"Non-finite endpoint: {column}")

parent = frame["parent"].to_numpy(np.float64)
neighbor_v27 = frame["selected"].to_numpy(np.float64)
truth = frame["tvt_true"].to_numpy(np.float64)
distance = frame["md_since"].to_numpy(np.float64)
fold = frame["fold"].to_numpy(np.int8)
tail_weight = _smoothstep(distance)
groups = {
    str(well): np.asarray(indices, np.int64)
    for well, indices in frame.groupby("well", sort=False).indices.items()
}

print(
    "TAIL-ENDPOINT CONFIG",
    {
        "rows": len(frame), "wells": len(groups), "endpoints": endpoint_columns,
        "lowpass_ft": LOWPASS_WINDOW_FT, "cap_ft": CAP_FT,
    },
    flush=True,
)


def _fold_metric(prediction, held_fold, endpoint):
    mask = fold == int(held_fold)
    return _metric(frame.loc[mask].reset_index(drop=True), prediction[mask], endpoint)


summary_rows = []
audit_rows = []
prediction_store = {}

for endpoint_number, endpoint_name in enumerate(endpoint_columns, 1):
    absolute = frame[endpoint_name].to_numpy(np.float64)
    raw_delta = absolute - parent
    filtered_delta = np.zeros(len(frame), np.float32)
    filtered_valid = np.zeros(len(frame), bool)
    for well, indices in groups.items():
        values, valid = _centred_lowpass(
            distance[indices], raw_delta[indices], np.isfinite(raw_delta[indices])
        )
        filtered_delta[indices] = values.astype(np.float32)
        filtered_valid[indices] = valid
    base_correction = tail_weight * np.where(
        filtered_valid, np.clip(filtered_delta.astype(np.float64), -CAP_FT, CAP_FT), 0.0
    )

    dose_scan = []
    for dose in DOSES:
        prediction = parent + float(dose) * base_correction
        metric = _fold_metric(prediction, SELECTION_FOLD, endpoint_name)
        parent_metric = _fold_metric(parent, SELECTION_FOLD, "parent")
        metric.update({
            "dose": float(dose),
            "p95_change_vs_parent": metric["p95_well_rmse"] - parent_metric["p95_well_rmse"],
        })
        metric["risk_pass"] = bool(metric["p95_change_vs_parent"] <= 0.10)
        metric["selection_score"] = float(
            metric["rmse"] + 0.5 * max(metric["p95_change_vs_parent"], 0.0)
        )
        dose_scan.append(metric)
    dose_table = pd.DataFrame(dose_scan).sort_values(
        ["risk_pass", "selection_score", "rmse"],
        ascending=[False, True, True],
        kind="stable",
    )
    selected_dose = float(dose_table.iloc[0]["dose"])
    endpoint_prediction = parent + selected_dose * base_correction

    blend_scan = []
    selection_mask = fold == SELECTION_FOLD
    local_selection = frame.loc[selection_mask].reset_index(drop=True)
    for weight in BLEND_WEIGHTS:
        prediction = neighbor_v27 + float(weight) * (endpoint_prediction - neighbor_v27)
        metric = _metric(local_selection, prediction[selection_mask], "fixed_blend")
        metric["weight_endpoint"] = float(weight)
        blend_scan.append(metric)
    blend_table = pd.DataFrame(blend_scan).sort_values(
        ["rmse", "p95_well_rmse"], kind="stable"
    )
    selected_weight = float(blend_table.iloc[0]["weight_endpoint"])
    fixed_blend = neighbor_v27 + selected_weight * (endpoint_prediction - neighbor_v27)
    prediction_store[endpoint_name] = endpoint_prediction.astype(np.float32)

    for held_fold, role in ((3, "selection"), (2, "confirmation"), (1, "final")):
        mask = fold == held_fold
        local = frame.loc[mask].reset_index(drop=True)
        choice, continuous, mean_oracle_weight = _perwell_oracles(
            local, neighbor_v27[mask], endpoint_prediction[mask]
        )
        endpoints = {
            "parent": parent[mask],
            "neighbor_v27": neighbor_v27[mask],
            "endpoint_tail": endpoint_prediction[mask],
            "fixed_blend": fixed_blend[mask],
            "choice_oracle": choice,
            "continuous_oracle": continuous,
        }
        local_metrics = {}
        for variant, prediction in endpoints.items():
            metric = _metric(local, prediction, variant)
            summary_rows.append({
                "source": endpoint_name,
                "selected_dose": selected_dose,
                "fixed_weight_endpoint": selected_weight,
                "fold": held_fold,
                "role": role,
                "mean_oracle_weight_endpoint": mean_oracle_weight,
                **metric,
            })
            local_metrics[variant] = metric

        tail = local["md_since"].to_numpy(np.float64) > 1000.0
        e_neighbor = neighbor_v27[mask][tail] - local["tvt_true"].to_numpy(np.float64)[tail]
        e_endpoint = endpoint_prediction[mask][tail] - local["tvt_true"].to_numpy(np.float64)[tail]
        correlation = float(np.corrcoef(e_neighbor, e_endpoint)[0, 1])
        audit_rows.append({
            "source": endpoint_name,
            "selected_dose": selected_dose,
            "fixed_weight_endpoint": selected_weight,
            "fold": held_fold,
            "role": role,
            "tail_error_correlation": correlation,
            "endpoint_gain_vs_neighbor": (
                local_metrics["neighbor_v27"]["rmse"] - local_metrics["endpoint_tail"]["rmse"]
            ),
            "fixed_gain_vs_neighbor": (
                local_metrics["neighbor_v27"]["rmse"] - local_metrics["fixed_blend"]["rmse"]
            ),
            "choice_oracle_gain": (
                local_metrics["neighbor_v27"]["rmse"] - local_metrics["choice_oracle"]["rmse"]
            ),
            "continuous_oracle_gain": (
                local_metrics["neighbor_v27"]["rmse"] - local_metrics["continuous_oracle"]["rmse"]
            ),
            "fixed_far_gain_vs_neighbor": (
                local_metrics["neighbor_v27"]["far_gt2000_rmse"]
                - local_metrics["fixed_blend"]["far_gt2000_rmse"]
            ),
            "fixed_p95_change_vs_neighbor": (
                local_metrics["fixed_blend"]["p95_well_rmse"]
                - local_metrics["neighbor_v27"]["p95_well_rmse"]
            ),
            "mean_oracle_weight_endpoint": mean_oracle_weight,
        })
    print(
        f"TAIL-ENDPOINT {endpoint_number}/{len(endpoint_columns)}",
        endpoint_name,
        {"dose": selected_dose, "fixed_weight": selected_weight},
        flush=True,
    )

summary = pd.DataFrame(summary_rows)
audit = pd.DataFrame(audit_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
audit.to_csv(OUT_DIR / "endpoint_audit.csv", index=False)

# Select the endpoint on Fold 3 only. Fold 1/2 oracle evidence remains a
# confirmation diagnostic and cannot change which endpoint enters the gates.
selection_ranking = audit[audit["fold"].eq(3)].sort_values(
    ["continuous_oracle_gain", "choice_oracle_gain", "fixed_gain_vs_neighbor"],
    ascending=False,
    kind="stable",
).reset_index(drop=True)
selected_source = str(selection_ranking.iloc[0]["source"])
selection_ranking.to_csv(OUT_DIR / "fold3_source_selection.csv", index=False)

# Rank investment value on Fold 1+2 for reporting only.
rank_rows = []
for source in endpoint_columns:
    local = audit[audit["source"].eq(source) & audit["fold"].isin((1, 2))]
    rank_rows.append({
        "source": source,
        "selected_dose": float(local["selected_dose"].iloc[0]),
        "fixed_weight_endpoint": float(local["fixed_weight_endpoint"].iloc[0]),
        "mean_tail_error_correlation": float(local["tail_error_correlation"].mean()),
        "minimum_fixed_gain": float(local["fixed_gain_vs_neighbor"].min()),
        "mean_fixed_gain": float(local["fixed_gain_vs_neighbor"].mean()),
        "minimum_choice_oracle_gain": float(local["choice_oracle_gain"].min()),
        "mean_choice_oracle_gain": float(local["choice_oracle_gain"].mean()),
        "minimum_continuous_oracle_gain": float(local["continuous_oracle_gain"].min()),
        "mean_continuous_oracle_gain": float(local["continuous_oracle_gain"].mean()),
        "minimum_fixed_far_gain": float(local["fixed_far_gain_vs_neighbor"].min()),
        "maximum_fixed_p95_change": float(local["fixed_p95_change_vs_neighbor"].max()),
        "mean_oracle_weight_endpoint": float(local["mean_oracle_weight_endpoint"].mean()),
    })
ranking = pd.DataFrame(rank_rows).sort_values(
    ["minimum_continuous_oracle_gain", "minimum_choice_oracle_gain", "mean_fixed_gain"],
    ascending=False,
    kind="stable",
).reset_index(drop=True)
ranking["selected_on_fold3"] = ranking["source"].eq(selected_source)
ranking.to_csv(OUT_DIR / "investment_ranking.csv", index=False)

best = ranking.loc[ranking["source"].eq(selected_source)].iloc[0].to_dict()
gates = {
    "choice_oracle_gain_each_fold_ge_0p08": bool(best["minimum_choice_oracle_gain"] >= 0.08),
    "continuous_oracle_gain_each_fold_ge_0p15": bool(
        best["minimum_continuous_oracle_gain"] >= 0.15
    ),
    "oracle_weight_nontrivial": bool(best["mean_oracle_weight_endpoint"] >= 0.10),
}
fixed_ready = bool(
    best["minimum_fixed_gain"] > 0.0
    and best["minimum_fixed_far_gain"] >= -0.01
    and best["maximum_fixed_p95_change"] <= 0.10
)

row_output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
row_output["parent"] = parent.astype(np.float32)
row_output["neighbor_v27"] = neighbor_v27.astype(np.float32)
for source, prediction in prediction_store.items():
    row_output[f"tail_{source}"] = prediction
row_output.to_parquet(OUT_DIR / "tail_endpoint_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": "frontier_6p39_tail_endpoint_oracle_v28",
    "stage": "Fold3_endpoint_dose_and_fixed_weight_Fold2_confirmation_Fold1_final",
    "sources": {
        "v27": str(V27_PATH), "core": str(CORE_PATH),
        "cp": str(CP_PATH) if CP_PATH.is_file() else None,
    },
    "endpoint_columns": endpoint_columns,
    "hidden_predictions_observed": False,
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD,
        "final_fold": FINAL_FOLD,
        "fold2_search_count": 0,
        "fold1_search_count": 0,
    },
    "tail_recipe": {
        "start_ft": START_FT, "transition_ft": TRANSITION_FT,
        "lowpass_window_ft": LOWPASS_WINDOW_FT, "cap_ft": CAP_FT,
    },
    "fold3_selected_source": selected_source,
    "fold3_source_selection": selection_ranking.to_dict(orient="records"),
    "selected_source_confirmation": best,
    "oracle_investment_gates": gates,
    "oracle_pass": bool(all(gates.values())),
    "fixed_rule_already_ready": fixed_ready,
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "free_gb": float(shutil.disk_usage(ROOT).free / (1024 ** 3)),
    "next": (
        "FIXED PASS -> freeze the two-endpoint blend and replay it on the full deployment chain"
        if fixed_ready else (
            "ORACLE PASS -> train a cross-fit same-direction confidence dose rule"
            if all(gates.values()) else
            "FAIL -> v27 Neighbor is the current tail postprocessing ceiling"
        )
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nTAIL ENDPOINT INVESTMENT RANKING", flush=True)
print(ranking.to_string(index=False), flush=True)
print("\nTAIL ENDPOINT AUDIT", flush=True)
print(audit.to_string(index=False), flush=True)
print("\nFINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
