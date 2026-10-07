"""Cross-fitted marker-event verifier after the Drift-V3 marker audit.

Run in the same live kernel after ``v3_constraint_profile_marker_audit_v0``.
The frozen Drift-V3 model is never retrained.  Candidate marker events are
generated from typewell-unique markers and the frozen absolute score profile.
A compact LightGBM classifier is cross-fitted by complete well inside the
untouched Fold-2 population.  Every reported path prediction therefore comes
from an event classifier that did not see that well's labels.

The path recipe is pre-registered and is not selected on Fold 2:

* at most two events per well;
* non-maximum suppression over 128 ft;
* correction dose 0.50;
* raw correction cap 16 ft (maximum applied move 8 ft).

The matched control keeps event scores and event locations fixed but rotates
the marker identity within each well.  A pass here authorises a proper outer-
train OOF marker stack; it is not itself a five-fold result.
"""

from pathlib import Path
import gc
import hashlib
import json
import math
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REQUIRED_MARKER_V1 = {
    "OOF_ROOT",
    "fold2_volumes_v3",
    "forward_audit_v0",
    "node_predictions_audit_v0",
    "_load_raw_record",
    "_native_z_relative_v2",
    "_robust_prefix_slope_v2",
    "_select_markers_audit_v0",
    "_path_payload_v2",
    "_summarise_payloads_v2",
    "STATE_OFFSETS_V2",
    "_write_json",
    "_json_default",
}
missing_marker_v1 = sorted(REQUIRED_MARKER_V1.difference(globals()))
if missing_marker_v1:
    raise RuntimeError(
        "Run after v3_constraint_profile_marker_audit_v0 in the same kernel; "
        f"missing: {missing_marker_v1}"
    )

EXPERIMENT_MARKER_V1 = "v3_marker_event_crossfit_gateway_v1"
OUT_DIR_MARKER_V1 = Path(OOF_ROOT) / EXPERIMENT_MARKER_V1
OUT_DIR_MARKER_V1.mkdir(parents=True, exist_ok=True)

META_FOLDS_MARKER_V1 = 5
MARKER_CANDIDATES_PER_NODE_V1 = 3
EVENT_TOLERANCE_FT_V1 = 4.0
EVENT_NMS_FT_V1 = 128.0

# Frozen before seeing any Fold-2 path score.
FIXED_TOP_EVENTS_V1 = 2
FIXED_DOSE_V1 = 0.50
FIXED_RAW_CAP_FT_V1 = 16.0

SEED_MARKER_V1 = 20260719


def _stable_fold_marker_v1(well):
    digest = hashlib.sha256(str(well).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % META_FOLDS_MARKER_V1 + 1


def _softmax_marker_v1(values):
    values = np.asarray(values, np.float64)
    shifted = values - np.nanmax(values)
    probability = np.exp(np.clip(shifted, -40.0, 0.0))
    probability /= max(float(probability.sum()), 1e-12)
    return probability


def _finite_marker_v1(value, default=0.0):
    return float(value) if np.isfinite(value) else float(default)


def _candidate_table_marker_v1():
    rows = []
    marker_bank = {}
    for position, well in enumerate(sorted(fold2_volumes_v3), 1):
        volume = fold2_volumes_v3[well]
        record = _load_raw_record(well)
        output = forward_audit_v0[well]
        base_prediction = node_predictions_audit_v0[well][
            "combined_learned_precision"
        ].astype(np.float64)
        markers, uniqueness, alias24, coverage = _select_markers_audit_v0(record)
        if len(markers) < MARKER_CANDIDATES_PER_NODE_V1:
            raise RuntimeError(f"{well}: only {len(markers)} marker states")
        marker_bank[well] = markers.astype(np.int64)

        score = volume.input[0].astype(np.float32).T  # nodes x states
        distance = volume.node_md.astype(np.float64) - float(volume.node_md[0])
        z_relative = volume.node_z_relative.astype(np.float64)
        slope = _robust_prefix_slope_v2(record, _native_z_relative_v2(well))
        prefix_prediction = slope * distance - z_relative
        truth = volume.truth_offset.astype(np.float64)
        absolute_precision = output["absolute_precision"].astype(np.float64)

        typewell_gr = record.typewell[0].astype(np.float64)
        typewell_valid = record.typewell[1].astype(np.float64) > 0.5
        centre = len(typewell_gr) // 2
        typewell_d1 = np.gradient(typewell_gr)
        typewell_d2 = np.gradient(typewell_d1)

        marker_score_matrix = score[:, markers]
        for node in range(len(score)):
            state_score = score[node].astype(np.float64)
            state_probability = _softmax_marker_v1(state_score)
            state_order = np.argsort(state_score)[::-1]
            state_best = float(state_score[state_order[0]])
            state_second = float(state_score[state_order[1]])
            entropy = float(
                -np.sum(state_probability * np.log(np.maximum(state_probability, 1e-12)))
            )

            marker_score = marker_score_matrix[node].astype(np.float64)
            marker_order = np.argsort(marker_score)[::-1]
            marker_mean = float(marker_score.mean())
            marker_std = float(marker_score.std()) + 1e-6
            for marker_rank, marker_position in enumerate(
                marker_order[:MARKER_CANDIDATES_PER_NODE_V1], 1
            ):
                marker = int(markers[marker_position])
                marker_offset = float(STATE_OFFSETS_V2[marker])
                value = float(state_score[marker])
                lo = max(0, marker - 4)
                hi = min(len(state_score), marker + 5)
                local_probability = float(state_probability[lo:hi].sum())
                global_rank = int(np.sum(state_score > value)) + 1

                left1 = float(score[max(0, node - 1), marker])
                right1 = float(score[min(len(score) - 1, node + 1), marker])
                left2 = float(score[max(0, node - 2), marker])
                right2 = float(score[min(len(score) - 1, node + 2), marker])
                temporal_peak1 = value - max(left1, right1)
                temporal_peak2 = value - max(left2, right2)
                temporal_curvature = 2.0 * value - left1 - right1

                alias_scores = []
                for shift in (-24, 24):
                    alias_state = marker + shift
                    if 0 <= alias_state < len(state_score):
                        alias_scores.append(float(state_score[alias_state]))
                best_alias_score = max(alias_scores) if alias_scores else value

                tw_index = centre + int(round(marker_offset))
                tw_index = int(np.clip(tw_index, 0, len(typewell_gr) - 1))
                tw_valid = float(typewell_valid[tw_index])
                tw_lo = max(0, tw_index - 8)
                tw_hi = min(len(typewell_gr), tw_index + 9)
                tw_local = typewell_gr[tw_lo:tw_hi]
                tw_local_valid = typewell_valid[tw_lo:tw_hi]
                tw_local_std = float(np.std(tw_local[tw_local_valid])) if tw_local_valid.any() else 0.0

                base_difference = marker_offset - float(base_prediction[node])
                prefix_difference = marker_offset - float(prefix_prediction[node])
                truth_difference = marker_offset - float(truth[node])
                rows.append(
                    {
                        "well": well,
                        "meta_fold": _stable_fold_marker_v1(well),
                        "node": int(node),
                        "node_md": float(volume.node_md[node]),
                        "distance_ft": float(distance[node]),
                        "marker": marker,
                        "marker_offset": marker_offset,
                        "marker_rank": marker_rank,
                        "label": int(abs(truth_difference) <= EVENT_TOLERANCE_FT_V1),
                        "truth_distance": abs(truth_difference),
                        "score": value,
                        "score_z_markers": (value - marker_mean) / marker_std,
                        "state_rank": global_rank,
                        "state_rank_fraction": global_rank / len(state_score),
                        "score_margin_state": value - state_second,
                        "score_margin_marker": value - float(
                            marker_score[marker_order[1]]
                            if marker_rank == 1
                            else marker_score[marker_order[0]]
                        ),
                        "best_state_score": state_best,
                        "best_state_margin": state_best - state_second,
                        "posterior_entropy": entropy,
                        "local_probability": local_probability,
                        "temporal_peak1": temporal_peak1,
                        "temporal_peak2": temporal_peak2,
                        "temporal_curvature": temporal_curvature,
                        "temporal_left_delta": value - left1,
                        "temporal_right_delta": value - right1,
                        "score_margin_alias24": value - best_alias_score,
                        "marker_uniqueness": _finite_marker_v1(uniqueness[marker], -1.0),
                        "marker_alias24_similarity": _finite_marker_v1(alias24[marker], 1.0),
                        "marker_coverage": _finite_marker_v1(coverage[marker], 0.0),
                        "marker_abs_offset": abs(marker_offset),
                        "base_difference": base_difference,
                        "base_abs_difference": abs(base_difference),
                        "prefix_difference": prefix_difference,
                        "prefix_abs_difference": abs(prefix_difference),
                        "base_prefix_disagreement": float(base_prediction[node] - prefix_prediction[node]),
                        "absolute_precision": float(absolute_precision[node]),
                        "progress": float(node / max(len(score) - 1, 1)),
                        "prefix_slope": float(slope),
                        "z_relative": float(z_relative[node]),
                        "typewell_gr": float(typewell_gr[tw_index]),
                        "typewell_d1": float(typewell_d1[tw_index]),
                        "typewell_d2": float(typewell_d2[tw_index]),
                        "typewell_local_std": tw_local_std,
                        "typewell_valid": tw_valid,
                    }
                )
        if position % 50 == 0 or position == len(fold2_volumes_v3):
            print(
                f"marker candidates {position}/{len(fold2_volumes_v3)} "
                f"rows={len(rows)}",
                flush=True,
            )
    return pd.DataFrame(rows), marker_bank


def _crossfit_marker_v1(frame, features):
    output = frame.copy()
    output["event_probability"] = np.nan
    importance_rows = []
    model_paths = []
    for meta_fold in range(1, META_FOLDS_MARKER_V1 + 1):
        train_mask = output["meta_fold"].ne(meta_fold)
        valid_mask = output["meta_fold"].eq(meta_fold)
        train = output.loc[train_mask]
        valid = output.loc[valid_mask]
        positives = max(int(train["label"].sum()), 1)
        negatives = max(len(train) - positives, 1)
        positive_weight = min(80.0, max(8.0, negatives / positives * 0.35))
        sample_weight = np.ones(len(train), np.float64)
        sample_weight[train["truth_distance"].to_numpy(float) <= 12.0] = 3.0
        sample_weight[train["label"].to_numpy(int) == 1] = positive_weight

        dataset = lgb.Dataset(
            train[features],
            label=train["label"],
            weight=sample_weight,
            feature_name=features,
            free_raw_data=True,
        )
        model = lgb.train(
            {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.035,
                "num_leaves": 23,
                "max_depth": 6,
                "min_data_in_leaf": 80,
                "feature_fraction": 0.82,
                "bagging_fraction": 0.85,
                "bagging_freq": 1,
                "lambda_l1": 1.0,
                "lambda_l2": 6.0,
                "max_bin": 127,
                "verbosity": -1,
                "seed": SEED_MARKER_V1 + meta_fold,
                "num_threads": 4,
            },
            dataset,
            num_boost_round=360,
        )
        output.loc[valid_mask, "event_probability"] = model.predict(valid[features])
        model_path = OUT_DIR_MARKER_V1 / f"marker_event_meta_fold{meta_fold}.txt"
        model.save_model(str(model_path))
        model_paths.append(str(model_path))
        importance_rows.extend(
            {
                "meta_fold": meta_fold,
                "feature": feature,
                "importance_gain": float(gain),
            }
            for feature, gain in zip(features, model.feature_importance("gain"))
        )
        print(
            {
                "meta_fold": meta_fold,
                "train_wells": int(train["well"].nunique()),
                "valid_wells": int(valid["well"].nunique()),
                "train_rows": len(train),
                "valid_rows": len(valid),
                "positive_weight": positive_weight,
            },
            flush=True,
        )
    if not np.isfinite(output["event_probability"]).all():
        raise RuntimeError("Cross-fitted event probabilities contain non-finite values")
    return output, pd.DataFrame(importance_rows), model_paths


def _select_events_marker_v1(frame):
    selected = []
    for row in frame.sort_values("event_probability", ascending=False).itertuples(index=False):
        if row.node <= 0 or abs(float(row.base_difference)) > 64.0:
            continue
        if any(abs(float(row.node_md) - float(previous.node_md)) < EVENT_NMS_FT_V1 for previous in selected):
            continue
        if any(int(row.marker) == int(previous.marker) for previous in selected):
            continue
        selected.append(row)
        if len(selected) >= FIXED_TOP_EVENTS_V1:
            break
    return selected


def _decode_marker_v1(well, frame, marker_bank, shuffled=False):
    volume = fold2_volumes_v3[well]
    base = node_predictions_audit_v0[well]["combined_learned_precision"].astype(np.float64)
    events = _select_events_marker_v1(frame)
    closure = {0: [0.0]}
    selected_rows = []
    markers = marker_bank[well]
    marker_position = {int(marker): index for index, marker in enumerate(markers)}
    for event in events:
        marker = int(event.marker)
        applied_marker = marker
        if shuffled:
            applied_marker = int(markers[(marker_position[marker] + 1) % len(markers)])
        marker_offset = float(STATE_OFFSETS_V2[applied_marker])
        raw = float(np.clip(marker_offset - base[int(event.node)], -FIXED_RAW_CAP_FT_V1, FIXED_RAW_CAP_FT_V1))
        correction = FIXED_DOSE_V1 * raw
        closure.setdefault(int(event.node), []).append(correction)
        selected_rows.append(
            {
                "well": well,
                "meta_fold": int(event.meta_fold),
                "node": int(event.node),
                "distance_ft": float(event.distance_ft),
                "marker": marker,
                "applied_marker": applied_marker,
                "event_probability": float(event.event_probability),
                "label": int(event.label),
                "raw_correction": raw,
                "applied_correction": correction,
                "shuffled": bool(shuffled),
            }
        )
    nodes = np.asarray(sorted(closure), np.int64)
    values = np.asarray([np.mean(closure[node]) for node in nodes], np.float64)
    correction = np.interp(
        np.arange(len(base), dtype=np.float64),
        nodes.astype(np.float64),
        values,
        left=0.0,
        right=float(values[-1]),
    )
    prediction = base + correction
    prediction -= prediction[0]
    return prediction, selected_rows


def _fold_report_marker_v1(base_merged, candidate_merged):
    rows = []
    for meta_fold in range(1, META_FOLDS_MARKER_V1 + 1):
        wells = {
            well for well in np.unique(base_merged["well"])
            if _stable_fold_marker_v1(well) == meta_fold
        }
        base_mask = np.isin(base_merged["well"], list(wells))
        candidate_mask = np.isin(candidate_merged["well"], list(wells))
        base_error = base_merged["prediction"][base_mask] - base_merged["truth"][base_mask]
        candidate_error = (
            candidate_merged["prediction"][candidate_mask]
            - candidate_merged["truth"][candidate_mask]
        )
        base_rmse = float(np.sqrt(np.mean(base_error**2)))
        candidate_rmse = float(np.sqrt(np.mean(candidate_error**2)))
        rows.append(
            {
                "meta_fold": meta_fold,
                "wells": len(wells),
                "rows": int(base_mask.sum()),
                "base_rmse": base_rmse,
                "candidate_rmse": candidate_rmse,
                "gain": base_rmse - candidate_rmse,
            }
        )
    return pd.DataFrame(rows)


started_marker_v1 = time.time()
candidates_marker_v1, marker_bank_v1 = _candidate_table_marker_v1()
identity_columns_marker_v1 = {
    "well",
    "meta_fold",
    "node",
    "node_md",
    "distance_ft",
    "marker",
    "marker_offset",
    "label",
    "truth_distance",
}
features_marker_v1 = [
    column for column in candidates_marker_v1.columns
    if column not in identity_columns_marker_v1
]
crossfit_marker_v1, importance_marker_v1, model_paths_marker_v1 = _crossfit_marker_v1(
    candidates_marker_v1, features_marker_v1
)
crossfit_marker_v1.to_pickle(OUT_DIR_MARKER_V1 / "crossfit_candidates.pkl")
importance_summary_marker_v1 = (
    importance_marker_v1.groupby("feature", as_index=False)["importance_gain"]
    .sum()
    .sort_values("importance_gain", ascending=False)
)
importance_summary_marker_v1.to_csv(
    OUT_DIR_MARKER_V1 / "feature_importance.csv", index=False
)

label_marker_v1 = crossfit_marker_v1["label"].to_numpy(int)
probability_marker_v1 = crossfit_marker_v1["event_probability"].to_numpy(float)
candidate_auc_marker_v1 = float(roc_auc_score(label_marker_v1, probability_marker_v1))
candidate_ap_marker_v1 = float(average_precision_score(label_marker_v1, probability_marker_v1))

base_payloads_marker_v1 = []
candidate_payloads_marker_v1 = []
control_payloads_marker_v1 = []
selected_rows_marker_v1 = []
control_rows_marker_v1 = []
for well in sorted(fold2_volumes_v3):
    well_frame = crossfit_marker_v1[crossfit_marker_v1["well"].eq(well)]
    candidate_prediction, selected = _decode_marker_v1(
        well, well_frame, marker_bank_v1, shuffled=False
    )
    control_prediction, control_selected = _decode_marker_v1(
        well, well_frame, marker_bank_v1, shuffled=True
    )
    base_prediction = node_predictions_audit_v0[well][
        "combined_learned_precision"
    ]
    base_payloads_marker_v1.append(
        _path_payload_v2(fold2_volumes_v3[well], base_prediction)
    )
    candidate_payloads_marker_v1.append(
        _path_payload_v2(fold2_volumes_v3[well], candidate_prediction)
    )
    control_payloads_marker_v1.append(
        _path_payload_v2(fold2_volumes_v3[well], control_prediction)
    )
    selected_rows_marker_v1.extend(selected)
    control_rows_marker_v1.extend(control_selected)

base_metrics_marker_v1, base_bywell_marker_v1, base_merged_marker_v1 = (
    _summarise_payloads_v2(base_payloads_marker_v1)
)
candidate_metrics_marker_v1, candidate_bywell_marker_v1, candidate_merged_marker_v1 = (
    _summarise_payloads_v2(candidate_payloads_marker_v1)
)
control_metrics_marker_v1, control_bywell_marker_v1, control_merged_marker_v1 = (
    _summarise_payloads_v2(control_payloads_marker_v1)
)
selected_marker_v1 = pd.DataFrame(selected_rows_marker_v1)
selected_marker_v1.to_csv(OUT_DIR_MARKER_V1 / "selected_events.csv", index=False)
pd.DataFrame(control_rows_marker_v1).to_csv(
    OUT_DIR_MARKER_V1 / "selected_events_shuffled_control.csv", index=False
)

fold_report_marker_v1 = _fold_report_marker_v1(
    base_merged_marker_v1, candidate_merged_marker_v1
)
fold_report_marker_v1.to_csv(OUT_DIR_MARKER_V1 / "meta_fold_scores.csv", index=False)
base_bywell_marker_v1.to_csv(OUT_DIR_MARKER_V1 / "base_bywell.csv", index=False)
candidate_bywell_marker_v1.to_csv(
    OUT_DIR_MARKER_V1 / "candidate_bywell.csv", index=False
)
control_bywell_marker_v1.to_csv(OUT_DIR_MARKER_V1 / "control_bywell.csv", index=False)

selected_event_precision_marker_v1 = float(selected_marker_v1["label"].mean())
selected_well_hit_rate_marker_v1 = float(
    selected_marker_v1.groupby("well")["label"].max().reindex(
        sorted(fold2_volumes_v3), fill_value=0
    ).mean()
)
overall_gain_marker_v1 = float(
    base_metrics_marker_v1["rmse"] - candidate_metrics_marker_v1["rmse"]
)
far_gain_marker_v1 = float(
    base_metrics_marker_v1["far_gt2000_rmse"]
    - candidate_metrics_marker_v1["far_gt2000_rmse"]
)
control_degradation_marker_v1 = float(
    control_metrics_marker_v1["rmse"] - candidate_metrics_marker_v1["rmse"]
)

path_pass_marker_v1 = bool(
    overall_gain_marker_v1 >= 0.20
    and far_gain_marker_v1 >= 0.50
    and int((fold_report_marker_v1["gain"] > 0).sum()) >= 4
    and float(fold_report_marker_v1["gain"].min()) >= -0.10
)
identity_pass_marker_v1 = bool(control_degradation_marker_v1 >= 0.10)
event_pass_marker_v1 = bool(
    candidate_auc_marker_v1 >= 0.80
    and selected_event_precision_marker_v1 >= 0.40
)

verdict_marker_v1 = {
    "experiment": EXPERIMENT_MARKER_V1,
    "stage": "fold2_complete_well_crossfit_marker_event_gateway",
    "base_model": "frozen Drift-V3 epoch 16",
    "validation_scope": "Fold2 only; five-way grouped crossfit by complete well",
    "classifier": {
        "features": len(features_marker_v1),
        "candidate_rows": len(crossfit_marker_v1),
        "positive_rows": int(label_marker_v1.sum()),
        "positive_fraction": float(label_marker_v1.mean()),
        "crossfit_auc": candidate_auc_marker_v1,
        "crossfit_average_precision": candidate_ap_marker_v1,
        "models": model_paths_marker_v1,
    },
    "frozen_recipe": {
        "top_events_per_well": FIXED_TOP_EVENTS_V1,
        "nms_ft": EVENT_NMS_FT_V1,
        "dose": FIXED_DOSE_V1,
        "raw_cap_ft": FIXED_RAW_CAP_FT_V1,
        "maximum_applied_move_ft": FIXED_DOSE_V1 * FIXED_RAW_CAP_FT_V1,
    },
    "selection": {
        "selected_events": len(selected_marker_v1),
        "event_precision": selected_event_precision_marker_v1,
        "well_hit_rate": selected_well_hit_rate_marker_v1,
    },
    "base": base_metrics_marker_v1,
    "candidate": candidate_metrics_marker_v1,
    "shuffled_marker_identity_control": control_metrics_marker_v1,
    "gains": {
        "overall": overall_gain_marker_v1,
        "far_gt2000": far_gain_marker_v1,
        "control_degradation": control_degradation_marker_v1,
        "positive_meta_folds": int((fold_report_marker_v1["gain"] > 0).sum()),
        "minimum_meta_fold_gain": float(fold_report_marker_v1["gain"].min()),
    },
    "gates": {
        "overall_gain_ge": 0.20,
        "far_gain_ge": 0.50,
        "positive_meta_folds_ge": 4,
        "minimum_meta_fold_gain_ge": -0.10,
        "crossfit_auc_ge": 0.80,
        "selected_event_precision_ge": 0.40,
        "shuffled_identity_degradation_ge": 0.10,
    },
    "path_pass": path_pass_marker_v1,
    "event_pass": event_pass_marker_v1,
    "identity_control_pass": identity_pass_marker_v1,
    "strict_pass": bool(
        path_pass_marker_v1 and event_pass_marker_v1 and identity_pass_marker_v1
    ),
    "elapsed_minutes": (time.time() - started_marker_v1) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> build proper outer-train OOF marker stack, then untouched Fold1"
        if path_pass_marker_v1 and event_pass_marker_v1 and identity_pass_marker_v1
        else "FAIL -> marker oracle is not deployable with current emission; train alias-aware absolute matcher"
    ),
}
_write_json(OUT_DIR_MARKER_V1 / "verdict.json", verdict_marker_v1)

print("\nCROSSFIT CLASSIFIER")
print(json.dumps(verdict_marker_v1["classifier"], indent=2), flush=True)
print("\nTOP IMPORTANCE")
print(importance_summary_marker_v1.head(30).to_string(index=False), flush=True)
print("\nMETA FOLDS")
print(fold_report_marker_v1.to_string(index=False), flush=True)
print("\nPATH SCORES")
print(
    pd.DataFrame(
        [
            {"variant": "base", **base_metrics_marker_v1},
            {"variant": "marker_crossfit", **candidate_metrics_marker_v1},
            {"variant": "shuffled_identity", **control_metrics_marker_v1},
        ]
    ).to_string(index=False),
    flush=True,
)
print("\nFINAL VERDICT")
print(json.dumps(verdict_marker_v1, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR_MARKER_V1, flush=True)

gc.collect()

