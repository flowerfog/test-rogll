"""Zero-training incremental audit for the Alias-V2 absolute emission.

Run in the same kernel after both the Drift-V3 audits and
``alias_aware_longcontext_absolute_matcher_fold2_v2_cell.py``.  The cell asks
whether Alias-V2 contributes modes or path accuracy beyond the frozen V3
emission.  No weights are selected on Fold 2: probability fusion is fixed at
50/50 and the existing V3 drift heads/precisions are reused unchanged.
"""

from pathlib import Path
import gc
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
import torch


REQUIRED_UNION_V0 = {
    "OOF_ROOT",
    "device",
    "model_v3",
    "model_alias_v2",
    "fold2_volumes_v3",
    "samples_alias_v2",
    "AliasSampleV2",
    "_load_raw_record",
    "_native_z_relative_v2",
    "_robust_prefix_slope_v2",
    "_window_bank_alias_v2",
    "QUERY_HALF_WIDTHS_ALIAS_V2",
    "_sample_tensors_alias_v2",
    "_tensors_v3",
    "_graph_solve_audit_v0",
    "_path_payload_v2",
    "_summarise_payloads_v2",
    "STATE_OFFSETS_ALIAS_V2",
    "_write_json",
    "_json_default",
}
missing_union_v0 = sorted(REQUIRED_UNION_V0.difference(globals()))
if missing_union_v0:
    raise RuntimeError(
        "Run after Drift-V3 audit and Alias-V2 in the same kernel; missing: "
        + repr(missing_union_v0)
    )

EXPERIMENT_UNION_V0 = "alias_v2_emission_union_path_audit_v0"
OUT_DIR_UNION_V0 = Path(OOF_ROOT) / EXPERIMENT_UNION_V0
OUT_DIR_UNION_V0.mkdir(parents=True, exist_ok=True)


def _node_sample_union_v0(well, volume):
    record = _load_raw_record(well)
    rows = volume.node_rows.astype(np.int64)
    source = samples_alias_v2[well]
    full_z = _native_z_relative_v2(well).astype(np.float64)
    anchor_md = float(record.md[record.anchor_index])
    distance = record.md[rows].astype(np.float64) - anchor_md
    z_relative = full_z[rows]
    slope = _robust_prefix_slope_v2(record, full_z)
    prior = slope * distance - z_relative
    labels = volume.truth_offset.astype(np.float64)
    label_index = np.rint(
        labels - float(STATE_OFFSETS_ALIAS_V2[0])
    ).astype(np.int64)
    query = _window_bank_alias_v2(
        record.md,
        record.horizontal[0],
        record.horizontal[1],
        record.md[rows],
        QUERY_HALF_WIDTHS_ALIAS_V2,
    )
    return AliasSampleV2(
        well=well,
        fold=2,
        reference=source.reference,
        state_valid=source.state_valid,
        query=query,
        labels=labels.astype(np.float32),
        label_index=label_index.astype(np.int64),
        prior_offset=prior.astype(np.float32),
        distance_ft=distance.astype(np.float32),
        z_relative=z_relative.astype(np.float32),
        row_weight=np.ones(len(rows), np.float32),
        requested_rows=len(rows),
        eval_rows=source.eval_rows,
    )


def _log_probability_union_v0(logits, valid):
    logits = np.asarray(logits, np.float64)
    logits[:, ~np.asarray(valid, bool)] = -1e4
    maximum = np.max(logits, axis=1, keepdims=True)
    probability = np.exp(np.clip(logits - maximum, -60.0, 0.0))
    probability[:, ~np.asarray(valid, bool)] = 0.0
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
    return np.log(np.maximum(probability, 1e-30)), probability


def _retrieval_union_v0(scores, sample):
    scores = np.asarray(scores, np.float64)
    truth = sample.labels.astype(np.float64)
    truth_index = sample.label_index.astype(np.int64)
    prediction = STATE_OFFSETS_ALIAS_V2[np.argmax(scores, axis=1)]
    truth_score = scores[np.arange(len(scores)), truth_index]
    rank = 1 + np.sum(scores > truth_score[:, None], axis=1)
    top8 = np.argpartition(scores, -8, axis=1)[:, -8:]
    return {
        "rows": len(scores),
        "argmax_rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "top1_within5": float(np.mean(np.abs(prediction - truth) <= 5.0)),
        "top8_within5": float(
            np.mean(
                np.any(
                    np.abs(STATE_OFFSETS_ALIAS_V2[top8] - truth[:, None]) <= 5.0,
                    axis=1,
                )
            )
        ),
        "median_rank": float(np.median(rank)),
        "p90_rank": float(np.quantile(rank, 0.90)),
        "top8_index": top8,
    }


def _posterior_path_union_v0(probability):
    return probability @ STATE_OFFSETS_ALIAS_V2.astype(np.float64)


started_union_v0 = time.time()
model_v3.eval()
model_alias_v2.eval()

retrieval_rows_union_v0 = []
union_hit_parts_v0 = []
path_payloads_union_v0 = {
    "v3_frozen": [],
    "alias_absolute_only": [],
    "alias_plus_v3_drift": [],
    "fused_absolute_only": [],
    "fused_plus_v3_drift": [],
}

for position_union_v0, well_union_v0 in enumerate(sorted(fold2_volumes_v3), 1):
    volume_union_v0 = fold2_volumes_v3[well_union_v0]
    sample_union_v0 = _node_sample_union_v0(well_union_v0, volume_union_v0)

    alias_tensors_union_v0 = _sample_tensors_alias_v2(sample_union_v0)
    (
        alias_query_union_v0,
        alias_reference_union_v0,
        alias_valid_union_v0,
        _,
        _,
        alias_prior_union_v0,
        alias_distance_union_v0,
        alias_z_union_v0,
        _,
    ) = alias_tensors_union_v0
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=device.type == "cuda"
    ):
        alias_logits_union_v0 = model_alias_v2(
            alias_query_union_v0,
            alias_reference_union_v0,
            alias_valid_union_v0,
            alias_prior_union_v0,
            alias_distance_union_v0,
            alias_z_union_v0,
            prior_jitter=False,
        )
        v3_values_union_v0, _, v3_z_union_v0, _ = _tensors_v3(volume_union_v0)
        v3_output_union_v0 = model_v3(v3_values_union_v0, v3_z_union_v0)

    alias_logits_np_union_v0 = alias_logits_union_v0.float().cpu().numpy()
    v3_logits_np_union_v0 = v3_output_union_v0["unary"].float().cpu().numpy().T
    state_valid_union_v0 = sample_union_v0.state_valid
    old_log_union_v0, old_probability_union_v0 = _log_probability_union_v0(
        v3_logits_np_union_v0, state_valid_union_v0
    )
    new_log_union_v0, new_probability_union_v0 = _log_probability_union_v0(
        alias_logits_np_union_v0, state_valid_union_v0
    )
    fused_probability_union_v0 = 0.5 * (
        old_probability_union_v0 + new_probability_union_v0
    )
    fused_probability_union_v0 /= np.maximum(
        fused_probability_union_v0.sum(axis=1, keepdims=True), 1e-12
    )
    fused_log_union_v0 = np.log(np.maximum(fused_probability_union_v0, 1e-30))

    old_retrieval_union_v0 = _retrieval_union_v0(old_log_union_v0, sample_union_v0)
    new_retrieval_union_v0 = _retrieval_union_v0(new_log_union_v0, sample_union_v0)
    fused_retrieval_union_v0 = _retrieval_union_v0(fused_log_union_v0, sample_union_v0)
    for label_union_v0, metrics_union_v0 in (
        ("v3", old_retrieval_union_v0),
        ("alias_v2", new_retrieval_union_v0),
        ("probability_mean", fused_retrieval_union_v0),
    ):
        retrieval_rows_union_v0.append(
            {
                "well": well_union_v0,
                "variant": label_union_v0,
                **{
                    key: value
                    for key, value in metrics_union_v0.items()
                    if key != "top8_index"
                },
            }
        )

    union_index_union_v0 = np.concatenate(
        [
            old_retrieval_union_v0["top8_index"],
            new_retrieval_union_v0["top8_index"],
        ],
        axis=1,
    )
    union_hit_parts_v0.append(
        np.any(
            np.abs(
                STATE_OFFSETS_ALIAS_V2[union_index_union_v0]
                - sample_union_v0.labels[:, None]
            )
            <= 5.0,
            axis=1,
        )
    )

    z_np_union_v0 = sample_union_v0.z_relative.astype(np.float64)
    alias_absolute_union_v0 = _posterior_path_union_v0(new_probability_union_v0)
    alias_absolute_union_v0 -= alias_absolute_union_v0[0]
    fused_absolute_union_v0 = _posterior_path_union_v0(fused_probability_union_v0)
    fused_absolute_union_v0 -= fused_absolute_union_v0[0]

    old_absolute_precision_union_v0 = (
        v3_output_union_v0["absolute_precision"].float().cpu().numpy()
    )
    old_horizons_union_v0 = {
        int(horizon): (
            delta.float().cpu().numpy(),
            precision.float().cpu().numpy(),
        )
        for horizon, (delta, precision) in v3_output_union_v0["horizons"].items()
    }

    def solve_union_v0(absolute_tvt_union_v0):
        solved_s_union_v0 = _graph_solve_audit_v0(
            absolute_tvt_union_v0 + z_np_union_v0,
            old_absolute_precision_union_v0,
            old_horizons_union_v0,
            len(z_np_union_v0),
            use_absolute=True,
            use_drift=True,
            learned_precision=True,
        )
        solved_tvt_union_v0 = solved_s_union_v0 - z_np_union_v0
        return solved_tvt_union_v0 - solved_tvt_union_v0[0]

    predictions_union_v0 = {
        "v3_frozen": (
            v3_output_union_v0["solved_tvt"].float().cpu().numpy()
        ),
        "alias_absolute_only": alias_absolute_union_v0,
        "alias_plus_v3_drift": solve_union_v0(alias_absolute_union_v0),
        "fused_absolute_only": fused_absolute_union_v0,
        "fused_plus_v3_drift": solve_union_v0(fused_absolute_union_v0),
    }
    for label_union_v0, prediction_union_v0 in predictions_union_v0.items():
        prediction_union_v0 = np.asarray(prediction_union_v0, np.float64)
        prediction_union_v0 -= prediction_union_v0[0]
        path_payloads_union_v0[label_union_v0].append(
            _path_payload_v2(volume_union_v0, prediction_union_v0)
        )

    if position_union_v0 % 50 == 0 or position_union_v0 == len(fold2_volumes_v3):
        print(
            f"emission union {position_union_v0}/{len(fold2_volumes_v3)}",
            flush=True,
        )
    del alias_tensors_union_v0, alias_query_union_v0, alias_reference_union_v0
    del alias_valid_union_v0, alias_prior_union_v0, alias_distance_union_v0
    del alias_z_union_v0, alias_logits_union_v0, v3_output_union_v0

retrieval_bywell_union_v0 = pd.DataFrame(retrieval_rows_union_v0)
retrieval_bywell_union_v0.to_csv(
    OUT_DIR_UNION_V0 / "retrieval_bywell.csv", index=False
)
retrieval_summary_rows_union_v0 = []
for variant_union_v0, frame_union_v0 in retrieval_bywell_union_v0.groupby(
    "variant", sort=True
):
    retrieval_summary_rows_union_v0.append(
        {
            "variant": variant_union_v0,
            "rows": int(frame_union_v0["rows"].sum()),
            "argmax_rmse_well_weighted": float(
                np.average(
                    frame_union_v0["argmax_rmse"],
                    weights=frame_union_v0["rows"],
                )
            ),
            "top1_within5": float(
                np.average(
                    frame_union_v0["top1_within5"],
                    weights=frame_union_v0["rows"],
                )
            ),
            "top8_within5": float(
                np.average(
                    frame_union_v0["top8_within5"],
                    weights=frame_union_v0["rows"],
                )
            ),
            "median_well_rank": float(frame_union_v0["median_rank"].median()),
        }
    )
retrieval_summary_union_v0 = pd.DataFrame(retrieval_summary_rows_union_v0)
retrieval_summary_union_v0.to_csv(
    OUT_DIR_UNION_V0 / "retrieval_summary.csv", index=False
)
top8_union_hit_v0 = float(np.mean(np.concatenate(union_hit_parts_v0)))

path_rows_union_v0 = []
path_bywell_union_v0 = {}
for label_union_v0, payloads_union_v0 in path_payloads_union_v0.items():
    metrics_union_v0, bywell_union_v0, _ = _summarise_payloads_v2(payloads_union_v0)
    path_rows_union_v0.append({"variant": label_union_v0, **metrics_union_v0})
    path_bywell_union_v0[label_union_v0] = bywell_union_v0
    bywell_union_v0.to_csv(
        OUT_DIR_UNION_V0 / f"path_{label_union_v0}_bywell.csv", index=False
    )
path_summary_union_v0 = pd.DataFrame(path_rows_union_v0).sort_values("rmse")
path_summary_union_v0.to_csv(OUT_DIR_UNION_V0 / "path_summary.csv", index=False)

base_path_union_v0 = next(
    row for row in path_rows_union_v0 if row["variant"] == "v3_frozen"
)
best_incremental_union_v0 = min(
    (row for row in path_rows_union_v0 if row["variant"] != "v3_frozen"),
    key=lambda row: row["rmse"],
)
path_gain_union_v0 = float(
    base_path_union_v0["rmse"] - best_incremental_union_v0["rmse"]
)
far_gain_union_v0 = float(
    base_path_union_v0["far_gt2000_rmse"]
    - best_incremental_union_v0["far_gt2000_rmse"]
)
old_top8_union_v0 = float(
    retrieval_summary_union_v0.loc[
        retrieval_summary_union_v0["variant"].eq("v3"), "top8_within5"
    ].iloc[0]
)
union_gain_union_v0 = top8_union_hit_v0 - old_top8_union_v0

verdict_union_v0 = {
    "experiment": EXPERIMENT_UNION_V0,
    "stage": "zero_training_alias_v2_incremental_emission_and_path_audit",
    "validation_fold": 2,
    "fusion": "fixed 0.5 probability mean; no Fold2 weight selection",
    "retrieval": {
        "summary": retrieval_summary_union_v0.to_dict(orient="records"),
        "top8_union_within5": top8_union_hit_v0,
        "top8_union_gain_vs_v3": union_gain_union_v0,
    },
    "path": {
        "summary": path_summary_union_v0.to_dict(orient="records"),
        "best_incremental_variant": best_incremental_union_v0["variant"],
        "gain_vs_v3": path_gain_union_v0,
        "far_gain_vs_v3": far_gain_union_v0,
    },
    "gates": {
        "top8_union_gain_ge": 0.05,
        "path_gain_ge": 0.20,
        "far_gain_ge": 0.20,
    },
    "retrieval_complement_pass": bool(union_gain_union_v0 >= 0.05),
    "path_pass": bool(path_gain_union_v0 >= 0.20 and far_gain_union_v0 >= 0.20),
    "strict_pass": bool(
        union_gain_union_v0 >= 0.05
        and path_gain_union_v0 >= 0.20
        and far_gain_union_v0 >= 0.20
    ),
    "elapsed_minutes": (time.time() - started_union_v0) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> rebuild Drift absolute volume with Alias-V2 and test untouched Fold1"
        if union_gain_union_v0 >= 0.05
        and path_gain_union_v0 >= 0.20
        and far_gain_union_v0 >= 0.20
        else "FAIL -> close local correspondence; proceed to complete-well direct alignment"
    ),
}
_write_json(OUT_DIR_UNION_V0 / "verdict.json", verdict_union_v0)

print("\nRETRIEVAL SUMMARY")
print(retrieval_summary_union_v0.to_string(index=False), flush=True)
print("top8 union within5:", top8_union_hit_v0, flush=True)
print("\nPATH SUMMARY")
print(path_summary_union_v0.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict_union_v0, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR_UNION_V0, flush=True)

gc.collect()
torch.cuda.empty_cache()
