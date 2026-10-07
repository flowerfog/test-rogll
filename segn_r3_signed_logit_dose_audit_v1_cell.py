"""Zero-training signed-logit audit for the frozen SEGN-R3 matcher.

This cell separates three possibilities behind R3's failed local gateway:

1. useful candidate ordering with excessive confidence (best 0 < alpha < 1),
2. reversed candidate ordering (best alpha < 0), or
3. no useful ordering (best alpha == 0, the uniform posterior).

Alpha is selected on the same 32 Fold-3 selection wells used by R3 and then
frozen for the remaining 122 Fold-3 confirmation wells. Fold 1 and Fold 2 are
not loaded. The evaluation remains a non-deployable teacher-anchor diagnostic.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SOURCE_CANDIDATES_R3D = (
    Path("/root/autodl-tmp/segn_r3_querycausal_absolute_matcher_fold3_v0_cell.py"),
    Path.cwd() / "segn_r3_querycausal_absolute_matcher_fold3_v0_cell.py",
)
SOURCE_PATH_R3D = next(
    (path for path in SOURCE_CANDIDATES_R3D if path.exists()), None
)
if SOURCE_PATH_R3D is None:
    raise FileNotFoundError(
        "Upload segn_r3_querycausal_absolute_matcher_fold3_v0_cell.py to "
        "/root/autodl-tmp before running this audit."
    )

source_r3d = SOURCE_PATH_R3D.read_text(encoding="utf-8")
marker_r3d = "\nstarted_r3c = time.time()"
if marker_r3d not in source_r3d:
    raise RuntimeError(f"Cannot find R3 bootstrap marker in {SOURCE_PATH_R3D}")
exec(
    compile(source_r3d.split(marker_r3d, 1)[0], str(SOURCE_PATH_R3D), "exec"),
    globals(),
)


EXPERIMENT_R3D = "segn_r3_signed_logit_dose_audit_v1"
OUT_DIR_R3D = OOF_ROOT / EXPERIMENT_R3D
OUT_DIR_R3D.mkdir(parents=True, exist_ok=True)

DOSES_R3D = np.asarray(
    (
        -4.0,
        -2.0,
        -1.0,
        -0.50,
        -0.25,
        -0.10,
        0.0,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.75,
        1.0,
        1.5,
        2.0,
        4.0,
    ),
    dtype=np.float32,
)


def _checkpoint_state_r3d(path):
    try:
        # The checkpoint contains NumPy metadata in addition to model tensors.
        # It was produced by our own R3 training cell, so full loading is trusted.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch versions before weights_only was introduced.
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload
    return payload, {}


@torch.no_grad()
def _decode_record_doses_r3d(model, record, device, control, shuffled_record):
    model.eval()
    node_count = len(record.node_rows)
    sums = np.zeros((len(DOSES_R3D), node_count), np.float64)
    counts = np.zeros(node_count, np.float64)
    sums[:, 0] = float(record.anchor_tvt)
    counts[0] = 1.0

    rank_values = []
    top1_values = []
    top5_values = []
    true_valid_values = []

    doses = torch.as_tensor(DOSES_R3D, device=device)[:, None, None]
    states = torch.as_tensor(STATE_OFFSETS, device=device, dtype=torch.float32)

    for raw_start in record.chunk_starts:
        start = int(raw_start)
        start_row = int(record.node_rows[start])
        deployed_anchor = float(record.truth[start_row])
        payload = _chunk_payload(
            record,
            start,
            deployed_anchor=deployed_anchor,
            previous_delta=0.0,
            previous_rate=0.0,
            control=control,
            shuffled_typewell=shuffled_record,
        )
        tensors = _tensor_payload(payload, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = model(*tensors)
        logits = logits.float()
        candidate_valid = logits > -1.0e3

        scaled = doses * logits[None]
        scaled = scaled.masked_fill(~candidate_valid[None], -1.0e4)
        probability = F.softmax(scaled, dim=1)
        expected = (probability * states[None, :, None]).sum(dim=1)
        expected = expected.cpu().numpy().astype(np.float64)

        end = start + expected.shape[1]
        sums[:, start + 1 : end + 1] += deployed_anchor + expected
        counts[start + 1 : end + 1] += 1.0

        if control == "full":
            target = torch.as_tensor(
                payload["target_state"], device=device, dtype=torch.float32
            )
            for column in range(logits.shape[1]):
                valid_indices = torch.nonzero(
                    candidate_valid[:, column], as_tuple=False
                ).flatten()
                if not len(valid_indices):
                    continue
                nearest_local = torch.argmin(
                    torch.abs(states[valid_indices] - target[column])
                )
                true_index = valid_indices[nearest_local]
                ordering = torch.argsort(logits[valid_indices, column], descending=True)
                rank = int(
                    torch.nonzero(ordering == nearest_local, as_tuple=False)[0, 0]
                ) + 1
                rank_values.append(rank)
                top1_values.append(rank <= 1)
                top5_values.append(rank <= 5)
                true_valid_values.append(
                    float(torch.min(torch.abs(states[valid_indices] - target[column])))
                    <= 0.5
                )

    valid = counts > 0
    eval_rows = record.eval_index
    query_md = record.md[eval_rows].astype(np.float64)
    source_md = record.md[record.node_rows[valid]].astype(np.float64)
    predictions = np.stack(
        [
            np.interp(query_md, source_md, sums[index, valid] / counts[valid])
            for index in range(len(DOSES_R3D))
        ],
        axis=0,
    )
    truth = record.truth[eval_rows].astype(np.float64)
    distance = query_md - float(record.md[record.anchor_index])
    rank_payload = {
        "ranks": rank_values,
        "top1": top1_values,
        "top5": top5_values,
        "true_valid": true_valid_values,
    }
    return predictions, truth, distance, rank_payload


def _dose_curve_r3d(model, records, device, control):
    predictions = [[] for _ in DOSES_R3D]
    truths = []
    distances = []
    wells = []
    all_ranks = []
    all_top1 = []
    all_top5 = []
    all_true_valid = []
    shuffled = list(records[1:]) + list(records[:1]) if len(records) > 1 else records

    for progress, (record, shuffled_record) in enumerate(zip(records, shuffled), 1):
        decoded, truth, distance, rank_payload = _decode_record_doses_r3d(
            model, record, device, control, shuffled_record
        )
        for index in range(len(DOSES_R3D)):
            predictions[index].append(decoded[index])
        truths.append(truth)
        distances.append(distance)
        wells.append(record.well)
        all_ranks.extend(rank_payload["ranks"])
        all_top1.extend(rank_payload["top1"])
        all_top5.extend(rank_payload["top5"])
        all_true_valid.extend(rank_payload["true_valid"])
        if progress % 50 == 0 or progress == len(records):
            print(f"R3 signed-dose {control} {progress}/{len(records)}", flush=True)

    rows = []
    for dose, dose_predictions in zip(DOSES_R3D, predictions):
        metrics = _metric_payload_r1(wells, truths, dose_predictions, distances)
        metrics["dose"] = float(dose)
        rows.append(metrics)
    rank_summary = {
        "columns": int(len(all_ranks)),
        "median_truth_rank": float(np.median(all_ranks)) if all_ranks else math.nan,
        "p90_truth_rank": float(np.quantile(all_ranks, 0.90)) if all_ranks else math.nan,
        "top1_truth_rate": float(np.mean(all_top1)) if all_top1 else math.nan,
        "top5_truth_rate": float(np.mean(all_top5)) if all_top5 else math.nan,
        "true_state_in_valid_axis": (
            float(np.mean(all_true_valid)) if all_true_valid else math.nan
        ),
    }
    return pd.DataFrame(rows), rank_summary


started_r3d = time.time()
_seed_everything(SEED)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the R3 signed-dose audit")
device_r3d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds_r3d, fold_path_r3d = _load_folds()
train_wells_r3d = folds_r3d.loc[
    folds_r3d["fold"].isin(TRAIN_FOLDS_R2), "well"
].tolist()
fold3_wells_r3d = folds_r3d.loc[
    folds_r3d["fold"].eq(VALID_FOLD_R2), "well"
].tolist()
if len(train_wells_r3d) != 309 or len(fold3_wells_r3d) != 154:
    raise RuntimeError(
        f"Unexpected Fold45/Fold3 sizes: {len(train_wells_r3d)}/{len(fold3_wells_r3d)}"
    )

records_r3d = {
    record.well: record
    for record in _load_records(fold3_wells_r3d, "r3_signed_dose_fold3")
}
fold3_records_r3d = [records_r3d[well] for well in fold3_wells_r3d]
order_r3d = np.argsort([record.eval_rows for record in fold3_records_r3d])
selection_positions_r3d = np.unique(
    np.rint(np.linspace(0, len(order_r3d) - 1, SELECTION_WELLS_R2)).astype(int)
)
selection_indices_r3d = set(order_r3d[selection_positions_r3d].tolist())
selection_records_r3d = [
    record
    for index, record in enumerate(fold3_records_r3d)
    if index in selection_indices_r3d
]
confirmation_records_r3d = [
    record
    for index, record in enumerate(fold3_records_r3d)
    if index not in selection_indices_r3d
]

checkpoint_candidates_r3d = (
    OOF_ROOT
    / "segn_r3_querycausal_absolute_matcher_fold3_v0"
    / "models"
    / "best_absolute_matcher.pt",
    Path("/root/autodl-tmp/oof_exports")
    / "segn_r3_querycausal_absolute_matcher_fold3_v0"
    / "models"
    / "best_absolute_matcher.pt",
)
checkpoint_r3d = next(
    (path for path in checkpoint_candidates_r3d if path.exists()), None
)
if checkpoint_r3d is None:
    raise FileNotFoundError(
        "Cannot find the frozen R3 best_absolute_matcher.pt checkpoint"
    )

state_r3d, checkpoint_payload_r3d = _checkpoint_state_r3d(checkpoint_r3d)
model_r3d = QueryCausalMatcherR3().to(device_r3d)
model_r3d.load_state_dict(state_r3d)
model_r3d.eval()

print(
    "R3 SIGNED-DOSE CONFIG",
    {
        "checkpoint": str(checkpoint_r3d),
        "selection_wells": len(selection_records_r3d),
        "confirmation_wells": len(confirmation_records_r3d),
        "doses": DOSES_R3D.tolist(),
        "fold2_touched": False,
        "fold1_touched": False,
    },
    flush=True,
)

selection_curve_r3d, selection_ranks_r3d = _dose_curve_r3d(
    model_r3d, selection_records_r3d, device_r3d, control="full"
)
selection_curve_r3d = selection_curve_r3d.sort_values("dose").reset_index(drop=True)
selected_row_r3d = selection_curve_r3d.sort_values(
    ["rmse", "dose"], ascending=[True, False]
).iloc[0]
selected_dose_r3d = float(selected_row_r3d["dose"])

confirmation_curve_r3d, confirmation_ranks_r3d = _dose_curve_r3d(
    model_r3d, confirmation_records_r3d, device_r3d, control="full"
)
shuffled_curve_r3d, _ = _dose_curve_r3d(
    model_r3d, confirmation_records_r3d, device_r3d, control="shuffled_typewell"
)

selection_curve_r3d.to_csv(OUT_DIR_R3D / "selection_dose_curve.csv", index=False)
confirmation_curve_r3d.to_csv(
    OUT_DIR_R3D / "confirmation_dose_curve.csv", index=False
)
shuffled_curve_r3d.to_csv(
    OUT_DIR_R3D / "confirmation_shuffled_dose_curve.csv", index=False
)


def _row_for_dose_r3d(frame, dose):
    index = int(np.argmin(np.abs(frame["dose"].to_numpy(float) - float(dose))))
    return frame.iloc[index].to_dict()


confirmation_uniform_r3d = _row_for_dose_r3d(confirmation_curve_r3d, 0.0)
confirmation_raw_r3d = _row_for_dose_r3d(confirmation_curve_r3d, 1.0)
confirmation_selected_r3d = _row_for_dose_r3d(
    confirmation_curve_r3d, selected_dose_r3d
)
shuffled_selected_r3d = _row_for_dose_r3d(shuffled_curve_r3d, selected_dose_r3d)

gain_vs_uniform_r3d = float(
    confirmation_uniform_r3d["rmse"] - confirmation_selected_r3d["rmse"]
)
shuffle_degradation_r3d = float(
    shuffled_selected_r3d["rmse"] - confirmation_selected_r3d["rmse"]
)

if abs(selected_dose_r3d) < 1.0e-12:
    interpretation_r3d = "uniform_wins_no_usable_candidate_ordering"
elif selected_dose_r3d < 0.0:
    interpretation_r3d = "negative_dose_wins_candidate_ordering_is_reversed"
elif selected_dose_r3d < 1.0:
    interpretation_r3d = "positive_shrunk_dose_wins_ordering_exists_but_is_overconfident"
else:
    interpretation_r3d = "positive_full_or_sharpened_dose_wins"

checks_r3d = {
    "selected_dose_positive": selected_dose_r3d > 0.0,
    "confirmation_gain_vs_uniform_ge_0p25": gain_vs_uniform_r3d >= 0.25,
    "confirmation_rmse_le_5p5": confirmation_selected_r3d["rmse"] <= 5.5,
    "shuffled_degradation_ge_0p30": shuffle_degradation_r3d >= 0.30,
}
strict_pass_r3d = bool(all(checks_r3d.values()))

verdict_r3d = {
    "experiment": EXPERIMENT_R3D,
    "stage": "fold3_frozen_signed_logit_dose_confirmation",
    "source": str(SOURCE_PATH_R3D),
    "checkpoint": str(checkpoint_r3d),
    "sample_unit": "teacher-anchored chunks nested within complete wells",
    "deployable": False,
    "optimizer_steps": 0,
    "protocol": {
        "selection_wells": len(selection_records_r3d),
        "confirmation_wells": len(confirmation_records_r3d),
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "selected_dose": selected_dose_r3d,
    "selection": selected_row_r3d.to_dict(),
    "confirmation_uniform": confirmation_uniform_r3d,
    "confirmation_raw_dose_1": confirmation_raw_r3d,
    "confirmation_selected": confirmation_selected_r3d,
    "confirmation_shuffled_selected": shuffled_selected_r3d,
    "confirmation_gain_vs_uniform": gain_vs_uniform_r3d,
    "confirmation_shuffle_degradation": shuffle_degradation_r3d,
    "selection_rank_diagnostics": selection_ranks_r3d,
    "confirmation_rank_diagnostics": confirmation_ranks_r3d,
    "interpretation": interpretation_r3d,
    "checks": checks_r3d,
    "strict_pass": strict_pass_r3d,
    "elapsed_minutes": (time.time() - started_r3d) / 60.0,
    "next": (
        "PASS -> retain R3 ordering and move the frozen dose into recursive A/B"
        if strict_pass_r3d
        else "FAIL -> R3 bilinear/cosine ordering is not useful; replace the evidence family"
    ),
}

with (OUT_DIR_R3D / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict_r3d, handle, indent=2)

print("\nR3 SELECTION DOSE CURVE")
print(
    selection_curve_r3d[
        [
            "dose",
            "rmse",
            "median_well_rmse",
            "p95_well_rmse",
            "near_0_500_rmse",
            "far_gt2000_rmse",
        ]
    ].to_string(index=False)
)
print("\nR3 CONFIRMATION DOSE CURVE")
print(
    confirmation_curve_r3d[
        [
            "dose",
            "rmse",
            "median_well_rmse",
            "p95_well_rmse",
            "near_0_500_rmse",
            "far_gt2000_rmse",
        ]
    ].to_string(index=False)
)
print("\nR3 SIGNED-DOSE FINAL VERDICT")
print(json.dumps(verdict_r3d, indent=2), flush=True)
print("saved:", OUT_DIR_R3D, flush=True)

if not strict_pass_r3d:
    raise RuntimeError(json.dumps(verdict_r3d, indent=2))
