"""Zero-training recursive rollout audit for the frozen SEGN-R1 prior.

R1 proved a 4.585-ft geometry prior under true chunk anchors, but that number
is not deployable.  This nested Fold-3 audit recursively feeds each candidate's
own posterior-mean anchor into the next chunk.  It compares prior-only rollout,
small matching doses, and the original learned fusion.  No optimizer is used;
Fold 1 and Fold 2 are never loaded.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SOURCE_CANDIDATES_R3 = (
    Path("/root/autodl-tmp/segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py"),
    Path.cwd() / "segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py",
)
SOURCE_PATH_R3 = next((path for path in SOURCE_CANDIDATES_R3 if path.exists()), None)
if SOURCE_PATH_R3 is None:
    raise FileNotFoundError(
        "Upload segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py to "
        "/root/autodl-tmp before running the rollout gateway."
    )

source_r3 = SOURCE_PATH_R3.read_text(encoding="utf-8")
marker_r3 = "\nstarted_r1 = time.time()"
if marker_r3 not in source_r3:
    raise RuntimeError(f"Cannot find gateway marker in {SOURCE_PATH_R3}")
exec(
    compile(source_r3.split(marker_r3, 1)[0], str(SOURCE_PATH_R3), "exec"),
    globals(),
)


EXPERIMENT_R3 = "segn_r1_prior_rollout_gateway_v3"
OUT_DIR_R3 = OOF_ROOT / EXPERIMENT_R3
OUT_DIR_R3.mkdir(parents=True, exist_ok=True)
CONFIGS_R3 = (
    {"config_id": "prior", "arm": "prior", "dose": 0.0},
    {"config_id": "costmix__dose0.005", "arm": "costmix", "dose": 0.005},
    {"config_id": "costmix__dose0.01", "arm": "costmix", "dose": 0.01},
    {"config_id": "costmix__dose0.02", "arm": "costmix", "dose": 0.02},
    {
        "config_id": "bilinear_costmix__dose0.01",
        "arm": "bilinear_costmix",
        "dose": 0.01,
    },
    {"config_id": "learned_fused", "arm": "learned_fused", "dose": math.nan},
)


@torch.no_grad()
def _rollout_map_r3(model, payload, device, config):
    tensors = _tensor_payload(payload, device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        fused, auxiliary = model(*tensors, return_aux=True)
    fused = fused.float()
    slots = auxiliary["slot_maps"].float()
    if config["arm"] == "learned_fused":
        logits = fused
    else:
        prior = slots[5]
        if config["arm"] == "prior":
            logits = prior
        elif config["arm"] == "costmix":
            dose = float(config["dose"])
            logits = (1.0 - dose) * prior + dose * slots[2]
        elif config["arm"] == "bilinear_costmix":
            dose = float(config["dose"])
            matching = 0.5 * (slots[0] + slots[2])
            logits = (1.0 - dose) * prior + dose * matching
        else:
            raise KeyError(config["arm"])
    states = torch.as_tensor(STATE_OFFSETS, device=device, dtype=torch.float32)
    probability = F.softmax(logits.float(), dim=0)
    expected = (probability * states[:, None]).sum(dim=0)
    return expected.cpu().numpy().astype(np.float64)


@torch.no_grad()
def _rollout_record_r3(
    model,
    record,
    device,
    config,
    control="full",
    shuffled_record=None,
):
    node_prediction = np.full(len(record.node_rows), np.nan, np.float64)
    node_prediction[0] = float(record.anchor_tvt)
    previous_start = 0
    previous_anchor = float(record.anchor_tvt)

    for raw_start in record.chunk_starts:
        start = int(raw_start)
        if np.isfinite(node_prediction[start]):
            deployed_anchor = float(node_prediction[start])
        else:
            prior_nodes = np.flatnonzero(np.isfinite(node_prediction[: start + 1]))
            if not len(prior_nodes):
                raise RuntimeError(f"{record.well}: rollout lost its anchor")
            deployed_anchor = float(node_prediction[prior_nodes[-1]])

        start_row = int(record.node_rows[start])
        previous_row = int(record.node_rows[previous_start])
        md_delta = max(float(record.md[start_row] - record.md[previous_row]), 1.0)
        previous_delta = deployed_anchor - previous_anchor if start > 0 else 0.0
        previous_rate = previous_delta / md_delta if start > 0 else 0.0
        payload = _chunk_payload(
            record,
            start,
            deployed_anchor=deployed_anchor,
            previous_delta=previous_delta,
            previous_rate=previous_rate,
            control=control,
            shuffled_typewell=shuffled_record,
        )
        expected = _rollout_map_r3(model, payload, device, config)
        end = start + len(expected)
        node_prediction[start + 1 : end + 1] = deployed_anchor + expected
        previous_start = start
        previous_anchor = deployed_anchor

    valid = np.isfinite(node_prediction)
    if int(valid.sum()) < 2:
        raise RuntimeError(f"{record.well}: rollout has too few valid nodes")
    eval_rows = record.eval_index
    query_md = record.md[eval_rows].astype(np.float64)
    prediction = np.interp(
        query_md,
        record.md[record.node_rows[valid]].astype(np.float64),
        node_prediction[valid],
    )
    truth = record.truth[eval_rows].astype(np.float64)
    distance = query_md - float(record.md[record.anchor_index])
    return truth, prediction, distance


@torch.no_grad()
def _evaluate_rollout_r3(model, records, configs, device, control="full"):
    model.eval()
    collectors = {
        config["config_id"]: {
            "wells": [],
            "truths": [],
            "predictions": [],
            "distances": [],
        }
        for config in configs
    }
    shuffled = list(records[1:]) + list(records[:1]) if len(records) > 1 else list(records)
    for number, (record, shuffled_record) in enumerate(zip(records, shuffled), 1):
        for config in configs:
            truth, prediction, distance = _rollout_record_r3(
                model,
                record,
                device,
                config,
                control=control,
                shuffled_record=shuffled_record,
            )
            collector = collectors[config["config_id"]]
            collector["wells"].append(record.well)
            collector["truths"].append(truth)
            collector["predictions"].append(prediction)
            collector["distances"].append(distance)
        if number % 25 == 0 or number == len(records):
            print(
                f"SEGN-R1 ROLLOUT evaluate {control} {number}/{len(records)}",
                flush=True,
            )
    return {
        config_id: _metric_payload_r1(**collector)
        for config_id, collector in collectors.items()
    }


started_r3 = time.time()
_seed_everything(SEED)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the SEGN-R1 rollout gateway")
device_r3 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path_r3 = (
    OOF_ROOT
    / "segn_r1_fiveslot_costmap_fold3_gateway_v0"
    / "models"
    / "best_fold3_selection.pt"
)
if not checkpoint_path_r3.exists():
    raise FileNotFoundError(checkpoint_path_r3)
checkpoint_r3 = torch.load(
    checkpoint_path_r3, map_location=device_r3, weights_only=False
)
model_r3 = FiveSlotCostMapR1().to(device_r3)
model_r3.load_state_dict(checkpoint_r3["model"])
model_r3.eval()

folds_r3, fold_path_r3 = _load_folds()
fold3_wells_r3 = folds_r3.loc[
    folds_r3["fold"].eq(SELECTION_FOLD_R1), "well"
].tolist()
if len(fold3_wells_r3) != 154:
    raise RuntimeError(f"Unexpected Fold-3 size: {len(fold3_wells_r3)}")
fold3_records_r3 = _load_records(fold3_wells_r3, "r1_rollout_fold3")

order_r3 = np.argsort([record.eval_rows for record in fold3_records_r3])
selection_positions_r3 = np.unique(
    np.rint(np.linspace(0, len(order_r3) - 1, SELECTION_WELLS_R1)).astype(int)
)
selection_indices_r3 = set(order_r3[selection_positions_r3].tolist())
selection_records_r3 = [
    record
    for index, record in enumerate(fold3_records_r3)
    if index in selection_indices_r3
]
confirmation_records_r3 = [
    record
    for index, record in enumerate(fold3_records_r3)
    if index not in selection_indices_r3
]

selection_metrics_r3 = _evaluate_rollout_r3(
    model_r3, selection_records_r3, CONFIGS_R3, device_r3, control="full"
)
selection_rows_r3 = []
for config_r3 in CONFIGS_R3:
    metrics_r3 = selection_metrics_r3[config_r3["config_id"]]
    selection_rows_r3.append(
        {
            **config_r3,
            "rows": metrics_r3["rows"],
            "wells": metrics_r3["wells"],
            "exact_sse": metrics_r3["exact_sse"],
            "rmse": metrics_r3["rmse"],
            "median_well_rmse": metrics_r3["median_well_rmse"],
            "p95_well_rmse": metrics_r3["p95_well_rmse"],
            "far_gt2000_rmse": metrics_r3["far_gt2000_rmse"],
        }
    )
selection_frame_r3 = pd.DataFrame(selection_rows_r3).sort_values(
    ["rmse", "config_id"]
)
selection_frame_r3.to_csv(OUT_DIR_R3 / "selection_rollout_scan.csv", index=False)
print("SEGN-R1 ROLLOUT SELECTION")
print(selection_frame_r3.to_string(index=False), flush=True)

selected_row_r3 = selection_frame_r3.iloc[0].to_dict()
selected_config_r3 = next(
    config for config in CONFIGS_R3 if config["config_id"] == selected_row_r3["config_id"]
)
confirmation_configs_r3 = [CONFIGS_R3[0]]
if selected_config_r3["config_id"] != "prior":
    confirmation_configs_r3.append(selected_config_r3)

confirmation_full_r3 = _evaluate_rollout_r3(
    model_r3,
    confirmation_records_r3,
    confirmation_configs_r3,
    device_r3,
    control="full",
)
confirmation_no_gr_r3 = _evaluate_rollout_r3(
    model_r3,
    confirmation_records_r3,
    confirmation_configs_r3,
    device_r3,
    control="no_gr",
)
confirmation_shuffled_r3 = _evaluate_rollout_r3(
    model_r3,
    confirmation_records_r3,
    confirmation_configs_r3,
    device_r3,
    control="shuffled_typewell",
)

selected_id_r3 = selected_config_r3["config_id"]
prior_full_r3 = confirmation_full_r3["prior"]
selected_full_r3 = confirmation_full_r3[selected_id_r3]
prior_no_gr_r3 = confirmation_no_gr_r3["prior"]
selected_no_gr_r3 = confirmation_no_gr_r3[selected_id_r3]
prior_shuffled_r3 = confirmation_shuffled_r3["prior"]
selected_shuffled_r3 = confirmation_shuffled_r3[selected_id_r3]

gain_r3 = prior_full_r3["rmse"] - selected_full_r3["rmse"]
far_gain_r3 = (
    prior_full_r3["far_gt2000_rmse"] - selected_full_r3["far_gt2000_rmse"]
)
no_gr_gain_r3 = prior_no_gr_r3["rmse"] - selected_no_gr_r3["rmse"]
shuffled_gain_r3 = prior_shuffled_r3["rmse"] - selected_shuffled_r3["rmse"]

checks_r3 = {
    "prior_rollout_rmse_le_10p5": prior_full_r3["rmse"] <= 10.5,
    "prior_rollout_far_rmse_le_12": prior_full_r3["far_gt2000_rmse"] <= 12.0,
    "selected_rollout_rmse_le_10": selected_full_r3["rmse"] <= 10.0,
    "selected_gain_vs_prior_ge_0p10": gain_r3 >= 0.10,
    "selected_far_gain_ge_0p10": far_gain_r3 >= 0.10,
}
strict_pass_r3 = bool(all(checks_r3.values()))

verdict_r3 = {
    "experiment": EXPERIMENT_R3,
    "stage": "nested_zero_training_recursive_prior_rollout_gateway",
    "source_experiment": "segn_r1_fiveslot_costmap_fold3_gateway_v0",
    "checkpoint": str(checkpoint_path_r3),
    "sample_unit": "one complete physical well",
    "deployable_inputs_only": True,
    "truth_usage": "metrics only; every post-prefix anchor is model-recursive",
    "protocol": {
        "train_folds_of_frozen_model": list(TRAIN_FOLDS_R1),
        "selection_fold": SELECTION_FOLD_R1,
        "selection_wells": len(selection_records_r3),
        "confirmation_wells": len(confirmation_records_r3),
        "fold2_touched": False,
        "fold1_touched": False,
        "optimizer_steps": 0,
    },
    "selected": selected_config_r3,
    "selection": selection_metrics_r3[selected_id_r3],
    "confirmation_prior": prior_full_r3,
    "confirmation_selected": selected_full_r3,
    "gain_vs_prior": gain_r3,
    "far_gain_vs_prior": far_gain_r3,
    "controls": {
        "no_gr_prior": prior_no_gr_r3,
        "no_gr_selected": selected_no_gr_r3,
        "no_gr_gain_vs_prior": no_gr_gain_r3,
        "shuffled_prior": prior_shuffled_r3,
        "shuffled_selected": selected_shuffled_r3,
        "shuffled_gain_vs_prior": shuffled_gain_r3,
    },
    "checks": checks_r3,
    "strict_pass": strict_pass_r3,
    "elapsed_minutes": (time.time() - started_r3) / 60.0,
    "next": (
        "PASS -> use this deployable endpoint in a nested SP45 correlation/blend audit"
        if strict_pass_r3
        else "FAIL -> local geometry is accurate only under teacher anchors; rebuild absolute anchoring"
    ),
}
_write_json(OUT_DIR_R3 / "verdict.json", verdict_r3)
print("SEGN-R1 ROLLOUT FINAL VERDICT", json.dumps(verdict_r3, indent=2), flush=True)
print("saved:", OUT_DIR_R3, flush=True)

