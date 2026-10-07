"""Zero-training nested dose audit for the frozen SEGN-R1 checkpoint.

The R1 confirmation showed that its geometry-only prior is strong under a
teacher anchor while the learned gate was forced to consume a weak matching
slot.  This cell asks the narrower investment question before any retraining:
does any frozen GR matching map improve that prior at a non-zero dose?

Fold-3 selection wells choose one arm and dose.  The remaining Fold-3 wells
confirm it once, followed by matched no-GR and shuffled-typewell controls.
Fold 1 and Fold 2 are never loaded, and no optimizer is created.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SOURCE_CANDIDATES_D2 = (
    Path("/root/autodl-tmp/segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py"),
    Path.cwd() / "segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py",
)
SOURCE_PATH_D2 = next((path for path in SOURCE_CANDIDATES_D2 if path.exists()), None)
if SOURCE_PATH_D2 is None:
    raise FileNotFoundError(
        "Upload segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py to "
        "/root/autodl-tmp before running the dose audit."
    )

source_d2 = SOURCE_PATH_D2.read_text(encoding="utf-8")
marker_d2 = "\nstarted_r1 = time.time()"
if marker_d2 not in source_d2:
    raise RuntimeError(f"Cannot find gateway marker in {SOURCE_PATH_D2}")
exec(
    compile(source_d2.split(marker_d2, 1)[0], str(SOURCE_PATH_D2), "exec"),
    globals(),
)


EXPERIMENT_D2 = "segn_r1_prior_matching_dose_audit_v2"
OUT_DIR_D2 = OOF_ROOT / EXPERIMENT_D2
OUT_DIR_D2.mkdir(parents=True, exist_ok=True)
DOSES_D2 = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
ARM_NAMES_D2 = (
    "bilinear",
    "pair_mlp",
    "costmix",
    "evidence",
    "w1",
    "bilinear_costmix",
    "learned_matching_gate",
)


def _configs_d2():
    output = [{"config_id": "prior", "arm": "prior", "dose": 0.0}]
    for arm in ARM_NAMES_D2:
        for dose in DOSES_D2:
            output.append(
                {
                    "config_id": f"{arm}__dose{dose:g}",
                    "arm": arm,
                    "dose": float(dose),
                }
            )
    return output


CONFIGS_D2 = _configs_d2()


@torch.no_grad()
def _maps_d2(model, payload, device):
    tensors = _tensor_payload(payload, device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        _, auxiliary = model(*tensors, return_aux=True)
    slots = auxiliary["slot_maps"].float()
    matching_gate = auxiliary["gate_weights"][:5].float()
    matching_gate = matching_gate / matching_gate.sum(dim=0, keepdim=True).clamp_min(1.0e-8)
    return {
        "prior": slots[5],
        "bilinear": slots[0],
        "pair_mlp": slots[1],
        "costmix": slots[2],
        "evidence": slots[3],
        "w1": slots[4],
        "bilinear_costmix": 0.5 * (slots[0] + slots[2]),
        "learned_matching_gate": (
            matching_gate[:, None, :] * slots[:5]
        ).sum(dim=0),
    }


@torch.no_grad()
def _evaluate_configs_d2(model, records, configs, device, control="full"):
    model.eval()
    config_ids = [config["config_id"] for config in configs]
    collectors = {
        config_id: {"wells": [], "truths": [], "predictions": [], "distances": []}
        for config_id in config_ids
    }
    states = torch.as_tensor(STATE_OFFSETS, device=device, dtype=torch.float32)
    shuffled = list(records[1:]) + list(records[:1]) if len(records) > 1 else list(records)

    for number, (record, shuffled_record) in enumerate(zip(records, shuffled), 1):
        node_count = len(record.node_rows)
        sums = np.zeros((len(configs), node_count), np.float64)
        counts = np.zeros((len(configs), node_count), np.float64)
        anchor_truth = float(record.truth[record.node_rows[0]])
        sums[:, 0] = anchor_truth
        counts[:, 0] = 1.0

        for raw_start in record.chunk_starts:
            start = int(raw_start)
            start_row = int(record.node_rows[start])
            payload = _chunk_payload(
                record,
                start,
                deployed_anchor=float(record.truth[start_row]),
                previous_delta=0.0,
                previous_rate=0.0,
                control=control,
                shuffled_typewell=shuffled_record,
            )
            maps = _maps_d2(model, payload, device)
            prior = maps["prior"]
            target_columns = prior.shape[1]
            end = start + target_columns
            anchor = float(payload["deployed_anchor"])

            mixed_maps = []
            for config in configs:
                if config["arm"] == "prior":
                    mixed = prior
                else:
                    dose = float(config["dose"])
                    mixed = (1.0 - dose) * prior + dose * maps[config["arm"]]
                mixed_maps.append(mixed)
            mixed_maps = torch.stack(mixed_maps, dim=0)
            probability = F.softmax(mixed_maps, dim=1)
            expected = (probability * states[None, :, None]).sum(dim=1)
            expected = expected.cpu().numpy().astype(np.float64)
            sums[:, start + 1 : end + 1] += anchor + expected
            counts[:, start + 1 : end + 1] += 1.0

        eval_rows = record.eval_index
        support_md = record.md[record.node_rows].astype(np.float64)
        query_md = record.md[eval_rows].astype(np.float64)
        truth = record.truth[eval_rows].astype(np.float64)
        distance = query_md - float(record.md[record.anchor_index])
        for config_index, config_id in enumerate(config_ids):
            valid = counts[config_index] > 0
            node_prediction = sums[config_index, valid] / counts[config_index, valid]
            prediction = np.interp(query_md, support_md[valid], node_prediction)
            collector = collectors[config_id]
            collector["wells"].append(record.well)
            collector["truths"].append(truth)
            collector["predictions"].append(prediction)
            collector["distances"].append(distance)

        if number % 25 == 0 or number == len(records):
            print(
                f"SEGN-R1 DOSE evaluate {control} {number}/{len(records)}",
                flush=True,
            )

    return {
        config_id: _metric_payload_r1(**collector)
        for config_id, collector in collectors.items()
    }


started_d2 = time.time()
_seed_everything(SEED)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the SEGN-R1 dose audit")
device_d2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path_d2 = (
    OOF_ROOT
    / "segn_r1_fiveslot_costmap_fold3_gateway_v0"
    / "models"
    / "best_fold3_selection.pt"
)
if not checkpoint_path_d2.exists():
    raise FileNotFoundError(checkpoint_path_d2)
checkpoint_d2 = torch.load(
    checkpoint_path_d2, map_location=device_d2, weights_only=False
)
model_d2 = FiveSlotCostMapR1().to(device_d2)
model_d2.load_state_dict(checkpoint_d2["model"])
model_d2.eval()

folds_d2, fold_path_d2 = _load_folds()
fold3_wells_d2 = folds_d2.loc[
    folds_d2["fold"].eq(SELECTION_FOLD_R1), "well"
].tolist()
if len(fold3_wells_d2) != 154:
    raise RuntimeError(f"Unexpected Fold-3 size: {len(fold3_wells_d2)}")
fold3_records_d2 = _load_records(fold3_wells_d2, "r1_dose_fold3")

order_d2 = np.argsort([record.eval_rows for record in fold3_records_d2])
selection_positions_d2 = np.unique(
    np.rint(np.linspace(0, len(order_d2) - 1, SELECTION_WELLS_R1)).astype(int)
)
selection_indices_d2 = set(order_d2[selection_positions_d2].tolist())
selection_records_d2 = [
    record
    for index, record in enumerate(fold3_records_d2)
    if index in selection_indices_d2
]
confirmation_records_d2 = [
    record
    for index, record in enumerate(fold3_records_d2)
    if index not in selection_indices_d2
]

selection_metrics_d2 = _evaluate_configs_d2(
    model_d2, selection_records_d2, CONFIGS_D2, device_d2, control="full"
)
selection_table_d2 = []
prior_selection_rmse_d2 = selection_metrics_d2["prior"]["rmse"]
for config_d2 in CONFIGS_D2:
    metrics_d2 = selection_metrics_d2[config_d2["config_id"]]
    selection_table_d2.append(
        {
            **config_d2,
            "rows": metrics_d2["rows"],
            "wells": metrics_d2["wells"],
            "exact_sse": metrics_d2["exact_sse"],
            "rmse": metrics_d2["rmse"],
            "gain_vs_prior": prior_selection_rmse_d2 - metrics_d2["rmse"],
            "far_gt2000_rmse": metrics_d2["far_gt2000_rmse"],
        }
    )
selection_frame_d2 = pd.DataFrame(selection_table_d2).sort_values(
    ["rmse", "dose", "config_id"]
)
selection_frame_d2.to_csv(OUT_DIR_D2 / "selection_dose_scan.csv", index=False)
print("SEGN-R1 DOSE SELECTION TOP")
print(selection_frame_d2.head(20).to_string(index=False), flush=True)

selected_d2 = selection_frame_d2.iloc[0].to_dict()
selected_config_d2 = next(
    config for config in CONFIGS_D2 if config["config_id"] == selected_d2["config_id"]
)
confirmation_configs_d2 = [CONFIGS_D2[0]]
if selected_config_d2["config_id"] != "prior":
    confirmation_configs_d2.append(selected_config_d2)

confirmation_full_d2 = _evaluate_configs_d2(
    model_d2,
    confirmation_records_d2,
    confirmation_configs_d2,
    device_d2,
    control="full",
)
confirmation_no_gr_d2 = _evaluate_configs_d2(
    model_d2,
    confirmation_records_d2,
    confirmation_configs_d2,
    device_d2,
    control="no_gr",
)
confirmation_shuffled_d2 = _evaluate_configs_d2(
    model_d2,
    confirmation_records_d2,
    confirmation_configs_d2,
    device_d2,
    control="shuffled_typewell",
)

candidate_id_d2 = selected_config_d2["config_id"]
prior_full_d2 = confirmation_full_d2["prior"]
candidate_full_d2 = confirmation_full_d2[candidate_id_d2]
prior_no_gr_d2 = confirmation_no_gr_d2["prior"]
candidate_no_gr_d2 = confirmation_no_gr_d2[candidate_id_d2]
prior_shuffled_d2 = confirmation_shuffled_d2["prior"]
candidate_shuffled_d2 = confirmation_shuffled_d2[candidate_id_d2]

full_gain_d2 = prior_full_d2["rmse"] - candidate_full_d2["rmse"]
no_gr_gain_d2 = prior_no_gr_d2["rmse"] - candidate_no_gr_d2["rmse"]
shuffled_gain_d2 = prior_shuffled_d2["rmse"] - candidate_shuffled_d2["rmse"]
no_gr_retention_d2 = no_gr_gain_d2 / full_gain_d2 if full_gain_d2 > 1.0e-8 else math.nan
shuffled_retention_d2 = (
    shuffled_gain_d2 / full_gain_d2 if full_gain_d2 > 1.0e-8 else math.nan
)

checks_d2 = {
    "selection_chooses_nonzero_matching_dose": selected_config_d2["dose"] > 0.0,
    "confirmation_gain_ge_0p10": full_gain_d2 >= 0.10,
    "confirmation_far_gain_ge_0p10": (
        prior_full_d2["far_gt2000_rmse"]
        - candidate_full_d2["far_gt2000_rmse"]
        >= 0.10
    ),
    "no_gr_retains_le_30pct_gain": (
        np.isfinite(no_gr_retention_d2) and no_gr_retention_d2 <= 0.30
    ),
    "shuffle_retains_le_50pct_gain": (
        np.isfinite(shuffled_retention_d2) and shuffled_retention_d2 <= 0.50
    ),
}
strict_pass_d2 = bool(all(checks_d2.values()))

verdict_d2 = {
    "experiment": EXPERIMENT_D2,
    "stage": "nested_zero_training_prior_to_matching_dose_audit",
    "source_experiment": "segn_r1_fiveslot_costmap_fold3_gateway_v0",
    "checkpoint": str(checkpoint_path_d2),
    "protocol": {
        "train_folds_of_frozen_model": list(TRAIN_FOLDS_R1),
        "selection_fold": SELECTION_FOLD_R1,
        "selection_wells": len(selection_records_d2),
        "confirmation_wells": len(confirmation_records_d2),
        "fold2_touched": False,
        "fold1_touched": False,
        "optimizer_steps": 0,
    },
    "selected": selected_config_d2,
    "selection_prior_rmse": prior_selection_rmse_d2,
    "selection_candidate": selection_metrics_d2[candidate_id_d2],
    "confirmation_prior": prior_full_d2,
    "confirmation_candidate": candidate_full_d2,
    "confirmation_gain": full_gain_d2,
    "confirmation_far_gain": (
        prior_full_d2["far_gt2000_rmse"]
        - candidate_full_d2["far_gt2000_rmse"]
    ),
    "controls": {
        "no_gr_prior": prior_no_gr_d2,
        "no_gr_candidate": candidate_no_gr_d2,
        "no_gr_gain": no_gr_gain_d2,
        "no_gr_gain_retention": no_gr_retention_d2,
        "shuffled_prior": prior_shuffled_d2,
        "shuffled_candidate": candidate_shuffled_d2,
        "shuffled_gain": shuffled_gain_d2,
        "shuffled_gain_retention": shuffled_retention_d2,
    },
    "checks": checks_d2,
    "strict_pass": strict_pass_d2,
    "elapsed_minutes": (time.time() - started_d2) / 60.0,
    "next": (
        "PASS -> retrain a prior-residual gate without anti-collapse regularisation"
        if strict_pass_d2
        else "FAIL -> frozen matching maps add no causal value over the local prior"
    ),
}
_write_json(OUT_DIR_D2 / "verdict.json", verdict_d2)
print("SEGN-R1 DOSE FINAL VERDICT", json.dumps(verdict_d2, indent=2), flush=True)
print("saved:", OUT_DIR_D2, flush=True)

