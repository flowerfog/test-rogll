"""V205: legal same-well visible-prefix test-time adaptation gateway.

The frozen V202 correlation model is adapted independently for each target
well using only that well's visible TVT_input prefix. A 24-well Fold3 gateway
selects adaptation steps and compares against donor-prefix adaptation. The
remaining 130 Fold3 wells are opened only if the discovery gate is strong.
Fold2 and Fold1 are untouched.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


STARTED_V205 = time.time()
EXPERIMENT_V205 = "samewell_prefix_adaptation_gateway_v205"
required_v205 = (
    "model", "select_records", "device", "_block_nodes", "_inputs",
    "_row_prediction", "_metrics", "STATE_OFFSETS", "OUT", "SEED",
)
missing_v205 = [name for name in required_v205 if name not in globals()]
if missing_v205:
    raise RuntimeError("Run V202 in this live kernel first. Missing: " + repr(missing_v205))

OUT_V205 = Path(OUT).parent / EXPERIMENT_V205
OUT_V205.mkdir(parents=True, exist_ok=True)
DISCOVERY_WELLS_V205 = 24
PREFIX_WINDOW_FT_V205 = 2048.0
PREFIX_STEP_FT_V205 = 24.0
CHECKPOINT_STEPS_V205 = (0, 4, 8, 16)
LEARNING_RATE_V205 = 1e-4
TARGET_SIGMA_V205 = 2.0
KL_WEIGHT_V205 = 0.05


def _clean_v205(value):
    if isinstance(value, dict):
        return {str(key): _clean_v205(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_v205(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _restore_v202_v205():
    if "base_state_v203" in globals():
        state = copy.deepcopy(base_state_v203)
        model.load_state_dict(state)
        return state, "live_base_state_v203"
    checkpoint_path = Path(OUT) / "models" / "fold45_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state", checkpoint))
    model.load_state_dict(state)
    return copy.deepcopy(model.state_dict()), str(checkpoint_path)


def _prefix_nodes_v205(record):
    known = np.flatnonzero(record.condition[0] > 0.5)
    anchor_md = float(record.md[record.anchor_index])
    known = known[record.md[known] >= anchor_md - PREFIX_WINDOW_FT_V205]
    grid = np.arange(
        max(float(record.md[known[0]]), anchor_md - PREFIX_WINDOW_FT_V205),
        anchor_md + 0.5 * PREFIX_STEP_FT_V205,
        PREFIX_STEP_FT_V205,
    )
    positions = np.searchsorted(record.md[known], grid)
    positions = np.clip(positions, 0, len(known) - 1)
    rows = np.unique(known[positions]).astype(np.int64)
    label = record.condition[1, rows].astype(np.float32) * 128.0
    valid = (
        np.isfinite(label)
        & (label >= float(STATE_OFFSETS[0]))
        & (label <= float(STATE_OFFSETS[-1]))
    )
    rows, label = rows[valid], label[valid]
    if len(rows) < 16:
        raise RuntimeError(f"{record.well}: only {len(rows)} legal prefix nodes")
    return rows, label


def _adaptation_loss_v205(logits, label, base_logits):
    states = torch.as_tensor(STATE_OFFSETS, device=logits.device)[:, None]
    target = torch.as_tensor(label, device=logits.device)[None]
    distribution = torch.exp(-0.5 * ((states - target) / TARGET_SIGMA_V205) ** 2)
    distribution /= distribution.sum(dim=0, keepdim=True).clamp_min(1e-8)
    logp = F.log_softmax(logits[0], dim=0)
    ce = -(distribution * logp).sum(dim=0).mean()
    probability = F.softmax(logits[0], dim=0)
    expected = (probability * states).sum(dim=0)
    path = F.smooth_l1_loss(expected / 8.0, target[0] / 8.0)
    base_probability = F.softmax(base_logits[0], dim=0)
    kl = (
        base_probability
        * (F.log_softmax(base_logits[0], dim=0) - logp)
    ).sum(dim=0).mean()
    return ce + 0.20 * path + KL_WEIGHT_V205 * kl


@torch.no_grad()
def _hidden_item_v205(record):
    rows, label, md, z = _block_nodes(record)
    logits = model(*_inputs(record, rows, device))[0].float().cpu().numpy()
    return {"record": record, "rows": rows, "label": label, "md": md, "z": z, "logits": logits}


def _mean_parts_v205(items, endpoint):
    parts = []
    for item in items:
        logits = item["logits"].astype(np.float64)
        logits -= np.max(logits, axis=0, keepdims=True)
        probability = np.exp(logits)
        probability /= np.sum(probability, axis=0, keepdims=True)
        offsets = STATE_OFFSETS.astype(np.float64) @ probability
        record = item["record"]
        prediction = _row_prediction(record, item["rows"], offsets)
        eval_rows = record.eval_index
        parts.append({
            "well": record.well,
            "row_idx": eval_rows.astype(np.int64),
            "truth": record.truth[eval_rows].astype(np.float64),
            "prediction": prediction,
            "distance": record.md[eval_rows].astype(np.float64) - float(record.md[record.anchor_index]),
        })
    return _metrics(parts, endpoint), parts


def _adapt_one_v205(target_record, adaptation_record, base_state, checkpoints):
    model.load_state_dict(base_state)
    prefix_rows, prefix_label = _prefix_nodes_v205(adaptation_record)
    model.eval()
    with torch.no_grad():
        base_prefix_logits = model(*_inputs(adaptation_record, prefix_rows, device)).detach()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_V205, weight_decay=1e-5)
    output = {}
    maximum = max(checkpoints)
    if 0 in checkpoints:
        output[0] = _hidden_item_v205(target_record)
    for step in range(1, maximum + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(*_inputs(adaptation_record, prefix_rows, device))
        loss = _adaptation_loss_v205(logits, prefix_label, base_prefix_logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in checkpoints:
            model.eval()
            output[step] = _hidden_item_v205(target_record)
    return output


base_state_v205, checkpoint_source_v205 = _restore_v202_v205()
ordered_v205 = sorted(select_records, key=lambda record: record.well)
discovery_v205 = ordered_v205[:DISCOVERY_WELLS_V205]
confirmation_v205 = ordered_v205[DISCOVERY_WELLS_V205:]
aligned_by_step_v205 = {step: [] for step in CHECKPOINT_STEPS_V205}
donor_by_step_v205 = {step: [] for step in CHECKPOINT_STEPS_V205}

for index_v205, record_v205 in enumerate(discovery_v205):
    aligned_v205 = _adapt_one_v205(
        record_v205, record_v205, base_state_v205, CHECKPOINT_STEPS_V205
    )
    donor_record_v205 = discovery_v205[
        (index_v205 + max(1, len(discovery_v205) // 2)) % len(discovery_v205)
    ]
    donor_v205 = _adapt_one_v205(
        record_v205, donor_record_v205, base_state_v205, CHECKPOINT_STEPS_V205
    )
    for step_v205 in CHECKPOINT_STEPS_V205:
        aligned_by_step_v205[step_v205].append(aligned_v205[step_v205])
        donor_by_step_v205[step_v205].append(donor_v205[step_v205])
    print(f"V205 discovery {index_v205 + 1}/{len(discovery_v205)}", flush=True)

scan_rows_v205 = []
for step_v205 in CHECKPOINT_STEPS_V205:
    aligned_metrics_v205, _ = _mean_parts_v205(
        aligned_by_step_v205[step_v205], f"aligned_step{step_v205}"
    )
    donor_metrics_v205, _ = _mean_parts_v205(
        donor_by_step_v205[step_v205], f"donor_step{step_v205}"
    )
    scan_rows_v205.append({
        "steps": step_v205,
        "rmse": aligned_metrics_v205["rmse"],
        "far_gt2000_rmse": aligned_metrics_v205["far_gt2000_rmse"],
        "p95_well_rmse": aligned_metrics_v205["p95_well_rmse"],
        "donor_rmse": donor_metrics_v205["rmse"],
        "aligned_increment_vs_donor": donor_metrics_v205["rmse"] - aligned_metrics_v205["rmse"],
    })
scan_v205 = pd.DataFrame(scan_rows_v205).sort_values("rmse", ignore_index=True)
scan_v205.to_csv(OUT_V205 / "discovery_scan.csv", index=False)
selected_steps_v205 = int(scan_v205.iloc[0]["steps"])
base_rmse_v205 = float(scan_v205.loc[scan_v205["steps"].eq(0), "rmse"].iloc[0])
discovery_gain_v205 = base_rmse_v205 - float(scan_v205.iloc[0]["rmse"])
discovery_pass_v205 = (
    selected_steps_v205 > 0
    and discovery_gain_v205 >= 0.50
    and float(scan_v205.iloc[0]["aligned_increment_vs_donor"]) >= 0.30
)

confirmation_metrics_v205 = None
if discovery_pass_v205:
    confirmation_items_v205 = []
    for index_v205, record_v205 in enumerate(confirmation_v205):
        adapted_v205 = _adapt_one_v205(
            record_v205, record_v205, base_state_v205, (selected_steps_v205,)
        )
        confirmation_items_v205.append(adapted_v205[selected_steps_v205])
        if (index_v205 + 1) % 20 == 0 or index_v205 + 1 == len(confirmation_v205):
            print(f"V205 confirmation {index_v205 + 1}/{len(confirmation_v205)}", flush=True)
    confirmation_metrics_v205, _ = _mean_parts_v205(
        confirmation_items_v205, "fold3_confirmation"
    )

checks_v205 = {
    "discovery_gain_ge_0p50": discovery_gain_v205 >= 0.50,
    "aligned_beats_donor_ge_0p30": float(scan_v205.iloc[0]["aligned_increment_vs_donor"]) >= 0.30,
    "confirmation_rmse_le_10": confirmation_metrics_v205 is not None and confirmation_metrics_v205["rmse"] <= 10.0,
}
strict_pass_v205 = bool(all(checks_v205.values()))
verdict_v205 = {
    "experiment": EXPERIMENT_V205,
    "stage": "Fold45_global_model_Fold3_samewell_prefix_adaptation_gateway",
    "checkpoint_source": checkpoint_source_v205,
    "new_legal_observable": "target-well visible TVT_input used for per-well test-time adaptation",
    "historical_prediction_inputs": [],
    "protocol": {
        "discovery_wells": len(discovery_v205),
        "confirmation_wells": len(confirmation_v205) if discovery_pass_v205 else 0,
        "fold2_touched": 0,
        "fold1_touched": 0,
    },
    "prefix": {
        "window_ft": PREFIX_WINDOW_FT_V205,
        "step_ft": PREFIX_STEP_FT_V205,
        "learning_rate": LEARNING_RATE_V205,
    },
    "selected_steps": selected_steps_v205,
    "discovery_base_rmse": base_rmse_v205,
    "discovery_gain": discovery_gain_v205,
    "discovery_scan": scan_rows_v205,
    "confirmation": confirmation_metrics_v205,
    "checks": checks_v205,
    "strict_pass": strict_pass_v205,
    "elapsed_minutes": (time.time() - STARTED_V205) / 60.0,
    "next": (
        "PASS -> freeze adaptation and evaluate Fold2 once"
        if strict_pass_v205
        else "FAIL -> same-well prefix supervision cannot adapt the V202 correlation engine"
    ),
}
(OUT_V205 / "verdict.json").write_text(
    json.dumps(_clean_v205(verdict_v205), indent=2, allow_nan=False), encoding="utf-8"
)
model.load_state_dict(base_state_v205)

print("\nV205 DISCOVERY SCAN")
print(scan_v205.to_string(index=False))
print("\nV205 FINAL VERDICT")
print(json.dumps(_clean_v205(verdict_v205), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_V205, flush=True)
