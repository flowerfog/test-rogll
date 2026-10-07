"""V203: hard-negative calibration of the high-recall V202 posterior.

Run in the same live kernel after V202/V202b. The frozen V202 best checkpoint
is fine-tuned on Folds 4/5 only with a candidate-ranking objective. Fold3
selects one posterior consumer; Fold2 is evaluated once; Fold1 is untouched.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


STARTED_V203 = time.time()
EXPERIMENT_V203 = "hard_negative_posterior_calibration_v203"
required_v203 = (
    "model", "train_records", "select_records", "gate_records", "device",
    "_block_nodes", "_inputs", "_logits_for_records", "_row_prediction",
    "_metrics", "_top8_oracle", "STATE_OFFSETS", "OUT", "SEED",
)
missing_v203 = [name for name in required_v203 if name not in globals()]
if missing_v203:
    raise RuntimeError(
        "Run the completed V202 pipeline in this live kernel first. Missing: "
        + repr(missing_v203)
    )

OUT_V203 = Path(OUT).parent / EXPERIMENT_V203
MODEL_DIR_V203 = OUT_V203 / "models"
OUT_V203.mkdir(parents=True, exist_ok=True)
MODEL_DIR_V203.mkdir(parents=True, exist_ok=True)
MAX_EPOCHS_V203 = 4
LEARNING_RATE_V203 = 5e-5
MARGIN_V203 = 0.75
POSITIVE_RADIUS_V203 = 2.0
NEGATIVE_RADIUS_V203 = 8.0
HARD_NEGATIVES_V203 = 16
TEMPERATURES_V203 = (0.25, 0.5, 0.75, 1.0, 1.5)
TOPK_V203 = (1, 4, 8, 16, 32)


def _clean_v203(value):
    if isinstance(value, dict):
        return {str(key): _clean_v203(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_v203(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _ranking_loss_v203(logits, label):
    scores = logits[0]
    states = torch.as_tensor(STATE_OFFSETS, device=scores.device)[:, None]
    target = torch.as_tensor(label, device=scores.device)[None]
    valid = torch.isfinite(target[0]) & (target[0] >= states[0, 0]) & (target[0] <= states[-1, 0])
    if not torch.any(valid):
        raise RuntimeError("V203 record has no in-axis labels")
    scores = scores[:, valid]
    target = target[:, valid]
    distance = torch.abs(states - target)
    positive = distance <= POSITIVE_RADIUS_V203
    negative = distance >= NEGATIVE_RADIUS_V203
    positive_score = scores.masked_fill(~positive, -torch.inf).amax(dim=0)
    negative_scores = scores.masked_fill(~negative, -torch.inf)
    hard_k = min(HARD_NEGATIVES_V203, negative_scores.shape[0])
    hard_negative = torch.topk(negative_scores, hard_k, dim=0).values
    ranking = F.softplus(hard_negative - positive_score[None] + MARGIN_V203).mean()

    distribution = torch.exp(-0.5 * (distance / 2.0) ** 2)
    distribution /= distribution.sum(dim=0, keepdim=True).clamp_min(1e-8)
    cross_entropy = -(distribution * F.log_softmax(scores, dim=0)).sum(dim=0).mean()
    return ranking + 0.20 * cross_entropy, {
        "ranking": float(ranking.detach()),
        "cross_entropy": float(cross_entropy.detach()),
    }


def _offsets_v203(logits, recipe):
    temperature = float(recipe["temperature"])
    topk = int(recipe["topk"])
    scaled = logits.astype(np.float64) / temperature
    if topk == 1:
        return STATE_OFFSETS[np.argmax(scaled, axis=0)].astype(np.float64)
    indices = np.argpartition(scaled, -topk, axis=0)[-topk:]
    columns = np.arange(scaled.shape[1])
    chosen = scaled[indices, columns]
    chosen -= np.max(chosen, axis=0, keepdims=True)
    weight = np.exp(chosen)
    weight /= np.sum(weight, axis=0, keepdims=True)
    return np.sum(STATE_OFFSETS[indices] * weight, axis=0)


def _evaluate_recipe_v203(items, recipe, endpoint):
    parts = []
    for item in items:
        record = item["record"]
        offsets = _offsets_v203(item["logits"], recipe)
        prediction = _row_prediction(record, item["rows"], offsets)
        eval_rows = record.eval_index
        parts.append({
            "well": record.well,
            "row_idx": eval_rows.astype(np.int64),
            "truth": record.truth[eval_rows].astype(np.float64),
            "prediction": prediction,
            "distance": (
                record.md[eval_rows].astype(np.float64)
                - float(record.md[record.anchor_index])
            ),
        })
    return _metrics(parts, endpoint), parts


base_state_v203 = copy.deepcopy(model.state_dict())
optimizer_v203 = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE_V203, weight_decay=1e-5
)
history_v203 = []
best_v203 = None

for epoch_v203 in range(1, MAX_EPOCHS_V203 + 1):
    model.train()
    rng_v203 = np.random.default_rng(int(SEED) + 20300 + epoch_v203)
    sums_v203 = {"loss": 0.0, "ranking": 0.0, "cross_entropy": 0.0}
    for number_v203, index_v203 in enumerate(
        rng_v203.permutation(len(train_records)), 1
    ):
        record_v203 = train_records[int(index_v203)]
        rows_v203, label_v203, _, _ = _block_nodes(record_v203)
        optimizer_v203.zero_grad(set_to_none=True)
        logits_v203 = model(*_inputs(record_v203, rows_v203, device))
        loss_v203, diagnostics_v203 = _ranking_loss_v203(logits_v203, label_v203)
        loss_v203.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer_v203.step()
        sums_v203["loss"] += float(loss_v203.detach())
        for key_v203, value_v203 in diagnostics_v203.items():
            sums_v203[key_v203] += value_v203
        if number_v203 % 100 == 0:
            print(
                f"V203 epoch{epoch_v203} train {number_v203}/{len(train_records)}",
                flush=True,
            )

    select_items_v203 = _logits_for_records(model, select_records, device, mode="full")
    epoch_recipe_v203 = {"topk": 8, "temperature": 0.75}
    epoch_metrics_v203, _ = _evaluate_recipe_v203(
        select_items_v203, epoch_recipe_v203, "fold3_epoch_monitor"
    )
    epoch_oracle_v203 = _metrics(
        _top8_oracle(select_items_v203), "fold3_top8_oracle"
    )
    row_v203 = {
        "epoch": epoch_v203,
        **{
            f"train_{key}": value / len(train_records)
            for key, value in sums_v203.items()
        },
        "fold3_rmse": epoch_metrics_v203["rmse"],
        "fold3_top8_oracle_rmse": epoch_oracle_v203["rmse"],
    }
    history_v203.append(row_v203)
    print("V203 EPOCH", row_v203, flush=True)
    selection_value_v203 = (
        epoch_metrics_v203["rmse"] + 0.25 * epoch_oracle_v203["rmse"]
    )
    if best_v203 is None or selection_value_v203 < best_v203["selection_value"]:
        best_v203 = {
            "epoch": epoch_v203,
            "selection_value": selection_value_v203,
            "state": copy.deepcopy(model.state_dict()),
            "fold3_monitor": epoch_metrics_v203,
            "fold3_top8_oracle": epoch_oracle_v203,
        }
        torch.save(best_v203, MODEL_DIR_V203 / "fold45_calibrated_best.pt")

pd.DataFrame(history_v203).to_csv(OUT_V203 / "training_history.csv", index=False)
model.load_state_dict(best_v203["state"])
select_items_v203 = _logits_for_records(model, select_records, device, mode="full")

recipe_rows_v203 = []
for topk_v203 in TOPK_V203:
    for temperature_v203 in TEMPERATURES_V203:
        recipe_v203 = {"topk": topk_v203, "temperature": temperature_v203}
        metrics_v203, _ = _evaluate_recipe_v203(
            select_items_v203, recipe_v203, "fold3_recipe"
        )
        recipe_rows_v203.append({**recipe_v203, **metrics_v203})
recipe_scan_v203 = pd.DataFrame(recipe_rows_v203).sort_values("rmse", ignore_index=True)
recipe_scan_v203.to_csv(OUT_V203 / "fold3_recipe_scan.csv", index=False)
selected_recipe_v203 = {
    "topk": int(recipe_scan_v203.iloc[0]["topk"]),
    "temperature": float(recipe_scan_v203.iloc[0]["temperature"]),
}

# Fold2 is a reused confirmation set: V202 already exposed its aggregate gate
# metrics, but V203 never consumes Fold2 labels during fitting or selection.
gate_full_v203 = _logits_for_records(model, gate_records, device, mode="full")
gate_no_gr_v203 = _logits_for_records(model, gate_records, device, mode="no_gr")
gate_shuffle_v203 = _logits_for_records(
    model, gate_records, device, mode="shuffled_typewell"
)
fold2_full_v203, fold2_parts_v203 = _evaluate_recipe_v203(
    gate_full_v203, selected_recipe_v203, "calibrated_full"
)
fold2_no_gr_v203, _ = _evaluate_recipe_v203(
    gate_no_gr_v203, selected_recipe_v203, "calibrated_no_gr"
)
fold2_shuffle_v203, _ = _evaluate_recipe_v203(
    gate_shuffle_v203, selected_recipe_v203, "calibrated_shuffled_typewell"
)
fold2_oracle_v203 = _metrics(
    _top8_oracle(gate_full_v203), "calibrated_top8_oracle"
)

checks_v203 = {
    "fold2_rmse_le_9p5": fold2_full_v203["rmse"] <= 9.5,
    "fold2_top8_oracle_rmse_le_7": fold2_oracle_v203["rmse"] <= 7.0,
    "no_gr_degradation_ge_0p30": (
        fold2_no_gr_v203["rmse"] - fold2_full_v203["rmse"] >= 0.30
    ),
    "shuffled_typewell_degradation_ge_0p30": (
        fold2_shuffle_v203["rmse"] - fold2_full_v203["rmse"] >= 0.30
    ),
}
strict_pass_v203 = bool(all(checks_v203.values()))
verdict_v203 = {
    "experiment": EXPERIMENT_V203,
    "stage": "Fold45_hard_negative_calibration_Fold3_selection_Fold2_reconfirmation",
    "base": "frozen V202 high-recall posterior",
    "changed_variable": "soft state regression replaced by hard-negative candidate ranking",
    "protocol": {
        "train_folds": [4, 5],
        "selection_fold": 3,
        "reused_confirmation_fold": 2,
        "untouched_fold": 1,
    },
    "best_epoch": best_v203["epoch"],
    "selected_recipe": selected_recipe_v203,
    "fold3_monitor": best_v203["fold3_monitor"],
    "fold3_top8_oracle": best_v203["fold3_top8_oracle"],
    "fold2": {
        "full": fold2_full_v203,
        "no_gr": fold2_no_gr_v203,
        "shuffled_typewell": fold2_shuffle_v203,
        "top8_oracle": fold2_oracle_v203,
    },
    "checks": checks_v203,
    "strict_pass": strict_pass_v203,
    "elapsed_minutes": (time.time() - STARTED_V203) / 60.0,
    "next": (
        "PASS -> crossfit calibrated posterior and test blend against V168"
        if strict_pass_v203
        else "FAIL -> hard-negative ranking cannot consume V202 support; replace correlation features"
    ),
}
(OUT_V203 / "verdict.json").write_text(
    json.dumps(_clean_v203(verdict_v203), indent=2, allow_nan=False),
    encoding="utf-8",
)

print("\nV203 FOLD3 RECIPE TOP")
print(
    recipe_scan_v203[
        ["topk", "temperature", "rmse", "median_well_rmse", "p95_well_rmse", "far_gt2000_rmse"]
    ].head(12).to_string(index=False),
    flush=True,
)
print("\nV203 FINAL VERDICT")
print(json.dumps(_clean_v203(verdict_v203), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_V203, flush=True)
