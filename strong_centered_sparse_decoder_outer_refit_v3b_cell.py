"""Protocol repair: fixed outer-train refit for sparse decoder v3.

V3 selected epoch=3 and posterior temperature/dose on Fold 1 but applied the
folds-3/4/5 checkpoint directly to Fold 2.  This cell performs the standard
nested refit: reinitialise, train for the already frozen epoch count on folds
1/3/4/5, then apply the already frozen decoder to Fold 2 exactly once.
"""

from __future__ import annotations

import gc
import json
import runpy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REQUIRED_V3B = {
    "OUT_DIR_V3",
    "MODEL_DIR_V3",
    "TRAIN_FOLDS_V3",
    "VALID_FOLD_V3",
    "TEST_FOLD_V3",
    "MAX_EPOCHS_V3",
    "GRAD_ACCUM_WELLS_V3",
    "LEARNING_RATE_V3",
    "WEIGHT_DECAY_V3",
    "SEED_V3",
    "SparseCompleteWellDecoderV3",
    "_build_records_v3",
    "_loss_v3",
    "_collect_logits_v3",
    "_evaluate_logits_v3",
    "_seed_v3",
    "_jsonable_v3",
    "best_v3",
    "selected_v3",
    "feature_count_v3",
    "context_count_v3",
    "device_v3",
    "valid_records_v3",
    "test_records_v3",
    "shuffled_records_v3",
    "no_alignment_records_v3",
    "test_nonzero_v3",
}


def _restore_v3b_if_needed():
    missing = sorted(REQUIRED_V3B.difference(globals()))
    if not missing:
        print("V3B bootstrap: reusing live v3 state", flush=True)
        return
    candidates = (
        Path("/root/autodl-tmp/strong_centered_sparse_decoder_v3_cell.py"),
        Path.cwd() / "strong_centered_sparse_decoder_v3_cell.py",
    )
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise RuntimeError(
            "Upload strong_centered_sparse_decoder_v3_cell.py or run v3 first. "
            "Missing: "
            + repr(missing)
        )
    print("V3B bootstrap: running v3 prerequisite", source, flush=True)
    namespace = runpy.run_path(str(source))
    globals().update(
        {key: value for key, value in namespace.items() if not key.startswith("__")}
    )


_restore_v3b_if_needed()

EXPERIMENT_V3B = "strong_centered_sparse_decoder_outer_refit_v3b"
OUT_DIR_V3B = Path(OUT_DIR_V3).parent / EXPERIMENT_V3B
MODEL_DIR_V3B = OUT_DIR_V3B / "models"
OUT_DIR_V3B.mkdir(parents=True, exist_ok=True)
MODEL_DIR_V3B.mkdir(parents=True, exist_ok=True)
OUTER_FOLDS_V3B = (1, 3, 4, 5)
FROZEN_EPOCHS_V3B = int(best_v3["epoch"])
FROZEN_TEMPERATURE_V3B = float(selected_v3["temperature"])
FROZEN_DOSE_V3B = float(selected_v3["dose"])

if FROZEN_DOSE_V3B <= 0:
    raise RuntimeError("Fold-1 selected dose=0; no nonzero outer refit is authorised")

started_v3b = time.time()
print(
    "V3B FROZEN CONFIG",
    {
        "outer_train_folds": OUTER_FOLDS_V3B,
        "test_fold": TEST_FOLD_V3,
        "epochs": FROZEN_EPOCHS_V3B,
        "temperature": FROZEN_TEMPERATURE_V3B,
        "dose": FROZEN_DOSE_V3B,
        "seed": SEED_V3,
    },
    flush=True,
)

# Fold 1 records already exist from the selection phase.  Rebuild only the
# former core set, then append Fold 1 without regenerating it.
core_records_v3b, core_failures_v3b = _build_records_v3(TRAIN_FOLDS_V3)
outer_records_v3b = list(core_records_v3b) + list(valid_records_v3)
outer_records_v3b.sort(key=lambda item: item["well"])
if len(outer_records_v3b) < 610:
    raise RuntimeError(f"only {len(outer_records_v3b)} outer-train wells")

_seed_v3(SEED_V3)
model_v3b = SparseCompleteWellDecoderV3(
    feature_count_v3, context_count_v3
).to(device_v3)
optimizer_v3b = torch.optim.AdamW(
    model_v3b.parameters(), lr=LEARNING_RATE_V3, weight_decay=WEIGHT_DECAY_V3
)
scheduler_v3b = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_v3b, T_max=MAX_EPOCHS_V3, eta_min=3e-5
)
scaler_v3b = torch.amp.GradScaler("cuda", enabled=device_v3.type == "cuda")

history_v3b = []
for epoch_v3b in range(1, FROZEN_EPOCHS_V3B + 1):
    model_v3b.train()
    order_v3b = np.random.permutation(len(outer_records_v3b))
    optimizer_v3b.zero_grad(set_to_none=True)
    totals_v3b = {"loss": 0.0, "ce": 0.0, "path_mse": 0.0, "delta_mse": 0.0}
    updates_v3b = 0
    for step_v3b, record_index_v3b in enumerate(order_v3b, 1):
        record_v3b = outer_records_v3b[int(record_index_v3b)]
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device_v3.type == "cuda"
        ):
            loss_v3b, parts_v3b = _loss_v3(model_v3b, record_v3b, device_v3)
            scaled_v3b = loss_v3b / GRAD_ACCUM_WELLS_V3
        scaler_v3b.scale(scaled_v3b).backward()
        if step_v3b % GRAD_ACCUM_WELLS_V3 == 0 or step_v3b == len(order_v3b):
            scaler_v3b.unscale_(optimizer_v3b)
            torch.nn.utils.clip_grad_norm_(model_v3b.parameters(), 5.0)
            scaler_v3b.step(optimizer_v3b)
            scaler_v3b.update()
            optimizer_v3b.zero_grad(set_to_none=True)
            updates_v3b += 1
        totals_v3b["loss"] += float(loss_v3b.detach().cpu())
        for key_v3b, value_v3b in parts_v3b.items():
            totals_v3b[key_v3b] += value_v3b
        if step_v3b % 100 == 0 or step_v3b == len(order_v3b):
            print(
                f"V3B epoch {epoch_v3b}/{FROZEN_EPOCHS_V3B} "
                f"wells {step_v3b}/{len(order_v3b)} "
                f"loss={totals_v3b['loss']/step_v3b:.4f}",
                flush=True,
            )
    scheduler_v3b.step()
    history_v3b.append(
        {
            "epoch": epoch_v3b,
            "train_loss": totals_v3b["loss"] / len(order_v3b),
            "train_ce": totals_v3b["ce"] / len(order_v3b),
            "train_path_rmse": float(
                np.sqrt(max(totals_v3b["path_mse"] / len(order_v3b), 0.0))
            ),
            "train_delta_rmse": float(
                np.sqrt(max(totals_v3b["delta_mse"] / len(order_v3b), 0.0))
            ),
            "updates": updates_v3b,
            "learning_rate": optimizer_v3b.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started_v3b) / 60.0,
        }
    )

pd.DataFrame(history_v3b).to_csv(OUT_DIR_V3B / "training_history.csv", index=False)
checkpoint_v3b = MODEL_DIR_V3B / "outer_refit_fold2.pt"
torch.save(
    {
        "model": {key: value.detach().cpu() for key, value in model_v3b.state_dict().items()},
        "epochs": FROZEN_EPOCHS_V3B,
        "temperature": FROZEN_TEMPERATURE_V3B,
        "dose": FROZEN_DOSE_V3B,
        "seed": SEED_V3,
        "outer_folds": OUTER_FOLDS_V3B,
    },
    checkpoint_v3b,
)

test_logits_v3b = _collect_logits_v3(model_v3b, test_records_v3, device_v3)
test_metrics_v3b, test_wells_v3b = _evaluate_logits_v3(
    test_records_v3,
    test_logits_v3b,
    FROZEN_TEMPERATURE_V3B,
    FROZEN_DOSE_V3B,
)
test_wells_v3b.to_csv(OUT_DIR_V3B / "fold2_well_metrics.csv", index=False)

shuffled_logits_v3b = _collect_logits_v3(
    model_v3b, shuffled_records_v3, device_v3
)
shuffled_metrics_v3b, _ = _evaluate_logits_v3(
    shuffled_records_v3,
    shuffled_logits_v3b,
    FROZEN_TEMPERATURE_V3B,
    FROZEN_DOSE_V3B,
)
no_alignment_logits_v3b = _collect_logits_v3(
    model_v3b, no_alignment_records_v3, device_v3
)
no_alignment_metrics_v3b, _ = _evaluate_logits_v3(
    no_alignment_records_v3,
    no_alignment_logits_v3b,
    FROZEN_TEMPERATURE_V3B,
    FROZEN_DOSE_V3B,
)

print("\nV3B FOLD2 OUTER REFIT")
print(json.dumps(_jsonable_v3(test_metrics_v3b), indent=2))
print("\nV3B CONTROLS")
print(
    json.dumps(
        _jsonable_v3(
            {
                "shuffled_alignment": shuffled_metrics_v3b,
                "no_alignment": no_alignment_metrics_v3b,
            }
        ),
        indent=2,
    )
)

gates_v3b = {
    "gain_ge": 0.50,
    "far_gain_ge": 1.00,
    "positive_well_fraction_ge": 0.55,
    "shuffle_degradation_ge": 0.50,
    "no_alignment_degradation_ge": 0.30,
}
checks_v3b = {
    "gain": bool(test_metrics_v3b["gain_vs_base"] >= gates_v3b["gain_ge"]),
    "far_gain": bool(
        test_metrics_v3b.get("far_gt2000_gain", -np.inf)
        >= gates_v3b["far_gain_ge"]
    ),
    "positive_wells": bool(
        test_metrics_v3b["positive_well_fraction"]
        >= gates_v3b["positive_well_fraction_ge"]
    ),
    "shuffle": bool(
        shuffled_metrics_v3b["rmse"] - test_metrics_v3b["rmse"]
        >= gates_v3b["shuffle_degradation_ge"]
    ),
    "no_alignment": bool(
        no_alignment_metrics_v3b["rmse"] - test_metrics_v3b["rmse"]
        >= gates_v3b["no_alignment_degradation_ge"]
    ),
}
strict_pass_v3b = bool(all(checks_v3b.values()))
verdict_v3b = _jsonable_v3(
    {
        "experiment": EXPERIMENT_V3B,
        "stage": "fixed_outer_train_refit_protocol_repair",
        "outer_train_folds": list(OUTER_FOLDS_V3B),
        "frozen_test_fold": TEST_FOLD_V3,
        "frozen_epochs": FROZEN_EPOCHS_V3B,
        "frozen_temperature": FROZEN_TEMPERATURE_V3B,
        "frozen_dose": FROZEN_DOSE_V3B,
        "outer_train_wells": len(outer_records_v3b),
        "previous_fold2": test_nonzero_v3,
        "outer_refit_fold2": test_metrics_v3b,
        "gain_delta_vs_previous": (
            test_metrics_v3b["gain_vs_base"] - test_nonzero_v3["gain_vs_base"]
        ),
        "controls": {
            "shuffled_alignment": shuffled_metrics_v3b,
            "no_alignment": no_alignment_metrics_v3b,
            "shuffle_degradation": shuffled_metrics_v3b["rmse"] - test_metrics_v3b["rmse"],
            "no_alignment_degradation": no_alignment_metrics_v3b["rmse"] - test_metrics_v3b["rmse"],
        },
        "gates": gates_v3b,
        "checks": checks_v3b,
        "strict_pass": strict_pass_v3b,
        "checkpoint": str(checkpoint_v3b),
        "elapsed_minutes": (time.time() - started_v3b) / 60.0,
        "next": (
            "PASS -> confirm another untouched outer fold"
            if strict_pass_v3b
            else "FAIL -> close this sparse decoder; more wells did not solve cross-well ranking"
        ),
    }
)
(OUT_DIR_V3B / "verdict.json").write_text(
    json.dumps(verdict_v3b, indent=2), encoding="utf-8"
)
print("\nFINAL VERDICT")
print(json.dumps(verdict_v3b, indent=2))
print("saved:", OUT_DIR_V3B)

del core_records_v3b, outer_records_v3b
gc.collect()
