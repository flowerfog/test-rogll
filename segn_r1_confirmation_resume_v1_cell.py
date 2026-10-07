"""Resume SEGN-R1 confirmation from its saved Fold-3 selection checkpoint.

This cell performs no training and never loads Fold 1 or Fold 2.  It exists to
recover from an optional pandas/pyarrow row-artifact serialization failure.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch


SOURCE_CANDIDATES_RESUME = (
    Path("/root/autodl-tmp/segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py"),
    Path.cwd() / "segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py",
)
SOURCE_PATH_RESUME = next(
    (path for path in SOURCE_CANDIDATES_RESUME if path.exists()), None
)
if SOURCE_PATH_RESUME is None:
    raise FileNotFoundError(
        "Upload segn_r1_fiveslot_costmap_fold3_gateway_v0_cell.py to "
        "/root/autodl-tmp before running this resume cell."
    )

source_resume = SOURCE_PATH_RESUME.read_text(encoding="utf-8")
marker_resume = "\nstarted_r1 = time.time()"
if marker_resume not in source_resume:
    raise RuntimeError(f"Cannot find gateway marker in {SOURCE_PATH_RESUME}")
exec(
    compile(source_resume.split(marker_resume, 1)[0], str(SOURCE_PATH_RESUME), "exec"),
    globals(),
)

started_resume = time.time()
_seed_everything(SEED)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SEGN-R1 confirmation")
device_resume = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path_resume = MODEL_DIR_R1 / "best_fold3_selection.pt"
if not checkpoint_path_resume.exists():
    raise FileNotFoundError(checkpoint_path_resume)
checkpoint_resume = torch.load(
    checkpoint_path_resume, map_location=device_resume, weights_only=False
)
model_resume = FiveSlotCostMapR1().to(device_resume)
model_resume.load_state_dict(checkpoint_resume["model"])
model_resume.eval()

folds_resume, fold_path_resume = _load_folds()
fold3_wells_resume = folds_resume.loc[
    folds_resume["fold"].eq(SELECTION_FOLD_R1), "well"
].tolist()
if len(fold3_wells_resume) != 154:
    raise RuntimeError(f"Unexpected Fold-3 size: {len(fold3_wells_resume)}")
fold3_records_resume = _load_records(fold3_wells_resume, "r1_resume_fold3")

order_resume = np.argsort([record.eval_rows for record in fold3_records_resume])
selection_positions_resume = np.unique(
    np.rint(np.linspace(0, len(order_resume) - 1, SELECTION_WELLS_R1)).astype(int)
)
selection_indices_resume = set(order_resume[selection_positions_resume].tolist())
confirmation_records_resume = [
    record
    for index, record in enumerate(fold3_records_resume)
    if index not in selection_indices_resume
]
if len(confirmation_records_resume) != 122:
    raise RuntimeError(
        f"Unexpected confirmation size: {len(confirmation_records_resume)}"
    )

confirmation_full_r1, confirmation_diagnostics_r1 = _evaluate_r1(
    model_resume,
    confirmation_records_resume,
    device_resume,
    control="full",
    save_prefix=None,
)
confirmation_no_gr_r1, no_gr_diagnostics_r1 = _evaluate_r1(
    model_resume,
    confirmation_records_resume,
    device_resume,
    control="no_gr",
    save_prefix=None,
)
confirmation_shuffled_r1, shuffled_diagnostics_r1 = _evaluate_r1(
    model_resume,
    confirmation_records_resume,
    device_resume,
    control="shuffled_typewell",
    save_prefix=None,
)

for label_resume, metrics_resume in (
    ("full", confirmation_full_r1),
    ("no_gr", confirmation_no_gr_r1),
    ("shuffled_typewell", confirmation_shuffled_r1),
):
    pd.DataFrame(metrics_resume).T.to_csv(
        OUT_DIR / f"fold3_confirmation_{label_resume}_metrics.csv",
        index_label="variant",
    )

slot_report_r1 = []
for slot_resume in ("fused",) + SLOT_NAMES_R1:
    mean_resume = confirmation_full_r1[f"{slot_resume}_mean"]
    mode_resume = confirmation_full_r1[f"{slot_resume}_mode"]
    slot_report_r1.append(
        {
            "slot": slot_resume,
            "mean_rmse": mean_resume["rmse"],
            "mode_rmse": mode_resume["rmse"],
            "mean_gain_vs_mode": mode_resume["rmse"] - mean_resume["rmse"],
            "local_top1_within5": mean_resume.get("local_top1_within5"),
            "median_truth_rank": mean_resume.get("median_truth_rank"),
            "mean_entropy": mean_resume.get("mean_entropy"),
            "no_gr_mean_rmse": confirmation_no_gr_r1[f"{slot_resume}_mean"]["rmse"],
            "no_gr_degradation": (
                confirmation_no_gr_r1[f"{slot_resume}_mean"]["rmse"]
                - mean_resume["rmse"]
            ),
            "shuffled_mean_rmse": confirmation_shuffled_r1[
                f"{slot_resume}_mean"
            ]["rmse"],
            "shuffled_degradation": (
                confirmation_shuffled_r1[f"{slot_resume}_mean"]["rmse"]
                - mean_resume["rmse"]
            ),
        }
    )

slot_report_frame_r1 = pd.DataFrame(slot_report_r1).sort_values("mean_rmse")
slot_report_frame_r1.to_csv(OUT_DIR / "slot_confirmation_report.csv", index=False)
print("SEGN-R1 SLOT REPORT")
print(slot_report_frame_r1.to_string(index=False), flush=True)

strong_slots_resume = [
    slot_report_r1[index + 1] for index in STRONG_SLOT_INDICES_R1
]
best_strong_slot_resume = min(
    strong_slots_resume, key=lambda row: row["mean_rmse"]
)
checks_resume = {
    "target_axis_coverage_ge_0p99": (
        _target_axis_coverage(fold3_records_resume) >= 0.99
    ),
    "fused_teacher_anchor_rmse_le_6": (
        confirmation_full_r1["fused_mean"]["rmse"] <= 6.0
    ),
    "best_strong_slot_rmse_le_8": best_strong_slot_resume["mean_rmse"] <= 8.0,
    "best_strong_slot_no_gr_degradation_ge_0p5": (
        best_strong_slot_resume["no_gr_degradation"] >= 0.5
    ),
    "best_strong_slot_shuffled_degradation_ge_0p5": (
        best_strong_slot_resume["shuffled_degradation"] >= 0.5
    ),
    "posterior_mean_gain_vs_mode_ge_0p5": (
        confirmation_diagnostics_r1["mean_minus_mode_gain"] >= 0.5
    ),
    "gate_max_mean_le_0p8": (
        confirmation_diagnostics_r1["gate_max_mean"] <= GATE_MAX_MEAN_R1
    ),
    "effective_slots_ge_2": (
        confirmation_diagnostics_r1["effective_slots"]
        >= GATE_MIN_EFFECTIVE_SLOTS_R1
    ),
}
strict_pass_resume = bool(all(checks_resume.values()))

verdict_resume = {
    "experiment": EXPERIMENT_R1,
    "stage": "fold3_five_slot_teacher_anchor_costmap_gateway_resume",
    "sample_unit": "chunks nested within complete physical wells",
    "deployable": False,
    "truth_usage": "teacher anchor for observability only",
    "legal_model_inputs_only": True,
    "uses_pf_sp45_strong_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS_R1),
        "selection_fold": SELECTION_FOLD_R1,
        "selection_wells": SELECTION_WELLS_R1,
        "confirmation_wells": len(confirmation_records_resume),
        "fold2_touched": False,
        "fold1_touched": False,
        "selected_epoch": int(checkpoint_resume["epoch"]),
        "selected_rmse": float(checkpoint_resume["selection_rmse"]),
    },
    "confirmation_fused_mean": confirmation_full_r1["fused_mean"],
    "confirmation_fused_mode": confirmation_full_r1["fused_mode"],
    "confirmation_diagnostics": confirmation_diagnostics_r1,
    "best_strong_matching_slot": best_strong_slot_resume,
    "slot_report": slot_report_r1,
    "controls": {
        "no_gr_fused_mean": confirmation_no_gr_r1["fused_mean"],
        "shuffled_typewell_fused_mean": confirmation_shuffled_r1["fused_mean"],
        "no_gr_diagnostics": no_gr_diagnostics_r1,
        "shuffled_diagnostics": shuffled_diagnostics_r1,
    },
    "checks": checks_resume,
    "strict_pass": strict_pass_resume,
    "checkpoint": str(checkpoint_path_resume),
    "elapsed_minutes": (time.time() - started_resume) / 60.0,
    "next": (
        "PASS -> freeze the five-slot emission and run anchor-jitter before recurrence"
        if strict_pass_resume
        else "FAIL -> inspect slot table; do not add recurrence to an unproven emission"
    ),
}
_write_json(OUT_DIR / "verdict_resume.json", verdict_resume)
print("SEGN-R1 FINAL VERDICT", json.dumps(verdict_resume, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)

if not strict_pass_resume:
    raise RuntimeError(json.dumps(verdict_resume, indent=2))

