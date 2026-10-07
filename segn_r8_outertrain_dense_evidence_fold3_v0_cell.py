"""SEGN-R8: full outer-train scaling audit for the R6 dense evidence trunk.

R5/R6 trained on only Folds 4/5 (309 wells), while the referenced SEGN work
used leave-well-out style training.  R6 also selected its last available epoch.
This experiment changes only statistical scale: train the unchanged R6 dense
bilinear/cosine evidence model on every well outside Fold 3, select the epoch
on 32 deterministic Fold-3 wells, and confirm once on the remaining 122.

No base path, PF, spatial feature, CP12, Gold, neighbor, prior, or recurrence is
used.  This is the final evidence-family attribution test before replacing the
matching encoder.
"""

from __future__ import annotations

from pathlib import Path
import copy
import json
import math
import time

import numpy as np
import pandas as pd
import torch


ROOT_R8 = Path("/root/autodl-tmp")
SOURCE_CANDIDATES_R8 = (
    ROOT_R8 / "segn_r6_densemd_evidence_fold3_v0_cell.py",
    Path.cwd() / "segn_r6_densemd_evidence_fold3_v0_cell.py",
)
SOURCE_PATH_R8 = next((path for path in SOURCE_CANDIDATES_R8 if path.exists()), None)
if SOURCE_PATH_R8 is None:
    raise FileNotFoundError(
        "Upload segn_r6_densemd_evidence_fold3_v0_cell.py to /root/autodl-tmp."
    )

source_r8 = SOURCE_PATH_R8.read_text(encoding="utf-8")
marker_r8 = "\nstarted_r6 = time.time()"
if marker_r8 not in source_r8:
    raise RuntimeError(f"Cannot find R6 execution marker in {SOURCE_PATH_R8}")
exec(compile(source_r8.split(marker_r8, 1)[0], str(SOURCE_PATH_R8), "exec"), globals())


EXPERIMENT_R8 = "segn_r8_outertrain_dense_evidence_fold3_v0"
OUT_DIR_R8 = OOF_ROOT / EXPERIMENT_R8
MODEL_DIR_R8 = OUT_DIR_R8 / "models"
OUT_DIR_R8.mkdir(parents=True, exist_ok=True)
MODEL_DIR_R8.mkdir(parents=True, exist_ok=True)

# R6 helpers resolve these names dynamically.
OUT_DIR_R6 = OUT_DIR_R8
MODEL_DIR_R6 = MODEL_DIR_R8
OUT_DIR = OUT_DIR_R8
TRAIN_FOLDS_R6 = (1, 2, 4, 5)
VALID_FOLD_R6 = 3
MAX_EPOCHS_R6 = 32
MIN_EPOCHS_R6 = 16
REPORT_EVERY_R6 = 4
PATIENCE_REPORTS_R6 = 4


def _json_default_r8(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json_r8(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default_r8), encoding="utf-8"
    )


def _load_records_lenient_r8(wells, label, allow_train_exclusions):
    records = []
    failures = []
    started = time.time()
    for number, well in enumerate(wells, 1):
        try:
            records.append(_load_record(well))
        except Exception as error:
            failures.append({"well": str(well), "reason": repr(error)})
        if number % 50 == 0 or number == len(wells):
            print(
                f"load {label} records {number}/{len(wells)} failures={len(failures)}",
                flush=True,
            )
    if failures:
        pd.DataFrame(failures).to_csv(
            OUT_DIR_R8 / f"{label}_record_failures.csv", index=False
        )
        if not allow_train_exclusions or len(failures) > 2:
            raise RuntimeError(f"{len(failures)} {label} failures: {failures[:5]}")
    print(
        f"loaded {label}: {len(records)} wells in "
        f"{(time.time() - started) / 60.0:.2f}m",
        flush=True,
    )
    return records, failures


def _fit_r8(train_records, selection_records, device):
    _seed_everything(SEED + 8001)
    model = AxisAwareEvidenceR5().to(device)
    zero_guard = _query_zero_guard_r5(model, device)
    if not zero_guard["finite"] or zero_guard["maximum_no_gr_logit_std"] > 1.0e-6:
        raise RuntimeError(f"R8 pre-training query-zero guard failed: {zero_guard}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_R6, weight_decay=WEIGHT_DECAY_R6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS_R6, eta_min=MIN_LEARNING_RATE_R6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = math.inf
    bad_reports = 0
    history = []
    started_fit = time.time()

    for epoch in range(1, MAX_EPOCHS_R6 + 1):
        training = _train_epoch_r6(
            model, train_records, optimizer, scaler, device, epoch
        )
        scheduler.step()
        report = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in training.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started_fit) / 60.0,
        }
        if epoch % REPORT_EVERY_R6 == 0 or epoch == MAX_EPOCHS_R6:
            selection, _ = _evaluate_r6(model, selection_records, device)
            report.update(
                {
                    "selection_local_expected_rmse": selection["local"][
                        "expected_rmse"
                    ],
                    "selection_local_expected_correlation": selection["local"][
                        "expected_correlation"
                    ],
                    "selection_median_truth_rank": selection["local"][
                        "median_truth_rank"
                    ],
                    "selection_p90_truth_rank": selection["local"]["p90_truth_rank"],
                    "selection_path_mean_rmse": selection["path_mean"]["rmse"],
                    "selection_shape_correlation": selection["path_mean"][
                        "within_well_shape_correlation"
                    ],
                    "selection_mass_within5": selection["local"][
                        "posterior_mass_within5"
                    ],
                }
            )
            score = float(selection["local"]["expected_rmse"])
            if score < best_score - 1.0e-4:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_reports = 0
            else:
                bad_reports += 1
        history.append(report)
        pd.DataFrame(history).to_csv(
            OUT_DIR_R8 / "training_history.csv", index=False
        )
        print("SEGN-R8 EPOCH", report, flush=True)
        if epoch >= MIN_EPOCHS_R6 and bad_reports >= PATIENCE_REPORTS_R6:
            print(f"SEGN-R8 early stopping at epoch {epoch}", flush=True)
            break

    if best_epoch == 0:
        raise RuntimeError("SEGN-R8 produced no selection checkpoint")
    model.load_state_dict(best_state)
    checkpoint = MODEL_DIR_R8 / "best_outertrain_dense_evidence.pt"
    torch.save(
        {
            "model": best_state,
            "epoch": best_epoch,
            "selection_local_expected_rmse": best_score,
            "state_offsets": STATE_OFFSETS,
            "dense_md_spacing_ft": DENSE_MD_SPACING_FT_R6,
            "train_folds": TRAIN_FOLDS_R6,
            "source": str(SOURCE_PATH_R8),
        },
        checkpoint,
    )
    return model, best_epoch, best_score, zero_guard, checkpoint, history


started_r8 = time.time()
_seed_everything(SEED + 8001)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SEGN-R8")
device_r8 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds_r8, fold_path_r8 = _load_folds()
train_wells_r8 = folds_r8.loc[
    folds_r8["fold"].isin(TRAIN_FOLDS_R6), "well"
].tolist()
fold3_wells_r8 = folds_r8.loc[
    folds_r8["fold"].eq(VALID_FOLD_R6), "well"
].tolist()
if len(train_wells_r8) != 619 or len(fold3_wells_r8) != 154:
    raise RuntimeError(
        f"Unexpected outer-train/Fold3 sizes: {len(train_wells_r8)}/{len(fold3_wells_r8)}"
    )

train_records_r8, train_failures_r8 = _load_records_lenient_r8(
    train_wells_r8, "r8_outertrain", allow_train_exclusions=True
)
fold3_records_r8, fold3_failures_r8 = _load_records_lenient_r8(
    fold3_wells_r8, "r8_fold3", allow_train_exclusions=False
)
if len(train_records_r8) < 617 or len(fold3_records_r8) != 154:
    raise RuntimeError(
        f"Insufficient eligible records: {len(train_records_r8)}/{len(fold3_records_r8)}"
    )

order_r8 = np.argsort([record.eval_rows for record in fold3_records_r8])
selection_positions_r8 = np.unique(
    np.rint(np.linspace(0, len(order_r8) - 1, SELECTION_WELLS_R6)).astype(int)
)
selection_indices_r8 = set(order_r8[selection_positions_r8].tolist())
selection_records_r8 = [
    record for index, record in enumerate(fold3_records_r8) if index in selection_indices_r8
]
confirmation_records_r8 = [
    record for index, record in enumerate(fold3_records_r8) if index not in selection_indices_r8
]

coverage_r8 = _dense_axis_coverage_r6(fold3_records_r8)
guards_r8 = {
    "source": str(SOURCE_PATH_R8),
    "fold_source": str(fold_path_r8),
    "train_folds": list(TRAIN_FOLDS_R6),
    "requested_train_wells": len(train_wells_r8),
    "eligible_train_wells": len(train_records_r8),
    "train_record_failures": train_failures_r8,
    "selection_fold": VALID_FOLD_R6,
    "selection_wells": len(selection_records_r8),
    "confirmation_wells": len(confirmation_records_r8),
    "dense_md_spacing_ft": DENSE_MD_SPACING_FT_R6,
    "target_axis": coverage_r8,
    "parameters": int(sum(p.numel() for p in AxisAwareEvidenceR5().parameters())),
    "epochs_budget": MAX_EPOCHS_R6,
}
print("SEGN-R8 GUARDS", json.dumps(guards_r8, indent=2), flush=True)
if coverage_r8["coverage"] < 0.99:
    raise RuntimeError(f"SEGN-R8 target-axis guard failed: {coverage_r8}")

model_r8, best_epoch_r8, best_score_r8, zero_guard_r8, checkpoint_r8, history_r8 = _fit_r8(
    train_records_r8, selection_records_r8, device_r8
)
confirmation_full_r8, confirmation_bywell_r8 = _evaluate_r6(
    model_r8, confirmation_records_r8, device_r8, control="full"
)
confirmation_no_gr_r8, _ = _evaluate_r6(
    model_r8, confirmation_records_r8, device_r8, control="no_gr"
)
confirmation_shuffled_r8, _ = _evaluate_r6(
    model_r8, confirmation_records_r8, device_r8, control="shuffled_typewell"
)
confirmation_bywell_r8.to_csv(
    OUT_DIR_R8 / "fold3_confirmation_bywell.csv", index=False
)
post_zero_guard_r8 = _query_zero_guard_r5(model_r8, device_r8)

path_gain_vs_uniform_r8 = (
    confirmation_no_gr_r8["path_mean"]["rmse"]
    - confirmation_full_r8["path_mean"]["rmse"]
)
local_gain_vs_uniform_r8 = (
    confirmation_no_gr_r8["local"]["expected_rmse"]
    - confirmation_full_r8["local"]["expected_rmse"]
)
shuffle_path_degradation_r8 = (
    confirmation_shuffled_r8["path_mean"]["rmse"]
    - confirmation_full_r8["path_mean"]["rmse"]
)
shuffle_local_degradation_r8 = (
    confirmation_shuffled_r8["local"]["expected_rmse"]
    - confirmation_full_r8["local"]["expected_rmse"]
)

checks_r8 = {
    "query_zero_logit_std_le_1e_6": post_zero_guard_r8[
        "maximum_no_gr_logit_std"
    ]
    <= 1.0e-6,
    "path_mean_rmse_le_8p5": confirmation_full_r8["path_mean"]["rmse"] <= 8.5,
    "local_expected_rmse_le_8p5": confirmation_full_r8["local"][
        "expected_rmse"
    ]
    <= 8.5,
    "local_expected_correlation_ge_0p20": confirmation_full_r8["local"][
        "expected_correlation"
    ]
    >= 0.20,
    "median_truth_rank_le_10": confirmation_full_r8["local"][
        "median_truth_rank"
    ]
    <= 10.0,
    "path_gain_vs_uniform_ge_0p30": path_gain_vs_uniform_r8 >= 0.30,
    "local_gain_vs_uniform_ge_0p30": local_gain_vs_uniform_r8 >= 0.30,
    "shuffled_path_degradation_ge_0p30": shuffle_path_degradation_r8 >= 0.30,
    "shuffled_local_degradation_ge_0p30": shuffle_local_degradation_r8 >= 0.30,
}
strict_pass_r8 = bool(all(checks_r8.values()))

verdict_r8 = {
    "experiment": EXPERIMENT_R8,
    "stage": "fold3_full_outertrain_dense_evidence_scaling_gateway",
    "sample_unit": "dense teacher-anchored chunks nested within complete wells",
    "deployable": False,
    "truth_usage": "teacher anchors for evidence observability only",
    "uses_pf_sp45_strong_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS_R6),
        "selection_fold": VALID_FOLD_R6,
        "selection_wells": len(selection_records_r8),
        "confirmation_wells": len(confirmation_records_r8),
        "selected_epoch": best_epoch_r8,
        "selected_local_expected_rmse": best_score_r8,
    },
    "attribution": {
        "architecture_change_vs_r6": False,
        "r6_train_wells": 309,
        "r8_train_wells": len(train_records_r8),
        "purpose": "test whether R6 failed because its outer-train sample was halved",
    },
    "guards": guards_r8,
    "query_zero_guard_before_training": zero_guard_r8,
    "query_zero_guard_after_training": post_zero_guard_r8,
    "confirmation": {
        "full": confirmation_full_r8,
        "uniform_teacher_anchor": confirmation_no_gr_r8,
        "shuffled_typewell": confirmation_shuffled_r8,
    },
    "diagnostics": {
        "path_gain_vs_uniform": path_gain_vs_uniform_r8,
        "local_gain_vs_uniform": local_gain_vs_uniform_r8,
        "shuffled_path_degradation": shuffle_path_degradation_r8,
        "shuffled_local_degradation": shuffle_local_degradation_r8,
    },
    "checks": checks_r8,
    "strict_pass": strict_pass_r8,
    "checkpoint": str(checkpoint_r8),
    "elapsed_minutes": (time.time() - started_r8) / 60.0,
    "next": (
        "PASS -> rerun the R7 base-prior cost-map audit with this outer-train checkpoint"
        if strict_pass_r8
        else "FAIL -> training scale is not the missing mechanism; close bilinear/cosine and replace the candidate evidence construction"
    ),
}
_write_json_r8(OUT_DIR_R8 / "verdict.json", verdict_r8)
print("SEGN-R8 FINAL VERDICT", json.dumps(verdict_r8, indent=2), flush=True)
print("saved:", OUT_DIR_R8, flush=True)

if not strict_pass_r8:
    raise RuntimeError(json.dumps(verdict_r8, indent=2))
