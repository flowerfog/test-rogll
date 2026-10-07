"""SEGN-R5D: frozen logit-dose audit for the axis-aware evidence model.

R5 learned non-random cross-GR ordering, but its posterior mean moved much
farther from the teacher anchor than a uniform posterior.  This zero-training
cell tests the work-note hypothesis that the evidence should be consumed as a
soft dose rather than at its raw learned scale.

The 32-well Fold-3 selection subset freezes one non-negative logit dose.  The
remaining 122 Fold-3 wells confirm it.  Dose zero is explicitly a diagnostic
teacher-anchor uniform baseline, not a deployable no-GR model.  Folds 1/2 are
not loaded and the checkpoint receives zero optimizer/search updates.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SOURCE_CANDIDATES_R5D = (
    Path("/root/autodl-tmp/segn_r5_axisaware_evidence_fold3_v0_cell.py"),
    Path.cwd() / "segn_r5_axisaware_evidence_fold3_v0_cell.py",
)
SOURCE_PATH_R5D = next(
    (path for path in SOURCE_CANDIDATES_R5D if path.exists()), None
)
if SOURCE_PATH_R5D is None:
    raise FileNotFoundError(
        "Upload segn_r5_axisaware_evidence_fold3_v0_cell.py to "
        "/root/autodl-tmp before running SEGN-R5D."
    )

source_r5d = SOURCE_PATH_R5D.read_text(encoding="utf-8")
marker_r5d = "\nstarted_r5 = time.time()"
if marker_r5d not in source_r5d:
    raise RuntimeError(f"Cannot find bootstrap marker in {SOURCE_PATH_R5D}")
exec(
    compile(source_r5d.split(marker_r5d, 1)[0], str(SOURCE_PATH_R5D), "exec"),
    globals(),
)


EXPERIMENT_R5D = "segn_r5_frozen_logit_dose_audit_v1"
OUT_DIR_R5D = OOF_ROOT / EXPERIMENT_R5D
OUT_DIR_R5D.mkdir(parents=True, exist_ok=True)

CHECKPOINT_CANDIDATES_R5D = (
    OOF_ROOT
    / "segn_r5_axisaware_evidence_fold3_v0"
    / "models"
    / "best_axisaware_evidence.pt",
    Path("/root/autodl-tmp/best_axisaware_evidence.pt"),
    Path.cwd() / "best_axisaware_evidence.pt",
)
CHECKPOINT_R5D = next(
    (path for path in CHECKPOINT_CANDIDATES_R5D if path.exists()), None
)
if CHECKPOINT_R5D is None:
    raise FileNotFoundError(
        "Cannot find the frozen R5 best_axisaware_evidence.pt checkpoint."
    )

DOSES_R5D = np.asarray(
    [
        0.0,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.0,
    ],
    dtype=np.float64,
)
SELECTION_WELLS_R5D = 32


def _checkpoint_state_r5d(path):
    # This checkpoint was produced locally by the trusted R5 training cell.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload
    if isinstance(payload, dict) and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        return payload, {"model": payload}
    raise RuntimeError(f"Unsupported checkpoint payload in {path}")


def _safe_correlation_r5d(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    if len(left) < 2 or np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


@torch.no_grad()
def _evaluate_doses_r5d(
    model,
    records,
    doses,
    device,
    control="full",
    save_selected_dose=None,
    save_prefix=None,
):
    model.eval()
    doses = np.asarray(doses, np.float64)
    dose_tensor = torch.as_tensor(
        doses, device=device, dtype=torch.float32
    )[:, None, None]
    dose_count = len(doses)
    local_targets = []
    local_expected = [[] for _ in doses]
    local_mass5 = [[] for _ in doses]
    local_nll = [[] for _ in doses]
    truths = []
    distances = []
    anchors = []
    wells = []
    predictions = [[] for _ in doses]
    selected_rows = []
    shuffled = (
        list(records[1:]) + list(records[:1])
        if len(records) > 1
        else list(records)
    )

    for number, (record, shuffled_record) in enumerate(zip(records, shuffled), 1):
        node_count = len(record.node_rows)
        sums = np.zeros((dose_count, node_count), np.float64)
        count = np.zeros(node_count, np.float64)
        sums[:, 0] = float(record.anchor_tvt)
        count[0] = 1.0

        for raw_start in record.chunk_starts:
            start = int(raw_start)
            start_row = int(record.node_rows[start])
            previous_delta, previous_rate = _teacher_history_r5(record, start)
            payload = _chunk_payload(
                record,
                start,
                deployed_anchor=float(record.truth[start_row]),
                previous_delta=previous_delta,
                previous_rate=previous_rate,
                control=control,
                shuffled_typewell=shuffled_record,
            )
            tensors = _tensor_payload(payload, device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, auxiliary = model(*tensors, return_aux=True)
            logits = logits.float()
            valid = auxiliary["candidate_valid"]
            scaled = dose_tensor * logits[None]
            scaled = scaled.masked_fill(~valid[None, :, None], -1.0e4)
            probability = F.softmax(scaled, dim=1)
            states = torch.as_tensor(
                STATE_OFFSETS, device=device, dtype=torch.float32
            )
            expected = (probability * states[None, :, None]).sum(dim=1)

            target = payload["target_state"].astype(np.float64)
            end = start + len(target)
            node_slice = slice(start + 1, end + 1)
            anchor = float(payload["deployed_anchor"])
            expected_np = expected.cpu().numpy().astype(np.float64)
            sums[:, node_slice] += anchor + expected_np
            count[node_slice] += 1.0
            local_targets.extend(target.tolist())

            probability_np = probability.cpu().numpy().astype(np.float64)
            for dose_index in range(dose_count):
                local_expected[dose_index].extend(expected_np[dose_index].tolist())
                for column, target_value in enumerate(target):
                    within5 = np.abs(STATE_OFFSETS - target_value) <= 5.0
                    local_mass5[dose_index].append(
                        float(probability_np[dose_index, within5, column].sum())
                    )
                    true_index = int(
                        np.argmin(np.abs(STATE_OFFSETS - target_value))
                    )
                    local_nll[dose_index].append(
                        float(
                            -np.log(
                                max(
                                    probability_np[
                                        dose_index, true_index, column
                                    ],
                                    1.0e-12,
                                )
                            )
                        )
                    )

        supported = count > 0
        support_md = record.md[record.node_rows[supported]].astype(np.float64)
        query_md = record.md[record.eval_index].astype(np.float64)
        truth = record.truth[record.eval_index].astype(np.float64)
        distance = query_md - float(record.md[record.anchor_index])
        wells.append(record.well)
        truths.append(truth)
        distances.append(distance)
        anchors.append(float(record.anchor_tvt))
        for dose_index in range(dose_count):
            nodes = sums[dose_index, supported] / count[supported]
            prediction = np.interp(query_md, support_md, nodes)
            predictions[dose_index].append(prediction)

        if number % 25 == 0 or number == len(records):
            print(
                f"SEGN-R5D evaluate {control} {number}/{len(records)}",
                flush=True,
            )

    local_target = np.asarray(local_targets, np.float64)
    reports = []
    for dose_index, dose in enumerate(doses):
        path = _metric_payload_r1(
            wells, truths, predictions[dose_index], distances
        )
        path.update(
            _shape_metrics_r4(
                wells, truths, predictions[dose_index], anchors
            )
        )
        expected = np.asarray(local_expected[dose_index], np.float64)
        report = {
            "control": control,
            "dose": float(dose),
            **path,
            "local_columns": int(len(local_target)),
            "local_expected_rmse": float(
                np.sqrt(np.mean((expected - local_target) ** 2))
            ),
            "local_expected_correlation": _safe_correlation_r5d(
                expected, local_target
            ),
            "posterior_mass_within5": float(
                np.mean(local_mass5[dose_index])
            ),
            "mean_true_state_nll": float(np.mean(local_nll[dose_index])),
        }
        reports.append(report)

    if save_selected_dose is not None and save_prefix is not None:
        selected_index = int(np.argmin(np.abs(doses - save_selected_dose)))
        np.savez_compressed(
            OUT_DIR_R5D / f"{save_prefix}_rows.npz",
            well=np.concatenate(
                [
                    np.repeat(well, len(values))
                    for well, values in zip(wells, truths)
                ]
            ).astype("U"),
            truth=np.concatenate(truths).astype(np.float32),
            prediction=np.concatenate(predictions[selected_index]).astype(
                np.float32
            ),
            distance_past_anchor=np.concatenate(distances).astype(np.float32),
        )
    return pd.DataFrame(reports)


started_r5d = time.time()
_seed_everything(SEED)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SEGN-R5D")
device_r5d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds_r5d, fold_path_r5d = _load_folds()
fold3_wells_r5d = folds_r5d.loc[folds_r5d["fold"].eq(3), "well"].tolist()
if len(fold3_wells_r5d) != 154:
    raise RuntimeError(f"Unexpected Fold3 size: {len(fold3_wells_r5d)}")
fold3_records_r5d = _load_records(fold3_wells_r5d, "r5d_fold3")
order_r5d = np.argsort([record.eval_rows for record in fold3_records_r5d])
selection_positions_r5d = np.unique(
    np.rint(
        np.linspace(0, len(order_r5d) - 1, SELECTION_WELLS_R5D)
    ).astype(int)
)
selection_indices_r5d = set(order_r5d[selection_positions_r5d].tolist())
selection_records_r5d = [
    record
    for index, record in enumerate(fold3_records_r5d)
    if index in selection_indices_r5d
]
confirmation_records_r5d = [
    record
    for index, record in enumerate(fold3_records_r5d)
    if index not in selection_indices_r5d
]

state_r5d, checkpoint_payload_r5d = _checkpoint_state_r5d(CHECKPOINT_R5D)
model_r5d = AxisAwareEvidenceR5().to(device_r5d)
model_r5d.load_state_dict(state_r5d)
model_r5d.eval()

selection_scan_r5d = _evaluate_doses_r5d(
    model_r5d,
    selection_records_r5d,
    DOSES_R5D,
    device_r5d,
    control="full",
)
selection_scan_r5d.to_csv(
    OUT_DIR_R5D / "selection_dose_scan.csv", index=False
)
print(
    "SEGN-R5D SELECTION SCAN\n"
    + selection_scan_r5d[
        [
            "dose",
            "rmse",
            "local_expected_rmse",
            "local_expected_correlation",
            "within_well_shape_correlation",
            "posterior_mass_within5",
            "mean_true_state_nll",
        ]
    ].to_string(index=False),
    flush=True,
)

selection_order_r5d = selection_scan_r5d.sort_values(
    ["rmse", "local_expected_rmse", "dose"],
    ascending=[True, True, True],
)
selected_dose_r5d = float(selection_order_r5d.iloc[0]["dose"])
confirmation_doses_r5d = np.unique(
    np.asarray([0.0, selected_dose_r5d, 1.0], np.float64)
)
confirmation_scan_r5d = _evaluate_doses_r5d(
    model_r5d,
    confirmation_records_r5d,
    confirmation_doses_r5d,
    device_r5d,
    control="full",
    save_selected_dose=selected_dose_r5d,
    save_prefix="fold3_confirmation_selected",
)
confirmation_scan_r5d.to_csv(
    OUT_DIR_R5D / "confirmation_dose_scan.csv", index=False
)
shuffled_scan_r5d = _evaluate_doses_r5d(
    model_r5d,
    confirmation_records_r5d,
    np.asarray([selected_dose_r5d], np.float64),
    device_r5d,
    control="shuffled_typewell",
)
shuffled_scan_r5d.to_csv(
    OUT_DIR_R5D / "confirmation_shuffled.csv", index=False
)

uniform_r5d = confirmation_scan_r5d.loc[
    np.isclose(confirmation_scan_r5d["dose"], 0.0)
].iloc[0].to_dict()
selected_r5d = confirmation_scan_r5d.loc[
    np.isclose(confirmation_scan_r5d["dose"], selected_dose_r5d)
].iloc[0].to_dict()
raw_r5d = confirmation_scan_r5d.loc[
    np.isclose(confirmation_scan_r5d["dose"], 1.0)
].iloc[0].to_dict()
shuffled_r5d = shuffled_scan_r5d.iloc[0].to_dict()

path_gain_r5d = float(uniform_r5d["rmse"] - selected_r5d["rmse"])
local_gain_r5d = float(
    uniform_r5d["local_expected_rmse"]
    - selected_r5d["local_expected_rmse"]
)
shuffle_path_degradation_r5d = float(
    shuffled_r5d["rmse"] - selected_r5d["rmse"]
)
shuffle_local_degradation_r5d = float(
    shuffled_r5d["local_expected_rmse"]
    - selected_r5d["local_expected_rmse"]
)
checks_r5d = {
    "selected_dose_positive": selected_dose_r5d > 0.0,
    "confirmation_path_gain_vs_uniform_ge_0p20": path_gain_r5d >= 0.20,
    "confirmation_local_gain_vs_uniform_ge_0p20": local_gain_r5d >= 0.20,
    "selected_local_expected_correlation_ge_0p20": float(
        selected_r5d["local_expected_correlation"]
    )
    >= 0.20,
    "shuffled_path_degradation_ge_0p30": shuffle_path_degradation_r5d >= 0.30,
    "shuffled_local_degradation_ge_0p30": shuffle_local_degradation_r5d >= 0.30,
}
strict_pass_r5d = bool(all(checks_r5d.values()))

verdict_r5d = {
    "experiment": EXPERIMENT_R5D,
    "stage": "fold3_frozen_axisaware_logit_dose_confirmation",
    "source": str(SOURCE_PATH_R5D),
    "checkpoint": str(CHECKPOINT_R5D),
    "checkpoint_epoch": checkpoint_payload_r5d.get("epoch"),
    "optimizer_steps": 0,
    "truth_usage": "teacher anchors for evidence observability only",
    "uniform_baseline_warning": (
        "dose=0 is a teacher-anchor uniform-posterior diagnostic; it is not "
        "a deployable no-GR model"
    ),
    "protocol": {
        "selection_fold": 3,
        "selection_wells": len(selection_records_r5d),
        "confirmation_wells": len(confirmation_records_r5d),
        "fold2_touched": False,
        "fold1_touched": False,
        "dose_grid": DOSES_R5D.tolist(),
        "selected_dose": selected_dose_r5d,
    },
    "selection": selection_order_r5d.iloc[0].to_dict(),
    "confirmation": {
        "uniform_dose_0": uniform_r5d,
        "selected": selected_r5d,
        "raw_dose_1": raw_r5d,
        "shuffled_selected": shuffled_r5d,
    },
    "diagnostics": {
        "path_gain_vs_uniform": path_gain_r5d,
        "local_gain_vs_uniform": local_gain_r5d,
        "shuffled_path_degradation": shuffle_path_degradation_r5d,
        "shuffled_local_degradation": shuffle_local_degradation_r5d,
    },
    "checks": checks_r5d,
    "strict_pass": strict_pass_r5d,
    "elapsed_minutes": (time.time() - started_r5d) / 60.0,
    "next": (
        "PASS -> freeze the calibrated evidence dose and test it inside R4 scheduled recurrence"
        if strict_pass_r5d
        else "FAIL -> close coarse R5 evidence; rebuild the work-note cost map at dense MD resolution"
    ),
}
_write_json(OUT_DIR_R5D / "verdict.json", verdict_r5d)
print(
    "SEGN-R5D CONFIRMATION\n"
    + confirmation_scan_r5d[
        [
            "dose",
            "rmse",
            "local_expected_rmse",
            "local_expected_correlation",
            "within_well_shape_correlation",
            "posterior_mass_within5",
        ]
    ].to_string(index=False),
    flush=True,
)
print("SEGN-R5D FINAL VERDICT", json.dumps(verdict_r5d, indent=2), flush=True)
print("saved:", OUT_DIR_R5D, flush=True)

if not strict_pass_r5d:
    raise RuntimeError(json.dumps(verdict_r5d, indent=2))
