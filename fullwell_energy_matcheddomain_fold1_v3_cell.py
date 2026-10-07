"""Matched-domain complete-well energy training and untouched Fold1 audit.

Prerequisite: run the latest
fullwell_energy_densemode_fold1_application_v2_cell.py in this kernel. Its
restore stage supplies the frozen raw banks, prefix matcher, dense-volume
builder, mode generator, complete-well energy architecture, and Fold1 bank.

Protocol:
* retrain the historical core matcher on folds 3/4/5 only;
* generate a label-free B192 bank for Fold2;
* split Fold2 wells deterministically into energy-train/calibration groups;
* select the energy epoch on the calibration wells;
* refit the energy model on all Fold2 wells for that frozen epoch count;
* apply once to the existing label-free Fold1 B192 bank.

Candidate MSE is a training label on Fold2 only. It is never an input feature.
Fold1 truth enters only after the tournament has frozen every selected index.
"""

from __future__ import annotations

from pathlib import Path
import gc
import hashlib
import json
import math
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


REQUIRED_V3 = {
    "OOF_ROOT",
    "CompleteWellAliasEnergy",
    "MIN_CANDIDATE_COVERAGE",
    "MODE_BUDGET_V2",
    "fold1_volumes_dense",
    "core_wells",
    "tune_wells",
    "outer_train_wells",
    "samples",
    "prefix_queries_v1",
    "_global_prefix_subset_v1",
    "_concat_query_v1",
    "_fit_mixed_v1",
    "mixed_epochs_v1",
    "SEED_DENSE",
    "_build_volumes_d0",
    "_candidate_bank_v2",
    "_calibrated_well_v2",
    "_alignment_panel_v2",
    "_tournament_v2",
    "_metrics_v2",
    "_merge_payloads_v2",
    "_json_default",
}
missing_v3 = sorted(REQUIRED_V3.difference(globals()))
if missing_v3:
    raise RuntimeError(
        "Run the latest fullwell_energy_densemode_fold1_application_v2_cell.py "
        "first; missing live state: " + repr(missing_v3)
    )


EXPERIMENT_V3 = "fullwell_energy_matcheddomain_fold1_v3"
OUT_DIR_V3 = Path(OOF_ROOT) / EXPERIMENT_V3
PANEL_DIR_V3 = OUT_DIR_V3 / "fold2_panels"
MODEL_DIR_V3 = OUT_DIR_V3 / "models"
for directory in (OUT_DIR_V3, PANEL_DIR_V3, MODEL_DIR_V3):
    directory.mkdir(parents=True, exist_ok=True)

SEED_V3 = 20260723
CALIBRATION_FRACTION_V3 = 0.23
MAX_EPOCHS_V3 = 12
MIN_EPOCHS_V3 = 3
PATIENCE_V3 = 4
LEARNING_RATE_V3 = 1.0e-4
MIN_LEARNING_RATE_V3 = 1.5e-5
WEIGHT_DECAY_V3 = 3.0e-5
GRAD_ACCUM_WELLS_V3 = 4
GRADIENT_CLIP_V3 = 4.0
GROUP_CANDIDATES_V3 = 49
NEAR_CANDIDATES_V3 = 24
HARD_CANDIDATES_V3 = 12
SOFT_LABEL_SCALE_FT_V3 = 4.0
LOSS_EXPECTED_V3 = 0.15
LOSS_MARGIN_V3 = 0.10
PAIR_MARGIN_V3 = 0.50

GATE_FOLD1_RMSE_V3 = 8.0
TARGET_FOLD1_RMSE_V3 = 5.5
GATE_GAIN_VS_COMMON_V3 = 0.50
GATE_POSITIVE_WELLS_V3 = 0.55
GATE_CONTROL_DEGRADATION_V3 = 0.50


def _stable_fraction_v3(well):
    digest = hashlib.blake2b(
        f"{well}|matched-domain|{SEED_V3}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") / float(2**64 - 1)


def _panel_path_v3(well):
    return PANEL_DIR_V3 / f"{well}.npz"


def _ensure_mixed_core_queries_v3():
    for well in outer_train_wells:
        query = samples[well].train
        if query is None:
            raise RuntimeError(f"{well}: missing suffix query")
        # The restored bank is suffix-only. A mixed query has the historical
        # fixed prefix count added and therefore a larger requested-row count.
        if int(query.requested_rows) <= 192:
            prefix = _global_prefix_subset_v1(prefix_queries_v1[well])
            samples[well].train = _concat_query_v1(query, prefix)


def _save_fold2_panel_v3(well, volume):
    paths, metadata, truth, distance = _candidate_bank_v2(well, volume)
    well_data = _calibrated_well_v2(well)
    query, reference, valid, coverage = _alignment_panel_v2(well_data, paths)
    candidate_mse = np.mean(
        (paths.astype(np.float64) - truth[None, :].astype(np.float64)) ** 2,
        axis=1,
    ).astype(np.float32)
    available = coverage >= MIN_CANDIDATE_COVERAGE
    if int(available.sum()) < GROUP_CANDIDATES_V3:
        raise RuntimeError(
            f"{well}: only {int(available.sum())} available candidates"
        )
    np.savez_compressed(
        _panel_path_v3(well),
        well=np.asarray(well),
        query=query.astype(np.float16),
        reference=reference.astype(np.float16),
        valid=valid.astype(np.uint8),
        coverage=coverage.astype(np.float32),
        candidate_mse=candidate_mse,
        common_energy=metadata["common_energy"].to_numpy(np.float32),
        eval_rows=np.asarray([len(truth)], np.int32),
        cache_version=np.asarray([1], np.int16),
    )
    return {
        "well": well,
        "candidates": len(paths),
        "available_candidates": int(available.sum()),
        "oracle_rmse": float(np.sqrt(candidate_mse.min())),
    }


def _load_fold2_panel_v3(well):
    with np.load(_panel_path_v3(well), allow_pickle=False) as data:
        return {
            "well": str(data["well"].item()),
            "query": data["query"].astype(np.float32),
            "reference": data["reference"].astype(np.float32),
            "valid": data["valid"].astype(bool),
            "coverage": data["coverage"].astype(np.float32),
            "candidate_mse": data["candidate_mse"].astype(np.float32),
            "common_energy": data["common_energy"].astype(np.float32),
            "eval_rows": int(data["eval_rows"][0]),
        }


def _training_group_v3(panel, rng):
    available = np.flatnonzero(
        panel["coverage"] >= MIN_CANDIDATE_COVERAGE
    )
    mse = panel["candidate_mse"]
    target = int(available[np.argmin(mse[available])])
    remaining = available[available != target]
    by_error = remaining[np.argsort(mse[remaining])]
    near = by_error[: min(NEAR_CANDIDATES_V3, len(by_error))]
    hard = by_error[::-1][: min(HARD_CANDIDATES_V3, len(by_error))]
    chosen = list(dict.fromkeys([target, *near.tolist(), *hard.tolist()]))
    pool = np.asarray(
        [candidate for candidate in remaining if candidate not in set(chosen)],
        np.int64,
    )
    need = GROUP_CANDIDATES_V3 - len(chosen)
    if need > 0:
        if len(pool) < need:
            raise RuntimeError(f"{panel['well']}: insufficient candidate pool")
        chosen.extend(rng.choice(pool, size=need, replace=False).tolist())
    indices = np.asarray(chosen[:GROUP_CANDIDATES_V3], np.int64)
    permutation = rng.permutation(len(indices))
    indices = indices[permutation]
    target_local = int(np.flatnonzero(indices == target)[0])
    return indices, target_local


def _panel_tensors_v3(panel, indices, device):
    return (
        torch.from_numpy(panel["query"]).to(device),
        torch.from_numpy(panel["reference"][indices]).to(device),
        torch.from_numpy(panel["valid"][indices].astype(np.float32)).to(device),
        torch.from_numpy(panel["candidate_mse"][indices]).to(device),
    )


def _loss_v3(model, panel, device, rng):
    indices, target_local = _training_group_v3(panel, rng)
    query, reference, valid, mse = _panel_tensors_v3(panel, indices, device)
    logits = model(query, reference, valid)
    excess = (mse - mse.min()).clamp_min(0.0)
    target_probability = torch.softmax(
        -excess / (2.0 * SOFT_LABEL_SCALE_FT_V3**2), dim=0
    )
    listwise = -(target_probability * torch.log_softmax(logits, dim=0)).sum()
    probability = torch.softmax(logits, dim=0)
    expected = torch.sum(probability * excess) / (12.0**2)
    hard = excess >= 12.0**2
    if hard.any():
        margin = F.relu(
            PAIR_MARGIN_V3 - logits[target_local] + logits[hard].amax()
        )
    else:
        margin = logits[target_local] * 0.0
    loss = listwise + LOSS_EXPECTED_V3 * expected + LOSS_MARGIN_V3 * margin
    return loss, {
        "listwise": float(listwise.detach().cpu()),
        "expected": float(expected.detach().cpu()),
        "margin": float(margin.detach().cpu()),
    }


def _train_epoch_v3(model, wells, optimizer, scaler, device, epoch):
    model.train()
    order = list(wells)
    random.Random(SEED_V3 + epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "listwise": 0.0, "expected": 0.0, "margin": 0.0}
    for step, well in enumerate(order, start=1):
        panel = _load_fold2_panel_v3(well)
        rng = np.random.default_rng(SEED_V3 + epoch * 10007 + step)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            loss, parts = _loss_v3(model, panel, device, rng)
            scaled = loss / GRAD_ACCUM_WELLS_V3
        scaler.scale(scaled).backward()
        if step % GRAD_ACCUM_WELLS_V3 == 0 or step == len(order):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_V3)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        totals["loss"] += float(loss.detach().cpu())
        for key, value in parts.items():
            totals[key] += value
    return {key: value / max(len(order), 1) for key, value in totals.items()}


@torch.no_grad()
def _evaluate_fold2_panels_v3(model, wells, device):
    rows = []
    total_sse = 0.0
    total_rows = 0
    for well in wells:
        panel = _load_fold2_panel_v3(well)
        final_indices, logits, _ = _tournament_v2(
            model,
            panel["query"],
            panel["reference"],
            panel["valid"],
            device,
            "full",
        )
        selected = int(final_indices[int(np.argmax(logits))])
        mse = float(panel["candidate_mse"][selected])
        oracle = float(panel["candidate_mse"].min())
        common = int(np.argmax(panel["common_energy"]))
        eval_rows = panel["eval_rows"]
        total_sse += eval_rows * mse
        total_rows += eval_rows
        rows.append(
            {
                "well": well,
                "eval_rows": eval_rows,
                "selected": selected,
                "rmse": math.sqrt(mse),
                "oracle_rmse": math.sqrt(oracle),
                "common_rmse": math.sqrt(
                    float(panel["candidate_mse"][common])
                ),
                "oracle_promoted": bool(
                    int(np.argmin(panel["candidate_mse"]))
                    in set(final_indices.tolist())
                ),
            }
        )
    return {
        "rows": total_rows,
        "wells": len(rows),
        "rmse": math.sqrt(total_sse / total_rows),
        "mean_well_rmse": float(pd.DataFrame(rows)["rmse"].mean()),
        "median_well_rmse": float(pd.DataFrame(rows)["rmse"].median()),
        "oracle_promoted_fraction": float(
            pd.DataFrame(rows)["oracle_promoted"].mean()
        ),
    }, pd.DataFrame(rows)


def _new_energy_model_v3(device):
    checkpoint = (
        Path(OOF_ROOT)
        / "fullwell_nonrigid_warp_observability_fold2_v1"
        / "models"
        / "best_fold2_nonrigid_observability.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = CompleteWellAliasEnergy().to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, checkpoint


def _fit_energy_v3(train_wells, epochs, device, phase):
    model, checkpoint = _new_energy_model_v3(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_V3, weight_decay=WEIGHT_DECAY_V3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=MIN_LEARNING_RATE_V3
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, epochs + 1):
        metrics = _train_epoch_v3(
            model, train_wells, optimizer, scaler, device, epoch
        )
        history.append({"phase": phase, "epoch": epoch, **metrics})
        scheduler.step()
    return model, pd.DataFrame(history), checkpoint


def _score_fold1_v3(model, device):
    controls = ("full", "no_gr", "shuffled_alignment")
    selected_payloads = {control: [] for control in controls}
    posterior_payloads = {control: [] for control in controls}
    oracle_payloads = []
    common_payloads = []
    rows = []

    for index, well in enumerate(sorted(fold1_volumes_dense), start=1):
        paths, metadata, truth, distance = _candidate_bank_v2(
            well, fold1_volumes_dense[well]
        )
        well_data = _calibrated_well_v2(well)
        query, reference, valid, coverage = _alignment_panel_v2(well_data, paths)
        common_index = int(metadata["common_energy"].to_numpy().argmax())
        row = {
            "well": well,
            "candidates": len(paths),
            "common_index": common_index,
        }
        frozen = {}
        for control in controls:
            final_indices, logits, _ = _tournament_v2(
                model, query, reference, valid, device, control
            )
            selected = int(final_indices[int(np.argmax(logits))])
            probability = np.exp(logits - np.max(logits))
            probability /= probability.sum()
            posterior = np.sum(
                probability[:, None] * paths[final_indices], axis=0
            ).astype(np.float32)
            frozen[control] = (selected, final_indices, posterior)
            selected_payloads[control].append(
                {
                    "well": np.full(len(truth), well, dtype="U16"),
                    "truth": truth,
                    "prediction": paths[selected],
                    "distance": distance,
                }
            )
            posterior_payloads[control].append(
                {
                    "well": np.full(len(truth), well, dtype="U16"),
                    "truth": truth,
                    "prediction": posterior,
                    "distance": distance,
                }
            )
            row[f"{control}_selected"] = selected

        # Frozen selection boundary: truth is consumed only below this line.
        candidate_sse = np.sum(
            (paths.astype(np.float64) - truth[None].astype(np.float64)) ** 2,
            axis=1,
        )
        oracle_index = int(np.argmin(candidate_sse))
        row["oracle_index"] = oracle_index
        row["oracle_rmse"] = math.sqrt(candidate_sse[oracle_index] / len(truth))
        row["common_rmse"] = math.sqrt(candidate_sse[common_index] / len(truth))
        oracle_payloads.append(
            {
                "well": np.full(len(truth), well, dtype="U16"),
                "truth": truth,
                "prediction": paths[oracle_index],
                "distance": distance,
            }
        )
        common_payloads.append(
            {
                "well": np.full(len(truth), well, dtype="U16"),
                "truth": truth,
                "prediction": paths[common_index],
                "distance": distance,
            }
        )
        for control, (selected, final_indices, posterior) in frozen.items():
            row[f"{control}_rmse"] = math.sqrt(
                candidate_sse[selected] / len(truth)
            )
            row[f"{control}_oracle_promoted"] = bool(
                oracle_index in set(final_indices.tolist())
            )
        rows.append(row)
        if index % 20 == 0 or index == len(fold1_volumes_dense):
            print(
                f"matched-domain Fold1 {index}/{len(fold1_volumes_dense)}",
                flush=True,
            )

    selected = {
        control: _merge_payloads_v2(payloads)
        for control, payloads in selected_payloads.items()
    }
    posterior = {
        control: _merge_payloads_v2(payloads)
        for control, payloads in posterior_payloads.items()
    }
    oracle = _merge_payloads_v2(oracle_payloads)
    common = _merge_payloads_v2(common_payloads)
    metrics_selected = {
        control: _metrics_v2(payload)[0] for control, payload in selected.items()
    }
    metrics_posterior = {
        control: _metrics_v2(payload)[0] for control, payload in posterior.items()
    }
    metrics_oracle, oracle_bywell = _metrics_v2(oracle)
    metrics_common, common_bywell = _metrics_v2(common)
    return {
        "selected": metrics_selected,
        "posterior": metrics_posterior,
        "oracle": metrics_oracle,
        "common": metrics_common,
        "rows": pd.DataFrame(rows),
        "selected_payload": selected,
        "posterior_payload": posterior,
        "oracle_payload": oracle,
        "common_payload": common,
        "oracle_bywell": oracle_bywell,
        "common_bywell": common_bywell,
    }


started_v3 = time.time()
device_v3 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device_v3.type != "cuda":
    raise RuntimeError("CUDA is required")

print("MATCHED DOMAIN: restoring fold-clean core matcher", flush=True)
_ensure_mixed_core_queries_v3()
core_model_v3, core_history_v3 = _fit_mixed_v1(
    core_wells, mixed_epochs_v1, SEED_DENSE, "energy_domain_core"
)
core_history_v3.to_csv(OUT_DIR_V3 / "core_matcher_history.csv", index=False)
fold2_volumes_v3, fold2_volume_report_v3 = _build_volumes_d0(
    core_model_v3, tune_wells, "fold2_energy_domain"
)
fold2_volume_report_v3.to_csv(
    OUT_DIR_V3 / "fold2_volume_report.csv", index=False
)
del core_model_v3
gc.collect()
torch.cuda.empty_cache()

panel_rows_v3 = []
for index, well in enumerate(tqdm(sorted(fold2_volumes_v3), desc="Fold2 B192 panels"), 1):
    if not _panel_path_v3(well).exists():
        panel_rows_v3.append(_save_fold2_panel_v3(well, fold2_volumes_v3[well]))
    else:
        panel = _load_fold2_panel_v3(well)
        panel_rows_v3.append(
            {
                "well": well,
                "candidates": len(panel["candidate_mse"]),
                "available_candidates": int(
                    (panel["coverage"] >= MIN_CANDIDATE_COVERAGE).sum()
                ),
                "oracle_rmse": float(np.sqrt(panel["candidate_mse"].min())),
            }
        )
panel_report_v3 = pd.DataFrame(panel_rows_v3)
panel_report_v3.to_csv(OUT_DIR_V3 / "fold2_panel_report.csv", index=False)
print("FOLD2 PANEL GUARDS", panel_report_v3.describe().to_dict(), flush=True)

energy_train_wells_v3 = sorted(
    well for well in tune_wells if _stable_fraction_v3(well) >= CALIBRATION_FRACTION_V3
)
energy_calibration_wells_v3 = sorted(
    set(tune_wells) - set(energy_train_wells_v3)
)
if len(energy_train_wells_v3) < 105 or len(energy_calibration_wells_v3) < 25:
    raise RuntimeError(
        f"bad energy split {len(energy_train_wells_v3)}/{len(energy_calibration_wells_v3)}"
    )

calibration_model_v3, warm_checkpoint_v3 = _new_energy_model_v3(device_v3)
optimizer_v3 = torch.optim.AdamW(
    calibration_model_v3.parameters(),
    lr=LEARNING_RATE_V3,
    weight_decay=WEIGHT_DECAY_V3,
)
scheduler_v3 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_v3, T_max=MAX_EPOCHS_V3, eta_min=MIN_LEARNING_RATE_V3
)
scaler_v3 = torch.amp.GradScaler("cuda", enabled=True)
history_v3 = []
best_v3 = {"rmse": np.inf, "epoch": 0, "state": None}
bad_v3 = 0
for epoch in range(1, MAX_EPOCHS_V3 + 1):
    training = _train_epoch_v3(
        calibration_model_v3,
        energy_train_wells_v3,
        optimizer_v3,
        scaler_v3,
        device_v3,
        epoch,
    )
    calibration, _ = _evaluate_fold2_panels_v3(
        calibration_model_v3, energy_calibration_wells_v3, device_v3
    )
    report = {
        "epoch": epoch,
        **{f"train_{key}": value for key, value in training.items()},
        **{f"calibration_{key}": value for key, value in calibration.items()},
        "elapsed_minutes": (time.time() - started_v3) / 60.0,
    }
    history_v3.append(report)
    print("MATCHED ENERGY EPOCH", report, flush=True)
    if calibration["rmse"] < best_v3["rmse"] - 1e-6:
        best_v3 = {
            "rmse": calibration["rmse"],
            "epoch": epoch,
            "state": {
                key: value.detach().cpu().clone()
                for key, value in calibration_model_v3.state_dict().items()
            },
        }
        bad_v3 = 0
    else:
        bad_v3 += 1
    scheduler_v3.step()
    if epoch >= MIN_EPOCHS_V3 and bad_v3 >= PATIENCE_V3:
        break
pd.DataFrame(history_v3).to_csv(
    OUT_DIR_V3 / "energy_epoch_selection.csv", index=False
)
del calibration_model_v3
gc.collect()
torch.cuda.empty_cache()

if best_v3["epoch"] <= 0:
    raise RuntimeError("No matched-domain epoch selected")
final_model_v3, refit_history_v3, _ = _fit_energy_v3(
    sorted(tune_wells), best_v3["epoch"], device_v3, "fold2_refit"
)
refit_history_v3.to_csv(OUT_DIR_V3 / "energy_refit_history.csv", index=False)
checkpoint_v3 = MODEL_DIR_V3 / "fold1_matched_domain_energy.pt"
torch.save(
    {
        "experiment": EXPERIMENT_V3,
        "epoch": best_v3["epoch"],
        "model": {
            key: value.detach().cpu().clone()
            for key, value in final_model_v3.state_dict().items()
        },
        "train_wells": sorted(tune_wells),
        "warm_start": str(warm_checkpoint_v3),
    },
    checkpoint_v3,
)

fold1_v3 = _score_fold1_v3(final_model_v3, device_v3)
fold1_v3["rows"].to_csv(OUT_DIR_V3 / "fold1_bywell.csv", index=False)
full_metrics_v3 = fold1_v3["selected"]["full"]
common_metrics_v3 = fold1_v3["common"]
gain_vs_common_v3 = common_metrics_v3["rmse"] - full_metrics_v3["rmse"]
no_gr_degradation_v3 = (
    fold1_v3["selected"]["no_gr"]["rmse"] - full_metrics_v3["rmse"]
)
shuffle_degradation_v3 = (
    fold1_v3["selected"]["shuffled_alignment"]["rmse"]
    - full_metrics_v3["rmse"]
)
positive_fraction_v3 = float(
    np.mean(
        fold1_v3["rows"]["full_rmse"]
        < fold1_v3["rows"]["common_rmse"]
    )
)

strict_pass_v3 = bool(
    full_metrics_v3["rmse"] <= GATE_FOLD1_RMSE_V3
    and gain_vs_common_v3 >= GATE_GAIN_VS_COMMON_V3
    and positive_fraction_v3 >= GATE_POSITIVE_WELLS_V3
    and no_gr_degradation_v3 >= GATE_CONTROL_DEGRADATION_V3
    and shuffle_degradation_v3 >= GATE_CONTROL_DEGRADATION_V3
)
target_pass_v3 = bool(full_metrics_v3["rmse"] <= TARGET_FOLD1_RMSE_V3)

np.savez_compressed(
    OUT_DIR_V3 / "fold1_predictions.npz",
    well=fold1_v3["selected_payload"]["full"]["well"],
    truth=fold1_v3["selected_payload"]["full"]["truth"],
    distance=fold1_v3["selected_payload"]["full"]["distance"],
    selected=fold1_v3["selected_payload"]["full"]["prediction"],
    posterior=fold1_v3["posterior_payload"]["full"]["prediction"],
    common=fold1_v3["common_payload"]["prediction"],
    oracle=fold1_v3["oracle_payload"]["prediction"],
)

verdict_v3 = {
    "experiment": EXPERIMENT_V3,
    "stage": "matched_domain_fold2_training_fold1_application",
    "sample_unit": "one complete physical well candidate set",
    "candidate_generation_uses_truth": False,
    "candidate_training_labels": "Fold2 exact candidate SSE only",
    "candidate_features_include_mse": False,
    "protocol": {
        "matcher_train_folds": [3, 4, 5],
        "energy_domain_fold": 2,
        "energy_train_wells": len(energy_train_wells_v3),
        "energy_calibration_wells": len(energy_calibration_wells_v3),
        "selected_epoch": best_v3["epoch"],
        "energy_refit_wells": len(tune_wells),
        "application_fold": 1,
    },
    "selected": fold1_v3["selected"],
    "posterior": fold1_v3["posterior"],
    "common_energy_baseline": fold1_v3["common"],
    "oracle": fold1_v3["oracle"],
    "diagnostics": {
        "gain_vs_common": gain_vs_common_v3,
        "positive_well_fraction": positive_fraction_v3,
        "no_gr_degradation": no_gr_degradation_v3,
        "shuffled_alignment_degradation": shuffle_degradation_v3,
    },
    "gates": {
        "rmse_le": GATE_FOLD1_RMSE_V3,
        "target_rmse_le": TARGET_FOLD1_RMSE_V3,
        "gain_vs_common_ge": GATE_GAIN_VS_COMMON_V3,
        "positive_wells_ge": GATE_POSITIVE_WELLS_V3,
        "each_control_degradation_ge": GATE_CONTROL_DEGRADATION_V3,
    },
    "strict_pass": strict_pass_v3,
    "target_5p5_pass": target_pass_v3,
    "checkpoint": checkpoint_v3,
    "elapsed_minutes": (time.time() - started_v3) / 60.0,
    "free_gb": shutil.disk_usage(OUT_DIR_V3).free / 2**30,
    "next": (
        "PASS -> expand matched-domain training to five cross-fitted banks"
        if strict_pass_v3
        else "FAIL -> close generated-mode selection; complete-well observability did not survive proposal error"
    ),
}
(OUT_DIR_V3 / "verdict.json").write_text(
    json.dumps(verdict_v3, indent=2, default=_json_default), encoding="utf-8"
)

print("SELECTED", json.dumps(fold1_v3["selected"], indent=2), flush=True)
print("POSTERIOR", json.dumps(fold1_v3["posterior"], indent=2), flush=True)
print("ORACLE", json.dumps(fold1_v3["oracle"], indent=2), flush=True)
print("FINAL VERDICT", json.dumps(verdict_v3, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR_V3, flush=True)

del final_model_v3, fold2_volumes_v3
gc.collect()
torch.cuda.empty_cache()
