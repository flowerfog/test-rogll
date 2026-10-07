"""E1 capacity gateway for the frozen Fold-1 PCA16 anchored-S basis.

The PCA basis and mean curve come exclusively from E0 outer-train wells.  This
cell trains the existing full-well encoder to predict orthogonal PCA scores and
evaluates exact native-row pooled TVT RMSE.  It runs independent 1/4/24-well
capacity tests and never starts held-out Fold-1 training.
"""

from pathlib import Path
import gc
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


REQUIRED_E1 = {
    "records",
    "train_wells",
    "valid_wells",
    "train_records",
    "z_arrays_s",
    "FastTrajectorySHead",
    "_input_fast",
    "_rmse",
    "_seed_everything",
    "OOF_ROOT",
}
missing_e1 = sorted(REQUIRED_E1.difference(globals()))
if missing_e1:
    raise RuntimeError(
        "Run E1 in the live kernel after the native model and E0 cells; missing "
        + repr(missing_e1)
    )


EXPERIMENT_E1 = "deltacorr_steerspline_e1_pca16_capacity_v0"
OUT_DIR_E1 = OOF_ROOT / EXPERIMENT_E1
MODEL_DIR_E1 = OUT_DIR_E1 / "models"
for directory in (OUT_DIR_E1, MODEL_DIR_E1):
    directory.mkdir(parents=True, exist_ok=True)

BASIS_PATH_E1 = (
    OOF_ROOT
    / "deltacorr_steerspline_e0_basis_audit_v0"
    / "fold1_basis_bank.npz"
)
if not BASIS_PATH_E1.exists():
    raise FileNotFoundError(BASIS_PATH_E1)

PCA_DIM_E1 = 16
RIDGE_E1 = 1e-5
PATH_SCALE_E1 = 8.0
LEARNING_RATE_E1 = 3e-4
MIN_LEARNING_RATE_E1 = 3e-5
WEIGHT_DECAY_E1 = 2e-5
GRADIENT_CLIP_E1 = 3.0
REPORT_EVERY_E1 = 10
ABSOLUTE_CAPACITY_GATE_E1 = 1.0
SCENARIOS_E1 = (
    ("one_well", 1, 140),
    ("four_wells", 4, 200),
    ("twentyfour_wells", 24, 400),
)
SEED_E1 = 20260725


def _json_e1(path, payload):
    def default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


basis_bank_e1 = np.load(BASIS_PATH_E1, allow_pickle=False)
progress_grid_e1 = basis_bank_e1["progress_grid"].astype(np.float64)
mean_grid_e1 = basis_bank_e1["train_mean"].astype(np.float64)
components_grid_e1 = basis_bank_e1["pca_components"][:PCA_DIM_E1].astype(
    np.float64
)

if abs(mean_grid_e1[0]) > 1e-7 or np.max(np.abs(components_grid_e1[:, 0])) > 1e-7:
    raise RuntimeError("PCA16 basis is not hard anchored at prediction-start")


def _target_e1(record):
    eval_index = record.eval_index
    denominator = max(
        float(record.md[-1] - record.md[record.anchor_index]), 1.0
    )
    progress = (
        record.md[eval_index].astype(np.float64)
        - float(record.md[record.anchor_index])
    ) / denominator
    progress = np.clip(progress, 0.0, 1.0)
    z = z_arrays_s[record.well]
    z_delta = (
        z[eval_index].astype(np.float64) - float(z[record.anchor_index])
    )
    target_delta_s = (
        record.truth[eval_index].astype(np.float64)
        - float(record.anchor_tvt)
        + z_delta
    )
    offset = np.interp(progress, progress_grid_e1, mean_grid_e1)
    design = np.stack(
        [
            np.interp(progress, progress_grid_e1, component)
            for component in components_grid_e1
        ],
        axis=1,
    )
    gram = design.T @ design
    rhs = design.T @ (target_delta_s - offset)
    coefficients = np.linalg.solve(
        gram + RIDGE_E1 * np.eye(PCA_DIM_E1), rhs
    )
    oracle_delta_s = offset + design @ coefficients
    truth_tvt = record.truth[eval_index].astype(np.float64)
    oracle_tvt = (
        float(record.anchor_tvt) + oracle_delta_s - z_delta
    )
    return {
        "progress": progress.astype(np.float32),
        "offset": offset.astype(np.float32),
        "design": design.astype(np.float32),
        "target_delta_s": target_delta_s.astype(np.float32),
        "coefficients": coefficients.astype(np.float32),
        "z_delta": z_delta.astype(np.float32),
        "truth_tvt": truth_tvt.astype(np.float32),
        "oracle_tvt": oracle_tvt.astype(np.float32),
    }


target_cache_e1 = {
    record.well: _target_e1(record)
    for record in train_records + [records[well] for well in valid_wells]
}
train_coefficients_e1 = np.stack(
    [target_cache_e1[well]["coefficients"] for well in train_wells]
).astype(np.float64)
coefficient_center_e1 = train_coefficients_e1.mean(axis=0).astype(np.float32)
coefficient_scale_e1 = train_coefficients_e1.std(axis=0).astype(np.float32)
coefficient_scale_e1 = np.maximum(coefficient_scale_e1, 1.0).astype(np.float32)


def _decode_e1(record, coefficients):
    target = target_cache_e1[record.well]
    delta_s = target["offset"] + target["design"] @ np.asarray(
        coefficients, np.float32
    )
    prediction = (
        float(record.anchor_tvt) + delta_s - target["z_delta"]
    ).astype(np.float64)
    return target["truth_tvt"].astype(np.float64), prediction


def _fixed_metrics_e1(evaluation_records, coefficients, oracle=False):
    truths, predictions, bywell = [], [], []
    for record in evaluation_records:
        target = target_cache_e1[record.well]
        if oracle:
            truth = target["truth_tvt"].astype(np.float64)
            prediction = target["oracle_tvt"].astype(np.float64)
        else:
            truth, prediction = _decode_e1(record, coefficients)
        truths.append(truth)
        predictions.append(prediction)
        bywell.append(_rmse(truth, prediction))
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    return {
        "rmse": _rmse(truth, prediction),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
        "max_well_rmse": float(np.max(bywell)),
    }


def _loss_e1(standardised_prediction, record, device, row_weight):
    target = target_cache_e1[record.well]
    offset = torch.from_numpy(target["offset"]).to(device)
    design = torch.from_numpy(target["design"]).to(device)
    target_delta_s = torch.from_numpy(target["target_delta_s"]).to(device)
    target_coefficients = torch.from_numpy(target["coefficients"]).to(device)
    center = torch.from_numpy(coefficient_center_e1).to(device)
    scale = torch.from_numpy(coefficient_scale_e1).to(device)
    predicted_coefficients = center + scale * standardised_prediction[0]
    predicted_delta_s = offset + design @ predicted_coefficients
    target_standard = (target_coefficients - center) / scale
    path_loss = torch.mean(
        ((predicted_delta_s - target_delta_s) / PATH_SCALE_E1) ** 2
    )
    coefficient_loss = torch.mean(
        (standardised_prediction[0] - target_standard) ** 2
    )
    slope_loss = F.smooth_l1_loss(
        torch.diff(predicted_delta_s) / 2.0,
        torch.diff(target_delta_s) / 2.0,
    )
    loss = row_weight * (
        path_loss + 0.08 * coefficient_loss + 0.025 * slope_loss
    )
    return loss, path_loss.detach(), coefficient_loss.detach()


@torch.no_grad()
def _evaluate_e1(model, evaluation_records, device):
    model.eval()
    truths, predictions, bywell = [], [], []
    for record in evaluation_records:
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            standard = model(_input_fast(record, device))[0]
        coefficients = (
            coefficient_center_e1
            + coefficient_scale_e1 * standard.float().cpu().numpy()
        )
        truth, prediction = _decode_e1(record, coefficients)
        truths.append(truth)
        predictions.append(prediction)
        bywell.append(
            {
                "well": record.well,
                "rows": len(truth),
                "rmse": _rmse(truth, prediction),
                "mae": float(np.mean(np.abs(prediction - truth))),
            }
        )
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    bywell = pd.DataFrame(bywell)
    return {
        "rows": len(truth),
        "wells": len(evaluation_records),
        "rmse": _rmse(truth, prediction),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
        "wells_above_12ft": int((bywell["rmse"] > 12).sum()),
    }, bywell


def _capacity_pool_e1():
    rows = []
    for well in train_wells:
        record = records[well]
        rows.append(
            {
                "well": well,
                "eval_rows": record.eval_rows,
                "gr_missing": float(1.0 - record.horizontal[1].mean()),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["gr_missing", "eval_rows"])
    indices = np.linspace(0, len(frame) - 1, 24).round().astype(int)
    return [records[well] for well in frame.iloc[indices]["well"]]


def _train_capacity_e1(model, training_records, device, epochs, seed, gate):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_E1,
        weight_decay=WEIGHT_DECAY_E1,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=MIN_LEARNING_RATE_E1
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=512.0
    )
    mean_rows = float(np.mean([record.eval_rows for record in training_records]))
    rng = np.random.default_rng(seed)
    history, best = [], None
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = path_sum = coefficient_sum = 0.0
        skipped = 0
        for index in rng.permutation(len(training_records)):
            record = training_records[int(index)]
            optimizer.zero_grad(set_to_none=True)
            values = _input_fast(record, device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                standard = model(values)
                loss, path_loss, coefficient_loss = _loss_e1(
                    standard,
                    record,
                    device,
                    record.eval_rows / max(mean_rows, 1.0),
                )
            before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_E1)
            scaler.step(optimizer)
            scaler.update()
            skipped += int(float(scaler.get_scale()) < before)
            loss_sum += float(loss.detach().cpu())
            path_sum += float(path_loss.cpu())
            coefficient_sum += float(coefficient_loss.cpu())
        scheduler.step()
        if epoch != 1 and epoch % REPORT_EVERY_E1 != 0 and epoch != epochs:
            continue
        metrics, bywell = _evaluate_e1(model, training_records, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(training_records),
            "train_path_mse_scaled": path_sum / len(training_records),
            "train_coefficient_loss": coefficient_sum / len(training_records),
            **metrics,
            "skipped_steps": skipped,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("E1 PCA16 CAPACITY EPOCH", row, flush=True)
        if best is None or metrics["rmse"] < best["metrics"]["rmse"]:
            best = {
                "epoch": epoch,
                "metrics": metrics,
                "bywell": bywell.copy(),
                "state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            }
        if epoch >= 40 and metrics["rmse"] <= gate:
            break
    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history), (time.time() - started) / 60.0


started_e1 = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("E1 capacity requires CUDA")
device_e1 = torch.device("cuda")
pool_e1 = _capacity_pool_e1()
capacity_results_e1 = []
capacity_histories_e1 = []
parameters_e1 = None

print(
    "E1 PCA16 CONFIG",
    {
        "basis": str(BASIS_PATH_E1),
        "basis_fit_scope": "outer_train_only",
        "pca_dimensions": PCA_DIM_E1,
        "absolute_capacity_gate_ft": ABSOLUTE_CAPACITY_GATE_E1,
        "train_coefficient_abs_mean_max": float(
            np.max(np.abs(coefficient_center_e1))
        ),
        "train_coefficient_scale_max": float(np.max(coefficient_scale_e1)),
    },
    flush=True,
)

for scenario_index, (scenario, well_count, epochs) in enumerate(SCENARIOS_E1):
    indices = np.linspace(0, len(pool_e1) - 1, well_count).round().astype(int)
    scenario_records = [pool_e1[index] for index in indices]
    oracle = _fixed_metrics_e1(
        scenario_records, np.zeros(PCA_DIM_E1), oracle=True
    )
    baseline = _fixed_metrics_e1(
        scenario_records, coefficient_center_e1, oracle=False
    )
    if oracle["rmse"] >= ABSOLUTE_CAPACITY_GATE_E1:
        scenario_result = {
            "scenario": scenario,
            "wells": well_count,
            "representation_oracle": oracle,
            "gate_rmse": ABSOLUTE_CAPACITY_GATE_E1,
            "pass": False,
            "failure": "representation oracle makes the absolute gate impossible",
        }
        capacity_results_e1.append(scenario_result)
        break
    _seed_everything(SEED_E1 + scenario_index)
    model = FastTrajectorySHead(PCA_DIM_E1).to(device_e1)
    parameters_e1 = int(sum(parameter.numel() for parameter in model.parameters()))
    print(
        "E1 PCA16 CAPACITY START",
        {
            "scenario": scenario,
            "wells": well_count,
            "epochs": epochs,
            "parameters": parameters_e1,
            "baseline": baseline,
            "oracle": oracle,
        },
        flush=True,
    )
    model, best, history, runtime = _train_capacity_e1(
        model,
        scenario_records,
        device_e1,
        epochs,
        SEED_E1 + scenario_index,
        ABSOLUTE_CAPACITY_GATE_E1,
    )
    history.insert(0, "scenario", scenario)
    capacity_histories_e1.append(history)
    history.to_csv(OUT_DIR_E1 / f"history_{scenario}.csv", index=False)
    best["bywell"].to_csv(OUT_DIR_E1 / f"bywell_{scenario}.csv", index=False)
    passed = bool(best["metrics"]["rmse"] <= ABSOLUTE_CAPACITY_GATE_E1)
    checkpoint = MODEL_DIR_E1 / f"best_{scenario}.pt"
    torch.save(
        {
            "experiment": EXPERIMENT_E1,
            "scenario": scenario,
            "epoch": int(best["epoch"]),
            "model": best["state"],
            "metrics": best["metrics"],
            "pca_dimensions": PCA_DIM_E1,
            "coefficient_center": coefficient_center_e1,
            "coefficient_scale": coefficient_scale_e1,
            "basis_path": str(BASIS_PATH_E1),
        },
        checkpoint,
    )
    scenario_result = {
        "scenario": scenario,
        "wells": well_count,
        "epochs_budget": epochs,
        "best_epoch": int(best["epoch"]),
        "baseline": baseline,
        "representation_oracle": oracle,
        "gate_rmse": ABSOLUTE_CAPACITY_GATE_E1,
        "result": best["metrics"],
        "runtime_minutes": runtime,
        "checkpoint": str(checkpoint),
        "pass": passed,
    }
    capacity_results_e1.append(scenario_result)
    print("E1 PCA16 CAPACITY RESULT", scenario_result, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if not passed:
        break

strict_pass_e1 = bool(
    len(capacity_results_e1) == len(SCENARIOS_E1)
    and all(result["pass"] for result in capacity_results_e1)
)
verdict_e1 = {
    "experiment": EXPERIMENT_E1,
    "stage": "E1_PCA16_capacity",
    "basis_method": "fold_clean_pca",
    "pca_dimensions": PCA_DIM_E1,
    "parameters": parameters_e1,
    "capacity_gate_rmse_le": ABSOLUTE_CAPACITY_GATE_E1,
    "capacity_results": capacity_results_e1,
    "strict_pass": strict_pass_e1,
    "elapsed_minutes": (time.time() - started_e1) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> train the held-out PCA16 E1 baseline before adding DeltaCorr"
        if strict_pass_e1
        else "FAIL -> stop deterministic PCA SteerSpline before DeltaCorr"
    ),
}
_json_e1(OUT_DIR_E1 / "verdict.json", verdict_e1)

if capacity_histories_e1:
    print("\nE1 CAPACITY HISTORY")
    print(pd.concat(capacity_histories_e1, ignore_index=True).to_string(index=False))
print("\nE1 FINAL VERDICT")
print(json.dumps(verdict_e1, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_E1, flush=True)
