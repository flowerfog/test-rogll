"""Corrected E1 PCA16 1/4/24-well capacity ladder.

Each physical well receives the same number of optimiser updates.  Training
uses a PCA-coefficient warm-up before exact native-row path fine-tuning, as
validated by the one-well loss diagnostic.  This cell never starts Fold 1.
"""

from pathlib import Path
import gc
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch


REQUIRED_V2 = {
    "pool_e1",
    "target_cache_e1",
    "coefficient_center_e1",
    "coefficient_scale_e1",
    "PCA_DIM_E1",
    "FastTrajectorySHead",
    "_input_fast",
    "_loss_e1",
    "_evaluate_e1",
    "_fixed_metrics_e1",
    "_seed_everything",
    "device_e1",
    "OOF_ROOT",
}
missing_v2 = sorted(REQUIRED_V2.difference(globals()))
if missing_v2:
    raise RuntimeError(
        "Run this in the E1 live kernel after the loss diagnostic; missing "
        + repr(missing_v2)
    )


EXPERIMENT_V2 = "deltacorr_steerspline_e1_pca16_capacity_v2"
OUT_DIR_V2 = OOF_ROOT / EXPERIMENT_V2
MODEL_DIR_V2 = OUT_DIR_V2 / "models"
for directory in (OUT_DIR_V2, MODEL_DIR_V2):
    directory.mkdir(parents=True, exist_ok=True)

SEED_V2 = 20260727
WARMUP_CYCLES_V2 = 300
PATH_CYCLES_V2 = 300
REPORT_CYCLES_V2 = 20
CAPACITY_GATE_V2 = 1.0
SCENARIOS_V2 = (("one_well", 1), ("four_wells", 4), ("twentyfour_wells", 24))


def _json_v2(path, payload):
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


def _target_standard_v2(record, device):
    coefficients = torch.from_numpy(
        target_cache_e1[record.well]["coefficients"]
    ).to(device)
    center = torch.from_numpy(coefficient_center_e1).to(device)
    scale = torch.from_numpy(coefficient_scale_e1).to(device)
    return (coefficients - center) / scale


def _train_scenario_v2(model, scenario_records, device, seed):
    rng = np.random.default_rng(seed)
    mean_rows = float(np.mean([record.eval_rows for record in scenario_records]))
    history, best = [], None
    started = time.time()
    total_updates = 0

    phases = (
        {
            "name": "coefficient_warmup",
            "cycles": WARMUP_CYCLES_V2,
            "lr": 8e-4,
            "min_lr": 1.5e-4,
            "clip": 10.0,
        },
        {
            "name": "exact_path",
            "cycles": PATH_CYCLES_V2,
            "lr": 3e-4,
            "min_lr": 3e-5,
            "clip": 5.0,
        },
    )

    for phase in phases:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=phase["lr"], weight_decay=1e-5
        )
        phase_updates = phase["cycles"] * len(scenario_records)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase_updates, eta_min=phase["min_lr"]
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda", init_scale=512.0
        )

        for cycle in range(1, phase["cycles"] + 1):
            model.train()
            loss_sum = coefficient_sum = path_sum = 0.0
            skipped = 0
            for index in rng.permutation(len(scenario_records)):
                record = scenario_records[int(index)]
                optimizer.zero_grad(set_to_none=True)
                values = _input_fast(record, device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    standard = model(values)
                    target_standard = _target_standard_v2(record, device)
                    coefficient_loss = torch.mean(
                        (standard[0] - target_standard) ** 2
                    )
                    if phase["name"] == "coefficient_warmup":
                        loss = coefficient_loss
                        path_loss = torch.zeros((), device=device)
                    else:
                        path_objective, path_loss, _ = _loss_e1(
                            standard,
                            record,
                            device,
                            record.eval_rows / max(mean_rows, 1.0),
                        )
                        # _loss_e1 already includes a light coefficient term.
                        loss = path_objective
                before = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), phase["clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                skipped += int(float(scaler.get_scale()) < before)
                total_updates += 1
                loss_sum += float(loss.detach().cpu())
                coefficient_sum += float(coefficient_loss.detach().cpu())
                path_sum += float(path_loss.detach().cpu())

            should_report = (
                cycle == 1
                or cycle % REPORT_CYCLES_V2 == 0
                or cycle == phase["cycles"]
            )
            if not should_report:
                continue
            metrics, bywell = _evaluate_e1(model, scenario_records, device)
            row = {
                "phase": phase["name"],
                "cycle": cycle,
                "updates": total_updates,
                "loss": loss_sum / len(scenario_records),
                "coefficient_loss": coefficient_sum / len(scenario_records),
                "path_mse_scaled": path_sum / len(scenario_records),
                **metrics,
                "skipped_steps": skipped,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_minutes": (time.time() - started) / 60.0,
            }
            history.append(row)
            print("E1 PCA16 CAPACITY V2", row, flush=True)
            if best is None or metrics["rmse"] < best["metrics"]["rmse"]:
                best = {
                    "phase": phase["name"],
                    "cycle": cycle,
                    "updates": total_updates,
                    "metrics": metrics,
                    "bywell": bywell.copy(),
                    "state": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                }

            # Complete at least 40 warm-up cycles.  A passing warm-up receives
            # 20 exact-path cycles before final acceptance.
            if (
                phase["name"] == "coefficient_warmup"
                and cycle >= 40
                and metrics["rmse"] <= CAPACITY_GATE_V2
            ):
                break
            if (
                phase["name"] == "exact_path"
                and cycle >= 20
                and best["metrics"]["rmse"] <= CAPACITY_GATE_V2
            ):
                break

    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history), (time.time() - started) / 60.0


started_v2 = time.time()
results_v2, histories_v2 = [], []
parameters_v2 = None

print(
    "E1 PCA16 CAPACITY V2 CONFIG",
    {
        "pca_dimensions": PCA_DIM_E1,
        "warmup_updates_per_well": WARMUP_CYCLES_V2,
        "path_updates_per_well": PATH_CYCLES_V2,
        "absolute_gate_ft": CAPACITY_GATE_V2,
    },
    flush=True,
)

for scenario_index, (scenario, well_count) in enumerate(SCENARIOS_V2):
    indices = np.linspace(0, len(pool_e1) - 1, well_count).round().astype(int)
    scenario_records = [pool_e1[index] for index in indices]
    oracle = _fixed_metrics_e1(
        scenario_records, np.zeros(PCA_DIM_E1), oracle=True
    )
    baseline = _fixed_metrics_e1(
        scenario_records, coefficient_center_e1, oracle=False
    )
    if oracle["rmse"] >= CAPACITY_GATE_V2:
        result = {
            "scenario": scenario,
            "wells": well_count,
            "baseline": baseline,
            "representation_oracle": oracle,
            "gate_rmse": CAPACITY_GATE_V2,
            "pass": False,
            "failure": "PCA16 oracle makes the absolute gate impossible",
        }
        results_v2.append(result)
        print("E1 PCA16 CAPACITY V2 RESULT", result, flush=True)
        break

    _seed_everything(SEED_V2 + scenario_index)
    model = FastTrajectorySHead(PCA_DIM_E1).to(device_e1)
    parameters_v2 = int(sum(parameter.numel() for parameter in model.parameters()))
    print(
        "E1 PCA16 CAPACITY V2 START",
        {
            "scenario": scenario,
            "wells": well_count,
            "parameters": parameters_v2,
            "baseline": baseline,
            "representation_oracle": oracle,
        },
        flush=True,
    )
    model, best, history, runtime = _train_scenario_v2(
        model, scenario_records, device_e1, SEED_V2 + scenario_index
    )
    history.insert(0, "scenario", scenario)
    histories_v2.append(history)
    history.to_csv(OUT_DIR_V2 / f"history_{scenario}.csv", index=False)
    best["bywell"].to_csv(OUT_DIR_V2 / f"bywell_{scenario}.csv", index=False)
    checkpoint = MODEL_DIR_V2 / f"best_{scenario}.pt"
    torch.save(
        {
            "experiment": EXPERIMENT_V2,
            "scenario": scenario,
            "phase": best["phase"],
            "cycle": int(best["cycle"]),
            "updates": int(best["updates"]),
            "model": best["state"],
            "metrics": best["metrics"],
            "pca_dimensions": PCA_DIM_E1,
            "coefficient_center": coefficient_center_e1,
            "coefficient_scale": coefficient_scale_e1,
        },
        checkpoint,
    )
    passed = bool(best["metrics"]["rmse"] <= CAPACITY_GATE_V2)
    result = {
        "scenario": scenario,
        "wells": well_count,
        "baseline": baseline,
        "representation_oracle": oracle,
        "gate_rmse": CAPACITY_GATE_V2,
        "best_phase": best["phase"],
        "best_cycle": int(best["cycle"]),
        "best_updates": int(best["updates"]),
        "result": best["metrics"],
        "runtime_minutes": runtime,
        "checkpoint": str(checkpoint),
        "pass": passed,
    }
    results_v2.append(result)
    print("E1 PCA16 CAPACITY V2 RESULT", result, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if not passed:
        break

strict_pass_v2 = bool(
    len(results_v2) == len(SCENARIOS_V2)
    and all(result["pass"] for result in results_v2)
)
verdict_v2 = {
    "experiment": EXPERIMENT_V2,
    "stage": "E1_PCA16_corrected_capacity",
    "pca_dimensions": PCA_DIM_E1,
    "parameters": parameters_v2,
    "warmup_updates_per_well": WARMUP_CYCLES_V2,
    "path_updates_per_well": PATH_CYCLES_V2,
    "capacity_gate_rmse_le": CAPACITY_GATE_V2,
    "capacity_results": results_v2,
    "strict_pass": strict_pass_v2,
    "elapsed_minutes": (time.time() - started_v2) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> train held-out PCA16 E1 with the frozen two-stage schedule"
        if strict_pass_v2
        else "FAIL -> stop deterministic PCA SteerSpline before DeltaCorr"
    ),
}
_json_v2(OUT_DIR_V2 / "verdict.json", verdict_v2)

if histories_v2:
    print("\nE1 PCA16 CAPACITY V2 HISTORY")
    print(pd.concat(histories_v2, ignore_index=True).to_string(index=False))
print("\nE1 PCA16 CAPACITY V2 FINAL VERDICT")
print(json.dumps(verdict_v2, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_V2, flush=True)
