"""Fold-2 gateway for a standalone complete-well surface posterior model.

This addon is appended to the definition prefix of
native_resolution_dualref_capacity_v0_cell.py by the local build script.  The
resulting merged cell is self-contained on AutoDL and consumes only raw train
horizontal/typewell CSV files plus fold-clean well metadata.

The native dual-reference network produces a 257-state TVT posterior at every
latent MD column.  A bidirectional complete-well decoder then consumes a
low-rank view of the entire posterior, geometry and prefix channels.  It is
trained jointly on absolute TVT offset, structural surface (TVT + Z), local
slope, and 32/128/512-ft displacement targets.
"""

from pathlib import Path
import gc
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERIMENT_FWSP = "native_fullwell_surface_posterior_fold2_v0"
OUT_DIR_FWSP = OOF_ROOT / EXPERIMENT_FWSP
MODEL_DIR_FWSP = OUT_DIR_FWSP / "models"
OUT_DIR_FWSP.mkdir(parents=True, exist_ok=True)
MODEL_DIR_FWSP.mkdir(parents=True, exist_ok=True)

SEED_FWSP = 20260724
TRAIN_FOLDS_FWSP = (3, 4, 5)
VALID_FOLD_FWSP = 2
MAX_EPOCHS_FWSP = 36
MIN_EPOCHS_FWSP = 10
REPORT_EVERY_FWSP = 2
PATIENCE_REPORTS_FWSP = 5
LEARNING_RATE_FWSP = 2.5e-4
MIN_LEARNING_RATE_FWSP = 3.0e-5
WEIGHT_DECAY_FWSP = 3.0e-5
GRADIENT_CLIP_FWSP = 3.0
POSTERIOR_CHANNELS_FWSP = 40
GLOBAL_HIDDEN_FWSP = 96
GLOBAL_LAYERS_FWSP = 2
GLOBAL_DROPOUT_FWSP = 0.12
PATH_SCALE_FWSP = 24.0
TARGET_SIGMA_FWSP = 1.5
HORIZONS_FWSP = (8, 32, 128)  # latent columns = about 32/128/512 ft
LOSS_CE_FWSP = 0.20
LOSS_PATH_FWSP = 1.00
LOSS_SURFACE_SLOPE_FWSP = 0.20
LOSS_SURFACE_CURVATURE_FWSP = 0.04
LOSS_HORIZON_FWSP = 0.22
ARCHITECTURE_GATE_FWSP = 8.0
TARGET_GATE_FWSP = 5.5

FOLD_CANDIDATES_FWSP = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    ROOT / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    DATA_ROOT / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
)


def _write_json_fwsp(path, payload):
    def default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


def _load_folds_fwsp(wells):
    fold_path = next((path for path in FOLD_CANDIDATES_FWSP if path.exists()), None)
    if fold_path is None:
        raise FileNotFoundError(
            "Missing fold-clean well_metadata.csv; run foldclean_full_fleongg_lgb_v0 first"
        )
    frame = pd.read_csv(fold_path)
    well_column = next((c for c in ("well", "well_id", "Well") if c in frame), None)
    fold_column = next((c for c in ("fold", "Fold") if c in frame), None)
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Cannot identify well/fold columns in {fold_path}")
    frame = frame[[well_column, fold_column]].rename(
        columns={well_column: "well", fold_column: "fold"}
    )
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.drop_duplicates("well")
    missing = sorted(set(wells).difference(frame["well"]))
    if missing:
        raise RuntimeError(f"Fold metadata misses {len(missing)} wells: {missing[:5]}")
    return frame.loc[frame["well"].isin(wells)].copy(), fold_path


def _ensure_records_fwsp(wells):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for number, well in enumerate(wells, 1):
        path = _record_path(well)
        try:
            if not path.exists():
                _save_record(_build_record(well))
            record = _load_record(well)
            reports.append(
                {
                    "well": well,
                    "status": "ok",
                    "eval_rows": record.eval_rows,
                    "teacher_rmse": record.teacher_rmse,
                    "axis_coverage": record.axis_coverage,
                }
            )
        except Exception as exc:
            reports.append({"well": well, "status": "failed", "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"build/load native records {number}/{len(wells)}", flush=True)
    report = pd.DataFrame(reports)
    report.to_csv(OUT_DIR_FWSP / "record_build_report.csv", index=False)
    failed = report.loc[report["status"].eq("failed")]
    if len(failed):
        raise RuntimeError(
            f"{len(failed)} native records failed: "
            + repr(failed.head(5).to_dict(orient="records"))
        )
    return report


class FullWellPosteriorSurfaceNetFWSP(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = NativeDualReferenceWarpNet()
        self.posterior_projection = nn.Sequential(
            nn.Conv1d(N_STATES, POSTERIOR_CHANNELS_FWSP, 1),
            nn.GELU(),
            nn.Conv1d(POSTERIOR_CHANNELS_FWSP, POSTERIOR_CHANNELS_FWSP, 3, padding=1),
            nn.GELU(),
        )
        global_inputs = POSTERIOR_CHANNELS_FWSP + 10 + 5
        self.global_input = nn.Sequential(
            nn.Conv1d(global_inputs, GLOBAL_HIDDEN_FWSP, 5, padding=2),
            nn.GroupNorm(8, GLOBAL_HIDDEN_FWSP),
            nn.GELU(),
        )
        self.global_gru = nn.GRU(
            GLOBAL_HIDDEN_FWSP,
            GLOBAL_HIDDEN_FWSP,
            num_layers=GLOBAL_LAYERS_FWSP,
            batch_first=True,
            bidirectional=True,
            dropout=GLOBAL_DROPOUT_FWSP,
        )
        self.path_head = nn.Sequential(
            nn.Linear(2 * GLOBAL_HIDDEN_FWSP, GLOBAL_HIDDEN_FWSP),
            nn.GELU(),
            nn.Dropout(GLOBAL_DROPOUT_FWSP),
            nn.Linear(GLOBAL_HIDDEN_FWSP, 1),
        )
        self.mix_logit = nn.Parameter(torch.tensor(-0.5))

    def forward(self, horizontal, condition, typewell, heel):
        logits = self.local(horizontal, condition, typewell, heel)
        probability = torch.softmax(logits, dim=1)
        states = torch.as_tensor(
            STATE_OFFSETS, device=logits.device, dtype=logits.dtype
        )[None, :, None]
        local_mean = torch.sum(probability * states, dim=1)
        local_var = torch.sum(probability * (states - local_mean[:, None]) ** 2, dim=1)
        local_std = torch.sqrt(local_var.clamp_min(1e-5))
        local_entropy = -torch.sum(
            probability * torch.log(probability.clamp_min(1e-8)), dim=1
        ) / np.log(N_STATES)
        local_peak = probability.amax(dim=1)
        local_margin = torch.topk(probability, 2, dim=1).values
        local_margin = local_margin[:, 0] - local_margin[:, 1]

        posterior = self.posterior_projection(probability)
        columns = posterior.shape[-1]
        condition_latent = condition[..., ::HORIZONTAL_STRIDE][..., :columns]
        summary = torch.stack(
            [
                local_mean / STATE_HALF_WIDTH_FT,
                local_std / STATE_HALF_WIDTH_FT,
                local_entropy,
                local_peak,
                local_margin,
            ],
            dim=1,
        )
        features = torch.cat([posterior, condition_latent, summary], dim=1)
        encoded = self.global_input(features).transpose(1, 2)
        encoded, _ = self.global_gru(encoded)
        global_offset = STATE_HALF_WIDTH_FT * torch.tanh(
            self.path_head(encoded).squeeze(-1)
        )
        mix = torch.sigmoid(self.mix_logit)
        path = mix * global_offset + (1.0 - mix) * local_mean
        return logits, path, {
            "local_mean": local_mean,
            "global_offset": global_offset,
            "mix": mix,
            "entropy": local_entropy,
        }


def _tensor_record_fwsp(record, device, mode="full", shuffled_typewell=None):
    horizontal = record.horizontal.copy()
    typewell = record.typewell.copy()
    heel = record.heel.copy()
    if mode == "no_gr":
        horizontal[[0, 2, 3]] = 0.0
        typewell[[0, 2, 3]] = 0.0
        heel[[0, 2, 3]] = 0.0
    elif mode == "shuffled_typewell":
        if shuffled_typewell is None:
            raise ValueError("shuffled_typewell record is required")
        typewell = shuffled_typewell.typewell.copy()
    return (
        torch.from_numpy(horizontal)[None].to(device),
        torch.from_numpy(record.condition)[None].to(device),
        torch.from_numpy(typewell)[None].to(device),
        torch.from_numpy(heel)[None].to(device),
    )


def _soft_targets_fwsp(target, device):
    states = torch.as_tensor(STATE_OFFSETS, device=device, dtype=target.dtype)[None]
    distribution = torch.exp(-0.5 * ((states - target[:, None]) / TARGET_SIGMA_FWSP) ** 2)
    return distribution / distribution.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _record_targets_fwsp(record, columns, device):
    target = torch.from_numpy(record.latent_truth_offset[:columns]).to(device)
    mask = torch.from_numpy(record.latent_eval_mask[:columns]).to(device)
    latent_index = record.latent_index[:columns]
    z_relative = torch.from_numpy(record.condition[4, latent_index]).to(device) * 512.0
    surface = target + z_relative
    return target, surface, mask


def _loss_fwsp(logits, path, record, device):
    columns = min(logits.shape[-1], path.shape[-1])
    logits = logits[..., :columns]
    path = path[..., :columns]
    target, surface_target, mask = _record_targets_fwsp(record, columns, device)
    valid = mask & torch.isfinite(target) & (torch.abs(target) <= STATE_HALF_WIDTH_FT)
    if int(valid.sum()) < 32:
        raise RuntimeError(f"{record.well}: too few supervised columns")

    soft = _soft_targets_fwsp(target[valid], device)
    ce = -(soft * torch.log_softmax(logits[0, :, valid].T, dim=1)).sum(dim=1).mean()
    path_error = (path[0, valid] - target[valid]) / PATH_SCALE_FWSP
    path_mse = path_error.square().mean()

    z_relative = surface_target - target
    surface_prediction = path[0] + z_relative
    supervised_indices = torch.nonzero(valid, as_tuple=False).flatten()
    pred_surface = surface_prediction[supervised_indices]
    true_surface = surface_target[supervised_indices]
    slope = F.smooth_l1_loss(
        torch.diff(pred_surface) / 4.0,
        torch.diff(true_surface) / 4.0,
        beta=0.25,
    )
    if len(pred_surface) > 2:
        curvature = F.smooth_l1_loss(
            torch.diff(pred_surface, n=2) / 2.0,
            torch.diff(true_surface, n=2) / 2.0,
            beta=0.20,
        )
    else:
        curvature = path_mse.new_tensor(0.0)

    horizon_losses = []
    for horizon in HORIZONS_FWSP:
        if columns <= horizon:
            continue
        pair_mask = valid[horizon:] & valid[:-horizon]
        if int(pair_mask.sum()) < 8:
            continue
        pred_delta = surface_prediction[horizon:] - surface_prediction[:-horizon]
        true_delta = surface_target[horizon:] - surface_target[:-horizon]
        scale = max(float(np.sqrt(horizon) * 4.0), 8.0)
        horizon_losses.append(
            F.smooth_l1_loss(
                pred_delta[pair_mask] / scale,
                true_delta[pair_mask] / scale,
                beta=0.25,
            )
        )
    horizon_loss = (
        torch.stack(horizon_losses).mean()
        if horizon_losses
        else path_mse.new_tensor(0.0)
    )
    total = (
        LOSS_CE_FWSP * ce
        + LOSS_PATH_FWSP * path_mse
        + LOSS_SURFACE_SLOPE_FWSP * slope
        + LOSS_SURFACE_CURVATURE_FWSP * curvature
        + LOSS_HORIZON_FWSP * horizon_loss
    )
    return total, {
        "ce": ce.detach(),
        "path_mse": path_mse.detach(),
        "slope": slope.detach(),
        "curvature": curvature.detach(),
        "horizon": horizon_loss.detach(),
    }


def _decode_record_fwsp(record, latent_path):
    columns = min(len(record.latent_index), len(latent_path))
    latent_md = record.md[record.latent_index[:columns]]
    latent_tvt = record.anchor_tvt + np.asarray(latent_path[:columns], np.float64)
    support_md = np.r_[record.md[record.anchor_index], latent_md]
    support_tvt = np.r_[record.anchor_tvt, latent_tvt]
    order = np.argsort(support_md, kind="mergesort")
    support_md = support_md[order]
    support_tvt = support_tvt[order]
    unique, indices = np.unique(support_md, return_index=True)
    return np.interp(record.md[record.eval_index], unique, support_tvt[indices])


def _metrics_fwsp(rows):
    truth = rows["truth"].to_numpy(np.float64)
    prediction = rows["prediction"].to_numpy(np.float64)
    error = prediction - truth
    bywell = rows.assign(error2=error**2).groupby("well", sort=False)["error2"].mean().pow(0.5)
    output = {
        "rows": int(len(rows)),
        "wells": int(rows["well"].nunique()),
        "exact_sse": float(np.sum(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell.median()),
        "p75_well_rmse": float(bywell.quantile(0.75)),
        "p95_well_rmse": float(bywell.quantile(0.95)),
        "max_well_rmse": float(bywell.max()),
    }
    bands = (
        ("near_0_500_rmse", 0.0, 500.0),
        ("mid_500_1000_rmse", 500.0, 1000.0),
        ("mid_1000_2000_rmse", 1000.0, 2000.0),
        ("far_gt2000_rmse", 2000.0, np.inf),
    )
    distance = rows["md_since"].to_numpy(np.float64)
    for name, low, high in bands:
        selected = (distance >= low) & (distance < high)
        output[name] = (
            float(np.sqrt(np.mean(error[selected] ** 2))) if selected.any() else float("nan")
        )
    return output, bywell.reset_index(name="rmse")


@torch.no_grad()
def _evaluate_fwsp(model, records, device, mode="full", save_rows=False):
    model.eval()
    rows = []
    shuffled = list(reversed(records))
    entropies = []
    mixes = []
    for number, record in enumerate(records):
        other = shuffled[number]
        inputs = _tensor_record_fwsp(record, device, mode, other)
        logits, path, diagnostics = model(*inputs)
        prediction = _decode_record_fwsp(record, path[0].cpu().numpy())
        md_since = record.md[record.eval_index] - record.md[record.anchor_index]
        rows.append(
            pd.DataFrame(
                {
                    "well": record.well,
                    "row_idx": record.eval_index,
                    "truth": record.truth[record.eval_index],
                    "prediction": prediction,
                    "md_since": md_since,
                }
            )
        )
        entropies.append(float(diagnostics["entropy"].mean().cpu()))
        mixes.append(float(diagnostics["mix"].cpu()))
    frame = pd.concat(rows, ignore_index=True)
    metrics, bywell = _metrics_fwsp(frame)
    metrics["mean_entropy"] = float(np.mean(entropies))
    metrics["mean_global_mix"] = float(np.mean(mixes))
    return metrics, bywell, frame if save_rows else None


def _train_fwsp(train_records, valid_records, device):
    _seed_everything(SEED_FWSP)
    model = FullWellPosteriorSurfaceNetFWSP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_FWSP,
        weight_decay=WEIGHT_DECAY_FWSP,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS_FWSP, eta_min=MIN_LEARNING_RATE_FWSP
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=512.0)
    rng = np.random.default_rng(SEED_FWSP)
    best = None
    bad_reports = 0
    history = []
    started = time.time()
    for epoch in range(1, MAX_EPOCHS_FWSP + 1):
        model.train()
        order = rng.permutation(len(train_records))
        sums = {key: 0.0 for key in ("loss", "ce", "path_mse", "slope", "curvature", "horizon")}
        skipped = 0
        for step, index in enumerate(order, 1):
            record = train_records[int(index)]
            inputs = _tensor_record_fwsp(record, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits, path, _ = model(*inputs)
                loss, parts = _loss_fwsp(logits, path, record, device)
            before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_FWSP)
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) < before:
                skipped += 1
            sums["loss"] += float(loss.detach().cpu())
            for key in parts:
                sums[key] += float(parts[key].cpu())
            if step % 100 == 0 or step == len(order):
                print(
                    f"FWSP epoch {epoch}/{MAX_EPOCHS_FWSP} wells {step}/{len(order)} "
                    f"loss={sums['loss']/step:.4f}",
                    flush=True,
                )
            del inputs, logits, path, loss
        scheduler.step()

        if epoch not in (1, 2, MAX_EPOCHS_FWSP) and epoch % REPORT_EVERY_FWSP:
            continue
        valid_metrics, _, _ = _evaluate_fwsp(model, valid_records, device)
        report = {
            "epoch": epoch,
            **{f"train_{key}": value / len(order) for key, value in sums.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "skipped_steps": skipped,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(report)
        print("FWSP EPOCH", report, flush=True)
        if best is None or valid_metrics["rmse"] < best["metrics"]["rmse"] - 1e-4:
            best = {
                "epoch": epoch,
                "metrics": valid_metrics,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
            torch.save(
                {"model": best["state"], "epoch": epoch, "metrics": valid_metrics},
                MODEL_DIR_FWSP / "best_fold2.pt",
            )
            bad_reports = 0
        else:
            bad_reports += 1
        pd.DataFrame(history).to_csv(OUT_DIR_FWSP / "training_history.csv", index=False)
        if epoch >= MIN_EPOCHS_FWSP and bad_reports >= PATIENCE_REPORTS_FWSP:
            print(f"FWSP early stopping: best epoch {best['epoch']}", flush=True)
            break
    if best is None:
        raise RuntimeError("No FWSP checkpoint was produced")
    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history), (time.time() - started) / 60.0


STARTED_FWSP = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("This experiment requires a CUDA GPU")
DEVICE_FWSP = torch.device("cuda")
_seed_everything(SEED_FWSP)

wells_fwsp = [path.name.split("__", 1)[0] for path in _horizontal_paths()]
folds_fwsp, fold_path_fwsp = _load_folds_fwsp(wells_fwsp)
build_report_fwsp = _ensure_records_fwsp(wells_fwsp)
train_wells_fwsp = folds_fwsp.loc[
    folds_fwsp["fold"].isin(TRAIN_FOLDS_FWSP), "well"
].tolist()
valid_wells_fwsp = folds_fwsp.loc[
    folds_fwsp["fold"].eq(VALID_FOLD_FWSP), "well"
].tolist()
if len(train_wells_fwsp) < 460 or len(valid_wells_fwsp) < 154:
    raise RuntimeError(
        f"Unexpected Fold345/Fold2 sizes: {len(train_wells_fwsp)}/{len(valid_wells_fwsp)}"
    )
train_records_fwsp = [_load_record(well) for well in train_wells_fwsp]
valid_records_fwsp = [_load_record(well) for well in valid_wells_fwsp]

guards_fwsp = {
    "data_root": str(DATA_ROOT),
    "fold_source": str(fold_path_fwsp),
    "train_folds": list(TRAIN_FOLDS_FWSP),
    "validation_fold": VALID_FOLD_FWSP,
    "train_wells": len(train_records_fwsp),
    "valid_wells": len(valid_records_fwsp),
    "train_rows": int(sum(record.eval_rows for record in train_records_fwsp)),
    "valid_rows": int(sum(record.eval_rows for record in valid_records_fwsp)),
    "minimum_axis_coverage": float(build_report_fwsp["axis_coverage"].min()),
    "maximum_teacher_rmse": float(build_report_fwsp["teacher_rmse"].max()),
    "uses_historical_prediction": False,
    "uses_neighbor_spatial_cp12_gold": False,
    "sample_unit": "one complete physical well",
}
print("FWSP CONFIG", json.dumps(guards_fwsp, indent=2), flush=True)

model_fwsp, best_fwsp, history_fwsp, training_minutes_fwsp = _train_fwsp(
    train_records_fwsp, valid_records_fwsp, DEVICE_FWSP
)
full_fwsp, bywell_fwsp, rows_fwsp = _evaluate_fwsp(
    model_fwsp, valid_records_fwsp, DEVICE_FWSP, save_rows=True
)
rows_fwsp.to_parquet(OUT_DIR_FWSP / "fold2_oof.parquet", index=False)
bywell_fwsp.sort_values("rmse", ascending=False).to_csv(
    OUT_DIR_FWSP / "fold2_bywell.csv", index=False
)

controls_fwsp = []
for mode_fwsp in ("full", "no_gr", "shuffled_typewell"):
    metrics_fwsp, _, _ = _evaluate_fwsp(
        model_fwsp, valid_records_fwsp, DEVICE_FWSP, mode=mode_fwsp
    )
    controls_fwsp.append(
        {
            "mode": mode_fwsp,
            **metrics_fwsp,
            "degradation_vs_full": metrics_fwsp["rmse"] - full_fwsp["rmse"],
        }
    )
controls_frame_fwsp = pd.DataFrame(controls_fwsp)
controls_frame_fwsp.to_csv(OUT_DIR_FWSP / "causal_controls.csv", index=False)
no_gr_drop_fwsp = float(
    controls_frame_fwsp.loc[controls_frame_fwsp["mode"].eq("no_gr"), "degradation_vs_full"].iloc[0]
)
shuffle_drop_fwsp = float(
    controls_frame_fwsp.loc[
        controls_frame_fwsp["mode"].eq("shuffled_typewell"), "degradation_vs_full"
    ].iloc[0]
)

checks_fwsp = {
    "architecture_rmse_le_8": full_fwsp["rmse"] <= ARCHITECTURE_GATE_FWSP,
    "target_rmse_le_5p5": full_fwsp["rmse"] <= TARGET_GATE_FWSP,
    "far_rmse_le_10": full_fwsp["far_gt2000_rmse"] <= 10.0,
    "no_gr_degradation_ge_0p5": no_gr_drop_fwsp >= 0.5,
    "shuffled_typewell_degradation_ge_0p5": shuffle_drop_fwsp >= 0.5,
}
strict_pass_fwsp = bool(
    checks_fwsp["architecture_rmse_le_8"]
    and checks_fwsp["no_gr_degradation_ge_0p5"]
    and checks_fwsp["shuffled_typewell_degradation_ge_0p5"]
)
verdict_fwsp = {
    "experiment": EXPERIMENT_FWSP,
    "stage": "fold2_native_costmap_completewell_surface_gateway",
    "architecture": "native dual-reference 257-state posterior + bidirectional whole-well decoder",
    "legal_inputs_only": True,
    "guards": guards_fwsp,
    "parameters": int(sum(parameter.numel() for parameter in model_fwsp.parameters())),
    "selected_epoch": int(best_fwsp["epoch"]),
    "fold2": full_fwsp,
    "controls": {
        "no_gr_degradation": no_gr_drop_fwsp,
        "shuffled_typewell_degradation": shuffle_drop_fwsp,
    },
    "checks": checks_fwsp,
    "strict_pass": strict_pass_fwsp,
    "training_minutes": training_minutes_fwsp,
    "elapsed_minutes": (time.time() - STARTED_FWSP) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "TARGET PASS -> repeat on all five folds"
        if checks_fwsp["target_rmse_le_5p5"] and strict_pass_fwsp
        else (
            "ARCHITECTURE PASS -> inspect worst wells and add official-mask augmentation"
            if strict_pass_fwsp
            else "FAIL -> posterior ordering still does not transfer; stop this evidence family"
        )
    ),
}
_write_json_fwsp(OUT_DIR_FWSP / "verdict.json", verdict_fwsp)
print("\nFWSP CONTROLS")
print(controls_frame_fwsp.to_string(index=False), flush=True)
print("\nFWSP FINAL VERDICT")
print(json.dumps(verdict_fwsp, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_FWSP, flush=True)
