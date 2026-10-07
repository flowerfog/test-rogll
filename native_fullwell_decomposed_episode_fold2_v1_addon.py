import gc
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERIMENT_DCE = "native_fullwell_decomposed_episode_fold2_v1"
OUT_DIR_DCE = OOF_ROOT / EXPERIMENT_DCE
MODEL_DIR_DCE = OUT_DIR_DCE / "models"
OUT_DIR_DCE.mkdir(parents=True, exist_ok=True)
MODEL_DIR_DCE.mkdir(parents=True, exist_ok=True)

SEED_DCE = 20260724
TRAIN_FOLDS_DCE = (3, 4, 5)
VALID_FOLD_DCE = 2
MAX_EPOCHS_DCE = 24
MIN_EPOCHS_DCE = 8
REPORT_EVERY_DCE = 2
PATIENCE_REPORTS_DCE = 4
FAIL_FAST_EPOCH_DCE = 8
FAIL_FAST_RMSE_DCE = 11.5
EPISODES_PER_WELL_DCE = 2
OFFICIAL_EPISODE_PROBABILITY_DCE = 0.30
MAX_PSEUDO_ANCHOR_SHIFT_FT_DCE = 40.0
MIN_HIDDEN_ROWS_DCE = 256

LEARNING_RATE_DCE = 2.0e-4
MIN_LEARNING_RATE_DCE = 2.0e-5
WEIGHT_DECAY_DCE = 4.0e-5
GRADIENT_CLIP_DCE = 3.0
MAP_CHANNELS_DCE = 16
GLOBAL_HIDDEN_DCE = 96
GLOBAL_LAYERS_DCE = 2
GLOBAL_DROPOUT_DCE = 0.12
TARGET_SIGMA_DCE = 1.25
PATH_SCALE_DCE = 24.0
ALIAS_OFFSETS_DCE = (12, 24, 48)
HORIZONS_DCE = (8, 32, 128)

LOSS_CE_DCE = 0.70
LOSS_PATH_DCE = 0.55
LOSS_ALIAS_DCE = 0.20
LOSS_HORIZON_DCE = 0.14
ARCHITECTURE_GATE_DCE = 8.0
TARGET_GATE_DCE = 5.5

FOLD_CANDIDATES_DCE = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    ROOT / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    DATA_ROOT / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
)


def _json_dce(path, payload):
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


def _load_folds_dce(wells):
    path = next((candidate for candidate in FOLD_CANDIDATES_DCE if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("Missing fold-clean well_metadata.csv")
    frame = pd.read_csv(path)
    well_column = next((c for c in ("well", "well_id", "Well") if c in frame), None)
    fold_column = next((c for c in ("fold", "Fold") if c in frame), None)
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Cannot identify well/fold columns in {path}")
    frame = frame[[well_column, fold_column]].rename(
        columns={well_column: "well", fold_column: "fold"}
    )
    frame["well"] = frame["well"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.drop_duplicates("well")
    missing = sorted(set(wells).difference(frame["well"]))
    if missing:
        raise RuntimeError(f"Fold metadata misses {len(missing)} wells: {missing[:5]}")
    return frame.loc[frame["well"].isin(wells)].copy(), path


def _ensure_records_dce(wells):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for number, well in enumerate(wells, 1):
        try:
            if not _record_path(well).exists():
                _save_record(_build_record(well))
            record = _load_record(well)
            reports.append(
                {
                    "well": well,
                    "status": "ok",
                    "eval_rows": record.eval_rows,
                    "axis_coverage": record.axis_coverage,
                    "teacher_rmse": record.teacher_rmse,
                }
            )
        except Exception as exc:
            reports.append({"well": well, "status": "failed", "reason": repr(exc)})
        if number % 50 == 0 or number == len(wells):
            print(f"build/load DCE records {number}/{len(wells)}", flush=True)
    report = pd.DataFrame(reports)
    report.to_csv(OUT_DIR_DCE / "record_build_report.csv", index=False)
    failed = report.loc[report["status"].eq("failed")]
    if len(failed):
        raise RuntimeError(
            f"{len(failed)} DCE records failed: "
            + repr(failed.head(5).to_dict(orient="records"))
        )
    return report


def _shift_reference_dce(features, old_anchor_tvt, new_anchor_tvt):
    old_axis = old_anchor_tvt + REFERENCE_OFFSETS
    new_axis = new_anchor_tvt + REFERENCE_OFFSETS
    output = np.zeros_like(features, dtype=np.float32)
    for channel in range(features.shape[0]):
        output[channel] = np.interp(
            new_axis,
            old_axis,
            features[channel],
            left=0.0,
            right=0.0,
        )
    output[1] = (output[1] >= 0.999).astype(np.float32)
    return output


def _episode_dce(record, rng=None, official=False):
    if official or rng is None or rng.random() < OFFICIAL_EPISODE_PROBABILITY_DCE:
        return record

    original = record.anchor_index
    candidates = np.arange(original, len(record.md) - MIN_HIDDEN_ROWS_DCE, dtype=np.int32)
    if not len(candidates):
        return record
    truth = record.truth
    valid = np.isfinite(truth[candidates])
    shift = np.abs(truth[candidates] - record.anchor_tvt)
    candidates = candidates[valid & (shift <= MAX_PSEUDO_ANCHOR_SHIFT_FT_DCE)]
    if len(candidates) < 8:
        return record

    # Sample uniformly in progress rather than by row density.
    quantile = float(rng.uniform(0.0, 1.0))
    anchor_index = int(candidates[min(int(quantile * len(candidates)), len(candidates) - 1)])
    if anchor_index == original:
        return record
    anchor_tvt = float(truth[anchor_index])

    condition = record.condition.copy()
    known_mask = np.arange(len(record.md)) <= anchor_index
    condition[0] = known_mask.astype(np.float32)
    condition[1] = 0.0
    finite_known = known_mask & np.isfinite(truth)
    condition[1, finite_known] = np.clip(
        (truth[finite_known] - anchor_tvt) / STATE_HALF_WIDTH_FT, -2.0, 2.0
    )
    denominator = max(float(record.md[-1] - record.md[anchor_index]), 1.0)
    condition[3] = np.clip(
        (record.md - record.md[anchor_index]) / denominator, -1.0, 1.0
    )
    z_relative = record.condition[4].astype(np.float64)
    condition[4] = np.clip(z_relative - z_relative[anchor_index], -4.0, 4.0)

    typewell = _shift_reference_dce(
        record.typewell, record.anchor_tvt, anchor_tvt
    )
    reference_tvt = anchor_tvt + REFERENCE_OFFSETS
    known = np.flatnonzero(finite_known)
    heel_values, heel_valid = _heel_reference(
        truth[known], record.horizontal[0, known], reference_tvt
    )
    heel = _reference_features(heel_values, heel_valid)

    latent_truth = truth[record.latent_index] - anchor_tvt
    latent_eval_mask = (
        (record.latent_index > anchor_index)
        & np.isfinite(latent_truth)
        & (np.abs(latent_truth) <= STATE_HALF_WIDTH_FT)
    )
    eval_index = np.flatnonzero(
        (np.arange(len(record.md)) > anchor_index) & np.isfinite(truth)
    )
    if int(latent_eval_mask.sum()) < 64 or len(eval_index) < MIN_HIDDEN_ROWS_DCE:
        return record

    return Record(
        well=record.well,
        horizontal=record.horizontal,
        condition=condition.astype(np.float32),
        typewell=typewell.astype(np.float32),
        heel=heel.astype(np.float32),
        latent_index=record.latent_index,
        latent_truth_offset=latent_truth.astype(np.float32),
        latent_eval_mask=latent_eval_mask,
        md=record.md,
        eval_index=eval_index.astype(np.int32),
        truth=record.truth,
        anchor_tvt=anchor_tvt,
        anchor_index=anchor_index,
        eval_rows=len(eval_index),
        calibration_a=record.calibration_a,
        calibration_b=record.calibration_b,
        calibration_pairs=record.calibration_pairs,
        calibration_rmse=record.calibration_rmse,
        teacher_rmse=record.teacher_rmse,
        axis_coverage=float(np.mean(np.abs(truth[eval_index] - anchor_tvt) <= STATE_HALF_WIDTH_FT)),
    )


def _smooth_same_dce(values, kernel):
    kernel = int(kernel)
    if kernel <= 1:
        return values
    if kernel % 2 == 0:
        kernel += 1
    return F.avg_pool1d(
        F.pad(values, (kernel // 2, kernel // 2), mode="replicate"),
        kernel,
        stride=1,
    )


class DecomposedEpisodeCostNetDCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = NativeDualReferenceWarpNet()
        # Ten evidence maps: learned logits/probability plus four scales for
        # typewell and heel. State-axis striding retains mode location.
        self.map_encoder = nn.Sequential(
            nn.Conv2d(10, MAP_CHANNELS_DCE, (9, 5), stride=(4, 1), padding=(4, 2)),
            nn.GroupNorm(4, MAP_CHANNELS_DCE),
            nn.GELU(),
            nn.Conv2d(
                MAP_CHANNELS_DCE,
                MAP_CHANNELS_DCE,
                (7, 3),
                stride=(4, 1),
                padding=(3, 1),
            ),
            nn.GroupNorm(4, MAP_CHANNELS_DCE),
            nn.GELU(),
        )
        encoded_states = (N_STATES + 15) // 16
        global_inputs = MAP_CHANNELS_DCE * encoded_states + 10 + 5
        self.global_input = nn.Sequential(
            nn.Conv1d(global_inputs, GLOBAL_HIDDEN_DCE, 5, padding=2),
            nn.GroupNorm(8, GLOBAL_HIDDEN_DCE),
            nn.GELU(),
        )
        self.global_gru = nn.GRU(
            GLOBAL_HIDDEN_DCE,
            GLOBAL_HIDDEN_DCE,
            num_layers=GLOBAL_LAYERS_DCE,
            batch_first=True,
            bidirectional=True,
            dropout=GLOBAL_DROPOUT_DCE,
        )
        self.logit_head = nn.Sequential(
            nn.Linear(2 * GLOBAL_HIDDEN_DCE, GLOBAL_HIDDEN_DCE),
            nn.GELU(),
            nn.Dropout(GLOBAL_DROPOUT_DCE),
            nn.Linear(GLOBAL_HIDDEN_DCE, N_STATES),
        )
        self.correction_scale = nn.Parameter(torch.tensor(-1.0))

    def _direct_maps(self, horizontal, typewell, heel, columns):
        query = horizontal[:, 0:1, ::HORIZONTAL_STRIDE][..., :columns]
        state_slice = slice(REFERENCE_STATE_START, REFERENCE_STATE_START + N_STATES)
        type_value = typewell[:, 0:1, state_slice]
        heel_value = heel[:, 0:1, state_slice]
        maps = []
        # Comparable physical windows: query spacing is about 4 ft, references 1 ft.
        for query_kernel, reference_kernel in ((1, 1), (3, 9), (9, 33), (31, 125)):
            q = _smooth_same_dce(query, query_kernel).unsqueeze(2)
            t = _smooth_same_dce(type_value, reference_kernel).unsqueeze(-1)
            h = _smooth_same_dce(heel_value, reference_kernel).unsqueeze(-1)
            maps.append(-torch.log1p(((q - t) / 1.5).square()))
            maps.append(-torch.log1p(((q - h) / 1.5).square()))
        return maps

    def forward(self, horizontal, condition, typewell, heel):
        local_logits = self.local(horizontal, condition, typewell, heel)
        columns = local_logits.shape[-1]
        probability = torch.softmax(local_logits, dim=1)
        centred_logits = local_logits - local_logits.mean(dim=1, keepdim=True)
        scaled_logits = centred_logits / centred_logits.std(
            dim=1, keepdim=True
        ).clamp_min(0.2)
        evidence = [scaled_logits[:, None], probability[:, None]]
        evidence.extend(self._direct_maps(horizontal, typewell, heel, columns))
        maps = torch.cat(evidence, dim=1)
        encoded_map = self.map_encoder(maps)
        encoded_map = encoded_map.flatten(1, 2)

        states = torch.as_tensor(
            STATE_OFFSETS, device=local_logits.device, dtype=local_logits.dtype
        )[None, :, None]
        local_mean = torch.sum(probability * states, dim=1)
        local_var = torch.sum(
            probability * (states - local_mean[:, None]) ** 2, dim=1
        )
        local_std = torch.sqrt(local_var.clamp_min(1e-5))
        entropy = -torch.sum(
            probability * torch.log(probability.clamp_min(1e-8)), dim=1
        ) / np.log(N_STATES)
        peak = probability.amax(dim=1)
        top2 = torch.topk(probability, 2, dim=1).values
        summary = torch.stack(
            [
                local_mean / STATE_HALF_WIDTH_FT,
                local_std / STATE_HALF_WIDTH_FT,
                entropy,
                peak,
                top2[:, 0] - top2[:, 1],
            ],
            dim=1,
        )
        condition_latent = condition[..., ::HORIZONTAL_STRIDE][..., :columns]
        features = torch.cat([encoded_map, condition_latent, summary], dim=1)
        sequence = self.global_input(features).transpose(1, 2)
        sequence, _ = self.global_gru(sequence)
        correction = self.logit_head(sequence).transpose(1, 2)
        dose = F.softplus(self.correction_scale)
        logits = local_logits + dose * correction
        final_probability = torch.softmax(logits, dim=1)
        path = torch.sum(final_probability * states, dim=1)
        return logits, path, {
            "local_logits": local_logits,
            "entropy": entropy,
            "dose": dose,
        }


def _tensor_dce(record, device, mode="full", shuffled=None):
    horizontal = record.horizontal.copy()
    typewell = record.typewell.copy()
    heel = record.heel.copy()
    if mode == "no_gr":
        horizontal[[0, 2, 3]] = 0.0
        typewell[[0, 2, 3]] = 0.0
        heel[[0, 2, 3]] = 0.0
    elif mode == "shuffled_typewell":
        if shuffled is None:
            raise ValueError("shuffled record is required")
        typewell = shuffled.typewell.copy()
    return (
        torch.from_numpy(horizontal)[None].to(device),
        torch.from_numpy(record.condition)[None].to(device),
        torch.from_numpy(typewell)[None].to(device),
        torch.from_numpy(heel)[None].to(device),
    )


def _targets_dce(record, columns, device):
    target = torch.from_numpy(record.latent_truth_offset[:columns]).to(device)
    valid = torch.from_numpy(record.latent_eval_mask[:columns]).to(device)
    valid = valid & torch.isfinite(target) & (torch.abs(target) <= STATE_HALF_WIDTH_FT)
    return target, valid


def _loss_dce(logits, path, record, device):
    columns = min(logits.shape[-1], path.shape[-1])
    logits = logits[..., :columns]
    path = path[..., :columns]
    target, valid = _targets_dce(record, columns, device)
    if int(valid.sum()) < 32:
        raise RuntimeError(f"{record.well}: too few supervised columns")
    states = torch.as_tensor(STATE_OFFSETS, device=device, dtype=path.dtype)[None]
    soft = torch.exp(-0.5 * ((states - target[valid, None]) / TARGET_SIGMA_DCE) ** 2)
    soft = soft / soft.sum(dim=1, keepdim=True).clamp_min(1e-8)
    selected_logits = logits[0, :, valid].T
    ce = -(soft * torch.log_softmax(selected_logits, dim=1)).sum(dim=1).mean()
    path_mse = ((path[0, valid] - target[valid]) / PATH_SCALE_DCE).square().mean()

    truth_index = torch.round(target[valid] - float(STATE_OFFSETS[0])).long()
    row = torch.arange(len(truth_index), device=device)
    truth_score = selected_logits[row, truth_index]
    alias_terms = []
    for offset in ALIAS_OFFSETS_DCE:
        for sign in (-1, 1):
            alias_index = truth_index + sign * int(offset)
            keep = (alias_index >= 0) & (alias_index < N_STATES)
            if int(keep.sum()):
                alias_score = selected_logits[row[keep], alias_index[keep]]
                alias_terms.append(F.relu(1.0 - truth_score[keep] + alias_score).mean())
    alias = torch.stack(alias_terms).mean() if alias_terms else ce.new_tensor(0.0)

    horizon_terms = []
    for horizon in HORIZONS_DCE:
        if columns <= horizon:
            continue
        pair = valid[horizon:] & valid[:-horizon]
        if int(pair.sum()) < 8:
            continue
        predicted_delta = path[0, horizon:] - path[0, :-horizon]
        true_delta = target[horizon:] - target[:-horizon]
        scale = max(float(np.sqrt(horizon) * 4.0), 8.0)
        horizon_terms.append(
            F.smooth_l1_loss(
                predicted_delta[pair] / scale,
                true_delta[pair] / scale,
                beta=0.25,
            )
        )
    horizon_loss = torch.stack(horizon_terms).mean() if horizon_terms else ce.new_tensor(0.0)
    total = (
        LOSS_CE_DCE * ce
        + LOSS_PATH_DCE * path_mse
        + LOSS_ALIAS_DCE * alias
        + LOSS_HORIZON_DCE * horizon_loss
    )
    return total, {
        "ce": ce.detach(),
        "path_mse": path_mse.detach(),
        "alias": alias.detach(),
        "horizon": horizon_loss.detach(),
    }


def _decode_dce(record, latent_path):
    columns = min(len(record.latent_index), len(latent_path))
    support_md = np.r_[record.md[record.anchor_index], record.md[record.latent_index[:columns]]]
    support_tvt = np.r_[record.anchor_tvt, record.anchor_tvt + np.asarray(latent_path[:columns])]
    order = np.argsort(support_md, kind="mergesort")
    support_md = support_md[order]
    support_tvt = support_tvt[order]
    unique, index = np.unique(support_md, return_index=True)
    return np.interp(record.md[record.eval_index], unique, support_tvt[index])


def _metrics_dce(frame):
    error = frame["prediction"].to_numpy(np.float64) - frame["truth"].to_numpy(np.float64)
    bywell = frame.assign(error2=error**2).groupby("well", sort=False)["error2"].mean().pow(0.5)
    output = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.sum(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell.median()),
        "p75_well_rmse": float(bywell.quantile(0.75)),
        "p95_well_rmse": float(bywell.quantile(0.95)),
        "max_well_rmse": float(bywell.max()),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    for name, mask in (
        ("near_0_500_rmse", distance <= 500),
        ("mid_500_1000_rmse", (distance > 500) & (distance <= 1000)),
        ("mid_1000_2000_rmse", (distance > 1000) & (distance <= 2000)),
        ("far_gt2000_rmse", distance > 2000),
    ):
        output[name] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return output, bywell


@torch.no_grad()
def _evaluate_dce(model, records, device, mode="full", save_rows=False):
    model.eval()
    rows = []
    entropy = []
    dose = []
    for number, record in enumerate(records, 1):
        shuffled = records[(number * 37) % len(records)] if mode == "shuffled_typewell" else None
        inputs = _tensor_dce(record, device, mode=mode, shuffled=shuffled)
        logits, path, diagnostics = model(*inputs)
        prediction = _decode_dce(record, path[0].float().cpu().numpy())
        index = record.eval_index
        rows.append(
            pd.DataFrame(
                {
                    "well": record.well,
                    "row_idx": index,
                    "truth": record.truth[index],
                    "prediction": prediction,
                    "md_since": record.md[index] - record.md[record.anchor_index],
                }
            )
        )
        probability = torch.softmax(logits, dim=1)
        entropy.append(float((-(probability * probability.clamp_min(1e-8).log()).sum(1) / np.log(N_STATES)).mean().cpu()))
        dose.append(float(diagnostics["dose"].cpu()))
        if number % 50 == 0 or number == len(records):
            print(f"DCE evaluate {mode} {number}/{len(records)}", flush=True)
    frame = pd.concat(rows, ignore_index=True)
    metrics, bywell = _metrics_dce(frame)
    metrics["mean_entropy"] = float(np.mean(entropy))
    metrics["mean_correction_dose"] = float(np.mean(dose))
    bywell_frame = bywell.rename("rmse").reset_index()
    return metrics, bywell_frame, frame if save_rows else None


def _train_dce(train_records, valid_records, device):
    _seed_everything(SEED_DCE)
    model = DecomposedEpisodeCostNetDCE().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_DCE, weight_decay=WEIGHT_DECAY_DCE
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS_DCE, eta_min=MIN_LEARNING_RATE_DCE
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=512.0)
    rng = np.random.default_rng(SEED_DCE)
    best = None
    bad_reports = 0
    history = []
    started = time.time()
    for epoch in range(1, MAX_EPOCHS_DCE + 1):
        model.train()
        jobs = np.repeat(np.arange(len(train_records)), EPISODES_PER_WELL_DCE)
        rng.shuffle(jobs)
        sums = {key: 0.0 for key in ("loss", "ce", "path_mse", "alias", "horizon")}
        skipped = 0
        for step, index in enumerate(jobs, 1):
            episode = _episode_dce(train_records[int(index)], rng=rng)
            inputs = _tensor_dce(episode, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits, path, _ = model(*inputs)
                loss, parts = _loss_dce(logits, path, episode, device)
            before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_DCE)
            scaler.step(optimizer)
            scaler.update()
            skipped += int(float(scaler.get_scale()) < before)
            sums["loss"] += float(loss.detach().cpu())
            for key, value in parts.items():
                sums[key] += float(value.cpu())
            if step % 100 == 0 or step == len(jobs):
                print(
                    f"DCE epoch {epoch}/{MAX_EPOCHS_DCE} episodes {step}/{len(jobs)} "
                    f"loss={sums['loss']/step:.4f}",
                    flush=True,
                )
            del inputs, logits, path, loss
        scheduler.step()
        if epoch not in (1, 2, MAX_EPOCHS_DCE) and epoch % REPORT_EVERY_DCE:
            continue
        valid_metrics, _, _ = _evaluate_dce(model, valid_records, device)
        report = {
            "epoch": epoch,
            **{f"train_{key}": value / len(jobs) for key, value in sums.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "skipped_steps": skipped,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(report)
        print("DCE EPOCH", report, flush=True)
        if best is None or valid_metrics["rmse"] < best["metrics"]["rmse"] - 1e-4:
            best = {
                "epoch": epoch,
                "metrics": valid_metrics,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            torch.save(
                {"model": best["state"], "epoch": epoch, "metrics": valid_metrics},
                MODEL_DIR_DCE / "best_fold2.pt",
            )
            bad_reports = 0
        else:
            bad_reports += 1
        pd.DataFrame(history).to_csv(OUT_DIR_DCE / "training_history.csv", index=False)
        if epoch >= FAIL_FAST_EPOCH_DCE and best["metrics"]["rmse"] > FAIL_FAST_RMSE_DCE:
            print(
                f"DCE fail-fast: best Fold2 RMSE {best['metrics']['rmse']:.4f} "
                f"> {FAIL_FAST_RMSE_DCE:.1f} at epoch {epoch}",
                flush=True,
            )
            break
        if epoch >= MIN_EPOCHS_DCE and bad_reports >= PATIENCE_REPORTS_DCE:
            print(f"DCE early stopping: best epoch {best['epoch']}", flush=True)
            break
    if best is None:
        raise RuntimeError("No DCE checkpoint was produced")
    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history), (time.time() - started) / 60.0


STARTED_DCE = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("This experiment requires a CUDA GPU")
DEVICE_DCE = torch.device("cuda")
_seed_everything(SEED_DCE)

wells_dce = [path.name.split("__", 1)[0] for path in _horizontal_paths()]
folds_dce, fold_path_dce = _load_folds_dce(wells_dce)
build_report_dce = _ensure_records_dce(wells_dce)
train_wells_dce = folds_dce.loc[folds_dce["fold"].isin(TRAIN_FOLDS_DCE), "well"].tolist()
valid_wells_dce = folds_dce.loc[folds_dce["fold"].eq(VALID_FOLD_DCE), "well"].tolist()
if len(train_wells_dce) < 460 or len(valid_wells_dce) < 154:
    raise RuntimeError(
        f"Unexpected Fold345/Fold2 sizes: {len(train_wells_dce)}/{len(valid_wells_dce)}"
    )
train_records_dce = [_load_record(well) for well in train_wells_dce]
valid_records_dce = [_load_record(well) for well in valid_wells_dce]

guards_dce = {
    "data_root": str(DATA_ROOT),
    "fold_source": str(fold_path_dce),
    "train_folds": list(TRAIN_FOLDS_DCE),
    "validation_fold": VALID_FOLD_DCE,
    "train_wells": len(train_records_dce),
    "valid_wells": len(valid_records_dce),
    "episodes_per_well_per_epoch": EPISODES_PER_WELL_DCE,
    "official_episode_probability": OFFICIAL_EPISODE_PROBABILITY_DCE,
    "maximum_pseudo_anchor_shift_ft": MAX_PSEUDO_ANCHOR_SHIFT_FT_DCE,
    "fail_fast": {"epoch": FAIL_FAST_EPOCH_DCE, "rmse_gt": FAIL_FAST_RMSE_DCE},
    "uses_train_only_formation_columns": False,
    "uses_historical_prediction": False,
    "uses_neighbor_spatial_cp12_gold": False,
    "sample_unit": "complete physical well with randomized legal prefix episodes",
}
print("DCE CONFIG", json.dumps(guards_dce, indent=2), flush=True)

model_dce, best_dce, history_dce, training_minutes_dce = _train_dce(
    train_records_dce, valid_records_dce, DEVICE_DCE
)
full_dce, bywell_dce, rows_dce = _evaluate_dce(
    model_dce, valid_records_dce, DEVICE_DCE, save_rows=True
)
rows_dce.to_parquet(OUT_DIR_DCE / "fold2_oof.parquet", index=False)
bywell_dce.sort_values("rmse", ascending=False).to_csv(
    OUT_DIR_DCE / "fold2_bywell.csv", index=False
)

control_rows_dce = []
for mode_dce in ("full", "no_gr", "shuffled_typewell"):
    metrics_dce, _, _ = _evaluate_dce(
        model_dce, valid_records_dce, DEVICE_DCE, mode=mode_dce
    )
    control_rows_dce.append(
        {
            "mode": mode_dce,
            **metrics_dce,
            "degradation_vs_full": metrics_dce["rmse"] - full_dce["rmse"],
        }
    )
controls_dce = pd.DataFrame(control_rows_dce)
controls_dce.to_csv(OUT_DIR_DCE / "causal_controls.csv", index=False)
no_gr_drop_dce = float(
    controls_dce.loc[controls_dce["mode"].eq("no_gr"), "degradation_vs_full"].iloc[0]
)
shuffle_drop_dce = float(
    controls_dce.loc[
        controls_dce["mode"].eq("shuffled_typewell"), "degradation_vs_full"
    ].iloc[0]
)
checks_dce = {
    "architecture_rmse_le_8": full_dce["rmse"] <= ARCHITECTURE_GATE_DCE,
    "target_rmse_le_5p5": full_dce["rmse"] <= TARGET_GATE_DCE,
    "far_rmse_le_10": full_dce["far_gt2000_rmse"] <= 10.0,
    "no_gr_degradation_ge_0p5": no_gr_drop_dce >= 0.5,
    "shuffled_typewell_degradation_ge_0p5": shuffle_drop_dce >= 0.5,
}
strict_pass_dce = bool(
    checks_dce["architecture_rmse_le_8"]
    and checks_dce["no_gr_degradation_ge_0p5"]
    and checks_dce["shuffled_typewell_degradation_ge_0p5"]
)
verdict_dce = {
    "experiment": EXPERIMENT_DCE,
    "stage": "fold2_decomposed_costmap_random_episode_gateway",
    "architecture": "native 257-state decomposed multiscale maps plus whole-well categorical decoder",
    "legal_inputs_only": True,
    "guards": guards_dce,
    "parameters": int(sum(parameter.numel() for parameter in model_dce.parameters())),
    "selected_epoch": int(best_dce["epoch"]),
    "fold2": full_dce,
    "controls": {
        "no_gr_degradation": no_gr_drop_dce,
        "shuffled_typewell_degradation": shuffle_drop_dce,
    },
    "checks": checks_dce,
    "strict_pass": strict_pass_dce,
    "training_minutes": training_minutes_dce,
    "elapsed_minutes": (time.time() - STARTED_DCE) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "TARGET PASS -> repeat all five folds"
        if checks_dce["target_rmse_le_5p5"] and strict_pass_dce
        else (
            "ARCHITECTURE PASS -> add structured transition decoding"
            if strict_pass_dce
            else "FAIL -> decomposed candidate evidence still does not transfer"
        )
    ),
}
_json_dce(OUT_DIR_DCE / "verdict.json", verdict_dce)
print("\nDCE CONTROLS")
print(controls_dce.to_string(index=False), flush=True)
print("\nDCE FINAL VERDICT")
print(json.dumps(verdict_dce, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_DCE, flush=True)
