"""Dense per-column SSE energy model for the label-free full-well path bank.

Why this experiment exists
--------------------------
The 3216-path bank has a 3.91-ft Fold-2 oracle, but the previous unary model
reduced each complete candidate to one scalar and learned from only one
well-level ranking target.  This gateway keeps the exact same label-free bank
and full-well GR alignment, but predicts a dense error-energy trace for every
candidate column.  Each physical well therefore contributes hundreds of
thousands of supervised candidate/column comparisons while remaining one
complete-well sample.

Protocol
--------
* Folds 3/4/5 train one model; Fold 2 is the only gateway fold.
* Fold 1 is never loaded.
* Training candidates are selected without truth: base-family coverage,
  heuristic GR rank and deterministic random coverage.
* Truth appears only in the dense squared-error loss and frozen metrics.
* No neighbour, spatial strong, CP12, Gold, or leaderboard signal is used.

Run after the V0 candidate gateway and V1 full-bank retriever sources/checkpoint
have been uploaded to /root/autodl-tmp.  The V1 execution tail is not rerun.
"""

from __future__ import annotations

from pathlib import Path
import gc
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Restore the frozen candidate generator and shared metric utilities only.
# ---------------------------------------------------------------------------


SOURCE_CANDIDATES_V2 = (
    Path("/root/autodl-tmp/candidate_conditioned_fullbank_retriever_v1_cell.py"),
    Path.cwd() / "candidate_conditioned_fullbank_retriever_v1_cell.py",
)
SOURCE_PATH_V2 = next((path for path in SOURCE_CANDIDATES_V2 if path.exists()), None)
if SOURCE_PATH_V2 is None:
    raise FileNotFoundError(
        "Upload candidate_conditioned_fullbank_retriever_v1_cell.py and its "
        "V0 dependency to /root/autodl-tmp before running V2."
    )
source_v2 = SOURCE_PATH_V2.read_text(encoding="utf-8")
marker_v2 = "# Execute\n"
if marker_v2 not in source_v2:
    raise RuntimeError(f"Cannot locate V1 execution marker in {SOURCE_PATH_V2}")
exec(
    compile(source_v2.split(marker_v2, 1)[0], str(SOURCE_PATH_V2), "exec"),
    globals(),
)

REQUIRED_V2 = {
    "OOF_ROOT",
    "OUT_DIR_V1",
    "PANEL_DIR",
    "BASE_MENU",
    "SEQUENCE_LENGTH",
    "MIN_REFERENCE_COVERAGE",
    "TRAIN_FOLDS",
    "VALID_FOLD",
    "STRATA_NAMES",
    "_load_folds",
    "_well_payload",
    "_render_full_bank_v1",
    "_load_panel",
    "_row_for_index_v1",
    "_summarize_rows_v1",
    "_seed_everything",
    "ResidualBlockV1",
}
missing_v2 = sorted(REQUIRED_V2.difference(globals()))
if missing_v2:
    raise RuntimeError("V1 bootstrap missing: " + repr(missing_v2))
print("DENSE-SSE bootstrap:", SOURCE_PATH_V2, flush=True)


# ---------------------------------------------------------------------------
# Frozen gateway configuration.
# ---------------------------------------------------------------------------


EXPERIMENT_V2 = "candidate_dense_sse_energy_gateway_v2"
OUT_DIR_V2 = Path(OOF_ROOT) / EXPERIMENT_V2
TRAIN_CACHE_V2 = OUT_DIR_V2 / "train_samples"
MODEL_DIR_V2 = OUT_DIR_V2 / "models"
for directory_v2 in (OUT_DIR_V2, TRAIN_CACHE_V2, MODEL_DIR_V2):
    directory_v2.mkdir(parents=True, exist_ok=True)

SEED_V2 = 20260801
TRAIN_CANDIDATES_V2 = 192
INFERENCE_CHUNK_V2 = 192
MAX_EPOCHS_V2 = 8
MIN_EPOCHS_V2 = 3
PATIENCE_V2 = 3
LEARNING_RATE_V2 = 1.5e-4
MIN_LEARNING_RATE_V2 = 2.0e-5
WEIGHT_DECAY_V2 = 4.0e-5
GRAD_ACCUM_WELLS_V2 = 2
GRADIENT_CLIP_V2 = 4.0

CHANNELS_V2 = 48
DILATIONS_V2 = (1, 2, 4, 8, 16, 32, 64)
ERROR_CAP_FT_V2 = 128.0
SOFT_TARGET_TEMPERATURE_FT_V2 = 1.5
HARD_GAP_FT_V2 = 4.0
LOGIT_MARGIN_V2 = 0.50
LOSS_DENSE_V2 = 0.70
LOSS_SOFT_RANK_V2 = 0.30
LOSS_MARGIN_V2 = 0.10

GATE_FULL_MODEL_RMSE_V2 = 8.0
GATE_TOP32_ORACLE_RMSE_V2 = 5.20
GATE_TOP8_ORACLE_RMSE_V2 = 5.75
GATE_MEDIAN_RANK_V2 = 8.0
GATE_CONTROL_DEGRADATION_V2 = 1.0


def _json_default_v2(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json_v2(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default_v2), encoding="utf-8"
    )


def _sample_path_v2(well):
    return TRAIN_CACHE_V2 / f"{well}.npz"


def _truth_sequence_v2(well):
    payload = _well_payload(well)
    sequence_md = np.linspace(
        payload["eval_md"][0], payload["eval_md"][-1], SEQUENCE_LENGTH
    )
    truth = np.interp(sequence_md, payload["eval_md"], payload["truth"])
    return truth.astype(np.float32), float(payload["anchor_tvt"])


def _label_free_train_indices_v2(bank, well):
    deployable = np.flatnonzero(np.isfinite(bank["candidate_mse"]))
    if len(deployable) < TRAIN_CANDIDATES_V2:
        raise RuntimeError(f"only {len(deployable)} deployable candidates")
    selected = []
    seen = set()

    def add(values):
        for value in values:
            value = int(value)
            if value in seen or value not in deployable_set:
                continue
            seen.add(value)
            selected.append(value)
            if len(selected) >= TRAIN_CANDIDATES_V2:
                return

    deployable_set = set(int(value) for value in deployable)
    # V0's percentile heuristic is lower-is-better.
    add(deployable[np.argsort(bank["heuristic"][deployable], kind="mergesort")[:48]])
    # Every base family receives four deterministic shape proposals.
    for base_index in range(len(BASE_MENU)):
        members = deployable[bank["candidate_ids"][deployable, 0] == base_index]
        if not len(members):
            continue
        ordered = members[np.argsort(bank["heuristic"][members], kind="mergesort")]
        add(ordered[:2])
        add(members[np.linspace(0, len(members) - 1, 2, dtype=np.int32)])
    rng = np.random.default_rng(SEED_V2 + int(str(well), 16) % 1000003)
    add(rng.permutation(deployable))
    output = np.asarray(selected[:TRAIN_CANDIDATES_V2], np.int32)
    if len(output) != TRAIN_CANDIDATES_V2:
        raise RuntimeError(f"selected only {len(output)} candidates")
    return output


def _build_sample_v2(well):
    bank = _render_full_bank_v1(well)
    selected = _label_free_train_indices_v2(bank, well)
    truth_sequence, anchor = _truth_sequence_v2(well)
    np.savez_compressed(
        _sample_path_v2(well),
        well=np.asarray(str(well)),
        query=bank["query"].astype(np.float16),
        reference=bank["reference"][selected].astype(np.float16),
        valid=bank["valid"][selected].astype(np.uint8),
        path_delta=bank["path_delta"][selected].astype(np.float16),
        heuristic=bank["heuristic"][selected].astype(np.float16),
        truth_sequence=truth_sequence.astype(np.float32),
        anchor=np.asarray([anchor], np.float32),
        candidate_mse=bank["candidate_mse"][selected].astype(np.float32),
        candidate_ids=bank["candidate_ids"][selected].astype(np.int16),
        eval_rows=np.asarray([bank["eval_rows"]], np.int32),
        cache_version=np.asarray([2], np.int16),
    )
    return {
        "well": well,
        "status": "ok",
        "candidates": len(selected),
        "sample_oracle_rmse": float(np.sqrt(bank["candidate_mse"][selected].min())),
    }


def _load_sample_v2(well):
    with np.load(_sample_path_v2(well), allow_pickle=False) as data:
        return {
            "well": str(data["well"].item()),
            "query": data["query"].astype(np.float32),
            "reference": data["reference"].astype(np.float32),
            "valid": data["valid"].astype(bool),
            "path_delta": data["path_delta"].astype(np.float32),
            "heuristic": data["heuristic"].astype(np.float32),
            "truth_sequence": data["truth_sequence"].astype(np.float32),
            "anchor": float(data["anchor"][0]),
            "candidate_mse": data["candidate_mse"].astype(np.float32),
            "eval_rows": int(data["eval_rows"][0]),
        }


# ---------------------------------------------------------------------------
# Complete-well dense error-energy network.
# ---------------------------------------------------------------------------


class DenseSSEEnergyV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(11, CHANNELS_V2, 7, padding=3)
        self.blocks = nn.ModuleList(
            [ResidualBlockV1(CHANNELS_V2, dilation) for dilation in DILATIONS_V2]
        )
        self.head = nn.Sequential(
            nn.Conv1d(CHANNELS_V2 * 2 + 4, CHANNELS_V2, 1),
            nn.GELU(),
            nn.Conv1d(CHANNELS_V2, 1, 1),
        )

    def forward(self, query, reference, valid, path_delta, heuristic):
        candidates, length = reference.shape
        q = query[None, :].expand(candidates, -1)
        difference = (q - reference) * valid
        absolute = difference.abs()
        q_derivative = torch.diff(q, dim=1, prepend=q[:, :1])
        r_derivative = torch.diff(reference, dim=1, prepend=reference[:, :1])
        path = torch.clamp(path_delta / 96.0, -8.0, 8.0)
        path_derivative = torch.diff(path, dim=1, prepend=path[:, :1]) * 16.0
        progress = torch.linspace(
            0.0, 1.0, length, device=query.device, dtype=query.dtype
        )[None, :].expand(candidates, -1)
        heuristic_sequence = heuristic[:, None].expand(-1, length)
        inputs = torch.stack(
            [
                q,
                reference,
                difference,
                absolute,
                q_derivative,
                r_derivative,
                path,
                path_derivative,
                valid,
                progress,
                heuristic_sequence,
            ],
            dim=1,
        )
        hidden = self.stem(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        mask = valid[:, None, :]
        count = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        global_mean = (hidden * mask).sum(dim=2, keepdim=True) / count
        global_mean = global_mean.expand(-1, -1, length)
        summaries = torch.stack([absolute, path, path_derivative, valid], dim=1)
        descriptor = torch.cat([hidden, global_mean, summaries], dim=1)
        # log(1 + squared error); softplus keeps inferred SSE non-negative.
        return F.softplus(self.head(descriptor)[:, 0, :])


def _tensors_v2(query, reference, valid, path_delta, heuristic, device):
    return (
        torch.from_numpy(np.asarray(query, np.float32)).to(device),
        torch.from_numpy(np.asarray(reference, np.float32)).to(device),
        torch.from_numpy(np.asarray(valid, np.float32)).to(device),
        torch.from_numpy(np.asarray(path_delta, np.float32)).to(device),
        torch.from_numpy(np.asarray(heuristic, np.float32)).to(device),
    )


def _loss_v2(model, panel, device):
    permutation = np.random.permutation(len(panel["candidate_mse"]))
    tensors = _tensors_v2(
        panel["query"],
        panel["reference"][permutation],
        panel["valid"][permutation],
        panel["path_delta"][permutation],
        panel["heuristic"][permutation],
        device,
    )
    predicted_log = model(*tensors).float()
    path = torch.from_numpy(panel["path_delta"][permutation]).to(device)
    truth = torch.from_numpy(panel["truth_sequence"] - panel["anchor"]).to(device)
    error2 = (path - truth[None, :]).square()
    capped_error2 = error2.clamp_max(ERROR_CAP_FT_V2**2)
    target_log = torch.log1p(capped_error2)
    dense = F.smooth_l1_loss(predicted_log, target_log)

    predicted_mse = torch.expm1(predicted_log.clamp_max(10.0)).mean(dim=1)
    true_mse = error2.mean(dim=1)
    true_rmse = torch.sqrt(true_mse + 1e-8)
    soft_target = torch.softmax(
        -(true_rmse - true_rmse.min()) / SOFT_TARGET_TEMPERATURE_FT_V2, dim=0
    )
    scores = -predicted_mse
    soft_rank = -(soft_target * F.log_softmax(scores, dim=0)).sum()
    target = torch.argmin(true_mse)
    hard = true_rmse >= true_rmse[target] + HARD_GAP_FT_V2
    margin = (
        F.relu(LOGIT_MARGIN_V2 - scores[target] + scores[hard].amax())
        if hard.any()
        else scores[target] * 0.0
    )
    loss = (
        LOSS_DENSE_V2 * dense
        + LOSS_SOFT_RANK_V2 * soft_rank
        + LOSS_MARGIN_V2 * margin
    )
    return loss, {
        "dense": float(dense.detach().cpu()),
        "soft_rank": float(soft_rank.detach().cpu()),
        "margin": float(margin.detach().cpu()),
    }


@torch.no_grad()
def _score_arrays_v2(model, query, reference, valid, path_delta, heuristic, device, control="full"):
    model.eval()
    output = []
    for left in range(0, len(reference), INFERENCE_CHUNK_V2):
        right = min(left + INFERENCE_CHUNK_V2, len(reference))
        chunk_query = query.copy()
        chunk_reference = reference[left:right].copy()
        chunk_valid = valid[left:right].copy()
        chunk_heuristic = heuristic[left:right].copy()
        if control == "no_gr":
            chunk_query.fill(0.0)
            chunk_heuristic.fill(0.5)
        elif control == "shuffled_alignment":
            indices = (np.arange(left, right) + 37) % len(reference)
            chunk_reference = reference[indices]
            chunk_valid = valid[indices]
            chunk_heuristic = heuristic[indices]
        elif control != "full":
            raise ValueError(control)
        predicted_log = model(
            *_tensors_v2(
                chunk_query,
                chunk_reference,
                chunk_valid,
                path_delta[left:right],
                chunk_heuristic,
                device,
            )
        ).float()
        predicted_mse = torch.expm1(predicted_log.clamp_max(10.0)).mean(dim=1)
        output.append(predicted_mse.detach().cpu().numpy())
    return np.concatenate(output)


def _evaluate_pruned_v2(model, wells, device):
    rows = []
    for well in wells:
        panel = _load_panel(well)
        predicted = _score_arrays_v2(
            model,
            panel["query"],
            panel["reference"],
            panel["valid"],
            panel["path_delta"],
            panel["heuristic"],
            device,
        )
        index = int(np.argmin(predicted))
        row = {
            "well": well,
            "rows": int(panel["eval_rows"]),
            "mse": float(panel["candidate_mse"][index]),
            "mae": float(panel["candidate_mae"][index]),
        }
        for stratum, name in enumerate(STRATA_NAMES):
            row[f"{name}_rows"] = int(panel["strata_counts"][stratum])
            row[f"{name}_mse"] = float(panel["strata_mse"][index, stratum])
        rows.append(row)
    frame = pd.DataFrame(rows)
    return _summarize_rows_v1(frame), frame


def _evaluate_full_v2(model, wells, device):
    modes = ("model", "no_gr", "shuffled_alignment", "top8_oracle", "top32_oracle", "top192_oracle")
    rows = {mode: [] for mode in modes}
    rank_rows = []
    started = time.time()
    for number, well in enumerate(wells, start=1):
        bank = _render_full_bank_v1(well)
        deployable = np.asarray(bank["deployable"], dtype=bool)
        deployable_indices = np.flatnonzero(deployable)
        if len(deployable_indices) < 192:
            raise RuntimeError(
                f"{well} has only {len(deployable_indices)} deployable full-bank candidates"
            )
        if not np.isfinite(bank["candidate_mse"][deployable]).all():
            raise RuntimeError(f"{well} has non-finite deployable candidate MSE values")

        predicted = {}
        for control in ("full", "no_gr", "shuffled_alignment"):
            scores = _score_arrays_v2(
                model,
                bank["query"],
                bank["reference"],
                bank["valid"],
                bank["path_delta"],
                bank["heuristic"],
                device,
                control=control,
            )
            scores = np.asarray(scores, dtype=np.float64)
            scores[~np.isfinite(scores)] = np.inf
            scores[~deployable] = np.inf
            if not np.isfinite(scores[deployable]).any():
                raise RuntimeError(f"{well}/{control} has no finite deployable score")
            predicted[control] = scores
        for control in ("full", "no_gr", "shuffled_alignment"):
            index = int(np.argmin(predicted[control]))
            rows["model" if control == "full" else control].append(
                _row_for_index_v1(well, bank, index, control)
            )
        order = deployable_indices[
            np.argsort(predicted["full"][deployable_indices], kind="mergesort")
        ]
        best_truth = int(
            deployable_indices[np.argmin(bank["candidate_mse"][deployable_indices])]
        )
        rank = int(np.flatnonzero(order == best_truth)[0] + 1)
        rank_rows.append({"well": well, "truth_candidate_rank": rank})
        for top_k in (8, 32, 192):
            active = order[:top_k]
            index = int(active[np.argmin(bank["candidate_mse"][active])])
            rows[f"top{top_k}_oracle"].append(
                _row_for_index_v1(well, bank, index, f"top{top_k}_oracle")
            )
        del bank, predicted
        gc.collect()
        if number % 10 == 0 or number == len(wells):
            print(
                f"DENSE-SSE fullbank {number}/{len(wells)} "
                f"elapsed={(time.time() - started) / 60.0:.1f}m",
                flush=True,
            )
    reports = {}
    frames = {}
    for mode, values in rows.items():
        frame = pd.DataFrame(values)
        frames[mode] = frame
        reports[mode] = _summarize_rows_v1(frame)
        frame.to_csv(OUT_DIR_V2 / f"fold2_{mode}_bywell.csv", index=False)
    rank_frame = pd.DataFrame(rank_rows)
    rank_frame.to_csv(OUT_DIR_V2 / "fold2_candidate_ranks.csv", index=False)
    reports["rank"] = {
        "median": float(rank_frame["truth_candidate_rank"].median()),
        "p90": float(rank_frame["truth_candidate_rank"].quantile(0.90)),
    }
    return reports, frames


def _train_epoch_v2(model, wells, optimizer, scaler, device, epoch):
    model.train()
    order = list(wells)
    random.Random(SEED_V2 + epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "dense": 0.0, "soft_rank": 0.0, "margin": 0.0}
    steps = 0
    for number, well in enumerate(order, start=1):
        panel = _load_sample_v2(well)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            loss, parts = _loss_v2(model, panel, device)
            scaled = loss / GRAD_ACCUM_WELLS_V2
        scaler.scale(scaled).backward()
        if number % GRAD_ACCUM_WELLS_V2 == 0 or number == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_V2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        totals["loss"] += float(loss.detach().cpu())
        for key, value in parts.items():
            totals[key] += value
        if number % 100 == 0 or number == len(order):
            print(
                f"DENSE-SSE epoch={epoch} wells={number}/{len(order)} "
                f"loss={totals['loss'] / number:.4f}",
                flush=True,
            )
    output = {key: value / len(order) for key, value in totals.items()}
    output["optimizer_steps"] = steps
    return output


# ---------------------------------------------------------------------------
# Build label-free training panels, train, then run the one full-bank audit.
# ---------------------------------------------------------------------------


_seed_everything(SEED_V2)
folds_v2 = _load_folds()
train_wells_v2 = sorted(
    folds_v2.loc[folds_v2["fold"].isin(TRAIN_FOLDS), "well"].astype(str).tolist()
)
valid_wells_v2 = sorted(
    folds_v2.loc[folds_v2["fold"].eq(VALID_FOLD), "well"].astype(str).tolist()
)
if len(train_wells_v2) < 450 or len(valid_wells_v2) < 150:
    raise RuntimeError(f"Unexpected folds {len(train_wells_v2)}/{len(valid_wells_v2)}")
for well in valid_wells_v2:
    if not (PANEL_DIR / f"{well}.npz").exists():
        raise FileNotFoundError(PANEL_DIR / f"{well}.npz", "Run V0 through its oracle gate first")

started_v2 = time.time()
build_rows_v2 = []
for number, well in enumerate(train_wells_v2, start=1):
    try:
        if _sample_path_v2(well).exists():
            panel = _load_sample_v2(well)
            report = {
                "well": well,
                "status": "cached",
                "candidates": len(panel["candidate_mse"]),
                "sample_oracle_rmse": float(np.sqrt(panel["candidate_mse"].min())),
            }
        else:
            report = _build_sample_v2(well)
        build_rows_v2.append(report)
    except Exception as error:
        build_rows_v2.append({"well": well, "status": "failed", "reason": repr(error)})
    if number % 25 == 0 or number == len(train_wells_v2):
        print(
            f"build dense-SSE samples {number}/{len(train_wells_v2)} "
            f"elapsed={(time.time() - started_v2) / 60.0:.1f}m",
            flush=True,
        )
build_frame_v2 = pd.DataFrame(build_rows_v2)
build_frame_v2.to_csv(OUT_DIR_V2 / "train_sample_build_report.csv", index=False)
failed_v2 = build_frame_v2[build_frame_v2["status"].eq("failed")]
if len(failed_v2):
    raise RuntimeError(
        f"{len(failed_v2)} dense-SSE samples failed: "
        + repr(failed_v2.head(10).to_dict(orient="records"))
    )

guards_v2 = {
    "train_wells": len(train_wells_v2),
    "valid_wells": len(valid_wells_v2),
    "train_candidates": TRAIN_CANDIDATES_V2,
    "sample_oracle_rmse_median": float(build_frame_v2["sample_oracle_rmse"].median()),
    "fold1_touched": False,
    "label_free_candidate_selection": True,
}
print("DENSE-SSE GUARDS", json.dumps(guards_v2, indent=2), flush=True)

device_v2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_v2 = DenseSSEEnergyV2().to(device_v2)
parameters_v2 = int(sum(parameter.numel() for parameter in model_v2.parameters()))
optimizer_v2 = torch.optim.AdamW(
    model_v2.parameters(), lr=LEARNING_RATE_V2, weight_decay=WEIGHT_DECAY_V2
)
scheduler_v2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_v2, T_max=MAX_EPOCHS_V2, eta_min=MIN_LEARNING_RATE_V2
)
scaler_v2 = torch.amp.GradScaler("cuda", enabled=device_v2.type == "cuda")
checkpoint_path_v2 = MODEL_DIR_V2 / "best_fold2_dense_sse_energy.pt"
history_v2 = []
best_v2 = None
bad_epochs_v2 = 0

for epoch in range(1, MAX_EPOCHS_V2 + 1):
    training = _train_epoch_v2(
        model_v2, train_wells_v2, optimizer_v2, scaler_v2, device_v2, epoch
    )
    validation, _ = _evaluate_pruned_v2(model_v2, valid_wells_v2, device_v2)
    row = {
        "epoch": epoch,
        **{f"train_{key}": value for key, value in training.items()},
        **{f"valid_pruned_{key}": value for key, value in validation.items()},
        "learning_rate": float(optimizer_v2.param_groups[0]["lr"]),
        "elapsed_minutes": (time.time() - started_v2) / 60.0,
    }
    history_v2.append(row)
    print("DENSE-SSE EPOCH", row, flush=True)
    if best_v2 is None or validation["rmse"] < best_v2["rmse"] - 1e-4:
        best_v2 = {"epoch": epoch, **validation}
        torch.save(
            {"model": model_v2.state_dict(), "epoch": epoch, "pruned_metrics": validation},
            checkpoint_path_v2,
        )
        bad_epochs_v2 = 0
    else:
        bad_epochs_v2 += 1
    scheduler_v2.step()
    if epoch >= MIN_EPOCHS_V2 and bad_epochs_v2 >= PATIENCE_V2:
        print("DENSE-SSE early stopping", flush=True)
        break

pd.DataFrame(history_v2).to_csv(OUT_DIR_V2 / "training_history.csv", index=False)
checkpoint_v2 = torch.load(checkpoint_path_v2, map_location=device_v2, weights_only=False)
model_v2.load_state_dict(checkpoint_v2["model"])
reports_v2, frames_v2 = _evaluate_full_v2(model_v2, valid_wells_v2, device_v2)

controls_v2 = {
    "no_gr_rmse_degradation": float(reports_v2["no_gr"]["rmse"] - reports_v2["model"]["rmse"]),
    "shuffled_alignment_rmse_degradation": float(
        reports_v2["shuffled_alignment"]["rmse"] - reports_v2["model"]["rmse"]
    ),
}
checks_v2 = {
    "model": bool(reports_v2["model"]["rmse"] <= GATE_FULL_MODEL_RMSE_V2),
    "top32": bool(reports_v2["top32_oracle"]["rmse"] <= GATE_TOP32_ORACLE_RMSE_V2),
    "top8": bool(reports_v2["top8_oracle"]["rmse"] <= GATE_TOP8_ORACLE_RMSE_V2),
    "rank": bool(reports_v2["rank"]["median"] <= GATE_MEDIAN_RANK_V2),
    "controls": bool(
        controls_v2["no_gr_rmse_degradation"] >= GATE_CONTROL_DEGRADATION_V2
        and controls_v2["shuffled_alignment_rmse_degradation"] >= GATE_CONTROL_DEGRADATION_V2
    ),
}
strict_pass_v2 = bool(all(checks_v2.values()))
verdict_v2 = {
    "experiment": EXPERIMENT_V2,
    "stage": "fold2_dense_candidate_sse_energy_gateway",
    "sample_unit": "one complete physical well with dense candidate-column labels",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_strong_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
        "selected_epoch": int(checkpoint_v2["epoch"]),
    },
    "parameters": parameters_v2,
    "guards": guards_v2,
    "reports": reports_v2,
    "controls": controls_v2,
    "gates": {
        "model_rmse_le": GATE_FULL_MODEL_RMSE_V2,
        "top32_oracle_rmse_le": GATE_TOP32_ORACLE_RMSE_V2,
        "top8_oracle_rmse_le": GATE_TOP8_ORACLE_RMSE_V2,
        "median_truth_candidate_rank_le": GATE_MEDIAN_RANK_V2,
        "each_control_degradation_ge": GATE_CONTROL_DEGRADATION_V2,
    },
    "checks": checks_v2,
    "strict_pass": strict_pass_v2,
    "checkpoint": str(checkpoint_path_v2),
    "elapsed_minutes": (time.time() - started_v2) / 60.0,
}
verdict_v2["next"] = (
    "PASS -> one untouched Fold1 confirmation, then 5-seed averaging"
    if strict_pass_v2
    else "FAIL -> dense labels do not rescue the label-free candidate family"
)
_write_json_v2(OUT_DIR_V2 / "verdict.json", verdict_v2)
print("DENSE-SSE FINAL VERDICT")
print(json.dumps(verdict_v2, indent=2, default=_json_default_v2), flush=True)
print("saved:", OUT_DIR_V2, flush=True)
