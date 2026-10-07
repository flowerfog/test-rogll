"""E6a held-out DeltaCorr-257 local teacher gateway.

The sample unit for splitting remains a complete physical well.  Within each
outer-train well, fixed suffix positions provide dense local correspondence
supervision.  Two matched models are trained: a paired-typewell teacher and a
no-reference baseline.  The paired checkpoint is also evaluated with shuffled
typewells as a causal control.  No SteerSpline, neighbour, CP12, Gold, or OOF
prediction enters this experiment.
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


REQUIRED_E6A = {
    "records",
    "train_wells",
    "valid_wells",
    "_rmse",
    "_seed_everything",
    "OOF_ROOT",
}
missing_e6a = sorted(REQUIRED_E6A.difference(globals()))
if missing_e6a:
    raise RuntimeError(
        "Run E6a in the live native-resolution kernel; missing "
        + repr(missing_e6a)
    )


EXPERIMENT_E6A = "deltacorr_e6a_local_teacher_fold1_v0"
OUT_DIR_E6A = OOF_ROOT / EXPERIMENT_E6A
MODEL_DIR_E6A = OUT_DIR_E6A / "models"
for directory in (OUT_DIR_E6A, MODEL_DIR_E6A):
    directory.mkdir(parents=True, exist_ok=True)

SEED_E6A = 20260730
STATE_OFFSETS_E6A = np.arange(-128.0, 129.0, 1.0, dtype=np.float32)
N_STATES_E6A = len(STATE_OFFSETS_E6A)
REFERENCE_CONTEXT_E6A = 64
REFERENCE_START_E6A = REFERENCE_CONTEXT_E6A
PATCH_RADIUS_E6A = 32
PATCH_LENGTH_E6A = 2 * PATCH_RADIUS_E6A + 1
TRAIN_POSITIONS_E6A = 128
VALID_POSITIONS_E6A = 192
TARGET_SIGMA_E6A = 1.5

EMBEDDING_E6A = 32
EPOCHS_E6A = 10
LEARNING_RATE_E6A = 4e-4
MIN_LEARNING_RATE_E6A = 4e-5
WEIGHT_DECAY_E6A = 1e-5
GRADIENT_CLIP_E6A = 5.0
EARLY_STOP_REPORTS_E6A = 3

PRIMARY_RELATIVE_MAE_GAIN_E6A = 0.20
SHUFFLE_MAE_DEGRADATION_E6A = 0.10
POSITIVE_WELL_FRACTION_E6A = 0.55
BOOTSTRAP_ITERATIONS_E6A = 2000


def _json_e6a(path, payload):
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


def _sample_rows_e6a(record, positions):
    candidates = record.eval_index.astype(np.int64)
    if not len(candidates):
        raise RuntimeError(f"{record.well}: no evaluation rows")
    selected = np.linspace(0, len(candidates) - 1, positions).round().astype(int)
    return candidates[selected]


def _patches_e6a(record, rows):
    offsets = np.arange(-PATCH_RADIUS_E6A, PATCH_RADIUS_E6A + 1)
    indices = np.clip(rows[:, None] + offsets[None], 0, record.horizontal.shape[1] - 1)
    patches = record.horizontal[:, indices].transpose(1, 0, 2)
    return patches.astype(np.float16)


def _build_samples_e6a(wells, positions):
    samples = {}
    for index, well in enumerate(wells, 1):
        record = records[well]
        rows = _sample_rows_e6a(record, positions)
        labels = record.truth[rows].astype(np.float64) - float(record.anchor_tvt)
        if not np.isfinite(labels).all() or np.max(np.abs(labels)) > 128.0001:
            raise RuntimeError(
                f"{well}: invalid Delta-TVT labels, range="
                f"{np.nanmin(labels):.3f}..{np.nanmax(labels):.3f}"
            )
        reference = record.typewell[
            :,
            REFERENCE_START_E6A : REFERENCE_START_E6A + N_STATES_E6A,
        ]
        if reference.shape != (4, N_STATES_E6A):
            raise RuntimeError(f"{well}: bad typewell state slice {reference.shape}")
        samples[well] = {
            "patches": _patches_e6a(record, rows),
            "condition": record.condition[:, rows].T.astype(np.float16),
            "labels": labels.astype(np.float32),
            "reference": reference.astype(np.float16),
            "eval_rows": int(record.eval_rows),
            "sample_rows": rows.astype(np.int32),
        }
        if index % 100 == 0 or index == len(wells):
            print(
                f"build E6a samples {index}/{len(wells)} positions={positions}",
                flush=True,
            )
    return samples


class ChannelNormE6A(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values):
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class ResidualE6A(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                5,
                padding=2 * dilation,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv1d(channels, channels, 1),
            ChannelNormE6A(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return F.gelu(values + self.block(values))


class DeltaCorrTeacherE6A(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Conv1d(4, EMBEDDING_E6A, 9, padding=4),
            ChannelNormE6A(EMBEDDING_E6A),
            nn.GELU(),
            ResidualE6A(EMBEDDING_E6A, 1),
            ResidualE6A(EMBEDDING_E6A, 2),
            nn.Conv1d(EMBEDDING_E6A, EMBEDDING_E6A, 5, stride=2, padding=2),
            ChannelNormE6A(EMBEDDING_E6A),
            nn.GELU(),
            ResidualE6A(EMBEDDING_E6A, 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(10, 48),
            nn.GELU(),
            nn.Linear(48, EMBEDDING_E6A),
        )
        self.reference_encoder = nn.Sequential(
            nn.Conv1d(4, EMBEDDING_E6A, 7, padding=3),
            ChannelNormE6A(EMBEDDING_E6A),
            nn.GELU(),
            ResidualE6A(EMBEDDING_E6A, 1),
            ResidualE6A(EMBEDDING_E6A, 2),
            ResidualE6A(EMBEDDING_E6A, 4),
        )
        pair_channels = 4 * EMBEDDING_E6A + 3
        self.score = nn.Sequential(
            nn.Conv1d(pair_channels, 64, 1),
            ChannelNormE6A(64),
            nn.GELU(),
            nn.Conv1d(64, 32, 1),
            nn.GELU(),
            nn.Conv1d(32, 1, 1),
        )
        state = torch.from_numpy(STATE_OFFSETS_E6A / 128.0)
        self.register_buffer("state", state[None, None], persistent=False)

    def forward(self, patches, condition, reference):
        query = self.query_encoder(patches).squeeze(-1)
        query = query + self.condition_encoder(condition)
        reference_embedding = self.reference_encoder(reference)
        if reference_embedding.shape[0] == 1 and query.shape[0] > 1:
            reference_embedding = reference_embedding.expand(query.shape[0], -1, -1)
        query_volume = query.unsqueeze(-1).expand(-1, -1, N_STATES_E6A)
        difference = torch.abs(query_volume - reference_embedding)
        product = query_volume * reference_embedding
        reference_valid = reference[:, 1:2]
        if reference_valid.shape[0] == 1 and query.shape[0] > 1:
            reference_valid = reference_valid.expand(query.shape[0], -1, -1)
        state = self.state.expand(query.shape[0], -1, -1)
        state_squared = state.square()
        pair = torch.cat(
            [
                query_volume,
                reference_embedding,
                difference,
                product,
                reference_valid,
                state,
                state_squared,
            ],
            dim=1,
        )
        return self.score(pair).squeeze(1)


def _tensors_e6a(sample, device, reference=None):
    if reference is None:
        reference = sample["reference"]
    return (
        torch.from_numpy(sample["patches"].astype(np.float32)).to(device),
        torch.from_numpy(sample["condition"].astype(np.float32)).to(device),
        torch.from_numpy(np.asarray(reference, np.float32))[None].to(device),
        torch.from_numpy(sample["labels"]).to(device),
    )


def _reference_mode_e6a(sample, mode, shuffled_sample=None):
    if mode == "paired":
        return sample["reference"]
    if mode == "no_reference":
        return np.zeros_like(sample["reference"])
    if mode == "shuffled_typewell":
        return shuffled_sample["reference"]
    raise ValueError(mode)


def _loss_e6a(logits, labels):
    states = torch.from_numpy(STATE_OFFSETS_E6A).to(logits.device)
    target = torch.exp(
        -0.5 * ((states[None] - labels[:, None]) / TARGET_SIGMA_E6A) ** 2
    )
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_probability = F.log_softmax(logits, dim=1)
    ce = -(target * log_probability).sum(dim=1).mean()
    probability = torch.softmax(logits, dim=1)
    mean = (probability * states[None]).sum(dim=1)
    offset = F.smooth_l1_loss(mean / 16.0, labels / 16.0)
    loss = ce + 0.25 * offset
    return loss, ce.detach(), offset.detach()


@torch.no_grad()
def _evaluate_e6a(model, samples, wells, device, mode):
    model.eval()
    states = torch.from_numpy(STATE_OFFSETS_E6A).to(device)
    labels_all, predictions_all, nll_all, well_labels = [], [], [], []
    bywell = []
    shuffled = list(wells[1:]) + [wells[0]]
    for well, shuffled_well in zip(wells, shuffled):
        sample = samples[well]
        shuffled_sample = samples[shuffled_well]
        reference = _reference_mode_e6a(sample, mode, shuffled_sample)
        patches, condition, reference_tensor, labels = _tensors_e6a(
            sample, device, reference
        )
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(patches, condition, reference_tensor)
        probability = torch.softmax(logits.float(), dim=1)
        prediction = (probability * states[None]).sum(dim=1)
        nearest = torch.clamp(torch.round(labels + 128.0), 0, 256).long()
        nll = -torch.log_softmax(logits.float(), dim=1).gather(
            1, nearest[:, None]
        )[:, 0]
        labels_np = labels.cpu().numpy().astype(np.float64)
        prediction_np = prediction.cpu().numpy().astype(np.float64)
        nll_np = nll.cpu().numpy().astype(np.float64)
        labels_all.append(labels_np)
        predictions_all.append(prediction_np)
        nll_all.append(nll_np)
        well_labels.append(np.full(len(labels_np), well, dtype="U16"))
        bywell.append(
            {
                "well": well,
                "rows": len(labels_np),
                "eval_rows": sample["eval_rows"],
                "mae": float(np.mean(np.abs(prediction_np - labels_np))),
                "rmse": _rmse(labels_np, prediction_np),
                "nll": float(np.mean(nll_np)),
                "within_5": float(np.mean(np.abs(prediction_np - labels_np) <= 5.0)),
            }
        )
    labels = np.concatenate(labels_all)
    predictions = np.concatenate(predictions_all)
    nll = np.concatenate(nll_all)
    bywell = pd.DataFrame(bywell)
    metrics = {
        "mode": mode,
        "rows": len(labels),
        "wells": len(wells),
        "delta_mae": float(np.mean(np.abs(predictions - labels))),
        "delta_rmse": _rmse(labels, predictions),
        "nll": float(np.mean(nll)),
        "within_5": float(np.mean(np.abs(predictions - labels) <= 5.0)),
        "median_well_mae": float(bywell["mae"].median()),
        "p95_well_mae": float(bywell["mae"].quantile(0.95)),
    }
    payload = {
        "well": np.concatenate(well_labels),
        "label": labels.astype(np.float32),
        "prediction": predictions.astype(np.float32),
        "nll": nll.astype(np.float32),
    }
    return metrics, bywell, payload


def _train_arm_e6a(arm, train_samples, valid_samples, device):
    _seed_everything(SEED_E6A + (arm == "no_reference"))
    model = DeltaCorrTeacherE6A().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_E6A,
        weight_decay=WEIGHT_DECAY_E6A,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_E6A, eta_min=MIN_LEARNING_RATE_E6A
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=512.0
    )
    mean_rows = float(np.mean([train_samples[well]["eval_rows"] for well in train_wells]))
    rng = np.random.default_rng(SEED_E6A + (arm == "no_reference"))
    history, best = [], None
    reports_without_improvement = 0
    started = time.time()

    for epoch in range(1, EPOCHS_E6A + 1):
        model.train()
        sums = {"loss": 0.0, "ce": 0.0, "offset": 0.0, "skipped": 0}
        for step, index in enumerate(rng.permutation(len(train_wells)), 1):
            well = train_wells[int(index)]
            sample = train_samples[well]
            reference = _reference_mode_e6a(sample, arm)
            patches, condition, reference_tensor, labels = _tensors_e6a(
                sample, device, reference
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(patches, condition, reference_tensor)
                loss, ce, offset = _loss_e6a(logits, labels)
                loss = loss * sample["eval_rows"] / max(mean_rows, 1.0)
            before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_E6A)
            scaler.step(optimizer)
            scaler.update()
            sums["skipped"] += int(float(scaler.get_scale()) < before)
            sums["loss"] += float(loss.detach().cpu())
            sums["ce"] += float(ce.cpu())
            sums["offset"] += float(offset.cpu())
            if step % 100 == 0:
                print(
                    f"E6a {arm} epoch {epoch}/{EPOCHS_E6A} wells {step}/{len(train_wells)}",
                    flush=True,
                )
        scheduler.step()
        metrics, bywell, _ = _evaluate_e6a(
            model, valid_samples, valid_wells, device, arm
        )
        row = {
            "arm": arm,
            "epoch": epoch,
            "train_loss": sums["loss"] / len(train_wells),
            "train_ce": sums["ce"] / len(train_wells),
            "train_offset": sums["offset"] / len(train_wells),
            **metrics,
            "skipped_steps": sums["skipped"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("E6a EPOCH", row, flush=True)
        if best is None or metrics["delta_mae"] < best["metrics"]["delta_mae"] - 1e-4:
            best = {
                "epoch": epoch,
                "metrics": metrics,
                "bywell": bywell.copy(),
                "state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            }
            reports_without_improvement = 0
        else:
            reports_without_improvement += 1
        if epoch >= 5 and reports_without_improvement >= EARLY_STOP_REPORTS_E6A:
            break
    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history), (time.time() - started) / 60.0


def _well_bootstrap_e6a(full_bywell, baseline_bywell):
    merged = full_bywell[["well", "mae"]].merge(
        baseline_bywell[["well", "mae"]], on="well", suffixes=("_full", "_baseline")
    )
    gains = merged["mae_baseline"].to_numpy() - merged["mae_full"].to_numpy()
    rng = np.random.default_rng(SEED_E6A + 99)
    boot = np.empty(BOOTSTRAP_ITERATIONS_E6A, np.float64)
    for iteration in range(BOOTSTRAP_ITERATIONS_E6A):
        indices = rng.integers(0, len(gains), len(gains))
        boot[iteration] = float(np.mean(gains[indices]))
    return {
        "mean_well_mae_gain": float(np.mean(gains)),
        "median_well_mae_gain": float(np.median(gains)),
        "positive_well_fraction": float(np.mean(gains > 0)),
        "bootstrap_p05_gain": float(np.quantile(boot, 0.05)),
        "bootstrap_p50_gain": float(np.quantile(boot, 0.50)),
        "bootstrap_p95_gain": float(np.quantile(boot, 0.95)),
    }


started_e6a = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("E6a requires CUDA")
device_e6a = torch.device("cuda")

print(
    "E6a CONFIG",
    {
        "train_wells": len(train_wells),
        "valid_wells": len(valid_wells),
        "train_positions_per_well": TRAIN_POSITIONS_E6A,
        "valid_positions_per_well": VALID_POSITIONS_E6A,
        "states": N_STATES_E6A,
        "patch_length": PATCH_LENGTH_E6A,
        "arms": ["paired", "no_reference"],
        "intervention": "shuffled_typewell",
    },
    flush=True,
)

train_samples_e6a = _build_samples_e6a(train_wells, TRAIN_POSITIONS_E6A)
valid_samples_e6a = _build_samples_e6a(valid_wells, VALID_POSITIONS_E6A)

probe_e6a = DeltaCorrTeacherE6A().to(device_e6a)
probe_sample_e6a = train_samples_e6a[train_wells[0]]
probe_tensors_e6a = _tensors_e6a(probe_sample_e6a, device_e6a)
with torch.no_grad(), torch.amp.autocast("cuda"):
    probe_output_e6a = probe_e6a(*probe_tensors_e6a[:3])
parameters_e6a = int(sum(parameter.numel() for parameter in probe_e6a.parameters()))
print(
    "E6a MODEL SMOKE",
    {
        "parameters": parameters_e6a,
        "query": list(probe_tensors_e6a[0].shape),
        "reference": list(probe_tensors_e6a[2].shape),
        "output": list(probe_output_e6a.shape),
        "finite": bool(torch.isfinite(probe_output_e6a).all()),
    },
    flush=True,
)
del probe_e6a, probe_tensors_e6a, probe_output_e6a
gc.collect()
torch.cuda.empty_cache()

models_e6a, bests_e6a, histories_e6a, runtimes_e6a = {}, {}, [], {}
for arm in ("no_reference", "paired"):
    model, best, history, runtime = _train_arm_e6a(
        arm, train_samples_e6a, valid_samples_e6a, device_e6a
    )
    models_e6a[arm] = model
    bests_e6a[arm] = best
    histories_e6a.append(history)
    runtimes_e6a[arm] = runtime
    history.to_csv(OUT_DIR_E6A / f"history_{arm}.csv", index=False)
    best["bywell"].to_csv(OUT_DIR_E6A / f"bywell_{arm}.csv", index=False)
    torch.save(
        {
            "experiment": EXPERIMENT_E6A,
            "arm": arm,
            "epoch": int(best["epoch"]),
            "model": best["state"],
            "metrics": best["metrics"],
        },
        MODEL_DIR_E6A / f"best_{arm}.pt",
    )

full_metrics_e6a, full_bywell_e6a, full_payload_e6a = _evaluate_e6a(
    models_e6a["paired"], valid_samples_e6a, valid_wells, device_e6a, "paired"
)
shuffle_metrics_e6a, shuffle_bywell_e6a, shuffle_payload_e6a = _evaluate_e6a(
    models_e6a["paired"],
    valid_samples_e6a,
    valid_wells,
    device_e6a,
    "shuffled_typewell",
)
baseline_metrics_e6a, baseline_bywell_e6a, baseline_payload_e6a = _evaluate_e6a(
    models_e6a["no_reference"],
    valid_samples_e6a,
    valid_wells,
    device_e6a,
    "no_reference",
)

full_bywell_e6a.to_csv(OUT_DIR_E6A / "final_bywell_paired.csv", index=False)
shuffle_bywell_e6a.to_csv(OUT_DIR_E6A / "final_bywell_shuffled.csv", index=False)
baseline_bywell_e6a.to_csv(OUT_DIR_E6A / "final_bywell_no_reference.csv", index=False)
np.savez_compressed(OUT_DIR_E6A / "predictions_paired.npz", **full_payload_e6a)
np.savez_compressed(OUT_DIR_E6A / "predictions_shuffled.npz", **shuffle_payload_e6a)
np.savez_compressed(OUT_DIR_E6A / "predictions_no_reference.npz", **baseline_payload_e6a)

relative_mae_gain_e6a = float(
    (baseline_metrics_e6a["delta_mae"] - full_metrics_e6a["delta_mae"])
    / max(baseline_metrics_e6a["delta_mae"], 1e-8)
)
shuffle_degradation_e6a = float(
    shuffle_metrics_e6a["delta_mae"] - full_metrics_e6a["delta_mae"]
)
bootstrap_e6a = _well_bootstrap_e6a(full_bywell_e6a, baseline_bywell_e6a)

primary_pass_e6a = bool(relative_mae_gain_e6a >= PRIMARY_RELATIVE_MAE_GAIN_E6A)
shuffle_pass_e6a = bool(
    shuffle_degradation_e6a >= SHUFFLE_MAE_DEGRADATION_E6A
)
well_pass_e6a = bool(
    bootstrap_e6a["positive_well_fraction"] >= POSITIVE_WELL_FRACTION_E6A
    and bootstrap_e6a["bootstrap_p05_gain"] > 0.0
)

verdict_e6a = {
    "experiment": EXPERIMENT_E6A,
    "stage": "E6a_heldout_local_DeltaCorr_teacher",
    "sample_split": "outer_fold_grouped_by_physical_well",
    "train_wells": len(train_wells),
    "valid_wells": len(valid_wells),
    "train_positions_per_well": TRAIN_POSITIONS_E6A,
    "valid_positions_per_well": VALID_POSITIONS_E6A,
    "parameters": parameters_e6a,
    "paired": full_metrics_e6a,
    "no_reference": baseline_metrics_e6a,
    "shuffled_typewell": shuffle_metrics_e6a,
    "relative_mae_gain_vs_no_reference": relative_mae_gain_e6a,
    "shuffled_typewell_mae_degradation": shuffle_degradation_e6a,
    "well_bootstrap": bootstrap_e6a,
    "gates": {
        "relative_mae_gain_ge": PRIMARY_RELATIVE_MAE_GAIN_E6A,
        "shuffle_mae_degradation_ge": SHUFFLE_MAE_DEGRADATION_E6A,
        "positive_well_fraction_ge": POSITIVE_WELL_FRACTION_E6A,
        "bootstrap_p05_gain_gt": 0.0,
    },
    "primary_pass": primary_pass_e6a,
    "shuffle_pass": shuffle_pass_e6a,
    "well_pass": well_pass_e6a,
    "strict_pass": bool(primary_pass_e6a and shuffle_pass_e6a and well_pass_e6a),
    "best_epochs": {
        arm: int(best["epoch"]) for arm, best in bests_e6a.items()
    },
    "runtime_minutes": runtimes_e6a,
    "elapsed_minutes": (time.time() - started_e6a) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> freeze DeltaCorr teacher and test evidence injection into PCA16 head"
        if primary_pass_e6a and shuffle_pass_e6a and well_pass_e6a
        else "FAIL -> close DeltaCorr and the remaining PDF roadmap"
    ),
}
_json_e6a(OUT_DIR_E6A / "verdict.json", verdict_e6a)

print("\nE6a TRAINING HISTORY")
print(pd.concat(histories_e6a, ignore_index=True).to_string(index=False), flush=True)
print("\nE6a FINAL METRICS")
print(
    pd.DataFrame(
        [full_metrics_e6a, baseline_metrics_e6a, shuffle_metrics_e6a]
    ).to_string(index=False),
    flush=True,
)
print("\nE6a FINAL VERDICT")
print(json.dumps(verdict_e6a, indent=2, ensure_ascii=False), flush=True)
print("saved:", OUT_DIR_E6A, flush=True)
