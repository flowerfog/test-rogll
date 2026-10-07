"""Fold-2 gateway for a three-dimensional anchored Strong residual head.

Prerequisite
------------
Run ``strongbase_completewell_residual_gateway_v0_cell.py`` once far enough to
build its ``well_cache``.  This cell does not use that experiment's model or
predictions; it reuses only the legal complete-well input tensors and the
fold-clean Strong OOF trajectory stored in each cache record.

The model predicts three per-well coefficients for the frozen basis selected
by ``anchored_residual_basis_oracle_v1_cell.py``:

    1 - exp(-distance / 512), distance / 1000, progress ** 2

The basis is zero at prediction start.  Coefficients are whitened separately
for each physical well so coefficient MSE is equivalent to row-weighted path
MSE.  Folds 3/4/5 train the network and Fold 2 selects epoch and dose.  Fold 1
is never read.
"""

from pathlib import Path
import gc
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
from torch.utils.data import DataLoader, Dataset


EXPERIMENT = "anchored_residual_coefficient_gateway_v0"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
SOURCE_CACHE = OOF_ROOT / "strongbase_completewell_residual_gateway_v0" / "well_cache"
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260721
TRAIN_FOLDS = (3, 4, 5)
VALID_FOLD = 2
TAU_FT = 512.0
DIMENSIONS = 3
HIDDEN = 96
DILATIONS = (1, 2, 4, 8, 16, 32, 64)
DROPOUT = 0.10
BATCH_SIZE = 8
MAX_EPOCHS = 24
MIN_EPOCHS = 6
PATIENCE = 6
LEARNING_RATE = 4e-4
MIN_LEARNING_RATE = 4e-5
WEIGHT_DECAY = 1e-4
COEFFICIENT_LOSS_WEIGHT = 0.10
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
STRATA = (
    ("near_0_500", 0.0, 500.0),
    ("mid_500_1000", 500.0, 1000.0),
    ("mid_1000_2000", 1000.0, 2000.0),
    ("far_gt2000", 2000.0, np.inf),
)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _design(distance):
    distance = np.maximum(np.asarray(distance, np.float64), 0.0)
    maximum = max(float(np.max(distance)), 1.0)
    progress = distance / maximum
    return np.column_stack(
        [
            1.0 - np.exp(-distance / TAU_FT),
            distance / 1000.0,
            progress * progress,
        ]
    ).astype(np.float64)


def _weighted_geometry(design, target, weight):
    design = np.asarray(design, np.float64)
    target = np.asarray(target, np.float64)
    weight = np.asarray(weight, np.float64)
    weight = np.maximum(weight, 0.0)
    weight /= max(float(weight.sum()), 1e-12)
    gram = design.T @ (weight[:, None] * design)
    jitter = max(float(np.trace(gram)) * 1e-8 / DIMENSIONS, 1e-10)
    chol = np.linalg.cholesky(gram + jitter * np.eye(DIMENSIONS))
    root_weight = np.sqrt(weight)
    beta = np.linalg.lstsq(
        design * root_weight[:, None], target * root_weight, rcond=None
    )[0]
    whitened = chol.T @ beta
    transform = design @ np.linalg.inv(chol.T)
    oracle = design @ beta
    condition = float(np.linalg.cond(gram + jitter * np.eye(DIMENSIONS)))
    return (
        whitened.astype(np.float32),
        transform.astype(np.float32),
        beta.astype(np.float32),
        oracle.astype(np.float32),
        condition,
    )


if FOLD_PATH is None:
    raise FileNotFoundError("fold-clean well_metadata.csv not found")
if not SOURCE_CACHE.exists():
    raise FileNotFoundError(
        f"Missing {SOURCE_CACHE}. Run strongbase_completewell_residual_gateway_v0_cell.py "
        "once to build its legal-input well_cache."
    )

fold_frame = pd.read_csv(FOLD_PATH, dtype={"well": str})
fold_frame["well"] = fold_frame["well"].astype(str)
fold_frame["fold"] = pd.to_numeric(fold_frame["fold"], errors="raise").astype(int)
fold_map = fold_frame.drop_duplicates("well").set_index("well")["fold"].to_dict()

cache_wells = sorted(path.stem for path in SOURCE_CACHE.glob("*.npz"))
train_wells = [well for well in cache_wells if fold_map.get(well) in TRAIN_FOLDS]
valid_wells = [well for well in cache_wells if fold_map.get(well) == VALID_FOLD]
if len(train_wells) != 463 or len(valid_wells) != 155:
    raise RuntimeError(
        f"Unexpected cache/fold sizes: train={len(train_wells)}, valid={len(valid_wells)}. "
        "Expected 463/155; rebuild the Strong-base cache if files are incomplete."
    )


RAW_CACHE = {}


def _load_raw(well):
    if well not in RAW_CACHE:
        with np.load(SOURCE_CACHE / f"{well}.npz", allow_pickle=False) as data:
            RAW_CACHE[well] = {key: data[key].copy() for key in data.files}
    return RAW_CACHE[well]


with np.load(SOURCE_CACHE / f"{train_wells[0]}.npz", allow_pickle=False) as probe:
    feature_names = probe["feature_names"].astype(str).tolist()
INPUT_CHANNELS = len(feature_names)
GR_CHANNELS = np.asarray(
    [name.startswith(("gr_", "match_", "tw_")) for name in feature_names], bool
)
ALIGNMENT_CHANNELS = np.asarray(
    [name.startswith(("match_", "tw_")) for name in feature_names], bool
)
if INPUT_CHANNELS < 10 or not GR_CHANNELS.any() or not ALIGNMENT_CHANNELS.any():
    raise RuntimeError("Strong-base cache feature schema is not compatible")


class CoefficientDataset(Dataset):
    def __init__(self, wells):
        self.wells = list(wells)

    def __len__(self):
        return len(self.wells)

    def __getitem__(self, index):
        well = self.wells[index]
        data = _load_raw(well)
        node_md = data["node_md"].astype(np.float64)
        eval_md = data["eval_md"].astype(np.float64)
        distance = np.interp(
            node_md, eval_md, data["distance_full"].astype(np.float64)
        )
        target = data["target"].astype(np.float32)
        weight = data["node_weight"].astype(np.float32)
        whitened, transform, _, _, _ = _weighted_geometry(
            _design(distance), target, weight
        )
        return {
            "well": well,
            "features": torch.from_numpy(data["features"].astype(np.float32)),
            "target": torch.from_numpy(target),
            "weight": torch.from_numpy(weight),
            "whitened": torch.from_numpy(whitened),
            "transform": torch.from_numpy(transform),
        }


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels, channels, 5, padding=2 * dilation,
                dilation=dilation, groups=channels,
            ),
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return F.gelu(values + self.network(values))


class AnchoredCoefficientNet(nn.Module):
    def __init__(self, input_channels, target_scale):
        super().__init__()
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))
        self.input = nn.Sequential(
            nn.Conv1d(input_channels, HIDDEN, 1),
            nn.GroupNorm(8, HIDDEN),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(HIDDEN, dilation) for dilation in DILATIONS]
        )
        self.attention = nn.Conv1d(HIDDEN, 1, 1)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN * 4, HIDDEN * 2),
            nn.LayerNorm(HIDDEN * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN * 2, DIMENSIONS),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features):
        hidden = self.blocks(self.input(features))
        attention = torch.softmax(self.attention(hidden), dim=-1)
        attended = torch.sum(hidden * attention, dim=-1)
        mean = hidden.mean(dim=-1)
        maximum = hidden.amax(dim=-1)
        endpoints = 0.5 * (hidden[:, :, 0] + hidden[:, :, -1])
        summary = torch.cat([attended, mean, maximum, endpoints], dim=1)
        return self.head(summary) * self.target_scale


def _target_scale(wells):
    targets = []
    conditions = []
    oracle_sse = 0.0
    base_sse = 0.0
    rows = 0
    for well in wells:
        data = _load_raw(well)
        node_md = data["node_md"].astype(np.float64)
        distance = np.interp(
            node_md,
            data["eval_md"].astype(np.float64),
            data["distance_full"].astype(np.float64),
        )
        target = data["target"].astype(np.float64)
        weight = data["node_weight"].astype(np.float64)
        whitened, _, _, oracle, condition = _weighted_geometry(
            _design(distance), target, weight
        )
        targets.append(whitened)
        conditions.append(condition)
        base_sse += float(np.sum(weight * target**2))
        oracle_sse += float(np.sum(weight * (target - oracle)**2))
        rows += len(target)
    targets = np.stack(targets)
    scale = np.maximum(np.std(targets, axis=0), 1.0).astype(np.float32)
    return scale, {
        "target_mean": np.mean(targets, axis=0),
        "target_std": np.std(targets, axis=0),
        "condition_p50": float(np.median(conditions)),
        "condition_p95": float(np.quantile(conditions, 0.95)),
        "node_weighted_base_rmse": float(np.sqrt(base_sse / max(len(wells) * 512, 1))),
        "node_weighted_oracle_rmse": float(np.sqrt(oracle_sse / max(len(wells) * 512, 1))),
    }


def _loss(model, batch, device):
    features = batch["features"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    weight = batch["weight"].to(device, non_blocking=True)
    whitened = batch["whitened"].to(device, non_blocking=True)
    transform = batch["transform"].to(device, non_blocking=True)
    prediction = model(features)
    correction = torch.einsum("blc,bc->bl", transform, prediction)
    path_mse = torch.sum(weight * (correction - target) ** 2, dim=1) / torch.clamp(
        torch.sum(weight, dim=1), min=1e-6
    )
    coefficient_mse = torch.mean(
        ((prediction - whitened) / model.target_scale) ** 2, dim=1
    )
    loss = path_mse.mean() + COEFFICIENT_LOSS_WEIGHT * coefficient_mse.mean()
    return loss, float(path_mse.mean().detach().cpu()), float(coefficient_mse.mean().detach().cpu())


def _predict_whitened(model, wells, device, control="full"):
    model.eval()
    outputs = {}
    donor_features = None
    if control == "shuffled_alignment":
        donor_features = {
            well: _load_raw(wells[(index + 37) % len(wells)])["features"].astype(np.float32)
            for index, well in enumerate(wells)
        }
    with torch.no_grad():
        for well in wells:
            features = _load_raw(well)["features"].astype(np.float32)
            if control == "no_gr":
                features[GR_CHANNELS] = 0.0
            elif control == "shuffled_alignment":
                features[ALIGNMENT_CHANNELS] = donor_features[well][ALIGNMENT_CHANNELS]
            elif control != "full":
                raise ValueError(control)
            tensor = torch.from_numpy(features[None]).to(device)
            outputs[well] = model(tensor)[0].float().cpu().numpy()
    return outputs


def _evaluate_predictions(predictions, wells, dose):
    frames = []
    bywell_rows = []
    oracle_sse = 0.0
    for well in wells:
        data = _load_raw(well)
        eval_md = data["eval_md"].astype(np.float64)
        distance = data["distance_full"].astype(np.float64)
        truth = data["truth_full"].astype(np.float64)
        base = data["base_full"].astype(np.float64)
        node_md = data["node_md"].astype(np.float64)
        node_distance = np.interp(node_md, eval_md, distance)
        node_target = data["target"].astype(np.float64)
        node_weight = data["node_weight"].astype(np.float64)
        _, _, beta_oracle, _, _ = _weighted_geometry(
            _design(node_distance), node_target, node_weight
        )
        node_gram = _design(node_distance).T @ (
            (node_weight / node_weight.sum())[:, None] * _design(node_distance)
        )
        jitter = max(float(np.trace(node_gram)) * 1e-8 / DIMENSIONS, 1e-10)
        chol = np.linalg.cholesky(node_gram + jitter * np.eye(DIMENSIONS))
        beta = np.linalg.solve(chol.T, predictions[well].astype(np.float64))
        correction = float(dose) * (_design(distance) @ beta)
        oracle_correction = _design(distance) @ beta_oracle
        prediction = base + correction
        error = prediction - truth
        base_error = base - truth
        oracle_error = base + oracle_correction - truth
        oracle_sse += float(np.sum(oracle_error**2))
        bywell_rows.append(
            {
                "well": well,
                "rows": len(truth),
                "base_rmse": float(np.sqrt(np.mean(base_error**2))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "oracle_rmse": float(np.sqrt(np.mean(oracle_error**2))),
            }
        )
        frames.append(
            pd.DataFrame(
                {
                    "well": well,
                    "truth": truth,
                    "base": base,
                    "prediction": prediction,
                    "distance_ft": distance,
                    "correction": correction,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    bywell = pd.DataFrame(bywell_rows)
    error = frame["prediction"].to_numpy(float) - frame["truth"].to_numpy(float)
    base_error = frame["base"].to_numpy(float) - frame["truth"].to_numpy(float)
    base_sse = float(np.sum(base_error**2))
    sse = float(np.sum(error**2))
    denominator = max(base_sse - oracle_sse, 1e-12)
    metrics = {
        "rows": len(frame),
        "wells": len(bywell),
        "dose": float(dose),
        "exact_base_sse": base_sse,
        "exact_sse": sse,
        "exact_oracle_sse": oracle_sse,
        "base_rmse": float(np.sqrt(base_sse / len(frame))),
        "rmse": float(np.sqrt(sse / len(frame))),
        "oracle_rmse": float(np.sqrt(oracle_sse / len(frame))),
        "gain": float(np.sqrt(base_sse / len(frame)) - np.sqrt(sse / len(frame))),
        "oracle_sse_recovery": float((base_sse - sse) / denominator),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
        "mean_abs_move": float(np.mean(np.abs(frame["correction"]))),
        "p95_abs_move": float(np.quantile(np.abs(frame["correction"]), 0.95)),
    }
    distance = frame["distance_ft"].to_numpy(float)
    for name, low, high in STRATA:
        mask = (distance >= low) & (distance < high)
        metrics[f"{name}_base_rmse"] = float(np.sqrt(np.mean(base_error[mask] ** 2)))
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2)))
        metrics[f"{name}_gain"] = metrics[f"{name}_base_rmse"] - metrics[f"{name}_rmse"]
    return metrics, bywell, frame


def _evaluate(model, wells, device, control="full", dose=None):
    predictions = _predict_whitened(model, wells, device, control=control)
    if dose is not None:
        metrics, bywell, frame = _evaluate_predictions(predictions, wells, dose)
        return metrics, bywell, frame, pd.DataFrame([metrics])
    scan_rows = []
    payloads = {}
    for candidate_dose in DOSES:
        metrics, bywell, frame = _evaluate_predictions(predictions, wells, candidate_dose)
        scan_rows.append(metrics)
        payloads[candidate_dose] = (metrics, bywell, frame)
    scan = pd.DataFrame(scan_rows).sort_values(["rmse", "dose"]).reset_index(drop=True)
    selected_dose = float(scan.iloc[0]["dose"])
    metrics, bywell, frame = payloads[selected_dose]
    return metrics, bywell, frame, scan


started = time.time()
_seed_everything(SEED)
target_scale, target_diagnostics = _target_scale(train_wells)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnchoredCoefficientNet(INPUT_CHANNELS, target_scale).to(device)
parameters = int(sum(parameter.numel() for parameter in model.parameters()))

print(
    "COEFFICIENT GATEWAY CONFIG",
    {
        "train_wells": len(train_wells),
        "valid_wells": len(valid_wells),
        "input_channels": INPUT_CHANNELS,
        "parameters": parameters,
        "target_scale": target_scale.tolist(),
        "device": str(device),
    },
    flush=True,
)

loader = DataLoader(
    CoefficientDataset(train_wells),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=device.type == "cuda",
)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS, eta_min=MIN_LEARNING_RATE
)

base_metrics, _, _, base_scan = _evaluate(model, valid_wells, device, control="full")
best = {
    "epoch": 0,
    "metrics": base_metrics,
    "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
}
history = [{"epoch": 0, "train_loss": np.nan, **base_metrics}]
stale = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    totals = {"loss": 0.0, "path": 0.0, "coefficient": 0.0, "batches": 0}
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, path_loss, coefficient_loss = _loss(model, batch, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["path"] += path_loss
        totals["coefficient"] += coefficient_loss
        totals["batches"] += 1
    scheduler.step()
    metrics, _, _, _ = _evaluate(model, valid_wells, device, control="full")
    row = {
        "epoch": epoch,
        "train_loss": totals["loss"] / totals["batches"],
        "train_path_mse": totals["path"] / totals["batches"],
        "train_coefficient_mse": totals["coefficient"] / totals["batches"],
        **metrics,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    history.append(row)
    print("COEFFICIENT EPOCH", row, flush=True)
    if metrics["rmse"] < best["metrics"]["rmse"] - 1e-5:
        best = {
            "epoch": epoch,
            "metrics": metrics,
            "state": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
        }
        stale = 0
    else:
        stale += 1
    if epoch >= MIN_EPOCHS and stale >= PATIENCE:
        print("COEFFICIENT early stopping", flush=True)
        break

history = pd.DataFrame(history)
history.to_csv(OUT_DIR / "training_history.csv", index=False)
model.load_state_dict(best["state"])
full_metrics, bywell, rows, dose_scan = _evaluate(
    model, valid_wells, device, control="full"
)
selected_dose = float(full_metrics["dose"])
bywell.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)
dose_scan.to_csv(OUT_DIR / "fold2_dose_scan.csv", index=False)
rows[["well", "truth", "base", "prediction", "distance_ft", "correction"]].to_csv(
    OUT_DIR / "fold2_rows.csv.gz", index=False, compression="gzip"
)

control_rows = []
for control in ("full", "no_gr", "shuffled_alignment"):
    metrics, _, _, _ = _evaluate(
        model, valid_wells, device, control=control, dose=selected_dose
    )
    control_rows.append(
        {
            "mode": control,
            **metrics,
            "degradation_vs_full": metrics["rmse"] - full_metrics["rmse"],
            "gain_collapse_vs_full": full_metrics["gain"] - metrics["gain"],
        }
    )
controls = pd.DataFrame(control_rows)
controls.to_csv(OUT_DIR / "causal_controls.csv", index=False)

checkpoint = MODEL_DIR / "best_fold2_coefficient_head.pt"
torch.save(
    {
        "experiment": EXPERIMENT,
        "epoch": int(best["epoch"]),
        "dose": selected_dose,
        "model": best["state"],
        "feature_names": feature_names,
        "target_scale": target_scale,
        "basis": ["ramp512", "linear_per_1000ft", "progress_squared"],
        "metrics": full_metrics,
    },
    checkpoint,
)

no_gr = controls.loc[controls["mode"].eq("no_gr")].iloc[0]
shuffled = controls.loc[controls["mode"].eq("shuffled_alignment")].iloc[0]
representation_pass = bool(full_metrics["oracle_rmse"] <= 3.25)
transfer_pass = bool(
    full_metrics["gain"] >= 0.50
    and full_metrics["far_gt2000_gain"] >= 0.50
    and full_metrics["oracle_sse_recovery"] >= 0.15
)
causal_pass = bool(
    max(float(no_gr["gain_collapse_vs_full"]), float(shuffled["gain_collapse_vs_full"]))
    >= 0.25
)
target_pass = bool(full_metrics["rmse"] <= 5.5)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "fold2_anchored_residual_coefficient_gateway",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_cp12_gold": False,
    "base": "fold-clean strong OOF",
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
        "selected_epoch": int(best["epoch"]),
        "selected_dose": selected_dose,
    },
    "basis": {
        "terms": ["ramp512", "linear_per_1000ft", "progress_squared"],
        "dimensions": DIMENSIONS,
        "per_well_gram_whitening": True,
        "anchor_correction": 0.0,
    },
    "parameters": parameters,
    "target_diagnostics": target_diagnostics,
    "fold2": full_metrics,
    "controls": {
        "no_gr": no_gr.to_dict(),
        "shuffled_alignment": shuffled.to_dict(),
    },
    "gates": {
        "representation_oracle_rmse_le": 3.25,
        "gain_vs_strong_ge": 0.50,
        "far_gain_vs_strong_ge": 0.50,
        "oracle_sse_recovery_ge": 0.15,
        "one_control_gain_collapse_ge": 0.25,
        "target_rmse_le": 5.5,
    },
    "representation_pass": representation_pass,
    "transfer_pass": transfer_pass,
    "causal_pass": causal_pass,
    "target_5p5_pass": target_pass,
    "strict_pass": bool(representation_pass and transfer_pass and causal_pass),
    "checkpoint": str(checkpoint),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> freeze epoch/dose and run a separately trained untouched Fold1 confirmation"
        if representation_pass and transfer_pass and causal_pass
        else "FAIL -> the three-dimensional residual exists but is not predictable from this input encoder"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nDOSE SCAN")
print(dose_scan.to_string(index=False), flush=True)
print("\nCONTROLS")
print(controls.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)

del model, loader
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
