"""Axial/RAFT-style iterative TVT correspondence, fold-1 gateway.

Run after ``tvt_corrnet_prefix_meta_adapter_v0_cell.py`` in the same kernel.
It reuses the isolated v4 cost-volume panels but not the v4 U-Net weights.
Horizontal context is encoded with axial self-attention.  A recurrent update
block repeatedly samples the typewell/heel cost volume around its current
continuous TVT correction and refines the complete trajectory.

No strong, strong_bias, CP12, Gold, Neighbor, or neighbor-well data is used.
The likelihood-PF trajectory is only the fixed centre of the narrow TVT panel,
matching the PDF's CorrNet representation.
"""

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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


REQUIRED_GLOBALS = (
    "V4",
    "BASE_OUT_DIR",
    "manifest",
    "fold",
    "valid_store",
    "PanelStore",
    "_panel_key",
)
missing = [name for name in REQUIRED_GLOBALS if name not in globals()]
if missing:
    raise RuntimeError(
        "Run tvt_corrnet_prefix_meta_adapter_v0_cell.py first; missing: "
        + repr(missing)
    )

OOF_ROOT_RAFT = Path(V4["OOF_ROOT"])
OUT_DIR_RAFT = OOF_ROOT_RAFT / "tvt_corrnet_axial_raft_onefold_v0"
MODEL_DIR_RAFT = OUT_DIR_RAFT / "models"
for directory in (OUT_DIR_RAFT, MODEL_DIR_RAFT):
    directory.mkdir(parents=True, exist_ok=True)

STATE_OFFSETS_RAFT = np.asarray(V4["STATE_OFFSETS"], np.float32)
PREFIX_COLUMNS_RAFT = int(V4["PREFIX_COLUMNS"])
N_COLUMNS_RAFT = int(V4["N_COLUMNS"])
VALID_FOLD_RAFT = int(V4["VALID_FOLD"])
SEED_RAFT = 20260721
BATCH_SIZE_RAFT = 3
EPOCHS_RAFT = 24
EARLY_STOPPING_RAFT = 6
LEARNING_RATE_RAFT = 3e-4
MIN_LEARNING_RATE_RAFT = 2e-5
WEIGHT_DECAY_RAFT = 2e-4
HIDDEN_CHANNELS = 64
COST_CHANNELS = 16
RAFT_ITERATIONS = 8
ITERATION_GAMMA = 0.82
LOCAL_OFFSETS_FT = (-16.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 16.0)

device_raft = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device_raft.type == "cuda":
    torch.set_float32_matmul_precision("high")


def _seed_raft(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


guard_path = Path(BASE_OUT_DIR) / "panel_guards.csv"
if not guard_path.exists():
    raise FileNotFoundError(guard_path)
guard = pd.read_csv(guard_path, dtype={"well": str, "tag": str})
guard["well"] = guard["well"].astype(str)
guard["tag"] = guard["tag"].astype(str)
usable_flag = guard["usable"].astype(str).str.lower().isin(("true", "1", "yes"))
usable = set(
    zip(
        guard.loc[usable_flag, "well"],
        guard.loc[usable_flag, "tag"],
    )
)
train_wells_raft = set(
    fold.loc[fold["fold"].astype(int) != VALID_FOLD_RAFT, "well"].astype(str)
)
valid_wells_raft = set(
    fold.loc[fold["fold"].astype(int) == VALID_FOLD_RAFT, "well"].astype(str)
)
train_tasks_raft = manifest[
    manifest["well"].astype(str).isin(train_wells_raft)
].copy()
train_tasks_raft = train_tasks_raft[
    train_tasks_raft.apply(
        lambda row: (str(row["well"]), str(row["tag"])) in usable, axis=1
    )
]
valid_tasks_raft = manifest[
    manifest["well"].astype(str).isin(valid_wells_raft)
    & (manifest["tag"].astype(str) == "official")
].copy()
train_keys_raft = [
    _panel_key(str(row.well), str(row.tag))
    for row in train_tasks_raft.itertuples(index=False)
]
valid_keys_raft = [
    _panel_key(str(row.well), str(row.tag))
    for row in valid_tasks_raft.itertuples(index=False)
]

# Release meta-support panels before loading the full outer-train bank.
for stale_name in ("support_store", "posterior_model", "active_model"):
    stale = globals().pop(stale_name, None)
    if stale is not None:
        del stale
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

train_store_raft = PanelStore(train_keys_raft)
valid_store_raft = PanelStore(valid_keys_raft)


class RAFTPanelDataset(Dataset):
    def __init__(self, store, mode="full"):
        self.store = store
        self.mode = mode

    def __len__(self):
        return len(self.store.records)

    def __getitem__(self, index):
        record = self.store.records[index]
        horizontal = record["horizontal_gr"].copy()
        horizontal_valid = record["horizontal_valid"].copy()
        typewell = record["typewell_gr"].copy()
        typewell_valid = record["typewell_valid"].copy()
        heel = record["heel_reference_gr"].copy()
        heel_valid = record["heel_reference_valid"].copy()
        if self.mode == "shuffled_typewell":
            donor = self.store.records[
                (index + max(1, len(self.store.records) // 3)) % len(self.store.records)
            ]
            typewell = donor["typewell_gr"].copy()
            typewell_valid = donor["typewell_valid"].copy()
        elif self.mode == "no_gr":
            horizontal[:] = 0.0
            typewell[:] = 0.0
            heel[:] = 0.0
        elif self.mode != "full":
            raise ValueError(self.mode)
        return {
            "horizontal": torch.from_numpy(horizontal),
            "horizontal_valid": torch.from_numpy(horizontal_valid),
            "typewell": torch.from_numpy(typewell),
            "typewell_valid": torch.from_numpy(typewell_valid),
            "heel": torch.from_numpy(heel),
            "heel_valid": torch.from_numpy(heel_valid),
            "geometry": torch.from_numpy(record["geometry"]),
            "truth_residual": torch.from_numpy(record["truth_residual"]),
            "eval_mask": torch.from_numpy(record["eval_mask"]),
            "column_md": torch.from_numpy(record["column_md"]),
            "column_z": torch.from_numpy(record["column_z"]),
            "panel_weight": torch.tensor(record["panel_weight"], dtype=torch.float32),
        }


class ConvResidual1D(nn.Module):
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
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return F.gelu(values + self.block(values))


class AxialContextEncoder(nn.Module):
    def __init__(self, input_channels, channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels, 5, padding=2),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            ConvResidual1D(channels, 1),
            ConvResidual1D(channels, 2),
            ConvResidual1D(channels, 4),
            ConvResidual1D(channels, 8),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=channels * 3,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.axial = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, values):
        hidden = self.tcn(self.stem(values))
        hidden = self.axial(hidden.transpose(1, 2)).transpose(1, 2)
        return self.output(hidden)


class ConvGRU1D(nn.Module):
    def __init__(self, hidden_channels, input_channels):
        super().__init__()
        self.gates = nn.Conv1d(
            hidden_channels + input_channels, hidden_channels * 2, 5, padding=2
        )
        self.candidate = nn.Conv1d(
            hidden_channels + input_channels, hidden_channels, 5, padding=2
        )

    def forward(self, hidden, values):
        combined = torch.cat([hidden, values], dim=1)
        reset, update = torch.sigmoid(self.gates(combined)).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat([reset * hidden, values], dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class AxialRAFTTVT(nn.Module):
    def __init__(self, geometry_channels):
        super().__init__()
        context_input = 2 + geometry_channels
        self.context = AxialContextEncoder(context_input, HIDDEN_CHANNELS)
        self.cost_encoder = nn.Sequential(
            nn.Conv2d(10, 24, 3, padding=1),
            nn.GroupNorm(8, 24),
            nn.GELU(),
            nn.Conv2d(24, COST_CHANNELS, 3, padding=1),
            nn.GroupNorm(8, COST_CHANNELS),
            nn.GELU(),
        )
        sampled_channels = COST_CHANNELS * len(LOCAL_OFFSETS_FT)
        update_input = HIDDEN_CHANNELS + sampled_channels + 3
        self.update_projection = nn.Sequential(
            nn.Conv1d(update_input, HIDDEN_CHANNELS * 2, 1),
            nn.GELU(),
            ConvResidual1D(HIDDEN_CHANNELS * 2, 1),
            nn.Conv1d(HIDDEN_CHANNELS * 2, HIDDEN_CHANNELS, 1),
        )
        self.gru = ConvGRU1D(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.delta_head = nn.Sequential(
            ConvResidual1D(HIDDEN_CHANNELS, 1),
            nn.Conv1d(HIDDEN_CHANNELS, 1, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    @staticmethod
    def _raw_cost(horizontal, horizontal_valid, typewell, typewell_valid, heel, heel_valid):
        h = horizontal[:, None, None, :].expand_as(typewell[:, None])[:, 0]
        hv = horizontal_valid[:, None, :].expand_as(typewell)
        return torch.stack(
            [
                h,
                hv,
                typewell,
                typewell_valid,
                h * typewell,
                -torch.abs(h - typewell),
                heel,
                heel_valid,
                h * heel,
                -torch.abs(h - heel),
            ],
            dim=1,
        )

    @staticmethod
    def _sample_local(cost, correction):
        batch, channels, states, columns = cost.shape
        x = torch.linspace(-1.0, 1.0, columns, device=cost.device)
        sampled = []
        state_min = float(STATE_OFFSETS_RAFT[0])
        state_span = float(STATE_OFFSETS_RAFT[-1] - STATE_OFFSETS_RAFT[0])
        for local_offset in LOCAL_OFFSETS_FT:
            candidate = correction + float(local_offset)
            y = 2.0 * (candidate - state_min) / state_span - 1.0
            grid = torch.stack(
                [x[None].expand(batch, -1), y], dim=-1
            )[:, None]
            values = F.grid_sample(
                cost,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled.append(values[:, :, 0])
        return torch.cat(sampled, dim=1)

    def forward(
        self,
        horizontal,
        horizontal_valid,
        typewell,
        typewell_valid,
        heel,
        heel_valid,
        geometry,
        eval_mask,
    ):
        context_values = torch.cat(
            [horizontal[:, None], horizontal_valid[:, None], geometry], dim=1
        )
        context = self.context(context_values)
        hidden = context
        cost = self.cost_encoder(
            self._raw_cost(
                horizontal,
                horizontal_valid,
                typewell,
                typewell_valid,
                heel,
                heel_valid,
            )
        )
        correction = torch.zeros_like(horizontal)
        predictions = []
        progress = torch.linspace(0.0, 1.0, horizontal.shape[1], device=horizontal.device)
        progress = progress[None].expand_as(horizontal)
        for _ in range(RAFT_ITERATIONS):
            local_cost = self._sample_local(cost, correction)
            update_values = torch.cat(
                [
                    context,
                    local_cost,
                    correction[:, None] / 96.0,
                    eval_mask[:, None],
                    progress[:, None],
                ],
                dim=1,
            )
            update_values = self.update_projection(update_values)
            hidden = self.gru(hidden, update_values)
            delta = 8.0 * torch.tanh(self.delta_head(hidden).squeeze(1))
            correction = torch.clamp(correction + delta, -96.0, 96.0)
            correction = correction * eval_mask
            predictions.append(correction)
        return predictions


def _trajectory_loss(predictions, truth, eval_mask, column_md, column_z, panel_weight):
    denominator = eval_mask.sum(dim=1).clamp_min(1.0)
    panel_scale = (panel_weight / panel_weight.mean().clamp_min(1.0)).clamp(0.2, 5.0)
    total = truth.new_tensor(0.0)
    component_rows = []
    for iteration, prediction in enumerate(predictions):
        squared = ((prediction - truth).square() * eval_mask).sum(dim=1) / denominator
        robust = (
            F.smooth_l1_loss(prediction, truth, beta=2.0, reduction="none")
            * eval_mask
        ).sum(dim=1) / denominator
        predicted_s = prediction + column_z
        true_s = truth + column_z
        dmd = (column_md[:, 1:] - column_md[:, :-1]).clamp_min(1e-3)
        predicted_dip = (predicted_s[:, 1:] - predicted_s[:, :-1]) / dmd
        true_dip = (true_s[:, 1:] - true_s[:, :-1]) / dmd
        pair_mask = torch.minimum(eval_mask[:, 1:], eval_mask[:, :-1])
        slope = (
            F.smooth_l1_loss(
                predicted_dip, true_dip, beta=0.01, reduction="none"
            )
            * pair_mask
        ).sum(dim=1) / pair_mask.sum(dim=1).clamp_min(1.0)
        per_panel = squared / 64.0 + 0.20 * robust + 0.03 * slope
        iteration_loss = (per_panel * panel_scale).mean()
        weight = ITERATION_GAMMA ** (len(predictions) - iteration - 1)
        total = total + weight * iteration_loss
        component_rows.append((squared.mean(), robust.mean(), slope.mean()))
    normaliser = sum(
        ITERATION_GAMMA ** (len(predictions) - iteration - 1)
        for iteration in range(len(predictions))
    )
    final_squared, final_robust, final_slope = component_rows[-1]
    return total / normaliser, {
        "mse": final_squared,
        "robust": final_robust,
        "slope": final_slope,
    }


def _model_forward(model, batch):
    return model(
        batch["horizontal"].to(device_raft, non_blocking=True),
        batch["horizontal_valid"].to(device_raft, non_blocking=True),
        batch["typewell"].to(device_raft, non_blocking=True),
        batch["typewell_valid"].to(device_raft, non_blocking=True),
        batch["heel"].to(device_raft, non_blocking=True),
        batch["heel_valid"].to(device_raft, non_blocking=True),
        batch["geometry"].to(device_raft, non_blocking=True),
        batch["eval_mask"].to(device_raft, non_blocking=True),
    )


@torch.no_grad()
def _evaluate_raft(model, store, mode="full", return_rows=False):
    model.eval()
    dataset = RAFTPanelDataset(store, mode=mode)
    errors = []
    center_errors = []
    by_well = []
    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        batch = {
            key: value[None] for key, value in item.items() if key != "panel_weight"
        }
        with torch.autocast(
            device_type=device_raft.type,
            dtype=torch.float16,
            enabled=(device_raft.type == "cuda"),
        ):
            correction = _model_forward(model, batch)[-1][0].float().cpu().numpy()
        record = store.records[index]
        correction[:PREFIX_COLUMNS_RAFT] = 0.0
        column_s = record["center_tvt"] + correction + record["column_z"]
        prediction = (
            np.interp(record["eval_md"], record["column_md"], column_s)
            - record["eval_z"]
        )
        center_s = record["center_tvt"] + record["column_z"]
        center = (
            np.interp(record["eval_md"], record["column_md"], center_s)
            - record["eval_z"]
        )
        truth = record["eval_truth"].astype(np.float64)
        error = prediction - truth
        center_error = center - truth
        errors.append(error)
        center_errors.append(center_error)
        by_well.append(
            {
                "well": record["well"],
                "rows": len(error),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "center_rmse": float(np.sqrt(np.mean(center_error**2))),
            }
        )
        if return_rows:
            rows.append(
                pd.DataFrame(
                    {
                        "well": record["well"],
                        "row_idx": record["eval_rows"],
                        "truth": truth,
                        "center": center,
                        "prediction": prediction,
                    }
                )
            )
    errors = np.concatenate(errors)
    center_errors = np.concatenate(center_errors)
    by_well = pd.DataFrame(by_well)
    metrics = {
        "rows": int(len(errors)),
        "wells": int(len(by_well)),
        "center_rmse": float(np.sqrt(np.mean(center_errors**2))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "gain_vs_center": float(
            np.sqrt(np.mean(center_errors**2)) - np.sqrt(np.mean(errors**2))
        ),
        "mae": float(np.mean(np.abs(errors))),
        "median_well_rmse": float(by_well["rmse"].median()),
        "p95_well_rmse": float(by_well["rmse"].quantile(0.95)),
    }
    row_frame = pd.concat(rows, ignore_index=True) if return_rows else None
    return metrics, by_well, row_frame


def _train_raft():
    _seed_raft(SEED_RAFT)
    geometry_channels = int(train_store_raft.records[0]["geometry"].shape[0])
    model = AxialRAFTTVT(geometry_channels).to(device_raft)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_RAFT, weight_decay=WEIGHT_DECAY_RAFT
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_RAFT, eta_min=MIN_LEARNING_RATE_RAFT
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device_raft.type == "cuda"))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(device_raft.type == "cuda"))
    dataset = RAFTPanelDataset(train_store_raft, mode="full")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_RAFT,
        shuffle=True,
        num_workers=0,
        pin_memory=(device_raft.type == "cuda"),
    )
    checkpoint_path = MODEL_DIR_RAFT / f"seed_{SEED_RAFT}.pt"
    history = []
    best_rmse = np.inf
    stale = 0
    started = time.time()
    for epoch in range(1, EPOCHS_RAFT + 1):
        model.train()
        batch_rows = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device_raft.type,
                dtype=torch.float16,
                enabled=(device_raft.type == "cuda"),
            ):
                predictions = _model_forward(model, batch)
            loss, components = _trajectory_loss(
                [prediction.float() for prediction in predictions],
                batch["truth_residual"].to(device_raft),
                batch["eval_mask"].to(device_raft),
                batch["column_md"].to(device_raft),
                batch["column_z"].to(device_raft),
                batch["panel_weight"].to(device_raft),
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    **{
                        key: float(value.detach().cpu())
                        for key, value in components.items()
                    },
                }
            )
        scheduler.step()
        metrics, by_well, _ = _evaluate_raft(model, valid_store_raft, mode="full")
        record = {
            "epoch": epoch,
            **{
                f"train_{key}": float(np.mean([row[key] for row in batch_rows]))
                for key in batch_rows[0]
            },
            **{f"valid_{key}": value for key, value in metrics.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(record)
        print("AXIAL-RAFT EPOCH", record, flush=True)
        if metrics["rmse"] < best_rmse - 1e-4:
            best_rmse = float(metrics["rmse"])
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "geometry_channels": geometry_channels,
                    "seed": SEED_RAFT,
                },
                checkpoint_path,
            )
            by_well.to_csv(OUT_DIR_RAFT / "best_bywell.csv", index=False)
        else:
            stale += 1
        if epoch >= 8 and stale >= EARLY_STOPPING_RAFT:
            print(f"axial RAFT early stopping at epoch {epoch}", flush=True)
            break
    pd.DataFrame(history).to_csv(OUT_DIR_RAFT / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device_raft, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics, by_well, rows = _evaluate_raft(
        model, valid_store_raft, mode="full", return_rows=True
    )
    by_well.to_csv(OUT_DIR_RAFT / "final_bywell.csv", index=False)
    rows.to_pickle(OUT_DIR_RAFT / "fold1_predictions.pkl")
    return model, checkpoint, {
        "best_epoch": int(checkpoint["epoch"]),
        **metrics,
        "runtime_minutes": (time.time() - started) / 60.0,
    }


_seed_raft(SEED_RAFT)
probe = AxialRAFTTVT(int(train_store_raft.records[0]["geometry"].shape[0])).to(device_raft)
parameters_raft = int(sum(parameter.numel() for parameter in probe.parameters()))
probe_dataset = RAFTPanelDataset(train_store_raft, mode="full")
probe_batch = {
    key: value[None].to(device_raft)
    for key, value in probe_dataset[0].items()
    if key != "panel_weight"
}
with torch.no_grad():
    probe_predictions = _model_forward(probe, probe_batch)
if len(probe_predictions) != RAFT_ITERATIONS or not all(
    torch.isfinite(prediction).all() for prediction in probe_predictions
):
    raise RuntimeError("axial RAFT forward guard failed")
del probe, probe_predictions, probe_dataset, probe_batch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(
    {
        "experiment": "tvt_corrnet_axial_raft_onefold_v0",
        "device": str(device_raft),
        "train_wells": len(train_wells_raft),
        "train_panels": len(train_store_raft.records),
        "valid_wells": len(valid_store_raft.records),
        "states": len(STATE_OFFSETS_RAFT),
        "columns": N_COLUMNS_RAFT,
        "iterations": RAFT_ITERATIONS,
        "parameters": parameters_raft,
        "standalone": True,
    },
    flush=True,
)

model_raft, checkpoint_raft, result_raft = _train_raft()
intervention_rows = []
for mode in ("full", "no_gr", "shuffled_typewell"):
    metrics, _, rows = _evaluate_raft(
        model_raft, valid_store_raft, mode=mode, return_rows=True
    )
    rows.to_pickle(OUT_DIR_RAFT / f"intervention_{mode}.pkl")
    intervention_rows.append({"mode": mode, **metrics})
interventions = pd.DataFrame(intervention_rows)
full_rmse = float(interventions.loc[interventions["mode"] == "full", "rmse"].iloc[0])
interventions["degradation_vs_full"] = interventions["rmse"] - full_rmse
interventions.to_csv(OUT_DIR_RAFT / "interventions.csv", index=False)
no_gr_degradation = float(
    interventions.loc[interventions["mode"] == "no_gr", "degradation_vs_full"].iloc[0]
)
shuffle_degradation = float(
    interventions.loc[
        interventions["mode"] == "shuffled_typewell", "degradation_vs_full"
    ].iloc[0]
)
architecture_pass = bool(result_raft["rmse"] <= 7.0)
target_pass = bool(result_raft["rmse"] <= 5.5)
causal_pass = bool(no_gr_degradation >= 0.30 and shuffle_degradation >= 0.30)
verdict_raft = {
    "experiment": "tvt_corrnet_axial_raft_onefold_v0",
    "sample_unit": "complete panel/well state",
    "standalone": True,
    "fold": VALID_FOLD_RAFT,
    "train_wells": len(train_wells_raft),
    "train_panels": len(train_store_raft.records),
    "valid_wells": len(valid_store_raft.records),
    "parameters": parameters_raft,
    "result": result_raft,
    "no_gr_degradation": no_gr_degradation,
    "shuffled_typewell_degradation": shuffle_degradation,
    "architecture_pass": architecture_pass,
    "target_5p5_pass": target_pass,
    "causal_pass": causal_pass,
    "strict_pass": bool(architecture_pass and causal_pass),
    "next": (
        "TARGET PASS -> run second seed and five folds"
        if target_pass and causal_pass
        else (
            "ARCHITECTURE PASS -> run second seed before synthetic pretraining"
            if architecture_pass and causal_pass
            else "FAIL -> close S3 and proceed to S4 synthetic pretraining gateway"
        )
    ),
}
with open(OUT_DIR_RAFT / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict_raft, handle, indent=2)

print("\nRESULT")
print(json.dumps(result_raft, indent=2))
print("\nINTERVENTIONS")
print(interventions.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict_raft, indent=2))
print("saved:", OUT_DIR_RAFT)
