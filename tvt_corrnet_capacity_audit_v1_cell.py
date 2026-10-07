"""Capacity ladder for TVT-CorrNet before any held-out OOF run.

This cell reuses the 24 panels built by tvt_corrnet_overfit_v0_cell.py. It
trains progressively on 1, 4, and 24 wells. Each stage must reconstruct its
own training wells below the registered threshold before the next stage is
allowed to run.

Changes from v0:
* roughly 1M parameters instead of 128K;
* learnable correlation scale instead of a fixed 5x prior;
* CE-only warm-up, followed by weak TVT/S regularisation;
* lower learning rate, no weight decay, longer training;
* automatic early stop and ladder stop on failure.
"""

from pathlib import Path
import gc
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE_DIR = OOF_ROOT / "tvt_corrnet_overfit_v0"
PANEL_DIR = SOURCE_DIR / "panels"
OUT_DIR = OOF_ROOT / "tvt_corrnet_capacity_audit_v1"
MODEL_DIR = OUT_DIR / "models"

SCENARIOS = (
    {"name": "one_well", "wells": 1, "epochs": 600},
    {"name": "four_wells", "wells": 4, "epochs": 800},
    {"name": "twentyfour_wells", "wells": 24, "epochs": 1200},
)

EMBED_DIM = 64
BASE_CHANNELS = 48
BATCH_SIZE = 2
LEARNING_RATE = 5e-4
MIN_LEARNING_RATE = 5e-5
CE_ONLY_FRACTION = 0.60
REPORT_EVERY = 25
RANDOM_STATE = 20260717
TARGET_SIGMA_FT = 2.0

# Phase-two losses remain deliberately weak. The first question is whether
# the panel and decoder can represent the training paths at all.
LOSS_CE = 1.0
LOSS_TVT = 0.08
LOSS_S_SLOPE = 0.02
LOSS_S_CURVATURE = 0.01

POOLED_GATE_FT = 1.0
P95_WELL_GATE_FT = 2.0
EARLY_POOLED_FT = 0.85
EARLY_P95_WELL_FT = 1.70


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ordered_panel_paths():
    audit_path = SOURCE_DIR / "well_selection.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path, dtype={"well": str})
        flag = audit["selected"]
        if flag.dtype != bool:
            flag = flag.astype(str).str.lower().isin(["true", "1", "yes"])
        wells = audit.loc[flag, "well"].astype(str).tolist()
        paths = [PANEL_DIR / f"{well}.npz" for well in wells]
        paths = [path for path in paths if path.exists()]
    else:
        paths = sorted(PANEL_DIR.glob("*.npz"))
    if len(paths) < 24:
        raise FileNotFoundError(
            f"Need 24 v0 panels under {PANEL_DIR}; found {len(paths)}. "
            "Run tvt_corrnet_overfit_v0_cell.py first."
        )
    return paths[:24]


def _panel_guards(paths):
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "horizontal_patch",
                "typewell_patch",
                "geometry",
                "state_valid",
                "state_offsets",
                "column_md",
                "column_z",
                "truth_delta",
                "eval_md",
                "eval_truth",
                "anchor_tvt",
                "teacher_rmse",
                "label_coverage",
                "true_state_valid_fraction",
            }
            missing = required.difference(data.files)
            if missing:
                raise RuntimeError(f"{path.name} misses {sorted(missing)}")
            rows.append(
                {
                    "well": path.stem,
                    "teacher_rmse": float(data["teacher_rmse"][0]),
                    "label_coverage": float(data["label_coverage"][0]),
                    "true_state_valid_fraction": float(
                        data["true_state_valid_fraction"][0]
                    ),
                    "columns": int(data["horizontal_patch"].shape[0]),
                    "states": int(data["typewell_patch"].shape[0]),
                }
            )
    frame = pd.DataFrame(rows)
    summary = {
        "wells": len(frame),
        "teacher_decode_rmse": float(frame["teacher_rmse"].mean()),
        "min_label_coverage": float(frame["label_coverage"].min()),
        "min_true_state_valid": float(frame["true_state_valid_fraction"].min()),
        "columns": int(frame["columns"].iloc[0]),
        "states": int(frame["states"].iloc[0]),
    }
    if (
        summary["teacher_decode_rmse"] >= 1.0
        or summary["min_label_coverage"] < 0.995
        or summary["min_true_state_valid"] < 0.95
    ):
        raise RuntimeError(f"source panel guard failed: {summary}")
    return frame, summary


class CorrPanelDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            return {
                "horizontal": torch.from_numpy(data["horizontal_patch"]),
                "typewell": torch.from_numpy(data["typewell_patch"]),
                "geometry": torch.from_numpy(data["geometry"]),
                "state_valid": torch.from_numpy(
                    data["state_valid"].astype(bool)
                ),
                "truth_delta": torch.from_numpy(data["truth_delta"]),
                "column_z": torch.from_numpy(data["column_z"]),
            }


class PatchEncoder(nn.Module):
    def __init__(self, embedding=EMBED_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 32, 5, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 48, 5, stride=2, padding=2),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv1d(48, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, embedding, 3, padding=1),
            nn.GroupNorm(8, embedding),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, values):
        return self.network(values).squeeze(-1)


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.project = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
        )

    def forward(self, values):
        return F.gelu(self.block(values) + self.project(values))


class CorrNetV1(nn.Module):
    def __init__(self, geometry_channels):
        super().__init__()
        self.patch_encoder = PatchEncoder()
        self.correlation_scale = nn.Parameter(torch.tensor(0.5))
        input_channels = 2 + geometry_channels
        base = BASE_CHANNELS
        self.enc1 = ConvBlock(input_channels, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            ConvBlock(base * 4, base * 4),
            ConvBlock(base * 4, base * 4),
        )
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2 + base, base)
        self.head = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, horizontal, typewell, geometry, state_offsets):
        batch, columns, _, patch = horizontal.shape
        states = typewell.shape[1]
        horizontal_embedding = self.patch_encoder(
            horizontal.reshape(batch * columns, 2, patch)
        )
        typewell_embedding = self.patch_encoder(
            typewell.reshape(batch * states, 2, patch)
        )
        horizontal_embedding = F.normalize(
            horizontal_embedding.reshape(batch, columns, -1), dim=-1
        )
        typewell_embedding = F.normalize(
            typewell_embedding.reshape(batch, states, -1), dim=-1
        )
        correlation = torch.einsum(
            "bhd,bwd->bhw", typewell_embedding, horizontal_embedding
        )
        state_channel = state_offsets[None, :, None].expand(
            batch, states, columns
        )
        geometry_channels = geometry[:, :, None, :].expand(
            batch, geometry.shape[1], states, columns
        )
        values = torch.cat(
            [
                correlation[:, None],
                state_channel[:, None] / 128.0,
                geometry_channels,
            ],
            dim=1,
        )
        enc1 = self.enc1(values)
        enc2 = self.enc2(self.down1(enc1))
        bottleneck = self.bottleneck(self.down2(enc2))
        up2 = F.interpolate(
            bottleneck,
            size=enc2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = F.interpolate(
            dec2,
            size=enc1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        residual_logits = self.head(dec1).squeeze(1)
        return self.correlation_scale * correlation + residual_logits


def _loss(
    logits,
    truth_delta,
    column_z,
    state_valid,
    state_offsets,
    ce_only,
):
    valid_logits = logits.masked_fill(~state_valid[:, :, None], -30.0)
    distance = state_offsets[None, :, None] - truth_delta[:, None, :]
    target = torch.softmax(
        -0.5 * (distance / TARGET_SIGMA_FT) ** 2, dim=1
    )
    target = target * state_valid[:, :, None]
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_probability = F.log_softmax(valid_logits, dim=1)
    column_weight = torch.ones_like(truth_delta)
    column_weight[:, 0] = 4.0
    ce_by_column = -(target * log_probability).sum(dim=1)
    ce = (ce_by_column * column_weight).sum() / column_weight.sum()

    probability = torch.softmax(valid_logits, dim=1)
    prediction = (
        probability * state_offsets[None, :, None]
    ).sum(dim=1)
    tvt = F.smooth_l1_loss(prediction, truth_delta, beta=2.0)
    predicted_s = prediction + column_z - column_z[:, :1]
    true_s = truth_delta + column_z - column_z[:, :1]
    slope = F.smooth_l1_loss(
        predicted_s[:, 1:] - predicted_s[:, :-1],
        true_s[:, 1:] - true_s[:, :-1],
        beta=0.5,
    )
    curvature = F.smooth_l1_loss(
        predicted_s[:, 2:]
        - 2.0 * predicted_s[:, 1:-1]
        + predicted_s[:, :-2],
        true_s[:, 2:] - 2.0 * true_s[:, 1:-1] + true_s[:, :-2],
        beta=0.25,
    )
    if ce_only:
        total = ce
    else:
        total = (
            LOSS_CE * ce
            + LOSS_TVT * tvt
            + LOSS_S_SLOPE * slope
            + LOSS_S_CURVATURE * curvature
        )
    return total, {
        "ce": ce,
        "tvt": tvt,
        "slope": slope,
        "curvature": curvature,
    }


@torch.no_grad()
def _predict(model, panel_path, device):
    with np.load(panel_path, allow_pickle=False) as data:
        horizontal = torch.from_numpy(data["horizontal_patch"])[None].to(device)
        typewell = torch.from_numpy(data["typewell_patch"])[None].to(device)
        geometry = torch.from_numpy(data["geometry"])[None].to(device)
        state_valid = torch.from_numpy(
            data["state_valid"].astype(bool)
        )[None].to(device)
        state_offsets = torch.from_numpy(data["state_offsets"]).to(device)
        logits = model(
            horizontal, typewell, geometry, state_offsets
        ).masked_fill(~state_valid[:, :, None], -30.0)
        probability = torch.softmax(logits, dim=1)
        delta = (
            probability * state_offsets[None, :, None]
        ).sum(dim=1)[0].cpu().numpy()
        delta[0] = 0.0
        prediction = float(data["anchor_tvt"][0]) + np.interp(
            data["eval_md"].astype(float),
            data["column_md"].astype(float),
            delta,
        )
        truth = data["eval_truth"].astype(float)
    return prediction, truth


@torch.no_grad()
def _evaluate(model, paths, device):
    rows = []
    pooled_errors = []
    for path in paths:
        prediction, truth = _predict(model, path, device)
        error = prediction - truth
        pooled_errors.append(error)
        rows.append(
            {
                "well": path.stem,
                "rows": len(error),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
            }
        )
    pooled_errors = np.concatenate(pooled_errors)
    frame = pd.DataFrame(rows)
    metrics = {
        "pooled_rmse": float(np.sqrt(np.mean(pooled_errors**2))),
        "pooled_mae": float(np.mean(np.abs(pooled_errors))),
        "median_well_rmse": float(frame["rmse"].median()),
        "p95_well_rmse": float(frame["rmse"].quantile(0.95)),
    }
    return metrics, frame


def _run_scenario(config, paths, device, state_offsets, geometry_channels):
    name = config["name"]
    epochs = int(config["epochs"])
    scenario_paths = paths[: int(config["wells"])]
    _seed_everything(RANDOM_STATE + len(scenario_paths))
    dataset = CorrPanelDataset(scenario_paths)
    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    model = CorrNetV1(geometry_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=MIN_LEARNING_RATE
    )
    checkpoint_path = MODEL_DIR / f"{name}.pt"
    history = []
    best_rmse = np.inf
    warmup_epochs = int(round(epochs * CE_ONLY_FRACTION))
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_rows = []
        ce_only = epoch <= warmup_epochs
        for batch in loader:
            horizontal = batch["horizontal"].to(device, non_blocking=True)
            typewell = batch["typewell"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            state_valid = batch["state_valid"].to(device, non_blocking=True)
            truth_delta = batch["truth_delta"].to(device, non_blocking=True)
            column_z = batch["column_z"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                horizontal, typewell, geometry, state_offsets
            )
            loss, components = _loss(
                logits,
                truth_delta,
                column_z,
                state_valid,
                state_offsets,
                ce_only,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    **{
                        key: float(value.detach().cpu())
                        for key, value in components.items()
                    },
                }
            )
        scheduler.step()

        should_report = (
            epoch == 1
            or epoch % REPORT_EVERY == 0
            or epoch == warmup_epochs
            or epoch == warmup_epochs + 1
            or epoch == epochs
        )
        if not should_report:
            continue
        model.eval()
        metrics, by_well = _evaluate(model, scenario_paths, device)
        record = {
            "scenario": name,
            "wells": len(scenario_paths),
            "epoch": epoch,
            "phase": "ce_only" if ce_only else "joint",
            **{
                f"train_{key}": float(
                    np.mean([row[key] for row in loss_rows])
                )
                for key in loss_rows[0]
            },
            **metrics,
            "correlation_scale": float(
                model.correlation_scale.detach().cpu()
            ),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(record)
        print("epoch", record, flush=True)

        if metrics["pooled_rmse"] < best_rmse:
            best_rmse = metrics["pooled_rmse"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "correlation_scale": float(
                        model.correlation_scale.detach().cpu()
                    ),
                    "wells": [path.stem for path in scenario_paths],
                },
                checkpoint_path,
            )
            by_well.to_csv(
                OUT_DIR / f"{name}_best_bywell.csv", index=False
            )

        if (
            epoch >= max(100, warmup_epochs // 2)
            and metrics["pooled_rmse"] < EARLY_POOLED_FT
            and metrics["p95_well_rmse"] < EARLY_P95_WELL_FT
        ):
            print(f"{name}: early overfit gate cleared", flush=True)
            break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(OUT_DIR / f"{name}_history.csv", index=False)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    final_metrics, final_by_well = _evaluate(
        model, scenario_paths, device
    )
    final_by_well.to_csv(
        OUT_DIR / f"{name}_final_bywell.csv", index=False
    )
    passed = bool(
        final_metrics["pooled_rmse"] < POOLED_GATE_FT
        and final_metrics["p95_well_rmse"] < P95_WELL_GATE_FT
    )
    result = {
        "scenario": name,
        "wells": len(scenario_paths),
        "best_epoch": int(checkpoint["epoch"]),
        **final_metrics,
        "correlation_scale": float(checkpoint["correlation_scale"]),
        "parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "runtime_minutes": (time.time() - started) / 60.0,
        "pass": passed,
    }
    del model, optimizer, scheduler, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")

panel_paths = _ordered_panel_paths()
guard_frame, guards = _panel_guards(panel_paths)
guard_frame.to_csv(OUT_DIR / "panel_guards.csv", index=False)
with np.load(panel_paths[0], allow_pickle=False) as first:
    state_offsets = torch.from_numpy(first["state_offsets"]).to(DEVICE)
    geometry_channels = int(first["geometry"].shape[0])

print(
    {
        "experiment": "tvt_corrnet_capacity_audit_v1",
        "device": str(DEVICE),
        "source": str(SOURCE_DIR),
        "scenarios": SCENARIOS,
        "guards": guards,
        "base_channels": BASE_CHANNELS,
        "embedding": EMBED_DIM,
        "ce_only_fraction": CE_ONLY_FRACTION,
    }
)

results = []
for scenario in SCENARIOS:
    print("\nSTART", scenario, flush=True)
    result = _run_scenario(
        scenario,
        panel_paths,
        DEVICE,
        state_offsets,
        geometry_channels,
    )
    results.append(result)
    print("SCENARIO RESULT", result, flush=True)
    if not result["pass"]:
        print(
            f"STOP LADDER: {scenario['name']} did not clear the overfit gate",
            flush=True,
        )
        break

result_frame = pd.DataFrame(results)
result_frame.to_csv(OUT_DIR / "scenario_results.csv", index=False)
all_passed = bool(
    len(results) == len(SCENARIOS)
    and all(result["pass"] for result in results)
)
verdict = {
    "experiment": "tvt_corrnet_capacity_audit_v1",
    "device": str(DEVICE),
    "guards": guards,
    "results": results,
    "strict_pass": all_passed,
    "next": (
        "PASS -> build one held-out fold with negative controls"
        if all_passed
        else (
            "FAIL AT ONE WELL -> implementation/optimisation is invalid"
            if results and results[-1]["wells"] == 1
            else "FAIL AT SCALE -> representation does not identify all training paths"
        )
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2), encoding="utf-8"
)

print("\nSCENARIO RESULTS")
print(result_frame.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
