from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

from .config import ReproductionConfig, VERSION_RECIPES
from .data import append_pf_channels, append_xy_channels, build_base_panel, list_wells, load_record, normalize_panel
from .model import ARCHITECTURE, CONTRACT_VERSION, AlignmentUNet
from .normalization import PopulationStats
from .pf import MultiProfileParticleFilter
from .xy import XYSurfacePrior


def load_member(path: str | Path, in_channels: int, device: torch.device) -> AlignmentUNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise RuntimeError(
            {
                "message": "checkpoint architecture mismatch",
                "path": str(path),
                "expected": ARCHITECTURE,
                "found": checkpoint.get("architecture"),
            }
        )
    if checkpoint.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError({
            "message": "checkpoint data/training contract mismatch",
            "path": str(path),
            "expected": CONTRACT_VERSION,
            "found": checkpoint.get("contract_version"),
        })
    recipe = VERSION_RECIPES[checkpoint["version"]]
    model = AlignmentUNet(in_channels, pretrained=False, dropout=recipe.dropout)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict_panel(models: list[AlignmentUNet], values: np.ndarray, offsets: np.ndarray, device) -> np.ndarray:
    tensor = torch.from_numpy(values[None]).to(device, memory_format=torch.channels_last)
    probability = None
    for model in models:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            current = model(tensor).float().softmax(dim=-1)
        probability = current if probability is None else probability + current
    probability = probability / len(models)
    state = torch.from_numpy(offsets).to(device)
    return (probability[0] * state[None]).sum(dim=-1).cpu().numpy().astype(np.float32)


def decode_well(record, sample, bin_offset: np.ndarray, smooth_window: int,
                smooth_blend: float) -> pd.DataFrame:
    selected = sample.selected_rows
    valid = (selected >= 0) & (selected > record.anchor_index)
    expanded_all = np.repeat(bin_offset, len(selected) // len(bin_offset))
    expanded = expanded_all[valid]
    rows = selected[valid]
    prediction = sample.anchor_tvt + expanded
    if len(prediction) >= 7:
        window = min(smooth_window, len(prediction) if len(prediction) % 2 else len(prediction) - 1)
        window = max(5, window)
        surface = prediction + record.z[rows]
        smoothed = savgol_filter(surface, window_length=window, polyorder=1, mode="interp")
        smoothed_tvt = smoothed - record.z[rows]
        prediction = (
            smooth_blend * smoothed_tvt
            + (1.0 - smooth_blend) * prediction
        )
    return pd.DataFrame(
        {
            "well": record.well,
            "row": rows,
            "id": [f"{record.well}_{int(row)}" for row in rows],
            "tvt": prediction.astype(np.float32),
        }
    )


def discover_checkpoints(model_root: str | Path, version: str) -> list[Path]:
    paths = sorted(Path(model_root).glob(f"{version}/repeat*_fold*_seed*/best.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints for {version} under {model_root}")
    return paths


def predict_version(
    data_root: str | Path,
    model_root: str | Path,
    version: str,
    split: str = "test",
    xy_prior: XYSurfacePrior | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, bool]]:
    config = ReproductionConfig()
    recipe = VERSION_RECIPES[version]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs, reliability = {}, {}
    models = None
    checkpoint_paths = discover_checkpoints(model_root, version)
    first_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    population_stats = PopulationStats.from_dict(first_checkpoint["population_stats"])
    for well_index, well in enumerate(list_wells(data_root, split)):
        record = load_record(data_root, split, well)
        sample = build_base_panel(
            record, config.grid, config.loss.target_sigma_ft,
            population_stats=population_stats,
            apply_channel_normalization=False,
        )
        if recipe.use_pf:
            pf = MultiProfileParticleFilter(config.pf, 7919 + well_index).run(sample)
            sample = append_pf_channels(sample, pf.density, pf.ffbsi_offset)
        xy_result = None
        if recipe.use_xy:
            if xy_prior is None:
                raise ValueError("XY version requires all-train XY prior")
            xy_result = xy_prior.predict(record, sample)
            sample = append_xy_channels(sample, xy_result)
            reliability[well] = xy_result.reliable
        sample = sample.select(recipe.channels)
        sample = normalize_panel(sample, population_stats)
        if models is None:
            models = [load_member(path, len(sample.channels), device) for path in checkpoint_paths]
        bin_offset = predict_panel(models, sample.values, sample.state_offsets, device)
        outputs[well] = decode_well(
            record, sample, bin_offset, recipe.sg_window, recipe.sg_blend
        )
    return outputs, reliability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", choices=VERSION_RECIPES, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    xy_prior = None
    if VERSION_RECIPES[args.version].use_xy:
        xy_prior = XYSurfacePrior().fit_root(args.data_root)
    outputs, reliability = predict_version(
        args.data_root, args.model_root, args.version, args.split, xy_prior
    )
    frame = pd.concat(outputs.values(), ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    Path(args.output).with_suffix(".json").write_text(
        json.dumps({"version": args.version, "xy_reliable": reliability}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
