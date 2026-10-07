from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ReproductionConfig, VERSION_RECIPES
from .data import build_base_panel, load_record
from .infer import decode_well
from .loss import AlignmentLoss
from .model import ARCHITECTURE, AlignmentUNet


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold-file", required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--limit-valid", type=int)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise RuntimeError(
            {
                "message": "checkpoint architecture mismatch",
                "expected": ARCHITECTURE,
                "found": checkpoint.get("architecture"),
            }
        )
    version = checkpoint["version"]
    recipe = VERSION_RECIPES[version]
    if recipe.use_pf or recipe.use_xy:
        raise NotImplementedError("checkpoint audit currently targets the default branch")
    folds = pd.read_csv(args.fold_file, dtype={"well": str})
    wells = folds.loc[folds[f"repeat_{args.repeat}"] == args.fold, "well"].tolist()
    if args.limit_valid:
        wells = wells[: args.limit_valid]
    config = ReproductionConfig()
    first = build_base_panel(load_record(args.data_root, "train", wells[0]), config.grid, config.loss.target_sigma_ft)
    model = AlignmentUNet(len(first.channels), pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    criterion = AlignmentLoss(config.loss)
    bin_sse = row_sse = 0.0
    bin_count = row_count = 0
    components = {"cross_entropy": [], "path_huber": [], "gr_penalty": []}
    by_well = []
    for well in wells:
        record = load_record(args.data_root, "train", well)
        sample = build_base_panel(record, config.grid, config.loss.target_sigma_ft)
        values = torch.from_numpy(sample.values)[None].to(device, memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(values)
        loss, parts = criterion(
            logits,
            torch.from_numpy(sample.target_probability)[None].to(device),
            torch.from_numpy(sample.target_offset)[None].to(device),
            torch.from_numpy(sample.target_mask)[None].to(device),
            torch.from_numpy(sample.state_offsets)[None].to(device),
            torch.from_numpy(sample.gr_gap)[None].to(device),
        )
        for name in components:
            components[name].append(float(parts[name]))
        expected = parts["expected"][0].cpu().numpy()
        mask = sample.target_mask
        bin_error = expected[mask] - sample.target_offset[mask]
        bin_sse += float(np.sum(bin_error**2))
        bin_count += len(bin_error)
        frame = decode_well(record, sample, expected)
        rows = frame.row.to_numpy(dtype=int)
        truth = record.truth[rows]
        prediction = frame.tvt.to_numpy()
        finite = np.isfinite(truth) & np.isfinite(prediction)
        error = prediction[finite] - truth[finite]
        row_sse += float(np.sum(error**2))
        row_count += len(error)
        by_well.append({"well": well, "rows": len(error), "rmse": float(np.sqrt(np.mean(error**2)))})
    verdict = {
        "version": version,
        "wells": len(wells),
        "bin_rmse": float(np.sqrt(bin_sse / bin_count)),
        "row_rmse": float(np.sqrt(row_sse / row_count)),
        "rows": row_count,
        "loss_components": {name: float(np.mean(values)) for name, values in components.items()},
        "checkpoint_metrics": checkpoint.get("metrics"),
    }
    print("CHECKPOINT AUDIT", json.dumps(verdict, indent=2), flush=True)
    output = Path(args.checkpoint).with_name("checkpoint_audit.json")
    output.write_text(json.dumps({**verdict, "by_well": by_well}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
