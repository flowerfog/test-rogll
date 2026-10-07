from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .contract import validate_contract
from .data import build_base_panel, list_wells, load_record
from .loss import AlignmentLoss
from .model import AlignmentUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--matrix", default=str(Path(__file__).resolve().parents[1] / "reproduction-matrix.csv"))
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    validate_contract(args.matrix)
    if not torch.cuda.is_available():
        raise RuntimeError("AutoDL validation requires CUDA")
    well = list_wells(args.data_root, "train")[0]
    sample = build_base_panel(load_record(args.data_root, "train", well))
    audit = {
        "well": well,
        "shape": list(sample.values.shape),
        "channels": list(sample.channels),
        "finite": bool(np.isfinite(sample.values).all()),
        "axis_coverage": float(sample.target_mask.mean()),
    }
    print("DATA CONTRACT", json.dumps(audit), flush=True)
    model = AlignmentUNet(
        len(sample.channels),
        pretrained=args.encoder_checkpoint is None,
        encoder_checkpoint=args.encoder_checkpoint,
    ).cuda().to(memory_format=torch.channels_last)
    criterion = AlignmentLoss()
    # Use a real anchor-centered crop for a fast learning gate; production keeps 345x400.
    horizontal = slice(16, 48)
    state_slice = slice(120, 280)
    values = torch.from_numpy(sample.values[:, horizontal, state_slice])[None].cuda(
        memory_format=torch.channels_last
    )
    target = torch.from_numpy(sample.target_probability[horizontal, state_slice])[None].cuda()
    denominator = target.sum(-1, keepdim=True)
    mask = torch.from_numpy(sample.target_mask[horizontal])[None].cuda() & (denominator[..., 0] > 0)
    target = torch.where(denominator > 0, target / denominator.clamp_min(1e-12), target)
    truth = torch.from_numpy(sample.target_offset[horizontal])[None].cuda()
    gap = torch.from_numpy(sample.gr_gap[horizontal, state_slice])[None].cuda()
    state = torch.from_numpy(sample.state_offsets[state_slice])[None].cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(values)
            loss, _ = criterion(logits, target, truth, mask, state, gap)
        if not torch.isfinite(loss):
            raise RuntimeError({"message": "nonfinite loss", "step": step})
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        print("OVERFIT", {"step": step, "loss": losses[-1]}, flush=True)
    ratio = min(losses[-5:]) / losses[0]
    if ratio >= 0.8:
        raise RuntimeError({"message": "overfit gate failed", "ratio": ratio, "losses": losses})
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = model(torch.from_numpy(sample.values)[None].cuda(memory_format=torch.channels_last))
    if full.shape != (1, 345, 400) or not torch.isfinite(full).all():
        raise RuntimeError({"message": "full forward gate failed", "shape": list(full.shape)})
    print(
        "AUTODL VALIDATION PASS",
        json.dumps({"overfit_ratio": ratio, "full_shape": list(full.shape), "parameters": sum(p.numel() for p in model.parameters())}),
        flush=True,
    )


if __name__ == "__main__":
    main()
