from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import HEIGHT, VERSION_BY_NAME, WIDTH
from .data import build_panel, list_wells, load_well
from .loss import weighted_huber_ce
from .model import AlignmentUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--version", default="0719_V1", choices=tuple(VERSION_BY_NAME))
    parser.add_argument("--model-forward", action="store_true")
    parser.add_argument("--network-layout", choices=("state_horizontal", "horizontal_state"),
                        default="state_horizontal")
    args = parser.parse_args()
    version = VERSION_BY_NAME[args.version]
    well = load_well(args.data_root, list_wells(args.data_root)[0])
    # Auxiliary channels are audited by prepare.py; base representation is audited here.
    if any(name.startswith(("pf_", "geo_")) for name in version.channels):
        raise ValueError("Use a default version for the representation audit")
    panel = build_panel(well, version.channels)
    assert panel.values.shape == (len(version.channels), HEIGHT, WIDTH)
    assert panel.target.shape == (HEIGHT, WIDTH)
    assert np.isfinite(panel.values).all()
    assert np.allclose(panel.target[:, panel.target_mask].sum(0), 1, atol=1e-5)
    truth_index = np.abs(panel.state_tvt[:, None] - panel.truth_tvt[None]).argmin(0)
    perfect = torch.full((1, HEIGHT, WIDTH), -12.0)
    shifted = torch.full_like(perfect, -12.0)
    columns = torch.arange(WIDTH)
    perfect[0, truth_index, columns] = 12
    shifted[0, np.clip(truth_index + 20, 0, HEIGHT - 1), columns] = 12
    target = torch.from_numpy(panel.target)[None]
    mask = torch.from_numpy(panel.target_mask)[None]
    state = torch.from_numpy(panel.state_tvt)[None]
    truth = torch.from_numpy(panel.truth_tvt)[None]
    gap = torch.from_numpy(panel.gr_gap)[None]
    perfect_loss, _ = weighted_huber_ce(perfect, target, mask, state, truth, gap)
    shifted_loss, _ = weighted_huber_ce(shifted, target, mask, state, truth, gap)
    assert perfect_loss < shifted_loss, (perfect_loss, shifted_loss)
    report = {"well": well.well, "version": version.name, "shape": panel.values.shape,
              "target_columns": int(panel.target_mask.sum()), "perfect_loss": float(perfect_loss),
              "shifted_loss": float(shifted_loss)}
    if args.model_forward:
        model = AlignmentUNet(len(version.channels), layout=args.network_layout).eval()
        with torch.no_grad():
            output = model(torch.from_numpy(panel.values)[None])
        assert output.shape == (1, HEIGHT, WIDTH)
        report["parameters"] = sum(p.numel() for p in model.parameters())
        report["forward_finite"] = bool(torch.isfinite(output).all())
    print("AUDIT", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
