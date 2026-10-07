from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from .config import GRID, TRAINING, VERSIONS
from .contract import assert_clean_room, validate_reproduction_matrix
from .data import build_base_panel, build_target, load_record
from .loss import AlignmentLoss
from .model import AlignmentUNet
from .normalization import PopulationStats
from .train import decode_row_offsets


def _write_synthetic_well(root: Path, well: str = "synthetic") -> None:
    folder = root / "train"
    folder.mkdir(parents=True, exist_ok=True)
    rows = 3200
    md = np.arange(rows, dtype=np.float32)
    z = 1000 - .02 * md
    truth = 25 + .01 * md + 2 * np.sin(md / 350)
    gr = 80 + 25 * np.sin(truth / 8) + 8 * np.cos(truth / 2.3)
    visible = np.full(rows, np.nan, np.float32)
    visible[:1024] = truth[:1024]
    pd.DataFrame({
        "MD": md, "X": md, "Y": md * .2, "Z": z, "GR": gr,
        "TVT_input": visible, "TVT": truth,
    }).to_csv(folder / f"{well}__horizontal_well.csv", index=False)
    tvt = np.arange(-100, 200, .25, dtype=np.float32)
    tw_gr = 80 + 25 * np.sin(tvt / 8) + 8 * np.cos(tvt / 2.3)
    pd.DataFrame({"TVT": tvt, "GR": tw_gr}).to_csv(
        folder / f"{well}__typewell.csv", index=False
    )


def data_and_loss_gate() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _write_synthetic_well(root)
        record = load_record(root, "train", "synthetic")
        stats = PopulationStats(80.0, 25.0)
        panel = build_base_panel(record, population_stats=stats)
    assert panel.values.shape == (16, 345, 400)
    assert panel.target_probability.shape == (345, 400)
    assert np.isfinite(panel.values).all()
    z_diff = panel.values[panel.channels.index("z_diff")]
    assert float(np.std(z_diff, axis=1).max()) < 1e-7
    assert float(np.std(z_diff, axis=0).mean()) > 0.01
    supervised = np.flatnonzero(panel.target_mask)
    assert len(supervised)
    column = int(supervised[len(supervised) // 2])
    peak = int(panel.target_probability[column].argmax())
    nearest = int(np.abs(panel.state_offsets - panel.target_offset[column]).argmin())
    assert abs(peak - nearest) <= 1
    target = torch.from_numpy(panel.target_probability[None])
    logits_good = torch.log(target.clamp_min(1e-8))
    logits_bad = torch.roll(logits_good, 24, dims=-1)
    args = (
        target,
        torch.from_numpy(panel.target_offset[None]),
        torch.from_numpy(panel.target_mask[None]),
        torch.from_numpy(panel.state_offsets[None]),
        torch.from_numpy(panel.gr_gap[None]),
    )
    loss = AlignmentLoss()
    good, _ = loss(logits_good, *args)
    bad, _ = loss(logits_bad, *args)
    assert torch.isfinite(good) and good < bad
    padded_z = np.full(GRID.raster_rows, np.nan, dtype=np.float32)
    padded_z[:3200] = np.linspace(1000, 900, 3200)
    decoded = decode_row_offsets(
        np.zeros(GRID.horizontal_bins, np.float32), padded_z, 25.0, 1025, .6828376753426914
    )
    assert np.nanmax(np.abs(decoded[:3200] - 25.0)) < 1e-4
    assert np.isnan(decoded[3200:]).all()
    return {
        "shape": list(panel.values.shape),
        "state_first": float(panel.state_offsets[0]),
        "state_last": float(panel.state_offsets[-1]),
        "good_loss": float(good),
        "shifted_loss": float(bad),
        "padding_decode_max_gap": float(np.nanmax(np.abs(decoded[:3200] - 25.0))),
        "z_diff_state_std": float(np.std(z_diff, axis=1).max()),
        "z_diff_horizontal_std": float(np.std(z_diff, axis=0).mean()),
    }


def model_gate(skip_model: bool = False) -> dict[str, object]:
    if skip_model:
        return {"skipped": True}
    model = AlignmentUNet(16, pretrained=False)
    # Reduced spatial tensor checks graph semantics without changing the model.
    values = torch.randn(1, 16, 96, 128)
    output = model(values)
    assert output.shape == (1, 96, 128)
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "checkpoint.pt"
        torch.save(model.state_dict(), path)
        restored = AlignmentUNet(16, pretrained=False)
        restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        restored.eval()
        model.eval()
        with torch.no_grad():
            gap = float((model(values) - restored(values)).abs().max())
        assert gap == 0.0
    return {"output_shape": list(output.shape), "parameters": sum(p.numel() for p in model.parameters()), "checkpoint_max_gap": gap}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    counts = validate_reproduction_matrix(args.project_root / "reproduction-matrix.csv")
    assert_clean_room(args.project_root)
    report = {
        "level": "behavioral reproduction",
        "contract": counts,
        "data_loss": data_and_loss_gate(),
        "model": model_gate(args.skip_model),
        "versions": {name: list(recipe.channels) for name, recipe in VERSIONS.items()},
        "reference": {"0719_V1_oof_rmse": 5.091, "ensemble_oof_rmse": 4.627},
        "status": "COMPONENT_PARITY" if not args.skip_model else "PARTIAL_PARITY",
    }
    print("VALIDATION", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
