from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import HEIGHT, PF, SEEDS, STATE_AXIS, TRAIN, VERSIONS, WIDTH, XY
from .data import _AUGMENTATION, build_panel, list_wells, load_well
from .loss import weighted_huber_ce
from .model import AlignmentUNet


EXPECTED_CHANNELS = {
    "0719_V1": 16, "0724_V1": 17, "0729_V3": 18, "0801_V1": 16,
    "0801_V2": 16, "0802_V2": 18, "0803_V2": 20,
}
EXPECTED_WEIGHTS = {
    "0719_V1": (.07, .40), "0724_V1": (.28, 0.0), "0729_V3": (0.0, .20),
    "0801_V1": (.07, .40), "0801_V2": (.28, 0.0), "0802_V2": (0.0, 0.0),
    "0803_V2": (.28, 0.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--model-forward", action="store_true")
    args = parser.parse_args()

    assert (HEIGHT, WIDTH) == (400, 345)
    axis = np.asarray(STATE_AXIS)
    assert np.isclose(axis[0], -100) and np.isclose(axis[-1], 99.5)
    assert np.allclose(np.diff(axis), .5)
    assert tuple(SEEDS) == (0, 1, 2)
    assert (TRAIN.batch_size, TRAIN.learning_rate, TRAIN.min_learning_rate,
            TRAIN.weight_decay, TRAIN.ema_decay) == (16, 2e-4, 5e-7, .02, .99)
    assert (PF.particles, PF.seeds, PF.base_seed, PF.likelihood_scale, PF.update_rows) == (
        500, 128, 20_265_667, 6.0, 64)
    assert (XY.neighbor_wells, XY.point_neighbors, XY.idw_power) == (12, 112, 2.0)

    versions = {version.name: version for version in VERSIONS}
    assert set(versions) == set(EXPECTED_CHANNELS) == set(_AUGMENTATION)
    for name, count in EXPECTED_CHANNELS.items():
        version = versions[name]
        assert len(version.channels) == count, (name, version.channels)
        assert (version.general_weight, version.no_xy_weight) == EXPECTED_WEIGHTS[name]

    model = AlignmentUNet(len(versions["0719_V1"].channels))
    assert not any(isinstance(module, nn.LayerNorm) for module in model.modules())
    assert sum(isinstance(module, nn.BatchNorm2d) for module in model.modules()) > 30
    assert sum(isinstance(module, nn.AvgPool2d) for module in model.modules()) == 4
    if args.model_forward:
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 16, HEIGHT, WIDTH))
        assert output.shape == (1, HEIGHT, WIDTH) and torch.isfinite(output).all()

    well = load_well(args.data_root, list_wells(args.data_root, "train")[0], "train")
    panel = build_panel(well, versions["0719_V1"].channels)
    assert panel.values.shape == (16, HEIGHT, WIDTH) and np.isfinite(panel.values).all()
    columns = panel.target_mask
    assert np.allclose(panel.target[:, columns].sum(0), 1, atol=1e-5)
    logits = torch.from_numpy(np.log(np.maximum(panel.target, 1e-8)))[None]
    loss, terms = weighted_huber_ce(
        logits, torch.from_numpy(panel.target)[None], torch.from_numpy(panel.target_mask)[None],
        torch.from_numpy(panel.state_tvt)[None], torch.from_numpy(panel.truth_tvt)[None],
        torch.from_numpy(panel.gr_gap)[None])
    assert torch.isfinite(loss) and all(torch.isfinite(value) for value in terms.values())

    fold_report = None
    cache_report = None
    if args.cache_root:
        folds = pd.read_csv(args.cache_root / "folds.csv")
        sizes = folds.groupby("fold").size().sort_index().to_dict()
        assert len(folds) == 773 and set(sizes) == {1, 2, 3, 4, 5}
        assert max(sizes.values()) - min(sizes.values()) <= 1
        assert folds.geo_cluster.nunique() == 20
        fold_report = {str(key): int(value) for key, value in sizes.items()}
        train_wells = len(list_wells(args.data_root, "train"))
        test_wells = len(list_wells(args.data_root, "test"))
        expected = {
            "pf_train": (args.cache_root / "pf" / "train", train_wells),
            "pf_test": (args.cache_root / "pf" / "test", test_wells),
            "xy_test": (args.cache_root / "xy" / "test", test_wells),
        }
        for fold in range(1, 6):
            expected[f"xy_train_fold{fold}"] = (
                args.cache_root / "xy" / "train" / f"fold{fold}", train_wells
            )
        cache_report = {}
        for name, (folder, count) in expected.items():
            found = len(list(folder.glob("*.npz"))) if folder.is_dir() else 0
            assert found == count, (name, str(folder), found, count)
            cache_report[name] = found

    report = {
        "reproduction_level": "behavioral_reproduction",
        "geometry": [HEIGHT, WIDTH], "versions": EXPECTED_CHANNELS,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fold_sizes": fold_report, "cache_files": cache_report, "contract_pass": True,
        "remaining_inferred": [
            "visible typewell blend coefficient", "exact custom decoder wiring",
            "Huber delta", "TVT_bias_v2 window implementation",
            "PF diverse profile values and FFBSi path count",
            "original geo-map coordinate construction",
        ],
    }
    print("CONTRACT AUDIT", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
