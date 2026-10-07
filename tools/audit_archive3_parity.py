from __future__ import annotations

import json
import pathlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "archive (3)"
OUTPUT = ROOT / "postmortem" / "archive3_cleanroom_parity.json"


def _summary(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values)
    finite = np.isfinite(values)
    finite_values = values[finite]
    return {
        "shape": list(values.shape),
        "finite_fraction": float(finite.mean()),
        "mean": float(finite_values.mean()) if finite_values.size else None,
        "std": float(finite_values.std()) if finite_values.size else None,
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
    }


def main() -> None:
    # The supplied pickle was produced on Linux and contains PosixPath values.
    pathlib.PosixPath = pathlib.WindowsPath
    sys.path.insert(0, str(REFERENCE))
    sys.path.insert(0, str(ROOT))

    with (REFERENCE / "cfg.pkl").open("rb") as handle:
        cfg = pickle.load(handle)

    from seq_NN_dataset import SeqUNetDataset
    from first_place_cleanroom.config import VERSION_BY_NAME
    from first_place_cleanroom.data import AlignmentDataset

    wells = sorted(
        path.name.removesuffix("__horizontal_well.csv")
        for path in (ROOT / "train").glob("*__horizontal_well.csv")
    )
    if not wells:
        raise FileNotFoundError(ROOT / "train")
    well = wells[0]
    horizontal = pd.read_csv(ROOT / "train" / f"{well}__horizontal_well.csv")
    zero_geo = {well: np.zeros(len(horizontal), dtype=np.float32)}

    reference_dataset = SeqUNetDataset(
        ROOT / "train",
        [well],
        cfg,
        training=True,
        simulation=False,
        geo_prior=zero_geo,
    )
    reference_item = reference_dataset[0]

    version = VERSION_BY_NAME["0719_V1"]
    cleanroom_item = AlignmentDataset(
        ROOT,
        [well],
        "train",
        version.channels,
        auxiliary_store={},
        training=False,
        version_name=version.name,
    )[0]

    reference_x = np.asarray(reference_item["unet_static"], dtype=np.float32)
    cleanroom_x = cleanroom_item["x"].numpy().astype(np.float32)
    cleanroom_horizontal_state = cleanroom_x.transpose(0, 2, 1)

    channels = {}
    for index, name in enumerate(version.channels):
        reference_channel = reference_x[index]
        cleanroom_channel = cleanroom_horizontal_state[index]
        difference = cleanroom_channel - reference_channel
        channels[name] = {
            "reference": _summary(reference_channel),
            "cleanroom": _summary(cleanroom_channel),
            "difference": {
                "mae": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "max_abs": float(np.max(np.abs(difference))),
                "correlation": float(np.corrcoef(reference_channel.ravel(), cleanroom_channel.ravel())[0, 1]),
            },
        }

    reference_probs = np.asarray(reference_item["typewell_target_probs"], dtype=np.float32)
    cleanroom_probs = cleanroom_item["target"].numpy().T.astype(np.float32)
    target_difference = cleanroom_probs - reference_probs

    report = {
        "reference": str(REFERENCE),
        "well": well,
        "tensor_layout": {
            "reference": list(reference_x.shape),
            "cleanroom_native": list(cleanroom_x.shape),
            "cleanroom_transposed_for_comparison": list(cleanroom_horizontal_state.shape),
        },
        "state_axis": {
            "reference_first_last": [
                float(-cfg.typewell_window + cfg.typewell_window / cfg.typewell_len),
                float(cfg.typewell_window - cfg.typewell_window / cfg.typewell_len),
            ],
            "cleanroom_first_last": [
                float(cleanroom_item["state_tvt"][0] - cleanroom_item["anchor_tvt"]),
                float(cleanroom_item["state_tvt"][-1] - cleanroom_item["anchor_tvt"]),
            ],
        },
        "regression_target_stats": {
            "reference_cfg": list(cfg.target_stats["unet"]),
            "cleanroom_absolute_truth": True,
        },
        "alignment_target": {
            "reference": "Gaussian exp(-0.5 * (distance / 1.25)^2), downsampled mean TVT",
            "cleanroom": "Laplace exp(-abs(distance) / 1.25), last TVT in bin",
            "mae": float(np.mean(np.abs(target_difference))),
            "rmse": float(np.sqrt(np.mean(np.square(target_difference)))),
            "max_abs": float(np.max(np.abs(target_difference))),
        },
        "channels": channels,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
