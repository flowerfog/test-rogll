from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import FOLDS, RECONSTRUCTION_REVISION, SEEDS, VERSION_BY_NAME, VERSIONS
from .data import AlignmentDataset, list_wells
from .loss import decode_expected
from .model import AlignmentUNet
from .train import load_auxiliary
from .xy import xy_is_safe
from .infer_utils import smooth_raw


@torch.no_grad()
def predict_version(data_root: Path, cache_root: Path, model_root: Path, split: str,
                    version_name: str, wells: list[str]) -> dict[str, np.ndarray]:
    version = VERSION_BY_NAME[version_name]
    auxiliary = load_auxiliary(cache_root, split, wells, version.channels)
    dataset = AlignmentDataset(data_root, wells, split, version.channels, auxiliary)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True,
                        persistent_workers=True, prefetch_factor=1)
    accum: dict[str, np.ndarray] = {}
    members = 0
    for fold in FOLDS:
        for seed in SEEDS:
            path = model_root / version.name / f"fold{fold}_seed{seed}" / "best.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            revision = payload.get("config", {}).get("revision")
            if revision != RECONSTRUCTION_REVISION:
                raise RuntimeError({"message": "stale checkpoint", "path": str(path),
                                    "found": revision, "expected": RECONSTRUCTION_REVISION})
            layout = payload.get("config", {}).get("network_layout", "state_horizontal")
            model = AlignmentUNet(len(version.channels), version.dropout, layout=layout).cuda().eval()
            model.load_state_dict(payload["model"], strict=True)
            for batch in loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(batch["x"].float().cuda(non_blocking=True))
                _, expected = decode_expected(logits, batch["state_tvt"].cuda())
                for index, well in enumerate(batch["well"]):
                    value = expected[index].float().cpu().numpy()
                    accum[well] = accum.get(well, np.zeros_like(value)) + value
            members += 1
            del model
    return {well: prediction / members for well, prediction in accum.items()}


def raw_predictions(data_root: Path, split: str, version_name: str,
                    binned: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    version = VERSION_BY_NAME[version_name]
    result = {}
    for well, prediction in binned.items():
        hz = pd.read_csv(data_root / split / f"{well}__horizontal_well.csv")
        visible = np.flatnonzero(np.isfinite(hz.TVT_input.to_numpy(float)))
        anchor = int(visible[-1])
        start = anchor - 1023
        columns = start + np.minimum(np.arange(345) * 32 + 31, 11023)
        raw = np.interp(np.arange(len(hz)), columns, prediction, left=prediction[0], right=prediction[-1])
        raw = smooth_raw(raw, version.sg_window, version.sg_blend)
        raw[: anchor + 1] = hz.TVT_input.to_numpy(float)[: anchor + 1]
        result[well] = raw
    return result


def load_xy_stats(cache_root: Path, split: str, well: str) -> dict[str, float]:
    with np.load(cache_root / "xy" / split / f"{well}.npz") as source:
        return {key.removeprefix("stat__"): float(source[key]) for key in source.files if key.startswith("stat__")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Inference requires a BF16 CUDA device")
    wells = list_wells(args.data_root, args.split)
    version_predictions: dict[str, dict[str, np.ndarray]] = {}
    for version in VERSIONS:
        if version.general_weight == 0 and version.no_xy_weight == 0:
            continue
        print("INFER VERSION", version.name, flush=True)
        version_predictions[version.name] = raw_predictions(
            args.data_root, args.split, version.name,
            predict_version(args.data_root, args.cache_root, args.model_root, args.split, version.name, wells))
    rows = []
    routing = {}
    for well in wells:
        hz = pd.read_csv(args.data_root / args.split / f"{well}__horizontal_well.csv")
        stats = load_xy_stats(args.cache_root, args.split, well)
        safe = xy_is_safe(stats)
        routing[well] = {"xy_safe": safe, **stats}
        weighted = np.zeros(len(hz), np.float64)
        total = 0.0
        for version in VERSIONS:
            if version.name not in version_predictions:
                continue
            weight = version.general_weight if safe else version.no_xy_weight
            weighted += weight * version_predictions[version.name][well]
            total += weight
        prediction = weighted / total
        hidden = ~np.isfinite(hz.TVT_input.to_numpy(float))
        for index in np.flatnonzero(hidden):
            rows.append({"id": f"{well}_{index}", "tvt": float(prediction[index])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    (args.output.parent / "routing.json").write_text(json.dumps(routing, indent=2), encoding="utf-8")
    print("WROTE", {"path": str(args.output), "rows": len(rows), "wells": len(wells)}, flush=True)


if __name__ == "__main__":
    main()
