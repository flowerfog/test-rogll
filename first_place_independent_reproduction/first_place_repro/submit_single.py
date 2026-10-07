from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ReproductionConfig, VERSION_RECIPES
from .data import (
    append_pf_channels,
    append_xy_channels,
    build_base_panel,
    list_wells,
    load_record,
    normalize_panel,
)
from .infer import decode_well, load_member, predict_panel
from .normalization import PopulationStats
from .pf import MultiProfileParticleFilter
from .xy import XYSurfacePrior


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_submission(sample_path: str | Path, predictions: pd.DataFrame) -> pd.DataFrame:
    sample = pd.read_csv(sample_path, dtype={"id": str})
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"unexpected sample submission columns: {list(sample.columns)}")
    if sample["id"].duplicated().any():
        raise ValueError("sample submission contains duplicate ids")
    if predictions["id"].duplicated().any():
        duplicates = predictions.loc[predictions["id"].duplicated(), "id"].head().tolist()
        raise ValueError(f"prediction contains duplicate ids: {duplicates}")

    values = predictions[["id", "tvt"]].copy()
    values["id"] = values["id"].astype(str)
    submission = sample[["id"]].merge(values, on="id", how="left", validate="one_to_one")
    missing = submission["tvt"].isna()
    if missing.any():
        raise RuntimeError({
            "message": "single model does not cover the complete Kaggle submission",
            "missing_rows": int(missing.sum()),
            "sample_missing_ids": submission.loc[missing, "id"].head(10).tolist(),
        })
    output = submission["tvt"].to_numpy(dtype=np.float64)
    if not np.isfinite(output).all():
        raise RuntimeError("submission contains non-finite TVT values")
    return submission


def predict_checkpoint(
    data_root: str | Path,
    checkpoint_path: str | Path,
    split: str = "test",
) -> tuple[pd.DataFrame, dict]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model", "version", "population_stats", "epoch", "metrics"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"checkpoint is missing required fields: {missing}")
    version = str(checkpoint["version"])
    if version not in VERSION_RECIPES:
        raise RuntimeError(f"unknown checkpoint version: {version}")

    config = ReproductionConfig()
    recipe = VERSION_RECIPES[version]
    population_stats = PopulationStats.from_dict(checkpoint["population_stats"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xy_prior = XYSurfacePrior(config.xy).fit_root(data_root) if recipe.use_xy else None
    model = None
    frames = []
    input_shapes = {}
    for well_index, well in enumerate(list_wells(data_root, split)):
        record = load_record(data_root, split, well)
        sample = build_base_panel(
            record,
            config.grid,
            config.loss.target_sigma_ft,
            population_stats=population_stats,
            apply_channel_normalization=False,
        )
        if recipe.use_pf:
            pf_result = MultiProfileParticleFilter(config.pf, 7919 + well_index).run(sample)
            sample = append_pf_channels(sample, pf_result.density, pf_result.ffbsi_offset)
        if recipe.use_xy:
            sample = append_xy_channels(sample, xy_prior.predict(record, sample))
        sample = normalize_panel(sample.select(recipe.channels), population_stats)
        if not np.isfinite(sample.values).all():
            raise RuntimeError(f"{well}: input tensor contains non-finite values")
        input_shapes[well] = list(sample.values.shape)
        if model is None:
            model = load_member(checkpoint_path, len(sample.channels), device)
        offsets = predict_panel([model], sample.values, sample.state_offsets, device)
        frames.append(decode_well(record, sample, offsets, recipe.sg_window, recipe.sg_blend))

    if not frames:
        raise RuntimeError(f"no {split} wells found under {data_root}")
    predictions = pd.concat(frames, ignore_index=True)
    audit = {
        "reproduction_level": "single-member approximate reconstruction submission",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "version": version,
        "selected_model": checkpoint.get("selected_model", "unknown"),
        "epoch": int(checkpoint["epoch"]),
        "validation_metrics": checkpoint["metrics"],
        "members": 1,
        "channels": list(recipe.channels),
        "smoothing": {"window": recipe.sg_window, "blend": recipe.sg_blend},
        "prediction_rows_before_sample_merge": int(len(predictions)),
        "wells": int(predictions["well"].nunique()),
        "input_shapes": input_shapes,
    }
    return predictions, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Kaggle submission from exactly one checkpoint.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predictions, audit = predict_checkpoint(args.data_root, args.checkpoint)
    submission = build_submission(args.sample_submission, predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    values = submission["tvt"].to_numpy(dtype=np.float64)
    audit.update({
        "submission": str(output),
        "submission_rows": int(len(submission)),
        "submission_sha256": sha256(output),
        "tvt_min": float(values.min()),
        "tvt_mean": float(values.mean()),
        "tvt_max": float(values.max()),
        "finite": bool(np.isfinite(values).all()),
    })
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("SINGLE MODEL SUBMISSION", json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
