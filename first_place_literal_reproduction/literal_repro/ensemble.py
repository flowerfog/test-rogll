from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import QUALITY_COLUMNS, VERSIONS


def load_predictions(prediction_root: Path) -> pd.DataFrame:
    merged = None
    missing = []
    for version in VERSIONS:
        path = prediction_root / version / "submission.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        one = pd.read_csv(path, usecols=["id", "tvt"]).rename(columns={"tvt": version})
        merged = one if merged is None else merged.merge(one, on="id", how="inner", validate="one_to_one")
    if missing:
        raise FileNotFoundError({"message": "all six branch predictions are required", "missing": missing})
    if merged is None or len(merged) == 0:
        raise RuntimeError("no branch predictions loaded")
    return merged


def quality_gate(quality: pd.DataFrame) -> tuple[pd.Series, dict]:
    missing = [name for name in ("well", "split", *QUALITY_COLUMNS) if name not in quality]
    if missing:
        raise ValueError({"message": "XY routing quality columns missing", "missing": missing})
    train = quality[quality["split"].eq("train")]
    test = quality[quality["split"].eq("test")].copy()
    if train.empty or test.empty:
        raise ValueError("quality file requires both train and test rows")
    no_xy = pd.Series(False, index=test.index)
    thresholds = {}
    for name in QUALITY_COLUMNS:
        train_values = pd.to_numeric(train[name], errors="coerce")
        values = pd.to_numeric(test[name], errors="coerce")
        threshold = float(train_values.quantile(0.95))
        thresholds[name] = threshold
        no_xy |= values.isna() | values.gt(threshold)
    return pd.Series(no_xy.to_numpy(), index=test["well"].astype(str)), thresholds


def normalized_weights(mode: str) -> dict[str, float]:
    raw = {version: float(spec[mode]) for version, spec in VERSIONS.items()}
    total = sum(raw.values())
    if total <= 0:
        raise RuntimeError(f"zero ensemble weight for {mode}")
    return {version: weight / total for version, weight in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Published six-branch routed ensemble.")
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--quality-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = load_predictions(args.prediction_root)
    quality = pd.read_csv(args.quality_csv, dtype={"well": str})
    no_xy_by_well, thresholds = quality_gate(quality)
    well = frame["id"].str.split("_", n=1).str[0]
    no_xy = well.map(no_xy_by_well).fillna(True).to_numpy(bool)
    general_weights = normalized_weights("general")
    no_xy_weights = normalized_weights("no_xy")
    general = sum(frame[name].to_numpy(float) * weight for name, weight in general_weights.items())
    fallback = sum(frame[name].to_numpy(float) * weight for name, weight in no_xy_weights.items())
    output = pd.DataFrame({"id": frame["id"], "tvt": np.where(no_xy, fallback, general)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    report = {
        "rows": len(output),
        "no_xy_fraction": float(no_xy.mean()),
        "general_weights_normalized": general_weights,
        "no_xy_weights": no_xy_weights,
        "quality_thresholds": thresholds,
        "output": str(args.output),
    }
    print("ENSEMBLE", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
