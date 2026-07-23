"""Grouped out-of-fold evaluation and full-data prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .data import WellPaths, discover_wells, read_well
from .features import FEATURE_NAMES, FeatureSet, build_features


@dataclass(frozen=True)
class ModelConfig:
    max_train_rows_per_well: int = 512
    max_iter: int = 160
    learning_rate: float = 0.06
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 0.2
    random_state: int = 20260723


@dataclass(frozen=True)
class TrainingSample:
    x: np.ndarray
    residual: np.ndarray


def _even_sample(features: FeatureSet, limit: int) -> TrainingSample:
    if features.target_residual is None:
        raise ValueError("Training features have no target residual")
    count = len(features.row_indices)
    if limit <= 0:
        raise ValueError("max_train_rows_per_well must be positive")
    selected = (
        np.arange(count)
        if count <= limit
        else np.unique(np.linspace(0, count - 1, num=limit, dtype=int))
    )
    return TrainingSample(features.x[selected], features.target_residual[selected])


def prepare_training_samples(
    wells: list[WellPaths], config: ModelConfig
) -> dict[str, TrainingSample]:
    samples: dict[str, TrainingSample] = {}
    for number, paths in enumerate(wells, start=1):
        horizontal, typewell = read_well(paths, require_target=True)
        features = build_features(horizontal, typewell, include_target=True)
        samples[paths.well_id] = _even_sample(
            features, config.max_train_rows_per_well
        )
        if number % 100 == 0 or number == len(wells):
            print(f"prepared_wells={number}/{len(wells)}", flush=True)
    return samples


def fit_model(
    samples: dict[str, TrainingSample], well_ids: list[str], config: ModelConfig
) -> HistGradientBoostingRegressor:
    if not well_ids:
        raise ValueError("At least one training well is required")
    train_x = np.concatenate([samples[well_id].x for well_id in well_ids])
    train_y = np.concatenate([samples[well_id].residual for well_id in well_ids])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        max_bins=127,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=config.random_state,
    )
    model.fit(train_x, train_y)
    return model


def _score_wells(
    model: HistGradientBoostingRegressor, wells: list[WellPaths]
) -> tuple[float, int]:
    squared_error_sum = 0.0
    row_count = 0
    for paths in wells:
        horizontal, typewell = read_well(paths, require_target=True)
        features = build_features(horizontal, typewell, include_target=True)
        predicted = features.baseline_tvt + model.predict(features.x)
        actual = horizontal["TVT"].to_numpy(dtype=float)[features.row_indices]
        squared_error_sum += float(np.square(predicted - actual).sum())
        row_count += len(actual)
    return squared_error_sum, row_count


def grouped_oof(
    train_dir: Path, *, folds: int, config: ModelConfig
) -> dict[str, object]:
    wells = discover_wells(train_dir)
    if folds < 2 or folds > len(wells):
        raise ValueError(f"folds must be between 2 and {len(wells)}")
    samples = prepare_training_samples(wells, config)
    well_ids = np.array([well.well_id for well in wells], dtype=object)
    path_by_id = {well.well_id: well for well in wells}
    splitter = KFold(
        n_splits=folds, shuffle=True, random_state=config.random_state
    )

    total_squared_error = 0.0
    total_rows = 0
    fold_results: list[dict[str, object]] = []
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(well_ids), 1):
        train_ids = well_ids[train_indices].tolist()
        valid_ids = well_ids[valid_indices].tolist()
        model = fit_model(samples, train_ids, config)
        fold_sse, fold_rows = _score_wells(
            model, [path_by_id[well_id] for well_id in valid_ids]
        )
        fold_rmse = float(np.sqrt(fold_sse / fold_rows))
        result = {
            "fold": fold,
            "train_wells": len(train_ids),
            "valid_wells": len(valid_ids),
            "valid_rows": fold_rows,
            "pooled_rmse": fold_rmse,
        }
        fold_results.append(result)
        total_squared_error += fold_sse
        total_rows += fold_rows
        print(json.dumps(result, sort_keys=True), flush=True)

    return {
        "pooled_oof_rmse": float(np.sqrt(total_squared_error / total_rows)),
        "oof_rows": total_rows,
        "wells": len(wells),
        "folds": fold_results,
        "feature_count": len(FEATURE_NAMES),
    }


def predict_hidden_test(
    data_root: Path, output: Path, *, config: ModelConfig
) -> dict[str, object]:
    train_wells = discover_wells(data_root / "train")
    test_wells = discover_wells(data_root / "test")
    samples = prepare_training_samples(train_wells, config)
    model = fit_model(samples, [well.well_id for well in train_wells], config)

    predictions: dict[str, float] = {}
    for paths in test_wells:
        horizontal, typewell = read_well(paths, require_target=False)
        features = build_features(horizontal, typewell, include_target=False)
        values = features.baseline_tvt + model.predict(features.x)
        for row_index, value in zip(features.row_indices, values, strict=True):
            prediction_id = f"{paths.well_id}_{row_index}"
            if prediction_id in predictions:
                raise ValueError(f"Duplicate prediction ID: {prediction_id}")
            predictions[prediction_id] = float(value)

    sample_path = data_root / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected sample submission columns in {sample_path}")
    sample_ids = sample["id"].astype(str)
    missing = sorted(set(sample_ids) - predictions.keys())
    extra = sorted(predictions.keys() - set(sample_ids))
    if missing or extra:
        raise ValueError(
            "Prediction IDs do not match sample submission: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    submission = pd.DataFrame(
        {"id": sample_ids, "tvt": sample_ids.map(predictions).astype(float)}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    return {
        "output": str(output),
        "prediction_rows": len(submission),
        "train_wells": len(train_wells),
        "test_wells": len(test_wells),
        "feature_count": len(FEATURE_NAMES),
    }

