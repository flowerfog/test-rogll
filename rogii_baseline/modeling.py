"""Grouped out-of-fold evaluation and CUDA residual modeling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn

from .data import WellPaths, discover_wells, read_well
from .features import FEATURE_NAMES, FeatureSet, build_features


@dataclass(frozen=True)
class ModelConfig:
    max_train_rows_per_well: int = 512
    max_iter: int = 120
    learning_rate: float = 1e-3
    batch_size: int = 8192
    hidden_width: int = 128
    weight_decay: float = 1e-4
    device: str = "cuda"
    random_state: int = 20260723


@dataclass(frozen=True)
class TrainingSample:
    x: np.ndarray
    residual: np.ndarray


class ResidualMLP(nn.Module):
    def __init__(self, input_width: int, hidden_width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


@dataclass
class TorchResidualRegressor:
    network: ResidualMLP
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    device: torch.device
    prediction_batch_size: int

    def predict(self, values: np.ndarray) -> np.ndarray:
        normalized = np.clip(
            (np.asarray(values, dtype=np.float32) - self.x_mean) / self.x_scale,
            -12.0,
            12.0,
        )
        outputs: list[np.ndarray] = []
        self.network.eval()
        with torch.inference_mode():
            for start in range(0, len(normalized), self.prediction_batch_size):
                batch = torch.as_tensor(
                    normalized[start : start + self.prediction_batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                prediction = self.network(batch)
                outputs.append(prediction.cpu().numpy())
        standardized = np.concatenate(outputs) if outputs else np.empty(0)
        return standardized.astype(np.float64) * self.y_scale + self.y_mean


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


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def fit_model(
    samples: dict[str, TrainingSample], well_ids: list[str], config: ModelConfig
) -> TorchResidualRegressor:
    if not well_ids:
        raise ValueError("At least one training well is required")
    if config.max_iter < 1 or config.batch_size < 1:
        raise ValueError("max_iter and batch_size must be positive")

    train_x = np.concatenate([samples[well_id].x for well_id in well_ids]).astype(
        np.float32, copy=False
    )
    train_y = np.concatenate(
        [samples[well_id].residual for well_id in well_ids]
    ).astype(np.float32, copy=False)
    x_mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = np.where(x_scale > 1e-6, x_scale, 1.0).astype(np.float32)
    y_mean = float(train_y.mean(dtype=np.float64))
    y_scale = max(float(train_y.std(dtype=np.float64)), 1e-6)
    normalized_x = np.clip((train_x - x_mean) / x_scale, -12.0, 12.0)
    normalized_y = (train_y - y_mean) / y_scale

    device = _resolve_device(config.device)
    torch.manual_seed(config.random_state)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.random_state)
        torch.set_float32_matmul_precision("high")
    network = ResidualMLP(train_x.shape[1], config.hidden_width).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()
    x_tensor = torch.as_tensor(normalized_x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(normalized_y, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.random_state)

    for epoch in range(1, config.max_iter + 1):
        network.train()
        permutation = torch.randperm(len(x_tensor), generator=generator, device=device)
        epoch_squared_error = 0.0
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = network(x_tensor[indices])
            loss = loss_function(prediction, y_tensor[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_squared_error += float(loss.detach()) * len(indices)
        if epoch == 1 or epoch % 20 == 0 or epoch == config.max_iter:
            mean_loss = epoch_squared_error / len(x_tensor)
            print(
                f"device={device} epoch={epoch}/{config.max_iter} "
                f"standardized_mse={mean_loss:.8f}",
                flush=True,
            )

    return TorchResidualRegressor(
        network=network,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        device=device,
        prediction_batch_size=config.batch_size,
    )


def _score_wells(
    model: TorchResidualRegressor, wells: list[WellPaths]
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
        "device": config.device,
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
        "device": config.device,
    }
