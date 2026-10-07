"""Five-fold complete-well structural-surface sequence endpoint.

V4 proved that the cumulative structural-surface representation is accurate
(~0.57 ft oracle) but an independent segment GBDT cannot infer its long-range
trend.  This cell changes the learning unit, not the blend dose: one sample is
one complete well and a bidirectional GRU consumes the full sequence of 256-ft
segment summaries before predicting cumulative surface displacement at every
segment endpoint.

The frozen reference endpoint is evaluation-only.  It is neither an input nor
part of the training objective.  All normalizers, epoch selection and model
weights are outer-train derived.

Fast path: run immediately after segment_surface_decorrelated_frontier_v4_cell.py
in the same kernel.  Resume path: upload the V2/V3/V4 source cells; this script
will execute only V4's data-bank prefix and will not retrain V4.
"""

from pathlib import Path
import gc
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


ROOT_V5 = Path("/root/autodl-tmp")
OOF_ROOT_V5 = ROOT_V5 / "oof_exports"
EXPERIMENT_V5 = "fullwell_surface_bigru_frontier_v5"
OUT_DIR_V5 = OOF_ROOT_V5 / EXPERIMENT_V5
MODEL_DIR_V5 = OUT_DIR_V5 / "models"
OUT_DIR_V5.mkdir(parents=True, exist_ok=True)
MODEL_DIR_V5.mkdir(parents=True, exist_ok=True)


def _bootstrap_v4_bank():
    required = {
        "rows_v4",
        "segments_v4",
        "model_features_v4",
        "reference_path_v4",
        "reference_column_v4",
        "reference_exact_v4",
        "oracle_metrics_v4",
    }
    if required.issubset(globals()):
        print("V5 BOOTSTRAP: reusing live V4 segment bank", flush=True)
        return
    candidates = [
        ROOT_V5 / "segment_surface_decorrelated_frontier_v4_cell.py",
        Path("segment_surface_decorrelated_frontier_v4_cell.py"),
    ]
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Upload segment_surface_decorrelated_frontier_v4_cell.py to "
            "/root/autodl-tmp or run V4 in this kernel"
        )
    source = source_path.read_text(encoding="utf-8")
    marker = "oof_rates_v4 ="
    if marker not in source:
        raise RuntimeError(f"Cannot locate V4 bank boundary in {source_path}")
    print("V5 BOOTSTRAP: rebuilding only the V4 segment bank", flush=True)
    exec(
        compile(source.split(marker, 1)[0], str(source_path) + "::<bank>", "exec"),
        globals(),
    )


_bootstrap_v4_bank()


SEED_V5 = 20260724
FOLDS_V5 = [1, 2, 3, 4, 5]
MAX_EPOCHS_V5 = 80
PATIENCE_V5 = 12
BATCH_WELLS_V5 = 24
HIDDEN_V5 = 72
EMBED_V5 = 112
DROPOUT_V5 = 0.15
LEARNING_RATE_V5 = 8e-4
WEIGHT_DECAY_V5 = 2e-4
DELTA_SCALE_V5 = 32.0
BLEND_GRID_V5 = np.array(
    [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
    dtype=np.float64,
)
DEVICE_V5 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything_v5(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default_v5(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json_v5(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default_v5),
        encoding="utf-8",
    )


class FullWellSurfaceBiGRUV5(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, EMBED_V5),
            nn.GELU(),
            nn.LayerNorm(EMBED_V5),
            nn.Dropout(DROPOUT_V5),
        )
        self.gru = nn.GRU(
            EMBED_V5,
            HIDDEN_V5,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT_V5,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * HIDDEN_V5, HIDDEN_V5),
            nn.GELU(),
            nn.Dropout(DROPOUT_V5),
            nn.Linear(HIDDEN_V5, 1),
        )

    def forward(self, features, lengths):
        encoded = self.input(features)
        packed = pack_padded_sequence(
            encoded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed, _ = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed, batch_first=True, total_length=features.shape[1]
        )
        return self.head(output).squeeze(-1)


def _feature_matrix_v5(frame):
    return (
        frame.reindex(columns=model_features_v4)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


segments_v5 = segments_v4.copy().reset_index(drop=True)
segments_v5["_v5_index"] = np.arange(len(segments_v5), dtype=np.int32)
well_indices_v5 = {
    str(well): group.sort_values("seg_index")["_v5_index"].to_numpy(np.int64)
    for well, group in segments_v5.groupby("well", sort=False)
}
fold_by_well_v5 = (
    segments_v5.groupby("well", sort=False)["fold"].first().astype(int).to_dict()
)
features_raw_v5 = _feature_matrix_v5(segments_v5)
target_delta_v5 = (
    segments_v5["target_avg_rate"].to_numpy(np.float32)
    * segments_v5["seg_end_ft"].to_numpy(np.float32)
)
segment_rows_v5 = segments_v5["seg_rows"].to_numpy(np.float32)


def _normalizer_v5(wells):
    indices = np.concatenate([well_indices_v5[well] for well in wells])
    values = features_raw_v5[indices].astype(np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-5)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _batches_v5(wells, mean, scale, shuffle, seed):
    order = np.arange(len(wells))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), BATCH_WELLS_V5):
        selected = [wells[position] for position in order[start : start + BATCH_WELLS_V5]]
        lengths = np.array([len(well_indices_v5[well]) for well in selected], np.int64)
        width = int(lengths.max())
        x = np.zeros((len(selected), width, features_raw_v5.shape[1]), np.float32)
        y = np.zeros((len(selected), width), np.float32)
        well_weight = np.zeros((len(selected), width), np.float32)
        pooled_weight = np.zeros((len(selected), width), np.float32)
        mask = np.zeros((len(selected), width), bool)
        global_indices = []
        for row, well in enumerate(selected):
            indices = well_indices_v5[well]
            size = len(indices)
            x[row, :size] = (features_raw_v5[indices] - mean) / scale
            x[row, :size] = np.clip(x[row, :size], -8.0, 8.0)
            y[row, :size] = target_delta_v5[indices] / DELTA_SCALE_V5
            # Equalise wells while retaining the official row weighting inside a well.
            local_weight = segment_rows_v5[indices]
            local_weight = local_weight / max(float(local_weight.sum()), 1.0)
            well_weight[row, :size] = local_weight
            pooled_weight[row, :size] = segment_rows_v5[indices]
            mask[row, :size] = True
            global_indices.append(indices)
        yield {
            "features": torch.from_numpy(x).to(DEVICE_V5),
            "target": torch.from_numpy(y).to(DEVICE_V5),
            "well_weight": torch.from_numpy(well_weight).to(DEVICE_V5),
            "pooled_weight": torch.from_numpy(
                pooled_weight / max(float(pooled_weight.sum()), 1.0)
            ).to(DEVICE_V5),
            "mask": torch.from_numpy(mask).to(DEVICE_V5),
            "lengths": torch.from_numpy(lengths).to(DEVICE_V5),
            "indices": global_indices,
        }


def _loss_v5(prediction, batch):
    error = prediction - batch["target"]
    well_weight = batch["well_weight"]
    pooled_weight = batch["pooled_weight"]
    # Optimise the official pooled target while retaining a well-balanced guard.
    pooled_mse = (pooled_weight * error.square()).sum()
    well_mse = (well_weight * error.square()).sum() / max(prediction.shape[0], 1)
    robust = (
        well_weight
        * torch.nn.functional.smooth_l1_loss(
            prediction, batch["target"], reduction="none", beta=0.25
        )
    ).sum() / max(prediction.shape[0], 1)
    derivative_sum = prediction.new_tensor(0.0)
    derivative_count = 0
    for row, length in enumerate(batch["lengths"].tolist()):
        if length > 1:
            pred_delta = prediction[row, 1:length] - prediction[row, : length - 1]
            true_delta = batch["target"][row, 1:length] - batch["target"][row, : length - 1]
            derivative_sum = derivative_sum + torch.nn.functional.smooth_l1_loss(
                pred_delta, true_delta, reduction="mean", beta=0.20
            )
            derivative_count += 1
    derivative = derivative_sum / max(derivative_count, 1)
    return 0.60 * pooled_mse + 0.25 * well_mse + 0.15 * robust + 0.12 * derivative


@torch.no_grad()
def _predict_v5(model, wells, mean, scale):
    model.eval()
    output = np.full(len(segments_v5), np.nan, np.float32)
    for batch in _batches_v5(wells, mean, scale, False, 0):
        prediction = model(batch["features"], batch["lengths"]).cpu().numpy()
        for row, indices in enumerate(batch["indices"]):
            output[indices] = prediction[row, : len(indices)] * DELTA_SCALE_V5
    return output


def _endpoint_rmse_v5(predicted_delta, wells):
    indices = np.concatenate([well_indices_v5[well] for well in wells])
    error2 = (predicted_delta[indices] - target_delta_v5[indices]) ** 2
    weight = segment_rows_v5[indices].astype(np.float64)
    return float(np.sqrt(np.sum(weight * error2) / np.sum(weight)))


def _train_selection_v5(train_wells, valid_wells, mean, scale, fold):
    _seed_everything_v5(SEED_V5 + fold)
    model = FullWellSurfaceBiGRUV5(features_raw_v5.shape[1]).to(DEVICE_V5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_V5, weight_decay=WEIGHT_DECAY_V5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS_V5, eta_min=8e-5
    )
    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(1, MAX_EPOCHS_V5 + 1):
        model.train()
        losses = []
        for batch in _batches_v5(
            train_wells, mean, scale, True, SEED_V5 + 1000 * fold + epoch
        ):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["features"], batch["lengths"])
            loss = _loss_v5(prediction, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        valid_delta = _predict_v5(model, valid_wells, mean, scale)
        valid_rmse = _endpoint_rmse_v5(valid_delta, valid_wells)
        history.append(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "valid_endpoint_rmse": valid_rmse,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if valid_rmse < best_rmse - 1e-4:
            best_rmse = valid_rmse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 5 == 0 or bad_epochs >= PATIENCE_V5:
            print(
                "V5 SELECT",
                {"fold": fold, "epoch": epoch, "loss": float(np.mean(losses)),
                 "valid_endpoint_rmse": valid_rmse, "best": best_rmse},
                flush=True,
            )
        if bad_epochs >= PATIENCE_V5:
            break
    del model, optimizer, scheduler
    gc.collect()
    if DEVICE_V5.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, best_rmse, best_state, history


def _train_final_v5(train_wells, mean, scale, fold, epochs):
    _seed_everything_v5(SEED_V5 + 100 + fold)
    model = FullWellSurfaceBiGRUV5(features_raw_v5.shape[1]).to(DEVICE_V5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_V5, weight_decay=WEIGHT_DECAY_V5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=8e-5
    )
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in _batches_v5(
            train_wells, mean, scale, True, SEED_V5 + 2000 * fold + epoch
        ):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["features"], batch["lengths"])
            loss = _loss_v5(prediction, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                "V5 REFIT",
                {"fold": fold, "epoch": epoch, "epochs": epochs,
                 "loss": float(np.mean(losses))},
                flush=True,
            )
    return model


def _selection_scan_v5(selection, candidate_column):
    truth = selection["truth"].to_numpy(np.float64)
    reference = selection["reference"].to_numpy(np.float64)
    candidate = selection[candidate_column].to_numpy(np.float64)
    records = []
    for weight in BLEND_GRID_V5:
        prediction = reference + weight * (candidate - reference)
        records.append(
            {"weight_candidate": float(weight), **_metrics(selection, prediction)}
        )
    scan = pd.DataFrame(records).sort_values(
        ["rmse", "weight_candidate"], ignore_index=True
    )
    return scan.iloc[0].to_dict(), scan


started_v5 = time.time()
all_wells_v5 = sorted(well_indices_v5)
oof_delta_v5 = np.full(len(segments_v5), np.nan, np.float32)
history_v5 = []
fold_models_v5 = []

print(
    "FULLWELL-SURFACE V5 CONFIG",
    {
        "device": str(DEVICE_V5),
        "wells": len(all_wells_v5),
        "segments": len(segments_v5),
        "features": features_raw_v5.shape[1],
        "parameters": sum(
            parameter.numel()
            for parameter in FullWellSurfaceBiGRUV5(features_raw_v5.shape[1]).parameters()
        ),
        "reference": str(reference_path_v4),
        "reference_column": reference_column_v4,
    },
    flush=True,
)

for outer_fold_v5 in FOLDS_V5:
    tune_fold_v5 = FOLDS_V5[outer_fold_v5 % len(FOLDS_V5)]
    fit_wells_v5 = [
        well for well in all_wells_v5
        if fold_by_well_v5[well] not in (outer_fold_v5, tune_fold_v5)
    ]
    tune_wells_v5 = [
        well for well in all_wells_v5 if fold_by_well_v5[well] == tune_fold_v5
    ]
    final_wells_v5 = [
        well for well in all_wells_v5 if fold_by_well_v5[well] != outer_fold_v5
    ]
    valid_wells_v5 = [
        well for well in all_wells_v5 if fold_by_well_v5[well] == outer_fold_v5
    ]
    mean_select_v5, scale_select_v5 = _normalizer_v5(fit_wells_v5)
    best_epoch_v5, best_endpoint_v5, _, local_history_v5 = _train_selection_v5(
        fit_wells_v5,
        tune_wells_v5,
        mean_select_v5,
        scale_select_v5,
        outer_fold_v5,
    )
    history_v5.extend(local_history_v5)
    mean_final_v5, scale_final_v5 = _normalizer_v5(final_wells_v5)
    final_model_v5 = _train_final_v5(
        final_wells_v5,
        mean_final_v5,
        scale_final_v5,
        outer_fold_v5,
        best_epoch_v5,
    )
    fold_prediction_v5 = _predict_v5(
        final_model_v5, valid_wells_v5, mean_final_v5, scale_final_v5
    )
    valid_indices_v5 = np.concatenate([well_indices_v5[well] for well in valid_wells_v5])
    oof_delta_v5[valid_indices_v5] = fold_prediction_v5[valid_indices_v5]
    checkpoint_v5 = {
        "model": final_model_v5.state_dict(),
        "feature_names": model_features_v4,
        "normalizer_mean": mean_final_v5,
        "normalizer_scale": scale_final_v5,
        "selected_epoch": best_epoch_v5,
        "outer_fold": outer_fold_v5,
    }
    torch.save(checkpoint_v5, MODEL_DIR_V5 / f"surface_bigru_fold{outer_fold_v5}.pt")
    fold_models_v5.append(
        {"fold": outer_fold_v5, "tune_fold": tune_fold_v5,
         "selected_epoch": best_epoch_v5,
         "tune_endpoint_rmse": best_endpoint_v5,
         "fit_wells": len(fit_wells_v5), "final_wells": len(final_wells_v5),
         "valid_wells": len(valid_wells_v5)}
    )
    del final_model_v5, fold_prediction_v5
    gc.collect()
    if DEVICE_V5.type == "cuda":
        torch.cuda.empty_cache()

if not np.isfinite(oof_delta_v5).all():
    missing_v5 = int((~np.isfinite(oof_delta_v5)).sum())
    raise RuntimeError(f"Incomplete V5 OOF cumulative deltas: {missing_v5}")

oof_rate_v5 = oof_delta_v5.astype(np.float64) / np.maximum(
    segments_v5["seg_end_ft"].to_numpy(np.float64), 1.0
)
rows_v5 = rows_v4.copy()
rows_v5["surface_bigru"] = _decode_average_rates_v3(
    rows_v5, segments_v5, oof_rate_v5
).astype(np.float32)

candidate_metrics_v5 = _metrics_extended_v4(rows_v5, rows_v5["surface_bigru"])
candidate_correlations_v5 = _correlations(rows_v5, rows_v5["surface_bigru"])
candidate_correlations_v5.update(
    _distance_correlations_v4(rows_v5, rows_v5["surface_bigru"])
)
reference_metrics_v5 = _metrics_extended_v4(rows_v5, rows_v5["reference"])

nested_blend_v5 = np.empty(len(rows_v5), np.float64)
fold_blend_records_v5 = []
scan_parts_v5 = []
for held_fold_v5 in FOLDS_V5:
    selection_v5 = rows_v5.loc[rows_v5["fold"].ne(held_fold_v5)]
    held_mask_v5 = rows_v5["fold"].eq(held_fold_v5).to_numpy()
    held_v5 = rows_v5.loc[held_mask_v5]
    selected_v5, scan_v5 = _selection_scan_v5(selection_v5, "surface_bigru")
    scan_v5.insert(0, "held_fold", held_fold_v5)
    scan_parts_v5.append(scan_v5)
    weight_v5 = float(selected_v5["weight_candidate"])
    reference_v5 = held_v5["reference"].to_numpy(np.float64)
    candidate_v5 = held_v5["surface_bigru"].to_numpy(np.float64)
    blend_v5 = reference_v5 + weight_v5 * (candidate_v5 - reference_v5)
    nested_blend_v5[held_mask_v5] = blend_v5
    base_rmse_v5 = _metrics(held_v5, reference_v5)["rmse"]
    blend_rmse_v5 = _metrics(held_v5, blend_v5)["rmse"]
    fold_blend_records_v5.append(
        {"fold": held_fold_v5, "weight_candidate": weight_v5,
         "reference_rmse": base_rmse_v5,
         "candidate_rmse": _metrics(held_v5, candidate_v5)["rmse"],
         "blend_rmse": blend_rmse_v5, "gain": base_rmse_v5 - blend_rmse_v5,
         **_correlations(held_v5, candidate_v5),
         **_distance_correlations_v4(held_v5, candidate_v5)}
    )

rows_v5["nested_blend"] = nested_blend_v5.astype(np.float32)
blend_metrics_v5 = _metrics_extended_v4(rows_v5, rows_v5["nested_blend"])
blend_gain_v5 = reference_metrics_v5["rmse"] - blend_metrics_v5["rmse"]
fold_report_v5 = pd.DataFrame(fold_blend_records_v5)

checks_v5 = {
    "representation_oracle_rmse_le_1p5": oracle_metrics_v4["rmse"] <= 1.5,
    "standalone_rmse_le_10p5": candidate_metrics_v5["rmse"] <= 10.5,
    "far_gt2000_rmse_le_13": candidate_metrics_v5["far_gt2000_rmse"] <= 13.0,
    "distance_detrended_rho_le_0p50": candidate_correlations_v5[
        "distance_detrended_error_rho"
    ] <= 0.50,
    "nested_blend_gain_ge_0p15": blend_gain_v5 >= 0.15,
    "minimum_weight_ge_0p05": float(fold_report_v5["weight_candidate"].min()) >= 0.05,
    "positive_folds_ge_4": int((fold_report_v5["gain"] > 0).sum()) >= 4,
}
strict_pass_v5 = bool(all(checks_v5.values()))

rows_v5[[
    "id", "well", "row_idx", "fold", "truth", "md_since", "reference",
    "surface_bigru", "nested_blend"
]].to_parquet(OUT_DIR_V5 / "fullwell_surface_bigru_oof.parquet", index=False)
pd.DataFrame(history_v5).to_csv(OUT_DIR_V5 / "training_history.csv", index=False)
pd.DataFrame(fold_models_v5).to_csv(OUT_DIR_V5 / "model_report.csv", index=False)
fold_report_v5.to_csv(OUT_DIR_V5 / "nested_fold_report.csv", index=False)
pd.concat(scan_parts_v5, ignore_index=True).to_csv(
    OUT_DIR_V5 / "nested_blend_scan.csv", index=False
)

verdict_v5 = {
    "experiment": EXPERIMENT_V5,
    "stage": "fivefold_complete_well_surface_sequence_gateway",
    "sample_unit": "one complete physical well",
    "architecture": "two-layer bidirectional GRU over 256-ft segment summaries",
    "reference": {"path": str(reference_path_v4), "column": reference_column_v4,
                  "exact_model_core": bool(reference_exact_v4)},
    "protocol": {
        "reference_used_as_input": False,
        "reference_used_in_training_objective": False,
        "normalizer": "outer-train only",
        "epoch_selection": "three folds train, next fold tune",
        "outer_refit": "all four outer-train folds for frozen epoch count",
    },
    "guards": {"wells": len(all_wells_v5), "segments": len(segments_v5),
               "features": features_raw_v5.shape[1], "device": str(DEVICE_V5)},
    "representation_oracle": oracle_metrics_v4,
    "reference_metrics": reference_metrics_v5,
    "candidate_metrics": candidate_metrics_v5,
    "candidate_correlations": candidate_correlations_v5,
    "nested_blend_metrics": blend_metrics_v5,
    "nested_blend_gain": blend_gain_v5,
    "fold_models": fold_models_v5,
    "folds": fold_report_v5.to_dict(orient="records"),
    "checks": checks_v5,
    "strict_pass": strict_pass_v5,
    "elapsed_minutes": (time.time() - started_v5) / 60.0,
    "next": (
        "PASS -> refit deployment checkpoints and confirm against exact 6.47 core"
        if strict_pass_v5
        else "FAIL -> complete-well context still cannot infer cumulative structure; "
             "replace the evidence representation before any further decoder work"
    ),
}
_write_json_v5(OUT_DIR_V5 / "verdict.json", verdict_v5)
print("\nFULLWELL-SURFACE V5 FOLDS")
print(fold_report_v5.to_string(index=False), flush=True)
print("\nFULLWELL-SURFACE V5 FINAL VERDICT")
print(json.dumps(verdict_v5, indent=2, ensure_ascii=False, default=_json_default_v5), flush=True)
print("saved:", OUT_DIR_V5, flush=True)
