"""V130: differentiable candidate-set mixture trained on final TVT SSE.

This is not a candidate ranker. A DeepSets model consumes the top-64 unique
candidate paths jointly and emits mixture weights. The weighted trajectory is
inserted into the exact V76/top33 formula and trained end-to-end against final
TVT. Fold4+5 train, the first 24 Fold3 wells select epoch/consumption strength,
and the remaining Fold3 wells confirm once.
"""

from __future__ import annotations

import json
import random
import sys
import time
import types
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "differentiable_candidate_mixture_v130"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

V110_SOURCE = ROOT / "rawdata_lattice_topk_path_barycenter_v110_cell.py"
V96_SOURCE = ROOT / "rawdata_lattice_v76_full773_training_v96_cell.py"
V96_COARSE = OOF_ROOT / "rawdata_lattice_v76_full773_training_v96" / "fivefold_coarse_oof.parquet"
V106_MODEL_DIR = OOF_ROOT / "rawdata_lattice_blendaware_shortlist_reranker_v106" / "models"
FOLD4_SCORER = V106_MODEL_DIR / "base_fold5_for_fold4.txt"
FOLD5_SCORER = V106_MODEL_DIR / "base_fold4_for_fold5.txt"
TRAIN_FOLDS = (4, 5)
EVAL_FOLD = 3
DISCOVERY_WELLS = 24
TOPK = 64
BASELINE_K = 33
NODES = 128
SUMMARY_NODES = 16
SEED = 20260801
MAX_EPOCHS = 30
PATIENCE = 7
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 5e-4
FREEZE_FT = 1000.0
RAMP_FT = 250.0
CAP_FT = 6.0
ALPHAS = (0.25, 0.50, 0.75, 1.00)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _bootstrap(path, marker, name):
    if not path.is_file():
        raise FileNotFoundError(path)
    source = path.read_text(encoding="utf-8")
    if marker not in source:
        raise RuntimeError(f"Definition marker missing from {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(path), "exec"), module.__dict__)
    return module


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


R = _bootstrap(V110_SOURCE, "# --- V110 EXECUTION ---", "v130_v110_definitions")
M = _bootstrap(
    V96_SOURCE,
    "for path in (V4_SOURCE, V66_PATH, V70_PATH, V76_BUNDLE, V76_OOF_WEIGHTS):",
    "v130_v96_definitions",
)
for required in (
    R.V81_SOURCE, R.PANEL_DIR, R.MODEL_PATH, V96_COARSE,
    M.V76_BUNDLE, M.V76_OOF_WEIGHTS, FOLD4_SCORER, FOLD5_SCORER,
):
    if not Path(required).exists():
        raise FileNotFoundError(required)

N = R._bootstrap_v81()
Q, V, B, prior = N.Q, N.V, N.B, N.prior
rankers = {
    3: lgb.Booster(model_file=str(R.MODEL_PATH)),
    4: lgb.Booster(model_file=str(FOLD4_SCORER)),
    5: lgb.Booster(model_file=str(FOLD5_SCORER)),
}

# Exact five-fold V76 reconstruction, restricted later to folds 3/4/5.
frame = pd.read_parquet(V96_COARSE).sort_values(
    ["fold", "well", "row_idx"], kind="stable"
).reset_index(drop=True)
frame["well"] = frame["well"].astype(str)
if len(frame) != 3_783_989 or frame["well"].nunique() != 773 or frame["id"].duplicated().any():
    raise RuntimeError("Invalid V96 coarse OOF")
well_features = pd.DataFrame([
    M._feature_row(local) for _, local in frame.groupby(["fold", "well"], sort=False)
]).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
x, _ = M._augment(well_features)
bundle = joblib.load(M.V76_BUNDLE)
weights = np.clip(
    bundle["hist_fraction"] * bundle["hist_model"].predict(x)
    + bundle["ridge_fraction"] * bundle["ridge_model"].predict(x),
    M.MIN_WEIGHT,
    M.MAX_WEIGHT,
)
old_oof = pd.read_parquet(
    M.V76_OOF_WEIGHTS, columns=["fold", "well", "ensemble_weight"]
)
old_map = dict(zip(old_oof["well"].astype(str), old_oof["ensemble_weight"].astype(float)))
honest = well_features["fold"].isin(M.PRIOR_FOLDS).to_numpy()
weights[honest] = well_features.loc[honest, "well"].astype(str).map(old_map).to_numpy(np.float64)
if not np.isfinite(weights).all():
    raise RuntimeError("Incomplete V76 dose reconstruction")
weight_map = dict(zip(well_features["well"].astype(str), weights))
frame["dose"] = frame["well"].map(weight_map).astype(np.float64)
frame["exact_v76"] = (
    frame[M.PARENT].to_numpy(np.float64)
    + frame["dose"].to_numpy(np.float64)
    * (frame[M.COARSE].to_numpy(np.float64) - frame[M.PARENT].to_numpy(np.float64))
).astype(np.float32)
frame = frame[frame["fold"].isin([3, 4, 5])].copy().reset_index(drop=True)
well_positions = {
    str(well): np.asarray(positions, np.int64)
    for well, positions in frame.groupby("well", sort=False).indices.items()
}


def _well_frame(well):
    return frame.iloc[well_positions[str(well)]]


candidate_feature_rows = []
trajectory_rows = []
exact_rows = []
coarse_rows = []
truth_rows = []
taper_rows = []
dose_rows = []
distance_max_rows = []
row_count_rows = []
fold_rows = []
well_order = []
grid = np.linspace(0.0, 1.0, NODES)
summary_grid = np.linspace(0.0, 1.0, SUMMARY_NODES)
report_rows = []

eligible_wells = sorted(
    well for well in well_positions if (R.PANEL_DIR / f"{well}.npz").is_file()
)
for number, well in enumerate(eligible_wells, 1):
    try:
        local = _well_frame(well)
        with np.load(R.PANEL_DIR / f"{well}.npz", allow_pickle=False) as data:
            features = data["features"].astype(np.float32, copy=False)
            paths = data["paths"].astype(np.int16, copy=False)
        record = V["_load_record"](well)
        eval_rows = record["eval_rows"]
        truth = record["truth"][eval_rows].astype(np.float64)
        if len(local) != len(eval_rows):
            raise RuntimeError(f"row mismatch {len(local)}/{len(eval_rows)}")
        truth_gap = float(np.max(np.abs(local[M.TRUTH].to_numpy(np.float64) - truth)))
        if truth_gap > 0.01:
            raise RuntimeError(f"truth gap {truth_gap}")
        segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
        fold = int(local["fold"].iloc[0])
        scores = rankers[fold].predict(features)
        order = R._deduplicated_order(paths, scores)
        if len(order) < TOPK:
            raise RuntimeError(f"only {len(order)} unique candidates")
        retained = order[:TOPK]
        row_distance = local["md_since"].to_numpy(np.float64)
        progress = row_distance / max(float(row_distance[-1]), 1.0)
        trajectories = np.vstack([
            np.interp(
                grid,
                progress,
                B["_path_prediction"](record, prior, paths[index], segments)[eval_rows],
            )
            for index in retained
        ]).astype(np.float32)
        exact = np.interp(grid, progress, local["exact_v76"].to_numpy(np.float64))
        coarse = np.interp(grid, progress, local[M.COARSE].to_numpy(np.float64))
        truth_grid = np.interp(grid, progress, truth)
        move = trajectories.astype(np.float64) - coarse[None, :]
        uniform = np.mean(trajectories[:BASELINE_K], axis=0, dtype=np.float64)
        consensus_rms = np.sqrt(np.mean(np.square(trajectories - uniform[None, :]), axis=1))
        median_path = np.median(trajectories[:BASELINE_K], axis=0)
        median_rms = np.sqrt(np.mean(np.square(trajectories - median_path[None, :]), axis=1))
        score_z = (scores[retained] - np.mean(scores[retained])) / max(float(np.std(scores[retained])), 1e-4)
        path_summary = np.vstack([
            np.interp(summary_grid, grid, values) for values in move
        ]) / 16.0
        extra = np.column_stack([
            np.arange(TOPK, dtype=np.float64) / max(TOPK - 1, 1),
            score_z,
            consensus_rms / 16.0,
            median_rms / 16.0,
            path_summary,
        ])
        candidate_feature_rows.append(np.concatenate([
            features[retained].astype(np.float64), extra
        ], axis=1).astype(np.float32))
        trajectory_rows.append(trajectories)
        exact_rows.append(exact.astype(np.float32))
        coarse_rows.append(coarse.astype(np.float32))
        truth_rows.append(truth_grid.astype(np.float32))
        maximum_distance = float(np.max(row_distance))
        taper_rows.append(
            np.clip((grid * maximum_distance - FREEZE_FT) / RAMP_FT, 0.0, 1.0)
            .astype(np.float32)
        )
        taper_rows[-1] = taper_rows[-1] * taper_rows[-1] * (3.0 - 2.0 * taper_rows[-1])
        dose_rows.append(float(local["dose"].iloc[0]))
        distance_max_rows.append(maximum_distance)
        row_count_rows.append(len(local))
        fold_rows.append(fold)
        well_order.append(well)
        report_rows.append({
            "well": well, "fold": int(local["fold"].iloc[0]), "status": "built",
            "candidates": len(paths), "unique_candidates": len(order), "reason": "",
        })
    except Exception as exc:
        report_rows.append({
            "well": well, "fold": int(_well_frame(well)["fold"].iloc[0]),
            "status": "failed", "candidates": 0, "unique_candidates": 0,
            "reason": repr(exc),
        })
    if number % 20 == 0 or number == len(eligible_wells):
        failures = sum(row["status"] == "failed" for row in report_rows)
        print(
            f"V130 tensors {number}/{len(eligible_wells)} failures={failures} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )

report = pd.DataFrame(report_rows)
report.to_csv(OUT_DIR / "build_report.csv", index=False)
fold_array = np.asarray(fold_rows, np.int8)
coverage = {str(fold): int(np.sum(fold_array == fold)) for fold in (3, 4, 5)}
if coverage["3"] < 153 or coverage["4"] < 150 or coverage["5"] < 150:
    raise RuntimeError(f"Insufficient V130 coverage: {coverage}")

candidate_features_raw = np.stack(candidate_feature_rows).astype(np.float32)
trajectories = np.stack(trajectory_rows).astype(np.float32)
exact = np.stack(exact_rows).astype(np.float32)
coarse = np.stack(coarse_rows).astype(np.float32)
truth = np.stack(truth_rows).astype(np.float32)
taper = np.stack(taper_rows).astype(np.float32)
dose = np.asarray(dose_rows, np.float32)
row_count = np.asarray(row_count_rows, np.float32)
train_index = np.flatnonzero(np.isin(fold_array, TRAIN_FOLDS))
fold3_index = np.flatnonzero(fold_array == EVAL_FOLD)
fold3_wells = [well_order[index] for index in fold3_index]
discovery_index = fold3_index[:DISCOVERY_WELLS]
confirmation_index = fold3_index[DISCOVERY_WELLS:]
if len(discovery_index) != 24 or len(confirmation_index) < 129:
    raise RuntimeError(f"Fold3 split {len(discovery_index)}/{len(confirmation_index)}")

feature_mean = np.mean(candidate_features_raw[train_index], axis=(0, 1), keepdims=True)
feature_std = np.maximum(
    np.std(candidate_features_raw[train_index], axis=(0, 1), keepdims=True), 1e-3
)
candidate_features = ((candidate_features_raw - feature_mean) / feature_std).astype(np.float32)


class CandidateMixture(nn.Module):
    def __init__(self, feature_count, width=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, width), nn.LayerNorm(width), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(width, width), nn.SiLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(width * 3, width), nn.LayerNorm(width), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(width, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, values):
        encoded = self.encoder(values)
        mean = encoded.mean(dim=1, keepdim=True).expand_as(encoded)
        maximum = encoded.max(dim=1, keepdim=True).values.expand_as(encoded)
        logits = self.scorer(torch.cat([encoded, mean, maximum], dim=2)).squeeze(2)
        prior = torch.zeros_like(logits)
        prior[:, BASELINE_K:] = -4.0
        return torch.softmax(logits + prior, dim=1)


def _endpoint(weights_value, trajectories_value, exact_value, coarse_value, taper_value, dose_value):
    center = torch.sum(weights_value[:, :, None] * trajectories_value, dim=1)
    move = torch.clamp(dose_value[:, None] * (center - coarse_value), -CAP_FT, CAP_FT)
    return exact_value + taper_value * move


def _uniform_weights(count, device):
    values = torch.zeros((count, TOPK), dtype=torch.float32, device=device)
    values[:, :BASELINE_K] = 1.0 / BASELINE_K
    return values


_seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CandidateMixture(candidate_features.shape[2]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
dataset = TensorDataset(
    torch.from_numpy(candidate_features[train_index]),
    torch.from_numpy(trajectories[train_index]),
    torch.from_numpy(exact[train_index]),
    torch.from_numpy(coarse[train_index]),
    torch.from_numpy(truth[train_index]),
    torch.from_numpy(taper[train_index]),
    torch.from_numpy(dose[train_index]),
    torch.from_numpy(row_count[train_index]),
)
generator = torch.Generator().manual_seed(SEED)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)


def _predict_weights(indices):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(candidate_features[indices]).to(device)).cpu().numpy()


def _grid_prediction(indices, weights_value):
    values = torch.from_numpy(np.asarray(weights_value, np.float32)).to(device)
    with torch.no_grad():
        output = _endpoint(
            values,
            torch.from_numpy(trajectories[indices]).to(device),
            torch.from_numpy(exact[indices]).to(device),
            torch.from_numpy(coarse[indices]).to(device),
            torch.from_numpy(taper[indices]).to(device),
            torch.from_numpy(dose[indices]).to(device),
        )
    return output.cpu().numpy()


uniform_discovery = np.zeros((len(discovery_index), TOPK), np.float32)
uniform_discovery[:, :BASELINE_K] = 1.0 / BASELINE_K
baseline_discovery_grid = _grid_prediction(discovery_index, uniform_discovery)
best_loss = float("inf")
best_epoch = 0
best_state = None
stale = 0
history = []
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    losses = []
    for batch in loader:
        features_b, trajectories_b, exact_b, coarse_b, truth_b, taper_b, dose_b, row_count_b = [
            value.to(device) for value in batch
        ]
        optimizer.zero_grad(set_to_none=True)
        weights_b = model(features_b)
        prediction_b = _endpoint(
            weights_b, trajectories_b, exact_b, coarse_b, taper_b, dose_b
        )
        well_mse = torch.mean(torch.square(prediction_b - truth_b), dim=1)
        mse = torch.sum(row_count_b * well_mse) / torch.sum(row_count_b)
        entropy = -torch.mean(torch.sum(weights_b * torch.log(weights_b + 1e-8), dim=1))
        loss = mse - 0.002 * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    discovery_weights = _predict_weights(discovery_index)
    discovery_grid = _grid_prediction(discovery_index, discovery_weights)
    discovery_well_mse = np.mean(
        np.square(discovery_grid - truth[discovery_index]), axis=1
    )
    baseline_well_mse = np.mean(
        np.square(baseline_discovery_grid - truth[discovery_index]), axis=1
    )
    discovery_loss = float(np.average(
        discovery_well_mse, weights=row_count[discovery_index]
    ))
    baseline_loss = float(np.average(
        baseline_well_mse, weights=row_count[discovery_index]
    ))
    row = {
        "epoch": epoch, "train_loss": float(np.mean(losses)),
        "discovery_mse": discovery_loss, "uniform33_discovery_mse": baseline_loss,
    }
    history.append(row)
    if discovery_loss < best_loss - 1e-5:
        best_loss = discovery_loss
        best_epoch = epoch
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        stale = 0
    else:
        stale += 1
    print("V130 EPOCH", row, flush=True)
    if stale >= PATIENCE:
        break
if best_state is None:
    raise RuntimeError("V130 failed to select a checkpoint")
model.load_state_dict(best_state)
pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
torch.save({
    "state_dict": best_state, "feature_mean": feature_mean, "feature_std": feature_std,
    "topk": TOPK, "baseline_k": BASELINE_K, "nodes": NODES,
    "best_epoch": best_epoch,
}, MODEL_DIR / "best_candidate_mixture.pt")

learned_weights = _predict_weights(fold3_index)
uniform_weights = np.zeros_like(learned_weights)
uniform_weights[:, :BASELINE_K] = 1.0 / BASELINE_K
learned_grid = _grid_prediction(fold3_index, learned_weights)
uniform_grid = _grid_prediction(fold3_index, uniform_weights)


def _row_predictions(indices, grid_predictions):
    parts = []
    local_parts = []
    for local_index, global_index in enumerate(indices):
        well = well_order[global_index]
        local = _well_frame(well)
        distance = local["md_since"].to_numpy(np.float64)
        progress = distance / max(float(distance[-1]), 1.0)
        parts.append(np.interp(progress, grid, grid_predictions[local_index]))
        local_parts.append(local)
    return pd.concat(local_parts, ignore_index=True), np.concatenate(parts)


fold3_frame, uniform_prediction = _row_predictions(fold3_index, uniform_grid)
_, learned_prediction = _row_predictions(fold3_index, learned_grid)
discovery_mask = fold3_frame["well"].isin(set(fold3_wells[:DISCOVERY_WELLS])).to_numpy()
confirmation_mask = ~discovery_mask
baseline_discovery = R._metric(
    fold3_frame.loc[discovery_mask], uniform_prediction[discovery_mask], "uniform_top33"
)
baseline_all = R._metric(fold3_frame, uniform_prediction, "uniform_top33")
if abs(float(baseline_all["rmse"]) - 7.599560722) > 3e-3:
    raise RuntimeError(f"Uniform top33 parity failed: {baseline_all['rmse']}")

scan_rows = []
for alpha in ALPHAS:
    prediction = uniform_prediction + float(alpha) * (learned_prediction - uniform_prediction)
    result = R._metric(
        fold3_frame.loc[discovery_mask], prediction[discovery_mask], "learned_mixture"
    )
    result.update({
        "alpha": alpha,
        "gain": float(baseline_discovery["rmse"] - result["rmse"]),
        "far_gain": float(
            baseline_discovery["far_gt2000_rmse"] - result["far_gt2000_rmse"]
        ),
        "near_degradation": float(
            result["near_0_500_rmse"] - baseline_discovery["near_0_500_rmse"]
        ),
        "p95_gain": float(
            baseline_discovery["p95_well_rmse"] - result["p95_well_rmse"]
        ),
    })
    scan_rows.append(result)
scan = pd.DataFrame(scan_rows).sort_values("rmse", kind="stable")
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
eligible = scan[
    (scan["gain"] >= 0.30) & (scan["far_gain"] >= 0.20)
    & (scan["near_degradation"] <= 0.05) & (scan["p95_gain"] >= -0.20)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected_alpha = float(selected_row["alpha"])
selected_prediction = uniform_prediction + selected_alpha * (
    learned_prediction - uniform_prediction
)

summary_rows = []
for stage, mask in (
    ("fold3_discovery", discovery_mask),
    ("fold3_confirmation", confirmation_mask),
):
    summary_rows.append({"stage": stage, **R._metric(
        fold3_frame.loc[mask], uniform_prediction[mask], "uniform_top33"
    )})
    summary_rows.append({"stage": stage, **R._metric(
        fold3_frame.loc[mask], selected_prediction[mask], "learned_mixture"
    )})

rolled_weights = np.roll(
    learned_weights[DISCOVERY_WELLS:], max(1, len(confirmation_index) // 7), axis=0
)
rolled_grid = _grid_prediction(confirmation_index, rolled_weights)
confirmation_frame, rolled_raw = _row_predictions(confirmation_index, rolled_grid)
uniform_confirmation = uniform_prediction[confirmation_mask]
rolled_prediction = uniform_confirmation + selected_alpha * (
    rolled_raw - uniform_confirmation
)
summary_rows.append({"stage": "fold3_confirmation", **R._metric(
    confirmation_frame, rolled_prediction, "rolled_weight_control"
)})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)


def _summary_row(stage, endpoint):
    return summary[summary["stage"].eq(stage) & summary["endpoint"].eq(endpoint)].iloc[0]


conf_base = _summary_row("fold3_confirmation", "uniform_top33")
conf_candidate = _summary_row("fold3_confirmation", "learned_mixture")
conf_control = _summary_row("fold3_confirmation", "rolled_weight_control")
confirmation_values = {
    "gain": float(conf_base["rmse"] - conf_candidate["rmse"]),
    "far_gain": float(conf_base["far_gt2000_rmse"] - conf_candidate["far_gt2000_rmse"]),
    "near_degradation": float(conf_candidate["near_0_500_rmse"] - conf_base["near_0_500_rmse"]),
    "p95_gain": float(conf_base["p95_well_rmse"] - conf_candidate["p95_well_rmse"]),
    "aligned_increment_vs_rolled": float(conf_control["rmse"] - conf_candidate["rmse"]),
}
checks = {
    "discovery_gain_ge_0p30": float(selected_row["gain"]) >= 0.30,
    "confirmation_gain_ge_0p20": confirmation_values["gain"] >= 0.20,
    "confirmation_far_gain_ge_0p10": confirmation_values["far_gain"] >= 0.10,
    "near_degradation_le_0p05": confirmation_values["near_degradation"] <= 0.05,
    "p95_not_worse_by_gt_0p20": confirmation_values["p95_gain"] >= -0.20,
    "aligned_beats_rolled_ge_0p05": confirmation_values["aligned_increment_vs_rolled"] >= 0.05,
}
strict_pass = bool(len(eligible) and all(checks.values()))
output = fold3_frame[[
    "id", "well", "row_idx", "fold", M.TRUTH, "md_since", M.PARENT, M.COARSE,
    "exact_v76",
]].copy()
output["uniform_top33"] = uniform_prediction.astype(np.float32)
output["learned_mixture"] = selected_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fold3_mixture_oof.parquet", index=False, compression="zstd")
np.savez_compressed(
    OUT_DIR / "fold3_candidate_weights.npz",
    wells=np.asarray(fold3_wells),
    uniform=uniform_weights.astype(np.float32),
    learned=learned_weights.astype(np.float32),
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_direct_final_SSE_training_Fold3_discovery_confirmation",
    "sample_unit": "one complete physical well candidate set",
    "model_family": "DeepSets differentiable convex candidate mixture",
    "changed_objective": "final blended TVT SSE rather than candidate ranking",
    "candidate_count": TOPK,
    "matched_baseline": f"uniform top{BASELINE_K}",
    "training_shortlists": "Fold4 scored by Fold5 ranker; Fold5 scored by Fold4 ranker",
    "training_objective_weighting": "128-node trajectory MSE weighted by original evaluation-row count",
    "coverage": coverage,
    "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    "best_epoch": best_epoch,
    "selected_alpha": selected_alpha,
    "uniform_top33_fold3_rmse": float(baseline_all["rmse"]),
    "discovery": {
        "gain": float(selected_row["gain"]),
        "far_gain": float(selected_row["far_gain"]),
        "near_degradation": float(selected_row["near_degradation"]),
        "p95_gain": float(selected_row["p95_gain"]),
    },
    "confirmation": confirmation_values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> cross-fit the differentiable mixture and build deployment"
        if strict_pass else
        "FAIL -> candidate-bank evidence cannot learn a useful convex mixture; close bank consumption"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV130 DISCOVERY SCAN")
print(scan.to_string(index=False), flush=True)
print("\nV130 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV130 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
