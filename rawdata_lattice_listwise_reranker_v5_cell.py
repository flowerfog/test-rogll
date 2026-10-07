"""Neural complete-well listwise reranker for the frozen V4 top-384 bank.

V4 retained the V3 ceiling (Fold-2 top-384 oracle 4.595 ft) but unary MAP was
11.193 ft.  V5 compares candidates within one physical well and optimises a
soft whole-path target plus expected candidate MSE.  Fold 4/5 select epoch on
Fold 3; the frozen epoch is refit on Fold 3/4/5 and applied once to Fold 2.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("V5 requires the LightGBM used by V4") from exc


EXPERIMENT = "rawdata_lattice_listwise_reranker_v5"
ROOT = Path("/root/autodl-tmp/oof_exports") if Path("/root/autodl-tmp").exists() else Path.cwd() / "output"
V4_DIR = ROOT / "rawdata_lattice_coarse_ranker_v4"
OUT_DIR = ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not (V4_DIR / "verdict.json").exists():
    raise FileNotFoundError(f"Run V4 first: {V4_DIR / 'verdict.json'}")
V4_VERDICT = json.loads((V4_DIR / "verdict.json").read_text(encoding="utf-8"))
PANEL_DIR = V4_DIR / "panels"
FEATURE_NAMES = json.loads((V4_DIR / "feature_names.json").read_text(encoding="utf-8"))
INNER_COARSE_PATH = V4_DIR / "coarse_ranker.txt"
OUTER_COARSE_PATH = V4_DIR / "coarse_ranker_outer345.txt"
for required in (INNER_COARSE_PATH, OUTER_COARSE_PATH):
    if not required.exists():
        raise FileNotFoundError(required)

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
TOPK = int(V4_VERDICT["selected_topk"])
MAX_EPOCHS = 24
PATIENCE = 6
SEED = 20260721
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("V5 listwise reranker requires CUDA")


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _load_panel(well):
    with np.load(PANEL_DIR / f"{well}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _fold_table():
    # Recover fold membership from the V4 panel set and its original stable split.
    v3_candidates = [
        Path.cwd() / "rawdata_segment_lattice_nested_v1_cell.py",
        Path("/root/autodl-tmp/rawdata_segment_lattice_nested_v1_cell.py"),
    ]
    source = next((path for path in v3_candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError("Upload rawdata_segment_lattice_nested_v1_cell.py")
    text = source.read_text(encoding="utf-8")
    namespace = {"__name__": "v5_fold_definitions"}
    exec(compile(text.split("started = time.time()", 1)[0], str(source), "exec"), namespace)
    all_wells = [
        path.name.split("__", 1)[0]
        for path in sorted(
            (namespace["DATA_ROOT"] / "train").glob("*__horizontal_well.csv")
        )
    ]
    eligible = {path.stem for path in PANEL_DIR.glob("*.npz")}
    table = namespace["_stable_folds"](all_wells)
    return table[table["well"].isin(eligible)].reset_index(drop=True)


FOLDS = _fold_table()
FOLD_BY_WELL = dict(zip(FOLDS["well"], FOLDS["fold"]))
TRAIN_WELLS = [well for well in FOLD_BY_WELL if FOLD_BY_WELL[well] in TRAIN_FOLDS]
SELECTION_WELLS = [well for well in FOLD_BY_WELL if FOLD_BY_WELL[well] == SELECTION_FOLD]
APPLICATION_WELLS = [well for well in FOLD_BY_WELL if FOLD_BY_WELL[well] == APPLICATION_FOLD]


def _coarse_indices(panel, booster):
    score = booster.predict(panel["features"])
    return np.argsort(-score)[: min(TOPK, len(score))].astype(np.int64)


def _training_indices(panel, well, epoch):
    # Label-free training menu: half best generator-rank paths, half deterministic
    # rotating random paths.  It does not use candidate MSE.
    n = len(panel["mse"])
    count = min(max(TOPK, 512), n)
    first_count = min(count // 2, n)
    first = np.argsort(np.min(panel["arm_rank"], axis=1))[:first_count]
    remaining = np.setdiff1d(np.arange(n), first, assume_unique=False)
    random_count = min(count - len(first), len(remaining))
    seed = int.from_bytes(
        hashlib.sha256(f"{well}:{epoch}:{SEED}".encode("ascii")).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    random = rng.choice(remaining, size=random_count, replace=False) if random_count else np.empty(0, int)
    return np.r_[first, random].astype(np.int64)


def _normalizer(wells):
    total = np.zeros(len(FEATURE_NAMES), np.float64)
    total_sq = np.zeros(len(FEATURE_NAMES), np.float64)
    count = 0
    for well in wells:
        panel = _load_panel(well)
        indices = _training_indices(panel, well, 0)
        values = panel["features"][indices].astype(np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += len(values)
    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 1e-6)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)


class CompleteWellSetRanker(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_count, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=2)
        self.score = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 1))

    def forward(self, features):
        encoded = self.input(features)
        encoded = self.context(encoded[None])[0]
        return self.score(encoded).squeeze(-1)


def _loss(scores, mse, row_weight):
    rmse = torch.sqrt(mse.clamp_min(1e-6))
    relative = rmse - rmse.min()
    target = torch.softmax(-relative / 2.0, dim=0)
    log_probability = F.log_softmax(scores.float(), dim=0)
    probability = torch.softmax(scores.float(), dim=0)
    listwise = -(target * log_probability).sum()
    expected = torch.sum(probability * torch.clamp(mse / 64.0, max=64.0))
    best = torch.argmin(mse)
    hard = torch.topk(mse, k=min(32, len(mse)), largest=True).indices
    margin = F.softplus(1.0 - scores[best] + scores[hard]).mean()
    loss = row_weight * (listwise + 0.35 * expected + 0.10 * margin)
    return loss, {
        "listwise": float(listwise.detach().cpu()),
        "expected": float(expected.detach().cpu()),
        "margin": float(margin.detach().cpu()),
    }


def _features(panel, indices, mean, std, control="full", seed=0):
    values = panel["features"][indices].copy()
    gr_columns = np.asarray(
        [name.startswith("gr_") or name.startswith("gen_") for name in FEATURE_NAMES]
    )
    if control == "no_gr":
        values[:, gr_columns] = mean[gr_columns]
    elif control == "shuffled_alignment":
        rng = np.random.default_rng(seed)
        values[:, gr_columns] = values[rng.permutation(len(values))][:, gr_columns]
    elif control != "full":
        raise ValueError(control)
    return np.clip((values - mean) / std, -8.0, 8.0).astype(np.float32)


@torch.no_grad()
def _evaluate(model, wells, booster, mean, std, control="full"):
    model.eval()
    rows = []
    for well in wells:
        panel = _load_panel(well)
        indices = _coarse_indices(panel, booster)
        seed = int.from_bytes(hashlib.sha256(f"{control}:{well}".encode()).digest()[:8], "little")
        values = _features(panel, indices, mean, std, control, seed)
        scores = model(torch.from_numpy(values).to(DEVICE)).float().cpu().numpy()
        chosen = int(indices[int(np.argmax(scores))])
        oracle = int(indices[int(np.argmin(panel["mse"][indices]))])
        rows.append(
            {
                "well": well,
                "rows": int(panel["rows"][0]),
                "far_rows": int(panel["far_rows"][0]),
                "map_sse": float(panel["sse"][chosen]),
                "map_sae": float(panel["sae"][chosen]),
                "map_far_sse": float(panel["far_sse"][chosen]),
                "oracle_sse": float(panel["sse"][oracle]),
                "oracle_sae": float(panel["sae"][oracle]),
                "oracle_far_sse": float(panel["far_sse"][oracle]),
            }
        )
    frame = pd.DataFrame(rows)

    def metrics(prefix):
        count = int(frame["rows"].sum())
        far_count = int(frame["far_rows"].sum())
        sse = float(frame[f"{prefix}_sse"].sum())
        well_rmse = np.sqrt(frame[f"{prefix}_sse"] / frame["rows"])
        return {
            "rows": count,
            "wells": int(frame["well"].nunique()),
            "exact_sse": sse,
            "rmse": math.sqrt(sse / count),
            "mae": float(frame[f"{prefix}_sae"].sum() / count),
            "median_well_rmse": float(np.median(well_rmse)),
            "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
            "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
            "far_gt2000_rmse": math.sqrt(float(frame[f"{prefix}_far_sse"].sum()) / far_count),
        }

    return {"map": metrics("map"), "oracle": metrics("oracle")}, frame


def _train(train_wells, valid_wells, valid_booster, epochs, mean, std, phase):
    torch.manual_seed(SEED + (0 if phase == "inner" else 1000))
    torch.cuda.manual_seed_all(SEED + (0 if phase == "inner" else 1000))
    model = CompleteWellSetRanker(len(FEATURE_NAMES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best_rmse = np.inf
    best_epoch = 0
    bad_epochs = 0
    checkpoint = MODEL_DIR / f"best_{phase}.pt"
    median_rows = float(np.median([_load_panel(w)["rows"][0] for w in train_wells]))

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(train_wells)
        np.random.default_rng(SEED + epoch).shuffle(order)
        losses = []
        components = []
        for well in order:
            panel = _load_panel(well)
            indices = _training_indices(panel, well, epoch)
            permutation = np.random.default_rng(
                int.from_bytes(hashlib.sha256(f"perm:{well}:{epoch}".encode()).digest()[:8], "little")
            ).permutation(len(indices))
            indices = indices[permutation]
            values = _features(panel, indices, mean, std)
            mse = panel["mse"][indices].astype(np.float32)
            row_weight = float(np.clip(panel["rows"][0] / median_rows, 0.5, 2.0))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                scores = model(torch.from_numpy(values).to(DEVICE))
                loss, parts = _loss(
                    scores,
                    torch.from_numpy(mse).to(DEVICE),
                    row_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            components.append(parts)

        report = {
            "phase": phase,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_listwise": float(np.mean([row["listwise"] for row in components])),
            "train_expected": float(np.mean([row["expected"] for row in components])),
            "train_margin": float(np.mean([row["margin"] for row in components])),
        }
        if valid_wells:
            metrics, _ = _evaluate(model, valid_wells, valid_booster, mean, std)
            report.update({f"valid_{key}": value for key, value in metrics["map"].items()})
            current = metrics["map"]["rmse"]
            if current < best_rmse - 1e-4:
                best_rmse = current
                best_epoch = epoch
                bad_epochs = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                bad_epochs += 1
        history.append(report)
        print("V5 EPOCH", report, flush=True)
        if valid_wells and bad_epochs >= PATIENCE:
            break

    if valid_wells:
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    else:
        best_epoch = epochs
        torch.save(model.state_dict(), checkpoint)
    return model, best_epoch, pd.DataFrame(history), checkpoint


# --- V5 EXECUTION ---
started = time.time()
inner_booster = lgb.Booster(model_file=str(INNER_COARSE_PATH))
outer_booster = lgb.Booster(model_file=str(OUTER_COARSE_PATH))
mean_inner, std_inner = _normalizer(TRAIN_WELLS)
np.savez(OUT_DIR / "inner_normalizer.npz", mean=mean_inner, std=std_inner)

inner_model, selected_epoch, inner_history, inner_checkpoint = _train(
    TRAIN_WELLS,
    SELECTION_WELLS,
    inner_booster,
    MAX_EPOCHS,
    mean_inner,
    std_inner,
    "inner45_to_3",
)
inner_history.to_csv(OUT_DIR / "inner_history.csv", index=False)
fold3_metrics, fold3_bywell = _evaluate(
    inner_model, SELECTION_WELLS, inner_booster, mean_inner, std_inner
)
fold3_bywell.to_csv(OUT_DIR / "fold3_bywell.csv", index=False)

outer_wells = TRAIN_WELLS + SELECTION_WELLS
mean_outer, std_outer = _normalizer(outer_wells)
np.savez(OUT_DIR / "outer_normalizer.npz", mean=mean_outer, std=std_outer)
outer_model, _, outer_history, outer_checkpoint = _train(
    outer_wells,
    [],
    outer_booster,
    selected_epoch,
    mean_outer,
    std_outer,
    "outer345_to_2",
)
outer_history.to_csv(OUT_DIR / "outer_history.csv", index=False)

fold2_full, fold2_bywell = _evaluate(
    outer_model, APPLICATION_WELLS, outer_booster, mean_outer, std_outer, "full"
)
fold2_no_gr, _ = _evaluate(
    outer_model, APPLICATION_WELLS, outer_booster, mean_outer, std_outer, "no_gr"
)
fold2_shuffle, _ = _evaluate(
    outer_model,
    APPLICATION_WELLS,
    outer_booster,
    mean_outer,
    std_outer,
    "shuffled_alignment",
)
fold2_bywell.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)

controls = {
    "no_gr_map_degradation": fold2_no_gr["map"]["rmse"] - fold2_full["map"]["rmse"],
    "shuffled_alignment_map_degradation": fold2_shuffle["map"]["rmse"] - fold2_full["map"]["rmse"],
}
gates = {
    "top384_oracle_rmse_le_4p9": fold2_full["oracle"]["rmse"] <= 4.9,
    "reranker_map_rmse_le_8": fold2_full["map"]["rmse"] <= 8.0,
    "target_5p5": fold2_full["map"]["rmse"] <= 5.5,
    "each_control_degradation_ge_0p5": min(controls.values()) >= 0.5,
}
if gates["target_5p5"]:
    next_action = "PASS TARGET -> refit Fold2/3/4/5 and perform the untouched Fold1 confirmation"
elif gates["reranker_map_rmse_le_8"]:
    next_action = "ARCHITECTURE PASS -> audit tail disagreements before the Fold1 investment"
else:
    next_action = "FAIL -> candidate evidence cannot rank the 4.6-ft bank; stop model scaling"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_complete_well_neural_listwise_reranking",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "inner_train_folds": list(TRAIN_FOLDS),
        "epoch_selection_fold": SELECTION_FOLD,
        "outer_refit_folds": [3, 4, 5],
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "topk": TOPK,
    "features": len(FEATURE_NAMES),
    "parameters": int(sum(parameter.numel() for parameter in outer_model.parameters())),
    "selected_epoch": int(selected_epoch),
    "inner_training_wells": len(TRAIN_WELLS),
    "outer_refit_wells": len(outer_wells),
    "fold3": fold3_metrics,
    "fold2_full": fold2_full,
    "fold2_no_gr": fold2_no_gr,
    "fold2_shuffled_alignment": fold2_shuffle,
    "controls": controls,
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "next": next_action,
    "checkpoints": {
        "inner": str(inner_checkpoint),
        "outer": str(outer_checkpoint),
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV5 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)
