"""Complete-well DeepSets reranker with dynamic alias hard negatives.

The frozen V4 panels retain a strong top-384 oracle, but previous rankers did
not generalise.  V5 used easy worst-path negatives and V6 remained unary.
V60 scores the exact deployment menu as a set and mines the model's current
high-scoring non-top8 paths as hard negatives.  Fold 4/5 select the epoch on
Fold 3; a fresh model is then fit on Fold 3/4/5 and applied once to Fold 2.
Fold 1 remains untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERIMENT = "rawdata_lattice_deepset_hardnegative_v60b"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SOURCE = next(
    (
        path
        for path in (
            ROOT / "rawdata_lattice_oracle_mined_ranker_v6_cell.py",
            Path.cwd() / "rawdata_lattice_oracle_mined_ranker_v6_cell.py",
        )
        if path.is_file()
    ),
    None,
)
if SOURCE is None:
    raise FileNotFoundError("Upload rawdata_lattice_oracle_mined_ranker_v6_cell.py")

SEED = 20260728
MAX_EPOCHS = 18
PATIENCE = 5
TOP_POSITIVES = 8
HARD_NEGATIVES = 48
TARGET_TEMPERATURE_FT = 1.5
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 2e-4
GRAD_CLIP = 2.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("V60 requires CUDA")


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    marker = "\nstarted = time.time()\n"
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    module = types.ModuleType("raw_lattice_v6_definitions_for_v60")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(path), "exec"), module.__dict__)
    return module


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


def _seed(text):
    return int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:8], "little")


N = _bootstrap(SOURCE)


def _canonical_fold_sets():
    """Recover the exact fold lineage used when V4 built its panels/models."""
    candidates = (
        ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py",
        Path.cwd() / "rawdata_lattice_coarse_ranker_v4_cell.py",
    )
    source_path = next((path for path in candidates if path.is_file()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Upload rawdata_lattice_coarse_ranker_v4_cell.py so V60b can verify "
            "the canonical panel/fold lineage"
        )
    source = source_path.read_text(encoding="utf-8")
    marker = "\nwells = [\n"
    if marker not in source:
        raise RuntimeError(f"V4 execution marker missing from {source_path}")
    module = types.ModuleType("raw_lattice_v4_definitions_for_v60b")
    module.__file__ = str(source_path)
    sys.modules[module.__name__] = module
    exec(
        compile(source.split(marker, 1)[0] + "\n", str(source_path), "exec"),
        module.__dict__,
    )
    data_root = module.DATA_ROOT
    all_wells = [
        path.name.split("__", 1)[0]
        for path in sorted((data_root / "train").glob("*__horizontal_well.csv"))
    ]
    folds = module.V["_stable_folds"](all_wells)
    fold_by_well = dict(zip(folds["well"].astype(str), folds["fold"].astype(int)))
    available = {path.stem for path in N.PANEL_DIR.glob("*.npz")}

    def select(allowed):
        return [
            well
            for well in all_wells
            if fold_by_well.get(well) in allowed and well in available
        ]

    train = select(set(N.TRAIN_FOLDS))
    selection = select({int(N.SELECTION_FOLD)})
    application = select({int(N.APPLICATION_FOLD)})
    coverage = {
        "panel_files_total": len(available),
        "canonical_train_wells": len(train),
        "canonical_selection_wells": len(selection),
        "canonical_application_wells": len(application),
        "v6_train_wells": len(N.TRAIN_WELLS),
        "v6_selection_wells": len(N.SELECTION_WELLS),
        "v6_application_wells": len(N.APPLICATION_WELLS),
    }
    if len(train) < 307 or len(selection) < 154 or len(application) < 154:
        raise RuntimeError(
            "Canonical V4 panel coverage is incomplete. Rerun "
            "rawdata_lattice_coarse_ranker_v4_cell.py once; it will build the "
            f"missing canonical panels and refit both coarse models. Coverage: {coverage}"
        )
    print("V60b CANONICAL FOLD COVERAGE", coverage, flush=True)
    return train, selection, application, coverage


TRAIN_WELLS, SELECTION_WELLS, APPLICATION_WELLS, CANONICAL_COVERAGE = (
    _canonical_fold_sets()
)


def _build_menus(wells, booster, label):
    menus = {}
    row_counts = []
    oracle_coverage = []
    for number, well in enumerate(wells, 1):
        panel = N._load_panel(well)
        indices, coarse_score = N._coarse_menu(panel, booster)
        values = N._augment(panel, indices, coarse_score, "full", _seed(f"full:{well}"))
        rmse = np.sqrt(np.maximum(panel["mse"][indices], 0.0)).astype(np.float32)
        menus[well] = {
            "x": values.astype(np.float32),
            "rmse": rmse,
            "indices": indices,
            "rows": int(panel["rows"][0]),
            "far_rows": int(panel["far_rows"][0]),
        }
        row_counts.append(int(panel["rows"][0]))
        oracle_coverage.append(bool(int(np.argmin(panel["mse"])) in set(indices.tolist())))
        if number % 50 == 0 or number == len(wells):
            print(f"V60 menus {label} {number}/{len(wells)}", flush=True)
    median_rows = float(np.median(row_counts))
    for menu in menus.values():
        menu["query_weight"] = float(np.clip(menu["rows"] / median_rows, 0.5, 2.0))
    return menus, float(np.mean(oracle_coverage))


def _normalizer(menus):
    total = None
    total_sq = None
    count = 0
    for menu in menus.values():
        values = menu["x"].astype(np.float64)
        if total is None:
            total = np.zeros(values.shape[1], np.float64)
            total_sq = np.zeros(values.shape[1], np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += len(values)
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 1e-5))
    return mean.astype(np.float32), std.astype(np.float32)


def _normalise(values, mean, std):
    return np.clip((values - mean) / std, -8.0, 8.0).astype(np.float32)


class CandidateDeepSet(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(192, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(96 * 3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(128, 1),
        )

    def forward(self, features):
        encoded = self.encoder(features)
        mean = encoded.mean(dim=0, keepdim=True).expand_as(encoded)
        maximum = encoded.max(dim=0, keepdim=True).values.expand_as(encoded)
        return self.scorer(torch.cat([encoded, mean, maximum], dim=1)).squeeze(1)


def _loss(scores, candidate_rmse, query_weight):
    regret = candidate_rmse - candidate_rmse.min()
    target = torch.softmax(-regret / TARGET_TEMPERATURE_FT, dim=0)
    probability = torch.softmax(scores.float(), dim=0)
    listwise = -(target * F.log_softmax(scores.float(), dim=0)).sum()
    expected = torch.sum(probability * torch.clamp(regret / 8.0, max=8.0))

    positive = torch.topk(
        candidate_rmse, k=min(TOP_POSITIVES, len(candidate_rmse)), largest=False
    ).indices
    positive_mask = torch.zeros_like(scores, dtype=torch.bool)
    positive_mask[positive] = True
    negative_order = torch.argsort(scores.detach(), descending=True)
    hard = negative_order[~positive_mask[negative_order]][:HARD_NEGATIVES]
    if len(hard):
        best_positive = positive[torch.argmax(scores[positive].detach())]
        required_margin = torch.clamp(
            (candidate_rmse[hard] - candidate_rmse[best_positive]) / 8.0,
            min=0.25,
            max=3.0,
        )
        hard_margin = F.softplus(
            required_margin - scores[best_positive] + scores[hard]
        ).mean()
    else:
        hard_margin = scores.sum() * 0.0
    loss = float(query_weight) * (listwise + 0.60 * expected + 0.35 * hard_margin)
    return loss, {
        "listwise": float(listwise.detach().cpu()),
        "expected": float(expected.detach().cpu()),
        "hard_margin": float(hard_margin.detach().cpu()),
    }


def _metric_frame(rows):
    frame = pd.DataFrame(rows)
    output = {}
    for prefix in ("map", "coarse", "oracle"):
        count = int(frame["rows"].sum())
        far_count = int(frame["far_rows"].sum())
        sse = float(frame[f"{prefix}_sse"].sum())
        well_rmse = np.sqrt(frame[f"{prefix}_sse"] / frame["rows"])
        output[prefix] = {
            "rows": count,
            "wells": int(frame["well"].nunique()),
            "exact_sse": sse,
            "rmse": math.sqrt(sse / count),
            "mae": float(frame[f"{prefix}_sae"].sum() / count),
            "median_well_rmse": float(np.median(well_rmse)),
            "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
            "far_gt2000_rmse": math.sqrt(
                float(frame[f"{prefix}_far_sse"].sum()) / max(far_count, 1)
            ),
        }
    return output, frame


@torch.no_grad()
def _evaluate(model, wells, booster, mean, std, control="full"):
    model.eval()
    rows = []
    for number, well in enumerate(wells, 1):
        panel = N._load_panel(well)
        indices, coarse_score = N._coarse_menu(panel, booster)
        values = N._augment(
            panel, indices, coarse_score, control, _seed(f"{control}:{well}")
        )
        values = _normalise(values, mean, std)
        score = model(torch.from_numpy(values).to(DEVICE)).float().cpu().numpy()
        chosen = int(indices[int(np.argmax(score))])
        coarse = int(indices[0])
        oracle = int(indices[int(np.argmin(panel["mse"][indices]))])
        row = {
            "well": well,
            "rows": int(panel["rows"][0]),
            "far_rows": int(panel["far_rows"][0]),
            "chosen_regret_rmse": float(
                np.sqrt(panel["mse"][chosen]) - np.sqrt(panel["mse"][oracle])
            ),
        }
        for prefix, index in (("map", chosen), ("coarse", coarse), ("oracle", oracle)):
            row[f"{prefix}_sse"] = float(panel["sse"][index])
            row[f"{prefix}_sae"] = float(panel["sae"][index])
            row[f"{prefix}_far_sse"] = float(panel["far_sse"][index])
        rows.append(row)
        if number % 50 == 0 or number == len(wells):
            print(f"V60 evaluate {control} {number}/{len(wells)}", flush=True)
    return _metric_frame(rows)


def _train(train_wells, valid_wells, train_booster, valid_booster, epochs, phase):
    torch.manual_seed(SEED + (0 if phase == "inner45_to_3" else 1000))
    torch.cuda.manual_seed_all(SEED + (0 if phase == "inner45_to_3" else 1000))
    menus, coverage = _build_menus(train_wells, train_booster, f"{phase}_train")
    mean, std = _normalizer(menus)
    feature_count = next(iter(menus.values()))["x"].shape[1]
    model = CandidateDeepSet(feature_count).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    checkpoint = MODEL_DIR / f"best_{phase}.pt"
    history = []
    best_rmse = np.inf
    best_epoch = 0
    bad_epochs = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = list(train_wells)
        np.random.default_rng(SEED + epoch + (0 if phase == "inner45_to_3" else 1000)).shuffle(order)
        losses = []
        components = []
        for well in order:
            menu = menus[well]
            values = _normalise(menu["x"], mean, std)
            features = torch.from_numpy(values).to(DEVICE)
            candidate_rmse = torch.from_numpy(menu["rmse"]).to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            scores = model(features)
            loss, component = _loss(scores, candidate_rmse, menu["query_weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            components.append(component)
        report = {
            "phase": phase,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_listwise": float(np.mean([x["listwise"] for x in components])),
            "train_expected": float(np.mean([x["expected"] for x in components])),
            "train_hard_margin": float(np.mean([x["hard_margin"] for x in components])),
        }
        if valid_wells:
            metrics, _ = _evaluate(model, valid_wells, valid_booster, mean, std)
            current = float(metrics["map"]["rmse"])
            report.update(
                {
                    "valid_rmse": current,
                    "valid_coarse_rmse": metrics["coarse"]["rmse"],
                    "valid_oracle_rmse": metrics["oracle"]["rmse"],
                    "valid_far_rmse": metrics["map"]["far_gt2000_rmse"],
                }
            )
            if current < best_rmse - 1e-4:
                best_rmse = current
                best_epoch = epoch
                bad_epochs = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                bad_epochs += 1
        else:
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        history.append(report)
        print("V60 EPOCH", report, flush=True)
        if valid_wells and bad_epochs >= PATIENCE:
            break
    if valid_wells:
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    del menus
    return model, best_epoch, mean, std, coverage, pd.DataFrame(history), checkpoint


started = time.time()
inner_booster = lgb.Booster(model_file=str(N.INNER_COARSE_PATH))
outer_booster = lgb.Booster(model_file=str(N.OUTER_COARSE_PATH))

inner_model, selected_epoch, inner_mean, inner_std, inner_coverage, history, _ = _train(
    TRAIN_WELLS,
    SELECTION_WELLS,
    inner_booster,
    inner_booster,
    MAX_EPOCHS,
    "inner45_to_3",
)
history.to_csv(OUT_DIR / "inner_history.csv", index=False)
fold3_metrics, fold3_frame = _evaluate(
    inner_model, SELECTION_WELLS, inner_booster, inner_mean, inner_std
)
fold3_frame.to_csv(OUT_DIR / "fold3_bywell.csv", index=False)

outer_wells = TRAIN_WELLS + SELECTION_WELLS
outer_model, _, outer_mean, outer_std, outer_coverage, outer_history, outer_checkpoint = _train(
    outer_wells,
    [],
    outer_booster,
    outer_booster,
    selected_epoch,
    "outer345_to_2",
)
outer_history.to_csv(OUT_DIR / "outer_history.csv", index=False)
np.savez(OUT_DIR / "outer_normalizer.npz", mean=outer_mean, std=outer_std)

fold2_metrics, fold2_frame = _evaluate(
    outer_model, APPLICATION_WELLS, outer_booster, outer_mean, outer_std, "full"
)
fold2_no_gr, _ = _evaluate(
    outer_model, APPLICATION_WELLS, outer_booster, outer_mean, outer_std, "no_gr"
)
fold2_shuffle, _ = _evaluate(
    outer_model,
    APPLICATION_WELLS,
    outer_booster,
    outer_mean,
    outer_std,
    "shuffled_alignment",
)
fold2_frame.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)

controls = {
    "no_gr_degradation": float(fold2_no_gr["map"]["rmse"] - fold2_metrics["map"]["rmse"]),
    "shuffled_alignment_degradation": float(
        fold2_shuffle["map"]["rmse"] - fold2_metrics["map"]["rmse"]
    ),
}
gates = {
    "fold2_oracle_rmse_le_4p9": bool(fold2_metrics["oracle"]["rmse"] <= 4.9),
    "gain_vs_coarse_ge_0p30": bool(
        fold2_metrics["coarse"]["rmse"] - fold2_metrics["map"]["rmse"] >= 0.30
    ),
    "map_rmse_le_9p5": bool(fold2_metrics["map"]["rmse"] <= 9.5),
    "each_control_degradation_ge_0p25": bool(min(controls.values()) >= 0.25),
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold45_complete_well_hardnegative_training_Fold3_epoch_selection_Fold2_confirmation",
    "sample_unit": "one complete physical well with exact V4 top384 menu",
    "legal_inputs_only": True,
    "candidate_truth_usage": "outer-train supervised regret and hard-negative labels only",
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "inner_train_folds": list(N.TRAIN_FOLDS),
        "epoch_selection_fold": N.SELECTION_FOLD,
        "outer_refit_folds": [3, 4, 5],
        "application_fold": N.APPLICATION_FOLD,
        "untouched_fold": N.UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "canonical_fold_coverage": CANONICAL_COVERAGE,
    "architecture": "candidate encoder plus mean/max DeepSets context",
    "parameters": int(sum(parameter.numel() for parameter in outer_model.parameters())),
    "topk": N.TOPK,
    "selected_epoch": int(selected_epoch),
    "top_positives": TOP_POSITIVES,
    "dynamic_hard_negatives": HARD_NEGATIVES,
    "inner_menu_global_oracle_coverage": inner_coverage,
    "outer_menu_global_oracle_coverage": outer_coverage,
    "fold3": fold3_metrics,
    "fold2": fold2_metrics,
    "fold2_no_gr": fold2_no_gr,
    "fold2_shuffled_alignment": fold2_shuffle,
    "controls": controls,
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "checkpoint": outer_checkpoint,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> confirm on untouched Fold1 and test cross-evidence against 6.377"
        if all(gates.values())
        else "FAIL -> supervised set ranking still cannot generalise; close the V3 bank selector"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV60 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
