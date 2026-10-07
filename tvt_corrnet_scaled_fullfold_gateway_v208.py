"""V208: staged scaling gateway for the synthetic-pretrained axial CorrNet.

This file is intentionally standalone.  It consumes only the frozen V4 panel
cache and fold metadata, trains on folds 4/5, and evaluates complete fold 2.
Fold 3 and fold 1 are never loaded.  The likelihood-PF path is used only as the
centre of the state axis already encoded in the frozen panels; no V144/V168
prediction or candidate-bank endpoint is consumed.

The experiment answers one narrow question: does model scaling improve a
causally GR-conditioned trajectory decoder?  Small is always run.  Medium is
run next.  Large is launched only when medium improves complete-well RMSE by
the preregistered margin and passes no-GR / shuffled-typewell controls.

The script is safe to rerun: pretraining and fine-tuning resume from per-epoch
checkpoints, while completed profiles are loaded from their final reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import json
import math
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


EXPERIMENT = "tvt_corrnet_scaled_fullfold_gateway_v208"
ROOT = Path(os.environ.get("V208_ROOT", "/root/autodl-tmp"))
OOF_ROOT = ROOT / "oof_exports"
V4_DIR = OOF_ROOT / "tvt_corrnet_heel_conditioned_onefold_v4"
PANEL_DIR = V4_DIR / "panels"
GUARD_PATH = V4_DIR / "panel_guards.csv"
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (4, 5)
GATE_FOLD = 2
UNTOUCHED_FOLDS = (1, 3)
TRAIN_TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")
STATE_OFFSETS = np.arange(-96.0, 96.01, 2.0, dtype=np.float32)
PREFIX_COLUMNS = 64
N_COLUMNS = 320
LOCAL_OFFSETS_FT = (-16.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 16.0)
ITERATION_GAMMA = 0.82
SEED = 20260804
PRETRAIN_SEED = 20260805
WEIGHT_DECAY = 2e-4
MIN_LEARNING_RATE = 2e-5
PATH_TRANSPLANT_PROBABILITY = 0.50
RESIDUAL_SCALE = 1.0
MIN_REFERENCE_COVERAGE = 0.85
MEDIUM_SCALE_GATE = 0.30
CAUSAL_GATE = 0.30


@dataclass(frozen=True)
class ScaleProfile:
    name: str
    hidden: int
    cost: int
    tcn_depth: int
    transformer_layers: int
    heads: int
    iterations: int
    batch_size: int
    synthetic_multiplier: int
    pretrain_epochs: int
    real_epochs: int
    min_real_epochs: int
    patience: int
    learning_rate: float


PROFILES = (
    ScaleProfile("small", 64, 16, 4, 2, 4, 8, 3, 2, 4, 12, 8, 4, 3e-4),
    ScaleProfile("medium", 128, 32, 6, 4, 8, 12, 2, 2, 5, 18, 10, 6, 2e-4),
    ScaleProfile("large", 256, 48, 8, 6, 8, 16, 1, 2, 6, 24, 12, 8, 1.5e-4),
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_clean(value), indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _panel_cache_compatible(directory: Path) -> bool:
    guard_path = directory / "panel_guards.csv"
    panel_path = directory / "panels"
    if not guard_path.is_file() or not panel_path.is_dir():
        return False
    samples = list(panel_path.glob("*.npz"))[:1]
    if not samples:
        return False
    required = {
        "horizontal_gr", "typewell_gr", "heel_reference_gr", "geometry",
        "center_tvt", "truth_residual", "eval_rows", "eval_truth",
    }
    try:
        with np.load(samples[0], allow_pickle=False) as data:
            return required.issubset(set(data.files))
    except (OSError, ValueError):
        return False


def _ensure_panel_cache() -> None:
    """Find a moved V4 cache or rebuild only the folds required by V208."""
    global PANEL_DIR, GUARD_PATH
    if _panel_cache_compatible(V4_DIR):
        return

    candidates = []
    if OOF_ROOT.is_dir():
        for guard_path in OOF_ROOT.rglob("panel_guards.csv"):
            candidate = guard_path.parent
            if candidate != V4_DIR and _panel_cache_compatible(candidate):
                candidates.append(candidate)
    if candidates:
        candidates.sort(
            key=lambda path: sum(1 for _ in (path / "panels").glob("*.npz")),
            reverse=True,
        )
        selected = candidates[0]
        PANEL_DIR = selected / "panels"
        GUARD_PATH = selected / "panel_guards.csv"
        print(
            "V208 discovered compatible panel cache",
            {"directory": str(selected), "candidates": len(candidates)},
            flush=True,
        )
        return

    source_candidates = (
        ROOT / "tvt_corrnet_heel_conditioned_onefold_v4_cell.py",
        Path.cwd() / "tvt_corrnet_heel_conditioned_onefold_v4_cell.py",
    )
    source_path = next((path for path in source_candidates if path.is_file()), None)
    if source_path is None:
        raise FileNotFoundError(
            "V208 panel cache is absent. Upload "
            "tvt_corrnet_heel_conditioned_onefold_v4_cell.py beside V208; "
            "V208 will rebuild only Fold4/5 training panels and Fold2 official panels."
        )
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if free_gb < 2.0:
        raise RuntimeError(
            f"V208 panel rebuild needs at least 2 GB free disk; found {free_gb:.2f} GB"
        )
    source = source_path.read_text(encoding="utf-8")
    replacements = {
        "VALID_FOLD = 1": "VALID_FOLD = 2",
        'train_wells = fold.loc[fold["fold"] != VALID_FOLD, "well"].tolist()':
            'train_wells = fold.loc[fold["fold"].isin((4, 5)), "well"].tolist()',
        'valid_wells = fold.loc[fold["fold"] == VALID_FOLD, "well"].tolist()':
            'valid_wells = fold.loc[fold["fold"] == 2, "well"].tolist()',
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"V208 cannot patch V4 panel builder; marker missing: {old}")
        source = source.replace(old, new, 1)
    stop_marker = 'train_tasks["key"] = train_tasks.apply('
    if stop_marker not in source:
        raise RuntimeError("V208 cannot locate the V4 build/train boundary")
    build_source = source.split(stop_marker, 1)[0]
    print(
        "V208 rebuilding required panel cache",
        {
            "source": str(source_path),
            "train_folds": TRAIN_FOLDS,
            "official_gate_fold": GATE_FOLD,
            "free_gb": free_gb,
        },
        flush=True,
    )
    namespace = {"__name__": "v208_panel_rebuild", "__file__": str(source_path)}
    exec(compile(build_source, str(source_path), "exec"), namespace)
    namespace.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not _panel_cache_compatible(V4_DIR):
        raise RuntimeError("V208 panel-only rebuild completed without a compatible cache")
    PANEL_DIR = V4_DIR / "panels"
    GUARD_PATH = V4_DIR / "panel_guards.csv"


def _amp_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _cpu_state(model: nn.Module):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


class PanelStore:
    def __init__(self, keys, label):
        self.records = []
        started = time.time()
        for number, key in enumerate(keys, 1):
            path = PANEL_DIR / f"{key}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data:
                self.records.append(
                    {
                        "well": str(data["well"][0]),
                        "tag": str(data["tag"][0]),
                        "horizontal_gr": data["horizontal_gr"].astype(np.float32),
                        "horizontal_valid": data["horizontal_valid"].astype(np.float32),
                        "typewell_gr": data["typewell_gr"].astype(np.float32),
                        "typewell_valid": data["typewell_valid"].astype(np.float32),
                        "heel_reference_gr": data["heel_reference_gr"].astype(np.float32),
                        "heel_reference_valid": data["heel_reference_valid"].astype(np.float32),
                        "geometry": data["geometry"].astype(np.float32),
                        "eval_mask": data["eval_mask"].astype(np.float32),
                        "center_tvt": data["center_tvt"].astype(np.float32),
                        "truth_residual": data["truth_residual"].astype(np.float32),
                        "column_md": data["column_md"].astype(np.float32),
                        "column_z": data["column_z"].astype(np.float32),
                        "eval_rows": data["eval_rows"].astype(np.int32),
                        "eval_md": data["eval_md"].astype(np.float32),
                        "eval_z": data["eval_z"].astype(np.float32),
                        "eval_truth": data["eval_truth"].astype(np.float32),
                        "panel_weight": float(data["panel_weight"][0]),
                    }
                )
            if number % 200 == 0 or number == len(keys):
                print(
                    f"V208 load {label} {number}/{len(keys)} "
                    f"elapsed_min={(time.time() - started) / 60.0:.1f}",
                    flush=True,
                )


def _load_stores():
    _ensure_panel_cache()
    for required in (PANEL_DIR, GUARD_PATH, FOLD_PATH):
        if not required.exists():
            raise FileNotFoundError(required)
    fold = pd.read_csv(FOLD_PATH, dtype={"well": str})
    fold["well"] = fold["well"].astype(str)
    fold["fold"] = pd.to_numeric(fold["fold"], errors="raise").astype(int)
    coverage = fold.groupby("fold")["well"].nunique().to_dict()
    if any(int(coverage.get(value, 0)) < 154 for value in range(1, 6)):
        raise RuntimeError(f"Incomplete five-fold metadata: {coverage}")

    guard = pd.read_csv(GUARD_PATH, dtype={"well": str, "tag": str})
    guard["well"] = guard["well"].astype(str)
    guard["tag"] = guard["tag"].astype(str)
    usable = guard["usable"].astype(str).str.lower().isin(("true", "1", "yes"))
    guard = guard.loc[usable & guard["tag"].isin(TRAIN_TAGS), ["well", "tag"]]
    guard = guard.merge(fold[["well", "fold"]], on="well", how="left", validate="many_to_one")
    if guard["fold"].isna().any():
        raise RuntimeError("Panel guard contains wells absent from fold metadata")

    train = guard[guard["fold"].isin(TRAIN_FOLDS)].sort_values(["well", "tag"])
    gate = guard[(guard["fold"].eq(GATE_FOLD)) & guard["tag"].eq("official")]
    gate = gate.sort_values(["well", "tag"])
    train_wells = train["well"].nunique()
    gate_wells = gate["well"].nunique()
    if train_wells < 309 or gate_wells < 155:
        raise RuntimeError(
            f"Incomplete V208 coverage: train={train_wells}, gate={gate_wells}"
        )
    train_keys = [f"{row.well}__{row.tag}" for row in train.itertuples(index=False)]
    gate_keys = [f"{row.well}__{row.tag}" for row in gate.itertuples(index=False)]
    return PanelStore(train_keys, "train45"), PanelStore(gate_keys, "fold2"), coverage


class PanelDataset(Dataset):
    def __init__(self, store, mode="full"):
        self.store = store
        self.mode = mode

    def __len__(self):
        return len(self.store.records)

    def __getitem__(self, index):
        record = self.store.records[index]
        horizontal = record["horizontal_gr"].copy()
        horizontal_valid = record["horizontal_valid"].copy()
        typewell = record["typewell_gr"].copy()
        typewell_valid = record["typewell_valid"].copy()
        heel = record["heel_reference_gr"].copy()
        heel_valid = record["heel_reference_valid"].copy()
        if self.mode == "shuffled_typewell":
            donor = self.store.records[
                (index + max(1, len(self.store.records) // 3)) % len(self.store.records)
            ]
            typewell = donor["typewell_gr"].copy()
            typewell_valid = donor["typewell_valid"].copy()
        elif self.mode == "no_gr":
            horizontal[:] = 0.0
            typewell[:] = 0.0
            heel[:] = 0.0
        elif self.mode != "full":
            raise ValueError(self.mode)
        return {
            "horizontal": torch.from_numpy(horizontal),
            "horizontal_valid": torch.from_numpy(horizontal_valid),
            "typewell": torch.from_numpy(typewell),
            "typewell_valid": torch.from_numpy(typewell_valid),
            "heel": torch.from_numpy(heel),
            "heel_valid": torch.from_numpy(heel_valid),
            "geometry": torch.from_numpy(record["geometry"]),
            "truth_residual": torch.from_numpy(record["truth_residual"]),
            "eval_mask": torch.from_numpy(record["eval_mask"]),
            "column_md": torch.from_numpy(record["column_md"]),
            "column_z": torch.from_numpy(record["column_z"]),
            "panel_weight": torch.tensor(record["panel_weight"], dtype=torch.float32),
        }


def _sample_surface(surface, validity, correction):
    surface = np.asarray(surface, np.float32)
    validity = np.asarray(validity, np.float32)
    correction = np.asarray(correction, np.float32)
    step = float(np.median(np.diff(STATE_OFFSETS)))
    position = (correction - float(STATE_OFFSETS[0])) / step
    inside = (position >= 0.0) & (position <= len(STATE_OFFSETS) - 1)
    position = np.clip(position, 0.0, len(STATE_OFFSETS) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(STATE_OFFSETS) - 1)
    fraction = position - lower
    columns = np.arange(surface.shape[1], dtype=np.int64)
    sampled = (1.0 - fraction) * surface[lower, columns] + fraction * surface[upper, columns]
    valid = inside & (validity[lower, columns] > 0.5) & (validity[upper, columns] > 0.5)
    sampled = sampled.astype(np.float32)
    sampled[~valid] = 0.0
    return sampled, valid.astype(np.float32)


def _fill(values, valid):
    values = np.asarray(values, np.float32).copy()
    valid = np.asarray(valid, bool)
    positions = np.arange(len(values), dtype=np.float32)
    if int(valid.sum()) >= 2:
        return np.interp(positions, positions[valid], values[valid]).astype(np.float32)
    if int(valid.sum()) == 1:
        values[:] = float(values[valid][0])
    else:
        values[:] = 0.0
    return values


class ResidualBank:
    def __init__(self, store):
        self.rows = []
        for record in store.records:
            truth = np.asarray(record["truth_residual"], np.float32)
            reference, reference_valid = _sample_surface(
                record["typewell_gr"], record["typewell_valid"], truth
            )
            horizontal = np.asarray(record["horizontal_gr"], np.float32)
            horizontal_valid = np.asarray(record["horizontal_valid"], np.float32) > 0.5
            valid = horizontal_valid & (reference_valid > 0.5)
            residual = _fill(horizontal - reference, valid)
            if int(valid.sum()) >= 12 and float(np.var(reference[valid])) > 1e-6:
                x = reference[valid].astype(np.float64)
                y = residual[valid].astype(np.float64)
                xc = x - x.mean()
                denominator = float(np.dot(xc, xc))
                beta = float(np.clip(np.dot(xc, y - y.mean()) / denominator, -1.5, 1.5))
                intercept = float(np.mean(y) - beta * np.mean(x))
            else:
                beta = 0.0
                intercept = float(np.mean(residual[valid])) if np.any(valid) else 0.0
            self.rows.append(
                {
                    "horizontal_valid": horizontal_valid.astype(np.float32),
                    "orthogonal": (residual - (intercept + beta * reference)).astype(np.float32),
                    "beta": beta,
                    "intercept": intercept,
                }
            )


class SyntheticPanelDataset(Dataset):
    def __init__(self, store, bank, multiplier, seed):
        self.store = store
        self.bank = bank
        self.multiplier = int(multiplier)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.store.records) * self.multiplier

    def _render(self, index):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 104_729)
        target_index = int(index) % len(self.store.records)
        target = self.store.records[target_index]
        donor_index = int(rng.integers(0, len(self.store.records) - 1))
        if donor_index >= target_index:
            donor_index += 1
        donor = self.bank.rows[donor_index]
        target_truth = np.asarray(target["truth_residual"], np.float32)
        correction = target_truth.copy()
        transplanted = bool(rng.random() < PATH_TRANSPLANT_PROBABILITY)
        if transplanted:
            path_index = int(rng.integers(0, len(self.store.records) - 1))
            if path_index >= target_index:
                path_index += 1
            candidate = np.asarray(self.store.records[path_index]["truth_residual"], np.float32).copy()
            eval_columns = np.flatnonzero(np.asarray(target["eval_mask"]) > 0.5)
            if len(eval_columns):
                first = int(eval_columns[0])
                candidate += float(target_truth[first] - candidate[first])
            candidate[:PREFIX_COLUMNS] = 0.0
            candidate = np.clip(candidate, STATE_OFFSETS[1], STATE_OFFSETS[-2]).astype(np.float32)
            _, candidate_valid = _sample_surface(
                target["typewell_gr"], target["typewell_valid"], candidate
            )
            coverage = float(candidate_valid[np.asarray(target["eval_mask"]) > 0.5].mean())
            if coverage >= MIN_REFERENCE_COVERAGE:
                correction = candidate
            else:
                transplanted = False

        reference, reference_valid = _sample_surface(
            target["typewell_gr"], target["typewell_valid"], correction
        )
        transferred = donor["intercept"] + donor["beta"] * reference + donor["orthogonal"]
        horizontal = np.clip(reference + RESIDUAL_SCALE * transferred, -8.0, 8.0).astype(np.float32)
        horizontal_valid = (
            (reference_valid > 0.5) & (np.asarray(donor["horizontal_valid"]) > 0.5)
        ).astype(np.float32)
        horizontal[horizontal_valid < 0.5] = 0.0
        horizontal[:PREFIX_COLUMNS] = target["horizontal_gr"][:PREFIX_COLUMNS]
        horizontal_valid[:PREFIX_COLUMNS] = target["horizontal_valid"][:PREFIX_COLUMNS]
        correction[:PREFIX_COLUMNS] = 0.0
        return target, horizontal, horizontal_valid, correction, transplanted

    def audit(self, index):
        target, horizontal, valid, correction, transplanted = self._render(index)
        mask = np.asarray(target["eval_mask"]) > 0.5
        return {
            "transplanted": transplanted,
            "horizontal_coverage": float(valid[mask].mean()),
            "correction_p95": float(np.quantile(np.abs(correction[mask]), 0.95)),
        }

    def __getitem__(self, index):
        target, horizontal, horizontal_valid, correction, _ = self._render(index)
        return {
            "horizontal": torch.from_numpy(horizontal.copy()),
            "horizontal_valid": torch.from_numpy(horizontal_valid.copy()),
            "typewell": torch.from_numpy(target["typewell_gr"].copy()),
            "typewell_valid": torch.from_numpy(target["typewell_valid"].copy()),
            "heel": torch.from_numpy(target["heel_reference_gr"].copy()),
            "heel_valid": torch.from_numpy(target["heel_reference_valid"].copy()),
            "geometry": torch.from_numpy(target["geometry"].copy()),
            "truth_residual": torch.from_numpy(correction.copy()),
            "eval_mask": torch.from_numpy(target["eval_mask"].copy()),
            "column_md": torch.from_numpy(target["column_md"].copy()),
            "column_z": torch.from_numpy(target["column_z"].copy()),
            "panel_weight": torch.tensor(target["panel_weight"], dtype=torch.float32),
        }


class ConvResidual1D(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 4
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return F.gelu(values + self.block(values))


class AxialContextEncoder(nn.Module):
    def __init__(self, input_channels, profile):
        super().__init__()
        channels = profile.hidden
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels, 5, padding=2),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        dilations = [2 ** (index % 6) for index in range(profile.tcn_depth)]
        self.tcn = nn.Sequential(*[ConvResidual1D(channels, dilation) for dilation in dilations])
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=profile.heads,
            dim_feedforward=channels * 4,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.axial = nn.TransformerEncoder(layer, num_layers=profile.transformer_layers)
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.GroupNorm(8, channels), nn.GELU()
        )

    def forward(self, values):
        hidden = self.tcn(self.stem(values))
        hidden = self.axial(hidden.transpose(1, 2)).transpose(1, 2)
        return self.output(hidden)


class ConvGRU1D(nn.Module):
    def __init__(self, hidden_channels, input_channels):
        super().__init__()
        self.gates = nn.Conv1d(hidden_channels + input_channels, hidden_channels * 2, 5, padding=2)
        self.candidate = nn.Conv1d(hidden_channels + input_channels, hidden_channels, 5, padding=2)

    def forward(self, hidden, values):
        combined = torch.cat([hidden, values], dim=1)
        reset, update = torch.sigmoid(self.gates(combined)).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([reset * hidden, values], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class ScaledAxialRAFT(nn.Module):
    def __init__(self, geometry_channels, profile):
        super().__init__()
        self.profile = profile
        hidden = profile.hidden
        cost = profile.cost
        self.context = AxialContextEncoder(2 + geometry_channels, profile)
        self.cost_encoder = nn.Sequential(
            nn.Conv2d(10, max(24, cost), 3, padding=1),
            nn.GroupNorm(8, max(24, cost)),
            nn.GELU(),
            nn.Conv2d(max(24, cost), cost, 3, padding=1),
            nn.GroupNorm(8, cost),
            nn.GELU(),
        )
        sampled_channels = cost * len(LOCAL_OFFSETS_FT)
        self.update_projection = nn.Sequential(
            nn.Conv1d(hidden + sampled_channels + 3, hidden * 2, 1),
            nn.GELU(),
            ConvResidual1D(hidden * 2, 1),
            nn.Conv1d(hidden * 2, hidden, 1),
        )
        self.gru = ConvGRU1D(hidden, hidden)
        self.delta_head = nn.Sequential(ConvResidual1D(hidden, 1), nn.Conv1d(hidden, 1, 1))
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    @staticmethod
    def _raw_cost(horizontal, horizontal_valid, typewell, typewell_valid, heel, heel_valid):
        h = horizontal[:, None, :].expand_as(typewell)
        hv = horizontal_valid[:, None, :].expand_as(typewell)
        return torch.stack(
            [h, hv, typewell, typewell_valid, h * typewell, -torch.abs(h - typewell),
             heel, heel_valid, h * heel, -torch.abs(h - heel)], dim=1
        )

    @staticmethod
    def _sample_local(cost, correction):
        batch, _, _, columns = cost.shape
        x = torch.linspace(-1.0, 1.0, columns, device=cost.device)
        output = []
        state_min = float(STATE_OFFSETS[0])
        state_span = float(STATE_OFFSETS[-1] - STATE_OFFSETS[0])
        for offset in LOCAL_OFFSETS_FT:
            y = 2.0 * (correction + offset - state_min) / state_span - 1.0
            grid = torch.stack([x[None].expand(batch, -1), y], dim=-1)[:, None]
            sampled = F.grid_sample(cost, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            output.append(sampled[:, :, 0])
        return torch.cat(output, dim=1)

    def forward(self, horizontal, horizontal_valid, typewell, typewell_valid,
                heel, heel_valid, geometry, eval_mask):
        context = self.context(torch.cat([horizontal[:, None], horizontal_valid[:, None], geometry], dim=1))
        hidden = context
        cost = self.cost_encoder(
            self._raw_cost(horizontal, horizontal_valid, typewell, typewell_valid, heel, heel_valid)
        )
        correction = torch.zeros_like(horizontal)
        predictions = []
        progress = torch.linspace(0.0, 1.0, horizontal.shape[1], device=horizontal.device)
        progress = progress[None].expand_as(horizontal)
        for _ in range(self.profile.iterations):
            local_cost = self._sample_local(cost, correction)
            values = torch.cat(
                [context, local_cost, correction[:, None] / 96.0, eval_mask[:, None], progress[:, None]], dim=1
            )
            hidden = self.gru(hidden, self.update_projection(values))
            delta = 8.0 * torch.tanh(self.delta_head(hidden).squeeze(1))
            correction = torch.clamp(correction + delta, -96.0, 96.0) * eval_mask
            predictions.append(correction)
        return predictions


def _forward(model, batch, device):
    return model(
        batch["horizontal"].to(device, non_blocking=True),
        batch["horizontal_valid"].to(device, non_blocking=True),
        batch["typewell"].to(device, non_blocking=True),
        batch["typewell_valid"].to(device, non_blocking=True),
        batch["heel"].to(device, non_blocking=True),
        batch["heel_valid"].to(device, non_blocking=True),
        batch["geometry"].to(device, non_blocking=True),
        batch["eval_mask"].to(device, non_blocking=True),
    )


def _trajectory_loss(predictions, batch, device):
    truth = batch["truth_residual"].to(device, non_blocking=True)
    eval_mask = batch["eval_mask"].to(device, non_blocking=True)
    column_md = batch["column_md"].to(device, non_blocking=True)
    column_z = batch["column_z"].to(device, non_blocking=True)
    panel_weight = batch["panel_weight"].to(device, non_blocking=True)
    denominator = eval_mask.sum(dim=1).clamp_min(1.0)
    panel_scale = (panel_weight / panel_weight.mean().clamp_min(1.0)).clamp(0.2, 5.0)
    total = truth.new_tensor(0.0)
    last = None
    for iteration, prediction in enumerate(predictions):
        squared = ((prediction - truth).square() * eval_mask).sum(dim=1) / denominator
        robust = (F.smooth_l1_loss(prediction, truth, beta=2.0, reduction="none") * eval_mask).sum(dim=1) / denominator
        predicted_s = prediction + column_z
        true_s = truth + column_z
        dmd = (column_md[:, 1:] - column_md[:, :-1]).clamp_min(1e-3)
        predicted_dip = (predicted_s[:, 1:] - predicted_s[:, :-1]) / dmd
        true_dip = (true_s[:, 1:] - true_s[:, :-1]) / dmd
        pair_mask = torch.minimum(eval_mask[:, 1:], eval_mask[:, :-1])
        slope = (F.smooth_l1_loss(predicted_dip, true_dip, beta=0.01, reduction="none") * pair_mask).sum(dim=1) / pair_mask.sum(dim=1).clamp_min(1.0)
        per_panel = squared / 64.0 + 0.20 * robust + 0.03 * slope
        weight = ITERATION_GAMMA ** (len(predictions) - iteration - 1)
        total = total + weight * (per_panel * panel_scale).mean()
        last = (squared.mean(), robust.mean(), slope.mean())
    normalizer = sum(ITERATION_GAMMA ** (len(predictions) - i - 1) for i in range(len(predictions)))
    return total / normalizer, {
        "mse": float(last[0].detach().cpu()),
        "robust": float(last[1].detach().cpu()),
        "slope": float(last[2].detach().cpu()),
    }


def _loader(dataset, profile, epoch, phase):
    generator = torch.Generator()
    phase_offset = 0 if phase == "real" else 10_000_000
    generator.manual_seed(SEED + phase_offset + epoch * 10_007)
    return DataLoader(
        dataset,
        batch_size=profile.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _train_epoch(model, loader, optimizer, scaler, device, label):
    model.train()
    totals = []
    started = time.time()
    for number, batch in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            predictions = _forward(model, batch, device)
        loss, components = _trajectory_loss([item.float() for item in predictions], batch, device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label}: non-finite loss at batch {number}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        totals.append({"loss": float(loss.detach().cpu()), **components})
        if number % 100 == 0 or number == len(loader):
            print(
                f"V208 {label} batch {number}/{len(loader)} "
                f"loss={np.mean([row['loss'] for row in totals[-100:]]):.4f} "
                f"elapsed_min={(time.time() - started) / 60.0:.1f}",
                flush=True,
            )
    return {key: float(np.mean([row[key] for row in totals])) for key in totals[0]}


@torch.no_grad()
def _evaluate(model, store, device, mode="full", return_rows=False):
    model.eval()
    dataset = PanelDataset(store, mode=mode)
    row_parts = []
    by_well = []
    for index in range(len(dataset)):
        item = dataset[index]
        batch = {key: value[None] for key, value in item.items() if key != "panel_weight"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            correction = _forward(model, batch, device)[-1][0].float().cpu().numpy()
        record = store.records[index]
        correction[:PREFIX_COLUMNS] = 0.0
        column_s = record["center_tvt"] + correction + record["column_z"]
        prediction = np.interp(record["eval_md"], record["column_md"], column_s) - record["eval_z"]
        center_s = record["center_tvt"] + record["column_z"]
        center = np.interp(record["eval_md"], record["column_md"], center_s) - record["eval_z"]
        truth = record["eval_truth"].astype(np.float64)
        distance = record["eval_md"].astype(np.float64) - float(record["column_md"][PREFIX_COLUMNS - 1])
        error = prediction - truth
        by_well.append({"well": record["well"], "rows": len(error), "rmse": float(np.sqrt(np.mean(error ** 2)))})
        row_parts.append(pd.DataFrame({
            "well": record["well"], "row_idx": record["eval_rows"], "truth": truth,
            "center": center, "prediction": prediction, "distance": distance,
        }))
    rows = pd.concat(row_parts, ignore_index=True)
    wells = pd.DataFrame(by_well)
    error = rows["prediction"].to_numpy() - rows["truth"].to_numpy()
    center_error = rows["center"].to_numpy() - rows["truth"].to_numpy()
    distance = rows["distance"].to_numpy()
    metrics = {
        "rows": int(len(rows)), "wells": int(len(wells)),
        "exact_sse": float(np.sum(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "center_rmse": float(np.sqrt(np.mean(center_error ** 2))),
        "gain_vs_center": float(np.sqrt(np.mean(center_error ** 2)) - np.sqrt(np.mean(error ** 2))),
        "median_well_rmse": float(wells["rmse"].median()),
        "p95_well_rmse": float(wells["rmse"].quantile(0.95)),
    }
    for name, mask in (
        ("near_0_500_rmse", (distance >= 0) & (distance < 500)),
        ("mid_500_1000_rmse", (distance >= 500) & (distance < 1000)),
        ("mid_1000_2000_rmse", (distance >= 1000) & (distance < 2000)),
        ("far_gt2000_rmse", distance >= 2000),
    ):
        metrics[name] = float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else None
    return metrics, wells, rows if return_rows else None


def _checkpoint_path(profile, phase):
    return MODEL_DIR / f"{profile.name}_{phase}_last.pt"


def _pretrain(profile, model, synthetic, device):
    last_path = _checkpoint_path(profile, "pretrain")
    optimizer = torch.optim.AdamW(model.parameters(), lr=profile.learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=profile.pretrain_epochs, eta_min=MIN_LEARNING_RATE
    )
    scaler = _amp_scaler(device.type == "cuda")
    start_epoch = 1
    history = []
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"V208 resume {profile.name} pretrain at epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, profile.pretrain_epochs + 1):
        synthetic.set_epoch(epoch)
        metrics = _train_epoch(
            model, _loader(synthetic, profile, epoch, "synthetic"), optimizer, scaler,
            device, f"{profile.name} pretrain epoch {epoch}",
        )
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in metrics.items()}, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        torch.save({
            "epoch": epoch, "model": _cpu_state(model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
        }, last_path)
        pd.DataFrame(history).to_csv(OUT_DIR / f"{profile.name}_pretrain_history.csv", index=False)
        print("V208 PRETRAIN EPOCH", {"profile": profile.name, **row}, flush=True)
    return model


def _finetune(profile, model, train_store, gate_store, device):
    last_path = _checkpoint_path(profile, "finetune")
    best_path = MODEL_DIR / f"{profile.name}_best.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=profile.learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=profile.real_epochs, eta_min=MIN_LEARNING_RATE
    )
    scaler = _amp_scaler(device.type == "cuda")
    start_epoch = 1
    history = []
    best_rmse = math.inf
    best_epoch = 0
    stale = 0
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_rmse = float(checkpoint.get("best_rmse", math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        stale = int(checkpoint.get("stale", 0))
        print(f"V208 resume {profile.name} finetune at epoch {start_epoch}", flush=True)
    dataset = PanelDataset(train_store, mode="full")
    for epoch in range(start_epoch, profile.real_epochs + 1):
        train_metrics = _train_epoch(
            model, _loader(dataset, profile, epoch, "real"), optimizer, scaler,
            device, f"{profile.name} real epoch {epoch}",
        )
        scheduler.step()
        valid_metrics, by_well, _ = _evaluate(model, gate_store, device, mode="full")
        row = {
            "epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        improved = float(valid_metrics["rmse"]) < best_rmse - 1e-4
        if improved:
            best_rmse = float(valid_metrics["rmse"])
            best_epoch = epoch
            stale = 0
            torch.save({"epoch": epoch, "model": _cpu_state(model), "metrics": valid_metrics, "profile": asdict(profile)}, best_path)
            by_well.to_csv(OUT_DIR / f"{profile.name}_best_bywell.csv", index=False)
        else:
            stale += 1
        torch.save({
            "epoch": epoch, "model": _cpu_state(model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
            "best_rmse": best_rmse, "best_epoch": best_epoch, "stale": stale,
        }, last_path)
        pd.DataFrame(history).to_csv(OUT_DIR / f"{profile.name}_finetune_history.csv", index=False)
        print("V208 REAL EPOCH", {"profile": profile.name, **row, "best_rmse": best_rmse, "stale": stale}, flush=True)
        if epoch >= profile.min_real_epochs and stale >= profile.patience:
            print(f"V208 {profile.name} early stop epoch={epoch}", flush=True)
            break
    if not best_path.is_file():
        raise RuntimeError(f"{profile.name}: no best checkpoint")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model, int(checkpoint["epoch"])


def _run_profile(profile, train_store, gate_store, bank, device):
    final_path = OUT_DIR / f"{profile.name}_report.json"
    if final_path.is_file():
        report = json.loads(final_path.read_text(encoding="utf-8"))
        print(f"V208 skip completed profile {profile.name}: rmse={report['metrics']['rmse']:.6f}", flush=True)
        return report
    _seed_everything(SEED)
    geometry_channels = int(train_store.records[0]["geometry"].shape[0])
    model = ScaledAxialRAFT(geometry_channels, profile).to(device)
    parameters = int(sum(value.numel() for value in model.parameters()))
    synthetic = SyntheticPanelDataset(
        train_store, bank, profile.synthetic_multiplier, PRETRAIN_SEED + parameters % 100_000
    )
    audit = pd.DataFrame([synthetic.audit(index) for index in range(min(256, len(synthetic)))])
    if float(audit["horizontal_coverage"].mean()) < 0.85:
        raise RuntimeError(f"{profile.name}: synthetic coverage guard failed")
    print("V208 PROFILE", {**asdict(profile), "parameters": parameters, "synthetic_audit": audit.mean(numeric_only=True).to_dict()}, flush=True)
    started = time.time()
    model = _pretrain(profile, model, synthetic, device)
    model, best_epoch = _finetune(profile, model, train_store, gate_store, device)
    metrics, by_well, rows = _evaluate(model, gate_store, device, mode="full", return_rows=True)
    interventions = []
    for mode in ("full", "no_gr", "shuffled_typewell"):
        mode_metrics, _, _ = _evaluate(model, gate_store, device, mode=mode)
        interventions.append({"mode": mode, **mode_metrics})
    intervention_frame = pd.DataFrame(interventions)
    full_rmse = float(intervention_frame.loc[intervention_frame["mode"].eq("full"), "rmse"].iloc[0])
    intervention_frame["degradation_vs_full"] = intervention_frame["rmse"] - full_rmse
    intervention_frame.to_csv(OUT_DIR / f"{profile.name}_interventions.csv", index=False)
    by_well.to_csv(OUT_DIR / f"{profile.name}_final_bywell.csv", index=False)
    rows.to_parquet(OUT_DIR / f"{profile.name}_fold2_fullrow.parquet", index=False)
    report = {
        "profile": asdict(profile), "parameters": parameters, "best_epoch": best_epoch,
        "metrics": metrics,
        "no_gr_degradation": float(intervention_frame.loc[intervention_frame["mode"].eq("no_gr"), "degradation_vs_full"].iloc[0]),
        "shuffled_typewell_degradation": float(intervention_frame.loc[intervention_frame["mode"].eq("shuffled_typewell"), "degradation_vs_full"].iloc[0]),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    _write_json(final_path, report)
    del model, synthetic
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main():
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("V208 requires a CUDA GPU")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    _seed_everything(SEED)
    train_store, gate_store, fold_coverage = _load_stores()
    bank = ResidualBank(train_store)
    config = {
        "experiment": EXPERIMENT,
        "train_folds": TRAIN_FOLDS,
        "gate_fold": GATE_FOLD,
        "untouched_folds": UNTOUCHED_FOLDS,
        "train_wells": len(set(record["well"] for record in train_store.records)),
        "train_panels": len(train_store.records),
        "gate_wells": len(gate_store.records),
        "fold_coverage": fold_coverage,
        "state_axis": [float(STATE_OFFSETS[0]), float(STATE_OFFSETS[-1]), 2.0],
        "historical_endpoint_inputs": [],
        "state_axis_center": "legal likelihood-PF trajectory frozen in V4 panels",
        "profiles": [asdict(profile) for profile in PROFILES],
    }
    _write_json(OUT_DIR / "config.json", config)
    print("V208 CONFIG", json.dumps(_clean(config), indent=2), flush=True)

    reports = []
    small = _run_profile(PROFILES[0], train_store, gate_store, bank, device)
    reports.append(small)
    medium = _run_profile(PROFILES[1], train_store, gate_store, bank, device)
    reports.append(medium)
    medium_gain = float(small["metrics"]["rmse"] - medium["metrics"]["rmse"])
    medium_causal = bool(
        medium["no_gr_degradation"] >= CAUSAL_GATE
        and medium["shuffled_typewell_degradation"] >= CAUSAL_GATE
    )
    large_launched = bool(medium_gain >= MEDIUM_SCALE_GATE and medium_causal)
    if large_launched:
        reports.append(_run_profile(PROFILES[2], train_store, gate_store, bank, device))
    else:
        print(
            "V208 LARGE SKIPPED",
            {"medium_gain_vs_small": medium_gain, "medium_causal": medium_causal},
            flush=True,
        )

    summary = pd.DataFrame([
        {
            "profile": report["profile"]["name"], "parameters": report["parameters"],
            "best_epoch": report["best_epoch"], **report["metrics"],
            "no_gr_degradation": report["no_gr_degradation"],
            "shuffled_typewell_degradation": report["shuffled_typewell_degradation"],
        }
        for report in reports
    ]).sort_values("rmse")
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    best = min(reports, key=lambda item: item["metrics"]["rmse"])
    scale_monotonic = all(
        reports[index]["metrics"]["rmse"] < reports[index - 1]["metrics"]["rmse"]
        for index in range(1, len(reports))
    )
    strict_pass = bool(
        best["metrics"]["rmse"] <= 8.5
        and best["no_gr_degradation"] >= CAUSAL_GATE
        and best["shuffled_typewell_degradation"] >= CAUSAL_GATE
        and scale_monotonic
    )
    verdict = {
        "experiment": EXPERIMENT,
        "stage": "Fold45_scaled_synthetic_CorrNet_training_Fold2_completewell_gateway",
        "protocol": {
            "train_folds": TRAIN_FOLDS, "gate_fold": GATE_FOLD,
            "fold3_touched": 0, "fold1_touched": 0,
        },
        "reports": reports,
        "medium_gain_vs_small": medium_gain,
        "large_launched": large_launched,
        "scale_monotonic": scale_monotonic,
        "best_profile": best["profile"]["name"],
        "best_rmse": best["metrics"]["rmse"],
        "target_6p873_pass": int(best["metrics"]["rmse"] <= 6.873),
        "checks": {
            "best_rmse_le_8p5": int(best["metrics"]["rmse"] <= 8.5),
            "no_gr_degradation_ge_0p30": int(best["no_gr_degradation"] >= CAUSAL_GATE),
            "shuffled_typewell_degradation_ge_0p30": int(best["shuffled_typewell_degradation"] >= CAUSAL_GATE),
            "scale_monotonic": int(scale_monotonic),
        },
        "strict_pass": int(strict_pass),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "next": (
            "PASS -> retrain selected scale on Fold245 and confirm once on untouched Fold3"
            if strict_pass
            else "FAIL -> scaling does not rescue the GR correlation decoder; do not spend Fold3"
        ),
    }
    _write_json(OUT_DIR / "verdict.json", verdict)
    print("\nV208 SUMMARY")
    print(summary.to_string(index=False), flush=True)
    print("\nV208 FINAL VERDICT")
    print(json.dumps(_clean(verdict), indent=2), flush=True)
    print("saved:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
