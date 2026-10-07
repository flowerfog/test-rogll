# Paste this entire cell into Kaggle.
import sys
import shutil
import torch
from pathlib import Path


def find_data_root():
    for root in [Path('/kaggle/working'), Path('/kaggle/input')]:
        if not root.is_dir():
            continue
        for sample in root.rglob('sample_submission.csv'):
            candidate = sample.parent
            if (candidate / 'train').is_dir() and (candidate / 'test').is_dir():
                return candidate
    raise FileNotFoundError('Cannot locate competition data root')


MODEL_DIR = Path('/kaggle/input/datasets/fogflower/refiner')
MODEL_FILE = MODEL_DIR / 'emission_model.json'
if not MODEL_FILE.is_file():
    raise FileNotFoundError(MODEL_FILE)
for seed in (1103, 2207):
    checkpoint = MODEL_DIR / f'refiner_seed{seed}.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

DATA_ROOT = find_data_root()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT = Path('/kaggle/working/seventh_place_hmm_unet_inference_only')
OUTPUT.mkdir(parents=True, exist_ok=True)
shutil.copy2(MODEL_FILE, OUTPUT / 'emission_model.json')
for seed in (1103, 2207):
    shutil.copy2(MODEL_DIR / f'refiner_seed{seed}.pt', OUTPUT / f'refiner_seed{seed}.pt')
SOURCE = '''"""Clean-room reconstruction of the public 7th-place HMM + UNet recipe.

This implementation follows the published stages literally:
HMM core -> learned emissions -> synthetic-mistake refiner pretraining ->
short real-well finetuning -> two-refiner disagreement gate -> hedges.

The author's private checkpoints and undocumented feature code were not
available, so this is a reproducible clean-room implementation, not a claim
of byte-identical source recovery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 20260806
GRID_POINTS = 512
STATE_STEP_FT = 2.0
STATE_RADIUS_FT = 96.0
HMM_TEMPERATURE = 18.0
HMM_TRANSITION_SIGMA_FT = 8.0
SYNTHETIC_EPOCHS = 60
REAL_EPOCHS = 8
SYNTHETIC_GOOD_FRACTION = 0.15
SYNTHETIC_LOG_MEDIAN = math.log(2.9)
SYNTHETIC_LOG_SIGMA = 0.68
REFINER_SEEDS = (1103, 2207)
GATE_OFFSET = 1.0
HEDGE_HMM = 0.10
HEDGE_NEIGHBOUR = 0.15
HEDGE_FUSE = 0.15
HEDGE_NEIGHBOUR_REPEAT = 0.05
HEDGE_CLIP_FT = 24.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _num(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float64)


def _interp(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.full(len(query), np.nan, np.float64)
    order = np.argsort(x[valid], kind="stable")
    xx, yy = x[valid][order], y[valid][order]
    xx, unique = np.unique(xx, return_index=True)
    yy = yy[unique]
    return np.interp(query, xx, yy, left=yy[0], right=yy[-1])


@dataclass
class Well:
    well: str
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt_input: np.ndarray
    truth: np.ndarray | None
    type_tvt: np.ndarray
    type_gr: np.ndarray

    @property
    def known(self) -> np.ndarray:
        return np.isfinite(self.tvt_input)

    @property
    def anchor(self) -> int:
        positions = np.flatnonzero(self.known)
        return int(positions[-1]) if len(positions) else 0


def load_well(data_root: Path, well: str, split: str) -> Well:
    folder = data_root / split
    horizontal = pd.read_csv(folder / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(folder / f"{well}__typewell.csv")
    truth = _num(horizontal, "TVT") if "TVT" in horizontal else None
    return Well(
        well=well,
        md=_num(horizontal, "MD"),
        x=_num(horizontal, "X"),
        y=_num(horizontal, "Y"),
        z=_num(horizontal, "Z"),
        gr=_num(horizontal, "GR"),
        tvt_input=_num(horizontal, "TVT_input"),
        truth=truth,
        type_tvt=_num(typewell, "TVT"),
        type_gr=_num(typewell, "GR"),
    )


def _grid(well: Well) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(well.md, kind="stable")
    md = well.md[order]
    grid = np.linspace(md[0], md[-1], GRID_POINTS)
    z = _interp(md, well.z[order], grid)
    gr = _interp(md, well.gr[order], grid)
    tvt_input = _interp(md, well.tvt_input[order], grid)
    truth = _interp(md, well.truth[order], grid) if well.truth is not None else None
    return grid, z, gr, tvt_input, truth


def _base_path(well: Well, grid: np.ndarray, visible: np.ndarray) -> np.ndarray:
    known = np.flatnonzero(np.isfinite(visible))
    if len(known) < 2:
        return np.full(len(grid), float(np.nanmedian(visible)), np.float64)
    tail = known[-min(64, len(known)):]
    slope = np.polyfit(grid[tail] - grid[tail[-1]], visible[tail], 1)[0]
    return visible[tail[-1]] + slope * (grid - grid[tail[-1]])


def _emission_features(
    observed: np.ndarray,
    type_gr: np.ndarray,
    candidate: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Build candidate-state features shared by emission fitting and decode."""
    observed = np.nan_to_num(observed, nan=float(np.nanmedian(observed)))
    centre = float(np.nanmedian(observed))
    scale = max(1.0, float(1.4826 * np.nanmedian(np.abs(observed - centre))))
    value_delta = (type_gr - observed[None, :]) / scale
    observed_slope = np.gradient(observed, grid)
    type_slope = np.gradient(type_gr, grid, axis=1)
    slope_scale = max(1e-3, float(np.nanmedian(np.abs(observed_slope - np.nanmedian(observed_slope)))))
    slope_delta = (type_slope - observed_slope[None, :]) / slope_scale
    offset = candidate - candidate[len(candidate) // 2 : len(candidate) // 2 + 1]
    return np.stack(
        [
            np.ones_like(value_delta),
            np.abs(value_delta),
            value_delta * value_delta,
            np.abs(slope_delta),
            slope_delta * slope_delta,
            np.abs(offset) / max(STATE_RADIUS_FT, 1.0),
        ],
        axis=-1,
    )


def fit_learned_emissions(wells: list[Well], out_dir: Path) -> dict[str, object]:
    """Fit a frozen emission scorer from complete training wells.

    Each grid column contributes one positive state (nearest to the known
    training TVT) and a compact set of hard negatives.  Weighted ridge on
    these state labels is deterministic, memory bounded, and deployable on
    unseen wells because it uses only GR, typewell GR, and the HMM candidate.
    """
    feature_count = 6
    xtx = np.zeros((feature_count, feature_count), np.float64)
    xty = np.zeros(feature_count, np.float64)
    rng = np.random.default_rng(SEED + 71)
    used = 0
    for well in wells:
        grid, _, gr, visible, truth = _grid(well)
        if truth is None or np.isfinite(truth).sum() < 16:
            continue
        base = _base_path(well, grid, visible)
        offsets = np.arange(-STATE_RADIUS_FT, STATE_RADIUS_FT + STATE_STEP_FT, STATE_STEP_FT)
        candidate = base[None, :] + offsets[:, None]
        type_gr = _interp(well.type_tvt, well.type_gr, candidate)
        features = _emission_features(gr, type_gr, candidate, grid)
        target_offset = truth - base
        truth_state = np.argmin(np.abs(offsets[:, None] - target_offset[None, :]), axis=0)
        for column in rng.choice(len(grid), size=min(96, len(grid)), replace=False):
            positive = int(truth_state[column])
            negatives = {0, len(offsets) - 1, positive}
            negatives.update(
                int(index)
                for index in rng.choice(len(offsets), size=min(8, len(offsets)), replace=False)
            )
            for state in negatives:
                row = features[state, column]
                weight = 8.0 if state == positive else 1.0
                xtx += weight * np.outer(row, row)
                xty += weight * row * (1.0 if state == positive else 0.0)
                used += 1
    if used == 0:
        raise RuntimeError("No complete training wells available for learned emissions")
    ridge = 1e-3 * np.eye(feature_count, dtype=np.float64)
    coefficients = np.linalg.solve(xtx / used + ridge, xty / used)
    # The ridge target is high for the true state and low for hard negatives.
    # Its fitted mismatch coefficients are therefore already negative; keep
    # that direction when using the score as an HMM emission.
    coefficients *= 12.0
    model = {
        "feature_names": ["bias", "abs_value_delta", "value_delta_sq", "abs_slope_delta", "slope_delta_sq", "abs_offset"],
        "coefficients": coefficients.tolist(),
        "training_wells": int(len(wells)),
        "training_rows": int(used),
    }
    (out_dir / "emission_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def hmm_decode(well: Well, emission_model: dict[str, object] | None = None) -> np.ndarray:
    grid, z, gr, visible, _ = _grid(well)
    base = _base_path(well, grid, visible)
    offsets = np.arange(-STATE_RADIUS_FT, STATE_RADIUS_FT + STATE_STEP_FT, STATE_STEP_FT)
    candidate = base[None, :] + offsets[:, None]
    type_gr = _interp(well.type_tvt, well.type_gr, candidate)
    if emission_model is None:
        observed = np.nan_to_num(gr, nan=np.nanmedian(gr))
        local_scale = np.maximum(1.0, gaussian_filter1d(np.abs(observed - np.median(observed)), 8))
        emission = -0.5 * ((type_gr - observed[None, :]) / (HMM_TEMPERATURE + local_scale[None, :])) ** 2
    else:
        features = _emission_features(gr, type_gr, candidate, grid)
        coefficients = np.asarray(emission_model["coefficients"], np.float64)
        emission = np.einsum("cij,j->ci", features, coefficients)
        emission -= np.nanmax(emission, axis=0, keepdims=True)
    transition = -0.5 * ((offsets[:, None] - offsets[None, :]) / HMM_TRANSITION_SIGMA_FT) ** 2
    score = np.full((len(offsets), len(grid)), -np.inf, np.float64)
    back = np.zeros_like(score, np.int16)
    score[:, 0] = emission[:, 0]
    for column in range(1, len(grid)):
        values = score[:, column - 1][:, None] + transition
        back[:, column] = np.argmax(values, axis=0).astype(np.int16)
        score[:, column] = emission[:, column] + values[back[:, column], np.arange(len(offsets))]
    state = int(np.argmax(score[:, -1]))
    path = np.empty(len(grid), np.float64)
    for column in range(len(grid) - 1, -1, -1):
        path[column] = candidate[state, column]
        state = int(back[state, column]) if column else state
    known = np.isfinite(visible)
    path[known] = visible[known]
    return path


class RefinerUNet(nn.Module):
    def __init__(self, channels: int = 6):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(channels, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 32, 5, padding=2), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(), nn.Conv1d(64, 64, 5, padding=2), nn.GELU())
        self.mid = nn.Sequential(nn.Conv1d(64, 96, 5, padding=2), nn.GELU(), nn.Conv1d(96, 96, 5, padding=2), nn.GELU())
        # The decoder concatenates the 96-channel bottleneck, the 64-channel
        # second encoder skip, and the 32-channel first encoder skip before
        # its first convolution.
        self.dec = nn.Sequential(nn.Conv1d(96 + 64 + 32, 64, 5, padding=2), nn.GELU(), nn.Conv1d(64, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 1, 1))
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        mid = self.mid(e2)
        up = nn.functional.interpolate(mid, size=e2.shape[-1], mode="linear", align_corners=False)
        up = torch.cat([up, e2], dim=1)
        up = nn.functional.interpolate(up, size=e1.shape[-1], mode="linear", align_corners=False)
        up = torch.cat([up, e1], dim=1)
        return self.dec(up).squeeze(1)


def _normalise(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float32)
    centre = np.nanmedian(value)
    scale = 1.4826 * np.nanmedian(np.abs(value - centre))
    return np.nan_to_num((value - centre) / max(float(scale), 1e-3), nan=0.0, posinf=0.0, neginf=0.0)


def _synthetic_condition(truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    amplitude = float(rng.lognormal(SYNTHETIC_LOG_MEDIAN, SYNTHETIC_LOG_SIGMA))
    if rng.random() < SYNTHETIC_GOOD_FRACTION:
        amplitude = float(rng.uniform(0.3, 1.0))
    walk = rng.normal(0.0, 1.0, len(truth))
    walk = gaussian_filter1d(walk, max(3, len(truth) // 48))
    walk -= walk[0]
    walk /= max(float(np.std(walk)), 1e-6)
    return truth + amplitude * walk


def _features(well: Well, condition: np.ndarray, first_pass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid, z, gr, visible, truth = _grid(well)
    scale = max(float(grid[-1] - grid[0]), 1.0)
    channels = np.vstack([
        _normalise(gr),
        _normalise(z),
        (grid - grid[0]) / scale,
        np.isfinite(visible).astype(np.float32),
        _normalise(condition),
        _normalise(first_pass),
    ]).astype(np.float32)
    return channels, truth


def train_refiners(
    wells: list[Well],
    out_dir: Path,
    device: torch.device,
    emission_model: dict[str, object],
) -> list[RefinerUNet]:
    arrays, targets = [], []
    for well in wells:
        first = hmm_decode(well, emission_model)
        _, _, _, _, truth = _grid(well)
        if truth is None or not np.isfinite(truth).all():
            continue
        rng = np.random.default_rng(abs(hash(well.well)) % (2**32))
        for _ in range(4):
            condition = _synthetic_condition(truth, rng)
            features, _ = _features(well, condition, first)
            arrays.append(features)
            targets.append((truth - condition).astype(np.float32))
    x = torch.from_numpy(np.stack(arrays)).to(device)
    y = torch.from_numpy(np.stack(targets)).to(device)
    refiners = []
    for seed in REFINER_SEEDS:
        torch.manual_seed(seed)
        model = RefinerUNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
        model.train()
        for epoch in range(SYNTHETIC_EPOCHS):
            for values, target in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.smooth_l1_loss(model(values), target)
                loss.backward()
                optimizer.step()
        real_x, real_y = [], []
        for well in wells:
            first = hmm_decode(well, emission_model)
            grid, _, _, visible, truth = _grid(well)
            if truth is None:
                continue
            condition = np.where(np.isfinite(visible), visible, first)
            values, _ = _features(well, condition, first)
            real_x.append(values)
            real_y.append((truth - condition).astype(np.float32))
        real_loader = DataLoader(TensorDataset(torch.from_numpy(np.stack(real_x)).to(device), torch.from_numpy(np.stack(real_y)).to(device)), batch_size=16, shuffle=True)
        model.train()
        for _ in range(REAL_EPOCHS):
            for values, target in real_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.smooth_l1_loss(model(values), target)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), out_dir / f"refiner_seed{seed}.pt")
        refiners.append(model.eval())
    return refiners


def load_refiners(out_dir: Path, device: torch.device) -> list[RefinerUNet]:
    refiners = []
    for seed in REFINER_SEEDS:
        path = out_dir / f"refiner_seed{seed}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        model = RefinerUNet().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        refiners.append(model.eval())
    return refiners


def _nearest_neighbour(train_wells: list[Well], test: Well) -> np.ndarray:
    train_xy = np.array([[np.nanmedian(w.x), np.nanmedian(w.y)] for w in train_wells], float)
    tree = cKDTree(train_xy)
    _, index = tree.query([[np.nanmedian(test.x), np.nanmedian(test.y)]], k=1)
    donor = train_wells[int(np.asarray(index).reshape(-1)[0])]
    grid, _, _, visible, _ = _grid(test)
    donor_grid, _, _, _, donor_truth = _grid(donor)
    if donor_truth is None:
        return np.full(len(grid), visible[np.flatnonzero(np.isfinite(visible))[-1]], float)
    anchor = np.flatnonzero(np.isfinite(visible))[-1]
    donor_anchor = np.flatnonzero(np.isfinite(donor.tvt_input))[-1] if donor.known.any() else 0
    donor_relative = donor_truth - donor_truth[min(donor_anchor, len(donor_truth) - 1)]
    return visible[anchor] + np.interp(grid, donor_grid, donor_relative)


def predict_one(
    well: Well,
    train_wells: list[Well],
    refiners: list[RefinerUNet],
    emission_model: dict[str, object],
    device: torch.device,
) -> np.ndarray:
    first = hmm_decode(well, emission_model)
    grid, _, _, visible, _ = _grid(well)
    condition = np.where(np.isfinite(visible), visible, first)
    values, _ = _features(well, condition, first)
    with torch.no_grad():
        tensor = torch.from_numpy(values)[None].to(device)
        corrections = np.vstack([model(tensor).cpu().numpy()[0] for model in refiners])
    refined = condition + corrections.mean(axis=0)
    disagreement = corrections.std(axis=0)
    gate = disagreement / (disagreement + GATE_OFFSET)
    neighbour = _nearest_neighbour(train_wells, well)
    fuse = first
    final = refined + HEDGE_HMM * gate * (fuse - refined)
    final += HEDGE_NEIGHBOUR * gate * np.clip(neighbour - refined, -HEDGE_CLIP_FT, HEDGE_CLIP_FT)
    final += HEDGE_FUSE * gate * np.clip(fuse - refined, -HEDGE_CLIP_FT, HEDGE_CLIP_FT)
    final += HEDGE_NEIGHBOUR_REPEAT * gate * np.clip(neighbour - refined, -HEDGE_CLIP_FT, HEDGE_CLIP_FT)
    final[np.isfinite(visible)] = visible[np.isfinite(visible)]
    return np.interp(well.md, grid, final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("seventh_place_hmm_unet_cleanroom_v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()
    seed_everything(SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    train_ids = sorted(path.stem.replace("__horizontal_well", "") for path in (args.data_root / "train").glob("*__horizontal_well.csv"))
    test_ids = sorted(path.stem.replace("__horizontal_well", "") for path in (args.data_root / "test").glob("*__horizontal_well.csv"))
    train_wells = [load_well(args.data_root, well, "train") for well in train_ids]
    test_wells = [load_well(args.data_root, well, "test") for well in test_ids]
    sample_path = args.data_root / "sample_submission.csv"
    sample = pd.read_csv(sample_path) if sample_path.is_file() else None
    device = torch.device(args.device)
    print({"train_wells": len(train_wells), "test_wells": len(test_wells), "device": str(device)}, flush=True)
    emission_path = args.output / "emission_model.json"
    if args.inference_only:
        if not emission_path.is_file():
            raise FileNotFoundError(emission_path)
        emission_model = json.loads(emission_path.read_text(encoding="utf-8"))
        refiners = load_refiners(args.output, device)
        print({"stage": "inference_only", "model_dir": str(args.output)}, flush=True)
    else:
        emission_model = fit_learned_emissions(train_wells, args.output)
        print(
            {"stage": "learned_emissions", "training_rows": emission_model["training_rows"]},
            flush=True,
        )
        refiners = train_refiners(train_wells, args.output, device, emission_model)
    rows = []
    for number, well in enumerate(test_wells, 1):
        prediction = predict_one(well, train_wells, refiners, emission_model, device)
        if sample is None:
            row_idx = np.arange(len(prediction), dtype=int)
            ids = [f"{well.well}_{index}" for index in row_idx]
        else:
            local_sample = sample[sample["id"].astype(str).str.startswith(f"{well.well}_")].copy()
            row_idx = local_sample["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
            if len(row_idx) == 0 or int(row_idx.max()) >= len(prediction):
                raise RuntimeError(f"Invalid sample row alignment for {well.well}")
            ids = local_sample["id"].astype(str).tolist()
        rows.append(pd.DataFrame({"id": ids, "tvt": prediction[row_idx]}))
        print(f"inference {number}/{len(test_wells)} {well.well}", flush=True)
    submission = pd.concat(rows, ignore_index=True)
    if sample_path.is_file():
        submission = sample[["id"]].merge(submission, on="id", how="left")
    if submission["tvt"].isna().any() or submission["id"].duplicated().any():
        raise RuntimeError("Invalid submission alignment")
    submission.to_csv(args.output / "submission.csv", index=False)
    (args.output / "recipe.json").write_text(json.dumps({"experiment": "seventh_place_hmm_unet_cleanroom_v2", "hmm": {"state_radius_ft": STATE_RADIUS_FT, "state_step_ft": STATE_STEP_FT, "emission": "trained_weighted_ridge"}, "pretrain_epochs": SYNTHETIC_EPOCHS, "finetune_epochs": REAL_EPOCHS, "refiner_seeds": REFINER_SEEDS, "gate": "std/(std+1)", "hedges": {"hmm": HEDGE_HMM, "neighbor": HEDGE_NEIGHBOUR, "fuse": HEDGE_FUSE, "neighbor_repeat": HEDGE_NEIGHBOUR_REPEAT}}, indent=2), encoding="utf-8")
    print(f"saved {args.output / 'submission.csv'} rows={len(submission)}", flush=True)


if __name__ == "__main__":
    main()
'''
namespace = {'__name__': '__main__', '__file__': '/kaggle/working/embedded_hmm_unet.py'}
sys.argv = ['/kaggle/working/embedded_hmm_unet.py', '--data-root', str(DATA_ROOT), '--output', str(OUTPUT), '--device', DEVICE, '--inference-only']
print({'data_root': str(DATA_ROOT), 'model_dir': str(MODEL_DIR), 'device': DEVICE, 'stage': 'inference_only'}, flush=True)
exec(compile(SOURCE, '/kaggle/working/embedded_hmm_unet.py', 'exec'), namespace)
submission = OUTPUT / 'submission.csv'
if not submission.is_file():
    raise FileNotFoundError(submission)
shutil.copy2(submission, '/kaggle/working/submission.csv')
print({'submission': '/kaggle/working/submission.csv', 'rows': sum(1 for _ in open('/kaggle/working/submission.csv', encoding='utf-8')) - 1}, flush=True)
