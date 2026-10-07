"""S5: fold-clean complete-path listwise scorer with posterior-mean decoding.

Run after the v4/S3/S4 cells in the same AutoDL kernel.  This experiment is a
standalone, no-neighbour model.  Candidate trajectories are physical S(MD)
templates learned from other training folds and anchored to the target well's
visible heel.  No PF/beam path, strong, strong_bias, CP12, Gold, Neighbor, or
validation labels enter a candidate or model input.

Every outer-train panel uses a template bank built without its well fold.  Fold
1 uses a bank built from folds 2-5.  A shared sequence encoder scores the whole
candidate-aligned GR path, compares candidates with a set transformer, and
outputs a posterior-weighted trajectory rather than a hard route.
"""

from pathlib import Path
import copy
import gc
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


S5_REQUIRED_GLOBALS = (
    "V4",
    "fold",
    "train_store_raft",
    "valid_store_raft",
    "PREFIX_COLUMNS_RAFT",
    "device_raft",
    "_seed_raft",
)
s5_missing = [name for name in S5_REQUIRED_GLOBALS if name not in globals()]
if s5_missing:
    raise RuntimeError(
        "Run the v4/S3 cells first; missing S5 prerequisites: " + repr(s5_missing)
    )


S5_DATA_ROOT = Path(V4["DATA_ROOT"])
S5_OOF_ROOT = Path(V4["OOF_ROOT"])
S5_OUT_DIR = S5_OOF_ROOT / "fullwell_listwise_path_scorer_s5_onefold_v0"
S5_MODEL_DIR = S5_OUT_DIR / "models"
for s5_directory in (S5_OUT_DIR, S5_MODEL_DIR):
    s5_directory.mkdir(parents=True, exist_ok=True)

S5_VALID_FOLD = 1
S5_SEED = 20260724
S5_RAW_CLUSTERS = 48
S5_LENGTH_CLUSTERS = 64
S5_LENGTH_ALPHA = 0.5
S5_LINE_QUANTILES = np.linspace(0.02, 0.98, 17)
S5_MODEL_CANDIDATES = 64
S5_SEQUENCE_LENGTH = 256
S5_BATCH_SIZE = 2
S5_EPOCHS = 14
S5_PATIENCE = 5
S5_LEARNING_RATE = 3e-4
S5_MIN_LEARNING_RATE = 3e-5
S5_WEIGHT_DECAY = 2e-4
S5_TARGET_TEMPERATURE_FT = 2.5
S5_POSTERIOR_PATH_WEIGHT = 0.50
S5_EXPECTED_MSE_WEIGHT = 0.05
S5_TRAIN_ORACLE_GATE_FT = 8.0
S5_ARCHITECTURE_GATE_FT = 7.0
S5_TARGET_GATE_FT = 5.5


def _s5_rmse(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _s5_robust_normalise(values):
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    output = np.zeros(values.shape, np.float32)
    if int(valid.sum()) < 4:
        return output
    centre = float(np.median(values[valid]))
    scale = float(1.4826 * np.median(np.abs(values[valid] - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values[valid]), 1.0))
    output[valid] = np.clip((values[valid] - centre) / scale, -8.0, 8.0)
    return output


def _s5_rank(values, descending=False):
    values = np.asarray(values, np.float64)
    safe = np.nan_to_num(
        values,
        nan=(-1e9 if descending else 1e9),
        posinf=(1e9 if not descending else -1e9),
        neginf=(-1e9 if descending else 1e9),
    )
    order = np.argsort(-safe if descending else safe, kind="mergesort")
    ranks = np.empty(len(order), np.float32)
    ranks[order] = np.arange(len(order), dtype=np.float32)
    return ranks


def _s5_profile(record):
    prefix = int(PREFIX_COLUMNS_RAFT)
    truth_tvt = (
        np.asarray(record["center_tvt"], np.float64)
        + np.asarray(record["truth_residual"], np.float64)
    )
    z = np.asarray(record["column_z"], np.float64)
    md = np.asarray(record["column_md"], np.float64)
    anchor_s = float(truth_tvt[prefix - 1] + z[prefix - 1])
    delta_s = truth_tvt[prefix:] + z[prefix:] - anchor_s
    span = float(md[-1] - md[prefix - 1])
    return delta_s.astype(np.float32), max(span, 1.0)


class S5TemplateBank:
    def __init__(self, source_records, label, seed):
        profiles = []
        spans = []
        for record in source_records:
            profile, span = _s5_profile(record)
            profiles.append(profile)
            spans.append(span)
        profiles = np.stack(profiles).astype(np.float32)
        spans = np.asarray(spans, np.float64)
        self.label = str(label)
        self.median_span = float(np.median(spans))
        ratio = np.maximum(spans / max(self.median_span, 1.0), 0.1)
        length_normalised = profiles / ratio[:, None] ** S5_LENGTH_ALPHA

        raw_model = MiniBatchKMeans(
            n_clusters=min(S5_RAW_CLUSTERS, max(8, len(profiles) // 12)),
            random_state=int(seed),
            n_init=5,
            batch_size=256,
            max_iter=300,
        ).fit(profiles)
        length_model = MiniBatchKMeans(
            n_clusters=min(S5_LENGTH_CLUSTERS, max(8, len(profiles) // 10)),
            random_state=int(seed) + 101,
            n_init=5,
            batch_size=256,
            max_iter=300,
        ).fit(length_normalised)

        templates = []
        alphas = []
        priors = []
        names = []
        raw_counts = np.bincount(
            raw_model.labels_, minlength=len(raw_model.cluster_centers_)
        )
        for index, centre in enumerate(raw_model.cluster_centers_):
            templates.append(centre.astype(np.float32))
            alphas.append(0.0)
            priors.append(float(raw_counts[index] / max(raw_counts.sum(), 1)))
            names.append(f"raw_{index:03d}")
        length_counts = np.bincount(
            length_model.labels_, minlength=len(length_model.cluster_centers_)
        )
        for index, centre in enumerate(length_model.cluster_centers_):
            templates.append(centre.astype(np.float32))
            alphas.append(S5_LENGTH_ALPHA)
            priors.append(float(length_counts[index] / max(length_counts.sum(), 1)))
            names.append(f"length_{index:03d}")

        progress = np.arange(1, S5_SEQUENCE_LENGTH + 1, dtype=np.float32)
        progress /= float(S5_SEQUENCE_LENGTH)
        for index, endpoint in enumerate(
            np.quantile(profiles[:, -1], S5_LINE_QUANTILES)
        ):
            templates.append((float(endpoint) * progress).astype(np.float32))
            alphas.append(0.0)
            priors.append(1.0 / len(profiles))
            names.append(f"line_{index:02d}")

        self.templates = np.stack(templates).astype(np.float32)
        self.alphas = np.asarray(alphas, np.float32)
        self.priors = np.asarray(priors, np.float32)
        self.names = np.asarray(names, str)
        self.source_panels = int(len(source_records))

    def trajectories(self, record):
        prefix = int(PREFIX_COLUMNS_RAFT)
        md = np.asarray(record["column_md"], np.float32)
        z = np.asarray(record["column_z"], np.float32)
        anchor_tvt = float(record["center_tvt"][prefix - 1])
        anchor_s = anchor_tvt + float(z[prefix - 1])
        span = float(md[-1] - md[prefix - 1])
        ratio = max(span / max(self.median_span, 1.0), 0.1)
        delta_s = self.templates * ratio ** self.alphas[:, None]
        candidate_s = anchor_s + delta_s
        candidate_tvt = candidate_s - z[prefix:][None, :]
        return candidate_tvt.astype(np.float32), delta_s.astype(np.float32)


class S5ReferenceStore:
    def __init__(self, records):
        self.rows = {}
        wells = sorted({str(record["well"]) for record in records})
        for well in tqdm(wells, desc="load S5 target-well references"):
            typewell_path = S5_DATA_ROOT / "train" / f"{well}__typewell.csv"
            horizontal_path = S5_DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
            typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
            typewell = typewell.apply(pd.to_numeric, errors="coerce").dropna()
            typewell = (
                typewell.groupby("TVT", as_index=False)["GR"]
                .mean()
                .sort_values("TVT")
            )
            horizontal = pd.read_csv(horizontal_path, usecols=["MD", "X", "Y"])
            for column in ("MD", "X", "Y"):
                horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
            horizontal = horizontal.dropna().sort_values("MD")
            self.rows[well] = {
                "tvt": typewell["TVT"].to_numpy(np.float32),
                "gr": _s5_robust_normalise(typewell["GR"].to_numpy(float)),
                "md": horizontal["MD"].to_numpy(np.float32),
                "x": horizontal["X"].to_numpy(np.float32),
                "y": horizontal["Y"].to_numpy(np.float32),
            }


def _s5_affine_metrics(observed, observed_valid, candidate, candidate_valid):
    observed = np.asarray(observed, np.float32)
    candidate = np.asarray(candidate, np.float32)
    valid = candidate_valid & observed_valid[None, :]
    weight = valid.astype(np.float32)
    count = np.maximum(weight.sum(axis=1), 1.0)
    mean_x = (candidate * weight).sum(axis=1) / count
    mean_y = (observed[None, :] * weight).sum(axis=1) / count
    centred_x = (candidate - mean_x[:, None]) * weight
    centred_y = (observed[None, :] - mean_y[:, None]) * weight
    covariance = (centred_x * centred_y).sum(axis=1)
    variance_x = (centred_x * centred_x).sum(axis=1)
    variance_y = (centred_y * centred_y).sum(axis=1)
    slope = np.clip(covariance / np.maximum(variance_x, 1e-6), -5.0, 5.0)
    intercept = mean_y - slope * mean_x
    calibrated = slope[:, None] * candidate + intercept[:, None]
    difference = (calibrated - observed[None, :]) * weight
    mae = np.abs(difference).sum(axis=1) / count
    rmse = np.sqrt((difference * difference).sum(axis=1) / count)
    ncc = covariance / np.sqrt(np.maximum(variance_x * variance_y, 1e-8))
    coverage = weight.mean(axis=1)
    bad = count < 24
    ncc[bad] = -1.0
    mae[bad] = 8.0
    rmse[bad] = 8.0
    calibrated[~valid] = 0.0
    return {
        "valid": valid,
        "calibrated": calibrated.astype(np.float32),
        "slope": slope.astype(np.float32),
        "intercept": intercept.astype(np.float32),
        "mae": mae.astype(np.float32),
        "rmse": rmse.astype(np.float32),
        "ncc": ncc.astype(np.float32),
        "coverage": coverage.astype(np.float32),
    }


class S5PanelBuilder:
    def __init__(self, records, banks, fold_map, references):
        self.records = records
        self.banks = banks
        self.fold_map = fold_map
        self.references = references

    def _bank(self, record):
        well_fold = int(self.fold_map[str(record["well"])])
        return self.banks[well_fold]

    def build(self, index, mode="full", full_menu=False):
        record = self.records[index]
        bank = self._bank(record)
        prefix = int(PREFIX_COLUMNS_RAFT)
        candidate_tvt, delta_s = bank.trajectories(record)
        observed = np.asarray(record["horizontal_gr"], np.float32)[prefix:]
        observed_valid = (
            np.asarray(record["horizontal_valid"], np.float32)[prefix:] > 0.5
        )
        reference_row = self.references.rows[str(record["well"])]
        if mode == "shuffled_typewell":
            donor = self.records[
                (index + max(1, len(self.records) // 3)) % len(self.records)
            ]
            reference_row = self.references.rows[str(donor["well"])]
        elif mode == "no_gr":
            observed = np.zeros_like(observed)
            observed_valid = np.zeros_like(observed_valid)
        elif mode != "full":
            raise ValueError(mode)

        flat = np.interp(
            candidate_tvt.ravel(),
            reference_row["tvt"],
            reference_row["gr"],
            left=np.nan,
            right=np.nan,
        )
        candidate_gr = flat.reshape(candidate_tvt.shape).astype(np.float32)
        candidate_valid = np.isfinite(candidate_gr)
        candidate_gr = np.nan_to_num(candidate_gr, nan=0.0)
        metrics = _s5_affine_metrics(
            observed, observed_valid, candidate_gr, candidate_valid
        )

        md = np.asarray(record["column_md"], np.float32)
        z = np.asarray(record["column_z"], np.float32)
        eval_md = md[prefix:]
        eval_z = z[prefix:]
        span = max(float(eval_md[-1] - md[prefix - 1]), 1.0)
        heel_md = md[max(0, prefix - 17):prefix]
        heel_s = (
            np.asarray(record["center_tvt"], np.float32)[max(0, prefix - 17):prefix]
            + z[max(0, prefix - 17):prefix]
        )
        heel_slope = float(np.polyfit(heel_md, heel_s, 1)[0]) if len(heel_md) >= 4 else 0.0
        initial_columns = min(24, S5_SEQUENCE_LENGTH)
        candidate_slope = (
            delta_s[:, initial_columns - 1] - delta_s[:, 0]
        ) / max(float(eval_md[initial_columns - 1] - eval_md[0]), 1.0)
        slope_difference = np.abs(candidate_slope - heel_slope)

        match_rank = (
            _s5_rank(metrics["ncc"], descending=True)
            + _s5_rank(metrics["mae"], descending=False)
            + 0.25 * _s5_rank(slope_difference, descending=False)
        )
        match_order = np.argsort(match_rank, kind="mergesort")
        slope_order = np.argsort(slope_difference, kind="mergesort")
        selected = []
        primary_count = S5_MODEL_CANDIDATES if mode == "no_gr" else 48
        primary_order = slope_order if mode == "no_gr" else match_order
        for candidate_index in primary_order:
            if int(candidate_index) not in selected:
                selected.append(int(candidate_index))
            if len(selected) >= primary_count:
                break
        for candidate_index in slope_order:
            if int(candidate_index) not in selected:
                selected.append(int(candidate_index))
            if len(selected) >= S5_MODEL_CANDIDATES:
                break
        for candidate_index in match_order:
            if int(candidate_index) not in selected:
                selected.append(int(candidate_index))
            if len(selected) >= S5_MODEL_CANDIDATES:
                break
        selected = np.asarray(selected[:S5_MODEL_CANDIDATES], np.int64)
        if full_menu:
            selected = np.arange(len(candidate_tvt), dtype=np.int64)

        candidate_tvt = candidate_tvt[selected]
        delta_s = delta_s[selected]
        candidate_gr = candidate_gr[selected]
        candidate_valid = candidate_valid[selected]
        calibrated = metrics["calibrated"][selected]
        valid = metrics["valid"][selected]
        difference = (calibrated - observed[None, :]) * valid
        absolute_difference = np.abs(difference)
        window16 = uniform_filter1d(difference, 16, axis=1, mode="nearest")
        window64 = uniform_filter1d(difference, 64, axis=1, mode="nearest")
        progress = np.arange(1, S5_SEQUENCE_LENGTH + 1, dtype=np.float32)
        progress /= float(S5_SEQUENCE_LENGTH)
        candidate_dip = np.gradient(delta_s, eval_md, axis=1) * 100.0
        geometry = np.asarray(record["geometry"], np.float32)[:, prefix:]
        rel_x = geometry[5]
        rel_y = geometry[6]
        rel_z = geometry[7]
        dx = geometry[8]
        dy = geometry[9]
        dz = geometry[10]

        repeated = lambda values: np.broadcast_to(
            np.asarray(values, np.float32)[None, :], candidate_tvt.shape
        )
        sequence = np.stack(
            [
                repeated(observed),
                repeated(observed_valid.astype(np.float32)),
                candidate_gr,
                candidate_valid.astype(np.float32),
                calibrated,
                difference,
                absolute_difference,
                calibrated * repeated(observed),
                delta_s / 128.0,
                candidate_dip,
                repeated(progress),
                repeated((eval_z - float(z[prefix - 1])) / 128.0),
                window16,
                window64,
                repeated(rel_x),
                repeated(rel_y),
                repeated(rel_z),
                repeated(dx),
                repeated(dy),
                repeated(dz),
            ],
            axis=1,
        ).astype(np.float32)

        anchor_md = float(md[prefix - 1])
        anchor_x = float(np.interp(anchor_md, reference_row["md"], reference_row["x"]))
        anchor_y = float(np.interp(anchor_md, reference_row["md"], reference_row["y"]))
        total_variation = np.sum(np.abs(np.diff(delta_s, axis=1)), axis=1)
        selected_match_rank = _s5_rank(match_rank[selected], descending=False)
        summary = np.column_stack(
            [
                metrics["ncc"][selected],
                metrics["mae"][selected] / 4.0,
                metrics["rmse"][selected] / 4.0,
                metrics["coverage"][selected],
                metrics["slope"][selected],
                metrics["intercept"][selected] / 4.0,
                slope_difference[selected] * 100.0,
                delta_s[:, -1] / 128.0,
                total_variation / 256.0,
                selected_match_rank / max(len(selected) - 1, 1),
                np.log1p(bank.priors[selected] * bank.source_panels),
                np.full(len(selected), anchor_x / 10000.0),
                np.full(len(selected), anchor_y / 10000.0),
                np.full(len(selected), span / 5000.0),
                np.full(len(selected), float(geometry[15, 0])),
            ]
        ).astype(np.float32)

        truth_tvt = (
            np.asarray(record["center_tvt"], np.float32)[prefix:]
            + np.asarray(record["truth_residual"], np.float32)[prefix:]
        )
        candidate_mse = np.mean(
            (candidate_tvt - truth_tvt[None, :]) ** 2, axis=1
        ).astype(np.float32)
        return {
            "sequence": sequence,
            "summary": summary,
            "candidate_tvt": candidate_tvt.astype(np.float32),
            "candidate_mse": candidate_mse,
            "truth_tvt": truth_tvt.astype(np.float32),
            "panel_weight": np.float32(record["panel_weight"]),
            "selected_indices": selected,
            "bank_candidates": int(len(bank.templates)),
        }


class S5Dataset(Dataset):
    def __init__(self, builder, mode="full"):
        self.builder = builder
        self.mode = mode

    def __len__(self):
        return len(self.builder.records)

    def __getitem__(self, index):
        row = self.builder.build(index, mode=self.mode, full_menu=False)
        return {
            "sequence": torch.from_numpy(row["sequence"]),
            "summary": torch.from_numpy(row["summary"]),
            "candidate_tvt": torch.from_numpy(row["candidate_tvt"]),
            "candidate_mse": torch.from_numpy(row["candidate_mse"]),
            "truth_tvt": torch.from_numpy(row["truth_tvt"]),
            "panel_weight": torch.tensor(row["panel_weight"], dtype=torch.float32),
        }


class S5ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                5,
                padding=2 * dilation,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values):
        return F.gelu(values + self.net(values))


class S5WholePathScorer(nn.Module):
    def __init__(self, sequence_channels, summary_channels):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv1d(sequence_channels, 40, 7, stride=2, padding=3),
            nn.GroupNorm(8, 40),
            nn.GELU(),
            S5ResidualBlock(40, 1),
            S5ResidualBlock(40, 2),
            S5ResidualBlock(40, 4),
            nn.Conv1d(40, 48, 5, stride=2, padding=2),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            S5ResidualBlock(48, 2),
        )
        self.summary = nn.Sequential(
            nn.Linear(summary_channels, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_set = nn.TransformerEncoder(layer, num_layers=2)
        self.score = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.10), nn.Linear(64, 1)
        )

    def forward(self, sequence, summary):
        batch, candidates, channels, length = sequence.shape
        values = sequence.reshape(batch * candidates, channels, length)
        hidden = self.sequence(values)
        pooled = torch.cat([hidden.mean(dim=-1), hidden.amax(dim=-1)], dim=1)
        summary_hidden = self.summary(
            summary.reshape(batch * candidates, summary.shape[-1])
        )
        candidate = self.fusion(torch.cat([pooled, summary_hidden], dim=1))
        candidate = candidate.reshape(batch, candidates, 128)
        candidate = self.candidate_set(candidate)
        return self.score(candidate).squeeze(-1)


def _s5_loss(scores, candidate_tvt, candidate_mse, truth_tvt, panel_weight):
    candidate_rmse = torch.sqrt(candidate_mse.clamp_min(1e-6))
    target = torch.softmax(
        -(candidate_rmse - candidate_rmse.amin(dim=1, keepdim=True))
        / S5_TARGET_TEMPERATURE_FT,
        dim=1,
    )
    listwise = -(target * F.log_softmax(scores.float(), dim=1)).sum(dim=1)
    probability = torch.softmax(scores.float(), dim=1)
    posterior = torch.sum(probability[:, :, None] * candidate_tvt, dim=1)
    path_mse = torch.mean((posterior - truth_tvt) ** 2, dim=1) / 64.0
    expected_mse = torch.sum(probability * candidate_mse, dim=1) / 64.0
    per_panel = (
        listwise
        + S5_POSTERIOR_PATH_WEIGHT * path_mse
        + S5_EXPECTED_MSE_WEIGHT * expected_mse
    )
    weight = panel_weight / panel_weight.mean().clamp_min(1.0)
    return (per_panel * weight.clamp(0.2, 5.0)).mean(), {
        "listwise": listwise.mean(),
        "path_mse": (path_mse * 64.0).mean(),
        "expected_mse": (expected_mse * 64.0).mean(),
    }


def _s5_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=(device_raft.type == "cuda"))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=(device_raft.type == "cuda"))


def _s5_loader(dataset, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=S5_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device_raft.type == "cuda"),
        generator=generator,
    )


@torch.no_grad()
def _s5_evaluate(model, builder, mode="full", return_rows=False):
    model.eval()
    errors = []
    map_errors = []
    anchor_errors = []
    oracle_sse = 0.0
    total_rows = 0
    bywell = []
    rows = []
    for index, record in enumerate(builder.records):
        data = builder.build(index, mode=mode, full_menu=False)
        sequence = torch.from_numpy(data["sequence"])[None].to(device_raft)
        summary = torch.from_numpy(data["summary"])[None].to(device_raft)
        with torch.autocast(
            device_type=device_raft.type,
            dtype=torch.float16,
            enabled=(device_raft.type == "cuda"),
        ):
            scores = model(sequence, summary)[0].float()
        probability = torch.softmax(scores, dim=0).cpu().numpy()
        candidates = data["candidate_tvt"].astype(np.float64)
        posterior_columns = probability @ candidates
        map_columns = candidates[int(np.argmax(probability))]
        column_md = np.asarray(record["column_md"], np.float64)[PREFIX_COLUMNS_RAFT:]
        posterior = np.interp(record["eval_md"], column_md, posterior_columns)
        map_prediction = np.interp(record["eval_md"], column_md, map_columns)
        prefix = int(PREFIX_COLUMNS_RAFT)
        anchor_s = float(
            record["center_tvt"][prefix - 1] + record["column_z"][prefix - 1]
        )
        anchor = anchor_s - np.asarray(record["eval_z"], np.float64)
        truth = np.asarray(record["eval_truth"], np.float64)
        error = posterior - truth
        map_error = map_prediction - truth
        anchor_error = anchor - truth
        candidate_full = np.stack(
            [np.interp(record["eval_md"], column_md, path) for path in candidates]
        )
        candidate_sse = np.sum((candidate_full - truth[None, :]) ** 2, axis=1)
        oracle_sse += float(np.min(candidate_sse))
        total_rows += len(truth)
        errors.append(error)
        map_errors.append(map_error)
        anchor_errors.append(anchor_error)
        bywell.append(
            {
                "well": record["well"],
                "rows": len(truth),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "map_rmse": float(np.sqrt(np.mean(map_error**2))),
                "anchor_rmse": float(np.sqrt(np.mean(anchor_error**2))),
                "oracle_rmse": float(np.sqrt(np.min(candidate_sse) / len(truth))),
                "posterior_entropy": float(
                    -np.sum(probability * np.log(probability + 1e-12))
                ),
            }
        )
        if return_rows:
            rows.append(
                pd.DataFrame(
                    {
                        "well": record["well"],
                        "row_idx": record["eval_rows"],
                        "truth": truth,
                        "anchor": anchor,
                        "prediction": posterior,
                        "map_prediction": map_prediction,
                    }
                )
            )
    errors = np.concatenate(errors)
    map_errors = np.concatenate(map_errors)
    anchor_errors = np.concatenate(anchor_errors)
    bywell = pd.DataFrame(bywell)
    metrics = {
        "rows": int(len(errors)),
        "wells": int(len(bywell)),
        "anchor_rmse": float(np.sqrt(np.mean(anchor_errors**2))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "map_rmse": float(np.sqrt(np.mean(map_errors**2))),
        "oracle_rmse": float(np.sqrt(oracle_sse / max(total_rows, 1))),
        "mae": float(np.mean(np.abs(errors))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "mean_entropy": float(bywell["posterior_entropy"].mean()),
    }
    row_frame = pd.concat(rows, ignore_index=True) if return_rows else None
    return metrics, bywell, row_frame


def _s5_oracle_audit(builder):
    rows = []
    totals = {
        "rows": 0,
        "anchor_sse": 0.0,
        "full_oracle_sse": 0.0,
        "pruned_oracle_sse": 0.0,
        "heuristic_sse": 0.0,
    }
    for index, record in enumerate(
        tqdm(builder.records, desc="S5 physical path-bank oracle")
    ):
        full = builder.build(index, mode="full", full_menu=True)
        pruned = builder.build(index, mode="full", full_menu=False)
        prefix = int(PREFIX_COLUMNS_RAFT)
        column_md = np.asarray(record["column_md"], np.float64)[prefix:]
        truth = np.asarray(record["eval_truth"], np.float64)
        full_paths = np.stack(
            [
                np.interp(record["eval_md"], column_md, path)
                for path in full["candidate_tvt"]
            ]
        )
        pruned_paths = np.stack(
            [
                np.interp(record["eval_md"], column_md, path)
                for path in pruned["candidate_tvt"]
            ]
        )
        full_sse = np.sum((full_paths - truth[None, :]) ** 2, axis=1)
        pruned_sse = np.sum((pruned_paths - truth[None, :]) ** 2, axis=1)
        anchor_s = float(
            record["center_tvt"][prefix - 1] + record["column_z"][prefix - 1]
        )
        anchor = anchor_s - np.asarray(record["eval_z"], np.float64)
        anchor_sse = float(np.sum((anchor - truth) ** 2))
        totals["rows"] += len(truth)
        totals["anchor_sse"] += anchor_sse
        totals["full_oracle_sse"] += float(np.min(full_sse))
        totals["pruned_oracle_sse"] += float(np.min(pruned_sse))
        totals["heuristic_sse"] += float(pruned_sse[0])
        rows.append(
            {
                "well": record["well"],
                "rows": len(truth),
                "anchor_rmse": float(np.sqrt(anchor_sse / len(truth))),
                "full_oracle_rmse": float(np.sqrt(np.min(full_sse) / len(truth))),
                "pruned_oracle_rmse": float(
                    np.sqrt(np.min(pruned_sse) / len(truth))
                ),
                "heuristic_rmse": float(np.sqrt(pruned_sse[0] / len(truth))),
                "full_candidates": len(full_sse),
                "pruned_candidates": len(pruned_sse),
            }
        )
    bywell = pd.DataFrame(rows)
    count = max(int(totals["rows"]), 1)
    report = {
        "rows": int(totals["rows"]),
        "wells": int(len(bywell)),
        "anchor_rmse": float(np.sqrt(totals["anchor_sse"] / count)),
        "full_oracle_rmse": float(np.sqrt(totals["full_oracle_sse"] / count)),
        "pruned_oracle_rmse": float(
            np.sqrt(totals["pruned_oracle_sse"] / count)
        ),
        "heuristic_top1_rmse": float(np.sqrt(totals["heuristic_sse"] / count)),
        "median_pruned_oracle_rmse": float(bywell["pruned_oracle_rmse"].median()),
        "p95_pruned_oracle_rmse": float(
            bywell["pruned_oracle_rmse"].quantile(0.95)
        ),
    }
    return report, bywell


def _s5_train(train_builder, valid_builder, sequence_channels, summary_channels):
    _seed_raft(S5_SEED)
    model = S5WholePathScorer(sequence_channels, summary_channels).to(device_raft)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=S5_LEARNING_RATE, weight_decay=S5_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=S5_EPOCHS, eta_min=S5_MIN_LEARNING_RATE
    )
    scaler = _s5_scaler()
    train_dataset = S5Dataset(train_builder, mode="full")
    loader = _s5_loader(train_dataset, shuffle=True, seed=S5_SEED + 31)
    best_rmse = np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, S5_EPOCHS + 1):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            sequence = batch["sequence"].to(device_raft, non_blocking=True)
            summary = batch["summary"].to(device_raft, non_blocking=True)
            candidate_tvt = batch["candidate_tvt"].to(
                device_raft, non_blocking=True
            )
            candidate_mse = batch["candidate_mse"].to(
                device_raft, non_blocking=True
            )
            truth_tvt = batch["truth_tvt"].to(device_raft, non_blocking=True)
            panel_weight = batch["panel_weight"].to(device_raft, non_blocking=True)
            with torch.autocast(
                device_type=device_raft.type,
                dtype=torch.float16,
                enabled=(device_raft.type == "cuda"),
            ):
                scores = model(sequence, summary)
            loss, components = _s5_loss(
                scores,
                candidate_tvt,
                candidate_mse,
                truth_tvt,
                panel_weight,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(
                {
                    "loss": float(loss.detach().cpu()),
                    **{
                        name: float(value.detach().cpu())
                        for name, value in components.items()
                    },
                }
            )
        scheduler.step()
        metrics, _, _ = _s5_evaluate(
            model, valid_builder, mode="full", return_rows=False
        )
        row = {
            "epoch": epoch,
            **{
                f"train_{name}": float(np.mean([item[name] for item in losses]))
                for name in losses[0]
            },
            **{f"valid_{name}": value for name, value in metrics.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("S5 LISTWISE EPOCH", row, flush=True)
        if float(metrics["rmse"]) < best_rmse - 1e-4:
            best_rmse = float(metrics["rmse"])
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch >= 6 and stale >= S5_PATIENCE:
            print(f"S5 early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("S5 produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint_path = S5_MODEL_DIR / "best_fold1.pt"
    torch.save(
        {
            "model": best_state,
            "epoch": best_epoch,
            "sequence_channels": sequence_channels,
            "summary_channels": summary_channels,
            "model_candidates": S5_MODEL_CANDIDATES,
            "seed": S5_SEED,
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(S5_OUT_DIR / "training_history.csv", index=False)
    return model, best_epoch, checkpoint_path, (time.time() - started) / 60.0


def _s5_main():
    started = time.time()
    fold_map = (
        fold.groupby("well")["fold"].first().astype(int).to_dict()
    )
    train_records = list(train_store_raft.records)
    valid_records = list(valid_store_raft.records)
    all_records = train_records + valid_records
    references = S5ReferenceStore(all_records)

    banks = {}
    bank_rows = []
    # A panel from fold f is scored only against templates from the other folds.
    for held_out_fold in (1, 2, 3, 4, 5):
        source = [
            record
            for record in train_records
            if int(fold_map[str(record["well"])]) != held_out_fold
        ]
        if held_out_fold == 1:
            source = train_records
        bank = S5TemplateBank(
            source,
            label=f"heldout_{held_out_fold}",
            seed=S5_SEED + held_out_fold * 101,
        )
        banks[held_out_fold] = bank
        bank_rows.append(
            {
                "held_out_fold": held_out_fold,
                "source_panels": bank.source_panels,
                "candidates": len(bank.templates),
                "median_span": bank.median_span,
            }
        )
    bank_report = pd.DataFrame(bank_rows)
    bank_report.to_csv(S5_OUT_DIR / "bank_report.csv", index=False)
    print("\nS5 BANKS")
    print(bank_report.to_string(index=False))

    train_builder = S5PanelBuilder(train_records, banks, fold_map, references)
    valid_builder = S5PanelBuilder(valid_records, banks, fold_map, references)
    oracle_report, oracle_bywell = _s5_oracle_audit(valid_builder)
    oracle_bywell.to_csv(S5_OUT_DIR / "oracle_bywell.csv", index=False)
    with open(S5_OUT_DIR / "oracle_report.json", "w", encoding="utf-8") as handle:
        json.dump(oracle_report, handle, indent=2)
    print("\nS5 ORACLE GATE")
    print(json.dumps(oracle_report, indent=2))

    if oracle_report["pruned_oracle_rmse"] > S5_TRAIN_ORACLE_GATE_FT:
        verdict = {
            "experiment": "fullwell_listwise_path_scorer_s5_onefold_v0",
            "stage": "candidate_oracle_gate",
            "standalone": True,
            "oracle": oracle_report,
            "train_oracle_gate_ft": S5_TRAIN_ORACLE_GATE_FT,
            "strict_pass": False,
            "next": (
                "FAIL -> no-neighbour physical path menu lacks sufficient ceiling; "
                "do not train the S5 selector"
            ),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        with open(S5_OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
            json.dump(verdict, handle, indent=2)
        print("\nFINAL VERDICT")
        print(json.dumps(verdict, indent=2))
        print("saved:", S5_OUT_DIR)
        return verdict

    probe = train_builder.build(0, mode="full", full_menu=False)
    sequence_channels = int(probe["sequence"].shape[1])
    summary_channels = int(probe["summary"].shape[1])
    probe_model = S5WholePathScorer(sequence_channels, summary_channels)
    parameters = int(sum(parameter.numel() for parameter in probe_model.parameters()))
    del probe_model, probe
    print(
        {
            "experiment": "fullwell_listwise_path_scorer_s5_onefold_v0",
            "device": str(device_raft),
            "train_panels": len(train_records),
            "valid_wells": len(valid_records),
            "model_candidates": S5_MODEL_CANDIDATES,
            "sequence_channels": sequence_channels,
            "summary_channels": summary_channels,
            "parameters": parameters,
        },
        flush=True,
    )

    model, best_epoch, checkpoint_path, training_minutes = _s5_train(
        train_builder, valid_builder, sequence_channels, summary_channels
    )
    intervention_rows = []
    winner_rows = None
    for mode in ("full", "no_gr", "shuffled_typewell"):
        metrics, bywell, rows = _s5_evaluate(
            model,
            valid_builder,
            mode=mode,
            return_rows=(mode == "full"),
        )
        intervention_rows.append({"mode": mode, **metrics})
        bywell.to_csv(S5_OUT_DIR / f"{mode}_bywell.csv", index=False)
        if mode == "full":
            winner_rows = rows
    interventions = pd.DataFrame(intervention_rows)
    full_rmse = float(interventions.loc[interventions["mode"] == "full", "rmse"].iloc[0])
    interventions["degradation_vs_full"] = interventions["rmse"] - full_rmse
    interventions.to_csv(S5_OUT_DIR / "interventions.csv", index=False)
    winner_rows.to_pickle(S5_OUT_DIR / "fold1_predictions.pkl")
    no_gr_degradation = float(
        interventions.loc[
            interventions["mode"] == "no_gr", "degradation_vs_full"
        ].iloc[0]
    )
    shuffled_degradation = float(
        interventions.loc[
            interventions["mode"] == "shuffled_typewell", "degradation_vs_full"
        ].iloc[0]
    )
    architecture_pass = bool(full_rmse <= S5_ARCHITECTURE_GATE_FT)
    target_pass = bool(full_rmse <= S5_TARGET_GATE_FT)
    causal_pass = bool(no_gr_degradation >= 0.30 and shuffled_degradation >= 0.30)
    strict_pass = bool(architecture_pass and causal_pass)
    verdict = {
        "experiment": "fullwell_listwise_path_scorer_s5_onefold_v0",
        "stage": "trained_listwise_scorer",
        "sample_unit": "complete panel/well state",
        "standalone": True,
        "train_panels": len(train_records),
        "valid_wells": len(valid_records),
        "oracle": oracle_report,
        "model_candidates": S5_MODEL_CANDIDATES,
        "parameters": parameters,
        "best_epoch": int(best_epoch),
        "rmse": full_rmse,
        "no_gr_degradation": no_gr_degradation,
        "shuffled_typewell_degradation": shuffled_degradation,
        "architecture_pass": architecture_pass,
        "target_5p5_pass": target_pass,
        "causal_pass": causal_pass,
        "strict_pass": strict_pass,
        "checkpoint": str(checkpoint_path),
        "training_minutes": training_minutes,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "next": (
            "PASS -> run second seed before five-fold expansion"
            if strict_pass
            else "FAIL -> close S5 exact path-bank/listwise representation"
        ),
    }
    with open(S5_OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2)
    print("\nS5 INTERVENTIONS")
    print(interventions.to_string(index=False))
    print("\nFINAL VERDICT")
    print(json.dumps(verdict, indent=2))
    print("saved:", S5_OUT_DIR)
    return verdict


s5_verdict = _s5_main()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
