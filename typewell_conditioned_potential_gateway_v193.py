"""V193: typewell-conditioned datum-free structural potential gateway.

Learn a family of scalar potentials F(X, Y | typewell GR) from within-well differences:

    F(p_i) - F(p_j) = (TVT + Z)_i - (TVT + Z)_j

The unknown per-well datum cancels exactly. The legal typewell GR curve conditions
the field family, and the datum is recovered once from the visible TVT_input prefix.

No historical prediction, PF path, candidate bank, neighbour prediction, or
test/validation truth is consumed. Protocol: Fold4/5 train, Fold2 selects the
epoch, Fold3 is evaluated once, and Fold1 is never loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT = "typewell_conditioned_potential_gateway_v193"
SEED = 20260805
TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 2
GATEWAY_FOLD = 3
PAIR_STRIDE = 16
MAX_EPOCHS = 48
EVAL_EVERY = 2
PATIENCE_EVALS = 6
FAIL_FAST_EPOCH = 12
FAIL_FAST_RMSE = 12.0
BATCH_SIZE = 8192
TYPEWELL_NODES = 256


@dataclass
class Well:
    well: str
    fold: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    md: np.ndarray
    truth: np.ndarray
    tvt_input: np.ndarray
    hidden: np.ndarray
    typewell: np.ndarray


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _find_data_root(root: Path) -> Path:
    for candidate in (root / "rogii-wellbore-geology-prediction", root, Path.cwd()):
        if (candidate / "train").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate competition train directory")


def _find_metadata(root: Path) -> Path:
    candidates = (
        root / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
        root / "oof_exports" / "multicut_k5_full_spatial_lgb_bundle_v1" / "well_metadata.csv",
        Path.cwd() / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Frozen well_metadata.csv was not found")


def _load_well(data_root: Path, well: str, fold: int) -> Well:
    path = data_root / "train" / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame[["MD", "X", "Y", "Z", "TVT"]].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{well}: non-finite geometry or truth")
    tvt_input = frame["TVT_input"].to_numpy(np.float64)
    known = np.isfinite(tvt_input)
    hidden = ~known
    hidden_indices = np.flatnonzero(hidden)
    if not len(hidden_indices):
        raise RuntimeError(f"{well}: no hidden rows")
    first_hidden = int(hidden_indices[0])
    if first_hidden < 3 or not known[:first_hidden].all() or known[first_hidden:].any():
        raise RuntimeError(f"{well}: TVT_input is not a contiguous prefix")
    typewell_path = data_root / "train" / f"{well}__typewell.csv"
    typewell_frame = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    typewell_frame["TVT"] = pd.to_numeric(typewell_frame["TVT"], errors="coerce")
    typewell_frame["GR"] = pd.to_numeric(typewell_frame["GR"], errors="coerce")
    typewell_frame = typewell_frame.dropna(subset=["TVT"]).sort_values("TVT")
    typewell_frame = typewell_frame.groupby("TVT", as_index=False)["GR"].mean()
    tw_tvt = typewell_frame["TVT"].to_numpy(np.float64)
    tw_gr = typewell_frame["GR"].to_numpy(np.float64)
    finite_gr = np.isfinite(tw_gr)
    if len(tw_tvt) < 64 or finite_gr.sum() < 64 or np.ptp(tw_tvt) < 200.0:
        raise RuntimeError(f"{well}: insufficient legal typewell coverage")
    tw_gr = np.interp(tw_tvt, tw_tvt[finite_gr], tw_gr[finite_gr])
    query = np.linspace(tw_tvt[0], tw_tvt[-1], TYPEWELL_NODES)
    sampled_gr = np.interp(query, tw_tvt, tw_gr)
    median = float(np.median(sampled_gr))
    scale_gr = float(np.quantile(np.abs(sampled_gr - median), 0.75))
    scale_gr = max(scale_gr, 1.0)
    normalised_gr = np.clip((sampled_gr - median) / scale_gr, -6.0, 6.0)
    derivative = np.gradient(normalised_gr)
    typewell = np.stack((normalised_gr, derivative), axis=0).astype(np.float32)
    return Well(
        well=well,
        fold=fold,
        md=frame["MD"].to_numpy(np.float64),
        x=frame["X"].to_numpy(np.float64),
        y=frame["Y"].to_numpy(np.float64),
        z=frame["Z"].to_numpy(np.float64),
        truth=frame["TVT"].to_numpy(np.float64),
        tvt_input=tvt_input,
        hidden=hidden,
        typewell=typewell,
    )


def _coordinate_transform(train_wells: list[Well]) -> tuple[np.ndarray, np.ndarray]:
    xy = np.concatenate(
        [np.column_stack((well.x[::PAIR_STRIDE], well.y[::PAIR_STRIDE])) for well in train_wells]
    )
    centre = np.median(xy, axis=0)
    scale = np.quantile(np.abs(xy - centre), 0.90, axis=0)
    scale = np.where(scale < 1.0, 1.0, scale)
    return centre.astype(np.float32), scale.astype(np.float32)


def _normalise_xy(well: Well, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.column_stack((well.x, well.y)) - centre) / scale).astype(np.float32)


def _pair_dataset(
    wells: list[Well], centre: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    well_indices: list[np.ndarray] = []
    for well_index, well in enumerate(wells):
        xy = _normalise_xy(well, centre, scale)
        surface = well.truth + well.z
        sampled = np.unique(np.r_[np.arange(0, len(xy), PAIR_STRIDE), len(xy) - 1]).astype(int)
        anchor_pool = sampled[np.unique(np.linspace(0, len(sampled) - 1, 5).round().astype(int))]
        for index_position, index in enumerate(sampled):
            local_anchor = anchor_pool[np.argmin(np.abs(anchor_pool - index))]
            long_anchor = anchor_pool[index_position % len(anchor_pool)]
            for anchor, weight in ((local_anchor, 1.0), (long_anchor, 0.5)):
                if index == anchor:
                    continue
                points.append(xy[index])
                anchors.append(xy[anchor])
                targets.append(np.array(surface[index] - surface[anchor], np.float32))
                separation = abs(float(well.md[index] - well.md[anchor]))
                weights.append(np.array(weight * min(1.0, 256.0 / max(separation, 32.0)), np.float32))
                well_indices.append(np.array(well_index, np.int64))
        if (well_index + 1) % 100 == 0:
            print(f"V193 pairs {well_index + 1}/{len(wells)}", flush=True)
    point_array = np.asarray(points, np.float32)
    anchor_array = np.asarray(anchors, np.float32)
    target_array = np.asarray(targets, np.float32)
    weight_array = np.asarray(weights, np.float32)
    well_index_array = np.asarray(well_indices, np.int64)
    typewell_bank = np.stack([well.typewell for well in wells]).astype(np.float32)
    if not all(
        np.isfinite(array).all()
        for array in (point_array, anchor_array, target_array, weight_array, typewell_bank)
    ):
        raise RuntimeError("Non-finite V193 pair dataset")
    return point_array, anchor_array, target_array, weight_array, well_index_array, typewell_bank


class TypewellEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 24, 9, padding=4),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv1d(24, 32, 7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 48, 5, stride=2, padding=2),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
        )
        self.output = nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 32))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.network(values)
        pooled = torch.cat((hidden.mean(-1), hidden.amax(-1)), dim=1)
        return self.output(pooled)


class ConditionalFourierPotential(nn.Module):
    def __init__(self, seed: int = SEED):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        directions = torch.randn(64, 2, generator=generator)
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-6)
        frequencies = torch.logspace(math.log10(0.25), math.log10(32.0), 64)
        self.register_buffer("projection", directions * frequencies[:, None] * math.pi)
        input_features = 2 + 4 + 128 + 32
        self.typewell_encoder = TypewellEncoder()
        self.network = nn.Sequential(
            nn.Linear(input_features, 192),
            nn.SiLU(),
            nn.Linear(192, 192),
            nn.SiLU(),
            nn.Linear(192, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.linear = nn.Linear(2, 1, bias=False)

    def encode(self, xy: torch.Tensor) -> torch.Tensor:
        projection = xy @ self.projection.T
        polynomial = torch.stack(
            (xy[:, 0] ** 2, xy[:, 1] ** 2, xy[:, 0] * xy[:, 1], (xy**2).sum(1)), dim=1
        )
        return torch.cat((xy, polynomial, torch.sin(projection), torch.cos(projection)), dim=1)

    def encode_typewell(self, values: torch.Tensor) -> torch.Tensor:
        return self.typewell_encoder(values)

    def forward(self, xy: torch.Tensor, typewell_embedding: torch.Tensor) -> torch.Tensor:
        encoded = torch.cat((self.encode(xy), typewell_embedding), dim=1)
        return self.network(encoded).squeeze(1) + self.linear(xy).squeeze(1)


@torch.no_grad()
def _potential(
    model: ConditionalFourierPotential,
    xy: np.ndarray,
    typewell: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    typewell_tensor = torch.from_numpy(typewell[None]).to(device)
    embedding = model.encode_typewell(typewell_tensor)
    parts: list[np.ndarray] = []
    for start in range(0, len(xy), 32768):
        batch = torch.from_numpy(xy[start : start + 32768]).to(device)
        parts.append(model(batch, embedding.expand(len(batch), -1)).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def _predict_wells(
    model: ConditionalFourierPotential,
    wells: list[Well],
    centre: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    shuffled_typewells: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for well in wells:
        xy = _normalise_xy(well, centre, scale)
        typewell = well.typewell if shuffled_typewells is None else shuffled_typewells[well.well]
        field = _potential(model, xy, typewell, device)
        known = ~well.hidden
        visible_surface = well.tvt_input[known] + well.z[known]
        datum_values = visible_surface - field[known]
        datum_median = float(np.median(datum_values))
        hidden = np.flatnonzero(well.hidden)
        prediction = field[hidden] + datum_median - well.z[hidden]
        anchor = int(hidden[0] - 1)
        parts.append(
            pd.DataFrame(
                {
                    "well": well.well,
                    "fold": well.fold,
                    "row_idx": hidden,
                    "truth": well.truth[hidden],
                    "prediction": prediction,
                    "distance": well.md[hidden] - well.md[anchor],
                    "prefix_datum_mad": float(np.median(np.abs(datum_values - datum_median))),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _hold_surface(wells: list[Well]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for well in wells:
        hidden = np.flatnonzero(well.hidden)
        anchor = int(hidden[0] - 1)
        anchor_surface = float(well.tvt_input[anchor] + well.z[anchor])
        parts.append(
            pd.DataFrame(
                {
                    "well": well.well,
                    "fold": well.fold,
                    "row_idx": hidden,
                    "truth": well.truth[hidden],
                    "prediction": anchor_surface - well.z[hidden],
                    "distance": well.md[hidden] - well.md[anchor],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _metrics(frame: pd.DataFrame, endpoint: str) -> dict[str, object]:
    error = frame["prediction"].to_numpy(np.float64) - frame["truth"].to_numpy(np.float64)
    squared = error**2
    by_well = frame.assign(squared_error=squared).groupby("well")["squared_error"].mean().pow(0.5)
    report: dict[str, object] = {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "exact_sse": float(np.sum(squared)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(by_well.median()),
        "p95_well_rmse": float(by_well.quantile(0.95)),
    }
    for name, mask in (
        ("near_0_500_rmse", frame["distance"] <= 500.0),
        ("mid_500_1000_rmse", (frame["distance"] > 500.0) & (frame["distance"] <= 1000.0)),
        ("mid_1000_2000_rmse", (frame["distance"] > 1000.0) & (frame["distance"] <= 2000.0)),
        ("far_gt2000_rmse", frame["distance"] > 2000.0),
    ):
        local = squared[mask.to_numpy()]
        report[name] = float(np.sqrt(np.mean(local))) if len(local) else None
    return report


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--workers", type=int, default=4)
    args, _ = parser.parse_known_args()
    started = time.time()
    _seed_everything(SEED)

    root = args.root.resolve()
    data_root = _find_data_root(root)
    metadata_path = _find_metadata(root)
    out_dir = root / "oof_exports" / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    if not {"well", "fold"}.issubset(metadata.columns):
        raise RuntimeError(f"Unexpected fold metadata: {metadata.columns.tolist()}")
    metadata = metadata[["well", "fold"]].drop_duplicates("well")
    metadata["well"] = metadata["well"].astype(str).str.strip().str.lower()
    metadata["fold"] = metadata["fold"].astype(int)
    all_metadata = metadata.copy()
    allowed_folds = set(TRAIN_FOLDS) | {SELECTION_FOLD}
    metadata = metadata[metadata["fold"].isin(allowed_folds)].sort_values(["fold", "well"])

    wells: list[Well] = []
    failures: list[dict[str, object]] = []
    for number, row in enumerate(metadata.itertuples(index=False), 1):
        try:
            wells.append(_load_well(data_root, row.well, int(row.fold)))
        except Exception as error:
            failures.append({"well": row.well, "fold": int(row.fold), "reason": repr(error)})
        if number % 100 == 0:
            print(f"V193 records {number}/{len(metadata)} failures={len(failures)}", flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)
        raise RuntimeError(f"V193 record failures: {failures[:5]}")

    coverage = pd.Series([well.fold for well in wells]).value_counts().sort_index().to_dict()
    expected = {2: 155, 4: 155, 5: 154}
    if coverage != expected:
        raise RuntimeError(f"Unexpected coverage: {coverage}; expected {expected}")
    train_wells = [well for well in wells if well.fold in TRAIN_FOLDS]
    selection_wells = [well for well in wells if well.fold == SELECTION_FOLD]
    centre, scale = _coordinate_transform(train_wells)
    points, anchors, targets, weights, pair_well_indices, typewell_bank = _pair_dataset(
        train_wells, centre, scale
    )
    target_scale = float(np.quantile(np.abs(targets), 0.75))
    target_scale = max(target_scale, 1.0)

    print(
        "V193 CONFIG",
        json.dumps(
            {
                "coverage": coverage,
                "pairs": len(targets),
                "coordinate_centre": centre.tolist(),
                "coordinate_scale": scale.tolist(),
                "target_scale": target_scale,
                "typewell_nodes": TYPEWELL_NODES,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            indent=2,
        ),
        flush=True,
    )

    dataset = TensorDataset(
        torch.from_numpy(points),
        torch.from_numpy(anchors),
        torch.from_numpy(targets),
        torch.from_numpy(weights),
        torch.from_numpy(pair_well_indices),
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalFourierPotential().to(device)
    typewell_bank_tensor = torch.from_numpy(typewell_bank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=5e-5)

    best: dict[str, object] | None = None
    stale_evals = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for point, anchor, target, weight, well_index in loader:
            point = point.to(device, non_blocking=True)
            anchor = anchor.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            well_index = well_index.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            typewell_embeddings = model.encode_typewell(typewell_bank_tensor)
            pair_embedding = typewell_embeddings[well_index]
            difference = model(point, pair_embedding) - model(anchor, pair_embedding)
            element = F.smooth_l1_loss(difference / target_scale, target / target_scale, reduction="none")
            loss = torch.sum(element * weight) / torch.sum(weight).clamp_min(1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(torch.sum(element.detach() * weight).cpu())
            weight_sum += float(torch.sum(weight).cpu())
        scheduler.step()

        report: dict[str, object] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(weight_sum, 1e-9),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if epoch == 1 or epoch % EVAL_EVERY == 0:
            selection_prediction = _predict_wells(model, selection_wells, centre, scale, device)
            metric = _metrics(selection_prediction, "fold2_typewell_conditioned_potential")
            report.update({f"valid_{key}": value for key, value in metric.items() if key != "endpoint"})
            if best is None or float(metric["rmse"]) < float(best["rmse"]) - 1e-4:
                best = {
                    "epoch": epoch,
                    "rmse": float(metric["rmse"]),
                    "state": copy.deepcopy(model.state_dict()),
                    "metric": metric,
                }
                stale_evals = 0
            else:
                stale_evals += 1
        history.append(report)
        print("V193 EPOCH", report, flush=True)

        if epoch >= FAIL_FAST_EPOCH and best is not None and float(best["rmse"]) > FAIL_FAST_RMSE:
            print(
                f"V193 fail-fast: Fold2 best RMSE {best['rmse']:.4f} > {FAIL_FAST_RMSE:.1f}",
                flush=True,
            )
            break
        if stale_evals >= PATIENCE_EVALS:
            break

    if best is None:
        raise RuntimeError("V193 produced no validation checkpoint")
    model.load_state_dict(best["state"])
    del best["state"]
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    if float(best["rmse"]) > 8.0:
        verdict = _clean(
            {
                "experiment": EXPERIMENT,
                "stage": "Fold45_typewell_conditional_training_Fold2_fail_fast",
                "historical_prediction_inputs": [],
                "inference_inputs": ["X", "Y", "Z", "TVT_input visible prefix", "typewell GR"],
                "training_target": "within-well differences of S=TVT+Z",
                "protocol": {
                    "train_folds": list(TRAIN_FOLDS),
                    "selection_fold": SELECTION_FOLD,
                    "gateway_fold_loaded": False,
                    "fold1_loaded": False,
                },
                "pairs": len(targets),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "best_epoch": int(best["epoch"]),
                "fold2": best["metric"],
                "checks": {"fold2_rmse_le_8": False},
                "strict_pass": False,
                "elapsed_minutes": (time.time() - started) / 60.0,
                "next": "FAIL -> close conditional potential without loading Fold3 or repairing output",
            }
        )
        (out_dir / "verdict.json").write_text(
            json.dumps(verdict, indent=2, allow_nan=False), encoding="utf-8"
        )
        print("\nV193 FINAL VERDICT")
        print(json.dumps(verdict, indent=2, allow_nan=False))
        print(f"saved: {out_dir}")
        return

    gateway_metadata = all_metadata[all_metadata["fold"].eq(GATEWAY_FOLD)].sort_values("well")
    gateway_wells: list[Well] = []
    for row in gateway_metadata.itertuples(index=False):
        gateway_wells.append(_load_well(data_root, row.well, int(row.fold)))
    if len(gateway_wells) != 154:
        raise RuntimeError(f"Unexpected Fold3 gateway coverage: {len(gateway_wells)}")

    gateway_prediction = _predict_wells(model, gateway_wells, centre, scale, device)
    rolled = [well.typewell for well in gateway_wells]
    rolled = rolled[1:] + rolled[:1]
    shuffled_typewells = {
        well.well: typewell for well, typewell in zip(gateway_wells, rolled, strict=True)
    }
    shuffled_prediction = _predict_wells(
        model, gateway_wells, centre, scale, device, shuffled_typewells=shuffled_typewells
    )
    baseline_prediction = _hold_surface(gateway_wells)
    baseline_metric = _metrics(baseline_prediction, "hold_surface")
    candidate_metric = _metrics(gateway_prediction, "typewell_conditioned_potential")
    shuffled_metric = _metrics(shuffled_prediction, "rolled_typewell_control")
    summary = pd.DataFrame([baseline_metric, candidate_metric, shuffled_metric])
    summary.to_csv(out_dir / "matched_summary.csv", index=False)
    gateway_prediction.to_parquet(out_dir / "fold3_fullrow_oof.parquet", index=False)
    by_well = gateway_prediction.assign(
        squared_error=np.square(gateway_prediction["prediction"] - gateway_prediction["truth"])
    ).groupby("well", as_index=False).agg(
        rows=("row_idx", "size"),
        rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
        prefix_datum_mad=("prefix_datum_mad", "first"),
    )
    by_well.to_csv(out_dir / "fold3_by_well.csv", index=False)

    gain = float(baseline_metric["rmse"] - candidate_metric["rmse"])
    far_gain = float(baseline_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"])
    control_degradation = float(shuffled_metric["rmse"] - candidate_metric["rmse"])
    checks = {
        "fold2_rmse_le_8": float(best["rmse"]) <= 8.0,
        "fold3_rmse_le_7": float(candidate_metric["rmse"]) <= 7.0,
        "fold3_rmse_le_6": float(candidate_metric["rmse"]) <= 6.0,
        "gain_vs_hold_ge_1": gain >= 1.0,
        "far_gain_positive": far_gain > 0.0,
        "rolled_typewell_degradation_ge_0p20": control_degradation >= 0.20,
        "fold1_not_loaded": True,
    }
    strict_pass = bool(
        checks["fold2_rmse_le_8"]
        and checks["fold3_rmse_le_7"]
        and checks["gain_vs_hold_ge_1"]
        and checks["far_gain_positive"]
        and checks["rolled_typewell_degradation_ge_0p20"]
    )
    prediction_hash = hashlib.sha256(
        gateway_prediction["prediction"].to_numpy(np.float32).tobytes()
    ).hexdigest()
    verdict = {
        "experiment": EXPERIMENT,
        "stage": "Fold45_typewell_conditional_pairwise_training_Fold2_selection_Fold3_gateway",
        "historical_prediction_inputs": [],
        "inference_inputs": ["X", "Y", "Z", "TVT_input visible prefix", "typewell GR"],
        "training_target": "within-well differences of S=TVT+Z",
        "datum_handling": "c_well cancels in training and is recovered once from visible prefix",
        "protocol": {
            "train_folds": list(TRAIN_FOLDS),
            "selection_fold": SELECTION_FOLD,
            "gateway_fold": GATEWAY_FOLD,
            "fold1_loaded": False,
        },
        "pairs": len(targets),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(best["epoch"]),
        "fold2": best["metric"],
        "fold3": {
            "baseline": baseline_metric,
            "candidate": candidate_metric,
            "rolled_typewell_control": shuffled_metric,
        },
        "gain": gain,
        "far_gain": far_gain,
        "rolled_typewell_degradation": control_degradation,
        "checks": checks,
        "strict_pass": strict_pass,
        "prediction_sha256": prediction_hash,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "next": (
            "PASS -> freeze representation and repeat once on untouched Fold1"
            if strict_pass
            else "FAIL -> typewell-conditioned potential is not transferable; close without repair"
        ),
    }
    verdict = _clean(verdict)
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("\nV193 MATCHED SUMMARY")
    print(summary.to_string(index=False))
    print("\nV193 FINAL VERDICT")
    print(json.dumps(verdict, indent=2, allow_nan=False))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
