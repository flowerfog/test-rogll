"""Reconstructed full first-place-style model.

The public Default recipe is taken from V245.  PF and XY channels are rebuilt
from legal observations: PF is a per-column typewell-GR posterior heatmap;
XY is a leave-one-well-out inverse-distance local surface estimate.  The
winner's private implementations and weights are unavailable, so the report
labels these two parts as reconstructed rather than exact.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "first_place_default_rebuild_v245.py"
spec = importlib.util.spec_from_file_location("v245_impl_v247", SOURCE)
v245 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = v245
spec.loader.exec_module(v245)

OUT = v245.OOF_ROOT / os.environ.get("V247_OUT", "first_place_reconstructed_full_v247")
FOLDS = [int(x) for x in os.environ.get("V247_FOLDS", "1,2,3,4,5").split(",") if x.strip()]
SEEDS = [int(x) for x in os.environ.get("V247_SEEDS", "0,1,2").split(",") if x.strip()]
EPOCHS = int(os.environ.get("V247_EPOCHS", "18"))
BATCH = int(os.environ.get("V247_BATCH", "1"))
ACCUM = int(os.environ.get("V247_ACCUM", "2"))
WORKERS = int(os.environ.get("V247_WORKERS", "0"))
BRANCHES = [x.strip() for x in os.environ.get("V247_BRANCHES", "default,pf,xy").split(",") if x.strip()]
MAX_TRAIN_WELLS = int(os.environ.get("V247_MAX_TRAIN_WELLS", "0"))
MAX_VALID_WELLS = int(os.environ.get("V247_MAX_VALID_WELLS", "0"))

BASE_CHANNELS = len(v245.CHANNELS)
PF_CHANNELS = 6
EXTRA_CHANNELS = (
    "pf_probability_heatmap", "pf_multiscale_offset", "pf_std", "pf_ffbsi_offset",
    "pf_accumulated_tvt_offset", "pf_typewell_tvt_abs_gap",
    "xy_tvt_offset", "xy_tvt_difference", "xy_surface_offset",
    "xy_spread", "xy_distance", "xy_quality",
)
ALL_CHANNELS = tuple(v245.CHANNELS) + EXTRA_CHANNELS


def enrich(record):
    path = v245.DATA_ROOT / "train" / f"{record.well}__horizontal_well.csv"
    frame = pd.read_csv(path)
    record.x = v245.fill_coordinate(v245.numeric(frame, "X"))
    record.y = v245.fill_coordinate(v245.numeric(frame, "Y"))
    return record


class XYBank:
    def __init__(self, records):
        xs, ys, ss, owners = [], [], [], []
        for record in records:
            # Complete truth from outer-training donor wells is legal.  Only
            # the held-out target well must remain prefix-only.
            known = np.isfinite(record.tvt) & np.isfinite(record.x) & np.isfinite(record.y)
            xs.append(record.x[known]); ys.append(record.y[known])
            ss.append((record.tvt[known] + record.z[known]).astype(np.float32))
            owners.extend([record.well] * int(known.sum()))
        self.xy = np.column_stack((np.concatenate(xs), np.concatenate(ys))).astype(np.float32)
        self.surface = np.concatenate(ss).astype(np.float32)
        self.owner = np.asarray(owners, dtype=object)
        self.tree = cKDTree(self.xy)

    def query(self, record, md_axis, z_axis):
        qx = np.interp(md_axis, record.md, record.x)
        qy = np.interp(md_axis, record.md, record.y)
        query = np.column_stack((qx, qy))
        k = min(32, len(self.xy))
        distances, indices = self.tree.query(query, k=k)
        if k == 1:
            distances, indices = distances[:, None], indices[:, None]
        prediction = np.empty(len(query), np.float32)
        spread = np.empty(len(query), np.float32)
        nearest = np.empty(len(query), np.float32)
        for i in range(len(query)):
            keep = self.owner[indices[i]] != record.well
            if not np.any(keep):
                keep[:] = True
            d = np.maximum(distances[i][keep].astype(np.float64), 1.0)
            s = self.surface[indices[i][keep]].astype(np.float64)
            w = 1.0 / np.power(d, 1.5)
            w /= max(w.sum(), 1e-12)
            local_xy = self.xy[indices[i][keep]].astype(np.float64) - query[i].astype(np.float64)
            design = np.column_stack((np.ones(len(d)), local_xy))
            weighted = design * np.sqrt(w)[:, None]
            try:
                beta = np.linalg.lstsq(weighted, s * np.sqrt(w), rcond=None)[0]
                fitted = float(beta[0])
                residual = s - design @ beta
                plane_spread = float(np.sqrt(np.sum(w * residual * residual)))
                # Quality gate: reject ill-conditioned or poorly supported planes.
                condition = np.linalg.cond(weighted.T @ weighted)
                if not np.isfinite(condition) or condition > 1e8 or plane_spread > 12.0:
                    fitted = float(np.sum(w * s))
                    plane_spread = float(np.sqrt(np.sum(w * (s - fitted) ** 2)))
            except np.linalg.LinAlgError:
                fitted = float(np.sum(w * s))
                plane_spread = float(np.sqrt(np.sum(w * (s - fitted) ** 2)))
            prediction[i] = fitted - z_axis[i]
            spread[i] = plane_spread
            nearest[i] = float(np.min(d))
        return prediction, spread, nearest


def pf_channels(panel, record):
    """Multiscale bootstrap PF with systematic resampling and FFBSi trace."""
    _, axis = v245.alignment_axes(record)
    eval_rows = np.arange(record.anchor + 1, len(record.md))
    if len(eval_rows) < 2:
        return np.zeros((4, v245.HEIGHT, v245.WIDTH), np.float32)
    finite = np.isfinite(record.tw_tvt) & np.isfinite(record.tw_gr)
    tw_tvt, tw_gr = record.tw_tvt[finite], record.tw_gr[finite]
    gr = record.gr.copy()
    gr[~np.isfinite(gr)] = np.nanmedian(gr[np.isfinite(gr)])
    count = int(os.environ.get("V247_PF_PARTICLES", "256"))
    scales = (3.0, 5.0, 8.0, 12.0)
    visible = np.flatnonzero(np.isfinite(record.tvt_input) & np.isfinite(record.gr))
    reference = np.interp(record.tvt_input[visible], tw_tvt, tw_gr)
    if len(visible) >= 32 and np.nanstd(reference) > 1e-6:
        slope, intercept = np.polyfit(reference, record.gr[visible], 1)
        calibrated = slope * tw_gr + intercept
    else:
        calibrated = tw_gr.copy()
    kernel = np.ones(9, np.float64) / 9.0
    smooth = np.convolve(calibrated, kernel, mode="same")
    profiles = (tw_gr, calibrated, smooth, 0.5 * calibrated + 0.5 * smooth)
    expected_members, std_members = [], []
    ffbsi = None
    probability_rows = None
    for scale_index, scale in enumerate(scales):
        rng = np.random.default_rng(247000 + scale_index)
        surface = record.anchor_tvt + record.z[record.anchor] + rng.normal(0, 6, count)
        weights = np.full(count, 1.0 / count, np.float64)
        states = np.empty((len(eval_rows), count), np.float32)
        ancestors = np.empty((len(eval_rows), count), np.int32)
        filtered_surface = np.empty((len(eval_rows), count), np.float32)
        filtered_rate = np.empty((len(eval_rows), count), np.float32)
        filtered_weight = np.empty((len(eval_rows), count), np.float32)
        means = np.empty(len(eval_rows), np.float32)
        stds = np.empty(len(eval_rows), np.float32)
        ents = np.empty(len(eval_rows), np.float32)
        if scale_index == 1:
            probability_rows = np.empty((len(eval_rows), v245.HEIGHT), np.float32)
        previous = float(record.md[record.anchor])
        for step, row in enumerate(eval_rows):
            dm = max(float(record.md[row]) - previous, 1.0)
            surface += rng.normal(0, 0.01, count) * dm
            jump_mask = rng.random(count) < float(os.environ.get("V247_PF_JUMP_PROB", "0.002"))
            if np.any(jump_mask):
                surface[jump_mask] += rng.standard_t(df=3.0, size=int(jump_mask.sum())) * 8.0
            tvt = surface - record.z[row]
            expected_gr = np.interp(tvt, tw_tvt, profiles[scale_index])
            residual = (gr[row] - expected_gr) / max(float(np.nanstd(gr[:record.anchor + 1])), 10.0)
            weights *= np.exp(-0.5 * np.minimum(residual * residual / scale, 40.0))
            weights /= max(float(weights.sum()), 1e-12)
            filtered_surface[step] = surface.astype(np.float32)
            filtered_rate[step] = 0.0
            filtered_weight[step] = weights.astype(np.float32)
            ancestors[step] = np.arange(count, dtype=np.int32)
            # Accumulate row likelihoods, but only resample at disclosed
            # 64-sample block boundaries (or the final row).
            block_boundary = (step + 1) % 64 == 0 or step + 1 == len(eval_rows)
            if block_boundary and 1.0 / max(float(np.sum(weights * weights)), 1e-12) < count * 0.5:
                cdf = np.cumsum(weights)
                positions = (rng.random() + np.arange(count)) / count
                chosen = np.searchsorted(cdf, positions).clip(0, count - 1)
                surface = surface[chosen]
                tvt = surface - record.z[row]
                ancestors[step] = chosen.astype(np.int32)
                weights.fill(1.0 / count)
            states[step] = tvt
            means[step] = float(np.sum(weights * tvt))
            stds[step] = float(np.sqrt(np.sum(weights * (tvt - means[step]) ** 2)))
            hist, _ = np.histogram(tvt, bins=np.r_[axis - v245.TVT_STEP_FT / 2, axis[-1] + v245.TVT_STEP_FT / 2], weights=weights)
            hist /= max(float(hist.sum()), 1e-12)
            ents[step] = float(-np.sum(hist * np.log(np.maximum(hist, 1e-12))) / np.log(v245.HEIGHT))
            if scale_index == 1:
                probability_rows[step] = hist.astype(np.float32)
            previous = float(record.md[row])
        x = np.linspace(0, v245.WIDTH - 1, len(eval_rows))
        expected_members.append(np.interp(np.arange(v245.WIDTH), x, means))
        std_members.append(np.interp(np.arange(v245.WIDTH), x, stds))
        if scale_index == 0:
            # FFBSi backward simulation: sample each previous filtered state
            # using its filtering weight times the transition density to the
            # already sampled next state.
            trace_surface = np.empty(len(eval_rows), np.float32)
            final_weight = filtered_weight[-1].astype(np.float64)
            final_weight /= max(final_weight.sum(), 1e-12)
            index = int(np.searchsorted(np.cumsum(final_weight), rng.random(), side="right"))
            index = min(index, count - 1)
            trace_surface[-1] = filtered_surface[-1, index]
            for step in range(len(eval_rows) - 2, -1, -1):
                dm = max(float(record.md[eval_rows[step + 1]] - record.md[eval_rows[step]]), 1.0)
                sigma = max(0.01 * dm, 0.25)
                delta = (trace_surface[step + 1] - filtered_surface[step]) / sigma
                backward = filtered_weight[step].astype(np.float64) * np.exp(-0.5 * np.minimum(delta * delta, 80.0))
                total = float(backward.sum())
                if not np.isfinite(total) or total <= 1e-20:
                    backward = filtered_weight[step].astype(np.float64)
                    total = float(backward.sum())
                backward /= max(total, 1e-12)
                index = int(np.searchsorted(np.cumsum(backward), rng.random(), side="right"))
                index = min(index, count - 1)
                trace_surface[step] = filtered_surface[step, index]
            trace = trace_surface - record.z[eval_rows]
            ffbsi = np.interp(np.arange(v245.WIDTH), x, trace)
    offset = np.mean(expected_members, axis=0) - record.anchor_tvt
    std = np.mean(std_members, axis=0)
    ffbsi = ffbsi - record.anchor_tvt
    target_x = np.arange(v245.WIDTH)
    probability = np.stack([
        np.interp(target_x, x, probability_rows[:, state])
        for state in range(v245.HEIGHT)
    ]).astype(np.float32)
    probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-8)
    profiles = tuple(
        np.broadcast_to(value[None, :] / v245.TVT_RADIUS_FT, (v245.HEIGHT, v245.WIDTH))
        for value in (offset, std, ffbsi)
    )
    accumulated = np.r_[offset[0], offset[0] + np.cumsum(np.diff(offset))]
    accumulated_grid = np.broadcast_to(
        accumulated[None, :] / v245.TVT_RADIUS_FT,
        (v245.HEIGHT, v245.WIDTH),
    )
    state_offset = axis - record.anchor_tvt
    pf_gap = np.abs(state_offset[:, None] - offset[None, :]) / v245.TVT_RADIUS_FT
    return np.stack((probability, *profiles, accumulated_grid, pf_gap)).astype(np.float32)


def make_full_panel(record, xy_bank=None, training=False, bank=None, rng=None):
    panel = v245.make_panel(record, training=training, bank=bank, rng=rng)
    extra = [*pf_channels(panel, record)]
    if xy_bank is None:
        xy = np.zeros((6, v245.HEIGHT, v245.WIDTH), np.float32)
    else:
        _, z_axis = v245.alignment_axes(record)
        md_axis, _ = v245.alignment_axes(record)
        pred, spread, distance = xy_bank.query(record, md_axis, np.interp(md_axis, record.md, record.z))
        z_axis = np.interp(md_axis, record.md, record.z)
        baseline = record.anchor_tvt - (z_axis - record.z[record.anchor])
        surface_offset = pred + z_axis - (record.anchor_tvt + record.z[record.anchor])
        quality = np.exp(-spread / 12.0) / (1.0 + distance / 10000.0)
        xy = np.stack((
            np.broadcast_to(((pred - record.anchor_tvt) / v245.TVT_RADIUS_FT)[None, :], (v245.HEIGHT, v245.WIDTH)),
            np.broadcast_to(((pred - baseline) / v245.TVT_RADIUS_FT)[None, :], (v245.HEIGHT, v245.WIDTH)),
            np.broadcast_to((surface_offset / v245.TVT_RADIUS_FT)[None, :], (v245.HEIGHT, v245.WIDTH)),
            np.broadcast_to((spread / v245.TVT_RADIUS_FT)[None, :], (v245.HEIGHT, v245.WIDTH)),
            np.broadcast_to((np.log1p(distance) / 10.0)[None, :], (v245.HEIGHT, v245.WIDTH)),
            np.broadcast_to(quality[None, :], (v245.HEIGHT, v245.WIDTH)),
        )).astype(np.float32)
    image = np.nan_to_num(np.concatenate((panel["image"], np.asarray(extra), xy), axis=0),
                          nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    panel["image"] = image
    return panel


def static_extra(record, xy_bank):
    panel = v245.make_panel(record, training=False)
    pf = pf_channels(panel, record)
    md_axis, _ = v245.alignment_axes(record)
    z_axis = np.interp(md_axis, record.md, record.z)
    pred, spread, distance = xy_bank.query(record, md_axis, z_axis)
    baseline = record.anchor_tvt - (z_axis - record.z[record.anchor])
    surface_offset = pred + z_axis - (record.anchor_tvt + record.z[record.anchor])
    quality = np.exp(-spread / 12.0) / (1.0 + distance / 10000.0)
    xy = np.stack((
        np.broadcast_to(((pred - record.anchor_tvt) / v245.TVT_RADIUS_FT)[None, :], (v245.HEIGHT, v245.WIDTH)),
        np.broadcast_to(((pred - baseline) / v245.TVT_RADIUS_FT)[None, :], (v245.HEIGHT, v245.WIDTH)),
        np.broadcast_to((surface_offset / v245.TVT_RADIUS_FT)[None, :], (v245.HEIGHT, v245.WIDTH)),
        np.broadcast_to((spread / v245.TVT_RADIUS_FT)[None, :], (v245.HEIGHT, v245.WIDTH)),
        np.broadcast_to((np.log1p(distance) / 10.0)[None, :], (v245.HEIGHT, v245.WIDTH)),
        np.broadcast_to(quality[None, :], (v245.HEIGHT, v245.WIDTH)),
    )).astype(np.float32)
    return np.concatenate((pf, xy), axis=0).astype(np.float32)


class FullDataset(Dataset):
    def __init__(self, records, xy_bank, training, difference_bank, cache_dir, channel_indices):
        self.records, self.xy_bank = records, xy_bank
        self.training, self.difference_bank, self.epoch = training, difference_bank, 0
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.channel_indices = np.asarray(channel_indices, dtype=np.int64)
        self.built = 0
    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return len(self.records)
    def get_extra(self, record):
        if record.well in self.memory_cache:
            heatmap, compact = self.memory_cache[record.well]
            profiles = np.broadcast_to(compact[:, None, :], (len(EXTRA_CHANNELS) - 1, v245.HEIGHT, v245.WIDTH))
            return np.concatenate((heatmap[None], profiles), axis=0)
        path = self.cache_dir / f"{record.well}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                if "heatmap" not in saved.files or "compact" not in saved.files:
                    path.unlink()
                    return self.get_extra(record)
                heatmap = saved["heatmap"].astype(np.float32)
                compact = saved["compact"].astype(np.float32)
        else:
            full = static_extra(record, self.xy_bank)
            heatmap = full[0].astype(np.float32)
            compact = full[1:, 0, :].astype(np.float32)
            np.savez_compressed(path, heatmap=heatmap.astype(np.float16), compact=compact.astype(np.float16))
            self.built += 1
            print("V247 FEATURE CACHE", {"built": self.built, "well": record.well, "path": str(path)}, flush=True)
        if heatmap.shape != (v245.HEIGHT, v245.WIDTH) or compact.shape != (len(EXTRA_CHANNELS) - 1, v245.WIDTH):
            if path.is_file():
                path.unlink()
                return self.get_extra(record)
            raise RuntimeError({"message": "V247 cached feature shape mismatch", "well": record.well, "heatmap": list(heatmap.shape), "compact": list(compact.shape)})
        self.memory_cache[record.well] = (heatmap, compact)
        profiles = np.broadcast_to(compact[:, None, :], (len(EXTRA_CHANNELS) - 1, v245.HEIGHT, v245.WIDTH))
        return np.concatenate((heatmap[None], profiles), axis=0)
    def __getitem__(self, index):
        rng = np.random.default_rng(v245.SEED + self.epoch * 100003 + index)
        record = self.records[index]
        panel = v245.make_panel(record, training=self.training, bank=self.difference_bank, rng=rng)
        panel["image"] = np.concatenate((panel["image"], self.get_extra(record)), axis=0).astype(np.float32)
        panel["image"] = panel["image"][self.channel_indices]
        return {key: torch.from_numpy(panel[key]) for key in ("image", "target_offset", "mask")}


def train_member(train_records, valid_records, fold, seed, branch):
    v245.seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("V247 requires CUDA")
    branch_indices = {
        "default": list(range(BASE_CHANNELS)),
        "pf": list(range(BASE_CHANNELS + PF_CHANNELS)),
        "xy": list(range(BASE_CHANNELS)) + list(range(BASE_CHANNELS + PF_CHANNELS, len(ALL_CHANNELS))),
    }
    if branch not in branch_indices:
        raise ValueError(f"Unknown V247 branch: {branch}")
    indices = branch_indices[branch]
    model = v245.ConvNeXtBNUNet(len(indices)).to(device)
    xy_bank = XYBank(train_records)
    donor_signature = hashlib.sha1("\n".join(sorted(record.well for record in train_records)).encode("utf-8")).hexdigest()[:12]
    cache_dir = OUT / "feature_cache" / f"fold{fold}_{donor_signature}"
    dataset = FullDataset(train_records, xy_bank, True, v245.DifferenceBlockBank(train_records), cache_dir, indices)
    valid_dataset = FullDataset(valid_records, xy_bank, False, None, cache_dir, indices)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=v245.RECIPE.learning_rate, weight_decay=v245.RECIPE.weight_decay)
    best = None
    for epoch in range(1, EPOCHS + 1):
        dataset.set_epoch(epoch); model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(loader, 1):
            image = batch["image"].to(device, non_blocking=True); target = batch["target_offset"].to(device, non_blocking=True); mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(image); loss, _ = v245.alignment_loss(logits, target, mask, image[:, 14])
            (loss / ACCUM).backward()
            if step % ACCUM == 0 or step == len(loader): optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        # Reuse the proven evaluator by decoding through the full panel.
        model.eval(); errors = []
        with torch.no_grad():
            for record in valid_records:
                panel = v245.make_panel(record, training=False)
                panel["image"] = np.concatenate((panel["image"], valid_dataset.get_extra(record)), axis=0).astype(np.float32)[indices]
                image = torch.from_numpy(panel["image"])[None].to(device)
                logits = model(image); offset = v245.decode_panel(logits)[0]
                rows = np.arange(record.anchor + 1, len(record.md)); pred = record.anchor_tvt + np.interp(record.md[rows], panel["md"], offset)
                errors.append(pred - record.tvt[rows])
        error = np.concatenate(errors); row = {"branch": branch, "fold": fold, "seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)), "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))}
        print("V247 EPOCH", row, flush=True)
        if best is None or row["rmse"] < best["rmse"]:
            best = row; OUT.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "channels": [ALL_CHANNELS[i] for i in indices], "branch": branch, "metrics": row}, OUT / "checkpoints" / f"{branch}_fold{fold}_seed{seed}.pt")
    return best


def main():
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); folds = v245.load_folds()
    manifest = {"experiment": "first_place_reconstructed_full_v247", "channels": list(ALL_CHANNELS), "branches": BRANCHES, "folds": FOLDS, "seeds": SEEDS, "epochs": EPOCHS, "max_train_wells": MAX_TRAIN_WELLS, "max_valid_wells": MAX_VALID_WELLS, "pf": "four-scale bootstrap PF with systematic resampling, 2D probability heatmap, and FFBSi trace", "xy": "complete outer-train truth weighted local plane with condition/residual quality gate", "private_winner_code": False, "writes_submission": False}
    print("V247 CONFIG", json.dumps(manifest), flush=True)
    reports = []
    for fold in FOLDS:
        train_ids = folds.loc[folds.fold.ne(fold), "well"].astype(str).tolist(); valid_ids = folds.loc[folds.fold.eq(fold), "well"].astype(str).tolist()
        rng = np.random.default_rng(24700 + fold)
        if MAX_TRAIN_WELLS and len(train_ids) > MAX_TRAIN_WELLS:
            train_ids = sorted(rng.choice(train_ids, MAX_TRAIN_WELLS, replace=False).tolist())
        if MAX_VALID_WELLS and len(valid_ids) > MAX_VALID_WELLS:
            valid_ids = sorted(rng.choice(valid_ids, MAX_VALID_WELLS, replace=False).tolist())
        print("V247 FOLD DATA", {"fold": fold, "train_wells": len(train_ids), "valid_wells": len(valid_ids)}, flush=True)
        train_records = [enrich(v245.load_record(w)) for w in train_ids]; valid_records = [enrich(v245.load_record(w)) for w in valid_ids]
        for branch in BRANCHES:
            for seed in SEEDS:
                reports.append(train_member(train_records, valid_records, fold, seed, branch))
    manifest.update({"members": len(reports), "model_dir": str(OUT / "checkpoints"), "elapsed_minutes": (time.time() - started) / 60.0})
    OUT.joinpath("verdict.json").write_text(json.dumps({**manifest, "reports": reports}, indent=2), encoding="utf-8")
    print("V247 FINAL VERDICT", json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__": main()
