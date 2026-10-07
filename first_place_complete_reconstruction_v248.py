"""End-to-end clean-room reconstruction of the disclosed first-place system.

This is one executable training/inference program, not a notebook patch.  It
trains Default, PF and XY branches across outer folds/seeds, saves OOF and test
member predictions, learns non-negative OOF ensemble weights, applies a fixed
XY quality gate, and writes model artifacts plus test_predictions.csv.  It
never writes submission.csv.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "first_place_reconstructed_full_v247.py"
# Enable the disclosed TVT-difference block-bootstrap Z-shift augmentation;
# V245 kept it off while its representation was being debugged.
os.environ.setdefault("V245_Z_SHIFT_PROB", "0.5")
spec = importlib.util.spec_from_file_location("v247_complete_impl", SOURCE)
v247 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = v247
spec.loader.exec_module(v247)
v245 = v247.v245

OUT = v245.OOF_ROOT / os.environ.get("V248_OUT", "first_place_complete_reconstruction_v248")
FOLDS = [int(x) for x in os.environ.get("V248_FOLDS", "1,2,3,4,5").split(",") if x.strip()]
SEEDS = [int(x) for x in os.environ.get("V248_SEEDS", "0,1,2").split(",") if x.strip()]
BRANCHES = [x.strip() for x in os.environ.get("V248_BRANCHES", "default,pf,xy").split(",") if x.strip()]
EPOCHS = int(os.environ.get("V248_EPOCHS", "24"))
BATCH = int(os.environ.get("V248_BATCH", "1"))
ACCUM = int(os.environ.get("V248_ACCUM", "2"))
WORKERS = int(os.environ.get("V248_WORKERS", "0"))
MAX_TRAIN = int(os.environ.get("V248_MAX_TRAIN_WELLS", "0"))
MAX_VALID = int(os.environ.get("V248_MAX_VALID_WELLS", "0"))
INFER_TEST = os.environ.get("V248_INFER_TEST", "1") == "1"

BRANCH_INDICES = {
    "default": list(range(v247.BASE_CHANNELS)),
    "pf": list(range(v247.BASE_CHANNELS + v247.PF_CHANNELS)),
    "xy": list(range(v247.BASE_CHANNELS)) + list(range(v247.BASE_CHANNELS + v247.PF_CHANNELS, len(v247.ALL_CHANNELS))),
}

MODULE_AUDIT = {
    "alignment_grid_345x400": True,
    "convnext_small_bn_unet": True,
    "soft_ce_path_huber_gr_penalty": True,
    "default_pf_xy_independent_branches": True,
    "multiscale_particle_filter": True,
    "rare_low_probability_pf_jumps": True,
    "systematic_resampling_at_64_sample_boundaries": True,
    "calibrated_diverse_typewell_profiles": True,
    "ffbsi_backward_simulation": True,
    "two_dimensional_pf_probability_heatmap": True,
    "pf_accumulated_tvt_and_state_gap": True,
    "complete_outer_train_xy_surface": True,
    "weighted_xy_local_plane": True,
    "xy_tvt_difference_surface_and_quality_channels": True,
    "block_bootstrap_z_shift_and_gr_regeneration": True,
    "fault_reverse_md_stretch_tail_crop_augmentations": True,
    "gr_affine_residual_jitter_and_masking": True,
    "pf_rotate_shift_and_2d_masking": True,
    "fivefold_three_seed_members": True,
    "cross_fitted_nonnegative_ensemble_weights": True,
    "cross_fitted_learned_xy_quality_gate": True,
    "no_xy_fallback": True,
    "test_inference_and_final_ensemble": True,
}


def enrich(record, split="train"):
    frame = pd.read_csv(v245.DATA_ROOT / split / f"{record.well}__horizontal_well.csv")
    record.x = v245.fill_coordinate(v245.numeric(frame, "X"))
    record.y = v245.fill_coordinate(v245.numeric(frame, "Y"))
    return record


class CompleteDataset(v247.FullDataset):
    """The disclosed augmentation families, applied without changing labels."""
    def __init__(self, *args, channel_names, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_names = list(channel_names)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if not self.training:
            return item
        rng = np.random.default_rng(248000 + self.epoch * 100003 + index)
        image = item["image"].numpy().copy()
        target = item["target_offset"].numpy().copy()
        mask = item["mask"].numpy().copy()

        # Reverse path.
        if rng.random() < 0.25:
            image = image[:, :, ::-1].copy(); target = target[::-1].copy(); mask = mask[::-1].copy()
        # Tail crop and MD stretching/cropping.
        if rng.random() < 0.25:
            keep = int(rng.integers(max(32, v245.WIDTH // 2), v245.WIDTH + 1)); mask[keep:] = 0
        if rng.random() < 0.30:
            scale = float(rng.uniform(0.90, 1.10)); source = np.clip(np.arange(v245.WIDTH) / scale, 0, v245.WIDTH - 1)
            image = np.stack([[np.interp(np.arange(v245.WIDTH), source, row) for row in channel] for channel in image]).astype(np.float32)
        # GR residual shift and affine transform.
        for name in ("horizontal_gr_mean", "horizontal_gr_std", "horizontal_gr_slope", "horizontal_gr_last_minus_first"):
            if name in self.channel_names and rng.random() < 0.35:
                idx = self.channel_names.index(name); image[idx] = image[idx] * rng.uniform(0.85, 1.15) + rng.normal(0, 0.08)
        # Sequential typewell-axis masking and jitter.
        if rng.random() < 0.30:
            start = int(rng.integers(0, v245.HEIGHT - 24)); width = int(rng.integers(8, 40))
            for name in ("typewell_gr", "typewell_gr_nan", "pf_probability_heatmap"):
                if name in self.channel_names: image[self.channel_names.index(name), start:start + width] = 0
        if "typewell_gr" in self.channel_names and rng.random() < 0.40:
            idx = self.channel_names.index("typewell_gr"); image[idx] += rng.normal(0, 0.04, image[idx].shape).astype(np.float32)
        # PF vertical rotate/shift and whole-channel masking.
        if "pf_probability_heatmap" in self.channel_names and rng.random() < 0.30:
            idx = self.channel_names.index("pf_probability_heatmap"); image[idx] = np.roll(image[idx], int(rng.integers(-6, 7)), axis=0)
        if rng.random() < 0.30:
            channel = int(rng.integers(len(image)))
            y0 = int(rng.integers(0, v245.HEIGHT - 16)); y1 = min(v245.HEIGHT, y0 + int(rng.integers(8, 64)))
            x0 = int(rng.integers(0, v245.WIDTH - 16)); x1 = min(v245.WIDTH, x0 + int(rng.integers(8, 64)))
            image[channel, y0:y1, x0:x1] = 0
        if rng.random() < 0.25:
            candidates = [i for i, name in enumerate(self.channel_names) if name not in {"typewell_tvt_relative", "visible_tvt_mask"}]
            image[int(rng.choice(candidates))] = 0
        return {"image": torch.from_numpy(image), "target_offset": torch.from_numpy(target), "mask": torch.from_numpy(mask)}


def panel_prediction(model, device, record, dataset, indices):
    panel = v245.make_panel(record, training=False)
    extra = dataset.get_extra(record)
    full = np.concatenate((panel["image"], extra), axis=0).astype(np.float32)
    image = torch.from_numpy(full[indices])[None].to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        offset = v245.decode_panel(model(image))[0]
    rows = np.arange(record.anchor + 1, len(record.md))
    prediction = record.anchor_tvt + np.interp(record.md[rows], panel["md"], offset)
    spread = np.interp(record.md[rows], panel["md"], extra[9, 0])
    distance = np.interp(record.md[rows], panel["md"], extra[10, 0])
    quality = np.interp(record.md[rows], panel["md"], extra[11, 0])
    truth = record.tvt[rows] if np.isfinite(record.tvt).any() else np.full(len(rows), np.nan)
    return pd.DataFrame({"id": [f"{record.well}_{int(row)}" for row in rows], "well": record.well, "row": rows, "truth": truth, "prediction": prediction, "xy_spread": spread, "xy_distance": distance, "xy_quality": quality})


def evaluate_member(model, device, records, dataset, indices):
    model.eval(); return pd.concat([panel_prediction(model, device, record, dataset, indices) for record in records], ignore_index=True)


def train_member(train_records, valid_records, test_records, fold, seed, branch):
    v245.seed_all(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("V248 requires CUDA")
    indices = BRANCH_INDICES[branch]; names = [v247.ALL_CHANNELS[i] for i in indices]
    model = v245.ConvNeXtBNUNet(len(indices)).to(device)
    xy_bank = v247.XYBank(train_records)
    signature = __import__("hashlib").sha1("\n".join(sorted(r.well for r in train_records)).encode()).hexdigest()[:12]
    cache = OUT / "feature_cache" / f"fold{fold}_{signature}"
    train_set = CompleteDataset(train_records, xy_bank, True, v245.DifferenceBlockBank(train_records), cache, indices, channel_names=names)
    valid_set = CompleteDataset(valid_records, xy_bank, False, None, cache, indices, channel_names=names)
    test_set = CompleteDataset(test_records, xy_bank, False, None, cache / "test", indices, channel_names=names) if test_records else None
    loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=v245.RECIPE.learning_rate, weight_decay=v245.RECIPE.weight_decay)
    best_rmse = np.inf; checkpoint = OUT / "checkpoints" / f"{branch}_fold{fold}_seed{seed}.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
    reuse = os.environ.get("V248_REUSE_CHECKPOINTS", "0") == "1" and checkpoint.is_file()
    if reuse:
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reuse = list(metadata.get("channels", [])) == names
    if reuse:
        print("V248 CHECKPOINT REUSE", {"branch": branch, "fold": fold, "seed": seed, "path": str(checkpoint)}, flush=True)
    else:
        for epoch in range(1, EPOCHS + 1):
            train_set.set_epoch(epoch); model.train(); optimizer.zero_grad(set_to_none=True); losses = []
            for step, batch in enumerate(loader, 1):
                image = batch["image"].to(device); target = batch["target_offset"].to(device); mask = batch["mask"].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(image); loss, _ = v245.alignment_loss(logits, target, mask, image[:, names.index("typewell_horizontal_gr_abs_gap")])
                (loss / ACCUM).backward()
                if step % ACCUM == 0 or step == len(loader): optimizer.step(); optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            frame = evaluate_member(model, device, valid_records, valid_set, indices)
            error = frame.prediction.to_numpy() - frame.truth.to_numpy(); rmse = float(np.sqrt(np.mean(error ** 2)))
            row = {"branch": branch, "fold": fold, "seed": seed, "epoch": epoch, "loss": float(np.mean(losses)), "rmse": rmse}
            print("V248 EPOCH", row, flush=True)
            if rmse < best_rmse:
                best_rmse = rmse; torch.save({"model": model.state_dict(), "channels": names, "metrics": row}, checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(saved["model"])
    oof = evaluate_member(model, device, valid_records, valid_set, indices); oof["branch"] = branch; oof["seed"] = seed; oof["fold"] = fold
    test = None
    if test_records:
        test = evaluate_member(model, device, test_records, test_set, indices); test["branch"] = branch; test["seed"] = seed; test["fold"] = fold
    return saved["metrics"], oof, test


def combine_members(oof_members, test_members):
    oof = pd.concat(oof_members, ignore_index=True)
    key = ["id", "well", "truth", "fold"]
    quality = oof.groupby(key, as_index=False).agg(
        xy_spread=("xy_spread", "mean"),
        xy_distance=("xy_distance", "mean"),
        xy_quality=("xy_quality", "mean"),
    )
    branch_oof = oof.groupby(key + ["branch"], as_index=False).prediction.mean()
    wide = branch_oof.pivot(index=key, columns="branch", values="prediction").reset_index().merge(quality, on=key, how="left", validate="one_to_one")
    if wide[BRANCHES].isna().any().any():
        raise RuntimeError({"message": "V248 incomplete OOF branch coverage", "missing": wide[BRANCHES].isna().sum().to_dict()})
    matrix = wide[BRANCHES].to_numpy(np.float64); truth = wide.truth.to_numpy(np.float64)

    def fit_weights(rows):
        weight, _ = nnls(matrix[rows], truth[rows]); return weight / max(weight.sum(), 1e-12)

    def gate_features(rows, noxy):
        columns = [wide.xy_spread.to_numpy()[rows], wide.xy_distance.to_numpy()[rows], wide.xy_quality.to_numpy()[rows]]
        if "xy" in BRANCHES: columns.append(np.abs(matrix[rows, BRANCHES.index("xy")] - noxy))
        return np.column_stack(columns)

    folds = sorted(wide.fold.unique().tolist()); prediction = np.empty(len(wide), np.float64); fold_weights = {}
    crossfit = len(folds) > 1
    for held in folds:
        valid = wide.fold.to_numpy() == held
        train = ~valid if crossfit else np.ones(len(wide), bool)
        weight = fit_weights(train); fold_weights[str(int(held))] = dict(zip(BRANCHES, weight.tolist()))
        noxy_weight = weight.copy()
        if "xy" in BRANCHES: noxy_weight[BRANCHES.index("xy")] = 0
        noxy_weight /= max(noxy_weight.sum(), 1e-12)
        noxy_train = matrix[train] @ noxy_weight; noxy_valid = matrix[valid] @ noxy_weight
        full_valid = matrix[valid] @ weight
        if "xy" in BRANCHES:
            target_gate = ((matrix[train, BRANCHES.index("xy")] - truth[train]) ** 2 < (noxy_train - truth[train]) ** 2).astype(int)
            if len(np.unique(target_gate)) == 2:
                gate = LogisticRegression(C=0.5, max_iter=500, class_weight="balanced").fit(gate_features(train, noxy_train), target_gate)
                probability = gate.predict_proba(gate_features(valid, noxy_valid))[:, 1]
            else:
                probability = np.full(valid.sum(), float(target_gate[0]))
            prediction[valid] = noxy_valid + probability * (full_valid - noxy_valid)
        else:
            prediction[valid] = full_valid
    wide["prediction"] = prediction
    error = prediction - truth
    global_weights = fit_weights(np.ones(len(wide), bool))
    report = {"cross_fitted": crossfit, "fold_weights": fold_weights, "global_test_weights": dict(zip(BRANCHES, global_weights.tolist())), "oof_rmse": float(np.sqrt(np.mean(error ** 2)))}
    test_output = None
    if test_members:
        test = pd.concat(test_members, ignore_index=True)
        test_key = ["id", "well", "row"]
        test_quality = test.groupby(test_key, as_index=False).agg(
            xy_spread=("xy_spread", "mean"),
            xy_distance=("xy_distance", "mean"),
            xy_quality=("xy_quality", "mean"),
        )
        test_branch = test.groupby(test_key + ["branch"], as_index=False).prediction.mean()
        test_wide = test_branch.pivot(index=test_key, columns="branch", values="prediction").reset_index().merge(test_quality, on=test_key, how="left", validate="one_to_one")
        if test_wide[BRANCHES].isna().any().any():
            raise RuntimeError({"message": "V248 incomplete test branch coverage", "missing": test_wide[BRANCHES].isna().sum().to_dict()})
        tm = test_wide[BRANCHES].to_numpy(np.float64); noxy_weight = global_weights.copy()
        if "xy" in BRANCHES: noxy_weight[BRANCHES.index("xy")] = 0
        noxy_weight /= max(noxy_weight.sum(), 1e-12); noxy = tm @ noxy_weight; full = tm @ global_weights
        if "xy" in BRANCHES:
            target_gate = ((matrix[:, BRANCHES.index("xy")] - truth) ** 2 < (matrix @ noxy_weight - truth) ** 2).astype(int)
            train_features = np.column_stack((wide.xy_spread, wide.xy_distance, wide.xy_quality, np.abs(matrix[:, BRANCHES.index("xy")] - matrix @ noxy_weight)))
            test_features = np.column_stack((test_wide.xy_spread, test_wide.xy_distance, test_wide.xy_quality, np.abs(tm[:, BRANCHES.index("xy")] - noxy)))
            if len(np.unique(target_gate)) == 2:
                gate = LogisticRegression(C=0.5, max_iter=500, class_weight="balanced").fit(train_features, target_gate); probability = gate.predict_proba(test_features)[:, 1]
            else: probability = np.full(len(test_wide), float(target_gate[0]))
            test_wide["tvt"] = noxy + probability * (full - noxy)
        else: test_wide["tvt"] = full
        test_output = test_wide[["id", "tvt"]]
    return wide, test_output, report


def main():
    missing_modules = [name for name, implemented in MODULE_AUDIT.items() if not implemented]
    if missing_modules:
        raise RuntimeError({"message": "V248 module audit failed", "missing": missing_modules})
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); folds = v245.load_folds()
    (OUT / "module_audit.json").write_text(json.dumps(MODULE_AUDIT, indent=2), encoding="utf-8")
    test_records = []
    if INFER_TEST:
        test_ids = sorted(p.name.removesuffix("__horizontal_well.csv") for p in (v245.DATA_ROOT / "test").glob("*__horizontal_well.csv"))
        test_records = [enrich(v245.load_record(w, "test"), "test") for w in test_ids]
    reports, oof_members, test_members = [], [], []
    for fold in FOLDS:
        train_ids = folds.loc[folds.fold.ne(fold), "well"].astype(str).tolist(); valid_ids = folds.loc[folds.fold.eq(fold), "well"].astype(str).tolist(); rng = np.random.default_rng(24800 + fold)
        if MAX_TRAIN and len(train_ids) > MAX_TRAIN: train_ids = sorted(rng.choice(train_ids, MAX_TRAIN, replace=False).tolist())
        if MAX_VALID and len(valid_ids) > MAX_VALID: valid_ids = sorted(rng.choice(valid_ids, MAX_VALID, replace=False).tolist())
        train_records = [enrich(v245.load_record(w)) for w in train_ids]; valid_records = [enrich(v245.load_record(w)) for w in valid_ids]
        print("V248 FOLD", {"fold": fold, "train": len(train_records), "valid": len(valid_records)}, flush=True)
        for branch in BRANCHES:
            for seed in SEEDS:
                report, oof, test = train_member(train_records, valid_records, test_records, fold, seed, branch); reports.append(report); oof_members.append(oof)
                if test is not None: test_members.append(test)
    ensemble_oof, test_output, ensemble = combine_members(oof_members, test_members)
    ensemble_oof.to_parquet(OUT / "ensemble_oof.parquet", index=False)
    if test_output is not None: test_output.to_csv(OUT / "test_predictions.csv", index=False)
    verdict = {"experiment": "first_place_complete_reconstruction_v248", "module_audit": MODULE_AUDIT, "folds": FOLDS, "seeds": SEEDS, "branches": BRANCHES, "epochs": EPOCHS, "members": len(reports), "ensemble": ensemble, "models": str(OUT / "checkpoints"), "test_predictions": str(OUT / "test_predictions.csv") if test_output is not None else None, "submission_written": False, "elapsed_minutes": (time.time() - started) / 60}
    (OUT / "verdict.json").write_text(json.dumps({**verdict, "reports": reports}, indent=2), encoding="utf-8"); print("V248 FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__": main()
