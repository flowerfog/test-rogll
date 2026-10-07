"""Fast post-training entropy-weighted TTA audit for V239."""

import json
import os
from pathlib import Path

import numpy as np
import torch

import ninth_place_wholewell_unet_v239 as v


ROOT = Path(os.environ.get("ROGII_ROOT", "/root/autodl-tmp"))
OUT = ROOT / "oof_exports" / "ninth_place_wholewell_unet_v239"
CHECKPOINT = OUT / "fold1_full_seed239.pt"
TEMPERATURE = float(os.environ.get("V239_TTA_TEMPERATURE", "10"))


@torch.no_grad()
def predict_entropy_tta(model, record, device):
    shifts = (-18.0, -9.0, 0.0, 9.0, 18.0)
    views = []
    entropies = []
    for shift in shifts:
        for flip in (False, True):
            panel = v.compose_panel(record, shift, flip)
            values = torch.from_numpy(panel["input"])[None].to(device)
            valid = torch.from_numpy(panel["state_valid"])[None, :, :1].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(values).float().masked_fill(~valid, -30.0)
            probability = torch.softmax(logits, dim=1)
            entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=1).mean()
            offsets = torch.as_tensor(v.OFFSETS, device=device)[None, :, None]
            expected = (probability * offsets).sum(dim=1)[0].cpu().numpy()
            if flip:
                expected = expected[::-1].copy()
            views.append((shift, record.anchor_tvt + shift + expected))
            entropies.append(float(entropy.cpu()))
    weights = np.exp(-(np.asarray(entropies) - np.min(entropies)) / TEMPERATURE)
    weights /= np.sum(weights)
    columns_tvt = np.sum(
        np.asarray([prediction for _, prediction in views]) * weights[:, None], axis=0
    )
    columns_md = v.sample_md(record)
    columns_z = np.interp(columns_md, record.md, record.z)
    eval_rows = np.arange(record.anchor + 1, len(record.md))
    surface = columns_tvt + columns_z
    prediction = np.interp(record.md[eval_rows], columns_md, surface) - record.z[eval_rows]
    return eval_rows, prediction, float(np.sum(weights * np.asarray(entropies))), float(np.max(weights))


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    fold, fold_path = v.load_fold_map()
    wells = fold.loc[fold.fold.eq(1), "well"].astype(str).tolist()
    records = [v.load_record(well) for well in wells]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TTA audit requires CUDA")
    model = v.ResNet34UNet().to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    errors, entropy_rows = [], []
    for record in records:
        rows, prediction, entropy, max_weight = predict_entropy_tta(model, record, device)
        truth = record.truth[rows]
        finite = np.isfinite(truth) & np.isfinite(prediction)
        errors.append(prediction[finite] - truth[finite])
        entropy_rows.append({"well": record.well, "entropy": entropy, "max_view_weight": max_weight})
    error = np.concatenate(errors)
    result = {
        "experiment": "ninth_place_entropy_tta_eval_v239",
        "checkpoint": str(CHECKPOINT),
        "fold": 1,
        "temperature": TEMPERATURE,
        "rows": int(len(error)),
        "wells": len(records),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "base_rmse": float(checkpoint["metrics"]["rmse"]),
        "gain": float(checkpoint["metrics"]["rmse"] - np.sqrt(np.mean(error**2))),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "entropy_tta_verdict.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("V239 ENTROPY TTA", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
