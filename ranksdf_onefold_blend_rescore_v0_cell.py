"""Zero-training rescore of the RankSDF fold-1 checkpoint.

The standalone RankSDF v0 is weak but unusually decorrelated from strong.  This
cell reconstructs its held-out row predictions and evaluates fixed global
blends.  The preregistered primary candidate is:

    cp12 + clip(0.10 * (RankSDF - cp12), -6, 6)

The wider grid is diagnostic only; it must not override the primary gate on the
same fold.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
# Must match the hard-coded independent confirmation training cell.
VALID_FOLD = 3
RUN_DIR = OOF_ROOT / f"ranksdf_unet_onefold_v0_fold{VALID_FOLD}_s257_c512"
CACHE_DIR = OOF_ROOT / "ranksdf_panel_cache_v0_s257_c512"
SELECTED_MODEL_PATH = RUN_DIR / "models" / f"ranksdf_fold{VALID_FOLD}_selected.pt"
BEST_MODEL_PATH = RUN_DIR / "models" / f"ranksdf_fold{VALID_FOLD}_best.pt"
MODEL_PATH = SELECTED_MODEL_PATH if SELECTED_MODEL_PATH.exists() else BEST_MODEL_PATH
BUILD_REPORT_PATH = RUN_DIR / "panel_build_report.csv"
OUT_DIR = OOF_ROOT / f"ranksdf_onefold_blend_rescore_v0_fold{VALID_FOLD}"

BASE_CHANNELS = 16
TYPEWELL_VALID_CHANNEL = 7
WEIGHTS = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25)
APPLIED_CAPS = (3.0, 6.0, 12.0, 24.0, None)

PRIMARY_BASE = "cp12"
PRIMARY_WEIGHT = 0.10
PRIMARY_CAP = 6.0
PASS_PRIMARY_GAIN = 0.10
PASS_POSITIVE_WELL_FRACTION = 0.55


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class TinyRankUNet(nn.Module):
    def __init__(self, in_channels, base=16):
        super().__init__()
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 6)
        self.mid = ConvBlock(base * 6, base * 8)
        self.d4 = ConvBlock(base * 14, base * 6)
        self.d3 = ConvBlock(base * 10, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2)
        self.d1 = ConvBlock(base * 3, base)
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def up(x, skip):
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        x = self.mid(F.max_pool2d(e4, 2))
        x = self.d4(torch.cat([self.up(x, e4), e4], 1))
        x = self.d3(torch.cat([self.up(x, e3), e3], 1))
        x = self.d2(torch.cat([self.up(x, e2), e2], 1))
        x = self.d1(torch.cat([self.up(x, e1), e1], 1))
        return self.out(x).squeeze(1)


def _rmse(truth, prediction):
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))


def _cosine(error_a, error_b):
    denominator = np.sqrt(np.dot(error_a, error_a) * np.dot(error_b, error_b))
    return float(np.dot(error_a, error_b) / max(denominator, 1e-12))


def _score_variant(name, base_name, base, model_prediction, truth, wells, weight, cap):
    move = float(weight) * (model_prediction - base)
    if cap is not None:
        move = np.clip(move, -float(cap), float(cap))
    prediction = base + move
    base_rmse = _rmse(truth, base)
    prediction_rmse = _rmse(truth, prediction)
    rows = pd.DataFrame(
        {
            "well": wells,
            "truth": truth,
            "base": base,
            "prediction": prediction,
        }
    )
    per_well = []
    for well, frame in rows.groupby("well", sort=False):
        well_base = _rmse(frame["truth"], frame["base"])
        well_prediction = _rmse(frame["truth"], frame["prediction"])
        per_well.append(
            {
                "well": well,
                "rows": len(frame),
                "base_sse": float(np.sum((frame["base"] - frame["truth"]) ** 2)),
                "gain": well_base - well_prediction,
            }
        )
    per_well = pd.DataFrame(per_well)
    top_wells = set(per_well.nlargest(20, "base_sse")["well"])
    top = np.isin(wells, list(top_wells))
    return {
        "variant": name,
        "base": base_name,
        "weight_ranksdf": float(weight),
        "applied_cap_ft": np.nan if cap is None else float(cap),
        "base_rmse": base_rmse,
        "rmse": prediction_rmse,
        "gain": base_rmse - prediction_rmse,
        "positive_well_fraction": float(np.mean(per_well["gain"] > 0)),
        "median_well_gain": float(np.median(per_well["gain"])),
        "top20_gain": _rmse(truth[top], base[top]) - _rmse(truth[top], prediction[top]),
        "mean_abs_move": float(np.mean(np.abs(move))),
        "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
        "max_abs_move": float(np.max(np.abs(move))),
    }, prediction


for path in (MODEL_PATH, BUILD_REPORT_PATH):
    if not path.exists():
        raise FileNotFoundError(path)
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
channels = checkpoint.get("channels")
if not channels:
    raise RuntimeError("checkpoint has no channel manifest")
model = TinyRankUNet(len(channels), BASE_CHANNELS).to(DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

build_report = pd.read_csv(BUILD_REPORT_PATH)
valid_wells = build_report.loc[build_report["fold"] == VALID_FOLD, "well"].astype(str).tolist()
parts = []
with torch.no_grad():
    for well in tqdm(valid_wells, desc=f"reconstruct fold{VALID_FOLD} RankSDF"):
        path = CACHE_DIR / f"{well}.npz"
        with np.load(path, allow_pickle=False) as data:
            panel = data["panel"].astype(np.float32)
            offsets = data["offsets"].astype(np.float32)
            sample_md = data["sample_md"].astype(float)
            sample_z = data["sample_z"].astype(float)
            center_s = data["center_s"].astype(float)
            target_residual = data["target_residual"].astype(float)
            known = data["known_columns"].astype(bool)
            eval_md = data["eval_md"].astype(float)
            truth = data["eval_tvt"].astype(float)
            strong = data["eval_strong"].astype(float)
            cp12 = data["eval_cp12"].astype(float)
        tensor = torch.from_numpy(panel)[None].to(DEVICE)
        logits = model(tensor)
        valid = tensor[:, TYPEWELL_VALID_CHANNEL] > 0.5
        probability = torch.softmax(logits.masked_fill(~valid, -30.0), 1)
        offset_tensor = torch.from_numpy(offsets).to(DEVICE)
        predicted_offset = torch.sum(probability * offset_tensor[None, :, None], 1)[0].cpu().numpy()
        predicted_offset[known] = target_residual[known]
        sample_prediction = center_s + predicted_offset - sample_z
        prediction = np.interp(eval_md, sample_md, sample_prediction)
        parts.append(
            pd.DataFrame(
                {
                    "well": well,
                    "truth": truth,
                    "strong": strong,
                    "cp12": cp12,
                    "ranksdf": prediction,
                }
            )
        )

rows = pd.concat(parts, ignore_index=True)
truth = rows["truth"].to_numpy(float)
model_prediction = rows["ranksdf"].to_numpy(float)
wells = rows["well"].to_numpy(str)
strong = rows["strong"].to_numpy(float)
cp12 = rows["cp12"].to_numpy(float)

standalone = {
    "ranksdf_rmse": _rmse(truth, model_prediction),
    "strong_rmse": _rmse(truth, strong),
    "cp12_rmse": _rmse(truth, cp12),
    "cosine_vs_strong": _cosine(model_prediction - truth, strong - truth),
    "cosine_vs_cp12": _cosine(model_prediction - truth, cp12 - truth),
}

score_rows = []
prediction_map = {}
for base_name, base in (("strong", strong), ("cp12", cp12)):
    for weight in WEIGHTS:
        for cap in APPLIED_CAPS:
            cap_name = "full" if cap is None else str(int(cap))
            name = f"{base_name}_w{int(round(weight * 1000)):03d}_cap{cap_name}"
            score, prediction = _score_variant(
                name, base_name, base, model_prediction, truth, wells, weight, cap
            )
            score_rows.append(score)
            prediction_map[name] = prediction

score_report = pd.DataFrame(score_rows).sort_values(["base", "rmse"])
primary_name = f"{PRIMARY_BASE}_w{int(PRIMARY_WEIGHT * 1000):03d}_cap{int(PRIMARY_CAP)}"
primary = score_report[score_report["variant"] == primary_name]
if len(primary) != 1:
    raise RuntimeError(f"primary variant missing: {primary_name}")
primary = primary.iloc[0].to_dict()

# Closed-form uncapped optimum is diagnostic; it is not the gate configuration.
analytic = {}
for base_name, base in (("strong", strong), ("cp12", cp12)):
    base_error = base - truth
    model_error = model_prediction - truth
    a = float(np.mean((model_error - base_error) ** 2))
    b = float(-np.mean(base_error * (model_error - base_error)))
    weight = float(np.clip(b / max(a, 1e-12), 0.0, 1.0))
    prediction = (1.0 - weight) * base + weight * model_prediction
    analytic[base_name] = {
        "weight_ranksdf": weight,
        "predicted_rmse": _rmse(truth, prediction),
        "gain": _rmse(truth, base) - _rmse(truth, prediction),
    }

strict_pass = bool(
    primary["gain"] >= PASS_PRIMARY_GAIN
    and primary["positive_well_fraction"] >= PASS_POSITIVE_WELL_FRACTION
    and primary["top20_gain"] >= 0.0
)
verdict = {
    "experiment": "ranksdf_onefold_blend_rescore_v0",
    "valid_fold": VALID_FOLD,
    "model_path": str(MODEL_PATH),
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "standalone": standalone,
    "analytic_uncapped_optimum": analytic,
    "primary_variant": primary,
    "gates": {
        "gain_min": PASS_PRIMARY_GAIN,
        "positive_well_fraction_min": PASS_POSITIVE_WELL_FRACTION,
        "top20_gain_min": 0.0,
    },
    "strict_pass": strict_pass,
    "next": (
        "freeze primary blend and train remaining four folds"
        if strict_pass
        else "stop RankSDF v0; redesign inputs/training before more folds"
    ),
}

score_report.to_csv(OUT_DIR / "blend_scores.csv", index=False)
np.savez_compressed(
    OUT_DIR / f"fold{VALID_FOLD}_predictions.npz",
    truth=truth.astype(np.float32),
    strong=strong.astype(np.float32),
    cp12=cp12.astype(np.float32),
    ranksdf=model_prediction.astype(np.float32),
    primary=prediction_map[primary_name].astype(np.float32),
)
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("STANDALONE")
print(standalone)
print("\nBEST BY BASE")
print(score_report.groupby("base", sort=False).head(12).to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
