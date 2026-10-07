"""Rescore the learned GR path selection as a shrunken move from strong.

The hard-selected trajectory was slightly worse than strong.  This cell tests
whether the learned sequence scorer nevertheless chose a useful correction
direction whose full amplitude was excessive.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
MATCHER_DIR = OOF_ROOT / "gr_path_matcher_onefold_v0"
SCORER_DIR = OOF_ROOT / "gr_path_sequence_scorer_onefold_v0"
BANK_PATH = MATCHER_DIR / "restricted_path_bank.npz"
MODEL_PATH = SCORER_DIR / "best_sequence_scorer.pt"
CACHE_DIR = SCORER_DIR / "sequence_cache"
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
OUT_DIR = OOF_ROOT / "gr_path_sequence_shrink_rescore_v0"

VALID_FOLD = 1
WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.0)
APPLIED_CAPS_FT = (2.0, 4.0, 6.0, 8.0, 12.0, np.inf)
PASS_GAIN = 0.10
PASS_POSITIVE_WELLS = 0.52


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels),
        )
    def forward(self, values):
        return F.silu(values + self.net(values))


class PathSequenceScorer(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(inputs, 48, 7, padding=3, bias=False),
            nn.GroupNorm(8, 48), nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(48, 1), ResidualBlock(48, 2),
            ResidualBlock(48, 4), ResidualBlock(48, 8),
        )
        self.head = nn.Sequential(
            nn.Linear(96, 64), nn.SiLU(), nn.Dropout(0.10), nn.Linear(64, 1)
        )
    def forward(self, values):
        hidden = self.blocks(self.stem(values))
        return self.head(torch.cat([hidden.mean(-1), hidden.amax(-1)], 1)).squeeze(1)


def _rmse(truth, prediction):
    return float(np.sqrt(np.mean((np.asarray(truth, float) - np.asarray(prediction, float)) ** 2)))


for path in (BANK_PATH, MODEL_PATH, CACHE_DIR, DATA_ROOT / "train"):
    if not path.exists():
        raise FileNotFoundError(path)
pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("loading bank/checkpoint", flush=True)

with np.load(BANK_PATH, allow_pickle=False) as bank:
    templates = bank["templates"].astype(np.float32)
    alphas = bank["length_alphas"].astype(np.float32)
    names = bank["names"].astype(str)
    progress_grid = bank["progress_grid"].astype(np.float32)
    median_length = float(bank["median_length"][0])
    bank_cap = float(bank["cap"][0])

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
channels = list(checkpoint["channels"])
model = PathSequenceScorer(len(channels))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

print("loading candidate pool:", pool_path, flush=True)
header = pd.read_csv(pool_path, nrows=0).columns.tolist()
usecols = [column for column in ("id", "well", "row_idx", "tvt_true", "strong") if column in header]
pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pool["row_idx"].astype(np.int32)

# Cache contains both folds 1 and 5. Fold-1 membership is recovered from the
# original path features rather than inferred from filenames.
print("loading fold membership", flush=True)
path_features = pd.read_pickle(MATCHER_DIR / "path_features.pkl")
fold_map = path_features.groupby("well", sort=False)["fold"].first().astype(int).to_dict()
pool_wells = set(pool["well"])
valid_wells = sorted(
    path.stem
    for path in CACHE_DIR.glob("*.npz")
    if path.stem in pool_wells and fold_map.get(path.stem) == VALID_FOLD
)
pool = pool[pool["well"].isin(valid_wells)].copy()
pool_groups = {
    str(well): frame.sort_values("row_idx").copy()
    for well, frame in pool.groupby("well", sort=False)
}
print(
    "valid fold:",
    {"wells": len(valid_wells), "rows": len(pool), "candidates": len(names)},
    flush=True,
)

started = time.time()
parts = []
well_choices = []
with torch.no_grad():
    for well_number, well in enumerate(valid_wells, 1):
        with np.load(CACHE_DIR / f"{well}.npz", allow_pickle=False) as data:
            sequence = torch.from_numpy(data["sequence"].astype(np.float32))
        scores = model(sequence).cpu().numpy()
        order = np.argsort(scores)[::-1]
        selected = int(order[0])
        zero_index = int(np.flatnonzero(names == "zero")[0])
        score_margin_zero = float(scores[selected] - scores[zero_index])
        score_margin_second = float(scores[order[0]] - scores[order[1]])

        frame = pool_groups[well]
        horizontal = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv", usecols=["MD"]
        )
        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
        rows = frame["row_idx"].to_numpy(int)
        anchor = int(rows.min() - 1)
        eval_span = float(max(md[rows.max()] - md[anchor], 1.0))
        progress = np.clip((md[rows] - md[anchor]) / eval_span, 0.0, 1.0)
        length_ratio = max(eval_span / max(median_length, 1.0), 0.1)
        correction = np.interp(progress, progress_grid, templates[selected])
        correction *= length_ratio ** float(alphas[selected])
        correction = np.clip(correction, -bank_cap, bank_cap)
        frame["move"] = correction.astype(np.float32)
        parts.append(frame[["id", "well", "row_idx", "tvt_true", "strong", "move"]])
        well_choices.append(
            {
                "well": well,
                "selected_index": selected,
                "selected_name": str(names[selected]),
                "score_margin_zero": score_margin_zero,
                "score_margin_second": score_margin_second,
                "mean_abs_move": float(np.mean(np.abs(correction))),
                "p95_abs_move": float(np.quantile(np.abs(correction), 0.95)),
            }
        )
        if well_number % 25 == 0 or well_number == len(valid_wells):
            print(
                f"rescore {well_number}/{len(valid_wells)} "
                f"elapsed={(time.time() - started) / 60.0:.1f} min",
                flush=True,
            )

print("building shrink/cap scan", flush=True)
rows = pd.concat(parts, ignore_index=True)
choices = pd.DataFrame(well_choices)
truth = rows["tvt_true"].to_numpy(float)
strong = rows["strong"].to_numpy(float)
move = rows["move"].to_numpy(float)
error = strong - truth
move_energy = float(np.mean(move * move))
error_move = float(np.mean(error * move))
analytic_weight = float(-error_move / max(move_energy, 1e-12))
analytic_weight_clipped = float(np.clip(analytic_weight, 0.0, 1.0))

score_rows = []
bywell_rows = []
row_groups = {
    str(well): np.asarray(list(indices), dtype=np.int64)
    for well, indices in rows.groupby("well", sort=False).groups.items()
}
for weight in WEIGHTS:
    for cap in APPLIED_CAPS_FT:
        applied = weight * move
        if np.isfinite(cap):
            applied = np.clip(applied, -cap, cap)
        prediction = strong + applied
        name = f"w{weight:.2f}_cap{'full' if not np.isfinite(cap) else f'{cap:.0f}'}"
        perwell = []
        for well, indices in row_groups.items():
            base_rmse = _rmse(truth[indices], strong[indices])
            candidate_rmse = _rmse(truth[indices], prediction[indices])
            perwell.append(base_rmse - candidate_rmse)
            bywell_rows.append(
                {
                    "variant": name,
                    "well": well,
                    "strong_rmse": base_rmse,
                    "rmse": candidate_rmse,
                    "gain": base_rmse - candidate_rmse,
                }
            )
        score_rows.append(
            {
                "variant": name,
                "weight": weight,
                "cap_ft": cap,
                "strong_rmse": _rmse(truth, strong),
                "rmse": _rmse(truth, prediction),
                "gain_vs_strong": _rmse(truth, strong) - _rmse(truth, prediction),
                "mean_well_gain": float(np.mean(perwell)),
                "median_well_gain": float(np.median(perwell)),
                "positive_well_fraction": float(np.mean(np.asarray(perwell) > 0)),
                "mean_abs_applied_move": float(np.mean(np.abs(applied))),
                "p95_abs_applied_move": float(np.quantile(np.abs(applied), 0.95)),
            }
        )
    print(f"scan weight={weight:.2f} complete", flush=True)

scores = pd.DataFrame(score_rows).sort_values("rmse").reset_index(drop=True)
bywell = pd.DataFrame(bywell_rows)
scores.to_csv(OUT_DIR / "weight_cap_scan.csv", index=False)
bywell.to_csv(OUT_DIR / "weight_cap_bywell.csv", index=False)
choices.to_csv(OUT_DIR / "selected_paths.csv", index=False)
rows.to_pickle(OUT_DIR / "selected_move_oof.pkl")

nonzero = scores[scores["weight"] > 0]
best = nonzero.iloc[0].to_dict()
best = {
    key: (
        "full"
        if key == "cap_ft" and not np.isfinite(value)
        else value.item() if isinstance(value, np.generic) else value
    )
    for key, value in best.items()
}
strict_pass = bool(
    analytic_weight > 0
    and best["gain_vs_strong"] >= PASS_GAIN
    and best["positive_well_fraction"] >= PASS_POSITIVE_WELLS
)
verdict = {
    "experiment": "gr_path_sequence_shrink_rescore_v0",
    "source": str(SCORER_DIR),
    "valid_wells": len(valid_wells),
    "move_rms": float(np.sqrt(move_energy)),
    "error_move_inner_product": error_move,
    "analytic_weight": analytic_weight,
    "analytic_weight_clipped_0_1": analytic_weight_clipped,
    "best_scanned_nonzero": best,
    "gates": {
        "analytic_weight_positive": bool(analytic_weight > 0),
        "gain_vs_strong_min": PASS_GAIN,
        "positive_well_fraction_min": PASS_POSITIVE_WELLS,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> train the sequence scorer on cross-fit pseudo-PS states and validate fixed shrink"
        if strict_pass
        else "FAIL -> learned sequence scorer chose an unusable direction; close own-typewell GR path ranking"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nWEIGHT/CAP SCAN")
print(scores.head(30).to_string(index=False))
print("\nMARGIN DIAGNOSTICS")
print(choices[["score_margin_zero", "score_margin_second", "mean_abs_move", "p95_abs_move"]].describe().to_string())
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
