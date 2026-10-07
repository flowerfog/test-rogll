"""V221: physical-distance trajectory dynamics on the frozen V217 inputs.

Only new information is added: lagged slopes, scale-to-scale slope change, and
local-centre deviation for existing PF/beam/score/spatial trajectory channels.
All dynamics are computed on complete wells before the unchanged 1024-row
training sample is selected. Validation remains every official hidden row.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "likpf_residual_foldclean_spatial_v217.py"
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

source = SOURCE.read_text(encoding="utf-8")
source = source.replace(
    "from scipy.signal import savgol_filter",
    "from scipy.signal import savgol_filter\nfrom scipy.ndimage import uniform_filter1d",
    1,
)

feature_anchor = "features = base_features + spatial_features\n"
if source.count(feature_anchor) != 1:
    raise RuntimeError("V221 could not locate the V217 feature declaration")
dynamic_declaration = '''features = base_features + spatial_features

# Select genuinely path-valued channels. Static uncertainty, score, distance,
# calibration, and bias columns are intentionally excluded.
_preferred_channels = [
    "likpf_scale_3_d", "likpf_scale_5_d", "likpf_scale_8_d",
    "likpf_scale_12_d", "likpf_mean_d",
    "pf_ancc_delta", "pf_z_delta",
    "beam_cons_d", "beam_mean_d", "beam_med_d",
    "sc8_d", "sc15_d", "sc25_d", "sc_cons_d", "sc_ens_d", "hyb_d",
    "sig_mean_d", "form_mean_d",
    "tvt_dense_d", "tvt_densew_d", "tvt_dense50_d",
    "pf_vs_spatial", "pf_vs_dense", "spatial_vs_dense",
    "beam_vs_spatial", "sc_vs_beam",
]


def _is_dynamic_channel(name):
    lowered = name.lower()
    blocked = ("rmse", "_std", "dist", "bias", "trust", "score", "_sc", "unc", "entropy")
    if any(token in lowered for token in blocked):
        return False
    return (
        lowered.startswith(("beam_", "tvtf_", "tvtfw_", "tvtf50_"))
        or lowered in {
            "pf_ancc_delta", "pf_z_delta", "sc8_d", "sc15_d", "sc25_d",
            "sc_cons_d", "sc_ens_d", "hyb_d", "sig_mean_d", "form_mean_d",
            "tvt_dense_d", "tvt_densew_d", "tvt_dense50_d", "pf_vs_spatial",
            "pf_vs_dense", "spatial_vs_dense", "beam_vs_spatial", "sc_vs_beam",
        }
        or (lowered.startswith("likpf_") and lowered.endswith("_d"))
    )


trajectory_channels = []
for name in _preferred_channels + features:
    if name in features and name not in trajectory_channels and _is_dynamic_channel(name):
        trajectory_channels.append(name)
trajectory_channels = trajectory_channels[:36]
if len(trajectory_channels) < 16:
    raise RuntimeError(f"Only {len(trajectory_channels)} trajectory channels found")

DYNAMIC_LAGS = (32, 128, 512)
dynamic_features = []
for name in trajectory_channels:
    dynamic_features.extend([
        f"dyn__{name}__slope32",
        f"dyn__{name}__slope128",
        f"dyn__{name}__slope512",
        f"dyn__{name}__accel32_128",
        f"dyn__{name}__dev128",
    ])
features = features + dynamic_features
print({
    "trajectory_channels": trajectory_channels,
    "base_features": len(base_features),
    "spatial_features": len(spatial_features),
    "dynamic_features": len(dynamic_features),
    "total_features": len(features),
}, flush=True)
'''
source = source.replace(feature_anchor, dynamic_declaration, 1)

old_loader = '''def _load(wells, full):
    return pd.concat([_load_one(str(well), full) for well in wells], ignore_index=True, sort=False)
'''
new_loader = '''def _finite_values(values):
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return np.zeros(len(values), np.float64)
    index = np.arange(len(values))
    return np.interp(index, index[finite], values[finite])


def _lag_slope(values, md, lag):
    n = len(values)
    prior = np.maximum(np.arange(n) - lag, 0)
    denominator = md - md[prior]
    output = np.zeros(n, np.float64)
    valid = denominator > 1e-3
    output[valid] = (values[valid] - values[prior[valid]]) / denominator[valid]
    return output


def _add_dynamics(frame):
    frame = frame.sort_values("_row_idx", kind="stable").reset_index(drop=True)
    md = _finite_values(frame["md_since"].to_numpy(float))
    additions = {}
    for name in trajectory_channels:
        values = _finite_values(frame[name].to_numpy(float))
        slope32 = _lag_slope(values, md, 32)
        slope128 = _lag_slope(values, md, 128)
        slope512 = _lag_slope(values, md, 512)
        centre = uniform_filter1d(values, size=min(257, max(1, len(values))), mode="nearest")
        additions[f"dyn__{name}__slope32"] = slope32.astype(np.float32)
        additions[f"dyn__{name}__slope128"] = slope128.astype(np.float32)
        additions[f"dyn__{name}__slope512"] = slope512.astype(np.float32)
        additions[f"dyn__{name}__accel32_128"] = (slope32 - slope128).astype(np.float32)
        additions[f"dyn__{name}__dev128"] = (values - centre).astype(np.float32)
    return pd.concat([frame, pd.DataFrame(additions)], axis=1)


def _load(wells, full):
    frames = []
    for well in wells:
        # Dynamics must be computed before row sampling.
        frame = _add_dynamics(_load_one(str(well), True))
        if not full and len(frame) > 1024:
            positions = np.unique(
                np.round(np.linspace(0, len(frame) - 1, 1024)).astype(int)
            )
            frame = frame.iloc[positions].copy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)
'''
if source.count(old_loader) != 1:
    raise RuntimeError("V221 could not locate the audited V217 loader")
source = source.replace(old_loader, new_loader, 1)
source = source.replace('SEED = 217', 'SEED = 221', 1)
source = source.replace(
    'likpf_residual_foldclean_spatial_v217',
    'spatial_trajectory_dynamics_gbdt_v221',
)
source = source.replace('V217', 'V221').replace('v217', 'v221')

exec(compile(source, str(SOURCE), "exec"), globals())

# Exact paired comparison with the frozen V217 endpoint.
oof_root = ROOT / "oof_exports"
v217_path = oof_root / "likpf_residual_foldclean_spatial_v217" / "oof.parquet"
v221_path = oof_root / "spatial_trajectory_dynamics_gbdt_v221" / "oof.parquet"
for path in (v217_path, v221_path):
    if not path.is_file():
        raise FileNotFoundError(path)

v217 = pd.read_parquet(v217_path)[["id", "fold", "tvt_true", "candidate"]].rename(
    columns={"candidate": "v217"}
)
v221 = pd.read_parquet(v221_path)[["id", "candidate"]].rename(columns={"candidate": "v221"})
matched = v217.merge(v221, on="id", how="left", validate="one_to_one")
if len(matched) != 3_783_989 or matched["v221"].isna().any():
    raise RuntimeError("Incomplete V217/V221 alignment")


def _paired_rmse(group, column):
    error = group[column].to_numpy(float) - group["tvt_true"].to_numpy(float)
    return float(np.sqrt(np.mean(error * error)))


fold_rows = []
for held, group in matched.groupby("fold", sort=True):
    row = {
        "fold": int(held),
        "v217_rmse": _paired_rmse(group, "v217"),
        "v221_rmse": _paired_rmse(group, "v221"),
    }
    row["dynamic_gain"] = row["v217_rmse"] - row["v221_rmse"]
    fold_rows.append(row)
v217_rmse = _paired_rmse(matched, "v217")
v221_rmse = _paired_rmse(matched, "v221")
audit = {
    "experiment": "spatial_trajectory_dynamics_gbdt_v221_paired_audit",
    "changed_variable": "physical-distance multiscale dynamics for existing trajectory channels",
    "trajectory_channels": trajectory_channels,
    "dynamic_features": len(dynamic_features),
    "total_features": len(features),
    "v217_rmse": v217_rmse,
    "v221_rmse": v221_rmse,
    "dynamic_gain": v217_rmse - v221_rmse,
    "folds": fold_rows,
    "checks": {
        "dynamic_gain_ge_0p15": v217_rmse - v221_rmse >= 0.15,
        "positive_folds_ge_4": sum(row["dynamic_gain"] > 0 for row in fold_rows) >= 4,
        "minimum_fold_gain_ge_minus0p05": min(row["dynamic_gain"] for row in fold_rows) >= -0.05,
    },
}
audit["strict_pass"] = bool(all(audit["checks"].values()))
out = oof_root / "spatial_trajectory_dynamics_gbdt_v221"
pd.DataFrame(fold_rows).to_csv(out / "paired_v217_comparison.csv", index=False)
(out / "dynamic_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

print("\nV221 DYNAMIC FOLDS")
print(pd.DataFrame(fold_rows).to_string(index=False))
print("\nV221 DYNAMIC VERDICT")
print(json.dumps(audit, indent=2))
