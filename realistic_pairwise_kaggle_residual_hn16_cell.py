"""Final Kaggle tail cell: convert frozen HN13 score_mean into HN15 residual.

Run after the exact 6.377 mainline and the existing HN13 cell.  HN13 performs
the expensive candidate inference.  This cell reconstructs its pre-registered score_mean
endpoint from the frozen 0.275 blend, applies the independently validated HN15
residual model, and overwrites submission.csv.
"""

from pathlib import Path
import hashlib
import json
import os
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED_HN16 = time.time()
WORK_HN16 = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
PARENT_HN16 = WORK_HN16 / "submission_6p377_before_hn13.csv"
HN13_BLEND_HN16 = WORK_HN16 / "submission_hn13_score_mean_w0275.csv"
OUTPUT_HN16 = WORK_HN16 / "submission_hn16_score_mean_residual.csv"
FINAL_HN16 = WORK_HN16 / "submission.csv"
HN13_WEIGHT_HN16 = 0.275
CAP_HN16 = 8.0


def _find_data_root_hn16():
    candidates = []
    configured = os.environ.get("ROGII_DATA_ROOT")
    if configured:
        candidates.append(Path(configured))
    for name in ("COMPETITION_DATA_ROOT", "DATA_ROOT", "DATA_DIR"):
        value = globals().get(name)
        if value:
            candidates.append(Path(value))
    runtime_cfg = globals().get("CFG")
    if runtime_cfg is not None:
        for name in ("dataset_path", "DATA", "data_dir"):
            value = getattr(runtime_cfg, name, None)
            if value:
                candidates.append(Path(value))
    candidates.extend([
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
        Path("/root/autodl-tmp/replay_data"),
        Path("/root/autodl-tmp"),
        Path.cwd(),
    ])
    root = Path("/kaggle/input")
    if root.exists():
        candidates.extend(path.parent for path in root.rglob("sample_submission.csv"))
    checked = []
    for candidate in dict.fromkeys(Path(path).expanduser() for path in candidates):
        checked.append(str(candidate))
        if (
            candidate.is_dir()
            and (candidate / "test").is_dir()
            and (candidate / "sample_submission.csv").is_file()
        ):
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate competition data root. Checked: " + repr(checked))


def _find_model_hn16():
    configured = os.environ.get("ROGII_HN15_MODEL")
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise FileNotFoundError(path)
        model = lgb.Booster(model_file=str(path))
        if model.num_feature() != 23:
            raise RuntimeError(
                f"Configured HN15 model has {model.num_feature()} features, expected 23"
            )
        return path.resolve(), model
    candidates = []
    for root in (Path("/kaggle/input"), Path("/root/autodl-tmp/oof_exports"), Path.cwd()):
        if root.exists():
            candidates.extend(root.rglob("hn_features_full.txt"))
    valid = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            model = lgb.Booster(model_file=str(path))
        except Exception:
            continue
        if model.num_feature() == 23:
            priority = int("realistic_pairwise_deployment_residual_hn15" in str(path))
            valid.append((priority, path.stat().st_mtime, path.resolve(), model))
    if not valid:
        raise FileNotFoundError(
            "Upload HN15 models/hn_features_full.txt, or set ROGII_HN15_MODEL"
        )
    _, _, path, model = sorted(valid, reverse=True)[0]
    return path, model


def _sha256_hn16(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _features_hn16(frame):
    out = pd.DataFrame(index=frame.index)
    group = frame.groupby("well", sort=False)
    distance = frame["md_since"].to_numpy(np.float64)
    base = frame["base"].to_numpy(np.float64)
    hn = frame["hn"].to_numpy(np.float64)
    delta = hn - base
    max_distance = group["md_since"].transform("max").to_numpy(np.float64)
    progress = distance / np.maximum(max_distance, 1.0)
    first_base = group["base"].transform("first").to_numpy(np.float64)
    first_hn = group["hn"].transform("first").to_numpy(np.float64)
    out["distance_kft"] = distance / 1000.0
    out["progress"] = progress
    out["progress2"] = progress * progress
    out["log1p_distance"] = np.log1p(np.maximum(distance, 0.0))
    out["base_relative"] = base - first_base
    out["base_relative_per_kft"] = (base - first_base) / np.maximum(distance / 1000.0, 0.25)
    for lag in (32, 128, 512):
        shifted_base = group["base"].shift(lag).to_numpy(np.float64)
        shifted_hn = group["hn"].shift(lag).to_numpy(np.float64)
        out[f"base_delta_lag{lag}"] = np.nan_to_num(base - shifted_base)
        out[f"hn_delta_lag{lag}"] = np.nan_to_num(hn - shifted_hn)
        out[f"disagreement_delta_lag{lag}"] = np.nan_to_num(
            delta - (shifted_hn - shifted_base)
        )
    out["hn_relative"] = hn - first_hn
    out["disagreement"] = delta
    out["abs_disagreement"] = np.abs(delta)
    out["disagreement_per_kft"] = delta / np.maximum(distance / 1000.0, 0.25)
    delta_series = pd.Series(delta, index=frame.index)
    out["well_delta_mean"] = delta_series.groupby(frame["well"], sort=False).transform("mean")
    out["well_delta_std"] = (
        delta_series.groupby(frame["well"], sort=False).transform("std").fillna(0.0)
    )
    out["well_abs_delta_mean"] = (
        pd.Series(np.abs(delta), index=frame.index)
        .groupby(frame["well"], sort=False)
        .transform("mean")
    )
    out["delta_minus_well_mean"] = delta - out["well_delta_mean"].to_numpy(np.float64)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


for required_hn16 in (PARENT_HN16, HN13_BLEND_HN16):
    if not required_hn16.is_file():
        raise FileNotFoundError(
            f"Missing {required_hn16}. Run the exact mainline and HN13 cells first."
        )

data_root_hn16 = _find_data_root_hn16()
model_path_hn16, model_hn16 = _find_model_hn16()
sample_hn16 = pd.read_csv(data_root_hn16 / "sample_submission.csv")
parent_hn16 = pd.read_csv(PARENT_HN16)
blend_hn16 = pd.read_csv(HN13_BLEND_HN16)
for frame_hn16, name_hn16 in (
    (sample_hn16, "sample"), (parent_hn16, "parent"), (blend_hn16, "HN13")
):
    if "id" not in frame_hn16 or len(frame_hn16) != len(sample_hn16):
        raise RuntimeError(f"Invalid {name_hn16} rows")
    if not frame_hn16["id"].astype(str).equals(sample_hn16["id"].astype(str)):
        raise RuntimeError(f"{name_hn16} id alignment differs from sample submission")

sample_value_hn16 = next((name for name in ("tvt", "TVT") if name in sample_hn16), None)
parent_value_hn16 = next((name for name in ("tvt", "TVT") if name in parent_hn16), None)
blend_value_hn16 = next((name for name in ("tvt", "TVT") if name in blend_hn16), None)
if None in (sample_value_hn16, parent_value_hn16, blend_value_hn16):
    raise RuntimeError("Cannot identify TVT column")

parent_values_hn16 = parent_hn16[parent_value_hn16].to_numpy(np.float64)
blend_values_hn16 = blend_hn16[blend_value_hn16].to_numpy(np.float64)
hn_values_hn16 = (
    blend_values_hn16 - (1.0 - HN13_WEIGHT_HN16) * parent_values_hn16
) / HN13_WEIGHT_HN16
if not np.isfinite(hn_values_hn16).all():
    raise RuntimeError("Reconstructed HN13 endpoint is non-finite")

ids_hn16 = sample_hn16["id"].astype(str)
wells_hn16 = ids_hn16.str.rsplit("_", n=1).str[0]
rows_hn16 = pd.to_numeric(ids_hn16.str.rsplit("_", n=1).str[1], errors="coerce")
if rows_hn16.isna().any():
    raise RuntimeError("Cannot parse source row suffix from submission ids")
rows_hn16 = rows_hn16.astype(np.int64)
md_since_hn16 = np.full(len(sample_hn16), np.nan, np.float64)
for number_hn16, well_hn16 in enumerate(sorted(wells_hn16.unique()), 1):
    horizontal_hn16 = pd.read_csv(
        data_root_hn16 / "test" / f"{well_hn16}__horizontal_well.csv",
        usecols=["MD", "TVT_input"],
    )
    md_hn16 = pd.to_numeric(horizontal_hn16["MD"], errors="coerce").to_numpy(np.float64)
    tvt_input_hn16 = pd.to_numeric(
        horizontal_hn16["TVT_input"], errors="coerce"
    ).to_numpy(np.float64)
    known_hn16 = np.isfinite(tvt_input_hn16)
    if not known_hn16.any():
        raise RuntimeError(f"{well_hn16}: no known TVT_input anchor")
    anchor_hn16 = int(np.flatnonzero(known_hn16)[-1])
    positions_hn16 = np.flatnonzero(wells_hn16.to_numpy() == well_hn16)
    source_rows_hn16 = rows_hn16.iloc[positions_hn16].to_numpy(np.int64)
    if source_rows_hn16.min() < 0 or source_rows_hn16.max() >= len(md_hn16):
        raise RuntimeError(f"{well_hn16}: submission row suffix outside horizontal CSV")
    md_since_hn16[positions_hn16] = md_hn16[source_rows_hn16] - md_hn16[anchor_hn16]

endpoint_hn16 = pd.DataFrame(
    {
        "sample_position": np.arange(len(sample_hn16), dtype=np.int64),
        "well": wells_hn16.to_numpy(),
        "row_idx": rows_hn16.to_numpy(np.int64),
        "md_since": md_since_hn16,
        "base": parent_values_hn16,
        "hn": hn_values_hn16,
    }
).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if not np.isfinite(endpoint_hn16[["md_since", "base", "hn"]].to_numpy()).all():
    raise RuntimeError("Non-finite HN16 endpoint frame")

features_hn16 = _features_hn16(endpoint_hn16)
if features_hn16.shape[1] != 23 or model_hn16.num_feature() != 23:
    raise RuntimeError(
        f"HN15 schema mismatch: frame/model={features_hn16.shape[1]}/{model_hn16.num_feature()}"
    )
correction_hn16 = np.clip(
    model_hn16.predict(features_hn16), -CAP_HN16, CAP_HN16
)
prediction_sorted_hn16 = endpoint_hn16["base"].to_numpy(np.float64) + correction_hn16
final_values_hn16 = np.empty(len(sample_hn16), np.float64)
final_values_hn16[endpoint_hn16["sample_position"].to_numpy(np.int64)] = prediction_sorted_hn16
if not np.isfinite(final_values_hn16).all():
    raise RuntimeError("HN16 output is non-finite")

output_hn16 = sample_hn16.copy()
output_hn16[sample_value_hn16] = final_values_hn16
output_hn16.to_csv(OUTPUT_HN16, index=False)
shutil.copy2(OUTPUT_HN16, FINAL_HN16)

summary_hn16 = {
    "experiment": "realistic_pairwise_kaggle_residual_hn16",
    "parent": str(PARENT_HN16),
    "hn13_score_mean": str(HN13_BLEND_HN16),
    "hn15_model": str(model_path_hn16),
    "hn15_model_sha256": _sha256_hn16(model_path_hn16),
    "output": str(OUTPUT_HN16),
    "final_submission": str(FINAL_HN16),
    "rows": len(output_hn16),
    "wells": int(endpoint_hn16["well"].nunique()),
    "correction_cap_ft": CAP_HN16,
    "dose": 1.0,
    "mean_abs_correction_ft": float(np.mean(np.abs(correction_hn16))),
    "rms_correction_ft": float(np.sqrt(np.mean(correction_hn16 ** 2))),
    "p95_abs_correction_ft": float(np.quantile(np.abs(correction_hn16), 0.95)),
    "maximum_abs_correction_ft": float(np.max(np.abs(correction_hn16))),
    "cap_activation_fraction": float(np.mean(np.abs(correction_hn16) >= CAP_HN16 - 1e-9)),
    "mean_abs_hn_parent_disagreement_ft": float(
        np.mean(np.abs(hn_values_hn16 - parent_values_hn16))
    ),
    "elapsed_minutes": (time.time() - STARTED_HN16) / 60.0,
    "instruction": "Submit submission.csv; it is an exact copy of submission_hn16_score_mean_residual.csv.",
}
(WORK_HN16 / "hn16_inference_summary.json").write_text(
    json.dumps(summary_hn16, indent=2), encoding="utf-8"
)
print("\nHN16 FINAL SUMMARY")
print(json.dumps(summary_hn16, indent=2), flush=True)
print("Submission ready:", FINAL_HN16, flush=True)
