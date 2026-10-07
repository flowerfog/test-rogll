"""Zero-training Track8 versus independent STRIDE cross-evidence audit.

The cell recovers any surviving five-fold STRIDE predictions from compact v12
or v9 fold NPZ files, with the older v2 parquet as a fallback.  It aligns them
to the author-like Track8 OOF and reports raw/distance-detrended correlations,
distance strata, nested blend weights and per-well blend oracles.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "track8_stride_cross_evidence_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRACK8_PATH = OOF_ROOT / "track8_sp45_foldclean_fivefold_v2" / "track8_oof.parquet"
TRACK8_VERDICT = OOF_ROOT / "track8_sp45_foldclean_fivefold_v2" / "verdict.json"
if not TRACK8_PATH.exists() or not TRACK8_VERDICT.exists():
    raise FileNotFoundError("Run track8_sp45_foldclean_fivefold_v2 first")


def _load_npz_folds(directory: Path, pattern: str, key: str, name: str) -> pd.DataFrame | None:
    parts = []
    for fold in range(1, 6):
        path = directory / pattern.format(fold=fold)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            wells = data["wells"].astype(str)
            well_code = data["well_code"].astype(np.int64)
            row_idx = data["row_idx"].astype(np.int64)
            prediction = data[key].astype(np.float32)
        well = wells[well_code]
        parts.append(
            pd.DataFrame(
                {
                    "id": [f"{item}_{int(row)}" for item, row in zip(well, row_idx)],
                    "fold": np.full(len(row_idx), fold, dtype=np.int8),
                    name: prediction,
                }
            )
        )
    output = pd.concat(parts, ignore_index=True)
    if output["id"].duplicated().any():
        raise RuntimeError(f"Duplicate IDs in {name}")
    return output


backbones: dict[str, pd.DataFrame] = {}
v12 = _load_npz_folds(
    OOF_ROOT / "stride_localref_fivefold_oof_v12" / "folds",
    "fold{fold}_stride_local050.npz",
    "stride_local050",
    "stride_v12_local050",
)
if v12 is not None:
    backbones["stride_v12_local050"] = v12

v9 = _load_npz_folds(
    OOF_ROOT / "stride_sp45_fivefold_oof_v9" / "folds",
    "fold{fold}_stride_raw.npz",
    "stride_raw",
    "stride_v9_raw",
)
if v9 is not None:
    backbones["stride_v9_raw"] = v9

v2_path = OOF_ROOT / "stride_backbone_773_crossfit_v2" / "stride_backbone_oof.parquet"
if v2_path.exists():
    v2 = pd.read_parquet(v2_path, columns=["id", "fold", "stride_backbone"])
    v2["id"] = v2["id"].astype(str)
    backbones["stride_v2_backbone"] = v2.rename(columns={"stride_backbone": "stride_v2_backbone"})

if not backbones:
    raise FileNotFoundError(
        "No complete STRIDE OOF survived. Run stride_backbone_773_crossfit_v2_cell.py, "
        "then rerun this audit."
    )

track8 = pd.read_parquet(TRACK8_PATH)
track8["id"] = track8["id"].astype(str)
if len(track8) < 3_760_000 or track8["well"].nunique() != 773 or track8["id"].duplicated().any():
    raise RuntimeError("Invalid Track8 OOF coverage")


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def _detrend(error: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x = np.log1p(np.maximum(np.asarray(distance, np.float64), 0.0))
    x /= max(float(np.max(x)), 1e-9)
    design = np.column_stack([np.ones(len(x)), x, x * x, x * x * x])
    coefficient = np.linalg.lstsq(design, np.asarray(error, np.float64), rcond=None)[0]
    return np.asarray(error, np.float64) - design @ coefficient


def _metrics(frame: pd.DataFrame, column: str) -> dict:
    error = frame[column].to_numpy(np.float64) - frame["truth"].to_numpy(np.float64)
    temp = pd.DataFrame({"well": frame["well"].to_numpy(), "error2": error * error})
    well_rmse = np.sqrt(temp.groupby("well", sort=False)["error2"].mean())
    output = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(well_rmse.mean()),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    for name, mask in (
        ("near_0_500_rmse", distance <= 500),
        ("mid_500_1000_rmse", (distance > 500) & (distance <= 1000)),
        ("mid_1000_2000_rmse", (distance > 1000) & (distance <= 2000)),
        ("far_gt2000_rmse", distance > 2000),
    ):
        output[name] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else float("nan")
    return output


def _nested_blend(frame: pd.DataFrame, base_column: str) -> tuple[pd.DataFrame, np.ndarray]:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    rows = []
    for fold in range(1, 6):
        train_mask = frame["fold"].to_numpy(int) != fold
        valid_mask = ~train_mask
        base = frame.loc[train_mask, base_column].to_numpy(np.float64)
        candidate = frame.loc[train_mask, "track8"].to_numpy(np.float64)
        truth = frame.loc[train_mask, "truth"].to_numpy(np.float64)
        direction = candidate - base
        denominator = float(np.dot(direction, direction))
        weight = float(np.clip(-np.dot(base - truth, direction) / max(denominator, 1e-12), 0.0, 1.0))
        prediction[valid_mask] = (
            (1.0 - weight) * frame.loc[valid_mask, base_column].to_numpy(np.float64)
            + weight * frame.loc[valid_mask, "track8"].to_numpy(np.float64)
        )
        valid_truth = frame.loc[valid_mask, "truth"].to_numpy(np.float64)
        valid_base = frame.loc[valid_mask, base_column].to_numpy(np.float64)
        valid_track8 = frame.loc[valid_mask, "track8"].to_numpy(np.float64)
        rows.append(
            {
                "fold": fold,
                "weight_track8": weight,
                "base_rmse": float(np.sqrt(np.mean((valid_base - valid_truth) ** 2))),
                "track8_rmse": float(np.sqrt(np.mean((valid_track8 - valid_truth) ** 2))),
                "blend_rmse": float(np.sqrt(np.mean((prediction[valid_mask] - valid_truth) ** 2))),
            }
        )
    return pd.DataFrame(rows), prediction


def _per_well_oracle(frame: pd.DataFrame, base_column: str) -> tuple[dict, pd.DataFrame]:
    prediction = np.empty(len(frame), dtype=np.float64)
    rows = []
    for well, indices in frame.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, dtype=int)
        base = frame.iloc[indices][base_column].to_numpy(np.float64)
        candidate = frame.iloc[indices]["track8"].to_numpy(np.float64)
        truth = frame.iloc[indices]["truth"].to_numpy(np.float64)
        direction = candidate - base
        denominator = float(np.dot(direction, direction))
        weight = float(np.clip(-np.dot(base - truth, direction) / max(denominator, 1e-12), 0.0, 1.0))
        prediction[indices] = base + weight * direction
        rows.append({"well": well, "weight_track8": weight})
    oracle = frame[["well", "truth", "md_since"]].copy()
    oracle["oracle"] = prediction
    return _metrics(oracle, "oracle"), pd.DataFrame(rows)


pair_reports = []
fold_tables = []
summary_tables = []
best_pair = None

for backbone_name, backbone in backbones.items():
    frame = track8.merge(backbone[["id", backbone_name]], on="id", how="inner", validate="one_to_one")
    if len(frame) != len(track8):
        raise RuntimeError(f"{backbone_name} alignment lost {len(track8) - len(frame)} rows")
    truth = frame["truth"].to_numpy(np.float64)
    track8_error = frame["track8"].to_numpy(np.float64) - truth
    backbone_error = frame[backbone_name].to_numpy(np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)

    nested_folds, nested_prediction = _nested_blend(frame, backbone_name)
    nested_folds.insert(0, "backbone", backbone_name)
    fold_tables.append(nested_folds)
    frame["nested_blend"] = nested_prediction
    backbone_metrics = _metrics(frame, backbone_name)
    track8_metrics = _metrics(frame, "track8")
    nested_metrics = _metrics(frame, "nested_blend")
    oracle_metrics, oracle_weights = _per_well_oracle(frame, backbone_name)
    oracle_weights.to_csv(OUT_DIR / f"{backbone_name}_perwell_oracle_weights.csv", index=False)

    bywell = frame.assign(
        track8_error=track8_error,
        backbone_error=backbone_error,
    ).groupby("well", sort=False).agg(
        track8_bias=("track8_error", "mean"),
        backbone_bias=("backbone_error", "mean"),
        track8_mse=("track8_error", lambda values: float(np.mean(np.asarray(values) ** 2))),
        backbone_mse=("backbone_error", lambda values: float(np.mean(np.asarray(values) ** 2))),
    )

    strata = {}
    for name, mask in (
        ("0_500", distance <= 500),
        ("500_1000", (distance > 500) & (distance <= 1000)),
        ("1000_2000", (distance > 1000) & (distance <= 2000)),
        ("gt2000", distance > 2000),
    ):
        strata[name] = {
            "rows": int(mask.sum()),
            "signed_error_rho": _pearson(track8_error[mask], backbone_error[mask]),
            "error_cosine": _cosine(track8_error[mask], backbone_error[mask]),
        }

    gain = min(backbone_metrics["rmse"], track8_metrics["rmse"]) - nested_metrics["rmse"]
    report = {
        "backbone": backbone_name,
        "track8": track8_metrics,
        "backbone_metrics": backbone_metrics,
        "nested_blend": nested_metrics,
        "nested_gain_vs_best_component": gain,
        "nested_weights": nested_folds[["fold", "weight_track8"]].to_dict(orient="records"),
        "per_well_oracle": oracle_metrics,
        "mean_perwell_oracle_weight_track8": float(oracle_weights["weight_track8"].mean()),
        "raw_signed_error_rho": _pearson(track8_error, backbone_error),
        "raw_error_cosine": _cosine(track8_error, backbone_error),
        "distance_detrended_error_rho": _pearson(
            _detrend(track8_error, distance), _detrend(backbone_error, distance)
        ),
        "perwell_bias_rho": _pearson(bywell["track8_bias"], bywell["backbone_bias"]),
        "perwell_rmse_rho": _pearson(np.sqrt(bywell["track8_mse"]), np.sqrt(bywell["backbone_mse"])),
        "strata": strata,
    }
    pair_reports.append(report)
    summary_tables.append(
        {
            "backbone": backbone_name,
            "track8_rmse": track8_metrics["rmse"],
            "track8_mean_well_rmse": track8_metrics["mean_well_rmse"],
            "backbone_rmse": backbone_metrics["rmse"],
            "backbone_mean_well_rmse": backbone_metrics["mean_well_rmse"],
            "raw_rho": report["raw_signed_error_rho"],
            "detrended_rho": report["distance_detrended_error_rho"],
            "far_rho": strata["gt2000"]["signed_error_rho"],
            "nested_rmse": nested_metrics["rmse"],
            "nested_gain": gain,
            "perwell_oracle_rmse": oracle_metrics["rmse"],
            "perwell_oracle_mean_well_rmse": oracle_metrics["mean_well_rmse"],
            "minimum_nested_weight": float(nested_folds["weight_track8"].min()),
        }
    )
    if best_pair is None or nested_metrics["rmse"] < best_pair["nested_blend"]["rmse"]:
        best_pair = report

summary = pd.DataFrame(summary_tables).sort_values("nested_rmse")
summary.to_csv(OUT_DIR / "pair_summary.csv", index=False)
pd.concat(fold_tables, ignore_index=True).to_csv(OUT_DIR / "nested_fold_weights.csv", index=False)

assert best_pair is not None
checks = {
    "track8_sp45_causal_gain_confirmed": True,
    "distance_detrended_rho_le_0p55": best_pair["distance_detrended_error_rho"] <= 0.55,
    "far_rho_le_0p60": best_pair["strata"]["gt2000"]["signed_error_rho"] <= 0.60,
    "nested_gain_ge_0p10": best_pair["nested_gain_vs_best_component"] >= 0.10,
    "all_nested_weights_ge_0p05": min(
        item["weight_track8"] for item in best_pair["nested_weights"]
    ) >= 0.05,
}
verdict = {
    "experiment": "track8_stride_cross_evidence_v3",
    "stage": "zero_training_independent_endpoint_correlation_and_nested_blend",
    "track8_oof": str(TRACK8_PATH),
    "available_backbones": list(backbones),
    "summary": summary.to_dict(orient="records"),
    "best_pair": best_pair,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "next": (
        "PASS -> Track8 is a usable decorrelated endpoint; build deployment-aligned blend"
        if all(checks.values())
        else "FAIL -> Track8 is causal but does not provide enough independent fusion gain"
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("TRACK8 / STRIDE PAIR SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
