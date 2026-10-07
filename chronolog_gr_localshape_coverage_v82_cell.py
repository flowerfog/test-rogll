"""V82: remove V81's residual level confound before closing Chronolog pretraining.

V81 normalised each curve at well level. This audit normalises every fixed-size
window independently and also evaluates standardised first-difference shape.
It compares cross-domain nearest-neighbour distance with the nearest window
from another well in the same domain. No target or prediction is used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


STARTED_V82 = time.time()
ROOT_V82 = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE_V82 = ROOT_V82 / "chronolog_gr_domain_gateway_v81_cell.py"
if not SOURCE_V82.is_file():
    local = Path.cwd() / "chronolog_gr_domain_gateway_v81_cell.py"
    SOURCE_V82 = local if local.is_file() else SOURCE_V82
if not SOURCE_V82.is_file():
    raise FileNotFoundError(
        "Upload chronolog_gr_domain_gateway_v81_cell.py beside this V82 cell."
    )

source_v82 = SOURCE_V82.read_text(encoding="utf-8")
marker_v82 = "chronolog_windows, chronolog_info = load_chronolog()"
if marker_v82 not in source_v82:
    raise RuntimeError("V81 execution marker not found")
namespace_v82 = {"__file__": str(SOURCE_V82), "__name__": "chronolog_v81_definitions"}
exec(compile(source_v82.split(marker_v82, 1)[0], str(SOURCE_V82), "exec"), namespace_v82)

load_chronolog = namespace_v82["load_chronolog"]
load_competition = namespace_v82["load_competition"]
SCALES_FT = namespace_v82["SCALES_FT"]
RANDOM_STATE = namespace_v82["RANDOM_STATE"]
OUT_DIR_V82 = ROOT_V82 / "oof_exports" / "chronolog_gr_localshape_coverage_v82"
OUT_DIR_V82.mkdir(parents=True, exist_ok=True)


def local_standardise(values):
    values = np.asarray(values, np.float64)
    centre = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - centre)))
    if not np.isfinite(scale) or scale < 1e-4:
        scale = float(max(np.std(values), 1e-3))
    return np.clip((values - centre) / scale, -8.0, 8.0)


def acf(values, lag):
    values = np.asarray(values, np.float64)
    if len(values) <= lag:
        return 0.0
    left = values[:-lag] - values[:-lag].mean()
    right = values[lag:] - values[lag:].mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 1e-9 else 0.0


def shape_vector(window, mode):
    values = local_standardise(window["raw"])
    if mode == "derivative":
        values = np.diff(values)
        values = values / max(float(np.std(values)), 1e-3)
        values = np.clip(values, -8.0, 8.0)
    delta = np.diff(values)
    second = np.diff(values, 2)
    power = np.abs(np.fft.rfft(values - values.mean())) ** 2
    total = float(power[1:].sum() + 1e-9)
    n = len(power)
    return np.asarray(
        [
            np.std(values),
            np.quantile(values, 0.05),
            np.quantile(values, 0.10),
            np.quantile(values, 0.25),
            np.quantile(values, 0.50),
            np.quantile(values, 0.75),
            np.quantile(values, 0.90),
            np.quantile(values, 0.95),
            np.mean(np.abs(delta)),
            np.std(delta),
            np.quantile(np.abs(delta), 0.50),
            np.quantile(np.abs(delta), 0.90),
            np.mean(np.abs(second)),
            np.std(second),
            acf(values, 1),
            acf(values, 2),
            acf(values, 4),
            acf(values, 8),
            acf(values, 16),
            power[1 : max(2, n // 8)].sum() / total,
            power[max(2, n // 8) : max(3, n // 3)].sum() / total,
            power[max(3, n // 3) :].sum() / total,
            np.mean(np.sign(delta[:-1]) != np.sign(delta[1:])),
        ],
        np.float64,
    )


def grouped_domain_auc(x, labels, groups):
    prediction = np.full(len(labels), np.nan)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train, valid in splitter.split(x, labels, groups):
        model = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x[train], labels[train])
        prediction[valid] = model.predict_proba(x[valid])[:, 1]
    auc = float(roc_auc_score(labels, prediction))
    return max(auc, 1.0 - auc)


def different_well_nn(x, groups, query_k=32):
    model = NearestNeighbors(n_neighbors=min(query_k, len(x))).fit(x)
    distances, indices = model.kneighbors(x)
    output = np.full(len(x), np.nan)
    for row in range(len(x)):
        valid = groups[indices[row]] != groups[row]
        if np.any(valid):
            output[row] = distances[row, np.flatnonzero(valid)[0]]
    if not np.isfinite(output).all():
        raise RuntimeError("Could not find a different-well neighbour")
    return output


def coverage_metrics(x_left, groups_left, x_right, groups_right):
    scaler = StandardScaler().fit(np.vstack([x_left, x_right]))
    left = scaler.transform(x_left)
    right = scaler.transform(x_right)
    left_same = different_well_nn(left, groups_left)
    right_same = different_well_nn(right, groups_right)
    left_cross = NearestNeighbors(n_neighbors=1).fit(right).kneighbors(left, return_distance=True)[0][:, 0]
    right_cross = NearestNeighbors(n_neighbors=1).fit(left).kneighbors(right, return_distance=True)[0][:, 0]
    left_ratio = left_cross / np.maximum(left_same, 1e-6)
    right_ratio = right_cross / np.maximum(right_same, 1e-6)
    return {
        "chronolog_cross_same_ratio_median": float(np.median(left_ratio)),
        "competition_cross_same_ratio_median": float(np.median(right_ratio)),
        "chronolog_covered_at_1p25_fraction": float(np.mean(left_ratio <= 1.25)),
        "competition_covered_at_1p25_fraction": float(np.mean(right_ratio <= 1.25)),
        "chronolog_covered_at_1p50_fraction": float(np.mean(left_ratio <= 1.50)),
        "competition_covered_at_1p50_fraction": float(np.mean(right_ratio <= 1.50)),
    }


chronolog_windows_v82, chronolog_info_v82 = load_chronolog()
competition_windows_v82, competition_info_v82 = load_competition()
windows_v82 = chronolog_windows_v82 + competition_windows_v82

records_v82 = []
for scale_ft in SCALES_FT:
    for target_source in ("competition_typewell", "competition_prefix"):
        selected = [
            row
            for row in windows_v82
            if row["scale_ft"] == scale_ft
            and row["source"] in {"chronolog", target_source}
        ]
        labels = np.asarray([row["source"] == target_source for row in selected], np.int8)
        groups = np.asarray([row["group"] for row in selected])
        for mode in ("local_shape", "derivative"):
            x = np.stack([shape_vector(row, mode) for row in selected])
            auc = grouped_domain_auc(x, labels, groups)
            coverage = coverage_metrics(
                x[labels == 0], groups[labels == 0], x[labels == 1], groups[labels == 1]
            )
            records_v82.append(
                {
                    "scale_ft": float(scale_ft),
                    "target_domain": target_source,
                    "representation": mode,
                    "domain_auc": auc,
                    **coverage,
                }
            )
        print(f"V82 {scale_ft:g} {target_source} complete", flush=True)

summary_v82 = pd.DataFrame(records_v82)
decision_rows = summary_v82[
    (summary_v82["scale_ft"].isin([64.0, 128.0]))
    & (summary_v82["target_domain"] == "competition_typewell")
]
local_rows = decision_rows[decision_rows["representation"] == "local_shape"]
derivative_rows = decision_rows[decision_rows["representation"] == "derivative"]
checks_v82 = {
    "local_shape_auc_mean_le_0p85": bool(local_rows["domain_auc"].mean() <= 0.85),
    "derivative_auc_mean_le_0p85": bool(derivative_rows["domain_auc"].mean() <= 0.85),
    "local_competition_coverage_mean_ge_0p50": bool(
        local_rows["competition_covered_at_1p50_fraction"].mean() >= 0.50
    ),
    "derivative_competition_coverage_mean_ge_0p50": bool(
        derivative_rows["competition_covered_at_1p50_fraction"].mean() >= 0.50
    ),
}
strict_pass_v82 = bool(
    (checks_v82["local_shape_auc_mean_le_0p85"] or checks_v82["derivative_auc_mean_le_0p85"])
    and checks_v82["local_competition_coverage_mean_ge_0p50"]
    and checks_v82["derivative_competition_coverage_mean_ge_0p50"]
)
verdict_v82 = {
    "experiment": "chronolog_gr_localshape_coverage_v82",
    "stage": "per_window_normalised_shape_and_derivative_domain_coverage",
    "source": str(SOURCE_V82),
    "chronolog_wells": int(chronolog_info_v82["wells"]),
    "competition_typewell_wells": int(competition_info_v82["typewell_wells"]),
    "competition_prefix_wells": int(competition_info_v82["prefix_wells"]),
    "decision_scales_ft": [64.0, 128.0],
    "checks": checks_v82,
    "strict_pass": strict_pass_v82,
    "elapsed_minutes": (time.time() - STARTED_V82) / 60.0,
    "next": (
        "PASS -> pretrain only the passing local/derivative representation, then freeze it"
        if strict_pass_v82
        else "FAIL -> close descriptor pretraining; retain only empirical augmentation ranges"
    ),
}

summary_v82.to_csv(OUT_DIR_V82 / "localshape_coverage.csv", index=False)
(OUT_DIR_V82 / "verdict.json").write_text(json.dumps(verdict_v82, indent=2), encoding="utf-8")
print("\nV82 LOCAL-SHAPE COVERAGE")
print(summary_v82.to_string(index=False))
print("\nV82 FINAL VERDICT")
print(json.dumps(verdict_v82, indent=2), flush=True)
print("saved:", OUT_DIR_V82)
