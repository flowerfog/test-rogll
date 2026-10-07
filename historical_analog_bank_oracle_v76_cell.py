"""Route 6 Phase 0: historical-well analog bank oracle.

The target is matched to complete historical trajectories using only legal
target information: the visible surface prefix, full trajectory geometry and
horizontal GR.  Each historical suffix is transferred coherently from one
anchor; references are never switched from row to row.

This is an oracle/observability experiment, not a deployable selector.  It
tests whether prefix-conditioned retrieval puts useful complete paths in a
small top-k menu under both leave-one-well-out and whole-fold exclusion.
Run as one AutoDL notebook cell or as a script.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "historical_analog_bank_oracle_v76"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V66_PATH = (
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
DATA_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    ROOT / "replay_data",
    ROOT / "data",
)
DATA_ROOT = next((path for path in DATA_CANDIDATES if (path / "train").is_dir()), None)

EXPECTED_WELLS = 773
EXPECTED_ROWS = 3_783_989
QUERY_ROWS_PER_WELL = 192
TOP_K = 16
RANDOM_SEED = 20260729
PROTOCOLS = ("loo_all_other", "outer_fold_excluded")
FAMILIES = (
    "geometry",
    "surface_no_xy",
    "geometry_surface",
    "geometry_surface_gr",
    "full_context",
)
ORACLE_K = (1, 4, 8, 16)
BLEND_WEIGHTS = np.round(np.linspace(0.0, 0.40, 9), 2)

if DATA_ROOT is None:
    raise FileNotFoundError(f"No train directory under {[str(p) for p in DATA_CANDIDATES]}")
for required in (V66_PATH, FOLD_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)


def _metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    prediction = np.asarray(prediction, np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise RuntimeError(f"{endpoint}: invalid prediction")
    error = prediction - truth
    well_mse = (
        pd.DataFrame({"well": frame["well"].astype(str), "e2": error * error})
        .groupby("well", sort=False)["e2"].mean().to_numpy(np.float64)
    )
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(well_mse))),
        "p95_well_rmse": float(np.quantile(np.sqrt(well_mse), 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
    }


def _sample_vector(md: np.ndarray, values: np.ndarray, anchor: int, window: float,
                   nodes: int, relative: bool, standardise: bool) -> np.ndarray:
    anchor_md = float(md[anchor])
    grid = np.linspace(anchor_md - window, anchor_md, nodes)
    valid = np.isfinite(md) & np.isfinite(values) & (md <= anchor_md + 1e-6)
    if valid.sum() < 4:
        return np.full(nodes, np.nan, np.float64)
    x = md[valid]
    y = values[valid]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    vector = np.interp(grid, x, y, left=y[0], right=y[-1]).astype(np.float64)
    if relative:
        vector -= vector[-1]
    if standardise:
        vector -= np.mean(vector)
        scale = float(np.std(vector))
        if scale > 1e-6:
            vector /= scale
        else:
            vector[:] = 0.0
    return vector


def _slope(md: np.ndarray, surface_input: np.ndarray, anchor: int, window: float) -> float:
    edge = float(md[anchor])
    valid = (
        np.isfinite(md) & np.isfinite(surface_input)
        & (md >= edge - window) & (md <= edge + 1e-6)
    )
    if valid.sum() < 8:
        return np.nan
    x = md[valid] - edge
    y = surface_input[valid]
    return float(np.polyfit(x, y, 1)[0])


def _direction(xy: np.ndarray, anchor: int) -> np.ndarray:
    end = min(len(xy) - 1, anchor + max(64, (len(xy) - anchor) // 2))
    vector = xy[-1] - xy[anchor]
    if not np.isfinite(vector).all() or np.linalg.norm(vector) < 1.0:
        vector = xy[end] - xy[anchor]
    norm = float(np.linalg.norm(vector))
    if norm < 1.0:
        raise RuntimeError("degenerate trajectory direction")
    return vector / norm


def _interp_extrap(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    output = np.interp(query, x, y)
    if len(x) < 4:
        return output
    tail = x >= max(x[-1] - 384.0, x[0])
    if tail.sum() >= 4:
        slope = float(np.polyfit(x[tail] - x[-1], y[tail], 1)[0])
        slope = float(np.clip(slope, -0.12, 0.12))
        beyond = query > x[-1]
        output[beyond] = y[-1] + slope * (query[beyond] - x[-1])
    return output


def _robust_z(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    output = np.full(len(values), np.inf, np.float64)
    local = values[eligible]
    finite = np.isfinite(local)
    if finite.sum() == 0:
        output[eligible] = 0.0
        return output
    median = float(np.median(local[finite]))
    scale = 1.4826 * float(np.median(np.abs(local[finite] - median)))
    if scale < 1e-8:
        scale = float(np.std(local[finite])) + 1e-6
    transformed = (local - median) / scale
    transformed[~finite] = 8.0
    output[eligible] = np.clip(transformed, -4.0, 12.0)
    return output


def _rms_rows(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    difference = matrix - vector[None, :]
    return np.sqrt(np.nanmean(difference * difference, axis=1))


def _family_scores(target: dict, eligible: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    anchor_distance = np.linalg.norm(arrays["anchor_xy"] - target["anchor_xy"], axis=1)
    direction = np.clip(arrays["direction"] @ target["direction"], -1.0, 1.0)
    direction_penalty = 1.0 - direction
    length_penalty = np.abs(np.log(np.maximum(arrays["length"], 100.0) / max(target["length"], 100.0)))
    slope_penalty = np.sqrt(np.nanmean(
        (arrays["slopes"] - target["slopes"][None, :]) ** 2, axis=1
    ))
    surface_penalty = _rms_rows(arrays["surface_shape"], target["surface_shape"])
    z_penalty = _rms_rows(arrays["z_shape"], target["z_shape"])
    gr_penalty = _rms_rows(arrays["gr_shape"], target["gr_shape"])

    component = {
        "xy": _robust_z(anchor_distance, eligible),
        "dir": _robust_z(direction_penalty, eligible),
        "length": _robust_z(length_penalty, eligible),
        "slope": _robust_z(slope_penalty, eligible),
        "surface": _robust_z(surface_penalty, eligible),
        "z": _robust_z(z_penalty, eligible),
        "gr": _robust_z(gr_penalty, eligible),
    }
    scores = {
        "geometry": component["xy"] + 0.75 * component["dir"] + 0.35 * component["length"],
        "surface_no_xy": (
            0.50 * component["dir"] + 0.40 * component["length"]
            + component["slope"] + component["surface"]
        ),
        "geometry_surface": (
            0.50 * component["xy"] + 0.60 * component["dir"] + 0.35 * component["length"]
            + component["slope"] + component["surface"]
        ),
        "geometry_surface_gr": (
            0.40 * component["xy"] + 0.50 * component["dir"] + 0.30 * component["length"]
            + component["slope"] + component["surface"] + 0.60 * component["gr"]
        ),
        "full_context": (
            0.40 * component["xy"] + 0.50 * component["dir"] + 0.30 * component["length"]
            + component["slope"] + component["surface"]
            + 0.40 * component["gr"] + 0.35 * component["z"]
        ),
    }
    for score in scores.values():
        score[~eligible] = np.inf
    return scores


def _transfer(reference: dict, target: dict, query_distance: np.ndarray, target_z: np.ndarray) -> tuple[np.ndarray, float]:
    progress = reference["post_progress"]
    delta_surface = reference["post_surface_delta"]
    mapped_delta = _interp_extrap(progress, delta_surface, query_distance)
    prediction = (
        target["anchor_tvt"] + mapped_delta
        - (target_z - target["anchor_z"])
    )
    coverage = float(np.mean(query_distance <= progress[-1] + 1e-6))
    return prediction, coverage


# Load V66 and canonical folds.
header = pq.read_schema(V66_PATH).names
prediction_column = next((column for column in (
    "incumbent_6p377_replay", "prediction", "current_dose_0p20", "tvt_pred"
) if column in header), None)
if prediction_column is None:
    raise RuntimeError(f"Cannot identify V66 prediction: {header}")
columns = [column for column in (
    "id", "well", "row_idx", "fold", "tvt_true", "md_since", prediction_column
) if column in header]
v66 = pd.read_parquet(V66_PATH, columns=columns)
if len(v66) != EXPECTED_ROWS or v66["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(f"Unexpected V66 coverage: {len(v66)}/{v66['well'].nunique()}")
v66["well"] = v66["well"].astype(str)
fold_frame = pd.read_csv(FOLD_PATH)
fold_well = next(column for column in ("well", "well_id", "pair_id") if column in fold_frame)
fold_column = next(column for column in ("fold", "canonical_fold") if column in fold_frame)
fold_by_well = dict(zip(fold_frame[fold_well].astype(str), fold_frame[fold_column].astype(int)))

sample_parts = []
for _, group in v66.groupby("well", sort=False):
    take = np.unique(np.round(np.linspace(0, len(group) - 1, min(QUERY_ROWS_PER_WELL, len(group)))).astype(int))
    sample_parts.append(group.iloc[take])
sample = pd.concat(sample_parts, ignore_index=True).rename(columns={prediction_column: "base"})


# Build one legal prefix descriptor and one coherent suffix curve per well.
wells = sorted(sample["well"].unique())
records: dict[str, dict] = {}
failures = []
for number, well in enumerate(wells, 1):
    try:
        path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        frame = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"])
        frame = frame.apply(pd.to_numeric, errors="coerce")
        md = frame["MD"].to_numpy(np.float64)
        xy = frame[["X", "Y"]].to_numpy(np.float64)
        z = frame["Z"].to_numpy(np.float64)
        gr = frame["GR"].to_numpy(np.float64)
        tvt = frame["TVT"].to_numpy(np.float64)
        tvt_input = frame["TVT_input"].to_numpy(np.float64)
        known = np.flatnonzero(np.isfinite(tvt_input))
        if len(known) < 64:
            raise RuntimeError(f"only {len(known)} visible rows")
        anchor = int(known[-1])
        if len(md) - anchor < 100:
            raise RuntimeError("short suffix")
        surface = tvt + z
        surface_input = tvt_input + z
        post = np.arange(anchor, len(md), dtype=np.int32)
        post_progress = md[post] - md[anchor]
        post_delta = surface[post] - surface[anchor]
        if not np.isfinite(post_progress).all() or not np.isfinite(post_delta).all():
            raise RuntimeError("non-finite suffix")
        records[well] = {
            "well": well,
            "fold": int(fold_by_well[well]),
            "md": md,
            "xy": xy,
            "z": z,
            "anchor": anchor,
            "anchor_xy": xy[anchor],
            "anchor_z": float(z[anchor]),
            "anchor_tvt": float(tvt_input[anchor]),
            "direction": _direction(xy, anchor),
            "length": float(post_progress[-1]),
            "slopes": np.asarray([_slope(md, surface_input, anchor, window) for window in (128., 256., 512.)]),
            "surface_shape": _sample_vector(md, surface_input, anchor, 512., 24, True, False),
            "z_shape": _sample_vector(md, z, anchor, 512., 24, True, False),
            "gr_shape": _sample_vector(md, gr, anchor, 512., 32, False, True),
            "post_progress": post_progress,
            "post_surface_delta": post_delta,
        }
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 100 == 0 or number == len(wells):
        print(f"V76 load {number}/{len(wells)} success={len(records)} failures={len(failures)}", flush=True)

pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)
if len(records) < 760:
    raise RuntimeError(f"Only {len(records)} records: {failures[:5]}")
wells = sorted(records)
well_index = {well: index for index, well in enumerate(wells)}
arrays = {
    "anchor_xy": np.asarray([records[w]["anchor_xy"] for w in wells], np.float64),
    "direction": np.asarray([records[w]["direction"] for w in wells], np.float64),
    "length": np.asarray([records[w]["length"] for w in wells], np.float64),
    "slopes": np.asarray([records[w]["slopes"] for w in wells], np.float64),
    "surface_shape": np.asarray([records[w]["surface_shape"] for w in wells], np.float64),
    "z_shape": np.asarray([records[w]["z_shape"] for w in wells], np.float64),
    "gr_shape": np.asarray([records[w]["gr_shape"] for w in wells], np.float64),
}
fold_array = np.asarray([records[w]["fold"] for w in wells], np.int16)

print("V76 CONFIG", {
    "data": str(DATA_ROOT), "v66": str(V66_PATH), "sample_rows": len(sample),
    "sample_wells": sample["well"].nunique(), "records": len(records),
    "protocols": PROTOCOLS, "families": FAMILIES, "top_k": TOP_K,
}, flush=True)


summary_rows = []
blend_rows = []
choice_rows = []
protocol_payload = {}
for protocol in PROTOCOLS:
    aligned_parts = []
    family_legal = {family: [] for family in FAMILIES}
    family_median = {(family, k): [] for family in FAMILIES for k in (4, 8, 16)}
    family_oracle = {(family, k): [] for family in FAMILIES for k in ORACLE_K}
    global_oracle_parts = []

    for number, (well, group) in enumerate(sample[sample["well"].isin(records)].groupby("well", sort=False), 1):
        target = records[well]
        target_index = well_index[well]
        eligible = np.ones(len(wells), dtype=bool)
        eligible[target_index] = False
        if protocol == "outer_fold_excluded":
            eligible &= fold_array != target["fold"]
        scores = _family_scores(target, eligible, arrays)
        row_idx = group["row_idx"].to_numpy(np.int64)
        query_distance = target["md"][row_idx] - target["md"][target["anchor"]]
        target_z = target["z"][row_idx]
        truth = group["tvt_true"].to_numpy(np.float64)
        all_unique_paths = {}
        for family in FAMILIES:
            ranking = np.argsort(scores[family], kind="stable")[:TOP_K]
            paths = []
            for rank, reference_index in enumerate(ranking, 1):
                reference_well = wells[int(reference_index)]
                prediction, coverage = _transfer(records[reference_well], target, query_distance, target_z)
                mse = float(np.mean((prediction - truth) ** 2))
                paths.append(prediction)
                all_unique_paths.setdefault(reference_well, prediction)
                choice_rows.append({
                    "protocol": protocol, "well": well, "fold": target["fold"],
                    "family": family, "rank": rank, "reference_well": reference_well,
                    "retrieval_score": float(scores[family][reference_index]),
                    "path_rmse": float(np.sqrt(mse)), "coverage": coverage,
                    "anchor_distance": float(np.linalg.norm(target["anchor_xy"] - records[reference_well]["anchor_xy"])),
                    "direction_dot": float(np.dot(target["direction"], records[reference_well]["direction"])),
                })
            matrix = np.asarray(paths, np.float64)
            family_legal[family].append(matrix[0])
            for k in (4, 8, 16):
                family_median[(family, k)].append(np.median(matrix[:k], axis=0))
            per_path_sse = np.sum((matrix - truth[None, :]) ** 2, axis=1)
            for k in ORACLE_K:
                family_oracle[(family, k)].append(matrix[int(np.argmin(per_path_sse[:k]))])
        union_matrix = np.asarray(list(all_unique_paths.values()), np.float64)
        union_sse = np.sum((union_matrix - truth[None, :]) ** 2, axis=1)
        global_oracle_parts.append(union_matrix[int(np.argmin(union_sse))])
        aligned_parts.append(group)
        if number % 50 == 0 or number == len(records):
            print(f"V76 {protocol} {number}/{len(records)}", flush=True)

    frame = pd.concat(aligned_parts, ignore_index=True)
    base = frame["base"].to_numpy(np.float64)
    summary_rows.append({"protocol": protocol, "family": "base", **_metric(frame, base, "base_v66")})
    endpoint_payload = {}
    for family in FAMILIES:
        legal = np.concatenate(family_legal[family])
        endpoint_payload[f"{family}_top1"] = legal
        summary_rows.append({"protocol": protocol, "family": family, **_metric(frame, legal, "legal_top1")})
        for k in (4, 8, 16):
            median = np.concatenate(family_median[(family, k)])
            endpoint_payload[f"{family}_median{k}"] = median
            summary_rows.append({"protocol": protocol, "family": family, **_metric(frame, median, f"median_top{k}")})
        for k in ORACLE_K:
            oracle = np.concatenate(family_oracle[(family, k)])
            summary_rows.append({"protocol": protocol, "family": family, **_metric(frame, oracle, f"oracle_top{k}")})
        for weight in BLEND_WEIGHTS:
            blend = base + float(weight) * (legal - base)
            blend_rows.append({
                "protocol": protocol, "family": family, "weight": float(weight),
                **_metric(frame, blend, f"blend_{family}_w{weight:g}"),
            })
    global_oracle = np.concatenate(global_oracle_parts)
    summary_rows.append({"protocol": protocol, "family": "union", **_metric(frame, global_oracle, "oracle_union_top16")})
    protocol_payload[protocol] = {"frame": frame, "endpoints": endpoint_payload}

summary = pd.DataFrame(summary_rows)
blend_scan = pd.DataFrame(blend_rows)
choices = pd.DataFrame(choice_rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
blend_scan.to_csv(OUT_DIR / "blend_scan.csv", index=False)
choices.to_parquet(OUT_DIR / "analog_candidates.parquet", index=False, compression="zstd")

best_blends = (
    blend_scan.sort_values(["protocol", "rmse", "family", "weight"], kind="stable")
    .groupby("protocol", sort=False).head(1).reset_index(drop=True)
)
oracle_table = summary[summary["endpoint"].str.startswith("oracle_")].copy()
print("\nV76 ANALOG ORACLE SUMMARY")
print(oracle_table.sort_values(["protocol", "rmse"]).to_string(index=False), flush=True)
print("\nV76 LEGAL / MEDIAN SUMMARY")
print(summary[~summary["endpoint"].str.startswith("oracle_")].sort_values(["protocol", "rmse"]).to_string(index=False), flush=True)
print("\nV76 BEST FIXED BLENDS")
print(best_blends.to_string(index=False), flush=True)


def _row(protocol: str, family: str, endpoint: str) -> pd.Series:
    match = summary[
        summary["protocol"].eq(protocol) & summary["family"].eq(family)
        & summary["endpoint"].eq(endpoint)
    ]
    if len(match) != 1:
        raise RuntimeError(f"Missing summary row {protocol}/{family}/{endpoint}")
    return match.iloc[0]


def _best_oracle(protocol: str, endpoint: str) -> pd.Series:
    match = summary[summary["protocol"].eq(protocol) & summary["endpoint"].eq(endpoint)]
    if match.empty:
        raise RuntimeError(f"Missing oracle {protocol}/{endpoint}")
    return match.sort_values("rmse", kind="stable").iloc[0]


loo_top8 = _best_oracle("loo_all_other", "oracle_top8")
outer_top8 = _best_oracle("outer_fold_excluded", "oracle_top8")
loo_geometry = _row("loo_all_other", "geometry", "oracle_top8")
outer_geometry = _row("outer_fold_excluded", "geometry", "oracle_top8")
loo_union = _row("loo_all_other", "union", "oracle_union_top16")
outer_union = _row("outer_fold_excluded", "union", "oracle_union_top16")
loo_base = _row("loo_all_other", "base", "base_v66")
outer_base = _row("outer_fold_excluded", "base", "base_v66")

checks = {
    "loo_best_top8_oracle_rmse_le_7": bool(float(loo_top8["rmse"]) <= 7.0),
    "outer_best_top8_oracle_rmse_le_7p5": bool(float(outer_top8["rmse"]) <= 7.5),
    "loo_context_beats_geometry_by_ge_0p25": bool(float(loo_geometry["rmse"] - loo_top8["rmse"]) >= 0.25),
    "outer_context_beats_geometry_by_ge_0p15": bool(float(outer_geometry["rmse"] - outer_top8["rmse"]) >= 0.15),
    "loo_union_oracle_rmse_le_6p5": bool(float(loo_union["rmse"]) <= 6.5),
    "outer_union_oracle_rmse_le_7": bool(float(outer_union["rmse"]) <= 7.0),
}
strict_pass = all(checks.values())
verdict = {
    "experiment": EXPERIMENT,
    "stage": "route6_phase0_prefix_conditioned_complete_trajectory_analog_oracle",
    "sample_unit": "192 fixed hidden-suffix rows per complete physical well",
    "legal_retrieval_inputs": [
        "target visible-prefix surface shape and slopes", "target horizontal GR prefix shape",
        "target complete XY/Z trajectory geometry", "reference-well historical complete trajectories",
    ],
    "truth_usage": "reference history plus target audit/oracle labels only",
    "protocols": {
        "loo_all_other": "target well excluded; deployment-equivalent historical library",
        "outer_fold_excluded": "target canonical fold entirely excluded",
    },
    "sources": {"v66": str(V66_PATH), "folds": str(FOLD_PATH), "data": str(DATA_ROOT)},
    "coverage": {"rows": int(len(sample)), "wells": int(sample["well"].nunique()), "records": int(len(records))},
    "loo": {
        "base_rmse": float(loo_base["rmse"]), "best_top8": loo_top8.to_dict(),
        "geometry_top8": loo_geometry.to_dict(), "union": loo_union.to_dict(),
        "best_fixed_blend": best_blends[best_blends["protocol"].eq("loo_all_other")].iloc[0].to_dict(),
    },
    "outer_fold_excluded": {
        "base_rmse": float(outer_base["rmse"]), "best_top8": outer_top8.to_dict(),
        "geometry_top8": outer_geometry.to_dict(), "union": outer_union.to_dict(),
        "best_fixed_blend": best_blends[best_blends["protocol"].eq("outer_fold_excluded")].iloc[0].to_dict(),
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze the analog menu and train a nested complete-well retrieval scorer"
        if strict_pass else
        "FAIL -> close prefix-conditioned historical analog transfer and proceed to Route 5 geometry-only endpoint"
    ),
}
(OUT_DIR / "result.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nV76 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR)
