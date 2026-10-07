"""V131: transductive spatial-dip consensus over the V4 candidate bank.

Historical Neighbor transfers another well's TVT curve. V125 interpolates
truth-fitted residual coefficients. This experiment does neither: every well
keeps its own candidate trajectories, while nearby wells contribute only a
label-free low-frequency dip profile used to reorder those candidates.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "transductive_spatial_dip_consensus_v131"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V110_SOURCE = ROOT / "rawdata_lattice_topk_path_barycenter_v110_cell.py"
FOLD = 3
DISCOVERY_WELLS = 24
MAX_CANDIDATES = 66
OUTPUT_K = 33
GRID_NODES = 128
FREEZE_FT = 1000.0
RAMP_FT = 250.0
CAP_FT = 6.0
POINTS = ("anchor", "midpoint")
NEIGHBORS = (4, 8, 16)
SHAPE_NODES = (9, 17)
SUPPORTS = (33, 66)
PENALTIES = (0.02, 0.05, 0.10, 0.20)
TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _bootstrap_v110():
    if not V110_SOURCE.is_file():
        raise FileNotFoundError(V110_SOURCE)
    source = V110_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V110 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V110 marker missing from {V110_SOURCE}")
    module = types.ModuleType("v131_v110_definitions")
    module.__file__ = str(V110_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V110_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _robust_standardize(values):
    values = np.asarray(values, np.float64)
    center = float(np.median(values))
    scale = float(np.median(np.abs(values - center))) * 1.4826
    if scale < 1e-6:
        scale = max(float(np.std(values)), 1e-6)
    return (values - center) / scale


R = _bootstrap_v110()
for required in (R.V81_SOURCE, R.PANEL_DIR, R.MODEL_PATH, R.V70_PATH, R.V76_WEIGHTS):
    if not Path(required).exists():
        raise FileNotFoundError(required)
N = R._bootstrap_v81()
Q, V, B, prior = N.Q, N.V, N.B, N.prior
ranker = lgb.Booster(model_file=str(R.MODEL_PATH))
data_root = Path(V["DATA_ROOT"])

frame = pd.read_parquet(
    R.V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
    filters=[("fold", "=", FOLD)],
).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
frame["well"] = frame["well"].astype(str)
weights = pd.read_parquet(R.V76_WEIGHTS)
weights = weights[weights["fold"].eq(FOLD)][["well", "ensemble_weight"]].copy()
weights["well"] = weights["well"].astype(str)
weight_map = dict(zip(weights["well"], weights["ensemble_weight"].astype(float)))
frame["dose"] = frame["well"].map(weight_map)
if frame.empty or frame["dose"].isna().any() or frame["id"].duplicated().any():
    raise RuntimeError("Incomplete exact Fold3 V76 source")
frame["exact_v76"] = frame[PARENT].to_numpy(np.float64) + frame["dose"].to_numpy(np.float64) * (
    frame[COARSE].to_numpy(np.float64) - frame[PARENT].to_numpy(np.float64)
)

wells = sorted(w for w in frame["well"].unique() if (R.PANEL_DIR / f"{w}.npz").is_file())
if len(wells) < 153:
    raise RuntimeError(f"Only {len(wells)} Fold3 panels")

grid = np.linspace(0.0, 1.0, GRID_NODES)
trajectories_by_well = {}
maximum_distance = {}
coordinates = []
base_parts = []
build_rows = []
for number, well in enumerate(wells, 1):
    local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
    with np.load(R.PANEL_DIR / f"{well}.npz", allow_pickle=False) as data:
        features = data["features"].astype(np.float32, copy=False)
        paths = data["paths"].astype(np.int16, copy=False)
    record = V["_load_record"](well)
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    if len(local) != len(eval_rows) or np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
        raise RuntimeError(f"{well}: Fold3 alignment")
    scores = ranker.predict(features)
    order = R._deduplicated_order(paths, scores)
    if len(order) < MAX_CANDIDATES:
        raise RuntimeError(f"{well}: only {len(order)} unique candidates")
    segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
    distance = local["md_since"].to_numpy(np.float64)
    progress = distance / max(float(np.max(distance)), 1.0)
    trajectories = np.vstack([
        np.interp(
            grid,
            progress,
            B["_path_prediction"](record, prior, paths[index], segments)[eval_rows],
        )
        for index in order[:MAX_CANDIDATES]
    ]).astype(np.float32)
    trajectories_by_well[well] = trajectories
    maximum_distance[well] = float(np.max(distance))

    horizontal_path = data_root / "train" / f"{well}__horizontal_well.csv"
    horizontal = pd.read_csv(horizontal_path, usecols=["MD", "X", "Y", "TVT_input"])
    horizontal = horizontal.sort_values("MD", kind="stable").reset_index(drop=True)
    visible = np.flatnonzero(np.isfinite(horizontal["TVT_input"].to_numpy(np.float64)))
    if not len(visible):
        raise RuntimeError(f"{well}: no visible anchor")
    anchor = int(visible[-1])
    xy = horizontal[["X", "Y"]].to_numpy(np.float64)
    anchor_xy, end_xy = xy[anchor], xy[-1]
    direction = end_xy - anchor_xy
    direction /= max(float(np.linalg.norm(direction)), 1.0)
    coordinates.append({
        "well": well,
        "anchor_x": float(anchor_xy[0]), "anchor_y": float(anchor_xy[1]),
        "midpoint_x": float((anchor_xy[0] + end_xy[0]) * 0.5),
        "midpoint_y": float((anchor_xy[1] + end_xy[1]) * 0.5),
        "direction_x": float(direction[0]), "direction_y": float(direction[1]),
    })
    base_parts.append(local)
    build_rows.append({"well": well, "candidates": len(paths), "unique": len(order)})
    if number % 20 == 0 or number == len(wells):
        print(f"V131 build {number}/{len(wells)} elapsed_min={(time.time()-STARTED)/60:.1f}", flush=True)

pd.DataFrame(build_rows).to_csv(OUT_DIR / "build_report.csv", index=False)
coordinate_frame = pd.DataFrame(coordinates).set_index("well").loc[wells]
base_frame = pd.concat(base_parts, ignore_index=True)
discovery_wells = set(wells[:DISCOVERY_WELLS])
confirmation_wells = set(wells[DISCOVERY_WELLS:])


def _shape_matrix(shape_nodes):
    sample_grid = np.linspace(0.0, 1.0, int(shape_nodes))
    output = {}
    for well in wells:
        values = np.vstack([
            np.interp(sample_grid, grid, path) for path in trajectories_by_well[well]
        ])
        step_ft = maximum_distance[well] / max(int(shape_nodes) - 1, 1)
        output[well] = np.diff(values, axis=1) * (1000.0 / max(step_ft, 1.0))
    return output


shape_matrices = {nodes: _shape_matrix(nodes) for nodes in SHAPE_NODES}
neighbor_maps = {}
for point in POINTS:
    xy = coordinate_frame[[f"{point}_x", f"{point}_y"]].to_numpy(np.float64)
    distances = np.sqrt(np.sum(np.square(xy[:, None, :] - xy[None, :, :]), axis=2))
    direction = coordinate_frame[["direction_x", "direction_y"]].to_numpy(np.float64)
    direction_dot = np.clip(direction @ direction.T, -1.0, 1.0)
    effective_distance = distances / np.maximum(np.abs(direction_dot), 0.25)
    np.fill_diagonal(effective_distance, np.inf)
    for k in NEIGHBORS:
        chosen = np.argsort(effective_distance, axis=1, kind="stable")[:, :k]
        selected_distance = np.take_along_axis(effective_distance, chosen, axis=1)
        selected_weight = 1.0 / np.maximum(selected_distance, 100.0)
        selected_weight /= selected_weight.sum(axis=1, keepdims=True)
        selected_sign = np.sign(np.take_along_axis(direction_dot, chosen, axis=1))
        selected_sign[selected_sign == 0.0] = 1.0
        neighbor_maps[(point, k)] = (chosen, selected_weight, selected_sign)


def _centers(point, k, shape_nodes, support, penalty, rolled=False):
    shapes = shape_matrices[shape_nodes]
    baseline_shapes = np.stack([np.mean(shapes[well][:OUTPUT_K], axis=0) for well in wells])
    chosen, spatial_weight, direction_sign = neighbor_maps[(point, k)]
    neighbor_shapes = baseline_shapes[chosen]
    neighbor_shapes = np.where(
        direction_sign[:, :, None] >= 0.0,
        neighbor_shapes,
        -neighbor_shapes[:, :, ::-1],
    )
    references = np.sum(
        spatial_weight[:, :, None] * neighbor_shapes, axis=1,
    )
    if rolled:
        references = np.roll(references, max(1, len(wells) // 7), axis=0)
    centers = {}
    selected_rows = []
    for well_index, well in enumerate(wells):
        candidate_shapes = shapes[well][:support]
        residual = candidate_shapes - references[well_index][None, :]
        residual_center = np.median(residual, axis=0)
        scale = np.maximum(
            1.4826 * np.median(np.abs(residual - residual_center[None, :]), axis=0),
            1.0,
        )
        robust_cost = np.mean(np.log1p(np.square(residual / scale[None, :])), axis=1)
        geometry_score = _robust_standardize(robust_cost)
        rank_score = np.arange(support, dtype=np.float64) / max(support - 1, 1)
        selected = np.argsort(rank_score + float(penalty) * geometry_score, kind="stable")[:OUTPUT_K]
        centers[well] = np.mean(trajectories_by_well[well][selected], axis=0)
        selected_rows.append(selected)
    return centers, np.asarray(selected_rows, np.int16)


def _prediction(centers):
    parts = []
    for well in wells:
        local = frame[frame["well"].eq(well)].sort_values("row_idx", kind="stable")
        distance = local["md_since"].to_numpy(np.float64)
        progress = distance / max(float(np.max(distance)), 1.0)
        center = np.interp(progress, grid, centers[well])
        exact = local["exact_v76"].to_numpy(np.float64)
        coarse = local[COARSE].to_numpy(np.float64)
        dose = float(local["dose"].iloc[0])
        taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
        move = np.clip(dose * (center - coarse), -CAP_FT, CAP_FT)
        parts.append(exact + taper * move)
    return np.concatenate(parts)


baseline_centers = {well: np.mean(trajectories_by_well[well][:OUTPUT_K], axis=0) for well in wells}
baseline_prediction = _prediction(baseline_centers)
discovery_mask = base_frame["well"].isin(discovery_wells).to_numpy()
confirmation_mask = ~discovery_mask
base_discovery = R._metric(
    base_frame.loc[discovery_mask], baseline_prediction[discovery_mask], "uniform_top33"
)
base_all = R._metric(base_frame, baseline_prediction, "uniform_top33")
if abs(float(base_all["rmse"]) - 7.600041213) > 3e-3:
    raise RuntimeError(f"V131 top33 parity failed: {base_all['rmse']}")

recipes = [
    (point, k, nodes, support, penalty)
    for point in POINTS for k in NEIGHBORS for nodes in SHAPE_NODES
    for support in SUPPORTS for penalty in PENALTIES
]
scan_rows = []
prediction_cache = {}
selection_cache = {}
for recipe in recipes:
    centers, selected = _centers(*recipe)
    prediction = _prediction(centers)
    prediction_cache[recipe] = prediction
    selection_cache[recipe] = selected
    metric = R._metric(
        base_frame.loc[discovery_mask], prediction[discovery_mask], "spatial_dip_consensus"
    )
    metric.update({
        "point": recipe[0], "neighbors": recipe[1], "shape_nodes": recipe[2],
        "support": recipe[3], "penalty": recipe[4],
        "gain": float(base_discovery["rmse"] - metric["rmse"]),
        "far_gain": float(base_discovery["far_gt2000_rmse"] - metric["far_gt2000_rmse"]),
        "near_degradation": float(metric["near_0_500_rmse"] - base_discovery["near_0_500_rmse"]),
        "p95_gain": float(base_discovery["p95_well_rmse"] - metric["p95_well_rmse"]),
    })
    scan_rows.append(metric)

scan = pd.DataFrame(scan_rows).sort_values(["rmse", "p95_well_rmse"], kind="stable")
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
eligible = scan[
    (scan["gain"] >= 0.05) & (scan["far_gain"] >= 0.05)
    & (scan["near_degradation"] <= 1e-6) & (scan["p95_gain"] >= -0.20)
]
selected_row = (eligible if len(eligible) else scan).iloc[0]
selected_recipe = (
    str(selected_row["point"]), int(selected_row["neighbors"]),
    int(selected_row["shape_nodes"]), int(selected_row["support"]),
    float(selected_row["penalty"]),
)
selected_prediction = prediction_cache[selected_recipe]
rolled_centers, _ = _centers(*selected_recipe, rolled=True)
rolled_prediction = _prediction(rolled_centers)

summary_rows = []
for stage, mask in (("fold3_discovery", discovery_mask), ("fold3_confirmation", confirmation_mask)):
    summary_rows.extend([
        {"stage": stage, **R._metric(base_frame.loc[mask], baseline_prediction[mask], "uniform_top33")},
        {"stage": stage, **R._metric(base_frame.loc[mask], selected_prediction[mask], "spatial_dip_consensus")},
        {"stage": stage, **R._metric(base_frame.loc[mask], rolled_prediction[mask], "rolled_spatial_control")},
    ])
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)


def _row(stage, endpoint):
    return summary[summary["stage"].eq(stage) & summary["endpoint"].eq(endpoint)].iloc[0]


conf_base = _row("fold3_confirmation", "uniform_top33")
conf_candidate = _row("fold3_confirmation", "spatial_dip_consensus")
conf_control = _row("fold3_confirmation", "rolled_spatial_control")
values = {
    "gain": float(conf_base["rmse"] - conf_candidate["rmse"]),
    "far_gain": float(conf_base["far_gt2000_rmse"] - conf_candidate["far_gt2000_rmse"]),
    "near_degradation": float(conf_candidate["near_0_500_rmse"] - conf_base["near_0_500_rmse"]),
    "p95_gain": float(conf_base["p95_well_rmse"] - conf_candidate["p95_well_rmse"]),
    "aligned_increment_vs_rolled": float(conf_control["rmse"] - conf_candidate["rmse"]),
}
checks = {
    "discovery_gain_ge_0p05": float(selected_row["gain"]) >= 0.05,
    "confirmation_gain_ge_0p05": values["gain"] >= 0.05,
    "confirmation_far_gain_nonnegative": values["far_gain"] >= 0.0,
    "near_frozen_exact": abs(values["near_degradation"]) <= 1e-6,
    "p95_not_worse_by_gt_0p20": values["p95_gain"] >= -0.20,
    "aligned_beats_rolled_ge_0p03": values["aligned_increment_vs_rolled"] >= 0.03,
}
strict_pass = bool(len(eligible) and all(checks.values()))
output = base_frame[["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE, "exact_v76"]].copy()
output["uniform_top33"] = baseline_prediction.astype(np.float32)
output["spatial_dip_consensus"] = selected_prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "fold3_spatial_dip_oof.parquet", index=False, compression="zstd")
np.savez_compressed(
    OUT_DIR / "selected_candidate_indices.npz",
    wells=np.asarray(wells), indices=selection_cache[selected_recipe],
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_transductive_discovery_confirmation",
    "candidate_generation_uses_truth": False,
    "cross_well_information": "nearby-well low-frequency candidate dip only",
    "curve_transfer": False,
    "residual_target_training": False,
    "coverage": {"wells": len(wells), "rows": len(base_frame)},
    "search_count": len(recipes),
    "selected_recipe": {
        "point": selected_recipe[0], "neighbors": selected_recipe[1],
        "shape_nodes": selected_recipe[2], "support": selected_recipe[3],
        "penalty": selected_recipe[4], "output_k": OUTPUT_K,
        "freeze_ft": FREEZE_FT, "ramp_ft": RAMP_FT, "cap_ft": CAP_FT,
    },
    "discovery": {
        "gain": float(selected_row["gain"]), "far_gain": float(selected_row["far_gain"]),
        "near_degradation": float(selected_row["near_degradation"]),
        "p95_gain": float(selected_row["p95_gain"]),
    },
    "confirmation": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> cross-fit the spatial-dip graph on Folds1/2/3 and build deployment"
        if strict_pass else
        "FAIL -> spatial dip does not identify candidate modes; close cross-well graph branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV131 DISCOVERY TOP 20")
print(scan.head(20).to_string(index=False), flush=True)
print("\nV131 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV131 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
