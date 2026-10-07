from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


NEIGHBOR_WELLS = 12
POINT_NEIGHBORS = 112
IDW_POWER = 2.0
EPS = 1e-6
ANGLE_DEG = 50.0
ANISOTROPY_RATIO = 1.3


def transform(xy: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(ANGLE_DEG)
    c, s = np.cos(angle), np.sin(angle)
    along = xy[:, 0] * c + xy[:, 1] * s
    cross = -xy[:, 0] * s + xy[:, 1] * c
    return np.column_stack([along / ANISOTROPY_RATIO, cross])


def discover(path: Path) -> list[str]:
    return sorted(file.name.removesuffix("__horizontal_well.csv") for file in path.glob("*__horizontal_well.csv"))


def read_well(path: Path, well: str) -> dict:
    frame = pd.read_csv(path / f"{well}__horizontal_well.csv", usecols=["X", "Y", "TVT_input"])
    xy = frame[["X", "Y"]].to_numpy(np.float64)
    visible = frame["TVT_input"].notna().to_numpy()
    suffix_start = int(np.flatnonzero(~visible)[0]) if (~visible).any() else len(frame)
    return {"well": well, "xy": xy, "suffix_start": suffix_start, "centroid": xy.mean(axis=0)}


def support_points(record: dict, prefix_only: bool) -> np.ndarray:
    end = record["suffix_start"] if prefix_only else len(record["xy"])
    start = max(0, end - 512) if prefix_only else 0
    return record["xy"][start:end:64]


def score_query(query: dict, supports: dict[str, dict]) -> dict:
    query_xy = query["xy"][query["suffix_start"]::32]
    if len(query_xy) == 0:
        return {name: np.nan for name in (
            "mean_neighbor_distance", "p10_neighbor_distance",
            "prefix_visible_neighbor_weight_ratio", "query_center_distance_ratio",
        )}
    candidates = list(supports.values())
    centers = transform(np.stack([record["centroid"] for record in candidates]))
    q_center = transform(query["centroid"][None])[0]
    nearest = np.argsort(np.linalg.norm(centers - q_center, axis=1))[:NEIGHBOR_WELLS]
    other = np.concatenate([support_points(candidates[i], False) for i in nearest], axis=0)
    own = support_points(query, True)
    points = np.concatenate([other, own], axis=0)
    is_prefix = np.concatenate([np.zeros(len(other), bool), np.ones(len(own), bool)])
    transformed = transform(points)
    tree = cKDTree(transformed)
    dist, idx = tree.query(transform(query_xy), k=min(POINT_NEIGHBORS, len(points)))
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]
    weights = 1.0 / np.maximum(dist, EPS) ** IDW_POWER
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    neighbor_xy = transformed[idx]
    mean_xy = (neighbor_xy * weights[:, :, None]).sum(axis=1)
    spread = np.sqrt((weights * np.square(neighbor_xy - mean_xy[:, None, :]).sum(axis=2)).sum(axis=1))
    radial = np.linalg.norm(transform(query_xy) - mean_xy, axis=1) / np.maximum(spread, EPS)
    prefix_weight = (weights * is_prefix[idx]).sum(axis=1)
    other_weight = (weights * ~is_prefix[idx]).sum(axis=1)
    return {
        "mean_neighbor_distance": float(np.mean(dist)),
        "p10_neighbor_distance": float(np.quantile(dist, 0.10)),
        "prefix_visible_neighbor_weight_ratio": float(np.mean(prefix_weight / np.maximum(other_weight, EPS))),
        "query_center_distance_ratio": float(np.mean(radial)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the four disclosed XY reliability statistics.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_path, test_path = args.data_root / "train", args.data_root / "test"
    train = {well: read_well(train_path, well) for well in discover(train_path)}
    test = {well: read_well(test_path, well) for well in discover(test_path)}
    rows = []
    for index, (well, query) in enumerate(train.items(), 1):
        supports = {name: record for name, record in train.items() if name != well}
        rows.append({"well": well, "split": "train", **score_query(query, supports)})
        if index % 50 == 0:
            print(f"XY QUALITY train {index}/{len(train)}", flush=True)
    for well, query in test.items():
        rows.append({"well": well, "split": "test", **score_query(query, train)})
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    report = {
        "rows": len(output), "train_wells": len(train), "test_wells": len(test),
        "neighbor_wells": NEIGHBOR_WELLS, "point_neighbors": POINT_NEIGHBORS,
        "idw_power": IDW_POWER, "anisotropy_angle_deg": ANGLE_DEG,
        "anisotropy_ratio": ANISOTROPY_RATIO,
        "aggregation_status": "inferred because the write-up did not disclose row-to-well aggregation",
    }
    print("XY QUALITY", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
