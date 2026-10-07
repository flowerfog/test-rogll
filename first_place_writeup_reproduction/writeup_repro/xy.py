from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .data import PanelSample, WellRecord, list_wells, load_record
from .inferred_defaults import XYDefaults


@dataclass(frozen=True)
class XYResult:
    tvt_offset: np.ndarray
    mean_distance: float
    p10_distance: float
    prefix_weight_ratio: float
    center_distance_ratio: float
    reliable: bool


class XYSurfacePrior:
    """Leave-one-well-out robust local WLS for S = TVT + Z + constant."""

    def __init__(self, defaults: XYDefaults = XYDefaults()):
        self.defaults = defaults
        self.points: np.ndarray | None = None
        self.surface: np.ndarray | None = None
        self.wells: np.ndarray | None = None
        self.centers: dict[str, np.ndarray] = {}
        self.tree: cKDTree | None = None
        self.quality_thresholds: dict[str, float] = {}

    def fit_records(self, records: list[WellRecord], points_per_well: int = 32) -> "XYSurfacePrior":
        points, surfaces, wells = [], [], []
        for record in records:
            visible = np.flatnonzero(np.isfinite(record.tvt_input) & np.isfinite(record.z))
            if not len(visible):
                continue
            chosen = visible[np.linspace(0, len(visible) - 1, min(points_per_well, len(visible))).astype(int)]
            xy = np.column_stack([record.x[chosen], record.y[chosen]])
            points.append(xy)
            surfaces.append(record.tvt_input[chosen] + record.z[chosen])
            wells.extend([record.well] * len(chosen))
            self.centers[record.well] = np.nanmean(xy, axis=0)
        self.points = np.concatenate(points).astype(np.float64)
        self.surface = np.concatenate(surfaces).astype(np.float64)
        self.wells = np.asarray(wells)
        self.tree = cKDTree(self.points)
        return self

    def fit_root(self, data_root: str | Path, well_ids: list[str] | None = None) -> "XYSurfacePrior":
        well_ids = well_ids or list_wells(data_root, "train")
        return self.fit_records([load_record(data_root, "train", well) for well in well_ids])

    def _local_plane(self, query: np.ndarray, exclude_well: str | None) -> tuple[float, np.ndarray, np.ndarray]:
        if self.tree is None or self.points is None or self.surface is None or self.wells is None:
            raise RuntimeError("XYSurfacePrior is not fitted")
        candidate_count = min(len(self.points), self.defaults.neighbors * 6)
        distance, index = self.tree.query(query, k=candidate_count)
        distance, index = np.atleast_1d(distance), np.atleast_1d(index)
        keep = np.ones(len(index), dtype=bool)
        if exclude_well is not None:
            keep &= self.wells[index] != exclude_well
        index, distance = index[keep][: self.defaults.neighbors], distance[keep][: self.defaults.neighbors]
        if len(index) < 3:
            return np.nan, distance, index
        delta = self.points[index] - query
        covariance = np.cov(delta.T) if len(delta) > 2 else np.eye(2)
        eigenvalue, eigenvector = np.linalg.eigh(covariance + np.eye(2) * 1e-6)
        transformed = delta @ eigenvector
        transformed[:, 0] /= self.defaults.anisotropy
        anisotropic = np.sqrt(np.sum(transformed**2, axis=1))
        scale = max(float(np.median(anisotropic)), self.defaults.distance_floor)
        base_weight = np.exp(-0.5 * (anisotropic / scale) ** 2) / np.maximum(
            anisotropic, self.defaults.distance_floor
        )
        design = np.column_stack([np.ones(len(index)), delta])
        weight = base_weight.copy()
        coefficient = np.array([np.average(self.surface[index], weights=weight), 0.0, 0.0])
        for _ in range(self.defaults.robust_iterations):
            lhs = design.T @ (weight[:, None] * design)
            lhs += np.diag([self.defaults.ridge * 0.01, self.defaults.ridge, self.defaults.ridge])
            rhs = design.T @ (weight * self.surface[index])
            try:
                coefficient = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                coefficient = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            residual = self.surface[index] - design @ coefficient
            mad = max(float(np.median(np.abs(residual - np.median(residual))) * 1.4826), 0.5)
            robust = np.minimum(1.0, 1.5 * mad / np.maximum(np.abs(residual), 1e-6))
            weight = base_weight * robust
        return float(coefficient[0]), distance, index

    def predict(self, record: WellRecord, sample: PanelSample) -> XYResult:
        grid_h = sample.values.shape[1]
        rows = sample.selected_rows.reshape(grid_h, -1)
        path = np.zeros(grid_h, dtype=np.float32)
        all_distance = []
        center_ratios = []
        anchor_surface = record.anchor_tvt + float(record.z[record.anchor_index])
        for bin_index, group in enumerate(rows):
            group = group[group >= 0]
            if not len(group):
                path[bin_index] = path[bin_index - 1] if bin_index else 0.0
                continue
            query = np.array([np.nanmean(record.x[group]), np.nanmean(record.y[group])], dtype=np.float64)
            surface, distance, neighbor_index = self._local_plane(query, record.well)
            z = float(np.nanmean(record.z[group]))
            path[bin_index] = 0.0 if not np.isfinite(surface) else surface - z - record.anchor_tvt
            if len(distance):
                all_distance.extend(distance.tolist())
            if len(neighbor_index) and self.wells is not None:
                neighbor_centers = np.stack([self.centers[str(well)] for well in self.wells[neighbor_index]])
                denominator = np.maximum(
                    np.linalg.norm(self.points[neighbor_index] - neighbor_centers, axis=1).mean(), 1.0
                )
                center_ratios.append(float(np.linalg.norm(query - neighbor_centers, axis=1).mean() / denominator))
        distances = np.asarray(all_distance or [np.inf])
        mean_distance = float(np.mean(distances))
        p10_distance = float(np.percentile(distances, 10))
        anchor_query = np.array([record.x[record.anchor_index], record.y[record.anchor_index]])
        predicted_anchor, _, _ = self._local_plane(anchor_query, record.well)
        prefix_error = abs(predicted_anchor - anchor_surface) if np.isfinite(predicted_anchor) else np.inf
        prefix_weight_ratio = float(np.exp(-prefix_error / 20.0))
        center_ratio = float(np.mean(center_ratios or [np.inf]))
        reliable = bool(
            np.isfinite(mean_distance)
            and np.isfinite(center_ratio)
            and prefix_weight_ratio >= np.exp(-1.0)
        )
        return XYResult(path, mean_distance, p10_distance, prefix_weight_ratio, center_ratio, reliable)
