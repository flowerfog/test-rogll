from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HEIGHT, STATE_AXIS, WIDTH, XY
from .data import Well, _bin_last, _repeat_col, _repeat_row, _take, _window_indices


@dataclass
class XYResult:
    channels: dict[str, np.ndarray]
    stats: dict[str, float]
    prediction: np.ndarray


def anisotropic_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(XY.anisotropy_angle_deg)
    along = np.cos(angle) * x + np.sin(angle) * y
    across = (-np.sin(angle) * x + np.cos(angle) * y) * XY.anisotropy_ratio
    return np.c_[along, across]


def _fill_path(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        raise ValueError("trajectory coordinate has no finite values")
    return np.interp(np.arange(len(values)), valid, values[valid])


def _sample_support(wells: list[Well]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, surfaces, owners = [], [], []
    for owner, well in enumerate(wells):
        hz = well.horizontal
        rows = np.arange(0, len(hz), XY.train_bin_size)
        points.append(hz[["X", "Y"]].to_numpy(float)[rows])
        surfaces.append((hz.TVT.to_numpy(float) + hz.Z.to_numpy(float))[rows])
        owners.append(np.full(len(rows), owner, int))
    return np.vstack(points), np.concatenate(surfaces), np.concatenate(owners)


def build_xy_features(query: Well, support_wells: list[Well]) -> XYResult:
    centers = np.array([[w.horizontal.X.mean(), w.horizontal.Y.mean()] for w in support_wells])
    query_center = np.array([query.horizontal.X.mean(), query.horizontal.Y.mean()])
    center_distance = np.linalg.norm(
        anisotropic_xy(centers[:, 0] - query_center[0], centers[:, 1] - query_center[1]), axis=1)
    selected_wells = np.argsort(center_distance)[: XY.neighbor_wells]
    selected = [support_wells[i] for i in selected_wells]
    points, surfaces, _ = _sample_support(selected)
    transformed = anisotropic_xy(points[:, 0], points[:, 1])
    indices = _window_indices(query)
    query_xy_raw = np.c_[_take(query.horizontal.X.to_numpy(float), indices), _take(query.horizontal.Y.to_numpy(float), indices)]
    query_z_raw = _take(query.horizontal.Z.to_numpy(float), indices)
    query_x = _fill_path(_bin_last(query_xy_raw[:, 0]))
    query_y = _fill_path(_bin_last(query_xy_raw[:, 1]))
    query_z = _fill_path(_bin_last(query_z_raw))
    prediction = np.full(WIDTH, np.nan, np.float32)
    all_distances: list[np.ndarray] = []
    radial_scores, alignments = [], []
    support_center = transformed.mean(0)
    support_radius = np.mean(np.linalg.norm(transformed - support_center, axis=1)) + 1e-6
    previous_gradient = None
    for column in range(0, WIDTH, max(1, XY.query_bin_size // 32)):
        q = anisotropic_xy(np.array([query_x[column]]), np.array([query_y[column]]))[0]
        distance = np.linalg.norm(transformed - q, axis=1)
        nearest = np.argsort(distance)[: XY.point_neighbors]
        dxy = transformed[nearest] - q
        scale = np.maximum(np.std(dxy, axis=0), 1.0)
        design = np.c_[np.ones(len(nearest)), dxy / scale]
        weights = 1.0 / np.maximum(distance[nearest], 1.0) ** XY.idw_power
        matrix = design.T @ (weights[:, None] * design)
        eigen = np.linalg.eigvalsh(matrix[1:, 1:])
        ridge = XY.ridge_fraction * np.trace(matrix) / 3
        if eigen[0] / max(eigen[-1], 1e-12) < XY.min_eigen_ratio:
            ridge *= XY.min_eigen_ratio / max(eigen[0] / max(eigen[-1], 1e-12), 1e-3)
        target = surfaces[nearest]
        beta = np.linalg.solve(matrix + ridge * np.eye(3), design.T @ (weights * target))
        residual = target - design @ beta
        robust_scale = max(np.median(np.abs(residual)) / .6745, XY.tukey_scale)
        u = residual / (4.685 * robust_scale)
        tukey = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0)
        robust_weights = weights * tukey
        matrix = design.T @ (robust_weights[:, None] * design)
        beta = np.linalg.solve(matrix + ridge * np.eye(3), design.T @ (robust_weights * target))
        gradient = np.clip(beta[1:] / scale, -XY.max_gradient, XY.max_gradient)
        prediction[column] = beta[0] - query_z[column]
        all_distances.append(distance[nearest])
        radial_scores.append(np.linalg.norm(q - support_center) / support_radius)
        if previous_gradient is not None:
            alignments.append(float(np.dot(gradient, previous_gradient) /
                                    (np.linalg.norm(gradient) * np.linalg.norm(previous_gradient) + 1e-9)))
        previous_gradient = gradient
    valid = np.flatnonzero(np.isfinite(prediction))
    prediction = np.interp(np.arange(WIDTH), valid, prediction[valid]).astype(np.float32)
    visible = _bin_last(_take(query.horizontal.TVT_input.to_numpy(float), indices))
    prefix = np.isfinite(visible)
    if prefix.any():
        correction = np.nanmedian(visible[prefix][-16:] - prediction[prefix][-16:])
        prediction += correction
    prediction_rel = prediction - query.anchor_tvt
    prediction_diff = np.diff(prediction, prepend=query.anchor_tvt)
    state = np.asarray(STATE_AXIS, np.float32)[:, None]
    distance_flat = np.concatenate(all_distances)
    prefix_weight = np.sum(1 / np.maximum(distance_flat[: min(len(distance_flat), 512)], 1) ** 2)
    total_weight = np.sum(1 / np.maximum(distance_flat, 1) ** 2)
    stats = {
        "geo_nbr_distance": float(np.mean(distance_flat)),
        "geo_nbr_distance_q10": float(np.quantile(distance_flat, .1)),
        "geo_nbr_path_alignment": float(np.nanmean(alignments)) if alignments else 1.0,
        "geo_radial_extrap_score": float(np.mean(radial_scores)),
        "geo_prefix_weight_ratio": float(prefix_weight / max(total_weight, 1e-12)),
    }
    channels = {
        # The writeup distinguishes local TVT differences from accumulated
        # TVT. Members without accumulated PF evidence use the former; the
        # final PF+XY member carries both representations.
        "geo_tvt_diff": _repeat_row(prediction_diff / 100.0),
        "geo_tvt_rel": _repeat_row(prediction_rel / 100.0),
        "geo_tvt_abs_diff": (np.abs(state - prediction_rel[None]) / 100.0).astype(np.float32),
    }
    return XYResult(channels, stats, prediction)


def xy_is_safe(stats: dict[str, float]) -> bool:
    return (stats["geo_nbr_distance"] < 2459 and stats["geo_nbr_distance_q10"] < 1426
            and stats["geo_nbr_path_alignment"] > .96
            and stats["geo_prefix_weight_ratio"] < .315)
