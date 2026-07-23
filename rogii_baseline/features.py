"""Inference-available feature construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "delta_md",
    "delta_x",
    "delta_y",
    "delta_z",
    "horizontal_offset",
    "spatial_offset",
    "toe_progress",
    "x",
    "y",
    "z",
    "anchor_tvt",
    "geometry_tvt",
    "md_extrapolation_minus_geometry",
    "direction_x",
    "direction_y",
    "step_dx_dmd",
    "step_dy_dmd",
    "step_dz_dmd",
    "prefix_tvt_dmd_slope",
    "prefix_tvt_z_slope",
    "gr",
    "gr_missing",
    "trailing_gr",
    "gr_minus_anchor",
    "typewell_gr_at_geometry",
    "gr_minus_typewell",
    "typewell_gr_gradient",
    "typewell_position",
)


@dataclass(frozen=True)
class FeatureSet:
    x: np.ndarray
    baseline_tvt: np.ndarray
    row_indices: np.ndarray
    target_residual: np.ndarray | None


def _linear_slope(x: np.ndarray, y: np.ndarray, fallback: float) -> float:
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-10:
        return fallback
    slope = float(np.dot(centered, y - y.mean()) / denominator)
    return slope if np.isfinite(slope) else fallback


def _step_rate(values: np.ndarray, md: np.ndarray) -> np.ndarray:
    numerator = np.diff(values, prepend=values[0])
    denominator = np.diff(md, prepend=md[0])
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=np.abs(denominator) > 1e-10,
    )


def _reference_curve(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Duplicate TVT samples are averaged so interpolation remains deterministic.
    reference = (
        typewell.loc[:, ["TVT", "GR"]]
        .groupby("TVT", as_index=False, sort=True)["GR"]
        .mean()
    )
    ref_tvt = reference["TVT"].to_numpy(dtype=float)
    ref_gr = reference["GR"].to_numpy(dtype=float)
    if len(ref_tvt) < 2:
        raise ValueError("Typewell needs at least two distinct TVT values")
    ref_gradient = np.gradient(ref_gr, ref_tvt)
    return ref_tvt, ref_gr, ref_gradient


def build_features(
    horizontal: pd.DataFrame, typewell: pd.DataFrame, *, include_target: bool
) -> FeatureSet:
    """Build features for masked rows without consulting horizontal target TVT."""
    md = horizontal["MD"].to_numpy(dtype=float)
    x_coord = horizontal["X"].to_numpy(dtype=float)
    y_coord = horizontal["Y"].to_numpy(dtype=float)
    z_coord = horizontal["Z"].to_numpy(dtype=float)
    gr = horizontal["GR"].to_numpy(dtype=float)
    tvt_input = horizontal["TVT_input"].to_numpy(dtype=float)

    observed = np.isfinite(tvt_input)
    row_indices = np.flatnonzero(~observed)
    if not len(row_indices):
        raise ValueError("No masked suffix rows were found")
    anchor_index = int(np.flatnonzero(observed)[-1])
    anchor_tvt = float(tvt_input[anchor_index])

    prefix_start = max(0, anchor_index - 255)
    prefix = slice(prefix_start, anchor_index + 1)
    md_slope = np.clip(
        _linear_slope(md[prefix], tvt_input[prefix], fallback=0.0), -5.0, 5.0
    )
    z_slope = np.clip(
        _linear_slope(z_coord[prefix], tvt_input[prefix], fallback=-1.0), -5.0, 5.0
    )

    delta_md = md - md[anchor_index]
    delta_x = x_coord - x_coord[anchor_index]
    delta_y = y_coord - y_coord[anchor_index]
    delta_z = z_coord - z_coord[anchor_index]
    horizontal_offset = np.hypot(delta_x, delta_y)
    spatial_offset = np.sqrt(horizontal_offset**2 + delta_z**2)
    toe_span = max(float(delta_md[-1]), 1.0)
    toe_progress = np.clip(delta_md / toe_span, 0.0, 1.0)

    geometry_tvt = anchor_tvt + z_slope * delta_z
    md_extrapolation = anchor_tvt + md_slope * delta_md
    direction_x = np.divide(
        delta_x,
        horizontal_offset,
        out=np.zeros_like(delta_x),
        where=horizontal_offset > 1e-10,
    )
    direction_y = np.divide(
        delta_y,
        horizontal_offset,
        out=np.zeros_like(delta_y),
        where=horizontal_offset > 1e-10,
    )

    finite_prefix_gr = gr[: anchor_index + 1][np.isfinite(gr[: anchor_index + 1])]
    anchor_gr = float(finite_prefix_gr[-1]) if len(finite_prefix_gr) else 0.0
    trailing_gr = (
        pd.Series(gr).rolling(window=31, min_periods=1).mean().to_numpy(dtype=float)
    )

    ref_tvt, ref_gr, ref_gradient = _reference_curve(typewell)
    ref_at_geometry = np.interp(geometry_tvt, ref_tvt, ref_gr)
    ref_slope = np.interp(geometry_tvt, ref_tvt, ref_gradient)
    gr_filled = np.where(np.isfinite(gr), gr, ref_at_geometry)
    trailing_filled = np.where(np.isfinite(trailing_gr), trailing_gr, ref_at_geometry)
    ref_span = max(float(ref_tvt[-1] - ref_tvt[0]), 1.0)
    ref_position = (geometry_tvt - ref_tvt[0]) / ref_span

    all_features = np.column_stack(
        [
            delta_md,
            delta_x,
            delta_y,
            delta_z,
            horizontal_offset,
            spatial_offset,
            toe_progress,
            x_coord,
            y_coord,
            z_coord,
            np.full(len(md), anchor_tvt),
            geometry_tvt,
            md_extrapolation - geometry_tvt,
            direction_x,
            direction_y,
            _step_rate(x_coord, md),
            _step_rate(y_coord, md),
            _step_rate(z_coord, md),
            np.full(len(md), md_slope),
            np.full(len(md), z_slope),
            gr_filled,
            ~np.isfinite(gr),
            trailing_filled,
            gr_filled - anchor_gr,
            ref_at_geometry,
            gr_filled - ref_at_geometry,
            ref_slope,
            ref_position,
        ]
    )[row_indices]
    if all_features.shape[1] != len(FEATURE_NAMES):
        raise AssertionError("Feature name and matrix widths differ")
    if not np.isfinite(all_features).all():
        raise ValueError("Feature construction produced non-finite values")

    baseline = geometry_tvt[row_indices]
    target_residual = None
    if include_target:
        # Access to the horizontal target is deliberately isolated to labels.
        target = horizontal["TVT"].to_numpy(dtype=float)[row_indices]
        target_residual = target - baseline

    return FeatureSet(
        x=all_features.astype(np.float32),
        baseline_tvt=baseline.astype(np.float64),
        row_indices=row_indices,
        target_residual=(
            target_residual.astype(np.float64) if target_residual is not None else None
        ),
    )

