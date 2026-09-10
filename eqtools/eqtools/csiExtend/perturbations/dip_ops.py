"""Low-level geometry generators used after dip-profile resolution.

Profile declaration, projection, sampled/fixed mapping, transitions, and
one-dimensional interpolation live in :mod:`eqtools.csiExtend.dip_profile`.
This module intentionally contains only the two geometry primitives that turn
the resolved top-edge dip into strike and bottom coordinates.
"""

from __future__ import annotations

import numpy as np

from ..geom_ops import validate_top_bottom_cells


def generate_bottom_from_dips(
    top_coords: np.ndarray,
    dip_deg: np.ndarray,
    strike_deg: np.ndarray,
    fault_depth: float,
    fault_top: float,
    use_average_strike: bool = False,
    average_strike_source: str = "pca",
    user_direction_angle: float | None = None,
    interpolation_axis: str = "auto",
    verbose: bool = False,
) -> np.ndarray:
    """Compute bottom coordinates from aligned top, dip, and strike arrays.

    For vertical depth span ``h = fault_depth - fault_top`` and signed dip
    ``delta``, the down-dip width is ``h / sin(abs(delta))``. Negative dip is
    represented locally as ``strike + 180 degrees`` with positive magnitude.
    Row ``i`` of the bottom is always generated from row ``i`` of the top;
    neither boundary is sorted independently.

    ``interpolation_axis`` is retained as descriptive generator metadata. All
    profile interpolation has already completed before this function runs.
    """
    from numpy import cos, deg2rad, sin

    from ..DipInterpolation import (
        normalize_dip_to_neg90_90,
        validate_dip_angles_for_depth_projection,
    )

    top_coords = np.asarray(top_coords, dtype=float)
    dip_deg = np.asarray(dip_deg, dtype=float)
    strike_deg = np.asarray(strike_deg, dtype=float)
    if top_coords.ndim != 2 or top_coords.shape[1] < 2:
        raise ValueError("top_coords must have shape (n, 2+)")
    n = len(top_coords)
    if dip_deg.shape != (n,) or strike_deg.shape != (n,):
        raise ValueError("dip_deg and strike_deg must align one-to-one with top_coords")

    x = top_coords[:, 0]
    y = top_coords[:, 1]
    strike_rad = deg2rad(strike_deg.copy())
    dip_signed = normalize_dip_to_neg90_90(
        validate_dip_angles_for_depth_projection(dip_deg, name="dip")
    )
    dip_rad = deg2rad(dip_signed)

    strike_direction = np.array([x[-1] - x[0], y[-1] - y[0]])
    if use_average_strike:
        if average_strike_source == "pca":
            from sklearn.decomposition import PCA

            pca = PCA(n_components=1)
            pca.fit(np.column_stack([x, y]))
            principal = pca.components_[0]
            average_rad = np.pi / 2.0 - np.arctan2(principal[1], principal[0])
            if np.dot(principal, strike_direction) < 0:
                average_rad += np.pi
            strike_rad = np.full(n, average_rad)
        elif average_strike_source == "user" and user_direction_angle is not None:
            average_rad = deg2rad(user_direction_angle)
            user_vector = np.array([sin(average_rad), cos(average_rad)])
            direction_norm = np.linalg.norm(strike_direction)
            alignment = np.dot(user_vector, strike_direction)
            if (
                not np.isfinite(average_rad)
                or direction_norm <= np.finfo(float).eps
                or alignment <= np.finfo(float).eps * direction_norm
            ):
                raise ValueError(
                    "user_direction_angle must follow the positive trace "
                    "direction (geographic azimuth clockwise from North)"
                )
            strike_rad = np.full(n, average_rad)
        else:
            raise ValueError(
                "average_strike_source must be 'pca', or 'user' with a "
                "user_direction_angle"
            )
        if verbose:
            print(f"Average strike direction: {np.rad2deg(strike_rad[0]):.2f}")

    negative = dip_rad < 0.0
    strike_rad[negative] += np.pi
    dip_rad[negative] *= -1.0
    strike_rad = np.mod(strike_rad, 2.0 * np.pi)

    width = ((fault_depth - fault_top) / sin(dip_rad)).reshape(-1, 1)
    old_coords = np.column_stack([x, y, np.full(n, fault_top)])
    dip_vector = np.column_stack([
        cos(dip_rad) * cos(-strike_rad),
        cos(dip_rad) * sin(-strike_rad),
        sin(dip_rad),
    ])
    bottom_coords = old_coords + dip_vector * width
    validate_top_bottom_cells(old_coords, bottom_coords)
    return bottom_coords


def compute_strike(coords_xy: np.ndarray) -> np.ndarray:
    """Return per-node along-strike geographic azimuth in degrees."""
    from numpy import arctan2, concatenate, cos, deg2rad, diff, rad2deg, sin

    coords_xy = np.asarray(coords_xy, dtype=float)
    if coords_xy.ndim != 2 or coords_xy.shape[0] < 2 or coords_xy.shape[1] < 2:
        raise ValueError("coords_xy must have shape (n>=2, 2+)")
    x, y = coords_xy[:, 0], coords_xy[:, 1]
    segment_strike = 90.0 - rad2deg(arctan2(diff(y), diff(x)))
    segment_rad = deg2rad(segment_strike)
    average_rad = arctan2(
        (sin(segment_rad[:-1]) + sin(segment_rad[1:])) / 2.0,
        (cos(segment_rad[:-1]) + cos(segment_rad[1:])) / 2.0,
    )
    return concatenate((
        [segment_strike[0]],
        rad2deg(average_rad),
        [segment_strike[-1]],
    ))


__all__ = ["compute_strike", "generate_bottom_from_dips"]
