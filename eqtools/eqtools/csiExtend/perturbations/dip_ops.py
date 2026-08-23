"""
Pure functions for dip perturbation operations.

These functions contain the core mathematics extracted from
AdaptiveTriangularPatches methods, with no ``self`` dependency.
They accept arrays and scalars, return arrays — no side effects.

Used by DipGeneratorStage (future) and testable in isolation.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from ..geom_ops import validate_top_bottom_cells
from .angle_utils import angles_to_degrees


# ---------------------------------------------------------------------------
# 1. perturb_dip_values
# ---------------------------------------------------------------------------

def perturb_dip_values(
    dips: np.ndarray,
    perturbations: np.ndarray,
    fixed_nodes: list | None = None,
    angle_unit: str = 'degrees',
) -> np.ndarray:
    """Apply additive perturbations in continuous 0--180 dip space.

    Reference dips may use either the recommended continuous ``(0, 180)``
    representation or the historical signed representation
    ``[-90, 0) U (0, 90]``.  They are validated and converted to ``(0, 180)``
    *before* perturbations are added.  Consequently equivalent references such
    as ``100`` and ``-80`` produce the same proposal geometry.

    Parameters
    ----------
    dips : (n,) array
        Reference dip values in degrees. Accepted representations are
        ``[-90, 0) U (0, 180)``; zero and 180 degrees are invalid for a
        depth-projected fault surface.
    perturbations : array-like
        Perturbation increments (scalar broadcasts to all movable nodes).
        Units determined by *angle_unit*.
    fixed_nodes : list[int] or None
        Indices of nodes that should not be perturbed.
    angle_unit : str
        Unit of *perturbations* (``'degrees'`` or ``'radians'``).

    Returns
    -------
    (n,) ndarray
        Perturbed dips in the continuous ``(0, 180)`` proposal coordinate.

    Raises
    ------
    ValueError
        If a reference dip is invalid, the perturbation count is inconsistent,
        or a resulting proposal leaves the open interval ``(0, 180)``.
    """
    from ..DipInterpolation import (
        normalize_dip_to_0_180,
        validate_dip_angles_for_depth_projection,
    )

    reference = validate_dip_angles_for_depth_projection(
        np.asarray(dips, dtype=float),
        name='reference dip',
    )
    result = np.asarray(normalize_dip_to_0_180(reference), dtype=float).copy()
    if fixed_nodes is None:
        fixed_nodes = []

    movable = [i for i in range(len(result)) if i not in fixed_nodes]

    perts = np.asarray(angles_to_degrees(perturbations, angle_unit), dtype=float).ravel()
    if len(movable) == 0:
        return result
    if perts.size == 1:
        perts = np.full(len(movable), perts.item(), dtype=float)
    elif perts.size != len(movable):
        raise ValueError(
            f"perturbations must be scalar or match movable node count "
            f"({len(movable)}); got {perts.size}."
        )

    result[movable] += perts

    invalid = (
        ~np.isfinite(result)
        | (result <= 0.0)
        | (result >= 180.0)
    )
    if np.any(invalid):
        indices = np.argwhere(invalid).reshape(-1)[:5].tolist()
        raise ValueError(
            "perturbed dip must remain finite and in (0, 180) degrees "
            "after converting reference dips to the continuous proposal "
            f"coordinate; invalid value indices: {indices}"
        )
    return result


# ---------------------------------------------------------------------------
# 2. determine_interpolation_axis
# ---------------------------------------------------------------------------

def determine_interpolation_axis(
    x: np.ndarray,
    y: np.ndarray,
    verbose: bool = False,
) -> str:
    """Select optimal interpolation axis via PCA.

    Parameters
    ----------
    x, y : (n,) arrays
        Projected coordinates (UTM).
    verbose : bool
        Print diagnostic info.

    Returns
    -------
    str
        ``'x'`` or ``'y'``.
    """
    from sklearn.decomposition import PCA

    coords = np.column_stack([x, y])
    pca = PCA(n_components=1)
    pca.fit(coords)

    principal = pca.components_[0]
    ratio = pca.explained_variance_ratio_[0]

    if abs(principal[0]) > abs(principal[1]):
        axis = 'x'
        if verbose:
            print(f"PCA principal component aligns more with x-axis (variance: {ratio:.3f})")
    else:
        axis = 'y'
        if verbose:
            print(f"PCA principal component aligns more with y-axis (variance: {ratio:.3f})")

    return axis


# ---------------------------------------------------------------------------
# 3. interpolate_dip_onto_coords
# ---------------------------------------------------------------------------

def interpolate_dip_onto_coords(
    control_xy_dip: np.ndarray,
    target_coords: np.ndarray,
    interpolation_axis: str = 'auto',
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate dip values from control points onto target coordinates
    and compute per-node strike.

    Parameters
    ----------
    control_xy_dip : (n_ctrl, 3) array
        Columns ``[x_utm, y_utm, dip_degrees]``. Recommended dip form is
        signed ``[-90, 0) U (0, 90]``; ``(90, 180)`` is accepted as the
        equivalent opposite-side compatibility form.
    target_coords : (n_target, 2+) array
        Target coordinates in UTM (at least columns 0=x, 1=y).
    interpolation_axis : str
        ``'auto'``, ``'x'``, ``'y'``, or ``'arc_length'``.

        - ``'x'`` / ``'y'``: project onto the chosen axis and interpolate 1D.
        - ``'auto'``: PCA-select the dominant axis, then same as ``'x'``/``'y'``;
          it never selects ``'arc_length'``.
        - ``'arc_length'``: interpolate along the cumulative arc-length of
          *target_coords*. Control points are projected onto the trace to
          determine their arc-length positions. This is more physically
          meaningful for curved faults where axis-projection distorts distance.
          Values outside the outer control positions retain the nearest endpoint
          dip. Controls should project to distinct positions on the intended
          trace branch.
    verbose : bool
        Print diagnostic info.

    Returns
    -------
    interpolated_dip : (n_target,) ndarray
        Signed dip in ``[-90, 0) U (0, 90]`` at each target node.
    strike : (n_target,) ndarray
        Along-strike azimuth in degrees at each target node.
    """
    from ..DipInterpolation import (
        normalize_dip_to_0_180,
        normalize_dip_to_neg90_90,
        validate_dip_angles_for_depth_projection,
    )

    control_xy_dip = np.asarray(control_xy_dip, dtype=float).copy()
    valid_axes = {'auto', 'x', 'y', 'arc_length'}
    if interpolation_axis not in valid_axes:
        raise ValueError(
            "interpolation_axis must be one of 'auto', 'x', 'y', or "
            f"'arc_length'; got {interpolation_axis!r}"
        )
    validated_dip = validate_dip_angles_for_depth_projection(
        control_xy_dip[:, 2],
        name='control dip',
    )
    # Interpolate through the vertical orientation (90 deg), not through the
    # singular horizontal orientation (0 deg), when dip side changes.
    control_xy_dip[:, 2] = normalize_dip_to_0_180(validated_dip)

    if interpolation_axis == 'arc_length':
        interpolated_dip = _interpolate_dip_arc_length(
            control_xy_dip, target_coords, verbose=verbose)
    else:
        if interpolation_axis == 'auto':
            interpolation_axis = determine_interpolation_axis(
                target_coords[:, 0], target_coords[:, 1], verbose=verbose)
            if verbose:
                print(f"Auto-selected interpolation axis: {interpolation_axis}")

        axis_idx = 0 if interpolation_axis == 'x' else 1

        ctrl_axis = control_xy_dip[:, axis_idx]
        ctrl_dip = control_xy_dip[:, 2]

        order = np.argsort(ctrl_axis)
        sorted_axis = ctrl_axis[order]
        sorted_dip = ctrl_dip[order]

        interp_fn = interp1d(
            sorted_axis, sorted_dip,
            fill_value=(sorted_dip[0], sorted_dip[-1]),
            bounds_error=False,
        )
        interpolated_dip = interp_fn(target_coords[:, axis_idx])

    strike = compute_strike(target_coords)

    interpolated_dip = normalize_dip_to_neg90_90(interpolated_dip)

    return interpolated_dip, strike


def _interpolate_dip_arc_length(
    control_xy_dip: np.ndarray,
    target_coords: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Interpolate dip along the cumulative arc-length of the target trace.

    Each control point is projected onto the nearest segment of target_coords
    to determine its arc-length position. Then 1D interpolation is performed
    in arc-length space.
    """
    from scipy.spatial import cKDTree

    xy = target_coords[:, :2]
    n = len(xy)

    seg_dx = np.diff(xy[:, 0])
    seg_dy = np.diff(xy[:, 1])
    seg_len = np.sqrt(seg_dx**2 + seg_dy**2)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])

    ctrl_xy = control_xy_dip[:, :2]
    ctrl_dip = control_xy_dip[:, 2]

    ctrl_arclen = np.empty(len(ctrl_xy))
    for i, pt in enumerate(ctrl_xy):
        ctrl_arclen[i] = _project_point_to_polyline_arclen(pt, xy, cumlen, seg_len)

    order = np.argsort(ctrl_arclen)
    sorted_arclen = ctrl_arclen[order]
    sorted_dip = ctrl_dip[order]

    interp_fn = interp1d(
        sorted_arclen, sorted_dip,
        fill_value=(sorted_dip[0], sorted_dip[-1]),
        bounds_error=False,
    )
    interpolated_dip = interp_fn(cumlen)

    if verbose:
        print(f"Arc-length interpolation: trace length={cumlen[-1]:.2f}, "
              f"{len(ctrl_xy)} control points mapped to arc-lengths "
              f"{sorted_arclen}")

    return interpolated_dip


def _project_point_to_polyline_arclen(
    point: np.ndarray,
    polyline: np.ndarray,
    cumlen: np.ndarray,
    seg_len: np.ndarray,
) -> float:
    """Project a point onto a polyline and return its arc-length position.

    Uses perpendicular projection onto each segment, clamped to segment
    endpoints, then picks the segment with minimum distance.
    """
    px, py = point[0], point[1]
    n_seg = len(seg_len)

    best_dist = np.inf
    best_arclen = 0.0

    for j in range(n_seg):
        ax, ay = polyline[j, 0], polyline[j, 1]
        bx, by = polyline[j + 1, 0], polyline[j + 1, 1]

        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay

        L2 = seg_len[j] ** 2
        if L2 < 1e-30:
            t = 0.0
        else:
            t = (apx * abx + apy * aby) / L2
            t = max(0.0, min(1.0, t))

        proj_x = ax + t * abx
        proj_y = ay + t * aby
        dist = (px - proj_x) ** 2 + (py - proj_y) ** 2

        if dist < best_dist:
            best_dist = dist
            best_arclen = cumlen[j] + t * seg_len[j]

    return best_arclen


# ---------------------------------------------------------------------------
# 4. generate_bottom_from_dips
# ---------------------------------------------------------------------------

def generate_bottom_from_dips(
    top_coords: np.ndarray,
    dip_deg: np.ndarray,
    strike_deg: np.ndarray,
    fault_depth: float,
    fault_top: float,
    use_average_strike: bool = False,
    average_strike_source: str = 'pca',
    user_direction_angle: float | None = None,
    interpolation_axis: str = 'auto',
    verbose: bool = False,
) -> np.ndarray:
    """Compute bottom coordinates from top geometry, dip, and strike.

    Core formula::

        width = (fault_depth - fault_top) / sin(dip)
        bottom = old_coords + dip_vector * width

    Parameters
    ----------
    top_coords : (n, 2+) array
        Top coordinates in UTM (x, y[, z]).
    dip_deg : (n,) array
        Per-node dip in degrees. Positive values dip right of positive strike;
        negative values dip left. Values in ``(90, 180)`` are canonicalized to
        ``dip - 180``. Zero and 180 degrees are invalid for depth projection.
    strike_deg : (n,) array
        Per-node strike (geographic azimuth) in degrees.
    fault_depth : float
        Target depth of the fault bottom (km, positive down).
    fault_top : float
        Depth of the fault top edge (km, positive down).
    use_average_strike : bool
        Use a single average strike for all nodes.
    average_strike_source : str
        ``'pca'`` fits and orients the first principal axis of top x/y;
        ``'user'`` uses ``user_direction_angle``.
    user_direction_angle : float or None
        Explicit geographic strike azimuth in degrees clockwise from North
        (when source='user'). It must follow the positive trace direction.
    interpolation_axis : str
        Retained for API compatibility. Bottom rows preserve top-node order;
        this function never sorts them independently.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    bottom_coords : (n, 3) ndarray
        Bottom coordinates in UTM. Row ``i`` corresponds to
        ``top_coords[i]``.
    """
    from numpy import deg2rad, sin, cos
    from ..DipInterpolation import (
        normalize_dip_to_neg90_90,
        validate_dip_angles_for_depth_projection,
    )

    x = top_coords[:, 0]
    y = top_coords[:, 1]
    n = len(x)

    strike_rad = deg2rad(strike_deg.copy())
    dip_signed = normalize_dip_to_neg90_90(
        validate_dip_angles_for_depth_projection(dip_deg, name='dip')
    )
    dip_rad = deg2rad(dip_signed)

    strike_direction = np.array([x[-1] - x[0], y[-1] - y[0]])

    if use_average_strike:
        if average_strike_source == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pca.fit(np.column_stack([x, y]))
            principal = pca.components_[0]
            avg_rad = np.pi / 2.0 - np.arctan2(principal[1], principal[0])
            if np.dot(principal, strike_direction) < 0:
                avg_rad += np.pi
            strike_rad = np.full(n, avg_rad)
        elif average_strike_source == 'user' and user_direction_angle is not None:
            avg_rad = deg2rad(user_direction_angle)
            # Geographic azimuth maps to [east, north] as [sin, cos].
            user_strike_vector = np.array([sin(avg_rad), cos(avg_rad)])
            trace_direction_norm = np.linalg.norm(strike_direction)
            alignment = np.dot(user_strike_vector, strike_direction)
            if (
                not np.isfinite(avg_rad)
                or trace_direction_norm <= np.finfo(float).eps
                or alignment <= np.finfo(float).eps * trace_direction_norm
            ):
                raise ValueError(
                    "user_direction_angle must follow the positive trace "
                    "direction (geographic azimuth clockwise from North).")
            strike_rad = np.full(n, avg_rad)
        else:
            raise ValueError(
                "Invalid average_strike_source or user_direction_angle not provided.")
        if verbose:
            print(f"Average strike direction: {np.rad2deg(strike_rad[0]):.2f}")

    negative_mask = dip_rad < 0
    strike_rad[negative_mask] += np.pi
    dip_rad[negative_mask] = -dip_rad[negative_mask]

    strike_rad = np.mod(strike_rad, 2 * np.pi)

    width = ((fault_depth - fault_top) / sin(dip_rad)).reshape(-1, 1)
    old_coords = np.column_stack([x, y, np.full(n, fault_top)])

    dip_x = cos(dip_rad) * cos(-strike_rad)
    dip_y = cos(dip_rad) * sin(-strike_rad)
    dip_z = sin(dip_rad)
    dip_vector = np.column_stack([dip_x, dip_y, dip_z])

    bottom_coords = old_coords + dip_vector * width

    # Node i on the bottom is derived from node i on the top. Never sort the
    # bottom independently: doing so silently breaks top/bottom correspondence
    # for curved or non-monotonic traces.
    validate_top_bottom_cells(old_coords, bottom_coords)
    return bottom_coords


# ---------------------------------------------------------------------------
# 5. augment_control_points_with_buffers
# ---------------------------------------------------------------------------

def augment_control_points_with_buffers(
    control_xy_dip: np.ndarray,
    buffer_nodes_lonlat: np.ndarray,
    buffer_radius: float,
    interpolation_axis: str,
    top_coords_2d: np.ndarray,
    ll2xy: callable,
    xy2ll: callable,
) -> np.ndarray:
    """Add buffer-zone dip constraints around specified nodes.

    Mirrors the logic of ``AdaptiveTriangularPatches.handle_buffer_nodes``
    but operates on plain arrays.

    Parameters
    ----------
    control_xy_dip : (n, 3) array
        Columns ``[x_utm, y_utm, dip_degrees]``.
    buffer_nodes_lonlat : (m, 2) array
        Buffer node locations in lon/lat.
    buffer_radius : float
        Buffer distance (km, UTM units).
    interpolation_axis : str
        ``'x'`` or ``'y'``.
    top_coords_2d : (k, 2) array
        Top trace in UTM (for KDTree nearest-neighbour search).
    ll2xy : callable
        ``(lon, lat) -> (x_utm, y_utm)``
    xy2ll : callable
        ``(x_utm, y_utm) -> (lon, lat)``

    Returns
    -------
    augmented : (n + 2*m, 3) array
        Original control points plus buffer constraints,
        columns ``[x_utm, y_utm, dip_degrees]``.
    """
    from scipy.spatial import cKDTree as KDTree

    if interpolation_axis not in {'x', 'y'}:
        raise ValueError(
            "buffer augmentation requires interpolation_axis 'x' or 'y'; "
            f"got {interpolation_axis!r}"
        )
    buf_ll = np.asarray(buffer_nodes_lonlat)
    buf_x, buf_y = ll2xy(buf_ll[:, 0], buf_ll[:, 1])
    buf_xy = np.column_stack([buf_x, buf_y])

    if np.isscalar(buffer_radius):
        radii = np.full(len(buf_xy), buffer_radius)
    else:
        radii = np.asarray(buffer_radius)

    tree = KDTree(top_coords_2d)

    axis_idx = 0 if interpolation_axis == 'x' else 1

    ctrl_sorted_idx = np.argsort(control_xy_dip[:, axis_idx])
    ctrl_sorted = control_xy_dip[ctrl_sorted_idx]

    extra_rows = []
    for i, (node, r) in enumerate(zip(buf_xy, radii)):
        nearest_idx = tree.query(node)[1]
        trace = top_coords_2d

        dists_left = np.sqrt(
            (trace[:nearest_idx, 0] - node[0]) ** 2 +
            (trace[:nearest_idx, 1] - node[1]) ** 2
        )
        dists_right = np.sqrt(
            (trace[nearest_idx + 1:, 0] - node[0]) ** 2 +
            (trace[nearest_idx + 1:, 1] - node[1]) ** 2
        )

        left_in = np.where(dists_left <= r)[0]
        right_in = np.where(dists_right <= r)[0] + nearest_idx + 1

        if len(left_in) == 0 or len(right_in) == 0:
            continue

        left_pt = trace[left_in[0]]
        right_pt = trace[right_in[-1]]

        left_val_on_axis = left_pt[axis_idx]
        candidates = ctrl_sorted[ctrl_sorted[:, axis_idx] <= left_val_on_axis]
        if len(candidates) > 0:
            left_dip = candidates[-1, 2]
        else:
            left_dip = ctrl_sorted[ctrl_sorted[:, axis_idx] >= left_val_on_axis][0, 2]

        right_cand_idx = len(candidates)
        if right_cand_idx < len(ctrl_sorted):
            right_dip = ctrl_sorted[right_cand_idx, 2]
        else:
            right_dip = ctrl_sorted[-1, 2]

        extra_rows.append([left_pt[0], left_pt[1], left_dip])
        extra_rows.append([right_pt[0], right_pt[1], right_dip])

    if extra_rows:
        extra = np.array(extra_rows)
        augmented = np.vstack([control_xy_dip, extra])
        _, unique_idx = np.unique(
            np.round(augmented[:, :2], decimals=6), axis=0, return_index=True)
        augmented = augmented[np.sort(unique_idx)]
        return augmented

    return control_xy_dip


# ---------------------------------------------------------------------------
# 6. compute_strike (re-export from AdaptiveTriangularPatches @staticmethod)
# ---------------------------------------------------------------------------

def compute_strike(coords_xy: np.ndarray) -> np.ndarray:
    """Per-point along-strike azimuth (degrees, geographic convention).

    Standalone copy of ``AdaptiveTriangularPatches.compute_strike`` — the
    same algorithm, kept here so dip_ops is self-contained.

    Parameters
    ----------
    coords_xy : (N, 2+) array
        Columns 0, 1 are x, y in a projected CRS.

    Returns
    -------
    strike : (N,) ndarray
        Along-strike azimuth in degrees for each point.
    """
    from numpy import rad2deg, deg2rad, arctan2, sin, cos, diff, concatenate

    x, y = coords_xy[:, 0], coords_xy[:, 1]
    seg_strike = 90 - rad2deg(arctan2(diff(y), diff(x)))
    seg_rad = deg2rad(seg_strike)

    avg_rad = arctan2(
        (sin(seg_rad[:-1]) + sin(seg_rad[1:])) / 2,
        (cos(seg_rad[:-1]) + cos(seg_rad[1:])) / 2,
    )
    avg_deg = rad2deg(avg_rad)

    return concatenate(([seg_strike[0]], avg_deg, [seg_strike[-1]]))
