"""Utilities for selecting and validating fault patch indices.

This module keeps patch subset handling separate from constraint generation.
The helpers return patch ids only; callers such as Euler constraints,
zero-slip constraints, or post-processing decide how those ids are used.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _patch_count(fault: Any) -> int:
    if not hasattr(fault, "patch"):
        raise AttributeError("fault object has no 'patch' attribute")
    return len(fault.patch)


def _as_sequence(indices: Any, *, name: str) -> list[Any]:
    if np.isscalar(indices):
        return [indices]
    try:
        return list(indices)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer or an iterable of integers") from exc


def normalize_patch_indices(
    fault: Any,
    patch_indices: Iterable[int] | int | None = None,
    *,
    allow_none_all: bool = True,
    unique: bool = False,
    name: str = "patch_indices",
) -> np.ndarray:
    """Return validated patch indices as a one-dimensional integer array.

    Parameters
    ----------
    fault : object
        CSI fault-like object with a ``patch`` sequence.
    patch_indices : iterable of int or int, optional
        Patch ids to validate.  ``None`` means all patches when
        ``allow_none_all=True``.
    allow_none_all : bool, default True
        Whether ``None`` expands to all patch ids.
    unique : bool, default False
        If True, remove duplicates while preserving the first occurrence.
    name : str, default ``"patch_indices"``
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        Validated integer patch ids.
    """
    n_patches = _patch_count(fault)
    if patch_indices is None:
        if allow_none_all:
            return np.arange(n_patches, dtype=int)
        raise ValueError(f"{name} cannot be None")

    raw = _as_sequence(patch_indices, name=name)
    if not raw:
        return np.asarray([], dtype=int)

    indices: list[int] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must contain integers, got boolean {value!r}")
        if isinstance(value, (int, np.integer)):
            idx = int(value)
        elif isinstance(value, (float, np.floating)) and float(value).is_integer():
            idx = int(value)
        else:
            raise TypeError(f"{name} must contain integer patch ids, got {value!r}")
        indices.append(idx)

    arr = np.asarray(indices, dtype=int)
    invalid = arr[(arr < 0) | (arr >= n_patches)]
    if invalid.size:
        raise ValueError(
            f"{name} contains patch ids outside [0, {n_patches - 1}]: "
            f"{invalid.tolist()}"
        )

    if unique:
        seen: set[int] = set()
        arr = np.asarray([idx for idx in arr.tolist() if not (idx in seen or seen.add(idx))], dtype=int)

    return arr


def get_patch_centers(fault: Any, *, coord: str = "xy") -> np.ndarray:
    """Return patch centers in local ``xy`` or ``lonlat`` coordinates.

    Parameters
    ----------
    fault : object
        CSI fault-like object with ``getcenters()``.  For ``coord='lonlat'``,
        the object must also provide ``xy2ll``.
    coord : {"xy", "lonlat"}, default ``"xy"``
        Coordinate system for the first two columns.  The third column is depth
        in kilometers in both cases.

    Returns
    -------
    numpy.ndarray
        Array with columns ``x y depth`` or ``lon lat depth``.
    """
    centers = np.asarray(fault.getcenters(), dtype=float)
    if centers.ndim != 2 or centers.shape[1] < 3:
        raise ValueError("fault.getcenters() must return an array with at least 3 columns")
    centers = centers[:, :3].copy()

    key = coord.lower().replace("_", "").replace("-", "")
    if key in ("xy", "utm", "local"):
        return centers
    if key in ("lonlat", "ll"):
        if not hasattr(fault, "xy2ll"):
            raise AttributeError("fault object has no xy2ll() method for lonlat centers")
        lon, lat = fault.xy2ll(centers[:, 0], centers[:, 1])
        return np.column_stack((lon, lat, centers[:, 2]))
    raise ValueError("coord must be 'xy' or 'lonlat'")


def get_edge_patch_indices(
    fault: Any,
    edges: str | Sequence[str],
    *,
    unique: bool = True,
) -> np.ndarray:
    """Return patch ids attached to one or more named fault edges.

    Parameters
    ----------
    fault : object
        Fault object with ``edge_triangles_indices``.
    edges : str or sequence of str
        Edge names such as ``"top"``, ``"bottom"``, ``"left"`` or
        ``"right"``.
    unique : bool, default True
        If True, remove duplicate ids while preserving order.

    Returns
    -------
    numpy.ndarray
        Integer patch ids for the requested edges.
    """
    if not hasattr(fault, "edge_triangles_indices"):
        raise AttributeError(
            "fault object has no 'edge_triangles_indices'. Run edge detection first."
        )

    edge_names = [edges] if isinstance(edges, str) else list(edges)
    all_indices: list[int] = []
    available = list(fault.edge_triangles_indices.keys())
    for edge in edge_names:
        key = str(edge).lower()
        if key not in fault.edge_triangles_indices:
            raise KeyError(f"Edge '{edge}' not found. Available edges: {available}")
        all_indices.extend(np.asarray(fault.edge_triangles_indices[key], dtype=int).tolist())

    return normalize_patch_indices(fault, all_indices, allow_none_all=False, unique=unique, name="edge patch indices")


def _depth_mask(depths: np.ndarray, depth_range: Sequence[float] | None) -> np.ndarray:
    if depth_range is None:
        return np.ones(depths.shape[0], dtype=bool)
    if len(depth_range) != 2:
        raise ValueError("depth_range must be a two-item sequence: (min_depth, max_depth)")
    zmin, zmax = float(depth_range[0]), float(depth_range[1])
    if zmin > zmax:
        zmin, zmax = zmax, zmin
    return (depths >= zmin) & (depths <= zmax)


def get_patches_by_depth(
    fault: Any,
    depth_range: Sequence[float],
) -> np.ndarray:
    """Return patch ids whose center depth falls inside ``depth_range``.

    Parameters
    ----------
    fault : object
        CSI fault-like object.
    depth_range : sequence of float
        ``(min_depth, max_depth)`` in kilometers.

    Returns
    -------
    numpy.ndarray
        Integer patch ids selected by center depth.
    """
    centers = get_patch_centers(fault, coord="xy")
    return np.nonzero(_depth_mask(centers[:, 2], depth_range))[0].astype(int)


def get_patches_in_box(
    fault: Any,
    *,
    lon_range: Sequence[float] | None = None,
    lat_range: Sequence[float] | None = None,
    x_range: Sequence[float] | None = None,
    y_range: Sequence[float] | None = None,
    depth_range: Sequence[float] | None = None,
) -> np.ndarray:
    """Return patch ids whose centers fall inside a lon/lat or local xy box.

    Parameters
    ----------
    fault : object
        CSI fault-like object.
    lon_range, lat_range : sequence of float, optional
        Longitude and latitude bounds.  Provide both for geographic selection.
    x_range, y_range : sequence of float, optional
        Local coordinate bounds in kilometers.  Provide both for local selection.
    depth_range : sequence of float, optional
        Center-depth bounds in kilometers.

    Returns
    -------
    numpy.ndarray
        Integer patch ids selected by patch centers.
    """
    use_lonlat = lon_range is not None or lat_range is not None
    use_xy = x_range is not None or y_range is not None
    if use_lonlat == use_xy:
        raise ValueError("Provide either lon_range+lat_range or x_range+y_range")

    if use_lonlat:
        if lon_range is None or lat_range is None:
            raise ValueError("lon_range and lat_range must be provided together")
        centers = get_patch_centers(fault, coord="lonlat")
        xr, yr = lon_range, lat_range
    else:
        if x_range is None or y_range is None:
            raise ValueError("x_range and y_range must be provided together")
        centers = get_patch_centers(fault, coord="xy")
        xr, yr = x_range, y_range

    xmin, xmax = sorted((float(xr[0]), float(xr[1])))
    ymin, ymax = sorted((float(yr[0]), float(yr[1])))
    mask = (
        (centers[:, 0] >= xmin)
        & (centers[:, 0] <= xmax)
        & (centers[:, 1] >= ymin)
        & (centers[:, 1] <= ymax)
        & _depth_mask(centers[:, 2], depth_range)
    )
    return np.nonzero(mask)[0].astype(int)


def _trace_xy(fault: Any, *, use_discretized: bool) -> np.ndarray:
    if use_discretized and hasattr(fault, "xi") and getattr(fault, "xi") is not None:
        x = np.asarray(fault.xi, dtype=float)
        y = np.asarray(fault.yi, dtype=float)
    elif hasattr(fault, "xf") and getattr(fault, "xf") is not None:
        x = np.asarray(fault.xf, dtype=float)
        y = np.asarray(fault.yf, dtype=float)
    else:
        raise ValueError("fault trace is not available; expected xi/yi or xf/yf")

    if x.size != y.size or x.size < 2:
        raise ValueError("fault trace must contain at least two x/y points")
    return np.column_stack((x, y))


def _point_to_xy(fault: Any, point: Sequence[float], coord_system: str) -> tuple[float, float]:
    key = coord_system.lower().replace("_", "").replace("-", "")
    if key in ("lonlat", "ll"):
        if not hasattr(fault, "ll2xy"):
            raise AttributeError("fault object has no ll2xy() method for lonlat input")
        x, y = fault.ll2xy(point[0], point[1])
        return float(np.asarray(x)), float(np.asarray(y))
    if key in ("xy", "utm", "local"):
        return float(point[0]), float(point[1])
    raise ValueError("coord_system must be 'lonlat' or 'xy'")


def _project_points_to_polyline(points: np.ndarray, polyline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seg_start = polyline[:-1]
    seg_end = polyline[1:]
    seg_vec = seg_end - seg_start
    seg_len = np.linalg.norm(seg_vec, axis=1)
    valid = seg_len > 0.0
    if not np.any(valid):
        raise ValueError("fault trace contains only zero-length segments")
    seg_start = seg_start[valid]
    seg_vec = seg_vec[valid]
    seg_len = seg_len[valid]
    seg_len2 = seg_len**2
    seg_cum = np.concatenate(([0.0], np.cumsum(seg_len[:-1])))

    along = np.empty(points.shape[0], dtype=float)
    distance = np.empty(points.shape[0], dtype=float)
    for i, point in enumerate(points):
        rel = point - seg_start
        t = np.clip(np.sum(rel * seg_vec, axis=1) / seg_len2, 0.0, 1.0)
        projected = seg_start + t[:, None] * seg_vec
        dist = np.linalg.norm(point - projected, axis=1)
        j = int(np.argmin(dist))
        along[i] = seg_cum[j] + t[j] * seg_len[j]
        distance[i] = dist[j]
    return along, distance


def get_patches_in_trace_range(
    fault: Any,
    point1: Sequence[float],
    point2: Sequence[float],
    *,
    buffer_distance: float | None = None,
    depth_range: Sequence[float] | None = None,
    coord_system: str = "lonlat",
    use_discretized: bool = True,
) -> np.ndarray:
    """Return patch ids between two points along the fault trace.

    The selection is center-based.  ``point1`` and ``point2`` are projected to
    the fault trace, then patch centers are selected by along-trace position.
    ``buffer_distance`` optionally limits the perpendicular distance from the
    trace in kilometers.

    Parameters
    ----------
    fault : object
        CSI fault-like object with trace coordinates and patch centers.
    point1, point2 : sequence of float
        End points in ``coord_system``.
    buffer_distance : float, optional
        Maximum perpendicular distance from the trace, in kilometers.
    depth_range : sequence of float, optional
        Center-depth bounds in kilometers.
    coord_system : {"lonlat", "xy"}, default ``"lonlat"``
        Coordinate system of ``point1`` and ``point2``.
    use_discretized : bool, default True
        Prefer ``xi/yi`` over ``xf/yf`` when available.

    Returns
    -------
    numpy.ndarray
        Integer patch ids selected by patch centers.
    """
    trace = _trace_xy(fault, use_discretized=use_discretized)
    p1 = np.asarray(_point_to_xy(fault, point1, coord_system), dtype=float)
    p2 = np.asarray(_point_to_xy(fault, point2, coord_system), dtype=float)
    query_s, _ = _project_points_to_polyline(np.vstack((p1, p2)), trace)
    smin, smax = sorted((query_s[0], query_s[1]))

    centers = get_patch_centers(fault, coord="xy")
    patch_s, patch_d = _project_points_to_polyline(centers[:, :2], trace)
    mask = (patch_s >= smin) & (patch_s <= smax) & _depth_mask(centers[:, 2], depth_range)
    if buffer_distance is not None:
        if buffer_distance < 0:
            raise ValueError("buffer_distance must be non-negative")
        mask &= patch_d <= float(buffer_distance)
    return np.nonzero(mask)[0].astype(int)


def select_patch_indices(
    fault: Any,
    selector: Any = None,
    *,
    allow_none_all: bool = True,
    unique: bool = True,
    name: str = "selector",
) -> np.ndarray:
    """Select patch ids from a compact selector object.

    Parameters
    ----------
    fault : object
        CSI fault-like object.
    selector : object, optional
        ``None`` selects all patches when ``allow_none_all=True``.  A sequence
        of integers is treated as explicit patch ids.  A mapping may use one of
        the following forms:

        - ``{"patches": [...]}`` or ``{"patch_indices": [...]}``
        - ``{"edge": "top"}`` or ``{"edges": ["top", "bottom"]}``
        - ``{"depth_range": [zmin, zmax]}``
        - ``{"trace_range": {"point1": [...], "point2": [...], ...}}``
        - ``{"box": {"lon_range": [...], "lat_range": [...]}}``

        ``depth_range`` may be combined with ``edge`` or ``trace_range`` to
        further restrict the selected patch centers.
    allow_none_all : bool, default True
        Whether ``None`` expands to all patch ids.
    unique : bool, default True
        Remove duplicates while preserving order.
    name : str, default ``"selector"``
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        Validated integer patch ids.
    """
    if selector is None:
        return normalize_patch_indices(
            fault,
            None,
            allow_none_all=allow_none_all,
            unique=unique,
            name=name,
        )

    if not isinstance(selector, Mapping):
        return normalize_patch_indices(
            fault,
            selector,
            allow_none_all=False,
            unique=unique,
            name=name,
        )

    if "patches" in selector:
        selected = normalize_patch_indices(
            fault,
            selector["patches"],
            allow_none_all=False,
            unique=unique,
            name=f"{name}.patches",
        )
    elif "patch_indices" in selector:
        selected = normalize_patch_indices(
            fault,
            selector["patch_indices"],
            allow_none_all=False,
            unique=unique,
            name=f"{name}.patch_indices",
        )
    elif "edge" in selector or "edges" in selector:
        selected = get_edge_patch_indices(fault, selector.get("edge", selector.get("edges")), unique=unique)
    elif "trace_range" in selector:
        trace_range = dict(selector["trace_range"])
        selected = get_patches_in_trace_range(
            fault,
            trace_range["point1"],
            trace_range["point2"],
            buffer_distance=trace_range.get("buffer_distance"),
            depth_range=trace_range.get("depth_range", selector.get("depth_range")),
            coord_system=trace_range.get("coord_system", selector.get("coord_system", "lonlat")),
            use_discretized=trace_range.get("use_discretized", True),
        )
    elif "box" in selector:
        box = dict(selector["box"])
        selected = get_patches_in_box(
            fault,
            lon_range=box.get("lon_range"),
            lat_range=box.get("lat_range"),
            x_range=box.get("x_range"),
            y_range=box.get("y_range"),
            depth_range=box.get("depth_range", selector.get("depth_range")),
        )
    elif any(key in selector for key in ("lon_range", "lat_range", "x_range", "y_range")):
        selected = get_patches_in_box(
            fault,
            lon_range=selector.get("lon_range"),
            lat_range=selector.get("lat_range"),
            x_range=selector.get("x_range"),
            y_range=selector.get("y_range"),
            depth_range=selector.get("depth_range"),
        )
    elif "depth_range" in selector:
        selected = get_patches_by_depth(fault, selector["depth_range"])
    else:
        allowed = "patches, patch_indices, edge/edges, depth_range, trace_range, box"
        raise ValueError(f"{name} must define one of: {allowed}")

    if "depth_range" in selector and "trace_range" not in selector and "box" not in selector:
        depth_selected = set(get_patches_by_depth(fault, selector["depth_range"]).tolist())
        selected = np.asarray([idx for idx in selected.tolist() if idx in depth_selected], dtype=int)

    return normalize_patch_indices(
        fault,
        selected,
        allow_none_all=False,
        unique=unique,
        name=name,
    )
