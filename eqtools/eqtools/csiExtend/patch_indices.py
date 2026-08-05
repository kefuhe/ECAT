"""Utilities for selecting and validating fault patch indices.

This module keeps patch subset handling separate from constraint generation.
The helpers return patch ids only; callers such as Euler constraints,
zero-slip constraints, or post-processing decide how those ids are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .trace_ops import (
    point_at_trace_distance,
    project_points_to_trace,
    sample_trace_distances,
    trace_length,
)


@dataclass(frozen=True)
class TraceMarker:
    """A resolved point on a fault trace.

    Distances are measured along the fault trace in local ``x/y`` kilometers,
    not along longitude, latitude, or a single projected axis.
    """

    x: float
    y: float
    trace_distance_km: float
    segment_index: int
    segment_fraction: float
    lon: float | None = None
    lat: float | None = None
    distance_to_trace_km: float = 0.0
    method: str = "unknown"

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def lonlat(self) -> tuple[float, float] | None:
        if self.lon is None or self.lat is None:
            return None
        return (self.lon, self.lat)

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "lon": self.lon,
            "lat": self.lat,
            "x": self.x,
            "y": self.y,
            "trace_distance_km": self.trace_distance_km,
            "segment_index": self.segment_index,
            "segment_fraction": self.segment_fraction,
            "distance_to_trace_km": self.distance_to_trace_km,
            "method": self.method,
        }


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


def _xy_to_lonlat(fault: Any, x: float, y: float) -> tuple[float | None, float | None]:
    if not hasattr(fault, "xy2ll"):
        return None, None
    lon, lat = fault.xy2ll(float(x), float(y))
    return float(np.asarray(lon)), float(np.asarray(lat))


def _make_trace_marker(
    fault: Any,
    *,
    x: float,
    y: float,
    trace_distance_km: float,
    segment_index: int,
    segment_fraction: float,
    distance_to_trace_km: float = 0.0,
    method: str,
) -> TraceMarker:
    lon, lat = _xy_to_lonlat(fault, x, y)
    return TraceMarker(
        x=float(x),
        y=float(y),
        lon=lon,
        lat=lat,
        trace_distance_km=float(trace_distance_km),
        segment_index=int(segment_index),
        segment_fraction=float(segment_fraction),
        distance_to_trace_km=float(distance_to_trace_km),
        method=method,
    )


def _trace_lonlat(fault: Any, trace_xy: np.ndarray) -> np.ndarray:
    if not hasattr(fault, "xy2ll"):
        raise AttributeError("fault object has no xy2ll() method for longitude/latitude trace markers")
    lon, lat = fault.xy2ll(trace_xy[:, 0], trace_xy[:, 1])
    return np.column_stack((np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)))


def _coordinate_intersections_on_trace(
    fault: Any,
    trace_xy: np.ndarray,
    *,
    axis: str,
    value: float,
    atol: float = 1e-12,
) -> list[TraceMarker]:
    axis_key = axis.lower()
    if axis_key in ("lon", "longitude"):
        coord = _trace_lonlat(fault, trace_xy)[:, 0]
        method = "longitude"
    elif axis_key in ("lat", "latitude"):
        coord = _trace_lonlat(fault, trace_xy)[:, 1]
        method = "latitude"
    elif axis_key == "x":
        coord = trace_xy[:, 0]
        method = "x"
    elif axis_key == "y":
        coord = trace_xy[:, 1]
        method = "y"
    else:
        raise ValueError("axis must be longitude/lon, latitude/lat, x, or y")

    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(trace_xy[:, :2], axis=0), axis=1))]
    target = float(value)
    markers: list[TraceMarker] = []
    seen: set[tuple[int, float]] = set()
    for idx in range(trace_xy.shape[0] - 1):
        a = float(coord[idx])
        b = float(coord[idx + 1])
        denom = b - a
        if abs(denom) <= atol:
            if abs(target - a) > atol:
                continue
            fractions = (0.0, 1.0)
        else:
            frac = (target - a) / denom
            if frac < -atol or frac > 1.0 + atol:
                continue
            fractions = (float(np.clip(frac, 0.0, 1.0)),)

        seg_len = s[idx + 1] - s[idx]
        for frac in fractions:
            key = (idx, round(frac, 12))
            if key in seen:
                continue
            seen.add(key)
            xy = trace_xy[idx] + frac * (trace_xy[idx + 1] - trace_xy[idx])
            markers.append(
                _make_trace_marker(
                    fault,
                    x=xy[0],
                    y=xy[1],
                    trace_distance_km=s[idx] + frac * seg_len,
                    segment_index=idx,
                    segment_fraction=frac,
                    method=method,
                )
            )

    markers.sort(key=lambda marker: marker.trace_distance_km)
    return markers


def _choose_marker(candidates: list[TraceMarker], *, which: Any = "first") -> TraceMarker:
    if not candidates:
        raise ValueError("trace marker does not intersect the fault trace")
    if isinstance(which, (int, np.integer)):
        idx = int(which)
        try:
            return candidates[idx]
        except IndexError as exc:
            raise IndexError(f"trace marker intersection index {idx} is out of range") from exc

    key = str(which).lower()
    if key == "first":
        return candidates[0]
    if key == "last":
        return candidates[-1]
    raise ValueError("which must be 'first', 'last', or an integer intersection index")


def resolve_trace_marker(
    fault: Any,
    marker: Any,
    *,
    coord_system: str = "lonlat",
    use_discretized: bool = True,
) -> TraceMarker:
    """Resolve a user marker to a true point on the fault trace.

    Supported marker forms include:

    - ``{"longitude": 101.5}`` or ``{"by": "longitude", "value": 101.5}``
    - ``{"latitude": 24.0}``
    - ``{"trace_distance_km": 35.0}``
    - ``{"fraction": 0.5}``
    - ``{"point": [lon, lat], "coord_system": "lonlat"}``
    - ``{"nearest": [lon, lat]}``
    - a bare point sequence, interpreted in ``coord_system``.

    Point-like markers are projected to the nearest point on the trace; they do
    not snap to the nearest existing trace vertex.
    """
    if isinstance(marker, TraceMarker):
        return marker

    trace = _trace_xy(fault, use_discretized=use_discretized)
    if isinstance(marker, Mapping):
        raw = dict(marker)
        which = raw.get("which", "first")
        by = raw.get("by")

        if "trace_distance_km" in raw or by == "trace_distance_km":
            value = raw.get("trace_distance_km", raw.get("value"))
            point = point_at_trace_distance(trace, float(value))
            return _make_trace_marker(
                fault,
                x=point["xy"][0],
                y=point["xy"][1],
                trace_distance_km=point["trace_distance_km"],
                segment_index=point["segment_index"],
                segment_fraction=point["segment_fraction"],
                method="trace_distance_km",
            )

        if "fraction" in raw or by == "fraction":
            value = raw.get("fraction", raw.get("value"))
            fraction = float(value)
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError("trace marker fraction must be in [0, 1]")
            point = point_at_trace_distance(trace, fraction * trace_length(trace))
            return _make_trace_marker(
                fault,
                x=point["xy"][0],
                y=point["xy"][1],
                trace_distance_km=point["trace_distance_km"],
                segment_index=point["segment_index"],
                segment_fraction=point["segment_fraction"],
                method="fraction",
            )

        if "longitude" in raw or "lon" in raw or by in ("longitude", "lon"):
            value = raw.get("longitude", raw.get("lon", raw.get("value")))
            candidates = _coordinate_intersections_on_trace(
                fault,
                trace,
                axis="longitude",
                value=float(value),
            )
            return _choose_marker(candidates, which=which)

        if "latitude" in raw or "lat" in raw or by in ("latitude", "lat"):
            value = raw.get("latitude", raw.get("lat", raw.get("value")))
            candidates = _coordinate_intersections_on_trace(
                fault,
                trace,
                axis="latitude",
                value=float(value),
            )
            return _choose_marker(candidates, which=which)

        if "x" in raw or by == "x":
            value = raw.get("x", raw.get("value"))
            candidates = _coordinate_intersections_on_trace(fault, trace, axis="x", value=float(value))
            return _choose_marker(candidates, which=which)

        if "y" in raw or by == "y":
            value = raw.get("y", raw.get("value"))
            candidates = _coordinate_intersections_on_trace(fault, trace, axis="y", value=float(value))
            return _choose_marker(candidates, which=which)

        if "xy" in raw:
            point = raw["xy"]
            point_coord_system = "xy"
        elif "lonlat" in raw:
            point = raw["lonlat"]
            point_coord_system = "lonlat"
        elif "point" in raw:
            point = raw["point"]
            point_coord_system = raw.get("coord_system", coord_system)
        elif "nearest" in raw:
            point = raw["nearest"]
            point_coord_system = raw.get("coord_system", coord_system)
        elif by in ("point", "nearest"):
            point = raw.get("value")
            point_coord_system = raw.get("coord_system", coord_system)
        else:
            allowed = "longitude, latitude, trace_distance_km, fraction, point/nearest"
            raise ValueError(f"trace marker mapping must define one of: {allowed}")
    else:
        point = marker
        point_coord_system = coord_system

    xy = np.asarray(_point_to_xy(fault, point, point_coord_system), dtype=float)
    projection = project_points_to_trace(xy, trace)
    projected_xy = projection["projected_xy"][0]
    return _make_trace_marker(
        fault,
        x=projected_xy[0],
        y=projected_xy[1],
        trace_distance_km=projection["trace_distance_km"][0],
        segment_index=projection["segment_index"][0],
        segment_fraction=projection["segment_fraction"][0],
        distance_to_trace_km=projection["distance_to_trace_km"][0],
        method="nearest",
    )


def sample_trace_markers(
    fault: Any,
    start: Any,
    end: Any,
    *,
    step_km: float,
    include_endpoint: bool = True,
    coord_system: str = "lonlat",
    use_discretized: bool = True,
) -> list[TraceMarker]:
    """Sample true along-trace markers between two trace markers."""
    trace = _trace_xy(fault, use_discretized=use_discretized)
    start_marker = resolve_trace_marker(
        fault,
        start,
        coord_system=coord_system,
        use_discretized=use_discretized,
    )
    end_marker = resolve_trace_marker(
        fault,
        end,
        coord_system=coord_system,
        use_discretized=use_discretized,
    )
    distances = sample_trace_distances(
        trace,
        start_marker.trace_distance_km,
        end_marker.trace_distance_km,
        step_km,
        include_endpoint=include_endpoint,
    )
    return [
        resolve_trace_marker(
            fault,
            {"trace_distance_km": float(distance)},
            coord_system="xy",
            use_discretized=use_discretized,
        )
        for distance in distances
    ]


def _project_points_to_polyline(points: np.ndarray, polyline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projection = project_points_to_trace(points, polyline)
    return projection["trace_distance_km"], projection["distance_to_trace_km"]


def get_patches_in_trace_segment(
    fault: Any,
    start: Any,
    end: Any,
    *,
    buffer_distance: float | None = None,
    depth_range: Sequence[float] | None = None,
    coord_system: str = "lonlat",
    use_discretized: bool = True,
    return_markers: bool = False,
) -> np.ndarray | tuple[np.ndarray, TraceMarker, TraceMarker]:
    """Return patch ids between two flexible markers along the fault trace.

    Unlike ``get_patches_in_trace_range()``, ``start`` and ``end`` may be
    longitude/latitude cuts, along-trace distances, fractions, or points that
    are projected to the nearest true trace point.
    """
    start_marker = resolve_trace_marker(
        fault,
        start,
        coord_system=coord_system,
        use_discretized=use_discretized,
    )
    end_marker = resolve_trace_marker(
        fault,
        end,
        coord_system=coord_system,
        use_discretized=use_discretized,
    )
    selected = get_patches_in_trace_range(
        fault,
        start_marker.xy,
        end_marker.xy,
        buffer_distance=buffer_distance,
        depth_range=depth_range,
        coord_system="xy",
        use_discretized=use_discretized,
    )
    if return_markers:
        return selected, start_marker, end_marker
    return selected


def trace_range_selector_from_markers(
    fault: Any,
    start: Any,
    end: Any,
    *,
    buffer_distance: float | None = None,
    depth_range: Sequence[float] | None = None,
    coord_system: str = "lonlat",
    use_discretized: bool = True,
    output_coord_system: str = "lonlat",
) -> dict[str, Any]:
    """Build a standard ``trace_range`` selector from flexible markers."""
    start_marker = resolve_trace_marker(
        fault,
        start,
        coord_system=coord_system,
        use_discretized=use_discretized,
    )
    end_marker = resolve_trace_marker(
        fault,
        end,
        coord_system=coord_system,
        use_discretized=use_discretized,
    )

    key = output_coord_system.lower().replace("_", "").replace("-", "")
    if key in ("lonlat", "ll"):
        if start_marker.lonlat is None or end_marker.lonlat is None:
            raise AttributeError("fault object has no xy2ll() method for lonlat selector output")
        point1 = list(start_marker.lonlat)
        point2 = list(end_marker.lonlat)
        selector_coord = "lonlat"
    elif key in ("xy", "utm", "local"):
        point1 = [start_marker.x, start_marker.y]
        point2 = [end_marker.x, end_marker.y]
        selector_coord = "xy"
    else:
        raise ValueError("output_coord_system must be 'lonlat' or 'xy'")

    trace_range: dict[str, Any] = {
        "point1": point1,
        "point2": point2,
        "coord_system": selector_coord,
        "use_discretized": bool(use_discretized),
    }
    if buffer_distance is not None:
        trace_range["buffer_distance"] = float(buffer_distance)
    if depth_range is not None:
        trace_range["depth_range"] = [float(depth_range[0]), float(depth_range[1])]
    return {"trace_range": trace_range}


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
        - ``{"trace_segment": {"start": {"longitude": ...}, "end": {...}, ...}}``
        - ``{"box": {"lon_range": [...], "lat_range": [...]}}``

        ``depth_range`` may be combined with ``edge``, ``trace_range``,
        ``trace_segment`` or ``box`` to further restrict selected patch centers.
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
        point1 = trace_range.get("point1", trace_range.get("start"))
        point2 = trace_range.get("point2", trace_range.get("end"))
        if point1 is None or point2 is None:
            raise ValueError(f"{name}.trace_range must define point1/point2 or start/end")
        if isinstance(point1, Mapping) or isinstance(point2, Mapping):
            selected = get_patches_in_trace_segment(
                fault,
                point1,
                point2,
                buffer_distance=trace_range.get("buffer_distance"),
                depth_range=trace_range.get("depth_range", selector.get("depth_range")),
                coord_system=trace_range.get("coord_system", selector.get("coord_system", "lonlat")),
                use_discretized=trace_range.get("use_discretized", True),
            )
        else:
            selected = get_patches_in_trace_range(
                fault,
                point1,
                point2,
                buffer_distance=trace_range.get("buffer_distance"),
                depth_range=trace_range.get("depth_range", selector.get("depth_range")),
                coord_system=trace_range.get("coord_system", selector.get("coord_system", "lonlat")),
                use_discretized=trace_range.get("use_discretized", True),
            )
    elif "trace_segment" in selector:
        trace_segment = dict(selector["trace_segment"])
        selected = get_patches_in_trace_segment(
            fault,
            trace_segment["start"],
            trace_segment["end"],
            buffer_distance=trace_segment.get("buffer_distance"),
            depth_range=trace_segment.get("depth_range", selector.get("depth_range")),
            coord_system=trace_segment.get("coord_system", selector.get("coord_system", "lonlat")),
            use_discretized=trace_segment.get("use_discretized", True),
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
        allowed = "patches, patch_indices, edge/edges, depth_range, trace_range/trace_segment, box"
        raise ValueError(f"{name} must define one of: {allowed}")

    if (
        "depth_range" in selector
        and "trace_range" not in selector
        and "trace_segment" not in selector
        and "box" not in selector
    ):
        depth_selected = set(get_patches_by_depth(fault, selector["depth_range"]).tolist())
        selected = np.asarray([idx for idx in selected.tolist() if idx in depth_selected], dtype=int)

    return normalize_patch_indices(
        fault,
        selected,
        allow_none_all=False,
        unique=unique,
        name=name,
    )
