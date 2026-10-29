"""Resolve user-facing locations to exact positions on a projected trace.

This module is intentionally independent of CSI fault and patch objects.  It
turns distances, fractions, longitude/latitude cuts, or nearest-point queries
into :class:`TraceMarker` instances.  Downstream code can then use the common
``trace_distance_km`` value for trimming, sampling, or patch selection.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .trace_ops import (
    clean_trace,
    point_at_trace_distance,
    project_points_to_trace,
    trace_length,
)


@dataclass(frozen=True)
class TraceMarker:
    """A resolved point on a projected fault trace.

    Distances are measured from the first trace point along the local ``x/y``
    polyline.  ``distance_to_trace_km`` is non-zero only for nearest-point
    queries.  ``candidate_count`` records longitude/latitude cuts that intersect
    a curved trace more than once.
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
    candidate_count: int = 1
    candidate_index: int = 0

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
            "candidate_count": self.candidate_count,
            "candidate_index": self.candidate_index,
        }


def _call_transform(
    transform: Callable[[Any, Any], tuple[Any, Any]],
    first: Any,
    second: Any,
) -> tuple[np.ndarray, np.ndarray]:
    out_first, out_second = transform(first, second)
    return np.asarray(out_first, dtype=float), np.asarray(out_second, dtype=float)


def _trace_lonlat(
    trace_xy: np.ndarray,
    *,
    lonlat: Any | None,
    xy2ll: Callable[[Any, Any], tuple[Any, Any]] | None,
) -> np.ndarray | None:
    if lonlat is not None:
        values = np.asarray(lonlat, dtype=float)
        if values.ndim != 2 or values.shape[0] != trace_xy.shape[0] or values.shape[1] < 2:
            raise ValueError("lonlat must align with coords and contain longitude/latitude columns.")
        if not np.all(np.isfinite(values[:, :2])):
            raise ValueError("lonlat contains NaN or infinite values.")
        return values[:, :2].copy()
    if xy2ll is None:
        return None
    longitude, latitude = _call_transform(xy2ll, trace_xy[:, 0], trace_xy[:, 1])
    return np.column_stack((longitude.reshape(-1), latitude.reshape(-1)))


def _prepare_trace(
    coords: Any,
    *,
    lonlat: Any | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Clean projected duplicates while preserving lon/lat row alignment."""
    values = np.asarray(coords, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] < 2:
        raise ValueError("coords must contain at least two x/y points.")
    projected = np.array(values[:, :2], dtype=float, copy=True)
    if not np.all(np.isfinite(projected)):
        raise ValueError("coords contains NaN or infinite x/y values.")

    geographic = None
    if lonlat is not None:
        geographic = np.asarray(lonlat, dtype=float)
        if (
            geographic.ndim != 2
            or geographic.shape[0] != projected.shape[0]
            or geographic.shape[1] < 2
        ):
            raise ValueError(
                "lonlat must align with coords and contain longitude/latitude columns."
            )
        geographic = np.array(geographic[:, :2], dtype=float, copy=True)
        if not np.all(np.isfinite(geographic)):
            raise ValueError("lonlat contains NaN or infinite values.")

    step = np.linalg.norm(np.diff(projected, axis=0), axis=1)
    keep = np.r_[True, step > 0.0]
    trace_xy = clean_trace(projected)
    trace_lonlat = None if geographic is None else geographic[keep]
    return trace_xy, trace_lonlat


def _xy_to_lonlat(
    trace_xy: np.ndarray,
    segment_index: int,
    segment_fraction: float,
    *,
    trace_lonlat: np.ndarray | None,
    xy2ll: Callable[[Any, Any], tuple[Any, Any]] | None,
    x: float,
    y: float,
) -> tuple[float | None, float | None]:
    if xy2ll is not None:
        longitude, latitude = _call_transform(xy2ll, float(x), float(y))
        return float(longitude), float(latitude)
    if trace_lonlat is None:
        return None, None
    idx = int(segment_index)
    frac = float(segment_fraction)
    point = trace_lonlat[idx] + frac * (trace_lonlat[idx + 1] - trace_lonlat[idx])
    return float(point[0]), float(point[1])


def _make_marker(
    trace_xy: np.ndarray,
    *,
    x: float,
    y: float,
    trace_distance_km: float,
    segment_index: int,
    segment_fraction: float,
    method: str,
    trace_lonlat: np.ndarray | None,
    xy2ll: Callable[[Any, Any], tuple[Any, Any]] | None,
    distance_to_trace_km: float = 0.0,
) -> TraceMarker:
    lon, lat = _xy_to_lonlat(
        trace_xy,
        segment_index,
        segment_fraction,
        trace_lonlat=trace_lonlat,
        xy2ll=xy2ll,
        x=x,
        y=y,
    )
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


def _align_longitude(value: float, reference: float) -> float:
    """Return the longitude branch nearest ``reference``."""
    return float(reference + ((float(value) - reference + 180.0) % 360.0) - 180.0)


def _bisect_fraction(function: Callable[[float], float], left_value: float, right_value: float) -> float:
    left = 0.0
    right = 1.0
    f_left = float(left_value)
    f_right = float(right_value)
    if np.signbit(f_left) == np.signbit(f_right):
        raise ValueError("coordinate root must be bracketed by the trace segment.")
    for _ in range(60):
        middle = 0.5 * (left + right)
        f_middle = float(function(middle))
        if abs(f_middle) <= 1.0e-12 or right - left <= 1.0e-13:
            return middle
        if np.signbit(f_middle) == np.signbit(f_left):
            left = middle
            f_left = f_middle
        else:
            right = middle
    return 0.5 * (left + right)


def _coordinate_candidates(
    trace_xy: np.ndarray,
    *,
    axis: str,
    value: float,
    trace_lonlat: np.ndarray | None,
    xy2ll: Callable[[Any, Any], tuple[Any, Any]] | None,
    atol: float = 1.0e-10,
) -> list[TraceMarker]:
    key = str(axis).lower()
    target = float(value)
    if key in {"longitude", "lon"}:
        if trace_lonlat is None:
            raise ValueError("longitude markers require lonlat coordinates or an xy2ll transform.")
        coordinate = np.asarray(
            [_align_longitude(item, target) for item in trace_lonlat[:, 0]],
            dtype=float,
        )
        method = "longitude"
        axis_index = 0
        geographic = True
    elif key in {"latitude", "lat"}:
        if trace_lonlat is None:
            raise ValueError("latitude markers require lonlat coordinates or an xy2ll transform.")
        coordinate = trace_lonlat[:, 1]
        method = "latitude"
        axis_index = 1
        geographic = True
    elif key == "x":
        coordinate = trace_xy[:, 0]
        method = "x"
        axis_index = 0
        geographic = False
    elif key == "y":
        coordinate = trace_xy[:, 1]
        method = "y"
        axis_index = 1
        geographic = False
    else:
        raise ValueError("axis must be longitude/lon, latitude/lat, x, or y.")

    cumulative = np.r_[
        0.0,
        np.cumsum(np.linalg.norm(np.diff(trace_xy[:, :2], axis=0), axis=1)),
    ]
    candidates: list[TraceMarker] = []

    def value_on_segment(index: int, fraction: float) -> float:
        if geographic and xy2ll is not None:
            xy = trace_xy[index] + fraction * (trace_xy[index + 1] - trace_xy[index])
            longitude, latitude = _call_transform(xy2ll, xy[0], xy[1])
            result = float(longitude) if axis_index == 0 else float(latitude)
            return _align_longitude(result, target) if axis_index == 0 else result
        return float(coordinate[index] + fraction * (coordinate[index + 1] - coordinate[index]))

    def append_candidate(index: int, fraction: float) -> None:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        segment_length = cumulative[index + 1] - cumulative[index]
        distance = float(cumulative[index] + fraction * segment_length)
        for existing in candidates:
            if abs(existing.trace_distance_km - distance) <= 1.0e-9:
                return
        xy = trace_xy[index] + fraction * (trace_xy[index + 1] - trace_xy[index])
        candidates.append(
            _make_marker(
                trace_xy,
                x=xy[0],
                y=xy[1],
                trace_distance_km=distance,
                segment_index=index,
                segment_fraction=fraction,
                method=method,
                trace_lonlat=trace_lonlat,
                xy2ll=xy2ll,
            )
        )

    for index in range(trace_xy.shape[0] - 1):
        left = float(coordinate[index] - target)
        right = float(coordinate[index + 1] - target)
        left_zero = abs(left) <= atol
        right_zero = abs(right) <= atol
        if left_zero and right_zero:
            append_candidate(index, 0.0)
            append_candidate(index, 1.0)
            continue
        if left_zero:
            append_candidate(index, 0.0)
            continue
        if right_zero:
            append_candidate(index, 1.0)
            continue
        if np.signbit(left) == np.signbit(right):
            continue
        fraction = _bisect_fraction(
            lambda item, idx=index: value_on_segment(idx, item) - target,
            left,
            right,
        )
        append_candidate(index, fraction)

    candidates.sort(key=lambda marker: marker.trace_distance_km)
    return candidates


def _point_to_xy(
    point: Sequence[float],
    *,
    coord_system: str,
    ll2xy: Callable[[Any, Any], tuple[Any, Any]] | None,
) -> tuple[float, float]:
    values = np.asarray(point, dtype=float).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values[:2])):
        raise ValueError("point markers require two finite coordinates.")
    key = str(coord_system).lower().replace("_", "").replace("-", "")
    if key in {"xy", "utm", "local"}:
        return float(values[0]), float(values[1])
    if key in {"lonlat", "ll"}:
        if ll2xy is None:
            raise ValueError("lonlat point markers require an ll2xy transform.")
        x, y = _call_transform(ll2xy, float(values[0]), float(values[1]))
        return float(x), float(y)
    raise ValueError("coord_system must be 'lonlat' or 'xy'.")


def resolve_trace_markers(
    coords: Any,
    marker: Any,
    *,
    coord_system: str = "lonlat",
    lonlat: Any | None = None,
    ll2xy: Callable[[Any, Any], tuple[Any, Any]] | None = None,
    xy2ll: Callable[[Any, Any], tuple[Any, Any]] | None = None,
) -> tuple[TraceMarker, ...]:
    """Resolve a marker specification to all matching trace locations.

    Longitude and latitude cuts may return several ordered candidates.  Other
    marker forms return exactly one candidate.
    """
    if isinstance(marker, TraceMarker):
        return (marker,)

    trace_xy, aligned_lonlat = _prepare_trace(coords, lonlat=lonlat)
    trace_lonlat = _trace_lonlat(trace_xy, lonlat=aligned_lonlat, xy2ll=xy2ll)
    if isinstance(marker, Mapping):
        raw = dict(marker)
        by = raw.get("by")
        if "trace_distance_km" in raw or by == "trace_distance_km":
            value = raw.get("trace_distance_km", raw.get("value"))
            point = point_at_trace_distance(trace_xy, float(value))
            return (
                _make_marker(
                    trace_xy,
                    x=point["xy"][0],
                    y=point["xy"][1],
                    trace_distance_km=point["trace_distance_km"],
                    segment_index=point["segment_index"],
                    segment_fraction=point["segment_fraction"],
                    method="trace_distance_km",
                    trace_lonlat=trace_lonlat,
                    xy2ll=xy2ll,
                ),
            )
        if "fraction" in raw or by == "fraction":
            value = raw.get("fraction", raw.get("value"))
            fraction = float(value)
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError("trace marker fraction must be in [0, 1].")
            point = point_at_trace_distance(trace_xy, fraction * trace_length(trace_xy))
            return (
                _make_marker(
                    trace_xy,
                    x=point["xy"][0],
                    y=point["xy"][1],
                    trace_distance_km=point["trace_distance_km"],
                    segment_index=point["segment_index"],
                    segment_fraction=point["segment_fraction"],
                    method="fraction",
                    trace_lonlat=trace_lonlat,
                    xy2ll=xy2ll,
                ),
            )
        if "longitude" in raw or "lon" in raw or by in {"longitude", "lon"}:
            value = raw.get("longitude", raw.get("lon", raw.get("value")))
            return tuple(
                _coordinate_candidates(
                    trace_xy,
                    axis="longitude",
                    value=float(value),
                    trace_lonlat=trace_lonlat,
                    xy2ll=xy2ll,
                )
            )
        if "latitude" in raw or "lat" in raw or by in {"latitude", "lat"}:
            value = raw.get("latitude", raw.get("lat", raw.get("value")))
            return tuple(
                _coordinate_candidates(
                    trace_xy,
                    axis="latitude",
                    value=float(value),
                    trace_lonlat=trace_lonlat,
                    xy2ll=xy2ll,
                )
            )
        if "x" in raw or by == "x":
            value = raw.get("x", raw.get("value"))
            return tuple(
                _coordinate_candidates(
                    trace_xy,
                    axis="x",
                    value=float(value),
                    trace_lonlat=trace_lonlat,
                    xy2ll=xy2ll,
                )
            )
        if "y" in raw or by == "y":
            value = raw.get("y", raw.get("value"))
            return tuple(
                _coordinate_candidates(
                    trace_xy,
                    axis="y",
                    value=float(value),
                    trace_lonlat=trace_lonlat,
                    xy2ll=xy2ll,
                )
            )
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
        elif by in {"point", "nearest"}:
            point = raw.get("value")
            point_coord_system = raw.get("coord_system", coord_system)
        else:
            allowed = "longitude, latitude, trace_distance_km, fraction, point/nearest"
            raise ValueError(f"trace marker mapping must define one of: {allowed}.")
        max_distance_km = raw.get("max_distance_km")
    else:
        point = marker
        point_coord_system = coord_system
        max_distance_km = None

    xy = np.asarray(
        _point_to_xy(point, coord_system=point_coord_system, ll2xy=ll2xy),
        dtype=float,
    )
    projection = project_points_to_trace(xy, trace_xy)
    distance_to_trace = float(projection["distance_to_trace_km"][0])
    if max_distance_km is not None and float(max_distance_km) < 0.0:
        raise ValueError("max_distance_km must be non-negative.")
    if max_distance_km is not None and distance_to_trace > float(max_distance_km):
        raise ValueError(
            f"nearest trace point is {distance_to_trace:.6g} km away, exceeding "
            f"max_distance_km={float(max_distance_km):.6g}."
        )
    projected_xy = projection["projected_xy"][0]
    return (
        _make_marker(
            trace_xy,
            x=projected_xy[0],
            y=projected_xy[1],
            trace_distance_km=projection["trace_distance_km"][0],
            segment_index=projection["segment_index"][0],
            segment_fraction=projection["segment_fraction"][0],
            distance_to_trace_km=distance_to_trace,
            method="nearest",
            trace_lonlat=trace_lonlat,
            xy2ll=xy2ll,
        ),
    )


def resolve_trace_marker(
    coords: Any,
    marker: Any,
    *,
    coord_system: str = "lonlat",
    lonlat: Any | None = None,
    ll2xy: Callable[[Any, Any], tuple[Any, Any]] | None = None,
    xy2ll: Callable[[Any, Any], tuple[Any, Any]] | None = None,
    which: str | int | None = None,
) -> TraceMarker:
    """Resolve one marker, selecting an ordered coordinate intersection."""
    candidates = resolve_trace_markers(
        coords,
        marker,
        coord_system=coord_system,
        lonlat=lonlat,
        ll2xy=ll2xy,
        xy2ll=xy2ll,
    )
    if not candidates:
        raise ValueError("trace marker does not intersect the fault trace.")
    if which is None and isinstance(marker, Mapping):
        which = marker.get("which", "first")
    if which is None:
        which = "first"
    if isinstance(which, (int, np.integer)):
        index = int(which)
    else:
        key = str(which).lower()
        if key == "first":
            index = 0
        elif key == "last":
            index = len(candidates) - 1
        else:
            try:
                index = int(key)
            except ValueError as exc:
                raise ValueError("which must be 'first', 'last', or an integer index.") from exc
    try:
        selected = candidates[index]
    except IndexError as exc:
        raise IndexError(f"trace marker intersection index {index} is out of range.") from exc
    normalized_index = index if index >= 0 else len(candidates) + index
    return replace(
        selected,
        candidate_count=len(candidates),
        candidate_index=normalized_index,
    )


__all__ = [
    "TraceMarker",
    "resolve_trace_marker",
    "resolve_trace_markers",
]
