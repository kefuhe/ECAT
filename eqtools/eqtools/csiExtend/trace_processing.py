"""High-level, immutable fault-trace preprocessing for scripts and CLIs.

``trace_ops`` remains the numerical kernel.  This module adds projection,
user-facing marker resolution, operation history, and ordered composition
without mutating CSI fault objects.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from .trace_markers import TraceMarker, resolve_trace_marker, resolve_trace_markers
from .trace_ops import (
    clean_trace,
    extend_trace,
    orient_trace,
    resample_trace,
    reverse_trace,
    simplify_trace,
    smooth_trace,
    trace_length,
    trim_trace,
)


def _automatic_longitude_center(longitude: np.ndarray) -> float:
    radians = np.deg2rad(np.asarray(longitude, dtype=float))
    angle = np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    return float(((angle + 180.0) % 360.0) - 180.0)


def _readonly_coordinates(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must be a 2-D array with at least two columns.")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two points.")
    array = np.array(array[:, :2], dtype=float, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite coordinates.")
    array.setflags(write=False)
    return array


def _freeze_parameter(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_parameter(item) for key, item in value.items()}
        )
    if isinstance(value, np.ndarray):
        return tuple(_freeze_parameter(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_parameter(item) for item in value)
    return deepcopy(value)


def _thaw_parameter(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_parameter(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_parameter(item) for item in value]
    return deepcopy(value)


class TraceProjection:
    """Small composition wrapper around the CSI projection engine."""

    def __init__(
        self,
        lon0: float,
        lat0: float,
        *,
        utmzone: int | None = None,
        ellps: str = "WGS84",
        name: str = "TraceProjection",
    ) -> None:
        from csi import SourceInv

        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.utmzone = utmzone
        self.ellps = str(ellps)
        self._source = SourceInv(
            name,
            utmzone=utmzone,
            ellps=ellps,
            lon0=self.lon0,
            lat0=self.lat0,
        )

    def ll2xy(self, longitude: Any, latitude: Any) -> tuple[Any, Any]:
        return self._source.ll2xy(longitude, latitude)

    def xy2ll(self, x: Any, y: Any) -> tuple[Any, Any]:
        return self._source.xy2ll(x, y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lon0": self.lon0,
            "lat0": self.lat0,
            "utmzone": self.utmzone,
            "ellps": self.ellps,
        }


@dataclass(frozen=True)
class TraceOperation:
    """One recorded immutable trace-processing step."""

    name: str
    parameters: Mapping[str, Any]
    input_points: int
    output_points: int
    input_length_km: float
    output_length_km: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _freeze_parameter(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.name,
            "parameters": _thaw_parameter(self.parameters),
            "input_points": self.input_points,
            "output_points": self.output_points,
            "input_length_km": self.input_length_km,
            "output_length_km": self.output_length_km,
        }


@dataclass(frozen=True)
class TracePath:
    """One projected trace with geographic coordinates and operation history.

    All transformation methods return a new instance.  ``projection`` may be a
    :class:`TraceProjection` or any existing object that provides compatible
    ``ll2xy`` and ``xy2ll`` methods, including a CSI fault used only as a
    projection provider.
    """

    xy: np.ndarray
    lonlat: np.ndarray
    projection: Any = field(repr=False, compare=False)
    history: tuple[TraceOperation, ...] = ()

    def __post_init__(self) -> None:
        xy = _readonly_coordinates(self.xy, name="xy")
        lonlat = _readonly_coordinates(self.lonlat, name="lonlat")
        if xy.shape != lonlat.shape:
            raise ValueError("xy and lonlat coordinates must be aligned.")
        if not hasattr(self.projection, "ll2xy") or not hasattr(self.projection, "xy2ll"):
            raise TypeError("projection must provide ll2xy() and xy2ll() methods.")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "lonlat", lonlat)
        object.__setattr__(self, "history", tuple(self.history))

    @classmethod
    def from_lonlat(
        cls,
        coords: Any,
        *,
        lon0: float | None = None,
        lat0: float | None = None,
        utmzone: int | None = None,
        ellps: str = "WGS84",
        projection: Any | None = None,
    ) -> "TracePath":
        lonlat = _readonly_coordinates(coords, name="coords")
        if projection is None:
            resolved_lon0 = (
                _automatic_longitude_center(lonlat[:, 0]) if lon0 is None else float(lon0)
            )
            resolved_lat0 = float(np.mean(lonlat[:, 1])) if lat0 is None else float(lat0)
            projection = TraceProjection(
                resolved_lon0,
                resolved_lat0,
                utmzone=utmzone,
                ellps=ellps,
            )
        x, y = projection.ll2xy(lonlat[:, 0], lonlat[:, 1])
        xy = np.column_stack((x, y))
        return cls(xy=xy, lonlat=lonlat, projection=projection)

    @property
    def point_count(self) -> int:
        return int(self.xy.shape[0])

    @property
    def length_km(self) -> float:
        return trace_length(self.xy)

    def resolve_marker(self, marker: Any, *, which: str | int | None = None) -> TraceMarker:
        return resolve_trace_marker(
            self.xy,
            marker,
            lonlat=self.lonlat,
            ll2xy=self.projection.ll2xy,
            xy2ll=self.projection.xy2ll,
            which=which,
        )

    def resolve_markers(self, marker: Any) -> tuple[TraceMarker, ...]:
        return resolve_trace_markers(
            self.xy,
            marker,
            lonlat=self.lonlat,
            ll2xy=self.projection.ll2xy,
            xy2ll=self.projection.xy2ll,
        )

    def _spawn(self, xy: Any, operation: str, parameters: Mapping[str, Any]) -> "TracePath":
        output_xy = clean_trace(xy)[:, :2]
        longitude, latitude = self.projection.xy2ll(output_xy[:, 0], output_xy[:, 1])
        output_lonlat = np.column_stack((longitude, latitude))
        record = TraceOperation(
            name=operation,
            parameters=dict(parameters),
            input_points=self.point_count,
            output_points=int(output_xy.shape[0]),
            input_length_km=self.length_km,
            output_length_km=trace_length(output_xy),
        )
        return TracePath(
            xy=output_xy,
            lonlat=output_lonlat,
            projection=self.projection,
            history=(*self.history, record),
        )

    def clean(self, *, atol_km: float = 0.0) -> "TracePath":
        return self._spawn(
            clean_trace(self.xy, atol=float(atol_km)),
            "clean",
            {"atol_km": float(atol_km)},
        )

    def orient(self, *, start: str = "west") -> "TracePath":
        return self._spawn(orient_trace(self.xy, start=start), "orient", {"start": start})

    def reverse(self) -> "TracePath":
        return self._spawn(reverse_trace(self.xy), "reverse", {})

    def trim(self, *, start: Any | None = None, end: Any | None = None) -> "TracePath":
        start_marker = None if start is None else self.resolve_marker(start)
        end_marker = None if end is None else self.resolve_marker(end)
        start_distance = None if start_marker is None else start_marker.trace_distance_km
        end_distance = None if end_marker is None else end_marker.trace_distance_km
        parameters: dict[str, Any] = {
            "start": start,
            "end": end,
        }
        if start_marker is not None:
            parameters["resolved_start"] = start_marker.to_dict()
        if end_marker is not None:
            parameters["resolved_end"] = end_marker.to_dict()
        return self._spawn(
            trim_trace(self.xy, start=start_distance, end=end_distance),
            "trim",
            parameters,
        )

    def extend(
        self,
        *,
        start_km: float = 0.0,
        end_km: float = 0.0,
        tangent_window: int = 1,
        mode: str = "endpoint_tangent",
    ) -> "TracePath":
        parameters = {
            "start_km": float(start_km),
            "end_km": float(end_km),
            "tangent_window": int(tangent_window),
            "mode": mode,
        }
        return self._spawn(
            extend_trace(
                self.xy,
                start=start_km,
                end=end_km,
                tangent_window=tangent_window,
                mode=mode,
            ),
            "extend",
            parameters,
        )

    def resample(
        self,
        *,
        every_km: float | None = None,
        num_points: int | None = None,
        keep_endpoints: bool = True,
    ) -> "TracePath":
        parameters = {
            "every_km": every_km,
            "num_points": num_points,
            "keep_endpoints": bool(keep_endpoints),
        }
        return self._spawn(
            resample_trace(
                self.xy,
                every=every_km,
                num_points=num_points,
                keep_endpoints=keep_endpoints,
            ),
            "resample",
            parameters,
        )

    def simplify(self, *, method: str = "rdp", tolerance: float = 1.0) -> "TracePath":
        return self._spawn(
            simplify_trace(self.xy, method=method, tolerance=tolerance),
            "simplify",
            {"method": method, "tolerance": float(tolerance)},
        )

    def smooth(
        self,
        *,
        method: str = "bspline",
        smoothing: float = 1.0,
        num_points: int | None = None,
        window: int = 5,
        polyorder: int = 2,
        preserve_endpoints: bool = True,
    ) -> "TracePath":
        parameters = {
            "method": method,
            "smoothing": float(smoothing),
            "num_points": num_points,
            "window": int(window),
            "polyorder": int(polyorder),
            "preserve_endpoints": bool(preserve_endpoints),
        }
        return self._spawn(
            smooth_trace(
                self.xy,
                method=method,
                smoothing=smoothing,
                num_points=num_points,
                window=window,
                polyorder=polyorder,
                preserve_endpoints=preserve_endpoints,
            ),
            "smooth",
            parameters,
        )

    def apply(self, operation: Mapping[str, Any]) -> "TracePath":
        """Apply one explicit operation mapping."""
        spec = dict(operation)
        name = str(spec.pop("op", spec.pop("operation", ""))).lower()
        if not name:
            raise ValueError("trace operation must define 'op'.")
        handlers = {
            "clean": self.clean,
            "orient": self.orient,
            "reverse": self.reverse,
            "trim": self.trim,
            "extend": self.extend,
            "resample": self.resample,
            "simplify": self.simplify,
            "smooth": self.smooth,
        }
        if name not in handlers:
            raise ValueError(f"unsupported trace operation: {name}.")
        return handlers[name](**spec)

    def report(self) -> dict[str, Any]:
        projection = (
            self.projection.to_dict()
            if hasattr(self.projection, "to_dict")
            else {"type": type(self.projection).__name__}
        )
        return {
            "point_count": self.point_count,
            "length_km": self.length_km,
            "start_lonlat": self.lonlat[0].tolist(),
            "end_lonlat": self.lonlat[-1].tolist(),
            "projection": projection,
            "operations": [item.to_dict() for item in self.history],
        }


def process_trace(trace: TracePath, operations: Iterable[Mapping[str, Any]]) -> TracePath:
    """Apply ordered operation mappings to a :class:`TracePath`."""
    if not isinstance(trace, TracePath):
        raise TypeError("process_trace expects a TracePath.")
    result = trace
    for operation in operations:
        result = result.apply(operation)
    return result


__all__ = [
    "TraceOperation",
    "TracePath",
    "TraceProjection",
    "process_trace",
]
