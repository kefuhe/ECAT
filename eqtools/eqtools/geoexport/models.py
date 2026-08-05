"""Detached display models used by :mod:`eqtools.geoexport`.

The classes in this module deliberately contain no CSI solver objects.  They
form a small, read-only boundary between scientific data adapters and display
writers such as Google Earth KMZ.
"""

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import math
import re

import numpy as np


_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def _readonly_array(values, *, dtype=None):
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _frozen_mapping(values):
    return MappingProxyType(dict(values or {}))


def _deep_freeze(value):
    """Recursively detach JSON-like geometry and property structures."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _validate_layer_id(value):
    text = str(value).strip()
    if not _ID_PATTERN.fullmatch(text):
        raise ValueError(
            "Layer id must start with a letter and contain only letters, "
            f"digits, '.', '_' or '-'; got {value!r}."
        )
    return text


def _validate_layer_name(value):
    if value is None:
        raise ValueError("Layer name must not be empty.")
    text = str(value).strip()
    if not text:
        raise ValueError("Layer name must not be empty.")
    return text


@dataclass(frozen=True)
class LayerStyle:
    """Display-only styling shared by raster and vector layers.

    Parameters
    ----------
    cmap : str, default "viridis"
        Matplotlib colormap used for quantitative values.
    vmin, vmax : float, optional
        Fixed color limits after applying ``display_factor``. Set both or
        neither. Explicit limits take priority over ``symmetry``.
    symmetry : bool, default False
        When limits are automatic, center the color range on zero. This is
        intended for signed deformation-like values.
    alpha : float, default 0.8
        Layer opacity from zero (transparent) to one (opaque).
    display_factor : float, default 1
        Value multiplier used only for labels and color mapping.
    display_unit : str, optional
        Unit shown in the legend after applying ``display_factor``.
    normalization : {"linear", "cyclic"}, default "linear"
        Display normalization. ``cyclic`` wraps values by ``cyclic_period``.
    cyclic_period : float, optional
        Positive display period required for cyclic normalization.
    line_color : str
        CSS-style ``#RRGGBB`` color for unscaled vector features.
    line_width : float, default 1.5
        Google Earth line width.
    point_scale : float, default 0.8
        Google Earth point-icon scale.
    """

    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    symmetry: bool = False
    alpha: float = 0.8
    display_factor: float = 1.0
    display_unit: str | None = None
    normalization: str = "linear"
    cyclic_period: float | None = None
    line_color: str = "#ffffff"
    line_width: float = 1.5
    point_scale: float = 0.8

    def __post_init__(self):
        if isinstance(self.alpha, bool):
            raise ValueError("Layer alpha must be finite and within [0, 1].")
        alpha = float(self.alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("Layer alpha must be finite and within [0, 1].")
        object.__setattr__(self, "alpha", alpha)
        if not isinstance(self.symmetry, bool):
            raise ValueError("symmetry must be a boolean.")
        limits_are_set = self.vmin is not None or self.vmax is not None
        if limits_are_set and (self.vmin is None or self.vmax is None):
            raise ValueError("vmin and vmax must be set together or both be None.")
        if limits_are_set:
            if isinstance(self.vmin, bool) or isinstance(self.vmax, bool):
                raise ValueError("vmin and vmax must be finite numbers.")
            vmin = float(self.vmin)
            vmax = float(self.vmax)
            if not np.all(np.isfinite((vmin, vmax))) or vmin >= vmax:
                raise ValueError(
                    "vmin and vmax must be finite and strictly increasing."
                )
            object.__setattr__(self, "vmin", vmin)
            object.__setattr__(self, "vmax", vmax)
        if (
            isinstance(self.display_factor, bool)
            or not np.isfinite(float(self.display_factor))
            or np.isclose(float(self.display_factor), 0.0)
        ):
            raise ValueError("display_factor must be finite and non-zero.")
        normalization = str(self.normalization).strip().lower()
        if normalization not in {"linear", "cyclic"}:
            raise ValueError("normalization must be 'linear' or 'cyclic'.")
        object.__setattr__(self, "normalization", normalization)
        if normalization == "cyclic":
            if self.symmetry:
                raise ValueError(
                    "symmetry is not supported with cyclic normalization."
                )
            if (
                isinstance(self.cyclic_period, bool)
                or self.cyclic_period is None
                or not math.isfinite(float(self.cyclic_period))
                or float(self.cyclic_period) <= 0.0
            ):
                raise ValueError(
                    "cyclic_period must be positive for cyclic normalization."
                )
            if limits_are_set and not np.isclose(
                float(self.vmax) - float(self.vmin),
                float(self.cyclic_period),
            ):
                raise ValueError(
                    "Cyclic vmin/vmax must span exactly one cyclic_period."
                )
        for field_name in ("line_color",):
            color = str(getattr(self, field_name))
            if len(color) != 7 or not color.startswith("#"):
                raise ValueError(f"{field_name} must use '#RRGGBB'.")
            try:
                int(color[1:], 16)
            except ValueError as exc:
                raise ValueError(f"{field_name} must use '#RRGGBB'.") from exc
        if not math.isfinite(float(self.line_width)) or float(self.line_width) <= 0.0:
            raise ValueError("line_width must be finite and positive.")
        if not math.isfinite(float(self.point_scale)) or float(self.point_scale) <= 0.0:
            raise ValueError("point_scale must be finite and positive.")


@dataclass(frozen=True)
class RasterLayer:
    """One scalar raster with exact per-pixel geographic coordinates."""

    id: str
    name: str
    values: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    mask: np.ndarray | None = None
    topology: str = "geographic_rectilinear"
    units: str | None = None
    convention: str | None = None
    metadata: Mapping = field(default_factory=dict)
    style: LayerStyle = field(default_factory=LayerStyle)
    visible: bool = True

    def __post_init__(self):
        object.__setattr__(self, "id", _validate_layer_id(self.id))
        object.__setattr__(self, "name", _validate_layer_name(self.name))
        values = _readonly_array(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError("Raster values must be two-dimensional.")
        longitude = np.asarray(self.longitude, dtype=float)
        latitude = np.asarray(self.latitude, dtype=float)
        if longitude.ndim == 1 and latitude.ndim == 1:
            if longitude.size != values.shape[1] or latitude.size != values.shape[0]:
                raise ValueError(
                    "Raster longitude/latitude axes do not match value shape."
                )
            longitude, latitude = np.meshgrid(longitude, latitude)
        if longitude.shape != values.shape or latitude.shape != values.shape:
            raise ValueError(
                "Raster longitude/latitude meshes must match value shape."
            )
        if self.mask is None:
            mask = np.isfinite(values)
        else:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != values.shape:
                raise ValueError("Raster mask must match value shape.")
            mask = mask & np.isfinite(values)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "longitude", _readonly_array(longitude, dtype=float))
        object.__setattr__(self, "latitude", _readonly_array(latitude, dtype=float))
        object.__setattr__(self, "mask", _readonly_array(mask, dtype=bool))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))
        if not isinstance(self.visible, bool):
            raise ValueError("Raster layer visible must be a boolean.")


@dataclass(frozen=True)
class VectorLayer:
    """GeoJSON-like point, line, or polygon features."""

    id: str
    name: str
    features: tuple
    units: str | None = None
    convention: str | None = None
    value_property: str | None = "value"
    metadata: Mapping = field(default_factory=dict)
    style: LayerStyle = field(default_factory=LayerStyle)
    visible: bool = True

    def __post_init__(self):
        object.__setattr__(self, "id", _validate_layer_id(self.id))
        object.__setattr__(self, "name", _validate_layer_name(self.name))
        frozen_features = []
        for index, feature in enumerate(self.features):
            if not isinstance(feature, Mapping):
                raise ValueError(f"Feature {index} must be a mapping.")
            geometry = feature.get("geometry")
            if not isinstance(geometry, Mapping):
                raise ValueError(f"Feature {index} requires a geometry mapping.")
            geometry_type = geometry.get("type")
            if geometry_type not in {"Point", "LineString", "Polygon"}:
                raise ValueError(
                    f"Unsupported feature geometry {geometry_type!r}; "
                    "expected Point, LineString, or Polygon."
                )
            raw_coordinates = geometry.get("coordinates")
            if geometry_type == "Point":
                coordinates = np.asarray(raw_coordinates, dtype=float)
                valid_shape = coordinates.ndim == 1 and coordinates.size in {2, 3}
                normalized_coordinates = coordinates.tolist()
            elif geometry_type == "LineString":
                coordinates = np.asarray(raw_coordinates, dtype=float)
                valid_shape = (
                    coordinates.ndim == 2
                    and coordinates.shape[0] >= 2
                    and coordinates.shape[1] in {2, 3}
                )
                normalized_coordinates = coordinates.tolist()
            else:
                valid_shape = isinstance(raw_coordinates, (list, tuple))
                normalized_coordinates = []
                dimensions = set()
                if valid_shape:
                    for ring in raw_coordinates:
                        ring = np.asarray(ring, dtype=float)
                        if (
                            ring.ndim != 2
                            or ring.shape[0] < 3
                            or ring.shape[1] not in {2, 3}
                            or not np.all(np.isfinite(ring))
                        ):
                            valid_shape = False
                            break
                        dimensions.add(ring.shape[1])
                        normalized_coordinates.append(ring.tolist())
                    valid_shape = (
                        valid_shape
                        and bool(normalized_coordinates)
                        and len(dimensions) == 1
                    )
                coordinates = np.asarray(
                    normalized_coordinates[0] if normalized_coordinates else [],
                    dtype=float,
                )
            if not valid_shape or not np.all(np.isfinite(coordinates)):
                raise ValueError(
                    f"Feature {index} has invalid {geometry_type} coordinates."
                )
            if geometry_type == "Point":
                geographic = np.asarray(normalized_coordinates, dtype=float)[None, :]
            elif geometry_type == "LineString":
                geographic = np.asarray(normalized_coordinates, dtype=float)
            else:
                geographic = np.vstack(
                    [
                        np.asarray(ring, dtype=float)
                        for ring in normalized_coordinates
                    ]
                )
            if np.any((geographic[:, 0] < -360.0) | (geographic[:, 0] > 360.0)):
                raise ValueError(
                    f"Feature {index} longitude is outside [-360, 360]."
                )
            if np.any((geographic[:, 1] < -90.0) | (geographic[:, 1] > 90.0)):
                raise ValueError(
                    f"Feature {index} latitude is outside [-90, 90]."
                )
            frozen_features.append(
                _deep_freeze(
                    {
                    "geometry": {
                        "type": geometry_type,
                        "coordinates": normalized_coordinates,
                    },
                    "properties": dict(feature.get("properties") or {}),
                    }
                )
            )
        object.__setattr__(self, "features", tuple(frozen_features))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))
        if not isinstance(self.visible, bool):
            raise ValueError("Vector layer visible must be a boolean.")


@dataclass(frozen=True)
class ExportResult:
    """Summary returned after one export operation."""

    output_files: tuple
    layer_ids: tuple
    warnings: tuple = ()
    package_mode: str = "single"

    def __post_init__(self):
        object.__setattr__(
            self,
            "output_files",
            tuple(Path(path) for path in self.output_files),
        )
        object.__setattr__(self, "layer_ids", tuple(self.layer_ids))
        object.__setattr__(self, "warnings", tuple(self.warnings))


__all__ = ["ExportResult", "LayerStyle", "RasterLayer", "VectorLayer"]
