"""Small, renderer-independent models for the ECAT research map viewer."""

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import math
import re


SUPPORTED_LAYER_KINDS = (
    "earthquake_catalog",
    "vector",
    "gnss_velocity",
    "observation_grid",
    "raster",
    "csi_varres",
)
SUPPORTED_BASEMAPS = (
    "open-street-map",
    "carto-positron",
    "carto-darkmatter",
    "streets",
    "outdoors",
    "satellite",
    "satellite-streets",
    "white-bg",
)

_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_COMMON_QUANTITATIVE_STYLE_FIELDS = {
    "alpha",
    "cmap",
    "display_factor",
    "display_unit",
    "symmetry",
    "vmax",
    "vmin",
}
_STYLE_FIELDS_BY_KIND = {
    "earthquake_catalog": _COMMON_QUANTITATIVE_STYLE_FIELDS,
    "vector": {
        "alpha",
        "color",
        "line_width",
        "marker_size",
    },
    "gnss_velocity": {
        "alpha",
        "color",
        "display_scale",
        "line_width",
        "marker_size",
    },
    "observation_grid": _COMMON_QUANTITATIVE_STYLE_FIELDS
    | {"marker_size"},
    "raster": _COMMON_QUANTITATIVE_STYLE_FIELDS | {"marker_size"},
    "csi_varres": _COMMON_QUANTITATIVE_STYLE_FIELDS
    | {
        "color",
        "line_width",
        "marker_size",
    },
}
_POSITIVE_STYLE_FIELDS = {
    "display_scale",
    "line_width",
    "marker_size",
}
_FORMAT_VALUES = {
    "vector": {"geojson", "gmt"},
}
_DATA_TYPE_VALUES = {
    "csi_varres": {"sar", "optical"},
}
_OBSERVATION_MASKS = {"source_valid", "analysis_valid", "finite"}


def _validate_style(style, *, kind, layer_id):
    """Return one validated canonical display-style mapping."""

    style = dict(style or {})
    allowed = _STYLE_FIELDS_BY_KIND[kind]
    unknown = sorted(set(style) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported style fields for {kind} layer {layer_id!r}: "
            f"{unknown}."
        )
    alpha = style.get("alpha")
    if alpha is not None:
        if isinstance(alpha, bool):
            raise ValueError("Layer alpha must be finite and within [0, 1].")
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("Layer alpha must be finite and within [0, 1].")
        style["alpha"] = alpha
    symmetry = style.get("symmetry")
    if symmetry is not None and not isinstance(symmetry, bool):
        raise ValueError("Layer style.symmetry must be a boolean.")
    limits_are_set = style.get("vmin") is not None or style.get("vmax") is not None
    if limits_are_set:
        if style.get("vmin") is None or style.get("vmax") is None:
            raise ValueError(
                "Layer style.vmin and style.vmax must be set together."
            )
        if isinstance(style["vmin"], bool) or isinstance(style["vmax"], bool):
            raise ValueError("Layer style.vmin/vmax must be finite numbers.")
        lower = float(style["vmin"])
        upper = float(style["vmax"])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(
                "Layer style.vmin/vmax must be finite and strictly increasing."
            )
        style["vmin"] = lower
        style["vmax"] = upper
    factor = style.get("display_factor")
    if factor is not None:
        if (
            isinstance(factor, bool)
            or not math.isfinite(float(factor))
            or math.isclose(float(factor), 0.0)
        ):
            raise ValueError(
                "Layer style.display_factor must be finite and non-zero."
            )
        style["display_factor"] = float(factor)
    unit = style.get("display_unit")
    if unit is not None:
        unit = str(unit).strip()
        if not unit:
            raise ValueError("Layer style.display_unit must not be empty.")
        style["display_unit"] = unit
    for field_name in _POSITIVE_STYLE_FIELDS:
        value = style.get(field_name)
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Layer style.{field_name} must be finite and positive."
                )
            style[field_name] = value
    for field_name in ("cmap", "color"):
        value = style.get(field_name)
        if value is not None:
            value = str(value).strip()
            if not value:
                raise ValueError(f"Layer style.{field_name} must not be empty.")
            style[field_name] = value
    return style


def _frozen_mapping(values):
    return MappingProxyType(dict(values or {}))


def _freeze_payload(value):
    """Recursively detach JSON-like payload containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_payload(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    return value


@dataclass(frozen=True)
class LayerSpec:
    """Stable declaration of one viewer layer.

    ``source`` is already resolved against the project file. Scientific
    interpretation is selected by ``kind``; ``format`` is only an optional
    file-encoding disambiguator, while ``mask`` applies only to a standard
    observation grid.
    """

    id: str
    name: str
    kind: str
    source: Path
    variable: str | None = None
    visible: bool = False
    style: Mapping[str, Any] = field(default_factory=dict)
    format: str | None = None
    data_type: str | None = None
    mask: str | None = None

    def __post_init__(self):
        layer_id = str(self.id).strip()
        if not _ID_PATTERN.fullmatch(layer_id):
            raise ValueError(
                "Layer id must start with a letter and contain only letters, "
                f"digits, '.', '_' or '-'; got {self.id!r}."
            )
        name = str(self.name).strip()
        if not name:
            raise ValueError("Layer name cannot be empty.")
        kind = str(self.kind).strip().lower()
        if kind not in SUPPORTED_LAYER_KINDS:
            raise ValueError(
                f"Unsupported layer kind {kind!r}; expected one of "
                f"{SUPPORTED_LAYER_KINDS}."
            )
        style = _validate_style(self.style, kind=kind, layer_id=layer_id)
        if not isinstance(self.visible, bool):
            raise ValueError("Layer visible must be a YAML boolean.")

        variable = self.variable
        if variable is not None:
            variable = str(variable).strip()
            if not variable:
                raise ValueError("Layer variable must not be empty.")
        if kind == "observation_grid" and not variable:
            raise ValueError(
                f"Observation-grid layer {layer_id!r} requires an explicit "
                "variable so raw and corrected values are never confused."
            )
        if kind == "raster" and variable not in {None, "band_1"}:
            raise ValueError(
                f"Raster layer {layer_id!r} currently reads only band_1; "
                f"got variable={variable!r}."
            )
        if kind not in {"observation_grid", "raster", "csi_varres"} and variable:
            raise ValueError(f"Layer kind {kind!r} does not use variable.")

        mask = self.mask
        if mask is not None:
            mask = str(mask).strip().lower()
        if kind == "observation_grid":
            mask = mask or "source_valid"
            if mask not in _OBSERVATION_MASKS:
                raise ValueError(
                    "Observation-grid mask must be source_valid, "
                    "analysis_valid, or finite."
                )
        elif mask is not None:
            raise ValueError(f"Layer kind {kind!r} does not use mask.")

        format_name = self.format
        if format_name is not None:
            format_name = str(format_name).strip().lower()
            if not format_name:
                raise ValueError("Layer format must not be empty.")
        allowed_formats = _FORMAT_VALUES.get(kind)
        if format_name is not None and allowed_formats is None:
            raise ValueError(f"Layer kind {kind!r} does not use format.")
        if (
            format_name is not None
            and allowed_formats is not None
            and format_name not in allowed_formats
        ):
            raise ValueError(
                f"Layer kind {kind!r} format must be one of "
                f"{sorted(allowed_formats)}; got {format_name!r}."
            )
        data_type = self.data_type
        if data_type is not None:
            data_type = str(data_type).strip().lower()
            if not data_type:
                raise ValueError("Layer data_type must not be empty.")
        allowed_data_types = _DATA_TYPE_VALUES.get(kind)
        if data_type is not None and allowed_data_types is None:
            raise ValueError(f"Layer kind {kind!r} does not use data_type.")
        if kind == "csi_varres":
            data_type = data_type or "sar"
        if (
            data_type is not None
            and allowed_data_types is not None
            and data_type not in allowed_data_types
        ):
            raise ValueError(
                f"Layer kind {kind!r} data_type must be one of "
                f"{sorted(allowed_data_types)}; got {data_type!r}."
            )
        object.__setattr__(self, "id", layer_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source", Path(self.source).resolve())
        object.__setattr__(self, "variable", variable)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "format", format_name)
        object.__setattr__(self, "data_type", data_type)
        object.__setattr__(self, "style", _frozen_mapping(style))


@dataclass(frozen=True)
class LayerMetadata:
    """Small scientific and file metadata returned by a layer loader."""

    layer_id: str
    bbox: tuple[float, float, float, float] | None
    fingerprint: str
    feature_count: int | None = None
    shape: tuple[int, ...] | None = None
    available_variables: tuple[str, ...] = ()
    units: str | None = None
    crs: str | None = None
    grid_topology: str | None = None
    loader_version: int = 1
    derived_display: bool = False


@dataclass(frozen=True)
class LayerPayload:
    """Detached loader result passed to a renderer without mutable containers."""

    spec: LayerSpec
    metadata: LayerMetadata
    data: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "data", _freeze_payload(self.data))


@dataclass(frozen=True)
class ViewerState:
    """Small per-browser state; scientific payloads never live here."""

    basemap: str = "open-street-map"
    viewport: Mapping[str, Any] = field(default_factory=dict)
    visible_layer_ids: tuple[str, ...] = ()
    active_layer_id: str | None = None
    style_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "viewport", _frozen_mapping(self.viewport))
        object.__setattr__(
            self,
            "visible_layer_ids",
            tuple(dict.fromkeys(self.visible_layer_ids)),
        )
        object.__setattr__(
            self,
            "style_overrides",
            _frozen_mapping(self.style_overrides),
        )
        active_layer_id = self.active_layer_id
        if active_layer_id is not None:
            active_layer_id = str(active_layer_id).strip() or None
        object.__setattr__(self, "active_layer_id", active_layer_id)

    def to_dict(self):
        """Return JSON-safe session state for ``dcc.Store``."""

        return {
            "basemap": self.basemap,
            "viewport": dict(self.viewport),
            "visible_layer_ids": list(self.visible_layer_ids),
            "active_layer_id": self.active_layer_id,
            "style_overrides": dict(self.style_overrides),
        }

    @classmethod
    def from_dict(cls, values):
        """Build state from a possibly missing browser-store value."""

        values = values or {}
        return cls(
            basemap=values.get("basemap", "open-street-map"),
            viewport=values.get("viewport") or {},
            visible_layer_ids=tuple(values.get("visible_layer_ids") or ()),
            active_layer_id=values.get("active_layer_id"),
            style_overrides=values.get("style_overrides") or {},
        )


@dataclass(frozen=True)
class ViewerProject:
    """Parsed viewer project with resolved layer sources."""

    name: str
    path: Path | None
    layers: tuple[LayerSpec, ...]
    region: tuple[float, float, float, float] | None = None
    basemap: str = "open-street-map"

    def __post_init__(self):
        name = str(self.name).strip()
        if not name:
            raise ValueError("Viewer project name cannot be empty.")
        basemap = str(self.basemap).strip()
        if basemap not in SUPPORTED_BASEMAPS:
            raise ValueError(
                f"Viewer basemap must be one of {SUPPORTED_BASEMAPS}; "
                f"got {self.basemap!r}."
            )
        ids = [layer.id for layer in self.layers]
        duplicates = sorted(
            layer_id for layer_id in set(ids) if ids.count(layer_id) > 1
        )
        if duplicates:
            raise ValueError(f"Viewer layer ids must be unique: {duplicates}.")
        if self.region is not None:
            if len(self.region) != 4:
                raise ValueError(
                    "Viewer region must be [min_lon, max_lon, min_lat, max_lat]."
                )
            west, east, south, north = map(float, self.region)
            if not all(
                math.isfinite(value)
                for value in (west, east, south, north)
            ):
                raise ValueError("Viewer region values must be finite.")
            if west >= east or south >= north:
                raise ValueError(
                    "Viewer region requires min_lon < max_lon and "
                    "min_lat < max_lat."
                )
            object.__setattr__(self, "region", (west, east, south, north))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "basemap", basemap)
        object.__setattr__(self, "layers", tuple(self.layers))


__all__ = [
    "LayerMetadata",
    "LayerPayload",
    "LayerSpec",
    "SUPPORTED_BASEMAPS",
    "SUPPORTED_LAYER_KINDS",
    "ViewerProject",
    "ViewerState",
]
