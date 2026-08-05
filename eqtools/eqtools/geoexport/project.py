"""Strict YAML orchestration for multi-layer Google Earth exports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import yaml

from .adapters import (
    cells_from_varres_file,
    earthquakes_from_client_catalog,
    raster_from_observation_file,
    vector_from_geojson,
    vector_from_gmt,
)
from .google_earth import write_kmz
from .models import LayerStyle


PROJECT_SCHEMA_VERSION = 2
_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
PROJECT_LAYER_KINDS = (
    "observation_grid",
    "csi_varres",
    "earthquake_catalog",
    "vector",
)
_TOP_LEVEL_KEYS = {"version", "output", "layers"}
_OUTPUT_KEYS = {"path", "document_name"}
_STYLE_KEYS = {
    "cmap",
    "vmin",
    "vmax",
    "symmetry",
    "alpha",
    "display_factor",
    "display_unit",
    "normalization",
    "cyclic_period",
    "line_color",
    "line_width",
    "point_scale",
}
_LAYER_KEYS = {
    "observation_grid": {
        "id",
        "name",
        "kind",
        "source",
        "visible",
        "variable",
        "mask",
        "style",
    },
    "csi_varres": {
        "id",
        "name",
        "kind",
        "source",
        "visible",
        "data_type",
        "geometry",
        "component",
        "units",
        "convention",
        "style",
    },
    "earthquake_catalog": {
        "id",
        "name",
        "kind",
        "source",
        "visible",
        "style",
    },
    "vector": {
        "id",
        "name",
        "kind",
        "source",
        "format",
        "visible",
        "style",
    },
}


def _require_mapping(value, path):
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _reject_unknown(mapping, allowed, path):
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValueError(f"{path} contains unknown field(s): {', '.join(unknown)}.")


def _required_text(mapping, field, path):
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{field} must be a non-empty string.")
    return value.strip()


def load_export_project(path):
    """Load and validate one canonical geoexport project file.

    Paths are resolved relative to the YAML file. Unknown keys are rejected;
    there are no aliases, presets, includes, or environment-variable
    substitutions.
    """

    project_file = Path(path).resolve()
    with project_file.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw = _require_mapping(raw, "project")
    _reject_unknown(raw, _TOP_LEVEL_KEYS, "project")
    if raw.get("version") != PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"project.version must be {PROJECT_SCHEMA_VERSION}; "
            f"got {raw.get('version')!r}."
        )

    output = _require_mapping(raw.get("output"), "project.output")
    _reject_unknown(output, _OUTPUT_KEYS, "project.output")
    output_path = Path(_required_text(output, "path", "project.output"))
    if not output_path.is_absolute():
        output_path = project_file.parent / output_path
    if output_path.suffix.lower() != ".kmz":
        raise ValueError("project.output.path must end in .kmz.")
    document_name = output.get("document_name")
    if document_name is not None and (
        not isinstance(document_name, str) or not document_name.strip()
    ):
        raise ValueError(
            "project.output.document_name must be a non-empty string."
        )

    layers_raw = raw.get("layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        raise ValueError("project.layers must be a non-empty list.")
    layers = []
    layer_ids = set()
    for index, item in enumerate(layers_raw):
        field_path = f"project.layers[{index}]"
        item = _require_mapping(item, field_path)
        kind = _required_text(item, "kind", field_path)
        if kind not in PROJECT_LAYER_KINDS:
            raise ValueError(
                f"{field_path}.kind must be one of {PROJECT_LAYER_KINDS}; "
                f"got {kind!r}."
            )
        _reject_unknown(item, _LAYER_KEYS[kind], field_path)
        layer_id = _required_text(item, "id", field_path)
        if not _ID_PATTERN.fullmatch(layer_id):
            raise ValueError(
                f"{field_path}.id must start with a letter and contain only "
                "letters, digits, '.', '_' or '-'."
            )
        if layer_id in layer_ids:
            raise ValueError(f"Duplicate layer id {layer_id!r}.")
        layer_ids.add(layer_id)
        source = Path(_required_text(item, "source", field_path))
        if not source.is_absolute():
            source = project_file.parent / source
        if kind != "csi_varres" and not source.is_file():
            raise FileNotFoundError(f"{field_path}.source not found: {source}.")
        if kind == "csi_varres":
            prefix = str(source)
            if prefix.lower().endswith((".txt", ".rsp")):
                prefix = prefix[:-4]
            if not Path(f"{prefix}.txt").is_file() or not Path(
                f"{prefix}.rsp"
            ).is_file():
                raise FileNotFoundError(
                    f"{field_path}.source requires paired .txt/.rsp files: "
                    f"{prefix}."
                )
            source = Path(prefix)
        style = item.get("style")
        if style is not None:
            style = _require_mapping(style, f"{field_path}.style")
            _reject_unknown(style, _STYLE_KEYS, f"{field_path}.style")
            style = LayerStyle(**style)
        visible = item.get("visible", True)
        if not isinstance(visible, bool):
            raise ValueError(f"{field_path}.visible must be a YAML boolean.")
        format_name = item.get("format")
        if kind == "vector":
            if format_name is None:
                suffix = source.suffix.lower()
                format_name = (
                    "geojson"
                    if suffix in {".json", ".geojson"}
                    else "gmt"
                    if suffix == ".gmt"
                    else None
                )
            format_name = str(format_name or "").strip().lower()
            if format_name not in {"geojson", "gmt"}:
                raise ValueError(
                    f"{field_path}.format must be geojson or gmt."
                )
        elif format_name is not None:
            raise ValueError(f"{field_path}.format is only valid for vector.")
        normalized = dict(item)
        normalized["kind"] = kind
        normalized["id"] = layer_id
        normalized["source"] = source
        normalized["style"] = style
        normalized["visible"] = visible
        normalized["format"] = format_name
        layers.append(normalized)
    return {
        "version": PROJECT_SCHEMA_VERSION,
        "project_file": project_file,
        "output": {
            "path": output_path.resolve(),
            "document_name": document_name,
        },
        "layers": layers,
    }


def _build_layer(spec):
    common = {
        "layer_id": spec["id"],
        "name": spec.get("name"),
        "style": spec.get("style"),
        "visible": spec["visible"],
    }
    kind = spec["kind"]
    if kind == "observation_grid":
        return raster_from_observation_file(
            spec["source"],
            variable=spec.get("variable"),
            mask=spec.get("mask", "source_valid"),
            **common,
        )
    if kind == "csi_varres":
        return cells_from_varres_file(
            spec["source"],
            data_type=spec.get("data_type", "sar"),
            geometry=spec.get("geometry", "auto"),
            component=spec.get("component"),
            units=spec.get("units", "m"),
            convention=spec.get("convention"),
            **common,
        )
    if kind == "earthquake_catalog":
        return earthquakes_from_client_catalog(spec["source"], **common)
    if kind == "vector":
        if spec["format"] == "geojson":
            return vector_from_geojson(spec["source"], **common)
        return vector_from_gmt(spec["source"], **common)
    raise AssertionError(f"Unhandled project layer kind: {kind}.")


def export_project(path, *, overwrite=False):
    """Build all YAML layers and write the configured KMZ."""

    project = load_export_project(path)
    layers = [_build_layer(spec) for spec in project["layers"]]
    return write_kmz(
        layers,
        project["output"]["path"],
        overwrite=overwrite,
        document_name=project["output"]["document_name"],
    )


__all__ = [
    "PROJECT_LAYER_KINDS",
    "PROJECT_SCHEMA_VERSION",
    "export_project",
    "load_export_project",
]
