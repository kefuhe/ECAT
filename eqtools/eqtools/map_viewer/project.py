"""Strict, small YAML project format for the ECAT research map viewer."""

from pathlib import Path
from typing import Mapping

from .models import LayerSpec, ViewerProject


VIEWER_PROJECT_SCHEMA_VERSION = 2
_ROOT_FIELDS = {"version", "project", "view", "layers"}
_PROJECT_FIELDS = {"name"}
_VIEW_FIELDS = {"region", "basemap"}
_LAYER_FIELDS = {
    "id",
    "name",
    "kind",
    "source",
    "variable",
    "mask",
    "visible",
    "style",
    "format",
    "data_type",
}


def _mapping(value, path):
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _reject_unknown(mapping, allowed, path):
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown fields at {path}: {unknown}.")


def _required_text(mapping, field, path):
    value = mapping.get(field)
    if value is None or not str(value).strip():
        raise ValueError(f"{path}.{field} is required.")
    return str(value).strip()


def _resolve_source(project_dir, source):
    text = str(source).strip()
    if "://" in text:
        raise ValueError(
            "Remote viewer sources are not supported; use a "
            "local project-relative path."
        )
    path = Path(text)
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def _layer_from_mapping(raw, project_dir, path):
    values = _mapping(raw, path)
    _reject_unknown(values, _LAYER_FIELDS, path)
    source = _resolve_source(
        project_dir,
        _required_text(values, "source", path),
    )
    visible = values.get("visible", False)
    if not isinstance(visible, bool):
        raise ValueError(f"{path}.visible must be a YAML boolean.")
    layer_id = _required_text(values, "id", path)
    if layer_id.startswith("background."):
        raise ValueError(
            f"{path}.id uses the reserved 'background.' namespace."
        )
    return LayerSpec(
        id=layer_id,
        name=str(values.get("name") or values["id"]),
        kind=_required_text(values, "kind", path),
        source=source,
        variable=values.get("variable"),
        mask=values.get("mask"),
        visible=visible,
        style=values.get("style") or {},
        format=values.get("format"),
        data_type=values.get("data_type"),
    )


def load_viewer_project(path):
    """Parse a canonical viewer project and resolve all local paths.

    User-owned project files intentionally require explicit ``layers``.
    Packaged fault, block and GNSS backgrounds are added by the CLI project
    factory, while arbitrary directory discovery remains unsupported so raw
    and corrected observation variables are never chosen silently.
    """

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading an ECAT map-viewer project requires PyYAML."
        ) from exc

    project_path = Path(path).resolve()
    if not project_path.is_file():
        raise FileNotFoundError(
            f"Viewer project file does not exist: {project_path}."
        )
    with project_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    root = _mapping(raw, "project file")
    _reject_unknown(root, _ROOT_FIELDS, "project file")
    if root.get("version") != VIEWER_PROJECT_SCHEMA_VERSION:
        raise ValueError(
            "Viewer project requires "
            f"version: {VIEWER_PROJECT_SCHEMA_VERSION}."
        )

    project_values = _mapping(root.get("project") or {}, "project")
    _reject_unknown(project_values, _PROJECT_FIELDS, "project")
    view = _mapping(root.get("view") or {}, "view")
    _reject_unknown(view, _VIEW_FIELDS, "view")
    raw_layers = root.get("layers") or []
    if not isinstance(raw_layers, list):
        raise ValueError("layers must be a list.")
    project_dir = project_path.parent
    layers = [
        _layer_from_mapping(item, project_dir, f"layers[{index}]")
        for index, item in enumerate(raw_layers)
    ]
    return ViewerProject(
        name=str(project_values.get("name") or project_path.stem),
        path=project_path,
        layers=tuple(layers),
        region=view.get("region"),
        basemap=str(view.get("basemap") or "open-street-map"),
    )


__all__ = ["VIEWER_PROJECT_SCHEMA_VERSION", "load_viewer_project"]
