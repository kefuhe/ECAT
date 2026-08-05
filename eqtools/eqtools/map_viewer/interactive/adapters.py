"""Small adapters from existing ECAT display products to editor inputs."""

import numpy as np

from ..models import LayerPayload
from .models import EditorBackground


def _style(values=None, **overrides):
    style = dict(values or {})
    style.update(
        {name: value for name, value in overrides.items() if value is not None}
    )
    return style


def background_from_payload(payload, *, style=None):
    """Build a detached editor background from a supported viewer payload."""

    if not isinstance(payload, LayerPayload):
        raise TypeError("background_from_payload expects a LayerPayload.")
    if payload.spec.kind not in {"observation_grid", "raster", "csi_varres"}:
        raise ValueError(
            "Trace-editor backgrounds must be observation_grid, raster, or "
            "csi_varres layers."
        )
    longitude = np.asarray(payload.data["longitude"], dtype=float)
    latitude = np.asarray(payload.data["latitude"], dtype=float)
    values = np.asarray(payload.data["values"], dtype=float)
    valid = np.asarray(
        payload.data.get("valid_mask", np.ones(values.shape, dtype=bool)),
        dtype=bool,
    )
    return EditorBackground(
        name=payload.spec.name,
        longitude=longitude,
        latitude=latitude,
        values=values,
        valid_mask=valid,
        units=payload.metadata.units,
        variable=payload.data.get("variable") or payload.spec.variable,
        source_identity=payload.metadata.fingerprint,
        style=_style(payload.spec.style, **(style or {})),
    )


def background_from_observation_grid(
    grid,
    component,
    *,
    name=None,
    style=None,
):
    """Build a read-only editor background from the active downsample grid.

    ``display_component`` intentionally selects the corrected component when
    phase-cycle or reference/ramp corrections have already been applied.
    """

    component = str(component).strip()
    if not component:
        raise ValueError("Trace-editor component must not be empty.")
    try:
        values = grid.display_component(component)
    except KeyError as exc:
        available = sorted(grid.components)
        raise ValueError(
            f"Unknown trace-editor component {component!r}; available: "
            f"{available}."
        ) from exc
    valid = (
        np.asarray(grid.analysis_valid_mask, dtype=bool)
        & np.isfinite(values)
        & np.isfinite(grid.longitude)
        & np.isfinite(grid.latitude)
    )
    corrected = component in getattr(grid, "corrected_components", {})
    variable = f"corrected_{component}" if corrected else component
    return EditorBackground(
        name=name or variable,
        longitude=grid.longitude,
        latitude=grid.latitude,
        values=values,
        valid_mask=valid,
        units=grid.component_units.get(component, "m"),
        variable=variable,
        source_identity=None,
        style=style or {},
    )


__all__ = [
    "background_from_observation_grid",
    "background_from_payload",
]
