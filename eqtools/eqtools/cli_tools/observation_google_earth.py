"""Optional full-resolution Google Earth export for the downsampling CLI.

This module is deliberately an orchestration adapter.  It reads a detached
``ObservationGrid`` and delegates display-only conversion to
:mod:`eqtools.geoexport`.  It never changes reader values, masks, projection
vectors, covariance, or downsampling results.
"""

from pathlib import Path


def google_earth_export_file(config, out_name):
    """Return the configured KMZ path for one observation export."""

    output_file = (config or {}).get("file", "auto")
    if output_file in (None, "auto"):
        return Path(f"{out_name}_google_earth.kmz")
    return Path(str(output_file))


def _automatic_variables(grid):
    variables = []
    for component in grid.components:
        if component in grid.corrected_components:
            variables.append(
                "corrected_observation"
                if component == "observation"
                else f"corrected_{component}"
            )
        else:
            variables.append(component)
    return variables


def export_observation_google_earth(grid, out_name, config):
    """Export full-resolution observations without interpolation.

    Parameters
    ----------
    grid : ObservationGrid
        Reader-normalized full-resolution values after optional deterministic
        corrections. Automatic selection uses each final corrected component
        when one exists.
    out_name : str
        Observation output prefix.
    config : mapping
        Normalized ``export.google_earth`` section.

    Returns
    -------
    dict
        Serializable runtime report containing the output file, layer ids, and
        selected variables.

    Notes
    -----
    The Google Earth writer accepts only exact, regularly spaced geographic
    grids. Projected, rotated, curvilinear, irregular, and antimeridian-crossing
    grids fail clearly rather than being interpolated or reduced to a bounding
    box.
    """

    config = dict(config or {})
    if not config.get("enabled", False):
        return {
            "enabled": False,
            "files": [],
            "layer_ids": [],
            "variables": [],
        }

    from eqtools.geoexport import (
        LayerStyle,
        raster_from_observation_grid,
        write_kmz,
    )

    configured_variables = config.get("variables", "auto")
    variables = (
        _automatic_variables(grid)
        if configured_variables == "auto"
        else list(configured_variables)
    )
    style = LayerStyle(**dict(config.get("style", {}) or {}))
    visible = config.get("visible", True)
    layers = [
        raster_from_observation_grid(
            grid,
            variable=variable,
            layer_id=f"full_resolution_{variable}",
            name=f"Full resolution: {variable}",
            mask=config.get("mask", "source_valid"),
            style=style,
            visible=visible,
        )
        for variable in variables
    ]
    if not layers:
        raise ValueError(
            "Google Earth observation export requires at least one variable."
        )

    output_file = google_earth_export_file(config, out_name)
    result = write_kmz(
        layers,
        output_file,
        overwrite=config.get("overwrite", True),
        document_name=config.get("document_name"),
    )
    return {
        "enabled": True,
        "files": [str(path) for path in result.output_files],
        "layer_ids": list(result.layer_ids),
        "variables": variables,
    }


def format_google_earth_export_report(report):
    """Return a concise terminal summary for an observation KMZ export."""

    if not report or not report.get("enabled", False):
        return ""
    return "\n".join(
        (
            "Google Earth observation export:",
            f"  file: {', '.join(report.get('files', []))}",
            f"  variables: {', '.join(report.get('variables', []))}",
        )
    )


__all__ = [
    "export_observation_google_earth",
    "format_google_earth_export_report",
    "google_earth_export_file",
]
