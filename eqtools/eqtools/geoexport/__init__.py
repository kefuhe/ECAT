"""Display-only geographic exports for ECAT scientific data.

The public API is lazy so importing :mod:`eqtools.geoexport` does not require
CSI, Matplotlib, xarray, pandas, or PyYAML until a corresponding adapter or
writer is used.
"""

from importlib import import_module


_PUBLIC_OBJECTS = {
    "ExportResult": (".models", "ExportResult"),
    "LayerStyle": (".models", "LayerStyle"),
    "RasterLayer": (".models", "RasterLayer"),
    "VectorLayer": (".models", "VectorLayer"),
    "cells_from_arrays": (".adapters", "cells_from_arrays"),
    "cells_from_varres": (".adapters", "cells_from_varres"),
    "cells_from_varres_file": (".adapters", "cells_from_varres_file"),
    "earthquakes_from_client_catalog": (
        ".adapters",
        "earthquakes_from_client_catalog",
    ),
    "earthquakes_from_seismiclocations": (
        ".adapters",
        "earthquakes_from_seismiclocations",
    ),
    "patches_from_fault": (".adapters", "patches_from_fault"),
    "raster_from_arrays": (".adapters", "raster_from_arrays"),
    "raster_from_observation_file": (
        ".adapters",
        "raster_from_observation_file",
    ),
    "raster_from_observation_grid": (
        ".adapters",
        "raster_from_observation_grid",
    ),
    "trace_from_fault": (".adapters", "trace_from_fault"),
    "vector_from_geojson": (".adapters", "vector_from_geojson"),
    "vector_from_gmt": (".adapters", "vector_from_gmt"),
    "write_kmz": (".google_earth", "write_kmz"),
    "export_project": (".project", "export_project"),
    "load_export_project": (".project", "load_export_project"),
}


def __getattr__(name):
    try:
        module_name, object_name = _PUBLIC_OBJECTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
    value = getattr(import_module(module_name, __name__), object_name)
    globals()[name] = value
    return value


__all__ = sorted(_PUBLIC_OBJECTS)
