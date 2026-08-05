"""Explicit adapters from ECAT/CSI data into detached display layers.

Adapters only read their inputs.  They do not call inversion routines, mutate
fault geometry, alter slip arrays, or convert a display product back into a
scientific object.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from .models import LayerStyle, RasterLayer, VectorLayer


def _coerce_style(style, **defaults):
    if style is None:
        return LayerStyle(**defaults)
    if isinstance(style, LayerStyle):
        return style
    if isinstance(style, Mapping):
        values = dict(defaults)
        values.update(dict(style))
        return LayerStyle(**values)
    raise TypeError("style must be a LayerStyle, mapping, or None.")


def _layer_name(name, fallback):
    text = str(name or fallback).strip()
    return text or str(fallback)


def _default_identifier(value):
    text = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value).strip()
    ).strip("_")
    return text or "layer"


def _feature(geometry_type, coordinates, **properties):
    return {
        "geometry": {
            "type": geometry_type,
            "coordinates": np.asarray(coordinates, dtype=float).tolist(),
        },
        "properties": properties,
    }


def raster_from_observation_grid(
    grid,
    *,
    variable=None,
    layer_id="observation",
    name=None,
    mask="source_valid",
    style=None,
    visible=True,
):
    """Adapt one variable from an ECAT :class:`ObservationGrid`.

    Parameters
    ----------
    grid : ObservationGrid
        Standard ECAT observation-grid object.
    variable : str, optional
        Stored or derived variable.  When omitted, the grid must resolve one
        unambiguous default component.
    layer_id, name : str
        Stable machine id and human-readable layer name.
    mask : {"source_valid", "analysis_valid", "finite", None}
        Display mask.  Scientific values are never rewritten.
    style : LayerStyle or mapping, optional
        Display-only style.
    """

    from eqtools.csiExtend.downsample.observation_grid import (
        resolve_observation_variable,
    )

    if variable is None:
        components = tuple(grid.components)
        if len(components) != 1:
            raise ValueError(
                "Observation grid has multiple components; select --variable "
                f"from {grid.export_variables()}."
            )
        component = components[0]
        if component in grid.corrected_components:
            corrected_name = (
                "corrected_observation"
                if component == "observation"
                else f"corrected_{component}"
            )
            raise ValueError(
                "Observation grid contains original and corrected values; "
                f"select {component!r} or {corrected_name!r}."
            )
        variable = component
    resolved = resolve_observation_variable(grid, variable)
    mask_name = None if mask is None else str(mask).strip().lower()
    if mask_name in {None, "finite"}:
        selected_mask = np.isfinite(resolved.values)
    elif mask_name == "source_valid":
        selected_mask = grid.source_valid_mask
    elif mask_name == "analysis_valid":
        selected_mask = grid.analysis_valid_mask
    else:
        raise ValueError(
            "mask must be source_valid, analysis_valid, finite, or None."
        )
    metadata = dict(getattr(grid, "attrs", {}) or {})
    if metadata.get("source_file"):
        metadata["source_file"] = Path(str(metadata["source_file"])).name
    metadata.update(
        {
            "source_kind": "observation_grid",
            "variable": resolved.name,
            "component": resolved.component,
            "role": resolved.role,
            "mask": mask_name or "finite",
        }
    )
    return RasterLayer(
        id=layer_id,
        name=_layer_name(name, resolved.name),
        values=resolved.values,
        longitude=grid.longitude,
        latitude=grid.latitude,
        mask=selected_mask,
        topology=grid.topology,
        units=resolved.units,
        convention=resolved.positive_convention,
        metadata=metadata,
        style=_coerce_style(style, cmap="RdBu_r"),
        visible=visible,
    )


def raster_from_observation_file(path, **kwargs):
    """Read an ECAT standard NetCDF/HDF5 grid and adapt one variable."""

    from eqtools.csiExtend.downsample.observation_grid import read_observation_grid

    return raster_from_observation_grid(read_observation_grid(path), **kwargs)


def raster_from_arrays(
    values,
    longitude,
    latitude,
    *,
    layer_id="raster",
    name="Raster",
    mask=None,
    topology="geographic_rectilinear",
    units=None,
    convention=None,
    metadata=None,
    style=None,
    visible=True,
):
    """Create a raster layer from explicit values and coordinates.

    This low-level adapter performs validation but no reprojection,
    interpolation, unit conversion, or sign conversion.
    """

    return RasterLayer(
        id=layer_id,
        name=name,
        values=values,
        longitude=longitude,
        latitude=latitude,
        mask=mask,
        topology=topology,
        units=units,
        convention=convention,
        metadata=dict(metadata or {}, source_kind="arrays"),
        style=_coerce_style(style),
        visible=visible,
    )


def cells_from_varres(
    result,
    *,
    component=None,
    layer_id="downsampled_cells",
    name=None,
    units="m",
    convention=None,
    style=None,
    visible=True,
):
    """Adapt a detached CSI varres result into colored cell polygons."""

    if component is None:
        component = "observation" if result.data_type == "sar" else "magnitude"
    values = np.asarray(result.component(component), dtype=float)
    errors = result.errors.get(component)
    features = []
    for index, (row_id, lon, lat, value, vertices) in enumerate(
        zip(
            result.row_ids,
            result.longitude,
            result.latitude,
            values,
            result.vertices,
        )
    ):
        ring = np.asarray(vertices, dtype=float)
        if ring.shape[1] > 2:
            ring = ring[:, :2]
        if not np.allclose(ring[0], ring[-1]):
            ring = np.vstack((ring, ring[0]))
        properties = {
            "id": int(row_id),
            "row": int(index),
            "center_longitude": float(lon),
            "center_latitude": float(lat),
            "component": str(component),
            "value": float(value),
        }
        if errors is not None:
            properties["error"] = float(np.asarray(errors)[index])
        if result.projection is not None:
            east, north, up = result.projection[index]
            properties.update(
                {
                    "projection_east": float(east),
                    "projection_north": float(north),
                    "projection_up": float(up),
                }
            )
        features.append(_feature("Polygon", [ring], **properties))
    return VectorLayer(
        id=layer_id,
        name=_layer_name(name, f"{component} cells"),
        features=tuple(features),
        units=units,
        convention=convention,
        value_property="value",
        metadata={
            "source_kind": "csi_varres",
            "component": str(component),
            "data_type": result.data_type,
            "geometry": result.geometry,
            **dict(result.metadata),
        },
        style=_coerce_style(style, cmap="RdBu_r"),
        visible=visible,
    )


def cells_from_varres_file(path, *, data_type="sar", geometry="auto", **kwargs):
    """Read paired CSI ``.txt/.rsp`` files and adapt their cells."""

    from eqtools.csiExtend.downsample.varres_io import read_csi_varres_result

    result = read_csi_varres_result(
        path,
        data_type=data_type,
        geometry=geometry,
    )
    return cells_from_varres(result, **kwargs)


def cells_from_arrays(
    vertices,
    values,
    *,
    layer_id="cells",
    name="Cells",
    ids=None,
    units=None,
    convention=None,
    metadata=None,
    style=None,
    visible=True,
):
    """Create colored polygon cells from explicit vertices and values."""

    values = np.asarray(values, dtype=float).reshape(-1)
    vertices = tuple(np.asarray(item, dtype=float) for item in vertices)
    if len(vertices) != values.size:
        raise ValueError("vertices and values must have the same length.")
    if ids is None:
        ids = np.arange(values.size)
    ids = np.asarray(ids).reshape(-1)
    if ids.size != values.size:
        raise ValueError("ids and values must have the same length.")
    features = []
    for item_id, value, item_vertices in zip(ids, values, vertices):
        if item_vertices.ndim != 2 or item_vertices.shape[1] not in {2, 3}:
            raise ValueError("Each cell must contain Nx2 or Nx3 vertices.")
        ring = item_vertices
        if not np.allclose(ring[0], ring[-1]):
            ring = np.vstack((ring, ring[0]))
        features.append(
            _feature("Polygon", [ring], id=str(item_id), value=float(value))
        )
    return VectorLayer(
        id=layer_id,
        name=_layer_name(name, "Cells"),
        features=tuple(features),
        units=units,
        convention=convention,
        value_property="value",
        metadata=dict(metadata or {}, source_kind="arrays"),
        style=_coerce_style(style, cmap="RdBu_r"),
        visible=visible,
    )


def trace_from_fault(
    fault,
    *,
    trace="original",
    layer_id=None,
    name=None,
    style=None,
    visible=True,
):
    """Adapt an original or discretized CSI fault trace without mutation."""

    trace = str(trace).strip().lower()
    if trace == "original":
        lon = getattr(fault, "lon", None)
        lat = getattr(fault, "lat", None)
    elif trace == "discretized":
        lon = getattr(fault, "loni", None)
        lat = getattr(fault, "lati", None)
    else:
        raise ValueError("trace must be 'original' or 'discretized'.")
    if lon is None or lat is None:
        raise ValueError(f"Fault has no {trace} lon/lat trace coordinates.")
    coordinates = np.column_stack(
        (
            np.asarray(lon, dtype=float).reshape(-1),
            np.asarray(lat, dtype=float).reshape(-1),
        )
    )
    if coordinates.shape[0] < 2 or not np.all(np.isfinite(coordinates)):
        raise ValueError("Fault trace requires at least two finite lon/lat points.")
    fault_name = str(getattr(fault, "name", "fault"))
    return VectorLayer(
        id=layer_id or _default_identifier(f"{fault_name}_{trace}_trace"),
        name=_layer_name(name, f"{fault_name} ({trace} trace)"),
        features=(
            _feature(
                "LineString",
                coordinates,
                fault=fault_name,
                trace=trace,
            ),
        ),
        value_property=None,
        metadata={"source_kind": "csi_fault", "trace": trace},
        style=_coerce_style(style),
        visible=visible,
    )


def _fault_patch_lonlat(fault):
    patches = getattr(fault, "patch", None)
    xy2ll = getattr(fault, "xy2ll", None)
    if patches is not None and callable(xy2ll):
        converted = []
        for patch in patches:
            patch = np.asarray(patch, dtype=float)
            if (
                patch.ndim != 2
                or patch.shape[0] not in {3, 4}
                or patch.shape[1] < 3
            ):
                raise ValueError(
                    "Fault patches must contain 3 or 4 finite x, y, depth "
                    "vertices."
                )
            if not np.all(np.isfinite(patch[:, :3])):
                raise ValueError("Fault patch coordinates must be finite.")
            lon, lat = xy2ll(patch[:, 0].copy(), patch[:, 1].copy())
            converted.append(
                np.column_stack(
                    (
                        np.asarray(lon, dtype=float),
                        np.asarray(lat, dtype=float),
                        patch[:, 2],
                    )
                )
            )
        return tuple(converted)
    patchll = getattr(fault, "patchll", None)
    if patchll is None:
        raise ValueError("Fault has neither patch+xy2ll nor patchll geometry.")
    converted = tuple(np.asarray(patch, dtype=float).copy() for patch in patchll)
    if any(
        patch.ndim != 2
        or patch.shape[0] not in {3, 4}
        or patch.shape[1] < 2
        or not np.all(np.isfinite(patch))
        for patch in converted
    ):
        raise ValueError(
            "Fault patchll entries must contain 3 or 4 finite lon/lat vertices."
        )
    return converted


def _fault_patch_values(fault, component, count):
    key = str(component).strip().lower().replace("-", "_")
    aliases = {
        "ss": "strikeslip",
        "strike_slip": "strikeslip",
        "ds": "dipslip",
        "dip_slip": "dipslip",
        "opening": "tensile",
        "magnitude": "total",
    }
    key = aliases.get(key, key)
    slip = np.asarray(getattr(fault, "slip", np.empty((count, 0))), dtype=float)
    if slip.ndim == 1:
        slip = slip[:, None]
    if slip.shape[0] != count:
        raise ValueError("Fault slip rows do not match patch count.")
    if key == "strikeslip" and slip.shape[1] >= 1:
        return key, slip[:, 0]
    if key == "dipslip" and slip.shape[1] >= 2:
        return key, slip[:, 1]
    if key == "tensile" and slip.shape[1] >= 3:
        return key, slip[:, 2]
    if key == "total" and slip.shape[1] >= 2:
        return key, np.hypot(slip[:, 0], slip[:, 1])
    if key == "rake" and slip.shape[1] >= 2:
        return key, np.degrees(np.arctan2(slip[:, 1], slip[:, 0]))
    if key == "coupling":
        coupling = np.asarray(getattr(fault, "coupling", []), dtype=float).reshape(-1)
        if coupling.size == count:
            return key, coupling
    raise ValueError(
        f"Fault does not provide component {component!r} for {count} patches."
    )


def patches_from_fault(
    fault,
    *,
    component="total",
    altitude_mode="surface",
    layer_id=None,
    name=None,
    units=None,
    convention=None,
    style=None,
    visible=True,
):
    """Adapt CSI fault patches and one slip/coupling component.

    ``surface`` places polygons on the globe. ``depth_3d`` uses KML absolute
    altitude ``-depth_km * 1000`` and is intended for geometric inspection,
    not numerical exchange.
    """

    altitude_mode = str(altitude_mode).strip().lower()
    if altitude_mode not in {"surface", "depth_3d"}:
        raise ValueError("altitude_mode must be 'surface' or 'depth_3d'.")
    patches = _fault_patch_lonlat(fault)
    component, values = _fault_patch_values(fault, component, len(patches))
    fault_name = str(getattr(fault, "name", "fault"))
    slip = np.asarray(getattr(fault, "slip", []), dtype=float)
    if slip.ndim == 1:
        slip = slip[:, None]
    coupling = np.asarray(getattr(fault, "coupling", []), dtype=float).reshape(-1)
    features = []
    for index, (patch, value) in enumerate(zip(patches, values)):
        if patch.ndim != 2 or patch.shape[1] < 2:
            raise ValueError("Fault patch lon/lat geometry must be Nx2 or Nx3.")
        if altitude_mode == "depth_3d":
            if patch.shape[1] < 3:
                raise ValueError("depth_3d requires patch depth in kilometers.")
            coordinates = np.column_stack(
                (patch[:, 0], patch[:, 1], -1000.0 * patch[:, 2])
            )
        else:
            coordinates = patch[:, :2]
        if not np.allclose(coordinates[0], coordinates[-1]):
            coordinates = np.vstack((coordinates, coordinates[0]))
        properties = {
            "id": index,
            "fault": fault_name,
            "component": component,
            "value": float(value),
            "altitude_mode": altitude_mode,
        }
        if slip.ndim == 2 and slip.shape[0] == len(patches):
            if slip.shape[1] >= 1:
                properties["strikeslip"] = float(slip[index, 0])
            if slip.shape[1] >= 2:
                properties["dipslip"] = float(slip[index, 1])
                properties["total_slip"] = float(
                    np.hypot(slip[index, 0], slip[index, 1])
                )
                properties["rake_deg"] = float(
                    np.degrees(np.arctan2(slip[index, 1], slip[index, 0]))
                )
            if slip.shape[1] >= 3:
                properties["tensile"] = float(slip[index, 2])
        if coupling.size == len(patches):
            properties["coupling"] = float(coupling[index])
        if patch.shape[1] >= 3:
            properties["min_depth_km"] = float(np.min(patch[:, 2]))
            properties["max_depth_km"] = float(np.max(patch[:, 2]))
        if units is not None:
            properties["value_unit"] = str(units)
        features.append(_feature("Polygon", [coordinates], **properties))
    return VectorLayer(
        id=layer_id or _default_identifier(f"{fault_name}_{component}"),
        name=_layer_name(name, f"{fault_name} {component}"),
        features=tuple(features),
        units="degrees" if component == "rake" else units,
        convention=convention,
        value_property="value",
        metadata={
            "source_kind": "csi_fault",
            "component": component,
            "altitude_mode": altitude_mode,
        },
        style=_coerce_style(style),
        visible=visible,
    )


def _records_from_catalog(catalog):
    if isinstance(catalog, (str, Path)):
        with Path(catalog).open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if hasattr(catalog, "to_dict"):
        try:
            return list(catalog.to_dict(orient="records"))
        except TypeError:
            pass
    if isinstance(catalog, Mapping):
        keys = list(catalog)
        columns = [np.asarray(catalog[key]).reshape(-1) for key in keys]
        if not columns:
            return []
        if len({column.size for column in columns}) != 1:
            raise ValueError("Catalog mapping columns must have equal length.")
        return [
            {key: column[index] for key, column in zip(keys, columns)}
            for index in range(columns[0].size)
        ]
    if isinstance(catalog, Iterable):
        records = list(catalog)
        if all(isinstance(record, Mapping) for record in records):
            return records
    raise TypeError("Catalog must be a CSV path, DataFrame, or record sequence.")


def _first_present(record, names, default=None):
    for name in names:
        if name in record:
            value = record[name]
            if value is not None and not (
                isinstance(value, str) and not value.strip()
            ):
                return value
    return default


def earthquakes_from_client_catalog(
    catalog,
    *,
    layer_id="earthquakes",
    name="Earthquakes",
    style=None,
    visible=True,
):
    """Adapt earthquake-client CSV/DataFrame records into point features."""

    records = _records_from_catalog(catalog)
    features = []
    for index, record in enumerate(records):
        try:
            lon = float(_first_present(record, ("longitude", "lon")))
            lat = float(_first_present(record, ("latitude", "lat")))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Earthquake record {index} lacks finite longitude/latitude."
            ) from exc
        if not np.isfinite(lon) or not np.isfinite(lat):
            raise ValueError(
                f"Earthquake record {index} lacks finite longitude/latitude."
            )
        properties = {
            str(key): value.item() if isinstance(value, np.generic) else value
            for key, value in record.items()
            if value is not None
            and not (isinstance(value, str) and not value.strip())
        }
        properties.setdefault("id", index)
        magnitude = _first_present(record, ("magnitude", "mag"))
        depth = _first_present(record, ("depth", "depth_km"))
        if magnitude is not None:
            properties["magnitude"] = float(magnitude)
        if depth is not None:
            properties["depth_km"] = float(depth)
        features.append(_feature("Point", [lon, lat], **properties))
    return VectorLayer(
        id=layer_id,
        name=_layer_name(name, "Earthquakes"),
        features=tuple(features),
        value_property="depth_km",
        units="km",
        metadata={
            "source_kind": "earthquake_client_catalog",
            "point_altitude": "surface",
        },
        style=_coerce_style(style),
        visible=visible,
    )


def earthquakes_from_seismiclocations(
    seismic,
    *,
    layer_id=None,
    name=None,
    style=None,
    visible=True,
):
    """Adapt a CSI ``seismiclocations`` object without importing CSI.

    The adapter reads ``lon`` and ``lat`` plus optional ``depth``, ``mag``,
    ``time``, and ``name`` arrays.  CSI ``CMTinfo`` is intentionally not
    interpreted because its angle/unit conventions are not uniform with the
    earthquake-client CSV contract.
    """

    dtype = str(getattr(seismic, "dtype", "")).strip().lower()
    if dtype and dtype != "seismiclocations":
        raise TypeError(
            "Expected a seismiclocations-like object; "
            f"got dtype={dtype!r}."
        )
    raw_lon = getattr(seismic, "lon", None)
    raw_lat = getattr(seismic, "lat", None)
    if raw_lon is None or raw_lat is None:
        raise ValueError("seismiclocations requires lon and lat arrays.")
    lon = np.asarray(raw_lon, dtype=float).reshape(-1)
    lat = np.asarray(raw_lat, dtype=float).reshape(-1)
    if lon.size == 0 or lon.size != lat.size:
        raise ValueError("seismiclocations lon/lat arrays must have equal length.")
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError("seismiclocations lon/lat values must be finite.")

    optional = {}
    aliases = {
        "depth_km": ("depth",),
        "magnitude": ("mag", "magnitude"),
        "time": ("time",),
        "event_name": ("event_name", "event_names"),
    }
    for output_name, input_names in aliases.items():
        values = None
        resolved_input_name = None
        for input_name in input_names:
            candidate = getattr(seismic, input_name, None)
            if candidate is not None:
                candidate_values = np.asarray(candidate).reshape(-1)
                if candidate_values.size:
                    values = candidate_values
                    resolved_input_name = input_name
                    break
        if values is not None:
            if values.size != lon.size:
                raise ValueError(
                    f"seismiclocations {resolved_input_name} length does not "
                    "match lon/lat."
                )
            optional[output_name] = values.copy()

    features = []
    for index, (event_lon, event_lat) in enumerate(zip(lon, lat)):
        properties = {"id": index}
        for field_name, values in optional.items():
            value = values[index]
            if isinstance(value, np.generic):
                value = value.item()
            if field_name in {"depth_km", "magnitude"}:
                value = float(value)
            elif hasattr(value, "isoformat"):
                value = value.isoformat()
            else:
                value = str(value)
            properties[field_name] = value
        features.append(
            _feature("Point", [event_lon, event_lat], **properties)
        )
    source_name = str(getattr(seismic, "name", "seismiclocations"))
    return VectorLayer(
        id=layer_id or _default_identifier(source_name),
        name=_layer_name(name, source_name),
        features=tuple(features),
        value_property="depth_km",
        units="km",
        metadata={
            "source_kind": "csi_seismiclocations",
            "depth_positive": "down",
            "point_altitude": "surface",
            "cmtinfo_interpreted": False,
        },
        style=_coerce_style(style),
        visible=visible,
    )


def _simple_geojson_geometries(geometry):
    """Expand GeoJSON Multi* and collection geometry into simple parts."""

    geometry = dict(geometry or {})
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type in {"Point", "LineString", "Polygon"}:
        return [geometry]
    part_type = {
        "MultiPoint": "Point",
        "MultiLineString": "LineString",
        "MultiPolygon": "Polygon",
    }.get(geometry_type)
    if part_type is not None:
        return [
            {"type": part_type, "coordinates": part}
            for part in (coordinates or ())
        ]
    if geometry_type == "GeometryCollection":
        parts = []
        for part in geometry.get("geometries") or ():
            parts.extend(_simple_geojson_geometries(part))
        return parts
    raise ValueError(f"Unsupported GeoJSON geometry {geometry_type!r}.")


def vector_from_geojson(
    path,
    *,
    layer_id=None,
    name=None,
    style=None,
    visible=True,
):
    """Read and deterministically expand GeoJSON vector features."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    kind = document.get("type")
    if kind == "FeatureCollection":
        raw_features = document.get("features", [])
    elif kind == "Feature":
        raw_features = [document]
    elif kind in {
        "Point",
        "LineString",
        "Polygon",
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
        "GeometryCollection",
    }:
        raw_features = [{"geometry": document, "properties": {}}]
    else:
        raise ValueError(
            "GeoJSON must be a FeatureCollection, Feature, or supported "
            "GeoJSON geometry."
        )
    features = []
    for index, item in enumerate(raw_features):
        geometry = item.get("geometry") or {}
        parts = _simple_geojson_geometries(geometry)
        properties = dict(item.get("properties") or {})
        for part_index, part in enumerate(parts):
            part_properties = dict(properties)
            if len(parts) > 1:
                part_properties["geometry_part"] = part_index
            features.append(
                {
                    "geometry": part,
                    "properties": part_properties,
                }
            )
    return VectorLayer(
        id=layer_id or _default_identifier(path.stem),
        name=_layer_name(name, path.stem),
        features=tuple(features),
        value_property=None,
        metadata={"source_kind": "geojson", "source_file": path.name},
        style=_coerce_style(style),
        visible=visible,
    )


def vector_from_gmt(
    path,
    *,
    layer_id=None,
    name=None,
    style=None,
    visible=True,
):
    """Read GMT multisegment lon/lat lines using the existing ECAT parser."""

    from eqtools.gmttools import read_gmt_lines

    path = Path(path)
    segments = read_gmt_lines(path, column_names=["longitude", "latitude"])
    features = []
    for index, segment in enumerate(segments):
        values = np.asarray(segment[["longitude", "latitude"]], dtype=float)
        features.append(_feature("LineString", values, segment=index))
    return VectorLayer(
        id=layer_id or _default_identifier(path.stem),
        name=_layer_name(name, path.stem),
        features=tuple(features),
        value_property=None,
        metadata={"source_kind": "gmt", "source_file": path.name},
        style=_coerce_style(style),
        visible=visible,
    )


__all__ = [
    "cells_from_arrays",
    "cells_from_varres",
    "cells_from_varres_file",
    "earthquakes_from_client_catalog",
    "earthquakes_from_seismiclocations",
    "patches_from_fault",
    "raster_from_arrays",
    "raster_from_observation_file",
    "raster_from_observation_grid",
    "trace_from_fault",
    "vector_from_geojson",
    "vector_from_gmt",
]
