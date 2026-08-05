"""Read-only semantic loaders used by the ECAT research map viewer.

This module interprets only documented ECAT products and a small set of
background formats. It does not run reader conversions, corrections,
downsampling, Green functions, or inversion.
"""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .models import LayerMetadata, LayerPayload, LayerSpec


LOADER_VERSION = 1


def _readonly(values, dtype=None):
    array = np.asarray(values, dtype=dtype)
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


def _bbox(longitude, latitude):
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    valid = np.isfinite(longitude) & np.isfinite(latitude)
    if not np.any(valid):
        return None
    return (
        float(np.min(longitude[valid])),
        float(np.max(longitude[valid])),
        float(np.min(latitude[valid])),
        float(np.max(latitude[valid])),
    )


def _source_files(spec):
    if spec.kind == "csi_varres":
        text = str(spec.source)
        lower = text.lower()
        if lower.endswith(".txt") or lower.endswith(".rsp"):
            text = text[:-4]
        return (Path(f"{text}.txt"), Path(f"{text}.rsp"))
    return (spec.source,)


def source_fingerprint(spec):
    """Return a cheap source-identity fingerprint without opening data."""

    identities = []
    for path in _source_files(spec):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Viewer source does not exist: {path}.")
        stat = path.stat()
        identities.append(
            (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
        )
    encoded = json.dumps(identities, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _metadata(spec, fingerprint, longitude, latitude, **kwargs):
    return LayerMetadata(
        layer_id=spec.id,
        bbox=_bbox(longitude, latitude),
        fingerprint=fingerprint,
        loader_version=LOADER_VERSION,
        **kwargs,
    )


def _load_earthquake_catalog(spec, fingerprint):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Earthquake-catalog map layers require pandas."
        ) from exc

    frame = pd.read_csv(spec.source)
    required = {"longitude", "latitude"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Earthquake catalog {spec.source} is missing columns {missing}."
        )
    data = {}
    for name in ("longitude", "latitude", "magnitude", "depth"):
        if name in frame:
            numeric = pd.to_numeric(frame[name], errors="coerce").to_numpy()
            data[name] = _readonly(numeric, dtype=float)
    if np.any(
        ~np.isfinite(data["longitude"]) | ~np.isfinite(data["latitude"])
    ):
        raise ValueError(
            "Earthquake catalog longitude/latitude must be finite."
        )
    if "time" in frame:
        parsed = pd.to_datetime(frame["time"], errors="coerce", utc=True)
        data["time"] = tuple(
            value.isoformat() if not pd.isna(value) else ""
            for value in parsed
        )
    for name in ("place", "id", "nodal_plane1", "nodal_plane2"):
        if name in frame:
            data[name] = tuple(
                "" if pd.isna(value) else str(value)
                for value in frame[name]
            )
    metadata = _metadata(
        spec,
        fingerprint,
        data["longitude"],
        data["latitude"],
        feature_count=len(frame),
        crs="EPSG:4326",
    )
    return LayerPayload(spec, metadata, data)


def _geometry_coordinate_arrays(geometry):
    if not geometry:
        return []
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Point":
        return [np.asarray([coordinates], dtype=float)]
    if geometry_type in {"LineString", "MultiPoint"}:
        return [np.asarray(coordinates, dtype=float)]
    if geometry_type in {"Polygon", "MultiLineString"}:
        return [np.asarray(part, dtype=float) for part in coordinates]
    if geometry_type == "MultiPolygon":
        return [
            np.asarray(ring, dtype=float)
            for polygon in coordinates
            for ring in polygon
        ]
    if geometry_type == "GeometryCollection":
        arrays = []
        for part in geometry.get("geometries") or ():
            arrays.extend(_geometry_coordinate_arrays(part))
        return arrays
    raise ValueError(f"Unsupported GeoJSON geometry type: {geometry_type!r}.")


def _load_geojson(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("type") != "FeatureCollection":
        raise ValueError("Viewer GeoJSON source must be a FeatureCollection.")
    return data


def _load_gmt(path):
    from ..gmttools import read_gmt_lines

    features = []
    for segment in read_gmt_lines(path):
        longitude = np.asarray(segment["X"], dtype=float)
        latitude = np.asarray(segment["Y"], dtype=float)
        coordinates = np.column_stack((longitude, latitude)).tolist()
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates,
                },
                "properties": {},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _vector_source(spec):
    suffix = spec.source.suffix.lower()
    format_name = str(spec.format or "").lower()
    suffix_format = None
    if suffix in {".json", ".geojson"}:
        suffix_format = "geojson"
    elif suffix == ".gmt":
        suffix_format = "gmt"
    if format_name and suffix_format and format_name != suffix_format:
        raise ValueError(
            f"Vector layer {spec.id!r} format {format_name!r} conflicts "
            f"with source extension {suffix!r}."
        )
    selected_format = format_name or suffix_format
    if selected_format == "geojson":
        return _load_geojson(spec.source)
    if selected_format == "gmt":
        return _load_gmt(spec.source)
    raise ValueError(
        f"Vector layer {spec.id!r} requires GeoJSON or GMT line data."
    )


def _load_vector(spec, fingerprint):
    geojson = _vector_source(spec)
    features = geojson.get("features") or []
    geometries = []
    properties = []
    all_longitude = []
    all_latitude = []
    for feature in features:
        geometry = feature.get("geometry")
        arrays = _geometry_coordinate_arrays(geometry)
        for array in arrays:
            if array.ndim != 2 or array.shape[1] < 2:
                raise ValueError("GeoJSON coordinates must contain lon/lat.")
            if not np.all(np.isfinite(array[:, :2])):
                raise ValueError("GeoJSON longitude/latitude must be finite.")
            all_longitude.append(array[:, 0])
            all_latitude.append(array[:, 1])
        geometries.append(geometry)
        properties.append(dict(feature.get("properties") or {}))
    longitude = (
        np.concatenate(all_longitude) if all_longitude else np.empty(0)
    )
    latitude = np.concatenate(all_latitude) if all_latitude else np.empty(0)
    metadata = _metadata(
        spec,
        fingerprint,
        longitude,
        latitude,
        feature_count=len(features),
        crs="EPSG:4326",
    )
    return LayerPayload(
        spec,
        metadata,
        {
            "geometries": tuple(geometries),
            "properties": tuple(properties),
        },
    )


def _load_gnss_velocity(spec, fingerprint):
    geojson = _vector_source(spec)
    longitude = []
    latitude = []
    east = []
    north = []
    station = []
    properties = []
    for index, feature in enumerate(geojson.get("features") or []):
        geometry = feature.get("geometry") or {}
        values = dict(feature.get("properties") or {})
        if geometry.get("type") != "Point":
            continue
        velocity = values.get("velocity")
        if not isinstance(velocity, (list, tuple)) or len(velocity) < 2:
            raise ValueError(
                f"GNSS feature {index} requires velocity [east, north, ...]."
            )
        lon, lat = geometry.get("coordinates", [None, None])[:2]
        longitude.append(float(lon))
        latitude.append(float(lat))
        east.append(float(velocity[0]))
        north.append(float(velocity[1]))
        station.append(
            str(
                values.get("station")
                or values.get("name")
                or values.get("site")
                or index
            )
        )
        properties.append(values)
    source_longitude = _readonly(longitude, dtype=float)
    longitude = _readonly(
        ((np.asarray(source_longitude) + 180.0) % 360.0) - 180.0,
        dtype=float,
    )
    latitude = _readonly(latitude, dtype=float)
    east = _readonly(east, dtype=float)
    north = _readonly(north, dtype=float)
    if np.any(
        ~np.isfinite(longitude)
        | ~np.isfinite(latitude)
        | ~np.isfinite(east)
        | ~np.isfinite(north)
    ):
        raise ValueError("GNSS coordinates and horizontal velocity must be finite.")
    metadata = _metadata(
        spec,
        fingerprint,
        longitude,
        latitude,
        feature_count=len(longitude),
        units=geojson.get("units"),
        crs="EPSG:4326",
    )
    return LayerPayload(
        spec,
        metadata,
        {
            "longitude": longitude,
            "source_longitude": source_longitude,
            "longitude_wrapped": bool(
                np.any(
                    ~np.isclose(
                        np.asarray(longitude),
                        np.asarray(source_longitude),
                        rtol=0.0,
                        atol=1.0e-10,
                    )
                )
            ),
            "latitude": latitude,
            "east": east,
            "north": north,
            "station": tuple(station),
            "properties": tuple(properties),
            "reference_frame": geojson.get("reference_frame"),
        },
    )


def _load_observation_grid(spec, fingerprint):
    from ..csiExtend.downsample.observation_grid import (
        read_observation_grid,
        resolve_observation_variable,
    )

    grid = read_observation_grid(spec.source)
    variable = resolve_observation_variable(grid, spec.variable)
    values = np.asarray(variable.values)
    if spec.mask == "source_valid":
        selected_mask = np.asarray(grid.source_valid_mask, dtype=bool)
    elif spec.mask == "analysis_valid":
        selected_mask = np.asarray(grid.analysis_valid_mask, dtype=bool)
    elif spec.mask == "finite":
        selected_mask = np.ones(grid.shape, dtype=bool)
    else:
        raise AssertionError(f"Unhandled observation mask: {spec.mask!r}.")
    valid = (
        selected_mask
        & np.isfinite(values)
        & np.isfinite(grid.longitude)
        & np.isfinite(grid.latitude)
    )
    data = {
        "longitude": _readonly(grid.longitude, dtype=float),
        "latitude": _readonly(grid.latitude, dtype=float),
        "values": _readonly(values),
        "valid_mask": _readonly(valid, dtype=bool),
        "variable": variable.name,
        "role": variable.role,
        "mask": spec.mask,
        "positive_convention": variable.positive_convention,
    }
    metadata = _metadata(
        spec,
        fingerprint,
        grid.longitude[valid],
        grid.latitude[valid],
        shape=tuple(grid.shape),
        available_variables=tuple(grid.export_variables()),
        units=variable.units,
        crs=grid.crs_wkt or "EPSG:4326",
        grid_topology=grid.topology,
        derived_display=False,
    )
    return LayerPayload(spec, metadata, data)


def _load_csi_varres(spec, fingerprint):
    from ..csiExtend.downsample.varres_io import read_csi_varres_result

    data_type = str(spec.data_type or "sar").strip().lower()
    result = read_csi_varres_result(spec.source, data_type=data_type)
    variable = spec.variable
    if not variable:
        variable = (
            "observation"
            if result.data_type == "sar"
            else result.available_components[0]
        )
    values = result.component(variable)
    metadata = _metadata(
        spec,
        fingerprint,
        result.longitude,
        result.latitude,
        feature_count=result.cell_count,
        available_variables=tuple(result.available_components),
        crs="EPSG:4326",
    )
    return LayerPayload(
        spec,
        metadata,
        {
            "longitude": _readonly(result.longitude, dtype=float),
            "latitude": _readonly(result.latitude, dtype=float),
            "values": _readonly(values),
            "vertices": tuple(
                _readonly(vertices, dtype=float)
                for vertices in result.vertices
            ),
            "variable": variable,
            "data_type": result.data_type,
            "projection": (
                _readonly(result.projection, dtype=float)
                if result.projection is not None
                else None
            ),
        },
    )


def _load_raster(spec, fingerprint):
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import xy
        from rasterio.warp import transform
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "GeoTIFF viewer layers require rasterio."
        ) from exc

    max_axis = 400
    with rasterio.open(spec.source) as dataset:
        if dataset.crs is None or dataset.transform is None:
            raise ValueError("Raster layers require a CRS and affine transform.")
        scale = max(1, int(np.ceil(max(dataset.height, dataset.width) / max_axis)))
        out_height = max(1, int(np.ceil(dataset.height / scale)))
        out_width = max(1, int(np.ceil(dataset.width / scale)))
        values = dataset.read(
            1,
            out_shape=(out_height, out_width),
            masked=True,
            resampling=Resampling.nearest,
        )
        transform_out = dataset.transform * dataset.transform.scale(
            dataset.width / out_width,
            dataset.height / out_height,
        )
        rows, columns = np.indices((out_height, out_width))
        x, y = xy(
            transform_out,
            rows.reshape(-1),
            columns.reshape(-1),
            offset="center",
        )
        if dataset.crs.to_epsg() == 4326:
            longitude = np.asarray(x).reshape(values.shape)
            latitude = np.asarray(y).reshape(values.shape)
        else:
            longitude, latitude = transform(
                dataset.crs,
                "EPSG:4326",
                x,
                y,
            )
            longitude = np.asarray(longitude).reshape(values.shape)
            latitude = np.asarray(latitude).reshape(values.shape)
        nodata_mask = np.ma.getmaskarray(values)
        values = np.asarray(values.filled(np.nan))
        valid = (
            ~nodata_mask
            & np.isfinite(values)
            & np.isfinite(longitude)
            & np.isfinite(latitude)
        )
        crs = dataset.crs.to_string()
        original_shape = (dataset.height, dataset.width)
        units = dataset.tags(1).get("units")
    metadata = _metadata(
        spec,
        fingerprint,
        longitude[valid],
        latitude[valid],
        shape=original_shape,
        available_variables=(spec.variable or "band_1",),
        units=units,
        crs=crs,
        grid_topology="affine",
        derived_display=scale > 1,
    )
    return LayerPayload(
        spec,
        metadata,
        {
            "longitude": _readonly(longitude, dtype=float),
            "latitude": _readonly(latitude, dtype=float),
            "values": _readonly(values),
            "valid_mask": _readonly(valid, dtype=bool),
            "variable": spec.variable or "band_1",
            "overview_stride": scale,
        },
    )


_LOADERS = {
    "earthquake_catalog": _load_earthquake_catalog,
    "vector": _load_vector,
    "gnss_velocity": _load_gnss_velocity,
    "observation_grid": _load_observation_grid,
    "raster": _load_raster,
    "csi_varres": _load_csi_varres,
}


def load_layer(spec):
    """Load one documented layer kind without modifying its source."""

    if not isinstance(spec, LayerSpec):
        raise TypeError("load_layer expects a LayerSpec.")
    fingerprint = source_fingerprint(spec)
    return _LOADERS[spec.kind](spec, fingerprint)


__all__ = [
    "LOADER_VERSION",
    "load_layer",
    "source_fingerprint",
]
