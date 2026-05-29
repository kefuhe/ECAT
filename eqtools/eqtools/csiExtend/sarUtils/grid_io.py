import os

import numpy as np
import xarray as xr


DEFAULT_GRID_ENGINES = ("netcdf4", "h5netcdf", "scipy", "rasterio")


def open_grid_dataset(filename, engine=None, engine_candidates=None):
    """Open a NetCDF/GRD-like grid with clear engine fallback errors."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Grid file not found: {filename}")
    if engine:
        try:
            return xr.open_dataset(filename, engine=engine)
        except Exception as exc:
            raise ValueError(
                f"Could not open grid {filename!r} with xarray engine {engine!r}. "
                "Check sar_config.grid.engine or install the matching IO backend."
            ) from exc

    errors = []
    try:
        return xr.open_dataset(filename)
    except Exception as exc:
        errors.append(f"auto: {exc}")

    for candidate in engine_candidates or DEFAULT_GRID_ENGINES:
        try:
            return xr.open_dataset(filename, engine=candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    joined = "\n  - ".join(errors)
    raise ValueError(
        f"Could not open grid {filename!r} with xarray. Tried automatic engine "
        f"selection and {tuple(engine_candidates or DEFAULT_GRID_ENGINES)}.\n"
        f"  - {joined}"
    )


def resolve_grid_variable(dataset, variable=None, preferred=("z",)):
    if variable is not None:
        if variable not in dataset:
            raise KeyError(f"Variable {variable!r} not found in grid.")
        return variable
    for name in preferred:
        if name in dataset.data_vars:
            return name
    data_vars = list(dataset.data_vars)
    if len(data_vars) == 1:
        return data_vars[0]
    raise ValueError(
        "Grid variable is ambiguous. Set sar_config.grid.value_variable "
        "or the corresponding projection component variable explicitly."
    )


def resolve_lonlat_coords(dataset, lon_name=None, lat_name=None):
    lon_candidates = tuple(name for name in (lon_name, "lon", "longitude", "x") if name)
    lat_candidates = tuple(name for name in (lat_name, "lat", "latitude", "y") if name)

    lon_coord = next((name for name in lon_candidates if name in dataset.coords), None)
    lat_coord = next((name for name in lat_candidates if name in dataset.coords), None)
    if lon_coord is None or lat_coord is None:
        raise ValueError(
            "Could not find longitude/latitude coordinates. Expected lon/lat "
            "or x/y coordinates, or set sar_config.grid.lon_name/lat_name."
        )
    return lon_coord, lat_coord


def _coord_values_are_lonlat(lon, lat):
    lon_values = np.asarray(lon, dtype=float)
    lat_values = np.asarray(lat, dtype=float)
    if not np.all(np.isfinite(lon_values)) or not np.all(np.isfinite(lat_values)):
        return False
    return (
        np.nanmin(lon_values) >= -360.0
        and np.nanmax(lon_values) <= 360.0
        and np.nanmin(lat_values) >= -90.0
        and np.nanmax(lat_values) <= 90.0
    )


def validate_lonlat_coords(lon_coord, lat_coord, lon, lat, coord_is_lonlat=None, filename=None):
    generic_names = {lon_coord, lat_coord} <= {"x", "y"}
    if coord_is_lonlat is False:
        raise ValueError(
            f"Grid {filename or ''} uses coordinates {lon_coord!r}/{lat_coord!r}, "
            "but sar_config.grid.coord_is_lonlat is false. Direct-projection SAR "
            "input requires geographic longitude/latitude coordinates."
        )
    if generic_names or coord_is_lonlat is True:
        if coord_is_lonlat is True:
            return
        if not _coord_values_are_lonlat(lon, lat):
            raise ValueError(
                f"Grid {filename or ''} uses generic coordinates {lon_coord!r}/"
                f"{lat_coord!r} whose values do not look like longitude/latitude. "
                "Provide lon/lat coordinates or set a manual projection origin only "
                "after converting the grid to geographic lon/lat."
            )


def _transpose_data_to_coords(data_array, lon_array, lat_array):
    if data_array.ndim != 2:
        raise ValueError(
            f"Grid variable {data_array.name!r} must be 2-D; got dims {data_array.dims}."
        )

    if lon_array.ndim == 1 and lat_array.ndim == 1:
        lon_dim = lon_array.dims[0]
        lat_dim = lat_array.dims[0]
        target_dims = (lat_dim, lon_dim)
    elif lon_array.ndim == 2 and lat_array.ndim == 2:
        if lon_array.dims != lat_array.dims:
            raise ValueError("2-D lon/lat coordinate grids must have matching dimensions.")
        target_dims = lat_array.dims
    else:
        raise ValueError("Grid coordinates must both be 1-D vectors or both be 2-D meshes.")

    if set(data_array.dims) != set(target_dims):
        return data_array
    if data_array.dims != target_dims:
        return data_array.transpose(*target_dims)
    return data_array


def read_lonlat_grid(
    filename,
    variable=None,
    lon_name=None,
    lat_name=None,
    engine=None,
    coord_is_lonlat=None,
):
    with open_grid_dataset(filename, engine=engine) as dataset:
        variable_name = resolve_grid_variable(dataset, variable=variable)
        lon_coord, lat_coord = resolve_lonlat_coords(
            dataset,
            lon_name=lon_name,
            lat_name=lat_name,
        )
        data_array = _transpose_data_to_coords(
            dataset[variable_name],
            dataset[lon_coord],
            dataset[lat_coord],
        )
        values = np.asarray(data_array.values)
        lon = np.asarray(dataset[lon_coord].values)
        lat = np.asarray(dataset[lat_coord].values)

    validate_lonlat_coords(
        lon_coord,
        lat_coord,
        lon,
        lat,
        coord_is_lonlat=coord_is_lonlat,
        filename=filename,
    )

    if lon.ndim == 1 and lat.ndim == 1:
        mesh_lon, mesh_lat = np.meshgrid(lon, lat)
    elif lon.shape == values.shape and lat.shape == values.shape:
        mesh_lon, mesh_lat = lon, lat
    else:
        raise ValueError(
            "Grid coordinates must be 1-D lon/lat vectors or 2-D meshes "
            "matching the value grid."
        )
    if values.shape != mesh_lon.shape:
        raise ValueError(
            f"Grid values shape {values.shape} does not match coordinate "
            f"mesh shape {mesh_lon.shape}."
        )
    return values, lon, lat, mesh_lon, mesh_lat
