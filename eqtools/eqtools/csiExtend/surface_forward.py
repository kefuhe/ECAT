"""Utilities for dense surface-displacement forward modeling.

The functions here are intentionally thin wrappers around the CSI/ECAT
``fault.compute_surface_displacement`` interface.  They organize repeated
multi-fault, LOS-projection, and file-output patterns without hiding the
scientific inputs prepared by the calling script.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import numpy as np


@dataclass
class SurfaceForwardResult:
    """Container returned by :func:`compute_multifault_surface_displacement`."""

    obs_pts: np.ndarray
    disp_total_enu: np.ndarray
    disp_by_fault_enu: "OrderedDict[str, np.ndarray]" = field(
        default_factory=OrderedDict
    )
    fault_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_multifault_surface_displacement(
    faults,
    *,
    box=None,
    disk=None,
    npoints=10,
    lonlat=None,
    profile=None,
    data=None,
    slip_vectors=None,
    nu=0.25,
    method="cutde",
    target_mem_gb=None,
    max_obs_batch=None,
    max_tri_batch=None,
    min_batch_count=5,
    output_coords="lonlat",
    verbose=False,
    return_each_fault=True,
    check_obs_pts=True,
    **kwargs,
) -> SurfaceForwardResult:
    """Compute and sum dense ENU displacement from one or more fault objects.

    Parameters
    ----------
    faults : fault object, sequence, or mapping
        Objects must expose ``compute_surface_displacement``.  Mapping keys are
        used as fault names.  For a sequence, each object's ``name`` attribute is
        used when present.
    box, disk, lonlat, profile, data
        Observation-point specification passed directly to
        ``fault.compute_surface_displacement``.
    slip_vectors : None, array, sequence, or mapping
        Optional slip vectors passed as ``slipVec``.  A mapping is keyed by
        fault name.  A sequence follows the normalized fault order.  A single
        array is allowed only for a single fault.
    return_each_fault : bool
        Store per-fault displacement fields in the result.  The total
        displacement is always returned.
    check_obs_pts : bool
        Check that each fault call returns the same observation coordinates.

    Returns
    -------
    SurfaceForwardResult
        Observation coordinates, total ENU displacement, optional per-fault ENU
        displacement, and lightweight metadata.
    """

    fault_items = _normalize_faults(faults)
    obs_reference = None
    disp_total = None
    disp_by_fault = OrderedDict()

    for index, (fault_name, fault) in enumerate(fault_items):
        if not hasattr(fault, "compute_surface_displacement"):
            raise TypeError(
                f"Fault {fault_name!r} does not provide "
                "compute_surface_displacement()."
            )

        slip_vec = _select_slip_vector(slip_vectors, fault_name, index, len(fault_items))
        obs_pts, disp_enu = fault.compute_surface_displacement(
            box=box,
            disk=disk,
            npoints=npoints,
            lonlat=lonlat,
            profile=profile,
            data=data,
            slipVec=slip_vec,
            nu=nu,
            method=method,
            target_mem_gb=target_mem_gb,
            max_obs_batch=max_obs_batch,
            max_tri_batch=max_tri_batch,
            min_batch_count=min_batch_count,
            verbose=verbose,
            output_file=None,
            output_coords=output_coords,
            **kwargs,
        )

        obs_pts = np.asarray(obs_pts, dtype=float)
        disp_enu = _as_enu_array(disp_enu, name=fault_name)

        if obs_reference is None:
            obs_reference = obs_pts
            disp_total = np.zeros_like(disp_enu, dtype=float)
        elif check_obs_pts and not np.allclose(obs_reference, obs_pts, equal_nan=True):
            raise ValueError(
                f"Fault {fault_name!r} returned observation coordinates that "
                "differ from the first fault."
            )

        if disp_total.shape != disp_enu.shape:
            raise ValueError(
                f"Fault {fault_name!r} returned displacement shape "
                f"{disp_enu.shape}, expected {disp_total.shape}."
            )

        disp_total += disp_enu
        if return_each_fault:
            disp_by_fault[fault_name] = disp_enu

    metadata = {
        "method": method,
        "nu": nu,
        "n_faults": len(fault_items),
        "output_coords": output_coords,
        "sampling": _sampling_summary(box, disk, lonlat, profile, data),
    }
    return SurfaceForwardResult(
        obs_pts=obs_reference,
        disp_total_enu=disp_total,
        disp_by_fault_enu=disp_by_fault,
        fault_names=tuple(name for name, _ in fault_items),
        metadata=metadata,
    )


def project_enu_to_los(disp_enu, projection):
    """Project ENU displacement to a scalar observation.

    The convention follows ECAT's SAR reader contract:

    ``scalar_observation = ENU_displacement dot projection``.

    ``projection`` can be an ``(N, 3)`` array, a single ``(3,)`` vector, or an
    object with a ``.los`` attribute.
    """

    disp = _as_enu_array(disp_enu, name="disp_enu")
    if hasattr(projection, "los"):
        projection = projection.los
    proj = np.asarray(projection, dtype=float)

    if proj.shape == (3,):
        proj = np.broadcast_to(proj, disp.shape)
    if proj.shape != disp.shape:
        raise ValueError(
            f"Projection shape {proj.shape} must be (3,) or match "
            f"displacement shape {disp.shape}."
        )
    return np.einsum("ij,ij->i", disp, proj)


def save_surface_forward_txt(
    output_file,
    result: SurfaceForwardResult,
    *,
    include_by_fault=False,
    fmt="%.8e",
    create_parent=True,
):
    """Save a surface-forward result as a whitespace-delimited text table."""

    path = Path(output_file)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    arrays = [np.asarray(result.obs_pts), np.asarray(result.disp_total_enu)]
    headers = [
        "coord_0",
        "coord_1",
        "coord_2",
        "total_east",
        "total_north",
        "total_up",
    ]
    if include_by_fault:
        for fault_name, disp in result.disp_by_fault_enu.items():
            arrays.append(np.asarray(disp))
            safe = _safe_name(fault_name)
            headers.extend([f"{safe}_east", f"{safe}_north", f"{safe}_up"])

    table = np.column_stack(arrays)
    np.savetxt(path, table, fmt=fmt, header=" ".join(headers), comments="")
    return path


def save_surface_forward_h5(
    output_file,
    result: SurfaceForwardResult,
    *,
    include_by_fault=True,
    create_parent=True,
):
    """Save a surface-forward result as HDF5."""

    import h5py

    path = Path(output_file)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    coord_names = _coordinate_dataset_names(result.metadata.get("output_coords"))
    with h5py.File(path, "w") as h5:
        coords = h5.create_group("coordinates")
        for idx, name in enumerate(coord_names):
            coords.create_dataset(name, data=result.obs_pts[:, idx])

        disp = h5.create_group("displacement")
        disp.create_dataset("east", data=result.disp_total_enu[:, 0])
        disp.create_dataset("north", data=result.disp_total_enu[:, 1])
        disp.create_dataset("up", data=result.disp_total_enu[:, 2])

        if include_by_fault and result.disp_by_fault_enu:
            by_fault = h5.create_group("displacement_by_fault")
            for fault_name, fault_disp in result.disp_by_fault_enu.items():
                group = by_fault.create_group(_safe_name(fault_name))
                group.attrs["display_name"] = str(fault_name)
                group.create_dataset("east", data=fault_disp[:, 0])
                group.create_dataset("north", data=fault_disp[:, 1])
                group.create_dataset("up", data=fault_disp[:, 2])

        h5.attrs["fault_names"] = ",".join(result.fault_names)
        for key, value in result.metadata.items():
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                h5.attrs[key] = value
            else:
                h5.attrs[key] = str(value)
    return path


def save_raster_like_geotiff(
    output_file,
    values,
    reference_raster,
    *,
    valid_index=None,
    dtype="float32",
    nodata=np.nan,
    compress="deflate",
    create_parent=True,
    warn_georeferencing=True,
):
    """Save one-dimensional values back to a GeoTIFF grid.

    This helper is useful for SAR workflows where forward-modeled values are
    computed only at valid image pixels.  ``reference_raster`` can be a path or
    an open rasterio dataset.  ``valid_index`` should be the flat pixel indices
    used by the SAR reader.  If ``valid_index`` is omitted, ``values`` must
    already cover the full raster.

    The output inherits the reference raster georeferencing.  If the reference
    raster has index-like or incomplete georeferencing, the saved GeoTIFF will
    also plot with index-like axes.
    """

    import rasterio

    path = Path(output_file)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(values)

    def _write(src):
        if warn_georeferencing:
            _warn_if_raster_georeferencing_is_limited(src, context="reference_raster")
        profile = src.profile.copy()
        profile.update(dtype=dtype, nodata=nodata, compress=compress)
        grid = _values_to_grid(
            values,
            shape=(src.height, src.width),
            valid_index=valid_index,
            dtype=dtype,
            nodata=nodata,
        )
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(grid, 1)

    if hasattr(reference_raster, "profile"):
        _write(reference_raster)
    else:
        with rasterio.open(reference_raster) as src:
            _write(src)
    return path


def save_lonlat_regular_geotiff(
    output_file,
    values,
    lon,
    lat,
    *,
    valid_index=None,
    dtype="float32",
    nodata=np.nan,
    compress="deflate",
    create_parent=True,
    crs="EPSG:4326",
    regular_atol=1e-6,
    regular_rtol=1e-6,
):
    """Save values to a GeoTIFF using a regular longitude/latitude grid.

    ``lon`` and ``lat`` may be 1-D coordinate vectors or 2-D mesh grids.  For
    2-D meshes the coordinates must be separable into one longitude vector and
    one latitude vector within the supplied tolerances.  Curvilinear grids
    should be saved to a format that can carry 2-D coordinates, such as
    NetCDF/xarray, rather than forced into an affine GeoTIFF.
    """

    import rasterio

    path = Path(output_file)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    lon_vec, lat_vec, shape = _regular_lonlat_vectors(
        lon,
        lat,
        atol=regular_atol,
        rtol=regular_rtol,
    )
    grid = _values_to_grid(
        values,
        shape=shape,
        valid_index=valid_index,
        dtype=dtype,
        nodata=nodata,
    )
    transform = _affine_from_center_vectors(lon_vec, lat_vec)
    profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": compress,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(grid, 1)
    return path


def _values_to_grid(values, *, shape, valid_index=None, dtype="float32", nodata=np.nan):
    values = np.asarray(values)
    height, width = shape
    total_size = height * width
    if valid_index is None:
        if values.size != total_size:
            raise ValueError(
                f"values has {values.size} entries, expected {total_size} "
                "when valid_index is not provided."
            )
        return values.astype(dtype).reshape(height, width)

    index = np.asarray(valid_index)
    if values.size != index.size:
        raise ValueError(
            f"values has {values.size} entries, but valid_index has "
            f"{index.size} entries."
        )
    if np.any(index < 0) or np.any(index >= total_size):
        raise ValueError("valid_index contains entries outside the raster grid.")
    grid = np.full(total_size, nodata, dtype=dtype)
    grid[index] = values.astype(dtype)
    return grid.reshape(height, width)


def _regular_lonlat_vectors(lon, lat, *, atol=1e-6, rtol=1e-6):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.ndim == 1 and lat.ndim == 1:
        if lon.size < 2 or lat.size < 2:
            raise ValueError("lon and lat coordinate vectors must each have at least two values.")
        return lon, lat, (lat.size, lon.size)

    if lon.ndim != 2 or lat.ndim != 2 or lon.shape != lat.shape:
        raise ValueError(
            "lon and lat must be either 1-D vectors or matching 2-D mesh grids."
        )

    lon_vec = np.nanmean(lon, axis=0)
    lat_vec = np.nanmean(lat, axis=1)
    lon_scale = max(float(np.nanmax(lon_vec) - np.nanmin(lon_vec)), 1.0)
    lat_scale = max(float(np.nanmax(lat_vec) - np.nanmin(lat_vec)), 1.0)
    lon_tol = max(float(atol), float(rtol) * lon_scale)
    lat_tol = max(float(atol), float(rtol) * lat_scale)
    lon_residual = np.nanmax(np.abs(lon - lon_vec[None, :]))
    lat_residual = np.nanmax(np.abs(lat - lat_vec[:, None]))
    if lon_residual > lon_tol or lat_residual > lat_tol:
        raise ValueError(
            "lon/lat meshes are not regular enough for an affine GeoTIFF "
            f"(lon residual={lon_residual:g}, tolerance={lon_tol:g}; "
            f"lat residual={lat_residual:g}, tolerance={lat_tol:g}). "
            "Use a NetCDF/xarray grid for curvilinear coordinates."
        )
    return lon_vec, lat_vec, lon.shape


def _affine_from_center_vectors(lon_vec, lat_vec):
    from affine import Affine

    lon_vec = np.asarray(lon_vec, dtype=float)
    lat_vec = np.asarray(lat_vec, dtype=float)
    dx = _constant_spacing(lon_vec, "lon")
    dy = _constant_spacing(lat_vec, "lat")
    return Affine(dx, 0.0, lon_vec[0] - dx / 2.0, 0.0, dy, lat_vec[0] - dy / 2.0)


def _constant_spacing(values, name):
    diffs = np.diff(np.asarray(values, dtype=float))
    if diffs.size == 0:
        raise ValueError(f"{name} coordinate vector must have at least two values.")
    step = float(np.nanmedian(diffs))
    if np.isclose(step, 0.0):
        raise ValueError(f"{name} coordinate spacing is zero.")
    tolerance = max(1e-10, 1e-6 * abs(step))
    if np.nanmax(np.abs(diffs - step)) > tolerance:
        raise ValueError(f"{name} coordinate vector is not evenly spaced.")
    return step


def _warn_if_raster_georeferencing_is_limited(src, *, context):
    message = _limited_raster_georeferencing_message(src, context=context)
    if message:
        warnings.warn(message, UserWarning, stacklevel=3)


def _limited_raster_georeferencing_message(src, *, context):
    reasons = []
    if getattr(src, "crs", None) is None:
        reasons.append("missing CRS")
    if _transform_is_identity_like(src.transform):
        reasons.append("identity-like transform")
    if _bounds_are_index_like(src):
        reasons.append("index-like bounds")

    if not reasons:
        return None
    bounds = src.bounds
    return (
        f"{context} has limited georeferencing ({', '.join(reasons)}; "
        f"bounds={bounds}). The saved GeoTIFF will inherit this metadata and "
        "may plot with pixel-index axes. Use save_lonlat_regular_geotiff() "
        "when a regular lon/lat grid is available."
    )


def _transform_is_identity_like(transform):
    coeffs = np.array(
        [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        dtype=float,
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    north_up_origin = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    return np.allclose(coeffs, identity) or np.allclose(coeffs, north_up_origin)


def _bounds_are_index_like(src):
    bounds = src.bounds
    width = float(src.width)
    height = float(src.height)
    x_span = abs(float(bounds.right) - float(bounds.left))
    y_span = abs(float(bounds.top) - float(bounds.bottom))
    starts_near_zero = np.isclose(bounds.left, 0.0) and (
        np.isclose(bounds.bottom, 0.0) or np.isclose(bounds.top, 0.0)
    )
    spans_match_shape = np.isclose(x_span, width) and np.isclose(y_span, height)
    return bool(starts_near_zero and spans_match_shape)


def _normalize_faults(faults):
    if isinstance(faults, Mapping):
        if not faults:
            raise ValueError("At least one fault object is required.")
        return [(str(name), fault) for name, fault in faults.items()]

    if isinstance(faults, Sequence) and not isinstance(faults, (str, bytes)):
        if not faults:
            raise ValueError("At least one fault object is required.")
        names = []
        items = []
        for index, fault in enumerate(faults):
            name = getattr(fault, "name", None) or f"fault_{index}"
            name = str(name)
            if name in names:
                name = f"{name}_{index}"
            names.append(name)
            items.append((name, fault))
        return items

    name = getattr(faults, "name", None) or "fault_0"
    return [(str(name), faults)]


def _select_slip_vector(slip_vectors, fault_name, index, n_faults):
    if slip_vectors is None:
        return None
    if isinstance(slip_vectors, Mapping):
        return slip_vectors.get(fault_name)
    if isinstance(slip_vectors, Sequence) and not isinstance(slip_vectors, np.ndarray):
        return slip_vectors[index]
    if n_faults == 1:
        return slip_vectors
    raise ValueError(
        "A single slip_vectors array is ambiguous for multiple faults. "
        "Use a mapping keyed by fault name or a sequence in fault order."
    )


def _as_enu_array(array, *, name):
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}.")
    return arr


def _sampling_summary(box, disk, lonlat, profile, data):
    if data is not None:
        return "data"
    if lonlat is not None:
        return "lonlat"
    if box is not None:
        return "box"
    if disk is not None:
        return "disk"
    if profile is not None:
        return "profile"
    return "default"


def _coordinate_dataset_names(output_coords):
    if output_coords == "lonlat":
        return ("longitude", "latitude", "depth")
    if output_coords == "xy":
        return ("x", "y", "z")
    return ("coord_0", "coord_1", "coord_2")


def _safe_name(name):
    return str(name).replace("/", "_").replace("\\", "_")
