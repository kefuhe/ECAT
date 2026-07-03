"""Lightweight scientific raster plotting helpers.

This module intentionally handles only prepared 2-D raster-like data.  Product
semantics such as SAR sign conventions, LOS projection, and unit conversion
belong in readers or workflow scripts before calling these helpers.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from ._core import PlotStyle
from ._style_utils import finish_fig


_X_CANDIDATES = ("lon", "longitude", "x", "easting")
_Y_CANDIDATES = ("lat", "latitude", "y", "northing")


def plot_raster(
    data,
    *,
    x=None,
    y=None,
    extent=None,
    ax=None,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    percentile=99.0,
    symmetric=False,
    center=0.0,
    nodata=None,
    colorbar=True,
    colorbar_label=None,
    colorbar_orientation="vertical",
    colorbar_kwargs=None,
    title=None,
    xlabel=None,
    ylabel=None,
    axis="equal",
    geo_formatter=True,
    geo_decimal_places=None,
    axis_max_major_ticks=None,
    axis_tick_direction="out",
    origin="upper",
    interpolation="nearest",
    style="science",
    figsize="single",
    fontsize=8,
    tickfontsize=None,
    labelfontsize=None,
    colorbar_max_major_ticks=None,
    colorbar_tick_direction="out",
    dpi=300,
    rcparams=None,
    save=None,
    show=False,
    screen_dpi=200,
    close=False,
    **artist_kwargs,
):
    """Plot a prepared 2-D raster and return ``(fig, ax, artist)``.

    Parameters
    ----------
    data : array-like or xarray.DataArray
        Two-dimensional raster values.  If an xarray DataArray is supplied,
        coordinate vectors named lon/lat, longitude/latitude, or x/y are used
        when available.
    x, y : array-like, optional
        Optional 1-D or 2-D coordinates.  When supplied, drawing uses
        ``pcolormesh`` so coordinate spacing is honored.
    extent : sequence, optional
        ``[xmin, xmax, ymin, ymax]`` used with ``imshow`` when ``x``/``y`` are
        not supplied.
    symmetric : bool, optional
        Use a symmetric color range around ``center``.
    percentile : float or None, optional
        Robust percentile used when ``vmin``/``vmax`` are not fully specified.
        Set to ``None`` to use the full finite range.
    save : str or path-like, optional
        Output figure path.  Passing a path saves the figure through
        :func:`eqtools.viztools.finish_fig`.
    axis : {"equal", "auto", "off", "geo"} or None, optional
        Axis display mode.  ``"geo"`` keeps geographic axes visible, applies
        longitude/latitude labels and formatters, and uses an equal aspect.
        ``"off"`` hides axes.  Other values are forwarded to
        ``Axes.set_aspect``.
    axis_max_major_ticks, colorbar_max_major_ticks : int, optional
        Maximum number of major ticks to draw on the map axes or colorbar.
    tickfontsize, labelfontsize : float, optional
        Basic tick-label and axis/colorbar-label font sizes.

    Notes
    -----
    This helper does not interpret SAR, fault, or LOS conventions.  Convert and
    label values before plotting.
    """

    values, x, y, inferred_xlabel, inferred_ylabel = _coerce_raster_input(data, x=x, y=y)
    if (x is None) != (y is None):
        raise ValueError("x and y coordinates must be supplied together.")
    values = _masked_values(values, nodata=nodata)
    vmin, vmax = raster_limits(
        values,
        vmin=vmin,
        vmax=vmax,
        percentile=percentile,
        symmetric=symmetric,
        center=center,
    )

    if isinstance(save, bool):
        raise ValueError("save must be a file path or None, not a boolean.")

    needs_style = ax is None and style is not None
    context = (
        PlotStyle(style, figsize=figsize, fontsize=fontsize, dpi=dpi, rcparams=rcparams)
        if needs_style
        else nullcontext()
    )

    with context:
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        artist = _draw_raster(
            ax,
            values,
            x=x,
            y=y,
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin=origin,
            interpolation=interpolation,
            **artist_kwargs,
        )

        if colorbar:
            cbar_kwargs = dict(colorbar_kwargs or {})
            cbar_kwargs.setdefault("orientation", colorbar_orientation)
            cbar = fig.colorbar(artist, ax=ax, **cbar_kwargs)
            _style_colorbar(
                cbar,
                orientation=cbar_kwargs["orientation"],
                max_major_ticks=colorbar_max_major_ticks,
                tickfontsize=tickfontsize,
                labelfontsize=labelfontsize,
                tick_direction=colorbar_tick_direction,
            )
            if colorbar_label:
                label_kwargs = {}
                if labelfontsize is not None:
                    label_kwargs["fontsize"] = labelfontsize
                cbar.set_label(colorbar_label, **label_kwargs)

        if title:
            ax.set_title(title)
        _style_raster_axis(
            ax,
            axis=axis,
            xlabel=xlabel,
            ylabel=ylabel,
            inferred_xlabel=inferred_xlabel,
            inferred_ylabel=inferred_ylabel,
            geo_formatter=geo_formatter,
            geo_decimal_places=geo_decimal_places,
            max_major_ticks=axis_max_major_ticks,
            tickfontsize=tickfontsize,
            labelfontsize=labelfontsize,
            tick_direction=axis_tick_direction,
        )

        finish_fig(
            fig,
            save,
            save=save is not None,
            show=show,
            dpi=dpi,
            screen_dpi=screen_dpi,
            close=close,
        )

    return fig, ax, artist


def plot_dataarray(data_array, **kwargs):
    """Plot an xarray DataArray with coordinate-aware defaults."""

    return plot_raster(data_array, **kwargs)


def plot_geotiff(
    path,
    *,
    band=1,
    masked=True,
    ax=None,
    title=None,
    **kwargs,
):
    """Plot a GeoTIFF band using rasterio bounds and nodata masking."""

    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("plot_geotiff requires rasterio.") from exc

    with rasterio.open(path) as src:
        values = src.read(band, masked=masked)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        nodata = src.nodata if not masked else None
        _warn_if_geo_axis_has_limited_georeferencing(src, axis=kwargs.get("axis"))

    if title is None:
        title = Path(path).stem
    kwargs.setdefault("nodata", nodata)
    return plot_raster(values, extent=extent, ax=ax, title=title, **kwargs)


def plot_netcdf_grid(
    path,
    *,
    variable=None,
    x_name=None,
    y_name=None,
    engine=None,
    ax=None,
    title=None,
    **kwargs,
):
    """Plot a 2-D variable from a NetCDF/GRD-like file with xarray.

    If the file contains multiple data variables, ``variable`` must be set.
    """

    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("plot_netcdf_grid requires xarray.") from exc

    open_kwargs: dict[str, Any] = {}
    if engine is not None:
        open_kwargs["engine"] = engine
    with xr.open_dataset(path, **open_kwargs) as dataset:
        var_name = _resolve_data_variable(dataset, variable)
        data_array = dataset[var_name].load()
        if x_name is not None or y_name is not None:
            x, y = _resolve_named_coords(data_array, x_name=x_name, y_name=y_name)
            kwargs.setdefault("x", x)
            kwargs.setdefault("y", y)

    if title is None:
        title = var_name
    return plot_raster(data_array, ax=ax, title=title, **kwargs)


def raster_limits(
    data,
    *,
    vmin=None,
    vmax=None,
    percentile=99.0,
    symmetric=False,
    center=0.0,
):
    """Return robust ``(vmin, vmax)`` for finite raster values."""

    finite = np.asarray(data).ravel()
    if np.ma.isMaskedArray(data):
        finite = np.asarray(data.compressed())
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Raster contains no finite values to plot.")

    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)

    if symmetric:
        if vmin is not None or vmax is not None:
            half_range = max(
                abs(float(vmin) - center) if vmin is not None else 0.0,
                abs(float(vmax) - center) if vmax is not None else 0.0,
            )
        elif percentile is None:
            half_range = np.nanmax(np.abs(finite - center))
        else:
            half_range = np.nanpercentile(np.abs(finite - center), float(percentile))
        return float(center - half_range), float(center + half_range)

    if percentile is None:
        auto_min = float(np.nanmin(finite))
        auto_max = float(np.nanmax(finite))
    else:
        tail = (100.0 - float(percentile)) / 2.0
        auto_min, auto_max = np.nanpercentile(finite, [tail, 100.0 - tail])

    if vmin is None:
        vmin = auto_min
    if vmax is None:
        vmax = auto_max
    return float(vmin), float(vmax)


def _coerce_raster_input(data, *, x=None, y=None):
    inferred_xlabel = ""
    inferred_ylabel = ""
    if _looks_like_dataarray(data):
        values = np.asarray(data.values)
        if x is None or y is None:
            inferred_x, inferred_y, x_name, y_name = _infer_dataarray_coords(data)
            if x is None:
                x = inferred_x
                inferred_xlabel = x_name
            if y is None:
                y = inferred_y
                inferred_ylabel = y_name
    else:
        values = np.asarray(data)

    if values.ndim != 2:
        raise ValueError(f"Raster data must be 2-D, got shape {values.shape}.")
    return values, x, y, inferred_xlabel, inferred_ylabel


def _looks_like_dataarray(obj):
    return hasattr(obj, "dims") and hasattr(obj, "coords") and hasattr(obj, "values")


def _infer_dataarray_coords(data_array):
    x_name = _first_existing_name(data_array.coords, _X_CANDIDATES)
    y_name = _first_existing_name(data_array.coords, _Y_CANDIDATES)

    if x_name is None and len(data_array.dims) >= 2:
        x_dim = data_array.dims[-1]
        if x_dim in data_array.coords:
            x_name = x_dim
    if y_name is None and len(data_array.dims) >= 2:
        y_dim = data_array.dims[-2]
        if y_dim in data_array.coords:
            y_name = y_dim

    x = np.asarray(data_array.coords[x_name].values) if x_name else None
    y = np.asarray(data_array.coords[y_name].values) if y_name else None
    return x, y, x_name or "", y_name or ""


def _resolve_named_coords(data_array, *, x_name=None, y_name=None):
    x = np.asarray(data_array.coords[x_name].values) if x_name is not None else None
    y = np.asarray(data_array.coords[y_name].values) if y_name is not None else None
    return x, y


def _first_existing_name(mapping, names):
    for name in names:
        if name in mapping:
            return name
    return None


def _resolve_data_variable(dataset, variable):
    if variable is not None:
        if variable not in dataset.data_vars:
            raise KeyError(f"Variable {variable!r} not found in {list(dataset.data_vars)}.")
        return variable
    data_vars = list(dataset.data_vars)
    if len(data_vars) == 1:
        return data_vars[0]
    raise ValueError(
        "NetCDF/GRD file contains multiple data variables. Set variable=..."
    )


def _masked_values(values, *, nodata=None):
    arr = np.ma.array(values, copy=False)
    arr = np.ma.masked_invalid(arr)
    if nodata is not None and np.isfinite(nodata):
        arr = np.ma.masked_where(np.asarray(arr) == nodata, arr)
    return arr


def _draw_raster(
    ax,
    values,
    *,
    x=None,
    y=None,
    extent=None,
    cmap=None,
    vmin=None,
    vmax=None,
    origin="upper",
    interpolation="nearest",
    **kwargs,
):
    if x is not None and y is not None:
        x_edges, y_edges = _coordinates_for_pcolormesh(x, y)
        return ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
    return ax.imshow(
        values,
        extent=extent,
        origin=origin,
        interpolation=interpolation,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )


def _style_raster_axis(
    ax,
    *,
    axis,
    xlabel,
    ylabel,
    inferred_xlabel,
    inferred_ylabel,
    geo_formatter,
    geo_decimal_places,
    max_major_ticks,
    tickfontsize,
    labelfontsize,
    tick_direction,
):
    if axis == "off":
        ax.set_axis_off()
        return

    is_geo = axis == "geo"
    if is_geo:
        xlabel = "Longitude" if xlabel is None else xlabel
        ylabel = "Latitude" if ylabel is None else ylabel
        if max_major_ticks is None:
            max_major_ticks = 5
    else:
        xlabel = xlabel if xlabel is not None else inferred_xlabel
        ylabel = ylabel if ylabel is not None else inferred_ylabel

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labelfontsize is not None:
        ax.xaxis.label.set_size(labelfontsize)
        ax.yaxis.label.set_size(labelfontsize)

    if max_major_ticks is not None:
        _set_axis_max_major_ticks(ax.xaxis, max_major_ticks)
        _set_axis_max_major_ticks(ax.yaxis, max_major_ticks)

    if is_geo and geo_formatter:
        _apply_geo_formatters(ax, decimal_places=geo_decimal_places)

    ax.tick_params(axis="both", which="major", direction=tick_direction)
    if tickfontsize is not None:
        ax.tick_params(axis="both", labelsize=tickfontsize)

    if is_geo:
        ax.set_aspect("equal", adjustable="box")
    elif axis is not None and axis != "auto":
        ax.set_aspect(axis, adjustable="box")


def _style_colorbar(
    cbar,
    *,
    orientation,
    max_major_ticks,
    tickfontsize,
    labelfontsize,
    tick_direction,
):
    if max_major_ticks is not None:
        cbar.locator = _limited_locator(cbar.vmin, cbar.vmax, max_major_ticks)
        cbar.update_ticks()
    cbar.ax.tick_params(which="major", direction=tick_direction)
    if tickfontsize is not None:
        cbar.ax.tick_params(labelsize=tickfontsize)
    if labelfontsize is not None:
        cbar.ax.xaxis.label.set_size(labelfontsize)
        cbar.ax.yaxis.label.set_size(labelfontsize)


def _set_axis_max_major_ticks(axis, max_major_ticks):
    vmin, vmax = axis.get_view_interval()
    axis.set_major_locator(_limited_locator(vmin, vmax, max_major_ticks))


def _limited_locator(vmin, vmax, max_major_ticks):
    from matplotlib.ticker import FixedLocator

    return FixedLocator(_limited_tick_values(vmin, vmax, max_major_ticks))


def _limited_tick_values(vmin, vmax, max_major_ticks):
    from matplotlib.ticker import MaxNLocator

    vmin = float(vmin)
    vmax = float(vmax)
    max_major_ticks = int(max_major_ticks)
    if max_major_ticks <= 0:
        raise ValueError("max_major_ticks must be positive.")
    if np.isclose(vmin, vmax):
        return np.array([vmin])
    locator = MaxNLocator(nbins=max(1, max_major_ticks - 1), min_n_ticks=2)
    lower, upper = sorted((vmin, vmax))
    ticks = np.asarray(locator.tick_values(lower, upper), dtype=float)
    ticks = ticks[(ticks >= lower) & (ticks <= upper)]
    if ticks.size == 0:
        ticks = np.linspace(lower, upper, min(max_major_ticks, 2))
    if ticks.size > max_major_ticks:
        ticks = np.linspace(lower, upper, max_major_ticks)
    if vmin > vmax:
        ticks = ticks[::-1]
    return ticks


def _warn_if_geo_axis_has_limited_georeferencing(src, *, axis):
    if axis != "geo":
        return
    reasons = []
    if getattr(src, "crs", None) is None:
        reasons.append("missing CRS")
    if _transform_is_identity_like(src.transform):
        reasons.append("identity-like transform")
    if _bounds_are_index_like(src):
        reasons.append("index-like bounds")
    if not reasons:
        return

    warnings.warn(
        "plot_geotiff(axis='geo') is using a GeoTIFF with limited "
        f"georeferencing ({', '.join(reasons)}). Axis values come from the "
        "file bounds and may be pixel indices rather than longitude/latitude.",
        UserWarning,
        stacklevel=3,
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


def _apply_geo_formatters(ax, *, decimal_places=None):
    from ._formatters import LatFormatter, LonFormatter

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if decimal_places is None:
        x_places = _auto_geo_decimal_places(abs(xlim[1] - xlim[0]))
        y_places = _auto_geo_decimal_places(abs(ylim[1] - ylim[0]))
    else:
        x_places = y_places = int(decimal_places)
    ax.xaxis.set_major_formatter(LonFormatter(decimal_places=x_places))
    ax.yaxis.set_major_formatter(LatFormatter(decimal_places=y_places))


def _auto_geo_decimal_places(span):
    if span < 0.05:
        return 3
    if span < 1.0:
        return 2
    if span < 10.0:
        return 1
    return 0


def _coordinates_for_pcolormesh(x, y):
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if x_arr.ndim == 1 and y_arr.ndim == 1:
        return _centers_to_edges(x_arr), _centers_to_edges(y_arr)
    if x_arr.ndim == 2 and y_arr.ndim == 2:
        return x_arr, y_arr
    raise ValueError("x and y must both be 1-D coordinate vectors or 2-D meshes.")


def _centers_to_edges(coord):
    coord = np.asarray(coord, dtype=float)
    if coord.ndim != 1:
        raise ValueError("Coordinate centers must be 1-D.")
    if coord.size == 0:
        raise ValueError("Coordinate vector is empty.")
    if coord.size == 1:
        step = 1.0
        return np.array([coord[0] - 0.5 * step, coord[0] + 0.5 * step])
    mid = 0.5 * (coord[:-1] + coord[1:])
    first = coord[0] - (mid[0] - coord[0])
    last = coord[-1] + (coord[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])
