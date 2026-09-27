"""Private spatial rendering helpers shared by geodetic data objects."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import PolyCollection

from .decimation_geometry import corner_array_to_polygons


_RENDER_MODES = {"points", "cells", "auto"}


def normalize_render_mode(render_mode):
    """Return a canonical spatial rendering mode."""
    mode = str(render_mode).strip().lower()
    if mode not in _RENDER_MODES:
        raise ValueError("render_mode must be 'points', 'cells', or 'auto'")
    return mode


def resolve_color_limits(
    values,
    *,
    vmin=None,
    vmax=None,
    antisymmetric=True,
    significant_digits=2,
):
    """Resolve finite color limits without overriding explicit bounds.

    Automatic limits are zero-centred when ``antisymmetric`` is true and use
    the finite data range otherwise.  A caller may provide either or both
    bounds; every explicit bound remains authoritative.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("no finite values are available for color scaling")
    if not isinstance(antisymmetric, (bool, np.bool_)):
        raise TypeError("antisymmetric must be a boolean")

    if bool(antisymmetric):
        maximum = float(np.max(np.abs(finite_values)))
        if maximum == 0.0:
            automatic_min, automatic_max = -1.0e-3, 1.0e-3
        else:
            exponent = int(np.floor(np.log10(maximum)))
            factor = 10.0 ** (int(significant_digits) - 1 - exponent)
            limit = float(np.ceil(maximum * factor) / factor)
            automatic_min, automatic_max = -limit, limit
    else:
        automatic_min = float(np.min(finite_values))
        automatic_max = float(np.max(finite_values))
        if automatic_min == automatic_max:
            padding = max(abs(automatic_min), 1.0) * 1.0e-6
            automatic_min -= padding
            automatic_max += padding

    lower = automatic_min if vmin is None else float(vmin)
    upper = automatic_max if vmax is None else float(vmax)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("vmin and vmax must be finite")
    if lower >= upper:
        raise ValueError("vmin must be smaller than vmax")
    return lower, upper


def colorbar_ticks(vmin, vmax):
    """Return endpoint ticks and include zero only when it lies in range."""
    lower = float(vmin)
    upper = float(vmax)
    if lower < 0.0 < upper:
        return [lower, 0.0, upper]
    return [lower, upper]


def draw_spatial_field(
    ax,
    lon,
    lat,
    values,
    *,
    corners=None,
    render_mode="points",
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    decim=1,
    markersize=2,
    alpha=1.0,
    cell_edge_width=0.25,
):
    """Draw one scalar field as sample points or decimation cells.

    This helper owns display geometry only. It does not modify the source
    arrays or infer observation semantics. ``auto`` uses cells when valid
    corner geometry exists and otherwise uses points; explicit ``cells``
    rejects point-only data.
    """
    mode = normalize_render_mode(render_mode)
    lon = np.asarray(lon, dtype=float).reshape(-1)
    lat = np.asarray(lat, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if lon.size != lat.size or lon.size != values.size:
        raise ValueError("lon, lat, and values must have the same length")

    if not isinstance(decim, (int, np.integer)) or int(decim) < 1:
        raise ValueError("decim must be a positive integer")
    decim = int(decim)
    try:
        edge_width = float(cell_edge_width)
    except (TypeError, ValueError) as exc:
        raise TypeError("cell_edge_width must be a finite non-negative number") from exc
    if not np.isfinite(edge_width) or edge_width < 0.0:
        raise ValueError("cell_edge_width must be a finite non-negative number")

    polygons = None
    if mode != "points":
        polygons = corner_array_to_polygons(corners, expected_rows=values.size)
        if mode == "cells" and polygons is None:
            raise ValueError(
                "render_mode='cells' requires decimation corner geometry"
            )
        if mode == "auto":
            mode = "cells" if polygons is not None else "points"

    index = np.arange(values.size)[::decim]
    finite = (
        np.isfinite(lon[index])
        & np.isfinite(lat[index])
        & np.isfinite(values[index])
    )
    index = index[finite]
    if index.size == 0:
        raise ValueError("no finite spatial samples are available for plotting")

    if mode == "cells":
        selected = polygons[index]
        collection = PolyCollection(
            selected,
            array=values[index],
            cmap=cmap,
            edgecolors="black" if edge_width > 0.0 else "none",
            linewidths=edge_width,
            alpha=alpha,
            rasterized=True,
        )
        collection.set_clim(vmin, vmax)
        ax.add_collection(collection)
        x_coords = selected[:, :, 0]
        y_coords = selected[:, :, 1]
        bounds = (
            float(np.nanmin(x_coords)),
            float(np.nanmax(x_coords)),
            float(np.nanmin(y_coords)),
            float(np.nanmax(y_coords)),
        )
        return collection, bounds, mode

    scatter = ax.scatter(
        lon[index], lat[index], c=values[index], s=markersize,
        cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
        edgecolors="none", rasterized=True,
    )
    bounds = (
        float(np.nanmin(lon[index])),
        float(np.nanmax(lon[index])),
        float(np.nanmin(lat[index])),
        float(np.nanmax(lat[index])),
    )
    return scatter, bounds, mode
