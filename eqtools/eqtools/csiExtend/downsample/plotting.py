"""
Plotting helpers for downsampled geodetic data.

The functions here are intentionally independent from CSI. They only need the
lon/lat centers, observation values, and optional cell corners already read from
CSI ``*_ifg``/``*.rsp`` files.
"""

from contextlib import nullcontext

import numpy as np


def _resolve_cmap(cmap):
    if isinstance(cmap, str) and cmap.startswith("cmc."):
        try:
            import cmcrameri  # noqa: F401
        except ModuleNotFoundError:
            return "RdBu_r"
    return cmap


def _normalize_coordrange(coordrange):
    if coordrange is None:
        return None
    if len(coordrange) != 4:
        raise ValueError("coordrange must be [minlon, maxlon, minlat, maxlat].")
    values = [float(value) for value in coordrange]
    minlon, maxlon, minlat, maxlat = values
    if minlon >= maxlon or minlat >= maxlat:
        raise ValueError("coordrange must satisfy minlon < maxlon and minlat < maxlat.")
    return values


def _mask_from_coordrange(lon, lat, coordrange):
    if coordrange is None:
        return np.ones(lon.shape, dtype=bool)
    minlon, maxlon, minlat, maxlat = coordrange
    return (lon >= minlon) & (lon <= maxlon) & (lat >= minlat) & (lat <= maxlat)


def _corner_to_vertices(corner):
    corner = np.asarray(corner, dtype=float).ravel()
    if not np.all(np.isfinite(corner)):
        return None
    if corner.size == 4:
        ul_lon, ul_lat, lr_lon, lr_lat = corner
        return [
            (ul_lon, ul_lat),
            (lr_lon, ul_lat),
            (lr_lon, lr_lat),
            (ul_lon, lr_lat),
        ]
    if corner.size in (6, 8):
        return list(zip(corner[0::2], corner[1::2]))
    raise ValueError(
        "Each corner entry must have 4 values (UL/LR rectangle), "
        "6 values (triangle), or 8 values (quadrilateral)."
    )


def _corners_to_polygons(corners):
    if corners is None:
        return None
    polygons = []
    for corner in corners:
        vertices = _corner_to_vertices(corner)
        if vertices is not None:
            polygons.append(vertices)
    return polygons


def _resolve_limits(values, vmin=None, vmax=None, symmetry=True):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return vmin, vmax

    low = float(np.nanmin(finite)) if vmin is None else float(vmin)
    high = float(np.nanmax(finite)) if vmax is None else float(vmax)
    if symmetry:
        absmax = max(abs(low), abs(high))
        low, high = -absmax, absmax
    return low, high


def _normalize_colorbar_orientation(orientation):
    key = str(orientation).replace("-", "_").lower()
    aliases = {
        "h": "horizontal",
        "horizontal": "horizontal",
        "v": "vertical",
        "vertical": "vertical",
    }
    if key not in aliases:
        raise ValueError("colorbar_orientation must be 'horizontal' or 'vertical'.")
    return aliases[key]


def _default_colorbar_pad(orientation, mode, loc):
    if mode == "inside" and orientation == "horizontal":
        if "lower" in loc or "bottom" in loc:
            return 0.10
        return 0.04
    if mode == "inside":
        return 0.04
    return 0.02


def _resolve_colorbar_layout(orientation, mode="auto", loc=None,
                             size=None, thickness=None, pad=None):
    orientation = _normalize_colorbar_orientation(orientation)
    mode_key = str(mode).replace("-", "_").lower()
    aliases = {
        "auto": "auto",
        "inside": "inside",
        "inset": "inside",
        "outside": "outside",
        "external": "outside",
        "manual": "manual",
        "figure": "manual",
    }
    if mode_key not in aliases:
        raise ValueError("colorbar_mode must be 'auto', 'inside', 'outside', or 'manual'.")
    mode = aliases[mode_key]
    if mode == "auto":
        mode = "inside" if orientation == "horizontal" else "outside"

    if loc is None:
        if mode == "inside":
            loc = "lower left" if orientation == "horizontal" else "lower right"
        elif mode == "outside":
            loc = "bottom" if orientation == "horizontal" else "lower right"
        else:
            loc = "manual"
    loc = str(loc).replace("_", " ").replace("-", " ").lower()

    size = 0.4 if size is None else float(size)
    thickness = 0.025 if thickness is None else float(thickness)
    pad = _default_colorbar_pad(orientation, mode, loc) if pad is None else float(pad)
    if size <= 0.0 or thickness <= 0.0:
        raise ValueError("colorbar_size and colorbar_thickness must be positive.")
    if mode != "manual" and (size > 1.0 or thickness > 1.0):
        raise ValueError(
            "colorbar_size and colorbar_thickness are axes-relative fractions "
            "and must be <= 1 outside manual mode."
        )
    if pad < 0.0:
        raise ValueError("colorbar_pad must be non-negative.")

    return {
        "orientation": orientation,
        "mode": mode,
        "loc": loc,
        "size": size,
        "thickness": thickness,
        "pad": pad,
    }


def _inside_colorbar_bounds(orientation, loc, size, thickness, pad):
    if orientation == "horizontal":
        width, height = size, thickness
        if "right" in loc:
            x0 = 1.0 - pad - width
        elif "center" in loc:
            x0 = (1.0 - width) / 2.0
        else:
            x0 = pad

        if "upper" in loc or "top" in loc:
            y0 = 1.0 - pad - height
        else:
            y0 = pad
    else:
        width, height = thickness, size
        if "left" in loc:
            x0 = pad
        else:
            x0 = 1.0 - pad - width

        if "upper" in loc or "top" in loc:
            y0 = 1.0 - pad - height
        elif "center" in loc:
            y0 = (1.0 - height) / 2.0
        else:
            y0 = pad

    return [x0, y0, width, height]


def _outside_colorbar_bounds(ax, orientation, loc, size, thickness, pad):
    bbox = ax.get_position()
    if orientation == "vertical":
        width = thickness * bbox.width
        height = size * bbox.height
        if "left" in loc:
            x0 = bbox.x0 - pad * bbox.width - width
        else:
            x0 = bbox.x1 + pad * bbox.width

        if "upper" in loc or "top" in loc:
            y0 = bbox.y1 - height
        elif "center" in loc or "middle" in loc:
            y0 = bbox.y0 + (bbox.height - height) / 2.0
        else:
            y0 = bbox.y0
    else:
        width = size * bbox.width
        height = thickness * bbox.height
        if "right" in loc:
            x0 = bbox.x1 - width
        elif "left" in loc:
            x0 = bbox.x0
        else:
            x0 = bbox.x0 + (bbox.width - width) / 2.0

        if "upper" in loc or "top" in loc:
            y0 = bbox.y1 + pad * bbox.height
        else:
            y0 = bbox.y0 - pad * bbox.height - height

    return [x0, y0, width, height]


def _make_colorbar_axes(fig, ax, layout, colorbar_x=None, colorbar_y=None,
                        colorbar_length=None, colorbar_height=None):
    if layout["mode"] == "manual":
        return fig.add_axes([
            0.1 if colorbar_x is None else colorbar_x,
            0.1 if colorbar_y is None else colorbar_y,
            0.4 if colorbar_length is None else colorbar_length,
            0.02 if colorbar_height is None else colorbar_height,
        ])

    if layout["mode"] == "inside":
        bounds = _inside_colorbar_bounds(
            layout["orientation"],
            layout["loc"],
            layout["size"],
            layout["thickness"],
            layout["pad"],
        )
        return ax.inset_axes(bounds, transform=ax.transAxes)

    bounds = _outside_colorbar_bounds(
        ax,
        layout["orientation"],
        layout["loc"],
        layout["size"],
        layout["thickness"],
        layout["pad"],
    )
    return fig.add_axes(bounds)


def _set_colorbar_label_position(cb, orientation, layout, cb_label_loc=None):
    if orientation == "vertical":
        if cb_label_loc is None:
            cb_label_loc = "left" if layout["mode"] == "inside" else "right"
        cb.ax.yaxis.set_label_position(cb_label_loc)
        cb.ax.yaxis.set_ticks_position(cb_label_loc)
        return

    if cb_label_loc is None:
        cb_label_loc = "bottom"
    cb.ax.xaxis.set_label_position(cb_label_loc)
    cb.ax.xaxis.set_ticks_position(cb_label_loc)


def _plot_faults(ax, faults, trace_color="black", trace_linewidth=0.5):
    if not faults:
        return
    for fault in faults:
        if hasattr(fault, "lon") and hasattr(fault, "lat"):
            ax.plot(fault.lon, fault.lat, color=trace_color, lw=trace_linewidth)
        elif hasattr(fault, "__getitem__") and "lon" in fault and "lat" in fault:
            ax.plot(fault["lon"], fault["lat"], color=trace_color, lw=trace_linewidth)


def plot_decimated_geodata(lon, lat, values, corners=None, *,
                           style="cells", coordrange=None, factor4plot=1.0,
                           vmin=None, vmax=None, symmetry=True, cmap="cmc.roma_r",
                           faults=None, trace_color="black", trace_linewidth=0.5,
                           figsize=(3.0, 5.0), dpi=300, savefig=None, show=False,
                           edgewidth=0.1, edgecolor="black", alpha=1.0,
                           markersize=10, add_colorbar=True,
                           colorbar_orientation="vertical", colorbar_mode="auto",
                           colorbar_loc=None, colorbar_size=None,
                           colorbar_thickness=None, colorbar_pad=None,
                           colorbar_x=None, colorbar_y=None,
                           colorbar_length=None, colorbar_height=None,
                           cb_label=None, cb_label_loc=None,
                           tickfontsize=10, labelfontsize=10,
                           xlabel="Longitude", ylabel="Latitude",
                           title=None, style_context=("science",), fontsize=None):
    """
    Plot downsampled SAR or optical data on lon/lat axes.

    Parameters
    ----------
    lon, lat, values : array-like
        Downsampled sample centers and values.
    corners : array-like, optional
        Cell corners from CSI ``read_from_varres``. Supported shapes are 4
        values for legacy rectangle UL/LR, 6 for triangular cells, and 8 for
        quadrilateral cells.
    style : {"cells", "points"}
        ``cells`` draws polygons when corners are available. ``points`` draws
        sample centers.
    coordrange : sequence, optional
        ``[minlon, maxlon, minlat, maxlat]`` for display extent. ``None`` uses
        all downsampled data.
    factor4plot : float
        Display multiplier only, for example 100 for meter-to-centimeter plots.

    Returns
    -------
    fig, ax
        Matplotlib figure and map axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    try:
        from eqtools.viztools import sci_plot_style, set_degree_formatter
    except Exception:
        sci_plot_style = None
        set_degree_formatter = None

    style_key = str(style).replace("-", "_").lower()
    if style_key not in ("cells", "points"):
        raise ValueError("style must be 'cells' or 'points'.")

    coordrange = _normalize_coordrange(coordrange)
    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if lon.size != lat.size or lon.size != values.size:
        raise ValueError("lon, lat, and values must have the same length.")

    display_values = values * float(factor4plot)
    finite_mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(display_values)
    if not np.any(finite_mask):
        raise ValueError("No finite lon/lat/value samples are available for plotting.")
    range_mask = _mask_from_coordrange(lon, lat, coordrange)
    visible_mask = finite_mask & range_mask
    scale_values = display_values[visible_mask]
    if scale_values.size == 0:
        scale_values = display_values[finite_mask]
    vmin, vmax = _resolve_limits(scale_values, vmin=vmin, vmax=vmax, symmetry=symmetry)

    context = nullcontext()
    if sci_plot_style is not None and style_context is not None:
        context = sci_plot_style(style=style_context, fontsize=fontsize, figsize=figsize)

    with context:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        cmap = _resolve_cmap(cmap)
        norm = Normalize(vmin=vmin, vmax=vmax)
        mappable = None

        polygons = _corners_to_polygons(corners) if style_key == "cells" else None
        if style_key == "cells" and polygons:
            if len(polygons) != values.size:
                raise ValueError("corners length must match values length.")
            selected_polygons = [poly for poly, keep in zip(polygons, visible_mask) if keep]
            selected_values = display_values[visible_mask]
            if not selected_polygons:
                selected_polygons = polygons
                selected_values = display_values
            collection = PolyCollection(
                selected_polygons,
                array=np.asarray(selected_values, dtype=float),
                cmap=cmap,
                norm=norm,
                edgecolors=edgecolor,
                linewidths=edgewidth,
                alpha=alpha,
            )
            ax.add_collection(collection)
            ax.autoscale_view()
            mappable = collection
        else:
            selected = visible_mask if np.any(visible_mask) else finite_mask
            mappable = ax.scatter(
                lon[selected],
                lat[selected],
                c=display_values[selected],
                s=markersize,
                cmap=cmap,
                norm=norm,
                linewidths=edgewidth,
                edgecolors=edgecolor,
                alpha=alpha,
            )

        _plot_faults(ax, faults, trace_color=trace_color, trace_linewidth=trace_linewidth)
        if coordrange is not None:
            ax.set_xlim(coordrange[0], coordrange[1])
            ax.set_ylim(coordrange[2], coordrange[3])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        if set_degree_formatter is not None:
            set_degree_formatter(ax, axis="both")

        if add_colorbar:
            layout = _resolve_colorbar_layout(
                colorbar_orientation,
                mode=colorbar_mode,
                loc=colorbar_loc,
                size=colorbar_size,
                thickness=colorbar_thickness,
                pad=colorbar_pad,
            )
            fig.canvas.draw()
            cax = _make_colorbar_axes(
                fig,
                ax,
                layout,
                colorbar_x=colorbar_x,
                colorbar_y=colorbar_y,
                colorbar_length=colorbar_length,
                colorbar_height=colorbar_height,
            )
            cb = fig.colorbar(mappable, cax=cax, orientation=layout["orientation"])
            cb.ax.tick_params(labelsize=tickfontsize)
            if cb_label:
                cb.set_label(cb_label, fontdict={"size": labelfontsize})
            _set_colorbar_label_position(
                cb,
                layout["orientation"],
                layout,
                cb_label_loc=cb_label_loc,
            )

        if savefig:
            fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig, ax
