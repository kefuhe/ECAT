"""Check-plot helpers for the downsampling super app.

This module is deliberately scoped to raw/decimated QC figures used by
``ecat-downsample``.  It does not own CSI fault, slip-distribution, or other
domain plotting.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .plotting import (
    _corners_to_polygons,
    _inside_colorbar_bounds,
    _make_colorbar_axes,
    _mask_from_coordrange,
    _normalize_coordrange,
    _plot_faults,
    _resolve_cmap,
    _resolve_colorbar_layout,
    _set_colorbar_label_position,
)


@dataclass
class ComponentMap:
    name: str
    lon: object
    lat: object
    values: object
    corners: object = None
    label: str = None
    vmin: float = None
    vmax: float = None


def _needs_column_colorbar_spacing(layout, n_components, colorbar_orientation, colorbar_mode):
    if layout != "columns" or n_components <= 1:
        return False
    orientation = str(colorbar_orientation).replace("-", "_").lower()
    mode = str(colorbar_mode).replace("-", "_").lower()
    return orientation in ("v", "vertical") and mode in ("auto", "outside", "external")


def _needs_column_horizontal_colorbar_pad(layout, n_components, colorbar_orientation, colorbar_mode):
    if layout != "columns" or n_components <= 1:
        return False
    orientation = str(colorbar_orientation).replace("-", "_").lower()
    mode = str(colorbar_mode).replace("-", "_").lower()
    return orientation in ("h", "horizontal") and mode in ("auto", "outside", "external")


def _resolve_component_colorbar_orientation(orientation, layout, n_components):
    key = str(orientation).replace("-", "_").lower()
    if key == "auto":
        return "horizontal" if layout == "columns" and n_components > 1 else "vertical"
    return orientation


def _anchor_column_axes(axes):
    """Pack equal-aspect column maps toward the center of their subplot cells."""
    if len(axes) == 2:
        axes[0].set_anchor("E")
        axes[1].set_anchor("W")


def _freeze_manual_layout(fig):
    """Stop automatic layout from moving data axes after manual colorbars exist."""
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)
    else:
        fig.set_constrained_layout(False)
    fig.canvas.draw()


def _normalize_auto_number(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "auto", "none", "null"):
        return None
    return float(value)


def _auto_base_fontsize(width_in, *, min_size=6.0, max_size=10.0,
                        single_width=3.5, double_width=7.0):
    """Map publication figure width to a conservative check-plot font size."""
    width = float(width_in)
    if double_width <= single_width:
        return max_size
    ratio = (width - single_width) / (double_width - single_width)
    size = min_size + ratio * (max_size - min_size)
    return min(max(size, min_size), max_size)


def _resolve_plot_font_sizes(figsize, *, fontsize=None, tickfontsize=None, labelfontsize=None):
    base = _normalize_auto_number(fontsize)
    if base is None:
        base = _auto_base_fontsize(figsize[0])
    tick = _normalize_auto_number(tickfontsize)
    if tick is None:
        tick = max(base - 1.0, 6.0)
    label = _normalize_auto_number(labelfontsize)
    if label is None:
        label = base
    return base, tick, label


def _cap_interactive_dpi(fig, *, max_dpi=200):
    """Keep interactive Matplotlib windows usable without changing saved output."""
    current = float(fig.get_dpi())
    capped = min(current, float(max_dpi))
    if capped != current:
        fig.set_dpi(capped)
        fig.canvas.draw()


class _AttachedColorbarLocator:
    """Locate an outside colorbar from the current active data-axis box."""

    def __init__(self, fig, data_ax, layout):
        self.fig = fig
        self.data_ax = data_ax
        self.layout = layout

    def __call__(self, cax, renderer):
        from matplotlib.transforms import Bbox

        axes_bbox = self.data_ax.get_position()
        orientation = self.layout["orientation"]
        loc = self.layout["loc"]
        size = self.layout["size"]
        thickness = self.layout["thickness"]
        pad = self.layout["pad"]

        if orientation == "horizontal":
            width = size * axes_bbox.width
            height = thickness * axes_bbox.height
            if "right" in loc:
                x0 = axes_bbox.x1 - width
            elif "left" in loc:
                x0 = axes_bbox.x0
            else:
                x0 = axes_bbox.x0 + (axes_bbox.width - width) / 2.0

            tight_bbox = self.data_ax.get_tightbbox(renderer)
            if tight_bbox is not None:
                tight_bbox = tight_bbox.transformed(self.fig.transFigure.inverted())
            else:
                tight_bbox = axes_bbox
            if "upper" in loc or "top" in loc:
                y0 = tight_bbox.y1 + pad * axes_bbox.height
            else:
                y0 = tight_bbox.y0 - pad * axes_bbox.height - height
            y0 = min(max(y0, 0.01), 0.99 - height)
        else:
            width = thickness * axes_bbox.width
            height = size * axes_bbox.height
            if "left" in loc:
                x0 = axes_bbox.x0 - pad * axes_bbox.width - width
            else:
                x0 = axes_bbox.x1 + pad * axes_bbox.width
            x0 = min(max(x0, 0.01), 0.99 - width)

            if "upper" in loc or "top" in loc:
                y0 = axes_bbox.y1 - height
            elif "center" in loc or "middle" in loc:
                y0 = axes_bbox.y0 + (axes_bbox.height - height) / 2.0
            else:
                y0 = axes_bbox.y0
            y0 = min(max(y0, 0.01), 0.99 - height)

        return Bbox.from_bounds(x0, y0, width, height)


def _compact_component_label(label):
    if not label:
        return label
    return str(label).replace(" disp.", "\ndisp.")


def _font_points_to_figure_fraction(fig, points, axis):
    size_inches = fig.get_figwidth() if axis == "x" else fig.get_figheight()
    return float(points) / 72.0 / size_inches


def _effective_colorbar_label(label, compact_label):
    if label and compact_label:
        return _compact_component_label(label)
    return label


def _max_label_line_count(labels):
    counts = [str(label).count("\n") + 1 for label in labels if label]
    return max(counts, default=0)


def _normalize_panel_pad(panel_pad):
    if panel_pad is None:
        return 0.0
    if isinstance(panel_pad, str) and panel_pad.strip().lower() in ("", "none", "null", "auto"):
        return 0.0
    panel_pad = float(panel_pad)
    if panel_pad < 0.0:
        raise ValueError("panel_pad must be non-negative.")
    return panel_pad


def _normalize_tick_direction(direction, *, default="out"):
    direction = str(direction or default).replace("-", "_").lower()
    return default if direction == "auto" else direction


def _normalize_optional_positive_int(value, *, name):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null", "auto"):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer or null.")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer or null.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer or null.")
    return value


def _normalize_bool(value, *, name):
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("true", "yes", "y", "1", "on"):
            return True
        if key in ("false", "no", "n", "0", "off"):
            return False
        raise ValueError(f"{name} must be true or false.")
    return bool(value)


def _limited_tick_values(vmin, vmax, max_major_ticks):
    max_major_ticks = _normalize_optional_positive_int(
        max_major_ticks,
        name="max_major_ticks",
    )
    if max_major_ticks is None:
        return None
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    from matplotlib.ticker import MaxNLocator

    locator = MaxNLocator(nbins=max(1, max_major_ticks - 1), min_n_ticks=2)
    ticks = np.asarray(locator.tick_values(vmin, vmax), dtype=float)
    eps = max(abs(vmax - vmin), 1.0) * 1e-9
    ticks = ticks[(ticks >= vmin - eps) & (ticks <= vmax + eps)]
    ticks = np.unique(ticks)
    if ticks.size == 0:
        ticks = np.linspace(vmin, vmax, min(max_major_ticks, 2))
    if ticks.size > max_major_ticks:
        indices = np.linspace(0, ticks.size - 1, max_major_ticks)
        ticks = ticks[np.unique(np.rint(indices).astype(int))]
    return ticks


def _set_limited_major_ticks(axis, vmin, vmax, max_major_ticks):
    ticks = _limited_tick_values(vmin, vmax, max_major_ticks)
    if ticks is None:
        return
    from matplotlib.ticker import FixedLocator

    axis.set_major_locator(FixedLocator(ticks))


def _set_minor_locator(axis, enabled, subdivisions, *, name):
    from matplotlib.ticker import AutoMinorLocator, NullLocator

    if _normalize_bool(enabled, name=name):
        subdivisions = _normalize_optional_positive_int(
            subdivisions,
            name=f"{name}_subdivisions",
        )
        axis.set_minor_locator(AutoMinorLocator(subdivisions or 2))
    else:
        axis.set_minor_locator(NullLocator())


def resolve_figsize(figsize, *, aspect=None, height=None):
    """Resolve a tuple or a registered viztools figure-size name."""
    if figsize is None or str(figsize).lower() == "auto":
        figsize = "single"
    if isinstance(figsize, (list, tuple)) and len(figsize) == 2:
        return (float(figsize[0]), float(figsize[1]))
    if isinstance(figsize, (int, float)):
        width = float(figsize)
        return (width, width * (0.75 if aspect is None else float(aspect)))
    if isinstance(figsize, str):
        from eqtools.viztools import list_column_widths, publication_figsize

        key = figsize.lower()
        if key not in list_column_widths():
            available = ", ".join(sorted(list_column_widths()))
            raise ValueError(
                f"Unknown figsize preset {figsize!r}. Available presets: {available}."
            )
        kwargs = {}
        if aspect is not None:
            kwargs["aspect"] = float(aspect)
        if height is not None:
            kwargs["height"] = float(height)
        return publication_figsize(key, **kwargs)
    raise ValueError("figsize must be a preset string, number, or [width, height].")


def robust_limits(values, *, vmin=None, vmax=None, percentile=99.0, symmetry=True):
    """Resolve display limits from explicit bounds or a central percentile."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return vmin, vmax
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)

    percentile = 99.0 if percentile is None else float(percentile)
    if not 0.0 < percentile <= 100.0:
        raise ValueError("plot percentile must be in the interval (0, 100].")
    tail = (100.0 - percentile) / 2.0
    lo, hi = np.nanpercentile(finite, [tail, 100.0 - tail])
    lo = float(lo) if vmin is None else float(vmin)
    hi = float(hi) if vmax is None else float(vmax)
    if symmetry:
        absmax = max(abs(lo), abs(hi))
        lo, hi = -absmax, absmax
    return lo, hi


def apply_plot_stride(lon, lat, values, stride):
    stride = int(stride or 1)
    if stride <= 0:
        raise ValueError("plot_stride must be a positive integer.")
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    values = np.asarray(values)
    if values.ndim >= 2:
        return lon[::stride, ::stride], lat[::stride, ::stride], values[::stride, ::stride]
    return lon[::stride], lat[::stride], values[::stride]


def _visible_values(component, coordrange, factor4plot):
    lon = np.asarray(component.lon, dtype=float)
    lat = np.asarray(component.lat, dtype=float)
    values = np.asarray(component.values, dtype=float) * float(factor4plot)
    flat_lon = lon.ravel()
    flat_lat = lat.ravel()
    flat_values = values.ravel()
    finite = np.isfinite(flat_lon) & np.isfinite(flat_lat) & np.isfinite(flat_values)
    if coordrange is not None:
        finite &= _mask_from_coordrange(flat_lon, flat_lat, coordrange)
    if np.any(finite):
        return flat_values[finite]
    return flat_values[np.isfinite(flat_values)]


def _draw_component(
    ax,
    component,
    *,
    cell_style,
    coordrange,
    factor4plot,
    cmap,
    vmin,
    vmax,
    edgewidth,
    edgecolor,
    alpha,
    markersize,
):
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    lon = np.asarray(component.lon, dtype=float)
    lat = np.asarray(component.lat, dtype=float)
    values = np.asarray(component.values, dtype=float) * float(factor4plot)
    cmap = _resolve_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)

    if component.corners is not None and cell_style == "cells":
        flat_lon = lon.ravel()
        flat_lat = lat.ravel()
        flat_values = values.ravel()
        finite = np.isfinite(flat_lon) & np.isfinite(flat_lat) & np.isfinite(flat_values)
        if coordrange is not None:
            finite &= _mask_from_coordrange(flat_lon, flat_lat, coordrange)
        polygons = _corners_to_polygons(component.corners)
        if len(polygons) != flat_values.size:
            raise ValueError("corners length must match component value length.")
        selected_polygons = [poly for poly, keep in zip(polygons, finite) if keep]
        selected_values = flat_values[finite]
        if not selected_polygons:
            selected_polygons = polygons
            selected_values = flat_values
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
        return collection

    if lon.ndim == 2 and lat.ndim == 2 and values.ndim == 2:
        return ax.pcolormesh(lon, lat, values, cmap=cmap, norm=norm, shading="auto")

    finite = np.isfinite(lon.ravel()) & np.isfinite(lat.ravel()) & np.isfinite(values.ravel())
    if coordrange is not None:
        finite &= _mask_from_coordrange(lon.ravel(), lat.ravel(), coordrange)
    selected = finite if np.any(finite) else np.isfinite(values.ravel())
    return ax.scatter(
        lon.ravel()[selected],
        lat.ravel()[selected],
        c=values.ravel()[selected],
        s=markersize,
        cmap=cmap,
        norm=norm,
        linewidths=edgewidth,
        edgecolors=edgecolor,
        alpha=alpha,
    )


def _add_panel_colorbar(fig, ax, mappable, *, orientation, mode, loc, size,
                        thickness, pad, label, tickfontsize, labelfontsize,
                        cb_label_loc, tick_direction="out",
                        max_major_ticks=5, minor_ticks=False,
                        minor_subdivisions=2,
                        compact_label=False,
                        reserve_outside_space=True,
                        keep_inside_canvas=True):
    label = _effective_colorbar_label(label, compact_label)
    layout = _resolve_colorbar_layout(
        orientation,
        mode=mode,
        loc=loc,
        size=size,
        thickness=thickness,
        pad=pad,
    )
    if layout["mode"] == "inside":
        cax = ax.inset_axes(
            _inside_colorbar_bounds(
                layout["orientation"],
                layout["loc"],
                layout["size"],
                layout["thickness"],
                layout["pad"],
            ),
            transform=ax.transAxes,
        )
    elif layout["mode"] == "outside":
        if reserve_outside_space:
            if layout["orientation"] == "horizontal":
                _reserve_horizontal_colorbar_space(
                    fig,
                    ax,
                    layout,
                    tickfontsize=tickfontsize,
                    labelfontsize=labelfontsize,
                    label=label,
                )
            else:
                _reserve_vertical_colorbar_space(
                    fig,
                    ax,
                    layout,
                    tickfontsize=tickfontsize,
                    labelfontsize=labelfontsize,
                    label=label,
                )
        cax = fig.add_axes([0.0, 0.0, 0.01, 0.01])
        cax.set_axes_locator(_AttachedColorbarLocator(fig, ax, layout))
        cax.set_in_layout(False)
    else:
        cax = _make_colorbar_axes(fig, ax, layout)
    cb = fig.colorbar(mappable, cax=cax, orientation=layout["orientation"])
    _configure_colorbar_ticks(
        cb,
        layout,
        max_major_ticks=max_major_ticks,
        minor_ticks=minor_ticks,
        minor_subdivisions=minor_subdivisions,
    )
    _style_colorbar_ticks(
        fig,
        cb,
        layout,
        tickfontsize=tickfontsize,
        tick_direction=tick_direction,
    )
    if label:
        cb.set_label(label, fontdict={"size": labelfontsize})
    _set_colorbar_label_position(cb, layout["orientation"], layout, cb_label_loc)
    if keep_inside_canvas and layout["mode"] != "outside":
        _keep_colorbar_tightbbox_inside_canvas(fig, ax, cb.ax, layout)
    return cb


def _configure_colorbar_ticks(cb, layout, *, max_major_ticks, minor_ticks, minor_subdivisions):
    axis = cb.ax.yaxis if layout["orientation"] == "vertical" else cb.ax.xaxis
    ticks = _limited_tick_values(cb.vmin, cb.vmax, max_major_ticks)
    if ticks is not None:
        cb.set_ticks(ticks)
    _set_minor_locator(
        axis,
        minor_ticks,
        minor_subdivisions,
        name="colorbar_minor_ticks",
    )


def _style_colorbar_ticks(fig, cb, layout, *, tickfontsize, tick_direction="out"):
    """Make colorbar major ticks visible relative to the bar thickness."""
    fig.canvas.draw()
    bbox = cb.ax.get_position()
    if layout["orientation"] == "vertical":
        thickness_inches = bbox.width * fig.get_figwidth()
    else:
        thickness_inches = bbox.height * fig.get_figheight()
    tick_length = max(1.5, 0.5 * thickness_inches * 72.0)
    tick_direction = _normalize_tick_direction(tick_direction)
    cb.ax.tick_params(
        which="major",
        labelsize=tickfontsize,
        length=tick_length,
        width=0.8,
        direction=tick_direction,
        colors="black",
    )
    cb.ax.tick_params(
        which="minor",
        length=max(1.0, 0.5 * tick_length),
        width=0.6,
        direction=tick_direction,
        colors="black",
    )
    for tick in (
        cb.ax.xaxis.get_major_ticks()
        + cb.ax.xaxis.get_minor_ticks()
        + cb.ax.yaxis.get_major_ticks()
        + cb.ax.yaxis.get_minor_ticks()
    ):
        tick.tick1line.set_zorder(10)
        tick.tick2line.set_zorder(10)
        tick.tick1line.set_color("black")
        tick.tick2line.set_color("black")
        tick.tick1line.set_markeredgewidth(0.8)
        tick.tick2line.set_markeredgewidth(0.8)


def _reserve_horizontal_colorbar_space(fig, ax, layout, *, tickfontsize, labelfontsize, label):
    """Shrink the data axis only when needed so outside horizontal cbar fits on screen."""
    tickfontsize = 10.0 if tickfontsize is None else float(tickfontsize)
    labelfontsize = 10.0 if labelfontsize is None else float(labelfontsize)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_position()
    tight_bbox = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    height = layout["thickness"] * axes_bbox.height
    pad = layout["pad"] * axes_bbox.height
    label_points = tickfontsize + (labelfontsize if label else 0.0) + 8.0
    text_margin = _font_points_to_figure_fraction(fig, label_points, "y")
    figure_margin = 0.01
    required = height + pad + text_margin + figure_margin

    if "upper" in layout["loc"] or "top" in layout["loc"]:
        shortage = tight_bbox.y1 + required - 1.0
        if shortage <= 0.0:
            return
        delta = min(shortage, axes_bbox.height * 0.35)
        ax.set_position([
            axes_bbox.x0,
            axes_bbox.y0,
            axes_bbox.width,
            max(axes_bbox.height - delta, axes_bbox.height * 0.5),
        ])
        fig.canvas.draw()
        return

    shortage = required - tight_bbox.y0
    if shortage <= 0.0:
        return
    delta = min(shortage, axes_bbox.height * 0.35)
    ax.set_position([
        axes_bbox.x0,
        axes_bbox.y0 + delta,
        axes_bbox.width,
        max(axes_bbox.height - delta, axes_bbox.height * 0.5),
    ])
    fig.canvas.draw()


def _reserve_vertical_colorbar_space(fig, ax, layout, *, tickfontsize, labelfontsize, label):
    """Shrink the data axis only when needed so outside vertical cbar fits on screen."""
    tickfontsize = 10.0 if tickfontsize is None else float(tickfontsize)
    labelfontsize = 10.0 if labelfontsize is None else float(labelfontsize)
    fig.canvas.draw()
    axes_bbox = ax.get_position()
    width = layout["thickness"] * axes_bbox.width
    pad = layout["pad"] * axes_bbox.width
    label_points = tickfontsize * 4.5 + (labelfontsize if label else 0.0) + 8.0
    text_margin = _font_points_to_figure_fraction(fig, label_points, "x")
    figure_margin = 0.01
    required = width + pad + text_margin + figure_margin

    if "left" in layout["loc"]:
        shortage = required - axes_bbox.x0
        if shortage <= 0.0:
            return
        delta = min(shortage, axes_bbox.width * 0.35)
        ax.set_position([
            axes_bbox.x0 + delta,
            axes_bbox.y0,
            max(axes_bbox.width - delta, axes_bbox.width * 0.5),
            axes_bbox.height,
        ])
        fig.canvas.draw()
        return

    shortage = axes_bbox.x1 + required - 1.0
    if shortage <= 0.0:
        return
    delta = min(shortage, axes_bbox.width * 0.35)
    ax.set_position([
        axes_bbox.x0,
        axes_bbox.y0,
        max(axes_bbox.width - delta, axes_bbox.width * 0.5),
        axes_bbox.height,
    ])
    fig.canvas.draw()


def _reserve_column_vertical_colorbar_space(fig, axes, layout, *, tickfontsize, labelfontsize, labels, panel_pad=None):
    """Reserve equal-width column panels for per-axis outside vertical colorbars."""
    if not axes:
        return
    tickfontsize = 10.0 if tickfontsize is None else float(tickfontsize)
    labelfontsize = 10.0 if labelfontsize is None else float(labelfontsize)
    panel_pad = _normalize_panel_pad(panel_pad)
    fig.canvas.draw()
    positions = [ax.get_position() for ax in axes]
    left = min(pos.x0 for pos in positions)
    right_limit = 0.99
    available = right_limit - left
    if available <= 0.0:
        return

    has_label = any(labels)
    label_points = tickfontsize * 4.5 + (labelfontsize if has_label else 0.0) + 8.0
    text_margin = _font_points_to_figure_fraction(fig, label_points, "x") + 0.01
    width_factor = layout["thickness"] + layout["pad"]
    n_axes = len(positions)
    panel_pad_total = max(n_axes - 1, 0) * panel_pad
    target_width = (
        available - n_axes * text_margin - panel_pad_total
    ) / (n_axes * (1.0 + width_factor))
    if target_width <= 0.0:
        return

    original_width = min(pos.width for pos in positions)
    if target_width >= original_width:
        return
    target_width = max(target_width, original_width * 0.45)
    gap = width_factor * target_width + text_margin
    base_y0 = min(pos.y0 for pos in positions)
    base_y1 = max(pos.y1 for pos in positions)
    base_height = base_y1 - base_y0
    colorbar_on_left = "left" in layout["loc"]

    for index, ax in enumerate(axes):
        if colorbar_on_left:
            x0 = left + gap + index * (target_width + gap + panel_pad)
        else:
            x0 = left + index * (target_width + gap + panel_pad)
        ax.set_position([x0, base_y0, target_width, base_height])
    fig.canvas.draw()


def _reserve_column_horizontal_colorbar_space(fig, axes, layout, *, tickfontsize, labelfontsize, labels, panel_pad=None):
    """Reserve equal-size column panels for per-axis outside horizontal colorbars."""
    if not axes:
        return
    tickfontsize = 10.0 if tickfontsize is None else float(tickfontsize)
    labelfontsize = 10.0 if labelfontsize is None else float(labelfontsize)
    panel_pad = _normalize_panel_pad(panel_pad)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    positions = [ax.get_position() for ax in axes]
    tight_bboxes = [
        ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        for ax in axes
    ]
    target_width = min(pos.width for pos in positions)
    target_height = min(pos.height for pos in positions)
    line_count = _max_label_line_count(labels)
    label_points = tickfontsize * 1.2 + (labelfontsize * 1.4 * line_count) + 10.0
    text_margin = _font_points_to_figure_fraction(fig, label_points, "y")
    height = layout["thickness"] * target_height
    pad = layout["pad"] * target_height
    figure_margin = 0.01
    required = height + pad + text_margin + figure_margin

    if "upper" in layout["loc"] or "top" in layout["loc"]:
        shortage = max(tight_bbox.y1 + required - 1.0 for tight_bbox in tight_bboxes)
        delta = min(max(shortage, 0.0), target_height * 0.35)
        base_y0 = max(pos.y0 for pos in positions)
    else:
        shortage = max(required - tight_bbox.y0 for tight_bbox in tight_bboxes)
        delta = min(max(shortage, 0.0), target_height * 0.35)
        base_y0 = max(pos.y0 for pos in positions) + delta
    target_height = max(target_height - delta, target_height * 0.5)

    for ax, pos in zip(axes, positions):
        center_x = pos.x0 + pos.width / 2.0
        ax.set_position([
            center_x - target_width / 2.0,
            base_y0,
            target_width,
            target_height,
        ])
    fig.canvas.draw()
    _ensure_column_panel_gap(fig, axes, panel_pad)


def _ensure_column_panel_gap(fig, axes, panel_pad):
    """Enforce a minimum figure-fraction gap between neighboring active map panels."""
    panel_pad = _normalize_panel_pad(panel_pad)
    if panel_pad <= 0.0 or len(axes) < 2:
        return
    fig.canvas.draw()
    active = [ax.get_position() for ax in axes]
    shortage = max(
        panel_pad - (right.x0 - left.x1)
        for left, right in zip(active[:-1], active[1:])
    )
    if shortage <= 0.0:
        return
    positions = [ax.get_position(original=True) for ax in axes]
    n_axes = len(positions)
    shrink = shortage * (n_axes - 1) / n_axes
    min_width = min(pos.width for pos in positions) * 0.45
    new_width = max(min(pos.width for pos in positions) - shrink, min_width)
    total_width = n_axes * new_width + (n_axes - 1) * panel_pad
    left = min(pos.x0 for pos in positions)
    right = max(pos.x1 for pos in positions)
    span_center = (left + right) / 2.0
    start = span_center - total_width / 2.0
    if start < 0.01 or start + total_width > 0.99:
        start = max(0.01, min(start, 0.99 - total_width))
    base_y0 = min(pos.y0 for pos in positions)
    base_y1 = max(pos.y1 for pos in positions)
    for index, ax in enumerate(axes):
        x0 = start + index * (new_width + panel_pad)
        if len(axes) == 2:
            x0 = min(max(x0, 0.01), 0.99 - new_width)
        ax.set_position([x0, base_y0, new_width, base_y1 - base_y0])
    fig.canvas.draw()


def _keep_colorbar_tightbbox_inside_canvas(fig, data_ax, cbar_ax, layout):
    """Final small nudge for interactive canvases; saving still uses tight bbox."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_bbox = cbar_ax.get_tightbbox(renderer)
    if tight_bbox is None:
        return
    bbox = tight_bbox.transformed(fig.transFigure.inverted())
    margin = 0.005
    dx = dy = 0.0
    if bbox.x0 < margin:
        dx = margin - bbox.x0
    elif bbox.x1 > 1.0 - margin:
        dx = 1.0 - margin - bbox.x1
    if bbox.y0 < margin:
        dy = margin - bbox.y0
    elif bbox.y1 > 1.0 - margin:
        dy = 1.0 - margin - bbox.y1
    if dx == 0.0 and dy == 0.0:
        return

    for axis in (data_ax, cbar_ax):
        pos = axis.get_position()
        axis.set_position([pos.x0 + dx, pos.y0 + dy, pos.width, pos.height])
    fig.canvas.draw()


def _style_map_axis_ticks(
    ax,
    *,
    tick_direction="out",
    max_major_ticks=5,
    minor_ticks=False,
    minor_subdivisions=2,
):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    _set_limited_major_ticks(ax.xaxis, xlim[0], xlim[1], max_major_ticks)
    _set_limited_major_ticks(ax.yaxis, ylim[0], ylim[1], max_major_ticks)
    _set_minor_locator(
        ax.xaxis,
        minor_ticks,
        minor_subdivisions,
        name="axis_minor_ticks",
    )
    _set_minor_locator(
        ax.yaxis,
        minor_ticks,
        minor_subdivisions,
        name="axis_minor_ticks",
    )
    tick_direction = _normalize_tick_direction(tick_direction)
    ax.tick_params(
        axis="both",
        which="major",
        direction=tick_direction,
        colors="black",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction=tick_direction,
        colors="black",
    )
    for tick in (
        ax.xaxis.get_major_ticks()
        + ax.xaxis.get_minor_ticks()
        + ax.yaxis.get_major_ticks()
        + ax.yaxis.get_minor_ticks()
    ):
        tick.tick1line.set_zorder(10)
        tick.tick2line.set_zorder(10)
        tick.tick1line.set_color("black")
        tick.tick2line.set_color("black")


def plot_component_maps(
    components,
    *,
    file_path=None,
    save_fig=True,
    show=False,
    layout="auto",
    coordrange=None,
    factor4plot=1.0,
    percentile=99.0,
    symmetry=True,
    cmap="cmc.roma_r",
    figsize="single",
    figsize_aspect=None,
    figsize_height=None,
    dpi=300,
    style_context="science",
    fontsize=None,
    cell_style="points",
    edgewidth=0.1,
    edgecolor="black",
    alpha=1.0,
    markersize=10,
    faults=None,
    trace_color="black",
    trace_linewidth=0.5,
    axis_tick_direction="out",
    axis_max_major_ticks=5,
    axis_minor_ticks=False,
    axis_minor_subdivisions=2,
    colorbar_label="auto",
    colorbar_orientation="auto",
    colorbar_mode="outside",
    colorbar_loc=None,
    colorbar_size=None,
    colorbar_thickness=None,
    colorbar_pad=None,
    panel_pad=None,
    colorbar_tick_direction="out",
    colorbar_max_major_ticks=3,
    colorbar_minor_ticks=False,
    colorbar_minor_subdivisions=2,
    tickfontsize=None,
    labelfontsize=None,
    cb_label_loc=None,
    xlabel=None,
    ylabel=None,
):
    """Plot one or more raw/decimated component maps."""
    import matplotlib.pyplot as plt

    try:
        from eqtools.viztools import PlotStyle, set_degree_formatter
    except Exception:
        PlotStyle = None
        set_degree_formatter = None

    components = list(components)
    if not components:
        raise ValueError("At least one component is required for check plotting.")

    coordrange = _normalize_coordrange(coordrange)
    layout = "columns" if str(layout).lower() == "auto" and len(components) > 1 else layout
    layout = "single" if str(layout).lower() == "auto" else str(layout).replace("-", "_").lower()
    if layout not in ("single", "columns"):
        raise ValueError("check plot layout must be 'auto', 'single', or 'columns'.")
    if layout == "single" and len(components) != 1:
        raise ValueError("layout='single' requires exactly one component.")

    figsize = resolve_figsize(figsize, aspect=figsize_aspect, height=figsize_height)
    fontsize, tickfontsize, labelfontsize = _resolve_plot_font_sizes(
        figsize,
        fontsize=fontsize,
        tickfontsize=tickfontsize,
        labelfontsize=labelfontsize,
    )
    context = (
        PlotStyle(style_context, fontsize=fontsize, dpi=dpi)
        if PlotStyle and style_context is not None
        else None
    )

    if context is None:
        context = plt.rc_context(
            {
                "font.size": fontsize,
                "axes.labelsize": fontsize,
                "xtick.labelsize": max(fontsize - 1.0, 6.0),
                "ytick.labelsize": max(fontsize - 1.0, 6.0),
                "savefig.dpi": float(dpi),
            }
        )

    with context:
        subplot_kwargs = {"constrained_layout": True}
        resolved_colorbar_orientation = _resolve_component_colorbar_orientation(
            colorbar_orientation,
            layout,
            len(components),
        )
        column_vertical_colorbar = _needs_column_colorbar_spacing(
            layout,
            len(components),
            resolved_colorbar_orientation,
            colorbar_mode,
        )
        if column_vertical_colorbar:
            subplot_kwargs["gridspec_kw"] = {"wspace": 0.20}
        column_horizontal_colorbar = _needs_column_horizontal_colorbar_pad(
            layout,
            len(components),
            resolved_colorbar_orientation,
            colorbar_mode,
        )
        effective_colorbar_pad = colorbar_pad
        if column_horizontal_colorbar and effective_colorbar_pad is None:
            effective_colorbar_pad = 0.02

        if layout == "single":
            fig, axes = plt.subplots(
                1,
                1,
                figsize=figsize,
                squeeze=False,
                **subplot_kwargs,
            )
            axes = axes.ravel()
        else:
            fig, axes = plt.subplots(
                1,
                len(components),
                figsize=figsize,
                squeeze=False,
                **subplot_kwargs,
            )
            axes = axes.ravel()

        mappables = []
        for ax, component in zip(axes, components):
            scale_values = _visible_values(component, coordrange, factor4plot)
            vmin, vmax = robust_limits(
                scale_values,
                vmin=component.vmin,
                vmax=component.vmax,
                percentile=percentile,
                symmetry=symmetry,
            )
            mappable = _draw_component(
                ax,
                component,
                cell_style=cell_style,
                coordrange=coordrange,
                factor4plot=factor4plot,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                edgewidth=edgewidth,
                edgecolor=edgecolor,
                alpha=alpha,
                markersize=markersize,
            )
            mappables.append(mappable)
            _plot_faults(ax, faults, trace_color=trace_color, trace_linewidth=trace_linewidth)
            if coordrange is not None:
                ax.set_xlim(coordrange[0], coordrange[1])
                ax.set_ylim(coordrange[2], coordrange[3])
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.set_aspect("equal", adjustable="box")
            if layout == "columns":
                _anchor_column_axes(axes)
            if set_degree_formatter is not None:
                set_degree_formatter(ax, axis="both")
            _style_map_axis_ticks(
                ax,
                tick_direction=axis_tick_direction,
                max_major_ticks=axis_max_major_ticks,
                minor_ticks=axis_minor_ticks,
                minor_subdivisions=axis_minor_subdivisions,
            )
            if layout == "columns" and ax is not axes[0]:
                ax.tick_params(axis="y", labelleft=False)

        fig.canvas.draw()
        if column_vertical_colorbar or column_horizontal_colorbar:
            column_colorbar_layout = _resolve_colorbar_layout(
                resolved_colorbar_orientation,
                mode=colorbar_mode,
                loc=colorbar_loc,
                size=colorbar_size,
                thickness=colorbar_thickness,
                pad=effective_colorbar_pad,
            )
            labels = []
            for component in components:
                label = component.label
                if colorbar_label not in (None, "auto"):
                    label = colorbar_label
                label = _effective_colorbar_label(label, column_horizontal_colorbar)
                labels.append(label)
            if column_vertical_colorbar:
                _reserve_column_vertical_colorbar_space(
                    fig,
                    list(axes),
                    column_colorbar_layout,
                    tickfontsize=tickfontsize,
                    labelfontsize=labelfontsize,
                    labels=labels,
                    panel_pad=panel_pad,
                )
            else:
                _reserve_column_horizontal_colorbar_space(
                    fig,
                    list(axes),
                    column_colorbar_layout,
                    tickfontsize=tickfontsize,
                    labelfontsize=labelfontsize,
                    labels=labels,
                    panel_pad=panel_pad,
                )
        for ax, component, mappable in zip(axes, components, mappables):
            label = component.label
            if colorbar_label not in (None, "auto"):
                label = colorbar_label
            _add_panel_colorbar(
                fig,
                ax,
                mappable,
                orientation=resolved_colorbar_orientation,
                mode=colorbar_mode,
                loc=colorbar_loc,
                size=colorbar_size,
                thickness=colorbar_thickness,
                pad=effective_colorbar_pad,
                label=label,
                tickfontsize=tickfontsize,
                labelfontsize=labelfontsize,
                cb_label_loc=cb_label_loc,
                tick_direction=colorbar_tick_direction,
                max_major_ticks=colorbar_max_major_ticks,
                minor_ticks=colorbar_minor_ticks,
                minor_subdivisions=colorbar_minor_subdivisions,
                compact_label=column_horizontal_colorbar,
                reserve_outside_space=not (column_vertical_colorbar or column_horizontal_colorbar),
                keep_inside_canvas=not (column_vertical_colorbar or column_horizontal_colorbar),
            )

        _freeze_manual_layout(fig)
        if save_fig and file_path:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
        if show:
            _cap_interactive_dpi(fig)
            plt.show()
        else:
            plt.close(fig)

    return fig, axes
