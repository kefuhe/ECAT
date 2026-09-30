"""Render a GPS fit as paired EN arrows and concentric signed-U markers.

This module owns display semantics, not inversion semantics. The caller supplies
already prepared station-by-component arrays and an explicit active-U flag.
There is no forward calculation, model activation, observation filtering in the
source object, or publication of synthetic fields here.

The rendering sequence is input validation, projected positions/directions,
paired finite masks, shared display scales, then Matplotlib artists. In
particular, map coordinates (km/degrees), displacement units, inches, and marker
areas (points squared) remain separate throughout the calculation.
"""

import warnings

import numpy as np


def _positive(value, name):
    """Validate one scalar display magnitude before creating any figure."""
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _station_values(values, count, name, columns):
    """Read station-by-component values without changing the source layout.

    Some historical readers squeeze one station to shape (3,). Only that
    unambiguous case is normalized locally; multi-station shape errors fail.
    """
    values = np.asarray(values, dtype=float)
    if count == 1 and values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] != count or values.shape[1] < columns:
        raise ValueError(f"{name} must have {count} station rows and at least {columns} columns")
    return values


def _project(data, lon, lat):
    """Use the data object's coordinate frame, returning flat km arrays."""
    lon = np.asarray(lon, dtype=float).reshape(-1)
    lat = np.asarray(lat, dtype=float).reshape(-1)
    # CSI ll2xy requires scalar/NumPy inputs; pyproj's scalar fast path must
    # receive scalars for one station, not a one-element NumPy array.
    if lon.size == 1:
        inputs = (float(lon[0]), float(lat[0]))
    else:
        inputs = (lon, lat)
    projected_x, projected_y = data.ll2xy(*inputs)
    return (np.asarray(projected_x, dtype=float).reshape(-1),
            np.asarray(projected_y, dtype=float).reshape(-1))


def _positions_and_basis(data, lon, lat, coordinates):
    """Return positions and EN basis directions in the displayed plane.

    Projected basis vectors are one-metre geodesic differences, normalized
    individually. Quiver lengths use the original EN magnitude, independently
    of projection scale and the map's coordinate units.
    """
    # Each station's 2x2 basis has projected East and North as its columns.
    # Geographic axes use EN as screen directions; their aspect below accounts
    # for the local longitude/latitude distance ratio.
    count = lon.size
    basis = np.broadcast_to(np.eye(2), (count, 2, 2)).copy()
    if coordinates == "lonlat":
        finite = np.isfinite(lon) & np.isfinite(lat)
        if not finite.any():
            raise ValueError("GPS has no finite station coordinates")
        if np.ptp(lon[finite]) > 180 or np.max(np.abs(lat[finite])) >= 89:
            raise ValueError("lonlat comparison requires a regional, non-polar extent; use xy")
        return lon, lat, basis
    if coordinates != "xy":
        raise ValueError("coordinates must be 'xy' or 'lonlat'")
    x, y = _project(data, lon, lat)
    # The existing CSI geodesic/projector determines the orientation. Do not
    # assume that true North equals UTM grid North away from the central meridian.
    for index, azimuth in enumerate((90.0, 0.0)):
        lo, la, _ = data.geod.fwd(lon.tolist(), lat.tolist(), [azimuth] * count, [1.0] * count)
        xp, yp = _project(data, lo, la)
        direction = np.column_stack((np.asarray(xp).reshape(-1) - x,
                                     np.asarray(yp).reshape(-1) - y))
        length = np.linalg.norm(direction, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            basis[:, :, index] = direction / length[:, None]
    return x, y, basis


def _display_vectors(values, basis):
    """Project EN direction, then restore the physical horizontal magnitude.

    Map projection determines orientation only. Arrow length must express
    sqrt(E**2 + N**2), not map-coordinate displacement or projection scale.
    Zero displacement remains zero instead of becoming an arbitrary direction.
    """
    projected = np.einsum("nij,nj->ni", basis, values)
    length = np.linalg.norm(projected, axis=1)
    magnitude = np.linalg.norm(values, axis=1)
    result = np.zeros_like(projected)
    np.divide(projected * magnitude[:, None], length[:, None],
              out=result, where=length[:, None] > 0)
    return result


def _set_geographic_formatters(ax, remove_direction_labels):
    """Use the shared public degree formatters without losing local precision.

    Choose precision from the actual major ticks, not the absolute longitude.
    The public hemisphere formatters otherwise default to whole degrees, which
    would collapse neighboring labels on a small regional map.
    """
    from ecat_viz import DegreeFormatter, LatFormatter, LonFormatter

    for axis, formatter in ((ax.xaxis, LonFormatter), (ax.yaxis, LatFormatter)):
        if remove_direction_labels:
            degrees = DegreeFormatter(useOffset=False)
            degrees.set_scientific(False)
            axis.set_major_formatter(degrees)
            continue
        ticks = np.unique(axis.get_majorticklocs())
        ticks = ticks[np.isfinite(ticks)]
        intervals = np.diff(ticks)
        step = intervals[intervals > 0].min(initial=1.0)
        places = 0
        while places < 10 and not np.allclose(
            ticks, np.round(ticks, places), rtol=0, atol=step * 1e-6,
        ):
            places += 1
        axis.set_major_formatter(formatter(decimal_places=places))


def _add_comparison_legend(ax, *, colors, length_inches, scale_label,
                           vertical_sizes, loc):
    """Draw one ordinary legend with physically calibrated EN color samples.

    A single private Matplotlib handler lays out the shared magnitude above
    two equal colored bars, with U role samples below when U is displayed.
    DrawingArea dimensions are points: one inch is 72 points regardless of
    figure DPI, map units or legend font. Native Legend still owns placement,
    including loc='best'; no callback or separate arrow-key artist is needed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from matplotlib.textpath import TextPath

    font = FontProperties(size=plt.rcParams["legend.fontsize"])
    fontsize = font.get_size_in_points()
    rows = []
    if length_inches is not None:
        rows.extend((label, color, None) for label, color in
                    zip(("Observed EN", "Model EN"), colors))
    if vertical_sizes is not None:
        rows.extend((label, "0.5", np.sqrt(area)) for label, area in
                    zip(("Observed U (outer)", "Model U (inner)"), vertical_sizes))

    def text_width(text):
        return TextPath((0, 0), text, prop=font).get_extents().width

    bar_length = 0.0 if length_inches is None else 72.0 * length_inches
    marker_size = max((size or 0.0 for _, _, size in rows), default=0.0)
    sample_width = max(bar_length, marker_size,
                       text_width(scale_label) if scale_label else 0.0)
    gap = 0.6 * fontsize
    total_width = sample_width + gap + max(text_width(label) for label, _, _ in rows) + 0.2 * fontsize
    row_height = max(1.5 * fontsize, marker_size + 2.0)
    total_height = row_height * (len(rows) + bool(scale_label))

    class ComparisonHandler(HandlerBase):
        """Populate the native legend's handle box; all sizes are in points."""

        def legend_artist(self, legend, original, fontsize, handlebox):
            # The supported handler hook runs before Legend packs its rows.
            # Size this one composite entry to its content instead of using
            # the ordinary, font-relative fixed-length line sample.
            handlebox.width = total_width
            handlebox.height = total_height
            handlebox.xdescent = handlebox.ydescent = 0.0
            transform = handlebox.get_transform()
            center = sample_width / 2.0
            y = total_height - row_height / 2.0
            artists = []
            if scale_label:
                artists.append(Text(center, y, scale_label, ha="center", va="center",
                                    fontproperties=font, transform=transform))
                y -= row_height
            for label, color, size in rows:
                if size is None:
                    sample = Line2D([center - bar_length / 2.0, center + bar_length / 2.0],
                                    [y, y], color=color, linewidth=1.5,
                                    solid_capstyle="butt", label=label, transform=transform)
                else:
                    sample = Line2D([center], [y], linestyle="none", marker="o",
                                    color=color, markeredgecolor="0.25", markeredgewidth=0.4,
                                    markersize=size, label=label, transform=transform)
                artists.extend((sample, Text(sample_width + gap, y, label, va="center",
                                             fontproperties=font, transform=transform)))
                y -= row_height
            for artist in artists:
                handlebox.add_artist(artist)
            return artists[0]

    token = object()
    return ax.legend(handles=[token], labels=[""], loc=loc, prop=font,
                     handler_map={token: ComparisonHandler()}, handletextpad=0)


def plot_gps_fit_comparison(
    data, *, vertical=False, coordinates="lonlat", extent=None,
    figsize="single", unit="inch", ax=None,
    value_scale=1.0, value_unit="", arrow_scale=None, legend_value=None,
    color=("#e33e1c", "#2e5b99"), width=0.005,
    headwidth=3, headlength=5, headaxislength=4.5,
    vertical_sizes=(64, 25), vertical_cmap="RdBu_r",
    vertical_vmin=None, vertical_vmax=None, cbaxis=None,
    colorbar_orientation="vertical", colorbar_size=0.35,
    name=False, title=True, error=False,
    legend_loc="best", key_position=None,
    faults=None, fault_color="k", fault_linewidth=0.8,
    xticks=None, yticks=None, remove_direction_labels=False,
    xlabel=None, ylabel=None,
    style="science", style_kwargs=None,
    save_path=None, show=True, close=False, dpi=300,
):
    """Render prepared GPS fields and return their ordinary ``(fig, ax)``.

    Parameters are exposed and documented by ``gps.plot_fit_comparison``.
    ``vertical`` is the caller's component contract, not a guess from finite U
    or from the number of storage columns. Both roles use one horizontal scale
    and identical arrow geometry. The larger observed U marker is drawn first;
    the smaller modeled marker leaves an observed ring visible underneath.

    This function only allocates display arrays. It never assigns to ``data``.
    A borrowed axes belongs to its caller and must not be closed by this helper.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, to_rgba
    from matplotlib.patches import Ellipse
    from matplotlib.transforms import Affine2D, ScaledTranslation
    from ecat_viz import PlotStyle, finish_fig, get_cmap, publication_figsize

    # 1. Validate display controls before figures or global styles are changed.
    if not isinstance(vertical, (bool, np.bool_)):
        raise ValueError("vertical must be an explicit boolean")
    if ax is not None and close:
        raise ValueError("close=True cannot close a borrowed ax")
    value_scale = _positive(value_scale, "value_scale")
    width = _positive(width, "width")
    for label, value in (("headwidth", headwidth), ("headlength", headlength),
                         ("headaxislength", headaxislength), ("dpi", dpi),
                         ("fault_linewidth", fault_linewidth)):
        _positive(value, label)
    if len(color) != 2:
        raise ValueError("color must contain observed and model colors")
    for item in color:
        to_rgba(item)
    sizes = np.asarray(vertical_sizes, dtype=float)
    if sizes.shape != (2,) or not np.isfinite(sizes).all() or not (sizes[0] > sizes[1] > 0):
        raise ValueError("vertical_sizes must be two positive areas: observed > model")
    if extent is not None:
        extent = np.asarray(extent, dtype=float)
        if (extent.shape != (4,) or not np.isfinite(extent).all()
                or extent[0] >= extent[1] or extent[2] >= extent[3]):
            raise ValueError("extent must be finite [xmin, xmax, ymin, ymax] with increasing bounds")
    if cbaxis is not None:
        cbaxis = np.asarray(cbaxis, dtype=float)
        if cbaxis.shape != (4,) or not np.isfinite(cbaxis).all() or np.any(cbaxis[2:] <= 0):
            raise ValueError("cbaxis must be [left, bottom, positive width, positive height]")
    if colorbar_orientation not in ("vertical", "horizontal"):
        raise ValueError("colorbar_orientation must be 'vertical' or 'horizontal'")
    colorbar_size = _positive(colorbar_size, "colorbar_size")
    if colorbar_size > 1:
        raise ValueError("colorbar_size must be <= 1 (fraction of the main axes)")
    if key_position is not None:
        old_position = np.asarray(key_position, dtype=float)
        if old_position.shape != (2,) or not np.isfinite(old_position).all():
            raise ValueError("key_position must be a finite (x, y) in axes fractions")
        warnings.warn("key_position is deprecated and no longer positions a separate "
                      "arrow key; use legend_loc for the combined calibrated legend",
                      FutureWarning, stacklevel=2)
    # 2. Read prepared arrays in station order. Display conversion is local;
    # self.factor from file reading cannot describe later manual conversions.
    final_figsize = publication_figsize(column=figsize, unit=unit, aspect=0.8)
    lon = np.asarray(data.lon, dtype=float).reshape(-1)
    lat = np.asarray(data.lat, dtype=float).reshape(-1)
    if not lon.size or lat.shape != lon.shape:
        raise ValueError("GPS longitude and latitude must have matching non-empty station rows")
    count = lon.size
    columns = 3 if vertical else 2
    observed = _station_values(
        data.vel_enu, count, "GPS observations", columns,
    ) * value_scale
    if getattr(data, "synth", None) is None:
        raise ValueError("GPS comparison requires an already prepared synth")
    modeled = _station_values(
        data.synth, count, "GPS synth", columns,
    ) * value_scale
    x, y, basis = _positions_and_basis(data, lon, lat, coordinates)
    # 3. Each component group uses one mask for both roles. Missing data are
    # not zero displacement, and drawing a model at an unobserved row is not a fit.
    finite_positions = (
        np.isfinite(x) & np.isfinite(y)
        & np.isfinite(basis).all(axis=(1, 2))
    )
    horizontal = (
        finite_positions
        & np.isfinite(observed[:, :2]).all(axis=1)
        & np.isfinite(modeled[:, :2]).all(axis=1)
    )
    if vertical:
        up = (
            finite_positions
            & np.isfinite(observed[:, 2])
            & np.isfinite(modeled[:, 2])
        )
    else:
        up = np.zeros(count, dtype=bool)
    if not horizontal.any() and not up.any():
        raise ValueError("GPS comparison has no finite observed/model station pairs")
    if not horizontal.all():
        warnings.warn(f"GPS comparison omitted {count - horizontal.sum()} unpaired horizontal station(s)", UserWarning, stacklevel=2)
    if vertical and not up.all():
        warnings.warn(f"GPS comparison omitted {count - up.sum()} unpaired vertical station(s)", UserWarning, stacklevel=2)
    # 4. Use one physical-to-paper scale for observed and modeled arrows.
    # scale_units='inches' makes the scale independent of km/degree map units.
    vectors = [
        _display_vectors(values[horizontal, :2], basis[horizontal])
        for values in (observed, modeled)
    ]
    if arrow_scale is None:
        largest = max(
            np.linalg.norm(vector, axis=1).max(initial=0.0)
            for vector in vectors
        )
        paper_width = ax.figure.get_figwidth() if ax is not None else final_figsize[0]
        arrow_scale = (largest if largest > 0 else 1.0) / (0.1 * paper_width)
    arrow_scale = _positive(arrow_scale, "arrow_scale")
    if legend_value is None:
        # Default calibrated color-bar length is 0.2 inch, in display units.
        legend_value = 0.2 * arrow_scale
    legend_value = _positive(legend_value, "legend_value")
    uncertainty = None
    if error:
        uncertainty = _station_values(data.err_enu, count, "GPS err_enu", 2) * value_scale
        if not np.isfinite(uncertainty[horizontal, :2]).all() or np.any(uncertainty[horizontal, :2] < 0):
            raise ValueError("horizontal err_enu must be finite and non-negative")
    # 5. U color means signed displacement, while marker area means role.
    # A single normalization includes both roles; explicit bounds always win.
    cmap = get_cmap(vertical_cmap)
    clim = None
    if vertical and up.any():
        values = np.concatenate((observed[up, 2], modeled[up, 2]))
        bound = np.max(np.abs(values)) or 1.0
        lo = -bound if vertical_vmin is None else float(vertical_vmin)
        hi = bound if vertical_vmax is None else float(vertical_vmax)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError("vertical color limits must be finite and increasing")
        clim = (lo, hi)
    elif vertical_vmin is not None or vertical_vmax is not None:
        # Still diagnose malformed explicit input when U is hidden/unavailable.
        for value in (vertical_vmin, vertical_vmax):
            if value is not None and not np.isfinite(float(value)):
                raise ValueError("vertical color limits must be finite")
        if vertical_vmin is not None and vertical_vmax is not None and vertical_vmin >= vertical_vmax:
            raise ValueError("vertical color limits must be increasing")
    # A small default extent also handles a single station or aligned stations.
    # Explicit extents are already in the chosen km/degree coordinate system.
    automatic_extent = extent is None
    if automatic_extent:
        active = horizontal | up
        bounds = [np.min(x[active]), np.max(x[active]), np.min(y[active]), np.max(y[active])]
        minimum = 1.0 if coordinates == "xy" else 0.01
        dx = max(bounds[1] - bounds[0], minimum) * 0.2
        dy = max(bounds[3] - bounds[2], minimum) * 0.2
        extent = (bounds[0] - dx, bounds[1] + dx, bounds[2] - dy, bounds[3] + dy)
    station_names = None
    if name:
        station_names = np.asarray(data.station).reshape(-1)
        if station_names.size != count:
            raise ValueError("GPS station names must match station rows")
    # Fault positions must use the GPS object's frame, even if fault objects
    # were constructed with another projection origin. Never reuse fault.x/y.
    traces = []
    if faults is not None:
        faults = faults if isinstance(faults, (list, tuple)) else [faults]
        for fault in faults:
            lo, la = getattr(fault, "lon", None), getattr(fault, "lat", None)
            if lo is not None and la is not None:
                tx, ty = _project(data, lo, la) if coordinates == "xy" else (lo, la)
                traces.append((tx, ty))

    # 6. Create only the requested 2-D axes; style is restored on all exits.
    with PlotStyle(style, **{"usetex": False, **dict(style_kwargs or {})}):
        if ax is None:
            fig, ax = plt.subplots(figsize=final_figsize, constrained_layout=True)
        else:
            fig = ax.figure
        for tx, ty in traces:
            ax.plot(tx, ty, color=fault_color, linewidth=fault_linewidth, zorder=1)
        # Draw observed U first. Model U overlays only the center, so the
        # observed ring and model disk remain distinguishable at the same station.
        if clim is not None:
            norm = Normalize(*clim)
            for index, values in enumerate((observed, modeled)):
                scatter = ax.scatter(x[up], y[up], s=sizes[index], c=values[up, 2],
                                     cmap=cmap, norm=norm, edgecolors="0.25",
                                     linewidths=0.4, alpha=1.0, zorder=2 + index)
            values = np.concatenate((observed[up, 2], modeled[up, 2]))
            below, above = np.any(values < clim[0]), np.any(values > clim[1])
            if below and above:
                extend = "both"
            elif below:
                extend = "min"
            elif above:
                extend = "max"
            else:
                extend = "neither"
            # Attach to the final main-axes rectangle, rather than shrinking
            # a pre-aspect subplot box. Right-side U starts on the bottom spine;
            # an explicit cbaxis owns geometry in either orientation.
            if cbaxis is None:
                bounds = ([1.04, 0.0, 0.025, colorbar_size]
                          if colorbar_orientation == "vertical" else
                          [(1.0 - colorbar_size) / 2.0, -0.15, colorbar_size, 0.025])
            else:
                bounds = cbaxis
            cax = ax.inset_axes(bounds, transform=ax.transAxes)
            cbar = fig.colorbar(scatter, cax=cax, orientation=colorbar_orientation,
                                extend=extend)
            # A short colorbar needs only a few readable major ticks.
            from matplotlib.ticker import MaxNLocator
            cbar.locator = MaxNLocator(nbins=3)
            cbar.update_ticks()
            cbar.minorticks_off()
            cbar.set_label(f"Up (+){' (' + str(value_unit) + ')' if value_unit else ''}")
        # Arrow geometry is shared. Role is encoded by red/blue color only;
        # neither thickness nor position is changed to separate the two roles.
        for index, (vector, c) in enumerate(zip(vectors, color)):
            ax.quiver(x[horizontal], y[horizontal], vector[:, 0], vector[:, 1],
                          angles="uv", scale_units="inches", scale=arrow_scale,
                          color=c, width=width, headwidth=headwidth, headlength=headlength,
                          headaxislength=headaxislength, minlength=0, zorder=4 + index)
        # Optional ellipses use marginal err_enu, not the solver's covariance.
        # Their inch-based transform follows the observed tip when axes resize.
        if uncertainty is not None:
            for index, row in enumerate(np.flatnonzero(horizontal)):
                b = basis[row]
                u, v = vectors[0][index] / arrow_scale
                transform = (Affine2D.from_values(b[0, 0], b[1, 0], b[0, 1], b[1, 1], 0, 0)
                             + fig.dpi_scale_trans + ScaledTranslation(x[row], y[row], ax.transData)
                             + ScaledTranslation(u, v, fig.dpi_scale_trans))
                ellipse = Ellipse((0, 0), width=2 * uncertainty[row, 0] / arrow_scale,
                                  height=2 * uncertainty[row, 1] / arrow_scale,
                                  fill=False, edgecolor=color[0], linewidth=0.5, transform=transform, zorder=6)
                ax.add_patch(ellipse)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        if coordinates == "xy":
            ax.set_aspect("equal")
            default_labels = ("Easting (km)", "Northing (km)")
        else:
            latitude = np.median(lat[finite_positions])
            ax.set_aspect(1 / np.cos(np.deg2rad(latitude)))
            # This geographic view is regional, not a general cartographic
            # projection. EN arrow directions stay in screen coordinates.
            default_labels = ("", "")
        ax.set_xlabel(default_labels[0] if xlabel is None else xlabel,
                      fontsize=plt.rcParams["axes.labelsize"])
        ax.set_ylabel(default_labels[1] if ylabel is None else ylabel,
                      fontsize=plt.rcParams["axes.labelsize"])
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        if station_names is not None:
            for row in np.flatnonzero(horizontal | up):
                ax.annotate(str(station_names[row]), (x[row], y[row]),
                            xytext=(4, 4), textcoords="offset points", zorder=7)
        if title:
            ax.set_title(str(data.name) if title is True else str(title))
        if coordinates == "lonlat":
            _set_geographic_formatters(ax, remove_direction_labels)
        _add_comparison_legend(
            ax, colors=color,
            length_inches=legend_value / arrow_scale if horizontal.any() else None,
            scale_label=(f"{legend_value:g}{' ' + str(value_unit) if value_unit else ''}"
                         if horizontal.any() else None),
            vertical_sizes=sizes if clim is not None else None, loc=legend_loc,
        )
        if automatic_extent and horizontal.any():
            # Paper-sized arrows can extend past station-only map bounds.
            # Resolve the real axes rectangle after aspect/colorbar layout, then
            # grow automatic bounds to include both sets of arrow tips. Explicit
            # user extents remain authoritative. A few bounded layout passes
            # cover the small changes caused by the expanding projected range.
            for _ in range(3):
                fig.canvas.draw()
                station_pixels = ax.transData.transform(
                    np.column_stack((x[horizontal], y[horizontal]))
                )
                tip_pixels = np.concatenate([
                    station_pixels + vector / arrow_scale * fig.dpi
                    for vector in vectors
                ])
                tips = ax.transData.inverted().transform(tip_pixels)
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                xmargin = (xmax - xmin) * 0.04
                ymargin = (ymax - ymin) * 0.04
                expanded = (
                    min(xmin, tips[:, 0].min() - xmargin),
                    max(xmax, tips[:, 0].max() + xmargin),
                    min(ymin, tips[:, 1].min() - ymargin),
                    max(ymax, tips[:, 1].max() + ymargin),
                )
                if np.allclose(expanded, (xmin, xmax, ymin, ymax), rtol=0, atol=1e-10):
                    break
                ax.set_xlim(expanded[0], expanded[1])
                ax.set_ylim(expanded[2], expanded[3])
        if coordinates == "lonlat":
            _set_geographic_formatters(ax, remove_direction_labels)
        # 7. Save/show the explicit figure and keep legacy data.fig untouched.
        finish_fig(fig, save_path, show=show, close=close, dpi=dpi)
    return fig, ax
