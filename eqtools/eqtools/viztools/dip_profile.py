"""Read-only diagnostics for a resolved non-layered dip profile.

The scientific resolution remains owned by the fault's perturbation layer.
This module only draws the returned immutable ``ResolvedDipProfile`` so a
diagnostic can never invent a second projection, interpolation, or transition
rule.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from ._core import PlotStyle
from ._formatters import LatFormatter, LonFormatter
from ._style_utils import finish_fig


def plot_dip_profile_diagnostics(
    fault,
    perturbations=None,
    *,
    angle_unit="degrees",
    coordinates="xy",
    show_connectors=True,
    title=None,
    style="science",
    figsize=(8.0, 3.6),
    fontsize=8,
    dpi=300,
    rcparams=None,
    save=None,
    show=False,
    screen_dpi=200,
    close=False,
):
    """Plot declared/projected controls and the resolved 1-D dip profile.

    Parameters
    ----------
    fault : Bayesian non-layered triangular-fault object
        Must expose the read-only ``resolve_dip_profile`` method and a frozen
        ``geometry_ref.dip_profile``.
    perturbations : array-like, optional
        Candidate dip increments. Omission resolves the reference (zero)
        profile. The values are never written to ``fault``.
    angle_unit : {'degrees', 'radians'}, default 'degrees'
        Unit of ``perturbations``.
    coordinates : {'xy', 'lonlat'}, default 'xy'
        Coordinate frame for the map panel. The profile panel always uses the
        resolved one-dimensional coordinate.

    Returns
    -------
    fig, axes
        Matplotlib figure and ``{'map': ..., 'profile': ...}`` axes mapping.

    Notes
    -----
    Hollow symbols are user-declared locations, filled symbols are their
    projections onto the frozen top edge, ``S``/``F`` denote sampled/fixed
    controls, and orange bands are transition intervals. Top-edge segment
    colour shows the resolved dip used by bottom generation. ``START`` and
    ``END`` expose the top ordering that defines ``s_km`` and
    ``s_from_end_km``; control labels report resolved distance from START.
    """
    if isinstance(save, bool):
        raise ValueError("save must be a file path or None, not a boolean")
    if coordinates not in {"xy", "lonlat"}:
        raise ValueError("coordinates must be 'xy' or 'lonlat'")
    resolver = getattr(fault, "resolve_dip_profile", None)
    if resolver is None:
        raise TypeError("fault does not expose resolve_dip_profile()")

    resolved = resolver(perturbations, angle_unit=angle_unit)
    top_xy = resolved.top_xy
    raw_xy = resolved.raw_control_xy
    projected_xy = resolved.projected_control_xy
    zone_paths = [
        _polyline_interval(
            top_xy,
            resolved.top_s,
            zone.lower_s,
            zone.upper_s,
            zone.lower_xy,
            zone.upper_xy,
        )
        for zone in resolved.transition_zones
    ]
    zone_raw = [zone.raw_anchor_xy for zone in resolved.transition_zones]
    zone_projected = [
        zone.projected_anchor_xy for zone in resolved.transition_zones
    ]
    if coordinates == "lonlat":
        top_plot = _xy_to_lonlat(fault, top_xy)
        raw_plot = _xy_to_lonlat(fault, raw_xy)
        projected_plot = _xy_to_lonlat(fault, projected_xy)
        zone_paths = [_xy_to_lonlat(fault, path) for path in zone_paths]
        zone_raw = [_xy_to_lonlat(fault, points) for points in zone_raw]
        zone_projected = [
            _xy_to_lonlat(fault, points) for points in zone_projected
        ]
    else:
        top_plot, raw_plot, projected_plot = top_xy, raw_xy, projected_xy

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    figure = None
    axes = {}
    try:
        context = (
            PlotStyle(
                style,
                figsize=figsize,
                fontsize=fontsize,
                dpi=dpi,
                rcparams=rcparams,
            )
            if style is not None
            else nullcontext()
        )
        with context:
            figure, (map_axis, profile_axis) = plt.subplots(
                1,
                2,
                constrained_layout=True,
            )
            axes = {"map": map_axis, "profile": profile_axis}

            segments = np.stack([top_plot[:-1], top_plot[1:]], axis=1)
            segment_dip = 0.5 * (
                resolved.top_dip_continuous[:-1]
                + resolved.top_dip_continuous[1:]
            )
            collection = LineCollection(
                segments,
                cmap="viridis",
                linewidths=2.0,
            )
            collection.set_array(segment_dip)
            map_axis.add_collection(collection)
            map_axis.autoscale()

            map_axis.annotate(
                "",
                xy=top_plot[1],
                xytext=top_plot[0],
                arrowprops={"arrowstyle": "->", "color": "0.2", "lw": 1.0},
                zorder=5,
            )
            map_axis.annotate(
                "START\ns=0 km",
                top_plot[0],
                xytext=(4, -4),
                textcoords="offset points",
                ha="left",
                va="top",
            )
            map_axis.annotate(
                f"END\ns={resolved.top_s[-1]:.3g} km",
                top_plot[-1],
                xytext=(-4, -4),
                textcoords="offset points",
                ha="right",
                va="top",
            )

            for path in zone_paths:
                map_axis.plot(
                    path[:, 0],
                    path[:, 1],
                    color="#E69F00",
                    linewidth=4.0,
                    alpha=0.55,
                    solid_capstyle="round",
                    zorder=2,
                )

            if show_connectors:
                for raw, projected in zip(raw_plot, projected_plot):
                    map_axis.plot(
                        [raw[0], projected[0]],
                        [raw[1], projected[1]],
                        color="0.55",
                        linestyle="--",
                        linewidth=0.8,
                        zorder=1,
                    )
                for raw_points, projected_points in zip(
                    zone_raw,
                    zone_projected,
                ):
                    for raw, projected in zip(raw_points, projected_points):
                        map_axis.plot(
                            [raw[0], projected[0]],
                            [raw[1], projected[1]],
                            color="#E69F00",
                            linestyle=":",
                            linewidth=0.9,
                            zorder=1,
                        )

            for sampled, marker, colour, label in (
                (True, "o", "#0072B2", "sampled control"),
                (False, "s", "#D55E00", "fixed control"),
            ):
                mask = resolved.sampled == sampled
                if not np.any(mask):
                    continue
                map_axis.scatter(
                    raw_plot[mask, 0],
                    raw_plot[mask, 1],
                    marker=marker,
                    facecolors="none",
                    edgecolors=colour,
                    linewidths=1.2,
                    label=f"declared {label}",
                    zorder=3,
                )
                map_axis.scatter(
                    projected_plot[mask, 0],
                    projected_plot[mask, 1],
                    marker=marker,
                    facecolors=colour,
                    edgecolors="white",
                    linewidths=0.5,
                    label=f"projected {label}",
                    zorder=4,
                )
                role = "S" if sampled else "F"
                indices = np.flatnonzero(mask)
                for control_index in indices:
                    map_axis.annotate(
                        f"{role}{control_index}\ns={resolved.control_s[control_index]:.3g} km",
                        projected_plot[control_index],
                        xytext=(3, 3),
                        textcoords="offset points",
                    )

            if zone_raw:
                raw_points = np.vstack(zone_raw)
                projected_points = np.vstack(zone_projected)
                map_axis.scatter(
                    raw_points[:, 0],
                    raw_points[:, 1],
                    marker="D",
                    facecolors="none",
                    edgecolors="#E69F00",
                    linewidths=1.2,
                    label="declared transition anchor",
                    zorder=3,
                )
                map_axis.scatter(
                    projected_points[:, 0],
                    projected_points[:, 1],
                    marker="D",
                    facecolors="#E69F00",
                    edgecolors="white",
                    linewidths=0.5,
                    label="projected transition anchor",
                    zorder=4,
                )

            map_axis.set_aspect("equal", adjustable="datalim")
            if coordinates == "lonlat":
                map_axis.set_xlabel("Longitude")
                map_axis.set_ylabel("Latitude")
                map_axis.xaxis.set_major_formatter(LonFormatter())
                map_axis.yaxis.set_major_formatter(LatFormatter())
            else:
                map_axis.set_xlabel("x (km)")
                map_axis.set_ylabel("y (km)")
            map_axis.set_title("Control projection and transition zones")
            map_axis.legend(loc="best", fontsize="small")

            profile_axis.plot(
                resolved.top_u,
                resolved.top_dip_continuous,
                color="0.15",
                linewidth=1.5,
                label="resolved dip",
            )
            for zone in resolved.transition_zones:
                profile_axis.axvspan(
                    zone.lower_u,
                    zone.upper_u,
                    color="#E69F00",
                    alpha=0.2,
                )
            for sampled, marker, colour, label in (
                (True, "o", "#0072B2", "sampled"),
                (False, "s", "#D55E00", "fixed"),
            ):
                mask = resolved.sampled == sampled
                if np.any(mask):
                    profile_axis.scatter(
                        resolved.control_u[mask],
                        resolved.control_dip[mask],
                        marker=marker,
                        color=colour,
                        label=label,
                        zorder=3,
                    )
            unit = "km" if resolved.interpolation_axis == "arc_length" else "km"
            profile_axis.set_xlabel(
                f"u ({resolved.interpolation_axis}, {unit})"
            )
            profile_axis.set_ylabel("Dip (degree, continuous 0–180)")
            profile_axis.set_title("Resolved one-dimensional profile")
            profile_axis.legend(loc="best", fontsize="small")

            if title is None:
                title = f"{getattr(fault, 'name', 'Fault')}: dip-profile diagnostics"
            if title:
                figure.suptitle(title)

            if save is not None or show:
                finish_fig(
                    figure,
                    save,
                    save=save is not None,
                    show=show,
                    dpi=dpi,
                    screen_dpi=screen_dpi,
                    close=close,
                )
            elif close:
                plt.close(figure)
        return figure, axes
    except Exception:
        if figure is not None:
            plt.close(figure)
        raise


def plot_dip_transition_analysis(
    fault,
    analysis,
    suggestion=None,
    *,
    coordinates="xy",
    title=None,
    style="science",
    figsize=(8.0, 3.6),
    fontsize=8,
    dpi=300,
    rcparams=None,
    save=None,
    show=False,
    screen_dpi=200,
    close=False,
):
    """Plot a frozen-top curvature preflight and optional transition interval.

    The function consumes :class:`TopCurvatureAnalysis` without recomputing
    curvature.  It validates the result against ``fault.geometry_ref`` so a
    report from an older or differently ordered top cannot be plotted or
    accepted silently.

    Returns
    -------
    fig, axes
        Matplotlib figure and ``{'map': ..., 'curvature': ...}`` axes mapping.
    """
    if isinstance(save, bool):
        raise ValueError("save must be a file path or None, not a boolean")
    if coordinates not in {"xy", "lonlat"}:
        raise ValueError("coordinates must be 'xy' or 'lonlat'")
    geometry_ref = getattr(fault, "geometry_ref", None)
    if geometry_ref is None or geometry_ref.top_coords is None:
        raise ValueError("fault has no frozen reference top")
    analysis.validate_reference(geometry_ref.top_coords)
    if suggestion is not None:
        suggestion.validate_reference(geometry_ref.top_coords)

    map_xy = analysis.sample_xy
    map_plot = _xy_to_lonlat(fault, map_xy) if coordinates == "lonlat" else map_xy

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    figure = None
    axes = {}
    try:
        context = (
            PlotStyle(
                style,
                figsize=figsize,
                fontsize=fontsize,
                dpi=dpi,
                rcparams=rcparams,
            )
            if style is not None
            else nullcontext()
        )
        with context:
            figure, (map_axis, curvature_axis) = plt.subplots(
                1,
                2,
                constrained_layout=True,
            )
            axes = {"map": map_axis, "curvature": curvature_axis}

            segments = np.stack([map_plot[:-1], map_plot[1:]], axis=1)
            segment_curvature = 0.5 * (
                analysis.normalized_abs_curvature[:-1]
                + analysis.normalized_abs_curvature[1:]
            )
            collection = LineCollection(
                segments,
                cmap="magma",
                linewidths=2.2,
            )
            collection.set_array(segment_curvature)
            collection.set_clim(0.0, 1.0)
            map_axis.add_collection(collection)
            map_axis.autoscale()
            colorbar = figure.colorbar(collection, ax=map_axis, pad=0.02)
            colorbar.set_label("Normalized |curvature|")

            peak_plot = map_plot[analysis.peak_indices]
            if peak_plot.size:
                map_axis.scatter(
                    peak_plot[:, 0],
                    peak_plot[:, 1],
                    marker="o",
                    facecolors="none",
                    edgecolors="#0072B2",
                    linewidths=1.0,
                    label="retained peak",
                    zorder=4,
                )

            if suggestion is not None:
                interval_mask = (
                    (analysis.sample_s_km >= suggestion.lower.s_km)
                    & (analysis.sample_s_km <= suggestion.upper.s_km)
                )
                interval_xy = map_plot[interval_mask]
                endpoint_xy = np.vstack([
                    suggestion.lower.xy,
                    suggestion.center.xy,
                    suggestion.upper.xy,
                ])
                if coordinates == "lonlat":
                    endpoint_plot = _xy_to_lonlat(fault, endpoint_xy)
                else:
                    endpoint_plot = endpoint_xy
                interval_xy = np.vstack([
                    endpoint_plot[0],
                    interval_xy,
                    endpoint_plot[-1],
                ])
                map_axis.plot(
                    interval_xy[:, 0], interval_xy[:, 1],
                    color="#E69F00", linewidth=4.0, alpha=0.65,
                    solid_capstyle="round", label="suggested interval",
                    zorder=3,
                )
                map_axis.scatter(
                    endpoint_plot[[0, 2], 0], endpoint_plot[[0, 2], 1],
                    marker="D", color="#E69F00", edgecolors="white",
                    linewidths=0.5, label="suggested endpoints", zorder=5,
                )
                map_axis.scatter(
                    endpoint_plot[1, 0], endpoint_plot[1, 1],
                    marker="*", s=70, color="#D55E00", edgecolors="white",
                    linewidths=0.5, label="selected centre", zorder=6,
                )

            map_axis.annotate(
                "START",
                map_plot[0],
                xytext=(4, -4),
                textcoords="offset points",
                ha="left",
                va="top",
            )
            map_axis.annotate(
                "END",
                map_plot[-1],
                xytext=(-4, -4),
                textcoords="offset points",
                ha="right",
                va="top",
            )
            map_axis.set_aspect("equal", adjustable="datalim")
            if coordinates == "lonlat":
                map_axis.set_xlabel("Longitude")
                map_axis.set_ylabel("Latitude")
                map_axis.xaxis.set_major_formatter(LonFormatter())
                map_axis.yaxis.set_major_formatter(LatFormatter())
            else:
                map_axis.set_xlabel("x (km)")
                map_axis.set_ylabel("y (km)")
            map_axis.set_title("Reference top and curvature peaks")
            handles, labels = map_axis.get_legend_handles_labels()
            if handles:
                map_axis.legend(handles, labels, loc="best", fontsize="small")

            curvature_axis.plot(
                analysis.sample_s_km,
                analysis.signed_curvature_per_km,
                color="#0072B2",
                linewidth=1.2,
                label="signed curvature",
            )
            curvature_axis.plot(
                analysis.sample_s_km,
                analysis.abs_curvature_per_km,
                color="0.2",
                linewidth=1.1,
                linestyle="--",
                label="absolute curvature",
            )
            if analysis.peak_indices.size:
                curvature_axis.scatter(
                    analysis.sample_s_km[analysis.peak_indices],
                    analysis.abs_curvature_per_km[analysis.peak_indices],
                    marker="o", facecolors="none", edgecolors="#0072B2",
                    linewidths=1.0, zorder=4,
                )
            if suggestion is not None:
                curvature_axis.axvspan(
                    suggestion.lower.s_km,
                    suggestion.upper.s_km,
                    color="#E69F00",
                    alpha=0.18,
                    label="suggested interval",
                )
                if suggestion.curvature_threshold_per_km is not None:
                    curvature_axis.axhline(
                        suggestion.curvature_threshold_per_km,
                        color="#D55E00",
                        linestyle=":",
                        linewidth=1.0,
                        label="selected threshold",
                    )
                elif suggestion.turning_interval_s_km is not None:
                    curvature_axis.axvspan(
                        *suggestion.turning_interval_s_km,
                        color="0.65",
                        alpha=0.10,
                        label="turning integration interval",
                    )
                curvature_axis.axvline(
                    suggestion.center.s_km,
                    color="#D55E00",
                    linestyle="-.",
                    linewidth=0.9,
                )
            curvature_axis.axhline(0.0, color="0.65", linewidth=0.7)
            curvature_axis.set_xlabel("Arc length from START, s (km)")
            curvature_axis.set_ylabel("Planar curvature (1/km)")
            curvature_axis.set_title("Curvature-scale preflight")
            curvature_axis.legend(loc="best", fontsize="small")

            if title is None:
                title = f"{getattr(fault, 'name', 'Fault')}: transition preflight"
            if title:
                figure.suptitle(title)

            if save is not None or show:
                finish_fig(
                    figure,
                    save,
                    save=save is not None,
                    show=show,
                    dpi=dpi,
                    screen_dpi=screen_dpi,
                    close=close,
                )
            elif close:
                plt.close(figure)
        return figure, axes
    except Exception:
        if figure is not None:
            plt.close(figure)
        raise


def _xy_to_lonlat(fault, points):
    lon, lat = fault.xy2ll(points[:, 0], points[:, 1])
    return np.column_stack([lon, lat])


def _polyline_interval(top_xy, top_s, lower_s, upper_s, lower_xy, upper_xy):
    if lower_s <= upper_s:
        start_s, end_s = lower_s, upper_s
        start_xy, end_xy = lower_xy, upper_xy
    else:
        start_s, end_s = upper_s, lower_s
        start_xy, end_xy = upper_xy, lower_xy
    inside = (top_s > start_s) & (top_s < end_s)
    return np.vstack([start_xy, top_xy[inside], end_xy])


__all__ = ["plot_dip_profile_diagnostics", "plot_dip_transition_analysis"]
