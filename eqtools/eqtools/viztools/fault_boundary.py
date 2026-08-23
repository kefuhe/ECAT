"""Read-only diagnostics for an already extracted fault boundary.

The helpers in this module visualize boundary state owned by CSI.  They do not
extract, repair, relabel, or fall back between boundary algorithms, and they do
not write attributes on the supplied fault object.  Keeping that dependency
direction makes plotting safe to use before an inversion without coupling
Matplotlib to mesh, Laplacian, or Bayesian update logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np

from ._core import PlotStyle
from ._formatters import LatFormatter, LonFormatter
from ._style_utils import finish_fig


_EDGE_NAMES = ("top", "bottom", "left", "right")
_LOOP_EDGE_NAMES = ("top", "right", "bottom", "left")

# Okabe-Ito-derived colors keep the four semantic edges distinguishable in
# print and for common color-vision deficiencies.  They are intentionally
# stable across presets so an edge name never changes color between figures.
_EDGE_COLORS = {
    "top": "#0072B2",
    "bottom": "#D55E00",
    "left": "#009E73",
    "right": "#CC79A7",
}
_CORNER_SHORT_LABELS = {
    "top_left": "TL",
    "top_right": "TR",
    "bottom_left": "BL",
    "bottom_right": "BR",
}


def plot_fault_boundary_diagnostics(
    fault,
    *,
    views=("3d", "map"),
    coordinates="xy",
    show_mesh=True,
    show_boundary_faces=True,
    annotate_corners=True,
    show_projection=True,
    elevation=22.0,
    azimuth=-60.0,
    shape=(1.0, 1.0, 0.45),
    title=None,
    style="science",
    figsize=(11.0, 4.0),
    fontsize=8,
    dpi=300,
    rcparams=None,
    save=None,
    show=False,
    screen_dpi=200,
    close=False,
):
    """Plot read-only diagnostics for the current four-edge fault result.

    The fault must already have a successful boundary result, normally from
    ``fault.find_fault_fouredge_vertices(...)``.  This function deliberately
    does not call that method: plotting therefore cannot select an extraction
    backend, trigger fallback, rebuild boundary caches, or change smoothing
    state.

    Parameters
    ----------
    fault : CSI triangular-fault-like object
        Object containing ``Vertices``, ``Faces``, ``edge_vertex_indices`` and
        ``edge_extraction_method``.  Optional segment, corner, boundary-face,
        and provenance fields are drawn when available.
    views : sequence of {"3d", "map", "sequence"}, optional
        Panels to draw, in the requested order. ``sequence`` is an optional
        unwrapped four-edge ordering diagnostic; its horizontal axis is not a
        physical distance or fault cross-section.
    coordinates : {"xy", "lonlat"}, optional
        Coordinate frame for the 3-D and map panels.  Depth remains the CSI
        positive-down depth stored in the third coordinate column.
    show_mesh : bool, optional
        Draw the complete triangular mesh in light gray.
    show_boundary_faces : bool, optional
        Shade the inclusive boundary-face membership associated with each edge.
    annotate_corners : bool, optional
        Mark and label available junction vertices.
    show_projection : bool, optional
        In ``xy`` map view, draw strike and left/right projection vectors from
        ``edge_extraction_info`` when present.  No direction is recomputed.
    elevation, azimuth : float, optional
        Matplotlib 3-D viewing angles in degrees.
    shape : 3-tuple, optional
        3-D box aspect passed to :func:`optimize_3d_plot`.
    title : str, optional
        Figure title.  The default includes fault name and resolved method.
    style, figsize, fontsize, dpi, rcparams : optional
        Standard :mod:`eqtools.viztools` style and output controls.
    save : path-like, optional
        Output path passed to :func:`eqtools.viztools.finish_fig`.
    show, close : bool, optional
        Display or close the figure after optional saving.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    axes : dict[str, matplotlib.axes.Axes]
        Axes keyed by the requested view names.

    Notes
    -----
    ``edge_vertex_indices`` index ``fault.Vertices`` and have a canonical edge
    direction.  ``edge_triangles_indices`` index face rows and are treated only
    as membership; their list order is never interpreted as a spatial order.
    """

    if isinstance(save, bool):
        raise ValueError("save must be a file path or None, not a boolean.")

    view_names = _normalize_views(views)
    state = _read_boundary_state(fault, coordinates=coordinates)

    import matplotlib.pyplot as plt

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
            figure = plt.figure(constrained_layout=True)
            grid = figure.add_gridspec(1, len(view_names))
            for column, view_name in enumerate(view_names):
                if view_name == "3d":
                    axis = figure.add_subplot(grid[0, column], projection="3d")
                    _draw_3d_panel(
                        axis,
                        state,
                        show_mesh=show_mesh,
                        show_boundary_faces=show_boundary_faces,
                        annotate_corners=annotate_corners,
                        elevation=elevation,
                        azimuth=azimuth,
                        shape=shape,
                    )
                elif view_name == "map":
                    axis = figure.add_subplot(grid[0, column])
                    _draw_map_panel(
                        axis,
                        state,
                        show_mesh=show_mesh,
                        show_boundary_faces=show_boundary_faces,
                        annotate_corners=annotate_corners,
                        show_projection=show_projection,
                    )
                else:
                    axis = figure.add_subplot(grid[0, column])
                    _draw_sequence_panel(axis, state)
                axes[view_name] = axis

            method = state["method"]
            if title is None:
                fault_name = getattr(fault, "name", "Fault")
                title = f"{fault_name}: boundary diagnostics ({method})"
            if title:
                figure.suptitle(title)

            finish_fig(
                figure,
                save,
                save=save is not None,
                show=show,
                dpi=dpi,
                screen_dpi=screen_dpi,
                close=close,
            )
    except Exception:
        # A plotting error must not leave a half-created GUI figure behind.
        # The fault itself is never mutated, so no model rollback is needed.
        if figure is not None:
            plt.close(figure)
        raise

    return figure, axes


def _normalize_views(views):
    if isinstance(views, str):
        views = (views,)
    elif isinstance(views, Sequence):
        views = tuple(views)
    else:
        raise TypeError("views must be a view name or a sequence of view names.")

    allowed = {"3d", "map", "sequence"}
    if not views:
        raise ValueError("views must contain at least one panel name.")
    unknown = [view for view in views if view not in allowed]
    if unknown:
        raise ValueError(
            "views may contain only '3d', 'map', and 'sequence'; "
            f"got {unknown}."
        )
    if len(set(views)) != len(views):
        raise ValueError("views must not contain duplicate panel names.")
    return views


def _read_boundary_state(fault, *, coordinates):
    if coordinates not in {"xy", "lonlat"}:
        raise ValueError("coordinates must be 'xy' or 'lonlat'.")

    method = getattr(fault, "edge_extraction_method", None)
    if method is None or not hasattr(fault, "edge_vertex_indices"):
        raise ValueError(
            "Fault boundary is not available. Call "
            "find_fault_fouredge_vertices(...) before plotting diagnostics."
        )

    vertices_xyz = np.asarray(getattr(fault, "Vertices", None), dtype=float)
    faces = np.asarray(getattr(fault, "Faces", None), dtype=int)
    if vertices_xyz.ndim != 2 or vertices_xyz.shape[1] != 3:
        raise ValueError("fault.Vertices must be an (N, 3) array.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("fault.Faces must be an (M, 3) triangular connectivity array.")

    if coordinates == "lonlat":
        vertices = np.asarray(getattr(fault, "Vertices_ll", None), dtype=float)
        if vertices.shape != vertices_xyz.shape:
            raise ValueError(
                "coordinates='lonlat' requires fault.Vertices_ll with the same "
                "shape as fault.Vertices."
            )
        x_label, y_label = "Longitude", "Latitude"
    else:
        vertices = vertices_xyz
        x_label, y_label = "X (km)", "Y (km)"

    raw_edge_indices = fault.edge_vertex_indices
    missing_edges = [name for name in _EDGE_NAMES if name not in raw_edge_indices]
    if missing_edges:
        raise ValueError(f"Boundary result is missing edges: {missing_edges}.")

    edge_indices = {}
    for name in _EDGE_NAMES:
        indices = np.asarray(raw_edge_indices[name], dtype=int).reshape(-1)
        _validate_indices(indices, len(vertices), f"edge_vertex_indices[{name!r}]")
        if indices.size == 0:
            raise ValueError(f"edge_vertex_indices[{name!r}] must not be empty.")
        edge_indices[name] = indices

    raw_segments = getattr(fault, "edge_index_segments", {})
    edge_segments = {}
    for name in _EDGE_NAMES:
        segments = raw_segments.get(name) if hasattr(raw_segments, "get") else None
        if not segments:
            segments = [edge_indices[name]]
        normalized = []
        for isegment, segment in enumerate(segments):
            indices = np.asarray(segment, dtype=int).reshape(-1)
            _validate_indices(
                indices,
                len(vertices),
                f"edge_index_segments[{name!r}][{isegment}]",
            )
            normalized.append(indices)
        edge_segments[name] = normalized

    boundary_faces = {}
    raw_boundary_faces = getattr(fault, "edge_triangles_indices", {})
    for name in _EDGE_NAMES:
        values = raw_boundary_faces.get(name, []) if hasattr(raw_boundary_faces, "get") else []
        indices = np.asarray(values, dtype=int).reshape(-1)
        _validate_indices(indices, len(faces), f"edge_triangles_indices[{name!r}]")
        boundary_faces[name] = indices

    raw_corners = getattr(fault, "corner_vertex_indices", {})
    corner_indices = {}
    if hasattr(raw_corners, "items"):
        for name, value in raw_corners.items():
            index = int(value)
            _validate_indices(np.array([index]), len(vertices), f"corner_vertex_indices[{name!r}]")
            corner_indices[str(name)] = index

    return {
        "vertices": vertices,
        "faces": faces,
        "edge_indices": edge_indices,
        "edge_segments": edge_segments,
        "boundary_faces": boundary_faces,
        "corner_indices": corner_indices,
        "info": dict(getattr(fault, "edge_extraction_info", {}) or {}),
        "method": str(method),
        "coordinates": coordinates,
        "x_label": x_label,
        "y_label": y_label,
    }


def _validate_indices(indices, size, label):
    if indices.size and (np.min(indices) < 0 or np.max(indices) >= size):
        raise ValueError(f"{label} contains an index outside its declared index space.")


def _draw_3d_panel(
    ax,
    state,
    *,
    show_mesh,
    show_boundary_faces,
    annotate_corners,
    elevation,
    azimuth,
    shape,
):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = state["vertices"]
    faces = state["faces"]
    if show_mesh:
        mesh = Poly3DCollection(
            vertices[faces],
            facecolor="#D9D9D9",
            edgecolor="#7F7F7F",
            linewidth=0.25,
            alpha=0.16,
        )
        ax.add_collection3d(mesh)

    if show_boundary_faces:
        for edge_name in _EDGE_NAMES:
            face_indices = state["boundary_faces"][edge_name]
            if face_indices.size:
                collection = Poly3DCollection(
                    vertices[faces[face_indices]],
                    facecolor=_EDGE_COLORS[edge_name],
                    edgecolor="none",
                    alpha=0.12,
                )
                ax.add_collection3d(collection)

    _plot_edge_segments_3d(ax, state)
    if annotate_corners:
        _plot_corners_3d(ax, state)

    ax.set_xlabel(state["x_label"])
    ax.set_ylabel(state["y_label"])
    ax.set_zlabel("Depth (km, positive down)")
    ax.set_title("3-D boundary and mesh")
    ax.view_init(elev=elevation, azim=azimuth)
    _set_3d_limits(ax, vertices)
    ax.invert_zaxis()

    from .viz_3d import optimize_3d_plot

    optimize_3d_plot(ax, shape=shape, background_color="white", show_grid=True)
    # Axes3D uses one geometric tick-length factor for both major and minor
    # ticks.  Keeping style-provided minor ticks would therefore create a dense
    # comb with no readable length hierarchy; this diagnostic shows majors only.
    from matplotlib.ticker import MaxNLocator, NullLocator

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_minor_locator(NullLocator())
    if state["coordinates"] == "lonlat":
        _apply_geo_formatters(ax, vertices)


def _draw_map_panel(
    ax,
    state,
    *,
    show_mesh,
    show_boundary_faces,
    annotate_corners,
    show_projection,
):
    from matplotlib.collections import LineCollection, PolyCollection

    vertices = state["vertices"]
    faces = state["faces"]
    xy = vertices[:, :2]
    if show_mesh:
        triangle_xy = xy[faces]
        segments = np.concatenate(
            (
                triangle_xy[:, [0, 1]],
                triangle_xy[:, [1, 2]],
                triangle_xy[:, [2, 0]],
            ),
            axis=0,
        )
        ax.add_collection(
            LineCollection(segments, colors="#7F7F7F", linewidths=0.25, alpha=0.25)
        )

    if show_boundary_faces:
        for edge_name in _EDGE_NAMES:
            face_indices = state["boundary_faces"][edge_name]
            if face_indices.size:
                ax.add_collection(
                    PolyCollection(
                        xy[faces[face_indices]],
                        facecolor=_EDGE_COLORS[edge_name],
                        edgecolor="none",
                        alpha=0.10,
                    )
                )

    _plot_edge_segments_2d(ax, state)
    if annotate_corners:
        _plot_corners_2d(ax, state)
    if show_projection:
        _plot_projection_vectors(ax, state)

    ax.margins(x=0.03, y=0.08)
    _set_plan_view_aspect(ax, state)
    ax.set_xlabel(state["x_label"])
    ax.set_ylabel(state["y_label"])
    ax.set_title("Map view and side naming")
    if state["coordinates"] == "lonlat":
        _apply_geo_formatters(ax, vertices)
    ax.legend(loc="best", ncols=2)


def _draw_sequence_panel(ax, state):
    cursor = 0
    labelled = set()
    for edge_name in _LOOP_EDGE_NAMES:
        segments = state["edge_segments"][edge_name]
        if edge_name in {"bottom", "left"}:
            segments = [segment[::-1] for segment in reversed(segments)]
        for segment in segments:
            if segment.size == 0:
                continue
            depths = state["vertices"][segment, 2]
            positions = np.arange(cursor, cursor + len(segment))
            label = edge_name if edge_name not in labelled else None
            ax.plot(
                positions,
                depths,
                color=_EDGE_COLORS[edge_name],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                label=label,
            )
            labelled.add(edge_name)
            cursor += len(segment) + 1

    ax.invert_yaxis()
    ax.set_xlabel("Ordered boundary vertex position (not distance)")
    ax.set_ylabel("Depth (km, positive down)")
    ax.set_title("Unwrapped boundary sequence")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncols=2)

    summary = _diagnostic_summary(state)
    if summary:
        ax.text(
            0.02,
            0.50,
            summary,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize="x-small",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82},
        )


def _plot_edge_segments_3d(ax, state):
    vertices = state["vertices"]
    for edge_name in _EDGE_NAMES:
        for isegment, indices in enumerate(state["edge_segments"][edge_name]):
            points = vertices[indices]
            ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=_EDGE_COLORS[edge_name],
                linewidth=2.0,
                label=edge_name if isegment == 0 else None,
            )
    ax.legend(loc="best", ncols=2)


def _plot_edge_segments_2d(ax, state):
    vertices = state["vertices"]
    for edge_name in _EDGE_NAMES:
        for isegment, indices in enumerate(state["edge_segments"][edge_name]):
            points = vertices[indices]
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=_EDGE_COLORS[edge_name],
                linewidth=1.8,
                label=edge_name if isegment == 0 else None,
            )


def _plot_corners_3d(ax, state):
    vertices = state["vertices"]
    for corner_name, index in state["corner_indices"].items():
        point = vertices[index]
        ax.scatter(*point, color="#000000", marker="o", s=18, depthshade=False)
        label = _CORNER_SHORT_LABELS.get(corner_name, corner_name)
        ax.text(*point, f" {label}", fontsize="x-small")


def _plot_corners_2d(ax, state):
    vertices = state["vertices"]
    for corner_name, index in state["corner_indices"].items():
        point = vertices[index]
        ax.scatter(point[0], point[1], color="#000000", marker="o", s=18, zorder=5)
        label = _CORNER_SHORT_LABELS.get(corner_name, corner_name)
        ax.annotate(
            label,
            point[:2],
            xytext=(3, 3),
            textcoords="offset points",
            fontsize="x-small",
        )


def _plot_projection_vectors(ax, state):
    # Projection vectors are stored in the fault's projected x/y frame.  In a
    # lon/lat panel we report their provenance as text instead of pretending
    # degree and kilometre components share a scale.
    info = state["info"]
    if state["coordinates"] != "xy":
        return
    vertices = state["vertices"]
    center = np.mean(vertices[:, :2], axis=0)
    span = np.ptp(vertices[:, :2], axis=0)
    scale = 0.18 * max(float(np.max(span)), 1.0)
    vectors = (
        ("strike_vector_xy", "strike", "#4D4D4D", "--"),
        ("projection_vector_xy", "side projection", "#000000", "-"),
    )
    for key, label, color, linestyle in vectors:
        value = info.get(key)
        if value is None:
            continue
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.size != 2 or not np.all(np.isfinite(vector)):
            continue
        norm = np.linalg.norm(vector)
        if norm == 0:
            continue
        vector = vector / norm * scale
        ax.annotate(
            "",
            xy=center + vector,
            xytext=center,
            arrowprops={"arrowstyle": "->", "color": color, "linestyle": linestyle},
        )
        ax.plot([], [], color=color, linestyle=linestyle, label=label)


def _diagnostic_summary(state):
    info = state["info"]
    lines = [f"method: {state['method']}"]
    extraction_parameters = info.get("extraction_parameters", {}) or {}
    gap_policy = info.get("gap_policy", extraction_parameters.get("gap_policy"))
    if gap_policy is not None:
        lines.append(f"gap policy: {gap_policy}")
    if info.get("left_right_naming_rule") is not None:
        lines.append(f"side naming: {info['left_right_naming_rule']}")
    if info.get("component_node_counts") is not None:
        lines.append(f"components: {info['component_node_counts']}")
    if info.get("raw_run_summary") is not None:
        lines.append(f"raw runs: {info['raw_run_summary']}")
    return "\n".join(lines)


def _set_3d_limits(ax, vertices):
    for setter, column in (
        (ax.set_xlim, 0),
        (ax.set_ylim, 1),
        (ax.set_zlim, 2),
    ):
        lower = float(np.min(vertices[:, column]))
        upper = float(np.max(vertices[:, column]))
        if np.isclose(lower, upper):
            pad = max(abs(lower) * 0.01, 0.5)
            lower -= pad
            upper += pad
        setter(lower, upper)


def _set_plan_view_aspect(ax, state):
    """Preserve map scale without expanding the plotted coordinate limits."""
    if state["coordinates"] == "xy":
        ax.set_aspect("equal", adjustable="box")
        return

    latitude = state["vertices"][:, 1]
    mean_latitude = float(np.nanmean(latitude))
    cosine = abs(float(np.cos(np.deg2rad(mean_latitude))))
    # Longitude degrees shorten by cos(latitude).  ``adjustable='box'`` keeps
    # the tight data limits and changes the axes box instead of inventing a
    # much wider latitude range to fill a square subplot.
    ax.set_aspect(1.0 / max(cosine, 1.0e-6), adjustable="box")


def _apply_geo_formatters(ax, vertices):
    ax.xaxis.set_major_formatter(
        LonFormatter(decimal_places=_decimal_places_for_span(vertices[:, 0]))
    )
    ax.yaxis.set_major_formatter(
        LatFormatter(decimal_places=_decimal_places_for_span(vertices[:, 1]))
    )


def _decimal_places_for_span(values, target_intervals=5):
    """Choose enough decimal places to keep nearby geographic ticks distinct."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 2
    span = float(np.ptp(finite))
    if span <= 0.0:
        return 2
    approximate_step = span / target_intervals
    return int(np.clip(np.ceil(-np.log10(approximate_step)), 0, 5))


__all__ = ["plot_fault_boundary_diagnostics"]
