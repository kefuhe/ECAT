"""Plotly renderer for detached ECAT map-viewer payloads."""

from dataclasses import replace
import math

import numpy as np


DEFAULT_MAX_POINTS = 60_000


def _payload_with_style_overrides(payload, style_overrides):
    if not style_overrides:
        return payload
    style = dict(payload.spec.style)
    style.update(dict(style_overrides))
    return replace(payload, spec=replace(payload.spec, style=style))


def renderer_backend():
    """Return the single supported Plotly map backend."""

    import plotly.graph_objects as go

    if not hasattr(go, "Scattermap"):
        raise ImportError(
            "The ECAT map viewer requires Plotly 5.24 or newer with "
            "MapLibre Scattermap support."
        )
    return "maplibre"


def map_layout_key():
    """Plotly layout key used by the active renderer backend."""

    renderer_backend()
    return "map"


def _scattermap(**kwargs):
    import plotly.graph_objects as go

    renderer_backend()
    return go.Scattermap(**kwargs)


def _layer_meta(payload, role, *, colorbar_capable=False):
    return {
        "layer_id": payload.spec.id,
        "layer_kind": payload.spec.kind,
        "layer_role": role,
        "colorbar_capable": bool(colorbar_capable),
        "source": str(payload.spec.source),
        "variable": payload.data.get("variable", payload.spec.variable),
        "mask": payload.data.get("mask", payload.spec.mask),
        "units": payload.metadata.units,
        "positive_convention": payload.data.get("positive_convention"),
        "grid_topology": payload.metadata.grid_topology,
        "derived_display": payload.metadata.derived_display or (
            role == "display_overview"
        ),
    }


def _display_values(values, style):
    return np.asarray(values, dtype=float) * float(
        style.get("display_factor", 1.0)
    )


def _finite_limits(display_values, style):
    if style.get("vmin") is not None:
        return float(style["vmin"]), float(style["vmax"])
    values = np.asarray(display_values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    lower, upper = np.nanpercentile(finite, [2.0, 98.0])
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None, None
    if lower == upper:
        padding = max(abs(lower) * 0.01, 1.0e-12)
        lower -= padding
        upper += padding
    if style.get("symmetry", False):
        bound = max(abs(float(lower)), abs(float(upper)))
        lower, upper = -bound, bound
    return float(lower), float(upper)


def _sample_indices(valid_mask, max_points=DEFAULT_MAX_POINTS):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    indices = np.flatnonzero(valid_mask.reshape(-1))
    if indices.size <= max_points:
        return indices
    if valid_mask.ndim != 2:
        positions = np.linspace(0, indices.size - 1, max_points, dtype=int)
        return indices[positions]
    spatial_step = int(math.ceil(math.sqrt(indices.size / max_points)))
    row, column = np.nonzero(valid_mask)
    keep = (row % spatial_step == 0) & (column % spatial_step == 0)
    sampled = np.ravel_multi_index(
        (row[keep], column[keep]),
        valid_mask.shape,
    )
    if sampled.size == 0:
        positions = np.linspace(0, indices.size - 1, max_points, dtype=int)
        return indices[positions]
    if sampled.size > max_points:
        positions = np.linspace(0, sampled.size - 1, max_points, dtype=int)
        sampled = sampled[positions]
    return sampled


def _properties_text(properties):
    values = []
    for key, value in sorted(dict(properties or {}).items()):
        if value not in (None, "", [], {}):
            values.append(f"{key}: {value}")
    return "<br>".join(values)


def _geometry_lines(geometry):
    if not geometry:
        return []
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "LineString":
        return [coordinates]
    if geometry_type == "MultiLineString":
        return list(coordinates)
    if geometry_type == "Polygon":
        return list(coordinates)
    if geometry_type == "MultiPolygon":
        return [ring for polygon in coordinates for ring in polygon]
    if geometry_type == "GeometryCollection":
        lines = []
        for part in geometry.get("geometries") or ():
            lines.extend(_geometry_lines(part))
        return lines
    return []


def _geometry_points(geometry):
    if not geometry:
        return []
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Point":
        return [coordinates]
    if geometry_type == "MultiPoint":
        return list(coordinates)
    if geometry_type == "GeometryCollection":
        points = []
        for part in geometry.get("geometries") or ():
            points.extend(_geometry_points(part))
        return points
    return []


def _earthquake_traces(payload, *, show_colorbar=False):
    data = payload.data
    style = payload.spec.style
    count = len(data["longitude"])
    magnitude = np.asarray(data.get("magnitude", np.full(count, 4.0)))
    depth = np.asarray(data.get("depth", np.zeros(count)))
    display_depth = _display_values(depth, style)
    lower, upper = _finite_limits(display_depth, style)
    display_unit = style.get("display_unit") or "km"
    marker_size = np.clip((magnitude - 1.0) * 3.0, 4.0, 22.0)
    text = []
    for index in range(count):
        fields = []
        for name, label in (
            ("place", "Place"),
            ("time", "Time"),
            ("magnitude", "Magnitude"),
            ("depth", "Depth (km)"),
        ):
            if name in data:
                value = data[name][index]
                if name == "depth":
                    value = display_depth[index]
                    label = f"Depth ({display_unit})"
                fields.append(f"{label}: {value}")
        text.append("<br>".join(fields))
    marker = {
        "size": marker_size,
        "color": display_depth,
        "colorscale": style.get("cmap", "Viridis"),
        "showscale": bool(show_colorbar and "depth" in data),
        "colorbar": {"title": f"Depth ({display_unit})"},
    }
    if lower is not None:
        marker.update(cmin=lower, cmax=upper)
    return [
        _scattermap(
            lon=np.asarray(data["longitude"]),
            lat=np.asarray(data["latitude"]),
            mode="markers",
            marker=marker,
            opacity=float(style.get("alpha", 0.8)),
            text=text,
            hoverinfo="text",
            name=payload.spec.name,
            meta=_layer_meta(
                payload,
                "events",
                colorbar_capable="depth" in data,
            ),
        )
    ]


def _vector_traces(payload):
    line_longitude = []
    line_latitude = []
    line_hover = []
    point_longitude = []
    point_latitude = []
    point_hover = []
    for geometry, properties in zip(
        payload.data["geometries"],
        payload.data["properties"],
    ):
        text = (
            ""
            if payload.spec.id.startswith("background.")
            else _properties_text(properties)
        )
        for line in _geometry_lines(geometry):
            for point in line:
                line_longitude.append(point[0])
                line_latitude.append(point[1])
                line_hover.append(text)
            line_longitude.append(None)
            line_latitude.append(None)
            line_hover.append(None)
        for point in _geometry_points(geometry):
            point_longitude.append(point[0])
            point_latitude.append(point[1])
            point_hover.append(text)

    color = payload.spec.style.get("color", "#d62728")
    alpha = float(payload.spec.style.get("alpha", 0.85))
    hoverinfo = "skip" if payload.spec.id.startswith("background.") else "text"
    traces = []
    if line_longitude:
        traces.append(
            _scattermap(
                lon=line_longitude,
                lat=line_latitude,
                mode="lines",
                line={
                    "color": color,
                    "width": float(payload.spec.style.get("line_width", 1.5)),
                },
                opacity=alpha,
                text=None if hoverinfo == "skip" else line_hover,
                hoverinfo=hoverinfo,
                name=payload.spec.name,
                meta=_layer_meta(payload, "geometry"),
            )
        )
    if point_longitude:
        traces.append(
            _scattermap(
                lon=point_longitude,
                lat=point_latitude,
                mode="markers",
                marker={
                    "color": color,
                    "size": float(payload.spec.style.get("marker_size", 7.0)),
                },
                opacity=alpha,
                text=None if hoverinfo == "skip" else point_hover,
                hoverinfo=hoverinfo,
                name=payload.spec.name,
                meta=_layer_meta(payload, "points"),
            )
        )
    if not traces:
        raise ValueError(
            f"Vector layer {payload.spec.id!r} contains no renderable "
            "point, line, or polygon geometry."
        )
    return traces


def _gnss_traces(payload):
    data = payload.data
    longitude = np.asarray(data["longitude"])
    latitude = np.asarray(data["latitude"])
    east = np.asarray(data["east"])
    north = np.asarray(data["north"])
    scale = float(payload.spec.style.get("display_scale", 0.02))
    line_longitude = []
    line_latitude = []
    endpoint_longitude = []
    endpoint_latitude = []
    for lon, lat, east_value, north_value in zip(
        longitude,
        latitude,
        east,
        north,
    ):
        cosine_latitude = max(abs(math.cos(math.radians(float(lat)))), 1.0e-6)
        end_lon = lon + east_value * scale / cosine_latitude
        end_lat = lat + north_value * scale
        line_longitude.extend([lon, end_lon, None])
        line_latitude.extend([lat, end_lat, None])
        endpoint_longitude.append(end_lon)
        endpoint_latitude.append(end_lat)
    units = payload.metadata.units or "unknown"
    reference_frame = payload.data.get("reference_frame") or "unknown"
    text = [
        (
            f"Station: {station}<br>"
            f"East: {east_value:g} {units}<br>"
            f"North: {north_value:g} {units}<br>"
            f"Reference frame: {reference_frame}<br>"
            f"Display scale: {scale:g}"
        )
        for station, east_value, north_value in zip(
            data["station"],
            east,
            north,
        )
    ]
    color = payload.spec.style.get("color", "#2ca02c")
    alpha = float(payload.spec.style.get("alpha", 0.85))
    return [
        _scattermap(
            lon=line_longitude,
            lat=line_latitude,
            mode="lines",
            line={
                "color": color,
                "width": float(payload.spec.style.get("line_width", 1.5)),
            },
            opacity=alpha,
            hoverinfo="skip",
            showlegend=False,
            name=payload.spec.name,
            meta=_layer_meta(payload, "vectors"),
        ),
        _scattermap(
            lon=longitude,
            lat=latitude,
            mode="markers",
            marker={
                "color": color,
                "size": float(payload.spec.style.get("marker_size", 7.0)),
            },
            opacity=alpha,
            text=text,
            hoverinfo="text",
            name=payload.spec.name,
            meta=_layer_meta(payload, "stations"),
        ),
        _scattermap(
            lon=endpoint_longitude,
            lat=endpoint_latitude,
            mode="markers",
            marker={
                "color": color,
                "size": max(
                    3.0,
                    float(payload.spec.style.get("marker_size", 7.0)) * 0.65,
                ),
            },
            opacity=alpha,
            hoverinfo="skip",
            showlegend=False,
            name=payload.spec.name,
            meta=_layer_meta(payload, "vector_endpoints"),
        ),
    ]


def _grid_trace(payload, *, show_colorbar=False):
    data = payload.data
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    style = payload.spec.style
    all_display_values = _display_values(data["values"], style)
    lower, upper = _finite_limits(
        all_display_values[valid_mask],
        style,
    )
    indices = _sample_indices(valid_mask)
    longitude = np.asarray(data["longitude"]).reshape(-1)[indices]
    latitude = np.asarray(data["latitude"]).reshape(-1)[indices]
    values = all_display_values.reshape(-1)[indices]
    units = style.get("display_unit") or payload.metadata.units or "unknown"
    variable = data["variable"]
    convention = data.get("positive_convention") or "not declared"
    text = [
        (
            f"{variable}: {value:g} {units}<br>"
            f"Longitude: {lon:.6f}<br>Latitude: {lat:.6f}<br>"
            f"Positive convention: {convention}"
        )
        for lon, lat, value in zip(longitude, latitude, values)
    ]
    marker = {
        "size": float(payload.spec.style.get("marker_size", 5.0)),
        "color": values,
        "colorscale": style.get("cmap", "RdBu_r"),
        "showscale": bool(show_colorbar),
        "colorbar": {"title": f"{variable} ({units})"},
    }
    if lower is not None:
        marker.update(cmin=lower, cmax=upper)
    return _scattermap(
        lon=longitude,
        lat=latitude,
        mode="markers",
        marker=marker,
        opacity=float(style.get("alpha", 0.8)),
        text=text,
        hoverinfo="text",
        name=payload.spec.name,
        meta=_layer_meta(
            payload,
            "display_overview",
            colorbar_capable=True,
        ),
    )


def _varres_traces(payload, *, show_colorbar=False):
    data = payload.data
    style = payload.spec.style
    boundary_longitude = []
    boundary_latitude = []
    for vertices in data["vertices"]:
        vertices = np.asarray(vertices)
        closed = np.vstack((vertices, vertices[0]))
        boundary_longitude.extend(closed[:, 0].tolist() + [None])
        boundary_latitude.extend(closed[:, 1].tolist() + [None])
    display_values = _display_values(data["values"], style)
    lower, upper = _finite_limits(display_values, style)
    units = style.get("display_unit") or payload.metadata.units
    colorbar_title = data["variable"]
    if units:
        colorbar_title = f"{colorbar_title} ({units})"
    marker = {
        "size": float(payload.spec.style.get("marker_size", 6.0)),
        "color": display_values,
        "colorscale": style.get("cmap", "RdBu_r"),
        "showscale": bool(show_colorbar),
        "colorbar": {"title": colorbar_title},
    }
    if lower is not None:
        marker.update(cmin=lower, cmax=upper)
    return [
        _scattermap(
            lon=boundary_longitude,
            lat=boundary_latitude,
            mode="lines",
            line={
                "color": payload.spec.style.get("color", "#444444"),
                "width": float(payload.spec.style.get("line_width", 0.7)),
            },
            opacity=float(style.get("alpha", 0.65)),
            hoverinfo="skip",
            showlegend=False,
            name=payload.spec.name,
            meta=_layer_meta(payload, "cells"),
        ),
        _scattermap(
            lon=np.asarray(data["longitude"]),
            lat=np.asarray(data["latitude"]),
            mode="markers",
            marker=marker,
            opacity=float(style.get("alpha", 0.85)),
            text=[
                (
                    f"{data['variable']}: {value:g}"
                    + (f" {units}" if units else "")
                )
                for value in display_values
            ],
            hoverinfo="text",
            name=payload.spec.name,
            meta=_layer_meta(
                payload,
                "values",
                colorbar_capable=True,
            ),
        ),
    ]


def traces_for_payload(
    payload,
    *,
    show_colorbar=False,
    alpha=None,
    style_overrides=None,
):
    """Create traces with stable parent identity and runtime display options."""

    payload = _payload_with_style_overrides(payload, style_overrides)
    if payload.spec.kind == "earthquake_catalog":
        traces = _earthquake_traces(
            payload,
            show_colorbar=show_colorbar,
        )
    elif payload.spec.kind == "vector":
        traces = _vector_traces(payload)
    elif payload.spec.kind == "gnss_velocity":
        traces = _gnss_traces(payload)
    elif payload.spec.kind in {"observation_grid", "raster"}:
        traces = [
            _grid_trace(
                payload,
                show_colorbar=show_colorbar,
            )
        ]
    elif payload.spec.kind == "csi_varres":
        traces = _varres_traces(
            payload,
            show_colorbar=show_colorbar,
        )
    else:
        raise ValueError(f"No renderer for layer kind {payload.spec.kind!r}.")
    if alpha is not None:
        alpha = float(alpha)
        for trace in traces:
            trace.opacity = alpha
    return traces


def view_for_region(region):
    """Return a Plotly map center and zoom for a lon/lat bounding box."""

    if region is None:
        return {"lat": 0.0, "lon": 0.0}, 1.0
    west, east, south, north = region
    center = {"lon": (west + east) / 2.0, "lat": (south + north) / 2.0}
    span = max(east - west, north - south, 1.0e-6)
    zoom = max(0.0, min(12.0, math.log2(360.0 / span) - 0.75))
    return center, zoom


def create_base_figure(project, *, session_key=None):
    """Create a payload-free base figure with stable view persistence."""

    import plotly.graph_objects as go

    center, zoom = view_for_region(project.region)
    key = session_key or f"ecat-map:{project.name}"
    figure = go.Figure()
    figure.add_trace(
        _scattermap(
            lon=[],
            lat=[],
            mode="markers",
            marker={"size": 1, "opacity": 0},
            hoverinfo="skip",
            showlegend=False,
            meta={"layer_role": "base"},
        )
    )
    map_config = {
        "style": project.basemap,
        "center": center,
        "zoom": zoom,
        "uirevision": key,
    }
    figure.update_layout(
        **{
            map_layout_key(): map_config,
            "uirevision": key,
            "margin": {"r": 0, "t": 0, "l": 0, "b": 0},
            "dragmode": "pan",
            # The grouped layer panel is the authoritative visibility legend.
            # Plotly's trace legend duplicates those names and consumes map
            # width, especially when displaying global scientific context.
            "showlegend": False,
        }
    )
    return figure


def add_payload(
    figure,
    payload,
    *,
    visible=True,
    show_colorbar=False,
    alpha=None,
    style_overrides=None,
):
    """Append one payload's renderer traces to a figure."""

    for trace in traces_for_payload(
        payload,
        show_colorbar=show_colorbar,
        alpha=alpha,
        style_overrides=style_overrides,
    ):
        trace.visible = visible
        figure.add_trace(trace)
    return figure


def layer_trace_indices(figure, layer_id):
    """Find every renderer child trace for one stable layer id."""

    data = figure.get("data", ()) if isinstance(figure, dict) else figure.data
    indices = []
    for index, trace in enumerate(data):
        meta = (
            trace.get("meta") if isinstance(trace, dict)
            else getattr(trace, "meta", None)
        )
        if isinstance(meta, dict) and meta.get("layer_id") == layer_id:
            indices.append(index)
    return indices


__all__ = [
    "DEFAULT_MAX_POINTS",
    "add_payload",
    "create_base_figure",
    "layer_trace_indices",
    "map_layout_key",
    "renderer_backend",
    "traces_for_payload",
    "view_for_region",
]
