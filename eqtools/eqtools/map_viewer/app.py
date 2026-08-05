"""Dash application orchestration for the ECAT research map viewer."""

import json

from .catalog import LayerCatalog
from .models import ViewerState
from .renderer_plotly import (
    add_payload,
    create_base_figure,
    layer_trace_indices,
    map_layout_key,
    renderer_backend,
    traces_for_payload,
    view_for_region,
)
from .ui import (
    build_layout,
    default_active_layer_id,
    default_layer_alpha,
    default_color_controls,
    flatten_group_values,
    layer_details_component,
    layer_groups,
    supports_color_controls,
)


BASEMAPS = (
    ("OpenStreetMap", "open-street-map", True),
    ("Light — Carto", "carto-positron", True),
    ("Dark — Carto", "carto-darkmatter", True),
    ("Streets", "streets", True),
    ("Terrain / outdoors", "outdoors", True),
    ("Satellite imagery", "satellite", True),
    ("Satellite with streets", "satellite-streets", True),
    ("No external tiles", "white-bg", False),
)

_RENDER_KIND_RANK = {
    "observation_grid": 10,
    "raster": 10,
    "csi_varres": 10,
    "vector": 20,
    "gnss_velocity": 30,
    "earthquake_catalog": 40,
}


def basemap_options():
    """Return labeled dropdown options supported by the active backend."""

    renderer_backend()
    return [
        {
            "label": (
                f"{label} (online)"
                if online and value != "open-street-map"
                else label
            ),
            "value": value,
            "disabled": False,
        }
        for label, value, online in BASEMAPS
    ]


def basemap_patch(map_style):
    """Return a layout-only Dash patch for a basemap change."""

    from dash import Patch

    patch = Patch()
    patch["layout"][map_layout_key()]["style"] = map_style
    return patch


def _layer_render_key(spec, catalog):
    order = {layer.id: index for index, layer in enumerate(catalog.layers)}
    return (_RENDER_KIND_RANK.get(spec.kind, 20), order[spec.id])


def _alpha_override(style_overrides, spec):
    values = dict((style_overrides or {}).get(spec.id) or {})
    value = values.get("alpha")
    return default_layer_alpha(spec) if value is None else float(value)


def _layer_style_overrides(style_overrides, spec):
    return dict((style_overrides or {}).get(spec.id) or {})


def _shift_indices(mapping, insert_at, count):
    return {
        layer_id: [
            index + count if index >= insert_at else index
            for index in indices
        ]
        for layer_id, indices in mapping.items()
    }


def _trace_meta(trace):
    meta = getattr(trace, "meta", None)
    return meta if isinstance(meta, dict) else {}


def _insertion_index(spec, catalog, layer_traces, trace_count):
    target_key = _layer_render_key(spec, catalog)
    for other in catalog.layers:
        indices = layer_traces.get(other.id)
        if indices and _layer_render_key(other, catalog) > target_key:
            return min(indices)
    return int(trace_count)


def visibility_patch(
    render_state,
    selected_layer_ids,
    catalog,
    *,
    active_layer_id=None,
    style_overrides=None,
):
    """Patch visibility and insert newly loaded traces in stable render order.

    Returns
    -------
    dash.Patch
        Partial figure update. Existing unrelated trace payloads are absent.
    list of str
        Per-layer load messages; one failure does not clear other layers.
    dict
        Updated trace-index bookkeeping, including colorbar-capable traces.
    """

    from dash import Patch

    selected = set(selected_layer_ids or ())
    patch = Patch()
    messages = []
    render_state = render_state or {}
    trace_count = int(render_state.get("trace_count", 1))
    layer_traces = {
        str(layer_id): [int(index) for index in indices]
        for layer_id, indices in (
            render_state.get("layer_traces") or {}
        ).items()
    }
    colorbar_traces = {
        str(layer_id): [int(index) for index in indices]
        for layer_id, indices in (
            render_state.get("colorbar_traces") or {}
        ).items()
    }

    for spec in catalog.layers:
        indices = layer_traces.get(spec.id, [])
        if spec.id not in selected:
            for index in indices:
                patch["data"][index]["visible"] = False
            continue
        if indices:
            for index in indices:
                patch["data"][index]["visible"] = True
            continue
        try:
            payload = catalog.load(spec.id)
            traces = traces_for_payload(
                payload,
                show_colorbar=spec.id == active_layer_id,
                alpha=_alpha_override(style_overrides, spec),
                style_overrides=_layer_style_overrides(
                    style_overrides,
                    spec,
                ),
            )
            insert_at = _insertion_index(
                spec,
                catalog,
                layer_traces,
                trace_count,
            )
            layer_traces = _shift_indices(
                layer_traces,
                insert_at,
                len(traces),
            )
            colorbar_traces = _shift_indices(
                colorbar_traces,
                insert_at,
                len(traces),
            )
            indices = list(range(insert_at, insert_at + len(traces)))
            layer_traces[spec.id] = indices
            colorbar_traces[spec.id] = [
                index
                for index, trace in zip(indices, traces)
                if _trace_meta(trace).get("colorbar_capable")
            ]
            trace_count += len(traces)
            for offset, trace in enumerate(traces):
                trace.visible = True
                patch["data"].insert(
                    insert_at + offset,
                    trace.to_plotly_json(),
                )
            messages.append(f"Loaded {spec.name}.")
        except Exception as exc:
            messages.append(f"{spec.name}: {type(exc).__name__}: {exc}")
    updated_render_state = {
        "trace_count": trace_count,
        "layer_traces": layer_traces,
        "colorbar_traces": colorbar_traces,
    }
    return patch, messages, updated_render_state


def active_layer_patch(render_state, active_layer_id):
    """Show a colorbar only for the active layer."""

    from dash import Patch

    patch = Patch()
    for layer_id, indices in (
        (render_state or {}).get("colorbar_traces") or {}
    ).items():
        show = layer_id == active_layer_id
        for index in indices:
            patch["data"][int(index)]["marker"]["showscale"] = show
    return patch


def alpha_patch(render_state, layer_id, alpha):
    """Set runtime alpha for every loaded child trace of one layer."""

    from dash import Patch

    patch = Patch()
    indices = (
        (render_state or {}).get("layer_traces") or {}
    ).get(layer_id, ())
    for index in indices:
        patch["data"][int(index)]["opacity"] = float(alpha)
    return patch


def color_limits_patch(
    render_state,
    layer_id,
    payload,
    style_overrides,
):
    """Patch effective quantitative color limits for one loaded layer."""

    from dash import Patch

    patch = Patch()
    indices = (
        (render_state or {}).get("layer_traces") or {}
    ).get(layer_id, ())
    if not indices:
        return patch
    traces = traces_for_payload(
        payload,
        show_colorbar=True,
        style_overrides=style_overrides,
    )
    for index, trace in zip(indices, traces):
        meta = _trace_meta(trace)
        if not meta.get("colorbar_capable"):
            continue
        cmin = getattr(trace.marker, "cmin", None)
        cmax = getattr(trace.marker, "cmax", None)
        if cmin is not None and cmax is not None:
            patch["data"][int(index)]["marker"]["cmin"] = float(cmin)
            patch["data"][int(index)]["marker"]["cmax"] = float(cmax)
    return patch


def viewport_patch(region):
    """Return a layout-only patch that fits a lon/lat bounding box."""

    from dash import Patch

    center, zoom = view_for_region(region)
    patch = Patch()
    patch["layout"][map_layout_key()]["center"] = center
    patch["layout"][map_layout_key()]["zoom"] = zoom
    return patch, center, zoom


def viewport_from_relayout(relayout_data):
    """Extract small center/zoom/bearing/pitch state from Plotly relayout."""

    renderer_backend()
    prefix = "map"
    relayout_data = relayout_data or {}
    viewport = {}
    center = relayout_data.get(f"{prefix}.center")
    if isinstance(center, dict):
        viewport["center"] = {
            "lat": float(center["lat"]),
            "lon": float(center["lon"]),
        }
    for field in ("zoom", "bearing", "pitch"):
        key = f"{prefix}.{field}"
        if key in relayout_data:
            viewport[field] = float(relayout_data[key])
    return viewport


def build_initial_figure(
    project,
    catalog,
    *,
    active_layer_id=None,
    style_overrides=None,
):
    """Load visible layers in deterministic scientific display order."""

    figure = create_base_figure(project)
    messages = []
    visible = sorted(
        (spec for spec in catalog.layers if spec.visible),
        key=lambda spec: _layer_render_key(spec, catalog),
    )
    for spec in visible:
        try:
            add_payload(
                figure,
                catalog.load(spec.id),
                show_colorbar=spec.id == active_layer_id,
                alpha=_alpha_override(style_overrides, spec),
                style_overrides=_layer_style_overrides(
                    style_overrides,
                    spec,
                ),
            )
        except Exception as exc:
            messages.append(f"{spec.name}: {type(exc).__name__}: {exc}")
    layer_traces = {}
    colorbar_traces = {}
    for spec in catalog.layers:
        indices = layer_trace_indices(figure, spec.id)
        if not indices:
            continue
        layer_traces[spec.id] = indices
        colorbar_traces[spec.id] = [
            index
            for index in indices
            if (
                isinstance(getattr(figure.data[index], "meta", None), dict)
                and figure.data[index].meta.get("colorbar_capable")
            )
        ]
    render_state = {
        "trace_count": len(figure.data),
        "layer_traces": layer_traces,
        "colorbar_traces": colorbar_traces,
    }
    return figure, messages, render_state


def _inspector_text(click_data):
    if not click_data or not click_data.get("points"):
        return "Click a displayed feature to inspect its plotted attributes."
    point = dict(click_data["points"][0])
    keep = {
        key: value
        for key, value in point.items()
        if key in {"curveNumber", "pointIndex", "lon", "lat", "text"}
    }
    return json.dumps(keep, ensure_ascii=False, indent=2, default=str)


def _updated_state(state, **changes):
    values = {
        "basemap": state.basemap,
        "viewport": state.viewport,
        "visible_layer_ids": state.visible_layer_ids,
        "active_layer_id": state.active_layer_id,
        "style_overrides": state.style_overrides,
    }
    values.update(changes)
    return ViewerState(**values)


def create_app(project, *, catalog=None):
    """Create a session-isolated Dash viewer for a parsed project."""

    try:
        import dash
        from dash import ALL, Input, Output, Patch, State, no_update
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The ECAT map viewer requires the optional 'viewer' dependencies."
        ) from exc

    catalog = catalog or LayerCatalog(project.layers)
    visible_ids = tuple(layer.id for layer in project.layers if layer.visible)
    active_layer_id = default_active_layer_id(project.layers)
    initial_state = ViewerState(
        basemap=project.basemap,
        visible_layer_ids=visible_ids,
        active_layer_id=active_layer_id,
    )
    figure, initial_messages, initial_render_state = build_initial_figure(
        project,
        catalog,
        active_layer_id=active_layer_id,
    )
    groups = layer_groups(catalog.layers, project.region)

    app = dash.Dash(__name__)
    app.title = f"ECAT Map — {project.name}"
    app.layout = build_layout(
        project,
        catalog,
        figure,
        initial_state,
        initial_render_state,
        initial_messages,
        basemap_options(),
    )

    @app.callback(
        Output("viewer-map", "figure", allow_duplicate=True),
        Output("viewer-session-state", "data", allow_duplicate=True),
        Input("viewer-basemap", "value"),
        State("viewer-session-state", "data"),
        prevent_initial_call=True,
    )
    def _change_basemap(map_style, state_data):
        state = ViewerState.from_dict(state_data)
        updated = _updated_state(state, basemap=map_style)
        return basemap_patch(map_style), updated.to_dict()

    @app.callback(
        Output("viewer-map", "figure", allow_duplicate=True),
        Output("viewer-session-state", "data", allow_duplicate=True),
        Output("viewer-status", "children"),
        Output("viewer-render-state", "data"),
        Input({"type": "viewer-layer-group", "group": ALL}, "value"),
        State("viewer-session-state", "data"),
        State("viewer-render-state", "data"),
        prevent_initial_call=True,
    )
    def _change_layers(group_values, state_data, render_state):
        selected = flatten_group_values(group_values)
        state = ViewerState.from_dict(state_data)
        patch, messages, updated_render_state = visibility_patch(
            render_state,
            selected,
            catalog,
            active_layer_id=state.active_layer_id,
            style_overrides=state.style_overrides,
        )
        updated = _updated_state(
            state,
            visible_layer_ids=selected,
        )
        return (
            patch,
            updated.to_dict(),
            "\n".join(messages),
            updated_render_state,
        )

    @app.callback(
        Output(
            {"type": "viewer-layer-group", "group": ALL},
            "value",
        ),
        Input("viewer-hide-all", "n_clicks"),
        Input("viewer-solo-active", "n_clicks"),
        State("viewer-active-layer", "value"),
        prevent_initial_call=True,
    )
    def _layer_shortcuts(_hide_clicks, _solo_clicks, active):
        triggered = dash.ctx.triggered_id
        if triggered == "viewer-hide-all":
            return [[] for _group in groups]
        if triggered == "viewer-solo-active":
            return [
                [active] if any(layer.id == active for layer in group.layers) else []
                for group in groups
            ]
        return no_update

    @app.callback(
        Output("viewer-map", "figure", allow_duplicate=True),
        Output("viewer-session-state", "data", allow_duplicate=True),
        Output("viewer-alpha", "value"),
        Output("viewer-vmin", "value"),
        Output("viewer-vmax", "value"),
        Output("viewer-symmetry", "value"),
        Output("viewer-color-controls", "style"),
        Output("viewer-active-details", "children"),
        Input("viewer-active-layer", "value"),
        State("viewer-session-state", "data"),
        State("viewer-render-state", "data"),
        prevent_initial_call=True,
    )
    def _change_active_layer(active, state_data, render_state):
        state = ViewerState.from_dict(state_data)
        layer = catalog.get(active)
        override = dict(state.style_overrides.get(active) or {})
        alpha = float(
            override.get("alpha", default_layer_alpha(layer))
        )
        default_vmin, default_vmax, default_symmetry = default_color_controls(
            layer
        )
        vmin = override.get("vmin", default_vmin)
        vmax = override.get("vmax", default_vmax)
        symmetry = bool(override.get("symmetry", default_symmetry))
        color_controls_style = {
            "display": (
                "block" if supports_color_controls(layer) else "none"
            ),
            "marginTop": "8px",
        }
        updated = _updated_state(state, active_layer_id=active)
        return (
            active_layer_patch(render_state, active),
            updated.to_dict(),
            alpha,
            vmin,
            vmax,
            ["symmetric"] if symmetry else [],
            color_controls_style,
            layer_details_component(layer),
        )

    @app.callback(
        Output("viewer-map", "figure", allow_duplicate=True),
        Output("viewer-session-state", "data", allow_duplicate=True),
        Input("viewer-apply-alpha", "n_clicks"),
        State("viewer-alpha", "value"),
        State("viewer-active-layer", "value"),
        State("viewer-session-state", "data"),
        State("viewer-render-state", "data"),
        prevent_initial_call=True,
    )
    def _apply_alpha(_clicks, alpha, active, state_data, render_state):
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            return no_update, no_update
        state = ViewerState.from_dict(state_data)
        overrides = {
            layer_id: dict(values)
            for layer_id, values in state.style_overrides.items()
        }
        layer_values = dict(overrides.get(active) or {})
        layer_values["alpha"] = alpha
        overrides[active] = layer_values
        updated = _updated_state(state, style_overrides=overrides)
        return (
            alpha_patch(render_state, active, alpha),
            updated.to_dict(),
        )

    @app.callback(
        Output("viewer-map", "figure", allow_duplicate=True),
        Output("viewer-session-state", "data", allow_duplicate=True),
        Output("viewer-status", "children", allow_duplicate=True),
        Input("viewer-apply-color-limits", "n_clicks"),
        Input("viewer-reset-color-limits", "n_clicks"),
        State("viewer-vmin", "value"),
        State("viewer-vmax", "value"),
        State("viewer-symmetry", "value"),
        State("viewer-active-layer", "value"),
        State("viewer-session-state", "data"),
        State("viewer-render-state", "data"),
        prevent_initial_call=True,
    )
    def _apply_color_limits(
        _apply_clicks,
        _reset_clicks,
        vmin,
        vmax,
        symmetry_values,
        active,
        state_data,
        render_state,
    ):
        layer = catalog.get(active)
        if not supports_color_controls(layer):
            return no_update, no_update, "Active layer is not quantitative."
        reset = dash.ctx.triggered_id == "viewer-reset-color-limits"
        if not reset and ((vmin is None) != (vmax is None)):
            return (
                no_update,
                no_update,
                "Set both vmin and vmax, or reset to automatic limits.",
            )
        state = ViewerState.from_dict(state_data)
        overrides = {
            layer_id: dict(values)
            for layer_id, values in state.style_overrides.items()
        }
        layer_values = dict(overrides.get(active) or {})
        layer_values["symmetry"] = "symmetric" in (symmetry_values or ())
        if reset or vmin is None:
            layer_values.pop("vmin", None)
            layer_values.pop("vmax", None)
        else:
            layer_values["vmin"] = float(vmin)
            layer_values["vmax"] = float(vmax)
        try:
            payload = catalog.load(active)
            patch = color_limits_patch(
                render_state,
                active,
                payload,
                layer_values,
            )
        except (TypeError, ValueError) as exc:
            return no_update, no_update, str(exc)
        overrides[active] = layer_values
        updated = _updated_state(state, style_overrides=overrides)
        return patch, updated.to_dict(), ""

    @app.callback(
        Output("viewer-map", "figure", allow_duplicate=True),
        Output("viewer-session-state", "data", allow_duplicate=True),
        Output("viewer-status", "children", allow_duplicate=True),
        Input("viewer-fit-active", "n_clicks"),
        State("viewer-active-layer", "value"),
        State("viewer-session-state", "data"),
        prevent_initial_call=True,
    )
    def _fit_active_layer(_clicks, active, state_data):
        try:
            payload = catalog.load(active)
            if payload.metadata.bbox is None:
                raise ValueError("the layer does not declare a finite extent")
            patch, center, zoom = viewport_patch(payload.metadata.bbox)
        except Exception as exc:
            return (
                Patch(),
                no_update,
                f"Cannot fit {active}: {type(exc).__name__}: {exc}",
            )
        state = ViewerState.from_dict(state_data)
        viewport = dict(state.viewport)
        viewport.update({"center": center, "zoom": zoom})
        updated = _updated_state(state, viewport=viewport)
        return patch, updated.to_dict(), ""

    @app.callback(
        Output("viewer-session-state", "data", allow_duplicate=True),
        Input("viewer-map", "relayoutData"),
        State("viewer-session-state", "data"),
        prevent_initial_call=True,
    )
    def _remember_viewport(relayout_data, state_data):
        viewport = viewport_from_relayout(relayout_data)
        if not viewport:
            return Patch()
        state = ViewerState.from_dict(state_data)
        updated_viewport = dict(state.viewport)
        updated_viewport.update(viewport)
        return _updated_state(
            state,
            viewport=updated_viewport,
        ).to_dict()

    @app.callback(
        Output("viewer-inspector", "children"),
        Input("viewer-map", "clickData"),
    )
    def _inspect_feature(click_data):
        return _inspector_text(click_data)

    return app


__all__ = [
    "BASEMAPS",
    "active_layer_patch",
    "basemap_options",
    "basemap_patch",
    "build_initial_figure",
    "create_app",
    "alpha_patch",
    "color_limits_patch",
    "viewport_from_relayout",
    "viewport_patch",
    "visibility_patch",
]
