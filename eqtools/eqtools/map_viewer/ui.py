"""Dash layout helpers for the ECAT research map viewer."""

from dataclasses import dataclass

from .backgrounds import (
    background_info_for_layer,
    background_matches_region,
)


@dataclass(frozen=True)
class LayerGroup:
    """One user-facing group of immutable layer declarations."""

    key: str
    label: str
    layers: tuple


_GROUP_ORDER = (
    ("project", "Project data"),
    ("global_context", "Global tectonic context"),
    ("regional_faults", "Regional fault data"),
    ("regional_blocks", "Regional blocks"),
    ("gnss", "GNSS velocity fields"),
    ("other_background", "Other packaged backgrounds"),
)
_QUANTITATIVE_LAYER_KINDS = {
    "earthquake_catalog",
    "observation_grid",
    "raster",
    "csi_varres",
}


def layer_groups(layers, region=None):
    """Group layers and put study-region matches first within each group."""

    grouped = {key: [] for key, _ in _GROUP_ORDER}
    labels = dict(_GROUP_ORDER)
    for layer in layers:
        info = background_info_for_layer(layer)
        if info is None:
            key = "project"
        else:
            key = info.group if info.group in grouped else "other_background"
            labels[key] = info.group_label
        grouped[key].append(layer)
    return tuple(
        LayerGroup(
            key,
            labels[key],
            tuple(
                sorted(
                    grouped[key],
                    key=lambda layer: (
                        not background_matches_region(layer, region),
                    ),
                )
            ),
        )
        for key, _ in _GROUP_ORDER
        if grouped[key]
    )


def flatten_group_values(values):
    """Return stable layer ids from grouped Checklist values."""

    selected = []
    for group_values in values or ():
        selected.extend(group_values or ())
    return tuple(dict.fromkeys(str(layer_id) for layer_id in selected))


def default_active_layer_id(layers):
    """Prefer the first visible layer, otherwise the first declared layer."""

    layers = tuple(layers)
    for layer in layers:
        if layer.visible:
            return layer.id
    return layers[0].id if layers else None


def default_layer_alpha(layer):
    """Return the renderer's effective default alpha for one layer."""

    if "alpha" in layer.style:
        return float(layer.style["alpha"])
    if layer.kind in {"earthquake_catalog", "observation_grid", "raster"}:
        return 0.8
    return 0.85


def supports_color_controls(layer):
    """Return whether one layer has a quantitative color mapping."""

    return layer is not None and layer.kind in _QUANTITATIVE_LAYER_KINDS


def default_color_controls(layer):
    """Return configured vmin, vmax, and automatic symmetry."""

    if not supports_color_controls(layer):
        return None, None, False
    return (
        layer.style.get("vmin"),
        layer.style.get("vmax"),
        bool(layer.style.get("symmetry", False)),
    )


def layer_details_component(layer):
    """Build the compact metadata card for the active layer."""

    from dash import html

    info = background_info_for_layer(layer)
    rows = [
        ("Kind", layer.kind),
        ("Source", str(layer.source)),
        ("Variable", layer.variable or "n/a"),
        ("Mask", layer.mask or "n/a"),
        ("Format", layer.format or "inferred"),
        ("Data type", layer.data_type or "n/a"),
    ]
    if info is not None:
        rows[0:0] = [
            ("Scope", info.scope),
            ("Role", info.description),
            ("Citation", info.citation),
            ("Units", info.units or "not applicable / not declared"),
            ("Reference frame", info.reference_frame or "not applicable"),
        ]
    children = [
        html.Strong(layer.name),
        *[
            html.Div(
                [
                    html.Span(
                        f"{name}: ",
                        style={"fontWeight": "600"},
                    ),
                    html.Span(value),
                ],
                style={
                    "fontSize": "11px",
                    "marginTop": "4px",
                    "overflowWrap": "anywhere",
                },
            )
            for name, value in rows
        ],
    ]
    if info is not None and info.url:
        children.append(
            html.A(
                "Source / citation page",
                href=info.url,
                target="_blank",
                rel="noopener noreferrer",
                style={"display": "inline-block", "fontSize": "11px", "marginTop": "6px"},
            )
        )
    return html.Div(children)


def _checklist_label(layer, region=None):
    from dash import html

    info = background_info_for_layer(layer)
    scope = info.scope if info is not None else layer.kind
    badges = [
        html.Span(
            scope,
            style={
                "display": "inline-block",
                "marginLeft": "6px",
                "padding": "1px 5px",
                "borderRadius": "8px",
                "background": "#e8edf3",
                "color": "#445",
                "fontSize": "9px",
            },
        )
    ]
    if (
        info is not None
        and info.group != "global_context"
        and background_matches_region(layer, region)
    ):
        badges.append(
            html.Span(
                "study region",
                style={
                    "display": "inline-block",
                    "marginLeft": "4px",
                    "padding": "1px 5px",
                    "borderRadius": "8px",
                    "background": "#dff3df",
                    "color": "#275b27",
                    "fontSize": "9px",
                },
            )
        )
    return html.Div(
        [html.Span(layer.name), *badges],
        style={"display": "inline"},
    )


def build_layout(
    project,
    catalog,
    figure,
    initial_state,
    initial_render_state,
    initial_messages,
    basemap_options,
):
    """Build the viewer layout while keeping callback wiring in ``app.py``."""

    from dash import dcc, html

    groups = layer_groups(catalog.layers, project.region)
    visible = set(initial_state.visible_layer_ids)
    active_layer_id = initial_state.active_layer_id
    active_layer = (
        catalog.get(active_layer_id)
        if active_layer_id is not None
        else None
    )
    active_alpha = (
        default_layer_alpha(active_layer)
        if active_layer is not None
        else 0.85
    )
    active_vmin, active_vmax, active_symmetry = default_color_controls(
        active_layer
    )
    color_controls_style = {
        "display": "block" if supports_color_controls(active_layer) else "none",
        "marginTop": "8px",
    }
    layer_checklists = []
    for group in groups:
        layer_checklists.append(
            html.Div(
                [
                    html.H4(
                        group.label,
                        style={"fontSize": "12px", "margin": "12px 0 4px"},
                    ),
                    dcc.Checklist(
                        id={"type": "viewer-layer-group", "group": group.key},
                        options=[
                            {
                                "label": _checklist_label(
                                    layer,
                                    project.region,
                                ),
                                "value": layer.id,
                            }
                            for layer in group.layers
                        ],
                        value=[
                            layer.id
                            for layer in group.layers
                            if layer.id in visible
                        ],
                        labelStyle={"display": "block", "margin": "5px 0"},
                    ),
                ]
            )
        )

    return html.Div(
        [
            dcc.Store(
                id="viewer-session-state",
                storage_type="memory",
                data=initial_state.to_dict(),
            ),
            dcc.Store(
                id="viewer-render-state",
                storage_type="memory",
                data=initial_render_state,
            ),
            html.Div(
                [
                    html.H3(project.name, style={"margin": "0 0 12px 0"}),
                    html.Label("Basemap"),
                    dcc.Dropdown(
                        id="viewer-basemap",
                        options=basemap_options,
                        value=project.basemap,
                        clearable=False,
                    ),
                    html.Div(
                        (
                            "Satellite, terrain and street basemaps require "
                            "network access; scientific overlays remain local."
                        ),
                        style={"fontSize": "10px", "color": "#666", "marginTop": "4px"},
                    ),
                    html.Label(
                        "Layers",
                        style={"display": "block", "marginTop": "14px"},
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Hide all",
                                id="viewer-hide-all",
                                n_clicks=0,
                                style={"marginRight": "6px"},
                            ),
                            html.Button(
                                "Show active only",
                                id="viewer-solo-active",
                                n_clicks=0,
                            ),
                        ],
                        style={"margin": "6px 0"},
                    ),
                    *layer_checklists,
                    html.H4(
                        "Active layer",
                        style={"fontSize": "12px", "margin": "14px 0 4px"},
                    ),
                    dcc.Dropdown(
                        id="viewer-active-layer",
                        options=[
                            {"label": layer.name, "value": layer.id}
                            for layer in catalog.layers
                        ],
                        value=active_layer_id,
                        clearable=False,
                    ),
                    html.Div(
                        (
                            "The active layer owns the visible colorbar and "
                            "runtime display controls; it need not be visible."
                        ),
                        style={"fontSize": "10px", "color": "#666", "marginTop": "4px"},
                    ),
                    html.Label(
                        "Alpha",
                        style={"display": "block", "fontSize": "11px", "marginTop": "8px"},
                    ),
                    dcc.Slider(
                        id="viewer-alpha",
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        value=active_alpha,
                        marks={0.0: "0", 0.5: "0.5", 1.0: "1"},
                        tooltip={"placement": "bottom"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Color limits (display units)",
                                style={"fontSize": "11px"},
                            ),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="viewer-vmin",
                                        type="number",
                                        value=active_vmin,
                                        placeholder="auto min",
                                        debounce=True,
                                        style={"width": "47%"},
                                    ),
                                    dcc.Input(
                                        id="viewer-vmax",
                                        type="number",
                                        value=active_vmax,
                                        placeholder="auto max",
                                        debounce=True,
                                        style={
                                            "width": "47%",
                                            "marginLeft": "6%",
                                        },
                                    ),
                                ],
                            ),
                            dcc.Checklist(
                                id="viewer-symmetry",
                                options=[
                                    {
                                        "label": " Symmetric automatic range",
                                        "value": "symmetric",
                                    }
                                ],
                                value=(
                                    ["symmetric"] if active_symmetry else []
                                ),
                                style={"fontSize": "11px", "marginTop": "4px"},
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Apply color limits",
                                        id="viewer-apply-color-limits",
                                        n_clicks=0,
                                        style={"marginRight": "6px"},
                                    ),
                                    html.Button(
                                        "Reset to auto",
                                        id="viewer-reset-color-limits",
                                        n_clicks=0,
                                    ),
                                ],
                                style={"marginTop": "5px"},
                            ),
                            html.Div(
                                (
                                    "Explicit vmin/vmax win over symmetry and "
                                    "do not alter source values."
                                ),
                                style={
                                    "fontSize": "10px",
                                    "color": "#666",
                                    "marginTop": "3px",
                                },
                            ),
                        ],
                        id="viewer-color-controls",
                        style=color_controls_style,
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Apply alpha",
                                id="viewer-apply-alpha",
                                n_clicks=0,
                                style={"marginRight": "6px"},
                            ),
                            html.Button(
                                "Fit active layer",
                                id="viewer-fit-active",
                                n_clicks=0,
                            ),
                        ],
                        style={"marginTop": "8px"},
                    ),
                    html.Details(
                        [
                            html.Summary("Active layer metadata"),
                            html.Div(
                                (
                                    layer_details_component(active_layer)
                                    if active_layer is not None
                                    else "No layer is available."
                                ),
                                id="viewer-active-details",
                                style={"marginTop": "6px"},
                            ),
                        ],
                        open=True,
                        style={"fontSize": "12px", "marginTop": "10px"},
                    ),
                    html.Div(
                        "Hidden layers are parsed only when first shown.",
                        style={"fontSize": "10px", "color": "#666", "marginTop": "12px"},
                    ),
                    html.Pre(
                        "\n".join(initial_messages),
                        id="viewer-status",
                        style={
                            "whiteSpace": "pre-wrap",
                            "fontSize": "11px",
                            "color": "#8b0000",
                        },
                    ),
                ],
                style={
                    "width": "350px",
                    "padding": "14px",
                    "overflowY": "auto",
                    "borderRight": "1px solid #ddd",
                    "background": "#fafafa",
                },
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="viewer-map",
                        figure=figure,
                        config={
                            "displaylogo": False,
                            "scrollZoom": True,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": project.name,
                            },
                        },
                        style={"height": "100vh"},
                    ),
                    html.Pre(
                        id="viewer-inspector",
                        children=(
                            "Click a displayed feature to inspect its plotted "
                            "attributes."
                        ),
                        style={
                            "position": "absolute",
                            "right": "12px",
                            "top": "12px",
                            "maxWidth": "360px",
                            "maxHeight": "40vh",
                            "overflow": "auto",
                            "background": "rgba(255,255,255,0.92)",
                            "padding": "10px",
                            "fontSize": "11px",
                            "border": "1px solid #ddd",
                        },
                    ),
                ],
                style={"flex": "1", "position": "relative"},
            ),
        ],
        style={
            "display": "flex",
            "width": "100%",
            "height": "100vh",
            "fontFamily": "Arial, sans-serif",
        },
    )


__all__ = [
    "LayerGroup",
    "build_layout",
    "default_active_layer_id",
    "default_layer_alpha",
    "default_color_controls",
    "flatten_group_values",
    "layer_details_component",
    "layer_groups",
    "supports_color_controls",
]
