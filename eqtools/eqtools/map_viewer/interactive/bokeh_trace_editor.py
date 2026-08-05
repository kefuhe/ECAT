"""Optional local Bokeh UI for one ECAT working trace."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np

from .coordinates import lonlat_to_web_mercator, web_mercator_to_lonlat
from .display import color_limits, display_values, matplotlib_palette
from .models import TraceEditorSession
from .raster import QuadmeshRasterizer, supports_quadmesh
from .trace_io import save_trace


_BOKEH_INSTALL_HINT = (
    'Trace editing requires the optional interaction dependencies. Install '
    'them from the ECAT source tree with: python -m pip install -e '
    '".[interaction]"'
)

_BASEMAP_PROVIDERS = {
    "gray": "CartoDB.Positron",
    "street": "OpenStreetMap.Mapnik",
    "terrain": "Esri.WorldTopoMap",
    "satellite": "Esri.WorldImagery",
}


def require_bokeh():
    """Import Bokeh lazily and raise one actionable dependency error."""

    try:
        import bokeh
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_BOKEH_INSTALL_HINT) from exc
    version_parts = str(bokeh.__version__).split(".")
    version = tuple(int(part) for part in version_parts[:2])
    if version < (3, 6) or version >= (4, 0):
        raise RuntimeError(
            "ECAT trace editing requires Bokeh >=3.6,<4; installed version is "
            f"{bokeh.__version__}."
        )
    return bokeh


def _display_indices(background, max_points=80_000):
    """Return stable display-only indices without changing source values."""

    valid = np.asarray(background.valid_mask, dtype=bool)
    if valid.ndim == 2:
        valid_count = int(np.count_nonzero(valid))
        if valid_count <= max_points:
            return np.flatnonzero(valid.reshape(-1))
        stride = max(1, int(np.ceil(np.sqrt(valid_count / max_points))))
        sampled = np.zeros(valid.shape, dtype=bool)
        sampled[::stride, ::stride] = True
        indices = np.flatnonzero((valid & sampled).reshape(-1))
        if len(indices) <= max_points:
            return indices
        positions = np.linspace(0, len(indices) - 1, max_points, dtype=int)
        return indices[positions]
    indices = np.flatnonzero(valid.reshape(-1))
    if len(indices) <= max_points:
        return indices
    positions = np.linspace(0, len(indices) - 1, max_points, dtype=int)
    return indices[positions]


def _point_source_data(background):
    """Return a bounded exact-value point source for hover/fallback display."""

    indices = _display_indices(background)
    longitude = background.longitude.reshape(-1)[indices]
    latitude = background.latitude.reshape(-1)[indices]
    values, _ = display_values(background)
    x, y = lonlat_to_web_mercator(longitude, latitude)
    return {
        "x": x,
        "y": y,
        "longitude": longitude,
        "latitude": latitude,
        "value": values.reshape(-1)[indices],
    }


def _padded_point_ranges(point_data, *, fraction=0.02):
    """Return initial Web Mercator ranges for a non-structured background."""

    x = np.asarray(point_data["x"], dtype=float)
    y = np.asarray(point_data["y"], dtype=float)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    xpad = max((xmax - xmin) * float(fraction), 1.0)
    ypad = max((ymax - ymin) * float(fraction), 1.0)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


class DynamicRasterController:
    """Debounced Bokeh-server bridge for zoom-aware quadmesh frames."""

    def __init__(
        self,
        document,
        plot,
        source,
        rasterizer,
        *,
        width=1200,
        height=800,
        delay_ms=250,
        on_error=None,
    ):
        self.document = document
        self.plot = plot
        self.source = source
        self.rasterizer = rasterizer
        self.width = int(width)
        self.height = int(height)
        self.delay_ms = int(delay_ms)
        self.on_error = on_error
        self._pending = None
        self.last_frame = None

    def render_now(self):
        """Render the current plot range immediately and update the image source."""

        self._pending = None
        try:
            frame = self.rasterizer.render(
                x_range=(self.plot.x_range.start, self.plot.x_range.end),
                y_range=(self.plot.y_range.start, self.plot.y_range.end),
                width=self.width,
                height=self.height,
            )
            self.source.data = frame.source_data()
            self.last_frame = frame
            return frame
        except Exception as exc:  # Bokeh callback boundary; report without exit.
            if self.on_error is not None:
                self.on_error(exc)
                return None
            raise

    def schedule(self, _attr, _old, _new):
        """Debounce range events so interactive pan/zoom does not queue work."""

        if self._pending is not None:
            try:
                self.document.remove_timeout_callback(self._pending)
            except ValueError:
                pass
        self._pending = self.document.add_timeout_callback(
            self.render_now,
            self.delay_ms,
        )


def _working_source_data(workspace):
    if workspace.active is None or len(workspace.active.coordinates) == 0:
        return {"xs": [], "ys": []}
    coordinates = workspace.active.coordinates
    x, y = lonlat_to_web_mercator(coordinates[:, 0], coordinates[:, 1])
    return {"xs": [x.tolist()], "ys": [y.tolist()]}


def _table_source_data(workspace):
    if workspace.active is None:
        return {"index": [], "longitude": [], "latitude": []}
    coordinates = workspace.active.coordinates
    return {
        "index": list(range(len(coordinates))),
        "longitude": [float(value) for value in coordinates[:, 0]],
        "latitude": [float(value) for value in coordinates[:, 1]],
    }


def _reference_source_data(references):
    xs = []
    ys = []
    names = []
    ids = []
    for reference in references:
        x, y = lonlat_to_web_mercator(
            reference.coordinates[:, 0],
            reference.coordinates[:, 1],
        )
        xs.append(x.tolist())
        ys.append(y.tolist())
        names.append(reference.name)
        ids.append(reference.id)
    return {"xs": xs, "ys": ys, "name": names, "reference_id": ids}


@dataclass
class TraceEditorView:
    """Named Bokeh models returned for focused UI tests and extensions."""

    plot: object
    working_source: object
    node_source: object
    table_source: object
    reference_source: object
    status: object
    dirty_status: object
    mode: object
    output_path: object
    action_source: object
    background_source: object
    background_renderer: object
    color_mapper: object
    color_bar: object
    basemap_select: object
    display_options: object
    opacity: object
    buttons: dict
    tools: dict


def build_trace_editor_document(document, session):
    """Populate one Bokeh document from a backend-neutral editor session."""

    require_bokeh()
    if not isinstance(session, TraceEditorSession):
        raise TypeError("build_trace_editor_document expects TraceEditorSession.")

    from bokeh.layouts import column, row
    from bokeh.models import (
        Button,
        CheckboxGroup,
        ColorBar,
        ColumnDataSource,
        CustomJS,
        DataTable,
        Div,
        HoverTool,
        LinearColorMapper,
        NumberFormatter,
        PanTool,
        PolyDrawTool,
        PolyEditTool,
        RadioButtonGroup,
        Range1d,
        ResetTool,
        SaveTool,
        Select,
        Slider,
        TableColumn,
        TextInput,
        WheelZoomTool,
    )
    from bokeh.events import DocumentReady
    from bokeh.plotting import figure

    background = session.background
    workspace = session.workspace
    point_data = _point_source_data(background)
    lower, upper = color_limits(background)
    palette = matplotlib_palette(background.style.get("cmap", "RdBu_r"))
    alpha = float(background.style.get("alpha", 0.82))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Trace-editor observation alpha must be within [0, 1].")

    rasterizer = None
    initial_frame = None
    if supports_quadmesh(background):
        rasterizer = QuadmeshRasterizer(
            background,
            palette=palette,
            low=lower,
            high=upper,
        )
        x_range, y_range = rasterizer.padded_ranges()
        initial_frame = rasterizer.render(
            x_range=x_range,
            y_range=y_range,
            width=1200,
            height=800,
        )
        background_source = ColumnDataSource(initial_frame.source_data())
    else:
        x_range, y_range = _padded_point_ranges(point_data)
        background_source = ColumnDataSource(point_data)

    hover_source = ColumnDataSource(point_data)
    reference_source = ColumnDataSource(
        _reference_source_data(workspace.references)
    )
    working_source = ColumnDataSource(_working_source_data(workspace))
    # PolyEditTool owns this transient source. It is populated only after
    # the user selects a working line and is never a second geometry store.
    node_source = ColumnDataSource({"x": [], "y": []})
    table_source = ColumnDataSource(_table_source_data(workspace))
    action_source = ColumnDataSource({"action": [""], "token": [0]})
    status = Div(text="Ready. Reference traces are read-only.")

    plot = figure(
        title=session.title,
        x_axis_type="mercator",
        y_axis_type="mercator",
        x_range=Range1d(*x_range),
        y_range=Range1d(*y_range),
        sizing_mode="stretch_both",
        min_height=620,
        tools=[],
        active_scroll=None,
    )
    requested_basemap = str(
        background.style.get("basemap", "gray")
    ).strip().lower()
    if requested_basemap not in {*_BASEMAP_PROVIDERS, "none"}:
        choices = ", ".join([*_BASEMAP_PROVIDERS, "none"])
        raise ValueError(
            f"Unknown trace-editor basemap {requested_basemap!r}; "
            f"choose one of: {choices}."
        )
    tile_renderers = {}
    for key, provider in _BASEMAP_PROVIDERS.items():
        renderer = plot.add_tile(provider)
        renderer.visible = key == requested_basemap
        tile_renderers[key] = renderer

    color_mapper = LinearColorMapper(
        palette=palette,
        low=lower,
        high=upper,
        nan_color="#00000000",
    )
    if rasterizer is not None:
        background_renderer = plot.image_rgba(
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            source=background_source,
            global_alpha=alpha,
            name="observation-background",
        )
        hover_renderer = plot.scatter(
            x="x",
            y="y",
            source=hover_source,
            marker="square",
            size=8,
            fill_alpha=0.0,
            line_alpha=0.0,
            name="observation-hover",
        )
    else:
        background_renderer = plot.scatter(
            x="x",
            y="y",
            source=background_source,
            marker="square",
            size=float(background.style.get("marker_size", 5.0)),
            color={"field": "value", "transform": color_mapper},
            alpha=alpha,
            line_alpha=0.0,
            name="observation-background",
        )
        hover_renderer = background_renderer

    display_unit = background.style.get("display_unit") or background.units or ""
    color_bar = ColorBar(
        color_mapper=color_mapper,
        title=(
            f"{background.variable or background.name} ({display_unit})"
            if display_unit
            else (background.variable or background.name)
        ),
    )
    plot.add_layout(color_bar, "right")
    plot.add_tools(
        HoverTool(
            renderers=[hover_renderer],
            tooltips=[
                ("lon", "@longitude{0.000000}"),
                ("lat", "@latitude{0.000000}"),
                ("value", "@value{0.000000}"),
            ],
        )
    )

    raster_controller = None
    if rasterizer is not None:
        def _raster_error(exc):
            status.text = (
                '<span style="color:#b30000">Background redraw failed: '
                f"{escape(str(exc))}</span>"
            )

        raster_controller = DynamicRasterController(
            document,
            plot,
            background_source,
            rasterizer,
            width=1200,
            height=800,
            delay_ms=250,
            on_error=_raster_error,
        )
        raster_controller.last_frame = initial_frame
        for axis_range in (plot.x_range, plot.y_range):
            axis_range.on_change("start", raster_controller.schedule)
            axis_range.on_change("end", raster_controller.schedule)

    plot.multi_line(
        xs="xs",
        ys="ys",
        source=reference_source,
        line_color="#555555",
        line_dash="dashed",
        line_width=2,
        line_alpha=0.9,
        name="reference-traces",
    )
    working_renderer = plot.multi_line(
        xs="xs",
        ys="ys",
        source=working_source,
        line_color="#ffcc00",
        line_width=3,
        line_alpha=1.0,
        name="working-trace",
    )
    node_renderer = plot.scatter(
        x="x",
        y="y",
        source=node_source,
        marker="circle",
        size=9,
        fill_color="#ffffff",
        fill_alpha=0.9,
        line_color="#d94801",
        line_width=2,
        selection_fill_color="#00ffff",
        selection_line_color="#000000",
        name="working-vertices",
    )

    pan_tool = PanTool()
    wheel_tool = WheelZoomTool()
    draw_tool = PolyDrawTool(
        renderers=[working_renderer],
        vertex_renderer=node_renderer,
        num_objects=1,
        description="Draw one connected working trace",
    )
    edit_tool = PolyEditTool(
        renderers=[working_renderer],
        vertex_renderer=node_renderer,
        description="Add, move, or delete working-trace vertices",
    )
    plot.add_tools(
        pan_tool,
        wheel_tool,
        draw_tool,
        edit_tool,
        ResetTool(),
        SaveTool(),
    )
    plot.toolbar.active_drag = pan_tool
    plot.toolbar.active_scroll = wheel_tool
    plot.toolbar.active_tap = None

    table = DataTable(
        source=table_source,
        columns=[
            TableColumn(field="index", title="Index"),
            TableColumn(
                field="longitude",
                title="Longitude",
                formatter=NumberFormatter(format="0.00000000"),
            ),
            TableColumn(
                field="latitude",
                title="Latitude",
                formatter=NumberFormatter(format="0.00000000"),
            ),
        ],
        height=280,
        sizing_mode="stretch_width",
        selectable=True,
        index_position=None,
    )
    node_source.selected.js_on_change(
        "indices",
        CustomJS(
            args={"other": table_source},
            code=(
                "const a = JSON.stringify(cb_obj.indices);"
                "const b = JSON.stringify(other.selected.indices);"
                "if (a !== b) other.selected.indices = cb_obj.indices;"
            ),
        ),
    )
    table_source.selected.js_on_change(
        "indices",
        CustomJS(
            args={"other": node_source},
            code=(
                "const a = JSON.stringify(cb_obj.indices);"
                "const b = JSON.stringify(other.selected.indices);"
                "if (a !== b) other.selected.indices = cb_obj.indices;"
            ),
        ),
    )

    reference_options = [
        (reference.id, reference.name) for reference in workspace.references
    ]
    reference_select = Select(
        title="Reference trace",
        options=reference_options,
        value=reference_options[0][0] if reference_options else "",
        disabled=not reference_options,
    )
    basemap_select = Select(
        title="Basemap",
        options=[
            ("gray", "Gray"),
            ("street", "Street"),
            ("terrain", "Terrain"),
            ("satellite", "Satellite"),
            ("none", "None"),
        ],
        value=requested_basemap,
    )
    display_options = CheckboxGroup(
        labels=["Show basemap", "Show observation", "Show colorbar"],
        active=[0, 1, 2],
    )
    opacity = Slider(
        title="Observation opacity",
        start=0.0,
        end=1.0,
        value=alpha,
        step=0.05,
    )
    mode = RadioButtonGroup(labels=["Browse", "Draw", "Edit"], active=0)
    output_path = TextInput(
        title="Save adjusted trace as",
        value=str(session.output_path),
    )
    overwrite = CheckboxGroup(labels=["Allow explicit overwrite"], active=[])
    dirty_status = Div(text="No active working trace.")
    instructions = Div(
        text=(
            "<b>Modes:</b> Browse pans or zooms the map; Draw creates one "
            "connected trace. Copying a reference enters Edit; click the "
            "yellow working trace once to show its editable vertices. Move, "
            "insert, or select vertices in Edit. Use Delete selected vertex "
            "(or the Delete key). Save As never overwrites unless explicitly "
            "enabled."
        )
    )
    buttons = {
        "new": Button(label="New trace", button_type="primary"),
        "copy": Button(
            label="Copy reference as working",
            disabled=not reference_options,
        ),
        "finish": Button(label="Finish drawing / Browse"),
        "clear": Button(label="Clear vertices"),
        "delete_vertex": Button(label="Delete selected vertex"),
        "delete": Button(label="Delete working trace", button_type="danger"),
        "copy_selected": Button(label="Copy selected coordinate"),
        "copy_all": Button(label="Copy all coordinates"),
        "undo": Button(label="Undo", disabled=not workspace.history.can_undo),
        "redo": Button(label="Redo", disabled=not workspace.history.can_redo),
        "validate": Button(label="Validate"),
        "save": Button(label="Save As", button_type="success"),
    }
    syncing = {"active": False}

    def _set_status(message, *, error=False):
        color = "#b30000" if error else "#1b5e20"
        status.text = f'<span style="color:{color}">{escape(str(message))}</span>'

    def _sync_display(_attr, _old, _new):
        show_basemap = 0 in display_options.active
        show_observation = 1 in display_options.active
        show_colorbar = 2 in display_options.active
        for key, renderer in tile_renderers.items():
            renderer.visible = (
                show_basemap
                and basemap_select.value != "none"
                and key == basemap_select.value
            )
        background_renderer.visible = show_observation
        hover_renderer.visible = show_observation
        color_bar.visible = show_observation and show_colorbar
        opacity.disabled = not show_observation

    def _opacity_changed(_attr, _old, value):
        value = float(value)
        glyph = background_renderer.glyph
        if hasattr(glyph, "global_alpha"):
            glyph.global_alpha = value
        else:
            glyph.fill_alpha = value

    basemap_select.on_change("value", _sync_display)
    display_options.on_change("active", _sync_display)
    opacity.on_change("value", _opacity_changed)
    _sync_display(None, None, None)

    def _refresh_controls():
        buttons["undo"].disabled = not workspace.history.can_undo
        buttons["redo"].disabled = not workspace.history.can_redo
        disabled = workspace.active is None
        for name in (
            "finish",
            "clear",
            "delete_vertex",
            "delete",
            "copy_selected",
            "copy_all",
            "validate",
            "save",
        ):
            buttons[name].disabled = disabled
        if workspace.active is None:
            dirty_status.text = "No active working trace."
        else:
            state = "unsaved changes" if workspace.active.dirty else "saved"
            dirty_status.text = (
                f"Working trace: {escape(workspace.active.name)} | "
                f"{len(workspace.active.coordinates)} vertices | {state}."
            )

    def _sync_from_workspace(*, working=True, table=True):
        """Project authoritative workspace state into persistent UI sources."""

        syncing["active"] = True
        try:
            if working:
                working_source.data = _working_source_data(workspace)
            if table:
                table_source.data = _table_source_data(workspace)
        finally:
            syncing["active"] = False

    def _refresh():
        _sync_from_workspace()
        _refresh_controls()

    def _coordinates_match(coordinates):
        if workspace.active is None:
            return False
        current = np.asarray(workspace.active.coordinates, dtype=float)
        candidate = np.asarray(coordinates, dtype=float)
        return current.shape == candidate.shape and np.allclose(
            current,
            candidate,
            rtol=0.0,
            atol=1.0e-10,
        )

    def _commit_coordinates(coordinates):
        """Commit one completed working-line edit to the canonical workspace."""

        coordinates = np.asarray(coordinates, dtype=float)
        changed = not _coordinates_match(coordinates)
        if changed:
            workspace.replace_coordinates(coordinates)
        # Do not write back to working_source or node_source here. Bokeh owns
        # the current MultiLine/vertex array relationship during an edit.
        _sync_from_workspace(working=False, table=True)
        _refresh_controls()
        if changed:
            _set_status(
                f"Working trace updated: {len(coordinates)} vertices."
            )
        return changed

    def _coordinates_from_working_source(data):
        xs = list(data.get("xs") or [])
        ys = list(data.get("ys") or [])
        if len(xs) != len(ys) or len(xs) > 1:
            raise ValueError("The editor supports exactly one working polyline.")
        if not xs:
            return np.empty((0, 2), dtype=float)
        x_values = np.asarray(xs[0], dtype=float)
        y_values = np.asarray(ys[0], dtype=float)
        if x_values.shape != y_values.shape:
            raise ValueError("Working x/y vertex arrays are not aligned.")
        longitude_values, latitude_values = web_mercator_to_lonlat(
            x_values,
            y_values,
        )
        return np.column_stack((longitude_values, latitude_values))

    def _source_changed(_attr, _old, new):
        if syncing["active"]:
            return
        try:
            coordinates = _coordinates_from_working_source(new)
            _commit_coordinates(coordinates)
        except (TypeError, ValueError) as exc:
            _set_status(exc, error=True)
            _refresh()

    working_source.on_change("data", _source_changed)

    def _mode_changed(_attr, _old, active):
        if active == 0:
            plot.toolbar.active_drag = pan_tool
            plot.toolbar.active_tap = None
            _set_status("Browse mode: pan/zoom without adding vertices.")
        elif active == 1:
            if workspace.active is None:
                workspace.new_path()
                _refresh()
            plot.toolbar.active_drag = draw_tool
            plot.toolbar.active_tap = draw_tool
            _set_status("Draw mode: add one connected working trace.")
        else:
            if workspace.active is None:
                workspace.new_path()
                _refresh()
            plot.toolbar.active_drag = edit_tool
            plot.toolbar.active_tap = edit_tool
            _set_status(
                "Edit mode: click the yellow working trace to show vertices, "
                "then drag, insert, or select them."
            )

    mode.on_change("active", _mode_changed)

    def _new():
        workspace.new_path()
        _refresh()
        mode.active = 1
        _set_status("Created a new trace; Draw mode is active.")

    def _copy():
        try:
            workspace.copy_reference(reference_select.value)
            _refresh()
            mode.active = 2
            _set_status(
                "Reference copied; Edit mode is active. Click the yellow "
                "working trace to show editable vertices."
            )
        except (KeyError, ValueError) as exc:
            _set_status(exc, error=True)

    def _finish():
        mode.active = 0
        _set_status("Drawing finished; Browse mode is active.")

    def _clear():
        workspace.clear_path()
        _refresh()
        _set_status("Working vertices cleared; Undo can restore them.")

    def _delete_vertex():
        if workspace.active is None:
            _set_status("There is no active working trace.", error=True)
            return
        indices = sorted(
            {
                int(index)
                for index in (
                    node_source.selected.indices
                    or table_source.selected.indices
                )
                if 0 <= int(index) < len(workspace.active.coordinates)
            }
        )
        if not indices:
            _set_status("Select at least one vertex to delete.", error=True)
            return
        coordinates = np.delete(workspace.active.coordinates, indices, axis=0)
        workspace.replace_coordinates(coordinates)
        node_source.selected.indices = []
        table_source.selected.indices = []
        _refresh()
        _set_status(
            f"Deleted {len(indices)} selected vertex/vertices; "
            "the remaining trace was reconnected."
        )

    def _delete():
        workspace.delete_path()
        _refresh()
        _set_status("Working trace deleted; references remain unchanged.")

    def _undo():
        workspace.undo()
        _refresh()
        _set_status("Undo applied.")

    def _redo():
        workspace.redo()
        _refresh()
        _set_status("Redo applied.")

    def _validate():
        try:
            draft = workspace.validate_for_save()
            _set_status(f"Trace is valid: {len(draft.coordinates)} vertices.")
        except ValueError as exc:
            _set_status(exc, error=True)

    def _save():
        try:
            draft = workspace.validate_for_save()
            target = save_trace(
                Path(output_path.value),
                draft,
                overwrite=0 in overwrite.active,
            )
            workspace.mark_saved(target)
            _refresh()
            _set_status(f"Saved adjusted trace: {target}")
        except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
            _set_status(exc, error=True)

    buttons["new"].on_click(_new)
    buttons["copy"].on_click(_copy)
    buttons["finish"].on_click(_finish)
    buttons["clear"].on_click(_clear)
    buttons["delete_vertex"].on_click(_delete_vertex)
    buttons["delete"].on_click(_delete)
    buttons["undo"].on_click(_undo)
    buttons["redo"].on_click(_redo)
    buttons["validate"].on_click(_validate)
    buttons["save"].on_click(_save)

    copy_code = """
        const indices = selected_only ? table_source.selected.indices :
            Array.from({length: table_source.data.index.length}, (_, i) => i);
        if (indices.length === 0) {
            status.text = '<span style="color:#b30000">No coordinates selected.</span>';
            return;
        }
        const lines = indices.map((i) =>
            `${Number(table_source.data.longitude[i]).toFixed(8)} ` +
            `${Number(table_source.data.latitude[i]).toFixed(8)}`
        );
        navigator.clipboard.writeText(lines.join('\\n')).then(() => {
            status.text = '<span style="color:#1b5e20">Copied ' +
                `${lines.length} coordinate row(s).</span>`;
        }).catch((error) => {
            status.text = `<span style="color:#b30000">Clipboard failed: ${error}</span>`;
        });
    """
    buttons["copy_selected"].js_on_click(
        CustomJS(
            args={
                "table_source": table_source,
                "status": status,
                "selected_only": True,
            },
            code=copy_code,
        )
    )
    buttons["copy_all"].js_on_click(
        CustomJS(
            args={
                "table_source": table_source,
                "status": status,
                "selected_only": False,
            },
            code=copy_code,
        )
    )

    def _shortcut_changed(_attr, _old, data):
        action = str((data.get("action") or [""])[0])
        if action == "undo":
            _undo()
        elif action == "redo":
            _redo()
        elif action == "finish":
            _finish()
        elif action == "delete_vertex":
            _delete_vertex()

    action_source.on_change("data", _shortcut_changed)
    document.js_on_event(
        DocumentReady,
        CustomJS(
            args={"source": action_source},
            code="""
                window.__ecatTraceEditorHandlers = window.__ecatTraceEditorHandlers || {};
                const key = source.id;
                const previous = window.__ecatTraceEditorHandlers[key];
                if (previous) document.removeEventListener('keydown', previous);
                const handler = (event) => {
                    const target = event.target;
                    if (target && (
                        target.matches('input, textarea, select') ||
                        target.isContentEditable
                    )) return;
                    let action = null;
                    const modifier = event.ctrlKey || event.metaKey;
                    if (modifier && event.key.toLowerCase() === 'z') {
                        action = event.shiftKey ? 'redo' : 'undo';
                    } else if (modifier && event.key.toLowerCase() === 'y') {
                        action = 'redo';
                    } else if (event.key === 'Escape' || event.key === 'Enter') {
                        action = 'finish';
                    } else if (event.key === 'Delete') {
                        action = 'delete_vertex';
                    }
                    if (action === null) return;
                    event.preventDefault();
                    const token = Number(source.data.token[0] || 0) + 1;
                    source.data = {action: [action], token: [token]};
                    source.change.emit();
                };
                window.__ecatTraceEditorHandlers[key] = handler;
                document.addEventListener('keydown', handler);
            """,
        ),
    )

    controls = column(
        instructions,
        Div(text="<b>Display</b>"),
        basemap_select,
        display_options,
        opacity,
        mode,
        reference_select,
        row(buttons["new"], buttons["copy"], sizing_mode="stretch_width"),
        row(buttons["undo"], buttons["redo"], sizing_mode="stretch_width"),
        buttons["finish"],
        row(buttons["clear"], buttons["delete_vertex"], sizing_mode="stretch_width"),
        buttons["delete"],
        table,
        row(
            buttons["copy_selected"],
            buttons["copy_all"],
            sizing_mode="stretch_width",
        ),
        output_path,
        overwrite,
        row(buttons["validate"], buttons["save"], sizing_mode="stretch_width"),
        dirty_status,
        status,
        width=420,
        sizing_mode="stretch_height",
    )
    document.add_root(row(plot, controls, sizing_mode="stretch_both"))
    document.title = session.title
    _refresh()
    return TraceEditorView(
        plot=plot,
        working_source=working_source,
        node_source=node_source,
        table_source=table_source,
        reference_source=reference_source,
        status=status,
        dirty_status=dirty_status,
        mode=mode,
        output_path=output_path,
        action_source=action_source,
        background_source=background_source,
        background_renderer=background_renderer,
        color_mapper=color_mapper,
        color_bar=color_bar,
        basemap_select=basemap_select,
        display_options=display_options,
        opacity=opacity,
        buttons=buttons,
        tools={
            "draw": draw_tool,
            "edit": edit_tool,
            "pan": pan_tool,
            "raster": raster_controller,
        },
    )


def run_trace_editor(
    session,
    *,
    host="127.0.0.1",
    port=5006,
    open_browser=True,
):
    """Run one blocking local Bokeh server until interrupted in the terminal."""

    require_bokeh()
    from bokeh.server.server import Server

    host = str(host).strip() or "127.0.0.1"
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError("Trace-editor port must be within [1, 65535].")

    def _document_factory(document):
        build_trace_editor_document(document, session)

    server = Server(
        {"/": _document_factory},
        address=host,
        port=port,
        allow_websocket_origin=[f"{host}:{port}"],
    )
    server.start()
    print(f"ECAT trace editor: http://{host}:{port}/")
    print("Keep this terminal open; press Ctrl+C here to stop the editor.")
    if open_browser:
        server.io_loop.add_callback(server.show, "/")
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        print("Stopping ECAT trace editor.")
    finally:
        server.stop()


__all__ = [
    "TraceEditorView",
    "build_trace_editor_document",
    "require_bokeh",
    "run_trace_editor",
]
