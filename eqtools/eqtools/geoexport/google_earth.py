"""Google Earth KML/KMZ writers for detached geoexport layers.

The writer is intentionally display-only.  Raster pixels are never
interpolated or reprojected: only exact geographic rectilinear grids can be
written as ``GroundOverlay`` images in the first implementation.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .models import ExportResult, RasterLayer, VectorLayer


KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
ET.register_namespace("", KML_NAMESPACE)


def _tag(name):
    return f"{{{KML_NAMESPACE}}}{name}"


def _child(parent, name, text=None, **attributes):
    element = ET.SubElement(parent, _tag(name), attributes)
    if text is not None:
        element.text = str(text)
    return element


def _safe_asset_name(layer_id):
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(layer_id)
    )


def _hex_to_kml(color, alpha=1.0):
    red = color[1:3]
    green = color[3:5]
    blue = color[5:7]
    alpha_hex = f"{int(round(255.0 * float(alpha))):02x}"
    return f"{alpha_hex}{blue}{green}{red}".lower()


def _rgba_to_kml(rgba, alpha=1.0):
    red, green, blue, rgba_alpha = np.asarray(rgba, dtype=float)
    rgba_alpha *= float(alpha)
    return (
        f"{int(round(255 * rgba_alpha)):02x}"
        f"{int(round(255 * blue)):02x}"
        f"{int(round(255 * green)):02x}"
        f"{int(round(255 * red)):02x}"
    )


def _resolve_display_limits(values, style):
    """Return scaled display values and their resolved color limits.

    Explicit ``vmin/vmax`` always win. Otherwise linear data use the robust
    2nd/98th percentile display range, optionally made symmetric around zero.
    Scientific source values are never modified.
    """

    display_values = np.asarray(values, dtype=float) * float(style.display_factor)
    finite = display_values[np.isfinite(display_values)]
    if finite.size == 0:
        raise ValueError("A quantitative layer contains no finite values.")
    if style.normalization == "cyclic":
        period = float(style.cyclic_period)
        if style.vmin is None:
            limits = (0.0, period)
        else:
            limits = (style.vmin, style.vmax)
        display_values = np.mod(display_values - limits[0], period) + limits[0]
    elif style.vmin is not None:
        limits = (style.vmin, style.vmax)
    else:
        limits = tuple(
            float(value) for value in np.nanpercentile(finite, [2.0, 98.0])
        )
        if style.symmetry:
            half_range = max(abs(limits[0]), abs(limits[1]))
            limits = (-half_range, half_range)
        if np.isclose(limits[0], limits[1]):
            padding = max(abs(limits[0]) * 0.01, 1.0e-12)
            limits = (limits[0] - padding, limits[1] + padding)
    return display_values, tuple(float(value) for value in limits)


def _matplotlib_color_tools(cmap_name, limits):
    try:
        from matplotlib import colormaps
        from matplotlib.colors import Normalize
    except ImportError as exc:
        raise ImportError(
            "Google Earth quantitative export requires matplotlib."
        ) from exc
    try:
        cmap = colormaps.get_cmap(cmap_name)
    except ValueError as exc:
        raise ValueError(f"Unknown matplotlib colormap {cmap_name!r}.") from exc
    return cmap, Normalize(vmin=limits[0], vmax=limits[1], clip=True)


def _rectilinear_axes(layer):
    if layer.topology != "geographic_rectilinear":
        raise ValueError(
            f"Raster layer {layer.id!r} has topology {layer.topology!r}. "
            "Google Earth GroundOverlay export currently requires "
            "geographic_rectilinear coordinates; no interpolation was applied."
        )
    longitude = np.asarray(layer.longitude, dtype=float)
    latitude = np.asarray(layer.latitude, dtype=float)
    lon_axis = longitude[0, :]
    lat_axis = latitude[:, 0]
    tolerance = 1.0e-10
    if not np.allclose(
        longitude,
        np.broadcast_to(lon_axis, longitude.shape),
        rtol=0.0,
        atol=tolerance,
    ) or not np.allclose(
        latitude,
        np.broadcast_to(lat_axis[:, None], latitude.shape),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            f"Raster layer {layer.id!r} is not exactly rectilinear; "
            "no approximate axis reduction was performed."
        )
    if not np.all(np.isfinite(lon_axis)) or not np.all(np.isfinite(lat_axis)):
        raise ValueError("Raster coordinate axes must be finite.")
    lon_axis = _normalize_longitudes(lon_axis)
    lon_diff = np.diff(lon_axis)
    lat_diff = np.diff(lat_axis)
    if lon_axis.size < 2 or lat_axis.size < 2:
        raise ValueError("GroundOverlay rasters require at least 2x2 pixels.")
    if not (np.all(lon_diff > 0.0) or np.all(lon_diff < 0.0)):
        raise ValueError("Raster longitude axis must be strictly monotonic.")
    if not (np.all(lat_diff > 0.0) or np.all(lat_diff < 0.0)):
        raise ValueError("Raster latitude axis must be strictly monotonic.")
    for axis_name, differences in (
        ("longitude", lon_diff),
        ("latitude", lat_diff),
    ):
        spacing = float(differences[0])
        if not np.allclose(
            differences,
            spacing,
            rtol=1.0e-7,
            atol=max(abs(spacing) * 1.0e-10, 1.0e-12),
        ):
            raise ValueError(
                f"Raster {axis_name} axis must be regularly spaced for an "
                "exact Google Earth GroundOverlay; no bounding-box "
                "approximation was applied."
            )
    if float(np.max(lon_axis) - np.min(lon_axis)) > 180.0:
        raise ValueError(
            "Raster appears to cross the antimeridian; first-version "
            "GroundOverlay export does not split wrapped grids."
        )
    lat_edges = _axis_edges(lat_axis)
    if np.min(lat_edges) < -90.0 or np.max(lat_edges) > 90.0:
        raise ValueError(
            "Raster latitude pixel edges must remain within [-90, 90]."
        )
    return lon_axis.copy(), lat_axis.copy()


def _normalize_longitudes(values):
    """Return the equivalent KML longitude representation in [-180, 180]."""

    values = np.asarray(values, dtype=float)
    normalized = np.mod(values + 180.0, 360.0) - 180.0
    normalized[np.isclose(normalized, -180.0) & (values > 0.0)] = 180.0
    return normalized


def _axis_edges(axis):
    axis = np.asarray(axis, dtype=float)
    centers = 0.5 * (axis[:-1] + axis[1:])
    first = axis[0] - (centers[0] - axis[0])
    last = axis[-1] + (axis[-1] - centers[-1])
    return np.concatenate(([first], centers, [last]))


def _write_raster_png(layer, path):
    lon_axis, lat_axis = _rectilinear_axes(layer)
    values = np.asarray(layer.values, dtype=float)
    mask = np.asarray(layer.mask, dtype=bool)
    if lon_axis[0] > lon_axis[-1]:
        lon_axis = lon_axis[::-1]
        values = values[:, ::-1]
        mask = mask[:, ::-1]
    if lat_axis[0] < lat_axis[-1]:
        lat_axis = lat_axis[::-1]
        values = values[::-1, :]
        mask = mask[::-1, :]

    display_values, limits = _resolve_display_limits(values, layer.style)
    cmap, normalization = _matplotlib_color_tools(layer.style.cmap, limits)
    rgba = np.asarray(cmap(normalization(display_values)), dtype=float)
    rgba[..., 3] = np.where(mask, float(layer.style.alpha), 0.0)
    rgba[~np.isfinite(display_values), 3] = 0.0
    try:
        from matplotlib.image import imsave
    except ImportError as exc:
        raise ImportError("Raster KMZ export requires matplotlib.") from exc
    imsave(path, rgba)

    lon_edges = _axis_edges(lon_axis)
    lat_edges = _axis_edges(lat_axis)
    bounds = {
        "west": float(min(lon_edges[0], lon_edges[-1])),
        "east": float(max(lon_edges[0], lon_edges[-1])),
        "south": float(min(lat_edges[0], lat_edges[-1])),
        "north": float(max(lat_edges[0], lat_edges[-1])),
    }
    return limits, bounds


def _write_colorbar(layer, limits, path):
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError("Colorbar export requires matplotlib.") from exc
    cmap, normalization = _matplotlib_color_tools(layer.style.cmap, limits)
    figure = Figure(figsize=(4.4, 0.75), dpi=160)
    FigureCanvasAgg(figure)
    axis = figure.add_axes((0.08, 0.42, 0.84, 0.28))
    ColorbarBase(axis, cmap=cmap, norm=normalization, orientation="horizontal")
    unit = layer.style.display_unit or layer.units
    label = layer.name if not unit else f"{layer.name} ({unit})"
    axis.set_xlabel(label, fontsize=7)
    axis.tick_params(labelsize=6)
    figure.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.04)


def _layer_description(layer):
    lines = ["<table>"]
    if layer.units:
        lines.append(f"<tr><th>Stored unit</th><td>{layer.units}</td></tr>")
    if layer.convention:
        lines.append(
            f"<tr><th>Positive convention</th><td>{layer.convention}</td></tr>"
        )
    lines.append(
        "<tr><th>Display factor</th>"
        f"<td>{float(layer.style.display_factor):g}</td></tr>"
    )
    for key, value in sorted(dict(layer.metadata).items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"<tr><th>{key}</th><td>{value}</td></tr>")
    lines.append("</table>")
    return "".join(lines)


def _add_screen_overlay(document, layer, href, index):
    overlay = _child(document, "ScreenOverlay")
    _child(overlay, "name", f"{layer.name} legend")
    _child(overlay, "visibility", "1" if layer.visible else "0")
    icon = _child(overlay, "Icon")
    _child(icon, "href", href)
    ET.SubElement(
        overlay,
        _tag("overlayXY"),
        {"x": "0", "y": "0", "xunits": "fraction", "yunits": "fraction"},
    )
    ET.SubElement(
        overlay,
        _tag("screenXY"),
        {
            "x": "0.02",
            "y": str(0.03 + index * 0.10),
            "xunits": "fraction",
            "yunits": "fraction",
        },
    )
    ET.SubElement(
        overlay,
        _tag("size"),
        {"x": "0", "y": "0", "xunits": "pixels", "yunits": "pixels"},
    )


def _add_raster_layer(
    document,
    layer,
    asset_dir,
    asset_prefix,
    legend_index,
    draw_order,
):
    image_name = f"{asset_prefix}.png"
    legend_name = f"{asset_prefix}_legend.png"
    limits, bounds = _write_raster_png(layer, asset_dir / image_name)
    _write_colorbar(layer, limits, asset_dir / legend_name)

    folder = _child(document, "Folder")
    _child(folder, "name", layer.name)
    _child(folder, "visibility", "1" if layer.visible else "0")
    _child(folder, "description", _layer_description(layer))
    overlay = _child(folder, "GroundOverlay")
    _child(overlay, "name", f"{layer.name} raster")
    _child(overlay, "drawOrder", str(int(draw_order)))
    icon = _child(overlay, "Icon")
    _child(icon, "href", f"images/{image_name}")
    box = _child(overlay, "LatLonBox")
    for field in ("north", "south", "east", "west"):
        _child(box, field, f"{bounds[field]:.12g}")
    _child(box, "rotation", "0")
    _add_screen_overlay(
        folder,
        layer,
        f"images/{legend_name}",
        legend_index,
    )
    return limits


def _serializable_value(value):
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _feature_value(feature, property_name):
    if not property_name:
        return None
    value = feature["properties"].get(property_name)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _add_vector_styles(document, layer, display_values, limits):
    style_ids = []
    if len(display_values):
        cmap, normalization = _matplotlib_color_tools(layer.style.cmap, limits)
        colors = [
            _rgba_to_kml(cmap(normalization(value)), layer.style.alpha)
            for value in display_values
        ]
    else:
        colors = [_hex_to_kml(layer.style.line_color, layer.style.alpha)]
    for index, color in enumerate(colors):
        style_id = f"{_safe_asset_name(layer.id)}_style_{index}"
        style = ET.SubElement(document, _tag("Style"), {"id": style_id})
        icon_style = _child(style, "IconStyle")
        _child(icon_style, "color", color)
        _child(icon_style, "scale", f"{float(layer.style.point_scale):g}")
        line_style = _child(style, "LineStyle")
        _child(line_style, "color", color)
        _child(line_style, "width", f"{float(layer.style.line_width):g}")
        polygon_style = _child(style, "PolyStyle")
        _child(polygon_style, "color", color)
        outline_style = _child(style, "BalloonStyle")
        _child(outline_style, "text", "$[description]")
        style_ids.append(style_id)
    return style_ids


def _add_no_data_style(document, layer):
    """Add one neutral style that cannot be mistaken for a low value."""

    style_id = f"{_safe_asset_name(layer.id)}_style_no_data"
    style = ET.SubElement(document, _tag("Style"), {"id": style_id})
    color = _hex_to_kml("#808080", min(float(layer.style.alpha), 0.55))
    icon_style = _child(style, "IconStyle")
    _child(icon_style, "color", color)
    _child(icon_style, "scale", f"{float(layer.style.point_scale):g}")
    line_style = _child(style, "LineStyle")
    _child(line_style, "color", color)
    _child(line_style, "width", f"{float(layer.style.line_width):g}")
    polygon_style = _child(style, "PolyStyle")
    _child(polygon_style, "color", color)
    balloon_style = _child(style, "BalloonStyle")
    _child(balloon_style, "text", "$[description]")
    return style_id


def _coordinates_text(coordinates):
    array = np.asarray(coordinates, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    array = array.copy()
    array[:, 0] = _normalize_longitudes(array[:, 0])
    if array.shape[0] > 1 and np.ptp(array[:, 0]) > 180.0:
        raise ValueError(
            "Vector feature crosses the antimeridian; first-version export "
            "does not split the geometry."
        )
    return " ".join(",".join(f"{value:.12g}" for value in row) for row in array)


def _oriented_ring(ring, *, counterclockwise):
    """Close and orient one lon/lat ring using the KML right-hand rule."""

    ring = np.asarray(ring, dtype=float)
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack((ring, ring[0]))
    x = ring[:, 0]
    y = ring[:, 1]
    signed_area = 0.5 * float(
        np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    )
    is_counterclockwise = signed_area > 0.0
    if is_counterclockwise != bool(counterclockwise):
        ring = ring[::-1].copy()
    return ring


def _add_geometry(parent, geometry):
    geometry_type = geometry["type"]
    coordinates = geometry["coordinates"]
    if geometry_type == "Point":
        node = _child(parent, "Point")
        _child(node, "coordinates", _coordinates_text(coordinates))
        return
    if geometry_type == "LineString":
        node = _child(parent, "LineString")
        _child(node, "tessellate", "1")
        _child(node, "coordinates", _coordinates_text(coordinates))
        return
    polygon = _child(parent, "Polygon")
    _child(polygon, "tessellate", "1")
    raw_rings = [np.asarray(ring, dtype=float) for ring in coordinates]
    rings = [
        _oriented_ring(raw_rings[0], counterclockwise=True),
        *[
            _oriented_ring(ring, counterclockwise=False)
            for ring in raw_rings[1:]
        ],
    ]
    if rings[0].shape[1] == 3:
        _child(polygon, "altitudeMode", "absolute")
    outer = _child(polygon, "outerBoundaryIs")
    linear_ring = _child(outer, "LinearRing")
    _child(linear_ring, "coordinates", _coordinates_text(rings[0]))
    for ring in rings[1:]:
        inner = _child(polygon, "innerBoundaryIs")
        linear_ring = _child(inner, "LinearRing")
        _child(linear_ring, "coordinates", _coordinates_text(ring))


def _add_vector_layer(document, layer, asset_dir, asset_prefix, legend_index):
    feature_values = [
        _feature_value(feature, layer.value_property) for feature in layer.features
    ]
    finite_values = [value for value in feature_values if value is not None]
    folder = _child(document, "Folder")
    _child(folder, "name", layer.name)
    _child(folder, "visibility", "1" if layer.visible else "0")
    _child(folder, "description", _layer_description(layer))
    warnings = []
    no_data_style_id = None
    if finite_values:
        display_values, limits = _resolve_display_limits(
            finite_values,
            layer.style,
        )
        bins = np.linspace(limits[0], limits[1], 33)
        centers = 0.5 * (bins[:-1] + bins[1:])
        style_ids = _add_vector_styles(document, layer, centers, limits)
        legend_name = f"{asset_prefix}_legend.png"
        _write_colorbar(layer, limits, asset_dir / legend_name)
        _add_screen_overlay(
            folder,
            layer,
            f"images/{legend_name}",
            legend_index,
        )
        if len(finite_values) != len(feature_values):
            no_data_style_id = _add_no_data_style(document, layer)
            warnings.append(
                f"Layer {layer.id!r} contains "
                f"{len(feature_values) - len(finite_values)} feature(s) "
                "without a finite quantitative value; a neutral no-data "
                "style was used."
            )
    else:
        display_values = []
        limits = None
        bins = None
        style_ids = _add_vector_styles(document, layer, [], None)
        if layer.value_property:
            warnings.append(
                f"Layer {layer.id!r} declares quantitative property "
                f"{layer.value_property!r}, but no feature contains a finite "
                "value; the unscaled line color was used."
            )

    for feature_index, (feature, stored_value) in enumerate(
        zip(layer.features, feature_values)
    ):
        placemark = _child(folder, "Placemark")
        properties = feature["properties"]
        feature_name = properties.get("name", properties.get("id", feature_index))
        _child(placemark, "name", feature_name)
        if stored_value is not None:
            display_value = stored_value * float(layer.style.display_factor)
            if layer.style.normalization == "cyclic":
                display_value = (
                    np.mod(
                        display_value - limits[0],
                        float(layer.style.cyclic_period),
                    )
                    + limits[0]
                )
            style_index = int(
                np.clip(np.digitize(display_value, bins) - 1, 0, 31)
            )
        else:
            display_value = None
            style_index = None
        description = ["<table>"]
        for key, value in properties.items():
            description.append(
                f"<tr><th>{key}</th><td>{_serializable_value(value)}</td></tr>"
            )
        description.append("</table>")
        _child(placemark, "description", "".join(description))
        style_id = (
            no_data_style_id
            if style_index is None and no_data_style_id is not None
            else style_ids[0] if style_index is None else style_ids[style_index]
        )
        _child(placemark, "styleUrl", f"#{style_id}")
        extended = _child(placemark, "ExtendedData")
        for key, value in properties.items():
            data = ET.SubElement(extended, _tag("Data"), {"name": str(key)})
            _child(data, "value", _serializable_value(value))
        if stored_value is not None:
            data = ET.SubElement(
                extended,
                _tag("Data"),
                {"name": "display_value"},
            )
            _child(data, "value", f"{display_value:.12g}")
        _add_geometry(placemark, feature["geometry"])
    return limits, warnings


def _write_kml_tree(layers, staging_dir, document_name):
    root = ET.Element(_tag("kml"))
    document = _child(root, "Document")
    _child(document, "name", document_name)
    _child(
        document,
        "description",
        "Display-only export created by ECAT/eqtools geoexport. "
        "Do not use KML/KMZ values as an inversion input.",
    )
    image_dir = staging_dir / "images"
    image_dir.mkdir()
    manifest_layers = []
    warnings = []
    legend_index = 0
    for draw_order, layer in enumerate(layers):
        asset_prefix = _safe_asset_name(layer.id)
        if isinstance(layer, RasterLayer):
            limits = _add_raster_layer(
                document,
                layer,
                image_dir,
                asset_prefix,
                legend_index,
                draw_order,
            )
            legend_index += 1
            layer_type = "raster"
            feature_count = int(np.count_nonzero(layer.mask))
            estimated_mib = float(layer.values.size * 49) / (1024.0 ** 2)
            if estimated_mib >= 128.0:
                warnings.append(
                    f"Raster layer {layer.id!r} may require roughly "
                    f"{estimated_mib:.0f} MiB of temporary array memory while "
                    "creating its full-resolution PNG."
                )
        elif isinstance(layer, VectorLayer):
            limits, layer_warnings = _add_vector_layer(
                document,
                layer,
                image_dir,
                asset_prefix,
                legend_index,
            )
            warnings.extend(layer_warnings)
            if limits is not None:
                legend_index += 1
            layer_type = "vector"
            feature_count = len(layer.features)
        else:
            raise TypeError(
                "write_kmz accepts only RasterLayer and VectorLayer objects."
            )
        manifest_layers.append(
            {
                "id": layer.id,
                "name": layer.name,
                "type": layer_type,
                "count": feature_count,
                "stored_unit": layer.units,
                "positive_convention": layer.convention,
                "visible": bool(layer.visible),
                "cmap": layer.style.cmap,
                "alpha": float(layer.style.alpha),
                "symmetry": bool(layer.style.symmetry),
                "display_factor": float(layer.style.display_factor),
                "display_unit": layer.style.display_unit or layer.units,
                "display_limits": limits,
                "requested_vmin": layer.style.vmin,
                "requested_vmax": layer.style.vmax,
                "automatic_limits": (
                    "percentile_2_98"
                    if layer.style.vmin is None
                    and layer.style.normalization == "linear"
                    else None
                ),
                "normalization": layer.style.normalization,
                "cyclic_period": layer.style.cyclic_period,
                "metadata": dict(layer.metadata),
            }
        )
    tree = ET.ElementTree(root)
    tree.write(
        staging_dir / "doc.kml",
        encoding="utf-8",
        xml_declaration=True,
    )
    manifest = {
        "schema": "ecat-google-earth-export",
        "version": 2,
        "display_only": True,
        "layers": manifest_layers,
    }
    with (staging_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, default=str)
    return tuple(warnings)


def write_kmz(
    layers,
    output_file,
    *,
    overwrite=False,
    document_name=None,
):
    """Write one self-contained KMZ with one or more detached layers.

    Parameters
    ----------
    layers : sequence of RasterLayer or VectorLayer
        Layers in Google Earth drawing order.
    output_file : path-like
        Destination ending in ``.kmz``.
    overwrite : bool, default False
        Permit replacement of this exact output file.
    document_name : str, optional
        Name shown at the top of the Google Earth layer tree.

    Returns
    -------
    ExportResult
        Output and layer summary.

    Notes
    -----
    Stored arrays and geometry are read only. Display factors, colormaps, and
    alpha affect rendering and labels but never rewrite scientific values.
    """

    layers = tuple(layers)
    if not layers:
        raise ValueError("At least one export layer is required.")
    layer_ids = [layer.id for layer in layers]
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("Layer ids must be unique within one KMZ.")
    asset_names = [_safe_asset_name(layer_id) for layer_id in layer_ids]
    if len(set(asset_names)) != len(asset_names):
        raise ValueError(
            "Layer ids must remain unique after filename normalization."
        )
    output_file = Path(output_file)
    if output_file.suffix.lower() != ".kmz":
        raise ValueError("Google Earth archive output must end in .kmz.")
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_file}. Use overwrite=True."
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".ecat_geoexport_", dir=output_file.parent)
    )
    temporary_archive = staging_dir.with_suffix(".kmz.tmp")
    try:
        warnings = _write_kml_tree(
            layers,
            staging_dir,
            document_name or output_file.stem,
        )
        with zipfile.ZipFile(
            temporary_archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for path in sorted(staging_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging_dir).as_posix())
        os.replace(temporary_archive, output_file)
    finally:
        if temporary_archive.exists():
            temporary_archive.unlink()
        shutil.rmtree(staging_dir, ignore_errors=True)
    return ExportResult(
        output_files=(output_file,),
        layer_ids=tuple(layer_ids),
        warnings=warnings,
        package_mode="single",
    )


__all__ = ["KML_NAMESPACE", "write_kmz"]
