import numpy as np
import yaml

from .region_utils import (
    as_vector,
    point_count,
    points_in_polygon,
    polygon_points,
    read_polygon_file,
)


def processing_region_report_file(config, out_name):
    report_file = config.get("report_file", "auto")
    if report_file in (None, False):
        return None
    if str(report_file).lower() == "auto":
        return f"{out_name}_processing_region_report.yml"
    return str(report_file)


def _box_values(box, coord_type="lonlat"):
    if isinstance(box, dict):
        if coord_type == "xy":
            key_groups = (
                ("minX", "maxX", "minY", "maxY"),
                ("minx", "maxx", "miny", "maxy"),
                ("x_min", "x_max", "y_min", "y_max"),
            )
        else:
            key_groups = (
                ("minLon", "maxLon", "minLat", "maxLat"),
                ("minlon", "maxlon", "minlat", "maxlat"),
                ("lon_min", "lon_max", "lat_min", "lat_max"),
            )
        for keys in key_groups:
            if all(key in box for key in keys):
                return tuple(float(box[key]) for key in keys)
        if coord_type == "xy":
            raise ValueError(
                "processing_region.box must define minX/maxX/minY/maxY "
                "or x_min/x_max/y_min/y_max when coord_type='xy'."
            )
        raise ValueError(
            "processing_region.box must define minLon/maxLon/minLat/maxLat "
            "or lon_min/lon_max/lat_min/lat_max when coord_type='lonlat'."
        )
    if len(box) != 4:
        raise ValueError("processing_region.box must contain four values.")
    return tuple(float(value) for value in box)


def _inside_box(first, second, box, coord_type="lonlat"):
    first_min, first_max, second_min, second_max = _box_values(box, coord_type=coord_type)
    return (
        (first >= first_min)
        & (first <= first_max)
        & (second >= second_min)
        & (second <= second_max)
    )


def _region_geometry(config):
    geometry = config.get("geometry")
    if geometry is not None:
        return str(geometry).replace("-", "_").lower()
    if config.get("polygon_file") is not None:
        return "polygon_file"
    if config.get("polygon") is not None:
        return "polygon"
    return "box"


def _coordinate_vectors(data, coord_type, n_points):
    if coord_type == "xy":
        return (
            as_vector(data.x, "x", n_points, context="processing_region"),
            as_vector(data.y, "y", n_points, context="processing_region"),
        )
    return (
        as_vector(data.lon, "lon", n_points, context="processing_region"),
        as_vector(data.lat, "lat", n_points, context="processing_region"),
    )


def processing_region_keep_mask(data, config, base_dir=None):
    n_points = point_count(data)
    coord_type = str(config.get("coord_type", "lonlat")).replace("-", "_").lower()
    first, second = _coordinate_vectors(data, coord_type, n_points)
    geometry = _region_geometry(config)

    if geometry == "box":
        if config.get("box") is None:
            raise ValueError("processing_region requires box when geometry='box'.")
        return _inside_box(first, second, config["box"], coord_type=coord_type)
    if geometry == "polygon":
        if config.get("polygon") is None:
            raise ValueError("processing_region requires polygon when geometry='polygon'.")
        polygon = polygon_points(config["polygon"], label="processing_region.polygon")
        return points_in_polygon(first, second, polygon)
    if geometry == "polygon_file":
        if config.get("polygon_file") is None:
            raise ValueError("processing_region requires polygon_file when geometry='polygon_file'.")
        polygon, resolved_path = read_polygon_file(
            config["polygon_file"],
            base_dir=base_dir,
            min_points=3,
            label="Processing-region polygon file",
        )
        config["resolved_polygon_file"] = str(resolved_path)
        return points_in_polygon(first, second, polygon)
    raise ValueError(f"Unsupported processing_region.geometry: {geometry!r}.")


def keep_processing_indices(data, indices, n_points):
    if hasattr(data, "keepPixels"):
        data.keepPixels(indices)
        return
    if hasattr(data, "reject_pixel"):
        keep = np.zeros(n_points, dtype=bool)
        keep[np.asarray(indices, dtype=int)] = True
        rejected = np.flatnonzero(~keep)
        if rejected.size:
            data.reject_pixel(rejected)
        return
    raise AttributeError(
        "processing_region requires the data object to provide keepPixels() "
        "or reject_pixel()."
    )


def apply_processing_region(data, config, out_name="sar", base_dir=None, write_report=True):
    config = config or {}
    enabled = bool(config.get("enabled", False))
    n_points = point_count(data)
    report = {
        "enabled": enabled,
        "input_count": n_points,
        "final_count": n_points,
        "removed_count": 0,
        "coord_type": str(config.get("coord_type", "lonlat")).replace("-", "_").lower(),
        "geometry": _region_geometry(config),
    }
    if not enabled:
        return report

    keep = processing_region_keep_mask(data, config, base_dir=base_dir)
    final_indices = np.flatnonzero(keep)
    if final_indices.size == 0:
        raise ValueError("processing_region removed all processing points.")

    report["final_count"] = int(final_indices.size)
    report["removed_count"] = int(n_points - final_indices.size)
    report["details"] = {
        key: value
        for key, value in config.items()
        if key not in ("enabled", "report", "report_file")
    }
    if final_indices.size != n_points:
        keep_processing_indices(data, final_indices, n_points)

    report_file = processing_region_report_file(config, out_name)
    if write_report and config.get("report", True) and report_file is not None:
        with open(report_file, "w", encoding="utf-8") as file:
            yaml.safe_dump(report, file, allow_unicode=True, sort_keys=False)
        report["report_file"] = report_file
    return report


def format_processing_region_report(report):
    if not report.get("enabled"):
        return ""
    lines = [
        "Processing region:",
        f"  input points : {report['input_count']}",
        f"  geometry     : {report['geometry']} ({report['coord_type']})",
        f"  removed      : {report['removed_count']}",
        f"  final points : {report['final_count']}",
    ]
    if report.get("report_file"):
        lines.append(f"  report file  : {report['report_file']}")
    return "\n".join(lines)
