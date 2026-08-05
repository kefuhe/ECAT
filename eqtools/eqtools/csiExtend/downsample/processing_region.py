import numpy as np
import yaml

from .region_utils import (
    align_longitude_interval_to_reference,
    align_longitudes,
    as_vector,
    finite_value_range,
    longitude_intervals_for_data,
    point_count,
    points_in_lonlat_polygon,
    points_in_polygon,
    polygon_points,
    read_polygon_file,
    unwrap_polygon_longitudes,
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
    if coord_type == "lonlat":
        first = align_longitudes(first, 0.5 * (first_min + first_max))
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
        if coord_type == "lonlat":
            return points_in_lonlat_polygon(first, second, polygon)
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
        if coord_type == "lonlat":
            return points_in_lonlat_polygon(first, second, polygon)
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


def _input_coordinate_range(data, coord_type, n_points):
    first, second = _coordinate_vectors(data, coord_type, n_points)
    if coord_type == "xy":
        return {
            "x": finite_value_range(first),
            "y": finite_value_range(second),
        }
    return {
        "longitude": finite_value_range(first),
        "latitude": finite_value_range(second),
    }


def _resolved_geometry_for_data(data, config, coord_type, n_points, base_dir=None):
    geometry = _region_geometry(config)
    if coord_type != "lonlat":
        return None
    longitude, _latitude = _coordinate_vectors(data, coord_type, n_points)
    if geometry == "box" and config.get("box") is not None:
        minimum, maximum, min_latitude, max_latitude = _box_values(
            config["box"],
            coord_type=coord_type,
        )
        intervals = longitude_intervals_for_data(
            minimum,
            maximum,
            longitude,
        )
        boxes = [
            [interval[0], interval[1], min_latitude, max_latitude]
            for interval in intervals
        ]
        return {"box": boxes[0]} if len(boxes) == 1 else {"boxes": boxes}

    if geometry == "polygon":
        polygon = polygon_points(
            config.get("polygon"),
            label="processing_region.polygon",
        )
    elif geometry == "polygon_file" and config.get("polygon_file") is not None:
        polygon, _resolved_path = read_polygon_file(
            config["polygon_file"],
            base_dir=base_dir,
            min_points=3,
            label="Processing-region polygon file",
        )
    else:
        return None

    polygon = unwrap_polygon_longitudes(polygon)
    data_range = finite_value_range(longitude)
    if data_range is not None:
        reference = 0.5 * sum(data_range)
        polygon_center = float(np.mean(polygon[:, 0]))
        shift = align_longitude_interval_to_reference(
            polygon_center,
            polygon_center,
            reference,
        )[0] - polygon_center
        polygon[:, 0] += shift
    return {"polygon": polygon.tolist()}


def _empty_region_message(report, config):
    coordinate_range = report.get("input_coordinate_range", {})
    longitude_range = coordinate_range.get("longitude")
    latitude_range = coordinate_range.get("latitude")
    if longitude_range is None:
        return "processing_region removed all processing points."
    message = (
        "processing_region removed all processing points; "
        f"data longitude range={longitude_range}, latitude range={latitude_range}"
    )
    if _region_geometry(config) == "box":
        message += f", configured box={config.get('box')}"
        resolved = report.get("resolved_geometry") or {}
        equivalent = resolved.get("box", resolved.get("boxes"))
        message += f", equivalent box near data={equivalent}"
    return message + "."


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

    report["input_coordinate_range"] = _input_coordinate_range(
        data,
        report["coord_type"],
        n_points,
    )
    resolved_geometry = _resolved_geometry_for_data(
        data,
        config,
        report["coord_type"],
        n_points,
        base_dir=base_dir,
    )
    if resolved_geometry is not None:
        report["resolved_geometry"] = resolved_geometry

    keep = processing_region_keep_mask(data, config, base_dir=base_dir)
    final_indices = np.flatnonzero(keep)
    if final_indices.size == 0:
        raise ValueError(_empty_region_message(report, config))

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
    coordinate_range = report.get("input_coordinate_range", {})
    if coordinate_range.get("longitude") is not None:
        lines.append(
            "  input lon/lat: "
            f"{coordinate_range['longitude']} / {coordinate_range.get('latitude')}"
        )
    resolved_geometry = report.get("resolved_geometry")
    if resolved_geometry:
        lines.append(f"  resolved geom: {resolved_geometry}")
    if report.get("report_file"):
        lines.append(f"  report file  : {report['report_file']}")
    return "\n".join(lines)
