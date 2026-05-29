from pathlib import Path

import numpy as np


def point_count(data):
    if hasattr(data, "vel"):
        return int(np.asarray(data.vel).size)
    if hasattr(data, "east") and hasattr(data, "north"):
        east_count = int(np.asarray(data.east).size)
        north_count = int(np.asarray(data.north).size)
        if east_count != north_count:
            raise ValueError(
                "Expected optical east and north arrays to have the same number of points; "
                f"got {east_count} and {north_count}."
            )
        return east_count
    raise ValueError("Cannot determine point count: expected vel or east/north values.")


def as_vector(values, name, expected_size=None, *, context="data"):
    array = np.asarray(values, dtype=float).reshape(-1)
    if expected_size is not None and array.size != expected_size:
        raise ValueError(
            f"{context} expected {name} to have {expected_size} values; got {array.size}."
        )
    return array


def box_values(box, label, key_groups=None):
    if isinstance(box, dict):
        key_groups = key_groups or (
            ("minLon", "maxLon", "minLat", "maxLat"),
            ("minlon", "maxlon", "minlat", "maxlat"),
            ("lon_min", "lon_max", "lat_min", "lat_max"),
        )
        for keys in key_groups:
            if all(key in box for key in keys):
                return tuple(float(box[key]) for key in keys)
        expected = " or ".join("/".join(keys) for keys in key_groups)
        raise ValueError(f"{label} must define {expected}.")
    if len(box) != 4:
        raise ValueError(f"{label} must contain four values.")
    return tuple(float(value) for value in box)


def inside_boxes(first, second, rule, *, label="region", key_groups=None):
    boxes = rule.get("boxes")
    if boxes is None:
        boxes = [rule.get("box", rule)]
    if not isinstance(boxes, (list, tuple)):
        raise ValueError(f"{label} requires boxes to be a list.")
    if len(boxes) == 4 and not isinstance(boxes[0], (dict, list, tuple)):
        boxes = [boxes]

    selected = np.zeros(first.size, dtype=bool)
    for index, box in enumerate(boxes):
        first_min, first_max, second_min, second_max = box_values(
            box,
            f"{label}[{index}]",
            key_groups=key_groups,
        )
        selected |= (
            (first >= first_min)
            & (first <= first_max)
            & (second >= second_min)
            & (second <= second_max)
        )
    return selected


def read_polygon_file(path, base_dir=None, *, min_points=3, label="Polygon file"):
    path = Path(path)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    points = np.loadtxt(path, comments="#", dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] < min_points:
        raise ValueError(
            f"{label} {path} must contain at least {min_points} rows "
            "and two columns."
        )
    return points[:, :2], path


def polygon_points(item, base_dir=None, *, min_points=3, label="polygon"):
    if isinstance(item, dict):
        if item.get("file") is not None:
            points, _ = read_polygon_file(
                item["file"],
                base_dir=base_dir,
                min_points=min_points,
                label=label,
            )
            return points
        if item.get("path") is not None:
            points, _ = read_polygon_file(
                item["path"],
                base_dir=base_dir,
                min_points=min_points,
                label=label,
            )
            return points
        if item.get("points") is not None:
            item = item["points"]
        else:
            raise ValueError(f"{label} entries require points, file, or path.")
    points = np.asarray(item, dtype=float)
    if points.ndim != 2 or points.shape[0] < min_points or points.shape[1] < 2:
        raise ValueError(
            f"{label} points must be a list of at least {min_points} coordinate pairs."
        )
    return points[:, :2]


def points_in_polygon(x, y, polygon):
    polygon = np.asarray(polygon, dtype=float)
    poly_x = polygon[:, 0]
    poly_y = polygon[:, 1]
    inside = np.zeros(x.size, dtype=bool)
    previous = polygon.shape[0] - 1
    for current in range(polygon.shape[0]):
        yi = poly_y[current]
        yj = poly_y[previous]
        xi = poly_x[current]
        xj = poly_x[previous]
        denominator = (yj - yi) if yj != yi else np.finfo(float).eps
        intersects = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / denominator + xi
        )
        inside ^= intersects
        previous = current
    return inside


def inside_polygons(first, second, rule, base_dir=None, *, label="polygon"):
    polygons = rule.get("polygons")
    if polygons is None:
        if rule.get("polygon") is not None:
            polygons = [rule["polygon"]]
        elif (
            rule.get("file") is not None
            or rule.get("path") is not None
            or rule.get("points") is not None
        ):
            polygons = [rule]
        else:
            raise ValueError(f"{label} requires polygon, polygons, points, file, or path.")

    selected = np.zeros(first.size, dtype=bool)
    for item in polygons:
        selected |= points_in_polygon(
            first,
            second,
            polygon_points(item, base_dir=base_dir, label=label),
        )
    return selected


def region_keep_mask(selected, action):
    action = str(action or "remove_inside").replace("-", "_").lower()
    if action in ("remove_inside", "keep_outside"):
        return ~selected
    if action in ("keep_inside", "remove_outside"):
        return selected
    raise ValueError(f"Unsupported data filter action: {action!r}.")
