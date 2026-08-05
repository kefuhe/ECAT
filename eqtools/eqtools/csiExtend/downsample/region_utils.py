from pathlib import Path

import numpy as np


LONGITUDE_PERIOD_DEGREES = 360.0


def finite_value_range(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    return [float(np.min(finite)), float(np.max(finite))]


def align_longitudes(values, reference):
    """Return longitudes on the periodic branch nearest ``reference``.

    The input is never mutated.  This makes geometrically equivalent
    ``[-180, 180]`` and ``[0, 360]`` representations compare consistently
    without changing the longitude convention stored by CSI.
    """

    values = np.asarray(values, dtype=float)
    reference = float(reference)
    return (
        reference
        + np.mod(
            values - reference + 0.5 * LONGITUDE_PERIOD_DEGREES,
            LONGITUDE_PERIOD_DEGREES,
        )
        - 0.5 * LONGITUDE_PERIOD_DEGREES
    )


def align_longitude_interval_to_reference(minimum, maximum, reference):
    """Shift one continuous longitude interval near a reference longitude."""

    minimum = float(minimum)
    maximum = float(maximum)
    center = 0.5 * (minimum + maximum)
    shift = LONGITUDE_PERIOD_DEGREES * np.floor(
        (float(reference) - center) / LONGITUDE_PERIOD_DEGREES + 0.5
    )
    return [minimum + float(shift), maximum + float(shift)]


def longitude_intervals_for_data(minimum, maximum, longitude):
    """Return equivalent interval branches that overlap numeric data longitudes."""

    minimum = float(minimum)
    maximum = float(maximum)
    data_range = finite_value_range(longitude)
    if data_range is None:
        return [[minimum, maximum]]
    data_minimum, data_maximum = data_range
    first_shift = int(
        np.ceil((data_minimum - maximum) / LONGITUDE_PERIOD_DEGREES)
    )
    last_shift = int(
        np.floor((data_maximum - minimum) / LONGITUDE_PERIOD_DEGREES)
    )
    intervals = [
        [
            minimum + shift * LONGITUDE_PERIOD_DEGREES,
            maximum + shift * LONGITUDE_PERIOD_DEGREES,
        ]
        for shift in range(first_shift, last_shift + 1)
    ]
    if intervals:
        return intervals
    reference = 0.5 * (data_minimum + data_maximum)
    return [
        align_longitude_interval_to_reference(
            minimum,
            maximum,
            reference,
        )
    ]


def unwrap_polygon_longitudes(polygon):
    """Return a polygon whose longitude path is continuous across the dateline."""

    polygon = np.asarray(polygon, dtype=float)
    unwrapped = np.array(polygon, dtype=float, copy=True)
    unwrapped[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(unwrapped[:, 0])))
    return unwrapped


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


def inside_boxes(
    first,
    second,
    rule,
    *,
    label="region",
    key_groups=None,
    periodic_first=False,
):
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
        comparison_first = (
            align_longitudes(first, 0.5 * (first_min + first_max))
            if periodic_first
            else first
        )
        selected |= (
            (comparison_first >= first_min)
            & (comparison_first <= first_max)
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


def points_in_lonlat_polygon(longitude, latitude, polygon):
    """Test longitude/latitude points using periodic longitude equivalence."""

    polygon = unwrap_polygon_longitudes(polygon)
    reference = float(np.mean(polygon[:, 0]))
    longitude = align_longitudes(longitude, reference)
    return points_in_polygon(longitude, latitude, polygon)


def project_lonlat(data, longitude, latitude, *, context="region"):
    """Project matching longitude/latitude arrays with a CSI-like data object."""

    if not hasattr(data, "ll2xy"):
        raise AttributeError(
            f"{context} requires the data object to provide ll2xy()."
        )
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    if longitude.shape != latitude.shape:
        raise ValueError(
            f"{context} longitude and latitude arrays must have matching shapes."
        )
    shape = longitude.shape
    x, y = data.ll2xy(longitude.reshape(-1), latitude.reshape(-1))
    return (
        np.asarray(x, dtype=float).reshape(shape),
        np.asarray(y, dtype=float).reshape(shape),
    )


def lonlat_region_mask(
    data,
    longitude,
    latitude,
    region,
    *,
    base_dir=None,
    label="region",
):
    """Return a lon/lat mask for one circle, box, polygon, or polygon file."""

    if not isinstance(region, dict):
        raise ValueError(f"{label} must be a mapping.")
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    if longitude.shape != latitude.shape:
        raise ValueError(f"{label} longitude and latitude shapes must match.")

    kind = str(region.get("kind", "")).replace("-", "_").lower()
    supported = ("circle", "box", "polygon", "polygon_file")
    if kind not in supported:
        raise ValueError(
            f"{label}.kind must be one of {supported}; got {kind!r}."
        )

    if kind == "circle":
        center = region.get("center")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError(f"{label}.center must be [lon, lat].")
        radius = float(region.get("radius_km", 0.0))
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"{label}.radius_km must be positive.")
        x, y = project_lonlat(
            data,
            longitude,
            latitude,
            context=label,
        )
        center_x, center_y = project_lonlat(
            data,
            np.asarray([float(center[0])]),
            np.asarray([float(center[1])]),
            context=label,
        )
        return np.hypot(x - center_x[0], y - center_y[0]) <= radius

    if kind == "box":
        bounds = region.get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            raise ValueError(
                f"{label}.bounds must be "
                "[min_lon, max_lon, min_lat, max_lat]."
            )
        min_lon, max_lon, min_lat, max_lat = (
            float(value) for value in bounds
        )
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(
                f"{label}.bounds must satisfy min_lon < max_lon and "
                "min_lat < max_lat."
            )
        comparison_longitude = align_longitudes(
            longitude,
            0.5 * (min_lon + max_lon),
        )
        return (
            (comparison_longitude >= min_lon)
            & (comparison_longitude <= max_lon)
            & (latitude >= min_lat)
            & (latitude <= max_lat)
        )

    if kind == "polygon":
        polygon = polygon_points(
            region.get("polygon"),
            base_dir=base_dir,
            label=f"{label}.polygon",
        )
    else:
        polygon_file = region.get("polygon_file")
        if polygon_file is None:
            raise ValueError(f"{label}.polygon_file is required.")
        polygon, _resolved = read_polygon_file(
            polygon_file,
            base_dir=base_dir,
            min_points=3,
            label=f"{label}.polygon_file",
        )
    return points_in_lonlat_polygon(
        longitude.reshape(-1),
        latitude.reshape(-1),
        polygon,
    ).reshape(longitude.shape)


def inside_polygons(
    first,
    second,
    rule,
    base_dir=None,
    *,
    label="polygon",
    periodic_first=False,
):
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
        polygon = polygon_points(item, base_dir=base_dir, label=label)
        if periodic_first:
            selected |= points_in_lonlat_polygon(first, second, polygon)
        else:
            selected |= points_in_polygon(first, second, polygon)
    return selected


def region_keep_mask(selected, action):
    action = str(action or "remove_inside").replace("-", "_").lower()
    if action in ("remove_inside", "keep_outside"):
        return ~selected
    if action in ("keep_inside", "remove_outside"):
        return selected
    raise ValueError(f"Unsupported data filter action: {action!r}.")
