import numpy as np
import yaml

from .region_utils import (
    as_vector,
    inside_boxes,
    inside_polygons,
    point_count,
    region_keep_mask as _region_keep_mask,
)


SHARED_DATA_FILTER_KIND_CHOICES = (
    "finite",
    "lonlat_box",
    "lonlat_polygon",
)

SAR_DATA_FILTER_KIND_CHOICES = SHARED_DATA_FILTER_KIND_CHOICES + (
    "value_abs",
    "value_range",
    "projection_norm",
)

OPTICAL_DATA_FILTER_KIND_CHOICES = SHARED_DATA_FILTER_KIND_CHOICES + (
    "component_abs",
    "component_range",
    "vector_norm_range",
)

DATA_FILTER_KIND_CHOICES = SAR_DATA_FILTER_KIND_CHOICES

DATA_FILTER_ACTION_CHOICES = (
    "remove_inside",
    "keep_inside",
    "remove_outside",
    "keep_outside",
)


def _as_vector(values, name, expected_size=None):
    return as_vector(values, name, expected_size, context="data_filters")


def _inside_boxes(lon, lat, rule):
    return inside_boxes(lon, lat, rule, label="lonlat_box")


def _inside_polygons(lon, lat, rule, base_dir=None):
    return inside_polygons(lon, lat, rule, base_dir=base_dir, label="lonlat_polygon")


def _projection(data, expected_size):
    projection = getattr(data, "los", None)
    if projection is None:
        raise ValueError("data_filters.projection_norm requires data.los/projection values.")
    projection = np.asarray(projection, dtype=float)
    if projection.ndim != 2 or projection.shape[1] != 3 or projection.shape[0] != expected_size:
        raise ValueError(
            "data_filters expected data.los/projection to have shape (n_points, 3)."
        )
    return projection


def _observation_values(data, rule, expected_size):
    if not hasattr(data, "vel"):
        raise ValueError(
            "data_filters value rules require scalar observation data.vel. "
            "Use component_* or vector_norm_range rules for optical data."
        )
    value_space = str(rule.get("value_space", "observation")).replace("-", "_").lower()
    if value_space != "observation":
        raise ValueError(
            "data_filters currently applies value rules to value_space='observation' "
            "(the converted data.vel used by inversion)."
        )
    return _as_vector(data.vel, "vel", expected_size)


def _component_names(rule):
    if rule.get("components") is not None:
        names = rule["components"]
    elif rule.get("component") is not None:
        names = [rule["component"]]
    else:
        names = ["east", "north"]
    if isinstance(names, str):
        names = [names]
    normalized = [str(name).replace("-", "_").lower() for name in names]
    invalid = [name for name in normalized if name not in ("east", "north")]
    if invalid:
        raise ValueError(
            "data_filters component rules support only 'east' and 'north'; "
            f"got {invalid!r}."
        )
    return normalized


def _component_values(data, rule, expected_size):
    values = []
    for component in _component_names(rule):
        if not hasattr(data, component):
            raise ValueError(
                f"data_filters component rule requires data.{component}."
            )
        values.append(_as_vector(getattr(data, component), component, expected_size))
    return values


def _horizontal_norm(data, expected_size):
    east = _as_vector(data.east, "east", expected_size)
    north = _as_vector(data.north, "north", expected_size)
    return np.sqrt(east**2 + north**2)


def _finite_keep_mask(data, n_points):
    if hasattr(data, "vel"):
        keep = np.isfinite(_as_vector(data.vel, "vel", n_points))
    elif hasattr(data, "east") and hasattr(data, "north"):
        keep = np.isfinite(_as_vector(data.east, "east", n_points))
        keep &= np.isfinite(_as_vector(data.north, "north", n_points))
    else:
        raise ValueError(
            "data_filters finite rule requires scalar data.vel or optical "
            "data.east/data.north."
        )
    keep &= np.isfinite(_as_vector(data.lon, "lon", n_points))
    keep &= np.isfinite(_as_vector(data.lat, "lat", n_points))
    projection = getattr(data, "los", None)
    if projection is not None:
        keep &= np.all(np.isfinite(_projection(data, n_points)), axis=1)
    return keep


def _rule_keep_mask(data, rule, n_points, base_dir=None):
    kind = str(rule.get("kind", "")).replace("-", "_").lower()
    if kind == "finite":
        return _finite_keep_mask(data, n_points)
    if kind == "value_abs":
        if rule.get("threshold") is None:
            raise ValueError("value_abs filter requires threshold.")
        values = _observation_values(data, rule, n_points)
        return np.abs(values) <= float(rule["threshold"])
    if kind == "component_abs":
        if rule.get("threshold") is None:
            raise ValueError("component_abs filter requires threshold.")
        keep = np.ones(n_points, dtype=bool)
        threshold = float(rule["threshold"])
        for values in _component_values(data, rule, n_points):
            keep &= np.abs(values) <= threshold
        return keep
    if kind == "value_range":
        values = _observation_values(data, rule, n_points)
        keep = np.ones(n_points, dtype=bool)
        if rule.get("min") is not None:
            keep &= values >= float(rule["min"])
        if rule.get("max") is not None:
            keep &= values <= float(rule["max"])
        return keep
    if kind == "component_range":
        keep = np.ones(n_points, dtype=bool)
        for values in _component_values(data, rule, n_points):
            if rule.get("min") is not None:
                keep &= values >= float(rule["min"])
            if rule.get("max") is not None:
                keep &= values <= float(rule["max"])
        return keep
    if kind == "lonlat_box":
        lon = _as_vector(data.lon, "lon", n_points)
        lat = _as_vector(data.lat, "lat", n_points)
        return _region_keep_mask(_inside_boxes(lon, lat, rule), rule.get("action"))
    if kind == "lonlat_polygon":
        lon = _as_vector(data.lon, "lon", n_points)
        lat = _as_vector(data.lat, "lat", n_points)
        return _region_keep_mask(_inside_polygons(lon, lat, rule, base_dir=base_dir), rule.get("action"))
    if kind == "projection_norm":
        norms = np.linalg.norm(_projection(data, n_points), axis=1)
        if rule.get("tolerance") is not None:
            target = float(rule.get("target", 1.0))
            tolerance = float(rule["tolerance"])
            lower = target - tolerance
            upper = target + tolerance
        else:
            lower = float(rule["min"]) if rule.get("min") is not None else -np.inf
            upper = float(rule["max"]) if rule.get("max") is not None else np.inf
        return (norms >= lower) & (norms <= upper)
    if kind == "vector_norm_range":
        norms = _horizontal_norm(data, n_points)
        if rule.get("tolerance") is not None:
            target = float(rule.get("target", 0.0))
            tolerance = float(rule["tolerance"])
            lower = target - tolerance
            upper = target + tolerance
        else:
            lower = float(rule["min"]) if rule.get("min") is not None else -np.inf
            upper = float(rule["max"]) if rule.get("max") is not None else np.inf
        return (norms >= lower) & (norms <= upper)
    raise ValueError(f"Unsupported data filter kind: {kind!r}.")


def _enabled_rules(config):
    rules = config.get("rules") or []
    if not isinstance(rules, list):
        raise ValueError("sar_config.data_filters.rules must be a list.")
    return [rule for rule in rules if rule.get("enabled", True)]


def filter_report_file(config, out_name):
    report_file = config.get("report_file", "auto")
    if report_file in (None, False):
        return None
    if str(report_file).lower() == "auto":
        return f"{out_name}_filter_report.yml"
    return str(report_file)


def apply_data_filters(data, config, out_name="sar", base_dir=None, write_report=True):
    config = config or {}
    enabled = bool(config.get("enabled", False))
    n_points = point_count(data)
    report = {
        "enabled": enabled,
        "input_count": n_points,
        "final_count": n_points,
        "rules": [],
    }
    if not enabled:
        return report

    cumulative_keep = np.ones(n_points, dtype=bool)
    rules = [{"name": "finite", "kind": "finite", "enabled": True}]
    rules.extend(_enabled_rules(config))

    for index, rule in enumerate(rules):
        name = rule.get("name") or rule.get("kind") or f"rule_{index}"
        rule_keep = _rule_keep_mask(data, rule, n_points, base_dir=base_dir)
        current = cumulative_keep.copy()
        input_count = int(current.sum())
        cumulative_keep &= rule_keep
        kept_count = int(cumulative_keep.sum())
        removed_count = input_count - kept_count
        report["rules"].append(
            {
                "name": str(name),
                "kind": str(rule.get("kind")),
                "input_count": input_count,
                "removed_count": int(removed_count),
                "kept_count": kept_count,
                "details": {
                    key: value
                    for key, value in rule.items()
                    if key not in ("enabled",)
                },
            }
        )

    final_indices = np.flatnonzero(cumulative_keep)
    report["final_count"] = int(final_indices.size)
    raw_valid_index = getattr(data, "projection_raw_valid_index", None)
    if raw_valid_index is not None:
        data.data_filter_raw_valid_index = np.asarray(raw_valid_index, dtype=int)[final_indices]
    if final_indices.size != n_points:
        if hasattr(data, "keepPixels"):
            data.keepPixels(final_indices)
        elif hasattr(data, "reject_pixel"):
            keep = np.zeros(n_points, dtype=bool)
            keep[final_indices] = True
            rejected = np.flatnonzero(~keep)
            if rejected.size:
                data.reject_pixel(rejected)
        else:
            raise AttributeError(
                "data_filters requires the data object to provide keepPixels() "
                "or reject_pixel()."
            )
        if hasattr(data, "projection_downsampled_valid_index"):
            data.projection_downsampled_valid_index = np.asarray(
                data.projection_downsampled_valid_index,
                dtype=int,
            )[final_indices]
        if hasattr(data, "projection_valid"):
            data.projection_valid = data.los

    report_file = filter_report_file(config, out_name)
    if write_report and config.get("report", True) and report_file is not None:
        with open(report_file, "w", encoding="utf-8") as file:
            yaml.safe_dump(report, file, allow_unicode=True, sort_keys=False)
        report["report_file"] = report_file

    return report


def format_filter_report(report):
    if not report.get("enabled"):
        return ""
    lines = [
        "Data filters:",
        f"  input points : {report['input_count']}",
    ]
    for rule in report.get("rules", []):
        if rule["kind"] == "finite" and rule["removed_count"] == 0:
            continue
        details = rule.get("details", {})
        suffix = ""
        if rule["kind"] == "value_abs":
            suffix = (
                " by abs(observation)>"
                f"{details.get('threshold')}"
            )
        elif rule["kind"] == "component_abs":
            suffix = (
                " by abs(component)>"
                f"{details.get('threshold')}"
            )
        elif rule["kind"] == "value_range":
            suffix = f" by observation outside [{details.get('min')}, {details.get('max')}]"
        elif rule["kind"] == "component_range":
            suffix = f" by component outside [{details.get('min')}, {details.get('max')}]"
        elif rule["kind"] in ("lonlat_box", "lonlat_polygon"):
            suffix = f" by action={details.get('action', 'remove_inside')}"
        elif rule["kind"] == "projection_norm":
            suffix = " by projection norm limits"
        elif rule["kind"] == "vector_norm_range":
            suffix = " by horizontal-vector norm limits"
        lines.append(
            f"  {rule['name']}: removed {rule['removed_count']}/{rule['input_count']}{suffix}"
        )
    lines.append(f"  final points : {report['final_count']}")
    if report.get("report_file"):
        lines.append(f"  report file  : {report['report_file']}")
    return "\n".join(lines)
