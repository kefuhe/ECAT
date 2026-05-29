from copy import deepcopy


SAR_MODE_CHOICES = ("unwrapped_phase", "phase_los", "los_displacement", "range_offset", "azimuth_offset")
SAR_READER_CHOICES = ("gamma", "gamma_tiff", "gmtsar", "hyp3")
SAR_READER_ALIASES = {
    "gamma": "gamma",
    "gammasar": "gamma",
    "gamma_binary": "gamma",
    "gamma_tiff": "gamma_tiff",
    "gammatiff": "gamma_tiff",
    "gmtsar": "gmtsar",
    "gmt_sar": "gmtsar",
    "hyp3": "hyp3",
    "hyp3_tiff": "hyp3",
}
ORIGIN_MODE_CHOICES = ("auto", "manual")
COVAR_MISSING_POLICY_CHOICES = ("existing_or_identity", "identity", "error")
DOWNSAMPLE_METHOD_CHOICES = ("std", "data", "trirb", "from_rsp")
SPLIT_METRIC_CHOICES = ("curvature", "gradient")
FOCUS_REGION_COORD_CHOICES = ("lonlat", "xy")
PROCESSING_REGION_GEOMETRY_CHOICES = ("box", "polygon", "polygon_file")
AMPLITUDE_STAT_CHOICES = ("mean_abs", "abs_mean", "median_abs")
STD_CORRECTION_CHOICES = ("std", "mean", "median", "bilinear")
LEGACY_DOWNSAMPLE_FIELD_RENAMES = {
    "downsample.std_config.tolerance": "downsample.std_config.min_valid_fraction",
    "downsample.std_config.std_threshold": "downsample.std_config.split_std_threshold",
    "downsample.std_config.correction": "downsample.std_config.split_metric_correction",
    "downsample.std_config.smooth": "downsample.std_config.split_metric_smoothing",
    "downsample.data_config.tolerance": "downsample.data_config.min_valid_fraction",
    "downsample.data_config.threshold": "downsample.data_config.split_metric_threshold",
    "downsample.data_config.quantity": "downsample.data_config.split_metric",
    "downsample.data_config.smooth": "downsample.data_config.split_metric_smoothing",
    "downsample.trirb_config.tolerance": "downsample.trirb_config.min_valid_fraction",
    "downsample.from_rsp_config.tolerance": "downsample.from_rsp_config.min_valid_fraction",
}
EXTRACTION_VALUE_STAT_CHOICES = ("mean", "median", "center_nearest", "trimmed_mean")
EXTRACTION_ERROR_STAT_CHOICES = ("std", "mad", "sem", "none")
EXTRACTION_COORDINATE_STAT_CHOICES = ("mean", "block_center", "center_nearest")
EXTRACTION_PROJECTION_STAT_CHOICES = ("mean_normalized", "center_nearest")
GUIDE_GRID_SOURCE_CHOICES = ("filtered_observation",)
GUIDE_GRID_FILTER_KIND_CHOICES = ("gaussian",)
GUIDE_GRID_UNIT_CHOICES = ("km", "pixel")
GUIDE_GRID_COMPONENT_CHOICES = ("auto", "observation", "magnitude", "east", "north", "both")
RSP_GEOMETRY_CHOICES = ("auto", "rectangle", "triangle")
CUTDE_BACKEND_CHOICES = ("cpp", "cuda", "opencl", "auto")
FAULT_MODEL_TYPE_CHOICES = ("generated_from_trace", "csi_gmt")
FAULT_MODEL_GEOMETRY_CHOICES = ("triangular", "rectangular")
FAULT_MODEL_USE_CHOICES = ("trirb",)
FAULT_MODEL_PLOT_MODE_CHOICES = ("trace", "outline", "edges", "patch_edges", "both")
FAULT_PLOT_STAGE_CHOICES = ("raw", "decim", "all")
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

SAR_OUTPUT_SUFFIX_BY_MODE = {
    "range_offset": "_RngOff",
    "range": "_RngOff",
    "azimuth_offset": "_AziOff",
    "azimuth": "_AziOff",
    "az": "_AziOff",
}

DEFAULT_PROCESSING_REGION = {
    "enabled": False,
    "report": True,
    "report_file": "auto",
    "coord_type": "lonlat",
    "geometry": "box",
    "box": None,
    "polygon": None,
    "polygon_file": None,
}

DEFAULT_INPUT_ADAPTER_CONFIG = {
    "enabled": False,
}

DEFAULT_EXTRACTION_CONFIG = {
    "value_statistic": "mean",
    "error_statistic": "std",
    "coordinate_statistic": "mean",
    "projection_statistic": "mean_normalized",
    "trim_fraction": 0.1,
}

DEFAULT_GUIDE_GRID_CONFIG = {
    "enabled": False,
    "source": "filtered_observation",
    "component": "auto",
    "filter": {
        "kind": "gaussian",
        "sigma": None,
        "unit": "km",
        "radius_sigma": 3.0,
    },
}

DEFAULT_DOWNSAMPLE_REPORT_CONFIG = {
    "enabled": True,
    "report_file": "auto",
    "quality": True,
}

DEFAULT_SAR_CONFIG = {
    "reader": "gamma",
    "mode": None,
    "preset": None,
    "convention": None,
    "directory": "..",
    "outName": "sar",
    "output_check": True,
    "output_suffix": "auto",
    "files": {
        "prefix": None,
        "value": None,
        "metadata": None,
        "geometry": {
            "azimuth": None,
            "incidence": None,
        },
        "projection": {
            "east": None,
            "north": None,
            "up": None,
        },
    },
    "read": {
        "downsample": 1,
        "downsample_for_covar": 1,
        "zero2nan": True,
        "wavelength": None,
        "factor_to_m": 1.0,
    },
    "grid": {
        "engine": None,
        "phase_band": 1,
        "azi_band": 1,
        "inc_band": 1,
        "coord_is_lonlat": None,
        "value_variable": None,
        "projection_variable": None,
        "east_variable": None,
        "north_variable": None,
        "up_variable": None,
        "lon_name": None,
        "lat_name": None,
    },
    "data_filters": {
        "enabled": False,
        "report": True,
        "report_file": "auto",
        "rules": [
            {
                "name": "valid_observation_range",
                "enabled": False,
                "kind": "value_range",
                "value_space": "observation",
                "min": None,
                "max": None,
            },
        ],
    },
    "qc": {
        "summary_percentile": 99.0,
        "plot": {
            "save_fig": True,
            "file_path": "sar_values.png",
            "show": True,
            "rawdownsample4plot": 1,
            "coordrange": None,
            "value_space": "observation",
            "factor4plot": 100.0,
            "vmin": None,
            "vmax": None,
            "symmetry": True,
            "cmap": "cmc.roma_r",
            "colorbar_orientation": "vertical",
            "colorbar_mode": "auto",
            "colorbar_loc": None,
            "colorbar_size": None,
            "colorbar_thickness": None,
            "colorbar_pad": None,
            "cb_label": None,
        },
    },
}

SAR_CONFIG_KEYS = {
    "reader",
    "mode",
    "preset",
    "convention",
    "directory",
    "outName",
    "output_check",
    "output_suffix",
    "files",
    "read",
    "grid",
    "data_filters",
    "qc",
}

DEFAULT_OPTICAL_CONFIG = {
    "directory": "..",
    "outName": "Optical_S2_part1",
    "filename": None,
    "vel_type": "north",
    "factor_to_m": 10.0,
    "ew_band": 1,
    "sn_band": 2,
    "remove_nan": True,
    "output_check": True,
    "data_filters": {
        "enabled": False,
        "report": True,
        "report_file": "auto",
        "rules": [
            {
                "name": "valid_horizontal_component_range",
                "enabled": False,
                "kind": "component_range",
                "components": ["east", "north"],
                "min": None,
                "max": None,
            },
        ],
    },
    "qc": {
        "summary_percentile": 99.0,
        "plot": {
            "save_fig": True,
            "file_path": None,
            "show": True,
            "rawdownsample4plot": 1,
            "coordrange": None,
            "data": ["east", "north"],
            "factor4plot": 1.0,
            "vmin": None,
            "vmax": None,
            "symmetry": True,
            "cmap": "cmc.roma_r",
            "figsize": [3.5, 4.0],
            "title": ["East Component", "North Component"],
            "unified_colorbar": False,
            "dpi": 300,
        },
    },
}

OPTICAL_CONFIG_KEYS = {
    "directory",
    "outName",
    "filename",
    "vel_type",
    "factor_to_m",
    "ew_band",
    "sn_band",
    "remove_nan",
    "output_check",
    "data_filters",
    "qc",
}

SAR_QC_KEYS = {"summary_percentile", "plot"}
SAR_QC_PLOT_KEYS = set(DEFAULT_SAR_CONFIG["qc"]["plot"]) | {
    "colorbar_x",
    "colorbar_y",
    "colorbar_length",
    "colorbar_height",
    "cb_label_loc",
    "tickfontsize",
    "labelfontsize",
    "figsize",
    "fontsize",
    "style",
    "text",
    "text_position",
    "text_fontsize",
    "text_color",
}
OPTICAL_QC_KEYS = {"summary_percentile", "plot"}
OPTICAL_QC_PLOT_KEYS = set(DEFAULT_OPTICAL_CONFIG["qc"]["plot"]) | {
    "add_colorbar",
    "cb_label",
    "trace_color",
    "trace_linewidth",
    "style",
    "fontsize",
    "colorbar_length",
    "colorbar_height",
    "colorbar_x",
    "colorbar_y",
    "colorbar_orientation",
    "cb_label_loc",
    "tickfontsize",
    "labelfontsize",
    "sharey",
    "equal_aspect",
}

def deep_update(base, updates):
    result = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def compact_kwargs(kwargs):
    return {key: value for key, value in kwargs.items() if value is not None}


def _require_mapping(container, key, errors, label):
    value = container.get(key)
    if not isinstance(value, dict):
        errors.append(f"{label}.{key} must be a mapping.")
        return None
    return value


def _require_keys(container, keys, errors, label):
    missing = [key for key in keys if key not in container or container.get(key) is None]
    if missing:
        errors.append(f"{label} missing required keys: {', '.join(missing)}.")


def _require_present_keys(container, keys, errors, label):
    missing = [key for key in keys if key not in container]
    if missing:
        errors.append(f"{label} missing required keys: {', '.join(missing)}.")


def _check_positive_number(container, key, errors, label, allow_zero=False):
    value = container.get(key)
    if value is None:
        return
    try:
        value = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label}.{key} must be numeric.")
        return
    if allow_zero:
        if value < 0.0:
            errors.append(f"{label}.{key} must be non-negative.")
    elif value <= 0.0:
        errors.append(f"{label}.{key} must be positive.")


def _check_positive_int(container, key, errors, label):
    value = container.get(key)
    if value is None:
        return
    if isinstance(value, bool):
        errors.append(f"{label}.{key} must be an integer.")
        return
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label}.{key} must be an integer.")
        return
    if not numeric_value.is_integer():
        errors.append(f"{label}.{key} must be an integer.")
        return
    if numeric_value <= 0:
        errors.append(f"{label}.{key} must be a positive integer.")


def _check_fraction(container, key, errors, label, *, allow_zero=False):
    value = container.get(key)
    if value is None:
        return
    _check_positive_number(container, key, errors, label, allow_zero=allow_zero)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return
    lower_ok = numeric_value >= 0.0 if allow_zero else numeric_value > 0.0
    if not lower_ok or numeric_value > 1.0:
        bracket = "[0, 1]" if allow_zero else "(0, 1]"
        errors.append(f"{label}.{key} must be in {bracket}.")


def _check_number_sequence(value, length, errors, label, allow_none=False):
    if value is None and allow_none:
        return None
    if not isinstance(value, (list, tuple)):
        errors.append(f"{label} must be a sequence with {length} values.")
        return None
    if len(value) != length:
        errors.append(f"{label} must contain {length} values.")
        return None
    converted = []
    for item in value:
        if item is None and allow_none:
            converted.append(None)
            continue
        try:
            converted.append(float(item))
        except (TypeError, ValueError):
            errors.append(f"{label} values must be numeric.")
            return None
    return converted


def _validate_box(value, errors, label, required_keys):
    if value is None:
        return
    if isinstance(value, dict):
        if required_keys and isinstance(required_keys[0], str):
            key_groups = (required_keys,)
        else:
            key_groups = required_keys
        for keys in key_groups:
            if all(key in value for key in keys):
                _require_keys(value, keys, errors, label)
                values = [value.get(key) for key in keys]
                break
        else:
            keys_text = " or ".join("/".join(keys) for keys in key_groups)
            errors.append(f"{label} missing required keys: {keys_text}.")
            return
    else:
        values = value
    converted = _check_number_sequence(values, 4, errors, label)
    if converted is None:
        return
    if converted[0] >= converted[1] or converted[2] >= converted[3]:
        errors.append(f"{label} must satisfy minLon < maxLon and minLat < maxLat.")


def _validate_polygon(value, errors, label):
    if not isinstance(value, (list, tuple)):
        errors.append(f"{label} must be a list of [x, y] or [lon, lat] pairs.")
        return
    if len(value) < 3:
        errors.append(f"{label} must contain at least three points.")
        return
    for index, point in enumerate(value):
        _check_number_sequence(point, 2, errors, f"{label}[{index}]")


def _check_optional_number(container, key, errors, label):
    if container.get(key) is None:
        return
    try:
        float(container[key])
    except (TypeError, ValueError):
        errors.append(f"{label}.{key} must be numeric.")


def _validate_data_filter_box(rule, errors, label):
    if rule.get("boxes") is not None:
        boxes = rule["boxes"]
        if not isinstance(boxes, (list, tuple)):
            errors.append(f"{label}.boxes must be a list of boxes.")
            return
        if len(boxes) == 4 and not isinstance(boxes[0], (dict, list, tuple)):
            _validate_box(boxes, errors, f"{label}.boxes", (
                ("minLon", "maxLon", "minLat", "maxLat"),
                ("minlon", "maxlon", "minlat", "maxlat"),
                ("lon_min", "lon_max", "lat_min", "lat_max"),
            ))
            return
        for index, box in enumerate(boxes):
            _validate_box(box, errors, f"{label}.boxes[{index}]", (
                ("minLon", "maxLon", "minLat", "maxLat"),
                ("minlon", "maxlon", "minlat", "maxlat"),
                ("lon_min", "lon_max", "lat_min", "lat_max"),
            ))
        return
    if rule.get("box") is not None:
        _validate_box(rule["box"], errors, f"{label}.box", (
            ("minLon", "maxLon", "minLat", "maxLat"),
            ("minlon", "maxlon", "minlat", "maxlat"),
            ("lon_min", "lon_max", "lat_min", "lat_max"),
        ))
        return
    _validate_box(rule, errors, label, (
        ("minLon", "maxLon", "minLat", "maxLat"),
        ("minlon", "maxlon", "minlat", "maxlat"),
        ("lon_min", "lon_max", "lat_min", "lat_max"),
    ))


def _validate_data_filter_polygon_entry(value, errors, label):
    if isinstance(value, dict):
        if value.get("file") is not None or value.get("path") is not None:
            return
        if value.get("points") is not None:
            _validate_polygon(value["points"], errors, f"{label}.points")
            return
        errors.append(f"{label} requires points, file, or path.")
        return
    _validate_polygon(value, errors, label)


def _validate_data_filter_polygon(rule, errors, label):
    if rule.get("polygons") is not None:
        polygons = rule["polygons"]
        if not isinstance(polygons, (list, tuple)):
            errors.append(f"{label}.polygons must be a list.")
            return
        for index, polygon in enumerate(polygons):
            _validate_data_filter_polygon_entry(polygon, errors, f"{label}.polygons[{index}]")
        return
    if rule.get("polygon") is not None:
        _validate_data_filter_polygon_entry(rule["polygon"], errors, f"{label}.polygon")
        return
    if rule.get("points") is not None or rule.get("file") is not None or rule.get("path") is not None:
        _validate_data_filter_polygon_entry(rule, errors, label)
        return
    errors.append(f"{label} requires polygon, polygons, points, file, or path.")


def _validate_data_filters(
    data_filters,
    errors,
    label_prefix="sar_config.data_filters",
    kind_choices=SAR_DATA_FILTER_KIND_CHOICES,
):
    if not isinstance(data_filters, dict):
        errors.append(f"{label_prefix} must be a mapping.")
        return
    for key in ("enabled", "report"):
        if key in data_filters and not isinstance(data_filters[key], bool):
            errors.append(f"{label_prefix}.{key} must be true or false.")
    rules = data_filters.get("rules", [])
    if rules is None:
        return
    if not isinstance(rules, list):
        errors.append(f"{label_prefix}.rules must be a list.")
        return
    for index, rule in enumerate(rules):
        label = f"{label_prefix}.rules[{index}]"
        if not isinstance(rule, dict):
            errors.append(f"{label} must be a mapping.")
            continue
        if "enabled" in rule and not isinstance(rule["enabled"], bool):
            errors.append(f"{label}.enabled must be true or false.")
        kind = str(rule.get("kind", "")).replace("-", "_").lower()
        if kind not in kind_choices:
            errors.append(f"{label}.kind must be one of {kind_choices}; got {rule.get('kind')!r}.")
            continue
        if rule.get("enabled") is False:
            continue
        action = str(rule.get("action", "remove_inside")).replace("-", "_").lower()
        if kind in ("lonlat_box", "lonlat_polygon") and action not in DATA_FILTER_ACTION_CHOICES:
            errors.append(f"{label}.action must be one of {DATA_FILTER_ACTION_CHOICES}; got {rule.get('action')!r}.")
        if kind in ("value_abs", "value_range"):
            value_space = str(rule.get("value_space", "observation")).replace("-", "_").lower()
            if value_space != "observation":
                errors.append(f"{label}.value_space currently supports only 'observation'.")
        if kind in ("component_abs", "component_range"):
            components = rule.get("components", rule.get("component", ["east", "north"]))
            if isinstance(components, str):
                components = [components]
            if not isinstance(components, (list, tuple)):
                errors.append(f"{label}.components must be 'east', 'north', or a list of components.")
            else:
                normalized = [str(component).replace("-", "_").lower() for component in components]
                if any(component not in ("east", "north") for component in normalized):
                    errors.append(f"{label}.components supports only 'east' and 'north'.")
        if kind == "value_abs":
            _require_keys(rule, ("threshold",), errors, label)
            _check_positive_number(rule, "threshold", errors, label, allow_zero=True)
        elif kind == "component_abs":
            _require_keys(rule, ("threshold",), errors, label)
            _check_positive_number(rule, "threshold", errors, label, allow_zero=True)
        elif kind == "value_range":
            if rule.get("min") is None and rule.get("max") is None:
                errors.append(f"{label} requires min and/or max.")
            _check_optional_number(rule, "min", errors, label)
            _check_optional_number(rule, "max", errors, label)
            if rule.get("min") is not None and rule.get("max") is not None:
                try:
                    if float(rule["min"]) >= float(rule["max"]):
                        errors.append(f"{label}.min must be smaller than max.")
                except (TypeError, ValueError):
                    pass
        elif kind == "component_range":
            if rule.get("min") is None and rule.get("max") is None:
                errors.append(f"{label} requires min and/or max.")
            _check_optional_number(rule, "min", errors, label)
            _check_optional_number(rule, "max", errors, label)
            if rule.get("min") is not None and rule.get("max") is not None:
                try:
                    if float(rule["min"]) >= float(rule["max"]):
                        errors.append(f"{label}.min must be smaller than max.")
                except (TypeError, ValueError):
                    pass
        elif kind == "lonlat_box":
            _validate_data_filter_box(rule, errors, label)
        elif kind == "lonlat_polygon":
            _validate_data_filter_polygon(rule, errors, label)
        elif kind == "projection_norm":
            if rule.get("tolerance") is None and rule.get("min") is None and rule.get("max") is None:
                errors.append(f"{label} requires tolerance or min/max.")
            _check_optional_number(rule, "target", errors, label)
            _check_positive_number(rule, "tolerance", errors, label, allow_zero=True)
            _check_optional_number(rule, "min", errors, label)
            _check_optional_number(rule, "max", errors, label)
            if rule.get("min") is not None and rule.get("max") is not None:
                try:
                    if float(rule["min"]) >= float(rule["max"]):
                        errors.append(f"{label}.min must be smaller than max.")
                except (TypeError, ValueError):
                    pass
        elif kind == "vector_norm_range":
            if rule.get("tolerance") is None and rule.get("min") is None and rule.get("max") is None:
                errors.append(f"{label} requires tolerance or min/max.")
            _check_optional_number(rule, "target", errors, label)
            _check_positive_number(rule, "tolerance", errors, label, allow_zero=True)
            _check_optional_number(rule, "min", errors, label)
            _check_optional_number(rule, "max", errors, label)
            if rule.get("min") is not None and rule.get("max") is not None:
                try:
                    if float(rule["min"]) >= float(rule["max"]):
                        errors.append(f"{label}.min must be smaller than max.")
                except (TypeError, ValueError):
                    pass


def _processing_region_box_keys(coord_type):
    if coord_type == "xy":
        return (
            ("minX", "maxX", "minY", "maxY"),
            ("minx", "maxx", "miny", "maxy"),
            ("x_min", "x_max", "y_min", "y_max"),
        )
    return (
        ("minLon", "maxLon", "minLat", "maxLat"),
        ("minlon", "maxlon", "minlat", "maxlat"),
        ("lon_min", "lon_max", "lat_min", "lat_max"),
    )


def _validate_processing_region(region, errors, label="processing_region"):
    if not isinstance(region, dict):
        errors.append(f"{label} must be a mapping.")
        return
    for key in ("enabled", "report"):
        if key in region and not isinstance(region[key], bool):
            errors.append(f"{label}.{key} must be true or false.")

    coord_type = _mode_key(region.get("coord_type", "lonlat"))
    if coord_type not in FOCUS_REGION_COORD_CHOICES:
        errors.append(
            f"{label}.coord_type must be one of "
            f"{FOCUS_REGION_COORD_CHOICES}; got {region.get('coord_type')!r}."
        )
        coord_type = "lonlat"

    geometry = _mode_key(region.get("geometry", "box"))
    if geometry not in PROCESSING_REGION_GEOMETRY_CHOICES:
        errors.append(
            f"{label}.geometry must be one of "
            f"{PROCESSING_REGION_GEOMETRY_CHOICES}; got {region.get('geometry')!r}."
        )
        return

    enabled = region.get("enabled", False)
    provided_geometry_fields = [
        key for key in ("box", "polygon", "polygon_file")
        if region.get(key) is not None
    ]
    if len(provided_geometry_fields) > 1:
        errors.append(
            f"Use only one of {label}.box, polygon, or polygon_file."
        )

    if geometry == "box":
        if enabled and region.get("box") is None:
            errors.append(f"{label}.box is required when enabled and geometry='box'.")
        if region.get("box") is not None:
            _validate_box(
                region["box"],
                errors,
                f"{label}.box",
                _processing_region_box_keys(coord_type),
            )
    elif geometry == "polygon":
        if enabled and region.get("polygon") is None:
            errors.append(f"{label}.polygon is required when enabled and geometry='polygon'.")
        if region.get("polygon") is not None:
            _validate_polygon(region["polygon"], errors, f"{label}.polygon")
    elif geometry == "polygon_file":
        if enabled and region.get("polygon_file") is None:
            errors.append(
                f"{label}.polygon_file is required when enabled "
                "and geometry='polygon_file'."
            )
        if region.get("polygon_file") is not None and not isinstance(region["polygon_file"], str):
            errors.append(f"{label}.polygon_file must be a string path.")


def _validate_stage_list(value, errors, label):
    stages = value if isinstance(value, (list, tuple)) else [value]
    for stage in stages:
        key = _mode_key(stage)
        if key not in FAULT_PLOT_STAGE_CHOICES:
            errors.append(f"{label} must contain only {FAULT_PLOT_STAGE_CHOICES}; got {stage!r}.")


def _validate_fault_traces(config, errors):
    entries = config.get("fault_traces", [])
    if isinstance(entries, dict):
        entries = entries.get("sources", [])
    if entries is None:
        return
    if not isinstance(entries, list):
        errors.append("fault_traces must be a list or a mapping with sources.")
        return
    for index, entry in enumerate(entries):
        label = f"fault_traces[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be a mapping.")
            continue
        if "enabled" in entry and not isinstance(entry["enabled"], bool):
            errors.append(f"{label}.enabled must be true or false.")
        if entry.get("enabled", False):
            _require_keys(entry, ("file",), errors, label)
        if entry.get("stages") is not None:
            _validate_stage_list(entry["stages"], errors, f"{label}.stages")
        if entry.get("columns") is not None:
            columns = entry["columns"]
            if not isinstance(columns, (list, tuple)) or len(columns) < 2:
                errors.append(f"{label}.columns must contain at least lon and lat.")


def _validate_fault_model_plot(plot, errors, label):
    if plot in (None, False):
        return
    if plot is True:
        return
    if not isinstance(plot, dict):
        errors.append(f"{label}.plot must be a mapping, true, or false.")
        return
    if plot.get("stages") is not None:
        _validate_stage_list(plot["stages"], errors, f"{label}.plot.stages")
    mode = _mode_key(plot.get("mode", "edges"))
    if mode not in FAULT_MODEL_PLOT_MODE_CHOICES:
        errors.append(
            f"{label}.plot.mode must be one of {FAULT_MODEL_PLOT_MODE_CHOICES}; "
            f"got {plot.get('mode')!r}."
        )


def _validate_fault_models(config, errors):
    entries = config.get("fault_models", [])
    if isinstance(entries, dict):
        entries = entries.get("sources", [])
    if entries is None:
        return
    if not isinstance(entries, list):
        errors.append("fault_models must be a list or a mapping with sources.")
        return
    for index, entry in enumerate(entries):
        label = f"fault_models[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be a mapping.")
            continue
        if "enabled" in entry and not isinstance(entry["enabled"], bool):
            errors.append(f"{label}.enabled must be true or false.")
        model_type = _mode_key(entry.get("type", "csi_gmt" if entry.get("file") is not None else "generated_from_trace"))
        geometry = _mode_key(entry.get("geometry", "triangular"))
        if model_type not in FAULT_MODEL_TYPE_CHOICES:
            errors.append(f"{label}.type must be one of {FAULT_MODEL_TYPE_CHOICES}; got {entry.get('type')!r}.")
        if geometry not in FAULT_MODEL_GEOMETRY_CHOICES:
            errors.append(f"{label}.geometry must be one of {FAULT_MODEL_GEOMETRY_CHOICES}; got {entry.get('geometry')!r}.")
        if entry.get("enabled", False):
            if model_type == "csi_gmt":
                _require_keys(entry, ("file",), errors, label)
            elif model_type == "generated_from_trace":
                _require_keys(
                    entry,
                    ("trace_file", "dip_angle", "dip_direction", "top_size", "bottom_size"),
                    errors,
                    label,
                )
                if geometry != "triangular":
                    errors.append(f"{label} generated_from_trace currently requires geometry: triangular.")
        if entry.get("use_for") is not None:
            use_for = entry["use_for"] if isinstance(entry["use_for"], (list, tuple)) else [entry["use_for"]]
            for value in use_for:
                key = _mode_key(value)
                if key not in FAULT_MODEL_USE_CHOICES:
                    errors.append(f"{label}.use_for must contain only {FAULT_MODEL_USE_CHOICES}; got {value!r}.")
                if key == "trirb" and geometry != "triangular":
                    errors.append(f"{label}.use_for: trirb requires geometry: triangular.")
        _validate_fault_model_plot(entry.get("plot"), errors, label)


def _validate_extraction_config(extraction, errors, label="downsample.extraction"):
    if not isinstance(extraction, dict):
        errors.append(f"{label} must be a mapping.")
        return
    choices = {
        "value_statistic": EXTRACTION_VALUE_STAT_CHOICES,
        "error_statistic": EXTRACTION_ERROR_STAT_CHOICES,
        "coordinate_statistic": EXTRACTION_COORDINATE_STAT_CHOICES,
        "projection_statistic": EXTRACTION_PROJECTION_STAT_CHOICES,
    }
    for key, allowed in choices.items():
        value = extraction.get(key, DEFAULT_EXTRACTION_CONFIG[key])
        if value not in allowed:
            errors.append(f"{label}.{key} must be one of {allowed}; got {value!r}.")
    _check_positive_number(extraction, "trim_fraction", errors, label, allow_zero=True)
    if extraction.get("trim_fraction") is not None:
        try:
            trim_fraction = float(extraction["trim_fraction"])
        except (TypeError, ValueError):
            return
        if trim_fraction >= 0.5:
            errors.append(f"{label}.trim_fraction must be smaller than 0.5.")


def _validate_guide_grid_config(guide_grid, method, errors, label="downsample.guide_grid"):
    if not isinstance(guide_grid, dict):
        errors.append(f"{label} must be a mapping.")
        return
    enabled = guide_grid.get("enabled", False)
    if not isinstance(enabled, bool):
        errors.append(f"{label}.enabled must be true or false.")
        enabled = False
    if enabled and method not in ("std", "data"):
        errors.append(f"{label} can be enabled only with downsample.method 'std' or 'data'.")
    source = guide_grid.get("source", "filtered_observation")
    if source not in GUIDE_GRID_SOURCE_CHOICES:
        errors.append(f"{label}.source must be one of {GUIDE_GRID_SOURCE_CHOICES}; got {source!r}.")
    component = guide_grid.get("component", "auto")
    if component not in GUIDE_GRID_COMPONENT_CHOICES:
        errors.append(f"{label}.component must be one of {GUIDE_GRID_COMPONENT_CHOICES}; got {component!r}.")
    guide_filter = guide_grid.get("filter", {})
    if not isinstance(guide_filter, dict):
        errors.append(f"{label}.filter must be a mapping.")
        return
    kind = guide_filter.get("kind", "gaussian")
    if kind not in GUIDE_GRID_FILTER_KIND_CHOICES:
        errors.append(f"{label}.filter.kind must be one of {GUIDE_GRID_FILTER_KIND_CHOICES}; got {kind!r}.")
    unit = guide_filter.get("unit", "km")
    if unit not in GUIDE_GRID_UNIT_CHOICES:
        errors.append(f"{label}.filter.unit must be one of {GUIDE_GRID_UNIT_CHOICES}; got {unit!r}.")
    if enabled and guide_filter.get("sigma") is None:
        errors.append(f"{label}.filter.sigma is required when guide_grid.enabled is true.")
    _check_positive_number(guide_filter, "sigma", errors, f"{label}.filter")
    _check_positive_number(guide_filter, "radius_sigma", errors, f"{label}.filter")


def _validate_downsample_report_config(report, errors, label="downsample.report"):
    if not isinstance(report, dict):
        errors.append(f"{label} must be a mapping.")
        return
    if "enabled" in report and not isinstance(report["enabled"], bool):
        errors.append(f"{label}.enabled must be true or false.")
    if "quality" in report and not isinstance(report["quality"], bool):
        errors.append(f"{label}.quality must be true or false.")
    report_file = report.get("report_file", "auto")
    if report_file not in (None, False) and not isinstance(report_file, str):
        errors.append(f"{label}.report_file must be 'auto', a path string, null, or false.")


def _reject_legacy_downsample_fields(config, field_names, errors, label):
    for field_name in field_names:
        if field_name in config:
            old_key = f"{label}.{field_name}"
            new_key = LEGACY_DOWNSAMPLE_FIELD_RENAMES.get(old_key)
            if new_key:
                errors.append(f"{old_key} was renamed to {new_key}.")
            else:
                errors.append(f"{old_key} is no longer supported.")


def _mode_key(mode):
    return None if mode is None else str(mode).replace("-", "_").lower()


def normalize_sar_reader_key(reader):
    key = _mode_key(reader)
    try:
        return SAR_READER_ALIASES[key]
    except KeyError:
        return key


def _reject_unsupported_keys(config, allowed_keys, label):
    unknown = sorted(set(config) - set(allowed_keys))
    if unknown:
        allowed = ", ".join(sorted(allowed_keys))
        raise ValueError(
            f"Unsupported {label} keys: {', '.join(unknown)}. "
            f"Allowed keys are: {allowed}."
        )


def normalize_sar_config(sar_config, validate=True):
    raw = deepcopy(sar_config or {})
    _reject_unsupported_keys(raw, SAR_CONFIG_KEYS, "sar_config")
    normalized = deep_update(DEFAULT_SAR_CONFIG, {})

    for key in ("reader", "mode", "preset", "directory", "outName", "output_check", "output_suffix"):
        if key in raw:
            normalized[key] = raw[key]

    normalized["reader"] = normalize_sar_reader_key(normalized["reader"])
    normalized["mode"] = _mode_key(normalized["mode"])
    normalized["convention"] = raw.get("convention", normalized["convention"])

    files = deepcopy(normalized["files"])
    files = deep_update(files, raw.get("files", {}) or {})
    normalized["files"] = files

    read = deepcopy(normalized["read"])
    read.update(raw.get("read", {}) or {})
    normalized["read"] = read

    grid = deepcopy(normalized["grid"])
    grid.update(raw.get("grid", {}) or {})
    normalized["grid"] = grid

    data_filters = deepcopy(normalized["data_filters"])
    data_filters.update(raw.get("data_filters", {}) or {})
    normalized["data_filters"] = data_filters

    qc = deepcopy(normalized["qc"])
    qc.update(raw.get("qc", {}) or {})
    plot = deepcopy(DEFAULT_SAR_CONFIG["qc"]["plot"])
    plot.update((raw.get("qc", {}) or {}).get("plot", {}) or {})
    qc["plot"] = plot
    normalized["qc"] = qc

    if validate:
        validate_sar_config(normalized)
    return normalized


def validate_sar_config(sar_config):
    errors = []
    reader = sar_config.get("reader")
    mode = sar_config.get("mode")

    if reader not in SAR_READER_CHOICES:
        errors.append(f"sar_config.reader must be one of {SAR_READER_CHOICES}; got {reader!r}.")

    semantic_count = sum(
        sar_config.get(key) is not None for key in ("mode", "preset", "convention")
    )
    if semantic_count != 1:
        errors.append("Use exactly one of sar_config.mode, sar_config.preset, or sar_config.convention.")

    if mode is not None and mode not in SAR_MODE_CHOICES:
        errors.append(f"sar_config.mode must be one of {SAR_MODE_CHOICES}; got {mode!r}.")
    if reader == "hyp3" and mode not in (None, "los_displacement", "unwrapped_phase", "phase_los"):
        errors.append(
            "reader='hyp3' supports mode='los_displacement' or "
            "mode='unwrapped_phase'/'phase_los' in the built-in config."
        )
    convention = sar_config.get("convention") or {}
    if convention and not isinstance(convention, dict):
        errors.append("sar_config.convention must be a mapping when provided.")
        convention = {}
    direct_convention_keys = {
        "input_projection_role",
        "input_projection_convention",
    }
    angle_convention_keys = {
        "azimuth_reference",
        "azimuth_unit",
        "azimuth_direction",
        "incidence_reference",
        "incidence_unit",
        "input_azimuth_role",
    }
    if reader == "gmtsar":
        illegal = sorted(key for key in angle_convention_keys if key in convention)
        if illegal:
            errors.append(
                "reader='gmtsar' uses direct ENU projection conventions; "
                f"remove angle-raster convention keys: {', '.join(illegal)}."
            )
    else:
        illegal = sorted(key for key in direct_convention_keys if key in convention)
        if illegal:
            errors.append(
                f"reader={reader!r} uses azimuth/incidence angle conventions; "
                f"remove direct-projection convention keys: {', '.join(illegal)}."
            )

    files = sar_config.get("files", {})
    if not isinstance(files, dict):
        errors.append("sar_config.files must be a mapping.")
        files = {}
    allowed_file_keys = {"prefix", "value", "metadata", "geometry", "projection"}
    unknown_file_keys = sorted(set(files) - allowed_file_keys)
    if unknown_file_keys:
        errors.append(
            "Unsupported sar_config.files keys: "
            f"{', '.join(unknown_file_keys)}. Use prefix, value, metadata, "
            "geometry.azimuth/incidence, and projection.east/north/up."
        )
    geometry = files.get("geometry") or {}
    projection = files.get("projection") or {}
    if not isinstance(geometry, dict):
        errors.append("sar_config.files.geometry must be a mapping.")
        geometry = {}
    if not isinstance(projection, dict):
        errors.append("sar_config.files.projection must be a mapping.")
        projection = {}
    allowed_geometry_keys = {"azimuth", "incidence"}
    unknown_geometry_keys = sorted(set(geometry) - allowed_geometry_keys)
    if unknown_geometry_keys:
        errors.append(
            "Unsupported sar_config.files.geometry keys: "
            f"{', '.join(unknown_geometry_keys)}. Use azimuth and incidence."
        )
    allowed_projection_keys = {"east", "north", "up"}
    unknown_projection_keys = sorted(set(projection) - allowed_projection_keys)
    if unknown_projection_keys:
        errors.append(
            "Unsupported sar_config.files.projection keys: "
            f"{', '.join(unknown_projection_keys)}. Use east, north, and up."
        )

    prefix = files.get("prefix")
    explicit_value = files.get("value")
    explicit_angle_files = [
        explicit_value,
        files.get("metadata"),
        geometry.get("azimuth"),
        geometry.get("incidence"),
    ]
    direct_explicit = [projection.get(key) for key in ("east", "north", "up")]
    if reader == "gmtsar":
        if prefix:
            errors.append("reader='gmtsar' requires explicit files.value and files.projection; do not use files.prefix.")
        convention_observation_type = _mode_key(convention.get("observation_type"))
        preset_key = _mode_key(sar_config.get("preset"))
        is_azimuth_observation = (
            mode in ("azimuth_offset", "azimuth", "az")
            or convention_observation_type == "azimuth_offset"
            or (preset_key is not None and "azimuth" in preset_key)
        )
        if not explicit_value:
            errors.append("reader='gmtsar' requires files.value.")
        if not projection.get("east"):
            errors.append("reader='gmtsar' requires files.projection.east.")
        if not projection.get("north"):
            errors.append("reader='gmtsar' requires files.projection.north.")
        if not is_azimuth_observation and not projection.get("up"):
            errors.append("reader='gmtsar' requires files.projection.up except for direct azimuth_offset inputs.")
        if files.get("metadata") or geometry.get("azimuth") or geometry.get("incidence"):
            errors.append("reader='gmtsar' does not use files.metadata or files.geometry.")
    else:
        if any(direct_explicit):
            errors.append("files.projection.* is only supported for reader='gmtsar'.")
        if prefix and any(explicit_angle_files):
            errors.append("Use either sar_config.files.prefix or explicit files.value/metadata/geometry entries, not both.")
        if not prefix and not any(explicit_angle_files):
            errors.append("Set sar_config.files.prefix or explicit files.value/metadata/geometry entries.")
    if reader != "gmtsar" and any(explicit_angle_files):
        required_files = {
            "files.value": explicit_value,
            "files.metadata": files.get("metadata"),
            "files.geometry.azimuth": geometry.get("azimuth"),
            "files.geometry.incidence": geometry.get("incidence"),
        }
        if reader in ("gamma_tiff", "hyp3"):
            required_files.pop("files.metadata")
        missing_files = [key for key, value in required_files.items() if not value]
        if missing_files:
            errors.append(
                "Explicit SAR files are incomplete for "
                f"reader={reader!r}; missing: {', '.join(missing_files)}."
            )

    read = sar_config.get("read", {})
    if not isinstance(read, dict):
        errors.append("sar_config.read must be a mapping.")
        read = {}
    allowed_read_keys = {
        "downsample",
        "downsample_for_covar",
        "zero2nan",
        "wavelength",
        "factor_to_m",
        "overrides",
    }
    unknown_read_keys = sorted(set(read) - allowed_read_keys)
    if unknown_read_keys:
        errors.append(
            "Unsupported sar_config.read keys: "
            f"{', '.join(unknown_read_keys)}. Put band, variable, engine, "
            "and coordinate settings under sar_config.grid."
        )
    for key in ("downsample", "downsample_for_covar"):
        _check_positive_int(read, key, errors, "sar_config.read")
    _check_positive_number(read, "factor_to_m", errors, "sar_config.read")
    grid = sar_config.get("grid", {})
    if not isinstance(grid, dict):
        errors.append("sar_config.grid must be a mapping.")
        grid = {}
    allowed_grid_keys = {
        "engine",
        "phase_band",
        "azi_band",
        "inc_band",
        "coord_is_lonlat",
        "value_variable",
        "projection_variable",
        "east_variable",
        "north_variable",
        "up_variable",
        "lon_name",
        "lat_name",
    }
    unknown_grid_keys = sorted(set(grid) - allowed_grid_keys)
    if unknown_grid_keys:
        errors.append(
            "Unsupported sar_config.grid keys: "
            f"{', '.join(unknown_grid_keys)}."
        )
    for key in ("phase_band", "azi_band", "inc_band"):
        _check_positive_int(grid, key, errors, "sar_config.grid")
    if grid.get("coord_is_lonlat") not in (None, True, False):
        errors.append("sar_config.grid.coord_is_lonlat must be true, false, or null.")

    data_filters = sar_config.get("data_filters", {})
    _validate_data_filters(data_filters, errors)

    qc = sar_config.get("qc", {})
    if not isinstance(qc, dict):
        errors.append("sar_config.qc must be a mapping.")
        qc = {}
    unknown_qc_keys = sorted(set(qc) - SAR_QC_KEYS)
    if unknown_qc_keys:
        errors.append(
            "Unsupported sar_config.qc keys: "
            f"{', '.join(unknown_qc_keys)}."
        )
    if "outlier_threshold" in qc:
        errors.append(
            "sar_config.qc.outlier_threshold has been replaced by "
            "sar_config.data_filters.rules with kind='value_abs'."
        )

    plot = qc.get("plot", {})
    if not isinstance(plot, dict):
        errors.append("sar_config.qc.plot must be a mapping.")
        plot = {}
    unknown_plot_keys = sorted(set(plot) - SAR_QC_PLOT_KEYS)
    if unknown_plot_keys:
        errors.append(
            "Unsupported sar_config.qc.plot keys: "
            f"{', '.join(unknown_plot_keys)}."
        )
    if plot.get("value_space") not in ("observation", "raw"):
        errors.append("sar_config.qc.plot.value_space must be 'observation' or 'raw'.")
    if plot.get("factor4plot") is not None:
        try:
            factor4plot = float(plot["factor4plot"])
        except (TypeError, ValueError):
            errors.append("sar_config.qc.plot.factor4plot must be numeric.")
        else:
            if factor4plot == 0.0:
                errors.append("sar_config.qc.plot.factor4plot must be non-zero.")
    if plot.get("coordrange") is not None:
        values = _check_number_sequence(plot["coordrange"], 4, errors, "sar_config.qc.plot.coordrange")
        if values is not None and (values[0] >= values[1] or values[2] >= values[3]):
            errors.append(
                "sar_config.qc.plot.coordrange must satisfy "
                "minlon < maxlon and minlat < maxlat."
            )

    if errors:
        raise ValueError("Invalid SAR downsample config:\n- " + "\n- ".join(errors))
    return True


def _validate_scalar_or_pair(value, errors, label):
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            errors.append(f"{label} must be numeric or contain two values for east/north.")
            return
        for index, item in enumerate(value):
            if item is None:
                continue
            try:
                float(item)
            except (TypeError, ValueError):
                errors.append(f"{label}[{index}] must be numeric or null.")
        return
    try:
        float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be numeric, null, or a two-value list.")


def normalize_optical_config(optical_config, validate=True):
    raw = deepcopy(optical_config or {})
    _reject_unsupported_keys(raw, OPTICAL_CONFIG_KEYS, "optical_config")
    normalized = deep_update(DEFAULT_OPTICAL_CONFIG, {})

    for key in (
        "directory",
        "outName",
        "filename",
        "vel_type",
        "factor_to_m",
        "ew_band",
        "sn_band",
        "remove_nan",
        "output_check",
    ):
        if key in raw:
            normalized[key] = raw[key]

    if normalized.get("vel_type") is not None:
        normalized["vel_type"] = _mode_key(normalized["vel_type"])

    data_filters = deepcopy(normalized["data_filters"])
    data_filters.update(raw.get("data_filters", {}) or {})
    normalized["data_filters"] = data_filters

    qc = deepcopy(normalized["qc"])
    qc.update(raw.get("qc", {}) or {})
    plot = deepcopy(DEFAULT_OPTICAL_CONFIG["qc"]["plot"])
    plot.update((raw.get("qc", {}) or {}).get("plot", {}) or {})
    qc["plot"] = plot
    normalized["qc"] = qc

    if validate:
        validate_optical_config(normalized)
    return normalized


def validate_optical_config(optical_config):
    errors = []
    _require_keys(optical_config, ("outName", "filename"), errors, "optical_config")
    if "directory" in optical_config and optical_config["directory"] is not None and not isinstance(optical_config["directory"], str):
        errors.append("optical_config.directory must be a string path.")
    if "factor_to_m" in optical_config:
        _check_positive_number(optical_config, "factor_to_m", errors, "optical_config")
    for key in ("ew_band", "sn_band"):
        _check_positive_int(optical_config, key, errors, "optical_config")
    if "remove_nan" in optical_config and not isinstance(optical_config["remove_nan"], bool):
        errors.append("optical_config.remove_nan must be true or false.")
    if "output_check" in optical_config and not isinstance(optical_config["output_check"], bool):
        errors.append("optical_config.output_check must be true or false.")
    if "vel_type" in optical_config and optical_config["vel_type"] not in ("east", "north"):
        errors.append("optical_config.vel_type must be 'east' or 'north'.")

    _validate_data_filters(
        optical_config.get("data_filters", {}),
        errors,
        label_prefix="optical_config.data_filters",
        kind_choices=OPTICAL_DATA_FILTER_KIND_CHOICES,
    )

    qc = optical_config.get("qc", {})
    if not isinstance(qc, dict):
        errors.append("optical_config.qc must be a mapping.")
        qc = {}
    unknown_qc_keys = sorted(set(qc) - OPTICAL_QC_KEYS)
    if unknown_qc_keys:
        errors.append(
            "Unsupported optical_config.qc keys: "
            f"{', '.join(unknown_qc_keys)}."
        )
    _check_positive_number(qc, "summary_percentile", errors, "optical_config.qc")
    plot = qc.get("plot", {})
    if not isinstance(plot, dict):
        errors.append("optical_config.qc.plot must be a mapping.")
        plot = {}
    unknown_plot_keys = sorted(set(plot) - OPTICAL_QC_PLOT_KEYS)
    if unknown_plot_keys:
        errors.append(
            "Unsupported optical_config.qc.plot keys: "
            f"{', '.join(unknown_plot_keys)}."
        )
    for key in ("save_fig", "show", "unified_colorbar", "symmetry"):
        if key in plot and not isinstance(plot[key], bool):
            errors.append(f"optical_config.qc.plot.{key} must be true or false.")
    _check_positive_int(plot, "rawdownsample4plot", errors, "optical_config.qc.plot")
    _check_positive_number(plot, "factor4plot", errors, "optical_config.qc.plot")
    _check_positive_int(plot, "dpi", errors, "optical_config.qc.plot")
    if plot.get("coordrange") is not None:
        values = _check_number_sequence(plot["coordrange"], 4, errors, "optical_config.qc.plot.coordrange")
        if values is not None and (values[0] >= values[1] or values[2] >= values[3]):
            errors.append(
                "optical_config.qc.plot.coordrange must satisfy "
                "minlon < maxlon and minlat < maxlat."
            )
    if plot.get("data") is not None:
        data = plot["data"]
        if isinstance(data, str):
            data = [data]
        if not isinstance(data, (list, tuple)):
            errors.append("optical_config.qc.plot.data must be 'east', 'north', or a list.")
        else:
            allowed = ("east", "north")
            if any(str(item).replace("-", "_").lower() not in allowed for item in data):
                errors.append("optical_config.qc.plot.data supports only 'east' and 'north'.")
    if plot.get("figsize") is not None:
        figsize = _check_number_sequence(plot["figsize"], 2, errors, "optical_config.qc.plot.figsize")
        if figsize is not None and any(value <= 0.0 for value in figsize):
            errors.append("optical_config.qc.plot.figsize values must be positive.")
    _validate_scalar_or_pair(plot.get("vmin"), errors, "optical_config.qc.plot.vmin")
    _validate_scalar_or_pair(plot.get("vmax"), errors, "optical_config.qc.plot.vmax")

    if errors:
        raise ValueError("Invalid optical downsample config:\n- " + "\n- ".join(errors))
    return True


def normalize_downsample_config(config, validate=True):
    normalized = deepcopy(config or {})
    data_type = normalized.get("data_type")
    if data_type not in ("sar", "optical"):
        raise ValueError("data_type must be 'sar' or 'optical'.")

    input_adapter = deepcopy(DEFAULT_INPUT_ADAPTER_CONFIG)
    input_adapter.update(normalized.get("input_adapter", {}) or {})
    normalized["input_adapter"] = input_adapter

    general = deepcopy(normalized.get("general", {}) or {})
    origin = general.get("origin")
    has_lon0 = general.get("lon0") is not None
    has_lat0 = general.get("lat0") is not None
    if origin is None:
        origin = "manual" if has_lon0 and has_lat0 else "auto"
    origin = str(origin).replace("-", "_").lower()
    if has_lon0 and has_lat0:
        origin = "manual"
    general["origin"] = origin
    normalized["general"] = general

    processing_region = deepcopy(DEFAULT_PROCESSING_REGION)
    processing_region.update(normalized.get("processing_region", {}) or {})
    if processing_region.get("coord_type") is not None:
        processing_region["coord_type"] = _mode_key(processing_region["coord_type"])
    if processing_region.get("geometry") is not None:
        processing_region["geometry"] = _mode_key(processing_region["geometry"])
    normalized["processing_region"] = processing_region

    if data_type == "sar":
        normalized["sar_config"] = normalize_sar_config(
            normalized.get("sar_config", {}),
            validate=validate and not input_adapter.get("enabled", False),
        )
    elif data_type == "optical":
        if "optical_config" not in normalized and not input_adapter.get("enabled", False):
            raise ValueError("data_type='optical' requires optical_config.")
        normalized["optical_config"] = normalize_optical_config(
            normalized.get("optical_config", {}),
            validate=validate and not input_adapter.get("enabled", False),
        )

    downsample = normalized.get("downsample")
    if isinstance(downsample, dict) and downsample.get("method") is not None:
        downsample["method"] = _mode_key(downsample["method"])
        raw_compute = downsample.get("compute", {}) or {}
        if isinstance(raw_compute, dict):
            compute = deepcopy(raw_compute)
            cutde_backend = compute.get("cutde_backend", "cpp")
            compute["cutde_backend"] = _mode_key(cutde_backend)
            downsample["compute"] = compute
        std_config = downsample.get("std_config")
        if isinstance(std_config, dict):
            focus_region = std_config.get("focus_region")
            if isinstance(focus_region, dict) and focus_region.get("coord_type") is not None:
                focus_region["coord_type"] = _mode_key(focus_region["coord_type"])
        data_config = downsample.get("data_config")
        if isinstance(data_config, dict) and data_config.get("split_metric") is not None:
            data_config["split_metric"] = _mode_key(data_config["split_metric"])
        if isinstance(std_config, dict) and std_config.get("split_metric_correction") is not None:
            std_config["split_metric_correction"] = _mode_key(std_config["split_metric_correction"])
        raw_extraction = downsample.get("extraction", {}) or {}
        if isinstance(raw_extraction, dict):
            extraction = deepcopy(DEFAULT_EXTRACTION_CONFIG)
            extraction.update(raw_extraction)
            for key in ("value_statistic", "error_statistic", "coordinate_statistic", "projection_statistic"):
                if extraction.get(key) is not None:
                    extraction[key] = _mode_key(extraction[key])
            downsample["extraction"] = extraction

        raw_guide_grid = downsample.get("guide_grid", {}) or {}
        if isinstance(raw_guide_grid, dict):
            guide_grid = deepcopy(DEFAULT_GUIDE_GRID_CONFIG)
            guide_grid.update(raw_guide_grid)
            guide_filter = deepcopy(DEFAULT_GUIDE_GRID_CONFIG["filter"])
            raw_guide_filter = raw_guide_grid.get("filter", {}) or {}
            if isinstance(raw_guide_filter, dict):
                guide_filter.update(raw_guide_filter)
            for key in ("source", "component"):
                if guide_grid.get(key) is not None:
                    guide_grid[key] = _mode_key(guide_grid[key])
            for key in ("kind", "unit"):
                if guide_filter.get(key) is not None:
                    guide_filter[key] = _mode_key(guide_filter[key])
            guide_grid["filter"] = guide_filter
            downsample["guide_grid"] = guide_grid
        raw_report = downsample.get("report", {}) or {}
        if isinstance(raw_report, dict):
            report = deepcopy(DEFAULT_DOWNSAMPLE_REPORT_CONFIG)
            report.update(raw_report)
            downsample["report"] = report

    normalized.setdefault("fault_traces", [])
    normalized.setdefault("fault_models", [])
    if validate:
        validate_downsample_config(normalized)
    return normalized


def validate_downsample_config(config):
    errors = []
    data_type = config.get("data_type")
    if data_type not in ("sar", "optical"):
        errors.append("data_type must be 'sar' or 'optical'.")

    input_adapter = config.get("input_adapter", {}) or {}
    if not isinstance(input_adapter, dict):
        errors.append("input_adapter must be a mapping.")
        input_adapter = {}
    if "enabled" in input_adapter and not isinstance(input_adapter["enabled"], bool):
        errors.append("input_adapter.enabled must be true or false.")
    adapter_enabled = bool(input_adapter.get("enabled", False))

    general = _require_mapping(config, "general", errors, "config")
    if general is not None:
        origin = general.get("origin")
        if origin not in ORIGIN_MODE_CHOICES:
            errors.append(f"general.origin must be one of {ORIGIN_MODE_CHOICES}; got {origin!r}.")
        has_lon0 = general.get("lon0") is not None
        has_lat0 = general.get("lat0") is not None
        if origin == "manual":
            _require_keys(general, ("lon0", "lat0"), errors, "general")
        elif has_lon0 != has_lat0:
            errors.append("general.lon0 and general.lat0 must be set together, or both left null for origin='auto'.")
        for key in ("lon0", "lat0"):
            if general.get(key) is not None:
                try:
                    float(general[key])
                except (TypeError, ValueError):
                    errors.append(f"general.{key} must be numeric when provided.")

    _validate_processing_region(config.get("processing_region", {}), errors)
    _validate_fault_traces(config, errors)
    _validate_fault_models(config, errors)

    covar = _require_mapping(config, "covar", errors, "config")
    if covar is not None:
        _require_keys(covar, ("function", "frac", "every", "distmax", "rampEst"), errors, "covar")
        missing_policy = str(covar.get("missing_policy", "existing_or_identity")).replace("-", "_").lower()
        if missing_policy not in COVAR_MISSING_POLICY_CHOICES:
            errors.append(
                "covar.missing_policy must be one of "
                f"{COVAR_MISSING_POLICY_CHOICES}; got {covar.get('missing_policy')!r}."
            )
        _check_positive_number(covar, "frac", errors, "covar")
        _check_positive_number(covar, "every", errors, "covar")
        _check_positive_number(covar, "distmax", errors, "covar")
        if covar.get("mask_out") is not None:
            mask_out = covar["mask_out"]
            mask_keys = (
                ("minLon", "maxLon", "minLat", "maxLat"),
                ("minlon", "maxlon", "minlat", "maxlat"),
            )
            if isinstance(mask_out, dict):
                _validate_box(mask_out, errors, "covar.mask_out", mask_keys)
            elif isinstance(mask_out, (list, tuple)) and len(mask_out) == 4 and not isinstance(mask_out[0], (dict, list, tuple)):
                _validate_box(mask_out, errors, "covar.mask_out", mask_keys)
            elif isinstance(mask_out, (list, tuple)):
                for index, item in enumerate(mask_out):
                    _validate_box(item, errors, f"covar.mask_out[{index}]", mask_keys)
            else:
                errors.append("covar.mask_out must be null, one box, or a list of boxes.")

    downsample = config.get("downsample", {})
    if not isinstance(downsample, dict):
        errors.append("config.downsample must be a mapping.")
        downsample = {}
    compute = downsample.get("compute", {})
    if compute is None:
        compute = {}
    if not isinstance(compute, dict):
        errors.append("downsample.compute must be a mapping.")
    else:
        cutde_backend = compute.get("cutde_backend", "cpp")
        if cutde_backend not in CUTDE_BACKEND_CHOICES:
            errors.append(
                "downsample.compute.cutde_backend must be one of "
                f"{CUTDE_BACKEND_CHOICES}; got {cutde_backend!r}."
            )
    method = downsample.get("method")
    if method not in DOWNSAMPLE_METHOD_CHOICES:
        errors.append(
            "downsample.method must be one of "
            f"{DOWNSAMPLE_METHOD_CHOICES}; got {method!r}."
        )
    _validate_extraction_config(
        downsample.get("extraction", DEFAULT_EXTRACTION_CONFIG),
        errors,
    )
    _validate_guide_grid_config(
        downsample.get("guide_grid", DEFAULT_GUIDE_GRID_CONFIG),
        method,
        errors,
    )
    _validate_downsample_report_config(
        downsample.get("report", DEFAULT_DOWNSAMPLE_REPORT_CONFIG),
        errors,
    )
    if method == "std":
        std_config = _require_mapping(downsample, "std_config", errors, "downsample")
        if std_config is not None:
            _reject_legacy_downsample_fields(
                std_config,
                ("tolerance", "std_threshold", "correction", "smooth"),
                errors,
                "downsample.std_config",
            )
            _require_keys(
                std_config,
                ("startingsize", "minimumsize", "min_valid_fraction", "split_std_threshold"),
                errors,
                "downsample.std_config",
            )
            for key in ("startingsize", "minimumsize", "split_std_threshold"):
                _check_positive_number(std_config, key, errors, "downsample.std_config")
            _check_fraction(std_config, "min_valid_fraction", errors, "downsample.std_config")
            _check_positive_int(std_config, "decimorig", errors, "downsample.std_config")
            _check_positive_int(std_config, "itmax", errors, "downsample.std_config")
            _check_positive_number(std_config, "split_metric_smoothing", errors, "downsample.std_config")
            if "use_variance" in std_config and not isinstance(std_config["use_variance"], bool):
                errors.append("downsample.std_config.use_variance must be true or false.")
            correction = std_config.get("split_metric_correction", "std")
            if correction not in STD_CORRECTION_CHOICES:
                errors.append(
                    "downsample.std_config.split_metric_correction must be one of "
                    f"{STD_CORRECTION_CHOICES}; got {correction!r}."
                )
            amplitude_stat = std_config.get("amplitude_stat", "mean_abs")
            if amplitude_stat not in AMPLITUDE_STAT_CHOICES:
                errors.append(
                    "downsample.std_config.amplitude_stat must be one of "
                    f"{AMPLITUDE_STAT_CHOICES}; got {amplitude_stat!r}."
                )

            focus_region = std_config.get("focus_region")
            if focus_region is not None:
                if not isinstance(focus_region, dict):
                    errors.append("downsample.std_config.focus_region must be a mapping.")
                else:
                    enabled = focus_region.get("enabled", False)
                    if not isinstance(enabled, bool):
                        errors.append("downsample.std_config.focus_region.enabled must be true or false.")
                    coord_type = focus_region.get("coord_type", "lonlat")
                    if coord_type not in FOCUS_REGION_COORD_CHOICES:
                        errors.append(
                            "downsample.std_config.focus_region.coord_type must be one of "
                            f"{FOCUS_REGION_COORD_CHOICES}; got {focus_region.get('coord_type')!r}."
                        )
                    polygon = focus_region.get("polygon")
                    polygon_file = focus_region.get("polygon_file")
                    if polygon is not None and polygon_file is not None:
                        errors.append(
                            "Use only one of downsample.std_config.focus_region.polygon "
                            "or polygon_file."
                        )
                    if enabled and polygon is None and polygon_file is None:
                        errors.append(
                            "downsample.std_config.focus_region requires polygon or "
                            "polygon_file when focus_region.enabled is true."
                        )
                    if polygon is not None:
                        _validate_polygon(polygon, errors, "downsample.std_config.focus_region.polygon")
                    if polygon_file is not None and not isinstance(polygon_file, str):
                        errors.append("downsample.std_config.focus_region.polygon_file must be a string path.")
                    _check_positive_int(
                        focus_region,
                        "max_splits_outside",
                        errors,
                        "downsample.std_config.focus_region",
                    )

            high_value_refinement = std_config.get("high_value_refinement")
            if high_value_refinement is not None:
                if not isinstance(high_value_refinement, dict):
                    errors.append("downsample.std_config.high_value_refinement must be a mapping.")
                else:
                    enabled = high_value_refinement.get("enabled", False)
                    if not isinstance(enabled, bool):
                        errors.append("downsample.std_config.high_value_refinement.enabled must be true or false.")
                    _check_positive_number(
                        high_value_refinement,
                        "high_value_ratio",
                        errors,
                        "downsample.std_config.high_value_refinement",
                    )
                    _check_positive_int(
                        high_value_refinement,
                        "min_splits",
                        errors,
                        "downsample.std_config.high_value_refinement",
                    )
                    _check_positive_number(
                        high_value_refinement,
                        "reference_max_value",
                        errors,
                        "downsample.std_config.high_value_refinement",
                    )

            low_amplitude_cap = std_config.get("low_amplitude_cap")
            if low_amplitude_cap is not None:
                if not isinstance(low_amplitude_cap, dict):
                    errors.append("downsample.std_config.low_amplitude_cap must be a mapping.")
                else:
                    enabled = low_amplitude_cap.get("enabled", False)
                    if not isinstance(enabled, bool):
                        errors.append("downsample.std_config.low_amplitude_cap.enabled must be true or false.")
                    _check_positive_number(
                        low_amplitude_cap,
                        "amplitude_ratio",
                        errors,
                        "downsample.std_config.low_amplitude_cap",
                    )
                    _check_positive_int(
                        low_amplitude_cap,
                        "max_splits",
                        errors,
                        "downsample.std_config.low_amplitude_cap",
                    )
                    _check_positive_number(
                        low_amplitude_cap,
                        "reference_max_value",
                        errors,
                        "downsample.std_config.low_amplitude_cap",
                    )
                    if (
                        "apply_inside_focus_region" in low_amplitude_cap
                        and not isinstance(low_amplitude_cap["apply_inside_focus_region"], bool)
                    ):
                        errors.append(
                            "downsample.std_config.low_amplitude_cap.apply_inside_focus_region "
                            "must be true or false."
                        )
    if method == "data":
        data_config = _require_mapping(downsample, "data_config", errors, "downsample")
        if data_config is not None:
            _reject_legacy_downsample_fields(
                data_config,
                ("tolerance", "threshold", "quantity", "smooth"),
                errors,
                "downsample.data_config",
            )
            _require_keys(
                data_config,
                ("startingsize", "minimumsize", "min_valid_fraction", "split_metric_threshold"),
                errors,
                "downsample.data_config",
            )
            for key in ("startingsize", "minimumsize", "split_metric_threshold"):
                _check_positive_number(data_config, key, errors, "downsample.data_config")
            _check_fraction(data_config, "min_valid_fraction", errors, "downsample.data_config")
            split_metric = data_config.get("split_metric", "curvature")
            if split_metric not in SPLIT_METRIC_CHOICES:
                errors.append(
                    "downsample.data_config.split_metric must be one of "
                    f"{SPLIT_METRIC_CHOICES}; got {data_config.get('split_metric')!r}."
                )
            _check_positive_int(data_config, "decimorig", errors, "downsample.data_config")
            _check_positive_int(data_config, "itmax", errors, "downsample.data_config")
            _check_positive_number(data_config, "split_metric_smoothing", errors, "downsample.data_config")
    if method == "trirb":
        trirb_config = _require_mapping(downsample, "trirb_config", errors, "downsample")
        if trirb_config is not None:
            _reject_legacy_downsample_fields(
                trirb_config,
                ("tolerance",),
                errors,
                "downsample.trirb_config",
            )
            _require_keys(
                trirb_config,
                ("minimumsize", "min_valid_fraction", "max_samples", "change_threshold", "smooth_factor"),
                errors,
                "downsample.trirb_config",
            )
            for key in ("minimumsize", "change_threshold", "smooth_factor"):
                _check_positive_number(trirb_config, key, errors, "downsample.trirb_config")
            _check_fraction(trirb_config, "min_valid_fraction", errors, "downsample.trirb_config")
            _check_positive_int(trirb_config, "max_samples", errors, "downsample.trirb_config")
            _check_positive_int(trirb_config, "decimorig", errors, "downsample.trirb_config")
    if method == "from_rsp":
        from_rsp_config = _require_mapping(downsample, "from_rsp_config", errors, "downsample")
        if from_rsp_config is not None:
            _reject_legacy_downsample_fields(
                from_rsp_config,
                ("tolerance",),
                errors,
                "downsample.from_rsp_config",
            )
            _require_keys(
                from_rsp_config,
                ("rsp_file", "min_valid_fraction"),
                errors,
                "downsample.from_rsp_config",
            )
            geometry = str(from_rsp_config.get("geometry", "auto")).replace("-", "_").lower()
            if geometry not in RSP_GEOMETRY_CHOICES:
                errors.append(
                    "downsample.from_rsp_config.geometry must be one of "
                    f"{RSP_GEOMETRY_CHOICES}; got {from_rsp_config.get('geometry')!r}."
                )
            _check_fraction(
                from_rsp_config,
                "min_valid_fraction",
                errors,
                "downsample.from_rsp_config",
                allow_zero=True,
            )
            _check_positive_int(from_rsp_config, "decimorig", errors, "downsample.from_rsp_config")

    plot_decim = downsample.get("plot_decim")
    if plot_decim is not None:
        if not isinstance(plot_decim, dict):
            errors.append("downsample.plot_decim must be a mapping.")
        else:
            style = plot_decim.get("style", "cells")
            if style not in ("cells", "points"):
                errors.append("downsample.plot_decim.style must be 'cells' or 'points'.")
            coordrange = plot_decim.get("coordrange")
            if coordrange is not None:
                values = _check_number_sequence(coordrange, 4, errors, "downsample.plot_decim.coordrange")
                if values is not None and (values[0] >= values[1] or values[2] >= values[3]):
                    errors.append(
                        "downsample.plot_decim.coordrange must satisfy "
                        "minlon < maxlon and minlat < maxlat."
                    )
            _check_positive_number(plot_decim, "factor4plot", errors, "downsample.plot_decim")
            _check_positive_number(plot_decim, "edgewidth", errors, "downsample.plot_decim", allow_zero=True)
            _check_positive_number(plot_decim, "markersize", errors, "downsample.plot_decim")
            _check_positive_number(plot_decim, "alpha", errors, "downsample.plot_decim", allow_zero=True)
            if plot_decim.get("alpha") is not None:
                try:
                    alpha = float(plot_decim["alpha"])
                except (TypeError, ValueError):
                    alpha = None
                if alpha is not None and alpha > 1.0:
                    errors.append("downsample.plot_decim.alpha must be <= 1.")
            _check_positive_int(plot_decim, "dpi", errors, "downsample.plot_decim")
            _check_positive_number(plot_decim, "colorbar_size", errors, "downsample.plot_decim")
            _check_positive_number(plot_decim, "colorbar_thickness", errors, "downsample.plot_decim")
            _check_positive_number(plot_decim, "colorbar_pad", errors, "downsample.plot_decim", allow_zero=True)
            if plot_decim.get("figsize") is not None:
                figsize = _check_number_sequence(plot_decim["figsize"], 2, errors, "downsample.plot_decim.figsize")
                if figsize is not None and any(value <= 0.0 for value in figsize):
                    errors.append("downsample.plot_decim.figsize values must be positive.")
            if plot_decim.get("colorbar_orientation") not in (None, "horizontal", "vertical", "h", "v"):
                errors.append(
                    "downsample.plot_decim.colorbar_orientation must be "
                    "'horizontal' or 'vertical'."
                )
            if plot_decim.get("colorbar_mode") not in (None, "auto", "inside", "outside", "manual"):
                errors.append(
                    "downsample.plot_decim.colorbar_mode must be "
                    "'auto', 'inside', 'outside', or 'manual'."
                )

    if data_type == "optical" and not adapter_enabled:
        optical_config = _require_mapping(config, "optical_config", errors, "config")
        if optical_config is not None:
            try:
                validate_optical_config(optical_config)
            except ValueError as exc:
                errors.extend(str(exc).splitlines()[1:])

    if errors:
        raise ValueError("Invalid downsample config:\n- " + "\n- ".join(errors))
    return True


def get_sar_output_name(sar_config):
    out_name = sar_config["outName"]
    suffix = sar_config.get("output_suffix", "auto")
    if suffix is None or str(suffix).lower() in ("", "none", "false"):
        return out_name
    if str(suffix).lower() != "auto":
        return out_name + str(suffix)
    mode_key = _mode_key(sar_config.get("mode") or "")
    auto_suffix = SAR_OUTPUT_SUFFIX_BY_MODE.get(mode_key, "")
    if auto_suffix and str(out_name).endswith(auto_suffix):
        return out_name
    return out_name + auto_suffix
