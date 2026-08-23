"""Deterministic reference correction for SAR and optical observations."""

import warnings

import numpy as np
import yaml

from .region_utils import lonlat_region_mask, project_lonlat


CORRECTION_MODELS = ("offset", "plane")
COEFFICIENT_MODES = ("estimate", "fixed")
CORRECTION_REGION_KINDS = ("circle", "box", "polygon", "polygon_file")


def observation_correction_report_file(config, out_name):
    report_file = config.get("report_file", "auto")
    if report_file in (None, False):
        return None
    if str(report_file).lower() == "auto":
        return f"{out_name}_observation_correction.yml"
    return str(report_file)


def _region_mask(data, grid, region, base_dir=None, label="region"):
    return lonlat_region_mask(
        data,
        grid.longitude,
        grid.latitude,
        region,
        base_dir=base_dir,
        label=label,
    )


def _correction_fit_selection(data, grid, fit, model, base_dir=None):
    """Build the model-aware fit mask and its report diagnostics.

    Offset estimation needs an explicit zero-reference region.  Plane
    estimation uses every analysis-valid observation when ``regions`` is
    empty; explicit regions remain available when a restricted plane fit is
    scientifically preferable.  Exclusions are applied last in both modes.
    """

    if not isinstance(fit, dict):
        raise ValueError("observation_correction.fit must be a mapping.")
    coord_type = str(fit.get("coord_type", "lonlat")).replace("-", "_").lower()
    if coord_type != "lonlat":
        raise ValueError(
            "observation_correction.fit.coord_type currently supports only 'lonlat'."
        )
    regions = fit.get("regions", []) or []
    if not isinstance(regions, list):
        raise ValueError(
            "observation_correction.fit.regions must be a list."
        )
    model = str(model).replace("-", "_").lower()
    analysis_valid = np.asarray(grid.analysis_valid_mask, dtype=bool)
    if regions:
        fit_scope = "explicit_regions"
        selected = np.zeros(grid.shape, dtype=bool)
        for index, region in enumerate(regions):
            selected |= _region_mask(
                data,
                grid,
                region,
                base_dir=base_dir,
                label=f"observation_correction.fit.regions[{index}]",
            )
    elif model == "plane":
        fit_scope = "all_valid"
        selected = np.array(analysis_valid, copy=True)
    elif model == "offset":
        raise ValueError(
            "observation_correction.fit.regions must contain at least one "
            "zero-reference region for model='offset'."
        )
    else:
        raise ValueError(
            f"observation_correction.model must be one of {CORRECTION_MODELS}."
        )

    selected &= analysis_valid
    candidate_pixel_count = int(np.count_nonzero(selected))
    excludes = fit.get("exclude_regions", []) or []
    if not isinstance(excludes, list):
        raise ValueError(
            "observation_correction.fit.exclude_regions must be a list."
        )
    excluded = np.zeros(grid.shape, dtype=bool)
    for index, region in enumerate(excludes):
        excluded |= _region_mask(
            data,
            grid,
            region,
            base_dir=base_dir,
            label=f"observation_correction.fit.exclude_regions[{index}]",
        )
    excluded_pixel_count = int(np.count_nonzero(selected & excluded))
    selected &= ~excluded
    details = {
        "scope": fit_scope,
        "analysis_valid_pixel_count": int(np.count_nonzero(analysis_valid)),
        "candidate_pixel_count": candidate_pixel_count,
        "excluded_pixel_count": excluded_pixel_count,
        "pixel_count": int(np.count_nonzero(selected)),
    }
    return selected, details


def correction_fit_mask(data, grid, fit, base_dir=None, model="offset"):
    """Return the model-aware observation-correction fit mask."""

    selected, _details = _correction_fit_selection(
        data,
        grid,
        fit,
        model=model,
        base_dir=base_dir,
    )
    return selected


def _finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "mad": None,
            "std": None,
            "min": None,
            "max": None,
        }
    median = float(np.median(values))
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": median,
        "mad": float(np.median(np.abs(values - median))),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _huber_plane(x, y, values, max_iterations=25, tolerance=1.0e-10):
    x0 = float(np.mean(x))
    y0 = float(np.mean(y))
    design = np.column_stack(
        (
            np.ones(values.size, dtype=float),
            x - x0,
            y - y0,
        )
    )
    rank = int(np.linalg.matrix_rank(design))
    if rank < 3:
        raise ValueError(
            "observation_correction plane fit is spatially degenerate "
            f"(design rank {rank}, expected 3)."
        )
    scaled = np.column_stack(
        (
            np.ones(values.size, dtype=float),
            (x - x0) / max(float(np.std(x)), np.finfo(float).eps),
            (y - y0) / max(float(np.std(y)), np.finfo(float).eps),
        )
    )
    condition_number = float(np.linalg.cond(scaled))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        residual = values - design @ coefficients
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        robust_sigma = 1.4826 * mad
        if not np.isfinite(robust_sigma) or robust_sigma <= np.finfo(float).eps:
            break
        threshold = 1.345 * robust_sigma
        absolute = np.abs(residual - median)
        weights = np.ones(values.size, dtype=float)
        outside = absolute > threshold
        weights[outside] = threshold / absolute[outside]
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_values = values * np.sqrt(weights)
        updated, *_ = np.linalg.lstsq(
            weighted_design,
            weighted_values,
            rcond=None,
        )
        if np.linalg.norm(updated - coefficients) <= tolerance * (
            1.0 + np.linalg.norm(coefficients)
        ):
            coefficients = updated
            break
        coefficients = updated
    return {
        "offset": float(coefficients[0]),
        "east_gradient": float(coefficients[1]),
        "north_gradient": float(coefficients[2]),
        "origin_x_km": x0,
        "origin_y_km": y0,
        "rank": rank,
        "condition_number": condition_number,
        "iterations": int(iterations),
    }


def _estimate_coefficients(data, grid, values, component, model, fit_mask):
    values = np.asarray(values, dtype=float)
    valid = fit_mask & np.isfinite(values)
    count = int(np.count_nonzero(valid))
    if count == 0:
        raise ValueError(
            f"observation_correction fit selection contains no valid "
            f"{component} observations."
        )
    if count < 10:
        warnings.warn(
            f"observation_correction uses only {count} valid {component} "
            "fit pixels; inspect the correction report.",
            UserWarning,
            stacklevel=3,
        )

    reference_values = values[valid]
    if model == "offset":
        coefficients = {
            "offset": float(np.median(reference_values)),
            "east_gradient": 0.0,
            "north_gradient": 0.0,
            "origin_x_km": 0.0,
            "origin_y_km": 0.0,
            "rank": 1,
            "condition_number": 1.0,
            "iterations": 0,
        }
    else:
        x, y = project_lonlat(
            data,
            grid.longitude[valid],
            grid.latitude[valid],
            context="observation_correction",
        )
        coefficients = _huber_plane(x, y, reference_values)
    return coefficients, _finite_stats(reference_values)


def _fixed_coefficients(data, config, component, model):
    fixed = config.get("fixed_coefficients")
    if not isinstance(fixed, dict):
        raise ValueError(
            "observation_correction.fixed_coefficients must be a mapping."
        )
    component_values = fixed.get(component)
    if not isinstance(component_values, dict):
        raise ValueError(
            f"observation_correction.fixed_coefficients.{component} is required."
        )
    coefficients = {
        "offset": float(component_values.get("offset", 0.0)),
        "east_gradient": float(component_values.get("east_gradient", 0.0)),
        "north_gradient": float(component_values.get("north_gradient", 0.0)),
        "rank": None,
        "condition_number": None,
        "iterations": 0,
    }
    if model == "offset":
        coefficients["east_gradient"] = 0.0
        coefficients["north_gradient"] = 0.0
        coefficients["origin_x_km"] = 0.0
        coefficients["origin_y_km"] = 0.0
        return coefficients

    origin = fixed.get("origin")
    if not isinstance(origin, (list, tuple)) or len(origin) != 2:
        raise ValueError(
            "observation_correction.fixed_coefficients.origin must be "
            "[lon, lat] for model='plane'."
        )
    origin_x, origin_y = project_lonlat(
        data,
        np.asarray([float(origin[0])]),
        np.asarray([float(origin[1])]),
        context="observation_correction",
    )
    coefficients["origin_x_km"] = float(origin_x[0])
    coefficients["origin_y_km"] = float(origin_y[0])
    coefficients["origin_lonlat"] = [float(origin[0]), float(origin[1])]
    return coefficients


def _surface(data, grid, coefficients):
    if (
        coefficients["east_gradient"] == 0.0
        and coefficients["north_gradient"] == 0.0
    ):
        return np.full(grid.shape, coefficients["offset"], dtype=float)
    x, y = project_lonlat(
        data,
        grid.longitude,
        grid.latitude,
        context="observation_correction",
    )
    return (
        coefficients["offset"]
        + coefficients["east_gradient"] * (x - coefficients["origin_x_km"])
        + coefficients["north_gradient"] * (y - coefficients["origin_y_km"])
    )


def _apply_to_csi_data(data, grid, component, corrected_grid):
    raw_indices = grid.raw_flat_indices()
    corrected_values = np.asarray(corrected_grid).reshape(-1)[raw_indices]
    attribute = "vel" if component == "observation" else component
    current = np.asarray(getattr(data, attribute))
    if corrected_values.size != current.size:
        raise ValueError(
            f"observation_correction cannot align raw {component} grid with "
            f"current CSI values: {corrected_values.size} versus {current.size}."
        )
    original_attribute = (
        "observation_before_correction"
        if component == "observation"
        else f"{component}_before_correction"
    )
    setattr(data, original_attribute, np.array(current, copy=True))
    setattr(data, attribute, np.array(corrected_values, dtype=float, copy=True))


def apply_observation_correction(
    data,
    grid,
    config,
    out_name="observation",
    base_dir=None,
    write_report=True,
):
    """Estimate/apply one deterministic correction model to each component."""

    config = config or {}
    enabled = bool(config.get("enabled", False))
    model = str(config.get("model", "offset")).replace("-", "_").lower()
    coefficient_mode = str(
        config.get("coefficient_mode", "estimate")
    ).replace("-", "_").lower()
    report = {
        "enabled": enabled,
        "model": model,
        "coefficient_mode": coefficient_mode,
        "formula": "corrected = observation - correction_surface",
        "components": {},
    }
    if not enabled:
        return report
    if model not in CORRECTION_MODELS:
        raise ValueError(
            f"observation_correction.model must be one of {CORRECTION_MODELS}."
        )
    if coefficient_mode not in COEFFICIENT_MODES:
        raise ValueError(
            "observation_correction.coefficient_mode must be one of "
            f"{COEFFICIENT_MODES}."
        )

    fit_mask = None
    if coefficient_mode == "estimate":
        fit_mask, fit_details = _correction_fit_selection(
            data,
            grid,
            config.get("fit"),
            model=model,
            base_dir=base_dir,
        )
        report["fit"] = fit_details
        # Kept for readers of reports written before fit diagnostics were
        # grouped explicitly under ``fit``.
        report["reference_pixel_count"] = fit_details["pixel_count"]

    for component in grid.components:
        before = np.asarray(grid.display_component(component), dtype=float)
        if coefficient_mode == "estimate":
            coefficients, reference_stats = _estimate_coefficients(
                data,
                grid,
                before,
                component,
                model,
                fit_mask,
            )
        else:
            coefficients = _fixed_coefficients(
                data,
                config,
                component,
                model,
            )
            reference_stats = None
        surface = _surface(data, grid, coefficients)
        corrected = before - surface
        corrected[~np.isfinite(before)] = np.nan
        grid.set_correction(component, surface, corrected)
        _apply_to_csi_data(data, grid, component, corrected)

        component_report = {
            "coefficients": coefficients,
            "before": _finite_stats(before[grid.analysis_valid_mask]),
            "after": _finite_stats(corrected[grid.analysis_valid_mask]),
        }
        if reference_stats is not None:
            component_report["reference"] = reference_stats
        report["components"][component] = component_report

    report_file = observation_correction_report_file(config, out_name)
    if write_report and config.get("report", True) and report_file:
        with open(report_file, "w", encoding="utf-8") as stream:
            yaml.safe_dump(report, stream, allow_unicode=True, sort_keys=False)
        report["report_file"] = report_file
    return report


def format_observation_correction_report(report):
    if not report.get("enabled"):
        return ""
    lines = [
        "Observation correction:",
        f"  model       : {report['model']}",
        f"  coefficients: {report['coefficient_mode']}",
        "  formula     : corrected = observation - correction_surface",
    ]
    fit_details = report.get("fit")
    if fit_details:
        lines.extend(
            [
                f"  fit scope   : {fit_details['scope']}",
                f"  fit pixels  : {fit_details['pixel_count']}",
                f"  excluded    : {fit_details['excluded_pixel_count']}",
            ]
        )
    for component, details in report.get("components", {}).items():
        coefficients = details["coefficients"]
        lines.extend(
            [
                f"  {component}:",
                f"    offset         : {coefficients['offset']:.8g}",
                f"    east gradient  : {coefficients['east_gradient']:.8g} / km",
                f"    north gradient : {coefficients['north_gradient']:.8g} / km",
                f"    median before  : {details['before']['median']}",
                f"    median after   : {details['after']['median']}",
            ]
        )
    if report.get("report_file"):
        lines.append(f"  report file : {report['report_file']}")
    return "\n".join(lines)
