"""Fault summary helpers for CSI/eqtools fault objects.

The functions in this module follow the same pattern as other csiExtend
diagnostics: ``summarize_*`` returns structured dictionaries, while
``print_*`` formats the same information for interactive checks.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np


def summarize_fault(
    fault: Any,
    *,
    mu: float = 3.0e10,
    slip_factor: float = 1.0,
    slip_unit_label: str = "m",
    moment_slip_factor: float | None = None,
    moment_kind: str = "moment",
    moment_unit_label: str | None = None,
    equivalent_duration_years: float | None = None,
    include_slip: bool = True,
    include_moment: bool = True,
    moment_skip_reason: str | None = None,
) -> dict[str, Any]:
    """Return a compact geometry and slip summary for one fault object.

    Parameters
    ----------
    fault
        A CSI-compatible fault object, for example ``TriangularPatches`` or
        ``RectangularPatches``.
    mu
        Shear modulus in Pa used when a seismic moment can be computed.
    slip_factor
        Multiplicative factor applied to slip before reporting slip statistics.
    slip_unit_label
        Label for the slip statistics after applying ``slip_factor``.
    moment_slip_factor
        Multiplicative factor applied to slip for moment or moment-rate
        calculations. When omitted, ``slip_factor`` is reused.
    moment_kind
        ``"moment"`` for cumulative displacement or ``"moment_rate"`` for
        slip-rate models. Rate summaries report ``N*m/yr`` and an equivalent
        one-year magnitude label rather than event magnitude.
    include_slip, include_moment
        Control whether slip statistics and seismic moment are collected.
    moment_skip_reason
        Optional warning recorded when moment calculation is intentionally
        skipped by a custom workflow.

    Returns
    -------
    dict
        A JSON-serializable summary dictionary. Missing optional quantities are
        represented by ``None`` and explained in the ``warnings`` list.
    """
    warnings: list[str] = []
    patch_type = _infer_patch_type(fault)
    area = _patch_areas(fault, warnings)

    summary = {
        "name": str(getattr(fault, "name", fault.__class__.__name__)),
        "class_name": fault.__class__.__name__,
        "patch_type": patch_type,
        "basic": _basic_info(fault, patch_type),
        "trace": _trace_info(fault, warnings),
        "bounds": _geometry_bounds(fault, warnings),
        "mesh": _mesh_info(fault, patch_type, area, warnings),
        "orientation": _orientation_info(fault, warnings),
        "warnings": warnings,
    }

    if include_slip:
        summary["slip"] = _slip_info(
            fault,
            slip_factor=slip_factor,
            slip_unit_label=slip_unit_label,
            warnings=warnings,
        )
    else:
        summary["slip"] = None

    if include_moment:
        summary["moment"] = _moment_info(
            fault,
            area,
            mu=mu,
            slip_factor=slip_factor if moment_slip_factor is None else moment_slip_factor,
            moment_kind=moment_kind,
            moment_unit_label=moment_unit_label,
            equivalent_duration_years=equivalent_duration_years,
            warnings=warnings,
        )
    else:
        summary["moment"] = None
        if moment_skip_reason:
            warnings.append(moment_skip_reason)

    return summary


def get_fault_summary(fault: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`summarize_fault` for API consistency."""
    return summarize_fault(fault, **kwargs)


def summarize_faults(
    faults: Iterable[Any],
    *,
    fault_groups: Iterable[Iterable[str] | str] | None = None,
    mu: float = 3.0e10,
    slip_factor: float = 1.0,
    slip_unit_label: str = "m",
    moment_slip_factor: float | None = None,
    moment_kind: str = "moment",
    moment_unit_label: str | None = None,
    equivalent_duration_years: float | None = None,
    include_slip: bool = True,
    include_moment: bool = True,
    moment_skip_reason: str | None = None,
) -> dict[str, Any]:
    """Return summaries for a collection of fault objects."""
    target_faults = list(faults)
    fault_summaries = [
        summarize_fault(
            fault,
            mu=mu,
            slip_factor=slip_factor,
            slip_unit_label=slip_unit_label,
            moment_slip_factor=moment_slip_factor,
            moment_kind=moment_kind,
            moment_unit_label=moment_unit_label,
            equivalent_duration_years=equivalent_duration_years,
            include_slip=include_slip,
            include_moment=include_moment,
            moment_skip_reason=moment_skip_reason,
        )
        for fault in target_faults
    ]

    group_summaries, total_moment = _group_moment_summaries(
        fault_summaries,
        fault_groups=fault_groups,
    )

    return {
        "faults": fault_summaries,
        "groups": group_summaries,
        "total": {
            "fault_count": len(fault_summaries),
            "patch_count": _sum_present(
                item.get("mesh", {}).get("patch_count") for item in fault_summaries
            ),
            "area_km2": _sum_present(
                item.get("mesh", {}).get("area", {}).get("total_km2")
                for item in fault_summaries
            ),
            **total_moment,
        },
    }


def get_faults_summary(faults: Iterable[Any], **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`summarize_faults` for API consistency."""
    return summarize_faults(faults, **kwargs)


def print_fault_summary(
    fault: Any,
    *,
    file: Any = None,
    tablefmt: str = "grid",
    **kwargs: Any,
) -> dict[str, Any]:
    """Print and return a summary for one fault object."""
    summary = summarize_fault(fault, **kwargs)
    print(_format_single_fault_summary(summary, tablefmt=tablefmt), file=file)
    return summary


def show_fault_summary(fault: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`print_fault_summary` for interactive use."""
    return print_fault_summary(fault, **kwargs)


def print_faults_summary(
    faults: Iterable[Any],
    *,
    fault_groups: Iterable[Iterable[str] | str] | None = None,
    file: Any = None,
    tablefmt: str = "grid",
    **kwargs: Any,
) -> dict[str, Any]:
    """Print and return summaries for multiple faults."""
    summary = summarize_faults(faults, fault_groups=fault_groups, **kwargs)
    print(_format_faults_summary(summary, tablefmt=tablefmt), file=file)
    return summary


def show_faults_summary(faults: Iterable[Any], **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`print_faults_summary` for interactive use."""
    return print_faults_summary(faults, **kwargs)


def _basic_info(fault: Any, patch_type: str | None) -> dict[str, Any]:
    return {
        "patch_type": patch_type,
        "has_trace": _has_xy_pair(fault, "xf", "yf"),
        "has_discretized_trace": _has_xy_pair(fault, "xi", "yi"),
        "has_slip": getattr(fault, "slip", None) is not None,
        "lon0": _to_float_or_none(getattr(fault, "lon0", None)),
        "lat0": _to_float_or_none(getattr(fault, "lat0", None)),
        "utmzone": getattr(fault, "utmzone", None),
    }


def _trace_info(fault: Any, warnings: list[str]) -> dict[str, Any]:
    trace = {
        "point_count": _point_count(fault, "xf", "yf"),
        "length_km": _trace_length(fault, discretized=False, warnings=warnings),
        "discretized_point_count": _point_count(fault, "xi", "yi"),
        "discretized_length_km": _trace_length(
            fault,
            discretized=True,
            warnings=warnings,
        ),
    }
    return trace


def _geometry_bounds(fault: Any, warnings: list[str]) -> dict[str, Any]:
    vertices = _collect_local_vertices(fault)
    bounds = {}
    if vertices is not None and vertices.size:
        labels = ("x_km", "y_km", "depth_km")
        for index, label in enumerate(labels):
            if vertices.shape[1] > index:
                bounds[label] = _range_stat(vertices[:, index])

    lonlat = _collect_lonlat_vertices(fault)
    if lonlat is not None and lonlat.size:
        if lonlat.shape[1] > 0:
            bounds["lon"] = _range_stat(lonlat[:, 0])
        if lonlat.shape[1] > 1:
            bounds["lat"] = _range_stat(lonlat[:, 1])

    if not bounds:
        warnings.append("No vertex or patch coordinates were available for geometry bounds.")
    return bounds


def _mesh_info(
    fault: Any,
    patch_type: str | None,
    area: np.ndarray | None,
    warnings: list[str],
) -> dict[str, Any]:
    mesh = {
        "patch_count": _patch_count(fault),
        "vertex_count": _vertex_count(fault),
        "face_count": _face_count(fault),
        "area": _area_stats(area),
    }
    if patch_type == "rectangular":
        mesh.update(_rectangular_patch_size_stats(fault, warnings))
    return mesh


def _orientation_info(fault: Any, warnings: list[str]) -> dict[str, Any]:
    strike = _angle_array_from_method(fault, "getStrikes", "strike", warnings)
    dip = _angle_array_from_method(fault, "getDips", "dip", warnings)
    return {
        "strike_deg": _angle_stats(strike),
        "dip_deg": _angle_stats(dip),
    }


def _slip_info(
    fault: Any,
    *,
    slip_factor: float,
    slip_unit_label: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    slip = getattr(fault, "slip", None)
    if slip is None:
        return None
    try:
        slip_array = np.asarray(slip, dtype=float)
    except (TypeError, ValueError):
        warnings.append("Slip array could not be converted to float.")
        return None

    if slip_array.ndim == 1:
        slip_array = slip_array.reshape((-1, 1))
    if slip_array.size == 0:
        return None

    slip_array = slip_array * float(slip_factor)
    component_names = ("strike_slip", "dip_slip", "tensile", "coupling")
    components = {}
    for index in range(slip_array.shape[1]):
        name = component_names[index] if index < len(component_names) else f"component_{index}"
        components[name] = _array_stat(slip_array[:, index])

    if slip_array.shape[1] >= 2:
        total = np.sqrt(slip_array[:, 0] ** 2 + slip_array[:, 1] ** 2)
    else:
        total = np.abs(slip_array[:, 0])

    return {
        "unit": str(slip_unit_label),
        "component_count": int(slip_array.shape[1]),
        "patch_count": int(slip_array.shape[0]),
        "components": components,
        "total": _array_stat(total),
    }


def _moment_info(
    fault: Any,
    area: np.ndarray | None,
    *,
    mu: float,
    slip_factor: float,
    moment_kind: str,
    moment_unit_label: str | None,
    equivalent_duration_years: float | None,
    warnings: list[str],
) -> dict[str, Any] | None:
    if area is None or area.size == 0:
        return None
    slip = _total_slip_array(fault, slip_factor=slip_factor, warnings=warnings)
    if slip is None or slip.size == 0:
        return None

    count = min(area.size, slip.size)
    if count == 0:
        return None
    if area.size != slip.size:
        warnings.append(
            f"Area count ({area.size}) and slip count ({slip.size}) differ; "
            f"moment uses the first {count} entries."
        )

    area_m2 = np.asarray(area[:count], dtype=float) * 1.0e6
    slip_m = np.asarray(slip[:count], dtype=float)
    valid = np.isfinite(area_m2) & np.isfinite(slip_m)
    if not valid.any():
        return None

    moment = float(np.sum(float(mu) * area_m2[valid] * slip_m[valid]))
    kind = str(moment_kind or "moment").lower()
    if kind == "moment_rate":
        duration = 1.0 if equivalent_duration_years is None else float(equivalent_duration_years)
        return {
            "kind": "moment_rate",
            "unit": moment_unit_label or "N*m/yr",
            "moment_N_m": None,
            "magnitude": None,
            "moment_rate_N_m_per_yr": moment,
            "mw_equivalent_1yr": _moment_magnitude(moment * duration),
            "equivalent_duration_years": duration,
            "mu_Pa": float(mu),
            "slip_factor": float(slip_factor),
            "patch_count": int(valid.sum()),
        }

    if kind != "moment":
        warnings.append(f"Unknown moment kind '{moment_kind}'; treating it as cumulative moment.")
    return {
        "kind": "moment",
        "unit": moment_unit_label or "N*m",
        "moment_N_m": moment,
        "magnitude": _moment_magnitude(moment),
        "mu_Pa": float(mu),
        "slip_factor": float(slip_factor),
        "patch_count": int(valid.sum()),
    }


def _infer_patch_type(fault: Any) -> str | None:
    raw = getattr(fault, "patchType", None)
    if raw is None:
        raw = fault.__class__.__name__
    text = str(raw).lower()
    if "tri" in text:
        return "triangular"
    if "rect" in text or "quad" in text:
        return "rectangular"
    return str(raw) if raw is not None else None


def _has_xy_pair(fault: Any, x_name: str, y_name: str) -> bool:
    return getattr(fault, x_name, None) is not None and getattr(fault, y_name, None) is not None


def _point_count(fault: Any, x_name: str, y_name: str) -> int:
    if not _has_xy_pair(fault, x_name, y_name):
        return 0
    try:
        return int(min(np.asarray(getattr(fault, x_name)).size, np.asarray(getattr(fault, y_name)).size))
    except Exception:
        return 0


def _trace_length(fault: Any, *, discretized: bool, warnings: list[str]) -> float | None:
    x_name, y_name = ("xi", "yi") if discretized else ("xf", "yf")
    if not _has_xy_pair(fault, x_name, y_name):
        return None

    if hasattr(fault, "cumdistance"):
        try:
            distances = np.asarray(fault.cumdistance(discretized=discretized), dtype=float)
            finite = distances[np.isfinite(distances)]
            if finite.size:
                return float(finite[-1])
        except Exception as exc:
            fallback = _polyline_length(fault, x_name, y_name)
            if fallback is not None:
                return fallback
            label = "discretized trace" if discretized else "trace"
            warnings.append(f"Could not compute {label} length with cumdistance(): {exc}")

    return _polyline_length(fault, x_name, y_name)


def _polyline_length(fault: Any, x_name: str, y_name: str) -> float | None:
    if not _has_xy_pair(fault, x_name, y_name):
        return None
    try:
        x = np.asarray(getattr(fault, x_name), dtype=float).reshape(-1)
        y = np.asarray(getattr(fault, y_name), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    count = min(x.size, y.size)
    if count < 2:
        return 0.0 if count == 1 else None
    dx = np.diff(x[:count])
    dy = np.diff(y[:count])
    return float(np.nansum(np.sqrt(dx * dx + dy * dy)))


def _patch_count(fault: Any) -> int:
    for name in ("numpatch", "N_slip"):
        value = getattr(fault, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    patch = getattr(fault, "patch", None)
    if patch is not None:
        try:
            return int(len(patch))
        except TypeError:
            pass
    faces = getattr(fault, "Faces", None)
    if faces is not None:
        try:
            return int(np.asarray(faces).shape[0])
        except Exception:
            pass
    return 0


def _vertex_count(fault: Any) -> int | None:
    vertices = getattr(fault, "Vertices", None)
    if vertices is None:
        return None
    try:
        return int(np.asarray(vertices).shape[0])
    except Exception:
        return None


def _face_count(fault: Any) -> int | None:
    faces = getattr(fault, "Faces", None)
    if faces is None:
        return None
    try:
        return int(np.asarray(faces).shape[0])
    except Exception:
        return None


def _patch_areas(fault: Any, warnings: list[str]) -> np.ndarray | None:
    area = getattr(fault, "area", None)
    if area is None and hasattr(fault, "compute_patch_areas"):
        try:
            area = fault.compute_patch_areas()
        except Exception as exc:
            warnings.append(f"Could not compute patch areas: {exc}")
            return None
    if area is None:
        return None
    try:
        area_array = np.asarray(area, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        warnings.append("Patch areas could not be converted to float.")
        return None
    finite = area_array[np.isfinite(area_array)]
    return finite if finite.size else None


def _area_stats(area: np.ndarray | None) -> dict[str, Any] | None:
    if area is None or area.size == 0:
        return None
    stat = _array_stat(area)
    return {
        "total_km2": float(np.nansum(area)),
        "mean_km2": stat["mean"],
        "median_km2": stat["median"],
        "min_km2": stat["min"],
        "max_km2": stat["max"],
        "count": stat["count"],
    }


def _rectangular_patch_size_stats(fault: Any, warnings: list[str]) -> dict[str, Any]:
    if not hasattr(fault, "getpatchgeometry"):
        return {}
    patch_count = _patch_count(fault)
    if patch_count <= 0:
        return {}

    lengths = []
    widths = []
    for index in range(patch_count):
        try:
            geometry = fault.getpatchgeometry(index)
        except Exception as exc:
            warnings.append(f"Could not read rectangular patch geometry for patch {index}: {exc}")
            return {}
        if len(geometry) >= 6:
            widths.append(geometry[3])
            lengths.append(geometry[4])

    result = {}
    length_stats = _array_stat_or_none(lengths)
    width_stats = _array_stat_or_none(widths)
    if length_stats is not None:
        result["patch_length_km"] = length_stats
    if width_stats is not None:
        result["patch_width_km"] = width_stats
    return result


def _collect_local_vertices(fault: Any) -> np.ndarray | None:
    vertices = _as_2d_float_array(getattr(fault, "Vertices", None))
    if vertices is not None and vertices.shape[1] >= 3:
        return vertices[:, :3]

    patch = getattr(fault, "patch", None)
    if patch is None:
        return None

    arrays = []
    try:
        iterator = list(patch)
    except TypeError:
        return None
    for item in iterator:
        array = _as_2d_float_array(item)
        if array is not None and array.shape[1] >= 3:
            arrays.append(array[:, :3])
    if not arrays:
        return None
    return np.vstack(arrays)


def _collect_lonlat_vertices(fault: Any) -> np.ndarray | None:
    for name in ("Vertices_ll", "vertices_ll", "ll_vertices"):
        array = _as_2d_float_array(getattr(fault, name, None))
        if array is not None and array.shape[1] >= 2:
            return array[:, :2]

    for name in ("llpatch", "patchll"):
        patches = getattr(fault, name, None)
        if patches is None:
            continue
        arrays = []
        try:
            iterator = list(patches)
        except TypeError:
            continue
        for item in iterator:
            array = _as_2d_float_array(item)
            if array is not None and array.shape[1] >= 2:
                arrays.append(array[:, :2])
        if arrays:
            return np.vstack(arrays)
    return None


def _angle_array_from_method(
    fault: Any,
    method_name: str,
    label: str,
    warnings: list[str],
) -> np.ndarray | None:
    method = getattr(fault, method_name, None)
    if method is None:
        return None
    try:
        values = np.asarray(method(), dtype=float).reshape(-1)
    except Exception as exc:
        warnings.append(f"Could not collect {label} angles from {method_name}(): {exc}")
        return None
    values = values[np.isfinite(values)]
    if not values.size:
        return None

    if np.nanmax(np.abs(values)) <= 2.0 * np.pi + 1.0e-6:
        values = np.degrees(values)
    return values


def _angle_stats(values: np.ndarray | None) -> dict[str, Any] | None:
    if values is None or values.size == 0:
        return None
    return _array_stat(values)


def _total_slip_array(
    fault: Any,
    *,
    slip_factor: float,
    warnings: list[str],
) -> np.ndarray | None:
    slip = getattr(fault, "slip", None)
    if slip is None:
        return None
    try:
        slip_array = np.asarray(slip, dtype=float)
    except (TypeError, ValueError):
        warnings.append("Slip array could not be converted to float for moment calculation.")
        return None
    if slip_array.ndim == 1:
        slip_array = slip_array.reshape((-1, 1))
    if slip_array.size == 0:
        return None

    slip_array = slip_array * float(slip_factor)
    if slip_array.shape[1] >= 2:
        total = np.sqrt(slip_array[:, 0] ** 2 + slip_array[:, 1] ** 2)
    else:
        total = np.abs(slip_array[:, 0])
    return total[np.isfinite(total)]


def _group_moment_summaries(
    fault_summaries: list[dict[str, Any]],
    *,
    fault_groups: Iterable[Iterable[str] | str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kind = _summary_moment_kind_from_faults(fault_summaries)
    value_key = _moment_value_key(kind)
    moment_by_name = {
        item["name"]: item.get("moment", {}).get(value_key)
        for item in fault_summaries
        if item.get("moment")
    }

    if fault_groups is None:
        return [], _moment_total_summary(moment_by_name.values(), kind)

    groups = _normalize_fault_groups(fault_summaries, fault_groups)
    group_summaries = []
    grouped_values = []
    for index, group in enumerate(groups):
        names = [name for name in group if name in moment_by_name and moment_by_name[name] is not None]
        value = float(sum(moment_by_name[name] for name in names))
        if value > 0:
            grouped_values.append(value)
        group_summaries.append(_moment_group_summary(index, group, names, value, kind))
    return group_summaries, _moment_total_summary(grouped_values, kind)


def _summary_moment_kind_from_faults(fault_summaries: list[dict[str, Any]]) -> str:
    kinds = {
        str(item.get("moment", {}).get("kind") or "moment")
        for item in fault_summaries
        if item.get("moment")
    }
    if "moment_rate" in kinds:
        return "moment_rate"
    return "moment"


def _moment_value_key(kind: str) -> str:
    return "moment_rate_N_m_per_yr" if kind == "moment_rate" else "moment_N_m"


def _moment_magnitude_key(kind: str) -> str:
    return "mw_equivalent_1yr" if kind == "moment_rate" else "magnitude"


def _moment_group_summary(
    index: int,
    group: Iterable[str],
    names: list[str],
    value: float,
    kind: str,
) -> dict[str, Any]:
    group_names = list(group)
    label = f"Event {index + 1}: {', '.join(names) if names else ', '.join(group_names)}"
    result: dict[str, Any] = {
        "name": label,
        "faults": group_names,
        "moment_kind": kind,
    }
    if kind == "moment_rate":
        result.update(
            {
                "moment_N_m": None,
                "magnitude": None,
                "moment_rate_N_m_per_yr": value if value > 0 else None,
                "mw_equivalent_1yr": _moment_magnitude(value),
            }
        )
    else:
        result.update(
            {
                "moment_N_m": value if value > 0 else None,
                "magnitude": _moment_magnitude(value),
            }
        )
    return result


def _moment_total_summary(values: Iterable[Any], kind: str) -> dict[str, Any]:
    total = 0.0
    for value in values:
        if value is not None:
            total += float(value)

    if kind == "moment_rate":
        return {
            "moment_kind": "moment_rate",
            "moment_unit": "N*m/yr",
            "moment_N_m": None,
            "magnitude": None,
            "moment_rate_N_m_per_yr": total if total > 0 else None,
            "mw_equivalent_1yr": _moment_magnitude(total),
            "equivalent_duration_years": 1.0,
        }

    return {
        "moment_kind": "moment",
        "moment_unit": "N*m",
        "moment_N_m": total if total > 0 else None,
        "magnitude": _moment_magnitude(total),
    }


def _normalize_fault_groups(
    fault_summaries: list[dict[str, Any]],
    fault_groups: Iterable[Iterable[str] | str] | None,
) -> list[list[str]]:
    if fault_groups is None:
        return [[item["name"]] for item in fault_summaries]
    normalized = []
    for group in fault_groups:
        if isinstance(group, str):
            normalized.append([group])
        else:
            normalized.append([str(name) for name in group])
    return normalized


def _moment_magnitude(moment: float | None) -> float | None:
    if moment is None or moment <= 0 or not np.isfinite(moment):
        return None
    return float(2.0 / 3.0 * (math.log10(moment) - 9.1))


def _array_stat_or_none(values: Iterable[Any]) -> dict[str, Any] | None:
    try:
        array = np.asarray(list(values), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    finite = array[np.isfinite(array)]
    if not finite.size:
        return None
    return _array_stat(finite)


def _array_stat(values: Iterable[Any]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.nanmean(finite)),
        "median": float(np.nanmedian(finite)),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "std": float(np.nanstd(finite)),
    }


def _range_stat(values: Iterable[Any]) -> dict[str, Any]:
    stat = _array_stat(values)
    return {
        "min": stat["min"],
        "max": stat["max"],
        "count": stat["count"],
    }


def _as_2d_float_array(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim == 1:
        array = array.reshape((1, -1))
    if array.ndim != 2 or array.size == 0:
        return None
    return array


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _sum_present(values: Iterable[Any]) -> float | int | None:
    total = 0.0
    seen = False
    for value in values:
        if value is None:
            continue
        total += float(value)
        seen = True
    if not seen:
        return None
    return int(total) if total.is_integer() else total


def _moment_total_for_single_fault(summary: dict[str, Any]) -> dict[str, Any]:
    moment = summary.get("moment") or {}
    kind = str(moment.get("kind") or "moment")
    if kind == "moment_rate":
        return {
            "moment_kind": "moment_rate",
            "moment_unit": moment.get("unit") or "N*m/yr",
            "moment_N_m": None,
            "magnitude": None,
            "moment_rate_N_m_per_yr": moment.get("moment_rate_N_m_per_yr"),
            "mw_equivalent_1yr": moment.get("mw_equivalent_1yr"),
            "equivalent_duration_years": moment.get("equivalent_duration_years", 1.0),
        }
    return {
        "moment_kind": "moment",
        "moment_unit": moment.get("unit") or "N*m",
        "moment_N_m": moment.get("moment_N_m"),
        "magnitude": moment.get("magnitude"),
    }


def _format_single_fault_summary(summary: dict[str, Any], *, tablefmt: str) -> str:
    moment_total = _moment_total_for_single_fault(summary)
    return _format_faults_summary(
        {
            "faults": [summary],
            "groups": [],
            "total": {
                "fault_count": 1,
                "patch_count": summary.get("mesh", {}).get("patch_count"),
                "area_km2": summary.get("mesh", {}).get("area", {}).get("total_km2")
                if summary.get("mesh", {}).get("area")
                else None,
                **moment_total,
            },
        },
        tablefmt=tablefmt,
        title=f"Fault Summary: {summary['name']}",
    )


def _format_faults_summary(
    summary: dict[str, Any],
    *,
    tablefmt: str,
    title: str = "Fault Summary",
) -> str:
    lines = [title, "=" * len(title)]
    fault_summaries = summary.get("faults", [])

    geometry_rows = []
    for item in fault_summaries:
        mesh = item.get("mesh") or {}
        area = mesh.get("area") or {}
        trace = item.get("trace") or {}
        bounds = item.get("bounds") or {}
        depth = bounds.get("depth_km") or {}
        orientation = item.get("orientation") or {}
        strike = orientation.get("strike_deg") or {}
        dip = orientation.get("dip_deg") or {}
        geometry_rows.append(
            [
                item.get("name"),
                item.get("patch_type") or "N/A",
                _format_int(mesh.get("patch_count")),
                _format_number(trace.get("length_km"), suffix=" km"),
                _format_number(area.get("total_km2"), suffix=" km^2"),
                _format_range(depth, suffix=" km"),
                _format_number(strike.get("mean"), suffix=" deg"),
                _format_number(dip.get("mean"), suffix=" deg"),
            ]
        )
    lines.append("")
    lines.append("Geometry")
    lines.append(
        _tabulate(
            geometry_rows,
            [
                "Fault",
                "Patch Type",
                "Patches",
                "Trace Length",
                "Area",
                "Depth Range",
                "Mean Strike",
                "Mean Dip",
            ],
            tablefmt=tablefmt,
        )
    )

    slip_rows = _slip_rows(fault_summaries)
    if slip_rows:
        slip_unit = _summary_slip_unit_label(fault_summaries)
        lines.append("")
        lines.append("Slip")
        lines.append(
            _tabulate(
                slip_rows,
                [
                    "Fault",
                    "Component",
                    f"Mean ({slip_unit})",
                    f"Max ({slip_unit})",
                    f"Min ({slip_unit})",
                ],
                tablefmt=tablefmt,
            )
        )

    moment_rows = _moment_rows(summary)
    if moment_rows:
        lines.append("")
        lines.append(_moment_table_title(summary))
        lines.append(
            _tabulate(
                moment_rows,
                _moment_table_headers(summary, fault_summaries),
                tablefmt=tablefmt,
            )
        )

    warnings = []
    for item in fault_summaries:
        for warning in item.get("warnings", []):
            warnings.append(f"{item.get('name')}: {warning}")
    if warnings:
        lines.append("")
        lines.append("Warnings")
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines)


def _slip_rows(fault_summaries: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    preferred = ("strike_slip", "dip_slip", "tensile", "coupling")
    for item in fault_summaries:
        slip = item.get("slip")
        if not slip:
            continue
        components = slip.get("components", {})
        for name in preferred:
            stat = components.get(name)
            if stat:
                rows.append(
                    [
                        item.get("name"),
                        name,
                        _format_number(stat.get("mean")),
                        _format_number(stat.get("max")),
                        _format_number(stat.get("min")),
                    ]
                )
        total = slip.get("total")
        if total:
            rows.append(
                [
                    item.get("name"),
                    "total",
                    _format_number(total.get("mean")),
                    _format_number(total.get("max")),
                    _format_number(total.get("min")),
                ]
            )
    return rows


def _summary_slip_unit_label(fault_summaries: list[dict[str, Any]]) -> str:
    units = {
        str(item.get("slip", {}).get("unit"))
        for item in fault_summaries
        if item.get("slip") and item.get("slip", {}).get("unit")
    }
    if not units:
        return "m"
    if len(units) == 1:
        return next(iter(units))
    return "mixed"


def _summary_moment_kind(summary: dict[str, Any]) -> str:
    total = summary.get("total") or {}
    if total.get("moment_kind"):
        return str(total["moment_kind"])
    return _summary_moment_kind_from_faults(summary.get("faults", []))


def _moment_table_title(summary: dict[str, Any]) -> str:
    return "Moment Rate" if _summary_moment_kind(summary) == "moment_rate" else "Moment"


def _moment_table_headers(
    summary: dict[str, Any],
    fault_summaries: list[dict[str, Any]],
) -> list[str]:
    if _summary_moment_kind(summary) == "moment_rate":
        slip_unit = _summary_slip_unit_label(fault_summaries)
        return [
            "Fault/Group",
            "Moment Rate (N*m/yr)",
            "Mw equiv. (1 yr)",
            f"Mean Slip Rate ({slip_unit})",
            f"Max Slip Rate ({slip_unit})",
        ]
    return ["Fault/Group", "Moment (N*m)", "Mw", "Mean Slip (m)", "Max Slip (m)"]


def _moment_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    kind = _summary_moment_kind(summary)
    value_key = _moment_value_key(kind)
    magnitude_key = _moment_magnitude_key(kind)
    for item in summary.get("faults", []):
        moment = item.get("moment")
        slip = item.get("slip") or {}
        total_slip = slip.get("total") or {}
        if not moment:
            continue
        rows.append(
            [
                item.get("name"),
                _format_scientific(moment.get(value_key)),
                _format_number(moment.get(magnitude_key)),
                _format_number(total_slip.get("mean")),
                _format_number(total_slip.get("max")),
            ]
        )

    groups = [
        group for group in summary.get("groups", [])
        if group.get(value_key) is not None
    ]
    if groups:
        rows.append(["-" * 8, "-" * 8, "-" * 8, "-" * 8, "-" * 8])
        for group in groups:
            rows.append(
                [
                    group.get("name"),
                    _format_scientific(group.get(value_key)),
                    _format_number(group.get(magnitude_key)),
                    "-",
                    "-",
                ]
            )
    total = summary.get("total") or {}
    if total.get(value_key) is not None:
        rows.append(
            [
                "TOTAL",
                _format_scientific(total.get(value_key)),
                _format_number(total.get(magnitude_key)),
                "-",
                "-",
            ]
        )
    return rows


def _tabulate(rows: list[list[Any]], headers: list[str], *, tablefmt: str) -> str:
    try:
        from tabulate import tabulate

        return tabulate(
            rows,
            headers=headers,
            tablefmt=tablefmt,
            stralign="left",
            disable_numparse=True,
        )
    except Exception:
        table = [headers, *rows]
        widths = [max(len(str(row[index])) for row in table) for index in range(len(headers))]
        lines = []
        for row_index, row in enumerate(table):
            lines.append("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))
            if row_index == 0:
                lines.append("  ".join("-" * width for width in widths))
        return "\n".join(lines)


def _format_int(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "N/A"


def _format_number(value: Any, *, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.2f}{suffix}"


def _format_scientific(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.3e}"


def _format_range(stat: dict[str, Any], *, suffix: str = "") -> str:
    if not stat or stat.get("min") is None or stat.get("max") is None:
        return "N/A"
    return f"{_format_number(stat.get('min'))} - {_format_number(stat.get('max'))}{suffix}"


__all__ = (
    "get_fault_summary",
    "get_faults_summary",
    "print_fault_summary",
    "print_faults_summary",
    "show_fault_summary",
    "show_faults_summary",
    "summarize_fault",
    "summarize_faults",
)
