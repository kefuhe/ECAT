"""Lightweight convergence diagnostics for eqtools SMC outputs."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import yaml


def evaluate_smc_convergence(
    samples: Sequence[Sequence[float]],
    *,
    postval: Optional[Sequence[float]] = None,
    beta: Optional[Sequence[float]] = None,
    sample_stats: Optional[Mapping[str, Sequence[float]]] = None,
    fault_parameter_stage_summary: Optional[Mapping[str, Sequence[float]]] = None,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    parameter_names: Optional[Sequence[str]] = None,
    key_parameters: Optional[Iterable[str]] = None,
    min_particles: int = 100,
    boundary_fraction_max: float = 0.10,
    boundary_tol_fraction: float = 0.01,
    normalized_ess_min: float = 0.20,
    max_weight_fraction_max: float = 0.50,
    unique_ancestor_fraction_min: float = 0.10,
    acceptance_rate_late_mean_min: float = 0.01,
    fault_trend_late_fraction: float = 0.25,
    fault_trend_beta_min: float = 0.50,
    fault_trend_min_stages: int = 2,
    fault_trend_drift_ratio_review: float = 0.50,
    fault_trend_drift_ratio_warning: float = 1.00,
    fault_ci_change_ratio_max: float = 0.30,
    fault_final_ci_to_max_ci_max: float = 0.90,
    fault_final_ci_to_prior_max: float = 0.70,
    fault_bound_distance_min: float = 0.02,
) -> Dict[str, Any]:
    """Evaluate diagnostics that are meaningful for one eqtools SMC run.

    The current SMC output stores the final particle ensemble, not independent
    long-chain traces.  The foreground report therefore focuses on the saved
    SMC process and on the stage evolution of fault geometry parameters.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError("samples must be a 2D array with shape (n_particles, n_parameters)")

    n_particles, n_parameters = arr.shape
    names = _resolve_parameter_names(parameter_names, n_parameters)
    selected = set(key_parameters or names)
    warnings = []

    beta_final = _final_scalar(beta)
    completed = bool(beta_final is not None and np.isclose(beta_final, 1.0))
    if not completed:
        warnings.append("beta_final_not_reached")

    finite_postval_ratio = None
    if postval is not None:
        post = np.asarray(postval, dtype=float)
        finite_postval_ratio = float(np.isfinite(post).mean()) if post.size else 0.0
        if finite_postval_ratio < 1.0:
            warnings.append("non_finite_postval")

    if n_particles < min_particles:
        warnings.append("too_few_particles")

    process_report = _smc_process_report(
        sample_stats,
        normalized_ess_min=normalized_ess_min,
        max_weight_fraction_max=max_weight_fraction_max,
        unique_ancestor_fraction_min=unique_ancestor_fraction_min,
        acceptance_rate_late_mean_min=acceptance_rate_late_mean_min,
    )
    warnings.extend(process_report.get("warnings", []))
    fault_trend_report = _fault_parameter_trend_report(
        fault_parameter_stage_summary,
        sample_stats=sample_stats,
        boundary_fraction_max=boundary_fraction_max,
        late_fraction=fault_trend_late_fraction,
        beta_min=fault_trend_beta_min,
        min_stages=fault_trend_min_stages,
        drift_ratio_review=fault_trend_drift_ratio_review,
        drift_ratio_warning=fault_trend_drift_ratio_warning,
    )
    warnings.extend(fault_trend_report.get("warnings", []))

    parameter_reports = {}
    lb = ub = None
    if lower_bounds is not None and upper_bounds is not None:
        lb = np.asarray(lower_bounds, dtype=float)
        ub = np.asarray(upper_bounds, dtype=float)
        if lb.shape[0] != n_parameters or ub.shape[0] != n_parameters:
            raise ValueError("lower_bounds and upper_bounds must match sample parameter count")
        parameter_reports = _boundary_reports(
            arr,
            names,
            selected,
            lb,
            ub,
            boundary_fraction_max=boundary_fraction_max,
            boundary_tol_fraction=boundary_tol_fraction,
        )
        if any(item["status"] == "WARNING" for item in parameter_reports.values()):
            warnings.append("boundary_hit")
    fault_checks_report = _fault_parameter_check_report(
        fault_parameter_stage_summary,
        sample_stats=sample_stats,
        samples=arr,
        parameter_names=names,
        lower_bounds=lb,
        upper_bounds=ub,
        beta_min=fault_trend_beta_min,
        min_stages=fault_trend_min_stages,
        late_fraction=fault_trend_late_fraction,
        median_trend_ratio_max=fault_trend_drift_ratio_review,
        median_trend_ratio_warning=fault_trend_drift_ratio_warning,
        ci_change_ratio_max=fault_ci_change_ratio_max,
        final_ci_to_max_ci_max=fault_final_ci_to_max_ci_max,
        final_ci_to_prior_max=fault_final_ci_to_prior_max,
        bound_distance_min=fault_bound_distance_min,
        boundary_mass_max=boundary_fraction_max,
        boundary_tol_fraction=boundary_tol_fraction,
    )
    warnings.extend(fault_checks_report.get("warnings", []))
    warnings = _unique_list(warnings)

    if not completed:
        status = "INCOMPLETE"
    elif finite_postval_ratio == 0.0:
        status = "FAILED"
    elif (
        process_report.get("status") == "warning"
        or fault_trend_report.get("status") == "drifting"
        or "too_few_particles" in warnings
        or "non_finite_postval" in warnings
    ):
        status = "WARNING"
    elif warnings:
        status = "REVIEW_RECOMMENDED"
    elif process_report.get("available") or fault_trend_report.get("available"):
        status = "OK_SINGLE_RUN"
    else:
        status = "DIAGNOSTICS_LIMITED"
    return {
        "status": status,
        "completed": {
            "beta_final": beta_final,
            "beta_reached_1": completed,
        },
        "samples": {
            "particles": int(n_particles),
            "parameters": int(n_parameters),
            "min_particles": int(min_particles),
        },
        "postval": {
            "finite_ratio": finite_postval_ratio,
        },
        "smc_process": process_report,
        "fault_parameter_trend": fault_trend_report,
        "fault_parameter_checks": fault_checks_report,
        "chains": {
            "independent_runs": 1,
            "rhat_available": False,
            "ess_available": False,
            "note": (
                "Traditional R-hat/ESS are not computed for one SMC run. "
                "Use independent random seeds for replicated convergence checks."
            ),
        },
        "parameters": parameter_reports,
        "warnings": warnings,
    }


def _resolve_parameter_names(parameter_names, n_parameters):
    if parameter_names is None:
        return [f"param_{i}" for i in range(n_parameters)]
    names = list(parameter_names)
    if len(names) != n_parameters:
        raise ValueError("parameter_names length must match sample parameter count")
    return names


def _final_scalar(values):
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[-1])


def _smc_process_report(
    sample_stats,
    *,
    normalized_ess_min,
    max_weight_fraction_max,
    unique_ancestor_fraction_min,
    acceptance_rate_late_mean_min,
):
    if not sample_stats:
        return {
            "available": False,
            "note": (
                "DIAGNOSTICS_LIMITED: this result stores only the final particle "
                "set; SMC process diagnostics are unavailable."
            ),
            "warnings": [],
        }

    stats = {
        key: np.asarray(value, dtype=float).reshape(-1)
        for key, value in sample_stats.items()
    }
    stages = _size_of(stats.get("stage"))
    if stages == 0:
        return {
            "available": False,
            "note": (
                "DIAGNOSTICS_LIMITED: sample_stats is present but contains no "
                "stage records."
            ),
            "warnings": [],
        }
    warnings = []

    normalized_ess = stats.get("normalized_ess")
    normalized_ess_summary = _threshold_summary(
        normalized_ess,
        stats.get("stage"),
        normalized_ess_min,
        direction="below",
    )
    normalized_ess_min_value = _nanmin(normalized_ess)
    if (
        normalized_ess_min_value is not None
        and normalized_ess_min_value < normalized_ess_min
    ):
        warnings.append("low_stage_ess")

    max_weight_fraction = stats.get("max_weight_fraction")
    max_weight_fraction_summary = _threshold_summary(
        max_weight_fraction,
        stats.get("stage"),
        max_weight_fraction_max,
        direction="above",
    )
    max_weight_fraction_value = _nanmax(max_weight_fraction)
    if (
        max_weight_fraction_value is not None
        and max_weight_fraction_value > max_weight_fraction_max
    ):
        warnings.append("weight_degeneracy")

    unique_ancestor_fraction = stats.get("unique_ancestor_fraction")
    unique_ancestor_fraction_summary = _threshold_summary(
        unique_ancestor_fraction,
        stats.get("stage"),
        unique_ancestor_fraction_min,
        direction="below",
    )
    unique_ancestor_fraction_min_value = _nanmin(unique_ancestor_fraction)
    if (
        unique_ancestor_fraction_min_value is not None
        and unique_ancestor_fraction_min_value < unique_ancestor_fraction_min
    ):
        warnings.append("ancestor_collapse")

    acceptance_rate_mean = stats.get("acceptance_rate_mean")
    acceptance_rate_summary = _threshold_summary(
        acceptance_rate_mean,
        stats.get("stage"),
        acceptance_rate_late_mean_min,
        direction="below",
    )
    acceptance_rate_late_mean = _late_mean(acceptance_rate_mean)
    if (
        acceptance_rate_late_mean is not None
        and acceptance_rate_late_mean < acceptance_rate_late_mean_min
    ):
        warnings.append("low_mutation_acceptance")

    delta_beta = stats.get("delta_beta")
    process_status, process_note = _smc_process_status_note(
        stages=stats.get("stage"),
        warnings=warnings,
        normalized_ess_summary=normalized_ess_summary,
        max_weight_fraction_summary=max_weight_fraction_summary,
        unique_ancestor_fraction_summary=unique_ancestor_fraction_summary,
        acceptance_rate_late_mean=acceptance_rate_late_mean,
        acceptance_rate_late_mean_min=acceptance_rate_late_mean_min,
    )
    return {
        "available": True,
        "status": process_status,
        "note": process_note,
        "stages": int(stages),
        "beta_final": _last_finite(stats.get("beta")),
        "delta_beta_min": _nanmin(delta_beta),
        "delta_beta_median": _nanmedian(delta_beta),
        "normalized_ess_threshold": float(normalized_ess_min),
        "normalized_ess_min": normalized_ess_min_value,
        "normalized_ess_min_stage": normalized_ess_summary["extreme_stage"],
        "normalized_ess_below_threshold_count": normalized_ess_summary["count"],
        "normalized_ess_valid_stage_count": normalized_ess_summary["valid_count"],
        "conditional_ess_min": _nanmin(stats.get("conditional_ess")),
        "max_weight_fraction_threshold": float(max_weight_fraction_max),
        "max_weight_fraction_max": max_weight_fraction_value,
        "max_weight_fraction_max_stage": max_weight_fraction_summary["extreme_stage"],
        "max_weight_fraction_above_threshold_count": max_weight_fraction_summary["count"],
        "max_weight_fraction_valid_stage_count": max_weight_fraction_summary["valid_count"],
        "unique_ancestor_fraction_threshold": float(unique_ancestor_fraction_min),
        "unique_ancestor_fraction_min": unique_ancestor_fraction_min_value,
        "unique_ancestor_fraction_min_stage": unique_ancestor_fraction_summary["extreme_stage"],
        "unique_ancestor_fraction_below_threshold_count": unique_ancestor_fraction_summary["count"],
        "unique_ancestor_fraction_valid_stage_count": unique_ancestor_fraction_summary["valid_count"],
        "acceptance_rate_threshold": float(acceptance_rate_late_mean_min),
        "acceptance_rate_late_mean": acceptance_rate_late_mean,
        "acceptance_rate_mean_min": _nanmin(acceptance_rate_mean),
        "acceptance_rate_mean_min_stage": acceptance_rate_summary["extreme_stage"],
        "acceptance_rate_below_threshold_count": acceptance_rate_summary["count"],
        "acceptance_rate_valid_stage_count": acceptance_rate_summary["valid_count"],
        "jump_distance_mean_median": _nanmedian(stats.get("jump_distance_mean")),
        "covariance_condition_max": _nanmax(stats.get("covariance_condition")),
        "warnings": warnings,
    }


def _smc_process_status_note(
    *,
    stages,
    warnings,
    normalized_ess_summary,
    max_weight_fraction_summary,
    unique_ancestor_fraction_summary,
    acceptance_rate_late_mean,
    acceptance_rate_late_mean_min,
):
    if not warnings:
        return "ok", "stage diagnostics stable"

    stage_values = _stage_values(stages)
    stage_count = stage_values.size
    late_start_index = max(0, int(np.floor(stage_count * 0.75)))
    late_stage_values = set(stage_values[late_start_index:].astype(int).tolist())
    process_trigger_stages = set()
    for summary in [
        normalized_ess_summary,
        max_weight_fraction_summary,
        unique_ancestor_fraction_summary,
    ]:
        process_trigger_stages.update(int(s) for s in summary.get("trigger_stages", []))

    if "low_mutation_acceptance" in warnings:
        return (
            "warning",
            (
                "low late-stage mutation acceptance "
                f"({acceptance_rate_late_mean:.3g} < "
                f"{acceptance_rate_late_mean_min:g})"
            ),
        )

    if not process_trigger_stages:
        return "review", "SMC process diagnostics require review"

    late_triggers = process_trigger_stages.intersection(late_stage_values)
    valid_count = max(
        normalized_ess_summary.get("valid_count", 0),
        max_weight_fraction_summary.get("valid_count", 0),
        unique_ancestor_fraction_summary.get("valid_count", 0),
    )
    trigger_count = len(process_trigger_stages)
    if valid_count <= 2:
        stages_text = _compact_stage_list(sorted(process_trigger_stages))
        return (
            "review",
            f"limited-stage particle degeneracy ({stages_text})",
        )
    if late_triggers or (valid_count and trigger_count / valid_count >= 0.25):
        stages_text = _compact_stage_list(sorted(process_trigger_stages))
        return (
            "warning",
            f"persistent or late particle degeneracy ({stages_text})",
        )

    stages_text = _compact_stage_list(sorted(process_trigger_stages))
    return (
        "review",
        f"early particle degeneracy only ({stages_text})",
    )


def _stage_values(stages):
    if stages is None:
        return np.asarray([], dtype=int)
    arr = np.asarray(stages, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.asarray([], dtype=int)
    return finite.astype(int)


def _compact_stage_list(stages, max_items=4):
    if not stages:
        return "no stage"
    text = ", ".join(f"stage {stage}" for stage in stages[:max_items])
    if len(stages) > max_items:
        text += f", +{len(stages) - max_items} more"
    return text


def _unique_list(items):
    unique = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _size_of(values):
    if values is None:
        return 0
    return np.asarray(values).size


def _finite(values):
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _nanmin(values):
    finite = _finite(values)
    if finite.size == 0:
        return None
    return float(np.min(finite))


def _nanmax(values):
    finite = _finite(values)
    if finite.size == 0:
        return None
    return float(np.max(finite))


def _nanmedian(values):
    finite = _finite(values)
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _last_finite(values):
    finite = _finite(values)
    if finite.size == 0:
        return None
    return float(finite[-1])


def _late_mean(values):
    finite = _finite(values)
    if finite.size == 0:
        return None
    start = max(0, int(np.floor(finite.size * 0.75)))
    return float(np.mean(finite[start:]))


def _threshold_summary(values, stages, threshold, *, direction):
    if values is None:
        return {
            "count": 0,
            "valid_count": 0,
            "extreme_stage": None,
            "trigger_stages": [],
        }
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite_mask = np.isfinite(arr)
    valid_count = int(np.count_nonzero(finite_mask))
    if valid_count == 0:
        return {
            "count": 0,
            "valid_count": 0,
            "extreme_stage": None,
            "trigger_stages": [],
        }
    if direction == "below":
        trigger_mask = finite_mask & (arr < threshold)
        extreme_index = int(np.nanargmin(np.where(finite_mask, arr, np.nan)))
    elif direction == "above":
        trigger_mask = finite_mask & (arr > threshold)
        extreme_index = int(np.nanargmax(np.where(finite_mask, arr, np.nan)))
    else:
        raise ValueError("direction must be 'below' or 'above'")
    return {
        "count": int(np.count_nonzero(trigger_mask)),
        "valid_count": valid_count,
        "extreme_stage": _stage_at(stages, extreme_index),
        "trigger_stages": [
            _stage_at(stages, int(index))
            for index in np.flatnonzero(trigger_mask)
        ],
    }


def _stage_at(stages, index):
    if stages is None:
        return int(index + 1)
    stage_arr = np.asarray(stages).reshape(-1)
    if index >= stage_arr.size:
        return int(index + 1)
    value = stage_arr[index]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(value) and value.is_integer():
        return int(value)
    if np.isfinite(value):
        return value
    return int(index + 1)


def _fault_parameter_trend_report(
    summary,
    *,
    sample_stats,
    boundary_fraction_max,
    late_fraction,
    beta_min,
    min_stages,
    drift_ratio_review,
    drift_ratio_warning,
):
    if not summary:
        return {
            "available": False,
            "status": "unavailable",
            "note": "fault parameter stage trends are unavailable",
            "warnings": [],
            "flagged_parameters": [],
        }

    try:
        stages = np.asarray(summary.get("stage"), dtype=float).reshape(-1)
        median = np.asarray(summary.get("median"), dtype=float)
    except (TypeError, ValueError):
        return {
            "available": False,
            "status": "unavailable",
            "note": "fault parameter stage trends are malformed",
            "warnings": [],
            "flagged_parameters": [],
        }

    if median.ndim == 1:
        median = median.reshape((-1, 1))
    if stages.size < 2 or median.shape[0] < 2:
        return {
            "available": False,
            "status": "unavailable",
            "note": "fault parameter stage trends need at least two stages",
            "warnings": [],
            "flagged_parameters": [],
        }

    names = [str(item) for item in summary.get("parameter_names", [])]
    if not names:
        names = [f"fault_param_{i}" for i in range(median.shape[1])]
    labels = [str(item) for item in summary.get("display_names", names)]
    if len(labels) < len(names):
        labels.extend(names[len(labels):])

    ci_width = _trend_ci_width(summary, median.shape)
    boundary_fraction = _trend_boundary_fraction(summary, median.shape)
    window = _fault_trend_window(
        stages,
        median.shape[0],
        sample_stats=sample_stats,
        beta_min=beta_min,
        min_stages=min_stages,
        late_fraction=late_fraction,
    )
    trend_indices = window["indices"]
    late_median = median[trend_indices, :]
    final_width = ci_width[-1, :] if ci_width.size else np.full(median.shape[1], np.nan)
    late_range = np.nanmax(late_median, axis=0) - np.nanmin(late_median, axis=0)

    drift_ratio = np.full(median.shape[1], np.nan)
    for idx in range(median.shape[1]):
        width = final_width[idx]
        if np.isfinite(width) and width > 0:
            drift_ratio[idx] = late_range[idx] / width
        elif np.isfinite(late_range[idx]) and late_range[idx] == 0:
            drift_ratio[idx] = 0.0
        elif np.isfinite(late_range[idx]):
            drift_ratio[idx] = np.inf

    final_boundary = (
        boundary_fraction[-1, :]
        if boundary_fraction.size
        else np.full(median.shape[1], np.nan)
    )
    flagged = []
    for idx, name in enumerate(names[: median.shape[1]]):
        reasons = []
        ratio = drift_ratio[idx]
        boundary = final_boundary[idx]
        if (np.isfinite(ratio) and ratio > drift_ratio_review) or np.isinf(ratio):
            reasons.append("late_stage_drift")
        if (np.isfinite(ratio) and ratio > drift_ratio_warning) or np.isinf(ratio):
            reasons.append("late_stage_warning_drift")
        if np.isfinite(boundary) and boundary > boundary_fraction_max:
            reasons.append("boundary_limited")
        if reasons:
            flagged.append({
                "name": name,
                "label": labels[idx] if idx < len(labels) else name,
                "reasons": reasons,
                "drift_ratio": None if not np.isfinite(ratio) else float(ratio),
                "boundary_fraction": None if not np.isfinite(boundary) else float(boundary),
            })

    warnings = []
    has_drift = any("late_stage_drift" in item["reasons"] for item in flagged)
    has_warning_drift = any(
        "late_stage_warning_drift" in item["reasons"] for item in flagged
    )
    has_boundary = any("boundary_limited" in item["reasons"] for item in flagged)
    if has_warning_drift:
        status = "drifting"
        warnings.append("fault_parameter_drift")
    elif has_drift:
        status = "review"
        warnings.append("fault_parameter_drift")
    else:
        status = "stable"
    if has_boundary:
        warnings.append("fault_parameter_boundary_limited")

    note = _fault_trend_note(status, flagged, drift_ratio, names, labels, window)
    max_ratio = _trend_nanmax(drift_ratio)
    max_ratio_idx = _trend_nanargmax(drift_ratio)
    return {
        "available": True,
        "status": status,
        "note": note,
        "late_stage_count": int(trend_indices.size),
        "trend_window": {
            key: value
            for key, value in window.items()
            if key != "indices"
        },
        "drift_ratio_review": float(drift_ratio_review),
        "drift_ratio_warning": float(drift_ratio_warning),
        "boundary_fraction_max": float(boundary_fraction_max),
        "max_drift_ratio": max_ratio,
        "max_drift_parameter": (
            names[max_ratio_idx]
            if max_ratio_idx is not None and max_ratio_idx < len(names)
            else None
        ),
        "flagged_parameters": flagged,
        "warnings": warnings,
    }


def _fault_trend_window(
    stages,
    n_stages,
    *,
    sample_stats,
    beta_min,
    min_stages,
    late_fraction,
):
    min_stages = max(2, int(min_stages))
    beta_by_stage = _fault_trend_beta_by_stage(stages, sample_stats)
    if beta_by_stage is not None:
        posterior_indices = np.flatnonzero(beta_by_stage >= beta_min)
        if posterior_indices.size >= min_stages:
            return _trend_window_record(
                posterior_indices,
                stages,
                kind="posterior_near_beta",
                beta_min=beta_min,
                beta_values=beta_by_stage,
                posterior_near_stage_count=int(posterior_indices.size),
                used_fallback=False,
            )
        fallback = np.arange(max(0, n_stages - min_stages), n_stages, dtype=int)
        return _trend_window_record(
            fallback,
            stages,
            kind="last_stages_limited_posterior_beta",
            beta_min=beta_min,
            beta_values=beta_by_stage,
            posterior_near_stage_count=int(posterior_indices.size),
            used_fallback=True,
        )

    fallback_count = min(min_stages, n_stages)
    indices = np.arange(max(0, n_stages - fallback_count), n_stages, dtype=int)
    return _trend_window_record(
        indices,
        stages,
        kind="last_stages_no_beta",
        beta_min=beta_min,
        beta_values=None,
        posterior_near_stage_count=None,
        used_fallback=True,
    )


def _fault_trend_beta_by_stage(stages, sample_stats):
    if not sample_stats or "beta" not in sample_stats:
        return None
    beta = np.asarray(sample_stats["beta"], dtype=float).reshape(-1)
    if beta.size == 0:
        return None
    if beta.size == stages.size:
        return beta
    if "stage" not in sample_stats:
        return None
    stat_stages = np.asarray(sample_stats["stage"], dtype=float).reshape(-1)
    if stat_stages.size != beta.size:
        return None
    beta_by_stage = np.full(stages.size, np.nan)
    beta_map = {
        int(stage): beta_value
        for stage, beta_value in zip(stat_stages, beta)
        if np.isfinite(stage) and np.isfinite(beta_value)
    }
    for idx, stage in enumerate(stages):
        if np.isfinite(stage):
            beta_by_stage[idx] = beta_map.get(int(stage), np.nan)
    if np.all(np.isnan(beta_by_stage)):
        return None
    return beta_by_stage


def _trend_window_record(
    indices,
    stages,
    *,
    kind,
    beta_min,
    beta_values,
    posterior_near_stage_count,
    used_fallback,
):
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        indices = np.asarray([max(0, len(stages) - 1)], dtype=int)
    stage_values = np.asarray(stages, dtype=float).reshape(-1)
    selected_stages = stage_values[indices]
    record = {
        "indices": indices,
        "kind": kind,
        "stage_count": int(indices.size),
        "stage_start": _clean_number(selected_stages[0]),
        "stage_end": _clean_number(selected_stages[-1]),
        "beta_min": float(beta_min),
        "posterior_near_stage_count": posterior_near_stage_count,
        "used_fallback": bool(used_fallback),
    }
    if beta_values is not None:
        selected_beta = np.asarray(beta_values, dtype=float).reshape(-1)[indices]
        record["beta_start"] = _clean_number(selected_beta[0])
        record["beta_end"] = _clean_number(selected_beta[-1])
    return record


def _clean_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    if value.is_integer():
        return int(value)
    return value


def _trend_ci_width(summary, target_shape):
    if "ci_width" in summary:
        width = np.asarray(summary["ci_width"], dtype=float)
    elif "ci_lower" in summary and "ci_upper" in summary:
        width = np.asarray(summary["ci_upper"], dtype=float) - np.asarray(
            summary["ci_lower"],
            dtype=float,
        )
    else:
        width = np.full(target_shape, np.nan)
    if width.ndim == 1:
        width = width.reshape((-1, 1))
    return width


def _trend_boundary_fraction(summary, target_shape):
    if "boundary_fraction" not in summary:
        return np.full(target_shape, np.nan)
    arr = np.asarray(summary["boundary_fraction"], dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    return arr


def _fault_parameter_check_report(
    summary,
    *,
    sample_stats,
    samples,
    parameter_names,
    lower_bounds,
    upper_bounds,
    beta_min,
    min_stages,
    late_fraction,
    median_trend_ratio_max,
    median_trend_ratio_warning,
    ci_change_ratio_max,
    final_ci_to_max_ci_max,
    final_ci_to_prior_max,
    bound_distance_min,
    boundary_mass_max,
    boundary_tol_fraction,
):
    if not summary:
        return {
            "available": False,
            "note": "fault parameter checks are unavailable",
            "warnings": [],
            "rows": [],
            "summaries": {},
        }

    try:
        stages = np.asarray(summary.get("stage"), dtype=float).reshape(-1)
        median = np.asarray(summary.get("median"), dtype=float)
    except (TypeError, ValueError):
        return {
            "available": False,
            "note": "fault parameter checks are malformed",
            "warnings": [],
            "rows": [],
            "summaries": {},
        }
    if median.ndim == 1:
        median = median.reshape((-1, 1))
    if stages.size < 2 or median.shape[0] < 2:
        return {
            "available": False,
            "note": "fault parameter checks need at least two stages",
            "warnings": [],
            "rows": [],
            "summaries": {},
        }

    names = [str(item) for item in summary.get("parameter_names", [])]
    if not names:
        names = [f"fault_param_{i}" for i in range(median.shape[1])]
    labels = [str(item) for item in summary.get("display_names", names)]
    if len(labels) < len(names):
        labels.extend(names[len(labels):])

    ci_width = _trend_ci_width(summary, median.shape)
    ci_lower = _trend_optional_matrix(summary, "ci_lower", median.shape)
    ci_upper = _trend_optional_matrix(summary, "ci_upper", median.shape)
    boundary_fraction = _trend_boundary_fraction(summary, median.shape)
    window = _fault_trend_window(
        stages,
        median.shape[0],
        sample_stats=sample_stats,
        beta_min=beta_min,
        min_stages=min_stages,
        late_fraction=late_fraction,
    )
    indices = window["indices"]
    bound_lookup = _parameter_bound_lookup(parameter_names, lower_bounds, upper_bounds)
    sample_lookup = _parameter_sample_lookup(parameter_names, samples)
    rows = []
    for idx, name in enumerate(names[: median.shape[1]]):
        label = labels[idx] if idx < len(labels) else name
        final_ci_width = _safe_float(ci_width[-1, idx])
        max_stage_ci_width = _safe_float(_nanmax(ci_width[:, idx]))
        median_trend_ratio = _ratio(
            np.nanmax(median[indices, idx]) - np.nanmin(median[indices, idx]),
            final_ci_width,
        )
        ci_width_delta = _safe_float(ci_width[indices[-1], idx] - ci_width[indices[0], idx])
        if ci_width_delta is None:
            ci_change_ratio = None
            ci_widening_ratio = None
            ci_narrowing_ratio = None
        else:
            ci_change_ratio = _ratio(abs(ci_width_delta), final_ci_width)
            ci_widening_ratio = _ratio(max(ci_width_delta, 0.0), final_ci_width)
            ci_narrowing_ratio = _ratio(max(-ci_width_delta, 0.0), final_ci_width)
        final_ci_to_max_ci = _ratio(final_ci_width, max_stage_ci_width)

        lower = upper = prior_width = None
        final_ci_to_prior = None
        bound_distance = None
        bound_direction = None
        lower_boundary_mass = None
        upper_boundary_mass = None
        boundary_mass = None
        boundary_mass_direction = None
        median_prior_position = None
        if name in bound_lookup:
            lower, upper = bound_lookup[name]
            prior_width = upper - lower
            if np.isfinite(prior_width) and prior_width > 0:
                final_ci_to_prior = _ratio(final_ci_width, prior_width)
                median_prior_position = _ratio(median[-1, idx] - lower, prior_width)
                if ci_lower is not None and ci_upper is not None:
                    lower_dist = _ratio(ci_lower[-1, idx] - lower, prior_width)
                    upper_dist = _ratio(upper - ci_upper[-1, idx], prior_width)
                    if _is_number(lower_dist) and _is_number(upper_dist):
                        if lower_dist <= upper_dist:
                            bound_distance = lower_dist
                            bound_direction = "lower"
                        else:
                            bound_distance = upper_dist
                            bound_direction = "upper"
                if name in sample_lookup:
                    values = np.asarray(sample_lookup[name], dtype=float)
                    tol = prior_width * boundary_tol_fraction
                    near_lower = values <= lower + tol
                    near_upper = values >= upper - tol
                    lower_boundary_mass = _safe_float(np.mean(near_lower))
                    upper_boundary_mass = _safe_float(np.mean(near_upper))
                    if (
                        lower_boundary_mass is not None
                        and upper_boundary_mass is not None
                    ):
                        if lower_boundary_mass >= upper_boundary_mass:
                            boundary_mass = lower_boundary_mass
                            boundary_mass_direction = "lower"
                        else:
                            boundary_mass = upper_boundary_mass
                            boundary_mass_direction = "upper"

        flags = []
        info_flags = []
        if _gt(median_trend_ratio, median_trend_ratio_max):
            flags.append("median_trend")
        if _gt(ci_widening_ratio, ci_change_ratio_max):
            flags.append("ci_width_widening")
        if _gt(ci_narrowing_ratio, ci_change_ratio_max):
            info_flags.append("ci_width_narrowing")
        if _gt(final_ci_to_max_ci, final_ci_to_max_ci_max):
            flags.append("ci_not_contracted")
        if _gt(final_ci_to_prior, final_ci_to_prior_max):
            flags.append("broad_ci")
        if _lt(bound_distance, bound_distance_min):
            info_flags.append("ci_edge_near_bound")
        if _gte(boundary_mass, boundary_mass_max):
            flags.append("boundary_mass")

        rows.append({
            "name": name,
            "label": label,
            "final_ci_to_max_ci": final_ci_to_max_ci,
            "final_ci_to_prior": final_ci_to_prior,
            "median_trend_ratio": median_trend_ratio,
            "ci_change_ratio": ci_change_ratio,
            "ci_width_delta": ci_width_delta,
            "ci_widening_ratio": ci_widening_ratio,
            "ci_narrowing_ratio": ci_narrowing_ratio,
            "bound_distance": bound_distance,
            "bound_direction": bound_direction,
            "boundary_mass": boundary_mass,
            "boundary_mass_direction": boundary_mass_direction,
            "lower_boundary_mass": lower_boundary_mass,
            "upper_boundary_mass": upper_boundary_mass,
            "median_prior_position": median_prior_position,
            "boundary_fraction": _safe_float(boundary_fraction[-1, idx]),
            "final_ci_width": final_ci_width,
            "max_stage_ci_width": max_stage_ci_width,
            "flags": flags,
            "info_flags": info_flags,
        })

    summaries = {
        "median_trend": _check_summary(
            rows,
            flag="median_trend",
            ok_note="median stable over posterior-near stages",
            flagged_note="median still drifting in",
        ),
        "uncertainty": _uncertainty_summary(rows),
        "late_ci_behavior": _late_ci_behavior_summary(rows),
    }
    bound_proximity = _bound_proximity_summary(rows)
    summaries["bound_proximity"] = bound_proximity
    summaries["bound_pressure"] = bound_proximity
    warnings = []
    if summaries["median_trend"]["flagged_parameters"]:
        warnings.append("fault_parameter_drift")
    if summaries["uncertainty"]["flagged_parameters"]:
        warnings.append("fault_parameter_uncertainty")
    if summaries["bound_proximity"]["flagged_parameters"]:
        warnings.append("fault_parameter_bound_proximity")

    return {
        "available": True,
        "note": "fault parameter checks computed",
        "thresholds": {
            "median_trend_ratio_max": float(median_trend_ratio_max),
            "median_trend_ratio_warning": float(median_trend_ratio_warning),
            "ci_change_ratio_max": float(ci_change_ratio_max),
            "ci_widening_ratio_max": float(ci_change_ratio_max),
            "ci_narrowing_ratio_display_min": float(ci_change_ratio_max),
            "final_ci_to_max_ci_max": float(final_ci_to_max_ci_max),
            "final_ci_to_prior_max": float(final_ci_to_prior_max),
            "bound_distance_min": float(bound_distance_min),
            "boundary_mass_max": float(boundary_mass_max),
            "boundary_tol_fraction": float(boundary_tol_fraction),
            "beta_min": float(beta_min),
        },
        "trend_window": {
            key: value
            for key, value in window.items()
            if key != "indices"
        },
        "summaries": summaries,
        "rows": rows,
        "warnings": warnings,
    }


def _trend_optional_matrix(summary, key, target_shape):
    if key not in summary:
        return None
    arr = np.asarray(summary[key], dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    if arr.shape != target_shape:
        return None
    return arr


def _parameter_bound_lookup(parameter_names, lower_bounds, upper_bounds):
    if parameter_names is None or lower_bounds is None or upper_bounds is None:
        return {}
    lb = np.asarray(lower_bounds, dtype=float).reshape(-1)
    ub = np.asarray(upper_bounds, dtype=float).reshape(-1)
    lookup = {}
    for idx, name in enumerate(parameter_names):
        if idx < lb.size and idx < ub.size:
            lookup[str(name)] = (float(lb[idx]), float(ub[idx]))
    return lookup


def _parameter_sample_lookup(parameter_names, samples):
    if parameter_names is None or samples is None:
        return {}
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        return {}
    lookup = {}
    for idx, name in enumerate(parameter_names):
        if idx < arr.shape[1]:
            lookup[str(name)] = arr[:, idx]
    return lookup


def _check_summary(rows, *, flag, ok_note, flagged_note):
    flagged = [
        {
            "name": row["name"],
            "label": row["label"],
            "flags": list(row["flags"]),
        }
        for row in rows
        if flag in row["flags"]
    ]
    if not flagged:
        return {
            "status": "ok",
            "note": ok_note,
            "flagged_parameters": [],
        }
    labels = [item["label"] for item in flagged]
    return {
        "status": "review",
        "note": f"{flagged_note} {_compact_name_list(labels)}",
        "flagged_parameters": flagged,
    }


def _bound_proximity_summary(rows):
    flags = {"boundary_mass"}
    flagged = [
        {
            "name": row["name"],
            "label": row["label"],
            "flags": [flag for flag in row["flags"] if flag in flags],
            "bound_direction": row.get("bound_direction"),
            "boundary_mass_direction": row.get("boundary_mass_direction"),
            "bound_distance": row.get("bound_distance"),
            "boundary_mass": row.get("boundary_mass"),
        }
        for row in rows
        if flags.intersection(row["flags"])
    ]
    info = [
        {
            "name": row["name"],
            "label": row["label"],
            "info_flags": [
                flag
                for flag in row.get("info_flags", [])
                if flag == "ci_edge_near_bound"
            ],
            "bound_direction": row.get("bound_direction"),
            "bound_distance": row.get("bound_distance"),
            "boundary_mass": row.get("boundary_mass"),
            "boundary_mass_direction": row.get("boundary_mass_direction"),
        }
        for row in rows
        if "ci_edge_near_bound" in row.get("info_flags", [])
    ]
    if not flagged:
        if info:
            labels = [item["label"] for item in info]
            return {
                "status": "info",
                "note": f"CI edge near prior bound in {_compact_name_list(labels)}",
                "flagged_parameters": [],
                "info_parameters": info,
            }
        return {
            "status": "ok",
            "note": "none",
            "flagged_parameters": [],
            "info_parameters": [],
        }
    labels = [item["label"] for item in flagged]
    return {
        "status": "review",
        "note": f"posterior near prior bound in {_compact_name_list(labels)}",
        "flagged_parameters": flagged,
        "info_parameters": info,
    }


def _uncertainty_summary(rows):
    flags = {"ci_width_widening", "ci_not_contracted", "broad_ci"}
    flagged = [
        {
            "name": row["name"],
            "label": row["label"],
            "flags": [flag for flag in row["flags"] if flag in flags],
        }
        for row in rows
        if flags.intersection(row["flags"])
    ]
    if not flagged:
        return {
            "status": "ok",
            "note": "none",
            "flagged_parameters": [],
        }
    labels = [item["label"] for item in flagged]
    return {
        "status": "review",
        "note": f"broad or unstable CI in {_compact_name_list(labels)}",
        "flagged_parameters": flagged,
    }


def _late_ci_behavior_summary(rows):
    widening = [
        {
            "name": row["name"],
            "label": row["label"],
            "ci_widening_ratio": row.get("ci_widening_ratio"),
        }
        for row in rows
        if "ci_width_widening" in row["flags"]
    ]
    narrowing = [
        {
            "name": row["name"],
            "label": row["label"],
            "ci_narrowing_ratio": row.get("ci_narrowing_ratio"),
        }
        for row in rows
        if "ci_width_narrowing" in row.get("info_flags", [])
    ]
    if widening:
        labels = [item["label"] for item in widening]
        return {
            "status": "review",
            "note": f"late CI widening in {_compact_name_list(labels)}",
            "widening_parameters": widening,
            "narrowing_parameters": narrowing,
        }
    if narrowing:
        labels = [item["label"] for item in narrowing]
        return {
            "status": "informative",
            "note": f"late CI narrowing in {_compact_name_list(labels)}",
            "widening_parameters": [],
            "narrowing_parameters": narrowing,
        }
    return {
        "status": "stable",
        "note": "late CI widths are stable within the display threshold",
        "widening_parameters": [],
        "narrowing_parameters": [],
    }


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _ratio(numerator, denominator):
    numerator = _safe_float(numerator)
    denominator = _safe_float(denominator)
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator)


def _is_number(value):
    return value is not None and np.isfinite(value)


def _gt(value, threshold):
    return value is not None and value > threshold


def _gte(value, threshold):
    return value is not None and threshold is not None and value >= threshold


def _lt(value, threshold):
    return value is not None and value < threshold


def _fault_trend_note(status, flagged, drift_ratio, names, labels, window):
    window_text = _trend_window_text(window)
    if status == "stable":
        max_ratio = _trend_nanmax(drift_ratio)
        if max_ratio is None:
            return f"fault parameter medians stable over {window_text}"
        idx = _trend_nanargmax(drift_ratio)
        label = labels[idx] if idx is not None and idx < len(labels) else names[idx]
        return (
            f"fault parameter medians stable over {window_text} "
            f"(largest drift {label}: {max_ratio:.3g} x final CI width)"
        )
    if status == "drifting":
        items = [
            item["label"]
            for item in flagged
            if "late_stage_warning_drift" in item["reasons"]
        ]
        return "warning-level trend-window drift in " + _compact_name_list(items)
    if status == "review":
        items = [
            item["label"]
            for item in flagged
            if "late_stage_drift" in item["reasons"]
        ]
        return "review-level trend-window drift in " + _compact_name_list(items)
    items = [
        item["label"]
        for item in flagged
        if "boundary_limited" in item["reasons"]
    ]
    return "posterior mass near prior boundary in " + _compact_name_list(items)


def _trend_window_text(window):
    kind = window.get("kind")
    count = window.get("stage_count")
    stage_start = window.get("stage_start")
    stage_end = window.get("stage_end")
    if kind == "posterior_near_beta":
        beta_min = window.get("beta_min")
        return (
            f"{count} posterior-near stages "
            f"(beta >= {beta_min:g}, stage {stage_start}-{stage_end})"
        )
    if kind == "last_stages_limited_posterior_beta":
        beta_min = window.get("beta_min")
        available = window.get("posterior_near_stage_count")
        return (
            f"last {count} stages (only {available} stage(s) with "
            f"beta >= {beta_min:g})"
        )
    return f"last {count} stages (stage {stage_start}-{stage_end}; beta unavailable)"


def _compact_name_list(names, max_items=4):
    names = [str(name) for name in names]
    if not names:
        return "none"
    text = ", ".join(names[:max_items])
    if len(names) > max_items:
        text += f", +{len(names) - max_items} more"
    return text


def _trend_nanmax(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return None
    return float(np.max(valid))


def _trend_nanargmax(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return None
    return int(np.nanargmax(np.where(valid_mask, arr, np.nan)))


def _boundary_reports(
    samples,
    names,
    selected,
    lower_bounds,
    upper_bounds,
    *,
    boundary_fraction_max,
    boundary_tol_fraction,
):
    reports = {}
    for idx, name in enumerate(names):
        if name not in selected:
            continue
        lb = lower_bounds[idx]
        ub = upper_bounds[idx]
        width = ub - lb
        if not np.isfinite(width) or width <= 0:
            reports[name] = {
                "status": "WARNING",
                "boundary_fraction": None,
            }
            continue
        tol = width * boundary_tol_fraction
        near_lower = samples[:, idx] <= lb + tol
        near_upper = samples[:, idx] >= ub - tol
        frac = float(np.logical_or(near_lower, near_upper).mean())
        ci95 = [
            float(np.percentile(samples[:, idx], 2.5)),
            float(np.percentile(samples[:, idx], 97.5)),
        ]
        status = "WARNING" if frac > boundary_fraction_max else "OK"
        reports[name] = {
            "status": status,
            "median": float(np.median(samples[:, idx])),
            "ci95": ci95,
            "boundary_fraction": frac,
            "boundary_direction": _boundary_direction(near_lower, near_upper),
        }
    return reports


def _boundary_direction(near_lower, near_upper):
    lower_count = int(np.count_nonzero(near_lower))
    upper_count = int(np.count_nonzero(near_upper))
    if lower_count == 0 and upper_count == 0:
        return None
    if lower_count >= upper_count:
        return "lower"
    return "upper"


def evaluate_smc_h5(
    filename: str,
    *,
    lower_bounds=None,
    upper_bounds=None,
    parameter_names=None,
    key_parameters=None,
    **kwargs,
) -> Dict[str, Any]:
    """Read a standard eqtools SMC HDF5 file and evaluate diagnostics."""
    import h5py

    with h5py.File(filename, "r") as h5:
        samples = np.asarray(h5["allsamples"])
        postval = np.asarray(h5["postval"]) if "postval" in h5 else None
        beta = np.asarray(h5["beta"]) if "beta" in h5 else None
        sample_stats = None
        if "sample_stats" in h5:
            sample_stats = {
                key: np.asarray(h5["sample_stats"][key])
                for key in h5["sample_stats"].keys()
            }
        fault_parameter_stage_summary = None
        if "fault_parameter_stage_summary" in h5:
            group = h5["fault_parameter_stage_summary"]
            fault_parameter_stage_summary = {
                key: _read_h5_value(group[key])
                for key in group.keys()
            }
    return evaluate_smc_convergence(
        samples,
        postval=postval,
        beta=beta,
        sample_stats=sample_stats,
        fault_parameter_stage_summary=fault_parameter_stage_summary,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        parameter_names=parameter_names,
        key_parameters=key_parameters,
        **kwargs,
    )


def write_convergence_report(report: Dict[str, Any], filename: str) -> None:
    """Write a convergence report to YAML or JSON."""
    if filename.lower().endswith(".json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return
    with open(filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)


def _read_h5_value(dataset):
    value = dataset[()]
    arr = np.asarray(value)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8").tolist()
    if arr.dtype.kind == "O":
        return [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in arr.reshape(-1)
        ]
    return value
