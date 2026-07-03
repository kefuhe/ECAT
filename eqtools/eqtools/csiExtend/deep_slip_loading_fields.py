"""Derived fields for the deep-slip loading proxy model."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .interseismic_fields import (
    _safe_ratio,
    extract_inverted_slip,
    get_fault_by_name,
    normalize_slip_component,
    summarize_values,
)


DEEP_SLIP_FIELD_ALIASES = {
    "deep_loading": "deep_loading_proxy_rate",
    "loading_proxy": "deep_loading_proxy_rate",
    "deep_loading_proxy": "deep_loading_proxy_rate",
    "deep_loading_proxy_rate": "deep_loading_proxy_rate",
    "tectonic_loading_proxy": "deep_loading_proxy_rate",
    "shallow_slip": "shallow_slip_rate",
    "shallow_slip_rate": "shallow_slip_rate",
    "creep": "shallow_slip_rate",
    "creep_rate": "shallow_slip_rate",
    "slip_deficit": "slip_deficit_to_deep_signed",
    "slip_deficit_signed": "slip_deficit_to_deep_signed",
    "slip_deficit_to_deep": "slip_deficit_to_deep_signed",
    "slip_deficit_to_deep_signed": "slip_deficit_to_deep_signed",
    "slip_deficit_abs": "slip_deficit_to_deep_magnitude",
    "slip_deficit_magnitude": "slip_deficit_to_deep_magnitude",
    "slip_deficit_to_deep_magnitude": "slip_deficit_to_deep_magnitude",
    "coupling": "coupling_to_deep",
    "coupling_ratio": "coupling_to_deep",
    "coupling_to_deep": "coupling_to_deep",
    "coupling_abs": "coupling_to_deep_magnitude",
    "coupling_magnitude": "coupling_to_deep_magnitude",
    "coupling_to_deep_magnitude": "coupling_to_deep_magnitude",
    "creep_fraction": "creep_fraction_to_deep",
    "creep_fraction_to_deep": "creep_fraction_to_deep",
    "mapping_distance": "mapping_distance",
    "distance": "mapping_distance",
    "matched_deep_patch": "matched_deep_patch",
}


def normalize_deep_slip_loading_field(field: str) -> str:
    """Return a canonical deep-slip loading proxy field name."""
    key = str(field).lower().replace("-", "_")
    try:
        return DEEP_SLIP_FIELD_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(DEEP_SLIP_FIELD_ALIASES))
        raise ValueError(f"Unknown deep-slip loading field '{field}'. Accepted values: {allowed}") from exc


def calculate_deep_slip_loading_fields(
    inversion: Any,
    mapping: Mapping[str, Any],
    *,
    solution: Sequence[float] | None = None,
    component: str | None = None,
    zero_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Calculate deep-proxy loading, shallow slip and coupling fields.

    The result is aligned with ``mapping["shallow_patch_indices"]`` rather than
    all patches of the shallow fault.  Use ``expand_deep_slip_loading_field`` if
    a full-fault array is needed for CSI patch GMT writing or plotting.

    Parameters
    ----------
    inversion : object
        BLSE/Bayesian-like inversion object.
    mapping : mapping
        Output from ``map_shallow_patches_to_deep_top_trace``.
    solution : sequence of float, optional
        Linear solution vector.  Defaults to ``inversion.mpost``.
    component : str, optional
        Slip component.  Defaults to ``mapping["component"]``.
    zero_tolerance : float, default 1e-12
        Denominator tolerance for ratio fields.

    Returns
    -------
    dict
        Result dictionary with ``fields``, ``stats`` and ``metadata``.
    """
    component_key = normalize_slip_component(component or mapping.get("component", "strikeslip"))
    shallow_fault_name = str(mapping["shallow_fault"])
    shallow_fault = get_fault_by_name(inversion, shallow_fault_name)
    shallow_indices = np.asarray(mapping["shallow_patch_indices"], dtype=int)
    deep_fault_names = np.asarray(mapping["deep_fault_names"], dtype=object)
    deep_patch_indices = np.asarray(mapping["deep_patch_indices"], dtype=int)
    if not (
        shallow_indices.shape[0] == deep_fault_names.shape[0] == deep_patch_indices.shape[0]
    ):
        raise ValueError("mapping arrays must have the same length")

    shallow_all = extract_inverted_slip(
        inversion,
        shallow_fault_name,
        solution=solution,
        slip_component=component_key,
    )
    shallow_slip = shallow_all[shallow_indices]

    deep_cache: dict[str, np.ndarray] = {}
    deep_loading = np.empty(shallow_indices.size, dtype=float)
    for i, (fault_name, patch_idx) in enumerate(zip(deep_fault_names.tolist(), deep_patch_indices.tolist())):
        fault_name = str(fault_name)
        if fault_name not in deep_cache:
            deep_cache[fault_name] = extract_inverted_slip(
                inversion,
                fault_name,
                solution=solution,
                slip_component=component_key,
            )
        deep_loading[i] = deep_cache[fault_name][int(patch_idx)]

    slip_deficit = deep_loading - shallow_slip
    slip_deficit_abs = np.abs(slip_deficit)
    coupling = _safe_ratio(slip_deficit, deep_loading, zero_tolerance)
    coupling_abs = _safe_ratio(slip_deficit_abs, np.abs(deep_loading), zero_tolerance)
    creep_fraction = _safe_ratio(shallow_slip, deep_loading, zero_tolerance)
    mapping_distance = np.asarray(mapping.get("mapping_distance", np.full(shallow_indices.size, np.nan)), dtype=float)
    matched_deep_patch = deep_patch_indices.astype(float)

    fields = {
        "deep_loading_proxy_rate": deep_loading,
        "shallow_slip_rate": shallow_slip,
        "slip_deficit_to_deep_signed": slip_deficit,
        "slip_deficit_to_deep_magnitude": slip_deficit_abs,
        "coupling_to_deep": coupling,
        "coupling_to_deep_magnitude": coupling_abs,
        "creep_fraction_to_deep": creep_fraction,
        "mapping_distance": mapping_distance,
        "matched_deep_patch": matched_deep_patch,
    }
    stats = {
        name: summarize_values(values)
        for name, values in fields.items()
        if name != "matched_deep_patch"
    }

    near_zero = int(np.sum(np.abs(deep_loading) < zero_tolerance))
    return {
        "fault_name": shallow_fault_name,
        "shallow_fault": shallow_fault_name,
        "fields": fields,
        "stats": stats,
        "metadata": {
            "model": "deep_slip_loading_proxy",
            "field_convention": "deep_slip_proxy",
            "slip_component": component_key,
            "unit": "same_as_solution_per_year",
            "n_selected_patches": int(shallow_indices.size),
            "n_fault_patches": len(shallow_fault.patch),
            "shallow_patch_indices": shallow_indices.tolist(),
            "deep_faults": sorted({str(name) for name in deep_fault_names.tolist()}),
            "near_zero_deep_loading_count": near_zero,
            "formulas": {
                "deep_loading_proxy_rate": "b",
                "shallow_slip_rate": "s",
                "slip_deficit_to_deep_signed": "b - s",
                "coupling_to_deep": "(b - s) / b",
                "creep_fraction_to_deep": "s / b",
            },
            "mapping": {
                "deep_fault_names": [str(name) for name in deep_fault_names.tolist()],
                "deep_patch_indices": deep_patch_indices.astype(int).tolist(),
                "mapping_distance": mapping_distance.astype(float).tolist(),
                "coord_frame": mapping.get("coord_frame"),
                "shallow_selector": mapping.get("shallow_selector"),
                "deep_selectors": mapping.get("deep_selectors"),
            },
        },
    }


def get_deep_slip_loading_field_values(result: Mapping[str, Any], field: str) -> np.ndarray:
    """Return one deep-slip loading field array using aliases."""
    canonical = normalize_deep_slip_loading_field(field)
    fields = result.get("fields", {})
    if canonical not in fields:
        available = ", ".join(sorted(fields))
        raise ValueError(f"Field '{field}' is not available. Available fields: {available}")
    return np.asarray(fields[canonical], dtype=float)


def expand_deep_slip_loading_field(
    result: Mapping[str, Any],
    fault: Any,
    field: str,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Expand a selected-patch deep-proxy field to all patches of the fault."""
    values = get_deep_slip_loading_field_values(result, field)
    patch_indices = np.asarray(result["metadata"]["shallow_patch_indices"], dtype=int)
    full = np.full(len(fault.patch), fill_value, dtype=float)
    full[patch_indices] = values
    return full
