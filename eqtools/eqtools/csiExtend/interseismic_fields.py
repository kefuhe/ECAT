"""
Interseismic kinematic fields for fault slip inversions.

The functions in this module are intentionally stateless: they read a current
    linear solution vector and fault geometry, then return arrays for tectonic
    loading, direct backslip, slip deficit, coupling, and creep.  They do not
    modify ``fault.slip``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config.config_utils import get_observation_unit_info, m_per_year_to_observation_factor
from .interseismic_parameter_model import (
    calculate_loading_from_terms,
    get_fault_loading_params,
    resolve_euler_block_vectors as _resolve_euler_block_vectors,
)
from .patch_indices import normalize_patch_indices


FIELD_ALIASES = {
    "block_slip_rate": "block_slip_rate_signed",
    "block_slip_rate_signed": "block_slip_rate_signed",
    "loading": "tectonic_loading_rate",
    "loading_rate": "tectonic_loading_rate",
    "tectonic_loading": "tectonic_loading_rate",
    "tectonic_loading_rate": "tectonic_loading_rate",
    "slip": "backslip_rate",
    "backslip": "backslip_rate",
    "backslip_rate": "backslip_rate",
    "backslip_strikeslip": "backslip_strikeslip",
    "backslip_dipslip": "backslip_dipslip",
    "backslip_total": "backslip_total",
    "direct_backslip": "backslip_rate",
    "inverted": "inverted_slip",
    "inverted_slip": "inverted_slip",
    "inverted_strikeslip": "inverted_strikeslip",
    "inverted_dipslip": "inverted_dipslip",
    "inverted_total": "inverted_total",
    "slip_deficit": "slip_deficit_signed",
    "slip_deficit_signed": "slip_deficit_signed",
    "signed_slip_deficit": "slip_deficit_signed",
    "slip_deficit_magnitude": "slip_deficit_magnitude",
    "slip_deficit_abs": "slip_deficit_magnitude",
    "absolute_slip_deficit": "slip_deficit_magnitude",
    "coupling": "coupling_ratio",
    "coupling_ratio": "coupling_ratio",
    "coupling_rate": "coupling_ratio",
    "coupling_magnitude": "coupling_magnitude",
    "coupling_abs": "coupling_magnitude",
    "creep": "creep_rate_signed",
    "creep_rate": "creep_rate_signed",
    "creep_rate_signed": "creep_rate_signed",
    "creep_ratio": "creep_ratio",
    "unlocked_ratio": "creep_ratio",
}

SLIP_COMPONENT_ALIASES = {
    "s": "strikeslip",
    "ss": "strikeslip",
    "strike": "strikeslip",
    "strike_slip": "strikeslip",
    "strikeslip": "strikeslip",
    "d": "dipslip",
    "ds": "dipslip",
    "dip": "dipslip",
    "dip_slip": "dipslip",
    "dipslip": "dipslip",
    "total": "total",
    "magnitude": "total",
}


def normalize_slip_component(component: str) -> str:
    """Return the canonical slip component name.

    Parameters
    ----------
    component : str
        Slip component requested by the user.  Accepted aliases are
        ``strikeslip``/``ss``/``strike``, ``dipslip``/``ds``/``dip``, and
        ``total``/``magnitude``.

    Returns
    -------
    str
        One of ``"strikeslip"``, ``"dipslip"``, or ``"total"``.
    """
    key = str(component).lower().replace("-", "_")
    try:
        return SLIP_COMPONENT_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(SLIP_COMPONENT_ALIASES))
        raise ValueError(f"Invalid slip_component '{component}'. Accepted values: {allowed}") from exc


def normalize_interseismic_field(field: str) -> str:
    """Return the canonical interseismic field name.

    Parameters
    ----------
    field : str
        Field name or alias.  Common aliases include ``loading``,
        ``backslip``, ``slip_deficit``, ``coupling``, ``creep``, and ``slip``.

    Returns
    -------
    str
        Canonical field name used in result dictionaries.
    """
    key = str(field).lower().replace("-", "_")
    try:
        return FIELD_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(FIELD_ALIASES))
        raise ValueError(f"Unknown interseismic field '{field}'. Accepted values: {allowed}") from exc


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    """Calculate compact descriptive statistics for one patch field.

    Parameters
    ----------
    values : sequence of float
        Per-patch scalar field.

    Returns
    -------
    dict
        ``min``, ``max``, ``mean``, ``std``, ``median`` and ``count`` values.
        Empty arrays return ``nan`` for statistics and ``count=0``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "count": 0,
        }
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "median": float(np.nanmedian(arr)),
        "count": int(arr.size),
    }


def _safe_ratio(
    numerator: Sequence[float],
    denominator: Sequence[float],
    zero_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return numerator / denominator and zero values with near-zero denominator."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if num.shape != den.shape:
        raise ValueError(f"numerator shape {num.shape} does not match denominator shape {den.shape}")
    values = np.zeros_like(num, dtype=float)
    mask = np.abs(den) >= zero_tolerance
    values[mask] = num[mask] / den[mask]
    return values


def get_faults_from_inversion(inversion: Any) -> Sequence[Any]:
    """Return fault/source objects from a BLSE or Bayesian inversion object."""
    if hasattr(inversion, "_get_faults"):
        return inversion._get_faults()
    if hasattr(inversion, "multifaults") and hasattr(inversion.multifaults, "faults"):
        return inversion.multifaults.faults
    if hasattr(inversion, "faults"):
        return inversion.faults
    raise AttributeError("Cannot find fault list on inversion object")


def get_fault_by_name(inversion: Any, fault_name: str) -> Any:
    """Return one fault by name from a BLSE or Bayesian inversion object."""
    for fault in get_faults_from_inversion(inversion):
        if getattr(fault, "name", None) == fault_name:
            return fault
    raise ValueError(f"Fault '{fault_name}' not found")


def _get_solution_vector(inversion: Any, solution: Optional[Sequence[float]] = None) -> np.ndarray:
    if solution is None:
        if not hasattr(inversion, "mpost") or inversion.mpost is None:
            raise ValueError(
                "No current linear solution was found. Run the inversion first, "
                "or call returnModel(..., print_stat=False) for Bayesian results, "
                "or pass solution explicitly."
            )
        solution = inversion.mpost
    return np.asarray(solution, dtype=float)


def _linear_solution_offset(inversion: Any, solution: np.ndarray) -> int:
    """Return the offset between full Bayesian sample positions and mpost indices."""
    offset = int(getattr(inversion, "linear_sample_start_position", 0) or 0)
    if offset <= 0:
        return 0
    try:
        n_linear = int(getattr(inversion, "lsq_parameters"))
    except Exception:
        n_linear = None
    if n_linear is not None and len(solution) == n_linear:
        return offset
    return 0


def _get_source_start(inversion: Any, fault_name: str, solution: np.ndarray) -> int:
    """Return source-parameter start index in the current linear solution."""
    if hasattr(inversion, "slip_positions") and fault_name in inversion.slip_positions:
        start = int(inversion.slip_positions[fault_name][0])
        return start - _linear_solution_offset(inversion, solution)

    for holder in (inversion, getattr(inversion, "multifaults", None)):
        if holder is not None and hasattr(holder, "fault_indexes") and fault_name in holder.fault_indexes:
            return int(holder.fault_indexes[fault_name][0])

    raise AttributeError(
        f"Cannot determine parameter start for fault '{fault_name}'. "
        "Expected slip_positions or fault_indexes."
    )


def _get_source_param_names(inversion: Any, fault: Any) -> Sequence[str]:
    adapters = getattr(inversion, "adapters", None)
    if adapters is None and hasattr(inversion, "multifaults"):
        adapters = getattr(inversion.multifaults, "adapters", None)

    if adapters and fault.name in adapters:
        adapter = adapters[fault.name]
        if getattr(adapter, "source_type", "Fault") != "Fault":
            raise ValueError(
                f"Interseismic fields are only defined for Fault sources; "
                f"'{fault.name}' is {adapter.source_type}."
            )
        return list(adapter.get_param_names())

    from .source_adapters import FaultAdapter
    slipdir = FaultAdapter._canonicalize_slipdir(getattr(fault, "slipdir", "sd"))
    mapping = {"s": "strikeslip", "d": "dipslip", "t": "tensile", "c": "coupling"}
    return [mapping[c] for c in slipdir if c in mapping]


def extract_inverted_slip(
    inversion: Any,
    fault_name: str,
    solution: Optional[Sequence[float]] = None,
    slip_component: str = "strikeslip",
) -> np.ndarray:
    """Extract one per-patch slip component from the current linear solution.

    Parameters
    ----------
    inversion : object
        ``BoundLSEMultiFaultsInversion`` or ``BayesianMultiFaultsInversion``-like
        object with ``mpost`` and source parameter positions.
    fault_name : str
        Name of the target fault.
    solution : sequence of float, optional
        Linear solution vector.  If omitted, ``inversion.mpost`` is used.
    slip_component : str, default ``"strikeslip"``
        ``"strikeslip"``, ``"dipslip"``, or ``"total"``.  Aliases such as
        ``"ss"``, ``"ds"`` and ``"magnitude"`` are accepted.

    Returns
    -------
    numpy.ndarray
        One value per patch.  The input fault object is not modified.
    """
    component = normalize_slip_component(slip_component)
    fault = get_fault_by_name(inversion, fault_name)
    sol = _get_solution_vector(inversion, solution)
    n_patches = len(fault.patch)
    source_start = _get_source_start(inversion, fault_name, sol)
    param_names = list(_get_source_param_names(inversion, fault))

    def _component_values(name: str) -> np.ndarray:
        if name not in param_names:
            raise ValueError(f"Fault '{fault_name}' has no '{name}' component in the current model")
        comp_index = param_names.index(name)
        start = source_start + comp_index * n_patches
        end = start + n_patches
        if start < 0 or end > len(sol):
            raise ValueError(
                f"Component '{name}' for fault '{fault_name}' maps to solution "
                f"indices [{start}, {end}), outside solution length {len(sol)}."
            )
        return sol[start:end].copy()

    if component in ("strikeslip", "dipslip"):
        return _component_values(component)

    ss = _component_values("strikeslip")
    ds = _component_values("dipslip")
    return np.sqrt(ss**2 + ds**2)


def resolve_patch_indices(
    fault: Any,
    patch_indices: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Return explicit patch indices, or all patches when omitted.

    Parameters
    ----------
    fault : object
        CSI fault object.
    patch_indices : iterable of int, optional
        Explicit patch subset.  ``None`` means all patches.

    Returns
    -------
    numpy.ndarray
        Integer patch indices.  Empty subsets are allowed only when explicitly
        supplied.
    """
    return normalize_patch_indices(
        fault,
        patch_indices,
        allow_none_all=True,
        name="patch_indices",
    )


def _get_transform_indices(inversion: Any, fault: Any) -> Mapping[str, Mapping[str, Tuple[int, int]]]:
    if hasattr(fault, "transform_indices") and fault.transform_indices:
        return fault.transform_indices
    faults = get_faults_from_inversion(inversion)
    if faults and hasattr(faults[0], "transform_indices"):
        return faults[0].transform_indices
    if hasattr(inversion, "transform_indices"):
        return inversion.transform_indices
    if hasattr(inversion, "multifaults") and hasattr(inversion.multifaults, "transform_indices"):
        return inversion.multifaults.transform_indices
    return {}


def resolve_euler_block_vectors(
    inversion: Any,
    fault_name: str,
    params: Mapping[str, Any],
    solution: Optional[Sequence[float]] = None,
    euler_params1: Optional[Sequence[float]] = None,
    euler_params2: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve the two Euler block vectors for one fault.

    Parameters
    ----------
    inversion : object
        Inversion object containing a current linear solution and transform
        indices when dataset-estimated Euler rotations are used.
    fault_name : str
        Target fault name.
    params : mapping
        Parsed per-fault Euler constraint configuration.  The public config
        convention for fixed poles is ``[lon, lat, omega]`` in the units
        declared by ``euler_pole_units``; parsed fixed poles/vectors are stored
        as Cartesian Euler vectors in radians/year before reaching this helper.
    solution : sequence of float, optional
        Linear solution vector.  Defaults to ``inversion.mpost``.
    euler_params1, euler_params2 : sequence of float, optional
        Explicit Cartesian Euler vectors ``[wx, wy, wz]`` in radians/year.
        Provided values override the corresponding configured block.

    Returns
    -------
    tuple of numpy.ndarray
        Two Cartesian Euler vectors in radians/year.
    """
    return _resolve_euler_block_vectors(
        inversion,
        fault_name,
        params,
        solution=solution,
        euler_params1=euler_params1,
        euler_params2=euler_params2,
    )


def calculate_tectonic_loading_rate(
    inversion: Any,
    fault_name: str,
    euler_params1: Optional[Sequence[float]] = None,
    euler_params2: Optional[Sequence[float]] = None,
    solution: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Calculate long-term strike-parallel tectonic loading rate.

    Parameters
    ----------
    inversion : object
        BLSE or Bayesian inversion object with parsed
        ``interseismic_config.fault_loading``.
    fault_name : str
        Target fault name.
    euler_params1, euler_params2 : sequence of float, optional
        Explicit Euler vectors ``[wx, wy, wz]`` in radians/year for the two
        blocks.  If omitted, vectors are read from the Euler config and current
        solution.
    solution : sequence of float, optional
        Linear solution vector.  Defaults to ``inversion.mpost``.
    Returns
    -------
    numpy.ndarray
        One loading-rate value per patch.
    """
    fault = get_fault_by_name(inversion, fault_name)
    patch_indices = resolve_patch_indices(fault)
    loading_rate = np.zeros(len(fault.patch), dtype=float)

    if euler_params1 is not None or euler_params2 is not None:
        params = get_fault_loading_params(inversion, fault_name)
        if params.get("loading_regions"):
            raise ValueError(
                "Explicit euler_params1/euler_params2 cannot be used with "
                "fault_loading.loading_regions because one pair of vectors "
                "cannot represent multiple regional block pairs. Update the "
                "interseismic_config instead or remove loading_regions."
            )
        vec1, vec2 = resolve_euler_block_vectors(
            inversion,
            fault_name,
            params,
            solution=solution,
            euler_params1=euler_params1,
            euler_params2=euler_params2,
        )
        from .interseismic_parameter_model import _project_loading_coefficients

        euler_strike = _project_loading_coefficients(inversion, fault_name, patch_indices, params)
        fixed_scale = m_per_year_to_observation_factor(inversion, default="m/yr")
        loading_rate[patch_indices] = fixed_scale * np.sum(euler_strike * (vec1 - vec2)[None, :], axis=1)
    else:
        loading_rate[patch_indices] = calculate_loading_from_terms(
            inversion,
            fault_name,
            patch_indices,
            solution=solution,
        )
    return loading_rate


def calculate_interseismic_fields(
    inversion: Any,
    fault_name: str,
    euler_params1: Optional[Sequence[float]] = None,
    euler_params2: Optional[Sequence[float]] = None,
    solution: Optional[Sequence[float]] = None,
    slip_component: str = "strikeslip",
) -> Dict[str, Any]:
    """Calculate the standard interseismic fields for one fault.

    The linear slip variable is interpreted as direct backslip ``q``.  With
    Euler/block loading ``b`` projected to the same strike direction:

    ``slip_deficit_signed = -q``, ``coupling_ratio = -q / b`` and
    ``creep_rate_signed = b + q``.

    Parameters
    ----------
    inversion : object
        BLSE or Bayesian inversion object.  Bayesian objects must already have a
        representative model loaded through ``returnModel()`` unless
        ``solution`` is supplied.
    fault_name : str
        Target fault name.
    euler_params1, euler_params2 : sequence of float, optional
        Explicit Euler vectors ``[wx, wy, wz]`` in radians/year.
    solution : sequence of float, optional
        Linear solution vector.  Defaults to ``inversion.mpost``.
    slip_component : str, default ``"strikeslip"``
        Slip component used in backslip, slip-deficit, coupling, and creep
        calculations.
    Returns
    -------
    dict
        Result dictionary with ``fields``, ``stats`` and ``metadata`` sections.
        All arrays have one value per patch and the input fault is not modified.
    """
    component = normalize_slip_component(slip_component)
    fault = get_fault_by_name(inversion, fault_name)
    params = get_fault_loading_params(inversion, fault_name)

    sol = _get_solution_vector(inversion, solution)
    loading = calculate_tectonic_loading_rate(
        inversion,
        fault_name,
        euler_params1=euler_params1,
        euler_params2=euler_params2,
        solution=sol,
    )
    backslip = extract_inverted_slip(
        inversion,
        fault_name,
        solution=sol,
        slip_component=component,
    )
    slip_deficit_signed = -backslip
    slip_deficit_magnitude = np.abs(backslip)
    coupling_ratio = _safe_ratio(slip_deficit_signed, loading)
    coupling_magnitude = _safe_ratio(slip_deficit_magnitude, np.abs(loading))
    creep_rate_signed = loading + backslip
    creep_ratio = _safe_ratio(np.abs(creep_rate_signed), np.abs(loading))

    inverted_key = f"inverted_{component}"
    backslip_key = f"backslip_{component}"
    fields = {
        "block_slip_rate_signed": loading,
        "tectonic_loading_rate": loading,
        "backslip_rate": backslip,
        backslip_key: backslip,
        "inverted_slip": backslip,
        inverted_key: backslip,
        "slip_deficit_signed": slip_deficit_signed,
        "slip_deficit_magnitude": slip_deficit_magnitude,
        "coupling_ratio": coupling_ratio,
        "coupling_magnitude": coupling_magnitude,
        "creep_rate_signed": creep_rate_signed,
        "creep_ratio": creep_ratio,
    }
    stats = {name: summarize_values(values) for name, values in fields.items()}
    unit_info = get_observation_unit_info(inversion, default="m/yr")

    return {
        "fault_name": fault_name,
        "fields": fields,
        "stats": stats,
        "metadata": {
            "slip_component": component,
            "unit": unit_info["observation"],
            "unit_assumed": bool(unit_info["assumed"]),
            "n_patches": len(fault.patch),
            "fault_loading": {
                "block_types": list(params.get("block_types", [])),
                "blocks": list(params.get("blocks", params.get("blocks_standard", []))),
                "block_names": list(params.get("block_names", [])),
                "loading_regions": [
                    {
                        "name": region.get("name"),
                        "block_types": list(region.get("block_types", [])),
                        "blocks": list(region.get("blocks", region.get("blocks_standard", []))),
                        "block_names": list(region.get("block_names", [])),
                        "reference_strike": region.get("reference_strike"),
                        "motion_sense": region.get("motion_sense"),
                    }
                    for region in params.get("loading_regions", []) or []
                ],
            },
            "reference_strike": params.get("reference_strike"),
            "motion_sense": params.get("motion_sense"),
            "field_convention": "direct_backslip",
            "formulas": {
                "block_slip_rate_signed": "b",
                "backslip_rate": "q",
                "slip_deficit_signed": "-q",
                "coupling_ratio": "-q / b",
                "creep_rate_signed": "b + q",
            },
            "compatibility_aliases": {
                "inverted_slip": "backslip_rate",
            },
        },
    }


def get_interseismic_field_values(result: Mapping[str, Any], field: str) -> np.ndarray:
    """Return one field array from a result dictionary using aliases."""
    canonical = normalize_interseismic_field(field)
    fields = result.get("fields", {})
    if canonical == "inverted_slip" and canonical not in fields:
        for key in ("inverted_strikeslip", "inverted_dipslip", "inverted_total"):
            if key in fields:
                canonical = key
                break
    if canonical not in fields:
        available = ", ".join(sorted(fields))
        raise ValueError(f"Field '{field}' is not available. Available fields: {available}")
    return np.asarray(fields[canonical], dtype=float)
