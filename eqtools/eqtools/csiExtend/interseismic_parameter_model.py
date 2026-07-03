"""Interseismic block and fault-loading parameter model helpers.

This module keeps the physical block-motion definition separate from optional
backslip/coupling constraints.  The same resolver is used by field
calculation, Euler-cap inequalities, and hard backslip equalities so that a
constraint selector cannot accidentally redefine the tectonic loading rate.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config.config_utils import (
    m_per_year_to_observation_factor,
    observation_to_m_per_year_factor,
)


def get_faults_from_inversion(inversion: Any) -> Sequence[Any]:
    """Return source/fault objects from a BLSE, VCE, or Bayesian inversion."""
    if hasattr(inversion, "_get_faults"):
        return inversion._get_faults()
    if hasattr(inversion, "multifaults") and hasattr(inversion.multifaults, "faults"):
        return inversion.multifaults.faults
    if hasattr(inversion, "faults"):
        return inversion.faults
    raise AttributeError("Cannot find fault list on inversion object")


def get_fault_by_name(inversion: Any, fault_name: str) -> Any:
    """Return one source/fault object by name."""
    for fault in get_faults_from_inversion(inversion):
        if getattr(fault, "name", None) == fault_name:
            return fault
    raise ValueError(f"Fault/source '{fault_name}' not found")


def get_interseismic_config(inversion: Any) -> Mapping[str, Any]:
    """Return the parsed interseismic configuration from an inversion object."""
    config = getattr(getattr(inversion, "config", None), "interseismic_config", None)
    if config:
        return config
    return getattr(inversion, "interseismic_config", {}) or {}


def get_fault_loading_config(interseismic_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the active fault-loading section."""
    return interseismic_config.get("fault_loading", {})


def get_fault_loading_params(inversion: Any, fault_name: str) -> Mapping[str, Any]:
    """Return parsed loading parameters for one fault."""
    loading = get_fault_loading_config(get_interseismic_config(inversion))
    if not loading.get("enabled", False):
        raise ValueError("Interseismic fault_loading is not enabled in configuration")
    if fault_name not in loading.get("faults", {}):
        raise ValueError(f"Fault '{fault_name}' not found in interseismic fault_loading")
    return loading["faults"][fault_name]


def get_blocks_config(interseismic_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the optional named-block registry."""
    return interseismic_config.get("blocks", {}) or {}


def _get_solution_vector(inversion: Any, solution: Optional[Sequence[float]] = None) -> np.ndarray:
    if solution is None:
        if not hasattr(inversion, "mpost") or inversion.mpost is None:
            raise ValueError(
                "No current linear solution was found. Run the inversion first, "
                "or pass solution explicitly."
            )
        solution = inversion.mpost
    return np.asarray(solution, dtype=float)


def _linear_parameter_offset(
    inversion: Any,
    solution: Optional[Sequence[float]] = None,
    n_total: Optional[int] = None,
) -> int:
    """Return offset from full Bayesian sample indices to the linear block."""
    offset = int(getattr(inversion, "linear_sample_start_position", 0) or 0)
    if offset <= 0:
        return 0

    n_linear = None
    for attr in ("lsq_parameters", "mcmc_samples"):
        if hasattr(inversion, attr):
            try:
                value = int(getattr(inversion, attr))
            except Exception:
                continue
            if attr == "mcmc_samples":
                value -= offset
            n_linear = value
            break

    if n_total is not None and n_linear is not None and int(n_total) == n_linear:
        return offset
    if solution is not None and n_linear is not None and len(solution) == n_linear:
        return offset
    return 0


def get_source_start(
    inversion: Any,
    source_name: str,
    solution: Optional[Sequence[float]] = None,
    n_total: Optional[int] = None,
) -> int:
    """Return a source start index in the active linear parameter vector."""
    offset = _linear_parameter_offset(inversion, solution=solution, n_total=n_total)
    if hasattr(inversion, "slip_positions") and source_name in inversion.slip_positions:
        return int(inversion.slip_positions[source_name][0]) - offset

    holders = (inversion, getattr(inversion, "multifaults", None))
    for holder in holders:
        if holder is not None and hasattr(holder, "fault_indexes") and source_name in holder.fault_indexes:
            return int(holder.fault_indexes[source_name][0]) - offset

    raise AttributeError(
        f"Cannot determine parameter start for source '{source_name}'. "
        "Expected slip_positions or fault_indexes."
    )


def _get_transform_indices_from_source(source: Any) -> Mapping[str, Mapping[str, Tuple[int, int]]]:
    indices = getattr(source, "transform_indices", None)
    return indices or {}


def _first_source_name(inversion: Any) -> Optional[str]:
    faults = get_faults_from_inversion(inversion)
    if not faults:
        return None
    return getattr(faults[0], "name", None)


def find_transform_indices(
    inversion: Any,
    dataset_name: str,
    transform_name: str = "eulerrotation",
    owner_source: Optional[str] = None,
) -> Tuple[Mapping[str, Mapping[str, Tuple[int, int]]], str]:
    """Find transform-index metadata and the source that owns the columns."""
    if owner_source:
        source = get_fault_by_name(inversion, owner_source)
        indices = _get_transform_indices_from_source(source)
        if dataset_name in indices and transform_name in indices[dataset_name]:
            return indices, owner_source
        raise ValueError(
            f"No {transform_name} transform for dataset '{dataset_name}' on owner_source '{owner_source}'"
        )

    for source in get_faults_from_inversion(inversion):
        source_name = getattr(source, "name", None)
        indices = _get_transform_indices_from_source(source)
        if dataset_name in indices and transform_name in indices[dataset_name]:
            return indices, source_name

    for holder in (inversion, getattr(inversion, "multifaults", None)):
        indices = getattr(holder, "transform_indices", None)
        if indices and dataset_name in indices and transform_name in indices[dataset_name]:
            source_name = _first_source_name(inversion)
            if source_name is None:
                raise ValueError(f"Cannot infer owner source for dataset '{dataset_name}' transform")
            return indices, source_name

    raise ValueError(f"No {transform_name} transform found for dataset '{dataset_name}'")


def resolve_transform_columns(
    inversion: Any,
    dataset_name: str,
    transform_name: str = "eulerrotation",
    owner_source: Optional[str] = None,
    solution: Optional[Sequence[float]] = None,
    n_total: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    """Return global/linear parameter columns for a dataset transform."""
    indices, source_name = find_transform_indices(
        inversion,
        dataset_name,
        transform_name=transform_name,
        owner_source=owner_source,
    )
    local_indices = indices[dataset_name].get(transform_name)
    if local_indices is None:
        raise ValueError(f"No {transform_name} transform found for dataset '{dataset_name}'")
    local_start, local_end = map(int, local_indices)
    if local_end <= local_start:
        raise ValueError(f"Invalid {transform_name} index range for dataset '{dataset_name}'")
    source_start = get_source_start(inversion, source_name, solution=solution, n_total=n_total)
    columns = np.arange(source_start + local_start, source_start + local_end, dtype=int)
    if n_total is not None and (columns[0] < 0 or columns[-1] >= int(n_total)):
        raise ValueError(
            f"{transform_name} parameters for dataset '{dataset_name}' map to "
            f"indices [{columns[0]}, {columns[-1] + 1}), outside {n_total}."
        )
    return columns, source_name


def _block_euler_mode(block: Mapping[str, Any]) -> str:
    euler_config = block.get("euler", {})
    mode = euler_config.get("mode")
    if mode:
        return str(mode)
    source = euler_config.get("source")
    if source == "dataset":
        return "estimate"
    if source == "euler_pole":
        return "fixed_pole"
    if source == "euler_vector":
        return "fixed_vector"
    raise ValueError(f"Unsupported block Euler source '{source}'")


def _block_datasets(block: Mapping[str, Any]) -> list[str]:
    euler_config = block.get("euler", {})
    datasets = block.get("datasets", euler_config.get("datasets", []))
    return [str(name) for name in list(datasets or [])]


def _block_anchor_dataset(block: Mapping[str, Any]) -> str:
    euler_config = block.get("euler", {})
    datasets = _block_datasets(block)
    anchor = euler_config.get("anchor_dataset")
    if anchor:
        return str(anchor)
    if datasets:
        return str(datasets[0])
    raise ValueError("Estimated Euler block must define at least one dataset")


def _block_fixed_euler_vector(block: Mapping[str, Any]) -> np.ndarray:
    euler_config = block.get("euler", {})
    vector = euler_config.get("vector_radians_per_year", euler_config.get("value_standard"))
    vector = np.asarray(vector, dtype=float)
    if vector.size != 3:
        raise ValueError("Fixed block Euler vector must contain exactly three values")
    return vector


def _fixed_euler_vector(block_type: str, block_data: Sequence[float]) -> np.ndarray:
    if block_type in {"euler_pole", "euler_vector"}:
        vector = np.asarray(block_data, dtype=float)
        if vector.size != 3:
            raise ValueError(f"Fixed Euler block type '{block_type}' must contain three values")
        return vector
    raise ValueError(f"Unsupported fixed Euler block type '{block_type}'")


def _physical_euler_to_model_units(inversion: Any, vector: Sequence[float]) -> np.ndarray:
    """Convert physical rad/yr Euler vector to the active matrix units."""
    factor = m_per_year_to_observation_factor(inversion, default="m/yr")
    return np.asarray(vector, dtype=float) * factor


def _model_euler_to_physical_units(inversion: Any, vector: Sequence[float]) -> np.ndarray:
    """Convert solved eulerrotation coefficients to physical rad/yr."""
    factor = observation_to_m_per_year_factor(inversion, default="m/yr")
    return np.asarray(vector, dtype=float) * factor


def resolve_euler_block_vectors(
    inversion: Any,
    fault_name: str,
    params: Mapping[str, Any],
    solution: Optional[Sequence[float]] = None,
    euler_params1: Optional[Sequence[float]] = None,
    euler_params2: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve the two Cartesian Euler vectors used by one fault loading model."""
    sol = _get_solution_vector(inversion, solution)
    interseismic_config = get_interseismic_config(inversion)
    blocks_config = get_blocks_config(interseismic_config)
    block_types = params.get("block_types", [])
    blocks_standard = params.get("blocks_standard", params.get("blocks", []))

    resolved = []
    for block_type, block_data in zip(block_types, blocks_standard):
        if block_type == "block":
            block = blocks_config.get("items", {}).get(block_data)
            if block is None:
                raise ValueError(f"Fault '{fault_name}' references undefined block '{block_data}'")
            mode = _block_euler_mode(block)
            if mode == "estimate":
                euler_config = block.get("euler", {})
                dataset_name = _block_anchor_dataset(block)
                columns, _ = resolve_transform_columns(
                    inversion,
                    dataset_name,
                    owner_source=euler_config.get("owner_source"),
                    solution=sol,
                    n_total=len(sol),
                )
                if columns.size != 3:
                    raise ValueError(f"Expected 3 Euler parameters for dataset '{dataset_name}'")
                resolved.append(_model_euler_to_physical_units(inversion, sol[columns]))
            else:
                resolved.append(_block_fixed_euler_vector(block))
        elif block_type == "dataset":
            columns, _ = resolve_transform_columns(
                inversion,
                block_data,
                owner_source=params.get("owner_source"),
                solution=sol,
                n_total=len(sol),
            )
            if columns.size != 3:
                raise ValueError(f"Expected 3 Euler parameters for dataset '{block_data}'")
            resolved.append(_model_euler_to_physical_units(inversion, sol[columns]))
        elif block_type in {"euler_pole", "euler_vector"}:
            resolved.append(_fixed_euler_vector(block_type, block_data))
        else:
            raise ValueError(f"Unknown Euler block_type: {block_type}")

    if len(resolved) != 2:
        raise ValueError(f"Fault '{fault_name}' must define exactly two Euler blocks")
    if euler_params1 is not None:
        resolved[0] = np.asarray(euler_params1, dtype=float)
    if euler_params2 is not None:
        resolved[1] = np.asarray(euler_params2, dtype=float)
    return resolved[0], resolved[1]


def _project_loading_coefficients(inversion: Any, fault_name: str, patch_indices: np.ndarray, params) -> np.ndarray:
    from .euler_inequality_constraints import calculate_euler_matrix_for_points, project_euler_to_strike

    fault = get_fault_by_name(inversion, fault_name)
    centers = np.asarray(fault.getcenters(), dtype=float)[patch_indices]
    lonc, latc = fault.xy2ll(centers[:, 0], centers[:, 1])
    euler_mat = calculate_euler_matrix_for_points(np.radians(lonc), np.radians(latc))
    return project_euler_to_strike(
        euler_mat,
        fault,
        patch_indices.tolist(),
        params.get("reference_strike", 0.0),
        len(patch_indices),
    )


def build_loading_linear_terms(
    inversion: Any,
    fault_name: str,
    patch_indices: Iterable[int],
    n_total: int,
    interseismic_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``A_loading`` and fixed terms for ``b = block1 - block2``.

    The returned arrays satisfy ``b = A_loading @ x + fixed_loading`` for the
    active linear parameter vector ``x``.
    """
    if interseismic_config is None:
        params = get_fault_loading_params(inversion, fault_name)
        interseismic_config = get_interseismic_config(inversion)
    else:
        loading = get_fault_loading_config(interseismic_config)
        if not loading.get("enabled", False) or fault_name not in loading.get("faults", {}):
            raise ValueError(f"Interseismic fault_loading is not enabled for fault '{fault_name}'.")
        params = loading["faults"][fault_name]
    patch_indices = np.asarray(list(patch_indices), dtype=int)
    A_loading = np.zeros((len(patch_indices), int(n_total)), dtype=float)
    fixed_loading = np.zeros(len(patch_indices), dtype=float)

    regions = list(params.get("loading_regions", []) or [])
    if regions:
        _add_regioned_loading_terms(
            inversion,
            fault_name,
            params,
            regions,
            patch_indices,
            n_total,
            interseismic_config,
            A_loading,
            fixed_loading,
        )
        return A_loading, fixed_loading

    _add_loading_terms_for_params(
        inversion,
        fault_name,
        params,
        patch_indices,
        n_total,
        interseismic_config,
        A_loading,
        fixed_loading,
    )
    return A_loading, fixed_loading


def _add_regioned_loading_terms(
    inversion: Any,
    fault_name: str,
    default_params: Mapping[str, Any],
    regions: Sequence[Mapping[str, Any]],
    patch_indices: np.ndarray,
    n_total: int,
    interseismic_config: Mapping[str, Any],
    A_loading: np.ndarray,
    fixed_loading: np.ndarray,
) -> None:
    """Fill loading terms using optional manual region overrides."""
    from .patch_indices import select_patch_indices

    fault = get_fault_by_name(inversion, fault_name)
    assigned = np.zeros(len(patch_indices), dtype=bool)
    assigned_by_patch: Dict[int, str] = {}

    for region_index, region in enumerate(regions):
        region_name = str(region.get("name", f"region_{region_index}"))
        selected = select_patch_indices(
            fault,
            region.get("selector"),
            allow_none_all=False,
            unique=True,
            name=f"loading region '{region_name}' selector for fault '{fault_name}'",
        )
        selected_set = {int(idx) for idx in selected.tolist()}
        row_mask = np.asarray([int(idx) in selected_set for idx in patch_indices], dtype=bool)
        if not np.any(row_mask):
            continue

        overlapping_rows = np.nonzero(assigned & row_mask)[0]
        if overlapping_rows.size:
            overlap_patches = [int(patch_indices[row]) for row in overlapping_rows.tolist()]
            previous = sorted({assigned_by_patch.get(idx, "unknown") for idx in overlap_patches})
            raise ValueError(
                f"fault_loading.faults.{fault_name}.loading_regions overlap on patches "
                f"{overlap_patches}; region '{region_name}' overlaps with {previous}."
            )

        row_indices = np.nonzero(row_mask)[0]
        _add_loading_terms_for_params(
            inversion,
            fault_name,
            region,
            patch_indices[row_indices],
            n_total,
            interseismic_config,
            A_loading,
            fixed_loading,
            row_indices=row_indices,
        )
        assigned[row_indices] = True
        for patch_idx in patch_indices[row_indices].tolist():
            assigned_by_patch[int(patch_idx)] = region_name

    default_rows = np.nonzero(~assigned)[0]
    if default_rows.size:
        _add_loading_terms_for_params(
            inversion,
            fault_name,
            default_params,
            patch_indices[default_rows],
            n_total,
            interseismic_config,
            A_loading,
            fixed_loading,
            row_indices=default_rows,
        )


def _add_loading_terms_for_params(
    inversion: Any,
    fault_name: str,
    params: Mapping[str, Any],
    patch_indices: np.ndarray,
    n_total: int,
    interseismic_config: Mapping[str, Any],
    A_loading: np.ndarray,
    fixed_loading: np.ndarray,
    *,
    row_indices: Optional[np.ndarray] = None,
) -> None:
    """Add loading terms for one fault-level or region-level block pair."""
    if patch_indices.size == 0:
        return
    if row_indices is None:
        row_indices = np.arange(len(patch_indices), dtype=int)
    else:
        row_indices = np.asarray(row_indices, dtype=int)
    if row_indices.size != patch_indices.size:
        raise ValueError("row_indices and patch_indices must have the same length")

    euler_strike = _project_loading_coefficients(inversion, fault_name, patch_indices, params)
    fixed_loading_scale = m_per_year_to_observation_factor(inversion, default="m/yr")
    blocks_config = get_blocks_config(interseismic_config)

    block_types = params.get("block_types", [])
    blocks_standard = params.get("blocks_standard", params.get("blocks", []))
    if len(block_types) != 2 or len(blocks_standard) != 2:
        raise ValueError(f"Fault '{fault_name}' loading definition must define exactly two Euler blocks")

    local_A = np.zeros((len(patch_indices), int(n_total)), dtype=float)
    local_fixed = np.zeros(len(patch_indices), dtype=float)
    for block_idx, (block_type, block_data) in enumerate(zip(block_types, blocks_standard)):
        block_sign = 1.0 if block_idx == 0 else -1.0
        if block_type == "block":
            block = blocks_config.get("items", {}).get(block_data)
            if block is None:
                raise ValueError(f"Fault '{fault_name}' references undefined block '{block_data}'")
            _add_block_loading_terms(
                inversion,
                block,
                block_sign,
                euler_strike,
                local_A,
                local_fixed,
                n_total,
                fixed_loading_scale=fixed_loading_scale,
            )
        elif block_type == "dataset":
            _add_dataset_loading_terms(
                inversion,
                block_data,
                params.get("owner_source"),
                block_sign,
                euler_strike,
                local_A,
                n_total,
            )
        elif block_type in {"euler_pole", "euler_vector"}:
            euler_vector = _fixed_euler_vector(block_type, block_data)
            fixed_m_per_year = np.sum(euler_strike * euler_vector[None, :], axis=1)
            local_fixed += block_sign * fixed_loading_scale * fixed_m_per_year
        else:
            raise ValueError(f"Unknown Euler block_type: {block_type}")

    A_loading[row_indices, :] += local_A
    fixed_loading[row_indices] += local_fixed


def _add_block_loading_terms(
    inversion: Any,
    block: Mapping[str, Any],
    block_sign: float,
    euler_strike: np.ndarray,
    A_loading: np.ndarray,
    fixed_loading: np.ndarray,
    n_total: int,
    *,
    fixed_loading_scale: float,
) -> None:
    mode = _block_euler_mode(block)
    euler_config = block.get("euler", {})
    if mode == "estimate":
        dataset_name = _block_anchor_dataset(block)
        _add_dataset_loading_terms(
            inversion,
            dataset_name,
            euler_config.get("owner_source"),
            block_sign,
            euler_strike,
            A_loading,
            n_total,
        )
    elif mode in {"fixed_pole", "fixed_vector"}:
        euler_vector = _block_fixed_euler_vector(block)
        fixed_m_per_year = np.sum(euler_strike * euler_vector[None, :], axis=1)
        fixed_loading += block_sign * fixed_loading_scale * fixed_m_per_year
    else:
        raise ValueError(f"Unsupported block Euler mode '{mode}'")


def _add_dataset_loading_terms(
    inversion: Any,
    dataset_name: str,
    owner_source: Optional[str],
    block_sign: float,
    euler_strike: np.ndarray,
    A_loading: np.ndarray,
    n_total: int,
) -> None:
    columns, _ = resolve_transform_columns(
        inversion,
        dataset_name,
        owner_source=owner_source,
        n_total=n_total,
    )
    if columns.size != 3:
        raise ValueError(f"Expected 3 Euler parameters for dataset '{dataset_name}'")
    for k, column in enumerate(columns):
        A_loading[:, column] += block_sign * euler_strike[:, k]


def calculate_loading_from_terms(
    inversion: Any,
    fault_name: str,
    patch_indices: Iterable[int],
    solution: Optional[Sequence[float]] = None,
    interseismic_config: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Calculate loading for selected patches through the shared linear model."""
    sol = _get_solution_vector(inversion, solution)
    A_loading, fixed_loading = build_loading_linear_terms(
        inversion,
        fault_name,
        patch_indices,
        n_total=len(sol),
        interseismic_config=interseismic_config,
    )
    return A_loading @ sol + fixed_loading


def generate_block_euler_equality_constraints(
    inversion: Any,
    interseismic_config: Mapping[str, Any],
    n_total: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Generate equality rows tying dataset Euler vectors to block Euler.

    For ``mode: estimate``, every dataset in a block shares one estimated
    ``eulerrotation`` vector.  For ``mode: fixed_pole`` or ``mode:
    fixed_vector``, any listed dataset that has an ``eulerrotation`` transform
    is fixed to the block Euler vector.
    """
    loading = get_fault_loading_config(interseismic_config)
    blocks = get_blocks_config(interseismic_config)
    if not loading.get("enabled", False) or not blocks.get("enabled", False):
        return None, None

    rows = []
    rhs = []
    for block_name, block in blocks.get("items", {}).items():
        datasets = _block_datasets(block)
        if not datasets:
            continue
        euler_config = block.get("euler", {})
        mode = _block_euler_mode(block)
        owner_source = euler_config.get("owner_source")

        if mode == "estimate":
            if len(datasets) <= 1:
                continue
            anchor_dataset = _block_anchor_dataset(block)
            ref_columns, _ = resolve_transform_columns(
                inversion,
                anchor_dataset,
                owner_source=owner_source,
                n_total=n_total,
            )
            if ref_columns.size != 3:
                raise ValueError(f"Expected 3 Euler parameters for block '{block_name}' anchor dataset")
            for dataset in datasets:
                if dataset == anchor_dataset:
                    continue
                columns, _ = resolve_transform_columns(
                    inversion,
                    dataset,
                    owner_source=owner_source,
                    n_total=n_total,
                )
                if columns.size != 3:
                    raise ValueError(f"Expected 3 Euler parameters for block '{block_name}' dataset '{dataset}'")
                for local_idx in range(3):
                    row = np.zeros(int(n_total), dtype=float)
                    row[columns[local_idx]] = 1.0
                    row[ref_columns[local_idx]] = -1.0
                    rows.append(row)
                    rhs.append(0.0)
        elif mode in {"fixed_pole", "fixed_vector"}:
            fixed_vector = _physical_euler_to_model_units(inversion, _block_fixed_euler_vector(block))
            for dataset in datasets:
                try:
                    columns, _ = resolve_transform_columns(
                        inversion,
                        dataset,
                        owner_source=owner_source,
                        n_total=n_total,
                    )
                except ValueError:
                    # A fixed block may list datasets whose rigid Euler motion
                    # has already been removed before inversion.  Preflight
                    # diagnostics report that case; constraint generation can
                    # safely skip missing eulerrotation columns.
                    continue
                if columns.size != 3:
                    raise ValueError(f"Expected 3 Euler parameters for block '{block_name}' dataset '{dataset}'")
                for local_idx in range(3):
                    row = np.zeros(int(n_total), dtype=float)
                    row[columns[local_idx]] = 1.0
                    rows.append(row)
                    rhs.append(float(fixed_vector[local_idx]))
        else:
            raise ValueError(f"Unsupported block Euler mode '{mode}'")

    if not rows:
        return None, None
    return np.vstack(rows), np.asarray(rhs, dtype=float)
