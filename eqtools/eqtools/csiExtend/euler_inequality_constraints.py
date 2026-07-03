"""Euler/fault-loading projection and optional interseismic cap constraints."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .interseismic_parameter_model import build_loading_linear_terms, get_fault_loading_config
from .patch_indices import select_patch_indices


CAP_LOADING_TERMS_TOL = 1.0e-12
CAP_BOUND_TOL = 1.0e-12

HARD_BACKSLIP_STATE_ALIASES = {
    "free": "free",
    "creep": "zero_backslip",
    "zero": "zero_backslip",
    "zero_slip": "zero_backslip",
    "zero_backslip": "zero_backslip",
    "full": "full_coupling",
    "locked": "full_coupling",
    "full_locking": "full_coupling",
    "full_coupling": "full_coupling",
    "partial_coupling": "prescribed_coupling",
    "prescribed_coupling": "prescribed_coupling",
    "fixed_coupling": "prescribed_coupling",
    "prescribed_backslip": "prescribed_backslip",
    "fixed_backslip": "prescribed_backslip",
    "backslip": "prescribed_backslip",
}
HARD_BACKSLIP_STATES = {
    "zero_backslip",
    "full_coupling",
    "prescribed_coupling",
    "prescribed_backslip",
}
CAP_HARD_OVERLAP_ALIASES = {
    "skip": "skip",
    "exclude": "skip",
    "auto": "skip",
    "keep": "keep",
    "allow": "keep",
    "error": "error",
    "raise": "error",
}


def calculate_euler_matrix_for_points(lonc: np.ndarray, latc: np.ndarray) -> np.ndarray:
    """Return the matrix mapping Cartesian Euler vectors to EN velocities.

    Parameters
    ----------
    lonc, latc : numpy.ndarray
        Point longitudes and latitudes in radians.

    Returns
    -------
    numpy.ndarray
        Matrix with shape ``(2 * n_points, 3)``.  The first ``n_points`` rows
        are east-velocity coefficients, and the remaining rows are north.
    """
    num_points = len(lonc)
    euler_mat = np.zeros((2 * num_points, 3), dtype=float)
    radius = 6378137.0

    cos_lat = np.cos(latc)
    sin_lat = np.sin(latc)
    cos_lon = np.cos(lonc)
    sin_lon = np.sin(lonc)

    euler_mat[:num_points, 0] = -radius * sin_lat * cos_lon
    euler_mat[:num_points, 1] = -radius * sin_lat * sin_lon
    euler_mat[:num_points, 2] = radius * cos_lat
    euler_mat[num_points:, 0] = radius * sin_lon
    euler_mat[num_points:, 1] = -radius * cos_lon
    euler_mat[num_points:, 2] = 0.0
    return euler_mat


def calculate_reference_strike_vector(reference_strike_deg: float, num_patches: int) -> np.ndarray:
    """Return EN unit vectors for a reference strike angle."""
    reference_strike_rad = np.deg2rad(reference_strike_deg)
    vec_reference = np.zeros((num_patches, 2), dtype=float)
    vec_reference[:, 0] = np.sin(reference_strike_rad)
    vec_reference[:, 1] = np.cos(reference_strike_rad)
    return vec_reference


def convert_euler_pole_to_vector(lon_pole: float, lat_pole: float, omega: float) -> np.ndarray:
    """Convert Euler pole ``(lon, lat, omega)`` to Cartesian vector.

    Angles are radians and ``omega`` is radians/year.  The argument order
    matches the public ``fixed_pole`` convention used by interseismic configs.
    """
    return np.array([
        omega * np.cos(lat_pole) * np.cos(lon_pole),
        omega * np.cos(lat_pole) * np.sin(lon_pole),
        omega * np.sin(lat_pole),
    ], dtype=float)


def project_euler_to_strike(
    euler_mat: np.ndarray,
    fault: object,
    patch_indices: Sequence[int],
    reference_strike_deg: float,
    num_patches: int,
) -> np.ndarray:
    """Project EN Euler velocities to each selected patch strike direction.

    The reference strike only chooses the sign-consistent branch of each
    patch's own strike vector.  Projection is still done on the local patch
    strike.
    """
    vel_east = euler_mat[:num_patches, :]
    vel_north = euler_mat[num_patches:, :]

    all_strikes = np.asarray(fault.getStrikes(), dtype=float)
    patch_strikes_rad = all_strikes[np.asarray(patch_indices, dtype=int)]
    strike_east = np.sin(patch_strikes_rad)
    strike_north = np.cos(patch_strikes_rad)

    reference_strike_rad = np.deg2rad(reference_strike_deg)
    ref_east = np.sin(reference_strike_rad)
    ref_north = np.cos(reference_strike_rad)
    reverse_mask = strike_east * ref_east + strike_north * ref_north < 0
    strike_east[reverse_mask] = -strike_east[reverse_mask]
    strike_north[reverse_mask] = -strike_north[reverse_mask]

    return vel_east * strike_east[:, None] + vel_north * strike_north[:, None]


def determine_motion_sign(motion_sense: str) -> float:
    """Return ``+1`` for dextral/right-lateral and ``-1`` for sinistral/left-lateral."""
    key = str(motion_sense).lower()
    if key in {"dextral", "right_lateral", "right"}:
        return 1.0
    if key in {"sinistral", "left_lateral", "left"}:
        return -1.0
    raise ValueError(f"Invalid motion_sense: {motion_sense}")


def normalize_interseismic_backslip_state(state: Any) -> str:
    """Normalize public backslip/coupling state aliases.

    The normalized states are shared by cap generation and diagnostics so that
    the default cap filtering follows the same semantics as hard equality
    constraints.
    """
    key = str(state).lower().replace("-", "_").replace(" ", "_")
    try:
        return HARD_BACKSLIP_STATE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown interseismic backslip state '{state}'") from exc


def normalize_cap_hard_overlap_policy(policy: Any) -> str:
    """Normalize cap/hard-overlap policy names."""
    key = str(policy).lower().replace("-", "_").replace(" ", "_")
    try:
        return CAP_HARD_OVERLAP_ALIASES[key]
    except KeyError as exc:
        raise ValueError("cap_constraints hard_overlap must be 'skip', 'keep', or 'error'") from exc


def get_hard_interseismic_backslip_patch_indices(
    fault: Any,
    interseismic_config: Mapping[str, Any],
    fault_name: str,
    component: str = "strikeslip",
) -> np.ndarray:
    """Return patch ids already fixed by hard backslip/coupling equalities.

    ``state='free'`` is intentionally ignored because it does not add equality
    rows.  Component matching matters: a dip-slip equality does not make a
    strike-slip Euler cap redundant.
    """
    target_component = _normalize_cap_component(component)
    hard_indices: list[int] = []
    for index, spec in enumerate(interseismic_config.get("backslip_constraints", []) or []):
        if spec.get("fault") != fault_name:
            continue
        state = normalize_interseismic_backslip_state(spec.get("state"))
        if state not in HARD_BACKSLIP_STATES:
            continue
        spec_component = _normalize_cap_component(spec.get("component", "strikeslip"))
        if spec_component != target_component:
            continue
        selected = select_patch_indices(
            fault,
            spec.get("selector"),
            allow_none_all=True,
            unique=True,
            name=f"backslip selector {index} for fault '{fault_name}'",
        )
        hard_indices.extend(int(i) for i in selected.tolist())

    if not hard_indices:
        return np.asarray([], dtype=int)
    seen: set[int] = set()
    return np.asarray(
        [idx for idx in hard_indices if not (idx in seen or seen.add(idx))],
        dtype=int,
    )


def filter_cap_patch_indices_for_hard_constraints(
    fault: Any,
    interseismic_config: Mapping[str, Any],
    fault_name: str,
    patch_indices: Sequence[int],
    *,
    component: str = "strikeslip",
    hard_overlap: Any = "skip",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the cap/hard-overlap policy to selected cap patch ids.

    By default, cap inequalities are only generated for freely estimated
    patches.  Patches already fixed by hard equality constraints are skipped to
    avoid duplicate active rows such as ``q+b=0`` and ``q+b<=0`` on the same
    patch.
    """
    policy = normalize_cap_hard_overlap_policy(hard_overlap)
    configured = np.asarray(patch_indices, dtype=int)
    hard = get_hard_interseismic_backslip_patch_indices(
        fault,
        interseismic_config,
        fault_name,
        component=component,
    )
    hard_set = set(int(i) for i in hard.tolist())
    overlap = np.asarray([int(i) for i in configured.tolist() if int(i) in hard_set], dtype=int)

    if overlap.size and policy == "error":
        preview = ", ".join(str(int(i)) for i in overlap[:10])
        suffix = "" if overlap.size <= 10 else ", ..."
        raise ValueError(
            f"cap_constraints for fault '{fault_name}' overlap hard backslip/coupling "
            f"constraints on patch(es): {preview}{suffix}. "
            "Use hard_overlap: skip or exclude these patches from the cap selector."
        )

    if policy == "skip" and overlap.size:
        active = np.asarray([int(i) for i in configured.tolist() if int(i) not in hard_set], dtype=int)
        skipped = overlap
    else:
        active = configured
        skipped = np.asarray([], dtype=int)

    return active, {
        "policy": policy,
        "configured_patch_count": int(configured.size),
        "active_patch_count": int(active.size),
        "hard_overlap_patch_count": int(overlap.size),
        "skipped_hard_patch_count": int(skipped.size),
        "hard_patches": hard.tolist(),
        "overlap_patches": overlap.tolist(),
        "skipped_hard_patches": skipped.tolist(),
    }


def generate_euler_cap_constraints(multifault_solver: Any, interseismic_config: Mapping[str, Any], all_datasets):
    """Generate optional backslip/loading cap inequalities.

    Let ``q`` be ECAT's direct strike-slip backslip variable and ``b`` be the
    fault-loading rate projected to the same strike convention.  The default
    ``mode='motion_sense'`` cap is:

    - dextral/right-lateral: ``q + k*b <= 0``
    - sinistral/left-lateral: ``q + k*b >= 0``, implemented as ``-(q + k*b) <= 0``

    ``mode='loading_sign'`` is stricter and intended for fixed-loading tests:
    it enforces ``0 <= -q/b <= k`` from the actual projected loading sign on
    each selected patch.  Because that sign must be known before solving, this
    mode rejects loading terms that still depend on estimated Euler columns.

    ``k`` is ``cap_constraints.*.max_coupling`` and defaults to 1.0.  The
    legacy key ``factor`` is accepted by the config parser as an alias.

    Bounds such as ``q >= 0`` or ``q <= 0`` should be set separately in
    ``bounds_config.yml`` when the user wants the usual ``0 <= coupling <= 1``
    interval.
    """
    fault_loading = get_fault_loading_config(interseismic_config)
    cap_config = interseismic_config.get("cap_constraints", {})
    if not fault_loading.get("enabled", False) or not cap_config.get("enabled", False):
        return None, None

    faults = getattr(getattr(multifault_solver, "config", None), "faults_list", None)
    if faults is None:
        faults = getattr(multifault_solver, "faults", [])
    fault_name_to_obj = {fault.name: fault for fault in faults}

    constraint_info = []
    total_constraints = 0
    for fault_name, cap_params in cap_config.get("faults", {}).items():
        if fault_name not in fault_name_to_obj:
            continue
        if fault_name not in fault_loading.get("faults", {}):
            raise ValueError(f"cap_constraints for '{fault_name}' require matching fault_loading settings")

        fault = fault_name_to_obj[fault_name]
        motion_params = fault_loading["faults"][fault_name]
        patch_indices = select_patch_indices(
            fault,
            cap_params.get("selector"),
            allow_none_all=True,
            unique=True,
            name=f"cap_constraints selector for fault '{fault_name}'",
        )
        patch_indices, _ = filter_cap_patch_indices_for_hard_constraints(
            fault,
            interseismic_config,
            fault_name,
            patch_indices,
            component="strikeslip",
            hard_overlap=cap_params.get("hard_overlap", "skip"),
        )
        if patch_indices.size == 0:
            continue

        constraint_info.append({
            "fault_name": fault_name,
            "fault": fault,
            "params": motion_params,
            "fault_start": _get_fault_linear_start(multifault_solver, fault_name),
            "patch_indices": patch_indices,
            "mode": _normalize_cap_mode(cap_params.get("mode", "motion_sense")),
            "max_coupling": float(cap_params.get("max_coupling", 1.0)),
            "min_loading_abs": float(cap_params.get("min_loading_abs", 0.0)),
        })
        mode = constraint_info[-1]["mode"]
        total_constraints += 2 * len(patch_indices) if mode == "loading_sign" else len(patch_indices)

    if total_constraints == 0:
        return None, None

    total_params = int(getattr(multifault_solver, "lsq_parameters"))
    A_ineq = np.zeros((total_constraints, total_params), dtype=float)
    b_ineq = np.zeros(total_constraints, dtype=float)
    constraint_row = 0

    for info in constraint_info:
        patch_indices = np.asarray(info["patch_indices"], dtype=int)
        rows = np.arange(constraint_row, constraint_row + len(patch_indices))
        slip_indices = int(info["fault_start"]) + patch_indices
        mode = str(info.get("mode", "motion_sense"))
        max_coupling = float(info.get("max_coupling", 1.0))

        A_loading, fixed_loading = build_loading_linear_terms(
            multifault_solver,
            info["fault_name"],
            patch_indices,
            n_total=total_params,
            interseismic_config=interseismic_config,
        )

        if mode == "motion_sense":
            motion_signs = _motion_signs_for_patch_indices(info["fault"], info["params"], patch_indices)
            A_ineq[rows, slip_indices] = motion_signs
            A_ineq[rows, :] += motion_signs[:, None] * max_coupling * A_loading
            b_ineq[rows] -= motion_signs * max_coupling * fixed_loading
            constraint_row += len(patch_indices)
        elif mode == "loading_sign":
            _require_fixed_loading_for_loading_sign(
                info["fault_name"],
                patch_indices,
                A_loading,
                fixed_loading,
                float(info.get("min_loading_abs", 0.0)),
            )
            loading_signs = np.sign(fixed_loading).astype(float)
            _validate_loading_sign_bounds(
                multifault_solver,
                info["fault_name"],
                patch_indices,
                slip_indices,
                fixed_loading,
                max_coupling,
            )

            sign_rows = rows
            magnitude_rows = np.arange(rows[-1] + 1, rows[-1] + 1 + len(patch_indices))

            A_ineq[sign_rows, slip_indices] = loading_signs
            b_ineq[sign_rows] = 0.0
            A_ineq[magnitude_rows, slip_indices] = -loading_signs
            b_ineq[magnitude_rows] = max_coupling * loading_signs * fixed_loading
            constraint_row += 2 * len(patch_indices)
        else:
            raise ValueError(f"Unsupported cap constraint mode '{mode}'")

    return A_ineq, b_ineq


def apply_euler_cap_constraints(multifault_solver, interseismic_config, all_datasets, verbose=False):
    """Append configured cap constraints to ``multifault_solver.A_ueq/b_ueq``."""
    A_ineq, b_ineq = generate_euler_cap_constraints(multifault_solver, interseismic_config, all_datasets)
    if A_ineq is None or A_ineq.size == 0:
        if verbose:
            print("No interseismic Euler-cap constraints generated.")
        return
    if multifault_solver.A_ueq is None:
        multifault_solver.A_ueq = A_ineq
        multifault_solver.b_ueq = b_ineq
    else:
        multifault_solver.A_ueq = np.vstack([multifault_solver.A_ueq, A_ineq])
        multifault_solver.b_ueq = np.hstack([multifault_solver.b_ueq, b_ineq])
    if verbose:
        configured = interseismic_config.get("cap_constraints", {}).get("configured_faults", [])
        print(f"Applied {A_ineq.shape[0]} interseismic Euler-cap constraints: {configured}")


def validate_euler_cap_config(interseismic_config, faultnames, dataset_names):
    """Validate parsed fault-loading and cap settings."""
    fault_loading = get_fault_loading_config(interseismic_config)
    cap_config = interseismic_config.get("cap_constraints", {})
    if not fault_loading.get("enabled", False):
        return True

    for fault_name, params in fault_loading.get("faults", {}).items():
        if fault_name not in faultnames:
            raise ValueError(f"fault_loading references unknown fault '{fault_name}'")
        block_types = params.get("block_types", [])
        blocks = params.get("blocks_standard", [])
        if len(block_types) != 2 or len(blocks) != 2:
            raise ValueError(f"Fault '{fault_name}' must have exactly two fault-loading blocks")
        determine_motion_sign(params.get("motion_sense", "dextral"))
        for block_type, block_data in zip(block_types, blocks):
            if block_type == "dataset" and block_data not in dataset_names:
                raise ValueError(f"Dataset '{block_data}' for fault '{fault_name}' not found")

    if cap_config.get("enabled", False):
        for fault_name in cap_config.get("faults", {}):
            if fault_name not in fault_loading.get("faults", {}):
                raise ValueError(f"cap_constraints for '{fault_name}' require fault_loading settings")
    return True


def _get_transform_indices(multifault_solver, faults) -> Mapping[str, Mapping[str, tuple[int, int]]]:
    if faults and hasattr(faults[0], "transform_indices"):
        return faults[0].transform_indices
    if hasattr(multifault_solver, "transform_indices"):
        return multifault_solver.transform_indices
    if hasattr(multifault_solver, "multifaults") and hasattr(multifault_solver.multifaults, "transform_indices"):
        return multifault_solver.multifaults.transform_indices
    return {}


def _get_fault_linear_start(multifault_solver, fault_name: str) -> int:
    offset = int(getattr(multifault_solver, "linear_sample_start_position", 0) or 0)
    if hasattr(multifault_solver, "slip_positions") and fault_name in multifault_solver.slip_positions:
        return int(multifault_solver.slip_positions[fault_name][0]) - offset
    if hasattr(multifault_solver, "fault_indexes") and fault_name in multifault_solver.fault_indexes:
        return int(multifault_solver.fault_indexes[fault_name][0]) - offset
    if hasattr(multifault_solver, "multifaults"):
        holder = multifault_solver.multifaults
        if hasattr(holder, "fault_indexes") and fault_name in holder.fault_indexes:
            return int(holder.fault_indexes[fault_name][0]) - offset
    raise AttributeError(f"Cannot determine linear parameter start for fault '{fault_name}'")


def _normalize_cap_mode(mode: Any) -> str:
    key = str(mode).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "motion_sense": "motion_sense",
        "motion": "motion_sense",
        "fault_motion": "motion_sense",
        "loading_sign": "loading_sign",
        "projected_loading": "loading_sign",
    }
    normalized = aliases.get(key)
    if normalized not in {"motion_sense", "loading_sign"}:
        raise ValueError("cap_constraints mode must be 'motion_sense' or 'loading_sign'")
    return normalized


def _normalize_cap_component(component: Any) -> str:
    key = str(component).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "strikeslip": "strikeslip",
        "strike_slip": "strikeslip",
        "ss": "strikeslip",
        "dipslip": "dipslip",
        "dip_slip": "dipslip",
        "ds": "dipslip",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported interseismic backslip component '{component}'") from exc


def _motion_signs_for_patch_indices(fault: object, params: Mapping[str, Any], patch_indices: np.ndarray) -> np.ndarray:
    """Return cap signs for selected patches, honoring loading-region overrides."""
    patch_indices = np.asarray(patch_indices, dtype=int)
    signs = np.full(
        patch_indices.size,
        determine_motion_sign(params.get("motion_sense", "dextral")),
        dtype=float,
    )
    if not params.get("loading_regions"):
        return signs

    position_by_patch = {int(patch): pos for pos, patch in enumerate(patch_indices.tolist())}
    for region in params.get("loading_regions", []) or []:
        region_patches = select_patch_indices(
            fault,
            region.get("selector"),
            allow_none_all=False,
            unique=True,
            name=f"loading region '{region.get('name', 'region')}' selector",
        )
        region_sign = determine_motion_sign(region.get("motion_sense", params.get("motion_sense", "dextral")))
        for patch_idx in np.asarray(region_patches, dtype=int):
            pos = position_by_patch.get(int(patch_idx))
            if pos is not None:
                signs[pos] = region_sign
    return signs


def _require_fixed_loading_for_loading_sign(
    fault_name: str,
    patch_indices: np.ndarray,
    A_loading: np.ndarray,
    fixed_loading: np.ndarray,
    min_loading_abs: float,
) -> None:
    if np.any(np.abs(A_loading) > CAP_LOADING_TERMS_TOL):
        raise ValueError(
            f"cap_constraints mode 'loading_sign' for fault '{fault_name}' requires fixed fault_loading. "
            "Estimated block/dataset Euler terms make the projected loading sign unknown before solving; "
            "use mode 'motion_sense' for that case."
        )

    small = np.where(np.abs(fixed_loading) <= float(min_loading_abs))[0]
    if small.size:
        examples = ", ".join(str(int(patch_indices[i])) for i in small[:5])
        raise ValueError(
            f"cap_constraints mode 'loading_sign' for fault '{fault_name}' found loading "
            f"|b| <= min_loading_abs on selected patch(es): {examples}. "
            "The coupling ratio -q/b is undefined or unstable there."
        )


def _get_solver_bounds(multifault_solver: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    for holder in (multifault_solver, getattr(multifault_solver, "constraint_manager", None)):
        if holder is None:
            continue
        lb = getattr(holder, "lb", getattr(holder, "_lb", None))
        ub = getattr(holder, "ub", getattr(holder, "_ub", None))
        if lb is not None or ub is not None:
            lb_array = None if lb is None else np.asarray(lb, dtype=float)
            ub_array = None if ub is None else np.asarray(ub, dtype=float)
            return lb_array, ub_array
    return None, None


def _validate_loading_sign_bounds(
    multifault_solver: Any,
    fault_name: str,
    patch_indices: np.ndarray,
    slip_indices: np.ndarray,
    fixed_loading: np.ndarray,
    max_coupling: float,
) -> None:
    lb, ub = _get_solver_bounds(multifault_solver)
    if lb is None and ub is None:
        return

    conflicts = []
    for patch_idx, col, loading in zip(patch_indices, slip_indices, fixed_loading):
        allowed_a = 0.0
        allowed_b = -float(max_coupling) * float(loading)
        allowed_low = min(allowed_a, allowed_b)
        allowed_high = max(allowed_a, allowed_b)

        lower = -np.inf
        upper = np.inf
        if lb is not None and int(col) < lb.size:
            lower = float(lb[int(col)])
        if ub is not None and int(col) < ub.size:
            upper = float(ub[int(col)])
        if lower > allowed_high + CAP_BOUND_TOL or upper < allowed_low - CAP_BOUND_TOL:
            conflicts.append(
                f"patch {int(patch_idx)} column {int(col)}: bounds [{lower:g}, {upper:g}] "
                f"do not intersect cap interval [{allowed_low:g}, {allowed_high:g}]"
            )

    if conflicts:
        preview = "; ".join(conflicts[:5])
        raise ValueError(
            f"cap_constraints mode 'loading_sign' for fault '{fault_name}' conflicts with slip bounds: {preview}"
        )
