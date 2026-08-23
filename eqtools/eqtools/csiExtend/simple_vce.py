import numpy as np
from . import lsqlin
from .covariance_utils import prepare_block_covariance_metrics, whiten_data_blocks
from .quadratic_objective import (
    LeastSquaresBlock,
    assemble_quadratic_objective,
    assemble_residual_system,
)
from eqtools.csiExtend.config.parameter_groups import (
    normalize_group_vector,
    resolve_group_layout,
)


def _finite_block_array(values):
    """Return a finite float array, copying only when repair is required.

    Normal covariance whitening already produces finite arrays.  Reusing those
    arrays avoids retaining a second ``n_obs x n_params`` copy beside the
    prepared Gram block.  The historical ``nan_to_num`` behavior is preserved
    for exceptional non-finite inputs without mutating the caller's array.
    """
    values = np.asarray(values, dtype=float)
    if np.all(np.isfinite(values)):
        return values
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def simplified_vce(
    data_metrics,
    d, 
    G, 
    L, 
    bounds, 
    data_ranges=None,
    fault_ranges=None,
    smoothing_faults=None,
    sigma_mode='individual',
    sigma_groups=None,
    sigma_update=None,
    sigma_values=None,
    smooth_mode='single',
    smooth_groups=None,
    smooth_update=None,
    smooth_values=None,
    A_ueq=None,
    b_ueq=None, 
    Aeq=None, 
    beq=None,
    max_iter=20, 
    tol=1e-4, 
    verbose=False
):
    """
    Simplified Variance Component Estimation for geodetic inversions using lsqlin solver.
    
    Solves: minimize ||G*m - d||^2_Σd + Σ_i ||L_i*m||^2_Σα_i
    
    Subject to:
    - A_ueq*m <= b_ueq (inequality constraints)
    - Aeq*m = beq (equality constraints)  
    - lb <= m <= ub (bounds)
    
    Parameters:
    -----------
    data_metrics : mapping
        Prepared ``DataCovarianceMetric`` for each entry in ``data_ranges``.
    d : array (n_obs,)
        Observation vector
    G : array (n_obs, n_params)
        Green's function matrix
    L : array (n_reg, n_params)
        Complete smoothing/constraint matrix
    bounds : tuple (lb, ub)
        Parameter bounds (lower, upper)
    data_ranges : dict, optional
        Data ranges: {'dataset1': (start, end), 'dataset2': (start, end), ...}
        If None, assumes single dataset: {'data': (0, n_obs)}
    fault_ranges : dict, optional
        Fault parameter ranges: {'fault1': (start, end), 'fault2': (start, end), ...}
        Used to identify which L constraints belong to which fault
        If None, assumes single fault: {'fault': (0, n_params)}
    smoothing_faults : sequence of str, optional
        Sources that own Laplacian rows. ``fault_ranges`` still describes the
        complete model-column layout; this subset only controls alpha grouping.
        If None, every entry in ``fault_ranges`` is treated as smoothable for
        compatibility with direct low-level calls.
    sigma_mode : str
        - 'single': All datasets share one sigma
        - 'individual': Each dataset has its own sigma
        - 'grouped': Custom grouping via sigma_groups
        Defaults to ``'individual'``, matching the public BLSE/VCE
        configuration contract.
    sigma_groups : dict, optional
        For 'grouped' mode: {'group1': ['dataset1', 'dataset2'], 'group2': ['dataset3']}
    smooth_mode : str
        - 'single': All faults share one alpha
        - 'individual': Each fault has its own alpha
        - 'grouped': Custom grouping via smooth_groups
    smooth_groups : dict, optional
        For 'grouped' mode: {'group1': ['fault1', 'fault2'], 'group2': ['fault3']}
    A_ueq : array, optional
        Inequality constraint matrix (A_ueq*m <= b_ueq)
    b_ueq : array, optional
        Inequality constraint vector
    Aeq : array, optional
        Equality constraint matrix (Aeq*m = beq)
    beq : array, optional
        Equality constraint vector
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance. With no fixed effective component, convergence
        uses the historical spread of update factors. If any effective data or
        smoothing component is fixed, each updated factor must approach one.
    verbose : bool
        Print progress
    
    Returns:
    --------
    dict with keys:
        - 'm': estimated parameters
        - 'solved_sigma2_by_group': data variances used to solve ``m``
        - 'solved_alpha2_by_group': smoothing variances used to solve ``m``
        - 'proposed_sigma2_by_group': post-update values for a possible next
          iteration
        - 'proposed_alpha2_by_group': post-update values for a possible next
          iteration
        - 'sigma_groups'/'smooth_groups': resolved member mappings
        - 'component_diagnostics': group-level Qw and approximate reduced Q
        - 'convergence_mode'/'convergence_measure': stopping-rule diagnostics
        - 'converged': convergence flag
        - 'iterations': number of iterations
    """
    
    # Setup
    lb, ub = bounds
    n_obs = len(d)
    n_params = G.shape[1]
    n_reg = L.shape[0]
    sigma_only = (n_reg == 0)  # No smoothing constraints → sigma-only VCE
    
    # Configure data ranges
    if data_ranges is None:
        data_ranges = {'data': (0, n_obs)}

    # The base whitening is independent of the estimated variance components.
    # Compute W_k G_k and W_k d_k once, then only rescale them per iteration.
    whitened_data = whiten_data_blocks(data_metrics, data_ranges, G, d)
    
    # Configure fault ranges
    if fault_ranges is None:
        fault_ranges = {'fault': (0, n_params)}
    
    # Resolve membership first, then validate values in group space.  This is
    # intentionally independent of the number of data sets or sources.
    sigma_config = _setup_sigma_groups(data_ranges, sigma_mode, sigma_groups)
    sigma_group_names = list(sigma_config.keys())
    n_sigma = len(sigma_group_names)
    sigma_update = normalize_group_vector(
        sigma_update, sigma_group_names, value_name="sigma_update",
        default_value=True, dtype=bool,
    )
    sigma_values = normalize_group_vector(
        sigma_values, sigma_group_names, value_name="sigma_values",
        default_value=1.0, dtype=float,
    )
    sigma_updatable = [g for g, u in zip(sigma_group_names, sigma_update) if u]
    sigma_fixed = {g: v for g, u, v in zip(sigma_group_names, sigma_update, sigma_values) if not u}
    
    smooth_config = _setup_smooth_groups(
        fault_ranges,
        smooth_mode,
        smooth_groups,
        smoothing_faults=smoothing_faults,
    )
    smooth_group_names = list(smooth_config.keys())
    n_smooth = len(smooth_group_names)
    smooth_update = normalize_group_vector(
        smooth_update, smooth_group_names, value_name="smooth_update",
        default_value=True, dtype=bool,
    )
    smooth_values = normalize_group_vector(
        smooth_values, smooth_group_names, value_name="smooth_values",
        default_value=1.0, dtype=float,
    )
    smooth_updatable = [g for g, u in zip(smooth_group_names, smooth_update) if u]
    smooth_fixed = {g: v for g, u, v in zip(smooth_group_names, smooth_update, smooth_values) if not u}

    # Resolve smoothing rows once.  These rows depend only on the frozen model
    # column layout, not on the variance components updated in the loop.  Empty
    # groups are retained in result metadata but do not anchor convergence.
    smooth_group_rows = {}
    for group, faults in smooth_config.items():
        rows = []
        for fault in faults:
            start_param, end_param = fault_ranges[fault]
            rows.extend(_find_fault_constraints(L, start_param, end_param))
        smooth_group_rows[group] = rows

    # The residual blocks and their Gram/cross products are invariant across
    # VCE iterations.  Only the group variance-component weights change.
    data_quadratic_blocks = {
        dataset: LeastSquaresBlock.prepare(
            _finite_block_array(WG),
            _finite_block_array(Wd),
            name=f"data:{dataset}",
        )
        for dataset, (WG, Wd) in whitened_data.items()
    }
    smooth_quadratic_blocks = {}
    for group, rows in smooth_group_rows.items():
        if rows:
            smooth_quadratic_blocks[group] = LeastSquaresBlock.prepare(
                _finite_block_array(L[rows, :]),
                name=f"smoothing:{group}",
            )

    effective_smooth_groups = {
        group for group, rows in smooth_group_rows.items() if rows
    }
    effective_updatable = list(sigma_updatable) + [
        group for group in smooth_updatable if group in effective_smooth_groups
    ]
    has_fixed_effective_component = (
        any(not update for update in sigma_update)
        or any(
            not smooth_update[index]
            for index, group in enumerate(smooth_group_names)
            if group in effective_smooth_groups
        )
    )
    if not effective_updatable:
        convergence_mode = 'fixed'
    elif has_fixed_effective_component:
        convergence_mode = 'anchored'
    else:
        convergence_mode = 'relative'
    
    if verbose:
        print(f"VCE Setup: {n_obs} obs, {n_params} params, {n_reg} constraints")
        if sigma_only:
            print("  ** Sigma-only VCE: no smoothing constraints (L has 0 rows).")
            print("  ** Only data variance components will be estimated.")
        print(f"Data ranges: {len(data_ranges)} datasets")
        print(f"Fault ranges: {len(fault_ranges)} faults")
        print(f"Sigma groups: {n_sigma} groups")
        print(f"Smooth groups: {n_smooth} groups")
        for group, datasets in sigma_config.items():
            print(f"  Data group {group}: {datasets} (update={group in sigma_updatable}, value={sigma_fixed.get(group, 'auto')})")
        for fault, (start, end) in fault_ranges.items():
            print(f"  Fault {fault}: params [{start}:{end}]")
        if not sigma_only:
            for group, faults in smooth_config.items():
                print(f"  Smooth group {group}: faults {faults} (update={group in smooth_updatable}, value={smooth_fixed.get(group, 'auto')})")
    
    # Initialize variance components (σ^2)
    var_d = {g: sigma_values[i] for i, g in enumerate(sigma_group_names)}
    var_alpha = {g: smooth_values[i] for i, g in enumerate(smooth_group_names)}
    
    # Iteration
    converged = False
    data_effective_dof = {}
    smooth_effective_dof = {}
    solved_sigma2_by_group = dict(var_d)
    solved_alpha2_by_group = dict(var_alpha)
    for it in range(max_iter):
        # These are the variance components that scale the augmented system
        # solved in this iteration. If the iteration limit is reached, the
        # update below describes a possible next iteration, while ``m`` still
        # belongs to this frozen pair of dictionaries.
        solved_sigma2_by_group = dict(var_d)
        solved_alpha2_by_group = dict(var_alpha)
        weighted_blocks = []
        for group in sigma_group_names:
            weighted_blocks.extend(
                (data_quadratic_blocks[dataset], 1.0 / var_d[group])
                for dataset in sigma_config[group]
            )
        for group in smooth_group_names:
            if group in smooth_quadratic_blocks:
                weighted_blocks.append((
                    smooth_quadratic_blocks[group],
                    1.0 / var_alpha[group],
                ))
        N_total, q_total = assemble_quadratic_objective(
            weighted_blocks, n_parameters=n_params
        )

        # ======================================================================
        # Solve using lsqlin solver with constraints
        # ======================================================================

        # Solve using lsqlin solver with constraints
        opts = {'show_progress': False}
        ret = lsqlin.lsqlin_quadratic_auto(
            N_total, q_total,
            lambda: assemble_residual_system(
                weighted_blocks, n_parameters=n_params
            ),
            0,
            A_ueq, b_ueq,
            Aeq, beq,
            lb, ub,
            None, opts,
        )
        m = lsqlin.cvxopt_to_numpy_matrix(ret['x']).flatten()

        # ======================================================================
        # Compute total normal matrix (all datasets + all regularization)
        # ======================================================================

        # The same normal matrix solved above is also the VCE covariance
        # matrix.  Reusing it keeps the solve and variance update metrically
        # identical and removes a second Gram assembly.
        try:
            N_inv = np.linalg.inv(N_total)
        except np.linalg.LinAlgError:
            N_inv = np.linalg.pinv(N_total)

        # ======================================================================
        # Update variance components
        # ======================================================================

        # Update variance components
        update_factors_d = {}
        update_factors_alpha = {}
        # Data variance components
        for i, group in enumerate(sigma_group_names):
            group_whitened_residuals = []
            for dataset in sigma_config[group]:
                WG_sub, Wd_sub = whitened_data[dataset]
                group_whitened_residuals.append(WG_sub @ m - Wd_sub)
            whitened_residual = np.concatenate(group_whitened_residuals)
            # The group normal block is the exact sum of the per-dataset Gram
            # matrices prepared before the VCE loop.  Rebuilding a stacked WG
            # and multiplying it again would be algebraically identical but
            # repeats the dominant O(n p^2) work in every iteration.
            N_d_group = sum(
                data_quadratic_blocks[dataset].gram
                for dataset in sigma_config[group]
            ) / var_d[group]
            dof_eff = len(whitened_residual) - np.trace(N_inv @ N_d_group)
            if dof_eff <= 0:
                dof_eff = len(whitened_residual) * 0.1
            data_effective_dof[group] = float(dof_eff)
            rss = np.dot(whitened_residual, whitened_residual) / var_d[group]
            update_factors_d[group] = rss / dof_eff if sigma_update[i] else 1.0  # 固定sigma不更新
        # Smoothing variance components
        for i, group in enumerate(smooth_group_names):
            L_group_rows = smooth_group_rows[group]
            if L_group_rows:
                L_group = L[L_group_rows, :]
                reg_res = L_group @ m
                # Use the same immutable Gram block that contributed to the
                # solved N_total.  This keeps the effective-DoF trace metric
                # tied to the exact matrix used by the current VCE iteration.
                N_alpha_group = (
                    smooth_quadratic_blocks[group].gram / var_alpha[group]
                )
                dof_eff = len(reg_res) - np.trace(N_inv @ N_alpha_group)
                if dof_eff <= 0:
                    dof_eff = len(reg_res) * 0.1
                smooth_effective_dof[group] = float(dof_eff)
                rss = reg_res.T @ reg_res / var_alpha[group]
                update_factors_alpha[group] = rss / dof_eff if smooth_update[i] else 1.0
            else:
                update_factors_alpha[group] = 1.0
        # Check convergence.  When every effective component is updated, VCE
        # has a common-scale freedom and only relative update factors are
        # identifiable, so the historical spread criterion is retained.  A
        # fixed effective component anchors the scale; then every updated
        # factor must approach one.  In particular, one updated sigma against
        # fixed alpha must not converge merely because its factor list has one
        # element.
        all_update_factors = [update_factors_d[g] for g in sigma_updatable]
        all_update_factors.extend(
            update_factors_alpha[g]
            for g in smooth_updatable
            if g in effective_smooth_groups
        )
        update_factors = np.asarray(all_update_factors, dtype=float)
        if convergence_mode == 'relative':
            change = (
                np.max(update_factors) - np.min(update_factors)
                if update_factors.size else 0.0
            )
        elif convergence_mode == 'anchored':
            change = (
                np.max(np.abs(update_factors - 1.0))
                if update_factors.size else 0.0
            )
        else:
            change = 0.0
        if verbose:
            print(
                f"Iter {it+1}: convergence[{convergence_mode}] = "
                f"{change:.6f}"
            )
            for group, factor in update_factors_d.items():
                print(f"  update_factor_d[{group}]: {factor:.6f}")
            for group, factor in update_factors_alpha.items():
                print(f"  update_factor_alpha[{group}]: {factor:.6f}")
        if change < tol:
            converged = True
            if verbose:
                print(f"Converged after {it+1} iterations")
            break
        # Apply update factors
        for i, group in enumerate(sigma_group_names):
            if sigma_update[i]:
                var_d[group] = var_d[group] * update_factors_d[group]
        for i, group in enumerate(smooth_group_names):
            if smooth_update[i]:
                var_alpha[group] = var_alpha[group] * update_factors_alpha[group]
    
    # Final diagnostics reuse the already-whitened blocks. They do not enter
    # the VCE update and therefore cannot change the estimated model or
    # variance components. Effective dof is the last VCE linearization; it is
    # exact for the converged state and explicitly reported as approximate.
    component_diagnostics = {"data": {}, "smooth": {}}
    for group, datasets in sigma_config.items():
        weighted_quadratic = 0.0
        for dataset in datasets:
            WG_sub, Wd_sub = whitened_data[dataset]
            residual = WG_sub @ m - Wd_sub
            weighted_quadratic += (
                float(np.dot(residual, residual))
                / solved_sigma2_by_group[group]
            )
        dof = data_effective_dof.get(group)
        component_diagnostics["data"][group] = {
            "weighted_quadratic": weighted_quadratic,
            "effective_dof": dof,
            "reduced_weighted_misfit": (
                None if dof is None else weighted_quadratic / dof
            ),
        }

    if not sigma_only:
        for group, faults in smooth_config.items():
            rows = smooth_group_rows[group]
            weighted_quadratic = 0.0
            if rows:
                residual = L[rows, :] @ m
                weighted_quadratic = (
                    float(np.dot(residual, residual))
                    / solved_alpha2_by_group[group]
                )
            dof = smooth_effective_dof.get(group)
            component_diagnostics["smooth"][group] = {
                "weighted_quadratic": weighted_quadratic,
                "effective_dof": dof,
                "reduced_weighted_misfit": (
                    None if dof is None else weighted_quadratic / dof
                ),
            }

    return {
        'm': m,
        # ``solved_*`` is the only scale state scientifically associated with
        # the returned model.  ``proposed_*`` records the update that would
        # seed another iteration when the loop stops before convergence.
        'solved_sigma2_by_group': solved_sigma2_by_group,
        'solved_alpha2_by_group': solved_alpha2_by_group,
        'proposed_sigma2_by_group': dict(var_d),
        'proposed_alpha2_by_group': dict(var_alpha),
        'sigma_groups': {key: list(value) for key, value in sigma_config.items()},
        'smooth_groups': {key: list(value) for key, value in smooth_config.items()},
        'component_diagnostics': component_diagnostics,
        'convergence_mode': convergence_mode,
        'convergence_measure': float(change),
        'fault_ranges': fault_ranges,      # Fault parameter ranges
        'sigma_only': sigma_only,          # True when no smoothing constraints
        'converged': converged,
        'iterations': it + 1
    }


def _find_fault_constraints(L, start_param, end_param):
    """Find which L rows constrain parameters in the given range."""
    constraint_rows = []
    for i in range(L.shape[0]):
        # Check if this constraint row has non-zero elements in the fault parameter range
        if np.any(L[i, start_param:end_param] != 0):
            constraint_rows.append(i)
    return constraint_rows


def assemble_weighted_smoothing_matrix(
    L,
    fault_ranges,
    smooth_groups,
    solved_alpha2_by_group,
):
    """Recreate the smoothing rows used by one returned VCE model.

    The VCE solver orders regularization rows by smoothing group and then by
    fault membership.  Keeping that ordering here makes the published matrix
    an exact representation of the objective associated with ``result['m']``;
    it is not rebuilt later from mutable ``fault.GL`` attributes.
    """
    L = np.asarray(L, dtype=float)
    if L.ndim != 2:
        raise ValueError("L must be a two-dimensional smoothing matrix")
    blocks = []
    for group, faults in smooth_groups.items():
        variance = float(solved_alpha2_by_group[group])
        if not np.isfinite(variance) or variance <= 0.0:
            raise ValueError(
                f"Smoothing variance for group '{group}' must be positive"
            )
        rows = []
        for fault in faults:
            start_param, end_param = fault_ranges[fault]
            rows.extend(_find_fault_constraints(L, start_param, end_param))
        if rows:
            blocks.append(L[rows, :] / np.sqrt(variance))
    if blocks:
        return np.vstack(blocks)
    return np.zeros((0, L.shape[1]), dtype=float)


def _setup_smooth_groups(
    fault_ranges,
    smooth_mode,
    smooth_groups,
    *,
    smoothing_faults=None,
):
    """Set up alpha groups without changing the full model-column layout.

    ``fault_ranges`` includes every source because its ranges index columns of
    ``G`` and ``L``.  Alpha grouping is narrower: Pressure, Sbarbot, and future
    non-Laplacian sources must remain in the model layout without being forced
    into a smoothing group.
    """

    all_faults = list(fault_ranges)
    if smoothing_faults is None:
        faults = all_faults
    else:
        faults = list(smoothing_faults)
        if len(faults) != len(set(faults)):
            raise ValueError("smoothing_faults contains duplicate source names")
        unknown = set(faults) - set(all_faults)
        if unknown:
            raise ValueError(
                "Smoothing sources not found in fault_ranges: "
                + ", ".join(sorted(unknown))
            )

    if smooth_mode == 'grouped' and smooth_groups is not None:
        unsupported = [
            source
            for members in smooth_groups.values()
            for source in members
            if source in all_faults and source not in faults
        ]
        if unsupported:
            raise ValueError(
                f"Source '{unsupported[0]}' does not support smoothing"
            )

    return resolve_group_layout(
        faults,
        smooth_mode,
        smooth_groups,
        member_label="smoothing source",
        single_group_name="all",
        individual_prefix="smooth_",
    )["members_by_group"]


def _setup_sigma_groups(data_ranges, sigma_mode, sigma_groups):
    """Resolve the shared sigma grouping contract for VCE."""

    return resolve_group_layout(
        list(data_ranges),
        sigma_mode,
        sigma_groups,
        member_label="dataset",
        single_group_name="all",
        individual_prefix="group_",
    )["members_by_group"]


def test_multi_fault_vce():
    """Test VCE with multiple fault sets."""
    
    # Synthetic data
    np.random.seed(42)
    n_obs = 300
    n_params = 60  # Multiple fault sets + ramp parameters
    
    G = np.random.randn(n_obs, n_params) * 0.5
    
    # Define fault parameter ranges
    fault_ranges = {
        'main_fault': (0, 25),      # 25 parameters
        'branch_fault': (25, 40),   # 15 parameters
        'background': (40, 50),     # 10 parameters
        'ramp': (50, 60)           # 10 ramp parameters
    }
    
    # Build complete L matrix with different constraint types
    L_parts = []
    
    # Main fault smoothing (strong)
    for i in range(24):
        row = np.zeros(n_params)
        row[i] = -1
        row[i+1] = 1
        L_parts.append(row)
    
    # Branch fault smoothing (medium)
    for i in range(25, 39):
        row = np.zeros(n_params)
        row[i] = -1
        row[i+1] = 1
        L_parts.append(row)
    
    # Background smoothing (weak)
    for i in range(40, 49):
        row = np.zeros(n_params)
        row[i] = -1
        row[i+1] = 1
        L_parts.append(row)
    
    # Ramp constraints (minimal)
    for i in range(50, 59):
        row = np.zeros(n_params)
        row[i] = -1
        row[i+1] = 1
        L_parts.append(row)
    
    L = np.array(L_parts)
    
    # True model with different characteristics
    m_true = np.concatenate([
        np.sin(np.linspace(0, 2*np.pi, 25)) * 0.3,    # Smooth main fault
        np.sin(np.linspace(0, 6*np.pi, 15)) * 0.2,    # Rough branch fault
        np.sin(np.linspace(0, 4*np.pi, 10)) * 0.1,    # Medium background
        np.linspace(-0.05, 0.05, 10)                  # Linear ramp
    ])
    d_clean = G @ m_true
    
    # Add noise
    noise = np.concatenate([
        np.random.randn(100) * 0.02,  # Low noise
        np.random.randn(100) * 0.05,  # Medium noise
        np.random.randn(100) * 0.08   # High noise
    ])
    d_noisy = d_clean + noise
    
    # Data covariance
    Cd = np.diag(np.concatenate([
        np.full(100, 0.02**2),
        np.full(100, 0.05**2),
        np.full(100, 0.08**2)
    ]))
    
    bounds = (-1.0, 1.0)
    
    # Data ranges
    data_ranges = {
        'insar1': (0, 100),
        'insar2': (100, 200),
        'gps': (200, 300)
    }
    data_metrics = prepare_block_covariance_metrics(Cd, data_ranges)
    
    print("Testing Multi-Fault VCE")
    print("=" * 40)
    
    # Test 1: Single smoothing parameter for all faults
    print("\n1. Single smoothing parameter:")
    result1 = simplified_vce(
        data_metrics, d_noisy, G, L, bounds,
        data_ranges, fault_ranges,
        sigma_mode='individual',
        smooth_mode='single', 
        verbose=True
    )
    
    # Test 2: Individual smoothing parameters for each fault
    print("\n2. Individual smoothing parameters:")
    result2 = simplified_vce(
        data_metrics, d_noisy, G, L, bounds,
        data_ranges, fault_ranges,
        sigma_mode='individual',
        smooth_mode='individual',
        verbose=True
    )
    
    # Test 3: Grouped smoothing
    print("\n3. Grouped smoothing:")
    smooth_groups = {
        'fault_smooth': ['main_fault', 'branch_fault'],
        'background_smooth': ['background', 'ramp']
    }
    result3 = simplified_vce(
        data_metrics, d_noisy, G, L, bounds,
        data_ranges, fault_ranges,
        sigma_mode='individual',
        smooth_mode='grouped',
        smooth_groups=smooth_groups,
        verbose=True
    )
    
    # Compare
    print(f"\nParameter errors:")
    print(f"  Single smooth: {np.linalg.norm(result1['m'] - m_true):.6f}")
    print(f"  Individual smooth: {np.linalg.norm(result2['m'] - m_true):.6f}")
    print(f"  Grouped smooth: {np.linalg.norm(result3['m'] - m_true):.6f}")


if __name__ == "__main__":
    test_multi_fault_vce()
