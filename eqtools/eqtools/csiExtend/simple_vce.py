import numpy as np
from . import lsqlin

def simplified_vce(
    Cd_inv, 
    d, 
    G, 
    L, 
    bounds, 
    data_ranges=None,
    fault_ranges=None,
    sigma_mode='single',
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
    
    Solves: minimize ||G*m - d||²_Σd + Σᵢ ||Lᵢ*m||²_Σαᵢ
    
    Subject to:
    - A_ueq*m <= b_ueq (inequality constraints)
    - Aeq*m = beq (equality constraints)  
    - lb <= m <= ub (bounds)
    
    Parameters:
    -----------
    Cd_inv : array (n_obs, n_obs)
        Inverse data covariance matrix
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
    sigma_mode : str
        - 'single': All datasets share one sigma
        - 'individual': Each dataset has its own sigma
        - 'grouped': Custom grouping via sigma_groups
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
        Convergence tolerance (max difference between update factors)
    verbose : bool
        Print progress
    
    Returns:
    --------
    dict with keys:
        - 'm': estimated parameters
        - 'var_d': data variance components (σ²)
        - 'var_alpha': regularization variance components (σ²)
        - 'weights': regularization weights
        - 'converged': convergence flag
        - 'iterations': number of iterations
    """
    
    # Setup
    lb, ub = bounds
    n_obs = len(d)
    n_params = G.shape[1]
    n_reg = L.shape[0]
    
    # Configure data ranges
    if data_ranges is None:
        data_ranges = {'data': (0, n_obs)}
    
    # Configure fault ranges
    if fault_ranges is None:
        fault_ranges = {'fault': (0, n_params)}
    
    # 配置sigma分组
    sigma_config = _setup_sigma_groups(data_ranges, sigma_mode, sigma_groups)
    sigma_group_names = list(sigma_config.keys())
    n_sigma = len(sigma_group_names)
    # 新增：sigma update/fixed处理
    if sigma_update is None:
        sigma_update = [True] * n_sigma
    if sigma_values is None:
        sigma_values = [1.0] * n_sigma
    sigma_update = np.array(sigma_update, dtype=bool)
    sigma_values = np.array(sigma_values, dtype=float)
    sigma_updatable = [g for g, u in zip(sigma_group_names, sigma_update) if u]
    sigma_fixed = {g: v for g, u, v in zip(sigma_group_names, sigma_update, sigma_values) if not u}
    
    # 配置smoothing分组
    smooth_config = _setup_smooth_groups(fault_ranges, smooth_mode, smooth_groups)
    smooth_group_names = list(smooth_config.keys())
    n_smooth = len(smooth_group_names)
    # 新增：smooth update/fixed处理
    if smooth_update is None:
        smooth_update = [True] * n_smooth
    if smooth_values is None:
        smooth_values = [1.0] * n_smooth
    smooth_update = np.array(smooth_update, dtype=bool)
    smooth_values = np.array(smooth_values, dtype=float)
    smooth_updatable = [g for g, u in zip(smooth_group_names, smooth_update) if u]
    smooth_fixed = {g: v for g, u, v in zip(smooth_group_names, smooth_update, smooth_values) if not u}
    
    if verbose:
        print(f"VCE Setup: {n_obs} obs, {n_params} params, {n_reg} constraints")
        print(f"Data ranges: {len(data_ranges)} datasets")
        print(f"Fault ranges: {len(fault_ranges)} faults")
        print(f"Sigma groups: {n_sigma} groups")
        print(f"Smooth groups: {n_smooth} groups")
        for group, datasets in sigma_config.items():
            print(f"  Data group {group}: {datasets} (update={group in sigma_updatable}, value={sigma_fixed.get(group, 'auto')})")
        for fault, (start, end) in fault_ranges.items():
            print(f"  Fault {fault}: params [{start}:{end}]")
        for group, faults in smooth_config.items():
            print(f"  Smooth group {group}: faults {faults} (update={group in smooth_updatable}, value={smooth_fixed.get(group, 'auto')})")
    
    # Initialize variance components (σ²)
    var_d = {g: sigma_values[i] for i, g in enumerate(sigma_group_names)}
    var_alpha = {g: smooth_values[i] for i, g in enumerate(smooth_group_names)}
    
    # Iteration
    for it in range(max_iter):
        # Build weighted system
        G_blocks = []
        d_blocks = []
        
        # Add data blocks
        for i, group in enumerate(sigma_group_names):
            for dataset in sigma_config[group]:
                start, end = data_ranges[dataset]
                Cd_inv_sub = Cd_inv[start:end, start:end]
                G_sub = G[start:end, :]
                d_sub = d[start:end]
                # Weight: sqrt(Cd_inv) / sqrt(σ²_d) = sqrt(Cd_inv) / σ_d
                weight = 1.0 / np.sqrt(var_d[group])
                try:
                    L_chol = np.linalg.cholesky(Cd_inv_sub)
                    G_blocks.append(L_chol @ G_sub * weight)
                    d_blocks.append(L_chol @ d_sub * weight)
                except np.linalg.LinAlgError:
                    sqrt_Cd_inv = np.sqrt(np.diag(Cd_inv_sub))
                    G_blocks.append(np.diag(sqrt_Cd_inv) @ G_sub * weight)
                    d_blocks.append(sqrt_Cd_inv * d_sub * weight)
        
        # Add regularization blocks for each smoothing group
        for i, group in enumerate(smooth_group_names):
            faults = smooth_config[group]
            L_group_rows = []
            for fault in faults:
                start_param, end_param = fault_ranges[fault]
                fault_rows = _find_fault_constraints(L, start_param, end_param)
                L_group_rows.extend(fault_rows)
            if L_group_rows:
                L_group = L[L_group_rows, :]
                n_reg_group = L_group.shape[0]
                reg_weight = 1.0 / np.sqrt(var_alpha[group])
                G_blocks.append(L_group * reg_weight)
                d_blocks.append(np.zeros(n_reg_group) * reg_weight)
        
        # Combine all blocks
        G_aug = np.vstack(G_blocks)
        d_aug = np.concatenate(d_blocks)
        G_aug = np.nan_to_num(G_aug, nan=0.0, posinf=0.0, neginf=0.0)
        d_aug = np.nan_to_num(d_aug, nan=0.0, posinf=0.0, neginf=0.0)

        # ======================================================================
        # Solve using lsqlin solver with constraints
        # ======================================================================

        # Solve using lsqlin solver with constraints
        opts = {'show_progress': False}
        try:
            ret = lsqlin.lsqlin(G_aug, d_aug, 0, A_ueq, b_ueq, Aeq, beq, lb, ub, None, opts)
            m = lsqlin.cvxopt_to_numpy_matrix(ret['x']).flatten()
        except:
            try:
                ret = lsqlin.lsqlin(G_aug, d_aug, 0, A_ueq, b_ueq, None, None, lb, ub, None, opts)
                m = lsqlin.cvxopt_to_numpy_matrix(ret['x']).flatten()
            except:
                ret = lsqlin.lsqlin(G_aug, d_aug, 0, None, None, None, None, lb, ub, None, opts)
                m = lsqlin.cvxopt_to_numpy_matrix(ret['x']).flatten()

        # ======================================================================
        # Compute total normal matrix (all datasets + all regularization)
        # ======================================================================

        # Compute total normal matrix (all datasets + all regularization)
        N_d_total = np.zeros((n_params, n_params))
        for i, group in enumerate(sigma_group_names):
            for dataset in sigma_config[group]:
                start, end = data_ranges[dataset]
                Cd_inv_sub = Cd_inv[start:end, start:end]
                G_sub = G[start:end, :]
                N_d_total += G_sub.T @ Cd_inv_sub @ G_sub / var_d[group]
        N_alpha_total = np.zeros((n_params, n_params))
        for i, group in enumerate(smooth_group_names):
            faults = smooth_config[group]
            L_group_rows = []
            for fault in faults:
                start_param, end_param = fault_ranges[fault]
                fault_rows = _find_fault_constraints(L, start_param, end_param)
                L_group_rows.extend(fault_rows)
            if L_group_rows:
                L_group = L[L_group_rows, :]
                N_alpha_total += L_group.T @ L_group / var_alpha[group]
        N_total = N_d_total + N_alpha_total
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
            group_residuals = []
            group_Cd_inv = []
            group_G = []
            for dataset in sigma_config[group]:
                start, end = data_ranges[dataset]
                pred = G[start:end, :] @ m
                res = pred - d[start:end]
                group_residuals.append(res)
                group_Cd_inv.append(Cd_inv[start:end, start:end])
                group_G.append(G[start:end, :])
            res_combined = np.concatenate(group_residuals)
            Cd_inv_combined = block_diag(*group_Cd_inv)
            G_combined = np.vstack(group_G)
            N_d_group = G_combined.T @ Cd_inv_combined @ G_combined / var_d[group]
            dof_eff = len(res_combined) - np.trace(N_inv @ N_d_group)
            if dof_eff <= 0:
                dof_eff = len(res_combined) * 0.1
            rss = res_combined.T @ Cd_inv_combined @ res_combined / var_d[group]
            update_factors_d[group] = rss / dof_eff if sigma_update[i] else 1.0  # 固定sigma不更新
        # Smoothing variance components
        for i, group in enumerate(smooth_group_names):
            faults = smooth_config[group]
            L_group_rows = []
            for fault in faults:
                start_param, end_param = fault_ranges[fault]
                fault_rows = _find_fault_constraints(L, start_param, end_param)
                L_group_rows.extend(fault_rows)
            if L_group_rows:
                L_group = L[L_group_rows, :]
                reg_res = L_group @ m
                N_alpha_group = L_group.T @ L_group / var_alpha[group]
                dof_eff = len(reg_res) - np.trace(N_inv @ N_alpha_group)
                if dof_eff <= 0:
                    dof_eff = len(reg_res) * 0.1
                rss = reg_res.T @ reg_res / var_alpha[group]
                update_factors_alpha[group] = rss / dof_eff if smooth_update[i] else 1.0
            else:
                update_factors_alpha[group] = 1.0
        # Check convergence
        all_update_factors = [update_factors_d[g] for g in sigma_updatable] + [update_factors_alpha[g] for g in smooth_updatable]
        update_factors = np.array(all_update_factors)
        change = np.max(update_factors) - np.min(update_factors) if len(update_factors) > 0 else 0.0
        if verbose:
            print(f"Iter {it+1}: Max difference between update factors = {change:.6f}")
            for group, factor in update_factors_d.items():
                print(f"  update_factor_d[{group}]: {factor:.6f}")
            for group, factor in update_factors_alpha.items():
                print(f"  update_factor_alpha[{group}]: {factor:.6f}")
        if change < tol:
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
    
    # Compute weights (regularization parameter ratios)
    weights = {}
    for d_group in var_d.keys():
        weights[d_group] = {}
        for alpha_group, alpha_var in var_alpha.items():
            weights[d_group][alpha_group] = alpha_var / var_d[d_group]
    # Simplify output for single smoothing case
    if len(var_alpha) == 1:
        var_alpha_out = list(var_alpha.values())[0]
        std_alpha_out = np.sqrt(var_alpha_out)
        weights_out = {group: var_alpha_out / var for group, var in var_d.items()}
    else:
        var_alpha_out = var_alpha
        std_alpha_out = {k: np.sqrt(v) for k, v in var_alpha.items()}
        weights_out = weights

    # Print final results
    if verbose:
        print(f"\nFinal Results:")
        for group, var in var_d.items():
            print(f"  var_d[{group}] (σ²): {var:.6f}")
            print(f"  std_d[{group}] (σ): {np.sqrt(var):.6f}")
        if isinstance(var_alpha_out, dict):
            for group, var in var_alpha_out.items():
                print(f"  var_alpha[{group}] (σ²): {var:.6f}")
                print(f"  std_alpha[{group}] (σ): {np.sqrt(var):.6f}")
        else:
            print(f"  var_alpha (σ²): {var_alpha_out:.6f}")
            print(f"  std_alpha (σ): {np.sqrt(var_alpha_out):.6f}")
        print(f"\nWeights:")
        if isinstance(weights_out, dict) and any(isinstance(v, dict) for v in weights_out.values()):
            for d_group, weights_dict in weights_out.items():
                for alpha_group, weight in weights_dict.items():
                    print(f"  weight[{d_group}][{alpha_group}]: {weight:.6f}")
        else:
            for group, weight in weights_out.items():
                print(f"  weight[{group}]: {weight:.6f}")

    return {
        'm': m,
        'var_d': var_d,                    # Data variance components (σ²)
        'var_alpha': var_alpha_out,        # Smoothing variance components (σ²)
        'std_d': {k: np.sqrt(v) for k, v in var_d.items()},      # Data standard deviations
        'std_alpha': std_alpha_out,        # Smoothing standard deviations
        'weights': weights_out,            # Regularization weights
        'fault_ranges': fault_ranges,      # Fault parameter ranges
        'converged': it < max_iter - 1,
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


def _setup_smooth_groups(fault_ranges, smooth_mode, smooth_groups):
    """Setup smoothing grouping configuration."""
    
    faults = list(fault_ranges.keys())
    
    if smooth_mode == 'single':
        return {'all': faults}
    
    elif smooth_mode == 'individual':
        return {f'smooth_{fault}': [fault] for fault in faults}
    
    elif smooth_mode == 'grouped':
        if smooth_groups is None:
            raise ValueError("smooth_groups must be provided for 'grouped' mode")
        
        # Validate all faults are assigned
        assigned = set()
        for group, group_faults in smooth_groups.items():
            for fault in group_faults:
                if fault not in faults:
                    raise ValueError(f"Fault '{fault}' not found in fault_ranges")
                if fault in assigned:
                    raise ValueError(f"Fault '{fault}' assigned to multiple groups")
                assigned.add(fault)
        
        if assigned != set(faults):
            unassigned = set(faults) - assigned
            raise ValueError(f"Faults not assigned to any group: {unassigned}")
        
        return smooth_groups
    
    else:
        raise ValueError(f"Unknown smooth_mode: {smooth_mode}")


def _setup_sigma_groups(data_ranges, sigma_mode, sigma_groups):
    """Setup sigma grouping configuration."""
    
    datasets = list(data_ranges.keys())
    
    if sigma_mode == 'single':
        return {'all': datasets}
    
    elif sigma_mode == 'individual':
        return {f'group_{dataset}': [dataset] for dataset in datasets}
    
    elif sigma_mode == 'grouped':
        if sigma_groups is None:
            raise ValueError("sigma_groups must be provided for 'grouped' mode")
        
        # Validate all datasets are assigned
        assigned = set()
        for group, group_datasets in sigma_groups.items():
            for dataset in group_datasets:
                if dataset not in datasets:
                    raise ValueError(f"Dataset '{dataset}' not found in data_ranges")
                if dataset in assigned:
                    raise ValueError(f"Dataset '{dataset}' assigned to multiple groups")
                assigned.add(dataset)
        
        if assigned != set(datasets):
            unassigned = set(datasets) - assigned
            raise ValueError(f"Datasets not assigned to any group: {unassigned}")
        
        return sigma_groups
    
    else:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")


def block_diag(*arrays):
    """Simple block diagonal matrix construction."""
    if len(arrays) == 1:
        return arrays[0]
    
    shapes = [a.shape for a in arrays]
    out_shape = (sum(s[0] for s in shapes), sum(s[1] for s in shapes))
    out = np.zeros(out_shape)
    
    r, c = 0, 0
    for arr in arrays:
        h, w = arr.shape
        out[r:r+h, c:c+w] = arr
        r, c = r + h, c + w
    
    return out


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
    Cd_inv = np.diag(np.concatenate([
        np.full(100, 1/0.02**2),
        np.full(100, 1/0.05**2),
        np.full(100, 1/0.08**2)
    ]))
    
    bounds = (-1.0, 1.0)
    
    # Data ranges
    data_ranges = {
        'insar1': (0, 100),
        'insar2': (100, 200),
        'gps': (200, 300)
    }
    
    print("Testing Multi-Fault VCE")
    print("=" * 40)
    
    # Test 1: Single smoothing parameter for all faults
    print("\n1. Single smoothing parameter:")
    result1 = simplified_vce(
        Cd_inv, d_noisy, G, L, bounds, 
        data_ranges, fault_ranges,
        sigma_mode='individual',
        smooth_mode='single', 
        verbose=True
    )
    
    # Test 2: Individual smoothing parameters for each fault
    print("\n2. Individual smoothing parameters:")
    result2 = simplified_vce(
        Cd_inv, d_noisy, G, L, bounds,
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
        Cd_inv, d_noisy, G, L, bounds,
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