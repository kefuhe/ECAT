import numpy as np
from scipy.optimize import lsq_linear

def rigorous_vce(
    Cd_inv, 
    d, 
    G, 
    L, 
    bounds, 
    data_ranges=None,
    fault_ranges=None,
    sigma_mode='single',
    sigma_groups=None,
    smooth_mode='single',
    smooth_groups=None,
    max_iter=20, 
    tol=1e-4, 
    verbose=False
):
    """
    Rigorous Variance Component Estimation (VCE) for geodetic inversions.
    
    This function performs rigorous variance component estimation to determine
    the optimal weights between data fitting and regularization components using
    an iterative approach based on variance component estimation theory.
    
    Mathematical Framework:
    -----------------------
    The method solves the following optimization problem:
    
    minimize: ||G*m - d||^2_Σd + Σ_i ||L_i*m||^2_Σα_i
    
    where:
    - G: Design matrix relating model parameters to observations
    - L: Complete regularization matrix (all constraints)
    - m: Model parameters to be estimated
    - d: Observed data vector
    - Σd: Data variance components (multiple datasets allowed)
    - Σα: Regularization variance components (multiple fault groups allowed)
    
    The variance components are estimated using rigorous statistical theory:
    
    E[v_i^TP_iv_i] = σ_i^2 * tr(P_i - P_iGN^-^1G^TP_i)
    
    where P_i is the weight matrix for component i, and N is the normal matrix.
    
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
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance for variance component changes
    verbose : bool
        Print progress information
    
    Returns:
    --------
    dict with keys:
        - 'm': estimated parameters
        - 'var_d': data variance components (σ^2)
        - 'var_alpha': regularization variance components (σ^2)
        - 'weights': regularization weights
        - 'converged': convergence flag
        - 'iterations': number of iterations
        - 'residuals_d': data residuals for each group
        - 'residuals_alpha': regularization residuals for each group
    
    Mathematical Details:
    ---------------------
    The rigorous VCE approach uses the following statistical framework:
    
    1. For each variance component σ_i^2, compute the expected value:
       E[v_i^TP_iv_i] = σ_i^2 * (r_i - tr(P_iGN^-^1G^TP_i))
    
    2. Set up the variance component equation:
       v_i^TP_iv_i = σ_i^2 * (r_i - tr(P_iGN^-^1G^TP_i))
    
    3. Solve the system of variance component equations iteratively using
       the rigorous trace formulas and coefficient matrix approach
    
    where:
    - v_i: residual vector for component i
    - P_i: weight matrix for component i
    - r_i: redundancy (degrees of freedom) for component i
    - N: normal matrix of the combined system
    
    References:
    -----------
    - Koch, K.R. (1999). Parameter Estimation and Hypothesis Testing in Linear Models
    - Aster, R.C., et al. (2018). Parameter Estimation and Inverse Problems
    - Teunissen, P.J.G. (2000). Adjustment Theory: An Introduction
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
    
    # Configure sigma groups
    sigma_config = _setup_sigma_groups(data_ranges, sigma_mode, sigma_groups)
    
    # Configure smoothing groups
    smooth_config = _setup_smooth_groups(fault_ranges, smooth_mode, smooth_groups)
    
    if verbose:
        print(f"Rigorous VCE Setup: {n_obs} obs, {n_params} params, {n_reg} constraints")
        print(f"Data ranges: {len(data_ranges)} datasets")
        print(f"Fault ranges: {len(fault_ranges)} faults")
        print(f"Sigma groups: {len(sigma_config)} groups")
        print(f"Smooth groups: {len(smooth_config)} groups")
        for group, datasets in sigma_config.items():
            print(f"  Data group {group}: {datasets}")
        for fault, (start, end) in fault_ranges.items():
            print(f"  Fault {fault}: params [{start}:{end}]")
        for group, faults in smooth_config.items():
            print(f"  Smooth group {group}: faults {faults}")
    
    # Initialize variance components (σ^2)
    var_d = {group: 1.0 for group in sigma_config.keys()}
    var_alpha = {group: 1.0 for group in smooth_config.keys()}
    
    # Storage for residuals
    residuals_d = {}
    residuals_alpha = {}
    
    # Iteration
    for it in range(max_iter):
        
        # ======================================================================
        # Step 1: Build weighted system matrices
        # ======================================================================
        
        G_blocks = []
        d_blocks = []
        
        # Add data blocks
        for group, datasets in sigma_config.items():
            for dataset in datasets:
                start, end = data_ranges[dataset]
                Cd_inv_sub = Cd_inv[start:end, start:end]
                G_sub = G[start:end, :]
                d_sub = d[start:end]
                
                # Weight: sqrt(Cd_inv) / sqrt(σ^2_d) = sqrt(Cd_inv) / σ_d
                weight = 1.0 / np.sqrt(var_d[group])
                try:
                    L_chol = np.linalg.cholesky(Cd_inv_sub)
                    G_blocks.append(L_chol @ G_sub * weight)
                    d_blocks.append(L_chol @ d_sub * weight)
                except np.linalg.LinAlgError:
                    # Fallback for non-positive definite matrices
                    sqrt_Cd_inv = np.sqrt(np.diag(Cd_inv_sub))
                    G_blocks.append(np.diag(sqrt_Cd_inv) @ G_sub * weight)
                    d_blocks.append(sqrt_Cd_inv * d_sub * weight)
        
        # Add regularization blocks for each smoothing group
        for group, faults in smooth_config.items():
            # Extract L rows that correspond to this group's faults
            L_group_rows = []
            for fault in faults:
                start_param, end_param = fault_ranges[fault]
                # Find L rows that act on this fault's parameters
                fault_rows = _find_fault_constraints(L, start_param, end_param)
                L_group_rows.extend(fault_rows)
            
            if L_group_rows:
                L_group = L[L_group_rows, :]
                n_reg_group = L_group.shape[0]
                
                # Weight: 1/sqrt(σ^2_α) = 1/σ_α
                reg_weight = 1.0 / np.sqrt(var_alpha[group])
                G_blocks.append(L_group * reg_weight)
                d_blocks.append(np.zeros(n_reg_group) * reg_weight)
        
        # ======================================================================
        # Step 2: Solve weighted least squares problem
        # ======================================================================
        
        # Combine all blocks
        G_aug = np.vstack(G_blocks)
        d_aug = np.concatenate(d_blocks)
        
        # Handle numerical issues
        G_aug = np.nan_to_num(G_aug, nan=0.0, posinf=0.0, neginf=0.0)
        d_aug = np.nan_to_num(d_aug, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Solve: minimize ||G_aug*m - d_aug||^2 subject to bounds
        result = lsq_linear(G_aug, d_aug, bounds=(lb, ub), method='trf')
        m = result.x
        
        # ======================================================================
        # Step 3: Compute residuals for each component
        # ======================================================================
        
        # Data residuals for each group
        residuals_d = {}
        for group, datasets in sigma_config.items():
            group_residuals = []
            for dataset in datasets:
                start, end = data_ranges[dataset]
                pred = G[start:end, :] @ m
                res = pred - d[start:end]
                group_residuals.append(res)
            residuals_d[group] = np.concatenate(group_residuals)
        
        # Regularization residuals for each group
        residuals_alpha = {}
        for group, faults in smooth_config.items():
            L_group_rows = []
            for fault in faults:
                start_param, end_param = fault_ranges[fault]
                fault_rows = _find_fault_constraints(L, start_param, end_param)
                L_group_rows.extend(fault_rows)
            
            if L_group_rows:
                L_group = L[L_group_rows, :]
                residuals_alpha[group] = L_group @ m
            else:
                residuals_alpha[group] = np.array([])
        
        # ======================================================================
        # Step 4: Compute normal matrices for rigorous VCE
        # ======================================================================
        
        # Collect all normal matrices for each component
        N_components = {}
        
        # Data normal matrices
        for group, datasets in sigma_config.items():
            group_Cd_inv = []
            group_G = []
            for dataset in datasets:
                start, end = data_ranges[dataset]
                Cd_inv_sub = Cd_inv[start:end, start:end]
                G_sub = G[start:end, :]
                group_Cd_inv.append(Cd_inv_sub)
                group_G.append(G_sub)
            
            # Combine for this group
            Cd_inv_combined = block_diag(*group_Cd_inv)
            G_combined = np.vstack(group_G)
            N_components[f'd_{group}'] = G_combined.T @ Cd_inv_combined @ G_combined / var_d[group]
        
        # Regularization normal matrices
        for group, faults in smooth_config.items():
            L_group_rows = []
            for fault in faults:
                start_param, end_param = fault_ranges[fault]
                fault_rows = _find_fault_constraints(L, start_param, end_param)
                L_group_rows.extend(fault_rows)
            
            if L_group_rows:
                L_group = L[L_group_rows, :]
                N_components[f'alpha_{group}'] = L_group.T @ L_group / var_alpha[group]
            else:
                N_components[f'alpha_{group}'] = np.zeros((n_params, n_params))
        
        # Total normal matrix
        N_total = sum(N_components.values())
        
        try:
            N_inv = np.linalg.inv(N_total)
        except np.linalg.LinAlgError:
            N_inv = np.linalg.pinv(N_total)
        
        # ======================================================================
        # Step 5: Rigorous variance component estimation using trace formulas
        # ======================================================================
        
        # Collect quadratic forms (w vector)
        w_vector = []
        component_names = []
        component_sizes = []
        
        # Data components
        for group, datasets in sigma_config.items():
            v = residuals_d[group]
            group_Cd_inv = []
            for dataset in datasets:
                start, end = data_ranges[dataset]
                Cd_inv_sub = Cd_inv[start:end, start:end]
                group_Cd_inv.append(Cd_inv_sub)
            
            Cd_inv_combined = block_diag(*group_Cd_inv)
            quad_form = v.T @ Cd_inv_combined @ v / var_d[group]
            w_vector.append(quad_form)
            component_names.append(f'd_{group}')
            component_sizes.append(len(v))
        
        # Regularization components
        for group, faults in smooth_config.items():
            if group in residuals_alpha and len(residuals_alpha[group]) > 0:
                v = residuals_alpha[group]
                quad_form = v.T @ v / var_alpha[group]
                w_vector.append(quad_form)
                component_names.append(f'alpha_{group}')
                component_sizes.append(len(v))
        
        w_vector = np.array(w_vector)
        n_components = len(w_vector)
        
        # ======================================================================
        # Step 6: Compute rigorous trace matrix S
        # ======================================================================
        
        S_matrix = np.zeros((n_components, n_components))
        
        for i, comp_i in enumerate(component_names):
            for j, comp_j in enumerate(component_names):
                Ni = N_components[comp_i]
                Nj = N_components[comp_j]
                
                if i == j:
                    # Diagonal terms: s_i_i = n_i - 2*tr(N^-^1N_i) + tr(N^-^1N_iN^-^1N_i)
                    ni = component_sizes[i]
                    trace1 = np.trace(N_inv @ Ni)
                    trace2 = np.trace(N_inv @ Ni @ N_inv @ Ni)
                    S_matrix[i, j] = ni - 2 * trace1 + trace2
                else:
                    # Off-diagonal terms: s_i_j = tr(N^-^1N_iN^-^1N_j)
                    S_matrix[i, j] = np.trace(N_inv @ Ni @ N_inv @ Nj)
        
        # ======================================================================
        # Step 7: Solve for variance component updates
        # ======================================================================
        
        try:
            # Solve: S * c = w for variance component multipliers
            c = np.linalg.solve(S_matrix, w_vector)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse if S is singular
            c = np.linalg.pinv(S_matrix) @ w_vector
        
        # ======================================================================
        # Step 8: Check convergence and update variance components
        # ======================================================================
        update_factors_d = {}
        update_factors_alpha = {}
        for d, group_name in zip(c[:len(var_d)], var_d.keys()):
            update_factors_d[group_name] = d
        for alpha, group_name in zip(c[len(var_d):], var_alpha.keys()):
            update_factors_alpha[group_name] = alpha
        # Check convergence based on update factors
        all_update_factors = list(update_factors_d.values()) + list(update_factors_alpha.values())
        update_factors = np.array(all_update_factors)
        change = np.max(update_factors) - np.min(update_factors)
        
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
        
        # Apply update factors to get new variance components
        var_d = {group: var_d[group] * update_factors_d[group] for group in var_d.keys()}
        var_alpha = {group: var_alpha[group] * update_factors_alpha[group] for group in var_alpha.keys()}
    
    # ======================================================================
    # Final Results
    # ======================================================================
    
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
    
    if verbose:
        print(f"\nFinal Results:")
        for group, var in var_d.items():
            print(f"  var_d[{group}] (σ^2): {var:.6f}")
            print(f"  std_d[{group}] (σ): {np.sqrt(var):.6f}")
        
        if isinstance(var_alpha_out, dict):
            for group, var in var_alpha_out.items():
                print(f"  var_alpha[{group}] (σ^2): {var:.6f}")
                print(f"  std_alpha[{group}] (σ): {np.sqrt(var):.6f}")
        else:
            print(f"  var_alpha (σ^2): {var_alpha_out:.6f}")
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
        'var_d': var_d,                    # Data variance components (σ^2)
        'var_alpha': var_alpha_out,        # Smoothing variance components (σ^2)
        'std_d': {k: np.sqrt(v) for k, v in var_d.items()},      # Data standard deviations
        'std_alpha': std_alpha_out,        # Smoothing standard deviations
        'weights': weights_out,            # Regularization weights
        'fault_ranges': fault_ranges,      # Fault parameter ranges
        'converged': it < max_iter - 1,
        'iterations': it + 1,
        'residuals_d': residuals_d,        # Data residuals for each group
        'residuals_alpha': residuals_alpha # Regularization residuals for each group
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


def test_rigorous_multi_fault_vce():
    """Test rigorous VCE with multiple fault sets."""
    
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
    
    print("Testing Rigorous Multi-Fault VCE")
    print("=" * 50)
    
    # Test 1: Single smoothing parameter for all faults
    print("\n1. Single smoothing parameter:")
    result1 = rigorous_vce(
        Cd_inv, d_noisy, G, L, bounds, 
        data_ranges, fault_ranges,
        sigma_mode='individual',
        smooth_mode='single', 
        verbose=True
    )
    
    # Test 2: Individual smoothing parameters for each fault
    print("\n2. Individual smoothing parameters:")
    result2 = rigorous_vce(
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
    result3 = rigorous_vce(
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
    
    # Show parameter breakdown
    print(f"\nParameter breakdown (Individual smooth case):")
    m_est = result2['m']
    print(f"  Main fault error: {np.linalg.norm(m_est[:25] - m_true[:25]):.6f}")
    print(f"  Branch fault error: {np.linalg.norm(m_est[25:40] - m_true[25:40]):.6f}")
    print(f"  Background error: {np.linalg.norm(m_est[40:50] - m_true[40:50]):.6f}")
    print(f"  Ramp error: {np.linalg.norm(m_est[50:60] - m_true[50:60]):.6f}")


if __name__ == "__main__":
    test_rigorous_multi_fault_vce()