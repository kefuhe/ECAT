"""
Depth-Equalized Smoothing (DES) from Zhang et al., 2025
"""

import numpy as np

def compute_Gn_norm(G, G_norm="l2"):
    """
    Compute the norm of each column G_n in the Green's function matrix.
    
    Parameters:
    -----------
    G : array_like, shape (M, N)
        Green's function matrix
    G_norm : str, optional
        Norm type: "l2" (default) or "l1"
        
    Returns:
    --------
    array_like, shape (N,)
        Column-wise norms of G
    """
    if G_norm == "l2":
        norm2_fault = np.linalg.norm(G, axis=0)
        return norm2_fault  # ||G_n|| L2 norm
    elif G_norm == "l1":
        return np.sum(np.abs(G), axis=0)          # ||G_n|| L1 norm
    else:
        raise ValueError("G_norm must be 'l2' or 'l1'")


def compute_Gprime_with_poly(G, poly_positions=None, mode="per_column", groups=None, G_norm="l2", 
                           depth_grouping_config=None):
    """
    Compute the scaled G' matrix for DES smoothing, handling polynomial terms separately.
    
    This function implements the scaling procedure for fault slip parameters only,
    leaving polynomial terms unchanged.
    
    Parameters:
    -----------
    G : array_like, shape (M, N)
        Green's function matrix including fault slip and polynomial terms
    poly_positions : dict or None, optional
        Dictionary mapping fault names to polynomial position ranges (start, end).
        Format: {fault_name: (start_idx, end_idx)} or {fault_name: [start_idx, end_idx]}
        If None, assumes no polynomial terms exist.
    mode : str, optional
        Scaling mode: "per_column" (column-wise scaling) or "per_depth" (group-wise scaling)
    groups : array_like, shape (N_fault,), optional
        Deprecated: use depth_grouping_config instead
    G_norm : str, optional
        Norm type: "l2" or "l1"
    depth_grouping_config : dict, optional
        Configuration for depth-based grouping when mode="per_depth". Format:
        {
            'strategy': 'uniform' | 'custom' | 'values',
            'depths': array of depth values for each fault parameter,
            'interval': float (for uniform strategy),
            'custom_groups': array of group boundaries (for custom strategy),
            'tolerance': float (for values strategy, default 1e-6)
        }
        
    Returns:
    --------
    G_prime : array_like, shape (M, N)
        Scaled Green's function matrix (fault slip scaled, polynomials unchanged)
    alpha : float
        Scaling parameter (mean of squared norms for fault slip only)
    norm2_fault : array_like, shape (N_fault,)
        Squared norms of fault slip columns only
    fault_indices : array_like, shape (N_fault,)
        Indices of fault slip columns in the original matrix
    scale_factors : array_like, shape (N_fault,)
        Scaling factors applied to fault slip columns
    """
    M, N = G.shape
    
    # Identify polynomial positions from ranges
    poly_indices = set()
    if poly_positions is not None:
        for fault_name, positions in poly_positions.items():
            if positions is not None:
                # Handle both tuple and list formats: (start, end) or [start, end]
                if isinstance(positions, (tuple, list)) and len(positions) == 2:
                    start_idx, end_idx = positions
                    # Add all indices in the range [start_idx, end_idx)
                    poly_indices.update(range(start_idx, end_idx))
                else:
                    raise ValueError(f"Invalid poly_positions format for fault {fault_name}: {positions}. "
                                   "Expected (start_idx, end_idx) or [start_idx, end_idx]")
    
    # Get fault slip indices (all non-polynomial columns)
    fault_indices = np.array([i for i in range(N) if i not in poly_indices])
    
    # Extract fault slip part of G
    G_fault = G[:, fault_indices]
    
    # Compute norms and scaling for fault slip part only
    norm2_fault = compute_Gn_norm(G_fault, G_norm)
    alpha = np.mean(norm2_fault)
    
    # Compute scaling factors
    if mode == "per_column":
        # Column-wise scaling for fault slip
        scale_factors = alpha / norm2_fault
    elif mode == "per_depth":
        # Group-wise scaling by depth layers
        if groups is not None:
            # Backward compatibility: use legacy groups array
            import warnings
            warnings.warn("Parameter 'groups' is deprecated. Use 'depth_grouping_config' instead.", 
                         DeprecationWarning, stacklevel=2)
            group_assignments = _assign_groups_legacy(groups, len(fault_indices))
        elif depth_grouping_config is not None:
            group_assignments = _assign_depth_groups(depth_grouping_config, len(fault_indices))
        else:
            raise ValueError("per_depth mode requires either 'groups' or 'depth_grouping_config'")
        
        scale_factors = np.zeros_like(norm2_fault)
        for g in np.unique(group_assignments):
            idx = (group_assignments == g)
            mean_norm2 = np.mean(norm2_fault[idx])
            scale_factors[idx] = alpha / mean_norm2
    else:
        raise ValueError("mode must be 'per_column' or 'per_depth'")
    
    # Apply scaling to create G_prime
    G_prime = G.copy()
    G_prime[:, fault_indices] = G_fault * scale_factors
    # Polynomial columns remain unchanged
    
    return G_prime, alpha, norm2_fault, fault_indices, scale_factors


def _assign_groups_legacy(groups, n_fault):
    """
    Legacy group assignment for backward compatibility.
    """
    if len(groups) != n_fault:
        raise ValueError(f"groups length ({len(groups)}) must match number of fault columns ({n_fault})")
    return np.array(groups)


def _assign_depth_groups(depth_grouping_config, n_fault):
    """
    Assign depth groups based on configuration.
    
    Parameters:
    -----------
    depth_grouping_config : dict
        Configuration dictionary with strategy and parameters
    n_fault : int
        Number of fault parameters
        
    Returns:
    --------
    array_like
        Group assignments for each fault parameter
    """
    strategy = depth_grouping_config.get('strategy', 'uniform')
    depths = depth_grouping_config.get('depths')
    
    if depths is None:
        raise ValueError("depths array is required in depth_grouping_config")
    
    depths = np.array(depths)
    if len(depths) != n_fault:
        raise ValueError(f"depths length ({len(depths)}) must match number of fault columns ({n_fault})")
    
    if strategy == 'uniform':
        # Uniform depth intervals
        interval = depth_grouping_config.get('interval')
        if interval is None:
            raise ValueError("interval is required for uniform strategy")
        
        min_depth = np.min(depths)
        group_assignments = np.floor((depths - min_depth) / interval).astype(int)
        
    elif strategy == 'custom':
        # Custom group boundaries
        custom_groups = depth_grouping_config.get('custom_groups')
        if custom_groups is None:
            raise ValueError("custom_groups is required for custom strategy")
        
        custom_groups = np.array(custom_groups)
        group_assignments = np.digitize(depths, custom_groups) - 1
        group_assignments = np.clip(group_assignments, 0, len(custom_groups) - 1)
        
    elif strategy == 'values':
        # Group by unique depth values with tolerance
        tolerance = depth_grouping_config.get('tolerance', 1e-6)
        
        # Sort unique depths
        unique_depths = []
        group_assignments = np.zeros(n_fault, dtype=int)
        
        for i, depth in enumerate(depths):
            # Find if this depth already exists within tolerance
            assigned = False
            for j, unique_depth in enumerate(unique_depths):
                if abs(depth - unique_depth) <= tolerance:
                    group_assignments[i] = j
                    assigned = True
                    break
            
            if not assigned:
                # Create new group
                unique_depths.append(depth)
                group_assignments[i] = len(unique_depths) - 1
                
    else:
        raise ValueError(f"Unknown grouping strategy: {strategy}")
    
    return group_assignments


def create_depth_grouping_config(strategy='uniform', depths=None, **kwargs):
    """
    Helper function to create depth grouping configuration.
    
    Parameters:
    -----------
    strategy : str
        Grouping strategy: 'uniform', 'custom', or 'values'
    depths : array_like
        Depth values for each fault parameter
    **kwargs : additional parameters
        - interval: float (for uniform strategy)
        - custom_groups: array (for custom strategy) 
        - tolerance: float (for values strategy)
        
    Returns:
    --------
    dict
        Configuration dictionary
        
    Examples:
    --------
    # Uniform grouping with 2km intervals
    config = create_depth_grouping_config('uniform', depths=depths, interval=2000)
    
    # Custom grouping with specified boundaries
    config = create_depth_grouping_config('custom', depths=depths, 
                                         custom_groups=[0, 5000, 10000, 15000])
    
    # Group by unique values with 100m tolerance
    config = create_depth_grouping_config('values', depths=depths, tolerance=100)
    """
    config = {
        'strategy': strategy,
        'depths': depths
    }
    config.update(kwargs)
    return config


def compute_Dprime_with_poly(D, norm2_fault, fault_indices):
    """
    Compute the scaled smoothing operator D' handling both fault slip and polynomial parameters.
    
    The scaling follows DES theory: D'_kn = D_kn * (||G_k||^2 / ||G_n||^2)
    where k and n are indices in the fault slip parameter space.
    
    Parameters:
    -----------
    D : array_like, shape (K, N_total)
        Original smoothing operator matrix for all parameters (fault slip + polynomials)
    norm2_fault : array_like, shape (N_fault,)
        Squared norms ||G_n||^2 for fault slip from compute_Gprime_with_poly
    fault_indices : array_like, shape (N_fault,)
        Indices of fault slip columns in the full parameter vector
        
    Returns:
    --------
    D_prime : array_like, shape (K, N_total)
        Scaled smoothing operator matrix with fault slip columns scaled, polynomial unchanged
    """
    K, N_total = D.shape
    D_prime = D.copy()
    
    # Apply DES scaling to fault slip columns only
    # For each row k and each fault slip column n:
    # D'_kn = D_kn * (||G_k|| / ||G_n||)
    for k in range(K):
        # Assume row k corresponds to constraint equation k
        # We need to map k to a fault parameter index for ||G_k||
        k_fault_idx = k  # min(k, len(norm2_fault) - 1)  # Safe mapping
        norm2_k = norm2_fault[k_fault_idx]
        
        D_prime[k, fault_indices] = D[k, fault_indices] * (norm2_k / norm2_fault)
    
    # Polynomial columns remain unchanged (no scaling applied)
    
    return D_prime


def transform_inequality_constraints(A_ineq, b_ineq, scale_factors, fault_indices):
    """
    Transform inequality constraints As ≤ b for DES scaling.
    
    For fault slip parameters, the constraint becomes:
    A'_prime * s'_fault ≤ b where A'_prime = A * scale_factors

    Parameters:
    -----------
    A_ineq : array_like, shape (M_ineq, N)
        Original inequality constraint matrix
    b_ineq : array_like, shape (M_ineq,)
        Original inequality constraint bounds  
    scale_factors : array_like, shape (N_fault,)
        Scaling factors for fault slip parameters
    fault_indices : array_like, shape (N_fault,)
        Indices of fault slip columns
        
    Returns:
    --------
    A_ineq_prime : array_like, shape (M_ineq, N)
        Scaled inequality constraint matrix
    b_ineq : array_like, shape (M_ineq,)
        Unchanged constraint bounds
    """
    A_ineq_prime = A_ineq.copy()
    
    # Vectorized scaling for fault slip columns
    A_ineq_prime[:, fault_indices] = A_ineq[:, fault_indices] * scale_factors
    
    # Polynomial columns and bounds remain unchanged
    return A_ineq_prime, b_ineq


def transform_equality_constraints(A_eq, b_eq, scale_factors, fault_indices):
    """
    Transform equality constraints As = b for DES scaling.
    
    For fault slip parameters, the constraint becomes:
    A'_prime * s'_fault = b where A'_prime = A * scale_factors
    
    Parameters:
    -----------
    A_eq : array_like, shape (M_eq, N)
        Original equality constraint matrix
    b_eq : array_like, shape (M_eq,)
        Original equality constraint bounds
    scale_factors : array_like, shape (N_fault,)
        Scaling factors for fault slip parameters
    fault_indices : array_like, shape (N_fault,)
        Indices of fault slip columns
        
    Returns:
    --------
    A_eq_prime : array_like, shape (M_eq, N)
        Scaled equality constraint matrix
    b_eq : array_like, shape (M_eq,)
        Unchanged constraint bounds
    """
    A_eq_prime = A_eq.copy()
    
    # Vectorized scaling for fault slip columns
    A_eq_prime[:, fault_indices] = A_eq[:, fault_indices] * scale_factors
    
    # Polynomial columns and bounds remain unchanged
    return A_eq_prime, b_eq


def transform_bounds(lb, ub, scale_factors, fault_indices):
    """
    Transform lower and upper bounds for DES scaling.
    
    For fault slip parameters: lb' = lb / scale_factors, ub' = ub / scale_factors
    
    Parameters:
    -----------
    lb : array_like, shape (N,)
        Original lower bounds
    ub : array_like, shape (N,)
        Original upper bounds
    scale_factors : array_like, shape (N_fault,)
        Scaling factors for fault slip parameters
    fault_indices : array_like, shape (N_fault,)
        Indices of fault slip columns
        
    Returns:
    --------
    lb_prime : array_like, shape (N,)
        Scaled lower bounds
    ub_prime : array_like, shape (N,)
        Scaled upper bounds
    """
    lb_prime = lb.copy()
    ub_prime = ub.copy()
    
    # Vectorized scaling for fault slip bounds
    lb_prime[fault_indices] = lb[fault_indices] / scale_factors

    ub_prime[fault_indices] = ub[fault_indices] / scale_factors
    
    # Polynomial bounds remain unchanged
    return lb_prime, ub_prime


def recover_sf_with_poly(s_prime, alpha, norm2_fault, fault_indices):
    """
    Recover the final slip solution from the scaled solution, handling polynomials.
    
    Parameters:
    -----------
    s_prime : array_like, shape (N,)
        Scaled solution from DES inversion (including fault slip and polynomials)
    alpha : float
        Scaling parameter from compute_Gprime_with_poly
    norm2_fault : array_like, shape (N_fault,)
        Squared norms for fault slip from compute_Gprime_with_poly
    fault_indices : array_like, shape (N_fault,)
        Indices of fault slip columns
        
    Returns:
    --------
    array_like, shape (N,)
        Final solution with fault slip recovered and polynomials unchanged
    """
    s_final = s_prime.copy()
    
    # Recover fault slip part: s_fault = alpha * s'_fault / ||G_n||^2
    s_final[fault_indices] = alpha * s_prime[fault_indices] / norm2_fault
    
    # Polynomial part remains unchanged (s_final[poly_indices] = s_prime[poly_indices])
    
    return s_final


def get_poly_positions_from_multifaults(multifaults):
    """
    Extract polynomial positions from multifaults object.
    
    Parameters:
    -----------
    multifaults : object
        Multifaults object containing poly_positions attribute
        
    Returns:
    --------
    dict
        Dictionary mapping fault names to polynomial position ranges (start, end)
    """
    if hasattr(multifaults, 'poly_positions'):
        return multifaults.poly_positions
    else:
        # If no poly_positions attribute, assume no polynomials
        return {}


def apply_des_transformation(G, D=None, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, 
                           lb=None, ub=None, poly_positions=None, mode="per_column", 
                           groups=None, G_norm="l2", depth_grouping_config=None):
    """
    Apply complete DES transformation to all matrices and constraints.
    
    Parameters:
    -----------
    G : array_like, shape (M, N)
        Green's function matrix
    D : array_like, shape (K, N), optional
        Smoothing operator matrix for all parameters (fault slip + polynomials)
    A_ineq : array_like, shape (M_ineq, N), optional
        Inequality constraint matrix
    b_ineq : array_like, shape (M_ineq,), optional
        Inequality constraint bounds
    A_eq : array_like, shape (M_eq, N), optional
        Equality constraint matrix
    b_eq : array_like, shape (M_eq,), optional
        Equality constraint bounds
    lb : array_like, shape (N,), optional
        Lower bounds
    ub : array_like, shape (N,), optional
        Upper bounds
    poly_positions : dict, optional
        Dictionary mapping fault names to polynomial position ranges (start, end)
        Format: {fault_name: (start_idx, end_idx)} or {fault_name: [start_idx, end_idx]}
    mode : str, optional
        Scaling mode: "per_column" or "per_depth"
    groups : array_like, optional
        Deprecated: use depth_grouping_config instead
    G_norm : str, optional
        Norm type: "l2" or "l1"
    depth_grouping_config : dict, optional
        Configuration for depth-based grouping when mode="per_depth"
        
    Returns:
    --------
    dict
        Dictionary containing all transformed matrices and scaling information
    """
    # Transform Green's function matrix
    G_prime, alpha, norm2_fault, fault_indices, scale_factors = compute_Gprime_with_poly(
        G, poly_positions, mode, groups, G_norm, depth_grouping_config
    )
    
    result = {
        'G_prime': G_prime,
        'alpha': alpha,
        'norm2_fault': norm2_fault,
        'fault_indices': fault_indices,
        'scale_factors': scale_factors
    }
    
    # Transform smoothing operator if provided
    if D is not None:
        D_prime = compute_Dprime_with_poly(D, norm2_fault, fault_indices)
        result['D_prime'] = D_prime
    
    # Transform inequality constraints if provided
    if A_ineq is not None:
        if b_ineq is None:
            b_ineq = np.zeros(A_ineq.shape[0])
        A_ineq_prime, b_ineq_out = transform_inequality_constraints(
            A_ineq, b_ineq, scale_factors, fault_indices
        )
        result['A_ineq_prime'] = A_ineq_prime
        result['b_ineq'] = b_ineq_out
    
    # Transform equality constraints if provided
    if A_eq is not None:
        if b_eq is None:
            b_eq = np.zeros(A_eq.shape[0])
        A_eq_prime, b_eq_out = transform_equality_constraints(
            A_eq, b_eq, scale_factors, fault_indices
        )
        result['A_eq_prime'] = A_eq_prime
        result['b_eq'] = b_eq_out
    
    # Transform bounds if provided
    if lb is not None and ub is not None:
        lb_prime, ub_prime = transform_bounds(lb, ub, scale_factors, fault_indices)
        result['lb_prime'] = lb_prime
        result['ub_prime'] = ub_prime
    
    return result


if __name__ == "__main__":
    # Example usage with polynomial position ranges
    
    # Example poly_positions format:
    # poly_positions = {
    #     'fault1': (100, 105),  # polynomial indices from 100 to 104
    #     'fault2': [200, 203],  # polynomial indices from 200 to 202
    #     'fault3': None         # no polynomials for this fault
    # }
    
    # Extract polynomial positions from multifaults
    poly_positions = get_poly_positions_from_multifaults(multifaults)
    
    # Apply complete DES transformation
    des_result = apply_des_transformation(
        G=G, D=D, A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
        lb=lb, ub=ub, poly_positions=poly_positions, mode="per_column", G_norm="l2"
    )
    
    # Use transformed matrices for inversion
    G_prime = des_result['G_prime']
    D_prime = des_result['D_prime']
    A_ineq_prime = des_result['A_ineq_prime']
    A_eq_prime = des_result['A_eq_prime']
    lb_prime = des_result['lb_prime']
    ub_prime = des_result['ub_prime']
    
    # ... perform inversion to get s_prime ...
    
    # Recover final solution
    sf = recover_sf_with_poly(
        s_prime, des_result['alpha'], des_result['norm2_fault'], des_result['fault_indices']
    )
    
    print(f"DES transformation completed successfully")
    print(f"Total parameters: {G.shape[1]}")
    print(f"Fault slip parameters: {len(des_result['fault_indices'])}")
    print(f"Polynomial parameters: {G.shape[1] - len(des_result['fault_indices'])}")