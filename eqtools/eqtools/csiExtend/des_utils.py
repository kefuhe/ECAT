"""
Depth-Equalized Smoothing (DES) Utilities
Based on Zhang et al. (2025)

This module implements the scaling of Green's function matrices to equalize 
smoothing effects across different depths or fault patches.

Key features:
1. Supports multiple scaling modes (per_patch, per_depth, per_column).
2. Preserves physical rake consistency by default (per_patch/per_depth).
3. Handles multi-fault systems and polynomial parameters explicitly.
4. Integrated Logging for easy debugging and process tracking.
"""

import numpy as np
import logging

# Initialize module-level logger
logger = logging.getLogger(__name__)

def setup_des_logging(level=logging.INFO):
    """
    Helper to configure the DES logger from the main script.
    
    Parameters:
    -----------
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def compute_Gn_norm(G, G_norm="l2"):
    """
    Compute the norm of each column G_n in the Green's function matrix.
    
    Parameters:
    -----------
    G : array_like, shape (M, N)
        Green's function matrix.
    G_norm : str, optional
        Norm type: "l2" (default, Euclidean length) or "l1".
        
    Returns:
    --------
    array_like, shape (N,)
        Column-wise norms of G.
    """
    if G_norm == "l2":
        return np.linalg.norm(G, axis=0)  # ||G_n|| L2 norm (Length)
    elif G_norm == "l1":
        return np.sum(np.abs(G), axis=0)  # ||G_n|| L1 norm
    else:
        logger.error(f"Invalid G_norm type: {G_norm}")
        raise ValueError("G_norm must be 'l2' or 'l1'")


def _assign_depth_groups(depth_grouping_config, n_items):
    """
    Assign depth groups based on configuration strategy.
    Internal helper function.
    """
    strategy = depth_grouping_config.get('strategy', 'uniform')
    depths = depth_grouping_config.get('depths')
    
    if depths is None:
        msg = "depths array is required in depth_grouping_config"
        logger.error(msg)
        raise ValueError(msg)
    
    depths = np.array(depths)
    if len(depths) != n_items:
        msg = f"depths length ({len(depths)}) must match number of items ({n_items})"
        logger.error(msg)
        raise ValueError(msg)
    
    if strategy == 'uniform':
        interval = depth_grouping_config.get('interval')
        if interval is None:
            msg = "interval is required for uniform strategy"
            logger.error(msg)
            raise ValueError(msg)
        min_depth = np.min(depths)
        group_assignments = np.floor((depths - min_depth) / interval).astype(int)
        
    elif strategy == 'custom':
        custom_groups = depth_grouping_config.get('custom_groups')
        if custom_groups is None:
            msg = "custom_groups is required for custom strategy"
            logger.error(msg)
            raise ValueError(msg)
        custom_groups = np.array(custom_groups)
        group_assignments = np.digitize(depths, custom_groups) - 1
        group_assignments = np.clip(group_assignments, 0, len(custom_groups) - 1)
        
    elif strategy == 'values':
        tolerance = depth_grouping_config.get('tolerance', 1e-6)
        unique_depths = []
        group_assignments = np.zeros(n_items, dtype=int)
        for i, depth in enumerate(depths):
            assigned = False
            for j, unique_depth in enumerate(unique_depths):
                if abs(depth - unique_depth) <= tolerance:
                    group_assignments[i] = j
                    assigned = True
                    break
            if not assigned:
                unique_depths.append(depth)
                group_assignments[i] = len(unique_depths) - 1
    else:
        msg = f"Unknown grouping strategy: {strategy}"
        logger.error(msg)
        raise ValueError(msg)
    
    return group_assignments


def create_depth_grouping_config(strategy='uniform', depths=None, **kwargs):
    """
    Helper function to create depth grouping configuration.
    
    Parameters:
    -----------
    strategy : str
        Strategy for grouping patches by depth:
        - 'uniform': Groups depths into bins of fixed interval.
        - 'values':  Groups depths by unique values (within tolerance).
        - 'custom':  Uses provided depth edges to define groups.
    depths : array_like
        Array of depth values for all patches.
    **kwargs : 
        Additional arguments for specific strategies:
        - interval (float): Bin size for 'uniform' strategy.
        - custom_groups (array): Edges for 'custom' strategy.
        - tolerance (float): Tolerance for 'values' strategy (default 1e-6).
        
    Returns:
    --------
    dict
        A configuration dictionary suitable for use in apply_des_transformation.
        
    Example:
    --------
    >>> config = create_depth_grouping_config(strategy='uniform', depths=depth_array, interval=2.0)
    """
    config = {
        'strategy': strategy,
        'depths': depths
    }
    config.update(kwargs)
    return config


def compute_Gprime_explicit(G, fault_indices_config, mode="per_patch", G_norm="l2", 
                            depth_grouping_config=None):
    """
    Compute the scaled G' matrix for DES smoothing using explicit parameter indexing.
    
    This is the core function for the DES method.
    
    Modes Explanation:
    ------------------
    1. "per_patch" (DEFAULT):
       - Calculates the Combined Patch Norm: sqrt(||SS||^2 + ||DS||^2).
       - Scales both SS and DS components of the same patch by this single value.
       - BENEFIT: Preserves the rake (slip angle) and physical consistency.
    
    2. "per_depth":
       - Calculates the Combined Patch Norm first (preserving consistency).
       - Groups patches by depth layers.
       - Averages the norms within each depth group.
       - Scales all parameters in that depth layer by the group average.
       - BENEFIT: Reduces noise for uneven station distribution while keeping rake consistent.
    
    3. "per_column" (Legacy / Zhang 2025 Original):
       - Scales every column independently based on its own norm.
       - BENEFIT: Maximum resolution recovery.
       - DRAWBACK: May distort rake or amplify noise in weak components.

    Parameters:
    -----------
    G : array_like (M, N)
        Original Green's function matrix.
    fault_indices_config : list of dict
        Configuration for each fault, containing indices and depths.
        Structure:
        [
            {
                'name': 'FaultA', 
                'ss': [idx...], 
                'ds': [idx...], 
                'poly': [idx...], 
                'depths': [d1, d2...] # Depth per patch
            },
            ...
        ]
    mode : str, optional
        "per_patch" (default), "per_depth", or "per_column".
    G_norm : str, optional
        "l2" (Euclidean length) or "l1".
    depth_grouping_config : dict, optional
        Configuration for depth grouping (required for "per_depth" mode).

    Returns:
    --------
    G_prime : array_like
        Scaled Green's function matrix.
    alpha : float
        The global mean effective norm used for scaling.
    final_effective_norms : array_like
        The effective norms used for scaling each parameter (used for D matrix).
    all_fault_indices : array_like
        Indices of all fault parameters (excluding polynomials).
    final_scales : array_like
        The actual scaling factors applied to each column.
    """
    logger.info(f"Starting DES computation. Mode: {mode}, Norm: {G_norm}")

    M, N = G.shape
    all_fault_indices = []
    # Store the calculated "effective norm" for every column in G
    full_effective_norms = np.zeros(N) 
    # Lists to collect data for per_depth processing
    depth_processing_list = [] # Stores (indices, combined_norm, depth) tuples
    # List to collect norms for calculating the global Alpha
    norms_for_alpha = [] 
    
    # --- 2. Iterate through each fault configuration ---
    for f_cfg in fault_indices_config:
        name = f_cfg.get('name', 'Unknown')
        logger.info(f"Processing fault: {name}")
        # Extract indices and depths
        ss_idx = np.array(f_cfg.get('ss', []), dtype=int)
        ds_idx = np.array(f_cfg.get('ds', []), dtype=int)
        depths = np.array(f_cfg.get('depths', [])) 
        
        # Aggregate all fault indices (to isolate them from polynomials later)
        current_fault_indices = np.concatenate([ss_idx, ds_idx]).astype(int)
        all_fault_indices.extend(current_fault_indices)
        
        # Compute Raw Column Norms ||G_n||
        norms_ss = compute_Gn_norm(G[:, ss_idx], G_norm) if len(ss_idx) > 0 else np.array([])
        norms_ds = compute_Gn_norm(G[:, ds_idx], G_norm) if len(ds_idx) > 0 else np.array([])
        
        # Log basic stats for this fault
        max_ss = np.max(norms_ss) if len(norms_ss) > 0 else 0
        logger.debug(f"Fault '{name}': {len(ss_idx)} SS patches (max norm={max_ss:.2e}), {len(ds_idx)} DS patches.")
        # --- 3. Mode-Specific Logic ---
        
        if mode == "per_column":
            # --- Legacy Mode: Independent Scaling ---
            # Directly use raw norms as effective norms
            if len(ss_idx) > 0:
                full_effective_norms[ss_idx] = norms_ss
                norms_for_alpha.extend(norms_ss)
            if len(ds_idx) > 0:
                full_effective_norms[ds_idx] = norms_ds
                norms_for_alpha.extend(norms_ds)
                
        else:
            # --- Consistent Modes: "per_patch" and "per_depth" ---
            # Both start by calculating the Combined Patch Norm to preserve rake
            
            # Handle Single-Component vs Dual-Component faults
            if len(ds_idx) > 0:
                if len(ss_idx) != len(ds_idx):
                    msg = f"Fault {f_cfg.get('name')} SS/DS count mismatch."
                    logger.error(msg)
                    raise ValueError(msg)
                # Combined Norm = sqrt(||SS||^2 + ||DS||^2)
                combined_norms = np.sqrt(norms_ss**2 + norms_ds**2)
            else:
                # If only SS exists, Combined Norm is just SS norm
                combined_norms = norms_ss
            
            if mode == "per_patch":
                # Assign combined norm to both SS and DS columns
                if len(ss_idx) > 0:
                    full_effective_norms[ss_idx] = combined_norms
                if len(ds_idx) > 0:
                    full_effective_norms[ds_idx] = combined_norms
                
                # Add to alpha calculation (twice if two components, to weight properly)
                if len(ss_idx) > 0: norms_for_alpha.extend(combined_norms)
                if len(ds_idx) > 0: norms_for_alpha.extend(combined_norms)
            
            elif mode == "per_depth":
                # Collect data for depth grouping
                # We store the *indices* and the *combined norm* for that patch
                if len(depths) != len(combined_norms):
                     msg = f"Fault {f_cfg.get('name')} depths length mismatch."
                     logger.error(msg)
                     raise ValueError(msg)
                
                # Store SS info
                if len(ss_idx) > 0:
                    depth_processing_list.append({
                        'indices': ss_idx,
                        'norms': combined_norms, # Use combined norm!
                        'depths': depths
                    })
                # Store DS info
                if len(ds_idx) > 0:
                    depth_processing_list.append({
                        'indices': ds_idx,
                        'norms': combined_norms, # Use combined norm!
                        'depths': depths
                    })

    # Ensure indices are sorted
    all_fault_indices = np.array(sorted(all_fault_indices))

    # --- 4. Post-Processing for "per_depth" ---
    if mode == "per_depth":
        if depth_grouping_config is None:
            msg = "per_depth mode requires depth_grouping_config"
            logger.error(msg)
            raise ValueError(msg)
            
        # Flatten the lists collected above
        all_depth_indices = []
        all_depth_values = []
        all_depth_norms = []
        
        for item in depth_processing_list:
            all_depth_indices.extend(item['indices'])
            all_depth_values.extend(item['depths'])
            all_depth_norms.extend(item['norms'])
            
        all_depth_indices = np.array(all_depth_indices)
        all_depth_values = np.array(all_depth_values)
        all_depth_norms = np.array(all_depth_norms)
        
        # Prepare config for grouping helper
        # We override 'depths' with the collected global depths
        temp_config = depth_grouping_config.copy()
        temp_config['depths'] = all_depth_values
        
        # Assign groups
        group_assignments = _assign_depth_groups(temp_config, len(all_depth_values))
        
        # Average norms within groups
        unique_groups = np.unique(group_assignments)
        for g in unique_groups:
            mask = (group_assignments == g)
            
            # Indices belonging to this depth group
            target_cols = all_depth_indices[mask]
            
            # The combined norms associated with these parameters
            norms_in_group = all_depth_norms[mask]
            
            # Calculate mean effective norm for this depth
            mean_norm = np.mean(norms_in_group)
            
            # Assign this mean value to all parameters in this depth layer
            full_effective_norms[target_cols] = mean_norm
            
        # Re-collect norms for Alpha from the updated full array
        norms_for_alpha = full_effective_norms[all_fault_indices]

    # --- 5. Final Calculation of Alpha and Scales ---
    # Alpha is the global mean of effective norms
    alpha = np.mean(norms_for_alpha)
    
    scale_factors_full = np.zeros(N)
    
    # Calculate scale factors only for fault columns
    with np.errstate(divide='ignore', invalid='ignore'):
        # Extract relevant norms
        fault_norms = full_effective_norms[all_fault_indices]

        # Check for zero norms (dead patches)
        zero_mask = (fault_norms == 0)
        if np.any(zero_mask):
            logger.warning(f"Found {np.sum(zero_mask)} fault columns with zero sensitivity. Scaling set to 0.")

        # Formula: Scale = Alpha / ||G_n||_effective
        scales = alpha / fault_norms
        scales[fault_norms == 0] = 0.0 # Handle zero-sensitivity columns
        
        scale_factors_full[all_fault_indices] = scales

    # Log scale statistics
    min_scale = np.min(scales[~zero_mask]) if np.any(~zero_mask) else 0
    max_scale = np.max(scales[~zero_mask]) if np.any(~zero_mask) else 0
    logger.info(f"Scale factors range: [{min_scale:.2f}, {max_scale:.2f}]")

    # --- 6. Apply Scaling ---
    G_prime = G.copy()
    # Vectorized in-place multiplication
    G_prime[:, all_fault_indices] *= scale_factors_full[all_fault_indices]
    
    # Extract outputs for fault parameters only (to match API expectations)
    final_effective_norms = full_effective_norms[all_fault_indices]
    final_scales = scale_factors_full[all_fault_indices]
    
    return G_prime, alpha, final_effective_norms, all_fault_indices, final_scales

def compute_Dprime_with_poly(D, effective_norms, fault_indices):
    """
    Compute the scaled smoothing operator D'.
    
    D'_kn = D_kn * (||G_k|| / ||G_n||)
    
    Parameters:
    -----------
    D : array_like (K, N_total)
        Original smoothing matrix.
    effective_norms : array_like (N_fault,)
        The effective norms (||G_n||) used to scale G.
    fault_indices : array_like (N_fault,)
        Indices of fault parameters in D.
        
    Returns:
    --------
    D_prime : array_like
    """
    logger.info("Scaling smoothing matrix D...")

    K, N_total = D.shape
    D_prime = D.copy()
    
    # Determine the number of fault parameters
    N_fault = len(fault_indices)
    
    # Create the scaling matrix using broadcasting
    # Rows (k): assume 1-to-1 mapping between smoothing rows and fault params
    # We clip indices to ensure we don't go out of bounds if K > N_fault
    row_indices = np.arange(K)
    norm_k_idx = np.minimum(row_indices, N_fault - 1)
    
    norm_k = effective_norms[norm_k_idx] # (K,)
    norm_n = effective_norms             # (N_fault,)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # scaling_matrix[k, n] = ||G_k|| / ||G_n||
        scaling_matrix = norm_k[:, np.newaxis] / norm_n[np.newaxis, :]
        scaling_matrix[~np.isfinite(scaling_matrix)] = 0
        
    # Apply scaling only to fault parameter columns
    D_prime[:, fault_indices] = D[:, fault_indices] * scaling_matrix
    
    return D_prime

# =============================================================================
# 3. TRANSFORMATION & RECOVERY
# =============================================================================

def recover_sf_with_poly(s_prime, alpha, effective_norms, fault_indices):
    """
    Recover the physical slip s_final from the scaled solution s_prime.
    
    This acts as the inverse operation to the DES scaling.
    Formula: s_final = alpha * s_prime / ||G_n||_effective

    Parameters:
    -----------
    s_prime : array_like, shape (N,)
        The solution vector obtained from the inverse problem using scaled matrices (G', D', etc.).
    alpha : float
        The global scaling factor returned by apply_des_transformation.
    effective_norms : array_like, shape (N_fault,)
        The effective norms (||G_n||) for fault parameters returned by apply_des_transformation.
    fault_indices : array_like, shape (N_fault,)
        Indices of the fault slip parameters in the solution vector.
    
    Returns:
    --------
    array_like, shape (N,)
        The recovered physical slip vector. Polynomial parameters are left unchanged from s_prime.
    """
    logger.info("Recovering physical slip from scaled solution...")
    s_final = s_prime.copy()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        s_restored = alpha * s_prime[fault_indices] / effective_norms
        s_restored[effective_norms == 0] = 0
        s_final[fault_indices] = s_restored

    logger.debug("Solution recovered successfully.")
    return s_final


def apply_des_transformation(G, fault_indices_config, D=None, 
                           A_ineq=None, b_ineq=None, 
                           A_eq=None, b_eq=None, 
                           lb=None, ub=None, 
                           mode="per_patch", 
                           G_norm="l2", 
                           depth_grouping_config=None):
    """
    Main entry point to apply complete DES transformation.

    This function orchestrates the depth-equalized smoothing (DES) process by:
    1. Calculating the scaled Green's function matrix (G').
    2. Scaling the smoothing matrix (D).
    3. Transforming all constraints (Equality, Inequality, Bounds) to the new scaled space.

    Parameters:
    -----------
    G : array_like, shape (M, N)
        The original Green's function matrix relating slip to data.
    fault_indices_config : list of dict
        Configuration for each fault component indices.
        Example structure:
        [
            {
                'name': 'fault1',
                'ss': [0, 1, 2],       # Strike-slip indices
                'ds': [3, 4, 5],       # Dip-slip indices
                'poly': [6, 7],        # Polynomial indices
                'depths': [5.0, 5.0, 10.0] # Depths for patches (length matches len(ss))
            }
        ]
    D : array_like, shape (K, N), optional
        Smoothing matrix (Laplacian). If provided, D' will be computed.
    A_ineq, b_ineq : array_like, optional
        Inequality constraints matrices (A_ineq * x <= b_ineq).
    A_eq, b_eq : array_like, optional
        Equality constraints matrices (A_eq * x = b_eq).
    lb, ub : array_like, optional
        Lower and upper bounds for parameters.
    mode : str, optional
        Scaling mode: "per_patch" (default), "per_depth", or "per_column".
        - "per_patch": Scales patch components together (preserves rake).
        - "per_depth": Scales patches at same depth together (preserves rake, reduces noise).
        - "per_column": Scales each column independently (maximizes resolution).
    G_norm : str, optional
        Norm type for column scaling: "l2" (default) or "l1".
    depth_grouping_config : dict, optional
        Configuration for depth grouping strategy (required if mode="per_depth").
        Example: {'strategy': 'uniform', 'interval': 2.0, 'depths': all_depths}

    Returns:
    --------
    dict
        A dictionary containing transformed matrices and recovery info. Key output keys:
        - 'G_prime': Scaled Green's function (M, N).
        - 'D_prime': Scaled smoothing matrix (K, N) (if D provided).
        - 'A_ineq_prime', 'b_ineq': Transformed inequality constraints.
        - 'A_eq_prime', 'b_eq': Transformed equality constraints.
        - 'lb_prime', 'ub_prime': Transformed bounds.
        - 'alpha': Global scaling factor (scalar).
        - 'scale_factors': Array of scaling factors per parameter (N,).
        - 'fault_indices': Indices of slip parameters (non-polynomial).
        - 'norm2_fault': Effective norms used for scaling (same as ||G_n||).

    Example:
    --------
    >>> # 1. Prepare Fault Configuration
    >>> fault_config = [{
    ...     'name': 'Fault_A',
    ...     'ss': [0, 1], 'ds': [2, 3], 'poly': [4],
    ...     'depths': [5.0, 10.0]
    ... }]
    >>>
    >>> # 2. Apply DES Transformation
    >>> des_res = apply_des_transformation(G, fault_config, D=Laplacian, 
    ...                                    lb=lb, ub=ub, mode="per_patch")
    >>>
    >>> # 3. Solve Inverse Problem (using scaled matrices)
    >>> # m_prime = solver(des_res['G_prime'], d, des_res['D_prime'], ...)
    >>>
    >>> # 4. Recover Physical Slip
    >>> m_physical = recover_sf_with_poly(
    ...     m_prime, 
    ...     des_res['alpha'], 
    ...     des_res['norm2_fault'], 
    ...     des_res['fault_indices']
    ... )
    """
    logger.info("=== Applying DES Transformation ===")

    # 1. Transform G and get scaling parameters
    G_prime, alpha, effective_norms, fault_indices, scale_factors = compute_Gprime_explicit(
        G, fault_indices_config, mode, G_norm, depth_grouping_config
    )
    
    result = {
        'G_prime': G_prime,
        'alpha': alpha,
        'norm2_fault': effective_norms, # Keeping name for compatibility, actually norms
        'fault_indices': fault_indices,
        'scale_factors': scale_factors
    }
    
    # 2. Transform D (Smoothing Matrix)
    if D is not None:
        result['D_prime'] = compute_Dprime_with_poly(D, effective_norms, fault_indices)

    # 3. Transform Constraints & Bounds
    logger.info("Scaling constraints and bounds...")

    # 3. Transform Constraints (Inequalities) -> A' = A * Scale
    if A_ineq is not None:
        if b_ineq is None: b_ineq = np.zeros(A_ineq.shape[0])
        A_ineq_prime = A_ineq.copy()
        A_ineq_prime[:, fault_indices] *= scale_factors
        result['A_ineq_prime'] = A_ineq_prime
        result['b_ineq'] = b_ineq
    
    # 4. Transform Constraints (Equalities) -> A' = A * Scale
    if A_eq is not None:
        if b_eq is None: b_eq = np.zeros(A_eq.shape[0])
        A_eq_prime = A_eq.copy()
        A_eq_prime[:, fault_indices] *= scale_factors
        result['A_eq_prime'] = A_eq_prime
        result['b_eq'] = b_eq
        
    # 5. Transform Bounds -> lb' = lb / Scale
    if lb is not None and ub is not None:
        lb_prime = lb.copy()
        ub_prime = ub.copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            lb_prime[fault_indices] /= scale_factors
            ub_prime[fault_indices] /= scale_factors
            # Handle division by zero or inf
            lb_prime[~np.isfinite(lb_prime)] = -np.inf
            ub_prime[~np.isfinite(ub_prime)] = np.inf
            
        result['lb_prime'] = lb_prime
        result['ub_prime'] = ub_prime

    logger.info("=== DES Transformation Complete ===")
    return result

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