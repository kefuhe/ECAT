"""
Euler Pole Inequality Constraints for Multi-Fault Inversion

This module provides functions to generate inequality constraints based on Euler pole motion
for tectonic block modeling. The constraints ensure that fault slip rates are consistent
with block motion predicted by Euler poles.

Key concepts:
- Reference strike: defines the projection direction for Euler velocities
- Motion sense: dextral (+1) vs sinistral (-1) motion sense
- Inequality form: slip + euler_motion <= 0 (for dextral) or >= 0 (for sinistral)

Author: Kfhe
Date: 08/22/2023
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional


def calculate_euler_matrix_for_points(lonc: np.ndarray, latc: np.ndarray) -> np.ndarray:
    """
    Calculate the standard Euler matrix for given observation points.
    This matrix converts Euler vector [wx, wy, wz] to velocity [ve, vn].
    
    Parameters:
    -----------
    lonc : np.ndarray
        Longitude coordinates in radians
    latc : np.ndarray
        Latitude coordinates in radians
        
    Returns:
    --------
    euler_mat : np.ndarray
        Euler matrix with shape (2*num_patches, 3)
        First num_patches rows: East velocity coefficients for [wx, wy, wz]
        Next num_patches rows: North velocity coefficients for [wx, wy, wz]
    """
    num_patches = len(lonc)
    euler_mat = np.zeros((2 * num_patches, 3))
    
    # Earth radius in meters
    R = 6378137.0
    
    # Vectorized calculations
    cos_lat = np.cos(latc)
    sin_lat = np.sin(latc)
    cos_lon = np.cos(lonc)
    sin_lon = np.sin(lonc)
    
    # East component coefficients (first num_patches rows)
    euler_mat[:num_patches, 0] = -R * sin_lat * cos_lon  # wx coefficient
    euler_mat[:num_patches, 1] = -R * sin_lat * sin_lon  # wy coefficient  
    euler_mat[:num_patches, 2] = R * cos_lat             # wz coefficient
    
    # North component coefficients (next num_patches rows)
    euler_mat[num_patches:, 0] = R * sin_lon             # wx coefficient
    euler_mat[num_patches:, 1] = -R * cos_lon            # wy coefficient
    euler_mat[num_patches:, 2] = 0                       # wz coefficient
    
    return euler_mat


def calculate_reference_strike_vector(reference_strike_deg: float, num_patches: int) -> np.ndarray:
    """
    Calculate reference strike unit vectors for projection.
    
    Parameters:
    -----------
    reference_strike_deg : float
        Reference strike angle in degrees (measured clockwise from north)
    num_patches : int
        Number of patches
        
    Returns:
    --------
    vec_reference : np.ndarray
        Reference direction unit vectors (shape: [num_patches, 2])
        Each row is [east_component, north_component]
    """
    reference_strike_rad = np.deg2rad(reference_strike_deg)
    
    vec_reference = np.zeros((num_patches, 2))
    vec_reference[:, 0] = np.sin(reference_strike_rad)  # East component
    vec_reference[:, 1] = np.cos(reference_strike_rad)  # North component

    return vec_reference


def convert_euler_pole_to_vector(lat_pole: float, lon_pole: float, omega: float) -> np.ndarray:
    """
    Convert Euler pole (lat, lon, omega) to Cartesian Euler vector.
    
    Parameters:
    -----------
    lat_pole : float
        Latitude of Euler pole in radians
    lon_pole : float
        Longitude of Euler pole in radians
    omega : float
        Angular velocity in radians/year
        
    Returns:
    --------
    euler_vector : np.ndarray
        Cartesian Euler vector [wx, wy, wz] in radians/year
    """
    omega_x = omega * np.cos(lat_pole) * np.cos(lon_pole)
    omega_y = omega * np.cos(lat_pole) * np.sin(lon_pole)
    omega_z = omega * np.sin(lat_pole)
    
    return np.array([omega_x, omega_y, omega_z])

def project_euler_to_strike(euler_mat: np.ndarray, fault: object, patch_indices: list, 
                           reference_strike_deg: float, num_patches: int) -> np.ndarray:
    """
    Project Euler velocities to each patch's strike-slip direction.
    
    The reference strike is used to ensure consistent strike direction convention
    (avoiding opposite directions), but projection is done to each patch's own strike.
    
    Parameters:
    -----------
    euler_mat : np.ndarray
        Euler matrix (2*num_patches, 3) with East and North velocity components
    fault : object
        Fault object containing patch information
    patch_indices : list
        List of patch indices to process
    reference_strike_deg : float
        Reference strike angle in degrees (for direction consistency)
    num_patches : int
        Number of patches
        
    Returns:
    --------
    euler_strike : np.ndarray
        Strike-slip velocity components (num_patches, 3) for [wx, wy, wz]
        Positive values indicate motion in each patch's strike direction
    """
    # Extract East and North velocity components from Euler matrix
    vel_east = euler_mat[:num_patches, :]   # East velocity coefficients
    vel_north = euler_mat[num_patches:, :]  # North velocity coefficients
    
    # Get all patch strikes from fault object (in radians)
    all_strikes = np.array(fault.getStrikes())
    patch_strikes_rad = all_strikes[patch_indices]  # Selected patch strikes
    
    # Convert reference strike to radians
    reference_strike_rad = np.deg2rad(reference_strike_deg)
    
    # Vectorized calculation of strike direction vectors
    strike_east = np.sin(patch_strikes_rad)
    strike_north = np.cos(patch_strikes_rad)
    
    # Reference strike direction vector
    ref_east = np.sin(reference_strike_rad)
    ref_north = np.cos(reference_strike_rad)
    
    # Check direction consistency using vectorized dot product
    dot_products = strike_east * ref_east + strike_north * ref_north
    
    # Reverse directions where dot product is negative
    reverse_mask = dot_products < 0
    strike_east[reverse_mask] = -strike_east[reverse_mask]
    strike_north[reverse_mask] = -strike_north[reverse_mask]
    
    # Vectorized projection of Euler velocities to strike directions
    euler_strike = (vel_east * strike_east[:, np.newaxis] + 
                   vel_north * strike_north[:, np.newaxis])
    
    return euler_strike


def determine_motion_sign(motion_sense: str) -> float:
    """
    Determine motion sign based on motion sense.
    
    Parameters:
    -----------
    motion_sense : str
        Motion sense: 'dextral', 'sinistral', 'right_lateral', 'left_lateral'
        
    Returns:
    --------
    motion_sign : float
        +1.0 for dextral (right-lateral), -1.0 for sinistral (left-lateral)
    """
    if motion_sense.lower() in ['dextral', 'right_lateral', 'right']:
        return 1.0  # Positive for dextral motion
    elif motion_sense.lower() in ['sinistral', 'left_lateral', 'left']:
        return -1.0  # Negative for sinistral motion
    else:
        raise ValueError(f"Invalid motion_sense: {motion_sense}")

def generate_euler_inequality_constraints(multifault_solver, euler_config, all_datasets):
    """
    Generate Euler pole inequality constraints for multi-fault inversion.
    
    The constraints are in the form: A_ineq * x <= b_ineq
    Where: slip_strike + motion_sign * (euler1_strike - euler2_strike) <= 0
    
    For dextral motion (motion_sign = +1):
        slip_strike + (euler1_strike - euler2_strike) <= 0
    For sinistral motion (motion_sign = -1):
        -(slip_strike + (euler1_strike - euler2_strike)) <= 0
        
    Parameters:
    -----------
    multifault_solver : multifaultsolve_boundLSE or BayesianMultiFaultsInversion
        The multi-fault solver object
    euler_config : dict
        Parsed Euler constraints configuration
    all_datasets : list
        List of all dataset objects
        
    Returns:
    --------
    tuple : (A_ineq, b_ineq)
        A_ineq : np.ndarray
            Inequality constraint matrix
        b_ineq : np.ndarray
            Inequality constraint vector
    """
    
    if not euler_config.get('enabled', False):
        return None, None
    
    # Get fault and data information
    faults = multifault_solver.config.faults_list
    fault_name_to_obj = {f.name: f for f in faults}
    
    # Get transform_indices from the first fault (where it's stored)
    transform_indices = {}
    if len(faults) > 0 and hasattr(faults[0], 'transform_indices'):
        transform_indices = faults[0].transform_indices
    elif hasattr(multifault_solver, 'transform_indices'):
        transform_indices = multifault_solver.transform_indices
    
    # Get dataset objects from config
    dataset_name_to_obj = {d.name: d for d in all_datasets}
    
    # Pre-calculate constraint information for all faults
    constraint_info = []
    total_constraints = 0
    
    for fault_name, params in euler_config['faults'].items():
        if fault_name not in fault_name_to_obj:
            continue
            
        fault = fault_name_to_obj[fault_name]
        
        # Get patch indices to apply constraints
        apply_patches = params.get('apply_to_patches', None)
        if apply_patches is None:
            patch_indices = list(range(len(fault.patch)))
        else:
            patch_indices = apply_patches
        
        if not patch_indices:
            continue
        
        # Get fault parameter positions
        fault_start = multifault_solver.fault_indexes[fault_name][0]
        
        # Get patch centers
        centers = np.array(fault.getcenters())[patch_indices]
        
        # Convert patch centers to lon/lat
        xc, yc = centers[:, 0], centers[:, 1]
        lonc, latc = fault.xy2ll(xc, yc)
        lonc, latc = np.radians(lonc), np.radians(latc)
        
        # Get reference strike
        reference_strike_deg = params.get('reference_strike', 0.0)
        
        # Determine motion sign
        motion_sense = params.get('motion_sense', 'dextral')
        motion_sign = determine_motion_sign(motion_sense)
        
        # Pre-calculate Euler matrix
        euler_mat = calculate_euler_matrix_for_points(lonc, latc)
        
        # Calculate strike-slip projection to each patch's own strike
        euler_strike = project_euler_to_strike(euler_mat, fault, patch_indices, 
                                              reference_strike_deg, len(patch_indices))
        
        constraint_info.append({
            'fault_name': fault_name,
            'fault': fault,
            'params': params,
            'fault_start': fault_start,
            'patch_indices': np.array(patch_indices),
            'euler_strike': euler_strike,
            'motion_sign': motion_sign,
            'num_patches': len(patch_indices)
        })
        
        total_constraints += len(patch_indices)
    
    if total_constraints == 0:
        return None, None
    
    # Initialize constraint matrices
    total_params = multifault_solver.lsq_parameters
    A_ineq = np.zeros((total_constraints, total_params))
    b_ineq = np.zeros(total_constraints)
    
    constraint_row = 0
    
    # Process each fault with constraints
    for info in constraint_info:
        fault_name = info['fault_name']
        params = info['params']
        fault_start = info['fault_start']
        patch_indices = info['patch_indices']
        euler_strike = info['euler_strike']
        motion_sign = info['motion_sign']  # +1 for dextral, -1 for sinistral
        num_patches = info['num_patches']
        
        # Set slip coefficients for all patches at once
        # For dextral: +1 * slip_strike
        # For sinistral: -1 * slip_strike (entire constraint multiplied by -1)
        slip_indices = fault_start + np.array(patch_indices, dtype=int)
        constraint_rows = np.arange(constraint_row, constraint_row + num_patches)
        A_ineq[constraint_rows, slip_indices] = motion_sign  # +1 or -1
        
        # Get block parameters
        block_types = params['block_types']
        blocks_standard = params['blocks_standard']
        
        # Process each block
        for block_idx, (block_type, block_data) in enumerate(zip(block_types, blocks_standard)):
            
            # For the Euler terms: first block positive, second block negative
            # Then multiply by motion_sign for the entire constraint
            if block_idx == 0:
                block_sign = motion_sign  # +motion_sign for first block
            else:
                block_sign = -motion_sign  # -motion_sign for second block
            
            if block_type == 'dataset':
                # Get dataset name and check for estimated parameters
                dataset_name = block_data
                
                if dataset_name not in dataset_name_to_obj:
                    raise ValueError(f"Dataset '{dataset_name}' not found in datasets")
                
                # Get parameter indices from transform_indices
                euler_indices = None
                if dataset_name in transform_indices:
                    dataset_transforms = transform_indices[dataset_name]
                    euler_indices = dataset_transforms.get('eulerrotation')
                
                if euler_indices is not None:
                    # Estimated parameters - add to constraint matrix
                    start_idx, end_idx = euler_indices
                    start_idx += fault_start  # Adjust to global index
                    end_idx += fault_start  # Adjust to global index
                    for k in range(3):  # wx, wy, wz components
                        if start_idx + k < end_idx:
                            A_ineq[constraint_rows, start_idx + k] += block_sign * euler_strike[:, k]
                
            elif block_type in ['euler_pole', 'euler_vector']:
                # Fixed parameters - add to RHS
                if block_type == 'euler_pole':
                    lon_pole, lat_pole, omega = block_data
                    euler_vector = convert_euler_pole_to_vector(lat_pole, lon_pole, omega)
                else:  # euler_vector
                    euler_vector = np.array(block_data)
                
                # Calculate fixed velocities
                fixed_velocities = np.sum(euler_strike * euler_vector[None, :], axis=1)
                b_ineq[constraint_rows] -= block_sign * fixed_velocities
        
        constraint_row += num_patches
    
    return A_ineq, b_ineq


def apply_euler_inequality_constraints(multifault_solver, euler_config, all_datasets, verbose=False):
    """
    Apply Euler pole inequality constraints to the multifault solver.
    
    Parameters:
    -----------
    multifault_solver : multifaultsolve_boundLSE
        The multi-fault solver object
    euler_config : dict
        Parsed Euler constraints configuration
    verbose : bool
        Enable verbose output
    """
    
    if not euler_config.get('enabled', False):
        if verbose:
            print("Euler inequality constraints are disabled.")
        return
    
    # Generate Euler inequality constraints
    A_ineq, b_ineq = generate_euler_inequality_constraints(multifault_solver, euler_config, all_datasets)
    
    if A_ineq is None or A_ineq.size == 0:
        if verbose:
            print("No Euler inequality constraints generated.")
        return
    
    # Add to existing inequality constraints
    if multifault_solver.A_ueq is None:
        multifault_solver.A_ueq = A_ineq
        multifault_solver.b_ueq = b_ineq
    else:
        multifault_solver.A_ueq = np.vstack([multifault_solver.A_ueq, A_ineq])
        multifault_solver.b_ueq = np.hstack([multifault_solver.b_ueq, b_ineq])
    
    if verbose:
        print(f"Applied {A_ineq.shape[0]} Euler inequality constraints to solver.")
        configured_faults = euler_config.get('configured_faults', [])
        print(f"Constrained faults: {configured_faults}")
        
        # Print constraint details
        for fault_name, params in euler_config['faults'].items():
            if fault_name in configured_faults:
                motion_sense = params.get('motion_sense', 'dextral')
                reference_strike = params.get('reference_strike', 0.0)
                print(f"  {fault_name}: {motion_sense} motion, reference strike = {reference_strike}°")
        
        # Print transform information if available
        faults = multifault_solver.faults
        transform_indices = {}
        if len(faults) > 0 and hasattr(faults[0], 'transform_indices'):
            transform_indices = faults[0].transform_indices
        elif hasattr(multifault_solver, 'transform_indices'):
            transform_indices = multifault_solver.transform_indices
        
        if transform_indices:
            print("Transform parameter indices found:")
            datasets_used = set()
            for fault_name, params in euler_config['faults'].items():
                if fault_name in configured_faults:
                    for i, block_type in enumerate(params['block_types']):
                        if block_type == 'dataset':
                            datasets_used.add(params['blocks_standard'][i])
            
            for dataset_name, transforms in transform_indices.items():
                if dataset_name in datasets_used:
                    print(f"  {dataset_name}: {transforms}")


# Example usage and configuration validation
def validate_euler_inequality_config(euler_config, faultnames, dataset_names):
    """
    Validate Euler inequality constraints configuration.
    
    Parameters:
    -----------
    euler_config : dict
        Euler constraints configuration
    faultnames : list
        List of fault names
    dataset_names : list
        List of dataset names
        
    Returns:
    --------
    bool
        True if configuration is valid
    """
    
    if not euler_config.get('enabled', False):
        return True
    
    faults_config = euler_config.get('faults', {})
    
    for fault_name, params in faults_config.items():
        if fault_name not in faultnames:
            raise ValueError(f"Fault '{fault_name}' not found in solver faults")
        
        # Validate required parameters
        if 'blocks' not in params or 'block_types' not in params:
            raise ValueError(f"Missing required parameters for fault '{fault_name}'")
        
        block_types = params['block_types']
        blocks = params['blocks_standard']
        
        if len(block_types) != 2 or len(blocks) != 2:
            raise ValueError(f"Fault '{fault_name}' must have exactly 2 blocks")
        
        # Validate dataset references
        for i, (block_type, block_data) in enumerate(zip(block_types, blocks)):
            if block_type == 'dataset' and block_data not in dataset_names:
                raise ValueError(f"Dataset '{block_data}' for fault '{fault_name}' not found")
        
        # Validate motion_sense
        motion_sense = params.get('motion_sense', 'dextral')
        if motion_sense not in ['dextral', 'sinistral', 'right_lateral', 'left_lateral']:
            raise ValueError(f"Invalid motion_sense '{motion_sense}' for fault '{fault_name}'")
        
        # Validate reference_strike
        reference_strike = params.get('reference_strike', 0.0)
        if not isinstance(reference_strike, (int, float)):
            raise ValueError(f"reference_strike for fault '{fault_name}' must be numeric")
    
    return True


def example_euler_inequality_config():
    """
    Example configuration for Euler inequality constraints.
    """
    return {
        'enabled': True,
        'defaults': {
            'reference_strike': 45.0,  # degrees, default reference direction
            'motion_sense': 'dextral'  # default motion sense
        },
        'faults': {
            # Fault with two datasets (both estimated)
            'HH_Main': {
                'block_types': ['dataset', 'dataset'],
                'blocks': ['GPS_South_China', 'GPS_Burma'], 
                'blocks_standard': ['GPS_South_China', 'GPS_Burma'],
                'block_names': ['South_China_Block', 'Burma_Block'],
                'reference_strike': 30.0,  # degrees
                'motion_sense': 'dextral',
                'apply_to_patches': None  # All patches
            },
            
            # Fault with dataset and fixed Euler pole
            'HH_North': {
                'block_types': ['dataset', 'euler_pole'],
                'blocks': ['GPS_Tibet', [26.8, 98.5, 0.38]],
                'blocks_standard': ['GPS_Tibet', [0.4678, 1.7191, 1.2042e-08]],  # converted to rad, rad, rad/yr
                'block_names': ['Tibetan_Plateau', 'Fixed_Reference'],
                'reference_strike': 120.0,  # degrees
                'motion_sense': 'sinistral',
                'apply_to_patches': [0, 1, 2, 3, 4]  # Specific patches
            },
            
            # Fault with two fixed Euler vectors
            'HH_South': {
                'block_types': ['euler_vector', 'euler_vector'],
                'blocks': [[1.2e-8, -0.8e-8, 2.1e-8], [0.8e-8, 1.5e-8, -0.6e-8]],
                'blocks_standard': [[1.2e-8, -0.8e-8, 2.1e-8], [0.8e-8, 1.5e-8, -0.6e-8]],
                'block_names': ['Indian_Plate', 'Sunda_Block'],
                'reference_strike': 60.0,  # degrees
                'motion_sense': 'dextral'
            }
        },
        'configured_faults': ['HH_Main', 'HH_North', 'HH_South']
    }


# ------------------------For Visualization------------------------#
def assign_euler_slip_to_fault(fault, euler_config, fault_name, euler_params1, euler_params2):
    """
    根据两个欧拉参数计算走滑并赋值给断层对象。
    
    Parameters:
    -----------
    fault : object
        断层对象
    euler_config : dict
        欧拉约束配置
    fault_name : str
        断层名称
    euler_params1 : np.ndarray or list
        第一个块的欧拉参数 [wx, wy, wz] (rad/year)
    euler_params2 : np.ndarray or list
        第二个块的欧拉参数 [wx, wy, wz] (rad/year)
        
    Returns:
    --------
    strike_slip : np.ndarray
        计算得到的走滑分量
    """
    
    if fault_name not in euler_config['faults']:
        raise ValueError(f"Fault '{fault_name}' not found in euler_config")
    
    params = euler_config['faults'][fault_name]
    
    # 获取应用patch索引
    apply_patches = params.get('apply_to_patches', None)
    if apply_patches is None:
        patch_indices = list(range(len(fault.patch)))
    else:
        patch_indices = apply_patches
    
    # 获取patch中心点
    centers = np.array(fault.getcenters())[patch_indices]
    xc, yc = centers[:, 0], centers[:, 1]
    lonc, latc = fault.xy2ll(xc, yc)
    lonc, latc = np.radians(lonc), np.radians(latc)
    # 计算欧拉矩阵
    euler_mat = calculate_euler_matrix_for_points(lonc, latc)
    
    # 获取参考走向
    reference_strike_deg = params.get('reference_strike', 0.0)
    
    # 计算每个块的走滑速度
    euler_strike1 = project_euler_to_strike(euler_mat, fault, patch_indices, 
                                           reference_strike_deg, len(patch_indices))
    euler_strike2 = project_euler_to_strike(euler_mat, fault, patch_indices, 
                                           reference_strike_deg, len(patch_indices))
    
    # 转换为numpy数组
    euler_params1 = np.array(euler_params1)
    euler_params2 = np.array(euler_params2)
    
    # 计算速度差
    vel1 = np.sum(euler_strike1 * euler_params1[None, :], axis=1)
    vel2 = np.sum(euler_strike2 * euler_params2[None, :], axis=1)
    
    # 计算走滑分量：vel1 - vel2
    strike_slip_selected = vel1 - vel2
    
    # 初始化所有patch的走滑为0
    strike_slip_full = np.zeros(len(fault.patch))
    
    # 将计算值赋给选定的patch
    for i, patch_idx in enumerate(patch_indices):
        strike_slip_full[patch_idx] = strike_slip_selected[i]
    
    # 将走滑分量赋值给断层对象
    fault.slip[patch_indices, 0] = strike_slip_full  # 走滑分量赋值给断层对象的slip属性
    
    return strike_slip_selected


def visualize_euler_slip(fault, euler_config, fault_name, euler_params1, euler_params2, 
                        plot_kwargs=None, save_path=None):
    """
    计算并可视化欧拉参数预测的走滑分布。
    
    Parameters:
    -----------
    fault : object
        断层对象
    euler_config : dict
        欧拉约束配置
    fault_name : str
        断层名称
    euler_params1 : np.ndarray or list
        第一个块的欧拉参数 [wx, wy, wz] (rad/year)
    euler_params2 : np.ndarray or list
        第二个块的欧拉参数 [wx, wy, wz] (rad/year)
    plot_kwargs : dict, optional
        传递给fault.plot()的额外参数
    save_path : str, optional
        保存图像的路径
        
    Returns:
    --------
    strike_slip : np.ndarray
        计算得到的走滑分量
    """
    
    # 计算并赋值走滑
    strike_slip = assign_euler_slip_to_fault(fault, euler_config, fault_name, 
                                           euler_params1, euler_params2)
    
    # 设置默认绘图参数
    default_plot_kwargs = {
        'slip': 'strikeslip'
    }
    
    if plot_kwargs:
        default_plot_kwargs.update(plot_kwargs)
    
    # 绘制断层
    fig = fault.plot(**default_plot_kwargs)
    
    # 打印统计信息
    params = euler_config['faults'][fault_name]
    motion_sense = params.get('motion_sense', 'dextral')
    reference_strike = params.get('reference_strike', 0.0)
    
    print(f"\n=== Euler Strike-slip Analysis for {fault_name} ===")
    print(f"Motion sense: {motion_sense}")
    print(f"Reference strike: {reference_strike}°")
    print(f"Applied patches: {len(strike_slip)}")
    print(f"Strike-slip range: [{np.min(strike_slip):.2e}, {np.max(strike_slip):.2e}] m/year")
    print(f"Mean strike-slip: {np.mean(strike_slip):.2e} m/year")
    
    return strike_slip


def compare_euler_scenarios(fault, euler_config, fault_name, scenarios, 
                           plot_kwargs=None, save_dir=None):
    """
    比较多种欧拉参数组合的走滑预测结果。
    
    Parameters:
    -----------
    fault : object
        断层对象
    euler_config : dict
        欧拉约束配置
    fault_name : str
        断层名称
    scenarios : dict
        多个场景的欧拉参数
        格式: {'scenario_name': [euler_params1, euler_params2], ...}
    plot_kwargs : dict, optional
        绘图参数
    save_dir : str, optional
        保存目录
        
    Returns:
    --------
    results : dict
        各场景的走滑结果
    """
    
    import matplotlib.pyplot as plt
    
    results = {}
    
    for scenario_name, (euler_params1, euler_params2) in scenarios.items():
        print(f"\n--- Processing scenario: {scenario_name} ---")
        
        # 计算走滑
        strike_slip = assign_euler_slip_to_fault(fault, euler_config, fault_name,
                                               euler_params1, euler_params2)
        results[scenario_name] = strike_slip.copy()
        
        # 绘图参数
        plot_params = {
            'slip': 'strikeslip',
            'colorbar': True,
            'title': f'{fault_name} - {scenario_name}'
        }
        if plot_kwargs:
            plot_params.update(plot_kwargs)
        
        # 绘制
        fig = fault.plot(**plot_params)
        
        # 保存
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{fault_name}_{scenario_name}_slip.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    return results


def create_euler_slip_summary(fault, euler_config, fault_name, euler_params1, euler_params2):
    """
    创建欧拉走滑预测的详细摘要。
    
    Parameters:
    -----------
    fault : object
        断层对象
    euler_config : dict
        欧拉约束配置
    fault_name : str
        断层名称
    euler_params1, euler_params2 : array-like
        两个块的欧拉参数
        
    Returns:
    --------
    summary : dict
        详细摘要信息
    """
    
    # 计算走滑
    strike_slip = assign_euler_slip_to_fault(fault, euler_config, fault_name,
                                           euler_params1, euler_params2)
    
    params = euler_config['faults'][fault_name]
    
    # 获取patch信息
    apply_patches = params.get('apply_to_patches', None)
    if apply_patches is None:
        patch_indices = list(range(len(fault.patch)))
    else:
        patch_indices = apply_patches
    
    # 创建摘要
    summary = {
        'fault_name': fault_name,
        'motion_sense': params.get('motion_sense', 'dextral'),
        'reference_strike': params.get('reference_strike', 0.0),
        'euler_params1': np.array(euler_params1),
        'euler_params2': np.array(euler_params2),
        'applied_patches': len(patch_indices),
        'total_patches': len(fault.patch),
        'strike_slip_stats': {
            'min': np.min(strike_slip),
            'max': np.max(strike_slip),
            'mean': np.mean(strike_slip),
            'std': np.std(strike_slip),
            'median': np.median(strike_slip)
        }
    }
    
    return summary