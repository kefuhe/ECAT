"""
Batch extraction of orthogonal profiles along a fault trace and step function fitting to estimate fault-perpendicular displacement (step height).

Features:
- Sample a center point every `spacing` (e.g., 1 km) along the fault trace.
- At each center, extract a profile orthogonal to the fault trace, with length `profile_length` and width `profile_width`.
- Fit a step function to each profile, with the initial step position at the fault intersection (can be adjusted).
- Output the step position and step height (fault-perpendicular displacement) for each profile.

Dependencies:
- numpy, scipy, shapely
- Your ProfileAnalyzer class (must support xy2ll, ll2xy, define_profile, extract_from_matrix)

See usage example at the end of this file.
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from scipy.optimize import curve_fit
from shapely.geometry import LineString
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from shapely.geometry import LineString
from math import erf, sqrt, pi

@dataclass
class ProfileConfig:
    """
    Configuration for extracting fault-perpendicular profiles.

    Attributes:
        profile_length: Length of each profile (in fault-perpendicular direction).
        profile_width: Width of each profile (swath width).
        profile_spacing: Sampling interval along the profile.
        interpolation_method: Interpolation method for data extraction ('nearest', 'linear', etc.).
        envelope_method: Method for envelope calculation ('minmax', etc.).
        envelope_percentiles: Percentiles for envelope calculation (e.g., (1, 99)).
        outlier_threshold: Threshold for outlier removal.
        outlier_method: Outlier removal method ('absolute', etc.).
        outlier_removal: Whether to remove outliers.
        swath_method: Swath extraction method ('optimized', etc.).
        exclude_near_fault_range: Range to exclude near the fault (tuple: min, max).
        distance_range: Range of distances to include in the profile (tuple: min, max).
    """
    profile_length: float = 5.0
    profile_width: float = 1.0
    profile_spacing: float = 0.2
    interpolation_method: str = 'nearest'
    envelope_method: str = 'minmax'
    envelope_percentiles: Tuple[int, int] = (1, 99)
    outlier_threshold: float = 1000.0
    outlier_method: str = 'absolute'
    outlier_removal: bool = False
    swath_method: str = 'optimized'
    exclude_near_fault_range: Optional[Tuple[float, float]] = None
    distance_range: Optional[Tuple[float, float]] = None

@dataclass
class FitConfig:
    """
    Configuration for profile fitting.

    Attributes:
        fit_mode: Fitting mode ('backslip', 'step_linear', 'backslip_C').
        step_range: Allowed range for step position (centered at fault).
        S0_range: Allowed range for step height parameter.
        d_range: Allowed range for locked depth parameter.
        C_range: Allowed range for C parameter (backslip_C mode).
        d2_range: Allowed range for d2 parameter (backslip_C mode).
    """
    fit_mode: str = "backslip"
    step_range: float = 2.0
    S0_range: Tuple[float, float] = (-10, 10)
    d_range: Tuple[float, float] = (0.01, 30)
    C_range: Tuple[float, float] = (-10, 10)
    d2_range: Tuple[float, float] = (0.01, 30)

@dataclass
class BatchConfig:
    """
    Configuration for batch profile extraction and fitting.

    Attributes:
        spacing: Sampling interval along the fault trace.
        max_workers: Number of parallel workers for extraction/fitting.
        i_range: Index range of profiles to process (tuple: start, end).
        indices: Specific indices of profiles to process.
        save_path: Path to save the batch results (pickle file).
    """
    spacing: float = 1.0
    max_workers: int = 4
    i_range: Optional[Tuple[int, int]] = None
    indices: Optional[List[int]] = None
    save_path: Optional[str] = None

def sample_fault_profiles(
    fault_points: List[Tuple[float, float]],
    analyzer,
    data: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    profile_cfg: ProfileConfig = ProfileConfig(),
    fit_cfg: FitConfig = FitConfig(),
    batch_cfg: BatchConfig = BatchConfig(),
    fault_points_type: str = 'lonlat',  # 'lonlat' or 'xy'
    fixed_x0_dict: Optional[Dict[int, float]] = None,
    x0_mode: str = 'fixed'  # 'fixed' or 'init'
) -> List[Dict[str, Any]]:
    """
    Batch extract and fit profiles along a fault trace.

    Args:
        fault_points: List of fault trace points. Type controlled by fault_points_type.
        analyzer: ProfileAnalyzer instance.
        data: 2D data array.
        x_coords, y_coords: grid coordinates (always longitude/latitude or projected grid, NOT affected by fault_points_type).
        profile_cfg: ProfileConfig instance.
        fit_cfg: FitConfig instance.
        batch_cfg: BatchConfig instance.
        fault_points_type: 'lonlat' if fault_points are longitude/latitude, 'xy' if already projected.
        fixed_x0_dict: Optional dictionary mapping profile indices to fixed x0 values.

    Returns:
        List of profile fit results.
    """
    if fault_points_type == 'lonlat':
        fault_points_xy = [analyzer.ll2xy(lon, lat) for lon, lat in fault_points]
    else:
        fault_points_xy = fault_points
    fault_line = LineString(fault_points_xy)
    total_length = fault_line.length
    n_samples = int(total_length // batch_cfg.spacing) + 1

    if batch_cfg.indices is not None:
        sample_indices = batch_cfg.indices
    else:
        start, end = batch_cfg.i_range if batch_cfg.i_range else (0, n_samples)
        sample_indices = list(range(start, end))
    sample_distances = [i * batch_cfg.spacing for i in sample_indices]

    def process_one(i, d):
        profile_data, normal_vec = extract_profile(
            fault_line, analyzer, data, x_coords, y_coords, d,
            profile_cfg.profile_length, profile_cfg.profile_width, profile_cfg.profile_spacing,
            profile_cfg.interpolation_method, profile_cfg.envelope_method, profile_cfg.envelope_percentiles,
            profile_cfg.outlier_threshold, profile_cfg.outlier_method, profile_cfg.outlier_removal, profile_cfg.swath_method
        )
        if profile_data is None:
            return i, None
        distances = getattr(profile_data, 'swath_distances', None)
        values = getattr(profile_data, 'swath_values', None)
        if distances is None or values is None:
            return i, None
        distances = np.asarray(distances, dtype=float)
        values = np.asarray(values, dtype=float)
        if profile_cfg.distance_range is not None:
            mask = (distances >= profile_cfg.distance_range[0]) & (distances <= profile_cfg.distance_range[1])
            distances = distances[mask]
            values = values[mask]
        if profile_cfg.exclude_near_fault_range is not None:
            mask = (distances < profile_cfg.exclude_near_fault_range[0]) | (distances > profile_cfg.exclude_near_fault_range[1])
            distances = distances[mask]
            values = values[mask]
        fit_result = fit_profile(
            distances, values,
            mode=fit_cfg.fit_mode,
            fault_pos=0.0,
            step_range=fit_cfg.step_range,
            S0_range=fit_cfg.S0_range,
            d_range=fit_cfg.d_range,
            C_range=fit_cfg.C_range,
            d2_range=fit_cfg.d2_range,
            fixed_x0=fixed_x0_dict.get(i, None) if fixed_x0_dict else None,
            x0_mode=x0_mode,
            profile_index=i
        )
        if fit_result is None:
            return i, None
        fit_info = {
            'fit_mode': fit_result['fit_mode'],
            'fit_params': fit_result['fit_params'],
            'ith_profile': i,
            'distance_along_fault': d,
            'center_xy': (fault_line.interpolate(d).x, fault_line.interpolate(d).y),
            'center_lonlat': analyzer.xy2ll(fault_line.interpolate(d).x, fault_line.interpolate(d).y),
            "normal_vector": normal_vec,
            'step_position': fit_result['step_position'],
            'step_value': fit_result['step_value'],
            'damage_zone_width': fit_result['damage_zone_width'],
            'left_point': fit_result.get('left_point', None),
            'right_point': fit_result.get('right_point', None),
            'profile_distances': distances,
            'profile_values': call_fit_func(fit_result['fit_mode'], fit_result['fit_params'], distances),
            'profile_data': profile_data
        }
        x = fit_info['profile_distances']
        if profile_cfg.distance_range is not None:
            mark = (x >= profile_cfg.distance_range[0]) & (x <= profile_cfg.distance_range[1])
            x = x[mark]
        if profile_cfg.exclude_near_fault_range is not None:
            mark = (x < profile_cfg.exclude_near_fault_range[0]) | (x > profile_cfg.exclude_near_fault_range[1])
            x = x[mark]
        y = call_fit_func(fit_info['fit_mode'], fit_info['fit_params'], x)
        profile_data.profile_values = y
        profile_data.profile_distances = x
        fit_info['profile_values'] = y
        fit_info['profile_distances'] = x
        return i, fit_info

    results_dict = {}
    # Parallel processing
    if len(sample_indices) > 1 and batch_cfg.max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=batch_cfg.max_workers) as executor:
            futures = [
                executor.submit(process_one, i, d)
                for i, d in zip(sample_indices, sample_distances)
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Extracting profiles (parallel)"):
                i, r = f.result()
                if r is not None:
                    results_dict[i] = r
    else:
        # Serial processing
        for i, d in zip(sample_indices, sample_distances):
            idx, r = process_one(i, d)
            if r is not None:
                results_dict[idx] = r

    results = [results_dict[i] for i in sample_indices if i in results_dict]
    if batch_cfg.save_path:
        with open(batch_cfg.save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {batch_cfg.save_path}")
    return results

# --- Fit Functions ---

def step_linear_func(x, x0, a1, b1, a2, b2):
    """Piecewise linear step function."""
    return np.where(
        x < x0,
        a1 + b1 * x,
        a2 + b2 * x
    )

def backslip_func(x, x0, S, d, a, b):
    """Backslip model: -S/pi * arctan((x-x0)/d) + a + b*x"""
    return -S/np.pi * np.arctan((x - x0)/d) + a + b * x

def backslip_C_func(x, x0, S, d1, C, d2, a, b):
    """Backslip model with C term."""
    x_shift = x - x0
    d2_safe = np.where(np.abs(d2) < 1e-6, 1e-6, d2)
    return (
        -S/np.pi * np.arctan(x_shift/d1)
        + C * (1/np.pi * np.arctan(x_shift/d2_safe) - np.heaviside(x_shift, 0.5))
        + a + b * x_shift
    )

def erf_profile(x, a, b, c, w_s, eps_el):
    """
    Erf + linear dislocation profile model.

    Model:
        y(x) = a + b * erf((x - c) / (sqrt(2) * w_s)) + eps_el * x

    Parameters:
        x      : across-fault distance [m]
        a      : intercept [m]
        b      : total dislocation amplitude D_GI [m]
        c      : fault center position [m]
        w_s    : shear-band width [m]
        eps_el : near-field elastic gradient [m/m]

    Returns:
        y      : displacement profile [m]
    """
    z = (x - c) / (sqrt(2.0) * w_s)
    v_erf = np.vectorize(erf)(z)
    return a + b * v_erf + eps_el * x

def erf_profile_derivative(x, a, b, c, w_s, eps_el):
    """
    First derivative of erf + linear profile.

    dy/dx = eps_el + (b / (sqrt(pi) * w_s)) * exp(-((x - c) / w_s)^2)

    Parameters:
        x      : across-fault distance [m]
        a, b, c, w_s, eps_el : model parameters

    Returns:
        dy/dx  : strain profile [m/m]
    """
    return eps_el + (b / (sqrt(pi) * w_s)) * np.exp(-((x - c) / w_s) ** 2)

def call_fit_func(fit_mode: str, fit_params: List[float], x: np.ndarray) -> np.ndarray:
    """Call the corresponding fit function based on fit_mode and fit_params."""
    if fit_mode == "step_linear":
        return step_linear_func(x, *fit_params)
    elif fit_mode == "backslip":
        return backslip_func(x, *fit_params)
    elif fit_mode == "erf_linear":
        return erf_profile(x, *fit_params)
    elif fit_mode == "backslip_C":
        return backslip_C_func(x, *fit_params)
    else:
        raise ValueError(f"Unknown fit_mode: {fit_mode}")

# --- Profile Fitting ---

def fit_profile(
    distances: np.ndarray,
    values: np.ndarray,
    mode: str = "backslip",
    fault_pos: float = 0.0,
    step_range: float = 2.0,
    S0_range: Tuple[float, float] = (-10, 10),
    d_range: Tuple[float, float] = (0.01, 30),
    C_range: Tuple[float, float] = (-10, 10),
    d2_range: Tuple[float, float] = (0.01, 30),
    fixed_x0: Optional[float] = None,
    x0_mode: str = "fixed", # 'fixed' or 'init'
    profile_index: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Fit a step function or backslip model to the given profile data.

    Returns:
        dict with fit_mode, fit_params, step_position, step_value, or None if fitting fails.
    """
    mask = ~np.isnan(distances) & ~np.isnan(values)
    x = distances[mask]
    y = values[mask]
    if len(x) < 6:
        return None

    if mode == "step_linear":
        a1 = np.nanmean(y[x < fault_pos]) if np.any(x < fault_pos) else np.nanmean(y)
        b1 = 0
        a2 = np.nanmean(y[x >= fault_pos]) if np.any(x >= fault_pos) else np.nanmean(y)
        b2 = 0
        x0_bounds = (fault_pos - step_range, fault_pos + step_range)
        try:
            if fixed_x0 is not None and x0_mode == "fixed":
                def fit_func(x, a1, b1, a2, b2):
                    return step_linear_func(x, fixed_x0, a1, b1, a2, b2)
                p0 = [a1, b1, a2, b2]
                bounds = (
                    [-np.inf, -np.inf, -np.inf, -np.inf],
                    [np.inf, np.inf, np.inf, np.inf]
                )
                popt, _ = curve_fit(fit_func, x, y, p0=p0, bounds=bounds)
                popt_full = [fixed_x0] + list(popt)
            elif fixed_x0 is not None and x0_mode == "init":
                def fit_func(x, x0, a1, b1, a2, b2):
                    return step_linear_func(x, x0, a1, b1, a2, b2)
                p0 = [fixed_x0, a1, b1, a2, b2]
                bounds = (
                    [fixed_x0-step_range, -np.inf, -np.inf, -np.inf, -np.inf],
                    [fixed_x0+step_range, np.inf, np.inf, np.inf, np.inf]
                )
                popt, _ = curve_fit(fit_func, x, y, p0=p0, bounds=bounds)
                popt_full = list(popt)
            else:
                def fit_func(x, x0, a1, b1, a2, b2):
                    return step_linear_func(x, x0, a1, b1, a2, b2)
                popt, _ = curve_fit(
                    fit_func,
                    x, y,
                    p0=[fault_pos, a1, b1, a2, b2],
                    bounds=(
                        [x0_bounds[0], -np.inf, -np.inf, -np.inf, -np.inf],
                        [x0_bounds[1], np.inf, np.inf, np.inf, np.inf]
                    )
                )
                popt_full = list(popt)
            x0_fit, a1_fit, b1_fit, a2_fit, b2_fit = popt_full
            x_right = np.nanmin(x[x >= x0_fit])
            x_left = np.nanmax(x[x < x0_fit])
            step_value = (a2_fit + b2_fit * x_right) - (a1_fit + b1_fit * x_left)
            fit_params = [x0_fit, a1_fit, b1_fit, a2_fit, b2_fit]

            # Extract two extreme values
            return {
                'fit_mode': 'step_linear',
                'fit_params': fit_params,
                'step_position': x0_fit,
                'step_value': step_value,
                # distance between extreme values
                'damage_zone_width': x_right - x_left,
                'left_point': (x_left, a1_fit + b1_fit * x_left),
                'right_point': (x_right, a2_fit + b2_fit * x_right)
            }
        except Exception as e:
            if profile_index is not None:
                print(f"Step fit failed for profile {profile_index}: {e}")
            else:
                print(f"Step fit failed: {e}")
            return None

    elif mode == "backslip":
        x_mean = np.nanmean(x)
        x0_0 = fault_pos
        S0 = (np.nanmax(y) - np.nanmin(y))
        d0 = 1.0
        a0 = np.nanmean(y)
        b0 = 0
        try:
            if fixed_x0 is not None and x0_mode == "fixed":
                fixed_x0 = fixed_x0 - x_mean
                def fit_func(x, S, d, a, b):
                    return backslip_func(x, fixed_x0, S, d, a, b)
                p0 = [S0, d0, a0, b0]
                bounds = (
                    [S0_range[0], d_range[0], -np.inf, -np.inf],
                    [S0_range[1], d_range[1], np.inf, np.inf]
                )
                popt, _ = curve_fit(fit_func, x-x_mean, y, p0=p0, bounds=bounds)
                popt_full = [fixed_x0] + list(popt)
            elif fixed_x0 is not None and x0_mode == "init":
                fixed_x0 = fixed_x0 - x_mean
                def fit_func(x, x0, S, d, a, b):
                    return backslip_func(x, x0, S, d, a, b)
                p0 = [fixed_x0, S0, d0, a0, b0]
                bounds=(
                    [fixed_x0 - step_range, S0_range[0], d_range[0], -np.inf, -np.inf],
                    [fixed_x0 + step_range, S0_range[1], d_range[1], np.inf, np.inf]
                )
                popt, _ = curve_fit(fit_func, x-x_mean, y, p0=p0, bounds=bounds)
                popt_full = list(popt)
            else:
                popt, _ = curve_fit(
                    backslip_func,
                    x-x_mean, y,
                    p0=[x0_0, S0, d0, a0, b0],
                    bounds=(
                        [x0_0 - step_range, S0_range[0], d_range[0], -np.inf, -np.inf],
                        [x0_0 + step_range, S0_range[1], d_range[1], np.inf, np.inf]
                    )
                )
                popt_full = list(popt)
            x_fit, S_fit, d_fit, a_fit, b_fit = popt_full
            fit_params = [x_fit + x_mean, S_fit, d_fit, a_fit - b_fit * x_mean, b_fit]

            # Two extreme values
            C = (np.pi*d_fit*b_fit)/S_fit
            if 0 < C < 1:
                delta = d_fit*np.sqrt(1/C - 1)
            else:
                delta = 0
            x_left = x_fit - delta + x_mean
            x_right = x_fit + delta + x_mean
            if C < 0 or C > 1:
                step_value = np.nan
            else:
                y_right_value = backslip_func(x_right - x_mean, *popt_full)
                y_left_value = backslip_func(x_left - x_mean, *popt_full)
                step_value = y_right_value - y_left_value

            # Calculate profile bounds
            profile_x_min = np.nanmin(x)
            profile_x_max = np.nanmax(x)

            # Check if extrema are at the edges
            is_left_at_edge = np.isclose(x_left, profile_x_min, atol=1e-3)
            is_right_at_edge = np.isclose(x_right, profile_x_max, atol=1e-3)

            if is_left_at_edge and is_right_at_edge:
                delta_slope = max_curvature_position(d_fit)
                x_left = x_fit - delta_slope + x_mean
                x_right = x_fit + delta_slope + x_mean
                y_left_value = backslip_func(x_left - x_mean, *popt_full)
                y_right_value = backslip_func(x_right - x_mean, *popt_full)
                step_value = y_right_value - y_left_value
            return {
                'fit_mode': 'backslip',
                'fit_params': fit_params,
                'step_position': x_fit + x_mean,
                'step_value': step_value,
                'damage_zone_width': x_right - x_left,
                'left_point': (x_left, y_left_value),
                'right_point': (x_right, y_right_value)
            }
        except Exception as e:
            if profile_index is not None:
                print(f"Backslip fit failed for profile {profile_index}: {e}")
            else:
                print(f"Backslip fit failed: {e}")
            return None
    elif mode == "erf_linear":
        x_mean = np.nanmean(x)
        a0 = np.nanmean(y)
        b0 = (np.nanmax(y) - np.nanmin(y)) / 2
        c0 = fault_pos
        w_s0 = 1.0
        eps_el0 = 0.0
        try:
            if fixed_x0 is not None and x0_mode == "fixed":
                fixed_x0 = fixed_x0 - x_mean
                def fit_func(x, a, b, w_s, eps_el):
                    return erf_profile(x, a, b, fixed_x0, w_s, eps_el)
                p0 = [a0, b0, w_s0, eps_el0]
                bounds = (
                    [-np.inf, S0_range[0], 1e-3, -np.inf],
                    [np.inf, S0_range[1], np.inf, np.inf]
                )
                popt, _ = curve_fit(fit_func, x - x_mean, y, p0=p0, bounds=bounds)
                popt_full = list(popt)[:2] + [fixed_x0] + list(popt)[2:]
            elif fixed_x0 is not None and x0_mode == "init":
                fixed_x0 = fixed_x0 - x_mean
                def fit_func(x, a, b, c, w_s, eps_el):
                    return erf_profile(x, a, b, c, w_s, eps_el)
                p0 = [a0, b0, fixed_x0, w_s0, eps_el0]
                bounds = (
                    [-np.inf, S0_range[0], fixed_x0 - step_range, 1e-3, -np.inf],
                    [np.inf, S0_range[1], fixed_x0 + step_range, np.inf, np.inf]
                )
                popt, _ = curve_fit(fit_func, x - x_mean, y, p0=p0, bounds=bounds)
                popt_full = list(popt)
            else:
                popt, _ = curve_fit(
                    erf_profile,
                    x - x_mean, y,
                    p0=[a0, b0, c0, w_s0, eps_el0],
                    bounds=(
                        [-np.inf, S0_range[0], c0 - step_range, 1e-3, -np.inf],
                        [np.inf, S0_range[1], c0 + step_range, np.inf, np.inf]
                    )
                )
                popt_full = list(popt)

            a_fit, b_fit, c_fit, w_s_fit, eps_el_fit = popt
            fit_params = [a_fit, b_fit, c_fit + x_mean, w_s_fit, eps_el_fit]
            return {
                'fit_mode': 'erf_linear',
                'fit_params': fit_params,
                'step_position': c_fit,
                'step_value': b_fit,
                'damage_zone_width': 2 * w_s_fit * 2.5631  # ~99% transition width
            }
        except Exception as e:
            if profile_index is not None:
                print(f"Erf fit failed for profile {profile_index}: {e}")
            else:
                print(f"Erf fit failed: {e}")
            return None

    elif mode == "backslip_C":
        x0_0 = fault_pos
        S0 = (np.nanmax(values) - np.nanmin(values))
        d1_0 = 1.0
        C0 = 0.0
        d2_0 = 1.0
        a0 = np.nanmean(values)
        b0 = 0.0
        try:
            popt, _ = curve_fit(
                backslip_C_func,
                distances, values,
                p0=[x0_0, S0, d1_0, C0, d2_0, a0, b0],
                bounds=(
                    [x0_0 - step_range, S0_range[0], d_range[0], C_range[0], max(d2_range[0], 1e-3), -np.inf, -np.inf],
                    [x0_0 + step_range, S0_range[1], d_range[1], C_range[1], d2_range[1], np.inf, np.inf]
                )
            )
            x0_fit, S_fit, d1_fit, C_fit, d2_fit, a_fit, b_fit = popt
            fit_params = [x0_fit, S_fit, d1_fit, C_fit, d2_fit, a_fit, b_fit]
            return {
                'fit_mode': 'backslip_C',
                'fit_params': fit_params,
                'step_position': x0_fit,
                'step_value': S_fit
            }
        except Exception as e:
            print(f"Backslip_C fit failed: {e}")
            return None

    else:
        raise ValueError("Unknown mode: choose 'step_linear', 'backslip', or 'backslip_C'")

def max_curvature_position(d):
    """
    Given locking depth d (km), return the position Δx* (km) where the curvature (second derivative) of y(x) = arctan((x-x0)/d) is maximized.

    Example:
    for d in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        print(f"d={d:.2f} km -> Δx*={max_curvature_position(d):.3f} km")
    """

    # Approximate formulas for small and large d
    if d < 0.3:
        return np.sqrt(d)
    if d > 5:
        return 0.57735 * d  # 1/sqrt(3) ≈ 0.57735

    # Equation: d^2(1+s)^2 = (3s+1)/(3s-1)
    def func(s):
        return d**2 * (1+s)**2 - (3*s+1)/(3*s-1)

    # Find root for s > 1/3
    s_root = brentq(func, 0.34, 1000)
    u_star = np.sqrt(s_root)
    return d

# --- Profile Extraction ---

def extract_profile(
    fault_line: LineString,
    analyzer,
    data: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    d: float,
    profile_length: float,
    profile_width: float,
    profile_spacing: float,
    interpolation_method: str,
    envelope_method: str,
    envelope_percentiles: Tuple[int, int],
    outlier_threshold: float,
    outlier_method: str,
    outlier_removal: bool,
    swath_method: str
) -> Optional[Any]:
    """
    Extract a single profile at distance d along the fault line.
    Returns profile_data or None.
    """
    center = fault_line.interpolate(d)
    total_length = fault_line.length
    if d + 0.01 < total_length:
        p1 = fault_line.interpolate(d)
        p2 = fault_line.interpolate(min(d + 0.01, total_length))
    else:
        p1 = fault_line.interpolate(max(d - 0.01, 0))
        p2 = fault_line.interpolate(d)
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    normal = np.array([-dy, dx])
    normal = normal / np.linalg.norm(normal)
    half_len = profile_length / 2
    start_xy = np.array([center.x, center.y]) - normal * half_len
    end_xy = np.array([center.x, center.y]) + normal * half_len
    start_lonlat = analyzer.xy2ll(*start_xy)
    end_lonlat = analyzer.xy2ll(*end_xy)
    ref_lonlat = analyzer.xy2ll(center.x, center.y)
    analyzer.define_profile(start_lonlat, end_lonlat,
                            spacing=profile_spacing,
                            width=profile_width,
                            coord_type='lonlat',
                            reference_point=ref_lonlat,
                            reference_coord_type='lonlat')
    analyzer.config.interpolation_method = interpolation_method
    analyzer.config.envelope_method = envelope_method
    analyzer.config.envelope_percentiles = envelope_percentiles
    analyzer.config.outlier_threshold = outlier_threshold
    analyzer.config.outlier_method = outlier_method
    analyzer.config.outlier_removal = outlier_removal
    analyzer.config.swath_method = swath_method
    profile_data = analyzer.extract_from_matrix(data, x_coords, y_coords, coord_type='lonlat')
    return profile_data, normal

def save_results(save_path: str, results: List[Dict[str, Any]]):
    """Save profile results to file."""
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_path}")

def load_fault_profile_results(path: str) -> List[Dict[str, Any]]:
    """Load saved fault profile results."""
    with open(path, "rb") as f:
        return pickle.load(f)

def plot_all_profiles(
    results: List[Dict[str, Any]],
    profiles_per_row: int = 6,
    profiles_per_fig: int = 36,
    save_prefix: str = 'profiles_batch',
    show: bool = False
):
    """
    Batch plot all profile fitting curves and raw scatter points, and save images.
    Each profile is a subplot, 6 columns per row, 36 profiles per figure.
    """
    n = len(results)
    for batch_idx, fig_start in enumerate(range(0, n, profiles_per_fig)):
        fig_end = min(fig_start + profiles_per_fig, n)
        fig_profiles = results[fig_start:fig_end]
        rows = (len(fig_profiles) + profiles_per_row - 1) // profiles_per_row
        fig, axes = plt.subplots(rows, profiles_per_row, figsize=(4*profiles_per_row, 3*rows), squeeze=False)
        for idx, r in enumerate(fig_profiles):
            row = idx // profiles_per_row
            col = idx % profiles_per_row
            ax = axes[row][col]
            # Raw scatter points
            distances = getattr(r.get('profile_data', r), 'swath_distances', r.get('profile_distances', []))
            values = getattr(r.get('profile_data', r), 'swath_values', r.get('profile_values', []))
            ax.scatter(distances, values, s=20, color='gray', label='Raw')
            # Fitted curve
            fit_distances = r.get('profile_distances', [])
            fit_values = r.get('profile_values', [])
            ax.plot(fit_distances, fit_values, color='r', label='Fit')
            left_point = r.get('left_point', (None, None))
            right_point = r.get('right_point', (None, None))
            ax.scatter(*left_point, color='b', label='Left Point')
            ax.scatter(*right_point, color='g', label='Right Point')
            ax.set_title(f'Profile {r.get("ith_profile", idx)}')
            ax.legend()
        # Remove unused subplots
        for idx in range(len(fig_profiles), rows * profiles_per_row):
            row = idx // profiles_per_row
            col = idx % profiles_per_row
            fig.delaxes(axes[row][col])
        plt.tight_layout()
        save_name = f"{save_prefix}_{batch_idx+1}.png"
        plt.savefig(save_name)
        if not show:
            plt.close(fig)
        else:
            plt.show()

if __name__ == "__main__":
    # Example usage
    from eqtools.csiExtend.statUtils.profile_analyzer import ProfileAnalyzer

    fault_points = [(120.0, 30.0), (121.0, 31.0), (122.0, 32.0)]
    lon = np.linspace(119, 123, 201)
    lat = np.linspace(29, 33, 201)
    LON, LAT = np.meshgrid(lon, lat)
    data = np.sin((LON-121.0)*np.pi) + np.cos((LAT-31.0)*np.pi) + np.random.normal(0, 0.1, LON.shape)
    analyzer = ProfileAnalyzer(lon0=121.0, lat0=31.0, utmzone=None)

    profile_cfg = ProfileConfig(profile_length=2.0, exclude_near_fault_range=(-0.2, 0.2))
    fit_cfg = FitConfig(fit_mode="backslip")
    batch_cfg = BatchConfig(spacing=1.0, max_workers=8)
    results = sample_fault_profiles(
        fault_points, analyzer, data, lon, lat,
        profile_cfg=profile_cfg, fit_cfg=fit_cfg, batch_cfg=batch_cfg
    )
    for r in results:
        print(f"Profile center: {r['center_lonlat']}, Step position: {r['step_position']:.2f} km, Step value: {r['step_value']:.2f}")

    # Plot offset (step value) along the fault trace
    if results:
        fault_line = LineString(fault_points)
        fault_distances = []
        for r in results:
            pt = r['center_lonlat']
            proj_pt = fault_line.project(LineString([pt]))
            fault_distances.append(proj_pt)
        offsets = [r['step_value'] for r in results]

        plt.figure(figsize=(8, 4))
        plt.plot(fault_distances, offsets, marker='o')
        plt.xlabel('Distance along fault trace (degrees or projected units)')
        plt.ylabel('Offset (step value)')
        plt.title('Extracted Offset (Step) Along Fault Trace')
        plt.grid(True)
        plt.tight_layout()
        plt.show()