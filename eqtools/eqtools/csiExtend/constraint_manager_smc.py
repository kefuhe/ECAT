"""
Bounds Manager Module

This module provides the BoundsManager class for managing parameter bounds
and constraints in Bayesian fault inversion processes.
"""

import numpy as np
import copy
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime

from .constraint_manager_base import ConstraintManagerBase


class ConstraintManagerSMC(ConstraintManagerBase):
    """
    Enhanced bounds and constraints management system for Bayesian/SMC inversions.
    
    Handles parameter bounds and linear constraints for Bayesian inversion:
    - Configuration loading and parsing
    - Bounds constraints (lb <= x <= ub) with per-fault customization
    - Inequality constraints (A_ineq * x <= b_ineq) for SMC_F_J mode
    - Equality constraints (A_eq * x = b_eq) for SMC_F_J mode
    - Rake angle constraints
    - Euler constraints
    """
    
    def __init__(self, inversion_instance, verbose: bool = True):
        """
        Initialize bounds manager.
        
        Parameters:
        -----------
        inversion_instance : object
            The inversion instance with config and multifaults
        verbose : bool
            Enable verbose output
        """
        self.inversion_instance = inversion_instance
        self.config = inversion_instance.config
        self.multifaults = inversion_instance.multifaults
        self.verbose = verbose
        
        # Basic parameters
        self.mcmc_samples = inversion_instance.mcmc_samples
        self.geometry_positions = inversion_instance.geometry_positions
        self.sigmas_position = inversion_instance.sigmas_position
        self.alpha_position = inversion_instance.alpha_position
        self.slip_positions = inversion_instance.slip_positions.copy()
        self.poly_positions = inversion_instance.poly_positions.copy()
        
        # Sampling modes
        self.slip_sampling_mode = self.config.slip_sampling_mode
        self.bayesian_sampling_mode = getattr(self.config, 'bayesian_sampling_mode', 'SMC_F_J')

        # Adjust positions for rake_fixed mode
        self._adjust_positions_for_rake_fixed()
        
        # Shared storage (constraints, cache, common bounds keys)
        self._init_shared_storage()
        
        # Extend bounds with SMC-specific keys
        self._bounds.update({
            'geometry': {},          # Per-fault geometry bounds
            'slip_magnitude': {},    # Per-fault slip magnitude bounds
            'rake_angle': {},        # Per-fault rake angle bounds
            'sigmas': None,          # Global sigmas bounds
            'alpha': None,           # Global alpha bounds
        })
        
        if self.verbose:
            print(f"[OK] BoundsManager initialized (mode: {self.bayesian_sampling_mode}/{self.slip_sampling_mode})")

    def _adjust_positions_for_rake_fixed(self):
        """Adjust slip and poly positions for rake_fixed mode."""
        if self.slip_sampling_mode == 'rake_fixed':
            total_half = 0
            for ifault in self.config.faults_list:
                lb_slip, ub_slip = self.slip_positions[ifault.name]
                lb_poly, ub_poly = self.poly_positions[ifault.name]
                # Use adapter for spatial element count; fallback to patch/volumes/1
                n_spatial = self._get_n_spatial_elements(ifault)
                self.slip_positions[ifault.name] = [lb_slip - total_half, ub_slip - total_half - n_spatial]
                total_half += n_spatial
                self.poly_positions[ifault.name] = [lb_poly - total_half, ub_poly - total_half]

    def _get_parallel_rank(self):
        return getattr(self.config, 'parallel_rank', None)

    def _get_source_type(self, fault_name):
        """Get source type string for a given source name, using adapter if available."""
        if hasattr(self.multifaults, 'adapters') and fault_name in self.multifaults.adapters:
            return self.multifaults.adapters[fault_name].source_type
        fault_obj = next((f for f in self.config.faults_list if f.name == fault_name), None)
        return getattr(fault_obj, 'type', 'Fault') if fault_obj else 'Fault'

    def _get_n_spatial_elements(self, fault_obj):
        """Get number of spatial elements using adapter or fallback attributes."""
        if hasattr(self.multifaults, 'adapters') and fault_obj.name in self.multifaults.adapters:
            return self.multifaults.adapters[fault_obj.name].get_n_spatial_elements()
        if hasattr(fault_obj, 'patch'):
            return len(fault_obj.patch)
        if hasattr(fault_obj, 'volumes'):
            return len(fault_obj.volumes)
        return 1  # Point source (Pressure)

    def _is_smc_fj_mode(self) -> bool:
        """Check if in SMC_F_J mode with ss_ds sampling (supports linear constraints)."""
        return (self.bayesian_sampling_mode == 'SMC_F_J' and 
                self.slip_sampling_mode == 'ss_ds')

    def _fault_exists(self, fault_name: str) -> bool:
        """Check if fault exists in config.faults_list."""
        return any(fault.name == fault_name for fault in self.config.faults_list)

    def _on_bounds_config_loaded(self):
        """Sync bounds config to inversion_instance for SMC compatibility."""
        self.inversion_instance.bounds_config = self._bounds_config

    # ==================== Configuration Loading ====================

    def _get_extra_config_summary_items(self) -> list:
        """Return SMC-specific config summary items (geometry, sigmas, alpha, etc.)."""
        items = []
        if not self._bounds_config:
            return items
        if 'geometry' in self._bounds_config:
            items.append(f"geometry bounds for {len(self._bounds_config['geometry'])} fault(s)")
        if 'slip_magnitude' in self._bounds_config:
            items.append(f"slip magnitude bounds for {len(self._bounds_config['slip_magnitude'])} fault(s)")
        if 'sigmas' in self._bounds_config:
            items.append("sigmas bounds")
        if 'alpha' in self._bounds_config:
            items.append("alpha bounds")
        return items

    # ==================== Global Bounds Management ====================
    
    def set_global_bounds(self, lb: float = None, ub: float = None, source: str = "manual"):
        """
        Set global bounds that apply to all parameters by default.
        
        Parameters:
        -----------
        lb : float, optional
            Global lower bound
        ub : float, optional
            Global upper bound
        source : str
            Source description
        """
        if lb is not None and ub is not None and lb > ub:
            raise ValueError("Global lower bound should be less than upper bound")
        
        # Initialize parameter arrays if needed
        self._initialize_bounds_arrays()
        
        if lb is not None:
            self._bounds['global']['lb'] = lb
            self._bounds['lb'][:] = lb  # Direct global update, no nan check
        if ub is not None:
            self._bounds['global']['ub'] = ub
            self._bounds['ub'][:] = ub  # Direct global update, no nan check
        
        if self.verbose:
            print(f"[GLB] Set global bounds: lb={lb}, ub={ub} (source: {source})")

    def _initialize_bounds_arrays(self):
        """Initialize bounds arrays based on mcmc_samples."""
        if self._bounds['lb'] is None:
            self._bounds['lb'] = np.ones(self.mcmc_samples) * np.nan
        if self._bounds['ub'] is None:
            self._bounds['ub'] = np.ones(self.mcmc_samples) * np.nan

    # ==================== Hyperparameter Bounds ====================
    
    def set_hyperparameter_bounds(self, geometry=None, sigmas=None, alpha=None, source: str = "manual"):
        """Set bounds for hyperparameters (geometry, sigmas, alpha)."""
        if geometry is not None:
            self.set_geometry_bounds(geometry, source)
        if sigmas is not None:
            self.set_sigmas_bounds(sigmas, source)
        if alpha is not None:
            self.set_alpha_bounds(alpha, source)

    def set_geometry_bounds(self, geometry_bounds, source: str = "manual"):
        """Set bounds for geometry parameters (per fault) with per-parameter support."""
        if geometry_bounds is None:
            return
        
        self._initialize_bounds_arrays()
        
        for fault_name, bounds in geometry_bounds.items():
            if not self._fault_exists(fault_name):
                if self.verbose:
                    print(f"[!]  Warning: Fault '{fault_name}' not found in faults_list, skipping geometry bounds")
                continue
                
            if (fault_name in self.config.faults and 
                self.config.faults[fault_name]['geometry']['update']):
                start, end = self.config.faults[fault_name]['geometry']['sample_positions']
                expected_params = end - start
                
                lb_vals, ub_vals = self._process_parameter_bounds(
                    bounds, expected_params, f"geometry for {fault_name}"
                )
                
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['lb'][start:end] = lb_vals
                    self._bounds['ub'][start:end] = ub_vals
                    self._bounds['geometry'][fault_name] = (lb_vals, ub_vals)
                    
                    if self.verbose:
                        print(f"[*]  Set geometry bounds for '{fault_name}': {len(lb_vals)} parameters (source: {source})")

    def set_sigmas_bounds(self, sigmas_bounds, source: str = "manual"):
        """Set bounds for sigma parameters with per-sigma support."""
        if not any(self.config.sigmas['update']) or sigmas_bounds is None:
            return
        
        self._initialize_bounds_arrays()
        start, end = self.sigmas_position
        expected_sigmas = end - start
        
        lb_vals, ub_vals = self._process_parameter_bounds(
            sigmas_bounds, expected_sigmas, "sigmas"
        )
        
        if lb_vals is not None and ub_vals is not None:
            self._bounds['lb'][start:end] = lb_vals
            self._bounds['ub'][start:end] = ub_vals
            self._bounds['sigmas'] = (lb_vals, ub_vals)
            
            if self.verbose:
                print(f"[STAT] Set sigmas bounds: {len(lb_vals)} sigmas (source: {source})")

    def set_alpha_bounds(self, alpha_bounds, source: str = "manual"):
        """Set bounds for alpha parameters with per-alpha support."""
        if not any(self.config.alpha['update']) or alpha_bounds is None:
            return
        
        self._initialize_bounds_arrays()
        start, end = self.alpha_position
        expected_alphas = end - start
        
        lb_vals, ub_vals = self._process_parameter_bounds(
            alpha_bounds, expected_alphas, "alpha"
        )
        
        if lb_vals is not None and ub_vals is not None:
            self._bounds['lb'][start:end] = lb_vals
            self._bounds['ub'][start:end] = ub_vals
            self._bounds['alpha'] = (lb_vals, ub_vals)
            
            if self.verbose:
                print(f"[*] Set alpha bounds: {len(lb_vals)} alphas (source: {source})")

    # ==================== Linear Parameter Bounds ====================
    
    def set_linear_parameter_bounds(self, slip_magnitude=None, rake_angle=None, 
                                   strikeslip=None, dipslip=None, poly=None, source: str = "manual"):
        """Set bounds for linear parameters (slip, poly)."""
        self.set_slip_bounds_for_all_faults(slip_magnitude, rake_angle, strikeslip, dipslip, source)
        self.set_poly_bounds_for_all_faults(poly, source)

    def set_slip_bounds_for_all_faults(self, slip_magnitude=None, rake_angle=None, 
                                      strikeslip=None, dipslip=None, source: str = "manual"):
        """Set slip bounds for all faults based on slip_sampling_mode.
        
        Only applies to Fault-type sources. Pressure/Sbarbot sources should
        use set_source_component_bounds() instead.
        """
        self._initialize_bounds_arrays()
        
        for fault_name in self.config.faultnames:
            if not self._fault_exists(fault_name):
                if self.verbose:
                    print(f"[!]  Warning: Fault '{fault_name}' not found in faults_list, skipping slip bounds")
                continue
            
            # Skip non-Fault sources — they don't have strikeslip/dipslip/rake semantics
            if self._get_source_type(fault_name) != 'Fault':
                continue
            
            self._set_slip_bounds_for_fault(fault_name, slip_magnitude, rake_angle, strikeslip, dipslip, source)

    def _set_slip_bounds_for_fault(self, fault_name, slip_magnitude=None, rake_angle=None, 
                                  strikeslip=None, dipslip=None, source: str = "manual"):
        """Set slip bounds for a specific fault based on sampling mode."""
        if self.slip_sampling_mode == "rake_fixed":
            if slip_magnitude and slip_magnitude.get(fault_name):
                self.set_slip_bounds_based_on_mode(fault_name, slip_magnitude=slip_magnitude[fault_name], source=source)
                
        elif self.slip_sampling_mode == "ss_ds":
            ss = strikeslip.get(fault_name) if strikeslip else None
            ds = dipslip.get(fault_name) if dipslip else None
            if ss is not None or ds is not None:
                self.set_slip_bounds_based_on_mode(fault_name, strikeslip=ss, dipslip=ds, source=source)
                
        elif self.slip_sampling_mode == "magnitude_rake":
            mag = slip_magnitude.get(fault_name) if slip_magnitude else None
            rake = rake_angle.get(fault_name) if rake_angle else None
            if mag is not None or rake is not None:
                self.set_slip_bounds_based_on_mode(fault_name, slip_magnitude=mag, rake_angle=rake, source=source)

    def set_slip_bounds_based_on_mode(self, fault_name, slip_magnitude=None, rake_angle=None, 
                                     strikeslip=None, dipslip=None, source: str = "manual"):
        """Set slip parameter bounds based on the slip_sampling_mode with per-patch support."""
        slip_start, slip_end = self.slip_positions[fault_name]
        slip_half = (slip_end + slip_start) // 2
        n_patches = slip_end - slip_start
        
        # Get fault object for patch count validation
        fault_obj = next((f for f in self.config.faults_list if f.name == fault_name), None)
        if fault_obj is None:
            if self.verbose:
                print(f"[!]  Warning: Fault '{fault_name}' not found in faults_list")
            return
        
        # Use adapter if available for spatial element count, fallback to len(patch)
        if hasattr(self.config, 'multifaults') and hasattr(self.config.multifaults, 'adapters') \
                and fault_name in self.config.multifaults.adapters:
            expected_patches = self.config.multifaults.adapters[fault_name].get_n_spatial_elements()
        elif hasattr(fault_obj, 'patch'):
            expected_patches = len(fault_obj.patch)
        elif hasattr(fault_obj, 'volumes'):
            expected_patches = len(fault_obj.volumes)
        else:
            expected_patches = n_patches

        if self.slip_sampling_mode == 'rake_fixed':
            # Only magnitude bounds (n_patches values)
            if slip_magnitude is not None:
                lb_vals, ub_vals = self._process_parameter_bounds(
                    slip_magnitude, expected_patches, f"slip_magnitude for {fault_name}"
                )
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['slip_magnitude'][fault_name] = (lb_vals, ub_vals)
                    self._bounds['lb'][slip_start:slip_end] = lb_vals
                    self._bounds['ub'][slip_start:slip_end] = ub_vals
                    
                    if self.verbose:
                        print(f"[*] Set slip magnitude bounds for '{fault_name}': {len(lb_vals)} patches (source: {source})")
                
        elif self.slip_sampling_mode == 'magnitude_rake':
            # Magnitude and rake bounds (each n_patches values)
            if slip_magnitude is not None:
                lb_vals, ub_vals = self._process_parameter_bounds(
                    slip_magnitude, expected_patches, f"slip_magnitude for {fault_name}"
                )
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['slip_magnitude'][fault_name] = (lb_vals, ub_vals)
                    self._bounds['lb'][slip_start:slip_half] = lb_vals
                    self._bounds['ub'][slip_start:slip_half] = ub_vals
                    
                    if self.verbose:
                        print(f"[*] Set slip magnitude bounds for '{fault_name}': {len(lb_vals)} patches (source: {source})")
                
            if rake_angle is not None:
                lb_vals, ub_vals = self._process_parameter_bounds(
                    rake_angle, expected_patches, f"rake_angle for {fault_name}"
                )
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['rake_angle'][fault_name] = (lb_vals, ub_vals)
                    self._bounds['lb'][slip_half:slip_end] = lb_vals
                    self._bounds['ub'][slip_half:slip_end] = ub_vals
                    
                    if self.verbose:
                        print(f"[SYNC] Set rake angle bounds for '{fault_name}': {len(lb_vals)} patches (source: {source})")
                
        elif self.slip_sampling_mode == 'ss_ds':
            # Strike-slip and dip-slip bounds (each n_patches values)
            if strikeslip is not None:
                lb_vals, ub_vals = self._process_parameter_bounds(
                    strikeslip, expected_patches, f"strikeslip for {fault_name}"
                )
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['strikeslip'][fault_name] = (lb_vals, ub_vals)
                    self._bounds['lb'][slip_start:slip_half] = lb_vals
                    self._bounds['ub'][slip_start:slip_half] = ub_vals
                    
                    if self.verbose:
                        print(f"[<>]  Set strike-slip bounds for '{fault_name}': {len(lb_vals)} patches (source: {source})")
                
            if dipslip is not None:
                lb_vals, ub_vals = self._process_parameter_bounds(
                    dipslip, expected_patches, f"dipslip for {fault_name}"
                )
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['dipslip'][fault_name] = (lb_vals, ub_vals)
                    self._bounds['lb'][slip_half:slip_end] = lb_vals
                    self._bounds['ub'][slip_half:slip_end] = ub_vals
                    
                    if self.verbose:
                        print(f"[UD]  Set dip-slip bounds for '{fault_name}': {len(lb_vals)} patches (source: {source})")


    def set_poly_bounds_for_all_faults(self, poly_bounds, source: str = "manual"):
        """Set polynomial bounds for all faults."""
        if poly_bounds is None:
            return
        
        self._initialize_bounds_arrays()
        
        for fault_name, bounds in poly_bounds.items():
            if not self._fault_exists(fault_name):
                if self.verbose:
                    print(f"[!]  Warning: Fault '{fault_name}' not found in faults_list, skipping poly bounds")
                continue
            
            self.set_poly_bounds(fault_name, bounds, source)

    def set_poly_bounds(self, fault_name: str, poly_bounds, source: str = "manual"):
        """Set bounds for polynomial parameters of a specific fault with per-coefficient support."""
        start, end = self.poly_positions[fault_name]
        expected_coeffs = end - start
        
        lb_vals, ub_vals = self._process_parameter_bounds(
            poly_bounds, expected_coeffs, f"poly for {fault_name}"
        )
        
        if lb_vals is not None and ub_vals is not None:
            self._bounds['lb'][start:end] = lb_vals
            self._bounds['ub'][start:end] = ub_vals
            self._bounds['poly'][fault_name] = (lb_vals, ub_vals)
            
            if self.verbose:
                print(f"[GEO] Set poly bounds for '{fault_name}': {len(lb_vals)} coefficients (source: {source})")

    def set_source_component_bounds(self, source_name: str, comp_bounds: Dict[str, Any],
                                     source: str = "manual"):
        """
        Set per-component bounds for any source type using source adapters.
        
        Parameters
        ----------
        source_name : str
            Name of the source (fault/pressure/sbarbot).
        comp_bounds : dict
            {component_name: bounds_input} — component names from adapter.get_param_names().
            Each bounds_input is processed by _process_parameter_bounds (supports
            [lb, ub] uniform or per-element formats).
        source : str
            Source description for audit trail.
        """
        if not self._fault_exists(source_name):
            if self.verbose:
                print(f"[!]  Warning: Source '{source_name}' not found in faults_list, skipping source_bounds")
            return
        
        if not hasattr(self.multifaults, 'adapters') or source_name not in self.multifaults.adapters:
            if self.verbose:
                print(f"[!]  Warning: No adapter found for source '{source_name}', skipping source_bounds")
            return
        
        self._initialize_bounds_arrays()
        
        adapter = self.multifaults.adapters[source_name]
        params_per_comp = adapter.get_n_params_per_component()
        slip_st, _ = self.slip_positions[source_name]
        
        offset = slip_st
        for comp_name in adapter.get_param_names():
            n = params_per_comp[comp_name]
            if comp_name in comp_bounds:
                lb_vals, ub_vals = self._process_parameter_bounds(
                    comp_bounds[comp_name], n, f"{comp_name} for {source_name}"
                )
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['lb'][offset:offset + n] = lb_vals
                    self._bounds['ub'][offset:offset + n] = ub_vals
            offset += n
        
        if self.verbose:
            print(f"[SRC] Set component bounds for '{source_name}': {comp_bounds} (source: {source})")

    def _process_parameter_bounds(self, bounds_input, expected_length: int, param_name: str):
        """
        Process parameter bounds input to support both uniform and per-element bounds.
        
        Parameters:
        -----------
        bounds_input : various types
            Can be:
            - [lb, ub]: uniform bounds for all elements
            - [[lb1, lb2, ...], [ub1, ub2, ...]]: per-element bounds  
            - {'lb': [lb1, lb2, ...], 'ub': [ub1, ub2, ...]}: per-element bounds dict
            - {'lb': lb_val, 'ub': ub_val}: uniform bounds dict
        expected_length : int
            Expected number of elements
        param_name : str
            Parameter name for error messages
            
        Returns:
        --------
        tuple: (lb_array, ub_array) or (None, None) if invalid
        """
        if bounds_input is None:
            return None, None
        
        try:
            # Case 1: Dictionary format
            if isinstance(bounds_input, dict):
                if 'lb' in bounds_input and 'ub' in bounds_input:
                    lb_input = bounds_input['lb']
                    ub_input = bounds_input['ub']
                    
                    # Convert to arrays
                    lb_array = self._convert_to_array(lb_input, expected_length, f"{param_name} lower bounds")
                    ub_array = self._convert_to_array(ub_input, expected_length, f"{param_name} upper bounds")
                    
                    if lb_array is not None and ub_array is not None:
                        # Validate bounds consistency
                        if np.any(lb_array > ub_array):
                            raise ValueError(f"Lower bounds > upper bounds for {param_name}")
                        return lb_array, ub_array
                else:
                    raise ValueError(f"Dictionary bounds for {param_name} must contain 'lb' and 'ub' keys")
            
            # Case 2: List/array format
            elif isinstance(bounds_input, (list, tuple, np.ndarray)):
                bounds_array = np.asarray(bounds_input)
                
                # Case 2a: Simple [lb, ub] format (uniform bounds)
                if bounds_array.ndim == 1 and len(bounds_array) == 2:
                    lb_val, ub_val = bounds_array
                    if lb_val > ub_val:
                        raise ValueError(f"Lower bound > upper bound for {param_name}: {lb_val} > {ub_val}")
                    
                    lb_array = np.full(expected_length, lb_val)
                    ub_array = np.full(expected_length, ub_val)
                    return lb_array, ub_array
                
                # Case 2b: [[lb1, lb2, ...], [ub1, ub2, ...]] format (per-element bounds)
                elif bounds_array.ndim == 2 and bounds_array.shape[0] == 2:
                    lb_input, ub_input = bounds_array[0], bounds_array[1]
                    
                    lb_array = self._convert_to_array(lb_input, expected_length, f"{param_name} lower bounds")
                    ub_array = self._convert_to_array(ub_input, expected_length, f"{param_name} upper bounds")
                    
                    if lb_array is not None and ub_array is not None:
                        if np.any(lb_array > ub_array):
                            raise ValueError(f"Lower bounds > upper bounds for {param_name}")
                        return lb_array, ub_array
                
                # Case 2c: Single array interpreted as uniform bounds
                elif bounds_array.ndim == 1 and len(bounds_array) == expected_length:
                    # This could be ambiguous, but we'll treat it as lower bounds only
                    if self.verbose:
                        print(f"[!]  Warning: Single array for {param_name} interpreted as lower bounds only")
                    return bounds_array, np.full(expected_length, np.inf)
                
                else:
                    raise ValueError(f"Invalid bounds array shape for {param_name}: {bounds_array.shape}")
            
            else:
                raise ValueError(f"Unsupported bounds format for {param_name}: {type(bounds_input)}")
        
        except Exception as e:
            if self.verbose:
                print(f"[X] Error processing bounds for {param_name}: {e}")
            return None, None

    def _convert_to_array(self, input_val, expected_length: int, param_desc: str):
        """
        Convert input to numpy array with proper length validation.
        
        Parameters:
        -----------
        input_val : scalar, list, or array
            Input value to convert
        expected_length : int
            Expected array length
        param_desc : str
            Parameter description for error messages
            
        Returns:
        --------
        np.ndarray or None: Converted array or None if invalid
        """
        try:
            # Scalar case - broadcast to expected length
            if np.isscalar(input_val):
                return np.full(expected_length, float(input_val))
            
            # Array case - validate length
            array_val = np.asarray(input_val, dtype=float)
            
            if array_val.ndim != 1:
                raise ValueError(f"Expected 1D array for {param_desc}")
            
            if len(array_val) == 1:
                # Single element - broadcast
                return np.full(expected_length, array_val[0])
            elif len(array_val) == expected_length:
                # Correct length
                return array_val
            else:
                raise ValueError(f"Length mismatch for {param_desc}: expected {expected_length}, got {len(array_val)}")
        
        except Exception as e:
            if self.verbose:
                print(f"[X] Error converting {param_desc}: {e}")
            return None

    # ==================== Apply Bounds from Config ====================
    
    def apply_bounds_from_config(self):
        """Apply all bounds from loaded configuration."""
        if not self._bounds_config:
            if self.verbose:
                print("[!]  No bounds config loaded")
            return
        
        # Apply global bounds
        lb = self._bounds_config.get('lb', None)
        ub = self._bounds_config.get('ub', None)
        if lb is not None or ub is not None:
            self.set_global_bounds(lb, ub, source="config_file")
        
        # Apply hyperparameter bounds
        self.set_hyperparameter_bounds(
            geometry=self._bounds_config.get('geometry', None),
            sigmas=self._bounds_config.get('sigmas', None),
            alpha=self._bounds_config.get('alpha', None),
            source="config_file"
        )
        
        # Apply linear parameter bounds (Fault-only: strikeslip/dipslip/magnitude/rake)
        self.set_linear_parameter_bounds(
            slip_magnitude=self._bounds_config.get('slip_magnitude', None),
            rake_angle=self._bounds_config.get('rake_angle', None),
            strikeslip=self._bounds_config.get('strikeslip', None),
            dipslip=self._bounds_config.get('dipslip', None),
            poly=self._bounds_config.get('poly', None),
            source="config_file"
        )
        
        # Apply generic source component bounds (works for Fault, Pressure, Sbarbot)
        source_bounds_config = self._bounds_config.get('source_bounds', {})
        for source_name, comp_bounds in source_bounds_config.items():
            if self._fault_exists(source_name):
                self.set_source_component_bounds(source_name, comp_bounds, source="config_file")
        
        self._bounds['source'] = "config_file"
        self._bounds['applied_time'] = datetime.now()

    # ==================== Bounds Retrieval Methods ====================
    
    def get_bounds_for_fullsmc(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds for FULLSMC mode - all parameters with lb/ub constraints only."""
        self._initialize_bounds_arrays()
        
        lb_full = self._bounds['lb'].copy()
        ub_full = self._bounds['ub'].copy()
        
        # Handle NaN values with reasonable defaults
        lb_full[np.isnan(lb_full)] = -10.0
        ub_full[np.isnan(ub_full)] = 10.0
        
        return lb_full, ub_full

    def get_bounds_for_hyperparameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds for hyperparameters in SMC_F_J mode."""
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        
        self._initialize_bounds_arrays()
        lb_hyper = self._bounds['lb'][:linear_sample_start].copy()
        ub_hyper = self._bounds['ub'][:linear_sample_start].copy()
        
        # Handle NaN values
        lb_hyper[np.isnan(lb_hyper)] = -10.0
        ub_hyper[np.isnan(ub_hyper)] = 10.0
        
        return lb_hyper, ub_hyper

    def get_bounds_for_linear_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds for linear parameters in SMC_F_J mode."""
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        
        self._initialize_bounds_arrays()
        lb_linear = self._bounds['lb'][linear_sample_start:].copy()
        ub_linear = self._bounds['ub'][linear_sample_start:].copy()
        
        # Handle NaN values
        lb_linear[np.isnan(lb_linear)] = -10.0
        ub_linear[np.isnan(ub_linear)] = 10.0
        
        return lb_linear, ub_linear

    # ==================== SMC_F_J Linear Constraints Generation ====================
    
    def generate_rake_angle_constraints(self, rake_angle=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate linear inequality constraints for rake angle bounds.
        Only applies to SMC_F_J mode with ss_ds sampling.
        """
        if not self._is_smc_fj_mode():
            return np.zeros((0, 0)), np.zeros(0)
        
        # Get rake angle constraints
        if rake_angle is None:
            if (not hasattr(self, '_bounds_config') or self._bounds_config is None or 
                'rake_angle' not in self._bounds_config):
                return np.zeros((0, 0)), np.zeros(0)
            rake_angle = self._bounds_config['rake_angle']
        else:
            if not hasattr(self, '_bounds_config') or self._bounds_config is None:
                self._bounds_config = {}
            self._bounds_config['rake_angle'].update(rake_angle)
        
        return self._generate_rake_inequality_constraints(rake_angle)

    def generate_fixed_rake_constraints(self, fixed_rake) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate linear equality constraints for fixed rake angles.
        Only applies to SMC_F_J mode with ss_ds sampling.
        """
        if not self._is_smc_fj_mode():
            return np.zeros((0, 0)), np.zeros(0)
        
        return self._generate_rake_equality_constraints(fixed_rake)

    def _generate_rake_inequality_constraints(self, rake_angle):
        """Generate rake angle inequality constraints A*x <= b.
        
        Only Fault-type sources are included; non-Fault sources in
        rake_angle dict are silently skipped.
        """
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        nlinear = self.mcmc_samples - linear_sample_start
        
        # Filter faults with rake constraints that exist in faults_list AND are Fault type
        constrained_faults = [fault for fault in self.config.faults_list 
                             if fault.name in rake_angle
                             and self._get_source_type(fault.name) == 'Fault']
        
        if not constrained_faults:
            return np.zeros((0, nlinear)), np.zeros(0)
        
        npatch = sum(self._get_n_spatial_elements(fault) for fault in constrained_faults)
        A = np.zeros((2 * npatch, nlinear))
        b = np.zeros(2 * npatch)

        patch_count = 0
        for fault in constrained_faults:
            start, end = self.slip_positions[fault.name]
            start -= linear_sample_start
            end -= linear_sample_start
            half = (start + end) // 2
            
            rake_start, rake_end = rake_angle[fault.name]
            inpatch = self._get_n_spatial_elements(fault)
            
            for i in range(inpatch):
                # Lower bound: ss*sin(rake_start) - ds*cos(rake_start) <= 0
                A[patch_count + i, start + i] = np.sin(np.deg2rad(rake_start))
                A[patch_count + i, half + i] = -np.cos(np.deg2rad(rake_start))
                
                # Upper bound: -ss*sin(rake_end) + ds*cos(rake_end) <= 0
                A[patch_count + inpatch + i, start + i] = -np.sin(np.deg2rad(rake_end))
                A[patch_count + inpatch + i, half + i] = np.cos(np.deg2rad(rake_end))
                
            patch_count += inpatch
        
        return A, b

    def _generate_rake_equality_constraints(self, fixed_rake):
        """Generate fixed rake equality constraints A*x = b.
        
        Only Fault-type sources are included; non-Fault sources are skipped.
        """
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        nlinear = self.mcmc_samples - linear_sample_start
        
        # Filter faults with fixed rake constraints that exist AND are Fault type
        constrained_faults = [fault for fault in self.config.faults_list 
                             if fault.name in fixed_rake
                             and self._get_source_type(fault.name) == 'Fault']
        
        if not constrained_faults:
            return np.zeros((0, nlinear)), np.zeros(0)
        
        npatch = sum(self._get_n_spatial_elements(fault) for fault in constrained_faults)
        A_eq = np.zeros((npatch, nlinear))
        b_eq = np.zeros(npatch)

        patch_count = 0
        for fault in constrained_faults:
            start, end = self.slip_positions[fault.name]
            start -= linear_sample_start
            end -= linear_sample_start
            half = (start + end) // 2
            
            rake = fixed_rake[fault.name]
            inpatch = self._get_n_spatial_elements(fault)
            
            for i in range(inpatch):
                # Fixed rake: ss*sin(rake) - ds*cos(rake) = 0
                A_eq[patch_count + i, start + i] = np.sin(np.deg2rad(rake))
                A_eq[patch_count + i, half + i] = -np.cos(np.deg2rad(rake))
                
            patch_count += inpatch
        
        return A_eq, b_eq

    # ==================== Constraint Management API ====================
    
    def add_inequality_constraint(self, A: np.ndarray, b: np.ndarray, name: str, 
                                source: str = "manual", overwrite: bool = False):
        """Add inequality constraint A @ x <= b (SMC_F_J mode only).

        Adds a mode guard and validates that A columns match the linear
        parameter count before delegating to the base implementation.
        """
        if not self._is_smc_fj_mode():
            if self._should_warn():
                warnings.warn("Linear constraints only applicable in SMC_F_J mode with ss_ds sampling")
            return
        # SMC-specific: validate column count matches linear parameter space
        A_arr = np.asarray(A)
        expected_cols = self.mcmc_samples - self.inversion_instance.linear_sample_start_position
        if A_arr.ndim == 2 and A_arr.shape[1] != expected_cols:
            raise ValueError(
                f"Constraint A columns ({A_arr.shape[1]}) != "
                f"linear parameters ({expected_cols})")
        super().add_inequality_constraint(A, b, name, source, overwrite)

    def add_equality_constraint(self, A: np.ndarray, b: np.ndarray, name: str,
                              source: str = "manual", overwrite: bool = False):
        """Add equality constraint A @ x = b (SMC_F_J mode only).

        Adds a mode guard and validates that A columns match the linear
        parameter count before delegating to the base implementation.
        """
        if not self._is_smc_fj_mode():
            if self._should_warn():
                warnings.warn("Linear constraints only applicable in SMC_F_J mode with ss_ds sampling")
            return
        # SMC-specific: validate column count matches linear parameter space
        A_arr = np.asarray(A)
        expected_cols = self.mcmc_samples - self.inversion_instance.linear_sample_start_position
        if A_arr.ndim == 2 and A_arr.shape[1] != expected_cols:
            raise ValueError(
                f"Constraint A columns ({A_arr.shape[1]}) != "
                f"linear parameters ({expected_cols})")
        super().add_equality_constraint(A, b, name, source, overwrite)

    def add_rake_angle_constraints(self, rake_angle=None):
        """Add rake angle inequality constraints for SMC_F_J mode."""
        if not self._is_smc_fj_mode():
            return
        
        A, b = self.generate_rake_angle_constraints(rake_angle)
        if A.size > 0:
            source = 'bounds_config' if rake_angle is None else 'manual'
            self.add_inequality_constraint(A, b, name='rake_constraints', source=source, overwrite=True)

    def add_fixed_rake_constraints(self, fixed_rake):
        """Add fixed rake equality constraints for SMC_F_J mode."""
        if not self._is_smc_fj_mode():
            return
        
        A_eq, b_eq = self.generate_fixed_rake_constraints(fixed_rake)
        if A_eq.size > 0:
            self.add_equality_constraint(A_eq, b_eq, name='fixed_rake_constraints', source='manual', overwrite=True)

    def add_euler_constraints(self):
        """Add Euler constraints if enabled in config (SMC_F_J mode only, Fault-only)."""
        if not self._is_smc_fj_mode():
            return
            
        try:
            if (not hasattr(self.config, 'euler_constraints') or 
                not self.config.euler_constraints.get('enabled', False)):
                return
            
            from .euler_inequality_constraints import generate_euler_inequality_constraints
            
            euler_config = copy.deepcopy(self.config.euler_constraints)
            
            # Filter out non-Fault sources from euler config (Euler is Fault-specific)
            if 'faults' in euler_config:
                non_fault_names = [fn for fn in euler_config['faults']
                                   if self._get_source_type(fn) != 'Fault']
                for fn in non_fault_names:
                    if self.verbose:
                        print(f"[!]  Warning: Euler constraint skipping non-Fault source '{fn}'")
                    del euler_config['faults'][fn]
            
            all_datasets = self.config.geodata['data']
            
            A_ineq, b_ineq = generate_euler_inequality_constraints(
                self.inversion_instance, euler_config, all_datasets
            )
            
            if A_ineq is not None and A_ineq.size > 0:
                self.add_inequality_constraint(
                    A_ineq, b_ineq, name='euler_constraints', 
                    source='config.euler_constraints', overwrite=True
                )
                
        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply Euler constraints: {e}")
            raise

    def remove_constraint(self, name: str, constraint_type: Optional[str] = None):
        """Remove constraint by name (SMC_F_J mode only)."""
        if not self._is_smc_fj_mode():
            if self._should_warn():
                warnings.warn("Linear constraints only applicable in SMC_F_J mode")
            return
        super().remove_constraint(name, constraint_type)

    def get_combined_inequality_constraints(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get combined inequality constraints as (A, b) for SMC_F_J mode."""
        if not self._is_smc_fj_mode():
            return None, None
        return super().get_combined_inequality_constraints()

    def get_combined_equality_constraints(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get combined equality constraints as (A, b) for SMC_F_J mode."""
        if not self._is_smc_fj_mode():
            return None, None
        return super().get_combined_equality_constraints()

    # ==================== Complete Constraint Application ====================
    
    def apply_source_constraints_from_config(self):
        """Apply source-specific inequality/equality constraints from ``source_constraints`` config.

        The ``source_constraints`` section in the bounds config YAML maps each source
        name to a list of constraint definitions.  Each definition contains:
        - ``name``: constraint identifier
        - ``type``: ``'inequality'`` or ``'equality'``
        - ``rule``: a recognised pattern (e.g. ``'pressure >= 0'``)

        Only effective in SMC_F_J mode (linear constraints).
        """
        if not self._is_smc_fj_mode():
            return

        if not self._bounds_config or 'source_constraints' not in self._bounds_config:
            return

        source_constraints_cfg = self._bounds_config['source_constraints']
        if not source_constraints_cfg:
            return

        if not hasattr(self.multifaults, 'adapters'):
            if self.verbose:
                print("[!]  Warning: multifaults has no adapters, skipping source_constraints")
            return

        linear_start = self.inversion_instance.linear_sample_start_position
        n_linear = self.mcmc_samples - linear_start

        for source_name, src_cfg in source_constraints_cfg.items():
            if not self._fault_exists(source_name):
                if self.verbose:
                    print(f"[!]  Warning: Source '{source_name}' not found, skipping source_constraints")
                continue
            if source_name not in self.multifaults.adapters:
                if self.verbose:
                    print(f"[!]  Warning: No adapter for '{source_name}', skipping source_constraints")
                continue

            adapter = self.multifaults.adapters[source_name]
            # param_start relative to linear parameter block
            param_start = self.slip_positions[source_name][0] - linear_start

            # Normalise list-of-dicts → dict-of-dicts keyed by constraint name
            constraints_dict = self._normalise_constraint_list(src_cfg)

            # Inequality constraints
            for cname, A, b in adapter.generate_source_inequality_constraints(
                    constraints_dict, param_start, n_linear):
                full_name = f"src_{source_name}_{cname}"
                self.add_inequality_constraint(A, b, name=full_name,
                                               source=f"source_constraints/{source_name}",
                                               overwrite=True)

            # Equality constraints
            for cname, A, b in adapter.generate_source_equality_constraints(
                    constraints_dict, param_start, n_linear):
                full_name = f"src_{source_name}_{cname}"
                self.add_equality_constraint(A, b, name=full_name,
                                             source=f"source_constraints/{source_name}",
                                             overwrite=True)

    def apply_all_constraints_from_config(self, bounds_config_file: str = None, 
                                        rake_limits: Dict = None, 
                                        encoding: str = 'utf-8'):
        """
        Apply all constraints based on configuration settings.
        
        Parameters:
        -----------
        bounds_config_file : str, optional
            Path to bounds configuration file
        rake_limits : dict, optional
            Additional rake angle limits
        encoding : str
            File encoding
        """
        if self.verbose:
            print("\n[RUN] Applying all constraints from configuration...")
        
        # Load bounds config if provided
        if bounds_config_file is not None:
            self.load_bounds_config(bounds_config_file, encoding)
        
        # Apply bounds from config
        if (self._bounds_config is not None and 
            hasattr(self.config, 'use_bounds_constraints') and 
            self.config.use_bounds_constraints):
            self.apply_bounds_from_config()
        
        # Apply rake angle constraints
        if (hasattr(self.config, 'use_rake_angle_constraints') and 
            self.config.use_rake_angle_constraints):
            self.apply_rake_constraints(rake_limits)
        
        # Apply Euler constraints
        if (hasattr(self.config, 'euler_constraints') and 
            self.config.euler_constraints.get('enabled', False)):
            self.add_euler_constraints()
        
        # Apply source-specific constraints from source_constraints config
        if self._bounds_config and 'source_constraints' in self._bounds_config:
            self.apply_source_constraints_from_config()
        
        if self.verbose:
            print("[OK] All constraints applied successfully")
            self.print_summary()

    def apply_rake_constraints(self, additional_rake_limits: Dict = None):
        """Apply rake angle constraints from config and additional limits."""
        try:
            # Collect rake limits from multiple sources
            final_rake_limits = {}
            
            # From bounds config
            if self._bounds_config and 'rake_angle' in self._bounds_config:
                final_rake_limits.update(self._bounds_config['rake_angle'])
            
            # From additional parameters
            if additional_rake_limits:
                final_rake_limits.update(additional_rake_limits)
            
            if not final_rake_limits:
                if self.verbose:
                    print("[i]  No rake angle limits specified")
                return
            
            # Apply rake angle constraints
            source = 'bounds_config'
            if additional_rake_limits:
                source += ' + additional_limits'
            self.add_rake_angle_constraints(final_rake_limits)

        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply rake constraints: {e}")
            raise

    # ==================== Summary and Diagnostics ====================
    
    def print_summary(self):
        """Print comprehensive summary of bounds and constraints with detailed parameter info."""
        print("\n" + "="*70)
        print("[*] BOUNDS MANAGER SUMMARY")
        print("="*70)
        
        print(f"[OK] Configuration:")
        print(f"   Slip sampling mode: {self.slip_sampling_mode}")
        print(f"   Bayesian sampling mode: {self.bayesian_sampling_mode}")
        print(f"   Linear constraints supported: {'[OK] Yes' if self._is_smc_fj_mode() else '[X] No'}")
        
        # Configuration file info
        if self._bounds['config_file']:
            print(f"   Config file: {self._bounds['config_file']}")
        
        # Bounds summary with detailed parameter breakdown
        print(f"\n[STAT] Bounds Status:")
        if self._bounds['lb'] is not None or self._bounds['ub'] is not None:
            n_params = self.mcmc_samples
            n_lb = np.sum(~np.isnan(self._bounds['lb'])) if self._bounds['lb'] is not None else 0
            n_ub = np.sum(~np.isnan(self._bounds['ub'])) if self._bounds['ub'] is not None else 0
            n_both = 0
            if self._bounds['lb'] is not None and self._bounds['ub'] is not None:
                n_both = np.sum((~np.isnan(self._bounds['lb'])) & (~np.isnan(self._bounds['ub'])))
            
            print(f"   Total parameters: {n_params}")
            print(f"   Lower bounded: {n_lb}")
            print(f"   Upper bounded: {n_ub}")
            print(f"   Fully bounded: {n_both}")
            
            # Global bounds
            global_lb = self._bounds['global']['lb']
            global_ub = self._bounds['global']['ub']
            if global_lb is not None or global_ub is not None:
                print(f"   Global defaults: lb={global_lb}, ub={global_ub}")
            
            # Parameter-specific bounds with detailed breakdown
            if hasattr(self.inversion_instance, 'linear_sample_start_position'):
                linear_start = self.inversion_instance.linear_sample_start_position
                print(f"   Hyperparameters: [0:{linear_start}] (geometry, sigma, alpha)")
                print(f"   Linear parameters: [{linear_start}:{self.mcmc_samples}] (slip, poly)")
            
            # Per-parameter bounds summary with element count
            bound_types = ['geometry', 'strikeslip', 'dipslip', 'slip_magnitude', 'rake_angle', 'poly']
            for bound_type in bound_types:
                bounds_dict = self._bounds[bound_type]
                if bounds_dict:
                    print(f"\n   {bound_type.capitalize()} bounds: {len(bounds_dict)} fault(s)")
                    for fault, bounds in bounds_dict.items():
                        if isinstance(bounds[0], np.ndarray):
                            # Per-element bounds
                            lb_array, ub_array = bounds
                            if len(lb_array) > 0 and len(ub_array) > 0:  # Check for non-empty arrays
                                print(f"     - {fault}: {len(lb_array)} elements")
                                print(f"       lb: [{lb_array.min():.3f} ... {lb_array.max():.3f}]")
                                print(f"       ub: [{ub_array.min():.3f} ... {ub_array.max():.3f}]")
                            else:
                                print(f"     - {fault}: {len(lb_array)} elements")

                        else:
                            # Uniform bounds (legacy)
                            print(f"     - {fault}: {bounds} (uniform)")
            
            # Global parameter bounds with element details
            if self._bounds['sigmas']:
                sigmas_bounds = self._bounds['sigmas']
                if isinstance(sigmas_bounds[0], np.ndarray):
                    lb_array, ub_array = sigmas_bounds
                    print(f"\n   Sigmas bounds: {len(lb_array)} elements")
                    print(f"     lb: [{lb_array.min():.3f} ... {lb_array.max():.3f}]")
                    print(f"     ub: [{ub_array.min():.3f} ... {ub_array.max():.3f}]")
                else:
                    print(f"\n   Sigmas bounds: {sigmas_bounds} (uniform)")
            
            if self._bounds['alpha']:
                alpha_bounds = self._bounds['alpha']
                if isinstance(alpha_bounds[0], np.ndarray):
                    lb_array, ub_array = alpha_bounds
                    print(f"\n   Alpha bounds: {len(lb_array)} elements")
                    print(f"     lb: [{lb_array.min():.3f} ... {lb_array.max():.3f}]")
                    print(f"     ub: [{ub_array.min():.3f} ... {ub_array.max():.3f}]")
                else:
                    print(f"\n   Alpha bounds: {alpha_bounds} (uniform)")
                    
            print(f"\n   Source: {self._bounds['source']}")
        else:
            print("   No bounds set")
        
        # Constraints summary (SMC_F_J only)
        if self._is_smc_fj_mode():
            print(f"\n[INQ] Inequality Constraints: {len(self._inequality_constraints)} groups")
            total_ineq = sum(c['A'].shape[0] for c in self._inequality_constraints.values())
            print(f"   Total constraints: {total_ineq}")
            
            for name, constraint in self._inequality_constraints.items():
                print(f"   - {name}: {constraint['A'].shape[0]} constraints (source: {constraint['source']})")
            
            print(f"\n[EQ] Equality Constraints: {len(self._equality_constraints)} groups")
            total_eq = sum(c['A'].shape[0] for c in self._equality_constraints.values())
            print(f"   Total constraints: {total_eq}")
            
            for name, constraint in self._equality_constraints.items():
                print(f"   - {name}: {constraint['A'].shape[0]} constraints (source: {constraint['source']})")
        
        print("="*70)

    def validate(self) -> Dict[str, Any]:
        """Validate current configuration.

        Extends base validation with SMC-specific checks (mode compatibility,
        undefined bounds).
        """
        result = super().validate()

        # SMC-specific: mode compatibility
        if self.bayesian_sampling_mode == 'SMC_F_J' and self.slip_sampling_mode != 'ss_ds':
            result['warnings'].append("Linear constraints only supported in SMC_F_J mode with ss_ds sampling")

        # SMC-specific: undefined bounds warning
        undefined_count = 0
        if self._bounds['lb'] is not None:
            undefined_count += int(np.sum(np.isnan(self._bounds['lb'])))
        if self._bounds['ub'] is not None:
            undefined_count += int(np.sum(np.isnan(self._bounds['ub'])))
        if undefined_count > 0:
            result['warnings'].append(f"{undefined_count} bounds are undefined (will use defaults)")

        return result

    # Backward-compatible alias
    validate_configuration = validate

    # ==================== Legacy Compatibility Methods ====================
    
    def update_bounds_from_config(self, config_file=None, encoding='utf-8'):
        """Legacy method - use load_bounds_config and apply_bounds_from_config instead."""
        if config_file is not None:
            self.load_bounds_config(config_file, encoding)
        self.apply_bounds_from_config()

    # SMC-specific property
    @property
    def bounds_config(self):
        """Bounds config for compatibility."""
        return self._bounds_config


# Backward-compatible alias
ConstraintManager = ConstraintManagerSMC