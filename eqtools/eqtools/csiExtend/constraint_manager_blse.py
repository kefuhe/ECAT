import numpy as np
import copy
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from datetime import datetime
from pathlib import Path

from .constraint_manager_base import ConstraintManagerBase
from .source_adapters import FaultAdapter


class ConstraintManagerBLSE(ConstraintManagerBase):
    """
    BLSE constraint and bounds management system.
    
    Handles ALL constraint-related operations including:
    - Configuration loading and parsing
    - Bounds constraints (lb <= x <= ub) with per-fault customization
    - Inequality constraints (A_ineq * x <= b_ineq)
    - Equality constraints (A_eq * x = b_eq)
    - Rake angle constraints
    - Euler constraints
    """
    
    def __init__(self, solver, config=None, verbose: bool = True):
        """
        Initialize constraint manager.
        
        Parameters:
        -----------
        solver : object
            The solver instance (multifaultsolve_boundLSE or BoundLSEMultiFaultsInversion)
        config : object, optional
            Configuration object with constraint settings
        verbose : bool
            Enable verbose output
        """
        self.solver = solver
        self.config = config
        self.verbose = verbose
        
        # Shared storage (constraints, cache, common bounds keys)
        self._init_shared_storage()
        
        if self.verbose:
            print(f"[OK] Complete ConstraintManager initialized")

    def _on_bounds_config_loaded(self):
        """Sync bounds config to solver for BLSE compatibility."""
        self.solver.bounds_config = self._bounds_config

    def set_rake_angle_constraints(self, rake_limits: Dict[str, Tuple[float, float]], source: str = "manual"):
        """
        Set rake angle inequality constraints for specified faults.
        
        Parameters:
        -----------
        rake_limits : dict
            Dictionary with fault names as keys and (min_rake, max_rake) tuples as values
        source : str
            Source description
        """
        try:
            # Get fault names that exist in both rake_limits and self.solver.faults
            fault_names = [fault.name for fault in self.solver.faults]
            constrained_fault_names = [name for name in rake_limits.keys() if name in fault_names]
            
            if not constrained_fault_names:
                if self.verbose:
                    print("[!]  Warning: No faults found that match both rake_limits keys and solver.faults")
                return
            
            # Calculate two rake half-plane rows per constrained patch.
            n_rake_rows = 0
            Np = self.solver.lsq_parameters  # Total number of parameters
            
            # Get constrained fault objects. Only Fault-type sources support rake constraints.
            constrained_faults = [fault for fault in self.solver.faults
                                  if fault.name in constrained_fault_names
                                  and self._get_source_type(fault.name) == 'Fault']
            constrained_fault_names = [f.name for f in constrained_faults]
            
            if not constrained_faults:
                if self.verbose:
                    print("[!]  Warning: No Fault-type sources found for rake angle constraints")
                return
            
            for ifault in constrained_faults:
                inpatch = len(ifault.patch)
                n_rake_rows += 2 * inpatch
            
            A = np.zeros((n_rake_rows, Np))
            b = np.zeros((n_rake_rows,))
            
            row_offset = 0
            for ifault in constrained_faults:
                inpatch = len(ifault.patch)
                start = self.solver.fault_indexes[ifault.name][0]
                adapter = getattr(self.solver, 'adapters', {}).get(ifault.name)
                ss_start, ds_start = self._rake_component_starts(
                    ifault, start, inpatch, adapter=adapter
                )
                
                # Get the rake angle bounds
                rake_start, rake_end = self._validate_rake_interval(
                    ifault.name, rake_limits[ifault.name]
                )
                
                # Generate the linear constraints for each patch
                for i in range(inpatch):
                    # Lower bound constraint: ss*sin(rake_start) - ds*cos(rake_start) <= 0
                    A[row_offset + i, ss_start + i] = np.sin(np.deg2rad(rake_start))
                    A[row_offset + i, ds_start + i] = -np.cos(np.deg2rad(rake_start))
                    
                    # Upper bound constraint: -ss*sin(rake_end) + ds*cos(rake_end) <= 0
                    A[row_offset + inpatch + i, ss_start + i] = -np.sin(np.deg2rad(rake_end))
                    A[row_offset + inpatch + i, ds_start + i] = np.cos(np.deg2rad(rake_end))
                
                row_offset += 2 * inpatch
            
            # Store constraint
            self._inequality_constraints['rake_angle'] = {
                'A': A,
                'b': b,
                'source': source,
                'shape': A.shape,
                'added_time': datetime.now()
            }
            
            # Invalidate cache
            self._combined_cache['inequality']['valid'] = False
            
            if self.verbose:
                print(f"[INQ] Applied rake angle constraints: {A.shape[0]} constraints for {len(constrained_fault_names)} fault(s)")
                for fault_name, (min_rake, max_rake) in rake_limits.items():
                    if fault_name in constrained_fault_names:
                        print(f"   - {fault_name}: {min_rake} deg <= rake <= {max_rake} deg")
                        
        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply rake angle constraints: {e}")
            raise

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
            self.set_rake_angle_constraints(final_rake_limits, source=source)

        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply rake constraints: {e}")
            raise

    def set_fixed_rake_constraints(self, fixed_rake: Dict[str, float], source: str = "manual"):
        """
        Set fixed rake angle equality constraints for specified faults.
        
        Parameters:
        -----------
        fixed_rake : dict
            Dictionary with fault names as keys and rake angle values as values (in degrees)
        source : str
            Source description
        """
        try:
            # Get fault names that exist in both fixed_rake and self.solver.faults
            fault_names = [fault.name for fault in self.solver.faults]
            constrained_fault_names = [name for name in fixed_rake.keys() if name in fault_names]
            
            if not constrained_fault_names:
                if self.verbose:
                    print("[!]  Warning: No faults found that match both fixed_rake keys and solver.faults")
                return
            
            # Calculate total patches for constrained faults
            npatch = 0
            Np = self.solver.lsq_parameters
            
            # Get constrained fault objects. Only Fault-type sources have rake.
            constrained_faults = [fault for fault in self.solver.faults
                                  if fault.name in constrained_fault_names
                                  and self._get_source_type(fault.name) == 'Fault']
            constrained_fault_names = [f.name for f in constrained_faults]
            
            if not constrained_faults:
                if self.verbose:
                    print("[!]  Warning: No Fault-type sources found for fixed rake constraints")
                return
            
            for ifault in constrained_faults:
                npatch += len(ifault.patch)
            
            # Create constraint matrices
            Aeq = np.zeros((npatch, Np))
            beq = np.zeros((npatch,))
            
            patch_count = 0
            for ifault in constrained_faults:
                irake = fixed_rake[ifault.name]
                inpatch = len(ifault.patch)
                start = self.solver.fault_indexes[ifault.name][0]
                adapter = getattr(self.solver, 'adapters', {}).get(ifault.name)
                ss_start, ds_start = self._rake_component_starts(
                    ifault, start, inpatch, adapter=adapter
                )
                rake_angle = np.deg2rad(irake)
                
                # Generate equality constraints for each patch
                for i in range(inpatch):
                    # Fixed rake constraint: ss*sin(rake) - ds*cos(rake) = 0
                    Aeq[patch_count + i, ss_start + i] = np.sin(rake_angle)
                    Aeq[patch_count + i, ds_start + i] = -np.cos(rake_angle)
                
                patch_count += inpatch
            
            # Store constraint
            self._equality_constraints['fixed_rake'] = {
                'A': Aeq,
                'b': beq,
                'source': source,
                'shape': Aeq.shape,
                'added_time': datetime.now()
            }
            
            # Invalidate cache
            self._combined_cache['equality']['valid'] = False
            
            if self.verbose:
                print(f"[EQ] Applied fixed rake constraints: {Aeq.shape[0]} constraints for {len(constrained_fault_names)} fault(s)")
                for fault_name, rake_angle in fixed_rake.items():
                    if fault_name in constrained_fault_names:
                        print(f"   - {fault_name}: rake = {rake_angle} deg")
                        
        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply fixed rake constraints: {e}")
            raise

    def apply_euler_cap_constraints(self):
        """Apply optional interseismic Euler-cap constraints for Fault sources."""
        try:
            interseismic_config = getattr(self.config, 'interseismic_config', {})
            if not interseismic_config.get('cap_constraints', {}).get('enabled', False):
                if self.verbose:
                    print("[i]  Interseismic Euler-cap constraints not enabled")
                return

            from .euler_inequality_constraints import generate_euler_cap_constraints

            active_config = copy.deepcopy(interseismic_config)
            cap_faults = active_config.get('cap_constraints', {}).get('faults', {})
            non_fault_names = [
                fn for fn in list(cap_faults)
                if self._get_source_type(fn) != 'Fault'
            ]
            for fn in non_fault_names:
                if self.verbose:
                    print(f"[!]  Warning: Euler-cap constraint skipping non-Fault source '{fn}'")
                del cap_faults[fn]

            all_datasets = self.config.geodata['data']
            A_ineq, b_ineq = generate_euler_cap_constraints(self.solver, active_config, all_datasets)

            if A_ineq is not None and A_ineq.size > 0:
                self._inequality_constraints['euler_cap_constraints'] = {
                    'A': A_ineq.copy(),
                    'b': b_ineq.copy(),
                    'source': 'interseismic_config.cap_constraints',
                    'shape': A_ineq.shape,
                    'added_time': datetime.now()
                }
                self._combined_cache['inequality']['valid'] = False

                if self.verbose:
                    print(f"[INQ] Applied Euler-cap constraints: {A_ineq.shape[0]} constraints")
                    configured_faults = active_config.get('cap_constraints', {}).get('configured_faults', [])
                    print(f"   - Constrained faults: {configured_faults}")
            elif self.verbose:
                print("[i]  No Euler-cap constraints generated")

        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply Euler-cap constraints: {e}")
            raise

    def apply_interseismic_backslip_constraints(self):
        """Apply hard backslip/coupling constraints from interseismic_config."""
        constraints = getattr(self.config, 'interseismic_config', {}).get('backslip_constraints', [])
        for index, spec in enumerate(constraints):
            if self._get_source_type(spec['fault']) != 'Fault':
                if self.verbose:
                    print(f"[!]  Warning: Interseismic backslip constraint skipping non-Fault source '{spec['fault']}'")
                continue
            self.solver.add_interseismic_backslip_constraint(
                spec['fault'],
                spec['state'],
                selector=spec.get('selector'),
                component=spec.get('component', 'strikeslip'),
                coupling=spec.get('coupling'),
                value=spec.get('value'),
                name=spec.get('name', f"interseismic_backslip_{index}"),
                overwrite=spec.get('overwrite', True),
                source='interseismic_config.backslip_constraints',
            )

    def apply_interseismic_block_constraints(self):
        """Apply block-level Euler sharing constraints from interseismic_config."""
        interseismic_config = getattr(self.config, 'interseismic_config', {})
        try:
            from .interseismic_parameter_model import generate_block_euler_equality_constraints

            A_eq, b_eq = generate_block_euler_equality_constraints(
                self.solver,
                interseismic_config,
                n_total=int(self.solver.lsq_parameters),
            )
            if A_eq is None or A_eq.size == 0:
                if self.verbose:
                    print("[i]  No interseismic block Euler-sharing constraints generated")
                return
            self.add_equality_constraint(
                A_eq,
                b_eq,
                name='interseismic_block_euler_constraints',
                source='interseismic_config.blocks',
                overwrite=True,
            )
        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply interseismic block constraints: {e}")
            raise

    def apply_source_constraints_from_config(self):
        """Apply source-specific inequality/equality constraints from ``source_constraints`` config.

        The ``source_constraints`` section in the bounds config YAML maps each source
        name to a list of constraint definitions.  Each definition contains:
        - ``name``: constraint identifier
        - ``type``: ``'inequality'`` or ``'equality'``
        - ``rule``: a recognised pattern (e.g. ``'pressure >= 0'``)

        The method delegates to each source's adapter
        ``generate_source_inequality_constraints`` /
        ``generate_source_equality_constraints`` to build the actual ``A, b`` matrices.
        """
        if not self._bounds_config or 'source_constraints' not in self._bounds_config:
            return

        source_constraints_cfg = self._bounds_config['source_constraints']
        if not source_constraints_cfg:
            return

        if not hasattr(self.solver, 'adapters'):
            if self.verbose:
                print("[!]  Warning: solver has no adapters, skipping source_constraints")
            return

        n_total = self.solver.lsq_parameters if hasattr(self.solver, 'lsq_parameters') else 0
        if n_total == 0:
            return

        for source_name, src_cfg in source_constraints_cfg.items():
            if not self._fault_exists(source_name):
                if self.verbose:
                    print(f"[!]  Warning: Source '{source_name}' not found, skipping source_constraints")
                continue
            if source_name not in self.solver.adapters:
                if self.verbose:
                    print(f"[!]  Warning: No adapter for '{source_name}', skipping source_constraints")
                continue

            adapter = self.solver.adapters[source_name]
            param_start = self.solver.slip_positions[source_name][0] if hasattr(self.solver, 'slip_positions') else 0

            # Normalise list-of-dicts 鈫?dict-of-dicts keyed by constraint name
            constraints_dict = self._normalise_constraint_list(src_cfg)

            # Inequality constraints
            for cname, A, b in adapter.generate_source_inequality_constraints(
                    constraints_dict, param_start, n_total):
                full_name = f"src_{source_name}_{cname}"
                self.add_inequality_constraint(A, b, name=full_name,
                                               source=f"source_constraints/{source_name}",
                                               overwrite=True)

            # Equality constraints
            for cname, A, b in adapter.generate_source_equality_constraints(
                    constraints_dict, param_start, n_total):
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
        if self._bounds_config is not None and hasattr(self.config, 'use_bounds_constraints') and self.config.use_bounds_constraints:
            self.apply_bounds_from_config()
        
        # Apply rake angle constraints
        if hasattr(self.config, 'use_rake_angle_constraints') and self.config.use_rake_angle_constraints:
            self.apply_rake_constraints(rake_limits)
        
        # Apply optional interseismic constraints.
        interseismic_config = getattr(self.config, 'interseismic_config', {})
        if interseismic_config.get('blocks', {}).get('enabled', False):
            self.apply_interseismic_block_constraints()
        if interseismic_config.get('cap_constraints', {}).get('enabled', False):
            self.apply_euler_cap_constraints()
        if interseismic_config.get('backslip_constraints'):
            self.apply_interseismic_backslip_constraints()
        
        # Apply source-specific constraints from source_constraints config
        if self._bounds_config and 'source_constraints' in self._bounds_config:
            self.apply_source_constraints_from_config()
        
        if self.verbose:
            print("[OK] All constraints applied successfully")
            self.print_summary()

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
        
        if lb is not None:
            self._bounds['global']['lb'] = lb
        if ub is not None:
            self._bounds['global']['ub'] = ub
        
        # Apply to parameter arrays
        self._apply_global_bounds_to_arrays(lb, ub)
        
        if self.verbose:
            print(f"[GLB] Set global bounds: lb={lb}, ub={ub} (source: {source})")

    def set_fault_slip_bounds(self, fault_name: str, strikeslip: Tuple[float, float] = None, 
                            dipslip: Tuple[float, float] = None, source: str = "manual"):
        """
        Set slip bounds for a specific fault.
        
        Parameters:
        -----------
        fault_name : str
            Name of the fault
        strikeslip : tuple, optional
            (lower_bound, upper_bound) for strike-slip
        dipslip : tuple, optional
            (lower_bound, upper_bound) for dip-slip
        source : str
            Source description
        """
        if not self._fault_exists(fault_name):
            raise ValueError(f"Fault '{fault_name}' not found in solver")
        
        if strikeslip is not None:
            slb, sub = strikeslip
            if slb > sub:
                raise ValueError(f"Strike-slip lower bound ({slb}) > upper bound ({sub})")
            self._bounds['strikeslip'][fault_name] = strikeslip
            self._apply_strikeslip_bounds(fault_name, strikeslip)
            
        if dipslip is not None:
            dlb, dub = dipslip
            if dlb > dub:
                raise ValueError(f"Dip-slip lower bound ({dlb}) > upper bound ({dub})")
            self._bounds['dipslip'][fault_name] = dipslip
            self._apply_dipslip_bounds(fault_name, dipslip)
        
        if self.verbose:
            print(f"[*] Set slip bounds for '{fault_name}': ss={strikeslip}, ds={dipslip} (source: {source})")

    def set_fault_poly_bounds(self, fault_name: str, poly_bounds: Tuple[float, float], source: str = "manual"):
        """
        Set polynomial parameter bounds for a specific fault.
        
        Parameters:
        -----------
        fault_name : str
            Name of the fault
        poly_bounds : tuple
            (lower_bound, upper_bound) for polynomial parameters
        source : str
            Source description
        """
        if not self._fault_exists(fault_name):
            raise ValueError(f"Fault '{fault_name}' not found in solver")
        
        plb, pub = poly_bounds
        if plb > pub:
            raise ValueError(f"Polynomial lower bound ({plb}) > upper bound ({pub})")
        
        self._bounds['poly'][fault_name] = poly_bounds
        self._apply_poly_bounds(fault_name, poly_bounds)
        
        if self.verbose:
            print(f"[GEO] Set poly bounds for '{fault_name}': {poly_bounds} (source: {source})")

    def set_source_component_bounds(self, source_name: str, comp_bounds: Dict[str, Tuple[float, float]], 
                                     source: str = "manual"):
        """
        Set per-component bounds for any source type using source adapters.
        
        Parameters
        ----------
        source_name : str
            Name of the source (fault/pressure/sbarbot).
        comp_bounds : dict
            {component_name: (lb, ub)}; component names come from
            adapter.get_param_names(), e.g. {'eps12': (-1e-4, 1e-4)}.
        source : str
            Source description for audit trail.
        """
        if not self._fault_exists(source_name):
            raise ValueError(f"Source '{source_name}' not found in solver")
        
        if not hasattr(self.solver, 'adapters') or source_name not in self.solver.adapters:
            raise ValueError(f"No adapter found for source '{source_name}'")
        
        adapter = self.solver.adapters[source_name]
        params_per_comp = adapter.get_n_params_per_component()
        slip_st, _ = self.solver.slip_positions[source_name]
        
        self._initialize_bounds_arrays()
        
        offset = slip_st
        for comp_name in adapter.get_param_names():
            n = params_per_comp[comp_name]
            if comp_name in comp_bounds:
                clb, cub = comp_bounds[comp_name]
                if clb > cub:
                    raise ValueError(f"Lower bound ({clb}) > upper bound ({cub}) for {comp_name}")
                self._bounds['lb'][offset:offset + n] = clb
                self._bounds['ub'][offset:offset + n] = cub
            offset += n
        
        if self.verbose:
            print(f"[SRC] Set component bounds for '{source_name}': {comp_bounds} (source: {source})")

    def apply_bounds_from_config(self):
        """Apply all bounds from loaded configuration."""
        if not self._bounds_config:
            if self.verbose:
                print("[!]  No bounds config loaded")
            return
        
        # Initialize parameter arrays if needed
        self._initialize_bounds_arrays()
        
        # Apply global bounds
        lb = self._bounds_config.get('lb', None)
        ub = self._bounds_config.get('ub', None)
        if lb is not None or ub is not None:
            self.set_global_bounds(lb, ub, source="config_file")
        
        # Apply fault-specific slip bounds (legacy Fault-only keys)
        strikeslip_config = self._bounds_config.get('strikeslip', {})
        dipslip_config = self._bounds_config.get('dipslip', {})
        
        all_slip_faults = set(strikeslip_config.keys()) | set(dipslip_config.keys())
        for fault_name in all_slip_faults:
            if self._fault_exists(fault_name):
                # Only Fault-type sources have strikeslip/dipslip semantics
                if self._get_source_type(fault_name) != 'Fault':
                    if self.verbose:
                        print(f"[!]  Warning: '{fault_name}' is not a Fault source, "
                              f"skipping strikeslip/dipslip bounds. Use 'source_bounds' instead.")
                    continue
                ss_bounds = strikeslip_config.get(fault_name, None)
                ds_bounds = dipslip_config.get(fault_name, None)
                self.set_fault_slip_bounds(fault_name, ss_bounds, ds_bounds, source="config_file")
        
        # Apply polynomial bounds
        poly_config = self._bounds_config.get('poly', {})
        for fault_name, poly_bounds in poly_config.items():
            if self._fault_exists(fault_name):
                self.set_fault_poly_bounds(fault_name, poly_bounds, source="config_file")
        
        # Apply generic source component bounds (works for Pressure, Sbarbot, etc.)
        source_bounds_config = self._bounds_config.get('source_bounds', {})
        for source_name, comp_bounds in source_bounds_config.items():
            if self._fault_exists(source_name):
                self.set_source_component_bounds(source_name, comp_bounds, source="config_file")
        
        self._bounds['source'] = "config_file"
        self._bounds['applied_time'] = datetime.now()

    def _initialize_bounds_arrays(self):
        """Initialize bounds arrays based on solver parameters."""
        if hasattr(self.solver, 'lsq_parameters'):
            n_params = self.solver.lsq_parameters
        else:
            # Fallback: calculate from adapters if available, else from source attributes
            n_params = 0
            for fault in self.solver.faults:
                if hasattr(self.solver, 'adapters') and fault.name in self.solver.adapters:
                    n_params += self.solver.adapters[fault.name].get_n_source_params()
                elif hasattr(fault, 'patch') and hasattr(fault, 'slipdir'):
                    n_params += len(fault.patch) * len(FaultAdapter._canonicalize_slipdir(fault.slipdir))
                elif hasattr(fault, 'volumes') and hasattr(fault, 'strain_components'):
                    n_params += len(fault.volumes) * len(fault.strain_components)
                else:
                    n_params += 1  # Point source (Pressure)
                if hasattr(fault, 'numberofpolys'):
                    n_params += sum(fault.numberofpolys.values())
        
        if self._bounds['lb'] is None:
            self._bounds['lb'] = np.ones(n_params) * np.nan
        if self._bounds['ub'] is None:
            self._bounds['ub'] = np.ones(n_params) * np.nan

    def _apply_global_bounds_to_arrays(self, lb: float = None, ub: float = None):
        """Apply global bounds to parameter arrays."""
        self._initialize_bounds_arrays()
        
        if lb is not None:
            self._bounds['lb'][np.isnan(self._bounds['lb'])] = lb
        if ub is not None:
            self._bounds['ub'][np.isnan(self._bounds['ub'])] = ub

    def _apply_strikeslip_bounds(self, fault_name: str, bounds: Tuple[float, float]):
        """Apply strike-slip bounds to specific fault."""
        if hasattr(self.solver, 'slip_positions'):
            st, _ = self.solver.slip_positions[fault_name]
            fault = next(f for f in self.solver.faults if f.name == fault_name)
            adapter = getattr(self.solver, 'adapters', {}).get(fault_name)
            component_slices = self._source_component_slices(
                fault, st, adapter=adapter
            )
            if 'strikeslip' not in component_slices:
                raise ValueError(
                    f"Fault '{fault_name}' has no strikeslip component for bounds"
                )
            slb, sub = bounds
            ss_slice = component_slices['strikeslip']
            self._bounds['lb'][ss_slice] = slb
            self._bounds['ub'][ss_slice] = sub

    def _apply_dipslip_bounds(self, fault_name: str, bounds: Tuple[float, float]):
        """Apply dip-slip bounds to specific fault."""
        if hasattr(self.solver, 'slip_positions'):
            st, _ = self.solver.slip_positions[fault_name]
            fault = next(f for f in self.solver.faults if f.name == fault_name)
            adapter = getattr(self.solver, 'adapters', {}).get(fault_name)
            component_slices = self._source_component_slices(
                fault, st, adapter=adapter
            )
            if 'dipslip' not in component_slices:
                raise ValueError(
                    f"Fault '{fault_name}' has no dipslip component for bounds"
                )
            dlb, dub = bounds
            ds_slice = component_slices['dipslip']
            self._bounds['lb'][ds_slice] = dlb
            self._bounds['ub'][ds_slice] = dub

    def _apply_poly_bounds(self, fault_name: str, bounds: Tuple[float, float]):
        """Apply polynomial bounds to specific fault."""
        if hasattr(self.solver, 'poly_positions'):
            st, se = self.solver.poly_positions[fault_name]
            plb, pub = bounds
            self._bounds['lb'][st:se] = plb
            self._bounds['ub'][st:se] = pub

    def _fault_exists(self, fault_name: str) -> bool:
        """Check if fault exists in solver."""
        return any(fault.name == fault_name for fault in self.solver.faults)

    def _get_source_type(self, fault_name):
        """Get source type string for a given source name, using adapter if available."""
        if hasattr(self.solver, 'adapters') and fault_name in self.solver.adapters:
            return self.solver.adapters[fault_name].source_type
        fault_obj = next((f for f in self.solver.faults if f.name == fault_name), None)
        return getattr(fault_obj, 'type', 'Fault') if fault_obj else 'Fault'

    def sync_to_solver(self):
        """Refresh legacy solver-side caches from the constraint manager.

        BLSE/VCE solve paths read bounds and linear constraints directly from
        ``constraint_manager``.  This method keeps older inspection attributes
        such as ``solver.lb`` and ``solver.A_ueq`` current without creating a
        second writable constraint state on the solver.
        """
        # Sync legacy bounds arrays
        if self._bounds['lb'] is not None:
            self.solver.lb = self._bounds['lb'].copy()
        if self._bounds['ub'] is not None:
            self.solver.ub = self._bounds['ub'].copy()

        # Sync combined constraints for backward-compatible readers.
        A_ineq, b_ineq = self.get_combined_inequality_constraints()
        A_eq, b_eq = self.get_combined_equality_constraints()

        # Use setters to avoid AttributeError.
        self.solver.A_ueq = A_ineq
        self.solver.b_ueq = b_ineq
        self.solver.Aeq = A_eq
        self.solver.beq = b_eq

        if self.verbose:
            print("[SYNC] Refreshed legacy solver constraint cache")

    def validate(self) -> Dict[str, Any]:
        """Validate all constraints for consistency.

        Extends base validation with a summary dict.
        """
        result = super().validate()
        result['summary'] = {
            'bounds_set': self._bounds['lb'] is not None or self._bounds['ub'] is not None,
            'inequality_groups': len(self._inequality_constraints),
            'equality_groups': len(self._equality_constraints),
            'total_inequality_constraints': sum(c['A'].shape[0] for c in self._inequality_constraints.values()),
            'total_equality_constraints': sum(c['A'].shape[0] for c in self._equality_constraints.values())
        }
        return result

    # Backward-compatible alias
    validate_constraints = validate

    def print_summary(self):
        """Print comprehensive constraint and bounds summary."""
        print("\n" + "="*70)
        print("COMPLETE CONSTRAINT MANAGER SUMMARY")
        print("="*70)
        
        # Configuration info
        print(f"[OK] CONFIGURATION")
        if self._bounds['config_file']:
            print(f"   Config file: {self._bounds['config_file']}")
        
        if self.config:
            bounds_enabled = getattr(self.config, 'use_bounds_constraints', False)
            rake_enabled = getattr(self.config, 'use_rake_angle_constraints', False)
            interseismic_config = getattr(self.config, 'interseismic_config', {})
            cap_enabled = interseismic_config.get('cap_constraints', {}).get('enabled', False)
            
            print(f"   Bounds constraints: {'[OK] Enabled' if bounds_enabled else '[X] Disabled'}")
            print(f"   Rake constraints: {'[OK] Enabled' if rake_enabled else '[X] Disabled'}")
            print(f"   Euler-cap constraints: {'[OK] Enabled' if cap_enabled else '[X] Disabled'}")
        
        # Bounds info
        print(f"\n[STAT] BOUNDS MANAGEMENT")
        if self._bounds['lb'] is not None or self._bounds['ub'] is not None:
            n_params = len(self._bounds['lb']) if self._bounds['lb'] is not None else len(self._bounds['ub'])
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
            
            # Per-fault bounds
            if self._bounds['strikeslip']:
                print(f"   Strike-slip bounds: {len(self._bounds['strikeslip'])} fault(s)")
                for fault, bounds in self._bounds['strikeslip'].items():
                    print(f"     - {fault}: {bounds}")
            
            if self._bounds['dipslip']:
                print(f"   Dip-slip bounds: {len(self._bounds['dipslip'])} fault(s)")
                for fault, bounds in self._bounds['dipslip'].items():
                    print(f"     - {fault}: {bounds}")
            
            if self._bounds['poly']:
                print(f"   Polynomial bounds: {len(self._bounds['poly'])} fault(s)")
                for fault, bounds in self._bounds['poly'].items():
                    print(f"     - {fault}: {bounds}")
                    
            print(f"   Source: {self._bounds['source']}")
        else:
            print("   No bounds set")
        
        # Inequality constraints
        print(f"\n[INQ] INEQUALITY CONSTRAINTS")
        print(f"   Groups: {len(self._inequality_constraints)}")
        total_ineq = sum(c['A'].shape[0] for c in self._inequality_constraints.values())
        print(f"   Total constraints: {total_ineq}")
        
        for name, constraint in self._inequality_constraints.items():
            print(f"   - {name}: {constraint['A'].shape[0]} constraints (source: {constraint['source']})")
        
        # Equality constraints
        print(f"\n[EQ] EQUALITY CONSTRAINTS")
        print(f"   Groups: {len(self._equality_constraints)}")
        total_eq = sum(c['A'].shape[0] for c in self._equality_constraints.values())
        print(f"   Total constraints: {total_eq}")
        
        for name, constraint in self._equality_constraints.items():
            print(f"   - {name}: {constraint['A'].shape[0]} constraints (source: {constraint['source']})")
        
        # Validation status
        validation = self.validate_constraints()
        print(f"\n[OK] VALIDATION: {'PASSED' if validation['valid'] else 'FAILED'}")
        
        if validation['errors']:
            print("   [X] Errors:")
            for error in validation['errors']:
                print(f"      {error}")
        
        if validation['warnings']:
            print("   [!]  Warnings:")
            for warning in validation['warnings']:
                print(f"      {warning}")
        
        print("="*70)

    # BLSE-specific property
    @property
    def bounds_config_file(self) -> Optional[str]:
        """Path to bounds config file."""
        return self._bounds['config_file']


# Backward-compatible alias
ConstraintManager = ConstraintManagerBLSE
