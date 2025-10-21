import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from datetime import datetime
from pathlib import Path

class ConstraintManager:
    """
    Complete constraint and bounds management system.
    
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
        
        # Initialize bounds storage with detailed tracking
        self._bounds = {
            'lb': None,  # Global lower bounds array
            'ub': None,  # Global upper bounds array
            'global': {'lb': None, 'ub': None},  # Global defaults
            'strikeslip': {},  # Per-fault strike-slip bounds
            'dipslip': {},     # Per-fault dip-slip bounds  
            'poly': {},        # Per-fault polynomial bounds
            'source': None,
            'config_file': None,
            'applied_time': None
        }
        
        # Configuration storage
        self._bounds_config = None
        
        # Constraint storage
        self._inequality_constraints = {}  # name -> constraint_dict
        self._equality_constraints = {}    # name -> constraint_dict
        
        # Combined constraints cache
        self._combined_cache = {
            'inequality': {'A': None, 'b': None, 'valid': False},
            'equality': {'A': None, 'b': None, 'valid': False}
        }
        
        if self.verbose:
            print(f"🔧 Complete ConstraintManager initialized")

    def load_bounds_config(self, config_file: str, encoding: str = 'utf-8'):
        """
        Load bounds configuration from file.
        
        Parameters:
        -----------
        config_file : str
            Path to bounds configuration file
        encoding : str
            File encoding
        """
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Bounds config file not found: {config_file}")
            
            with open(config_path, 'r', encoding=encoding) as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    self._bounds_config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            self._bounds['config_file'] = str(config_path)
            
            # Store bounds_config in solver for compatibility
            self.solver.bounds_config = self._bounds_config
            
            if self.verbose:
                print(f"📁 Loaded bounds config from: {config_file}")
                self._print_config_summary()
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to load bounds config: {e}")
            raise

    def _print_config_summary(self):
        """Print summary of loaded configuration."""
        if not self._bounds_config:
            return
            
        config_items = []
        if 'lb' in self._bounds_config or 'ub' in self._bounds_config:
            config_items.append("global bounds")
        if 'strikeslip' in self._bounds_config:
            config_items.append(f"strike-slip bounds for {len(self._bounds_config['strikeslip'])} fault(s)")
        if 'dipslip' in self._bounds_config:
            config_items.append(f"dip-slip bounds for {len(self._bounds_config['dipslip'])} fault(s)")
        if 'poly' in self._bounds_config:
            config_items.append(f"polynomial bounds for {len(self._bounds_config['poly'])} fault(s)")
        if 'rake_angle' in self._bounds_config:
            config_items.append(f"rake angle constraints for {len(self._bounds_config['rake_angle'])} fault(s)")
        
        if config_items:
            print(f"   - Configuration contains: {', '.join(config_items)}")

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
                    print("⚠️  Warning: No faults found that match both rake_limits keys and solver.faults")
                return
            
            # Calculate total patches for constrained faults only
            npatch = 0
            Nsd = 0
            Np = self.solver.lsq_parameters  # Total number of parameters
            
            # Get constrained fault objects
            constrained_faults = [fault for fault in self.solver.faults if fault.name in constrained_fault_names]
            
            for ifault in constrained_faults:
                inpatch = len(ifault.patch)
                npatch += inpatch
                Nsd += int(inpatch * len(ifault.slipdir))
            
            # Create constraint matrices - note: Nsd constraints for Nsd/2 patches (2 constraints per patch)
            A = np.zeros((Nsd, Np))
            b = np.zeros((Nsd,))
            
            patch_count = 0
            for ifault in constrained_faults:
                inpatch = len(ifault.patch)
                start = self.solver.fault_indexes[ifault.name][0]
                half = start + inpatch
                
                # Get the rake angle bounds
                rake_start, rake_end = rake_limits[ifault.name]
                
                # Generate the linear constraints for each patch
                for i in range(inpatch):
                    # Lower bound constraint: ss*sin(rake_start) - ds*cos(rake_start) <= 0
                    A[patch_count + i, start + i] = np.sin(np.deg2rad(rake_start))
                    A[patch_count + i, half + i] = -np.cos(np.deg2rad(rake_start))
                    
                    # Upper bound constraint: -ss*sin(rake_end) + ds*cos(rake_end) <= 0
                    A[patch_count + inpatch + i, start + i] = -np.sin(np.deg2rad(rake_end))
                    A[patch_count + inpatch + i, half + i] = np.cos(np.deg2rad(rake_end))
                
                patch_count += inpatch
            
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
                print(f"🔻 Applied rake angle constraints: {A.shape[0]} constraints for {len(constrained_fault_names)} fault(s)")
                for fault_name, (min_rake, max_rake) in rake_limits.items():
                    if fault_name in constrained_fault_names:
                        print(f"   • {fault_name}: {min_rake}° ≤ rake ≤ {max_rake}°")
                        
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to apply rake angle constraints: {e}")
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
                    print("ℹ️  No rake angle limits specified")
                return
            
            # Apply rake angle constraints
            source = 'bounds_config'
            if additional_rake_limits:
                source += ' + additional_limits'
            self.set_rake_angle_constraints(final_rake_limits, source=source)

        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to apply rake constraints: {e}")
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
                    print("⚠️  Warning: No faults found that match both fixed_rake keys and solver.faults")
                return
            
            # Calculate total patches for constrained faults
            npatch = 0
            Np = self.solver.lsq_parameters
            
            # Get constrained fault objects
            constrained_faults = [fault for fault in self.solver.faults if fault.name in constrained_fault_names]
            
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
                half = start + inpatch
                rake_angle = np.deg2rad(irake)
                
                # Generate equality constraints for each patch
                for i in range(inpatch):
                    # Fixed rake constraint: ss*sin(rake) - ds*cos(rake) = 0
                    Aeq[patch_count + i, start + i] = np.sin(rake_angle)
                    Aeq[patch_count + i, half + i] = -np.cos(rake_angle)
                
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
                print(f"🔺 Applied fixed rake constraints: {Aeq.shape[0]} constraints for {len(constrained_fault_names)} fault(s)")
                for fault_name, rake_angle in fixed_rake.items():
                    if fault_name in constrained_fault_names:
                        print(f"   • {fault_name}: rake = {rake_angle}°")
                        
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to apply fixed rake constraints: {e}")
            raise

    def apply_euler_constraints(self):
        """Apply Euler constraints."""
        try:
            if not hasattr(self.config, 'euler_constraints') or not self.config.euler_constraints.get('enabled', False):
                if self.verbose:
                    print("ℹ️  Euler constraints not enabled in config")
                return
            
            from .euler_inequality_constraints import generate_euler_inequality_constraints
            
            euler_config = self.config.euler_constraints
            all_datasets = self.config.geodata['data']
            
            # Generate constraints
            A_ineq, b_ineq = generate_euler_inequality_constraints(self.solver, euler_config, all_datasets)
            
            if A_ineq is not None and A_ineq.size > 0:
                self._inequality_constraints['euler_constraints'] = {
                    'A': A_ineq.copy(),
                    'b': b_ineq.copy(),
                    'source': 'config.euler_constraints',
                    'shape': A_ineq.shape,
                    'added_time': datetime.now()
                }
                # Invalidate cache
                self._combined_cache['inequality']['valid'] = False
                
                if self.verbose:
                    print(f"🔻 Applied Euler constraints: {A_ineq.shape[0]} constraints")
                    configured_faults = euler_config.get('configured_faults', [])
                    print(f"   - Constrained faults: {configured_faults}")
            elif self.verbose:
                print("ℹ️  No Euler constraints generated")
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to apply Euler constraints: {e}")
            raise

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
            print("\n🚀 Applying all constraints from configuration...")
        
        # Load bounds config if provided
        if bounds_config_file is not None:
            self.load_bounds_config(bounds_config_file, encoding)
        
        # Apply bounds from config
        if self._bounds_config is not None and hasattr(self.config, 'use_bounds_constraints') and self.config.use_bounds_constraints:
            self.apply_bounds_from_config()
        
        # Apply rake angle constraints
        if hasattr(self.config, 'use_rake_angle_constraints') and self.config.use_rake_angle_constraints:
            self.apply_rake_constraints(rake_limits)
        
        # Apply Euler constraints
        if hasattr(self.config, 'euler_constraints') and self.config.euler_constraints.get('enabled', False):
            self.apply_euler_constraints()
        
        if self.verbose:
            print("✅ All constraints applied successfully")
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
            print(f"🌐 Set global bounds: lb={lb}, ub={ub} (source: {source})")

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
            print(f"⚡ Set slip bounds for '{fault_name}': ss={strikeslip}, ds={dipslip} (source: {source})")

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
            print(f"📐 Set poly bounds for '{fault_name}': {poly_bounds} (source: {source})")

    def apply_bounds_from_config(self):
        """Apply all bounds from loaded configuration."""
        if not self._bounds_config:
            if self.verbose:
                print("⚠️  No bounds config loaded")
            return
        
        # Initialize parameter arrays if needed
        self._initialize_bounds_arrays()
        
        # Apply global bounds
        lb = self._bounds_config.get('lb', None)
        ub = self._bounds_config.get('ub', None)
        if lb is not None or ub is not None:
            self.set_global_bounds(lb, ub, source="config_file")
        
        # Apply fault-specific slip bounds
        strikeslip_config = self._bounds_config.get('strikeslip', {})
        dipslip_config = self._bounds_config.get('dipslip', {})
        
        all_slip_faults = set(strikeslip_config.keys()) | set(dipslip_config.keys())
        for fault_name in all_slip_faults:
            if self._fault_exists(fault_name):
                ss_bounds = strikeslip_config.get(fault_name, None)
                ds_bounds = dipslip_config.get(fault_name, None)
                self.set_fault_slip_bounds(fault_name, ss_bounds, ds_bounds, source="config_file")
        
        # Apply polynomial bounds
        poly_config = self._bounds_config.get('poly', {})
        for fault_name, poly_bounds in poly_config.items():
            if self._fault_exists(fault_name):
                self.set_fault_poly_bounds(fault_name, poly_bounds, source="config_file")
        
        self._bounds['source'] = "config_file"
        self._bounds['applied_time'] = datetime.now()

    def _initialize_bounds_arrays(self):
        """Initialize bounds arrays based on solver parameters."""
        if hasattr(self.solver, 'lsq_parameters'):
            n_params = self.solver.lsq_parameters
        else:
            # Try to calculate from faults
            n_params = 0
            for fault in self.solver.faults:
                n_params += len(fault.patch) * len(fault.slipdir)
                # Add polynomial parameters if any
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
            st, se = self.solver.slip_positions[fault_name]
            half = (st + se) // 2
            slb, sub = bounds
            self._bounds['lb'][st:half] = slb
            self._bounds['ub'][st:half] = sub

    def _apply_dipslip_bounds(self, fault_name: str, bounds: Tuple[float, float]):
        """Apply dip-slip bounds to specific fault."""
        if hasattr(self.solver, 'slip_positions'):
            st, se = self.solver.slip_positions[fault_name]
            half = (st + se) // 2
            dlb, dub = bounds
            self._bounds['lb'][half:se] = dlb
            self._bounds['ub'][half:se] = dub

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

    def add_inequality_constraint(self, A: np.ndarray, b: np.ndarray, name: str, 
                                source: str = "manual", overwrite: bool = False):
        """Add inequality constraint A @ x <= b."""
        if name in self._inequality_constraints and not overwrite:
            raise ValueError(f"Inequality constraint '{name}' already exists. Use overwrite=True to replace.")
        
        A = np.asarray(A)
        b = np.asarray(b)
        
        if A.ndim != 2:
            raise ValueError(f"Constraint matrix A must be 2D, got shape {A.shape}")
        
        if A.shape[0] != len(b):
            raise ValueError(f"A.shape[0] ({A.shape[0]}) != len(b) ({len(b)})")
        
        # Store constraint
        self._inequality_constraints[name] = {
            'A': A.copy(),
            'b': b.copy(),
            'source': source,
            'shape': A.shape,
            'added_time': datetime.now()
        }
        
        # Invalidate cache
        self._combined_cache['inequality']['valid'] = False
        
        if self.verbose:
            print(f"🔻 Added inequality constraint '{name}': {A.shape[0]} constraints (source: {source})")

    def add_equality_constraint(self, A: np.ndarray, b: np.ndarray, name: str,
                              source: str = "manual", overwrite: bool = False):
        """Add equality constraint A @ x = b."""
        if name in self._equality_constraints and not overwrite:
            raise ValueError(f"Equality constraint '{name}' already exists. Use overwrite=True to replace.")
        
        A = np.asarray(A)
        b = np.asarray(b)
        
        if A.ndim != 2:
            raise ValueError(f"Constraint matrix A must be 2D, got shape {A.shape}")
        
        if A.shape[0] != len(b):
            raise ValueError(f"A.shape[0] ({A.shape[0]}) != len(b) ({len(b)})")
        
        # Store constraint
        self._equality_constraints[name] = {
            'A': A.copy(),
            'b': b.copy(),
            'source': source,
            'shape': A.shape,
            'added_time': datetime.now()
        }
        
        # Invalidate cache
        self._combined_cache['equality']['valid'] = False
        
        if self.verbose:
            print(f"🔺 Added equality constraint '{name}': {A.shape[0]} constraints (source: {source})")

    def remove_constraint(self, name: str, constraint_type: Optional[str] = None):
        """Remove constraint by name."""
        removed = False
        
        if constraint_type is None or constraint_type == 'inequality':
            if name in self._inequality_constraints:
                del self._inequality_constraints[name]
                self._combined_cache['inequality']['valid'] = False
                removed = True
                constraint_type = 'inequality'
        
        if constraint_type is None or constraint_type == 'equality':
            if name in self._equality_constraints:
                del self._equality_constraints[name]
                self._combined_cache['equality']['valid'] = False
                removed = True
                constraint_type = 'equality'
        
        if removed:
            if self.verbose:
                print(f"❌ Removed {constraint_type} constraint: '{name}'")
        else:
            raise ValueError(f"Constraint '{name}' not found")

    def get_combined_inequality_constraints(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get combined inequality constraints as (A, b)."""
        if not self._inequality_constraints:
            return None, None
        
        cache = self._combined_cache['inequality']
        if cache['valid']:
            return cache['A'].copy() if cache['A'] is not None else None, \
                   cache['b'].copy() if cache['b'] is not None else None
        
        # Rebuild combined constraints
        A_list = []
        b_list = []
        
        for constraint in self._inequality_constraints.values():
            A_list.append(constraint['A'])
            b_list.append(constraint['b'])
        
        A_combined = np.vstack(A_list)
        b_combined = np.hstack(b_list)
        
        # Update cache
        cache['A'] = A_combined.copy()
        cache['b'] = b_combined.copy()
        cache['valid'] = True
        
        return A_combined, b_combined

    def get_combined_equality_constraints(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get combined equality constraints as (A, b)."""
        if not self._equality_constraints:
            return None, None
        
        cache = self._combined_cache['equality']
        if cache['valid']:
            return cache['A'].copy() if cache['A'] is not None else None, \
                   cache['b'].copy() if cache['b'] is not None else None
        
        # Rebuild combined constraints
        A_list = []
        b_list = []
        
        for constraint in self._equality_constraints.values():
            A_list.append(constraint['A'])
            b_list.append(constraint['b'])
        
        A_combined = np.vstack(A_list)
        b_combined = np.hstack(b_list)
        
        # Update cache
        cache['A'] = A_combined.copy()
        cache['b'] = b_combined.copy()
        cache['valid'] = True
        
        return A_combined, b_combined

    def sync_to_solver(self):
        """Synchronize all constraints and bounds to solver attributes."""
        # Sync bounds arrays
        if self._bounds['lb'] is not None:
            self.solver.lb = self._bounds['lb'].copy()
        if self._bounds['ub'] is not None:
            self.solver.ub = self._bounds['ub'].copy()
        
        # Sync bounds metadata
        if not hasattr(self.solver, 'bounds'):
            self.solver.bounds = {}
        
        self.solver.bounds.update({
            'lb': self._bounds['global']['lb'],
            'ub': self._bounds['global']['ub'],
            'strikeslip': self._bounds['strikeslip'].copy(),
            'dipslip': self._bounds['dipslip'].copy(),
            'poly': self._bounds['poly'].copy()
        })
        
        # Sync inequality constraints
        if not hasattr(self.solver, 'inequality_constraints'):
            self.solver.inequality_constraints = {}
        
        for name, constraint in self._inequality_constraints.items():
            self.solver.inequality_constraints[name] = {'A': constraint['A'], 'b': constraint['b']}
        
        # Sync equality constraints  
        if not hasattr(self.solver, 'equality_constraints'):
            self.solver.equality_constraints = {}
        
        for name, constraint in self._equality_constraints.items():
            self.solver.equality_constraints[name] = {'A': constraint['A'], 'b': constraint['b']}
        
        # Sync combined constraints for backward compatibility
        A_ineq, b_ineq = self.get_combined_inequality_constraints()
        A_eq, b_eq = self.get_combined_equality_constraints()
        
        # Use setters to avoid AttributeError
        self.solver.A_ueq = A_ineq
        self.solver.b_ueq = b_ineq
        self.solver.Aeq = A_eq
        self.solver.beq = b_eq
        
        if self.verbose:
            print("🔄 Synchronized all constraints and bounds to solver")

    def validate_constraints(self) -> Dict[str, Any]:
        """Validate all constraints for consistency."""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Validate bounds
        if self._bounds['lb'] is not None and self._bounds['ub'] is not None:
            inconsistent = np.where(self._bounds['lb'] > self._bounds['ub'])[0]
            if len(inconsistent) > 0:
                result['errors'].append(
                    f"Inconsistent bounds at {len(inconsistent)} parameter(s): "
                    f"indices {inconsistent[:10].tolist()}" + 
                    ("..." if len(inconsistent) > 10 else "")
                )
                result['valid'] = False
        
        # Validate inequality constraints
        for name, constraint in self._inequality_constraints.items():
            A, b = constraint['A'], constraint['b']
            if A.shape[0] != len(b):
                result['errors'].append(
                    f"Inequality '{name}': A.shape[0] ({A.shape[0]}) != len(b) ({len(b)})"
                )
                result['valid'] = False
        
        # Validate equality constraints
        for name, constraint in self._equality_constraints.items():
            A, b = constraint['A'], constraint['b']
            if A.shape[0] != len(b):
                result['errors'].append(
                    f"Equality '{name}': A.shape[0] ({A.shape[0]}) != len(b) ({len(b)})"
                )
                result['valid'] = False
        
        # Create summary
        result['summary'] = {
            'bounds_set': self._bounds['lb'] is not None or self._bounds['ub'] is not None,
            'inequality_groups': len(self._inequality_constraints),
            'equality_groups': len(self._equality_constraints),
            'total_inequality_constraints': sum(c['A'].shape[0] for c in self._inequality_constraints.values()),
            'total_equality_constraints': sum(c['A'].shape[0] for c in self._equality_constraints.values())
        }
        
        return result

    def print_summary(self):
        """Print comprehensive constraint and bounds summary."""
        print("\n" + "="*70)
        print("COMPLETE CONSTRAINT MANAGER SUMMARY")
        print("="*70)
        
        # Configuration info
        print(f"🔧 CONFIGURATION")
        if self._bounds['config_file']:
            print(f"   Config file: {self._bounds['config_file']}")
        
        if self.config:
            bounds_enabled = getattr(self.config, 'use_bounds_constraints', False)
            rake_enabled = getattr(self.config, 'use_rake_angle_constraints', False)
            euler_enabled = hasattr(self.config, 'euler_constraints') and self.config.euler_constraints.get('enabled', False)
            
            print(f"   Bounds constraints: {'✅ Enabled' if bounds_enabled else '❌ Disabled'}")
            print(f"   Rake constraints: {'✅ Enabled' if rake_enabled else '❌ Disabled'}")
            print(f"   Euler constraints: {'✅ Enabled' if euler_enabled else '❌ Disabled'}")
        
        # Bounds info
        print(f"\n📊 BOUNDS MANAGEMENT")
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
                    print(f"     • {fault}: {bounds}")
            
            if self._bounds['dipslip']:
                print(f"   Dip-slip bounds: {len(self._bounds['dipslip'])} fault(s)")
                for fault, bounds in self._bounds['dipslip'].items():
                    print(f"     • {fault}: {bounds}")
            
            if self._bounds['poly']:
                print(f"   Polynomial bounds: {len(self._bounds['poly'])} fault(s)")
                for fault, bounds in self._bounds['poly'].items():
                    print(f"     • {fault}: {bounds}")
                    
            print(f"   Source: {self._bounds['source']}")
        else:
            print("   No bounds set")
        
        # Inequality constraints
        print(f"\n🔻 INEQUALITY CONSTRAINTS")
        print(f"   Groups: {len(self._inequality_constraints)}")
        total_ineq = sum(c['A'].shape[0] for c in self._inequality_constraints.values())
        print(f"   Total constraints: {total_ineq}")
        
        for name, constraint in self._inequality_constraints.items():
            print(f"   • {name}: {constraint['A'].shape[0]} constraints (source: {constraint['source']})")
        
        # Equality constraints
        print(f"\n🔺 EQUALITY CONSTRAINTS")
        print(f"   Groups: {len(self._equality_constraints)}")
        total_eq = sum(c['A'].shape[0] for c in self._equality_constraints.values())
        print(f"   Total constraints: {total_eq}")
        
        for name, constraint in self._equality_constraints.items():
            print(f"   • {name}: {constraint['A'].shape[0]} constraints (source: {constraint['source']})")
        
        # Validation status
        validation = self.validate_constraints()
        print(f"\n✅ VALIDATION: {'PASSED' if validation['valid'] else 'FAILED'}")
        
        if validation['errors']:
            print("   ❌ Errors:")
            for error in validation['errors']:
                print(f"      {error}")
        
        if validation['warnings']:
            print("   ⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"      {warning}")
        
        print("="*70)

    # Properties for backward compatibility
    @property
    def lb(self) -> Optional[np.ndarray]:
        """Lower bounds array for compatibility."""
        return self._bounds['lb'].copy() if self._bounds['lb'] is not None else None

    @property
    def ub(self) -> Optional[np.ndarray]:
        """Upper bounds array for compatibility."""
        return self._bounds['ub'].copy() if self._bounds['ub'] is not None else None

    @property
    def inequality_constraints(self) -> Dict[str, Dict]:
        """Inequality constraints dict for compatibility."""
        return {name: {'A': constraint['A'], 'b': constraint['b']} 
                for name, constraint in self._inequality_constraints.items()}

    @property
    def equality_constraints(self) -> Dict[str, Dict]:
        """Equality constraints dict for compatibility."""
        return {name: {'A': constraint['A'], 'b': constraint['b']} 
                for name, constraint in self._equality_constraints.items()}

    @property
    def bounds_config_file(self) -> Optional[str]:
        """Path to bounds config file."""
        return self._bounds['config_file']