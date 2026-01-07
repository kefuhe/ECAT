# import the necessary libraries
from csi import multifaultsolve
import copy
import yaml
import numpy as np
import pyproj as pp
from .fnnls import fnnls
from scipy.linalg import block_diag as blkdiag
# import self-written library
from . import lsqlin
from ..plottools import sci_plot_style, DegreeFormatter
from .fault_analysis_mixin import FaultAnalysisMixin
from .constraint_manager_blse import ConstraintManager

# Plot
from eqtools.getcpt import get_cpt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import cmcrameri # cmc.devon_r cmc.lajolla_r cmc.batlow

class multifaultsolve_boundLSE(multifaultsolve, FaultAnalysisMixin):
    '''
    Enhanced multi-fault solver with unified constraint management.
    
    Features:
    - Unified constraint and bounds management via ConstraintManager
    - Laplace smoothing constraints
    - Depth-Equalized Smoothing (DES) support
    - Simple Variance Component Estimation (VCE)
    - Multiple solver backends (lsqlin, fnnls)
    '''
    
    def __init__(self, name, faults, verbose=True, extra_parameters=None, des_enabled=False, des_config=None):
        super(multifaultsolve_boundLSE, self).__init__(name,
                                                faults,
                                                verbose=verbose)
        
        # Calculate the covariance matrix and the inverse of the covariance matrix
        self.calculate_Icovd_chol()
        self.calculate_slip_and_poly_positions()

        # Configure DES Logger
        from .des_utils import setup_des_logging
        import logging
        
        if verbose:
            # Shows High-level info (Alpha, Mode, Ranges)
            setup_des_logging(logging.INFO)
            # OR for deep debugging:
            # setup_des_logging(logging.DEBUG)
        else:
            setup_des_logging(logging.WARNING)

        # Initialize storage for bounds and constraints
        self._lb = None
        self._ub = None
        self._A_ueq = None
        self._b_ueq = None
        self._Aeq = None
        self._beq = None

        # Initialize unified constraint manager
        self.constraint_manager = ConstraintManager(solver=self, verbose=verbose)
        
        # Initialize parameter count
        self.lsq_parameters = self._calculate_total_parameters()
        
        if extra_parameters is not None:
            self.ramp_switch = len(extra_parameters)
        else:
            self.ramp_switch = 0
        
        # DES (Depth-Equalized Smoothing) parameters
        self.des_enabled = des_enabled
        self.des_config = des_config if des_config is not None else {
            'mode': 'per_patch',
            'G_norm': 'l2',
            'depth_grouping': {
                'strategy': 'uniform',
                'interval': 1.0
                }
        }
        
        if verbose:
            print(f"🔧 Enhanced multifaultsolve_boundLSE initialized with unified constraint management")
        
        return
    
    def _calculate_total_parameters(self):
        """Calculate total number of parameters for all faults."""
        total = 0
        for fault in self.faults:
            npatches = len(fault.patch)
            num_slip_samples = len(fault.slipdir) * npatches
            num_poly_samples = np.sum([fault.numberofpolys[ikey] for ikey in fault.numberofpolys], dtype=int)
            total += num_slip_samples + num_poly_samples
        return total
    
    def calculate_Icovd_chol(self):
        '''
        Calculate the Cholesky decomposition of the inverse of the covariance matrix.
        '''
        Icovd = np.linalg.inv(self.Cd)
        self.Icovd_chol = np.linalg.cholesky(Icovd)
        return

    def calculate_slip_and_poly_positions(self):
        """
        Calculates indices for SS, DS, Poly, AND Depths for DES.
        """
        self.slip_positions = {}
        self.poly_positions = {}
        self.des_indices_config = [] 
        
        current_idx = 0
        
        for fault in self.faults:
            n_patches = len(fault.patch)
            
            # --- 获取深度信息 ---
            # 假设 fault.getcenters() 返回 [x, y, z] (km)，取 z 为深度
            # 注意：确保单位一致 (通常是 km 或 m，只要配置里对得上即可)
            centers = fault.getcenters()
            depths = [c[2] for c in centers] # List of depth for each patch
            
            # --- 1. SS 索引 ---
            ss_start = current_idx
            ss_end = ss_start + n_patches
            ss_indices = list(range(ss_start, ss_end))
            current_idx = ss_end
            
            # --- 2. DS 索引 ---
            ds_indices = []
            if len(fault.slipdir) > 1:
                ds_start = current_idx
                ds_end = ds_start + n_patches
                ds_indices = list(range(ds_start, ds_end))
                current_idx = ds_end
            
            # Legacy positions update
            slip_end_idx = current_idx
            self.slip_positions[fault.name] = (ss_start, slip_end_idx)
            
            # --- 3. Poly 索引 ---
            num_poly = np.sum([fault.numberofpolys[ikey] for ikey in fault.numberofpolys], dtype=int)
            if num_poly > 0:
                poly_start = current_idx
                poly_end = poly_start + num_poly
                poly_indices = list(range(poly_start, poly_end))
                current_idx = poly_end
            else:
                poly_indices = []
            
            self.poly_positions[fault.name] = (slip_end_idx, current_idx)

            # --- 4. 生成 DES 配置 ---
            fault_config = {
                'name': fault.name,
                'ss': ss_indices,
                'ds': ds_indices,
                'poly': poly_indices,
                'depths': depths  # <--- 新增：保存该断层每个 Patch 的深度
            }
            self.des_indices_config.append(fault_config)

        self.lsq_parameters = current_idx

    def set_bounds(self, lb=None, ub=None, strikeslip_limits=None, dipslip_limits=None, poly_limits=None):
        """Set bounds using constraint manager."""
        # Set global bounds
        if lb is not None or ub is not None:
            self.constraint_manager.set_global_bounds(lb, ub, source="manual")
        
        # Set fault-specific bounds
        if strikeslip_limits or dipslip_limits:
            for fault in self.faults:
                ss_bounds = strikeslip_limits.get(fault.name) if strikeslip_limits else None
                ds_bounds = dipslip_limits.get(fault.name) if dipslip_limits else None
                if ss_bounds or ds_bounds:
                    self.constraint_manager.set_fault_slip_bounds(fault.name, ss_bounds, ds_bounds, source="manual")
        
        # Set polynomial bounds
        if poly_limits:
            for fault in self.faults:
                if fault.name in poly_limits:
                    self.constraint_manager.set_fault_poly_bounds(fault.name, poly_limits[fault.name], source="manual")
        
        # Sync to solver
        self.constraint_manager.sync_to_solver()

    # Legacy properties for backward compatibility
    @property
    def lb(self):
        """Lower bounds for backward compatibility."""
        return self._lb

    @lb.setter
    def lb(self, value):
        """Set lower bounds."""
        self._lb = value

    @property 
    def ub(self):
        """Upper bounds for backward compatibility."""
        return self._ub

    @ub.setter
    def ub(self, value):
        """Set upper bounds."""
        self._ub = value

    @property
    def A_ueq(self):
        """Combined inequality constraint matrix for backward compatibility."""
        if self._A_ueq is not None:
            return self._A_ueq
        # Fallback to constraint manager
        A, _ = self.constraint_manager.get_combined_inequality_constraints()
        return A

    @A_ueq.setter
    def A_ueq(self, value):
        """Set inequality constraint matrix."""
        self._A_ueq = value

    @property
    def b_ueq(self):
        """Combined inequality constraint vector for backward compatibility."""
        if self._b_ueq is not None:
            return self._b_ueq
        # Fallback to constraint manager
        _, b = self.constraint_manager.get_combined_inequality_constraints()
        return b

    @b_ueq.setter
    def b_ueq(self, value):
        """Set inequality constraint vector."""
        self._b_ueq = value

    @property
    def Aeq(self):
        """Combined equality constraint matrix for backward compatibility."""
        if self._Aeq is not None:
            return self._Aeq
        # Fallback to constraint manager
        A, _ = self.constraint_manager.get_combined_equality_constraints()
        return A

    @Aeq.setter
    def Aeq(self, value):
        """Set equality constraint matrix."""
        self._Aeq = value

    @property
    def beq(self):
        """Combined equality constraint vector for backward compatibility."""
        if self._beq is not None:
            return self._beq
        # Fallback to constraint manager
        _, b = self.constraint_manager.get_combined_equality_constraints()
        return b

    @beq.setter
    def beq(self, value):
        """Set equality constraint vector."""
        self._beq = value

    @property
    def inequality_constraints(self):
        """Inequality constraints for backward compatibility."""
        return self.constraint_manager._inequality_constraints

    @property
    def equality_constraints(self):
        """Equality constraints for backward compatibility."""
        return self.constraint_manager._equality_constraints

    @property
    def inequality_constraints(self):
        """Inequality constraints for backward compatibility."""
        return self.constraint_manager.inequality_constraints

    @property
    def equality_constraints(self):
        """Equality constraints for backward compatibility."""
        return self.constraint_manager.equality_constraints

    def add_inequality_constraint(self, A, b, name=None, source=None):
        """Add inequality constraint using constraint manager."""
        if name is None:
            name = f"ineq_{len(self.constraint_manager._inequality_constraints) + 1}"
        self.constraint_manager.add_inequality_constraint(A, b, name, source or "external")
        # Sync back to solver
        self.constraint_manager.sync_to_solver()

    def add_equality_constraint(self, A, b, name=None, source=None):
        """Add equality constraint using constraint manager."""
        if name is None:
            name = f"eq_{len(self.constraint_manager._equality_constraints) + 1}"
        self.constraint_manager.add_equality_constraint(A, b, name, source or "external")
        # Sync back to solver
        self.constraint_manager.sync_to_solver()

    def set_bounds_from_config(self, config_file, encoding='utf-8'):
        """Load and apply bounds from config file using constraint manager."""
        self.constraint_manager.load_bounds_config(config_file, encoding)
        self.constraint_manager.apply_bounds_from_config()
        self.constraint_manager.sync_to_solver()

    def set_inequality_constraints_for_rake_angle(self, rake_limits):
        """Set rake angle constraints using constraint manager."""
        self.constraint_manager.set_rake_angle_constraints(rake_limits, source="manual")
        self.constraint_manager.sync_to_solver()
        
    def set_equality_constraints_for_fixed_rake(self, fixed_rake):
        """Set fixed rake angle constraints using constraint manager."""
        self.constraint_manager.set_fixed_rake_constraints(fixed_rake, source="manual")
        self.constraint_manager.sync_to_solver()

    def print_constraint_summary(self):
        """Print constraint summary using constraint manager."""
        self.constraint_manager.print_summary()

    @property
    def bounds(self):
        """Bounds dict for backward compatibility."""
        return {
            'lb': self.constraint_manager._bounds['global']['lb'],
            'ub': self.constraint_manager._bounds['global']['ub'], 
            'strikeslip': self.constraint_manager._bounds['strikeslip'],
            'dipslip': self.constraint_manager._bounds['dipslip'],
            'poly': self.constraint_manager._bounds['poly']
        }

    # Convenience methods (delegate to constraint manager)
    def add_parameter_equality(self, indices, name=None):
        """Convenience method for adding parameter equality constraints."""
        self.add_equality_constraint(indices=indices, name=name)

    def add_linear_combination_constraint(self, coefficients, indices, value, constraint_type='equality', name=None):
        """Convenience method for adding linear combination constraints."""
        coefficients = np.array(coefficients)
        indices = np.array(indices)
        
        if len(coefficients) != len(indices):
            raise ValueError("Coefficients and indices arrays must have the same length")
        
        A = np.zeros((1, self.lsq_parameters))
        A[0, indices] = coefficients
        b = np.array([value])
        
        if constraint_type == 'equality':
            self.add_equality_constraint(A=A, b=b, name=name)
        elif constraint_type == 'inequality':
            self.add_inequality_constraint(A, b, name=name)
        else:
            raise ValueError("constraint_type must be 'equality' or 'inequality'")

    def ConstrainedLeastSquareSoln(self, penalty_weight=1., smoothing_matrix=None, data_weight=1.,
                                smoothing_constraints=None, method='mudpy', Aueq=None, bueq=None, 
                                Aeq=None, beq=None, verbose=False, extra_parameters=None,
                                iterations=1000, tolerance=None, maxfun=100000, des_enabled=None,
                                validate_constraints=True):
        '''
        Enhanced constrained least squares solution with unified constraint management.
        '''

        # Validate constraints using constraint manager
        if validate_constraints:
            validation = self.constraint_manager.validate_constraints()
            if not validation['valid']:
                print("❌ Constraint validation failed:")
                for error in validation['errors']:
                    print(f"   Error: {error}")
                raise ValueError("Invalid constraints detected")
            
            if validation['warnings'] and verbose:
                print("⚠️  Constraint validation warnings:")
                for warning in validation['warnings']:
                    print(f"   Warning: {warning}")

        if verbose:
            print("\n🚀 Starting constrained least squares with unified constraint management:")
            if not self.constraint_manager.verbose:
                self.constraint_manager.print_summary()

        # Ensure constraints are synced
        self.constraint_manager.sync_to_solver()

        # Import DES utilities
        from .des_utils import apply_des_transformation, recover_sf_with_poly

        # Determine if DES should be used
        use_des = des_enabled if des_enabled is not None else self.des_enabled

        # Get the faults
        faults = self.faults

        # Get the matrixes and vectors
        G = self.G
        Cd = self.Cd
        d = self.d

        # Nd = d.shape[0]
        Np = G.shape[1]
        Ns = 0
        Ns_st = []
        Ns_se = []
        # build Laplace
        for fault in faults:
            Ns_st.append(Ns)
            Ns += int(fault.slip.shape[0]*len(fault.slipdir))
            Ns_se.append(Ns)
        G_lap = np.zeros((Ns, Np))
        d_lap = np.zeros((Ns, ))

        # ----------------------------Smoothing matrix-----------------------------#
        if smoothing_matrix is None:
            if isinstance(penalty_weight, (int, float)):
                penalty_weight = np.ones(len(faults)) * penalty_weight
            elif isinstance(penalty_weight, (list, np.ndarray)):
                assert len(penalty_weight) == len(faults), "The length of penalty_weight should be equal to the number of faults."
            else:
                raise ValueError("penalty_weight should be a scalar or a list of scalars.")

            # Handle smoothing constraints
            faultnames = [ifault.name for ifault in faults]
            if smoothing_constraints is not None:
                if isinstance(smoothing_constraints, tuple) and len(smoothing_constraints) == 4:
                    smoothing_constraints = {fault_name: smoothing_constraints for fault_name in faultnames}
                elif isinstance(smoothing_constraints, dict):
                    assert all(fault_name in smoothing_constraints for fault_name in faultnames), "All fault names must be in smoothing_constraints."
                else:
                    raise ValueError("smoothing_constraints should be a 4-tuple or a dictionary with fault names as keys and 4-tuples as values.")

            smoothing_constraints = [smoothing_constraints[ifaultname] for ifaultname in faultnames]
            for ii, (fault, ipenalty_weight, ismoothing_constraints) in enumerate(zip(faults, penalty_weight, smoothing_constraints)):
                st = self.fault_indexes[fault.name][0]
                if fault.type == 'Fault':
                    if fault.patchType in ('rectangle'):
                        lap = fault.buildLaplacian(method=method, bounds=ismoothing_constraints)
                    else:
                        lap = fault.buildLaplacian(method=method, bounds=ismoothing_constraints)
                    lap_sd = blkdiag(lap, lap)
                    Nsd = len(fault.slipdir)
                    # TODO: The following code is not clear, need to be modified
                    if Nsd == 1:
                        lap_sd = lap
                    se = st + Nsd*lap.shape[0]
                    G_lap[Ns_st[ii]:Ns_se[ii], st:se] = lap_sd * ipenalty_weight
        else:
            G_lap = smoothing_matrix
            d_lap = np.zeros((G_lap.shape[0], ))
        self.G_lap = G_lap

        # ----------------------------Data weight-----------------------------#
        if isinstance(data_weight, (int, float)):
            data_weight = np.ones(d.shape[0]) * data_weight
        elif isinstance(data_weight, (list, np.ndarray)):
            assert len(data_weight) == len(self.faults[0].datanames), "The length of data_weight should be equal to the number of data sets."
            data_weight = np.array(data_weight)
        else:
            raise ValueError("data_weight should be a scalar or a list of scalars.")
        
        Icovd_chol = self.Icovd_chol.copy()
        st = 0
        ed = 0
        datanames = self.faults[0].datanames
        for idataname, iwgt in zip(datanames, data_weight):
            idata = self.faults[0].d[idataname]
            ed = st + idata.shape[0]
            Icovd_chol[st:ed, st:ed] *= iwgt
            st = ed

        W = Icovd_chol
        self.dataweight = W

        # ----------------------------Set constraints using constraint manager-----------------------------#
        # Get constraints from constraint manager
        A_ueq_cm, b_ueq_cm = self.constraint_manager.get_combined_inequality_constraints()
        Aeq_cm, beq_cm = self.constraint_manager.get_combined_equality_constraints()
        
        # Combine with external constraints if provided
        A_ueq = A_ueq_cm
        b_ueq = b_ueq_cm
        if Aueq is not None and bueq is not None:
            A_ueq = np.vstack((A_ueq, Aueq)) if A_ueq is not None else Aueq
            b_ueq = np.hstack((b_ueq, bueq)) if b_ueq is not None else bueq

        Aeq_final = Aeq_cm
        beq_final = beq_cm
        if Aeq is not None and beq is not None:
            Aeq_final = np.vstack((Aeq_final, Aeq)) if Aeq_final is not None else Aeq
            beq_final = np.hstack((beq_final, beq)) if beq_final is not None else beq

        # Get bounds from constraint manager
        lb = self.constraint_manager.lb
        ub = self.constraint_manager.ub
        if lb is None or ub is None or any(np.isnan(lb)) or any(np.isnan(ub)):
            raise ValueError("You should set bounds first using set_bounds() or constraint manager methods")

        # ----------------------------Apply DES transformation if enabled-----------------------------#
        if use_des:
            if verbose:
                print("Applying Depth-Equalized Smoothing (DES) transformation...")
            
            # Get polynomial positions
            # poly_positions = get_poly_positions_from_multifaults(self)
            
            # Apply DES transformation to the original Green's function matrix G
            des_result = apply_des_transformation(
                G=G,  # Use original G matrix for DES parameter calculation
                D=G_lap,
                A_ineq=A_ueq,
                b_ineq=b_ueq,
                A_eq=Aeq_final,
                b_eq=beq_final,
                lb=lb,
                ub=ub,
                fault_indices_config=self.des_indices_config,
                mode=self.des_config.get('mode', 'per_column'),
                # groups=self.des_config.get('groups', None),
                G_norm=self.des_config.get('G_norm', 'l2'),
                depth_grouping_config=self.des_config.get('depth_grouping', None)
            )
            
            # Apply DES scaling to get G_prime
            G_prime = des_result['G_prime']
            
            # Now construct the augmented system with DES-scaled matrices
            d2I = np.vstack((np.dot(W, d)[:, None], d_lap[:, None])).flatten()
            G2I = np.vstack((np.dot(W, G_prime), des_result['D_prime']))
            
            # Update constraints with DES-transformed versions
            A_ueq_prime = des_result.get('A_ineq_prime', A_ueq)
            b_ueq_prime = des_result.get('b_ineq', b_ueq)
            Aeq_prime = des_result.get('A_eq_prime', Aeq_final)
            beq_prime = des_result.get('b_eq', beq_final)
            lb_prime = des_result['lb_prime']
            ub_prime = des_result['ub_prime']
            
            # Store DES information for recovery
            self.des_result = des_result
            
            if verbose:
                print(f"DES applied: {len(des_result['fault_indices'])} fault parameters scaled")
                print(f"Scaling factor range: [{des_result['scale_factors'].min():.3f}, {des_result['scale_factors'].max():.3f}]")
        else:
            # No DES transformation - use original matrices
            d2I = np.vstack((np.dot(W, d)[:, None], d_lap[:, None])).flatten()
            G2I = np.vstack((np.dot(W, G), G_lap))
            
            A_ueq_prime, b_ueq_prime = A_ueq, b_ueq
            Aeq_prime, beq_prime = Aeq_final, beq_final
            lb_prime, ub_prime = lb, ub

        # ----------------------------Inverse using lsqlin-----------------------------#
        # Compute using lsqlin
        opts = {'show_progress': False}
        try:
            ret = lsqlin.lsqlin(G2I, d2I, 0, A_ueq_prime, b_ueq_prime, Aeq_prime, beq_prime, lb_prime, ub_prime, None, opts)
        except:
            ret = lsqlin.lsqlin(G2I, d2I, 0, A_ueq_prime, b_ueq_prime, None, None, lb_prime, ub_prime, None, opts)
        mpost_prime = lsqlin.cvxopt_to_numpy_matrix(ret['x'])
        
        # ----------------------------Recover solution if DES was used-----------------------------#
        if use_des:
            # Recover the final solution
            mpost = recover_sf_with_poly(
                mpost_prime, 
                des_result['alpha'], 
                des_result['norm2_fault'], 
                des_result['fault_indices']
            )
        else:
            mpost = mpost_prime
        
        # Store mpost
        self.mpost = mpost

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def simple_vce(self, smoothing_matrix=None, smoothing_constraints=None, method='mudpy',
                   verbose=False, max_iter=10, tol=1e-4, des_enabled=None,
                   sigma_mode='individual', sigma_groups=None, sigma_update=None, sigma_values=None,
                   smooth_mode='single', smooth_groups=None, smooth_update=None, smooth_values=None,
                   validate_constraints=True):
        """
        Perform Simple Variance Component Estimation (VCE) for multi-fault inversion.

        This method iteratively estimates optimal weights between data fitting and
        regularization components using a simplified VCE approach with lsqlin solver.
        The penalty weights are automatically determined through VCE iterations.

        Parameters
        ----------
        smoothing_matrix : array, optional
            Pre-computed smoothing matrix (if None, will build Laplacian)
        smoothing_constraints : tuple or dict, optional
            Smoothing constraints for Laplacian construction
        method : str
            Method for building Laplacian ('mudpy')
        verbose : bool
            Enable verbose output
        max_iter : int
            Maximum VCE iterations
        tol : float
            Convergence tolerance
        des_enabled : bool, optional
            Whether to use DES (if None, uses self.des_enabled)
        sigma_mode : str
            'single', 'individual', or 'grouped' for data variance components
        sigma_groups : dict, optional
            Custom grouping for data variance components
        sigma_update : list of bool, optional
            Whether to update each sigma group (same order as sigma groups)
        sigma_values : list of float, optional
            Initial/fixed values for each sigma group (same order as sigma groups)
        smooth_mode : str
            'single', 'individual', or 'grouped' for smoothing variance components
        smooth_groups : dict, optional
            Custom grouping for smoothing variance components
        smooth_update : list of bool, optional
            Whether to update each smoothing group (same order as smoothing groups)
        smooth_values : list of float, optional
            Initial/fixed values for each smoothing group (same order as smoothing groups)

        Returns
        -------
        dict with keys:
            - 'm': estimated parameters
            - 'var_d': data variance components
            - 'var_alpha': regularization variance components
            - 'weights': final weight ratios
            - 'converged': convergence flag
            - 'iterations': number of iterations
        """
        # Validate constraints using constraint manager
        if validate_constraints:
            validation = self.constraint_manager.validate_constraints()
            if not validation['valid']:
                print("❌ Constraint validation failed:")
                for error in validation['errors']:
                    print(f"   Error: {error}")
                raise ValueError("Invalid constraints detected")

        if verbose:
            print("\n🚀 Starting VCE with unified constraint management:")
            self.constraint_manager.print_summary()

        # Ensure constraints are synced
        self.constraint_manager.sync_to_solver()

        from .simple_vce import simplified_vce
        from .des_utils import apply_des_transformation, recover_sf_with_poly

        use_des = des_enabled if des_enabled is not None else self.des_enabled

        if verbose:
            print("="*60)
            print("Starting Simple VCE for Multi-Fault Inversion")
            print(f"Number of faults: {len(self.faults)}")
            print(f"DES enabled: {use_des}")
            print(f"Sigma mode: {sigma_mode}")
            print(f"Smooth mode: {smooth_mode}")
            print("="*60)

        # Get basic matrices
        G = self.G
        d = self.d
        Cd_inv = np.linalg.inv(self.Cd)
        # Icovd_chol = np.linalg.cholesky(Cd_inv)

        # Set bounds
        lb = self.constraint_manager.lb
        ub = self.constraint_manager.ub
        if lb is None or ub is None or any(np.isnan(lb)) or any(np.isnan(ub)):
            raise ValueError("You should set bounds first using set_bounds() method")

        # Setup data ranges
        data_ranges = {}
        start = 0
        for dataname in self.faults[0].datanames:
            idata = self.faults[0].d[dataname]
            end = start + idata.shape[0]
            data_ranges[dataname] = (start, end)
            start = end

        # Setup fault ranges
        fault_ranges = {}
        for fault in self.faults:
            start, end = self.slip_positions[fault.name]
            fault_ranges[fault.name] = (start, end)

        # Build smoothing matrix if not provided
        if smoothing_matrix is None:
            if verbose:
                print("Building smoothing matrix...")

            faults = self.faults
            Np = G.shape[1]
            Ns = 0
            Ns_st = []
            Ns_se = []

            for fault in faults:
                Ns_st.append(Ns)
                Ns += int(fault.slip.shape[0] * len(fault.slipdir))
                Ns_se.append(Ns)

            G_lap = np.zeros((Ns, Np))

            faultnames = [ifault.name for ifault in faults]
            if smoothing_constraints is not None:
                if isinstance(smoothing_constraints, (tuple, list)) and len(smoothing_constraints) == 4:
                    smoothing_constraints = {fault_name: smoothing_constraints for fault_name in faultnames}
                elif isinstance(smoothing_constraints, dict):
                    assert all(fault_name in smoothing_constraints for fault_name in faultnames), \
                        "All fault names must be in smoothing_constraints"
            else:
                smoothing_constraints = {fault_name: (None, None, None, None) for fault_name in faultnames}

            for ii, fault in enumerate(faults):
                st = self.fault_indexes[fault.name][0]
                ismoothing_constraints = smoothing_constraints[fault.name]

                if fault.type == 'Fault':
                    lap = fault.buildLaplacian(method=method, bounds=ismoothing_constraints)
                    from scipy.linalg import block_diag as blkdiag
                    lap_sd = blkdiag(lap, lap)
                    Nsd = len(fault.slipdir)
                    if Nsd == 1:
                        lap_sd = lap
                    se = st + Nsd * lap.shape[0]
                    G_lap[Ns_st[ii]:Ns_se[ii], st:se] = lap_sd

            smoothing_matrix = G_lap

            if verbose:
                print(f"Smoothing matrix built: {smoothing_matrix.shape}")

        # Get constraints from constraint manager
        A_ueq, b_ueq = self.constraint_manager.get_combined_inequality_constraints()
        Aeq, beq = self.constraint_manager.get_combined_equality_constraints()

        # Prepare for DES transformation if enabled
        if use_des:
            des_result = apply_des_transformation(
                G=G,  # Use original G matrix for DES parameter calculation
                D=smoothing_matrix,
                A_ineq=A_ueq,
                b_ineq=b_ueq,
                A_eq=Aeq,
                b_eq=beq,
                lb=lb,
                ub=ub,
                fault_indices_config=self.des_indices_config,
                mode=self.des_config.get('mode', 'per_column'),
                G_norm=self.des_config.get('G_norm', 'l2'),
                depth_grouping_config=self.des_config.get('depth_grouping', None)
            )

            G_vce = des_result['G_prime']
            L_vce = des_result['D_prime']
            lb_vce = des_result['lb_prime']
            ub_vce = des_result['ub_prime']
            A_ueq_vce = des_result.get('A_ineq_prime', A_ueq)
            b_ueq_vce = des_result.get('b_ineq', b_ueq)
            Aeq_vce = des_result.get('A_eq_prime', Aeq)
            beq_vce = des_result.get('b_eq', beq)
            fault_ranges_vce = fault_ranges  # Keep original for now

            self.des_result = des_result
        else:
            G_vce = G
            L_vce = smoothing_matrix
            lb_vce = lb
            ub_vce = ub
            A_ueq_vce = A_ueq
            b_ueq_vce = b_ueq
            Aeq_vce = Aeq
            beq_vce = beq
            fault_ranges_vce = fault_ranges

        # Run Simple VCE with lsqlin solver
        if verbose:
            print(f"Running Simple VCE with lsqlin solver (max_iter={max_iter}, tol={tol})...")

        vce_result = simplified_vce(
            Cd_inv=Cd_inv,
            d=d,
            G=G_vce,
            L=L_vce,
            bounds=(lb_vce, ub_vce),
            data_ranges=data_ranges,
            fault_ranges=fault_ranges_vce,
            sigma_mode=sigma_mode,
            sigma_groups=sigma_groups,
            sigma_update=sigma_update,
            sigma_values=sigma_values,
            smooth_mode=smooth_mode,
            smooth_groups=smooth_groups,
            smooth_update=smooth_update,
            smooth_values=smooth_values,
            A_ueq=A_ueq_vce,
            b_ueq=b_ueq_vce,
            Aeq=Aeq_vce,
            beq=beq_vce,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose
        )

        # Recover solution if DES was used
        if use_des:
            m_prime = vce_result['m']
            m_recovered = recover_sf_with_poly(
                m_prime,
                des_result['alpha'],
                des_result['norm2_fault'],
                des_result['fault_indices']
            )

            vce_result['m'] = m_recovered

        # Store results
        self.mpost = vce_result['m']
        self.vce_result = vce_result

        if verbose:
            print(f"VCE completed in {vce_result['iterations']} iterations")
            print(f"Converged: {vce_result['converged']}")

            if isinstance(vce_result['var_d'], dict):
                for group, var in vce_result['var_d'].items():
                    print(f"Data variance [{group}]: {var:.6f}")
            else:
                print(f"Data variance: {vce_result['var_d']:.6f}")

            if isinstance(vce_result['var_alpha'], dict):
                for group, var in vce_result['var_alpha'].items():
                    print(f"Regularization variance [{group}]: {var:.6f}")
            else:
                print(f"Regularization variance: {vce_result['var_alpha']:.6f}")

            if 'weights' in vce_result:
                print(f"\nFinal weights:")
                weights = vce_result['weights']
                if isinstance(weights, dict):
                    if any(isinstance(v, dict) for v in weights.values()):
                        for d_group, w_dict in weights.items():
                            for alpha_group, weight in w_dict.items():
                                print(f"  weight[{d_group}][{alpha_group}]: {weight:.6f}")
                    else:
                        for group, weight in weights.items():
                            print(f"  weight[{group}]: {weight:.6f}")
                else:
                    print(f"  weight: {weights:.6f}")

            print("="*60)

        return vce_result
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def solve_with_fnnls(self, penalty_weight=1., smoothing_matrix=None, data_weight=1.,
                         smoothing_constraints=None, method='mudpy', verbose=False,
                         validate_constraints=True):
        """
        Solve the constrained least squares problem using fnnls.
    
        Parameters:
        - penalty_weight: Weight for the smoothing matrix penalty.
        - smoothing_matrix: Matrix used for smoothing (Laplacian if provided).
        - data_weight: Weight for the data.
        - smoothing_constraints: Smoothing constraints to apply during the least squares process.
        - method: Solver method to use.
        - verbose: Enable verbose output.
    
        Returns:
        - mpost: The solution vector.
        """
        # Validate constraints
        if validate_constraints:
            validation = self.constraint_manager.validate_constraints()
            if not validation['valid']:
                print("❌ Constraint validation failed:")
                for error in validation['errors']:
                    print(f"   Error: {error}")
                raise ValueError("Invalid constraints detected")

        if verbose:
            print("\n🚀 Starting fnnls solver with unified constraint management:")
            self.constraint_manager.print_summary()

        # Ensure constraints are synced
        self.constraint_manager.sync_to_solver()

        # Get the faults
        faults = self.faults
    
        # Get the matrixes and vectors
        G = self.G
        Cd = self.Cd
        d = self.d
    
        # Nd = d.shape[0]
        Np = G.shape[1]
        Ns = 0
        Ns_st = []
        Ns_se = []
        # build Laplace
        for fault in faults:
            Ns_st.append(Ns)
            Ns += int(fault.slip.shape[0] * len(fault.slipdir))
            Ns_se.append(Ns)
        G_lap = np.zeros((Ns, Np))
        d_lap = np.zeros((Ns, ))
    
        # ----------------------------Smoothing matrix-----------------------------#
        if smoothing_matrix is None:
            if isinstance(penalty_weight, (int, float)):
                penalty_weight = np.ones(len(faults)) * penalty_weight
            elif isinstance(penalty_weight, (list, np.ndarray)):
                assert len(penalty_weight) == len(faults), "The length of penalty_weight should be equal to the number of faults."
            else:
                raise ValueError("penalty_weight should be a scalar or a list of scalars.")
    
            # Handle smoothing constraints
            faultnames = [ifault.name for ifault in faults]
            if smoothing_constraints is not None:
                if isinstance(smoothing_constraints, tuple) and len(smoothing_constraints) == 4:
                    smoothing_constraints = {fault_name: smoothing_constraints for fault_name in faultnames}
                elif isinstance(smoothing_constraints, dict):
                    assert all(fault_name in smoothing_constraints for fault_name in faultnames), "All fault names must be in smoothing_constraints."
                else:
                    raise ValueError("smoothing_constraints should be a 4-tuple or a dictionary with fault names as keys and 4-tuples as values.")
    
            smoothing_constraints = [smoothing_constraints[ifaultname] for ifaultname in faultnames]
            for ii, (fault, ipenalty_weight, ismoothing_constraints) in enumerate(zip(faults, penalty_weight, smoothing_constraints)):
                st = self.fault_indexes[fault.name][0]
                if fault.type == 'Fault':
                    if fault.patchType in ('rectangle'):
                        lap = fault.buildLaplacian(method=method, bounds=ismoothing_constraints)
                    else:
                        lap = fault.buildLaplacian(method=method, bounds=ismoothing_constraints)
                    lap_sd = blkdiag(lap, lap)
                    Nsd = len(fault.slipdir)
                    # TODO: The following code is not clear, need to be modified
                    if Nsd == 1:
                        lap_sd = lap
                    se = st + Nsd * lap.shape[0]
                    G_lap[Ns_st[ii]:Ns_se[ii], st:se] = lap_sd * ipenalty_weight
        else:
            G_lap = np.zeros((smoothing_matrix.shape[0], Np))
            G_lap[:, :Ns] = smoothing_matrix
            d_lap = np.zeros((G_lap.shape[0], ))
        self.G_lap = G_lap
    
        G_lap2I = G_lap
    
        # ----------------------------Data weight-----------------------------#
        if isinstance(data_weight, (int, float)):
            data_weight = np.ones(d.shape[0]) * data_weight
        elif isinstance(data_weight, (list, np.ndarray)):
            assert len(data_weight) == len(self.faults[0].datanames), "The length of data_weight should be equal to the number of data sets."
            data_weight = np.array(data_weight)
        else:
            raise ValueError("data_weight should be a scalar or a list of scalars.")
        # Icovd = np.linalg.inv(Cd)
        # Icovd_chol = np.linalg.cholesky(Icovd)
        Icovd_chol = self.Icovd_chol
        st = 0
        ed = 0
        datanames = self.faults[0].datanames
        for idataname, iwgt in zip(datanames, data_weight):
            idata = self.faults[0].d[idataname]
            ed = st + idata.shape[0]
            Icovd_chol[st:ed, st:ed] *= iwgt
            st = ed
    
        W = Icovd_chol
        self.dataweight = W
        d2I = np.vstack((np.dot(W, d)[:, None], d_lap[:, None])).flatten()
    
        G2I = np.vstack((np.dot(W, G), G_lap2I))
    
        # ----------------------------Inverse using fnnls-----------------------------#
        # Get bounds from constraint manager
        lb = self.constraint_manager.lb
        ub = self.constraint_manager.ub
        if lb is None or ub is None or any(np.isnan(lb)) or any(np.isnan(ub)):
            raise ValueError("You should set bounds first using set_bounds() method")
        
        # Ensure lb and ub are numpy arrays
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        
        # Initialize masks for variables that need to be flipped
        flip_mask = (lb < 0) | (ub < 0)
        
        # Flip the necessary parts of G2I
        G2I_flipped = G2I.copy()
        G2I_flipped[:, flip_mask] = -G2I[:, flip_mask]
        
        # Use fnnls to solve the problem
        mpost_flipped, res = fnnls(G2I_flipped, d2I)
        
        # Flip the necessary parts of the solution back
        mpost = mpost_flipped.copy()
        mpost[flip_mask] = -mpost_flipped[flip_mask]
        
        # Store mpost
        self.mpost = mpost
        
        # All done
        return

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def distributem(self, verbose=False):
        '''
        After computing the m_post model, this routine distributes the m parameters to the faults.

        Kwargs:
            * verbose   : talk to me

        Returns:
            * None
        '''

        # Get the faults
        faults = self.faults

        # Loop over the faults
        for fault in faults:

            if verbose:
                print ("---------------------------------")
                print ("---------------------------------")
                print("Distribute the slip values to fault {}".format(fault.name))

            # Store the mpost
            st = self.fault_indexes[fault.name][0]
            se = self.fault_indexes[fault.name][1]
            fault.mpost = self.mpost[st:se]

            # Transformation object
            if fault.type=='transformation':
                
                # Distribute simply
                fault.distributem()

            # Fault object
            if fault.type == "Fault":

                # Affect the indexes
                self.affectIndexParameters(fault)

                # put the slip values in slip
                st = 0
                if 's' in fault.slipdir:
                    se = st + fault.slip.shape[0]
                    fault.slip[:,0] = fault.mpost[st:se]
                    st += fault.slip.shape[0]
                if 'd' in fault.slipdir:
                    se = st + fault.slip.shape[0]
                    fault.slip[:,1] = fault.mpost[st:se]
                    st += fault.slip.shape[0]
                if 't' in fault.slipdir:
                    se = st + fault.slip.shape[0]
                    fault.slip[:,2] = fault.mpost[st:se]
                    st += fault.slip.shape[0]
                if 'c' in fault.slipdir:
                    se = st + fault.slip.shape[0]
                    fault.coupling = fault.mpost[st:se]
                    st += fault.slip.shape[0]

                # check
                if hasattr(fault, 'NumberCustom'):
                    fault.custom = {} # Initialize dictionnary
                    # Get custom params for each dataset
                    for dset in fault.datanames:
                        if 'custom' in fault.G[dset].keys():
                            nc = fault.G[dset]['custom'].shape[1] # Get number of param for this dset
                            se = st + nc
                            fault.custom[dset] = fault.mpost[st:se]
                            st += nc

            # Pressure object
            elif fault.type == "Pressure":

                st = 0
                if fault.source in {"Mogi", "Yang"}:
                    se = st + 1
                    print(np.asscalar(fault.mpost[st:se]*fault.mu))
                    fault.deltapressure = np.asscalar(fault.mpost[st:se]*fault.mu)
                    st += 1
                elif fault.source == "pCDM":
                    se = st + 1
                    fault.DVx = np.asscalar(fault.mpost[st:se]*fault.scale)
                    st += 1
                    se = st + 1
                    fault.DVy = np.asscalar(fault.mpost[st:se]*fault.scale)
                    st += 1
                    se = st + 1
                    fault.DVz = np.asscalar(fault.mpost[st:se]*fault.scale)
                    st += 1
                    print("Total potency scaled by", fault.scale)

                    if fault.DVtot is None:
                        fault.computeTotalpotency()
                elif fault.source == "CDM":
                    se = st + 1
                    print(np.asscalar(fault.mpost[st:se]*fault.mu))
                    fault.deltaopening = np.asscalar(fault.mpost[st:se])
                    st += 1

            # Get the polynomial/orbital/helmert values if they exist
            if fault.type in ('Fault', 'Pressure'):
                fault.polysol = {}
                fault.polysolindex = {}
                for dset in fault.datanames:
                    if dset in fault.poly.keys():
                        if (fault.poly[dset] is None):
                            fault.polysol[dset] = None
                        else:

                            if (fault.poly[dset].__class__ is not str) and (fault.poly[dset].__class__ is not list):
                                if (fault.poly[dset] > 0):
                                    se = st + fault.poly[dset]
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += fault.poly[dset]
                            elif (fault.poly[dset].__class__ is str):
                                if fault.poly[dset] == 'full':
                                    nh = fault.helmert[dset]
                                    se = st + nh
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += nh
                                if fault.poly[dset] in ('strain', 'strainnorotation', 'strainonly', 'strainnotranslation', 'translation', 'translationrotation'):
                                    nh = fault.strain[dset]
                                    se = st + nh
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += nh
                                # Added by kfhe, at 10/12/2021
                                if fault.poly[dset] == 'eulerrotation':
                                    nh = fault.eulerrot[dset]
                                    se = st + nh
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += nh
                                if fault.poly[dset] == 'internalstrain':
                                    nh = fault.intstrain[dset]
                                    se = st + nh
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += nh
                            elif (fault.poly[dset].__class__ is list):
                                nh = fault.transformation[dset]
                                se = st + nh
                                fault.polysol[dset] = fault.mpost[st:se]
                                fault.polysolindex[dset] = range(st,se)
                                st += nh

        # All done
        return
    # ----------------------------------------------------------------------

if __name__ == "__main__":
    solver = multifaultsolve_boundLSE()
    # 设置边界约束
    solver.set_bounds(lb=-10, ub=10)
    solver.set_bounds(strikeslip_limits={'main_fault': (-5, 5)})
    
    # 运行VCE - 每个数据集和断层都有独立的方差分量
    result = solver.simple_vce(
        sigma_mode='individual',    # 每个数据集独立sigma
        smooth_mode='individual',   # 每个断层独立alpha
        verbose=True
    )
    
    # 或者使用分组模式
    result = solver.simple_vce(
        sigma_mode='grouped',
        sigma_groups={'sar': ['insar1', 'insar2'], 'gnss': ['gps']},
        smooth_mode='grouped', 
        smooth_groups={'main': ['main_fault'], 'secondary': ['branch_fault', 'background']},
        verbose=True
    )
    
    # 分发结果
    solver.distributem()

# EOF