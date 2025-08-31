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

# Plot
from eqtools.getcpt import get_cpt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import cmcrameri # cmc.devon_r cmc.lajolla_r cmc.batlow

class multifaultsolve_boundLSE(multifaultsolve, FaultAnalysisMixin):
    '''
    Invert for slip distribution and orbital parameters
        1. Add Laplace smoothing constraints
        2. Construct a new function to generate boundary constraints
        3. Write a function to assemble the smoothing matrix and corresponding data objects
        4. Add a constrained least squares inversion function
    '''
    
    def __init__(self, name, faults, verbose=True, extra_parameters=None, des_enabled=False, des_config=None):
        super(multifaultsolve_boundLSE, self).__init__(name,
                                                faults,
                                                verbose=verbose)
        
        # Calculate the covariance matrix and the inverse of the covariance matrix
        self.calculate_Icovd_chol()
        self.calculate_slip_and_poly_positions()
        self.lb = np.ones(self.lsq_parameters) * np.nan
        self.ub = np.ones(self.lsq_parameters) * np.nan

        # Set the un-equality constraints for rake angle, with the form of A*x <= b
        self.A_ueq = None
        self.b_ueq = None

        # Set the equality constraints for fixed rake angle, with the form of Aeq*x = beq
        self.Aeq = None
        self.beq = None

        self.bounds = {
            'lb': None,
            'ub': None,
            'strikeslip': None,
            'dipslip': None,
            'poly': None,
            'rake_angle': None
        }

        if extra_parameters is not None:
            self.ramp_switch = len(extra_parameters)
        else:
            self.ramp_switch = 0
        
        # DES (Depth-Equalized Smoothing) parameters
        self.des_enabled = des_enabled
        depth_lists = [np.array(fault.getcenters(), dtype=float)[:, 2] for fault in self.faults]
        # Duplicate each fault's depth array once for strike-slip and dip-slip components estimation
        self.depths = np.unique(np.concatenate([np.tile(depth_list, 2) for depth_list in depth_lists]))
        self.des_config = des_config if des_config is not None else {
            'mode': 'per_column',
            'G_norm': 'l2',
            'depth_grouping_config': {
                'strategy': 'uniform',
                'depths': self.depths,
                'interval': 1.0
                }
        }
        
        return
    
    def calculate_Icovd_chol(self):
        '''
        Calculate the Cholesky decomposition of the inverse of the covariance matrix.
        '''
        Icovd = np.linalg.inv(self.Cd)
        self.Icovd_chol = np.linalg.cholesky(Icovd)
        return

    def calculate_slip_and_poly_positions(self):
        self.slip_positions = {}
        self.poly_positions = {}
        start_position = 0
        for fault in self.faults:
            # npatches = fault.Faces.shape[0]
            npatches = len(fault.patch)
            num_slip_samples = len(fault.slipdir)*npatches
            num_poly_samples = np.sum([npoly for npoly in fault.poly.values() if npoly is not None], dtype=int)
            self.slip_positions[fault.name] = (start_position, start_position + num_slip_samples)
            self.poly_positions[fault.name] = (start_position + num_slip_samples, start_position + num_slip_samples + num_poly_samples)
            start_position += num_slip_samples + num_poly_samples
        self.lsq_parameters = start_position
    
    def set_inequality_constraints_for_rake_angle(self, rake_limits):
        '''
        Generate linear constraints (A, b) for the rake angle range based on strike-slip and dip-slip components.
        Satify the following constraints:
         A * x <= b, where b = 0 vector
        rake_limits: (dict) {fault_name: (lower_bound, upper_bound)}, unit: degree
        rake: (-180, 180) or (0, 360), anti-clockwise is positive
        rake_ub - rake_lb <= 180
        '''

        npatch = 0
        Nsd = 0
        Np = self.lsq_parameters # self.G.shape[1]
        for ifault in self.faults:
            inpatch = len(ifault.patch)
            npatch += inpatch
            Nsd += int(inpatch*len(ifault.slipdir))
        A = np.zeros((Nsd, Np))
        b = np.zeros((Nsd,))

        patch_count = 0
        for ifault in self.faults:
            inpatch = len(ifault.patch)
            start = self.fault_indexes[ifault.name][0]
            half = start + inpatch
            # Get the rake angle bounds
            rake_start, rake_end = rake_limits[ifault.name]
            # Generate the linear constraints
            for i in range(inpatch):
                # cross product of slip and rake,
                # x: (ss, ds), y: (cos, sin) of rake with x.cross(y) = z, z = (ss*sin(rake) - ds*cos(rake)) < 0
                A[patch_count + i, start+i] = np.sin(np.deg2rad(rake_start))
                A[patch_count + i, half+i] = -np.cos(np.deg2rad(rake_start))
                # x: (ss, ds), y: (cos, sin) of rake with x.cross(y) = z, z = (ss*sin(rake) - ds*cos(rake)) > 0
                A[patch_count + inpatch+i, start+i] = -np.sin(np.deg2rad(rake_end))
                A[patch_count + inpatch+i, half+i] =   np.cos(np.deg2rad(rake_end))
            patch_count += inpatch

        self.A_ueq = A
        self.b_ueq = b
        return
    
    def set_equality_constraints_for_fixed_rake(self, fixed_rake):
        '''
        Generate equality constraints (Aeq, beq) for the fixed rake angle based on strike-slip and dip-slip components.
        Satify the following constraints:
         Aeq * x = beq, where beq = 0 vector
        fixed_rake: (dict) {fault_name: rake_angle}, unit: degree
        rake: -180 <= rake <= 180, anti-clockwise is positive
        '''
        npatch = 0
        Nsd = 0
        Np = self.lsq_parameters
        for fault in self.faults:
            npatch += fault.slip.shape[0]
            Nsd += int(fault.slip.shape[0]*len(fault.slipdir))
        Aeq = np.zeros((npatch, Np))
        beq = np.zeros((npatch,))
        patch_count = 0
        for fault in self.faults:
            irake = fixed_rake[fault.name]
            inpatch = fault.slip.shape[0]
            start = self.fault_indexes[fault.name][0]
            half = start + inpatch
            rake_angle = np.deg2rad(irake)
            # cross product
            for i in range(inpatch):
                # cross product of slip and rake,
                # x: (ss, ds), y: (cos, sin) of rake with x.cross(y) = z, z = (ss*sin(rake) - ds*cos(rake)) = 0
                Aeq[patch_count+i, start+i] = np.sin(rake_angle)
                Aeq[patch_count+i, half+i] = -np.cos(rake_angle)
            patch_count += inpatch

        self.Aeq = Aeq
        self.beq = beq
        return
    
    def set_bounds(self, lb=None, ub=None, strikeslip_limits=None, dipslip_limits=None, poly_limits=None):
        '''
        strikeslip_limits: (dict) {fault_name: (lower_bound, upper_bound)}
        dipslip_limits: (dict) {fault_name: (lower_bound, upper_bound)}
        poly_limits: (dict) {fault_name: (lower_bound, upper_bound)}
        '''
        # Set the bounds for the whole model
        self.update_global_bounds(lb, ub)
        # Set the bounds for each fault's strike-slip and dip-slip
        for fault in self.faults:
            # Strike-slip limits
            if strikeslip_limits is not None and strikeslip_limits.get(fault.name) is not None:
                self.update_slip_bounds(fault.name, strikeslip_limits[fault.name], None)
            # Dip-slip limits
            if dipslip_limits is not None and dipslip_limits.get(fault.name) is not None:
                self.update_slip_bounds(fault.name, None, dipslip_limits[fault.name])
            # Polynomial limits
            if poly_limits is not None and poly_limits.get(fault.name) is not None:
                self.update_bounds_for_poly(fault.name, poly_limits[fault.name])
        return
    
    def set_bounds_from_config(self, config_file, encoding='utf-8'):
        """
        Set parameter bounds from a configuration file.
        """
        if config_file is not None:
            try:
                with open(config_file, 'r', encoding=encoding) as file:
                    self.bounds_config = yaml.safe_load(file)
            except FileNotFoundError:
                print(f"Configuration file {config_file} not found.")
                return
            except yaml.YAMLError as e:
                print(f"Error parsing configuration file: {e}")
                return
    
        lb = self.bounds_config.get('lb', None)
        ub = self.bounds_config.get('ub', None)
        self.update_global_bounds(lb, ub)

        strikeslip = self.bounds_config.get('strikeslip', None)
        dipslip = self.bounds_config.get('dipslip', None)
        poly = self.bounds_config.get('poly', None)

        for fault in self.faults:
            fault_name = fault.name
            if strikeslip is not None or dipslip is not None:
                istrikeslip = strikeslip.get(fault_name, None) if strikeslip is not None else None
                idipslip = dipslip.get(fault_name, None) if dipslip is not None else None
                if istrikeslip is not None or idipslip is not None:
                    self.update_slip_bounds(fault_name, istrikeslip, idipslip)
        
            if poly is not None and poly.get(fault_name, None) is not None:
                self.update_bounds_for_poly(fault_name, poly[fault_name])
    
    def update_global_bounds(self, lb=None, ub=None):
        """
        Update the global bounds for the model.
        """
        if lb is not None and ub is not None and lb > ub:
            print("Global lower bound should be less than upper bound.")
            return
        self.bounds['lb'] = lb if lb is not None else self.bounds.get('lb', None)
        self.bounds['ub'] = ub if ub is not None else self.bounds.get('ub', None)
        
        if lb is not None:
            self.lb[np.isnan(self.lb)] = lb
        if ub is not None:
            self.ub[np.isnan(self.ub)] = ub
        return

    def update_slip_bounds(self, fault_name, strikeslip=None, dipslip=None):
        """
        Update the bounds for the strike-slip and dip-slip components of a fault.
        """
        fault_names = [fault.name for fault in self.faults]
        if fault_name in fault_names:
            st, se = self.slip_positions[fault_name]
            half = (st + se) // 2
            if strikeslip is not None:
                slb, sub = strikeslip
                self.lb[st:half] = slb
                self.ub[st:half] = sub
                if self.bounds['strikeslip'] is None:
                    self.bounds['strikeslip'] = {}
                self.bounds['strikeslip'][fault_name] = strikeslip
            if dipslip is not None:
                st = half
                dlb, dub = dipslip
                self.lb[st:se] = dlb
                self.ub[st:se] = dub
                if self.bounds['dipslip'] is None:
                    self.bounds['dipslip'] = {}
                self.bounds['dipslip'][fault_name] = dipslip
        else:
            print(f"Fault {fault_name} not found.")
        return
    
    def update_bounds_for_poly(self, fault_name, poly_bounds):
        """
        Update the bounds for the polynomial parameters of a fault.
        """
        fault_names = [fault.name for fault in self.faults]
        if fault_name in fault_names:
            st, se = self.poly_positions[fault_name]
            plb, pub = poly_bounds
            self.lb[st:se] = plb
            self.ub[st:se] = pub
            if self.bounds['poly'] is None:
                self.bounds['poly'] = {}
            self.bounds['poly'][fault_name] = poly_bounds
        else:
            print(f"Fault {fault_name} not found.")
        return

    def ConstrainedLeastSquareSoln(self, penalty_weight=1., smoothing_matrix=None, data_weight=1.,
                                smoothing_constraints=None, method='mudpy', Aueq=None, bueq=None, 
                                Aeq=None, beq=None, verbose=False, extra_parameters=None,
                                iterations=1000, tolerance=None, maxfun=100000, des_enabled=None):
        '''
        Perform a constrained least squares solution with optional smoothing.

        Parameters:
        - extra_parameters: Additional parameters for the solver.
        - penalty_weight: Weight for the smoothing matrix penalty.
        - iterations: Maximum number of iterations for the solver.
        - tolerance: Tolerance for the solver convergence.
        - maxfun: Maximum number of function evaluations.
        - smoothing_matrix: Matrix used for smoothing (Laplacian if provided).
        - smoothing_constraints : tuple or dict, optional. Ignored if smoothing_matrix is provided.
            Smoothing constraints to apply during the least squares process. If None, the function will use the combined Green's functions matrix.
            If a tuple, it should be a 4-tuple. If a dict, the keys should be fault names and the values should be 4-tuples.
            (top, bottom, left, right) for the smoothing constraints.
        - method: Solver method to use.
        - Aueq, bueq: Matrices for external inequality constraints (Aueq*x <= bueq).
        - Aeq, beq: Matrices for equality constraints (Aeq*x = beq).
        - verbose: Enable verbose output.
        - des_enabled: Whether to use Depth-Equalized Smoothing (DES). If None, uses self.des_enabled.

        Note:
        If smoothing_matrix is provided, smoothing_constraints will be ignored, as the smoothing_matrix directly
        incorporates smoothing into the solution.
        '''

        # Import DES utilities
        from .des_utils import apply_des_transformation, recover_sf_with_poly, get_poly_positions_from_multifaults

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
            # G_lap = np.zeros((smoothing_matrix.shape[0], Np))
            # G_lap[:, :Ns] = smoothing_matrix
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
        # Icovd = np.linalg.inv(Cd)
        # Icovd_chol = np.linalg.cholesky(Icovd)
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

        # ----------------------------Set constraints-----------------------------#
        A_ueq, b_ueq = self.A_ueq, self.b_ueq
        if Aueq is not None and bueq is not None:
            A_ueq = np.vstack((A_ueq, Aueq)) if A_ueq is not None else Aueq
            b_ueq = np.hstack((b_ueq, bueq)) if b_ueq is not None else bueq
        # Set the un-equality constraints for rake angle, with the form of A_ueq*x <= b_ueq
        self.A_ueq, self.b_ueq = A_ueq, b_ueq

        # Set the equality constraints for fixed rake angle, with the form of Aeq*x = beq
        if Aeq is not None and beq is not None:
            self.Aeq, self.beq = Aeq, beq

        # Set the constraint of the upper/lower Bounds
        lb, ub = self.lb, self.ub
        if any(np.isnan(lb)) or any(np.isnan(ub)):
            raise ValueError("You should assemble the upper/lower bounds first")

        # ----------------------------Apply DES transformation if enabled-----------------------------#
        if use_des:
            if verbose:
                print("Applying Depth-Equalized Smoothing (DES) transformation...")
            
            # Get polynomial positions
            poly_positions = get_poly_positions_from_multifaults(self)
            
            # Apply DES transformation to the original Green's function matrix G
            des_result = apply_des_transformation(
                G=G,  # Use original G matrix for DES parameter calculation
                D=G_lap,
                A_ineq=A_ueq,
                b_ineq=b_ueq,
                A_eq=self.Aeq,
                b_eq=self.beq,
                lb=lb,
                ub=ub,
                poly_positions=poly_positions,
                mode=self.des_config.get('mode', 'per_column'),
                groups=self.des_config.get('groups', None),
                G_norm=self.des_config.get('G_norm', 'l2'),
                depth_grouping_config=self.des_config.get('depth_grouping_config', None)
            )
            
            # Apply DES scaling to get G_prime
            G_prime = des_result['G_prime']
            
            # Now construct the augmented system with DES-scaled matrices
            d2I = np.vstack((np.dot(W, d)[:, None], d_lap[:, None])).flatten()
            G2I = np.vstack((np.dot(W, G_prime), des_result['D_prime']))
            
            # Update constraints with DES-transformed versions
            A_ueq_prime = des_result.get('A_ineq_prime', A_ueq)
            b_ueq_prime = des_result.get('b_ineq', b_ueq)
            Aeq_prime = des_result.get('A_eq_prime', self.Aeq)
            beq_prime = des_result.get('b_eq', self.beq)
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
            Aeq_prime, beq_prime = self.Aeq, self.beq
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
            if verbose:
                print("Recovering solution from DES transformation...")
            
            # Recover the final solution
            mpost = recover_sf_with_poly(
                mpost_prime, 
                des_result['alpha'], 
                des_result['norm2_fault'], 
                des_result['fault_indices']
            )
            
            if verbose:
                print("DES recovery completed")
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
                   smooth_mode='single', smooth_groups=None, smooth_update=None, smooth_values=None):
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

        from .simple_vce import simplified_vce
        from .des_utils import apply_des_transformation, recover_sf_with_poly, get_poly_positions_from_multifaults

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
        lb, ub = self.lb, self.ub
        if any(np.isnan(lb)) or any(np.isnan(ub)):
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

        # Prepare constraint matrices
        A_ueq = self.A_ueq
        b_ueq = self.b_ueq
        Aeq = self.Aeq
        beq = self.beq

        # Prepare for DES transformation if enabled
        if use_des:
            if verbose:
                print("Preparing DES transformation...")

            poly_positions = get_poly_positions_from_multifaults(self)

            des_result = apply_des_transformation(
                G=G,
                D=smoothing_matrix,
                A_ineq=A_ueq,
                b_ineq=b_ueq,
                A_eq=Aeq,
                b_eq=beq,
                lb=lb,
                ub=ub,
                poly_positions=poly_positions,
                mode=self.des_config.get('mode', 'per_column'),
                groups=self.des_config.get('groups', None),
                G_norm=self.des_config.get('G_norm', 'l2'),
                depth_grouping_config=self.des_config.get('depth_grouping_config', None)
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

            if verbose:
                print(f"DES applied: scaling factor range [{des_result['scale_factors'].min():.3f}, {des_result['scale_factors'].max():.3f}]")
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
            if verbose:
                print("Recovering solution from DES transformation...")

            m_prime = vce_result['m']
            m_recovered = recover_sf_with_poly(
                m_prime,
                des_result['alpha'],
                des_result['norm2_fault'],
                des_result['fault_indices']
            )

            vce_result['m'] = m_recovered

            if verbose:
                print("DES recovery completed")

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
                         smoothing_constraints=None, method='mudpy', verbose=False):
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
        # Set the constraint of the upper/lower Bounds
        lb, ub = self.lb, self.ub
        if any(np.isnan(lb)) or any(np.isnan(ub)):
            raise ValueError("You should assemble the upper/lower bounds first")
        
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