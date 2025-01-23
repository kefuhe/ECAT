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

# Plot
from eqtools.getcpt import get_cpt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import cmcrameri # cmc.devon_r cmc.lajolla_r cmc.batlow

class multifaultsolve_boundLSE(multifaultsolve):
    '''
    Invert for slip distribution and orbital parameters
        1. Add Laplace smoothing constraints
        2. Construct a new function to generate boundary constraints
        3. Write a function to assemble the smoothing matrix and corresponding data objects
        4. Add a constrained least squares inversion function
    '''
    
    def __init__(self, name, faults, verbose=True, extra_parameters=None):
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
                                   iterations=1000, tolerance=None, maxfun=100000):
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

        Note:
        If smoothing_matrix is provided, smoothing_constraints will be ignored, as the smoothing_matrix directly
        incorporates smoothing into the solution.
        '''
    
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

        # ----------------------------Inverse using lsqlin-----------------------------#
        # Set constraint
        A_ueq, b_ueq = self.A_ueq, self.b_ueq
        if Aueq is not None and bueq is not None:
            A_ueq = np.vstack((A_ueq, Aueq)) if A_ueq is not None else Aueq
            b_ueq = np.hstack((b_ueq, Aueq)) if b_ueq is not None else bueq
        # Set the un-equality constraints for rake angle, with the form of A_ueq*x <= b_ueq
        self.A_ueq, self.b_ueq = A_ueq, b_ueq

        # Set the equality constraints for fixed rake angle, with the form of Aeq*x = beq
        if Aeq is not None and beq is not None:
            self.Aeq, self.beq = Aeq, beq

        # Set the constraint of the upper/lower Bounds
        lb, ub = self.lb, self.ub
        if any(np.isnan(lb)) or any(np.isnan(ub)):
            raise ValueError("You should assemble the upper/lower bounds first")

        # Compute using lsqlin
        opts = {'show_progress': False}
        try:
            ret = lsqlin.lsqlin(G2I, d2I, 0, self.A_ueq, self.b_ueq, self.Aeq, self.beq, lb, ub, None, opts)
        except:
            ret = lsqlin.lsqlin(G2I, d2I, 0, self.A_ueq, self.b_ueq, None, None, lb, ub, None, opts)
        mpost = lsqlin.cvxopt_to_numpy_matrix(ret['x'])
        # Store mpost
        self.mpost = mpost

        # All done
        return
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

    def print_moment_magnitude(self, mu=3.e10, slip_factor=1.0):
        import csi.faultpostproc as faultpp
        # Get the first fault
        first_fault = self.faults[0]
        lon0, lat0, utmzone = first_fault.lon0, first_fault.lat0, first_fault.utmzone
    
        # Combine the faults
        combined_fault = first_fault.duplicateFault()
    
        # Add the patches and slip
        if len(self.faults) > 1:
            for ifault in self.faults[1:]:
                for patch, slip in zip(ifault.patch, ifault.slip):
                    combined_fault.N_slip = combined_fault.slip.shape[0] + 1
                    combined_fault.addpatch(patch, slip)
    
        # Get the combined fault name
        fault_names = [fault.name for fault in self.faults]
        combined_name = '_'.join(fault_names)

        # Scale the slip
        combined_fault.slip *= slip_factor
    
        # Patches 2 vertices
        combined_fault.setVerticesFromPatches()
        combined_fault.numpatch = combined_fault.Faces.shape[0]
        # Compute the triangle areas, moments, moment tensor and magnitude
        combined_fault.compute_triangle_areas()
        fault_processor = faultpp(combined_name, combined_fault, mu, lon0=lon0, lat0=lat0, utmzone=utmzone)
        fault_processor.computeMoments()
        fault_processor.computeMomentTensor()
        fault_processor.computeMagnitude()
    
        # Print the moment magnitude
        self.tripproc = fault_processor
        print(f"Mo is: {fault_processor.Mo:.8e}")
        print(f"Mw is {fault_processor.Mw:.1f}")
    
    def plot_multifaults_slip(self, figsize=(None, None), slip='total', cmap='precip3_16lev_change.cpt', norm=None,
                              show=True, savefig=False, ftype='pdf', dpi=600, bbox_inches=None, 
                              method='cdict', N=None, drawCoastlines=False, plot_on_2d=True,
                              style=['notebook'], cbaxis=[0.1, 0.2, 0.1, 0.02], cblabel='',
                              xlabelpad=None, ylabelpad=None, zlabelpad=None,
                              xtickpad=None, ytickpad=None, ztickpad=None,
                              elevation=None, azimuth=None, shape=(1.0, 1.0, 1.0), zratio=None, plotTrace=True,
                              depth=None, zticks=None, map_expand=0.2, fault_expand=0.1,
                              plot_faultEdges=False, faultEdges_color='k', faultEdges_linewidth=1.0, suffix='',
                              remove_direction_labels=False, zaxis_position='bottom-left', show_grid=True, grid_color='#bebebe',
                              background_color='white', axis_color=None, cbticks=None, cblinewidth=None, cbfontsize=None, cb_label_side='opposite',
                              map_cbaxis=None):
        """
        Plot the slip distribution of multiple faults.
    
        Parameters:
        - figsize (tuple): Size of the figure and map (default is (None, None)).
        - slip (str): Type of slip to plot (default is 'total').
        - cmap (str): Colormap to use (default is 'precip3_16lev_change.cpt').
        - norm: Normalization for the colormap (default is None).
        - show (bool): Whether to show the plot (default is True).
        - savefig (bool): Whether to save the figure (default is False).
        - ftype (str): File type for saving the figure (default is 'pdf').
        - dpi (int): Dots per inch for the saved figure (default is 600).
        - bbox_inches: Bounding box in inches for saving the figure (default is None).
        - method (str): Method for getting the colormap (default is 'cdict').
        - N (int): Number of colors in the colormap (default is None).
        - drawCoastlines (bool): Whether to draw coastlines (default is False).
        - plot_on_2d (bool): Whether to plot on a 2D map (default is True).
        - style (list): Style for the plot (default is ['notebook']).
        - cbaxis (list): Colorbar axis position (default is [0.1, 0.2, 0.1, 0.02]).
        - cblabel (str): Label for the colorbar (default is '').
        - xlabelpad, ylabelpad, zlabelpad (float): Padding for the axis labels (default is None).
        - xtickpad, ytickpad, ztickpad (float): Padding for the axis ticks (default is None).
        - elevation, azimuth (float): Elevation and azimuth angles for the 3D plot (default is None).
        - shape (tuple): Shape of the 3D plot (default is (1.0, 1.0, 1.0)).
        - zratio (float): Ratio for the z-axis (default is None).
        - plotTrace (bool): Whether to plot the fault trace (default is True).
        - depth (float): Depth for the z-axis (default is None).
        - zticks (list): Ticks for the z-axis (default is None).
        - map_expand (float): Expansion factor for the map (default is 0.2).
        - fault_expand (float): Expansion factor for the fault (default is 0.1).
        - plot_faultEdges (bool): Whether to plot the fault edges (default is False).
        - faultEdges_color (str): Color for the fault edges (default is 'k').
        - faultEdges_linewidth (float): Line width for the fault edges (default is 1.0).
        - suffix (str): Suffix for the saved figure filename (default is '').
        - remove_direction_labels : If True, remove E, N, S, W from axis labels (default is False)
        - zaxis_position (str): Position of the z-axis (bottom-left, top-right) (default is 'bottom-left').
        - show_grid (bool): Whether to show grid lines (default is True).
        - grid_color (str): Color of the grid lines (default is 'gray').
        - background_color (str): Background color of the plot (default is 'white').
        - axis_color (str): Color of the axes (default is None).
        - cbticks (list): List of ticks to set on the colorbar (default is None).
        - cblinewidth (int): Width of the colorbar label border and tick lines (default is 1).
        - cbfontsize (int): Font size of the colorbar label (default is 10).
        - cb_label_side (str): Position of the label relative to the ticks ('opposite' or 'same', default is 'opposite').
        - map_cbaxis    : Axis for the colorbar on the map plot, default is None
    
        Returns:
        - None
        """
        from ..plottools import plot_slip_distribution
    
        # Print Base Information
        for ifault in self.faults:
            print(f'Fault {ifault.name}: Mean Strike: {np.mean(ifault.getStrikes()*180/np.pi):.2f}°, Mean Dip: {np.mean(ifault.getDips()*180/np.pi):.2f}°')
    
        if len(self.faults) > 1:
            # Combine faults if there are more than one
            combined_fault = self.faults[0].duplicateFault()
            combined_fault.name = 'Combined Fault'
            for fault in self.faults[1:]:
                for ipatch, islip in zip(fault.patch, fault.slip):
                    combined_fault.N_slip = combined_fault.slip.shape[0] + 1
                    combined_fault.addpatch(ipatch, islip)
            combined_fault.setTrace(0.1)
            mfault = combined_fault
            add_faults = self.faults
        else:
            # Directly plot if there is only one fault
            mfault = self.faults[0]
            add_faults = [mfault]
    
        # Plot the slip distribution
        plot_slip_distribution(mfault, add_faults=add_faults, slip=slip, cmap=cmap, norm=norm, figsize=figsize, method=method, N=N,
                               drawCoastlines=drawCoastlines, plot_on_2d=plot_on_2d, cbaxis=cbaxis, cblabel=cblabel,
                               show=show, savefig=savefig, ftype=ftype, dpi=dpi, bbox_inches=bbox_inches,
                               remove_direction_labels=remove_direction_labels, cbticks=cbticks, cblinewidth=cblinewidth,
                               cbfontsize=cbfontsize, cb_label_side=cb_label_side, map_cbaxis=map_cbaxis, style=style,
                               xlabelpad=xlabelpad, ylabelpad=ylabelpad, zlabelpad=zlabelpad, xtickpad=xtickpad,
                               ytickpad=ytickpad, ztickpad=ztickpad, elevation=elevation, azimuth=azimuth, shape=shape,
                               zratio=zratio, plotTrace=plotTrace, depth=depth, zticks=zticks, map_expand=map_expand,
                               fault_expand=fault_expand, plot_faultEdges=plot_faultEdges, faultEdges_color=faultEdges_color,
                               faultEdges_linewidth=faultEdges_linewidth, suffix=suffix, show_grid=show_grid,
                               grid_color=grid_color, background_color=background_color, axis_color=axis_color,
                               zaxis_position=zaxis_position)
    
        # All Done
        return