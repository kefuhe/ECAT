# import the necessary libraries
from csi import multifaultsolve
import copy
import yaml
import numpy as np
import pyproj as pp
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
    
    def calculate_slip_and_poly_positions(self):
        self.slip_positions = {}
        self.poly_positions = {}
        start_position = 0
        for fault in self.faults:
            npatches = fault.Faces.shape[0]
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
        rake: -180 <= rake <= 180, anti-clockwise is positive
        rake_ub - rake_lb <= 180
        '''

        npatch = 0
        Nsd = 0
        Np = self.lsq_parameters # self.G.shape[1]
        for fault in self.faults:
            npatch += fault.slip.shape[0]
            Nsd += int(fault.slip.shape[0]*len(fault.slipdir))
        A = np.zeros((Nsd, Np))
        b = np.zeros((Nsd,))
        patch_count = 0
        for fault in self.faults:
            irake = rake_limits[fault.name]
            inpatch = fault.slip.shape[0]
            start = self.fault_indexes[fault.name][0]
            half = start + inpatch
            rake_bound = np.zeros((inpatch, 2))
            rake_bound[:, 0] = irake[0]
            rake_bound[:, 1] = irake[1]
            # cross product
            for i in range(inpatch):
                ilb, iub = rake_bound[i]
                # cross product of slip and rake,
                # x: (ss, ds), y: (cos, sin) of rake with x.cross(y) = z, z = (ss*sin(rake) - ds*cos(rake)) < 0
                A[patch_count+i, start+i] = np.sin(np.deg2rad(ilb))
                A[patch_count+i, half+i] = -np.cos(np.deg2rad(ilb))
                # x: (ss, ds), y: (cos, sin) of rake with x.cross(y) = z, z = (ss*sin(rake) - ds*cos(rake)) > 0
                A[patch_count + i+inpatch, start+i] = -np.sin(np.deg2rad(iub))
                A[patch_count + i+inpatch, half+i] = np.cos(np.deg2rad(iub))
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
        - smoothing_constraints: Constraints for the smoothing matrix. Ignored if smoothing_matrix is provided.
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

        if smoothing_matrix is None:
            for ii, fault in enumerate(faults):
                st = self.fault_indexes[fault.name][0]
                if fault.type == 'Fault':
                    if fault.patchType in ('rectangle'):
                        lap = fault.buildLaplacian(method=method, bounds=smoothing_constraints)
                    else:
                        lap = fault.buildLaplacian(method=method, bounds=smoothing_constraints)
                    lap_sd = blkdiag(lap, lap)
                    Nsd = len(fault.slipdir)
                    # TODO: The following code is not clear, need to be modified
                    if Nsd == 1:
                        lap_sd = lap
                    se = st + Nsd*lap.shape[0]
                    G_lap[Ns_st[ii]:Ns_se[ii], st:se] = lap_sd
        else:
            G_lap = np.zeros((smoothing_matrix.shape[0], Np))
            G_lap[:, :Ns] = smoothing_matrix
            d_lap = np.zeros((G_lap.shape[0], ))
        self.G_lap = G_lap

        G_lap2I = penalty_weight*G_lap

        Icovd = np.linalg.inv(Cd)
        W = np.linalg.cholesky(Icovd) * data_weight
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
        mpost = ret['x']
        # Store mpost
        self.mpost = lsqlin.cvxopt_to_numpy_matrix(mpost)

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
    
    def plot_multifaults_slip(self, figsize=(None, None), slip='total', cmap='precip3_16lev_change.cpt', 
                              show=True, savefig=False, ftype='pdf', dpi=600, bbox_inches=None, 
                              method='cdict', N=None, drawCoastlines=False, plot_on_2d=True,
                              style=['notebook'], cbaxis=[0.1, 0.2, 0.1, 0.02], cblabel='',
                              xlabelpad=None, ylabelpad=None, zlabelpad=None,
                              xtickpad=None, ytickpad=None, ztickpad=None,
                              elevation=None, azimuth=None, shape=(1.0, 1.0, 1.0), plotTrace=True,
                              depth=None, zticks=None, map_expand=0.2, fault_expand=0.1,
                              plot_faultEdges=False, faultEdges_color='k', faultEdges_linewidth=1.0,):
        '''
        figsize: default (None, None), the size of the figure and map
        '''
        if isinstance(cmap, str) and cmap.endswith('.cpt'):
            cmap = get_cpt.get_cmap(cmap, method=method, N=N)

        with sci_plot_style(style=style):
            if len(self.faults) > 1:
                # Combine faults if there are more than one
                combined_fault = self.faults[0].duplicateFault()
                combined_fault.name = 'Combined Fault'
                for fault in self.faults[1:]:
                    for ipatch, islip in zip(fault.patch, fault.slip):
                        combined_fault.N_slip = combined_fault.slip.shape[0] + 1
                        combined_fault.addpatch(ipatch, islip)
                combined_fault.setTrace(0.1)
                combined_fault.plot(drawCoastlines=drawCoastlines, slip=slip, cmap=cmap, savefig=False, 
                                    ftype=ftype, dpi=dpi, bbox_inches=bbox_inches, plot_on_2d=plot_on_2d, 
                                    figsize=figsize, cbaxis=cbaxis, cblabel=cblabel, show=False, expand=map_expand)
                ax = combined_fault.slipfig.faille
                fig = combined_fault.slipfig
                name = combined_fault.name
                mfault = combined_fault
            else:
                # Directly plot if there is only one fault
                self.faults[0].plot(drawCoastlines=drawCoastlines, slip=slip, cmap=cmap, savefig=False, 
                                                ftype=ftype, dpi=dpi, bbox_inches=bbox_inches, plot_on_2d=plot_on_2d,
                                                figsize=figsize, cbaxis=cbaxis, cblabel=cblabel, show=False, expand=map_expand)
                ax = self.faults[0].slipfig.faille
                fig = self.faults[0].slipfig
                name = self.faults[0].name
                mfault = self.faults[0]

            if plot_faultEdges:
                for fault in self.faults:
                    fault.find_fault_fouredge_vertices(refind=True)
                    for edgename in fault.fouredgepntsInVertices:
                        edge = fault.fouredgepntsInVertices[edgename]
                        x, y, z = edge[:, 0], edge[:, 1], -edge[:, 2]
                        lon, lat = fault.xy2ll(x, y)
                        ax.plot(lon, lat, z, color=faultEdges_color, linewidth=faultEdges_linewidth)            
            
            if plotTrace:
                for fault in self.faults:
                    fig.faulttrace(fault, color='r', discretized=False,  linewidth=1, zorder=1)

            # Set labels and title with optional labelpad
            ax.set_xlabel('Longitude', labelpad=xlabelpad)
            ax.set_ylabel('Latitude', labelpad=ylabelpad)
            ax.set_zlabel('Depth (km)', labelpad=zlabelpad)
            
            # Adjust tick parameters with optional pad
            if xtickpad is not None:
                ax.tick_params(axis='x', pad=xtickpad)
            if ytickpad is not None:
                ax.tick_params(axis='y', pad=ytickpad)
            if ztickpad is not None:
                ax.tick_params(axis='z', pad=ztickpad)
            
            # Set Z tick labels
            if depth is not None and zticks is not None:
                ax.set_zlim3d([-depth, 0])
                ax.set_zticks(zticks)
            if fault_expand is not None:
                # Get lons lats
                lon = np.unique(np.array([p[:,0] for p in mfault.patchll]))
                lat = np.unique(np.array([p[:,1] for p in mfault.patchll]))
                lonmin, lonmax = lon.min(), lon.max()
                latmin, latmax = lat.min(), lat.max()
                ax.set_xlim(lonmin-fault_expand, lonmax+fault_expand)
                ax.set_ylim(latmin-fault_expand, latmax+fault_expand)
            ax.zaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{abs(val)}'))
            
            # Set View, reference to csi.geodeticplot.set_view
            if elevation is not None and azimuth is not None:
                ax.view_init(elev=elevation, azim=azimuth)
            # Set Z axis ratio
            ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), 
                np.diag([shape[0], shape[1], shape[2], 1]))

            if savefig:
                prefix = name.replace(' ','_')
                saveFig = ['fault']
                if plot_on_2d:
                    saveFig.append('map')
                fig.savefig(prefix+'_{}'.format(slip), ftype=ftype, dpi=dpi, bbox_inches=bbox_inches, saveFig=saveFig)

            if show:
                showFig = ['fault']
                if plot_on_2d:
                    showFig.append('map')
                fig.show(showFig=showFig)
                plt.show()