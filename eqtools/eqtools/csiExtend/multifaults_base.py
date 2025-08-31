"""
Multi-Faults Base Classes Module

This module provides base classes for multi-fault inversion operations,
including fault geometry updates, Green's functions computation, and 
Laplacian matrix operations.
"""
# import external libraries
import numpy as np
from scipy.sparse import csr_matrix, block_diag

# import internal modules
from .multifaultsolve_boundLSE import multifaultsolve_boundLSE as multifaultsolve


class MyMultiFaultsInversion(multifaultsolve):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.faults_dict = {fault.name: fault for fault in self.faults}
        # To order the G matrix based on the order of the faults
        self.faultnames = [fault.name for fault in self.faults]

    def _apply_to_faults(self, func, fault_names=None):
        # Apply a function to the specified faults
        if fault_names is None:
            faults = self.faults_dict.values()  # Update all faults if fault_names is None
        else:
            faults = [self.faults_dict[name] for name in fault_names if name in self.faults_dict]  # Update specified faults
        for fault in faults:
            func(fault)

    def update_fault(self, method_name, fault_names=None, *args, **kwargs):
        # Update the specified faults
        def func(fault):
            try:
                method = getattr(fault, method_name)
                method(*args, **kwargs)  # Call the method with the specified arguments
            except AttributeError:
                raise ValueError(f"Fault object has no method named '{method_name}'")

        self._apply_to_faults(func, fault_names)

    def update_fault_geometry(self, method='perturb_bottom_coords', fault_names=None, perturbations=None, **kwargs):
        # Update the geometry of the specified faults using the specified method and perturbations (if any)
        method = kwargs.pop('method', method)
        # Update the geometry of the specified faults
        self.update_fault(method, fault_names=fault_names, perturbations=perturbations, **kwargs)

    def update_mesh(self, method='generate_mesh', fault_names=None, verbose=0, show=False, **kwargs):
        # Update the mesh of the specified faults using the specified method and arguments (if any)
        method = kwargs.pop('method', method)
        # Update the mesh of the specified faults
        self.update_fault(method, fault_names=fault_names, verbose=verbose, show=show, **kwargs)

    def update_GFs(self, geodata=None, verticals=None, fault_names=None, dataFaults=None, method=None, options=None):
        # Update the Green's functions of the specified faults
        def func(fault):
            # get the good indexes
            st_row = 0
            sliplist = [slip for slip, char in zip(['strikeslip', 'dipslip', 'tensile', 'coupling'], 'sdtc') if char in fault.slipdir]
            for obsdata, vertical, dataFault in zip(geodata, verticals, dataFaults or [None]*len(geodata)):
                # Determine method based on fault presence in dataFault
                gfmethod = 'empty' if dataFault is not None and fault.name not in dataFault else method
                # print(f"Method for {fault.name}: {gfmethod}")
                fault.buildGFs(obsdata, vertical=vertical, slipdir=fault.slipdir, method=gfmethod, verbose=False, **(options or {}))
                st = 0
                for sp in sliplist:
                    Nclocal = fault.G[obsdata.name][sp].shape[1]
                    Nrowlocal = fault.G[obsdata.name][sp].shape[0]

                    fault.Gassembled[st_row:st_row+Nrowlocal, st:st+Nclocal] = fault.G[obsdata.name][sp]
                    st += Nclocal
                st_row += Nrowlocal

            # get the good indexes for self.G
            st, se = self.fault_indexes[fault.name]
            # Store the G matrix
            self.G[:, st:se] = fault.Gassembled

        self._apply_to_faults(func, fault_names)

    def update_Laplacian(self, method='Mudpy', bounds=('free', 'free', 'free', 'free'), 
                         topscale=0.25, bottomscale=0.03, fault_names=None):
        # Update the Laplacian matrix of the specified faults using the specified method and arguments (if any)
        if not hasattr(self, 'GLs') or self.GLs is None:
            self.GLs = {}

        def func(fault):
            lap = fault.buildLaplacian(method=method, bounds=bounds, 
                                       topscale=topscale, bottomscale=bottomscale)
            if len(fault.slipdir) == 1:
                fault.GL = csr_matrix(lap)
            else:
                fault.GL = block_diag([lap for _ in fault.slipdir]).tocsr()

            self.GLs[fault.name] = fault.GL

        self._apply_to_faults(func, fault_names)
    
    def compute_data_inv_covs_and_logdets(self, geodata):
        inv_covs = {}
        chol_decomps = {}  # Store the Cholesky decomposition of the inverse covariance matrix
        logdets = {}
        if not self.faults:
            raise ValueError("No faults available.")
        for idataname, idata in zip(self.faults[0].datanames, geodata):
            if idata.name != idataname:
                raise ValueError(f"Data name mismatch: expected {idataname}, got {idata.name}")
            inv_cov = np.linalg.inv(idata.Cd)
            chol_decomp = np.linalg.cholesky(inv_cov)  # Compute the Cholesky decomposition of the inverse covariance matrix
            det = np.linalg.slogdet(idata.Cd)[1]
            inv_covs[idataname] = inv_cov
            chol_decomps[idataname] = chol_decomp  # Store the Cholesky decomposition of the inverse covariance matrix
            logdets[idataname] = det
        return inv_covs, chol_decomps, logdets  # Return the inverse covariance matrices, Cholesky decompositions, and log determinants

    def compute_fault_areas(self, fault_names=None):
        # Compute the areas of the patches of the specified faults
        if not hasattr(self, 'patch_areas') or self.patch_areas is None:
            self.patch_areas = {}

        def func(fault):
            areas = fault.compute_patch_areas()
            self.patch_areas[fault.name] = areas

        self._apply_to_faults(func, fault_names)
        return self.patch_areas


if __name__ == "__main__":
    pass