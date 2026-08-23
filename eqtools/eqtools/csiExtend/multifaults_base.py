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
from .covariance_utils import prepare_covariance_metric
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
            except AttributeError as exc:
                raise ValueError(
                    f"Fault object has no method named '{method_name}'"
                ) from exc
            # Keep method-body AttributeError exceptions intact.  Recasting
            # them as lookup failures hides the real scientific-state error
            # and its traceback during preflight or candidate evaluation.
            method(*args, **kwargs)

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
        targets = (
            self.faults_dict.values() if fault_names is None
            else (
                self.faults_dict[name]
                for name in fault_names
                if name in self.faults_dict
            )
        )
        for fault in targets:
            if hasattr(fault, 'mesh_valid'):
                fault.mesh_valid = (
                    getattr(fault, 'Vertices', None) is not None
                    and getattr(fault, 'Faces', None) is not None
                )
            if hasattr(fault, 'laplacian_valid'):
                fault.laplacian_valid = getattr(fault, 'GL', None) is not None
            if hasattr(fault, 'area_valid'):
                area = getattr(fault, 'area', None)
                fault.area_valid = (
                    area is not None and np.asarray(area).size > 0
                )

    def update_GFs(self, geodata=None, verticals=None, fault_names=None, dataFaults=None, method=None, options=None):
        """
        Update the Green's functions of the specified faults.
        Uses source adapters for type-safe GF building.
        """
        def func(fault):
            adapter = self.adapters[fault.name]
            gf_keys = adapter.get_gf_column_keys()
            st_row = 0
            
            for obsdata, vertical, dataFault in zip(geodata, verticals, dataFaults or [None]*len(geodata)):
                # Determine method based on fault presence in dataFault
                gfmethod = 'empty' if dataFault is not None and fault.name not in dataFault else method
                
                # Build GFs using adapter (handles type-specific call signatures)
                adapter.build_gfs(obsdata, vertical, method=gfmethod, options=options)
                
                # Assemble GFs into fault.Gassembled
                st = 0
                for key in gf_keys:
                    if key in fault.G[obsdata.name]:
                        Gsp = fault.G[obsdata.name][key]
                        Nclocal = Gsp.shape[1]
                        Nrowlocal = Gsp.shape[0]
                        
                        fault.Gassembled[st_row:st_row+Nrowlocal, st:st+Nclocal] = Gsp
                        st += Nclocal
                
                st_row += Nrowlocal

            # Get the good indexes for self.G
            st, se = self.fault_indexes[fault.name]
            # Store the G matrix
            self.G[:, st:se] = fault.Gassembled

        self._apply_to_faults(func, fault_names)

    def update_Laplacian(self, method='Mudpy', bounds=('free', 'free', 'free', 'free'), 
                         topscale=0.25, bottomscale=0.03, fault_names=None):
        """
        Update the Laplacian matrix of the specified faults.
        Only applicable to sources that support smoothing (via adapter).
        """
        if not hasattr(self, 'GLs') or self.GLs is None:
            self.GLs = {}

        def func(fault):
            adapter = self.adapters[fault.name]
            if not adapter.supports_smoothing():
                if hasattr(fault, 'laplacian_valid'):
                    fault.laplacian_valid = False
                return
            
            # Build Laplacian matrix using adapter
            fault.GL = adapter.build_laplacian(method=method, bounds=bounds, 
                                               topscale=topscale, bottomscale=bottomscale)
            self.GLs[fault.name] = fault.GL
            if hasattr(fault, 'laplacian_valid'):
                fault.laplacian_valid = fault.GL is not None

        self._apply_to_faults(func, fault_names)
    
    def compute_data_covariance_metrics(self, geodata):
        """Prepare one reusable covariance metric per named data set.

        Each metric is derived directly from that data set's covariance and
        contains ``W`` and ``log(det(C))``. No full precision matrix or global
        block-diagonal whitener is formed.
        """
        metrics = {}
        if not self.faults:
            raise ValueError("No faults available.")
        datanames = list(self.faults[0].datanames)
        geodata = list(geodata)
        if len(set(datanames)) != len(datanames):
            raise ValueError("Data names must be unique.")
        if len(geodata) != len(datanames):
            raise ValueError(
                f"Expected {len(datanames)} data sets, got {len(geodata)}"
            )
        for idataname, idata in zip(datanames, geodata):
            if idata.name != idataname:
                raise ValueError(
                    f"Data name mismatch: expected {idataname}, got {idata.name}"
                )
            metrics[idataname] = prepare_covariance_metric(
                idata.Cd,
                name=f"covariance for data set '{idataname}'",
            )
            expected_size = np.asarray(self.faults[0].d[idataname]).shape[0]
            if metrics[idataname].size != expected_size:
                raise ValueError(
                    f"{idataname}: covariance size "
                    f"{metrics[idataname].size} does not match observation "
                    f"length {expected_size}"
                )
        return metrics

    def compute_fault_areas(self, fault_names=None):
        """Force area computation and update the compatibility snapshot."""
        if not hasattr(self, 'patch_areas') or self.patch_areas is None:
            self.patch_areas = {}

        def func(fault):
            adapter = self.adapters[fault.name]
            areas = adapter.compute_patch_areas()
            if areas is not None:
                self.patch_areas[fault.name] = areas
            if hasattr(fault, 'area_valid'):
                fault.area_valid = areas is not None and np.asarray(areas).size > 0

        self._apply_to_faults(func, fault_names)
        return self.patch_areas

    def get_fault_areas(self, fault_names=None):
        """Return a call-local snapshot of current source areas.

        CSI owns each source cache. This aggregator deliberately does not
        retain a second candidate cache: fixed geometries are reused by the
        source getter, while deformed geometries are recomputed only after CSI
        invalidates their source-local cache.
        """
        current_areas = {}

        def func(fault):
            areas = self.adapters[fault.name].get_patch_areas()
            if areas is not None:
                current_areas[fault.name] = areas
            if hasattr(fault, 'area_valid'):
                fault.area_valid = areas is not None and np.asarray(areas).size > 0

        self._apply_to_faults(func, fault_names)
        return current_areas


if __name__ == "__main__":
    pass
