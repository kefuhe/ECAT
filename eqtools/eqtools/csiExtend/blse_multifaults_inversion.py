import scipy
import numpy as np
import copy
import os
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

from .multifaults_base import MyMultiFaultsInversion
from .config.blse_config import BoundLSEInversionConfig
from .constraint_manager_blse import ConstraintManager
from .data_correction_constraints import DataCorrectionConstraintMixin
from .data_correction_report_mixin import DataCorrectionReportMixin
from .deep_slip_loading_mixin import DeepSlipLoadingMixin
from .interseismic_mixin import InterseismicKinematicsMixin
from .patch_indices import normalize_patch_indices
from ..plottools import sci_plot_style
from .data_plot_utils import _plot_leveling_fit, _plot_crossfaultoffset_fit

class BoundLSEMultiFaultsInversion(
    DataCorrectionReportMixin,
    DataCorrectionConstraintMixin,
    DeepSlipLoadingMixin,
    InterseismicKinematicsMixin,
    MyMultiFaultsInversion,
):
    def __init__(self, name, faults_list, geodata=None, config='default_config_BLSE.yml', encoding='utf-8',
                 gfmethods=None, bounds_config='bounds_config.yml', interseismic_config=None,
                 rake_limits=None, extra_parameters=None, verbose=True, des_enabled=False, des_config=None):
        """
        Initialize BoundLSEMultiFaultsInversion with DES support.
        
        Parameters:
        -----------
        name : str
            Name of the inversion
        faults_list : list
            List of fault objects
        geodata : object, optional
            Geodetic data object
        config : str or object, optional
            Configuration file path or config object (default: 'default_config_BLSE.yml')
        encoding : str, optional
            File encoding (default: 'utf-8')
        gfmethods : dict, optional
            Green's function methods
        bounds_config : str, optional
            Bounds configuration file (default: 'bounds_config.yml')
        interseismic_config : str or dict, optional
            Interseismic block-motion and optional cap/backslip constraint
            configuration.  If omitted, ``config.interseismic_config_file`` is
            used when present.
        rake_limits : dict, optional
            Rake angle limits
        extra_parameters : dict, optional
            Additional parameters for the solver
        verbose : bool, optional
            Enable verbose output (default: True)
        des_enabled : bool, optional
            Whether to enable Depth-Equalized Smoothing (DES) (default: False)
        des_config : dict, optional
            DES configuration parameters (default: None)
        """
        # Initialize the faults ahead of the configuration
        self.faults = faults_list
        self.faults_dict = {fault.name: fault for fault in self.faults}
        # To order the G matrix based on the order of the faults
        self.faultnames = [fault.name for fault in self.faults]
        
        # Initialize BoundLSEInversionConfig first
        if isinstance(config, str):
            assert geodata is not None, "geodata must be provided when config is a file"
            self.config = BoundLSEInversionConfig(config, multifaults=None, 
                                                  geodata=geodata, 
                                                  faults_list=faults_list, 
                                                  gfmethods=gfmethods, 
                                                  encoding=encoding,
                                                  verbose=verbose)
        else:
            self.config = config

        if interseismic_config is None:
            interseismic_config = getattr(self.config, 'interseismic_config_file', None)
        if interseismic_config is not None:
            self.config.load_interseismic_config(interseismic_config, encoding=encoding)

        # Initialize MyMultiFaultsInversion with DES support
        super(BoundLSEMultiFaultsInversion, self).__init__(name, 
                                                           faults_list, 
                                                           extra_parameters=extra_parameters, 
                                                           verbose=verbose,
                                                           des_enabled=des_enabled,
                                                           des_config=des_config)

        self.assembleGFs()
        
        self.update_config(self.config)

        # DES (Depth-Equalized Smoothing) parameters
        des_from_config = getattr(self.config, 'des', {'enabled': False})
        des_config = des_config if des_config is not None else des_from_config
        self.des_enabled = des_enabled or des_config.get('enabled', False)
        self.des_config = des_config if des_config is not None else {
            'mode': 'per_patch',
            'G_norm': 'l2',
            'depth_grouping': {
                'strategy': 'uniform',
                'interval': 1.0
                }
        }
        
        # Apply all constraints using the constraint manager
        self.constraint_manager.apply_all_constraints_from_config(
            bounds_config_file=bounds_config,
            rake_limits=rake_limits,
            encoding=encoding
        )
        
        # Sync constraints to solver for backward compatibility
        self.constraint_manager.sync_to_solver()

    def update_config(self, config):
        self.config = config
        if hasattr(self, 'constraint_manager'):
            self.constraint_manager.config = config
        self._update_faults()

    def update_interseismic_config(self, interseismic_config, reapply=True):
        """Load a new interseismic config and optionally rebuild its constraints."""
        parsed = self.config.load_interseismic_config(interseismic_config)
        if reapply and hasattr(self, 'constraint_manager'):
            self.constraint_manager.remove_constraint('euler_cap_constraints', 'inequality')
            for constraint_name in ('interseismic_block_euler_constraints', 'interseismic_block_euler_sharing'):
                if constraint_name in getattr(self.constraint_manager, '_equality_constraints', {}):
                    self.constraint_manager.remove_constraint(constraint_name, 'equality')
            for name in list(getattr(self.constraint_manager, '_equality_constraints', {})):
                group = self.constraint_manager._equality_constraints[name]
                if group.get('source') == 'interseismic_config.backslip_constraints':
                    self.constraint_manager.remove_constraint(name, 'equality')
            if parsed.get('blocks', {}).get('enabled', False):
                self.constraint_manager.apply_interseismic_block_constraints()
            self.constraint_manager.apply_euler_cap_constraints()
            self.constraint_manager.apply_interseismic_backslip_constraints()
            self.constraint_manager.sync_to_solver()
        return parsed

    def update_euler_cap_constraint(
        self,
        fault_name,
        *,
        selector=None,
        max_coupling=None,
        mode=None,
        min_loading_abs=None,
        enabled=None,
        reapply=True,
    ):
        """Update optional cap-constraint selector for one fault.

        Parameters
        ----------
        fault_name : str
            Fault whose cap constraint should be updated.
        selector : dict or iterable of int, optional
            Patch selector for cap rows only.  It does not affect tectonic
            loading-rate calculation.
        max_coupling : float, optional
            Upper multiplier ``k`` in ``|backslip| <= k * |loading|``.
            Defaults to the value already stored in config, or 1.0.
        mode : {"motion_sense", "loading_sign"}, optional
            Cap construction mode.  ``motion_sense`` is the default and works
            with estimated Euler loading.  ``loading_sign`` requires fixed
            loading and constrains ``0 <= -q / b <= k`` from the projected sign.
        min_loading_abs : float, optional
            Minimum absolute loading accepted by ``mode="loading_sign"``.
        enabled : bool, optional
            If provided, update ``cap_constraints.enabled``.
        reapply : bool, default True
            If True, rebuild the cap constraint matrix after updating config.

        Returns
        -------
        dict
            Current parsed ``config.interseismic_config`` dictionary.
        """
        if fault_name not in self.faults_dict:
            raise ValueError(
                f"Fault '{fault_name}' not found. Available: {list(self.faults_dict.keys())}"
            )

        interseismic = copy.deepcopy(getattr(self.config, 'interseismic_config', {}))
        cap = interseismic.setdefault('cap_constraints', {})
        if enabled is not None:
            cap['enabled'] = bool(enabled)
        cap.setdefault('faults', {})
        cap['faults'].setdefault(fault_name, {})
        if selector is not None:
            if isinstance(selector, (list, tuple, np.ndarray)):
                indices = normalize_patch_indices(
                    self.faults_dict[fault_name],
                    selector,
                    allow_none_all=False,
                    unique=True,
                    name=f"cap selector for fault '{fault_name}'",
                )
                selector = {'patches': indices.tolist()}
            cap['faults'][fault_name]['selector'] = selector
        if max_coupling is not None:
            max_coupling = float(max_coupling)
            if max_coupling < 0.0:
                raise ValueError("max_coupling must be non-negative")
            cap['faults'][fault_name]['max_coupling'] = max_coupling
        if mode is not None:
            cap['faults'][fault_name]['mode'] = str(mode)
        if min_loading_abs is not None:
            min_loading_abs = float(min_loading_abs)
            if min_loading_abs < 0.0:
                raise ValueError("min_loading_abs must be non-negative")
            cap['faults'][fault_name]['min_loading_abs'] = min_loading_abs
        return self.update_interseismic_config(interseismic, reapply=reapply)

    def _update_faults(self):
        # Update the faults based on the configuration parameters and method parameters for each fault 
        datanames = [d.name for d in self.config.geodata.get('data', [])]
        Nd = len(datanames)
        faultnames = self.faultnames
        for fault_name, fault_config in self.config.faults.items():
            if fault_name != 'defaults':
                # Update Green's functions
                dataFaults = self.config.dataFaults
                # Check if dataFaults is a list of lists, each equal to faultnames
                if not (isinstance(dataFaults, list) and len(dataFaults) == Nd and all(fault_name in flist for flist in dataFaults)):
                    self.update_GFs(fault_names=[fault_name], **fault_config['method_parameters']['update_GFs'])
                # Update Laplacian
                self.update_Laplacian(fault_names=[fault_name], **fault_config['method_parameters']['update_Laplacian'])

    def run(self, penalty_weight=None, smoothing_constraints=None, data_weight=None, data_log_scaled=None, 
            penalty_log_scaled=None, sigma=None, alpha=None, verbose=True, des_enabled=None):
        """
        Start the boundary-constrained least squares process.
    
        Parameters:
        -----------
        penalty_weight : int, float, list, or np.ndarray, optional
            Penalty weights to apply to the Green's functions. If None, the function will use the initial values from the configuration.
        smoothing_constraints : tuple or dict, optional
            Smoothing constraints to apply during the least squares process. If None, the function will use the combined Green's functions matrix.
            If a tuple, it should be a 4-tuple. If a dict, the keys should be fault names and the values should be 4-tuples.
            (top, bottom, left, right) for the smoothing constraints.
        data_weight : np.ndarray, optional
            Weights to apply to the data. If None, the function will use the initial values from the configuration.
        data_log_scaled : bool, optional
            Whether to apply log scaling to the data weights. If None, the function will use the log_scaled value from the configuration.
        penalty_log_scaled : bool, optional
            Whether to apply log scaling to the penalty weights. If None, the function will use the log_scaled value from the configuration.
        sigma : np.ndarray, optional
            Data standard deviations. If None, the function will use the initial values from the configuration.
        alpha : np.ndarray, optional
            Smoothing standard deviations. If None, the function will use the initial values from the configuration.
        verbose : bool, optional
            Whether to print the results of the inversion. Default is True.
        des_enabled : bool, optional
            Whether to use Depth-Equalized Smoothing (DES). If None, uses self.des_enabled.
    
        Returns:
        --------
        None
        """
        from .config.config_utils import parse_initial_values

        # Ensure data_weight and sigma are either both None or only one is provided
        if (data_weight is not None) and (sigma is not None):
            raise ValueError("data_weight and sigma must either both be None or only one is provided.")
    
        # Ensure penalty_weight and alpha are either both None or only one is provided
        if (penalty_weight is not None) and (alpha is not None):
            raise ValueError("penalty_weight and alpha must either both be None or only one is provided.")
    
        # Handle data weights
        n_datasets = len(self.config.sigmas['update'])
        if self.config.sigmas['mode'] == 'single':
            data_names = ['All_data']
        elif self.config.sigmas['mode'] == 'individual':
            data_names = [d.name for d in self.config.geodata.get('data', [])]
        elif self.config.sigmas['mode'] == 'grouped':
            data_names = list(self.config.sigmas['groups'].keys())
        data_indices = self.config.sigmas['dataset_param_indices']
        if data_weight is None:
            if sigma is None:
                sigma = self.config.sigmas['initial_value']
            else:
                sigma = parse_initial_values({'initial_value': sigma},
                                                n_datasets=n_datasets,
                                                param_name='initial_value',  # initial_value or 'values'
                                                dataset_names=data_names,
                                                print_name='sigma')
            sigma = np.array(sigma)
            if data_log_scaled is None:
                data_log_scaled = self.config.sigmas['log_scaled']
            if data_log_scaled:
                sigma = np.power(10, sigma)
            data_weight = 1.0 / sigma
        else:
            wgt_dict = {'initial_value': data_weight}
            data_weight = parse_initial_values(wgt_dict, n_datasets=n_datasets,
                                                param_name='initial_value',  # initial_value or 'values'
                                                dataset_names=data_names,
                                                print_name='data_weight')
            data_weight = np.array(data_weight)
        data_weight = data_weight[data_indices]

        # Handle penalty weights
        # If alpha smoothing is disabled, use uniform weight (no regularization penalty)
        if not self.config.alpha_enabled:
            penalty_weight = np.ones(len(self.faults))
            self.current_penalty_weight = penalty_weight
            # Alpha disabled: use empty smoothing matrix, ignore smoothing_constraints
            self.combine_GL_poly(penalty_weight=penalty_weight)
            self.ConstrainedLeastSquareSoln(penalty_weight=penalty_weight,
                                            smoothing_matrix=self.GL_combined_poly,
                                            data_weight=data_weight,
                                            des_enabled=des_enabled,
                                            verbose=True)
            self.distributem()
            return
        else:
            n_faults = len(self.config.alpha['update'])
            if self.config.alpha['mode'] == 'single':
                fault_names = ['All_faults']
            elif self.config.alpha['mode'] == 'individual':
                fault_names = [fault.name for fault in self.faults]
            elif self.config.alpha['mode'] == 'grouped':
                fault_names = [f'Event_{i}' for i in range(n_faults)]
            fault_indices = self.config.alpha['fault_param_indices']

            if penalty_weight is None:
                if alpha is None:
                    alpha = self.config.alpha['initial_value']
                    # print('alpha is from config:', alpha)
                else:
                    alpha = parse_initial_values({'initial_value': alpha},
                                                    n_datasets=n_faults,
                                                    param_name='initial_value',  # initial_value or 'values'
                                                    dataset_names=fault_names,
                                                    print_name='alpha')
                alpha = np.array(alpha)
                if penalty_log_scaled is None:
                    penalty_log_scaled = self.config.alpha['log_scaled']
                if penalty_log_scaled:
                    alpha = np.power(10, alpha)
                penalty_weight = 1.0 / alpha
            else:
                penalty_weight = parse_initial_values({'initial_value': penalty_weight},
                                                      n_datasets=n_faults,
                                                      param_name='initial_value',  # initial_value or 'values'
                                                      dataset_names=fault_names,
                                                      print_name='penalty_weight')
                penalty_weight = np.array(penalty_weight)
            penalty_weight = penalty_weight[fault_indices]

            self.current_penalty_weight = penalty_weight
        # Handle smoothing constraints
        if smoothing_constraints is not None:
            if isinstance(smoothing_constraints, (tuple, list)) and len(smoothing_constraints) == 4:
                smoothing_constraints = {fault_name: smoothing_constraints for fault_name in self.faultnames}
            elif isinstance(smoothing_constraints, dict):
                assert all(fault_name in smoothing_constraints for fault_name in self.faultnames), "All fault names must be in smoothing_constraints."
            else:
                raise ValueError("smoothing_constraints should be a 4-tuple or a dictionary with fault names as keys and 4-tuples as values.")
    
        if smoothing_constraints is not None:
            self.ConstrainedLeastSquareSoln(penalty_weight=penalty_weight, 
                                            smoothing_constraints=smoothing_constraints, 
                                            data_weight=data_weight,
                                            des_enabled=des_enabled,
                                            verbose=True)
        else:
            self.combine_GL_poly(penalty_weight=penalty_weight)
            self.ConstrainedLeastSquareSoln(penalty_weight=penalty_weight, 
                                            smoothing_matrix=self.GL_combined_poly,
                                            data_weight=data_weight,
                                            des_enabled=des_enabled,
                                            verbose=True)
        self.distributem()

    def run_simple_vce(self, smoothing_constraints=None, verbose=True, max_iter=20, tol=1e-4, 
                       des_enabled=None, sigma_mode=None, sigma_groups=None, sigma_update=None, sigma_values=None,
                       smooth_mode=None, smooth_groups=None, smooth_update=None, smooth_values=None):
        """
        Run Simple Variance Component Estimation (VCE) for multi-fault inversion.
    
        This method automatically determines optimal weights between data fitting and
        regularization components using iterative VCE approach with lsqlin solver.
        No manual penalty weights are needed - they are estimated through VCE iterations.
    
        Parameters
        ----------
        smoothing_constraints : tuple or dict, optional
            Smoothing constraints to apply during the least squares process. If None,
            the function will use the combined Green's functions matrix.
            If a tuple, it should be a 4-tuple. If a dict, the keys should be fault
            names and the values should be 4-tuples.
            (top, bottom, left, right) for the smoothing constraints.
        verbose : bool, optional
            Whether to print detailed progress information. Default is True.
        max_iter : int, optional
            Maximum number of VCE iterations. Default is 10.
        tol : float, optional
            Convergence tolerance for VCE. Default is 1e-4.
        des_enabled : bool, optional
            Whether to use Depth-Equalized Smoothing (DES). If None, uses self.des_enabled.
        sigma_mode : str, optional
            Mode for data variance components: 'single', 'individual', or 'grouped'.
        sigma_groups : dict, optional
            Custom grouping for data variance components when sigma_mode='grouped'.
        sigma_update : list of bool, optional
            Whether to update each sigma group (same order as sigma groups)
        sigma_values : list of float, optional
            Initial/fixed values for each sigma group (same order as sigma groups)
        smooth_mode : str, optional
            Mode for smoothing variance components: 'single', 'individual', or 'grouped'.
        smooth_groups : dict, optional
            Custom grouping for smoothing variance components when smooth_mode='grouped'.
        smooth_update : list of bool, optional
            Whether to update each smoothing group (same order as smoothing groups)
        smooth_values : list of float, optional
            Initial/fixed values for each smoothing group (same order as smoothing groups)
    
        Returns
        -------
        dict
            VCE results containing:
            - 'm': estimated parameters
            - 'var_d': data variance components
            - 'var_alpha': regularization variance components
            - 'weights': final weight ratios
            - 'converged': convergence flag
            - 'iterations': number of iterations
        """
        sigma_mode = self.config.sigmas.get('mode', 'individual') if sigma_mode is None else sigma_mode
        sigma_groups = self.config.sigmas.get('groups', None) if sigma_groups is None else sigma_groups
        sigma_update = self.config.sigmas['update'] if sigma_update is None else sigma_update
        sigma_values = self.config.sigmas['initial_value'] if sigma_values is None else sigma_values
        if self.config.sigmas['log_scaled']:
            sigma_values = np.power(10, sigma_values)**2
        else:
            sigma_values = np.array(sigma_values)**2
        # print(sigma_mode, sigma_groups, sigma_update, sigma_values)

        # Check if alpha (smoothing) is disabled
        alpha_disabled = not self.config.alpha_enabled

        if alpha_disabled:
            # Sigma-only VCE: no smoothing estimation, only data weighting
            smooth_mode = 'single'
            smooth_groups = {'no_smooth': self.faultnames}
            smooth_values = [1.0]
            smooth_update = [False]  # Never update alpha 鈥?nothing to estimate
        else:
            alphaFaults = self.config.alphaFaults
            if len(alphaFaults) == 1:
                smooth_mode = 'single'
                smooth_groups = {'Event_all': alphaFaults[0]}
                smooth_values = [self.config.alpha['initial_value'][0]]
                smooth_update = [True]
            else:
                smooth_mode = 'grouped'
                smooth_groups = {f'Event_{i}': alphaFaults[i] for i in range(len(alphaFaults))}
                smooth_values = self.config.alpha['initial_value']
                smooth_update = self.config.alpha['update']
            if self.config.alpha['log_scaled']:
                smooth_values = np.power(10, smooth_values)**2
            else:
                smooth_values = np.array(smooth_values)**2
        # print(smooth_mode, smooth_groups, smooth_update, smooth_values)
        if des_enabled is None:
            des_enabled = getattr(self, 'des_enabled', False)

        if verbose:
            print("="*70)
            print("Starting Simple VCE for Multi-Fault Inversion")
            if alpha_disabled:
                print("Alpha smoothing is DISABLED 鈥?running sigma-only VCE (data weighting only).")
            else:
                print("Automatically determining optimal regularization weights...")
            print(f"Number of faults: {len(self.faults)}")
            print(f"Data variance mode: {sigma_mode}")
            print(f"Smoothing variance mode: {smooth_mode}")
            print(f"DES enabled: {des_enabled}")
            print("="*70)
    
        # Ensure bounds are set through the constraint manager, which is the
        # canonical source for BLSE/VCE constraints.
        lb = self.constraint_manager.lb
        ub = self.constraint_manager.ub
        if lb is None or ub is None:
            raise ValueError("Bounds must be set before running VCE. Use set_bounds_from_config() or set_bounds().")
        if np.any(np.isnan(lb)) or np.any(np.isnan(ub)):
            raise ValueError("Some bounds are not set (NaN values found). Please set all bounds first.")
    
        # Handle smoothing constraints (ignored when alpha is disabled)
        if alpha_disabled:
            smoothing_constraints = None
            if verbose:
                print("Alpha disabled 鈥?smoothing_constraints ignored, using empty smoothing matrix.")
        elif smoothing_constraints is not None:
            if isinstance(smoothing_constraints, (tuple, list)) and len(smoothing_constraints) == 4:
                smoothing_constraints = {fault_name: smoothing_constraints for fault_name in self.faultnames}
            elif isinstance(smoothing_constraints, dict):
                missing_faults = set(self.faultnames) - set(smoothing_constraints.keys())
                if missing_faults:
                    if verbose:
                        print(f"Warning: Smoothing constraints not specified for faults: {missing_faults}")
                        print("Using default constraints (None, None, None, None) for these faults.")
                    for fault_name in missing_faults:
                        smoothing_constraints[fault_name] = (None, None, None, None)
            else:
                raise ValueError("smoothing_constraints should be a 4-tuple or a dictionary with fault names as keys and 4-tuples as values.")
    
        # Prepare smoothing matrix if using custom constraints
        if smoothing_constraints is not None:
            if verbose:
                print("Using custom smoothing constraints...")
                for fault_name, constraints in smoothing_constraints.items():
                    if all(c is not None for c in constraints):
                        print(f"  {fault_name}: top={constraints[0]}, bottom={constraints[1]}, left={constraints[2]}, right={constraints[3]} km")
                    else:
                        print(f"  {fault_name}: using default constraints")
    
            vce_result = self.simple_vce(
                smoothing_matrix=None,
                smoothing_constraints=smoothing_constraints,
                method='mudpy',
                verbose=verbose,
                max_iter=max_iter,
                tol=tol,
                des_enabled=des_enabled,
                sigma_mode=sigma_mode,
                sigma_groups=sigma_groups,
                sigma_update=sigma_update,
                sigma_values=sigma_values,
                smooth_mode=smooth_mode,
                smooth_groups=smooth_groups,
                smooth_update=smooth_update,
                smooth_values=smooth_values
            )
        else:
            if verbose:
                print("Using default Laplacian smoothing matrix...")
    
            self.combine_GL_poly(penalty_weight=1.0)
    
            vce_result = self.simple_vce(
                smoothing_matrix=self.GL_combined_poly,
                smoothing_constraints=None,
                method='mudpy',
                verbose=verbose,
                max_iter=max_iter,
                tol=tol,
                des_enabled=des_enabled,
                sigma_mode=sigma_mode,
                sigma_groups=sigma_groups,
                sigma_update=sigma_update,
                sigma_values=sigma_values,
                smooth_mode=smooth_mode,
                smooth_groups=smooth_groups,
                smooth_update=smooth_update,
                smooth_values=smooth_values
            )
    
        self.distributem()

        # Post-process penalty weights (only meaningful when smoothing is enabled)
        if alpha_disabled:
            self.current_penalty_weight = np.zeros(len(self.faults))
        else:
            var_alpha = vce_result.get('var_alpha', None)
            if isinstance(var_alpha, dict):
                penalty_weight = np.array([1.0 / np.sqrt(v) for v in var_alpha.values()])
            else:
                penalty_weight = np.array([1.0 / np.sqrt(var_alpha)])
            self.current_penalty_weight = penalty_weight[self.config.alpha['fault_param_indices']]
    
        if verbose:
            print("\n" + "="*70)
            print("VCE Results Summary:")
            print(f"Converged: {vce_result['converged']}")
            print(f"Iterations: {vce_result['iterations']}")
    
            print("\nVariance Components:")
            if isinstance(vce_result['var_d'], dict):
                for group, var in vce_result['var_d'].items():
                    print(f"  Data variance [{group}]: {var:.6e}")
            else:
                print(f"  Data variance: {vce_result['var_d']:.6e}")
    
            if not alpha_disabled:
                if isinstance(vce_result['var_alpha'], dict):
                    for group, var in vce_result['var_alpha'].items():
                        print(f"  Regularization variance [{group}]: {var:.6e}")
                else:
                    print(f"  Regularization variance: {vce_result['var_alpha']:.6e}")
            else:
                print("  Regularization variance: N/A (smoothing disabled)")
    
            print("\nFinal Model Statistics:")
            self.returnModel(print_stat=True)
    
            print("="*70)

        return vce_result
    
    def simple_run_loop(self, penalty_weights=None, output_file='run_loop.dat', preferred_penalty_weight=None, rms_unit='m', verbose=True, equal_aspect=False):
        """
        Run the inversion for a range of penalty weights.
    
        Parameters:
        -----------
        penalty_weights : list, np.ndarray, optional
            Penalty weights to apply to the Green's functions. If None, the function will use the initial values from the configuration.
        output_file : str, optional
            Path to the output file. If None, the results will only be printed to the screen. Default is 'run_loop.dat'.
        preferred_penalty_weight : float, optional
            The preferred penalty weight to highlight in the plot. If None, no preferred point will be highlighted.
        rms_unit : str, optional
            The unit for RMS. Default is 'm'. If set to other values, RMS will be scaled accordingly.
        verbose : bool, optional
            Whether to print the results of the inversion. Default is True.
        equal_aspect : bool, optional
            If True, set equal aspect ratio for the plot. Default is False.
        """
        results = []
    
        # ---------------------------------Loop Penalty Weight---------------------------------------------#
        # penalty_weight = [1.0, 10.0, 30.0, 50.0, 80.0, 100.0, 125.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0]
        for ipenalty in penalty_weights:
            alpha = [np.log10(1.0/ipenalty)] * len(self.faults)
            self.run(penalty_weight=None, alpha=alpha, verbose=True)
            if verbose:
                self.returnModel(print_stat=True)

            # Calculate RMS and VR for the solution and print the results
            rms = np.sqrt(np.mean((np.dot(self.G, self.mpost) - self.d)**2))
            vr = (1 - np.sum((np.dot(self.G, self.mpost) - self.d)**2) / np.sum(self.d**2)) * 100
            self.combine_GL_poly()
            roughness_vec = np.dot(self.GL_combined_poly, self.mpost)
            roughness = np.sqrt(np.mean(roughness_vec**2)) if roughness_vec.size > 0 else 0.0
            result = {
                'Penalty_weight': ipenalty,
                'Roughness': roughness,
                'RMS': rms,
                'VR': vr
            }
            results.append(result)
            # # Format penalty weight with up to 4 decimals, removing trailing zeros but keeping at least 1 decimal
            # penalty_str = f'{ipenalty:.4f}'.rstrip('0')
            # if penalty_str.endswith('.'):
            #     penalty_str += '0'
            # output = f'Penalty_weight: {penalty_str}, Roughness: {roughness:.4f}, RMS: {rms:.4f}, VR: {vr:.2f}%'
            # print(output)
    
        # Convert results to DataFrame
        df = pd.DataFrame(results)
    
        # Save DataFrame to file if output_file is specified
        if output_file:
            df.to_csv(output_file, index=False)
    
        # Default plot
        self.plot_roughness_vs_rms(df, output_file='Roughness_vs_RMS.png', show=True, 
                                   preferred_penalty_weight=preferred_penalty_weight, rms_unit=rms_unit, equal_aspect=equal_aspect)

        return df
    
    def plot_roughness_vs_rms(self, df, output_file='Roughness_vs_RMS.png', show=True, preferred_penalty_weight=None, rms_unit='m', equal_aspect=False):
        """
        Plot Roughness vs RMS and save the plot to a file.
    
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the results with columns 'Roughness' and 'RMS'.
        output_file : str, optional
            Path to the output file. Default is 'Roughness_vs_RMS.png'.
        show : bool, optional
            Whether to display the plot. Default is True.
        preferred_penalty_weight : float, optional
            The preferred penalty weight to highlight in the plot. If None, no preferred point will be highlighted.
        rms_unit : str, optional
            The unit for RMS. Default is 'm'. If set to other values, RMS will be scaled accordingly.
        equal_aspect : bool, optional
            If True, set equal aspect ratio for the plot. Default is False.
        """
        # Scale RMS values if necessary
        rms_values = df.RMS.values
        if rms_unit != 'm':
            if rms_unit == 'cm':
                rms_values *= 100
            elif rms_unit == 'mm':
                rms_values *= 1000
            else:
                raise ValueError(f"Unsupported RMS unit: {rms_unit}")
    
        with sci_plot_style():
            plt.plot(df.Roughness.values[:], rms_values[:], marker='o', linestyle='-', label='L-Curve')
            
            # Highlight the preferred penalty weight point if specified
            if preferred_penalty_weight is not None:
                preferred_point = df[df.Penalty_weight == preferred_penalty_weight]
                if not preferred_point.empty:
                    plt.plot(preferred_point.Roughness.values, preferred_point.RMS.values, marker='o', c='#e54726', label='Preferred')
            
            plt.xlabel('Roughness')
            plt.ylabel(f'RMS ({rms_unit})')
            plt.legend()
            plt.grid(True)
            if equal_aspect:
                plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(output_file, dpi=600)
            if show:
                plt.show()
            else:
                plt.close()

    def reassemble_data(self, geodata=None, trifaults_list=None, verticals=None):
        """
        Assemble data for inversion.

        Parameters:
        - geodata: list
            List of geodetic data objects (e.g., InSAR, GPS, Optical).
        - trifaults_list: list
            List of fault objects.
        - verticals: list
            List of vertical data objects.
        """
        faults = self.faults if trifaults_list is None else trifaults_list
        geodata = self.config.geodata['data'] if geodata is None else geodata
        vertical = self.config.geodata['verticals'] if verticals is None else verticals
        for data, vert in zip(geodata, vertical):
            for ifault in faults:
                if data.dtype in ('insar', 'tsunami'):
                    ifault.d[data.name] = data.vel if data.dtype == 'insar' else data.d
                elif data.dtype in ('gps', 'multigps'):
                    ifault.d[data.name] = data.vel_enu[:, :data.obs_per_station].T.flatten()
                    ifault.d[data.name] = ifault.d[data.name][np.isfinite(ifault.d[data.name])]
                elif data.dtype == 'opticorr':
                    ifault.d[data.name] = np.hstack((data.east.T.flatten(), data.north.T.flatten()))
                    if vert:
                        ifault.d[data.name] = np.hstack((ifault.d[data.name], np.zeros_like(data.east.T.ravel())))
                elif data.dtype == 'leveling':
                    ifault.d[data.name] = data.vel
                elif data.dtype == 'crossfaultoffset':
                    ifault.d[data.name] = data.data_vector

        for ifault in faults:
            ifault.assembled(geodata)

        self.d = faults[0].dassembled

    def returnModel(self, mpost=None, print_stat=True):
        if mpost is not None:
            mpost_backup = copy.deepcopy(self.mpost)
            self.mpost = mpost
        self.distributem()
        if mpost is not None:
            self.mpost = mpost_backup
        
        # Calculate and print fit statistics
        if print_stat:
            self.calculate_and_print_fit_statistics()

        # Caluculate RMS and VR for the solution and print the results
        rms = np.sqrt(np.mean((np.dot(self.G, self.mpost) - self.d)**2))
        vr = (1 - np.sum((np.dot(self.G, self.mpost) - self.d)**2) / np.sum(self.d**2)) * 100
        self.combine_GL_poly()
        roughness_vec = np.dot(self.GL_combined_poly, self.mpost)
        roughness = np.sqrt(np.mean(roughness_vec**2)) if roughness_vec.size > 0 else 0.0
        self.combine_GL_poly(penalty_weight=self.current_penalty_weight)
        if print_stat:
            # Format penalty weight with up to 4 decimals, removing trailing zeros but keeping at least 1 decimal
            penalty_str = [f'{ipenalty:.4f}'.rstrip('0') for ipenalty in self.current_penalty_weight]
            penalty_str = [s + '0' if s.endswith('.') else s for s in penalty_str]
            penalty_str = ', '.join(penalty_str)
            output = f'Penalty_weight: {penalty_str}, Roughness: {roughness:.4f}, RMS: {rms:.4f}, VR: {vr:.2f}%'
            print(output)
        return roughness, rms, vr
    
    def calculate_and_print_fit_statistics(self):
        """
        Calculate and print fit statistics for all datasets.
        """
        # Call parent class method with 'BLSE' model
        super().calculate_and_print_fit_statistics(model='BLSE')
        

    def combine_GL_poly(self, GL_combined=None, penalty_weight=None):
        """
        Combine Green's functions (GL) with polynomial constraints and apply penalty weights.
    
        Parameters:
        -----------
        GL_combined : np.ndarray, optional
            Pre-combined Green's functions matrix. If None, the function will combine the Green's functions.
        penalty_weight : int, float, list, or np.ndarray, optional
            Penalty weights to apply to the Green's functions. If None, the function will use the initial values from the configuration.
    
        Returns:
        --------
        GL_combined_poly : np.ndarray
            Combined Green's functions matrix with polynomial constraints and applied penalty weights.
        """
        if penalty_weight is not None:
            if isinstance(penalty_weight, (int, float)):
                penalty_weight = np.ones(len(self.faults)) * penalty_weight
            elif isinstance(penalty_weight, (list, np.ndarray)):
                if len(penalty_weight) == 1:
                    # Single value in list, expand it
                    penalty_weight = np.ones(len(self.faults)) * penalty_weight[0]
                assert len(penalty_weight) == len(self.faults) or len(penalty_weight) == 1, "The length of penalty_weight should be equal to the number of faults or a single value."
            else:
                raise ValueError("penalty_weight should be a scalar or a list of scalars.")
        else:
            # When alpha is disabled, use uniform weight=1.0 (no regularization penalty)
            if not self.config.alpha_enabled:
                penalty_weight = np.ones(len(self.faults))
            else:
                alpha = np.array(self.config.alpha['initial_value'])
                fault_index = self.config.alpha['fault_param_indices']
                alpha = alpha[fault_index]
                assert len(alpha) == len(self.faults), "The length of alpha should be equal to the number of faults."
                if self.config.alpha['log_scaled']:
                    penalty_weight = 1.0 / np.power(10, alpha)
                else:
                    penalty_weight = 1.0 / alpha

        # When alpha is disabled, skip all GL blocks to prevent smoothing
        alpha_disabled = not self.config.alpha_enabled

        if GL_combined is None:
            GL_combined_poly = []
            for fault, ipenalty_weight in zip(self.faults, penalty_weight):
                has_gl = hasattr(fault, 'GL') and fault.GL is not None
                if has_gl and not alpha_disabled:
                    poly_positions = self.poly_positions.get(fault.name, (0, 0))
                    # Create a zero matrix with the correct size
                    combined = np.zeros((fault.GL.shape[0], fault.GL.shape[1] + poly_positions[1] - poly_positions[0]))
                    # Copy the values from the original matrix to the combined matrix at the correct positions
                    combined[:, :fault.GL.shape[1]] = fault.GL.toarray() * ipenalty_weight
                    GL_combined_poly.append(combined)
                else:
                    # Sources without smoothing (Pressure/Sbarbot): zero-row placeholder
                    slip_st, slip_end = self.slip_positions.get(fault.name, (0, 0))
                    poly_st, poly_end = self.poly_positions.get(fault.name, (0, 0))
                    n_params = (slip_end - slip_st) + (poly_end - poly_st)
                    if n_params > 0:
                        GL_combined_poly.append(np.zeros((0, n_params)))
            if GL_combined_poly:
                self.GL_combined_poly = scipy.linalg.block_diag(*GL_combined_poly)
            else:
                self.GL_combined_poly = np.zeros((0, 0))
    
        return self.GL_combined_poly

    def extract_and_plot_blse_results(self, rank=0, 
                                          plot_faults=True, plot_data=True,
                                          antisymmetric=True, res_use_data_norm=True, cmap='RdBu_r', azimuth=None, elevation=None,
                                          slip_cmap='cmc.roma_r', depth_range=None, z_ticks=None, 
                                          axis_shape=(1.0, 1.0, 0.6), 
                                          gps_title=True, sar_title=True, sar_cbaxis=[0.1, 0.15, 0.35, 0.04], # [0.15, 0.25, 0.25, 0.02],
                                          gps_figsize=None, sar_figsize='double', gps_scale=0.05, gps_legendscale=0.2,
                                          file_type='png',
                                          remove_direction_labels=False,
                                          fault_cbaxis=[0.15, 0.22, 0.15, 0.02], 
                                          data_poly=None,
                                          print_fit_statistics=True,
                                          print_fault_statistics=True
                                          ):
        """
        Extract and plot the Bayesian results.
    
        args:
        rank: process rank (default is 0)
        filename: name of the HDF5 file to save the samples (default is 'samples_mag_rake_multifaults.h5')
        plot_faults: whether to plot faults (default is True)
        plot_data: whether to plot data (default is True)
        antisymmetric: whether to set the colormap to be antisymmetric (default is True)
        res_use_data_norm: whether to make the norm of 'res' consistent with 'data' and 'synth' (default is True)
        cmap: colormap to use (default is 'RdBu_r')
        slip_cmap: colormap for slip (default is 'precip3_16lev_change.cpt')
        depth_range: depth range for the plot (default is None)
        z_ticks: z-axis ticks for the plot (default is None)
        gps_title: whether to show title for GPS data plots (default is True)
        sar_title: whether to show title for SAR data plots (default is True)
        sar_cbaxis: colorbar axis position for SAR data plots (default is [0.1, 0.15, 0.35, 0.04])
        gps_figsize: figure size for GPS data plots (default is None)
        sar_figsize: figure size for SAR data plots (default is (3.5, 2.7))
        gps_scale: scale for GPS data plots (default is 0.05)
        gps_legendscale: legend scale for GPS data plots (default is 0.2)
        file_type: file type to save the figures (default is 'png')
        remove_direction_labels : If True, remove E, N, S, W from axis labels (default is False)
        fault_cbaxis: colorbar axis position for fault plots (default is [0.15, 0.22, 0.15, 0.02])
        data_poly: whether to include polynomial constraints in the data (default is None), options are 'include' or None
        print_fit_statistics: whether to print fit statistics (default is True)
        print_fault_statistics: whether to print fault statistics (default is True)
        """
        if rank == 0:
            import cmcrameri
            from ..getcpt import get_cpt 
    
            if slip_cmap is not None and slip_cmap.endswith('.cpt'):
                # 'precip3_16lev_change.cpt'
                cmap_slip = get_cpt.get_cmap(slip_cmap, method='list', N=15)
            else:
                cmap_slip = slip_cmap
            if slip_cmap is None:
                cmap_slip = get_cpt.get_cmap('precip3_16lev_change.cpt', method='list', N=15)
            self.returnModel(print_stat=print_fit_statistics)
            if print_fault_statistics:
                self._print_fault_statistics()
    
            if plot_faults:
                self.plot_multifaults_slip(slip='total', cmap=cmap_slip,
                                                drawCoastlines=False, cblabel='Slip (m)',
                                                savefig=True, style=['notebook'], cbaxis=fault_cbaxis,
                                                xtickpad=5, ytickpad=5, ztickpad=5,
                                                xlabelpad=15, ylabelpad=15, zlabelpad=15,
                                                shape=axis_shape, elevation=elevation, azimuth=azimuth,
                                                depth=depth_range, zticks=z_ticks, fault_expand=0.0,
                                                plot_faultEdges=False, suffix='_slip', outdir='output', ftype=file_type,
                                                remove_direction_labels=remove_direction_labels,
                                                )

            # Build the synthetic data and plot the results
            faults = self.faults
            cogps_vertical_list = []
            cosar_list = []
            coleveling_list = []
            cocrossfault_list = []
            datas = self.config.geodata.get('data', [])
            verticals = self.config.geodata.get('verticals', [])
            for data, vertical in zip(datas, verticals):
                if data.dtype == 'gps':
                    cogps_vertical_list.append([data, vertical])
                elif data.dtype == 'insar':
                    cosar_list.append(data)
                elif data.dtype == 'leveling':
                    coleveling_list.append(data)
                elif data.dtype == 'crossfaultoffset':
                    cocrossfault_list.append(data)

            # Plot GPS data
            for fault in faults:
                if fault.lon is None or fault.lat is None:
                    fault.setTrace(0.1)
                fault.color = 'k' # Set the color to black
                fault.linewidth = 2.0 # Set the line width to 2.0
            for cogps, vertical in cogps_vertical_list:
                cogps.buildsynth(faults, vertical=vertical, poly=data_poly)
                if plot_data:
                    box = [cogps.lon.min(), cogps.lon.max(), cogps.lat.min(), cogps.lat.max()]
                    cogps.plot(faults=faults, drawCoastlines=True, data=['data', 'synth'], 
                                scale=gps_scale, legendscale=gps_legendscale, color=['#e33e1c', '#2e5b99'],
                                seacolor='lightblue', box=box, titleyoffset=1.02, title=gps_title, figsize=gps_figsize,
                                remove_direction_labels=remove_direction_labels)
                    cogps.fig.savefig(f'gps_{cogps.name}', ftype=file_type, dpi=600, 
                                    bbox_inches='tight', mapaxis=None, saveFig=['map'])
            # Plot SAR data
            for fault in faults:
                fault.color = 'k'
            for cosar in cosar_list:
                cosar.buildsynth(faults, vertical=True, poly=data_poly)
                if plot_data:
                    datamin, datamax = cosar.vel.min(), cosar.vel.max()
                    absmax = max(abs(datamin), abs(datamax))
                    data_norm = [-absmax, absmax] if antisymmetric else [datamin, datamax]
                    # for data in ['data', 'synth', 'res']:
                    #     if data == 'res':
                    #         cosar.res = cosar.vel - cosar.synth
                    #         absmax = max(abs(cosar.res.min()), abs(cosar.res.max()))
                    #         res_norm = [-absmax, absmax] if antisymmetric else [cosar.res.min(), cosar.res.max()]
                    #         res_norm = data_norm if res_use_data_norm else res_norm
                    #         cosar.plot(faults=faults, data=data, seacolor='lightblue', figsize=sar_figsize, norm=res_norm, cmap=cmap,
                    #             cbaxis=sar_cbaxis, drawCoastlines=True, titleyoffset=1.02, title=sar_title,
                    #             remove_direction_labels=remove_direction_labels)
                    #     else:
                    #         cosar.plot(faults=faults, data=data, seacolor='lightblue', figsize=sar_figsize, norm=data_norm, cmap=cmap,
                    #                 cbaxis=sar_cbaxis, drawCoastlines=True, titleyoffset=1.02, title=sar_title,
                    #                 remove_direction_labels=remove_direction_labels)
                    #     cosar.fig.savefig(f'sar_{cosar.name}_{data}', ftype=file_type, dpi=600, saveFig=['map'], 
                    #                     bbox_inches='tight', mapaxis=None)
                    
                    # Make directory for fit comparison plots
                    out_modeling_dir = pathlib.Path('Modeling')
                    out_modeling_dir.mkdir(parents=True, exist_ok=True)
                    cosar.plot_fit_comparison(
                                                faults=faults,
                                                cmap=cmap,
                                                vmin=data_norm[0],
                                                vmax=data_norm[1],
                                                share_colorbar=res_use_data_norm,
                                                cbaxis=sar_cbaxis,
                                                save_path=out_modeling_dir / f'{cosar.name}_fit_comparison.pdf',
                                                figsize=sar_figsize,
                                                show=True
                                            )

            # Build synthetics and save/plot leveling data
            for colev in coleveling_list:
                colev.buildsynth(faults, vertical=True, poly=data_poly)
            if plot_data and coleveling_list:
                out_modeling_dir = pathlib.Path('Modeling')
                out_modeling_dir.mkdir(parents=True, exist_ok=True)
                for colev in coleveling_list:
                    for itype in ['data', 'synth']:
                        colev.write2file(f'{colev.name}_{itype}.txt', outDir=str(out_modeling_dir), data=itype)
                    _plot_leveling_fit(colev, save_dir=out_modeling_dir, file_type=file_type)
            
            # Build synthetics and save/plot cross-fault offset data
            for cocf in cocrossfault_list:
                cocf.buildsynth(faults, poly=data_poly)
            if plot_data and cocrossfault_list:
                out_modeling_dir = pathlib.Path('Modeling')
                out_modeling_dir.mkdir(parents=True, exist_ok=True)
                for cocf in cocrossfault_list:
                    for itype in ['data', 'synth']:
                        cocf.write2file(f'{cocf.name}_{itype}.txt', outDir=str(out_modeling_dir), data=itype)
                    _plot_crossfaultoffset_fit(cocf, save_dir=out_modeling_dir, file_type=file_type)


#EOF
