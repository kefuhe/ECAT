import scipy
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd

from .multifaults_base import MyMultiFaultsInversion
from .config.blse_config import BoundLSEInversionConfig
from .constraint_manager_blse import ConstraintManager
from .euler_inequality_constraints import (
    apply_euler_inequality_constraints,
    validate_euler_inequality_config)
from ..plottools import sci_plot_style

class BoundLSEMultiFaultsInversion(MyMultiFaultsInversion):
    def __init__(self, name, faults_list, geodata=None, config='default_config_BLSE.yml', encoding='utf-8',
                 gfmethods=None, bounds_config='bounds_config.yml', rake_limits=None,
                 extra_parameters=None, verbose=True, des_enabled=False, des_config=None):
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

    def _update_faults(self):
        # Update the faults based on the configuration parameters and method parameters for each fault 
        datanames = [d.name for d in self.config.geodata.get('data', [])]
        Nd = len(datanames)
        faultnames = self.faultnames
        for fault_name, fault_config in self.config.faults.items():
            if fault_name != 'default':
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
        n_faults = len(self.config.alpha['update'])
        if self.config.alpha['mode'] == 'single':
            fault_names = ['All_faults']
        elif self.config.alpha['mode'] == 'individual':
            fault_names = [fault.name for fault in self.faults]
        elif self.config.alpha['mode'] == 'grouped':
            fault_names = [f'Event_{i}' for i in range(n_faults)]
        fault_indices = self.config.alpha['faults_index']

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
        if verbose:
            print("="*70)
            print("Starting Simple VCE for Multi-Fault Inversion")
            print("Automatically determining optimal regularization weights...")
            print(f"Number of faults: {len(self.faults)}")
            print(f"Data variance mode: {sigma_mode}")
            print(f"Smoothing variance mode: {smooth_mode}")
            if des_enabled is None:
                des_enabled = getattr(self, 'des_enabled', False)
            print(f"DES enabled: {des_enabled}")
            print("="*70)
    
        # Ensure bounds are set
        if not hasattr(self, 'lb') or not hasattr(self, 'ub'):
            raise ValueError("Bounds must be set before running VCE. Use set_bounds_from_config() or set_bounds().")
    
        if hasattr(self, 'lb') and hasattr(self, 'ub'):
            if np.any(np.isnan(self.lb)) or np.any(np.isnan(self.ub)):
                raise ValueError("Some bounds are not set (NaN values found). Please set all bounds first.")
    
        # Handle smoothing constraints
        if smoothing_constraints is not None:
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
        penalty_weight = 1.0/np.sqrt(vce_result.get('var_alpha', None))
        if isinstance(penalty_weight, (float,)):
            penalty_weight = np.array([penalty_weight])
        else:
            penalty_weight = np.array(penalty_weight)
        self.current_penalty_weight = penalty_weight[self.config.alpha['faults_index']]
    
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
    
            if isinstance(vce_result['var_alpha'], dict):
                for group, var in vce_result['var_alpha'].items():
                    print(f"  Regularization variance [{group}]: {var:.6e}")
            else:
                print(f"  Regularization variance: {vce_result['var_alpha']:.6e}")
    
            # print(f"\nEffective penalty weights: {[f'{w:.4e}' for w in self.vce_penalty_weights]}")
    
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
            roughness = np.dot(self.GL_combined_poly, self.mpost)
            roughness = np.sqrt(np.mean(roughness**2))
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
        roughness = np.dot(self.GL_combined_poly, self.mpost)
        roughness = np.sqrt(np.mean(roughness**2))
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
            alpha = np.array(self.config.alpha['initial_value'])
            fault_index = self.config.alpha['faults_index']
            alpha = alpha[fault_index]
            assert len(alpha) == len(self.faults), "The length of alpha should be equal to the number of faults."
            if self.config.alpha['log_scaled']:
                penalty_weight = 1.0 / np.power(10, alpha)
            else:
                penalty_weight = 1.0 / alpha

        if GL_combined is None:
            GL_combined_poly = []
            for fault, ipenalty_weight in zip(self.faults, penalty_weight):
                poly_positions = self.poly_positions.get(fault.name, (0, 0))
                # Create a zero matrix with the correct size
                combined = np.zeros((fault.GL.shape[0], fault.GL.shape[1] + poly_positions[1] - poly_positions[0]))
                # Copy the values from the original matrix to the combined matrix at the correct positions
                combined[:, :fault.GL.shape[1]] = fault.GL.toarray() * ipenalty_weight
                GL_combined_poly.append(combined)
            self.GL_combined_poly = scipy.linalg.block_diag(*GL_combined_poly)
    
        return self.GL_combined_poly

    def extract_and_plot_blse_results(self, rank=0, 
                                          plot_faults=True, plot_data=True,
                                          antisymmetric=True, res_use_data_norm=True, cmap='jet', azimuth=None, elevation=None,
                                          slip_cmap='precip3_16lev_change.cpt', depth_range=None, z_ticks=None, 
                                          axis_shape=(1.0, 1.0, 0.6), 
                                          gps_title=True, sar_title=True, sar_cbaxis=[0.15, 0.25, 0.25, 0.02],
                                          gps_figsize=None, sar_figsize=(3.5, 2.7), gps_scale=0.05, gps_legendscale=0.2,
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
        cmap: colormap to use (default is 'jet')
        slip_cmap: colormap for slip (default is 'precip3_16lev_change.cpt')
        depth_range: depth range for the plot (default is None)
        z_ticks: z-axis ticks for the plot (default is None)
        gps_title: whether to show title for GPS data plots (default is True)
        sar_title: whether to show title for SAR data plots (default is True)
        sar_cbaxis: colorbar axis position for SAR data plots (default is [0.15, 0.25, 0.25, 0.02])
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
                                                plot_faultEdges=False, suffix='_slip', ftype=file_type,
                                                remove_direction_labels=remove_direction_labels,
                                                )

            # Build the synthetic data and plot the results
            faults = self.faults
            cogps_vertical_list = []
            cosar_list = []
            datas = self.config.geodata.get('data', [])
            verticals = self.config.geodata.get('verticals', [])
            for data, vertical in zip(datas, verticals):
                if data.dtype == 'gps':
                    cogps_vertical_list.append([data, vertical])
                elif data.dtype == 'insar':
                    cosar_list.append(data)

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
                    for data in ['data', 'synth', 'res']:
                        if data == 'res':
                            cosar.res = cosar.vel - cosar.synth
                            absmax = max(abs(cosar.res.min()), abs(cosar.res.max()))
                            res_norm = [-absmax, absmax] if antisymmetric else [cosar.res.min(), cosar.res.max()]
                            res_norm = data_norm if res_use_data_norm else res_norm
                            cosar.plot(faults=faults, data=data, seacolor='lightblue', figsize=sar_figsize, norm=res_norm, cmap=cmap,
                                cbaxis=sar_cbaxis, drawCoastlines=True, titleyoffset=1.02, title=sar_title,
                                remove_direction_labels=remove_direction_labels)
                        else:
                            cosar.plot(faults=faults, data=data, seacolor='lightblue', figsize=sar_figsize, norm=data_norm, cmap=cmap,
                                    cbaxis=sar_cbaxis, drawCoastlines=True, titleyoffset=1.02, title=sar_title,
                                    remove_direction_labels=remove_direction_labels)
                        cosar.fig.savefig(f'sar_{cosar.name}_{data}', ftype=file_type, dpi=600, saveFig=['map'], 
                                        bbox_inches='tight', mapaxis=None)

    def calculate_tectonic_loading_rate(self, fault_name, euler_params1=None, euler_params2=None):
        """
        Calculate tectonic loading rate on fault due to Euler motion difference (strike-slip component).
        
        Parameters:
        -----------
        fault_name : str
            Name of the fault
        euler_params1 : np.ndarray or list, optional
            First block's Euler parameters [wx, wy, wz] (rad/year)
            If None, will extract from inversion results based on config
        euler_params2 : np.ndarray or list, optional
            Second block's Euler parameters [wx, wy, wz] (rad/year)
            If None, will extract from inversion results based on config
            
        Returns:
        --------
        loading_rate : np.ndarray
            Long-term tectonic loading rate (m/year)
        """
        
        # Get fault object
        fault = None
        for f in self.faults:
            if f.name == fault_name:
                fault = f
                break
        
        if fault is None:
            raise ValueError(f"Fault '{fault_name}' not found in solver")
        
        # Get Euler configuration
        euler_config = getattr(self.config, 'euler_constraints', {})
        if not euler_config.get('enabled', False):
            raise ValueError("Euler constraints are not enabled in configuration")
        
        if fault_name not in euler_config['faults']:
            raise ValueError(f"Fault '{fault_name}' not found in euler_config")
        
        params = euler_config['faults'][fault_name]
        
        # Extract Euler parameters if not provided
        if euler_params1 is None or euler_params2 is None:
            if not hasattr(self, 'mpost') or self.mpost is None:
                raise ValueError("No inversion results found and no Euler parameters provided. "
                               "Run inversion first or provide euler_params1 and euler_params2.")
            
            # Get transform indices
            transform_indices = {}
            if hasattr(self.faults[0], 'transform_indices'):
                transform_indices = self.faults[0].transform_indices
            elif hasattr(self, 'transform_indices'):
                transform_indices = self.transform_indices
            else:
                raise ValueError("No transform_indices found for extracting Euler parameters")
            
            block_types = params['block_types']
            blocks_standard = params['blocks_standard']
            
            extracted_params = []
            for block_idx, (block_type, block_data) in enumerate(zip(block_types, blocks_standard)):
                if block_type == 'dataset':
                    # Extract from inversion results
                    dataset_name = block_data
                    if dataset_name not in transform_indices:
                        raise ValueError(f"Dataset '{dataset_name}' not found in transform_indices")
                    
                    euler_indices = transform_indices[dataset_name].get('eulerrotation')
                    if euler_indices is None:
                        raise ValueError(f"No Euler parameters found for dataset '{dataset_name}'")
                    
                    start_idx, end_idx = euler_indices
                    if end_idx - start_idx != 3:
                        raise ValueError(f"Expected 3 Euler parameters for dataset '{dataset_name}', "
                                       f"got {end_idx - start_idx}")
                    
                    euler_params = self.mpost[start_idx:end_idx]
                    extracted_params.append(euler_params)
                    
                elif block_type == 'euler_pole':
                    # Convert Euler pole to vector
                    from .euler_inequality_constraints import convert_euler_pole_to_vector
                    lon_pole, lat_pole, omega = block_data
                    euler_vector = convert_euler_pole_to_vector(lat_pole, lon_pole, omega)
                    extracted_params.append(euler_vector)
                    
                elif block_type == 'euler_vector':
                    # Use directly
                    extracted_params.append(np.array(block_data))
                
                else:
                    raise ValueError(f"Unknown block_type: {block_type}")
            
            if len(extracted_params) != 2:
                raise ValueError(f"Expected exactly 2 blocks for fault '{fault_name}', "
                               f"got {len(extracted_params)}")
            
            if euler_params1 is None:
                euler_params1 = extracted_params[0]
            if euler_params2 is None:
                euler_params2 = extracted_params[1]
        
        # Get patch indices to apply constraints
        apply_patches = None # params.get('apply_to_patches', None)
        if apply_patches is None:
            patch_indices = list(range(len(fault.patch)))
        else:
            patch_indices = apply_patches
        
        # Get patch centers
        centers = np.array(fault.getcenters())[patch_indices]
        xc, yc = centers[:, 0], centers[:, 1]
        lonc, latc = fault.xy2ll(xc, yc)
        lonc, latc = np.radians(lonc), np.radians(latc)
        # Calculate Euler matrix
        from .euler_inequality_constraints import calculate_euler_matrix_for_points, project_euler_to_strike
        euler_mat = calculate_euler_matrix_for_points(lonc, latc)
        
        # Get reference strike
        reference_strike_deg = params.get('reference_strike', 0.0)
        
        # Calculate strike-slip projection matrix
        euler_strike = project_euler_to_strike(euler_mat, fault, patch_indices, 
                                             reference_strike_deg, len(patch_indices))
        
        # Convert to numpy arrays
        euler_params1 = np.array(euler_params1)
        euler_params2 = np.array(euler_params2)
        
        # Calculate velocity for each block
        vel1 = np.sum(euler_strike * euler_params1[None, :], axis=1)
        vel2 = np.sum(euler_strike * euler_params2[None, :], axis=1)
        
        # Calculate loading rate: block1 - block2
        loading_rate_selected = vel1 - vel2
        
        # Extend to all patches (unconstrained patches set to 0)
        loading_rate = np.zeros(len(fault.patch))
        loading_rate[patch_indices] = loading_rate_selected
        
        # Store in fault attributes (optional)
        if not hasattr(fault, 'tectonic_loading_rate'):
            fault.tectonic_loading_rate = loading_rate
        else:
            fault.tectonic_loading_rate = loading_rate
        
        return loading_rate
    
    
    def calculate_locking_degree(self, fault_name, euler_params1=None, euler_params2=None, 
                               method='absolute', slip_component='strikeslip'):
        """
        Calculate fault locking degree.
        
        Parameters:
        -----------
        fault_name : str
            Name of the fault
        euler_params1, euler_params2 : np.ndarray or list, optional
            Two blocks' Euler parameters [wx, wy, wz] (rad/year)
            If None, will extract from inversion results based on config
        method : str
            Calculation method: 'absolute' or 'relative'
            - 'absolute': long-term rate + inverted slip
            - 'relative': (long-term rate + inverted slip) / long-term rate
        slip_component : str
            Slip component: 'strikeslip', 'dipslip', or 'total'
            
        Returns:
        --------
        locking_degree : np.ndarray
            Locking degree
        """
        
        # Get fault object
        fault = None
        fault_idx = None
        for i, f in enumerate(self.faults):
            if f.name == fault_name:
                fault = f
                fault_idx = i
                break
        
        if fault is None:
            raise ValueError(f"Fault '{fault_name}' not found in solver")
        
        # Calculate long-term tectonic loading rate
        loading_rate = self.calculate_tectonic_loading_rate(fault_name, euler_params1, euler_params2)
        
        # Get inverted slip
        if not hasattr(self, 'mpost') or self.mpost is None:
            raise ValueError("No inversion results found. Run inversion first.")
        
        # Get fault parameter indices
        fault_start, fault_end = self.fault_indexes[fault_name]
        fault_params = self.mpost[fault_start:fault_end]
        npatches = len(fault.patch)
        # Extract slip component
        if slip_component == 'strikeslip':
            slip_index = 0
        elif slip_component == 'dipslip':
            slip_index = npatches
        elif slip_component == 'total':
            slip_index = None
        else:
            raise ValueError(f"Invalid slip_component: {slip_component}")
        
        if slip_index is not None:
            # Single slip component
            inverted_slip = fault_params[slip_index:slip_index+npatches]
        else:
            # Total slip magnitude
            ss_slip = fault_params[0:npatches]
            ds_slip = fault_params[npatches:npatches+npatches]
            inverted_slip = np.sqrt(ss_slip**2 + ds_slip**2)
            # For total slip, use absolute loading rate as reference
            # abs_loading_rate = np.abs(loading_rate)
        
        # Ensure array lengths match
        if len(inverted_slip) != len(loading_rate):
            raise ValueError(f"Slip array length ({len(inverted_slip)}) doesn't match "
                            f"loading rate array length ({len(loading_rate)})")
        
        # Calculate locking degree
        if method == 'absolute':
            # Absolute locking degree: long-term rate + inverted slip (note: inverted slip is usually negative)
            locking_degree = loading_rate + inverted_slip
            
        elif method == 'relative':
            # Relative locking degree: (long-term rate + inverted slip) / long-term rate
            # Avoid division by zero
            loading_rate_safe = np.where(np.abs(loading_rate) < 1e-12, 1e-12, loading_rate)
            locking_degree = np.abs(loading_rate + inverted_slip) / np.abs(loading_rate_safe)
            
            # Handle special case: when long-term rate is zero, set locking degree to 0
            locking_degree = np.where(np.abs(loading_rate) < 1e-12, 0.0, locking_degree)
            
        else:
            raise ValueError(f"Invalid method: {method}. Use 'absolute' or 'relative'")
        
        # Store results in fault attributes
        fault.locking_degree = locking_degree
        
        return locking_degree
    
    
    def analyze_fault_kinematics(self, fault_name, euler_params1=None, euler_params2=None,
                                slip_component='strikeslip', save_results=True):
        """
        Comprehensive analysis of fault kinematics: long-term loading, inverted slip, locking degree.
        
        Parameters:
        -----------
        fault_name : str
            Name of the fault
        euler_params1, euler_params2 : np.ndarray or list, optional
            Two blocks' Euler parameters
            If None, will extract from inversion results based on config
        slip_component : str
            Slip component to analyze
        save_results : bool
            Whether to save results to fault attributes
            
        Returns:
        --------
        analysis : dict
            Analysis results dictionary
        """
        
        # Calculate long-term tectonic loading rate
        loading_rate = self.calculate_tectonic_loading_rate(fault_name, euler_params1, euler_params2)
        
        # Calculate two types of locking degree
        locking_abs = self.calculate_locking_degree(fault_name, euler_params1, euler_params2,
                                                  method='absolute', slip_component=slip_component)
        
        locking_rel = self.calculate_locking_degree(fault_name, euler_params1, euler_params2, 
                                                  method='relative', slip_component=slip_component)
        
        # Get inverted slip
        fault_start, fault_end = self.fault_indexes[fault_name]
        fault_params = self.mpost[fault_start:fault_end]
        
        fault = None
        for f in self.faults:
            if f.name == fault_name:
                fault = f
                break
        
        npatches = len(fault.patch)
        if slip_component == 'strikeslip':
            inverted_slip = fault_params[0:npatches]
        elif slip_component == 'dipslip':
            inverted_slip = fault_params[npatches:npatches+npatches]
        else:
            ss_slip = fault_params[0:npatches]
            ds_slip = fault_params[npatches:npatches+npatches]
            inverted_slip = np.sqrt(ss_slip**2 + ds_slip**2)
        
        # Compile analysis results
        analysis = {
            'fault_name': fault_name,
            'slip_component': slip_component,
            'num_patches': len(fault.patch),
            'loading_rate': {
                'values': loading_rate,
                'stats': {
                    'min': np.min(loading_rate),
                    'max': np.max(loading_rate), 
                    'mean': np.mean(loading_rate),
                    'std': np.std(loading_rate),
                    'median': np.median(loading_rate)
                }
            },
            'inverted_slip': {
                'values': inverted_slip,
                'stats': {
                    'min': np.min(inverted_slip),
                    'max': np.max(inverted_slip),
                    'mean': np.mean(inverted_slip),
                    'std': np.std(inverted_slip),
                    'median': np.median(inverted_slip)
                }
            },
            'locking_degree_absolute': {
                'values': locking_abs,
                'stats': {
                    'min': np.min(locking_abs),
                    'max': np.max(locking_abs),
                    'mean': np.mean(locking_abs),
                    'std': np.std(locking_abs),
                    'median': np.median(locking_abs)
                }
            },
            'locking_degree_relative': {
                'values': locking_rel,
                'stats': {
                    'min': np.min(locking_rel),
                    'max': np.max(locking_rel),
                    'mean': np.mean(locking_rel),
                    'std': np.std(locking_rel),
                    'median': np.median(locking_rel)
                }
            }
        }
        
        return analysis
    
    
    def plot_fault_kinematics(self, fault_name, euler_params1=None, euler_params2=None,
                             slip_component='strikeslip', plot_type='all', save_path=None):
        """
        Plot fault kinematics analysis results.
        
        Parameters:
        -----------
        fault_name : str
            Name of the fault
        euler_params1, euler_params2 : array-like, optional
            Euler parameters
            If None, will extract from inversion results based on config
        slip_component : str
            Slip component
        plot_type : str
            Plot type: 'loading', 'slip', 'locking_abs', 'locking_rel', 'all'
        save_path : str, optional
            Save path
        """
        
        import matplotlib.pyplot as plt
        
        # Get fault object
        fault = None
        for f in self.faults:
            if f.name == fault_name:
                fault = f
                break
        
        if plot_type == 'all':
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            # 1. Long-term tectonic loading rate
            loading_rate = self.calculate_tectonic_loading_rate(fault_name, euler_params1, euler_params2)
            # Temporarily assign for plotting
            original_slip = fault.slip.copy()
            fault.slip[:, 0] = loading_rate
            fault.plot(slip='strikeslip', ax=axes[0], colorbar=True)
            axes[0].set_title(f'Tectonic Loading Rate\n{fault_name}')
            
            # 2. Inverted slip
            fault.slip = original_slip  # Restore original slip
            fault.plot(slip=slip_component, ax=axes[1], colorbar=True)
            axes[1].set_title(f'Inverted {slip_component.title()}\n{fault_name}')
            
            # 3. Absolute locking degree
            locking_abs = self.calculate_locking_degree(fault_name, euler_params1, euler_params2,
                                                      method='absolute', slip_component=slip_component)
            fault.slip[:, 0] = locking_abs
            fault.plot(slip='strikeslip', ax=axes[2], colorbar=True)
            axes[2].set_title(f'Absolute Locking Degree\n{fault_name}')
            
            # 4. Relative locking degree  
            locking_rel = self.calculate_locking_degree(fault_name, euler_params1, euler_params2,
                                                      method='relative', slip_component=slip_component)
            fault.slip[:, 0] = locking_rel
            fault.plot(slip='strikeslip', ax=axes[3], colorbar=True)
            axes[3].set_title(f'Relative Locking Degree\n{fault_name}')
            
            # Restore original slip
            fault.slip = original_slip
            
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            original_slip = fault.slip.copy()
            
            if plot_type == 'loading':
                loading_rate = self.calculate_tectonic_loading_rate(fault_name, euler_params1, euler_params2)
                fault.slip[:, 0] = loading_rate
                fault.plot(slip='strikeslip', ax=ax, colorbar=True)
                ax.set_title(f'Tectonic Loading Rate - {fault_name}')
                
            elif plot_type == 'slip':
                fault.plot(slip=slip_component, ax=ax, colorbar=True)
                ax.set_title(f'Inverted {slip_component.title()} - {fault_name}')
                
            elif plot_type == 'locking_abs':
                locking_abs = self.calculate_locking_degree(fault_name, euler_params1, euler_params2,
                                                          method='absolute', slip_component=slip_component)
                fault.slip[:, 0] = locking_abs
                fault.plot(slip='strikeslip', ax=ax, colorbar=True)
                ax.set_title(f'Absolute Locking Degree - {fault_name}')
                
            elif plot_type == 'locking_rel':
                locking_rel = self.calculate_locking_degree(fault_name, euler_params1, euler_params2,
                                                          method='relative', slip_component=slip_component)
                fault.slip[:, 0] = locking_rel
                fault.plot(slip='strikeslip', ax=ax, colorbar=True)
                ax.set_title(f'Relative Locking Degree - {fault_name}')
            
            fault.slip = original_slip
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return fig

#EOF