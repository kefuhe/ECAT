import scipy
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd

from .bayesian_multifaults_inversion import MyMultiFaultsInversion
from .bayesian_config import BoundLSEInversionConfig
from ..plottools import sci_plot_style

class BoundLSEMultiFaultsInversion(MyMultiFaultsInversion):
    def __init__(self, name, faults_list, geodata=None, config='default_config_BLSE.yml', encoding='utf-8',
                 gfmethods=None, bounds_config='bounds_config.yml', rake_limits=None,
                 extra_parameters=None, verbose=True):
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

        # Initialize MyMultiFaultsInversion
        super(BoundLSEMultiFaultsInversion, self).__init__(name, 
                                                           faults_list, 
                                                           extra_parameters=extra_parameters, 
                                                           verbose=verbose)

        self.assembleGFs() # assemble the Green's functions because the data is already assembled
        self.update_config(self.config)
        if self.config.use_bounds_constraints:
            self.set_bounds_from_config(bounds_config, encoding=encoding)
        if self.config.use_rake_angle_constraints:
            if rake_limits is not None:
                self.set_inequality_constraints_for_rake_angle(rake_limits)
            elif self.config.use_bounds_constraints:
                self.set_inequality_constraints_for_rake_angle(self.bounds_config['rake_angle']) 
            else:
                assert False, "Rake angle constraints can only be set when bounds constraints are used or rake_limits is provided."

    def update_config(self, config):
        self.config = config
        self._update_faults()

    def _update_faults(self):
        # Update the faults based on the configuration parameters and method parameters for each fault 
        for fault_name, fault_config in self.config.faults.items():
            if fault_name != 'default':
                # Update Green's functions
                self.update_GFs(fault_names=[fault_name], **fault_config['method_parameters']['update_GFs'])
                # Update Laplacian
                self.update_Laplacian(fault_names=[fault_name], **fault_config['method_parameters']['update_Laplacian'])
    
    def run(self, penalty_weight=None, smoothing_constraints=None, data_weight=None, data_log_scaled=None, 
            penalty_log_scaled=None, sigma=None, alpha=None, verbose=True):
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
    
        Returns:
        --------
        None
        """
        from .config_utils import parse_initial_values

        # Ensure data_weight and sigma are either both None or only one is provided
        if (data_weight is not None) and (sigma is not None):
            raise ValueError("data_weight and sigma must either both be None or only one is provided.")
    
        # Ensure penalty_weight and alpha are either both None or only one is provided
        if (penalty_weight is not None) and (alpha is not None):
            raise ValueError("penalty_weight and alpha must either both be None or only one is provided.")
    
        # Handle data weights
        if data_weight is None:
            if sigma is None:
                sigma = self.config.sigmas['initial_value']
            sigma = np.array(sigma)
            if data_log_scaled is None:
                data_log_scaled = self.config.sigmas['log_scaled']
            if data_log_scaled:
                sigma = np.power(10, sigma)
            data_weight = 1.0 / sigma

            # if data_log_scaled:
            #     print(f'Using formula: data_weight = 1.0 / 10^sigma, with sigma = {sigma}')
            # else:
            #     print(f'Using formula: data_weight = 1.0 / sigma, with sigma = {sigma}')
        else:
            n_datasets = len(self.config.geodata.get('data', []))
            data_names = [d.name for d in self.config.geodata.get('data', [])]

            wgt_dict = {'initial_value': data_weight}
            data_weight = parse_initial_values(wgt_dict, n_datasets=n_datasets,
                                                param_name='initial_value',  # initial_value or 'values'
                                                dataset_names=data_names,
                                                print_name='data_weight')
            data_weight = np.array(data_weight)
            # print(f"Parsed data_weight: {data_weight}")
    
        # Handle penalty weights
        if penalty_weight is None:
            if alpha is None:
                alpha = self.config.alpha['initial_value']
                # print('alpha is from config:', alpha)
            else:
                n_faults = len(self.faults)
                fault_names = [fault.name for fault in self.faults]
                alpha = parse_initial_values({'initial_value': alpha},
                                                n_datasets=n_faults,
                                                param_name='initial_value',  # initial_value or 'values'
                                                dataset_names=fault_names,
                                                print_name='alpha')
                # print('alpha is from input:', alpha)
            alpha = np.array(alpha)
            fault_index = self.config.alpha['faults_index']
            alpha = alpha[fault_index]
            if penalty_log_scaled is None:
                penalty_log_scaled = self.config.alpha['log_scaled']
            if penalty_log_scaled:
                penalty_weight = 1.0 / np.power(10, alpha)
            else:
                penalty_weight = 1.0 / alpha
            # if penalty_log_scaled:
            #     print(f'Using formula: penalty_weight = 1.0 / 10^alpha, with alpha = {alpha} and fault_index = {fault_index}')
            # else:
            #     print(f'Using formula: penalty_weight = 1.0 / alpha, with alpha = {alpha} and fault_index = {fault_index}')
        else:
            n_faults = len(self.faults)
            fault_names = [fault.name for fault in self.faults]
            penalty_weight = parse_initial_values({'initial_value': penalty_weight},
                                                  n_datasets=n_faults,
                                                  param_name='initial_value',  # initial_value or 'values'
                                                  dataset_names=fault_names,
                                                  print_name='penalty_weight')
            fault_index = self.config.alpha['faults_index']
            penalty_weight = np.array(penalty_weight)
            penalty_weight = penalty_weight[fault_index]
            # print(f"Parsed penalty_weight: {penalty_weight} with fault_index: {fault_index}")
    
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
                                            verbose=True)
        else:
            self.combine_GL_poly(penalty_weight=penalty_weight)
            self.ConstrainedLeastSquareSoln(penalty_weight=penalty_weight, 
                                            smoothing_matrix=self.GL_combined_poly,
                                            data_weight=data_weight,
                                            verbose=True)
        self.distributem()

        # if verbose:
        #     # Caluculate RMS and VR for the solution and print the results
        #     rms = np.sqrt(np.mean((np.dot(self.G, self.mpost) - self.d)**2))
        #     vr = (1 - np.sum((np.dot(self.G, self.mpost) - self.d)**2) / np.sum(self.d**2)) * 100
        #     self.combine_GL_poly()
        #     roughness = np.dot(self.GL_combined_poly, self.mpost)
        #     roughness = np.sqrt(np.mean(roughness**2))
        #     self.returnModel()
        #     print(f'Roughness: {roughness:.4f}, RMS: {rms:.4f}, VR: {vr:.2f}%')
        #     # self._print_fault_statistics()
        #     self.combine_GL_poly(penalty_weight=penalty_weight)
    
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

    def calculate_data_fit_metrics(self, data, vertical=True):
        """
        Calculate RMS and VR for different data types.
        
        Parameters:
        -----------
        data : csi data object
            GPS, InSAR, or optical correlation data object
        vertical : bool
            Whether to include vertical component (for GPS data)
            
        Returns:
        --------
        tuple : (rms, vr)
            Root Mean Square error and Variance Reduction percentage
        """
        if data.dtype == 'insar':
            observed = data.vel
            synthetic = data.synth
        elif data.dtype == 'gps':
            if vertical:
                observed = data.vel_enu.flatten()  # Flatten all components
                synthetic = data.synth.flatten()
            else:
                observed = data.vel_enu[:, :-1].flatten()  # Only E-N components
                synthetic = data.synth[:, :-1].flatten()
        elif data.dtype in ('opticorr', 'optical'):
            observed = np.hstack((data.east, data.north))
            synthetic = np.hstack((data.east_synth, data.north_synth))
        else:
            raise ValueError(f"Unsupported data type: {data.dtype}")
        
        # Calculate RMS
        residuals = synthetic - observed
        rms = np.sqrt(np.mean(residuals**2))
        
        # Calculate Variance Reduction
        ss_res = np.sum(residuals**2)  # Sum of squares of residuals
        ss_tot = np.sum(observed**2)   # Total sum of squares
        vr = (1 - ss_res / ss_tot) * 100 if ss_tot != 0 else 0.0
        
        return rms, vr
    
    def calculate_and_print_fit_statistics(self):
        """
        Calculate and print fit statistics for all datasets.
        """
        
        print("\n" + "="*70)
        print(f"Data Fit Statistics (BLSE model)")
        print("="*70)
        
        # Build synthetics and calculate statistics for each dataset
        for idata, ivert, ipoly in zip(self.config.geodata['data'], self.config.geodata['verticals'], self.config.geodata['polys']):
            ipoly = ipoly if ipoly is None else 'include'
            idata.buildsynth(self.faults, direction='sd', poly=ipoly, vertical=ivert)
            # Calculate RMS and VR using the helper method
            rms, vr = self.calculate_data_fit_metrics(idata, ivert)
            print(f"{idata.name:<15} | RMS: {rms:8.4f} | VR: {vr:6.2f}%")

        print("="*70)

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
                fault.color = 'b' # Set the color to blue
            for cogps, vertical in cogps_vertical_list:
                cogps.buildsynth(faults, vertical=vertical, poly=data_poly)
                if plot_data:
                    box = [cogps.lon.min(), cogps.lon.max(), cogps.lat.min(), cogps.lat.max()]
                    cogps.plot(faults=faults, drawCoastlines=True, data=['data', 'synth'], 
                                scale=gps_scale, legendscale=gps_legendscale, color=['k', 'r'],
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

#EOF