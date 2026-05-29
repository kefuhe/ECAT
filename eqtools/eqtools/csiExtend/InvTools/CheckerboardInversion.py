import numpy as np
import copy
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from typing import Union, Dict, List, Optional, Any
from pathlib import Path

# CSI / EqTools Imports
from ..blse_multifaults_inversion import BoundLSEMultiFaultsInversion
from ...plottools import sci_plot_style, publication_figsize, set_degree_formatter

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DPI = 300
DEFAULT_GPS_SCALE = 0.1
DEFAULT_COLORMAP = 'viridis'
DEFAULT_SLIP_COLORMAP = 'cmc.roma_r'

class CheckerboardInversion(BoundLSEMultiFaultsInversion):
    """
    Specialized inversion class for checkerboard tests.
    Inherits from BoundLSEMultiFaultsInversion.
    
    Main features:
    1. Fix potential missing self.geodata in the base class.
    2. Provide explicit single fault/multi-fault checkerboard pattern setting interface.
    3. Automate the workflow: forward modeling -> add noise (supports differential settings) 
       -> replace observation data -> update global d vector.
    4. Provide visualization interface for input models and synthetic data.
    """

    def __init__(self, name: str, faults: List[Any], data: List, 
                 verbose: bool = True, config: Optional[Any] = None, 
                 bounds_config: Optional[Any] = None) -> None:
        """Initialize CheckerboardInversion instance.
        
        Args:
            name: Name identifier for this inversion instance
            faults: List of fault objects
            data: Geodetic data (list format)
            verbose: Enable verbose logging
            config: Configuration object
            bounds_config: Bounds configuration object
        """
        # 1. Initialize base class
        super().__init__(name, faults, data, verbose=verbose, config=config, bounds_config=bounds_config)
        
        # 2. [Key fix] Ensure self.geodata exists
        if not hasattr(self, 'geodata') or self.geodata is None:
            if hasattr(self, 'config') and hasattr(self.config, 'geodata'):
                self.geodata = self.config.geodata.get('data', [])
            else:
                if isinstance(data, list):
                    self.geodata = data
                else:
                    self.geodata = []
        
        # Create mapping from name to data for convenient access by name
        self.geodata_dict = {d.name: d for d in self.geodata}
        self.true_slips: Dict[str, np.ndarray] = {}
        if self.verbose:
            logger.info(f"[{self.name}] Initialized for Checkerboard Test.")
            logger.info(f"    Faults: {self.faultnames}")
            logger.info(f"    Datasets: {list(self.geodata_dict.keys())}")

        # 3. Backup original observation data (deep copy to prevent modification)
        if hasattr(self, 'd'):
            self.d_obs_original = copy.deepcopy(self.d)
        
        self.geodata_original_vel = []
        for d in self.geodata:
            if d.dtype == 'opticorr':
                self.geodata_original_vel.append({
                    'east': d.east.copy() if d.east is not None else None,
                    'north': d.north.copy() if d.north is not None else None
                })
            elif d.dtype == 'crossfaultoffset':
                self.geodata_original_vel.append({
                    'fault_parallel': d.fault_parallel.copy() if d.fault_parallel is not None else None,
                    'fault_perpendicular': d.fault_perpendicular.copy() if d.fault_perpendicular is not None else None,
                    'fault_vertical': d.fault_vertical.copy() if d.fault_vertical is not None else None
                })
            else:
                self.geodata_original_vel.append(d.vel.copy() if hasattr(d, 'vel') else None)

    # =================================================================
    # 1. Checkerboard pattern setting interface
    # =================================================================
    def add_checkerboard_pattern(self, fault_name: str, **kwargs) -> None:
        """
        Set checkerboard slip distribution for a specified single fault.
        
        Args:
            fault_name: Fault name.
            **kwargs: Parameters to pass to fault.generate_checkboard_slip.
                      (Mu, horizontal_discretization, depth_ranges, rake_angle, etc.)
                      
        Raises:
            ValueError: If fault_name does not exist
            AttributeError: If fault doesn't support checkerboard generation
        """
        # Input validation
        if not isinstance(fault_name, str) or not fault_name.strip():
            logger.error("Invalid fault_name: must be a non-empty string")
            raise ValueError("fault_name must be a non-empty string")
            
        if fault_name not in self.faults_dict:
            logger.error(f"Fault '{fault_name}' does not exist. Available: {self.faultnames}")
            raise ValueError(f"Fault '{fault_name}' does not exist. Available: {self.faultnames}")
            
        target_fault = self.faults_dict[fault_name]
        
        if not hasattr(target_fault, 'generate_checkboard_slip'):
            logger.error(f"Fault '{fault_name}' does not support 'generate_checkboard_slip'")
            raise AttributeError(f"Fault '{fault_name}' does not support 'generate_checkboard_slip'.")

        if self.verbose:
            logger.info(f"--> Setting Checkerboard for '{fault_name}'")

        # Parameter names are already defined in the Fault class (Mu, depth_ranges, etc.)
        target_fault.generate_checkboard_slip(**kwargs)
        self.true_slips[fault_name] = target_fault.slip.copy() if hasattr(target_fault, 'slip') else None

        if hasattr(target_fault, 'slip'):
            max_slip = np.linalg.norm(target_fault.slip, axis=1).max()
            if self.verbose:
                logger.info(f"    [Success] Slip generated. Max slip: {max_slip:.4f} m")

    def set_multi_fault_checkerboard(self, patterns_config: Dict[str, Dict[str, Any]]) -> None:
        """Batch setting checkerboard patterns for multiple faults.
        
        Args:
            patterns_config: Dictionary mapping fault names to their parameter dicts.
                            Format: {'FaultName': {params...}}
        """
        logger.info(f"\n--> Batch setting checkerboard for {len(patterns_config)} faults...")
        for fname, params in patterns_config.items():
            self.add_checkerboard_pattern(fname, **params)

    # =================================================================
    # 2. Forward modeling and data injection (enhanced version)
    # =================================================================
    def _determine_noise_sigma(self, noise_sigma: Union[float, List[float], Dict[str, float]], 
                               data_index: int, data_name: str) -> float:
        """Determine noise level for specific dataset.
        
        Args:
            noise_sigma: Noise specification (float, list, or dict)
            data_index: Index of current dataset
            data_name: Name of current dataset
            
        Returns:
            Noise sigma value for this dataset
        """
        if isinstance(noise_sigma, dict):
            isigma = noise_sigma.get(data_name, 0.0)
            if self.verbose:
                if isigma == 0.0:
                    logger.warning(f"    No noise specified for '{data_name}' in dict; defaulting to 0.0")
            return isigma
        elif isinstance(noise_sigma, list):
            isigma = noise_sigma[data_index] if data_index < len(noise_sigma) else 0.0
            if self.verbose and data_index >= len(noise_sigma):
                logger.warning(f"    Noise list too short for '{data_name}'; defaulting to 0.0")
            return isigma
        else:
            return float(noise_sigma)
    
    def _get_vertical_setting(self, data_index: int) -> bool:
        """Determine whether to compute vertical component for dataset.
        
        Args:
            data_index: Index of current dataset
            
        Returns:
            Whether to use vertical component
        """
        if not (self.config and hasattr(self.config, 'geodata') and 'verticals' in self.config.geodata):
            return True
            
        verticals_conf = self.config.geodata['verticals']
        if isinstance(verticals_conf, list) and data_index < len(verticals_conf):
            return verticals_conf[data_index]
        elif isinstance(verticals_conf, bool):
            return verticals_conf
        return True
    
    def _inject_noise_and_update_weights(self, data: Any, sigma: float, update_weight: bool) -> None:
        """Add noise to synthetic data and optionally update weights.
        
        Args:
            data: Dataset object
            sigma: Noise standard deviation
            update_weight: Whether to update weight matrix
        """
        if sigma <= 0:
            return
            
        if self.verbose:
            logger.info(f"    Injecting noise sigma={sigma} into '{data.name}'")
        
        try:
            data.add_random_noise(sigma=sigma, data='synth')
            
            if update_weight:
                if hasattr(data, 'err'):
                    data.err[:] = sigma
                if hasattr(data, 'buildDiagCd'):
                    data.buildDiagCd()
        except Exception as e:
            logger.warning(f"Failed to add noise or update weights for '{data.name}': {e}")
    
    def _replace_observation_with_synthetic(self, data: Any) -> None:
        """Replace observation values with synthetic data.
        
        Args:
            data: Dataset object
        """
        if data.dtype == 'opticorr':
            if hasattr(data, 'synth_east'): 
                data.east = data.synth_east.copy()
            if hasattr(data, 'synth_north'): 
                data.north = data.synth_north.copy()
        elif data.dtype == 'crossfaultoffset':
            if hasattr(data, 'synth_parallel') and data.synth_parallel is not None:
                data.fault_parallel = data.synth_parallel.copy()
            if hasattr(data, 'synth_perpendicular') and data.synth_perpendicular is not None:
                data.fault_perpendicular = data.synth_perpendicular.copy()
            if hasattr(data, 'synth_vertical') and data.synth_vertical is not None:
                data.fault_vertical = data.synth_vertical.copy()
        else:
            if hasattr(data, 'synth'):
                data.vel = data.synth.copy()
            else:
                logger.warning(f"Dataset '{data.name}' has no 'synth' attribute")
    
    def apply_synthetics(self, noise_sigma: Union[float, List[float], Dict[str, float]] = 0.0, 
                        update_weight: bool = True, save_dir: Optional[str] = 'Modeling') -> None:
        """
        Generate synthetic data -> add noise -> replace observation values -> (optional) update weights 
        -> update d vector.
        
        Args:
            noise_sigma: Noise standard deviation.
                - float: Use same noise for all data.
                - list: Noise values in order of geodata.
                - dict: {'T012A': 0.005, 'GPS': 0.002} Specify by name.
            update_weight: Whether to update data weight matrix (Cd) based on new noise.
                          Recommended as True to ensure inversion weights match actual noise level.
            save_dir: Directory to save output. None to skip saving.
        """
        if self.verbose:
            logger.info(f"\n--> [Forward Modeling] Generating synthetics...")

        # Backup slip information before processing (moved outside loop - bug fix)
        faults_checkslip = [f.slip.copy() for f in self.faults if hasattr(f, 'slip')]

        for i, data in enumerate(self.geodata):
            try:
                # 1. Determine noise level for this data
                sigma = self._determine_noise_sigma(noise_sigma, i, data.name)

                # 2. Determine whether to compute vertical component
                use_vertical = self._get_vertical_setting(i)

                # 3. Forward modeling (G * m_true)
                # poly=None: Only compute tectonic deformation
                data.buildsynth(faults=self.faults, direction='sd', poly=None, vertical=use_vertical)
                
                # 4. Add noise and update weights
                self._inject_noise_and_update_weights(data, sigma, update_weight)
                
                # 5. Replace observation values with synthetic
                self._replace_observation_with_synthetic(data)
                
                # 6. Save files
                if save_dir:
                    self._save_data_to_file(data, save_dir, suffix='data')
                    
            except Exception as e:
                logger.error(f"Failed to process dataset '{data.name}': {e}")
                raise

        # Restore slip information to prevent inconsistency
        if hasattr(self, 'config') and hasattr(self.config, '_initialize_faults_and_assemble_data'):
            self.config._initialize_faults_and_assemble_data(faults_list=None, geodata=None)
            for f, slip in zip(self.faults, faults_checkslip):
                if hasattr(f, 'slip'):
                    f.slip = slip

        # 7. Update global d vector and Cd matrix (if update_weight=True)
        self._update_global_vectors(update_Cd=update_weight)

    def _update_global_vectors(self, update_Cd: bool = True) -> None:
        """Update d vector and Cd matrix in the inversion class.
        
        Args:
            update_Cd: Whether to update covariance matrix
        """
        try:
            # Call fault.assembled to reassemble d and Cd
            for fault in self.faults:
                fault.assembled(self.geodata, verbose=False)
                if update_Cd:
                    fault.assembleCd(self.geodata, verbose=False)

            # Update BoundLSE's own d (typically take from first fault)
            if len(self.faults) > 0:
                if hasattr(self.faults[0], 'dassembled'):
                    self.d = self.faults[0].dassembled
                else:
                    logger.warning("First fault has no 'dassembled' attribute")
                    
                if update_Cd and hasattr(self.faults[0], 'Cd'):
                    self.Cd = self.faults[0].Cd
            
            if self.verbose:
                logger.info(f"    [System] Global vectors updated. d shape: {self.d.shape}")
        except Exception as e:
            logger.error(f"Failed to update global vectors: {e}")
            raise

    def _save_data_to_file(self, data: Any, out_dir: str, suffix: str = 'synth') -> None:
        """Save dataset to file.
        
        Args:
            data: Dataset object to save
            out_dir: Output directory path
            suffix: Filename suffix
        """
        try:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            if data.dtype == 'opticorr':
                for idir in ['east', 'north']:
                    data.writeDecim2file(
                        f'{data.name}_{suffix}_{idir}.txt', 
                        f'data{idir}', 
                        outDir=str(out_path), 
                        triangular=True
                    )
            elif data.dtype in ('leveling', 'crossfaultoffset'):
                data.write2file(
                    f'{data.name}_{suffix}.txt',
                    outDir=str(out_path),
                    data='data'
                )
            else:
                data.writeDecim2file(
                    f'{data.name}_{suffix}.txt', 
                    'data', 
                    outDir=str(out_path)
                )
        except Exception as e:
            logger.warning(f"Failed to save data file for '{data.name}': {e}")

    def save_true_model(self, output_dir: str = 'output') -> None:
        """Save true checkerboard model to files.
        
        Args:
            output_dir: Directory to save output files
        """
        try:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            for fault in self.faults:
                try:
                    fault.writePatches2File(
                        str(out_path / f'checkerboard_truth_{fault.name}.gmt'), 
                        add_slip='total'
                    )
                    fault.writeSlipDirection2File(
                        filename=str(out_path / f'slipdir_truth_{fault.name}.txt'), 
                        scale='total'
                    )
                except Exception as e:
                    logger.warning(f"Failed to save true model for fault '{fault.name}': {e}")
        except Exception as e:
            logger.error(f"Failed to create output directory '{output_dir}': {e}")
            raise

    # =================================================================
    # 3. Visualization interface (new)
    # =================================================================
    def _plot_fault_slip(self, save_dir: Optional[Path]) -> None:
        """Plot true slip model.
        
        Args:
            save_dir: Directory to save figures (None to skip saving)
        """
        try:
            logger.info("    Plotting True Slip Model...")
            from eqtools.plottools import plot_slip_distribution
            
            figname = str(save_dir / "input_true_model") if save_dir else None
            plot_slip_distribution(
                self.faults, 
                cmap=DEFAULT_SLIP_COLORMAP, 
                plot_on_2d=False, 
                savefig=True, 
                figname=figname
            )
        except Exception as e:
            logger.error(f"Failed to plot fault slip: {e}")
    
    def _plot_single_dataset(self, data: Any, cmap: str, figsize: tuple, 
                            save_dir: Optional[Path], show: bool) -> None:
        """Plot a single dataset.
        
        Args:
            data: Dataset object to plot
            cmap: Colormap name
            figsize: Figure size tuple
            save_dir: Directory to save figures
            show: Whether to show the plot
        """
        try:
            # GPS data plotting vectors
            if data.dtype in ('gps', 'multigps'):
                data.plot(
                    faults=self.faults, 
                    data='data', 
                    figsize=figsize,
                    drawCoastlines=False, 
                    scale=DEFAULT_GPS_SCALE, 
                    remove_direction_labels=True
                )
            
            # InSAR data plotting layers
            elif data.dtype == 'insar':
                data.plot(
                    faults=self.faults, 
                    data='data', 
                    cbaxis=[0.15, 0.25, 0.25, 0.02],
                    titleyoffset=1.02,
                    figsize=figsize,
                    drawCoastlines=False, 
                    cmap=cmap, 
                    remove_direction_labels=True
                )
            
            # Optical data
            elif data.dtype == 'opticorr':
                # Opticorr is complex, typically has east/north two figures
                logger.info(f"    Skipping opticorr data '{data.name}' (not yet implemented)")
                return

            # Leveling data
            elif data.dtype == 'leveling':
                if hasattr(data, 'plot'):
                    data.plot(show=False)
                else:
                    logger.info(f"    Skipping leveling data '{data.name}' (no plot method)")
                    return

            # Cross-fault offset data
            elif data.dtype == 'crossfaultoffset':
                if hasattr(data, 'plot'):
                    data.plot(show=False)
                else:
                    logger.info(f"    Skipping crossfaultoffset data '{data.name}' (no plot method)")
                    return

            if save_dir and hasattr(data, 'fig'):
                figname = save_dir / f"input_data_{data.name}"
                data.fig.savefig(str(figname), dpi=DEFAULT_DPI, saveFig=['map'])
                logger.info(f"    Saved: {figname}")
            
            if not show:
                plt.close()
                
        except Exception as e:
            logger.warning(f"Failed to plot dataset '{data.name}': {e}")
            if not show:
                plt.close()
    
    def plot_inputs(self, plot_faults: bool = True, plot_data: bool = True, 
                    cmap: str = DEFAULT_COLORMAP, figsize: tuple = (10, 8), 
                    save_dir: Optional[str] = None, show: bool = True) -> None:
        """
        Plot input state before inversion:
        1. True checkerboard slip model (Ground Truth).
        2. Synthetic observation data (Synthetic Data).
        
        Args:
            plot_faults: Whether to plot fault slip.
            plot_data: Whether to plot synthetic data.
            cmap: Colormap name.
            save_dir: If path provided, save figures.
            show: Whether to call plt.show().
        """
        if not (plot_faults or plot_data):
            logger.info("No plots requested (plot_faults=False, plot_data=False)")
            return

        logger.info("\n--> [Plotting] Visualizing Inputs (Ground Truth & Synthetics)...")
        
        # Prepare save directory
        save_path = None
        if save_dir:
            save_path = Path(save_dir)
            try:
                save_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create save directory '{save_dir}': {e}")
                save_path = None

        # --- 1. Plot true slip model ---
        if plot_faults:
            self._plot_fault_slip(save_path)

        # --- 2. Plot synthetic data ---
        if plot_data:
            logger.info("    Plotting Synthetic Data...")
            for data in self.geodata:
                self._plot_single_dataset(data, cmap, figsize, save_path, show)
            
            if show:
                plt.show()
            
            logger.info("    Plot completed.")

    # =================================================================
    # 3. New Visualization Interface (SciencePlots)
    # =================================================================
    
    def plot_model_comparison(self, slip_type: str = 'totalslip', cmap: str = 'cmc.roma_r', 
                            save_path: str = None, show: bool = True):
        """
        Plot 1x2 Comparison: [Input Truth] vs [Inverted Result] using PolyCollection (patches).
        
        Args:
            slip_type: 'totalslip' (default), 'strikeslip', or 'dipslip'.
            cmap: Colormap name.
            save_path: Directory to save the figure.
            show: Whether to show the plot.
        """
        logger.info(f"--> [Plotting] Generating Model Comparison ({slip_type})...")
        from matplotlib.collections import PolyCollection  # Ensure import
        
        # Helper to extract component
        def _get_slip_values(slip_arr, s_type):
            if slip_arr is None: return None
            # Handle 1D array (assume generic scalar)
            if slip_arr.ndim == 1: return slip_arr
            
            # Handle (N, 3) array: [strike, dip, tensile]
            if s_type == 'totalslip':
                return np.linalg.norm(slip_arr, axis=1)
            elif s_type == 'strikeslip':
                return slip_arr[:, 0]
            elif s_type == 'dipslip':
                return slip_arr[:, 1]
            else:
                raise ValueError(f"Unknown slip_type: {s_type}. Use 'totalslip', 'strikeslip', or 'dipslip'.")

        # Use SciencePlots context
        with sci_plot_style(figsize='double', figsize_height=3.5):
            for fault in self.faults:
                if fault.name not in self.true_slips:
                    logger.warning(f"    No true slip found for {fault.name}, skipping.")
                    continue
                
                if not hasattr(fault, 'patchll'):
                    logger.error(f"    Fault {fault.name} has no 'patchll' attribute. Cannot plot patches.")
                    continue

                # 1. Prepare Data
                val_true = _get_slip_values(self.true_slips[fault.name], slip_type)
                val_inv = _get_slip_values(fault.slip, slip_type)
                
                # 2. Determine Limits (Shared Colorbar)
                # For 'totalslip', usually start at 0. For components, use min/max (can be negative)
                if slip_type == 'totalslip':
                    vmin = 0.0
                    vmax = max(val_true.max(), val_inv.max())
                else:
                    # For components, allow negative ranges
                    vmin = min(val_true.min(), val_inv.min())
                    vmax = max(val_true.max(), val_inv.max())
                    
                    # Optional: specific check for checkerboard (often 0 is background)
                    # If signals are purely positive, clamp vmin to 0 for better contrast?
                    # Let's stick to data limits to be safe.
                    if vmin > 0: vmin = 0 # If all positive, anchor at 0

                # 3. Extract Geometry
                patches_verts = [p[:, :2] for p in fault.patchll]
                
                # Calculate extent
                all_verts = np.vstack(patches_verts)
                lon_min, lon_max = all_verts[:, 0].min(), all_verts[:, 0].max()
                lat_min, lat_max = all_verts[:, 1].min(), all_verts[:, 1].max()
                pad_lon = (lon_max - lon_min) * 0.05
                pad_lat = (lat_max - lat_min) * 0.05

                # 4. Create Plot
                fig, axes = plt.subplots(1, 2, constrained_layout=True, sharex=True, sharey=True)
                
                def _add_poly_collection(ax, values, title):
                    # edgecolors='face' avoids white lines between patches better than 'none' in some viewers
                    pc = PolyCollection(patches_verts, cmap=cmap, edgecolors='none') 
                    
                    pc.set_array(values)
                    pc.set_clim(vmin, vmax)
                    ax.add_collection(pc)
                    
                    # Formatting
                    ax.set_title(title, fontsize=10)
                    set_degree_formatter(ax)
                    
                    # Manually set limits
                    ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
                    ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)
                    
                    ax.set_aspect('equal', adjustable='box')
                    return pc

                # Plot Truth
                _add_poly_collection(axes[0], val_true, f"Input ({slip_type})")
                
                # Plot Inversion
                im = _add_poly_collection(axes[1], val_inv, "Inverted Result")
                
                # Colorbar
                cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
                
                # Proper label based on type
                if slip_type == 'total':
                    cbar.set_label('Total Slip (m)')
                elif slip_type == 'strikeslip':
                    cbar.set_label('Strike-Slip (m)')
                elif slip_type == 'dipslip':
                    cbar.set_label('Dip-Slip (m)')
                else:
                    cbar.set_label('Slip (m)')

                if save_path:
                    p = Path(save_path)
                    p.mkdir(parents=True, exist_ok=True)
                    # Filename includes slip_type
                    fname = p / f'Comp_Model_{fault.name}_{slip_type}.png'
                    plt.savefig(fname, dpi=DEFAULT_DPI)
                    logger.info(f"    Saved: {fname}")
                
                if show: plt.show()
                else: plt.close()

    def plot_data_fit_comparison(self, cmap='RdBu_r', save_path=None, show=True, 
                                 share_colorbar=True, figsize=None):
        """
        Plot N Rows x 3 Cols: [Observed] [Modeled] [Residual].
        
        Args:
            cmap: Colormap for data (e.g., 'RdBu_r' or 'jet').
            save_path: Directory to save figure.
            show: Show plot interactively.
            share_colorbar: If True, Residuals use the same color scale (vmin/vmax) as Data.
                            If False, Residuals use their own auto-scaled limits.
            figsize: Tuple (width, height). If None, calculated automatically based on 'double' column width.
        """
        logger.info("--> [Plotting] Generating Data Fit Comparison...")
        
        # 1. Prepare Data
        plot_data = []
        for data in self.geodata:
            if data.dtype == 'opticorr': continue
            if data.dtype == 'crossfaultoffset': continue

            d_obs = data.vel.copy() if hasattr(data, 'vel') else None
            if d_obs is None: continue
            d_mod = data.synth.copy()
            d_res = d_obs - d_mod

            plot_data.append({
                'name': data.name,
                'lon': data.lon,
                'lat': data.lat,
                'obs': d_obs,
                'mod': d_mod,
                'res': d_res
            })

        if not plot_data: return

        # 2. Setup Figure Size
        n_data = len(plot_data)
        if figsize is None:
            # Default: Double column width (~7.2 in), height adaptive (approx 2.5 inch per row)
            width, _ = publication_figsize(column='double') 
            height = 2.5 * n_data
            figsize = (width, height)

        # 3. Plotting with SciencePlots Style
        with sci_plot_style(style=['science', 'no-latex'], figsize=figsize):
            fig, axes = plt.subplots(n_data, 3, constrained_layout=True)
            if n_data == 1: axes = np.array([axes])

            for i, p in enumerate(plot_data):
                # Limits
                # Abs max for Data (Obs & Mod)
                abs_max_data = np.nanmax(np.abs(np.concatenate((p['obs'], p['mod']))))
                v_lim = (-abs_max_data, abs_max_data) # Symmetrical limits
                
                # Abs max for Residual
                if share_colorbar:
                    r_lim = v_lim
                else:
                    abs_max_res = np.nanmax(np.abs(p['res']))
                    r_lim = (-abs_max_res, abs_max_res)

                # --- Plot Helper ---
                def _scatter(ax, val, title, vmin, vmax):
                    sc = ax.scatter(p['lon'], p['lat'], c=val, cmap=cmap, 
                                  vmin=vmin, vmax=vmax, s=5, edgecolors='none')
                    # Use set_degree_formatter for all axes
                    set_degree_formatter(ax)
                    ax.tick_params(labelsize=8)
                    ax.axis('equal') # Important for maps
                    # Remove minor ticks
                    ax.minorticks_off()
                    
                    if i == 0: ax.set_title(title, fontsize=10, fontweight='bold')
                    if ax == axes[i, 0]: ax.set_ylabel(f"{p['name']}", fontsize=10, rotation=90)
                    return sc

                # Plot Cols
                # Plot Fault traces in background if needed
                for fault in self.faults:
                    lonf, latf = fault.lon, fault.lat
                    for j in range(3):
                        axes[i, j].plot(lonf, latf, color='k', linewidth=1.0, alpha=0.5)
                # Plot Observations
                sc_obs = _scatter(axes[i, 0], p['obs'], "Observed", v_lim[0], v_lim[1])
                sc_mod = _scatter(axes[i, 1], p['mod'], "Modeled", v_lim[0], v_lim[1])
                sc_res = _scatter(axes[i, 2], p['res'], "Residual", r_lim[0], r_lim[1])

                # Colorbars
                # 1. Shared CB for Obs/Mod (Placed across first two columns)
                cb1 = fig.colorbar(sc_mod, ax=[axes[i, 0], axes[i, 1]], 
                                 fraction=0.046, pad=0.04, aspect=30)
                # cb1.set_label('LOS (m)', fontsize=8) # Optional label

                # 2. CB for Residual
                if share_colorbar:
                    # If sharing, just put a colorbar on the residual plot that matches the range
                    cb2 = fig.colorbar(sc_res, ax=axes[i, 2], 
                                     fraction=0.046 * 2, pad=0.04, aspect=30)
                else:
                    cb2 = fig.colorbar(sc_res, ax=axes[i, 2], 
                                     fraction=0.046 * 2, pad=0.04, aspect=30)
                    cb2.set_label('Res (m)', fontsize=8)

            if save_path:
                p = Path(save_path)
                p.mkdir(parents=True, exist_ok=True)
                fname = p / 'Comp_Data_Fit.png'
                plt.savefig(fname, dpi=DEFAULT_DPI)
                logger.info(f"    Saved: {fname}")

            if show: plt.show()
            else: plt.close()