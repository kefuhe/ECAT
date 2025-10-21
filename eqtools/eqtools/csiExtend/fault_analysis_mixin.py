"""
Fault Analysis Mixin Module

This module provides a mixin class that contains common fault analysis methods
for seismic moment and moment magnitude calculations, fault statistics, and plotting.
This mixin can be used by different fault inversion classes to avoid code duplication.

Author: Kefeng He
Date: 2025-08-01
Version: 1.0.0
"""

import numpy as np
from tabulate import tabulate


class FaultAnalysisMixin:
    """
    A mixin class that provides fault analysis functionality including seismic moment
    and moment magnitude calculations, fault statistics printing, and slip distribution plotting.
    
    This mixin is designed to be inherited by classes that have fault objects,
    regardless of whether they are accessed via self.faults or self.multifaults.faults.
    
    Methods:
        _get_faults(): Abstract method to get fault list, implemented by subclasses
        calculate_moment_magnitude(): Calculate seismic moment and magnitude
        _calculate_moment_magnitude_csi(): CSI method implementation
        _calculate_moment_magnitude_hankel(): Hankel formula implementation
        print_moment_magnitude(): Print moment magnitude results
        _print_fault_statistics(): Print comprehensive fault statistics
        plot_multifaults_slip(): Plot slip distribution of multiple faults
    """
    
    def _get_faults(self):
        """
        Get the list of fault objects. This method should be overridden by subclasses
        to return the appropriate fault list based on their internal structure.
        
        Returns:
            list: List of fault objects
            
        Raises:
            AttributeError: If no fault list can be found in the object
        """
        if hasattr(self, 'multifaults') and hasattr(self.multifaults, 'faults'):
            return self.multifaults.faults
        elif hasattr(self, 'faults'):
            return self.faults
        else:
            raise AttributeError("Cannot find faults in this object. "
                               "Please ensure the class has either 'faults' or 'multifaults.faults' attribute.")
    
    def calculate_moment_magnitude(self, faults=None, mu=3.e10, slip_factor=1.0, mode='hankel'):
        """
        Calculate seismic moment and moment magnitude for specified faults.
        
        This method provides two calculation modes:
        - 'csi': Uses CSI faultpostproc method for combined fault processing
        - 'hankel': Uses standard Hankel formula (Mw = 2/3 * (log10(Mo) - 9.1))
        
        Args:
            faults (list, optional): List of fault objects or fault names to analyze.
                                   If None, uses all available faults. Defaults to None.
            mu (float, optional): Shear modulus in Pa. Defaults to 3.e10.
            slip_factor (float, optional): Factor to scale the slip values. Defaults to 1.0.
            mode (str, optional): Calculation mode ('csi' or 'hankel'). Defaults to 'hankel'.
        
        Returns:
            dict: Dictionary containing moment and magnitude information with keys:
                - For 'csi' mode: 'total_moment', 'total_magnitude', 'mode', 'processor', 'faults'
                - For 'hankel' mode: 'fault_moments', 'total_moment', 'total_magnitude', 
                  'mode', 'mu', 'slip_factor', 'faults'
        
        Raises:
            ValueError: If mode is not 'csi' or 'hankel'
        """
        # Handle faults parameter
        if faults is None:
            target_faults = self._get_faults()
        elif isinstance(faults, list):
            if len(faults) > 0:
                if isinstance(faults[0], str):
                    # List of fault names
                    all_faults = self._get_faults()
                    fault_dict = {fault.name: fault for fault in all_faults}
                    target_faults = []
                    for fault_name in faults:
                        if fault_name in fault_dict:
                            target_faults.append(fault_dict[fault_name])
                        else:
                            print(f"Warning: Fault '{fault_name}' not found in available faults")
                else:
                    # List of fault objects
                    target_faults = faults
            else:
                target_faults = self._get_faults()
        else:
            target_faults = self._get_faults()
        
        if mode == 'csi':
            return self._calculate_moment_magnitude_csi(target_faults, mu, slip_factor)
        elif mode == 'hankel':
            return self._calculate_moment_magnitude_hankel(target_faults, mu, slip_factor)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'csi' or 'hankel'.")
    
    def _calculate_moment_magnitude_csi(self, target_faults, mu=3.e10, slip_factor=1.0):
        """
        Calculate moment magnitude using CSI faultpostproc method.
        
        This method combines multiple faults into a single fault object and uses
        the CSI library's faultpostproc class to compute seismic moments, moment
        tensor, and magnitude.
        
        Args:
            target_faults (list): List of fault objects to process
            mu (float, optional): Shear modulus in Pa. Defaults to 3.e10.
            slip_factor (float, optional): Factor to scale slip values. Defaults to 1.0.
        
        Returns:
            dict: Dictionary containing:
                - 'total_moment': Total seismic moment in N·m
                - 'total_magnitude': Moment magnitude
                - 'mode': 'csi'
                - 'processor': CSI faultpostproc object for further analysis
                - 'faults': List of fault names processed
        """
        import csi.faultpostproc as faultpp
        
        # Get the first fault for reference parameters
        first_fault = target_faults[0]
        lon0, lat0, utmzone = first_fault.lon0, first_fault.lat0, first_fault.utmzone

        # Combine the faults by duplicating the first fault
        combined_fault = first_fault.duplicateFault()

        # Add patches and slip from remaining faults
        if len(target_faults) > 1:
            for ifault in target_faults[1:]:
                for patch, slip in zip(ifault.patch, ifault.slip):
                    combined_fault.N_slip = combined_fault.slip.shape[0] + 1
                    combined_fault.addpatch(patch, slip)

        # Create combined fault name
        fault_names = [fault.name for fault in target_faults]
        combined_name = '_'.join(fault_names)

        # Scale the slip values
        combined_fault.slip *= slip_factor

        # Convert patches to vertices
        combined_fault.setVerticesFromPatches()
        combined_fault.numpatch = combined_fault.Faces.shape[0]
        
        # Compute triangle areas, moments, moment tensor and magnitude
        combined_fault.compute_patch_areas()
        fault_processor = faultpp(combined_name, combined_fault, mu, 
                                lon0=lon0, lat0=lat0, utmzone=utmzone)
        fault_processor.computeMoments()
        fault_processor.computeMomentTensor()
        fault_processor.computeMagnitude()

        # Store processor for later use
        self.tripproc = fault_processor
        
        return {
            'total_moment': fault_processor.Mo,
            'total_magnitude': fault_processor.Mw,
            'mode': 'csi',
            'processor': fault_processor,
            'faults': [fault.name for fault in target_faults]
        }
    
    def _calculate_moment_magnitude_hankel(self, target_faults, mu=3.e10, slip_factor=1.0):
        """
        Calculate moment magnitude using the standard Hankel formula.
        
        Uses the relationship: Mw = 2/3 * (log10(Mo) - 9.1)
        where Mo is the seismic moment in N·m.
        
        Args:
            target_faults (list): List of fault objects to process
            mu (float, optional): Shear modulus in Pa. Defaults to 3.e10.
            slip_factor (float, optional): Factor to scale slip values. Defaults to 1.0.
        
        Returns:
            dict: Dictionary containing:
                - 'fault_moments': Dictionary with individual fault moment data
                - 'total_moment': Total seismic moment in N·m
                - 'total_magnitude': Total moment magnitude
                - 'mode': 'hankel'
                - 'mu': Shear modulus used
                - 'slip_factor': Slip scaling factor used
                - 'faults': List of fault names processed
        """
        fault_moments = {}
        total_moment = 0
        
        for ifault in target_faults:
            if hasattr(ifault, 'slip') and ifault.slip is not None:
                # Calculate total slip magnitude
                if ifault.slip.shape[1] >= 2:
                    # For strike-slip and dip-slip components
                    total_slip = np.sqrt(ifault.slip[:, 0]**2 + ifault.slip[:, 1]**2) * slip_factor
                else:
                    # For single slip component
                    total_slip = np.abs(ifault.slip[:, 0]) * slip_factor
                
                # Calculate seismic moment: Mo = μ * A * D
                patch_areas = getattr(ifault, 'area', ifault.compute_patch_areas())
                areas = np.array(patch_areas) * 1e6  # Convert km² to m²
                moment = np.sum(mu * areas * total_slip)
                
                # Calculate moment magnitude using Hankel formula
                moment_magnitude = 2.0 / 3.0 * (np.log10(moment) - 9.1)
                
                # Store individual fault data
                fault_moments[ifault.name] = {
                    'moment': moment,
                    'magnitude': moment_magnitude,
                    'mean_slip': np.mean(total_slip),
                    'max_slip': np.max(total_slip)
                }
                
                total_moment += moment
        
        # Calculate total magnitude
        total_magnitude = 2.0 / 3.0 * (np.log10(total_moment) - 9.1) if total_moment > 0 else 0.0
        
        return {
            'fault_moments': fault_moments,
            'total_moment': total_moment,
            'total_magnitude': total_magnitude,
            'mode': 'hankel',
            'mu': mu,
            'slip_factor': slip_factor,
            'faults': [fault.name for fault in target_faults]
        }
    
    def print_moment_magnitude(self, faults=None, mu=3.e10, slip_factor=1.0, mode='hankel'):
        """
        Print seismic moment and moment magnitude information in a formatted table.
        
        This method calculates and displays the seismic moment and magnitude information
        for the specified faults using either CSI or Hankel calculation methods.
        
        Args:
            faults (list, optional): List of fault objects or fault names to analyze.
                                   If None, uses all available faults. Defaults to None.
            mu (float, optional): Shear modulus in Pa. Defaults to 3.e10.
            slip_factor (float, optional): Factor to scale slip values. Defaults to 1.0.
            mode (str, optional): Calculation mode ('csi' or 'hankel'). Defaults to 'hankel'.
        
        Returns:
            dict: Results dictionary from calculate_moment_magnitude method
        """
        # Calculate moment and magnitude
        results = self.calculate_moment_magnitude(faults=faults, mu=mu, 
                                                slip_factor=slip_factor, mode=mode)
        
        fault_list_str = ', '.join(results['faults']) if 'faults' in results else 'All faults'
        
        print("\n" + "="*60)
        print(f"Seismic Moment and Magnitude ({mode.upper()} mode)")
        print(f"Analyzed faults: {fault_list_str}")
        print("="*60)
        
        if mode == 'csi':
            # Simple output for CSI mode
            print(f"Mo is: {results['total_moment']:.8e} N·m")
            print(f"Mw is: {results['total_magnitude']:.2f}")
            print(f"Shear modulus: {mu:.2e} Pa")
            print(f"Slip scaling factor: {slip_factor:.2f}")
            
        elif mode == 'hankel':
            # Detailed table output for Hankel mode
            if 'fault_moments' in results:
                table_data = []
                for fault_name, fault_data in results['fault_moments'].items():
                    table_data.append([
                        fault_name,
                        f"{fault_data['moment']:.3e}",
                        f"{fault_data['magnitude']:.2f}",
                        f"{fault_data['mean_slip']:.4f}",
                        f"{fault_data['max_slip']:.4f}"
                    ])
                
                # Add total row
                table_data.append([
                    'TOTAL',
                    f"{results['total_moment']:.3e}",
                    f"{results['total_magnitude']:.2f}",
                    '-',
                    '-'
                ])
                
                headers = ['Fault Name', 'Moment (N·m)', 'Magnitude', 'Mean Slip (m)', 'Max Slip (m)']
                print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='left'))
            
            print(f"\nParameters:")
            print(f"Shear modulus: {results['mu']:.2e} Pa")
            print(f"Slip scaling factor: {results['slip_factor']:.2f}")
            print(f"Formula: Mw = 2/3 * (log10(Mo) - 9.1)")
        
        print("="*60)
        
        return results

    def _print_fault_statistics(self, faults=None, fault_groups=None):
        """
        Print comprehensive fault statistics in formatted tables.
        
        This method displays:
        1. Fault geometry statistics (strike, dip, patches, length)
        2. Slip statistics (strike-slip, dip-slip, total slip)
        3. Seismic moment and magnitude calculations with grouping support
        
        Args:
            faults (list, optional): List of fault objects or fault names to analyze.
                                   If None, uses all available faults. Defaults to None.
            fault_groups (list, optional): List of fault groups for moment calculation.
                                         Each group should be a list of fault names.
                                         If None, treats each fault as its own group.
                                         Defaults to None.
        """
        # Handle faults parameter
        if faults is None:
            target_faults = self._get_faults()
        elif isinstance(faults, list):
            if len(faults) > 0:
                if isinstance(faults[0], str):
                    # List of fault names
                    all_faults = self._get_faults()
                    fault_dict = {fault.name: fault for fault in all_faults}
                    target_faults = []
                    for fault_name in faults:
                        if fault_name in fault_dict:
                            target_faults.append(fault_dict[fault_name])
                        else:
                            print(f"Warning: Fault '{fault_name}' not found in available faults")
                else:
                    # List of fault objects
                    target_faults = faults
            else:
                target_faults = self._get_faults()
        else:
            print("Warning: Invalid faults parameter, using available faults")
            target_faults = self._get_faults()
        
        print("\n" + "="*80)
        print("Fault Statistics Summary")
        print("="*80)
        
        # Fault geometry statistics table
        geometry_table_data = []
        for ifault in target_faults:
            mean_strike = np.mean(ifault.getStrikes() * 180 / np.pi)
            mean_dip = np.mean(ifault.getDips() * 180 / np.pi)
            num_patches = len(ifault.patch)
            
            # Calculate fault length using cumdistance method
            try:
                fault_length = ifault.cumdistance(discretized=False)[-1]
                length_str = f"{fault_length:.2f} km"
            except:
                length_str = 'N/A'
            
            geometry_table_data.append([
                ifault.name,
                f"{mean_strike:.2f}°",
                f"{mean_dip:.2f}°",
                num_patches,
                length_str
            ])
        
        geometry_headers = ['Fault Name', 'Mean Strike', 'Mean Dip', 'Patches', 'Length']
        print("\nFault Geometry:")
        print(tabulate(geometry_table_data, headers=geometry_headers, 
                      tablefmt='grid', stralign='left'))
        
        # Fault slip statistics table
        slip_table_data = []
        for ifault in target_faults:
            if hasattr(ifault, 'slip') and ifault.slip is not None:
                # Extract slip components
                if ifault.slip.shape[1] >= 2:
                    strike_slip = ifault.slip[:, 0]  # Strike-slip component
                    dip_slip = ifault.slip[:, 1]     # Dip-slip component
                    total_slip = np.sqrt(strike_slip**2 + dip_slip**2)  # Total slip magnitude
                    
                    # Calculate statistics (min, mean, max)
                    ss_stats = {
                        'mean': np.mean(strike_slip),
                        'max': np.max(strike_slip),
                        'min': np.min(strike_slip)
                    }
                    
                    ds_stats = {
                        'mean': np.mean(dip_slip),
                        'max': np.max(dip_slip),
                        'min': np.min(dip_slip)
                    }
                    
                    total_stats = {
                        'mean': np.mean(total_slip),
                        'max': np.max(total_slip),
                        'min': np.min(total_slip)
                    }
                    
                    # Add rows for each slip component
                    slip_table_data.extend([
                        [ifault.name, 'Strike-slip', f"{ss_stats['mean']:.4f}", 
                         f"{ss_stats['max']:.4f}", f"{ss_stats['min']:.4f}"],
                        ['', 'Dip-slip', f"{ds_stats['mean']:.4f}", 
                         f"{ds_stats['max']:.4f}", f"{ds_stats['min']:.4f}"],
                        ['', 'Total slip', f"{total_stats['mean']:.4f}", 
                         f"{total_stats['max']:.4f}", f"{total_stats['min']:.4f}"]
                    ])
                else:
                    # Single slip component only
                    slip = ifault.slip[:, 0]
                    slip_stats = {
                        'mean': np.mean(slip),
                        'max': np.max(slip),
                        'min': np.min(slip)
                    }
                    
                    slip_table_data.append([
                        ifault.name, 'Slip', f"{slip_stats['mean']:.4f}",
                        f"{slip_stats['max']:.4f}", f"{slip_stats['min']:.4f}"
                    ])
            else:
                slip_table_data.append([
                    ifault.name, 'No slip data', 'N/A', 'N/A', 'N/A'
                ])
        
        slip_headers = ['Fault Name', 'Component', 'Mean (m)', 'Max (m)', 'Min (m)']
        print("\nFault Slip Statistics:")
        print(tabulate(slip_table_data, headers=slip_headers, 
                      tablefmt='grid', stralign='left'))
        
        # Seismic moment calculations
        moment_table_data = []
        
        # Use default fault groups if not provided
        if fault_groups is None:
            # Check for configuration-based fault groups
            if (hasattr(self, 'config') and hasattr(self.config, 'alpha') 
                and 'faults' in self.config.alpha):
                fault_groups = self.config.alpha['faults']
            else:
                # Fallback: treat each fault as its own group
                fault_groups = [[fault.name] for fault in target_faults]
        
        # Calculate individual fault moments
        fault_moments = {}
        for ifault in target_faults:
            if hasattr(ifault, 'slip') and ifault.slip is not None:
                # Calculate total slip magnitude
                if ifault.slip.shape[1] >= 2:
                    total_slip = np.sqrt(ifault.slip[:, 0]**2 + ifault.slip[:, 1]**2)
                else:
                    total_slip = np.abs(ifault.slip[:, 0])
                
                # Calculate seismic moment
                patch_areas = getattr(ifault, 'area', ifault.compute_patch_areas())
                areas = np.array(patch_areas) * 1e6  # Convert km² to m²
                mu = getattr(self, 'shear_modulus', 3e10)  # Default shear modulus
                moment = np.sum(mu * areas * total_slip)
                moment_magnitude = 2.0 / 3.0 * (np.log10(moment) - 9.1)
                
                fault_moments[ifault.name] = {
                    'moment': moment,
                    'magnitude': moment_magnitude,
                    'mean_slip': np.mean(total_slip),
                    'max_slip': np.max(total_slip)
                }
                
                moment_table_data.append([
                    ifault.name,
                    f"{moment:.3e}",
                    f"{moment_magnitude:.2f}",
                    f"{np.mean(total_slip):.4f}",
                    f"{np.max(total_slip):.4f}"
                ])
        
        # Add separator line for grouped results
        if len(fault_groups) > 1 or (len(fault_groups) == 1 and len(fault_groups[0]) > 1):
            moment_table_data.append(['-'*10, '-'*10, '-'*10, '-'*10, '-'*10])
        
        # Calculate group moments
        total_moment = 0
        for i, group in enumerate(fault_groups):
            if isinstance(group, str):
                group = [group]  # Convert single fault name to list
            
            group_moment = 0
            group_names = []
            
            for fault_name in group:
                if fault_name in fault_moments:
                    group_moment += fault_moments[fault_name]['moment']
                    group_names.append(fault_name)
            
            if group_moment > 0:
                group_magnitude = 2.0 / 3.0 * (np.log10(group_moment) - 9.1)
                total_moment += group_moment
                
                # Create group display name
                if len(group_names) == 1:
                    group_display_name = f"Event {i+1}: {group_names[0]}"
                else:
                    group_display_name = f"Event {i+1}: {', '.join(group_names)}"

                moment_table_data.append([
                    group_display_name,
                    f"{group_moment:.3e}",
                    f"{group_magnitude:.2f}",
                    '-',
                    '-'
                ])
        
        # Add total row
        if total_moment > 0:
            total_magnitude = 2.0 / 3.0 * (np.log10(total_moment) - 9.1)
            moment_table_data.append([
                'TOTAL',
                f"{total_moment:.3e}",
                f"{total_magnitude:.2f}",
                '-',
                '-'
            ])
        
        moment_headers = ['Fault Name/Group', 'Moment (N·m)', 'Magnitude', 
                         'Mean Slip (m)', 'Max Slip (m)']
        print("\nSeismic Moment and Magnitude:")
        print(tabulate(moment_table_data, headers=moment_headers, 
                      tablefmt='grid', stralign='left'))
        
        print("="*80)
    
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
        import numpy as np
        
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
    
    def calculate_and_print_fit_statistics(self, model='median'):
        """
        Calculate and print fit statistics for all datasets.
        
        Parameters:
        -----------
        model : str
            Model type to use ('median', 'mean', 'MAP', etc.) or BLSE for BLSE model.
        """
        print("\n" + "="*70)
        print(f"Data Fit Statistics ({model.upper()} model)")
        print("="*70)
        
        # Get the faults - different access patterns for different classes
        target_faults = self._get_faults()
        
        # Get geodata and verticals from configuration
        geodata = self.config.geodata['data']
        verticals = self.config.geodata['verticals']  
        polys = self.config.geodata['polys']
        
        # Build synthetics and calculate statistics for each dataset
        for idata, ivert, ipoly in zip(geodata, verticals, polys):
            ipoly = ipoly if ipoly is None else 'include'
            idata.buildsynth(target_faults, direction='sd', poly=ipoly, vertical=ivert)
            # Calculate RMS and VR using the helper method
            rms, vr = self.calculate_data_fit_metrics(idata, ivert)
            print(f"{idata.name:<15} | RMS: {rms:8.4f} | VR: {vr:6.2f}%")
    
        print("="*70)

    def plot_multifaults_slip(self, faults=None, figsize=(None, None), slip='total', 
                             cmap='precip3_16lev_change.cpt', norm=None, show=True, savefig=False, 
                             ftype='pdf', dpi=600, bbox_inches=None, method='cdict', N=None, 
                             drawCoastlines=False, plot_on_2d=True, style=['notebook'], 
                             cbaxis=[0.1, 0.2, 0.1, 0.02], cblabel='', xlabelpad=None, 
                             ylabelpad=None, zlabelpad=None, xtickpad=None, ytickpad=None, 
                             ztickpad=None, elevation=None, azimuth=None, shape=(1.0, 1.0, 1.0), 
                             zratio=None, plotTrace=True, depth=None, zticks=None, map_expand=0.2, 
                             fault_expand=0.1, plot_faultEdges=False, faultEdges_color='k', 
                             faultEdges_linewidth=1.0, suffix='', remove_direction_labels=False, 
                             zaxis_position='bottom-left', show_grid=True, grid_color='#bebebe',
                             background_color='white', axis_color=None, cbticks=None, 
                             cblinewidth=None, cbfontsize=None, cb_label_side='opposite',
                             map_cbaxis=None):
        """
        Plot the slip distribution of multiple faults.
        
        This method creates a comprehensive visualization of slip distribution across
        multiple fault surfaces, combining them if necessary and providing both 2D
        map and 3D fault plane visualizations.
        
        Args:
            faults (list, optional): List of fault objects or fault names to plot.
                                   If None, uses all available faults. Defaults to None.
            figsize (tuple, optional): Size of the figure and map. Defaults to (None, None).
            slip (str, optional): Type of slip to plot ('total', 'strike', 'dip'). 
                                Defaults to 'total'.
            cmap (str, optional): Colormap to use. Defaults to 'precip3_16lev_change.cpt'.
            norm (optional): Normalization for the colormap. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to True.
            savefig (bool, optional): Whether to save the figure. Defaults to False.
            ftype (str, optional): File type for saving the figure. Defaults to 'pdf'.
            dpi (int, optional): Dots per inch for the saved figure. Defaults to 600.
            bbox_inches (optional): Bounding box in inches for saving the figure. 
                                  Defaults to None.
            method (str, optional): Method for getting the colormap. Defaults to 'cdict'.
            N (int, optional): Number of colors in the colormap. Defaults to None.
            drawCoastlines (bool, optional): Whether to draw coastlines. Defaults to False.
            plot_on_2d (bool, optional): Whether to plot on a 2D map. Defaults to True.
            style (list, optional): Style for the plot. Defaults to ['notebook'].
            cbaxis (list, optional): Colorbar axis position. Defaults to [0.1, 0.2, 0.1, 0.02].
            cblabel (str, optional): Label for the colorbar. Defaults to ''.
            xlabelpad, ylabelpad, zlabelpad (float, optional): Padding for the axis labels. 
                                                             Defaults to None.
            xtickpad, ytickpad, ztickpad (float, optional): Padding for the axis ticks. 
                                                          Defaults to None.
            elevation, azimuth (float, optional): Elevation and azimuth angles for the 3D plot. 
                                                Defaults to None.
            shape (tuple, optional): Shape of the 3D plot. Defaults to (1.0, 1.0, 1.0).
            zratio (float, optional): Ratio for the z-axis. Defaults to None.
            plotTrace (bool, optional): Whether to plot the fault trace. Defaults to True.
            depth (float, optional): Depth for the z-axis. Defaults to None.
            zticks (list, optional): Ticks for the z-axis. Defaults to None.
            map_expand (float, optional): Expansion factor for the map. Defaults to 0.2.
            fault_expand (float, optional): Expansion factor for the fault. Defaults to 0.1.
            plot_faultEdges (bool, optional): Whether to plot the fault edges. Defaults to False.
            faultEdges_color (str, optional): Color for the fault edges. Defaults to 'k'.
            faultEdges_linewidth (float, optional): Line width for the fault edges. 
                                                  Defaults to 1.0.
            suffix (str, optional): Suffix for the saved figure filename. Defaults to ''.
            remove_direction_labels (bool, optional): If True, remove E, N, S, W from axis labels. 
                                                    Defaults to False.
            zaxis_position (str, optional): Position of the z-axis ('bottom-left', 'top-right'). 
                                          Defaults to 'bottom-left'.
            show_grid (bool, optional): Whether to show grid lines. Defaults to True.
            grid_color (str, optional): Color of the grid lines. Defaults to '#bebebe'.
            background_color (str, optional): Background color of the plot. Defaults to 'white'.
            axis_color (str, optional): Color of the axes. Defaults to None.
            cbticks (list, optional): List of ticks to set on the colorbar. Defaults to None.
            cblinewidth (int, optional): Width of the colorbar label border and tick lines. 
                                       Defaults to None.
            cbfontsize (int, optional): Font size of the colorbar label. Defaults to None.
            cb_label_side (str, optional): Position of the label relative to the ticks 
                                         ('opposite' or 'same'). Defaults to 'opposite'.
            map_cbaxis (optional): Axis for the colorbar on the map plot. Defaults to None.
        
        Returns:
            None
        """
        from ..plottools import plot_slip_distribution
        
        # Handle faults parameter
        if faults is None:
            target_faults = self._get_faults()
        elif isinstance(faults, list):
            if len(faults) > 0:
                if isinstance(faults[0], str):
                    # List of fault names
                    all_faults = self._get_faults()
                    fault_dict = {fault.name: fault for fault in all_faults}
                    target_faults = []
                    for fault_name in faults:
                        if fault_name in fault_dict:
                            target_faults.append(fault_dict[fault_name])
                        else:
                            print(f"Warning: Fault '{fault_name}' not found in available faults")
                else:
                    # List of fault objects
                    target_faults = faults
            else:
                target_faults = self._get_faults()
        else:
            print("Warning: Invalid faults parameter, using all available faults")
            target_faults = self._get_faults()

        if len(target_faults) > 1:
            # 是要判断所有fault的patchType是一样的，可选triangle或rectangle
            if all(fault.patchType == target_faults[0].patchType for fault in target_faults):
                # Combine faults if there are more than one
                combined_fault = target_faults[0].duplicateFault()
                combined_fault.name = 'Combined Fault'
                for fault in target_faults[1:]:
                    for ipatch, islip in zip(fault.patch, fault.slip):
                        combined_fault.N_slip = combined_fault.slip.shape[0] + 1
                        combined_fault.addpatch(ipatch, islip)
                combined_fault.setTrace(0.1)
                mfault = combined_fault
                add_faults = target_faults
            else:
                mfault = target_faults
                add_faults = target_faults
        else:
            # Directly plot if there is only one fault
            mfault = target_faults[0]
            add_faults = [mfault]

        # Plot the slip distribution using the external plotting function
        plot_slip_distribution(mfault, add_faults=add_faults, slip=slip, cmap=cmap, norm=norm, 
                             figsize=figsize, method=method, N=N, drawCoastlines=drawCoastlines, 
                             plot_on_2d=plot_on_2d, cbaxis=cbaxis, cblabel=cblabel, show=show, 
                             savefig=savefig, ftype=ftype, dpi=dpi, bbox_inches=bbox_inches,
                             remove_direction_labels=remove_direction_labels, cbticks=cbticks, 
                             cblinewidth=cblinewidth, cbfontsize=cbfontsize, 
                             cb_label_side=cb_label_side, map_cbaxis=map_cbaxis, style=style,
                             xlabelpad=xlabelpad, ylabelpad=ylabelpad, zlabelpad=zlabelpad, 
                             xtickpad=xtickpad, ytickpad=ytickpad, ztickpad=ztickpad, 
                             elevation=elevation, azimuth=azimuth, shape=shape, zratio=zratio, 
                             plotTrace=plotTrace, depth=depth, zticks=zticks, 
                             map_expand=map_expand, fault_expand=fault_expand, 
                             plot_faultEdges=plot_faultEdges, faultEdges_color=faultEdges_color,
                             faultEdges_linewidth=faultEdges_linewidth, suffix=suffix, 
                             show_grid=show_grid, grid_color=grid_color, 
                             background_color=background_color, axis_color=axis_color,
                             zaxis_position=zaxis_position)

        # All Done
        return
    
# EOF