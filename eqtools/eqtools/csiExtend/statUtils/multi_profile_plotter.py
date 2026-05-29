"""
Multi-Profile Line Plotter

Simple tool to plot multiple profile datasets with swath data visualization options.
Reads saved profile data and creates comparison plots with optional swath envelope and median lines.

Example JSON configuration:
{
    "title": "Multi-Dataset Profile Comparison", 
    "datasets": [
        {
            "name": "Elevation",
            "file": "elevation_profile.hdf5",
            "format": "hdf5",
            "color": "blue",
            "linestyle": "-",
            "linewidth": 2,
            "show_swath_points": true,
            "show_swath_envelope": true,
            "show_swath_median": true,
            "show_interpolated_line": false
        },
        {
            "name": "Gravity",
            "file": "gravity_profile.json",
            "format": "json", 
            "color": "red",
            "linestyle": "--",
            "linewidth": 1.5,
            "show_swath_points": false,
            "show_swath_envelope": true,
            "show_swath_median": true,
            "show_interpolated_line": true
        }
    ],
    "plot_settings": {
        "figsize": [14, 8],
        "xlabel": "Distance along profile (km)",
        "ylabel": "Value",
        "grid": true,
        "legend": true,
        "save_figure": false,
        "output_file": "multi_profile_comparison.png",
        "dpi": 300,
        "swath_point_alpha": 0.6,
        "swath_point_size": 10,
        "envelope_alpha": 0.3
    }
}

Authors:
    [Your Name]

Version:
    1.0.0

Last Updated:
    2025-08-03
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import h5py

# Import existing profile tools for data structures
from .profile_analyzer import ProfileData
from ...plottools import sci_plot_style


class SwathProfileData:
    """Enhanced profile data container with swath processing capabilities."""
    
    def __init__(self, profile_data: ProfileData):
        """Initialize from ProfileData object."""
        self.profile_distances = profile_data.profile_distances
        self.profile_values = profile_data.profile_values
        self.profile_uncertainties = profile_data.profile_uncertainties
        self.profile_coordinates_lonlat = profile_data.profile_coordinates_lonlat
        
        # Swath data
        self.swath_distances = profile_data.swath_distances
        self.swath_values = profile_data.swath_values
        self.swath_coordinates_lonlat = profile_data.swath_coordinates_lonlat
        self.swath_offsets = profile_data.swath_offsets
        
        # Metadata
        self.units = profile_data.units
        self.description = profile_data.description
        
        # Computed swath statistics
        self._swath_envelope = None
        self._swath_median = None
    
    def compute_swath_envelope(self, 
                              method: str = 'minmax',
                              percentiles: Tuple[float, float] = (5, 95),
                              window_size: Optional[float] = None,
                              min_points: int = 0,
                              max_std: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute swath envelope (upper and lower bounds).
        
        Parameters:
        -----------
        method : str
            'minmax' for min/max envelope, 'percentile' for percentile-based
        percentiles : tuple
            Lower and upper percentiles if using percentile method
        window_size : float, optional
            Window size for computing envelope. If None, auto-determined.
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Upper and lower envelope arrays
        """
        if (self.swath_distances is None or 
            self.swath_values is None or 
            len(self.swath_distances) == 0):
            return None, None
        
        # Use swath data directly for smoother results
        return self._compute_swath_statistics_smooth(
            statistic=method, 
            percentiles=percentiles,
            window_size=window_size,
            return_envelope=True,
            min_points=min_points,
            max_std=max_std
        )
    
    def compute_swath_median(self, 
                           window_size: Optional[float] = None,
                           smooth_factor: float = 0.5,
                           min_points: int = 0,
                           max_std: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute swath median line using smooth windowing approach.
        
        Parameters:
        -----------
        window_size : float, optional
            Window size for computing median. If None, auto-determined.
        smooth_factor : float
            Smoothing factor for window overlap (0-1)
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Distances and median values
        """
        if (self.swath_distances is None or 
            self.swath_values is None or 
            len(self.swath_distances) == 0):
            return None, None
        
        return self._compute_swath_statistics_smooth(
            statistic='median',
            window_size=window_size,
            smooth_factor=smooth_factor,
            return_envelope=False,
            min_points=min_points,
            max_std=max_std
        )
    
    def _compute_swath_statistics_smooth(self,
                                       statistic: str = 'median',
                                       percentiles: Tuple[float, float] = (5, 95),
                                       window_size: Optional[float] = None,
                                       smooth_factor: float = 0.5,
                                       return_envelope: bool = False,
                                       min_points: int = 5,
                                       max_std: Optional[float] = None) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """
        Compute swath statistics using smooth windowing approach similar to profile_analyzer.
        
        Parameters:
        -----------
        statistic : str
            'median', 'mean', 'minmax', or 'percentile'
        percentiles : tuple
            Percentiles for envelope calculation
        window_size : float, optional
            Window size. If None, auto-determined from data density
        smooth_factor : float
            Window overlap factor (0-1)
        return_envelope : bool
            Whether to return envelope bounds
            
        Returns:
        --------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]
            For median/mean: (distances, values)
            For envelope: ((distances, upper_values), (distances, lower_values))
        """
        # Sort swath data by distance
        sort_indices = np.argsort(self.swath_distances)
        sorted_distances = self.swath_distances[sort_indices]
        sorted_values = self.swath_values[sort_indices]
        
        # Remove NaN values
        valid_mask = ~np.isnan(sorted_values)
        sorted_distances = sorted_distances[valid_mask]
        sorted_values = sorted_values[valid_mask]
        
        if len(sorted_distances) == 0:
            return None, None
        
        # Determine window size automatically if not provided
        if window_size is None:
            distance_range = sorted_distances[-1] - sorted_distances[0]
            n_points = len(sorted_distances)
            
            # Adaptive window size based on data density
            if n_points > 100:
                window_size = distance_range / (n_points / 20)  # About 20 points per window
            else:
                window_size = distance_range / 10  # 10 windows across the profile
                
            # Ensure minimum window size
            window_size = max(window_size, distance_range / 50)
        
        # Create output arrays
        min_dist, max_dist = sorted_distances[0], sorted_distances[-1]
        step_size = window_size * smooth_factor
        
        # Generate distance grid for output
        n_steps = int((max_dist - min_dist) / step_size) + 1
        output_distances = np.linspace(min_dist, max_dist, n_steps)
        
        if return_envelope:
            upper_values = np.full_like(output_distances, np.nan)
            lower_values = np.full_like(output_distances, np.nan)
        else:
            output_values = np.full_like(output_distances, np.nan)
        
        # Compute statistics in sliding windows
        for i, center_dist in enumerate(output_distances):
            # Define window bounds
            window_start = center_dist - window_size / 2
            window_end = center_dist + window_size / 2
            
            # Find points within window
            window_mask = (sorted_distances >= window_start) & (sorted_distances <= window_end)
            window_values = sorted_values[window_mask]
            
            if len(window_values) <= min_points:
                continue
            if max_std is not None and np.std(window_values) > max_std:
                continue

            # Compute requested statistic
            if return_envelope:
                if statistic == 'minmax':
                    upper_values[i] = np.max(window_values)
                    lower_values[i] = np.min(window_values)
                elif statistic == 'percentile' and len(window_values) >= 3:
                    upper_values[i] = np.percentile(window_values, percentiles[1])
                    lower_values[i] = np.percentile(window_values, percentiles[0])
                elif statistic == 'percentile':
                    # Fallback for small samples
                    upper_values[i] = np.max(window_values)
                    lower_values[i] = np.min(window_values)
            else:
                if statistic == 'median':
                    output_values[i] = np.median(window_values)
                elif statistic == 'mean':
                    output_values[i] = np.mean(window_values)
        
        # Interpolate any remaining gaps
        if return_envelope:
            upper_values = self._interpolate_gaps(upper_values)
            lower_values = self._interpolate_gaps(lower_values)
            self._swath_envelope = (upper_values, lower_values)
            return upper_values, lower_values
        else:
            output_values = self._interpolate_gaps(output_values)
            if statistic == 'median':
                self._swath_median = (output_distances, output_values)
            return output_distances, output_values
    
    def compute_swath_mean(self, 
                          window_size: Optional[float] = None,
                          smooth_factor: float = 0.5,
                          min_points: int = 0,
                          max_std: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute swath mean line.
        
        Parameters:
        -----------
        window_size : float, optional
            Window size for computing mean
        smooth_factor : float
            Smoothing factor for window overlap
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Distances and mean values
        """
        if (self.swath_distances is None or 
            self.swath_values is None or 
            len(self.swath_distances) == 0):
            return None, None
        
        return self._compute_swath_statistics_smooth(
            statistic='mean',
            window_size=window_size,
            smooth_factor=smooth_factor,
            return_envelope=False,
            min_points=min_points,
            max_std=max_std
        )
    
    def _interpolate_gaps(self, values: np.ndarray) -> np.ndarray:
        """Interpolate NaN gaps in arrays."""
        valid_mask = ~np.isnan(values)
        
        if not np.any(valid_mask) or np.all(valid_mask):
            return values
        
        valid_indices = np.where(valid_mask)[0]
        valid_values = values[valid_mask]
        
        if len(valid_values) > 1:
            all_indices = np.arange(len(values))
            interp_mask = (all_indices >= valid_indices[0]) & (all_indices <= valid_indices[-1])
            values[interp_mask] = np.interp(
                all_indices[interp_mask], 
                valid_indices, 
                valid_values
            )
        
        return values


class MultiSwathProfilePlotter:
    """
    Enhanced plotter for multiple profile datasets with swath visualization.
    """
    
    def __init__(self, config_file: Union[str, Path, Dict]):
        """
        Initialize plotter with configuration.
        
        Parameters:
        -----------
        config_file : str, Path, or dict
            Path to JSON configuration file or configuration dictionary
        """
        if isinstance(config_file, (str, Path)):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config_file.copy()
        
        self.datasets = {}
        self.swath_profile_data = {}
    
    def load_profile_data(self, dataset_name: str) -> SwathProfileData:
        """
        Load profile data from file and wrap in SwathProfileData.
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset in configuration
            
        Returns:
        --------
        SwathProfileData
            Loaded and wrapped profile data
        """
        if dataset_name in self.swath_profile_data:
            return self.swath_profile_data[dataset_name]
        
        dataset_config = None
        for ds in self.config['datasets']:
            if ds['name'] == dataset_name:
                dataset_config = ds
                break
        
        if dataset_config is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        # Load data based on format
        file_path = Path(dataset_config['file'])
        file_format = dataset_config.get('format', 'auto')
        
        if file_format == 'auto':
            file_format = file_path.suffix.lower().lstrip('.')
            if file_format in ['h5', 'hdf']:
                file_format = 'hdf5'
        
        if file_format == 'csv':
            profile_data = self._load_csv_data(file_path)
        elif file_format == 'hdf5':
            profile_data = self._load_hdf5_data(file_path)
        elif file_format == 'json':
            profile_data = self._load_json_data(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        # Wrap in SwathProfileData
        swath_data = SwathProfileData(profile_data)
        self.swath_profile_data[dataset_name] = swath_data
        return swath_data
    
    def _load_csv_data(self, file_path: Path) -> ProfileData:
        """Load profile data from CSV file."""
        df = pd.read_csv(file_path)
        
        # Required columns
        if 'distance' not in df.columns:
            raise ValueError("CSV file must contain 'distance' column")
        
        # Try different possible value column names
        value_col = None
        for col_name in ['value', 'values', 'data', 'z']:
            if col_name in df.columns:
                value_col = col_name
                break
        
        if value_col is None:
            raise ValueError("CSV file must contain a value column (value, values, data, or z)")
        
        # Extract basic data
        distances = df['distance'].values
        values = df[value_col].values
        
        # Extract coordinates if available
        coordinates_lonlat = None
        if 'longitude' in df.columns and 'latitude' in df.columns:
            coordinates_lonlat = np.column_stack([df['longitude'], df['latitude']])
        
        # Extract uncertainties if available
        uncertainties = None
        for unc_col in ['uncertainty', 'error', 'std']:
            if unc_col in df.columns:
                uncertainties = df[unc_col].values
                break
        
        return ProfileData(
            profile_distances=distances,
            profile_values=values,
            profile_uncertainties=uncertainties,
            profile_coordinates_lonlat=coordinates_lonlat,
            data_type='csv_loaded'
        )
    
    def _load_hdf5_data(self, file_path: Path) -> ProfileData:
        """Load profile data from HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            # Load required data
            distances = f['profile_distances'][:]
            values = f['profile_values'][:]
            
            # Load optional data
            uncertainties = None
            if 'profile_uncertainties' in f:
                uncertainties = f['profile_uncertainties'][:]
            
            coordinates_lonlat = None
            if 'profile_coordinates_lonlat' in f:
                coordinates_lonlat = f['profile_coordinates_lonlat'][:]
            
            # Load swath data if available
            swath_coordinates_lonlat = None
            swath_values = None
            swath_distances = None
            swath_offsets = None
            
            if 'swath_data' in f:
                swath_group = f['swath_data']
                if 'coordinates_lonlat' in swath_group:
                    swath_coordinates_lonlat = swath_group['coordinates_lonlat'][:]
                if 'values' in swath_group:
                    swath_values = swath_group['values'][:]
                if 'distances' in swath_group:
                    swath_distances = swath_group['distances'][:]
                if 'offsets' in swath_group:
                    swath_offsets = swath_group['offsets'][:]
            
            # Load metadata
            units = f.attrs.get('units', None)
            if isinstance(units, bytes):
                units = units.decode()
            
            description = f.attrs.get('description', None)
            if isinstance(description, bytes):
                description = description.decode()
        
        return ProfileData(
            profile_distances=distances,
            profile_values=values,
            profile_uncertainties=uncertainties,
            profile_coordinates_lonlat=coordinates_lonlat,
            swath_coordinates_lonlat=swath_coordinates_lonlat,
            swath_values=swath_values,
            swath_distances=swath_distances,
            swath_offsets=swath_offsets,
            units=units,
            description=description,
            data_type='hdf5_loaded'
        )
    
    def _load_json_data(self, file_path: Path) -> ProfileData:
        """Load profile data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to arrays
        distances = np.array(data['profile_distances'])
        values = np.array(data['profile_values'])
        
        uncertainties = None
        if 'profile_uncertainties' in data and data['profile_uncertainties'] is not None:
            uncertainties = np.array(data['profile_uncertainties'])
        
        coordinates_lonlat = None
        if 'profile_coordinates_lonlat' in data and data['profile_coordinates_lonlat'] is not None:
            coordinates_lonlat = np.array(data['profile_coordinates_lonlat'])
        
        # Load swath data if available
        swath_coordinates_lonlat = None
        swath_values = None
        swath_distances = None
        swath_offsets = None
        
        if 'swath_coordinates_lonlat' in data and data['swath_coordinates_lonlat'] is not None:
            swath_coordinates_lonlat = np.array(data['swath_coordinates_lonlat'])
        if 'swath_values' in data and data['swath_values'] is not None:
            swath_values = np.array(data['swath_values'])
        if 'swath_distances' in data and data['swath_distances'] is not None:
            swath_distances = np.array(data['swath_distances'])
        if 'swath_offsets' in data and data['swath_offsets'] is not None:
            swath_offsets = np.array(data['swath_offsets'])
        
        return ProfileData(
            profile_distances=distances,
            profile_values=values,
            profile_uncertainties=uncertainties,
            profile_coordinates_lonlat=coordinates_lonlat,
            swath_coordinates_lonlat=swath_coordinates_lonlat,
            swath_values=swath_values,
            swath_distances=swath_distances,
            swath_offsets=swath_offsets,
            units=data.get('units'),
            description=data.get('description'),
            data_type='json_loaded'
        )
    
    def load_all_datasets(self):
        """Load all datasets defined in configuration."""
        for dataset in self.config['datasets']:
            self.load_profile_data(dataset['name'])
    
    def plot_comparison(self, 
                       dataset_names: Optional[List[str]] = None,
                       **kwargs) -> plt.Figure:
        """
        Plot comparison of multiple profile datasets with swath options.
        
        Parameters:
        -----------
        dataset_names : list, optional
            List of dataset names to plot. If None, plots all datasets defined in configuration.
        
        **kwargs : dict
            Dataset-specific and plot-specific parameters.
            ------------------
            x_offset : float, default 0
                单个数据集横坐标整体偏移（单位与distance一致），可在datasets配置或kwargs中设置
            y_offset : float, default 0
                单个数据集纵坐标整体偏移，可在datasets配置或kwargs中设置

            Additional plotting parameters that override configuration settings.
            Supported parameters:
            
            Figure and Layout:
            ------------------
            figsize : list, default [14, 8]
                Figure size as [width, height] in inches
            xlabel : str, default 'Distance along profile (km)'
                X-axis label
            ylabel : str, default 'Value'
                Y-axis label
            xlim : list or tuple, optional
                Set x-axis range, e.g. [xmin, xmax]
            ylim : list or tuple, optional
                Set y-axis range, e.g. [ymin, ymax]
            title : str, default from config
                Plot title (overrides config title)
            grid : bool, default True
                Whether to show grid
            save_figure : bool, default False
                Whether to save the figure
            output_file : str, default 'multi_profile_comparison.png'
                Output filename for saved figure
            dpi : int, default 300
                DPI for saved figure
            
            Legend Settings:
            ----------------
            legend : bool, default True
                Whether to show legend
            legend_fontsize : int/str, default 9
                Legend font size
            legend_loc : str, default 'best'
                Legend location ('best', 'upper right', 'upper left', 'lower left', 
                'lower right', 'right', 'center left', 'center right', 'lower center', 
                'upper center', 'center')
            legend_ncol : int, default 1
                Number of columns in legend
            legend_nrow : int, optional
                Number of rows in legend (auto-calculates ncol if specified)
            legend_markerscale : float, default 1.0
                Size of legend markers relative to plot markers
            legend_columnspacing : float, default 2.0
                Spacing between legend columns
            legend_handletextpad : float, default 0.8
                Padding between legend handle and text
            legend_handlelength : float, default 2.0
                Length of legend handles
            legend_borderaxespad : float, default 0.5
                Whitespace inside legend border
            legend_frameon : bool, default True
                Whether to draw legend frame
            legend_fancybox : bool, default True
                Whether to use rounded corners for legend frame
            legend_shadow : bool, default False
                Whether to draw shadow behind legend
            legend_framealpha : float, default 0.8
                Legend frame transparency (0-1)
            legend_bbox_to_anchor : list, optional
                Custom legend position as [x, y] or [x, y, width, height]
            legend_edgecolor : str, default 'gray'
                Legend frame edge color
            legend_facecolor : str, default 'white'
                Legend frame face color
            legend_linewidth : float, default 0.5
                Legend frame line width
            
            Swath Point Settings:
            ---------------------
            swath_point_alpha : float, default 0.6
                Transparency of swath points (0-1)
            swath_point_size : float, default 10
                Size of swath points
            
            Swath Envelope Settings:
            ------------------------
            envelope_alpha : float, default 0.3
                Transparency of swath envelope fill (0-1)
            
        Dataset Configuration Override:
        -------------------------------
        You can also override individual dataset settings by passing dataset-specific
        parameters. These will apply to ALL datasets unless overridden in the dataset
        configuration itself:
        
            color : str
                Line/marker color for all datasets
            linestyle : str
                Line style ('-', '--', '-.', ':') for all datasets
            linewidth : float
                Line width for all datasets
            alpha : float
                Line transparency (0-1) for all datasets
            show_swath_points : bool
                Whether to show swath points for all datasets
            show_swath_envelope : bool
                Whether to show swath envelope for all datasets
            show_swath_median : bool
                Whether to show swath median line for all datasets
            show_swath_mean : bool
                Whether to show swath mean line for all datasets
            show_interpolated_line : bool
                Whether to show interpolated profile line for all datasets
            show_envelope_lines : bool, default True
                是否绘制包络线（上下边界线）
            envelope_line_color : str, optional
                包络线颜色，默认与主色一致
            envelope_line_style : str, default '-'
                包络线线型
            envelope_line_width : float, default 1.0
                包络线宽度
            envelope_method : str
                Envelope computation method ('minmax' or 'percentile') for all datasets
            envelope_percentiles : tuple
                Percentiles for envelope (e.g., (5, 95)) for all datasets
            swath_window_size : float
                Window size for swath statistics computation for all datasets
            swath_smooth_factor : float
                Smoothing factor for swath statistics (0-1) for all datasets
                
        Returns:
        --------
        plt.Figure
            Figure object
            
        Examples:
        --------
        Basic usage:
        >>> plotter = MultiSwathProfilePlotter('config.json')
        >>> fig = plotter.plot_comparison()
        
        Plot specific datasets with custom settings:
        >>> fig = plotter.plot_comparison(
        ...     dataset_names=['Elevation', 'Gravity'],
        ...     figsize=[16, 10],
        ...     legend_ncol=2,
        ...     legend_markerscale=2.0,
        ...     show_swath_envelope=True,
        ...     envelope_alpha=0.4
        ... )
        
        Custom legend positioning:
        >>> fig = plotter.plot_comparison(
        ...     legend_bbox_to_anchor=[1.02, 1.0],
        ...     legend_loc='upper left',
        ...     legend_fontsize=12,
        ...     legend_shadow=True
        ... )
        
        Override all datasets to show only swath data:
        >>> fig = plotter.plot_comparison(
        ...     show_interpolated_line=False,
        ...     show_swath_median=True,
        ...     show_swath_envelope=True,
        ...     swath_point_alpha=0.8
        ... )
        
        Save high-quality figure:
        >>> fig = plotter.plot_comparison(
        ...     save_figure=True,
        ...     output_file='comparison_high_res.png',
        ...     dpi=600,
        ...     figsize=[20, 12]
        ... )
            
        Notes:
        -----
        - Parameters passed via kwargs override configuration file settings
        - Dataset-specific parameters in configuration take precedence over kwargs
        - Use legend_nrow to automatically calculate ncol based on number of entries
        - Use legend_bbox_to_anchor for precise legend positioning outside plot area
        - Swath statistics are computed using sliding window approach with configurable overlap
        """
        # Use all datasets if none specified
        if dataset_names is None:
            dataset_names = [ds['name'] for ds in self.config['datasets']]
        
        # Load datasets if not already loaded
        for name in dataset_names:
            if name not in self.swath_profile_data:
                self.load_profile_data(name)
        
        # Get plot settings
        plot_settings = self.config.get('plot_settings', {})
        
        # Override with kwargs
        for key, value in kwargs.items():
            plot_settings[key] = value
        
        show_envelope_lines = plot_settings.get('show_envelope_lines', True)
        envelope_line_color = plot_settings.get('envelope_line_color', None)
        envelope_line_style = plot_settings.get('envelope_line_style', '-')
        envelope_line_width = plot_settings.get('envelope_line_width', 1.0)
        # envelope_upper_label = plot_settings.get('envelope_upper_label', None)
        # envelope_lower_label = plot_settings.get('envelope_lower_label', None)

        # Create figure
        figsize = plot_settings.get('figsize', [14, 8])
        
        with sci_plot_style():
            fig, ax = plt.subplots(figsize=figsize)
            
            # Track which datasets have been added to legend
            legend_added = set()
            
            # Plot each dataset
            for name in dataset_names:
                swath_data = self.swath_profile_data[name]
                dataset_config = next(ds for ds in self.config['datasets'] if ds['name'] == name)

                # Get offsets
                x_offset = dataset_config.get('x_offset', plot_settings.get('x_offset', 0))
                y_offset = dataset_config.get('y_offset', plot_settings.get('y_offset', 0))
                # Get plotting parameters
                color = dataset_config.get('color', 'blue')
                linestyle = dataset_config.get('linestyle', '-')
                linewidth = dataset_config.get('linewidth', 2)
                alpha = dataset_config.get('alpha', 1.0)
                
                # Swath visualization options
                show_swath_points = dataset_config.get('show_swath_points', False)
                show_swath_envelope = dataset_config.get('show_swath_envelope', False)
                show_swath_median = dataset_config.get('show_swath_median', False)
                show_swath_mean = dataset_config.get('show_swath_mean', False)
                show_interpolated_line = dataset_config.get('show_interpolated_line', True)
                
                # Swath processing parameters
                swath_window_size = dataset_config.get('swath_window_size', None)
                swath_smooth_factor = dataset_config.get('swath_smooth_factor', 0.5)
                swath_min_points = dataset_config.get('swath_min_points', 0)
                swath_max_std = dataset_config.get('swath_max_std', None)
                # print(swath_min_points, swath_max_std)

                # Determine legend label for this dataset
                legend_label = name if name not in legend_added else None
                
                # Plot swath points if requested and available
                if (show_swath_points and 
                    swath_data.swath_distances is not None and 
                    swath_data.swath_values is not None):
                    
                    swath_point_alpha = plot_settings.get('swath_point_alpha', 0.6)
                    swath_point_size = plot_settings.get('swath_point_size', 10)
                    
                    valid_mask = ~np.isnan(swath_data.swath_values)
                    ax.scatter(
                        swath_data.swath_distances[valid_mask] + x_offset, 
                        swath_data.swath_values[valid_mask] + y_offset,
                        c=color, alpha=swath_point_alpha, s=swath_point_size,
                        label=legend_label, marker='.'
                    )
                    if legend_label:
                        legend_added.add(name)
                        legend_label = None  # Don't use label again for this dataset
                
                # Plot swath envelope if requested and available
                # 包络绘制
                if (show_swath_envelope and 
                    swath_data.swath_distances is not None and 
                    swath_data.swath_values is not None):
                    
                    envelope_method = dataset_config.get('envelope_method', 'minmax')
                    envelope_percentiles = dataset_config.get('envelope_percentiles', (5, 95))
                    envelope_alpha = plot_settings.get('envelope_alpha', 0.3)
                    
                    upper_env, lower_env = swath_data.compute_swath_envelope(
                        method=envelope_method,
                        percentiles=envelope_percentiles,
                        window_size=swath_window_size,
                        min_points=swath_min_points,
                        max_std=swath_max_std
                    )
                    env_distances, _ = swath_data.compute_swath_median(
                        window_size=swath_window_size,
                        smooth_factor=swath_smooth_factor,
                        min_points=swath_min_points,
                        max_std=swath_max_std
                    )
                    if upper_env is not None and lower_env is not None and env_distances is not None:
                        # 填充包络
                        ax.fill_between(
                            env_distances + x_offset, lower_env + y_offset, upper_env + y_offset,
                            alpha=envelope_alpha, color=color,
                            label=legend_label
                        )
                        if legend_label:
                            legend_added.add(name)
                            legend_label = None
                        # 包络线
                        if show_envelope_lines:
                            line_color = envelope_line_color if envelope_line_color else color
                            # 上边界线
                            ax.plot(env_distances + x_offset, upper_env + y_offset, 
                                    color=line_color, linestyle=envelope_line_style, 
                                    alpha=0.7, linewidth=envelope_line_width,
                                    # label=envelope_upper_label
                                    )
                            # 下边界线
                            ax.plot(env_distances + x_offset, lower_env + y_offset, 
                                    color=line_color, linestyle=envelope_line_style, 
                                    alpha=0.7, linewidth=envelope_line_width,
                                    # label=envelope_lower_label
                                    )
                
                # Plot swath median line if requested and available
                if (show_swath_median and 
                    swath_data.swath_distances is not None and 
                    swath_data.swath_values is not None):
                    
                    median_distances, median_values = swath_data.compute_swath_median(
                        window_size=swath_window_size,
                        smooth_factor=swath_smooth_factor,
                        min_points=swath_min_points,
                        max_std=swath_max_std
                    )
                    
                    if median_distances is not None and median_values is not None:
                        valid_mask = ~np.isnan(median_values)
                        ax.plot(
                            median_distances[valid_mask] + x_offset, median_values[valid_mask] + y_offset,
                            color=color, linestyle='-', linewidth=linewidth + 0.5,
                            alpha=alpha, label=legend_label
                        )
                        if legend_label:
                            legend_added.add(name)
                            legend_label = None
                
                # Plot swath mean line if requested and available
                if (show_swath_mean and 
                    swath_data.swath_distances is not None and 
                    swath_data.swath_values is not None):
                    
                    mean_distances, mean_values = swath_data.compute_swath_mean(
                        window_size=swath_window_size,
                        smooth_factor=swath_smooth_factor,
                        min_points=swath_min_points,
                        max_std=swath_max_std
                    )
                    
                    if mean_distances is not None and mean_values is not None:
                        valid_mask = ~np.isnan(mean_values)
                        ax.plot(
                            mean_distances[valid_mask] + x_offset, mean_values[valid_mask] + y_offset,
                            color=color, linestyle='-.', linewidth=linewidth,
                            alpha=alpha, label=legend_label
                        )
                        if legend_label:
                            legend_added.add(name)
                            legend_label = None
                
                # Plot interpolated profile line if requested
                if show_interpolated_line and swath_data.profile_values is not None:
                    valid_mask = ~np.isnan(swath_data.profile_values)
                    ax.plot(
                        swath_data.profile_distances[valid_mask] + x_offset, 
                        swath_data.profile_values[valid_mask] + y_offset,
                        color=color, linestyle=linestyle, linewidth=linewidth,
                        alpha=alpha, label=legend_label
                    )
                    if legend_label:
                        legend_added.add(name)
                        legend_label = None
                
                # Plot uncertainty bands if available and interpolated line is shown
                if (show_interpolated_line and 
                    swath_data.profile_uncertainties is not None and
                    swath_data.profile_values is not None):
                    
                    valid_mask = ~np.isnan(swath_data.profile_values) & ~np.isnan(swath_data.profile_uncertainties)
                    distances = swath_data.profile_distances[valid_mask]
                    values = swath_data.profile_values[valid_mask]
                    uncertainties = swath_data.profile_uncertainties[valid_mask]
                    
                    # Don't add uncertainty bands to legend
                    ax.fill_between(
                        distances + x_offset,
                        values - uncertainties + y_offset,
                        values + uncertainties + y_offset,
                        alpha=0.2, color=color
                    )
            
            # Set labels and title
            xlabel = plot_settings.get('xlabel', 'Distance along profile (km)')
            ylabel = plot_settings.get('ylabel', 'Value')
            title = self.config.get('title', 'Multi-Dataset Profile Comparison')
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

            xlim = plot_settings.get('xlim', None)
            ylim = plot_settings.get('ylim', None)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            
            # Grid and legend
            if plot_settings.get('grid', True):
                ax.grid(True, alpha=0.3)
            
            if plot_settings.get('legend', True):
                # Get legend parameters with defaults
                legend_fontsize = plot_settings.get('legend_fontsize', 9)
                legend_loc = plot_settings.get('legend_loc', 'best')
                legend_ncol = plot_settings.get('legend_ncol', 1)
                legend_nrow = plot_settings.get('legend_nrow', None)  # If specified, auto-calculate ncol
                legend_markerscale = plot_settings.get('legend_markerscale', 1.0)
                legend_columnspacing = plot_settings.get('legend_columnspacing', 2.0)
                legend_handletextpad = plot_settings.get('legend_handletextpad', 0.8)
                legend_frameon = plot_settings.get('legend_frameon', True)
                legend_fancybox = plot_settings.get('legend_fancybox', True)
                legend_shadow = plot_settings.get('legend_shadow', False)
                legend_framealpha = plot_settings.get('legend_framealpha', 0.8)
                legend_bbox_to_anchor = plot_settings.get('legend_bbox_to_anchor', None)
                legend_borderaxespad = plot_settings.get('legend_borderaxespad', 0.5)
                legend_handlelength = plot_settings.get('legend_handlelength', 2.0)
                
                # Calculate ncol from nrow if specified
                if legend_nrow is not None:
                    # Get number of legend entries
                    handles, labels = ax.get_legend_handles_labels()
                    n_entries = len(handles)
                    if n_entries > 0:
                        legend_ncol = int(np.ceil(n_entries / legend_nrow))
                
                # Create legend parameters dictionary
                legend_kwargs = {
                    'fontsize': legend_fontsize,
                    'loc': legend_loc,
                    'ncol': legend_ncol,
                    'markerscale': legend_markerscale,
                    'columnspacing': legend_columnspacing,
                    'handletextpad': legend_handletextpad,
                    'frameon': legend_frameon,
                    'fancybox': legend_fancybox,
                    'shadow': legend_shadow,
                    'framealpha': legend_framealpha,
                    'borderaxespad': legend_borderaxespad,
                    'handlelength': legend_handlelength
                }
                
                # Add bbox_to_anchor if specified
                if legend_bbox_to_anchor is not None:
                    legend_kwargs['bbox_to_anchor'] = tuple(legend_bbox_to_anchor)
                
                # Create the legend
                legend = ax.legend(**legend_kwargs)
                
                # Additional legend customization
                if legend_frameon:
                    # Customize legend frame
                    legend_edgecolor = plot_settings.get('legend_edgecolor', 'gray')
                    legend_facecolor = plot_settings.get('legend_facecolor', 'white')
                    legend_linewidth = plot_settings.get('legend_linewidth', 0.5)
                    
                    legend.get_frame().set_edgecolor(legend_edgecolor)
                    legend.get_frame().set_facecolor(legend_facecolor)
                    legend.get_frame().set_linewidth(legend_linewidth)
            
            plt.tight_layout()
            
            # Save figure if requested
            if plot_settings.get('save_figure', False):
                output_file = plot_settings.get('output_file', 'multi_profile_comparison.png')
                dpi = plot_settings.get('dpi', 300)
                plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
                print(f"Figure saved to: {output_file}")
            
            return fig
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about loaded datasets."""
        info = {}
        
        for name, swath_data in self.swath_profile_data.items():
            profile_info = {
                'n_profile_points': len(swath_data.profile_distances) if swath_data.profile_distances is not None else 0,
                'has_swath_data': swath_data.swath_values is not None,
                'units': swath_data.units
            }
            
            # Profile data statistics
            if swath_data.profile_values is not None:
                valid_profile = swath_data.profile_values[~np.isnan(swath_data.profile_values)]
                if len(valid_profile) > 0:
                    profile_info.update({
                        'profile_value_range': [float(np.min(valid_profile)), float(np.max(valid_profile))],
                        'profile_mean': float(np.mean(valid_profile)),
                        'profile_std': float(np.std(valid_profile))
                    })
            
            # Swath data statistics
            if swath_data.swath_values is not None:
                valid_swath = swath_data.swath_values[~np.isnan(swath_data.swath_values)]
                if len(valid_swath) > 0:
                    profile_info.update({
                        'n_swath_points': len(valid_swath),
                        'swath_value_range': [float(np.min(valid_swath)), float(np.max(valid_swath))],
                        'swath_mean': float(np.mean(valid_swath)),
                        'swath_std': float(np.std(valid_swath))
                    })
            
            info[name] = profile_info
        
        return info

# Convenience functions
def plot_multiple_swath_profiles_from_config(config_file: Union[str, Path, Dict],
                                           dataset_names: Optional[List[str]] = None,
                                           **kwargs) -> plt.Figure:
    """
    Quick function to plot multiple profiles with swath data from configuration.
    
    Parameters:
    -----------
    config_file : str, Path, or dict
        Configuration file or dictionary
    dataset_names : list, optional
        Datasets to plot
    **kwargs : dict
        Additional plot parameters
        
    Returns:
    --------
    plt.Figure
        Figure object
    """
    plotter = MultiSwathProfilePlotter(config_file)
    return plotter.plot_comparison(dataset_names, **kwargs)

def plot_multiple_swath_profiles_simple(profile_files: List[str],
                                       names: Optional[List[str]] = None,
                                       colors: Optional[List[str]] = None,
                                       title: str = "Profile Comparison with Swath Data",
                                       show_swath_points: bool = True,
                                       show_swath_envelope: bool = True,
                                       show_swath_median: bool = True,
                                       show_interpolated_line: bool = False,
                                       **kwargs) -> plt.Figure:
    """
    Simple function to plot multiple profile files with swath options.
    
    Parameters:
    -----------
    profile_files : list
        List of profile data files
    names : list, optional
        Names for each dataset
    colors : list, optional
        Colors for each dataset
    title : str
        Plot title
    show_swath_points : bool
        Show swath points for all datasets
    show_swath_envelope : bool
        Show swath envelope for all datasets
    show_swath_median : bool
        Show swath median line for all datasets
    show_interpolated_line : bool
        Show interpolated profile line for all datasets
    **kwargs : dict
        Additional plot parameters
        
    Returns:
    --------
    plt.Figure
        Figure object
    """
    if names is None:
        names = [f"Dataset {i+1}" for i in range(len(profile_files))]
    
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Create simple configuration
    config = {
        "title": title,
        "datasets": [],
        "plot_settings": {
            "figsize": kwargs.get('figsize', [14, 8]),
            "grid": kwargs.get('grid', True),
            "legend": kwargs.get('legend', True),
            "swath_point_alpha": kwargs.get('swath_point_alpha', 0.6),
            "swath_point_size": kwargs.get('swath_point_size', 10),
            "envelope_alpha": kwargs.get('envelope_alpha', 0.3)
        }
    }
    
    for i, (file_path, name) in enumerate(zip(profile_files, names)):
        config["datasets"].append({
            "name": name,
            "file": file_path,
            "format": "auto",
            "color": colors[i % len(colors)],
            "linestyle": "-",
            "linewidth": 2,
            "show_swath_points": show_swath_points,
            "show_swath_envelope": show_swath_envelope,
            "show_swath_median": show_swath_median,
            "show_interpolated_line": show_interpolated_line
        })
    
    return plot_multiple_swath_profiles_from_config(config, **kwargs)

def create_example_swath_config():
    """Create an example configuration file for swath profile plotting."""
    config = {
        "title": "Multi-Dataset Profile Comparison with Swath Data",
        "datasets": [
            {
                "name": "Topography",
                "file": "elevation_profile.hdf5",
                "format": "hdf5",
                "color": "blue",
                "linestyle": "-",
                "linewidth": 2.5,
                "show_swath_points": True,
                "show_swath_envelope": True,
                "show_swath_median": True,
                "show_swath_mean": False,
                "show_interpolated_line": False,
                "envelope_method": "minmax",
                "swath_window_size": None,  # Auto-determine
                "swath_smooth_factor": 0.5  # 50% overlap
            },
            {
                "name": "Gravity Anomaly",
                "file": "gravity_profile.json", 
                "format": "json",
                "color": "red",
                "linestyle": "--",
                "linewidth": 2,
                "show_swath_points": False,
                "show_swath_envelope": True,
                "show_swath_median": True,
                "show_swath_mean": False,
                "show_interpolated_line": True,
                "envelope_method": "percentile",
                "envelope_percentiles": [10, 90],
                "swath_window_size": 2.0,  # Fixed 2km window
                "swath_smooth_factor": 0.3  # 30% overlap for smoother curves
            },
            {
                "name": "Magnetic Field",
                "file": "magnetic_profile.hdf5",
                "format": "hdf5", 
                "color": "green",
                "linestyle": "-.",
                "linewidth": 1.5,
                "show_swath_points": True,
                "show_swath_envelope": False,
                "show_swath_median": True,
                "show_swath_mean": True,  # Show both median and mean
                "show_interpolated_line": False,
                "swath_window_size": None,
                "swath_smooth_factor": 0.4
            }
        ],
        "plot_settings": {
            "figsize": [16, 8],
            "xlabel": "Distance along profile (km)",
            "ylabel": "Value",
            "grid": True,
            "legend": True,
            "save_figure": True,
            "output_file": "multi_swath_profile_comparison.png",
            "dpi": 300,
            "swath_point_alpha": 0.6,
            "swath_point_size": 8,
            "envelope_alpha": 0.25
        }
    }
    
    with open('example_swath_profile_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Example swath profile configuration saved to: example_swath_profile_config.json")
    return config

if __name__ == "__main__":
    # Create example configuration
    example_config = create_example_swath_config()
    
    # Example usage
    print("\nExample usage:")
    print("1. Using configuration file:")
    print("   plotter = MultiSwathProfilePlotter('config.json')")
    print("   fig = plotter.plot_comparison()")
    
    print("\n2. Simple plotting with swath options:")
    print("   fig = plot_multiple_swath_profiles_simple(")
    print("       ['file1.hdf5', 'file2.json'], ")
    print("       names=['Dataset 1', 'Dataset 2'],")
    print("       show_swath_points=True,")
    print("       show_swath_envelope=True,")
    print("       show_swath_median=True")
    print("   )")
    
    print("\n3. Configuration dictionary:")
    print("   fig = plot_multiple_swath_profiles_from_config(config_dict)")

    # 方法1：使用配置文件
    plotter = MultiSwathProfilePlotter('swath_config.json')
    fig = plotter.plot_comparison()

    # 方法2：简单快速绘制，只显示swath数据
    fig = plot_multiple_swath_profiles_simple(
        profile_files=['topo.hdf5', 'gravity.json'],
        names=['地形', '重力'],
        show_swath_points=True,
        show_swath_envelope=True,
        show_swath_median=True,
        show_interpolated_line=False  # 不显示插值线
    )

    # 方法3：混合显示 - 一些显示swath，一些显示插值
    config = {
        "datasets": [
            {
                "name": "地形", "file": "topo.hdf5",
                "show_swath_median": True,
                "show_interpolated_line": False
            },
            {
                "name": "重力", "file": "gravity.hdf5", 
                "show_swath_envelope": True,
                "show_interpolated_line": True
            }
        ]
    }
    fig = plot_multiple_swath_profiles_from_config(config)

    # 4. 获取数据集信息
    plotter = MultiSwathProfilePlotter('config.json')
    info = plotter.get_dataset_info()
    print(f"数据集信息: {info}")