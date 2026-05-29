"""
Regional Ramp Removal Module

This module provides comprehensive tools for fitting and removing regional trends
from spatial data. It supports various polynomial surface fitting methods and
can handle both gridded and scattered point data.

Key Features:
- Multiple ramp types: constant, linear, quadratic surfaces
- Flexible geographic masking (exclude/include regions)
- Support for both gridded matrices and scattered point data
- Coordinate system integration with CSI
- Robust outlier handling
- Comprehensive visualization tools

Classes:
    RegionalRampRemover: Main class for regional trend analysis and removal
    RampConfiguration: Configuration container for ramp fitting parameters
    RampResult: Data container for ramp fitting results

Example:
    >>> from eqtools.statUtils import RegionalRampRemover
    >>> # Initialize with coordinate system
    >>> remover = RegionalRampRemover(lon0=120.0, lat0=30.0, utmzone=51)
    >>> 
    >>> # Remove linear trend excluding central region
    >>> detrended_data, ramp_info = remover.remove_ramp_from_matrix(
    ...     data=elevation_matrix,
    ...     x_coords=lon_grid,
    ...     y_coords=lat_grid,
    ...     coord_type='lonlat',
    ...     exclude_box=(120.5, 122.5, 30.5, 32.5),
    ...     ramp_type='linear'
    ... )

Authors:
    Kefeng He

Version:
    1.0.0

Last Updated:
    2025-08-07
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional, Any, Literal
from dataclasses import dataclass, field
import warnings
import json

# Scientific computing imports
from scipy import interpolate
from scipy.spatial import cKDTree

# Geospatial imports
from shapely.geometry import Point, Polygon, box
import geopandas as gpd

# CSI imports
from csi.SourceInv import SourceInv

# Local imports
from ...plottools import sci_plot_style, set_degree_formatter


@dataclass
class RampConfiguration:
    """Configuration class for regional ramp removal parameters."""
    
    # Ramp fitting parameters
    ramp_type: Literal['constant', 'linear', 'quadratic'] = 'linear'
    min_points_required: int = 20
    use_robust_fitting: bool = False
    robust_threshold: float = 2.0  # For robust fitting outlier detection
    
    # Reference point configuration
    reference_point_mode: Literal['mean', 'none', 'user'] = 'mean'
    reference_point: Optional[Tuple[float, float]] = None
    reference_coord_type: str = 'lonlat'  # 'lonlat' or 'xy'
    
    # Geographic masking
    exclude_regions: List[Dict[str, Any]] = field(default_factory=list)
    include_regions: List[Dict[str, Any]] = field(default_factory=list)
    coord_type: str = 'lonlat'  # 'lonlat' or 'xy'
    
    # Data preprocessing
    remove_outliers_before_fitting: bool = False
    outlier_method: Literal['zscore', 'iqr', 'absolute'] = 'zscore'
    outlier_threshold: float = 3.0
    
    # Interpolation for scattered data
    interpolation_method: str = 'linear'  # 'linear', 'cubic', 'nearest'
    interpolation_grid_spacing: Optional[float] = None  # Auto if None

    def set_reference_point(self, 
                           mode: Literal['mean', 'none', 'user'],
                           point: Optional[Tuple[float, float]] = None,
                           coord_type: str = 'lonlat') -> None:
        """
        Set reference point for coordinate centering.
        
        Parameters:
        -----------
        mode : str
            Reference point mode:
            - 'mean': Use center of fit points (default)
            - 'none': No coordinate centering (use original coordinates)
            - 'user': Use user-provided reference point
        point : tuple, optional
            Reference point coordinates (required if mode='user')
        coord_type : str
            Coordinate type for reference point
            
        Examples:
        --------
        # Use mean of fit points as reference
        config.set_reference_point('mean')
        
        # Use specific location as reference
        config.set_reference_point('user', (121.5, 32.0), 'lonlat')
        
        # No coordinate centering
        config.set_reference_point('none')
        """
        self.reference_point_mode = mode
        self.reference_point = point
        self.reference_coord_type = coord_type
        
        if mode == 'user' and point is None:
            raise ValueError("Reference point must be provided when mode='user'")

    def add_exclude_box(self, 
                       min_x: float, max_x: float, 
                       min_y: float, max_y: float,
                       coord_type: str = None) -> None:
        """
        Add a rectangular exclusion region.
        
        Parameters:
        -----------
        min_x, max_x : float
            X coordinate bounds (longitude or x)
        min_y, max_y : float
            Y coordinate bounds (latitude or y)
        coord_type : str, optional
            Coordinate type. Uses self.coord_type if None.
        """
        coord_type = coord_type or self.coord_type
        self.exclude_regions.append({
            'type': 'box',
            'bounds': (min_x, max_x, min_y, max_y),
            'coord_type': coord_type
        })
    
    def add_exclude_polygon(self, 
                           vertices: List[Tuple[float, float]],
                           coord_type: str = None) -> None:
        """
        Add a polygonal exclusion region.
        
        Parameters:
        -----------
        vertices : list
            List of (x, y) or (lon, lat) vertices defining the polygon
        coord_type : str, optional
            Coordinate type. Uses self.coord_type if None.
        """
        coord_type = coord_type or self.coord_type
        self.exclude_regions.append({
            'type': 'polygon',
            'vertices': vertices,
            'coord_type': coord_type
        })
    
    def add_exclude_circle(self, 
                          center: Tuple[float, float], 
                          radius: float,
                          coord_type: str = None) -> None:
        """
        Add a circular exclusion region.
        
        Parameters:
        -----------
        center : tuple
            Center point (x, y) or (lon, lat)
        radius : float
            Radius in km (for lonlat) or same units as coordinates (for xy)
        coord_type : str, optional
            Coordinate type. Uses self.coord_type if None.
        """
        coord_type = coord_type or self.coord_type
        self.exclude_regions.append({
            'type': 'circle',
            'center': center,
            'radius': radius,
            'coord_type': coord_type
        })
    
    def clear_regions(self) -> None:
        """Clear all defined regions."""
        self.exclude_regions.clear()
        self.include_regions.clear()


@dataclass
class RampResult:
    """Container for regional ramp removal results."""
    
    # Input data info
    original_data_shape: Tuple[int, ...] = None
    n_input_points: int = 0
    coord_type: str = 'lonlat'
    
    # Reference point info
    reference_point_mode: str = 'mean'
    reference_point_used: Optional[Tuple[float, float]] = None
    reference_coord_type: str = 'lonlat'
    
    # Fitting results
    ramp_type: str = 'linear'
    coefficients: List[float] = field(default_factory=list)
    n_fit_points: int = 0
    n_excluded_points: int = 0
    fit_residual_rms: float = 0.0
    fit_r_squared: float = 0.0
    
    # Ramp surface info
    ramp_value_range: Tuple[float, float] = (0.0, 0.0)
    original_data_range: Tuple[float, float] = (0.0, 0.0)
    detrended_data_range: Tuple[float, float] = (0.0, 0.0)
    
    # Processing info
    regions_used: List[Dict[str, Any]] = field(default_factory=list)
    outliers_removed: bool = False
    outlier_info: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    condition_number: float = 0.0
    fit_uncertainty: List[float] = field(default_factory=list)
    
    def get_ramp_equation(self) -> str:
        """Get human-readable equation for the fitted ramp."""
        ref_info = ""
        if self.reference_point_mode != 'none' and self.reference_point_used:
            ref_x, ref_y = self.reference_point_used
            ref_info = f" (relative to {ref_x:.3f}, {ref_y:.3f})"
        
        if self.ramp_type == 'constant':
            return f"z = {self.coefficients[0]:.4f}{ref_info}"
        elif self.ramp_type == 'linear':
            a, b, c = self.coefficients
            return f"z = {a:.6f}*(x-x0) + {b:.6f}*(y-y0) + {c:.4f}{ref_info}"
        elif self.ramp_type == 'quadratic':
            a, b, c, d, e, f = self.coefficients
            return (f"z = {a:.8f}*(x-x0)^2 + {b:.8f}*(y-y0)^2 + {c:.8f}*(x-x0)*(y-y0) + "
                   f"{d:.6f}*(x-x0) + {e:.6f}*(y-y0) + {f:.4f}{ref_info}")
        else:
            return "Unknown ramp type"
    
    def print_summary(self) -> None:
        """Print a summary of the ramp removal results."""
        print("=" * 60)
        print("REGIONAL RAMP REMOVAL SUMMARY")
        print("=" * 60)
        print(f"Ramp type: {self.ramp_type}")
        print(f"Reference point mode: {self.reference_point_mode}")
        if self.reference_point_used:
            print(f"Reference point: ({self.reference_point_used[0]:.4f}, {self.reference_point_used[1]:.4f})")
        print(f"Equation: {self.get_ramp_equation()}")
        print(f"Fit points: {self.n_fit_points} (excluded: {self.n_excluded_points})")
        print(f"RMS residual: {self.fit_residual_rms:.4f}")
        print(f"R-squared: {self.fit_r_squared:.4f}")
        print(f"Condition number: {self.condition_number:.2e}")
        print(f"Ramp range: [{self.ramp_value_range[0]:.4f}, {self.ramp_value_range[1]:.4f}]")
        print(f"Original data range: [{self.original_data_range[0]:.4f}, {self.original_data_range[1]:.4f}]")
        print(f"Detrended data range: [{self.detrended_data_range[0]:.4f}, {self.detrended_data_range[1]:.4f}]")
        if self.outliers_removed:
            print(f"Outliers removed: {self.outlier_info.get('n_outliers', 0)}")
        print("=" * 60)


class RegionalRampRemover(SourceInv):
    """
    Comprehensive tool for regional ramp/trend removal from spatial data.
    
    This class provides methods to fit and remove polynomial surfaces from
    spatial data while allowing flexible geographic masking and reference point
    configuration for improved numerical stability.
    
    Attributes:
    -----------
    config : RampConfiguration
        Configuration object containing analysis parameters
        
    Example:
    --------
    >>> remover = RegionalRampRemover(
    ...     name='ramp_analysis',
    ...     lon0=120.0, lat0=30.0, utmzone=51
    ... )
    >>> 
    >>> # Configure exclusion region
    >>> config = RampConfiguration()
    >>> config.add_exclude_box(120.5, 122.5, 30.5, 32.5)  # Central region
    >>> remover.configure(config)
    >>> 
    >>> # Remove linear trend from matrix data
    >>> detrended, result = remover.remove_ramp_from_matrix(
    ...     data_matrix, lon_coords, lat_coords, coord_type='lonlat'
    ... )
    """
    
    def __init__(self, name='RegionalRampRemover', lon0=None, lat0=None, utmzone=None, 
                 ellps='WGS84', config: Optional[RampConfiguration] = None):
        """
        Initialize RegionalRampRemover with CSI coordinate system.
        
        Parameters:
        -----------
        name : str
            Name for the remover
        lon0 : float
            Reference longitude for UTM projection
        lat0 : float
            Reference latitude for UTM projection
        utmzone : int
            UTM zone number
        ellps : str
            Ellipsoid definition
        config : RampConfiguration, optional
            Initial configuration. If None, uses default settings.
        """
        # Initialize parent SourceInv class
        super().__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0)
        
        self.config = config or RampConfiguration()
    
    def configure(self, config: RampConfiguration) -> None:
        """
        Update remover configuration.
        
        Parameters:
        -----------
        config : RampConfiguration
            New configuration object
        """
        self.config = config
    
    def remove_ramp_from_matrix(self,
                               data: np.ndarray,
                               x_coords: np.ndarray,
                               y_coords: np.ndarray,
                               coord_type: str = 'lonlat',
                               ramp_type: str = None,
                               exclude_box: Tuple[float, float, float, float] = None,
                               reference_point_mode: str = None,
                               reference_point: Tuple[float, float] = None,
                               return_result_object: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, RampResult]]:
        """
        Remove regional ramp from gridded matrix data.
        
        Parameters:
        -----------
        data : np.ndarray
            2D data matrix to detrend
        x_coords : np.ndarray
            X-coordinate array (longitude or x in km)
        y_coords : np.ndarray  
            Y-coordinate array (latitude or y in km)
        coord_type : str
            'lonlat' if coordinates are in degrees, 'xy' if in km
        ramp_type : str, optional
            Type of surface to fit: 'constant', 'linear', 'quadratic'.
            If None, uses config setting.
        exclude_box : tuple, optional
            Geographic box to exclude: (min_x, max_x, min_y, max_y).
            If provided, temporarily overrides config regions.
        reference_point_mode : str, optional
            Reference point mode: 'mean', 'none', 'user'. Overrides config if provided.
        reference_point : tuple, optional
            User reference point (lon, lat) or (x, y). Required if mode='user'.
        return_result_object : bool
            Whether to return detailed RampResult object
            
        Returns:
        --------
        np.ndarray or Tuple[np.ndarray, RampResult]
            Detrended data matrix, optionally with detailed results
        """
        # Use provided parameters or fall back to config
        ramp_type = ramp_type or self.config.ramp_type
        ref_mode = reference_point_mode or self.config.reference_point_mode
        ref_point = reference_point or self.config.reference_point
        
        # Validate inputs
        if data.shape != (len(y_coords), len(x_coords)) and x_coords.ndim == 1:
            if len(x_coords) != data.shape[1] or len(y_coords) != data.shape[0]:
                raise ValueError("Coordinate arrays don't match data dimensions")
        
        # Create coordinate grids if 1D arrays provided
        if x_coords.ndim == 1 and y_coords.ndim == 1:
            X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='xy')
        else:
            X_grid, Y_grid = x_coords, y_coords
        
        # Convert coordinates to xy (km) for fitting if needed
        if coord_type == 'lonlat':
            X_km, Y_km = self._convert_lonlat_grid_to_xy(X_grid, Y_grid)
        else:
            X_km, Y_km = X_grid, Y_grid
        
        # Flatten arrays for processing
        x_flat = X_km.flatten()
        y_flat = Y_km.flatten()
        data_flat = data.flatten()
        
        # Remove NaN values
        valid_mask = ~np.isnan(data_flat)
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        data_valid = data_flat[valid_mask]
        
        if len(data_valid) == 0:
            warnings.warn("No valid data points found")
            if return_result_object:
                return data, RampResult()
            return data
        
        # Apply geographic masking
        if exclude_box is not None:
            # Use temporary exclude box
            fit_mask = self._apply_exclude_box_mask(x_valid, y_valid, exclude_box, coord_type)
        else:
            # Use configured regions
            fit_mask = self._apply_geographic_masking(x_valid, y_valid, coord_type)
        
        x_fit = x_valid[fit_mask]
        y_fit = y_valid[fit_mask]
        data_fit = data_valid[fit_mask]
        
        # Apply outlier removal if configured
        if self.config.remove_outliers_before_fitting:
            outlier_mask = self._remove_outliers(data_fit)
            x_fit = x_fit[outlier_mask]
            y_fit = y_fit[outlier_mask]
            data_fit = data_fit[outlier_mask]
            
            outlier_info = {
                'method': self.config.outlier_method,
                'threshold': self.config.outlier_threshold,
                'n_outliers': len(data_fit) - np.sum(outlier_mask)
            }
        else:
            outlier_info = {}
        
        # Check minimum points requirement
        if len(data_fit) < self.config.min_points_required:
            warnings.warn(f"Insufficient data points ({len(data_fit)}) for {ramp_type} "
                         f"ramp fitting (requires >= {self.config.min_points_required})")
            if return_result_object:
                return data, RampResult()
            return data
        
        # Determine and apply reference point
        ref_point_xy, actual_ref_point = self._determine_reference_point(
            x_fit, y_fit, ref_mode, ref_point, coord_type
        )
        
        # Center coordinates relative to reference point
        x_fit_centered = x_fit - ref_point_xy[0]
        y_fit_centered = y_fit - ref_point_xy[1]
        x_valid_centered = x_valid - ref_point_xy[0]
        y_valid_centered = y_valid - ref_point_xy[1]
        
        # Fit ramp surface using centered coordinates
        try:
            coeffs, fit_stats = self._fit_ramp_surface(x_fit_centered, y_fit_centered, data_fit, ramp_type)
        except Exception as e:
            warnings.warn(f"Failed to fit {ramp_type} ramp: {e}")
            if return_result_object:
                return data, RampResult()
            return data
        
        # Evaluate ramp surface on full grid using centered coordinates
        X_km_centered = X_km - ref_point_xy[0]
        Y_km_centered = Y_km - ref_point_xy[1]
        ramp_surface = self._evaluate_ramp_surface(X_km_centered, Y_km_centered, ramp_type, coeffs)
        
        # Remove ramp from data
        detrended_data = data - ramp_surface
        
        # Create result object if requested
        if return_result_object:
            result = RampResult(
                original_data_shape=data.shape,
                n_input_points=len(data_valid),
                coord_type=coord_type,
                reference_point_mode=ref_mode,
                reference_point_used=actual_ref_point,
                reference_coord_type=self.config.reference_coord_type,
                ramp_type=ramp_type,
                coefficients=coeffs.tolist(),
                n_fit_points=len(data_fit),
                n_excluded_points=len(data_valid) - len(data_fit),
                fit_residual_rms=fit_stats['rms_residual'],
                fit_r_squared=fit_stats['r_squared'],
                ramp_value_range=(float(np.min(ramp_surface)), float(np.max(ramp_surface))),
                original_data_range=(float(np.min(data_valid)), float(np.max(data_valid))),
                detrended_data_range=(float(np.min(detrended_data[valid_mask.reshape(data.shape)])), 
                                    float(np.max(detrended_data[valid_mask.reshape(data.shape)]))),
                regions_used=self._get_current_regions_info(exclude_box, coord_type),
                outliers_removed=self.config.remove_outliers_before_fitting,
                outlier_info=outlier_info,
                condition_number=fit_stats.get('condition_number', 0.0),
                fit_uncertainty=fit_stats.get('parameter_uncertainty', [])
            )
            
            return detrended_data, result
        else:
            return detrended_data
    
    def remove_ramp_from_points(self,
                               points: np.ndarray,
                               values: np.ndarray,
                               coord_type: str = 'lonlat',
                               ramp_type: str = None,
                               exclude_regions: List[Dict[str, Any]] = None,
                               reference_point_mode: str = None,
                               reference_point: Tuple[float, float] = None,
                               return_result_object: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, RampResult]]:
        """
        Remove regional ramp from scattered point data.
        
        Parameters:
        -----------
        points : np.ndarray
            Point coordinates (n_points, 2) - (lon,lat) or (x,y)
        values : np.ndarray
            Values at each point
        coord_type : str
            'lonlat' if coordinates are in degrees, 'xy' if in km
        ramp_type : str, optional
            Type of surface to fit. If None, uses config setting.
        exclude_regions : list, optional
            List of region definitions to exclude. If None, uses config.
        reference_point_mode : str, optional
            Reference point mode: 'mean', 'none', 'user'. Overrides config if provided.
        reference_point : tuple, optional
            User reference point coordinates. Required if mode='user'.
        return_result_object : bool
            Whether to return detailed RampResult object
            
        Returns:
        --------
        np.ndarray or Tuple[np.ndarray, RampResult]
            Detrended values, optionally with detailed results
        """
        # Use provided parameters or fall back to config
        ramp_type = ramp_type or self.config.ramp_type
        ref_mode = reference_point_mode or self.config.reference_point_mode
        ref_point = reference_point or self.config.reference_point
        
        # Convert point coordinates to xy if needed
        if coord_type == 'lonlat':
            points_xy = np.zeros_like(points)
            for i, (lon, lat) in enumerate(points):
                x, y = self.ll2xy(lon, lat)
                points_xy[i] = [x, y]
        else:
            points_xy = points.copy()
        
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        points_valid = points_xy[valid_mask]
        values_valid = values[valid_mask]
        
        if len(values_valid) == 0:
            warnings.warn("No valid data points found")
            if return_result_object:
                return values, RampResult()
            return values
        
        # Apply geographic masking
        if exclude_regions is not None:
            # Use temporary regions
            fit_mask = self._apply_regions_mask(points_valid, exclude_regions, coord_type)
        else:
            # Use configured regions
            fit_mask = self._apply_geographic_masking(
                points_valid[:, 0], points_valid[:, 1], coord_type
            )
        
        points_fit = points_valid[fit_mask]
        values_fit = values_valid[fit_mask]
        
        # Apply outlier removal if configured
        if self.config.remove_outliers_before_fitting:
            outlier_mask = self._remove_outliers(values_fit)
            points_fit = points_fit[outlier_mask]
            values_fit = values_fit[outlier_mask]
            
            outlier_info = {
                'method': self.config.outlier_method,
                'threshold': self.config.outlier_threshold,
                'n_outliers': len(values_fit) - np.sum(outlier_mask)
            }
        else:
            outlier_info = {}
        
        # Check minimum points requirement
        if len(values_fit) < self.config.min_points_required:
            warnings.warn(f"Insufficient data points ({len(values_fit)}) for {ramp_type} "
                         f"ramp fitting (requires >= {self.config.min_points_required})")
            if return_result_object:
                return values, RampResult()
            return values
        
        # Determine and apply reference point
        ref_point_xy, actual_ref_point = self._determine_reference_point(
            points_fit[:, 0], points_fit[:, 1], ref_mode, ref_point, coord_type
        )
        
        # Center coordinates relative to reference point
        points_fit_centered = points_fit - ref_point_xy
        points_valid_centered = points_valid - ref_point_xy
        
        # Fit ramp surface using centered coordinates
        try:
            coeffs, fit_stats = self._fit_ramp_surface(
                points_fit_centered[:, 0], points_fit_centered[:, 1], values_fit, ramp_type
            )
        except Exception as e:
            warnings.warn(f"Failed to fit {ramp_type} ramp: {e}")
            if return_result_object:
                return values, RampResult()
            return values
        
        # Evaluate ramp at all valid points using centered coordinates
        ramp_values = self._evaluate_ramp_surface_points(points_valid_centered, ramp_type, coeffs)
        
        # Remove ramp from values
        detrended_values = values.copy()
        detrended_values[valid_mask] = values_valid - ramp_values
        
        # Create result object if requested
        if return_result_object:
            result = RampResult(
                original_data_shape=(len(values),),
                n_input_points=len(values_valid),
                coord_type=coord_type,
                reference_point_mode=ref_mode,
                reference_point_used=actual_ref_point,
                reference_coord_type=self.config.reference_coord_type,
                ramp_type=ramp_type,
                coefficients=coeffs.tolist(),
                n_fit_points=len(values_fit),
                n_excluded_points=len(values_valid) - len(values_fit),
                fit_residual_rms=fit_stats['rms_residual'],
                fit_r_squared=fit_stats['r_squared'],
                ramp_value_range=(float(np.min(ramp_values)), float(np.max(ramp_values))),
                original_data_range=(float(np.min(values_valid)), float(np.max(values_valid))),
                detrended_data_range=(float(np.min(detrended_values[valid_mask])), 
                                    float(np.max(detrended_values[valid_mask]))),
                regions_used=exclude_regions or self.config.exclude_regions,
                outliers_removed=self.config.remove_outliers_before_fitting,
                outlier_info=outlier_info,
                condition_number=fit_stats.get('condition_number', 0.0),
                fit_uncertainty=fit_stats.get('parameter_uncertainty', [])
            )
            
            return detrended_values, result
        else:
            return detrended_values
    
    def apply_ramp_to_data(self,
                          data: np.ndarray,
                          x_coords: np.ndarray,
                          y_coords: np.ndarray,
                          ramp_result: RampResult,
                          coord_type: str = None) -> np.ndarray:
        """
        Apply a previously fitted ramp to new data.
        
        Parameters:
        -----------
        data : np.ndarray
            Data to which ramp should be applied (added back)
        x_coords : np.ndarray
            X coordinates
        y_coords : np.ndarray
            Y coordinates
        ramp_result : RampResult
            Previous ramp fitting result
        coord_type : str, optional
            Coordinate type. Uses ramp_result.coord_type if None.
            
        Returns:
        --------
        np.ndarray
            Data with ramp applied
        """
        coord_type = coord_type or ramp_result.coord_type
        
        # Create coordinate grids if needed
        if x_coords.ndim == 1 and y_coords.ndim == 1:
            X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='xy')
        else:
            X_grid, Y_grid = x_coords, y_coords
        
        # Convert to xy if needed
        if coord_type == 'lonlat':
            X_km, Y_km = self._convert_lonlat_grid_to_xy(X_grid, Y_grid)
        else:
            X_km, Y_km = X_grid, Y_grid
        
        # Evaluate ramp surface
        ramp_surface = self._evaluate_ramp_surface(
            X_km, Y_km, ramp_result.ramp_type, np.array(ramp_result.coefficients)
        )
        
        return data + ramp_surface
    
    def evaluate_ramp_at_points(self,
                               points: np.ndarray,
                               ramp_result: RampResult,
                               coord_type: str = None) -> np.ndarray:
        """
        Evaluate a fitted ramp at specific points.
        
        Parameters:
        -----------
        points : np.ndarray
            Points where to evaluate ramp (n_points, 2)
        ramp_result : RampResult
            Previous ramp fitting result
        coord_type : str, optional
            Coordinate type. Uses ramp_result.coord_type if None.
            
        Returns:
        --------
        np.ndarray
            Ramp values at the specified points
        """
        coord_type = coord_type or ramp_result.coord_type
        
        # Convert points to xy if needed
        if coord_type == 'lonlat':
            points_xy = np.zeros_like(points)
            for i, (lon, lat) in enumerate(points):
                x, y = self.ll2xy(lon, lat)
                points_xy[i] = [x, y]
        else:
            points_xy = points.copy()
        
        # Get reference point in xy coordinates
        if ramp_result.reference_point_used is None:
            ref_point_xy = np.array([0.0, 0.0])
        else:
            if ramp_result.reference_coord_type == 'lonlat':
                ref_x, ref_y = self.ll2xy(ramp_result.reference_point_used[0], 
                                        ramp_result.reference_point_used[1])
                ref_point_xy = np.array([ref_x, ref_y])
            else:
                ref_point_xy = np.array(ramp_result.reference_point_used)
        
        # Center points relative to reference
        points_centered = points_xy - ref_point_xy
        
        return self._evaluate_ramp_surface_points(
            points_centered, ramp_result.ramp_type, np.array(ramp_result.coefficients)
        )

    def _determine_reference_point(self, 
                                  x_fit: np.ndarray, 
                                  y_fit: np.ndarray,
                                  ref_mode: str,
                                  ref_point: Optional[Tuple[float, float]],
                                  coord_type: str) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
        """
        Determine the reference point for coordinate centering.
        
        Parameters:
        -----------
        x_fit, y_fit : np.ndarray
            Fit coordinates in xy (km)
        ref_mode : str
            Reference point mode: 'mean', 'none', 'user'
        ref_point : tuple, optional
            User-provided reference point
        coord_type : str
            Original coordinate type
            
        Returns:
        --------
        Tuple[np.ndarray, Optional[Tuple]]
            Reference point in xy coordinates and in original coordinates
        """
        if ref_mode == 'none':
            # No centering
            return np.array([0.0, 0.0]), None
            
        elif ref_mode == 'mean':
            # Use mean of fit points
            ref_x_km = np.mean(x_fit)
            ref_y_km = np.mean(y_fit)
            ref_point_xy = np.array([ref_x_km, ref_y_km])
            
            # Convert back to original coordinate type for reporting
            if coord_type == 'lonlat':
                ref_lon, ref_lat = self.xy2ll(ref_x_km, ref_y_km)
                actual_ref_point = (ref_lon, ref_lat)
            else:
                actual_ref_point = (ref_x_km, ref_y_km)
                
            return ref_point_xy, actual_ref_point
            
        elif ref_mode == 'user':
            if ref_point is None:
                raise ValueError("Reference point must be provided when mode='user'")
            
            # Convert user point to xy if needed
            if self.config.reference_coord_type == 'lonlat':
                ref_x_km, ref_y_km = self.ll2xy(ref_point[0], ref_point[1])
                actual_ref_point = ref_point
            else:
                ref_x_km, ref_y_km = ref_point
                # Convert to lonlat for consistent reporting
                if coord_type == 'lonlat':
                    actual_ref_point = self.xy2ll(ref_x_km, ref_y_km)
                else:
                    actual_ref_point = ref_point
                    
            ref_point_xy = np.array([ref_x_km, ref_y_km])
            return ref_point_xy, actual_ref_point
            
        else:
            raise ValueError(f"Unknown reference point mode: {ref_mode}")
    
    def plot_ramp_analysis(self,
                          data: np.ndarray,
                          x_coords: np.ndarray,
                          y_coords: np.ndarray,
                          ramp_result: RampResult,
                          coord_type: str = None,
                          figsize: Tuple[int, int] = (15, 10),
                          cmap: str = 'RdBu_r',
                          show_fit_points: bool = True,
                          show_excluded_regions: bool = True) -> plt.Figure:
        """
        Create comprehensive visualization of ramp analysis results.
        
        Parameters:
        -----------
        data : np.ndarray
            Original data matrix
        x_coords : np.ndarray
            X coordinates
        y_coords : np.ndarray
            Y coordinates
        ramp_result : RampResult
            Ramp fitting results
        coord_type : str, optional
            Coordinate type
        figsize : tuple
            Figure size
        cmap : str
            Colormap
        show_fit_points : bool
            Whether to show points used for fitting
        show_excluded_regions : bool
            Whether to highlight excluded regions
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        coord_type = coord_type or ramp_result.coord_type
        
        with sci_plot_style():
            fig = plt.figure(figsize=figsize)
            
            # Create subplot layout
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1],
                                hspace=0.3, wspace=0.3)
            
            # Original data
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_data_map(ax1, data, x_coords, y_coords, coord_type, 
                              title='Original Data', cmap=cmap)
            
            # Fitted ramp surface
            ax2 = fig.add_subplot(gs[0, 1])
            ramp_surface = self._create_ramp_surface_for_plotting(
                x_coords, y_coords, ramp_result, coord_type
            )
            self._plot_data_map(ax2, ramp_surface, x_coords, y_coords, coord_type,
                              title='Fitted Ramp Surface', cmap='viridis')
            
            # Detrended data
            ax3 = fig.add_subplot(gs[0, 2])
            detrended = data - ramp_surface
            self._plot_data_map(ax3, detrended, x_coords, y_coords, coord_type,
                              title='Detrended Data', cmap=cmap)
            
            # Fit diagnostics
            ax4 = fig.add_subplot(gs[1, :2])
            self._plot_fit_diagnostics(ax4, data, x_coords, y_coords, ramp_result, 
                                     coord_type, show_fit_points, show_excluded_regions)
            
            # Statistics
            ax5 = fig.add_subplot(gs[1, 2])
            self._plot_ramp_statistics(ax5, ramp_result)
            
            plt.suptitle(f'Regional {ramp_result.ramp_type.title()} Ramp Analysis', 
                        fontsize=14, fontweight='bold')
            
            return fig
    
    # Internal methods
    def _convert_lonlat_grid_to_xy(self, lon_grid: np.ndarray, lat_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert lon/lat grids to xy coordinates in km."""
        x_grid = np.zeros_like(lon_grid)
        y_grid = np.zeros_like(lat_grid)
        
        for i in range(lon_grid.shape[0]):
            for j in range(lon_grid.shape[1]):
                x, y = self.ll2xy(lon_grid[i, j], lat_grid[i, j])
                x_grid[i, j] = x
                y_grid[i, j] = y
        
        return x_grid, y_grid
    
    def _apply_exclude_box_mask(self, x: np.ndarray, y: np.ndarray, 
                               exclude_box: Tuple[float, float, float, float],
                               coord_type: str) -> np.ndarray:
        """Apply a single exclude box mask."""
        min_x, max_x, min_y, max_y = exclude_box
        
        if coord_type == 'lonlat':
            # Convert box to xy coordinates
            min_x_km, min_y_km = self.ll2xy(min_x, min_y)
            max_x_km, max_y_km = self.ll2xy(max_x, max_y)
            
            outside_mask = ((x < min_x_km) | (x > max_x_km) | 
                          (y < min_y_km) | (y > max_y_km))
        else:
            outside_mask = ((x < min_x) | (x > max_x) | 
                          (y < min_y) | (y > max_y))
        
        return outside_mask
    
    def _apply_geographic_masking(self, x: np.ndarray, y: np.ndarray, coord_type: str) -> np.ndarray:
        """Apply configured geographic masking."""
        n_points = len(x)
        include_mask = np.ones(n_points, dtype=bool)
        
        # Apply exclude regions
        for region in self.config.exclude_regions:
            if region['type'] == 'box':
                box_mask = self._apply_exclude_box_mask(x, y, region['bounds'], 
                                                       region.get('coord_type', coord_type))
                include_mask &= box_mask
            elif region['type'] == 'polygon':
                poly_mask = self._apply_polygon_mask(x, y, region['vertices'],
                                                   region.get('coord_type', coord_type))
                include_mask &= ~poly_mask  # Exclude points inside polygon
            elif region['type'] == 'circle':
                circle_mask = self._apply_circle_mask(x, y, region['center'], region['radius'],
                                                    region.get('coord_type', coord_type))
                include_mask &= ~circle_mask  # Exclude points inside circle
        
        # Apply include regions (if any defined, only keep points in these regions)
        if self.config.include_regions:
            include_any = np.zeros(n_points, dtype=bool)
            for region in self.config.include_regions:
                if region['type'] == 'box':
                    box_mask = ~self._apply_exclude_box_mask(x, y, region['bounds'],
                                                           region.get('coord_type', coord_type))
                    include_any |= box_mask
                elif region['type'] == 'polygon':
                    poly_mask = self._apply_polygon_mask(x, y, region['vertices'],
                                                       region.get('coord_type', coord_type))
                    include_any |= poly_mask
                elif region['type'] == 'circle':
                    circle_mask = self._apply_circle_mask(x, y, region['center'], region['radius'],
                                                        region.get('coord_type', coord_type))
                    include_any |= circle_mask
            
            include_mask &= include_any
        
        return include_mask
    
    def _apply_polygon_mask(self, x: np.ndarray, y: np.ndarray, 
                           vertices: List[Tuple[float, float]], coord_type: str) -> np.ndarray:
        """Check which points are inside a polygon."""
        # Convert vertices to xy if needed
        if coord_type == 'lonlat':
            vertices_xy = []
            for lon, lat in vertices:
                x_v, y_v = self.ll2xy(lon, lat)
                vertices_xy.append((x_v, y_v))
        else:
            vertices_xy = vertices
        
        # Create polygon and check containment
        polygon = Polygon(vertices_xy)
        points = [Point(xi, yi) for xi, yi in zip(x, y)]
        
        return np.array([polygon.contains(point) for point in points])
    
    def _apply_circle_mask(self, x: np.ndarray, y: np.ndarray,
                          center: Tuple[float, float], radius: float, coord_type: str) -> np.ndarray:
        """Check which points are inside a circle."""
        # Convert center to xy if needed
        if coord_type == 'lonlat':
            center_x, center_y = self.ll2xy(center[0], center[1])
        else:
            center_x, center_y = center
        
        # Calculate distances
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        return distances <= radius
    
    def _remove_outliers(self, values: np.ndarray) -> np.ndarray:
        """Remove outliers using configured method."""
        if self.config.outlier_method == 'zscore':
            if len(values) > 2:
                z_scores = np.abs((values - np.mean(values)) / np.std(values))
                return z_scores <= self.config.outlier_threshold
        elif self.config.outlier_method == 'iqr':
            if len(values) > 4:
                q25 = np.percentile(values, 25)
                q75 = np.percentile(values, 75)
                iqr = q75 - q25
                if iqr > 0:
                    lower_bound = q25 - self.config.outlier_threshold * iqr
                    upper_bound = q75 + self.config.outlier_threshold * iqr
                    return (values >= lower_bound) & (values <= upper_bound)
        elif self.config.outlier_method == 'absolute':
            return np.abs(values) <= self.config.outlier_threshold
        
        return np.ones(len(values), dtype=bool)
    
    def _fit_ramp_surface(self, x: np.ndarray, y: np.ndarray, values: np.ndarray, 
                         ramp_type: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """Fit polynomial surface to data."""
        n_points = len(values)
        
        if ramp_type == 'constant':
            if n_points < 1:
                raise ValueError("Need at least 1 point for constant fit")
            coeffs = np.array([np.mean(values)])
            
            # Calculate fit statistics
            predicted = np.full_like(values, coeffs[0])
            residuals = values - predicted
            rms_residual = np.sqrt(np.mean(residuals**2))
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((values - np.mean(values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
        elif ramp_type == 'linear':
            if n_points < 3:
                raise ValueError("Need at least 3 points for linear fit")
            
            # Design matrix [x, y, 1]
            A = np.column_stack([x, y, np.ones(n_points)])
            
        elif ramp_type == 'quadratic':
            if n_points < 6:
                raise ValueError("Need at least 6 points for quadratic fit")
            
            # Design matrix [x^2, y^2, xy, x, y, 1]
            A = np.column_stack([x**2, y**2, x*y, x, y, np.ones(n_points)])
        
        else:
            raise ValueError(f"Unknown ramp_type: {ramp_type}")
        
        # Solve for non-constant cases
        if ramp_type != 'constant':
            if self.config.use_robust_fitting:
                coeffs = self._robust_least_squares(A, values)
            else:
                coeffs, residuals, rank, s = np.linalg.lstsq(A, values, rcond=None)
            
            # Calculate fit statistics
            predicted = A @ coeffs
            residuals = values - predicted
            rms_residual = np.sqrt(np.mean(residuals**2))
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((values - np.mean(values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate condition number
            condition_number = np.linalg.cond(A)
            
            # Estimate parameter uncertainties (for non-robust case)
            if not self.config.use_robust_fitting and len(residuals) > len(coeffs):
                try:
                    cov_matrix = np.linalg.inv(A.T @ A) * (ss_res / (len(residuals) - len(coeffs)))
                    parameter_uncertainty = np.sqrt(np.diag(cov_matrix))
                except:
                    parameter_uncertainty = np.zeros_like(coeffs)
            else:
                parameter_uncertainty = np.zeros_like(coeffs)
        else:
            condition_number = 1.0
            parameter_uncertainty = [0.0]
        
        fit_stats = {
            'rms_residual': rms_residual,
            'r_squared': r_squared,
            'condition_number': condition_number,
            'parameter_uncertainty': parameter_uncertainty.tolist()
        }
        
        return coeffs, fit_stats
    
    def _robust_least_squares(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Robust least squares fitting using iterative reweighting."""
        # Initial least squares solution
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Iterative reweighting
        for _ in range(5):  # Maximum 5 iterations
            residuals = b - A @ coeffs
            abs_residuals = np.abs(residuals)
            
            # Compute robust scale estimate
            mad = np.median(abs_residuals)
            if mad == 0:
                mad = np.mean(abs_residuals)
            if mad == 0:
                break
            
            # Compute weights (Huber weights)
            normalized_residuals = abs_residuals / (1.4826 * mad)  # 1.4826 for consistency with normal distribution
            weights = np.where(normalized_residuals <= self.config.robust_threshold,
                             1.0,
                             self.config.robust_threshold / normalized_residuals)
            
            # Weighted least squares
            W = np.diag(weights)
            try:
                coeffs = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
            except np.linalg.LinAlgError:
                # Fall back to lstsq if solve fails
                coeffs = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ b, rcond=None)[0]
        
        return coeffs
    
    def _evaluate_ramp_surface(self, x: np.ndarray, y: np.ndarray, 
                              ramp_type: str, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate fitted ramp surface at given coordinates."""
        if ramp_type == 'constant':
            return np.full_like(x, coeffs[0])
        elif ramp_type == 'linear':
            return coeffs[0] * x + coeffs[1] * y + coeffs[2]
        elif ramp_type == 'quadratic':
            return (coeffs[0] * x**2 + coeffs[1] * y**2 + coeffs[2] * x * y + 
                    coeffs[3] * x + coeffs[4] * y + coeffs[5])
        else:
            raise ValueError(f"Unknown ramp_type: {ramp_type}")
    
    def _evaluate_ramp_surface_points(self, points: np.ndarray, 
                                     ramp_type: str, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate ramp surface at point coordinates."""
        x = points[:, 0]
        y = points[:, 1]
        return self._evaluate_ramp_surface(x, y, ramp_type, coeffs)
    
    def _get_current_regions_info(self, exclude_box: Optional[Tuple[float, float, float, float]], 
                                 coord_type: str) -> List[Dict[str, Any]]:
        """Get current regions configuration for result storage."""
        if exclude_box is not None:
            return [{
                'type': 'box',
                'bounds': exclude_box,
                'coord_type': coord_type
            }]
        else:
            return self.config.exclude_regions.copy()
    
    def _create_ramp_surface_for_plotting(self, x_coords: np.ndarray, y_coords: np.ndarray,
                                         ramp_result: RampResult, coord_type: str) -> np.ndarray:
        """Create ramp surface for plotting purposes."""
        if x_coords.ndim == 1 and y_coords.ndim == 1:
            X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='xy')
        else:
            X_grid, Y_grid = x_coords, y_coords
        
        if coord_type == 'lonlat':
            X_km, Y_km = self._convert_lonlat_grid_to_xy(X_grid, Y_grid)
        else:
            X_km, Y_km = X_grid, Y_grid
        
        # Get reference point in xy coordinates
        if ramp_result.reference_point_used is None:
            ref_point_xy = np.array([0.0, 0.0])
        else:
            if ramp_result.reference_coord_type == 'lonlat':
                ref_x, ref_y = self.ll2xy(ramp_result.reference_point_used[0], 
                                        ramp_result.reference_point_used[1])
                ref_point_xy = np.array([ref_x, ref_y])
            else:
                ref_point_xy = np.array(ramp_result.reference_point_used)
        
        # Center coordinates
        X_km_centered = X_km - ref_point_xy[0]
        Y_km_centered = Y_km - ref_point_xy[1]
        
        return self._evaluate_ramp_surface(
            X_km_centered, Y_km_centered, ramp_result.ramp_type, np.array(ramp_result.coefficients)
        )
    
    def _plot_data_map(self, ax: plt.Axes, data: np.ndarray, 
                      x_coords: np.ndarray, y_coords: np.ndarray,
                      coord_type: str, title: str, cmap: str) -> None:
        """Plot data as a map."""
        if x_coords.ndim == 1 and y_coords.ndim == 1:
            extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
        else:
            extent = [np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)]
        
        im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        ax.set_title(title, fontsize=10)
        
        if coord_type == 'lonlat':
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            set_degree_formatter(ax, axis='both')
        else:
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
    
    def _plot_fit_diagnostics(self, ax: plt.Axes, data: np.ndarray,
                             x_coords: np.ndarray, y_coords: np.ndarray,
                             ramp_result: RampResult, coord_type: str,
                             show_fit_points: bool, show_excluded_regions: bool) -> None:
        """Plot fit diagnostics."""
        # This would show fit points, excluded regions, etc.
        # Implementation depends on specific requirements
        ax.text(0.5, 0.5, f'Fit Diagnostics\n\n'
                          f'Ramp type: {ramp_result.ramp_type}\n'
                          f'Fit points: {ramp_result.n_fit_points}\n'
                          f'Excluded: {ramp_result.n_excluded_points}\n'
                          f'RMS residual: {ramp_result.fit_residual_rms:.4f}\n'
                          f'R^2: {ramp_result.fit_r_squared:.4f}',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_title('Fit Diagnostics', fontsize=10)
        ax.axis('off')
    
    def _plot_ramp_statistics(self, ax: plt.Axes, ramp_result: RampResult) -> None:
        """Plot ramp statistics."""
        ax.text(0.05, 0.95, ramp_result.get_ramp_equation(),
                transform=ax.transAxes, verticalalignment='top',
                fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Add more statistics
        stats_text = (f'Original range: [{ramp_result.original_data_range[0]:.2f}, '
                     f'{ramp_result.original_data_range[1]:.2f}]\n'
                     f'Ramp range: [{ramp_result.ramp_value_range[0]:.2f}, '
                     f'{ramp_result.ramp_value_range[1]:.2f}]\n'
                     f'Detrended range: [{ramp_result.detrended_data_range[0]:.2f}, '
                     f'{ramp_result.detrended_data_range[1]:.2f}]\n'
                     f'Condition #: {ramp_result.condition_number:.2e}')
        
        ax.text(0.05, 0.6, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.set_title('Ramp Statistics', fontsize=10)
        ax.axis('off')


# Update convenience functions to support reference points
def remove_linear_trend_from_matrix(data: np.ndarray,
                                   x_coords: np.ndarray,
                                   y_coords: np.ndarray,
                                   exclude_box: Tuple[float, float, float, float],
                                   coord_type: str = 'lonlat',
                                   reference_point_mode: str = 'mean',
                                   reference_point: Tuple[float, float] = None,
                                   lon0: Optional[float] = None,
                                   lat0: Optional[float] = None,
                                   utmzone: Optional[int] = None) -> np.ndarray:
    """
    Quick function to remove linear trend from matrix data.
    
    Parameters:
    -----------
    data : np.ndarray
        2D data matrix
    x_coords : np.ndarray
        X coordinates
    y_coords : np.ndarray
        Y coordinates
    exclude_box : tuple
        Box to exclude: (min_x, max_x, min_y, max_y)
    coord_type : str
        'lonlat' or 'xy'
    reference_point_mode : str
        'mean', 'none', or 'user'
    reference_point : tuple, optional
        Reference point for centering (required if mode='user')
    lon0, lat0, utmzone : projection parameters
        
    Returns:
    --------
    np.ndarray
        Detrended data
    """
    remover = RegionalRampRemover(lon0=lon0, lat0=lat0, utmzone=utmzone)
    return remover.remove_ramp_from_matrix(
        data, x_coords, y_coords, coord_type, 'linear', exclude_box, 
        reference_point_mode, reference_point, False
    )

def remove_quadratic_trend_from_matrix(data: np.ndarray,
                                      x_coords: np.ndarray,
                                      y_coords: np.ndarray,
                                      exclude_box: Tuple[float, float, float, float],
                                      coord_type: str = 'lonlat',
                                      reference_point_mode: str = 'mean',
                                      reference_point: Tuple[float, float] = None,
                                      lon0: Optional[float] = None,
                                      lat0: Optional[float] = None,
                                      utmzone: Optional[int] = None) -> np.ndarray:
    """
    Quick function to remove quadratic trend from matrix data.
    
    Parameters:
    -----------
    data : np.ndarray
        2D data matrix
    x_coords : np.ndarray
        X coordinates
    y_coords : np.ndarray
        Y coordinates
    exclude_box : tuple
        Box to exclude: (min_x, max_x, min_y, max_y)
    coord_type : str
        'lonlat' or 'xy'
    reference_point_mode : str
        'mean', 'none', or 'user'
    reference_point : tuple, optional
        Reference point for centering (required if mode='user')
    lon0, lat0, utmzone : projection parameters
        
    Returns:
    --------
    np.ndarray
        Detrended data
    """
    remover = RegionalRampRemover(lon0=lon0, lat0=lat0, utmzone=utmzone)
    return remover.remove_ramp_from_matrix(
        data, x_coords, y_coords, coord_type, 'quadratic', exclude_box,
        reference_point_mode, reference_point, False
    )

def remove_trend_from_points(points: np.ndarray,
                            values: np.ndarray,
                            ramp_type: str = 'linear',
                            exclude_regions: List[Dict[str, Any]] = None,
                            coord_type: str = 'lonlat',
                            reference_point_mode: str = 'mean',
                            reference_point: Tuple[float, float] = None,
                            lon0: Optional[float] = None,
                            lat0: Optional[float] = None,
                            utmzone: Optional[int] = None) -> np.ndarray:
    """
    Quick function to remove trend from scattered point data.
    
    Parameters:
    -----------
    points : np.ndarray
        Point coordinates (n_points, 2)
    values : np.ndarray
        Values at points
    ramp_type : str
        Type of trend: 'constant', 'linear', 'quadratic'
    exclude_regions : list
        List of regions to exclude from fitting
    coord_type : str
        'lonlat' or 'xy'
    reference_point_mode : str
        Reference point mode: 'mean', 'none', 'user'
    reference_point : tuple, optional
        Reference point for centering (required if mode='user')
    lon0, lat0, utmzone : projection parameters
        
    Returns:
    --------
    np.ndarray
        Detrended values
    """
    remover = RegionalRampRemover(lon0=lon0, lat0=lat0, utmzone=utmzone)
    return remover.remove_ramp_from_points(
        points, values, coord_type, ramp_type, exclude_regions, 
        reference_point_mode, reference_point, False
    )


# Example usage
if __name__ == "__main__":
    # Create test data
    lon = np.linspace(120, 125, 101)
    lat = np.linspace(30, 35, 101)
    LON, LAT = np.meshgrid(lon, lat)
    
    # Create synthetic data with regional trend + local anomaly
    regional_trend = 0.01 * LON + 0.02 * LAT  # Linear regional trend
    local_anomaly = 50 * np.exp(-((LON-122.5)**2 + (LAT-32.5)**2) / (2 * 0.5**2))
    Z = regional_trend + local_anomaly + np.random.normal(0, 2, LON.shape)
    
    # Remove regional trend excluding central anomaly region
    remover = RegionalRampRemover(lon0=122.5, lat0=32.5, utmzone=51)
    
    # Configure to exclude central region containing the anomaly
    config = RampConfiguration()
    config.add_exclude_box(121.8, 123.2, 31.8, 33.2)  # Exclude central anomaly
    config.ramp_type = 'linear'
    remover.configure(config)
    
    # Remove trend
    detrended, result = remover.remove_ramp_from_matrix(
        Z, lon, lat, coord_type='lonlat'
    )
    
    # Print results
    result.print_summary()
    
    # Create visualization
    fig = remover.plot_ramp_analysis(Z, lon, lat, result, coord_type='lonlat')
    plt.show()
    
    print(f"Original data range: {np.min(Z):.2f} to {np.max(Z):.2f}")
    print(f"Detrended data range: {np.min(detrended):.2f} to {np.max(detrended):.2f}")
    print(f"Regional trend range: {result.ramp_value_range[0]:.2f} to {result.ramp_value_range[1]:.2f}")