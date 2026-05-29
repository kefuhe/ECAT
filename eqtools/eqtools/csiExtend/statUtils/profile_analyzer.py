"""
Profile Analysis Module

This module provides comprehensive tools for extracting, analyzing, visualizing,
and storing statistical information along profiles for various data formats including
matrices, scatter point sequences, and other spatial data types.

Key Features:
- Flexible profile definition with UTM projection support
- Support for multiple data formats (matrices, point clouds, grids)
- Statistical analysis along profiles (mean, median, percentiles, etc.)
- Advanced visualization capabilities
- Data export in multiple formats
- Integration with CSI coordinate system

Classes:
    ProfileAnalyzer: Main class for profile-based statistical analysis
    ProfileData: Data container for profile results
    ProfileVisualizer: Specialized plotting and visualization tools

Example:
    >>> from eqtools.statUtils import ProfileAnalyzer
    >>> # Initialize analyzer with CSI coordinate system
    >>> analyzer = ProfileAnalyzer(lon0=120.0, lat0=30.0, utmzone=51)
    >>> # Define profile in lon/lat
    >>> profile = analyzer.define_profile(
    ...     start=(120.0, 30.0), end=(125.0, 35.0),
    ...     spacing=1.0  # 1km spacing
    ... )
    >>> # Extract data along profile
    >>> results = analyzer.extract_from_matrix(
    ...     data=elevation_matrix, 
    ...     coordinates=(lon_grid, lat_grid),
    ...     coord_type='lonlat'
    ... )

Authors:
    [Your Name]

Version:
    1.0.0

Last Updated:
    2025-08-01
"""

# Standard library imports
import os
import warnings
from typing import Union, Tuple, List, Dict, Optional, Any, Literal
from dataclasses import dataclass, field
from pathlib import Path

# Scientific computing imports
import numpy as np
import pandas as pd
from scipy import interpolate, stats
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

# Geospatial imports
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points
from shapely.affinity import scale, translate
import geopandas as gpd
from rtree import index

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri as cmc
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

# Data format imports
import h5py
import json
import pickle

# CSI imports
from csi.SourceInv import SourceInv

# Local imports
from ...plottools import sci_plot_style
from ...plottools import set_degree_formatter
from .regional_ramp_removal import RegionalRampRemover, RampConfiguration, RampResult


@dataclass
class ProfileConfiguration:
    """Configuration class for profile definition and analysis parameters."""
    
    # Profile geometry - will be set by define_profile_* methods
    start_point: Tuple[float, float] = None
    end_point: Tuple[float, float] = None
    spacing: float = 1.0  # km spacing
    width: Optional[float] = None  # Profile width for swath analysis (km)
    
    # Reference point for zero distance
    reference_point: Optional[Tuple[float, float]] = None  # Reference point for distance zero
    reference_coord_type: str = 'lonlat'  # Coordinate type for reference point
    
    # Profile definition parameters (for reference)
    profile_definition_method: Optional[str] = None
    profile_params: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis parameters
    interpolation_method: str = 'linear'  # 'linear', 'cubic', 'nearest'
    aggregation_method: str = 'mean'  # 'mean', 'median', 'max', 'min'
    statistical_measures: List[str] = field(default_factory=lambda: ['mean', 'std', 'median', 'q25', 'q75'])
    
    # Data handling
    handle_nan: str = 'skip'  # 'skip', 'interpolate', 'zero'
    outlier_removal: bool = False
    outlier_threshold: float = 3.0  # 阈值含义取决于方法
    outlier_method: str = 'zscore'  # 'zscore', 'iqr', 'absolute'
    
    # Swath analysis settings
    swath_method: str = 'optimized'  # 'auto', 'shapely', 'optimized', 'geopandas'

    # Swath envelope settings
    envelope_method: str = 'minmax'  # 'minmax', 'percentile'
    envelope_percentiles: Tuple[float, float] = (5, 95)  # (lower, upper) percentiles
    envelope_smooth: bool = True  # Whether to smooth envelope curves
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_interpolation = ['linear', 'cubic', 'nearest']
        if self.interpolation_method not in valid_interpolation:
            raise ValueError(f"interpolation_method must be one of {valid_interpolation}")
        
        # Validate swath method
        valid_swath_methods = ['auto', 'shapely', 'optimized', 'geopandas']
        if self.swath_method not in valid_swath_methods:
            raise ValueError(f"swath_method must be one of {valid_swath_methods}")
        
        # Validate envelope method
        valid_envelope_methods = ['minmax', 'percentile']
        if self.envelope_method not in valid_envelope_methods:
            raise ValueError(f"envelope_method must be one of {valid_envelope_methods}")
        
        # Validate outlier method
        valid_outlier_methods = ['zscore', 'iqr', 'absolute']
        if self.outlier_method not in valid_outlier_methods:
            raise ValueError(f"outlier_method must be one of {valid_outlier_methods}")
        
        # Validate outlier threshold based on method
        if self.outlier_removal:
            if self.outlier_method == 'zscore' and self.outlier_threshold <= 0:
                raise ValueError("For zscore method, outlier_threshold must be positive (typically 2-3)")
            elif self.outlier_method == 'iqr' and self.outlier_threshold <= 0:
                raise ValueError("For iqr method, outlier_threshold must be positive (typically 1.5-3)")
            elif self.outlier_method == 'absolute' and self.outlier_threshold < 0:
                raise ValueError("For absolute method, outlier_threshold must be non-negative")
        
        # Validate percentiles
        if self.envelope_method == 'percentile':
            lower, upper = self.envelope_percentiles
            if not (0 <= lower < upper <= 100):
                raise ValueError(f"envelope_percentiles must satisfy 0 <= lower < upper <= 100, got {self.envelope_percentiles}")

@dataclass
class ProfileData:
    """Data container for profile analysis results."""
    
    # Profile geometry
    profile_coordinates_lonlat: np.ndarray = None  # (n_points, 2) - lon/lat
    profile_coordinates_xy: np.ndarray = None      # (n_points, 2) - x/y in km
    profile_distances: np.ndarray = None          # Distance along profile in km
    profile_values: np.ndarray = None             # Extracted values along profile
    profile_uncertainties: Optional[np.ndarray] = None  # Uncertainties if available
    
    # Swath data (if width > 0)
    swath_coordinates_lonlat: Optional[np.ndarray] = None  # All points within swath (lon/lat)
    swath_coordinates_xy: Optional[np.ndarray] = None      # All points within swath (x/y km)
    swath_values: Optional[np.ndarray] = None              # All values within swath
    swath_distances: Optional[np.ndarray] = None           # Distances along profile for swath points
    swath_offsets: Optional[np.ndarray] = None             # Perpendicular distances from profile (km)
    
    # Statistics
    statistics: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Metadata
    data_type: str = 'unknown'
    units: Optional[str] = None
    description: Optional[str] = None
    processing_info: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self):
        """Return number of points along profile."""
        return len(self.profile_distances) if self.profile_distances is not None else 0
    
    def get_bounds(self) -> Tuple[float, float]:
        """Get value bounds (min, max) for the profile data."""
        if self.profile_values is not None:
            valid_values = self.profile_values[~np.isnan(self.profile_values)]
            if len(valid_values) > 0:
                return float(np.min(valid_values)), float(np.max(valid_values))
        return 0.0, 1.0



class ProfileAnalyzer(SourceInv):
    """
    Comprehensive profile analysis tool for spatial data.

    This class inherits from CSI's SourceInv class to provide coordinate
    transformation capabilities and follows CSI conventions for coordinate systems.
    
    Attributes:
    -----------
    config : ProfileConfiguration
        Configuration object containing analysis parameters
    current_profile : ProfileData
        Currently loaded profile data
    visualizer : ProfileVisualizer
        Associated visualization object
    ramp_remover : RegionalRampRemover, optional
        Regional ramp removal tool
        
    Example:
    --------
    >>> analyzer = ProfileAnalyzer(
    ...     name='profile_analysis',
    ...     lon0=120.0, lat0=30.0, utmzone=51
    ... )
    >>> # Set up profile in lon/lat
    >>> config = ProfileConfiguration(
    ...     start_point=(120.0, 30.0),
    ...     end_point=(125.0, 35.0),
    ...     spacing=1.0,  # 1km
    ...     width=5.0     # 5km
    ... )
    >>> analyzer.configure(config)
    >>> 
    >>> # Extract data from matrix
    >>> results = analyzer.extract_from_matrix(
    ...     data_matrix, lon_coords, lat_coords, coord_type='lonlat'
    ... )
    >>> 
    >>> # Compute statistics
    >>> stats = analyzer.compute_statistics(results)
    >>> 
    >>> # Visualize results
    >>> fig = analyzer.plot_profile_summary(results)
    """
    
    def __init__(self, name='ProfileAnalyzer', lon0=None, lat0=None, utmzone=None, 
                 ellps='WGS84', config: Optional[ProfileConfiguration] = None):
        """
        Initialize ProfileAnalyzer with CSI coordinate system.
        
        Parameters:
        -----------
        name : str
            Name for the analyzer
        lon0 : float
            Reference longitude for UTM projection
        lat0 : float
            Reference latitude for UTM projection
        utmzone : int
            UTM zone number
        ellps : str
            Ellipsoid definition
        config : ProfileConfiguration, optional
            Initial configuration. If None, uses default settings.
        """
        # Initialize parent sourceinv class
        super().__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0)
        
        self.config = config or ProfileConfiguration()
        self.current_profile = None
        self.visualizer = ProfileVisualizer(self)
        
        # Initialize ramp remover with same coordinate system
        self.ramp_remover = RegionalRampRemover(
            name=f"{name}_ramp_remover",
            lon0=lon0, lat0=lat0, utmzone=utmzone, ellps=ellps
        )
    
    def configure(self, config: ProfileConfiguration) -> None:
        """
        Update analyzer configuration.
        
        Parameters:
        -----------
        config : ProfileConfiguration
            New configuration object
        """
        self.config = config

    def set_reference_point(self, 
                           reference_point: Tuple[float, float],
                           coord_type: str = 'lonlat') -> None:
        """
        Set reference point for distance zero on the profile.
        
        Parameters:
        -----------
        reference_point : tuple
            Reference point coordinates (lon, lat) or (x, y)
        coord_type : str
            Coordinate type: 'lonlat' or 'xy'
        """
        self.config.reference_point = reference_point
        self.config.reference_coord_type = coord_type
        
        # If we already have profile data, recalculate distances
        if self.current_profile is not None:
            self._recalculate_distances_with_reference()
    
    def _recalculate_distances_with_reference(self) -> None:
        """Recalculate profile distances using the reference point."""
        if (self.config.reference_point is None or 
            self.current_profile is None or 
            self.current_profile.profile_coordinates_xy is None):
            return
        
        # Get reference point in xy coordinates
        if self.config.reference_coord_type == 'lonlat':
            ref_x, ref_y = self.ll2xy(
                self.config.reference_point[0], 
                self.config.reference_point[1]
            )
            ref_point_xy = np.array([ref_x, ref_y])
        else:
            ref_point_xy = np.array(self.config.reference_point)
        
        # Find projection of reference point onto profile
        profile_coords = self.current_profile.profile_coordinates_xy
        ref_distance = self._project_point_onto_profile(ref_point_xy, profile_coords)
        
        # Recalculate distances relative to reference
        original_distances = self._calculate_profile_distances_km(profile_coords)
        self.current_profile.profile_distances = original_distances - ref_distance
        
        # Also update swath distances if they exist
        if (self.current_profile.swath_distances is not None and 
            len(self.current_profile.swath_distances) > 0):
            self.current_profile.swath_distances = self.current_profile.swath_distances - ref_distance
    
    def _project_point_onto_profile(self, 
                                   point_xy: np.ndarray, 
                                   profile_coords: np.ndarray) -> float:
        """
        Project a point onto the profile line and return the distance along profile.
        
        Parameters:
        -----------
        point_xy : np.ndarray
            Point coordinates in xy (km)
        profile_coords : np.ndarray
            Profile coordinates in xy (km)
            
        Returns:
        --------
        float
            Distance along profile to the projected point (km)
        """
        from shapely.geometry import Point, LineString
        
        # Create profile line
        profile_line = LineString(profile_coords)
        
        # Create point
        point = Point(point_xy[0], point_xy[1])
        
        # Project point onto line
        distance_along_profile = profile_line.project(point)
        
        return distance_along_profile
    
    def _calculate_profile_distances_km(self, profile_coords: np.ndarray) -> np.ndarray:
        """Calculate cumulative distances along profile in km."""
        # Use Euclidean distances (coordinates already in km)
        diff = np.diff(profile_coords, axis=0)
        distances = np.concatenate([[0], np.cumsum(np.sqrt(np.sum(diff**2, axis=1)))])
        
        # Apply reference point offset if specified
        if self.config.reference_point is not None:
            # Get reference point in xy coordinates
            if self.config.reference_coord_type == 'lonlat':
                ref_x, ref_y = self.ll2xy(
                    self.config.reference_point[0], 
                    self.config.reference_point[1]
                )
                ref_point_xy = np.array([ref_x, ref_y])
            else:
                ref_point_xy = np.array(self.config.reference_point)
            
            # Find projection of reference point onto profile
            ref_distance = self._project_point_onto_profile(ref_point_xy, profile_coords)
            
            # Adjust distances relative to reference
            distances = distances - ref_distance
        
        return distances
    
    def get_reference_info(self) -> Dict[str, Any]:
        """
        Get information about the reference point and its projection.
        
        Returns:
        --------
        Dict[str, Any]
            Reference point information
        """
        if self.config.reference_point is None:
            return {"status": "No reference point set"}
        
        info = {
            "status": "Reference point set",
            "reference_point": self.config.reference_point,
            "reference_coord_type": self.config.reference_coord_type
        }
        
        # If we have profile data, calculate projection details
        if (self.current_profile is not None and 
            self.current_profile.profile_coordinates_xy is not None):
            
            # Get reference point in xy coordinates
            if self.config.reference_coord_type == 'lonlat':
                ref_x, ref_y = self.ll2xy(
                    self.config.reference_point[0], 
                    self.config.reference_point[1]
                )
                ref_point_xy = np.array([ref_x, ref_y])
                ref_lon, ref_lat = self.config.reference_point
            else:
                ref_point_xy = np.array(self.config.reference_point)
                ref_lon, ref_lat = self.xy2ll(ref_point_xy[0], ref_point_xy[1])
            
            # Find projection onto profile
            profile_coords = self.current_profile.profile_coordinates_xy
            ref_distance_original = self._project_point_onto_profile(ref_point_xy, profile_coords)
            
            # Find closest point on profile
            from shapely.geometry import Point, LineString
            profile_line = LineString(profile_coords)
            ref_point = Point(ref_point_xy[0], ref_point_xy[1])
            projected_point = profile_line.interpolate(ref_distance_original)
            
            # Distance from reference point to profile
            perpendicular_distance = ref_point.distance(projected_point)
            
            # Convert projected point back to lonlat
            proj_lon, proj_lat = self.xy2ll(projected_point.x, projected_point.y)
            
            info.update({
                "reference_lonlat": (ref_lon, ref_lat),
                "projected_point_lonlat": (proj_lon, proj_lat),
                "projected_point_xy": (projected_point.x, projected_point.y),
                "perpendicular_distance_km": perpendicular_distance,
                "distance_along_original_profile_km": ref_distance_original
            })
        
        return info

    # Update existing methods to handle reference point
    def define_profile_from_endpoints(self,
                                    start: Tuple[float, float],
                                    end: Tuple[float, float],
                                    spacing: float = 1.0,
                                    width: Optional[float] = None,
                                    coord_type: str = 'lonlat',
                                    reference_point: Optional[Tuple[float, float]] = None,
                                    reference_coord_type: str = 'lonlat') -> Tuple[np.ndarray, np.ndarray]:
        """
        Define profile from start and end points with optional reference point.
        
        Parameters:
        -----------
        start : tuple
            Starting point (lon, lat) or (x, y)
        end : tuple
            Ending point (lon, lat) or (x, y)
        spacing : float
            Point spacing along profile in km
        width : float, optional
            Profile width for swath analysis in km
        coord_type : str
            Coordinate type: 'lonlat' or 'xy'
        reference_point : tuple, optional
            Reference point for distance zero (lon, lat) or (x, y)
        reference_coord_type : str
            Coordinate type for reference point: 'lonlat' or 'xy'
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Profile coordinates in both lonlat and xy formats
        """
        self.config.start_point = start
        self.config.end_point = end
        self.config.spacing = spacing
        self.config.width = width
        self.config.reference_point = reference_point
        self.config.reference_coord_type = reference_coord_type
        self.config.profile_definition_method = 'endpoints'
        self.config.profile_params = {
            'start': start,
            'end': end,
            'coord_type': coord_type,
            'reference_point': reference_point,
            'reference_coord_type': reference_coord_type
        }
        
        return self._create_profile_coordinates(start, end, spacing, coord_type)
    
    def define_profile_from_center_azimuth(self,
                                         center: Tuple[float, float],
                                         azimuth: float,
                                         length: float,
                                         spacing: float = 1.0,
                                         width: Optional[float] = None,
                                         coord_type: str = 'lonlat',
                                         reference_point: Optional[Tuple[float, float]] = None,
                                         reference_coord_type: str = 'lonlat') -> Tuple[np.ndarray, np.ndarray]:
        """
        Define profile from center point, azimuth, and total length with optional reference point.
        
        Parameters:
        -----------
        center : tuple
            Center point (lon, lat) or (x, y)
        azimuth : float
            Azimuth angle in degrees (0=North, 90=East, clockwise)
        length : float
            Total profile length in km
        spacing : float
            Point spacing along profile in km
        width : float, optional
            Profile width for swath analysis in km
        coord_type : str
            Coordinate type: 'lonlat' or 'xy'
        reference_point : tuple, optional
            Reference point for distance zero (lon, lat) or (x, y)
        reference_coord_type : str
            Coordinate type for reference point: 'lonlat' or 'xy'
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Profile coordinates in both lonlat and xy formats
        """
        # Calculate start and end points from center, azimuth, and length
        if coord_type == 'lonlat':
            # Convert center to xy for calculation
            center_xy = self.ll2xy(center[0], center[1])
            center_xy = np.array([center_xy[0], center_xy[1]])
        else:
            center_xy = np.array(center)
        
        # Calculate half-length offset in xy coordinates
        half_length = length / 2.0
        
        # Convert azimuth to radians (0=North, clockwise)
        azimuth_rad = np.radians(90 - azimuth)
        
        # Calculate offset vector
        dx = half_length * np.cos(azimuth_rad)
        dy = half_length * np.sin(azimuth_rad)
        
        # Calculate start and end points in xy
        start_xy = center_xy - np.array([dx, dy])
        end_xy = center_xy + np.array([dx, dy])
        
        # Convert back to original coordinate system if needed
        if coord_type == 'lonlat':
            start_lon, start_lat = self.xy2ll(start_xy[0], start_xy[1])
            end_lon, end_lat = self.xy2ll(end_xy[0], end_xy[1])
            start = (start_lon, start_lat)
            end = (end_lon, end_lat)
        else:
            start = tuple(start_xy)
            end = tuple(end_xy)
        
        # Store configuration
        self.config.start_point = start
        self.config.end_point = end
        self.config.spacing = spacing
        self.config.width = width
        self.config.reference_point = reference_point
        self.config.reference_coord_type = reference_coord_type
        self.config.profile_definition_method = 'center_azimuth'
        self.config.profile_params = {
            'center': center,
            'azimuth': azimuth,
            'length': length,
            'coord_type': coord_type,
            'reference_point': reference_point,
            'reference_coord_type': reference_coord_type
        }
        
        return self._create_profile_coordinates(start, end, spacing, coord_type)
    
    def define_profile_from_points_list(self,
                                      points: List[Tuple[float, float]],
                                      spacing: float = 1.0,
                                      width: Optional[float] = None,
                                      coord_type: str = 'lonlat',
                                      reference_point: Optional[Tuple[float, float]] = None,
                                      reference_coord_type: str = 'lonlat') -> Tuple[np.ndarray, np.ndarray]:
        """
        Define profile from a list of points with optional reference point.
        
        Parameters:
        -----------
        points : list
            List of points [(lon1, lat1), (lon2, lat2), ...] or [(x1, y1), ...]
        spacing : float
            Point spacing along profile in km
        width : float, optional
            Profile width for swath analysis in km
        coord_type : str
            Coordinate type: 'lonlat' or 'xy'
        reference_point : tuple, optional
            Reference point for distance zero (lon, lat) or (x, y)
        reference_coord_type : str
            Coordinate type for reference point: 'lonlat' or 'xy'
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Profile coordinates in both lonlat and xy formats
        """
        if len(points) < 2:
            raise ValueError("At least 2 points required for profile definition")
        
        # Store configuration
        self.config.start_point = points[0]
        self.config.end_point = points[-1]
        self.config.spacing = spacing
        self.config.width = width
        self.config.reference_point = reference_point
        self.config.reference_coord_type = reference_coord_type
        self.config.profile_definition_method = 'points_list'
        self.config.profile_params = {
            'points': points,
            'coord_type': coord_type,
            'reference_point': reference_point,
            'reference_coord_type': reference_coord_type
        }
        
        # Convert all points to xy coordinates for calculation
        if coord_type == 'lonlat':
            # Vectorized version for efficiency
            lons = np.array([pt[0] for pt in points])
            lats = np.array([pt[1] for pt in points])
            xs, ys = self.ll2xy(lons, lats)
            points_xy = np.column_stack([xs, ys])
        else:
            points_xy = np.array(points)
        
        # Create profile by interpolating between all points
        profile_xy = self._create_multi_segment_profile(points_xy, spacing)
        
        # Convert back to lonlat
        profile_lonlat = np.zeros_like(profile_xy)
        for i, (x, y) in enumerate(profile_xy):
            lon, lat = self.xy2ll(x, y)
            profile_lonlat[i] = [lon, lat]
        
        return profile_lonlat, profile_xy
    
    def define_profile_interactive(self,
                                 method: Literal['endpoints', 'center_azimuth', 'points_list'] = 'endpoints',
                                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interactive profile definition with automatic method selection.
        
        Parameters:
        -----------
        method : str
            Profile definition method
        **kwargs : dict
            Method-specific parameters
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Profile coordinates in both lonlat and xy formats
        """
        if method == 'endpoints':
            required_params = ['start', 'end']
            for param in required_params:
                if param not in kwargs:
                    raise ValueError(f"Parameter '{param}' required for endpoints method")
            return self.define_profile_from_endpoints(**kwargs)
        
        elif method == 'center_azimuth':
            required_params = ['center', 'azimuth', 'length']
            for param in required_params:
                if param not in kwargs:
                    raise ValueError(f"Parameter '{param}' required for center_azimuth method")
            return self.define_profile_from_center_azimuth(**kwargs)
        
        elif method == 'points_list':
            required_params = ['points']
            for param in required_params:
                if param not in kwargs:
                    raise ValueError(f"Parameter '{param}' required for points_list method")
            return self.define_profile_from_points_list(**kwargs)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _create_profile_coordinates(self,
                                  start: Tuple[float, float],
                                  end: Tuple[float, float],
                                  spacing: float,
                                  coord_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Internal method to create profile coordinates from start/end points."""
        if coord_type == 'lonlat':
            # Convert to xy coordinates for calculation
            start_xy = self.ll2xy(start[0], start[1])
            end_xy = self.ll2xy(end[0], end[1])
            start_xy = np.array([start_xy[0], start_xy[1]])
            end_xy = np.array([end_xy[0], end_xy[1]])
        else:
            start_xy = np.array(start)
            end_xy = np.array(end)
        
        # Calculate profile in xy coordinates (km)
        profile_xy = self._define_straight_line_profile_xy(start_xy, end_xy, spacing)
        
        # Convert back to lonlat
        profile_lonlat = np.stack(self.xy2ll(profile_xy[:, 0], profile_xy[:, 1]), axis=-1)
        
        return profile_lonlat, profile_xy
    
    def _create_multi_segment_profile(self,
                                    points_xy: np.ndarray,
                                    spacing: float) -> np.ndarray:
        """Create profile with multiple segments between points."""
        all_profile_coords = []
        
        for i in range(len(points_xy) - 1):
            start_xy = points_xy[i]
            end_xy = points_xy[i + 1]
            
            # Create segment profile
            segment_coords = self._define_straight_line_profile_xy(start_xy, end_xy, spacing)
            
            # Avoid duplicating points at segment boundaries
            if i > 0:
                segment_coords = segment_coords[1:]  # Skip first point (duplicate)
            
            all_profile_coords.append(segment_coords)
        
        # Concatenate all segments
        return np.vstack(all_profile_coords)
    
    # Update the legacy define_profile method to use new system
    def define_profile(self, 
                      start: Tuple[float, float],
                      end: Tuple[float, float],
                      spacing: float = 1.0,
                      width: Optional[float] = None,
                      coord_type: str = 'lonlat',
                      reference_point: Optional[Tuple[float, float]] = None,
                      reference_coord_type: str = 'lonlat') -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy method for backward compatibility.
        Use define_profile_from_endpoints for new code.
        """
        return self.define_profile_from_endpoints(start, end, spacing, width, coord_type, reference_point, reference_coord_type)
    
    def get_profile_info(self) -> Dict[str, Any]:
        """
        Get information about the currently defined profile.
        
        Returns:
        --------
        Dict[str, Any]
            Profile information including definition method and parameters
        """
        if self.config.start_point is None or self.config.end_point is None:
            return {"status": "No profile defined"}
        
        # Calculate profile statistics
        if self.config.start_point and self.config.end_point:
            if hasattr(self, 'current_profile') and self.current_profile is not None:
                total_length = self.current_profile.profile_distances[-1]
                n_points = len(self.current_profile.profile_distances)
            else:
                # Estimate from start/end points
                start_xy = self.ll2xy(self.config.start_point[0], self.config.start_point[1])
                end_xy = self.ll2xy(self.config.end_point[0], self.config.end_point[1])
                total_length = np.linalg.norm(np.array(end_xy) - np.array(start_xy))
                n_points = int(total_length / self.config.spacing) + 1
        
        info = {
            "status": "Profile defined",
            "definition_method": self.config.profile_definition_method or "endpoints",
            "definition_params": self.config.profile_params,
            "start_point": self.config.start_point,
            "end_point": self.config.end_point,
            "spacing": self.config.spacing,
            "width": self.config.width,
            "estimated_length": f"{total_length:.2f} km",
            "estimated_points": n_points
        }
        
        return info
    
    def _define_straight_line_profile_xy(self,
                                       start_xy: np.ndarray,
                                       end_xy: np.ndarray,
                                       spacing: float) -> np.ndarray:
        """Create straight line profile in xy coordinates (km)."""
        total_distance = np.linalg.norm(end_xy - start_xy)
        n_points = int(total_distance / spacing) + 1
        
        # Create profile points
        t = np.linspace(0, 1, n_points)
        profile_coords = start_xy[np.newaxis, :] + t[:, np.newaxis] * (end_xy - start_xy)[np.newaxis, :]
        
        return profile_coords
    
    def extract_from_matrix(self,
                           data: np.ndarray,
                           x_coords: np.ndarray,
                           y_coords: np.ndarray,
                           coord_type: str = 'lonlat',
                           uncertainties: Optional[np.ndarray] = None,
                           auto_apply_regional_ramp: bool = False) -> ProfileData:
        """
        Extract profile data from a 2D matrix/grid.
        
        Parameters:
        -----------
        data : np.ndarray
            2D data matrix
        x_coords : np.ndarray
            X-coordinate array (longitude or x in km)
        y_coords : np.ndarray
            Y-coordinate array (latitude or y in km)
        coord_type : str
            'lonlat' if coordinates are in degrees, 'xy' if in km
        uncertainties : np.ndarray, optional
            Uncertainty matrix matching data shape
        auto_apply_regional_ramp : bool
            Whether to automatically apply regional ramp removal if configured
            
        Returns:
        --------
        ProfileData
            Extracted profile data
        """
        # Check if profile is already defined
        if self.config.start_point is None or self.config.end_point is None:
            raise ValueError("Profile not defined. Call define_profile_* method first.")
        
        # Generate profile coordinates using existing configuration
        profile_lonlat, profile_xy = self._create_profile_coordinates(
            self.config.start_point,
            self.config.end_point,
            self.config.spacing,
            'lonlat'  # Profile points are always stored in lonlat
        )
        
        # Convert coordinate grids to appropriate format for interpolation
        if coord_type == 'lonlat':
            # Convert grid coordinates to xy for interpolation
            if x_coords.ndim == 1 and y_coords.ndim == 1:
                X_deg, Y_deg = np.meshgrid(x_coords, y_coords, indexing='xy')
            else:
                X_deg, Y_deg = x_coords, y_coords
            
            # Convert grid to xy coordinates
            X_km, Y_km = self.ll2xy(X_deg, Y_deg)
        else:
            # Coordinates already in km
            if x_coords.ndim == 1 and y_coords.ndim == 1:
                X_km, Y_km = np.meshgrid(x_coords, y_coords, indexing='xy')
            else:
                X_km, Y_km = x_coords, y_coords
        
        # Interpolate data at profile points (use xy coordinates)
        profile_values = self._interpolate_grid_data(
            data, X_km, Y_km, profile_xy
        )
        
        # Handle uncertainties if provided
        profile_uncertainties = None
        if uncertainties is not None:
            profile_uncertainties = self._interpolate_grid_data(
                uncertainties, X_km, Y_km, profile_xy
            )

        # Calculate distances along profile in km (with reference point offset)
        profile_distances = self._calculate_profile_distances_km(profile_xy)

        # Handle swath data if width is specified
        swath_data = None
        if self.config.width is not None:
            # 打印时间
            # import time
            # print("Extracting swath data...")
            # start_time = time.time()
            swath_data = self._extract_swath_data(
                data, X_km, Y_km, profile_xy, self.config.width
            )
            # end_time = time.time()
            # print(f"Swath data extraction completed in {end_time - start_time:.2f} seconds.")

            # Apply reference point offset to swath distances
            if swath_data and swath_data['distances'] is not None and self.config.reference_point is not None:
                ref_x, ref_y = self.ll2xy(
                    self.config.reference_point[0], 
                    self.config.reference_point[1]
                ) if self.config.reference_coord_type == 'lonlat' else self.config.reference_point
                ref_point_xy = np.array([ref_x, ref_y])
                ref_distance = self._project_point_onto_profile(ref_point_xy, profile_xy)
                swath_data['distances'] = swath_data['distances'] - ref_distance

        # Create ProfileData object
        result = ProfileData(
            profile_coordinates_lonlat=profile_lonlat,
            profile_coordinates_xy=profile_xy,
            profile_distances=profile_distances,
            profile_values=profile_values,
            profile_uncertainties=profile_uncertainties,
            data_type='matrix',
            processing_info={
                'source_shape': data.shape,
                'interpolation_method': self.config.interpolation_method,
                'coord_type': coord_type,
                'reference_point': self.config.reference_point,
                'reference_coord_type': self.config.reference_coord_type
            }
        )
        
        # Add swath data if available
        if swath_data:
            result.swath_coordinates_xy = swath_data['coordinates']
            result.swath_values = swath_data['values']
            result.swath_distances = swath_data['distances']
            result.swath_offsets = swath_data['offsets']
            
            # Convert swath coordinates to lonlat
            if swath_data['coordinates'] is not None:
                xs = swath_data['coordinates'][:, 0]
                ys = swath_data['coordinates'][:, 1]
                lons, lats = self.xy2ll(xs, ys)
                result.swath_coordinates_lonlat = np.column_stack([lons, lats])
        
        # Apply regional ramp removal if configured and requestedf
        if auto_apply_regional_ramp and hasattr(self, '_ramp_result'):
            result = self.apply_ramp_to_profile(result, self._ramp_result)
        
        self.current_profile = result
        return result
    
    def extract_from_points(self,
                           points: np.ndarray,
                           values: np.ndarray,
                           coord_type: str = 'lonlat',
                           uncertainties: Optional[np.ndarray] = None) -> ProfileData:
        """
        Extract profile data from scattered point data.
        
        Parameters:
        -----------
        points : np.ndarray
            Point coordinates (n_points, 2) - (lon,lat) or (x,y)
        values : np.ndarray
            Values at each point
        coord_type : str
            'lonlat' if coordinates are in degrees, 'xy' if in km
        uncertainties : np.ndarray, optional
            Uncertainties for each point
            
        Returns:
        --------
        ProfileData
            Extracted profile data
        """
        # Check if profile is already defined
        if self.config.start_point is None or self.config.end_point is None:
            raise ValueError("Profile not defined. Call define_profile_* method first.")
        
        # Generate profile coordinates using existing configuration
        profile_lonlat, profile_xy = self._create_profile_coordinates(
            self.config.start_point,
            self.config.end_point,
            self.config.spacing,
            'lonlat'  # Profile points are always stored in lonlat
        )
        
        # Convert point coordinates to xy if needed
        if coord_type == 'lonlat':
            # Vectorized version for efficiency
            lons = points[:, 0]
            lats = points[:, 1]
            xs, ys = self.ll2xy(lons, lats)
            points_xy = np.column_stack([xs, ys])
        else:
            points_xy = points.copy()
        
        # Build KDTree for efficient nearest neighbor search
        tree = cKDTree(points_xy)
        
        # Find nearest points and interpolate
        if self.config.width is None:
            # Simple nearest neighbor or interpolation
            profile_values = self._interpolate_scattered_data(
                points_xy, values, profile_xy
            )
            profile_uncertainties = None
            if uncertainties is not None:
                profile_uncertainties = self._interpolate_scattered_data(
                    points_xy, uncertainties, profile_xy
                )
            
            # No swath data for simple extraction
            swath_data = None
            
        else:
            # Swath analysis - find all points within width
            swath_data = self._extract_scattered_swath_data(
                points_xy, values, profile_xy, self.config.width, uncertainties
            )
            
            # Apply reference point offset to swath distances
            if swath_data and swath_data['distances'] is not None and self.config.reference_point is not None:
                ref_x, ref_y = self.ll2xy(
                    self.config.reference_point[0], 
                    self.config.reference_point[1]
                ) if self.config.reference_coord_type == 'lonlat' else self.config.reference_point
                ref_point_xy = np.array([ref_x, ref_y])
                ref_distance = self._project_point_onto_profile(ref_point_xy, profile_xy)
                swath_data['distances'] = swath_data['distances'] - ref_distance
            
            # Aggregate swath data to profile
            profile_values = self._aggregate_swath_to_profile(swath_data['values_by_segment'])
            if uncertainties is not None:
                profile_uncertainties = self._aggregate_swath_to_profile(
                    swath_data['uncertainties_by_segment']
                )
            else:
                profile_uncertainties = None
        
        # Calculate distances along profile (with reference point offset)
        profile_distances = self._calculate_profile_distances_km(profile_xy)
        
        # Create ProfileData object
        result = ProfileData(
            profile_coordinates_lonlat=profile_lonlat,
            profile_coordinates_xy=profile_xy,
            profile_distances=profile_distances,
            profile_values=profile_values,
            profile_uncertainties=profile_uncertainties,
            data_type='scattered_points',
            processing_info={
                'n_source_points': len(points),
                'interpolation_method': self.config.interpolation_method,
                'coord_type': coord_type,
                'reference_point': self.config.reference_point,
                'reference_coord_type': self.config.reference_coord_type
            }
        )
        
        # Add swath data if available
        if swath_data is not None:
            result.swath_coordinates_xy = swath_data['coordinates']
            result.swath_values = swath_data['values']
            result.swath_distances = swath_data['distances']
            result.swath_offsets = swath_data['offsets']
            
            # Convert swath coordinates to lonlat
            if swath_data['coordinates'] is not None:
                xs = swath_data['coordinates'][:, 0]
                ys = swath_data['coordinates'][:, 1]
                lons, lats = self.xy2ll(xs, ys)
                result.swath_coordinates_lonlat = np.column_stack([lons, lats])
        
        self.current_profile = result
        return result
    
    def _remove_outliers(self, values: np.ndarray, method: str = None) -> np.ndarray:
        """
        Remove outliers from data array using various methods.
        
        Parameters:
        -----------
        values : np.ndarray
            Input values
        method : str, optional
            Outlier detection method. If None, uses self.config.outlier_method.
            Available methods:
            - 'zscore': Z-score method (标准差倍数法)
            - 'iqr': Interquartile range method (四分位距法)
            - 'absolute': Absolute threshold method (绝对值阈值法)
            
        Returns:
        --------
        np.ndarray
            Boolean mask where True indicates valid (non-outlier) data
            
        Method Descriptions:
        -------------------
        1. 'zscore' (Z-score method / 标准差倍数法):
           - 计算每个数据点的 Z-score: z = (x - mean) / std
           - 移除 |z| > threshold 的数据点
           - threshold 通常设为 2-3，表示超过几个标准差
           - 适用于正态分布或近似正态分布的数据
           - 公式: |x - μ| > k × σ 的点被视为异常值
           
        2. 'iqr' (Interquartile Range method / 四分位距法):
           - 计算第一四分位数 Q1 (25%) 和第三四分位数 Q3 (75%)
           - 计算四分位距 IQR = Q3 - Q1
           - 定义异常值边界: [Q1 - k×IQR, Q3 + k×IQR]
           - threshold 通常设为 1.5 (经典值) 或 2.0-3.0 (更宽松)
           - 对非正态分布数据更稳健
           - 公式: x < Q1 - k×IQR 或 x > Q3 + k×IQR 的点被视为异常值
           
        3. 'absolute' (Absolute threshold method / 绝对值阈值法):
           - 直接使用用户提供的绝对阈值
           - 移除绝对值超过阈值的数据点
           - threshold 是具体的数值阈值 (与数据同单位)
           - 适用于已知合理数据范围的情况
           - 公式: |x| > threshold 的点被视为异常值
           
        Examples:
        --------
        # Z-score method (去除超过2个标准差的点)
        config.outlier_method = 'zscore'
        config.outlier_threshold = 2.0
        
        # IQR method (去除超过1.5倍IQR的点)
        config.outlier_method = 'iqr' 
        config.outlier_threshold = 1.5
        
        # Absolute threshold (去除绝对值超过100的点)
        config.outlier_method = 'absolute'
        config.outlier_threshold = 100.0
        """
        if not self.config.outlier_removal:
            return np.ones(len(values), dtype=bool)
        
        # Use provided method or fall back to config
        if method is None:
            method = self.config.outlier_method
        
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            return valid_mask
        
        valid_values = values[valid_mask]
        outlier_mask = np.ones(len(values), dtype=bool)
        
        if method == 'zscore':
            # Z-score method (标准差倍数法)
            # 移除超过 threshold 个标准差的数据点
            if len(valid_values) > 2:  # 至少需要3个点来计算标准差
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                
                if std_val > 0:
                    z_scores = np.abs((values - mean_val) / std_val)
                    outlier_mask = z_scores <= self.config.outlier_threshold
                    
                    # 记录移除的异常值信息
                    n_outliers = np.sum(~outlier_mask & valid_mask)
                    if n_outliers > 0:
                        print(f"Z-score method: Removed {n_outliers} outliers "
                              f"(threshold: {self.config.outlier_threshold} std devs)")
                
        elif method == 'iqr':
            # Interquartile range method (四分位距法)
            # 移除超出 Q1-k*IQR 到 Q3+k*IQR 范围的数据点
            if len(valid_values) > 4:  # 需要足够的点来计算四分位数
                q25 = np.percentile(valid_values, 25)
                q75 = np.percentile(valid_values, 75)
                iqr = q75 - q25
                
                if iqr > 0:
                    lower_bound = q25 - self.config.outlier_threshold * iqr
                    upper_bound = q75 + self.config.outlier_threshold * iqr
                    outlier_mask = (values >= lower_bound) & (values <= upper_bound)
                    
                    # 记录移除的异常值信息
                    n_outliers = np.sum(~outlier_mask & valid_mask)
                    if n_outliers > 0:
                        print(f"IQR method: Removed {n_outliers} outliers "
                              f"(threshold: {self.config.outlier_threshold} × IQR = {self.config.outlier_threshold * iqr:.2f})")
                        print(f"  Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")
                        
        elif method == 'absolute':
            # Absolute threshold method (绝对值阈值法)
            # 移除绝对值超过阈值的数据点
            outlier_mask = np.abs(values) <= self.config.outlier_threshold
            
            # 记录移除的异常值信息
            n_outliers = np.sum(~outlier_mask & valid_mask)
            if n_outliers > 0:
                print(f"Absolute method: Removed {n_outliers} outliers "
                      f"(threshold: |value| > {self.config.outlier_threshold})")
                
        else:
            raise ValueError(f"Unknown outlier removal method: {method}. "
                            f"Available methods: 'zscore', 'iqr', 'absolute'")
        
        # 结合有效数据掩码
        return valid_mask & outlier_mask
    
    def configure_zscore_outlier_removal(config: ProfileConfiguration, 
                                        threshold: float = 3.0) -> None:
        """
        Configure Z-score based outlier removal.
        
        Parameters:
        -----------
        config : ProfileConfiguration
            Configuration object to modify
        threshold : float
            Number of standard deviations (typically 2-3)
            
        Example:
        --------
        config = ProfileConfiguration()
        configure_zscore_outlier_removal(config, threshold=2.5)
        # 移除超过2.5个标准差的数据点
        """
        config.outlier_removal = True
        config.outlier_method = 'zscore'
        config.outlier_threshold = threshold
    
    def configure_iqr_outlier_removal(config: ProfileConfiguration, 
                                     threshold: float = 1.5) -> None:
        """
        Configure IQR based outlier removal.
        
        Parameters:
        -----------
        config : ProfileConfiguration
            Configuration object to modify
        threshold : float
            IQR multiplier (typically 1.5 for classical, 2-3 for lenient)
            
        Example:
        --------
        config = ProfileConfiguration()
        configure_iqr_outlier_removal(config, threshold=2.0)
        # 使用2倍IQR范围作为异常值边界
        """
        config.outlier_removal = True
        config.outlier_method = 'iqr'
        config.outlier_threshold = threshold
    
    def configure_absolute_outlier_removal(config: ProfileConfiguration, 
                                          threshold: float) -> None:
        """
        Configure absolute threshold based outlier removal.
        
        Parameters:
        -----------
        config : ProfileConfiguration
            Configuration object to modify
        threshold : float
            Absolute threshold value (same units as data)
            
        Example:
        --------
        config = ProfileConfiguration()
        configure_absolute_outlier_removal(config, threshold=1000.0)
        # 移除绝对值超过1000的数据点
        """
        config.outlier_removal = True
        config.outlier_method = 'absolute'
        config.outlier_threshold = threshold
    
    def set_outlier_removal(self, 
                           method: Literal['zscore', 'iqr', 'absolute'],
                           threshold: float,
                           enable: bool = True) -> None:
        """
        Configure outlier removal settings.
        
        Parameters:
        -----------
        method : str
            Outlier detection method
        threshold : float
            Threshold value (interpretation depends on method)
        enable : bool
            Whether to enable outlier removal
            
        Examples:
        --------
        # Z-score method: 移除超过2个标准差的点
        analyzer.set_outlier_removal('zscore', 2.0)
        
        # IQR method: 移除超过1.5倍IQR的点
        analyzer.set_outlier_removal('iqr', 1.5)
        
        # Absolute threshold: 移除绝对值超过100的点
        analyzer.set_outlier_removal('absolute', 100.0)
        """
        self.config.outlier_removal = enable
        self.config.outlier_method = method
        self.config.outlier_threshold = threshold
        
        # 验证设置
        if enable:
            if method == 'zscore' and threshold <= 0:
                raise ValueError("Z-score threshold must be positive")
            elif method == 'iqr' and threshold <= 0:
                raise ValueError("IQR threshold must be positive")
            elif method == 'absolute' and threshold < 0:
                raise ValueError("Absolute threshold must be non-negative")
    
    def _interpolate_grid_data(self,
                              data: np.ndarray,
                              X: np.ndarray,
                              Y: np.ndarray,
                              profile_coords: np.ndarray) -> np.ndarray:
        """
        Interpolate gridded data at profile coordinates (all in km).
        For large grids, only a small window around the profile is used to speed up interpolation.
        Handles NaN and outlier values before interpolation.
        """
        # 1. Mask out NaN and outlier values
        mask = ~np.isnan(data)
        if self.config.outlier_removal:
            outlier_mask = self._remove_outliers(data.flatten()).reshape(data.shape)
            mask = mask & outlier_mask
    
        # 2. Compute bounding box of profile points and add buffer
        buffer_km = max(self.config.width or 2.0, 2.0) * 2  # Buffer: 2x swath width or 2km
        min_x, max_x = profile_coords[:, 0].min() - buffer_km, profile_coords[:, 0].max() + buffer_km
        min_y, max_y = profile_coords[:, 1].min() - buffer_km, profile_coords[:, 1].max() + buffer_km
    
        # 3. Find index range for X/Y (assume regular grid)
        x1d = X[0, :]
        y1d = Y[:, 0]
        x_mask = (x1d >= min_x) & (x1d <= max_x)
        y_mask = (y1d >= min_y) & (y1d <= max_y)

        if not np.any(x_mask) or not np.any(y_mask):
            return np.full(len(profile_coords), np.nan)
    
        # 4. Extract subregion
        X_sub = X[np.ix_(y_mask, x_mask)]
        Y_sub = Y[np.ix_(y_mask, x_mask)]
        data_sub = data[np.ix_(y_mask, x_mask)]
        mask_sub = mask[np.ix_(y_mask, x_mask)]
    
        # 5. Only keep profile points inside subregion
        profile_mask = (
            (profile_coords[:, 0] >= x1d[x_mask][0]) & (profile_coords[:, 0] <= x1d[x_mask][-1]) &
            (profile_coords[:, 1] >= y1d[y_mask][0]) & (profile_coords[:, 1] <= y1d[y_mask][-1])
        )
        profile_coords_sub = profile_coords[profile_mask]
    
        # 6. Interpolation
        profile_values = np.full(len(profile_coords), np.nan)
        if self.config.interpolation_method == 'nearest':
            # Use map_coordinates for nearest neighbor
            x_indices = np.interp(profile_coords_sub[:, 0], X_sub[0, :], np.arange(X_sub.shape[1]))
            y_indices = np.interp(profile_coords_sub[:, 1], Y_sub[:, 0], np.arange(Y_sub.shape[0]))
            coords = np.vstack([y_indices, x_indices])
            # For masked data, set to nan
            data_sub_masked = np.where(mask_sub, data_sub, np.nan)
            values = map_coordinates(data_sub_masked, coords, order=0, mode='nearest')
            profile_values[profile_mask] = values
            # Outlier removal for interpolated values
            if self.config.outlier_removal:
                profile_outlier_mask = self._remove_outliers(profile_values)
                profile_values[~profile_outlier_mask] = np.nan
        elif self.config.interpolation_method == 'linear':
            from scipy.interpolate import RegularGridInterpolator
            data_sub_masked = np.where(mask_sub, data_sub, np.nan)
            interpolator = RegularGridInterpolator(
                (Y_sub[:, 0], X_sub[0, :]), data_sub_masked, method='linear', bounds_error=False, fill_value=np.nan
            )
            values = interpolator(profile_coords_sub)
            profile_values[profile_mask] = values
            if self.config.outlier_removal:
                profile_outlier_mask = self._remove_outliers(profile_values)
                profile_values[~profile_outlier_mask] = np.nan
        elif self.config.interpolation_method == 'cubic':
            # For cubic, use CloughTocher2DInterpolator on subregion
            mask_flat = mask_sub.flatten()
            x_flat = X_sub.flatten()[mask_flat]
            y_flat = Y_sub.flatten()[mask_flat]
            data_flat = data_sub.flatten()[mask_flat]
            from scipy.interpolate import CloughTocher2DInterpolator
            if len(data_flat) == 0:
                return profile_values
            interpolator = CloughTocher2DInterpolator(np.column_stack([x_flat, y_flat]), data_flat)
            values = interpolator(profile_coords_sub)
            profile_values[profile_mask] = values
            if self.config.outlier_removal:
                profile_outlier_mask = self._remove_outliers(profile_values)
                profile_values[~profile_outlier_mask] = np.nan
        else:
            raise ValueError(f"Unknown interpolation method: {self.config.interpolation_method}")
    
        return profile_values
    
    def _interpolate_scattered_data(self,
                                   points: np.ndarray,
                                   values: np.ndarray,
                                   profile_coords: np.ndarray) -> np.ndarray:
        """
        Interpolate scattered point data at profile coordinates (all in km).
        For large point clouds, only use points within a buffer around the profile to speed up interpolation.
        Handles NaN and outlier values before interpolation.
        """
        # Remove NaN values and outliers
        mask = ~np.isnan(values)
        if self.config.outlier_removal:
            outlier_mask = self._remove_outliers(values)
            mask = mask & outlier_mask
    
        points_clean = points[mask]
        values_clean = values[mask]
    
        if len(values_clean) == 0:
            return np.full(len(profile_coords), np.nan)
    
        # 1. Compute bounding box of profile points and add buffer
        buffer_km = max(self.config.width or 2.0, 2.0) * 2  # Buffer: 2x swath width or 2km
        min_x, max_x = profile_coords[:, 0].min() - buffer_km, profile_coords[:, 0].max() + buffer_km
        min_y, max_y = profile_coords[:, 1].min() - buffer_km, profile_coords[:, 1].max() + buffer_km
    
        # 2. Only keep scattered points inside subregion
        sub_mask = (
            (points_clean[:, 0] >= min_x) & (points_clean[:, 0] <= max_x) &
            (points_clean[:, 1] >= min_y) & (points_clean[:, 1] <= max_y)
        )
        points_sub = points_clean[sub_mask]
        values_sub = values_clean[sub_mask]
    
        if len(values_sub) == 0:
            return np.full(len(profile_coords), np.nan)
    
        # 3. Only keep profile points inside subregion
        profile_mask = (
            (profile_coords[:, 0] >= min_x) & (profile_coords[:, 0] <= max_x) &
            (profile_coords[:, 1] >= min_y) & (profile_coords[:, 1] <= max_y)
        )
        profile_coords_sub = profile_coords[profile_mask]
    
        profile_values = np.full(len(profile_coords), np.nan)
        if len(profile_coords_sub) == 0:
            return profile_values
    
        # 4. Interpolation
        method = self.config.interpolation_method
        if method == 'nearest':
            tree = cKDTree(points_sub)
            _, indices = tree.query(profile_coords_sub)
            profile_values[profile_mask] = values_sub[indices]
        elif method == 'linear':
            interpolator = interpolate.LinearNDInterpolator(points_sub, values_sub)
            values_interp = interpolator(profile_coords_sub)
            profile_values[profile_mask] = values_interp
        elif method == 'cubic':
            interpolator = interpolate.CloughTocher2DInterpolator(points_sub, values_sub)
            values_interp = interpolator(profile_coords_sub)
            profile_values[profile_mask] = values_interp
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
        # Outlier removal for interpolated values
        if self.config.outlier_removal:
            profile_outlier_mask = self._remove_outliers(profile_values)
            profile_values[~profile_outlier_mask] = np.nan
    
    def _extract_swath_data(self,
                           data: np.ndarray,
                           X: np.ndarray,
                           Y: np.ndarray,
                           profile_coords: np.ndarray,
                           width: float) -> Dict:
        """
        Extract all data within swath width of profile using optimized method.
        All coordinates are in km.
        """
        # 1. Compute bounding box of profile and add buffer
        buffer_km = max(width, 2.0) * 2
        min_x, max_x = profile_coords[:, 0].min() - buffer_km, profile_coords[:, 0].max() + buffer_km
        min_y, max_y = profile_coords[:, 1].min() - buffer_km, profile_coords[:, 1].max() + buffer_km
    
        # 2. Find index range for X/Y (assume regular grid)
        x1d = X[0, :]
        y1d = Y[:, 0]
        x_mask = (x1d >= min_x) & (x1d <= max_x)
        y_mask = (y1d >= min_y) & (y1d <= max_y)
    
        # 3. Extract subregion
        X_sub = X[np.ix_(y_mask, x_mask)]
        Y_sub = Y[np.ix_(y_mask, x_mask)]
        data_sub = data[np.ix_(y_mask, x_mask)]
    
        # 4. Method selection (user or auto)
        method = getattr(self.config, 'swath_method', 'auto')
        total_points = data_sub.size
    
        # Default: use optimized for large data, shapely/geopandas for small data
        if method == 'auto':
            if total_points < 50000:
                return self._extract_swath_data_shapely(data_sub, X_sub, Y_sub, profile_coords, width)
            elif total_points < 500000:
                return self._extract_swath_data_geopandas(data_sub, X_sub, Y_sub, profile_coords, width)
            else:
                return self._extract_swath_data_optimized(data_sub, X_sub, Y_sub, profile_coords, width)
        elif method == 'shapely':
            return self._extract_swath_data_shapely(data_sub, X_sub, Y_sub, profile_coords, width)
        elif method == 'geopandas':
            return self._extract_swath_data_geopandas(data_sub, X_sub, Y_sub, profile_coords, width)
        elif method == 'optimized':
            return self._extract_swath_data_optimized(data_sub, X_sub, Y_sub, profile_coords, width)
        else:
            raise ValueError(f"Unknown swath extraction method: {method}")
    
    def _extract_swath_data_geopandas(self,
                                     data: np.ndarray,
                                     X: np.ndarray,
                                     Y: np.ndarray,
                                     profile_coords: np.ndarray,
                                     width: float) -> Dict:
        """
        Extract swath data using GeoPandas for efficient spatial operations (coordinates in km).
        """
        from shapely.geometry import Point, LineString
        
        # Flatten the grid coordinates and data
        x_flat = X.flatten()
        y_flat = Y.flatten()
        data_flat = data.flatten()
        
        # Apply outlier removal if enabled
        if self.config.outlier_removal:
            outlier_mask = self._remove_outliers(data_flat)
            valid_mask = ~np.isnan(data_flat) & outlier_mask
        else:
            valid_mask = ~np.isnan(data_flat)
        
        # Filter data
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        data_valid = data_flat[valid_mask]
        
        if len(data_valid) == 0:
            return {
                'coordinates': None,
                'values': None,
                'distances': None,
                'offsets': None
            }
        
        # Create GeoDataFrame with grid points
        grid_points = [Point(x, y) for x, y in zip(x_valid, y_valid)]
        gdf = gpd.GeoDataFrame({
            'geometry': grid_points,
            'value': data_valid,
            'x_coord': x_valid,
            'y_coord': y_valid
        })
        
        # Create profile line and buffer
        profile_line = LineString(profile_coords)
        swath_polygon = profile_line.buffer(width / 2)
        
        # Create GeoDataFrame for swath polygon
        swath_gdf = gpd.GeoDataFrame([1], geometry=[swath_polygon])
        
        # Spatial join to find points within swath
        points_in_swath = gpd.sjoin(gdf, swath_gdf, how='inner', predicate='within')
        
        if len(points_in_swath) == 0:
            return {
                'coordinates': None,
                'values': None,
                'distances': None,
                'offsets': None
            }
        
        # Extract coordinates and values
        swath_coords = []
        swath_values = []
        swath_distances = []
        swath_offsets = []
        
        for idx, row in points_in_swath.iterrows():
            point = row.geometry
            value = row['value']
            x_coord, y_coord = point.x, point.y
            
            # Calculate distance along profile
            distance_along_profile = profile_line.project(point)
            
            # Calculate perpendicular distance from profile line
            nearest_point_on_line = profile_line.interpolate(distance_along_profile)
            offset_distance = point.distance(nearest_point_on_line)
            
            # Determine sign of offset (left/right of profile)
            if len(profile_coords) >= 2:
                line_fraction = distance_along_profile / profile_line.length
                segment_idx = min(int(line_fraction * (len(profile_coords) - 1)), 
                                len(profile_coords) - 2)
                
                p1 = profile_coords[segment_idx]
                p2 = profile_coords[segment_idx + 1]
                
                profile_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                point_vec = np.array([x_coord - nearest_point_on_line.x, 
                                    y_coord - nearest_point_on_line.y])
                
                cross_product = np.cross(profile_vec, point_vec)
                offset_distance = offset_distance if cross_product >= 0 else -offset_distance
            
            swath_coords.append([x_coord, y_coord])
            swath_values.append(value)
            swath_distances.append(distance_along_profile)
            swath_offsets.append(offset_distance)
        
        return {
            'coordinates': np.array(swath_coords) if swath_coords else None,
            'values': np.array(swath_values) if swath_values else None,
            'distances': np.array(swath_distances) if swath_distances else None,
            'offsets': np.array(swath_offsets) if swath_offsets else None
        }
    
    def _extract_swath_data_shapely(self,
                                   data: np.ndarray,
                                   X: np.ndarray,
                                   Y: np.ndarray,
                                   profile_coords: np.ndarray,
                                   width: float) -> Dict:
        """
        Extract swath data using Shapely geometry (coordinates in km).
        """
        swath_coords = []
        swath_values = []
        swath_distances = []
        swath_offsets = []
        
        # Create profile line using Shapely
        profile_line = LineString(profile_coords)
        
        # Create swath polygon by buffering the profile line
        swath_polygon = profile_line.buffer(width / 2)
        
        # Flatten the grid coordinates and data
        x_flat = X.flatten()
        y_flat = Y.flatten()
        data_flat = data.flatten()
        
        # Apply outlier removal if enabled
        if self.config.outlier_removal:
            outlier_mask = self._remove_outliers(data_flat)
            valid_mask = ~np.isnan(data_flat) & outlier_mask
        else:
            valid_mask = ~np.isnan(data_flat)
        
        # Filter data
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        data_valid = data_flat[valid_mask]
        
        if len(data_valid) == 0:
            return {
                'coordinates': None,
                'values': None,
                'distances': None,
                'offsets': None
            }
        
        # Create points for valid grid locations
        grid_points = [Point(x, y) for x, y in zip(x_valid, y_valid)]
        
        # Use spatial indexing for efficiency with large datasets
        spatial_index = index.Index()
        for i, point in enumerate(grid_points):
            spatial_index.insert(i, point.bounds)
        
        # Get candidate points within bounding box of swath
        swath_bounds = swath_polygon.bounds
        candidate_indices = list(spatial_index.intersection(swath_bounds))
        
        # Check which candidate points are actually within the swath
        for idx in candidate_indices:
            point = grid_points[idx]
            
            if swath_polygon.contains(point) or swath_polygon.touches(point):
                # Point is within swath
                x_coord, y_coord = point.x, point.y
                value = data_valid[idx]
                
                # Calculate distance along profile
                distance_along_profile = profile_line.project(point)
                
                # Calculate perpendicular distance from profile line
                nearest_point_on_line = profile_line.interpolate(distance_along_profile)
                offset_distance = point.distance(nearest_point_on_line)
                
                # Determine sign of offset (left/right of profile)
                if len(profile_coords) >= 2:
                    line_fraction = distance_along_profile / profile_line.length
                    segment_idx = min(int(line_fraction * (len(profile_coords) - 1)), 
                                    len(profile_coords) - 2)
                    
                    p1 = profile_coords[segment_idx]
                    p2 = profile_coords[segment_idx + 1]
                    
                    profile_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                    point_vec = np.array([x_coord - nearest_point_on_line.x, 
                                        y_coord - nearest_point_on_line.y])
                    
                    cross_product = np.cross(profile_vec, point_vec)
                    offset_distance = offset_distance if cross_product >= 0 else -offset_distance
                
                swath_coords.append([x_coord, y_coord])
                swath_values.append(value)
                swath_distances.append(distance_along_profile)
                swath_offsets.append(offset_distance)
        
        return {
            'coordinates': np.array(swath_coords) if swath_coords else None,
            'values': np.array(swath_values) if swath_values else None,
            'distances': np.array(swath_distances) if swath_distances else None,
            'offsets': np.array(swath_offsets) if swath_offsets else None
        }
    
    def _extract_swath_data_optimized(self,
                                     data: np.ndarray,
                                     X: np.ndarray,
                                     Y: np.ndarray,
                                     profile_coords: np.ndarray,
                                     width: float) -> Dict:
        """
        Optimized swath extraction using vectorized operations (coordinates in km).
        This method is much faster than shapely/geopandas for large grids.
        Returns all points within swath width of the profile, with their distances and offsets.
        """
        # Flatten grid data
        x_flat = X.flatten()
        y_flat = Y.flatten()
        data_flat = data.flatten()
    
        # Outlier removal and NaN mask
        if self.config.outlier_removal:
            outlier_mask = self._remove_outliers(data_flat)
            valid_mask = ~np.isnan(data_flat) & outlier_mask
        else:
            valid_mask = ~np.isnan(data_flat)
    
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        data_valid = data_flat[valid_mask]
    
        if len(data_valid) == 0:
            return {
                'coordinates': None,
                'values': None,
                'distances': None,
                'offsets': None
            }
    
        # Prepare for vectorized calculation
        points = np.column_stack([x_valid, y_valid])
        n_segments = len(profile_coords) - 1
        half_width = width / 2
    
        # Precompute cumulative distances along profile for each segment
        segment_lengths = np.linalg.norm(np.diff(profile_coords, axis=0), axis=1)
        segment_cumlen = np.concatenate([[0], np.cumsum(segment_lengths)])
    
        # Collect results
        swath_coords = []
        swath_values = []
        swath_distances = []
        swath_offsets = []
    
        # For each segment, vectorized calculation
        for i in range(n_segments):
            p1 = profile_coords[i]
            p2 = profile_coords[i + 1]
            seg_vec = p2 - p1
            seg_len = np.linalg.norm(seg_vec)
            if seg_len == 0:
                continue
            seg_unit = seg_vec / seg_len
    
            # Vector from p1 to all points
            vecs = points - p1
            # Projection of each point onto the segment (distance along segment)
            proj = np.dot(vecs, seg_unit)
            # Only keep points whose projection falls within the segment
            mask_on_segment = (proj >= 0) & (proj <= seg_len)
            if not np.any(mask_on_segment):
                continue
    
            # For those points, compute perpendicular offset
            vecs_on = vecs[mask_on_segment]
            proj_on = proj[mask_on_segment]
            points_on = points[mask_on_segment]
            data_on = data_valid[mask_on_segment]
    
            # Perpendicular vector from segment to point
            perp_vecs = vecs_on - np.outer(proj_on, seg_unit)
            perp_dist = np.linalg.norm(perp_vecs, axis=1)
    
            # Only keep points within swath width
            mask_within_width = perp_dist <= half_width
            if not np.any(mask_within_width):
                continue
    
            final_points = points_on[mask_within_width]
            final_values = data_on[mask_within_width]
            final_proj = proj_on[mask_within_width]
            final_perp_vecs = perp_vecs[mask_within_width]
            final_perp_dist = perp_dist[mask_within_width]
    
            # Signed offset: left/right of profile
            cross = np.cross(seg_unit, final_perp_vecs)
            signed_offset = final_perp_dist * np.sign(cross)
    
            # Distance along profile = segment start + projection
            dist_along = segment_cumlen[i] + final_proj
    
            swath_coords.append(final_points)
            swath_values.append(final_values)
            swath_distances.append(dist_along)
            swath_offsets.append(signed_offset)
    
        # Concatenate all segments' results
        if len(swath_coords) == 0:
            return {
                'coordinates': None,
                'values': None,
                'distances': None,
                'offsets': None
            }
    
        swath_coords = np.vstack(swath_coords)
        swath_values = np.concatenate(swath_values)
        swath_distances = np.concatenate(swath_distances)
        swath_offsets = np.concatenate(swath_offsets)
    
        # Remove duplicates (same grid point may be close to multiple segments)
        coords_view = swath_coords.view([('', swath_coords.dtype)] * swath_coords.shape[1])
        _, unique_idx = np.unique(coords_view, return_index=True)
        swath_coords = swath_coords[unique_idx]
        swath_values = swath_values[unique_idx]
        swath_distances = swath_distances[unique_idx]
        swath_offsets = swath_offsets[unique_idx]
    
        return {
            'coordinates': swath_coords,
            'values': swath_values,
            'distances': swath_distances,
            'offsets': swath_offsets
        }
    
    def _extract_scattered_swath_data(self,
                                      points: np.ndarray,
                                      values: np.ndarray,
                                      profile_coords: np.ndarray,
                                      width: float,
                                      uncertainties: Optional[np.ndarray] = None) -> Dict:
        """
        Extract scattered points within swath of profile (coordinates in km).
        Optimized: first crop points using a bounding box with buffer, then spatial index.
        """
        # Apply outlier removal if enabled
        if self.config.outlier_removal:
            outlier_mask = self._remove_outliers(values)
            valid_mask = ~np.isnan(values) & outlier_mask
        else:
            valid_mask = ~np.isnan(values)
    
        points_clean = points[valid_mask]
        values_clean = values[valid_mask]
        uncertainties_clean = uncertainties[valid_mask] if uncertainties is not None else None
    
        if len(values_clean) == 0:
            return {
                'coordinates': None,
                'values': None,
                'distances': None,
                'offsets': None,
                'values_by_segment': [],
                'uncertainties': None,
                'uncertainties_by_segment': None
            }

        # --- Optimization: Use buffer to reduce point search area ---
        buffer_km = max(width, 2.0) * 2
        min_x, max_x = profile_coords[:, 0].min() - buffer_km, profile_coords[:, 0].max() + buffer_km
        min_y, max_y = profile_coords[:, 1].min() - buffer_km, profile_coords[:, 1].max() + buffer_km
        bbox_mask = (
            (points_clean[:, 0] >= min_x) & (points_clean[:, 0] <= max_x) &
            (points_clean[:, 1] >= min_y) & (points_clean[:, 1] <= max_y)
        )
        points_bbox = points_clean[bbox_mask]
        values_bbox = values_clean[bbox_mask]
        uncertainties_bbox = uncertainties_clean[bbox_mask] if uncertainties_clean is not None else None
    
        if len(values_bbox) == 0:
            return {
                'coordinates': None,
                'values': None,
                'distances': None,
                'offsets': None,
                'values_by_segment': [],
                'uncertainties': None,
                'uncertainties_by_segment': None
            }
    
        # Create profile line using Shapely
        profile_line = LineString(profile_coords)
        swath_polygon = profile_line.buffer(width / 2)
    
        # Convert points to Shapely Point objects
        point_geometries = [Point(x, y) for x, y in points_bbox]
    
        # Use spatial indexing for efficiency
        spatial_index = index.Index()
        for i, point in enumerate(point_geometries):
            spatial_index.insert(i, point.bounds)
    
        # Find points within swath
        swath_indices = []
        swath_coords = []
        swath_values = []
        swath_distances = []
        swath_offsets = []
        swath_uncertainties = []
    
        # Get candidate points within bounding box
        swath_bounds = swath_polygon.bounds
        candidate_indices = list(spatial_index.intersection(swath_bounds))
    
        for idx in candidate_indices:
            point = point_geometries[idx]
            if swath_polygon.contains(point) or swath_polygon.touches(point):
                value = values_bbox[idx]
                distance_along_profile = profile_line.project(point)
                nearest_point_on_line = profile_line.interpolate(distance_along_profile)
                offset_distance = point.distance(nearest_point_on_line)
                # Determine sign of offset
                if len(profile_coords) >= 2:
                    line_fraction = distance_along_profile / profile_line.length
                    segment_idx = min(int(line_fraction * (len(profile_coords) - 1)), len(profile_coords) - 2)
                    p1 = profile_coords[segment_idx]
                    p2 = profile_coords[segment_idx + 1]
                    profile_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                    point_vec = np.array([point.x - nearest_point_on_line.x, point.y - nearest_point_on_line.y])
                    cross_product = np.cross(profile_vec, point_vec)
                    offset_distance = offset_distance if cross_product >= 0 else -offset_distance
                swath_indices.append(idx)
                swath_coords.append([point.x, point.y])
                swath_values.append(value)
                swath_distances.append(distance_along_profile)
                swath_offsets.append(offset_distance)
                if uncertainties_bbox is not None:
                    swath_uncertainties.append(uncertainties_bbox[idx])
    
        # Organize data by profile segments for aggregation
        if len(swath_distances) > 0:
            profile_distances = self._calculate_profile_distances_km(profile_coords)
            n_segments = len(profile_coords) - 1
            values_by_segment = [[] for _ in range(n_segments)]
            uncertainties_by_segment = [[] for _ in range(n_segments)] if uncertainties_bbox is not None else None
            for i, dist_along in enumerate(swath_distances):
                segment_idx = np.searchsorted(profile_distances[1:], dist_along)
                segment_idx = min(segment_idx, n_segments - 1)
                values_by_segment[segment_idx].append(swath_values[i])
                if uncertainties_bbox is not None:
                    uncertainties_by_segment[segment_idx].append(swath_uncertainties[i])
        else:
            values_by_segment = []
            uncertainties_by_segment = None

        result = {
            'coordinates': np.array(swath_coords) if swath_coords else None,
            'values': np.array(swath_values) if swath_values else None,
            'distances': np.array(swath_distances) if swath_distances else None,
            'offsets': np.array(swath_offsets) if swath_offsets else None,
            'values_by_segment': values_by_segment
        }
        if uncertainties_bbox is not None:
            result['uncertainties'] = np.array(swath_uncertainties) if swath_uncertainties else None
            result['uncertainties_by_segment'] = uncertainties_by_segment

        return result
    
    def _aggregate_swath_to_profile(self, values_by_segment: List[np.ndarray]) -> np.ndarray:
        """Aggregate swath values to profile using specified method."""
        profile_values = []
        
        for segment_values in values_by_segment:
            if len(segment_values) == 0:
                profile_values.append(np.nan)
            else:
                if self.config.aggregation_method == 'mean':
                    profile_values.append(np.nanmean(segment_values))
                elif self.config.aggregation_method == 'median':
                    profile_values.append(np.nanmedian(segment_values))
                elif self.config.aggregation_method == 'max':
                    profile_values.append(np.nanmax(segment_values))
                elif self.config.aggregation_method == 'min':
                    profile_values.append(np.nanmin(segment_values))
        
        return np.array(profile_values)
    
    def compute_statistics(self, profile_data: ProfileData) -> Dict[str, np.ndarray]:
        """
        Compute statistical measures along the profile.
        
        Parameters:
        -----------
        profile_data : ProfileData
            Profile data to analyze
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary of statistical measures
        """
        stats_dict = {}
        values = profile_data.profile_values
        
        if values is None or len(values) == 0:
            return stats_dict
        
        # Remove NaN values for statistics
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        
        if len(valid_values) == 0:
            return stats_dict
        
        # Compute requested statistics
        for measure in self.config.statistical_measures:
            if measure == 'mean':
                stats_dict['mean'] = np.full_like(values, np.nanmean(values))
            elif measure == 'median':
                stats_dict['median'] = np.full_like(values, np.nanmedian(values))
            elif measure == 'std':
                stats_dict['std'] = np.full_like(values, np.nanstd(values))
            elif measure == 'q25':
                stats_dict['q25'] = np.full_like(values, np.nanpercentile(values, 25))
            elif measure == 'q75':
                stats_dict['q75'] = np.full_like(values, np.nanpercentile(values, 75))
            elif measure == 'min':
                stats_dict['min'] = np.full_like(values, np.nanmin(values))
            elif measure == 'max':
                stats_dict['max'] = np.full_like(values, np.nanmax(values))
        
        # Store in profile data
        profile_data.statistics = stats_dict
        
        return stats_dict
    
    def save_profile_data(self,
                         profile_data: ProfileData,
                         filepath: Union[str, Path],
                         format: str = 'auto') -> None:
        """
        Save profile data to file.
        
        Parameters:
        -----------
        profile_data : ProfileData
            Profile data to save
        filepath : str or Path
            Output file path
        format : str
            Output format ('csv', 'hdf5', 'json', 'pickle', 'auto')
        """
        filepath = Path(filepath)
        
        if format == 'auto':
            format = filepath.suffix.lower().lstrip('.')
            if format == 'h5':
                format = 'hdf5'
        
        if format == 'csv':
            self._save_csv(profile_data, filepath)
        elif format == 'hdf5':
            self._save_hdf5(profile_data, filepath)
        elif format == 'json':
            self._save_json(profile_data, filepath)
        elif format == 'pickle':
            self._save_pickle(profile_data, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_csv(self, profile_data: ProfileData, filepath: Path) -> None:
        """Save profile data as CSV."""
        # Create DataFrame
        data_dict = {
            'distance': profile_data.profile_distances,
            'longitude': profile_data.profile_coordinates_lonlat[:, 0],
            'latitude': profile_data.profile_coordinates_lonlat[:, 1],
            'value': profile_data.profile_values
        }
        
        if profile_data.profile_uncertainties is not None:
            data_dict['uncertainty'] = profile_data.profile_uncertainties
        
        # Add statistics
        for stat_name, stat_values in profile_data.statistics.items():
            data_dict[f'stat_{stat_name}'] = stat_values
        
        df = pd.DataFrame(data_dict)
        df.to_csv(filepath, index=False)
    
    def _save_hdf5(self, profile_data: ProfileData, filepath: Path) -> None:
        """Save profile data as HDF5."""
        with h5py.File(filepath, 'w') as f:
            # Main profile data
            f.create_dataset('profile_coordinates_lonlat', data=profile_data.profile_coordinates_lonlat)
            f.create_dataset('profile_coordinates_xy', data=profile_data.profile_coordinates_xy)
            f.create_dataset('profile_distances', data=profile_data.profile_distances)
            f.create_dataset('profile_values', data=profile_data.profile_values)
            
            if profile_data.profile_uncertainties is not None:
                f.create_dataset('profile_uncertainties', data=profile_data.profile_uncertainties)
            
            # Swath data if available
            if profile_data.swath_coordinates_lonlat is not None:
                swath_group = f.create_group('swath_data')
                swath_group.create_dataset('coordinates_lonlat', data=profile_data.swath_coordinates_lonlat)
                swath_group.create_dataset('coordinates_xy', data=profile_data.swath_coordinates_xy)
                swath_group.create_dataset('values', data=profile_data.swath_values)
                swath_group.create_dataset('distances', data=profile_data.swath_distances)
                swath_group.create_dataset('offsets', data=profile_data.swath_offsets)
            
            # Statistics
            if profile_data.statistics:
                stats_group = f.create_group('statistics')
                for stat_name, stat_values in profile_data.statistics.items():
                    stats_group.create_dataset(stat_name, data=stat_values)
            
            # Metadata
            f.attrs['data_type'] = profile_data.data_type
            if profile_data.units:
                f.attrs['units'] = profile_data.units
            if profile_data.description:
                f.attrs['description'] = profile_data.description
    
    def _save_json(self, profile_data: ProfileData, filepath: Path) -> None:
        """Save profile data as JSON."""
        # Convert numpy arrays to lists for JSON serialization
        data_dict = {
            'profile_coordinates_lonlat': profile_data.profile_coordinates_lonlat.tolist(),
            'profile_coordinates_xy': profile_data.profile_coordinates_xy.tolist(),
            'profile_distances': profile_data.profile_distances.tolist(),
            'profile_values': profile_data.profile_values.tolist(),
            'data_type': profile_data.data_type,
            'units': profile_data.units,
            'description': profile_data.description,
            'processing_info': profile_data.processing_info
        }
        
        if profile_data.profile_uncertainties is not None:
            data_dict['profile_uncertainties'] = profile_data.profile_uncertainties.tolist()
        
        # Add swath data if available
        if profile_data.swath_coordinates_lonlat is not None:
            data_dict['swath_coordinates_lonlat'] = profile_data.swath_coordinates_lonlat.tolist()
            data_dict['swath_coordinates_xy'] = profile_data.swath_coordinates_xy.tolist()
            data_dict['swath_values'] = profile_data.swath_values.tolist()
            data_dict['swath_distances'] = profile_data.swath_distances.tolist()
            data_dict['swath_offsets'] = profile_data.swath_offsets.tolist()
        
        # Add statistics
        if profile_data.statistics:
            data_dict['statistics'] = {
                name: values.tolist() for name, values in profile_data.statistics.items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(data_dict, f, indent=2)
    
    def _save_pickle(self, profile_data: ProfileData, filepath: Path) -> None:
        """Save profile data as pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(profile_data, f)
    
    def load_profile_data(self, filepath: Union[str, Path], format: str = 'auto') -> ProfileData:
        """
        Load profile data from file.
        
        Parameters:
        -----------
        filepath : str or Path
            Input file path
        format : str
            Input format ('csv', 'hdf5', 'json', 'pickle', 'auto')
            
        Returns:
        --------
        ProfileData
            Loaded profile data
        """
        filepath = Path(filepath)
        
        if format == 'auto':
            format = filepath.suffix.lower().lstrip('.')
            if format == 'h5':
                format = 'hdf5'
        
        if format == 'csv':
            return self._load_csv(filepath)
        elif format == 'hdf5':
            return self._load_hdf5(filepath)
        elif format == 'json':
            return self._load_json(filepath)
        elif format == 'pickle':
            return self._load_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _load_csv(self, filepath: Path) -> ProfileData:
        """Load profile data from CSV."""
        df = pd.read_csv(filepath)
        
        # Extract main profile data
        profile_coordinates_lonlat = np.column_stack([df['longitude'], df['latitude']])
        profile_distances = df['distance'].values
        profile_values = df['value'].values
        
        # Extract uncertainties if available
        profile_uncertainties = None
        if 'uncertainty' in df.columns:
            profile_uncertainties = df['uncertainty'].values
        
        # Extract statistics
        statistics = {}
        for col in df.columns:
            if col.startswith('stat_'):
                stat_name = col[5:]  # Remove 'stat_' prefix
                statistics[stat_name] = df[col].values
        
        # Create ProfileData object
        result = ProfileData(
            profile_coordinates_lonlat=profile_coordinates_lonlat,
            profile_distances=profile_distances,
            profile_values=profile_values,
            profile_uncertainties=profile_uncertainties,
            statistics=statistics,
            data_type='loaded_from_csv'
        )
        
        return result
    
    def _load_hdf5(self, filepath: Path) -> ProfileData:
        """Load profile data from HDF5."""
        with h5py.File(filepath, 'r') as f:
            # Load main profile data
            profile_coordinates_lonlat = f['profile_coordinates_lonlat'][:]
            profile_distances = f['profile_distances'][:]
            profile_values = f['profile_values'][:]
            
            # Load coordinates_xy if available
            profile_coordinates_xy = None
            if 'profile_coordinates_xy' in f:
                profile_coordinates_xy = f['profile_coordinates_xy'][:]
            
            # Load uncertainties if available
            profile_uncertainties = None
            if 'profile_uncertainties' in f:
                profile_uncertainties = f['profile_uncertainties'][:]
            
            # Load swath data if available
            swath_coordinates_lonlat = None
            swath_coordinates_xy = None
            swath_values = None
            swath_distances = None
            swath_offsets = None
            
            if 'swath_data' in f:
                swath_group = f['swath_data']
                if 'coordinates_lonlat' in swath_group:
                    swath_coordinates_lonlat = swath_group['coordinates_lonlat'][:]
                if 'coordinates_xy' in swath_group:
                    swath_coordinates_xy = swath_group['coordinates_xy'][:]
                if 'values' in swath_group:
                    swath_values = swath_group['values'][:]
                if 'distances' in swath_group:
                    swath_distances = swath_group['distances'][:]
                if 'offsets' in swath_group:
                    swath_offsets = swath_group['offsets'][:]
            
            # Load statistics if available
            statistics = {}
            if 'statistics' in f:
                stats_group = f['statistics']
                for stat_name in stats_group.keys():
                    statistics[stat_name] = stats_group[stat_name][:]
            
            # Load metadata
            data_type = f.attrs.get('data_type', 'loaded_from_hdf5')
            units = f.attrs.get('units', None)
            description = f.attrs.get('description', None)
            
            # Create ProfileData object
            result = ProfileData(
                profile_coordinates_lonlat=profile_coordinates_lonlat,
                profile_coordinates_xy=profile_coordinates_xy,
                profile_distances=profile_distances,
                profile_values=profile_values,
                profile_uncertainties=profile_uncertainties,
                swath_coordinates_lonlat=swath_coordinates_lonlat,
                swath_coordinates_xy=swath_coordinates_xy,
                swath_values=swath_values,
                swath_distances=swath_distances,
                swath_offsets=swath_offsets,
                statistics=statistics,
                data_type=data_type,
                units=units,
                description=description
            )
            
            return result
    
    def _load_json(self, filepath: Path) -> ProfileData:
        """Load profile data from JSON."""
        with open(filepath, 'r') as f:
            data_dict = json.load(f)
        
        # Convert lists back to numpy arrays
        profile_coordinates_lonlat = np.array(data_dict['profile_coordinates_lonlat'])
        profile_distances = np.array(data_dict['profile_distances'])
        profile_values = np.array(data_dict['profile_values'])
        
        # Load coordinates_xy if available
        profile_coordinates_xy = None
        if 'profile_coordinates_xy' in data_dict:
            profile_coordinates_xy = np.array(data_dict['profile_coordinates_xy'])
        
        # Load uncertainties if available
        profile_uncertainties = None
        if 'profile_uncertainties' in data_dict:
            profile_uncertainties = np.array(data_dict['profile_uncertainties'])
        
        # Load swath data if available
        swath_coordinates_lonlat = None
        swath_coordinates_xy = None
        swath_values = None
        swath_distances = None
        swath_offsets = None
        
        if 'swath_coordinates_lonlat' in data_dict:
            swath_coordinates_lonlat = np.array(data_dict['swath_coordinates_lonlat'])
        if 'swath_coordinates_xy' in data_dict:
            swath_coordinates_xy = np.array(data_dict['swath_coordinates_xy'])
        if 'swath_values' in data_dict:
            swath_values = np.array(data_dict['swath_values'])
        if 'swath_distances' in data_dict:
            swath_distances = np.array(data_dict['swath_distances'])
        if 'swath_offsets' in data_dict:
            swath_offsets = np.array(data_dict['swath_offsets'])
        
        # Load statistics if available
        statistics = {}
        if 'statistics' in data_dict:
            for stat_name, stat_values in data_dict['statistics'].items():
                statistics[stat_name] = np.array(stat_values)
        
        # Create ProfileData object
        result = ProfileData(
            profile_coordinates_lonlat=profile_coordinates_lonlat,
            profile_coordinates_xy=profile_coordinates_xy,
            profile_distances=profile_distances,
            profile_values=profile_values,
            profile_uncertainties=profile_uncertainties,
            swath_coordinates_lonlat=swath_coordinates_lonlat,
            swath_coordinates_xy=swath_coordinates_xy,
            swath_values=swath_values,
            swath_distances=swath_distances,
            swath_offsets=swath_offsets,
            statistics=statistics,
            data_type=data_dict.get('data_type', 'loaded_from_json'),
            units=data_dict.get('units', None),
            description=data_dict.get('description', None),
            processing_info=data_dict.get('processing_info', {})
        )
        
        return result

    def _load_pickle(self, filepath: Path) -> ProfileData:
        """Load profile data from pickle."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def configure_regional_ramp_removal(self,
                                       ramp_config: RampConfiguration = None,
                                       reference_point_mode: str = 'mean',
                                       reference_point: Tuple[float, float] = None,
                                       **kwargs) -> None:
        """
        Configure regional ramp removal settings.
        
        Parameters:
        -----------
        ramp_config : RampConfiguration, optional
            Complete ramp configuration object
        reference_point_mode : str
            Reference point mode: 'mean', 'none', 'user'
        reference_point : tuple, optional
            Reference point coordinates (required if mode='user')
        **kwargs : dict
            Configuration parameters to update
            
        Examples:
        --------
        # Using RampConfiguration object with custom reference point
        ramp_config = RampConfiguration()
        ramp_config.add_exclude_box(120.5, 122.5, 30.5, 32.5)
        ramp_config.set_reference_point('user', (121.5, 32.0))
        ramp_config.ramp_type = 'linear'
        analyzer.configure_regional_ramp_removal(ramp_config)
        
        # Using keyword arguments with mean reference point
        analyzer.configure_regional_ramp_removal(
            ramp_type='quadratic',
            exclude_box=(120.5, 122.5, 30.5, 32.5),
            reference_point_mode='mean',
            coord_type='lonlat'
        )
        
        # Using specific reference point
        analyzer.configure_regional_ramp_removal(
            ramp_type='linear',
            exclude_box=(120.5, 122.5, 30.5, 32.5),
            reference_point_mode='user',
            reference_point=(121.0, 31.5),
            coord_type='lonlat'
        )
        
        # No coordinate centering (for comparison)
        analyzer.configure_regional_ramp_removal(
            ramp_type='linear',
            exclude_box=(120.5, 122.5, 30.5, 32.5),
            reference_point_mode='none'
        )
        """
        if ramp_config is not None:
            self.ramp_remover.configure(ramp_config)
        else:
            # Create configuration from kwargs
            config = RampConfiguration()
            
            # Set reference point configuration
            config.set_reference_point(
                reference_point_mode, 
                reference_point, 
                kwargs.get('coord_type', 'lonlat')
            )
            
            # Handle exclude_box shortcut
            if 'exclude_box' in kwargs:
                exclude_box = kwargs.pop('exclude_box')
                coord_type = kwargs.get('coord_type', 'lonlat')
                config.add_exclude_box(
                    exclude_box[0], exclude_box[1], 
                    exclude_box[2], exclude_box[3], 
                    coord_type
                )
            
            # Apply other parameters
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            self.ramp_remover.configure(config)
    
    def remove_regional_ramp(self,
                           data: np.ndarray,
                           x_coords: np.ndarray,
                           y_coords: np.ndarray,
                           coord_type: str = 'lonlat',
                           reference_point_mode: str = None,
                           reference_point: Tuple[float, float] = None,
                           return_result_object: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, RampResult]]:
        """
        Remove regional ramp from matrix data using configured settings.
        
        Parameters:
        -----------
        data : np.ndarray
            2D data matrix to detrend
        x_coords : np.ndarray
            X-coordinate array
        y_coords : np.ndarray  
            Y-coordinate array
        coord_type : str
            'lonlat' if coordinates are in degrees, 'xy' if in km
        reference_point_mode : str, optional
            Reference point mode: 'mean', 'none', 'user'. Overrides config if provided.
        reference_point : tuple, optional
            Reference point coordinates. Required if mode='user'.
        return_result_object : bool
            Whether to return detailed RampResult object
            
        Returns:
        --------
        np.ndarray or Tuple[np.ndarray, RampResult]
            Detrended data matrix, optionally with detailed results
            
        Examples:
        --------
        # Configure and remove regional ramp with mean reference point
        analyzer.configure_regional_ramp_removal(
            ramp_type='linear',
            exclude_box=(120.5, 122.5, 30.5, 32.5),
            reference_point_mode='mean'
        )
        detrended_data = analyzer.remove_regional_ramp(
            elevation_matrix, lon_grid, lat_grid, coord_type='lonlat'
        )
        
        # Use specific reference point
        detrended_data, ramp_result = analyzer.remove_regional_ramp(
            elevation_matrix, lon_grid, lat_grid, 
            coord_type='lonlat',
            reference_point_mode='user',
            reference_point=(121.0, 31.5),
            return_result_object=True
        )
        ramp_result.print_summary()
        print(f"Reference point used: {ramp_result.reference_point_used}")
        
        # No coordinate centering for comparison
        detrended_data_no_center = analyzer.remove_regional_ramp(
            elevation_matrix, lon_grid, lat_grid, 
            coord_type='lonlat',
            reference_point_mode='none'
        )
        """
        result = self.ramp_remover.remove_ramp_from_matrix(
            data, x_coords, y_coords, coord_type, 
            reference_point_mode=reference_point_mode,
            reference_point=reference_point,
            return_result_object=True
        )
        
        if return_result_object:
            detrended_data, ramp_result = result
            # Store for automatic application to profile data
            self._ramp_result = ramp_result
            return detrended_data, ramp_result
        else:
            detrended_data, ramp_result = result
            # Store for automatic application to profile data
            self._ramp_result = ramp_result
            return detrended_data
    
    def apply_ramp_to_profile(self, 
                             profile_data: ProfileData, 
                             ramp_result: RampResult) -> ProfileData:
        """
        Apply previously fitted ramp to profile data.
        
        Parameters:
        -----------
        profile_data : ProfileData
            Profile data to which ramp should be applied
        ramp_result : RampResult
            Previous ramp fitting result
            
        Returns:
        --------
        ProfileData
            Profile data with ramp applied
        """
        import copy
        result_data = copy.deepcopy(profile_data)
        
        # Apply to profile values
        if result_data.profile_coordinates_xy is not None and result_data.profile_values is not None:
            ramp_values = self.ramp_remover.evaluate_ramp_at_points(
                result_data.profile_coordinates_xy, ramp_result, coord_type='xy'
            )
            result_data.profile_values = result_data.profile_values - ramp_values
        
        # Apply to swath data if available
        if (result_data.swath_coordinates_xy is not None and 
            result_data.swath_values is not None):
            
            ramp_swath = self.ramp_remover.evaluate_ramp_at_points(
                result_data.swath_coordinates_xy, ramp_result, coord_type='xy'
            )
            result_data.swath_values = result_data.swath_values - ramp_swath
        
        # Update processing info
        if result_data.processing_info is None:
            result_data.processing_info = {}
        result_data.processing_info['regional_ramp_removed'] = {
            'ramp_type': ramp_result.ramp_type,
            'coefficients': ramp_result.coefficients,
            'n_fit_points': ramp_result.n_fit_points,
            'fit_residual_rms': ramp_result.fit_residual_rms
        }
        
        print(f"Applied regional {ramp_result.ramp_type} ramp removal to profile data")
        
        return result_data
    
    def get_ramp_info(self) -> Dict[str, Any]:
        """
        Get information about regional ramp removal settings.
        
        Returns:
        --------
        Dict[str, Any]
            Ramp removal information including reference point details
        """
        if hasattr(self, '_ramp_result'):
            info = {
                "status": "Regional ramp fitted",
                "ramp_type": self._ramp_result.ramp_type,
                "equation": self._ramp_result.get_ramp_equation(),
                "reference_point_mode": self._ramp_result.reference_point_mode,
                "reference_point_used": self._ramp_result.reference_point_used,
                "reference_coord_type": self._ramp_result.reference_coord_type,
                "n_fit_points": self._ramp_result.n_fit_points,
                "fit_residual_rms": self._ramp_result.fit_residual_rms,
                "fit_r_squared": self._ramp_result.fit_r_squared,
                "condition_number": self._ramp_result.condition_number
            }
        else:
            info = {"status": "No regional ramp fitted"}
        
        return info
    
    def clear_ramp(self) -> None:
        """Clear regional ramp removal settings."""
        if hasattr(self, '_ramp_result'):
            delattr(self, '_ramp_result')
            print("Regional ramp settings cleared")
        else:
            print("No regional ramp settings to clear")
    
    def plot_ramp_analysis(self,
                          data: np.ndarray,
                          x_coords: np.ndarray,
                          y_coords: np.ndarray,
                          coord_type: str = 'lonlat',
                          **kwargs) -> plt.Figure:
        """
        Create visualization of ramp analysis results.
        
        Parameters:
        -----------
        data : np.ndarray
            Original data matrix
        x_coords : np.ndarray
            X coordinates
        y_coords : np.ndarray
            Y coordinates
        coord_type : str
            Coordinate type
        **kwargs : dict
            Additional plotting parameters
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        if not hasattr(self, '_ramp_result'):
            raise ValueError("No ramp result available. Run remove_regional_ramp first.")
        
        return self.ramp_remover.plot_ramp_analysis(
            data, x_coords, y_coords, self._ramp_result, coord_type, **kwargs
        )
    
    def plot_profile_summary(self, 
                           profile_data: Optional[ProfileData] = None,
                           show_statistics: bool = True,
                           show_swath: bool = True,
                           show_swath_points: bool = False,
                           show_interpolated_profile: bool = True,
                           show_swath_median: bool = False,
                           show_titles: bool = True,
                           figsize: Tuple[int, int] = (12, 8),
                           legend_position: str = 'auto',
                           cmap: str = 'jet',
                           vmin: Optional[float] = None,
                           vmax: Optional[float] = None,
                           symmetric_colorbar: bool = False,
                           colorbar_extend: str = 'both') -> plt.Figure:
        """
        Create comprehensive profile summary plot with enhanced colorbar control.
        
        Parameters:
        -----------
        profile_data : ProfileData, optional
            Profile data to plot. Uses current_profile if None.
        show_statistics : bool
            Whether to show statistical measures
        show_swath : bool
            Whether to show swath data if available
        show_swath_points : bool
            Whether to plot individual swath points as scatter in main profile
        show_interpolated_profile : bool
            Whether to show the interpolated profile line
        show_swath_median : bool
            Whether to show swath median line instead of/in addition to interpolated profile
        show_titles : bool
            Whether to show subplot titles
        figsize : tuple
            Figure size
        legend_position : str
            Legend position for map plot: 'auto', 'upper_left', 'upper_right', 
            'lower_left', 'lower_right', 'outside_right', 'outside_bottom', 'best'
        cmap : str
            Colormap name. Examples:
            - 'jet', 'viridis', 'plasma', 'inferno', 'magma'
            - 'RdBu_r', 'seismic', 'coolwarm' (for symmetric data)
            - 'cmc.roma_r', 'cmc.vik' (scientific colormaps)
        vmin : float, optional
            Minimum value for colorbar. If None, uses data minimum
        vmax : float, optional
            Maximum value for colorbar. If None, uses data maximum
        symmetric_colorbar : bool
            If True, makes colorbar symmetric around zero using max(abs(vmin), abs(vmax))
            Useful for displacement data or anomalies
        colorbar_extend : str
            How to extend colorbar beyond vmin/vmax:
            - 'neither': no extension
            - 'both': extend both ends
            - 'min': extend lower end only
            - 'max': extend upper end only
            
        Returns:
        --------
        plt.Figure
            Figure object
            
        Examples:
        --------
        # Basic usage with default jet colormap
        fig = analyzer.plot_profile_summary(result)
        
        # Use scientific colormap with custom range
        fig = analyzer.plot_profile_summary(result, cmap='cmc.roma_r', vmin=-50, vmax=50)
        
        # Symmetric colorbar for displacement data
        fig = analyzer.plot_profile_summary(result, cmap='RdBu_r', 
                                           vmin=-10, vmax=15, symmetric_colorbar=True)
        
        # Custom range with extension indicators
        fig = analyzer.plot_profile_summary(result, cmap='viridis', 
                                           vmin=0, vmax=100, colorbar_extend='max')
        """
        if profile_data is None:
            profile_data = self.current_profile
        
        if profile_data is None:
            raise ValueError("No profile data available. Run extraction first.")
        
        return self.visualizer.plot_profile_summary(
            profile_data, show_statistics, show_swath, 
            show_swath_points, show_interpolated_profile, show_swath_median, 
            show_titles, figsize, legend_position,
            cmap, vmin, vmax, symmetric_colorbar, colorbar_extend
        )

# Update the ProfileVisualizer class methods
class ProfileVisualizer:
    """Specialized visualization tools for profile analysis."""
    
    def __init__(self, analyzer: ProfileAnalyzer):
        """Initialize visualizer with reference to analyzer."""
        self.analyzer = analyzer
    
    # 在 ProfileVisualizer 类中添加新的方法和参数
    
    def plot_profile_summary(self,
                           profile_data: ProfileData,
                           show_statistics: bool = True,
                           show_swath: bool = True,
                           show_swath_points: bool = False,
                           show_interpolated_profile: bool = True,
                           show_swath_median: bool = False,
                           show_titles: bool = True,
                           figsize: Tuple[int, int] = (12, 8),
                           legend_position: str = 'auto',
                           cmap: str = 'jet',
                           vmin: Optional[float] = None,
                           vmax: Optional[float] = None,
                           symmetric_colorbar: bool = False,
                           colorbar_extend: str = 'both') -> plt.Figure:
        """
        Create comprehensive profile summary plot.
        
        Parameters:
        -----------
        profile_data : ProfileData
            Profile data to plot
        show_statistics : bool
            Whether to show statistical measures
        show_swath : bool
            Whether to show swath data if available
        show_swath_points : bool
            Whether to plot individual swath points as scatter
        show_interpolated_profile : bool
            Whether to show the interpolated profile line
        show_swath_median : bool
            Whether to show swath median line instead of interpolated profile
        show_titles : bool
            Whether to show subplot titles
        figsize : tuple
            Figure size
        legend_position : str
            Legend position for map plot: 'auto', 'upper_left', 'upper_right', 
            'lower_left', 'lower_right', 'outside_right', 'outside_bottom', 'best'
        cmap : str
            Colormap name (e.g., 'jet', 'viridis', 'RdBu_r', 'cmc.roma_r')
        vmin : float, optional
            Minimum value for colorbar. If None, uses data minimum
        vmax : float, optional
            Maximum value for colorbar. If None, uses data maximum
        symmetric_colorbar : bool
            If True, makes colorbar symmetric around zero using max(abs(vmin), abs(vmax))
        colorbar_extend : str
            Colorbar extension: 'neither', 'both', 'min', 'max'
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        
        # Set up the plotting style
        with sci_plot_style():
            # Determine subplot layout
            n_subplots = 2  # Profile + map
            if show_swath and profile_data.swath_values is not None:
                n_subplots = 3
            
            fig = plt.figure(figsize=figsize)
            
            # Create subplot layout with better spacing
            if n_subplots == 2:
                gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1],
                                    hspace=0.3, wspace=0.3)
                ax_profile = fig.add_subplot(gs[0, :])
                ax_map = fig.add_subplot(gs[1, 0])
                ax_hist = fig.add_subplot(gs[1, 1])
            else:
                gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2, 1],
                                    hspace=0.35, wspace=0.3)
                ax_profile = fig.add_subplot(gs[0, :])
                ax_swath = fig.add_subplot(gs[1, :])
                ax_map = fig.add_subplot(gs[2, 0])
                ax_hist = fig.add_subplot(gs[2, 1])
            
            # 计算统一的颜色范围
            colorbar_settings = self._calculate_colorbar_settings(
                profile_data, cmap, vmin, vmax, symmetric_colorbar, colorbar_extend
            )
            
            # Plot main profile
            self._plot_profile_line(
                ax_profile, profile_data, show_statistics,
                show_swath_points=show_swath_points,
                show_interpolated_profile=show_interpolated_profile,
                show_swath_median=show_swath_median,
                show_title=show_titles,
                colorbar_settings=colorbar_settings
            )
            
            # Plot swath if available
            if n_subplots == 3:
                self._plot_swath_data(ax_swath, profile_data, show_title=show_titles,
                                     colorbar_settings=colorbar_settings)
            
            # Plot map view
            self._plot_profile_map(ax_map, profile_data, legend_position=legend_position, 
                                 show_title=show_titles, colorbar_settings=colorbar_settings)
            
            # Plot histogram
            self._plot_value_histogram(ax_hist, profile_data, show_title=show_titles)
            
            # 调整整体布局
            # plt.tight_layout()
            
            return fig
    
    def _calculate_colorbar_settings(self,
                                   profile_data: ProfileData,
                                   cmap: str,
                                   vmin: Optional[float],
                                   vmax: Optional[float],
                                   symmetric_colorbar: bool,
                                   colorbar_extend: str) -> Dict[str, Any]:
        """
        Calculate unified colorbar settings for all subplots.
        
        Parameters:
        -----------
        profile_data : ProfileData
            Profile data
        cmap : str
            Colormap name
        vmin : float, optional
            User-specified minimum value
        vmax : float, optional
            User-specified maximum value
        symmetric_colorbar : bool
            Whether to make colorbar symmetric
        colorbar_extend : str
            Colorbar extension setting
            
        Returns:
        --------
        Dict[str, Any]
            Colorbar settings dictionary
        """
        
        # 收集所有可用的数值数据
        all_values = []
        
        # 添加剖面数值
        if profile_data.profile_values is not None:
            profile_valid = profile_data.profile_values[~np.isnan(profile_data.profile_values)]
            if len(profile_valid) > 0:
                all_values.extend(profile_valid)
        
        # 添加swath数值
        if profile_data.swath_values is not None:
            swath_valid = profile_data.swath_values[~np.isnan(profile_data.swath_values)]
            if len(swath_valid) > 0:
                all_values.extend(swath_valid)
        
        if len(all_values) == 0:
            # 没有有效数据，使用默认值
            data_min, data_max = 0.0, 1.0
        else:
            all_values = np.array(all_values)
            data_min = np.min(all_values)
            data_max = np.max(all_values)
        
        # 确定最终的 vmin 和 vmax
        final_vmin = vmin if vmin is not None else data_min
        final_vmax = vmax if vmax is not None else data_max
        
        # 应用对称设置
        if symmetric_colorbar:
            max_abs = max(abs(final_vmin), abs(final_vmax))
            final_vmin = -max_abs
            final_vmax = max_abs
        
        # 确定colorbar扩展设置
        actual_extend = colorbar_extend
        if colorbar_extend == 'neither':
            # 自动判断是否需要扩展
            if vmin is not None and data_min < final_vmin:
                if vmax is not None and data_max > final_vmax:
                    actual_extend = 'both'
                else:
                    actual_extend = 'min'
            elif vmax is not None and data_max > final_vmax:
                actual_extend = 'max'
        
        return {
            'cmap': cmap,
            'vmin': final_vmin,
            'vmax': final_vmax,
            'extend': actual_extend,
            'data_min': data_min,
            'data_max': data_max
        }
    
    def _plot_profile_line(self,
                         ax: plt.Axes,
                         profile_data: ProfileData,
                         show_statistics: bool,
                         show_swath_points: bool = False,
                         show_interpolated_profile: bool = True,
                         show_swath_median: bool = False,
                         show_title: bool = True,
                         colorbar_settings: Optional[Dict[str, Any]] = None) -> None:
        """
        Plot the main profile line with colorbar settings.
        """
        
        distances = profile_data.profile_distances
        values = profile_data.profile_values
        
        # 设置默认colorbar设置
        if colorbar_settings is None:
            colorbar_settings = self._calculate_colorbar_settings(
                profile_data, 'jet', None, None, False, 'neither'
            )
        
        # Plot reference line at distance = 0 if reference point is set
        if self.analyzer.config.reference_point is not None:
            ax.axvline(x=0, color='red', linestyle=':', linewidth=2, 
                      alpha=0.8, label='Reference point', zorder=5)
    
        # Plot swath points as scatter if requested
        if (show_swath_points and 
            profile_data.swath_distances is not None and 
            profile_data.swath_values is not None and 
            len(profile_data.swath_distances) > 0):
            
            swath_distances = profile_data.swath_distances
            swath_values = profile_data.swath_values
            
            # Apply filtering based on envelope method
            if self.analyzer.config.envelope_method == 'percentile':
                lower_pct, upper_pct = self.analyzer.config.envelope_percentiles
                
                # Calculate global percentiles for filtering
                valid_swath_values = swath_values[~np.isnan(swath_values)]
                if len(valid_swath_values) > 0:
                    global_lower = np.percentile(valid_swath_values, lower_pct)
                    global_upper = np.percentile(valid_swath_values, upper_pct)
                    
                    # Mask data within percentile range
                    percentile_mask = ((swath_values >= global_lower) & 
                                     (swath_values <= global_upper) & 
                                     ~np.isnan(swath_values))
                    
                    # Plot data within percentile range
                    if np.any(percentile_mask):
                        scatter = ax.scatter(
                            swath_distances[percentile_mask], 
                            swath_values[percentile_mask],
                            c=swath_values[percentile_mask], 
                            alpha=0.6, s=15, 
                            cmap=colorbar_settings['cmap'],
                            vmin=colorbar_settings['vmin'],
                            vmax=colorbar_settings['vmax'],
                            label=f'Swath points ({lower_pct:.0f}%-{upper_pct:.0f}%)',
                            zorder=2
                        )
                    
                    # Plot outliers with different style
                    outlier_mask = (~percentile_mask & ~np.isnan(swath_values))
                    if np.any(outlier_mask):
                        ax.scatter(
                            swath_distances[outlier_mask], 
                            swath_values[outlier_mask],
                            c='red', alpha=0.4, s=10, marker='x',
                            label='Outlier points',
                            zorder=2
                        )
            else:
                # Plot all swath points
                valid_mask = ~np.isnan(swath_values)
                if np.any(valid_mask):
                    scatter = ax.scatter(
                        swath_distances[valid_mask], 
                        swath_values[valid_mask],
                        c=swath_values[valid_mask], 
                        alpha=0.6, s=15, 
                        cmap=colorbar_settings['cmap'],
                        vmin=colorbar_settings['vmin'],
                        vmax=colorbar_settings['vmax'],
                        label='Swath points',
                        zorder=2
                    )
        
        # Calculate and plot swath median line if requested
        if (show_swath_median and 
            profile_data.swath_distances is not None and 
            profile_data.swath_values is not None and 
            len(profile_data.swath_distances) > 0):
            
            swath_median_distances, swath_median_values = self._calculate_swath_median_line(
                profile_data.profile_distances,
                profile_data.swath_distances,
                profile_data.swath_values
            )
            
            ax.plot(
                swath_median_distances, swath_median_values, 
                'g-', linewidth=2.5, 
                label='Swath median line', 
                zorder=4
            )
        
        # Plot interpolated profile line if requested
        if show_interpolated_profile and values is not None:
            ax.plot(distances, values, 'b-', linewidth=2, label='Interpolated profile', zorder=3)
        
        # Plot swath envelope if swath data is available
        if (profile_data.swath_distances is not None and 
            profile_data.swath_values is not None and 
            len(profile_data.swath_distances) > 0):
            
            # Calculate envelope curves for swath data
            if self.analyzer.config.envelope_method == 'percentile':
                swath_upper, swath_lower = self._calculate_swath_envelope_percentile(
                    profile_data.profile_distances,
                    profile_data.swath_distances,
                    profile_data.swath_values,
                    upper_percentile=self.analyzer.config.envelope_percentiles[1],
                    lower_percentile=self.analyzer.config.envelope_percentiles[0]
                )
            else:
                # Default to min/max envelope
                swath_upper, swath_lower = self._calculate_swath_envelope(
                    profile_data.profile_distances,
                    profile_data.swath_distances,
                    profile_data.swath_values
                )
            
            # Plot swath envelope
            ax.fill_between(
                distances, swath_lower, swath_upper,
                alpha=0.2, color='blue', 
                label=f'Swath envelope (±{self.analyzer.config.width/2:.1f} km)',
                zorder=1
            )
            
            # Plot envelope boundary lines
            ax.plot(distances, swath_upper, 'b--', alpha=0.6, linewidth=1, zorder=2)
            ax.plot(distances, swath_lower, 'b--', alpha=0.6, linewidth=1, zorder=2)
        
        # Error bars if uncertainties available and interpolated profile is shown
        if (show_interpolated_profile and 
            profile_data.profile_uncertainties is not None and 
            values is not None):
            ax.fill_between(
                distances,
                values - profile_data.profile_uncertainties,
                values + profile_data.profile_uncertainties,
                alpha=0.3, color='red', label='Profile uncertainty', zorder=2
            )
        
        # Statistical measures
        if show_statistics and profile_data.statistics:
            colors = ['red', 'green', 'orange', 'purple', 'brown']
            for i, (stat_name, stat_values) in enumerate(profile_data.statistics.items()):
                if stat_name in ['mean', 'median']:
                    ax.axhline(
                        stat_values[0], 
                        color=colors[i % len(colors)], 
                        linestyle='--', 
                        alpha=0.7,
                        label=f'{stat_name.capitalize()}: {stat_values[0]:.2f}',
                        zorder=3
                    )
            
        ax.set_xlabel('Distance along profile (km)')
        if self.analyzer.config.reference_point is not None:
            ax.set_xlabel('Distance along profile (km, relative to reference)')
        
        ax.set_ylabel(f'Value ({profile_data.units or "units"})')
        
        # Set title only if requested
        if show_title:
            # Update title based on what's being shown
            title_parts = []
            if show_interpolated_profile:
                title_parts.append('Interpolated Profile')
            if show_swath_median:
                title_parts.append('Swath Median')
            if show_swath_points:
                title_parts.append('Swath Points')
            
            if not title_parts:
                title_parts.append('Profile Analysis')
            
            if self.analyzer.config.reference_point is not None:
                title_parts.append('(Referenced)')
            
            ax.set_title(' & '.join(title_parts))
        
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_swath_data(self, ax: plt.Axes, profile_data: ProfileData, 
                        show_title: bool = True,
                        colorbar_settings: Optional[Dict[str, Any]] = None) -> None:
        """Plot swath data as scatter with unified colorbar settings."""
        
        if profile_data.swath_values is None:
            return
        
        # 设置默认colorbar设置
        if colorbar_settings is None:
            colorbar_settings = self._calculate_colorbar_settings(
                profile_data, 'jet', None, None, False, 'neither'
            )
        
        distances = profile_data.swath_distances
        offsets = profile_data.swath_offsets
        values = profile_data.swath_values
        
        # Plot based on envelope method
        if self.analyzer.config.envelope_method != 'percentile':
            scatter = ax.scatter(
                distances, offsets, c=values, 
                alpha=0.6, s=10, 
                cmap=colorbar_settings['cmap'],
                vmin=colorbar_settings['vmin'],
                vmax=colorbar_settings['vmax']
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, extend=colorbar_settings['extend'])
            cbar.set_label(f'Value ({profile_data.units or "units"})', fontsize=8)
            cbar.ax.tick_params(labelsize=8)
            
            # 添加数值范围信息
            if colorbar_settings['vmin'] != colorbar_settings['data_min'] or \
               colorbar_settings['vmax'] != colorbar_settings['data_max']:
                range_text = f"Data range: [{colorbar_settings['data_min']:.2f}, {colorbar_settings['data_max']:.2f}]\n"
                range_text += f"Display range: [{colorbar_settings['vmin']:.2f}, {colorbar_settings['vmax']:.2f}]"
                ax.text(0.02, 0.98, range_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=7,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            # For percentile method, show filtered data
            lower_pct, upper_pct = self.analyzer.config.envelope_percentiles
            
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                global_lower = np.percentile(valid_values, lower_pct)
                global_upper = np.percentile(valid_values, upper_pct)
                
                # Mask data outside percentile range
                percentile_mask = (values >= global_lower) & (values <= global_upper)
                
                # Plot data within percentile range
                scatter = ax.scatter(
                    distances[percentile_mask], offsets[percentile_mask], 
                    c=values[percentile_mask], 
                    alpha=0.6, s=10, 
                    cmap=colorbar_settings['cmap'],
                    vmin=colorbar_settings['vmin'],
                    vmax=colorbar_settings['vmax'],
                    label=f'{lower_pct:.0f}%-{upper_pct:.0f}% percentile data'
                )
                
                # Plot outliers with different style
                outlier_mask = ~percentile_mask & ~np.isnan(values)
                if np.any(outlier_mask):
                    ax.scatter(
                        distances[outlier_mask], offsets[outlier_mask],
                        c='red', alpha=0.3, s=5, marker='x',
                        label='Outliers'
                    )
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax, extend=colorbar_settings['extend'])
                cbar.set_label(f'Value ({profile_data.units or "units"})', fontsize=8)
                cbar.ax.tick_params(labelsize=8)
                ax.legend(fontsize=8)
        
        # Plot profile line
        profile_distances = profile_data.profile_distances
        ax.plot(profile_distances, np.zeros_like(profile_distances), 'r-', linewidth=2)
        
        ax.set_xlabel('Distance along profile (km)', fontsize=9)
        ax.set_ylabel('Distance from profile (km)', fontsize=9)
        ax.tick_params(labelsize=8)
        
        # Set title only if requested
        if show_title:
            if self.analyzer.config.envelope_method == 'percentile':
                lower_pct, upper_pct = self.analyzer.config.envelope_percentiles
                ax.set_title(f'Swath Data ({lower_pct:.0f}%-{upper_pct:.0f}% percentile filtered)', fontsize=10)
            else:
                ax.set_title('Swath Data', fontsize=10)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_profile_map(self, ax: plt.Axes, profile_data: ProfileData, 
                         legend_position: str = 'auto', show_title: bool = True,
                         colorbar_settings: Optional[Dict[str, Any]] = None) -> None:
        """Plot map view of the profile with unified colorbar settings."""
        
        # 设置默认colorbar设置
        if colorbar_settings is None:
            colorbar_settings = self._calculate_colorbar_settings(
                profile_data, 'jet', None, None, False, 'neither'
            )
        
        coords = profile_data.profile_coordinates_lonlat
        
        # Plot profile line
        ax.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=2, label='Profile')
        
        # Mark start and end points
        ax.plot(coords[0, 0], coords[0, 1], 'go', markersize=6, label='Start')
        ax.plot(coords[-1, 0], coords[-1, 1], 'ro', markersize=6, label='End')
    
        # Plot reference point and its projection if set
        if self.analyzer.config.reference_point is not None:
            ref_info = self.analyzer.get_reference_info()
            
            if "reference_lonlat" in ref_info:
                ref_lon, ref_lat = ref_info["reference_lonlat"]
                proj_lon, proj_lat = ref_info["projected_point_lonlat"]
                
                # Plot projected point on profile
                ax.plot(proj_lon, proj_lat, 'ko', markersize=6, markerfacecolor='orange',
                       markeredgecolor='black', markeredgewidth=1, label='Reference point')
        
        # Plot swath buffer zone if width is specified
        if hasattr(self.analyzer.config, 'width') and self.analyzer.config.width is not None:
            # Create swath buffer polygon using the profile coordinates in xy
            profile_xy = profile_data.profile_coordinates_xy
            
            if profile_xy is not None and len(profile_xy) >= 2:
                # Create buffer around profile line in xy coordinates (km)
                from shapely.geometry import LineString
                profile_line_xy = LineString(profile_xy)
                swath_polygon_xy = profile_line_xy.buffer(self.analyzer.config.width / 2)
                
                # Convert buffer polygon back to lonlat for plotting
                if hasattr(swath_polygon_xy, 'exterior'):
                    # Single polygon
                    buffer_coords_xy = np.array(swath_polygon_xy.exterior.coords)
                    buffer_coords_lonlat = np.zeros_like(buffer_coords_xy)
                    
                    for i, (x, y) in enumerate(buffer_coords_xy):
                        lon, lat = self.analyzer.xy2ll(x, y)
                        buffer_coords_lonlat[i] = [lon, lat]
                    
                    # Plot buffer area
                    ax.fill(buffer_coords_lonlat[:, 0], buffer_coords_lonlat[:, 1], 
                           alpha=0.2, color='red', label=f'Swath ({self.analyzer.config.width:.1f} km)')
                    ax.plot(buffer_coords_lonlat[:, 0], buffer_coords_lonlat[:, 1], 
                           'r--', alpha=0.5, linewidth=1)
                else:
                    # MultiPolygon - handle multiple polygons
                    for polygon in swath_polygon_xy.geoms:
                        if hasattr(polygon, 'exterior'):
                            buffer_coords_xy = np.array(polygon.exterior.coords)
                            buffer_coords_lonlat = np.zeros_like(buffer_coords_xy)
                            
                            for i, (x, y) in enumerate(buffer_coords_xy):
                                lon, lat = self.analyzer.xy2ll(x, y)
                                buffer_coords_lonlat[i] = [lon, lat]
                            
                            ax.fill(buffer_coords_lonlat[:, 0], buffer_coords_lonlat[:, 1], 
                                   alpha=0.2, color='red')
                            ax.plot(buffer_coords_lonlat[:, 0], buffer_coords_lonlat[:, 1], 
                                   'r--', alpha=0.5, linewidth=1)
        
        # Plot swath data points if available - filtered based on envelope method
        if profile_data.swath_coordinates_lonlat is not None:
            swath_coords = profile_data.swath_coordinates_lonlat
            
            if self.analyzer.config.envelope_method == 'percentile':
                # Filter swath data based on percentiles
                lower_pct, upper_pct = self.analyzer.config.envelope_percentiles
                values = profile_data.swath_values
                
                # Calculate global percentiles for filtering
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    global_lower = np.percentile(valid_values, lower_pct)
                    global_upper = np.percentile(valid_values, upper_pct)
                    
                    # Mask data within percentile range
                    percentile_mask = (values >= global_lower) & (values <= global_upper)
                    
                    # Plot data within percentile range
                    if np.any(percentile_mask):
                        scatter = ax.scatter(
                            swath_coords[percentile_mask, 0], swath_coords[percentile_mask, 1],
                            c=values[percentile_mask], alpha=0.6, s=10, 
                            cmap=colorbar_settings['cmap'],
                            vmin=colorbar_settings['vmin'],
                            vmax=colorbar_settings['vmax'],
                            label=f'Swath data ({lower_pct:.0f}%-{upper_pct:.0f}%)'
                        )
                    
                    # Plot outliers with different style
                    outlier_mask = ~percentile_mask & ~np.isnan(values)
                    if np.any(outlier_mask):
                        ax.scatter(
                            swath_coords[outlier_mask, 0], swath_coords[outlier_mask, 1],
                            c='red', alpha=0.3, s=5, marker='x',
                            label='Outliers'
                        )
            else:
                # Plot all swath data for minmax method
                scatter = ax.scatter(
                    swath_coords[:, 0], swath_coords[:, 1],
                    c=profile_data.swath_values, alpha=0.6, s=10, 
                    cmap=colorbar_settings['cmap'],
                    vmin=colorbar_settings['vmin'],
                    vmax=colorbar_settings['vmax'],
                    label='Swath data'
                )
            
            # Add a small colorbar for swath data only if scatter plot was created
            if 'scatter' in locals():
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                cbar_ax = inset_axes(ax, width="3%", height="40%", loc='center right', 
                                    bbox_to_anchor=(-0.05, 0, 1, 1), bbox_transform=ax.transAxes)
                cbar = plt.colorbar(scatter, cax=cbar_ax, extend=colorbar_settings['extend'])
                cbar.set_label(f'Value ({profile_data.units or "units"})', fontsize=7)
                cbar.ax.tick_params(labelsize=7)
        
        set_degree_formatter(ax, axis='both')
        ax.tick_params(labelsize=8)
        
        # Set title only if requested
        if show_title:
            # Update title based on envelope method
            if self.analyzer.config.envelope_method == 'percentile':
                lower_pct, upper_pct = self.analyzer.config.envelope_percentiles
                ax.set_title(f'Profile Location ({lower_pct:.0f}%-{upper_pct:.0f}% data shown)', fontsize=10)
            else:
                ax.set_title('Profile Location', fontsize=10)
        
        # 智能图例位置选择
        if legend_position == 'auto':
            # 获取数据范围和轴范围
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # 获取所有绘制元素的边界
            data_bounds = self._get_plot_data_bounds(coords, profile_data)
            
            # 计算可用空间并选择最佳图例位置
            legend_location = self._find_best_legend_position(xlim, ylim, data_bounds)
        else:
            legend_location = legend_position
    
        # 根据位置设置图例，使用更小的字体
        if legend_location == 'outside_right':
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                              fontsize=7, frameon=True, fancybox=True, shadow=True)
        elif legend_location == 'outside_bottom':
            legend = ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', 
                              ncol=3, fontsize=7, frameon=True, fancybox=True, shadow=True)
        elif legend_location == 'upper_right':
            legend = ax.legend(loc='upper right', fontsize=7, frameon=True, 
                              fancybox=True, shadow=True, framealpha=0.9)
        elif legend_location == 'upper_left':
            legend = ax.legend(loc='upper left', fontsize=7, frameon=True, 
                              fancybox=True, shadow=True, framealpha=0.9)
        elif legend_location == 'lower_left':
            legend = ax.legend(loc='lower left', fontsize=7, frameon=True, 
                              fancybox=True, shadow=True, framealpha=0.9)
        elif legend_location == 'lower_right':
            legend = ax.legend(loc='lower right', fontsize=7, frameon=True, 
                              fancybox=True, shadow=True, framealpha=0.9)
        else:
            legend = ax.legend(loc='best', fontsize=7, frameon=True, 
                              fancybox=True, shadow=True, framealpha=0.9)
        
        # 设置图例背景色以增强可读性
        if legend:
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            legend.get_frame().set_edgecolor('gray')
            legend.get_frame().set_linewidth(0.5)
        
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    def _plot_value_histogram(self, ax: plt.Axes, profile_data: ProfileData, show_title: bool = True) -> None:
        """Plot histogram of swath values or profile values if no swath data."""
        
        # Prioritize swath data if available
        if profile_data.swath_values is not None and len(profile_data.swath_values) > 0:
            values = profile_data.swath_values
            data_source = 'swath'
        else:
            values = profile_data.profile_values
            data_source = 'profile'
        
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Apply filtering based on envelope method for swath data
        if (data_source == 'swath' and 
            self.analyzer.config.envelope_method == 'percentile'):
            
            lower_pct, upper_pct = self.analyzer.config.envelope_percentiles
            global_lower = np.percentile(valid_values, lower_pct)
            global_upper = np.percentile(valid_values, upper_pct)
            
            # Separate filtered and outlier data
            percentile_mask = (valid_values >= global_lower) & (valid_values <= global_upper)
            filtered_values = valid_values[percentile_mask]
            outlier_values = valid_values[~percentile_mask]
            
            # Plot histogram of filtered data
            if len(filtered_values) > 0:
                n1, bins1, patches1 = ax.hist(
                    filtered_values, bins=30, alpha=0.7, edgecolor='black', 
                    color='blue', label=f'{lower_pct:.0f}%-{upper_pct:.0f}% data'
                )
            
            # Plot histogram of outliers
            if len(outlier_values) > 0:
                n2, bins2, patches2 = ax.hist(
                    outlier_values, bins=20, alpha=0.5, edgecolor='red', 
                    color='red', label='Outliers'
                )
            
            # Calculate statistics on filtered data
            if len(filtered_values) > 0:
                mean_val = np.mean(filtered_values)
                median_val = np.median(filtered_values)
                
                ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2,
                        label=f'Mean (filtered): {mean_val:.2f}')
                ax.axvline(median_val, color='darkgreen', linestyle='--', linewidth=2,
                        label=f'Median (filtered): {median_val:.2f}')
            
            # Add percentile boundaries
            ax.axvline(global_lower, color='orange', linestyle=':', alpha=0.8,
                    label=f'{lower_pct:.0f}% percentile: {global_lower:.2f}')
            ax.axvline(global_upper, color='orange', linestyle=':', alpha=0.8,
                    label=f'{upper_pct:.0f}% percentile: {global_upper:.2f}')
            
            title_suffix = f' ({lower_pct:.0f}%-{upper_pct:.0f}% filtered)'
            
        else:
            # Plot histogram of all data
            n, bins, patches = ax.hist(valid_values, bins=30, alpha=0.7, edgecolor='black')
            title_suffix = ''
        
        # Add standard deviation info if data_source is swath
        if data_source == 'swath':
            std_val = np.std(valid_values)
            ax.text(0.02, 0.98, f'Std: {std_val:.2f}\nN: {len(valid_values)}', 
                    transform=ax.transAxes, verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(f'Value ({profile_data.units or "units"})', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.tick_params(labelsize=8)
        
        # Set title only if requested
        if show_title:
            # Update title based on data source
            if data_source == 'swath':
                ax.set_title(f'Swath Value Distribution{title_suffix}', fontsize=10)
            else:
                ax.set_title('Profile Value Distribution', fontsize=10)
        
        ax.grid(True, alpha=0.3)
    
    def _calculate_swath_median_line(self,
                                   profile_distances: np.ndarray,
                                   swath_distances: np.ndarray,
                                   swath_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate median line for swath data.
        
        Parameters:
        -----------
        profile_distances : np.ndarray
            Distances along main profile
        swath_distances : np.ndarray
            Distances along profile for swath points
        swath_values : np.ndarray
            Values at swath points
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Distances and median values for the swath median line
        """
        median_distances = []
        median_values = []
        
        # For each profile point, find the median of swath values nearby
        for i, dist in enumerate(profile_distances):
            # Define a search window around this profile point
            window_size = self.analyzer.config.spacing / 2  # Half spacing on each side
            
            # Find swath points within the window
            mask = np.abs(swath_distances - dist) <= window_size
            
            if np.any(mask):
                window_values = swath_values[mask]
                valid_values = window_values[~np.isnan(window_values)]
                
                if len(valid_values) > 0:
                    # Apply filtering based on envelope method
                    if self.analyzer.config.envelope_method == 'percentile':
                        lower_pct, upper_pct = self.analyzer.config.envelope_percentiles
                        
                        # Calculate local percentiles for this window
                        if len(valid_values) >= 3:
                            local_lower = np.percentile(valid_values, lower_pct)
                            local_upper = np.percentile(valid_values, upper_pct)
                            
                            # Filter values within percentile range
                            filtered_values = valid_values[
                                (valid_values >= local_lower) & (valid_values <= local_upper)
                            ]
                            
                            if len(filtered_values) > 0:
                                median_val = np.median(filtered_values)
                            else:
                                median_val = np.median(valid_values)  # Fallback
                        else:
                            median_val = np.median(valid_values)
                    else:
                        median_val = np.median(valid_values)
                    
                    median_distances.append(dist)
                    median_values.append(median_val)
        
        return np.array(median_distances), np.array(median_values)
    
    def _calculate_swath_envelope(self,
                                profile_distances: np.ndarray,
                                swath_distances: np.ndarray,
                                swath_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate upper and lower envelope curves for swath data.
        
        Parameters:
        -----------
        profile_distances : np.ndarray
            Distances along main profile
        swath_distances : np.ndarray
            Distances along profile for swath points
        swath_values : np.ndarray
            Values at swath points
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Upper and lower envelope values at profile distances
        """
        upper_envelope = np.full_like(profile_distances, np.nan)
        lower_envelope = np.full_like(profile_distances, np.nan)
        
        # For each profile point, find the range of swath values nearby
        for i, dist in enumerate(profile_distances):
            # Define a search window around this profile point
            window_size = self.analyzer.config.spacing / 2  # Half spacing on each side
            
            # Find swath points within the window
            mask = np.abs(swath_distances - dist) <= window_size
            
            if np.any(mask):
                window_values = swath_values[mask]
                valid_values = window_values[~np.isnan(window_values)]
                
                if len(valid_values) > 0:
                    upper_envelope[i] = np.max(valid_values)
                    lower_envelope[i] = np.min(valid_values)
        
        # Interpolate gaps in envelope
        upper_envelope = self._interpolate_envelope_gaps(upper_envelope)
        lower_envelope = self._interpolate_envelope_gaps(lower_envelope)
        
        return upper_envelope, lower_envelope
    
    def _calculate_swath_envelope_percentile(self,
                                           profile_distances: np.ndarray,
                                           swath_distances: np.ndarray,
                                           swath_values: np.ndarray,
                                           upper_percentile: float = 95,
                                           lower_percentile: float = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate percentile-based envelope curves for swath data.
        
        Parameters:
        -----------
        profile_distances : np.ndarray
            Distances along main profile
        swath_distances : np.ndarray
            Distances along profile for swath points
        swath_values : np.ndarray
            Values at swath points
        upper_percentile : float
            Upper percentile (default 95%)
        lower_percentile : float
            Lower percentile (default 5%)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Upper and lower envelope values at profile distances
        """
        upper_envelope = np.full_like(profile_distances, np.nan)
        lower_envelope = np.full_like(profile_distances, np.nan)
        
        # For each profile point, find the range of swath values nearby
        for i, dist in enumerate(profile_distances):
            # Define a search window around this profile point
            window_size = self.analyzer.config.spacing / 2  # Half spacing on each side
            
            # Find swath points within the window
            mask = np.abs(swath_distances - dist) <= window_size
            
            if np.any(mask):
                window_values = swath_values[mask]
                valid_values = window_values[~np.isnan(window_values)]
                
                if len(valid_values) >= 3:  # Need at least 3 points for percentiles
                    upper_envelope[i] = np.percentile(valid_values, upper_percentile)
                    lower_envelope[i] = np.percentile(valid_values, lower_percentile)
                elif len(valid_values) > 0:
                    # Fallback to min/max for small samples
                    upper_envelope[i] = np.max(valid_values)
                    lower_envelope[i] = np.min(valid_values)
        
        # Interpolate gaps in envelope
        upper_envelope = self._interpolate_envelope_gaps(upper_envelope)
        lower_envelope = self._interpolate_envelope_gaps(lower_envelope)
        
        return upper_envelope, lower_envelope
    
    def _interpolate_envelope_gaps(self, envelope: np.ndarray) -> np.ndarray:
        """Interpolate NaN gaps in envelope curves."""
        valid_mask = ~np.isnan(envelope)
        
        if not np.any(valid_mask):
            return envelope
        
        if np.all(valid_mask):
            return envelope
        
        # Use linear interpolation to fill gaps
        valid_indices = np.where(valid_mask)[0]
        valid_values = envelope[valid_mask]
        
        # Interpolate only within the range of valid data
        all_indices = np.arange(len(envelope))
        
        # Only interpolate between existing points, don't extrapolate
        interp_mask = (all_indices >= valid_indices[0]) & (all_indices <= valid_indices[-1])
        
        if len(valid_values) > 1:
            envelope[interp_mask] = np.interp(
                all_indices[interp_mask], 
                valid_indices, 
                valid_values
            )
        
        return envelope
    
    def _get_plot_data_bounds(self, coords: np.ndarray, profile_data: ProfileData) -> Dict[str, float]:
        """获取绘制数据的边界范围"""
        bounds = {
            'lon_min': np.min(coords[:, 0]),
            'lon_max': np.max(coords[:, 0]),
            'lat_min': np.min(coords[:, 1]),
            'lat_max': np.max(coords[:, 1])
        }
        
        # 如果有swath数据，扩展边界
        if profile_data.swath_coordinates_lonlat is not None:
            swath_coords = profile_data.swath_coordinates_lonlat
            bounds['lon_min'] = min(bounds['lon_min'], np.min(swath_coords[:, 0]))
            bounds['lon_max'] = max(bounds['lon_max'], np.max(swath_coords[:, 0]))
            bounds['lat_min'] = min(bounds['lat_min'], np.min(swath_coords[:, 1]))
            bounds['lat_max'] = max(bounds['lat_max'], np.max(swath_coords[:, 1]))
        
        return bounds
    
    def _find_best_legend_position(self, xlim: Tuple[float, float], ylim: Tuple[float, float], 
                                  data_bounds: Dict[str, float]) -> str:
        """智能选择最佳图例位置"""
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # 计算数据占用的相对空间
        data_width_ratio = (data_bounds['lon_max'] - data_bounds['lon_min']) / x_range
        data_height_ratio = (data_bounds['lat_max'] - data_bounds['lat_min']) / y_range
        
        # 计算各个角落的空闲空间
        # 左上角空闲空间
        upper_left_space = (data_bounds['lon_min'] - xlim[0]) * (ylim[1] - data_bounds['lat_max'])
        # 右上角空闲空间  
        upper_right_space = (xlim[1] - data_bounds['lon_max']) * (ylim[1] - data_bounds['lat_max'])
        # 左下角空闲空间
        lower_left_space = (data_bounds['lon_min'] - xlim[0]) * (data_bounds['lat_min'] - ylim[0])
        # 右下角空闲空间
        lower_right_space = (xlim[1] - data_bounds['lon_max']) * (data_bounds['lat_min'] - ylim[0])
        
        # 如果数据占用空间太大（>80%），优先选择外部位置
        if data_width_ratio > 0.8 or data_height_ratio > 0.8:
            if x_range > y_range:
                return 'outside_bottom'  # 宽度更大，放底部
            else:
                return 'outside_right'   # 高度更大，放右侧
        
        # 找到最大空闲空间的角落
        spaces = {
            'upper_left': upper_left_space,
            'upper_right': upper_right_space, 
            'lower_left': lower_left_space,
            'lower_right': lower_right_space
        }
        
        best_position = max(spaces, key=spaces.get)
        
        # 如果最大空间还是太小，使用外部位置
        max_space = spaces[best_position]
        total_area = x_range * y_range
        
        if max_space < 0.1 * total_area:  # 小于10%的总面积
            if x_range > y_range:
                return 'outside_bottom'
            else:
                return 'outside_right'
        
        return best_position



# Update convenience functions to include reference point
def create_profile_from_endpoints(start: Tuple[float, float],
                                 end: Tuple[float, float],
                                 spacing: float = 1.0,
                                 width: Optional[float] = None,
                                 coord_type: str = 'lonlat',
                                 reference_point: Optional[Tuple[float, float]] = None,
                                 reference_coord_type: str = 'lonlat',
                                 lon0: float = None,
                                 lat0: float = None,
                                 utmzone: int = None) -> ProfileAnalyzer:
    """
    Create ProfileAnalyzer with profile defined by endpoints and optional reference point.
    
    Parameters:
    -----------
    start : tuple
        Starting point (lon, lat) or (x, y)
    end : tuple
        Ending point (lon, lat) or (x, y)
    spacing : float
        Point spacing in km
    width : float, optional
        Profile width for swath analysis in km
    coord_type : str
        'lonlat' or 'xy'
    reference_point : tuple, optional
        Reference point for distance zero (lon, lat) or (x, y)
    reference_coord_type : str
        Coordinate type for reference point: 'lonlat' or 'xy'
    lon0, lat0, utmzone : projection parameters
        
    Returns:
    --------
    ProfileAnalyzer
        Configured analyzer
    """
    analyzer = ProfileAnalyzer(
        name='endpoints_profile',
        lon0=lon0, lat0=lat0, utmzone=utmzone
    )
    analyzer.define_profile_from_endpoints(
        start, end, spacing, width, coord_type, reference_point, reference_coord_type
    )
    return analyzer

def create_profile_from_center_azimuth(center: Tuple[float, float],
                                      azimuth: float,
                                      length: float,
                                      spacing: float = 1.0,
                                      width: Optional[float] = None,
                                      coord_type: str = 'lonlat',
                                      reference_point: Optional[Tuple[float, float]] = None,
                                      reference_coord_type: str = 'lonlat',
                                      lon0: float = None,
                                      lat0: float = None,
                                      utmzone: int = None) -> ProfileAnalyzer:
    """
    Create ProfileAnalyzer with profile defined by center point and azimuth.
    
    Parameters:
    -----------
    center : tuple
        Center point (lon, lat) or (x, y)
    azimuth : float
        Azimuth angle in degrees (0=North, 90=East, clockwise)
    length : float
        Total profile length in km
    spacing : float
        Point spacing in km
    width : float, optional
        Profile width for swath analysis in km
    coord_type : str
        'lonlat' or 'xy'
    reference_point : tuple, optional
        Reference point for distance zero (lon, lat) or (x, y)
    reference_coord_type : str
        Coordinate type for reference point: 'lonlat' or 'xy'
    lon0, lat0, utmzone : projection parameters
        
    Returns:
    --------
    ProfileAnalyzer
        Configured analyzer
    """
    analyzer = ProfileAnalyzer(
        name='center_azimuth_profile',
        lon0=lon0, lat0=lat0, utmzone=utmzone
    )
    analyzer.define_profile_from_center_azimuth(center, azimuth, length, spacing, width, coord_type, 
                                                reference_point, reference_coord_type)
    return analyzer

def create_profile_from_points(points: List[Tuple[float, float]],
                              spacing: float = 1.0,
                              width: Optional[float] = None,
                              coord_type: str = 'lonlat',
                              reference_point: Optional[Tuple[float, float]] = None,
                              reference_coord_type: str = 'lonlat',
                              lon0: float = None,
                              lat0: float = None,
                              utmzone: int = None) -> ProfileAnalyzer:
    """
    Create ProfileAnalyzer with profile defined by multiple points.
    
    Parameters:
    -----------
    points : list
        List of points [(lon1, lat1), (lon2, lat2), ...] or [(x1, y1), ...]
    spacing : float
        Point spacing in km
    width : float, optional
        Profile width for swath analysis in km
    coord_type : str
        'lonlat' or 'xy'
    reference_point : tuple, optional
        Reference point for distance zero (lon, lat) or (x, y)
    reference_coord_type : str
        Coordinate type for reference point: 'lonlat' or 'xy'
    lon0, lat0, utmzone : projection parameters
        
    Returns:
    --------
    ProfileAnalyzer
        Configured analyzer
    """
    analyzer = ProfileAnalyzer(
        name='points_profile',
        lon0=lon0, lat0=lat0, utmzone=utmzone
    )
    analyzer.define_profile_from_points_list(points, spacing, width, coord_type,
                                              reference_point=reference_point,
                                              reference_coord_type=reference_coord_type)
    return analyzer

# Update convenience functions to support reference points
def remove_regional_trend(data,
                         x_coords,
                         y_coords,
                         exclude_region,
                         trend_type: Literal['constant', 'linear', 'quadratic'] = 'linear',
                         coord_type: str = 'lonlat',
                         reference_point_mode: str = 'mean',
                         reference_point: Optional[Tuple[float, float]] = None,
                         lon0: Optional[float] = None,
                         lat0: Optional[float] = None,
                         utmzone: Optional[int] = None,
                         return_details: bool = False):
    """
    Universal regional trend removal function with automatic data type identification (matrix or scatter points).
    
    Parameters:
    -----------
    data : np.ndarray
        Data array, can be:
        - 2D matrix (ny, nx): regular gridded data
        - 1D array (n_points,): scatter data values
        - 2D array (n_points, 3): scatter data with coordinates [x, y, value]
    x_coords : np.ndarray
        X coordinate array:
        - For matrix data: 1D array (nx,) or 2D array (ny, nx)
        - For scatter data: 1D array (n_points,) or None when data is (n_points,3)
    y_coords : np.ndarray
        Y coordinate array:
        - For matrix data: 1D array (ny,) or 2D array (ny, nx)  
        - For scatter data: 1D array (n_points,) or None when data is (n_points,3)
    exclude_region : tuple or dict or callable
        Exclusion region definition:
        - tuple: (min_x, max_x, min_y, max_y) rectangular box
        - dict: {'type': 'circle', 'center': (x, y), 'radius': r} circular region
        - callable: function accepting (x, y) returning bool, True means exclude
    trend_type : str
        Trend type: 'constant' (constant offset), 'linear' (linear), 'quadratic' (quadratic)
    coord_type : str
        Coordinate type: 'lonlat' or 'xy'
    reference_point_mode : str
        Reference point mode: 'mean' (fit points mean), 'none' (no centering), 'user' (user specified)
    reference_point : tuple, optional
        User specified reference point coordinates (required when mode='user')
    lon0, lat0, utmzone : float, optional
        Projection parameters
    return_details : bool
        Whether to return detailed results
        
    Returns:
    --------
    np.ndarray or Tuple[np.ndarray, RampResult]
        Detrended data, with detailed results if return_details=True
        
    Examples:
    --------
    # Matrix data - linear detrending using fit points mean as reference
    Z_detrend = remove_regional_trend(
        elevation_matrix, lon_grid, lat_grid,
        exclude_region=(120.5, 122.5, 30.5, 32.5),
        trend_type='linear', coord_type='lonlat',
        reference_point_mode='mean'
    )
    
    # Scatter data - constant offset removal using specific reference point
    points_detrend = remove_regional_trend(
        point_values, point_x, point_y,
        exclude_region=(120.5, 122.5, 30.5, 32.5),
        trend_type='constant', coord_type='lonlat',
        reference_point_mode='user',
        reference_point=(121.0, 31.5)
    )
    
    # No coordinate centering (for comparison)
    Z_detrend_no_center = remove_regional_trend(
        elevation_matrix, lon_grid, lat_grid,
        exclude_region=(120.5, 122.5, 30.5, 32.5),
        trend_type='linear', coord_type='lonlat',
        reference_point_mode='none'
    )
    
    # Get detailed results
    Z_detrend, ramp_result = remove_regional_trend(
        elevation_matrix, lon_grid, lat_grid,
        exclude_region=(120.5, 122.5, 30.5, 32.5),
        trend_type='linear', coord_type='lonlat',
        reference_point_mode='mean',
        return_details=True
    )
    print(f"Fit used {ramp_result.n_fit_points} points")
    print(f"Reference point: {ramp_result.reference_point_used}")
    print(f"Fit equation: {ramp_result.get_ramp_equation()}")
    """
    from .regional_ramp_removal import RegionalRampRemover, RampConfiguration
    
    # Automatically identify data type and format
    data_info = _identify_data_format(data, x_coords, y_coords)
    
    # Create RegionalRampRemover
    remover = RegionalRampRemover(
        name='trend_remover',
        lon0=lon0, lat0=lat0, utmzone=utmzone
    )
    
    # Configure trend removal parameters
    config = RampConfiguration()
    config.ramp_type = trend_type
    config.set_reference_point(reference_point_mode, reference_point, coord_type)
    
    # Handle exclude regions
    _configure_exclude_region(config, exclude_region, coord_type)
    
    remover.configure(config)
    
    # Call appropriate processing method based on data type
    if data_info['type'] == 'matrix':
        # Matrix data
        result = remover.remove_ramp_from_matrix(
            data_info['data'], data_info['x_coords'], data_info['y_coords'],
            coord_type, return_result_object=True
        )
    else:
        # Scattered point data
        result = remover.remove_ramp_from_points(
            data_info['coordinates'], data_info['values'],
            coord_type, return_result_object=True
        )
    
    detrended_data, ramp_result = result
    
    if return_details:
        return detrended_data, ramp_result
    else:
        return detrended_data

def _identify_data_format(data, x_coords, y_coords):
    """
    Automatically identify data format.
    
    Returns:
    --------
    dict: Contains data type and processed data
    """
    data = np.asarray(data)
    
    # Case 1: data is 2D matrix
    if data.ndim == 2 and data.shape[0] > 1 and data.shape[1] > 1:
        # Matrix data
        x_coords = np.asarray(x_coords) if x_coords is not None else None
        y_coords = np.asarray(y_coords) if y_coords is not None else None
        
        return {
            'type': 'matrix',
            'data': data,
            'x_coords': x_coords,
            'y_coords': y_coords
        }
    
    # Case 2: data is (n_points, 3) format [x, y, value]
    elif data.ndim == 2 and data.shape[1] == 3:
        coordinates = data[:, :2]  # x, y coordinates
        values = data[:, 2]        # values
        
        return {
            'type': 'points',
            'coordinates': coordinates,
            'values': values
        }
    
    # Case 3: data is 1D value array, with x_coords, y_coords
    elif data.ndim == 1:
        if x_coords is None or y_coords is None:
            raise ValueError("For 1D data array, both x_coords and y_coords must be provided")
        
        x_coords = np.asarray(x_coords).flatten()
        y_coords = np.asarray(y_coords).flatten()
        
        if len(x_coords) != len(data) or len(y_coords) != len(data):
            raise ValueError(f"Coordinate arrays length ({len(x_coords)}, {len(y_coords)}) "
                           f"must match data length ({len(data)})")
        
        coordinates = np.column_stack([x_coords, y_coords])
        
        return {
            'type': 'points',
            'coordinates': coordinates,
            'values': data
        }
    
    else:
        raise ValueError(f"Unsupported data format. Data shape: {data.shape}")

def _configure_exclude_region(config: RampConfiguration, exclude_region, coord_type: str):
    """
    Configure exclude regions.
    
    Parameters:
    -----------
    config : RampConfiguration
        Configuration object
    exclude_region : various types
        Exclude region definition
    coord_type : str
        Coordinate type
    """
    if exclude_region is None:
        return
    
    if isinstance(exclude_region, tuple) and len(exclude_region) == 4:
        # Rectangle box: (min_x, max_x, min_y, max_y)
        min_x, max_x, min_y, max_y = exclude_region
        config.add_exclude_box(min_x, max_x, min_y, max_y, coord_type)
        
    elif isinstance(exclude_region, dict):
        # Dictionary format region definition
        if exclude_region.get('type') == 'circle':
            center = exclude_region['center']
            radius = exclude_region['radius']
            config.add_exclude_circle(center[0], center[1], radius, coord_type)
            
        elif exclude_region.get('type') == 'polygon':
            vertices = exclude_region['vertices']
            config.add_exclude_polygon(vertices, coord_type)
            
        elif exclude_region.get('type') == 'box':
            bounds = exclude_region['bounds']  # [min_x, max_x, min_y, max_y]
            config.add_exclude_box(bounds[0], bounds[1], bounds[2], bounds[3], coord_type)
            
        else:
            raise ValueError(f"Unknown exclude region type: {exclude_region.get('type')}")
            
    elif callable(exclude_region):
        # Custom function
        config.add_exclude_function(exclude_region, coord_type)
        
    else:
        raise ValueError(f"Unsupported exclude_region format: {type(exclude_region)}")

# Update specific functions
def remove_regional_constant_trend(data,
                                  x_coords,
                                  y_coords,
                                  exclude_region,
                                  coord_type: str = 'lonlat',
                                  reference_point_mode: str = 'mean',
                                  reference_point: Optional[Tuple[float, float]] = None,
                                  lon0: Optional[float] = None,
                                  lat0: Optional[float] = None,
                                  utmzone: Optional[int] = None,
                                  return_details: bool = False):
    """
    Remove constant offset trend (global data uplift/subsidence).
    
    This is the simplest trend removal, removing only a global constant offset:
    corrected_data = original_data - constant_offset
    
    When using reference_point_mode='mean', coordinates are centered at the mean of
    fit points, so the constant term represents the data value at the reference point.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array (automatically detects matrix or scatter format)
    x_coords, y_coords : np.ndarray
        Coordinate arrays
    exclude_region : tuple or dict or callable
        Exclude region definition
    coord_type : str
        Coordinate type
    reference_point_mode : str
        Reference point mode: 'mean', 'none', 'user'
    reference_point : tuple, optional
        User-specified reference point (required when mode='user')
    lon0, lat0, utmzone : projection parameters
    return_details : bool
        Whether to return detailed results
        
    Returns:
    --------
    np.ndarray or Tuple[np.ndarray, RampResult]
        Data with constant offset removed
        
    Examples:
    --------
    # Remove GPS reference station bias using station mean as reference
    gps_corrected = remove_regional_constant_trend(
        gps_displacements, station_lons, station_lats,
        exclude_region=(epicenter_lon-0.5, epicenter_lon+0.5, 
                       epicenter_lat-0.5, epicenter_lat+0.5),
        coord_type='lonlat',
        reference_point_mode='mean'
    )
    
    # Use specific station as reference point
    gps_corrected, details = remove_regional_constant_trend(
        gps_displacements, station_lons, station_lats,
        exclude_region=fault_zone,
        coord_type='lonlat',
        reference_point_mode='user',
        reference_point=(reference_station_lon, reference_station_lat),
        return_details=True
    )
    print(f"Reference point: {details.reference_point_used}")
    print(f"Constant offset: {details.coefficients[0]:.3f}")
    """
    return remove_regional_trend(
        data, x_coords, y_coords, exclude_region,
        trend_type='constant', coord_type=coord_type,
        reference_point_mode=reference_point_mode,
        reference_point=reference_point,
        lon0=lon0, lat0=lat0, utmzone=utmzone,
        return_details=return_details
    )

def remove_regional_linear_trend(data,
                                x_coords,
                                y_coords,
                                exclude_region,
                                coord_type: str = 'lonlat',
                                reference_point_mode: str = 'mean',
                                reference_point: Optional[Tuple[float, float]] = None,
                                lon0: Optional[float] = None,
                                lat0: Optional[float] = None,
                                utmzone: Optional[int] = None,
                                return_details: bool = False):
    """
    Remove linear regional trend.
    
    Fit and remove linear trend surface: z = a*(x-x0) + b*(y-y0) + c
    where (x0, y0) is the reference point, which can be the mean of fit points,
    user-specified point, or origin.
    
    Benefits of using reference point:
    1. Improve numerical stability (avoid rounding errors from large coordinates)
    2. Make parameters more physically meaningful (c represents value at reference point)
    3. Improve condition number of the fit
    
    Parameters:
    -----------
    data : np.ndarray
        Data array (automatically detects matrix or scatter format)
    x_coords, y_coords : np.ndarray
        Coordinate arrays
    exclude_region : tuple or dict or callable
        Exclude region definition
    coord_type : str
        Coordinate type
    reference_point_mode : str
        Reference point mode: 'mean', 'none', 'user'
    reference_point : tuple, optional
        User-specified reference point (required when mode='user')
    lon0, lat0, utmzone : projection parameters
    return_details : bool
        Whether to return detailed results
        
    Returns:
    --------
    np.ndarray or Tuple[np.ndarray, RampResult]
        Data with linear trend removed
        
    Examples:
    --------
    # Remove orbital error from InSAR data using regional center as reference
    insar_corrected = remove_regional_linear_trend(
        insar_phase, lon_grid, lat_grid,
        exclude_region=(fault_zone_box),
        coord_type='lonlat',
        reference_point_mode='mean'
    )
    
    # 使用已知稳定点作为参考
    insar_corrected, result = remove_regional_linear_trend(
        insar_phase, lon_grid, lat_grid,
        exclude_region=(fault_zone_box),
        coord_type='lonlat',
        reference_point_mode='user',
        reference_point=(stable_point_lon, stable_point_lat),
        return_details=True
    )
    print(f"线性梯度: x方向 {result.coefficients[0]:.2e}/km, y方向 {result.coefficients[1]:.2e}/km")
    print(f"参考点 {result.reference_point_used} 处的值: {result.coefficients[2]:.3f}")
    """
    return remove_regional_trend(
        data, x_coords, y_coords, exclude_region,
        trend_type='linear', coord_type=coord_type,
        reference_point_mode=reference_point_mode,
        reference_point=reference_point,
        lon0=lon0, lat0=lat0, utmzone=utmzone,
        return_details=return_details
    )

def remove_regional_quadratic_trend(data,
                                   x_coords,
                                   y_coords,
                                   exclude_region,
                                   coord_type: str = 'lonlat',
                                   reference_point_mode: str = 'mean',
                                   reference_point: Optional[Tuple[float, float]] = None,
                                   lon0: Optional[float] = None,
                                   lat0: Optional[float] = None,
                                   utmzone: Optional[int] = None,
                                   return_details: bool = False):
    """
    去除二次区域趋势。
    
    拟合并去除二次趋势面：
    z = a*(x-x0)^2 + b*(y-y0)^2 + c*(x-x0)*(y-y0) + d*(x-x0) + e*(y-y0) + f
    
    其中(x0, y0)是参考点。二次项可以描述大尺度的弯曲变形。
    
    Parameters:
    -----------
    data : np.ndarray
        数据数组（自动识别矩阵或散点格式）
    x_coords, y_coords : np.ndarray
        坐标数组
    exclude_region : tuple or dict or callable
        排除区域定义
    coord_type : str
        坐标类型
    reference_point_mode : str
        参考点模式：'mean', 'none', 'user'
    reference_point : tuple, optional
        用户指定参考点（当mode='user'时需要）
    lon0, lat0, utmzone : projection parameters
    return_details : bool
        是否返回详细结果
        
    Returns:
    --------
    np.ndarray or Tuple[np.ndarray, RampResult]
        去除二次趋势后的数据
        
    Examples:
    --------
    # 去除地形相关的大气延迟，使用区域中心作为参考
    insar_corrected = remove_regional_quadratic_trend(
        insar_los_displacement, lon_grid, lat_grid,
        exclude_region=landslide_area,
        coord_type='lonlat',
        reference_point_mode='mean'
    )
    
    # 详细分析二次趋势
    insar_corrected, result = remove_regional_quadratic_trend(
        insar_los_displacement, lon_grid, lat_grid,
        exclude_region=landslide_area,
        coord_type='lonlat',
        reference_point_mode='user',
        reference_point=(region_center_lon, region_center_lat),
        return_details=True
    )
    
    a, b, c, d, e, f = result.coefficients
    print(f"二次项系数: x^2 {a:.2e}, y^2 {b:.2e}, xy {c:.2e}")
    print(f"线性项系数: x {d:.2e}, y {e:.2e}")
    print(f"参考点常数项: {f:.3f}")
    print(f"拟合条件数: {result.condition_number:.2e}")
    """
    return remove_regional_trend(
        data, x_coords, y_coords, exclude_region,
        trend_type='quadratic', coord_type=coord_type,
        reference_point_mode=reference_point_mode,
        reference_point=reference_point,
        lon0=lon0, lat0=lat0, utmzone=utmzone,
        return_details=return_details
    )

# 添加批量趋势去除函数
def compare_trend_removal_methods(data,
                                 x_coords,
                                 y_coords,
                                 exclude_region,
                                 coord_type: str = 'lonlat',
                                 lon0: Optional[float] = None,
                                 lat0: Optional[float] = None,
                                 utmzone: Optional[int] = None,
                                 plot_results: bool = True) -> Dict[str, Any]:
    """
    比较不同趋势去除方法的效果。
    
    Parameters:
    -----------
    data : np.ndarray
        原始数据
    x_coords, y_coords : np.ndarray
        坐标数组
    exclude_region : various
        排除区域
    coord_type : str
        坐标类型
    lon0, lat0, utmzone : projection parameters
    plot_results : bool
        是否绘制比较图
        
    Returns:
    --------
    Dict[str, Any]
        包含各种方法结果的字典
        
    Examples:
    --------
    results = compare_trend_removal_methods(
        elevation_data, lon_coords, lat_coords,
        exclude_region=(120.5, 122.5, 30.5, 32.5),
        coord_type='lonlat', plot_results=True
    )
    
    print("拟合质量比较:")
    for method in ['constant', 'linear', 'quadratic']:
        rms = results[method]['ramp_result'].fit_residual_rms
        r2 = results[method]['ramp_result'].fit_r_squared
        print(f"{method:>10}: RMS={rms:.3f}, R^2={r2:.3f}")
    """
    results = {}
    
    # 测试各种趋势去除方法
    methods = ['constant', 'linear', 'quadratic']
    
    for method in methods:
        try:
            detrended_data, ramp_result = remove_regional_trend(
                data, x_coords, y_coords, exclude_region,
                trend_type=method, coord_type=coord_type,
                lon0=lon0, lat0=lat0, utmzone=utmzone,
                return_details=True
            )
            
            results[method] = {
                'detrended_data': detrended_data,
                'ramp_result': ramp_result,
                'fit_quality': {
                    'rms': ramp_result.fit_residual_rms,
                    'r_squared': ramp_result.fit_r_squared,
                    'n_points': ramp_result.n_fit_points
                }
            }
            
            print(f"{method.capitalize()} trend removal:")
            print(f"  Fit points: {ramp_result.n_fit_points}")
            print(f"  RMS residual: {ramp_result.fit_residual_rms:.4f}")
            print(f"  R-squared: {ramp_result.fit_r_squared:.4f}")
            print(f"  Equation: {ramp_result.get_ramp_equation()}")
            print()
            
        except Exception as e:
            print(f"Warning: {method} trend removal failed: {e}")
            results[method] = None
    
    # 绘制比较图
    if plot_results and any(r is not None for r in results.values()):
        _plot_trend_comparison(data, x_coords, y_coords, results, coord_type)
    
    return results

def _plot_trend_comparison(original_data, x_coords, y_coords, results, coord_type):
    """绘制趋势去除方法比较图。"""
    import matplotlib.pyplot as plt
    from ...plottools import sci_plot_style
    
    with sci_plot_style():
        # 统计有效结果
        valid_methods = [method for method, result in results.items() if result is not None]
        n_methods = len(valid_methods)
        
        if n_methods == 0:
            print("No valid results to plot")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
        if n_methods == 0:
            axes = axes.reshape(2, -1)
        
        # 计算统一的颜色范围
        all_data = [original_data]
        for method in valid_methods:
            all_data.append(results[method]['detrended_data'])
        
        vmin = np.nanmin([np.nanmin(data) for data in all_data])
        vmax = np.nanmax([np.nanmax(data) for data in all_data])
        
        # 绘制原始数据
        _plot_single_data(axes[0, 0], original_data, x_coords, y_coords, 
                         "Original Data", vmin, vmax, coord_type)
        _plot_histogram(axes[1, 0], original_data, "Original")
        
        # 绘制各种去趋势结果
        for i, method in enumerate(valid_methods):
            result = results[method]
            detrended_data = result['detrended_data']
            ramp_result = result['ramp_result']
            
            title = f"{method.capitalize()} Detrended\n(RMS: {ramp_result.fit_residual_rms:.3f})"
            
            _plot_single_data(axes[0, i + 1], detrended_data, x_coords, y_coords,
                             title, vmin, vmax, coord_type)
            _plot_histogram(axes[1, i + 1], detrended_data, f"{method.capitalize()}")
        
        plt.tight_layout()
        plt.show()

def _plot_single_data(ax, data, x_coords, y_coords, title, vmin, vmax, coord_type):
    """绘制单个数据图。"""
    if data.ndim == 2:
        # 矩阵数据
        if x_coords.ndim == 1 and y_coords.ndim == 1:
            X, Y = np.meshgrid(x_coords, y_coords)
        else:
            X, Y = x_coords, y_coords
            
        im = ax.contourf(X, Y, data, levels=50, vmin=vmin, vmax=vmax, cmap='jet')
        ax.contour(X, Y, data, levels=10, colors='k', alpha=0.3, linewidths=0.5)
        plt.colorbar(im, ax=ax, shrink=0.8)
        
    else:
        # 散点数据 - 需要从 _identify_data_format 的逻辑推断坐标
        if hasattr(x_coords, '__len__') and len(x_coords) == len(data):
            scatter = ax.scatter(x_coords, y_coords, c=data, vmin=vmin, vmax=vmax, cmap='jet', s=10)
            plt.colorbar(scatter, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'Cannot plot\nscatter data', ha='center', va='center',
                   transform=ax.transAxes)
    
    ax.set_title(title, fontsize=10)
    if coord_type == 'lonlat':
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    else:
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')

def _plot_histogram(ax, data, label):
    """绘制数据直方图。"""
    valid_data = data[~np.isnan(data)]
    if len(valid_data) > 0:
        ax.hist(valid_data, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax.axvline(np.mean(valid_data), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(valid_data):.2f}')
        ax.axvline(np.median(valid_data), color='green', linestyle='--',
                  label=f'Median: {np.median(valid_data):.2f}')
        ax.legend(fontsize=8)
    
    ax.set_title(f'{label} Distribution', fontsize=10)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

# 高级应用函数
def auto_trend_removal(data,
                      x_coords,
                      y_coords,
                      exclude_region,
                      coord_type: str = 'lonlat',
                      selection_criteria: str = 'rms',
                      lon0: Optional[float] = None,
                      lat0: Optional[float] = None,
                      utmzone: Optional[int] = None) -> Tuple[np.ndarray, str, Any]:
    """
    自动选择最佳趋势去除方法。
    
    Parameters:
    -----------
    data : np.ndarray
        原始数据
    x_coords, y_coords : np.ndarray
        坐标数组
    exclude_region : various
        排除区域
    coord_type : str
        坐标类型
    selection_criteria : str
        选择标准：'rms'（最小残差）, 'r_squared'（最大相关系数）, 'aic'（AIC准则）
    lon0, lat0, utmzone : projection parameters
        
    Returns:
    --------
    Tuple[np.ndarray, str, RampResult]
        最佳去趋势数据、选择的方法、详细结果
        
    Examples:
    --------
    best_data, best_method, ramp_result = auto_trend_removal(
        insar_data, lon_coords, lat_coords,
        exclude_region=fault_area,
        selection_criteria='aic'
    )
    print(f"自动选择的最佳方法: {best_method}")
    """
    # 比较所有方法
    results = compare_trend_removal_methods(
        data, x_coords, y_coords, exclude_region,
        coord_type=coord_type, lon0=lon0, lat0=lat0, utmzone=utmzone,
        plot_results=False
    )
    
    # 根据选择标准找到最佳方法
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        raise ValueError("No trend removal method succeeded")
    
    best_method = None
    best_score = None
    
    for method, result in valid_results.items():
        fit_quality = result['fit_quality']
        
        if selection_criteria == 'rms':
            # 最小残差
            score = fit_quality['rms']
            if best_score is None or score < best_score:
                best_score = score
                best_method = method
                
        elif selection_criteria == 'r_squared':
            # 最大相关系数
            score = fit_quality['r_squared']
            if best_score is None or score > best_score:
                best_score = score
                best_method = method
                
        elif selection_criteria == 'aic':
            # AIC准则（考虑模型复杂度）
            n = fit_quality['n_points']
            rms = fit_quality['rms']
            
            # 参数个数：constant=1, linear=3, quadratic=6
            k = {'constant': 1, 'linear': 3, 'quadratic': 6}[method]
            
            # 计算AIC (Akaike Information Criterion)
            aic = n * np.log(rms**2) + 2 * k
            
            if best_score is None or aic < best_score:
                best_score = aic
                best_method = method
        
        else:
            raise ValueError(f"Unknown selection criteria: {selection_criteria}")
    
    print(f"自动选择最佳趋势去除方法: {best_method}")
    print(f"选择标准 ({selection_criteria}): {best_score:.4f}")
    
    return (results[best_method]['detrended_data'], 
            best_method, 
            results[best_method]['ramp_result'])

# Convenience functions for quick analysis
def quick_profile_analysis(data: np.ndarray,
                          x_coords: np.ndarray,
                          y_coords: np.ndarray,
                          start: Tuple[float, float],
                          end: Tuple[float, float],
                          spacing: float = 1.0,
                          coord_type: str = 'lonlat',
                          lon0: float = None,
                          lat0: float = None,
                          utmzone: int = None) -> ProfileData:
    """
    Quick profile analysis function.
    
    Parameters:
    -----------
    data : np.ndarray
        2D data array
    x_coords : np.ndarray
        X coordinates (lon or x)
    y_coords : np.ndarray
        Y coordinates (lat or y)
    start : tuple
        Start point
    end : tuple
        End point
    spacing : float
        Point spacing in km
    coord_type : str
        'lonlat' or 'xy'
    lon0, lat0, utmzone : float, float, int
        Projection parameters
        
    Returns:
    --------
    ProfileData
        Profile analysis results
    """
    config = ProfileConfiguration(
        start_point=start,
        end_point=end,
        spacing=spacing
    )
    
    analyzer = ProfileAnalyzer(
        name='quick_analysis',
        lon0=lon0, lat0=lat0, utmzone=utmzone,
        config=config
    )
    result = analyzer.extract_from_matrix(data, x_coords, y_coords, coord_type=coord_type)
    analyzer.compute_statistics(result)
    
    return result


# Example usage
if __name__ == "__main__":
    # Create test data
    lon = np.linspace(120, 125, 101)
    lat = np.linspace(30, 35, 101)
    LON, LAT = np.meshgrid(lon, lat)
    
    # Create synthetic elevation data
    Z = 1000 * np.exp(-((LON-122.5)**2 + (LAT-32.5)**2) / (2 * 1**2)) + np.random.normal(0, 50, LON.shape)
    
    # Quick analysis
    result = quick_profile_analysis(
        Z, lon, lat, 
        start=(120.5, 30.5), 
        end=(124.5, 34.5), 
        spacing=1.0,  # 1km
        coord_type='lonlat',
        lon0=122.5, lat0=32.5, utmzone=51
    )
    
    print(f"Profile extracted {len(result)} points")
    print(f"Value range: {result.get_bounds()}")
    print(f"Distance range: {result.profile_distances[0]:.1f} - {result.profile_distances[-1]:.1f} km")