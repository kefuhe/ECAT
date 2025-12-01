"""
LayeredDipFault - A class for handling faults with depth-varying dip angles.

This module provides functionality for:
- Interpolating dip angles that vary with depth
- Generating multi-layer fault geometry
- Supporting various input formats (functions, discrete profiles, Slab2.0 data)

Author: Added by kfhe
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Union, List, Dict, Callable, Optional, Tuple
from .AdaptiveTriangularPatches import AdaptiveTriangularPatches


def normalize_dip_to_0_180(dip: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Normalize dip angle from [-90, 90] to [0, 180] range.
    
    This conversion ensures smooth interpolation across the vertical (90°) boundary.
    - Positive dips (0 to 90) remain unchanged
    - Negative dips (-90 to 0) are converted to (90 to 180)
    
    Parameters:
    -----------
    dip : float or np.ndarray
        Dip angle(s) in range [-90, 90].
        
    Returns:
    --------
    float or np.ndarray
        Normalized dip angle(s) in range [0, 180].
    
    Examples:
    ---------
    >>> normalize_dip_to_0_180(30)   # -> 30
    >>> normalize_dip_to_0_180(-30)  # -> 150
    >>> normalize_dip_to_0_180(90)   # -> 90
    >>> normalize_dip_to_0_180(-90)  # -> 180 (or 0, depending on convention)
    """
    dip = np.asarray(dip)
    normalized = np.where(dip < 0, 180 + dip, dip)
    return float(normalized) if normalized.ndim == 0 else normalized


def normalize_dip_to_neg90_90(dip: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert dip angle from [0, 180] back to [-90, 90] range.
    
    Parameters:
    -----------
    dip : float or np.ndarray
        Dip angle(s) in range [0, 180].
        
    Returns:
    --------
    float or np.ndarray
        Dip angle(s) in range [-90, 90].
    """
    dip = np.asarray(dip)
    normalized = np.where(dip > 90, dip - 180, dip)
    return float(normalized) if normalized.ndim == 0 else normalized


class DepthDipProfile:
    """
    A class to represent depth-dip relationship at a reference node.
    
    Note: Internally stores dip in [0, 180] range for smooth interpolation.
    Returns dip in [-90, 90] range when queried.
    
    Attributes:
    -----------
    x : float
        X coordinate of the reference node (UTM).
    y : float
        Y coordinate of the reference node (UTM).
    lon : float
        Longitude of the reference node.
    lat : float
        Latitude of the reference node.
    depth_dip_pairs : np.ndarray
        Array of shape (n, 2) with columns [depth, dip] (dip in 0-180 range internally).
    interpolator : callable
        Interpolation function for dip(depth).
    interpolation_method : str
        Method used for interpolation.
    """
    
    def __init__(self, x: float, y: float, lon: float, lat: float, 
                 depth_dip_pairs: np.ndarray, interpolation_method: str = 'linear',
                 input_dip_range: str = 'neg90_90'):
        """
        Initialize a DepthDipProfile.
        
        Parameters:
        -----------
        x, y : float
            UTM coordinates.
        lon, lat : float
            Geographic coordinates.
        depth_dip_pairs : np.ndarray
            Array of shape (n, 2) with columns [depth, dip].
        interpolation_method : str
            Interpolation method:
            - 'linear': Linear interpolation between depth points
            - 'cubic': Cubic spline interpolation (smooth curves)
            - 'nearest': Nearest neighbor (same as 'step' but scipy style)
            - 'step' or 'previous': Step function, dip equals the value at 
              the previous (shallower) depth point. Creates hard transitions
              at depth boundaries. Extrapolation uses linear method.
        input_dip_range : str
            Range of input dip values:
            - 'neg90_90': Input is in [-90, 90] range (default, will be converted)
            - '0_180': Input is already in [0, 180] range
        """
        self.x = x
        self.y = y
        self.lon = lon
        self.lat = lat
        self.interpolation_method = interpolation_method
        self.depth_dip_pairs = np.atleast_2d(depth_dip_pairs).copy()
        
        # Normalize dip to [0, 180] for internal storage
        if input_dip_range == 'neg90_90':
            self.depth_dip_pairs[:, 1] = normalize_dip_to_0_180(self.depth_dip_pairs[:, 1])
        
        # Sort by depth
        sort_idx = np.argsort(self.depth_dip_pairs[:, 0])
        self.depth_dip_pairs = self.depth_dip_pairs[sort_idx]
        
        # Create interpolator based on method
        self._create_interpolator(interpolation_method)
    
    def _create_interpolator(self, method: str):
        """Create the appropriate interpolator based on method."""
        depths = self.depth_dip_pairs[:, 0]
        dips = self.depth_dip_pairs[:, 1]
        
        if method in ('step', 'previous'):
            # Step function: dip at depth z equals dip at the largest depth_i <= z
            # For extrapolation beyond boundaries, use linear extrapolation
            self._step_depths = depths
            self._step_dips = dips
            
            # Create linear interpolator for extrapolation only
            self._linear_extrap = interp1d(
                depths, dips,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            
            # Use custom interpolator
            self.interpolator = self._step_interpolate
        else:
            # Standard scipy interpolation
            self.interpolator = interp1d(
                depths, dips,
                kind=method,
                fill_value='extrapolate',
                bounds_error=False
            )
    
    def _step_interpolate(self, depth: float) -> float:
        """
        Step interpolation: returns dip at the previous (shallower) depth point.
        
        For depth z:
        - If z < min_depth: linear extrapolation
        - If z > max_depth: linear extrapolation  
        - Otherwise: dip = dip[i] where depth[i] is largest depth <= z
        """
        depth = float(depth)
        min_depth = self._step_depths[0]
        max_depth = self._step_depths[-1]
        
        if depth < min_depth or depth > max_depth:
            # Extrapolation: use linear
            return float(self._linear_extrap(depth))
        
        # Find the largest depth_i <= depth
        idx = np.searchsorted(self._step_depths, depth, side='right') - 1
        idx = max(0, min(idx, len(self._step_dips) - 1))
        
        return float(self._step_dips[idx])
    
    def get_dip(self, depth: float, output_range: str = 'neg90_90') -> float:
        """
        Get dip angle at a specific depth.
        
        Parameters:
        -----------
        depth : float
            Depth in km.
        output_range : str
            Output range: 'neg90_90' (default) or '0_180'.
            
        Returns:
        --------
        float
            Dip angle in specified range.
        """
        dip_0_180 = float(self.interpolator(depth))
        
        if output_range == 'neg90_90':
            return normalize_dip_to_neg90_90(dip_0_180)
        return dip_0_180
    
    def get_dip_0_180(self, depth: float) -> float:
        """Get dip angle in [0, 180] range (for internal calculations)."""
        return float(self.interpolator(depth))
    
    def get_depth_points(self) -> np.ndarray:
        """Get the depth points defined in this profile."""
        return self.depth_dip_pairs[:, 0].copy()
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.
        Note: Exports dip in [-90, 90] range for user readability.
        """
        # Convert back to [-90, 90] for export
        export_pairs = self.depth_dip_pairs.copy()
        export_pairs[:, 1] = normalize_dip_to_neg90_90(export_pairs[:, 1])
        
        return {
            'x': self.x,
            'y': self.y,
            'lon': self.lon,
            'lat': self.lat,
            'depth_dip_pairs': export_pairs.tolist(),
            'interpolation_method': self.interpolation_method
        }
    
    @classmethod
    def from_dict(cls, data: dict, interpolation_method: str = None) -> 'DepthDipProfile':
        """Create from dictionary (assumes dip in [-90, 90] range)."""
        # Use stored method if available, otherwise use provided or default
        method = interpolation_method or data.get('interpolation_method', 'linear')
        return cls(
            x=data['x'],
            y=data['y'],
            lon=data['lon'],
            lat=data['lat'],
            depth_dip_pairs=np.array(data['depth_dip_pairs']),
            interpolation_method=method,
            input_dip_range='neg90_90'
        )


class LayeredDipInterpolator:
    """
    Interpolator for depth-varying dip angles across a fault surface.
    
    This class handles:
    - Multiple reference nodes with depth-dip profiles
    - Spatial interpolation between nodes (operates in 0-180 dip space)
    - Buffer zone handling for segmented faults
    
    Attributes:
    -----------
    profiles : List[DepthDipProfile]
        List of depth-dip profiles at reference nodes.
    interpolation_axis : str
        Axis for spatial interpolation ('x' or 'y').
    depth_function : callable, optional
        Function to compute dip from base_dip and depth.
    """
    
    def __init__(self, interpolation_axis: str = 'auto'):
        """
        Initialize the interpolator.
        
        Parameters:
        -----------
        interpolation_axis : str
            Axis for spatial interpolation ('auto', 'x', or 'y').
        """
        self.profiles: List[DepthDipProfile] = []
        self.interpolation_axis = interpolation_axis
        self.depth_function: Optional[Callable] = None
        self._spatial_interpolators: Dict[float, Callable] = {}
    
    def clear_profiles(self):
        """Clear all profiles and cached interpolators."""
        self.profiles.clear()
        self._spatial_interpolators.clear()
        self.depth_function = None
    
    def add_profile(self, profile: DepthDipProfile):
        """Add a depth-dip profile."""
        self.profiles.append(profile)
        # Clear cached interpolators
        self._spatial_interpolators.clear()
    
    def set_depth_function(self, func: Callable[[float, float], float]):
        """
        Set a function to compute dip from base_dip and depth.
        
        Parameters:
        -----------
        func : callable
            Function with signature func(base_dip, depth) -> dip.
            Note: Input and output should be in [-90, 90] range.
            Example: lambda dip, depth: dip + 0.5 * depth
        """
        self.depth_function = func
        # Clear cached interpolators when function changes
        self._spatial_interpolators.clear()
    
    def get_all_depth_points(self) -> np.ndarray:
        """
        Get all unique depth points from all profiles.
        
        Returns:
        --------
        np.ndarray
            Sorted array of unique depth values.
        """
        if len(self.profiles) == 0:
            return np.array([])
        
        all_depths = []
        for p in self.profiles:
            all_depths.extend(p.get_depth_points())
        
        return np.unique(all_depths)
    
    def _get_spatial_interpolator_0_180(self, depth: float) -> Callable:
        """
        Get or create a spatial interpolator for a specific depth.
        Returns dip in [0, 180] range for internal use.
        """
        cache_key = ('0_180', depth)
        
        if cache_key not in self._spatial_interpolators:
            if len(self.profiles) == 0:
                raise ValueError("No profiles available for interpolation")
            
            if len(self.profiles) == 1:
                # Single profile - no spatial interpolation needed
                if self.depth_function is not None:
                    base_dip = self.profiles[0].get_dip(self.profiles[0].depth_dip_pairs[0, 0])
                    dip = self.depth_function(base_dip, depth)
                    dip_0_180 = normalize_dip_to_0_180(dip)
                else:
                    dip_0_180 = self.profiles[0].get_dip_0_180(depth)
                self._spatial_interpolators[cache_key] = lambda x, y, d=dip_0_180: d
            else:
                # Multiple profiles - create spatial interpolator in 0-180 space
                axis_values = []
                dip_values_0_180 = []
                
                for p in self.profiles:
                    axis_val = p.x if self.interpolation_axis == 'x' else p.y
                    axis_values.append(axis_val)
                    
                    if self.depth_function is not None:
                        base_dip = p.get_dip(p.depth_dip_pairs[0, 0])
                        dip = self.depth_function(base_dip, depth)
                        dip_values_0_180.append(normalize_dip_to_0_180(dip))
                    else:
                        dip_values_0_180.append(p.get_dip_0_180(depth))
                
                # Sort by axis value
                sort_idx = np.argsort(axis_values)
                axis_values = np.array(axis_values)[sort_idx]
                dip_values_0_180 = np.array(dip_values_0_180)[sort_idx]
                
                interp_func = interp1d(
                    axis_values, dip_values_0_180,
                    kind='linear',
                    fill_value=(dip_values_0_180[0], dip_values_0_180[-1]),
                    bounds_error=False
                )
                
                if self.interpolation_axis == 'x':
                    self._spatial_interpolators[cache_key] = lambda x, y, f=interp_func: float(f(x))
                else:
                    self._spatial_interpolators[cache_key] = lambda x, y, f=interp_func: float(f(y))
        
        return self._spatial_interpolators[cache_key]
    
    def get_dip(self, x: float, y: float, depth: float, output_range: str = 'neg90_90') -> float:
        """
        Get dip angle at a specific location and depth.
        
        Parameters:
        -----------
        x, y : float
            UTM coordinates.
        depth : float
            Depth in km.
        output_range : str
            Output range: 'neg90_90' (default) or '0_180'.
            
        Returns:
        --------
        float
            Dip angle in specified range.
        """
        spatial_interp = self._get_spatial_interpolator_0_180(depth)
        dip_0_180 = spatial_interp(x, y)
        
        if output_range == 'neg90_90':
            return normalize_dip_to_neg90_90(dip_0_180)
        return dip_0_180
    
    def get_dip_array(self, x: np.ndarray, y: np.ndarray, depth: float, 
                      output_range: str = 'neg90_90') -> np.ndarray:
        """
        Get dip angles for arrays of coordinates at a specific depth.
        
        Parameters:
        -----------
        x, y : np.ndarray
            UTM coordinates.
        depth : float
            Depth in km.
        output_range : str
            Output range: 'neg90_90' (default) or '0_180'.
            
        Returns:
        --------
        np.ndarray
            Dip angles in specified range.
        """
        return np.array([self.get_dip(xi, yi, depth, output_range) for xi, yi in zip(x, y)])
    
    def sample_at_depth(self, depth: float, x_coords: np.ndarray = None, 
                        y_coords: np.ndarray = None) -> pd.DataFrame:
        """
        Sample dip values at a specific depth along the fault trace.
        
        This is useful for generating buffer-node-style data at each depth layer.
        
        Parameters:
        -----------
        depth : float
            Depth to sample at.
        x_coords, y_coords : np.ndarray, optional
            Coordinates to sample at. If None, uses profile locations.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns [x, y, lon, lat, dip].
        """
        if x_coords is None:
            # Use profile locations
            records = []
            for p in self.profiles:
                records.append({
                    'x': p.x, 'y': p.y,
                    'lon': p.lon, 'lat': p.lat,
                    'dip': self.get_dip(p.x, p.y, depth)
                })
            return pd.DataFrame(records)
        else:
            # Sample at provided coordinates
            dips = self.get_dip_array(x_coords, y_coords, depth)
            return pd.DataFrame({
                'x': x_coords, 'y': y_coords,
                'dip': dips
            })
    
    def to_dataframe(self, depths: np.ndarray = None) -> pd.DataFrame:
        """
        Export all profiles to a DataFrame.
        
        Parameters:
        -----------
        depths : np.ndarray, optional
            Depths to sample. If None, uses depths from profiles.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns [x, y, lon, lat, depth, dip].
        """
        records = []
        for p in self.profiles:
            if depths is None:
                sample_depths = p.depth_dip_pairs[:, 0]
            else:
                sample_depths = depths
            
            for d in sample_depths:
                records.append({
                    'x': p.x, 'y': p.y,
                    'lon': p.lon, 'lat': p.lat,
                    'depth': d, 
                    'dip': p.get_dip(d)  # Returns in [-90, 90]
                })
        
        return pd.DataFrame(records)


class AdaptiveLayeredDipTriangularPatches(AdaptiveTriangularPatches):
    """
    A fault class that supports depth-varying dip angles.
    
    This class extends AdaptiveTriangularPatches to provide:
    - Depth-dependent dip angle interpolation
    - Multi-layer geometry generation
    - Support for Slab2.0 style isodepth curves
    - Buffer zone handling for segmented faults
    
    Attributes:
    -----------
    dip_interpolator : LayeredDipInterpolator
        Interpolator for depth-varying dip angles.
    layer_coords : List[np.ndarray]
        List of layer coordinates.
    layer_depths : np.ndarray
        Array of layer depths.
    """
    
    def __init__(self, name: str, utmzone=None, ellps='WGS84', 
                 lon0=None, lat0=None, verbose=True):
        """Initialize LayeredDipFault."""
        super().__init__(name, utmzone=utmzone, ellps=ellps, 
                        lon0=lon0, lat0=lat0, verbose=verbose)
        
        self.dip_interpolator = LayeredDipInterpolator()
        self.layer_coords: List[np.ndarray] = []
        self.layer_coords_ll: List[np.ndarray] = []
        self.layer_depths: np.ndarray = np.array([])
    
    # ==================== Input Methods ====================
    
    def set_depth_dip_from_function(
            self,
            reference_nodes: np.ndarray,
            depth_function: Callable[[float, float], float],
            depth_range: Tuple[float, float] = None,
            num_depth_samples: int = 10,
            is_utm: bool = False,
            buffer_nodes: np.ndarray = None,
            buffer_radius: float = None
        ):
        """
        Set depth-dip relationship using a function.
        
        Parameters:
        -----------
        reference_nodes : np.ndarray
            Reference nodes with shape (n, 3): [lon/x, lat/y, base_dip].
            base_dip should be in [-90, 90] range.
        depth_function : callable
            Function: depth_function(base_dip, depth) -> dip.
            Both input and output dip should be in [-90, 90] range.
            Example: lambda dip, depth: dip + 0.5 * depth
        depth_range : tuple, optional
            (min_depth, max_depth). If None, uses (self.top, self.depth).
        num_depth_samples : int
            Number of depth samples for storing the profile.
        is_utm : bool
            If True, input coordinates are in UTM.
        buffer_nodes : np.ndarray, optional
            Buffer nodes for segmented faults. Shape (n, 2) with [lon, lat].
        buffer_radius : float, optional
            Buffer radius in km.
            
        Example:
        --------
        >>> fault.set_depth_dip_from_function(
        ...     reference_nodes=np.array([[lon0, lat0, 30.0]]),
        ...     depth_function=lambda dip, depth: dip + 1.0 * depth
        ... )
        """
        if depth_range is None:
            depth_range = (self.top, self.depth)
        
        depths = np.linspace(depth_range[0], depth_range[1], num_depth_samples)
        reference_nodes = np.atleast_2d(reference_nodes)
        
        # Reset interpolator
        self.dip_interpolator.clear_profiles()
        self.dip_interpolator.set_depth_function(depth_function)
        
        # Process reference nodes
        processed_nodes = self._process_reference_nodes(
            reference_nodes, is_utm, buffer_nodes, buffer_radius)
        
        for node in processed_nodes:
            x, y, lon, lat, base_dip = node['x'], node['y'], node['lon'], node['lat'], node['dip']
            
            # Generate depth-dip pairs using the function
            depth_dip_pairs = np.column_stack([
                depths,
                [depth_function(base_dip, d) for d in depths]
            ])
            
            profile = DepthDipProfile(x, y, lon, lat, depth_dip_pairs, 
                                      input_dip_range='neg90_90')
            self.dip_interpolator.add_profile(profile)
    
    def set_depth_dip_from_profiles(
            self,
            profiles_data: Union[Dict, pd.DataFrame, np.ndarray],
            is_utm: bool = False,
            interpolation_method: str = 'linear',
            buffer_nodes: np.ndarray = None,
            buffer_radius: float = None
        ):
        """
        Set depth-dip relationship from discrete profiles.
        
        Parameters:
        -----------
        profiles_data : dict, DataFrame, or ndarray
            Input data in one of these formats:
            
            1. Dictionary format:
               {
                   'reference_nodes': np.array([[lon1, lat1], [lon2, lat2], ...]),
                   'depth_dip_profiles': [
                       np.array([[depth1, dip1], [depth2, dip2], ...]),  # Node 1
                       np.array([[depth1, dip1], [depth2, dip2], ...]),  # Node 2
                   ]
               }
               Note: dip values should be in [-90, 90] range.
            
            2. DataFrame format:
               Columns: [lon, lat, depth, dip] or [x, y, depth, dip]
               Multiple rows per location with different depths.
               Note: dip values should be in [-90, 90] range.
            
            3. Array format:
               Shape (n, 4) with columns [lon/x, lat/y, depth, dip]
               Note: dip values should be in [-90, 90] range.
               
        is_utm : bool
            If True, coordinates are in UTM.
        interpolation_method : str
            Interpolation method for depth-dip relationship:
            - 'linear': Linear interpolation between depth points
            - 'cubic': Cubic spline interpolation
            - 'step' or 'previous': Step function (hard transitions at depth boundaries)
        buffer_nodes : np.ndarray, optional
            Buffer nodes for additional reference points.
        buffer_radius : float, optional
            Buffer radius in km.
        """
        self.dip_interpolator.clear_profiles()
        
        if isinstance(profiles_data, dict):
            self._parse_dict_profiles(profiles_data, is_utm, interpolation_method)
        elif isinstance(profiles_data, pd.DataFrame):
            self._parse_dataframe_profiles(profiles_data, is_utm, interpolation_method)
        elif isinstance(profiles_data, np.ndarray):
            self._parse_array_profiles(profiles_data, is_utm, interpolation_method)
        else:
            raise ValueError(f"Unsupported profiles_data type: {type(profiles_data)}")
        
        # Handle buffer nodes if provided
        if buffer_nodes is not None and buffer_radius is not None:
            self._add_buffer_profiles(buffer_nodes, buffer_radius, interpolation_method)
    
    def _process_reference_nodes(
            self,
            reference_nodes: np.ndarray,
            is_utm: bool,
            buffer_nodes: np.ndarray = None,
            buffer_radius: float = None
        ) -> List[Dict]:
        """
        Process reference nodes and optionally add buffer nodes.
        
        Returns list of dicts with keys: x, y, lon, lat, dip
        """
        processed = []
        
        for node in reference_nodes:
            if is_utm:
                x, y = node[0], node[1]
                lon, lat = self.xy2ll(x, y)
            else:
                lon, lat = node[0], node[1]
                x, y = self.ll2xy(lon, lat)
            
            base_dip = node[2] if len(node) > 2 else None
            processed.append({'x': x, 'y': y, 'lon': lon, 'lat': lat, 'dip': base_dip})
        
        # Handle buffer nodes using parent class method
        if buffer_nodes is not None and buffer_radius is not None:
            # Create xydip DataFrame from processed nodes
            xydip = pd.DataFrame(processed)
            
            # Use parent's handle_buffer_nodes
            interpolation_axis = self.dip_interpolator.interpolation_axis
            if interpolation_axis == 'auto':
                interpolation_axis = self._determine_optimal_interpolation_axis(
                    self.top_coords[:, 0], self.top_coords[:, 1])
                self.dip_interpolator.interpolation_axis = interpolation_axis
            
            xydip_expanded = self.handle_buffer_nodes(
                xydip, buffer_nodes, buffer_radius, 
                interpolation_axis, update_ref=False)
            
            # Convert back to list of dicts
            processed = xydip_expanded.to_dict('records')
        
        return processed
    
    def _parse_dict_profiles(self, data: dict, is_utm: bool, interp_method: str):
        """Parse dictionary format profiles."""
        ref_nodes = np.atleast_2d(data['reference_nodes'])
        profiles = data['depth_dip_profiles']
        
        for node, depth_dip in zip(ref_nodes, profiles):
            if is_utm:
                x, y = node[0], node[1]
                lon, lat = self.xy2ll(x, y)
            else:
                lon, lat = node[0], node[1]
                x, y = self.ll2xy(lon, lat)
            
            profile = DepthDipProfile(x, y, lon, lat, np.array(depth_dip), 
                                      interp_method, input_dip_range='neg90_90')
            self.dip_interpolator.add_profile(profile)
    
    def _parse_dataframe_profiles(self, df: pd.DataFrame, is_utm: bool, interp_method: str):
        """Parse DataFrame format profiles."""
        coord_cols = ['x', 'y'] if is_utm else ['lon', 'lat']
        
        # Group by location
        grouped = df.groupby(coord_cols)
        
        for coords, group in grouped:
            if is_utm:
                x, y = coords
                lon, lat = self.xy2ll(x, y)
            else:
                lon, lat = coords
                x, y = self.ll2xy(lon, lat)
            
            depth_dip = group[['depth', 'dip']].sort_values('depth').values
            profile = DepthDipProfile(x, y, lon, lat, depth_dip, 
                                      interp_method, input_dip_range='neg90_90')
            self.dip_interpolator.add_profile(profile)
    
    def _parse_array_profiles(self, arr: np.ndarray, is_utm: bool, interp_method: str):
        """Parse array format profiles."""
        if arr.shape[1] != 4:
            raise ValueError("Array must have 4 columns: [lon/x, lat/y, depth, dip]")
        
        # Convert to DataFrame and use DataFrame parser
        coord_cols = ['x', 'y'] if is_utm else ['lon', 'lat']
        df = pd.DataFrame(arr, columns=coord_cols + ['depth', 'dip'])
        self._parse_dataframe_profiles(df, is_utm, interp_method)
    
    def _add_buffer_profiles(self, buffer_nodes: np.ndarray, buffer_radius: float,
                             interpolation_method: str = 'linear'):
        """
        Add profiles at buffer node locations by interpolating from existing profiles.
        """
        if len(self.dip_interpolator.profiles) == 0:
            raise ValueError("Must have existing profiles before adding buffer profiles")
        
        # Get depths from existing profiles
        depths = self.dip_interpolator.profiles[0].depth_dip_pairs[:, 0]
        
        # Create xydip at each depth for buffer node handling
        for depth in depths:
            # Get dips at this depth for existing profiles
            xydip = self.dip_interpolator.sample_at_depth(depth)
            
            # Apply buffer node handling
            interpolation_axis = self.dip_interpolator.interpolation_axis
            xydip_expanded = self.handle_buffer_nodes(
                xydip, buffer_nodes, buffer_radius,
                interpolation_axis, update_ref=False)
            
            # Find new nodes (buffer nodes)
            existing_locs = set(zip(xydip['x'], xydip['y']))
            for _, row in xydip_expanded.iterrows():
                if (row['x'], row['y']) not in existing_locs:
                    # This is a buffer node - need to create profile if not exists
                    # Check if profile already exists at this location
                    profile_exists = any(
                        np.isclose(p.x, row['x']) and np.isclose(p.y, row['y'])
                        for p in self.dip_interpolator.profiles
                    )
                    if not profile_exists:
                        # Create new profile by sampling at all depths
                        depth_dip_pairs = []
                        for d in depths:
                            dip = self.dip_interpolator.get_dip(row['x'], row['y'], d)
                            depth_dip_pairs.append([d, dip])
                        
                        lon, lat = row.get('lon', None), row.get('lat', None)
                        if lon is None or lat is None:
                            lon, lat = self.xy2ll(row['x'], row['y'])
                        
                        profile = DepthDipProfile(
                            row['x'], row['y'], lon, lat,
                            np.array(depth_dip_pairs),
                            interpolation_method=interpolation_method,
                            input_dip_range='neg90_90'
                        )
                        self.dip_interpolator.add_profile(profile)
    
    def set_depth_dip_from_constant(
            self,
            dip_angle: float,
            reference_nodes: np.ndarray = None,
            is_utm: bool = False,
            buffer_nodes: np.ndarray = None,
            buffer_radius: float = None
        ):
        """
        Set constant dip angle (no depth variation).
        
        Parameters:
        -----------
        dip_angle : float
            Constant dip angle in degrees (in [-90, 90] range).
        reference_nodes : np.ndarray, optional
            Reference nodes. If None, uses top_coords centroid.
        is_utm : bool
            If True, coordinates are in UTM.
        buffer_nodes : np.ndarray, optional
            Buffer nodes for segmented faults.
        buffer_radius : float, optional
            Buffer radius in km.
        """
        if reference_nodes is None:
            # Use centroid of top_coords
            x_center = np.mean(self.top_coords[:, 0])
            y_center = np.mean(self.top_coords[:, 1])
            lon_center, lat_center = self.xy2ll(x_center, y_center)
            reference_nodes = np.array([[lon_center, lat_center]])
        
        # Create constant dip profiles
        reference_nodes = np.atleast_2d(reference_nodes)
        depth_dip_data = {
            'reference_nodes': reference_nodes,
            'depth_dip_profiles': [
                np.array([[self.top, dip_angle], [self.depth, dip_angle]])
                for _ in reference_nodes
            ]
        }
        
        self.set_depth_dip_from_profiles(depth_dip_data, is_utm, 
                                         buffer_nodes=buffer_nodes,
                                         buffer_radius=buffer_radius)

    # ==================== Interpolation Setup ====================
    
    def setup_interpolation(
            self,
            discretization_interval: float = None,
            interpolation_axis: str = 'auto',
            calculate_strike_along_trace: bool = True
        ) -> str:
        """
        Setup spatial interpolation parameters.
        
        This method:
        1. Discretizes the trace if needed
        2. Sets up the interpolation axis
        3. Calculates strike along trace
        
        Parameters:
        -----------
        discretization_interval : float, optional
            Interval for discretizing the trace.
        interpolation_axis : str
            Axis for interpolation ('auto', 'x', or 'y').
        calculate_strike_along_trace : bool
            If True, calculate strike along trace.
        
        Returns:
        --------
        str
            Selected interpolation axis.
        """
        # Discretize trace if needed
        if discretization_interval is not None:
            self.discretize_top_coords(every=discretization_interval)
            self.xi, self.yi = self.top_coords[:, 0], self.top_coords[:, 1]
            self.loni, self.lati = self.xy2ll(self.xi, self.yi)
        
        # Auto-select interpolation axis
        if interpolation_axis == 'auto':
            interpolation_axis = self._determine_optimal_interpolation_axis(
                self.top_coords[:, 0], self.top_coords[:, 1])
            if self.verbose:
                print(f"Auto-selected interpolation axis: {interpolation_axis}")
        
        self.dip_interpolator.interpolation_axis = interpolation_axis
        
        # Calculate strike if not already done
        if not hasattr(self, 'strikei') or self.strikei is None:
            self.calculate_trace_strike(
                use_discretized_trace=True,
                calculate_strike_along_trace=calculate_strike_along_trace)
        
        self.top_strike = self.calculate_top_strike(discretized=True)
        
        return interpolation_axis

    # ==================== Layer Generation ====================
    
    def generate_layer_depths(
            self,
            num_layers: int = None,
            layer_depths: np.ndarray = None,
            use_profile_depths: bool = False,
            include_endpoints: bool = True
        ) -> np.ndarray:
        """
        Generate depth values for intermediate layers.
        
        Parameters:
        -----------
        num_layers : int, optional
            Number of intermediate layers (excluding top and bottom).
        layer_depths : np.ndarray, optional
            Explicit depth values. Overrides num_layers if provided.
        use_profile_depths : bool
            If True, use depth points from existing profiles.
            This is useful when profiles define specific depth layers.
            Overrides num_layers if True.
        include_endpoints : bool
            If True, include top and bottom depths in returned array.
            
        Returns:
        --------
        np.ndarray
            Array of layer depths.
            
        Examples:
        ---------
        >>> # Use profile depths (recommended for discrete profiles)
        >>> depths = fault.generate_layer_depths(use_profile_depths=True)
        
        >>> # Use explicit depths
        >>> depths = fault.generate_layer_depths(layer_depths=np.array([5, 10, 15, 20]))
        
        >>> # Use evenly spaced depths
        >>> depths = fault.generate_layer_depths(num_layers=4)
        """
        if layer_depths is not None:
            self.layer_depths = np.array(layer_depths)
        elif use_profile_depths:
            # Get all depth points from profiles
            all_depths = self.dip_interpolator.get_all_depth_points()
            if len(all_depths) == 0:
                raise ValueError("No profiles available. Set profiles first.")
            # Exclude top and bottom if they exist
            intermediate = all_depths[(all_depths > self.top) & (all_depths < self.depth)]
            self.layer_depths = intermediate
            if self.verbose and len(intermediate) > 0:
                print(f"Using {len(intermediate)} intermediate depths from profiles: {intermediate}")
        elif num_layers is not None:
            # Generate evenly spaced depths
            self.layer_depths = np.linspace(self.top, self.depth, num_layers + 2)[1:-1]
        else:
            raise ValueError("Either num_layers, layer_depths, or use_profile_depths=True must be provided")
        
        if include_endpoints:
            all_depths = np.concatenate([[self.top], self.layer_depths, [self.depth]])
            return np.unique(all_depths)
        
        return self.layer_depths
    
    def generate_layer_coords(
            self,
            layer_depths: np.ndarray = None,
            num_layers: int = None,
            use_profile_depths: bool = False,
            use_average_strike: bool = False,
            average_strike_source: str = 'pca',
            user_direction_angle: float = None,
            integration_steps: int = 10,
            verbose: bool = False
        ) -> List[np.ndarray]:
        """
        Generate coordinates for all layers using depth-varying dips.
        
        Parameters:
        -----------
        layer_depths : np.ndarray, optional
            Depths for intermediate layers.
        num_layers : int, optional
            Number of intermediate layers.
        use_profile_depths : bool
            If True, automatically use depth points from existing profiles.
            This is recommended when using set_depth_dip_from_profiles().
        use_average_strike : bool
            If True, use average strike direction.
        average_strike_source : str
            Source for average strike ('pca' or 'user').
        user_direction_angle : float, optional
            User-specified direction angle.
        integration_steps : int
            Number of steps for integrating along dip direction.
        verbose : bool
            If True, print debug information.
            
        Returns:
        --------
        List[np.ndarray]
            List of layer coordinates (excluding top and bottom).
            
        Examples:
        ---------
        >>> # Automatically use depths from profile definitions
        >>> fault.set_depth_dip_from_profiles(profiles_data, interpolation_method='step')
        >>> coords = fault.generate_layer_coords(use_profile_depths=True)
        
        >>> # Or specify number of layers
        >>> coords = fault.generate_layer_coords(num_layers=5)
        """
        # Generate layer depths
        if layer_depths is None and num_layers is None and not use_profile_depths:
            raise ValueError("Either layer_depths, num_layers, or use_profile_depths=True must be provided")
        
        depths = self.generate_layer_depths(
            num_layers=num_layers, 
            layer_depths=layer_depths, 
            use_profile_depths=use_profile_depths,
            include_endpoints=False
        )
        
        # Get strike direction
        strike_rad = self._compute_strike_rad(
            use_average_strike, average_strike_source, user_direction_angle, verbose)
        
        # Initialize with top coords
        x, y = self.top_coords[:, 0], self.top_coords[:, 1]
        current_z = np.ones_like(x) * self.top
        
        self.layer_coords = []
        self.layer_coords_ll = []
        
        # All depths including top for iteration
        all_depths = np.concatenate([[self.top], depths])
        
        for i, target_depth in enumerate(depths):
            # Generate coords for this layer
            layer_xyz = self._propagate_to_depth(
                x, y, current_z, target_depth, strike_rad, integration_steps)
            
            self.layer_coords.append(layer_xyz)
            
            # Convert to lon/lat
            lon, lat = self.xy2ll(layer_xyz[:, 0], layer_xyz[:, 1])
            layer_ll = np.column_stack([lon, lat, layer_xyz[:, 2]])
            self.layer_coords_ll.append(layer_ll)
            
            # Update current position for next iteration
            x, y = layer_xyz[:, 0], layer_xyz[:, 1]
            current_z = layer_xyz[:, 2]
        
        return self.layer_coords
    
    def _compute_strike_rad(
            self,
            use_average_strike: bool,
            average_strike_source: str,
            user_direction_angle: float,
            verbose: bool
        ) -> np.ndarray:
        """Compute strike angles in radians."""
        from numpy import deg2rad
        
        x, y = self.top_coords[:, 0], self.top_coords[:, 1]
        strike_direction = np.array([x[-1] - x[0], y[-1] - y[0]])
        
        if use_average_strike:
            if average_strike_source == 'pca':
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                pca.fit(np.vstack((x, y)).T)
                principal_direction = pca.components_[0]
                average_strike_rad = np.arctan2(principal_direction[1], principal_direction[0])
                average_strike_rad = np.pi/2.0 - average_strike_rad
                if np.dot(principal_direction, strike_direction) < 0:
                    average_strike_rad += np.pi
            elif average_strike_source == 'user' and user_direction_angle is not None:
                average_strike_rad = deg2rad(user_direction_angle)
            else:
                raise ValueError("Invalid average_strike_source or user_direction_angle not provided.")
            
            if verbose:
                print(f"Average strike direction: {np.rad2deg(average_strike_rad):.2f}")
            
            return np.full(len(x), average_strike_rad)
        else:
            return deg2rad(self.top_strike)
    
    def _propagate_to_depth(
            self,
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray,
            target_depth: float,
            strike_rad: np.ndarray,
            integration_steps: int
        ) -> np.ndarray:
        """
        Propagate coordinates from current depth to target depth using depth-varying dips.
        
        Uses numerical integration along the dip direction.
        Operates in [0, 180] dip space for correct geometry computation.
        """
        from numpy import deg2rad, sin, cos
        
        # Integration along depth
        depth_steps = np.linspace(z[0], target_depth, integration_steps + 1)
        
        current_x = x.copy()
        current_y = y.copy()
        current_z = z.copy()
        
        for i in range(len(depth_steps) - 1):
            d1, d2 = depth_steps[i], depth_steps[i + 1]
            dz = d2 - d1
            
            # Get dip at midpoint depth (in [-90, 90] for geometry)
            mid_depth = (d1 + d2) / 2
            dips = self.dip_interpolator.get_dip_array(current_x, current_y, mid_depth, 
                                                        output_range='neg90_90')
            dip_rad = deg2rad(dips)
            
            # Adjust for negative dips (dipping in opposite direction)
            negative_mask = dip_rad < 0
            adjusted_strike = strike_rad.copy()
            adjusted_strike[negative_mask] += np.pi
            dip_rad = np.abs(dip_rad)
            
            # Compute displacement
            # Handle near-zero dips (horizontal layers)
            dip_rad = np.maximum(dip_rad, 1e-6)  # Avoid division by zero
            
            width = dz / np.sin(dip_rad)
            dx = width * np.cos(dip_rad) * np.cos(-adjusted_strike)
            dy = width * np.cos(dip_rad) * np.sin(-adjusted_strike)
            
            current_x += dx
            current_y += dy
            current_z = np.full_like(current_x, d2)
        
        return np.column_stack([current_x, current_y, current_z])
    
    def generate_bottom_coords(
            self,
            use_average_strike: bool = False,
            average_strike_source: str = 'pca',
            user_direction_angle: float = None,
            integration_steps: int = 10,
            update_self: bool = True,
            verbose: bool = False
        ) -> np.ndarray:
        """
        Generate bottom coordinates using depth-varying dips.
        
        Parameters:
        -----------
        use_average_strike : bool
            If True, use average strike direction.
        average_strike_source : str
            Source for average strike ('pca' or 'user').
        user_direction_angle : float, optional
            User-specified direction angle.
        integration_steps : int
            Number of steps for integrating along dip direction.
        update_self : bool
            If True, update self.bottom_coords.
        verbose : bool
            If True, print debug information.
            
        Returns:
        --------
        np.ndarray
            Bottom coordinates.
        """
        # Get strike direction
        strike_rad = self._compute_strike_rad(
            use_average_strike, average_strike_source, user_direction_angle, verbose)
        
        # Propagate from top to bottom
        x, y = self.top_coords[:, 0], self.top_coords[:, 1]
        z = np.ones_like(x) * self.top
        
        bottom_xyz = self._propagate_to_depth(
            x, y, z, self.depth, strike_rad, integration_steps)
        
        # Sort by interpolation axis
        axis_idx = 0 if self.dip_interpolator.interpolation_axis == 'x' else 1
        sort_order = np.argsort(bottom_xyz[:, axis_idx])
        bottom_xyz = bottom_xyz[sort_order]
        
        # Ensure consistent direction with top
        strike_direction = np.array([
            self.top_coords[-1, 0] - self.top_coords[0, 0],
            self.top_coords[-1, 1] - self.top_coords[0, 1]
        ])
        bottom_direction = np.array([
            bottom_xyz[-1, 0] - bottom_xyz[0, 0],
            bottom_xyz[-1, 1] - bottom_xyz[0, 1]
        ])
        if np.dot(strike_direction, bottom_direction) < 0:
            bottom_xyz = bottom_xyz[::-1]
        
        if update_self:
            self.set_bottom_coords(bottom_xyz, lonlat=False)
        
        return bottom_xyz
    
    def _smooth_coords(
            self,
            coords: np.ndarray,
            smooth_window: int = 5,
            smooth_method: str = 'savgol',
            polyorder: int = 2
        ) -> np.ndarray:
        """
        Apply smoothing to coordinates to remove small fluctuations.
        
        Parameters:
        -----------
        coords : np.ndarray
            Coordinates array, shape (n, 2) or (n, 3).
        smooth_window : int
            Smoothing window size (must be odd, will be adjusted if even).
        smooth_method : str
            Smoothing method:
            - 'savgol': Savitzky-Golay filter (preserves shape, recommended)
            - 'moving_average': Simple moving average
            - 'gaussian': Gaussian filter
        polyorder : int
            Polynomial order for Savitzky-Golay filter (must be < smooth_window).
            
        Returns:
        --------
        np.ndarray
            Smoothed coordinates (same shape as input).
        """
        if len(coords) < smooth_window:
            return coords
        
        # Ensure window is odd
        if smooth_window % 2 == 0:
            smooth_window += 1
        
        smoothed = coords.copy()
        
        # Determine number of spatial dimensions to smooth (exclude depth if present)
        n_dims = min(coords.shape[1], 2)  # Only smooth x, y
        
        for dim in range(n_dims):
            arr = coords[:, dim]
            
            if smooth_method == 'savgol':
                from scipy.signal import savgol_filter
                order = min(polyorder, smooth_window - 1)
                smoothed[:, dim] = savgol_filter(arr, smooth_window, order, mode='interp')
                
            elif smooth_method == 'moving_average':
                kernel = np.ones(smooth_window) / smooth_window
                smoothed_arr = np.convolve(arr, kernel, mode='same')
                # Fix edge effects
                half_w = smooth_window // 2
                for i in range(half_w):
                    smoothed_arr[i] = np.mean(arr[:i + half_w + 1])
                    smoothed_arr[-(i + 1)] = np.mean(arr[-(i + half_w + 1):])
                smoothed[:, dim] = smoothed_arr
                
            elif smooth_method == 'gaussian':
                from scipy.ndimage import gaussian_filter1d
                sigma = smooth_window / 4.0
                smoothed[:, dim] = gaussian_filter1d(arr, sigma, mode='nearest')
        
        return smoothed
    
    def smooth_layer_coords(
            self,
            smooth_window: int = 5,
            smooth_method: str = 'savgol',
            include_bottom: bool = True
        ):
        """
        Smooth all layer coordinates and optionally bottom coordinates.
        
        Call this before generate_layered_mesh() if you notice jagged edges
        or small fluctuations in the generated layers.
        
        Parameters:
        -----------
        smooth_window : int
            Smoothing window size. Larger = more smoothing.
            Recommended: 5-9 for moderate smoothing, 11+ for heavy smoothing.
        smooth_method : str
            'savgol' (default, preserves shape), 'moving_average', or 'gaussian'.
        include_bottom : bool
            If True, also smooth bottom_coords.
            
        Examples:
        ---------
        >>> fault.generate_layer_coords(num_layers=5)
        >>> fault.generate_bottom_coords()
        >>> fault.smooth_layer_coords(smooth_window=7)  # Apply smoothing
        >>> fault.generate_layered_mesh()
        """
        # Smooth intermediate layers
        for i, layer in enumerate(self.layer_coords):
            self.layer_coords[i] = self._smooth_coords(layer, smooth_window, smooth_method)
            # Update lon/lat
            lon, lat = self.xy2ll(self.layer_coords[i][:, 0], self.layer_coords[i][:, 1])
            self.layer_coords_ll[i] = np.column_stack([lon, lat, self.layer_coords[i][:, 2]])
        
        # Smooth bottom
        if include_bottom and hasattr(self, 'bottom_coords') and self.bottom_coords is not None:
            # bottom_coords is (n, 2), add temporary z column for smoothing
            bottom_with_z = np.column_stack([
                self.bottom_coords[:, 0], 
                self.bottom_coords[:, 1],
                np.full(len(self.bottom_coords), self.depth)
            ])
            smoothed = self._smooth_coords(bottom_with_z, smooth_window, smooth_method)
            self.bottom_coords = smoothed[:, :2]
            
            # Update lon/lat
            lon, lat = self.xy2ll(self.bottom_coords[:, 0], self.bottom_coords[:, 1])
            self.bottom_coords_ll = np.column_stack([lon, lat])
        
        if self.verbose:
            print(f"Smoothed {len(self.layer_coords)} layers with window={smooth_window}, method={smooth_method}")

    # ==================== Mesh Generation ====================
    
    def generate_layered_mesh(
            self,
            num_layers: int = None,
            layer_depths: np.ndarray = None,
            use_profile_depths: bool = False,
            nodes_on_layers: bool = True,
            mesh_func: bool = True,
            field_size_dict: dict = {'min_dx': 3, 'bias': 1.05},
            mesh_algorithm: int = 2,
            optimize_method: str = 'Laplace2D',
            show: bool = True,
            verbose: int = 0,
            out_mesh: str = None,
            write2file: bool = False,
            smooth_layers: bool = False,
            smooth_window: int = 5,
            smooth_method: str = 'savgol',
            **kwargs
        ):
        """
        Generate mesh using depth-varying dip layers.
        
        Parameters:
        -----------
        num_layers : int, optional
            Number of intermediate layers.
        layer_depths : np.ndarray, optional
            Explicit depths for layers.
        use_profile_depths : bool
            If True, use depth points from profiles as layer depths.
        nodes_on_layers : bool
            If True, ensure nodes are placed on layer interfaces.
        mesh_func : bool
            If True, use field-based mesh sizing.
        field_size_dict : dict
            Mesh size field parameters.
        mesh_algorithm : int
            Gmsh algorithm (2: default, 5: Delaunay, 6: Frontal-Delaunay).
        optimize_method : str
            Mesh optimization method.
        show : bool
            If True, show mesh in Gmsh GUI.
        verbose : int
            Gmsh verbosity level.
        out_mesh : str, optional
            Output mesh file path.
        write2file : bool
            If True, write mesh to file.
        smooth_layers : bool
            If True, smooth layer and bottom coordinates before meshing.
            Useful when not using average strike to remove small fluctuations.
        smooth_window : int
            Smoothing window size (only used if smooth_layers=True).
        smooth_method : str
            Smoothing method: 'savgol', 'moving_average', or 'gaussian'.
        **kwargs
            Additional arguments for generate_multilayer_mesh.
            
        Examples:
        ---------
        >>> # With smoothing for variable strike
        >>> fault.generate_layered_mesh(
        ...     num_layers=5,
        ...     smooth_layers=True,
        ...     smooth_window=7
        ... )
        """
        # Generate layer coordinates if not already done
        if len(self.layer_coords) == 0:
            if num_layers is None and layer_depths is None and not use_profile_depths:
                raise ValueError("Either num_layers, layer_depths, or use_profile_depths=True must be provided")
            self.generate_layer_coords(
                layer_depths=layer_depths, 
                num_layers=num_layers,
                use_profile_depths=use_profile_depths
            )
        
        # Generate bottom if not set
        if not hasattr(self, 'bottom_coords') or self.bottom_coords is None:
            self.generate_bottom_coords()
        
        # Apply smoothing if requested
        if smooth_layers:
            self.smooth_layer_coords(
                smooth_window=smooth_window,
                smooth_method=smooth_method,
                include_bottom=True
            )
        
        # Use parent's generate_multilayer_mesh
        if out_mesh is None:
            out_mesh = f'gmsh_layered_fault_mesh_{self.name}.msh'
        
        # Prepare top_coords with depth (shape: n x 3)
        top_coords_3d = np.column_stack([
            self.top_coords[:, 0],
            self.top_coords[:, 1],
            np.full(len(self.top_coords), self.top)
        ])
        
        # Prepare bottom_coords with depth (shape: n x 3)
        bottom_coords_3d = np.column_stack([
            self.bottom_coords[:, 0],
            self.bottom_coords[:, 1],
            np.full(len(self.bottom_coords), self.depth)
        ])
        
        self.mesh_generator.set_coordinates(top_coords_3d, bottom_coords_3d)
        
        # Remove smooth-related kwargs to avoid passing to mesh generator
        smooth_related_keys = ['smooth_window', 'smooth_method', 'smooth_layers', 'smooth_coords']
        for key in smooth_related_keys:
            kwargs.pop(key, None)
        
        vertices, faces = self.mesh_generator.generate_multilayer_gmsh_mesh(
            layers_coords=self.layer_coords,
            mesh_func=mesh_func,
            out_mesh=out_mesh,
            write2file=write2file,
            show=show,
            read_mesh=True,
            field_size_dict=field_size_dict,
            mesh_algorithm=mesh_algorithm,
            optimize_method=optimize_method,
            verbose=verbose,
            nodes_on_layers=nodes_on_layers,
            **kwargs
        )
        self.VertFace2csifault(vertices, faces)
    
    # ==================== Export Methods ====================
    
    def export_dip_structure(self, filepath: str = None) -> dict:
        """
        Export the complete dip structure to a dictionary or file.
        
        Parameters:
        -----------
        filepath : str, optional
            If provided, save to JSON file.
            
        Returns:
        --------
        dict
            Dictionary containing all dip profile information.
            Note: dip values are exported in [-90, 90] range.
        """
        import json
        
        structure = {
            'name': self.name,
            'top': self.top,
            'depth': self.depth,
            'interpolation_axis': self.dip_interpolator.interpolation_axis,
            'profiles': [p.to_dict() for p in self.dip_interpolator.profiles],
            'layer_depths': self.layer_depths.tolist() if len(self.layer_depths) > 0 else []
        }
        
        if filepath is not None:
            with open(filepath, 'w') as f:
                json.dump(structure, f, indent=2)
        
        return structure
    
    def load_dip_structure(self, filepath: str):
        """
        Load dip structure from a JSON file.
        
        Parameters:
        -----------
        filepath : str
            Path to JSON file.
        """
        import json
        
        with open(filepath, 'r') as f:
            structure = json.load(f)
        
        self.top = structure['top']
        self.depth = structure['depth']
        self.layer_depths = np.array(structure['layer_depths'])
        
        self.dip_interpolator = LayeredDipInterpolator(structure['interpolation_axis'])
        for p_dict in structure['profiles']:
            profile = DepthDipProfile.from_dict(p_dict)
            self.dip_interpolator.add_profile(profile)
    
    # ==================== Visualization ====================
    
    def plot_dip_profiles(
            self,
            depths: np.ndarray = None,
            figsize: tuple = (10, 6),
            save_fig: str = None,
            dpi: int = 300,
            show: bool = True
        ):
        """
        Plot depth-dip profiles for all reference nodes.
        
        Parameters:
        -----------
        depths : np.ndarray, optional
            Depths to sample. If None, uses 20 evenly spaced depths.
        figsize : tuple
            Figure size.
        save_fig : str, optional
            Path to save figure.
        dpi : int
            DPI for saved figure.
        show : bool
            If True, display the plot.
        """
        import matplotlib.pyplot as plt
        
        if depths is None:
            depths = np.linspace(self.top, self.depth, 20)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, profile in enumerate(self.dip_interpolator.profiles):
            dips = [profile.get_dip(d) for d in depths]  # Returns in [-90, 90]
            label = f'Node {i+1} ({profile.lon:.3f}, {profile.lat:.3f})'
            ax.plot(dips, depths, label=label, linewidth=2)
        
        ax.set_xlabel('Dip (°)')
        ax.set_ylabel('Depth (km)')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Depth-Dip Profiles')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(save_fig, dpi=dpi)
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, ax
    
    def plot_dip_map(
            self,
            depth: float = None,
            figsize: tuple = (12, 8),
            save_fig: str = None,
            dpi: int = 300,
            show: bool = True
        ):
        """
        Plot a map of dip angles at a specific depth.
        
        Parameters:
        -----------
        depth : float, optional
            Depth to plot. If None, uses top depth.
        figsize : tuple
            Figure size.
        save_fig : str, optional
            Path to save figure.
        dpi : int
            DPI for saved figure.
        show : bool
            If True, display the plot.
        """
        import matplotlib.pyplot as plt
        
        if depth is None:
            depth = self.top
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot fault trace
        ax.plot(self.top_coords[:, 0], self.top_coords[:, 1], 
                'k-', linewidth=2, label='Fault trace')
        
        # Plot reference nodes with dip values
        for profile in self.dip_interpolator.profiles:
            dip = profile.get_dip(depth)
            ax.scatter(profile.x, profile.y, c='red', s=100, zorder=5)
            ax.annotate(f'{dip:.1f}°', (profile.x, profile.y), 
                       textcoords='offset points', xytext=(5, 5))
        
        # Sample dip along trace
        x_trace = self.top_coords[:, 0]
        y_trace = self.top_coords[:, 1]
        dips_trace = self.dip_interpolator.get_dip_array(x_trace, y_trace, depth)
        
        # Color the trace by dip
        scatter = ax.scatter(x_trace, y_trace, c=dips_trace, 
                            cmap='RdYlBu_r', s=20, alpha=0.7)
        plt.colorbar(scatter, label=f'Dip at {depth} km (°)')
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title(f'Dip Distribution at Depth = {depth} km')
        ax.legend()
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(save_fig, dpi=dpi)
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, ax