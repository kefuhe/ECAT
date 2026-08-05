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
from .fault_angle_conventions import (
    normalize_oriented_reference_dip,
    validate_oriented_reference_dip,
)

_DIP_RANGES = frozenset({'neg90_90', '0_180'})
_INTERPOLATION_METHODS = frozenset({
    'linear', 'cubic', 'nearest', 'step', 'previous',
})


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
    >>> normalize_dip_to_0_180(-90)  # -> 90 (vertical; side is immaterial)
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


def validate_dip_angles_for_depth_projection(
        dip: Union[float, np.ndarray],
        name: str = 'dip'
    ) -> np.ndarray:
    """Validate dip values used to project a top edge to a deeper edge.

    Accepted external representations are ``[-90, 0) U (0, 180)`` degrees.
    The recommended representation is signed ``[-90, 0) U (0, 90]``:
    positive values dip to the right of positive strike, while negative values
    dip to the left. Values in ``(90, 180)`` are a compatibility form and are
    equivalent to ``dip - 180``. Zero and 180 degrees are horizontal and
    cannot reach a different target depth, so they are rejected.

    Parameters
    ----------
    dip : float or array-like
        Dip angle(s) in degrees.
    name : str
        Input label used in error messages.

    Returns
    -------
    ndarray
        Finite validated values as ``float`` without changing representation.
    """
    return validate_oriented_reference_dip(dip, name=name)


def _validate_internal_dip_0_180(dip, name='dip') -> np.ndarray:
    """Return finite internal dip values strictly inside ``(0, 180)``."""
    values = np.asarray(dip, dtype=float)
    invalid = ~np.isfinite(values) | (values <= 0.0) | (values >= 180.0)
    if np.any(invalid):
        indices = np.argwhere(invalid).reshape(-1)[:5].tolist()
        raise ValueError(
            f"{name} must be finite and strictly inside (0, 180) degrees; "
            f"invalid value indices: {indices}"
        )
    return values


class DepthDipProfile:
    """
    A class to represent depth-dip relationship at a reference node.
    
    Internally stores dip strictly in ``(0, 180)`` for continuous
    interpolation. Query methods return either the signed reference form or
    the continuous internal form according to ``output_range``.
    
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
            Strict low-level input convention. ``'neg90_90'`` accepts
            ``[-90, 0) U (0, 90]``; ``'0_180'`` accepts ``(0, 180)``.
            High-level geometry APIs accept either equivalent reference form
            and call :meth:`from_oriented_reference`.
        """
        if input_dip_range not in _DIP_RANGES:
            raise ValueError(
                "input_dip_range must be 'neg90_90' or '0_180'; "
                f"got {input_dip_range!r}"
            )
        if interpolation_method not in _INTERPOLATION_METHODS:
            raise ValueError(
                "interpolation_method must be one of "
                f"{sorted(_INTERPOLATION_METHODS)}; got {interpolation_method!r}"
            )

        coordinates = np.asarray([x, y, lon, lat], dtype=float)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("profile x, y, lon and lat must be finite")

        pairs = np.asarray(depth_dip_pairs, dtype=float)
        pairs = np.atleast_2d(pairs)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(
                "depth_dip_pairs must have shape (n, 2) with columns "
                "[depth, dip]"
            )
        minimum_points = 4 if interpolation_method == 'cubic' else 2
        if pairs.shape[0] < minimum_points:
            raise ValueError(
                f"interpolation_method={interpolation_method!r} requires at "
                f"least {minimum_points} depth-dip points; got {pairs.shape[0]}"
            )
        if not np.all(np.isfinite(pairs)):
            raise ValueError("depth_dip_pairs must contain only finite values")

        self.x = float(x)
        self.y = float(y)
        self.lon = float(lon)
        self.lat = float(lat)
        self.interpolation_method = interpolation_method
        self.depth_dip_pairs = pairs.copy()

        if input_dip_range == 'neg90_90':
            signed = self.depth_dip_pairs[:, 1]
            invalid = (
                (signed < -90.0)
                | (signed > 90.0)
                | np.isclose(signed, 0.0, rtol=0.0, atol=1e-12)
            )
            if np.any(invalid):
                raise ValueError(
                    "depth_dip_pairs dip values with "
                    "input_dip_range='neg90_90' must be in "
                    "[-90, 0) U (0, 90] degrees"
                )
            self.depth_dip_pairs[:, 1] = normalize_dip_to_0_180(signed)
        else:
            self.depth_dip_pairs[:, 1] = _validate_internal_dip_0_180(
                self.depth_dip_pairs[:, 1],
                name='depth_dip_pairs dip',
            )

        sort_idx = np.argsort(self.depth_dip_pairs[:, 0])
        self.depth_dip_pairs = self.depth_dip_pairs[sort_idx]
        if np.any(np.diff(self.depth_dip_pairs[:, 0]) <= 0.0):
            raise ValueError(
                "depth_dip_pairs depths must be unique; duplicate depths "
                "cannot define an unambiguous profile"
            )

        self._create_interpolator(interpolation_method)

    @classmethod
    def from_oriented_reference(
            cls, x, y, lon, lat, depth_dip_pairs,
            interpolation_method='linear'):
        """Build a profile from the public oriented-reference dip protocol.

        The public input may use signed or continuous equivalent values. This
        constructor validates and normalizes once at the API boundary, then
        delegates to the strict 0-180 internal constructor.
        """
        pairs = np.asarray(depth_dip_pairs, dtype=float)
        pairs = np.atleast_2d(pairs)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(
                "depth_dip_pairs must have shape (n, 2) with columns "
                "[depth, dip]"
            )
        pairs = pairs.copy()
        pairs[:, 1] = normalize_oriented_reference_dip(
            pairs[:, 1],
            name='depth_dip_pairs dip',
        )
        return cls(
            x, y, lon, lat, pairs,
            interpolation_method=interpolation_method,
            input_dip_range='0_180',
        )


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
        if output_range not in _DIP_RANGES:
            raise ValueError(
                "output_range must be 'neg90_90' or '0_180'; "
                f"got {output_range!r}"
            )
        dip_0_180 = float(self.interpolator(depth))
        _validate_internal_dip_0_180(dip_0_180, name='interpolated dip')

        if output_range == 'neg90_90':
            return normalize_dip_to_neg90_90(dip_0_180)
        return dip_0_180
    
    def get_dip_0_180(self, depth: float) -> float:
        """Get dip angle in [0, 180] range (for internal calculations)."""
        dip = float(self.interpolator(depth))
        _validate_internal_dip_0_180(dip, name='interpolated dip')
        return dip
    
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
        """Create from serialized public reference-dip values."""
        # Use stored method if available, otherwise use provided or default
        method = interpolation_method or data.get('interpolation_method', 'linear')
        return cls.from_oriented_reference(
            x=data['x'],
            y=data['y'],
            lon=data['lon'],
            lat=data['lat'],
            depth_dip_pairs=np.array(data['depth_dip_pairs']),
            interpolation_method=method,
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
        if output_range not in _DIP_RANGES:
            raise ValueError(
                "output_range must be 'neg90_90' or '0_180'; "
                f"got {output_range!r}"
            )
        spatial_interp = self._get_spatial_interpolator_0_180(depth)
        dip_0_180 = float(spatial_interp(x, y))
        _validate_internal_dip_0_180(
            dip_0_180,
            name='spatially interpolated dip',
        )

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
