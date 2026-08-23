from enum import Enum
import numpy as np
from pyproj import Proj

class ReferencePoint(Enum):
    """Enumeration for fault patch reference point."""
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"

class FaultGeometry:
    """
    A general class for fault patch geometry calculations.
    
    This class provides methods to calculate fault patch corners with different
    reference points, units, and coordinate systems.
    """
    
    def __init__(self, proj_or_converter=None):
        """
        Initialize fault geometry calculator.
        
        Args:
            proj_or_converter: Either pyproj.Proj object or coordinate converter
        """
        self.proj_or_converter = proj_or_converter
    
    def _normalize_units(self, value, unit_from, unit_to):
        """
        Convert between different units.
        
        Args:
            value (float): Value to convert
            unit_from (str): Source unit ('m', 'km', 'cm')
            unit_to (str): Target unit ('m', 'km', 'cm')
            
        Returns:
            float: Converted value
        """
        conversions = {
            ('m', 'km'): 0.001,
            ('km', 'm'): 1000.0,
            ('cm', 'm'): 0.01,
            ('m', 'cm'): 100.0,
            ('cm', 'km'): 0.00001,
            ('km', 'cm'): 100000.0,
            ('m', 'm'): 1.0,
            ('km', 'km'): 1.0,
            ('cm', 'cm'): 1.0
        }
        return value * conversions.get((unit_from, unit_to), 1.0)
    
    def _convert_to_utm(self, lon, lat, converter_type='pyproj'):
        """
        Convert lon/lat to UTM coordinates.
        
        Args:
            lon (float): Longitude
            lat (float): Latitude
            converter_type (str): Type of converter ('pyproj' or 'custom')
            
        Returns:
            tuple: (x_utm, y_utm) in meters
        """
        if converter_type == 'pyproj':
            return self.proj_or_converter(lon, lat)
        elif converter_type == 'custom':
            # For JinInv-style converter
            x_rel, y_rel = self.proj_or_converter.lonlat_to_utm(lon, lat)
            x_utm = x_rel + self.proj_or_converter.center_utm_x
            y_utm = y_rel + self.proj_or_converter.center_utm_y
            return x_utm, y_utm
        else:
            raise ValueError(f"Unknown converter type: {converter_type}")
    
    def _convert_from_utm(self, x_utm, y_utm, converter_type='pyproj'):
        """
        Convert UTM coordinates to lon/lat.
        
        Args:
            x_utm (float): UTM X coordinate
            y_utm (float): UTM Y coordinate
            converter_type (str): Type of converter ('pyproj' or 'custom')
            
        Returns:
            tuple: (lon, lat) in degrees
        """
        if converter_type == 'pyproj':
            return self.proj_or_converter(x_utm, y_utm, inverse=True)
        elif converter_type == 'custom':
            # For JinInv-style converter
            x_rel = x_utm - self.proj_or_converter.center_utm_x
            y_rel = y_utm - self.proj_or_converter.center_utm_y
            return self.proj_or_converter.utm_to_lonlat(x_rel, y_rel)
        else:
            raise ValueError(f"Unknown converter type: {converter_type}")
    
    def calculate_fault_corners(self, lon, lat, depth, length, width, strike, dip,
                                pos_s=0.0, pos_d=0.0,
                                reference_point=ReferencePoint.CENTER,
                                depth_unit='km', length_unit='km', width_unit='km',
                                converter_type='pyproj'):
        """
        Calculate the four corner points of a rectangular fault patch.
        
        Args:
            lon (float): Longitude of reference point (degrees)
            lat (float): Latitude of reference point (degrees)
            depth (float): Depth of reference point (positive downward)
            length (float): Fault length along strike
            width (float): Fault width along dip
            strike (float): Strike angle (degrees, 0-360)
            dip (float): Dip angle (degrees, 0-90)
            pos_s (float): Position along strike relative to reference point (default: 0.0)
            pos_d (float): Position along dip relative to reference point (default: 0.0)
            reference_point (ReferencePoint): Reference point on fault patch
            depth_unit (str): Unit for depth ('m', 'km', 'cm')
            length_unit (str): Unit for length ('m', 'km', 'cm')
            width_unit (str): Unit for width ('m', 'km', 'cm')
            converter_type (str): Type of coordinate converter ('pyproj' or 'custom')
            
        Returns:
            list: Four corner points as [(lon1, lat1, depth1), ...]
                    Order: top-left, top-right, bottom-right, bottom-left
        """
        if self.proj_or_converter is None:
            raise ValueError("Coordinate converter not initialized.")
        
        # Convert all dimensions to meters for calculation
        depth_m = self._normalize_units(depth, depth_unit, 'm')
        length_m = self._normalize_units(length, length_unit, 'm')
        width_m = self._normalize_units(width, width_unit, 'm')
        pos_s_m = self._normalize_units(pos_s, length_unit, 'm')
        pos_d_m = self._normalize_units(pos_d, width_unit, 'm')
        
        # Convert to UTM coordinates
        x_ref, y_ref = self._convert_to_utm(lon, lat, converter_type)
        
        # Handle dip angle > 90 degrees (for JinInv compatibility)
        if dip > 90:
            strike = (strike - 180) % 360
            dip = 180 - dip
        
        # Convert angles to radians
        strike_rad = np.radians(-strike)  # Negative for proper rotation
        dip_rad = np.radians(dip)
        
        # Calculate half dimensions
        half_length = length_m / 2
        half_width = width_m / 2
        
        # Calculate projection of width onto horizontal and vertical
        dx_horz = half_width * np.cos(dip_rad)
        dz = half_width * np.sin(dip_rad)
        
        # Calculate position offset due to pos_s and pos_d
        pos_d_horz = pos_d_m * np.cos(dip_rad)  # Horizontal component of pos_d
        pos_d_vert = pos_d_m * np.sin(dip_rad)  # Vertical component of pos_d
        
        # Define corner offsets based on reference point
        if reference_point == ReferencePoint.CENTER:
            # For PSCMP and SDM-style: reference point is center, pos_s and pos_d are offsets
            base_depth = depth_m + pos_d_vert  # Adjust depth by pos_d vertical component
            
            corner_offsets = [
                (pos_d_horz - dx_horz, pos_s_m - half_length, base_depth - dz),  # Top-left
                (pos_d_horz - dx_horz, pos_s_m + half_length, base_depth - dz),  # Top-right
                (pos_d_horz + dx_horz, pos_s_m + half_length, base_depth + dz),  # Bottom-right
                (pos_d_horz + dx_horz, pos_s_m - half_length, base_depth + dz)   # Bottom-left
            ]
        elif reference_point == ReferencePoint.TOP_LEFT:
            # For JinInv-style: reference point is top-left corner
            base_depth = depth_m + pos_d_vert
            
            corner_offsets = [
                (pos_d_horz, pos_s_m, base_depth),                                    # Top-left (reference)
                (pos_d_horz, pos_s_m + length_m, base_depth),                         # Top-right
                (pos_d_horz + 2*dx_horz, pos_s_m + length_m, base_depth + 2*dz),     # Bottom-right
                (pos_d_horz + 2*dx_horz, pos_s_m, base_depth + 2*dz)                 # Bottom-left
            ]
        elif reference_point == ReferencePoint.TOP_RIGHT:
            base_depth = depth_m + pos_d_vert
            
            corner_offsets = [
                (pos_d_horz, pos_s_m - length_m, base_depth),                        # Top-left
                (pos_d_horz, pos_s_m, base_depth),                                   # Top-right (reference)
                (pos_d_horz + 2*dx_horz, pos_s_m, base_depth + 2*dz),               # Bottom-right
                (pos_d_horz + 2*dx_horz, pos_s_m - length_m, base_depth + 2*dz)     # Bottom-left
            ]
        elif reference_point == ReferencePoint.BOTTOM_LEFT:
            base_depth = depth_m + pos_d_vert
            
            corner_offsets = [
                (pos_d_horz - 2*dx_horz, pos_s_m, base_depth - 2*dz),               # Top-left
                (pos_d_horz - 2*dx_horz, pos_s_m + length_m, base_depth - 2*dz),    # Top-right
                (pos_d_horz, pos_s_m + length_m, base_depth),                        # Bottom-right
                (pos_d_horz, pos_s_m, base_depth)                                    # Bottom-left (reference)
            ]
        elif reference_point == ReferencePoint.BOTTOM_RIGHT:
            base_depth = depth_m + pos_d_vert
            
            corner_offsets = [
                (pos_d_horz - 2*dx_horz, pos_s_m - length_m, base_depth - 2*dz),    # Top-left
                (pos_d_horz - 2*dx_horz, pos_s_m, base_depth - 2*dz),               # Top-right
                (pos_d_horz, pos_s_m, base_depth),                                   # Bottom-right (reference)
                (pos_d_horz, pos_s_m - length_m, base_depth)                         # Bottom-left
            ]
        else:
            raise ValueError(f"Unknown reference point: {reference_point}")
        
        # Rotate corners around z-axis by strike angle
        corners_utm = []
        for dx, dy, z in corner_offsets:
            # Complex number rotation
            xy_rotated = (dx + dy * 1j) * np.exp(1j * strike_rad)
            x_rot, y_rot = xy_rotated.real, xy_rotated.imag
            # Translate to reference point
            corners_utm.append((x_rot + x_ref, y_rot + y_ref, z))
        
        # Convert back to lat/lon coordinates
        corners_geo = []
        for x, y, z in corners_utm:
            lon_corner, lat_corner = self._convert_from_utm(x, y, converter_type)
            depth_corner = self._normalize_units(z, 'm', depth_unit)
            corners_geo.append((lon_corner, lat_corner, depth_corner))
        
        return corners_geo


class CoordinateConverter:
    """Coordinate converter class for UTM and geographic coordinate transformations"""
    
    def __init__(self, lon0, lat0):
        """
        Initialize coordinate converter
        
        Parameters:
        -----------
        lon0, lat0 : float
            Longitude and latitude of projection center (degrees)
        """
        from pyproj import Transformer
        
        self.lon0 = lon0
        self.lat0 = lat0
        
        # Determine UTM zone
        self.utm_zone = int((lon0 + 180) / 6) + 1
        
        # Determine northern/southern hemisphere
        self.hemisphere = 'north' if lat0 >= 0 else 'south'
        
        # Define projections
        self.utm_proj = Proj(proj='utm', zone=self.utm_zone, ellps='WGS84', datum='WGS84')
        self.wgs84_proj = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        
        # Create coordinate transformers
        self.transformer_to_utm = Transformer.from_proj(self.wgs84_proj, self.utm_proj, always_xy=True)
        self.transformer_to_wgs84 = Transformer.from_proj(self.utm_proj, self.wgs84_proj, always_xy=True)
        
        # Calculate center point UTM coordinates
        self.center_utm_x, self.center_utm_y = self.transformer_to_utm.transform(lon0, lat0)
        
        print(f"Coordinate converter initialized:")
        print(f"  Projection center: {lon0}°E, {lat0}°N")
        print(f"  UTM zone: {self.utm_zone}{self.hemisphere[0].upper()}")
        print(f"  Center UTM coordinates: {self.center_utm_x:.2f}, {self.center_utm_y:.2f}")
    
    def lonlat_to_utm(self, lon, lat):
        """
        Convert longitude/latitude to relative UTM coordinates
        
        Parameters:
        -----------
        lon, lat : array_like
            Longitude and latitude coordinates (degrees)
            
        Returns:
        --------
        x, y : ndarray
            UTM coordinates relative to center (meters)
        """
        # Convert to absolute UTM coordinates
        utm_x, utm_y = self.transformer_to_utm.transform(lon, lat)
        
        # Calculate relative coordinates
        x = utm_x - self.center_utm_x
        y = utm_y - self.center_utm_y
        
        return x, y
    
    def utm_to_lonlat(self, x, y):
        """
        Convert relative UTM coordinates to longitude/latitude
        
        Parameters:
        -----------
        x, y : array_like
            UTM coordinates relative to center (meters)
            
        Returns:
        --------
        lon, lat : ndarray
            Longitude and latitude coordinates (degrees)
        """
        # Convert to absolute UTM coordinates
        utm_x = self.center_utm_x + x
        utm_y = self.center_utm_y + y
        
        # Convert to longitude/latitude
        lon, lat = self.transformer_to_wgs84.transform(utm_x, utm_y)
        
        return lon, lat