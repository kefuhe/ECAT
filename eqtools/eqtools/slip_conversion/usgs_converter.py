import json
import os

import numpy as np
from pyproj import Geod

from .base_converter import BaseSlipConverter
from .geometry_utils import ReferencePoint


_WGS84_GEOD = Geod(ellps="WGS84")


class USGSGeoJSONConverter(BaseSlipConverter):
    """
    USGS GeoJSON slip distribution data converter.
    
    Handles USGS finite fault model data in GeoJSON format.
    Format: GeoJSON with Polygon features containing slip properties.
    """
    
    def __init__(self, slip_file='FFM.geojson'):
        super().__init__(slip_file)
        
        # Set USGS-specific parameters
        self.reference_point = ReferencePoint.CENTER
        self.depth_unit = 'km'  # Will convert from meters in GeoJSON
        self.length_unit = 'km'
        self.width_unit = 'km'
        self.converter_type = 'geojson'
        
        # USGS-specific attributes
        self.geojson_data = None
        self.patches_data = []
        
    def read_slip_file(self):
        """
        Read USGS GeoJSON slip distribution file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(self.input_file):
            print(f"Error: USGS GeoJSON file {self.input_file} not found!")
            return False
        
        try:
            # Read GeoJSON file
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.geojson_data = json.load(f)
            
            # Parse GeoJSON data
            self._parse_geojson_data()
            
            # Convert to standard format
            self._convert_to_standard_format()
            
            print(f"Loaded {len(self.slip_data['longitude'])} USGS fault patches")
            print(f"Slip range: {np.min(self.slip_data['total_slip']):.3f} - {np.max(self.slip_data['total_slip']):.3f} m")
            print(f"Depth range: {np.min(self.slip_data['depth']):.1f} - {np.max(self.slip_data['depth']):.1f} km")
            
            return True
            
        except Exception as e:
            print(f"Error reading USGS GeoJSON file: {e}")
            return False
    
    def _parse_geojson_data(self):
        """
        Parse GeoJSON data and extract fault patches.
        """
        self.patches_data = []
        
        for feature in self.geojson_data['features']:
            geometry = feature['geometry']
            properties = feature['properties']
            
            # Get coordinates (GeoJSON format: [lon, lat, depth])
            coordinates = geometry['coordinates'][0]  # Polygon exterior ring
            
            # Get slip properties
            rake = properties.get('rake', 0)
            slip = properties.get('slip', 0)
            strike = properties.get('strike', 0)
            dip = properties.get('dip', 90)
            
            # Calculate strike-slip and dip-slip components
            ss = slip * np.cos(np.radians(rake))
            ds = slip * np.sin(np.radians(rake))
            
            # Calculate patch center and dimensions
            lons = [coord[0] for coord in coordinates[:-1]]  # Exclude last point (same as first)
            lats = [coord[1] for coord in coordinates[:-1]]
            depths = [coord[2] for coord in coordinates[:-1]]
            
            # Calculate center point
            center_lon = np.mean(lons)
            center_lat = np.mean(lats)
            center_depth = np.mean(depths) / 1000.0  # Convert to km
            
            # Estimate horizontal patch dimensions on the WGS84 ellipsoid.
            # pyproj is already a core eqtools dependency, so GeoJSON support
            # does not need separate geojson or geopy distributions.
            _, _, length_m = _WGS84_GEOD.inv(
                lons[0], lats[0], lons[1], lats[1]
            )
            _, _, width_m = _WGS84_GEOD.inv(
                lons[0], lats[0], lons[-1], lats[-1]
            )
            
            length_km = length_m / 1000.0
            width_km = width_m / 1000.0
            
            # Store patch data with original coordinates
            patch_data = {
                'longitude': center_lon,
                'latitude': center_lat,
                'depth': center_depth,
                'length': length_km,
                'width': width_km,
                'strike': strike,
                'dip': dip,
                'rake': rake,
                'strike_slip': ss,
                'dip_slip': ds,
                'total_slip': slip,
                'corners': [(coord[0], coord[1], coord[2] / 1000.0) for coord in coordinates[:-1]]
            }
            
            self.patches_data.append(patch_data)
    
    def _convert_to_standard_format(self):
        """
        Convert parsed data to standard slip_data format.
        """
        # Initialize arrays
        self.slip_data = {
            'longitude': np.array([patch['longitude'] for patch in self.patches_data]),
            'latitude': np.array([patch['latitude'] for patch in self.patches_data]),
            'depth': np.array([patch['depth'] for patch in self.patches_data]),
            'length': np.array([patch['length'] for patch in self.patches_data]),
            'width': np.array([patch['width'] for patch in self.patches_data]),
            'strike': np.array([patch['strike'] for patch in self.patches_data]),
            'dip': np.array([patch['dip'] for patch in self.patches_data]),
            'strike_slip': np.array([patch['strike_slip'] for patch in self.patches_data]),
            'dip_slip': np.array([patch['dip_slip'] for patch in self.patches_data]),
            'total_slip': np.array([patch['total_slip'] for patch in self.patches_data])
        }
    
    def _setup_projection(self):
        """
        Setup projection (not needed for GeoJSON format).
        """
        # GeoJSON already provides geographic coordinates
        # No projection setup needed
        pass
    
    def calculate_patch_corners(self, lon, lat, depth, length, width, strike, dip):
        """
        Calculate patch corners using the original GeoJSON coordinates.
        
        For USGS GeoJSON data, we use the actual polygon coordinates.
        """
        # Find the patch data for this location
        for patch in self.patches_data:
            if (abs(patch['longitude'] - lon) < 1e-6 and 
                abs(patch['latitude'] - lat) < 1e-6 and 
                abs(patch['depth'] - depth) < 1e-6):
                
                return patch['corners']
        
        # Fallback: return a simple rectangle
        return [(lon, lat, depth), (lon+0.01, lat, depth), 
                (lon+0.01, lat+0.01, depth), (lon, lat+0.01, depth)]


def main():
    """
    Main function to demonstrate the USGS GeoJSON converter usage.
    """
    converter = USGSGeoJSONConverter('FFM.geojson')
    converter.convert_all()


if __name__ == "__main__":
    main()
