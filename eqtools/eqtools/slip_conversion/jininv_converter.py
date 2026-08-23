"""
JinInv slip distribution data converter and visualizer.

This module provides tools to convert JinInv format slip model data to GMT format
and create 3D visualizations.

JinInv data format:
[indx_fault,indx_patch,indx_depth,x_start,y_start,z_start,length_patch(m),width_patch(m),strike,dip,tp,strike_slip(cm),dip_slip(cm)]
"""

from .base_converter import BaseSlipConverter
from .geometry_utils import CoordinateConverter, FaultGeometry, ReferencePoint
import numpy as np
import scipy.io as sio
import os


class JinInvSlipConverter(BaseSlipConverter):
    """
    JinInv slip distribution data converter.
    
    Handles JinInv MATLAB format files and converts them to standard formats.
    """
    
    def __init__(self, mat_file='slip.mat', lon0=None, lat0=None):
        super().__init__(mat_file)
        
        # Set JinInv-specific parameters
        self.reference_point = ReferencePoint.TOP_LEFT
        self.depth_unit = 'm'
        self.length_unit = 'm'
        self.width_unit = 'm'
        self.converter_type = 'custom'
        
        # JinInv-specific attributes
        self.converter = None
        self.slip_model = None
        
        # Manual projection parameters
        self.manual_lon0 = lon0
        self.manual_lat0 = lat0
    
    def read_slip_file(self):
        """
        Read JinInv MATLAB file with slip distribution data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(self.input_file):
            print(f"Error: JinInv file {self.input_file} not found!")
            return False
        
        try:
            # Read MATLAB file
            mat_data = sio.loadmat(self.input_file)
            
            # Get projection parameters - try file first, then manual input
            lon0 = None
            lat0 = None
            
            # Try to get from file
            if 'lon0' in mat_data and 'lat0' in mat_data:
                lon0 = float(mat_data['lon0'][0, 0])
                lat0 = float(mat_data['lat0'][0, 0])
                print(f"Using projection parameters from file: lon0={lon0:.4f}, lat0={lat0:.4f}")
            
            # Use manual input if file doesn't contain them or if manually specified
            if self.manual_lon0 is not None and self.manual_lat0 is not None:
                lon0 = self.manual_lon0
                lat0 = self.manual_lat0
                print(f"Using manual projection parameters: lon0={lon0:.4f}, lat0={lat0:.4f}")
            
            # Check if we have valid projection parameters
            if lon0 is None or lat0 is None:
                print("Error: Missing projection parameters (lon0, lat0)")
                print("Please provide them manually when creating the converter:")
                print("converter = JinInvSlipConverter('file.mat', lon0=your_lon0, lat0=your_lat0)")
                return False
            
            # Initialize coordinate converter
            self.converter = CoordinateConverter(lon0, lat0)
            self._setup_projection()
            
            # Get slip model data
            if 'slip_model' in mat_data:
                data_array = mat_data['slip_model']
            else:
                # Search for likely data arrays
                data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
                print(f"Available keys in MATLAB file: {data_keys}")
                
                for key in data_keys:
                    if 'model' in key.lower() or 'slip' in key.lower():
                        data_array = mat_data[key]
                        print(f"Using data array from key: '{key}'")
                        break
                else:
                    # If no obvious key found, use the first non-metadata key
                    if data_keys:
                        data_array = mat_data[data_keys[0]]
                        print(f"Using data array from first available key: '{data_keys[0]}'")
                    else:
                        raise ValueError("No data arrays found in MATLAB file")
            
            # Ensure data is 2D
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)
            
            print(f"Data array shape: {data_array.shape}")
            
            if data_array.shape[1] < 13:
                raise ValueError(f"Insufficient columns, expected at least 13, got {data_array.shape[1]}")
            
            # Parse JinInv format data
            self.slip_model = {
                'x_start': data_array[:, 3],
                'y_start': data_array[:, 4],
                'z_start': -data_array[:, 5],  # Convert to positive downward
                'length_patch': data_array[:, 6],
                'width_patch': data_array[:, 7],
                'strike': data_array[:, 8],
                'dip': data_array[:, 9],
                'strike_slip': data_array[:, 11]/100.0,  # Convert cm to m
                'dip_slip': -data_array[:, 12]/100.0,    # Convert cm to m, convert to positive for reverse slip
            }
            
            # Convert UTM coordinates to lat/lon
            lon, lat = self.converter.utm_to_lonlat(
                self.slip_model['x_start'], 
                self.slip_model['y_start']
            )
            
            # Calculate total slip
            total_slip = np.sqrt(self.slip_model['strike_slip']**2 + 
                               self.slip_model['dip_slip']**2)
            
            # Populate standard slip_data format
            self.slip_data = {
                'longitude': lon,
                'latitude': lat,
                'depth': self.slip_model['z_start'],
                'length': self.slip_model['length_patch'],
                'width': self.slip_model['width_patch'],
                'strike': self.slip_model['strike'],
                'dip': self.slip_model['dip'],
                'strike_slip': self.slip_model['strike_slip'],
                'dip_slip': self.slip_model['dip_slip'],
                'total_slip': total_slip
            }
            
            print(f"Loaded {len(total_slip)} JinInv fault patches")
            print(f"Longitude range: {np.min(lon):.4f} - {np.max(lon):.4f}°")
            print(f"Latitude range: {np.min(lat):.4f} - {np.max(lat):.4f}°")
            print(f"Slip range: {np.min(total_slip):.3f} - {np.max(total_slip):.3f} m")
            
            return True
            
        except Exception as e:
            print(f"Error reading JinInv file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_projection(self):
        """
        Setup fault geometry calculator for JinInv format.
        """
        if self.converter is not None:
            self.fault_geometry = FaultGeometry(self.converter)


def main():
    """
    Main function to demonstrate the JinInv slip converter usage.
    """
    # Example 1: Try to use projection parameters from file
    converter = JinInvSlipConverter('model_myanmar_2025_68E_82W.mat')
    
    # Example 2: Manually provide projection parameters
    # converter = JinInvSlipConverter('model_myanmar_2025_68E_82W.mat', 
    #                                lon0=95.0, lat0=23.0)
    
    converter.convert_all()


if __name__ == "__main__":
    main()