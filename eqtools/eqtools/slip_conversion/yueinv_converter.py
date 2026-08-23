"""
Yueinv slip distribution data converter.
Handles Yueinv format files from Yue Han and converts them to standard formats.
"""

from .base_converter import BaseSlipConverter
from .geometry_utils import ReferencePoint, FaultGeometry
import numpy as np
import pandas as pd
from pyproj import Proj
import os


class YueinvSlipConverter(BaseSlipConverter):
    """
    Yueinv slip distribution data converter.
    
    Handles Yueinv format files (e.g., science.adi1519_slipmodel_event.txt) 
    and converts them to standard formats.
    
    Yueinv format columns:
    time time-dur lat lon depth strike dip rake Mo(Nm) slip(m) x y xinc yinc
    """
    
    def __init__(self, slip_file='science.adi1519_slipmodel_event.txt', lon0=None, lat0=None, skiprows=25):
        super().__init__(slip_file)
        
        # Set Yueinv-specific parameters
        self.reference_point = ReferencePoint.CENTER
        self.depth_unit = 'km'
        self.length_unit = 'km'
        self.width_unit = 'km'
        self.converter_type = 'pyproj'
        
        # Yueinv-specific attributes
        self.df = None
        self.proj = None
        self.utm_zone = None
        self.skiprows = skiprows
        
        # Manual projection parameters
        self.manual_lon0 = lon0
        self.manual_lat0 = lat0
        
        # Define Yueinv column names
        self.yueinv_columns = [
            'time', 'time-dur', 'lat', 'lon', 'depth', 'strike', 'dip', 
            'rake', 'Mo(Nm)', 'slip(m)', 'x', 'y', 'xinc', 'yinc', 'segnum', 'subnum', 'parnum'
        ]
    
    def read_slip_file(self):
        """
        Read Yueinv slip distribution file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(self.input_file):
            print(f"Error: Yueinv slip file {self.input_file} not found!")
            return False
        
        try:
            # Read Yueinv data with skiprows for header
            self.df = pd.read_csv(
                self.input_file, 
                skiprows=self.skiprows,
                delim_whitespace=True,
                names=self.yueinv_columns
            )
            
            # Validate required columns
            required_cols = ['lat', 'lon', 'depth', 'strike', 'dip', 'rake', 'slip(m)', 'xinc', 'yinc']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Setup projection
            self._setup_projection()
            
            # Convert rake to strike-slip and dip-slip components
            rake_rad = np.deg2rad(self.df['rake'].values)
            slip_total = self.df['slip(m)'].values
            strike_slip = slip_total * np.cos(rake_rad)
            dip_slip = slip_total * np.sin(rake_rad)
            
            # Populate standard slip_data format
            self.slip_data = {
                'longitude': self.df['lon'].values,
                'latitude': self.df['lat'].values,
                'depth': self.df['depth'].values,
                'length': self.df['xinc'].values,
                'width': self.df['yinc'].values,
                'strike': self.df['strike'].values,
                'dip': self.df['dip'].values,
                'strike_slip': strike_slip,
                'dip_slip': dip_slip,
                'total_slip': slip_total
            }
            
            print(f"Loaded {len(self.df)} Yueinv fault patches")
            print(f"Time range: {self.df['time'].min():.1f} - {self.df['time'].max():.1f} s")
            print(f"Slip range: {slip_total.min():.3f} - {slip_total.max():.3f} m")
            print(f"Rake range: {self.df['rake'].min():.1f} - {self.df['rake'].max():.1f}°")
            
            return True
            
        except Exception as e:
            print(f"Error reading Yueinv file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_projection(self):
        """
        Setup UTM projection and fault geometry calculator.
        """
        if self.df is not None and len(self.df) > 0:
            # Determine projection parameters
            if self.manual_lon0 is not None and self.manual_lat0 is not None:
                lon0, lat0 = self.manual_lon0, self.manual_lat0
                print(f"Using manual projection parameters: lon0={lon0:.4f}, lat0={lat0:.4f}")
            else:
                # Use data center as projection origin
                lon0 = self.df['lon'].mean()
                lat0 = self.df['lat'].mean()
                print(f"Using data center as projection origin: lon0={lon0:.4f}, lat0={lat0:.4f}")
            
            # Setup UTM projection
            self.utm_zone = int((lon0 + 180) // 6) + 1
            self.proj = Proj(proj='utm', zone=self.utm_zone, ellps='WGS84')
            self.fault_geometry = FaultGeometry(self.proj)
            
            print(f"Using UTM zone: {self.utm_zone}")


def main():
    """
    Main function to demonstrate the Yueinv slip converter usage.
    """
    # Example usage with automatic projection
    converter = YueinvSlipConverter('science.adi1519_slipmodel_event2.txt')
    
    # Example with manual projection parameters
    # converter = YueinvSlipConverter('science.adi1519_slipmodel_event2.txt', 
    #                                lon0=37.0, lat0=38.0)
    
    converter.convert_all()


if __name__ == "__main__":
    main()