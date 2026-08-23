from .base_converter import BaseSlipConverter
from .geometry_utils import ReferencePoint, FaultGeometry
import numpy as np
import pandas as pd
from pyproj import Proj
import os


class SDMSlipConverter(BaseSlipConverter):
    """
    SDM slip distribution data converter.
    
    Handles SDM (Seismic Displacement Model) format files and converts them to standard formats.
    """
    
    def __init__(self, slip_file='slip.dat'):
        super().__init__(slip_file)
        
        # Set SDM-specific parameters
        self.reference_point = ReferencePoint.CENTER
        self.depth_unit = 'km'
        self.length_unit = 'km'
        self.width_unit = 'km'
        self.converter_type = 'pyproj'
        
        # SDM-specific attributes
        self.df = None
        self.proj = None
        self.utm_zone = None
        
        # Define the core columns we need from SDM output
        self.core_columns = ['lat_deg', 'lon_deg', 'depth_km', 'length_km', 'width_km', 
                           'slp_strk_m', 'slp_ddip_m', 'slp_am_m', 'strike_deg', 'dip_deg', 'rake_deg']
    
    def read_slip_file(self):
        """
        Read SDM slip distribution file with flexible column detection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(self.input_file):
            print(f"Error: SDM slip file {self.input_file} not found!")
            return False
        
        try:
            # Read SDM data with automatic whitespace separation
            df = pd.read_csv(self.input_file, delim_whitespace=True)
            
            # Column assignment based on SDM output format
            n_cols = len(df.columns)
            
            if n_cols == 11 or n_cols == 14:  # Basic SDM format or with MPa but no local coords
                col_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            elif n_cols == 13 or n_cols == 16:  # SDM with local coords with/ without MPa
                col_indices = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12]  # Skip x_local(3), y_local(4)
            else:
                raise ValueError(f"Unexpected SDM file format with {n_cols} columns")
            
            # Create new dataframe with standardized column names
            self.df = pd.DataFrame()
            for i, col_name in enumerate(self.core_columns):
                if i < len(col_indices):
                    self.df[col_name] = df.iloc[:, col_indices[i]]
            
            # Setup projection
            self._setup_projection()
            
            # Populate standard slip_data format
            self.slip_data = {
                'longitude': self.df['lon_deg'].values,
                'latitude': self.df['lat_deg'].values,
                'depth': self.df['depth_km'].values,
                'length': self.df['length_km'].values,
                'width': self.df['width_km'].values,
                'strike': self.df['strike_deg'].values,
                'dip': self.df['dip_deg'].values,
                'strike_slip': self.df['slp_strk_m'].values,
                'dip_slip': - self.df['slp_ddip_m'].values,
                'total_slip': self.df['slp_am_m'].values
            }
            
            print(f"Loaded {len(self.df)} SDM fault patches")
            print(f"Slip range: {self.df['slp_am_m'].min():.3f} - {self.df['slp_am_m'].max():.3f} m")
            
            return True
            
        except Exception as e:
            print(f"Error reading SDM file: {e}")
            return False
    
    def _setup_projection(self):
        """
        Setup UTM projection and fault geometry calculator.
        """
        if self.df is not None and len(self.df) > 0:
            lat_ref, lon_ref = self.df.iloc[0]['lat_deg'], self.df.iloc[0]['lon_deg']
            self.utm_zone = int((lon_ref + 180) // 6) + 1
            self.proj = Proj(proj='utm', zone=self.utm_zone, ellps='WGS84')
            self.fault_geometry = FaultGeometry(self.proj)
            
            print(f"Using UTM zone: {self.utm_zone}")
            print(f"Reference point: ({lat_ref:.4f}°, {lon_ref:.4f}°)")


def main():
    """
    Main function to demonstrate the SDM slip converter usage.
    """
    converter = SDMSlipConverter('slip.dat')
    converter.convert_all()


if __name__ == "__main__":
    main()