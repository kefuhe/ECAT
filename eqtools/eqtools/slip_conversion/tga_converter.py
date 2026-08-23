from .base_converter import BaseSlipConverter
from .geometry_utils import ReferencePoint, FaultGeometry
import numpy as np
import pandas as pd
from pyproj import Proj
import os


class TGASlipConverter(BaseSlipConverter):
    """
    TGA slip distribution data converter.
    
    Handles TGA (Tokyo Geodesy Archive) format files and converts them to standard formats.
    
    TGA format columns:
    Long(deg) Lati(deg) Z(m) Width(m) Length(m) Strike(deg) Dip(deg) U1(cm) U2(cm) U3(cm) 
    Adj1 Adj2 Adj3 Adj4 HWBlock FWBlock
    
    Where:
    - U1: strike slip (cm)
    - U2: dip slip (cm) 
    - U3: opening/tensile slip (cm)
    """
    
    def __init__(self, slip_file='tga_slip.txt'):
        super().__init__(slip_file)
        
        # Set TGA-specific parameters
        self.reference_point = ReferencePoint.BOTTOM_LEFT  # TGA typically uses bottom-left as reference
        self.depth_unit = 'm'
        self.length_unit = 'm'
        self.width_unit = 'm'
        self.converter_type = 'pyproj'
        
        # TGA-specific attributes
        self.df = None
        self.proj = None
        self.utm_zone = None
        
        # Define TGA column names
        self.tga_columns = [
            'Long(deg)', 'Lati(deg)', 'Z(m)', 'Width(m)', 'Length(m)', 
            'Strike(deg)', 'Dip(deg)', 'U1(cm)', 'U2(cm)', 'U3(cm)',
            'Adj1', 'Adj2', 'Adj3', 'Adj4', 'HWBlock', 'FWBlock'
        ]
    
    def read_slip_file(self):
        """
        Read TGA slip distribution file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(self.input_file):
            print(f"Error: TGA slip file {self.input_file} not found!")
            return False
        
        try:
            # Read TGA data - skip header if present
            with open(self.input_file, 'r') as f:
                first_line = f.readline().strip()
                
            # Check if first line is header
            has_header = 'Long(deg)' in first_line or 'Lati(deg)' in first_line
            skip_rows = 1 if has_header else 0
            
            # Read data with flexible whitespace separation
            self.df = pd.read_csv(
                self.input_file, 
                delim_whitespace=True, 
                skiprows=skip_rows,
                names=self.tga_columns[:16] if len(self.tga_columns) >= 16 else None
            )
            
            # Validate minimum required columns
            required_cols = 10  # Up to U3(cm)
            if len(self.df.columns) < required_cols:
                raise ValueError(f"TGA file must have at least {required_cols} columns, found {len(self.df.columns)}")
            
            # If we have more or fewer columns than expected, adjust
            if len(self.df.columns) != len(self.tga_columns):
                print(f"Warning: Expected {len(self.tga_columns)} columns, found {len(self.df.columns)}")
                # Use first 10 columns for core data
                col_names = self.tga_columns[:min(len(self.df.columns), len(self.tga_columns))]
                self.df.columns = col_names + [f'Col{i}' for i in range(len(col_names), len(self.df.columns))]
            else:
                self.df.columns = self.tga_columns
            
            # Setup projection
            self._setup_projection()
            
            # Convert units and calculate total slip
            # TGA: U1=strike_slip(cm), U2=dip_slip(cm), U3=opening(cm)
            strike_slip_m = self.df['U1(cm)'].values / 100.0  # cm to m
            dip_slip_m = - self.df['U2(cm)'].values / 100.0     # cm to m
            opening_m = self.df['U3(cm)'].values / 100.0      # cm to m
            
            # Calculate total slip magnitude (ignoring opening for now)
            total_slip = np.sqrt(strike_slip_m**2 + dip_slip_m**2)
            
            # Convert depth to positive downward (TGA Z is typically negative upward)
            depth_positive = np.abs(self.df['Z(m)'].values)
            
            # Populate standard slip_data format
            self.slip_data = {
                'longitude': self.df['Long(deg)'].values,
                'latitude': self.df['Lati(deg)'].values,
                'depth': depth_positive,  # Convert to positive downward
                'length': self.df['Length(m)'].values,
                'width': self.df['Width(m)'].values,
                'strike': self.df['Strike(deg)'].values,
                'dip': self.df['Dip(deg)'].values,
                'strike_slip': strike_slip_m,
                'dip_slip': dip_slip_m,
                'total_slip': total_slip,
                'opening': opening_m  # Additional field for TGA
            }
            
            print(f"Loaded {len(self.df)} TGA fault patches")
            print(f"Longitude range: {self.df['Long(deg)'].min():.4f} - {self.df['Long(deg)'].max():.4f}°")
            print(f"Latitude range: {self.df['Lati(deg)'].min():.4f} - {self.df['Lati(deg)'].max():.4f}°")
            if np.any(opening_m != 0):
                print(f"Opening range: {opening_m.min():.3f} - {opening_m.max():.3f} m")
            
            return True
            
        except Exception as e:
            print(f"Error reading TGA file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_projection(self):
        """
        Setup UTM projection and fault geometry calculator.
        """
        if self.df is not None and len(self.df) > 0:
            # Use center of data for projection
            lat_ref = self.df['Lati(deg)'].mean()
            lon_ref = self.df['Long(deg)'].mean()
            
            # Calculate UTM zone
            self.utm_zone = int((lon_ref + 180) // 6) + 1
            
            # Setup projection
            self.proj = Proj(proj='utm', zone=self.utm_zone, ellps='WGS84')
            self.fault_geometry = FaultGeometry(self.proj)
            
            print(f"Using UTM zone: {self.utm_zone}")
            print(f"Reference point: ({lat_ref:.4f}°, {lon_ref:.4f}°)")


def main():
    """
    Main function to demonstrate the TGA slip converter usage.
    """
    converter = TGASlipConverter('slip.rec')
    converter.convert_all()


if __name__ == "__main__":
    main()