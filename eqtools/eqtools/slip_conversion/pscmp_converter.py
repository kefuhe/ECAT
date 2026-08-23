"""
PSCMP to CSI GMT format converter

This module converts PSCMP fault definitions to CSI GMT format
by inheriting from BaseSlipConverter and overriding necessary methods.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict
from .base_converter import BaseSlipConverter
from .geometry_utils import ReferencePoint, FaultGeometry
from pyproj import Proj


class PSCMPConverter(BaseSlipConverter):
    """
    PSCMP slip distribution data converter.
    
    Handles PSCMP fault format files and converts them to standard formats
    by inheriting from BaseSlipConverter.
    """
    
    def __init__(self, slip_file='fault_sources.csv'):
        super().__init__(slip_file)
        
        # Set PSCMP-specific parameters
        self.reference_point = ReferencePoint.CENTER
        self.depth_unit = 'km'
        self.length_unit = 'km'
        self.width_unit = 'km'
        self.converter_type = 'pyproj'
        
        # PSCMP-specific attributes
        self.df = None
        self.proj = None
        self.utm_zone = None
    
    def read_slip_file(self):
        """
        Read PSCMP fault file with flexible format detection.
        Supports .csv, .xlsx, .json, .txt, .dat, and .inp formats.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(self.input_file):
            print(f"Error: PSCMP fault file {self.input_file} not found!")
            return False
        
        try:
            # Detect file format and read
            filepath = Path(self.input_file)
            file_format = filepath.suffix.lower()
            
            if file_format == '.csv':
                df = pd.read_csv(self.input_file)
            elif file_format in ['.xlsx', '.xls']:
                df = pd.read_excel(self.input_file)
            elif file_format == '.json':
                df = pd.read_json(self.input_file)
            elif file_format in ['.txt', '.dat']:
                df = pd.read_csv(self.input_file, sep='\s+', comment='#')
            elif file_format == '.inp':
                # Parse PSCMP input file
                df = self._parse_pscmp_inp_file(self.input_file)
            else:
                # Try CSV as default
                df = pd.read_csv(self.input_file)
            
            # Validate required columns
            required_cols = ['fault_id', 'o_lat', 'o_lon', 'o_depth', 'length', 'width', 'strike', 'dip']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Set default values for optional columns
            optional_defaults = {
                'np_st': 1, 'np_di': 1, 'start_time': 0.0,
                'slip_strike': 0.0, 'slip_downdip': 0.0, 'opening': 0.0,
                'pos_s': None, 'pos_d': None
            }
            
            for col, default_val in optional_defaults.items():
                if col not in df.columns:
                    if col == 'pos_d':
                        df[col] = df['width'] / 2.0
                    elif col == 'pos_s':
                        df[col] = df['length'] / 2.0
                    else:
                        df[col] = default_val

            # Standardize PSCMP data to patch-level format
            self.df = self._standardize_pscmp_data(df)
            
            # Setup projection
            self._setup_projection()
            
            # Populate standard slip_data format (following BaseSlipConverter interface)
            self.slip_data = {
                'longitude': self.df['o_lon'].values,
                'latitude': self.df['o_lat'].values,
                'depth': self.df['o_depth'].values,
                'length': self.df['patch_length'].values,
                'width': self.df['patch_width'].values,
                'strike': self.df['strike'].values,
                'dip': self.df['dip'].values,
                'strike_slip': self.df['slip_strike'].values,
                'dip_slip': -self.df['slip_downdip'].values,  # Convert to CSI convention
                'total_slip': np.sqrt(self.df['slip_strike'].values**2 + self.df['slip_downdip'].values**2),
                'pos_s': self.df['pos_s'].values,
                'pos_d': self.df['pos_d'].values
            }
            
            # Add opening if present
            if 'opening' in self.df.columns:
                self.slip_data['opening'] = self.df['opening'].values
            
            print(f"Loaded {len(self.df)} PSCMP fault patches")
            print(f"Slip range: {self.slip_data['total_slip'].min():.3f} - {self.slip_data['total_slip'].max():.3f} m")
            
            return True
            
        except Exception as e:
            print(f"Error reading PSCMP file: {e}")
            return False

    def _parse_pscmp_inp_file(self, inp_file: str) -> pd.DataFrame:
        """
        Parse PSCMP .inp input file format.
        
        PSCMP .inp file typically contains fault source definitions with format:
        # n_faults
        #-------------------------------------------------------------------------------
          1746
        #-------------------------------------------------------------------------------
        # n   O_lat   O_lon   O_depth length  width strike dip   np_st np_di start_time
        #     pos_s   pos_d   slp_stk slp_ddip open
        
        Parameters:
        -----------
        inp_file : str
            Path to PSCMP .inp file
            
        Returns:
        --------
        pd.DataFrame
            Parsed fault data
        """
        fault_data = []
        
        with open(inp_file, 'r') as f:
            lines = f.readlines()

        # Find n_faults value
        n_faults = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            # Look for n_faults comment line
            if 'n_faults' in line.lower() and line.startswith('#'):
                # Look for the number after the separator lines
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    
                    # Skip separator lines (lines with only # and -)
                    if next_line.startswith('#') and ('-' in next_line or '=' in next_line):
                        continue
                    
                    # Found a non-comment, non-separator line - should be n_faults
                    if next_line and not next_line.startswith('#'):
                        try:
                            n_faults = int(next_line)
                            print(f"Found n_faults = {n_faults}")
                            break
                        except ValueError:
                            continue
                break
        
        if n_faults is None:
            print("Warning: Could not find n_faults, using fallback parser")
            return self._parse_pscmp_inp_file_simple(inp_file)
        
        # Find the start of fault data section
        fault_section_start = None
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith('#') and len(line.split()) >= 10:
                parts = line.split()
                try:
                    fault_num = int(parts[0])
                    if fault_num == 1:  # Found start of fault definitions
                        fault_section_start = i
                        print(f"Found fault section starting at line {i+1}")
                        break
                except ValueError:
                    continue
        
        if fault_section_start is None:
            raise ValueError("Could not find fault data section in .inp file")
        
        # Parse fault data starting from fault_section_start
        i = fault_section_start
        fault_count = 0
        
        while i < len(lines) and fault_count < n_faults:
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
            
            try:
                # Parse first line: fault number and basic parameters
                parts = line.split()
                if len(parts) >= 10:  # PSCMP format: n O_lat O_lon O_depth length width strike dip np_st np_di start_time
                    fault_num = int(parts[0])
                    lat = float(parts[1])
                    lon = float(parts[2])
                    depth = float(parts[3])
                    length = float(parts[4])
                    width = float(parts[5])
                    strike = float(parts[6])
                    dip = float(parts[7])
                    np_st = int(parts[8])
                    np_di = int(parts[9])
                    start_time = float(parts[10]) if len(parts) > 10 else 0.0
                    
                    # Calculate patch dimensions
                    patch_length = length / np_st
                    patch_width = width / np_di
                    total_patches = np_st * np_di
                    
                    # Read slip data for all patches
                    patch_slip_data = []
                    for patch_idx in range(total_patches):
                        if i + 1 + patch_idx < len(lines):
                            slip_line = lines[i + 1 + patch_idx].strip()
                            if slip_line and not slip_line.startswith('#'):
                                try:
                                    slip_parts = slip_line.split()
                                    if len(slip_parts) >= 5:
                                        pos_s = float(slip_parts[0])
                                        pos_d = float(slip_parts[1])
                                        slip_strike = float(slip_parts[2])
                                        slip_downdip = float(slip_parts[3])
                                        opening = float(slip_parts[4])
                                        
                                        patch_slip_data.append({
                                            'pos_s': pos_s,
                                            'pos_d': pos_d,
                                            'slip_strike': slip_strike,
                                            'slip_downdip': slip_downdip,
                                            'opening': opening
                                        })
                                except (ValueError, IndexError):
                                    # Use default values if parsing fails
                                    patch_slip_data.append({
                                        'pos_s': patch_length / 2.0,
                                        'pos_d': patch_width / 2.0,
                                        'slip_strike': 0.0,
                                        'slip_downdip': 0.0,
                                        'opening': 0.0
                                    })
                    
                    # If we don't have enough slip data, fill with defaults
                    while len(patch_slip_data) < total_patches:
                        patch_slip_data.append({
                            'pos_s': patch_length / 2.0,
                            'pos_d': patch_width / 2.0,
                            'slip_strike': 0.0,
                            'slip_downdip': 0.0,
                            'opening': 0.0
                        })
                    
                    # Create records for each patch
                    for patch_idx, slip_data in enumerate(patch_slip_data):
                        fault_record = {
                            'fault_id': fault_num,
                            'patch_id': f"{fault_num}_{patch_idx + 1}",
                            'o_lat': lat,
                            'o_lon': lon,
                            'o_depth': depth,
                            'length': length,          # Original fault length
                            'width': width,            # Original fault width
                            'patch_length': patch_length,   # Individual patch length
                            'patch_width': patch_width,     # Individual patch width
                            'strike': strike,
                            'dip': dip,
                            'np_st': np_st,
                            'np_di': np_di,
                            'slip_strike': slip_data['slip_strike'],
                            'slip_downdip': slip_data['slip_downdip'],
                            'opening': slip_data['opening'],
                            'start_time': start_time,
                            'pos_s': slip_data['pos_s'],
                            'pos_d': slip_data['pos_d']
                        }
                        fault_data.append(fault_record)
                    
                    # Skip the slip lines we just processed
                    i += total_patches
                    fault_count += 1
                    
                    if fault_count % 100 == 0:
                        print(f"Parsed {fault_count}/{n_faults} faults...")
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line {i+1}: {line}")
                print(f"Error: {e}")
            
            i += 1
        
        if not fault_data:
            raise ValueError("No valid fault data found in .inp file")
        
        df = pd.DataFrame(fault_data)
        print(f"Successfully parsed {len(df)} fault patches from {inp_file}")
        print(f"Total faults: {fault_count}, Total patches: {len(df)}")
        
        return df

    def _parse_pscmp_inp_file_simple(self, inp_file: str) -> pd.DataFrame:
        """
        Simple fallback parser for PSCMP .inp files without n_faults header.
        
        Parameters:
        -----------
        inp_file : str
            Path to PSCMP .inp file
            
        Returns:
        --------
        pd.DataFrame
            Parsed fault data
        """
        fault_data = []
        fault_id = 1
        
        with open(inp_file, 'r') as f:
            lines = f.readlines()
        
        # Parse the file line by line
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#') or line.startswith('!'):
                i += 1
                continue
            
            try:
                # Split the line into components
                parts = line.split()
                
                if len(parts) >= 7:  # Minimum required parameters
                    # Parse basic fault parameters
                    lat = float(parts[0])
                    lon = float(parts[1])
                    depth = float(parts[2])
                    length = float(parts[3])
                    width = float(parts[4])
                    strike = float(parts[5])
                    dip = float(parts[6])
                    
                    # Parse optional parameters with defaults
                    np_st = int(parts[7]) if len(parts) > 7 else 1
                    np_di = int(parts[8]) if len(parts) > 8 else 1
                    slip_strike = float(parts[9]) if len(parts) > 9 else 0.0
                    slip_downdip = float(parts[10]) if len(parts) > 10 else 0.0
                    start_time = float(parts[11]) if len(parts) > 11 else 0.0
                    opening = float(parts[12]) if len(parts) > 12 else 0.0
                    
                    # Calculate patch dimensions
                    patch_length = length / np_st
                    patch_width = width / np_di
                    
                    # Create fault record
                    fault_record = {
                        'fault_id': fault_id,
                        'patch_id': f"{fault_id}_1",
                        'o_lat': lat,
                        'o_lon': lon,
                        'o_depth': depth,
                        'length': length,
                        'width': width,
                        'patch_length': patch_length,
                        'patch_width': patch_width,
                        'strike': strike,
                        'dip': dip,
                        'np_st': np_st,
                        'np_di': np_di,
                        'slip_strike': slip_strike,
                        'slip_downdip': slip_downdip,
                        'opening': opening,
                        'start_time': start_time,
                        'pos_s': patch_length / 2.0,  # Default center position
                        'pos_d': patch_width / 2.0
                    }
                    
                    fault_data.append(fault_record)
                    fault_id += 1
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line {i+1}: {line}")
                print(f"Error: {e}")
            
            i += 1
        
        if not fault_data:
            raise ValueError("No valid fault data found in .inp file")
        
        df = pd.DataFrame(fault_data)
        print(f"Successfully parsed {len(df)} fault sources from {inp_file}")
        
        return df

    def save_pscmp_inp_format(self, output_file: str):
        """
        Save fault data in PSCMP .inp format.
        
        Parameters:
        -----------
        output_file : str
            Output .inp filename
        """
        if self.df is None:
            raise ValueError("No data loaded. Call read_slip_file() first.")
        
        with open(output_file, 'w') as f:
            f.write("# PSCMP input file generated from fault data\n")
            f.write("# Format: lat lon depth(km) length(km) width(km) strike(deg) dip(deg) np_st np_di slip_strike(m) slip_downdip(m) start_time(s) opening(m)\n")
            f.write("#\n")
            
            # Group by fault_id to handle multi-patch faults
            for fault_id in self.df['fault_id'].unique():
                fault_patches = self.df[self.df['fault_id'] == fault_id]
                
                # For multi-patch faults, use the first patch as representative
                first_patch = fault_patches.iloc[0]
                
                f.write(f"{first_patch['o_lat']:10.6f} {first_patch['o_lon']:11.6f} ")
                f.write(f"{first_patch['o_depth']:8.3f} {first_patch['length']:8.3f} ")
                f.write(f"{first_patch['width']:8.3f} {first_patch['strike']:7.2f} ")
                f.write(f"{first_patch['dip']:6.2f} {first_patch['np_st']:3d} ")
                f.write(f"{first_patch['np_di']:3d} {first_patch['slip_strike']:8.3f} ")
                f.write(f"{first_patch['slip_downdip']:8.3f} {first_patch['start_time']:8.3f} ")
                f.write(f"{first_patch['opening']:8.3f}")
                f.write(f"  # Fault {fault_id}\n")
        
        print(f"PSCMP .inp format saved: {output_file}")
    
    def _setup_projection(self):
        """
        Setup UTM projection and fault geometry calculator.
        """
        if self.df is not None and len(self.df) > 0:
            lat_ref, lon_ref = self.df.iloc[0]['o_lat'], self.df.iloc[0]['o_lon']
            self.utm_zone = int((lon_ref + 180) // 6) + 1
            self.proj = Proj(proj='utm', zone=self.utm_zone, ellps='WGS84')
            self.fault_geometry = FaultGeometry(self.proj)
            
            print(f"Using UTM zone: {self.utm_zone}")
            print(f"Reference point: ({lat_ref:.4f}°, {lon_ref:.4f}°)")
    
    def _standardize_pscmp_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize PSCMP data format, expanding patches according to np_st and np_di.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw PSCMP data
            
        Returns:
        --------
        pd.DataFrame
            Standardized fault data with one row per patch
        """
        standardized_data = []
        
        for fault_id in df['fault_id'].unique():
            fault_rows = df[df['fault_id'] == fault_id]
            
            # Check consistency of np_st and np_di for this fault
            np_st_values = fault_rows['np_st'].unique()
            np_di_values = fault_rows['np_di'].unique()
            
            if len(np_st_values) > 1 or len(np_di_values) > 1:
                raise ValueError(f"Fault {fault_id} has inconsistent np_st or np_di values")
            
            np_st = int(np_st_values[0])
            np_di = int(np_di_values[0])
            
            # Process based on patch subdivision
            if np_st == 1 and np_di == 1:
                # Each row is an independent patch
                self._process_individual_patches(fault_rows, standardized_data)
            else:
                # Generate regular patch grid
                self._process_uniform_patches(fault_rows.iloc[0], np_st, np_di, standardized_data)
        
        return pd.DataFrame(standardized_data)
    
    def _process_individual_patches(self, fault_rows: pd.DataFrame, standardized_data: List[Dict]):
        """Process individual patches (np_st=1, np_di=1)."""
        patch_id = 1
        for _, row in fault_rows.iterrows():
            patch_data = {
                'fault_id': int(row['fault_id']),
                'patch_id': f"{int(row['fault_id'])}_{patch_id}",
                'o_lat': float(row['o_lat']),
                'o_lon': float(row['o_lon']),
                'o_depth': float(row['o_depth']),
                'length': float(row['length']),
                'width': float(row['width']),
                'patch_length': float(row['length']),  # For single patch, same as total
                'patch_width': float(row['width']),
                'strike': float(row['strike']),
                'dip': float(row['dip']),
                'pos_s': float(row['pos_s']),
                'pos_d': float(row['pos_d']),
                'slip_strike': float(row['slip_strike']),
                'slip_downdip': float(row['slip_downdip']),
                'opening': float(row['opening']),
                'start_time': float(row['start_time']),
                'np_st': int(row['np_st']),
                'np_di': int(row['np_di'])
            }
            standardized_data.append(patch_data)
            patch_id += 1
    
    def _process_uniform_patches(self, first_row: pd.Series, np_st: int, np_di: int, 
                               standardized_data: List[Dict]):
        """Process uniform patch grid (np_st>1 or np_di>1)."""
        patch_length = first_row['length'] / np_st
        patch_width = first_row['width'] / np_di
        
        patch_id = 1
        for i in range(np_st):
            for j in range(np_di):
                # Calculate patch center position (following PSCMP convention)
                pos_s = (i + 0.5) * patch_length
                pos_d = (j + 0.5) * patch_width
                
                patch_data = {
                    'fault_id': int(first_row['fault_id']),
                    'patch_id': f"{int(first_row['fault_id'])}_{patch_id}",
                    'o_lat': float(first_row['o_lat']),
                    'o_lon': float(first_row['o_lon']),
                    'o_depth': float(first_row['o_depth']),
                    'length': float(first_row['length']),  # Original fault length
                    'width': float(first_row['width']),   # Original fault width
                    'patch_length': patch_length,
                    'patch_width': patch_width,
                    'strike': float(first_row['strike']),
                    'dip': float(first_row['dip']),
                    'pos_s': pos_s,
                    'pos_d': pos_d,
                    'slip_strike': float(first_row['slip_strike']),
                    'slip_downdip': float(first_row['slip_downdip']),
                    'opening': float(first_row['opening']),
                    'start_time': float(first_row['start_time']),
                    'np_st': np_st,
                    'np_di': np_di
                }
                standardized_data.append(patch_data)
                patch_id += 1
        
    def write_to_format(self, output_file: str, **kwargs):
        """
        Write slip data back to PSCMP .inp format.
        
        Parameters:
        -----------
        output_file : str
            Output .inp file path
        **kwargs : dict
            PSCMP-specific options:
            - include_header : bool, default True
            - compact_format : bool, default False
        """
        if self.slip_data is None:
            raise ValueError("No slip data loaded. Call read_slip_file() first.")
        
        include_header = kwargs.get('include_header', True)
        compact_format = kwargs.get('compact_format', False)
        
        with open(output_file, 'w') as f:
            if include_header:
                f.write("#===============================================================================\n")
                f.write("# n_faults\n")
                f.write("#-------------------------------------------------------------------------------\n")
                f.write(f"  {len(self.slip_data['longitude'])}\n")
                f.write("#-------------------------------------------------------------------------------\n")
                f.write("# n   O_lat   O_lon   O_depth length  width strike dip   np_st np_di start_time\n")
                f.write("# [-] [deg]   [deg]   [km]    [km]     [km] [deg]  [deg] [-]   [-]   [day]\n")
                f.write("#     pos_s   pos_d   slp_stk slp_ddip open\n")
                f.write("#     [km]    [km]    [m]     [m]      [m]\n")
                f.write("#-------------------------------------------------------------------------------\n")
            
            # Choose format based on compact_format option
            if compact_format:
                pscmp_format = '{0:4d} {1:7.4f} {2:8.4f} {3:4.1f} {4:4.1f} {5:4.1f} {6:5.1f} {7:4.1f} {8:1d} {9:1d} {10:4.1f}\n    {11:7.4f} {12:7.4f} {13:6.3f} {14:6.3f} {15:6.3f}'
            else:
                pscmp_format = '{0:4d} {1:9.4f} {2:9.4f} {3:6.1f} {4:6.1f} {5:6.1f} {6:6.1f} {7:6.1f} {8:3d} {9:3d} {10:6.1f}\n    {11:9.4f} {12:9.4f} {13:9.3f} {14:9.3f} {15:9.3f}'
            
            for i in range(len(self.slip_data['longitude'])):
                # Convert dip_slip back to PSCMP convention (negative for reverse)
                slip_downdip = -self.slip_data['dip_slip'][i]
                
                # Get additional data from df if available
                if hasattr(self, 'df') and self.df is not None and i < len(self.df):
                    row = self.df.iloc[i]
                    fault_id = int(row.get('fault_id', i + 1))
                    np_st = int(row.get('np_st', 1))
                    np_di = int(row.get('np_di', 1))
                    start_time = float(row.get('start_time', 0.0))
                    pos_s = float(row.get('pos_s', self.slip_data['length'][i] / 2))
                    pos_d = float(row.get('pos_d', self.slip_data['width'][i] / 2))
                else:
                    fault_id = i + 1
                    np_st = np_di = 1
                    start_time = 0.0
                    pos_s = self.slip_data['length'][i] / 2
                    pos_d = self.slip_data['width'][i] / 2
                
                opening = self.slip_data.get('opening', [0.0] * len(self.slip_data['longitude']))[i]
                
                line = pscmp_format.format(
                    fault_id,
                    self.slip_data['latitude'][i],
                    self.slip_data['longitude'][i],
                    self.slip_data['depth'][i],
                    self.slip_data['length'][i],
                    self.slip_data['width'][i],
                    self.slip_data['strike'][i],
                    self.slip_data['dip'][i],
                    np_st, np_di, start_time,
                    pos_s, pos_d,
                    self.slip_data['strike_slip'][i],
                    slip_downdip,
                    opening
                )
                f.write(line + '\n')
        
        print(f"PSCMP format saved: {output_file}")


def main():
    """
    Main function to demonstrate the PSCMP converter usage.
    """
    converter = PSCMPConverter('fault_sources.csv')
    # 使用基类的完整转换流程
    converter.convert_all()


if __name__ == "__main__":
    main()