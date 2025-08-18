"""
PSCMP Configuration File Generator

This module provides classes and functions to generate PSCMP input files
for modeling post-seismic deformation using Green's functions.
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
import warnings


@dataclass
class FaultSource:
    """
    Single rectangular subfault definition for PSCMP calculation.
    
    Attributes:
    -----------
    fault_id : int
        Fault identification number
    o_lat : float
        Latitude of reference point in degrees
    o_lon : float
        Longitude of reference point in degrees
    o_depth : float
        Depth of reference point in km
    length : float
        Fault length along strike in km
    width : float
        Fault width along dip in km
    strike : float
        Strike angle in degrees
    dip : float
        Dip angle in degrees
    np_st : int
        Number of patches along strike
    np_di : int
        Number of patches along dip
    start_time : float
        Rupture start time in days
    patches : List[Dict]
        List of fault patches with slip parameters
    """
    fault_id: int
    o_lat: float
    o_lon: float
    o_depth: float
    length: float
    width: float
    strike: float
    dip: float
    np_st: int = 1
    np_di: int = 1
    start_time: float = 0.0
    patches: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate fault parameters."""
        if self.length <= 0 or self.width <= 0:
            raise ValueError("Fault length and width must be positive")
        if not (0 <= self.strike <= 360):
            raise ValueError("Strike must be between 0 and 360 degrees")
        if not (0 <= self.dip <= 90):
            raise ValueError("Dip must be between 0 and 90 degrees")
        if self.np_st < 1 or self.np_di < 1:
            raise ValueError("Number of patches must be >= 1")
        if self.o_depth < 0:
            raise ValueError("Depth must be non-negative")
    
    def add_patch(self, pos_s: float, pos_d: float, 
                  slip_strike: float, slip_downdip: float, opening: float = 0.0):
        """
        Add a fault patch with slip parameters.
        
        Parameters:
        -----------
        pos_s : float
            Position along strike in km
        pos_d : float
            Position along dip in km
        slip_strike : float
            Strike-slip component in m
        slip_downdip : float
            Dip-slip component in m
        opening : float
            Opening component in m
        """
        patch = {
            'pos_s': pos_s,
            'pos_d': pos_d,
            'slip_strike': slip_strike,
            'slip_downdip': slip_downdip,
            'opening': opening
        }
        self.patches.append(patch)
    
    def generate_uniform_patches(self, slip_strike: float = 0.0, 
                                slip_downdip: float = 0.0, opening: float = 0.0):
        """
        Generate uniform patch distribution with given slip values.
        
        Parameters:
        -----------
        slip_strike : float
            Uniform strike-slip in m
        slip_downdip : float
            Uniform dip-slip in m
        opening : float
            Uniform opening in m
        """
        self.patches.clear()
        
        # Ensure np_st and np_di are integers
        np_st = int(self.np_st)
        np_di = int(self.np_di)
        
        # Calculate patch positions
        for i in range(np_st):
            for j in range(np_di):
                pos_s = (i + 0.5) * self.length / np_st - self.length / 2.0
                pos_d = (j + 0.5) * self.width / np_di
                
                self.add_patch(pos_s, pos_d, slip_strike, slip_downdip, opening)


@dataclass
class PSCMPParameters:
    """
    Complete parameter set for PSCMP calculation.
    
    Attributes:
    -----------
    # Observation array configuration
    iposrec : int
        Position type: 0=irregular, 1=1D profile, 2=2D rectangular array
    observation_points : List or Dict
        Observation point coordinates or array parameters
    
    # Output configuration
    insar_los : bool
        Include InSAR line-of-sight displacement
    los_direction : Tuple[float, float, float]
        LOS direction cosines (x, y, z)
    coulomb_stress : bool
        Include Coulomb stress calculation
    coulomb_params : Dict
        Coulomb stress parameters
    output_dir : str
        Output directory
    
    # Time series outputs
    displacement_outputs : Tuple[bool, bool, bool]
        Enable displacement components (x, y, z)
    displacement_files : Tuple[str, str, str]
        Displacement output file names
    stress_outputs : Tuple[bool, bool, bool, bool, bool, bool]
        Enable stress components
    stress_files : Tuple[str, str, str, str, str, str]
        Stress output file names
    tilt_outputs : Tuple[bool, bool, bool, bool, bool]
        Enable tilt/geoid/gravity outputs
    tilt_files : Tuple[str, str, str, str, str]
        Tilt/geoid/gravity file names
    
    # Snapshot outputs
    snapshots : List[Dict]
        Snapshot time points and file names
    
    # Green's function database
    grn_dir : str
        Green's function directory
    grn_files : Tuple[str, ...]
        Green's function file names (13 components)
    
    # Fault sources
    fault_sources : List[FaultSource]
        List of fault source definitions
    """
    # Observation array
    iposrec: int = 2
    observation_points: Union[List, Dict] = field(default_factory=dict)
    
    # Output configuration
    insar_los: bool = True
    los_direction: Tuple[float, float, float] = (-0.072, 0.408, -0.910)
    coulomb_stress: bool = False
    coulomb_params: Dict = field(default_factory=dict)
    output_dir: str = './pscmpgrns/'

    # Time series outputs
    displacement_outputs: Tuple[bool, bool, bool] = (False, False, False)
    displacement_files: Tuple[str, str, str] = ('ux.dat', 'uy.dat', 'uz.dat')
    stress_outputs: Tuple[bool, bool, bool, bool, bool, bool] = (False, False, False, False, False, False)
    stress_files: Tuple[str, str, str, str, str, str] = ('sxx.dat', 'syy.dat', 'szz.dat', 'sxy.dat', 'syz.dat', 'szx.dat')
    tilt_outputs: Tuple[bool, bool, bool, bool, bool] = (False, False, False, False, False)
    tilt_files: Tuple[str, str, str, str, str] = ('tx.dat', 'ty.dat', 'rot.dat', 'gd.dat', 'gr.dat')
    
    # Snapshots
    snapshots: List[Dict] = field(default_factory=list)
    
    # Green's functions
    grn_dir: str = './psgrnfcts/'
    grn_files: Tuple[str, ...] = ('uz', 'ur', 'ut', 'szz', 'srr', 'stt', 'szr', 'srt', 'stz', 'tr', 'tt', 'rot', 'gd', 'gr')
    
    # Fault sources
    fault_sources: List[FaultSource] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default parameters."""
        if not self.observation_points:
            self.observation_points = {
                'nxrec': 61, 'xrec1': 0.00, 'xrec2': 15.00,
                'nyrec': 61, 'yrec1': 89.00, 'yrec2': 101.00
            }
        
        if not self.coulomb_params:
            self.coulomb_params = {
                'friction': 0.700, 'skempton': 0.000,
                'strike': 300.000, 'dip': 15.000, 'rake': 90.000,
                'sigma1': 1.0E+06, 'sigma2': -1.0E+06, 'sigma3': 0.0E+00
            }
        
        if not self.snapshots:
            self.snapshots = [
                {'time': 0.00, 'filename': 'snapshot_coseism.dat', 'comment': '0 co-seismic'},
                {'time': 30.42, 'filename': 'snapshot_1_month.dat', 'comment': '1 month'},
                {'time': 91.25, 'filename': 'snapshot_3_month.dat', 'comment': '3 months'},
                {'time': 182.50, 'filename': 'snapshot_6_month.dat', 'comment': '6 months'},
                {'time': 365.00, 'filename': 'snapshot_1_year.dat', 'comment': '1 year'},
                {'time': 730.00, 'filename': 'snapshot_2_year.dat', 'comment': '2 years'},
                {'time': 1825.00, 'filename': 'snapshot_5_year.dat', 'comment': '5 years'}
            ]

        # Auto modify the directory separators
        self.output_dir = self._fix_dir_sep(self.output_dir)
        self.grn_dir = self._fix_dir_sep(self.grn_dir)

        self._validate_parameters()
    
    @staticmethod
    def _fix_dir_sep(path: str) -> str:
        # Keep directory separators consistent
        sep = os.sep
        # Ensure the path ends with a separator
        path = path.replace('/', sep).replace('\\', sep)
        if not path.endswith(sep):
            path += sep
        return path

    def _validate_parameters(self):
        """Validate all parameters."""
        if self.iposrec not in [0, 1, 2]:
            raise ValueError("iposrec must be 0, 1, or 2")
        
        if len(self.los_direction) != 3:
            raise ValueError("LOS direction must have 3 components")
        
        if len(self.displacement_outputs) != 3:
            raise ValueError("Displacement outputs must have 3 components")
        
        if len(self.stress_outputs) != 6:
            raise ValueError("Stress outputs must have 6 components")
        
        if len(self.tilt_outputs) != 5:
            raise ValueError("Tilt outputs must have 5 components")
        
        if len(self.grn_files) != 14:
            raise ValueError("Green's function files must have 14 components")

    def load_observation_points_from_file(self, filename: Union[str, Path], 
                                         file_format: str = 'auto'):
        """
        Load observation points from file.
        
        Parameters:
        -----------
        filename : str or Path
            Input filename
        file_format : str
            File format: 'csv', 'excel', 'json', 'txt', or 'auto' (auto-detect)
        """
        filepath = Path(filename)
        
        if file_format == 'auto':
            file_format = filepath.suffix.lower()
            if file_format in ['.xlsx', '.xls']:
                file_format = 'excel'
            elif file_format == '.csv':
                file_format = 'csv'
            elif file_format == '.json':
                file_format = 'json'
            elif file_format in ['.txt', '.dat']:
                file_format = 'txt'
            else:
                raise ValueError(f"Cannot auto-detect format for {filepath}")
        
        if file_format == 'csv':
            df = pd.read_csv(filename)
        elif file_format == 'excel':
            df = pd.read_excel(filename)
        elif file_format == 'json':
            df = pd.read_json(filename)
        elif file_format == 'txt':
            df = pd.read_csv(filename, delim_whitespace=True, comment='#')
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        self.load_observation_points_from_dataframe(df)
    
    def load_observation_points_from_dataframe(self, df: pd.DataFrame):
        """
        Load observation points from pandas DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns: lat, lon (and optionally x, y for rectangular array)
        """
        # Check required columns
        if 'lat' in df.columns and 'lon' in df.columns:
            # Irregular observation points
            coordinates = [(row['lat'], row['lon']) for _, row in df.iterrows()]
            self.set_irregular_observation_points(coordinates)
        
        elif 'x' in df.columns and 'y' in df.columns:
            # Try to infer if it's a rectangular grid
            x_vals = sorted(df['x'].unique())
            y_vals = sorted(df['y'].unique())
            
            if len(x_vals) * len(y_vals) == len(df):
                # Rectangular grid
                self.set_rectangular_observation_array(
                    len(x_vals), min(x_vals), max(x_vals),
                    len(y_vals), min(y_vals), max(y_vals)
                )
            else:
                # Irregular points in Cartesian coordinates
                # Convert to lat/lon if possible or use as is
                coordinates = [(row['x'], row['y']) for _, row in df.iterrows()]
                self.observation_points = {
                    'nrec': len(coordinates),
                    'coordinates': coordinates
                }
                self.iposrec = 0
        else:
            raise ValueError("DataFrame must contain either 'lat'/'lon' or 'x'/'y' columns")
    
    def load_fault_sources_from_file(self, filename: Union[str, Path], 
                                    file_format: str = 'auto'):
        """
        Load fault sources from file.
        
        Parameters:
        -----------
        filename : str or Path
            Input filename
        file_format : str
            File format: 'csv', 'excel', 'json', 'txt', or 'auto'
        """
        filepath = Path(filename)
        
        if file_format == 'auto':
            file_format = filepath.suffix.lower()
            if file_format in ['.xlsx', '.xls']:
                file_format = 'excel'
            elif file_format == '.csv':
                file_format = 'csv'
            elif file_format == '.json':
                file_format = 'json'
            elif file_format in ['.txt', '.dat']:
                file_format = 'txt'
            else:
                raise ValueError(f"Cannot auto-detect format for {filepath}")
        
        if file_format == 'json':
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.load_fault_sources_from_dict(data)
        else:
            # Load as DataFrame first
            if file_format == 'csv':
                df = pd.read_csv(filename)
            elif file_format == 'excel':
                df = pd.read_excel(filename)
            elif file_format == 'txt':
                df = pd.read_csv(filename, delim_whitespace=True, comment='#')
            
            self.load_fault_sources_from_dataframe(df)
    
    def load_fault_sources_from_dataframe(self, df: pd.DataFrame):
        """
        Load fault sources from pandas DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with fault parameters
        """
        required_cols = ['fault_id', 'o_lat', 'o_lon', 'o_depth', 'length', 'width', 'strike', 'dip']
        optional_cols = ['np_st', 'np_di', 'start_time']
        patch_cols = ['pos_s', 'pos_d', 'slip_strike', 'slip_downdip', 'opening']
        
        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clear existing fault sources
        self.fault_sources.clear()
        
        # 处理每个唯一的断层ID
        for fault_id in df['fault_id'].unique():
            fault_rows = df[df['fault_id'] == fault_id]
            
            # 检查该断层的np_st和np_di是否一致
            np_st_values = fault_rows['np_st'].unique() if 'np_st' in df.columns else [1]
            np_di_values = fault_rows['np_di'].unique() if 'np_di' in df.columns else [1]
            
            if len(np_st_values) > 1 or len(np_di_values) > 1:
                raise ValueError(f"Fault {fault_id} has inconsistent np_st or np_di values")
            
            np_st = int(np_st_values[0])
            np_di = int(np_di_values[0])
            
            # 使用第一行的几何参数创建断层
            first_row = fault_rows.iloc[0]
            fault_kwargs = {
                'fault_id': int(first_row['fault_id']),
                'o_lat': float(first_row['o_lat']),
                'o_lon': float(first_row['o_lon']),
                'o_depth': float(first_row['o_depth']),
                'length': float(first_row['length']),
                'width': float(first_row['width']),
                'strike': float(first_row['strike']),
                'dip': float(first_row['dip']),
                'np_st': np_st,
                'np_di': np_di,
                'start_time': float(first_row.get('start_time', 0.0))
            }
            
            fault = FaultSource(**fault_kwargs)
            
            # 根据np_st和np_di的值决定如何处理patches
            if np_st == 1 and np_di == 1:
                # 每行都是一个独立的patch
                if all(col in df.columns for col in patch_cols):
                    # 有详细patch信息
                    for _, patch_row in fault_rows.iterrows():
                        if all(pd.notna(patch_row[col]) for col in patch_cols):
                            fault.add_patch(
                                float(patch_row['pos_s']), 
                                float(patch_row['pos_d']),
                                float(patch_row['slip_strike']), 
                                float(patch_row['slip_downdip']),
                                float(patch_row.get('opening', 0.0))
                            )
                else:
                    # 没有详细patch信息，每行用滑移信息生成单个patch
                    for _, patch_row in fault_rows.iterrows():
                        slip_strike = float(patch_row.get('slip_strike', 0.0))
                        slip_downdip = float(patch_row.get('slip_downdip', 0.0))
                        opening = float(patch_row.get('opening', 0.0))
                        # 对于单个patch，位置在断层中心
                        fault.add_patch(0.0, 0.0, slip_strike, slip_downdip, opening)
            else:
                # np_st>1 或 np_di>1，生成规则的patch网格
                # 使用第一行的滑移参数
                slip_strike = float(first_row.get('slip_strike', 0.0))
                slip_downdip = float(first_row.get('slip_downdip', 0.0))
                opening = float(first_row.get('opening', 0.0))
                fault.generate_uniform_patches(slip_strike, slip_downdip, opening)
            
            self.add_fault_source(fault)
    
    def load_fault_sources_from_dict(self, data: Union[Dict, List[Dict]]):
        """
        Load fault sources from dictionary or list of dictionaries.
        
        Parameters:
        -----------
        data : Dict or List[Dict]
            Fault source data
        """
        if isinstance(data, dict):
            if 'faults' in data:
                fault_list = data['faults']
            else:
                fault_list = [data]
        else:
            fault_list = data
        
        self.fault_sources.clear()
        
        for fault_dict in fault_list:
            # Extract fault parameters
            fault_params = {k: v for k, v in fault_dict.items() 
                          if k not in ['patches']}
            
            fault = FaultSource(**fault_params)
            
            # Add patches if provided
            if 'patches' in fault_dict:
                for patch in fault_dict['patches']:
                    fault.add_patch(**patch)
            
            self.add_fault_source(fault)
    
    def save_fault_sources_to_file(self, filename: Union[str, Path], 
                                  file_format: str = 'auto', include_patches: bool = True):
        """
        Save fault sources to file.
        
        Parameters:
        -----------
        filename : str or Path
            Output filename
        file_format : str
            File format: 'csv', 'excel', 'json', or 'auto'
        include_patches : bool
            Whether to include patch details
        """
        filepath = Path(filename)
        
        if file_format == 'auto':
            file_format = filepath.suffix.lower()
            if file_format in ['.xlsx', '.xls']:
                file_format = 'excel'
            elif file_format == '.csv':
                file_format = 'csv'
            elif file_format == '.json':
                file_format = 'json'
            else:
                file_format = 'csv'
        
        if file_format == 'json':
            fault_data = []
            for fault in self.fault_sources:
                fault_dict = {
                    'fault_id': fault.fault_id,
                    'o_lat': fault.o_lat,
                    'o_lon': fault.o_lon,
                    'o_depth': fault.o_depth,
                    'length': fault.length,
                    'width': fault.width,
                    'strike': fault.strike,
                    'dip': fault.dip,
                    'np_st': fault.np_st,
                    'np_di': fault.np_di,
                    'start_time': fault.start_time
                }
                
                if include_patches:
                    fault_dict['patches'] = fault.patches
                
                fault_data.append(fault_dict)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({'faults': fault_data}, f, indent=2)
        
        else:
            df = self.fault_sources_to_dataframe(include_patches)
            
            if file_format == 'csv':
                df.to_csv(filename, index=False)
            elif file_format == 'excel':
                df.to_excel(filename, index=False)
    
    def fault_sources_to_dataframe(self, include_patches: bool = True) -> pd.DataFrame:
        """
        Convert fault sources to pandas DataFrame.
        
        Parameters:
        -----------
        include_patches : bool
            Whether to include patch details
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with fault source data
        """
        data = []
        
        if include_patches:
            # Include patch details (one row per patch)
            for fault in self.fault_sources:
                for patch in fault.patches:
                    row = {
                        'fault_id': fault.fault_id,
                        'o_lat': fault.o_lat,
                        'o_lon': fault.o_lon,
                        'o_depth': fault.o_depth,
                        'length': fault.length,
                        'width': fault.width,
                        'strike': fault.strike,
                        'dip': fault.dip,
                        'np_st': fault.np_st,
                        'np_di': fault.np_di,
                        'start_time': fault.start_time,
                        'pos_s': patch['pos_s'],
                        'pos_d': patch['pos_d'],
                        'slip_strike': patch['slip_strike'],
                        'slip_downdip': patch['slip_downdip'],
                        'opening': patch['opening']
                    }
                    data.append(row)
        else:
            # Only fault geometry (one row per fault)
            for fault in self.fault_sources:
                row = {
                    'fault_id': fault.fault_id,
                    'o_lat': fault.o_lat,
                    'o_lon': fault.o_lon,
                    'o_depth': fault.o_depth,
                    'length': fault.length,
                    'width': fault.width,
                    'strike': fault.strike,
                    'dip': fault.dip,
                    'np_st': fault.np_st,
                    'np_di': fault.np_di,
                    'start_time': fault.start_time
                }
                data.append(row)
        
        return pd.DataFrame(data)    

    def add_fault_source(self, fault: FaultSource):
        """Add a fault source to the configuration."""
        self.fault_sources.append(fault)
    
    def add_snapshot(self, time: float, filename: str, comment: str = ""):
        """
        Add a snapshot output.
        
        Parameters:
        -----------
        time : float
            Snapshot time in days
        filename : str
            Output filename
        comment : str
            Optional comment
        """
        snapshot = {'time': time, 'filename': filename, 'comment': comment}
        self.snapshots.append(snapshot)
    
    def set_rectangular_observation_array(self, nxrec: int, xrec1: float, xrec2: float,
                                         nyrec: int, yrec1: float, yrec2: float):
        """
        Set rectangular 2D observation array.
        
        Parameters:
        -----------
        nxrec : int
            Number of x samples
        xrec1, xrec2 : float
            Start and end x values
        nyrec : int
            Number of y samples
        yrec1, yrec2 : float
            Start and end y values
        """
        self.iposrec = 2
        self.observation_points = {
            'nxrec': nxrec, 'xrec1': xrec1, 'xrec2': xrec2,
            'nyrec': nyrec, 'yrec1': yrec1, 'yrec2': yrec2
        }
    
    def set_profile_observation_array(self, nrec: int, lat1: float, lon1: float,
                                     lat2: float, lon2: float):
        """
        Set 1D profile observation array.
        
        Parameters:
        -----------
        nrec : int
            Number of profile points
        lat1, lon1 : float
            Start coordinates
        lat2, lon2 : float
            End coordinates
        """
        self.iposrec = 1
        self.observation_points = {
            'nrec': nrec,
            'lat1': lat1, 'lon1': lon1,
            'lat2': lat2, 'lon2': lon2
        }
    
    def set_irregular_observation_points(self, coordinates: List[Tuple[float, float]]):
        """
        Set irregular observation points.
        
        Parameters:
        -----------
        coordinates : List[Tuple[float, float]]
            List of (lat, lon) coordinates
        """
        self.iposrec = 0
        self.observation_points = {
            'nrec': len(coordinates),
            'coordinates': coordinates
        }


class PSCMPConfig:
    """
    PSCMP configuration file generator.
    
    This class creates properly formatted PSCMP input files with complete
    header comments and parameter validation.
    """
    
    def __init__(self, parameters: Optional[PSCMPParameters] = None):
        """
        Initialize PSCMP configuration generator.
        
        Parameters:
        -----------
        parameters : PSCMPParameters, optional
            Configuration parameters. If None, uses defaults.
        """
        self.parameters = parameters or PSCMPParameters()

    @classmethod
    def from_template(cls, template_name: str = 'simple'):
        """
        Create configuration from predefined template.
        
        Parameters:
        -----------
        template_name : str
            Template name: 'simple', 'insar', 'coulomb', 'detailed'
        """
        config = cls()
        
        if template_name == 'simple':
            # Simple displacement outputs only
            config.parameters.displacement_outputs = (True, True, True)
            
        elif template_name == 'insar':
            # InSAR-focused configuration
            config.parameters.insar_los = True
            config.parameters.displacement_outputs = (True, True, True)
            
        elif template_name == 'coulomb':
            # Coulomb stress analysis
            config.parameters.coulomb_stress = True
            config.parameters.stress_outputs = (True, True, True, True, True, True)
            
        elif template_name == 'detailed':
            # All outputs enabled
            config.parameters.displacement_outputs = (True, True, True)
            config.parameters.stress_outputs = (True, True, True, True, True, True)
            config.parameters.tilt_outputs = (True, True, True, True, True)
            config.parameters.insar_los = True
            config.parameters.coulomb_stress = True
            
        return config
    
    @classmethod
    def from_files(cls, observation_file: Union[str, Path] = None,
                   fault_file: Union[str, Path] = None, **kwargs):
        """
        Create configuration from data files.
        
        Parameters:
        -----------
        observation_file : str or Path, optional
            Observation points file
        fault_file : str or Path, optional
            Fault sources file
        **kwargs : dict
            Additional parameter updates
        """
        config = cls()
        
        if observation_file:
            config.parameters.load_observation_points_from_file(observation_file)
        
        if fault_file:
            config.parameters.load_fault_sources_from_file(fault_file)
        
        config.update_parameters(**kwargs)
        return config
    
    def save_template(self, filename: Union[str, Path]):
        """Save current configuration as template."""
        template_data = {
            'parameters': {
                'iposrec': self.parameters.iposrec,
                'insar_los': self.parameters.insar_los,
                'los_direction': self.parameters.los_direction,
                'coulomb_stress': self.parameters.coulomb_stress,
                'coulomb_params': self.parameters.coulomb_params,
                'output_dir': self.parameters._fix_dir_sep(self.parameters.output_dir),
                'displacement_outputs': self.parameters.displacement_outputs,
                'stress_outputs': self.parameters.stress_outputs,
                'tilt_outputs': self.parameters.tilt_outputs,
                'grn_dir': self.parameters._fix_dir_sep(self.parameters.grn_dir)
            },
            'observation_points': self.parameters.observation_points,
            'snapshots': self.parameters.snapshots,
            'fault_sources': [
                {
                    'fault_id': f.fault_id, 'o_lat': f.o_lat, 'o_lon': f.o_lon,
                    'o_depth': f.o_depth, 'length': f.length, 'width': f.width,
                    'strike': f.strike, 'dip': f.dip, 'np_st': f.np_st,
                    'np_di': f.np_di, 'start_time': f.start_time, 'patches': f.patches
                }
                for f in self.parameters.fault_sources
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2)

    def update_parameters(self, **kwargs):
        """
        Update configuration parameters.
        
        Parameters:
        -----------
        **kwargs : dict
            Parameter updates
        """
        for key, value in kwargs.items():
            if hasattr(self.parameters, key):
                setattr(self.parameters, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    def _fix_dir_sep(self, path: str) -> str:
        """
        Fix directory separator for the current operating system.
        """
        sep = os.sep
        path = path.replace('/', sep).replace('\\', sep)
        if not path.endswith(sep):
            path += sep
        return path

    def generate_config_string(self) -> str:
        """
        Generate complete PSCMP configuration file content.
        
        Returns:
        --------
        str
            Complete configuration file content
        """
        # Always fix output_dir and grn_dir before output
        self.parameters.output_dir = self._fix_dir_sep(self.parameters.output_dir)
        if hasattr(self.parameters, 'grn_dir'):
            self.parameters.grn_dir = self._fix_dir_sep(self.parameters.grn_dir)
        lines = []
        
        # Add header comments
        lines.extend(self._generate_header())
        
        # Add configuration sections
        lines.extend(self._generate_observation_config())
        lines.extend(self._generate_output_config())
        lines.extend(self._generate_green_function_config())
        lines.extend(self._generate_fault_config())
        
        # Add end marker
        lines.append("#================================end of input===================================")
        
        return '\n'.join(lines)
    
    def write_config_file(self, filename: Union[str, Path] = "pscmp_input.dat", verbose=True):
        """
        Write PSCMP configuration file.
        
        Parameters:
        -----------
        filename : str or Path
            Output filename
        """
        config_content = self.generate_config_string()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        if verbose:
            print(f"PSCMP configuration written to: {filename}")
    
    def _generate_header(self) -> List[str]:
        """Generate file header with comments."""
        return [
            "#===============================================================================",
            "# This is input file of FORTRAN77 program \"pscmp08\" for modeling post-seismic",
            "# deformation induced by earthquakes in multi-layered viscoelastic media using",
            "# the Green's function approach. The earthquke source is represented by an",
            "# arbitrary number of rectangular dislocation planes. For more details, please",
            "# read the accompanying READ.ME file.",
            "#",
            "# written by Rongjiang Wang",
            "# GeoForschungsZentrum Potsdam",
            "# e-mail: wang@gfz-potsdam.de",
            "# phone +49 331 2881209",
            "# fax +49 331 2881204",
            "#",
            "# Last modified: Potsdam, July, 2008",
            "#",
            "# References:",
            "#",
            "# (1) Wang, R., F. Lorenzo-Martín and F. Roth (2003), Computation of deformation",
            "#     induced by earthquakes in a multi-layered elastic crust - FORTRAN programs",
            "#     EDGRN/EDCMP, Computer and Geosciences, 29(2), 195-207.",
            "# (2) Wang, R., F. Lorenzo-Martin and F. Roth (2006), PSGRN/PSCMP - a new code for",
            "#     calculating co- and post-seismic deformation, geoid and gravity changes",
            "#     based on the viscoelastic-gravitational dislocation theory, Computers and",
            "#     Geosciences, 32, 527-541. DOI:10.1016/j.cageo.2005.08.006.",
            "# (3) Wang, R. (2005), The dislocation theory: a consistent way for including the",
            "#     gravity effect in (visco)elastic plane-earth models, Geophysical Journal",
            "#     International, 161, 191-196.",
            "#",
            "#################################################################",
            "##                                                             ##",
            "## Green's functions should have been prepared with the        ##",
            "## program \"psgrn08\" before the program \"pscmp08\" is started.  ##",
            "##                                                             ##",
            "## For local Cartesian coordinate system, the Aki's convention ##",
            "## is used, that is, x is northward, y is eastward, and z is   ##",
            "## downward.                                                   ##",
            "##                                                             ##",
            "## If not specified otherwise, SI Unit System is used overall! ##",
            "##                                                             ##",
            "#################################################################"
        ]
    
    def _generate_observation_config(self) -> List[str]:
        """Generate observation array configuration section."""
        p = self.parameters
        
        lines = [
            "#===============================================================================",
            "# OBSERVATION ARRAY",
            "# =================",
            "# 1. selection for irregular observation positions (= 0) or a 1D observation",
            "#    profile (= 1) or a rectangular 2D observation array (= 2): iposrec",
            "#",
            "#    IF (iposrec = 0 for irregular observation positions) THEN",
            "#",
            "# 2. number of positions: nrec",
            "#",
            "# 3. coordinates of the observations: (lat(i),lon(i)), i=1,nrec",
            "#",
            "#    ELSE IF (iposrec = 1 for regular 1D observation array) THEN",
            "#",
            "# 2. number of position samples of the profile: nrec",
            "#",
            "# 3. the start and end positions: (lat1,lon1), (lat2,lon2)",
            "#",
            "#    ELSE IF (iposrec = 2 for rectanglular 2D observation array) THEN",
            "#",
            "# 2. number of x samples, start and end values: nxrec, xrec1, xrec2",
            "#",
            "# 3. number of y samples, start and end values: nyrec, yrec1, yrec2",
            "#",
            "#    sequence of the positions in output data: lat(1),lon(1); ...; lat(nx),lon(1);",
            "#    lat(1),lon(2); ...; lat(nx),lon(2); ...; lat(1),lon(ny); ...; lat(nx),lon(ny).",
            "#",
            "#    Note that the total number of observation positions (nrec or nxrec*nyrec)",
            "#    should be <= NRECMAX (see pecglob.h)!",
            "#==============================================================================="
        ]
        
        # Add observation configuration based on type
        if p.iposrec == 0:  # Irregular points
            obs = p.observation_points
            lines.append(f" {p.iposrec}")
            lines.append(f" {obs['nrec']}")
            for i in range(0, len(obs['coordinates']), 3):
                coord_line = "   "
                for j in range(min(3, len(obs['coordinates']) - i)):
                    lat, lon = obs['coordinates'][i + j]
                    coord_line += f"({lat:.5f},{lon:.5f}), "
                lines.append(coord_line)
            lines[-1] = lines[-1].rstrip(' ').rstrip(',')
        
        elif p.iposrec == 1:  # 1D profile
            obs = p.observation_points
            lines.extend([
                f" {p.iposrec}",
                f" {obs['nrec']}",
                f" ({obs['lat1']:9.5f}, {obs['lon1']:9.5f}), ({obs['lat2']:9.5f}, {obs['lon2']:9.5f})"
            ])
        
        elif p.iposrec == 2:  # 2D rectangular array
            obs = p.observation_points
            lines.extend([
                f" {p.iposrec}",
                f" {obs['nxrec']:2d}     {obs['xrec1']:9.5f}  {obs['xrec2']:9.5f}",
                f" {obs['nyrec']:2d}    {obs['yrec1']:9.5f} {obs['yrec2']:9.5f}"
            ])
        
        return lines
    
    def _generate_output_config(self) -> List[str]:
        """Generate output configuration section."""
        p = self.parameters
        
        lines = [
            "#===============================================================================",
            "# OUTPUTS",
            "# =======",
            "#",
            "# 1. select output for los displacement (only for snapshots, see below), x, y,",
            "#    and z-cosines to the INSAR orbit: insar (1/0 = yes/no), xlos, ylos, zlos",
            "#",
            "#    if this option is selected (insar = 1), the snapshots will include additional",
            "#    data:",
            "#    LOS_Dsp = los displacement to the given satellite orbit.",
            "#",
            "# 2. select output for Coulomb stress changes (only for snapshots, see below):",
            "#    icmb (1/0 = yes/no), friction, Skempton ratio, strike, dip, and rake angles",
            "#    [deg] describing the uniform regional master fault mechanism, the uniform",
            "#    regional principal stresses: sigma1, sigma2 and sigma3 [Pa] in arbitrary",
            "#    order (the orietation of the pre-stress field will be derived by assuming",
            "#    that the master fault is optimally oriented according to Coulomb failure",
            "#    criterion)",
            "#",
            "#    if this option is selected (icmb = 1), the snapshots will include additional",
            "#    data:",
            "#    CMB_Fix, Sig_Fix = Coulomb and normal stress changes on master fault;",
            "#    CMB_Op1/2, Sig_Op1/2 = Coulomb and normal stress changes on the two optimally",
            "#                       oriented faults;",
            "#    Str_Op1/2, Dip_Op1/2, Slp_Op1/2 = strike, dip and rake angles of the two",
            "#                       optimally oriented faults.",
            "#",
            "#    Note: the 1. optimally orieted fault is the one closest to the master fault.",
            "#",
            "# 3. output directory in char format: outdir",
            "#",
            "# 4. select outputs for displacement components (1/0 = yes/no): itout(i), i=1-3",
            "#",
            "# 5. the file names in char format for the x, y, and z components:",
            "#    toutfile(i), i=1-3",
            "#",
            "# 6. select outputs for stress components (1/0 = yes/no): itout(i), i=4-9",
            "#",
            "# 7. the file names in char format for the xx, yy, zz, xy, yz, and zx components:",
            "#    toutfile(i), i=4-9",
            "#",
            "# 8. select outputs for vertical NS and EW tilt components, block rotation, geoid",
            "#    and gravity changes (1/0 = yes/no): itout(i), i=10-14",
            "#",
            "# 9. the file names in char format for the NS tilt (positive if borehole top",
            "#    tilts to north), EW tilt (positive if borehole top tilts to east), block",
            "#    rotation (clockwise positive), geoid and gravity changes: toutfile(i), i=10-14",
            "#",
            "#    Note that all above outputs are time series with the time window as same",
            "#    as used for the Green's functions",
            "#",
            "#10. number of scenario outputs (\"snapshots\": spatial distribution of all above",
            "#    observables at given time points; <= NSCENMAX (see pscglob.h): nsc",
            "#",
            "#11. the time [day], and file name (in char format) for the 1. snapshot;",
            "#12. the time [day], and file name (in char format) for the 2. snapshot;",
            "#13. ...",
            "#",
            "#    Note that all file or directory names should not be longer than 80",
            "#    characters. Directories must be ended by / (unix) or \\ (dos)!",
            "#==============================================================================="
        ]
        
        # InSAR LOS configuration
        insar_flag = 1 if p.insar_los else 0
        lines.append(f" {insar_flag}    {p.los_direction[0]:6.3f}  {p.los_direction[1]:6.3f}  {p.los_direction[2]:6.3f}")
        
        # Coulomb stress configuration
        coulomb_flag = 1 if p.coulomb_stress else 0
        if p.coulomb_stress:
            cp = p.coulomb_params
            lines.append(f" {coulomb_flag}     {cp['friction']:5.3f}  {cp['skempton']:5.3f}  {cp['strike']:7.3f}   {cp['dip']:6.3f}  {cp['rake']:6.3f}    {cp['sigma1']:8.1E}   {cp['sigma2']:8.1E}    {cp['sigma3']:8.1E}")
        else:
            lines.append(f" {coulomb_flag}     0.700  0.000  300.000   15.000  90.000    1.0E+06   -1.0E+06    0.0E+00")
        
        # Output directory
        lines.append(f" '{self.parameters._fix_dir_sep(p.output_dir)}'")

        # Displacement outputs
        disp_flags = "  ".join([f"{int(flag):10d}" for flag in p.displacement_outputs])
        lines.append(f"  {disp_flags}")
        disp_files = "    ".join([f"'{file}'" for file in p.displacement_files])
        lines.append(f"  {disp_files}")
        
        # Stress outputs
        stress_flags = "  ".join([f"{int(flag):10d}" for flag in p.stress_outputs])
        lines.append(f"  {stress_flags}")
        stress_files = "   ".join([f"'{file}'" for file in p.stress_files])
        lines.append(f"  {stress_files}")
        
        # Tilt/geoid/gravity outputs
        tilt_flags = "  ".join([f"{int(flag):10d}" for flag in p.tilt_outputs])
        lines.append(f"  {tilt_flags}")
        tilt_files = "    ".join([f"'{file}'" for file in p.tilt_files])
        lines.append(f"  {tilt_files}")
        
        # Snapshots
        lines.append(f"  {len(p.snapshots)}")
        for snapshot in p.snapshots:
            comment = f"      |{snapshot['comment']}" if snapshot['comment'] else ""
            lines.append(f"     {snapshot['time']:5.2f}  '{snapshot['filename']}'{comment}")
        
        return lines
    
    def _generate_green_function_config(self) -> List[str]:
        """Generate Green's function database configuration section."""
        p = self.parameters
        
        lines = [
            "#===============================================================================",
            "#",
            "# GREEN'S FUNCTION DATABASE",
            "# =========================",
            "# 1. directory where the Green's functions are stored: grndir",
            "#",
            "# 2. file names (without extensions!) for the 13 Green's functions:",
            "#    3 displacement komponents (uz, ur, ut): green(i), i=1-3",
            "#    6 stress components (szz, srr, stt, szr, srt, stz): green(i), i=4-9",
            "#    radial and tangential components measured by a borehole tiltmeter,",
            "#    rigid rotation around z-axis, geoid and gravity changes (tr, tt, rot, gd, gr):",
            "#    green(i), i=10-14",
            "#",
            "#    Note that all file or directory names should not be longer than 80",
            "#    characters. Directories must be ended by / (unix) or \\ (dos)! The",
            "#    extensions of the file names will be automatically considered. They",
            "#    are \".ep\", \".ss\", \".ds\" and \".cl\" denoting the explosion (inflation)",
            "#    strike-slip, the dip-slip and the compensated linear vector dipole",
            "#    sources, respectively.",
            "#",
            "#===============================================================================",
            f" '{self.parameters._fix_dir_sep(p.grn_dir)}'",
            f" '{p.grn_files[0]}'  '{p.grn_files[1]}'  '{p.grn_files[2]}'",
            f" '{p.grn_files[3]}' '{p.grn_files[4]}' '{p.grn_files[5]}' '{p.grn_files[6]}' '{p.grn_files[7]}' '{p.grn_files[8]}'",
            f" '{p.grn_files[9]}'  '{p.grn_files[10]}'  '{p.grn_files[11]}' '{p.grn_files[12]}'  '{p.grn_files[13]}'"
        ]
        
        return lines
    
    def _generate_fault_config(self) -> List[str]:
        """Generate fault source configuration section."""
        p = self.parameters
        
        lines = [
            "#===============================================================================",
            "# RECTANGULAR SUBFAULTS",
            "# =====================",
            "# 1. number of subfaults (<= NSMAX in pscglob.h): ns",
            "#",
            "# 2. parameters for the 1. rectangular subfault: geographic coordinates",
            "#    (O_lat, O_lon) [deg] and O_depth [km] of the local reference point on",
            "#    the present fault plane, length (along strike) [km] and width (along down",
            "#    dip) [km], strike [deg], dip [deg], number of equi-size fault patches along",
            "#    the strike (np_st) and along the dip (np_di) (total number of fault patches",
            "#    = np_st x np_di), and the start time of the rupture; the following data",
            "#    lines describe the slip distribution on the present sub-fault:",
            "#",
            "#    pos_s[km]  pos_d[km]  slip_strike[m]  slip_downdip[m]  opening[m]",
            "#",
            "#    where (pos_s,pos_d) defines the position of the center of each patch in",
            "#    the local coordinate system with the origin at the reference point:",
            "#    pos_s = distance along the length (positive in the strike direction)",
            "#    pos_d = distance along the width (positive in the down-dip direction)",
            "#",
            "#",
            "# 3. ... for the 2. subfault ...",
            "# ...",
            "#                   N",
            "#                  /",
            "#                 /| strike",
            "#                +------------------------",
            "#                |\\        p .            \\ W",
            "#                :-\\      i .              \\ i",
            "#                |  \\    l .                \\ d",
            "#                :90 \\  S .                  \\ t",
            "#                |-dip\\  .                    \\ h",
            "#                :     \\. | rake               \\",
            "#                Z      -------------------------",
            "#                              L e n g t h",
            "#",
            "#    Simulation of a Mogi source:",
            "#    (1) Calculate deformation caused by three small openning plates (each",
            "#        causes a third part of the volume of the point inflation) located",
            "#        at the same depth as the Mogi source but oriented orthogonal to",
            "#        each other.",
            "#    (2) Multiply the results by 3(1-nu)/(1+nu), where nu is the Poisson",
            "#        ratio at the source depth.",
            "#    The multiplication factor is the ratio of the seismic moment (energy) of",
            "#    the Mogi source to that of the plate openning with the same volume change.",
            "#===============================================================================",
            "# n_faults",
            "#-------------------------------------------------------------------------------",
            f"  {len(p.fault_sources)}",
            "#-------------------------------------------------------------------------------",
            "# n   O_lat   O_lon   O_depth length  width strike dip   np_st np_di start_time",
            "# [-] [deg]   [deg]   [km]    [km]     [km] [deg]  [deg] [-]   [-]   [day]",
            "#     pos_s   pos_d   slp_stk slp_ddip open",
            "#     [km]    [km]    [m]     [m]      [m]",
            "#-------------------------------------------------------------------------------"
        ]
        
        # Add fault definitions
        for fault in p.fault_sources:
            # Fault header line - ensure all values are properly typed
            fault_id = int(fault.fault_id)
            np_st = int(fault.np_st)
            np_di = int(fault.np_di)
            
            lines.append(
                f"  {fault_id:4d}    {fault.o_lat:7.4f}   {fault.o_lon:8.4f}    "
                f"{fault.o_depth:7.4f}   {fault.length:7.4f}   {fault.width:7.4f}  "
                f"{fault.strike:6.2f}   {fault.dip:5.2f}   {np_st:3d}   {np_di:3d}    "
                f"{fault.start_time:8.5f}"
            )
            
            # Patch definitions
            for patch in fault.patches:
                lines.append(
                    f"      {patch['pos_s']:7.4f}   {patch['pos_d']:7.4f}     "
                    f"{patch['slip_strike']:6.3f}    {patch['slip_downdip']:6.3f}     "
                    f"{patch['opening']:6.3f}"
                )
        
        return lines


# Utility functions for creating templates
def create_observation_template_csv(filename: str = "observation_points.csv", 
                                   array_type: str = 'rectangular'):
    """
    Create CSV template for observation points.
    
    Parameters:
    -----------
    filename : str
        Output CSV filename
    array_type : str
        'rectangular', 'profile', or 'irregular'
    """
    if array_type == 'rectangular':
        # Create a sample rectangular grid
        x = np.linspace(0, 15, 16)
        y = np.linspace(89, 101, 13)
        X, Y = np.meshgrid(x, y)
        
        df = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten()
        })
        
    elif array_type == 'profile':
        # Create a sample profile
        n_points = 50
        lat = np.linspace(1.5, 2.0, n_points)
        lon = np.linspace(95.0, 96.0, n_points)
        
        df = pd.DataFrame({
            'lat': lat,
            'lon': lon
        })
        
    elif array_type == 'irregular':
        # Create sample irregular points
        np.random.seed(42)
        n_points = 30
        lat = 1.5 + 0.5 * np.random.random(n_points)
        lon = 95.0 + 1.0 * np.random.random(n_points)
        
        df = pd.DataFrame({
            'lat': lat,
            'lon': lon
        })
    
    df.to_csv(filename, index=False)
    print(f"Observation points template saved to: {filename}")


def create_fault_template_csv(filename: str = "fault_sources.csv", 
                             template_type: str = 'mixed'):
    """
    Create CSV template for fault sources.
    
    Parameters:
    -----------
    filename : str
        Output CSV filename
    template_type : str
        'single_patches' - 每个patch单独设置 (np_st=1, np_di=1)
        'uniform_segments' - 断层段统一设置 (np_st>1 或 np_di>1)
        'mixed' - 混合模式示例
    """
    if template_type == 'single_patches':
        # 单个patch模式：np_st=1, np_di=1，每行一个patch
        data = {
            'fault_id': [1, 1, 1, 1, 2, 2],
            'o_lat': [1.7987, 1.7987, 1.7987, 1.7987, 2.0000, 2.0000],
            'o_lon': [95.4805, 95.4805, 95.4805, 95.4805, 95.5000, 95.5000],
            'o_depth': [0.0, 0.0, 0.0, 0.0, 5.0, 5.0],
            'length': [44.7, 44.7, 44.7, 44.7, 30.0, 30.0],
            'width': [24.6, 24.6, 24.6, 24.6, 20.0, 20.0],
            'strike': [299.5, 299.5, 299.5, 299.5, 45.0, 45.0],
            'dip': [11.7, 11.7, 11.7, 11.7, 90.0, 90.0],
            'np_st': [1, 1, 1, 1, 1, 1],  # 都是1，表示单个patch
            'np_di': [1, 1, 1, 1, 1, 1],  # 都是1，表示单个patch
            'start_time': [0.0, 0.0, 0.0, 0.0, 10.0, 10.0],
            'pos_s': [-22.35, -11.175, 11.175, 22.35, -15.0, 15.0],
            'pos_d': [12.3, 12.3, 12.3, 12.3, 10.0, 10.0],
            'slip_strike': [0.002, 0.005, 0.015, 0.008, 1.0, 0.5],
            'slip_downdip': [-0.010, -0.020, -0.040, -0.025, 0.0, 0.0],
            'opening': [0.000, 0.000, 0.000, 0.000, 0.0, 0.0]
        }
        
    elif template_type == 'uniform_segments':
        # 断层段统一模式：np_st>1 或 np_di>1，整个断层段统一滑移
        data = {
            'fault_id': [1, 2],
            'o_lat': [1.7987, 2.0000],
            'o_lon': [95.4805, 95.5000],
            'o_depth': [0.0, 5.0],
            'length': [44.7, 30.0],
            'width': [24.6, 20.0],
            'strike': [299.5, 45.0],
            'dip': [11.7, 90.0],
            'np_st': [4, 2],  # 多个patch，统一滑移
            'np_di': [1, 2],  # 多个patch，统一滑移
            'start_time': [0.0, 10.0],
            'slip_strike': [0.010, 1.0],      # 整个断层段的统一滑移
            'slip_downdip': [-0.030, 0.0],    # 整个断层段的统一滑移
            'opening': [0.000, 0.0]           # 整个断层段的统一开度
        }
        
    else:  # mixed
        # 混合模式：既有单个patch又有断层段
        data = {
            'fault_id': [1, 1, 2, 3],
            'o_lat': [1.7987, 1.7987, 2.0000, 1.5000],
            'o_lon': [95.4805, 95.4805, 95.5000, 96.0000],
            'o_depth': [0.0, 0.0, 5.0, 3.0],
            'length': [44.7, 44.7, 30.0, 20.0],
            'width': [24.6, 24.6, 20.0, 15.0],
            'strike': [299.5, 299.5, 45.0, 180.0],
            'dip': [11.7, 11.7, 90.0, 45.0],
            'np_st': [1, 1, 2, 3],        # fault1: 单个patch, fault2: 断层段, fault3: 断层段
            'np_di': [1, 1, 2, 1],        # fault1: 单个patch, fault2: 断层段, fault3: 断层段
            'start_time': [0.0, 0.0, 10.0, 5.0],
            'pos_s': [-11.175, 11.175, None, None],  # 只有单个patch模式才有具体位置
            'pos_d': [12.3, 12.3, None, None],       # 只有单个patch模式才有具体位置
            'slip_strike': [0.005, 0.015, 1.0, 0.8],
            'slip_downdip': [-0.020, -0.040, 0.0, -0.5],
            'opening': [0.000, 0.000, 0.0, 0.0]
        }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Fault sources template ({template_type}) saved to: {filename}")


# Example usage
if __name__ == "__main__":
    # Example 1: Create templates
    print("Creating template files...")
    create_observation_template_csv("obs_rectangular.csv", "rectangular")
    create_observation_template_csv("obs_irregular.csv", "irregular")
    create_fault_template_csv("faults_with_patches.csv", include_patches=True)
    create_fault_template_csv("faults_geometry.csv", include_patches=False)
    
    # Example 2: Load from files
    print("\nLoading configuration from files...")
    config1 = PSCMPConfig.from_files(
        observation_file="obs_rectangular.csv",
        fault_file="faults_with_patches.csv",
        output_dir="./results/",
        insar_los=True
    )
    
    # Example 3: Load from pandas DataFrames
    print("\nLoading from DataFrames...")
    
    # Create observation points
    obs_df = pd.DataFrame({
        'lat': [1.5, 1.6, 1.7, 1.8],
        'lon': [95.0, 95.1, 95.2, 95.3]
    })
    
    # Create fault sources
    fault_df = pd.DataFrame({
        'fault_id': [1],
        'o_lat': [1.7987],
        'o_lon': [95.4805],
        'o_depth': [0.0],
        'length': [44.7],
        'width': [24.6],
        'strike': [299.5],
        'dip': [11.7],
        'np_st': [1],
        'np_di': [1],
        'slip_strike': [0.01],
        'slip_downdip': [-0.03],
        'opening': [0.0]
    })
    
    config2 = PSCMPConfig()
    config2.parameters.load_observation_points_from_dataframe(obs_df)
    config2.parameters.load_fault_sources_from_dataframe(fault_df)
    
    # Example 4: Save configuration
    config1.write_config_file("example_pscmp.dat")
    config1.save_template("example_template.json")
    
    print("\nAll examples completed successfully!")