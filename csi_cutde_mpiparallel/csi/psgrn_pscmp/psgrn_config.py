"""
PSGRN Configuration File Generator

This module provides classes and functions to generate PSGRN input files
with flexible parameter configuration.
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
class LayerModel:
    """
    Single layer definition for PSGRN earth model.
    
    Attributes:
    -----------
    depth : float
        Depth to layer boundary in km
    vp : float
        P-wave velocity in km/s
    vs : float
        S-wave velocity in km/s
    rho : float
        Density in kg/m³
    eta1 : float
        Transient viscosity in Pa·s (<=0 means infinity)
    eta2 : float
        Steady-state viscosity in Pa·s (<=0 means infinity)
    alpha : float
        Ratio between effective and unrelaxed shear modulus (0 < alpha <= 1)
        Default: 1.0 for elastic/Maxwell, 0.5 for Burgers body
    """
    depth: float
    vp: float
    vs: float
    rho: float
    eta1: float = 0.0
    eta2: float = 0.0
    alpha: float = None  # Will be set automatically based on rheology
    
    def __post_init__(self):
        """Validate layer parameters and set default alpha."""
        if self.depth < 0:
            raise ValueError("Layer depth must be non-negative")
        if self.vp <= 0 or self.vs <= 0:
            raise ValueError("Velocities must be positive")
        if self.rho <= 0:
            raise ValueError("Density must be positive")
        
        # Set default alpha based on rheology type
        if self.alpha is None:
            if self.is_burgers():
                self.alpha = 0.5  # Default for Burgers body
            else:
                self.alpha = 1.0  # Default for elastic/Maxwell/SLS
        
        if not (0 < self.alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1")
    
    def is_elastic(self) -> bool:
        """Check if layer is elastic (both viscosities infinite)."""
        return self.eta1 <= 0 and self.eta2 <= 0
    
    def is_maxwell(self) -> bool:
        """Check if layer follows Maxwell rheology."""
        return (self.eta1 <= 0 or self.alpha == 1.0) and self.eta2 > 0
    
    def is_standard_linear_solid(self) -> bool:
        """Check if layer is standard linear solid."""
        return self.eta1 > 0 and self.eta2 <= 0
    
    def is_burgers(self) -> bool:
        """Check if layer is Burgers body (both viscosities finite)."""
        return self.eta1 > 0 and self.eta2 > 0
    
    def get_rheology_type(self) -> str:
        """Get string description of rheology type."""
        if self.is_elastic():
            return "Elastic"
        elif self.is_maxwell():
            return "Maxwell"
        elif self.is_standard_linear_solid():
            return "Standard Linear Solid"
        elif self.is_burgers():
            return "Burgers"
        else:
            return "Unknown"


@dataclass
class PSGRNParameters:
    """
    Complete parameter set for PSGRN calculation.
    
    Attributes:
    -----------
    # Source-observation configuration
    obs_depth : float
        Uniform depth of observation points in km
    earthquake_type : int
        0 for oceanic, 1 for continental earthquakes
    
    # Horizontal distances
    n_distances : int
        Number of horizontal observation distances
    min_distance : float
        Minimum distance in km
    max_distance : float
        Maximum distance in km
    distance_ratio : float
        Ratio between max and min sampling interval
    
    # Source depths
    n_depths : int
        Number of equidistant source depths
    min_depth : float
        Minimum source depth in km
    max_depth : float
        Maximum source depth in km
    
    # Time sampling
    n_time_samples : int
        Number of time samples (should be power of 2)
    time_window : float
        Time window in days
    
    # Wavenumber integration
    accuracy : float
        Relative accuracy of wavenumber integration
    gravity_factor : float
        Factor for gravity effect (0-1)
    
    # Output configuration
    output_dir : str
        Output directory path
    displacement_files : Tuple[str, str, str]
        File names for uz, ur, ut components
    stress_files : Tuple[str, str, str, str, str, str]
        File names for szz, srr, stt, szr, srt, stz components
    tilt_files : Tuple[str, str, str, str, str]
        File names for tr, tt, rot, gd, gr components
    
    # Earth model
    layers : List[LayerModel]
        List of layer definitions
    """
    # Source-observation configuration
    obs_depth: float = 0.0
    earthquake_type: int = 0
    
    # Horizontal distances
    n_distances: int = 73
    min_distance: float = 0.0
    max_distance: float = 2000.0
    distance_ratio: float = 10.0
    
    # Source depths  
    n_depths: int = 30
    min_depth: float = 1.0
    max_depth: float = 59.0
    
    # Time sampling
    n_time_samples: int = 512
    time_window: float = 46628.75
    
    # Wavenumber integration
    accuracy: float = 0.025
    gravity_factor: float = 1.00
    
    # Output configuration
    output_dir: str = './psgrnfcts/'
    displacement_files: Tuple[str, str, str] = ('uz', 'ur', 'ut')
    stress_files: Tuple[str, str, str, str, str, str] = ('szz', 'srr', 'stt', 'szr', 'srt', 'stz')
    tilt_files: Tuple[str, str, str, str, str] = ('tr', 'tt', 'rot', 'gd', 'gr')
    
    # Earth model
    layers: List[LayerModel] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default earth model if none provided."""
        if not self.layers:
            self.layers = self._create_default_model()

        # Auto modify the directory separators
        self.output_dir = self._fix_dir_sep(self.output_dir)

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
    
    def _create_default_model(self) -> List[LayerModel]:
        """Create default 3-layer crustal model."""
        return [
            LayerModel(0.0, 6.0, 3.46, 2600.0),      # Surface
            LayerModel(16.0, 6.0, 3.46, 2600.0),     # Upper crust
            LayerModel(16.0, 6.7, 3.87, 2800.0),     # Interface
            LayerModel(30.0, 6.7, 3.87, 2800.0),     # Lower crust
            LayerModel(30.0, 8.0, 4.62, 3400.0),     # Interface
            LayerModel(60.0, 8.0, 4.62, 3400.0),     # Upper mantle (elastic)
            LayerModel(60.0, 8.0, 4.62, 3400.0, 0.0, 1.0e19, 1.0)  # Lower mantle (viscous)
        ]
    
    def _validate_parameters(self):
        """Validate all parameters."""
        if self.n_distances < 1:
            raise ValueError("Number of distances must be >= 1")
        if self.max_distance <= self.min_distance:
            raise ValueError("Max distance must be > min distance")
        if self.distance_ratio < 1.0:
            raise ValueError("Distance ratio must be >= 1.0")
        
        if self.n_depths < 1:
            raise ValueError("Number of depths must be >= 1")
        if self.max_depth < self.min_depth:
            raise ValueError("Max depth must be >= min depth")
        if self.min_depth <= 0:
            raise ValueError("Min depth must be > 0")
        
        if self.n_time_samples <= 0:
            raise ValueError("Number of time samples must be > 0")
        if self.time_window <= 0:
            raise ValueError("Time window must be > 0")
        
        # Check if n_time_samples is power of 2
        if self.n_time_samples > 2 and (self.n_time_samples & (self.n_time_samples - 1)) != 0:
            warnings.warn("Number of time samples should be power of 2 for optimal FFT performance")
        
        if not (0 < self.accuracy <= 1.0):
            raise ValueError("Accuracy must be between 0 and 1")
        if not (0 <= self.gravity_factor <= 1.0):
            raise ValueError("Gravity factor must be between 0 and 1")
        
        if not self.layers:
            raise ValueError("At least one layer must be defined")
    
    def add_layer(self, depth: float, vp: float, vs: float, rho: float,
                  eta1: float = 0.0, eta2: float = 0.0, alpha: float = None):
        """
        Add a layer to the earth model.
        
        Parameters:
        -----------
        depth : float
            Depth to layer boundary in km
        vp : float
            P-wave velocity in km/s
        vs : float
            S-wave velocity in km/s
        rho : float
            Density in kg/m³
        eta1 : float
            Transient viscosity in Pa·s (<=0 means infinity)
        eta2 : float
            Steady-state viscosity in Pa·s (<=0 means infinity)
        alpha : float, optional
            Ratio between effective and unrelaxed shear modulus
            If None, will be set automatically:
            - 0.5 for Burgers body (eta1 > 0 and eta2 > 0)
            - 1.0 for other rheologies
        """
        layer = LayerModel(depth, vp, vs, rho, eta1, eta2, alpha)
        self.layers.append(layer)
        
        # Sort layers by depth
        self.layers.sort(key=lambda x: x.depth)
    
    def add_burgers_layer(self, depth: float, vp: float, vs: float, rho: float,
                         eta1: float, eta2: float, alpha: float = 0.5):
        """
        Add a Burgers body layer to the earth model.
        
        Parameters:
        -----------
        depth : float
            Depth to layer boundary in km
        vp : float
            P-wave velocity in km/s
        vs : float
            S-wave velocity in km/s
        rho : float
            Density in kg/m³
        eta1 : float
            Transient viscosity in Pa·s (must be > 0)
        eta2 : float
            Steady-state viscosity in Pa·s (must be > 0)
        alpha : float
            Ratio between effective and unrelaxed shear modulus (default: 0.5)
        """
        if eta1 <= 0 or eta2 <= 0:
            raise ValueError("Both eta1 and eta2 must be positive for Burgers body")
        
        self.add_layer(depth, vp, vs, rho, eta1, eta2, alpha)
    
    def set_elastic_halfspace(self, depth: float, vp: float, vs: float, rho: float):
        """
        Set parameters for elastic halfspace.
        
        Parameters:
        -----------
        depth : float
            Depth to halfspace in km
        vp : float
            P-wave velocity in km/s
        vs : float
            S-wave velocity in km/s
        rho : float
            Density in kg/m³
        """
        self.add_layer(depth, vp, vs, rho, 0.0, 0.0, 1.0)
    
    def set_viscous_halfspace(self, depth: float, vp: float, vs: float, rho: float,
                             eta2: float, alpha: float = 1.0):
        """
        Set parameters for viscous halfspace (Maxwell rheology).
        
        Parameters:
        -----------
        depth : float
            Depth to halfspace in km
        vp : float
            P-wave velocity in km/s
        vs : float
            S-wave velocity in km/s
        rho : float
            Density in kg/m³
        eta2 : float
            Steady-state viscosity in Pa·s
        alpha : float
            Ratio between effective and unrelaxed shear modulus (default: 1.0)
        """
        self.add_layer(depth, vp, vs, rho, 0.0, eta2, alpha)
    
    def set_burgers_halfspace(self, depth: float, vp: float, vs: float, rho: float,
                             eta1: float, eta2: float, alpha: float = 0.5):
        """
        Set parameters for Burgers body halfspace.
        
        Parameters:
        -----------
        depth : float
            Depth to halfspace in km
        vp : float
            P-wave velocity in km/s
        vs : float
            S-wave velocity in km/s
        rho : float
            Density in kg/m³
        eta1 : float
            Transient viscosity in Pa·s
        eta2 : float
            Steady-state viscosity in Pa·s
        alpha : float
            Ratio between effective and unrelaxed shear modulus (default: 0.5)
        """
        self.add_burgers_layer(depth, vp, vs, rho, eta1, eta2, alpha)
    
    def clear_layers(self):
        """Clear all layers from the model."""
        self.layers.clear()

    def load_layers_from_file(self, filename: Union[str, Path], file_format: str = 'auto'):
        """
        Load earth model layers from file.
        
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
            if file_format == '.xlsx' or file_format == '.xls':
                file_format = 'excel'
            elif file_format == '.csv':
                file_format = 'csv'
            elif file_format == '.json':
                file_format = 'json'
            elif file_format == '.txt' or file_format == '.dat':
                file_format = 'txt'
            else:
                raise ValueError(f"Cannot auto-detect format for {filepath}")
        
        if file_format == 'csv':
            self._load_from_csv(filename)
        elif file_format == 'excel':
            self._load_from_excel(filename)
        elif file_format == 'json':
            self._load_from_json(filename)
        elif file_format == 'txt':
            self._load_from_txt(filename)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def load_layers_from_dataframe(self, df: pd.DataFrame):
        """
        Load earth model layers from pandas DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns: depth, vp, vs, rho, eta1 (optional), eta2 (optional), alpha (optional)
        """
        required_columns = ['depth', 'vp', 'vs', 'rho']
        optional_columns = ['eta1', 'eta2', 'alpha']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clear existing layers
        self.clear_layers()
        
        # Add layers from DataFrame
        for _, row in df.iterrows():
            kwargs = {
                'depth': float(row['depth']),
                'vp': float(row['vp']),
                'vs': float(row['vs']),
                'rho': float(row['rho'])
            }
            
            # Add optional parameters if present
            for col in optional_columns:
                if col in df.columns and pd.notna(row[col]):
                    kwargs[col] = float(row[col])
            
            self.add_layer(**kwargs)
    
    def load_layers_from_dict(self, layer_data: List[Dict]):
        """
        Load earth model layers from list of dictionaries.
        
        Parameters:
        -----------
        layer_data : List[Dict]
            List of layer dictionaries with keys: depth, vp, vs, rho, eta1, eta2, alpha
        """
        self.clear_layers()
        
        for layer_dict in layer_data:
            # Extract required parameters
            required_params = ['depth', 'vp', 'vs', 'rho']
            kwargs = {}
            
            for param in required_params:
                if param not in layer_dict:
                    raise ValueError(f"Missing required parameter '{param}' in layer definition")
                kwargs[param] = float(layer_dict[param])
            
            # Extract optional parameters
            optional_params = ['eta1', 'eta2', 'alpha']
            for param in optional_params:
                if param in layer_dict and layer_dict[param] is not None:
                    kwargs[param] = float(layer_dict[param])
            
            self.add_layer(**kwargs)
    
    def _load_from_csv(self, filename: Union[str, Path]):
        """Load layers from CSV file."""
        df = pd.read_csv(filename)
        self.load_layers_from_dataframe(df)
    
    def _load_from_excel(self, filename: Union[str, Path], sheet_name: str = 'layers'):
        """Load layers from Excel file."""
        try:
            df = pd.read_excel(filename, sheet_name=sheet_name)
        except ValueError:
            # If specified sheet doesn't exist, try first sheet
            df = pd.read_excel(filename, sheet_name=0)
        self.load_layers_from_dataframe(df)
    
    def _load_from_json(self, filename: Union[str, Path]):
        """Load layers from JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'layers' in data:
            layer_data = data['layers']
        elif isinstance(data, list):
            layer_data = data
        else:
            raise ValueError("JSON file must contain 'layers' array or be an array of layer objects")
        
        self.load_layers_from_dict(layer_data)
    
    def _load_from_txt(self, filename: Union[str, Path]):
        """Load layers from space-separated text file."""
        # Try to read as space-separated values
        try:
            df = pd.read_csv(filename, delim_whitespace=True, comment='#')
            
            # Auto-detect column names if not provided
            if df.columns[0].isdigit() or df.columns[0].replace('.', '').isdigit():
                # First row is data, not headers
                expected_cols = ['depth', 'vp', 'vs', 'rho', 'eta1', 'eta2', 'alpha']
                df.columns = expected_cols[:len(df.columns)]
            
            self.load_layers_from_dataframe(df)
        except Exception as e:
            raise ValueError(f"Failed to load text file {filename}: {e}")
    
    def save_layers_to_file(self, filename: Union[str, Path], file_format: str = 'auto'):
        """
        Save earth model layers to file.
        
        Parameters:
        -----------
        filename : str or Path
            Output filename
        file_format : str
            File format: 'csv', 'excel', 'json', or 'auto' (auto-detect from extension)
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
                file_format = 'csv'  # Default to CSV
        
        df = self.to_dataframe()
        
        if file_format == 'csv':
            df.to_csv(filename, index=False)
        elif file_format == 'excel':
            df.to_excel(filename, sheet_name='layers', index=False)
        elif file_format == 'json':
            layer_data = df.to_dict('records')
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({'layers': layer_data}, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert earth model layers to pandas DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with layer parameters
        """
        data = []
        for layer in self.layers:
            data.append({
                'depth': layer.depth,
                'vp': layer.vp,
                'vs': layer.vs,
                'rho': layer.rho,
                'eta1': layer.eta1,
                'eta2': layer.eta2,
                'alpha': layer.alpha,
                'rheology': layer.get_rheology_type()
            })
        
        return pd.DataFrame(data)
    
    def create_standard_models(self, model_type: str = 'continental'):
        """
        Create standard earth models.
        
        Parameters:
        -----------
        model_type : str
            Model type: 'continental', 'oceanic', 'simple_viscous', 'layered_viscous'
        """
        self.clear_layers()
        
        if model_type == 'continental':
            # Standard continental crust model
            layers_data = [
                {'depth': 0.0, 'vp': 6.0, 'vs': 3.46, 'rho': 2600.0},  # Upper crust
                {'depth': 20.0, 'vp': 6.7, 'vs': 3.87, 'rho': 2800.0}, # Lower crust
                {'depth': 40.0, 'vp': 8.0, 'vs': 4.62, 'rho': 3400.0}, # Elastic mantle
                {'depth': 100.0, 'vp': 8.2, 'vs': 4.7, 'rho': 3500.0, 'eta2': 1.0e19} # Viscous mantle
            ]
        
        elif model_type == 'oceanic':
            # Standard oceanic crust model
            layers_data = [
                {'depth': 0.0, 'vp': 5.0, 'vs': 2.5, 'rho': 2200.0},   # Sediments
                {'depth': 2.0, 'vp': 6.5, 'vs': 3.7, 'rho': 2900.0},   # Oceanic crust
                {'depth': 8.0, 'vp': 8.0, 'vs': 4.62, 'rho': 3400.0},  # Elastic mantle
                {'depth': 60.0, 'vp': 8.2, 'vs': 4.7, 'rho': 3500.0, 'eta2': 5.0e18} # Viscous mantle
            ]
        
        elif model_type == 'simple_viscous':
            # Simple two-layer model
            layers_data = [
                {'depth': 0.0, 'vp': 6.5, 'vs': 3.75, 'rho': 2700.0},  # Elastic crust
                {'depth': 30.0, 'vp': 8.1, 'vs': 4.65, 'rho': 3400.0, 'eta2': 1.0e19} # Viscous mantle
            ]
        
        elif model_type == 'layered_viscous':
            # Multi-layer model with different viscosities
            layers_data = [
                {'depth': 0.0, 'vp': 6.0, 'vs': 3.46, 'rho': 2600.0},   # Elastic crust
                {'depth': 30.0, 'vp': 8.0, 'vs': 4.62, 'rho': 3400.0, 'eta2': 1.0e18}, # High viscosity
                {'depth': 100.0, 'vp': 8.2, 'vs': 4.7, 'rho': 3500.0, 'eta2': 1.0e19}, # Medium viscosity
                {'depth': 200.0, 'vp': 8.5, 'vs': 4.8, 'rho': 3600.0, 'eta2': 1.0e20}  # Low viscosity
            ]
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.load_layers_from_dict(layers_data)


class PSGRNConfig:
    """
    PSGRN configuration file generator.
    
    This class creates properly formatted PSGRN input files with complete
    header comments and parameter validation.
    """
    
    def __init__(self, parameters: Optional[PSGRNParameters] = None):
        """
        Initialize PSGRN configuration generator.
        
        Parameters:
        -----------
        parameters : PSGRNParameters, optional
            Configuration parameters. If None, uses defaults.
        """
        self.parameters = parameters or PSGRNParameters()

    @classmethod
    def from_template(cls, template_name: str = 'continental'):
        """
        Create configuration from predefined template.
        
        Parameters:
        -----------
        template_name : str
            Template name: 'continental', 'oceanic', 'simple_viscous', 'layered_viscous'
        """
        config = cls()
        config.parameters.create_standard_models(template_name)
        return config
    
    @classmethod
    def from_layer_file(cls, layer_file: Union[str, Path], **kwargs):
        """
        Create configuration from layer definition file.
        
        Parameters:
        -----------
        layer_file : str or Path
            Layer definition file (CSV, Excel, JSON, or TXT)
        **kwargs : dict
            Additional parameter updates
        """
        config = cls()
        config.parameters.load_layers_from_file(layer_file)
        
        # Update other parameters if provided
        config.update_parameters(**kwargs)
        
        return config
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, **kwargs):
        """
        Create configuration from pandas DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with layer definitions
        **kwargs : dict
            Additional parameter updates
        """
        config = cls()
        config.parameters.load_layers_from_dataframe(df)
        
        # Update other parameters if provided
        config.update_parameters(**kwargs)
        
        return config
    
    def save_template(self, filename: Union[str, Path]):
        """
        Save current configuration as a template file.
        
        Parameters:
        -----------
        filename : str or Path
            Template filename (will save as JSON)
        """
        template_data = {
            'parameters': {
                'obs_depth': self.parameters.obs_depth,
                'earthquake_type': self.parameters.earthquake_type,
                'n_distances': self.parameters.n_distances,
                'min_distance': self.parameters.min_distance,
                'max_distance': self.parameters.max_distance,
                'distance_ratio': self.parameters.distance_ratio,
                'n_depths': self.parameters.n_depths,
                'min_depth': self.parameters.min_depth,
                'max_depth': self.parameters.max_depth,
                'n_time_samples': self.parameters.n_time_samples,
                'time_window': self.parameters.time_window,
                'accuracy': self.parameters.accuracy,
                'gravity_factor': self.parameters.gravity_factor,
                'output_dir': self.parameters._fix_dir_sep(self.parameters.output_dir)
            },
            'layers': self.parameters.to_dataframe().to_dict('records')
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2)
    
    def load_template(self, filename: Union[str, Path]):
        """
        Load configuration from template file.
        
        Parameters:
        -----------
        filename : str or Path
            Template filename (JSON format)
        """
        with open(filename, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
        
        # Load parameters
        if 'parameters' in template_data:
            self.update_parameters(**template_data['parameters'])
        
        # Load layers
        if 'layers' in template_data:
            self.parameters.load_layers_from_dict(template_data['layers'])

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
        Generate complete PSGRN configuration file content.
        
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
        lines.extend(self._generate_source_obs_config())
        lines.extend(self._generate_time_config())
        lines.extend(self._generate_wavenumber_config())
        lines.extend(self._generate_output_config())
        lines.extend(self._generate_model_config())
        
        # Add end marker
        lines.append("#=======================end of input===========================================")
        
        return '\n'.join(lines)
    
    def write_config_file(self, filename: Union[str, Path] = "psgrn_input.dat"):
        """
        Write PSGRN configuration file.
        
        Parameters:
        -----------
        filename : str or Path
            Output filename
        """
        config_content = self.generate_config_string()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"PSGRN configuration written to: {filename}")
    
    def _generate_header(self) -> List[str]:
        """Generate file header with comments."""
        return [
            "#=============================================================================",
            "# This is input file of FORTRAN77 program \"psgrn08\" for computing responses",
            "# (Green's functions) of a multi-layered viscoelastic halfspace to point",
            "# dislocation sources buried at different depths. All results will be stored in",
            "# the given directory and provide the necessary data base for the program",
            "# \"pscmp07a\" for computing time-dependent deformation, geoid and gravity changes",
            "# induced by an earthquake with extended fault planes via linear superposition.",
            "# For more details, please read the accompanying READ.ME file.",
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
            "## Cylindrical coordinates (Z positive downwards!) are used.   ##",
            "##                                                             ##",
            "## If not specified otherwise, SI Unit System is used overall! ##",
            "##                                                             ##",
            "#################################################################",
            "#"
        ]
    
    def _generate_source_obs_config(self) -> List[str]:
        """Generate source-observation configuration section."""
        p = self.parameters
        
        return [
            "#------------------------------------------------------------------------------",
            "#",
            "#\tPARAMETERS FOR SOURCE-OBSERVATION CONFIGURATIONS",
            "#\t================================================",
            "# 1. the uniform depth of the observation points [km], switch for oceanic (0)",
            "#    or continental(1) earthquakes;",
            "# 2. number of (horizontal) observation distances (> 1 and <= nrmax defined in",
            "#    psgglob.h), start and end distances [km], ratio (>= 1.0) between max. and",
            "#    min. sampling interval (1.0 for equidistant sampling);",
            "# 3. number of equidistant source depths (>= 1 and <= nzsmax defined in",
            "#    psgglob.h), start and end source depths [km];",
            "#",
            "#    r1,r2 = minimum and maximum horizontal source-observation",
            "#            distances (r2 > r1).",
            "#    zs1,zs2 = minimum and maximum source depths (zs2 >= zs1 > 0).",
            "#",
            "#    Note that the same sampling rates dr_min and dzs will be used later by the",
            "#    program \"pscmp08\" for discretizing the finite source planes to a 2D grid",
            "#    of point sources.",
            "#------------------------------------------------------------------------------",
            f" {p.obs_depth:8.1f}       {p.earthquake_type}",
            f" {p.n_distances:3d}   {p.min_distance:6.1f}  {p.max_distance:7.1f}  {p.distance_ratio:5.1f}",
            f" {p.n_depths:3d}   {p.min_depth:6.1f}    {p.max_depth:5.1f}"
        ]
    
    def _generate_time_config(self) -> List[str]:
        """Generate time sampling configuration section."""
        p = self.parameters
        
        return [
            "#------------------------------------------------------------------------------",
            "#",
            "#\tPARAMETERS FOR TIME SAMPLING",
            "#\t============================",
            "# 1. number of time samples (<= ntmax def. in psgglob.h) and time window [days].",
            "#",
            "#    Note that nt (> 0) should be power of 2 (the fft-rule). If nt = 1, the",
            "#    coseismic (t = 0) changes will be computed; If nt = 2, the coseismic",
            "#    (t = 0) and steady-state (t -> infinity) changes will be computed;",
            "#    Otherwise, time series for the given time samples will be computed.",
            "#",
            "#------------------------------------------------------------------------------",
            f" {p.n_time_samples:3d}   {p.time_window:8.2f}"
        ]
    
    def _generate_wavenumber_config(self) -> List[str]:
        """Generate wavenumber integration configuration section."""
        p = self.parameters
        
        return [
            "#------------------------------------------------------------------------------",
            "#",
            "#\tPARAMETERS FOR WAVENUMBER INTEGRATION",
            "#\t=====================================",
            "# 1. relative accuracy of the wave-number integration (suggested: 0.1 - 0.01)",
            "# 2. factor (> 0 and < 1) for including influence of earth's gravity on the",
            "#    deformation field (e.g. 0/1 = without / with 100% gravity effect).",
            "#------------------------------------------------------------------------------",
            f" {p.accuracy:5.3f}",
            f" {p.gravity_factor:4.2f}"
        ]
    
    def _generate_output_config(self) -> List[str]:
        """Generate output files configuration section."""
        p = self.parameters
        
        return [
            "#------------------------------------------------------------------------------",
            "#",
            "#\tPARAMETERS FOR OUTPUT FILES",
            "#\t===========================",
            "#",
            "# 1. output directory",
            "# 2. file names for 3 displacement components (uz, ur, ut)",
            "# 3. file names for 6 stress components (szz, srr, stt, szr, srt, stz)",
            "# 4. file names for radial and tangential tilt components (as measured by a",
            "#    borehole tiltmeter), rigid rotation of horizontal plane, geoid and gravity",
            "#    changes (tr, tt, rot, gd, gr)",
            "#",
            "#    Note that all file or directory names should not be longer than 80",
            "#    characters. Directory and subdirectoy names must be separated and ended",
            "#    by / (unix) or \\ (dos)! All file names should be given without extensions",
            "#    that will be appended automatically by \".ep\" for the explosion (inflation)",
            "#    source, \".ss\" for the strike-slip source, \".ds\" for the dip-slip source,",
            "#    and \".cl\" for the compensated linear vector dipole source)",
            "#",
            "#------------------------------------------------------------------------------",
            f" '{self.parameters._fix_dir_sep(p.output_dir)}'",
            f" '{p.displacement_files[0]}'  '{p.displacement_files[1]}'  '{p.displacement_files[2]}'",
            f" '{p.stress_files[0]}' '{p.stress_files[1]}' '{p.stress_files[2]}' '{p.stress_files[3]}' '{p.stress_files[4]}' '{p.stress_files[5]}'",
            f" '{p.tilt_files[0]}'  '{p.tilt_files[1]}'  '{p.tilt_files[2]}' '{p.tilt_files[3]}'  '{p.tilt_files[4]}'"
        ]
    
    def _generate_model_config(self) -> List[str]:
        """Generate earth model configuration section."""
        p = self.parameters
        
        lines = [
            "#------------------------------------------------------------------------------",
            "#",
            "#\tGLOBAL MODEL PARAMETERS",
            "#\t=======================",
            "# 1. number of data lines of the layered model (<= lmax as defined in psgglob.h)",
            "#",
            "#    The surface and the upper boundary of the half-space as well as the",
            "#    interfaces at which the viscoelastic parameters are continuous, are all",
            "#    defined by a single data line; All other interfaces, at which the",
            "#    viscoelastic parameters are discontinuous, are all defined by two data",
            "#    lines (upper-side and lower-side values). This input format could also be",
            "#    used for a graphic plot of the layered model. Layers which have different",
            "#    parameter values at top and bottom, will be treated as layers with a",
            "#    constant gradient, and will be discretised to a number of homogeneous",
            "#    sublayers. Errors due to the discretisation are limited within about 5%",
            "#    (changeable, see psgglob.h).",
            "#",
            "# 2....	parameters of the multilayered model",
            "#",
            "#    Burgers rheology [a Kelvin-Voigt body (mu1, eta1) and a Maxwell body",
            "#    (mu2, eta2) in series connection] for relaxation of shear modulus is",
            "#    implemented. No relaxation of compressional modulus is considered.",
            "#",
            "#    eta1  = transient viscosity (dashpot of the Kelvin-Voigt body; <= 0 means",
            "#            infinity value)",
            "#    eta2  = steady-state viscosity (dashpot of the Maxwell body; <= 0 means",
            "#            infinity value)",
            "#    alpha = ratio between the effective and the unrelaxed shear modulus",
            "#            = mu1/(mu1+mu2) (> 0 and <= 1) (unrelaxed modulus mu2 is",
            "#            derived from S wave velocity and density)",
            "#",
            "#    Special cases and typical alpha values:",
            "#        (1) Elastic: eta1 and eta2 <= 0 (i.e. infinity); alpha meaningless",
            "#        (2) Maxwell body: eta1 <= 0 (i.e. eta1 = infinity)",
            "#                          or alpha = 1 (i.e. mu1 = infinity); alpha = 1.0",
            "#        (3) Standard-Linear-Solid: eta2 <= 0 (i.e. infinity); alpha = 1.0",
            "#            fully relaxed modulus = alpha*unrelaxed_modulus",
            "#            characteristic relaxation time = eta1*alpha/unrelaxed_modulus",
            "#        (4) Burgers body: eta1 > 0 and eta2 > 0; alpha typically = 0.5",
            "#            This represents the most general viscoelastic behavior",
            "#------------------------------------------------------------------------------",
            f" {len(p.layers):2d}                               |int: no_model_lines;",
            "#------------------------------------------------------------------------------",
            "# no  depth[km]  vp[km/s]  vs[km/s]  rho[kg/m^3] eta1[Pa*s] eta2[Pa*s] alpha",
            "#    Rheology types: Elastic | Maxwell | SLS | Burgers",
            "#------------------------------------------------------------------------------"
        ]
        
        # Add layer definitions with rheology type comments
        for i, layer in enumerate(p.layers, 1):
            rheology_comment = f"  # {layer.get_rheology_type()}"
            lines.append(
                f" {i:2d}      {layer.depth:6.3f}   {layer.vp:6.4f}    {layer.vs:6.4f}    "
                f"{layer.rho:6.1f}      {layer.eta1:8.1E}    {layer.eta2:8.1E}    {layer.alpha:5.3f}"
                f"{rheology_comment}"
            )
        
        return lines


# Utility functions for creating common layer configurations
def create_layer_template_csv(filename: str = "layer_template.csv"):
    """
    Create a CSV template file for layer definitions.
    
    Parameters:
    -----------
    filename : str
        Output CSV filename
    """
    template_data = {
        'depth': [0.0, 20.0, 40.0, 100.0],
        'vp': [6.0, 6.7, 8.0, 8.2],
        'vs': [3.46, 3.87, 4.62, 4.7],
        'rho': [2600.0, 2800.0, 3400.0, 3500.0],
        'eta1': [0.0, 0.0, 0.0, 0.0],
        'eta2': [0.0, 0.0, 0.0, 1.0e19],
        'alpha': [1.0, 1.0, 1.0, 1.0]
    }
    
    df = pd.DataFrame(template_data)
    df.to_csv(filename, index=False)
    print(f"Layer template saved to: {filename}")


def create_layer_template_excel(filename: str = "layer_template.xlsx"):
    """
    Create an Excel template file for layer definitions.
    
    Parameters:
    -----------
    filename : str
        Output Excel filename
    """
    template_data = {
        'depth': [0.0, 20.0, 40.0, 100.0],
        'vp': [6.0, 6.7, 8.0, 8.2],
        'vs': [3.46, 3.87, 4.62, 4.7],
        'rho': [2600.0, 2800.0, 3400.0, 3500.0],
        'eta1': [0.0, 0.0, 0.0, 0.0],
        'eta2': [0.0, 0.0, 0.0, 1.0e19],
        'alpha': [1.0, 1.0, 1.0, 1.0],
        'comment': ['Upper crust', 'Lower crust', 'Elastic mantle', 'Viscous mantle']
    }
    
    df = pd.DataFrame(template_data)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='layers', index=False)
        
        # Add formatting
        workbook = writer.book
        worksheet = writer.sheets['layers']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 20)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"Layer template saved to: {filename}")


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Create from template
    print("Example 1: Creating from template")
    config1 = PSGRNConfig.from_template('continental')
    config1.write_config_file("test_continental.dat")
    
    # Example 2: Create template files
    print("\nExample 2: Creating template files")
    create_layer_template_csv()
    create_layer_template_excel()
    
    # Example 3: Load from CSV file
    print("\nExample 3: Loading from CSV")
    config2 = PSGRNConfig.from_layer_file("layer_template.csv",
                                         n_distances=100,
                                         max_distance=1500.0,
                                         output_dir='./output/')
    
    # Example 4: Create from pandas DataFrame
    print("\nExample 4: Creating from DataFrame")
    import pandas as pd
    
    layer_data = pd.DataFrame({
        'depth': [0.0, 30.0, 100.0],
        'vp': [6.0, 8.0, 8.2],
        'vs': [3.46, 4.62, 4.7],
        'rho': [2600.0, 3400.0, 3500.0],
        'eta2': [0.0, 0.0, 1.0e19]
    })
    
    config3 = PSGRNConfig.from_dataframe(layer_data)
    
    # Example 5: Save and load template
    print("\nExample 5: Template save/load")
    config3.save_template("my_model_template.json")
    
    config4 = PSGRNConfig()
    config4.load_template("my_model_template.json")
    config4.write_config_file("test_from_template.dat")
    
    print("\nAll examples completed successfully!")