import yaml
import os
import numpy as np
import logging

# Setup module-level logger
logger = logging.getLogger(__name__)

# Load csi and its extensions
from csi.gps import gps
from csi.insar import insar
from ..multifaults_base import MyMultiFaultsInversion

# Import utility functions for parsing configuration updates
from .config_utils import parse_update, parse_initial_values
from .config_utils import parse_alpha_faults, parse_data_faults, parse_sigmas_config, parse_alpha_config
from .config_utils import parse_euler_constraints_config, parse_euler_units
from .base_config import CommonConfigBase


class LinearInversionConfig(CommonConfigBase):
    """
    Linear inversion configuration base class.
    This class provides common functionality for both Bayesian and BLSE inversions.
    """
    
    def __init__(self, config_file='default_config.yml', multifaults=None, geodata=None, 
                 verticals=None, polys=None, dataFaults=None, alphaFaults=None, faults_list=None,
                 gfmethods=None, encoding='utf-8', verbose=False, parallel_rank=None, **kwargs):
        
        super().__init__(config_file=config_file, geodata=geodata, verbose=verbose, parallel_rank=parallel_rank)
        
        # Common attributes for linear inversion
        self.multifaults = multifaults
        self.use_bounds_constraints = True
        self.use_rake_angle_constraints = True
        self.use_euler_constraints = False
        self.GLs = None
        self.moment_magnitude_threshold = None
        self.patch_areas = None
        self.shear_modulus = 3.0e10
        self.magnitude_tolerance = None
        self.nonlinear_inversion = False
        self.rake_angle = None
        self.faults = {}  # Dictionary to store fault parameters

        # Initialize alpha with default values
        self.alpha = {
            'enabled': True,
            'update': True,
            'initial_value': 0.0,
            'log_scaled': True,
            'faults': None
        }

        # Initialize DES configuration with default values
        self.des = self._parse_des_config(None)

        # Initialize Euler constraints with default values
        self.euler_constraints = {
            'enabled': False,
            'defaults': {},
            'faults': {},
            'configured_faults': []
        }

        # Initialize faults
        assert multifaults or faults_list, "Either multifaults or faults_list must be provided"
        if multifaults is not None:
            self.faultnames = multifaults.faultnames
            self.faults_list = multifaults.faults
        else:
            self.faults_list = faults_list
            self.faultnames = [fault.name for fault in faults_list]

        # Load configuration from file if provided
        if config_file is not None:
            self.load_from_file(config_file, encoding=encoding)

        self.shear_modulus = float(self.shear_modulus)

        # Set attributes from kwargs
        self.set_attributes(**kwargs)

        # Initialize geodata and related processing
        self._update_geodata(geodata)
        self._validate_verticals(verticals)
        self._select_data_sets()
        self._validate_polys(polys)

        # Set fault-related data
        self.set_data_faults(dataFaults)
        self.set_alpha_faults(alphaFaults)

        # [New] Parse DES configuration from kwargs
        des = getattr(self, 'des', None)
        self.des = self._parse_des_config(des)

        # Parse alpha parameters
        fault_names = self.faultnames
        self.alpha = parse_alpha_config(self.alpha, faultnames=fault_names)

        # Parse Euler constraints if enabled
        if self.use_euler_constraints:
            self._parse_euler_constraints()

        # Update Green's function parameters
        self.update_GFs_parameters(self.geodata['data'], self.geodata['verticals'],
                                  self.geodata['faults'], gfmethods)

        # Parse sigmas parameters
        # n_datasets = len(self.geodata.get('data', []))
        data_names = [d.name for d in self.geodata.get('data', [])]
        self.geodata['sigmas'] = parse_sigmas_config(self.geodata['sigmas'], dataset_names=data_names)

    def _parse_des_config(self, des_dict):
        """
        Parse DES configuration from a dictionary, merging with default values.
        """
        # Default configuration structure
        default_des = {
            'enabled': False,
            'mode': 'per_patch',
            'G_norm': 'l2',
            'depth_grouping': {
                'strategy': 'uniform',
                'interval': 1.0,
                'tolerance': 1e-6
            }
        }

        if des_dict is None:
            return default_des

        # Merge user configuration with default configuration
        config = default_des.copy()
        config.update(des_dict)
        
        # Handle nested depth_grouping
        if 'depth_grouping' in des_dict:
            default_depth = default_des['depth_grouping'].copy()
            default_depth.update(des_dict['depth_grouping'])
            config['depth_grouping'] = default_depth

        # Simple validation
        valid_modes = ['per_patch', 'per_depth', 'per_column']
        if config['mode'] not in valid_modes:
            if self.verbose:
                logger.warning(f"Invalid DES mode '{config['mode']}'. Fallback to 'per_patch'.")
            config['mode'] = 'per_patch'

        return config

    def _validate_polys(self, polys):
        """
        Parse and validate the 'polys' parameter from geodata configuration with enhanced flexibility.
        
        Parameters:
        -----------
        polys : int, str, list, or None
            The 'polys' parameter provided by the user. Multiple formats supported:
            - None: No polynomial corrections will be estimated for any dataset
            - int/str: Single polynomial order applied to all datasets
            - list: Polynomial settings for each dataset individually
        
        Examples:
        ---------
        >>> _validate_polys(None)  # No polynomial corrections
        geodata['polys'] = [None, None, None]
        
        >>> _validate_polys(3)  # Polynomial order 3 for all datasets
        geodata['polys'] = [3, 3, 3]
        
        >>> _validate_polys([3, None, 1])  # Per-dataset settings
        geodata['polys'] = [3, None, 1]
        
        Raises:
        -------
        ValueError: If the length of the 'polys' list does not match the number of datasets
        """
        # Get current polys configuration safely, default to input parameter or empty list
        current_polys = self.geodata.get('polys')
        if current_polys is None:
            self.geodata['polys'] = polys if polys is not None else []
        
        # If polys is still empty/None, set default based on data types
        if not self.geodata.get('polys'):
            self.geodata['polys'] = []
            for data in self.geodata.get('data', []):
                if data.dtype == 'insar':
                    self.geodata['polys'].append(None)  # No polynomial correction for InSAR by default
                else:
                    self.geodata['polys'].append(None)  # No polynomial correction for other data types
        
        # Handle different input formats
        polys_config = self.geodata.get('polys', [])
        n_datasets = len(self.geodata.get('data', []))
        
        if isinstance(polys_config, list):
            # List format - validate length matches number of datasets
            if len(polys_config) != n_datasets:
                msg = f"Length of 'polys' list ({len(polys_config)}) does not match the number of datasets ({n_datasets})"
                logger.error(msg)
                raise ValueError(msg)
        elif isinstance(polys_config, (int, str, type(None))):
            # Single value format - expand to all datasets
            self.geodata['polys'] = [polys_config] * n_datasets
        else:
            msg = f"'polys' must be None, int, str, or list, got {type(polys_config)}"
            logger.error(msg)
            raise ValueError(msg)
        # print(f"Polys configuration set to: {self.geodata['polys']}")

    @property
    def alphaFaults(self):
        return self.alpha.get('faults')
    
    @alphaFaults.setter
    def alphaFaults(self, value):
        all_faultnames = self.faultnames
        alphaFaults = value if value is not None else self.alpha['faults']
        result = parse_alpha_faults(alphaFaults, all_faultnames, param_name='alphaFaults')
        self.alpha['faults'] = result

    def load_from_file(self, config_file, encoding='utf-8'):
        """
        Load configuration from a file.
        """
        with open(config_file, 'r', encoding=encoding) as f:
            config_data = yaml.safe_load(f)
        
        # Set lon0 and lat0
        lon_lat_0 = config_data.get('lon_lat_0', None)
        if lon_lat_0 is not None:
            self.lon0 = lon_lat_0[0]
            self.lat0 = lon_lat_0[1]
        config_data.pop('lon_lat_0', None)

        # Handle alpha configuration
        if 'alpha' in config_data:
            self.alpha.update(config_data['alpha'])
            config_data.pop('alpha')

        # Handle Euler constraints configuration
        if 'euler_constraints' in config_data:
            self.euler_constraints.update(config_data['euler_constraints'])
            config_data.pop('euler_constraints')

        # Process faults configuration
        self._process_faults_config(config_data)
        
        # Set attributes
        self.set_attributes(**config_data)

    def _parse_euler_constraints(self):
        """
        Parse Euler pole constraints configuration and handle unit conversions.
        """
        try:
            # Get dataset names if available
            dataset_names = [d.name for d in self.geodata.get('data', [])] if self.geodata.get('data') else None
            
            # Parse Euler constraints configuration
            self.euler_constraints = parse_euler_constraints_config(
                self.euler_constraints, 
                self.faultnames, 
                dataset_names=dataset_names
            )
            
            # Process unit conversions for each fault
            if self.euler_constraints['enabled']:
                self._process_euler_unit_conversions()
                
            if self.verbose and self.euler_constraints['enabled']:
                logger.info(f"Euler constraints parsed for faults: {self.euler_constraints['configured_faults']}")
                
        except Exception as e:
            if self.verbose:
                logger.warning(f"Warning: Failed to parse Euler constraints: {e}")
            # Disable Euler constraints if parsing fails
            self.euler_constraints['enabled'] = False
            self.use_euler_constraints = False

    def _process_euler_unit_conversions(self):
        """
        Process unit conversions for Euler constraints.
        """
        # Process defaults
        defaults = self.euler_constraints['defaults']
        
        # Parse default units
        if 'euler_pole_units' in defaults:
            pole_units = parse_euler_units(defaults['euler_pole_units'], 'euler_pole')
            defaults['_euler_pole_conversion'] = pole_units
            
        if 'euler_vector_units' in defaults:
            vector_units = parse_euler_units(defaults['euler_vector_units'], 'euler_vector')
            defaults['_euler_vector_conversion'] = vector_units
        
        # Process fault-specific configurations
        for fault_name, fault_config in self.euler_constraints['faults'].items():
            # Use fault-specific units or fall back to defaults
            fault_units = fault_config.get('units', {})
            
            # Euler pole units
            pole_units_config = fault_units.get('euler_pole_units', defaults['euler_pole_units'])
            pole_conversion = parse_euler_units(pole_units_config, 'euler_pole')
            fault_config['_euler_pole_conversion'] = pole_conversion
            
            # Euler vector units
            vector_units_config = fault_units.get('euler_vector_units', defaults['euler_vector_units'])
            vector_conversion = parse_euler_units(vector_units_config, 'euler_vector')
            fault_config['_euler_vector_conversion'] = vector_conversion
            
            # Convert block parameters to standard units
            self._convert_block_parameters(fault_config)

    def _convert_block_parameters(self, fault_config):
        """
        Convert block parameters to standard units (radians and radians/year).
        """
        block_types = fault_config['block_types']
        blocks = fault_config['blocks']
        
        converted_blocks = []
        
        for i, (block_type, block) in enumerate(zip(block_types, blocks)):
            if block_type == 'dataset':
                # Dataset names don't need conversion
                converted_blocks.append(block)
            elif block_type == 'euler_pole':
                # Convert [lat, lon, omega] to standard units
                conversion = fault_config['_euler_pole_conversion']
                factors = conversion['conversion_factors']
                
                converted_block = [
                    block[0] * factors[0],  # latitude to radians
                    block[1] * factors[1],  # longitude to radians
                    block[2] * factors[2]   # angular velocity to radians/year
                ]
                converted_blocks.append(converted_block)
                
            elif block_type == 'euler_vector':
                # Convert [wx, wy, wz] to standard units
                conversion = fault_config['_euler_vector_conversion']
                factors = conversion['conversion_factors']
                
                converted_block = [
                    block[0] * factors[0],  # wx to radians/year
                    block[1] * factors[1],  # wy to radians/year
                    block[2] * factors[2]   # wz to radians/year
                ]
                converted_blocks.append(converted_block)
        
        # Store both original and converted blocks
        fault_config['blocks_original'] = blocks
        fault_config['blocks_standard'] = converted_blocks

    def get_euler_constraint_parameters(self, fault_name):
        """
        Get Euler constraint parameters for a specific fault in standard units.
        
        Parameters:
        -----------
        fault_name : str
            Name of the fault
            
        Returns:
        --------
        dict or None
            Dictionary containing constraint parameters in standard units, or None if not configured
        """
        if not self.euler_constraints['enabled']:
            return None
            
        if fault_name not in self.euler_constraints['faults']:
            return None
            
        fault_config = self.euler_constraints['faults'][fault_name]
        
        return {
            'block_types': fault_config['block_types'],
            'blocks_standard': fault_config['blocks_standard'],
            'block_names': fault_config['block_names'],
            'fix_reference_block': fault_config['fix_reference_block'],
            'apply_to_patches': fault_config['apply_to_patches'],
            'normalization': fault_config.get('normalization', self.euler_constraints['defaults']['normalization']),
            'regularization': fault_config.get('regularization', self.euler_constraints['defaults']['regularization'])
        }

    def get_all_euler_constraints(self):
        """
        Get all Euler constraint parameters for configured faults.
        
        Returns:
        --------
        dict
            Dictionary mapping fault names to their constraint parameters
        """
        if not self.euler_constraints['enabled']:
            return {}
            
        result = {}
        for fault_name in self.euler_constraints['configured_faults']:
            result[fault_name] = self.get_euler_constraint_parameters(fault_name)
            
        return result

    def convert_euler_parameter_units(self, values, from_units, to_units, param_type):
        """
        Convert Euler parameter units.
        
        Parameters:
        -----------
        values : list or array
            Parameter values to convert
        from_units : list
            Source units
        to_units : list  
            Target units
        param_type : str
            'euler_pole' or 'euler_vector'
            
        Returns:
        --------
        list
            Converted values
        """
        from_conversion = parse_euler_units(from_units, param_type)
        to_conversion = parse_euler_units(to_units, param_type)
        
        from_factors = from_conversion['conversion_factors']
        to_factors = to_conversion['conversion_factors']
        
        # Convert to standard units first, then to target units
        converted = []
        for i, value in enumerate(values):
            standard_value = value * from_factors[i]
            target_value = standard_value / to_factors[i]
            converted.append(target_value)
            
        return converted

    def _process_faults_config(self, config_data):
        """
        Process fault configuration from the config file.
        """
        default_fault_parameters = config_data.get('faults', {}).get('defaults', {})

        for fault_name, fault_parameters in config_data.get('faults', {}).items():
            if fault_name == 'defaults':
                continue

            if fault_parameters is None:
                config_data['faults'][fault_name] = default_fault_parameters.copy()
            else:
                merged_parameters = {**default_fault_parameters, **fault_parameters}
                
                # Special handling for 'geometry'
                merged_geometry = {**default_fault_parameters.get('geometry', {}), 
                                 **fault_parameters.get('geometry', {})}
                merged_parameters['geometry'] = merged_geometry
                
                # Special handling for 'method_parameters'
                merged_method_parameters = default_fault_parameters.get('method_parameters', {}).copy()
                for method_name, method_params in fault_parameters.get('method_parameters', {}).items():
                    if method_name in merged_method_parameters:
                        if 'method' in merged_method_parameters[method_name] or 'method' in method_params:
                            merged_method_parameters[method_name] = method_params
                        else:
                            merged_method_parameters[method_name] = {**merged_method_parameters[method_name], **method_params}
                    else:
                        merged_method_parameters[method_name] = method_params
                merged_parameters['method_parameters'] = merged_method_parameters
                
                config_data['faults'][fault_name] = merged_parameters

            self.faults[fault_name] = config_data['faults'][fault_name]
        
        # Ensure all faults in faultnames are configured
        for fault_name in self.faultnames:
            if fault_name not in self.faults:
                config_data['faults'][fault_name] = default_fault_parameters.copy()
                self.faults[fault_name] = config_data['faults'][fault_name]
                if self.verbose:
                    logger.info(f"Fault '{fault_name}' not found in config file. Using default parameters.")

    def _initialize_faults_and_assemble_data(self, faults_list=None, geodata=None):
        """
        Set up faults by building Green's functions, assembling data and Green's functions for inversion,
        and building covariance matrices for GPS/InSAR data.
        """
        # Build Green's functions for all faults
        faults_list = faults_list or self.faults_list
        geodata = self.geodata['data'] or geodata
        verticals = self.geodata['verticals']
        polys = self.geodata['polys']
        nonpolys = [None] * len(geodata)

        for ifault in faults_list:
            faultname = ifault.name
            gfconf = self.faults[faultname]['method_parameters']['update_GFs']
            gfmethod = gfconf.get('method', 'homogeneous')  # Default method is 'homogeneous'
            gfopts = gfconf.get('options', {})
            for obsdata, vertical in zip(geodata, verticals):
                ifault.buildGFs(obsdata, vertical=vertical, slipdir='sd',
                                method=gfmethod, verbose=False, **gfopts)
            ifault.initializeslip()

            # Set options['force_recompute'] to False after first computation if present
            if 'force_recompute' in gfopts:
                gfopts['force_recompute'] = False

        # Assemble data and Green's functions for inversion
        poly_assembled = False  # Flag to check if the polynomial is assembled
        for ifault in faults_list:
            # Assemble data
            ifault.assembled(geodata, verbose=False)
            # Assemble Green's functions
            if not poly_assembled:
                ifault.assembleGFs(geodata, polys=polys, slipdir='sd', verbose=False, custom=False)
                poly_assembled = True
            else:
                ifault.assembleGFs(geodata, polys=nonpolys, slipdir='sd', verbose=False, custom=False)

        # Build covariance matrices for GPS/InSAR data
        for ifault in faults_list:
            # Note: verbose=True may cause errors
            ifault.assembleCd(geodata, verbose=False, add_prediction=None)

    def update_GFs_parameters(self, geodata, verticals, dataFaults=None, gfmethods=None):
        """
        Update the update_GFs method parameters for all faults.
        """
        allowed_methods = (
            'okada', 'Okada', 'OKADA', 'ok92', 'meade', 'Meade', 'MEADE',
            'edks', 'EDKS',
            'PSCMP', 'pscmp', 'PSGRN', 'psgrn',
            'EDCMP', 'edcmp', 'EDGRN', 'edgrn',
            'cutde', 'CUTDE',
            'empty'
        )
    
        if len(geodata) != len(verticals):
            msg = "Length of 'geodata' and 'verticals' should be the same."
            logger.error(msg)
            raise ValueError(msg)
        
        if gfmethods is not None and len(self.faultnames) != len(gfmethods):
            msg = "Length of faultnames and gfmethods should be the same."
            logger.error(msg)
            raise ValueError(msg)
        
        for i, fault_name in enumerate(self.faultnames):
            fault_parameters = self.faults[fault_name]
            method = gfmethods[i] if gfmethods is not None else None
            
            # If method is None, try to get from config or use default by patchType
            if method is None:
                ifault = self.faults_list[i]
                method = fault_parameters['method_parameters']['update_GFs'].get('method')
                if method is None:
                    if hasattr(ifault, 'patchType'):
                        if ifault.patchType == 'triangle':
                            method = 'cutde'
                        elif ifault.patchType == 'rectangle':
                            method = 'okada'
                        else:
                            msg = f"Unknown patchType '{ifault.patchType}' for fault '{fault_name}'"
                            logger.error(msg)
                            raise ValueError(msg)
                    else:
                        msg = f"Cannot determine default method for fault '{fault_name}' (missing patchType)"
                        logger.error(msg)
                        raise ValueError(msg)
            
            # Check whether the method is allowed or not
            if method not in allowed_methods:
                msg = (
                    f"Invalid Green's function method '{method}' for fault '{fault_name}'. "
                    f"Allowed methods are: {allowed_methods}"
                )
                logger.error(msg)
                raise ValueError(msg)
    
            fault_parameters['method_parameters']['update_GFs'] = {
                'geodata': geodata,
                'verticals': verticals,
                'dataFaults': dataFaults,
                'method': method,
                'options': fault_parameters['method_parameters']['update_GFs'].get('options', {})
            }
            # print(fault_name, fault_parameters['method_parameters']['update_GFs']['dataFaults'])

    def set_data_faults(self, dataFaults=None):

        all_faultnames = self.faultnames
        all_datanames = [d.name for d in self.geodata['data']]
        dataFaults = dataFaults if dataFaults is not None else self.geodata['faults']
        result = parse_data_faults(dataFaults, all_faultnames, all_datanames, param_name='dataFaults')
        self.geodata['faults'] = result

        if self.verbose:
            logger.info(f"Data faults set to: {self.geodata['faults']}")

    def set_alpha_faults(self, alphaFaults=None):

        all_faultnames = self.faultnames
        alphaFaults = alphaFaults if alphaFaults is not None else self.alpha['faults']
        result = parse_alpha_faults(alphaFaults, all_faultnames, param_name='alphaFaults')
        self.alpha['faults'] = result
        if None in result:
            self.alphaFaultsIndex = [0] * len(self.faultnames)
        else:
            # Create a dictionary to map fault names to their indices
            fault_index_map = {fault: idx for idx, sublist in enumerate(result) for fault in sublist}
            # Generate the alphaFaultsIndex based on the order of faultnames
            self.alphaFaultsIndex = [fault_index_map[fault] for fault in self.faultnames]
        self.alpha['faults_index'] = self.alphaFaultsIndex

        if self.verbose:
            logger.info(f"Alpha faults set to: {self.alphaFaults} with indices {self.alphaFaultsIndex}")

    def set_faults_method_parameters(self, method_parameters_dict):
        """
        Update the method parameters for all faults.
        """
        for fault_name, method_parameters in method_parameters_dict.items():
            if fault_name in self.faults:
                self.faults[fault_name]['method_parameters'].update(method_parameters)
            else:
                msg = f"Fault {fault_name} does not exist in the configuration."
                logger.error(msg)
                raise ValueError(msg)

    def _validate_laplacian_bounds(self):
        """
        Validate fault configurations for smoothing bounds.
        
        All faults' update_Laplacian bounds must only contain 'free' and 'locked', with length 4
        """
        
        # 1. Validate smoothing bounds
        valid_bounds = {'free', 'locked'}
        
        for fault_name, fault_config in self.faults.items():
            laplacian_config = fault_config.get('method_parameters', {}).get('update_Laplacian', {})
            bounds = laplacian_config.get('bounds')
            
            # Handle null bounds - expand to ['free', 'free', 'free', 'free']
            if bounds is None:
                laplacian_config['bounds'] = ['free', 'free', 'free', 'free']
                if self.verbose:
                    logger.info(f"Fault '{fault_name}': bounds is null, setting to ['free', 'free', 'free', 'free']")
            else:
                # Validate bounds format
                if not isinstance(bounds, list) or len(bounds) != 4:
                    msg = f"Fault '{fault_name}': bounds must be a list of length 4, got {bounds}"
                    logger.error(msg)
                    raise ValueError(msg)
                
                # Validate bounds values
                invalid_bounds = set(bounds) - valid_bounds
                if invalid_bounds:
                    msg = f"Fault '{fault_name}': bounds can only contain 'free' and 'locked', found invalid values: {list(invalid_bounds)}"
                    logger.error(msg)
                    raise ValueError(msg)
    
    def set_attributes(self, **kwargs):
        """
        Set object attributes based on key-value pairs in kwargs.
        
        Parameters:
        -----------
        **kwargs : dict
            Dictionary of attribute names and values to set
            
        Raises:
        -------
        ValueError
            If an unknown attribute is provided
        """
        # Set the attributes based on the key-value pairs in kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                msg = f"Unknown attribute '{key}'"
                logger.error(msg)
                raise ValueError(msg)

    def export_config(self, filename=None, format='yaml'):
        """
        Export the current config object's internal state to a file (supports yaml/json).

        This method is useful for recording and reproducing the fully expanded, normalized,
        and internally used configuration after all automatic augmentation and processing.

        Parameters
        ----------
        filename : str
            Output file path.
        format : str, optional
            Output format: 'yaml' (default) or 'json'.

        Notes
        -----
        - The exported file contains all key configuration attributes, including
        faultnames, faults, geodata, alpha, and other inversion parameters.
        - Numpy arrays and other non-serializable objects will be converted to lists or basic types.
        - geodata['data'] will be replaced by a list of dataset names.
        - For each fault and defaults, if method_parameters.update_GFs.geodata is a list of data objects,
        it will be replaced by a list of their .name attributes.
        - geodata['faults'] will only include 'defaults' and faults in self.faultnames, with 'defaults' first and the rest in self.faultnames order.
        - You can add more attributes to export_dict as needed for your workflow.

        Example
        -------
        >>> config = LinearInversionConfig(config_file='your_config.yml', ...)
        >>> config.export_config('final_config_record.yaml')
        """

        import json
        from collections import OrderedDict
        import copy

        def _to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return OrderedDict((k, _to_serializable(v)) for k, v in obj.items())
            elif isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            else:
                return obj

        # Deepcopy to avoid modifying self.geodata/self.faults
        geodata_export = copy.deepcopy(self.geodata)
        faults_export_src = copy.deepcopy(self.faults)

        # geodata['data'] -> dataset names
        if 'data' in geodata_export and isinstance(geodata_export['data'], list):
            geodata_export['data'] = [d.name for d in geodata_export['data']]

        # geodata['faults']: only output 'defaults' and self.faultnames, defaults first, then by self.faultnames order
        geodata_faults = geodata_export.get('faults', {})
        faults_ordered = OrderedDict()
        if 'defaults' in geodata_faults:
            faults_ordered['defaults'] = _to_serializable(geodata_faults['defaults'])
        for fname in self.faultnames:
            if fname in geodata_faults:
                faults_ordered[fname] = _to_serializable(geodata_faults[fname])
        geodata_export['faults'] = faults_ordered

        # faults: only output 'defaults' and self.faultnames, defaults first, then by self.faultnames order
        faults_export = OrderedDict()
        if 'defaults' in faults_export_src:
            defaults = faults_export_src['defaults']
            mp = defaults.get('method_parameters', {})
            if 'update_GFs' in mp:
                ugfs = mp['update_GFs']
                if 'geodata' in ugfs and isinstance(ugfs['geodata'], list):
                    ugfs['geodata'] = [d.name if hasattr(d, 'name') else d for d in ugfs['geodata']]
                mp['update_GFs'] = ugfs
                defaults['method_parameters'] = mp
            faults_export['defaults'] = _to_serializable(defaults)
        for fname in self.faultnames:
            if fname in faults_export_src:
                fval = faults_export_src[fname]
                mp = fval.get('method_parameters', {})
                if 'update_GFs' in mp:
                    ugfs = mp['update_GFs']
                    if 'geodata' in ugfs and isinstance(ugfs['geodata'], list):
                        ugfs['geodata'] = [d.name if hasattr(d, 'name') else d for d in ugfs['geodata']]
                    mp['update_GFs'] = ugfs
                    fval['method_parameters'] = mp
                faults_export[fname] = _to_serializable(fval)

        # Build export dictionary
        export_dict = OrderedDict()

        def add_field(key, value):
            export_dict[key] = _to_serializable(value)

        def add_optional(attr_name, export_key=None):
            if hasattr(self, attr_name):
                add_field(export_key or attr_name, getattr(self, attr_name))

        add_field('class', self.__class__.__name__)
        add_field('config_file', self.config_file)

        is_bayesian = 'bayesian' in self.__class__.__name__.lower()
        add_field('inversion_type', 'Bayesian' if is_bayesian else 'BLSE')

        # Add optional fields if they exist
        if is_bayesian:
            add_optional('bayesian_sampling_mode')
            add_optional('nonlinear_inversion')
            add_optional('slip_sampling_mode')
            add_optional('nchains')
            add_optional('chain_length')
        else:
            add_optional('slip_sampling_mode')

        add_field('faultnames', self.faultnames)
        add_field('use_bounds_constraints', self.use_bounds_constraints)
        add_field('use_rake_angle_constraints', self.use_rake_angle_constraints)
        add_field('rake_angle', self.rake_angle)
        add_field('use_euler_constraints', self.use_euler_constraints)
        add_field('shear_modulus', '{:.1f} GPa'.format(self.shear_modulus/1e9))
        add_field('moment_magnitude_threshold', self.moment_magnitude_threshold)
        add_field('magnitude_tolerance', self.magnitude_tolerance)
        # add_field('nonlinear_inversion', self.nonlinear_inversion)

        add_field('geodata', geodata_export)
        add_field('alpha', self.alpha)
        add_field('euler_constraints', self.euler_constraints)
        add_field('faults', faults_export)

        if format == 'yaml':
            import yaml

            class OrderedDumper(yaml.SafeDumper):
                pass

            def _dict_representer(dumper, data):
                return dumper.represent_mapping(
                    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                    data.items()
                )

            def _list_representer(dumper, data):
                return dumper.represent_sequence(
                    yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG,
                    data,
                    flow_style=True
                )

            OrderedDumper.add_representer(OrderedDict, _dict_representer)
            OrderedDumper.add_representer(list, _list_representer)
            filename = 'parsed_' + self.config_file if filename is None else filename
            with open(filename, 'w', encoding='utf-8') as f:
                yaml.dump(export_dict, f, allow_unicode=True, Dumper=OrderedDumper, default_flow_style=None, indent=2)
        elif format == 'json':
            filename = 'parsed_' + self.config_file if filename is None else filename
            # Transfer '.yml' to '.json'
            filename = filename.replace('.yml', '.json')
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_dict, f, ensure_ascii=False, indent=2)
        else:
            msg = "Only 'yaml' and 'json' formats are supported."
            logger.error(msg)
            raise ValueError(msg)

# EOF