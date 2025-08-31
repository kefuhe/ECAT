import yaml
import os
import numpy as np

# Load csi and its extensions
from csi.gps import gps
from csi.insar import insar
from ..multifaults_base import MyMultiFaultsInversion

# Import utility functions for parsing configuration updates
from .config_utils import parse_update, parse_initial_values
from .config_utils import parse_alpha_faults, parse_data_faults, parse_sigmas_config, parse_alpha_config
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

        # Parse alpha parameters
        fault_names = self.faultnames
        self.alpha = parse_alpha_config(self.alpha, faultnames=fault_names)

        # Update Green's function parameters
        self.update_GFs_parameters(self.geodata['data'], self.geodata['verticals'],
                                  self.geodata['faults'], gfmethods)

        # Parse sigmas parameters
        # n_datasets = len(self.geodata.get('data', []))
        data_names = [d.name for d in self.geodata.get('data', [])]
        self.geodata['sigmas'] = parse_sigmas_config(self.geodata['sigmas'], dataset_names=data_names)

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
                raise ValueError(f"Length of 'polys' list ({len(polys_config)}) must equal the number of datasets ({n_datasets})")
        elif isinstance(polys_config, (int, str, type(None))):
            # Single value format - expand to all datasets
            self.geodata['polys'] = [polys_config] * n_datasets
        else:
            raise ValueError(f"'polys' must be None, int, str, or list, got {type(polys_config)}")
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

        # Process faults configuration
        self._process_faults_config(config_data)
        
        # Set attributes
        self.set_attributes(**config_data)

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
                    print(f"Fault '{fault_name}' not found in config file. Using default parameters.")

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
            raise ValueError("Length of geodata and verticals should be the same.")
        
        if gfmethods is not None and len(self.faultnames) != len(gfmethods):
            raise ValueError("Length of faultnames and gfmethods should be the same.")
        
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
                            raise ValueError(f"Unknown patchType '{ifault.patchType}' for fault '{fault_name}'")
                    else:
                        raise ValueError(f"Cannot determine default method for fault '{fault_name}' (missing patchType)")
            
            # Check whether the method is allowed or not
            if method not in allowed_methods:
                raise ValueError(
                    f"Invalid Green's function method '{method}' for fault '{fault_name}'. "
                    f"Allowed methods are: {allowed_methods}"
                )
    
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
            print(f"Data faults set to: {self.geodata['faults']}")

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
            print(f"Alpha faults set to: {self.alphaFaults} with indices {self.alphaFaultsIndex}")

    def set_faults_method_parameters(self, method_parameters_dict):
        """
        Update the method parameters for all faults.
        """
        for fault_name, method_parameters in method_parameters_dict.items():
            if fault_name in self.faults:
                self.faults[fault_name]['method_parameters'].update(method_parameters)
            else:
                raise ValueError(f"Fault {fault_name} does not exist in the configuration.")

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
                    print(f"Fault '{fault_name}': bounds is null, setting to ['free', 'free', 'free', 'free']")
            else:
                # Validate bounds format
                if not isinstance(bounds, list) or len(bounds) != 4:
                    raise ValueError(f"Fault '{fault_name}': bounds must be a list of length 4, got {bounds}")
                
                # Validate bounds values
                invalid_bounds = set(bounds) - valid_bounds
                if invalid_bounds:
                    raise ValueError(f"Fault '{fault_name}': bounds can only contain 'free' and 'locked', "
                                f"found invalid values: {list(invalid_bounds)}")
    
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

        export_dict = OrderedDict([
            ('class', self.__class__.__name__),
            ('config_file', self.config_file),
            ('faultnames', self.faultnames),
            ('shear_modulus', self.shear_modulus),
            ('use_bounds_constraints', self.use_bounds_constraints),
            ('use_rake_angle_constraints', self.use_rake_angle_constraints),
            ('rake_angle', self.rake_angle),
            ('moment_magnitude_threshold', self.moment_magnitude_threshold),
            ('magnitude_tolerance', self.magnitude_tolerance),
            ('nonlinear_inversion', self.nonlinear_inversion),
            ('geodata', _to_serializable(geodata_export)),
            ('alpha', _to_serializable(self.alpha)),
            ('faults', faults_export),
        ])

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
            raise ValueError("Only 'yaml' and 'json' formats are supported.")

# EOF