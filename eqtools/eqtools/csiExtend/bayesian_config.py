import yaml
import os
import glob
import numpy as np

# Load csi and its extensions
from csi.gps import gps
from csi.insar import insar
# from .bayesian_multifaults_inversion import MyMultiFaultsInversion

# Import utility functions for parsing configuration updates
from .config_utils import parse_update, parse_initial_values


class BaseBayesianConfig:
    def __init__(self, config_file='default_config.yml', geodata=None, encoding='utf-8', verbose=False, parallel_rank=None):
        self.verbose = verbose
        self.parallel_rank = parallel_rank # Rank for parallel processing, if applicable
        if self.verbose and (self.parallel_rank is None or self.parallel_rank == 0):
            self._print_initialization_message()

        self.config_file = config_file
        self.nchains = 100 # Number of chains for BayesianMultiFaultsInversion
        self.chain_length = 50 # Length of each chain for BayesianMultiFaultsInversion
        self.geodata = {}
        self.lon0 = None
        self.lat0 = None
        self.faultnames = [] # List of fault names
        self.faults_list = [] # List of fault objects
        self.clipping_options = {} # Dictionary to store the clipping options
        self.data_sources = {} # Dictionary to store the data sources

    @property
    def sigmas(self):
        return self.geodata.get('sigmas')
    
    @sigmas.setter
    def sigmas(self, value):
        self.geodata['sigmas'] = value
        # Check 'update' is in sigmas and is a boolean
        if 'update' not in self.geodata['sigmas']:
            self.geodata['sigmas']['update'] = [True] * len(self.geodata.get('data', []))
        elif not all(isinstance(item, bool) for item in self.geodata['sigmas']['update']):
            raise ValueError("The 'update' parameter in sigmas must be a boolean")
        
        # Check 'log_scaled' is in sigmas and is a boolean
        if 'log_scaled' not in self.geodata['sigmas']:
            self.geodata['sigmas']['log_scaled'] = True
        elif not isinstance(self.geodata['sigmas']['log_scaled'], bool):
            raise ValueError("The 'log_scaled' parameter in sigmas must be a boolean")

    @property
    def dataFaults(self):
        return self.geodata['faults']
    
    @dataFaults.setter
    def dataFaults(self, value):
        if value is None:
            self.geodata['faults'] = [self.faultnames] * len(self.geodata.get('data', []))
        elif isinstance(value, list):
            # Flatten the list of lists to a single list if sublist is a list
            flattened_dataFaults = [item for sublist in value for item in (sublist if isinstance(sublist, list) else [sublist])]
            
            # Check flattened_dataFaults is subset of self.faultnames
            if not set(flattened_dataFaults).issubset(set(self.faultnames + [None])):
                raise ValueError("The dataFaults must be a subset of the faultnames in self.multifaults")
            
            # Ensure the list is at most two levels deep
            for sublist in value:
                if isinstance(sublist, list):
                    if any(isinstance(item, list) for item in sublist):
                        raise ValueError("The dataFaults list must be at most two levels deep")
                    if all(item is None for item in sublist) or all(item is not None for item in sublist):
                        continue
                    else:
                        raise ValueError("The second level lists must either contain only None or no None")
            
            # Replace None with faultnames in geodata['faults']
            self.geodata['faults'] = [[item if item is not None else self.faultnames for item in sublist] if isinstance(sublist, list) else (sublist if sublist is not None else self.faultnames) for sublist in value]
        else:
            raise ValueError("dataFaults must be a list or None")

    def load_data(self, data_type):
        data_config = self.data_sources[data_type]
        data_files = []

        # Check if specific files are provided
        if 'files' in data_config:
            data_files.extend(data_config['files'])
        else:
            # Use file pattern to match files
            data_files.extend(glob.glob(os.path.join(data_config['directory'], data_config['file_pattern'])))

        for data_file in data_files:
            assert self.lon0 is not None and self.lat0 is not None, f"lon0 and lat0 must be set to read {data_type} data"
            data_name = os.path.basename(os.path.splitext(data_file)[0])
            if data_type == 'gps':
                data_instance = gps(name=data_name, utmzone=None, ellps='WGS84', lon0=self.lon0, lat0=self.lat0, verbose=True)
                data_instance.read_from_enu(data_file, factor=1., minerr=1., header=1, checkNaNs=True)
            elif data_type == 'insar':
                data_file_prefix = os.path.splitext(data_file)[0]
                data_instance = insar(data_name, lon0=self.lon0, lat0=self.lat0, verbose=True)
                data_instance.read_from_varres(data_file_prefix, cov=True)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            self.geodata['data'].append(data_instance)

    def load_all_data(self):
        if not self.geodata.get('data', []):
            for data_type in self.data_sources:
                self.load_data(data_type)
            if not self.geodata['data']:
                raise ValueError("Failed to load any geodata files.")
        else:
            print("Geodata already provided in configuration.")

    def _print_initialization_message(self):
        print("---------------------------------")
        print("---------------------------------")
        print(f"Initializing {self.__class__.__name__} object")

    def list_attributes(self):
        """
        List all attributes of the configuration
        """
        return [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

    def _update_geodata(self, geodata):
        if 'data' not in self.geodata or self.geodata['data'] is None:
            self.geodata['data'] = geodata if geodata else []
        if not self.geodata['data']:
            assert self.lon0 is not None and self.lat0 is not None, "lon0 and lat0 must be set to read geodata files with no geodata provided"
            self.load_all_data()

    def _set_geodata_attributes(self):
        '''
        To trigger the property setters for sigmas and dataFaults
        '''
        self.sigmas = self.geodata.get('sigmas', {})
        self.dataFaults = self.geodata.get('faults', None)

    def _validate_verticals(self, verticals=None):
        if 'verticals' not in self.geodata or self.geodata['verticals'] is None:
            self.geodata['verticals'] = verticals if verticals else True
        verticals = self.geodata.get('verticals', None)
        data_length = len(self.geodata['data'])
        if isinstance(verticals, list):
            if len(verticals) != data_length:
                raise ValueError(f"Length of 'verticals' list ({len(verticals)}) does not match length of 'data' ({data_length})")
        elif isinstance(verticals, bool):
            self.geodata['verticals'] = [verticals] * data_length
        else:
            raise ValueError("'verticals' must be either a list or a boolean")

    def _select_data_sets(self):
        data_verticals_dict = {d.name: v for d, v in zip(self.geodata['data'], self.geodata['verticals'])}
        if self.clipping_options.get('enabled', False):
            methods = self.clipping_options.get('methods', [])
            for method_config in methods:
                method = method_config.get('method', None)
                if method == 'lon_lat_range':
                    lon_lat_range = method_config.get('lon_lat_range', None)
                    if lon_lat_range is None:
                        raise ValueError("Clipping method 'lon_lat_range' requires 'lon_lat_range' to be set")
                    for data in self.geodata['data']:
                        if data.dtype == 'insar':
                            data.select_pixels(*lon_lat_range)
                        elif data.dtype == 'gps':
                            data.select_stations(*lon_lat_range)
                            if not data_verticals_dict[data.name]:
                                data.vel_enu[:, -1] = np.nan
                                data.buildCd(direction='en')
                            else:
                                data.buildCd(direction='enu')
                elif method == 'distance_to_fault':
                    distance_to_fault = method_config.get('distance_to_fault', None)
                    faults = self.faults_list
                    if distance_to_fault is None or not faults:
                        raise ValueError("Clipping method 'distance_to_fault' requires 'distance_to_fault' and non-empty 'faults_list' to be set")
                    for data in self.geodata['data']:
                        if data.dtype == 'insar':
                            data.reject_pixels_fault(distance_to_fault, faults)
                        elif data.dtype == 'gps':
                            # No clipping for GPS data
                            continue
                else:
                    raise ValueError(f"Unsupported clipping method: {method}")


class explorefaultConfig(BaseBayesianConfig):
    def __init__(self, config_file=None, geodata=None, verbose=False, parallel_rank=None):
        super().__init__(config_file=config_file, geodata=geodata, verbose=verbose, parallel_rank=parallel_rank)
        self.bounds = {}
        self.initial = {} # Initial parameters for each fault
        self.fixed_params = {} # Fixed parameters for each fault
        self.nfaults = 1 # Number of faults to be explored 
        self.faultnames = [f'fault_{i}' for i in range(self.nfaults)]
        self.slip_sampling_mode = 'mag_rake'

        if config_file:
            self.load_config(config_file, geodata=geodata)
        
        # Parse the 'update' parameter in sigmas
        # Added by kfhe at 07/11/2025
        n_datasets = len(self.geodata.get('data', []))
        data_names = [d.name for d in self.geodata.get('data', [])]
        # parsed_update = parse_update(self.geodata['sigmas'], n_datasets, param_name='update', 
        #                                 dataset_names=data_names)
        # self.geodata['sigmas']['update'] = parsed_update
        # print(f"Parsed update for sigmas: {self.geodata['sigmas']['update']}")
        
        # Parse initial values using the new generic function
        parsed_initial_values = parse_initial_values(
            self.geodata['sigmas'],
            n_datasets,
            param_name='values',  # initial_value or 'values'
            default_value=0.01,
            min_value=0.0,
            dataset_names=data_names
        )
        self.geodata['sigmas']['values'] = parsed_initial_values
        # print(f"Parsed initial values for sigmas: {self.geodata['sigmas']['values']}")

    def load_config(self, config_file, geodata=None):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.bounds = config.get('bounds', {})
        self.initial = config.get('initial', {})
        self.fixed_params = config.get('fixed_params', {})
        self.nfaults = config.get('nfaults', 1)
        self.faultnames = [f'fault_{i}' for i in range(self.nfaults)]
        self.slip_sampling_mode = config.get('slip_sampling_mode', 'mag_rake')
        self.clipping_options = config.get('clipping_options', {})
        self.geodata = config.get('geodata', {})
        lon_lat_0 = config.get('lon_lat_0', None)
        if lon_lat_0:
            self.lon0, self.lat0 = lon_lat_0
        self.data_sources = config.get('data_sources', {})

        self._update_geodata(geodata)
        self._validate_verticals()

        # Parse the 'update' parameter in sigmas
        n_datasets = len(self.geodata.get('data', []))
        data_names = [d.name for d in self.geodata.get('data', [])]
        parsed_update = parse_update(self.geodata['sigmas'], n_datasets, param_name='update', 
                                        dataset_names=data_names)
        self.geodata['sigmas']['update'] = parsed_update
        # print(f"Parsed update for sigmas: {self.geodata['sigmas']['update']}")

        self._set_geodata_attributes()
        self.update_polys_estimate_and_boundaries()
        self._select_data_sets()

    def update_polys_estimate_and_boundaries(self, datas=None):
        if self.geodata.get('polys', {}).get('enabled', False):
            if datas is not None:
                if type(datas) is not list:
                    datas = [datas]
                self.geodata['polys']['estimate'] = [d.name for d in datas]
            else:
                datas = self.geodata.get('polys', {}).get('estimate', [])

            insar_data = [d for d in self.geodata.get('data', []) if d.dtype == 'insar']
            default_bounds = self.geodata['polys']['boundaries'].get('defaults', None)
            
            if not datas:
                datas = [d.name for d in insar_data]
                self.geodata['polys']['estimate'] = datas
            
            for data in insar_data:
                if data.name in datas:
                    boundary_key = data.name
                    if boundary_key not in self.geodata['polys']['boundaries']:
                        if default_bounds:
                            self.geodata['polys']['boundaries'][boundary_key] = default_bounds
                        else:
                            raise ValueError(f"Bounds for {boundary_key} must be set as there is no default")
                else:
                    raise ValueError(f"Data name {data.name} is not in the estimate list")
    
    @BaseBayesianConfig.sigmas.setter
    def sigmas(self, value):
        super(explorefaultConfig, self.__class__).sigmas.fset(self, value)
        self.geodata['sigmas']['ndatas'] = self.ndatas
        self.geodata['sigmas']['names'] = ['sigma_{}'.format(i) for i in range(self.ndatas)]

        sigma_names = self.geodata['sigmas']['names']
        bounds = self.geodata['sigmas']['bounds']
    
        # Check if 'defaults' is in bounds
        if 'defaults' not in bounds:
            # Check if all sigmas are in bounds
            if not set(sigma_names).issubset(bounds.keys()):
                raise ValueError("The bounds dictionary must have keys for all sigmas or a 'defaults' key")
        else:
            # Fill in the missing sigmas with the defaults
            defaults = bounds['defaults']
            for name in sigma_names:
                if name not in bounds:
                    bounds[name] = defaults
    
    @property
    def ndatas(self):
        return len(self.geodata.get('data', []))



class BayesianMultiFaultsInversionConfig(BaseBayesianConfig):
    def __init__(self, config_file='default_config.yml', multifaults=None, geodata=None, 
                 verticals=None, polys=None, dataFaults=None, alphaFaults=None, faults_list=None,
                 gfmethods=None, slip_sampling_mode='ss_ds', bayesian_sampling_mode='SMC_F_J', encoding='utf-8', 
                 verbose=False, parallel_rank=None, **kwargs):
        """
        Initialize the BayesianMultiFaultsInversionConfig object.
        """
        from .bayesian_multifaults_inversion import MyMultiFaultsInversion

        super().__init__(config_file, geodata=geodata, verbose=verbose, parallel_rank=parallel_rank)
        self.multifaults = multifaults # Multifaults object for the inversion
        self.bayesian_sampling_mode = bayesian_sampling_mode # Bayesian sampling mode, default is SMC_F_J, other options are FULLSMC
        self.use_bounds_constraints = True # Use bounds constraints for the inversion, only for SMC_F_J mode
        self.use_rake_angle_constraints = True # Use rake angle constraints for the inversion, only for SMC_F_J mode
        self.GLs = None
        self.moment_magnitude_threshold = None
        self.patch_areas = None
        self.shear_modulus = 3.0e10
        self.magnitude_tolerance = None 
        self.nonlinear_inversion = False
        self.slip_sampling_mode = slip_sampling_mode
        self.rake_angle = None # Only used when slip_sampling_mode is 'rake_fixed'
        self.faults = {}  # Dictionary to store the fault parameters

        # Initialize alpha with default values
        self.alpha = {
            'enabled': True,
            'update': True,
            'initial_value': 0.0,
            'log_scaled': True,
            'faults': None
        }

        assert multifaults or faults_list, "Either multifaults or faults_list must be provided"
        if multifaults is not None:
            self.faultnames = multifaults.faultnames
            self.faults_list = multifaults.faults
        else:
            self.faults_list = faults_list
            self.faultnames = [fault.name for fault in faults_list]
        
        # Load the configuration from a file
        if config_file is not None:
            self.load_from_file(config_file, encoding=encoding)
        
        self.shear_modulus = float(self.shear_modulus)

        # Set the attributes based on the key-value pairs in kwargs
        self.set_attributes(**kwargs)

        # Set the geometry update to False for faults with shared info
        # True is to calculate green's functions for the fault each iteration
        # [0, 0] is no sampling for the fault geometry
        # Added by kfhe at 03/25/2025
        for ifault in self.faults_list:
            if hasattr(ifault, 'use_shared_info') and ifault.use_shared_info:
                self.faults[ifault.name]['geometry'] = {
                    'update': True,
                    'sample_positions': [0, 0]
                }

        # Set alpha['update'] to False if alpha is not enabled
        if not self.alpha['enabled']:
            self.alpha['update'] = False
            self.alpha['initial_value'] = 0.0

        # Enforce the slip sampling mode as 'magnitude_rake' in F-J inversion
        if self.bayesian_sampling_mode == 'SMC_F_J':
            self.slip_sampling_mode = 'ss_ds'

        # Set geodata attributes
        # Set the data of geodata
        self._update_geodata(geodata)
        # Validate the verticals
        self._validate_verticals(verticals)
        # Select the data sets based on the clipping options
        self._select_data_sets()
        # Validate the polys
        self._validate_polys(polys)

        # Update the GFs parameters based on the geodata and verticals
        self.update_GFs_parameters(self.geodata['data'], self.geodata['verticals'], self.geodata['faults'], gfmethods)

        # Set the dataFaults based on the data configuration
        self.set_data_faults(dataFaults)
        # Set the alphaFaults based on the alpha configuration
        self.set_alpha_faults(alphaFaults)

        # Parse the 'update' parameter in sigmas
        # Added by kfhe at 07/11/2025
        n_datasets = len(self.geodata.get('data', []))
        data_names = [d.name for d in self.geodata.get('data', [])]
        parsed_update = parse_update(self.geodata['sigmas'], n_datasets, param_name='update', 
                                        dataset_names=data_names)
        self.geodata['sigmas']['update'] = parsed_update
        # print(f"Parsed update for sigmas: {self.geodata['sigmas']['update']}")

        # Parse initial values using the new generic function
        parsed_initial_values = parse_initial_values(
            self.geodata['sigmas'],
            n_datasets,
            param_name='initial_value',  # initial_value or 'values'
            default_value=0.01,
            min_value=0.0,
            dataset_names=data_names
        )
        self.geodata['sigmas']['initial_value'] = parsed_initial_values
        # print(f"Parsed initial values for sigmas: {self.geodata['sigmas']['initial_value']}")

        if self.clipping_options.get('enabled', False):
            self._initialize_faults_and_assemble_data()
        if multifaults is None:
            self._initialize_faults_and_assemble_data()
            verbose = self.verbose and (self.parallel_rank is None or self.parallel_rank == 0)
            multifaults = MyMultiFaultsInversion('myfault', self.faults_list, verbose=verbose)
            multifaults.assembleGFs() # assemble the Green's functions because the data is already assembled
            self.multifaults = multifaults
    
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
        self.alpha['faults'] = value
    
    def load_from_file(self, config_file, encoding='utf-8'):
        """
        Load the configuration from a file and ensure defaults are always present with required attributes.
        """
        # Load the configuration from a file
        with open(config_file, 'r', encoding=encoding) as f:
            config_data = yaml.safe_load(f)
    
        # Ensure defaults exist and have required attributes, which is added by kfhe at 03/25/2025
        if 'faults' not in config_data:
            config_data['faults'] = {}
        if 'defaults' not in config_data['faults']:
            config_data['faults']['defaults'] = {}
        if 'geometry' not in config_data['faults']['defaults']:
            config_data['faults']['defaults']['geometry'] = {
                'update': False,
                'sample_positions': [0, 0]
            }
        else:
            # Ensure geometry has required attributes
            config_data['faults']['defaults']['geometry'].setdefault('update', False)
            config_data['faults']['defaults']['geometry'].setdefault('sample_positions', [0, 0])
    
        # Set lon0 and lat0 based on the configuration file
        lon_lat_0 = config_data.get('lon_lat_0', None)
        if lon_lat_0 is not None:
            self.lon0 = lon_lat_0[0]
            self.lat0 = lon_lat_0[1]
        config_data.pop('lon_lat_0', None)
    
        # Handle alpha configuration
        if 'alpha' in config_data:
            self.alpha.update(config_data['alpha'])
            config_data.pop('alpha')  # Remove alpha from config_data to avoid overwriting the alpha attribute
    
        # Handle the default parameters
        if 'slip_sampling_mode' in config_data:
            self.slip_sampling_mode = config_data['slip_sampling_mode']
            if self.slip_sampling_mode == 'rake_fixed':
                if 'rake_angle' in config_data:
                    self.rake_angle = config_data['rake_angle']
                else:
                    raise ValueError("When slip_sampling_mode is 'rake_fixed', a 'rake_angle' must be provided in the config file.")
            elif 'rake_angle' in config_data:
                if self.verbose and (self.parallel_rank is None or self.parallel_rank == 0):
                    print("Warning: 'rake_angle' is provided but 'slip_sampling_mode' is not 'rake_fixed'. 'rake_angle' will be ignored.")
    
        # Get the default parameters
        default_fault_parameters = config_data['faults']['defaults']
    
        # Handle the faults
        for fault_name, fault_parameters in config_data['faults'].items():
            if fault_name == 'defaults':
                continue
    
            # Use the default parameters if fault_parameters is None
            if fault_parameters is None:
                config_data['faults'][fault_name] = default_fault_parameters.copy()
            else:
                # Combine the default parameters with the fault parameters
                merged_parameters = {**default_fault_parameters, **fault_parameters}
    
                # Special handling for 'geometry'
                merged_geometry = {**default_fault_parameters.get('geometry', {}), **fault_parameters.get('geometry', {})}
                merged_parameters['geometry'] = merged_geometry
    
                # Special handling for 'method_parameters'
                merged_method_parameters = default_fault_parameters.get('method_parameters', {}).copy()
                for method_name, method_params in fault_parameters.get('method_parameters', {}).items():
                    if method_name in merged_method_parameters:
                        # Check if 'method' is in the method parameters
                        # If it is, then the method parameters are to be replaced
                        if 'method' in merged_method_parameters[method_name] or 'method' in method_params:
                            merged_method_parameters[method_name] = method_params
                        else:
                            merged_method_parameters[method_name] = {**merged_method_parameters[method_name], **method_params}
                    else:
                        merged_method_parameters[method_name] = method_params
                merged_parameters['method_parameters'] = merged_method_parameters
    
                config_data['faults'][fault_name] = merged_parameters
    
            self._process_fixed_nodes(config_data['faults'][fault_name])
            self.faults[fault_name] = config_data['faults'][fault_name]
    
        # Ensure all faults in faultnames are configured
        for fault_name in self.faultnames:
            if fault_name not in self.faults:
                config_data['faults'][fault_name] = default_fault_parameters.copy()
                self.faults[fault_name] = config_data['faults'][fault_name]
                if self.verbose and (self.parallel_rank is None or self.parallel_rank == 0):
                    print(f"Fault '{fault_name}' not found in config file. Using default parameters.")
    
        # Remove faults not in faultnames from config_data and self.faults
        for fault_name in list(config_data['faults'].keys()):
            if fault_name != 'defaults' and fault_name not in self.faultnames:
                del config_data['faults'][fault_name]
                if self.verbose and (self.parallel_rank is None or self.parallel_rank == 0):
                    print(f"Fault '{fault_name}' found in config file but not in faultnames. Removed from configuration.")
    
        for fault_name in list(self.faults.keys()):
            if fault_name not in self.faultnames:
                del self.faults[fault_name]
                if self.verbose and (self.parallel_rank is None or self.parallel_rank == 0):
                    print(f"Fault '{fault_name}' found in self.faults but not in faultnames. Removed from configuration.")
    
        # Set the attributes based on the key-value pairs in config_data
        self.set_attributes(**config_data)

        # Validate fault configurations
        self._validate_fault_configurations()

    def _process_fixed_nodes(self, fault_parameters):
        # Make sure the fixed nodes are in the correct format
        for method_parameters in fault_parameters.get('method_parameters', {}).values():
            if 'fixed_nodes' in method_parameters and method_parameters['fixed_nodes'] is not None:
                fixed_nodes = []
                for node in method_parameters['fixed_nodes']:
                    if isinstance(node, list):
                        fixed_nodes.extend(range(node[0], node[1] + 1))
                    else:
                        fixed_nodes.append(node)
                method_parameters['fixed_nodes'] = fixed_nodes

    def _initialize_faults_and_assemble_data(self, faults_list=None, geodata=None):
        """
        Setup faults by building Green's functions, assembling data and Green's functions for inversion,
        and building covariance matrices for GPS/InSAR data.
        """
        # --------------------------------Build GreenFns-----------------------------------------#
        faults_list = faults_list or self.faults_list
        geodata = self.geodata['data'] or geodata
        verticals = self.geodata['verticals']
        polys = self.geodata['polys']
        nonpolys = [None] * len(geodata)

        for ifault in faults_list:
            faultname = ifault.name
            gfmethod = self.faults[faultname]['method_parameters']['update_GFs']['method']
            for obsdata, vertical in zip(geodata, verticals):
                ifault.buildGFs(obsdata, vertical=vertical, slipdir='sd', method=gfmethod, verbose=False)
            ifault.initializeslip()

        # ----------------------Assemble data and GreenFns for Inversion------------------------#
        poly_assembled = False  # flag to check if the polynomial is assembled
        for ifault in faults_list:
            # assemble data
            ifault.assembled(geodata, verbose=False)
            # assemble GreensFns
            if not poly_assembled:
                ifault.assembleGFs(geodata, polys=polys, slipdir='sd', verbose=False, custom=False)
                poly_assembled = True
            else:
                ifault.assembleGFs(geodata, polys=nonpolys, slipdir='sd', verbose=False, custom=False)

        # --------------------------Build Covariance Matrix for GPS/InSAR data-------------------#
        # assemble data Covariance matrices, You should assemble the Green's function matrix first
        for ifault in faults_list:
            # Bug: the verbose may lead to the error if it is set to True
            ifault.assembleCd(geodata, verbose=False, add_prediction=None)

    def set_attributes(self, **kwargs):
        # Set the attributes based on the key-value pairs in kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown attribute '{key}'")
    
    def set_faults_method_parameters(self, method_parameters_dict):
        """
        Update the method parameters for all faults.
        """
        for fault_name, method_parameters in method_parameters_dict.items():
            if fault_name in self.faults:
                self.faults[fault_name]['method_parameters'].update(method_parameters)
            else:
                raise ValueError(f"Fault {fault_name} does not exist in the configuration.")

    def update_GFs_parameters(self, geodata, verticals, dataFaults=None, gfmethods=None):
        """
        Update the update_GFs method parameters for all faults.
        """
        if len(geodata) != len(verticals):
            raise ValueError("Length of geodata and verticals should be the same.")
        
        if gfmethods is not None and len(self.faultnames) != len(gfmethods):
            raise ValueError("Length of faultnames and gfmethods should be the same.")
        
        for i, fault_name in enumerate(self.faultnames):
            fault_parameters = self.faults[fault_name]
            method = gfmethods[i] if gfmethods is not None else None
            
            if method is None:
                ifault = self.faults_list[i]
                method = fault_parameters['method_parameters']['update_GFs'].get('method')
                
                if method is None:
                    if ifault.patchType == 'triangle':
                        method = 'cutde'
                    elif ifault.patchType == 'rectangle':
                        method = 'okada'
                    else:
                        raise ValueError("Unknown patchType")
            
            fault_parameters['method_parameters']['update_GFs'] = {
                'geodata': geodata,
                'verticals': verticals,
                'dataFaults': dataFaults,
                'method': method
            }

    def set_data_faults(self, dataFaults=None):
        if dataFaults is not None:
            self.dataFaults = dataFaults
        elif self.dataFaults is None:
            self.dataFaults = [self.faultnames]*len(self.geodata['data'])
        
        # Check if self.dataFaults is a list
        if not isinstance(self.dataFaults, list):
            raise ValueError("self.dataFaults must be a list")
        
        # Check the length of self.dataFaults
        if len(self.dataFaults) != len(self.geodata['data']):
            raise ValueError("Length of self.dataFaults should be equal to the length of geodata")
        
        # Check contents of self.dataFaults
        # flatten the list of lists to a single list if sublist is a list
        self.validate_faults(check_dataFaults=True)

    def validate_faults(self, check_dataFaults=False, check_alphaFaults=False):
        """
        Validate the dataFaults or alphaFaults.
        """

        if check_dataFaults:
            # Flatten the list of lists to a single list if sublist is a list
            flattened_dataFaults = [item for sublist in self.dataFaults for item in (sublist if isinstance(sublist, list) else [sublist])]
            
            # Check flattened_dataFaults is subset of self.faultnames
            if not set(flattened_dataFaults).issubset(set(self.faultnames + [None])):
                raise ValueError("The dataFaults must be a subset of the faultnames in self.multifaults")
        
        if check_alphaFaults:
            # Check if self.alphaFaults does not contain None and all other items are lists
            if None not in self.alphaFaults:
                if not all(isinstance(item, list) for item in self.alphaFaults):
                    raise ValueError("All items in self.alphaFaults must be lists")

                # Flatten the list of lists
                flattened_faults = [fault for sublist in self.alphaFaults for fault in sublist]
                
                # Check if self.alphaFaults contains duplicate items
                if len(flattened_faults) != len(set(flattened_faults)):
                    raise ValueError("self.alphaFaults cannot contain duplicate items")

                # Check if the union of self.alphaFaults equals self.faultnames
                if set(flattened_faults) != set(self.faultnames):
                    raise ValueError("The union of self.alphaFaults must equal the items in self.faultnames")

    def set_alpha_faults(self, alphaFaults=None):
        if alphaFaults is not None:
            self.alphaFaults = alphaFaults
        elif self.alphaFaults is None:
            self.alphaFaults = [None]

        # Check if self.alphaFaults is a list
        if not isinstance(self.alphaFaults, list):
            raise ValueError("self.alphaFaults must be a list")

        # Check if self.alphaFaults contains more than one None
        if None in self.alphaFaults and len(self.alphaFaults) > 1:
            raise ValueError("self.alphaFaults cannot contain None and other items simultaneously")

        # Check if self.alphaFaults does not contain None and all other items are lists
        self.validate_faults(check_alphaFaults=True)
        
        # 
        self.alpha['faults'] = self.alphaFaults
        if None in self.alphaFaults:
            self.alphaFaultsIndex = [0] * len(self.faultnames)
        else:
            # Create a dictionary to map fault names to their indices
            fault_index_map = {fault: idx for idx, sublist in enumerate(self.alphaFaults) for fault in sublist}
            # Generate the alphaFaultsIndex based on the order of faultnames
            self.alphaFaultsIndex = [fault_index_map[fault] for fault in self.faultnames]
        self.alpha['faults_index'] = self.alphaFaultsIndex

    def _validate_fault_configurations(self):
        """
        Validate fault configurations for smoothing bounds and geometry sample positions.
        
        1. All faults' update_Laplacian bounds must only contain 'free' and 'locked', with length 4
        2. All geometry-updating faults' sample_positions must form a complete 0-n sequence union
        """
        
        # 1. Validate smoothing bounds
        valid_bounds = {'free', 'locked'}
        
        for fault_name, fault_config in self.faults.items():
            laplacian_config = fault_config.get('method_parameters', {}).get('update_Laplacian', {})
            bounds = laplacian_config.get('bounds')
            
            # Handle null bounds - expand to ['free', 'free', 'free', 'free']
            if bounds is None:
                laplacian_config['bounds'] = ['free', 'free', 'free', 'free']
                if self.verbose and (self.parallel_rank is None or self.parallel_rank == 0):
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
        
        # 2. Validate geometry sample positions
        geometry_updating_faults = []
        all_sample_positions = set()
        
        for fault_name, fault_config in self.faults.items():
            geometry_config = fault_config.get('geometry', {})
            update_geometry = geometry_config.get('update', False)
            
            if update_geometry:
                sample_positions = geometry_config.get('sample_positions', [0, 0])
                
                # Validate sample_positions format
                if not isinstance(sample_positions, list) or len(sample_positions) != 2:
                    raise ValueError(f"Fault '{fault_name}': sample_positions must be a list of length 2, "
                                   f"got {sample_positions}")
                
                start, end = sample_positions
                if not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError(f"Fault '{fault_name}': sample_positions must contain integers, "
                                   f"got {sample_positions}")
                
                if start > end:
                    raise ValueError(f"Fault '{fault_name}': sample_positions start ({start}) "
                                   f"cannot be greater than end ({end})")
                
                # Check if this fault actually updates geometry (end > start)
                if end > start:
                    geometry_updating_faults.append(fault_name)
                    # Add all positions in the range to the set
                    all_sample_positions.update(range(start, end))
        
        # Validate that sample_positions form a complete 0-n sequence
        if geometry_updating_faults:
            if not all_sample_positions:
                raise ValueError("No valid geometry sample positions found in geometry-updating faults")
            
            min_pos = min(all_sample_positions)
            max_pos = max(all_sample_positions)
            expected_positions = set(range(min_pos, max_pos + 1))
            
            # Check if starting from 0
            if min_pos != 0:
                raise ValueError(f"Geometry sample positions must start from 0, but starts from {min_pos}")
            
            # Check if positions form a complete sequence
            missing_positions = expected_positions - all_sample_positions
            if missing_positions:
                raise ValueError(f"Geometry sample positions must form a complete sequence, "
                               f"missing positions: {sorted(missing_positions)}")
            
            # Check if n > 0 (at least one position)
            if max_pos == 0:
                raise ValueError("Geometry sample positions must have n > 0 (at least position 0 and 1)")
            
            if self.verbose and (self.parallel_rank is None or self.parallel_rank == 0):
                print(f"Geometry validation passed: {len(geometry_updating_faults)} faults updating geometry, "
                      f"sample positions [0, {max_pos+1}) complete")


class BoundLSEInversionConfig(BaseBayesianConfig):
    def __init__(self, config_file='default_config.yml', multifaults=None, geodata=None, 
                 verticals=None, polys=None, dataFaults=None, alphaFaults=None, faults_list=None,
                 gfmethods=None, encoding='utf-8', verbose=False, **kwargs):
        """
        Initialize the BayesianMultiFaultsInversionConfig object.
        """
        from .bayesian_multifaults_inversion import MyMultiFaultsInversion

        super().__init__(config_file, geodata=geodata, verbose=verbose)
        self.multifaults = multifaults # Multifaults object for the inversion
        self.use_bounds_constraints = True # Use bounds constraints for the inversion, only for SMC_F_J mode
        self.use_rake_angle_constraints = True # Use rake angle constraints for the inversion, only for SMC_F_J mode
        self.GLs = None
        self.moment_magnitude_threshold = None
        self.patch_areas = None
        self.shear_modulus = 3.0e10
        self.magnitude_tolerance = None 
        self.nonlinear_inversion = False
        self.rake_angle = None # Only used when slip_sampling_mode is 'rake_fixed'
        self.faults = {}  # Dictionary to store the fault parameters

        # Initialize alpha with default values
        self.alpha = {
            'enabled': True,
            'update': False,
            'initial_value': 0.0,
            'log_scaled': True,
            'faults': None
        }

        assert multifaults or faults_list, "Either multifaults or faults_list must be provided"
        if multifaults is not None:
            self.faultnames = multifaults.faultnames
            self.faults_list = multifaults.faults
        else:
            self.faults_list = faults_list
            self.faultnames = [fault.name for fault in faults_list]
        
        # Load the configuration from a file
        if config_file is not None:
            self.load_from_file(config_file, encoding=encoding)
        
        self.shear_modulus = float(self.shear_modulus)

        # Set the attributes based on the key-value pairs in kwargs
        self.set_attributes(**kwargs)

        # Set geodata attributes
        # Set the data of geodata
        self._update_geodata(geodata)
        # Validate the verticals
        self._validate_verticals(verticals)
        # Select the data sets based on the clipping options
        self._select_data_sets()
        # Validate the polys
        self._validate_polys(polys)
        # Validate the Laplacian bounds
        self._validate_laplacian_bounds()

        # Update the GFs parameters based on the geodata and verticals
        self.update_GFs_parameters(self.geodata['data'], self.geodata['verticals'], self.geodata['faults'], gfmethods)

        # Set the dataFaults based on the data configuration
        self.set_data_faults(dataFaults)
        # Set the alphaFaults based on the alpha configuration
        self.set_alpha_faults(alphaFaults)

        if self.clipping_options.get('enabled', False):
            self._initialize_faults_and_assemble_data()

        self._initialize_faults_and_assemble_data()
        # Initialize the faults and assemble the data
        if multifaults is None:
            multifaults = MyMultiFaultsInversion('myfault', self.faults_list, verbose=False)
            self.multifaults = multifaults
        multifaults.assembleGFs() # assemble the Green's functions because the data is already assembled
    
    def _validate_polys(self, polys):
        """
        Validate and initialize the 'polys' parameter in the geodata configuration.
    
        If the user sets 'polys' to None or does not provide this parameter, it indicates that
        polynomial corrections will not be estimated for the geodata. The method ensures that
        the 'polys' parameter is properly initialized and consistent with the length of the geodata.
    
        Parameters:
        polys (list, int, str, or None): The 'polys' parameter provided by the user. It can be:
            - None: Indicates no polynomial corrections will be estimated.
            - A list: Specifies the polynomial correction settings for each geodata.
            - An integer or string: A single value applied to all geodata.
    
        Raises:
        ValueError: If the length of the 'polys' list does not match the length of the geodata.
        """
        if 'polys' not in self.geodata or self.geodata['polys'] is None:
            self.geodata['polys'] = polys if polys else []
        if not self.geodata['polys']:
            for data in self.geodata['data']:
                if data.dtype == 'insar':
                    self.geodata['polys'].append(None)  # Indicates no polynomial correction for this data
                else:
                    self.geodata['polys'].append(None)
    
        if isinstance(self.geodata['polys'], list):
            if len(self.geodata['polys']) != len(self.geodata['data']):
                raise ValueError("Length of 'polys' list must be equal to the length of 'data'")
        elif isinstance(self.geodata['polys'], (int, str, type(None))):
            self.geodata['polys'] = [self.geodata['polys']] * len(self.geodata['data'])

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
                if self.verbose and (self.parallel_rank is None or self.parallel_rank == 0):
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

    @property
    def alphaFaults(self):
        return self.alpha.get('faults')
    
    @alphaFaults.setter
    def alphaFaults(self, value):
        self.alpha['faults'] = value
    
    def load_from_file(self, config_file, encoding='utf-8'):
        # Load the configuration from a file
        with open(config_file, 'r', encoding=encoding) as f:
            config_data = yaml.safe_load(f)
        
        # Set lon0 and lat0 based on the configuration file
        lon_lat_0 = config_data.get('lon_lat_0', None)
        if lon_lat_0 is not None:
            self.lon0 = lon_lat_0[0]
            self.lat0 = lon_lat_0[1]
        config_data.pop('lon_lat_0', None)

        # Handle alpha configuration
        if 'alpha' in config_data:
            self.alpha.update(config_data['alpha'])
            config_data.pop('alpha')  # Remove alpha from config_data to avoid overwriting the alpha attribute

        # Get the default parameters
        default_fault_parameters = config_data.get('faults', {}).get('defaults', {})

        # Handle the faults
        for fault_name, fault_parameters in config_data.get('faults', {}).items():
            if fault_name == 'defaults':
                continue

            # Use the default parameters if fault_parameters is None
            if fault_parameters is None:
                config_data['faults'][fault_name] = default_fault_parameters.copy()
            else:
                # Combine the default parameters with the fault parameters
                merged_parameters = {**default_fault_parameters, **fault_parameters}

                # Special handling for 'geometry'
                merged_geometry = {**default_fault_parameters.get('geometry', {}), **fault_parameters.get('geometry', {})}
                merged_parameters['geometry'] = merged_geometry

                # Special handling for 'method_parameters'
                merged_method_parameters = default_fault_parameters.get('method_parameters', {}).copy()
                for method_name, method_params in fault_parameters.get('method_parameters', {}).items():
                    if method_name in merged_method_parameters:
                        # Check if 'method' is in the method parameters
                        # If it is, then the method parameters are to be replaced
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
                if self.verbose and (self.parallel_rank is None or self.parallel_rank == 0):
                    print(f"Fault '{fault_name}' not found in config file. Using default parameters.")
    
        # Remove faults not in faultnames from config_data and self.faults
        for fault_name in list(config_data['faults'].keys()):
            if fault_name != 'defaults' and fault_name not in self.faultnames:
                del config_data['faults'][fault_name]
                if self.verbose and (self.parallel_rank is None or self.parallel_rank == 0):
                    print(f"Fault '{fault_name}' found in config file but not in faultnames. Removed from configuration.")

        for fault_name in list(self.faults.keys()):
            if fault_name not in self.faultnames:
                del self.faults[fault_name]
                if self.verbose and (self.parallel_rank is None or self.parallel_rank == 0):
                    print(f"Fault '{fault_name}' found in self.faults but not in faultnames. Removed from configuration.")

        self.set_attributes(**config_data)

    def _initialize_faults_and_assemble_data(self, faults_list=None, geodata=None):
        """
        Setup faults by building Green's functions, assembling data and Green's functions for inversion,
        and building covariance matrices for GPS/InSAR data.
        """
        # --------------------------------Build GreenFns-----------------------------------------#
        faults_list = faults_list or self.faults_list
        geodata = self.geodata['data'] or geodata
        verticals = self.geodata['verticals']
        polys = self.geodata['polys']
        nonpolys = [None] * len(geodata)

        for ifault in faults_list:
            faultname = ifault.name
            gfmethod = self.faults[faultname]['method_parameters']['update_GFs']['method']
            for obsdata, vertical in zip(geodata, verticals):
                ifault.buildGFs(obsdata, vertical=vertical, slipdir='sd', method=gfmethod, verbose=False)
            ifault.initializeslip()

        # ----------------------Assemble data and GreenFns for Inversion------------------------#
        poly_assembled = False  # flag to check if the polynomial is assembled
        for ifault in faults_list:
            # assemble data
            ifault.assembled(geodata, verbose=False)
            # assemble GreensFns
            if not poly_assembled:
                ifault.assembleGFs(geodata, polys=polys, slipdir='sd', verbose=False, custom=False)
                poly_assembled = True
            else:
                ifault.assembleGFs(geodata, polys=nonpolys, slipdir='sd', verbose=False, custom=False)

        # --------------------------Build Covariance Matrix for GPS/InSAR data-------------------#
        # assemble data Covariance matrices, You should assemble the Green's function matrix first
        for ifault in faults_list:
            # Bug: the verbose may lead to the error if it is set to True
            ifault.assembleCd(geodata, verbose=False, add_prediction=None)

    def set_attributes(self, **kwargs):
        # Set the attributes based on the key-value pairs in kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def set_faults_method_parameters(self, method_parameters_dict):
        """
        Update the method parameters for all faults.
        """
        for fault_name, method_parameters in method_parameters_dict.items():
            if fault_name in self.faults:
                self.faults[fault_name]['method_parameters'].update(method_parameters)
            else:
                raise ValueError(f"Fault {fault_name} does not exist in the configuration.")

    def update_GFs_parameters(self, geodata, verticals, dataFaults=None, gfmethods=None):
        """
        Update the update_GFs method parameters for all faults.
        """
        if len(geodata) != len(verticals):
            raise ValueError("Length of geodata and verticals should be the same.")
        
        if gfmethods is not None and len(self.faultnames) != len(gfmethods):
            raise ValueError("Length of faultnames and gfmethods should be the same.")
        
        for i, fault_name in enumerate(self.faultnames):
            fault_parameters = self.faults[fault_name]
            method = gfmethods[i] if gfmethods is not None else None
            
            if method is None:
                ifault = self.faults_list[i]
                method = fault_parameters['method_parameters']['update_GFs'].get('method')
                
                if method is None:
                    if ifault.patchType == 'triangle':
                        method = 'cutde'
                    elif ifault.patchType == 'rectangle':
                        method = 'okada'
                    else:
                        raise ValueError("Unknown patchType")
            
            fault_parameters['method_parameters']['update_GFs'] = {
                'geodata': geodata,
                'verticals': verticals,
                'dataFaults': dataFaults,
                'method': method
            }

    def set_data_faults(self, dataFaults=None):
        if dataFaults is not None:
            self.dataFaults = dataFaults
        elif self.dataFaults is None:
            self.dataFaults = [self.faultnames]*len(self.geodata['data'])
        
        # Check if self.dataFaults is a list
        if not isinstance(self.dataFaults, list):
            raise ValueError("self.dataFaults must be a list")
        
        # Check the length of self.dataFaults
        if len(self.dataFaults) != len(self.geodata['data']):
            raise ValueError("Length of self.dataFaults should be equal to the length of geodata")
        
        # Check contents of self.dataFaults
        # flatten the list of lists to a single list if sublist is a list
        self.validate_faults(check_dataFaults=True)

    def validate_faults(self, check_dataFaults=False, check_alphaFaults=False):
        """
        Validate the dataFaults or alphaFaults.
        """

        if check_dataFaults:
            # Flatten the list of lists to a single list if sublist is a list
            flattened_dataFaults = [item for sublist in self.dataFaults for item in (sublist if isinstance(sublist, list) else [sublist])]
            
            # Check flattened_dataFaults is subset of self.faultnames
            if not set(flattened_dataFaults).issubset(set(self.faultnames + [None])):
                raise ValueError("The dataFaults must be a subset of the faultnames in self.multifaults")
        
        if check_alphaFaults:
            # Check if self.alphaFaults does not contain None and all other items are lists
            if None not in self.alphaFaults:
                if not all(isinstance(item, list) for item in self.alphaFaults):
                    raise ValueError("All items in self.alphaFaults must be lists")

                # Flatten the list of lists
                flattened_faults = [fault for sublist in self.alphaFaults for fault in sublist]
                
                # Check if self.alphaFaults contains duplicate items
                if len(flattened_faults) != len(set(flattened_faults)):
                    raise ValueError("self.alphaFaults cannot contain duplicate items")

                # Check if the union of self.alphaFaults equals self.faultnames
                if set(flattened_faults) != set(self.faultnames):
                    raise ValueError("The union of self.alphaFaults must equal the items in self.faultnames")

    def set_alpha_faults(self, alphaFaults=None):
        if alphaFaults is not None:
            self.alphaFaults = alphaFaults
        elif self.alphaFaults is None:
            self.alphaFaults = [None]

        # Check if self.alphaFaults is a list
        if not isinstance(self.alphaFaults, list):
            raise ValueError("self.alphaFaults must be a list")

        # Check if self.alphaFaults contains more than one None
        if None in self.alphaFaults and len(self.alphaFaults) > 1:
            raise ValueError("self.alphaFaults cannot contain None and other items simultaneously")

        # Check if self.alphaFaults does not contain None and all other items are lists
        self.validate_faults(check_alphaFaults=True)
        
        # 
        self.alpha['faults'] = self.alphaFaults
        if None in self.alphaFaults:
            self.alphaFaultsIndex = [0] * len(self.faultnames)
        else:
            # Create a dictionary to map fault names to their indices
            fault_index_map = {fault: idx for idx, sublist in enumerate(self.alphaFaults) for fault in sublist}
            # Generate the alphaFaultsIndex based on the order of faultnames
            self.alphaFaultsIndex = [fault_index_map[fault] for fault in self.faultnames]
        self.alpha['faults_index'] = self.alphaFaultsIndex

