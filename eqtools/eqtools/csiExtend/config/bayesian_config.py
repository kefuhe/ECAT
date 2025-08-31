import yaml

from .linear_config import LinearInversionConfig
from ..multifaults_base import MyMultiFaultsInversion


class BayesianMultiFaultsInversionConfig(LinearInversionConfig):
    """
    Bayesian multi-faults inversion configuration.
    
    This class provides configuration management specifically for Bayesian inversion
    of multiple fault systems, inheriting common linear inversion functionality 
    from LinearInversionConfig and adding Bayesian-specific features.
    """
    
    def __init__(self, config_file='default_config.yml', multifaults=None, geodata=None, 
                 verticals=None, polys=None, dataFaults=None, alphaFaults=None, faults_list=None,
                 gfmethods=None, slip_sampling_mode='ss_ds', bayesian_sampling_mode='SMC_F_J', 
                 encoding='utf-8', verbose=False, parallel_rank=None, **kwargs):
        """
        Initialize the BayesianMultiFaultsInversionConfig object.
        
        Parameters:
        -----------
        config_file : str, optional
            Path to the configuration file (default: 'default_config.yml')
        multifaults : object, optional
            Multifaults object for the inversion
        geodata : list, optional
            List of geodetic data objects
        verticals : list, optional
            List of vertical displacement flags for each dataset
        polys : list, optional
            List of polynomial correction orders for each dataset
        dataFaults : list, optional
            List of fault names for each dataset
        alphaFaults : list, optional
            List of alpha parameter groups for faults
        faults_list : list, optional
            List of fault objects
        gfmethods : list, optional
            List of Green's function methods for each fault
        slip_sampling_mode : str, optional
            Slip sampling mode (default: 'ss_ds')
        bayesian_sampling_mode : str, optional
            Bayesian sampling mode (default: 'SMC_F_J')
        encoding : str, optional
            File encoding for configuration file (default: 'utf-8')
        verbose : bool, optional
            Enable verbose output (default: False)
        parallel_rank : int, optional
            Parallel processing rank
        **kwargs : dict
            Additional keyword arguments
        """
        
        # Bayesian-specific attributes
        self.bayesian_sampling_mode = bayesian_sampling_mode
        self.nchains = 100
        self.chain_length = 50
        self.slip_sampling_mode = slip_sampling_mode
        self._sigmas_param_name = 'initial_value'  # BayesianMultiFaultsInversionConfig uses 'initial_value' for sigmas
        # Call parent class initialization
        super().__init__(config_file=config_file, multifaults=multifaults, geodata=geodata,
                        verticals=verticals, polys=polys, dataFaults=dataFaults, 
                        alphaFaults=alphaFaults, faults_list=faults_list, gfmethods=gfmethods,
                        encoding=encoding, verbose=verbose, parallel_rank=parallel_rank, **kwargs)
        
        # Bayesian-specific post-processing
        self._process_bayesian_specific_config()
        
        # Bayesian-specific validation
        self._validate_fault_configurations()
        
        # Initialize data assembly
        if self.clipping_options.get('enabled', False):
            self._initialize_faults_and_assemble_data()
        if multifaults is None:
            self._initialize_faults_and_assemble_data()
            multifaults = MyMultiFaultsInversion('myfault', self.faults_list, verbose=self.verbose)
            multifaults.assembleGFs()
            self.multifaults = multifaults

        if self.parallel_rank is not None and self.parallel_rank == 0:
            self.export_config()

    def _process_bayesian_specific_config(self):
        """Process Bayesian-specific configuration settings."""
        # Set geometry parameters for faults with shared information
        for ifault in self.faults_list:
            if hasattr(ifault, 'use_shared_info') and ifault.use_shared_info:
                self.faults[ifault.name]['geometry'] = {
                    'update': True,
                    'sample_positions': [0, 0]
                }

        # Alpha configuration processing
        if not self.alpha['enabled']:
            self.alpha['update'] = False
            self.alpha['initial_value'] = 0.0

        # Force set sampling mode
        if self.bayesian_sampling_mode == 'SMC_F_J':
            self.slip_sampling_mode = 'ss_ds'

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
                if self.verbose:
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
                if self.verbose:
                    print(f"Fault '{fault_name}' not found in config file. Using default parameters.")
    
        # Remove faults not in faultnames from config_data and self.faults
        for fault_name in list(config_data['faults'].keys()):
            if fault_name != 'defaults' and fault_name not in self.faultnames:
                del config_data['faults'][fault_name]
                if self.verbose:
                    print(f"Fault '{fault_name}' found in config file but not in faultnames. Removed from configuration.")
    
        for fault_name in list(self.faults.keys()):
            if fault_name not in self.faultnames:
                del self.faults[fault_name]
                if self.verbose:
                    print(f"Fault '{fault_name}' found in self.faults but not in faultnames. Removed from configuration.")
    
        # Set the attributes based on the key-value pairs in config_data
        self.set_attributes(**config_data)

        # Validate fault configurations
        self._validate_fault_configurations()

    def _process_fixed_nodes(self, fault_parameters):
        """
        Process fixed nodes to ensure they are in the correct format.
        
        Parameters:
        -----------
        fault_parameters : dict
            Dictionary containing fault parameters
        """
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

    def _validate_fault_configurations(self):
        """Validate Bayesian fault configurations."""
        # 1. Validate Laplacian bounds
        self._validate_laplacian_bounds()
        
        # 2. Validate geometry sampling positions
        self._validate_geometry_sample_positions()

    def _validate_geometry_sample_positions(self):
        """Validate geometry sampling positions."""
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
            
            if self.verbose:
                print(f"Geometry validation passed: {len(geometry_updating_faults)} faults updating geometry, "
                      f"sample positions [0, {max_pos+1}) complete")

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
                raise ValueError(f"Unknown attribute '{key}'")