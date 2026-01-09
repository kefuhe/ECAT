import yaml
import os
import glob
import numpy as np
import logging

# Setup module-level logger
logger = logging.getLogger(__name__)

# Load csi and its extensions
from csi.gps import gps
from csi.insar import insar

# Import utility functions for parsing configuration updates
from .config_utils import parse_data_faults, parse_update, parse_initial_values, parse_sigmas_config


class CommonConfigBase:
    def __init__(self, config_file='default_config.yml', geodata=None, encoding='utf-8', verbose=False, parallel_rank=None):
        self.verbose = verbose and (parallel_rank is None or parallel_rank == 0)
        self.parallel_rank = parallel_rank # Rank for parallel processing, if applicable
        if self.verbose:
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
        # Get dataset information
        data_names = [d.name for d in self.geodata.get('data', [])]
        
        # Get the parameter name for this config class (can be overridden in subclasses)
        param_name = getattr(self, '_sigmas_param_name', 'initial_value')
        
        # Parse the entire sigmas configuration using the new function
        self.geodata['sigmas'] = parse_sigmas_config(value, dataset_names=data_names, param_name=param_name)

    @property
    def dataFaults(self):
        return self.geodata['faults']
    
    @dataFaults.setter
    def dataFaults(self, value):
        all_faultnames = self.faultnames
        all_datanames = [d.name for d in self.geodata['data']]
        dataFaults = value if value is not None else self.geodata['faults']
        result = parse_data_faults(dataFaults, all_faultnames, all_datanames, param_name='dataFaults')
        self.geodata['faults'] = result

    def load_data(self, data_type):
        data_config = self.data_sources[data_type]
        data_files = []

        # Check if specific files are provided
        if 'files' in data_config:
            data_files.extend(data_config['files'])
        else:
            # Use file pattern to match files
            found_files = glob.glob(os.path.join(data_config['directory'], data_config['file_pattern']))
            if not found_files and self.verbose:
                logger.debug(f"No files found for pattern: {os.path.join(data_config['directory'], data_config['file_pattern'])}")
            data_files.extend(found_files)

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
                msg = f"Unsupported data type: {data_type}"
                logger.error(msg)
                raise ValueError(msg)
            self.geodata['data'].append(data_instance)

    def load_all_data(self):
        if not self.geodata.get('data', []):
            for data_type in self.data_sources:
                self.load_data(data_type)
            if not self.geodata['data']:
                msg = "No geodata files were loaded. Please check your configuration."
                logger.error(msg)
                raise ValueError(msg)
        else:
            if self.verbose:
                logger.info("Geodata already provided in configuration.")

    def _print_initialization_message(self):
        logger.info("---------------------------------")
        logger.info("---------------------------------")
        logger.info(f"Initializing {self.__class__.__name__} object")

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
                msg = f"Length of 'verticals' list ({len(verticals)}) does not match length of 'data' ({data_length})"
                logger.error(msg)
                raise ValueError(msg)
        elif isinstance(verticals, bool):
            self.geodata['verticals'] = [verticals] * data_length
        else:
            msg = "'verticals' must be either a list or a boolean"
            logger.error(msg)
            raise ValueError(msg)

    def _select_data_sets(self):
        data_verticals_dict = {d.name: v for d, v in zip(self.geodata['data'], self.geodata['verticals'])}
        if self.clipping_options.get('enabled', False):
            methods = self.clipping_options.get('methods', [])
            for method_config in methods:
                method = method_config.get('method', None)
                if method == 'lon_lat_range':
                    lon_lat_range = method_config.get('lon_lat_range', None)
                    if lon_lat_range is None:
                        msg = "Clipping method 'lon_lat_range' requires 'lon_lat_range' to be set"
                        logger.error(msg)
                        raise ValueError(msg)
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
                        msg = "Clipping method 'distance_to_fault' requires 'distance_to_fault' and non-empty 'faults_list' to be set"
                        logger.error(msg)
                        raise ValueError(msg)
                    for data in self.geodata['data']:
                        if data.dtype == 'insar':
                            data.reject_pixels_fault(distance_to_fault, faults)
                        elif data.dtype == 'gps':
                            # No clipping for GPS data
                            continue
                else:
                    msg = f"Unsupported clipping method: {method}"
                    logger.error(msg)
                    raise ValueError(msg)