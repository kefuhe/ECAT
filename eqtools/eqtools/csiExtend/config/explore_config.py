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
from ..multifaults_base import MyMultiFaultsInversion

# Import utility functions for parsing configuration updates
from .config_utils import parse_update, parse_initial_values
from .config_utils import parse_alpha_faults, parse_data_faults, parse_sigmas_config
from .base_config import CommonConfigBase


class AliasManager:
    """Manage fault ID and alias mappings."""
    
    def __init__(self, num_faults, user_aliases=None):
        self.num_faults = num_faults
        self.user_aliases = user_aliases or []
        self.id_to_alias = {}
        self.alias_to_id = {}
        self._build_mappings()
    
    def _build_mappings(self):
        """Build ID <-> Alias mappings."""
        if self.user_aliases:
            if isinstance(self.user_aliases, str):
                if self.num_faults > 1:
                    raise ValueError(
                        f"Single alias '{self.user_aliases}' provided but nfaults={self.num_faults}"
                    )
                self.user_aliases = [self.user_aliases]
            
            if len(self.user_aliases) != self.num_faults:
                msg = f"Alias count ({len(self.user_aliases)}) != nfaults ({self.num_faults})"
                logger.error(msg)
                raise ValueError(msg)
        
        for i in range(self.num_faults):
            fault_id = f'fault_{i}'
            alias = self.user_aliases[i] if self.user_aliases else f'F{i}'
            self.id_to_alias[fault_id] = alias
            self.alias_to_id[alias] = fault_id
    
    def translate_config_keys(self, config_dict):
        """Translate aliases in configuration dictionary keys"""
        if not self.user_aliases:
            return config_dict
        
        translated = {}
        for key, value in config_dict.items():
            new_key = self.alias_to_id.get(key, key)
            translated[new_key] = value
        return translated
    
    def translate_faults_list(self, faults_list):
        """Translate aliases in faults list"""
        if not faults_list or not self.user_aliases:
            return faults_list
        
        if isinstance(faults_list, list):
            return [
                [self.alias_to_id.get(f, f) for f in item] 
                if isinstance(item, list) 
                else self.alias_to_id.get(item, item)
                for item in faults_list
            ]
        return faults_list

class ExploreFaultConfig(CommonConfigBase):
    def __init__(self, config_file=None, geodata=None, verbose=False, parallel_rank=None):
        self._sigmas_param_name = 'values'  # ExploreFaultConfig uses 'values' for sigmas
        super().__init__(config_file=config_file, geodata=geodata, verbose=verbose, parallel_rank=parallel_rank)
        self.bounds = {}
        self.initial = {} # Initial parameters for each fault
        self.fixed_params = {} # Fixed parameters for each fault
        self.nfaults = 1 # Number of faults to be explored 
        self.faultnames = [f'fault_{i}' for i in range(self.nfaults)]
        self.slip_sampling_mode = 'mag_rake'

        # Storage for user-defined aliases (e.g., ['ATF', 'Kunlun'])
        self.fault_aliasnames = None

        # Load configuration if file is provided
        if config_file:
            self.load_config(config_file, geodata=geodata)

        # Parse sigmas parameters after loading config
        if self.geodata and 'data' in self.geodata:
            data_names = [d.name for d in self.geodata.get('data', [])]
            
            if 'sigmas' in self.geodata:
                self.sigmas = parse_sigmas_config(
                    self.geodata['sigmas'], 
                    dataset_names=data_names,
                    param_name='values'
                )

    def load_config(self, config_file, geodata=None):
        """
        Load configuration from a YAML file.
        
        This method implements a "Translation Layer" strategy:
        1. Load raw YAML data.
        2. Apply alias mapping (translate User Aliases -> Internal IDs).
        3. Assign values to attributes (triggering built-in validation).
        """
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        nfaults = config.get('nfaults', 1)
        user_aliases = config.get('fault_aliasnames', config.get('fault_names', None))

        # Initialize AliasManager with user aliases and number of faults
        self.alias_manager = AliasManager(nfaults, user_aliases)
        self.fault_id_to_alias = self.alias_manager.id_to_alias

        # Translate configuration sections using AliasManager
        for section in ['bounds', 'initial', 'fixed_params']:
            if section in config:
                config[section] = self.alias_manager.translate_config_keys(config[section])
        
        if 'geodata' in config and 'faults' in config['geodata']:
            config['geodata']['faults'] = self.alias_manager.translate_faults_list(
                config['geodata']['faults']
            )

        # Load basic settings
        self.bounds = config.get('bounds', {})
        self.initial = config.get('initial', {})
        self.fixed_params = config.get('fixed_params', {})
        self.nfaults = config.get('nfaults', 1)
        
        # Store aliases for display purposes later
        self.fault_aliasnames = config.get('fault_aliasnames', config.get('fault_names', None))
        
        # Re-initialize internal faultnames based on loaded nfaults
        self.faultnames = [f'fault_{i}' for i in range(self.nfaults)]
        
        self.slip_sampling_mode = config.get('slip_sampling_mode', 'mag_rake')
        self.clipping_options = config.get('clipping_options', {})
        
        # Handle Geodata
        # Priority: arguments > config file
        self.geodata = config.get('geodata', {})

        lon_lat_0 = config.get('lon_lat_0', None)
        if lon_lat_0:
            self.lon0, self.lat0 = lon_lat_0
        self.data_sources = config.get('data_sources', {})

        # Load data files if not provided externally
        self._update_geodata(geodata)
        self._validate_verticals()

        # Parse sigmas (using 'values' as the key for ExploreFaultConfig)
        if 'data' in self.geodata:
            data_names = [d.name for d in self.geodata.get('data', [])]
            if 'sigmas' in self.geodata:
                self.sigmas = parse_sigmas_config(
                    self.geodata['sigmas'], 
                    dataset_names=data_names,
                    param_name='values'
                )
        
        # [VALIDATION TRIGGER]
        # Assigning to self.dataFaults triggers `parse_data_faults` in base_config.py
        # Since _apply_alias_mapping has run, 'faults' now contains 'fault_0' etc.
        # So validation will PASS.
        self.dataFaults = self.geodata.get('faults', None)

        # Update polygon boundaries
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
                            msg = f"Bounds for {boundary_key} must be set as there is no default"
                            logger.error(msg)
                            raise ValueError(msg)
                else:
                    msg = f"Data name {data.name} is not in the estimate list"
                    logger.error(msg)
                    raise ValueError(msg)

    @CommonConfigBase.sigmas.setter
    def sigmas(self, value):
        super(ExploreFaultConfig, self.__class__).sigmas.fset(self, value)
        self.geodata['sigmas']['ndatas'] = self.ndatas
        self.geodata['sigmas']['names'] = ['sigma_{}'.format(i) for i in range(self.ndatas)]

        sigma_names = self.geodata['sigmas']['names']
        bounds = self.geodata['sigmas']['bounds']
    
        # Check if 'defaults' is in bounds
        if 'defaults' not in bounds:
            # Check if all sigmas are in bounds
            if not set(sigma_names).issubset(bounds.keys()):
                msg = "The bounds dictionary must have keys for all sigmas or a 'defaults' key"
                logger.error(msg)
                raise ValueError(msg)
        else:
            # Fill in the missing sigmas with the defaults
            defaults = bounds['defaults']
            for name in sigma_names:
                if name not in bounds:
                    bounds[name] = defaults
    
    @property
    def ndatas(self):
        return len(self.geodata.get('data', []))
    
    def export_config(self, filename=None, format='yaml'):
        """
        Export the current ExploreFaultConfig object's internal state to a file (supports yaml/json).
    
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
          faultnames, bounds, initial, fixed_params, geodata, sigmas, and other parameters.
        - Numpy arrays and other non-serializable objects will be converted to lists or basic types.
        - geodata['data'] will be replaced by a list of dataset names.
                - For fault-related dictionaries such as bounds, initial, and fixed_params,
                    the order is 'defaults' first, then fault_0, fault_1, ... (consistent with self.faultnames).
                - You may add more attributes to export_dict as needed.
    
        Example
        -------
        >>> config = ExploreFaultConfig(config_file='your_config.yml', ...)
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
    
        # Deepcopy to avoid modifying the original objects
        bounds_export = copy.deepcopy(self.bounds)
        initial_export = copy.deepcopy(self.initial)
        fixed_params_export = copy.deepcopy(self.fixed_params)
        geodata_export = copy.deepcopy(self.geodata)
    
        # geodata['data'] -> dataset names
        if 'data' in geodata_export and isinstance(geodata_export['data'], list):
            geodata_export['data'] = [d.name for d in geodata_export['data']]
    
        # bounds/initial/fixed_params: output only 'defaults' and entries in the order of self.faultnames
        def _ordered_fault_dict(src):
            od = OrderedDict()
            if 'defaults' in src:
                od['defaults'] = _to_serializable(src['defaults'])
            for fname in self.faultnames:
                if fname in src:
                    od[fname] = _to_serializable(src[fname])
            return od
    
        bounds_ordered = _ordered_fault_dict(bounds_export)
        initial_ordered = _ordered_fault_dict(initial_export)
        fixed_params_ordered = _ordered_fault_dict(fixed_params_export)
    
        export_dict = OrderedDict([
            ('nchains', getattr(self, 'nchains', None)),
            ('chain_length', getattr(self, 'chain_length', None)),
            ('nfaults', self.nfaults),
            ('lon_lat_0', getattr(self, 'lon_lat_0', None) if hasattr(self, 'lon_lat_0') else None),
            ('slip_sampling_mode', self.slip_sampling_mode),
            ('clipping_options', getattr(self, 'clipping_options', {})),
            ('bounds', bounds_ordered),
            ('initial', initial_ordered),
            ('fixed_params', fixed_params_ordered),
            ('geodata', _to_serializable(geodata_export)),
            ('data_sources', getattr(self, 'data_sources', {})),
        ])
    
        # Remove None values for cleaner output
        export_dict = OrderedDict((k, v) for k, v in export_dict.items() if v is not None)
    
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
    
            if filename is None:
                filename = 'parsed_' + getattr(self, 'config_file', 'config.yml')
            with open(filename, 'w', encoding='utf-8') as f:
                yaml.dump(export_dict, f, allow_unicode=True, Dumper=OrderedDumper, default_flow_style=None, indent=2)
        elif format == 'json':
            if filename is None:
                filename = 'parsed_' + getattr(self, 'config_file', 'config.yml').replace('.yml', '.json')
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_dict, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError("Only 'yaml' and 'json' formats are supported for export.")

# EOF