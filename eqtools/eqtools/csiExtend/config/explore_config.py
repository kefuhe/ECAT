import yaml
import os
import glob
import numpy as np

# Load csi and its extensions
from csi.gps import gps
from csi.insar import insar
from ..multifaults_base import MyMultiFaultsInversion

# Import utility functions for parsing configuration updates
from .config_utils import parse_update, parse_initial_values
from .config_utils import parse_alpha_faults, parse_data_faults, parse_sigmas_config
from .base_config import CommonConfigBase

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

        if config_file:
            self.load_config(config_file, geodata=geodata)
        
        # Parse the 'update' parameter in sigmas
        n_datasets = len(self.geodata.get('data', []))
        data_names = [d.name for d in self.geodata.get('data', [])]
        # Parse sigmas parameters
        self.sigmas = parse_sigmas_config(self.geodata['sigmas'], 
                                                     dataset_names=data_names,
                                                     param_name='values')

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
        # Parse sigmas parameters
        self.sigmas = parse_sigmas_config(self.geodata['sigmas'], 
                                                     dataset_names=data_names,
                                                     param_name='values')
        self.dataFaults = self.geodata.get('faults', None)

        # self._set_geodata_attributes()
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
        - bounds、initial、fixed_params等fault相关字典，顺序为'defaults'在前，然后依次为fault_0, fault_1, ...（与self.faultnames一致）。
        - 你可以根据需要添加更多属性到export_dict。
    
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
    
        # Deepcopy to avoid修改原对象
        bounds_export = copy.deepcopy(self.bounds)
        initial_export = copy.deepcopy(self.initial)
        fixed_params_export = copy.deepcopy(self.fixed_params)
        geodata_export = copy.deepcopy(self.geodata)
    
        # geodata['data'] -> dataset names
        if 'data' in geodata_export and isinstance(geodata_export['data'], list):
            geodata_export['data'] = [d.name for d in geodata_export['data']]
    
        # bounds/initial/fixed_params: 只输出'defaults'和self.faultnames顺序
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