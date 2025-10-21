"""
Configuration parsing utilities for bayesian_config.py and related modules.
"""
import numpy as np

def parse_update(config, n_datasets, param_name="update", dataset_names=None):
    """
    Parse the 'update' parameter from configuration with enhanced flexibility.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing the update parameter
    n_datasets : int
        Number of datasets
    param_name : str, optional
        Name of the parameter being parsed. Default is "update"
    dataset_names : list, optional
        List of dataset names for name-based indexing
    
    Returns:
    --------
    list
        List of boolean values indicating update status for each dataset
        
    Examples:
    ---------
    >>> parse_update({"update": True}, 3)
    [True, True, True]
    
    >>> parse_update({"update": [True, False, True]}, 3)
    [True, False, True]
    
    >>> parse_update({"update": [0, 2]}, 3)  # Index list
    [True, False, True]
    
    >>> parse_update({"update": ["sar_a", "sar_c"]}, 3, dataset_names=["sar_a", "sar_b", "sar_c"])
    [True, False, True]
    
    >>> parse_update({"update": {"true_indices": [0, 2]}}, 3)  # Compatible with old format
    [True, False, True]
    """
    update = config[param_name]
    
    if isinstance(update, bool):
        return [update] * n_datasets
    
    elif isinstance(update, list):
        # Check the type of list contents
        if len(update) == n_datasets and all(isinstance(x, bool) for x in update):
            # Complete boolean list
            return update
        elif all(isinstance(x, int) for x in update):
            # Index list (more intuitive approach)
            flags = [False] * n_datasets
            for idx in update:
                if idx >= n_datasets:
                    raise ValueError(f"Index {idx} in {param_name} exceeds number of datasets ({n_datasets})")
                flags[idx] = True
            return flags
        elif dataset_names and all(isinstance(x, str) for x in update):
            # Name list
            flags = [False] * n_datasets
            for name in update:
                if name not in dataset_names:
                    raise ValueError(f"Dataset name '{name}' not found in dataset_names")
                idx = dataset_names.index(name)
                flags[idx] = True
            return flags
        else:
            raise ValueError(f"Invalid list format for {param_name}")
    
    elif isinstance(update, dict):
        # Compatible with old dictionary format
        if "true_indices" in update and "false_indices" in update:
            raise ValueError(f"Cannot specify both 'true_indices' and 'false_indices' in {param_name}")
        
        if "true_indices" in update:
            flags = [False] * n_datasets
            for idx in update["true_indices"]:
                if idx >= n_datasets:
                    raise ValueError(f"Index {idx} in {param_name}.true_indices exceeds number of datasets ({n_datasets})")
                flags[idx] = True
            return flags
        elif "false_indices" in update:
            flags = [True] * n_datasets
            for idx in update["false_indices"]:
                if idx >= n_datasets:
                    raise ValueError(f"Index {idx} in {param_name}.false_indices exceeds number of datasets ({n_datasets})")
                flags[idx] = False
            return flags
        else:
            raise ValueError(f"Dict format for {param_name} must contain either 'true_indices' or 'false_indices'")
    
    else:
        raise ValueError(f"Invalid format for {param_name}")


def parse_initial_values(config, n_datasets, param_name="initial_value", default_value=0.0, 
                        min_value=None, dataset_names=None, print_name=None):
    """
    Parse initial values from configuration with enhanced flexibility and validation.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing the initial values parameter
    n_datasets : int
        Number of datasets
    param_name : str, optional
        Name of the parameter being parsed. Default is "initial_value"
    print_name : str, optional
        Name to use for printing in error messages. If None, uses param_name.
    default_value : float, optional
        Default value to use if parameter is missing. Default is 0.0
    min_value : float, optional
        Minimum allowed value. Default is None (no minimum check)
    dataset_names : list, optional
        List of dataset names for name-based indexing
    
    Returns:
    --------
    list
        List of float values for each dataset
        
    Examples:
    ---------
    >>> parse_initial_values({"initial_value": 0.05}, 3)
    [0.05, 0.05, 0.05]

    >>> parse_initial_values({"initial_value": [0.01]}, 3)
    [0.01, 0.01, 0.01]
    
    >>> parse_initial_values({"initial_value": [0.01, 0.02, 0.015]}, 3)
    [0.01, 0.02, 0.015]
    
    >>> parse_initial_values({"initial_value": [0.01, 0, 0.015]}, 3)
    [0.01, 0.0, 0.015]
    
    >>> parse_initial_values({}, 3)  # Missing parameter
    [0.0, 0.0, 0.0]
    
    >>> parse_initial_values({"initial_value": 0}, 3)
    [0.0, 0.0, 0.0]
    
    >>> parse_initial_values({"initial_value": {"sar_a": 0.01, "sar_c": 0.02}}, 3, 
    ...                     dataset_names=["sar_a", "sar_b", "sar_c"])
    [0.01, 0.0, 0.02]
    """
    initial_value = config.get(param_name)

    if print_name is None:
        print_name = param_name
    
    # Handle missing parameter
    if initial_value is None:
        return [float(default_value)] * n_datasets
    
    # Handle single value (int or float)
    if isinstance(initial_value, (int, float)):
        return [float(initial_value)] * n_datasets
    
    # Handle list of values
    elif isinstance(initial_value, list):
        if len(initial_value) == 1:
            # Single value in list, expand it
            return [float(initial_value[0])] * n_datasets
        if len(initial_value) != n_datasets:
            raise ValueError(f"Length of '{print_name}' list ({len(initial_value)}) does not match number of datasets ({n_datasets})")
        
        # Convert all values to float
        processed_values = []
        for i, val in enumerate(initial_value):
            if not isinstance(val, (int, float)):
                raise ValueError(f"All values in '{print_name}' must be numbers, got {type(val)} at index {i}")
            processed_values.append(float(val))
        
        return processed_values
    
    # Handle dictionary format (dataset names to values mapping)
    elif isinstance(initial_value, dict):
        if dataset_names is None:
            # raise ValueError(f"Dataset names must be provided when using dictionary format for '{print_name}'")
            raise ValueError(f'Not be supported when using dictionary format for {print_name}')
        
        processed_values = []
        for i, dataset_name in enumerate(dataset_names):
            if dataset_name in initial_value:
                val = initial_value[dataset_name]
                if not isinstance(val, (int, float)):
                    raise ValueError(f"Value for dataset '{dataset_name}' in '{print_name}' must be a number, got {type(val)}")
                processed_values.append(float(val))
            else:
                # Use default value for datasets not specified in dictionary
                processed_values.append(float(default_value))
        
        return processed_values
    
    else:
        raise ValueError(f"'{param_name}' must be a number, list of numbers, or dictionary mapping dataset names to numbers, got {type(initial_value)}")

def parse_data_faults(data_faults_config, all_faultnames, all_datanames, param_name="dataFaults"):
    """
    Parse dataFaults configuration with enhanced flexibility.
    
    Parameters:
    -----------
    data_faults_config : None, list, or dict
        Configuration for data faults
    all_faultnames : list
        List of all available fault names
    all_datanames : list
        List of all data names
    param_name : str, optional
        Name of the parameter for error messages
        
    Returns:
    --------
    list
        List of fault name lists for each dataset
        
    Examples:
    ---------
    >>> parse_data_faults(None, ["f1", "f2"], ["d1", "d2"])
    [["f1", "f2"], ["f1", "f2"]]
    
    >>> parse_data_faults([None, ["f1"]], ["f1", "f2"], ["d1", "d2"])
    [["f1", "f2"], ["f1"]]
    
    >>> parse_data_faults({"d1": "f1", "d2": None}, ["f1", "f2"], ["d1", "d2"])
    [["f1"], ["f1", "f2"]]
    """
    
    def _normalize_fault_item(item, all_faultnames, param_name):
        """Normalize a single fault item to a list of fault names"""
        if item is None:
            return all_faultnames.copy()
        elif isinstance(item, str):
            if item not in all_faultnames:
                raise ValueError(f"Fault name '{item}' in {param_name} not found in all_faultnames")
            return [item]
        elif isinstance(item, list):
            # Check if it's exactly all_faultnames
            if set(item) == set(all_faultnames):
                return all_faultnames.copy()
            # Check if it's a subset
            elif set(item).issubset(set(all_faultnames)):
                return item.copy()
            else:
                invalid_names = set(item) - set(all_faultnames)
                raise ValueError(f"Invalid fault names in {param_name}: {invalid_names}")
        else:
            raise ValueError(f"Invalid fault specification in {param_name}: {item}")
    
    # Case 1: None - expand to all_faultnames for all datasets
    if data_faults_config is None:
        return [all_faultnames.copy() for _ in all_datanames]
    
    # Case 2: List format
    elif isinstance(data_faults_config, list):
        if len(data_faults_config) != len(all_datanames):
            raise ValueError(f"Length of {param_name} ({len(data_faults_config)}) must equal "
                           f"number of datasets ({len(all_datanames)})")
        
        result = []
        for i, item in enumerate(data_faults_config):
            try:
                normalized = _normalize_fault_item(item, all_faultnames, f"{param_name}[{i}]")
                result.append(normalized)
            except ValueError as e:
                raise ValueError(f"Error in {param_name}[{i}] for dataset '{all_datanames[i]}': {str(e)}")
        
        return result
    
    # Case 3: Dictionary format
    elif isinstance(data_faults_config, dict):
        result = []
        
        for dataname in all_datanames:
            if dataname in data_faults_config:
                try:
                    normalized = _normalize_fault_item(data_faults_config[dataname], 
                                                     all_faultnames, 
                                                     f"{param_name}['{dataname}']")
                    result.append(normalized)
                except ValueError as e:
                    raise ValueError(f"Error in {param_name}['{dataname}']: {str(e)}")
            else:
                # Default to all_faultnames for unspecified datasets
                result.append(all_faultnames.copy())
        
        # Check for invalid dataset names in config
        invalid_datasets = set(data_faults_config.keys()) - set(all_datanames)
        if invalid_datasets:
            raise ValueError(f"Invalid dataset names in {param_name}: {invalid_datasets}")
        
        return result
    
    else:
        raise ValueError(f"{param_name} must be None, a list, or a dictionary")


def parse_alpha_faults(alpha_faults_config, all_faultnames, param_name="alphaFaults"):
    """
    Parse alphaFaults configuration with enhanced flexibility.
    
    Parameters:
    -----------
    alpha_faults_config : None, or list
        Configuration for alpha faults
    all_faultnames : list
        List of all available fault names
    param_name : str, optional
        Name of the parameter for error messages
        
    Returns:
    --------
    list
        List of fault name lists for each alpha
        
    Examples:
    ---------
    >>> parse_alpha_faults(None, ["f1", "f2"])
    [["f1", "f2"]]
    
    >>> parse_alpha_faults([None], ["f1", "f2"])
    [["f1", "f2"]]

    >>> parse_alpha_faults([["f1"], ["f2"]], ["f1", "f2"])
    [["f1"], ["f2"]]
    """
    
    def _normalize_fault_subset(item, all_faultnames, param_name):
        """Normalize a fault subset item"""
        if isinstance(item, str):
            if item not in all_faultnames:
                raise ValueError(f"Fault name '{item}' in {param_name} not found in all_faultnames")
            return [item]
        elif isinstance(item, list):
            if not set(item).issubset(set(all_faultnames)):
                invalid_names = set(item) - set(all_faultnames)
                raise ValueError(f"Invalid fault names in {param_name}: {invalid_names}")
            return item.copy()
        else:
            raise ValueError(f"Invalid fault specification in {param_name}: {item}")
    
    # Case 1: None or [None] - single alpha case
    if alpha_faults_config is None or (isinstance(alpha_faults_config, list) and 
                                      len(alpha_faults_config) == 1 and 
                                      alpha_faults_config[0] is None):
        return [all_faultnames.copy()]
    
    # Case 2: List format
    elif isinstance(alpha_faults_config, list):
        if len(alpha_faults_config) > len(all_faultnames):
            raise ValueError(f"Length of {param_name} ({len(alpha_faults_config)}) must be less than or equal to "
                           f"number of all faults ({len(all_faultnames)})")
        
        result = []
        all_assigned_faults = set()
        
        for i, item in enumerate(alpha_faults_config):
            try:
                normalized = _normalize_fault_subset(item, all_faultnames, f"{param_name}[{i}]")
                result.append(normalized)
                
                # Check for overlaps
                item_set = set(normalized)
                overlap = all_assigned_faults.intersection(item_set)
                if overlap:
                    raise ValueError(f"Fault names {overlap} appear in multiple alpha groups")
                all_assigned_faults.update(item_set)
                
            except ValueError as e:
                raise ValueError(f"Error in {param_name}[{i}]: {str(e)}")
        
        # Check for complete coverage
        if all_assigned_faults != set(all_faultnames):
            missing = set(all_faultnames) - all_assigned_faults
            raise ValueError(f"{param_name} does not cover all fault names. Missing: {missing}")
        
        return result
    
    else:
        raise ValueError(f"{param_name} must be None, or a list")

def parse_sigmas_config(sigmas_config, dataset_names, param_name='initial_value'):
    """
    Parse sigmas configuration supporting single, individual, and grouped modes
    
    Args:
        sigmas_config (dict): Sigmas configuration dictionary
        dataset_names (list): Dataset names list, must be provided
        param_name (str): Name of the initial value parameter ('initial_value' or 'values'), default 'initial_value'
    
    Returns:
        dict: Dictionary containing parsed results
            - mode: str, Mode type
            - update: list, Update flags for each parameter group
            - initial_value: list, Initial values for each parameter group (float)
            - dataset_param_indices: list, Parameter group index for each dataset
            - log_scaled: bool, Whether log scaled
            - num_datasets: int, Number of datasets
            - total_params: int, Total number of parameter groups
            - updatable_params: int, Number of parameters to be updated
            - groups: dict, Groups definition (for grouped mode)
    
    Examples:
        # Example 1: Using default 'initial_value' parameter name
        sigmas_config = {
            'update': True,
            'initial_value': [0.3181, 0.6062],
            'log_scaled': True
        }
        dataset_names = ['InSAR_A', 'InSAR_D']
        result = parse_sigmas_config(sigmas_config, dataset_names)
        
        # Example 2: Using 'values' parameter name
        sigmas_config = {
            'update': True,
            'values': [0.3181, 0.6062],  # Using 'values' instead of 'initial_value'
            'log_scaled': True
        }
        dataset_names = ['InSAR_A', 'InSAR_D']
        result = parse_sigmas_config(sigmas_config, dataset_names, param_name='values')
        
        # Example 3: Individual mode with dictionary values
        sigmas_config = {
            'mode': 'individual',
            'update': [True, False, True, True, False],
            'initial_value': {
                'InSAR_A': 0.3181,
                'GPS_E': 0.5,
                'GPS_N': 0.6
                # Missing datasets default to 0.0
            },
            'log_scaled': True
        }
        dataset_names = ['InSAR_A', 'InSAR_D', 'GPS_E', 'GPS_N', 'GPS_U']
        result = parse_sigmas_config(sigmas_config, dataset_names)
        
        # Example 4: Single mode (all datasets share one parameter)
        sigmas_config = {
            'mode': 'single',
            'update': True,
            'values': 0.5,  # Using 'values' parameter name
            'log_scaled': True
        }
        dataset_names = ['InSAR_A', 'InSAR_D', 'GPS_E', 'GPS_N', 'GPS_U']
        result = parse_sigmas_config(sigmas_config, dataset_names, param_name='values')
        
        # Example 5: Grouped mode with list values
        sigmas_config = {
            'mode': 'grouped',
            'groups': {
                'InSAR_group': ['InSAR_A', 'InSAR_D'],
                'GPS_horizontal': ['GPS_E', 'GPS_N'],
                'GPS_vertical': ['GPS_U']
            },
            'update': [True, False, True],
            'initial_value': [0.3181, 0.6062, 0.8],
            'log_scaled': True
        }
        dataset_names = ['InSAR_A', 'InSAR_D', 'GPS_E', 'GPS_N', 'GPS_U']
        result = parse_sigmas_config(sigmas_config, dataset_names)
    """
    
    # Validate dataset_names must be provided
    if not dataset_names:
        raise ValueError("dataset_names must be provided and cannot be empty")
    
    # Default values
    default_config = {
        'update': True,
        param_name: 0.0,  # Use the specified parameter name
        'log_scaled': False
    }
    
    # Merge default configuration
    config = {**default_config, **sigmas_config}
    
    # Number of datasets
    num_datasets = len(dataset_names)
    
    # Detect mode - default to individual (backward compatibility)
    mode = config.get('mode', 'individual')
    
    # Parse different modes
    if mode == 'single':
        # Single mode: all datasets share one parameter
        update = config['update']
        initial_value = config[param_name]  # Use the specified parameter name
        
        # In single mode, do not allow multi-value lists (except single-value list)
        if isinstance(update, (list, tuple, np.ndarray)):
            if len(update) == 1:
                update_list = [update[0]]
            else:
                raise ValueError(f"In single mode, update cannot be a multi-value list, current length: {len(update)}")
        else:
            update_list = [update]
            
        if isinstance(initial_value, (list, tuple)):
            if len(initial_value) == 1:
                initial_value_list = [float(initial_value[0])]
            else:
                raise ValueError(f"In single mode, {param_name} cannot be a multi-value list, current length: {len(initial_value)}")
        elif isinstance(initial_value, dict):
            raise ValueError(f"In single mode, {param_name} cannot be a dictionary")
        else:
            initial_value_list = [float(initial_value)]
            
        dataset_param_indices = [0] * num_datasets
        total_params = 1
        groups = None
        
    elif mode == 'individual':
        # Individual mode: each dataset has independent parameters
        update = config['update']
        initial_value = config[param_name]  # Use the specified parameter name
        
        # Handle update parameters
        if isinstance(update, (list, tuple, np.ndarray)):
            if len(update) != num_datasets:
                raise ValueError(f"In individual mode, update list length ({len(update)}) must equal number of datasets ({num_datasets})")
            update_list = list(update)
        else:
            # Expand single value to dataset size
            update_list = [update] * num_datasets
            
        # Handle initial_value parameters
        if isinstance(initial_value, dict):
            # Dictionary mode: specify dataset names
            initial_value_list = []
            for dataset_name in dataset_names:
                if dataset_name in initial_value:
                    initial_value_list.append(float(initial_value[dataset_name]))
                else:
                    initial_value_list.append(0.0)  # Default to 0.0
        elif isinstance(initial_value, (list, tuple, np.ndarray)):
            if len(initial_value) != num_datasets:
                raise ValueError(f"In individual mode, {param_name} list length ({len(initial_value)}) must equal number of datasets ({num_datasets})")
            initial_value_list = [float(val) for val in initial_value]
        else:
            # Expand single value to dataset size
            initial_value_list = [float(initial_value)] * num_datasets
        
        # Each dataset corresponds to its own parameter group index
        dataset_param_indices = list(range(num_datasets))
        total_params = num_datasets
        groups = None
        
    elif mode == 'grouped':
        # Grouped mode: custom grouping
        groups = config.get('groups', {})
        update_config = config.get('update', [])
        initial_value_config = config.get(param_name, [])  # Use the specified parameter name
        
        if not groups:
            raise ValueError("In grouped mode, groups parameter must be provided")
        
        # Build mapping from dataset names to parameter groups
        dataset_to_group = {}
        group_names = list(groups.keys())
        
        for group_idx, (group_name, datasets) in enumerate(groups.items()):
            for dataset_name in datasets:
                if dataset_name in dataset_to_group:
                    raise ValueError(f"Dataset '{dataset_name}' is assigned to multiple groups")
                dataset_to_group[dataset_name] = group_idx
        
        # Verify all datasets are grouped
        missing_datasets = set(dataset_names) - set(dataset_to_group.keys())
        if missing_datasets:
            raise ValueError(f"The following datasets are not grouped: {missing_datasets}")
        
        # Build parameter mapping: [0, 0, 1, 1, 2, 2] style
        dataset_param_indices = [dataset_to_group[name] for name in dataset_names]
        
        # Handle update parameters
        num_groups = len(group_names)
        
        if isinstance(update_config, (list, tuple, np.ndarray)):
            if len(update_config) != num_groups:
                raise ValueError(f"In grouped mode, update list length ({len(update_config)}) must equal number of groups ({num_groups})")
            update_list = list(update_config)
        else:
            # Expand single value to number of groups
            update_list = [update_config] * num_groups
            
        # Handle initial_value parameters
        if isinstance(initial_value_config, dict):
            # Dictionary mode: specify group names
            initial_value_list = []
            for group_name in group_names:
                if group_name in initial_value_config:
                    initial_value_list.append(float(initial_value_config[group_name]))
                else:
                    initial_value_list.append(0.0)  # Default to 0.0
        elif isinstance(initial_value_config, (list, tuple, np.ndarray)):
            if len(initial_value_config) != num_groups:
                raise ValueError(f"In grouped mode, {param_name} list length ({len(initial_value_config)}) must equal number of groups ({num_groups})")
            initial_value_list = [float(val) for val in initial_value_config]
        else:
            # Expand single value to number of groups
            initial_value_list = [float(initial_value_config)] * num_groups
            
        total_params = num_groups
    
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # Calculate number of updatable parameters and create updatable index mapping
    updatable_param_indices = []
    updatable_counter = 0
    
    for update_flag in update_list:
        if update_flag:
            updatable_param_indices.append(updatable_counter)
            updatable_counter += 1
        else:
            updatable_param_indices.append(-1)  # -1 indicates not updatable
    
    updatable_params = updatable_counter

    result = {
            'mode': mode,
            'update': np.array(update_list, dtype=bool),
            param_name: np.array(initial_value_list, dtype=float),
            'dataset_param_indices': np.array(dataset_param_indices, dtype=int),
            'updatable_param_indices': np.array(updatable_param_indices, dtype=int),
            'log_scaled': config['log_scaled'],
            'num_datasets': num_datasets,
            'total_params': total_params,
            'updatable_params': updatable_params,
            'groups': groups
        }
    return {**config, **result}

def parse_alpha_config(alpha_config, faultnames, param_name='initial_value'):
    """
    Parse the alpha (smoothing/regularization) configuration for VCE, supporting 'single', 'individual', and 'grouped' modes.

    This function standardizes the parsing of alpha (regularization) configuration for flexible VCE workflows.
    It supports:
      - Single mode: all faults share one parameter
      - Individual mode: each fault has its own parameter
      - Grouped mode: custom grouping of faults, using 'faults' (list of lists) or legacy 'groups' (dict).

    Parameters
    ----------
    alpha_config : dict
        Configuration dictionary for alpha, e.g.:
            {
                'mode': 'grouped',
                'faults': [['faultA', 'faultB'], ['faultC']],
                'update': [True, False],
                'initial_value': [0.1, 0.2],
                'log_scaled': True
            }
    faultnames : list of str
        List of fault names (must match the faults in the problem)
    param_name : str, optional
        Name of the parameter for initial values (default: 'initial_value')

    Returns
    -------
    dict
        Dictionary with parsed alpha configuration:
            - mode: str, mode type ('single', 'individual', 'grouped')
            - update: np.ndarray, update flags for each parameter group
            - initial_value: np.ndarray, initial values for each parameter group
            - fault_param_indices: np.ndarray, parameter group index for each fault
            - updatable_param_indices: np.ndarray, updatable index for each group (-1 if not updatable)
            - log_scaled: bool, whether log scaling is used
            - num_faults: int, number of faults
            - total_params: int, number of parameter groups
            - updatable_params: int, number of updatable groups
            - faults: list of lists, group definitions (for grouped mode)

    Examples
    --------
    # Example 1: Single mode (all faults share one alpha)
    alpha_config = {
        'mode': 'single',
        'update': True,
        'initial_value': 0.1,
        'log_scaled': False
    }
    faultnames = ['faultA', 'faultB', 'faultC']
    result = parse_alpha_config(alpha_config, faultnames)
    # result['mode'] == 'single'
    # result['update'] == array([True])
    # result['initial_value'] == array([0.1])
    # result['fault_param_indices'] == array([0, 0, 0])

    # Example 2: Individual mode
    alpha_config = {
        'mode': 'individual',
        'update': [True, False, True],
        'initial_value': [0.1, 0.2, 0.3],
        'log_scaled': True
    }
    faultnames = ['faultA', 'faultB', 'faultC']
    result = parse_alpha_config(alpha_config, faultnames)

    # Example 3: Grouped mode with faults (recommended)
    alpha_config = {
        'mode': 'grouped',
        'faults': [['faultA', 'faultB'], ['faultC']],
        'update': [True, False],
        'initial_value': [0.1, 0.2],
        'log_scaled': False
    }
    faultnames = ['faultA', 'faultB', 'faultC']
    result = parse_alpha_config(alpha_config, faultnames)

    # Example 4: Grouped mode with legacy groups (dict)
    alpha_config = {
        'mode': 'grouped',
        'groups': {'g1': ['faultA', 'faultB'], 'g2': ['faultC']},
        'update': [True, False],
        'initial_value': [0.1, 0.2],
        'log_scaled': False
    }
    faultnames = ['faultA', 'faultB', 'faultC']
    result = parse_alpha_config(alpha_config, faultnames)

    Notes
    -----
    - In 'grouped' mode, all faults must be assigned to exactly one group.
    - The length of 'update' and 'initial_value' must match the number of groups (grouped mode) or faults (individual mode).
    - The function returns indices for mapping faults to parameter groups and updatable groups.
    """

    if not faultnames:
        raise ValueError("faultnames must be provided and cannot be empty")

    num_faults = len(faultnames)
    mode = alpha_config.get('mode', 'single')
    log_scaled = alpha_config.get('log_scaled', True)
    enabled = alpha_config.get('enabled', True)

    # --- Single mode: all faults share one parameter ---
    if mode == 'single':
        update = alpha_config.get('update', True)
        initial_value = alpha_config.get(param_name, 0.0)
        update_list = [update] if not isinstance(update, (list, tuple, np.ndarray)) else list(update)
        if len(update_list) != 1:
            raise ValueError("In 'single' mode, 'update' must be a single value")
        if isinstance(initial_value, (list, tuple, np.ndarray)):
            if len(initial_value) != 1:
                raise ValueError("In 'single' mode, 'initial_value' must be a single value")
            initial_value_list = [float(initial_value[0])]
        else:
            initial_value_list = [float(initial_value)]
        fault_param_indices = [0] * num_faults
        total_params = 1
        group_faults = [faultnames.copy()]

    # --- Individual mode: each fault has its own parameter ---
    elif mode == 'individual':
        update = alpha_config.get('update', True)
        initial_value = alpha_config.get(param_name, 0.0)
        if isinstance(update, (list, tuple, np.ndarray)):
            if len(update) != num_faults:
                raise ValueError("In 'individual' mode, 'update' length must match number of faults")
            update_list = list(update)
        else:
            update_list = [update] * num_faults
        if isinstance(initial_value, dict):
            initial_value_list = [float(initial_value.get(name, 0.0)) for name in faultnames]
        elif isinstance(initial_value, (list, tuple, np.ndarray)):
            if len(initial_value) != num_faults:
                raise ValueError("In 'individual' mode, 'initial_value' length must match number of faults")
            initial_value_list = [float(v) for v in initial_value]
        else:
            initial_value_list = [float(initial_value)] * num_faults
        fault_param_indices = list(range(num_faults))
        total_params = num_faults
        group_faults = [[f] for f in faultnames]

    # --- Grouped mode: custom grouping ---
    elif mode == 'grouped':
        # Prefer 'faults' (list of lists), fallback to 'groups' (dict)
        group_faults = alpha_config.get('faults', None)
        if group_faults is None:
            groups = alpha_config.get('groups', None)
            if groups is None:
                raise ValueError("In 'grouped' mode, 'faults' (list of lists) or 'groups' (dict) must be provided")
            # Convert groups dict to list of lists
            group_faults = [groups[k] for k in groups]
        if not isinstance(group_faults, list) or not all(isinstance(g, list) for g in group_faults):
            raise ValueError("'faults' must be a list of lists of fault names")
        num_groups = len(group_faults)
        # Build fault to group index mapping
        fault_to_group = {}
        for idx, flist in enumerate(group_faults):
            for f in flist:
                if f in fault_to_group:
                    raise ValueError(f"Fault '{f}' assigned to multiple groups")
                fault_to_group[f] = idx
        missing = set(faultnames) - set(fault_to_group.keys())
        if missing:
            raise ValueError(f"Faults not assigned to any group: {missing}")
        fault_param_indices = [fault_to_group[name] for name in faultnames]
        update = alpha_config.get('update', True)
        if isinstance(update, (list, tuple, np.ndarray)):
            if len(update) != num_groups:
                raise ValueError("In 'grouped' mode, 'update' length must match number of groups")
            update_list = list(update)
        else:
            update_list = [update] * num_groups
        initial_value = alpha_config.get(param_name, 0.0)
        if isinstance(initial_value, dict):
            initial_value_list = [float(initial_value.get(str(i), 0.0)) for i in range(num_groups)]
        elif isinstance(initial_value, (list, tuple, np.ndarray)):
            if len(initial_value) != num_groups:
                raise ValueError("In 'grouped' mode, 'initial_value' length must match number of groups")
            initial_value_list = [float(v) for v in initial_value]
        else:
            initial_value_list = [float(initial_value)] * num_groups
        total_params = num_groups

    else:
        raise ValueError(f"Unknown alpha mode: {mode}")

    # --- Updatable parameter indices ---
    updatable_param_indices = []
    updatable_counter = 0
    for flag in update_list:
        if flag:
            updatable_param_indices.append(updatable_counter)
            updatable_counter += 1
        else:
            updatable_param_indices.append(-1)
    updatable_params = updatable_counter

    result = {
        'enabled': enabled,
        'update': np.array(update_list, dtype=bool),
        'initial_value': np.array(initial_value_list, dtype=float),
        'log_scaled': log_scaled,
        'faults': group_faults,
        'mode': mode,
        'fault_param_indices': np.array(fault_param_indices, dtype=int),
        'updatable_param_indices': np.array(updatable_param_indices, dtype=int),
        'num_faults': num_faults,
        'total_params': total_params,
        'updatable_params': updatable_params
    }
    return {**alpha_config, **result}

def parse_bounds(bounds_config, param_names, param_type="parameter"):
    """
    Parse bounds configuration with support for defaults.
    
    Parameters:
    -----------
    bounds_config : dict
        Bounds configuration dictionary
    param_names : list
        List of parameter names that need bounds
    param_type : str, optional
        Type of parameter for error messages. Default is "parameter"
        
    Returns:
    --------
    dict
        Dictionary with bounds for each parameter
        
    Examples:
    ---------
    >>> bounds = {"defaults": [0, 1], "sigma_0": [0, 0.5]}
    >>> parse_bounds(bounds, ["sigma_0", "sigma_1"])
    {"sigma_0": [0, 0.5], "sigma_1": [0, 1]}
    """
    result = {}
    defaults = bounds_config.get('defaults', None)
    
    for name in param_names:
        if name in bounds_config:
            result[name] = bounds_config[name]
        elif defaults is not None:
            result[name] = defaults
        else:
            raise ValueError(f"No bounds specified for {param_type} '{name}' and no defaults provided")
    
    return result


def parse_log_scaled(config, n_datasets, param_name="log_scaled"):
    """
    Parse the 'log_scaled' parameter from configuration.
    
    Similar to parse_update but specifically for log_scaled parameters.
    """
    return parse_update(config, n_datasets, param_name)


def validate_config_list(config_list, expected_length, param_name):
    """
    Validate that a configuration list has the expected length.
    
    Parameters:
    -----------
    config_list : list
        The configuration list to validate
    expected_length : int
        Expected length of the list
    param_name : str
        Name of the parameter for error messages
        
    Raises:
    -------
    ValueError
        If the list length doesn't match expected length
    """
    if len(config_list) != expected_length:
        raise ValueError(f"Length of {param_name} ({len(config_list)}) does not match expected length ({expected_length})")


def expand_single_value(value, n_items):
    """
    Expand a single value to a list of specified length.
    
    Parameters:
    -----------
    value : any
        Single value to expand
    n_items : int
        Number of items in the resulting list
        
    Returns:
    --------
    list
        List with the value repeated n_items times
    """
    return [value] * n_items


def merge_with_defaults(specific_config, default_config):
    """
    Merge specific configuration with defaults, handling nested dictionaries.
    
    Parameters:
    -----------
    specific_config : dict
        Specific configuration parameters
    default_config : dict
        Default configuration parameters
        
    Returns:
    --------
    dict
        Merged configuration with specific values taking precedence
    """
    merged = default_config.copy()
    
    for key, value in specific_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_with_defaults(value, merged[key])
        else:
            merged[key] = value
            
    return merged

# ---------------------------------Euler config parsing---------------------------------#
def parse_euler_constraints_config(euler_config, faultnames, dataset_names=None):
    """
    Parse Euler pole constraints configuration for tectonic block motion modeling.
    
    Parameters:
    -----------
    euler_config : dict
        Euler constraints configuration dictionary
    faultnames : list
        List of fault names in the problem
    dataset_names : list, optional
        List of dataset names (required if any fault uses 'dataset' block type)
        
    Returns:
    --------
    dict
        Parsed Euler constraints configuration with validated parameters
        
    Examples:
    ---------
    # Basic configuration with reference strike
    euler_config = {
        'enabled': True,
        'defaults': {
            'reference_strike': 45.0,  # degrees, reference direction for projection
            'motion_sense': 'dextral'  # or 'sinistral' for motion sense
        },
        'faults': {
            'HH_Main': {
                'blocks': ['GPS_data_A', 'GPS_data_B'],
                'block_names': ['South_China_Block', 'North_Burma_Block'],
                'reference_strike': 30.0,  # fault-specific reference strike
                'motion_sense': 'sinistral'  # left-lateral motion sense
            }
        }
    }
    """
    
    if not euler_config.get('enabled', False):
        return {
            'enabled': False,
            'faults': {},
            'defaults': {}
        }
    
    # Define default values
    default_defaults = {
        'euler_pole_units': ['degrees', 'degrees', 'degrees_per_myr'],
        'euler_vector_units': ['radians_per_year', 'radians_per_year', 'radians_per_year'],
        'fix_reference_block': None,
        'apply_to_patches': None,
        'normalization': True,
        'regularization': 0.01,
        'block_types': ['dataset', 'dataset'],  # Default to two datasets
        'reference_strike': 0.0,  # degrees, reference direction for projection
        'motion_sense': 'dextral'  # default motion sense
    }
    
    # Valid units and parameters for validation
    valid_angle_units = ['degrees', 'radians']
    valid_angular_velocity_units = [
        'degrees_per_myr', 'radians_per_year', 'radians_per_myr', 
        'degrees_per_year', 'radians_per_second'
    ]
    valid_block_types = ['dataset', 'euler_pole', 'euler_vector']
    valid_motion_senses = ['dextral', 'sinistral', 'right_lateral', 'left_lateral']
    
    # Parse defaults
    defaults = merge_with_defaults(
        euler_config.get('defaults', {}), 
        default_defaults
    )
    
    # Validate default units
    def validate_units(units, unit_type):
        if unit_type == 'euler_pole':
            if len(units) != 3:
                raise ValueError(f"euler_pole_units must have 3 elements [lat, lon, omega], got {len(units)}")
            if units[0] not in valid_angle_units or units[1] not in valid_angle_units:
                raise ValueError(f"Invalid angle units in euler_pole_units: {units[:2]}")
            if units[2] not in valid_angular_velocity_units:
                raise ValueError(f"Invalid angular velocity unit in euler_pole_units: {units[2]}")
        elif unit_type == 'euler_vector':
            if len(units) != 3:
                raise ValueError(f"euler_vector_units must have 3 elements [wx, wy, wz], got {len(units)}")
            if not all(u in valid_angular_velocity_units for u in units):
                raise ValueError(f"Invalid angular velocity units in euler_vector_units: {units}")
    
    def validate_reference_strike(strike, context):
        """Validate reference strike angle"""
        if not isinstance(strike, (int, float)):
            raise ValueError(f"reference_strike in {context} must be a number")
        # Normalize to [0, 360) range
        return float(strike % 360)
    
    def validate_motion_sense(motion_sense, context):
        """Validate motion sense"""
        if motion_sense not in valid_motion_senses:
            raise ValueError(f"Invalid motion_sense '{motion_sense}' in {context}. "
                           f"Valid options: {valid_motion_senses}")
        return motion_sense
    
    validate_units(defaults['euler_pole_units'], 'euler_pole')
    validate_units(defaults['euler_vector_units'], 'euler_vector')
    
    # Validate defaults
    defaults['reference_strike'] = validate_reference_strike(defaults['reference_strike'], 'defaults')
    defaults['motion_sense'] = validate_motion_sense(defaults['motion_sense'], 'defaults')
    
    # Parse fault-specific configurations
    fault_configs = {}
    faults_config = euler_config.get('faults', {})
    
    for fault_name in faultnames:
        if fault_name not in faults_config:
            # Skip faults not configured for Euler constraints
            continue
            
        fault_config = faults_config[fault_name]
        
        # Validate required parameters
        if 'blocks' not in fault_config:
            raise ValueError(f"Missing 'blocks' for fault '{fault_name}'")
            
        blocks = fault_config['blocks']
        
        # Use default block_types if not provided
        block_types = fault_config.get('block_types', defaults['block_types'])
        
        # Validate block_types and blocks consistency
        if len(block_types) != 2:
            raise ValueError(f"Fault '{fault_name}' must have exactly 2 block_types, got {len(block_types)}")
        if len(blocks) != 2:
            raise ValueError(f"Fault '{fault_name}' must have exactly 2 blocks, got {len(blocks)}")
            
        for block_type in block_types:
            if block_type not in valid_block_types:
                raise ValueError(f"Invalid block_type '{block_type}' for fault '{fault_name}'")
        
        # Validate blocks based on block_types
        for i, (block_type, block) in enumerate(zip(block_types, blocks)):
            if block_type == 'dataset':
                if not isinstance(block, str):
                    raise ValueError(f"Block {i} for fault '{fault_name}' with type 'dataset' must be a string")
                if dataset_names and block not in dataset_names:
                    raise ValueError(f"Dataset '{block}' for fault '{fault_name}' not found in dataset_names")
            elif block_type in ['euler_pole', 'euler_vector']:
                if not isinstance(block, (list, tuple)) or len(block) != 3:
                    raise ValueError(f"Block {i} for fault '{fault_name}' with type '{block_type}' must be a 3-element array")
                if not all(isinstance(x, (int, float)) for x in block):
                    raise ValueError(f"Block {i} for fault '{fault_name}' must contain numeric values")
        
        # Merge with defaults
        merged_config = merge_with_defaults(fault_config, {
            'block_types': block_types,  # Use the resolved block_types
            'fix_reference_block': defaults['fix_reference_block'],
            'apply_to_patches': defaults['apply_to_patches'],
            'reference_strike': defaults['reference_strike'],
            'motion_sense': defaults['motion_sense'],
            'units': {
                'euler_pole_units': defaults['euler_pole_units'],
                'euler_vector_units': defaults['euler_vector_units']
            }
        })
        
        # Validate fault-specific parameters
        if 'reference_strike' in fault_config:
            merged_config['reference_strike'] = validate_reference_strike(
                fault_config['reference_strike'], 
                f"fault '{fault_name}'"
            )
        
        if 'motion_sense' in fault_config:
            merged_config['motion_sense'] = validate_motion_sense(
                fault_config['motion_sense'],
                f"fault '{fault_name}'"
            )
        
        # Generate default block names if not provided
        if 'block_names' not in merged_config:
            merged_config['block_names'] = [f"{fault_name}_blockA", f"{fault_name}_blockB"]
        elif len(merged_config['block_names']) != 2:
            raise ValueError(f"Fault '{fault_name}' must have exactly 2 block_names if provided")
        
        # Convert units to standard format and store conversion factors
        if 'units' in fault_config:
            fault_units = fault_config['units']
            if 'euler_pole_units' in fault_units:
                validate_units(fault_units['euler_pole_units'], 'euler_pole')
            if 'euler_vector_units' in fault_units:
                validate_units(fault_units['euler_vector_units'], 'euler_vector')
        
        # Convert blocks to standard units (store both original and standardized)
        blocks_standard = []
        for i, (block_type, block) in enumerate(zip(block_types, blocks)):
            if block_type == 'dataset':
                blocks_standard.append(block)  # Dataset name, no conversion needed
            elif block_type in ['euler_pole', 'euler_vector']:
                # Convert to standard units
                if block_type == 'euler_pole':
                    units_config = merged_config['units']['euler_pole_units']
                    unit_info = parse_euler_units(units_config, 'euler_pole')
                else:  # euler_vector
                    units_config = merged_config['units']['euler_vector_units']
                    unit_info = parse_euler_units(units_config, 'euler_vector')
                
                # Apply unit conversions
                converted_block = [
                    block[j] * unit_info['conversion_factors'][j] 
                    for j in range(3)
                ]
                blocks_standard.append(converted_block)
        
        merged_config['blocks_standard'] = blocks_standard
        
        # Validate fix_reference_block
        fix_ref = merged_config['fix_reference_block']
        if fix_ref is not None and fix_ref not in [0, 1]:
            raise ValueError(f"fix_reference_block for fault '{fault_name}' must be None, 0, or 1")
        
        # Validate apply_to_patches
        apply_patches = merged_config['apply_to_patches']
        if apply_patches is not None:
            if not isinstance(apply_patches, (list, tuple)):
                raise ValueError(f"apply_to_patches for fault '{fault_name}' must be a list or None")
            if not all(isinstance(x, int) and x >= 0 for x in apply_patches):
                raise ValueError(f"apply_to_patches for fault '{fault_name}' must contain non-negative integers")
        
        fault_configs[fault_name] = merged_config
    
    return {
        'enabled': True,
        'defaults': defaults,
        'faults': fault_configs,
        'configured_faults': list(fault_configs.keys())
    }


def parse_euler_units(units_config, unit_type):
    """
    Parse and validate Euler pole/vector units configuration.
    
    Parameters:
    -----------
    units_config : list
        List of unit strings
    unit_type : str
        Either 'euler_pole' or 'euler_vector'
        
    Returns:
    --------
    dict
        Dictionary with parsed unit information and conversion factors
    """
    
    # Conversion factors to standard units (radians and radians/year)
    angle_conversions = {
        'degrees': np.pi / 180.0,
        'radians': 1.0
    }
    
    angular_velocity_conversions = {
        'radians_per_year': 1.0,
        'radians_per_myr': 1.0e-6,
        'radians_per_second': 365.25 * 24 * 3600,  # Convert to per year
        'degrees_per_year': np.pi / 180.0,
        'degrees_per_myr': np.pi / 180.0 * 1.0e-6
    }
    
    if unit_type == 'euler_pole':
        if len(units_config) != 3:
            raise ValueError("Euler pole units must have 3 elements: [latitude, longitude, angular_velocity]")
        
        lat_factor = angle_conversions.get(units_config[0])
        lon_factor = angle_conversions.get(units_config[1])
        omega_factor = angular_velocity_conversions.get(units_config[2])
        
        if lat_factor is None:
            raise ValueError(f"Invalid latitude unit: {units_config[0]}")
        if lon_factor is None:
            raise ValueError(f"Invalid longitude unit: {units_config[1]}")
        if omega_factor is None:
            raise ValueError(f"Invalid angular velocity unit: {units_config[2]}")
            
        return {
            'units': units_config,
            'conversion_factors': [lat_factor, lon_factor, omega_factor],
            'standard_units': ['radians', 'radians', 'radians_per_year']
        }
    
    elif unit_type == 'euler_vector':
        if len(units_config) != 3:
            raise ValueError("Euler vector units must have 3 elements: [wx, wy, wz]")
        
        conversion_factors = []
        for unit in units_config:
            factor = angular_velocity_conversions.get(unit)
            if factor is None:
                raise ValueError(f"Invalid angular velocity unit: {unit}")
            conversion_factors.append(factor)
            
        return {
            'units': units_config,
            'conversion_factors': conversion_factors,
            'standard_units': ['radians_per_year'] * 3
        }
    
    else:
        raise ValueError(f"Invalid unit_type: {unit_type}")
# --------------------------------------------------------------------------------------#