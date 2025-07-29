"""
Configuration parsing utilities for bayesian_config.py and related modules.
"""

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