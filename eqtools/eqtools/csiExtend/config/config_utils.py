"""
Configuration parsing utilities for bayesian_config.py and related modules.
"""
import numpy as np
import logging 
# Setup module-level logger
logger = logging.getLogger(__name__)


_OBSERVATION_UNIT_ALIASES = {
    "m": ("m", "displacement", 1.0),
    "meter": ("m", "displacement", 1.0),
    "meters": ("m", "displacement", 1.0),
    "cm": ("cm", "displacement", 1.0e-2),
    "centimeter": ("cm", "displacement", 1.0e-2),
    "centimeters": ("cm", "displacement", 1.0e-2),
    "mm": ("mm", "displacement", 1.0e-3),
    "millimeter": ("mm", "displacement", 1.0e-3),
    "millimeters": ("mm", "displacement", 1.0e-3),
    "m/yr": ("m/yr", "rate", 1.0),
    "m/year": ("m/yr", "rate", 1.0),
    "meter/year": ("m/yr", "rate", 1.0),
    "meters/year": ("m/yr", "rate", 1.0),
    "cm/yr": ("cm/yr", "rate", 1.0e-2),
    "cm/year": ("cm/yr", "rate", 1.0e-2),
    "centimeter/year": ("cm/yr", "rate", 1.0e-2),
    "centimeters/year": ("cm/yr", "rate", 1.0e-2),
    "mm/yr": ("mm/yr", "rate", 1.0e-3),
    "mm/year": ("mm/yr", "rate", 1.0e-3),
    "millimeter/year": ("mm/yr", "rate", 1.0e-3),
    "millimeters/year": ("mm/yr", "rate", 1.0e-3),
}


def parse_observation_unit(unit, default=None):
    """Parse the unified observation unit used by the inversion matrix.

    ``units.observation`` is the unit after any reader/factor conversion and
    before linear inversion.  It is intentionally global: ECAT assumes data,
    Green's functions, slip variables and constraint right-hand sides are
    already in the same numerical unit.
    """
    assumed = unit is None
    if assumed:
        unit = default
    if unit is None:
        return {
            "observation": None,
            "kind": None,
            "to_si": None,
            "from_si": None,
            "assumed": True,
        }

    key = str(unit).strip().lower().replace(" ", "").replace("_", "")
    key = key.replace("peryear", "/year").replace("peryr", "/yr")
    key = key.replace("yr^-1", "/yr").replace("year^-1", "/year")
    try:
        canonical, kind, to_si = _OBSERVATION_UNIT_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted({value[0] for value in _OBSERVATION_UNIT_ALIASES.values()}))
        msg = f"Unsupported units.observation '{unit}'. Supported units: {allowed}"
        logger.error(msg)
        raise ValueError(msg) from exc
    return {
        "observation": canonical,
        "kind": kind,
        "to_si": float(to_si),
        "from_si": float(1.0 / to_si),
        "assumed": assumed,
    }


def normalize_units_config(units_config=None):
    """Normalize the optional top-level ``units`` config section."""
    if units_config is None:
        return {"observation": None}
    if isinstance(units_config, str):
        units_config = {"observation": units_config}
    if not isinstance(units_config, dict):
        msg = "units must be a mapping such as {'observation': 'm'}"
        logger.error(msg)
        raise ValueError(msg)
    observation = units_config.get("observation")
    if observation is not None:
        observation = parse_observation_unit(observation)["observation"]
    return {**units_config, "observation": observation}


def get_observation_unit_info(holder, default="m"):
    """Return parsed ``units.observation`` info from a config or inversion object."""
    config = getattr(holder, "config", holder)
    units = getattr(config, "units", None)
    if units is None and isinstance(config, dict):
        units = config.get("units")
    if units is None:
        units = {}
    if isinstance(units, str):
        units = {"observation": units}
    observation = units.get("observation") if isinstance(units, dict) else None
    return parse_observation_unit(observation, default=default)


def observation_to_m_factor(holder, default="m"):
    """Return factor converting observation displacement units to meters."""
    info = get_observation_unit_info(holder, default=default)
    if info["kind"] != "displacement":
        raise ValueError(f"Observation unit '{info['observation']}' is not a displacement unit")
    return info["to_si"]


def m_to_observation_factor(holder, default="m"):
    """Return factor converting meters to observation displacement units."""
    info = get_observation_unit_info(holder, default=default)
    if info["kind"] != "displacement":
        raise ValueError(f"Observation unit '{info['observation']}' is not a displacement unit")
    return info["from_si"]


def m_per_year_to_observation_factor(holder, default="m/yr"):
    """Return factor converting m/yr to the configured observation rate unit."""
    info = get_observation_unit_info(holder, default=default)
    if info["kind"] != "rate":
        raise ValueError(
            f"Observation unit '{info['observation']}' is not a rate unit; "
            "Euler/block interseismic loading requires units.observation like 'm/yr' or 'mm/yr'."
        )
    return info["from_si"]


def observation_to_m_per_year_factor(holder, default="m/yr"):
    """Return factor converting configured observation rate units to m/yr."""
    info = get_observation_unit_info(holder, default=default)
    if info["kind"] != "rate":
        raise ValueError(
            f"Observation unit '{info['observation']}' is not a rate unit; "
            "Euler/block interseismic loading requires units.observation like 'm/yr' or 'mm/yr'."
        )
    return info["to_si"]

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
                    msg = f"Index {idx} in {param_name} exceeds number of datasets ({n_datasets})"
                    logger.error(msg)
                    raise ValueError(msg)
                flags[idx] = True
            return flags
        elif dataset_names and all(isinstance(x, str) for x in update):
            # Name list
            flags = [False] * n_datasets
            for name in update:
                if name not in dataset_names:
                    msg = f"Dataset name '{name}' in {param_name} not found in dataset_names"
                    logger.error(msg)
                    raise ValueError(msg)
                idx = dataset_names.index(name)
                flags[idx] = True
            return flags
        else:
            msg = f"Invalid list format for {param_name}"
            logger.error(msg)
            raise ValueError(msg)
    
    elif isinstance(update, dict):
        # Compatible with old dictionary format
        if "true_indices" in update and "false_indices" in update:
            msg = f"Cannot specify both 'true_indices' and 'false_indices' in {param_name}"
            logger.error(msg)
            raise ValueError(msg)
        
        if "true_indices" in update:
            flags = [False] * n_datasets
            for idx in update["true_indices"]:
                if idx >= n_datasets:
                    msg = f"Index {idx} in {param_name}.true_indices exceeds number of datasets ({n_datasets})"
                    logger.error(msg)
                    raise ValueError(msg)
                flags[idx] = True
            return flags
        elif "false_indices" in update:
            flags = [True] * n_datasets
            for idx in update["false_indices"]:
                if idx >= n_datasets:
                    msg = f"Index {idx} in {param_name}.false_indices exceeds number of datasets ({n_datasets})"
                    logger.error(msg)
                    raise ValueError(msg)
                flags[idx] = False
            return flags
        else:
            msg = f"Dict format for {param_name} must contain either 'true_indices' or 'false_indices'"
            logger.error(msg)
            raise ValueError(msg)
    
    else:
        msg = f"Invalid format for {param_name}"
        logger.error(msg)
        raise ValueError(msg)


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
            msg = f"Length of '{print_name}' list ({len(initial_value)}) does not match number of datasets ({n_datasets})"
            logger.error(msg)
            raise ValueError(msg)
        
        # Convert all values to float
        processed_values = []
        for i, val in enumerate(initial_value):
            if not isinstance(val, (int, float)):
                msg = f"All values in '{print_name}' must be numbers, got {type(val)} at index {i}"
                logger.error(msg)
                raise ValueError(msg)
            processed_values.append(float(val))
        
        return processed_values
    
    # Handle dictionary format (dataset names to values mapping)
    elif isinstance(initial_value, dict):
        if dataset_names is None:
            msg = f"dataset_names must be provided when using dictionary format for {print_name}"
            logger.error(msg)
            raise ValueError(msg)
        
        processed_values = []
        for i, dataset_name in enumerate(dataset_names):
            if dataset_name in initial_value:
                val = initial_value[dataset_name]
                if not isinstance(val, (int, float)):
                    msg = f"Value for dataset '{dataset_name}' in '{print_name}' must be a number, got {type(val)}"
                    logger.error(msg)
                    raise ValueError(msg)
                processed_values.append(float(val))
            else:
                # Use default value for datasets not specified in dictionary
                processed_values.append(float(default_value))
        
        return processed_values
    
    else:
        msg = f"'{param_name}' must be a number, list of numbers, or dictionary mapping dataset names to numbers, got {type(initial_value)}"
        logger.error(msg)
        raise ValueError(msg)

def parse_data_faults(data_faults_config, all_faultnames, all_datanames, param_name="dataFaults"):
    """
    Parse dataFaults configuration with enhanced flexibility.
    
    Parameters:
    -----------
    data_faults_config : None, list, or dict
        Configuration for data faults. Supported forms:
        - None: every dataset uses all faults
        - list[None | str | list[str]]: one entry per dataset in all_datanames order
        - dict[str, None | str | list[str]]: map dataset name to fault selection
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

    Raises:
    -------
    ValueError
        Raised when:
        - data_faults_config is not None, a list, or a dict
        - list input length does not match the number of datasets
        - a fault name does not exist in all_faultnames
        - a dataset name in dict input does not exist in all_datanames
        - an item is not one of None, str, or list[str]

    Notes:
    ------
    Each dataset is validated independently. Unlike parse_alpha_faults,
    fault coverage does not need to be complete across datasets:
    - a dataset may use all faults, one fault, or any subset of faults
    - unspecified datasets in dict form default to all_faultnames
    - repeated use of the same fault across different datasets is allowed
        
    Examples:
    ---------
    >>> parse_data_faults(None, ["f1", "f2"], ["d1", "d2"])
    [["f1", "f2"], ["f1", "f2"]]
    
    >>> parse_data_faults(["f1", "f2"], ["f1", "f2"], ["d1", "d2"])
    [["f1"], ["f2"]]

    >>> parse_data_faults([None, ["f1"]], ["f1", "f2"], ["d1", "d2"])
    [["f1", "f2"], ["f1"]]

    >>> parse_data_faults([["f1", "f2"], "f1"], ["f1", "f2"], ["d1", "d2"])
    [["f1", "f2"], ["f1"]]
    
    >>> parse_data_faults({"d1": "f1", "d2": None}, ["f1", "f2"], ["d1", "d2"])
    [["f1"], ["f1", "f2"]]

    >>> parse_data_faults({"d1": ["f1"]}, ["f1", "f2"], ["d1", "d2"])
    [["f1"], ["f1", "f2"]]

    Common invalid cases:
    - ["f1"] with all_datanames=["d1", "d2"]
      Invalid because list input must provide one item per dataset.
    - {"d3": "f1"} with all_datanames=["d1", "d2"]
      Invalid because "d3" is not a known dataset name.
    - {"d1": "f3"} with all_faultnames=["f1", "f2"]
      Invalid because "f3" is not a known fault name.
    """
    
    def _normalize_fault_item(item, all_faultnames, param_name):
        """Normalize a single fault item to a list of fault names"""
        if item is None:
            return all_faultnames.copy()
        elif isinstance(item, str):
            if item not in all_faultnames:
                msg = f"Fault name '{item}' in {param_name} not found in all_faultnames"
                logger.error(msg)
                raise ValueError(msg)
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
                msg = f"Invalid fault names in {param_name}: {invalid_names}"
                logger.error(msg)
                raise ValueError(msg)
        else:
            msg = f"Invalid fault specification in {param_name}: {item}"
            logger.error(msg)
            raise ValueError(msg)
    
    # Case 1: None - expand to all_faultnames for all datasets
    if data_faults_config is None:
        return [all_faultnames.copy() for _ in all_datanames]
    
    # Case 2: List format
    elif isinstance(data_faults_config, list):
        if len(data_faults_config) != len(all_datanames):
            msg = f"Length of {param_name} ({len(data_faults_config)}) must equal number of datasets ({len(all_datanames)})"
            logger.error(msg)
            raise ValueError(msg)
        
        result = []
        for i, item in enumerate(data_faults_config):
            try:
                normalized = _normalize_fault_item(item, all_faultnames, f"{param_name}[{i}]")
                result.append(normalized)
            except ValueError as e:
                msg = f"Error in {param_name}[{i}] for dataset '{all_datanames[i]}': {str(e)}"
                logger.error(msg)
                raise ValueError(msg)
        
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
                    msg = f"Error in {param_name}['{dataname}']: {str(e)}"
                    logger.error(msg)
                    raise ValueError(msg)
            else:
                # Default to all_faultnames for unspecified datasets
                result.append(all_faultnames.copy())
        
        # Check for invalid dataset names in config
        invalid_datasets = set(data_faults_config.keys()) - set(all_datanames)
        if invalid_datasets:
            msg = f"Invalid dataset names in {param_name}: {invalid_datasets}"
            logger.error(msg)
            raise ValueError(msg)
        
        return result
    
    else:
        msg = f"{param_name} must be None, a list, or a dictionary"
        logger.error(msg)
        raise ValueError(msg)


def parse_alpha_faults(alpha_faults_config, all_faultnames, param_name="alphaFaults",
                       smoothing_faultnames=None):
    """
    Parse alphaFaults configuration with enhanced flexibility.
    
    Parameters:
    -----------
    alpha_faults_config : None or list
        Configuration for alpha faults. Supported forms:
        - None or [None]: all faults share one alpha group
        - list[str]: each fault name defines one alpha group
        - list[list[str]]: each sublist defines one alpha group
        - mixed list[str | list[str]]: strings and grouped fault lists can be mixed
    all_faultnames : list
        List of all available fault names
    param_name : str, optional
        Name of the parameter for error messages
    smoothing_faultnames : list, optional
        Subset of all_faultnames that support Laplacian smoothing.
        When provided, only these names participate in alpha grouping;
        non-smoothing sources are silently excluded from coverage
        validation. When None (default), all faults are required.
        
    Returns:
    --------
    list
        List of fault name lists for each alpha

    Raises:
    -------
    ValueError
        Raised when:
        - alpha_faults_config is not None or a list
        - a fault name does not exist in all_faultnames
        - the same fault appears in multiple alpha groups
        - some faults are missing from the final grouping
        - the number of configured groups exceeds the number of faults

    Notes:
    ------
    A list-style configuration is considered valid only if all of the
    following conditions are satisfied:
    - every referenced fault name exists in all_faultnames
    - each fault appears exactly once across all groups
    - all faults in all_faultnames are fully covered
        
    Examples:
    ---------
    >>> parse_alpha_faults(None, ["f1", "f2"])
    [["f1", "f2"]]
    
    >>> parse_alpha_faults([None], ["f1", "f2"])
    [["f1", "f2"]]

    >>> parse_alpha_faults(["f1", "f2"], ["f1", "f2"])
    [["f1"], ["f2"]]

    >>> parse_alpha_faults([["f1"], ["f2"]], ["f1", "f2"])
    [["f1"], ["f2"]]

    >>> parse_alpha_faults([["f1", "f2"]], ["f1", "f2"])
    [["f1", "f2"]]

    >>> parse_alpha_faults([["f1", "f2"], "f3"], ["f1", "f2", "f3"])
    [["f1", "f2"], ["f3"]]

    >>> parse_alpha_faults(["f1", ["f2", "f3"]], ["f1", "f2", "f3"])
    [["f1"], ["f2", "f3"]]

    Common invalid cases:
    - ["f1"] with all_faultnames=["f1", "f2"]
      Invalid because "f2" is missing and coverage is incomplete.
    - [["f1", "f2"], "f2"] with all_faultnames=["f1", "f2"]
      Invalid because "f2" appears in multiple groups.
    - ["f3"] with all_faultnames=["f1", "f2"]
      Invalid because "f3" is not a known fault name.
    """
    
    # When smoothing_faultnames is provided, only those names participate
    # in alpha grouping validation.  Non-smoothing sources are excluded.
    if smoothing_faultnames is not None:
        all_faultnames = [fn for fn in all_faultnames if fn in smoothing_faultnames]

    def _normalize_fault_subset(item, all_faultnames, param_name):
        """Normalize a fault subset item"""
        if isinstance(item, str):
            if item not in all_faultnames:
                msg = f"Fault name '{item}' in {param_name} not found in all_faultnames"
                logger.error(msg)
                raise ValueError(msg)
            return [item]
        elif isinstance(item, list):
            if not set(item).issubset(set(all_faultnames)):
                invalid_names = set(item) - set(all_faultnames)
                msg = f"Invalid fault names in {param_name}: {invalid_names}"
                logger.error(msg)
                raise ValueError(msg)
            return item.copy()
        else:
            msg = f"Invalid fault specification in {param_name}: {item}"
            logger.error(msg)
            raise ValueError(msg)
    
    # Case 1: None or [None] - single alpha case
    if alpha_faults_config is None or (isinstance(alpha_faults_config, list) and 
                                      len(alpha_faults_config) == 1 and 
                                      alpha_faults_config[0] is None):
        return [all_faultnames.copy()]
    
    # Case 2: List format
    elif isinstance(alpha_faults_config, list):
        if len(alpha_faults_config) > len(all_faultnames):
            msg = f"Length of {param_name} ({len(alpha_faults_config)}) must be less than or equal to " \
                  f"number of all faults ({len(all_faultnames)})"
            logger.error(msg)
            raise ValueError(msg)
        
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
                    msg = f"Fault names {overlap} appear in multiple alpha groups"
                    logger.error(msg)
                    raise ValueError(msg)
                all_assigned_faults.update(item_set)
                
            except ValueError as e:
                msg = f"Error in {param_name}[{i}]: {str(e)}"
                logger.error(msg)
                raise ValueError(msg)
        
        # Check for complete coverage
        if all_assigned_faults != set(all_faultnames):
            missing = set(all_faultnames) - all_assigned_faults
            msg = f"{param_name} does not cover all fault names. Missing: {missing}"
            logger.error(msg)
            raise ValueError(msg)
        
        return result
    
    else:
        msg = f"{param_name} must be None, or a list"
        logger.error(msg)
        raise ValueError(msg)

def parse_sigmas_config(sigmas_config, dataset_names, param_name='initial_value'):
    """
    Parse sigma configuration for VCE in `single`, `individual`, or
    `grouped` mode.

    Parameters:
    -----------
    sigmas_config : dict
        Sigma configuration dictionary. Supported keys depend on mode:
        - mode: one of 'single', 'individual', 'grouped'
        - update: bool or per-parameter list
        - param_name: scalar, list, or dict depending on mode
        - log_scaled: bool
        - groups: required in grouped mode
    dataset_names : list
        Dataset names. Must be provided and cannot be empty.
    param_name : str, optional
        Name of the value field to parse from sigmas_config, such as
        'initial_value' or 'values'. Default is 'initial_value'.

    Returns:
    --------
    dict
        Dictionary containing the merged input configuration and parsed
        arrays, including:
        - mode: str
        - update: np.ndarray of bool
        - param_name: np.ndarray of float under the same key name passed in
          param_name
        - dataset_param_indices: np.ndarray of int
        - updatable_param_indices: np.ndarray of int
        - log_scaled: bool
        - num_datasets: int
        - total_params: int
        - updatable_params: int
        - groups: dict or None

    Raises:
    -------
    ValueError
        Raised when:
        - dataset_names is empty
        - mode is not 'single', 'individual', or 'grouped'
        - single mode receives multi-value update or value lists
        - single mode receives dict values
        - individual mode list lengths do not match len(dataset_names)
        - grouped mode does not define groups
        - a dataset is assigned to multiple groups
        - some datasets are missing from grouped assignments
        - grouped mode list lengths do not match the number of groups

    Notes:
    ------
    Default values are applied before parsing:
    - update -> True
    - param_name -> 0.0
    - log_scaled -> False

    Mode-specific input rules:
    - single:
      update and param_name must be scalar-like, or a one-element list
    - individual:
      update may be scalar or dataset-length list
      param_name may be scalar, dataset-length list, or dict keyed by
      dataset name; missing dict entries default to 0.0
    - grouped:
      groups must be a dict mapping group name to dataset-name list
      every dataset in dataset_names must appear in exactly one group
      param_name may be scalar, group-length list, or dict keyed by
      group name; missing dict entries default to 0.0

    Examples:
    ---------
    >>> cfg = {"mode": "single", "update": True, "initial_value": 0.5}
    >>> out = parse_sigmas_config(cfg, ["d1", "d2"])
    >>> out["dataset_param_indices"].tolist()
    [0, 0]
    >>> out["initial_value"].tolist()
    [0.5]

    >>> cfg = {
    ...     "mode": "individual",
    ...     "update": [True, False, True],
    ...     "initial_value": {"d1": 0.3, "d3": 0.8},
    ... }
    >>> out = parse_sigmas_config(cfg, ["d1", "d2", "d3"])
    >>> out["initial_value"].tolist()
    [0.3, 0.0, 0.8]

    >>> cfg = {
    ...     "mode": "grouped",
    ...     "groups": {"insar": ["d1", "d2"], "gps": ["d3"]},
    ...     "update": [True, False],
    ...     "initial_value": {"insar": 0.2, "gps": 0.6},
    ... }
    >>> out = parse_sigmas_config(cfg, ["d1", "d2", "d3"])
    >>> out["dataset_param_indices"].tolist()
    [0, 0, 1]
    >>> out["initial_value"].tolist()
    [0.2, 0.6]

    >>> cfg = {"mode": "single", "values": [0.5], "update": [True]}
    >>> out = parse_sigmas_config(cfg, ["d1", "d2"], param_name="values")
    >>> out["values"].tolist()
    [0.5]

    Common invalid cases:
    - {"mode": "single", "update": [True, False]}
      Invalid because single mode accepts only one update flag.
    - {"mode": "individual", "initial_value": [0.1, 0.2]}
      Invalid when len(dataset_names) is not 2.
    - {"mode": "grouped", "groups": {"g1": ["d1"], "g2": ["d1", "d2"]}}
      Invalid because a dataset cannot belong to multiple groups.
    - {"mode": "grouped", "groups": {"g1": ["d1"]}}
      Invalid when some datasets are not assigned to any group.
    """
    
    # Validate dataset_names must be provided
    if not dataset_names:
        msg = "dataset_names must be provided and cannot be empty"
        logger.error(msg)
        raise ValueError(msg)
    
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
                msg = f"In single mode, update cannot be a multi-value list, current length: {len(update)}"
                logger.error(msg)
                raise ValueError(msg)
        else:
            update_list = [update]
            
        if isinstance(initial_value, (list, tuple)):
            if len(initial_value) == 1:
                initial_value_list = [float(initial_value[0])]
            else:
                msg = f"In single mode, {param_name} cannot be a multi-value list, current length: {len(initial_value)}"
                logger.error(msg)
                raise ValueError(msg)
        elif isinstance(initial_value, dict):
            msg = f"In single mode, {param_name} cannot be a dictionary"
            logger.error(msg)
            raise ValueError(msg)
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
                msg = f"In individual mode, update list length ({len(update)}) must equal number of datasets ({num_datasets})"
                logger.error(msg)
                raise ValueError(msg)
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
                msg = f"In individual mode, {param_name} list length ({len(initial_value)}) must equal number of datasets ({num_datasets})"
                logger.error(msg)
                raise ValueError(msg)
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
            msg = "In grouped mode, groups parameter must be provided and cannot be empty"
            logger.error(msg)
            raise ValueError(msg)
        
        # Build mapping from dataset names to parameter groups
        dataset_to_group = {}
        group_names = list(groups.keys())
        
        for group_idx, (group_name, datasets) in enumerate(groups.items()):
            for dataset_name in datasets:
                if dataset_name in dataset_to_group:
                    msg = f"Dataset '{dataset_name}' is assigned to multiple groups"
                    logger.error(msg)
                    raise ValueError(msg)
                dataset_to_group[dataset_name] = group_idx
        
        # Verify all datasets are grouped
        missing_datasets = set(dataset_names) - set(dataset_to_group.keys())
        if missing_datasets:
            msg = f"The following datasets are not assigned to any group: {missing_datasets}"
            logger.error(msg)
            raise ValueError(msg)
        
        # Build parameter mapping: [0, 0, 1, 1, 2, 2] style
        dataset_param_indices = [dataset_to_group[name] for name in dataset_names]
        
        # Handle update parameters
        num_groups = len(group_names)
        
        if isinstance(update_config, (list, tuple, np.ndarray)):
            if len(update_config) != num_groups:
                msg = f"In grouped mode, update list length ({len(update_config)}) must equal number of groups ({num_groups})"
                logger.error(msg)
                raise ValueError(msg)
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
                msg = f"In grouped mode, {param_name} list length ({len(initial_value_config)}) must equal number of groups ({num_groups})"
                logger.error(msg)
                raise ValueError(msg)
            initial_value_list = [float(val) for val in initial_value_config]
        else:
            # Expand single value to number of groups
            initial_value_list = [float(initial_value_config)] * num_groups
            
        total_params = num_groups
    
    else:
        msg = f"Unsupported mode: {mode}. Supported modes are 'single', 'individual', and 'grouped'."
        logger.error(msg)
        raise ValueError(msg)
    
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

def parse_alpha_config(alpha_config, faultnames, param_name='initial_value',
                       smoothing_faultnames=None):
    """
    Parse alpha (smoothing / regularization) configuration in `single`,
    `individual`, or `grouped` mode.

    Parameters:
    -----------
    alpha_config : dict
        Alpha configuration dictionary. Supported keys depend on mode:
        - mode: one of 'single', 'individual', 'grouped'
        - update: bool or per-parameter list
        - param_name: scalar, list, or dict depending on mode
        - log_scaled: bool
        - enabled: bool
        - faults: grouped mode definition as list[list[str]]
        - groups: legacy grouped mode definition as dict[str, list[str]]
    faultnames : list
        Fault names. Must be provided and cannot be empty.
    param_name : str, optional
        Input key to read alpha values from. Default is 'initial_value'.
    smoothing_faultnames : list, optional
        Subset of faultnames that support Laplacian smoothing.
        When provided, only these names participate in alpha grouping;
        non-smoothing sources are silently excluded. When None, all
        faultnames are used.

    Returns:
    --------
    dict
        Dictionary containing the merged input configuration and parsed
        outputs, including:
        - enabled: bool
        - mode: str
        - update: np.ndarray of bool
        - initial_value: np.ndarray of float
        - log_scaled: bool
        - faults: list[list[str]]
        - fault_param_indices: list of int (group index per smoothing fault)
        - updatable_param_indices: np.ndarray of int
        - num_alpha_faults: int
        - total_params: int
        - updatable_params: int

    Raises:
    -------
    ValueError
        Raised when:
        - faultnames is empty
        - mode is not 'single', 'individual', or 'grouped'
        - single mode receives more than one update flag
        - single mode receives more than one value
        - individual mode list lengths do not match len(faultnames)
        - grouped mode defines neither faults nor groups
        - grouped mode faults is not a list of lists
        - a fault is assigned to multiple groups
        - some faults in faultnames are missing from grouped assignments
        - grouped mode list lengths do not match the number of groups

    Notes:
    ------
    Default values are applied before parsing:
    - mode -> 'single'
    - log_scaled -> True
    - enabled -> True
    - update -> True
    - param_name -> 0.0

    Mode-specific input rules:
    - single:
      all faults share one alpha parameter
      update must be scalar-like or length-1
      param_name must be scalar-like or length-1
    - individual:
      each fault gets its own alpha parameter
      update may be scalar or fault-length list
      param_name may be scalar, fault-length list, or dict keyed by
      fault name; missing dict entries default to 0.0
    - grouped:
      prefer faults as list[list[str]]
      groups is accepted as a legacy dict and converted by dict order
      every fault in faultnames must appear exactly once across groups
      param_name may be scalar, group-length list, or dict keyed by
      stringified group indices such as "0", "1", ...; missing keys
      default to 0.0

    Examples:
    ---------
    >>> cfg = {"mode": "single", "update": True, "initial_value": 0.1}
    >>> out = parse_alpha_config(cfg, ["f1", "f2", "f3"])
    >>> out["fault_param_indices"]
    [0, 0, 0]
    >>> out["initial_value"].tolist()
    [0.1]

    >>> cfg = {
    ...     "mode": "individual",
    ...     "update": [True, False, True],
    ...     "initial_value": {"f1": 0.1, "f3": 0.3},
    ... }
    >>> out = parse_alpha_config(cfg, ["f1", "f2", "f3"])
    >>> out["initial_value"].tolist()
    [0.1, 0.0, 0.3]

    >>> cfg = {
    ...     "mode": "grouped",
    ...     "faults": [["f1", "f2"], ["f3"]],
    ...     "update": [True, False],
    ...     "initial_value": [0.1, 0.2],
    ... }
    >>> out = parse_alpha_config(cfg, ["f1", "f2", "f3"])
    >>> out["fault_param_indices"]
    [0, 0, 1]

    >>> cfg = {
    ...     "mode": "grouped",
    ...     "groups": {"g1": ["f1", "f2"], "g2": ["f3"]},
    ...     "initial_value": {"0": 0.1, "1": 0.2},
    ... }
    >>> out = parse_alpha_config(cfg, ["f1", "f2", "f3"])
    >>> out["initial_value"].tolist()
    [0.1, 0.2]

    Common invalid cases:
    - {"mode": "single", "update": [True, False]}
      Invalid because single mode accepts only one update flag.
    - {"mode": "individual", "initial_value": [0.1, 0.2]}
      Invalid when len(faultnames) is not 2.
    - {"mode": "grouped", "faults": [["f1", "f2"], ["f2"]]}
      Invalid because a fault cannot belong to multiple groups.
    - {"mode": "grouped", "faults": [["f1"]]}
      Invalid when some faults are not assigned to any group.
    """

    if not faultnames:
        msg = "faultnames must be provided and cannot be empty"
        logger.error(msg)
        raise ValueError(msg)

    # Filter to smoothing-capable sources when specified
    # Alpha grouping is built on smoothing faultnames only;
    # non-smoothing faults are auto-assigned to group 0 in fault_param_indices.
    all_faultnames = list(faultnames)   # preserve original full list
    if smoothing_faultnames is not None:
        faultnames = [fn for fn in faultnames if fn in smoothing_faultnames]
        if not faultnames:
            # No smoothing sources at all 鈥?return a minimal disabled config
            result = {
                'enabled': False,
                'update': np.array([False], dtype=bool),
                'initial_value': np.array([0.0], dtype=float),
                'log_scaled': alpha_config.get('log_scaled', True),
                'faults': [],
                'mode': 'single',
                'fault_param_indices': [0] * len(all_faultnames),
                'updatable_param_indices': np.array([-1], dtype=int),
                'num_alpha_faults': 0,
                'total_params': 1,
                'updatable_params': 0,
            }
            return {**alpha_config, **result}

    num_alpha_faults = len(faultnames)
    mode = alpha_config.get('mode', 'single')
    log_scaled = alpha_config.get('log_scaled', True)
    enabled = alpha_config.get('enabled', True)

    # --- Single mode: all faults share one parameter ---
    if mode == 'single':
        update = alpha_config.get('update', True)
        initial_value = alpha_config.get(param_name, 0.0)
        update_list = [update] if not isinstance(update, (list, tuple, np.ndarray)) else list(update)
        if len(update_list) != 1:
            msg = "In 'single' mode, 'update' must be a single value"
            logger.error(msg)
            raise ValueError(msg)
        if isinstance(initial_value, (list, tuple, np.ndarray)):
            if len(initial_value) != 1:
                msg = "In 'single' mode, 'initial_value' must be a single value"
                logger.error(msg)
                raise ValueError(msg)
            initial_value_list = [float(initial_value[0])]
        else:
            initial_value_list = [float(initial_value)]
        fault_param_indices = [0] * num_alpha_faults
        total_params = 1
        group_faults = [faultnames.copy()]

    # --- Individual mode: each fault has its own parameter ---
    elif mode == 'individual':
        update = alpha_config.get('update', True)
        initial_value = alpha_config.get(param_name, 0.0)
        if isinstance(update, (list, tuple, np.ndarray)):
            if len(update) != num_alpha_faults:
                msg = "In 'individual' mode, 'update' length must match number of faults"
                logger.error(msg)
                raise ValueError(msg)
            update_list = list(update)
        else:
            update_list = [update] * num_alpha_faults
        if isinstance(initial_value, dict):
            initial_value_list = [float(initial_value.get(name, 0.0)) for name in faultnames]
        elif isinstance(initial_value, (list, tuple, np.ndarray)):
            if len(initial_value) != num_alpha_faults:
                msg = "In 'individual' mode, 'initial_value' length must match number of faults"
                logger.error(msg)
                raise ValueError(msg)
            initial_value_list = [float(v) for v in initial_value]
        else:
            initial_value_list = [float(initial_value)] * num_alpha_faults
        fault_param_indices = list(range(num_alpha_faults))
        total_params = num_alpha_faults
        group_faults = [[f] for f in faultnames]

    # --- Grouped mode: custom grouping ---
    elif mode == 'grouped':
        # Prefer 'faults' (list of lists), fallback to 'groups' (dict)
        group_faults = alpha_config.get('faults', None)
        if group_faults is None:
            groups = alpha_config.get('groups', None)
            if groups is None:
                msg = "In 'grouped' mode, 'faults' (list of lists) or 'groups' (dict) must be provided"
                logger.error(msg)
                raise ValueError(msg)
            # Convert groups dict to list of lists
            group_faults = [groups[k] for k in groups]
        if not isinstance(group_faults, list) or not all(isinstance(g, list) for g in group_faults):
            msg = "'faults' must be a list of lists of fault names"
            logger.error(msg)
            raise ValueError(msg)
        num_groups = len(group_faults)
        # Build fault to group index mapping
        fault_to_group = {}
        for idx, flist in enumerate(group_faults):
            for f in flist:
                if f in fault_to_group:
                    msg = f"Fault '{f}' assigned to multiple groups"
                    logger.error(msg)
                    raise ValueError(msg)
                fault_to_group[f] = idx
        missing = set(faultnames) - set(fault_to_group.keys())
        if missing:
            msg = f"Faults not assigned to any group: {missing}"
            logger.error(msg)
            raise ValueError(msg)
        fault_param_indices = [fault_to_group[name] for name in faultnames]
        update = alpha_config.get('update', True)
        if isinstance(update, (list, tuple, np.ndarray)):
            if len(update) != num_groups:
                msg = "In 'grouped' mode, 'update' length must match number of groups"
                logger.error(msg)
                raise ValueError(msg)
            update_list = list(update)
        else:
            update_list = [update] * num_groups
        initial_value = alpha_config.get(param_name, 0.0)
        if isinstance(initial_value, dict):
            initial_value_list = [float(initial_value.get(str(i), 0.0)) for i in range(num_groups)]
        elif isinstance(initial_value, (list, tuple, np.ndarray)):
            if len(initial_value) != num_groups:
                msg = "In 'grouped' mode, 'initial_value' length must match number of groups"
                logger.error(msg)
                raise ValueError(msg)
            initial_value_list = [float(v) for v in initial_value]
        else:
            initial_value_list = [float(initial_value)] * num_groups
        total_params = num_groups

    else:
        msg = f"Unknown alpha mode: {mode}. Supported modes are 'single', 'individual', and 'grouped'."
        logger.error(msg)
        raise ValueError(msg)

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

    # If smoothing filtering was applied, expand fault_param_indices back to all faults.
    # Non-smoothing faults are auto-assigned to group 0.
    if smoothing_faultnames is not None and len(all_faultnames) != len(faultnames):
        smoothing_index_map = {name: idx for name, idx in zip(faultnames, fault_param_indices)}
        fault_param_indices = [smoothing_index_map.get(fn, 0) for fn in all_faultnames]

    result = {
        'enabled': enabled,
        'update': np.array(update_list, dtype=bool),
        'initial_value': np.array(initial_value_list, dtype=float),
        'log_scaled': log_scaled,
        'faults': group_faults,
        'mode': mode,
        'fault_param_indices': fault_param_indices,
        'updatable_param_indices': np.array(updatable_param_indices, dtype=int),
        'num_alpha_faults': num_alpha_faults,
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
            msg = f"No bounds specified for {param_type} '{name}' and no defaults provided"
            logger.error(msg)
            raise ValueError(msg)
    
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
        msg = f"Length of {param_name} ({len(config_list)}) must equal expected length ({expected_length})"
        logger.error(msg)
        raise ValueError(msg)


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
            msg = "Euler pole units must have 3 elements: [longitude, latitude, angular_velocity]"
            logger.error(msg)
            raise ValueError(msg)
        
        lon_factor = angle_conversions.get(units_config[0])
        lat_factor = angle_conversions.get(units_config[1])
        omega_factor = angular_velocity_conversions.get(units_config[2])
        
        if lon_factor is None:
            msg = f"Invalid longitude unit: {units_config[0]}"
            logger.error(msg)
            raise ValueError(msg)
        if lat_factor is None:
            msg = f"Invalid latitude unit: {units_config[1]}"
            logger.error(msg)
            raise ValueError(msg)
        if omega_factor is None:
            msg = f"Invalid angular velocity unit: {units_config[2]}"
            logger.error(msg)
            raise ValueError(msg)
            
        return {
            'units': units_config,
            'conversion_factors': [lon_factor, lat_factor, omega_factor],
            'standard_units': ['radians', 'radians', 'radians_per_year']
        }
    
    elif unit_type == 'euler_vector':
        if len(units_config) != 3:
            msg = "Euler vector units must have 3 elements: [wx, wy, wz]"
            logger.error(msg)
            raise ValueError(msg)
        
        conversion_factors = []
        for unit in units_config:
            factor = angular_velocity_conversions.get(unit)
            if factor is None:
                msg = f"Invalid angular velocity unit: {unit}"
                logger.error(msg)
                raise ValueError(msg)
            conversion_factors.append(factor)
            
        return {
            'units': units_config,
            'conversion_factors': conversion_factors,
            'standard_units': ['radians_per_year'] * 3
        }
    
    else:
        msg = f"Invalid unit_type: {unit_type}"
        logger.error(msg)
        raise ValueError(msg)


def euler_pole_to_cartesian(lon_rad, lat_rad, omega_rad_per_year):
    """Convert a physical Euler pole in radians to a Cartesian vector.

    Parameters
    ----------
    lon_rad, lat_rad : float
        Euler pole longitude and latitude in radians.
    omega_rad_per_year : float
        Angular velocity in radians/year.

    Returns
    -------
    numpy.ndarray
        Cartesian Euler vector ``[wx, wy, wz]`` in radians/year.
    """
    return np.array([
        omega_rad_per_year * np.cos(lat_rad) * np.cos(lon_rad),
        omega_rad_per_year * np.cos(lat_rad) * np.sin(lon_rad),
        omega_rad_per_year * np.sin(lat_rad),
    ], dtype=float)


def standardize_euler_pole(value, units):
    """Return physical pole and Cartesian vector from user Euler-pole input.

    The public order is always ``[lon, lat, omega]``.  The returned pole is
    ``[lon_rad, lat_rad, omega_rad_per_year]`` and the vector is Cartesian
    ``[wx, wy, wz]`` in radians/year.
    """
    factors = parse_euler_units(units, "euler_pole")["conversion_factors"]
    pole = np.asarray([float(value[i]) * factors[i] for i in range(3)], dtype=float)
    return pole, euler_pole_to_cartesian(pole[0], pole[1], pole[2])


def standardize_euler_vector(value, units):
    """Return Cartesian Euler vector in radians/year from user input."""
    factors = parse_euler_units(units, "euler_vector")["conversion_factors"]
    return np.asarray([float(value[i]) * factors[i] for i in range(3)], dtype=float)
# --------------------------------------------------------------------------------------#
