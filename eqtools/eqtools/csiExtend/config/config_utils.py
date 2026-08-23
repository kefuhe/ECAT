"""
Configuration parsing utilities for bayesian_config.py and related modules.
"""
import numpy as np
import logging 

from eqtools.csiExtend.config.parameter_groups import (
    attach_group_parameters,
    resolve_group_layout,
)

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
    if isinstance(initial_value, (int, float, np.number)):
        return [float(initial_value)] * n_datasets
    
    # Handle ordered sequences consistently across Python and NumPy callers.
    elif isinstance(initial_value, (list, tuple, np.ndarray)):
        initial_value = np.asarray(initial_value)
        if initial_value.ndim != 1:
            msg = f"'{print_name}' must be a one-dimensional sequence"
            logger.error(msg)
            raise ValueError(msg)
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
            if not isinstance(val, (int, float, np.number)):
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
    """Normalize sigma values in member, group, and sampled parameter space.

    ``single`` creates one group, ``individual`` creates one group per data
    set, and ``grouped`` consumes the named ``groups`` mapping.  A scalar value
    or update flag broadcasts.  Sequence and dictionary inputs must cover the
    resolved groups exactly; missing or unknown keys fail before inversion.

    The returned legacy fields remain available.  ``group_layout`` is the
    canonical internal contract used by BLSE, VCE, and Bayesian consumers.
    """

    if not dataset_names:
        raise ValueError("dataset_names must be provided and cannot be empty")
    config = {
        "update": True,
        param_name: 0.0,
        "log_scaled": False,
        **({} if sigmas_config is None else sigmas_config),
    }
    mode = str(config.get("mode", "individual")).lower()
    raw_groups = config.get("groups") if mode == "grouped" else None
    layout = resolve_group_layout(
        dataset_names,
        mode,
        raw_groups,
        member_label="dataset",
        single_group_name="all",
        individual_prefix="group_",
    )

    raw_values = config.get(param_name, 0.0)
    if mode == "single" and isinstance(raw_values, dict):
        raise ValueError(f"In single mode, {param_name} cannot be a dictionary")
    aliases = None
    if mode == "individual":
        aliases = {
            group_name: member
            for group_name, (member,) in layout["members_by_group"].items()
        }
    contract = attach_group_parameters(
        layout,
        values=raw_values,
        update=config.get("update", True),
        value_name=param_name,
        default_value=0.0,
        value_key_aliases=aliases,
    )
    result = {
        "mode": mode,
        "update": contract["update_by_group"],
        param_name: contract["values_by_group"],
        "dataset_param_indices": contract["member_param_indices"],
        "updatable_param_indices": contract["sample_index_by_group"],
        "log_scaled": bool(config["log_scaled"]),
        "num_datasets": len(dataset_names),
        "total_params": contract["total_params"],
        "updatable_params": contract["updatable_params"],
        "groups": raw_groups,
        "group_layout": contract,
    }
    return {**config, **result}

def parse_alpha_config(alpha_config, faultnames, param_name='initial_value',
                       smoothing_faultnames=None):
    """Normalize alpha groups over smoothing-capable sources only.

    ``group_layout`` is canonical and never assigns a non-smoothing source to
    a real alpha group.  The historical full-source ``fault_param_indices`` is
    retained as a compatibility projection for existing consumers; new code
    must use ``group_layout`` when reasoning about cardinality.

    Scalar values broadcast.  List/array values and update flags must match the
    number of groups exactly.  Dictionary values must cover every group (or its
    documented alias) without unknown keys.
    """

    if not faultnames:
        raise ValueError("faultnames must be provided and cannot be empty")
    config = {} if alpha_config is None else dict(alpha_config)
    all_faultnames = list(faultnames)
    if len(all_faultnames) != len(set(all_faultnames)):
        raise ValueError("faultnames must be unique")

    if smoothing_faultnames is None:
        smoothing_names = all_faultnames.copy()
    else:
        smoothing_names = list(smoothing_faultnames)
        if len(smoothing_names) != len(set(smoothing_names)):
            raise ValueError("smoothing_faultnames contains duplicate names")
        unknown = [name for name in smoothing_names if name not in all_faultnames]
        if unknown:
            raise ValueError(
                "Unknown smoothing source name(s): " + ", ".join(unknown)
            )
        smoothing_names = [
            name for name in all_faultnames if name in set(smoothing_names)
        ]

    if not smoothing_names:
        empty_layout = {
            "mode": "single",
            "member_names": [],
            "group_names": [],
            "members_by_group": {},
            "member_to_group": {},
            "member_param_indices": np.array([], dtype=int),
            "total_params": 0,
            "values_by_group": np.array([], dtype=float),
            "update_by_group": np.array([], dtype=bool),
            "sample_index_by_group": np.array([], dtype=int),
            "updatable_params": 0,
        }
        result = {
            "enabled": False,
            "update": np.array([False], dtype=bool),
            param_name: np.array([0.0], dtype=float),
            "log_scaled": bool(config.get("log_scaled", True)),
            "faults": [],
            "mode": "single",
            "fault_param_indices": [0] * len(all_faultnames),
            "updatable_param_indices": np.array([-1], dtype=int),
            "num_alpha_faults": 0,
            "total_params": 1,
            "updatable_params": 0,
            "group_layout": empty_layout,
        }
        return {**config, **result}

    mode = str(config.get("mode", "single")).lower()
    raw_groups = None
    value_aliases = None
    if mode == "grouped":
        group_faults = config.get("faults")
        named_groups = config.get("groups")
        if group_faults is not None:
            if not isinstance(group_faults, list) or not all(
                isinstance(group, list) for group in group_faults
            ):
                raise ValueError("'faults' must be a list of fault-name lists")
            raw_groups = {
                f"Event_{index}": members
                for index, members in enumerate(group_faults)
            }
        elif named_groups is not None:
            raw_groups = named_groups
        else:
            raise ValueError(
                "In grouped mode, 'faults' or 'groups' must define the alpha groups"
            )

    layout = resolve_group_layout(
        smoothing_names,
        mode,
        raw_groups,
        member_label="smoothing source",
        single_group_name="all",
        individual_prefix="smooth_",
    )
    if mode == "individual":
        value_aliases = {
            group_name: member
            for group_name, (member,) in layout["members_by_group"].items()
        }
    elif mode == "grouped":
        # Stringified indices were accepted historically for alpha values.
        value_aliases = {
            group_name: str(index)
            for index, group_name in enumerate(layout["group_names"])
        }

    raw_values = config.get(param_name, 0.0)
    if mode == "single" and isinstance(raw_values, dict):
        raise ValueError(f"In single mode, {param_name} cannot be a dictionary")
    contract = attach_group_parameters(
        layout,
        values=raw_values,
        update=config.get("update", True),
        value_name=param_name,
        default_value=0.0,
        value_key_aliases=value_aliases,
    )

    smoothing_indices = {
        name: int(index)
        for name, index in zip(
            contract["member_names"], contract["member_param_indices"]
        )
    }
    # Compatibility boundary: old consumers expect one index per source.  A
    # non-smoothing source receives a neutral placeholder here only; it is not
    # part of the canonical group_layout and creates no parameter.
    fault_param_indices = [
        smoothing_indices.get(name, 0) for name in all_faultnames
    ]
    group_faults = [
        list(contract["members_by_group"][name])
        for name in contract["group_names"]
    ]
    result = {
        "enabled": bool(config.get("enabled", True)),
        "update": contract["update_by_group"],
        param_name: contract["values_by_group"],
        "log_scaled": bool(config.get("log_scaled", True)),
        "faults": group_faults,
        "mode": mode,
        "fault_param_indices": fault_param_indices,
        "updatable_param_indices": contract["sample_index_by_group"],
        "num_alpha_faults": len(smoothing_names),
        "total_params": contract["total_params"],
        "updatable_params": contract["updatable_params"],
        "group_layout": contract,
    }
    return {**config, **result}

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
