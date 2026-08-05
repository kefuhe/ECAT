"""Configuration parsing for interseismic block motion and constraints."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from .config_utils import (
    merge_with_defaults,
    parse_euler_units,
    standardize_euler_pole,
    standardize_euler_vector,
)


VALID_BLOCK_TYPES = {"dataset", "euler_pole", "euler_vector"}
VALID_BLOCK_EULER_MODES = {"estimate", "fixed_pole", "fixed_vector"}
BLOCK_EULER_MODE_ALIASES = {
    "estimate": "estimate",
    "estimated": "estimate",
    "dataset": "estimate",
    "fixed_pole": "fixed_pole",
    "euler_pole": "fixed_pole",
    "pole": "fixed_pole",
    "fixed_vector": "fixed_vector",
    "euler_vector": "fixed_vector",
    "vector": "fixed_vector",
}
BLOCK_EULER_MODE_TO_SOURCE = {
    "estimate": "dataset",
    "fixed_pole": "euler_pole",
    "fixed_vector": "euler_vector",
}
VALID_FAULT_LOADING_BLOCK_TYPES = VALID_BLOCK_TYPES | {"block"}
VALID_MOTION_SENSES = {"dextral", "sinistral", "right_lateral", "left_lateral", "right", "left"}
VALID_CAP_CONSTRAINT_MODES = {"motion_sense", "loading_sign"}
CAP_CONSTRAINT_MODE_ALIASES = {
    "motion_sense": "motion_sense",
    "motion": "motion_sense",
    "fault_motion": "motion_sense",
    "loading_sign": "loading_sign",
    "projected_loading": "loading_sign",
}
CAP_CONSTRAINT_TOP_LEVEL_KEYS = {
    "enabled", "defaults", "faults", "configured_faults",
}
CAP_CONSTRAINT_DEFAULT_KEYS = {
    "selector",
    "max_coupling",
    "factor",
    "mode",
    "hard_overlap",
    "min_loading_abs",
}
CAP_CONSTRAINT_FAULT_KEYS = CAP_CONSTRAINT_DEFAULT_KEYS | {"enabled"}
FAULT_LOADING_ONLY_KEYS = {
    "blocks",
    "block_types",
    "block_names",
    "loading_overrides",
    "loading_regions",
    "motion_sense_overrides",
    "motion_sense",
    "owner_source",
    "reference_strike",
    "units",
    "euler_pole_units",
    "euler_vector_units",
}
LOADING_OVERRIDE_KEYS = {
    "name",
    "id",
    "selector",
    "blocks",
    "block_types",
    "block_names",
    "reference_strike",
    "motion_sense",
    "owner_source",
    "units",
    # Normalized-output fields accepted for parser idempotence.
    "blocks_original",
    "blocks_standard",
}
MOTION_SENSE_OVERRIDE_KEYS = {"name", "id", "selector", "motion_sense"}
VALID_CAP_HARD_OVERLAP_POLICIES = {"skip", "keep", "error"}
CAP_HARD_OVERLAP_ALIASES = {
    "skip": "skip",
    "exclude": "skip",
    "auto": "skip",
    "keep": "keep",
    "allow": "keep",
    "error": "error",
    "raise": "error",
}

INTERSEISMIC_TOP_LEVEL_KEYS = {
    "version", "blocks", "fault_loading", "cap_constraints",
    "backslip_constraints", "outputs",
}
BLOCKS_CONTROL_KEYS = {"enabled", "defaults", "items", "configured_blocks"}
BLOCK_DEFAULT_KEYS = {
    "euler_mode", "euler_source", "euler_pole_units",
    "euler_vector_units", "owner_source",
}
BLOCK_ITEM_KEYS = {"name", "datasets", "owner_source", "euler"}
BLOCK_EULER_KEYS = {
    "mode", "source", "type", "value", "pole", "vector", "units",
    "datasets", "dataset", "anchor_dataset", "owner_source",
    # Normalized-output fields accepted so parsing is idempotent.
    "value_standard", "pole_standard", "vector_radians_per_year",
}
FAULT_LOADING_TOP_LEVEL_KEYS = {
    "enabled", "defaults", "faults", "configured_faults",
}
FAULT_LOADING_DEFAULT_KEYS = {
    "block_types", "euler_pole_units", "euler_vector_units",
    "reference_strike", "motion_sense", "owner_source",
}
FAULT_LOADING_FAULT_KEYS = {
    "blocks", "block_types", "block_names", "reference_strike",
    "motion_sense", "owner_source", "units", "loading_overrides",
    "loading_regions", "motion_sense_overrides",
    # Normalized-output fields accepted so script-side updates can reparse the
    # current configuration without first serializing it back to raw YAML.
    "blocks_standard", "blocks_original",
}
BACKSLIP_CONSTRAINT_KEYS = {
    "fault", "state", "selector", "component", "coupling", "value",
    "name",
}


def _validate_mapping_keys(raw_mapping, allowed_keys, context: str) -> None:
    """Reject unknown configuration keys with their full YAML context."""
    unknown = sorted(set(raw_mapping) - set(allowed_keys))
    if not unknown:
        return
    allowed = ", ".join(sorted(allowed_keys))
    unknown_text = ", ".join(unknown)
    raise ValueError(
        f"Unsupported key(s) in {context}: {unknown_text}. "
        f"Allowed keys: {allowed}"
    )


def empty_interseismic_config() -> Dict[str, Any]:
    """Return the disabled interseismic configuration structure."""
    return {
        "version": 1,
        "blocks": {
            "enabled": False,
            "defaults": {},
            "items": {},
            "configured_blocks": [],
        },
        "fault_loading": {
            "enabled": False,
            "defaults": {},
            "faults": {},
            "configured_faults": [],
        },
        "cap_constraints": {
            "enabled": False,
            "defaults": {},
            "faults": {},
            "configured_faults": [],
        },
        "backslip_constraints": [],
        "outputs": {},
    }


def load_interseismic_config_data(config: Any, encoding: str = "utf-8") -> Dict[str, Any]:
    """Load an interseismic configuration from a path or mapping."""
    if config is None:
        return empty_interseismic_config()
    if isinstance(config, Mapping):
        return deepcopy(dict(config))
    path = Path(config)
    with path.open("r", encoding=encoding) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Interseismic config '{path}' must contain a mapping at top level")
    return dict(data)


def parse_interseismic_config(
    config: Optional[Mapping[str, Any]],
    faultnames: Iterable[str],
    dataset_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Parse interseismic blocks, fault loading, and constraint configuration.

    Parameters
    ----------
    config : mapping
        Raw ``interseismic_config.yml`` content.
    faultnames : iterable of str
        Fault/source names in the current inversion problem.
    dataset_names : iterable of str, optional
        Available geodetic dataset names, used when a block is supplied by an
        estimated ``eulerrotation`` transform.

    Returns
    -------
    dict
        Normalized configuration with standardized fixed Euler parameters.

    Notes
    -----
    ``blocks`` and ``fault_loading`` define the physical loading rate ``b`` on
    each patch. ``cap_constraints`` only controls optional inequality rows.
    These roles are deliberately separate.
    """
    if not config:
        return empty_interseismic_config()
    if "euler_constraints" in config or "use_euler_constraints" in config:
        raise ValueError(
            "Old 'euler_constraints'/'use_euler_constraints' entries are no longer supported. "
            "Move block settings to interseismic_config.yml:blocks, fault-loading settings "
            "to fault_loading, and optional inequality caps to cap_constraints."
        )
    if "block_motion" in config:
        raise ValueError(
            "Old 'block_motion' entries are no longer supported. "
            "Rename the section to 'fault_loading'; block-motion loading is now "
            "defined only by interseismic_config.yml:fault_loading."
        )
    _validate_mapping_keys(
        config, INTERSEISMIC_TOP_LEVEL_KEYS, "interseismic_config"
    )

    faultnames = list(faultnames)
    dataset_names = set(dataset_names or [])
    raw = deepcopy(dict(config))
    blocks = _parse_blocks(raw.get("blocks", {}), dataset_names)
    fault_loading = _parse_fault_loading(raw.get("fault_loading", {}), faultnames, blocks)
    cap_constraints = _parse_cap_constraints(raw.get("cap_constraints", {}), fault_loading, faultnames)
    backslip_constraints = _parse_backslip_constraints(raw.get("backslip_constraints", []), faultnames)

    return {
        "version": int(raw.get("version", 1)),
        "blocks": blocks,
        "fault_loading": fault_loading,
        "cap_constraints": cap_constraints,
        "backslip_constraints": backslip_constraints,
        "outputs": deepcopy(raw.get("outputs", {})),
    }


def _parse_blocks(raw_blocks: Mapping[str, Any], dataset_names) -> Dict[str, Any]:
    raw_blocks = dict(raw_blocks or {})
    item_keys = BLOCKS_CONTROL_KEYS
    raw_items = raw_blocks.get("items")
    if raw_items is not None:
        _validate_mapping_keys(raw_blocks, BLOCKS_CONTROL_KEYS, "blocks")
    if raw_items is None:
        raw_items = {key: value for key, value in raw_blocks.items() if key not in item_keys}
    if raw_items is None:
        raw_items = {}
    if not isinstance(raw_items, Mapping):
        raise ValueError("blocks.items must be a mapping")

    default_defaults = {
        "euler_mode": "estimate",
        "euler_source": "dataset",
        "euler_pole_units": ["degrees", "degrees", "degrees_per_myr"],
        "euler_vector_units": ["radians_per_year", "radians_per_year", "radians_per_year"],
        "owner_source": None,
    }
    raw_defaults = dict(raw_blocks.get("defaults", {}) or {})
    _validate_mapping_keys(raw_defaults, BLOCK_DEFAULT_KEYS, "blocks.defaults")
    defaults = merge_with_defaults(raw_defaults, default_defaults)
    _validate_units(defaults["euler_pole_units"], "euler_pole")
    _validate_units(defaults["euler_vector_units"], "euler_vector")

    parsed_items = {}
    for block_name, block_config in raw_items.items():
        if not isinstance(block_config, Mapping):
            raise ValueError(f"blocks.{block_name} must be a mapping")
        parsed_items[str(block_name)] = _parse_one_block(block_name, block_config, defaults, dataset_names)

    enabled = bool(raw_blocks.get("enabled", bool(parsed_items)))
    return {
        "enabled": enabled,
        "defaults": defaults,
        "items": parsed_items if enabled else {},
        "configured_blocks": list(parsed_items.keys()) if enabled else [],
    }


def _parse_one_block(block_name, block_config, defaults, dataset_names) -> Dict[str, Any]:
    block_config = dict(block_config or {})
    euler_raw = dict(block_config.get("euler", block_config))
    _reject_block_share_flag(block_name, block_config, euler_raw)
    if "euler" in block_config:
        _validate_mapping_keys(
            block_config, BLOCK_ITEM_KEYS, f"blocks.{block_name}"
        )
        euler_mapping = dict(block_config.get("euler") or {})
        _validate_mapping_keys(
            euler_mapping, BLOCK_EULER_KEYS, f"blocks.{block_name}.euler"
        )
    else:
        _validate_mapping_keys(
            block_config,
            (BLOCK_ITEM_KEYS - {"euler"}) | BLOCK_EULER_KEYS,
            f"blocks.{block_name}",
        )
    raw_mode = euler_raw.get(
        "mode",
        euler_raw.get("source", euler_raw.get("type", defaults.get("euler_mode", defaults["euler_source"]))),
    )
    mode = _normalize_block_euler_mode(raw_mode, f"blocks.{block_name}.euler.mode")
    source = BLOCK_EULER_MODE_TO_SOURCE[mode]

    owner_source = euler_raw.get("owner_source", block_config.get("owner_source", defaults.get("owner_source")))
    datasets = _parse_block_datasets(block_name, block_config, euler_raw, mode, dataset_names)
    anchor_dataset = euler_raw.get("anchor_dataset")
    if anchor_dataset is None and datasets:
        anchor_dataset = datasets[0]
    if anchor_dataset is not None:
        anchor_dataset = str(anchor_dataset)
        if anchor_dataset not in datasets:
            raise ValueError(f"blocks.{block_name}.euler.anchor_dataset must be one of block datasets")

    parsed = {
        "name": str(block_name),
        "datasets": datasets,
        "owner_source": owner_source,
        "euler": {
            "mode": mode,
            "source": source,
            "owner_source": owner_source,
            "datasets": datasets,
            "anchor_dataset": anchor_dataset,
        },
    }

    if mode == "estimate":
        pass
    elif mode == "fixed_pole":
        value = euler_raw.get("value", euler_raw.get("pole"))
        _validate_block_value(str(block_name), 0, "euler_pole", value, dataset_names)
        units = euler_raw.get("units", defaults["euler_pole_units"])
        _validate_units(units, "euler_pole")
        pole_standard, vector_standard = standardize_euler_pole(value, units)
        parsed["euler"].update({
            "value": deepcopy(value),
            "value_standard": vector_standard.tolist(),
            "pole_standard": pole_standard.tolist(),
            "vector_radians_per_year": vector_standard.tolist(),
            "units": list(units),
        })
    elif mode == "fixed_vector":
        value = euler_raw.get("value", euler_raw.get("vector"))
        _validate_block_value(str(block_name), 0, "euler_vector", value, dataset_names)
        units = euler_raw.get("units", defaults["euler_vector_units"])
        _validate_units(units, "euler_vector")
        vector_standard = standardize_euler_vector(value, units)
        parsed["euler"].update({
            "value": deepcopy(value),
            "value_standard": vector_standard.tolist(),
            "vector_radians_per_year": vector_standard.tolist(),
            "units": list(units),
        })
    return parsed


def _reject_block_share_flag(block_name, block_config, euler_raw) -> None:
    for key in ("share", "share_euler"):
        if key in block_config:
            raise ValueError(
                f"blocks.{block_name}.{key} is no longer supported. "
                "Datasets listed in the same block always share one Euler vector; "
                "use separate blocks if they should not share Euler parameters."
            )
        if key in euler_raw:
            raise ValueError(
                f"blocks.{block_name}.euler.{key} is no longer supported. "
                "Datasets listed in the same block always share one Euler vector; "
                "use separate blocks if they should not share Euler parameters."
            )


def _normalize_block_euler_mode(value, context: str) -> str:
    key = str(value).lower()
    if key not in BLOCK_EULER_MODE_ALIASES:
        allowed = ", ".join(sorted(VALID_BLOCK_EULER_MODES))
        raise ValueError(f"{context} must be one of {allowed}")
    return BLOCK_EULER_MODE_ALIASES[key]


def _parse_block_datasets(block_name, block_config, euler_raw, mode, dataset_names) -> list[str]:
    datasets = block_config.get("datasets")
    if datasets is None:
        datasets = euler_raw.get("datasets")
    if datasets is None:
        datasets = euler_raw.get("dataset")
    if datasets is None and mode == "estimate":
        datasets = euler_raw.get("value")
    if datasets is None:
        datasets = []
    if isinstance(datasets, str):
        datasets = [datasets]
    if not isinstance(datasets, (list, tuple)):
        raise ValueError(f"blocks.{block_name}.datasets must be a list of dataset names")
    datasets = [str(name) for name in datasets]
    if mode == "estimate" and not datasets:
        raise ValueError(f"blocks.{block_name}.datasets must contain at least one dataset name")
    for dataset in datasets:
        if dataset_names and dataset not in dataset_names:
            raise ValueError(f"Dataset '{dataset}' for block '{block_name}' not found in geodata")
    return datasets


def _parse_fault_loading(raw_fault_loading: Mapping[str, Any], faultnames, blocks_config) -> Dict[str, Any]:
    raw_fault_loading = dict(raw_fault_loading or {})
    _validate_mapping_keys(
        raw_fault_loading, FAULT_LOADING_TOP_LEVEL_KEYS, "fault_loading"
    )
    enabled = bool(raw_fault_loading.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "defaults": {},
            "faults": {},
            "configured_faults": [],
        }

    default_defaults = {
        "block_types": ["block", "block"],
        "euler_pole_units": ["degrees", "degrees", "degrees_per_myr"],
        "euler_vector_units": ["radians_per_year", "radians_per_year", "radians_per_year"],
        "reference_strike": 0.0,
        "motion_sense": "dextral",
        "owner_source": None,
    }
    raw_defaults = dict(raw_fault_loading.get("defaults", {}) or {})
    _validate_mapping_keys(
        raw_defaults, FAULT_LOADING_DEFAULT_KEYS, "fault_loading.defaults"
    )
    defaults = merge_with_defaults(raw_defaults, default_defaults)
    _validate_units(defaults["euler_pole_units"], "euler_pole")
    _validate_units(defaults["euler_vector_units"], "euler_vector")
    defaults["reference_strike"] = _validate_reference_strike(defaults["reference_strike"], "fault_loading.defaults")
    defaults["motion_sense"] = _validate_motion_sense(defaults["motion_sense"], "fault_loading.defaults")

    faults_config = raw_fault_loading.get("faults", {})
    if not isinstance(faults_config, Mapping):
        raise ValueError("fault_loading.faults must be a mapping")
    unknown_faults = sorted(set(faults_config) - set(faultnames))
    if unknown_faults:
        raise ValueError(
            "fault_loading.faults references unknown fault(s): "
            + ", ".join(unknown_faults)
        )

    parsed_faults = {}
    for fault_name in faultnames:
        if fault_name not in faults_config:
            continue
        parsed_faults[fault_name] = _parse_one_fault_loading_fault(
            fault_name,
            dict(faults_config[fault_name] or {}),
            defaults,
            blocks_config,
        )

    return {
        "enabled": True,
        "defaults": defaults,
        "faults": parsed_faults,
        "configured_faults": list(parsed_faults.keys()),
    }


def _parse_one_fault_loading_fault(fault_name, fault_config, defaults, blocks_config) -> Dict[str, Any]:
    _validate_mapping_keys(
        fault_config,
        FAULT_LOADING_FAULT_KEYS,
        f"fault_loading.faults.{fault_name}",
    )
    if "blocks" not in fault_config:
        raise ValueError(f"fault_loading.faults.{fault_name} is missing required 'blocks'")
    blocks = fault_config["blocks"]
    block_types = fault_config.get("block_types", defaults["block_types"])
    if len(block_types) != 2:
        raise ValueError(f"fault_loading.faults.{fault_name}.block_types must contain exactly two entries")
    if len(blocks) != 2:
        raise ValueError(f"fault_loading.faults.{fault_name}.blocks must contain exactly two entries")
    for block_type in block_types:
        if block_type not in VALID_FAULT_LOADING_BLOCK_TYPES:
            raise ValueError(f"Invalid fault_loading block_type '{block_type}' for fault '{fault_name}'")
    known_blocks = blocks_config.get("items", {})
    for idx, (block_type, block) in enumerate(zip(block_types, blocks)):
        if block_type == "block" and block not in known_blocks:
            raise ValueError(f"fault_loading.faults.{fault_name} references undefined block '{block}'")
        if block_type in VALID_BLOCK_TYPES:
            _validate_block_value(fault_name, idx, block_type, block, set())

    merged = merge_with_defaults(fault_config, {
        "block_types": list(block_types),
        "reference_strike": defaults["reference_strike"],
        "motion_sense": defaults["motion_sense"],
        "owner_source": defaults.get("owner_source"),
        "units": {
            "euler_pole_units": defaults["euler_pole_units"],
            "euler_vector_units": defaults["euler_vector_units"],
        },
    })
    merged["reference_strike"] = _validate_reference_strike(
        merged["reference_strike"], f"fault_loading.faults.{fault_name}"
    )
    merged["motion_sense"] = _validate_motion_sense(
        merged["motion_sense"], f"fault_loading.faults.{fault_name}"
    )
    if "block_names" not in merged:
        merged["block_names"] = list(blocks)
    elif len(merged["block_names"]) != 2:
        raise ValueError(f"fault_loading.faults.{fault_name}.block_names must contain exactly two entries")
    merged["blocks_standard"] = _standardize_fault_loading_blocks(merged["block_types"], blocks, merged["units"])
    merged["blocks_original"] = deepcopy(list(blocks))
    raw_loading_overrides = _read_override_list(
        fault_config,
        primary_key="loading_overrides",
        legacy_key="loading_regions",
        context=f"fault_loading.faults.{fault_name}",
    )
    loading_overrides = _parse_loading_overrides(
        fault_name,
        raw_loading_overrides,
        merged,
        blocks_config,
    )
    merged["loading_overrides"] = loading_overrides
    # Compatibility alias for older internal callers and local scripts.
    merged["loading_regions"] = loading_overrides
    merged["motion_sense_overrides"] = _parse_motion_sense_overrides(
        fault_name,
        fault_config.get("motion_sense_overrides", []),
    )
    return merged


def _read_override_list(config, *, primary_key: str, legacy_key: str, context: str):
    has_primary = primary_key in config
    has_legacy = legacy_key in config
    if has_primary and has_legacy:
        # Parsed output intentionally carries the legacy alias for old internal
        # callers.  Accept the pair only when both names describe the same
        # declaration; differing values remain ambiguous and are rejected.
        if config[primary_key] != config[legacy_key]:
            raise ValueError(
                f"{context} cannot define different values for '{primary_key}' "
                f"and legacy '{legacy_key}'. Use '{primary_key}' only."
            )
        return config[primary_key]
    return config.get(primary_key, config.get(legacy_key, []))


def _parse_loading_overrides(fault_name, raw_regions, fault_defaults, blocks_config) -> list[Dict[str, Any]]:
    """Parse optional local loading-definition overrides for one fault.

    Loading overrides are a deliberately small local layer: each item uses a
    selector to choose patches and may override the fault-level block pair,
    reference strike, and motion sense.  Overlap and coverage depend on fault
    geometry and are checked when loading terms are built.
    """
    if raw_regions in (None, []):
        return []
    if not isinstance(raw_regions, list):
        raise ValueError(f"fault_loading.faults.{fault_name}.loading_overrides must be a list")

    parsed = []
    for index, raw_region in enumerate(raw_regions):
        if not isinstance(raw_region, Mapping):
            raise ValueError(f"fault_loading.faults.{fault_name}.loading_overrides[{index}] must be a mapping")
        unknown = sorted(set(raw_region) - LOADING_OVERRIDE_KEYS)
        if unknown:
            raise ValueError(
                f"Unsupported key(s) in fault_loading.faults.{fault_name}.loading_overrides[{index}]: "
                f"{', '.join(unknown)}"
            )
        if "selector" not in raw_region:
            raise ValueError(
                f"fault_loading.faults.{fault_name}.loading_overrides[{index}] is missing required 'selector'"
            )

        name = str(raw_region.get("name", f"region_{index}"))
        block_types = raw_region.get("block_types", fault_defaults["block_types"])
        blocks = raw_region.get("blocks", fault_defaults.get("blocks_original", fault_defaults.get("blocks", [])))
        units = merge_with_defaults(
            raw_region.get("units", {}),
            fault_defaults.get("units", {}),
        )
        if len(block_types) != 2:
            raise ValueError(
                f"fault_loading.faults.{fault_name}.loading_overrides[{index}].block_types "
                "must contain exactly two entries"
            )
        if len(blocks) != 2:
            raise ValueError(
                f"fault_loading.faults.{fault_name}.loading_overrides[{index}].blocks "
                "must contain exactly two entries"
            )
        for block_type in block_types:
            if block_type not in VALID_FAULT_LOADING_BLOCK_TYPES:
                raise ValueError(
                    f"Invalid fault_loading block_type '{block_type}' in "
                    f"fault_loading.faults.{fault_name}.loading_overrides[{index}]"
                )

        known_blocks = blocks_config.get("items", {})
        for block_idx, (block_type, block) in enumerate(zip(block_types, blocks)):
            if block_type == "block" and block not in known_blocks:
                raise ValueError(
                    f"fault_loading.faults.{fault_name}.loading_overrides[{index}] "
                    f"references undefined block '{block}'"
                )
            if block_type in VALID_BLOCK_TYPES:
                _validate_block_value(f"{fault_name}.loading_overrides[{index}]", block_idx, block_type, block, set())

        block_names = raw_region.get("block_names", list(blocks))
        if len(block_names) != 2:
            raise ValueError(
                f"fault_loading.faults.{fault_name}.loading_overrides[{index}].block_names "
                "must contain exactly two entries"
            )

        reference_strike = _validate_reference_strike(
            raw_region.get("reference_strike", fault_defaults["reference_strike"]),
            f"fault_loading.faults.{fault_name}.loading_overrides[{index}]",
        )
        motion_sense = _validate_motion_sense(
            raw_region.get("motion_sense", fault_defaults["motion_sense"]),
            f"fault_loading.faults.{fault_name}.loading_overrides[{index}]",
        )

        parsed.append({
            "name": name,
            "selector": deepcopy(raw_region["selector"]),
            "block_types": list(block_types),
            "blocks": deepcopy(list(blocks)),
            "blocks_original": deepcopy(list(blocks)),
            "blocks_standard": _standardize_fault_loading_blocks(block_types, blocks, units),
            "block_names": deepcopy(list(block_names)),
            "reference_strike": reference_strike,
            "motion_sense": motion_sense,
            "owner_source": raw_region.get("owner_source", fault_defaults.get("owner_source")),
            "units": units,
        })
    return parsed


def _parse_motion_sense_overrides(fault_name, raw_overrides) -> list[Dict[str, Any]]:
    """Parse local motion-sense overrides that do not redefine loading ``b``."""
    if raw_overrides in (None, []):
        return []
    if not isinstance(raw_overrides, list):
        raise ValueError(f"fault_loading.faults.{fault_name}.motion_sense_overrides must be a list")

    parsed = []
    for index, raw_override in enumerate(raw_overrides):
        if not isinstance(raw_override, Mapping):
            raise ValueError(
                f"fault_loading.faults.{fault_name}.motion_sense_overrides[{index}] must be a mapping"
            )
        unknown = sorted(set(raw_override) - MOTION_SENSE_OVERRIDE_KEYS)
        if unknown:
            raise ValueError(
                f"Unsupported key(s) in fault_loading.faults.{fault_name}.motion_sense_overrides[{index}]: "
                f"{', '.join(unknown)}"
            )
        if "selector" not in raw_override:
            raise ValueError(
                f"fault_loading.faults.{fault_name}.motion_sense_overrides[{index}] "
                "is missing required 'selector'"
            )
        if "motion_sense" not in raw_override:
            raise ValueError(
                f"fault_loading.faults.{fault_name}.motion_sense_overrides[{index}] "
                "is missing required 'motion_sense'"
            )
        parsed.append({
            "name": str(raw_override.get("name", raw_override.get("id", f"motion_sense_{index}"))),
            "selector": deepcopy(raw_override["selector"]),
            "motion_sense": _validate_motion_sense(
                raw_override["motion_sense"],
                f"fault_loading.faults.{fault_name}.motion_sense_overrides[{index}]",
            ),
        })
    return parsed


def _standardize_fault_loading_blocks(block_types, blocks, units) -> list[Any]:
    standardized = []
    for block_type, block in zip(block_types, blocks):
        if block_type == "block":
            standardized.append(block)
        elif block_type == "dataset":
            standardized.append(block)
        elif block_type == "euler_pole":
            _, vector = standardize_euler_pole(block, units["euler_pole_units"])
            standardized.append(vector.tolist())
        elif block_type == "euler_vector":
            standardized.append(standardize_euler_vector(block, units["euler_vector_units"]).tolist())
    return standardized


def _parse_cap_constraints(raw_cap: Mapping[str, Any], fault_loading: Mapping[str, Any], faultnames) -> Dict[str, Any]:
    raw_cap = dict(raw_cap or {})
    _validate_cap_constraint_keys(raw_cap, CAP_CONSTRAINT_TOP_LEVEL_KEYS, "cap_constraints")
    enabled = bool(raw_cap.get("enabled", False))
    defaults = {
        "selector": None,
        "max_coupling": 1.0,
        "mode": "motion_sense",
        "hard_overlap": "skip",
        "min_loading_abs": 0.0,
    }
    raw_defaults = dict(raw_cap.get("defaults", {}) or {})
    _validate_cap_constraint_keys(raw_defaults, CAP_CONSTRAINT_DEFAULT_KEYS, "cap_constraints.defaults")
    if "factor" in raw_defaults:
        raw_defaults["max_coupling"] = raw_defaults["factor"]
    defaults = merge_with_defaults(raw_defaults, defaults)
    defaults["max_coupling"] = _validate_cap_max_coupling(
        defaults.get("max_coupling", defaults.get("factor", 1.0)),
        "cap_constraints.defaults.max_coupling",
    )
    defaults["mode"] = _validate_cap_mode(defaults.get("mode", "motion_sense"), "cap_constraints.defaults.mode")
    defaults["hard_overlap"] = _validate_cap_hard_overlap(
        defaults.get("hard_overlap", "skip"),
        "cap_constraints.defaults.hard_overlap",
    )
    defaults["min_loading_abs"] = _validate_cap_min_loading_abs(
        defaults.get("min_loading_abs", 0.0),
        "cap_constraints.defaults.min_loading_abs",
    )
    if not enabled:
        return {
            "enabled": False,
            "defaults": defaults,
            "faults": {},
            "configured_faults": [],
        }

    raw_faults = raw_cap.get("faults")
    if raw_faults is None:
        raw_faults = {name: {} for name in fault_loading.get("configured_faults", [])}
    if not isinstance(raw_faults, Mapping):
        raise ValueError("cap_constraints.faults must be a mapping when provided")

    parsed_faults = {}
    for fault_name, fault_config in raw_faults.items():
        if fault_name not in faultnames:
            raise ValueError(f"cap_constraints references unknown fault '{fault_name}'")
        if fault_name not in fault_loading.get("faults", {}):
            raise ValueError(
                f"cap_constraints.faults.{fault_name} requires a matching fault_loading fault entry"
            )
        fault_config = dict(fault_config or {})
        _validate_cap_constraint_keys(
            fault_config,
            CAP_CONSTRAINT_FAULT_KEYS,
            f"cap_constraints.faults.{fault_name}",
        )
        if "factor" in fault_config:
            fault_config["max_coupling"] = fault_config["factor"]
        fault_cfg = merge_with_defaults(dict(fault_config or {}), defaults)
        if bool(fault_cfg.get("enabled", True)):
            parsed_faults[fault_name] = {
                "enabled": True,
                "selector": deepcopy(fault_cfg.get("selector")),
                "max_coupling": _validate_cap_max_coupling(
                    fault_cfg.get("max_coupling", 1.0),
                    f"cap_constraints.faults.{fault_name}.max_coupling",
                ),
                "mode": _validate_cap_mode(
                    fault_cfg.get("mode", "motion_sense"),
                    f"cap_constraints.faults.{fault_name}.mode",
                ),
                "hard_overlap": _validate_cap_hard_overlap(
                    fault_cfg.get("hard_overlap", "skip"),
                    f"cap_constraints.faults.{fault_name}.hard_overlap",
                ),
                "min_loading_abs": _validate_cap_min_loading_abs(
                    fault_cfg.get("min_loading_abs", 0.0),
                    f"cap_constraints.faults.{fault_name}.min_loading_abs",
                ),
            }

    return {
        "enabled": True,
        "defaults": defaults,
        "faults": parsed_faults,
        "configured_faults": list(parsed_faults.keys()),
    }


def _validate_cap_constraint_keys(raw_mapping: Mapping[str, Any], allowed_keys: set[str], context: str) -> None:
    """Reject silent no-op keys in cap constraints.

    ``cap_constraints`` defines where and how cap inequalities are built.  The
    loading definition itself belongs to ``fault_loading``.  Raising here keeps
    misplaced fields such as ``motion_sense`` from being silently ignored.
    """
    unknown = sorted(set(raw_mapping) - allowed_keys)
    if not unknown:
        return

    misplaced_loading_keys = [key for key in unknown if key in FAULT_LOADING_ONLY_KEYS]
    if misplaced_loading_keys:
        key = misplaced_loading_keys[0]
        if context == "cap_constraints":
            target = f"fault_loading.defaults.{key}"
        elif context == "cap_constraints.defaults":
            target = f"fault_loading.defaults.{key}"
        elif context.startswith("cap_constraints.faults."):
            fault_name = context.rsplit(".", 1)[-1]
            target = f"fault_loading.faults.{fault_name}.{key}"
        else:
            target = f"fault_loading.{key}"
        raise ValueError(
            f"{context}.{key} is not supported. Put loading-definition fields "
            f"under {target}; cap_constraints only selects cap inequalities "
            "and their mode/max_coupling/overlap behavior."
        )

    allowed = ", ".join(sorted(allowed_keys))
    unknown_text = ", ".join(unknown)
    raise ValueError(f"Unsupported key(s) in {context}: {unknown_text}. Allowed keys: {allowed}")


def _parse_backslip_constraints(raw_constraints, faultnames) -> list[Dict[str, Any]]:
    if raw_constraints is None:
        return []
    if not isinstance(raw_constraints, list):
        raise ValueError("backslip_constraints must be a list of mappings")
    parsed = []
    for idx, item in enumerate(raw_constraints):
        if not isinstance(item, Mapping):
            raise ValueError(f"backslip_constraints[{idx}] must be a mapping")
        spec = dict(item)
        _validate_mapping_keys(
            spec, BACKSLIP_CONSTRAINT_KEYS, f"backslip_constraints[{idx}]"
        )
        fault_name = spec.get("fault")
        if not fault_name:
            raise ValueError(f"backslip_constraints[{idx}] is missing required 'fault'")
        if fault_name not in faultnames:
            raise ValueError(f"backslip_constraints[{idx}] references unknown fault '{fault_name}'")
        if "state" not in spec:
            raise ValueError(f"backslip_constraints[{idx}] is missing required 'state'")
        parsed.append(deepcopy(spec))
    return parsed


def _validate_units(units, unit_type: str) -> None:
    parse_euler_units(units, unit_type)


def _validate_reference_strike(strike, context: str) -> float:
    if not isinstance(strike, (int, float)):
        raise ValueError(f"reference_strike in {context} must be numeric")
    return float(strike % 360)


def _validate_motion_sense(motion_sense, context: str) -> str:
    if motion_sense not in VALID_MOTION_SENSES:
        allowed = ", ".join(sorted(VALID_MOTION_SENSES))
        raise ValueError(f"Invalid motion_sense '{motion_sense}' in {context}; expected one of {allowed}")
    return str(motion_sense)


def _validate_cap_max_coupling(value, context: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{context} must be non-negative")
    return value


def _validate_cap_mode(value, context: str) -> str:
    key = str(value).lower().replace("-", "_").replace(" ", "_")
    mode = CAP_CONSTRAINT_MODE_ALIASES.get(key)
    if mode not in VALID_CAP_CONSTRAINT_MODES:
        allowed = ", ".join(sorted(VALID_CAP_CONSTRAINT_MODES))
        raise ValueError(f"Invalid cap constraint mode '{value}' in {context}; expected one of {allowed}")
    return mode


def _validate_cap_hard_overlap(value, context: str) -> str:
    key = str(value).lower().replace("-", "_").replace(" ", "_")
    policy = CAP_HARD_OVERLAP_ALIASES.get(key)
    if policy not in VALID_CAP_HARD_OVERLAP_POLICIES:
        allowed = ", ".join(sorted(VALID_CAP_HARD_OVERLAP_POLICIES))
        raise ValueError(f"Invalid cap hard_overlap '{value}' in {context}; expected one of {allowed}")
    return policy


def _validate_cap_min_loading_abs(value, context: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{context} must be non-negative")
    return value


def _validate_block_value(fault_name, idx, block_type, block, dataset_names) -> None:
    if block_type == "dataset":
        if not isinstance(block, str):
            raise ValueError(f"Block {idx} for fault '{fault_name}' with type dataset must be a dataset name")
        if dataset_names and block not in dataset_names:
            raise ValueError(f"Dataset '{block}' for fault '{fault_name}' not found in geodata")
    elif block_type in {"euler_pole", "euler_vector"}:
        if not isinstance(block, (list, tuple)) or len(block) != 3:
            raise ValueError(f"Block {idx} for fault '{fault_name}' with type {block_type} must have three values")
        if not all(isinstance(value, (int, float)) for value in block):
            raise ValueError(f"Block {idx} for fault '{fault_name}' with type {block_type} must be numeric")
