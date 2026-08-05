"""Small helpers for script-side interseismic configuration updates.

These helpers keep dynamic case logic in Python scripts while writing standard
``fault_loading.loading_overrides[].selector`` dictionaries back to the existing
interseismic configuration structure.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .patch_indices import (
    get_patches_in_trace_segment,
    resolve_trace_marker,
    trace_range_selector_from_markers,
)


def _override_items(config: Mapping[str, Any], fault_name: str, section: str) -> list[dict[str, Any]]:
    try:
        fault_cfg = config["fault_loading"]["faults"][fault_name]
        if section == "loading_overrides" and section not in fault_cfg and "loading_regions" in fault_cfg:
            section = "loading_regions"
        overrides = fault_cfg[section]
    except KeyError as exc:
        raise KeyError(
            f"interseismic_config.fault_loading.faults.{fault_name}.{section} not found"
        ) from exc
    if not isinstance(overrides, list):
        raise TypeError(
            f"interseismic_config.fault_loading.faults.{fault_name}.{section} must be a list"
        )
    return overrides


def _override_index(overrides: list[dict[str, Any]], override: int | str) -> int:
    if isinstance(override, int):
        if override < 0 or override >= len(overrides):
            raise IndexError(f"override index {override} is out of range")
        return override

    target = str(override)
    matches = [
        index
        for index, item in enumerate(overrides)
        if str(item.get("name", item.get("id", ""))) == target
    ]
    if not matches:
        raise KeyError(f"override named '{target}' was not found")
    if len(matches) > 1:
        raise ValueError(f"override name '{target}' is not unique")
    return matches[0]


def set_fault_loading_override_selector(
    interseismic_config: Mapping[str, Any],
    fault_name: str,
    override: int | str,
    selector: Mapping[str, Any],
    *,
    inplace: bool = False,
) -> dict[str, Any]:
    """Set one ``fault_loading.loading_overrides`` selector.

    Parameters
    ----------
    interseismic_config : mapping
        Parsed or raw interseismic configuration dictionary.
    fault_name : str
        Fault name under ``fault_loading.faults``.
    override : int or str
        Override index, or an override ``name``/``id``.
    selector : mapping
        Standard selector such as ``{"trace_range": ...}`` or
        ``{"patches": [...]}``.
    inplace : bool, default False
        If False, return a deep copy and leave the input unchanged.
    """
    config = interseismic_config if inplace else deepcopy(interseismic_config)
    overrides = _override_items(config, fault_name, "loading_overrides")
    overrides[_override_index(overrides, override)]["selector"] = dict(selector)
    return config


def set_fault_motion_sense_override_selector(
    interseismic_config: Mapping[str, Any],
    fault_name: str,
    override: int | str,
    selector: Mapping[str, Any],
    *,
    inplace: bool = False,
) -> dict[str, Any]:
    """Set one ``fault_loading.motion_sense_overrides`` selector."""
    config = interseismic_config if inplace else deepcopy(interseismic_config)
    overrides = _override_items(config, fault_name, "motion_sense_overrides")
    overrides[_override_index(overrides, override)]["selector"] = dict(selector)
    return config


def _selector_from_trace_segment(
    fault: Any,
    start: Any,
    end: Any,
    *,
    buffer_distance: float | None,
    depth_range,
    coord_system: str,
    use_discretized: bool,
    selector_format: str,
    output_coord_system: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start_marker = resolve_trace_marker(
        fault,
        start,
        coord_system=coord_system,
        use_discretized=use_discretized,
    )
    end_marker = resolve_trace_marker(
        fault,
        end,
        coord_system=coord_system,
        use_discretized=use_discretized,
    )

    key = selector_format.lower().replace("-", "_")
    if key == "trace_range":
        selector = trace_range_selector_from_markers(
            fault,
            start_marker,
            end_marker,
            buffer_distance=buffer_distance,
            depth_range=depth_range,
            coord_system="xy",
            use_discretized=use_discretized,
            output_coord_system=output_coord_system,
        )
        patch_indices = None
    elif key in ("patches", "patch_indices"):
        patch_indices = get_patches_in_trace_segment(
            fault,
            start_marker,
            end_marker,
            buffer_distance=buffer_distance,
            depth_range=depth_range,
            coord_system="xy",
            use_discretized=use_discretized,
        )
        selector = {"patches": [int(index) for index in patch_indices.tolist()]}
    else:
        raise ValueError("selector_format must be 'trace_range' or 'patches'")

    metadata = {
        "selector": selector,
        "start_marker": start_marker.to_dict(),
        "end_marker": end_marker.to_dict(),
        "trace_distance_km": start_marker.trace_distance_km,
        "segment_length_km": abs(end_marker.trace_distance_km - start_marker.trace_distance_km),
    }
    if patch_indices is not None:
        metadata["patch_indices"] = [int(index) for index in patch_indices.tolist()]
    return selector, metadata


def update_fault_loading_override_from_trace_segment(
    interseismic_config: Mapping[str, Any],
    fault: Any,
    fault_name: str,
    override: int | str,
    start: Any,
    end: Any,
    *,
    buffer_distance: float | None = None,
    depth_range=None,
    coord_system: str = "lonlat",
    use_discretized: bool = True,
    selector_format: str = "trace_range",
    output_coord_system: str = "lonlat",
    inplace: bool = False,
    return_metadata: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Update one loading-override selector from flexible trace markers.

    ``selector_format="trace_range"`` stores compact resolved endpoints.
    ``selector_format="patches"`` stores explicit patch ids.  Both formats are
    standard selectors consumed by the existing constraint and field code.
    """
    selector, metadata = _selector_from_trace_segment(
        fault,
        start,
        end,
        buffer_distance=buffer_distance,
        depth_range=depth_range,
        coord_system=coord_system,
        use_discretized=use_discretized,
        selector_format=selector_format,
        output_coord_system=output_coord_system,
    )

    config = set_fault_loading_override_selector(
        interseismic_config,
        fault_name,
        override,
        selector,
        inplace=inplace,
    )

    if not return_metadata:
        return config

    return config, metadata


def update_fault_motion_sense_override_from_trace_segment(
    interseismic_config: Mapping[str, Any],
    fault: Any,
    fault_name: str,
    override: int | str,
    start: Any,
    end: Any,
    *,
    buffer_distance: float | None = None,
    depth_range=None,
    coord_system: str = "lonlat",
    use_discretized: bool = True,
    selector_format: str = "trace_range",
    output_coord_system: str = "lonlat",
    inplace: bool = False,
    return_metadata: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Update one motion-sense override selector from flexible trace markers."""
    selector, metadata = _selector_from_trace_segment(
        fault,
        start,
        end,
        buffer_distance=buffer_distance,
        depth_range=depth_range,
        coord_system=coord_system,
        use_discretized=use_discretized,
        selector_format=selector_format,
        output_coord_system=output_coord_system,
    )
    config = interseismic_config if inplace else deepcopy(interseismic_config)
    set_fault_motion_sense_override_selector(
        config,
        fault_name,
        override,
        selector,
        inplace=True,
    )
    if return_metadata:
        return config, metadata
    return config


# Backward-compatible aliases for older local scripts.
set_fault_loading_region_selector = set_fault_loading_override_selector
update_fault_loading_region_from_trace_segment = update_fault_loading_override_from_trace_segment
