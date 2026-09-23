"""Read-only reporting for inversion parameter layouts.

This module describes parameter coordinates that have already been resolved by
configuration, source adapters, and constraint managers.  It never allocates
parameters, rebuilds constraints, reads posterior values, or changes solver
state.  The small dictionary protocol intentionally separates three concerns:

* inversion classes collect authoritative layout facts;
* these helpers validate and detach the facts;
* the formatter renders a compact human-readable table.

Half-open ranges are used throughout.  A range such as ``S[2:5)`` therefore
contains indices 2, 3, and 4 in the space identified by ``S``.
"""

from __future__ import annotations

import copy
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_VALID_USES = {"sample", "solve", "estimate", "fixed", "configured", "derived"}


def _as_int_pair(value: Any, *, name: str) -> tuple[int, int]:
    if value is None or len(value) != 2:
        raise ValueError(f"{name} must be a two-element half-open range")
    start, stop = (int(item) for item in value)
    if start < 0 or stop < start:
        raise ValueError(f"{name} has invalid range [{start}, {stop})")
    return start, stop


def make_parameter_layout_report(
    *,
    mode: str,
    spaces: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate and detach one solver-neutral parameter-layout report."""

    detached_spaces = []
    spaces_by_key = {}
    for raw in spaces:
        key = str(raw["key"])
        if not key or key in spaces_by_key:
            raise ValueError(f"Parameter space key must be unique; got {key!r}")
        width = int(raw["width"])
        offset = int(raw.get("global_offset", 0))
        if width < 0 or offset < 0:
            raise ValueError("Parameter space width and global_offset must be non-negative")
        space = {
            "key": key,
            "name": str(raw["name"]),
            "width": width,
            "global_offset": offset,
        }
        detached_spaces.append(space)
        spaces_by_key[key] = space

    detached_rows = []
    for raw in rows:
        use = str(raw["use"])
        if use not in _VALID_USES:
            raise ValueError(f"Unknown parameter-layout use {use!r}")
        space_key = raw.get("space")
        start = raw.get("start")
        stop = raw.get("stop")
        if space_key is None:
            if start is not None or stop is not None:
                raise ValueError("Rows without a parameter space cannot have a range")
            count = int(raw.get("count", 1))
        else:
            space_key = str(space_key)
            if space_key not in spaces_by_key:
                raise ValueError(f"Unknown parameter space {space_key!r}")
            start, stop = _as_int_pair((start, stop), name="parameter row")
            if stop > spaces_by_key[space_key]["width"]:
                raise ValueError(
                    f"Parameter row {space_key}[{start}:{stop}) exceeds space width "
                    f"{spaces_by_key[space_key]['width']}"
                )
            count = stop - start
        if count < 0:
            raise ValueError("Parameter row count must be non-negative")
        row = {
            "kind": str(raw["kind"]),
            "owner": str(raw.get("owner", "-")),
            "space": space_key,
            "start": start,
            "stop": stop,
            "count": count,
            "use": use,
            "content": str(raw.get("content", "-")),
            "units": tuple(str(value) for value in raw.get("units", ())),
            "details": copy.deepcopy(raw.get("details", {})),
        }
        detached_rows.append(row)

    return {
        "schema_version": 1,
        "mode": str(mode),
        "spaces": detached_spaces,
        "rows": detached_rows,
        "notes": [str(note) for note in notes],
    }


def build_scale_layout_rows(
    *,
    kind: str,
    layout: Mapping[str, Any],
    sample_slice: Sequence[int] | None,
    sampled_use: str,
    fixed_use: str = "fixed",
    space: str = "S",
    log_scaled: bool = False,
) -> list[dict[str, Any]]:
    """Build one structural row per canonical sigma/alpha group."""

    group_names = list(layout.get("group_names", ()))
    members_by_group = layout.get("members_by_group", {})
    update = np.asarray(
        layout.get("update_by_group", np.zeros(len(group_names), dtype=bool)),
        dtype=bool,
    )
    sample_indices = np.asarray(
        layout.get("sample_index_by_group", np.full(len(group_names), -1, dtype=int)),
        dtype=int,
    )
    if update.shape != (len(group_names),) or sample_indices.shape != (len(group_names),):
        raise ValueError(f"{kind} group layout has inconsistent update/sample arrays")
    unknown = [name for name in group_names if name not in members_by_group]
    if unknown:
        raise ValueError(f"{kind} group layout is missing members for: {', '.join(unknown)}")

    sample_start = None
    sample_stop = None
    if sample_slice is not None:
        sample_start, sample_stop = _as_int_pair(sample_slice, name=f"{kind} sample slice")
        if sample_stop - sample_start != int(np.sum(update)):
            raise ValueError(
                f"{kind} sample slice width does not match the number of sampled groups"
            )
    elif np.any(update) and sampled_use == "sample":
        raise ValueError(f"{kind} sampled groups require a sample slice")

    coordinate = "log10(s)" if log_scaled else "s"
    rows = []
    for index, group_name in enumerate(group_names):
        members = [str(value) for value in members_by_group[group_name]]
        is_active = bool(update[index])
        start = stop = None
        row_space = None
        if is_active and sampled_use == "sample":
            local = int(sample_indices[index])
            if local < 0:
                raise ValueError(f"Sampled {kind} group '{group_name}' has no sample index")
            start = sample_start + local
            stop = start + 1
            row_space = space
        member_text = ", ".join(members) if members else "-"
        if is_active:
            content = f"{coordinate}; members={member_text}"
            use = sampled_use
        else:
            content = f"fixed {coordinate}; members={member_text}"
            use = fixed_use
        rows.append(
            {
                "kind": str(kind),
                "owner": str(group_name),
                "space": row_space,
                "start": start,
                "stop": stop,
                "count": 1,
                "use": use,
                "content": content,
                "units": ("scale",),
                "details": {
                    "members": members,
                    "sampling_coordinate": coordinate if is_active else None,
                    "physical_coordinate": "s",
                    "update": is_active,
                },
            }
        )
    return rows


def build_active_scale_layout_rows(
    *,
    kind: str,
    result_rows: Sequence[Mapping[str, Any]],
    log_scaled: bool = False,
) -> list[dict[str, Any]]:
    """Build structural rows from scales frozen by a completed BLSE/VCE run.

    The active scale report is authoritative after a solve because VCE may
    override the configured grouping and update flags at runtime.  This
    adapter deliberately copies only structural facts (group, members, and
    estimated/fixed state); numerical scale values remain owned by the scale
    report and are never fed back into the layout.
    """

    selected = [row for row in result_rows if str(row.get("kind")) == str(kind)]
    group_names = [str(row["group"]) for row in selected]
    if len(group_names) != len(set(group_names)):
        raise ValueError(f"Active {kind} scale rows contain duplicate groups")

    members_by_group = {}
    update_by_group = []
    sample_index_by_group = []
    next_update = 0
    for row, group_name in zip(selected, group_names):
        state = str(row.get("state"))
        if state not in {"estimated", "fixed"}:
            raise ValueError(
                f"Active {kind} scale group '{group_name}' has unsupported "
                f"state {state!r}"
            )
        members_by_group[group_name] = [str(value) for value in row.get("members", ())]
        estimated = state == "estimated"
        update_by_group.append(estimated)
        sample_index_by_group.append(next_update if estimated else -1)
        next_update += int(estimated)

    layout = {
        "group_names": group_names,
        "members_by_group": members_by_group,
        "update_by_group": np.asarray(update_by_group, dtype=bool),
        "sample_index_by_group": np.asarray(sample_index_by_group, dtype=int),
    }
    return build_scale_layout_rows(
        kind=kind,
        layout=layout,
        sample_slice=None,
        sampled_use="estimate",
        fixed_use="fixed",
        log_scaled=log_scaled,
    )


def build_geometry_layout_rows(resolved_updates: Sequence[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Build compact rows from resolved Bayesian geometry contracts."""

    by_slice: dict[tuple[int, int], list[Any]] = {}
    for resolved in resolved_updates:
        start, stop = _as_int_pair(resolved.sample_slice, name="geometry sample slice")
        by_slice.setdefault((start, stop), []).append(resolved)

    rows = []
    notes = []
    for (start, stop), owners in sorted(by_slice.items()):
        count = stop - start
        owner_names = [str(owner.fault_name) for owner in owners]
        methods = list(dict.fromkeys(str(owner.method_name) for owner in owners))
        declared_names = [tuple(getattr(owner, "parameter_names", ()) or ()) for owner in owners]
        shared_names = declared_names[0] if declared_names and all(
            names == declared_names[0] for names in declared_names
        ) else ()
        contracts = [owner.registry_contract or {} for owner in owners]
        specs = [tuple((contract.get("parameter_spec") or {}).get("items") or ()) for contract in contracts]
        shared_spec = specs[0] if specs and all(spec == specs[0] for spec in specs) else ()

        parameters = []
        units = []
        for local_index in range(count):
            item = {}
            repeated = False
            if len(shared_spec) == count:
                item = shared_spec[local_index]
            elif len(shared_spec) == 1:
                item = shared_spec[0]
                repeated = count > 1
            if len(shared_names) == count:
                role = str(shared_names[local_index])
            else:
                role = str(item.get("role") or "parameter")
                if repeated or (not item and count > 1):
                    role += f"[{local_index}]"
            unit = item.get("unit")
            unit_from = item.get("unit_from")
            if unit is None and unit_from:
                values = []
                for owner in owners:
                    value = owner.method_kwargs.get(unit_from)
                    if value is None:
                        value = (owner.registry_contract.get("kwargs") or {}).get(unit_from)
                    values.append(value)
                if values and all(value == values[0] for value in values):
                    unit = values[0]
            parameters.append(role)
            units.append("-" if unit is None else str(unit))

        content_items = [
            name if unit == "-" else f"{name}:{unit}"
            for name, unit in zip(parameters, units)
        ]
        rows.append(
            {
                "kind": "geom",
                "owner": ",".join(owner_names),
                "space": "S",
                "start": start,
                "stop": stop,
                "use": "sample",
                "content": ", ".join(content_items),
                "units": units,
                "details": {
                    "faults": owner_names,
                    "methods": methods,
                    "parameters": parameters,
                },
            }
        )
        notes.append(f"geometry {','.join(owner_names)} -> {','.join(methods)}")
    return rows, notes


def build_linear_layout_rows(
    *,
    layout: Mapping[str, Any],
    adapters: Mapping[str, Any],
    observation_unit: str,
    use: str,
    space: str,
    poly_descriptions: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate a validated constraint-manager layout into compact rows."""

    if not layout.get("active", False):
        return []
    blocks = list(layout.get("blocks", ()))
    poly_descriptions = {} if poly_descriptions is None else poly_descriptions
    rows = []
    source_blocks: dict[str, list[Mapping[str, Any]]] = {}
    poly_blocks = {}
    for block in blocks:
        source = str(block["source"])
        if block.get("role") == "data_correction":
            poly_blocks[source] = block
        else:
            source_blocks.setdefault(source, []).append(block)

    ordered_sources = []
    for block in blocks:
        source = str(block["source"])
        if source not in ordered_sources:
            ordered_sources.append(source)
    for source_name in ordered_sources:
        components = source_blocks.get(source_name, [])
        if components:
            start = int(components[0]["start"])
            stop = int(components[-1]["stop"])
            labels = [
                f"{block['component']}[{int(block['stop']) - int(block['start'])}]"
                for block in components
            ]
            adapter = adapters.get(source_name)
            source_type = getattr(adapter, "source_type", None)
            unit = observation_unit if source_type == "Fault" else "-"
            content = ", ".join(labels)
            if unit != "-":
                content += f"; unit={unit}"
            rows.append(
                {
                    "kind": "slip" if source_type == "Fault" else "source",
                    "owner": source_name,
                    "space": space,
                    "start": start,
                    "stop": stop,
                    "use": use,
                    "content": content,
                    "units": (unit,),
                    "details": {"source_type": source_type, "components": labels},
                }
            )

        poly_block = poly_blocks.get(source_name)
        if poly_block is None:
            continue
        descriptions = list(poly_descriptions.get(source_name, ()))
        if descriptions:
            for description in descriptions:
                rows.append(
                    {
                        "kind": "poly",
                        "owner": f"{source_name}/{description['dataset']}",
                        "space": space,
                        "start": int(description["start"]),
                        "stop": int(description["stop"]),
                        "use": use,
                        "content": str(description["content"]),
                        "units": ("raw",),
                        "details": copy.deepcopy(description.get("details", {})),
                    }
                )
        else:
            start = int(poly_block["start"])
            stop = int(poly_block["stop"])
            rows.append(
                {
                    "kind": "poly",
                    "owner": source_name,
                    "space": space,
                    "start": start,
                    "stop": stop,
                    "use": use,
                    "content": f"data correction[{stop - start}]; raw coordinates",
                    "units": ("raw",),
                    "details": {},
                }
            )
    return rows


def describe_data_correction_blocks(
    inversion: Any,
    source: Any,
    *,
    start: int,
    stop: int,
) -> list[dict[str, Any]]:
    """Describe one source's data-correction columns in configured order.

    The parameter counts and ordering remain owned by CSI's ``poly`` and
    ``numberofpolys`` state.  Existing transform-component resolvers are used
    only for labels; an unknown transform falls back to deterministic ``p[i]``
    names and never changes a column.
    """

    from .data_correction_constraints import (
        _configured_dataset_order,
        _get_data_by_name,
        _iter_transform_blocks,
        _transform_components,
    )

    start, stop = _as_int_pair((start, stop), name="data-correction block")
    numberofpolys = getattr(source, "numberofpolys", {}) or {}
    poly = getattr(source, "poly", {}) or {}
    cursor = start
    result = []
    for dataset in _configured_dataset_order(source):
        count = int(numberofpolys.get(dataset, 0))
        if count <= 0:
            continue
        dataset_start = cursor
        dataset_stop = cursor + count
        if dataset_stop > stop:
            raise ValueError(
                f"Data-correction layout for '{source.name}/{dataset}' exceeds "
                f"the declared polynomial block [{start}, {stop})"
            )
        transform_config = poly.get(dataset)
        try:
            data = _get_data_by_name(inversion, dataset)
        except ValueError:
            data = None
        labels = []
        details = []
        if transform_config is None:
            labels = [f"p[{index}]" for index in range(count)]
        else:
            for transform, _local_start, n_params in _iter_transform_blocks(
                transform_config, data, count
            ):
                components = _transform_components(transform, n_params, data=data)
                components = list(components[:n_params])
                if len(components) < n_params:
                    components.extend(
                        f"p[{index}]" for index in range(len(components), n_params)
                    )
                labels.extend(components)
                details.append(
                    {"transform": str(transform), "components": components}
                )
        if len(labels) != count:
            labels = [f"p[{index}]" for index in range(count)]
            details = []
        result.append(
            {
                "dataset": str(dataset),
                "start": dataset_start,
                "stop": dataset_stop,
                "content": ", ".join(labels) + "; raw coordinates",
                "details": {"transforms": details, "components": labels},
            }
        )
        cursor = dataset_stop
    if cursor != stop:
        raise ValueError(
            f"Data-correction labels for '{source.name}' cover [{start}, {cursor}), "
            f"but the declared block ends at {stop}"
        )
    return result


def build_sampled_source_rows(
    *,
    inversion: Any,
    sources: Sequence[Any],
    adapters: Mapping[str, Any],
    source_positions: Mapping[str, Sequence[int]],
    poly_positions: Mapping[str, Sequence[int]],
    observation_unit: str,
    slip_sampling_mode: str,
    use: str = "sample",
    space: str = "S",
) -> list[dict[str, Any]]:
    """Describe FULLSMC source/poly blocks from canonical sampled positions."""

    rows = []
    mode = str(slip_sampling_mode)
    for source in sources:
        name = str(source.name)
        adapter = adapters[name]
        start, stop = _as_int_pair(source_positions[name], name=f"{name} source sample")
        width = stop - start
        source_type = getattr(adapter, "source_type", None)
        labels = []
        units = []
        if source_type == "Fault" and mode == "rake_fixed":
            labels = [f"magnitude[{width}]", "ss/ds derived from fixed rake"]
            units = [observation_unit, "degrees"]
        elif source_type == "Fault" and mode == "magnitude_rake":
            if width % 2:
                raise ValueError(
                    f"Magnitude/rake sample block for '{name}' has odd width {width}"
                )
            half = width // 2
            labels = [f"magnitude[{half}]", f"rake[{half}]"]
            units = [observation_unit, "degrees"]
        else:
            counts = adapter.get_n_params_per_component()
            component_names = list(adapter.get_param_names())
            if sum(int(counts[value]) for value in component_names) != width:
                raise ValueError(
                    f"Sampled source layout for '{name}' has width {width}, but "
                    "adapter component counts disagree"
                )
            labels = [f"{value}[{int(counts[value])}]" for value in component_names]
            units = [observation_unit if source_type == "Fault" else "-"]
        content = ", ".join(labels)
        if source_type == "Fault":
            content += f"; unit={observation_unit}"
        rows.append(
            {
                "kind": "slip" if source_type == "Fault" else "source",
                "owner": name,
                "space": space,
                "start": start,
                "stop": stop,
                "use": use,
                "content": content,
                "units": units,
                "details": {
                    "source_type": source_type,
                    "parameterization": mode if source_type == "Fault" else "native",
                },
            }
        )

        poly_start, poly_stop = _as_int_pair(
            poly_positions[name], name=f"{name} data-correction sample"
        )
        if poly_start == poly_stop:
            continue
        descriptions = describe_data_correction_blocks(
            inversion, source, start=poly_start, stop=poly_stop
        )
        for description in descriptions:
            rows.append(
                {
                    "kind": "poly",
                    "owner": f"{name}/{description['dataset']}",
                    "space": space,
                    "start": int(description["start"]),
                    "stop": int(description["stop"]),
                    "use": use,
                    "content": str(description["content"]),
                    "units": ("raw",),
                    "details": copy.deepcopy(description.get("details", {})),
                }
            )
    return rows


def _wrapped(value: Any, width: int) -> str:
    text = str(value)
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or "-"


def format_parameter_layout_report(
    report: Mapping[str, Any],
    *,
    summary_only: bool = False,
    tablefmt: str = "simple",
) -> str:
    """Render a compact layout table without changing the report."""

    spaces = list(report.get("spaces", ()))
    counts = []
    for space in spaces:
        label = "sampled" if space["key"] == "S" else "linear"
        counts.append(f"{label}={int(space['width'])}")
    header = "Parameter layout | mode=" + str(report.get("mode", "-"))
    if counts:
        header += " | " + " | ".join(counts)
    if summary_only:
        return header

    lines = [header]
    if spaces:
        definitions = []
        for item in spaces:
            text = f"{item['key']}={item['name']}"
            if int(item.get("global_offset", 0)):
                text += f" (global offset {int(item['global_offset'])})"
            definitions.append(text)
        lines.append("; ".join(definitions) + "; ranges are [start:end).")

    rows = list(report.get("rows", ()))
    if rows:
        from tabulate import tabulate

        table = []
        for row in rows:
            if row.get("space") is None:
                range_text = "-"
            else:
                range_text = (
                    f"{row['space']}[{int(row['start'])}:{int(row['stop'])})"
                )
            table.append(
                [
                    row.get("kind", "-"),
                    _wrapped(row.get("owner", "-"), 24),
                    range_text,
                    int(row.get("count", 0)),
                    row.get("use", "-"),
                    _wrapped(row.get("content", "-"), 46),
                ]
            )
        lines.append(
            tabulate(
                table,
                headers=["Kind", "Owner/group", "Range", "N", "Use", "Content"],
                tablefmt=tablefmt,
                stralign="left",
                numalign="right",
            )
        )
    else:
        lines.append("  No active parameter rows.")
    notes = list(report.get("notes", ()))
    if notes:
        lines.append("Details:")
        lines.extend(f"  {note}" for note in notes)
    return "\n".join(lines)
