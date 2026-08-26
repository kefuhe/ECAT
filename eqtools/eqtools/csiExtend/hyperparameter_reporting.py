"""Read-only reporting helpers for sigma and alpha scale parameters.

The inversion stack stores scale parameters in two coordinate systems:
samplers may use either ``s`` or ``log10(s)``, while likelihoods and weighted
least-squares systems always consume the positive physical scale ``s``.  This
module creates one report record per canonical parameter group without
changing either representation or touching solver state.

The helpers deliberately consume the canonical ``group_layout`` contract.
They never infer parameter cardinality from the number of datasets or sources,
which keeps fixed, sampled, individual, single, and grouped configurations
aligned with the likelihood adapters that own the numerical calculation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def build_scale_parameter_rows(
    *,
    kind: str,
    layout: Mapping[str, Any],
    active_scales_by_group: Mapping[str, float],
    update_state: str,
    log_scaled: bool = False,
    posterior_samples: Any = None,
    sample_index_offset: int | None = None,
    variance_by_group: Mapping[str, float] | None = None,
    diagnostics_by_group: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build canonical, physical-scale report rows.

    Parameters
    ----------
    kind : {"sigma", "alpha"}
        Scientific role of the scale.
    layout : mapping
        Canonical group layout returned by
        :func:`config.parameter_groups.attach_group_parameters`.
    active_scales_by_group : mapping
        Physical positive ``s`` associated with the active model.  These values
        must come from the solver/likelihood adapter, not from report-time
        exponentiation of a representative sample.
    update_state : str
        Label for an active non-fixed group, normally ``"sampled"`` for
        Bayesian output or ``"estimated"`` for VCE output.
    log_scaled : bool, optional
        Whether sampled coordinates are ``log10(s)``.  This affects only the
        interpretation of ``posterior_samples``.
    posterior_samples : array-like, optional
        Two-dimensional sampled-coordinate array containing only the updated
        groups, ordered by ``layout['sample_index_by_group']``.  Physical and
        log-space posterior standard deviations are calculated by transforming
        the full columns, so no delta-method approximation is introduced.
    sample_index_offset : int, optional
        Global vector offset used only for display.
    variance_by_group, diagnostics_by_group : mappings, optional
        VCE-specific values associated with the same active model.

    Returns
    -------
    list of dict
        Serializable, read-only report records.  No input is mutated.
    """

    group_names = list(layout.get("group_names", ()))
    members_by_group = layout.get("members_by_group", {})
    update_by_group = np.asarray(
        layout.get("update_by_group", np.ones(len(group_names), dtype=bool)),
        dtype=bool,
    )
    sample_index_by_group = np.asarray(
        layout.get("sample_index_by_group", np.full(len(group_names), -1, dtype=int)),
        dtype=int,
    )
    if update_by_group.shape != (len(group_names),):
        raise ValueError("group_layout update_by_group does not match group_names")
    if sample_index_by_group.shape != (len(group_names),):
        raise ValueError("group_layout sample_index_by_group does not match group_names")

    samples = None
    if posterior_samples is not None:
        samples = np.asarray(posterior_samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError("posterior_samples must be a two-dimensional array")
        required_columns = int(np.sum(update_by_group))
        if samples.shape[1] != required_columns:
            raise ValueError(
                "posterior_samples column count does not match updated parameter groups: "
                f"{samples.shape[1]} != {required_columns}"
            )

    variance_by_group = variance_by_group or {}
    diagnostics_by_group = diagnostics_by_group or {}
    rows = []
    for group_index, group_name in enumerate(group_names):
        if group_name not in active_scales_by_group:
            raise ValueError(f"Missing active physical scale for group '{group_name}'")
        scale = float(active_scales_by_group[group_name])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"Active physical scale for group '{group_name}' must be finite and positive"
            )

        updated = bool(update_by_group[group_index])
        local_sample_index = int(sample_index_by_group[group_index])
        if updated and local_sample_index < 0:
            raise ValueError(
                f"Updated group '{group_name}' has no sampled/update parameter index"
            )
        if not updated and local_sample_index != -1:
            raise ValueError(
                f"Fixed group '{group_name}' must use sample index -1"
            )

        posterior_scale_std = None
        posterior_log10_std = None
        if updated and samples is not None:
            sample_coordinate = samples[:, local_sample_index]
            physical_samples = (
                np.power(10.0, sample_coordinate)
                if log_scaled else sample_coordinate
            )
            if (
                np.any(~np.isfinite(physical_samples))
                or np.any(physical_samples <= 0.0)
            ):
                raise ValueError(
                    f"Posterior physical scales for group '{group_name}' "
                    "must be finite and positive"
                )
            posterior_scale_std = float(np.std(physical_samples))
            posterior_log10_std = float(np.std(np.log10(physical_samples)))

        diagnostics = diagnostics_by_group.get(group_name, {}) or {}
        variance = variance_by_group.get(group_name)
        has_sampling_coordinate = updated and str(update_state) == "sampled"
        rows.append(
            {
                "index": (
                    None
                    if not updated or sample_index_offset is None
                    else int(sample_index_offset) + local_sample_index
                ),
                "kind": str(kind),
                "group": str(group_name),
                "members": list(members_by_group.get(group_name, ())),
                "state": str(update_state) if updated else "fixed",
                # VCE estimates a physical scale but has no Bayesian sampling
                # coordinate.  Keep the shared column for a stable table
                # layout while publishing ``-`` for estimated/fixed groups.
                "sampling_space": (
                    "log10(s)"
                    if has_sampling_coordinate and log_scaled
                    else "s" if has_sampling_coordinate else "-"
                ),
                "scale": scale,
                "posterior_scale_std": posterior_scale_std,
                "log10_scale": float(np.log10(scale)),
                "posterior_log10_std": posterior_log10_std,
                "row_multiplier": float(1.0 / scale),
                "variance": None if variance is None else float(variance),
                "weighted_quadratic": diagnostics.get("weighted_quadratic"),
                "reduced_weighted_misfit": diagnostics.get(
                    "reduced_weighted_misfit"
                ),
            }
        )
    return rows


def format_scale_parameter_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    show_index: bool = True,
    show_posterior_uncertainty: bool = True,
    show_variance: bool = False,
    show_diagnostics: bool = False,
    tablefmt: str = "simple",
) -> str:
    """Format scale rows while keeping physical and sampling spaces explicit."""

    from tabulate import tabulate

    rows = list(rows)
    if not rows:
        return f"{title}\n  No scale parameters available."

    headers = []
    if show_index:
        headers.append("Index")
    headers.extend(["Kind", "Group", "Members", "State", "Sampling"])
    if show_variance:
        headers.append("Variance (v)")
    headers.append("Scale (s)")
    if show_posterior_uncertainty:
        headers.append("Post. SD(s)")
    headers.append("log10(s)")
    if show_posterior_uncertainty:
        headers.append("SD[log10(s)]")
    headers.append("Row mult. (1/s)")
    if show_diagnostics:
        headers.extend(["Qw", "Approx. red.Q"])

    table = []
    for row in rows:
        values = []
        if show_index:
            values.append("-" if row.get("index") is None else str(row["index"]))
        values.extend(
            [
                row.get("kind", ""),
                row.get("group", ""),
                ", ".join(str(value) for value in row.get("members", ())) or "-",
                row.get("state", ""),
                row.get("sampling_space", "-"),
            ]
        )
        if show_variance:
            values.append(_format_optional_float(row.get("variance")))
        values.append(_format_optional_float(row.get("scale")))
        if show_posterior_uncertainty:
            values.append(_format_optional_float(row.get("posterior_scale_std")))
        values.append(_format_optional_float(row.get("log10_scale")))
        if show_posterior_uncertainty:
            values.append(_format_optional_float(row.get("posterior_log10_std")))
        values.append(_format_optional_float(row.get("row_multiplier")))
        if show_diagnostics:
            values.extend(
                [
                    _format_optional_float(row.get("weighted_quadratic")),
                    _format_optional_float(row.get("reduced_weighted_misfit")),
                ]
            )
        table.append(values)
    return title + "\n" + tabulate(table, headers=headers, tablefmt=tablefmt)


def build_geometry_parameter_rows(
    resolved_updates: Sequence[Any],
    *,
    active_vector: Any,
    posterior_samples: Any = None,
) -> list[dict[str, Any]]:
    """Describe sampled geometry coordinates from resolved registry contracts.

    Shared sample slices are emitted once and list every consuming fault.  The
    function reads only the preflight records produced by Bayesian config; it
    does not inspect or rebuild current fault geometry.
    """

    active = np.asarray(active_vector, dtype=float).reshape(-1)
    samples = None
    if posterior_samples is not None:
        samples = np.asarray(posterior_samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError("posterior_samples must be a two-dimensional array")

    by_slice: dict[tuple[int, int], list[Any]] = {}
    for resolved in resolved_updates:
        start, end = (int(value) for value in resolved.sample_slice)
        by_slice.setdefault((start, end), []).append(resolved)

    rows = []
    for (start, end), owners in sorted(by_slice.items()):
        if start < 0 or end < start or end > active.size:
            raise ValueError(
                f"Geometry sample slice [{start}, {end}) is outside active vector"
            )
        if samples is not None and end > samples.shape[1]:
            raise ValueError(
                f"Geometry sample slice [{start}, {end}) is outside posterior samples"
            )

        methods = list(dict.fromkeys(owner.method_name for owner in owners))
        contracts = [owner.registry_contract or {} for owner in owners]
        item_specs = [
            tuple((contract.get("parameter_spec") or {}).get("items") or ())
            for contract in contracts
        ]
        shared_contract = item_specs[0] if all(spec == item_specs[0] for spec in item_specs) else ()
        count = end - start
        for local_index in range(count):
            item = {}
            repeated = False
            if len(shared_contract) == count:
                item = shared_contract[local_index]
            elif len(shared_contract) == 1:
                item = shared_contract[0]
                repeated = count > 1

            role = str(item.get("role") or "parameter")
            if repeated or (not item and count > 1):
                role += f"[{local_index}]"
            unit = item.get("unit")
            unit_from = item.get("unit_from")
            if unit is None and unit_from:
                owner = owners[0]
                unit = owner.method_kwargs.get(unit_from)
                if unit is None:
                    unit = (owner.registry_contract.get("kwargs") or {}).get(unit_from)
            unit = "-" if unit is None else str(unit)
            index = start + local_index
            rows.append(
                {
                    "index": index,
                    "faults": [owner.fault_name for owner in owners],
                    "method": ", ".join(methods),
                    "parameter": role,
                    "unit": unit,
                    "value": float(active[index]),
                    "posterior_std": (
                        None if samples is None else float(np.std(samples[:, index]))
                    ),
                }
            )
    return rows


def format_geometry_parameter_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str = "Bayesian geometry parameters",
    tablefmt: str = "simple",
) -> str:
    """Format geometry rows with explicit parameter roles and units."""

    from tabulate import tabulate

    rows = list(rows)
    if not rows:
        return f"{title}\n  No sampled geometry parameters."
    table = [
        [
            row.get("index", "-"),
            ", ".join(str(value) for value in row.get("faults", ())) or "-",
            row.get("method", "-"),
            row.get("parameter", "parameter"),
            row.get("unit", "-"),
            _format_optional_float(row.get("value")),
            _format_optional_float(row.get("posterior_std")),
        ]
        for row in rows
    ]
    return title + "\n" + tabulate(
        table,
        headers=["Index", "Fault(s)", "Method", "Parameter", "Unit", "Value", "Post. SD"],
        tablefmt=tablefmt,
    )


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "-"
    value = float(value)
    if not np.isfinite(value):
        return "-"
    return f"{value:.6g}"
