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

from .parameter_layout_reporting import build_geometry_layout_rows


def _require_exact_group_keys(name, values, group_names):
    """Reject missing or extra group keys at a reporting boundary."""

    expected = set(group_names)
    actual = set(values)
    if actual == expected:
        return
    missing = [group for group in group_names if group not in actual]
    unknown = [group for group in values if group not in expected]
    details = []
    if missing:
        details.append("missing groups: " + ", ".join(map(str, missing)))
    if unknown:
        details.append("unknown groups: " + ", ".join(map(str, unknown)))
    raise ValueError(
        f"{name} does not match group_layout (" + "; ".join(details) + ")"
    )


def describe_bayesian_value_source(model: Any) -> str:
    """Describe which posterior coordinate was activated for prediction."""

    if isinstance(model, str):
        normalized = model.lower()
        labels = {
            "mean": "posterior mean coordinate",
            "median": "posterior median coordinate",
            "map": "posterior MAP sample",
            "max_prob": "posterior marginal-mode coordinate",
        }
        return labels.get(normalized, f"posterior representative '{model}'")
    if isinstance(model, (int, np.integer)):
        return f"posterior sample [{int(model)}]"
    return "explicit model vector"


def collapse_member_scales_to_groups(
    *,
    layout: Mapping[str, Any],
    active_scales_by_member: Mapping[str, float],
) -> dict[str, float]:
    """Collapse member scales only when every group member is exactly aligned."""

    result = {}
    group_names = list(layout.get("group_names", ()))
    members_by_group = layout.get("members_by_group", {})
    _require_exact_group_keys("members_by_group", members_by_group, group_names)
    expected_members = []
    for group_name in group_names:
        expected_members.extend(members_by_group[group_name])
    if len(expected_members) != len(set(expected_members)):
        raise ValueError("group_layout assigns a member to more than one scale group")
    _require_exact_group_keys(
        "active_scales_by_member", active_scales_by_member, expected_members
    )

    for group_name in group_names:
        members = list(members_by_group[group_name])
        if not members:
            raise ValueError(f"Scale group '{group_name}' has no members")
        missing = [name for name in members if name not in active_scales_by_member]
        if missing:
            raise ValueError(
                f"Scale group '{group_name}' is missing active member values: "
                + ", ".join(missing)
            )
        values = np.asarray(
            [active_scales_by_member[name] for name in members], dtype=float
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(
                f"Scale group '{group_name}' has a non-positive or non-finite active scale"
            )
        if not np.all(values == values[0]):
            raise ValueError(
                f"Scale group '{group_name}' members do not share one exact active scale"
            )
        result[str(group_name)] = float(values[0])
    return result


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
    value_source_by_group: Mapping[str, str] | None = None,
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
    value_source_by_group : mapping, optional
        Human-readable provenance for each active group value.  Provenance is
        descriptive only; it must be frozen by the solver or model-activation
        layer and is never used to reconstruct a numerical scale.

    Returns
    -------
    list of dict
        Serializable, read-only report records.  No input is mutated.
    """

    group_names = list(layout.get("group_names", ()))
    members_by_group = layout.get("members_by_group", {})
    _require_exact_group_keys("members_by_group", members_by_group, group_names)
    _require_exact_group_keys(
        "active_scales_by_group", active_scales_by_group, group_names
    )
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
    value_source_by_group = value_source_by_group or {}
    if value_source_by_group:
        _require_exact_group_keys(
            "value_source_by_group", value_source_by_group, group_names
        )
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
                "value_source": str(value_source_by_group.get(group_name, "-")),
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


def build_fixed_scale_parameter_rows(
    *,
    kind: str,
    layout: Mapping[str, Any],
    row_multipliers_by_group: Mapping[str, float],
    value_source_by_group: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Build BLSE rows from the exact group multipliers used by the solver.

    BLSE consumes row multipliers directly.  A positive finite multiplier
    ``w`` has the physical-scale interpretation ``s = 1 / w``.  A non-positive
    or non-finite direct multiplier is preserved verbatim but is not assigned a
    fictitious scale or logarithm.  The function never re-reads configuration
    or converts an input coordinate after the solve.
    """

    group_names = list(layout.get("group_names", ()))
    members_by_group = layout.get("members_by_group", {})
    _require_exact_group_keys("members_by_group", members_by_group, group_names)
    _require_exact_group_keys(
        "row_multipliers_by_group", row_multipliers_by_group, group_names
    )
    _require_exact_group_keys(
        "value_source_by_group", value_source_by_group, group_names
    )

    rows = []
    for group_name in group_names:
        multiplier = float(row_multipliers_by_group[group_name])
        has_scale = np.isfinite(multiplier) and multiplier > 0.0
        scale = float(1.0 / multiplier) if has_scale else None
        rows.append(
            {
                "index": None,
                "kind": str(kind),
                "group": str(group_name),
                "members": list(members_by_group.get(group_name, ())),
                "state": "fixed",
                "value_source": str(value_source_by_group.get(group_name, "-")),
                "sampling_space": "-",
                "scale": scale,
                "posterior_scale_std": None,
                "log10_scale": None if scale is None else float(np.log10(scale)),
                "posterior_log10_std": None,
                "row_multiplier": multiplier,
                "variance": None,
                "weighted_quadratic": None,
                "reduced_weighted_misfit": None,
            }
        )
    return rows


def format_scale_parameter_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    show_index: bool = True,
    show_value_source: bool = False,
    show_sampling_space: bool = True,
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
    headers.extend(["Kind", "Group", "Members", "State"])
    if show_value_source:
        headers.append("Value source")
    if show_sampling_space:
        headers.append("Sample coord.")
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
            ]
        )
        if show_value_source:
            values.append(row.get("value_source", "-"))
        if show_sampling_space:
            values.append(row.get("sampling_space", "-"))
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

    rows = []
    layout_rows, _notes = build_geometry_layout_rows(resolved_updates)
    for layout_row in layout_rows:
        start = int(layout_row["start"])
        end = int(layout_row["stop"])
        if start < 0 or end < start or end > active.size:
            raise ValueError(
                f"Geometry sample slice [{start}, {end}) is outside active vector"
            )
        if samples is not None and end > samples.shape[1]:
            raise ValueError(
                f"Geometry sample slice [{start}, {end}) is outside posterior samples"
            )

        details = layout_row.get("details", {})
        parameters = list(details.get("parameters", ()))
        units = list(layout_row.get("units", ()))
        faults = list(details.get("faults", ()))
        methods = list(details.get("methods", ()))
        if len(parameters) != end - start or len(units) != end - start:
            raise ValueError("Geometry layout row does not match its sample slice width")
        for local_index, (role, unit) in enumerate(zip(parameters, units)):
            index = start + local_index
            rows.append(
                {
                    "index": index,
                    "faults": faults,
                    "method": ", ".join(methods),
                    "parameter": str(role),
                    "unit": str(unit),
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
