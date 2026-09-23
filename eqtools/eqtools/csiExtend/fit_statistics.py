"""Shared RMS/VR helpers for ECAT inversion diagnostics.

This module keeps the numerical definitions used by existing BLSE/Bayesian
reporting in one place.  It is intentionally read-only: helpers inspect
observed and synthetic vectors or solver matrices, but never rebuild Green's
functions or alter model parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data_vector_layout import gps_component_major_vector
from .hyperparameter_reporting import (
    build_scale_parameter_rows,
    format_scale_parameter_report,
)


def data_fit_vectors(data: Any, vertical: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return observed and synthetic vectors using existing CSI data fields.

    GPS uses the same component-major row order as CSI ``d/G/Cd``; scalar
    datasets use their native flat arrays; optical offsets concatenate east
    and north.
    """
    dtype = getattr(data, "dtype", None)
    if dtype == "insar":
        observed = data.vel
        synthetic = data.synth
    elif dtype == "gps":
        observed = gps_component_major_vector(
            data.vel_enu, vertical=vertical, name=f"{data.name} GPS observations"
        )
        synthetic = gps_component_major_vector(
            data.synth, vertical=vertical, name=f"{data.name} GPS synthetics"
        )
    elif dtype in ("opticorr", "optical"):
        observed = np.hstack((data.east, data.north))
        synthetic = np.hstack((data.east_synth, data.north_synth))
    elif dtype == "leveling":
        observed = data.vel
        synthetic = data.synth
    elif dtype == "crossfaultoffset":
        observed = data.data_vector
        synthetic = data.synth_vector
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

    return np.asarray(observed, dtype=float).reshape(-1), np.asarray(synthetic, dtype=float).reshape(-1)


def fit_metrics_from_vectors(observed: Any, synthetic: Any) -> dict[str, float | int]:
    """Compute RMS and VR with the established ECAT formula."""
    observed = np.asarray(observed, dtype=float).reshape(-1)
    synthetic = np.asarray(synthetic, dtype=float).reshape(-1)
    if observed.shape != synthetic.shape:
        raise ValueError(f"Observed/synthetic shape mismatch: {observed.shape} != {synthetic.shape}")
    residuals = synthetic - observed
    ss_res = float(np.sum(residuals ** 2))
    ss_obs = float(np.sum(observed ** 2))
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    vr = float((1.0 - ss_res / ss_obs) * 100.0) if ss_obs != 0.0 else 0.0
    return {
        "rms": rms,
        "vr": vr,
        "ss_res": ss_res,
        "ss_obs": ss_obs,
        "n_observations": int(residuals.size),
    }


def aggregate_dataset_fit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str = "global_observation_vector",
    model: str | None = None,
) -> dict[str, Any] | None:
    """Aggregate dataset rows into the exact concatenated-data fit metrics.

    The aggregation uses the sufficient statistics already computed for each
    dataset.  It is therefore numerically equivalent to concatenating every
    observation and prediction vector, without copying those potentially
    large arrays or rebuilding synthetics.  Dataset RMS/VR values are never
    averaged.

    Weighted diagnostics are published only when every dataset row contains a
    complete weighted quadratic.  A single effective standard deviation or
    reduced misfit is not inferred across heterogeneous sigma groups.
    """

    dataset_rows = [row for row in rows if row.get("scope") == "dataset"]
    if not dataset_rows:
        return None

    n_observations = int(
        np.sum([int(row["n_observations"]) for row in dataset_rows])
    )
    if n_observations <= 0:
        return None

    ss_res = float(np.sum([float(row["ss_res"]) for row in dataset_rows]))
    ss_obs = float(np.sum([float(row["ss_obs"]) for row in dataset_rows]))
    result = {
        "scope": str(scope),
        "model": model if model is not None else dataset_rows[0].get("model"),
        "dataset": None,
        "data_type": "assembled",
        "vertical": None,
        "poly": None,
        "sigma_group": None,
        "rms": float(np.sqrt(ss_res / n_observations)),
        "vr": float((1.0 - ss_res / ss_obs) * 100.0) if ss_obs != 0.0 else 0.0,
        "ss_res": ss_res,
        "ss_obs": ss_obs,
        "n_observations": n_observations,
    }

    weighted_values = [row.get("weighted_quadratic") for row in dataset_rows]
    if all(value is not None for value in weighted_values):
        weighted_quadratic = float(np.sum([float(value) for value in weighted_values]))
        result.update(
            {
                "sigma_scale": None,
                "base_marginal_std": None,
                "effective_marginal_std": None,
                "weighted_quadratic": weighted_quadratic,
                "weighted_rms": float(
                    np.sqrt(weighted_quadratic / n_observations)
                ),
                "weighted_effective_dof": None,
                "reduced_weighted_misfit": None,
            }
        )
    return result


def solver_fit_metrics(G: Any, m: Any, d: Any) -> dict[str, float | int]:
    """Compute RMS/VR from a solver matrix and vector, matching BLSE output."""
    m_array = np.asarray(m, dtype=float).reshape(-1)
    predicted = G.dot(m_array) if hasattr(G, "dot") else np.dot(G, m_array)
    return fit_metrics_from_vectors(d, predicted)


def weighted_fit_metrics_from_residual(
    residual: Any,
    covariance_metric: Any,
    *,
    sigma: float = 1.0,
    effective_dof: float | None = None,
) -> dict[str, float | int | None]:
    """Return covariance-aware residual diagnostics for one data set.

    The effective covariance is ``sigma**2 * C`` and the prepared metric
    represents ``C``.  Thus ``Qw = ||W r / sigma||**2`` is exactly
    ``r.T @ (sigma**2 C)**-1 @ r``.  ``weighted_rms`` is
    ``sqrt(Qw / n)`` and is dimensionless.  A reduced value is returned only
    when the caller supplies a positive effective degree of freedom; ECAT
    does not invent one for constrained BLSE or Bayesian models.
    """
    residual = np.asarray(residual, dtype=float).reshape(-1)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")

    whitened = covariance_metric.whiten(residual) / sigma
    weighted_quadratic = float(np.dot(whitened, whitened))
    n_observations = int(residual.size)
    weighted_rms = float(np.sqrt(weighted_quadratic / n_observations))

    reduced = None
    normalized_dof = None
    if effective_dof is not None:
        normalized_dof = float(effective_dof)
        if not np.isfinite(normalized_dof) or normalized_dof <= 0.0:
            raise ValueError("effective_dof must be finite and positive")
        reduced = float(weighted_quadratic / normalized_dof)

    base_std = getattr(covariance_metric, "marginal_rms_std", None)
    effective_std = None if base_std is None else float(base_std) * sigma
    return {
        "sigma_scale": sigma,
        "base_marginal_std": base_std,
        "effective_marginal_std": effective_std,
        "weighted_quadratic": weighted_quadratic,
        "weighted_rms": weighted_rms,
        "weighted_effective_dof": normalized_dof,
        "reduced_weighted_misfit": reduced,
    }


def format_fit_statistics_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
) -> str:
    """Format dataset rows followed by one optional global fit row."""
    from tabulate import tabulate

    dataset_rows = [row for row in rows if row.get("scope") == "dataset"]
    global_rows = [
        row
        for row in rows
        if row.get("scope") in {
            "global_observation_vector",
            "global_solver_vector",
        }
    ]
    display_rows = [*dataset_rows, *global_rows[:1]]
    title = "Data Fit Statistics"
    if model is not None:
        title += f" ({str(model).upper()} model)"
    if not display_rows:
        return f"{title}\n  No fit rows available."

    weighted = any(row.get("weighted_quadratic") is not None for row in display_rows)
    headers = ["Data", "Group", "N"]
    if weighted:
        headers.append("Eff. std")
    headers.extend(["RMS", "VR (%)"])
    if weighted:
        headers.extend(["Qw", "wRMS", "Approx. red.Q"])

    table = []
    for row in display_rows:
        is_global = row.get("scope") != "dataset"
        values = [
            "Global" if is_global else row.get("dataset", ""),
            row.get("sigma_group", "") or "-",
            int(row.get("n_observations", 0)),
        ]
        if weighted:
            values.append(_format_optional_float(row.get("effective_marginal_std")))
        values.extend(
            [
                f"{float(row['rms']):.6g}",
                f"{float(row['vr']):.4g}",
            ]
        )
        if weighted:
            values.extend(
                [
                    _format_optional_float(row.get("weighted_quadratic")),
                    _format_optional_float(row.get("weighted_rms")),
                    _format_optional_float(row.get("reduced_weighted_misfit")),
                ]
            )
        table.append(values)
    return title + "\n" + tabulate(table, headers=headers, tablefmt="simple")


def build_vce_component_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return rows for the variance components used by a VCE model.

    The caller-provided ``solved_*`` dictionaries are the sole numerical
    source.  Optional value-source metadata is report-only and never causes a
    configuration value or proposed next-iteration scale to be substituted.
    """
    rows = []
    diagnostics = result.get("component_diagnostics", {})
    sections = [
        (
            "data",
            "sigma",
            result.get("solved_sigma2_by_group", {}),
            result.get("sigma_groups", {}),
            result.get("sigma_update_by_group", {}),
            result.get("sigma_value_source_by_group", {}),
        )
    ]
    if not result.get("sigma_only", False):
        sections.append(
            (
                "smooth",
                "alpha",
                result.get("solved_alpha2_by_group", {}),
                result.get("smooth_groups", {}),
                result.get("smooth_update_by_group", {}),
                result.get("smooth_value_source_by_group", {}),
            )
        )
    for kind, symbol, values, groups, update_by_group, sources in sections:
        if not isinstance(values, Mapping):
            raise ValueError(f"solved {symbol} variances must be a group mapping")
        group_names = list(values)
        if set(groups) != set(group_names):
            raise ValueError(
                f"{symbol} group members do not match solved variance groups"
            )
        if update_by_group and set(update_by_group) != set(group_names):
            raise ValueError(
                f"{symbol} update states do not match solved variance groups"
            )
        if sources and set(sources) != set(group_names):
            raise ValueError(
                f"{symbol} value sources do not match solved variance groups"
            )
        # Older low-level callers did not publish update metadata.  Preserve
        # their established interpretation as estimated components, while new
        # solver results distinguish estimated and fixed groups explicitly.
        updates = np.asarray(
            [bool(update_by_group.get(name, True)) for name in group_names],
            dtype=bool,
        )
        next_update = 0
        sample_indices = []
        value_sources = {}
        for should_update in updates:
            sample_indices.append(next_update if should_update else -1)
            next_update += int(should_update)
        for group_name, should_update in zip(group_names, updates):
            # An estimated VCE value is not inherited from its initial value.
            # Provenance is meaningful only for components held fixed.
            fallback = "-" if should_update else "fixed input"
            value_sources[group_name] = str(sources.get(group_name, fallback))
        layout = {
            "group_names": group_names,
            "members_by_group": groups,
            "update_by_group": updates,
            "sample_index_by_group": np.asarray(sample_indices, dtype=int),
        }
        rows.extend(
            build_scale_parameter_rows(
                kind=symbol,
                layout=layout,
                active_scales_by_group={
                    group: float(np.sqrt(variance))
                    for group, variance in values.items()
                },
                update_state="estimated",
                variance_by_group=values,
                diagnostics_by_group=diagnostics.get(kind, {}),
                value_source_by_group=value_sources,
            )
        )
    return rows


def format_vce_component_report(result: Mapping[str, Any]) -> str:
    """Format final VCE variance components and group diagnostics.

    ``1/s`` is the row multiplier used by the augmented least-squares system;
    ``Qw`` and ``Approx. red.Q`` are evaluated for the final reported model
    and the explicit ``solved_*`` variance scales associated with it.  VCE has
    no Bayesian sample coordinate, so that column is intentionally omitted.
    """

    rows = build_vce_component_rows(result)

    status = "converged" if result.get("converged") else "not converged"
    title = f"VCE variance components ({status}, {result.get('iterations', 0)} iterations)"
    return format_scale_parameter_report(
        rows,
        title=title,
        show_index=False,
        show_value_source=True,
        show_sampling_space=False,
        show_posterior_uncertainty=False,
        show_variance=True,
        show_diagnostics=True,
        tablefmt="simple",
    )


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "-"
    value = float(value)
    if not np.isfinite(value):
        return "-"
    return f"{value:.6g}"


def fit_statistics_rows_to_dataframe(rows: Sequence[Mapping[str, Any]]):
    """Return a pandas DataFrame from fit-statistics rows."""
    import pandas as pd

    return pd.DataFrame([dict(row) for row in rows])


def format_fit_statistics_report(rows: Sequence[Mapping[str, Any]], *, model: str | None = None) -> str:
    """Format fit statistics as a compact human-readable report."""
    title = "Fit statistics report"
    if model is not None:
        title += f" ({str(model).upper()} model)"
    lines = [title, f"  rows: {len(rows)}"]
    for row in rows:
        scope = row.get("scope")
        label = (
            row.get("dataset")
            if scope == "dataset"
            else (
                "Global"
                if scope in {"global_observation_vector", "global_solver_vector"}
                else scope
            )
        )
        details = []
        if row.get("data_type") is not None:
            details.append(f"type={row.get('data_type')}")
        if row.get("poly") is not None:
            details.append(f"poly={row.get('poly')}")
        details.append(f"n={row.get('n_observations')}")
        suffix = f" ({', '.join(details)})" if details else ""
        metrics = [
            f"RMS={float(row['rms']):.6g}",
            f"VR={float(row['vr']):.4g}%",
        ]
        if row.get("weighted_quadratic") is not None:
            metrics.extend(
                [
                    f"Qw={float(row['weighted_quadratic']):.6g}",
                    f"wRMS={float(row['weighted_rms']):.6g}",
                ]
            )
            if row.get("reduced_weighted_misfit") is not None:
                metrics.append(
                    "Approx.red.Q="
                    f"{float(row['reduced_weighted_misfit']):.6g}"
                )
        lines.append(f"  - {label}{suffix}: " + ", ".join(metrics))
    return "\n".join(lines) + "\n"


def write_fit_statistics_report_files(
    rows: Sequence[Mapping[str, Any]],
    outdir,
    *,
    basename: str = "fit_statistics",
    formats: Sequence[str] = ("txt", "tsv"),
    model: str | None = None,
) -> dict[str, Path]:
    """Write fit-statistics text/table files and return paths by format."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    requested = {str(fmt).lower().lstrip(".") for fmt in formats}
    unsupported = requested - {"txt", "tsv"}
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported fit-statistics report format(s): {names}")

    written: dict[str, Path] = {}
    if "txt" in requested:
        path = outdir / f"{basename}.txt"
        path.write_text(format_fit_statistics_report(rows, model=model), encoding="utf-8")
        written["txt"] = path

    if "tsv" in requested:
        path = outdir / f"{basename}.tsv"
        dataframe = fit_statistics_rows_to_dataframe(rows)
        dataframe.to_csv(path, sep="\t", index=False)
        written["tsv"] = path

    return written
