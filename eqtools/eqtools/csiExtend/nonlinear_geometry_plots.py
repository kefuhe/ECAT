"""Plotting helpers for nonlinear geometry SMC results.

The public plotting entry points keep the old nonlinear SMC plotting
conventions where they are useful, especially ``plot_kde_matrix``.  The
implementation, however, selects sample columns from explicit parameter names
and groups instead of the old ``param_keys`` / ``param_index`` layout.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from eqtools.viztools import PlotStyle, finish_fig


_DEFAULT_STYLE = dict(
    preset="science",
    fontsize=9,
    legend_frame=False,
)

_SEABORN_STYLE_ALIASES = {"white", "ticks", "darkgrid", "whitegrid", "dark"}

_FAULT_PARAMETER_ORDER = [
    "lon",
    "lat",
    "depth",
    "length",
    "width",
    "strike",
    "dip",
    "slip",
    "magnitude",
    "rake",
    "strikeslip",
    "dipslip",
]

_DEFAULT_LABEL_MAP = {
    "lon": r"Lon",
    "lat": r"Lat",
    "depth": r"Z",
    "dip": r"$\delta$",
    "width": r"W",
    "length": r"L",
    "strike": r"Str",
    "slip": r"S",
    "magnitude": r"S",
    "rake": r"$\lambda$",
    "strikeslip": r"SS",
    "dipslip": r"DS",
}


@dataclass
class SMCPlotData:
    """Samples and parameter metadata used by nonlinear geometry plots."""

    samples: np.ndarray
    parameter_names: list[str]
    display_names: list[str]
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None
    postval: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None
    sample_stats: Optional[Dict[str, np.ndarray]] = None
    fault_parameter_stage_summary: Optional[Dict[str, Any]] = None
    groups: Optional[Dict[str, list[int]]] = None

    @classmethod
    def from_arrays(
        cls,
        samples,
        *,
        parameter_names=None,
        display_names=None,
        lower_bounds=None,
        upper_bounds=None,
        postval=None,
        beta=None,
        sample_stats=None,
        fault_parameter_stage_summary=None,
        groups=None,
    ):
        samples = _as_2d_samples(samples)
        n_parameters = samples.shape[1]
        names = _resolve_names(parameter_names, n_parameters)
        labels = _resolve_display_names(display_names, names)
        return cls(
            samples=samples,
            parameter_names=names,
            display_names=labels,
            lower_bounds=_optional_1d_array(lower_bounds, n_parameters, "lower_bounds"),
            upper_bounds=_optional_1d_array(upper_bounds, n_parameters, "upper_bounds"),
            postval=_optional_array(postval),
            beta=_optional_array(beta),
            sample_stats=_optional_sample_stats(sample_stats),
            fault_parameter_stage_summary=_optional_fault_parameter_stage_summary(
                fault_parameter_stage_summary
            ),
            groups=_resolve_groups(groups, names),
        )

    @classmethod
    def from_sampler(
        cls,
        sampler: Mapping[str, Any],
        *,
        parameter_names=None,
        display_names=None,
        lower_bounds=None,
        upper_bounds=None,
        groups=None,
    ):
        if "allsamples" not in sampler:
            raise ValueError("sampler must contain an 'allsamples' entry")
        return cls.from_arrays(
            sampler["allsamples"],
            parameter_names=parameter_names,
            display_names=display_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            postval=sampler.get("postval"),
            beta=sampler.get("beta"),
            sample_stats=sampler.get("sample_stats"),
            fault_parameter_stage_summary=sampler.get("fault_parameter_stage_summary"),
            groups=groups,
        )

    @classmethod
    def from_inversion(cls, inversion):
        if not hasattr(inversion, "sampler"):
            raise ValueError("inversion object has no sampler results")
        names = _call_or_attr(inversion, "parameter_names")
        labels = _call_or_attr(inversion, "parameter_display_names")
        lower_bounds = upper_bounds = None
        if hasattr(inversion, "parameter_bounds"):
            lower_bounds, upper_bounds = inversion.parameter_bounds()
        groups = _groups_from_parameter_specs(getattr(inversion, "parameter_specs", None))
        return cls.from_sampler(
            inversion.sampler,
            parameter_names=names,
            display_names=labels,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            groups=groups,
        )

    @classmethod
    def from_h5(cls, filename):
        try:
            import h5py
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("h5py is required to read SMC HDF5 files") from exc

        with h5py.File(filename, "r") as h5:
            if "allsamples" not in h5:
                raise ValueError(f"{filename}: missing 'allsamples' dataset")
            samples = h5["allsamples"][:]
            postval = h5["postval"][:] if "postval" in h5 else None
            beta = h5["beta"][:] if "beta" in h5 else None
            sample_stats = None
            if "sample_stats" in h5:
                sample_stats = {
                    key: h5["sample_stats"][key][:]
                    for key in h5["sample_stats"].keys()
                }
            fault_parameter_stage_summary = None
            if "fault_parameter_stage_summary" in h5:
                group = h5["fault_parameter_stage_summary"]
                fault_parameter_stage_summary = {
                    key: _read_h5_dataset(group[key])
                    for key in group.keys()
                }
            lower_bounds = h5["lower_bounds"][:] if "lower_bounds" in h5 else None
            upper_bounds = h5["upper_bounds"][:] if "upper_bounds" in h5 else None
            names = _json_attr(h5.attrs, "parameter_names_json")
            labels = _json_attr(h5.attrs, "parameter_display_names_json")
        return cls.from_arrays(
            samples,
            parameter_names=names,
            display_names=labels,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            postval=postval,
            beta=beta,
            sample_stats=sample_stats,
            fault_parameter_stage_summary=fault_parameter_stage_summary,
        )


def posterior_summary(
    source,
    *,
    parameters=None,
    groups=None,
    credible_interval=0.95,
):
    """Return posterior summary records for selected parameters."""
    data = coerce_plot_data(source)
    indices = _select_parameter_indices(data, parameters=parameters, groups=groups)
    alpha = (1.0 - credible_interval) / 2.0
    lo_pct = 100.0 * alpha
    hi_pct = 100.0 * (1.0 - alpha)

    rows = []
    for idx in indices:
        values = data.samples[:, idx]
        row = {
            "index": int(idx),
            "name": data.parameter_names[idx],
            "label": data.display_names[idx],
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "ci_lower": float(np.percentile(values, lo_pct)),
            "ci_upper": float(np.percentile(values, hi_pct)),
        }
        if data.lower_bounds is not None and data.upper_bounds is not None:
            row["lower_bound"] = float(data.lower_bounds[idx])
            row["upper_bound"] = float(data.upper_bounds[idx])
            row["boundary_fraction"] = _boundary_fraction(
                values,
                data.lower_bounds[idx],
                data.upper_bounds[idx],
            )
        rows.append(row)
    return rows


def plot_kde_matrix(
    source,
    *,
    figsize=None,
    save=False,
    filename="kde_matrix.png",
    save_path=None,
    show=True,
    style="white",
    fill=True,
    scatter=False,
    scatter_size=15,
    plot_sigmas=False,
    plot_faults=True,
    faults=None,
    plot_data_corrections=False,
    parameters=None,
    groups=None,
    max_parameters=None,
    axis_labels=None,
    label_map=None,
    wspace=None,
    hspace=None,
    center_lon_lat=False,
    xtick_rotation=None,
    ytick_rotation=None,
    lonlat_decimal=3,
    use_sigma_alias=True,
    tick_fontsize=None,
    label_fontsize=None,
    show_minor_ticks=False,
    tick_direction="in",
    major_tick_length=3,
    minor_tick_length=1.5,
    tick_width=0.5,
    bins=40,
    kde_bw_method="scott",
    dpi=600,
    screen_dpi=200,
    **style_kwargs,
):
    """Plot a publication-style KDE matrix for selected SMC parameters.

    The argument names mirror the old nonlinear SMC ``plot_kde_matrix`` where
    possible.  New nonlinear-geometry parameters are selected by canonical
    names/groups, so data-correction coefficients can be plotted without
    reintroducing the old index bookkeeping.
    """
    data = coerce_plot_data(source)
    selected = _select_kde_indices(
        source,
        data,
        parameters=parameters,
        groups=groups,
        plot_faults=plot_faults,
        faults=faults,
        plot_sigmas=plot_sigmas,
        plot_data_corrections=plot_data_corrections,
        max_parameters=max_parameters,
    )
    selected = _drop_constant_indices(data, selected)
    if not selected:
        raise ValueError("No non-constant parameters selected")

    labels = _labels_for_indices(
        source,
        data,
        selected,
        axis_labels=axis_labels,
        label_map=label_map,
        use_sigma_alias=use_sigma_alias,
    )
    values = data.samples[:, selected].copy()
    local_names = [_parse_parameter_name(data.parameter_names[idx])[2] for idx in selected]
    if center_lon_lat:
        _center_lon_lat_columns(values, local_names)
    parameter_limits = None

    n = len(selected)
    figure_kwargs, style_kwargs = _split_figsize_for_matplotlib(figsize, style_kwargs)
    if not figure_kwargs and figsize is None:
        figure_kwargs["figsize"] = (max(2.4, 2.35 * n), max(2.4, 2.35 * n))
    resolved_save_path = _resolve_save_path(
        save=save,
        filename=filename,
        save_path=save_path,
    )

    skw = _merge_style_kwargs(
        style=style,
        tick_fontsize=tick_fontsize,
        save_path=resolved_save_path,
        user_kwargs=style_kwargs,
    )
    with PlotStyle(**skw):
        fig, axes = _build_pairgrid_kde_matrix(
            values,
            labels,
            parameter_limits=parameter_limits,
            figure_kwargs=figure_kwargs,
            style=style,
            fill=fill,
            scatter=scatter,
            scatter_size=scatter_size,
            bins=bins,
            kde_bw_method=kde_bw_method,
        )

        _apply_kde_matrix_formatting(
            axes,
            local_names=local_names,
            xtick_rotation=xtick_rotation,
            ytick_rotation=ytick_rotation,
            lonlat_decimal=lonlat_decimal,
            tick_fontsize=tick_fontsize,
            label_fontsize=label_fontsize,
            show_minor_ticks=show_minor_ticks,
            tick_direction=tick_direction,
            major_tick_length=major_tick_length,
            minor_tick_length=minor_tick_length,
            tick_width=tick_width,
        )
        _tight_layout(fig)
        if wspace is not None or hspace is not None:
            fig.subplots_adjust(wspace=wspace, hspace=hspace)
        _save_show(fig, save_path=resolved_save_path, show=show, dpi=dpi, screen_dpi=screen_dpi)
    return fig, axes


def plot_parameter_marginals(
    source,
    *,
    parameters=None,
    groups=None,
    max_parameters=12,
    bins=40,
    credible_interval=0.95,
    diagonal="hist",
    show_bounds=True,
    figsize=None,
    save_path=None,
    show=False,
    style="science",
    dpi=600,
    **style_kwargs,
):
    """Plot one-dimensional posterior marginals."""
    data = coerce_plot_data(source)
    indices = _select_parameter_indices(
        data,
        parameters=parameters,
        groups=groups,
        max_parameters=max_parameters,
    )
    n = len(indices)
    if n == 0:
        raise ValueError("No parameters selected")
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 2.8 * nrows)
    figure_kwargs, style_kwargs = _split_figsize_for_matplotlib(figsize, style_kwargs)
    skw = _merge_style_kwargs(
        style=style,
        save_path=save_path,
        user_kwargs=style_kwargs,
    )
    with PlotStyle(**skw):
        plt = _pyplot()
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, **figure_kwargs)
        axes_flat = axes.reshape(-1)
        summaries = posterior_summary(
            data,
            parameters=[data.parameter_names[i] for i in indices],
            credible_interval=credible_interval,
        )

        for ax, idx, summary in zip(axes_flat, indices, summaries):
            values = data.samples[:, idx]
            _plot_1d_density(ax, values, bins=bins, mode=diagonal, fill=True)
            ax.axvline(summary["median"], color="0.15", lw=1.2)
            ax.axvspan(summary["ci_lower"], summary["ci_upper"], color="0.2", alpha=0.12)
            if show_bounds and data.lower_bounds is not None and data.upper_bounds is not None:
                ax.axvline(data.lower_bounds[idx], color="tab:red", lw=0.8, ls="--")
                ax.axvline(data.upper_bounds[idx], color="tab:red", lw=0.8, ls="--")
            ax.set_title(data.display_names[idx])
            ax.set_ylabel("Density")

        for ax in axes_flat[n:]:
            ax.set_visible(False)
        _tight_layout(fig)
        _save_show(fig, save_path=save_path, show=show, dpi=dpi)
    return fig, axes_flat[:n], summaries


def plot_parameter_pairs(
    source,
    *,
    parameters=None,
    groups=None,
    max_parameters=8,
    bins=35,
    diagonal="hist",
    scatter_size=8,
    alpha=0.35,
    figsize=None,
    save_path=None,
    show=False,
    style="science",
    dpi=600,
    **style_kwargs,
):
    """Plot a lightweight corner-style matrix for selected parameters."""
    data = coerce_plot_data(source)
    indices = _select_parameter_indices(
        data,
        parameters=parameters,
        groups=groups,
        max_parameters=max_parameters,
    )
    if not indices:
        raise ValueError("No parameters selected")
    n = len(indices)
    if figsize is None:
        figsize = (2.4 * n, 2.4 * n)
    figure_kwargs, style_kwargs = _split_figsize_for_matplotlib(figsize, style_kwargs)
    skw = _merge_style_kwargs(
        style=style,
        save_path=save_path,
        user_kwargs=style_kwargs,
    )
    with PlotStyle(**skw):
        plt = _pyplot()
        fig, axes = plt.subplots(n, n, squeeze=False, **figure_kwargs)

        for row, i in enumerate(indices):
            for col, j in enumerate(indices):
                ax = axes[row, col]
                if row == col:
                    _plot_1d_density(ax, data.samples[:, i], bins=bins, mode=diagonal, fill=True)
                elif row > col:
                    ax.scatter(
                        data.samples[:, j],
                        data.samples[:, i],
                        s=scatter_size,
                        alpha=alpha,
                        linewidths=0,
                        color="tab:blue",
                    )
                else:
                    ax.set_visible(False)
                    continue
                if row == n - 1:
                    ax.set_xlabel(data.display_names[j])
                else:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel(data.display_names[i])
                else:
                    ax.set_yticklabels([])
        _tight_layout(fig)
        _save_show(fig, save_path=save_path, show=show, dpi=dpi)
    return fig, axes


def plot_sample_traces(
    source,
    *,
    parameters=None,
    groups=None,
    max_parameters=12,
    thin=1,
    figsize=None,
    save_path=None,
    show=False,
    style="science",
    dpi=600,
    **style_kwargs,
):
    """Plot posterior samples in saved order for quick stability inspection."""
    data = coerce_plot_data(source)
    indices = _select_parameter_indices(
        data,
        parameters=parameters,
        groups=groups,
        max_parameters=max_parameters,
    )
    if not indices:
        raise ValueError("No parameters selected")
    if thin < 1:
        raise ValueError("thin must be >= 1")
    n = len(indices)
    if figsize is None:
        figsize = (8.0, max(2.0, 1.8 * n))
    figure_kwargs, style_kwargs = _split_figsize_for_matplotlib(figsize, style_kwargs)
    skw = _merge_style_kwargs(
        style=style,
        save_path=save_path,
        user_kwargs=style_kwargs,
    )
    with PlotStyle(**skw):
        plt = _pyplot()
        fig, axes = plt.subplots(n, 1, sharex=True, squeeze=False, **figure_kwargs)
        axes_flat = axes.reshape(-1)
        x = np.arange(data.samples.shape[0])[::thin]
        for ax, idx in zip(axes_flat, indices):
            ax.plot(x, data.samples[::thin, idx], lw=0.8, color="tab:blue")
            ax.set_ylabel(data.display_names[idx])
        axes_flat[-1].set_xlabel("Sample index")
        _tight_layout(fig)
        _save_show(fig, save_path=save_path, show=show, dpi=dpi)
    return fig, axes_flat


def plot_parameter_intervals(
    source,
    *,
    parameters=None,
    groups=None,
    credible_interval=0.95,
    max_parameters=40,
    sort_by="index",
    show_bounds=True,
    figsize=None,
    save_path=None,
    show=False,
    style="science",
    dpi=600,
    **style_kwargs,
):
    """Plot posterior median and credible intervals for selected parameters."""
    data = coerce_plot_data(source)
    rows = posterior_summary(
        data,
        parameters=parameters,
        groups=groups,
        credible_interval=credible_interval,
    )
    if max_parameters is not None:
        rows = rows[:max_parameters]
    rows = _sort_summary_rows(rows, sort_by)
    if not rows:
        raise ValueError("No parameters selected")
    n = len(rows)
    if figsize is None:
        figsize = (8.0, max(3.0, 0.34 * n + 1.4))
    figure_kwargs, style_kwargs = _split_figsize_for_matplotlib(figsize, style_kwargs)
    skw = _merge_style_kwargs(
        style=style,
        save_path=save_path,
        user_kwargs=style_kwargs,
    )
    with PlotStyle(**skw):
        plt = _pyplot()
        fig, ax = plt.subplots(**figure_kwargs)
        y = np.arange(n)
        med = np.array([row["median"] for row in rows])
        lo = np.array([row["ci_lower"] for row in rows])
        hi = np.array([row["ci_upper"] for row in rows])
        ax.errorbar(
            med,
            y,
            xerr=[med - lo, hi - med],
            fmt="o",
            markersize=4,
            color="tab:blue",
            ecolor="0.35",
            elinewidth=1.0,
            capsize=2,
        )
        if show_bounds and data.lower_bounds is not None and data.upper_bounds is not None:
            name_to_index = {name: i for i, name in enumerate(data.parameter_names)}
            for ypos, row in enumerate(rows):
                idx = name_to_index[row["name"]]
                ax.plot(
                    [data.lower_bounds[idx], data.upper_bounds[idx]],
                    [ypos, ypos],
                    color="tab:red",
                    alpha=0.18,
                    lw=4,
                    solid_capstyle="round",
                )
        ax.set_yticks(y)
        ax.set_yticklabels([row["label"] for row in rows])
        ax.invert_yaxis()
        ax.set_xlabel("Parameter value")
        _tight_layout(fig)
        _save_show(fig, save_path=save_path, show=show, dpi=dpi)
    return fig, ax, rows


def plot_smc_progress(
    source,
    *,
    figsize=(8.0, 4.5),
    save_path=None,
    show=False,
    style="science",
    dpi=600,
    **style_kwargs,
):
    """Plot saved SMC beta, posterior values, and process stats when available."""
    data = coerce_plot_data(source)
    stat_rows = _smc_progress_stat_rows(data.sample_stats)
    if data.beta is None and data.postval is None and not stat_rows:
        raise ValueError("No beta, postval, or sample_stats arrays are available")
    figure_kwargs, style_kwargs = _split_figsize_for_matplotlib(figsize, style_kwargs)
    skw = _merge_style_kwargs(
        style=style,
        save_path=save_path,
        user_kwargs=style_kwargs,
    )
    with PlotStyle(**skw):
        plt = _pyplot()
        nrows = int(data.beta is not None) + int(data.postval is not None) + len(stat_rows)
        fig, axes = plt.subplots(nrows, 1, squeeze=False, **figure_kwargs)
        axes_flat = axes.reshape(-1)
        row = 0
        if data.beta is not None:
            beta = np.asarray(data.beta).reshape(-1)
            axes_flat[row].plot(np.arange(beta.size), beta, marker="o", ms=3, lw=1.0)
            axes_flat[row].set_ylabel("beta")
            row += 1
        if data.postval is not None:
            post = np.asarray(data.postval).reshape(-1)
            axes_flat[row].plot(np.arange(post.size), post, lw=0.8, color="tab:green")
            axes_flat[row].set_ylabel("postval")
            row += 1
        for ylabel, values in stat_rows:
            arr = np.asarray(values, dtype=float).reshape(-1)
            axes_flat[row].plot(np.arange(arr.size), arr, marker="o", ms=3, lw=1.0)
            axes_flat[row].set_ylabel(ylabel)
            row += 1
        axes_flat[-1].set_xlabel("Saved index / SMC stage")
        _tight_layout(fig)
        _save_show(fig, save_path=save_path, show=show, dpi=dpi)
    return fig, axes_flat


def plot_fault_parameter_trends(
    source,
    *,
    parameters=None,
    max_parameters=12,
    details=False,
    credible_interval_label="95% CI",
    figsize=None,
    save_path=None,
    show=False,
    style="science",
    dpi=600,
    **style_kwargs,
):
    """Plot SMC-stage evolution of fault-parameter medians and intervals."""
    data = coerce_plot_data(source)
    summary = data.fault_parameter_stage_summary
    if not summary:
        raise ValueError("No fault_parameter_stage_summary is available")

    stages = np.asarray(summary.get("stage"), dtype=float).reshape(-1)
    median = _stage_summary_matrix(summary, "median")
    ci_lower = _stage_summary_matrix(summary, "ci_lower")
    ci_upper = _stage_summary_matrix(summary, "ci_upper")
    names = [str(name) for name in summary.get("parameter_names", [])]
    labels = [str(label) for label in summary.get("display_names", names)]
    if not names:
        names = [f"fault_param_{i}" for i in range(median.shape[1])]
    if len(labels) < len(names):
        labels.extend(names[len(labels):])

    selected = _select_stage_summary_indices(
        names,
        labels,
        parameters=parameters,
        max_parameters=max_parameters,
    )
    if not selected:
        raise ValueError("No fault parameters selected")

    detail_rows = _smc_progress_stat_rows(data.sample_stats) if details else []
    nrows = len(selected) + len(detail_rows)
    if figsize is None:
        figsize = (8.0, max(2.4, 1.45 * nrows + 0.7))
    figure_kwargs, style_kwargs = _split_figsize_for_matplotlib(figsize, style_kwargs)
    skw = _merge_style_kwargs(
        style=style,
        save_path=save_path,
        user_kwargs=style_kwargs,
    )
    x = stages if stages.size == median.shape[0] else np.arange(median.shape[0])
    with PlotStyle(**skw):
        plt = _pyplot()
        fig, axes = plt.subplots(nrows, 1, sharex=False, squeeze=False, **figure_kwargs)
        axes_flat = axes.reshape(-1)
        row = 0
        for idx in selected:
            ax = axes_flat[row]
            ax.plot(x, median[:, idx], marker="o", ms=3, lw=1.1, color="tab:blue")
            if ci_lower.shape == median.shape and ci_upper.shape == median.shape:
                ax.fill_between(
                    x,
                    ci_lower[:, idx],
                    ci_upper[:, idx],
                    color="tab:blue",
                    alpha=0.16,
                    linewidth=0,
                    label=credible_interval_label,
                )
            ax.set_ylabel(labels[idx] if idx < len(labels) else names[idx])
            if row == 0 and ci_lower.shape == median.shape:
                ax.legend(loc="best", fontsize="small", frameon=False)
            row += 1

        for ylabel, values in detail_rows:
            arr = np.asarray(values, dtype=float).reshape(-1)
            stat_stage = None
            if data.sample_stats and "stage" in data.sample_stats:
                stat_stage = np.asarray(data.sample_stats["stage"], dtype=float).reshape(-1)
            sx = stat_stage if stat_stage is not None and stat_stage.size == arr.size else np.arange(arr.size)
            axes_flat[row].plot(sx, arr, marker="o", ms=3, lw=1.0, color="0.25")
            axes_flat[row].set_ylabel(ylabel)
            row += 1

        axes_flat[-1].set_xlabel("SMC stage")
        _tight_layout(fig)
        _save_show(fig, save_path=save_path, show=show, dpi=dpi)
    return fig, axes_flat


def coerce_plot_data(source) -> SMCPlotData:
    """Convert supported inputs to ``SMCPlotData``."""
    if isinstance(source, SMCPlotData):
        return source
    if isinstance(source, (str, Path)):
        return SMCPlotData.from_h5(source)
    if isinstance(source, Mapping):
        return SMCPlotData.from_sampler(source)
    if hasattr(source, "sampler"):
        return SMCPlotData.from_inversion(source)
    return SMCPlotData.from_arrays(source)


def _as_2d_samples(samples):
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError("samples must be a 2D array with shape (n_samples, n_parameters)")
    return arr


def _resolve_names(parameter_names, n_parameters):
    if parameter_names is None:
        return [f"param_{i}" for i in range(n_parameters)]
    names = list(parameter_names)
    if len(names) != n_parameters:
        raise ValueError("parameter_names length must match sample parameter count")
    return [str(name) for name in names]


def _resolve_display_names(display_names, names):
    if display_names is None:
        return list(names)
    labels = list(display_names)
    if len(labels) != len(names):
        raise ValueError("display_names length must match parameter_names")
    return [str(label) for label in labels]


def _optional_1d_array(values, n_parameters, name):
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != n_parameters:
        raise ValueError(f"{name} length must match sample parameter count")
    return arr


def _optional_array(values):
    if values is None:
        return None
    return np.asarray(values, dtype=float)


def _optional_sample_stats(values):
    if not values:
        return None
    return {
        str(key): np.asarray(value, dtype=float)
        for key, value in values.items()
    }


def _optional_fault_parameter_stage_summary(values):
    if not values:
        return None
    summary: Dict[str, Any] = {}
    for key, value in values.items():
        key = str(key)
        if key in {"parameter_names", "display_names"}:
            summary[key] = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in np.asarray(value).reshape(-1)
            ]
        else:
            summary[key] = np.asarray(value, dtype=float)
    return summary


def _smc_progress_stat_rows(sample_stats):
    if not sample_stats:
        return []
    rows = []
    for key, label in [
        ("normalized_ess", "norm. ESS"),
        ("acceptance_rate_mean", "accept. mean"),
        ("unique_ancestor_fraction", "ancestor frac."),
    ]:
        if key in sample_stats:
            arr = np.asarray(sample_stats[key], dtype=float).reshape(-1)
            if arr.size:
                rows.append((label, arr))
    return rows


def _stage_summary_matrix(summary, key):
    if key not in summary:
        raise ValueError(f"fault_parameter_stage_summary is missing '{key}'")
    arr = np.asarray(summary[key], dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    if arr.ndim != 2:
        raise ValueError(f"fault_parameter_stage_summary['{key}'] must be 2D")
    return arr


def _select_stage_summary_indices(names, labels, *, parameters=None, max_parameters=None):
    if parameters is None:
        indices = list(range(len(names)))
    else:
        lookup = {name: i for i, name in enumerate(names)}
        lookup.update({label: i for i, label in enumerate(labels)})
        indices = []
        for item in parameters:
            if isinstance(item, int):
                idx = int(item)
                if idx < 0 or idx >= len(names):
                    raise ValueError(f"Fault parameter index out of range: {item}")
                indices.append(idx)
                continue
            text = str(item)
            if text not in lookup:
                raise ValueError(f"Unknown fault parameter: {item}")
            indices.append(lookup[text])
    indices = _unique_indices(indices)
    if max_parameters is not None:
        indices = indices[:max_parameters]
    return indices


def _read_h5_dataset(dataset):
    value = dataset[()]
    arr = np.asarray(value)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8").tolist()
    if arr.dtype.kind == "O":
        return [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in arr.reshape(-1)
        ]
    return value


def _call_or_attr(obj, name):
    value = getattr(obj, name, None)
    if callable(value):
        return value()
    return value


def _json_attr(attrs, key):
    if key not in attrs:
        return None
    value = attrs[key]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def _groups_from_parameter_specs(parameter_specs):
    if not parameter_specs:
        return None
    groups: Dict[str, list[int]] = {}
    for spec in parameter_specs:
        groups.setdefault(spec.group, []).append(spec.index)
    return groups


def _resolve_groups(groups, names):
    if groups is not None:
        return {str(key): [int(i) for i in value] for key, value in groups.items()}
    inferred: Dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        group = name.split(".", 1)[0] if "." in name else "parameters"
        inferred.setdefault(group, []).append(idx)
    return inferred


def _select_kde_indices(
    source,
    data,
    *,
    parameters=None,
    groups=None,
    plot_faults=True,
    faults=None,
    plot_sigmas=False,
    plot_data_corrections=False,
    max_parameters=None,
):
    if parameters is not None or groups is not None:
        return _select_parameter_indices(
            data,
            parameters=parameters,
            groups=groups,
            max_parameters=max_parameters,
        )

    indices = []
    if plot_faults:
        indices.extend(_fault_indices(source, data, faults=faults))
    if plot_sigmas:
        indices.extend(data.groups.get("sigmas", []))
    if plot_data_corrections:
        indices.extend(data.groups.get("data_corrections", []))

    unique = _unique_indices(indices)
    if max_parameters is not None:
        unique = unique[:max_parameters]
    return unique


def _select_parameter_indices(
    data: SMCPlotData,
    *,
    parameters=None,
    groups=None,
    max_parameters=None,
):
    if parameters is not None:
        indices = [_parameter_index(data, item) for item in parameters]
    elif groups is not None:
        indices = []
        group_list = groups if isinstance(groups, (list, tuple, set)) else [groups]
        for group in group_list:
            if data.groups is None or group not in data.groups:
                raise ValueError(f"Unknown parameter group: {group}")
            indices.extend(data.groups[group])
    else:
        indices = list(range(data.samples.shape[1]))
    unique = _unique_indices(indices)
    if max_parameters is not None:
        unique = unique[:max_parameters]
    return unique


def _unique_indices(indices):
    seen = set()
    unique = []
    for idx in indices:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


def _parameter_index(data, item):
    if isinstance(item, int):
        if item < 0 or item >= data.samples.shape[1]:
            raise ValueError(f"Parameter index out of range: {item}")
        return int(item)
    item = str(item)
    if item in data.parameter_names:
        return data.parameter_names.index(item)
    if item in data.display_names:
        return data.display_names.index(item)
    raise ValueError(f"Unknown parameter: {item}")


def _fault_indices(source, data, *, faults=None):
    fault_group = data.groups.get("faults", [])
    if not fault_group:
        return []

    target_faults = _normalize_faults_arg(source, data, faults)
    selected = []
    for fault_name in target_faults:
        fault_indices = [
            idx
            for idx in fault_group
            if _parse_parameter_name(data.parameter_names[idx])[1] == fault_name
        ]
        selected.extend(sorted(fault_indices, key=lambda idx: _fault_sort_key(data, idx)))

    if selected:
        return selected
    return sorted(fault_group, key=lambda idx: _fault_sort_key(data, idx))


def _normalize_faults_arg(source, data, faults):
    if faults is None:
        if hasattr(source, "faultnames"):
            return list(source.faultnames)
        names = []
        for idx in data.groups.get("faults", []):
            owner = _parse_parameter_name(data.parameter_names[idx])[1]
            if owner and owner not in names:
                names.append(owner)
        return names
    if isinstance(faults, str):
        return [faults]
    return list(faults)


def _fault_sort_key(data, idx):
    group, owner, local_name = _parse_parameter_name(data.parameter_names[idx])
    try:
        priority = _FAULT_PARAMETER_ORDER.index(local_name)
    except ValueError:
        priority = len(_FAULT_PARAMETER_ORDER) + 1
    return owner or "", priority, idx


def _parse_parameter_name(name):
    parts = str(name).split(".")
    if len(parts) >= 3:
        return parts[0], parts[1], ".".join(parts[2:])
    if len(parts) == 2:
        return parts[0], None, parts[1]
    text = str(name)
    if text.startswith("fault_"):
        tokens = text.split("_")
        if len(tokens) >= 3:
            return "faults", "_".join(tokens[:2]), "_".join(tokens[2:])
    return "parameters", None, text


def _drop_constant_indices(data, indices):
    kept = []
    for idx in indices:
        values = data.samples[:, idx]
        if np.nanvar(values) != 0:
            kept.append(idx)
    return kept


def _labels_for_indices(
    source,
    data,
    indices,
    *,
    axis_labels=None,
    label_map=None,
    use_sigma_alias=True,
):
    if axis_labels is not None:
        labels = list(axis_labels)
        if len(labels) != len(indices):
            raise ValueError("axis_labels length must match the number of plotted parameters")
        return [str(label) for label in labels]

    merged_map = dict(_DEFAULT_LABEL_MAP)
    if label_map:
        merged_map.update(label_map)

    labels = []
    n_faults = len(_normalize_faults_arg(source, data, None))
    data_correction_owners = set()
    for idx in indices:
        group, owner, _ = _parse_parameter_name(data.parameter_names[idx])
        if group == "data_corrections" and owner:
            data_correction_owners.add(owner)
    for idx in indices:
        name = data.parameter_names[idx]
        display = data.display_names[idx]
        group, owner, local_name = _parse_parameter_name(name)

        for key in (name, display, local_name):
            if key in merged_map:
                mapped = merged_map[key]
                break
        else:
            mapped = None

        if group == "faults":
            base = mapped or local_name.capitalize()
            if owner and n_faults > 1:
                labels.append(_format_fault_label(base, _fault_alias(source, owner)))
            else:
                labels.append(_as_math_label(base))
        elif group == "sigmas" and use_sigma_alias:
            labels.append(_format_sigma_label(source, local_name, display))
        elif group == "data_corrections":
            labels.append(
                _format_data_correction_label(
                    mapped=mapped,
                    display=display,
                    canonical_name=name,
                    owner=owner,
                    local_name=local_name,
                    include_owner=len(data_correction_owners) > 1,
                )
            )
        else:
            labels.append(mapped or display or name)
    return labels


def _fault_alias(source, fault_name):
    alias_map = getattr(source, "fault_alias_map", None)
    if isinstance(alias_map, Mapping) and fault_name in alias_map:
        return alias_map[fault_name]
    if str(fault_name).startswith("fault_"):
        return str(fault_name).replace("fault_", "F")
    return str(fault_name)


def _format_fault_label(symbol, subscript):
    clean = str(symbol).replace("$", "")
    return fr"${clean}_{{{subscript}}}$"


def _as_math_label(label):
    label = str(label)
    if label.startswith("$") and label.endswith("$"):
        return label
    return fr"${label}$"


def _format_sigma_label(source, local_name, display):
    sigmas = getattr(source, "sigmas", None)
    log_scaled = isinstance(sigmas, Mapping) and bool(sigmas.get("log_scaled", False))
    sub = str(local_name).replace("sigma_", "")
    if not sub:
        sub = str(display).replace("sigma_", "")
    if log_scaled:
        return fr"$\log(\sigma_{{{sub}}})$"
    return fr"$\sigma_{{{sub}}}$"


def _format_data_correction_label(
    *,
    mapped,
    display,
    canonical_name,
    owner,
    local_name,
    include_owner,
):
    if mapped:
        return str(mapped)
    display = str(display) if display is not None else ""
    if display and display != canonical_name and not display.startswith("data_corrections."):
        return display
    if include_owner and owner:
        return f"{owner}.{local_name}"
    return str(local_name)


def _center_lon_lat_columns(values, local_names):
    for col, local_name in enumerate(local_names):
        if str(local_name).lower() in {"lon", "lat"}:
            values[:, col] -= np.nanmean(values[:, col])


def _plot_1d_density(ax, values, *, bins, mode, fill=True, bw_method=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    if mode == "kde":
        try:
            from scipy.stats import gaussian_kde

            if np.allclose(values, values[0]):
                raise ValueError("constant samples")
            grid = np.linspace(np.min(values), np.max(values), 200)
            density = gaussian_kde(values, bw_method=bw_method)(grid)
            ax.plot(grid, density, color="tab:blue", lw=1.2)
            if fill:
                ax.fill_between(grid, 0.0, density, color="tab:blue", alpha=0.18)
            return
        except Exception:
            pass
    ax.hist(values, bins=bins, density=True, color="tab:blue", alpha=0.55, edgecolor="none")


def _plot_2d_density(
    ax,
    x,
    y,
    *,
    fill=True,
    scatter_size=15,
    bw_method="scott",
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 5:
        ax.scatter(x, y, s=scatter_size, alpha=0.35, linewidths=0, color="tab:blue")
        return
    try:
        import seaborn as sns

        kwargs = dict(
            x=x,
            y=y,
            ax=ax,
            fill=fill,
            levels=6,
            thresh=0.05,
            bw_method=bw_method,
            warn_singular=False,
        )
        if fill:
            kwargs["cmap"] = "Blues"
        else:
            kwargs["color"] = "tab:blue"
        sns.kdeplot(**kwargs)
        return
    except Exception:
        pass

    try:
        from scipy.stats import gaussian_kde

        if np.nanvar(x) == 0 or np.nanvar(y) == 0:
            raise ValueError("constant samples")
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, bw_method=bw_method)
        xmin, xmax = _padded_limits(x)
        ymin, ymax = _padded_limits(y)
        xx, yy = np.mgrid[xmin:xmax:70j, ymin:ymax:70j]
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        zmax = float(np.nanmax(zz))
        if not np.isfinite(zmax) or zmax <= 0:
            raise ValueError("invalid kde density")
        levels = np.linspace(zmax * 0.05, zmax, 6)
        if fill:
            ax.contourf(xx, yy, zz, levels=levels, cmap="Blues", alpha=0.72)
        ax.contour(xx, yy, zz, levels=levels, colors="tab:blue", linewidths=0.7)
    except Exception:
        ax.scatter(x, y, s=scatter_size, alpha=0.35, linewidths=0, color="tab:blue")


def _build_pairgrid_kde_matrix(
    values,
    labels,
    *,
    parameter_limits,
    figure_kwargs,
    style,
    fill=True,
    scatter=False,
    scatter_size=15,
    bins=40,
    kde_bw_method="scott",
):
    """Build a KDE matrix with the old Seaborn PairGrid visual path.

    The public API still uses the new named-parameter registry.  This helper
    intentionally delegates the matrix drawing to Seaborn so the visual result
    stays close to the legacy nonlinear SMC KDE figures.
    """
    try:
        import pandas as pd
        import seaborn as sns

        columns = [f"_p{idx}" for idx in range(values.shape[1])]
        df = pd.DataFrame(values, columns=columns)
        style_context = (
            sns.axes_style(style)
            if isinstance(style, str) and style in _SEABORN_STYLE_ALIASES
            else nullcontext()
        )
        with style_context:
            grid = sns.PairGrid(df, vars=columns, diag_sharey=False)
            if "figsize" in figure_kwargs:
                grid.figure.set_size_inches(*figure_kwargs["figsize"])
            if not scatter:
                for row, col in zip(*np.triu_indices_from(grid.axes, 1)):
                    grid.axes[row, col].set_visible(False)
            _map_pairgrid_kde(grid, "map_diag", sns, fill=fill, bw_method=kde_bw_method)
            _map_pairgrid_kde(grid, "map_lower", sns, fill=fill, bw_method=kde_bw_method)
            if scatter:
                grid.map_upper(
                    sns.scatterplot,
                    s=scatter_size,
                    alpha=0.35,
                    linewidth=0,
                )
            fig = grid.figure
            axes = np.asarray(grid.axes)
    except Exception:
        fig, axes = _build_manual_kde_matrix(
            values,
            figure_kwargs=figure_kwargs,
            fill=fill,
            scatter=scatter,
            scatter_size=scatter_size,
            bins=bins,
            kde_bw_method=kde_bw_method,
        )

    _set_kde_matrix_axes(
        axes,
        labels,
        parameter_limits=parameter_limits,
    )
    return fig, axes


def _map_pairgrid_kde(grid, method_name, sns, *, fill=True, bw_method="scott"):
    kwargs = dict(fill=fill, warn_singular=False)
    if bw_method is not None:
        kwargs["bw_method"] = bw_method
    try:
        getattr(grid, method_name)(sns.kdeplot, **kwargs)
    except TypeError:
        # Older seaborn releases did not accept all keyword arguments.
        getattr(grid, method_name)(sns.kdeplot, fill=fill)


def _build_manual_kde_matrix(
    values,
    *,
    figure_kwargs,
    fill=True,
    scatter=False,
    scatter_size=15,
    bins=40,
    kde_bw_method="scott",
):
    plt = _pyplot()
    n = values.shape[1]
    fig, axes = plt.subplots(n, n, squeeze=False, **figure_kwargs)
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            x = values[:, col]
            y = values[:, row]
            if row == col:
                _plot_1d_density(
                    ax,
                    x,
                    bins=bins,
                    mode="kde",
                    fill=fill,
                    bw_method=kde_bw_method,
                )
            elif row > col:
                _plot_2d_density(
                    ax,
                    x,
                    y,
                    fill=fill,
                    scatter_size=scatter_size,
                    bw_method=kde_bw_method,
                )
            elif scatter:
                ax.scatter(
                    x,
                    y,
                    s=scatter_size,
                    alpha=0.35,
                    linewidths=0,
                    color="tab:blue",
                )
            else:
                ax.set_visible(False)
    return fig, axes


def _set_kde_matrix_axes(axes, labels, *, parameter_limits):
    n = axes.shape[0]
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if not ax.get_visible():
                continue
            if parameter_limits is not None:
                ax.set_xlim(parameter_limits[col])
                if row != col:
                    ax.set_ylim(parameter_limits[row])

            if row == n - 1:
                ax.set_xlabel(labels[col])
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            if col == 0:
                ax.set_ylabel(labels[row])
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)


def _padded_limits(values):
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo else 1.0
    else:
        pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def _apply_kde_matrix_formatting(
    axes,
    *,
    local_names,
    xtick_rotation=None,
    ytick_rotation=None,
    lonlat_decimal=3,
    tick_fontsize=None,
    label_fontsize=None,
    show_minor_ticks=False,
    tick_direction="in",
    major_tick_length=3,
    minor_tick_length=1.5,
    tick_width=0.5,
):
    from matplotlib.ticker import AutoLocator, FormatStrFormatter

    n = axes.shape[0]
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if not ax.get_visible():
                continue
            if show_minor_ticks:
                ax.minorticks_on()
            else:
                ax.minorticks_off()
            _apply_pairgrid_like_spines(ax)
            ax.tick_params(
                axis="both",
                which="major",
                direction=tick_direction,
                length=major_tick_length,
                width=tick_width,
                bottom=True,
                left=True,
                top=False,
                right=False,
                labelbottom=row == n - 1,
                labelleft=col == 0,
                labeltop=False,
                labelright=False,
            )
            if show_minor_ticks:
                ax.tick_params(
                    axis="both",
                    which="minor",
                    direction=tick_direction,
                    length=minor_tick_length,
                    width=tick_width,
                    bottom=True,
                    left=True,
                    top=False,
                    right=False,
                )
            ax.xaxis.set_major_locator(AutoLocator())
            ax.yaxis.set_major_locator(AutoLocator())
            if local_names[col].lower() in {"lon", "lat"}:
                ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{lonlat_decimal}f"))
            else:
                _apply_smart_formatting(ax.xaxis)
            if row == col:
                _apply_smart_formatting(ax.yaxis)
            elif local_names[row].lower() in {"lon", "lat"}:
                ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{lonlat_decimal}f"))
            else:
                _apply_smart_formatting(ax.yaxis)

    default_tick_fontsize = tick_fontsize if tick_fontsize is not None else 10
    default_label_fontsize = label_fontsize if label_fontsize is not None else 12

    for ax in axes[-1, :]:
        if not ax.get_visible():
            continue
        ax.tick_params(axis="x", labelsize=default_tick_fontsize)
        if xtick_rotation is not None:
            for label in ax.get_xticklabels():
                label.set_rotation(xtick_rotation)
                label.set_ha("right")
                label.set_fontsize(default_tick_fontsize)
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=default_label_fontsize)

    for ax in axes[:, 0]:
        if not ax.get_visible():
            continue
        ax.tick_params(axis="y", labelsize=default_tick_fontsize)
        if ytick_rotation is not None:
            for label in ax.get_yticklabels():
                label.set_rotation(ytick_rotation)
                label.set_ha("right")
                label.set_fontsize(default_tick_fontsize)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=default_label_fontsize)


def _apply_pairgrid_like_spines(ax):
    """Match seaborn PairGrid(despine=True): left/bottom spines only."""
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _apply_smart_formatting(axis):
    from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

    vmin, vmax = axis.get_view_interval()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin == 0:
        return
    max_abs = max(abs(vmin), abs(vmax))
    if max_abs > 1e4 or (0 < max_abs < 1e-3):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 4))
        axis.set_major_formatter(formatter)
        return
    import math

    decimals = max(0, int(math.ceil(-math.log10(abs(vmax - vmin) / 5.0))))
    axis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))


def _boundary_fraction(values, lower, upper, tol_fraction=0.01):
    width = upper - lower
    if not np.isfinite(width) or width <= 0:
        return None
    tol = width * tol_fraction
    near = np.logical_or(values <= lower + tol, values >= upper - tol)
    return float(np.mean(near))


def _sort_summary_rows(rows, sort_by):
    if sort_by in (None, "index"):
        return sorted(rows, key=lambda row: row["index"])
    if sort_by == "median":
        return sorted(rows, key=lambda row: row["median"])
    if sort_by == "width":
        return sorted(rows, key=lambda row: row["ci_upper"] - row["ci_lower"])
    if sort_by == "name":
        return sorted(rows, key=lambda row: row["name"])
    raise ValueError("sort_by must be one of 'index', 'median', 'width', or 'name'")


def _split_figsize_for_matplotlib(figsize, style_kwargs):
    style_kwargs = dict(style_kwargs)
    if figsize is None:
        return {}, style_kwargs
    if isinstance(figsize, (tuple, list)) and len(figsize) == 2:
        return {"figsize": tuple(figsize)}, style_kwargs
    style_kwargs.setdefault("figsize", figsize)
    return {}, style_kwargs


def _merge_style_kwargs(
    *,
    style=None,
    tick_fontsize=None,
    save_path=None,
    user_kwargs=None,
):
    merged = dict(_DEFAULT_STYLE)
    if style is not None:
        if isinstance(style, str) and style in _SEABORN_STYLE_ALIASES:
            pass
        else:
            merged["preset"] = style
    if tick_fontsize is not None:
        merged["tick_fontsize"] = tick_fontsize
    if user_kwargs:
        merged.update(user_kwargs)
    if save_path and str(save_path).lower().endswith(".pdf"):
        merged.setdefault("pdf_fonttype", 42)
    return merged


def _resolve_save_path(*, save=False, filename=None, save_path=None):
    if save_path is not None:
        return Path(save_path)
    if save:
        return Path(filename or "figure.png")
    return None


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _tight_layout(fig):
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Tight layout not applied.*",
            category=UserWarning,
        )
        fig.tight_layout()


def _save_show(fig, *, save_path, show, dpi, screen_dpi=200):
    finish_fig(
        fig,
        save_path,
        show=show,
        dpi=dpi,
        screen_dpi=screen_dpi,
        bbox_inches="tight",
    )
