"""
Plotting utilities for leveling and cross-fault offset data fit comparison.

These functions are used by bayesian_multifaults_inversion and blse_multifaults_inversion
to visualize observed vs. synthetic data for the new data types.

Uses ``eqtools.viztools.PlotStyle`` for publication-quality styling and
``set_degree_formatter`` for geographic-coordinate axes when applicable.
All public functions accept ``**kwargs`` forwarded to ``PlotStyle`` so the
caller can override preset, fontsize, figsize, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from eqtools.viztools import PlotStyle, set_degree_formatter, save_fig

logger = logging.getLogger(__name__)

# Default PlotStyle keyword arguments used by every plot in this module.
# Users can override any of these via the ``**style_kwargs`` parameters.
_DEFAULT_STYLE = dict(
    preset='science',
    fontsize=9,
    legend_frame=False,
)


def _merge_style_kwargs(user_kwargs: dict) -> dict:
    """Return ``_DEFAULT_STYLE`` merged with *user_kwargs* (user wins)."""
    merged = _DEFAULT_STYLE.copy()
    merged.update(user_kwargs)
    return merged


def _plot_leveling_fit(data, save_dir=None, file_type='png', show=False,
                       **style_kwargs):
    """
    Plot leveling data fit: observed vs synthetic vertical displacement.

    Creates a three-panel figure:
    1. Observed vs Synthetic scatter plot with 1:1 line
    2. Bar chart of observed and synthetic values per station
    3. Residual bar chart

    Args:
        data : leveling data object (csi.leveling)
        save_dir : Path or str, directory to save figures (None to skip)
        file_type : str, file extension for saving ('png', 'pdf')
        show : bool, whether to show the plot interactively
        **style_kwargs : forwarded to ``PlotStyle`` (e.g. preset, fontsize,
            figsize, dpi, usetex, rcparams …).
    """
    if data.vel is None or data.synth is None:
        logger.warning(f"Leveling data '{data.name}' missing vel or synth, skipping plot.")
        return

    obs = data.vel
    syn = data.synth
    res = obs - syn
    stations = data.station if data.station is not None else np.arange(len(obs))

    skw = _merge_style_kwargs(style_kwargs)
    with PlotStyle(**skw):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        fig.suptitle(f'Leveling Fit: {data.name}', fontweight='bold')

        # Panel 1: Scatter observed vs synthetic
        ax = axes[0]
        vmin = min(obs.min(), syn.min())
        vmax = max(obs.max(), syn.max())
        margin = (vmax - vmin) * 0.1 if vmax != vmin else 0.01
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'k--', lw=0.8, alpha=0.5, label='1:1')
        ax.scatter(obs, syn, s=20, c='steelblue', edgecolors='k', linewidths=0.3, zorder=3)
        ax.set_xlabel('Observed (m)')
        ax.set_ylabel('Synthetic (m)')
        ax.set_title('Observed vs Synthetic')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()

        # Panel 2: Bar chart comparison
        ax = axes[1]
        n = len(obs)
        x = np.arange(n)
        w = 0.35
        ax.bar(x - w / 2, obs, w, label='Observed', color='steelblue', alpha=0.8)
        ax.bar(x + w / 2, syn, w, label='Synthetic', color='coral', alpha=0.8)
        ax.set_xlabel('Station Index')
        ax.set_ylabel('Vertical Disp. (m)')
        ax.set_title('Data Comparison')
        ax.legend()
        if n <= 30:
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in stations], rotation=90, fontsize=6)

        # Panel 3: Residuals
        ax = axes[2]
        colors = ['#d62728' if r > 0 else '#2ca02c' for r in res]
        ax.bar(x, res, color=colors, alpha=0.8)
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel('Station Index')
        ax.set_ylabel('Residual (m)')
        rms = np.sqrt(np.mean(res ** 2))
        ax.set_title(f'Residuals (RMS={rms:.4f} m)')
        if n <= 30:
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in stations], rotation=90, fontsize=6)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_fig(fig, str(save_dir / f'{data.name}_leveling_fit.{file_type}'),
                     dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)


def _plot_leveling_map(data, save_dir=None, file_type='png', show=False,
                       **style_kwargs):
    """
    Plot leveling station locations coloured by observed & synthetic values.

    Geographic axes are decorated with degree formatters via
    ``set_degree_formatter``.

    Args:
        data : leveling data object (csi.leveling)
        save_dir : Path or str, directory to save figures (None to skip)
        file_type : str, file extension for saving ('png', 'pdf')
        show : bool, whether to show the plot interactively
        **style_kwargs : forwarded to ``PlotStyle``.
    """
    if data.vel is None or data.synth is None:
        logger.warning(f"Leveling data '{data.name}' missing vel or synth, skipping map plot.")
        return
    if data.lon is None or data.lat is None:
        logger.warning(f"Leveling data '{data.name}' has no lon/lat, skipping map plot.")
        return

    obs = data.vel
    syn = data.synth
    absmax = max(abs(obs).max(), abs(syn).max())
    if absmax == 0:
        absmax = 0.01

    skw = _merge_style_kwargs(style_kwargs)
    with PlotStyle(**skw):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        fig.suptitle(f'Leveling Map: {data.name}', fontweight='bold')

        for ax, values, title in zip(axes, [obs, syn], ['Observed', 'Synthetic']):
            sc = ax.scatter(data.lon, data.lat, c=values, s=40,
                            cmap='RdBu_r', vmin=-absmax, vmax=absmax,
                            edgecolors='k', linewidths=0.3, zorder=3)
            ax.set_title(title)
            ax.set_aspect('equal')
            set_degree_formatter(ax, axis='both')
            plt.colorbar(sc, ax=ax, label='Vertical Disp. (m)', shrink=0.8)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_fig(fig, str(save_dir / f'{data.name}_leveling_map.{file_type}'),
                     dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)


def _plot_crossfaultoffset_fit(data, save_dir=None, file_type='png', show=False,
                               **style_kwargs):
    """
    Plot cross-fault offset data fit for each observed component.

    Creates one row per observed component (fault_parallel, fault_perpendicular,
    fault_vertical), each row with 2 panels: observed vs synthetic scatter, and
    residual bar chart.

    Args:
        data : crossfaultoffset data object (csi.crossfaultoffset)
        save_dir : Path or str, directory to save figures (None to skip)
        file_type : str, file extension for saving ('png', 'pdf')
        show : bool, whether to show the plot interactively
        **style_kwargs : forwarded to ``PlotStyle`` (e.g. preset, fontsize,
            figsize, dpi, usetex, rcparams …).
    """
    # Collect available components
    components = []
    if data.fault_parallel is not None and data.synth_parallel is not None:
        components.append(('Fault-Parallel', data.fault_parallel, data.synth_parallel))
    if data.fault_perpendicular is not None and data.synth_perpendicular is not None:
        components.append(('Fault-Perpendicular', data.fault_perpendicular, data.synth_perpendicular))
    if data.fault_vertical is not None and data.synth_vertical is not None:
        components.append(('Fault-Vertical', data.fault_vertical, data.synth_vertical))

    if not components:
        logger.warning(f"Cross-fault offset '{data.name}' has no observed/synth pairs, skipping plot.")
        return

    stations = data.station if data.station is not None else np.arange(len(components[0][1]))
    n_comp = len(components)

    skw = _merge_style_kwargs(style_kwargs)
    with PlotStyle(**skw):
        fig, axes = plt.subplots(n_comp, 2, figsize=(10, 3.5 * n_comp),
                                 constrained_layout=True, squeeze=False)
        fig.suptitle(f'Cross-Fault Offset Fit: {data.name}', fontweight='bold')

        for i, (comp_name, obs, syn) in enumerate(components):
            res = obs - syn
            n = len(obs)
            x = np.arange(n)

            # Scatter: observed vs synthetic
            ax = axes[i, 0]
            vmin = min(obs.min(), syn.min())
            vmax = max(obs.max(), syn.max())
            margin = (vmax - vmin) * 0.1 if vmax != vmin else 0.01
            ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                    'k--', lw=0.8, alpha=0.5)
            ax.scatter(obs, syn, s=20, c='steelblue', edgecolors='k', linewidths=0.3, zorder=3)
            ax.set_xlabel('Observed (m)')
            ax.set_ylabel('Synthetic (m)')
            ax.set_title(f'{comp_name}: Obs vs Syn')
            ax.set_aspect('equal', adjustable='box')

            # Residual bar
            ax = axes[i, 1]
            colors = ['#d62728' if r > 0 else '#2ca02c' for r in res]
            ax.bar(x, res, color=colors, alpha=0.8)
            ax.axhline(0, color='k', lw=0.5, ls='--')
            ax.set_xlabel('Station Index')
            ax.set_ylabel('Residual (m)')
            rms = np.sqrt(np.mean(res ** 2))
            ax.set_title(f'{comp_name} Residuals (RMS={rms:.4f} m)')
            if n <= 30:
                ax.set_xticks(x)
                ax.set_xticklabels([str(s) for s in stations], rotation=90, fontsize=6)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_fig(fig, str(save_dir / f'{data.name}_crossfault_fit.{file_type}'),
                     dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)


def _plot_crossfaultoffset_map(data, save_dir=None, file_type='png', show=False,
                               **style_kwargs):
    """
    Plot cross-fault offset point-pair locations coloured by observed value.

    Geographic axes are decorated with ``set_degree_formatter``.

    Args:
        data : crossfaultoffset data object (csi.crossfaultoffset)
        save_dir : Path or str, directory to save figures (None to skip)
        file_type : str, file extension for saving ('png', 'pdf')
        show : bool, whether to show the plot interactively
        **style_kwargs : forwarded to ``PlotStyle``.
    """
    components = []
    if data.fault_parallel is not None:
        components.append(('Fault-Parallel', data.fault_parallel))
    if data.fault_perpendicular is not None:
        components.append(('Fault-Perpendicular', data.fault_perpendicular))
    if data.fault_vertical is not None:
        components.append(('Fault-Vertical', data.fault_vertical))
    if not components:
        logger.warning(f"Cross-fault offset '{data.name}' has no data for map plot.")
        return
    if not hasattr(data, 'lon1') or data.lon1 is None:
        logger.warning(f"Cross-fault offset '{data.name}' missing lon1/lat1, skipping map plot.")
        return

    skw = _merge_style_kwargs(style_kwargs)
    with PlotStyle(**skw):
        n_comp = len(components)
        fig, axes = plt.subplots(1, n_comp, figsize=(5 * n_comp, 4),
                                 constrained_layout=True, squeeze=False)
        fig.suptitle(f'Cross-Fault Offset Map: {data.name}', fontweight='bold')

        mid_lon = (data.lon1 + data.lon2) / 2.0
        mid_lat = (data.lat1 + data.lat2) / 2.0

        for j, (comp_name, values) in enumerate(components):
            ax = axes[0, j]
            absmax = max(abs(values).max(), 1e-6)
            # Plot lines connecting point pairs
            for k in range(len(data.lon1)):
                ax.plot([data.lon1[k], data.lon2[k]],
                        [data.lat1[k], data.lat2[k]],
                        'k-', lw=0.3, alpha=0.4)
            sc = ax.scatter(mid_lon, mid_lat, c=values, s=30,
                            cmap='RdBu_r', vmin=-absmax, vmax=absmax,
                            edgecolors='k', linewidths=0.3, zorder=3)
            ax.set_title(comp_name)
            ax.set_aspect('equal')
            set_degree_formatter(ax, axis='both')
            plt.colorbar(sc, ax=ax, label='Offset (m)', shrink=0.8)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_fig(fig, str(save_dir / f'{data.name}_crossfault_map.{file_type}'),
                     dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)
