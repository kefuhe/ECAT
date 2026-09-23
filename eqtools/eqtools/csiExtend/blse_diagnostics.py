"""Domain plots for fixed-geometry BLSE smoothing diagnostics.

The functions in this module consume the canonical table returned by
``BoundLSEMultiFaultsInversion.scan_penalty_weights``.  They do not run an
inversion, rebuild statistics, convert units, save files, or display figures.
Legacy table and unit handling remains at the legacy BLSE method boundary.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd

from eqtools.viztools import PlotStyle, bake_text_fonts


_SUMMARY_COLUMNS = {
    "candidate_index",
    "penalty_weight",
    "roughness",
    "global_rms",
    "global_vr_percent",
    "observation_unit",
}


def _canonical_summary(summary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Validate and copy one canonical BLSE penalty-scan summary."""
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame")
    missing = sorted(_SUMMARY_COLUMNS.difference(summary.columns))
    if missing:
        raise ValueError(
            "BLSE diagnostic summary is missing canonical column(s): "
            + ", ".join(missing)
        )
    if summary.empty:
        raise ValueError("BLSE diagnostic summary must contain at least one candidate")

    units = summary["observation_unit"].dropna().astype(str).unique()
    if units.size != 1 or not units[0].strip():
        raise ValueError(
            "BLSE diagnostic summary must contain one unambiguous observation_unit"
        )
    return summary.copy(), units[0]


def _preferred_indices(penalties, preferred_penalty_weight):
    if preferred_penalty_weight is None:
        return np.array([], dtype=int)
    return np.flatnonzero(
        np.isclose(np.asarray(penalties, dtype=float), preferred_penalty_weight)
    )


def plot_blse_lcurve_summary(
    summary,
    preferred_penalty_weight=None,
    *,
    figsize="double",
    figsize_unit="inch",
    label_fontsize=9,
    tick_fontsize=8,
    style="science",
    legend_loc="upper left",
):
    """Plot penalty--RMS, penalty--VR and roughness--RMS diagnostics.

    RMS values remain in the canonical table's ``observation_unit``.  Unit
    conversion belongs to legacy compatibility wrappers or explicit user data
    preparation, not to this domain plot.
    """
    frame, observation_unit = _canonical_summary(summary)
    penalty_values = frame["penalty_weight"].to_numpy(dtype=float)
    roughness_values = frame["roughness"].to_numpy(dtype=float)
    rms_values = frame["global_rms"].to_numpy(dtype=float)
    vr_values = frame["global_vr_percent"].to_numpy(dtype=float)
    preferred = _preferred_indices(penalty_values, preferred_penalty_weight)

    with PlotStyle(
        style,
        figsize=figsize,
        figsize_unit="inch" if isinstance(figsize, str) else figsize_unit,
        figsize_aspect=3.8 / 12.0,
        fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=tick_fontsize,
    ):
        fig, axes = plt.subplots(1, 3)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        panel_colors = [colors[index % len(colors)] for index in range(3)]

        axes[0].semilogx(
            penalty_values, rms_values, "o-", color=panel_colors[0]
        )
        axes[0].set_xlabel("Penalty weight")
        axes[0].set_ylabel(f"Global RMS ({observation_unit})")

        axes[1].semilogx(
            penalty_values, vr_values, "o-", color=panel_colors[1]
        )
        axes[1].set_xlabel("Penalty weight")
        axes[1].set_ylabel("Global VR (%)")

        for axis in axes[:2]:
            axis.xaxis.set_major_locator(
                LogLocator(base=10, subs=(1.0,), numticks=12)
            )
            axis.xaxis.set_minor_locator(
                LogLocator(
                    base=10,
                    subs=np.arange(2, 10) * 0.1,
                    numticks=100,
                )
            )
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.tick_params(
                axis="x",
                which="minor",
                bottom=True,
                top=True,
                length=2.0,
                width=0.6,
            )

        axes[2].plot(
            roughness_values, rms_values, "o-", color=panel_colors[2]
        )
        axes[2].set_xlabel("Roughness")
        axes[2].set_ylabel(f"Global RMS ({observation_unit})")

        for axis in axes:
            axis.grid(alpha=0.25)

        if preferred.size:
            label = f"Preferred weight = {preferred_penalty_weight:g}"
            axes[0].plot(
                penalty_values[preferred],
                rms_values[preferred],
                marker="*",
                markersize=12,
                color="red",
                linestyle="none",
                label=label,
            )
            axes[0].legend(loc=legend_loc)
            axes[1].plot(
                penalty_values[preferred],
                vr_values[preferred],
                marker="*",
                markersize=12,
                color="red",
                linestyle="none",
            )
            axes[2].plot(
                roughness_values[preferred],
                rms_values[preferred],
                marker="*",
                markersize=12,
                color="red",
                linestyle="none",
            )

        fig.tight_layout()
        bake_text_fonts(fig)
    return fig, axes


def plot_blse_roughness_rms(
    summary,
    preferred_penalty_weight=None,
    *,
    figsize="single",
    figsize_unit="inch",
    label_fontsize=9,
    tick_fontsize=8,
    style="science",
    legend_loc="best",
    equal_aspect=False,
):
    """Plot the single roughness--RMS diagnostic from a canonical summary."""
    frame, observation_unit = _canonical_summary(summary)
    penalties = frame["penalty_weight"].to_numpy(dtype=float)
    roughness = frame["roughness"].to_numpy(dtype=float)
    rms = frame["global_rms"].to_numpy(dtype=float)
    preferred = _preferred_indices(penalties, preferred_penalty_weight)

    with PlotStyle(
        style,
        figsize=figsize,
        figsize_unit="inch" if isinstance(figsize, str) else figsize_unit,
        fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=tick_fontsize,
    ):
        fig, ax = plt.subplots()
        ax.plot(roughness, rms, "o-", label="L-curve")
        if preferred.size:
            ax.plot(
                roughness[preferred],
                rms[preferred],
                marker="o",
                color="#e54726",
                linestyle="none",
                label="Preferred",
            )
            ax.legend(loc=legend_loc)
        ax.set_xlabel("Roughness")
        ax.set_ylabel(f"RMS ({observation_unit})")
        ax.grid(True)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        bake_text_fonts(fig)
    return fig, ax


__all__ = [
    "plot_blse_lcurve_summary",
    "plot_blse_roughness_rms",
]
