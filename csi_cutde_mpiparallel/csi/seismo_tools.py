"""
General-purpose earthquake analysis and plotting tools.

This module provides functions for earthquake catalog analysis, including:
- b-value calculation (Least Squares and Aki's Maximum Likelihood methods)
- Gutenberg-Richter frequency-magnitude distribution plotting
- Cumulative event count calculation and plotting
- Epicenter distribution plotting with optional KDE and histogram
- Time-magnitude stick plot
- All plotting functions are separated from calculation functions for clarity and reusability.

Author: Kefeng He
Date: 2025-06-13
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eqtools.plottools import sci_plot_style, set_degree_formatter
from scipy.stats import linregress
from collections.abc import Sequence

def calc_b_value_least_squares(mags, bin_width=0.1, fit_min_mag=None):
    """
    Calculate b-value using Least Squares (linear regression) method.
    Gutenberg-Richter formula: log10(N) = a - b*M

    Parameters:
        mags: sequence of magnitudes
        bin_width: float, magnitude bin width
        fit_min_mag: float or None, only fit bins with M >= fit_min_mag

    Returns:
        b: float, b-value
        a: float, a-value (intercept)
        fit_info: tuple, regression result (slope, intercept, r_value, p_value, std_err)
    """
    mags = np.asarray(mags)
    m_min, m_max = mags.min(), mags.max()
    bins = np.arange(m_min, m_max + bin_width, bin_width)
    hist, edges = np.histogram(mags, bins=bins)
    N = np.cumsum(hist[::-1])[::-1]
    M = edges[:-1]
    logN = np.log10(N)
    valid = N > 0
    if fit_min_mag is not None:
        valid = valid & (M >= fit_min_mag)
    fit_M = M[valid]
    fit_logN = logN[valid]
    if len(fit_M) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(fit_M, fit_logN)
        b = -slope
        a = intercept
        return b, a, (slope, intercept, r_value, p_value, std_err)
    else:
        return np.nan, np.nan, None

def calc_b_value_mle(mags, min_mag=None, delta_mag=0.0):
    """
    Calculate b-value using Aki's Maximum Likelihood Estimation (MLE) method.
    Formula: b = log10(e) / (mean(M) - (Mmin - ΔM/2))

    Parameters:
        mags: sequence of magnitudes
        min_mag: float or None, only use M >= min_mag for calculation
        delta_mag: float, magnitude precision (default 0.0)

    Returns:
        b: float, b-value
    """
    mags = np.asarray(mags)
    if min_mag is not None:
        mags = mags[mags >= min_mag]
    if len(mags) < 2:
        return np.nan
    Mmin = mags.min()
    meanM = mags.mean()
    b = np.log10(np.e) / (meanM - (Mmin - delta_mag / 2))
    return b

def calc_cumulative_counts(dt_seq):
    """
    Calculate cumulative event counts sorted by time.

    Parameters:
        dt_seq: sequence (list, tuple, np.ndarray, pd.Series, etc.) of datetime-like objects

    Returns:
        dt_sorted: sorted list of datetime-like objects
        cum_counts: np.ndarray, cumulative counts
    """
    # Convert to list for sorting and compatibility
    dt_seq = list(dt_seq)
    dt_sorted = sorted(dt_seq)
    cum_counts = np.arange(1, len(dt_sorted) + 1)
    return dt_sorted, cum_counts

def filter_by_time(dts, *args, start_time=None, end_time=None):
    """
    Filter events by time range.

    Parameters:
        dts: sequence of datetime objects
        *args: sequences of other event properties (e.g., lats, lons, mags)
        start_time: datetime or None
        end_time: datetime or None

    Returns:
        filtered_dts, filtered_args...
    """
    filtered = []
    for items in zip(dts, *args):
        dt = items[0]
        if (start_time is None or dt >= start_time) and (end_time is None or dt <= end_time):
            filtered.append(items)
    if not filtered:
        return ([],) * (1 + len(args))
    return tuple([list(x) for x in zip(*filtered)])

# -------------------------------------------------------------------------#
#                         Plotting Functions                               #
# -------------------------------------------------------------------------#
def get_figsize(figsize=None):
    """
    Return a tuple for matplotlib figsize based on input.
    Supports:
        - "single": single-column width (e.g., (3.5, 3))
        - "double": double-column width (e.g., (7.2, 3))
        - tuple/list: user-defined size, e.g., (6, 4)
        - None: use matplotlib default
    """
    if figsize is None:
        return None
    elif isinstance(figsize, str):
        if figsize.lower() == "single":
            return (3.5, 3)
        elif figsize.lower() == "double":
            return (7.2, 3)
        else:
            raise ValueError(f"Unknown figsize string: {figsize}")
    elif isinstance(figsize, (tuple, list)) and len(figsize) == 2:
        return tuple(figsize)
    else:
        raise ValueError("figsize must be None, 'single', 'double', or a tuple/list of length 2.")

def plot_epicenter_combo(
    lons, lats, xlim=None, ylim=None,
    show_kde=True, show_hist=True, bins=50, kde_levels=5,
    style=['science', 'no-latex'], figsize=None, plot_style=None,
    title="Epicenter Distribution (with Density)", save_path=None
):
    """
    Plot epicenter distribution with optional KDE and 2D histogram.

    Parameters:
        lons, lats: sequences of longitude and latitude (any sequence type)
        xlim, ylim: axis limits (tuple or None)
        show_kde: bool, whether to plot KDE contours
        show_hist: bool, whether to plot 2D histogram
        bins: int, histogram bins
        kde_levels: int, KDE contour levels
        style: list of seaborn styles (default ['science', 'no-latex'])
        figsize: str or tuple, figure size ("single", "double", or (width, height))
        plot_style: dict or None, additional plot style parameters for sci_plot_style
        title: str, plot title
        save_path: str or None, if given, save figure to this path
    """
    # Convert to numpy arrays for compatibility
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    figsize = get_figsize(figsize)
    custom_style = {'style': style, 'figsize': figsize}
    plot_style = {**plot_style, **custom_style} if plot_style else custom_style
    # Set plot style
    with sci_plot_style(**plot_style):
        ax = plt.gca()
        sns.scatterplot(x=lons, y=lats, s=10, color=".15", ax=ax)
        if show_hist:
            sns.histplot(x=lons, y=lats, bins=bins, pthresh=.1, cmap="mako", ax=ax)
        if show_kde:
            sns.kdeplot(x=lons, y=lats, levels=kde_levels, color="w", linewidths=1, ax=ax)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        # from matplotlib.ticker import FuncFormatter
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x}°"))
        # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y}°"))
        set_degree_formatter(ax)
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

def plot_time_mag(dt_seq, mag_seq, 
                  style=['science', 'no-latex'], figsize='double', plot_style=None,
                  title="Time-Magnitude Stick Plot", save_path=None):
    """
    Plot time-magnitude stick plot.

    Parameters:
        dt_seq: sequence (list, tuple, np.ndarray, pd.Series, etc.) of datetime-like objects
        mag_seq: sequence of magnitudes
        style: list of seaborn styles (default ['science', 'no-latex'])
        figsize: str or tuple, figure size ("single", "double", or (width, height))
        plot_style: dict or None, additional plot style parameters for sci_plot_style
        title: str, plot title
        save_path: str or None, if given, save figure to this path
    """
    # Convert to suitable types for plotting
    dt_seq = list(dt_seq)
    mag_seq = np.asarray(mag_seq)

    figsize = get_figsize(figsize)
    custom_style = {'style': style, 'figsize': figsize}
    plot_style = {**plot_style, **custom_style} if plot_style else custom_style
    # Set plot style
    with sci_plot_style(**plot_style):
        plt.vlines(dt_seq, [0], mag_seq, color='gray', alpha=0.4, zorder=1)
        plt.scatter(dt_seq, mag_seq, color='black', s=10, zorder=2)
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

def plot_cumulative(dt_seq, 
                    style=['science', 'no-latex'], figsize='double', plot_style=None,
                    title="Cumulative Number vs. Time", save_path=None):
    """
    Plot cumulative event count over time.

    Parameters:
        dt_seq: sequence (list, tuple, np.ndarray, pd.Series, etc.) of datetime-like objects
        style: list of seaborn styles (default ['science', 'no-latex'])
        figsize: str or tuple, figure size ("single", "double", or (width, height))
        plot_style: dict or None, additional plot style parameters for sci_plot_style
        title: str, plot title
        save_path: str or None, if given, save figure to this path
    """
    dt_sorted, cum_counts = calc_cumulative_counts(dt_seq)
    
    figsize = get_figsize(figsize)
    custom_style = {'style': style, 'figsize': figsize}
    plot_style = {**plot_style, **custom_style} if plot_style else custom_style
    # Set plot style
    with sci_plot_style(**plot_style):
        plt.step(dt_sorted, cum_counts, where='post')
        plt.xlabel("Time")
        plt.ylabel("Cumulative Number")
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

def plot_mag_and_cumulative(dt_seq, mag_seq, 
                            style=['science', 'no-latex'], figsize='double', plot_style=None,
                            title="Time-Magnitude Stick Plot & Cumulative Number", save_path=None):
    """
    Plot time-magnitude stick plot and cumulative count on twin y-axes.

    Parameters:
        dt_seq: sequence (list, tuple, np.ndarray, pd.Series, etc.) of datetime-like objects
        mag_seq: sequence of magnitudes
        style: list of seaborn styles (default ['science', 'no-latex'])
        figsize: str or tuple, figure size ("single", "double", or (width, height))
        plot_style: dict or None, additional plot style parameters for sci_plot_style
        title: str, plot title
        save_path: str or None, if given, save figure to this path
    """
    dt_sorted, cum_counts = calc_cumulative_counts(dt_seq)
    dt_seq = list(dt_seq)
    mag_seq = np.asarray(mag_seq)

    figsize = get_figsize(figsize)
    custom_style = {'style': style, 'figsize': figsize}
    plot_style = {**plot_style, **custom_style} if plot_style else custom_style
    # fig, ax1 = plt.subplots(figsize=(10, 4))
    with sci_plot_style(**plot_style):
        ax1 = plt.gca()
        ax1.vlines(dt_seq, [0], mag_seq, color='gray', alpha=0.4, zorder=1)
        ax1.scatter(dt_seq, mag_seq, color='black', s=10, zorder=2)
        ax1.set_ylabel("Magnitude", color='black')
        ax1.set_xlabel("Time")
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_title(title)
        ax2 = ax1.twinx()
        ax2.step(dt_sorted, cum_counts, where='post', color='tab:red', linewidth=1.2)
        ax2.set_ylabel("Cumulative Number", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

def plot_gutenberg_richter(
    mag_seq, bin_width=0.1, fit_min_mag=None, fit_method="ls", delta_mag=0.0,
    style=['science'], figsize=None, plot_style=None,
    title="Gutenberg-Richter (logN-M) Relation", save_path=None
):
    """
    Plot Gutenberg-Richter frequency-magnitude distribution and fit b-value.

    Parameters:
        mag_seq: sequence (list, tuple, np.ndarray, pd.Series, etc.) of magnitudes
        bin_width: float, magnitude bin width
        fit_min_mag: float or None, only fit bins with M >= fit_min_mag
        fit_method: "ls" for least squares, "mle" for Aki's MLE
        delta_mag: float, magnitude precision for MLE
        style: list of seaborn styles (default ['science', 'no-latex'])
        figsize: str or tuple, figure size ("single", "double", or (width, height))
        plot_style: dict or None, additional plot style parameters for sci_plot_style
        title: str, plot title
        save_path: str or None, if given, save figure to this path
    """
    mags = np.asarray(mag_seq)
    m_min, m_max = mags.min(), mags.max()
    bins = np.arange(m_min, m_max + bin_width, bin_width)
    hist, edges = np.histogram(mags, bins=bins)
    N = np.cumsum(hist[::-1])[::-1]
    M = edges[:-1]
    logN = np.log10(N)
    valid = N > 0
    if fit_min_mag is not None:
        valid = valid & (M >= fit_min_mag)
    fit_M = M[valid]
    fit_logN = logN[valid]

    figsize = get_figsize(figsize)
    custom_style = {'style': style, 'figsize': figsize}
    plot_style = {**plot_style, **custom_style} if plot_style else custom_style
    # fig, ax1 = plt.subplots(figsize=(10, 4))
    with sci_plot_style(**plot_style):
        plt.plot(M, logN, 'o-', color='tab:blue', label="Data")

        if fit_method == "ls" and len(fit_M) > 1:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(fit_M, fit_logN)
            b = -slope
            a = intercept
            fit_label = rf"$\log_{{10}}(N) = {a:.2f} - {b:.2f}\times M$"
            plt.plot(fit_M, intercept + slope * fit_M, 'r--', label=f"Fit: {fit_label}")
        elif fit_method == "mle" and len(fit_M) > 1:
            b = np.log10(np.e) / (fit_M.mean() - (fit_M.min() - delta_mag / 2))
            fit_label = f"MLE b = {b:.2f}"
            plt.axhline(np.nan, color='r', linestyle='--', label=fit_label)  # Just for legend
        else:
            fit_label = "Not enough data for fit"

        plt.xlabel("Magnitude")
        plt.ylabel("$log_{10}(N≥M)$")
        plt.title(title)
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()