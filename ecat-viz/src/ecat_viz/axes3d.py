"""
viz_3d.py — Generic Matplotlib 3-D axis tools.

Public API
----------
optimize_3d_plot       : Optimise 3-D axis appearance

"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def optimize_3d_plot(ax, zratio=None, shape=(1.0, 1.0, 0.25), zaxis_position='bottom-left',
                     show_grid=True, grid_color='#bebebe', background_color='white', axis_color=None, grid_which='major',
                     show_xy_grid=True, show_xz_grid=True, show_yz_grid=True):
    """Optimize the appearance of a 3D Axes for publication-quality figures.

    This function provides fine-grained control over 3D plot aesthetics including
    axis ratios, grid lines, pane colors, and tick mark appearance. It's designed
    to create clean, professional-looking 3D visualizations suitable for papers
    and presentations.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axes object to optimize.
    zratio : float, optional
        Z-axis scaling ratio relative to X and Y axes. When specified, applies
        a projection transformation to compress or expand the Z dimension.
        If None, uses ``shape`` parameter instead. Default is None.
    shape : tuple of float, optional
        3D box aspect ratio as (x_scale, y_scale, z_scale). Only used when
        ``zratio`` is None. Default is (1.0, 1.0, 0.25), which compresses
        the Z-axis to 25% of X/Y dimensions.
    zaxis_position : {'bottom-left', 'top-right'}, optional
        Position of the Z-axis labels and ticks:

        - 'bottom-left': Place Z-axis on the lower-left corner (default)
        - 'top-right': Place Z-axis on the upper-right corner

    show_grid : bool, optional
        Whether to display grid lines on the 3D plot. Default is True.
    grid_color : str, optional
        Color for grid lines. Accepts any matplotlib color specification.
        Default is '#bebebe' (light gray).
    background_color : str or None, optional
        Background color for the 3D panes (XY, XZ, YZ planes):

        - Color string: Fill panes with this color (default is 'white')
        - None: Make panes transparent

    axis_color : str, optional
        Color for axis lines and ticks. If None, uses matplotlib default.
        Default is None.
    grid_which : {'major', 'minor', 'both'}, optional
        Which grid lines to display. Default is 'major'.
    show_xy_grid : bool, optional
        Whether to show grid lines on the XY plane. Default is True.
    show_xz_grid : bool, optional
        Whether to show grid lines on the XZ plane. Default is True.
    show_yz_grid : bool, optional
        Whether to show grid lines on the YZ plane. Default is True.

    Returns
    -------
    None
        Modifies the axes object in-place.

    Notes
    -----
    - The function modifies low-level 3D axes properties via the ``_axinfo``
      attribute, which provides finer control than public APIs.
    - Tick marks are configured to point outward for better visibility.
    - Grid lines are only shown where tick labels are present, reducing
      visual clutter.
    - The function is safe to call multiple times on the same axes.

    Examples
    --------
    Basic usage with default settings:

    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> import matplotlib.pyplot as plt
    >>> from ecat_viz import optimize_3d_plot
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> ax.plot([0, 1], [0, 1], [0, 1])
    >>> optimize_3d_plot(ax)
    >>> plt.show()

    Create a flattened Z-axis view:

    >>> optimize_3d_plot(ax, shape=(1.0, 1.0, 0.1))

    Transparent background with custom grid color:

    >>> optimize_3d_plot(ax, background_color=None, grid_color='#cccccc')

    Position Z-axis on the top-right:

    >>> optimize_3d_plot(ax, zaxis_position='top-right')

    Show grid only on XY plane (useful for depth plots):

    >>> optimize_3d_plot(ax, show_xy_grid=True, show_xz_grid=False, show_yz_grid=False)

    See Also
    --------
    matplotlib.axes.Axes.set_aspect : Set 2D aspect ratio
    mpl_toolkits.mplot3d.Axes3D.set_box_aspect : Set 3D box aspect
    plot_slip_distribution : Plot fault slip distribution with optimized 3D view

    Warnings
    --------
    This function accesses private attributes (``_axinfo``, ``_PLANES``) of
    matplotlib's 3D axes. While stable in recent matplotlib versions, these
    may change in future releases.
    """
    # Set Z axis ratio
    if zratio is not None:
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax),
                                     np.diag([1.0, 1.0, zratio, 1]))
    else:
        ax.set_box_aspect([shape[0], shape[1], shape[2]])

    # Set z-axis position
    if zaxis_position == 'bottom-left':
        ax.zaxis.set_ticks_position('lower')
        ax.zaxis.set_label_position('lower')
    elif zaxis_position == 'top-right':
        tmp_planes = ax.zaxis._PLANES
        ax.zaxis._PLANES = (tmp_planes[0], tmp_planes[1],
                            tmp_planes[2], tmp_planes[3],
                            tmp_planes[4], tmp_planes[5])

    # Set grid lines
    if show_grid:
        ax.grid(True, which=grid_which)
    else:
        ax.grid(False)

    # Set grid line colors
    ax.xaxis._axinfo['grid'].update(color=grid_color)
    ax.yaxis._axinfo['grid'].update(color=grid_color)
    ax.zaxis._axinfo['grid'].update(color=grid_color)

    # Set pane background color (background_color controls fill; None = transparent)
    _transparent = (1.0, 1.0, 1.0, 0.0)
    _pane_color = _transparent if background_color is None else background_color
    for _a in [ax.xaxis, ax.yaxis, ax.zaxis]:
        _a.set_pane_color(_pane_color)

    # Set axis line / tick color (axis_color is separate from pane fill)
    if axis_color is not None:
        for _a in [ax.xaxis, ax.yaxis, ax.zaxis]:
            _a._axinfo['color'] = axis_color

    # Set tick lines to be outside
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', direction='out')
    ax.tick_params(axis='z', direction='out')

    # Ensure tick lines are mainly outside
    ax.xaxis._axinfo['tick']['inward_factor'] = 0.4
    ax.xaxis._axinfo['tick']['outward_factor'] = 0
    ax.yaxis._axinfo['tick']['inward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['outward_factor'] = 0
    ax.zaxis._axinfo['tick']['inward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['outward_factor'] = 0

    # Only show tick lines and grid lines where there are tick labels
    ax.xaxis._axinfo['tick']['tick1On'] = False
    ax.xaxis._axinfo['tick']['tick2On'] = False
    ax.yaxis._axinfo['tick']['tick1On'] = False
    ax.yaxis._axinfo['tick']['tick2On'] = False
    ax.zaxis._axinfo['tick']['tick1On'] = False
    ax.zaxis._axinfo['tick']['tick2On'] = False

    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(True)
        tick.tick2line.set_visible(True)
        tick.gridline.set_visible(show_xy_grid)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(True)
        tick.tick2line.set_visible(True)
        tick.gridline.set_visible(show_xy_grid)
    for tick in ax.zaxis.get_major_ticks():
        tick.tick1line.set_visible(True)
        tick.tick2line.set_visible(True)
        tick.gridline.set_visible(show_xz_grid or show_yz_grid)

    # Hide grid lines where there are no tick labels
    for line in ax.xaxis.get_gridlines():
        line.set_visible(False)
    for line in ax.yaxis.get_gridlines():
        line.set_visible(False)
    for line in ax.zaxis.get_gridlines():
        line.set_visible(False)

    for tick in ax.xaxis.get_major_ticks():
        if tick.label1.get_visible():
            tick.gridline.set_visible(show_xy_grid)
    for tick in ax.yaxis.get_major_ticks():
        if tick.label1.get_visible():
            tick.gridline.set_visible(show_xy_grid)
    for tick in ax.zaxis.get_major_ticks():
        if tick.label1.get_visible():
            tick.gridline.set_visible(show_xz_grid or show_yz_grid)

    # Set background color
    ax.set_facecolor(background_color)


