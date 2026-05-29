"""
viz_3d.py — 3-D plot helpers and CSI fault slip-distribution plotter.

Public API
----------
optimize_3d_plot       : Optimise 3-D axis appearance
plot_slip_distribution : Plot CSI fault slip distribution (3-D + map)
"""

import os

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
    >>> from eqtools.viztools import optimize_3d_plot
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


def plot_slip_distribution(fault, slip='total', add_faults=None, cmap='precip3_16lev_change.cpt', norm=None,
                           figsize=(None, None), drawCoastlines=False, plot_on_2d=True, method='cdict', N=None,
                           cbaxis=[0.1, 0.2, 0.1, 0.02], cblabel='', show=True, savefig=False,
                           ftype='pdf', dpi=600, bbox_inches=None, remove_direction_labels=False,
                           cbticks=None, cblinewidth=None, cbfontsize=None, cb_label_side='opposite',
                           map_cbaxis=None, style=['notebook'], xlabelpad=None, ylabelpad=None, zlabelpad=None,
                           xtickpad=None, ytickpad=None, ztickpad=None, elevation=None, azimuth=None,
                           shape=(1.0, 1.0, 1.0), zratio=None, plotTrace=True, depth=None, zticks=None,
                           map_expand=0.2, fault_expand=0.1, plot_faultEdges=False, faultEdges_color='k',
                           faultEdges_linewidth=1.0, suffix='', outdir=None, show_grid=True, grid_color='#bebebe',
                           background_color='white', axis_color=None, zaxis_position='bottom-left', figname=None,
                           show_xy_grid=True, show_xz_grid=True, show_yz_grid=True):
    """Plot the slip distribution of a fault.

    Parameters
    ----------
    fault : fault object or list of fault objects
        The fault object to plot.
    slip : str, array, or list
        Type of slip to plot or slip array(s). Can be:
        * str: 'total', 'strikeslip', 'dipslip', etc. (default is 'total')
        * 1D array (n,): slip for single fault
        * 2D array (1, n): slip for single fault
        * List of arrays: slip for multiple faults (when fault is a list)
    add_faults : list, optional
        Additional faults to plot the trace of (default is None).
    cmap : str
        Colormap to use (default is 'precip3_16lev_change.cpt').
    norm : optional
        Normalization for the colormap (default is None).
    figsize : tuple
        Size of the figure and map (default is (None, None)).
    drawCoastlines : bool
        Whether to draw coastlines (default is False).
    plot_on_2d : bool
        Whether to plot on a 2D map (default is True).
    method : str
        Method for getting the colormap (default is 'cdict').
    N : int, optional
        Number of colors in the colormap (default is None).
    cbaxis : list
        Colorbar axis position (default is [0.1, 0.2, 0.1, 0.02]).
    cblabel : str
        Label for the colorbar (default is '').
    show : bool
        Whether to show the plot (default is True).
    savefig : bool
        Whether to save the figure (default is False).
    ftype : str
        File type for saving the figure (default is 'pdf').
    dpi : int
        Dots per inch for the saved figure (default is 600).
    bbox_inches : optional
        Bounding box in inches for saving the figure (default is None).
    remove_direction_labels : bool
        If True, remove E, N, S, W from axis labels (default is False).
    cbticks : list, optional
        List of ticks to set on the colorbar (default is None).
    cblinewidth : int, optional
        Width of the colorbar label border and tick lines (default is 1).
    cbfontsize : int, optional
        Font size of the colorbar label (default is None).
    cb_label_side : str
        Position of the label relative to the ticks ('opposite' or 'same', default is 'opposite').
    map_cbaxis : optional
        Axis for the colorbar on the map plot, default is None.
    style : list
        Style for the plot (default is ['notebook']).
    xlabelpad, ylabelpad, zlabelpad : float, optional
        Padding for the axis labels (default is None).
    xtickpad, ytickpad, ztickpad : float, optional
        Padding for the axis ticks (default is None).
    elevation, azimuth : float, optional
        Elevation and azimuth angles for the 3D plot (default is None).
    shape : tuple
        Shape of the 3D plot (default is (1.0, 1.0, 1.0)).
    zratio : float, optional
        Ratio for the z-axis (default is None).
    plotTrace : bool
        Whether to plot the fault trace (default is True).
    depth : float, optional
        Depth for the z-axis (default is None).
    zticks : list, optional
        Ticks for the z-axis (default is None).
    map_expand : float
        Expansion factor for the map (default is 0.2).
    fault_expand : float
        Expansion factor for the fault (default is 0.1).
    plot_faultEdges : bool
        Whether to plot the fault edges (default is False).
    faultEdges_color : str
        Color for the fault edges (default is 'k').
    faultEdges_linewidth : float
        Line width for the fault edges (default is 1.0).
    suffix : str
        Suffix for the saved figure filename (default is '').
    outdir : str, optional
        Output directory for saving the figure (default is None).
    show_grid : bool
        Whether to show grid lines (default is True).
    grid_color : str
        Color of the grid lines (default is '#bebebe').
    background_color : str
        Background color of the plot (default is 'white').
    axis_color : str, optional
        Color of the axes (default is None).
    zaxis_position : str
        Position of the z-axis (bottom-left, top-right) (default is 'bottom-left').
    figname : str, optional
        Name of the figure (default is None).
    show_xy_grid : bool
        Whether to show grid lines on the xy plane (default is True).
    show_xz_grid : bool
        Whether to show grid lines on the xz plane (default is True).
    show_yz_grid : bool
        Whether to show grid lines on the yz plane (default is True).
    """
    from ..getcpt import get_cpt
    import cmcrameri
    from matplotlib.ticker import FuncFormatter
    from ._compat import sci_plot_style

    if isinstance(cmap, str) and cmap.endswith('.cpt'):
        cmap = get_cpt.get_cmap(cmap, method, N)

    cbfontsize = cbfontsize if cbfontsize is not None else plt.rcParams['axes.labelsize']
    cblinewidth = cblinewidth if cblinewidth is not None else plt.rcParams['axes.linewidth']

    # Process slip parameter - check if it's an array or string
    slip_type = 'total'  # default for figname
    original_slip = None  # Store original slip for restoration

    if not isinstance(slip, str):
        # slip is an array or list of arrays
        if not isinstance(fault, list):
            # Single fault case
            slip_array = np.asarray(slip)
            if slip_array.ndim == 2 and slip_array.shape[0] == 1:
                slip_array = slip_array.flatten()

            # Save original slip and set custom slip to strike-slip component
            original_slip = fault.slip.copy() if hasattr(fault, 'slip') and fault.slip is not None else None
            if original_slip is None:
                fault.slip = np.zeros((len(slip_array), 3))
            fault.slip[:, 0] = slip_array  # Set to strike-slip component

            slip = 'strikeslip'  # Use 'strikeslip' for plotting
            slip_type = 'custom'
        else:
            # Multiple faults case
            if not isinstance(slip, list):
                raise ValueError("For multiple faults, slip must be a list of arrays.")
            if len(slip) != len(fault):
                raise ValueError(f"Number of slip arrays ({len(slip)}) must match number of faults ({len(fault)})")

            # Save original slip for each fault and set custom slip
            original_slip = []
            for ifault, islip in zip(fault, slip):
                islip_array = np.asarray(islip)
                if islip_array.ndim == 2 and islip_array.shape[0] == 1:
                    islip_array = islip_array.flatten()

                # Save original and set custom
                orig = ifault.slip.copy() if hasattr(ifault, 'slip') and ifault.slip is not None else None
                original_slip.append(orig)
                if orig is None:
                    ifault.slip = np.zeros((len(islip_array), 3))
                ifault.slip[:, 0] = islip_array  # Set to strike-slip component

            slip = 'strikeslip'  # Use 'strikeslip' for plotting
            slip_type = 'custom'
    else:
        # slip is a string like 'total', 'strikeslip', etc.
        slip_type = slip

    try:
        with sci_plot_style(style=style):
            if not isinstance(fault, list):
                fault.plot(drawCoastlines=drawCoastlines, slip=slip, cmap=cmap, norm=norm, savefig=False,
                        ftype=ftype, dpi=dpi, bbox_inches=bbox_inches, plot_on_2d=plot_on_2d,
                        figsize=figsize, cbaxis=cbaxis, cblabel=cblabel, show=False, expand=map_expand,
                        remove_direction_labels=remove_direction_labels, cbticks=cbticks,
                        cblinewidth=cblinewidth, cbfontsize=cbfontsize, cb_label_side=cb_label_side, map_cbaxis=map_cbaxis)
                ax = fault.slipfig.faille
                fig = fault.slipfig
                name = fault.name
            else:
                # Make a plot
                from csi.geodeticplot import geodeticplot as geoplt
                lon_min = min([p[:, 0].min() for f in fault for p in f.patchll])
                lon_max = max([p[:, 0].max() for f in fault for p in f.patchll])
                lat_min = min([p[:, 1].min() for f in fault for p in f.patchll])
                lat_max = max([p[:, 1].max() for f in fault for p in f.patchll])
                depth_max = max([p[:, 2].max() for f in fault for p in f.patchll])
                gp = geoplt(lon_min, lat_min, lon_max, lat_max, figsize=figsize)
                plot_colorbar = True
                for ifault in fault:
                    gp.faultpatches(ifault, slip=slip, colorbar=plot_colorbar,
                                    plot_on_2d=False, norm=norm, cmap=cmap,
                                    cbaxis=cbaxis, cblabel=cblabel,
                                    cbticks=cbticks, cblinewidth=cblinewidth, cbfontsize=cbfontsize,
                                    cb_label_side=cb_label_side, map_cbaxis=map_cbaxis,
                                    alpha=1.0 if plot_colorbar else 0.4)
                    plot_colorbar = False  # Only add one colorbar for multiple faults

                ax = gp.faille
                fig = gp
                name = 'multiple_faults'

            # Only for triangular faults at current stage
            if plot_faultEdges and add_faults is not None:
                for ifault in add_faults:
                    if ifault.patchType == 'triangle':
                        ifault.find_fault_fouredge_vertices(refind=True)
                        for edgename in ifault.edge_vertices:
                            edge = ifault.edge_vertices[edgename]
                            x, y, z = edge[:, 0], edge[:, 1], -edge[:, 2]
                            lon, lat = ifault.xy2ll(x, y)
                            ax.plot(lon, lat, z, color=faultEdges_color, linewidth=faultEdges_linewidth)
                    else:
                        Warning(f"Fault {ifault.name} is not a triangular fault. Currently, finding edge vertices is not supported.")

            if plotTrace and add_faults is not None:
                for ifault in add_faults:
                    if ifault.lon is not None and ifault.lat is not None:
                        fig.faulttrace(ifault, color='r', discretized=False, linewidth=1, zorder=1)
                    else:
                        Warning(f"Fault {fault.name} has no trace data.")

            # Set labels and title with optional labelpad
            ax.set_xlabel('Longitude', labelpad=xlabelpad)
            ax.set_ylabel('Latitude', labelpad=ylabelpad)
            ax.set_zlabel('Depth (km)', labelpad=zlabelpad)

            # Adjust tick parameters with optional pad
            if xtickpad is not None:
                ax.tick_params(axis='x', pad=xtickpad)
            if ytickpad is not None:
                ax.tick_params(axis='y', pad=ytickpad)
            if ztickpad is not None:
                ax.tick_params(axis='z', pad=ztickpad)

            # Set Z tick labels
            if depth is not None and zticks is not None:
                ax.set_zticks(zticks)
                ax.set_zlim3d([-depth, 0])
            if fault_expand is not None:
                faults_list = fault if isinstance(fault, list) else [fault]
                lon_min = min([p[:, 0].min() for f in faults_list for p in f.patchll])
                lon_max = max([p[:, 0].max() for f in faults_list for p in f.patchll])
                lat_min = min([p[:, 1].min() for f in faults_list for p in f.patchll])
                lat_max = max([p[:, 1].max() for f in faults_list for p in f.patchll])
                ax.set_xlim(lon_min - fault_expand, lon_max + fault_expand)
                ax.set_ylim(lat_min - fault_expand, lat_max + fault_expand)
            ax.zaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{abs(val)}'))

            # Set View
            if elevation is not None and azimuth is not None:
                ax.view_init(elev=elevation, azim=azimuth)
            else:
                if isinstance(fault, list):
                    strike = np.mean(np.hstack([f.getStrikes() for f in fault]) * 180 / np.pi)
                    dip = np.mean(np.hstack([f.getDips() for f in fault]) * 180 / np.pi)
                else:
                    strike = np.mean(fault.getStrikes() * 180 / np.pi)
                    dip = np.mean(fault.getDips() * 180 / np.pi)
                azimuth = -strike
                elevation = 90 - dip + 10
                ax.view_init(elev=elevation, azim=azimuth)

            if isinstance(fault, list):
                fig.setzaxis(depth_max)

            # Set 3D plot shape
            optimize_3d_plot(ax, shape=shape, zratio=zratio, zaxis_position=zaxis_position,
                             show_grid=show_grid, grid_color=grid_color,
                             background_color=background_color, axis_color=axis_color,
                             show_xy_grid=show_xy_grid, show_xz_grid=show_xz_grid, show_yz_grid=show_yz_grid)

            if savefig:
                saveFig = ['fault']
                if figname is None:
                    clean_name = name.replace(' ', '_')
                    if outdir is not None:
                        if not os.path.exists(outdir):
                            os.makedirs(outdir)
                        prefix = os.path.join(outdir, clean_name)
                    else:
                        prefix = clean_name
                    suffix = f'_{suffix}' if suffix != '' else ''
                    figname = prefix + '{0}_{1}'.format(suffix, slip_type)
                else:
                    if outdir is not None:
                        if not os.path.exists(outdir):
                            os.makedirs(outdir)
                    figname = os.path.join(outdir, figname) if outdir is not None else figname
                if plot_on_2d:
                    saveFig.append('map')
                fig.savefig(figname, ftype=ftype, dpi=dpi, bbox_inches=bbox_inches, saveFig=saveFig)

            if show:
                showFig = ['fault']
                if plot_on_2d:
                    showFig.append('map')
                fig.show(showFig=showFig)
                plt.show()

    finally:
        # Restore original slip values
        if original_slip is not None:
            if not isinstance(fault, list):
                if original_slip is not None:
                    fault.slip = original_slip
            else:
                for ifault, orig in zip(fault, original_slip):
                    if orig is not None:
                        ifault.slip = orig
