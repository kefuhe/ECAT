"""CSI fault slip-distribution plotting; generic axes tools live in ecat_viz."""
import os
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ecat_viz import optimize_3d_plot

_MISSING_SLIP = object()


def _snapshot_slip(fault):
    """Return enough state to restore ``fault.slip`` exactly."""
    if not hasattr(fault, 'slip'):
        return _MISSING_SLIP
    value = fault.slip
    return None if value is None else value.copy()


def _restore_slip(fault, snapshot):
    """Restore a snapshot made by :func:`_snapshot_slip`."""
    if snapshot is _MISSING_SLIP:
        if hasattr(fault, 'slip'):
            delattr(fault, 'slip')
    else:
        fault.slip = snapshot


def _install_custom_slip(faults, slips):
    """Temporarily install custom strike-slip arrays and return snapshots.

    Assignment is transactional: if any later fault fails validation, faults
    already modified by this call are restored before the error is raised.
    """
    snapshots = []
    try:
        for current_fault, current_slip in zip(faults, slips):
            snapshot = _snapshot_slip(current_fault)
            snapshots.append((current_fault, snapshot))
            slip_array = np.asarray(current_slip)
            if slip_array.ndim == 2 and slip_array.shape[0] == 1:
                slip_array = slip_array.ravel()
            if slip_array.ndim != 1:
                raise ValueError("Custom slip must be a one-dimensional array.")
            if snapshot is _MISSING_SLIP or snapshot is None:
                current_fault.slip = np.zeros((len(slip_array), 3))
            if np.asarray(current_fault.slip).ndim != 2 or current_fault.slip.shape[1] < 1:
                raise ValueError("fault.slip must be a two-dimensional component array.")
            if current_fault.slip.shape[0] != len(slip_array):
                raise ValueError(
                    "Custom slip length must match the number of rows in fault.slip."
                )
            current_fault.slip[:, 0] = slip_array
    except Exception:
        for current_fault, snapshot in reversed(snapshots):
            _restore_slip(current_fault, snapshot)
        raise
    return snapshots


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
    from ecat_viz import cpt as get_cpt
    import cmcrameri
    from matplotlib.ticker import FuncFormatter
    from ecat_viz import sci_plot_style

    if isinstance(cmap, str) and cmap.endswith('.cpt'):
        cmap = get_cpt.get_cmap(cmap, method=method, N=N)

    cbfontsize = cbfontsize if cbfontsize is not None else plt.rcParams['axes.labelsize']
    cblinewidth = cblinewidth if cblinewidth is not None else plt.rcParams['axes.linewidth']

    # Process slip parameter - check if it's an array or string
    slip_type = 'total'  # default for figname
    slip_snapshots = None

    if not isinstance(slip, str):
        # slip is an array or list of arrays
        if not isinstance(fault, list):
            # Single fault case
            slip_snapshots = _install_custom_slip([fault], [slip])

            slip = 'strikeslip'  # Use 'strikeslip' for plotting
            slip_type = 'custom'
        else:
            # Multiple faults case
            if not isinstance(slip, list):
                raise ValueError("For multiple faults, slip must be a list of arrays.")
            if len(slip) != len(fault):
                raise ValueError(f"Number of slip arrays ({len(slip)}) must match number of faults ({len(fault)})")

            slip_snapshots = _install_custom_slip(fault, slip)

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
                        warnings.warn(
                            f"Fault {ifault.name} is not triangular; plotting its edge vertices is unsupported.",
                            UserWarning,
                            stacklevel=2,
                        )

            if plotTrace and add_faults is not None:
                for ifault in add_faults:
                    if ifault.lon is not None and ifault.lat is not None:
                        fig.faulttrace(ifault, color='r', discretized=False, linewidth=1, zorder=1)
                    else:
                        warnings.warn(
                            f"Fault {ifault.name} has no trace data.",
                            UserWarning,
                            stacklevel=2,
                        )

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
        # A custom plotting array must never become persistent model state.
        if slip_snapshots is not None:
            for current_fault, snapshot in reversed(slip_snapshots):
                _restore_slip(current_fault, snapshot)
