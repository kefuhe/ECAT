from os import listdir
from os.path import isdir, join
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scienceplots
from matplotlib import cm
import matplotlib as mpl
from contextlib import contextmanager

# # 保存scienceplots的样式库
# scienceplots_styles = plt.style.library.copy()
# # 将scienceplots的样式库合并到当前的样式库中
# plt.style.library.update(scienceplots_styles)
# # 更新可用的样式列表
# plt.style.available[:] = sorted(plt.style.library.keys())

class DegreeFormatter(mpl.ticker.ScalarFormatter):
    def __call__(self, x, pos=None):
        # Call the parent class to get the original label
        label = super().__call__(x, pos)
        # Add the degree symbol
        return label + '°'


def register_science_styles():
    # register the included stylesheet in the matplotlib style library
    scienceplots_path = scienceplots.__path__[0]
    styles_path = join(scienceplots_path, 'styles')

    # Reads styles in /styles
    stylesheets = plt.style.core.read_style_directory(styles_path)
    # Reads styles in /styles subfolders
    for inode in listdir(styles_path):
        new_data_path = join(styles_path, inode)
        if isdir(new_data_path):
            new_stylesheets = plt.style.core.read_style_directory(new_data_path)
            stylesheets.update(new_stylesheets)

    # Update dictionary of styles
    plt.style.core.update_nested_dict(plt.style.library, stylesheets)
    # Update `plt.style.available`, ensuring all styles are registered
    plt.style.core.available[:] = sorted(plt.style.library.keys())

def set_plot_style(style=['science', 'no-latex'], figsize=(3.5, 2.8), use_degree=False, equal_aspect=False):
    register_science_styles()
    with plt.style.context(style=style):
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.formatter.use_mathtext'] = False
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.fontset'] = 'dejavusans'
        plt.rcParams['font.sans-serif'] = ['Arial']
        fig, ax = plt.subplots(figsize=figsize)
        if use_degree:
            formatter = DegreeFormatter()
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
        if equal_aspect:
            ax.set_aspect('equal', adjustable='box')
        return fig, ax

def set_degree_formatter(ax, axis='both'):
    formatter = DegreeFormatter()
    if axis in ['x', 'both']:
        ax.xaxis.set_major_formatter(formatter)
    if axis in ['y', 'both']:
        ax.yaxis.set_major_formatter(formatter)

def update_style_library():
    import matplotlib.pyplot as plt
    import scienceplots
    # 保存scienceplots的样式库
    scienceplots_styles = plt.style.library.copy()
    # 将scienceplots的样式库合并到当前的样式库中
    plt.style.library.update(scienceplots_styles)
    # 更新可用的样式列表
    plt.style.available[:] = sorted(plt.style.library.keys())

@contextmanager
def sci_plot_style(style=['science', 'no-latex'], legend_frame=False, use_tes=False, 
                   use_mathtext=False, serif=False, fontsize=None, figsize=None, pdf_fonttype=None):
    """
    Set scientific plotting style with context manager.
    
    Parameters:
    ----------
    style : list, optional
        List of styles to use.
    legend_frame : bool, optional
        Whether to add a frame to the legend.
    use_tes : bool, optional
        Whether to use TeX for text rendering.
    use_mathtext : bool, optional
        Whether to use mathtext for text rendering.
    serif : bool, optional
        Whether to use serif fonts.
    fontsize : int, optional
        Font size for text.
    figsize : tuple, optional
        Figure size.
    pdf_fonttype : int, optional
        Set pdf.fonttype in matplotlib. Common values are:
        - 3 (default) for Type 3 fonts
        - 42 for TrueType fonts
    """
    # Register scienceplots styles
    register_science_styles()

    plt.style.use(style)
    if legend_frame:
        plt.rc('legend', frameon=True, framealpha=0.7,
            fancybox=True, numpoints=1)
    if serif:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times', 'Palatino', 'New Century Schoolbook', 
                                       'Bookman', 'Computer Modern Roman']
    else:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Bitstream Vera Sans', 
                                            'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva', 
                                            'Lucid', 'Avant Garde', 'sans-serif']
    plt.rcParams['axes.formatter.use_mathtext'] = use_mathtext
    plt.rcParams['text.usetex'] = use_tes
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    if fontsize is not None:
        plt.rcParams['axes.labelsize'] = fontsize
        plt.rcParams['xtick.labelsize'] = fontsize
        plt.rcParams['ytick.labelsize'] = fontsize
        plt.rcParams['legend.fontsize'] = fontsize
        plt.rcParams['font.size'] = fontsize
    
    if figsize is not None:
        plt.rcParams['figure.figsize'] = figsize
        
    if pdf_fonttype is not None:
        plt.rcParams['pdf.fonttype'] = pdf_fonttype

    try:
        yield
    finally:
        plt.rcdefaults()

def optimize_3d_plot(ax, zratio=None, shape=(1.0, 1.0, 0.25), zaxis_position='bottom-left', 
                     show_grid=True, grid_color='#bebebe', background_color='white', axis_color=None, grid_which='major',
                     show_xy_grid=True, show_xz_grid=True, show_yz_grid=True):
    """
    Optimize the 3D plot appearance.

    Parameters:
    - ax: The 3D axis to optimize.
    - zratio (float): Ratio for the z-axis (default is None).
    - shape (tuple): Shape of the 3D plot (default is (1.0, 1.0, 0.25)).
    - zaxis_position (str): Position of the z-axis ('bottom-left', 'top-right', default is 'bottom-left').
    - show_grid (bool): Whether to show grid lines (default is True).
    - grid_color (str): Color of the grid lines (default is 'gray').
    - background_color (str): Background color of the plot (default is 'white').
    - axis_color (str): Color of the axes (default is None).
    - grid_which (str): Which grid lines to draw ('major', 'minor', 'both', default is 'major').
    - show_xy_grid (bool): Whether to show grid lines on the xy plane (default is True).
    - show_xz_grid (bool): Whether to show grid lines on the xz plane (default is True).
    - show_yz_grid (bool): Whether to show grid lines on the yz plane (default is True).

    Returns:
    - None
    """
    # Set Z axis ratio
    if zratio is not None:
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), 
                                     np.diag([1.0, 1.0, zratio, 1]))
    else:
        # Manually adjust the aspect ratio of the axes
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

    # Set background color
    if background_color is None:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Hide the x pane
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Hide the y pane
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Hide the z pane
    else:
        ax.xaxis.set_pane_color(background_color)
        ax.yaxis.set_pane_color(background_color)
        ax.zaxis.set_pane_color(background_color)

    if axis_color is None:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Hide the x pane
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Hide the y pane
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Hide the z pane
    else:
        ax.xaxis.set_pane_color(axis_color)
        ax.yaxis.set_pane_color(axis_color)
        ax.zaxis.set_pane_color(axis_color)

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
                           faultEdges_linewidth=1.0, suffix='', show_grid=True, grid_color='#bebebe',
                           background_color='white', axis_color=None, zaxis_position='bottom-left', figname=None,
                           show_xy_grid=True, show_xz_grid=True, show_yz_grid=True):
    """
    Plot the slip distribution of a fault.

    Parameters:
    - fault: The fault object to plot.
    - add_faults: Additional faults to plot the trace of the fault (default is None).
    - slip (str): Type of slip to plot (default is 'total').
    - cmap (str): Colormap to use (default is 'precip3_16lev_change.cpt').
    - norm: Normalization for the colormap (default is None).
    - figsize (tuple): Size of the figure and map (default is (None, None)).
    - drawCoastlines (bool): Whether to draw coastlines (default is False).
    - plot_on_2d (bool): Whether to plot on a 2D map (default is True).
    - method (str): Method for getting the colormap (default is 'cdict').
    - N (int): Number of colors in the colormap (default is None).
    - cbaxis (list): Colorbar axis position (default is [0.1, 0.2, 0.1, 0.02]).
    - cblabel (str): Label for the colorbar (default is '').
    - show (bool): Whether to show the plot (default is True).
    - savefig (bool): Whether to save the figure (default is False).
    - ftype (str): File type for saving the figure (default is 'pdf').
    - dpi (int): Dots per inch for the saved figure (default is 600).
    - bbox_inches: Bounding box in inches for saving the figure (default is None).
    - remove_direction_labels (bool): If True, remove E, N, S, W from axis labels (default is False).
    - cbticks (list): List of ticks to set on the colorbar (default is None).
    - cblinewidth (int): Width of the colorbar label border and tick lines (default is 1).
    - cbfontsize (int): Font size of the colorbar label (default is 10).
    - cb_label_side (str): Position of the label relative to the ticks ('opposite' or 'same', default is 'opposite').
    - map_cbaxis: Axis for the colorbar on the map plot, default is None.
    - style (list): Style for the plot (default is ['notebook']).
    - xlabelpad, ylabelpad, zlabelpad (float): Padding for the axis labels (default is None).
    - xtickpad, ytickpad, ztickpad (float): Padding for the axis ticks (default is None).
    - elevation, azimuth (float): Elevation and azimuth angles for the 3D plot (default is None).
    - shape (tuple): Shape of the 3D plot (default is (1.0, 1.0, 1.0)).
    - zratio (float): Ratio for the z-axis (default is None).
    - plotTrace (bool): Whether to plot the fault trace (default is True).
    - depth (float): Depth for the z-axis (default is None).
    - zticks (list): Ticks for the z-axis (default is None).
    - map_expand (float): Expansion factor for the map (default is 0.2).
    - fault_expand (float): Expansion factor for the fault (default is 0.1).
    - plot_faultEdges (bool): Whether to plot the fault edges (default is False).
    - faultEdges_color (str): Color for the fault edges (default is 'k').
    - faultEdges_linewidth (float): Line width for the fault edges (default is 1.0).
    - suffix (str): Suffix for the saved figure filename (default is '').
    - show_grid (bool): Whether to show grid lines (default is True).
    - grid_color (str): Color of the grid lines (default is 'gray').
    - background_color (str): Background color of the plot (default is 'white').
    - axis_color (str): Color of the axes (default is None).
    - zaxis_position (str): Position of the z-axis (bottom-left, top-right) (default is 'bottom-left').
    - figname (str): Name of the figure (default is None).

    Returns:
    - None
    """
    from .getcpt import get_cpt
    import cmcrameri
    from matplotlib.ticker import FuncFormatter

    if isinstance(cmap, str) and cmap.endswith('.cpt'):
        cmap = get_cpt.get_cmap(cmap, method, N)

    cbfontsize = cbfontsize if cbfontsize is not None else plt.rcParams['axes.labelsize']
    cblinewidth = cblinewidth if cblinewidth is not None else plt.rcParams['axes.linewidth']
    with sci_plot_style(style=style):
        fault.plot(drawCoastlines=drawCoastlines, slip=slip, cmap=cmap, norm=norm, savefig=False,
                   ftype=ftype, dpi=dpi, bbox_inches=bbox_inches, plot_on_2d=plot_on_2d,
                   figsize=figsize, cbaxis=cbaxis, cblabel=cblabel, show=False, expand=map_expand,
                   remove_direction_labels=remove_direction_labels, cbticks=cbticks,
                   cblinewidth=cblinewidth, cbfontsize=cbfontsize, cb_label_side=cb_label_side, map_cbaxis=map_cbaxis)
        ax = fault.slipfig.faille
        fig = fault.slipfig
        name = fault.name

        # Only for triangular faults at current stage
        if plot_faultEdges and add_faults is not None:
            for ifault in add_faults:
                ifault.find_fault_fouredge_vertices(refind=True)
                for edgename in ifault.edge_vertices:
                    edge = ifault.edge_vertices[edgename]
                    x, y, z = edge[:, 0], edge[:, 1], -edge[:, 2]
                    lon, lat = fault.xy2ll(x, y)
                    ax.plot(lon, lat, z, color=faultEdges_color, linewidth=faultEdges_linewidth)

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
            # Get lons lats
            lon = np.unique(np.array([p[:, 0] for p in fault.patchll]))
            lat = np.unique(np.array([p[:, 1] for p in fault.patchll]))
            lonmin, lonmax = lon.min(), lon.max()
            latmin, latmax = lat.min(), lat.max()
            ax.set_xlim(lonmin - fault_expand, lonmax + fault_expand)
            ax.set_ylim(latmin - fault_expand, latmax + fault_expand)
        ax.zaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{abs(val)}'))

        # Set View, reference to csi.geodeticplot.set_view
        if elevation is not None and azimuth is not None:
            ax.view_init(elev=elevation, azim=azimuth)
        else:
            strike = np.mean(fault.getStrikes() * 180 / np.pi)
            dip = np.mean(fault.getDips() * 180 / np.pi)
            azimuth = -strike
            elevation = 90 - dip + 10
            ax.view_init(elev=elevation, azim=azimuth)

        # Set 3D plot shape
        optimize_3d_plot(ax, shape=shape, zratio=zratio, zaxis_position=zaxis_position,
                         show_grid=show_grid, grid_color=grid_color,
                         background_color=background_color, axis_color=axis_color,
                         show_xy_grid=show_xy_grid, show_xz_grid=show_xz_grid, show_yz_grid=show_yz_grid)

        if savefig:
            saveFig = ['fault']
            if figname is None:
                prefix = name.replace(' ', '_')
                suffix = f'_{suffix}' if suffix != '' else ''
                figname = prefix + '{0}_{1}'.format(suffix, slip)
            if plot_on_2d:
                saveFig.append('map')
            fig.savefig(figname, ftype=ftype, dpi=dpi, bbox_inches=bbox_inches, saveFig=saveFig)

        if show:
            showFig = ['fault']
            if plot_on_2d:
                showFig.append('map')
            fig.show(showFig=showFig)
            plt.show()