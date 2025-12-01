from os import listdir
from os.path import isdir, join
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scienceplots
from matplotlib import cm
import matplotlib as mpl
from contextlib import contextmanager


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
    # Preserve the scienceplots style library
    scienceplots_styles = plt.style.library.copy()
    # Merge the scienceplots styles into the current style library
    plt.style.library.update(scienceplots_styles)
    # Refresh the list of available styles
    plt.style.available[:] = sorted(plt.style.library.keys())

def publication_figsize(column='single', fraction=1.0, aspect=0.75, height=None, unit='inch'):
    """
    Return figure size (width, height) in inches for common publication column widths.

    Parameters
    ----------
    column : str, float, or tuple
        - Predefined column names: 'single', 'double', 'nature', 'ieee', 'ieee_double', 'a4'.
        - Custom numeric width (interpreted as `unit`).
        - Direct (width, height) tuple (interpreted as `unit`, will be converted to inches).
    fraction : float, optional
        Fraction of the column width (0..1), default is 1.0.
    aspect : float, optional
        Height-to-width ratio (used when `height` is None), default is 0.75.
    height : float, optional
        Explicit height in `unit`. Overrides `aspect` if given.
    unit : str, optional
        Unit for input/output: 'inch' (default) or 'cm'.

    Returns
    -------
    tuple of float
        Figure size (width, height) in inches.

    Examples
    --------
    >>> publication_figsize('single')
    (3.5, 2.625)
    >>> publication_figsize('double', fraction=0.8)
    (5.76, 4.32)
    >>> publication_figsize(10, unit='cm')
    (3.937007874015748, 2.952755905511811)
    >>> publication_figsize((10, 8), unit='cm')
    (3.937007874015748, 3.1496062992125984)
    """
    # Predefined column widths in inches
    widths_inch = {
        'single': 3.5,
        'double': 7.2,
        'nature': 3.42,
        'ieee': 3.5,
        'ieee_double': 7.16,
        'a4': 8.27
    }

    cm_to_inch = 1 / 2.54

    # If column is already a (w, h) tuple/list
    if isinstance(column, (tuple, list)) and len(column) == 2:
        w_raw, h_raw = float(column[0]), float(column[1])
        if unit == 'cm':
            return (w_raw * cm_to_inch, h_raw * cm_to_inch)
        else:
            return (w_raw, h_raw)

    # Resolve width from column specification
    if isinstance(column, (int, float)):
        w = float(column)
    else:
        w = widths_inch.get(str(column).lower(), widths_inch['single'])

    # Convert width if specified in cm
    if unit == 'cm':
        w = w * cm_to_inch

    w = w * float(fraction)

    # Resolve height
    if height is not None:
        h = float(height)
        if unit == 'cm':
            h = h * cm_to_inch
    else:
        h = w * float(aspect)

    return (w, h)

@contextmanager
def sci_plot_style(style=['science', 'no-latex'], legend_frame=False, use_tes=False, 
                   use_mathtext=False, serif=False, fontsize=None, figsize=None, 
                   figsize_unit='inch', figsize_fraction=1.0, figsize_aspect=0.75, 
                   figsize_height=None, pdf_fonttype=None):
    """
    Context manager to set scientific plotting style.

    Parameters
    ----------
    style : list of str, optional
        Matplotlib style names to apply, default is ['science', 'no-latex'].
    legend_frame : bool, optional
        Whether to add frame to legends, default is False.
    use_tes : bool, optional
        Whether to use TeX for text rendering, default is False.
    use_mathtext : bool, optional
        Whether to use mathtext for rendering, default is False.
    serif : bool, optional
        Whether to use serif fonts, default is False.
    fontsize : int, optional
        Font size for text elements, default is None (use rcParams).
    figsize : str, float, or tuple, optional
        Figure size specification:
        - str: predefined column width name ('single', 'double', 'nature', 'ieee', 'ieee_double', 'a4')
        - float: custom width (height computed via figsize_aspect)
        - tuple: (width, height) in figsize_unit
        Default is None (use rcParams default).
    figsize_unit : str, optional
        Unit for figsize when numeric: 'inch' (default) or 'cm'.
    figsize_fraction : float, optional
        Fraction of column width (0..1) when figsize is a string, default is 1.0.
    figsize_aspect : float, optional
        Height-to-width ratio when figsize is a single number or string, default is 0.75.
    figsize_height : float, optional
        Explicit height in figsize_unit. Overrides figsize_aspect if given.
    pdf_fonttype : int, optional
        Set pdf.fonttype in matplotlib:
        - 3 (default): Type 3 fonts
        - 42: TrueType fonts (recommended for editable text in Illustrator)

    Yields
    ------
    None

    Examples
    --------
    >>> # Use predefined single column width
    >>> with sci_plot_style(figsize='single'):
    ...     fig, ax = plt.subplots()
    
    >>> # Use double column width with 80% width
    >>> with sci_plot_style(figsize='double', figsize_fraction=0.8):
    ...     fig, ax = plt.subplots()
    
    >>> # Custom width in cm (height auto-computed)
    >>> with sci_plot_style(figsize=15, figsize_unit='cm'):
    ...     fig, ax = plt.subplots()
    
    >>> # Direct (width, height) in cm
    >>> with sci_plot_style(figsize=(15, 10), figsize_unit='cm'):
    ...     fig, ax = plt.subplots()
    
    >>> # Custom width with explicit height in inches
    >>> with sci_plot_style(figsize=5.0, figsize_height=4.0):
    ...     fig, ax = plt.subplots()
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
    
    # Parse figsize parameter using publication_figsize
    if figsize is not None:
        resolved_figsize = publication_figsize(
            column=figsize,
            fraction=figsize_fraction,
            aspect=figsize_aspect,
            height=figsize_height,
            unit=figsize_unit
        )
        plt.rcParams['figure.figsize'] = resolved_figsize
        
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
    - fault: The fault object to plot, or a list of fault objects.
    - slip: Type of slip to plot or slip array(s). Can be:
        * str: 'total', 'strikeslip', 'dipslip', etc. (default is 'total')
        * 1D array (n,): slip for single fault
        * 2D array (1, n): slip for single fault
        * List of arrays: slip for multiple faults (when fault is a list)
    - add_faults: Additional faults to plot the trace of the fault (default is None).
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
    - show_xy_grid (bool): Whether to show grid lines on the xy plane (default is True).
    - show_xz_grid (bool): Whether to show grid lines on the xz plane (default is True).
    - show_yz_grid (bool): Whether to show grid lines on the yz plane (default is True).

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
                # Plot the whole thing
                from csi.geodeticplot import geodeticplot as geoplt
                # 用所有fault.patchll的经纬度范围来计算边界
                lon_min = min([p[:, 0].min() for f in fault for p in f.patchll])
                lon_max = max([p[:, 0].max() for f in fault for p in f.patchll])
                lat_min = min([p[:, 1].min() for f in fault for p in f.patchll])
                lat_max = max([p[:, 1].max() for f in fault for p in f.patchll])
                depth_max = max([p[:, 2].max() for f in fault for p in f.patchll])
                gp = geoplt(lon_min, lat_min, lon_max, lat_max, figsize=figsize)
                plot_colorbar = True
                for ifault in fault:
                    # Plot the faults
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
                # 用所有fault.patchll的经纬度范围来计算边界
                faults_list = fault if isinstance(fault, list) else [fault]
                lon_min = min([p[:, 0].min() for f in faults_list for p in f.patchll])
                lon_max = max([p[:, 0].max() for f in faults_list for p in f.patchll])
                lat_min = min([p[:, 1].min() for f in faults_list for p in f.patchll])
                lat_max = max([p[:, 1].max() for f in faults_list for p in f.patchll])
                ax.set_xlim(lon_min - fault_expand, lon_max + fault_expand)
                ax.set_ylim(lat_min - fault_expand, lat_max + fault_expand)
            ax.zaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{abs(val)}'))
    
            # Set View, reference to csi.geodeticplot.set_view
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
                    prefix = name.replace(' ', '_')
                    suffix = f'_{suffix}' if suffix != '' else ''
                    figname = prefix + '{0}_{1}'.format(suffix, slip_type)
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