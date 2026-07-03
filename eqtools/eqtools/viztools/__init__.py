"""
eqtools.viztools — Matplotlib style management for publication-quality figures.

This package provides PlotStyle and related utilities for managing matplotlib
styles, fonts, and figure sizes in a context-manager / persistent-apply /
decorator pattern.

Quick start
-----------
>>> from eqtools.viztools import PlotStyle, finish_fig
>>> with PlotStyle('science', figsize='single', fontsize=8):
...     fig, ax = plt.subplots()
...     ax.plot(x, y)
...     finish_fig(fig, 'figure.pdf', show=True, dpi=600)

One-step figure creation::

    fig, ax = PlotStyle('science', figsize='single', fontsize=8).subplots()

For the full API reference, see the eqtools.viztools documentation.
"""

# Version compatibility check
import warnings


def _check_matplotlib_version():
    """Check matplotlib version compatibility."""
    try:
        import matplotlib
        from ._constants import MIN_MATPLOTLIB_VERSION

        # Simple version comparison (works for most cases)
        current_version = matplotlib.__version__
        min_version = MIN_MATPLOTLIB_VERSION

        # Parse versions
        def parse_version(v):
            return tuple(int(x) for x in v.split('.')[:3])

        try:
            current_tuple = parse_version(current_version.split('+')[0])  # Remove dev suffixes
            min_tuple = parse_version(min_version)

            if current_tuple < min_tuple:
                warnings.warn(
                    f"eqtools.viztools requires matplotlib {min_version} or later, "
                    f"but version {current_version} is installed. "
                    f"Some features may not work correctly. "
                    f"Please upgrade: pip install --upgrade matplotlib",
                    UserWarning,
                    stacklevel=3
                )
        except (ValueError, AttributeError):
            # If version parsing fails, skip the check
            pass
    except ImportError:
        # matplotlib not installed (should not happen if we got here)
        pass


_check_matplotlib_version()

# Import public API
from ._core import (
    PlotStyle,
    register_preset,
    unregister_preset,
    list_presets,
)
from ._constants import Presets
from ._font_utils import list_chinese_fonts, bake_text_fonts
from ._style_utils import (
    publication_figsize,
    register_column_width,
    save_column_width,
    list_column_widths,
    save_fig,
    cap_interactive_dpi,
    show_fig,
    finish_fig,
)
from ._formatters import (
    DegreeFormatter,
    LatFormatter,
    LonFormatter,
    DMSFormatter,
    set_degree_formatter,
)
from ._color_utils import get_color_cycle
from ._compat import sci_plot_style, set_plot_style, update_style_library
from ._registry import _registry
from .raster import (
    plot_dataarray,
    plot_geotiff,
    plot_netcdf_grid,
    plot_raster,
    raster_limits,
)


def register_style_directory(path):
    """Register a custom directory containing .mplstyle files.

    Scans the directory for .mplstyle files and registers them as user presets.
    Styles can then be used with PlotStyle like built-in presets.

    Parameters
    ----------
    path : str or Path
        Path to directory containing .mplstyle files

    Raises
    ------
    FileNotFoundError
        If the directory does not exist
    ValueError
        If path is not a directory

    Examples
    --------
    >>> from eqtools.viztools import register_style_directory, PlotStyle
    >>> register_style_directory('~/my_styles')
    >>> # Use styles from that directory
    >>> with PlotStyle('my_custom_style'):
    ...     fig, ax = plt.subplots()

    Notes
    -----
    - Files starting with underscore (_) are ignored
    - Registered styles persist for the Python session
    - Styles from custom directories can be listed with list_presets()
    """
    _registry.register_style_directory(path)

# viz_3d: lazy import (depends on csi.geodeticplot + cmcrameri — optional)
try:
    from .viz_3d import optimize_3d_plot, plot_slip_distribution
    _HAS_VIZ3D = True
except ImportError:
    _HAS_VIZ3D = False

__all__ = [
    # Core style management
    'PlotStyle',
    'Presets',
    'register_preset',
    'unregister_preset',
    'list_presets',
    'register_style_directory',
    # Font utilities
    'list_chinese_fonts',
    'bake_text_fonts',
    # Figure size & saving
    'publication_figsize',
    'register_column_width',
    'save_column_width',
    'list_column_widths',
    'save_fig',
    'cap_interactive_dpi',
    'show_fig',
    'finish_fig',
    # Tick formatters
    'DegreeFormatter',
    'LatFormatter',
    'LonFormatter',
    'DMSFormatter',
    'set_degree_formatter',
    # Color utilities
    'get_color_cycle',
    # Raster plotting
    'plot_raster',
    'plot_dataarray',
    'plot_geotiff',
    'plot_netcdf_grid',
    'raster_limits',
    # Backward-compatible wrappers
    'sci_plot_style',
    'set_plot_style',
    'update_style_library',
]
if _HAS_VIZ3D:
    __all__ += ['optimize_3d_plot', 'plot_slip_distribution']
