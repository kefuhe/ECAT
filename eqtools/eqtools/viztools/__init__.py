"""Compatible plotting exports; domain functions remain in eqtools."""
from importlib import import_module

from ecat_viz import (
    PlotStyle,
    Presets,
    register_preset,
    unregister_preset,
    list_presets,
    register_style_directory,
    list_chinese_fonts,
    bake_text_fonts,
    publication_figsize,
    register_column_width,
    save_column_width,
    list_column_widths,
    save_fig,
    normalize_image_format,
    cap_interactive_dpi,
    show_fig,
    finish_fig,
    DegreeFormatter,
    LatFormatter,
    LonFormatter,
    DMSFormatter,
    set_degree_formatter,
    get_color_cycle,
    plot_raster,
    plot_dataarray,
    plot_geotiff,
    plot_netcdf_grid,
    raster_limits,
    sci_plot_style,
    set_plot_style,
    update_style_library,
    optimize_3d_plot,
)

_DOMAIN = {'plot_fault_boundary_diagnostics': '.fault_boundary', 'plot_dip_profile_diagnostics': '.dip_profile', 'plot_dip_transition_analysis': '.dip_profile', 'plot_slip_distribution': '.viz_3d'}

def __getattr__(name):
    if name in _DOMAIN:
        value = getattr(import_module(_DOMAIN[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = ['PlotStyle', 'Presets', 'register_preset', 'unregister_preset', 'list_presets', 'register_style_directory', 'list_chinese_fonts', 'bake_text_fonts', 'publication_figsize', 'register_column_width', 'save_column_width', 'list_column_widths', 'save_fig', 'normalize_image_format', 'cap_interactive_dpi', 'show_fig', 'finish_fig', 'DegreeFormatter', 'LatFormatter', 'LonFormatter', 'DMSFormatter', 'set_degree_formatter', 'get_color_cycle', 'plot_raster', 'plot_dataarray', 'plot_geotiff', 'plot_netcdf_grid', 'raster_limits', 'sci_plot_style', 'set_plot_style', 'update_style_library', 'optimize_3d_plot', 'plot_fault_boundary_diagnostics', 'plot_dip_profile_diagnostics', 'plot_dip_transition_analysis', 'plot_slip_distribution']
