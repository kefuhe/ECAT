"""
eqtools.plottools — Deprecated compatibility entry for plotting functions.

This module is kept for backward compatibility only.
Import general plotting from ecat_viz instead::

    from ecat_viz import PlotStyle, bake_text_fonts, save_fig

Fault, dip and slip diagnostics remain in eqtools.viztools.
"""
import warnings

warnings.warn(
    "eqtools.plottools is deprecated and will be removed in a future version. "
    "Use ecat_viz for general plotting and eqtools.viztools for fault diagnostics:\n"
    "    from ecat_viz import PlotStyle, bake_text_fonts, save_fig",
    DeprecationWarning,
    stacklevel=2,
)

from . import viztools as _viztools
__all__ = _viztools.__all__

def __getattr__(name):
    if name in __all__:
        return getattr(_viztools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | set(__all__))
