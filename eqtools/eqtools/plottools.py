"""
eqtools.plottools — Deprecated. API has moved to eqtools.viztools.

This module is kept for backward compatibility only.
Import from eqtools.viztools instead::

    from eqtools.viztools import PlotStyle, bake_text_fonts, save_fig
"""
import warnings

warnings.warn(
    "eqtools.plottools is deprecated and will be removed in a future version. "
    "Use eqtools.viztools instead:\n"
    "    from eqtools.viztools import PlotStyle, bake_text_fonts, save_fig",
    DeprecationWarning,
    stacklevel=2,
)

from .viztools import *        # noqa: F401, F403
from .viztools import __all__  # noqa: F401
