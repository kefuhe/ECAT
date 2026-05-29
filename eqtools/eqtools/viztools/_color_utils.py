"""
_color_utils.py — Color-cycle utilities for eqtools.viztools.

Public API
----------
get_color_cycle : Return color list from a preset or current rcParams
"""

from typing import List, Optional

import matplotlib as mpl


def get_color_cycle(preset: Optional[str] = None) -> List[str]:
    """Return the color list from a style preset (or the current rcParams).

    Parameters
    ----------
    preset : str, optional
        Registered preset name.  When *None* the current
        ``axes.prop_cycle`` rcParam is read directly.

    Returns
    -------
    list[str]
        List of hex color strings (entries without a 'color' key are skipped).

    Examples
    --------
    >>> get_color_cycle()                    # current active colors
    >>> get_color_cycle('colors-bright')     # bright palette colors
    """
    if preset is None:
        cycle = mpl.rcParams.get('axes.prop_cycle', None)
        if cycle is None:
            return []
        # Filter entries that actually have a 'color' key (P4-8 fix)
        return [c['color'] for c in cycle if 'color' in c]

    # Resolve via a temporary PlotStyle instance (does not modify global state)
    from ._core import PlotStyle, _ensure_initialized
    _ensure_initialized()
    tmp = PlotStyle(preset)
    rcp = tmp._build_final_rcparams()
    cycle = rcp.get('axes.prop_cycle', None)
    if cycle is None:
        return []
    return [c['color'] for c in cycle if 'color' in c]
