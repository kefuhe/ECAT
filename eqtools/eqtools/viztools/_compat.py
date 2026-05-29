"""
_compat.py — Backward-compatible wrappers for eqtools.viztools.

Public API
----------
sci_plot_style       : Legacy context manager (wraps PlotStyle)
set_plot_style       : Legacy (fig, ax) creator with style applied
update_style_library : Refresh matplotlib style library
"""

import warnings
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt


def _map_legacy_style_to_preset(style, serif: bool) -> Union[str, List[str]]:
    """Map the old sci_plot_style ``style`` argument to a preset name."""
    from ._core import _PRESET_REGISTRY
    if style is None:
        return 'science-serif' if serif else 'science'
    if isinstance(style, str):
        if style in _PRESET_REGISTRY:
            return style
        return style
    if isinstance(style, (list, tuple)):
        style_lower = {s.lower().replace('-', '') for s in style}
        if 'science' in style_lower:
            return 'science-serif' if (serif or 'serif' in style_lower) else 'science'
        return list(style)
    return 'science'


def sci_plot_style(
    style: Optional[Union[str, List[str]]] = None,
    legend_frame: bool = False,
    use_tes: bool = False,
    use_tex: bool = False,
    use_mathtext: bool = False,
    serif: bool = False,
    fontsize: Optional[float] = None,
    figsize: Optional[Union[str, float, tuple]] = None,
    figsize_unit: str = 'inch',
    figsize_fraction: float = 1.0,
    figsize_aspect: float = 0.75,
    figsize_height: Optional[float] = None,
    pdf_fonttype: Optional[int] = None,
) -> 'PlotStyle':
    """Context manager for scientific plotting style (backward-compatible wrapper).

    Backward-compatible wrapper around :class:`PlotStyle`.  Existing code
    using ``with sci_plot_style(...):`` continues to work unchanged.

    For new code, prefer :class:`PlotStyle` directly::

        with PlotStyle('science', figsize='single', fontsize=8):
            fig, ax = plt.subplots()

    Parameters
    ----------
    style : str or list, optional
        matplotlib style name(s) or a preset name.
    legend_frame : bool
    use_tes : bool
        Deprecated typo for ``use_tex``.  Emits a :class:`DeprecationWarning`.
    use_tex : bool
        Enable TeX rendering (``text.usetex=True``).
    use_mathtext : bool
        Enable mathtext (``axes.formatter.use_mathtext=True``).
    serif : bool
        Switch to the serif preset when *style* is not given.
    fontsize : float, optional
    figsize, figsize_unit, figsize_fraction, figsize_aspect, figsize_height
        Passed to :func:`publication_figsize`.
    pdf_fonttype : int, optional
    """
    from ._core import PlotStyle

    # P4-6: fix use_tes typo — emit DeprecationWarning, honour the intent
    if use_tes and not use_tex:
        warnings.warn(
            "'use_tes' is a misspelling; use 'use_tex=True' instead.",
            DeprecationWarning, stacklevel=2,
        )
    _do_tex = use_tex or use_tes

    preset = _map_legacy_style_to_preset(style, serif)
    extra_rc: Dict = {}
    if _do_tex:
        # Old interface: write directly to rcparams, no auto preamble injection
        extra_rc['text.usetex'] = True
    if use_mathtext:
        extra_rc['axes.formatter.use_mathtext'] = True
        extra_rc['mathtext.fontset'] = 'cm'

    return PlotStyle(
        preset,
        figsize=figsize,
        figsize_unit=figsize_unit,
        figsize_fraction=figsize_fraction,
        figsize_aspect=figsize_aspect,
        figsize_height=figsize_height,
        fontsize=fontsize,
        legend_frame=legend_frame,
        pdf_fonttype=pdf_fonttype,
        rcparams=extra_rc,
    )


def set_plot_style(
    style: Union[str, List[str]] = 'science',
    figsize: tuple = (3.5, 2.8),
    use_degree: bool = False,
    equal_aspect: bool = False,
) -> tuple:
    """Create a (fig, ax) pair with the given style applied (legacy helper).

    For new code, use :class:`PlotStyle` as a context manager and call
    ``plt.subplots()`` yourself.

    Parameters
    ----------
    style : str or list
        Preset or mplstyle name(s).
    figsize : tuple
        Figure size in inches.
    use_degree : bool
        Add degree symbol to both axes.
    equal_aspect : bool
        Set equal aspect ratio.

    Returns
    -------
    fig, ax
    """
    from ._core import PlotStyle
    from ._formatters import set_degree_formatter

    with PlotStyle(style, figsize=figsize):
        fig, ax = plt.subplots(figsize=figsize)
    if use_degree:
        set_degree_formatter(ax)
    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')
    return fig, ax


def update_style_library() -> None:
    """Refresh matplotlib's style library with both scienceplots and package styles."""
    from ._core import _register_package_styles, HAS_SCIENCEPLOTS, register_science_styles
    if HAS_SCIENCEPLOTS:
        register_science_styles()
    _register_package_styles()
