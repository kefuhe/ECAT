"""
_style_utils.py — Figure size utilities, column-width registry, save, and show.

Public API
----------
register_column_width : Register a named publication column width
publication_figsize   : Return figure size (w, h) in inches for publication columns
save_fig              : Save figure to one or more file formats
cap_interactive_dpi   : Cap interactive figure dpi without changing saved output
show_fig              : Show a figure after capping interactive dpi
finish_fig            : Common save/show helper for library plotting functions
normalize_image_format: Normalize and validate a requested image format
"""

import json
import os
import tempfile
import warnings
from numbers import Real
from pathlib import Path
from typing import Dict, Optional

# Import the centralized registry
from ._registry import _registry, _positive_finite, _validate_column_width
from ._core import _ensure_initialized


def normalize_image_format(file_type: str) -> str:
    """Return a canonical, supported image format name.

    The returned value is lowercase and has no leading dot.  Keeping this
    validation in the plotting layer gives every high-level ECAT figure
    product the same format contract.
    """
    from ._constants import KNOWN_IMAGE_FORMATS

    normalized = str(file_type).strip().lower().lstrip('.')
    if normalized not in KNOWN_IMAGE_FORMATS:
        raise ValueError(
            f"Unsupported file_type: {normalized}. "
            f"Supported formats: {sorted(KNOWN_IMAGE_FORMATS)}"
        )
    return normalized

# --------------------------------------------------------------------------
# Column-width registry (now managed by _registry)
# --------------------------------------------------------------------------

def register_column_width(name: str, width_inch: float) -> None:
    """Register a named column width for use with :func:`publication_figsize`.

    Parameters
    ----------
    name : str
        Case-insensitive key, e.g. ``'agu_single'``, ``'copernicus'``.
    width_inch : float
        Finite positive column width **in inches**.

    Example
    -------
    >>> register_column_width('agu_single', 3.37)
    >>> register_column_width('copernicus', 3.15)
    >>> publication_figsize('agu_single')           # (3.37, 2.5275)
    """
    name, width_inch = _validate_column_width(name, width_inch)
    _ensure_initialized()
    _registry.register_column_width(name, width_inch)


def _load_user_config() -> None:
    """Load user column-width overrides from the first existing config file.

    Search order (highest priority first):
    1. ``~/.config/eqtools/viztools.json``   (retained compatibility path)
    2. ``~/.config/eqtools/plottools.json``
    3. ``~/.config/statutils/plottools.json``  (legacy, emits DeprecationWarning)
    4. ``~/.plottools.json``                   (legacy, emits DeprecationWarning)

    No file is a no-op; malformed configuration emits a warning.
    """
    new_paths = [
        Path.home() / '.config' / 'eqtools' / 'viztools.json',
        Path.home() / '.config' / 'eqtools' / 'plottools.json',
    ]
    legacy_paths = [
        Path.home() / '.config' / 'statutils' / 'plottools.json',
        Path.home() / '.plottools.json',
    ]

    for cfg_path in new_paths:
        if cfg_path.exists():
            _load_config_file(cfg_path, legacy=False)
            return

    for cfg_path in legacy_paths:
        if cfg_path.exists():
            warnings.warn(
                f"ecat_viz: config file '{cfg_path}' is at a legacy path. "
                f"Move it to ~/.config/eqtools/viztools.json to suppress this warning.",
                DeprecationWarning, stacklevel=3,
            )
            _load_config_file(cfg_path, legacy=True)
            return


def _validated_config_widths(data):
    """Validate the complete width section before applying any entries."""
    if not isinstance(data, dict):
        raise ValueError("Column-width configuration must be a JSON object.")
    widths = data.get('column_widths', {})
    if not isinstance(widths, dict):
        raise ValueError("column_widths must be a JSON object.")
    normalized = {}
    for name, value in widths.items():
        key, width = _validate_column_width(name, value)
        if key in normalized and normalized[key] != width:
            raise ValueError(f"Column-width conflict for case-insensitive name '{key}'.")
        normalized[key] = width
    return normalized


def _load_config_file(cfg_path: Path, legacy: bool = False) -> None:
    """Read validated overrides without re-entering public initialization."""
    try:
        data = json.loads(cfg_path.read_text(encoding='utf-8'))
        widths = _validated_config_widths(data)
    except (OSError, ValueError) as exc:
        warnings.warn(f"ecat_viz: could not load config {cfg_path}: {exc}")
        return
    for name, width in widths.items():
        _registry.register_column_width(name, width)


def save_column_width(name: str, width_inch: float,
                      config_path: Optional[Path] = None) -> None:
    """Save a column width to the user configuration file.

    Parameters
    ----------
    name : str
        Column width name (case-insensitive).
    width_inch : float
        Width in inches.
    config_path : Path, optional
        Configuration file path. If None, uses the default path:
        ``~/.config/eqtools/viztools.json``

    Example
    -------
    >>> save_column_width('my_journal', 3.25)
    >>> publication_figsize('my_journal')  # Now available in future sessions
    (3.25, 2.4375)

    Notes
    -----
    - The column width is also registered in the current session
    - Creates the config directory if it doesn't exist
    - Preserves unrelated configuration entries and normalizes width names
    - Conflicting case aliases raise before changing the file or registry
    """
    name, width_inch = _validate_column_width(name, width_inch)
    if config_path is None:
        config_path = Path.home() / '.config' / 'eqtools' / 'viztools.json'
    config_path = Path(config_path)
    # Parse and validate before creating directories or changing the file.
    data = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    data['column_widths'] = _validated_config_widths(data)
    data['column_widths'][name] = width_inch
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    _ensure_initialized()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=config_path.parent, prefix='.' + config_path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, config_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    # A failed write must not register an unsaved override.
    _registry.register_column_width(name, width_inch)


def list_column_widths() -> Dict[str, float]:
    """List all registered column widths.

    Returns
    -------
    dict
        Mapping of column width names to widths (in inches).

    Example
    -------
    >>> widths = list_column_widths()
    >>> for name, width in sorted(widths.items()):
    ...     print(f'{name:15s} {width:.2f} inch')
    single          3.50 inch
    double          7.20 inch
    nature          3.42 inch
    ...
    """
    _ensure_initialized()
    return _registry.list_column_widths()


def publication_figsize(column='single', fraction=1.0, aspect=0.75, height=None, unit='inch'):
    """Return figure size (width, height) in inches for common publication column widths.

    Parameters
    ----------
    column : str, float, or tuple
        - Registered names: ``'single'``, ``'double'``, ``'nature'``,
          ``'ieee'``, ``'ieee_double'``, ``'a4'``, plus any name added via
          :func:`register_column_width`.
        - Custom numeric width (interpreted as *unit*).
        - ``(width, height)`` tuple (interpreted as *unit*).
    fraction : float
        Positive scale factor for the column width; ignored for a size tuple.
    aspect : float
        Height-to-width ratio when *height* is None.
    height : float, optional
        Explicit height in *unit*; overrides *aspect*.
    unit : {'inch', 'cm'}

    Returns
    -------
    tuple of float
        ``(width, height)`` in inches.

    Examples
    --------
    >>> publication_figsize('single')
    (3.5, 2.625)
    >>> publication_figsize('double', fraction=0.8)
    (5.76, 4.32)
    >>> publication_figsize(10, unit='cm')
    (3.937..., 2.952...)
    >>> publication_figsize((10, 8), unit='cm')
    (3.937..., 3.149...)
    """
    if unit not in {'inch', 'cm'}:
        raise ValueError("unit must be 'inch' or 'cm'.")
    _ensure_initialized()
    scale = 1 / 2.54 if unit == 'cm' else 1.0

    if isinstance(column, (tuple, list)):
        if len(column) != 2:
            raise ValueError("Figure size must contain exactly width and height.")
        return (_positive_finite(_positive_finite(column[0], 'Width') * scale, 'Width'),
                _positive_finite(_positive_finite(column[1], 'Height') * scale, 'Height'))

    if isinstance(column, Real):
        w = _positive_finite(column, 'Width') * scale
    else:
        w = _registry.get_column_width(str(column).lower())
        if w is None:
            available = list(_registry.list_column_widths().keys())
            warnings.warn(
                f"Column width '{column}' not found. "
                f"Available: {available}. Using 'single'.",
                UserWarning, stacklevel=2,
            )
            w = _registry.get_column_width('single')
    w = _positive_finite(w * _positive_finite(fraction, 'Fraction'), 'Width')
    h = (_positive_finite(height, 'Height') * scale if height is not None
         else w * _positive_finite(aspect, 'Aspect'))
    return w, _positive_finite(h, 'Height')


def save_fig(fig, path: str, fmts=None, dpi: int = 300,
             bbox_inches: str = 'tight', **kwargs) -> None:
    """Save *fig* to one or more file formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str
        Output path.  If it already has a recognised extension (pdf, png,
        svg, eps, jpg, tiff) that format is used; otherwise the extension(s)
        in *fmts* are appended.
    fmts : list[str], optional
        Format list, e.g. ``['pdf', 'png']``.  Ignored when *path* already
        has a format extension.  Defaults to ``['pdf']`` when *path* has no
        extension.
    dpi : int
        Resolution (default 300).
    bbox_inches : str
        Passed to ``fig.savefig`` (default ``'tight'``).
    **kwargs
        Additional keyword arguments forwarded to ``fig.savefig``.

    Examples
    --------
    >>> save_fig(fig, 'result.pdf')
    >>> save_fig(fig, 'result', fmts=['pdf', 'png'])
    >>> save_fig(fig, 'result.pdf', dpi=600, transparent=True)

    Raises
    ------
    TypeError
        If fig is not a matplotlib Figure object.
    ValueError
        If the format is not supported by matplotlib.
    """
    import matplotlib.figure
    from ._constants import KNOWN_IMAGE_FORMATS

    if not isinstance(fig, matplotlib.figure.Figure):
        raise TypeError(
            f"Expected matplotlib.figure.Figure, got {type(fig).__name__}. "
            f"Pass a Figure object from plt.figure() or fig, ax = plt.subplots()."
        )

    _KNOWN_EXTS = KNOWN_IMAGE_FORMATS
    p = Path(path)

    if p.suffix.lstrip('.').lower() in _KNOWN_EXTS:
        # Single file mode with explicit extension
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(str(p), dpi=dpi, bbox_inches=bbox_inches, **kwargs)
            print(f"Saved: {p}")
        except ValueError as e:
            raise ValueError(
                f"Failed to save figure as '{p.suffix}' format.\n"
                f"Matplotlib error: {e}\n"
                f"Supported formats: {', '.join(sorted(_KNOWN_EXTS))}"
            ) from e
    else:
        # Multiple file mode or no extension
        if fmts is None:
            fmts = ['pdf']

        # Validate formats before attempting to save
        invalid_fmts = [f for f in fmts if f.lstrip('.').lower() not in _KNOWN_EXTS]
        if invalid_fmts:
            raise ValueError(
                f"Unsupported format(s): {', '.join(invalid_fmts)}\n"
                f"Supported formats: {', '.join(sorted(_KNOWN_EXTS))}"
            )

        p.parent.mkdir(parents=True, exist_ok=True)
        for fmt in fmts:
            out = p.with_suffix(f'.{fmt.lstrip(".")}')
            try:
                fig.savefig(str(out), dpi=dpi, bbox_inches=bbox_inches, **kwargs)
                print(f"Saved: {out}")
            except ValueError as e:
                warnings.warn(
                    f"Failed to save {out}: {e}. Skipping this format.",
                    UserWarning
                )


def _ensure_figure(fig):
    import matplotlib.figure

    if not isinstance(fig, matplotlib.figure.Figure):
        raise TypeError(
            f"Expected matplotlib.figure.Figure, got {type(fig).__name__}. "
            f"Pass a Figure object from plt.figure() or fig, ax = plt.subplots()."
        )


def cap_interactive_dpi(fig, max_dpi: Optional[float] = 200, *, redraw: bool = True):
    """Cap a figure's interactive dpi while keeping its physical size unchanged.

    This is intended for ``plt.show()`` workflows where the saved figure may use
    a high dpi, such as 300 or 600, but the interactive window should remain
    usable on screen.  It changes only ``fig.dpi`` for the existing figure; use
    :func:`save_fig` or ``fig.savefig(..., dpi=...)`` to control saved-output
    resolution.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to adjust.
    max_dpi : float or None, optional
        Maximum interactive dpi.  ``None`` disables capping.
    redraw : bool, optional
        If True, request a canvas redraw after changing dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure instance.
    """
    _ensure_figure(fig)
    if max_dpi is None:
        return fig
    max_dpi = float(max_dpi)
    if max_dpi <= 0:
        raise ValueError("max_dpi must be positive or None")

    current = float(fig.get_dpi())
    capped = min(current, max_dpi)
    if capped != current:
        fig.set_dpi(capped)
        if redraw and getattr(fig, "canvas", None) is not None:
            try:
                fig.canvas.draw_idle()
            except Exception:
                try:
                    fig.canvas.draw()
                except Exception:
                    pass
    return fig


def show_fig(fig=None, *, max_dpi: Optional[float] = 200, block=None):
    """Show open Matplotlib figures, optionally capping one figure's dpi.

    ``fig`` selects the DPI adjustment target, not the windows shown.
    Display uses pyplot.show(), which may show all open figures.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure to show.  If omitted, ``matplotlib.pyplot.show`` is called for
        all open figures.
    max_dpi : float or None, optional
        Maximum interactive dpi.  ``None`` leaves figure dpi unchanged.
    block : bool, optional
        Forwarded to ``matplotlib.pyplot.show`` when provided.
    """
    import matplotlib.pyplot as plt

    if fig is not None:
        cap_interactive_dpi(fig, max_dpi=max_dpi)
    if block is None:
        plt.show()
    else:
        plt.show(block=block)


def finish_fig(
    fig,
    path=None,
    *,
    save=None,
    show: bool = False,
    dpi: int = 300,
    screen_dpi: Optional[float] = 200,
    fmts=None,
    bbox_inches: str = 'tight',
    bake_fonts: bool = True,
    close: bool = False,
    block=None,
    **savefig_kwargs,
):
    """Common save/show helper for ECAT plotting functions.

    ``PlotStyle`` should continue to manage style, fonts, and figure size.
    This helper manages the end of a plotting function: optional font baking,
    saved-output dpi, interactive dpi capping, and optional closing.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to finish.
    path : str or path-like, optional
        Output path.  If provided and ``save`` is None, the figure is saved.
    save : bool, optional
        Whether to save.  Defaults to ``path is not None``.
    show : bool, optional
        Whether to display the figure interactively.
    dpi : int, optional
        Saved-output dpi.  This does not control screen dpi.
    screen_dpi : float or None, optional
        Maximum interactive dpi before ``plt.show()``.  ``None`` disables
        capping.
    fmts, bbox_inches, **savefig_kwargs
        Forwarded to :func:`save_fig`.
    bake_fonts : bool, optional
        If True, call ``bake_text_fonts(fig)`` before save/show when possible.
    close : bool, optional
        If True, close the figure after saving/showing.  Defaults to False so
        callers can still return and edit ``fig``/``axes``.
    block : bool, optional
        Forwarded to :func:`show_fig`.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure instance.
    """
    _ensure_figure(fig)
    if save is None:
        save = path is not None
    if save and path is None:
        raise ValueError("path is required when save=True")

    if bake_fonts:
        try:
            from ._font_utils import bake_text_fonts

            bake_text_fonts(fig)
        except Exception as exc:
            warnings.warn(f"Could not fix figure fonts: {exc}", UserWarning, stacklevel=2)

    if save:
        save_fig(
            fig,
            str(path),
            fmts=fmts,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **savefig_kwargs,
        )
    if show:
        show_fig(fig, max_dpi=screen_dpi, block=block)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return fig
