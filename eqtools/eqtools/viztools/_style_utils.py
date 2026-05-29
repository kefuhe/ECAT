"""
_style_utils.py — Figure size utilities, column-width registry, and save_fig.

Public API
----------
register_column_width : Register a named publication column width
publication_figsize   : Return figure size (w, h) in inches for publication columns
save_fig              : Save figure to one or more file formats
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

# Import the centralized registry
from ._registry import _registry

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
        Column width **in inches**.

    Example
    -------
    >>> register_column_width('agu_single', 3.37)
    >>> register_column_width('copernicus', 3.15)
    >>> publication_figsize('agu_single')           # (3.37, 2.5275)
    """
    _registry.register_column_width(name, width_inch)


def _load_user_config() -> None:
    """Load user column-width overrides from the first existing config file.

    Search order (highest priority first):
    1. ``~/.config/eqtools/viztools.json``   (new standard path)
    2. ``~/.config/eqtools/plottools.json``
    3. ``~/.config/statutils/plottools.json``  (legacy, emits DeprecationWarning)
    4. ``~/.plottools.json``                   (legacy, emits DeprecationWarning)

    Silently skipped if no file exists or the JSON is malformed.
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
                f"eqtools.viztools: config file '{cfg_path}' is at a legacy path. "
                f"Move it to ~/.config/eqtools/viztools.json to suppress this warning.",
                DeprecationWarning, stacklevel=3,
            )
            _load_config_file(cfg_path, legacy=True)
            return


def _load_config_file(cfg_path: Path, legacy: bool = False) -> None:
    """Parse a JSON config file and register column widths from it."""
    try:
        data = json.loads(cfg_path.read_text(encoding='utf-8'))
        for name, w in data.get('column_widths', {}).items():
            register_column_width(name, float(w))
    except Exception as e:
        warnings.warn(
            f"eqtools.viztools: could not load config {cfg_path}: {e}"
        )


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
    - Preserves existing configuration entries
    """
    if config_path is None:
        config_path = Path.home() / '.config' / 'eqtools' / 'viztools.json'

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding='utf-8'))
        except Exception:
            data = {}
    else:
        data = {}

    # Update column_widths section
    if 'column_widths' not in data:
        data['column_widths'] = {}
    data['column_widths'][str(name).lower()] = float(width_inch)

    # Write back
    try:
        config_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
    except Exception as e:
        warnings.warn(f"Failed to save column width to {config_path}: {e}")
        return

    # Also register in current session
    register_column_width(name, width_inch)


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
        Fraction of the column width (0..1).
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
    cm_to_inch = 1 / 2.54

    if isinstance(column, (tuple, list)) and len(column) == 2:
        w_raw, h_raw = float(column[0]), float(column[1])
        if unit == 'cm':
            return (w_raw * cm_to_inch, h_raw * cm_to_inch)
        return (w_raw, h_raw)

    if isinstance(column, (int, float)):
        w = float(column)
    else:
        w = _registry.get_column_width(str(column).lower())
        if w is None:
            available = list(_registry.list_column_widths().keys())
            warnings.warn(
                f"Column width '{column}' not found. "
                f"Available: {available}. Using 'single' (3.5 in).",
                UserWarning,
                stacklevel=2
            )
            w = _registry.get_column_width('single')  # fallback

    if unit == 'cm':
        w = w * cm_to_inch

    w = w * float(fraction)

    if height is not None:
        h = float(height)
        if unit == 'cm':
            h = h * cm_to_inch
    else:
        h = w * float(aspect)

    return (w, h)


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
