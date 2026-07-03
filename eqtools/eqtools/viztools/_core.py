"""
_core.py — PlotStyle class, preset registry, and style initialization.

Public API
----------
PlotStyle        : Context-manager / persistent-apply / decorator style class
register_preset  : Register a named style preset
unregister_preset: Remove a user-registered preset
list_presets     : List all registered presets with descriptions
"""

import copy
import warnings
from os import listdir
from os.path import isdir, join
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt

# Import the centralized registry
from ._registry import _registry

# --------------------------------------------------------------------------
# Optional scienceplots
# --------------------------------------------------------------------------
try:
    import scienceplots
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False

# --------------------------------------------------------------------------
# Package styles directory
# --------------------------------------------------------------------------
_STYLES_DIR = Path(__file__).parent / 'styles'

# --------------------------------------------------------------------------
# Backward compatibility: direct access to registry (will be deprecated)
# --------------------------------------------------------------------------
_PRESET_REGISTRY = _registry._presets  # Direct reference for backward compat
_BUILTIN_PRESET_NAMES = _registry._builtin_presets  # Direct reference


def _ensure_initialized() -> None:
    """Run all one-time initialization exactly once (thread-safe)."""
    if _registry.is_initialized():
        return

    # Use registry's lock for thread-safe initialization
    with _registry._lock:
        # Double-check after acquiring lock
        if _registry.is_initialized():
            return

        _register_package_styles()
        _register_builtin_presets()
        from ._style_utils import _load_user_config
        _load_user_config()

        _registry.mark_initialized()


# --------------------------------------------------------------------------
# scienceplots registration (kept for backward compatibility)
# --------------------------------------------------------------------------
def _register_package_styles() -> None:
    """Register package .mplstyle files and scienceplots styles (once, lazily)."""
    if _registry.is_styles_registered():
        return
    # 1. Register scienceplots styles first (lower priority baseline)
    if HAS_SCIENCEPLOTS:
        try:
            register_science_styles()
        except Exception as e:
            warnings.warn(f"eqtools.viztools: could not register scienceplots styles: {e}")
    # 2. Register our own eqtools-*.mplstyle files (override / supplement)
    if _STYLES_DIR.exists():
        try:
            styles = plt.style.core.read_style_directory(str(_STYLES_DIR))
            plt.style.core.update_nested_dict(plt.style.library, styles)
            plt.style.core.available[:] = sorted(plt.style.library.keys())
        except Exception as e:
            warnings.warn(f"eqtools.viztools: could not register package styles: {e}")
    _registry.mark_styles_registered()


def register_science_styles() -> None:
    """Register scienceplots styles into matplotlib (requires scienceplots)."""
    if not HAS_SCIENCEPLOTS:
        return
    scienceplots_path = scienceplots.__path__[0]
    styles_path = join(scienceplots_path, 'styles')
    stylesheets = plt.style.core.read_style_directory(styles_path)
    for inode in listdir(styles_path):
        new_data_path = join(styles_path, inode)
        if isdir(new_data_path):
            new_stylesheets = plt.style.core.read_style_directory(new_data_path)
            stylesheets.update(new_stylesheets)
    plt.style.core.update_nested_dict(plt.style.library, stylesheets)
    plt.style.core.available[:] = sorted(plt.style.library.keys())


# LaTeX preamble constants (now imported from _constants)
from ._constants import (
    SANS_TEX_PREAMBLE as _SANS_TEX_PREAMBLE,
    SERIF_TEX_PREAMBLE as _SERIF_TEX_PREAMBLE,
    MATHFONT_ALIASES as _MATHFONT_ALIASES,
)

# --------------------------------------------------------------------------
# Preset Registry Functions (now delegate to _registry)
# --------------------------------------------------------------------------

def register_preset(
    name: str,
    *,
    base: Optional[Union[str, List[str]]] = None,
    mplstyles: Optional[List[str]] = None,
    rcparams: Optional[Dict[str, Any]] = None,
    chinese: bool = False,
    chinese_prefer_serif: bool = False,
    description: str = '',
) -> None:
    """Register a named style preset.

    Parameters
    ----------
    name : str
        Preset name, used in ``PlotStyle(name)``.
    base : str or list[str], optional
        Inherit from an existing preset or raw mplstyle name(s).
        When a list, parents are applied left-to-right.
    mplstyles : list[str], optional
        matplotlib style names (from ``plt.style.library``) to apply.
    rcparams : dict, optional
        Additional rcParam key/value overrides.
    chinese : bool
        Auto-inject the best available CJK font when True.
    chinese_prefer_serif : bool
        Prefer a CJK serif font (when ``chinese=True``).
    description : str
        Human-readable description shown by ``list_presets()``.

    Example
    -------
    >>> register_preset('my_lab', base='science',
    ...                 rcparams={'axes.grid': True},
    ...                 description='Lab house style')
    >>> with PlotStyle('my_lab', figsize='single'):
    ...     fig, ax = plt.subplots()
    """
    spec = {
        'base': base,
        'mplstyles': list(mplstyles or []),
        'rcparams': dict(rcparams or {}),
        'chinese': chinese,
        'chinese_prefer_serif': chinese_prefer_serif,
        'description': description,
    }
    _registry.register_preset(name, spec)


def unregister_preset(name: str) -> None:
    """Remove a user-registered preset.

    Built-in presets (science, science-serif, chinese, etc.) cannot be
    removed and will raise :class:`ValueError`.

    Parameters
    ----------
    name : str
        Preset name to remove.

    Raises
    ------
    ValueError
        If *name* is a built-in preset.

    Example
    -------
    >>> register_preset('test', base='science', rcparams={'axes.grid': True})
    >>> unregister_preset('test')
    """
    _registry.unregister_preset(name)


def list_presets() -> Dict[str, str]:
    """List all registered style presets.

    Returns
    -------
    dict
        ``{name: description}`` for every registered preset.

    Example
    -------
    >>> for name, desc in list_presets().items():
    ...     print(f'  {name:<20s}  {desc}')
    """
    _ensure_initialized()
    presets = _registry.list_presets()
    return {name: info['description'] for name, info in presets.items()}


# --------------------------------------------------------------------------
# PlotStyle — the main class
# --------------------------------------------------------------------------
class PlotStyle:
    """Style preset context manager, persistent-apply helper, and decorator.

    Applies a named preset (or list of presets) to matplotlib's rcParams,
    then restores **only the changed keys** on exit.

    Parameters
    ----------
    preset : str or list[str]
        Name(s) of registered presets.  Applied left-to-right so the last
        one wins when keys overlap.
    figsize : str, float, or tuple, optional
        Passed to :func:`publication_figsize`.  Accepted forms:

        * ``'single'``, ``'double'``, ``'nature'``, ``'ieee'``, ``'a4'``
        * Scalar width (height computed via *figsize_aspect*)
        * ``(width, height)`` tuple

    figsize_unit : {'inch', 'cm'}
    figsize_fraction : float
        Fraction of column width (default 1.0).
    figsize_aspect : float
        Height/width ratio when *figsize_height* is not given (default 0.75).
    figsize_height : float, optional
        Explicit height; overrides *figsize_aspect*.
    fontsize : float, optional
        Sets ``font.size`` and ``axes.labelsize``.
    tick_fontsize : float, optional
        Override ``xtick.labelsize`` / ``ytick.labelsize``.
    legend_fontsize : float, optional
        Override ``legend.fontsize``.
    title_fontsize : float, optional
        Override ``figure.titlesize``.
    legend_frame : bool
        Draw a frame around legends (default False).
    dpi : float, optional
        Saved-output dpi.  Sets ``savefig.dpi`` only.
    figure_dpi : float, optional
        Interactive figure dpi.  Use this only when the preview figure itself
        should use a non-default dpi.
    pdf_fonttype : {3, 42}, optional
        Sets ``pdf.fonttype`` and ``ps.fonttype``.
    usetex : bool, optional
        ``True``  — enable LaTeX rendering with auto preamble injection.
        ``False`` — disable LaTeX.
        ``None``  — leave the current setting unchanged (default).
    mathfont : str, optional
        Math font for matplotlib mathtext.  ``None`` (default) auto-selects.
    rcparams : dict, optional
        Raw rcParam overrides applied at the highest priority.

    Examples
    --------
    Context manager::

        with PlotStyle('science', figsize='single', fontsize=8):
            fig, ax = plt.subplots()

    Multiple presets::

        with PlotStyle(['science', 'chinese']):
            fig, ax = plt.subplots()

    Persistent apply::

        PlotStyle.apply('science', fontsize=9)
        PlotStyle.reset()

    Decorator::

        @PlotStyle.decorator('science', figsize='single')
        def my_plot():
            fig, ax = plt.subplots()
            return fig
    """

    # Note: _apply_stack and _MAX_STACK_DEPTH now managed by _registry
    # Direct access kept for backward compatibility (will access registry internally)

    # Handler chain: order matters (mathfont must come before usetex)
    # Users can extend this via register_handler()
    _HANDLERS = [
        '_apply_preset_layers',
        '_apply_figsize',
        '_apply_fontsize',
        '_apply_legend_frame',
        '_apply_dpi',
        '_apply_pdf_fonttype',
        '_apply_mathfont',   # determines _is_sans, needed by _apply_usetex
        '_apply_usetex',
        '_apply_extra',      # highest priority — always last
    ]

    # Custom handler registry: {priority: [(name, handler_fn), ...]}
    _CUSTOM_HANDLERS: Dict[int, list] = {}

    def __init__(
        self,
        preset: Union[str, List[str]] = 'science',
        *,
        figsize: Optional[Union[str, float, Tuple[float, float]]] = None,
        figsize_unit: str = 'inch',
        figsize_fraction: float = 1.0,
        figsize_aspect: float = 0.75,
        figsize_height: Optional[float] = None,
        fontsize: Optional[float] = None,
        tick_fontsize: Optional[float] = None,
        legend_fontsize: Optional[float] = None,
        title_fontsize: Optional[float] = None,
        legend_frame: bool = False,
        dpi: Optional[float] = None,
        figure_dpi: Optional[float] = None,
        pdf_fonttype: Optional[int] = None,
        usetex: Optional[bool] = None,
        mathfont: Optional[str] = None,
        rcparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        _ensure_initialized()

        self._presets = [preset] if isinstance(preset, str) else list(preset)
        self._figsize = figsize
        self._figsize_unit = figsize_unit
        self._figsize_fraction = figsize_fraction
        self._figsize_aspect = figsize_aspect
        self._figsize_height = figsize_height
        self._fontsize = fontsize
        self._tick_fontsize = tick_fontsize
        self._legend_fontsize = legend_fontsize
        self._title_fontsize = title_fontsize
        self._legend_frame = legend_frame
        self._dpi = dpi
        self._figure_dpi = figure_dpi
        self._pdf_fonttype = pdf_fonttype
        self._usetex = usetex
        self._mathfont = mathfont
        self._extra_rcparams = dict(rcparams or {})
        self._saved: Dict = {}

    # ------------------------------------------------------------------
    # Preset resolution helpers
    # ------------------------------------------------------------------
    def _resolve_preset(self, name: str, _visited: Optional[set] = None) -> Dict:
        """Recursively resolve preset inheritance, return merged entry dict."""
        if _visited is None:
            _visited = set()
        if name in _visited:
            return {'mplstyles': [], 'rcparams': {}, 'chinese': False, 'chinese_prefer_serif': False}
        _visited.add(name)

        if name not in _PRESET_REGISTRY:
            # Enhanced error message with suggestions
            from difflib import get_close_matches
            available = list(_PRESET_REGISTRY.keys())
            suggestions = get_close_matches(name, available, n=3, cutoff=0.6)

            msg = f"PlotStyle: preset '{name}' not found."
            if suggestions:
                msg += f"\n  Did you mean: {', '.join(suggestions)}?"
            msg += f"\n  Available presets: {', '.join(sorted(available))}"
            msg += f"\n  Use list_presets() to see descriptions."
            msg += f"\n  Falling back to 'science'."

            warnings.warn(msg, UserWarning, stacklevel=3)
            name = 'science'

        entry = _PRESET_REGISTRY[name]
        result: Dict = {'mplstyles': [], 'rcparams': {}, 'chinese': False, 'chinese_prefer_serif': False}

        raw_base = entry.get('base')
        if raw_base:
            bases = [raw_base] if isinstance(raw_base, str) else list(raw_base)
            for base_name in bases:
                if base_name in _PRESET_REGISTRY:
                    # Registered preset: recurse
                    base = self._resolve_preset(base_name, _visited.copy())
                    result['mplstyles'].extend(base['mplstyles'])
                    result['rcparams'].update(base['rcparams'])
                    if base['chinese']:
                        result['chinese'] = True
                    if base['chinese_prefer_serif']:
                        result['chinese_prefer_serif'] = True
                else:
                    # Treat as raw mplstyle name (P3: mixed base support)
                    result['mplstyles'].append(base_name)

        result['mplstyles'].extend(entry['mplstyles'])
        result['rcparams'].update(entry['rcparams'])
        if entry['chinese']:
            result['chinese'] = True
        if entry['chinese_prefer_serif']:
            result['chinese_prefer_serif'] = True
        return result

    # ------------------------------------------------------------------
    # Handler chain methods (each receives acc dict, modifies in-place)
    # ------------------------------------------------------------------
    def _apply_preset_layers(self, acc: Dict) -> None:
        """Apply mplstyles + preset rcparams + optional CJK font injection."""
        from ._font_utils import _probe_chinese_fonts
        from pathlib import Path

        for preset_name in self._presets:
            resolved = self._resolve_preset(preset_name)

            # Layer mplstyles from plt.style.library
            for style_name in resolved['mplstyles']:
                if style_name in plt.style.library:
                    acc.update(dict(plt.style.library[style_name]))
                else:
                    # Compatibility path: check if it's a file path
                    style_path = Path(style_name)
                    if style_path.exists() and style_path.suffix == '.mplstyle':
                        # Load the style file directly (deprecated path)
                        warnings.warn(
                            f"PlotStyle: loading style from file path '{style_name}' is deprecated. "
                            f"The style should be registered in plt.style.library. "
                            f"This compatibility path will be removed in a future version.",
                            DeprecationWarning,
                            stacklevel=3
                        )
                        try:
                            from matplotlib.style.core import rc_params_from_file
                            style_params = rc_params_from_file(str(style_path), use_default_template=False)
                            acc.update(style_params)
                        except Exception as e:
                            warnings.warn(f"PlotStyle: failed to load style from '{style_name}': {e}")
                    else:
                        warnings.warn(f"PlotStyle: mplstyle '{style_name}' not in library.")

            # Layer preset rcparams
            acc.update(resolved['rcparams'])

            # Chinese font injection
            if resolved['chinese']:
                fonts = _probe_chinese_fonts()
                prefer_serif = resolved['chinese_prefer_serif']
                cjk_font = fonts.get('serif' if prefer_serif else 'sans')
                if cjk_font:
                    family = 'serif' if prefer_serif else 'sans-serif'
                    current = list(acc.get(f'font.{family}', []))
                    if cjk_font not in current:
                        current.insert(0, cjk_font)
                    acc[f'font.{family}'] = current
                    acc['font.family'] = family
                    acc['axes.unicode_minus'] = False
                else:
                    warnings.warn(
                        "PlotStyle: no CJK font found on this system. "
                        "Chinese characters may not render correctly. "
                        "Install SimHei, Microsoft YaHei, or Noto Sans CJK SC."
                    )
                    acc['axes.unicode_minus'] = False

    def _apply_figsize(self, acc: Dict) -> None:
        from ._style_utils import publication_figsize
        if self._figsize is not None:
            fs = publication_figsize(
                column=self._figsize,
                fraction=self._figsize_fraction,
                aspect=self._figsize_aspect,
                height=self._figsize_height,
                unit=self._figsize_unit,
            )
            acc['figure.figsize'] = list(fs)

    def _apply_fontsize(self, acc: Dict) -> None:
        if self._fontsize is not None:
            fs = float(self._fontsize)
            acc['font.size']      = fs
            acc['axes.labelsize'] = fs
            _tick_fs  = float(self._tick_fontsize)   if self._tick_fontsize   is not None else max(fs - 1.0, 6.0)
            _leg_fs   = float(self._legend_fontsize) if self._legend_fontsize is not None else max(fs - 1.0, 6.0)
            _title_fs = float(self._title_fontsize)  if self._title_fontsize  is not None else fs + 1.0
            acc['xtick.labelsize']  = _tick_fs
            acc['ytick.labelsize']  = _tick_fs
            acc['legend.fontsize']  = _leg_fs
            acc['figure.titlesize'] = _title_fs

    def _apply_legend_frame(self, acc: Dict) -> None:
        if self._legend_frame:
            acc['legend.frameon']    = True
            acc['legend.framealpha'] = 0.7
            acc['legend.fancybox']   = True

    def _apply_dpi(self, acc: Dict) -> None:
        if self._dpi is not None:
            acc['savefig.dpi'] = float(self._dpi)
        if self._figure_dpi is not None:
            acc['figure.dpi'] = float(self._figure_dpi)

    def _apply_pdf_fonttype(self, acc: Dict) -> None:
        if self._pdf_fonttype is not None:
            acc['pdf.fonttype'] = int(self._pdf_fonttype)
            acc['ps.fonttype']  = int(self._pdf_fonttype)

    def _apply_mathfont(self, acc: Dict) -> None:
        """Determine _is_sans and apply mathfont (must run before _apply_usetex)."""
        _effective_family = acc.get('font.family', 'sans-serif')
        if isinstance(_effective_family, list):
            _effective_family = _effective_family[0] if _effective_family else 'sans-serif'
        # Store _is_sans as instance attribute for use by _apply_usetex
        self._is_sans = (_effective_family == 'sans-serif')

        if self._mathfont is not None:
            _mpl_fontset = _MATHFONT_ALIASES.get(
                str(self._mathfont).lower(), str(self._mathfont)
            )
            acc['mathtext.fontset'] = _mpl_fontset
        else:
            acc['mathtext.fontset'] = 'stixsans' if self._is_sans else 'stix'

    def _apply_usetex(self, acc: Dict) -> None:
        """Apply usetex setting with CJK-incompatibility guard and auto preamble."""
        _has_cjk = any(
            _PRESET_REGISTRY.get(p, {}).get('chinese', False)
            for p in self._presets
        )
        if self._usetex is True:
            if _has_cjk:
                warnings.warn(
                    "PlotStyle: usetex=True is incompatible with CJK presets "
                    "(pdflatex cannot render Chinese/Japanese/Korean characters). "
                    "usetex has been automatically disabled. "
                    "Use xelatex via the PGF backend for CJK + LaTeX math.",
                    UserWarning, stacklevel=3,
                )
                acc['text.usetex'] = False
            else:
                acc['text.usetex'] = True
                if 'text.latex.preamble' not in self._extra_rcparams:
                    preamble = _SANS_TEX_PREAMBLE if self._is_sans else _SERIF_TEX_PREAMBLE
                    acc['text.latex.preamble'] = preamble
        elif self._usetex is False:
            acc['text.usetex'] = False

    def _apply_extra(self, acc: Dict) -> None:
        """Apply caller's raw rcparams overrides (highest priority)."""
        acc.update(self._extra_rcparams)

    # ------------------------------------------------------------------
    # Main builder: dispatch through handler chain
    # ------------------------------------------------------------------
    def _build_final_rcparams(self) -> Dict:
        """Collect all rcParams to apply, lowest -> highest priority."""
        acc: Dict = {}

        # Built-in handlers
        for handler_name in self._HANDLERS:
            if handler_name == '_apply_extra':
                # Insert custom handlers before _apply_extra
                self._apply_custom_handlers(acc)
            getattr(self, handler_name)(acc)

        return acc

    def _apply_custom_handlers(self, acc: Dict) -> None:
        """Apply user-registered custom handlers in priority order."""
        for priority in sorted(self._CUSTOM_HANDLERS.keys()):
            for name, handler_fn in self._CUSTOM_HANDLERS[priority]:
                try:
                    handler_fn(self, acc)
                except Exception as e:
                    warnings.warn(
                        f"PlotStyle: custom handler '{name}' failed: {e}",
                        UserWarning, stacklevel=4
                    )

    # ------------------------------------------------------------------
    # Apply / restore
    # ------------------------------------------------------------------
    def _apply_to(self, save_target: Dict) -> None:
        """Apply rcParams and save original values into *save_target*.

        Only stores values that actually change to minimize memory usage.
        """
        import copy
        final = self._build_final_rcparams()
        for k, v in final.items():
            try:
                current = mpl.rcParams[k]
                # Only save and update if the value actually changes
                if current != v:
                    save_target[k] = copy.deepcopy(current)
                    mpl.rcParams[k] = v
            except (KeyError, ValueError):
                pass  # unknown or invalid key

    @staticmethod
    def _restore(saved: Dict) -> None:
        """Restore previously saved rcParams."""
        for k, v in saved.items():
            try:
                mpl.rcParams[k] = v
            except (KeyError, ValueError):
                pass

    # ------------------------------------------------------------------
    # Context manager interface
    # ------------------------------------------------------------------
    def __enter__(self) -> 'PlotStyle':
        self._saved = {}
        self._apply_to(self._saved)
        return self

    def __exit__(self, *args) -> None:
        self._restore(self._saved)
        self._saved = {}

    # ------------------------------------------------------------------
    # Persistent apply / reset
    # ------------------------------------------------------------------
    @classmethod
    def apply(cls, preset: Union[str, List[str]] = 'science', **kwargs) -> 'PlotStyle':
        """Apply a preset persistently (no context manager needed).

        Returns the :class:`PlotStyle` instance; call :meth:`reset` to undo.

        Parameters
        ----------
        preset : str or list[str]
        **kwargs
            Any ``PlotStyle.__init__`` keyword argument.

        Example
        -------
        >>> PlotStyle.apply('science', fontsize=9)
        >>> PlotStyle.reset()
        """
        ps = cls(preset, **kwargs)
        saved: Dict = {}
        ps._apply_to(saved)
        _registry.push_style(saved)  # Now uses registry (includes overflow check)
        return ps

    @classmethod
    def reset(cls) -> None:
        """Restore rcParams to the state before the last :meth:`apply` call.

        Example
        -------
        >>> PlotStyle.apply('science')
        >>> PlotStyle.reset()
        """
        try:
            saved = _registry.pop_style()
            cls._restore(saved)
        except IndexError:
            warnings.warn("PlotStyle.reset(): nothing to restore.")

    @classmethod
    def reset_all(cls) -> None:
        """Restore rcParams to the state before **all** :meth:`apply` calls.

        Example
        -------
        >>> PlotStyle.apply('science')
        >>> PlotStyle.apply('chinese')
        >>> PlotStyle.reset_all()   # both layers removed at once
        """
        while _registry.get_stack_depth() > 0:
            try:
                saved = _registry.pop_style()
                cls._restore(saved)
            except IndexError:
                break

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------
    @classmethod
    def register_handler(cls, handler_fn, name: Optional[str] = None, priority: int = 50) -> None:
        """Register a custom handler to extend the style processing chain.

        Custom handlers are called after built-in handlers but before _apply_extra.
        They receive (self, acc) where acc is the accumulated rcParams dict.

        Parameters
        ----------
        handler_fn : callable
            Function with signature ``handler_fn(plotstyle_instance, acc_dict) -> None``.
            Should modify acc_dict in-place.
        name : str, optional
            Handler name for debugging. Auto-generated if not provided.
        priority : int
            Execution priority (default 50). Lower values run first.
            Built-in handlers run at priority 0-100.

        Raises
        ------
        TypeError
            If handler_fn is not callable.
        ValueError
            If handler_fn doesn't have the correct signature.

        Example
        -------
        >>> def my_grid_handler(ps, acc):
        ...     if hasattr(ps, '_enable_grid') and ps._enable_grid:
        ...         acc['axes.grid'] = True
        ...         acc['grid.alpha'] = 0.3
        >>> PlotStyle.register_handler(my_grid_handler, 'grid', priority=60)
        """
        import inspect

        if name is None:
            name = getattr(handler_fn, '__name__', f'handler_{id(handler_fn)}')

        if not callable(handler_fn):
            raise TypeError(
                f"handler_fn must be callable, got {type(handler_fn).__name__}"
            )

        # Validate signature
        try:
            sig = inspect.signature(handler_fn)
            params = list(sig.parameters.keys())

            if len(params) != 2:
                raise ValueError(
                    f"Handler '{name}' must accept exactly 2 parameters, "
                    f"got {len(params)}: {params}. "
                    f"Expected signature: (plotstyle_instance, rcparams_dict) -> None"
                )
        except (ValueError, TypeError) as e:
            # If signature inspection fails, warn but continue
            if "must accept exactly 2 parameters" in str(e):
                raise  # Re-raise our validation error
            warnings.warn(
                f"Could not validate signature for handler '{name}': {e}. "
                f"Handler should have signature (plotstyle_instance, rcparams_dict) -> None.",
                UserWarning, stacklevel=2
            )

        if priority not in cls._CUSTOM_HANDLERS:
            cls._CUSTOM_HANDLERS[priority] = []

        # Check for duplicate names at same priority
        existing_names = [n for n, _ in cls._CUSTOM_HANDLERS[priority]]
        if name in existing_names:
            warnings.warn(
                f"PlotStyle: handler '{name}' already registered at priority {priority}. "
                f"Overwriting previous handler.",
                UserWarning, stacklevel=2
            )
            cls._CUSTOM_HANDLERS[priority] = [
                (n, fn) for n, fn in cls._CUSTOM_HANDLERS[priority] if n != name
            ]

        cls._CUSTOM_HANDLERS[priority].append((name, handler_fn))

    @classmethod
    def unregister_handler(cls, name: str, priority: Optional[int] = None) -> None:
        """Remove a custom handler by name.

        Parameters
        ----------
        name : str
            Handler name to remove.
        priority : int, optional
            If provided, only remove from this priority level.
            Otherwise, remove from all priority levels.

        Example
        -------
        >>> PlotStyle.unregister_handler('grid')
        """
        if priority is not None:
            if priority in cls._CUSTOM_HANDLERS:
                cls._CUSTOM_HANDLERS[priority] = [
                    (n, fn) for n, fn in cls._CUSTOM_HANDLERS[priority] if n != name
                ]
        else:
            for p in cls._CUSTOM_HANDLERS:
                cls._CUSTOM_HANDLERS[p] = [
                    (n, fn) for n, fn in cls._CUSTOM_HANDLERS[p] if n != name
                ]

    @classmethod
    def list_handlers(cls) -> Dict[int, List[str]]:
        """List all registered custom handlers.

        Returns
        -------
        dict
            ``{priority: [handler_names]}`` for all custom handlers.

        Example
        -------
        >>> PlotStyle.list_handlers()
        {50: ['grid'], 60: ['custom_colors']}
        """
        return {
            priority: [name for name, _ in handlers]
            for priority, handlers in cls._CUSTOM_HANDLERS.items()
            if handlers
        }

    def apply_to_axes(self, ax) -> None:
        """Apply this style to a specific Axes instance (non-global).

        This method applies style settings directly to an Axes object without
        modifying global rcParams. Useful for mixed-style figures (e.g., main
        plot + inset with different styles).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to style.

        Raises
        ------
        TypeError
            If ax is not a matplotlib Axes object.

        Notes
        -----
        Only a subset of rcParams can be applied at the Axes level:
        - Font sizes (labels, ticks, title)
        - Grid settings
        - Spine visibility
        - Tick parameters

        Global settings (figure.figsize, font.family, etc.) are ignored.

        Example
        -------
        >>> fig, (ax1, ax2) = plt.subplots(1, 2)
        >>> PlotStyle('science', fontsize=10).apply_to_axes(ax1)
        >>> PlotStyle('minimal', fontsize=8).apply_to_axes(ax2)
        """
        from matplotlib.axes import Axes

        if not isinstance(ax, Axes):
            raise TypeError(
                f"Expected matplotlib.axes.Axes, got {type(ax).__name__}. "
                f"Pass an Axes object from fig.add_subplot() or fig, ax = plt.subplots()."
            )

        final = self._build_final_rcparams()

        # Identify incompatible (figure-level) settings
        incompatible_prefixes = ('figure.', 'savefig.', 'font.', 'text.', 'mathtext.')
        incompatible = [k for k in final if any(k.startswith(p) for p in incompatible_prefixes)]

        if incompatible and len(incompatible) <= 10:
            warnings.warn(
                f"apply_to_axes: The following {len(incompatible)} rcParams are figure-level "
                f"and cannot be applied to individual axes: {incompatible}. "
                f"Use context manager or apply() for figure-level settings.",
                UserWarning,
                stacklevel=2
            )
        elif incompatible:
            warnings.warn(
                f"apply_to_axes: {len(incompatible)} figure-level rcParams "
                f"(figure.*, font.*, etc.) cannot be applied to axes. "
                f"Use context manager or apply() for figure-level settings.",
                UserWarning,
                stacklevel=2
            )

        # Map rcParams to Axes methods
        axes_mappings = {
            'axes.labelsize': lambda v: (
                ax.xaxis.label.set_fontsize(v),
                ax.yaxis.label.set_fontsize(v)
            ),
            'xtick.labelsize': lambda v: ax.tick_params(axis='x', labelsize=v),
            'ytick.labelsize': lambda v: ax.tick_params(axis='y', labelsize=v),
            'axes.titlesize': lambda v: ax.title.set_fontsize(v),
            'axes.grid': lambda v: ax.grid(v),
            'grid.alpha': lambda v: ax.grid(alpha=v) if final.get('axes.grid') else None,
            'grid.linewidth': lambda v: ax.grid(linewidth=v) if final.get('axes.grid') else None,
            'axes.spines.top': lambda v: ax.spines['top'].set_visible(v),
            'axes.spines.right': lambda v: ax.spines['right'].set_visible(v),
            'axes.spines.left': lambda v: ax.spines['left'].set_visible(v),
            'axes.spines.bottom': lambda v: ax.spines['bottom'].set_visible(v),
            'axes.linewidth': lambda v: [s.set_linewidth(v) for s in ax.spines.values()],
            'xtick.major.width': lambda v: ax.tick_params(axis='x', which='major', width=v),
            'ytick.major.width': lambda v: ax.tick_params(axis='y', which='major', width=v),
            'xtick.minor.width': lambda v: ax.tick_params(axis='x', which='minor', width=v),
            'ytick.minor.width': lambda v: ax.tick_params(axis='y', which='minor', width=v),
            'xtick.major.size': lambda v: ax.tick_params(axis='x', which='major', size=v),
            'ytick.major.size': lambda v: ax.tick_params(axis='y', which='major', size=v),
            'xtick.minor.size': lambda v: ax.tick_params(axis='x', which='minor', size=v),
            'ytick.minor.size': lambda v: ax.tick_params(axis='y', which='minor', size=v),
        }

        for key, value in final.items():
            if key in axes_mappings:
                try:
                    axes_mappings[key](value)
                except Exception:
                    pass  # Silently skip incompatible settings

    @classmethod
    def decorator(cls, preset: Union[str, List[str]] = 'science', **kwargs) -> Callable:
        """Decorator that applies a style around a function.

        Example
        -------
        >>> @PlotStyle.decorator('science', figsize='single')
        ... def my_plot():
        ...     fig, ax = plt.subplots()
        ...     return fig
        """
        def _decorator(func: Callable) -> Callable:
            def _wrapper(*args, **kw):
                with cls(preset, **kwargs):
                    return func(*args, **kw)
            _wrapper.__name__ = func.__name__
            _wrapper.__doc__  = func.__doc__
            return _wrapper
        return _decorator

    # ------------------------------------------------------------------
    # Convenience: subplots()
    # ------------------------------------------------------------------
    def subplots(self, *args, **kwargs) -> Tuple:
        """Apply this style and create a (fig, axes) pair.

        The style stays active for the lifetime of the returned figure and is
        automatically restored when the figure is closed.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`matplotlib.pyplot.subplots`.

        Returns
        -------
        tuple
            (fig, axes) as returned by plt.subplots()

        Example
        -------
        ::

            fig, ax = PlotStyle('science', figsize='single', fontsize=8).subplots()
            # No manual reset needed — style is restored when the figure closes.
        """
        self.__enter__()
        fig, axes = plt.subplots(*args, **kwargs)
        saved = self._saved  # capture the dict created by __enter__

        fig.canvas.mpl_connect('close_event', lambda _: PlotStyle._restore(saved))
        return fig, axes

    # ------------------------------------------------------------------
    # Inspector: describe()
    # ------------------------------------------------------------------
    @classmethod
    def describe(cls, name: str, print_result: bool = True,
                 filter_prefix: Optional[str] = None,
                 verbose: bool = False) -> Dict:
        """Print and return the fully resolved rcParams that a preset will apply.

        Parameters
        ----------
        name : str
            Registered preset name.
        print_result : bool
            If True (default) print the resolved parameters to stdout.
        filter_prefix : str, optional
            Only include keys starting with this prefix in the output and
            the returned dict (e.g. ``'font.'``).
        verbose : bool, optional
            If True, show the source of each parameter (mplstyle file or
            preset rcparams). Default is False.

        Returns
        -------
        dict
            Resolved rcParam dict (optionally filtered). If verbose=True,
            returns ``{'values': dict, 'sources': dict}``.

        Example
        -------
        >>> PlotStyle.describe('minimal')
        >>> rc = PlotStyle.describe('science', print_result=False)
        >>> rc_fonts = PlotStyle.describe('science', filter_prefix='font.')
        >>> # Show sources
        >>> result = PlotStyle.describe('science', verbose=True)
        >>> print(result['sources']['font.size'])  # e.g., "preset:science.rcparams"
        """
        _ensure_initialized()

        presets = _registry.list_presets()
        if name not in presets:
            available = list(presets.keys())
            print(f"Preset '{name}' not found. Available: {available}")
            return {} if not verbose else {'values': {}, 'sources': {}}

        entry = presets[name]
        base_str = f"base: {entry.get('base')}" if entry.get('base') else 'no base'

        if print_result:
            print(f"Preset: {name}  ({base_str})")
            print(f"mplstyles : {entry.get('mplstyles', [])}")

        tmp = cls(name)

        if verbose:
            # Track sources for each parameter
            sources = {}
            result = {}

            # Resolve preset to get mplstyles and rcparams
            resolved = tmp._resolve_preset(name)

            # Track mplstyle contributions
            for style_name in resolved['mplstyles']:
                if style_name in plt.style.library:
                    for key, value in plt.style.library[style_name].items():
                        result[key] = value
                        sources[key] = f"mplstyle:{style_name}"

            # Track preset rcparams contributions (override mplstyles)
            for key, value in resolved['rcparams'].items():
                result[key] = value
                sources[key] = f"preset:{name}.rcparams"

            # Apply filter if specified
            if filter_prefix is not None:
                result = {k: v for k, v in result.items() if k.startswith(filter_prefix)}
                sources = {k: v for k, v in sources.items() if k.startswith(filter_prefix)}

            if print_result:
                print(f"\nResolved rcparams ({len(result)} keys):")
                for k in sorted(result.keys()):
                    source = sources.get(k, 'unknown')
                    print(f"  {k:40s} = {result[k]!r:20s}  # from {source}")

            return {'values': result, 'sources': sources}

        else:
            # Original behavior (no source tracking)
            result = tmp._build_final_rcparams()

            if filter_prefix is not None:
                result = {k: v for k, v in result.items() if k.startswith(filter_prefix)}

            if print_result:
                print(f"Resolved rcparams ({len(result)} keys):")
                for k, v in sorted(result.items()):
                    print(f"  {k} = {v}")

            return result

    def __repr__(self) -> str:
        """Return detailed string representation for interactive exploration."""
        presets_str = ', '.join(f"'{p}'" for p in self._presets)
        params = []

        if self._figsize is not None:
            params.append(f"figsize={self._figsize!r}")
        if self._fontsize is not None:
            params.append(f"fontsize={self._fontsize}")
        if self._usetex is not None:
            params.append(f"usetex={self._usetex}")
        if self._dpi is not None:
            params.append(f"dpi={self._dpi}")
        if self._figure_dpi is not None:
            params.append(f"figure_dpi={self._figure_dpi}")
        if self._legend_frame:
            params.append("legend_frame=True")

        if params:
            params_str = ', '.join(params)
            return f"PlotStyle([{presets_str}], {params_str})"
        return f"PlotStyle([{presets_str}])"

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        return self.__repr__()


# --------------------------------------------------------------------------
# Built-in presets
# --------------------------------------------------------------------------
def _register_builtin_presets() -> None:
    register_preset(
        'science',
        mplstyles=['eqtools-science'],
        description='Sans-serif publication style (default)',
    )
    register_preset(
        'science-serif',
        mplstyles=['eqtools-science-serif'],
        description='Serif publication style',
    )
    register_preset(
        'chinese',
        base='science',
        chinese=True,
        description='Sans-serif + auto-detected CJK font',
    )
    register_preset(
        'chinese-serif',
        base='science-serif',
        chinese=True,
        chinese_prefer_serif=True,
        description='Serif + auto-detected CJK serif font',
    )
    register_preset(
        'presentation',
        mplstyles=['eqtools-presentation'],
        description='Large fonts for slides / posters',
    )
    register_preset(
        'notebook',
        mplstyles=['eqtools-notebook'],
        description='Moderate size for Jupyter notebooks',
    )
    register_preset(
        'minimal',
        base='science',
        mplstyles=['eqtools-minimal'],
        description='Clean: no top/right spines, no minor ticks',
    )
    # ── Color palette presets (overlay on top of any base preset) ──────────
    register_preset(
        'colors-bright',
        mplstyles=['eqtools-colors-bright'],
        description='Colorblind-safe bright palette (Paul Tol, 7 colors)',
    )
    register_preset(
        'colors-vibrant',
        mplstyles=['eqtools-colors-vibrant'],
        description='Colorblind-safe vibrant palette (Paul Tol, 7 colors)',
    )
    register_preset(
        'colors-contrast',
        mplstyles=['eqtools-colors-contrast'],
        description='High-contrast 3-color palette (colorblind + B&W print safe)',
    )
    # ── Plot-mode presets (overlay on top of any base preset) ───────────────
    register_preset(
        'scatter',
        base='science',
        mplstyles=['eqtools-scatter'],
        description='Scatter mode: markers only, no connecting lines (7 markers x std-colors)',
    )
    register_preset(
        'ieee',
        base='science-serif',
        mplstyles=['eqtools-ieee'],
        description='IEEE linestyle cycling: 4 colors x 4 linestyles, B&W print safe',
    )
    # Mark all registered presets as built-in (for unregister_preset guard)
    for name in _registry.list_presets().keys():
        _registry.mark_builtin_preset(name)
