"""
_font_utils.py — CJK font detection and text font baking utilities.

Public API
----------
list_chinese_fonts   : Probe system for available CJK fonts
bake_text_fonts      : Fix Text artist fonts before PlotStyle.reset()
"""

import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import matplotlib as mpl

# 导入常量
from ._constants import (
    CHINESE_SANS_CANDIDATES as _CHINESE_SANS_CANDIDATES,
    CHINESE_SERIF_CANDIDATES as _CHINESE_SERIF_CANDIDATES,
    FONT_CACHE_EXPIRY_DAYS,
    FONT_CACHE_DIR_NAME,
    FONT_CACHE_FILE_NAME,
)


# --------------------------------------------------------------------------
# Font cache persistence utilities
# --------------------------------------------------------------------------

def _get_font_cache_path() -> Path:
    """Get the path to the font cache file.

    Returns
    -------
    Path
        Path to font cache file (~/.cache/eqtools/font_cache.pkl)
    """
    cache_dir = Path.home() / FONT_CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / FONT_CACHE_FILE_NAME


def _load_font_cache() -> Optional[Dict]:
    """Load font cache from disk.

    Returns
    -------
    dict or None
        Cached font data if valid, None if cache doesn't exist or is expired
    """
    cache_path = _get_font_cache_path()

    if not cache_path.exists():
        return None

    # Check if cache is expired
    try:
        age_seconds = time.time() - cache_path.stat().st_mtime
        age_days = age_seconds / (24 * 3600)

        if age_days > FONT_CACHE_EXPIRY_DAYS:
            # Cache expired, delete it
            cache_path.unlink(missing_ok=True)
            return None
    except (OSError, AttributeError):
        return None

    # Load cache
    try:
        import pickle
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        # Cache corrupted, delete it
        cache_path.unlink(missing_ok=True)
        return None


def _save_font_cache(data: Dict) -> None:
    """Save font cache to disk.

    Parameters
    ----------
    data : dict
        Font data to cache
    """
    cache_path = _get_font_cache_path()

    try:
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # Silently fail if we can't write cache
        pass


def _font_exists(font_name: str) -> bool:
    """Check if a font exists in the system.

    Parameters
    ----------
    font_name : str
        Name of the font to check

    Returns
    -------
    bool
        True if font exists, False otherwise
    """
    try:
        from matplotlib.font_manager import fontManager
        available = {f.name for f in fontManager.ttflist}
        return font_name in available
    except Exception:
        return False


@lru_cache(maxsize=1)
def _probe_chinese_fonts() -> Dict[str, Optional[str]]:
    """Probe system fonts once with persistent caching; return best available CJK font names.

    This function checks a persistent disk cache first. If the cache is valid
    (less than 7 days old), it uses the cached result. Otherwise, it performs
    a full font probe and caches the result for future use.

    Returns
    -------
    dict
        {'sans': name_or_None, 'serif': name_or_None}
    """
    # Try to load from persistent cache first
    cache = _load_font_cache()
    if cache is not None and 'cjk_fonts' in cache:
        fonts = cache['cjk_fonts']

        # Validate that cached fonts still exist
        sans_valid = fonts['sans'] is None or _font_exists(fonts['sans'])
        serif_valid = fonts['serif'] is None or _font_exists(fonts['serif'])

        if sans_valid and serif_valid:
            return fonts

    # Cache miss or invalid - perform full probe
    try:
        from matplotlib.font_manager import fontManager
        available = {f.name for f in fontManager.ttflist}
    except Exception:
        return {'sans': None, 'serif': None}

    result = {
        'sans': next((f for f in _CHINESE_SANS_CANDIDATES if f in available), None),
        'serif': next((f for f in _CHINESE_SERIF_CANDIDATES if f in available), None),
    }

    # Save to persistent cache
    _save_font_cache({'cjk_fonts': result})

    return result


def list_chinese_fonts(refresh: bool = False) -> Dict[str, Optional[str]]:
    """
    Return the best available CJK font names on this system.

    Parameters
    ----------
    refresh : bool, optional
        If True, clear the cache and re-probe system fonts.
        Use this after installing new fonts. Default is False.

    Returns
    -------
    dict
        ``{'sans': name_or_None, 'serif': name_or_None}``

    Examples
    --------
    >>> fonts = list_chinese_fonts()
    >>> print(fonts)
    {'sans': 'SimHei', 'serif': 'SimSun'}

    >>> # After installing new fonts
    >>> fonts = list_chinese_fonts(refresh=True)
    """
    if refresh:
        _probe_chinese_fonts.cache_clear()
    return _probe_chinese_fonts()


def bake_text_fonts(fig) -> None:
    """Explicitly resolve and fix Text artist fonts while a PlotStyle is active.

    Call this **inside** a ``PlotStyle`` context (or before :meth:`PlotStyle.reset`
    in visualization functions) so that rendered fonts are independent of later
    ``rcParams`` changes.

    Background
    ----------
    matplotlib ``Text`` objects store the font *family* string (e.g.
    ``'sans-serif'``), not the specific font name.  At render time the
    family string is looked up against the current ``font.sans-serif``
    rcParam list.  If ``PlotStyle.reset()`` has already been called before
    ``plt.show()``, the lookup uses the *restored* (system default) list and
    the wrong font is rendered.

    This function walks all ``Text`` artists in *fig*, resolves the first
    available font from the *currently active* ``font.<family>`` list, and
    sets it explicitly via ``Text.set_fontname()``.

    Enhanced in Phase 2 to support:
    - ``Annotation`` objects
    - ``Text3D`` objects (if mpl_toolkits.mplot3d is available)
    - Colorbar text elements

    Note
    ----
    This function is **not effective** for ``usetex=True`` rendering, which
    uses an external pdflatex process.  In that case, call ``plt.show()``
    inside the PlotStyle context instead.

    Parameters
    ----------
    fig : matplotlib.figure.Figure

    Example
    -------
    ::

        with PlotStyle('science', fontsize=8):
            fig, ax = plt.subplots()
            ax.set_xlabel('Distance (km)')
            bake_text_fonts(fig)   # call while style is still active
        # now plt.show() uses the baked font, not the restored default
    """
    import matplotlib.text as _mtext
    import matplotlib.font_manager as _fm
    import warnings

    family = mpl.rcParams.get('font.family', ['sans-serif'])
    if isinstance(family, list):
        family = family[0] if family else 'sans-serif'
    font_list = mpl.rcParams.get(f'font.{family}', [])
    if isinstance(font_list, str):
        font_list = [font_list]
    available = {f.name for f in _fm.fontManager.ttflist}
    resolved = next((f for f in font_list if f in available), None)
    if resolved is None:
        warnings.warn(
            f"Could not resolve any font from '{family}' family: {font_list[:3]}... "
            f"Text fonts may revert to default after PlotStyle context exit. "
            f"Ensure matplotlib can find the font, or use a different font family.",
            UserWarning,
            stacklevel=2
        )
        return  # no matching font found, leave artists unchanged

    def _walk(artist):
        # Handle regular Text objects
        if isinstance(artist, _mtext.Text) and artist.get_text():
            try:
                artist.set_fontname(resolved)
            except Exception:
                pass

        # Handle Annotation objects (subclass of Text)
        if isinstance(artist, _mtext.Annotation):
            try:
                artist.set_fontname(resolved)
            except Exception:
                pass

        # Handle Text3D objects (if available)
        try:
            from mpl_toolkits.mplot3d.art3d import Text3D
            if isinstance(artist, Text3D):
                try:
                    artist.set_fontname(resolved)
                except Exception:
                    pass
        except ImportError:
            pass  # 3D toolkit not available

        # Handle colorbar axes
        if hasattr(artist, 'colorbar') and artist.colorbar is not None:
            try:
                _walk(artist.colorbar.ax)
            except Exception:
                pass

        # Handle axes with colorbars
        if hasattr(artist, 'get_axes') and hasattr(artist.get_axes(), 'collections'):
            ax = artist.get_axes()
            for collection in ax.collections:
                if hasattr(collection, 'colorbar') and collection.colorbar is not None:
                    try:
                        _walk(collection.colorbar.ax)
                    except Exception:
                        pass

        # Recursively walk children
        for child in getattr(artist, 'get_children', lambda: [])():
            _walk(child)

    _walk(fig)
