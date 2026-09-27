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
    """Freeze generic font-family lists on existing Text artists.

    Call inside the active PlotStyle context. Each artist retains its explicit
    font names, font file, fallback order, size, weight and style. Generic
    families are expanded using the current rcParams; no single font is forced
    onto the whole figure. TeX text is left unchanged. Text added afterwards
    still uses the settings active when it is created. Repeated calls are safe.
    Failures warn once per call and leave the affected artist unchanged.
    """
    import matplotlib.text as _mtext

    failures = []
    generic = {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}
    for text in fig.findobj(match=_mtext.Text):
        if text.get_usetex():
            continue
        try:
            props = text.get_fontproperties()
            if props.get_file() is not None:
                continue
            families = props.get_family()
            expanded = []
            for family in families:
                key = family.lower()
                if key in generic:
                    # Style lists may end in their own generic family.
                    # Keep concrete fallbacks only, otherwise later calls
                    # re-expand that tail using a different global style.
                    candidates = [name for name in mpl.rcParams[f'font.{key}']
                                  if name.lower() not in generic]
                    if not candidates:
                        raise ValueError(f'empty font.{key} fallback list')
                    expanded.extend(candidates)
                else:
                    expanded.append(family)
            if expanded != families:
                updated = props.copy()
                updated.set_family(expanded)
                text.set_fontproperties(updated)
        except Exception as exc:
            failures.append(f'{type(exc).__name__}: {exc}')
    if failures:
        warnings.warn(
            f'Could not fix font families for {len(failures)} text object(s): '
            + '; '.join(dict.fromkeys(failures)),
            UserWarning, stacklevel=2,
        )
