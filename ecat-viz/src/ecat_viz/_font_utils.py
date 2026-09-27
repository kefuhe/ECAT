"""
_font_utils.py — CJK font detection and text font baking utilities.

Public API
----------
list_chinese_fonts   : Probe system for available CJK fonts
bake_text_fonts      : Fix Text artist fonts before PlotStyle.reset()
"""

import json
import os
import tempfile
import time
import warnings
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

# None means unprobed; a dictionary may legitimately contain two None values.
_font_probe_cache: Optional[Dict[str, Optional[str]]] = None


def _get_font_cache_path() -> Path:
    """Locate the optional cache without creating directories on read."""
    return Path.home() / FONT_CACHE_DIR_NAME / FONT_CACHE_FILE_NAME


def _load_font_cache() -> Optional[Dict]:
    """Unreadable, expired or malformed JSON is a cache miss, never a plot error."""
    try:
        path = _get_font_cache_path()
        if time.time() - path.stat().st_mtime > FONT_CACHE_EXPIRY_DAYS * 86400:
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
        fonts = data.get('cjk_fonts') if isinstance(data, dict) else None
        if (not isinstance(fonts, dict) or set(fonts) != {'sans', 'serif'}
                or any(value is not None and not isinstance(value, str)
                       for value in fonts.values())):
            return None
        return data
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def _save_font_cache(data: Dict) -> None:
    """Best-effort atomic persistence; a read-only home does not block plotting."""
    temporary = None
    try:
        path = _get_font_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False)
        os.replace(temporary, path)
    except (OSError, RuntimeError, ValueError, TypeError):
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
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


def _probe_chinese_fonts(refresh: bool = False) -> Dict[str, Optional[str]]:
    """Own the memory cache; refresh replaces it without reading the disk cache."""
    global _font_probe_cache
    if not refresh and _font_probe_cache is not None:
        return _font_probe_cache.copy()
    cache = None if refresh else _load_font_cache()
    if cache is not None:
        cached = cache['cjk_fonts']
        if all(name is None or _font_exists(name) for name in cached.values()):
            _font_probe_cache = cached.copy()
            return cached.copy()

    try:
        from matplotlib.font_manager import fontManager
        available = {f.name for f in fontManager.ttflist}
    except Exception:
        available = set()
    result = {
        'sans': next((f for f in _CHINESE_SANS_CANDIDATES if f in available), None),
        'serif': next((f for f in _CHINESE_SERIF_CANDIDATES if f in available), None),
    }
    _font_probe_cache = result
    _save_font_cache({'cjk_fonts': result})
    return result.copy()


def list_chinese_fonts(refresh: bool = False) -> Dict[str, Optional[str]]:
    """
    Return the best available CJK font names on this system.

    Parameters
    ----------
    refresh : bool, optional
        If True, bypass both caches and re-probe Matplotlib's active fontManager.
        Newly installed fonts must first be visible to Matplotlib (restart or
        register them with fontManager.addfont). Default is False.

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
    return _probe_chinese_fonts(refresh=refresh)


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
