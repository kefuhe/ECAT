"""Display-only value, color-limit, and palette helpers for trace editing."""

from __future__ import annotations

import numpy as np


def display_values(background):
    """Return display-scaled values without changing the scientific array."""

    factor = float(background.style.get("display_factor", 1.0))
    return np.asarray(background.values, dtype=float) * factor, factor


def color_limits(background):
    """Resolve stable color limits from the full valid observation field.

    Explicit ``vmin``/``vmax`` and the automatic central percentile follow
    the downsample raw-check-plot contract.  The returned range is global for
    the layer and therefore remains unchanged while the map is zoomed.
    """

    values, _ = display_values(background)
    finite = values[np.asarray(background.valid_mask, dtype=bool)]
    if finite.size == 0:
        raise ValueError("Trace-editor background has no finite display values.")

    percentile = float(background.style.get("auto_percentile", 99.0))
    if not 0.0 < percentile <= 100.0:
        raise ValueError(
            "Trace-editor auto_percentile must be within (0, 100]."
        )
    tail = (100.0 - percentile) / 2.0
    auto_min, auto_max = np.nanpercentile(
        finite,
        [tail, 100.0 - tail],
    )
    explicit_min = background.style.get("vmin")
    explicit_max = background.style.get("vmax")
    lower = float(auto_min if explicit_min is None else explicit_min)
    upper = float(auto_max if explicit_max is None else explicit_max)

    if explicit_min is None or explicit_max is None:
        if bool(background.style.get("symmetry", True)):
            limit = max(abs(lower), abs(upper))
            lower, upper = -limit, limit

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Trace-editor color limits must be finite.")
    if lower >= upper:
        if explicit_min is not None or explicit_max is not None:
            raise ValueError("Trace-editor vmin must be smaller than vmax.")
        center = float(np.nanmedian(finite))
        spread = max(abs(center) * 0.05, 1.0)
        lower, upper = center - spread, center + spread
    return float(lower), float(upper)


def matplotlib_palette(name, *, size=256):
    """Sample one Matplotlib colormap in its authoritative value direction."""

    size = int(size)
    if size < 2:
        raise ValueError("A display palette requires at least two colors.")
    name = str(name or "RdBu_r").strip()
    if name.startswith("cmc."):
        try:
            import cmcrameri.cm  # noqa: F401 - registers cmc.* colormaps
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The selected cmc.* colormap requires cmcrameri. Install "
                "the ECAT interaction extra with: python -m pip install -e "
                "\".[interaction]\""
            ) from exc
    try:
        from matplotlib import colormaps
        from matplotlib.colors import to_hex

        cmap = colormaps[name]
    except (KeyError, ModuleNotFoundError) as exc:
        raise ValueError(f"Unknown trace-editor colormap: {name!r}.") from exc
    positions = np.linspace(0.0, 1.0, size)
    return [to_hex(cmap(position), keep_alpha=False) for position in positions]


__all__ = ["color_limits", "display_values", "matplotlib_palette"]
