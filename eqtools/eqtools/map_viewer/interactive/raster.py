"""Zoom-aware rasterization of structured observation grids for Bokeh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coordinates import lonlat_to_web_mercator
from .display import display_values
from .models import EditorBackground


_DATASHADER_INSTALL_HINT = (
    "Continuous trace-editor imagery requires Datashader >=0.19,<0.20. "
    "Install the ECAT interaction extra with: python -m pip install -e "
    '".[interaction]"'
)


def require_datashader():
    """Import the optional rasterizer and enforce ECAT's tested version range."""

    try:
        import datashader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_DATASHADER_INSTALL_HINT) from exc
    version = tuple(
        int(part) for part in str(datashader.__version__).split(".")[:2]
    )
    if version < (0, 19) or version >= (0, 20):
        raise RuntimeError(
            "ECAT trace editing requires Datashader >=0.19,<0.20; installed "
            f"version is {datashader.__version__}."
        )
    return datashader


def supports_quadmesh(background):
    """Return whether a background has structured 2-D grid topology."""

    if not isinstance(background, EditorBackground):
        raise TypeError("supports_quadmesh expects an EditorBackground.")
    return bool(
        background.values.ndim == 2
        and min(background.values.shape) >= 2
        and np.all(np.isfinite(background.longitude))
        and np.all(np.isfinite(background.latitude))
    )


@dataclass(frozen=True)
class RasterFrame:
    """One display-only RGBA frame anchored in Web Mercator coordinates."""

    image: np.ndarray
    x: float
    y: float
    dw: float
    dh: float

    def source_data(self):
        """Return the single-image columns consumed by Bokeh ``image_rgba``."""

        return {
            "image": [self.image],
            "x": [self.x],
            "y": [self.y],
            "dw": [self.dw],
            "dh": [self.dh],
        }


class QuadmeshRasterizer:
    """Rasterize a full immutable 2-D observation grid for the current view."""

    def __init__(self, background, *, palette, low, high):
        require_datashader()
        import xarray as xr

        if not supports_quadmesh(background):
            raise ValueError(
                "Quadmesh rendering requires a structured 2-D observation grid."
            )
        self.background = background
        self.palette = tuple(palette)
        self.low = float(low)
        self.high = float(high)
        if self.low >= self.high:
            raise ValueError("Quadmesh color limits must satisfy low < high.")

        longitude = np.asarray(background.longitude, dtype=float)
        latitude = np.asarray(background.latitude, dtype=float)
        self.x, self.y = lonlat_to_web_mercator(longitude, latitude)
        values, _ = display_values(background)
        self.values = np.where(background.valid_mask, values, np.nan)
        valid = (
            np.asarray(background.valid_mask, dtype=bool)
            & np.isfinite(self.x)
            & np.isfinite(self.y)
        )
        if not np.any(valid):
            raise ValueError("Quadmesh background has no finite coordinates.")
        self.bounds = (
            float(np.nanmin(self.x[valid])),
            float(np.nanmax(self.x[valid])),
            float(np.nanmin(self.y[valid])),
            float(np.nanmax(self.y[valid])),
        )
        self._data = xr.DataArray(
            self.values,
            name="value",
            dims=("row", "column"),
            coords={
                "mercator_x": (("row", "column"), self.x),
                "mercator_y": (("row", "column"), self.y),
            },
        )

    def padded_ranges(self, *, fraction=0.02):
        """Return initial x/y ranges with a small non-scientific view margin."""

        xmin, xmax, ymin, ymax = self.bounds
        xpad = max((xmax - xmin) * float(fraction), 1.0)
        ypad = max((ymax - ymin) * float(fraction), 1.0)
        return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)

    @staticmethod
    def _range(values, fallback):
        if values is None:
            return fallback
        start, end = (float(values[0]), float(values[1]))
        if not np.isfinite(start) or not np.isfinite(end):
            return fallback
        lower, upper = sorted((start, end))
        if lower == upper:
            pad = max(abs(lower) * 1e-9, 1.0)
            lower, upper = lower - pad, upper + pad
        return lower, upper

    def render(self, *, x_range=None, y_range=None, width=1200, height=800):
        """Rasterize the current view without changing source coordinates/values."""

        ds = require_datashader()
        from datashader import transfer_functions as tf

        width = min(max(int(width), 64), 4096)
        height = min(max(int(height), 64), 4096)
        fallback_x, fallback_y = self.padded_ranges()
        x_range = self._range(x_range, fallback_x)
        y_range = self._range(y_range, fallback_y)
        canvas = ds.Canvas(
            plot_width=width,
            plot_height=height,
            x_range=x_range,
            y_range=y_range,
        )
        aggregate = canvas.quadmesh(
            self._data,
            x="mercator_x",
            y="mercator_y",
            agg=ds.mean("value"),
        )
        shaded = tf.shade(
            aggregate,
            cmap=list(self.palette),
            span=(self.low, self.high),
            how="linear",
            alpha=255,
            min_alpha=255,
        )
        rgba = np.ascontiguousarray(shaded.data, dtype=np.uint32)
        return RasterFrame(
            image=rgba,
            x=x_range[0],
            y=y_range[0],
            dw=x_range[1] - x_range[0],
            dh=y_range[1] - y_range[0],
        )


__all__ = [
    "QuadmeshRasterizer",
    "RasterFrame",
    "require_datashader",
    "supports_quadmesh",
]
