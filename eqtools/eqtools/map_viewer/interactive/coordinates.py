"""Single coordinate-conversion boundary for the ECAT trace editor."""

import numpy as np

from .models import canonical_longitude


WEB_MERCATOR_RADIUS_M = 6_378_137.0
WEB_MERCATOR_MAX_LATITUDE = 85.0511287798066


def lonlat_to_web_mercator(longitude, latitude):
    """Convert WGS84 lon/lat degrees to spherical Web Mercator meters.

    The conversion is used only by the Bokeh display adapter. Scientific and
    exported path coordinates remain canonical longitude/latitude.
    """

    longitude = canonical_longitude(longitude)
    latitude = np.asarray(latitude, dtype=float)
    if longitude.shape != latitude.shape:
        raise ValueError("Longitude and latitude must have identical shapes.")
    if not np.all(np.isfinite(longitude)) or not np.all(np.isfinite(latitude)):
        raise ValueError("Longitude and latitude must be finite.")
    if np.any(np.abs(latitude) > WEB_MERCATOR_MAX_LATITUDE):
        raise ValueError(
            "Web Mercator display latitude must be within "
            f"[-{WEB_MERCATOR_MAX_LATITUDE}, {WEB_MERCATOR_MAX_LATITUDE}]."
        )
    longitude_rad = np.deg2rad(longitude)
    latitude_rad = np.deg2rad(latitude)
    x = WEB_MERCATOR_RADIUS_M * longitude_rad
    y = WEB_MERCATOR_RADIUS_M * np.log(
        np.tan(np.pi / 4.0 + latitude_rad / 2.0)
    )
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def web_mercator_to_lonlat(x, y):
    """Convert spherical Web Mercator meters to canonical WGS84 lon/lat."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Web Mercator x and y must have identical shapes.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Web Mercator x and y must be finite.")
    longitude = canonical_longitude(np.rad2deg(x / WEB_MERCATOR_RADIUS_M))
    latitude = np.rad2deg(
        2.0 * np.arctan(np.exp(y / WEB_MERCATOR_RADIUS_M)) - np.pi / 2.0
    )
    return np.asarray(longitude, dtype=float), np.asarray(latitude, dtype=float)


__all__ = [
    "WEB_MERCATOR_MAX_LATITUDE",
    "WEB_MERCATOR_RADIUS_M",
    "lonlat_to_web_mercator",
    "web_mercator_to_lonlat",
]
