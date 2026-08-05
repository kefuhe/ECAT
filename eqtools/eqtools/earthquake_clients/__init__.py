"""Earthquake download and plotting clients with lazy public imports.

Keeping this package initializer light lets the project-based map viewer use
packaged background resources without importing Cartopy, Matplotlib, or Plotly.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "USGSClient": (".clients.usgs_client", "USGSClient"),
    "GCMTClient": (".clients.gcmt_client", "GCMTClient"),
    "IRISClient": (".clients.iris_client", "IRISClient"),
    "EarthquakeClientFactory": (".clients", "EarthquakeClientFactory"),
    "logger": (".clients.logging_config", "logger"),
}


def __getattr__(name):
    if name == "read_gmt_lines":
        return import_module("..gmttools", __name__).read_gmt_lines
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_EXPORTS[name]
    return getattr(import_module(module_name, __name__), attribute)


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | {"read_gmt_lines"})


__all__ = [
    "EarthquakeClientFactory",
    "GCMTClient",
    "IRISClient",
    "USGSClient",
    "logger",
    "read_gmt_lines",
]
