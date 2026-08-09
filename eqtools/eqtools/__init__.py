from importlib import import_module

from ._version import __version__


def __getattr__(name):
    if name in {"csiExtend", "geoexport", "map_viewer"}:
        return import_module(f".{name}", __name__)
    if name == "ReadGMTLines":
        return import_module(".gmttools", __name__).ReadGMTLines
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", "ReadGMTLines", "csiExtend", "geoexport", "map_viewer"]
