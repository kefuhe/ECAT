from importlib import import_module

from .gmttools import ReadGMTLines


def __getattr__(name):
    if name == "csiExtend":
        return import_module(".csiExtend", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
