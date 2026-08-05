"""Read-only research map viewer for ECAT scientific products.

The package keeps its data model and loaders independent from Dash and Plotly.
Importing :mod:`eqtools.map_viewer` therefore does not add a web dependency to
downsampling, BLSE/VCE, SMC, or constraint code.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "LayerCatalog": (".catalog", "LayerCatalog"),
    "LayerMetadata": (".models", "LayerMetadata"),
    "LayerPayload": (".models", "LayerPayload"),
    "LayerSpec": (".models", "LayerSpec"),
    "ViewerProject": (".models", "ViewerProject"),
    "ViewerState": (".models", "ViewerState"),
    "create_app": (".app", "create_app"),
    "load_viewer_project": (".project", "load_viewer_project"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_EXPORTS[name]
    return getattr(import_module(module_name, __name__), attribute)


__all__ = list(_LAZY_EXPORTS)
