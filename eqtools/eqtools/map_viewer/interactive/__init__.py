"""Optional interactive geometry tools for ECAT map products.

The package keeps its models and trace I/O independent from Bokeh.  Importing
this module therefore does not add GUI dependencies to ECAT numerical code.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "EditHistory": (".models", "EditHistory"),
    "EditorBackground": (".models", "EditorBackground"),
    "InteractiveWorkspace": (".models", "InteractiveWorkspace"),
    "PathDraft": (".models", "PathDraft"),
    "ReferencePath": (".models", "ReferencePath"),
    "TraceEditorSession": (".models", "TraceEditorSession"),
    "read_reference_paths": (".trace_io", "read_reference_paths"),
    "run_trace_editor": (".bokeh_trace_editor", "run_trace_editor"),
    "save_trace": (".trace_io", "save_trace"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_EXPORTS[name]
    return getattr(import_module(module_name, __name__), attribute)


__all__ = list(_LAZY_EXPORTS)
