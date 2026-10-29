"""Compatibility import for :mod:`eqtools.csiExtend.geometry.fault_contours`.

The geometry subpackage owns the implementation. Existing imports, including
explicit helper imports, resolve to the same objects. Runtime code imports the
owner directly; no geometry, scientific defaults or registry state is copied.
"""

from ..geometry import fault_contours as _implementation

__all__ = getattr(
    _implementation, "__all__",
    [name for name in vars(_implementation) if not name.startswith("_")],
)


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
