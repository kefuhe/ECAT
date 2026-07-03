"""Lazy exports for csiExtend configuration classes.

Importing small helpers such as ``config.config_utils`` must not eagerly import
the inversion/config classes because those classes depend on solver modules.
Keeping this package initializer lazy avoids circular imports during solver
startup while preserving the historical public names.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "explorefaultConfig": (".explore_config", "ExploreFaultConfig"),
    "NonlinearGeometryConfig": (".nonlinear_geometry_config", "NonlinearGeometryConfig"),
    "BayesianMultiFaultsInversionConfig": (
        ".bayesian_config",
        "BayesianMultiFaultsInversionConfig",
    ),
    "BoundLSEInversionConfig": (".blse_config", "BoundLSEInversionConfig"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = tuple(_LAZY_EXPORTS)
