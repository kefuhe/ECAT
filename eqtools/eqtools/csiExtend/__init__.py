from importlib import import_module


_LAZY_EXPORTS = {
    "BayesianAdaptiveTriangularPatches": (
        ".BayesianAdaptiveTriangularPatches",
        "BayesianAdaptiveTriangularPatches",
    ),
    "BayesianTriFaultBase": (".bayesian_perturbation_base", "BayesianTriFaultBase"),
    "PerturbationBase": (".bayesian_perturbation_base", "PerturbationBase"),
    "PerturbationRegistry": (".bayesian_perturbation_base", "PerturbationRegistry"),
    "SharedFaultInfo": (".bayesian_perturbation_base", "SharedFaultInfo"),
    "track_mesh_update": (".bayesian_perturbation_base", "track_mesh_update"),
}


__all__ = tuple(_LAZY_EXPORTS)


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
