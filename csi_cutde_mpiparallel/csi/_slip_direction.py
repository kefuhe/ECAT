"""Shared parsing for CSI fault-source prediction components."""

_COMPONENT_KEYS = {
    "s": "strikeslip",
    "d": "dipslip",
    "t": "tensile",
    "c": "coupling",
}


def resolve_prediction_slipdir(fault, direction, green_functions):
    """Resolve one fault's prediction components without changing old modes.

    ``direction='source'`` follows the components declared when the source was
    assembled. Explicit legacy strings such as ``'sd'`` keep their established
    permissive behavior; only source-driven replay requires every declared GF.
    """
    if direction != "source":
        return direction

    slipdir = getattr(fault, "slipdir", None)
    if not isinstance(slipdir, str) or not slipdir:
        raise ValueError(
            f"Fault '{getattr(fault, 'name', '<unnamed>')}' has no active slipdir"
        )
    invalid = [char for char in slipdir if char not in _COMPONENT_KEYS]
    if invalid:
        raise ValueError(
            f"Fault '{getattr(fault, 'name', '<unnamed>')}' has unsupported "
            f"slipdir component(s): {''.join(dict.fromkeys(invalid))}"
        )
    for char in slipdir:
        key = _COMPONENT_KEYS[char]
        if green_functions.get(key) is None:
            raise ValueError(
                f"Fault '{getattr(fault, 'name', '<unnamed>')}' declares "
                f"component '{char}' but has no '{key}' Green function for "
                "the requested dataset"
            )
    return slipdir
