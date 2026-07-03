"""Utilities for nonlinear prior-bound format handling.

The SMC sampler currently consumes scipy-style Uniform distributions,
``[Uniform, loc, scale]``.  User-facing configs can be clearer when they use
``[Uniform, lower, upper]``.  This module keeps that conversion explicit and
centralized so old and new nonlinear config entry points do not silently
interpret the same numbers differently.
"""

from __future__ import annotations

import copy
from typing import Any


LOWER_RANGE = "lower_range"
LOWER_UPPER = "lower_upper"
VALID_PRIOR_BOUNDS_FORMATS = {LOWER_RANGE, LOWER_UPPER}


def validate_prior_bounds_format(bounds_format: str) -> str:
    """Return a normalized bounds-format name or raise ``ValueError``."""
    if bounds_format is None:
        raise ValueError("prior_bounds_format cannot be None")
    normalized = str(bounds_format).strip().lower()
    if normalized not in VALID_PRIOR_BOUNDS_FORMATS:
        valid = ", ".join(sorted(VALID_PRIOR_BOUNDS_FORMATS))
        raise ValueError(
            f"Unknown prior_bounds_format '{bounds_format}'. Expected one of: {valid}"
        )
    return normalized


def is_distribution_spec(value: Any) -> bool:
    """Return True when ``value`` looks like ``[Distribution, ...]``."""
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and isinstance(value[0], str)
    )


def normalize_prior_bound(
    bound: Any,
    bounds_format: str,
    *,
    context: str = "prior bound",
) -> Any:
    """Normalize one distribution spec to the sampler's internal format.

    Only ``Uniform`` changes between supported user formats.  Other
    distributions keep their existing scipy-style arguments.
    """
    if not is_distribution_spec(bound):
        return copy.deepcopy(bound)

    normalized_format = validate_prior_bounds_format(bounds_format)
    dist = bound[0]
    if dist != "Uniform":
        return list(copy.deepcopy(bound))

    if len(bound) != 3:
        raise ValueError(
            f"{context}: Uniform bounds must be [Uniform, lower, range] "
            f"or [Uniform, lower, upper], got {bound!r}"
        )

    lower = float(bound[1])
    third = float(bound[2])
    if normalized_format == LOWER_RANGE:
        scale = third
        if scale <= 0:
            raise ValueError(
                f"{context}: Uniform range must be positive under "
                f"prior_bounds_format={LOWER_RANGE}, got {scale!r}"
            )
    else:
        upper = third
        if upper <= lower:
            raise ValueError(
                f"{context}: upper must be greater than lower under "
                f"prior_bounds_format={LOWER_UPPER}, got lower={lower!r}, "
                f"upper={upper!r}"
            )
        scale = upper - lower

    return [dist, lower, scale]


def normalize_prior_bounds_tree(
    value: Any,
    bounds_format: str,
    *,
    context: str = "prior bounds",
) -> Any:
    """Recursively normalize distribution specs in dictionaries/lists."""
    if is_distribution_spec(value):
        return normalize_prior_bound(value, bounds_format, context=context)

    if isinstance(value, dict):
        return {
            key: normalize_prior_bounds_tree(
                item,
                bounds_format,
                context=f"{context}.{key}",
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            normalize_prior_bounds_tree(
                item,
                bounds_format,
                context=f"{context}[{idx}]",
            )
            for idx, item in enumerate(value)
        ]

    return copy.deepcopy(value)
