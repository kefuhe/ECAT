"""Shared corner-geometry contracts for decimated raster observations."""

from __future__ import annotations

import numpy as np


TRIANGLE = "triangle"
LEGACY_RECTANGLE = "legacy_rectangle"
QUADRILATERAL = "quadrilateral"

_RSP_COLUMN_MODES = {
    8: TRIANGLE,
    10: LEGACY_RECTANGLE,
    18: QUADRILATERAL,
}

_CORNER_WIDTH_MODES = {
    4: LEGACY_RECTANGLE,
    6: TRIANGLE,
    8: QUADRILATERAL,
}


def validate_triangular_hint(mode, triangular, *, context="corner geometry"):
    """Validate a legacy triangular hint against a resolved geometry mode."""
    if triangular is None:
        return
    if not isinstance(triangular, (bool, np.bool_)):
        raise TypeError("triangular must be True, False, or None")
    is_triangle = mode == TRIANGLE
    if bool(triangular) != is_triangle:
        requested = "triangular" if triangular else "rectangular"
        raise ValueError(
            f"{context} is {mode!r}, which conflicts with the explicit "
            f"triangular={triangular!r} ({requested}) hint"
        )


def resolve_rsp_corner_mode(num_columns, triangular=None):
    """Resolve a VarRes ``.rsp`` cell layout from its data-column count."""
    try:
        mode = _RSP_COLUMN_MODES[int(num_columns)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Unexpected .rsp format: "
            f"{num_columns} columns detected (expected 8, 10, or 18)"
        ) from exc
    validate_triangular_hint(mode, triangular, context=".rsp cell geometry")
    return mode


def classify_corner_array(corner, *, expected_rows=None):
    """Return the canonical geometry mode for an in-memory corner array.

    ``None`` and empty arrays represent point observations. Non-empty corner
    arrays are required to be finite ``(N, 4|6|8)`` arrays. The three widths
    represent a legacy diagonal rectangle, a triangle, and a full
    quadrilateral, respectively.
    """
    if corner is None:
        return None
    array = np.asarray(corner)
    if array.size == 0:
        return None
    if array.ndim != 2:
        raise ValueError("corner must be a two-dimensional array")
    try:
        mode = _CORNER_WIDTH_MODES[array.shape[1]]
    except KeyError as exc:
        raise ValueError(
            "corner must have 4 columns (legacy rectangle), 6 columns "
            "(triangle), or 8 columns (quadrilateral)"
        ) from exc
    if expected_rows is not None and array.shape[0] != int(expected_rows):
        raise ValueError(
            "corner row count does not match observation count: "
            f"{array.shape[0]} != {int(expected_rows)}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("corner coordinates must be finite")
    return mode
