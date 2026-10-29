"""Shared angle convention for compact planar-fault geometry.

This module is intentionally small.  It keeps the nonlinear SMC forward path
and the standard nonlinear-result-to-mesh bridges on the same strike/dip
conversion without coupling those classes to each other.
"""

from __future__ import annotations

import numpy as np


def validate_oriented_reference_dip(dip, name="dip"):
    """Validate oriented-reference dip and preserve its input representation.

    Accepted values are finite degrees in ``[-90, 0) U (0, 180)``. Values
    near zero and the horizontal endpoint 180 are rejected because they cannot
    project a top edge to a different depth.
    """
    try:
        values = np.asarray(dip, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be finite and in [-90, 0) U (0, 180) degrees"
        ) from exc
    invalid = (
        ~np.isfinite(values)
        | (values < -90.0)
        | (values >= 180.0)
        | np.isclose(values, 0.0, rtol=0.0, atol=1e-12)
    )
    if np.any(invalid):
        indices = np.argwhere(invalid).reshape(-1)[:5].tolist()
        raise ValueError(
            f"{name} must be finite and in [-90, 0) U (0, 180) degrees; "
            f"invalid value indices: {indices}"
        )
    return values


def normalize_oriented_reference_dip(dip, name="dip"):
    """Return validated oriented-reference dip in continuous ``(0, 180)``."""
    values = validate_oriented_reference_dip(dip, name=name)
    normalized = np.where(values < 0.0, 180.0 + values, values)
    return float(normalized) if normalized.ndim == 0 else normalized


def canonicalize_compact_fault_angles(strike, dip):
    """Return the CSI solver representation of one compact-fault angle pair.

    Parameters
    ----------
    strike, dip : float
        Input geographic strike and dip in degrees. Strike is clockwise from
        North. Dip accepts the native interval ``[0, 180]`` and the historical
        signed compatibility interval ``[-90, 0)``.

    Returns
    -------
    tuple of (float, float, bool)
        Canonical ``(strike, dip, side_flipped)``. The returned strike is in
        ``[0, 360)`` and dip is in ``[0, 90]``. ``side_flipped`` records
        whether conversion added 180 degrees to strike.

    Notes
    -----
    This conversion changes geometry coordinates only. Slip components and
    rake keep their sampled meaning and are never transformed here.
    """
    try:
        strike = float(strike)
        dip = float(dip)
    except (TypeError, ValueError) as exc:
        raise ValueError("strike and dip must be finite numeric values") from exc
    if not np.isfinite(strike) or not np.isfinite(dip):
        raise ValueError("strike and dip must be finite numeric values")
    if dip < -90.0 or dip > 180.0:
        raise ValueError(f"dip must be within [-90, 180] degrees; got {dip}")

    strike = strike % 360.0
    side_flipped = False
    if dip > 90.0:
        dip = 180.0 - dip
        strike = (strike + 180.0) % 360.0
        side_flipped = True
    elif dip < 0.0:
        dip = -dip
        strike = (strike + 180.0) % 360.0
        side_flipped = True
    return float(strike), float(dip), side_flipped

