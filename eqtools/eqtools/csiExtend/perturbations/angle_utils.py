"""Shared angle handling helpers for geometry perturbations."""

import numpy as np


_DEGREE_UNITS = {'degree', 'degrees', 'deg'}
_RADIAN_UNITS = {'radian', 'radians', 'rad'}


def normalize_angle_unit(angle_unit):
    """Return canonical angle unit name: ``degrees`` or ``radians``."""
    if angle_unit is None:
        return 'degrees'
    unit = str(angle_unit).lower()
    if unit in _DEGREE_UNITS:
        return 'degrees'
    if unit in _RADIAN_UNITS:
        return 'radians'
    raise ValueError(
        "angle_unit must be one of "
        f"{sorted(_DEGREE_UNITS | _RADIAN_UNITS)}, got {angle_unit!r}."
    )


def angles_to_radians(values, angle_unit='degrees'):
    """Convert scalar or array-like angle values to radians."""
    unit = normalize_angle_unit(angle_unit)
    angles = np.asarray(values, dtype=float)
    if unit == 'degrees':
        return np.radians(angles)
    return angles.copy()


def angles_to_degrees(values, angle_unit='degrees'):
    """Convert scalar or array-like angle values to degrees."""
    unit = normalize_angle_unit(angle_unit)
    angles = np.asarray(values, dtype=float)
    if unit == 'radians':
        return np.degrees(angles)
    return angles.copy()


def circular_mean_radians(angles):
    """Circular mean for radians, returned in radians."""
    angles = np.asarray(angles, dtype=float).ravel()
    if angles.size == 0:
        raise ValueError("Cannot compute circular mean of an empty angle array.")
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def expand_angles_to_radians(values, length, angle_unit='degrees',
                             use_average=False, name='angle'):
    """Convert angles to radians and expand scalar input to ``length``.

    Accepted inputs:
    - scalar or length-1 array: broadcast to all nodes;
    - array of exactly ``length`` values: used per-node;
    - when ``use_average`` is True, length ``length`` input is circularly
      averaged and then broadcast.
    """
    angles = np.asarray(angles_to_radians(values, angle_unit), dtype=float).ravel()
    if angles.size == 1:
        return np.full(length, angles.item(), dtype=float)
    if angles.size != length:
        raise ValueError(
            f"{name} must be scalar or have length {length}; got length {angles.size}."
        )
    if use_average:
        return np.full(length, circular_mean_radians(angles), dtype=float)
    return angles.copy()
