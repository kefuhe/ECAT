"""Canonical observation-row layouts shared by ECAT inversion paths.

CSI assembles GPS observations, Green functions, covariance blocks, and frame
transform estimators in component-major order: all east rows, then all north
rows, then (when requested) all up rows.  Nonlinear likelihoods must preserve
that order so every residual row is paired with the intended covariance row.
"""

from __future__ import annotations

import numpy as np


def gps_component_major_vector(values, *, vertical=True, name="GPS values"):
    """Return GPS values as ``E(all), N(all), [U(all)]``.

    Parameters
    ----------
    values : array-like, shape (n_stations, n_components)
        Station-by-component ENU values.
    vertical : bool
        Include the up component when true; otherwise return horizontal EN.
    name : str
        Context included in validation errors.
    """
    values = np.asarray(values, dtype=float)
    n_components = 3 if vertical else 2
    if values.ndim != 2 or values.shape[1] < n_components:
        raise ValueError(
            f"{name} must have shape (n_stations, at least {n_components}); "
            f"got {values.shape}"
        )
    return values[:, :n_components].T.reshape(-1)


def assign_gps_component_major_vector(
    values,
    vector,
    *,
    vertical=True,
    name="GPS vector",
):
    """Return a copy of ``values`` updated from a component-major vector.

    Horizontal assignment preserves the existing up column.  The inverse
    reshape is explicit because ordinary ``reshape(values.shape)`` would
    interpret the vector in station-major order.
    """
    values = np.asarray(values, dtype=float)
    n_components = 3 if vertical else 2
    if values.ndim != 2 or values.shape[1] < n_components:
        raise ValueError(
            f"GPS values must have shape (n_stations, at least {n_components}); "
            f"got {values.shape}"
        )

    vector = np.asarray(vector, dtype=float).reshape(-1)
    expected = values.shape[0] * n_components
    if vector.size != expected:
        raise ValueError(f"{name} has {vector.size} rows; expected {expected}")

    updated = values.copy()
    updated[:, :n_components] = vector.reshape(n_components, values.shape[0]).T
    return updated


def write_data_synthetic_vector(data, vector, *, vertical=True):
    """Publish one canonical prediction vector to CSI synthetic fields.

    Parameters
    ----------
    data : CSI data object
        Supported types are GPS, InSAR, leveling, and cross-fault offsets.
        The object must already contain a ``synth`` array whose shape defines
        the CSI-facing storage layout.
    vector : array-like, shape (n_observations,)
        Complete prediction in the same row order as the observation and
        covariance vectors.  For GPS this is component-major order:
        ``E(all), N(all), [U(all)]``.
    vertical : bool
        Include the GPS up component when true.  Ignored by scalar datasets.

    Returns
    -------
    numpy.ndarray
        The published ``data.synth`` array.

    Notes
    -----
    This function only translates between the inversion-vector layout and the
    CSI object layout.  It does not build Green functions or add correction
    terms; callers must pass the already complete prediction exactly once.
    """
    dtype = str(getattr(data, "dtype", "")).lower()
    vector = np.asarray(vector, dtype=float).reshape(-1)

    if dtype == "gps":
        data.synth = assign_gps_component_major_vector(
            data.synth,
            vector,
            vertical=vertical,
            name=f"{getattr(data, 'name', 'GPS')} GPS synthetic vector",
        )
        return data.synth

    if dtype in {"insar", "leveling", "crossfaultoffset"}:
        template = np.asarray(data.synth)
        if vector.size != template.size:
            name = getattr(data, "name", dtype)
            raise ValueError(
                f"{name} synthetic vector has {vector.size} rows; "
                f"expected {template.size}"
            )
        data.synth = vector.reshape(template.shape)
        if dtype == "crossfaultoffset":
            # Some CSI consumers read the flattened cross-fault prediction.
            data.synth_vector = vector.copy()
        return data.synth

    raise ValueError(f"Unsupported data type: {getattr(data, 'dtype', None)}")
