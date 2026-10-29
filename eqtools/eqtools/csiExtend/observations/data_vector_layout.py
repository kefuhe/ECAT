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


def data_observation_vector(data, *, vertical=True):
    """Return one CSI observation object in canonical inversion-row order."""
    dtype = str(getattr(data, "dtype", "")).lower()
    if dtype == "gps":
        return gps_component_major_vector(
            data.vel_enu,
            vertical=vertical,
            name=f"{getattr(data, 'name', 'GPS')} GPS observations",
        )
    if dtype in {"insar", "leveling"}:
        return np.asarray(data.vel, dtype=float).reshape(-1)
    if dtype in {"opticorr", "optical"}:
        return np.concatenate(
            (
                np.asarray(data.east, dtype=float).reshape(-1),
                np.asarray(data.north, dtype=float).reshape(-1),
            )
        )
    if dtype == "crossfaultoffset":
        return np.asarray(data.data_vector, dtype=float).reshape(-1)
    raise ValueError(f"Unsupported data type: {getattr(data, 'dtype', None)}")


def data_synthetic_vector(data, *, vertical=True):
    """Return one CSI synthetic object in canonical inversion-row order.

    This is the prediction-side inverse of :func:`prepare_data_synthetic_fields`.
    Keeping both directions beside the observation layout prevents statistics
    and publication code from maintaining separate data-type row conventions.
    """
    dtype = str(getattr(data, "dtype", "")).lower()
    if dtype == "gps":
        return gps_component_major_vector(
            data.synth,
            vertical=vertical,
            name=f"{getattr(data, 'name', 'GPS')} GPS synthetics",
        )
    if dtype in {"insar", "leveling"}:
        return np.asarray(data.synth, dtype=float).reshape(-1)
    if dtype in {"opticorr", "optical"}:
        return np.concatenate(
            (
                np.asarray(data.east_synth, dtype=float).reshape(-1),
                np.asarray(data.north_synth, dtype=float).reshape(-1),
            )
        )
    if dtype == "crossfaultoffset":
        synthetic = getattr(data, "synth_vector", None)
        if synthetic is None:
            synthetic = data.synth
        return np.asarray(synthetic, dtype=float).reshape(-1)
    raise ValueError(f"Unsupported data type: {getattr(data, 'dtype', None)}")


def prepare_data_synthetic_fields(data, vector, *, vertical=True):
    """Validate and prepare CSI synthetic fields without mutating ``data``.

    The returned mapping supports prevalidated multi-dataset publication: every
    dataset can be prepared first, then assigned only after all layouts have
    been validated. The assignments are not a general transaction for arbitrary
    custom setters.
    """
    dtype = str(getattr(data, "dtype", "")).lower()
    vector = np.asarray(vector, dtype=float).reshape(-1)
    name = str(getattr(data, "name", dtype or "dataset"))

    if dtype == "gps":
        template = np.zeros_like(np.asarray(data.vel_enu, dtype=float))
        return {
            "synth": assign_gps_component_major_vector(
                template,
                vector,
                vertical=vertical,
                name=f"{name} GPS synthetic vector",
            )
        }

    if dtype in {"insar", "leveling"}:
        template_source = getattr(data, "vel", None)
        if template_source is None:
            template_source = getattr(data, "synth", None)
        if template_source is None:
            raise ValueError(f"{name} has no observation or synthetic layout")
        template = np.asarray(template_source, dtype=float)
        if vector.size != template.size:
            raise ValueError(
                f"{name} synthetic vector has {vector.size} rows; "
                f"expected {template.size}"
            )
        return {"synth": vector.reshape(template.shape).copy()}

    if dtype in {"opticorr", "optical"}:
        east = np.asarray(data.east, dtype=float)
        north = np.asarray(data.north, dtype=float)
        expected = east.size + north.size
        if vector.size != expected:
            raise ValueError(
                f"{name} synthetic vector has {vector.size} rows; "
                f"expected {expected}"
            )
        split = east.size
        return {
            "east_synth": vector[:split].reshape(east.shape).copy(),
            "north_synth": vector[split:].reshape(north.shape).copy(),
        }

    if dtype == "crossfaultoffset":
        fields = (
            ("fault_parallel", "synth_parallel"),
            ("fault_perpendicular", "synth_perpendicular"),
            ("fault_vertical", "synth_vertical"),
        )
        prepared = {}
        offset = 0
        for observed_name, synthetic_name in fields:
            observed = getattr(data, observed_name, None)
            if observed is None:
                prepared[synthetic_name] = None
                continue
            observed = np.asarray(observed, dtype=float)
            end = offset + observed.size
            if end > vector.size:
                raise ValueError(
                    f"{name} synthetic vector ended before {observed_name}"
                )
            prepared[synthetic_name] = vector[offset:end].reshape(
                observed.shape
            ).copy()
            offset = end
        if offset != vector.size:
            raise ValueError(
                f"{name} synthetic vector has {vector.size} rows; expected {offset}"
            )
        prepared["synth"] = vector.copy()
        return prepared

    raise ValueError(f"Unsupported data type: {getattr(data, 'dtype', None)}")


def write_data_synthetic_vector(data, vector, *, vertical=True):
    """Publish one canonical prediction vector to CSI synthetic fields.

    Parameters
    ----------
    data : CSI data object
        Supported types are GPS, InSAR, optical correlation, leveling, and
        cross-fault offsets. The observation fields define the CSI-facing
        storage layout.
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
    prepared = prepare_data_synthetic_fields(
        data,
        vector,
        vertical=vertical,
    )
    for field, value in prepared.items():
        setattr(data, field, value)
    published = getattr(data, "synth", None)
    if published is not None:
        return published
    return np.asarray(vector, dtype=float).reshape(-1)
