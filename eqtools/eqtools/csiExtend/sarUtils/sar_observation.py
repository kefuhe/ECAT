"""Canonicalize SAR scalar observations before loading them into CSI.

Every returned scalar is paired with an ENU projection under the contract

    scalar_observation = ENU_displacement dot projection

Scalar conversion is deliberately independent of angle geometry and
acquisition look side.  Those inputs determine the projection axis, not the
positive direction of the raw scalar raster.
"""

import numpy as np

from .sar_conventions import ObservationType, RawValueConvention
from .sar_geometry import build_projection_vector


def require_positive_wavelength(wavelength):
    if wavelength is None:
        raise ValueError("unwrapped_phase requires a positive wavelength.")
    try:
        wavelength = float(wavelength)
    except (TypeError, ValueError):
        raise ValueError("unwrapped_phase requires a positive wavelength.") from None
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("unwrapped_phase requires a positive wavelength.")
    return wavelength


def unwrapped_phase_to_los(phase, wavelength):
    """Convert radians to ground-to-sensor LOS displacement.

    The reader uses

        phase = -(4*pi/wavelength) * los_displacement

    and therefore ``los_displacement = -wavelength*phase/(4*pi)``.
    """

    wavelength = require_positive_wavelength(wavelength)
    return np.asarray(phase) * wavelength / (-4.0 * np.pi)


def convert_observation_values(values, spec):
    """Convert only raw scalar values to their canonical positive direction.

    LOS/range targets are positive ground-to-sensor; azimuth targets are
    positive along heading.  Projection construction is intentionally absent
    here: changing scalar convention must never silently change angle meaning
    or acquisition look side.
    """

    values = np.asarray(values)

    if spec.observation_type == ObservationType.UNWRAPPED_PHASE:
        if spec.raw_value_convention != RawValueConvention.UNWRAPPED_PHASE:
            raise ValueError(
                "unwrapped_phase expects raw_value_convention='unwrapped_phase'."
            )
        return unwrapped_phase_to_los(values, spec.wavelength)

    if spec.observation_type == ObservationType.LOS_DISPLACEMENT:
        if spec.raw_value_convention == RawValueConvention.TOWARD_SENSOR:
            return values.copy()
        if spec.raw_value_convention == RawValueConvention.AWAY_FROM_SENSOR:
            # Express an away-positive input on the toward-sensor target axis.
            return -values
        raise ValueError(
            "los_displacement expects raw_value_convention='toward_sensor' "
            "or 'away_from_sensor'."
        )

    if spec.observation_type == ObservationType.AZIMUTH_OFFSET:
        if spec.raw_value_convention == RawValueConvention.ALONG_HEADING:
            return values.copy()
        if spec.raw_value_convention == RawValueConvention.OPPOSITE_HEADING:
            # Express an opposite-heading input on the along-heading target axis.
            return -values
        raise ValueError(
            "azimuth_offset expects raw_value_convention='along_heading' "
            "or 'opposite_heading'."
        )

    raise ValueError(f"Unsupported observation_type: {spec.observation_type}.")


def prepare_observation_for_csi(
    values, azimuth, incidence, observation_spec, geometry_spec
):
    """Return a scalar/projection pair satisfying the CSI observation contract.

    Scalar and projection are canonicalized independently and then paired.  A
    legacy representation may multiply both by ``-1`` without changing the
    physical observation equation, but flipping only one of them changes the
    modeled displacement sign and is invalid.
    """

    data = convert_observation_values(values, observation_spec)
    projection = build_projection_vector(
        azimuth, incidence, observation_spec, geometry_spec
    )
    return data, projection
