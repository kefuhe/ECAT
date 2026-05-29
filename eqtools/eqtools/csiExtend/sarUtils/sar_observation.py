import numpy as np

from .sar_conventions import InputValueConvention, ObservationType
from .sar_geometry import build_projection_vector


def require_positive_wavelength(wavelength):
    if wavelength is None:
        raise ValueError("phase_los requires a positive wavelength.")
    try:
        wavelength = float(wavelength)
    except (TypeError, ValueError):
        raise ValueError("phase_los requires a positive wavelength.") from None
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("phase_los requires a positive wavelength.")
    return wavelength


def unwrapped_phase_to_los(phase, wavelength):
    wavelength = require_positive_wavelength(wavelength)
    return np.asarray(phase) * wavelength / (-4.0 * np.pi)


def convert_observation_values(values, spec):
    """
    Convert raw product values to the scalar value represented by `spec`'s
    projection vector.
    """
    values = np.asarray(values)

    if spec.observation_type == ObservationType.PHASE_LOS:
        if spec.input_value_convention != InputValueConvention.UNWRAPPED_PHASE:
            raise ValueError("phase_los expects input_value_convention='unwrapped_phase'.")
        return unwrapped_phase_to_los(values, spec.wavelength)

    if spec.observation_type == ObservationType.LOS_DISPLACEMENT:
        if spec.input_value_convention == InputValueConvention.TOWARD_SATELLITE:
            return values.copy()
        if spec.input_value_convention == InputValueConvention.AWAY_FROM_SATELLITE:
            return -values
        raise ValueError(
            "los_displacement expects input_value_convention='toward_satellite' "
            "or 'away_from_satellite'."
        )

    if spec.observation_type == ObservationType.AZIMUTH_OFFSET:
        if spec.input_value_convention == InputValueConvention.ALONG_HEADING:
            return values.copy()
        if spec.input_value_convention == InputValueConvention.OPPOSITE_HEADING:
            return -values
        raise ValueError(
            "azimuth_offset expects input_value_convention='along_heading' "
            "or 'opposite_heading'."
        )

    raise ValueError(f"Unsupported observation_type: {spec.observation_type}.")


def prepare_observation_for_csi(values, azimuth, incidence, spec):
    data = convert_observation_values(values, spec)
    projection = build_projection_vector(azimuth, incidence, spec)
    return data, projection
