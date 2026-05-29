import numpy as np

from csi.projection import (
    projection_from_enu_heading,
    projection_from_enu_horizontal_incidence,
)

from .sar_conventions import (
    AngleDirection,
    AngleUnit,
    AzimuthReference,
    IncidenceReference,
    InputAzimuthRole,
    LookSide,
    ObservationType,
    coerce_enum,
)


def normalize_azimuth(azimuth, reference, unit, direction):
    """
    Convert an input azimuth raster to ENU degrees.

    Returned azimuth uses 0=East and counterclockwise positive.
    """
    reference = coerce_enum(AzimuthReference, reference, "azimuth_reference")
    unit = coerce_enum(AngleUnit, unit, "azimuth_unit")
    direction = coerce_enum(AngleDirection, direction, "azimuth_direction")

    azimuth = np.asarray(azimuth)
    if unit == AngleUnit.RADIAN:
        azimuth = np.rad2deg(azimuth)

    if reference == AzimuthReference.EAST and direction == AngleDirection.COUNTERCLOCKWISE:
        enu = azimuth
    elif reference == AzimuthReference.EAST and direction == AngleDirection.CLOCKWISE:
        enu = -azimuth
    elif reference == AzimuthReference.NORTH and direction == AngleDirection.COUNTERCLOCKWISE:
        enu = 90.0 + azimuth
    elif reference == AzimuthReference.NORTH and direction == AngleDirection.CLOCKWISE:
        enu = 90.0 - azimuth
    else:
        raise ValueError("Invalid azimuth convention.")

    return np.mod(enu, 360.0)


def normalize_incidence(incidence, reference, unit):
    """
    Convert incidence to degrees measured from vertical.
    """
    reference = coerce_enum(IncidenceReference, reference, "incidence_reference")
    unit = coerce_enum(AngleUnit, unit, "incidence_unit")

    incidence = np.asarray(incidence)
    if unit == AngleUnit.RADIAN:
        incidence = np.rad2deg(incidence)
    if reference == IncidenceReference.ELEVATION:
        incidence = 90.0 - incidence
    return incidence


def horizontal_unit(azimuth_deg):
    alpha = np.deg2rad(np.asarray(azimuth_deg).reshape(-1))
    return np.column_stack((np.cos(alpha), np.sin(alpha)))


def rotate_cw90(vectors):
    vectors = np.asarray(vectors)
    return np.column_stack((vectors[:, 1], -vectors[:, 0]))


def rotate_ccw90(vectors):
    vectors = np.asarray(vectors)
    return np.column_stack((-vectors[:, 1], vectors[:, 0]))


def infer_look_side(input_azimuth_role, configured_look_side=LookSide.RIGHT):
    role = coerce_enum(InputAzimuthRole, input_azimuth_role, "input_azimuth_role")
    if role in (InputAzimuthRole.RIGHT_LOOK_AWAY, InputAzimuthRole.RIGHT_LOS_TOWARD):
        return LookSide.RIGHT
    if role in (InputAzimuthRole.LEFT_LOOK_AWAY, InputAzimuthRole.LEFT_LOS_TOWARD):
        return LookSide.LEFT
    return coerce_enum(LookSide, configured_look_side, "look_side")


def heading_vector_from_input_azimuth(azimuth_deg, input_azimuth_role):
    role = coerce_enum(InputAzimuthRole, input_azimuth_role, "input_azimuth_role")
    input_vec = horizontal_unit(azimuth_deg)
    if role == InputAzimuthRole.HEADING:
        return input_vec
    if role == InputAzimuthRole.RIGHT_LOOK_AWAY:
        return rotate_ccw90(input_vec)
    if role == InputAzimuthRole.LEFT_LOOK_AWAY:
        return rotate_cw90(input_vec)
    if role == InputAzimuthRole.RIGHT_LOS_TOWARD:
        return rotate_cw90(input_vec)
    if role == InputAzimuthRole.LEFT_LOS_TOWARD:
        return rotate_ccw90(input_vec)
    raise ValueError(f"Unsupported input_azimuth_role: {role}.")


def los_toward_horizontal_from_heading(heading_vec, look_side):
    look_side = coerce_enum(LookSide, look_side, "look_side")
    if look_side == LookSide.RIGHT:
        return -rotate_cw90(heading_vec)
    if look_side == LookSide.LEFT:
        return -rotate_ccw90(heading_vec)
    raise ValueError(f"Unsupported look_side: {look_side}.")


def build_projection_vector(azimuth_deg, incidence_deg, spec):
    """
    Build the ENU projection vector for the scalar observation in `spec`.
    """
    heading_vec = heading_vector_from_input_azimuth(
        azimuth_deg,
        spec.input_azimuth_role,
    )

    if spec.observation_type in (
        ObservationType.PHASE_LOS,
        ObservationType.LOS_DISPLACEMENT,
    ):
        los_horizontal = los_toward_horizontal_from_heading(heading_vec, spec.look_side)
        return projection_from_enu_horizontal_incidence(los_horizontal, incidence_deg)

    if spec.observation_type == ObservationType.AZIMUTH_OFFSET:
        return projection_from_enu_heading(heading_vec, direction="along")

    raise ValueError(f"Unsupported observation_type: {spec.observation_type}.")
