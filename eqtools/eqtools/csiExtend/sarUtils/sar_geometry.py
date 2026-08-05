"""Normalize angle rasters and construct canonical ENU projections."""

import numpy as np

from csi.projection import (
    projection_from_enu_heading,
    projection_from_enu_horizontal_incidence,
)

from .sar_conventions import (
    AcquisitionLookSide,
    AngleDirection,
    AngleGeometrySpec,
    AngleUnit,
    AzimuthAngleRole,
    AzimuthReference,
    IncidenceReference,
    ObservationType,
    coerce_enum,
)


def normalize_azimuth(azimuth, reference, unit, direction):
    """Convert an azimuth angle to ENU degrees (0=East, CCW positive).

    In particular, a north-referenced clockwise angle ``beta`` becomes
    ``alpha_enu = (90 - beta) % 360``.
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
    """Convert incidence to degrees measured from vertical."""

    reference = coerce_enum(IncidenceReference, reference, "incidence_reference")
    unit = coerce_enum(AngleUnit, unit, "incidence_unit")

    incidence = np.asarray(incidence)
    if unit == AngleUnit.RADIAN:
        incidence = np.rad2deg(incidence)
    if reference == IncidenceReference.ELEVATION:
        incidence = 90.0 - incidence
    return incidence


def horizontal_unit(azimuth_deg):
    """Return ``[cos(alpha), sin(alpha)]`` in the horizontal ENU plane."""

    alpha = np.deg2rad(np.asarray(azimuth_deg).reshape(-1))
    return np.column_stack((np.cos(alpha), np.sin(alpha)))


def rotate_cw90(vectors):
    """Rotate horizontal ENU vectors clockwise: ``(E, N) -> (N, -E)``."""

    vectors = np.asarray(vectors)
    return np.column_stack((vectors[:, 1], -vectors[:, 0]))


def rotate_ccw90(vectors):
    """Rotate horizontal ENU vectors counterclockwise: ``(E, N) -> (-N, E)``."""

    vectors = np.asarray(vectors)
    return np.column_stack((-vectors[:, 1], vectors[:, 0]))


def _sensor_to_ground_from_heading(heading, acquisition_look_side):
    """Map heading to the horizontal sensor-to-ground direction.

    For a right-looking acquisition ``q_sg = Rcw(heading)``; for a
    left-looking acquisition ``q_sg = Rccw(heading)``.
    """

    side = coerce_enum(
        AcquisitionLookSide, acquisition_look_side, "acquisition_look_side"
    )
    if side == AcquisitionLookSide.RIGHT:
        return rotate_cw90(heading)
    return rotate_ccw90(heading)


def _heading_from_sensor_to_ground(sensor_to_ground, acquisition_look_side):
    """Invert the side-dependent heading-to-ground rotation.

    For right look ``heading = Rccw(q_sg)``; for left look
    ``heading = Rcw(q_sg)``.
    """

    side = coerce_enum(
        AcquisitionLookSide, acquisition_look_side, "acquisition_look_side"
    )
    if side == AcquisitionLookSide.RIGHT:
        return rotate_ccw90(sensor_to_ground)
    return rotate_cw90(sensor_to_ground)


def heading_vector_from_input_azimuth(azimuth_deg, geometry_spec):
    """Return along-heading ENU vectors from a normalized azimuth raster.

    Acquisition side is used only when the input angle represents a
    cross-track sensor/ground axis.  It never determines scalar sign.
    """

    if not isinstance(geometry_spec, AngleGeometrySpec):
        raise TypeError("geometry_spec must be an AngleGeometrySpec.")
    direction = horizontal_unit(azimuth_deg)
    role = geometry_spec.azimuth_angle_role

    if role == AzimuthAngleRole.HEADING:
        return direction
    if role == AzimuthAngleRole.SENSOR_TO_GROUND:
        return _heading_from_sensor_to_ground(
            direction, geometry_spec.acquisition_look_side
        )
    if role == AzimuthAngleRole.GROUND_TO_SENSOR:
        return _heading_from_sensor_to_ground(
            -direction, geometry_spec.acquisition_look_side
        )
    raise ValueError(f"Unsupported azimuth_angle_role: {role}.")


def ground_to_sensor_vector_from_input_azimuth(azimuth_deg, geometry_spec):
    """Return horizontal ground-to-sensor vectors using the shortest mapping.

    A sensor-to-ground input is negated directly, so look side is not used.
    Look side is required only when a heading input must be rotated onto the
    cross-track LOS axis.
    """

    if not isinstance(geometry_spec, AngleGeometrySpec):
        raise TypeError("geometry_spec must be an AngleGeometrySpec.")
    direction = horizontal_unit(azimuth_deg)
    role = geometry_spec.azimuth_angle_role

    if role == AzimuthAngleRole.GROUND_TO_SENSOR:
        return direction
    if role == AzimuthAngleRole.SENSOR_TO_GROUND:
        return -direction
    if role == AzimuthAngleRole.HEADING:
        return -_sensor_to_ground_from_heading(
            direction, geometry_spec.acquisition_look_side
        )
    raise ValueError(f"Unsupported azimuth_angle_role: {role}.")


def acquisition_look_side_is_used(observation_spec, geometry_spec):
    """Whether angle-to-target-axis conversion requires the acquisition side."""

    role = geometry_spec.azimuth_angle_role
    if observation_spec.observation_type in (
        ObservationType.UNWRAPPED_PHASE,
        ObservationType.LOS_DISPLACEMENT,
    ):
        return role == AzimuthAngleRole.HEADING
    if observation_spec.observation_type == ObservationType.AZIMUTH_OFFSET:
        return role != AzimuthAngleRole.HEADING
    raise ValueError(
        f"Unsupported observation_type: {observation_spec.observation_type}."
    )


def build_projection_vector(azimuth_deg, incidence_deg, observation_spec, geometry_spec):
    """Build the canonical ENU projection for an angle-raster product.

    LOS/range projections point from ground to sensor. Azimuth projections
    point along platform heading. Acquisition side is consulted only when the
    input angle axis must be rotated to the target axis.

    With horizontal ground-to-sensor unit vector ``g`` and incidence ``theta``
    measured from vertical,

        p_los = [g_E*sin(theta), g_N*sin(theta), cos(theta)]

    With horizontal along-heading unit vector ``h``,

        p_azimuth = [h_E, h_N, 0]
    """

    if observation_spec.observation_type in (
        ObservationType.UNWRAPPED_PHASE,
        ObservationType.LOS_DISPLACEMENT,
    ):
        horizontal = ground_to_sensor_vector_from_input_azimuth(
            azimuth_deg, geometry_spec
        )
        return projection_from_enu_horizontal_incidence(horizontal, incidence_deg)

    if observation_spec.observation_type == ObservationType.AZIMUTH_OFFSET:
        heading = heading_vector_from_input_azimuth(azimuth_deg, geometry_spec)
        return projection_from_enu_heading(heading, direction="along")

    raise ValueError(
        f"Unsupported observation_type: {observation_spec.observation_type}."
    )
