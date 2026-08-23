"""
Useful utilities for look vectors and coordinate systems.
"""

import numpy as np


def bearing_to_cartesian(heading):
    """
    Bearing (heading from North) to cartesian orientation CCW from east.

    :param heading: CW from North, in degrees
    :type heading: float
    :returns: Cartesian direction, CCW from East, in degrees
    :rtype: float
    """
    return 90 - heading


def cartesian_to_heading(cartesian_angle):
    """
    Cartesian orientation (CCW from east) to heading (CW from north).

    :param cartesian_angle: CCW from East, in degrees
    :type cartesian_angle: float
    :returns: heading direction, CW from North, in degrees
    :rtype: float
    """
    return 90 - cartesian_angle


def complement_angle(angle):
    """
    90 minus angle, in degrees.

    :param angle: degrees
    :returns: angle in degrees
    """
    return 90 - angle


def cartesian_to_ccw_from_north(angle):
    """
    angle minus 90, in degrees.

    :param angle: degrees
    :returns: angle in degrees
    """
    return angle - 90


def rotate_vector_by_angle(x0, y0, theta):
    """
    rotate a vector by an angle theta, in degrees CCW from East.

    :param x0: x component of vector
    :type x0: float
    :param y0: y component of vector
    :type y0: float
    :param theta: angle to rotate by, in degrees CCW from East like the mathematical definition
    :type theta: float
    :returns: xprime, yprime
    :rtype: float, float
    """
    x_prime = x0 * np.cos(np.deg2rad(-theta)) - y0 * np.sin(np.deg2rad(-theta))
    y_prime = x0 * np.sin(np.deg2rad(-theta)) + y0 * np.cos(np.deg2rad(-theta))
    return x_prime, y_prime


def normalize_vector(lkve, lkvn, lkvu):
    """Take a 3-component vector and normalize its components to a unit vector."""
    east_sq = np.square(lkve)
    north_sq = np.square(lkvn)
    up_sq = np.square(lkvu)
    sumarray = np.add(east_sq, north_sq)
    sumarray = np.add(sumarray, up_sq)
    magnitude = np.sqrt(sumarray)
    norm_lkve = np.divide(lkve, magnitude)
    norm_lkvn = np.divide(lkvn, magnitude)
    norm_lkvu = np.divide(lkvu, magnitude)
    return norm_lkve, norm_lkvn, norm_lkvu


def get_unit_vector_from_angle(angle):
    """
    Get the unit vector associated with a cartesian angle (CCW from east in degrees).

    :param angle: degrees CW from North
    :returns: list of two floats, x-component and y-component of unit vector
    """
    xcomponent, ycomponent = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    return [xcomponent, ycomponent]


def get_unit_vector_from_heading(heading):
    """
    Get the unit vector associated with a heading angle (CW from north in degrees).

    :param heading: degrees CW from North
    :returns: list of two floats, x-component and y-component of unit vector
    """
    cartesian_angle = bearing_to_cartesian(heading)
    [xcomponent, ycomponent] = get_unit_vector_from_angle(cartesian_angle)
    return [xcomponent, ycomponent]


def calc_rdr_azimuth_incidence_from_lkv_plane_down(lkve, lkvn, lkvu):
    """
    Function especially for UAVSAR interferograms to extract the incidence angle from lkv products.
    lkve, lkvn, lkvu describe vector from plane to ground.
    Convention: Azimuth angle measured from North in Anti-clockwise direction, in degrees, from ground to plane.
    Convention: Incidence angle measured from vertical at target.
    (aka degrees from vertical at satellite) (always +ve), in degrees.

    :param lkve: e component of look vector FROM PLANE TO GROUND
    :type lkve: float
    :param lkvn: n component of look vector FROM PLANE TO GROUND
    :type lkvn: float
    :param lkvu: u component of look vector FROM PLANE TO GROUND
    :type lkvu: float
    :returns: azimuth, incidence, angles in degrees
    :rtype: float, float
    """

    east_sq = np.square(lkve)
    north_sq = np.square(lkvn)
    sumarray = np.add(east_sq, north_sq)
    magnitude = np.sqrt(sumarray)
    azimuth_standard = np.arctan2(-lkvn, -lkve)   # azimuth is negative the direction of look vector from plane down
    azimuth_standard = np.rad2deg(azimuth_standard)
    azimuth = np.add(azimuth_standard, -90)

    incidence = np.arctan2(magnitude, -lkvu)
    incidence = np.rad2deg(incidence)
    return azimuth, incidence


def calc_lkv_from_rdr_azimuth_incidence(azimuth, incidence):
    """
    Function especially for ISCE interferograms to extract the lkv from azimuth/incidence.
    Convention: Azimuth angle measured from North in Anti-clockwise direction, in degrees, from ground to plane.
    Convention: Incidence angle measured from vertical at target.
    (aka degrees from vertical at satellite) (always +ve), in degrees.
    lkve, lkvn, lkvu describe vector from ground to plane.

    :param azimuth: CCW from north angle of vector from ground to plane, in degrees (ISCE LOS.RDR convention)
    :type azimuth: float
    :param incidence: angle from look vector to vertical, measured at the target (aka deg. from vertical at satellite)
    :type incidence: float
    :returns: lkve, lkvn, lkvu, unit vector from ground to plane
    """

    azimuth_standard = np.add(azimuth, 90)  # turning the CCW from N into CCW from East
    azimuth_rad = np.deg2rad(azimuth_standard)
    incidence_rad = np.deg2rad(incidence)
    lkv_u = np.cos(incidence_rad)
    lkv_n = np.sin(incidence_rad) * np.sin(azimuth_rad)
    lkv_e = np.sin(incidence_rad) * np.cos(azimuth_rad)
    return lkv_e, lkv_n, lkv_u


def look_vector2flight_incidence_angles(lkv_e, lkv_n, lkv_u):
    """
    General InSAR look vector math function, assuming right-looking satellite.
    lkv_e, lkv_n, lkv_u are the components of the look vector from ground to satellite.
    incidence angle is angle between look vector and vertical in degrees.
    Flight angle is clockwise from north in degrees.

    :param lkv_e: e component of look vector from ground to satellite
    :type lkv_e: float
    :param lkv_n: n component of look vector from ground to satellite
    :type lkv_n: float
    :param lkv_u: u component of look vector from ground to satellite
    :type lkv_u: float
    :returns: [flight_angle, incidence_angle] in degrees
    :rtype: list
    """
    unit_lkv = [lkv_e, lkv_n, lkv_u]
    unit_lkv = unit_lkv / np.linalg.norm(unit_lkv)
    vert_vector = [0, 0, 1]
    dotproduct = np.dot(unit_lkv, vert_vector)
    incidence_angle = np.rad2deg(np.arccos(dotproduct))

    lkv_horiz_angle = np.arctan2(lkv_n, lkv_e)  # cartesian angle of horiz look-vec. (negative small # for DESC)
    heading_deg = cartesian_to_heading(np.rad2deg(lkv_horiz_angle))
    flight_angle = heading_deg + 90  # satellite flies 90 degrees away from look vector direction
    return [flight_angle, incidence_angle]


def flight_incidence_angles2look_vector(flight_angle, incidence_angle):
    """
    General InSAR look vector math function, assuming right-looking satellite.
    lkv_e, lkv_n, lkv_u are the components of the look vector from ground to satellite.

    :param flight_angle: heading, clockwise from north, in degrees
    :type flight_angle: float
    :param incidence_angle: angle between look vector and vertical, in degrees
    :type incidence_angle: float
    :returns: [lkv_e, lkv_n, lkv_u]
    :rtype: list
    """
    lk_heading = flight_angle - 90  # heading, 90 degrees to the right of the satellite
    horizontal_lkv = np.sin(np.deg2rad(incidence_angle))
    lkv_u = np.cos(np.deg2rad(incidence_angle))
    lkv_cartesian_angle = bearing_to_cartesian(lk_heading)
    lkv_e = horizontal_lkv * np.cos(np.deg2rad(lkv_cartesian_angle))
    lkv_n = horizontal_lkv * np.sin(np.deg2rad(lkv_cartesian_angle))
    return [lkv_e, lkv_n, lkv_u]


def flight_incidence_angles2look_vector_leftlook(flight_angle, incidence_angle):
    """
    General InSAR look vector math function, assuming left-looking satellite.
    lkv_e, lkv_n, lkv_u are the components of the look vector from ground to satellite.

    :param flight_angle: heading, clockwise from north, in degrees
    :type flight_angle: float
    :param incidence_angle: angle between look vector and vertical, in degrees
    :type incidence_angle: float
    :returns: [lkv_e, lkv_n, lkv_u]
    :rtype: list
    """
    lk_heading = flight_angle + 90  # heading, 90 degrees to the left of the satellite
    horizontal_lkv = np.sin(np.deg2rad(incidence_angle))
    lkv_u = np.cos(np.deg2rad(incidence_angle))
    lkv_cartesian_angle = bearing_to_cartesian(lk_heading)
    lkv_e = horizontal_lkv * np.cos(np.deg2rad(lkv_cartesian_angle))
    lkv_n = horizontal_lkv * np.sin(np.deg2rad(lkv_cartesian_angle))
    return [lkv_e, lkv_n, lkv_u]


def def3D_into_LOS(U_e, U_n, U_u, flight_angle, incidence_angle, look_direction='right'):
    r"""
    Fialko, 2001, equation to project relative deformation into the LOS.

    .. math:: d_{los} = [U_n \sin(\phi) - U_e \cos(\phi)]*\sin(\lambda) + U_u \cos(\lambda).

    where
    lambda = incidence angle,
    phi = flight heading, and
    [U_e, U_n, U_u] are the east, north, and up components of the deformation.

    :param U_e: east component of deformation
    :type U_e: float
    :param U_n: north component of deformation
    :type U_n: float
    :param U_u: vertical component of deformation
    :type U_u: float
    :param flight_angle: azimuth of satellite heading vector in degrees, CW from north
    :type flight_angle: float
    :param incidence_angle: local incidence angle at the reflector (angle from the vertical), in degrees
    :type incidence_angle: float
    :param look_direction: 'left' or 'right' (default 'right')
    :type look_direction: string
    :returns: los deformation (in same units as U_e)
    :rtype: float
    """
    phi = np.deg2rad(flight_angle)
    lamda = np.deg2rad(incidence_angle)
    if look_direction == 'right':
        d_los = ((U_n * np.sin(phi) - U_e * np.cos(phi)) * np.sin(lamda) + U_u * np.cos(lamda))
    elif look_direction == 'left':
        d_los = ((U_n * np.sin(phi) + U_e * np.cos(phi)) * np.sin(lamda) + U_u * np.cos(lamda))
    else:
        raise ValueError("Error!  parameter look_direction must be 'right' or 'left', not %s" % look_direction)
    return d_los


def proj_los_into_vertical_no_horiz(los, lkv):
    """
    Project LOS deformation into a pseudo-vertical deformation,
    assuming horizontal deformation is zero.
    Compute the vertical deformation needed to produce given LOS deformation.

    :param los: float
    :param lkv: list of 3 floats, normalized look vector components E, N, U
    """
    lkv_horizontal = np.sqrt(lkv[0]*lkv[0] + lkv[1]*lkv[1])
    lkv_vertical = lkv[2]
    incidence_angle = np.arctan(lkv_horizontal/lkv_vertical)  # incidence angle from the vertical
    pseudo_vertical_disp = los / np.cos(incidence_angle)  # assuming no horizontal data contributes to LoS
    return pseudo_vertical_disp
