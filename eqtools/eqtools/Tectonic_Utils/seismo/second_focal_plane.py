"""
Utility tool for computing the second possible strike, rake, and dip in a focal mechanism,
given the strike, rake, and dip of the first focal plane.
Inspiration from Vince Cronin's website.
"""

# Internal information:
# Variable "a" is the reference strike azimuth.
# Variable "b" is the unit vector toward the reference strike azimuth in
# a coordinate system in which the +Y axis is north, the +X axis is east, and the +Z axis is up.
# Variable "c" is the unit dip vector.
# Variable "d" is the vector normal to the input nodal plane, and is the slip vector for the auxilary plane.
# Vector "d" points up if the input rake is a positive number, and down if the input rake is a negative number.
# Variable "f" is the basis vector for the X axis of the geographic coordinate system, which points east.
# Variable "g" is the basis vector for the Y axis of the geographic coordinate system, which points north.
# Variable "h" is the basis vector for the Z axis of the geographic coordinate system, which points up.
# Variable "i" is the basis vector for the X axis of the nodal plane coordinate system.
# If the rake is a negative number, the strike vector is the X axis and the dip vector is the Y axis.
# If the rake is a positive vector, the dip vector is the X axis and the strike vector is the Y axis.
# Variable "j" is the basis vec  tor for the Y axis of the nodal plane coordinate system.
# Variable "d" is the basis vector of the Z axis of the nodal plane coordinate system.
# Variable "k" is the unit vector in the nodal plane coordinate system that is coincident with the slip vector
# on the (input) nodal plane.
# Variable "m" is the transformation matrix from the nodal plane coordinate system to the geographic coordinate system.
# Variable "rakeVector" is the unit location vector along the slip direction on the nodal plane, expressed in the
# geographic coordinate system.
# Variable "n" is the dip vector of the auxilary plane, expressed in radian measure.
# Variable "s" is the unit vector projection of the rake vector onto the X - Y (horizontal) plane of the
# geographic coordinate system.
# Explanation of the result.
# The first element is the trend of the dip vector of the auxillary plane, in degrees.
# The second element is the dip angle of the auxillary plane, in degrees.
# The third element is the trend of the slip vector of the auxillary plane, in degrees.


import numpy as np


def vector_norm(v):
    """Computes norm of vector."""
    norm = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    return norm


def unit_vector(v):
    """Computes unit vector associated with 3-component vector."""
    norm = vector_norm(v)
    unit_vector = [v[0]/norm, v[1]/norm, v[2]/norm]
    return unit_vector


def vector_angle(v1, v2):
    """Returns angle between two vectors, in radians."""
    vectorAngle = np.arccos(np.dot(v1, v2)/(vector_norm(v1)*vector_norm(v2)))  # Returns in radians
    return vectorAngle


def find_aux_plane(strike, dip, rake):
    """
    Find the second auxiliary plane in a double couple focal mechanism from the given slip plane/vector.

    :param strike: strike of first focal plane, in degrees
    :type strike: float
    :param dip: dip of first focal plane, in degrees
    :type dip: float
    :param rake: rake of slip vector on first focal plane, in degrees
    :type rake: float
    :returns: [strike, dip, rake] of second focal plane
    :rtype: list of 3 floats
    """
    a = strike  # degrees
    dipaz = strike + 90  # degrees
    b = np.array([np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a)), 0])
    c = np.array([np.sin(np.deg2rad(dipaz))*np.cos(np.deg2rad(dip)),
                  np.cos(np.deg2rad(dipaz))*np.cos(np.deg2rad(dip)),
                  -np.sin(np.deg2rad(dip))])

    if rake < 0:
        d = np.cross(b, c)
    else:
        d = np.cross(c, b)

    f = np.array([1, 0, 0])
    g = np.array([0, 1, 0])
    h = np.array([0, 0, 1])

    if rake < 0:
        i = b
        j = c
        k = np.array([np.cos(np.deg2rad(rake)), -np.sin(np.deg2rad(rake)), 0])
    else:
        i = c
        j = b
        k = np.array([-np.sin(np.deg2rad(rake)), np.cos(np.deg2rad(rake)), 0])

    m = np.array([[np.dot(f, i), np.dot(f, j), np.dot(f, d)],
                  [np.dot(g, i), np.dot(g, j), np.dot(g, d)],
                  [np.dot(h, i), np.dot(h, j), np.dot(h, d)]])

    rakeVector = np.dot(m, k)
    n = np.arccos(abs(rakeVector[2]))  # auxDipAngle, radians

    s = unit_vector([rakeVector[0], rakeVector[1], 0])

    if rakeVector[0] < 0:
        rakeTrend = 360-vector_angle(s, g)*(180/np.pi)
    else:
        rakeTrend = vector_angle(s, g)*(180/np.pi)

    if rake < 0:
        if rakeTrend > 270:
            auxStrike = rakeTrend-270
        else:
            auxStrike = rakeTrend+90
    else:
        if rakeTrend < 90:
            auxStrike = rakeTrend+270
        else:
            auxStrike = rakeTrend-90
    auxStrikeRad = auxStrike*(np.pi/180)
    auxStrikeVector = [np.sin(auxStrikeRad), np.cos(auxStrikeRad), 0]

    if auxStrike > 270:
        auxDipAz = (auxStrike-270)
    else:
        auxDipAz = auxStrike+90
    auxDipAngle = n*180/np.pi

    if rake < 0:
        auxRake = -1*vector_angle(auxStrikeVector, d)*180/np.pi
    else:
        auxRake = vector_angle(auxStrikeVector, d)*180/np.pi

    auxStrikeAz = auxDipAz-90
    if auxStrikeAz < 0:
        auxStrikeAz = auxStrikeAz+360

    return [auxStrikeAz, auxDipAngle, auxRake]
