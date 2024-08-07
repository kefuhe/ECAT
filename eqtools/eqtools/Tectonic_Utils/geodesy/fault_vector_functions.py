"""
Useful utilities for defining fault planes and coordinate systems.
"""

import numpy as np
import math
from collections.abc import Iterable
from Tectonic_Utils.geodesy import haversine, utilities


def xy2lonlat_single(xi, yi, reflon, reflat):
    """
    Convert cartesian x/y coordinate into lon/lat coordinate using reference point.

    :param xi: x coordinate of target point, in km
    :type xi: float
    :param yi: y coordinate of target point, in km
    :type yi: float
    :param reflon: longitude of reference point
    :type reflon: float
    :param reflat: latitude of reference point
    :type reflat: float
    :returns: lon, lat of target point
    :rtype: float, float
    """
    lat = reflat + (yi * 1 / 111.000)
    lon = reflon + (xi * 1 / (111.000 * abs(np.cos(np.deg2rad(reflat)))))
    return lon, lat


def latlon2xy_single(loni, lati, lon0, lat0):
    """
    Convert lon/lat coordinate into cartesian x/y coordinate using reference point.

    :param loni: longitude of target point
    :type loni: float
    :param lati: latitude of target point
    :type lati: float
    :param lon0: longitude of reference point
    :type lon0: float
    :param lat0: latitude of reference point
    :type lat0: float
    :returns: x, y of target point, in km
    :rtype: float, float
    """
    radius = haversine.distance([lat0, lon0], [lati, loni])
    bearing = haversine.calculate_initial_compass_bearing((lat0, lon0), (lati, loni))
    azimuth = 90 - bearing
    x = radius * np.cos(np.deg2rad(azimuth))
    y = radius * np.sin(np.deg2rad(azimuth))
    return x, y


def xy2lonlat(xi, yi, reflon, reflat):
    """
    Convert cartesian x/y coordinates into lon/lat coordinates using reference point.

    :param xi: x coordinate of target point(s), in km
    :type xi: float or list
    :param yi: y coordinate of target point(s), in km
    :type yi: float or list
    :param reflon: longitude of reference point
    :type reflon: float
    :param reflat: latitude of reference point
    :type reflat: float
    :returns: lon, lat of target point(s)
    :rtype: list, list (float, float in case of single inputs)
    """

    if not isinstance(xi, Iterable):
        if not isinstance(yi, Iterable):  # if single value, return single value
            return xy2lonlat_single(xi, yi, reflon, reflat)
        else:
            raise ValueError(f"Error! Dimension of x and y does not agree: {xi} {yi}" )
    if len(list(xi)) != len(list(yi)):
        raise ValueError(f'Error! Length of x and y does not agree: {len(list(xi))} {len(list(yi))}')

    # if we are getting a list of values, we return a list of the same dimensions
    lat, lon = [], []
    for a, b in zip(xi, yi):
        loni, lati = xy2lonlat_single(a, b, reflon, reflat)
        lon.append(loni)
        lat.append(lati)
    return lon, lat


def latlon2xy(loni, lati, lon0, lat0):
    """
    Convert lon/lat coordinates into cartesian x/y coordinates using reference point.

    :param loni: longitude of target point(s)
    :type loni: float or list
    :param lati: latitude of target point(s)
    :type lati: float or list
    :param lon0: longitude of reference point
    :type lon0: float
    :param lat0: latitude of reference point
    :type lat0: float
    :returns: x, y of target point(s), in km
    :rtype: list, list  (float, float in case of single inputs)
    """
    if not isinstance(loni, Iterable):
        if not isinstance(lati, Iterable):  # if single value, return single value
            return latlon2xy_single(loni, lati, lon0, lat0)
        else:
            raise ValueError(f"Error! Dimension of x and y does not agree: {loni} {lati}" )
    if len(list(loni)) != len(list(lati)):
        raise ValueError(f'Error! Length of x and y does not agree: {len(list(loni))} {len(list(lati))}')

    # If we are getting a list, return a list of the same dimensions
    x, y = [], []
    for a, b in zip(loni, lati):
        xi, yi = latlon2xy_single(a, b, lon0, lat0)
        x.append(xi)
        y.append(yi)
    return x, y


def get_plane_normal(strike, dip):
    """
    Get the outward-facing unit normal to a plane of specified strike and dip (dip-cross-strike).

    :param strike: strike, degrees
    :type strike: float
    :param dip: dip, degrees
    :type dip: float
    :return: 3-component vector of outward facing unit normal vector, [x, y, z]
    :rtype: np.ndarray
    """
    # Inside, we first find the orthogonal unit vectors
    # aligned with strike and dip directions that sit within the plane. The plane normal is their cross product,
    # i.e. the outward facing unit normal vector, dip-cross-strike, in x-y-z coordinates.
    strike_vector = get_strike_vector(strike)  # unit vector
    dip_vector = get_dip_vector(strike, dip)  # unit vector
    plane_normal = simple_cross_product(dip_vector, strike_vector)  # dip x strike
    # for outward facing normal, by right-hand rule.
    return plane_normal


def simple_cross_product(a, b):
    """
    Implement the simple cross product for 2 vectors in 3-D space.

    :param a: array-like, 3-component vector
    :param b: array-like, 3-component vector
    :returns: 3-component array, [x, y, z]
    :rtype: [float, float, float]
    """
    s1 = a[1]*b[2] - a[2]*b[1]
    s2 = a[2]*b[0] - a[0]*b[2]
    s3 = a[0]*b[1] - a[1]*b[0]
    return [s1, s2, s3]


def get_dip_degrees(x0, y0, z0, x1, y1, z1):
    """
    Get the dip of the line that connects two points.

    :param x0: x coordinate of shallower point
    :type x0: float
    :param y0: y coordinate of shallower point
    :type y0: float
    :param z0: z coordinate of shallower point
    :type z0: float
    :param x1: x coordinate of deeper point
    :type x1: float
    :param y1: y coordinate of deeper point
    :type y1: float
    :param z1: z coordinate of deeper point
    :type z1: float
    :returns: dip, in degrees
    :rtype: float
    """
    horizontal_length = get_strike_length(x0, x1, y0, y1)
    vertical_distance = abs(z1 - z0)
    dip = np.rad2deg(math.atan2(vertical_distance, horizontal_length))
    return dip


def get_strike_vector(strike):
    """
    Get a unit vector along the strike direction of a plane.

    :param strike: strike, in degrees CW from N
    :type strike: float
    :returns: 3-component unit vector, in x, y, z coordinates
    :rtype: [float, float, float]
    """
    theta = np.deg2rad(90 - strike)
    strike_vector = [np.cos(theta), np.sin(theta), 0]
    return strike_vector


def get_dip_vector(strike, dip):
    """
    Get a unit vector along the dip direction of a plane.

    :param strike: strike, in degrees CW from N
    :type strike: float
    :param dip: dip, in degrees
    :type dip: float
    :returns: 3-component unit vector, in x, y, z coordinates
    :rtype: [float, float, float]
    """
    downdip_direction_theta = np.deg2rad(-strike)  # theta(strike+90)
    dip_unit_vector_z = np.sin(np.deg2rad(dip))  # the vertical component of the downdip unit vector
    dip_unit_vector_xy = np.sqrt(1-dip_unit_vector_z*dip_unit_vector_z)  # horizontal component of downdip unit vector
    dip_vector = [dip_unit_vector_xy * np.cos(downdip_direction_theta),
                  dip_unit_vector_xy * np.sin(downdip_direction_theta), -dip_unit_vector_z]
    return dip_vector


def get_rtlat_dip_slip(slip, rake):
    """
    Decompose slip into right lateral and reverse dip slip components.

    :param slip: slip, in any length unit
    :type slip: float
    :param rake: rake, in degrees
    :type rake: float
    :returns: rt-lat strike slip and reverse dip slip, in the same length units as `slip`
    :rtype: float, float
    """
    rt_strike_slip = -slip * np.cos(np.deg2rad(rake))  # negative sign for convention of right lateral slip
    dip_slip = slip * np.sin(np.deg2rad(rake))
    return rt_strike_slip, dip_slip


def get_leftlat_reverse_slip(slip, rake):
    """
    Decompose slip into left lateral and reverse slip components.

    :param slip: slip, in any length unit
    :type slip: float
    :param rake: rake, in degrees
    :type rake: float
    :returns: left-lat strike slip and reverse dip slip, in the same length units as `slip`
    :rtype: float, float
    """
    ll_strike_slip = slip * np.cos(np.deg2rad(rake))  # convention of left lateral slip
    dip_slip = slip * np.sin(np.deg2rad(rake))
    return ll_strike_slip, dip_slip


def get_strike(deltax, deltay):
    """
    Compute the strike of a vector x,y.

    :param deltax: displacement in x direction, in any length unit
    :type deltax: float
    :param deltay: displacement in y direction, in any length unit
    :type deltay: float
    :returns: strike of vector, in CW degrees from north
    :rtype: float
    """
    slope = math.atan2(deltay, deltax)
    strike = 90 - np.rad2deg(slope)
    if strike < 0:
        strike = strike + 360
    return strike


def get_downdip_width(top, bottom, dip):
    """
    Get total downdip-width of a rectangular fault plane given top depth, bottom depth, and dip.

    :param top: depth of top of fault plane, in km (positive down)
    :type top: float
    :param bottom: depth of top of fault plane, in km (positive down)
    :type bottom: float
    :param dip: dip of fault plane, in degrees (range 0 to 90)
    :type dip: float
    :returns: total down-dip width of the rectangle, in km
    :rtype: float
    """
    W = abs(top - bottom) / np.sin(np.deg2rad(dip))  # guaranteed to be between 0 and 90
    return W


def get_total_slip(strike_slip, dip_slip):
    """Just the pythagorean theorem."""
    return np.sqrt(strike_slip * strike_slip + dip_slip * dip_slip)


def get_strike_length(x0, x1, y0, y1):
    """Just the pythagorean theorem."""
    length = np.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
    return length


def get_top_bottom_from_center(center_depth, width, dip):
    """
    Get the top and bottom depth of a rectangular fault from its width, center, and dip.

    :param center_depth: depth of center of fault plane, in km (positive down)
    :type center_depth: float
    :param width: total downdip width of rectangular fault plane, in km
    :type width: float
    :param dip: dip of fault plane, in degrees (range 0 to 90)
    :type dip: float
    :returns: top and bottom of fault plane, in km
    :rtype: float, float
    """
    top = center_depth - (width / 2.0 * np.sin(np.deg2rad(dip)))
    bottom = center_depth + (width / 2.0 * np.sin(np.deg2rad(dip)))
    return top, bottom


def get_top_bottom_from_top(top_depth, width, dip):
    """
    Get the top and bottom depth of a rectangular fault from its width, top-edge depth, and dip.

    :param top_depth: depth of top edge of fault plane, in km (positive down)
    :type top_depth: float
    :param width: total downdip width of rectangular fault plane, in km
    :type width: float
    :param dip: dip of fault plane, in degrees (range 0 to 90)
    :type dip: float
    :returns: top and bottom of fault plane, in km
    :rtype: float, float
    """
    bottom = top_depth + (width * np.sin(np.deg2rad(dip)))
    return top_depth, bottom


def get_vector_magnitude(vector):
    """
    Get magnitude of a vector.

    :param vector: n-component vector, any units
    :type vector: array_like
    :return: magnitude
    :rtype: float
    """
    return utilities.get_vector_magnitude(vector)


def get_unit_vector(vec):
    """
    Get unit vector.

    :param vec: 3-component vector, any units
    :type vec: array_like
    :return: unit vector
    :rtype: array_like
    """
    return utilities.get_unit_vector(vec)


def add_vector_to_point(x0, y0, vector_mag, vector_heading):
    """
    :param x0: starting x-coordinate for vector
    :type x0: float
    :param y0: starting y-coordinate for vector
    :type y0: float
    :param vector_mag: magnitude of vector to be added to point
    :type vector_mag: float
    :param vector_heading: direction of vector, in degrees CW from north
    :type vector_heading: float
    :returns: x1, y1 coordinates of ending point
    :rtype: float, float
    """
    theta = np.deg2rad(90 - vector_heading)
    x1 = x0 + vector_mag * np.cos(theta)
    y1 = y0 + vector_mag * np.sin(theta)
    return x1, y1


def get_rake(rtlat_strike_slip, dip_slip):
    """
    Return the rake of a given slip vector.
    Positive strike-slip is right lateral, and positive dip-slip is reverse.
    Will return 0 if dipslip,strikeslip == 0,0.

    :param rtlat_strike_slip: quantity of right lateral slip, any length units
    :type rtlat_strike_slip: float
    :param dip_slip: quantity of reverse slip, any length units
    :type dip_slip: float
    :return: rake in range -180 to 180 degrees
    :rtype: float
    """
    rake = np.rad2deg(math.atan2(dip_slip, -rtlat_strike_slip))    # Aki and Richards definition: positive ll
    return rake
