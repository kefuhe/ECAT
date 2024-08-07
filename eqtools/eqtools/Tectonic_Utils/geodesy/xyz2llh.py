"""
Functions to convert between local enu, local llh, and global xyz coordinates.
Translated from Matlab.
"""

import numpy as np
from . import datums


def xyz2llh(xyz, datum='NAD83'):
    """
    XYZ2LLH calculates longitude, latitude, and height from global cartesisan coordinates.
    LLH = xyz2llh(XYZ, DATUM) calculates longitude(deg), latitude(deg), and height(m) on the ellipsoid
    specified by DATUM from the global cartestian coordinates given in the nx3(n=number of coordinate triples)
    matrix XYZ.
    DATUM can either be a vector the first two elements of which give da and df,
    or it can be a string containing the name of a datum that is resolved
    by the function DATUMS function.
    Note that longitude occupies the first row of LLH.
    See DATUMS for more information on datum parameters.

    :param xyz: [x, y, z]
    :type xyz: numpy array
    :param datum: name of datum
    :type datum: string
    :returns: [lon, lat, height]
    :rtype: numpy array
    """

    # Check input arguments
    if not isinstance(datum, str):
        raise ValueError(f'Could not parse given datum: {datum}')
    datum_array = datums.get_datums(datum)
    if np.sum(np.isnan(datum_array)) > 0:
        raise ValueError('Could not resolve datum name.')
    da, df = datum_array[0][0], datum_array[0][1]

    if np.shape(xyz)[1] != 3:
        raise TypeError('Input xyz MUST be nx3.')

    # Set constants
    a = 6378137 - da
    f = 1 / 298.2572235630 - df
    b = (1-f) * a
    e2 = 2 * f - np.square(f)
    E2 = (np.square(a) - np.square(b)) / np.square(b)

    # Calculate longitude, latitude, and height
    llh = np.zeros(np.shape(xyz))
    p = np.sqrt(np.square(xyz[:, 0]) + np.square(xyz[:, 1]))
    llh[:, 0] = np.arctan2(xyz[:, 1], xyz[:, 0])
    theta = np.arctan(np.divide((xyz[:, 2] * a), (p * b)))
    llh[:, 1] = np.arctan((xyz[:, 2] + E2 * b * np.power(np.sin(theta), 3)) /
                          (p - e2 * a * np.power(np.cos(theta), 3)) )
    N = a / np.sqrt(1 - e2 * np.square(np.sin(llh[:, 1])))
    llh[:, 2] = p / np.cos(llh[:, 1]) - N

    # Convert to degrees
    llh[:, 0:2] = llh[:, 0:2]*57.295779513082323
    return llh


def llh2xyz(llh, datum='NAD83'):
    """
    LLH2XYZ  Calculates global cartesisan coordinates from longitude, latitude, and height.
       XYZ=llh2xyz(llh,DATUM) calculates global cartestian coordinates
       given the nx3 (n = number of coordinate triples) matrix LLH that contains
       longitude (deg), latitude (deg), and height (m) on the ellipsoid
       specified by DATUM. DATUM can either be a vector the first two elements
       of which give da and df, or it can be a string containing the name of a
       datum that is resolved by the function DATUMS function.
       Note that longitude occupies the first row of LLH.
       See DATUMS for more information on datum parameters.

    :param llh: [lon, lat, height]
    :type llh: numpy  array
    :param datum: name of datum
    :type datum: string
    """

    # Check input arguments
    if not isinstance(datum, str):
        raise ValueError(f'Could not parse given datum: {datum}')
    datum_array = datums.get_datums(datum)
    if np.sum(np.isnan(datum_array)) > 0:
        raise ValueError(f'Could not resolve datum name: {datum}')
    da, df = datum_array[0][0], datum_array[0][1]

    if np.shape(llh)[1] != 3:
        raise TypeError('Input llh MUST be nx3.')

    # Ellipsoid parameters
    a = 6378137 - da
    f = 1 / 298.257223563 - df
    b = (1-f) * a

    # Convert degrees to radians
    phi = llh[:, 1] * np.pi / 180   # lat
    lam = llh[:, 0] * np.pi / 180   # lon

    # Convert llh to xyz
    XYZ = np.zeros(np.shape(llh))
    N = np.square(a) / np.sqrt(np.square(a) * np.square(np.cos(phi)) + np.square(b) * np.square(np.sin(phi)))
    XYZ[:, 0] = (N + llh[:, 2]) * np.cos(phi) * np.cos(lam)
    XYZ[:, 1] = (N + llh[:, 2]) * np.cos(phi) * np.sin(lam)
    XYZ[:, 2] = (np.square(b) * N / np.square(a) + llh[:, 2] ) * np.sin(phi)
    return XYZ


def xyz2enum(origin):
    """
    XYZ2ENUM   Returns a global to local transformation matrix.
       T=xyz2enum(ORIGIN) Returns a transformation matrix that
       tranforms coordinates in a global ECEF cartesian system
       into to a local coordinate system aligned with the geographic
       directions at the position ORIGIN.  ORIGIN should contain a
       longitude and a latitude pair (degrees). T is 3x3.

    :param origin: [longitude, latitude]
    :type origin: np array
    """

    # Check input arguments
    if len(origin) < 2:
        raise ValueError('Input origin must have 2 elements, longitude and latitude (degrees).')

    # Convert to radians and evaluate trigonometric functions
    origin = np.multiply(origin, np.pi / 180)
    s = np.sin(origin)
    c = np.cos(origin)

    # Make transformation matrix
    T = np.array([[-s[0], c[0], 0],
                  [-s[1]*c[0], -s[1]*s[0], c[1]],
                  [c[1]*c[0], c[1]*s[0], s[1]]])
    return T


def xyz2enu(d, origin, dcov=None):
    """
    XYZ2ENU   Transforms from global cartestian to local cartesian.
       [E,ECOV]=xyz2enu(D,DCOV,ORIGIN) transforms data vector D and
       data covariance DCOV from a global cartesian (XYZ) coordinate
       system to a local coordinate system aligned with the geographic
       directions at the position ORIGIN.
       D should be either 3nx1 or 3xn (n = number of individual vectors).
       DCOV should be 3nx3n.
       ORIGIN should be a vector of length 2 or 3.  If length 2, ORIGIN
       is taken as a longitude, latitude pair (degrees); if length 3,
       ORIGIN is taken as an XYZ station position. E is matrix (vector)
       of transformed coordinates the same size as input D.  ECOV is a
       matrix containing the transformed covariance.
       E=xyz2enu(D,ORIGIN) behaves as above without a data covariance matrix.

    :param d: nx3 np array of x, y, z values
    :type d: numpy array
    :param origin: 1x3 np array (x, y, z) or 1x2 np.array (lon, lat)
    :type origin: numpy array
    :param dcov: 3x3 np array
    :type dcov: numpy array
    """

    # Check input arguments
    if len(origin) > 2:
        origin = np.reshape(origin, (1, 3))
        origin = xyz2llh(origin)
        origin = origin[0]  # 1x3 1D array, contains llh

    # Make transformation matrix
    Tm = xyz2enum(origin)

    # Transform
    e = np.dot(Tm, d.T)
    if dcov is not None:
        ecov = np.dot(np.dot(Tm, dcov), Tm.T)
    else:
        ecov = None
    return e.T, ecov


def enu2xyz(d, origin, dcov=None):
    """
    ENU2XYZ   Transforms from global cartestian to local cartesian.
       [E,ECOV]=xyz2enu(D,DCOV,ORIGIN) transforms data vector D and
       data covariance DCOV from a local cartesian (ENU) coordinate
       system aligned with the geographic directions at the position ORIGIN
       to a global (XYZ) coordinate system.
       D should be either 3nx1 or 3xn (n = number of individual vectors).
       DCOV should be 3nx3n.
       ORIGIN should be a vector of length 2 or 3.  If length 2, ORIGIN
       is taken as a longitude, latitude pair (degrees); if length 3,
       ORIGIN is taken as an XYZ station position. E is matrix (vector)
       of transformed coordinates the same size as input D.  ECOV is a
       matrix containing the transformed covariance.
       E=xyz2enu(D,ORIGIN) behaves as above without a data covariance matrix.

    :param d: nx3 np array of e, n, u values
    :type d: numpy array
    :param origin: 1x3 np array (x, y, z) or 1x2 np.array (lon, lat)
    :type origin: numpy array
    :param dcov: 3x3 np array
    :type dcov: numpy array
    """
    # Check input arguments
    if len(origin) > 2:
        origin = np.reshape(origin, (1, 3))
        origin = xyz2llh(origin)
        origin = origin[0]  # 1x3 1D array, contains llh

    # Make transformation matrix
    Tm = xyz2enum(origin)
    Tminv = np.linalg.inv(Tm)

    # Transform
    e = np.dot(Tminv, d.T)
    if dcov is not None:
        ecov = np.dot(np.dot(Tminv, dcov), Tminv.T)
    else:
        ecov = None
    return e.T, ecov
