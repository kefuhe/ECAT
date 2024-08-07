import numpy as np


def wrap_lon(longitude):
    """
    Ensure longitude value is between -180° and 180° (e.g., not 240° E).

    :param longitude: float
    :returns: wrapped longitude, from -180° to 180°
    """
    if longitude > 360 or longitude < -360:
        raise ValueError("longitude outside normal range (-360 to 360)")
    if longitude > 180:
        longitude = longitude - 360
    if longitude < -180:
        longitude = longitude + 360
    return longitude


def get_vector_magnitude(vector):
    """
    Get magnitude of a vector.

    :param vector: n-component vector, any units
    :type vector: array_like
    :return: magnitude
    :rtype: float
    """
    total = 0
    for i in range(len(vector)):
        total = total + vector[i]*vector[i]
    magnitude = np.sqrt(total)
    return magnitude


def get_unit_vector(vec):
    """
    Get unit vector in the direction of a given vector.

    :param vec: 3-component vector, any units
    :type vec: array_like
    :return: unit vector
    :rtype: array_like
    """
    mag = np.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    vec = np.divide(vec, mag)
    return vec
