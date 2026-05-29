import numpy as np


def rotate_cw90(vectors):
    vectors = np.asarray(vectors)
    return np.column_stack((vectors[:, 1], -vectors[:, 0]))


def rotate_ccw90(vectors):
    vectors = np.asarray(vectors)
    return np.column_stack((-vectors[:, 1], vectors[:, 0]))


def unit_from_heading_north_cw(heading):
    """
    Convert CSI legacy heading angles to ENU horizontal unit vectors.

    Heading convention: 0=North and clockwise positive.
    """
    alpha = np.deg2rad(np.asarray(heading).reshape(-1))
    return np.column_stack((np.sin(alpha), np.cos(alpha)))


def unit_from_azimuth_east_ccw(azimuth):
    """
    Convert ENU azimuth angles to horizontal unit vectors.

    Azimuth convention: 0=East and counterclockwise positive.
    """
    alpha = np.deg2rad(np.asarray(azimuth).reshape(-1))
    return np.column_stack((np.cos(alpha), np.sin(alpha)))


def projection_from_heading_incidence(heading, incidence, look_side="right",
                                      output="toward_satellite"):
    """
    Build an ENU projection vector from satellite heading and incidence.

    Heading follows the legacy CSI convention: 0=North and clockwise positive.
    The returned projection points toward the satellite by default.
    """
    heading, incidence = np.broadcast_arrays(np.asarray(heading), np.asarray(incidence))
    heading_vec = unit_from_heading_north_cw(heading)
    look_side = look_side.lower()
    if look_side == "right":
        horizontal = -rotate_cw90(heading_vec)
    elif look_side == "left":
        horizontal = -rotate_ccw90(heading_vec)
    else:
        raise ValueError("look_side must be 'right' or 'left'.")

    output = output.lower()
    if output in ("away", "away_from_satellite"):
        horizontal *= -1.0
    elif output not in ("toward", "toward_satellite"):
        raise ValueError("output must be 'toward_satellite' or 'away_from_satellite'.")

    return projection_from_enu_horizontal_incidence(horizontal, incidence)


def projection_from_look_incidence(look_azimuth, incidence, look_sense="away"):
    """
    Build an ENU projection vector from a horizontal look/LOS azimuth.

    look_azimuth uses 0=East and counterclockwise positive. If look_sense is
    'away', the input is satellite-to-ground and the returned projection is
    flipped to ground-to-satellite.
    """
    look_azimuth, incidence = np.broadcast_arrays(np.asarray(look_azimuth), np.asarray(incidence))
    horizontal = unit_from_azimuth_east_ccw(look_azimuth)
    look_sense = look_sense.lower()
    if look_sense in ("away", "away_from_satellite"):
        horizontal *= -1.0
    elif look_sense not in ("toward", "toward_satellite"):
        raise ValueError("look_sense must be 'away' or 'toward'.")

    return projection_from_enu_horizontal_incidence(horizontal, incidence)


def projection_from_enu_horizontal_incidence(horizontal, incidence):
    """
    Build an ENU projection vector from a horizontal ENU direction and an
    incidence angle measured from vertical.
    """
    horizontal = np.asarray(horizontal)
    if horizontal.ndim == 1:
        horizontal = horizontal.reshape((1, 2))
    if horizontal.shape[1] != 2:
        raise ValueError("horizontal must have two ENU columns: east, north.")

    incidence = np.asarray(incidence).reshape(-1)
    if incidence.size == 1 and horizontal.shape[0] != 1:
        incidence = np.ones((horizontal.shape[0],)) * incidence[0]
    if incidence.shape[0] != horizontal.shape[0]:
        raise ValueError("incidence and horizontal direction sizes do not match.")

    norm = np.linalg.norm(horizontal, axis=1)
    if np.any(norm == 0.0):
        raise ValueError("horizontal direction contains a zero-length vector.")
    horizontal = horizontal / norm[:, np.newaxis]

    phi = np.deg2rad(incidence)
    return np.column_stack((
        horizontal[:, 0] * np.sin(phi),
        horizontal[:, 1] * np.sin(phi),
        np.cos(phi),
    ))


def projection_from_enu_heading(heading_vector, direction="along"):
    """
    Build a horizontal along-track ENU projection vector.
    """
    projection = np.asarray(heading_vector)
    if projection.ndim == 1:
        projection = projection.reshape((1, 2))
    if projection.shape[1] != 2:
        raise ValueError("heading_vector must have two ENU columns: east, north.")

    norm = np.linalg.norm(projection, axis=1)
    if np.any(norm == 0.0):
        raise ValueError("heading_vector contains a zero-length vector.")
    projection = projection / norm[:, np.newaxis]
    projection = np.column_stack((projection[:, 0], projection[:, 1], np.zeros(projection.shape[0])))

    direction = direction.lower()
    if direction in ("opposite", "opposite_heading"):
        projection *= -1.0
    elif direction not in ("along", "along_heading"):
        raise ValueError("direction must be 'along' or 'opposite'.")
    return projection
