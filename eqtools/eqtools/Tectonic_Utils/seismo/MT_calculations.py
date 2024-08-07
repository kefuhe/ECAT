"""
Calculations that deal with seismic moment tensors.
Notes from Lay and Wallace Chapter 8:

* Decomposition 1: Mij = isotropic + deviatoric
* Decomposition 2: Mij = isotropic + 3 vector dipoles
* Decomposition 3: Mij = isotropic + 3 double couples
* Decomposition 4: Mij = isotropic + 3 CLVDs
* Decomposition 5: Mij = isotropic + major DC + minor DC
* Decomposition 6: Mij = isotropic + DC + CLVD

The most useful in practice are Decomposition 1 and Decomposition 6.
"""

import numpy as np

def get_MT(mrr, mtt, mpp, mrt, mrp, mtp):
    """
    Build a matrix from the six components of the moment tensor

    :returns: np.array, 3x3 matrix
    """
    MT = np.array([[mrr, mrt, mrp], [mrt, mtt, mtp], [mrp, mtp, mpp]])
    return MT


def diagonalize_MT(MT):
    """
    Build a diagonal matrix whose elements are the ordered eigenvalues of original matrix MT.

    :returns: np.array, 3x3 matrix
    """
    eigvals, eigvecs = np.linalg.eig(MT)
    eigvals = sorted(eigvals)[::-1]
    return np.diag(eigvals)


def get_deviatoric_MT(MT):
    """
    Get deviatoric MT from a full MT.

    :returns: np.array, 3x3 matrix
    """
    iso_MT = get_iso_MT(MT)
    M_dev = np.subtract(MT, iso_MT)
    return M_dev


def get_iso_MT(MT):
    """
    Calculate the isotropic moment tensor from a full MT.

    :returns: np.array, 3x3 matrix
    """
    x = (1 / 3) * np.trace(MT)
    iso_MT = np.multiply(np.eye(3), x)
    return iso_MT


def get_clvd_dc_from_deviatoric_MT(MT):
    """
    Calculate the dc and clvd components of a deviatoric MT, from Shearer Equation 9.14.

    :returns: two np.arrays, each 3x3 matrix
    """
    eigenvalues = np.diag(MT)
    assert(eigenvalues[0] > eigenvalues[1] > eigenvalues[2]), ValueError("Deviatoric eigenvalues out of order.")
    dc_component = (1/2)*(eigenvalues[0]-eigenvalues[2])
    clvd_component = eigenvalues[1]*(1/2)
    M_dc = np.diag([dc_component, 0, -dc_component])
    M_clvd = np.diag([-clvd_component, 2*clvd_component, -clvd_component])
    return M_clvd, M_dc


def decompose_iso_dc_clvd(MT):
    """
    Decompose a full moment tensor into an isotropic part, a double-couple, and a CLVD component.

    :returns: three np.arrays, each 3x3 matrix
    """
    diag_MT = diagonalize_MT(MT)  # equivalent to a coordinate transformation
    M_iso = get_iso_MT(diag_MT)  # get the trace
    M_dev = get_deviatoric_MT(diag_MT)
    M_dev = diagonalize_MT(M_dev)  # diagonalized in the proper order
    M_clvd, M_dc = get_clvd_dc_from_deviatoric_MT(M_dev)
    return M_iso, M_clvd, M_dc


# def get_separate_scalar_moments(MT):
#     """return isotropic, clvd, and double couple moments. Not frequently used."""
#     M_iso, M_clvd, M_dc = decompose_iso_dc_clvd(MT);
#     iso_moment = abs(M_iso[0][0]);
#     clvd_moment = abs(M_clvd[0][0]);
#     dc_moment = abs(M_dc[0][0]);
#     return iso_moment, clvd_moment, dc_moment;


def get_total_scalar_moment(MT):
    """
    Shearer Equation 9.8: quadratic sum of element of moment tensor components, in newton-meters.

    :param MT: np.array, 3x3 matrix
    :returns: Mo, scalar moment
    :rtype: float
    """
    MT = np.divide(MT, 1e16)  # done to prevent computer buffer overflow
    total = 0
    for i in range(3):
        for j in range(3):
            total = total + MT[i][j]*MT[i][j]
    Mo = (1/np.sqrt(2)) * np.sqrt(total)
    Mo = np.multiply(Mo, 1e16)
    return Mo


def get_percent_double_couple(MT):
    """
    Get the percent double couple and percent clvd moment from a deviatoric moment tensor.
    When isotropic term is involved, this can get more complicated and there are several approaches.
    See Shearer equation 9.17 for epsilon.
    See Vavrycuk, 2001 for other approaches when isotropic component is involved.

    :param MT: np.array, 3x3 matrix
    :returns: percentage double couple, percentage CLVD
    :rtype: float, float
    """
    m_dev = diagonalize_MT(get_deviatoric_MT(MT))
    epsilon = np.diag(m_dev)[1] / np.max([np.abs(np.diag(m_dev)[0]), np.abs(np.diag(m_dev)[2])])
    fraction = epsilon * 2
    perc_clvd = 100 * (abs(fraction))
    perc_dc = 100 - perc_clvd
    return perc_dc, perc_clvd
