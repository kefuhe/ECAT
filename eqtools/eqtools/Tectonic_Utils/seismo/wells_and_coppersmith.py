"""
Set of functions that calculate important scaling relationships between earthquake magnitude and other parameters.
From Wells and Coppersmith, 1994.

* FAULT_TYPE = SS, R, N, All
* a, b = parameters in equation
* sa, sb = parameter uncertainties
* s = estimate uncertainty
* r = R correlation coefficient

Average displacements, Max displacements, and inverse relationships not included.
"""

import numpy as np
import sys
from . import moment_calculations


def check_fault_types(fault_type):
    """Ensure that provided fault type is within ["SS", "N", "R", "ALL"]"""
    if fault_type == "SS" or fault_type == "N" or fault_type == "R" or fault_type == "ALL":
        return
    else:
        sys.exit("Error! Fault Type should be SS, N, R, or ALL!  Exiting...")


def rectangular_slip(length, width, magnitude, mu=30*1e9):
    """Magnitude to slip on rectangular patch

    :param length: fault length, in m
    :type length: float
    :param width: fault width, in m
    :type width: float
    :param magnitude: magnitude
    :type magnitude: float
    :param mu: shear modulus, in Pa, defaults to 30 GPa
    :type mu: float, optional
    :returns: slip, in m
    :rtype: float
    """
    area = length * width  # meters^2
    moment_nm = moment_calculations.moment_from_mw(magnitude)
    slip = moment_nm / (area * mu)  # Moment = Area * slip * mu.
    return slip


def get_magnitude_from_rectangle(length, width, slip, mu=30*1e9):
    """
    :param length: fault length, in m
    :type length: float
    :param width: fault width, in m
    :type width: float
    :param slip: fault slip, in m
    :type slip: float
    :param mu: shear modulus, in Pa, defaults to 30 GPa
    :type mu: float, optional
    :returns: magnitude
    :rtype: float
    """
    area = length * width  # meters^2
    moment_nm = moment_calculations.moment_from_muad(mu, area, slip)  # newton-meters
    magnitude = moment_calculations.mw_from_moment(moment_nm)
    return magnitude


def SLR_from_M(M, fault_type):
    """
    Surface rupture width from Magnitude, computed using log(SLR)=a + b*M

    :param M: magnitude
    :type M: float
    :param fault_type: fault_type ["SS", "N", "R", "ALL"]
    :type fault_type: string
    :returns: surface rupture length, in km
    :rtype: float
    """
    # row has the form [a, sa, b, sb, s, r, mmin, mmax]
    check_fault_types(fault_type)
    if fault_type == "SS":
        [a, _, b, _, _, _, mmin, mmax] = [-3.55, 0.37, 0.74, 0.05, 0.23, 0.91, 5.6, 8.1]
    elif fault_type == "R":
        [a, _, b, _, _, _, mmin, mmax] = [-2.86, 0.55, 0.63, 0.08, 0.20, 0.88, 5.4, 7.4]
    elif fault_type == "N":
        [a, _, b, _, _, _, mmin, mmax] = [-2.01, 0.65, 0.50, 0.10, 0.21, 0.81, 5.2, 7.3]
    else:
        [a, _, b, _, _, _, mmin, mmax] = [-3.22, 0.27, 0.69, 0.04, 0.22, 0.89, 5.2, 8.1]
    assert(mmin <= M <= mmax), ValueError("M"+str(M)+" outside valid bounds for Wells and Coppersmith.")
    SLR = np.power(10, (a + b * M))
    return SLR


def RW_from_M(M, fault_type):
    """
    Downdip rupture width from Magnitude, computed using log(RW)=a + b*M

    :param M: magnitude
    :type M: float
    :param fault_type: fault_type ["SS", "N", "R", "ALL"]
    :type fault_type: string
    :returns: downdip rupture width, in km
    :rtype: float
    """
    # row has the form [a, sa, b, sb, s, r, mmin, mmax]
    check_fault_types(fault_type)
    if fault_type == "SS":
        [a, _, b, _, _, _, mmin, mmax] = [-0.76, 0.12, 0.27, 0.02, 0.14, 0.84, 4.8, 8.1]
    elif fault_type == "R":
        [a, _, b, _, _, _, mmin, mmax] = [-1.61, 0.20, 0.41, 0.03, 0.15, 0.90, 4.8, 7.6]
    elif fault_type == "N":
        [a, _, b, _, _, _, mmin, mmax] = [-1.14, 0.28, 0.35, 0.05, 0.12, 0.86, 5.2, 7.3]
    else:
        [a, _, b, _, _, _, mmin, mmax] = [-1.01, 0.10, 0.32, 0.02, 0.15, 0.84, 4.8, 8.1]
    assert(mmin <= M <= mmax), ValueError("M"+str(M)+" outside valid bounds for Wells and Coppersmith.")
    RW = np.power(10, (a + b * M))
    return RW


def RLD_from_M(M, fault_type):
    """
    Subsurface rupture length from Magnitude, computed using log(RLD)=a + b*M

    :param M: magnitude
    :type M: float
    :param fault_type: fault_type ["SS", "N", "R", "ALL"]
    :type fault_type: string
    :returns: subsurface rupture length, in km
    :rtype: float
    """
    # row has the form [a, sa, b, sb, s, r, mmin, mmax]
    check_fault_types(fault_type)
    if fault_type == "SS":
        [a, _, b, _, _, _, mmin, mmax] = [-2.57, 0.12, 0.62, 0.02, 0.15, 0.96, 4.8, 8.1]
    elif fault_type == "R":
        [a, _, b, _, _, _, mmin, mmax] = [-2.42, 0.21, 0.58, 0.03, 0.16, 0.93, 4.8, 7.6]
    elif fault_type == "N":
        [a, _, b, _, _, _, mmin, mmax] = [-1.88, 0.37, 0.50, 0.06, 0.17, 0.88, 5.2, 7.3]
    else:
        [a, _, b, _, _, _, mmin, mmax] = [-2.44, 0.11, 0.59, 0.02, 0.16, 0.94, 4.8, 8.1]
    assert(mmin <= M <= mmax), ValueError("M"+str(M)+" outside valid bounds for Wells and Coppersmith.")
    RLD = np.power(10, (a + b * M))
    return RLD


def RA_from_M(M, fault_type):
    """
    Rupture Area from Magnitude, computed using log(RA)=a + b*M

    :param M: magnitude
    :type M: float
    :param fault_type: fault_type ["SS", "N", "R", "ALL"]
    :type fault_type: string
    :returns: rupture area, in km^2
    :rtype: float
    """
    # row has the form [a, sa, b, sb, s, r, mmin, mmax]
    check_fault_types(fault_type)
    if fault_type == "SS":
        [a, _, b, _, _, _, mmin, mmax] = [-3.42, 0.18, 0.90, 0.03, 0.22, 0.96, 4.8, 7.9]
    elif fault_type == "R":
        [a, _, b, _, _, _, mmin, mmax] = [-3.99, 0.36, 0.98, 0.06, 0.26, 0.94, 4.8, 7.6]
    elif fault_type == "N":
        [a, _, b, _, _, _, mmin, mmax] = [-2.87, 0.50, 0.82, 0.08, 0.22, 0.92, 5.2, 7.3]
    else:  # ALL
        [a, _, b, _, _, _, mmin, mmax] = [-3.49, 0.16, 0.91, 0.03, 0.24, 0.95, 4.8, 7.9]
    assert(mmin <= M <= mmax), ValueError("M"+str(M)+" outside valid bounds for Wells and Coppersmith.")
    RA = np.power(10, (a + b * M))
    return RA
