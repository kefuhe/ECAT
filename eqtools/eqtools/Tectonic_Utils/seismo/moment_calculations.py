
"""Utilities that convert between moment and magnitude, etc. """

import numpy as np


def moment_from_muad(mu, A, d):
    """moment = mu * A * d.

    :param mu: shear modulus, in Pa
    :type mu: float
    :param A: area, in m^2
    :type A: float
    :param d: slip, in m
    :type d: float
    :returns: moment, in Newton-meters
    :rtype: float
    """
    return mu*A*d

def mw_from_moment(moment):
    """Definition of moment magnitude from Hanks and Kanamori (1979). Takes newton-meters, returns moment magnitude.

    :param moment: moment in Newton-meters
    :type moment: float
    :returns: magnitude
    :rtype: float
    """
    moment = moment*1e7   # convert to dyne-cm
    mw = (2/3)*np.log10(moment) - 10.7
    return mw

def moment_from_mw(Mw):
    """Definition of moment magnitude from Hanks and Kanamori (1979). Takes magnitude, returns moment in newton-meters.

    :param Mw: moment magnitude
    :type Mw: float
    :returns: moment in Newton-meters
    :rtype: float
    """
    exponent = 1.5*Mw + 1.5*10.7
    moment = np.power(10, exponent)
    moment_newton_meters = moment * 1e-7
    return moment_newton_meters
