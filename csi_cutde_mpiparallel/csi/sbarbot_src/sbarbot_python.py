'''
Pure-NumPy implementation of surface displacement from a vertical strain
volume in an elastic half-space (Barbot et al., 2017).

This is a direct translation of the Matlab/Python reference code with
``sympy`` dependencies replaced by NumPy equivalents so that no extra
packages are needed.  It is used as an automatic fallback when the
compiled Fortran shared library is unavailable.

Performance note: this implementation is ~100x slower than the Fortran
version for large observation arrays.  For production use, please compile
the Fortran library:
    cd csi/sbarbot_src && python build_sbarbot.py

Reference:
    Barbot S., J. D. P. Moore and V. Lambert, 2017.
    Displacement and Stress Associated with Distributed Anelastic
    Deformation in a Half Space,
    Bull. Seism. Soc. Am., 107(2), 10.1785/0120160237.

Original Python translation by Qiu Qiang (qiuqiang2012@gmail.com), 2017.
NumPy-only adaptation by kfhe, 2026.
'''

import numpy as np


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

def _xlogy(x, y):
    """Compute x * log(y), with the convention 0 * log(0) = 0."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = x * np.log(y)
    out = np.where(x == 0, 0.0, out)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _acoth(x):
    """Inverse hyperbolic cotangent: acoth(x) = 0.5 * ln((x+1)/(x-1)).

    Replaces ``sympy.acoth`` for numeric (non-symbolic) evaluation.
    Returns 0 where the result is not finite (removable singularity at |x|=1).
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 0.5 * np.log((x + 1.0) / (x - 1.0))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_arctan_ratio(y, x):
    """Compute arctan(y/x) safely when x or y may be zero.

    Equivalent to the Fortran helper ``atan3`` which returns
    sign(pi/2, y) when x == 0, and 0 when both are zero.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    both_zero = (x == 0) & (y == 0)
    x_is_zero = (x == 0) & ~both_zero
    safe_x = np.where(x == 0, 1.0, x)
    result = np.arctan(y / safe_x)
    result = np.where(x_is_zero, np.copysign(np.pi / 2.0, y), result)
    result = np.where(both_zero, 0.0, result)
    return result


# -----------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------

def displacement_python(x1, x2, x3, q1, q2, q3, L, T, W, theta,
                        eps11p=0., eps12p=0., eps13p=0.,
                        eps22p=0., eps23p=0., eps33p=0.,
                        G=30e9, nu=0.25):
    """Compute surface displacement from a single vertical strain volume.

    All coordinates and lengths must share the same unit (e.g. km).
    ``theta`` is the strike angle **in radians** (measured clockwise from
    north).  The output follows the Barbot (2017) convention:
    u1 = north, u2 = east, u3 = down.

    Source geometry (see Fortran source for ASCII diagram):

    (q1, q2, q3) is the **reference point** of the volume:

    - q1 (north), q2 (east): horizontal position at the **start** of
      the length (L) direction and the **center** of the thickness (T)
      direction.  In the rotated (primed) frame this maps to
      y1' = 0, y2' = 0.
    - q3: **top depth** of the volume (shallowest face).

    From this reference point the volume extends:

    - L along strike           (y1': 0     -> L)
    - T centered perp to strike (y2': -T/2  -> +T/2)
    - W downward in depth       (y3': q3    -> q3 + W)

    The geometric center of the volume is therefore at
    ``(q1 + L/2*cos(theta), q2 + L/2*sin(theta), q3 + W/2)``.

    Parameters
    ----------
    x1, x2, x3 : array_like
        Observation coordinates (northing, easting, depth >= 0).
    q1 : float
        Northing of volume reference point (start of L, center of T).
    q2 : float
        Easting of volume reference point.
    q3 : float
        Top depth of volume (>= 0).
    L : float
        Length along strike.
    T : float
        Thickness perpendicular to strike (centered about ref. point).
    W : float
        Depth extent; volume goes from q3 to q3 + W.
    theta : float
        Strike angle in **radians**, clockwise from north.
    eps11p .. eps33p : float
        Anelastic strain components in the source-aligned (primed)
        coordinate system.
    G : float
        Shear modulus (Pa).
    nu : float
        Poisson's ratio.

    Returns
    -------
    u1, u2, u3 : ndarray
        Displacement (north, east, down).
    """

    x1 = np.atleast_1d(np.asarray(x1, dtype=np.float64))
    x2 = np.atleast_1d(np.asarray(x2, dtype=np.float64))
    x3 = np.atleast_1d(np.asarray(x3, dtype=np.float64))

    # Lame parameter
    Lambda = G * 2.0 * nu / (1.0 - 2.0 * nu)

    # Isotropic strain
    epsvkk = eps11p + eps22p + eps33p

    # Rotate observation points into source-aligned frame
    ct = np.cos(theta)
    st = np.sin(theta)
    t1  = (x1 - q1) * ct + (x2 - q2) * st
    x2r = -(x1 - q1) * st + (x2 - q2) * ct
    x1r = t1
    # Overwrite for use in the Green's functions
    x1 = x1r
    x2 = x2r

    # Shorthand distance functions
    def r1(y1, y2, y3):
        return np.sqrt((x1 - y1)**2 + (x2 - y2)**2 + (x3 - y3)**2)

    def r2(y1, y2, y3):
        return np.sqrt((x1 - y1)**2 + (x2 - y2)**2 + (x3 + y3)**2)

    # Use safe arctan ratio for all np.arctan(y * x**(-1)) patterns
    _sar = _safe_arctan_ratio

    pi = np.pi

    # ------------------------------------------------------------------
    # Green's function components  (J_ij_kl)
    # Notation follows Barbot (2017) �?kept identical to reference code.
    # ------------------------------------------------------------------

    def J1112(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2 * _r2**(-1) * x3 * (x1 - y1) * (x2 - y2) * y3 *
            ((x1 - y1)**2 + (x3 + y3)**2)**(-1)
            - 4 * ((-1) + nu) * ((-1) + 2*nu) * (x3 + y3) *
            _sar(x2 - y2, x1 - y1)
            - x3 * np.arctan2(x3, x1 - y1)
            - 3*x3 * np.arctan2(3*x3, x1 - y1)
            + 4*nu*x3 * np.arctan2(-nu*x3, x1 - y1)
            + 4*((-1) + nu)*((-1) + 2*nu)*(x3 + y3) *
            np.arctan2(_r2*(-x1 + y1), (x2 - y2)*(x3 + y3))
            - 4*((-1) + nu)*(x3 - y3) *
            np.arctan2(r1(y1, y2, y3)*(x3 - y3), (x1 - y1)*(x2 - y2))
            + 3*y3*np.arctan2(-3*y3, x1 - y1)
            - y3*np.arctan2(y3, x1 - y1)
            - 4*nu*y3*np.arctan2(nu*y3, x1 - y1)
            - 4*((-1) + nu)*(x3 + y3) *
            np.arctan2(_r2*(x3 + y3), (x1 - y1)*(x2 - y2))
            + _xlogy(-((-3) + 4*nu)*(x1 - y1), r1(y1, y2, y3) + x2 - y2)
            + _xlogy((5 + 4*nu*((-3) + 2*nu))*(x1 - y1), _r2 + x2 - y2)
            + _xlogy(-4*((-1) + nu)*(x2 - y2), r1(y1, y2, y3) + x1 - y1)
            + _xlogy(-4*((-1) + nu)*(x2 - y2), _r2 + x1 - y1))

    def J1113(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*(x1 - y1)*((x1 - y1)**2 + (x2 - y2)**2)**(-1) * (
                -((-1) + nu)*((-1) + 2*nu)*_r2**2*(x3 + y3)
                + ((-1) + nu)*((-1) + 2*nu)*_r2*y3*(2*x3 + y3)
                + x3*((x1 - y1)**2 + (x2 - y2)**2 + x3*(x3 + y3)))
            + x2*np.arctan2(-x2, x1 - y1)
            - 3*x2*np.arctan2(3*x2, x1 - y1)
            + 4*nu*x2*np.arctan2(-nu*x2, x1 - y1)
            - 4*((-1) + nu)*(x2 - y2) *
            np.arctan2(r1(y1, y2, y3)*(x2 - y2), (x1 - y1)*(x3 - y3))
            + 4*((-1) + nu)*(x2 - y2) *
            np.arctan2(_r2*(x2 - y2), (x1 - y1)*(x3 + y3))
            + 3*y2*np.arctan2(-3*y2, x1 - y1)
            - y2*np.arctan2(y2, x1 - y1)
            - 4*nu*y2*np.arctan2(nu*y2, x1 - y1)
            + _xlogy(-((-3) + 4*nu)*(x1 - y1), r1(y1, y2, y3) + x3 - y3)
            + _xlogy(-(3 - 6*nu + 4*nu**2)*(x1 - y1), _r2 + x3 + y3)
            + _xlogy(-4*((-1) + nu)*(x3 - y3), r1(y1, y2, y3) + x1 - y1)
            + _xlogy(4*((-1) + nu)*(x3 + y3), _r2 + x1 - y1))

    def J1123(y1, y2, y3):
        _r1 = r1(y1, y2, y3)
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            -2*_r2**(-1)*((x1 - y1)**2 + (x2 - y2)**2)**(-1) *
            (x2 - y2)*((x1 - y1)**2 + (x3 + y3)**2)**(-1) * (
                x3*((x3**2 + (x1 - y1)**2)*(x3**2 + (x1 - y1)**2 + (x2 - y2)**2)
                    + x3*(3*x3**2 + 2*(x1 - y1)**2 + (x2 - y2)**2)*y3
                    + 3*x3**2*y3**2 + x3*y3**3)
                - ((-1) + nu)*((-1) + 2*nu)*_r2**2*(x3 + y3) *
                ((x1 - y1)**2 + (x3 + y3)**2)
                + ((-1) + nu)*((-1) + 2*nu)*_r2*y3*(2*x3 + y3) *
                ((x1 - y1)**2 + (x3 + y3)**2))
            + 2*((-1) + nu)*((-1) + 2*nu)*(x1 - y1) *
            _sar(x1 - y1, x2 - y2)
            + x1*np.arctan2(-x1, x2 - y2)
            - 3*x1*np.arctan2(3*x1, x2 - y2)
            + 4*nu*x1*np.arctan2(-nu*x1, x2 - y2)
            + 3*y1*np.arctan2(-3*y1, x2 - y2)
            - y1*np.arctan2(y1, x2 - y2)
            - 4*nu*y1*np.arctan2(nu*y1, x2 - y2)
            + 2*((-1) + 2*nu)*(x1 - y1) *
            np.arctan2(_r1*(-x1 + y1), (x2 - y2)*(x3 - y3))
            + 2*(1 - 2*nu)**2*(x1 - y1) *
            np.arctan2(_r2*(-x1 + y1), (x2 - y2)*(x3 + y3))
            + _xlogy(-2*x3, _r2 - x2 + y2)
            + _xlogy(-((-3) + 4*nu)*(x2 - y2), _r1 + x3 - y3)
            + _xlogy(-(3 - 6*nu + 4*nu**2)*(x2 - y2), _r2 + x3 + y3)
            + _xlogy(-((-3) + 4*nu)*(x3 - y3), _r1 + x2 - y2)
            + _xlogy(-(5 + 4*nu*((-3) + 2*nu))*(x3 + y3), _r2 + x2 - y2))

    def J2112(y1, y2, y3):
        _r1 = r1(y1, y2, y3)
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            -_r1 + (1 + 8*((-1) + nu)*nu)*_r2
            - 2*_r2**(-1)*x3*y3
            + _xlogy(-4*((-1) + nu)*((-1) + 2*nu)*(x3 + y3), _r2 + x3 + y3))

    def J2113(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*((x1 - y1)**2 + (x2 - y2)**2)**(-1) *
            (x2 - y2) * (
                -((-1) + nu)*((-1) + 2*nu)*_r2**2*(x3 + y3)
                + ((-1) + nu)*((-1) + 2*nu)*_r2*y3*(2*x3 + y3)
                + x3*((x1 - y1)**2 + (x2 - y2)**2 + x3*(x3 + y3)))
            + _xlogy(-((-1) - 2*nu + 4*nu**2)*(x2 - y2), _r2 + x3 + y3)
            + _xlogy(-x2 + y2, r1(y1, y2, y3) + x3 - y3))

    def J2123(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*(x1 - y1)*((x1 - y1)**2 + (x2 - y2)**2)**(-1) * (
                -((-1) + nu)*((-1) + 2*nu)*_r2**2*(x3 + y3)
                + ((-1) + nu)*((-1) + 2*nu)*_r2*y3*(2*x3 + y3)
                + x3*((x1 - y1)**2 + (x2 - y2)**2 + x3*(x3 + y3)))
            + _xlogy(-((-1) - 2*nu + 4*nu**2)*(x1 - y1), _r2 + x3 + y3)
            + _xlogy(-x1 + y1, r1(y1, y2, y3) + x3 - y3))

    def J3112(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return -(1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            -2*_r2**(-1)*x3*(x2 - y2)*y3*(x3 + y3) *
            ((x1 - y1)**2 + (x3 + y3)**2)**(-1)
            + 4*((-1) + nu)*((-1) + 2*nu)*(x1 - y1) *
            _sar(x1 - y1, x2 - y2)
            + 4*((-1) + nu)*((-1) + 2*nu)*(x1 - y1) *
            np.arctan2(_r2*(-x1 + y1), (x2 - y2)*(x3 + y3))
            + _xlogy(-4*((-1) + nu)*((-1) + 2*nu)*(x2 - y2), _r2 + x3 + y3)
            + _xlogy(x3 - y3, r1(y1, y2, y3) + x2 - y2)
            + _xlogy(-x3 - 7*y3 - 8*nu**2*(x3 + y3) + 8*nu*(x3 + 2*y3),
                     _r2 + x2 - y2))

    def J3113(y1, y2, y3):
        _r1 = r1(y1, y2, y3)
        _r2 = r2(y1, y2, y3)
        return -(1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            _r1 + ((-1) - 8*((-1) + nu)*nu)*_r2
            - 2*_r2**(-1)*x3*y3
            + 2*((-3) + 4*nu)*x3*_acoth(_r2**(-1)*(x3 + y3))
            + _xlogy(2*(3*x3 + 2*y3 - 6*nu*(x3 + y3)
                        + 4*nu**2*(x3 + y3)), _r2 + x3 + y3))

    def J3123(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return -(1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*x3*(x1 - y1)*(x2 - y2)*y3 *
            ((x1 - y1)**2 + (x3 + y3)**2)**(-1)
            + 4*((-1) + nu)*((-1) + 2*nu)*(x3 + y3) *
            _sar(x2 - y2, x1 - y1)
            + 4*((-1) + 2*nu)*(nu*x3 + ((-1) + nu)*y3) *
            np.arctan2(_r2*(x1 - y1), (x2 - y2)*(x3 + y3))
            + _xlogy(x1 - y1, r1(y1, y2, y3) + x2 - y2)
            + _xlogy(-(1 + 8*((-1) + nu)*nu)*(x1 - y1), _r2 + x2 - y2))

    def J1212(y1, y2, y3):
        _r1 = r1(y1, y2, y3)
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            -_r1 + (1 + 8*((-1) + nu)*nu)*_r2
            - 2*_r2**(-1)*x3*y3
            + _xlogy(-4*((-1) + nu)*((-1) + 2*nu)*(x3 + y3), _r2 + x3 + y3))

    def J1213(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*((x1 - y1)**2 + (x2 - y2)**2)**(-1) *
            (x2 - y2) * (
                -((-1) + nu)*((-1) + 2*nu)*_r2**2*(x3 + y3)
                + ((-1) + nu)*((-1) + 2*nu)*_r2*y3*(2*x3 + y3)
                + x3*((x1 - y1)**2 + (x2 - y2)**2 + x3*(x3 + y3)))
            + _xlogy(-((-1) - 2*nu + 4*nu**2)*(x2 - y2), _r2 + x3 + y3)
            + _xlogy(-x2 + y2, r1(y1, y2, y3) + x3 - y3))

    def J1223(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*(x1 - y1)*((x1 - y1)**2 + (x2 - y2)**2)**(-1) * (
                -((-1) + nu)*((-1) + 2*nu)*_r2**2*(x3 + y3)
                + ((-1) + nu)*((-1) + 2*nu)*_r2*y3*(2*x3 + y3)
                + x3*((x1 - y1)**2 + (x2 - y2)**2 + x3*(x3 + y3)))
            + _xlogy(-((-1) - 2*nu + 4*nu**2)*(x1 - y1), _r2 + x3 + y3)
            + _xlogy(-x1 + y1, r1(y1, y2, y3) + x3 - y3))

    def J2212(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*x3*(x1 - y1)*(x2 - y2)*y3 *
            ((x2 - y2)**2 + (x3 + y3)**2)**(-1)
            - 4*((-1) + nu)*((-1) + 2*nu)*(x3 + y3) *
            _sar(x1 - y1, x2 - y2)
            - x3*np.arctan2(x3, x1 - y1)
            - 3*x3*np.arctan2(3*x3, x1 - y1)
            + 4*nu*x3*np.arctan2(-nu*x3, x1 - y1)
            + 4*((-1) + nu)*((-1) + 2*nu)*(x3 + y3) *
            np.arctan2(_r2*(-x2 + y2), (x1 - y1)*(x3 + y3))
            - 4*((-1) + nu)*(x3 - y3) *
            np.arctan2(r1(y1, y2, y3)*(x3 - y3), (x1 - y1)*(x2 - y2))
            + 3*y3*np.arctan2(-3*y3, x1 - y1)
            - y3*np.arctan2(y3, x1 - y1)
            - 4*nu*y3*np.arctan2(nu*y3, x1 - y1)
            - 4*((-1) + nu)*(x3 + y3) *
            np.arctan2(_r2*(x3 + y3), (x1 - y1)*(x2 - y2))
            + _xlogy(-4*((-1) + nu)*(x1 - y1), r1(y1, y2, y3) + x2 - y2)
            + _xlogy(-4*((-1) + nu)*(x1 - y1), _r2 + x2 - y2)
            + _xlogy(-((-3) + 4*nu)*(x2 - y2), r1(y1, y2, y3) + x1 - y1)
            + _xlogy((5 + 4*nu*((-3) + 2*nu))*(x2 - y2), _r2 + x1 - y1))

    def J2213(y1, y2, y3):
        _r1 = r1(y1, y2, y3)
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            -2*_r2**(-1)*(x1 - y1) *
            ((x1 - y1)**2 + (x2 - y2)**2)**(-1) *
            ((x2 - y2)**2 + (x3 + y3)**2)**(-1) * (
                x3*((x3**2 + (x2 - y2)**2)*(x3**2 + (x1 - y1)**2 + (x2 - y2)**2)
                    + x3*(3*x3**2 + (x1 - y1)**2 + 2*(x2 - y2)**2)*y3
                    + 3*x3**2*y3**2 + x3*y3**3)
                - ((-1) + nu)*((-1) + 2*nu)*_r2**2*(x3 + y3) *
                ((x2 - y2)**2 + (x3 + y3)**2)
                + ((-1) + nu)*((-1) + 2*nu)*_r2*y3*(2*x3 + y3) *
                ((x2 - y2)**2 + (x3 + y3)**2))
            + 2*((-1) + nu)*((-1) + 2*nu)*(x2 - y2) *
            _sar(x2 - y2, x1 - y1)
            + x2*np.arctan2(-x2, x1 - y1)
            - 3*x2*np.arctan2(3*x2, x1 - y1)
            + 4*nu*x2*np.arctan2(-nu*x2, x1 - y1)
            + 3*y2*np.arctan2(-3*y2, x1 - y1)
            - y2*np.arctan2(y2, x1 - y1)
            - 4*nu*y2*np.arctan2(nu*y2, x1 - y1)
            + 2*((-1) + 2*nu)*(x2 - y2) *
            np.arctan2(_r1*(-x2 + y2), (x1 - y1)*(x3 - y3))
            + 2*(1 - 2*nu)**2*(x2 - y2) *
            np.arctan2(_r2*(-x2 + y2), (x1 - y1)*(x3 + y3))
            + _xlogy(-2*x3, _r2 - x1 + y1)
            + _xlogy(-((-3) + 4*nu)*(x1 - y1), _r1 + x3 - y3)
            + _xlogy(-(3 - 6*nu + 4*nu**2)*(x1 - y1), _r2 + x3 + y3)
            + _xlogy(-((-3) + 4*nu)*(x3 - y3), _r1 + x1 - y1)
            + _xlogy(-(5 + 4*nu*((-3) + 2*nu))*(x3 + y3), _r2 + x1 - y1))

    def J2223(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*((x1 - y1)**2 + (x2 - y2)**2)**(-1) *
            (x2 - y2) * (
                -((-1) + nu)*((-1) + 2*nu)*_r2**2*(x3 + y3)
                + ((-1) + nu)*((-1) + 2*nu)*_r2*y3*(2*x3 + y3)
                + x3*((x1 - y1)**2 + (x2 - y2)**2 + x3*(x3 + y3)))
            + x1*np.arctan2(-x1, x2 - y2)
            - 3*x1*np.arctan2(3*x1, x2 - y2)
            + 4*nu*x1*np.arctan2(-nu*x1, x2 - y2)
            - 4*((-1) + nu)*(x1 - y1) *
            np.arctan2(r1(y1, y2, y3)*(x1 - y1), (x2 - y2)*(x3 - y3))
            + 4*((-1) + nu)*(x1 - y1) *
            np.arctan2(_r2*(x1 - y1), (x2 - y2)*(x3 + y3))
            + 3*y1*np.arctan2(-3*y1, x2 - y2)
            - y1*np.arctan2(y1, x2 - y2)
            - 4*nu*y1*np.arctan2(nu*y1, x2 - y2)
            + _xlogy(-((-3) + 4*nu)*(x2 - y2), r1(y1, y2, y3) + x3 - y3)
            + _xlogy(-(3 - 6*nu + 4*nu**2)*(x2 - y2), _r2 + x3 + y3)
            + _xlogy(-4*((-1) + nu)*(x3 - y3), r1(y1, y2, y3) + x2 - y2)
            + _xlogy(4*((-1) + nu)*(x3 + y3), _r2 + x2 - y2))

    def J3212(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return -(1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            -2*_r2**(-1)*x3*(x1 - y1)*y3*(x3 + y3) *
            ((x2 - y2)**2 + (x3 + y3)**2)**(-1)
            + 4*((-1) + nu)*((-1) + 2*nu)*(x2 - y2) *
            _sar(x2 - y2, x1 - y1)
            + 4*((-1) + nu)*((-1) + 2*nu)*(x2 - y2) *
            np.arctan2(_r2*(-x2 + y2), (x1 - y1)*(x3 + y3))
            + _xlogy(-4*((-1) + nu)*((-1) + 2*nu)*(x1 - y1), _r2 + x3 + y3)
            + _xlogy(x3 - y3, r1(y1, y2, y3) + x1 - y1)
            + _xlogy(-x3 - 7*y3 - 8*nu**2*(x3 + y3) + 8*nu*(x3 + 2*y3),
                     _r2 + x1 - y1))

    def J3213(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return -(1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*x3*(x1 - y1)*(x2 - y2)*y3 *
            ((x2 - y2)**2 + (x3 + y3)**2)**(-1)
            + 4*((-1) + nu)*((-1) + 2*nu)*(x3 + y3) *
            _sar(x1 - y1, x2 - y2)
            + 4*((-1) + 2*nu)*(nu*x3 + ((-1) + nu)*y3) *
            np.arctan2(_r2*(x2 - y2), (x1 - y1)*(x3 + y3))
            + _xlogy(x2 - y2, r1(y1, y2, y3) + x1 - y1)
            + _xlogy(-(1 + 8*((-1) + nu)*nu)*(x2 - y2), _r2 + x1 - y1))

    def J3223(y1, y2, y3):
        _r1 = r1(y1, y2, y3)
        _r2 = r2(y1, y2, y3)
        return -(1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            _r1 + ((-1) - 8*((-1) + nu)*nu)*_r2
            - 2*_r2**(-1)*x3*y3
            + 2*((-3) + 4*nu)*x3*_acoth(_r2**(-1)*(x3 + y3))
            + _xlogy(2*(3*x3 + 2*y3 - 6*nu*(x3 + y3)
                        + 4*nu**2*(x3 + y3)), _r2 + x3 + y3))

    def J1312(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return -(1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*x3*(x2 - y2)*y3*(x3 + y3) *
            ((x1 - y1)**2 + (x3 + y3)**2)**(-1)
            - 4*((-1) + nu)*((-1) + 2*nu)*(x1 - y1) *
            _sar(x1 - y1, x2 - y2)
            + 4*((-1) + nu)*((-1) + 2*nu)*(x1 - y1) *
            np.arctan2(_r2*(x1 - y1), (x2 - y2)*(x3 + y3))
            + _xlogy(4*((-1) + nu)*((-1) + 2*nu)*(x2 - y2), _r2 + x3 + y3)
            + _xlogy(x3 - y3, r1(y1, y2, y3) + x2 - y2)
            + _xlogy((7 + 8*((-2) + nu)*nu)*x3 + y3 + 8*((-1) + nu)*nu*y3,
                     _r2 + x2 - y2))

    def J1313(y1, y2, y3):
        _r1 = r1(y1, y2, y3)
        _r2 = r2(y1, y2, y3)
        return -(1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            _r1 + _r2**(-1)*((7 + 8*((-2) + nu)*nu)*_r2**2 + 2*x3*y3)
            + 2*((-3) + 4*nu)*x3*_acoth(_r2**(-1)*(x3 + y3))
            + _xlogy(2*((-3)*x3 - 2*y3 + 6*nu*(x3 + y3)
                        - 4*nu**2*(x3 + y3)), _r2 + x3 + y3))

    def J1323(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return -(1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            -2*_r2**(-1)*x3*(x1 - y1)*(x2 - y2)*y3 *
            ((x1 - y1)**2 + (x3 + y3)**2)**(-1)
            - 4*((-1) + nu)*((-1) + 2*nu)*(x3 + y3) *
            _sar(x2 - y2, x1 - y1)
            - 4*((-1) + nu)*((-3)*x3 - y3 + 2*nu*(x3 + y3)) *
            np.arctan2(_r2*(x1 - y1), (x2 - y2)*(x3 + y3))
            + _xlogy(x1 - y1, r1(y1, y2, y3) + x2 - y2)
            + _xlogy((7 + 8*((-2) + nu)*nu)*(x1 - y1), _r2 + x2 - y2))

    def J2312(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return -(1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*x3*(x1 - y1)*y3*(x3 + y3) *
            ((x2 - y2)**2 + (x3 + y3)**2)**(-1)
            - 4*((-1) + nu)*((-1) + 2*nu)*(x2 - y2) *
            _sar(x2 - y2, x1 - y1)
            + 4*((-1) + nu)*((-1) + 2*nu)*(x2 - y2) *
            np.arctan2(_r2*(x2 - y2), (x1 - y1)*(x3 + y3))
            + _xlogy(4*((-1) + nu)*((-1) + 2*nu)*(x1 - y1), _r2 + x3 + y3)
            + _xlogy(x3 - y3, r1(y1, y2, y3) + x1 - y1)
            + _xlogy((7 + 8*((-2) + nu)*nu)*x3 + y3 + 8*((-1) + nu)*nu*y3,
                     _r2 + x1 - y1))

    def J2313(y1, y2, y3):
        _r2 = r2(y1, y2, y3)
        return -(1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            -2*_r2**(-1)*x3*(x1 - y1)*(x2 - y2)*y3 *
            ((x2 - y2)**2 + (x3 + y3)**2)**(-1)
            - 4*((-1) + nu)*((-1) + 2*nu)*(x3 + y3) *
            _sar(x1 - y1, x2 - y2)
            - 4*((-1) + nu)*((-3)*x3 - y3 + 2*nu*(x3 + y3)) *
            np.arctan2(_r2*(x2 - y2), (x1 - y1)*(x3 + y3))
            + _xlogy(x2 - y2, r1(y1, y2, y3) + x1 - y1)
            + _xlogy((7 + 8*((-2) + nu)*nu)*(x2 - y2), _r2 + x1 - y1))

    def J2323(y1, y2, y3):
        _r1 = r1(y1, y2, y3)
        _r2 = r2(y1, y2, y3)
        return -(1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            _r1 + _r2**(-1)*((7 + 8*((-2) + nu)*nu)*_r2**2 + 2*x3*y3)
            + 2*((-3) + 4*nu)*x3*_acoth(_r2**(-1)*(x3 + y3))
            + _xlogy(2*((-3)*x3 - 2*y3 + 6*nu*(x3 + y3)
                        - 4*nu**2*(x3 + y3)), _r2 + x3 + y3))

    def J3312(y1, y2, y3):
        _r1 = r1(y1, y2, y3)
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*x3*(x1 - y1)*(x2 - y2)*y3 *
            ((x1 - y1)**2 + (x3 + y3)**2)**(-1) *
            ((x2 - y2)**2 + (x3 + y3)**2)**(-1) *
            ((x1 - y1)**2 + (x2 - y2)**2 + 2*(x3 + y3)**2)
            - 3*x3*np.arctan2(3*x3, x1 - y1)
            - 5*x3*np.arctan2(5*x3, x2 - y2)
            + 12*nu*x3*np.arctan2(-3*nu*x3, x2 - y2)
            + 4*nu*x3*np.arctan2(-nu*x3, x1 - y1)
            - 8*nu**2*x3*np.arctan2(nu**2*x3, x2 - y2)
            + 3*y3*np.arctan2(-3*y3, x1 - y1)
            - 5*y3*np.arctan2(5*y3, x2 - y2)
            + 12*nu*y3*np.arctan2(-3*nu*y3, x2 - y2)
            - 4*nu*y3*np.arctan2(nu*y3, x1 - y1)
            - 8*nu**2*y3*np.arctan2(nu**2*y3, x2 - y2)
            + 2*((-1) + 2*nu)*(x3 - y3) *
            np.arctan2(_r1*(-x3 + y3), (x1 - y1)*(x2 - y2))
            + 2*(1 - 2*nu)**2*(x3 + y3) *
            np.arctan2(_r2*(x3 + y3), (x1 - y1)*(x2 - y2))
            + _xlogy(-((-3) + 4*nu)*(x1 - y1), _r1 + x2 - y2)
            + _xlogy((5 + 4*nu*((-3) + 2*nu))*(x1 - y1), _r2 + x2 - y2)
            + _xlogy(-((-3) + 4*nu)*(x2 - y2), _r1 + x1 - y1)
            + _xlogy((5 + 4*nu*((-3) + 2*nu))*(x2 - y2), _r2 + x1 - y1))

    def J3313(y1, y2, y3):
        _r1 = r1(y1, y2, y3)
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*x3*(x1 - y1)*y3*(x3 + y3) *
            ((x2 - y2)**2 + (x3 + y3)**2)**(-1)
            + 5*x2*np.arctan2(-5*x2, x1 - y1)
            - 3*x2*np.arctan2(3*x2, x1 - y1)
            + 4*nu*x2*np.arctan2(-nu*x2, x1 - y1)
            - 12*nu*x2*np.arctan2(3*nu*x2, x1 - y1)
            + 8*nu**2*x2*np.arctan2(-nu**2*x2, x1 - y1)
            - 4*((-1) + nu)*(x2 - y2) *
            np.arctan2(_r1*(x2 - y2), (x1 - y1)*(x3 - y3))
            - 8*((-1) + nu)**2*(x2 - y2) *
            np.arctan2(_r2*(x2 - y2), (x1 - y1)*(x3 + y3))
            + 3*y2*np.arctan2(-3*y2, x1 - y1)
            - 5*y2*np.arctan2(5*y2, x1 - y1)
            + 12*nu*y2*np.arctan2(-3*nu*y2, x1 - y1)
            - 4*nu*y2*np.arctan2(nu*y2, x1 - y1)
            - 8*nu**2*y2*np.arctan2(nu**2*y2, x1 - y1)
            + _xlogy(-4*x3, _r2 - x1 + y1)
            + _xlogy(-4*((-1) + nu)*(x1 - y1), _r1 + x3 - y3)
            + _xlogy(-8*((-1) + nu)**2*(x1 - y1), _r2 + x3 + y3)
            + _xlogy(-((-3) + 4*nu)*(x3 - y3), _r1 + x1 - y1)
            + _xlogy(-7*x3 - 5*y3 + 12*nu*(x3 + y3)
                     - 8*nu**2*(x3 + y3), _r2 + x1 - y1))

    def J3323(y1, y2, y3):
        _r1 = r1(y1, y2, y3)
        _r2 = r2(y1, y2, y3)
        return (1.0/16) * (1 - nu)**(-1) * pi**(-1) * G**(-1) * (
            2*_r2**(-1)*x3*(x2 - y2)*y3*(x3 + y3) *
            ((x1 - y1)**2 + (x3 + y3)**2)**(-1)
            + 5*x1*np.arctan2(-5*x1, x2 - y2)
            - 3*x1*np.arctan2(3*x1, x2 - y2)
            + 4*nu*x1*np.arctan2(-nu*x1, x2 - y2)
            - 12*nu*x1*np.arctan2(3*nu*x1, x2 - y2)
            + 8*nu**2*x1*np.arctan2(-nu**2*x1, x2 - y2)
            - 4*((-1) + nu)*(x1 - y1) *
            np.arctan2(_r1*(x1 - y1), (x2 - y2)*(x3 - y3))
            - 8*((-1) + nu)**2*(x1 - y1) *
            np.arctan2(_r2*(x1 - y1), (x2 - y2)*(x3 + y3))
            + 3*y1*np.arctan2(-3*y1, x2 - y2)
            - 5*y1*np.arctan2(5*y1, x2 - y2)
            + 12*nu*y1*np.arctan2(-3*nu*y1, x2 - y2)
            - 4*nu*y1*np.arctan2(nu*y1, x2 - y2)
            - 8*nu**2*y1*np.arctan2(nu**2*y1, x2 - y2)
            + _xlogy(-4*x3, _r2 - x2 + y2)
            + _xlogy(-4*((-1) + nu)*(x2 - y2), _r1 + x3 - y3)
            + _xlogy(-8*((-1) + nu)**2*(x2 - y2), _r2 + x3 + y3)
            + _xlogy(-((-3) + 4*nu)*(x3 - y3), _r1 + x2 - y2)
            + _xlogy(-7*x3 - 5*y3 + 12*nu*(x3 + y3)
                     - 8*nu**2*(x3 + y3), _r2 + x2 - y2))

    # ------------------------------------------------------------------
    # Integral combinations
    # ------------------------------------------------------------------

    def IU1(y1, y2, y3):
        return ((Lambda*epsvkk + 2*G*eps11p) * J1123(y1, y2, y3)
                + 2*G*eps12p * (J1223(y1, y2, y3) + J1113(y1, y2, y3))
                + 2*G*eps13p * (J1323(y1, y2, y3) + J1112(y1, y2, y3))
                + (Lambda*epsvkk + 2*G*eps22p) * J1213(y1, y2, y3)
                + 2*G*eps23p * (J1212(y1, y2, y3) + J1313(y1, y2, y3))
                + (Lambda*epsvkk + 2*G*eps33p) * J1312(y1, y2, y3))

    def IU2(y1, y2, y3):
        return ((Lambda*epsvkk + 2*G*eps11p) * J2123(y1, y2, y3)
                + 2*G*eps12p * (J2223(y1, y2, y3) + J2113(y1, y2, y3))
                + 2*G*eps13p * (J2323(y1, y2, y3) + J2112(y1, y2, y3))
                + (Lambda*epsvkk + 2*G*eps22p) * J2213(y1, y2, y3)
                + 2*G*eps23p * (J2212(y1, y2, y3) + J2313(y1, y2, y3))
                + (Lambda*epsvkk + 2*G*eps33p) * J2312(y1, y2, y3))

    def IU3(y1, y2, y3):
        return ((Lambda*epsvkk + 2*G*eps11p) * J3123(y1, y2, y3)
                + 2*G*eps12p * (J3223(y1, y2, y3) + J3113(y1, y2, y3))
                + 2*G*eps13p * (J3323(y1, y2, y3) + J3112(y1, y2, y3))
                + (Lambda*epsvkk + 2*G*eps22p) * J3213(y1, y2, y3)
                + 2*G*eps23p * (J3212(y1, y2, y3) + J3313(y1, y2, y3))
                + (Lambda*epsvkk + 2*G*eps33p) * J3312(y1, y2, y3))

    # ------------------------------------------------------------------
    # Volume integration (eight corners)
    # Suppress intermediate divide-by-zero / invalid warnings that arise
    # at degenerate geometry corners; they cancel in the summation.
    # ------------------------------------------------------------------
    with np.errstate(divide='ignore', invalid='ignore'):
        u1 = (IU1(L, T/2, q3+W) - IU1(L, -T/2, q3+W)
              + IU1(L, -T/2, q3) - IU1(L, T/2, q3)
              - IU1(0, T/2, q3+W) + IU1(0, -T/2, q3+W)
              - IU1(0, -T/2, q3) + IU1(0, T/2, q3))

        u2 = (IU2(L, T/2, q3+W) - IU2(L, -T/2, q3+W)
              + IU2(L, -T/2, q3) - IU2(L, T/2, q3)
              - IU2(0, T/2, q3+W) + IU2(0, -T/2, q3+W)
              - IU2(0, -T/2, q3) + IU2(0, T/2, q3))

        u3 = (IU3(L, T/2, q3+W) - IU3(L, -T/2, q3+W)
              + IU3(L, -T/2, q3) - IU3(L, T/2, q3)
              - IU3(0, T/2, q3+W) + IU3(0, -T/2, q3+W)
              - IU3(0, -T/2, q3) + IU3(0, T/2, q3))

    # Clean any residual NaN from removable singularities
    u1 = np.nan_to_num(u1, nan=0.0)
    u2 = np.nan_to_num(u2, nan=0.0)
    u3 = np.nan_to_num(u3, nan=0.0)

    # Rotate back to geographic frame
    t1 = u1*ct - u2*st
    u2  = u1*st + u2*ct
    u1  = t1

    return u1, u2, u3
