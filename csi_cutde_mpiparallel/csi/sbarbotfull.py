'''
ctypes wrapper for the compiled Fortran subroutine
computeDisplacementVerticalStrainVolume from Barbot (2017).

This module is the computational backend for the Sbarbot class,
analogous to mogifull.py for Mogi and yangfull.py for Yang.

When the compiled Fortran library is available it is used for maximum
performance.  If the library cannot be found the module automatically
falls back to a pure-NumPy implementation (slower but zero extra
dependencies).

To compile the Fortran library:
    cd csi/sbarbot_src && python build_sbarbot.py --install

Reference:
    Barbot S., J. D. P. Moore and V. Lambert, 2017.
    Displacement and Stress Associated with Distributed Anelastic
    Deformation in a Half Space,
    Bull. Seism. Soc. Am., 107(2), 10.1785/0120160237.

Written by kfhe, 2026.
'''

import ctypes
import numpy as np
import os
import platform
import logging

logger = logging.getLogger(__name__)

# Names of all 6 independent strain components
STRAIN_COMPONENTS = ('eps11', 'eps12', 'eps13', 'eps22', 'eps23', 'eps33')

# Module-level cached library handle
# None = not yet attempted, "PYTHON_FALLBACK" = no library found
_lib = None

# Whether the user has been warned about the fallback
_warned_fallback = False


def _find_library():
    '''
    Search for the compiled sbarbot shared library in known locations.

    Search order:
        1. csi/bin/<platform>/          (precompiled, shipped with pip install)
        2. csi/sbarbot_src/             (user compiled in source directory)
        3. csi/                         (library placed next to this file)

    Returns:
        str or None : path to the library if found
    '''
    base = os.path.dirname(os.path.abspath(__file__))

    system = platform.system()
    if system == 'Windows':
        names = ['sbarbot.dll']
    elif system == 'Darwin':
        names = ['libsbarbot.dylib', 'libsbarbot.so']
    else:
        names = ['libsbarbot.so']

    search_dirs = [
        # 1. Pre-built binaries shipped with the package
        os.path.join(base, 'bin', 'windows') if system == 'Windows' else
        os.path.join(base, 'bin', 'macos') if system == 'Darwin' else
        os.path.join(base, 'bin', 'ubuntu20.04'),
        # 2. User-compiled in sbarbot_src/
        os.path.join(base, 'sbarbot_src'),
        # 3. Next to this file
        base,
    ]

    for d in search_dirs:
        for name in names:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
    return None


def _load_library():
    '''
    Load the sbarbot shared library, or return the sentinel
    ``"PYTHON_FALLBACK"`` if the library cannot be found.
    '''
    global _lib
    if _lib is not None:
        return _lib

    lib_path = _find_library()
    if lib_path is None:
        _lib = "PYTHON_FALLBACK"
        return _lib

    logger.info(f"Loading sbarbot library from {lib_path}")
    try:
        _lib = ctypes.cdll.LoadLibrary(lib_path)
    except OSError as e:
        logger.warning(f"Failed to load sbarbot library ({e}), "
                       "falling back to pure-Python implementation.")
        _lib = "PYTHON_FALLBACK"
    return _lib


def _warn_fallback():
    """Issue a one-time warning when the Python fallback is used."""
    global _warned_fallback
    if not _warned_fallback:
        _warned_fallback = True
        logger.warning(
            "Sbarbot Fortran library not found – using pure-NumPy fallback "
            "(~100x slower).  To compile the Fortran library run:\n"
            "    cd csi/sbarbot_src && python build_sbarbot.py --install")


def displacement(x1, x2, x3, q1, q2, q3, L, T, W, theta,
                 eps11p=0., eps12p=0., eps13p=0.,
                 eps22p=0., eps23p=0., eps33p=0.,
                 G=30e9, nu=0.25):
    '''
    Compute surface displacement from a single vertical strain volume source
    at an array of observation points.

    Uses the compiled Fortran library when available, otherwise
    automatically falls back to a pure-NumPy implementation.

    Coordinate convention (Barbot 2017):
        x1 = northing, x2 = easting, x3 = depth (positive down).
        theta = strike angle, measured clockwise from north (radians).
        Output: u1 = north disp, u2 = east disp, u3 = down disp.

    Source geometry (see Fortran source and ASCII diagram)::

                          N (x1')
                         /
                        /| strike (theta)           E (x2')
            q1,q2,q3 ->@--------------------------+
                       |          T/2             |
                       |     <--+-->              | W
                       |        |                 | (depth
                       |        | (center line)   |  extent)
                       |     <--+-->              |
                       |          T/2             |
                       +--------------------------+
                       :       L (along strike)  /
                       :                        / T (thickness,
                       :                       /   into page)
                       Z (x3, depth)

    (q1, q2, q3) is the **reference point** of the volume:
      - q1 (north), q2 (east): horizontal position at the **start** of
        the length (L) direction and the **center** of the thickness (T)
        direction.  In the source-aligned (primed) frame, (q1,q2) maps
        to y1'=0, y2'=0.
      - q3: **top depth** of the volume (the shallowest face).
    From this reference point the volume extends:
      - L along strike       (y1': 0  -> L)
      - T centered perp to strike (y2': -T/2 -> +T/2)
      - W downward in depth  (y3': q3 -> q3+W)
    The geometric center of the volume is therefore at:
      (q1 + L/2*cos(theta), q2 + L/2*sin(theta), q3 + W/2).

    All length quantities must be in the same unit (e.g. km).
    G must be in Pa; nu is dimensionless.
    Displacement output is in the same length unit as the input.

    Args:
        x1, x2, x3 : array_like, observation coordinates
                      (northing, easting, depth; x3 >= 0).
        q1 : float, northing of the volume reference point (start of L,
             center of T).
        q2 : float, easting  of the volume reference point.
        q3 : float, top depth of the volume (>= 0).
        L  : float, length along strike.
        T  : float, thickness perpendicular to strike (centered).
        W  : float, depth extent (volume goes from q3 to q3+W).
        theta : float, strike angle in **radians**, clockwise from north.
        eps11p ... eps33p : float, anelastic strain components in the
                    source-aligned (primed) coordinate system.
        G  : float, shear modulus (Pa).
        nu : float, Poisson's ratio.

    Returns:
        u1, u2, u3 : ndarray, displacement (north, east, down).
    '''
    lib = _load_library()

    # ---- Python fallback path ----
    if lib == "PYTHON_FALLBACK":
        _warn_fallback()
        return displacement_python(x1, x2, x3, q1, q2, q3, L, T, W, theta,
                                   eps11p, eps12p, eps13p,
                                   eps22p, eps23p, eps33p, G, nu)

    # ---- Fortran fast path ----
    x1 = np.ascontiguousarray(x1, dtype=np.float64).ravel()
    x2 = np.ascontiguousarray(x2, dtype=np.float64).ravel()
    x3 = np.ascontiguousarray(x3, dtype=np.float64).ravel()
    N = x1.shape[0]
    assert x2.shape[0] == N and x3.shape[0] == N

    u1 = np.zeros(N, dtype=np.float64)
    u2 = np.zeros(N, dtype=np.float64)
    u3 = np.zeros(N, dtype=np.float64)

    c_int = ctypes.c_int
    c_double = ctypes.c_double
    c_double_p = ctypes.POINTER(c_double)

    n = c_int(N)
    scalars = [q1, q2, q3, L, T, W, theta,
               eps11p, eps12p, eps13p, eps22p, eps23p, eps33p,
               G, nu]
    c_scalars = [c_double(float(s)) for s in scalars]

    # Fortran symbol: lowercased name with trailing underscore (gfortran convention)
    func = lib.computedispverticalstrainvolumearray_

    func(ctypes.byref(n),
         x1.ctypes.data_as(c_double_p),
         x2.ctypes.data_as(c_double_p),
         x3.ctypes.data_as(c_double_p),
         *[ctypes.byref(s) for s in c_scalars],
         u1.ctypes.data_as(c_double_p),
         u2.ctypes.data_as(c_double_p),
         u3.ctypes.data_as(c_double_p))

    return u1, u2, u3


def displacement_python(x1, x2, x3, q1, q2, q3, L, T, W, theta,
                        eps11p=0., eps12p=0., eps13p=0.,
                        eps22p=0., eps23p=0., eps33p=0.,
                        G=30e9, nu=0.25):
    '''
    Pure-NumPy fallback for displacement computation.
    Supports both scalar and array inputs (no sympy dependency).
    '''
    from .sbarbot_src.sbarbot_python import displacement_python as _disp_py
    return _disp_py(x1, x2, x3, q1, q2, q3, L, T, W, theta,
                    eps11p, eps12p, eps13p, eps22p, eps23p, eps33p, G, nu)
