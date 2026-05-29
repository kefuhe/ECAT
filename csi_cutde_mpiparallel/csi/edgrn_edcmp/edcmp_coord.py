"""
Coordinate convention conversion between CSI and EDCMP.

CSI convention:  x = East,  y = North, z = depth (positive down)
EDCMP convention: x = North, y = East,  z = Down (seismological)

All functions in this module are pure, stateless transformations.
"""

import numpy as np


def csi_obs_to_edcmp(data_x, data_y, mean_x, mean_y):
    """
    Convert CSI observation coordinates to EDCMP receiver coordinates.

    Parameters
    ----------
    data_x, data_y : array-like
        CSI coordinates (x=East, y=North) in kilometres.
    mean_x, mean_y : float
        Reference origin in CSI coordinates (kilometres).

    Returns
    -------
    xrec, yrec : np.ndarray
        EDCMP receiver coordinates (x=North, y=East) in metres.
    """
    xrec = np.asarray((np.asarray(data_y, dtype=np.float64) - mean_y) * 1000.0, dtype=np.float64)
    yrec = np.asarray((np.asarray(data_x, dtype=np.float64) - mean_x) * 1000.0, dtype=np.float64)
    return xrec, yrec


def csi_source_to_edcmp(xs_csi, ys_csi):
    """
    Swap CSI source coordinates (x=East, y=North) to EDCMP (x=North, y=East).

    Parameters
    ----------
    xs_csi, ys_csi : float or array-like
        Source position(s) in CSI convention (metres, already offset from origin).

    Returns
    -------
    xs_edcmp, ys_edcmp : same type as input
        Source position(s) in EDCMP convention.
    """
    return ys_csi, xs_csi


def edcmp_disp_to_csi(disp_edcmp):
    """
    Convert EDCMP displacement output to CSI convention.

    EDCMP returns (Ux=North, Uy=East, Uz=Down).
    CSI expects  (dx=East,  dy=North, dz=Up).

    Parameters
    ----------
    disp_edcmp : np.ndarray, shape (N, 3)
        Displacement in EDCMP convention.

    Returns
    -------
    disp_csi : np.ndarray, shape (N, 3)
        Displacement in CSI convention.
    """
    disp_csi = np.empty_like(disp_edcmp)
    disp_csi[:, 0] = disp_edcmp[:, 1]   # dx_csi (East)  = Uy_edcmp (East)
    disp_csi[:, 1] = disp_edcmp[:, 0]   # dy_csi (North) = Ux_edcmp (North)
    disp_csi[:, 2] = -disp_edcmp[:, 2]  # dz_csi (Up)    = -Uz_edcmp (Down)
    return disp_csi
