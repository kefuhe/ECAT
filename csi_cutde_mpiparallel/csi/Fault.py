'''
A parent Fault class

Written by Z. Duputel, R. Jolivet, and B. Riel, March 2014
Edited by T. Shreve, May 2019
'''

# Import Externals stuff
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from . import triangularDisp as tdisp
from scipy.linalg import block_diag
import scipy.spatial.distance as scidis
import copy
import sys
import os
import logging
from types import SimpleNamespace
from dataclasses import dataclass

logger = logging.getLogger(__name__)
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz

# Personals
from .SourceInv import SourceInv
from .EDKSmp import sum_layered
from .EDKSmp import dropSourcesInPatches as Patches2Sources
from .psgrn_pscmp.PSGRNCmp import pscmpslip2dis
from .psgrn_pscmp.pscmp_options import PscmpOptions
from .edgrn_edcmp.EDGRNcmp import edcmpslip2dis, edcmpslip2dis_forward
from .edgrn_edcmp.edcmp_backends import (
    EdcmpOptions,
    compute_inmemory_edcmp_forward,
    compute_inmemory_edcmp_greens,
    resolve_edcmp_engine,
)
from .edgrn_edcmp.tri2rectpoints import patch_local2d_inv, patch_local2d, triangle_to_rectangles
from tqdm import tqdm

from . import VALID_GF_METHODS
from .gf_options import resolve_gf_options


@dataclass
class EdcmpPatchSourceCache:
    """Structured cache for EDCMP patch source parameters."""
    sources: list
    mean_x_km: float
    mean_y_km: float
    cache_key: tuple


def _lightweight_data(data):
    """
    Create a lightweight proxy of the data object for parallel dispatch.
    Only carries the fields needed by EDCMP workers (x, y, name),
    avoiding pickle of the full object (which includes the large Cd matrix).
    """
    return SimpleNamespace(
        x=np.asarray(data.x, dtype=np.float64),
        y=np.asarray(data.y, dtype=np.float64),
        name=getattr(data, 'name', ''),
    )


def _pscmp_patch_task(args):
    """
    Module-level function for parallel PSCMP patch computation.
    Automatically use fast version if available.
    """
    (self, p, SLP, data, workdir, psgrn_dir, out_dir, verbose, Np, 
     p_vertices, p_vertices_ll, patchType, faultname, 
     force_recompute, sourceparameters) = args
    dataname = data.name
    patch_prefix = f'p{p}'
    cx_km, cy_km, depth_km, width_km, length_km, strike_rad, dip_rad = sourceparameters
    clon = np.mean(p_vertices_ll[:, 0])
    clat = np.mean(p_vertices_ll[:, 1])
    strike_deg, dip_deg = np.rad2deg(strike_rad), np.rad2deg(dip_rad)
    if patchType == 'rectangle':
        ss, ds, ts = pscmpslip2dis(
            data, (clon, clat, depth_km, width_km, length_km, strike_deg, dip_deg),
            slip=SLP,
            grn_dir=psgrn_dir,
            output_dir=out_dir,
            filename_suffix=patch_prefix,
            workdir=workdir,
            force_recompute=force_recompute,
            faultname=faultname,
            dataname=dataname
        )
    elif patchType == 'triangle':
        vertices = p_vertices  # 3x3 array
        xyz = patch_local2d(vertices, cx_km, cy_km, depth_km, strike_rad, dip_rad)

        dx_km = 0.1  # 100m
        dy_km = 0.1  # 100m
        _, rect_corners = triangle_to_rectangles(xyz[:, :2], dx_km, dy_km)

        # Recover to original 3D coordinates
        rect_corners_3d = [
            patch_local2d_inv(
                np.column_stack([c, np.zeros((c.shape[0], 1))]),
                cx_km, cy_km, depth_km, strike_rad, dip_rad
            )
            for c in rect_corners
        ]
        xs_center_km = np.array([np.mean(c[:, 0]) for c in rect_corners_3d])
        ys_center_km = np.array([np.mean(c[:, 1]) for c in rect_corners_3d])
        depth_center_km = np.array([np.mean(c[:, 2]) for c in rect_corners_3d])
        xs_clon, xs_clat = self.xy2ll(xs_center_km, ys_center_km)
        ss, ds, ts = pscmpslip2dis(
            data, (xs_clon, xs_clat, depth_center_km, dy_km, dx_km, strike_deg, dip_deg),
            slip=SLP,
            grn_dir=psgrn_dir,
            output_dir=out_dir,
            filename_suffix=patch_prefix,
            workdir=workdir,
            force_recompute=force_recompute,
            faultname=faultname,
            dataname=dataname
        )

    return p, ss, ds, ts

# Prepare source parameters for edcmpslip2dis for each patch ---
def _prepare_patch_source(
    p,
    patchType,
    geometry,
    patch_vertices,
    mean_x_km,
    mean_y_km,
    rect_dx_km=0.1,
    rect_dy_km=0.1,
):
    """
    Prepare the source parameter tuple for edcmpslip2dis for a single patch.
    Returns (p, sourceparams) for order tracking.
    """
    import numpy as np
    cx_km, cy_km, depth_km, width_km, length_km, strike_rad, dip_rad = geometry
    strike_deg, dip_deg = np.rad2deg(strike_rad), np.rad2deg(dip_rad)
    if patchType == 'rectangle':
        half_width_horz_km = width_km * np.cos(dip_rad) / 2.0
        half_length_km = length_km / 2.0
        dxy = (-half_length_km + 1.j * half_width_horz_km) * np.exp(1.j * (np.pi/2.0 - strike_rad))
        x_top_left_km = cx_km + np.real(dxy)
        y_top_left_km = cy_km + np.imag(dxy)
        depth_top_left_km = depth_km - np.sin(dip_rad) * width_km / 2.0
        xs = (x_top_left_km - mean_x_km) * 1000.0
        ys = (y_top_left_km - mean_y_km) * 1000.0
        sourceparams = (xs, ys, depth_top_left_km*1000.0, width_km*1000.0, length_km*1000.0, strike_deg, dip_deg, mean_x_km, mean_y_km)
    elif patchType == 'triangle':
        vertices = patch_vertices
        xyz = patch_local2d(vertices, cx_km, cy_km, depth_km, strike_rad, dip_rad)
        dx_km = float(rect_dx_km)
        dy_km = float(rect_dy_km)
        _, rect_corners = triangle_to_rectangles(xyz[:, :2], dx_km, dy_km)
        rect_corners_3d = [
            patch_local2d_inv(
                np.column_stack([c, np.zeros((c.shape[0], 1))]),
                cx_km, cy_km, depth_km, strike_rad, dip_rad
            )
            for c in rect_corners
        ]
        xs_top_left_km = np.array([c[-1][0] for c in rect_corners_3d])
        ys_top_left_km = np.array([c[-1][1] for c in rect_corners_3d])
        depth_top_left_km = np.array([c[-1][2] for c in rect_corners_3d])
        xs = (xs_top_left_km - mean_x_km) * 1000.0
        ys = (ys_top_left_km - mean_y_km) * 1000.0
        valid = depth_top_left_km >= 0
        xs = xs[valid]
        ys = ys[valid]
        depth_top_left_km = depth_top_left_km[valid]
        sourceparams = (xs, ys, depth_top_left_km*1000.0, dy_km*1000.0, dx_km*1000.0, strike_deg, dip_deg, mean_x_km, mean_y_km)
    else:
        msg = f"Unknown patchType: {patchType}"
        logger.error(msg)
        raise ValueError(msg)
    return (p, sourceparams)

def _prepare_patch_source_wrapper(args):
    return _prepare_patch_source(*args)


def _edcmp_source_bundle_size(sourceparams):
    xs = np.atleast_1d(np.asarray(sourceparams[0], dtype=np.float64))
    return int(xs.size)


def _get_edcmp_patch_sources(fault, n_jobs, rect_dx_km=0.1, rect_dy_km=0.1):
    from concurrent.futures import ProcessPoolExecutor

    n_patch = len(fault.patch)
    cache_key = (
        n_patch,
        fault.patchType,
        round(float(rect_dx_km), 6),
        round(float(rect_dy_km), 6),
    )

    # Check structured cache
    cached = getattr(fault, '_edcmp_source_cache', None)
    if (
        cached is not None
        and isinstance(cached, EdcmpPatchSourceCache)
        and len(cached.sources) == n_patch
        and cached.cache_key == cache_key
    ):
        return cached.sources, cached.mean_x_km, cached.mean_y_km

    mean_x_km = np.mean([np.mean([p[i][0] for i in range(len(p))]) for p in fault.patch])
    mean_y_km = np.mean([np.mean([p[i][1] for i in range(len(p))]) for p in fault.patch])
    patch_geometries = [fault.getpatchgeometry(p, center=True) for p in range(n_patch)]
    patch_args = [
        (
            p,
            fault.patchType,
            patch_geometries[p],
            fault.patch[p],
            mean_x_km,
            mean_y_km,
            rect_dx_km,
            rect_dy_km,
        )
        for p in range(n_patch)
    ]

    max_workers = int(n_jobs) if n_jobs is not None else 1
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_prepare_patch_source_wrapper, patch_args))
    else:
        results = list(map(_prepare_patch_source_wrapper, patch_args))

    results.sort(key=lambda x: x[0])
    sources = [sourceparams for p, sourceparams in results]
    fault._edcmp_source_cache = EdcmpPatchSourceCache(
        sources=sources,
        mean_x_km=mean_x_km,
        mean_y_km=mean_y_km,
        cache_key=cache_key,
    )
    return sources, mean_x_km, mean_y_km

def _edcmp_patch_task(args):
    """
    Helper function for parallel EDCMP patch computation.
    Uses precomputed source parameters for each patch, so no geometry or projection is recalculated here.

    Parameters
    ----------
    args : tuple
        (p, SLP, data, workdir, grn_dir, output_dir, layered_model, verbose, Np,
         mean_x_km, mean_y_km, p_vertices, patchType, faultname, force_recompute, sourceparameters)
        - sourceparameters: tuple, precomputed and ready to be passed to edcmpslip2dis

    Returns
    -------
    p : int
        Patch index.
    ss, ds, ts : np.ndarray
        Green's functions for strike-slip, dip-slip, tensile-slip (Nd, 3).
    """
    (p, SLP, data, workdir, grn_dir, output_dir, layered_model, verbose, Np,
     mean_x_km, mean_y_km, p_vertices, patchType, faultname, force_recompute, sourceparameters) = args
    dataname = data.name
    patch_prefix = f'p{p}'

    # sourceparameters is a tuple: (xs, ys, depth, width, length, strike_deg, dip_deg, mean_x_km, mean_y_km)
    ss, ds, ts = edcmpslip2dis(
        data, sourceparameters,
        slip=SLP,
        grn_dir=grn_dir,
        output_dir=output_dir,
        filename_suffix=patch_prefix,
        workdir=workdir,
        layered_model=layered_model,
        force_recompute=force_recompute,
        faultname=faultname,
        dataname=dataname
    )

    return p, ss, ds, ts

def _single_patch_forward(args):
    p, sourceparams, slip, data, grn_dir, output_dir, workdir, layered_model, force_recompute, faultname = args
    return edcmpslip2dis_forward(
        data, sourceparams,
        slip=slip,
        grn_dir=grn_dir,
        output_dir=output_dir,
        filename_suffix=f'surface_p{p}',
        workdir=workdir,
        layered_model=layered_model,
        force_recompute=force_recompute,
        faultname=faultname,
        dataname=data.name
    )


def _single_patch_inmemory_forward(args):
    p, sourceparams, slip, data, grn_dir, workdir, engine, module_dir = args
    disp = compute_inmemory_edcmp_forward(
        data,
        sourceparams,
        slip=slip,
        engine=engine,
        grn_dir=grn_dir,
        workdir=workdir,
        module_dir=module_dir,
    )
    return p, disp


def _single_patch_inmemory_greens(args):
    p, sourceparams, slip, data, grn_dir, workdir, engine, module_dir = args
    ss, ds, ts = compute_inmemory_edcmp_greens(
        data,
        sourceparams,
        slip=slip,
        engine=engine,
        grn_dir=grn_dir,
        workdir=workdir,
        module_dir=module_dir,
        use_shared_memory=True,
    )
    return p, ss, ds, ts

#class Fault
class Fault(SourceInv):

    '''
        Parent class implementing what is common in all fault objects.

        You can specify either an official utm zone number or provide
        longitude and latitude for a custom zone.

        Args:
            * name          : Name of the fault.
            * utmzone       : UTM zone  (optional, default=None)
            * lon0          : Longitude defining the center of the custom utm zone
            * lat0          : Latitude defining the center of the custom utm zone
            * ellps         : ellipsoid (optional, default='WGS84')
    '''

    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):

        # Base class init
        super(Fault,self).__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0)

        # Initialize the fault
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing fault {}".format(self.name))
        self.verbose = verbose

        self.type = "Fault"

        # Specify the type of patch
        self.patchType = None

        # Set the reference point in the x,y domain (not implemented)
        self.xref = 0.0
        self.yref = 0.0

        # Allocate fault trace attributes
        self.xf   = None # original non-regularly spaced coordinates (UTM)
        self.yf   = None
        self.xi   = None # regularly spaced coordinates (UTM)
        self.yi   = None
        self.loni = None # regularly spaced coordinates (geographical)
        self.lati = None
        self.lon  = None
        self.lat  = None


        # Allocate depth attributes
        self.top = None             # Depth of the top of the fault
        self.depth = None           # Depth of the bottom of the fault

        # Allocate patches
        self.patch     = None
        self.slip      = None
        self.N_slip    = None # This will be the number of slip values
        self.totalslip = None
        self.Cm        = None
        self.mu        = None
        self.numz      = None

        # Remove files
        self.cleanUp = True

        # Create a dictionnary for the polysol
        self.polysol = {}

        # Create a dictionary for the Green's functions and the data vector
        self.G = {}
        self.d = {}

        # Create structure to store the GFs and the assembled d vector
        self.Gassembled = None
        self.dassembled = None

        # Adjacency map for the patches
        self.adjacencyMap = None

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Set up whats needed for an empty fault
    def initializeEmptyFault(self):
        '''
        Initializes what is required for a fualt with no patches

        Returns: 
            * None
        '''

        # Initialize
        self.patch = []
        self.patchll = []
        self.N_slip = 0
        self.initializeslip()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Returns a copy of the fault
    def duplicateFault(self):
        '''
        Returns a full copy (copy.deepcopy) of the fault object.

        Return:
            * fault         : fault object
        '''

        return copy.deepcopy(self)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Initialize the slip vector
    def initializeslip(self, n=None, values=None):
        '''
        Re-initializes the fault slip array to zero values.
        Slip array will be the size of the number of patches/tents times the
        3 components of slip (strike-slip, dip slip and tensile).

        - 1st Column is strike slip
        - 2nd Column is dip slip
        - 3rd Column is tensile

        Kwargs:
            * n             : Number of slip values. If None, it'll take the number of patches.
            * values        : Can be 'depth', 'strike', 'dip', 'length', 'width', 'area', 'index' or a numpy array. The array can be of size (n,3) or (n,1)

        Returns:
            * None
        '''

        # Shape
        if n is None:
           self.N_slip = len(self.patch)
        else:
            self.N_slip = n

        self.slip = np.zeros((self.N_slip,3))

        # Values
        if values is not None:
            # string type
            if type(values) is str:
                if values == 'depth':
                    values = np.array([self.getpatchgeometry(p, center=True)[2] for p in self.patch])
                elif values == 'strike':
                    values = np.array([self.getpatchgeometry(p, center=True)[5] for p in self.patch])
                elif values == 'dip':
                    values = np.array([self.getpatchgeometry(p, center=True)[6] for p in self.patch])
                elif values == 'length':
                    values = np.array([self.getpatchgeometry(p, center=True)[4] for p in self.patch])
                elif values == 'width':
                    values = np.array([self.getpatchgeometry(p, center=True)[3] for p in self.patch])
                elif values == 'area':
                    self.computeArea()
                    values = self.area
                elif values == 'index':
                    values = np.array([float(self.getindex(p)) for p in self.patch])
                self.slip[:,0] = values
            # Numpy array
            if type(values) is np.ndarray:
                try:
                    self.slip[:,:] = values
                except:
                    try:
                        self.slip[:,0] = values
                    except:
                        logger.error('Wrong size for the slip array provided')
                        return

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Add some fault traces to plot with self
    def addfaults(self, filename):
        '''
        Add some other faults to plot with the modeled one.

        Args:
            * filename  : Name of the file. File is ascii format. First column is longitude. Second column is latitude. Separator between faults is > as in GMT style

        Return:
            * None
        '''

        # Allocate a list
        self.addfaults = []

        # Read the file
        fin = open(filename, 'r')
        A = fin.readline()
        tmpflt=[]
        while len(A.split()) > 0:
            if A.split()[0] == '>':
                if len(tmpflt) > 0:
                    self.addfaults.append(np.array(tmpflt))
                tmpflt = []
            elif A.split()[0] == '#':
                pass # comment line, ignore
            else:
                lon = float(A.split()[0])
                lat = float(A.split()[1])
                tmpflt.append([lon,lat])
            A = fin.readline()
        fin.close()

        # Convert to utm
        self.addfaultsxy = []
        for fault in self.addfaults:
            x,y = self.ll2xy(fault[:,0], fault[:,1])
            self.addfaultsxy.append([x,y])

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def trace2xy(self):
        '''
        Transpose the fault trace lat/lon into the UTM reference.
        UTM coordinates are stored in self.xf and self.yf in km

        Returns:
            * None
        '''

        # do it
        self.xf, self.yf = self.ll2xy(self.lon, self.lat)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def trace2ll(self):
        '''
        Transpose the fault trace UTM coordinates into lat/lon.
        Lon/Lat coordinates are stored in self.lon and self.lat in degrees

        Returns:
            * None
        '''

        # do it
        self.lon, self.lat = self.xy2ll(self.xf, self.yf)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def patch2xy(self):
        '''
        Takes all the patches in self.patchll and convert them to xy
        Patches are stored in self.patch

        Returns:
            * None
        '''

        # Create list
        patch = []

        # Iterate
        for patchll in self.patchll:
            # Create a patch
            p = []
            # Iterate again
            for pll in patchll.tolist():
                x, y = self.ll2xy(pll[0], pll[1])
                p.append([x, y, pll[2]])
            patch.append(np.array(p))

        # Save
        self.patch = patch

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def patch2ll(self):
        '''
        Takes all the patches in self.patch and convert them to lonlat.
        Patches are stored in self.patchll

        Returns:
            * None
        '''

        # Create list
        patchll = []

        # Iterate
        for patch in self.patch:
            # Create a patch
            pll = []
            # Iterate again
            for p in patch.tolist():
                lon, lat = self.xy2ll(p[0], p[1])
                pll.append([lon, lat, p[2]])
            patchll.append(np.array(pll))

        # Save
        self.patchll = patchll

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setTrace(self,delta_depth=0., sort='y'):
        '''
        Uses the patches to build a fault trace. Fault trace is made of the
        vertices that are shallower than fault top + delta_depth
        Fault trace is in self.xf and self.yf

        Args:
            * delta_depth       : Depth extension below top of the fault

        '''
        self.xf = []
        self.yf = []

        # Set top
        if self.top is None:
            depth = [[p[2] for p in patch] for patch in self.patch]
            depth = np.unique(np.array(depth).flatten())
            self.top = np.min(depth)
            self.depth = np.max(depth)

        minz = np.round(self.top+delta_depth,1)
        for p in self.patch:
            for v in p:
                if np.round(v[2],1)>=minz:
                    continue
                self.xf.append(v[0])
                self.yf.append(v[1])
        self.xf = np.array(self.xf)
        self.yf = np.array(self.yf)
        if sort=='y':
            i = np.argsort(self.yf)
        elif sort=='x':
            i = np.argsort(self.xf)
        self.xf = self.xf[i]
        self.yf = self.yf[i]

        # Set lon lat
        self.trace2ll()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def trace(self, x, y, utm=False):
        '''
        Set the surface fault trace from Lat/Lon or UTM coordinates
        Surface fault trace is stored in self.xf, self.yf (UTM) and
        self.lon, self.lat (Lon/lat)

        Args:
            * Lon           : Array/List containing the Lon points.
            * Lat           : Array/List containing the Lat points.

        Kwargs:
            * utm           : If False, considers x and y are lon/lat. If True, considers x and y are utm in km

        Returns:
            * None
        '''

        # Set lon and lat
        if utm:
            self.xf  = np.array(x) # /1000.
            self.yf  = np.array(y) # /1000.
            # to lat/lon
            self.trace2ll()
        else:
            self.lon = np.array(x)
            self.lat = np.array(y)
            # utmize
            self.trace2xy()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def file2trace(self, filename, utm=False, header=0):
        '''
        Reads the fault trace from a text file (ascii 2 columns)
            - If utm is False, format is Lon Lat
            - If utm is True, format is X Y (in km)

        Args:
            * filename      : Name of the fault file.

        Kwargs:
            * utm           : Specify nature of coordinates
            * header        : Number of lines to skip at the beginning of the file

        Returns:
            * None
        '''

        # Open the file
        fin = open(filename, 'r')

        # Read the whole thing
        A = fin.readlines()

        # store these into Lon Lat
        x = []
        y = []
        for i in range(header, len(A)):
            x.append(float(A[i].split()[0]))
            y.append(float(A[i].split()[1]))

        # Create the trace
        self.trace(x, y, utm)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def discretize(self, every=2., tol=0.01, fracstep=0.2, xaxis='x',
                         cum_error=True):
        '''
        Refine the surface fault trace by setting a constant distance between
        each point. Pay attention, the fault cannot be exactly a straight
        line north-south. Descretized fault trace is stored in self.xi and
        self.yi

        Kwargs:
            * every         : Spacing between each point (in km)
            * tol           : Tolerance in the spacing (in km)
            * fracstep      : fractional step in the chosen direction for the discretization optimization
            * xaxis         : Axis for the discretization. 'x'= use x as the axis, 'y'= use y as the axis
            * cum_error     : if True, accounts for cumulated error to define the axis bound for the last patch

        Returns:
            * None
        '''

        # Check if the fault is in UTM coordinates
        if self.xf is None:
            self.trace2xy()

        if xaxis=='x':
            xf = self.xf
            yf = self.yf
        else:
            yf = self.xf
            xf = self.yf

        # Import the interpolation routines
        import scipy.interpolate as scint

        # Build the interpolation
        od = np.argsort(xf)
        f_inter = scint.interp1d(xf[od], yf[od], bounds_error=False)

        # Initialize the list of equally spaced points
        xi = [xf[od][0]]                               # Interpolated x fault
        yi = [yf[od][0]]                               # Interpolated y fault
        xlast = xf[od][-1]                             # Last point
        ylast = yf[od][-1]

        # First guess for the next point
        xt = xi[-1] + every * fracstep
        yt = f_inter(xt)
        # Check if first guess is in the domain
        if xt>xlast-tol:
            xt = xlast
            xi.append(xt)
            yi.append(f_inter(xt))
        # While the last point is not the last wanted point
        total_error = 0.
        mod_error   = 0.
        while (xi[-1] < xlast):
            # I compute the distance between me and the last accepted point
            d = np.sqrt( (xt-xi[-1])**2 + (yt-yi[-1])**2 )
            # Check if I am in the tolerated range
            if np.abs(d-every)<tol:
                xi.append(xt)
                yi.append(yt)
            else:
                # While I am to far away from my goal and I did not pass the last x
                while ((np.abs(d-every)>tol) and (xt<xlast)):
                    # I add the distance*frac that I need to go
                    xt += (every-d)*fracstep
                    # If I passed the last point (accounting for error in previous steps)
                    if (np.round(xt,decimals=2)>=np.round(xlast-mod_error-tol,decimals=2)):
                        xt = xlast
                    elif (xt<xi[-1]):  # If I passed the previous point
                        xt = xi[-1] + every
                    # I compute the corresponding yt
                    yt = f_inter(xt)
                    # I compute the corresponding distance
                    d = np.sqrt( (xt-xi[-1])**2 + (yt-yi[-1])**2 )
                # When I stepped out of that loop, append
                if cum_error:
                    total_error += every - d
                    mod_error    = np.abs(total_error)%(0.5*every)
                xi.append(xt)
                yi.append(yt)
            # Next guess for the loop
            xt = xi[-1] + every * fracstep

        # After interpolation, reorder the points to match the original xf order
        # Check if the original data was in ascending or descending order
        if xf[0] < xf[-1]:
            # If the original data was in ascending order, sort the interpolated data in ascending order
            xi, yi = zip(*sorted(zip(xi, yi)))
        else:
            # If the original data was in descending order, sort the interpolated data in descending order
            xi, yi = zip(*sorted(zip(xi, yi), reverse=True))

        # Store the result in self
        if xaxis=='x':
            self.xi = np.array(xi)
            self.yi = np.array(yi)
        else:
            self.yi = np.array(xi)
            self.xi = np.array(yi)

        # Compute the lon/lat
        self.loni, self.lati = self.xy2ll(self.xi, self.yi)

        # All done
        return

    def discretize_trace(self, every, threshold=2):
        """
        Discretize the fault trace at regular intervals.
    
        Args:
            * every (float): Interval at which to discretize the trace.
            * threshold (float): Threshold distance to check the first and last vertex. Default is 2 km.
                Add the first and last vertex in the rupture trace (self.xf, self.yf) if the distance to the nearest r_new point is greater than the threshold.
    
        Returns:
            * None
        """
        x, y = self.xf, self.yf
        # Calculate the length of the curve
        dx = np.insert(np.diff(x), 0, 0)
        dy = np.insert(np.diff(y), 0, 0)
        dr = np.sqrt(dx*dx + dy*dy)
        r = cumtrapz(dr, initial=0)  # Length of the curve
    
        # Create interpolation functions
        fx = interp1d(r, x, kind='linear')
        fy = interp1d(r, y, kind='linear')
    
        # Discretize the curve length at regular intervals
        num_points = int(np.floor(r[-1] / every))
        remainder = r[-1] - num_points * every
        if remainder >= every / 2:
            num_points += 1
        r_new = np.linspace(0, r[-1], num_points)
    
        # Check the distance of the first and last vertex to the nearest r_new point
        if r_new[0] - r[0] > threshold:
            r_new = np.insert(r_new, 0, r[0])
        if r[-1] - r_new[-1] > threshold:
            r_new = np.append(r_new, r[-1])
    
        # Calculate new x and y values
        x_new = fx(r_new)
        y_new = fy(r_new)
    
        self.xi, self.yi = x_new, y_new
        # Compute the lon/lat
        self.loni, self.lati = self.xy2ll(self.xi, self.yi)
    
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getPatchesInRange(self, method='buffer', **kwargs):
        """
        Get patch indices within specified range.
        
        Parameters
        ----------
        method : str
            Method to select patches:
            - 'buffer': Select patches within buffer distance perpendicular to fault trace between two points
            - 'box': Select patches within longitude/latitude bounding box
            
        **kwargs : dict
            For 'buffer' method:
                point1 : tuple or list
                    First reference point (lon, lat) or (x, y) depending on coord_system
                point2 : tuple or list  
                    Second reference point (lon, lat) or (x, y) depending on coord_system
                buffer_distance : float
                    Buffer distance in km (perpendicular to fault trace)
                coord_system : str, default='lonlat'
                    Coordinate system of input point ('lonlat' or 'xy')
                depth_range : tuple, optional
                    Depth range (min_depth, max_depth) in km. If None, all depths included.
                    
            For 'box' method:
                lon_range : tuple
                    Longitude range (min_lon, max_lon) in degrees
                lat_range : tuple  
                    Latitude range (min_lat, max_lat) in degrees
                depth_range : tuple, optional
                    Depth range (min_depth, max_depth) in km. If None, all depths included.
        
        Returns
        -------
        patch_indices : list
            List of patch indices within the specified range
            
        Examples
        --------
        # Buffer method - select patches between two points with perpendicular buffer
        >>> indices = fault.getPatchesInRange(
        ...     method='buffer',
        ...     point1=(120.5, 24.2),
        ...     point2=(121.0, 24.8),
        ...     buffer_distance=10.0,
        ...     coord_system='lonlat',
        ...     depth_range=(0, 20)
        ... )
        
        # Box method - select patches within lon/lat bounding box
        >>> indices = fault.getPatchesInRange(
        ...     method='box', 
        ...     lon_range=(120.0, 121.0),
        ...     lat_range=(24.0, 25.0),
        ...     depth_range=(0, 15)
        ... )
        """
        
        import numpy as np
        from shapely.geometry import LineString, Point
        from shapely.ops import nearest_points
        
        # Get patch centers
        centers = self.getcenters()  # Returns [(x, y, z), ...]
        patch_indices = []
        
        if method == 'buffer':
            # Extract required parameters
            point1 = kwargs.get('point1')
            point2 = kwargs.get('point2')
            buffer_distance = kwargs.get('buffer_distance')
            coord_system = kwargs.get('coord_system', 'lonlat')
            depth_range = kwargs.get('depth_range', None)
            
            if point1 is None or point2 is None or buffer_distance is None:
                msg = "For 'buffer' method, 'point1', 'point2' and 'buffer_distance' are required"
                logger.error(msg)
                raise ValueError(msg)
            
            # Convert point coordinates to xy if necessary
            if coord_system == 'lonlat':
                point1_x, point1_y = self.ll2xy(point1[0], point1[1])
                point2_x, point2_y = self.ll2xy(point2[0], point2[1])
            else:
                point1_x, point1_y = point1[0], point1[1]
                point2_x, point2_y = point2[0], point2[1]
            
            # Create fault trace LineString
            if not hasattr(self, 'xi') or self.xi is None:
                # Use original fault trace if discretized trace doesn't exist
                trace_x, trace_y = self.xf, self.yf
            else:
                # Use discretized fault trace
                trace_x, trace_y = self.xi, self.yi
                
            if trace_x is None or trace_y is None:
                msg = "Fault trace is not defined. Please set fault trace first."
                logger.error(msg)
                raise ValueError(msg)
            
            # Create LineString from fault trace
            trace_coords = list(zip(trace_x, trace_y))
            fault_linestring = LineString(trace_coords)
            
            # Find the nearest points on fault trace for both input points
            query_point1 = Point(point1_x, point1_y)
            query_point2 = Point(point2_x, point2_y)
            nearest_point1_on_trace = nearest_points(fault_linestring, query_point1)[0]
            nearest_point2_on_trace = nearest_points(fault_linestring, query_point2)[0]
            
            # Get the positions along the fault trace
            pos1 = fault_linestring.project(nearest_point1_on_trace)
            pos2 = fault_linestring.project(nearest_point2_on_trace)
            
            # Ensure pos1 < pos2
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            
            # Calculate perpendicular distance from each patch center to fault trace
            # and check if it's along the specified segment
            for i, (cx, cy, cz) in enumerate(centers):
                patch_point = Point(cx, cy)
                nearest_point_on_trace = nearest_points(fault_linestring, patch_point)[0]
                
                # Get position along fault trace
                patch_pos = fault_linestring.project(nearest_point_on_trace)
                
                # Check if patch is within the along-strike range
                if pos1 <= patch_pos <= pos2:
                    # Distance from patch center to nearest point on fault trace
                    distance_to_trace = np.sqrt((cx - nearest_point_on_trace.x)**2 + 
                                              (cy - nearest_point_on_trace.y)**2)
                    
                    # Check if within buffer distance
                    if distance_to_trace <= buffer_distance:
                        # Check depth range if specified
                        if depth_range is not None:
                            if depth_range[0] <= cz <= depth_range[1]:
                                patch_indices.append(i)
                        else:
                            patch_indices.append(i)
        
        elif method == 'box':
            # Extract required parameters
            lon_range = kwargs.get('lon_range')
            lat_range = kwargs.get('lat_range')
            depth_range = kwargs.get('depth_range', None)
            
            if lon_range is None or lat_range is None:
                msg = "For 'box' method, 'lon_range' and 'lat_range' are required"
                logger.error(msg)
                raise ValueError(msg)
            
            # Convert patch centers to lon/lat
            for i, (cx, cy, cz) in enumerate(centers):
                lon, lat = self.xy2ll(cx, cy)
                
                # Check if within lon/lat bounds
                if (lon_range[0] <= lon <= lon_range[1] and 
                    lat_range[0] <= lat <= lat_range[1]):
                    
                    # Check depth range if specified
                    if depth_range is not None:
                        if depth_range[0] <= cz <= depth_range[1]:
                            patch_indices.append(i)
                    else:
                        patch_indices.append(i)
        
        else:
            msg = f"Unknown method '{method}'. Use 'buffer' or 'box'"
            logger.error(msg)
            raise ValueError(msg)
        
        return patch_indices

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # def generate_checkboard_slip(self, Mu=3e10, horizontal_discretization=40, depth_ranges=[5, 15], 
    #                              normalize=True, target_moment=None, target_magnitude=None, rake_angle=0):
    #     """
    #     Generate a checkerboard slip distribution for testing.
    
    #     Parameters:
    #     - Mu: float, default=3e10
    #         Shear modulus.
    #     - horizontal_discretization: int, default=40
    #         Horizontal discretization distance for the fault trace.
    #     - depth_ranges: list, default=[5, 15]
    #         Depth ranges for vertical discretization.
    #     - normalize: bool, default=True
    #         Whether to normalize the slip to match a target moment magnitude.
    #     - target_moment: float, optional
    #         Target scalar moment for normalization. If None, normalize to the original slip moment.
    #     - target_magnitude: float, optional
    #         Target moment magnitude (Mw) for normalization. If provided, it will be converted to scalar moment.
    #     - rake_angle: float, default=0, unit: degree
    #         Rake angle for the slip distribution.
    
    #     Notes:
    #     - If both `target_moment` and `target_magnitude` are provided, `target_moment` takes precedence.
    #     """
    #     def selectPatches_trans(fault, tvert1, tvert2, mindep, maxdep, tol=0.2):
    #         '''
    #         Select patches based on the given criteria.
    #         '''
    #         pselect = []
    #         tx1, ty1 = tvert1[0], tvert1[1]
    #         tx2, ty2 = tvert2[0], tvert2[1]
    #         slope = np.arctan2(ty2 - ty1, tx2 - tx1)
    #         slp_len = np.sqrt((ty2 - ty1)**2 + (tx2 - tx1)**2)
    #         for p in range(len(fault.patch)):
    #             x1, x2, x3, width, length, strike, dip = fault.getpatchgeometry(p)
    #             xy_trans = ((x1 - tx1) + (x2 - ty1) * 1j) * np.exp(-slope * 1j)
    #             x_trans = xy_trans.real
    #             if -tol <= x_trans < slp_len + tol and mindep < x3 < maxdep:
    #                 pselect.append(p)
    #         return pselect
    
    #     from .faultpostproc import faultpostproc
    
    #     # Step 1: Compute raw moment
    #     if normalize:
    #         if target_moment is None and target_magnitude is None:
    #             rawmoment = 0.0
    #             self.type = 'Fault'
    #             self.computeArea()
    #             rawproces = faultpostproc('Calculating_Moment', self, Mu=Mu, lon0=self.lon0, lat0=self.lat0, utmzone=self.utmzone)
    #             rawproces.computeMomentTensor()
    #             rawmoment += rawproces.computeScalarMoment()
    #             print(f"Raw moment Mo={rawmoment:.2e}")
    #         elif target_magnitude is not None and target_moment is None:
    #             # Mw = 2/3 * (log10(Mo) - 9.1) -> Mo = 10**((Mw * 3/2) + 9.1)
    #             target_moment = 10**((target_magnitude * 3.0 / 2.0) + 9.1)
    #             print(f"Converted target magnitude Mw={target_magnitude} to scalar moment Mo={target_moment:.2e}")
    
    #     # Step 3: Horizontal discretization
    #     self.setTrace(0.1)
    #     self.discretize_trace(every=horizontal_discretization)
    
    #     # Step 4: Vertical discretization and slip assignment
    #     self.initializeslip()
    #     rake_rad = np.radians(rake_angle)
    #     for i in range(len(depth_ranges) - 1):
    #         mindep, maxdep = depth_ranges[i], depth_ranges[i + 1]
    #         layer_cnt = 0 if i % 2 == 0 else 1
    #         for k in range(layer_cnt, len(self.xi) - 1, 2):
    #             # xmin, xmax = np.sort([self.xi[k], self.xi[k+1]])
    #             # ymin, ymax = np.sort([self.yi[k], self.yi[k+1]])
    #             pselect = selectPatches_trans(self, [self.xi[k], self.yi[k]], [self.xi[k + 1], self.yi[k + 1]], mindep, maxdep)
    #             self.slip[pselect, 0] = np.cos(rake_rad) * 1.0  # Strike-slip component
    #             self.slip[pselect, 1] = np.sin(rake_rad) * 1.0  # Dip-slip component
    
    #     # Step 5: Normalize slip to match target moment or original moment
    #     if normalize:
    #         # Compute new moment
    #         newmoment = 0.0
    #         postfault1 = faultpostproc('Calculating_Moment', self, Mu=Mu, lon0=self.lon0, lat0=self.lat0, utmzone=self.utmzone)
    #         postfault1.computeMomentTensor()
    #         newmoment += postfault1.computeScalarMoment()
    
    #         # Determine normalization factor
    #         if target_moment is not None:
    #             moment_ratio = target_moment / newmoment
    #         else:
    #             moment_ratio = rawmoment / newmoment
    
    #         print(f"Normalizing slip: Raw moment to New moment ratio = {moment_ratio:.2f}")
    #         self.slip[:, :] *= moment_ratio
    #     else:
    #         print("Slip normalization skipped. Using unit slip (1 m).")


    def generate_checkboard_slip(self, Mu=3e10, horizontal_discretization=None, depth_ranges=[5, 15], 
                                 normalize=True, target_moment=None, target_magnitude=None, rake_angle=0,
                                 start_with_slip=True):
        """
        Generate a checkerboard slip distribution for resolution tests.
        
        This method supports flexible horizontal discretization (by count, by distance, or by explicit list)
        and allows controlling the starting phase (slip vs. no-slip).

        Parameters:
        -----------
        Mu : float, default=3e10
            Shear modulus (Pascal).
        
        horizontal_discretization : int, float, list, or np.ndarray, optional
            Controls how the fault is discretized along the strike direction.
            - If int: Specifies the EXACT NUMBER of columns (e.g., 10).
            - If float: Specifies the TARGET AVERAGE WIDTH (km) of each column. 
              The method will calculate the optimal integer number of columns to fit the trace length uniformly.
            - If list/array: Specifies the EXPLICIT BOUNDARIES (km) along strike (e.g., [0, 5, 10, 20]).
            - If None: Defaults to dividing the trace into 10 uniform columns.

        depth_ranges : list, default=[5, 15]
            List of depth boundaries (km) defining the vertical rows. 
            e.g., [0, 5, 10, 20] defines 3 rows: 0-5, 5-10, 10-20.

        normalize : bool, default=True
            If True, scales the slip to match a target moment or magnitude.

        target_moment : float, optional
            Target scalar seismic moment (Nm). Overrides target_magnitude if provided.

        target_magnitude : float, optional
            Target Moment Magnitude (Mw). Used if target_moment is None.

        rake_angle : float, default=0
            Rake angle in degrees (0 = Left-lateral, 90 = Reverse, -90 = Normal).

        start_with_slip : bool, default=True
            Controls the pattern phase of the top-left block (first row, first column).
            - True: The first block has SLIP ("Black").
            - False: The first block has NO SLIP ("White").

        Notes:
        ------
        This method uses vectorized operations and curvilinear projection for high performance 
        and geometric accuracy on complex fault surfaces.
        """
        import numpy as np
        from .faultpostproc import faultpostproc

        # ==========================================
        # 1. Trace Validation & Generation
        # ==========================================
        # Ensure the fault trace (self.xi, self.yi) exists for distance calculations.
        trace_valid = False
        if hasattr(self, 'xi') and self.xi is not None:
            if np.ndim(self.xi) > 0 and len(self.xi) > 1:
                trace_valid = True
                
        if not trace_valid:
            logger.warning("Trace (xi, yi) not detected or invalid. Automatically building trace...")
            if hasattr(self, 'setTrace'):
                self.setTrace(0.1)  # Set trace from top edge
            
            # Default discretization for trace construction if not specified
            trace_step = 5.0
            if isinstance(horizontal_discretization, (int, float)):
                trace_step = horizontal_discretization if isinstance(horizontal_discretization, float) else 5.0
            
            # Discretize the trace to  target step/10 for better accuracy
            if hasattr(self, 'discretize_trace'):
                self.discretize_trace(every=trace_step/10.0)
                
            if not hasattr(self, 'xi') or self.xi is None:
                 msg = "Failed to generate fault trace. Check fault geometry."
                 logger.error(msg)
                 raise ValueError(msg)

        # ==========================================
        # 2. Geometry Projection
        # ==========================================
        # Get geometric centers of all sub-faults (N, 3)
        centers = self.getcenters() 
        n_patches = centers.shape[0]

        trace_x = np.array(self.xi)
        trace_y = np.array(self.yi)
        
        # Calculate the curvilinear coordinate (distance along strike) for each patch
        # Uses the optimized vectorized projection method
        patch_along_strike_dist = self._project_to_trace_vectorized(centers[:, :2], trace_x, trace_y)

        # Calculate total length of the fault trace
        d_segments = np.sqrt(np.diff(trace_x)**2 + np.diff(trace_y)**2)
        total_trace_len = np.sum(d_segments)

        # ==========================================
        # 3. Grid Indexing (Discretization Logic)
        # ==========================================
        
        # --- Vertical (Depth) Discretization ---
        # Map depth to row indices. depth_ranges must be sorted.
        # digitize returns 1-based index, subtract 1 for 0-based.
        row_indices = np.digitize(centers[:, 2], depth_ranges) - 1
        n_rows_defined = len(depth_ranges) - 1
        
        # --- Horizontal (Strike) Discretization ---
        strike_edges = None
        
        # Case A: Default (if None)
        if horizontal_discretization is None:
            horizontal_discretization = 10 # Default to integer count

        # Case B: Integer -> Number of Columns (Uniform)
        if isinstance(horizontal_discretization, int):
            n_cols = max(1, horizontal_discretization)
            strike_edges = np.linspace(0, total_trace_len, n_cols + 1)
            logger.info(f"Checkerboard: Dividing trace ({total_trace_len:.2f} km) into {n_cols} uniform columns.")

        # Case C: Float -> Average Distance (Uniform Resampling)
        elif isinstance(horizontal_discretization, float):
            target_width = horizontal_discretization
            raw_n_cols = total_trace_len / target_width
            n_cols = int(np.round(raw_n_cols))
            n_cols = max(1, n_cols) # Ensure at least 1 column
            strike_edges = np.linspace(0, total_trace_len, n_cols + 1)
            actual_width = total_trace_len / n_cols
            logger.info(f"Checkerboard: Optimized {target_width} km -> {actual_width:.4f} km width ({n_cols} cols).")

        # Case D: List/Array -> Explicit Edges (Custom/Non-uniform)
        elif isinstance(horizontal_discretization, (list, np.ndarray, tuple)):
            strike_edges = np.array(horizontal_discretization)
            # Basic validation
            if strike_edges.ndim != 1 or len(strike_edges) < 2:
                msg = "Explicit horizontal_discretization must be a 1D list with at least 2 points."
                logger.error(msg)
                raise ValueError(msg)
            logger.info(f"Checkerboard: Using explicit horizontal edges: {strike_edges}")
        
        else:
            msg = "horizontal_discretization must be None, int, float, or list/array."
            logger.error(msg)
            raise TypeError(msg)

        # Map strike distance to column indices
        col_indices = np.digitize(patch_along_strike_dist, strike_edges) - 1
        n_cols_defined = len(strike_edges) - 1

        # ==========================================
        # 4. Pattern Generation & Assignment
        # ==========================================
        
        # Create masks for valid patches (those falling within the defined ranges)
        valid_depth_mask = (row_indices >= 0) & (row_indices < n_rows_defined)
        valid_strike_mask = (col_indices >= 0) & (col_indices < n_cols_defined)
        
        # --- Phase Logic ---
        # If start_with_slip is True (Black start): (0,0) -> 0%2==0 -> True (Slip)
        # If start_with_slip is False (White start): (0,0) -> 0%2!=1 -> False (No Slip)
        remainder_target = 0 if start_with_slip else 1
        
        # Determine which patches get slip
        checker_mask = ((row_indices + col_indices) % 2 == remainder_target) & valid_depth_mask & valid_strike_mask

        # Assign Slip
        self.initializeslip() # Reset all slip to 0
        rake_rad = np.radians(rake_angle)
        
        self.slip[checker_mask, 0] = np.cos(rake_rad) * 1.0 
        self.slip[checker_mask, 1] = np.sin(rake_rad) * 1.0 
        
        assigned_count = np.sum(checker_mask)
        logger.info(f"Checkerboard generated: {assigned_count}/{n_patches} patches assigned slip.")

        # ==========================================
        # 5. Normalization
        # ==========================================
        if normalize and assigned_count > 0:
            postfault = faultpostproc('Calculating_Moment', self, Mu=Mu, lon0=self.lon0, lat0=self.lat0, utmzone=self.utmzone)
            postfault.computeMomentTensor()
            current_moment = postfault.computeScalarMoment()
            
            target_mo = None
            if target_moment is not None:
                target_mo = target_moment
            elif target_magnitude is not None:
                target_mo = 10**((target_magnitude * 1.5) + 9.1)
            
            if target_mo is not None:
                if current_moment > 1e-9: # Avoid division by zero
                    ratio = target_mo / current_moment
                    self.slip *= ratio
                    logger.info(f"Normalization: Scaling slip by factor {ratio:.4f} to match Mo={target_mo:.2e}")
                else:
                    logger.warning("Current moment is effectively zero despite slip assignment. Check mesh area.")
            else:
                logger.info("Normalization: Skipped (No target moment/magnitude specified).")
        elif normalize and assigned_count == 0:
            logger.info("Normalization: Skipped (No patches assigned slip).")

    def _project_to_trace_vectorized(self, points, trace_x, trace_y):
        """
        Vectorized calculation of curvilinear coordinates (distance along trace).
        
        Projects points orthogonally onto the nearest segment of the polyline defined by trace_x, trace_y.
        Handles degenerate segments (length=0) robustly.

        Parameters:
        -----------
        points : np.ndarray (N, 2)
            The (x, y) coordinates of the points to project.
        trace_x, trace_y : np.ndarray
            Coordinates defining the trace vertices.

        Returns:
        --------
        s_coords : np.ndarray (N,)
            Distance along the trace from the start (0) to the projected point.
        """
        import numpy as np

        # 1. Prepare segment vectors
        p_start = np.vstack((trace_x[:-1], trace_y[:-1])).T
        p_end = np.vstack((trace_x[1:], trace_y[1:])).T
        
        seg_vectors = p_end - p_start 
        seg_lens_sq = np.sum(seg_vectors**2, axis=1) # Shape: (M,)
        seg_lens = np.sqrt(seg_lens_sq)
        
        # 2. Vector from segment start to points
        # points shape: (N, 2) -> (N, 1, 2)
        # p_start shape: (M, 2) -> (1, M, 2)
        diff = points[:, np.newaxis, :] - p_start[np.newaxis, :, :] # Shape: (N, M, 2)
        
        # 3. Calculate projection factor t
        # Using errstate to suppress divide-by-zero warnings for 0-length segments
        with np.errstate(divide='ignore', invalid='ignore'):
            # t = Dot(diff, seg) / |seg|^2
            t = np.sum(diff * seg_vectors[np.newaxis, :, :], axis=2) / seg_lens_sq[np.newaxis, :] # Shape: (N, M)
            
            # Fix: Handle zero-length segments by setting their t to 0
            zero_len_mask = (seg_lens_sq == 0)
            if np.any(zero_len_mask):
                t[:, zero_len_mask] = 0 
        
        # 4. Clamp t to segment bounds [0, 1]
        t_clamped = np.clip(t, 0, 1)
        
        # 5. Find closest point on each segment and distances
        closest_points_on_segs = p_start[np.newaxis, :, :] + t_clamped[:, :, np.newaxis] * seg_vectors[np.newaxis, :, :]
        dist_vecs = points[:, np.newaxis, :] - closest_points_on_segs
        distances_sq = np.sum(dist_vecs**2, axis=2) # Shape: (N, M)
        
        # 6. Identify the nearest segment for each point
        nearest_seg_indices = np.argmin(distances_sq, axis=1) # Shape: (N,)
        
        # 7. Calculate s-coordinate (Cumulative length + projection)
        segment_cum_dist = np.concatenate(([0], np.cumsum(seg_lens)))
        base_dist = segment_cum_dist[nearest_seg_indices]
        
        # Extract the t-value corresponding to the nearest segment
        n_range = np.arange(points.shape[0])
        best_t = t_clamped[n_range, nearest_seg_indices]
        best_seg_len = seg_lens[nearest_seg_indices]
        
        s_coords = base_dist + best_t * best_seg_len
        
        return s_coords
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def cumdistance(self, discretized=False):
        '''
        Computes the distance between the first point of the fault and every
        other point. The distance is cumulative along the fault.

        Args:
            * discretized           : if True, use the discretized fault trace (default False)

        Returns:
            * dis                   : Cumulative distance array
        '''

        # Get the x and y positions
        if discretized:
            x = self.xi
            y = self.yi
        else:
            x = self.xf
            y = self.yf

        # initialize
        dis = np.zeros((x.shape[0]))

        # Loop
        for i in range(1,x.shape[0]):
            d = np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
            dis[i] = dis[i-1] + d

        # all done
        return dis
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def distance2trace(self, lon, lat, discretized=False, coord='ll'):
        '''
        Computes the distance between a point and the trace of a fault.
        This is a slow method, so it has been recoded in a few places
        throughout the whole library.

        Args:
            * lon               : Longitude of the point.
            * lat               : Latitude of the point.

        Kwargs:
            * discretized       : Uses the discretized trace.
            * coord             : if 'll' or 'lonlat', input in degree. If 'xy' or 'utm', input in km

        Returns:
            * dalong            : Distance to the first point of the fault along the fault
            * dacross           : Shortest distance between the point and the fault
        '''

        # Get the cumulative distance along the fault
        cumdis = self.cumdistance(discretized=discretized)

        # ll2xy
        if coord in ('ll', 'lonlat'):
            x, y = self.ll2xy(lon, lat)
        elif coord in ('xy', 'utm'):
            x,y = lon, lat

        # Fault coordinates
        if discretized:
            xf = self.xi
            yf = self.yi
        else:
            xf = self.xf
            yf = self.yf

        # Compute the distance between the point and all the points
        d = scidis.cdist([[x,y]], [[xf[i], yf[i]] for i in range(len(xf))])[0]

        # Get the two closest points
        imin1 = d.argmin()
        dmin1 = d[imin1]
        d[imin1] = 999999.
        imin2 = d.argmin()
        dmin2 = d[imin2]
        d[imin2] = 999999.
        dtot = dmin1+dmin2

        # Along the fault?
        xc = (xf[imin1]*dmin1 + xf[imin2]*dmin2)/dtot
        yc = (yf[imin1]*dmin1 + yf[imin2]*dmin2)/dtot

        # Distance
        if dmin1<dmin2:
            jm = imin1
        else:
            jm = imin2
        dalong = cumdis[jm] + np.sqrt( (xc-xf[jm])**2 + (yc-yf[jm])**2 )
        dacross = np.sqrt((xc-x)**2 + (yc-y)**2)

        # All done
        return dalong, dacross
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getindex(self, p):
        '''
        Returns the index of a patch.

        Args:
            * p         : Patch from a fault object.

        Returns:
            * iout      : index of the patch
        '''

        # output index
        iout = None

        # Find the index of the patch
        for i in range(len(self.patch)):
            try:
                if (self.patch[i] == p).all():
                    iout = i
            except:
                if self.patch[i]==p:
                    iout = i

        # All done
        return iout
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getslip(self, p):
        '''
        Returns the slip vector for a patch or tent

        Args:
            * p         : patch or tent

        Returns:
            * iout      : Index of the patch or tent
        '''

        # Get patch index
        io = self.getindex(p)

        # All done
        return self.slip[io,:]
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def writeTrace2File(self, filename, ref='lonlat'):
        '''
        Writes the trace to a file. Format is ascii with two columns with
        either lon/lat (in degrees) or x/y (utm in km).

        Args:
            * filename      : Name of the file

        Kwargs:
            * ref           : can be lonlat or utm.

        Returns:
            * None
        '''

        # Get values
        if ref in ('utm'):
            x = self.xf*1000.
            y = self.yf*1000.
        elif ref in ('lonlat'):
            x = self.lon
            y = self.lat

        # Open file
        fout = open(filename, 'w')

        # Write
        for i in range(x.shape[0]):
            fout.write('{} {} \n'.format(x[i], y[i]))

        # Close file
        fout.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def saveGFs(self, dtype='d', outputDir='.',
                      suffix={'strikeslip':'SS',
                              'dipslip':'DS',
                              'tensile':'TS',
                              'coupling': 'Coupling'}):
        '''
        Saves the Green's functions in different files.

        Kwargs:
            * dtype       : Format of the binary data saved. 'd' for double. 'f' for float32
            * outputDir   : Directory to save binary data.
            * suffix      : suffix for GFs name (dictionary)

        Returns:
            * None
        '''

        # Print stuff
        if self.verbose:
            logger.info('Writing Greens functions to file for fault {}'.format(self.name))

        # Loop over the keys in self.G
        for data in self.G.keys():

            # Get the Green's function
            G = self.G[data]

            # Create one file for each slip componenets
            for c in G.keys():
                if G[c] is not None:
                    g = G[c].flatten()
                    n = self.name.replace(' ', '_')
                    d = data.replace(' ', '_')
                    filename = '{}_{}_{}.gf'.format(n, d, suffix[c])
                    g = g.astype(dtype)
                    g.tofile(os.path.join(outputDir, filename))

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def saveData(self, dtype='d', outputDir='.'):
        '''
        Saves the Data in binary files.

        Kwargs:
            * dtype       : Format of the binary data saved. 'd' for double. 'f' for float32
            * outputDir   : Directory to save binary data

        Returns:
            * None
        '''

        # Print stuff
        if self.verbose:
            logger.info('Writing Greens functions to file for fault {}'.format(self.name))

        # Loop over the data names in self.d
        for data in self.d.keys():

            # Get data
            D = self.d[data]

            # Write data file
            filename = '{}_{}.data'.format(self.name, data)
            D.tofile(os.path.join(outputDir, filename))

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildGFs(self, data, vertical=True, slipdir='sd',
                 method='homogeneous', verbose=True, convergence=None,
                 options=None):
        '''
        Builds the Green's function matrix based on the discretized fault.

        The Green's function matrix is stored in a dictionary.
        Each entry of the dictionary is named after the corresponding dataset.
        Each of these entry is a dictionary that contains 'strikeslip', 'dipslip',
        'tensile' and/or 'coupling'

        Args:
            * data          : Data object (gps, insar, optical, ...)

        Kwargs:
            * vertical      : If True, will produce green's functions for the vertical displacements in a gps object.
            * slipdir       : Direction of slip along the patches. Can be any combination of s (strikeslip), d (dipslip), t (tensile) and c (coupling)
            * method        : Can be 'okada', 'meade', 'edks', 'pscmp', 'edcmp', 'cutde', 'homogeneous', 'empty'
            * verbose       : Writes stuff to the screen (overwrites self.verbose)
            * convergence   : If coupling case, needs convergence azimuth and rate [azimuth in deg, rate]
            * options       : Method-specific configuration.
                EdcmpOptions, PscmpOptions, dict, or None.
                Use csi.describe_gf_options(method) to see available fields.

        Returns:
            * None
        '''

        if self.patchType == 'triangletent':
            assert method == 'edks', 'Homogeneous case not implemented for {} faults'.format(self.patchType)

        if method in ('homogeneous', 'Homogeneous'):
            if self.patchType == 'rectangle':
                method = 'Okada'
            elif self.patchType == 'triangle':
                method = 'cutde'
            elif self.patchType == 'triangletent':
                method = 'Meade'

        if verbose:
            logger.info('Greens functions computation method: {}'.format(method))

        if data.dtype == 'insar':
            if not vertical:
                if verbose:
                    logger.warning('---------------------------------')
                    logger.warning(' WARNING: You specified vertical=False')
                    logger.warning(' As this is dangerous for SAR data, switched to True...')
                    logger.warning(' SAR data are very sensitive to vertical displacements.')
                    logger.warning('---------------------------------')
                vertical = True

        opts = resolve_gf_options(method, options)

        method_lower = method.lower()
        if method_lower in ('okada', 'ok92', 'meade'):
            G = self.homogeneousGFs(data, vertical=vertical, slipdir=slipdir, verbose=verbose, convergence=convergence)
        elif method_lower in ('edks',):
            G = self.edksGFs(data, vertical=vertical, slipdir=slipdir, verbose=verbose, convergence=convergence)
        elif method_lower in ('pscmp', 'psgrn'):
            G = self.pscmpGFs(data, vertical=vertical, slipdir=slipdir, verbose=verbose,
                              convergence=convergence, options=opts)
        elif method_lower in ('edcmp', 'edgrn'):
            G = self.edcmpGFs(data, vertical=vertical, slipdir=slipdir, verbose=verbose,
                              convergence=convergence, options=opts)
        elif method_lower in ('cutde',):
            G = self.cutdeGFs(data, vertical=vertical, slipdir=slipdir, verbose=verbose, convergence=convergence)
        elif method_lower in ('empty',):
            G = self.emptyGFs(data, vertical=vertical, slipdir=slipdir, verbose=verbose, convergence=convergence)
        else:
            raise ValueError('Method {} not implemented'.format(method))

        data.setGFsInFault(self, G, vertical=vertical)

        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildGFs_multidataset_bycutde(self, datadict, verticaldict=None, slipdir='sd',
                                method='cutde', verbose=True, convergence=None,
                                pscmpinp='pscmp_template.inp', grn_dir='psgrnfcts'):
        '''
        Builds the Green's function matrix based on the discretized fault.

        The Green's function matrix is stored in a dictionary.
        Each entry of the dictionary is named after the corresponding dataset.
        Each of these entry is a dictionary that contains 'strikeslip', 'dipslip',
        'tensile' and/or 'coupling'

        Args:
            * datadict      : Data object (gps, insar, optical, ...)

        Kwargs:
            * vertical      : If True, will produce green's functions for the vertical displacements in a gps object.
            * slipdir       : Direction of slip along the patches. Can be any combination of s (strikeslip), d (dipslip), t (tensile) and c (coupling)
            * method        : Can be 'okada' (Okada, 1982) (rectangular patches only), 'meade' (Meade 2007) (triangular patches only), 'edks' (Zhao & Rivera, 2002), 'homogeneous' (Okada for rectangles, Meade for triangles)
            * verbose       : Writes stuff to the screen (overwrites self.verbose)
            * convergence   : If coupling case, needs convergence azimuth and rate [azimuth in deg, rate]

        Returns:
            * None

        **********************
        '''
        from .gps import gps
        tmpdata = gps(name='tmp', utmzone=self.utmzone, ellps=self.ellps, lon0=self.lon0, lat0=self.lat0, verbose=False)
        tmpx, tmpy = [], []
        key_order = []
        for ikey in datadict.keys():
            idata = datadict[ikey]
            # self.buildGFs(data, vertical=vertical, slipdir=slipdir, method='empty', verbose=verbose, convergence=convergence, pscmpinp=pscmpinp, psgrndir=psgrndir)
            tmpx.extend(idata.x.tolist())
            tmpy.extend(idata.y.tolist())
            key_order.append(ikey)
        tmpx = np.array(tmpx)
        tmpy = np.array(tmpy)
        tmp_sta = np.array(['{0:04d}'.format(i) for i in range(len(tmpx))])
        tmpdata.setStat(tmp_sta, tmpx, tmpy, loc_format='XY')

        # Compute the Green's functions Gss.shape is (3, npoints, npatches)
        Gss, Gds, Gts = self.cutdeGFsinGPS(tmpdata, vertical=True, slipdir='sdt', verbose=verbose, convergence=convergence)
        # print(Gss.shape, Gds.shape, Gts.shape, flush=True)

        # Separate the Green's functions for each type of data set
        st = 0
        ed = 0
        # ivertical =vertical
        for ikey in key_order:
            idata = datadict[ikey]
            ivertical = verticaldict[ikey]
            ed = st + len(idata.x)
            iGss = Gss[:, st:ed, :]
            iGds = Gds[:, st:ed, :]
            iGts = Gts[:, st:ed, :]
            st = ed

            # Data type check
            if idata.dtype == 'insar':
                if not ivertical:
                    if verbose:
                        logger.warning('---------------------------------')
                        logger.warning(' WARNING: You specified vertical=False')
                        logger.warning(' As this is dangerous for SAR data, switched to True...')
                        logger.warning(' SAR data are very sensitive to vertical displacements.')
                        logger.warning('---------------------------------')
                    ivertical = True

            # Build the dictionary
            iG = self._buildGFsdict(idata, iGss, iGds, iGts, slipdir=slipdir,
                                    convergence=convergence, vertical=ivertical)
            # Separate the Green's functions for each type of data set
            idata.setGFsInFault(self, iG, vertical=ivertical)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def surfaceGFs(self, data, slipdir='sd', verbose=True):
        '''
        Build the GFs for the surface slip case.
        We assume the data are within the bounds of the fault.

        Args:
            * data      : surfaceslip data ojbect

        Kwargs:
            * slipdir   : any combinatino of s and d. default: 'sd'
            * verbose   : Default True
        '''

        # Check
        assert data.dtype == 'surfaceslip', 'Only works for surfaceslip data type: {}'.format(data.dtype)

        # Number of parameters
        if self.patchType == 'triangletent':
            n = len(self.tent)
        else:
            raise NotImplementedError

        # Initialize
        Gss = None; Gds = None
        if 's' in slipdir: Gss = np.zeros((len(data.vel), n))
        if 'd' in slipdir: Gds = np.zeros((len(data.vel), n))

        # Find points at the surface
        if self.patchType == 'triangletent':
            # get the index of the points
            zeroD = np.flatnonzero(np.array([tent[2] for tent in self.tent])==0.)
            if len(zeroD)==0: 
                logger.info('No surface patches.')
                return None
            # Get their positions
            x = np.array([tent[0] for tent in self.tent])[zeroD]
            y = np.array([tent[1] for tent in self.tent])[zeroD]
            strike = self.getStrikes()[zeroD]
            # Iterate over the data points
            for i, (lon, lat) in enumerate(zip(data.lon, data.lat)):
                # Get the two closest points
                xd,yd = self.ll2xy(lon, lat)
                dis = np.sqrt((x-xd)**2 + (y-yd)**2)
                i1,i2 = np.argsort(dis)[:2]
                d1 = dis[i1]
                d2 = dis[i2]
                # Check that the data point is between the two fault points
                v1 = np.array([xd-x[i1], yd-y[i1]]); v1 /= np.linalg.norm(v1)
                v2 = np.array([xd-x[i2], yd-y[i2]]); v2 /= np.linalg.norm(v2)
                if sum(np.abs((np.arctan2(v1[1],v1[0])*180/np.pi, np.arctan2(v2[1],v2[0])*180/np.pi))) > 90.: # If the point is between the two, then interpolate
                    if 's' in slipdir:
                        Gss[i,zeroD[i1]] = d2/(d1+d2); Gss[i,zeroD[i2]] = d1/(d1+d2)
                    if 'd' in slipdir:
                        Gds[i,zeroD[i1]] = d2/(d1+d2); Gds[i,zeroD[i2]] = d1/(d1+d2)
                if data.los is not None:
                    los = data.los[i]
                    if 's' in slipdir:
                        v1 = np.array([-1.*np.sin(strike[i1]), np.cos(strike[i1]), 0])
                        v2 = np.array([-1.*np.sin(strike[i2]), np.cos(strike[i2]), 0])
                        Gss[i,zeroD[i1]] *= v1.dot(los)
                        Gss[i,zeroD[i2]] *= v2.dot(los)
                    if 'd' in slipdir:
                        v1 = np.array([0., 0., 1.])
                        v2 = np.array([0., 0., 1.])
                        Gds[i,zeroD[i1]] *= v1.dot(los)
                        Gds[i,zeroD[i2]] *= v2.dot(los)
        else:
            raise NotImplementedError

        # Build the dictionnary
        G = {'strikeslip':Gss, 'dipslip':Gds}

        # All done
        return G
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def emptyGFs(self, data, vertical=True, slipdir='sd', verbose=True, convergence=None):
        ''' 
        Build zero GFs.

        Args:
            * data          : Data object (gps, insar, optical, ...)

        Kwargs:
            * vertical      : If True, will produce green's functions for the vertical displacements in a gps object.
            * slipdir       : Direction of slip along the patches. Can be any combination of s (strikeslip), d (dipslip), t (tensile) and c (coupling)
            * verbose       : Writes stuff to the screen (overwrites self.verbose)

        Returns:
            * G             : Dictionnary of GFs
        '''

        # Create the dictionary
        G = {'strikeslip':None, 'dipslip':None, 'tensile':None, 'coupling':None}

        # Get shape
        if self.patchType == 'triangletent':
            nm = len(self.tent)
        else:
            nm = len(self.patch)

        # Get shape
        if data.dtype in ('insar', 'surfaceslip'):
            nd = len(data.vel)
        elif data.dtype in ('opticor'):
            nd = data.vel.shape[0]*2
        elif data.dtype in ('gps'):
            nd = data.vel_enu.shape[0]
            if vertical:
                nd *= 3
            else:
                nd *= 2
        elif data.dtype == 'crossfaultoffset':
            nd = data.obs_per_station * len(data.station)
        elif data.dtype == 'leveling':
            nd = len(data.vel)

        # Build dictionnary
        if 's' in slipdir:
            G['strikeslip'] = np.zeros((nd,nm))
        if 'd' in slipdir:
            G['dipslip'] = np.zeros((nd,nm))
        if 't' in slipdir:
            G['tensile'] = np.zeros((nd,nm))
        if 'c' in slipdir:
            G['coupling'] = np.zeros((nd,nm))

        # All done
        return G
    # ----------------------------------------------------------------------c

    # ----------------------------------------------------------------------
    def homogeneousGFs(self, data, vertical=True, slipdir='sd', verbose=True,
                             convergence=None):
        '''
        Builds the Green's functions for a homogeneous half-space.

        If your patches are rectangular, Okada's formulation is used (Okada, 1982)
        If your patches are triangular, Meade's formulation is used (Meade, 2007)


        Args:
            * data          : Data object (gps, insar, optical, ...)

        Kwargs:
            * vertical      : If True, will produce green's functions for the vertical displacements in a gps object.
            * slipdir       : Direction of slip along the patches. Can be any combination of s (strikeslip), d (dipslip), t (tensile) and c (coupling)
            * verbose       : Writes stuff to the screen (overwrites self.verbose)
            * convergence   : If coupling case, needs convergence azimuth and rate [azimuth in deg, rate]

        Returns:
            * G             : Dictionary of the built Green's functions
        '''

        # Dispatch for cross-fault offset data
        if data.dtype == 'crossfaultoffset':
            return self._build_gfs_crossfaultoffset_via_gps_concat(
                self.homogeneousGFs, data, slipdir=slipdir,
                vertical=True, verbose=verbose, convergence=convergence)

        # Check that we are not in this case
        assert self.patchType != 'triangletent',\
                'Need to run EDKS for that particular type of fault'

        # Print
        if verbose:
            logger.info('---------------------------------')
            logger.info("Building Green's functions for the data set ")
            logger.info("{} of type {} in a homogeneous half-space".format(data.name,
                                                                     data.dtype))

        # Initialize the slip vector
        SLP = []
        if 's' in slipdir:              # If strike slip is aksed
            SLP.append(1.0)
        else:                           # Else
            SLP.append(0.0)
        if 'd' in slipdir:              # If dip slip is asked
            SLP.append(1.0)
        else:                           # Else
            SLP.append(0.0)
        if convergence is not None:
            SLP = [1.0, 1.0]
            if 'c' not in slipdir:
                slipdir += 'c'
        if 'c' in slipdir:
            assert convergence is not None, 'No convergence azimuth and rate given'
        if 't' in slipdir:              # If tensile is asked
            SLP.append(1.0)
        else:                           # Else
            SLP.append(0.0)

        # Create the dictionary
        G = {'strikeslip':[], 'dipslip':[], 'tensile':[], 'coupling':[]}

        # Create the matrices to hold the whole thing
        Gss = np.zeros((3, len(data.x), len(self.patch)))
        Gds = np.zeros((3, len(data.x), len(self.patch)))
        Gts = np.zeros((3, len(data.x), len(self.patch)))

        # Loop over each patch
        for p in range(len(self.patch)):
            if verbose:
                sys.stdout.write('\r Patch: {} / {} '.format(p+1,len(self.patch)))
                sys.stdout.flush()

            # get the surface displacement corresponding to unit slip
            # ss,ds,op will all have shape (Nd,3) for 3 components
            ss, ds, ts = self.slip2dis(data, p, slip=SLP)

            # Store them
            Gss[:,:,p] = ss.T
            Gds[:,:,p] = ds.T
            Gts[:,:,p] = ts.T

        if verbose:
            logger.info(' ')

        # Build the dictionary
        G = self._buildGFsdict(data, Gss, Gds, Gts, slipdir=slipdir,
                               convergence=convergence, vertical=vertical)

        # All done
        return G
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setCustomGFs(self, data, G):
        '''
        Sets a custom Green's Functions matrix in the G dictionary.

        Args:
            * data          : Data concerned by the Green's function
            * G             : Green's function matrix

        Returns:
            * None
        '''

        # Check
        if not hasattr(self, 'G'):
            self.G = {}

        # Check
        if not data.name in self.G.keys():
            self.G[data.name] = {}

        # Set
        self.G[data.name]['custom'] = G

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def _edcmp_displacement_forward(self, obs_pts, slipVec,
                                   faultname='', dataname='', verbose=False,
                                   options=None):
        """Forward calculation of surface displacement using EDCMP for all patches.

        Parameters
        ----------
        obs_pts : np.ndarray
            Observation points, shape (N, 3), in fault/projected coordinates (meters).
        slipVec : np.ndarray
            Physical slip vector for each patch, shape (n_patch, 3) [ss, ds, ts].
        faultname, dataname : str
            Labels for file naming.
        verbose : bool
            Print progress information.
        options : EdcmpOptions or None
            EDCMP configuration.

        Returns
        -------
        disp_total : np.ndarray, shape (N, 3)
        """
        import numpy as np
        from concurrent.futures import ProcessPoolExecutor, as_completed

        opts = options if isinstance(options, EdcmpOptions) else EdcmpOptions()
        n_jobs = opts.n_jobs or 4

        N_obs = obs_pts.shape[0]
        N_patch = slipVec.shape[0]
        disp_total = np.zeros((N_obs, 3))

        if self.patchType == 'triangle' and not opts.allow_triangle:
            raise ValueError(
                "Triangle patches with method='edcmp' require edcmp_allow_triangle=True"
            )

        data = SimpleNamespace(
            x=obs_pts[:, 0],
            y=obs_pts[:, 1],
            name=dataname if dataname else 'surface',
        )

        patch_sources, _, _ = _get_edcmp_patch_sources(
            self,
            n_jobs,
            rect_dx_km=opts.triangle_rect_dx_km,
            rect_dy_km=opts.triangle_rect_dy_km,
        )
        resolved_engine = resolve_edcmp_engine(
            opts.engine,
            fallback_engines=opts.fallback_engines,
            module_dir=opts.module_dir,
        )

        if resolved_engine != 'exe' and not opts.layered_model:
            logger.warning(
                "EDCMP engine '%s' only supports layered Green's functions; falling back to exe backend.",
                resolved_engine,
            )
            resolved_engine = 'exe'

        bundle_sizes = [_edcmp_source_bundle_size(source) for source in patch_sources]
        total_rects = int(sum(bundle_sizes))
        if verbose:
            logger.info(
                "EDCMP-%s forward: %d patches, %d equivalent rectangles (data=%s)",
                resolved_engine, N_patch, total_rects, data.name,
            )

        data_lite = _lightweight_data(data)
        if resolved_engine == 'exe':
            patch_args = []
            for p in range(N_patch):
                patch_args.append((p, patch_sources[p], slipVec[p], data_lite, opts.grn_dir,
                                   opts.output_dir, opts.workdir, opts.layered_model,
                                   opts.force_recompute, faultname))

            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futures = [pool.submit(_single_patch_forward, arg) for arg in patch_args]
                progress_iter = as_completed(futures)
                if verbose:
                    progress_iter = tqdm(progress_iter, total=N_patch, desc="EDCMP-exe forward")
                for future in progress_iter:
                    disp_total += future.result()
        else:
            max_workers = int(n_jobs)
            if max_workers > 1:
                patch_args = [
                    (
                        p,
                        patch_sources[p],
                        slipVec[p],
                        data_lite,
                        opts.grn_dir,
                        opts.workdir,
                        resolved_engine,
                        opts.module_dir,
                    )
                    for p in range(N_patch)
                ]
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    progress_iter = as_completed([pool.submit(_single_patch_inmemory_forward, arg) for arg in patch_args])
                    if verbose:
                        progress_iter = tqdm(progress_iter, total=N_patch, desc=f"EDCMP-{resolved_engine} forward")
                    for future in progress_iter:
                        _, disp_patch = future.result()
                        disp_total += disp_patch
            else:
                for p in range(N_patch):
                    if verbose:
                        sys.stdout.write(
                            f"\r EDCMP-{resolved_engine} forward patch: {p+1} / {N_patch} "
                            f"(bundle={bundle_sizes[p]} rects) "
                        )
                        sys.stdout.flush()
                    disp_total += compute_inmemory_edcmp_forward(
                        data,
                        patch_sources[p],
                        slipVec[p],
                        engine=resolved_engine,
                        grn_dir=opts.grn_dir,
                        workdir=opts.workdir,
                        module_dir=opts.module_dir,
                    )
                if verbose:
                    sys.stdout.write('\n')
                    sys.stdout.flush()

        return disp_total
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def writePointSources2Pickle(self, filename):
        '''
        Writes the point sources to a pickle file.
        Always writes the Facet based point sources.

        Args:
            * filename      : Name of the pickle file.

        Returns:
            * None
        '''

        # Import
        try:
            import pickle
        except:
            logger.error('Needs the pickle module...')
            return

        # Assert
        assert hasattr(self, 'edksSources'), 'Need to compute sources'

        # Get the right source
        if len(self.edksSources)>7:
            edksSources = self.edksFacetSources
        else:
            edksSources = self.edksSources

        # Open file
        fout = open(filename, 'wb')

        # Save
        pickle.dump(edksSources, fout)

        # Close
        fout.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def readPointSourcesFromPickle(self, filename):
        '''
        Reads the point sources for computing Green's functions with EDKS
        from a pickle file. Sets the sources in self.edksSources

        Args:
            * filename      : Name of the pickle file

        Returns:
            * None
        '''

        # Import
        try:
            import pickle
        except:
            logger.error('Needs the pickle module...')
            return

        # Create lists, clean lists
        if hasattr(self, 'edksFacetSources'):
            del self.edksFacetSources

        # Read the whole file
        fin = open(filename, 'rb')
        sources = pickle.load(fin)
        fin.close()

        # Store
        self.edksSources = sources

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def edksGFs(self, data, vertical=True, slipdir='sd', verbose=True,
                      convergence=None):
        '''
        Builds the Green's functions based on the solution by Zhao & Rivera 2002.
        The corresponding functions are in the EDKS code that needs to be installed and
        the executables should be found in the directory set by the environment
        variable EDKS_BIN.

        A few variables need to be set in before running this method

            Required:
                - self.kernelsEDKS    : Filename of the EDKS kernels.

            One of the Three:
                - self.sourceSpacing  : Spacing between the sources in each patch.
                - self.sourceNumber   : Number of sources per patches.
                - self.sourceArea     : Maximum Area of the sources.

        Args:
            * data              : Data object

        Kwargs:
            * vertical      : If True, will produce green's functions for the vertical displacements in a gps object.
            * slipdir       : Direction of slip along the patches. Can be any combination of s (strikeslip), d (dipslip), t (tensile) and c (coupling)
            * verbose       : Writes stuff to the screen (overwrites self.verbose)
            * convergence   : If coupling case, needs convergence azimuth and rate [azimuth in deg, rate]

        Returns:
            * G             : Dictionary of the built Green's functions
        '''

        # Dispatch for cross-fault offset data
        if data.dtype == 'crossfaultoffset':
            return self._build_gfs_crossfaultoffset_via_gps_concat(
                self.edksGFs, data, slipdir=slipdir,
                vertical=True, verbose=verbose, convergence=convergence)

        # Print
        if verbose:
            logger.info('---------------------------------')
            logger.info("Building Green's functions for the data set")
            logger.info("{} of type {} using EDKS on fault {}".format(data.name, data.dtype, self.name))

        # Check if we can find kernels
        if not hasattr(self, 'kernelsEDKS'):
            if verbose:
                logger.warning('---------------------------------')
                logger.warning(' WARNING: Kernels for computation of')
                logger.warning(' stratified Greens functions not set in {}.kernelsEDKS'.format(self.name))
                logger.warning(' Looking for default kernels')
                logger.warning('---------------------------------')
            self.kernelsEDKS = 'kernels.edks'
        stratKernels = self.kernelsEDKS
        assert os.path.isfile(stratKernels), 'Kernels for EDKS not found...'

        # Show me
        if verbose:
            logger.info('Kernels used: {}'.format(stratKernels))

        # Check if we can find mention of the spacing between points
        if not hasattr(self, 'sourceSpacing') and not hasattr(self, 'sourceNumber')\
                and not hasattr(self, 'sourceArea'):
            logger.error('---------------------------------')
            logger.error(' ERROR: Cannot find sourceSpacing nor')
            logger.error(' sourceNumber nor sourceArea for stratified')
            logger.error(' Greens function computation. Dying here...')
            logger.error('---------------------------------')
            sys.exit(1)

        # Receivers to meters
        xr = data.x * 1000.
        yr = data.y * 1000.

        # Prefix for the files
        prefix = '{}_{}'.format(self.name.replace(' ','-'), data.name.replace(' ','-'))

        # Check
        if convergence is not None and 'c' not in slipdir:
            slipdir += 'c'
        if 'c' in slipdir:
            assert convergence is not None, 'No convergence azimuth and rate given'
            if 's' not in slipdir:
                slipdir += 's'
            if 'd' not in slipdir:
                slipdir += 'd'

        # Check something
        if not hasattr(self, 'keepTrackOfSources'):
            if self.patchType == 'triangletent':
                self.keepTrackOfSources = True
            else:
                self.keepTrackOfSources = False

        # If we have already done that step
        if self.keepTrackOfSources and hasattr(self, 'edksSources'):
            if verbose:
                logger.info('Get sources from saved sources')
            Ids, xs, ys, zs, strike, dip, Areas = self.edksSources[:7]
        # Else, drop sources in the patches
        else:
            if verbose:
                logger.info('Subdividing patches into point sources')
            Ids, xs, ys, zs, strike, dip, Areas = Patches2Sources(self, verbose=verbose)
            # All these guys need to be in meters
            xs *= 1000.
            ys *= 1000.
            zs *= 1000.
            Areas *= 1e6
            # Strike and dip in degrees
            strike = strike*180./np.pi
            dip = dip*180./np.pi
            # Keep track?
            self.edksSources = [Ids, xs, ys, zs, strike, dip, Areas]

        # Get the slip vector
        if self.patchType in ('triangle', 'rectangle'):
            slip = np.ones(dip.shape)
        if self.patchType == 'triangletent':
            # If saved, good
            if self.keepTrackOfSources and hasattr(self, 'edksSources') and (len(self.edksSources)>7):
                slip = self.edksSources[7]
            # Else, we have to re-organize the Ids from facet to nodes
            else:
                if hasattr(self, 'homogeneousStrike'):
                    homS = self.homogeneousStrike
                else:
                    homS = False
                if hasattr(self, 'homogeneousDip'):
                    homD = self.homogeneousDip
                else:
                    homD = False
                self.Facet2Nodes(homogeneousStrike=homS, homogeneousDip=homD)#, keepFacetsSeparated=TentCouplingCase)
                Ids, xs, ys, zs, strike, dip, Areas, slip = self.edksSources

        # Informations
        if verbose:
            logger.info('{} sources for {} patches and {} data points'.format(len(Ids), len(self.patch), len(xr)))

        # Run EDKS Strike slip
        if 's' in slipdir:
            if verbose:
                logger.info('Running Strike Slip component for data set {}'.format(data.name))
            iGss = np.array(sum_layered(xs, ys, zs,
                                        strike, dip, np.zeros(dip.shape), slip,
                                        np.sqrt(Areas), np.sqrt(Areas), 1, 1,
                                        xr, yr, stratKernels, prefix, BIN_EDKS='EDKS_BIN',
                                        cleanUp=self.cleanUp, verbose=verbose))
            if verbose:
                logger.info('Summing sub-sources...')
            Gss = np.zeros((3, iGss.shape[1],np.unique(Ids).shape[0]))
            for Id in np.unique(Ids):
                Gss[:,:,Id] = np.sum(iGss[:,:,np.flatnonzero(Ids==Id)], axis=2)
            del iGss
        else:
            Gss = np.zeros((3, len(data.x), len(self.patch)))

        # Run EDKS dip slip
        if 'd' in slipdir:
            if verbose:
                logger.info('Running Dip Slip component for data set {}'.format(data.name))
            iGds = np.array(sum_layered(xs, ys, zs,
                                        strike, dip, np.ones(dip.shape)*90.0, slip,
                                        np.sqrt(Areas), np.sqrt(Areas), 1, 1,
                                        xr, yr, stratKernels, prefix, BIN_EDKS='EDKS_BIN',
                                        cleanUp=self.cleanUp, verbose=verbose))
            if verbose:
                logger.info('Summing sub-sources...')
            Gds = np.zeros((3, iGds.shape[1], np.unique(Ids).shape[0]))
            for Id in np.unique(Ids):
                Gds[:,:,Id] = np.sum(iGds[:,:,np.flatnonzero(Ids==Id)], axis=2)
            del iGds
        else:
            Gds = np.zeros((3, len(data.x), len(self.patch)))

        # Run EDKS Tensile?
        if 't' in slipdir:
            assert False, 'Sorry, this is not working so far... Bryan should get it done soon...'
            if verbose:
                logger.info('Running tensile component for data set {}'.format(data.name))
            iGts = np.array(sum_layered(xs, ys, zs,
                                        strike, dip, np.zeros(dip.shape), slip,
                                        np.sqrt(Areas), np.sqrt(Areas), 1, 1,
                                        xr, yr, stratKernels, prefix,
                                        BIN_EDKS='EDKS_BIN', tensile=True, verbose=verbose))
            if verbose:
                logger.info('Summing sub-sources...')
            Gts = np.zeros((3, iGts.shape[1], np.unique(Ids).shape[0]))
            for Id in np.unique(Ids):
                Gts[:, :,Id] = np.sum(iGts[:,:,np.flatnonzero(Ids==Id)], axis=2)
            del iGts
        else:
            Gts = np.zeros((3, len(data.x), len(self.patch)))

        # Ordering
        G = self._buildGFsdict(data, Gss, Gds, Gts, slipdir=slipdir,
                               convergence=convergence, vertical=vertical)

        # All done
        return G
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def pscmpGFs(self, data, vertical=True, slipdir='sd', verbose=True, convergence=None,
                 options=None):
        """Compute PSCMP Green's functions for all patches in parallel.

        Parameters
        ----------
        data : object
            Observation data, must have .lat and .lon attributes (array-like).
        vertical : bool
            If True, include vertical displacements for GPS.
        slipdir : str
            Slip directions, any combination of 's' (strike), 'd' (dip), 't' (tensile), 'c' (coupling).
        verbose : bool
            Print progress information.
        convergence : list or None
            Convergence azimuth and rate for coupling GFs.
        options : PscmpOptions or None
            PSCMP configuration. Use PscmpOptions.describe_options() to see fields.

        Returns
        -------
        G : dict
            Dictionary of Green's functions.
        """
        opts = options if isinstance(options, PscmpOptions) else PscmpOptions()

        if data.dtype == 'crossfaultoffset':
            return self._build_gfs_crossfaultoffset_via_gps_concat(
                self.pscmpGFs, data, slipdir=slipdir,
                vertical=True, verbose=verbose, convergence=convergence,
                options=opts)

        import numpy as np
        import os
        import glob
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n_jobs = opts.n_jobs if opts.n_jobs is not None else max(os.cpu_count() // 2, 4)
        workdir = opts.workdir
        psgrn_dir = opts.grn_dir
        out_dir = opts.output_dir
        cleanup_inp = opts.cleanup_inp
        force_recompute = opts.force_recompute

        SLP = []
        if 's' in slipdir:
            SLP.append(1.0)
        else:
            SLP.append(0.0)
        if 'd' in slipdir:
            SLP.append(1.0)
        else:
            SLP.append(0.0)
        if convergence is not None:
            SLP = [1.0, 1.0]
            if 'c' not in slipdir:
                slipdir += 'c'
        if 'c' in slipdir:
            assert convergence is not None, 'No convergence azimuth and rate given'
        if 't' in slipdir:
            SLP.append(1.0)
        else:
            SLP.append(0.0)

        Nd = len(data.x)
        Np = len(self.patch)
        Gss = np.zeros((3, Nd, Np))
        Gds = np.zeros((3, Nd, Np))
        Gts = np.zeros((3, Nd, Np))

        mean_x_km = np.mean([np.mean([p[i][0] for i in range(len(p))]) for p in self.patch])
        mean_y_km = np.mean([np.mean([p[i][1] for i in range(len(p))]) for p in self.patch])

        patch_args = []
        for p in range(Np):
            cx_km, cy_km, depth_km, width_km, length_km, strike_rad, dip_rad = self.getpatchgeometry(p, center=True)
            patch_args.append((self, p, SLP, data, workdir, psgrn_dir, out_dir, verbose, Np,
                            self.patch[p], self.patchll[p], self.patchType, self.name, force_recompute,
                            [cx_km, cy_km, depth_km, width_km, length_km, strike_rad, dip_rad]))

        results = [None] * Np
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_pscmp_patch_task, arg) for arg in patch_args]
            for i, future in enumerate(tqdm(as_completed(futures), total=Np)):
                p, ss, ds, ts = future.result()
                Gss[:, :, p] = ss.T
                Gds[:, :, p] = ds.T
                Gts[:, :, p] = ts.T
                results[p] = (ss, ds, ts)

        if verbose:
            logger.info('')

        if cleanup_inp:
            inp_files = glob.glob(os.path.join(workdir, 'pscmp*.inp'))
            for f in inp_files:
                if 'p1.inp' in f:
                    continue
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning("Failed to remove %s: %s", f, e)

        G = self._buildGFsdict(data, Gss, Gds, Gts, slipdir=slipdir,
                               convergence=convergence, vertical=vertical)
        return G
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def edcmpGFs(self, data, vertical=True, slipdir='sd', verbose=True,
                convergence=None, options=None):
        """Compute EDCMP Green's functions for all patches in parallel.

        Parameters
        ----------
        data : object
            Observation data, must have .x and .y attributes (array-like, meters).
        vertical : bool
            If True, include vertical displacements.
        slipdir : str
            Slip directions, any combination of 's' (strike), 'd' (dip), 't' (tensile).
        verbose : bool
            Print progress information.
        convergence : list or None
            Convergence azimuth and rate for coupling GFs.
        options : EdcmpOptions or None
            EDCMP configuration. Use EdcmpOptions.describe_options() to see fields.

        Returns
        -------
        G : dict
            Dictionary of Green's functions.
        """
        opts = options if isinstance(options, EdcmpOptions) else EdcmpOptions()

        if data.dtype == 'crossfaultoffset':
            return self._build_gfs_crossfaultoffset_via_gps_concat(
                self.edcmpGFs, data, slipdir=slipdir,
                vertical=True, verbose=verbose,
                options=opts)

        import numpy as np
        import os
        import glob
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if self.patchType == 'triangle' and not opts.allow_triangle:
            raise ValueError(
                "Triangle patches with method='edcmp' require edcmp_allow_triangle=True"
            )

        SLP = [1.0 if c in slipdir else 0.0 for c in 'sdt']

        Nd = len(data.x)
        Np = len(self.patch)
        Gss = np.zeros((3, Nd, Np))
        Gds = np.zeros((3, Nd, Np))
        Gts = np.zeros((3, Nd, Np))

        if opts.n_jobs is None:
            n_jobs = max(os.cpu_count() // 2, 4)
            if verbose:
                logger.info("n_jobs=None, auto-detected %d workers", n_jobs)
        else:
            n_jobs = int(opts.n_jobs)

        resolved_engine = resolve_edcmp_engine(
            opts.engine,
            fallback_engines=opts.fallback_engines,
            module_dir=opts.module_dir,
        )
        if resolved_engine != 'exe' and not opts.layered_model:
            logger.warning(
                "EDCMP engine '%s' only supports layered Green's functions; falling back to exe backend.",
                resolved_engine,
            )
            resolved_engine = 'exe'
        if verbose:
            logger.info("EDCMP backend engine: %s", resolved_engine)

        patch_sources, mean_x_km, mean_y_km = _get_edcmp_patch_sources(
            self,
            n_jobs,
            rect_dx_km=opts.triangle_rect_dx_km,
            rect_dy_km=opts.triangle_rect_dy_km,
        )
        bundle_sizes = [_edcmp_source_bundle_size(source) for source in patch_sources]
        total_rects = int(sum(bundle_sizes))
        if verbose:
            logger.info(
                "EDCMP-%s: %d patches, %d equivalent rectangles",
                resolved_engine, Np, total_rects,
            )

        data_lite = _lightweight_data(data)
        if resolved_engine == 'exe':
            self._edcmpGFs_exe(
                data_lite, SLP, Np, Gss, Gds, Gts,
                patch_sources, mean_x_km, mean_y_km,
                opts.workdir, opts.grn_dir, opts.output_dir, opts.layered_model,
                n_jobs, opts.force_recompute, verbose,
            )
        else:
            self._edcmpGFs_inmemory(
                data, data_lite, SLP, Np, Gss, Gds, Gts,
                patch_sources, bundle_sizes,
                resolved_engine, opts.module_dir,
                opts.grn_dir, opts.workdir, n_jobs, verbose,
            )

        if verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        if opts.cleanup_inp and resolved_engine == 'exe':
            inp_files = glob.glob(os.path.join(opts.workdir, 'edcmp*.inp'))
            for f in inp_files:
                if 'p1.inp' in f:
                    continue
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning("Failed to remove %s: %s", f, e)

        G = self._buildGFsdict(data, Gss, Gds, Gts, slipdir=slipdir,
                            convergence=convergence, vertical=vertical)
        return G

    def _edcmpGFs_exe(self, data_lite, SLP, Np, Gss, Gds, Gts,
                      patch_sources, mean_x_km, mean_y_km,
                      workdir, grn_dir, output_dir, layered_model,
                      n_jobs, force_recompute, verbose):
        """Run EDCMP Green's function computation via exe subprocess."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        patch_args = []
        for p in range(Np):
            patch_args.append((p, SLP, data_lite, workdir, grn_dir, output_dir, layered_model, verbose, Np,
                            mean_x_km, mean_y_km, self.patch[p], self.patchType, self.name, force_recompute,
                            patch_sources[p]))

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_edcmp_patch_task, arg) for arg in patch_args]
            for future in tqdm(as_completed(futures), total=Np, desc="EDCMP-exe"):
                p, ss, ds, ts = future.result()
                Gss[:, :, p] = ss.T
                Gds[:, :, p] = ds.T
                Gts[:, :, p] = ts.T

    def _edcmpGFs_inmemory(self, data, data_lite, SLP, Np, Gss, Gds, Gts,
                           patch_sources, bundle_sizes,
                           resolved_engine, edcmp_module_dir,
                           grn_dir, workdir, n_jobs, verbose):
        """Run EDCMP Green's function computation via in-memory ctypes backend."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if n_jobs > 1:
            from csi.edgrn_edcmp.shared_memory_backend import create_shared_greens
            from csi.edgrn_edcmp.edcmp_backends import _init_shared_memory_worker

            if verbose:
                logger.info("Creating shared memory for Green's functions...")

            shared_grn, shm_metadata = create_shared_greens(
                engine=resolved_engine,
                grn_dir=grn_dir,
                workdir=workdir,
                module_dir=edcmp_module_dir
            )

            if verbose:
                total_bytes = sum(np.prod(arr['shape']) * 8 for arr in shm_metadata['arrays'].values())
                logger.info(f"Shared memory created, size: {total_bytes / 1024 / 1024:.1f} MB")

            try:
                patch_args = [
                    (
                        p,
                        patch_sources[p],
                        SLP,
                        data_lite,
                        grn_dir,
                        workdir,
                        resolved_engine,
                        edcmp_module_dir,
                    )
                    for p in range(Np)
                ]
                with ProcessPoolExecutor(
                    max_workers=n_jobs,
                    initializer=_init_shared_memory_worker,
                    initargs=(shm_metadata,),
                ) as pool:
                    progress_iter = as_completed([pool.submit(_single_patch_inmemory_greens, arg) for arg in patch_args])
                    if verbose:
                        progress_iter = tqdm(progress_iter, total=Np, desc=f"EDCMP-{resolved_engine}")
                    for future in progress_iter:
                        p, ss, ds, ts = future.result()
                        Gss[:, :, p] = ss.T
                        Gds[:, :, p] = ds.T
                        Gts[:, :, p] = ts.T
            finally:
                shared_grn.cleanup()
                if verbose:
                    logger.info("Shared memory cleaned up")
        else:
            for p in range(Np):
                if verbose:
                    sys.stdout.write(
                        f"\r EDCMP-{resolved_engine} patch: {p+1} / {Np} "
                        f"(bundle={bundle_sizes[p]} rects) "
                    )
                    sys.stdout.flush()
                ss, ds, ts = compute_inmemory_edcmp_greens(
                    data,
                    patch_sources[p],
                    slip=SLP,
                    engine=resolved_engine,
                    grn_dir=grn_dir,
                    workdir=workdir,
                    module_dir=edcmp_module_dir,
                )
                Gss[:, :, p] = ss.T
                Gds[:, :, p] = ds.T
                Gts[:, :, p] = ts.T
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def cutdeGFs(self, data, vertical=True, slipdir='sd', verbose=True, convergence=None, nu=0.25):
        """
        Builds the Green's functions based on the solution by cutde.
        The corresponding functions are in the cutde code that needs to be installed

        Note:
        The cutde code uses a different coordinate system with z-axis pointing upward instead of downward.
        The vertex order of each patch need to be counter-clockwise when viewed from above.
        """
        if data.dtype == 'crossfaultoffset':
            return self.cutdeGFs_crossfaultoffset(data, slipdir=slipdir,
                                                  verbose=verbose,
                                                  convergence=convergence,
                                                  nu=nu)

        from cutde.halfspace import disp_matrix

        # Print
        if verbose:
            logger.info('---------------------------------')
            logger.info("Building Green's functions for the data set ")
            logger.info("{} of type {} in a homogeneous elastic half-space by a parallel code cutde".format(data.name,
                                                                    data.dtype))

        # Create the dictionary
        G = {'strikeslip':[], 'dipslip':[], 'tensile':[], 'coupling':[]}

        # Create the matrices to hold the whole thing
        Gss, Gds, Gts = [np.zeros((3, len(data.x), len(self.patch))) for _ in range(3)]
        
        # calculate the Green's functions using cutde.halfspace.disp_matrix
        ## get the observation points
        obs_pts = np.array([data.x, data.y, np.zeros(data.x.shape)]).T
        ## get the source points
        src_tris = self.Vertices[self.Faces, :]
        # cutde uses a different coordinate system with z-axis pointing upward instead of downward
        src_tris[:, :, -1] *= -1

        # transfer obs_pts from F-order to C-order
        obs_pts = np.ascontiguousarray(obs_pts)

        disp_mat = disp_matrix(obs_pts, src_tris, nu)

        # disp_mat has shape (Nd, 3_component, Np, 3_slip).
        # disp_mat[:, :, :, i] has shape (Nd, 3_component, Np).
        # Transpose with (1, 0, 2) to get (3_component, Nd, Np) which is the
        # standard GF shape expected by _buildGFsdict (axis-0 = E/N/U).
        # Gss.shape = (3, Nd, Np), where Nd is number of data points, Np is number of patches
        Gss, Gds, Gts = [np.transpose(disp_mat[:, :, :, i], (1, 0, 2)) for i in range(3)]

        if verbose:
            logger.info(' ')

        # Build the dictionary
        G = self._buildGFsdict(data, Gss, Gds, Gts, slipdir=slipdir,
                                convergence=convergence, vertical=vertical)

        # All done
        return G
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def cutdeGFsinGPS(self, data, vertical=True, slipdir='sd', verbose=True, convergence=None, nu=0.25):
        """
        Builds the Green's functions based on the solution by cutde.
        The corresponding functions are in the cutde code that needs to be installed

        Note:
        The cutde code uses a different coordinate system with z-axis pointing upward instead of downward.
        The vertex order of each patch need to be counter-clockwise when viewed from above.
        """
        from cutde.halfspace import disp_matrix

        # Print
        if verbose:
            logger.info('---------------------------------')
            logger.info("Building Green's functions for the data set ")
            logger.info("{} of type {} in a homogeneous elastic half-space by a parallel code cutde".format(data.name,
                                                                    data.dtype))

        # Create the dictionary
        G = {'strikeslip':[], 'dipslip':[], 'tensile':[], 'coupling':[]}

        # Create the matrices to hold the whole thing
        Gss, Gds, Gts = [np.zeros((3, len(data.x), len(self.patch))) for _ in range(3)]
        
        # calculate the Green's functions using cutde.halfspace.disp_matrix
        ## get the observation points
        obs_pts = np.array([data.x, data.y, np.zeros(data.x.shape)]).T
        ## get the source points
        src_tris = self.Vertices[self.Faces, :]
        # cutde uses a different coordinate system with z-axis pointing upward instead of downward
        src_tris[:, :, -1] *= -1

        # transfer obs_pts from F-order to C-order
        obs_pts = np.ascontiguousarray(obs_pts)

        disp_mat = disp_matrix(obs_pts, src_tris, nu)

        # disp_mat has shape (Nd, 3_component, Np, 3_slip).
        # disp_mat[:, :, :, i] has shape (Nd, 3_component, Np).
        # Transpose with (1, 0, 2) to get (3_component, Nd, Np) — standard GF shape.
        Gss, Gds, Gts = [np.transpose(disp_mat[:, :, :, i], (1, 0, 2)) for i in range(3)]

        if verbose:
            logger.info(' ')

        # # Build the dictionary
        # G = self._buildGFsdict(data, Gss, Gds, Gts, slipdir=slipdir,
        #                         convergence=convergence, vertical=vertical)
        
        # All done
        return Gss, Gds, Gts
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def cutdeGFs_crossfaultoffset(self, data, slipdir='sd', verbose=True, 
                                   convergence=None, nu=0.25):
        """
        Builds the Green's functions for cross-fault offset data using cutde.

        For each point pair, computes the displacement difference (side2 - side1)
        and projects it onto fault-parallel and fault-perpendicular directions.

        Args:
            * data      : crossfaultoffset data object

        Kwargs:
            * slipdir       : Direction of slip ('s', 'd', 't', 'c')
            * verbose       : Talk to me
            * convergence   : Convergence azimuth and rate [azimuth in deg, rate]
            * nu            : Poisson's ratio (default 0.25)

        Returns:
            * G             : Dictionary of GFs
        """
        from cutde.halfspace import disp_matrix

        assert data.dtype == 'crossfaultoffset', \
            'Only works for crossfaultoffset data type: {}'.format(data.dtype)

        if verbose:
            logger.info('---------------------------------')
            logger.info("Building Green's functions for cross-fault offset data set ")
            logger.info("{} using cutde".format(data.name))

        # Source triangles
        src_tris = self.Vertices[self.Faces, :]
        src_tris[:, :, -1] *= -1  # cutde uses z-up

        n_pairs = len(data.station)
        n_patches = len(self.patch)

        # Observation points for side 1
        obs1 = np.column_stack([data.x1, data.y1, np.zeros(n_pairs)])
        obs1 = np.ascontiguousarray(obs1)

        # Observation points for side 2
        obs2 = np.column_stack([data.x2, data.y2, np.zeros(n_pairs)])
        obs2 = np.ascontiguousarray(obs2)

        # Compute displacement matrices
        # disp_mat shape: (n_obs, 3_disp, n_patches, 3_slip)
        disp_mat1 = disp_matrix(obs1, src_tris, nu)
        disp_mat2 = disp_matrix(obs2, src_tris, nu)

        # Differential displacement: side2 - side1
        diff_disp = disp_mat2 - disp_mat1  # (n_pairs, 3_disp, n_patches, 3_slip)

        # Convert to (3, n_pairs, n_patches) for shared projection helper
        # diff_disp[:,:,:,i] shape: (n_pairs, 3_E/N/U, n_patches)
        dGss = np.transpose(diff_disp[:, :, :, 0], (1, 0, 2))
        dGds = np.transpose(diff_disp[:, :, :, 1], (1, 0, 2))
        dGts = np.transpose(diff_disp[:, :, :, 2], (1, 0, 2))

        if verbose:
            logger.info('Done building GFs for {}'.format(data.name))

        return self._project_crossfaultoffset_from_diff_GFs(
            data, dGss, dGds, dGts, slipdir, convergence)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def _project_crossfaultoffset_from_diff_GFs(self, data, dGss, dGds, dGts,
                                                 slipdir, convergence):
        """
        Shared helper: project differential GF tensors (side2 - side1) onto
        crossfaultoffset observation space.

        Args:
            * data          : crossfaultoffset data object
            * dGss/dGds/dGts: shape (3, n_pairs, n_patches)
                              axis-0: [E, N, U] displacement components
                              differential = side2 - side1
            * slipdir       : slip direction string ('s','d','t','c')
            * convergence   : coupling params [azimuth_deg, rate] or None

        Returns:
            * G             : dictionary of projected GF matrices
        """
        n_pairs = len(data.station)

        # Build projection rules: parallel → perpendicular → vertical
        components = []
        if data.fault_parallel is not None:
            fp_e = -np.sin(data.strike)  # (-sin s, cos s) in (E, N)
            fp_n =  np.cos(data.strike)
            components.append(('horizontal', fp_e, fp_n))
        if data.fault_perpendicular is not None:
            fn_e =  np.cos(data.strike)  # (cos s, sin s) in (E, N)
            fn_n =  np.sin(data.strike)
            components.append(('horizontal', fn_e, fn_n))
        if data.fault_vertical is not None:
            components.append(('vertical', None, None))

        G = {'strikeslip': None, 'dipslip': None, 'tensile': None, 'coupling': None}

        for slip_char, slip_key, dG in zip('sdt',
                                            ['strikeslip', 'dipslip', 'tensile'],
                                            [dGss, dGds, dGts]):
            if slip_char not in slipdir:
                continue
            rows = []
            for comp_type, ve, vn in components:
                if comp_type == 'horizontal':
                    proj = ve[:, None] * dG[0] + vn[:, None] * dG[1]  # (n_pairs, n_patches)
                else:
                    proj = dG[2]  # U differential
                rows.append(proj)
            G[slip_key] = np.vstack(rows)  # (n_comp * n_pairs, n_patches)

        if 'c' in slipdir:
            assert convergence is not None, 'No convergence azimuth and rate given'
            azimuth, rate = convergence
            azimuth_rad = azimuth * np.pi / 180.
            strike_arr = self.getStrikes()
            dip_arr    = self.getDips()
            rotation = np.arctan2(
                np.tan(strike_arr) - np.tan(azimuth_rad),
                np.cos(dip_arr) * (1. + np.tan(azimuth_rad) * np.tan(strike_arr)))
            if azimuth > 90. and azimuth <= 270.:
                rotation += np.pi
            rows = []
            for comp_type, ve, vn in components:
                if comp_type == 'horizontal':
                    proj_ss = ve[:, None] * dGss[0] + vn[:, None] * dGss[1]
                    proj_ds = ve[:, None] * dGds[0] + vn[:, None] * dGds[1]
                else:
                    proj_ss = dGss[2]
                    proj_ds = dGds[2]
                proj_c = (proj_ss * np.cos(rotation)[None, :] +
                          proj_ds * np.sin(rotation)[None, :]) * rate
                rows.append(proj_c)
            G['coupling'] = np.vstack(rows)

        return G
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def _build_gfs_crossfaultoffset_via_gps_concat(self, method, data, **method_kwargs):
        """
        Generic crossfaultoffset GF builder via GPS-format concatenation.

        Concatenates side1 and side2 observation points into a mock GPS data object,
        calls the given GF method to obtain a 6N-row GPS-format G dict, then
        extracts E/N/U differentials (side2 - side1) and projects onto
        fault-parallel / fault-perpendicular / vertical directions.

        This decouples the crossfaultoffset projection logic from each backend
        (homogeneous, EDKS, PSCMP, EDCMP), letting them share one implementation.

        Args:
            * method        : bound GF method, e.g. self.homogeneousGFs
            * data          : crossfaultoffset data object
            * **method_kwargs : kwargs forwarded to method (slipdir, verbose, etc.)

        Returns:
            * G             : crossfaultoffset-projected GF dict
        """
        n = len(data.station)
        slipdir    = method_kwargs.pop('slipdir', 'sd')
        convergence = method_kwargs.pop('convergence', None)

        # Ensure ss and ds are computed when coupling is requested
        raw_slipdir = slipdir
        if 'c' in slipdir:
            if 's' not in slipdir:
                slipdir = slipdir + 's'
            if 'd' not in slipdir:
                slipdir = slipdir + 'd'

        # backend_slipdir: strip 'c' — coupling is computed by the projection helper,
        # not by the GPS-format backend method
        backend_slipdir = ''.join(c for c in slipdir if c != 'c')

        # Create a GPS-like mock with 2N observation points (side1 || side2)
        class _GPSMock:
            dtype = 'gps'
        mock = _GPSMock()
        mock.x   = np.concatenate([data.x1,   data.x2])
        mock.y   = np.concatenate([data.y1,   data.y2])
        mock.lon = np.concatenate([data.lon1, data.lon2])
        mock.lat = np.concatenate([data.lat1, data.lat2])
        mock.name = data.name

        # Call the backend method with the mock (dtype='gps' avoids re-dispatch).
        # We do NOT pass convergence here: coupling is handled entirely by
        # _project_crossfaultoffset_from_diff_GFs using dGss and dGds.
        G_gps = method(mock, slipdir=backend_slipdir, **method_kwargs)

        # G_gps format for GPS vertical=True: shape (6N, Np) per component
        # Row layout: [E_0..E_{2N-1}, N_0..N_{2N-1}, U_0..U_{2N-1}]
        # obs 0..N-1 = side1,  obs N..2N-1 = side2
        Np = len(self.patch)
        dGss = np.zeros((3, n, Np))
        dGds = np.zeros((3, n, Np))
        dGts = np.zeros((3, n, Np))

        for slip_char, dG, key in zip('sdt', [dGss, dGds, dGts],
                                       ['strikeslip', 'dipslip', 'tensile']):
            if slip_char not in slipdir or G_gps.get(key) is None:
                continue
            M = G_gps[key]  # (6N, Np)
            dG[0] = M[n:2*n,     :] - M[0:n,     :]  # dE
            dG[1] = M[3*n:4*n,   :] - M[2*n:3*n, :]  # dN
            dG[2] = M[5*n:6*n,   :] - M[4*n:5*n, :]  # dU

        return self._project_crossfaultoffset_from_diff_GFs(
            data, dGss, dGds, dGts, raw_slipdir, convergence)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setGFsFromFile(self, data, strikeslip=None, dipslip=None,
                                   tensile=None, coupling=None,
                                   custom=None, vertical=False, dtype='d'):
        '''
        Sets the Green's functions reading binary files. Be carefull, these have to be in the
        good format (i.e. if it is GPS, then GF are E, then N, then U, optional, and
        if insar, GF are projected already). Basically, it will work better if
        you have computed the GFs using csi...

        Args:
            * data          : Data object

        kwargs:
            * strikeslip    : File containing the Green's functions for strikeslip related displacements.
            * dipslip       : File containing the Green's functions for dipslip related displacements.
            * tensile       : File containing the Green's functions for tensile related displacements.
            * coupling      : File containing the Green's functions for coupling related displacements.
            * vertical      : Deal with the UP component (gps: default is false, insar: it will be true anyway).
            * dtype         : Type of binary data. 'd' for double/float64. 'f' for float32

        Returns:
            * None
        '''

        if self.verbose:
            logger.info('---------------------------------')
            logger.info("Set up Green's functions for fault {}".format(self.name))
            logger.info("and data {} from files: ".format(data.name))
            logger.info("     strike slip: {}".format(strikeslip))
            logger.info("     dip slip:    {}".format(dipslip))
            logger.info("     tensile:     {}".format(tensile))
            logger.info("     coupling:    {}".format(coupling))

        # Get the number of patches
        if self.N_slip is None:
            self.N_slip = self.slip.shape[0]

        # Read the files and reshape the GFs
        Gss = None; Gds = None; Gts = None; Gcp = None
        if strikeslip is not None:
            Gss = np.fromfile(strikeslip, dtype=dtype)
            ndl = int(Gss.shape[0]/self.N_slip)
            Gss = Gss.reshape((ndl, self.N_slip))
        if dipslip is not None:
            Gds = np.fromfile(dipslip, dtype=dtype)
            ndl = int(Gds.shape[0]/self.N_slip)
            Gds = Gds.reshape((ndl, self.N_slip))
        if tensile is not None:
            Gts = np.fromfile(tensile, dtype=dtype)
            ndl = int(Gts.shape[0]/self.N_slip)
            Gts = Gts.reshape((ndl, self.N_slip))
        if coupling is not None:
            Gcp = np.fromfile(coupling, dtype=dtype)
            ndl = int(Gcp.shape[0]/self.N_slip)
            Gcp = Gcp.reshape((ndl, self.N_slip))

        # Create the big dictionary
        G = {'strikeslip': Gss,
             'dipslip': Gds,
             'tensile': Gts,
             'coupling': Gcp}

        # The dataset sets the Green's functions itself
        data.setGFsInFault(self, G, vertical=vertical)

        # If custom
        if custom is not None:
            self.setCustomGFs(data, custom)

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setGFs(self, data, strikeslip=[None, None, None],
                           dipslip=[None, None, None],
                           tensile=[None, None, None],
                           coupling=[None, None, None],
                           vertical=False, synthetic=False):
        '''
        Stores the input Green's functions matrices into the fault structure.

        These GFs are organized in a dictionary structure in self.G
        Entries of self.G are the data set names (data.name). Entries of self.G[data.name] are 'strikeslip', 'dipslip', 'tensile' and/or 'coupling'

        If you provide GPS GFs, those are organised with E, N and U in lines

        If you provide Optical GFs, those are organised with E and N in lines

        If you provide InSAR GFs, these need to be projected onto the
        LOS direction already.

        Args:
            * data          : Data structure

        Kwargs:
            * strikeslip    : List of matrices of the Strikeslip Green's functions
            * dipslip       : List of matrices of the dipslip Green's functions
            * tensile       : List of matrices of the tensile Green's functions
            * coupling      : List of matrices of the coupling Green's function

        Returns:
            * None
        '''

        # Get the number of data per point
        data_types = ['insar', 'tsunami', 'gps', 'multigps', 'opticorr', 'crossfaultoffset', 'leveling']
        obs_per_station = [1, 1, 0, 0, 2, 0, 1]
        if data.dtype == 'crossfaultoffset':
            # obs_per_station is dynamic (property) for crossfaultoffset
            pass
        elif data.dtype == 'leveling':
            # leveling obs_per_station is always 1, already set in class
            pass
        else:
            data.obs_per_station = obs_per_station[data_types.index(data.dtype)]

        # Check components
        if data.dtype in ('gps', 'multigps'):
            data.obs_per_station += sum(~np.isnan(data.vel_enu[:, :2]).any(axis=0))
            if vertical:
                if np.isnan(data.vel_enu[:,2]).any():
                    msg = 'Vertical can only be true if all stations have vertical components'
                    logger.error(msg)
                    raise ValueError(msg)
                data.obs_per_station += 1
        elif data.dtype == 'opticorr':
            if vertical:
                data.obs_per_station += 1

        # Create the storage for that dataset
        self.G.setdefault(data.name, {})
        G = self.G[data.name]

        # Initializes the data vector
        if not synthetic:
            if data.dtype in ('insar', 'tsunami'):
                self.d[data.name] = data.vel if data.dtype == 'insar' else data.d
                vertical = True
            elif data.dtype in ('gps', 'multigps'):
                self.d[data.name] = data.vel_enu[:, :data.obs_per_station].T.flatten()
                self.d[data.name] = self.d[data.name][np.isfinite(self.d[data.name])]
            elif data.dtype == 'opticorr':
                self.d[data.name] = np.hstack((data.east.T.flatten(), data.north.T.flatten()))
                if vertical:
                    self.d[data.name] = np.hstack((self.d[data.name], np.zeros_like(data.east.T.ravel())))
            elif data.dtype == 'crossfaultoffset':
                self.d[data.name] = data.data_vector
            elif data.dtype == 'leveling':
                self.d[data.name] = data.vel

        for gf_type, gf_values in zip(['strikeslip', 'dipslip', 'tensile', 'coupling'], [strikeslip, dipslip, tensile, coupling]):
            if len(gf_values) == 3:
                gf_values = [gf for gf in gf_values if gf is not None]
                if gf_values:
                    d, m = gf_values[0].shape
                    G[gf_type] = np.array(gf_values).reshape((len(gf_values)*d, m))
            elif gf_values[0] is not None:
                G[gf_type] = gf_values[0]

        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def dropPointSources(self):
        '''
        Drops point sources along the fault. Point sources can then be used
        to compute GFs using the EDKS software.

        The process is controlled by the attributes:
            - self.sourceSpacing      : Distance between sources
            - self.sourceArea         : Area of the sources
            - self.sourceNumber       : Number of sources per patch
        One needs to set at least one of those three attributes.

        Sources are saved in self.plotSources and self.edksSources

        Returns:
            * None
        '''

        # Compute sources
        Ids, xs, ys, zs, strike, dip, Areas = Patches2Sources(self, verbose=True)

        # Save them
        self.plotSources = [Ids, xs, ys, zs, strike, dip, Areas]
        self.edksSources = [Ids, xs*1e3, ys*1e3, zs*1e3,
                            strike*180/np.pi, dip*180/np.pi, Areas*1e6]

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def rotateGFs(self, data, azimuth):
        '''
        For the data set data, returns the rotated GFs so that dip slip motion
        is aligned with the azimuth. It uses the Greens functions stored
        in self.G[data.name].

        Args:
            * data          : Name of the data set.
            * azimuth       : Direction in which to rotate the GFs

        Returns:
            * rotatedGar    : GFs along the azimuth direction
            * rotatedGrp    : GFs in the direction perpendicular to the azimuth direction
        '''

        # Check if strike and dip slip GFs have been computed
        assert 'strikeslip' in self.G[data.name].keys(), \
                        "No strike slip Green's function available..."
        assert 'dipslip' in self.G[data.name].keys(), \
                        "No dip slip Green's function available..."

        # Get the Green's functions
        Gss = self.G[data.name]['strikeslip']
        Gds = self.G[data.name]['dipslip']

        # Do the rotation
        rotatedGar, rotatedGrp = self._rotatedisp(Gss, Gds, azimuth)

        #Store it, it will be used to return the slip vector.
        self.azimuth = azimuth

        # All done
        return rotatedGar, rotatedGrp
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def assembled(self, datas, verbose=True):
        '''
        Assembles a data vector for inversion using the list datas
        Assembled vector is stored in self.dassembled

        Args:
            * datas         : list of data objects

        Returns:
            * None
        '''

        # Ensure datas is a list
        datas = datas if isinstance(datas, list) else [datas]

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling d vector")

        # Use list comprehension to get all local d and concatenate them into one vector
        self.dassembled = np.concatenate([self.d[data.name] for data in datas])

        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def assembleGFs(self, datas, polys=None, slipdir='sdt', verbose=True,
                                custom=False, computeNormFact=True, computeIntStrainNormFact=True):
        '''
        Assemble the Green's functions corresponding to the data in datas.
        This method allows to specify which transformation is going
        to be estimated in the data sets, through the polys argument.

        Assembled Green's function matrix is stored in self.Gassembled

        Args:
            * datas : list of data sets. If only one data set is used, can be a data instance only.

        Kwargs:
            * polys : None, nothing additional is estimated

                - For InSAR, Optical, GPS:
                    - 1: estimate a constant offset
                    - 3: estimate z = ax + by + c
                    - 4: estimate z = axy + bx + cy + d

                - For GPS only:
                    - 'full'      : Estimates a rotation, translation and scaling (Helmert transform).
                    - 'strain'    : Estimates the full strain tensor (Rotation + Translation + Internal strain)
                    - 'strainnorotation'   : Estimates the strain tensor and a translation
                    - 'strainonly'    : Estimates the strain tensor
                    - 'strainnotranslation'   : Estimates the strain tensor and a rotation
                    - 'translation'   : Estimates the translation
                    - 'translationrotation' : Estimates the translation and a rotation
                    - 'eulerrotation' : Estimates a rotation only (for tectonic plate motion)
                    - 'internalstrain' : Estimates the internal strain only (no rotation, no translation)

            * slipdir   : Directions of slip to include. Can be any combination of s (strike slip), d (dip slip), t (tensile), c (coupling)

            * custom    : If True, gets the additional Green's function from the dictionary self.G[data.name]['custom']

            * computeNormFact   : bool. if True, compute new OrbNormalizingFactor. if False, uses parameters in self.OrbNormalizingFactor

            * verbose   : Talk to me (overwrites self.verbose)

        Returns:
            * None
        '''

        # Check
        datas = datas if isinstance(datas, list) else [datas]

        # print
        if verbose:
            logger.info("---------------------------------")
            logger.info("Assembling G for fault {}".format(self.name))

        # Store the assembled slip directions
        self.slipdir = slipdir

        # Create a dictionary to keep track of the orbital forms
        self.poly = {}
        self.numberofpolys = {}
        
        # NEW: Track transformation parameter indices
        self.transform_indices = {}

        # Set poly right
        if polys.__class__ is not list:
            for data in datas:
                if (polys.__class__ is not str) and (polys is not None):
                    if data.dtype == 'crossfaultoffset':
                        self.poly[data.name] = polys
                    elif data.dtype == 'leveling':
                        self.poly[data.name] = polys
                    else:
                        self.poly[data.name] = polys*data.obs_per_station
                else:
                    self.poly[data.name] = polys
        elif polys.__class__ is list:
            for data, poly in zip(datas, polys):
                if (poly.__class__ is not str) and (poly is not None) and (poly.__class__ is not list):
                    if data.dtype == 'crossfaultoffset':
                        self.poly[data.name] = poly
                    elif data.dtype == 'leveling':
                        self.poly[data.name] = poly
                    else:
                        self.poly[data.name] = poly*data.obs_per_station
                else:
                    self.poly[data.name] = poly

        # Create the transformation holder
        for attr in ['helmert', 'strain', 'eulerrot', 'intstrain', 'transformation']:
            if not hasattr(self, attr):
                setattr(self, attr, {})

        # Get the number of parameters
        if self.N_slip is None:
            self.N_slip = self.slip.shape[0]
        Nps = self.N_slip*len(slipdir)
        Npo = 0
        global_transform_start = Nps  # Transform parameters start after slip parameters
        
        for data in datas :
            transformation = self.poly[data.name]
            if type(transformation) in (str, list):
                tmpNpo = data.getNumberOfTransformParameters(self.poly[data.name])
                self.numberofpolys[data.name] = tmpNpo
                
                # NEW: Track parameter indices for this dataset
                data_transform_dict = {}
                current_pos = 0
                
                # Handle transformation list
                if isinstance(transformation, list):
                    for i, trans in enumerate(transformation):
                        if isinstance(trans, str):
                            # String transformation
                            trans_params = data.getNumberOfTransformParameters(trans)
                            data_transform_dict[trans] = (current_pos, current_pos + trans_params)
                            current_pos += trans_params
                        elif isinstance(trans, (int, np.integer)):
                            # Integer transformation (polynomial)
                            data_transform_dict[f'polynomial_{i}'] = (current_pos, current_pos + trans)
                            current_pos += trans
                elif isinstance(transformation, str):
                    # Single string transformation
                    data_transform_dict[transformation] = (0, tmpNpo)
                
                # Convert to global indices
                global_transform_dict = {}
                for trans_name, (start, end) in data_transform_dict.items():
                    global_start = global_transform_start + Npo + start
                    global_end = global_transform_start + Npo + end
                    global_transform_dict[trans_name] = (global_start, global_end)
                
                self.transform_indices[data.name] = global_transform_dict
                Npo += tmpNpo
                
                if type(transformation) is str:
                    # If poly is a string, store it in the right attribute
                    if transformation in ('full', 'helmert'):
                        self.helmert[data.name] = tmpNpo
                    elif transformation in ('strain', 'strainonly',
                                            'strainnorotation', 'strainnotranslation',
                                            'translation', 'translationrotation'):
                        self.strain[data.name] = tmpNpo
                    # Added by kfhe, at 10/12/2021
                    elif transformation in ('eulerrotation'):
                        self.eulerrot[data.name] = tmpNpo
                    elif transformation in ('internalstrain'):
                        self.intstrain[data.name] = tmpNpo
                else:
                    # If poly is a list, store it in transformation
                    self.transformation[data.name] = tmpNpo
            elif transformation is not None:
                # 1 or 3 only represent ramp correction, not a real transformation
                tmpNpo = data.getNumberOfTransformParameters(transformation)
                self.transform_indices[data.name] = {
                    'polynomial': (global_transform_start + Npo, global_transform_start + Npo + tmpNpo)
                }
                Npo += tmpNpo
                self.numberofpolys[data.name] = tmpNpo
                
        Np = Nps + Npo

        # Save extra Parameters
        self.TransformationParameters = Npo

        # Custom?
        if custom:
            Npc = 0
            custom_start = Np
            for data in datas:
                if 'custom' in self.G[data.name].keys():
                    custom_params = self.G[data.name]['custom'].shape[1]
                    if data.name not in self.transform_indices:
                        self.transform_indices[data.name] = {}
                    self.transform_indices[data.name]['custom'] = (custom_start + Npc, custom_start + Npc + custom_params)
                    Npc += custom_params
            Np += Npc
            self.NumberCustom = Npc
        else:
            Npc = 0

        if verbose:
            logger.info(f"Parameter summary:")
            logger.info(f"  Slip parameters: {Nps} (indices 0-{Nps-1})")
            logger.info(f"  Transform parameters: {Npo} (indices {Nps}-{Nps+Npo-1})")
            if custom:
                logger.info(f"  Custom parameters: {Npc} (indices {Nps+Npo}-{Np-1})")
            logger.info(f"  Total parameters: {Np}")
            
            # Display transformation parameter positions for each dataset
            for data_name, transform_dict in self.transform_indices.items():
                logger.info(f"  {data_name} transforms:")
                for trans_name, (start, end) in transform_dict.items():
                    logger.info(f"    {trans_name}: indices {start}-{end-1}")

        # Get the number of data
        Nd = 0
        for data in datas:
            Nd += self.d[data.name].shape[0]

        # Build the desired slip list
        sliplist = [slip for slip, char in zip(['strikeslip', 'dipslip', 'tensile', 'coupling'], 'sdtc') if char in slipdir]

        # Allocate G and d
        G = np.zeros((Nd, Np))

        # Create the list of data names, to keep track of it
        self.datanames = []

        # loop over the datasets
        el = 0
        custstart = Nps # custom indices
        polstart = Nps + Npc # poly indices
        for data in datas:

            # Keep data name
            self.datanames.append(data.name)

            # print
            if verbose:
                logger.info("Dealing with {} of type {}".format(data.name, data.dtype))

            # Elastic Green's functions

            # Get the corresponding G
            Ndlocal = self.d[data.name].shape[0]
            Glocal = np.zeros((Ndlocal, Nps))

            # Fill Glocal
            ec = 0
            for sp in sliplist:
                Nclocal = self.G[data.name][sp].shape[1]
                Glocal[:,ec:ec+Nclocal] = self.G[data.name][sp]
                ec += Nclocal

            # Put Glocal into the big G
            G[el:el+Ndlocal,0:Nps] = Glocal

            # Custom
            if custom:
                # Check if data has custom GFs
                if 'custom' in self.G[data.name].keys():
                    nc = self.G[data.name]['custom'].shape[1] # Nb of custom param
                    custend = custstart + nc
                    G[el:el+Ndlocal,custstart:custend] = self.G[data.name]['custom']
                    custstart += nc

            # Polynomes and strain
            if self.poly[data.name] is not None:

                # Build the polynomial function
                if data.dtype in ('gps', 'multigps'):
                    orb = data.getTransformEstimator(self.poly[data.name], computeNormFact=computeNormFact, computeIntStrainNormFact=computeIntStrainNormFact)
                elif data.dtype in ('insar', ):
                    # orb = data.getPolyEstimator(self.poly[data.name],computeNormFact=computeNormFact)
                    orb = data.getTransformEstimator(self.poly[data.name], computeNormFact=computeNormFact, computeIntStrainNormFact=computeIntStrainNormFact, verbose=verbose)
                elif data.dtype == 'opticorr':
                    orb = data.getTransformEstimator(self.poly[data.name], computeNormFact=computeNormFact, verbose=verbose)
                elif data.dtype == 'tsunami':
                    orb = data.getRampEstimator(self.poly[data.name])
                elif data.dtype == 'crossfaultoffset':
                    orb = data.getTransformEstimator(self.poly[data.name], computeNormFact=computeNormFact)
                elif data.dtype == 'leveling':
                    orb = data.getTransformEstimator(self.poly[data.name], computeNormFact=computeNormFact)

                # Number of columns
                nc = orb.shape[1]

                # Put it into G for as much observable per station we have
                polend = polstart + nc
                G[el:el+Ndlocal, polstart:polend] = orb
                polstart += nc

            # Update el to check where we are
            el = el + Ndlocal

        # Store G in self
        self.Gassembled = G

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def assembleCd(self, datas, add_prediction=None, verbose=False):
        '''
        Assembles the data covariance matrices that have been built for each
        data structure.

        Args:
            * datas         : List of data instances or one data instance

        Kwargs:
            * add_prediction: Precentage of displacement to add to the Cd diagonal to simulate a Cp (dirty version of a prediction error covariance, see Duputel et al 2013, GJI).
            * verbose       : Talk to me (overwrites self.verbose)

        Returns:
            * None
        '''

        # Check if the Green's function are ready
        assert self.Gassembled is not None, \
                "You should assemble the Green's function matrix first"

        # Check
        datas = datas if isinstance(datas, list) else [datas]

        # Get the total number of data
        Nd = self.Gassembled.shape[0]
        Cd = np.zeros((Nd, Nd))

        # Loop over the data sets
        st = 0
        for data in datas:
            # Fill in Cd
            if verbose:
                logger.info("{0}: data vector shape {1}".format(data.name, self.d[data.name].shape))
            se = st + self.d[data.name].shape[0]
            Cd[st:se, st:se] = data.Cd
            # Add some Cp if asked
            if add_prediction is not None:
                Cd[st:se, st:se] += np.diag((self.d[data.name]*add_prediction/100.)**2)
            st = se

        # Store Cd in self
        self.Cd = Cd

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildCmGaussian(self, sigma, extra_params=None):
        '''
        Builds a diagonal Cm with sigma values on the diagonal.
        Sigma is a list of numbers, as long as you have components of slip (1, 2 or 3).
        extra_params allows to add some diagonal terms and expand the size
        of the matrix, in case the fault object is also hosting the estimation
        of transformation parameters.

        Model covariance is hold in self.Cm

        Args:
            * sigma         : List of numbers the size of the slip components requried for the modeling

        Kwargs:
            * extra_params   : a list of extra parameters.

        Returns:
            * None
        '''

        # Get the number of slip directions
        slipdir = len(self.slipdir)
        if self.N_slip is None:
            self.N_slip = self.slip.shape[0]

        # Number of parameters
        Np = self.N_slip * slipdir
        if extra_params is not None:
            Np += len(extra_params)

        # Create Cm
        Cm = np.zeros((Np, Np))

        # Loop over slip dir
        for i in range(slipdir):
            Cmt = np.diag(sigma[i] * np.ones(self.N_slip,))
            Cm[i*self.N_slip:(i+1)*self.N_slip,i*self.N_slip:(i+1)*self.N_slip] = Cmt

        # Put the extra parameter sigma values
        st = self.N_slip * slipdir
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Stores Cm
        self.Cm = Cm

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildCmLaplacian(self, lam, diagFact=None, extra_params=None,
                                    sensitivity=True, method='distance',
                                    sensitivityNormalizing=False, irregular=False):
        '''
        Implements the Laplacian smoothing with sensitivity (optional) into
        a model covariance matrix. Description can be found in
        F. Ortega-Culaciati's PhD thesis.

        extra_params allows to add some diagonal terms and expand the size
        of the matrix, in case the fault object is also hosting the estimation
        of transformation parameters.

        Model covariance is hold in self.Cm

        Args:
            * lam                       : Damping factor (list of size of slipdirections)

        Kwargs:
            * extra_params              : a list of extra parameters.
            * sensitivity               : Weights the Laplacian by Sensitivity (default True)
            * sensitivityNormalizing    : Normalizing the Sensitivity?
            * method                    : which method to use to build the Laplacian operator 
            * irregular                 : Only used for rectangular patches. Allows to account for irregular meshing along dip.

        Returns:
            * None
        '''

        # lambda
        if type(lam) is float:
            lam = [lam for i in range(len(self.slipdir))]

        # Get the number of patches
        nSlip = self.N_slip
        if extra_params is not None:
            nExtra = len(extra_params)
        else:
            nExtra = 0

        # How many parameters
        Np = self.N_slip * len(self.slipdir)
        if extra_params is not None:
            Np += nExtra

        # Create the matrix
        Cm = np.zeros((Np, Np))

        # Build the laplacian
        D = self.buildLaplacian(verbose=True, method=method, irregular=irregular)

        Sensitivity = {}

        # Normalizing
        if sensitivityNormalizing:
            self.slipIntegrate()
            Volumes = self.volume

        # Loop over directions:
        for i in range(len(self.slipdir)):

            # Start/Stop
            ist = nSlip*i
            ied = ist+nSlip

            if sensitivity:

                # Compute sensitivity matrix (see Loveless & Meade, 2011)
                G = self.Gassembled[:,ist:ied]
                if sensitivityNormalizing:
                    G = G/self.volume[np.newaxis,:]
                S = np.diag(np.dot(G.T, G))
                Sensitivity[self.slipdir[i]] = S

                # Weight Laplacian by sensitivity (see F. Ortega-Culaciati PhD Thesis)
                iS = np.sqrt(1./S)
                D = D*iS[:,np.newaxis]

            # LocalCm
            D2 = np.dot(D.T,D)
            localCm = 1./lam[i]*np.linalg.inv(D2)

            # Mingle with the diagonal
            if diagFact is not None:
                localCm -= np.diag(np.diag(localCm))
                localCm += np.diag(np.max(localCm, axis=1))*diagFact

            # Put it into Cm
            Cm[ist:ied, ist:ied] = localCm

        # Add extra params
        if nExtra>0:
            CmRamp = np.diag(extra_params)
            Cm[-nExtra:, -nExtra:] = CmRamp

        # Set inside the fault
        self.Cm = Cm
        self.Laplacian = D
        self.Sensitivity = Sensitivity

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildCm(self, sigma, lam, lam0=None, extra_params=None, lim=None,
                                  verbose=True):
        '''
        Builds a model covariance matrix using the equation described in
        Radiguet et al 2010. We use

        :math:`C_m(i,j) = \\frac{\sigma \lambda_0}{ \lambda }^2 e^{-\\frac{||i,j||_2}{ \lambda }}`

        extra_params allows to add some diagonal terms and expand the size
        of the matrix, in case the fault object is also hosting the estimation
        of transformation parameters.

        Model covariance is stored in self.Cm

        Args:
            * sigma         : Amplitude of the correlation.
            * lam           : Characteristic length scale.

        Kwargs:
            * lam0          : Normalizing distance. if None, lam0=min(distance between patches)
            * extra_params  : A list of extra parameters.
            * lim           : Limit distance parameter (see self.distancePatchToPatch)
            * verbose       : Talk to me (overwrites self.verrbose)

        Returns:
            * None
        '''

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling the Cm matrix ")
            print ("Sigma = {}".format(sigma))
            print ("Lambda = {}".format(lam))

        # Geth the desired slip directions
        slipdir = self.slipdir

        # Get the patch centers
        self.centers = np.array(self.getcenters())

        # Sets the lambda0 value
        if lam0 is None:
            xd = ((np.unique(self.centers[:,0]).max() - np.unique(self.centers[:,0]).min())
                / (np.unique(self.centers[:,0]).size))
            yd = ((np.unique(self.centers[:,1]).max() - np.unique(self.centers[:,1]).min())
                / (np.unique(self.centers[:,1]).size))
            zd = ((np.unique(self.centers[:,2]).max() - np.unique(self.centers[:,2]).min())
                / (np.unique(self.centers[:,2]).size))
            lam0 = np.sqrt(xd**2 + yd**2 + zd**2)
        if verbose:
            logger.info("Lambda0 = {}".format(lam0))
        C = (sigma * lam0 / lam)**2

        # Creates the principal Cm matrix
        if self.N_slip is None:
            self.N_slip = self.slip.shape[0]
        Np = self.N_slip * len(slipdir)
        if extra_params is not None:
            Np += len(extra_params)
        Cm = np.zeros((Np, Np))

        # Loop over the patches
        distances = self.distanceMatrix(distance='center', lim=lim)
        Cmt = C * np.exp(-distances / lam)

        # Store that into Cm
        st = 0
        for i in range(len(slipdir)):
            se = st + self.N_slip
            Cm[st:se, st:se] = Cmt
            st += self.N_slip

        # Put the extra values
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Store Cm into self
        self.Cm = Cm

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildCmXY(self, sigma, lam, lam0=None, extra_params=None, lim=None,
                                  verbose=True):
        '''
        Builds a model covariance matrix using the equation described in
        Radiguet et al 2010 with a different characteristic lengthscale along
        the horizontal and vertical directions. We use

        :math:`C_m(i,j) = \\frac{\sigma \lambda_0}{ \lambda_x }^2 e^{-\\frac{||i,j||_{x2}}{ \lambda_x }} \\frac{\sigma \lambda_0}{ \lambda_z }^2 e^{-\\frac{||i,j||_{z2}}{ \lambda_z }}`

        extra_params allows to add some diagonal terms and expand the size
        of the matrix, in case the fault object is also hosting the estimation
        of transformation parameters.

        Model covariance is stored in self.Cm

        Args:
            * sigma         : Amplitude of the correlation.
            * lam           : Characteristic length scale (lamx, lamz)

        Kwargs:
            * lam0          : Normalizing distance. if None, lam0=min(distance between patches)
            * extra_params  : A list of extra parameters.
            * lim           : Limit distance parameter (see self.distancePatchToPatch)
            * verbose       : Talk to me (overwrites self.verrbose)

        Returns:
            * None
        '''

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling the Cm matrix ")
            print ("Sigma = {}".format(sigma))
            print ("Lambda = {}".format(tuple(lam)))

        # Geth the desired slip directions
        slipdir = self.slipdir

        # Get the patch centers
        if self.patchType=='triangulartent':
            self.centers = self.Vertices
        else:
            self.centers = np.array(self.getcenters())

        # Sets the lambda0 value
        if lam0 is None:
            xd = ((np.unique(self.centers[:,0]).max() - np.unique(self.centers[:,0]).min())
                / (np.unique(self.centers[:,0]).size))
            yd = ((np.unique(self.centers[:,1]).max() - np.unique(self.centers[:,1]).min())
                / (np.unique(self.centers[:,1]).size))
            zd = ((np.unique(self.centers[:,2]).max() - np.unique(self.centers[:,2]).min())
                / (np.unique(self.centers[:,2]).size))
            lam0 = np.sqrt(xd**2 + yd**2 + zd**2)
        if verbose:
            logger.info("Lambda0 = {}".format(lam0))
        C = (sigma * lam0 / lam[0]) * (sigma * lam0 / lam[1])

        # Creates the principal Cm matrix
        if self.N_slip==None:
            self.N_slip = self.slip.shape[0]
        Np = self.N_slip * len(slipdir)
        if extra_params is not None:
            Np += len(extra_params)
        Cm = np.zeros((Np, Np))

        # Loop over the patches
        Hdistances,Vdistances = self.distancesMatrix(distance='center', lim=lim)
        Cmt = C * np.exp(-0.5 * Hdistances / lam[0]) * np.exp(-0.5 * Vdistances / lam[1])

        # Store that into Cm
        st = 0
        for i in range(len(slipdir)):
            se = st + self.N_slip
            Cm[st:se, st:se] = Cmt
            st += self.N_slip

        # Put the extra values
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Store Cm into self
        self.Cm = Cm

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildCmSlipDirs(self, sigma, lam, lam0=None, extra_params=None,
                                          lim=None, verbose=True):
        '''
        Builds a model covariance matrix using the equation described in
        Radiguet et al 2010. Here, Sigma and Lambda are lists specifying
        values for the slip directions. We use

        :math:`C_m(i,j) = \\frac{\sigma\lambda_0}{\lambda}^2 e^{-\\frac{||i,j||_2}{\lambda}}`

        Args:
            * sigma         : Amplitude of the correlation.
            * lam           : Characteristic length scale.

        Kwargs:
            * lam0          : Normalizing distance. If None, lam0=min(distance between patches)
            * extra_params  : A list of extra parameters.
            * lim           : Limit distance parameter (see self.distancePatchToPatch)
            * verbose       : Talk to me (overwrites self.verrbose)

        Returns:
            * None
        '''

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling the Cm matrix ")
            print ("Sigma = {}".format(sigma))
            print ("Lambda = {}".format(lam))

        # Need the patch geometry
        assert self.patch is not None,\
                "You should build the patches and the Green's functions first."

        # Get slip
        if self.N_slip is None:
            self.N_slip = self.slip.shape[0]

        # Geth the desired slip directions
        slipdir = self.slipdir

        # Get the patch centers
        self.centers = np.array(self.getcenters())

        # Sets the lambda0 value
        if lam0 is None:
            xd = (np.unique(self.centers[:,0]).max() - \
                    np.unique(self.centers[:,0]).min())/(np.unique(self.centers[:,0]).size)
            yd = (np.unique(self.centers[:,1]).max() - \
                    np.unique(self.centers[:,1]).min())/(np.unique(self.centers[:,1]).size)
            zd = (np.unique(self.centers[:,2]).max() - \
                    np.unique(self.centers[:,2]).min())/(np.unique(self.centers[:,2]).size)
            lam0 = np.sqrt( xd**2 + yd**2 + zd**2 )

        # Creates the principal Cm matrix
        Np = self.N_slip*len(slipdir)
        if extra_params is not None:
            Np += len(extra_params)
        Cmt = np.zeros((self.N_slip, self.N_slip))
        Cm = np.zeros((Np, Np))

        # Build the sigma and lambda lists
        if type(sigma) is not list:
            s = []; l = []
            for sl in range(len(slipdir)):
                s.append(sigma)
                l.append(lam)
            sigma = s
            lam = l
        assert (type(sigma) is list), 'Sigma is not a list, why???'
        assert(len(sigma)==len(lam)), 'Sigma and lambda must have the same length'
        assert(len(sigma)==len(slipdir)), \
                'Need one value of sigma and one value of lambda per slip direction'

        # Loop over the slipdirections
        st = 0
        for sl in range(len(slipdir)):
            # pick the right values
            la = lam[sl]
            C = (sigma[sl]*lam0/la)**2
            # Get distance matrix
            distance = self.distanceMatrix(distance='center', lim=lim)
            # Compute Cmt
            Cmt = C * np.exp( -1.0*distance/la)
            # Store that into Cm
            se = st + self.N_slip
            Cm[st:se, st:se] = Cmt
            st += self.N_slip

        # Put the extra values
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Store Cm into self
        self.Cm = Cm

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildCmSensitivity(self, sigma, lam, lam0=None, extra_params=None,
                                              lim=None, verbose=True):
        '''
        Builds a model covariance matrix using the equation described in Radiguet et al 2010.
        We use

        :math:`C_m(i,j) = \\frac{\sigma\lambda_0}{\lambda}^2 e^{-\\frac{||i,j||_2}{\lambda}}`

        Then correlation length is weighted by the sensitivity matrix described in Ortega's PhD thesis:
        :math:`S = diag(G'G)`
        
        Here, Sigma and Lambda are lists specifying values for the slip directions

        extra_params allows to add some diagonal terms and expand the size
        of the matrix, in case the fault object is also hosting the estimation
        of transformation parameters.

        Model covariance is stored in self.Cm

        Args:
            * sigma         : Amplitude of the correlation.
            * lam           : Characteristic length scale.

        Kwargs:
            * lam0          : Normalizing distance. if None, lam0=min(distance between patches)
            * extra_params  : a list of extra parameters.
            * lim           : Limit distance parameter (see self.distancePatchToPatch)
            * verbose       : Talk to me (overwrites self.verrbose)

        Returns:
            * None
        '''

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling the Cm matrix ")
            print ("Sigma = {}".format(sigma))
            print ("Lambda = {}".format(lam))

        # Assert
        assert hasattr(self, 'Gassembled'), "Need to assemble the Green's functions"

        # Need the patch geometry
        assert self.patch is not None, "You should build the patches and the Green's functions first."

        # Set
        self.N_slip = self.slip.shape[0]

        # Geth the desired slip directions
        slipdir = self.slipdir

        # Get the patch centers
        self.centers = np.array(self.getcenters())

        # Sets the lambda0 value
        if lam0 is None:
            xd = (np.unique(self.centers[:,0]).max() - \
                    np.unique(self.centers[:,0]).min())/(np.unique(self.centers[:,0]).size)
            yd = (np.unique(self.centers[:,1]).max() - \
                    np.unique(self.centers[:,1]).min())/(np.unique(self.centers[:,1]).size)
            zd = (np.unique(self.centers[:,2]).max() - \
                    np.unique(self.centers[:,2]).min())/(np.unique(self.centers[:,2]).size)
            lam0 = np.sqrt( xd**2 + yd**2 + zd**2 )

        # Creates the principal Cm matrix
        Np = self.N_slip*len(slipdir)
        if extra_params is not None:
            Np += len(extra_params)
        Cmt = np.zeros((self.N_slip, self.N_slip))
        lambdast = np.zeros((self.N_slip, self.N_slip))
        Cm = np.zeros((Np, Np))
        Lambdas = np.zeros((Np, Np))

        # Build the sigma and lambda lists
        if type(sigma) is not list:
            s = []; l = []
            for sl in range(len(slipdir)):
                s.append(sigma)
                l.append(lam)
            sigma = s
            lam = l
        assert type(sigma) is list, 'Sigma needs to be a list'
        assert(len(sigma)==len(lam)), 'Sigma and lambda must have the same length'
        assert(len(sigma)==len(slipdir)), 'Need one value of sigma and one value of lambda per slip direction'

        # Loop over the slipdirections
        st = 0
        for sl in range(len(slipdir)):

            # Update a counter
            se = st + self.N_slip

            # Get the greens functions and build sensitivity
            G = self.Gassembled[:,st:se]
            S = np.diag(np.dot(G.T, G)).copy()
            ss = S.max()
            S /= ss

            # pick the right values
            la = lam[sl]

            # Loop over the patches
            distance = self.distanceMatrix(distance='center', lim=lim)

            # Weight Lambda by the relative sensitivity
            s1, s2 = np.meshgrid(S, S)
            L = la/np.sqrt(s1*s2)
            # Compute Cm
            Cmt = ((sigma[sl]*lam0/L)**2) * np.exp( -1.0*distance/L)

            # Store that into Cm
            Cm[st:se, st:se] = Cmt
            Lambdas[st:se, st:se] = lambdast
            st += self.N_slip

        # Put the extra values
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Store Cm into self
        self.Cm = Cm
        self.Lambdas = Lambdas

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def writePatchesCenters2File(self, filename, slip=None, scale=1.0):
        '''
        Write the patch center coordinates in an ascii file
        the file format is so that it can by used directly in psxyz (GMT).

        Args:
            * filename      : Name of the file.

        Kwargs:
            * slip          : Put the slip as a value for the color. Can be None, strikeslip, dipslip, total, coupling
            * scale         : Multiply the slip value by a factor.

        Retunrs:
            * None
        '''

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.patch):
            raise NotImplementedError('Only works for len(slip)==len(patch)')

        # Select the string for the color
        if slip is not None:
            if slip == 'coupling':
                slp = self.coupling[:]
            elif slip == 'strikeslip':
                slp = self.slip[:,0]*scale
            elif slip == 'dipslip':
                slp = self.slip[:,1]*scale
            elif slip == 'total':
                slp = np.sqrt(self.slip[:,0]**2 + self.slip[:,1]**2)*scale
            else:
                try:
                    slp = getattr(self, slip)
                except:
                    assert False, 'No value called {}'.format(slip)

        # Write something
        logger.info('Writing geometry to file {}'.format(filename))

        # Open the file
        fout = open(filename, 'w')

        # Loop over the patches
        nPatches = len(self.patch)
        for patch in self.patch:

            # Get patch index
            pIndex = self.getindex(patch)

            # Get patch center
            xc, yc, zc = self.getcenter(patch)
            lonc, latc = self.xy2ll(xc, yc)

            # Write the string to file
            fout.write('{} {} {} {} \n'.format(lonc, latc, zc, slp[pIndex]))

        # Close the file
        fout.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def sumPatches(self, iPatches, finalPatch):
        '''
        Takes a list of indexes of patches, sums the corresponding GFs and
        replace the corresponding patches by the finalPatch in self.patch

        Args:
            * patches       : List of the patche indexes to sum
            * finalPatch    : Geometry of the final patch.

        Returns:
            * None
        '''

        # Needs to have Greens functions
        assert len(self.G.keys())>0, 'Need some Greens functions, otherwise this function is pointless'

        # Loop over the data sets
        for data in self.G:

            # Get it
            G = self.G[data]

            # Loop over the Green's functions
            for comp in G:

                # Get the matrix
                gf = G[comp]

                # Sum the columns
                col = np.sum(gf[:,iPatches], axis=1)

                # New matrix
                gf = np.delete(gf, iPatches[1:], axis=1)
                gf[:,iPatches[0]] = col

                # Set it
                G[comp] = gf

        # Replace the first of the patches by the new patch
        self.replacePatch(finalPatch, iPatches[0])

        # Delete the other patches
        self.deletepatches(iPatches[1:])

        # Equivalent Patches
        if self.patchType == 'rectangle':
            self.computeEquivRectangle()

        # Check
        self.N_slip = len(self.patch)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def estimateSeismicityRate(self, earthquake, extra_div=1.0, epsilon=0.00001):
        '''
        Counts the number of earthquakes per patches and divides by the area of the patches.
        Sets the results in
            self.earthquakeInPatch (Number of earthquakes per patch) and self.seismicityRate (Seismicity rate for this patch)

        Args:
            * earthquake    : seismiclocation object

        Kwargs:
            * extra_div     : Extra divider to get the seismicity rate.
            * epsilon       : Epsilon value for precision of earthquake location.

        Returns:
            * None
        '''

        # Make sure the area of the fault patches is computed
        self.computeArea()

        # Project the earthquakes on fault patches
        ipatch = earthquake.getEarthquakesOnPatches(self, epsilon=epsilon)

        # Count
        number = np.zeros(len(self.patch))

        # Loop
        for i in range(len(self.patch)):
            number[i] = len(ipatch[i].tolist())/(self.area[i]*extra_div)

        # Store that in the fault
        self.earthquakesInPatch = ipatch
        self.seismicityRate = number

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def gaussianSlipSmoothing(self, length):
        '''
        Smoothes the slip distribution using a Gaussian filter.
        Smooth slip distribution is in self.slip

        Args:
            * length        : Correlation length.

        Returns:
            * None
        '''

        # Number of patches
        nP = self.slip.shape[0]

        # Build the smoothing matrix
        S = self.distanceMatrix(distance='center', lim=None)**2

        # Compute
        S = np.exp(-0.5*S/(length**2))
        div = 1./S.sum(axis=0)
        S = np.multiply(S, div)
        self.Smooth = S

        # Smooth
        self.slip[:,0] = np.dot(S, self.slip[:,0])
        self.slip[:,1] = np.dot(S, self.slip[:,1])
        self.slip[:,2] = np.dot(S, self.slip[:,2])

        # All done
        return
    # ----------------------------------------------------------------------

    def slipIntegrate(self, slip=None):
        '''
        Integrates slip on the patch by simply multiplying slip by the
        patch area. Sets the results in self.volume

        Kwargs:
            * slip  : Can be strikeslip, dipslip, tensile, coupling or a list/array of floats.

        Returns:
            * None
        '''

        # Slip
        if type(slip) is str:
            if slip=='strikeslip':
                slip = self.slip[:,0]
            elif slip=='dipslip':
                slip = self.slip[:,1]
            elif slip=='tensile':
                slip = self.slip[:,2]
            elif slip=='coupling':
                slip = self.coupling
            else:
                slip = getattr(self, slip)
        elif type(slip) in (np.ndarray, list):
            assert len(slip)==len(self.patch), 'Slip vector is the wrong size'
        else:
            slip = np.ones((len(self.patch),))

        # Compute Volumes
        self.computeArea()
        self.volume = self.area*slip

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setmu(self, model_file, tents = False):
        '''
        Gets the shear modulus corresponding to each patch using a model
        file from the EDKS software. Shear moduli are set in self.mu

        The model file format is as follows:

        +-----+----+----+----+
        |  N  | F  |    |    | 
        +=====+====+====+====+
        |RHO_1|VP_1|VS_1|TH_1|
        +-----+----+----+----+
        |RHO_2|VP_2|VS_2|TH_2|
        +-----+----+----+----+
        | ... | ...| ...| ...|
        +-----+----+----+----+
        |RHO_N|VP_N|VS_N|TH_N|
        +-----+----+----+----+

        where N is the number of layers, F a conversion factor to SI units
        RHO_i is the density of the i-th layer
        VP_i is the P-wave velocity in the i-th layer
        VS_i is the S-wave velocity in the i-th layer
        TH_i is the thickness of the i-th layer

        Args:
            * model_file    : path to model file
            * tents         : if True, set mu values every point source in patches

        Returns:
            * None
        '''

        # Read model file
        mu = []
        depth  = 0.
        depths = []
        with open(model_file) as f:
            L = f.readlines()
            items = L[0].strip().split()
            N = int(items[0])
            F = float(items[1])
            for l in L[1:]:
                c = l.strip()
                if len(c) and c[0]=='#':
                    continue
                items = c.split()
                if len(items)!=4:
                    continue
                TH  = float(items[3])*F
                VS  = float(items[2])*F
                RHO = float(items[0])*F
                mu.append(VS*VS*RHO)
                if TH==0.:
                    TH = np.inf
                depths.append([depth,depth+TH])
                depth += TH
        depths = np.array(depths)*1e-3 # depth in km
        Nd = len(depths)
        if tents:
            if self.keepTrackOfSources and hasattr(self, 'edksSources'):
                Ids, xs, ys, zs, strike, dip, Areas = self.edksSources[:7]

            else:
                Ids, xs, ys, zs, strike, dip, Areas = Patches2Sources(self)
                # All these guys need to be in meters
                xs *= 1000. ; ys *= 1000. ; zs *= 1000.
                Areas *= 1e6
                # Strike and dip in degrees
                strike = strike*180./np.pi
                dip = dip*180./np.pi
                # Keep track?
                self.edksSources = [Ids, xs, ys, zs, strike, dip, Areas]

            Np = len(self.edksSources[0])

        else:
            Np = len(self.patch)

        # Set Mu for each patch
        self.mu = np.zeros((Np,))
        for p in range(Np):
            if tents:
               p_z = zs[p]/1000.
            else:
                p_x, p_y, p_z,width, length, strike_rad, dip_rad = self.getpatchgeometry(p,center=True)

            for d in range(Nd):
                if p_z>=depths[d][0] and p_z<depths[d][1]:
                    self.mu[p] = mu[d]

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Some building routines that should not be touched
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def _buildGFsdict(self, data, Gss, Gds, Gts,
                            slipdir='sd', convergence=None, vertical=True):
        '''
        Some ordering of the Gfs to make the computation routines simpler.

        Args:
            * data          : instance of data
            * Gss           : Strike slip greens functions
            * Gds           : Dip slip greens functions
            * Gts           : Tensile greens functions

        Kwargs:
            * slipdir       : Direction of slip. Can be any combination of 's', 'd', 't' or 'c'
            *convergence    : Convergence vector for coupling GFs [azimuth in degree, rate]
            *vertical       : If true, assumes verticals are used for the GPS case

        Returns:
            * G             : Dictionary of GFs
        '''

        # Compute Coupling GFs
        if 'c' in slipdir:
            Gcs = self._disp4coupling(Gss, Gds, convergence)
        else:
            Gcs = np.zeros(Gss.shape)

        # Verticals?
        Ncomp = 3
        if not vertical:
            Ncomp = 2
            Gss, Gds, Gts, Gcs = [G[:2,:,:] if s in slipdir else G for G, s in zip([Gss, Gds, Gts, Gcs], 'sdtc')]

        # Get some size info
        Nparm = Gss.shape[2]
        Npoints = Gss.shape[1]
        Ndata = Ncomp*Npoints

        # Check format
        if data.dtype in ['gps', 'opticorr', 'multigps']:
            # Flat arrays with e, then n, then u (optional)
            Gss, Gds, Gts, Gcs = [G.reshape((Ndata, Nparm)) if s in slipdir else G for G, s in zip([Gss, Gds, Gts, Gcs], 'sdtc')]
        elif data.dtype in ('insar', 'insartimeseries'):
            # If InSAR, do the dot product with the los
            # print(data.los.shape, Gss.shape)
            Gss, Gds, Gts, Gcs = [np.einsum('ij,ijk->ik', data.los, np.transpose(G, (1, 0, 2))) if s in slipdir else G for G, s in zip([Gss, Gds, Gts, Gcs], 'sdtc')]
        elif data.dtype == 'leveling':
            # Leveling uses only the vertical (U) component — index 2
            Gss, Gds, Gts, Gcs = [G[2, :, :] if s in slipdir else G for G, s in zip([Gss, Gds, Gts, Gcs], 'sdtc')]
        elif data.dtype == 'crossfaultoffset':
            raise RuntimeError(
                'crossfaultoffset GFs must be projected before _buildGFsdict. '
                'Use cutdeGFs / homogeneousGFs / edksGFs / pscmpGFs / edcmpGFs which '
                'dispatch to the crossfaultoffset-specific handler automatically.')

        # Create the dictionary
        G = {'strikeslip':[], 'dipslip':[], 'tensile':[], 'coupling':[]}

        # Reshape the Green's functions
        for s, key in zip('sdtc', ['strikeslip', 'dipslip', 'tensile', 'coupling']):
            G[key] = locals()['G' + s + 's'] if s in slipdir else None

        # All done
        return G
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def _rotatedisp(self, Gss, Gds, azimuth):
        '''
        A rotation function for Green function.

        Args:
            * Gss           : Strike slip GFs
            * Gds           : Dip slip GFs
            * azimtuh       : Direction to rotate (degrees)

        Return:
            * rotatedGar    : Displacements along azimuth
            * rotatedGrp    : Displacements perp. to azimuth direction
        '''

        # Make azimuth positive
        if azimuth < 0.:
            azimuth += 360.

        # Get strikes and dips
        #if self.patchType is 'triangletent':
        #    strike = super(self.__class__, self).getStrikes()
        #    dip = super(self.__class__, self).getDips()
        #else:
        strike, dip = self.getStrikes(), self.getDips()

        # Convert angle in radians
        azimuth *= ((np.pi) / 180.)
        rotation = np.arctan2(np.tan(strike) - np.tan(azimuth),
                            np.cos(dip)*(1.+np.tan(azimuth)*np.tan(strike)))

        # If azimuth within ]90, 270], change rotation
        if azimuth*(180./np.pi) > 90. and azimuth*(180./np.pi)<=270.:
            rotation += np.pi

        # Store rotation angles
        self.rotation = rotation.copy()

        # Rotate them (ar: along-rake; rp: rake-perpendicular)
        rotatedGar = Gss*np.cos(rotation) + Gds*np.sin(rotation)
        rotatedGrp = Gss*np.sin(rotation) - Gds*np.cos(rotation)

        # All done
        return rotatedGar, rotatedGrp
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def _disp4coupling(self, Gss, Gds, convergence):
        '''
        Converts the displacements into what we need to build coupling GFs
        Gss and Gds are of a shape (3xnumber of sites, number of fault patches)
        The 3 is for East, North and Up displacements

        Args:
            * Gss           : Strike slip GFs
            * Gds           : Dip slip GFs
            * convergence   : [azimuth in degrees, rate]

        Returns:
            * Gar           : Along coupling Greens functions

        '''

        # For now, convergence is constant alnog strike
        azimuth, rate = convergence

        # Create the holders
        Gar = np.zeros(Gss.shape)
        Grp = np.zeros(Gds.shape)

        # Rotate the GFs
        Gar[0,:,:], Grp[0,:,:] = self._rotatedisp(Gss[0,:,:], Gds[0,:,:], azimuth)
        Gar[1,:,:], Grp[1,:,:] = self._rotatedisp(Gss[1,:,:], Gds[1,:,:], azimuth)
        Gar[2,:,:], Grp[2,:,:] = self._rotatedisp(Gss[2,:,:], Gds[2,:,:], azimuth)

        # Multiply and sum
        Gar *= rate

        # All done (we only retun Gar as Grp should be 0)
        return Gar
    # ----------------------------------------------------------------------

#EOF
