'''
A class for multiple vertical strain volumes (Barbot, 2017).

Manages a collection of rectangular vertical strain volumes, analogous
to how Fault manages a collection of fault patches.  Each volume is
parameterised by position (q1, q2, q3), geometry (L, T, W), and strike
(theta), and can carry anelastic strain components as "unknowns" for
inversion, similar to slip on fault patches.

Reference:
    Barbot S., J. D. P. Moore and V. Lambert, 2017.
    Displacement and Stress Associated with Distributed Anelastic
    Deformation in a Half Space,
    Bull. Seism. Soc. Am., 107(2), 10.1785/0120160237.

Written by kfhe, 2026.
'''

import numpy as np
import copy
import os
import logging
from tqdm import tqdm
from scipy.linalg import block_diag

from .SourceInv import SourceInv
from . import sbarbotfull

logger = logging.getLogger(__name__)

# Canonical ordering of the six independent strain components
ALL_STRAIN_COMPONENTS = ('eps11', 'eps12', 'eps13', 'eps22', 'eps23', 'eps33')


class Sbarbot(SourceInv):
    '''
    Class implementing multiple vertical strain volumes in an elastic
    half-space using the analytical solution of Barbot et al. (2017).

    Volumes are organised like fault patches: each has fixed geometry
    and variable strain amplitude.  Green's functions map unit strain
    per volume to surface displacement, enabling geodetic inversion.

    Args:
        * name      : Name of the source set.
        * utmzone   : UTM zone (optional, default=None)
        * lon0      : Longitude of custom UTM centre
        * lat0      : Latitude of custom UTM centre
        * ellps     : Ellipsoid (default='WGS84')
    '''

    def __init__(self, name, utmzone=None, ellps='WGS84',
                 lon0=None, lat0=None, verbose=True):

        super(Sbarbot, self).__init__(name, utmzone=utmzone, ellps=ellps,
                                      lon0=lon0, lat0=lat0)

        if verbose:
            print("---------------------------------")
            print("---------------------------------")
            print("Initializing Sbarbot source set {}".format(self.name))
        self.verbose = verbose
        self.type = "Sbarbot"

        # Volume storage -------------------------------------------------
        # Each volume: (x_km, y_km, depth_km, L_km, T_km, W_km, strike_deg)
        #
        # (x, y) = UTM (easting, northing) in km.  These map to the
        # Barbot (2017) Fortran parameters as:
        #   q1 = y  (northing)   q2 = x  (easting)   q3 = depth
        #
        # (q1, q2) is the **reference point** of the volume, located at
        # the **start** of the length (L) direction and the **center**
        # of the thickness (T) direction.  q3 is the **top depth**.
        #
        # From this reference point the volume extends:
        #   L  along strike            (y1': 0    -> L)
        #   T  centred perp to strike  (y2': -T/2 -> +T/2)
        #   W  downward in depth       (y3': q3   -> q3+W)
        #
        # strike = azimuth from north (degrees, CW positive).
        self.volumes = []
        self.lon_vol = None
        self.lat_vol = None
        self.x_vol = None
        self.y_vol = None

        # Strain ----------------------------------------------------------
        self.strain = None              # (Nvol, N_strain) array
        self.strain_components = None   # e.g. ['eps12', 'eps13']
        self.N_strain = None

        # Elastic parameters
        self.mu = 30e9
        self.nu = 0.25

        # GFs and data
        self.G = {}
        self.d = {}
        self.Gassembled = None
        self.dassembled = None
        self.polysol = {}

        # For assembleGFs tracking
        self.slipdir = None
        self.poly = {}
        self.numberofpolys = {}
        self.transform_indices = {}
        self.TransformationParameters = 0
        self.NumberCustom = 0
        self.datanames = []
        self.cleanUp = True

    # ------------------------------------------------------------------
    # Volume management
    # ------------------------------------------------------------------

    def addVolume(self, lon, lat, depth, L, T, W, strike, latlon=True):
        '''
        Add a single strain volume.

        The position (lon, lat) / (x, y) maps to the Fortran reference
        point (q1, q2): the **start** of the length direction and the
        **center** of the thickness direction.  See class docstring for
        the full geometry diagram.

        Args:
            * lon, lat  : reference-point position (lon/lat if latlon=True,
                          UTM easting/northing km otherwise).
            * depth     : top depth of volume (km, positive down),
                          volume extends from depth to depth + W.
            * L         : length along strike (km), extends from
                          the reference point in the strike direction.
            * T         : thickness perpendicular to strike (km),
                          centred about the reference point (-T/2 to +T/2).
            * W         : depth extent (km), volume goes from depth to
                          depth + W.
            * strike    : azimuth from north, clockwise positive (degrees).
            * latlon    : if True, lon/lat are geographic, else UTM km.
        '''
        if latlon:
            x, y = self.ll2xy(lon, lat)
        else:
            x, y = lon, lat

        self.volumes.append((x, y, depth, L, T, W, strike))
        self._update_coord_arrays()

    def setVolumes(self, lons, lats, depths, Ls, Ts, Ws, strikes, latlon=True):
        '''
        Set all volumes at once.  All arguments are array-like of length Nvol.
        '''
        lons = np.atleast_1d(np.asarray(lons, dtype=float))
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        depths = np.atleast_1d(np.asarray(depths, dtype=float))
        Ls = np.atleast_1d(np.asarray(Ls, dtype=float))
        Ts = np.atleast_1d(np.asarray(Ts, dtype=float))
        Ws = np.atleast_1d(np.asarray(Ws, dtype=float))
        strikes = np.atleast_1d(np.asarray(strikes, dtype=float))

        Nvol = len(lons)
        assert all(len(a) == Nvol for a in [lats, depths, Ls, Ts, Ws, strikes])

        self.volumes = []
        for i in range(Nvol):
            if latlon:
                x, y = self.ll2xy(lons[i], lats[i])
            else:
                x, y = lons[i], lats[i]
            self.volumes.append((x, y, float(depths[i]), float(Ls[i]),
                                 float(Ts[i]), float(Ws[i]), float(strikes[i])))

        self._update_coord_arrays()

    def _update_coord_arrays(self):
        '''Rebuild the lon/lat/x/y arrays from self.volumes.'''
        if len(self.volumes) == 0:
            self.x_vol = np.array([])
            self.y_vol = np.array([])
            self.lon_vol = np.array([])
            self.lat_vol = np.array([])
            return
        arr = np.array(self.volumes)
        self.x_vol = arr[:, 0]
        self.y_vol = arr[:, 1]
        self.lon_vol, self.lat_vol = self.xy2ll(self.x_vol, self.y_vol)

    def readVolumesFromFile(self, filename, latlon=True, header=0):
        '''
        Read volumes from an ASCII file.

        Expected columns: lon  lat  depth  L  T  W  strike
        (or x  y  depth  L  T  W  strike if latlon=False)
        '''
        data = np.loadtxt(filename, skiprows=header)
        assert data.shape[1] >= 7, "Need at least 7 columns: lon lat depth L T W strike"
        self.setVolumes(data[:, 0], data[:, 1], data[:, 2],
                        data[:, 3], data[:, 4], data[:, 5], data[:, 6],
                        latlon=latlon)

    def writeVolumesToFile(self, filename):
        '''Write volumes to ASCII file (lon lat depth L T W strike).'''
        with open(filename, 'w') as f:
            f.write("# lon lat depth(km) L(km) T(km) W(km) strike(deg)\n")
            for vol in self.volumes:
                x, y, dep, L, T, W, strike = vol
                lo, la = self.xy2ll(x, y)
                f.write(f"{lo:.8f} {la:.8f} {dep:.4f} "
                        f"{L:.4f} {T:.4f} {W:.4f} {strike:.2f}\n")

    def getVolumeGeometry(self, v):
        '''
        Return the geometry of volume *v* in Fortran-ready units.

        Mapping from stored (x, y) to Barbot (2017) Fortran parameters:
            q1 = y  (northing, km)     -- start of length, center of thickness
            q2 = x  (easting,  km)     -- same horizontal reference point
            q3 = depth (km)            -- TOP of volume; extends to q3+W
            theta = np.deg2rad(strike) -- radians, CW from north

        Returns:
            q1, q2, q3, L, T, W, theta  (all in km / radians)
        '''
        x, y, depth, L, T, W, strike = self.volumes[v]
        # x = easting(km) -> q2, y = northing(km) -> q1
        q1 = y       # northing km – start of L, center of T
        q2 = x       # easting  km – same reference point
        q3 = depth   # top depth km – volume goes from q3 to q3+W
        theta = np.deg2rad(strike)
        return q1, q2, q3, L, T, W, theta

    @property
    def Nvol(self):
        '''Return the number of volumes.'''
        return len(self.volumes)

    def duplicateSource(self):
        '''Return a deep copy.'''
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Strain initialisation
    # ------------------------------------------------------------------

    def initializeStrain(self, strain_components=None, values=None):
        '''
        Initialise the strain array.

        Args:
            * strain_components : list of component names, e.g. ['eps12','eps13'].
                                  Default: ['eps12', 'eps13'].
            * values            : (Nvol, N_strain) initial array, or None for zeros.
        '''
        if strain_components is None:
            strain_components = ['eps12', 'eps13']
        for sc in strain_components:
            if sc not in ALL_STRAIN_COMPONENTS:
                raise ValueError(f"Unknown strain component '{sc}'. "
                                 f"Must be one of {ALL_STRAIN_COMPONENTS}")
        self.strain_components = list(strain_components)
        self.N_strain = len(strain_components)
        Nvol = self.Nvol

        if values is not None:
            self.strain = np.array(values, dtype=float)
            assert self.strain.shape == (Nvol, self.N_strain)
        else:
            self.strain = np.zeros((Nvol, self.N_strain))

    # ------------------------------------------------------------------
    # Green's function computation
    # ------------------------------------------------------------------

    def buildGFs(self, data, vertical=True, strain_components=None,
                 verbose=True):
        '''
        Build Green's functions for a data set.

        For each selected strain component and each volume, compute the
        displacement at data locations due to unit strain.

        Args:
            * data              : Data object (gps, insar, leveling, etc.)
            * vertical          : Use vertical component for GPS
            * strain_components : List of strain component names.
                                  If None, uses self.strain_components.
            * verbose           : Print progress
        '''
        if strain_components is not None:
            self.strain_components = list(strain_components)
            self.N_strain = len(strain_components)
        if self.strain_components is None:
            self.initializeStrain()

        if verbose:
            logger.info('---------------------------------')
            logger.info(f"Building Sbarbot GFs for {data.name} ({data.dtype})")
            logger.info(f"  Strain components: {self.strain_components}")
            logger.info(f"  Volumes: {self.Nvol}")

        Nvol = self.Nvol

        # Observation points (Fortran: x1=north, x2=east, x3=depth)
        obs_x1 = np.asarray(data.y, dtype=np.float64)   # northing km
        obs_x2 = np.asarray(data.x, dtype=np.float64)   # easting km
        Nobs = len(obs_x1)
        obs_x3 = np.zeros(Nobs, dtype=np.float64)       # surface

        # Strain component -> Fortran kwarg mapping
        comp_to_kwarg = {
            'eps11': 'eps11p', 'eps12': 'eps12p', 'eps13': 'eps13p',
            'eps22': 'eps22p', 'eps23': 'eps23p', 'eps33': 'eps33p',
        }

        # Build GFs for each strain component
        G_raw = {}
        for comp in self.strain_components:
            # (3, Nobs, Nvol): E, N, U components
            Gc = np.zeros((3, Nobs, Nvol))

            for v in tqdm(range(Nvol), desc=f'  {comp}', disable=not verbose):
                q1, q2, q3, L, T, W, theta = self.getVolumeGeometry(v)

                # Set this component to 1, others to 0
                eps_kwargs = {k: 0.0 for k in comp_to_kwarg.values()}
                eps_kwargs[comp_to_kwarg[comp]] = 1.0

                u1, u2, u3 = sbarbotfull.displacement(
                    obs_x1, obs_x2, obs_x3,
                    q1, q2, q3, L, T, W, theta,
                    G=self.mu, nu=self.nu,
                    **eps_kwargs
                )

                # Fortran output: u1=north, u2=east, u3=down
                # CSI convention: (E, N, U)
                Gc[0, :, v] = u2     # East
                Gc[1, :, v] = u1     # North
                Gc[2, :, v] = -u3    # Up = -Down

            G_raw[comp] = Gc

        # Format and store
        Gformatted = self._buildGFsdict(data, G_raw, vertical=vertical)
        self.G[data.name] = Gformatted

        if verbose:
            logger.info("  Done.")

        return Gformatted

    def _buildGFsdict(self, data, G_raw, vertical=True):
        '''
        Format raw (3, Nobs, Nvol) GF arrays for the specific data type.

        Args:
            * data    : data instance
            * G_raw   : dict {component_name: (3, Nobs, Nvol)}
            * vertical: Use vertical for GPS

        Returns:
            * G : dict {component_name: (Nd, Nvol)}
        '''
        G = {}

        for comp, Gc in G_raw.items():
            # Gc shape: (3, Nobs, Nvol)
            Ncomp = 3
            if not vertical and data.dtype in ('gps', 'multigps'):
                Gc = Gc[:2, :, :]
                Ncomp = 2

            Nobs = Gc.shape[1]
            Nvol = Gc.shape[2]
            Nd = Ncomp * Nobs

            if data.dtype in ('gps', 'multigps', 'opticorr'):
                # Flat: E(all pts), N(all pts), U(all pts) -> (Nd, Nvol)
                G[comp] = Gc.reshape((Nd, Nvol))

            elif data.dtype in ('insar', 'insartimeseries'):
                # Project onto LOS
                Gc_t = np.transpose(Gc, (1, 0, 2))  # (Nobs, 3, Nvol)
                G[comp] = np.einsum('ij,ijk->ik', data.los, Gc_t)

            elif data.dtype == 'leveling':
                # Vertical (U) component only -> (Nobs, Nvol)
                G[comp] = Gc[2, :, :]

            else:
                raise NotImplementedError(
                    f"Data type '{data.dtype}' not supported in _buildGFsdict")

        return G

    def setGFs(self, data, strain_gfs):
        '''
        Set Green's functions from pre-computed arrays.

        Args:
            * data       : Data object
            * strain_gfs : dict {component_name: (Nd, Nvol) array}
        '''
        self.G[data.name] = {}
        for comp, gf in strain_gfs.items():
            self.G[data.name][comp] = np.array(gf)

        if self.strain_components is None:
            self.strain_components = list(strain_gfs.keys())
            self.N_strain = len(self.strain_components)

    def setCustomGFs(self, data, G_custom):
        '''
        Set custom Green's functions for a data set.

        Args:
            * data     : Data object
            * G_custom : (Nd, Ncustom) array
        '''
        if data.name not in self.G:
            self.G[data.name] = {}
        self.G[data.name]['custom'] = np.array(G_custom)

    # ------------------------------------------------------------------
    # Data vector (d)
    # ------------------------------------------------------------------

    def _setDataVector(self, data, vertical=True):
        '''Set the data vector for a data set, consistent with the GF layout.'''
        if data.dtype in ('gps', 'multigps'):
            ncomp = 3 if vertical else 2
            d = data.vel_enu[:, :ncomp].T.flatten()
            d = d[np.isfinite(d)]
        elif data.dtype in ('insar', 'insartimeseries'):
            d = data.vel
        elif data.dtype == 'leveling':
            d = data.vel
        elif data.dtype == 'crossfaultoffset':
            d = data.data_vector
        else:
            raise NotImplementedError(f"Data type '{data.dtype}' not supported")
        self.d[data.name] = d

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assembleGFs(self, datas, polys=None, strain_components=None,
                    verbose=True, custom=False, computeNormFact=True):
        '''
        Assemble Green's functions into a single G matrix for inversion.

        Column layout:
            [comp1_vol1..comp1_volN | comp2_vol1..comp2_volN | ... | custom | polys]

        Args:
            * datas              : list of data objects (or single)
            * polys              : polynomial / transformation estimators
            * strain_components  : which components to include
            * verbose            : print info
            * custom             : include custom GFs
            * computeNormFact    : recompute normalising factors
        '''
        datas = datas if isinstance(datas, list) else [datas]

        if strain_components is not None:
            self.strain_components = list(strain_components)
            self.N_strain = len(strain_components)

        if verbose:
            logger.info("---------------------------------")
            logger.info("Assembling G for Sbarbot source set {}".format(self.name))

        Nvol = self.Nvol
        comp_list = self.strain_components
        Nps = Nvol * len(comp_list)  # total strain parameters

        self.slipdir = ''.join([c.replace('eps', '') for c in comp_list])

        # Polynomials / transformations
        self.poly = {}
        self.numberofpolys = {}
        self.transform_indices = {}

        if polys is None:
            for data in datas:
                self.poly[data.name] = None
        elif not isinstance(polys, list):
            for data in datas:
                self.poly[data.name] = polys
        else:
            for data, poly in zip(datas, polys):
                self.poly[data.name] = poly

        Npo = 0
        for data in datas:
            transformation = self.poly[data.name]
            if transformation is not None:
                tmpNpo = data.getNumberOfTransformParameters(transformation)
                self.numberofpolys[data.name] = tmpNpo
                Npo += tmpNpo

        # Custom
        Npc = 0
        if custom:
            for data in datas:
                if data.name in self.G and 'custom' in self.G[data.name]:
                    Npc += self.G[data.name]['custom'].shape[1]
        self.NumberCustom = Npc
        self.TransformationParameters = Npo

        Np = Nps + Npc + Npo

        if verbose:
            logger.info(f"  Strain parameters: {Nps} "
                        f"({len(comp_list)} comp x {Nvol} vol)")
            logger.info(f"  Transform parameters: {Npo}")
            if custom:
                logger.info(f"  Custom parameters: {Npc}")
            logger.info(f"  Total parameters: {Np}")

        # Count data
        Nd = 0
        for data in datas:
            self._setDataVector(data)
            Nd += self.d[data.name].shape[0]

        # Allocate
        G = np.zeros((Nd, Np))
        self.datanames = []

        el = 0
        custstart = Nps
        polstart = Nps + Npc

        for data in datas:
            self.datanames.append(data.name)
            if verbose:
                logger.info(f"  Dealing with {data.name} ({data.dtype})")

            Ndlocal = self.d[data.name].shape[0]
            Glocal = np.zeros((Ndlocal, Nps))

            # Fill Glocal: columns for each strain component
            ec = 0
            for comp in comp_list:
                Gc = self.G[data.name][comp]
                Nc = Gc.shape[1]
                Glocal[:, ec:ec + Nc] = Gc
                ec += Nc

            G[el:el + Ndlocal, 0:Nps] = Glocal

            # Custom
            if custom and data.name in self.G and 'custom' in self.G[data.name]:
                nc = self.G[data.name]['custom'].shape[1]
                G[el:el + Ndlocal, custstart:custstart + nc] = self.G[data.name]['custom']
                custstart += nc

            # Polynomials / transformations
            if self.poly[data.name] is not None:
                orb = data.getTransformEstimator(
                    self.poly[data.name],
                    computeNormFact=computeNormFact)
                nc = orb.shape[1]
                G[el:el + Ndlocal, polstart:polstart + nc] = orb
                polstart += nc

            el += Ndlocal

        self.Gassembled = G

    def assembleCd(self, datas, verbose=False):
        '''Assemble data covariance matrices.'''
        datas = datas if isinstance(datas, list) else [datas]
        Cd_blocks = []
        for data in datas:
            if verbose:
                logger.info(f"  Getting Cd for {data.name}")
            Cd_blocks.append(data.Cd)
        self.Cd = block_diag(*Cd_blocks)

    def assembledata(self, datas):
        '''Assemble the data vector d.'''
        datas = datas if isinstance(datas, list) else [datas]
        dvec = []
        for data in datas:
            self._setDataVector(data)
            dvec.append(self.d[data.name])
        self.dassembled = np.concatenate(dvec)

    # ------------------------------------------------------------------
    # Forward modelling
    # ------------------------------------------------------------------

    def buildsynth(self, datas, strain=None):
        '''
        Build synthetic data from strain values.

        Args:
            * datas  : list of data objects
            * strain : (Nvol, N_strain) array, or None to use self.strain
        '''
        datas = datas if isinstance(datas, list) else [datas]

        if strain is None:
            strain = self.strain
        assert strain is not None, "Strain values not set."

        for data in datas:
            Gdata = self.G[data.name]
            Nd = next(iter(Gdata.values())).shape[0]
            synth = np.zeros(Nd)

            for ic, comp in enumerate(self.strain_components):
                synth += Gdata[comp].dot(strain[:, ic])

            # Distribute to data object
            if data.dtype in ('gps', 'multigps'):
                ncomp = 3
                nsta = len(data.x)
                data.synth = synth.reshape((ncomp, nsta)).T
            elif data.dtype in ('insar', 'insartimeseries'):
                data.synth = synth
            elif data.dtype == 'leveling':
                data.synth = synth

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def saveGFs(self, dtype='d', outputDir='.'):
        '''Save Green's functions to binary files.'''
        if self.verbose:
            print(f"Writing GFs to files for {self.name}")
        for dataname in self.G:
            for comp in self.G[dataname]:
                g = self.G[dataname][comp]
                if g is not None:
                    g = g.flatten().astype(dtype)
                    n = self.name.replace(' ', '_')
                    d = dataname.replace(' ', '_')
                    fname = f'{n}_{d}_{comp}.gf'
                    g.tofile(os.path.join(outputDir, fname))

    def loadGFs(self, data, strain_components=None, dtype='d', inputDir='.'):
        '''
        Load Green's functions from binary files.

        Args:
            * data              : Data object
            * strain_components : list of component names to load
            * dtype             : binary dtype
            * inputDir          : directory containing .gf files
        '''
        if strain_components is None:
            strain_components = self.strain_components
        if strain_components is None:
            raise ValueError("strain_components must be specified")

        Nvol = self.Nvol
        self.G[data.name] = {}

        for comp in strain_components:
            n = self.name.replace(' ', '_')
            d = data.name.replace(' ', '_')
            fname = os.path.join(inputDir, f'{n}_{d}_{comp}.gf')
            g = np.fromfile(fname, dtype=dtype)
            ndl = g.size // Nvol
            self.G[data.name][comp] = g.reshape((ndl, Nvol))

        self.strain_components = list(strain_components)
        self.N_strain = len(strain_components)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def getVolumesCenters(self):
        '''
        Return the centres of all volumes in geographic and UTM coords.

        Returns:
            lons, lats, xs, ys, depths
        '''
        if len(self.volumes) == 0:
            return (np.array([]),) * 5
        arr = np.array(self.volumes)
        xs = arr[:, 0]
        ys = arr[:, 1]
        depths = arr[:, 2]
        lons, lats = self.xy2ll(xs, ys)
        return lons, lats, xs, ys, depths

    def plotVolumes(self, strain_component=0, ax=None, show=True, **kwargs):
        '''
        Simple top-view plot of volumes, coloured by strain.

        Args:
            * strain_component : index (int) or name (str)
            * ax    : matplotlib axes
            * show  : call plt.show()
        '''
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        if isinstance(strain_component, str):
            ic = self.strain_components.index(strain_component)
        else:
            ic = strain_component

        vals = self.strain[:, ic] if self.strain is not None else np.zeros(self.Nvol)
        vmax = np.max(np.abs(vals)) if np.any(vals != 0) else 1.0

        for i, vol in enumerate(self.volumes):
            x, y, dep, L, T, W, strike = vol
            rect = Rectangle((-L / 2, -T / 2), L, T,
                             linewidth=0.5, edgecolor='k',
                             facecolor=plt.cm.RdBu_r((vals[i] / vmax + 1) / 2))
            t = Affine2D().rotate_deg(-strike).translate(x, y) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)

        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.set_xlabel('Easting (km)')
        ax.set_ylabel('Northing (km)')
        if show:
            plt.show()
