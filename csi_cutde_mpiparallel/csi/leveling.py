'''
A class that deals with leveling (spirit leveling / precise leveling) data.

Leveling data provides vertical displacement at discrete benchmark locations.
Each observation is a scalar vertical value (uplift positive), with an associated
error.  The Green's function is simply the U (vertical) component evaluated at
each benchmark location – no LOS projection is needed.

Geometry:

    benchmark_i  ----(leveling rod)----  measured dU_i
        (lon_i, lat_i)

Written by kfhe, 2026.
'''

# Externals
import numpy as np
import copy
import os
import sys
import matplotlib.pyplot as plt

# Personals
from .SourceInv import SourceInv


class leveling(SourceInv):
    '''
    A class that handles leveling data (vertical displacements at benchmarks).

    Args:
        * name      : Name of the dataset.

    Kwargs:
        * utmzone   : UTM zone (optional, default=None)
        * lon0      : Longitude of the center of the UTM zone
        * lat0      : Latitude of the center of the UTM zone
        * ellps     : ellipsoid (optional, default='WGS84')
        * verbose   : Speak to me (default=True)
    '''

    def __init__(self, name, utmzone=None, ellps='WGS84',
                 lon0=None, lat0=None, verbose=True):

        # Base class init
        super(leveling, self).__init__(name,
                                       utmzone=utmzone,
                                       ellps=ellps,
                                       lon0=lon0,
                                       lat0=lat0)

        # Set things
        self.dtype = 'leveling'

        if verbose:
            print("---------------------------------")
            print("---------------------------------")
            print("Initialize Leveling data set {}".format(self.name))
        self.verbose = verbose

        # Initialize
        self.station = None          # (n,) benchmark names
        self.lon = None              # (n,) longitude
        self.lat = None              # (n,) latitude
        self.x = None                # (n,) UTM x (km)
        self.y = None                # (n,) UTM y (km)

        self.vel = None              # (n,) vertical displacement
        self.err = None              # (n,) vertical error
        self.synth = None            # (n,) synthetic vertical displacement

        self.Cd = None               # (n,n) data covariance matrix

        self.obs_per_station = 1     # always 1 scalar per benchmark

        # factor for getPolyEstimator (same convention as InSAR)
        self.factor = 1.0

        return

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------
    def set_station_data(self, station, lon, lat, vel, err=None):
        '''
        Set leveling data from arrays.

        Args:
            * station   : (n,) array of benchmark names (str)
            * lon       : (n,) longitude
            * lat       : (n,) latitude
            * vel       : (n,) vertical displacement

        Kwargs:
            * err       : (n,) error; defaults to ones if None
        '''
        self.station = np.array(station)
        self.lon = np.array(lon, dtype=float)
        self.lat = np.array(lat, dtype=float)
        self.x, self.y = self.ll2xy(self.lon, self.lat)
        self.vel = np.array(vel, dtype=float)
        if err is not None:
            self.err = np.array(err, dtype=float)
        else:
            self.err = np.ones_like(self.vel)
        return

    def read_from_ascii(self, filename, header=0):
        '''
        Read leveling data from an ASCII file.

        Format::

            station  lon  lat  vel  [err]

        Lines starting with ``#`` are skipped.

        Args:
            * filename  : path to input file

        Kwargs:
            * header    : number of header lines to skip (default 0)
        '''
        names = []
        data = []
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i < header:
                    continue
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                names.append(parts[0])
                data.append([float(v) for v in parts[1:]])

        data = np.array(data)
        lon = data[:, 0]
        lat = data[:, 1]
        vel = data[:, 2]
        err = data[:, 3] if data.shape[1] > 3 else None
        self.set_station_data(names, lon, lat, vel, err=err)
        return

    def write2file(self, filename, outDir='./', data='data', write_header=True):
        '''
        Write leveling data to an ASCII file.

        Args:
            * filename  : output file name

        Kwargs:
            * outDir    : output directory
            * data      : 'data' or 'synth'
            * write_header : write a header line as the first line (default: True)
        '''
        fout = open(os.path.join(outDir, filename), 'w')

        if data == 'synth':
            vel = self.synth if self.synth is not None else np.zeros_like(self.vel)
        else:
            vel = self.vel

        err = self.err if self.err is not None else np.zeros_like(self.vel)

        if write_header:
            fout.write('# station  lon  lat  vel  err\n')

        for i in range(len(self.station)):
            fout.write('{:s}  {:.6f}  {:.6f}  {:.6f}  {:.6f}\n'.format(
                str(self.station[i]), self.lon[i], self.lat[i],
                vel[i], err[i]))
        fout.close()
        return

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------
    def buildCd(self, sigma=None, lam=None, function='exp', add_prediction=None):
        '''
        Build the data covariance matrix.

        If *sigma* and *lam* are given, builds a spatially correlated Cd
        (same convention as InSAR).  Otherwise builds a diagonal Cd from
        ``self.err``.

        Kwargs:
            * sigma             : standard deviation (scalar)
            * lam               : correlation length (km)
            * function          : 'exp' or 'gauss'
            * add_prediction    : percentage of data values to add
                                  to diagonal (prediction uncertainty)
        '''
        n = self.vel.shape[0]

        if sigma is not None and lam is not None:
            import scipy.spatial.distance as scidis
            dist = scidis.squareform(scidis.pdist(
                np.column_stack([self.x, self.y])))
            if function == 'exp':
                self.Cd = sigma**2 * np.exp(-dist / lam)
            elif function == 'gauss':
                self.Cd = sigma**2 * np.exp(-dist**2 / (2.0 * lam**2))
            else:
                raise ValueError('Unsupported function: {}'.format(function))
            # Replace diagonal with individual errors if available
            if self.err is not None:
                np.fill_diagonal(self.Cd, self.err**2)
        else:
            self.Cd = np.diag(self.err**2)

        if add_prediction is not None:
            self.Cd += np.diag((self.vel * add_prediction / 100.0)**2)

        return

    # ------------------------------------------------------------------
    # Transform / polynomial estimators
    # ------------------------------------------------------------------
    def getNumberOfTransformParameters(self, transformation):
        '''
        Returns the number of transform parameters.

        Args:
            * transformation : int (polynomial order 1/3/4) or str ('strain',
                               'eulerrotation', 'internalstrain') or list thereof.
        '''
        if not isinstance(transformation, list):
            transformation = [transformation]

        Npo = 0
        for t in transformation:
            if isinstance(t, int):
                Npo += t
            elif t == 'strain':
                Npo += 3
            elif t == 'eulerrotation':
                Npo += 3
            elif t == 'internalstrain':
                Npo += 3
            else:
                return 0
        return Npo

    def getTransformEstimator(self, transformation, computeNormFact=True,
                              computeIntStrainNormFact=True, verbose=True):
        '''
        Returns the polynomial / transformation estimator matrix.

        Args:
            * transformation : int, str, or list

        Returns:
            * orb : (n, Npo) design matrix
        '''
        if not isinstance(transformation, list):
            transformation = [transformation]

        n = self.vel.shape[0]
        orb = np.empty((n, 0))

        for t in transformation:
            if isinstance(t, int):
                tmp = self._getPolyEstimator(t, computeNormFact=computeNormFact,
                                             verbose=verbose)
            elif t == 'strain':
                tmp = self._get2DstrainEst(computeNormFact=computeNormFact)
            elif t == 'eulerrotation':
                tmp = self._getEulerMatrix()
            elif t == 'internalstrain':
                tmp = self._getInternalStrain(
                    computeIntStrainNormFact=computeIntStrainNormFact)
            else:
                return None
            orb = np.hstack((orb, tmp))

        return orb

    # --- private poly / strain helpers (mirror InSAR conventions) ---

    def _getPolyEstimator(self, ptype, computeNormFact=True, verbose=True):
        '''
        Polynomial design matrix.  ptype: 1=constant, 3=+linear, 4=+cross.
        '''
        n = self.vel.shape[0]
        orb = np.zeros((n, ptype))
        if ptype > 0:
            orb[:, 0] = 1.0
        if ptype >= 3:
            if computeNormFact:
                self._computeTransformNormFact(verbose=verbose)
            normX = self._normFact['x']
            normY = self._normFact['y']
            x0, y0 = self._normFact['ref']
            orb[:, 1] = (self.x - x0) / normX
            orb[:, 2] = (self.y - y0) / normY
        if ptype >= 4:
            orb[:, 3] = orb[:, 1] * orb[:, 2]
        orb *= self.factor
        return orb

    def _computeTransformNormFact(self, verbose=True):
        '''Compute normalizing factors for polynomial estimator.'''
        x0 = 0.5 * (self.x.max() + self.x.min())
        y0 = 0.5 * (self.y.max() + self.y.min())
        normX = max(self.x.max() - self.x.min(), 1e-10)
        normY = max(self.y.max() - self.y.min(), 1e-10)
        self._normFact = {'x': normX, 'y': normY, 'ref': (x0, y0)}
        return

    def _get2DstrainEst(self, computeNormFact=True):
        '''Simple 2-D strain estimator (3 params: exx, eyy, exy).'''
        if computeNormFact:
            self._computeTransformNormFact()
        x0, y0 = self._normFact['ref']
        normX = self._normFact['x']
        normY = self._normFact['y']
        n = self.vel.shape[0]
        orb = np.zeros((n, 3))
        orb[:, 0] = (self.x - x0) / normX
        orb[:, 1] = (self.y - y0) / normY
        orb[:, 2] = orb[:, 0] * orb[:, 1]
        orb *= self.factor
        return orb

    def _getEulerMatrix(self):
        '''Euler rotation estimator (3 params).'''
        n = self.vel.shape[0]
        orb = np.zeros((n, 3))
        orb[:, 0] = self.y * 1e3   # dy contributes to vertical via curvature
        orb[:, 1] = self.x * 1e3
        orb[:, 2] = 1.0            # constant offset
        return orb

    def _getInternalStrain(self, computeIntStrainNormFact=True):
        '''Internal strain estimator (3 params).'''
        if computeIntStrainNormFact:
            self._computeTransformNormFact()
        return self._get2DstrainEst(computeNormFact=False)

    # ------------------------------------------------------------------
    # GF interface
    # ------------------------------------------------------------------
    def setGFsInFault(self, fault, G, vertical=True):
        '''
        Set the Green's functions into the fault object.

        For leveling, the GF is already a (n, Np) matrix of pure vertical
        response, so we wrap each key in a single-element list (same convention
        as InSAR).

        Args:
            * fault     : Fault instance
            * G         : dictionary with keys 'strikeslip', 'dipslip', etc.
        '''
        if fault.type == "Fault":
            fault.setGFs(self,
                         strikeslip=[G.get('strikeslip', None)],
                         dipslip=[G.get('dipslip', None)],
                         tensile=[G.get('tensile', None)],
                         coupling=[G.get('coupling', None)],
                         vertical=True)
        return

    # ------------------------------------------------------------------
    # Forward modelling
    # ------------------------------------------------------------------
    def buildsynth(self, faults, direction='sd', poly=None, vertical=True,
                   custom=False, computeNormFact=True,
                   computeIntStrainNormFact=True):
        '''
        Compute synthetic vertical displacements from fault slip models.

        Args:
            * faults    : Fault or list of Fault instances

        Kwargs:
            * direction         : slip components ('s', 'd', 't', 'c')
            * poly              : 'build'/'include', int, str or list
            * vertical          : ignored (leveling is always vertical)
            * custom            : include custom GFs
            * computeNormFact   : recompute normalization factors
        '''
        if not isinstance(faults, list):
            faults = [faults]

        n = self.vel.shape[0]
        self.synth = np.zeros(n)

        for fault in faults:
            if fault.type != "Fault":
                continue
            G = fault.G[self.name]

            if 's' in direction and G.get('strikeslip') is not None:
                self.synth += G['strikeslip'].dot(fault.slip[:, 0])
            if 'd' in direction and G.get('dipslip') is not None:
                self.synth += G['dipslip'].dot(fault.slip[:, 1])
            if 't' in direction and G.get('tensile') is not None:
                self.synth += G['tensile'].dot(fault.slip[:, 2])
            if 'c' in direction and G.get('coupling') is not None:
                self.synth += G['coupling'].dot(fault.coupling)

            if custom and G.get('custom') is not None:
                self.synth += G['custom'].dot(fault.custom[self.name])

            # Polynomial / transformation correction
            if poly in ('build', 'include') and self.name in fault.poly:
                ref = fault.poly[self.name]
                if isinstance(ref, int):
                    self._computePoly(fault, computeNormFact=computeNormFact)
                    self.synth += self._orbit
                elif isinstance(ref, str):
                    if ref == 'strain':
                        self._compute2Dstrain(fault, computeNormFact=computeNormFact)
                        self.synth += self._strainCorr
                    elif ref == 'internalstrain':
                        self._computeInternalStrain(
                            fault,
                            computeIntStrainNormFact=computeIntStrainNormFact)
                        self.synth += self._intStrainCorr
                elif isinstance(ref, list):
                    self._computeTransformation(
                        fault, computeNormFact=computeNormFact,
                        computeIntStrainNormFact=computeIntStrainNormFact)
                    self.synth += self._transformation

        return

    # --- private poly forward helpers ---

    def _computePoly(self, fault, computeNormFact=True):
        ref = fault.poly[self.name]
        orb = self._getPolyEstimator(ref, computeNormFact=computeNormFact,
                                     verbose=False)
        self._orbit = orb.dot(fault.polysol[self.name])

    def _compute2Dstrain(self, fault, computeNormFact=True):
        orb = self._get2DstrainEst(computeNormFact=computeNormFact)
        self._strainCorr = orb.dot(fault.polysol[self.name])

    def _computeInternalStrain(self, fault, computeIntStrainNormFact=True):
        orb = self._getInternalStrain(
            computeIntStrainNormFact=computeIntStrainNormFact)
        self._intStrainCorr = orb.dot(fault.polysol[self.name])

    def _computeTransformation(self, fault, computeNormFact=True,
                               computeIntStrainNormFact=True):
        orb = self.getTransformEstimator(
            fault.poly[self.name],
            computeNormFact=computeNormFact,
            computeIntStrainNormFact=computeIntStrainNormFact)
        self._transformation = orb.dot(fault.polysol[self.name])

    # ------------------------------------------------------------------
    def removeSynth(self, faults, direction='sd', poly=None, custom=False,
                    computeNormFact=True, computeIntStrainNormFact=True):
        '''
        Remove synthetic from observed data.
        '''
        self.buildsynth(faults, direction=direction, poly=poly, custom=custom,
                        computeNormFact=computeNormFact,
                        computeIntStrainNormFact=computeIntStrainNormFact)
        self.vel -= self.synth
        return

    # ------------------------------------------------------------------
    def getRMS(self):
        '''
        Compute root-mean-square of residual (vel - synth).
        '''
        if self.synth is not None:
            return np.sqrt(np.mean((self.vel - self.synth)**2))
        return np.sqrt(np.mean(self.vel**2))

    # ------------------------------------------------------------------
    # Reject / select benchmarks
    # ------------------------------------------------------------------
    def select_stations(self, minlon, maxlon, minlat, maxlat):
        '''
        Select benchmarks inside a lon/lat bounding box.

        Args:
            * minlon : Minimum longitude.
            * maxlon : Maximum longitude.
            * minlat : Minimum latitude.
            * maxlat : Maximum latitude.

        Returns:
            * None
        '''

        # Store the corners
        self.minlon = minlon
        self.maxlon = maxlon
        self.minlat = minlat
        self.maxlat = maxlat

        # Select on latitude and longitude
        u = np.flatnonzero((self.lat > minlat) & (self.lat < maxlat)
                           & (self.lon > minlon) & (self.lon < maxlon))

        # Select the stations
        self.station = self.station[u]
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.vel = self.vel[u]
        if self.err is not None:
            self.err = self.err[u]
        if self.synth is not None:
            self.synth = self.synth[u]
        if self.Cd is not None:
            self.Cd = self.Cd[np.ix_(u, u)]

        # All done
        return

    def reject(self, u):
        '''
        Reject benchmarks by index.

        Args:
            * u : index or boolean mask of benchmarks to reject
        '''
        if isinstance(u, (list, np.ndarray)):
            keep = np.ones(len(self.station), dtype=bool)
            keep[u] = False
        else:
            keep = np.ones(len(self.station), dtype=bool)
            keep[u] = False

        self.station = self.station[keep]
        self.lon = self.lon[keep]
        self.lat = self.lat[keep]
        self.x = self.x[keep]
        self.y = self.y[keep]
        self.vel = self.vel[keep]
        if self.err is not None:
            self.err = self.err[keep]
        if self.synth is not None:
            self.synth = self.synth[keep]
        if self.Cd is not None:
            self.Cd = self.Cd[np.ix_(keep, keep)]
        return

    # ------------------------------------------------------------------
    # PlotStyle helper
    # ------------------------------------------------------------------
    @staticmethod
    def _get_plot_style_context(style, style_kwargs, figsize='double',
                                figsize_aspect=0.75):
        '''
        Build a PlotStyle context manager with unified figsize handling.

        Returns (context_manager, resolved_figsize):
          - PlotStyle available → resolved_figsize is None (rcParams handles it)
          - PlotStyle missing   → resolved_figsize is a (w, h) tuple fallback
        '''
        import contextlib

        _FALLBACK_WIDTHS = {
            'single': 3.54, 'double': 7.08, 'nature': 3.54,
            'ieee': 3.5, 'a4': 6.3,
        }

        def _resolve_fallback(fs, aspect):
            if fs is None:
                return None
            if isinstance(fs, (list, tuple)):
                return tuple(fs)
            elif isinstance(fs, str):
                w = _FALLBACK_WIDTHS.get(fs, 7.08)
                return (w, w * aspect)
            elif isinstance(fs, (int, float)):
                return (float(fs), float(fs) * aspect)
            return None

        if style is None:
            return contextlib.nullcontext(), _resolve_fallback(figsize, figsize_aspect)

        try:
            from eqtools.viztools import PlotStyle
            kw = dict(style_kwargs or {})
            if figsize is not None and 'figsize' not in kw:
                kw['figsize'] = figsize
            if figsize_aspect is not None and 'figsize_aspect' not in kw:
                kw['figsize_aspect'] = figsize_aspect
            return PlotStyle(style, **kw), None
        except Exception:
            return contextlib.nullcontext(), _resolve_fallback(figsize, figsize_aspect)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self, show=True, savefig=None, axis='index',
             residuals=True, ylim=None, unit_label='m',
             figsize='double', figsize_aspect=0.4,
             style='science', style_kwargs=None):
        '''
        Plot leveling data, synthetics and residuals.

        Kwargs:
            * show          : call plt.show()
            * savefig       : file path to save
            * axis          : 'index', 'lon', 'lat', 'cumdist'
            * residuals     : overlay residual curve if synth exists
            * ylim          : (ymin, ymax) for vertical axis
            * unit_label    : y-axis unit label
            * figsize       : PlotStyle figsize — 'single', 'double',
                              scalar width, or (w, h) tuple.  Default 'double'.
            * figsize_aspect: height/width ratio (default 0.4 for wide profile)
            * style         : PlotStyle preset (default 'science'); None to disable
            * style_kwargs  : extra kwargs forwarded to PlotStyle
                              (e.g. fontsize, usetex, dpi, rcparams, ...)
        '''
        if self.vel is None:
            print('No data to plot')
            return

        ctx, fs = self._get_plot_style_context(
            style, style_kwargs, figsize=figsize, figsize_aspect=figsize_aspect)

        with ctx:
            # X-axis
            if axis == 'lon':
                xvals = self.lon
                xlabel = 'Longitude (°)'
            elif axis == 'lat':
                xvals = self.lat
                xlabel = 'Latitude (°)'
            elif axis == 'cumdist':
                dx = np.diff(self.x)
                dy = np.diff(self.y)
                seg = np.sqrt(dx**2 + dy**2)
                xvals = np.concatenate([[0.0], np.cumsum(seg)])
                xlabel = 'Distance along profile (km)'
            else:
                xvals = np.arange(len(self.station))
                xlabel = 'Benchmark index'

            fig, ax = plt.subplots(1, 1, figsize=fs)
            ax.errorbar(xvals, self.vel,
                        yerr=self.err if self.err is not None else None,
                        fmt='o', markersize=4, color='steelblue',
                        capsize=2, elinewidth=0.8, label='Data')

            if self.synth is not None:
                ax.plot(xvals, self.synth, 's-', markersize=3, linewidth=1.0,
                        color='crimson', label='Synth')
                if residuals:
                    resid = self.vel - self.synth
                    ax.plot(xvals, resid, '--', linewidth=1.0,
                            color='darkorange', label='Residual')
                    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')

            ax.set_xlabel(xlabel)
            ax.set_ylabel('Vertical displacement ({})'.format(unit_label))
            ax.set_title('Leveling: {}'.format(self.name))
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.25)
            plt.tight_layout()

            if savefig is not None:
                fig.savefig(savefig, dpi=150, bbox_inches='tight')
            if show:
                plt.show()

            return fig

    def plotMap(self, show=True, savefig=None,
               cmap='RdBu_r', vmin=None, vmax=None,
               figsize='double', figsize_aspect=0.75,
               style='science', style_kwargs=None):
        '''
        Plot benchmark locations colored by vertical displacement.

        Kwargs:
            * figsize       : 'single', 'double', scalar, or (w,h). Default 'double'.
            * figsize_aspect: height/width ratio (default 0.75)
            * style         : PlotStyle preset (default 'science'); None to disable
            * style_kwargs  : extra kwargs forwarded to PlotStyle
        '''
        if self.vel is None:
            print('No data to plot')
            return

        ctx, fs = self._get_plot_style_context(
            style, style_kwargs, figsize=figsize, figsize_aspect=figsize_aspect)

        with ctx:
            fig, ax = plt.subplots(1, 1, figsize=fs)
            sc = ax.scatter(self.lon, self.lat, c=self.vel,
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            s=30, edgecolors='k', linewidths=0.3)
            plt.colorbar(sc, ax=ax, label='Vertical displacement')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Leveling: {} (map)'.format(self.name))
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            plt.tight_layout()

            if savefig is not None:
                fig.savefig(savefig, dpi=150, bbox_inches='tight')
            if show:
                plt.show()

            return fig
