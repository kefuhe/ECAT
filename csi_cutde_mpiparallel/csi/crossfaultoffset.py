'''
A class that deals with cross-fault offset data.

Cross-fault offset data consists of point pairs (one on each side of the fault)
with differential displacement measurements projected onto fault-parallel and
fault-perpendicular directions.

Geometry (plan view, example: strike = 0 deg, fault striking North):

              ^ fp  (fault-parallel)
              |     = (-sin s,  cos s) in (E, N)
              |                          N
  side1 ------+------ side2         fp  ^    ^ strike direction
 (x1,y1)  ref (x,y) (x2,y2)             |   /
              |                         |  /  fault trace
              +-------->               -+-
              fn  (fault-perpendicular) |
                  = ( cos s,  sin s)
                    in (E, N)

  sign convention (differential displacement d2 - d1):
      fault_parallel      = (d_side2 - d_side1) . fp
      fault_perpendicular = (d_side2 - d_side1) . fn

  For strike = 0 deg (N-S fault):
      side1 = WEST,  side2 = EAST
      fp points NORTH,  fn points EAST

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


class crossfaultoffset(SourceInv):
    '''
    A class that handles cross-fault offset data.

    Each data point is a pair of observation points on opposite sides of the fault.
    The displacement difference (side2 - side1) is projected onto fault-parallel and
    fault-perpendicular directions.

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
        super(crossfaultoffset, self).__init__(name,
                                               utmzone=utmzone,
                                               ellps=ellps,
                                               lon0=lon0,
                                               lat0=lat0)

        # Set things
        self.dtype = 'crossfaultoffset'

        # print
        if verbose:
            print("---------------------------------")
            print("---------------------------------")
            print("Initialize Cross-Fault Offset data set {}".format(self.name))
        self.verbose = verbose

        # Initialize things
        self.station = None          # (n,) station/pair names
        self.lon = None              # (n,) reference point lon (midpoint)
        self.lat = None              # (n,) reference point lat (midpoint)
        self.x = None                # (n,) reference point UTM x
        self.y = None                # (n,) reference point UTM y

        self.lon1 = None             # (n,) side 1 lon
        self.lat1 = None             # (n,) side 1 lat
        self.x1 = None               # (n,) side 1 UTM x
        self.y1 = None               # (n,) side 1 UTM y

        self.lon2 = None             # (n,) side 2 lon
        self.lat2 = None             # (n,) side 2 lat
        self.x2 = None               # (n,) side 2 UTM x
        self.y2 = None               # (n,) side 2 UTM y

        self.strike = None           # (n,) local fault strike in radians

        self.fault_parallel = None       # (n,) fault-parallel offset data
        self.fault_perpendicular = None  # (n,) fault-perpendicular offset data
        self.fault_vertical = None       # (n,) cross-fault vertical offset data
        self.err_parallel = None         # (n,) fault-parallel errors
        self.err_perpendicular = None    # (n,) fault-perpendicular errors
        self.err_vertical = None         # (n,) cross-fault vertical errors

        self.synth_parallel = None       # (n,) synthetic fault-parallel
        self.synth_perpendicular = None  # (n,) synthetic fault-perpendicular
        self.synth_vertical = None       # (n,) synthetic cross-fault vertical
        self.synth = None                # combined synth vector

        self.Cd = None

        # All done
        return

    @property
    def obs_per_station(self):
        '''
        Number of observations per station pair (1, 2 or 3).
        '''
        n = 0
        if self.fault_parallel is not None:
            n += 1
        if self.fault_perpendicular is not None:
            n += 1
        if self.fault_vertical is not None:
            n += 1
        return n

    @obs_per_station.setter
    def obs_per_station(self, value):
        # Allow external setting for compatibility with Fault.setGFs
        self._obs_per_station_override = value

    @property
    def data_vector(self):
        '''
        Assemble the data into a single 1D vector.
         Order: fault_parallel (if present), then fault_perpendicular (if present),
             then fault_vertical (if present).
        '''
        parts = []
        if self.fault_parallel is not None:
            parts.append(self.fault_parallel)
        if self.fault_perpendicular is not None:
            parts.append(self.fault_perpendicular)
        if self.fault_vertical is not None:
            parts.append(self.fault_vertical)
        if len(parts) == 0:
            return np.array([])
        return np.concatenate(parts)

    @property
    def synth_vector(self):
        '''
        Assemble the synthetic data into a single 1D vector.
        '''
        parts = []
        if self.synth_parallel is not None:
            parts.append(self.synth_parallel)
        if self.synth_perpendicular is not None:
            parts.append(self.synth_perpendicular)
        if self.synth_vertical is not None:
            parts.append(self.synth_vertical)
        if len(parts) == 0:
            return np.array([])
        return np.concatenate(parts)

    def set_from_point_pairs(self, station, lon1, lat1, lon2, lat2,
                              strike, fault_parallel=None,
                              fault_perpendicular=None,
                              fault_vertical=None,
                              err_parallel=None, err_perpendicular=None,
                              err_vertical=None):
        '''
        Set data from directly specified point pair coordinates.

        Args:
            * station           : (n,) array of station/pair names
            * lon1, lat1        : (n,) side 1 (e.g., south/west side) coordinates
            * lon2, lat2        : (n,) side 2 (e.g., north/east side) coordinates
            * strike            : (n,) local fault strike in degrees (clockwise from N)

        Kwargs:
            * fault_parallel        : (n,) fault-parallel offset
            * fault_perpendicular   : (n,) fault-perpendicular offset
            * fault_vertical        : (n,) vertical offset difference (side2 - side1)
            * err_parallel          : (n,) fault-parallel error
            * err_perpendicular     : (n,) fault-perpendicular error
            * err_vertical          : (n,) fault-vertical error
        '''

        # Store names
        self.station = np.array(station)
        n = len(self.station)

        # Store coordinates
        self.lon1 = np.array(lon1, dtype=float)
        self.lat1 = np.array(lat1, dtype=float)
        self.lon2 = np.array(lon2, dtype=float)
        self.lat2 = np.array(lat2, dtype=float)

        # Convert to UTM
        self.x1, self.y1 = self.ll2xy(self.lon1, self.lat1)
        self.x2, self.y2 = self.ll2xy(self.lon2, self.lat2)

        # Reference point as midpoint
        self.lon = 0.5 * (self.lon1 + self.lon2)
        self.lat = 0.5 * (self.lat1 + self.lat2)
        self.x, self.y = self.ll2xy(self.lon, self.lat)

        # Strike in radians
        self.strike = np.array(strike, dtype=float) * np.pi / 180.

        # Data
        if fault_parallel is not None:
            self.fault_parallel = np.array(fault_parallel, dtype=float)
        if fault_perpendicular is not None:
            self.fault_perpendicular = np.array(fault_perpendicular, dtype=float)
        if fault_vertical is not None:
            self.fault_vertical = np.array(fault_vertical, dtype=float)

        # Errors
        if err_parallel is not None:
            self.err_parallel = np.array(err_parallel, dtype=float)
        elif self.fault_parallel is not None:
            self.err_parallel = np.ones(n) * 1.0
        if err_perpendicular is not None:
            self.err_perpendicular = np.array(err_perpendicular, dtype=float)
        elif self.fault_perpendicular is not None:
            self.err_perpendicular = np.ones(n) * 1.0
        if err_vertical is not None:
            self.err_vertical = np.array(err_vertical, dtype=float)
        elif self.fault_vertical is not None:
            self.err_vertical = np.ones(n) * 1.0

        # All done
        return

    def set_from_fault_points(self, station, lon, lat, width, strike,
                               fault_parallel=None, fault_perpendicular=None,
                               fault_vertical=None,
                               err_parallel=None, err_perpendicular=None,
                               err_vertical=None):
        '''
        Set data from fault reference points with symmetric offset width.

        Points on side 1 and side 2 are computed by offsetting perpendicularly
        from the fault reference point by +/- width.  Side 1 is to the LEFT,
        side 2 is to the RIGHT, when looking along the strike direction.

        Plan view (example: strike = 45 deg, fault striking NE):

                         NE (along-strike)
                        ^
                       /  fault trace
                      /
          side1 ---- ref ---- side2
         (left)   (lon,lat)  (right)
          -width              +width
                  along perp direction = (cos s, -sin s) in (E, N)

        The perpendicular direction (toward side2) is 90 deg clockwise from
        the strike direction::

            strike direction  = (sin s,  cos s) in (E, N)
            perp    (right)   = (cos s, -sin s) in (E, N)   [= rotate CW 90 deg]

            side1 UTM = ref_UTM - width * (cos s, -sin s)   [left of strike]
            side2 UTM = ref_UTM + width * (cos s, -sin s)   [right of strike]

        Args:
            * station       : (n,) array of station/pair names
            * lon, lat      : (n,) fault reference point coordinates
            * width         : (n,) or scalar, symmetric offset width in km
            * strike        : (n,) local fault strike in degrees (clockwise from N)

        Kwargs:
            * fault_parallel        : (n,) fault-parallel offset
            * fault_perpendicular   : (n,) fault-perpendicular offset
            * fault_vertical        : (n,) vertical offset difference (side2 - side1)
            * err_parallel          : (n,) fault-parallel error
            * err_perpendicular     : (n,) fault-perpendicular error
            * err_vertical          : (n,) fault-vertical error
        '''

        station = np.array(station)
        lon = np.array(lon, dtype=float)
        lat = np.array(lat, dtype=float)
        strike_deg = np.array(strike, dtype=float)
        width = np.atleast_1d(np.array(width, dtype=float))
        if width.size == 1:
            width = np.full(len(station), width[0])

        # Convert to UTM
        x_ref, y_ref = self.ll2xy(lon, lat)
        strike_rad = strike_deg * np.pi / 180.

        # Perpendicular direction: strike + 90 deg (to the right looking along strike)
        # Side 1: left of strike (strike - 90)
        # Side 2: right of strike (strike + 90)
        perp_x = np.cos(strike_rad)   # perpendicular to strike (East component)
        perp_y = -np.sin(strike_rad)  # perpendicular to strike (North component)
        # Note: strike is measured CW from N, so strike direction vector is (sin(strike), cos(strike))
        # Perpendicular (to the right of strike) is (cos(strike), -sin(strike))

        x1 = x_ref - width * perp_x
        y1 = y_ref - width * perp_y
        x2 = x_ref + width * perp_x
        y2 = y_ref + width * perp_y

        lon1, lat1 = self.xy2ll(x1, y1)
        lon2, lat2 = self.xy2ll(x2, y2)

        self.set_from_point_pairs(station, lon1, lat1, lon2, lat2, strike_deg,
                                   fault_parallel=fault_parallel,
                                   fault_perpendicular=fault_perpendicular,
                                   fault_vertical=fault_vertical,
                                   err_parallel=err_parallel,
                                   err_perpendicular=err_perpendicular,
                                   err_vertical=err_vertical)

        # All done
        return

    def set_from_fault_points_asymmetric(self, station, lon, lat,
                                          width1, width2, strike,
                                          fault_parallel=None,
                                          fault_perpendicular=None,
                                          fault_vertical=None,
                                          err_parallel=None,
                                          err_perpendicular=None,
                                          err_vertical=None):
        '''
        Set data from fault reference points with asymmetric offset widths.

        Args:
            * station       : (n,) array of station/pair names
            * lon, lat      : (n,) fault reference point coordinates
            * width1        : (n,) or scalar, offset width for side 1 (left of strike)
            * width2        : (n,) or scalar, offset width for side 2 (right of strike)
            * strike        : (n,) local fault strike in degrees

        Kwargs:
            * fault_parallel        : (n,) fault-parallel offset
            * fault_perpendicular   : (n,) fault-perpendicular offset
            * fault_vertical        : (n,) vertical offset difference (side2 - side1)
            * err_parallel          : (n,) fault-parallel error
            * err_perpendicular     : (n,) fault-perpendicular error
            * err_vertical          : (n,) fault-vertical error
        '''

        station = np.array(station)
        lon = np.array(lon, dtype=float)
        lat = np.array(lat, dtype=float)
        strike_deg = np.array(strike, dtype=float)
        width1 = np.atleast_1d(np.array(width1, dtype=float))
        width2 = np.atleast_1d(np.array(width2, dtype=float))
        if width1.size == 1:
            width1 = np.full(len(station), width1[0])
        if width2.size == 1:
            width2 = np.full(len(station), width2[0])

        # Convert to UTM
        x_ref, y_ref = self.ll2xy(lon, lat)
        strike_rad = strike_deg * np.pi / 180.

        # Perpendicular direction (to the right of strike)
        perp_x = np.cos(strike_rad)
        perp_y = -np.sin(strike_rad)

        x1 = x_ref - width1 * perp_x
        y1 = y_ref - width1 * perp_y
        x2 = x_ref + width2 * perp_x
        y2 = y_ref + width2 * perp_y

        lon1, lat1 = self.xy2ll(x1, y1)
        lon2, lat2 = self.xy2ll(x2, y2)

        self.set_from_point_pairs(station, lon1, lat1, lon2, lat2, strike_deg,
                                   fault_parallel=fault_parallel,
                                   fault_perpendicular=fault_perpendicular,
                                   fault_vertical=fault_vertical,
                                   err_parallel=err_parallel,
                                   err_perpendicular=err_perpendicular,
                                   err_vertical=err_vertical)

        # All done
        return

    def read_from_ascii(self, filename, header=0, fmt='point_pairs'):
        '''
        Read cross-fault offset data from an ASCII file.

        File format for fmt='point_pairs':
            station lon1 lat1 lon2 lat2 strike fp [fp_err] [fn] [fn_err] [fv] [fv_err]

        File format for fmt='fault_points':
            station lon lat width strike fp [fp_err] [fn] [fn_err] [fv] [fv_err]

        File format for fmt='fault_points_asym':
            station lon lat width1 width2 strike fp [fp_err] [fn] [fn_err] [fv] [fv_err]

        Args:
            * filename  : Input file path

        Kwargs:
            * header    : Number of header lines to skip
            * fmt       : 'point_pairs', 'fault_points', or 'fault_points_asym'
        '''

        assert os.path.exists(filename), 'Cannot find file {}'.format(filename)

        # Read lines
        with open(filename, 'r') as f:
            lines = f.readlines()[header:]

        # Parse
        station = []
        cols = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                continue
            items = line.split()
            station.append(items[0])
            cols.append([float(x) for x in items[1:]])

        station = np.array(station)
        cols = np.array(cols)

        if fmt == 'point_pairs':
            # station lon1 lat1 lon2 lat2 strike fp [fp_err] [fn] [fn_err]
            lon1, lat1, lon2, lat2, strike = cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3], cols[:, 4]
            fp = cols[:, 5] if cols.shape[1] > 5 else None
            fp_err = cols[:, 6] if cols.shape[1] > 6 else None
            fn = cols[:, 7] if cols.shape[1] > 7 else None
            fn_err = cols[:, 8] if cols.shape[1] > 8 else None
            fv = cols[:, 9] if cols.shape[1] > 9 else None
            fv_err = cols[:, 10] if cols.shape[1] > 10 else None
            self.set_from_point_pairs(station, lon1, lat1, lon2, lat2, strike,
                                       fault_parallel=fp,
                                       fault_perpendicular=fn,
                                       fault_vertical=fv,
                                       err_parallel=fp_err,
                                       err_perpendicular=fn_err,
                                       err_vertical=fv_err)

        elif fmt == 'fault_points':
            # station lon lat width strike fp [fp_err] [fn] [fn_err]
            lon, lat, width, strike = cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3]
            fp = cols[:, 4] if cols.shape[1] > 4 else None
            fp_err = cols[:, 5] if cols.shape[1] > 5 else None
            fn = cols[:, 6] if cols.shape[1] > 6 else None
            fn_err = cols[:, 7] if cols.shape[1] > 7 else None
            fv = cols[:, 8] if cols.shape[1] > 8 else None
            fv_err = cols[:, 9] if cols.shape[1] > 9 else None
            self.set_from_fault_points(station, lon, lat, width, strike,
                                        fault_parallel=fp,
                                        fault_perpendicular=fn,
                                        fault_vertical=fv,
                                        err_parallel=fp_err,
                                        err_perpendicular=fn_err,
                                        err_vertical=fv_err)

        elif fmt == 'fault_points_asym':
            # station lon lat width1 width2 strike fp [fp_err] [fn] [fn_err]
            lon, lat, w1, w2, strike = cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3], cols[:, 4]
            fp = cols[:, 5] if cols.shape[1] > 5 else None
            fp_err = cols[:, 6] if cols.shape[1] > 6 else None
            fn = cols[:, 7] if cols.shape[1] > 7 else None
            fn_err = cols[:, 8] if cols.shape[1] > 8 else None
            fv = cols[:, 9] if cols.shape[1] > 9 else None
            fv_err = cols[:, 10] if cols.shape[1] > 10 else None
            self.set_from_fault_points_asymmetric(station, lon, lat, w1, w2, strike,
                                                    fault_parallel=fp,
                                                    fault_perpendicular=fn,
                                                    fault_vertical=fv,
                                                    err_parallel=fp_err,
                                                    err_perpendicular=fn_err,
                                                    err_vertical=fv_err)

        # All done
        return

    def write2file(self, filename, outDir='./', data='data'):
        '''
        Write cross-fault offset data to file.

        Args:
            * filename  : Output file name

        Kwargs:
            * outDir    : Output directory
            * data      : 'data' or 'synth'
        '''

        filepath = os.path.join(outDir, filename)

        if data == 'data':
            fp = self.fault_parallel
            fn = self.fault_perpendicular
            fv = self.fault_vertical
        elif data == 'synth':
            fp = self.synth_parallel
            fn = self.synth_perpendicular
            fv = self.synth_vertical
        else:
            raise ValueError("data must be 'data' or 'synth'")

        with open(filepath, 'w') as f:
            f.write('# station lon1 lat1 lon2 lat2 strike_deg')
            if fp is not None:
                f.write(' fault_parallel err_parallel')
            if fn is not None:
                f.write(' fault_perpendicular err_perpendicular')
            if fv is not None:
                f.write(' fault_vertical err_vertical')
            f.write('\n')

            for i in range(len(self.station)):
                line = '{} {:.6f} {:.6f} {:.6f} {:.6f} {:.4f}'.format(
                    self.station[i],
                    self.lon1[i], self.lat1[i],
                    self.lon2[i], self.lat2[i],
                    np.degrees(self.strike[i]))
                if fp is not None:
                    err = self.err_parallel[i] if self.err_parallel is not None else 0.0
                    line += ' {:.6f} {:.6f}'.format(fp[i], err)
                if fn is not None:
                    err = self.err_perpendicular[i] if self.err_perpendicular is not None else 0.0
                    line += ' {:.6f} {:.6f}'.format(fn[i], err)
                if fv is not None:
                    err = self.err_vertical[i] if self.err_vertical is not None else 0.0
                    line += ' {:.6f} {:.6f}'.format(fv[i], err)
                f.write(line + '\n')

        # All done
        return

    def buildCd(self, add_prediction=None):
        '''
        Builds a diagonal data covariance matrix from the uncertainties.

        Kwargs:
            * add_prediction : percentage of data to add to diagonal

        Returns:
            * None
        '''

        errs = []
        if self.fault_parallel is not None:
            assert self.err_parallel is not None, 'Need errors for fault_parallel'
            errs.append(self.err_parallel)
        if self.fault_perpendicular is not None:
            assert self.err_perpendicular is not None, 'Need errors for fault_perpendicular'
            errs.append(self.err_perpendicular)
        if self.fault_vertical is not None:
            assert self.err_vertical is not None, 'Need errors for fault_vertical'
            errs.append(self.err_vertical)

        err = np.concatenate(errs)
        self.Cd = np.diag(err ** 2)

        if add_prediction is not None:
            d = self.data_vector
            self.Cd += np.diag((d * add_prediction / 100.) ** 2)

        # All done
        return

    def getNumberOfTransformParameters(self, transformation):
        '''
        Returns the number of transform parameters.

        For crossfaultoffset, integer polynomial transformations are expanded
        independently for each observed component.

        Args:
            * transformation : int (1 for constant, etc.)

        Returns:
            * Integer
        '''

        if type(transformation) is list:
            transformation = transformation

        if isinstance(transformation, (str, type(None), int, np.integer)):
            transformation = [transformation]

        Npo = 0
        for itransformation in transformation:
            if isinstance(itransformation, (int, np.integer)):
                Npo += itransformation * self.obs_per_station
            else:
                return 0
        return Npo

    def getTransformEstimator(self, transformation, computeNormFact=True,
                               computeIntStrainNormFact=True, verbose=True):
        '''
        Returns the estimator for the transformation.

        For crossfaultoffset, supports integer polynomial types (1, 3, 4).
        Polynomial is applied per component.

        Args:
            * transformation : int or list

        Returns:
            * 2d array
        '''

        if type(transformation) is list:
            transformation = transformation

        if isinstance(transformation, (str, type(None), int, np.integer)):
            transformation = [transformation]

        n = len(self.station)
        n_comp = self.obs_per_station
        nd = n_comp * n

        orb = np.empty((nd, 0))
        for itransformation in transformation:
            if isinstance(itransformation, (int, np.integer)):
                # Build polynomial estimator per component
                tmporb = np.zeros((nd, itransformation * n_comp))
                if itransformation >= 3:
                    if computeNormFact:
                        x0 = np.mean(self.x)
                        y0 = np.mean(self.y)
                        normX = np.abs(self.x - x0).max()
                        normY = np.abs(self.y - y0).max()
                        if normX == 0.:
                            normX = 1.
                        if normY == 0.:
                            normY = 1.
                        self.TransformNormalizingFactor = {
                            'ref': [x0, y0],
                            'x': normX,
                            'y': normY,
                        }
                    else:
                        assert hasattr(self, 'TransformNormalizingFactor'), \
                            'You must set TransformNormalizingFactor first'
                        x0, y0 = self.TransformNormalizingFactor['ref']
                        normX = self.TransformNormalizingFactor['x']
                        normY = self.TransformNormalizingFactor['y']
                for ic in range(n_comp):
                    row_start = ic * n
                    row_end = (ic + 1) * n
                    col_start = ic * itransformation
                    # Constant
                    if itransformation > 0:
                        tmporb[row_start:row_end, col_start] = 1.0
                    if itransformation >= 3:
                        tmporb[row_start:row_end, col_start + 1] = (self.x - x0) / normX
                        tmporb[row_start:row_end, col_start + 2] = (self.y - y0) / normY
                    if itransformation >= 4:
                        tmporb[row_start:row_end, col_start + 3] = tmporb[row_start:row_end, col_start + 1] * tmporb[row_start:row_end, col_start + 2]
                orb = np.hstack((orb, tmporb))
            else:
                print('No Transformation asked for object {}'.format(self.name))
                return None

        return orb

    def setGFsInFault(self, fault, G, vertical=True):
        '''
        From a dictionary of Green's functions, sets these correctly into the fault
        object fault for future computation.

        Args:
            * fault     : Instance of Fault
            * G         : Dictionary with entries 'strikeslip', 'dipslip', 'tensile', 'coupling'.

        Kwargs:
            * vertical  : Not used, for consistency.

        Returns:
            * None
        '''

        if fault.type == 'Fault':
            keys = ['strikeslip', 'dipslip', 'tensile', 'coupling']
            GFs = {key: G.get(key, None) for key in keys}
            fault.setGFs(self,
                         strikeslip=[GFs['strikeslip']],
                         dipslip=[GFs['dipslip']],
                         tensile=[GFs['tensile']],
                         coupling=[GFs['coupling']],
                         vertical=True)

        # All done
        return

    def buildsynth(self, faults, direction='sd', poly=None, vertical=True,
                    custom=False, computeNormFact=True):
        '''
        Builds the synthetic data from the fault slip models.

        Args:
            * faults    : list of fault instances

        Kwargs:
            * direction : combination of 's', 'd', 't', 'c'
            * poly      : polynomial correction (None, int)
            * vertical  : not used, for consistency
            * custom    : use custom GFs
            * computeNormFact : compute normalizing factors

        Returns:
            * None
        '''

        if type(faults) is not list:
            faults = [faults]

        n = len(self.station)

        # Initialize synthetics
        self.synth_parallel = np.zeros(n) if self.fault_parallel is not None else None
        self.synth_perpendicular = np.zeros(n) if self.fault_perpendicular is not None else None
        self.synth_vertical = np.zeros(n) if self.fault_vertical is not None else None

        for fault in faults:

            if fault.type == 'Fault':

                G = fault.G[self.name]

                for slip_char, slip_key, slip_col in [('s', 'strikeslip', 0),
                                                       ('d', 'dipslip', 1),
                                                       ('t', 'tensile', 2)]:
                    if slip_char in direction and slip_key in G and G[slip_key] is not None:
                        synth_full = G[slip_key].dot(fault.slip[:, slip_col])
                        idx = 0
                        if self.synth_parallel is not None:
                            self.synth_parallel += synth_full[idx * n:(idx + 1) * n]
                            idx += 1
                        if self.synth_perpendicular is not None:
                            self.synth_perpendicular += synth_full[idx * n:(idx + 1) * n]
                            idx += 1
                        if self.synth_vertical is not None:
                            self.synth_vertical += synth_full[idx * n:(idx + 1) * n]

                if 'c' in direction and 'coupling' in G and G['coupling'] is not None:
                    synth_full = G['coupling'].dot(fault.coupling)
                    idx = 0
                    if self.synth_parallel is not None:
                        self.synth_parallel += synth_full[idx * n:(idx + 1) * n]
                        idx += 1
                    if self.synth_perpendicular is not None:
                        self.synth_perpendicular += synth_full[idx * n:(idx + 1) * n]
                        idx += 1
                    if self.synth_vertical is not None:
                        self.synth_vertical += synth_full[idx * n:(idx + 1) * n]

        # Handle polynomial correction
        if poly is not None:
            for fault in faults:
                if fault.type == 'Fault' and hasattr(fault, 'polysol') and self.name in fault.polysol:
                    params = fault.polysol[self.name]
                    if type(params) is dict:
                        params = params[poly]
                    Horb = self.getTransformEstimator(poly, computeNormFact=computeNormFact)
                    orbit = Horb.dot(params)
                    idx = 0
                    if self.synth_parallel is not None:
                        self.synth_parallel += orbit[idx * n:(idx + 1) * n]
                        idx += 1
                    if self.synth_perpendicular is not None:
                        self.synth_perpendicular += orbit[idx * n:(idx + 1) * n]
                        idx += 1
                    if self.synth_vertical is not None:
                        self.synth_vertical += orbit[idx * n:(idx + 1) * n]

        # Build combined synth vector
        self.synth = self.synth_vector

        # All done
        return

    def removeTransformation(self, fault, verbose=False, custom=False):
        '''
        Wrapper for compatibility.
        '''
        # Not implemented for cross-fault offset
        return

    def removeSynth(self, faults, direction='sd', poly=None, vertical=True,
                     custom=False, computeNormFact=True):
        '''
        Computes the synthetics and removes them from the data.
        '''
        # Build synth first
        self.buildsynth(faults, direction=direction, poly=poly,
                         vertical=vertical, custom=custom,
                         computeNormFact=computeNormFact)

        # Remove
        if self.fault_parallel is not None and self.synth_parallel is not None:
            self.fault_parallel -= self.synth_parallel
        if self.fault_perpendicular is not None and self.synth_perpendicular is not None:
            self.fault_perpendicular -= self.synth_perpendicular
        if self.fault_vertical is not None and self.synth_vertical is not None:
            self.fault_vertical -= self.synth_vertical

        # All done
        return

    def getRMS(self):
        '''
        Computes the RMS of the data and the residuals.

        Returns:
            * dataRMS, synthRMS
        '''

        d = self.data_vector
        N = len(d)
        dataRMS = np.sqrt(np.sum(d ** 2) / N)

        if self.synth is not None:
            residual = d - self.synth
            synthRMS = np.sqrt(np.sum(residual ** 2) / N)
        else:
            synthRMS = 0.

        return dataRMS, synthRMS

    def select_stations(self, minlon, maxlon, minlat, maxlat):
        '''
        Select the station pairs inside a lon/lat bounding box.

        Selection is based on the reference (midpoint) coordinates.

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
        self.lon1 = self.lon1[u]
        self.lat1 = self.lat1[u]
        self.x1 = self.x1[u]
        self.y1 = self.y1[u]
        self.lon2 = self.lon2[u]
        self.lat2 = self.lat2[u]
        self.x2 = self.x2[u]
        self.y2 = self.y2[u]
        self.strike = self.strike[u]

        if self.fault_parallel is not None:
            self.fault_parallel = self.fault_parallel[u]
        if self.fault_perpendicular is not None:
            self.fault_perpendicular = self.fault_perpendicular[u]
        if self.fault_vertical is not None:
            self.fault_vertical = self.fault_vertical[u]
        if self.err_parallel is not None:
            self.err_parallel = self.err_parallel[u]
        if self.err_perpendicular is not None:
            self.err_perpendicular = self.err_perpendicular[u]
        if self.err_vertical is not None:
            self.err_vertical = self.err_vertical[u]

        if self.Cd is not None:
            ncomp = self.obs_per_station
            idx = np.concatenate([u + k * len(u) for k in range(ncomp)])
            self.Cd = self.Cd[np.ix_(idx, idx)]

        # All done
        return

    def reject(self, u):
        '''
        Reject data points by index.

        Args:
            * u : array of indices to remove
        '''
        u = np.array(u, dtype=int)
        if len(u) == 0:
            return

        keep = np.ones(len(self.station), dtype=bool)
        keep[u] = False

        self.station = self.station[keep]
        self.lon = self.lon[keep]
        self.lat = self.lat[keep]
        self.x = self.x[keep]
        self.y = self.y[keep]
        self.lon1 = self.lon1[keep]
        self.lat1 = self.lat1[keep]
        self.x1 = self.x1[keep]
        self.y1 = self.y1[keep]
        self.lon2 = self.lon2[keep]
        self.lat2 = self.lat2[keep]
        self.x2 = self.x2[keep]
        self.y2 = self.y2[keep]
        self.strike = self.strike[keep]

        if self.fault_parallel is not None:
            self.fault_parallel = self.fault_parallel[keep]
        if self.fault_perpendicular is not None:
            self.fault_perpendicular = self.fault_perpendicular[keep]
        if self.fault_vertical is not None:
            self.fault_vertical = self.fault_vertical[keep]
        if self.err_parallel is not None:
            self.err_parallel = self.err_parallel[keep]
        if self.err_perpendicular is not None:
            self.err_perpendicular = self.err_perpendicular[keep]
        if self.err_vertical is not None:
            self.err_vertical = self.err_vertical[keep]

        if self.Cd is not None:
            ncomp = self.obs_per_station
            n_new = int(keep.sum())
            keep_idx = np.flatnonzero(keep)
            n_old = len(keep)
            idx = np.concatenate([keep_idx + k * n_old for k in range(ncomp)])
            self.Cd = self.Cd[np.ix_(idx, idx)]

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

    def plot(self, show=True, axis='cumdist', savefig=None,
             residuals=True, ylim=None, unit_label='m',
             figsize='double', figsize_aspect=None,
             style='science', style_kwargs=None):
        '''
        Plot cross-fault offset data, synthetics, and optionally residuals.

        Each active component (fault-parallel, fault-perpendicular, fault-vertical)
        gets its own subplot.  When synthetics are available, residuals
        (data - synth) are drawn as a thin dashed line sharing the same axes.

        Kwargs:
            * show          : show the plot (default True)
            * axis          : x-axis type, one of
                              'cumdist'  - cumulative along-fault distance (km)
                              'lon'      - reference longitude
                              'lat'      - reference latitude
                              'index'    - integer station index
            * savefig       : file path to save figure (None = do not save)
            * residuals     : if True and synthetics exist, overlay data-synth residuals
                              as a dashed line (default True)
            * ylim          : (ymin, ymax) tuple applied to ALL subplots for a
                              common y-axis scale; None = auto-scale per subplot
            * unit_label    : label suffix for y-axis (default 'm')
            * figsize       : PlotStyle figsize — 'single', 'double',
                              scalar width, or (w, h) tuple.  Default 'double'.
            * figsize_aspect: height/width ratio; default auto-scales with
                              number of components (~0.35 * n_comp)
            * style         : PlotStyle preset (default 'science'); None to disable
            * style_kwargs  : extra kwargs forwarded to PlotStyle
                              (e.g. fontsize, usetex, dpi, rcparams, ...)

        Returns:
            * fig           : matplotlib Figure
        '''

        n_comp = self.obs_per_station
        if n_comp == 0:
            print('No data to plot')
            return

        # Auto-compute aspect based on number of component panels
        if figsize_aspect is None:
            figsize_aspect = 0.35 * n_comp

        ctx, fs = self._get_plot_style_context(
            style, style_kwargs, figsize=figsize, figsize_aspect=figsize_aspect)

        with ctx:
            # ----- X-axis -------------------------------------------------------
            if axis == 'lon':
                xvals  = self.lon
                xlabel = 'Longitude (\u00b0)'
            elif axis == 'lat':
                xvals  = self.lat
                xlabel = 'Latitude (\u00b0)'
            elif axis == 'index':
                xvals  = np.arange(len(self.station))
                xlabel = 'Station index'
            else:
                # Cumulative along-fault distance: project midpoints onto mean strike
                if self.x is not None and self.strike is not None:
                    mean_strike = float(np.mean(self.strike))
                    sp_e = -np.sin(mean_strike)
                    sp_n =  np.cos(mean_strike)
                    along = self.x * sp_e + self.y * sp_n  # km
                    xvals  = along - along.min()
                    xlabel = 'Along-fault distance (km)'
                else:
                    xvals  = np.arange(len(self.station))
                    xlabel = 'Station index'

            # ----- Figure layout ------------------------------------------------
            fig, axes = plt.subplots(n_comp, 1, figsize=fs, squeeze=False)
            fig.suptitle('Cross-Fault Offset: {}'.format(self.name), fontsize=11)

            # ----- Helper to draw one component ---------------------------------
            def _draw_component(ax, data_arr, err_arr, synth_arr, label, ylabel):
                ax.errorbar(xvals, data_arr,
                            yerr=err_arr if err_arr is not None else None,
                            fmt='o', markersize=4, color='steelblue', capsize=2,
                            elinewidth=0.8, label='Data')
                if synth_arr is not None:
                    ax.plot(xvals, synth_arr, 's-', markersize=3, linewidth=1.0,
                            color='crimson', label='Synth')
                    if residuals:
                        resid = data_arr - synth_arr
                        ax.plot(xvals, resid, '--', linewidth=1.0, color='darkorange',
                                label='Residual')
                        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
                ax.set_ylabel('{} ({})'.format(ylabel, unit_label))
                ax.set_xlabel(xlabel)
                if ylim is not None:
                    ax.set_ylim(ylim)
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.25)

            # ----- Per-component subplots ---------------------------------------
            idx = 0
            if self.fault_parallel is not None:
                _draw_component(axes[idx, 0],
                                self.fault_parallel, self.err_parallel,
                                self.synth_parallel,
                                'Fault-Parallel', 'fp')
                idx += 1

            if self.fault_perpendicular is not None:
                _draw_component(axes[idx, 0],
                                self.fault_perpendicular, self.err_perpendicular,
                                self.synth_perpendicular,
                                'Fault-Perpendicular', 'fn')
                idx += 1

            if self.fault_vertical is not None:
                _draw_component(axes[idx, 0],
                                self.fault_vertical, self.err_vertical,
                                self.synth_vertical,
                                'Fault-Vertical', 'fv')

            plt.tight_layout()

            if savefig is not None:
                fig.savefig(savefig, dpi=150, bbox_inches='tight')
            if show:
                plt.show()

            return fig

    def plotMap(self, show=True, savefig=None,
                figsize='double', figsize_aspect=1.0,
                style='science', style_kwargs=None):
        '''
        Plot the point pairs on a map.

        Kwargs:
            * figsize       : 'single', 'double', scalar, or (w,h). Default 'double'.
            * figsize_aspect: height/width ratio (default 1.0 for square map)
            * style         : PlotStyle preset (default 'science'); None to disable
            * style_kwargs  : extra kwargs forwarded to PlotStyle
        '''

        ctx, fs = self._get_plot_style_context(
            style, style_kwargs, figsize=figsize, figsize_aspect=figsize_aspect)

        with ctx:
            fig, ax = plt.subplots(1, 1, figsize=fs)

            for i in range(len(self.station)):
                ax.plot([self.lon1[i], self.lon2[i]], [self.lat1[i], self.lat2[i]],
                        'k-', linewidth=0.5)
                ax.plot(self.lon1[i], self.lat1[i], 'bv', markersize=4)
                ax.plot(self.lon2[i], self.lat2[i], 'r^', markersize=4)
                ax.plot(self.lon[i], self.lat[i], 'g+', markersize=6)

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Cross-Fault Offset Point Pairs: {}'.format(self.name))
            ax.set_aspect('equal')

            if savefig is not None:
                fig.savefig(savefig, dpi=150, bbox_inches='tight')
            if show:
                plt.show()

            return fig

#EOF
