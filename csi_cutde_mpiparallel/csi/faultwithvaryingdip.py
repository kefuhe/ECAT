'''
A class that deals with faults with varying dip along strike.

Written by R. Jolivet, April 2013
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from scipy.linalg import block_diag
import copy
import sys
import os

# Rectangular patches Fault class
from .RectangularPatches import RectangularPatches

# Personals
major, minor, micro, release, serial = sys.version_info
if major==2:
    import okada4py as ok


class faultwithvaryingdip(RectangularPatches):

    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None):
        '''
        Args:
            * name      : Name of the fault.
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        super(faultwithvaryingdip,self).__init__(name,
                                                 utmzone = utmzone,
                                                 ellps = ellps, 
                                                 lon0 = lon0,
                                                 lat0 = lat0)

        # All done
        return

    def dipevolution(self, dip):
        '''
        Compute dip angle at each discretized point along strike using piecewise 
        linear interpolation.
        
        Args:
            * dip : list of [distance, dip_angle] pairs defining dip evolution.
                    - distance (km): cumulative distance along fault trace from origin
                    - dip_angle (degrees): dip angle at that distance
                    Must be sorted in ascending order by distance.
                    
        Example:
            dip = [[0, 20], [10, 30], [80, 90]]
            means:
                - At 0 km from origin: dip = 20°
                - At 10 km from origin: dip = 30°
                - At 80 km from origin: dip = 90°
            For points between control points, dip is linearly interpolated.
            For points beyond the last control point, the last dip value is used.
            
        Sets:
            * self.dip   : list of dip angles (degrees) for each discretized point
            * self.track : list of cumulative distances (km) for each point
        '''

        # Create a structure
        self.dip = []
        self.track = []

        # Set a distance counter
        dis = 0.0

        # Set the previous x,y
        xp = self.xi[0]
        yp = self.yi[0]

        # Prepare control points arrays
        xdip = np.array([dip[i][0] for i in range(len(dip))])  # distances
        ydip = np.array([dip[i][1] for i in range(len(dip))])  # dip angles

        # Loop along the discretized fault trace
        for i in range(self.xi.shape[0]):

            # Update the cumulative distance
            dis += np.sqrt((self.xi[i]-xp)**2 + (self.yi[i]-yp)**2)
            self.track.append(dis)

            # Find which segment the current distance falls into
            indices = np.flatnonzero(dis >= xdip)
            
            if len(indices) == 0:
                # Before first control point: use first dip value
                d = ydip[0]
            else:
                u = indices[-1]
                
                if u < (len(ydip) - 1):
                    # Within interpolation range: linear interpolation
                    xa, ya = xdip[u], ydip[u]
                    xb, yb = xdip[u+1], ydip[u+1]
                    # Linear interpolation: d = ya + (yb-ya)/(xb-xa) * (dis-xa)
                    slope = (yb - ya) / (xb - xa)
                    d = ya + slope * (dis - xa)
                else:
                    # Beyond last control point: use last dip value
                    d = ydip[-1]

            # Store dip angle
            self.dip.append(d)

            # Update previous point coordinates
            xp = self.xi[i]
            yp = self.yi[i]

        # all done
        return

    def buildPatches(self, dip, dipdirection, every=10, minpatchsize=0.00001, trace_tol=0.1, trace_fracstep=0.2, 
                     trace_xaxis='x', trace_cum_error=True):
        '''
        Build fault patches with varying dip along strike.
        
        Args:
            * dip : list of [distance, dip_angle] pairs defining dip evolution.
                    - distance (km): cumulative distance along fault trace from origin
                    - dip_angle (degrees): dip angle at that distance
                    Example: [[0, 20], [10, 30], [80, 90]]
                    
            * dipdirection : azimuth (degrees from North, clockwise) towards which 
                             the fault dips.
                             
            * every : float, patch length (km) along strike for discretization.
            
            * minpatchsize : float, minimum patch size (km). Patches smaller than 
                             this will be skipped.
                             
            * trace_tol : float, tolerance for trace discretization optimization.
            
            * trace_fracstep : float, fractional step for discretization optimization.
            
            * trace_xaxis : str, axis used for discretization ('x' or 'y').
            
            * trace_cum_error : bool, if True, account for accumulated error in 
                                last patch boundary.
        
        Process:
            1. Discretize surface trace into points spaced ~every km apart
            2. Compute dip angle at each point via piecewise linear interpolation
            3. Drape fault downward layer by layer, each layer offset by self.width
            4. Connect adjacent points to form quadrilateral patches
            
        Sets:
            * self.patch    : list of patches in local coordinates (km)
            * self.patchll  : list of patches in lon/lat coordinates
            * self.slip     : slip vectors for each patch
            * self.patchdip : dip info for each patch
        '''

        # Print
        print("Building a dipping fault")
        print("         Dip Angle       : from {} to {} degrees".format(dip[0], dip[-1]))
        print("         Dip Direction   : {} degrees From North".format(dipdirection))

        # Initialize the structures
        self.patch = []
        self.patchll = []
        self.slip = []
        self.patchdip = []

        # Discretize the surface trace of the fault
        self.discretize(every,trace_tol,trace_fracstep,trace_xaxis,trace_cum_error)

        # Build the dip evolution along strike
        self.dipevolution(dip)

        # Convert degrees to radians
        self.dip = np.array(self.dip)
        self.dip = self.dip*np.pi/180.
        dipdirection_rad = dipdirection*np.pi/180.

        # Initialize the depth of the top row
        self.zi = np.ones((self.xi.shape))*self.top

        # Track depth levels
        D = [self.top]

        # Loop over the depth layers
        for i in range(self.numz):

            # Get the top of the row
            xt = self.xi
            yt = self.yi
            lont, latt = self.xy2ll(xt, yt)
            zt = self.zi

            # Compute the bottom row coordinates
            # Horizontal offset: width * cos(dip), in dipdirection
            # Vertical offset: width * sin(dip)
            xb = xt + self.width*np.cos(self.dip)*np.sin(dipdirection_rad)
            yb = yt + self.width*np.cos(self.dip)*np.cos(dipdirection_rad)
            lonb, latb = self.xy2ll(xb, yb)
            zb = zt + self.width*np.sin(self.dip)

            # Record max depth of this layer
            D.append(zb.max())

            # Build patches by connecting adjacent points
            for j in range(xt.shape[0]-1):
                # 1st corner (top-left)
                x1 = xt[j]
                y1 = yt[j]
                z1 = zt[j]
                lon1 = lont[j]
                lat1 = latt[j]
                # 2nd corner (top-right)
                x2 = xt[j+1]
                y2 = yt[j+1]
                z2 = zt[j+1]
                lon2 = lont[j+1]
                lat2 = latt[j+1]
                # 3rd corner (bottom-right)
                x3 = xb[j+1]
                y3 = yb[j+1]
                z3 = zb[j+1]
                lon3 = lonb[j+1]
                lat3 = latb[j+1]
                # 4th corner (bottom-left)
                x4 = xb[j]
                y4 = yb[j]
                z4 = zb[j]
                lon4 = lonb[j]
                lat4 = latb[j]
                # Order points to ensure consistent orientation
                if y1>y2:
                    p2 = [x1, y1, z1]; p2ll = [lon1, lat1, z1]
                    p1 = [x2, y2, z2]; p1ll = [lon2, lat2, z2]
                    p4 = [x3, y3, z3]; p4ll = [lon3, lat3, z3]
                    p3 = [x4, y4, z4]; p3ll = [lon4, lat4, z4]
                else:
                    p1 = [x1, y1, z1]; p1ll = [lon1, lat1, z1]
                    p2 = [x2, y2, z2]; p2ll = [lon2, lat2, z2]
                    p3 = [x3, y3, z3]; p3ll = [lon3, lat3, z3]
                    p4 = [x4, y4, z4]; p4ll = [lon4, lat4, z4]
                # Skip patches that are too small
                psize = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
                if psize<minpatchsize:
                    continue
                p = [p1, p2, p3, p4]
                pll = [p1ll, p2ll, p3ll, p4ll]
                p = np.array(p)
                pll = np.array(pll)
                # Store patch
                self.patch.append(p)
                self.patchll.append(pll)
                self.slip.append([0.0, 0.0, 0.0])
                self.patchdip.append(dip)

            # Move to next layer: current bottom becomes next top
            self.xi = xb
            self.yi = yb
            self.zi = zb

        # Set depth info
        D = np.array(D)
        self.z_patches = D
        self.depth = D.max()

        # Convert slip to array
        self.slip = np.array(self.slip)

        # Re-discretize to restore original fault trace
        self.discretize(every,trace_tol,trace_fracstep,trace_xaxis,trace_cum_error)

        # Compute equivalent rectangles
        self.computeEquivRectangle()
    
        # All done
        return

#EOF