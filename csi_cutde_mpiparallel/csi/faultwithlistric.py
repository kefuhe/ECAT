'''
A class for faults with varying strike and dip along strike and depth.

Supports:
    1. Curved fault traces (variable strike)
    2. Variable dip along strike (different dip at each node)
    3. Variable dip along depth (listric faults)
    4. Non-uniform patch width distribution (geometric series)

Written based on SDMFaultGeometry, adapted to CSI framework.
'''

# Externals
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import copy
import sys

# Rectangular patches Fault class
from .RectangularPatches import RectangularPatches


class faultwithlistric(RectangularPatches):
    '''
    Fault class with variable strike and dip along strike and depth.
    
    Features:
        - Curved fault trace (variable strike along strike)
        - Variable dip along strike
        - Variable dip along depth (listric faults)
        - Non-uniform patch width (geometric series distribution)
    '''

    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, 
                 verbose=True):
        '''
        Args:
            * name      : Name of the fault.
            * utmzone   : UTM zone (optional, default=None)
            * ellps     : Ellipsoid (optional, default='WGS84')
            * lon0, lat0: Reference longitude/latitude for local coordinates
            * verbose   : Print progress info
        '''
        
        super(faultwithlistric, self).__init__(name,
                                                        utmzone=utmzone,
                                                        ellps=ellps, 
                                                        lon0=lon0,
                                                        lat0=lat0,
                                                        verbose=verbose)
        
        # Initialize geometry attributes
        self.dip_top_array = None
        self.dip_bottom_array = None
        self.local_strikes = None
        self.layer_widths = None
        self.width_bias = None
        self.min_patch_width = None
        self.nodes = None
        self.dip_at_nodes = None
        
        # Initialize geometry parameters
        self.top = 0.0
        self.width = None
        self.numz = None
        
        return

    def setGeometry(self, top=0.0, width=None, numz=None):
        '''
        Set fault geometry parameters.
        
        Args:
            * top   : float, depth of fault top (km). Default: 0.0
            * width : float, fault width along dip direction (km).
            * numz  : int, number of layers in depth direction.
        '''
        self.top = top
        if width is not None:
            self.width = width
        if numz is not None:
            self.numz = numz
        
        return

    def setDipDistribution(self, dip_top, dip_bottom=None):
        '''
        Set dip distribution along fault trace.
        
        Args:
            * dip_top    : array-like, shape (n_trace,)
                           Dip angle at top of fault for each trace point (degrees).
            * dip_bottom : array-like, shape (n_trace,), optional
                           Dip angle at bottom of fault for each trace point (degrees).
                           If None, uses dip_top (constant dip with depth).
        
        Note:
            Must be called after trace() and before buildPatches().
            Array length must match number of trace points.
        '''
        self.dip_top_array = np.array(dip_top)
        
        if dip_bottom is None:
            self.dip_bottom_array = self.dip_top_array.copy()
        else:
            self.dip_bottom_array = np.array(dip_bottom)
        
        # Validate
        n_trace = len(self.xf) if hasattr(self, 'xf') else 0
        if n_trace > 0:
            assert len(self.dip_top_array) == n_trace, \
                f"dip_top length ({len(self.dip_top_array)}) must match trace length ({n_trace})"
            assert len(self.dip_bottom_array) == n_trace, \
                f"dip_bottom length ({len(self.dip_bottom_array)}) must match trace length ({n_trace})"
        
        return

    def setWidthDistribution(self, width_bias=None, min_patch_width=None):
        '''
        Set non-uniform patch width distribution along dip direction.
        
        Args:
            * width_bias      : float, optional
                                Geometric ratio for width distribution.
                                - None or 1.0: uniform width
                                - > 1.0: denser at shallow, sparser at deep
                                - < 1.0: sparser at shallow, denser at deep
            * min_patch_width : float, optional
                                Minimum patch width (first term of geometric series).
                                Required when width_bias is set.
        
        Note:
            Call before buildPatches() to take effect.
        '''
        self.width_bias = width_bias
        self.min_patch_width = min_patch_width
        
        if width_bias is not None and width_bias != 1.0:
            if min_patch_width is None:
                raise ValueError("min_patch_width is required when width_bias is set")
        
        return

    def _calculateTraceLength(self):
        '''Calculate total length of fault trace.'''
        if not hasattr(self, 'xf') or self.xf is None:
            return 0.0
        
        dx = np.diff(self.xf)
        dy = np.diff(self.yf)
        distances = np.sqrt(dx**2 + dy**2)
        return np.sum(distances)

    def _getCumulativeDistance(self):
        '''Calculate cumulative distance along trace.'''
        dx = np.diff(self.xf)
        dy = np.diff(self.yf)
        distances = np.sqrt(dx**2 + dy**2)
        return np.concatenate([[0], np.cumsum(distances)])

    def _interpolateTrace(self, distances):
        '''
        Interpolate fault trace at specified distances.
        
        Args:
            * distances : array-like, distances along trace (km)
        
        Returns:
            * positions : ndarray, shape (n, 2), interpolated (x, y)
            * strikes   : ndarray, shape (n,), local strike angles (degrees)
        '''
        cum_dist = self._getCumulativeDistance()
        
        # Interpolate coordinates
        interp_x = interp1d(cum_dist, self.xf, kind='linear', fill_value='extrapolate')
        interp_y = interp1d(cum_dist, self.yf, kind='linear', fill_value='extrapolate')
        
        x_interp = interp_x(distances)
        y_interp = interp_y(distances)
        positions = np.column_stack([x_interp, y_interp])
        
        # Calculate local strikes
        strikes = self._calculateLocalStrikes(positions)
        
        return positions, strikes

    def _calculateLocalStrikes(self, positions):
        '''
        Calculate local strike at each position using central difference.
        
        Strike is azimuth from North, clockwise (degrees).
        '''
        n = len(positions)
        strikes = np.zeros(n)
        
        for i in range(n):
            if i == 0:
                dx = positions[i+1, 0] - positions[i, 0]
                dy = positions[i+1, 1] - positions[i, 1]
            elif i == n - 1:
                dx = positions[i, 0] - positions[i-1, 0]
                dy = positions[i, 1] - positions[i-1, 1]
            else:
                dx = positions[i+1, 0] - positions[i-1, 0]
                dy = positions[i+1, 1] - positions[i-1, 1]
            
            # Strike: angle from North (y-axis), clockwise
            # atan2(dx, dy) gives angle from y-axis
            strikes[i] = np.rad2deg(np.arctan2(dx, dy)) % 360
        
        return strikes

    def _interpolateDip(self, distances):
        '''
        Interpolate dip angles at specified distances.
        
        Returns:
            * dip_top    : ndarray, top dip angles (degrees)
            * dip_bottom : ndarray, bottom dip angles (degrees)
        '''
        cum_dist = self._getCumulativeDistance()
        
        interp_top = interp1d(cum_dist, self.dip_top_array, 
                              kind='linear', fill_value='extrapolate')
        interp_bottom = interp1d(cum_dist, self.dip_bottom_array, 
                                 kind='linear', fill_value='extrapolate')
        
        return interp_top(distances), interp_bottom(distances)

    def _calculateLayerWidths(self, total_width, n_layers):
        '''
        Calculate width of each layer.
        
        Args:
            * total_width : float, total fault width (km)
            * n_layers    : int, number of layers
        
        Returns:
            * layer_widths : ndarray, shape (n_layers,), width of each layer
            * cum_widths   : ndarray, shape (n_layers+1,), cumulative widths from 0
        '''
        if self.width_bias is None or self.width_bias == 1.0:
            # Uniform distribution
            layer_widths = np.full(n_layers, total_width / n_layers)
        else:
            # Geometric series: w0, w0*r, w0*r^2, ...
            # Sum = w0 * (r^n - 1) / (r - 1) = total_width
            r = self.width_bias
            w0 = total_width * (r - 1) / (r**n_layers - 1)
            layer_widths = w0 * r**np.arange(n_layers)
        
        cum_widths = np.concatenate([[0], np.cumsum(layer_widths)])
        return layer_widths, cum_widths

    def buildPatches(self, dip_top, dip_bottom=None, dipdirection=None,
                        every=10, numz=None, width=None,
                        width_bias=None, min_patch_width=None,
                        minpatchsize=0.00001, verbose=None):
        '''
        Build fault patches with variable strike and dip.
        
        Args:
            * dip_top       : array-like or float
                                Dip angle at fault top for each trace point (degrees).
                                If float, uses constant dip along strike.
            * dip_bottom    : array-like or float, optional
                                Dip angle at fault bottom for each trace point (degrees).
                                If None, uses dip_top (constant dip with depth).
            * dipdirection  : float, optional
                                Azimuth towards which fault dips (degrees from North).
                                If None, computed as strike + 90° (dipping to the right).
            * every         : float, patch length along strike (km). Default: 10.
            * numz          : int, optional, number of layers in depth direction.
                                Overrides self.numz if provided.
            * width         : float, optional, fault width along dip (km).
                                Overrides self.width if provided.
            * width_bias    : float, optional, geometric ratio for width distribution.
                                - None or 1.0: uniform width
                                - > 1.0: denser at shallow
                                - < 1.0: denser at deep
            * min_patch_width: float, optional, minimum patch width (km).
                                Required when width_bias is set.
            * minpatchsize  : float, minimum patch size to include (km).
            * verbose       : bool, optional, print progress.
        
        Sets:
            * self.patch, self.patchll, self.slip, self.patchdip
        '''
        if verbose is None:
            verbose = self.verbose
        
        # Set width distribution parameters
        if width_bias is not None:
            self.setWidthDistribution(width_bias, min_patch_width)
        
        # Update geometry parameters if provided
        if numz is not None:
            self.numz = numz
        if width is not None:
            self.width = width
        
        # Prepare dip arrays
        n_trace = len(self.xf)
        
        if np.isscalar(dip_top):
            self.dip_top_array = np.full(n_trace, float(dip_top))
        else:
            self.dip_top_array = np.array(dip_top)
            assert len(self.dip_top_array) == n_trace, \
                f"dip_top length ({len(self.dip_top_array)}) must match trace length ({n_trace})"
        
        if dip_bottom is None:
            self.dip_bottom_array = self.dip_top_array.copy()
        elif np.isscalar(dip_bottom):
            self.dip_bottom_array = np.full(n_trace, float(dip_bottom))
        else:
            self.dip_bottom_array = np.array(dip_bottom)
            assert len(self.dip_bottom_array) == n_trace, \
                f"dip_bottom length ({len(self.dip_bottom_array)}) must match trace length ({n_trace})"
        
        if verbose:
            print("Building fault with variable geometry")
            print(f"         Dip (top)    : {self.dip_top_array.min():.1f}° - {self.dip_top_array.max():.1f}°")
            print(f"         Dip (bottom) : {self.dip_bottom_array.min():.1f}° - {self.dip_bottom_array.max():.1f}°")
            if self.width_bias is not None:
                print(f"         Width bias   : {self.width_bias}")
        
        # Initialize structures
        self.patch = []
        self.patchll = []
        self.slip = []
        self.patchdip = []
        
        # Discretize trace
        self.discretize_trace(every, threshold=every/3.0)
        
        # Calculate trace length and sampling points
        total_length = self._calculateTraceLength()
        nlength = max(2, int(np.round(total_length / every)) + 1)
        l_samples = np.linspace(0, total_length, nlength)
        
        # Interpolate trace, strike, and dip
        positions, strikes = self._interpolateTrace(l_samples)
        dip_top_interp, dip_bottom_interp = self._interpolateDip(l_samples)
        self.local_strikes = strikes
        
        # Calculate layer widths
        n_layers = self.numz
        layer_widths, cum_widths = self._calculateLayerWidths(self.width, n_layers)
        self.layer_widths = layer_widths
        
        # Initialize nodes array: (nlength, nwidth, 3)
        nwidth = n_layers + 1
        nodes = np.zeros((nlength, nwidth, 3))
        dip_at_nodes = np.zeros((nlength, nwidth))
        
        # Generate all nodes
        for il in range(nlength):
            x_trace, y_trace = positions[il]
            strike_rad = np.deg2rad(strikes[il])
            dip_t = dip_top_interp[il]
            dip_b = dip_bottom_interp[il]
            
            # Dip direction: perpendicular to strike, to the right
            if dipdirection is None:
                dip_dir_rad = strike_rad + np.pi/2
            else:
                dip_dir_rad = np.deg2rad(dipdirection)
            
            # First layer (surface trace)
            nodes[il, 0, 0] = x_trace
            nodes[il, 0, 1] = y_trace
            nodes[il, 0, 2] = self.top
            dip_at_nodes[il, 0] = dip_t
            
            # Build layers from top to bottom
            for iw in range(1, nwidth):
                prev_x = nodes[il, iw-1, 0]
                prev_y = nodes[il, iw-1, 1]
                prev_z = nodes[il, iw-1, 2]
                
                dw = layer_widths[iw-1]
                
                # Interpolate dip based on cumulative width ratio
                w_ratio = cum_widths[iw] / self.width
                dip_current = dip_t + (dip_b - dip_t) * w_ratio
                dip_rad = np.deg2rad(dip_current)
                dip_at_nodes[il, iw] = dip_current
                
                # Horizontal displacement along dip direction
                dx = dw * np.cos(dip_rad) * np.sin(dip_dir_rad)
                dy = dw * np.cos(dip_rad) * np.cos(dip_dir_rad)
                dz = dw * np.sin(dip_rad)
                
                nodes[il, iw, 0] = prev_x + dx
                nodes[il, iw, 1] = prev_y + dy
                nodes[il, iw, 2] = prev_z + dz
        
        self.nodes = nodes
        self.dip_at_nodes = dip_at_nodes
        
        # Track depth range
        D = [self.top]
        
        # Build patches from nodes
        for iw in range(n_layers):
            D.append(nodes[:, iw+1, 2].max())
            
            for il in range(nlength - 1):
                # Four corners
                x1, y1, z1 = nodes[il, iw]
                x2, y2, z2 = nodes[il+1, iw]
                x3, y3, z3 = nodes[il+1, iw+1]
                x4, y4, z4 = nodes[il, iw+1]
                
                # # CSI requires p1 and p2 (top edge) to have same depth
                # # Also p3 and p4 (bottom edge) to have same depth
                # z_top = (z1 + z2) / 2.0
                # z_bottom = (z3 + z4) / 2.0
                
                # # Use averaged depths
                # z1 = z_top
                # z2 = z_top
                # z3 = z_bottom
                # z4 = z_bottom
                
                lon1, lat1 = self.xy2ll(x1, y1)
                lon2, lat2 = self.xy2ll(x2, y2)
                lon3, lat3 = self.xy2ll(x3, y3)
                lon4, lat4 = self.xy2ll(x4, y4)
                
                # Ensure consistent ordering (p1 should be "left" of p2)
                if y1 > y2:
                    p1 = [x2, y2, z2]; p1ll = [lon2, lat2, z2]
                    p2 = [x1, y1, z1]; p2ll = [lon1, lat1, z1]
                    p3 = [x4, y4, z4]; p3ll = [lon4, lat4, z4]
                    p4 = [x3, y3, z3]; p4ll = [lon3, lat3, z3]
                else:
                    p1 = [x1, y1, z1]; p1ll = [lon1, lat1, z1]
                    p2 = [x2, y2, z2]; p2ll = [lon2, lat2, z2]
                    p3 = [x3, y3, z3]; p3ll = [lon3, lat3, z3]
                    p4 = [x4, y4, z4]; p4ll = [lon4, lat4, z4]
                
                # Skip small patches
                psize = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if psize < minpatchsize:
                    continue
                
                p = np.array([p1, p2, p3, p4])
                pll = np.array([p1ll, p2ll, p3ll, p4ll])
                
                # Average dip for this patch
                patch_dip = np.mean([dip_at_nodes[il, iw], dip_at_nodes[il+1, iw],
                                        dip_at_nodes[il+1, iw+1], dip_at_nodes[il, iw+1]])
                
                self.patch.append(p)
                self.patchll.append(pll)
                self.slip.append([0.0, 0.0, 0.0])
                self.patchdip.append(np.deg2rad(patch_dip))
        
        # Set depth info
        D = np.array(D)
        self.z_patches = D
        self.depth = D.max()
        
        # Convert to arrays
        self.slip = np.array(self.slip)
        
        # Re-discretize to restore original trace
        self.discretize_trace(every, threshold=every/3.0)
        
        # Compute equivalent rectangles
        self.computeEquivRectangle()
        
        if verbose:
            print(f"         Patches      : {len(self.patch)}")
            print(f"         Depth range  : {self.top:.1f} - {self.depth:.1f} km")
        
        return

    def plotCrossSections(self, n_sections=5, figsize=(16, 4)):
        '''
        Plot vertical cross-sections along the fault.
        
        Args:
            * n_sections : int, number of sections to plot
            * figsize    : tuple, figure size
        '''
        if self.nodes is None:
            raise ValueError("Call buildPatches() first")
        
        fig, axes = plt.subplots(1, n_sections, figsize=figsize)
        if n_sections == 1:
            axes = [axes]
        
        nlength = self.nodes.shape[0]
        nwidth = self.nodes.shape[1]
        section_indices = np.linspace(0, nlength-1, n_sections, dtype=int)
        
        for idx, il in enumerate(section_indices):
            ax = axes[idx]
            
            profile = self.nodes[il, :, :]
            
            # Distance along dip
            w_dist = np.zeros(nwidth)
            for iw in range(1, nwidth):
                w_dist[iw] = w_dist[iw-1] + np.linalg.norm(profile[iw] - profile[iw-1])
            
            ax.plot(w_dist, profile[:, 2], 'bo-', markersize=6, linewidth=2)
            ax.fill_between(w_dist, profile[:, 2], profile[:, 2].max() + 1,
                           alpha=0.3, color='brown')
            
            ax.set_xlabel('Distance along dip (km)')
            ax.set_ylabel('Depth (km)')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            
            dip_range = self.dip_at_nodes[il, :]
            ax.set_title(f'Section {idx+1}\nDip: {dip_range[0]:.1f}°→{dip_range[-1]:.1f}°')
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes


# ============= Examples =============

def example_straight_fault_variable_dip():
    '''Example: Straight fault with variable dip.'''
    print("=" * 60)
    print("Example: Straight fault with variable dip")
    print("=" * 60)
    
    # Create fault
    fault = faultwithlistric('test_fault', lon0=100.0, lat0=30.0, verbose=True)
    
    # Define trace (straight line)
    n_trace = 6
    lon = np.linspace(100.0, 100.5, n_trace)
    lat = np.linspace(30.0, 30.1, n_trace)
    fault.trace(lon, lat)
    
    # Set geometry
    fault.setGeometry(top=2.0, width=15.0, numz=5)
    
    # Dip varies along strike: 30° to 50° at top, 45° to 70° at bottom
    dip_top = np.linspace(30, 50, n_trace)
    dip_bottom = np.linspace(45, 70, n_trace)
    
    # Build patches
    fault.buildPatches(dip_top=dip_top, dip_bottom=dip_bottom, every=5.0)
    
    print(f"Number of patches: {len(fault.patch)}")
    
    # Visualize
    fault.plotCrossSections(n_sections=3)
    fault.initializeslip(values='depth')
    fault.plot(drawCoastlines=False)
    
    return fault


def example_curved_fault():
    '''Example: Curved fault with variable strike and dip.'''
    print("\n" + "=" * 60)
    print("Example: Curved fault with variable strike and dip")
    print("=" * 60)
    
    fault = faultwithlistric('curved_fault', lon0=100.0, lat0=30.0, verbose=True)
    
    # Arc-shaped trace
    theta = np.linspace(0, np.pi/3, 8)
    radius = 0.4  # degrees
    lon = 100.0 + radius * np.cos(theta)
    lat = 30.0 + radius * np.sin(theta)
    fault.trace(lon, lat)
    
    fault.setGeometry(top=1.0, width=12.0, numz=5)
    
    n_trace = len(lon)
    dip_top = np.linspace(25, 45, n_trace)
    dip_bottom = np.linspace(40, 65, n_trace)
    
    fault.buildPatches(dip_top=dip_top, dip_bottom=dip_bottom, every=4.0)
    fault.plotCrossSections(n_sections=3)
    fault.initializeslip(values='depth')
    fault.plot(drawCoastlines=False)
    print(f"Number of patches: {len(fault.patch)}")
    print(f"Strike range: {fault.local_strikes.min():.1f}° - {fault.local_strikes.max():.1f}°")
    
    return fault


def example_listric_fault():
    '''Example: Listric fault (dip decreases with depth).'''
    print("\n" + "=" * 60)
    print("Example: Listric fault")
    print("=" * 60)
    
    fault = faultwithlistric('listric_fault', lon0=100.0, lat0=30.0, verbose=True)
    
    # Straight trace
    lon = np.linspace(100.0, 100.6, 5)
    lat = np.full(5, 30.0)
    fault.trace(lon, lat)
    
    fault.setGeometry(top=0.5, width=20.0, numz=8)
    
    # Listric: steep at top (60°), gentle at bottom (25°)
    fault.buildPatches(dip_top=60.0, dip_bottom=25.0, every=6.0)
    
    fault.plotCrossSections(n_sections=3)
    fault.initializeslip(values='depth')
    fault.plot(drawCoastlines=False, equiv=True)
    return fault


def example_nonuniform_width():
    '''Example: Non-uniform patch width distribution.'''
    print("\n" + "=" * 60)
    print("Example: Non-uniform patch width (geometric series)")
    print("=" * 60)
    
    fault = faultwithlistric('nonuniform_fault', lon0=100.0, lat0=30.0, verbose=True)
    
    lon = np.linspace(100.0, 100.8, 10)
    lat = 30.0 + 0.1 * np.sin(np.linspace(0, 2*np.pi, 10))
    fault.trace(lon, lat)
    
    fault.setGeometry(top=1.5, width=18.0, numz=6)
    
    # Non-uniform width: denser at shallow
    n_trace = len(lon)
    dip_top = 35 + 15 * np.sin(np.linspace(0, 2*np.pi, n_trace))
    dip_bottom = dip_top + 20
    
    fault.buildPatches(dip_top=dip_top, dip_bottom=dip_bottom, every=5.0,
                       width_bias=1.3, min_patch_width=1.0)
    
    print(f"Layer widths: {fault.layer_widths}")
    fault.plotCrossSections(n_sections=4)
    fault.initializeslip(values='depth')
    fault.plot(drawCoastlines=False, equiv=True)
    return fault


if __name__ == "__main__":
    # Run examples
    fault1 = example_straight_fault_variable_dip()
    fault2 = example_curved_fault()
    fault3 = example_listric_fault()
    fault4 = example_nonuniform_width()