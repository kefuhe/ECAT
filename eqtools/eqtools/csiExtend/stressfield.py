'''
A class that deals with StressField data.

Written by R. Jolivet, Feb 2014.

Modified by kfhe, at 10/24/2022
Modified by kfhe, at 07/29/2025
'''

# Externals
import sys
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
try:
    import h5py
except:
    print('No hdf5 capabilities detected')

# Personals
from csi import stressfield as stressfield_base
from csi import okadafull as okada

class stressfield(stressfield_base):
    '''
    A class that handles a stress field. Not used in a long time, untested, could be incorrect.

    Args:
        * name          : Name of the StressField dataset.

    Kwargs:
        * utmzone       : UTM zone. Default is 10 (Western US).
        * lon0          : Longitude of the custom utmzone
        * lat0          : Latitude of the custom utmzone
        * ellps         : ellipsoid
        * verbose       : talk to me

    '''

    def __init__(self, name, utmzone=None, lon0=None, lat0=None, ellps='WGS84', verbose=True):

        # Base class init
        super(stressfield, self).__init__(name,
                                          utmzone=utmzone, 
                                          lon0=lon0, lat0=lat0,
                                          ellps=ellps)

        # Initialize the data set 
        self.name = name
        self.dtype = 'strainfield'

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize StressField data set {}".format(self.name))
        self.verbose=verbose

        # Initialize some things
        self.lon = None
        self.lat = None
        self.x = None
        self.y = None
        self.depth = None # Depth is positive downward, always is positive
        self.Stress = None
        self.trace = None

        # All done
        return

    def computeCoulombStress(self, rake, cof=0.6, return_unit='MPa'):
        '''
        Computes the Coulomb Failure Stress from stress tractions.
    
        This method calculates Coulomb failure stress on a fault plane given the rake angle 
        and friction coefficient. It works with stress tractions computed from either 
        triangular or rectangular dislocation sources.
    
        Parameters:
            rake : float
                The rake angle [Radian] of the coulomb failure stress.
                - rake = 0°: pure left-lateral strike-slip
                - rake = 90°: pure reverse/thrust
                - rake = ±180°: pure right-lateral strike-slip
                - rake = -90°: pure normal
            
            cof : float, optional
                The coefficient of friction μ [0, 1.0). Default is 0.6.
            
            return_unit : str, optional
                The unit of the returned Coulomb stress ('MPa', 'kPa', 'Pa', or 'bar').
                Default is 'MPa'.
    
        Returns:
            coulomb : np.ndarray
                The values of Coulomb failure stress in the specified unit.
    
        Raises:
            AssertionError: If stress tractions have not been computed first.
            ValueError: If an invalid return_unit is provided.
    
        Notes:
            - Coulomb stress formula: ΔCFS = τ + μσ
              where τ is shear stress on the fault plane (positive in slip direction)
              and σ is normal stress (positive = unclamping/extension)
            - Must call computeTriangularStressTraction() or computeRectangularStressTraction() first
            - Added by kfhe, 2024
    
        Examples:
            >>> # For triangular source
            >>> sf.computeTriangularStressTraction(source, receiver)
            >>> cfs = sf.computeCoulombStress(rake=np.pi/4, cof=0.6, return_unit='MPa')
            
            >>> # For rectangular source
            >>> sf.computeRectangularStressTraction(source, receiver)
            >>> cfs = sf.computeCoulombStress(rake=0.0, cof=0.4, return_unit='bar')
        '''
        # Check if stress tractions have been computed
        assert hasattr(self, 'TauStrike'), \
            'Must compute stress tractions first using computeTriangularStressTraction() or computeRectangularStressTraction()'
    
        # Compute the Coulomb Failure Stress
        # b is the unit vector in the slip direction on the fault plane
        # rake defines the slip direction: 
        # - rake component along strike (n2): cos(rake)
        # - rake component along dip (n3): sin(rake)
        b = np.cos(rake) * self.n2 + np.sin(rake) * self.n3
        Np = self.Sigma.shape[0]
        
        # Coulomb stress: τ (shear stress in slip direction) + μσ (friction term)
        # T · b gives the traction component in the slip direction
        coulomb = np.array([np.dot(b[:, i], self.T[i]) for i in range(Np)]) + cof * self.Sigma
    
        # Transfer the input unit of the Coulomb stress to lower case
        return_unit = return_unit.lower()
    
        # Transfer the Coulomb stress to the specified unit
        if return_unit == 'mpa':
            coulomb /= 1e6
        elif return_unit == 'kpa':
            coulomb /= 1e3
        elif return_unit == 'bar':
            coulomb /= 1e5
        elif return_unit != 'pa':
            raise ValueError("Invalid return_unit. Choose from 'MPa', 'kPa', 'Pa', or 'bar'.")
    
        return coulomb

    def fault2Stress(self, fault, factor=0.001, mu=30e9, nu=0.25, slipdirection='sd',
                     force_dip=None, stressonpatches=False, convert_to_triangle=False, verbose=False):
        """
        Compute stress tensor from fault slip.
        
        Args:
            fault: Source fault with slip distribution
            factor: Slip unit conversion factor (default: 0.001)
            mu: Shear modulus in Pa (default: 30 GPa)
            nu: Poisson's ratio (default: 0.25)
            slipdirection: Slip components 's', 'd', 't' (default: 'sd')
            force_dip: Override dip angle in radians (default: None)
            stressonpatches: Compute at patch centers (default: False)
            convert_to_triangle: Convert rectangles to triangles for CUTDE (default: False)
            verbose: Print progress (default: False)
        """
        if verbose:
            method = 'CUTDE' if fault.patchType == 'triangle' else 'Okada'
            if convert_to_triangle and fault.patchType == 'rectangle':
                method = 'CUTDE (rect→tri)'
            print(f'Computing stress from {fault.name} using {method}')
        
        # Prepare fault data
        geometry, slips = self._prepare_fault_data(fault, factor, slipdirection, force_dip)
        
        # Get observation points
        xs, ys, zs = self._prepare_observation_points(fault, None, stressonpatches)
        
        # Compute stress tensor
        self.Stress, self.flag, self.flag2 = self._compute_stress_tensor(
            fault, xs, ys, zs, geometry, slips, mu, nu, convert_to_triangle
        )
        self.stresstype = 'total'
        
        if verbose:
            print('Stress computation completed.')
    
    def computeStressTraction(self, source, receiver=None, strike=None, dip=None,
                              factor=0.001, mu=30e9, nu=0.25, slipdirection='sd',
                              force_dip=None, stressonpatches=False, convert_to_triangle=False, 
                              use_matrix_method=False, verbose=False):
        """
        Compute stress tractions on receiver planes.
        
        Args:
            source: Source fault with slip distribution
            receiver: Receiver fault (default: None)
            strike: Strike angle(s) in radians (default: None)
            dip: Dip angle(s) in radians (default: None)
            factor: Slip conversion factor (default: 0.001)
            mu: Shear modulus in Pa (default: 30 GPa)
            nu: Poisson's ratio (default: 0.25)
            slipdirection: Slip components (default: 'sd')
            force_dip: Override dip in radians (default: None)
            stressonpatches: Compute at patch centers (default: False)
            convert_to_triangle: Convert rectangles to triangles (default: False)
            use_matrix_method: Use strain_matrix method for triangular faults (default: False)
            verbose: Print progress (default: False)
        
        Returns:
            (n1, n2, n3, T, Sigma, TauStrike, TauDip)
        
        Notes:
            - For triangular faults, two computation methods are available:
              1. use_matrix_method=False (default): Computes full stress tensor using strain_free,
                 then projects onto receiver planes. More general and memory efficient.
              2. use_matrix_method=True: Uses strain_matrix to compute stress response matrix,
                 then contracts with slip vectors. More efficient for multiple slip scenarios.
            - For rectangular faults, use_matrix_method is ignored.
        """
        # Check if we should use matrix method for triangular faults
        if use_matrix_method and source.patchType == 'triangle':
            return self._compute_traction_matrix_method(
                source, receiver, strike, dip, factor, mu, nu, 
                slipdirection, force_dip, stressonpatches, verbose=verbose
            )
        
        # Standard method: compute stress tensor first, then project
        # Prepare source fault data
        geometry, slips = self._prepare_fault_data(source, factor, slipdirection, force_dip)
        
        # Get observation points
        xs, ys, zs = self._prepare_observation_points(source, receiver, stressonpatches)
        
        # Compute stress tensor
        Stress, _, _ = self._compute_stress_tensor(
            source, xs, ys, zs, geometry, slips, mu, nu, convert_to_triangle
        )
        
        # Determine receiver plane geometry
        N_obs = len(xs)
        if strike is not None and dip is not None:
            strike_rad = np.ones(N_obs) * strike
            dip_rad = np.ones(N_obs) * dip
        else:
            # Get from receiver fault
            strike_rad = receiver.getStrikes()
            dip_rad = receiver.getDips()
        
        if force_dip is not None:
            dip_rad[:] = force_dip
        
        # Compute unit vectors
        n1, n2, n3 = self.strikedip2normal(strike_rad, dip_rad)
        
        # Project stress onto receiver plane
        T = np.array([Stress[:, :, i] @ n1[:, i] for i in range(N_obs)])
        Sigma = np.array([T[i] @ n1[:, i] for i in range(N_obs)])
        TauStrike = np.array([T[i] @ n2[:, i] for i in range(N_obs)])
        TauDip = np.array([T[i] @ n3[:, i] for i in range(N_obs)])
        
        # Store results
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.T, self.Sigma = T, Sigma
        self.TauStrike, self.TauDip = TauStrike, TauDip
        
        if verbose:
            print('Traction computation completed.')
        
        return n1, n2, n3, T, Sigma, TauStrike, TauDip

    def compute_coulomb_stress_on_receiver(self, sources, receiver, strike_angle=None, dip_angle=None, 
                                           rake=0, cof=0.6, return_unit='MPa', # convert_to_triangle=False,
                                           use_matrix_method=False, verbose=False):
        """
        Compute Coulomb stress on receiver fault from multiple sources.

        Args:
            sources (list): List of source fault objects
            receiver: Receiver fault object
            strike_angle: Strike in degrees (overrides receiver geometry)
            dip_angle: Dip in degrees (overrides receiver geometry)
            rake: Rake angle in degrees (default: 0)
            cof: Friction coefficient (default: 0.6)
            return_unit: Output unit (default: 'MPa')
            # convert_to_triangle: Convert rectangles to triangles (default: False). Bug to be fixed later.
            use_matrix_method: Use strain_matrix method for triangular faults (default: False)
            verbose: Print progress (default: False)

        Returns:
            np.ndarray: Coulomb stress on receiver patches
        """
        convert_to_triangle = False  # Temporary fix for bug in receiver geometry handling
        # Override receiver geometry if specified
        if strike_angle is not None and dip_angle is not None:
            npatch = len(receiver.patch)
            strike = np.ones(npatch) * np.radians(strike_angle)
            dip = np.ones(npatch) * np.radians(dip_angle)
        else:
            strike = None
            dip = None
        
        # Accumulate stress from all sources
        coulomb_stress = 0
        rake_rad = np.radians(rake)
        for source in sources:
            self.computeStressTraction(source, receiver=receiver, strike=strike, dip=dip,
                                       convert_to_triangle=convert_to_triangle,
                                       use_matrix_method=use_matrix_method, verbose=verbose)
            coulomb_stress += self.computeCoulombStress(rake=rake_rad, cof=cof, return_unit=return_unit)
        
        return coulomb_stress

    def compute_coulomb_stress_field(self, sources, lon_range, lat_range, grid_size_lon, grid_size_lat, 
                                     strike_angle, dip_angle, rake=0, cof=0.6, return_unit='MPa', 
                                     depth=10.0, convert_to_triangle=False, use_matrix_method=False,
                                     plot=True, cmap='cmc.roma_r', vmin=-0.5, vmax=0.5, add_faults=None, 
                                     savefig=True, figname='coulomb_stress_field.png', dpi=600, verbose=False):
        """
        Compute Coulomb stress field on a uniform grid at specified depth.

        Args:
            sources (list): List of source fault objects
            lon_range (tuple): Longitude range (min, max)
            lat_range (tuple): Latitude range (min, max)
            grid_size_lon (int): Grid points in longitude
            grid_size_lat (int): Grid points in latitude
            strike_angle (float): Receiver plane strike in degrees
            dip_angle (float): Receiver plane dip in degrees
            rake (float): Rake angle in degrees (default: 0)
            cof (float): Friction coefficient (default: 0.6)
            return_unit (str): Output unit (default: 'MPa')
            depth (float): Depth in km (default: 10.0)
            convert_to_triangle (bool): Convert rectangles to triangles (default: False)
            use_matrix_method (bool): Use strain_matrix method (default: False)
            plot (bool): Plot result (default: True)
            cmap (str): Colormap name (default: 'cmc.roma_r')
            vmin (float): Colorbar min (default: -0.5)
            vmax (float): Colorbar max (default: 0.5)
            add_faults (list): Additional faults to plot
            savefig (bool): Save figure (default: True)
            figname (str): Output filename (default: 'coulomb_stress_field.png')
            dpi (int): Figure resolution (default: 600)

        Returns:
            np.ndarray: Coulomb stress field (grid_size_lat, grid_size_lon)
        """
        # Create observation grid
        lon = np.linspace(lon_range[0], lon_range[1], grid_size_lon)
        lat = np.linspace(lat_range[0], lat_range[1], grid_size_lat)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        lon_flat = lon_grid.flatten()
        lat_flat = lat_grid.flatten()
        
        # Set uniform receiver plane geometry
        strike = np.ones_like(lon_flat) * np.radians(strike_angle)
        dip = np.ones_like(lon_flat) * np.radians(dip_angle)
        self.setLonLatZ(lon_flat, lat_flat, np.ones_like(lon_flat) * depth)
        
        # Compute cumulative Coulomb stress
        coulomb_stress = 0
        rake_rad = np.radians(rake)
        for source in sources:
            self.computeStressTraction(source, strike=strike, dip=dip,
                                       convert_to_triangle=convert_to_triangle,
                                       use_matrix_method=use_matrix_method, verbose=verbose)
            coulomb_stress += self.computeCoulombStress(rake=rake_rad, cof=cof, return_unit=return_unit)
        
        # Reshape to grid
        coulomb_stress = coulomb_stress.reshape((grid_size_lat, grid_size_lon))

        # Plot if requested
        if plot:
            from ..plottools import sci_plot_style
            import cmcrameri
            import matplotlib.pyplot as plt

            with sci_plot_style():
                plt.imshow(coulomb_stress, extent=[lon_range[0], lon_range[1], 
                          lat_range[0], lat_range[1]], origin='lower', 
                          cmap=cmap, vmin=vmin, vmax=vmax)
                
                # Plot source faults
                for source in sources:
                    plt.plot(source.lon, source.lat, 'k', linewidth=0.5)
                
                # Plot additional faults
                if add_faults is not None:
                    for fault in add_faults:
                        plt.plot(fault.lon, fault.lat, 'k', linewidth=0.5)
                
                plt.colorbar(label=f'$\Delta CFS$ ({return_unit})')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                
                if savefig:
                    plt.savefig(figname, dpi=dpi, bbox_inches='tight')
                plt.show()

        return coulomb_stress
    
    def _compute_stress_tensor(self, fault, xs, ys, zs, geometry, slips, mu=30e9, nu=0.25, 
                               convert_to_triangle=False):
        """
        Compute full stress tensor using appropriate method.
        
        Args:
            fault: Fault object
            xs, ys, zs: Observation coordinates (ENU, z positive upward)
            geometry: Dict of patch geometry
            slips: Dict of slip components
            mu: Shear modulus in Pa
            nu: Poisson's ratio
            convert_to_triangle: Convert rectangles to triangles for CUTDE
        
        Returns:
            (Stress, flag, flag2): Stress tensor (3,3,N) and validity flags
        """
        # Convert rectangular to triangular if requested
        if convert_to_triangle and fault.patchType == 'rectangle':
            return self._compute_stress_cutde(
                fault, xs, ys, zs, 
                np.column_stack((slips['strike'], slips['dip'], slips['tensile'])),
                mu, nu
            )
        
        # Standard computation based on patch type
        if fault.patchType == 'rectangle':
            # Use Okada solution
            Stress, flag, flag2 = okada.stress(
                xs, ys, zs,
                geometry['xc'], geometry['yc'], geometry['zc'],
                geometry['width'], geometry['length'],
                geometry['strike'], geometry['dip'],
                slips['strike'], slips['dip'], slips['tensile'],
                mu, nu, full=True
            )
            return Stress, flag, flag2
            
        elif fault.patchType == 'triangle':
            # Use CUTDE directly with existing triangular data
            return self._compute_stress_cutde(
                fault, xs, ys, zs,
                np.column_stack((slips['strike'], slips['dip'], slips['tensile'])),
                mu, nu
            )
        
        else:
            raise ValueError(f"Unsupported patch type: {fault.patchType}")
    
    def _compute_stress_cutde(self, fault, xs, ys, zs, slip_vec, mu, nu):
        """
        Compute stress using CUTDE for triangular patches.
        
        Args:
            fault: Fault object (can be triangular or rectangular)
            xs, ys, zs: Observation coordinates (ENU, z positive upward)
            slip_vec: (N, 3) slip vectors [strike, dip, tensile]
            mu: Shear modulus in Pa
            nu: Poisson's ratio
        
        Returns:
            (Stress, flag, flag2): Stress tensor (3,3,N) and validity flags
        """
        from cutde.halfspace import strain_free, strain_to_stress
        
        # Prepare observation points
        obs_pts = np.column_stack((xs, ys, zs))
        obs_pts = np.ascontiguousarray(obs_pts)
        
        # Get triangular geometry
        if fault.patchType == 'triangle':
            # Use existing triangular data
            tris = fault.Vertices[fault.Faces].copy()
            tri_slip = np.ascontiguousarray(slip_vec)
        else:
            # Convert rectangular to triangular
            Faces, Vertices, tri_slip = fault._rect2triangular(slipVec=slip_vec)
            tris = Vertices[Faces].copy()
            tri_slip = np.ascontiguousarray(tri_slip)
        
        # Convert z to positive upward for CUTDE
        tris[:, :, -1] *= -1.0
        
        # Compute strain and stress
        strain = strain_free(obs_pts, tris, tri_slip, nu)
        stress = strain_to_stress(strain, mu, nu)
        
        # Assemble stress tensor (3, 3, N)
        Stress = np.zeros((3, 3, len(xs)))
        Stress[0, 0, :] = stress[:, 0]  # σ_xx
        Stress[1, 1, :] = stress[:, 1]  # σ_yy
        Stress[2, 2, :] = stress[:, 2]  # σ_zz
        Stress[0, 1, :] = Stress[1, 0, :] = stress[:, 3]  # σ_xy
        Stress[0, 2, :] = Stress[2, 0, :] = stress[:, 4]  # σ_xz
        Stress[1, 2, :] = Stress[2, 1, :] = stress[:, 5]  # σ_yz
        
        flag = np.ones(len(xs), dtype=bool)
        flag2 = True
        
        return Stress, flag, flag2
    
    def _prepare_fault_data(self, fault, factor=0.001, slipdirection='sd', force_dip=None):
        """
        Extract and prepare fault geometry and slip data.
        
        Returns:
            tuple: (geometry, slips) where geometry is dict and slips is dict
        """
        nPatch = len(fault.patch)
        
        # Initialize arrays
        geometry = {
            'xc': np.zeros(nPatch),
            'yc': np.zeros(nPatch),
            'zc': np.zeros(nPatch),
            'width': np.zeros(nPatch),
            'length': np.zeros(nPatch),
            'strike': np.zeros(nPatch),
            'dip': np.zeros(nPatch)
        }
        
        slips = {
            'strike': np.zeros(nPatch),
            'dip': np.zeros(nPatch),
            'tensile': np.zeros(nPatch)
        }
        
        # Extract patch data
        for ii in range(nPatch):
            geometry['xc'][ii], geometry['yc'][ii], geometry['zc'][ii], \
            geometry['width'][ii], geometry['length'][ii], \
            geometry['strike'][ii], geometry['dip'][ii] = \
                fault.getpatchgeometry(fault.patch[ii], center=True)
            slips['strike'][ii], slips['dip'][ii], slips['tensile'][ii] = fault.slip[ii, :]
        
        # Apply conversion factor
        slips['strike'] *= factor
        slips['dip'] *= factor
        slips['tensile'] *= factor
        
        # Filter slip components
        if 's' not in slipdirection:
            slips['strike'][:] = 0.0
        if 'd' not in slipdirection:
            slips['dip'][:] = 0.0
        if 't' not in slipdirection:
            slips['tensile'][:] = 0.0
        
        # Override dip if specified
        if force_dip is not None:
            geometry['dip'][:] = force_dip
        
        return geometry, slips
    
    def _prepare_observation_points(self, source=None, receiver=None, stressonpatches=False):
        """
        Determine observation point locations.
        
        Returns:
            tuple: (xs, ys, zs) observation coordinates in ENU with z positive upward
        """
        if receiver is None:
            if stressonpatches:
                # Use source patch centers
                if source.patchType == 'triangle':
                    xyzc = np.mean(source.Vertices[source.Faces, :], axis=1)
                    xs, ys, zs = xyzc[:, 0], xyzc[:, 1], -xyzc[:, 2]
                else:  # rectangle
                    nPatch = len(source.patch)
                    xs = np.zeros(nPatch)
                    ys = np.zeros(nPatch)
                    zs = np.zeros(nPatch)
                    for ii in range(nPatch):
                        xs[ii], ys[ii], zs[ii], *_ = \
                            source.getpatchgeometry(source.patch[ii], center=True)
                    zs = -zs
            else:
                # Use predefined observation points
                xs = self.x
                ys = self.y
                zs = -self.depth
        else:
            # Use receiver fault centers
            if receiver.patchType == 'triangle':
                xyzc = np.mean(receiver.Vertices[receiver.Faces, :], axis=1)
                xs, ys, zs = xyzc[:, 0], xyzc[:, 1], -xyzc[:, 2]
            else:  # rectangle
                N_receiver = len(receiver.patch)
                xs = np.zeros(N_receiver)
                ys = np.zeros(N_receiver)
                zs = np.zeros(N_receiver)
                for ii in range(N_receiver):
                    xs[ii], ys[ii], zs[ii], *_ = \
                        receiver.getpatchgeometry(receiver.patch[ii], center=True)
                zs = -zs
        
        return xs, ys, zs

    def _compute_traction_matrix_method(self, source, receiver=None, strike=None, dip=None,
                                        factor=0.001, mu=30e9, nu=0.25, slipdirection='sd',
                                        force_dip=None, stressonpatches=False, 
                                        target_mem_gb=None, max_obs_batch=None, max_tri_batch=None,
                                        min_batch_count=5, verbose=False):
        """
        Compute stress tractions using strain_matrix method with automatic memory management.
        
        This method computes the stress response matrix and contracts it directly with
        slip vectors. Includes intelligent batching for large-scale problems.
        
        Args:
            source: Source fault (triangular only)
            receiver: Receiver fault (default: None)
            strike: Strike angle(s) in radians (default: None)
            dip: Dip angle(s) in radians (default: None)
            factor: Slip conversion factor (default: 0.001)
            mu: Shear modulus in Pa (default: 30 GPa)
            nu: Poisson's ratio (default: 0.25)
            slipdirection: Slip components (default: 'sd')
            force_dip: Override dip in radians (default: None)
            stressonpatches: Compute at patch centers (default: False)
            target_mem_gb: Maximum memory (GB) for strain_matrix. If None, auto-detect as 60% of RAM.
            max_obs_batch: Maximum observation points per batch. If None, auto-calculate.
            max_tri_batch: Maximum triangles per batch. If None, auto-calculate.
            min_batch_count: Minimum number of batches (default: 5)
            verbose: Print progress (default: False)
        
        Returns:
            (n1, n2, n3, T, Sigma, TauStrike, TauDip)
        
        Notes:
            - For large problems, automatically batches computation to fit memory constraints
            - Memory formula: strain_matrix requires (N_obs × 6 × N_tri × 3) × 8 bytes
            - Stress tensor adds: (N_obs × 3 × 3 × N_tri × 3) × 8 bytes
        """
        from cutde.halfspace import strain_matrix, strain_to_stress
        import psutil
        
        assert source.patchType == 'triangle', 'Matrix method only works with triangular faults!'
        
        if verbose:
            print(f'Computing stress tractions from {source.name} using strain_matrix method')
        
        # Get observation points
        obs_pts = self._get_observation_points_for_traction(
            source, receiver, strike, dip, stressonpatches
        )
        obs_pts = np.ascontiguousarray(obs_pts)
        
        # Prepare triangular patches (z positive upward for CUTDE)
        tris = source.Vertices[source.Faces].copy()
        tris[:, :, -1] *= -1.0
        
        N_obs = obs_pts.shape[0]
        N_tri = tris.shape[0]
        
        # Get receiver plane geometry
        strike_rad, dip_rad = self._get_receiver_geometry(
            receiver, strike, dip, force_dip, N_obs
        )
        
        # Compute unit vectors
        n1, n2, n3 = self.strikedip2normal(strike_rad, dip_rad)
        
        # Prepare slip vectors
        slip = self._prepare_slip_vector(source, factor, slipdirection, N_tri)
        
        # Auto-detect available memory
        if target_mem_gb is None:
            available_gb = psutil.virtual_memory().available / (1024**3)
            target_mem_gb = available_gb * 0.6
            if verbose:
                print(f"Auto-detected target memory: {target_mem_gb:.2f} GB")
        
        # Memory calculation
        bytes_per_element = 8
        # strain_matrix: (N_obs, 6, N_tri, 3)
        mem_per_strain = 6 * 3 * bytes_per_element  # 144 bytes per obs-tri pair
        # stress_tensor: (N_obs, 3, 3, N_tri, 3)
        mem_per_stress = 3 * 3 * 3 * bytes_per_element  # 216 bytes per obs-tri pair
        mem_per_obs_tri = mem_per_strain + mem_per_stress  # ~360 bytes total
        
        # Full matrix memory requirement
        bytes_needed = N_obs * N_tri * mem_per_obs_tri
        gb_needed = bytes_needed / (1024**3)
        
        # Track user input
        user_provided_obs = max_obs_batch is not None
        user_provided_tri = max_tri_batch is not None
        
        # Calculate maximum obs with ALL triangles
        max_obs_with_full_tri = int((target_mem_gb * (1024**3)) / (N_tri * mem_per_obs_tri))
        
        # Intelligent batch size calculation
        if not user_provided_obs and not user_provided_tri:
            # Fully automatic
            if max_obs_with_full_tri >= N_obs:
                max_obs_batch = N_obs
                max_tri_batch = N_tri
            else:
                max_obs_batch = min(max_obs_with_full_tri, 100000)
                
                # Apply minimum batch count
                n_batches_candidate = (N_obs + max_obs_batch - 1) // max_obs_batch
                if n_batches_candidate < min_batch_count:
                    max_obs_batch = (N_obs + min_batch_count - 1) // min_batch_count
                
                max_obs_batch = max(2000, max_obs_batch)
                
                # Check if tri_batch needed
                mem_per_batch = (max_obs_batch * N_tri * mem_per_obs_tri) / (1024**3)
                if mem_per_batch <= target_mem_gb:
                    max_tri_batch = N_tri
                else:
                    max_tri_batch = int((target_mem_gb * (1024**3)) / (max_obs_batch * mem_per_obs_tri))
                    max_tri_batch = max(100, min(max_tri_batch, N_tri))
        
        elif user_provided_obs and not user_provided_tri:
            # Only obs provided
            mem_per_batch = (max_obs_batch * N_tri * mem_per_obs_tri) / (1024**3)
            
            if mem_per_batch > target_mem_gb:
                if verbose:
                    print(f"WARNING: obs_batch={max_obs_batch:,} requires {mem_per_batch:.2f} GB")
                max_tri_batch = int((target_mem_gb * (1024**3)) / (max_obs_batch * mem_per_obs_tri))
                max_tri_batch = max(100, min(max_tri_batch, N_tri))
            else:
                max_tri_batch = N_tri
        
        elif not user_provided_obs and user_provided_tri:
            # Only tri provided
            max_obs_with_tri = int((target_mem_gb * (1024**3)) / (max_tri_batch * mem_per_obs_tri))
            
            if max_obs_with_tri >= N_obs:
                max_obs_batch = N_obs
            else:
                max_obs_batch = min(max_obs_with_tri, 100000)
                n_batches_candidate = (N_obs + max_obs_batch - 1) // max_obs_batch
                if n_batches_candidate < min_batch_count:
                    max_obs_batch = (N_obs + min_batch_count - 1) // min_batch_count
                max_obs_batch = max(2000, max_obs_batch)
        
        else:
            # Both provided - check memory
            mem_per_batch = (max_obs_batch * max_tri_batch * mem_per_obs_tri) / (1024**3)
            
            if mem_per_batch > target_mem_gb:
                if verbose:
                    print(f"WARNING: User batch sizes require {mem_per_batch:.2f} GB, recalculating...")
                # Recalculate automatically
                if max_obs_with_full_tri >= N_obs:
                    max_obs_batch = N_obs
                    max_tri_batch = N_tri
                else:
                    max_obs_batch = min(max_obs_with_full_tri, 100000)
                    n_batches_candidate = (N_obs + max_obs_batch - 1) // max_obs_batch
                    if n_batches_candidate < min_batch_count:
                        max_obs_batch = (N_obs + min_batch_count - 1) // min_batch_count
                    max_obs_batch = max(2000, max_obs_batch)
                    
                    mem_per_batch = (max_obs_batch * N_tri * mem_per_obs_tri) / (1024**3)
                    if mem_per_batch <= target_mem_gb:
                        max_tri_batch = N_tri
                    else:
                        max_tri_batch = int((target_mem_gb * (1024**3)) / (max_obs_batch * mem_per_obs_tri))
                        max_tri_batch = max(100, min(max_tri_batch, N_tri))
        
        # Calculate batch info
        n_obs_batches = (N_obs + max_obs_batch - 1) // max_obs_batch
        n_tri_batches = (N_tri + max_tri_batch - 1) // max_tri_batch
        mem_per_iter = (max_obs_batch * max_tri_batch * mem_per_obs_tri) / (1024**3)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Stress Traction Matrix Calculation Summary")
            print(f"{'='*70}")
            print(f"Observation points: {N_obs:,}")
            print(f"Source triangles: {N_tri:,}")
            print(f"Estimated memory for full matrix: {gb_needed:.2f} GB")
            print(f"Target memory limit: {target_mem_gb:.2f} GB")
            print(f"Observation batch size: {max_obs_batch:,}")
            print(f"Triangle batch size: {max_tri_batch:,}")
            print(f"Memory per iteration: {mem_per_iter:.2f} GB")
            print(f"Total batches: {n_obs_batches} obs × {n_tri_batches} tri = {n_obs_batches * n_tri_batches} iterations")
            print(f"{'='*70}\n")
        
        # Choose computation strategy
        use_direct = gb_needed <= target_mem_gb and N_obs <= max_obs_batch and N_tri <= max_tri_batch
        
        if use_direct:
            if verbose:
                print("Method: Direct computation (single pass)")
            
            T, Sigma, TauStrike, TauDip = self._compute_traction_direct(
                obs_pts, tris, nu, mu, n1, n2, n3, slip
            )
        else:
            if verbose:
                print("Method: Batched computation")
            
            T, Sigma, TauStrike, TauDip = self._compute_traction_batched(
                obs_pts, tris, nu, mu, n1, n2, n3, slip,
                max_obs_batch, max_tri_batch, verbose
            )
        
        # Store results
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.T, self.Sigma = T, Sigma
        self.TauStrike, self.TauDip = TauStrike, TauDip
        
        if verbose:
            print('Traction computation completed (matrix method).\n')
        
        return n1, n2, n3, T, Sigma, TauStrike, TauDip

    def _get_observation_points_for_traction(self, source, receiver, strike, dip, stressonpatches):
        """Helper to get observation points for traction computation."""
        if receiver is None:
            if stressonpatches:
                xyzc = np.mean(source.Vertices[source.Faces, :], axis=1)
                obs_pts = np.column_stack((xyzc[:, 0], xyzc[:, 1], -xyzc[:, 2]))
            else:
                obs_pts = np.column_stack((self.x, self.y, -self.depth))
                assert strike is not None and dip is not None, \
                    'Must provide strike and dip when receiver is None and stressonpatches is False!'
        else:
            if receiver.patchType == 'triangle':
                xyzc = np.mean(receiver.Vertices[receiver.Faces, :], axis=1)
                obs_pts = np.column_stack((xyzc[:, 0], xyzc[:, 1], -xyzc[:, 2]))
            else:
                N_receiver = len(receiver.patch)
                xs = np.zeros(N_receiver)
                ys = np.zeros(N_receiver)
                zs = np.zeros(N_receiver)
                for ii in range(N_receiver):
                    xs[ii], ys[ii], zs[ii], *_ = \
                        receiver.getpatchgeometry(receiver.patch[ii], center=True)
                obs_pts = np.column_stack((xs, ys, -zs))
        
        return obs_pts

    def _get_receiver_geometry(self, receiver, strike, dip, force_dip, N_obs):
        """Helper to get receiver plane geometry."""
        if strike is not None and dip is not None:
            strike_rad = np.ones(N_obs) * strike
            dip_rad = np.ones(N_obs) * dip
        else:
            strike_rad = receiver.getStrikes()
            dip_rad = receiver.getDips()
        
        if force_dip is not None:
            dip_rad[:] = force_dip
        
        return strike_rad, dip_rad

    def _prepare_slip_vector(self, source, factor, slipdirection, N_tri):
        """Helper to prepare slip vectors."""
        strikeslip = np.zeros(N_tri)
        dipslip = np.zeros(N_tri)
        tensileslip = np.zeros(N_tri)
        
        if 's' in slipdirection:
            strikeslip = source.slip[:, 0] * factor
        if 'd' in slipdirection:
            dipslip = source.slip[:, 1] * factor
        if 't' in slipdirection:
            tensileslip = source.slip[:, 2] * factor
        
        return np.column_stack((strikeslip, dipslip, tensileslip))

    def _compute_traction_direct(self, obs_pts, tris, nu, mu, n1, n2, n3, slip):
        '''
        Compute stress tractions on a plane with given strike and dip angles using CUTDE for triangular faults.
        
        This method calculates the stress traction components (normal and shear) on a receiver plane 
        caused by slip on triangular source patches. It uses the CUTDE (Curve Triangle Dislocation Elements) 
        method to compute the full stress tensor and then projects it onto the specified receiver plane.

        Args:
            source : TriangularFault, optional
                Source fault object containing triangular patches with slip distribution.
                Must be provided when receiver is None and stressonpatches is True.
            
            receiver : TriangularFault/RectangularFault, optional
                Receiver fault object defining the planes where tractions are computed.
                If None, uses either source patch centers (if stressonpatches=True) or 
                predefined observation points (self.x, self.y, self.depth).
            
            strike : float or array, optional
                Strike angle(s) in radians.
                - If float: all receiver planes share the same strike
                - If array: must match the number of observation points
                - Required when receiver is None and stressonpatches is False
                Default is None.
            
            dip : float or array, optional
                Dip angle(s) in radians.
                - If float: all receiver planes share the same dip
                - If array: must match the number of observation points
                - Required when receiver is None and stressonpatches is False
                Default is None.

        Kwargs:
            factor : float, optional
                Conversion factor between slip units and distance units.
                Default is 0.001 (e.g., distances in km, slip in m).
                Example: Use 1e-6 if distances are in km and slip in mm.
            
            mu : float, optional
                Shear modulus in Pa. Default is 30e9 (30 GPa).
            
            nu : float, optional
                Poisson's ratio (dimensionless). Default is 0.25.
            
            slipdirection : str, optional
                String indicating which slip components to include:
                - 's': strike-slip component
                - 'd': dip-slip component
                - 't': tensile component
                - Any combination, e.g., 'sd' for strike and dip slip only
                Default is 'sd'.
            
            force_dip : float, optional
                If provided, overrides dip angles for all receiver planes (in radians).
                Default is None (uses original dip angles).
            
            stressonpatches : bool, optional
                If True, computes tractions at the center of source fault patches.
                If False, uses predefined observation points or receiver fault centers.
                Default is False.

        Returns:
            tuple : (n1, n2, n3, T, Sigma, TauStrike, TauDip)
                n1 : ndarray, shape (3, N_obs)
                    Unit normal vectors (positive = opening/extension).
                
                n2 : ndarray, shape (3, N_obs)
                    Unit strike direction vectors (positive = left-lateral/sinistral).
                
                n3 : ndarray, shape (3, N_obs)
                    Unit dip direction vectors (positive = reverse/thrust).
                
                T : ndarray, shape (N_obs, 3)
                    Total traction vectors at observation points.
                
                Sigma : ndarray, shape (N_obs,)
                    Normal traction components (positive = tensile/extension).
                
                TauStrike : ndarray, shape (N_obs,)
                    Strike-parallel shear traction (positive = left-lateral).
                
                TauDip : ndarray, shape (N_obs,)
                    Dip-parallel shear traction (positive = reverse).

        Notes:
            - Only compatible with TriangularFault objects using CUTDE
            - Coordinate system:
                * Input: ENU (East-North-Up), z positive downward for depths
                * CUTDE: z positive upward (automatically converted)
            - Sign conventions:
                * Normal traction: positive = extension/opening
                * Strike-slip traction: positive = left-lateral/sinistral
                * Dip-slip traction: positive = reverse/thrust
            - Results are stored in instance attributes:
                self.n1, self.n2, self.n3: Direction vectors
                self.T: Total traction tensor
                self.Sigma: Normal traction
                self.TauStrike: Strike-parallel shear traction
                self.TauDip: Dip-parallel shear traction

        Examples:
            >>> # Compute tractions on receiver fault from source fault
            >>> sf = stressfield('mystress')
            >>> n1, n2, n3, T, Sigma, TauStrike, TauDip = sf.computeTriangularStressTraction(
            ...     source=source_fault, 
            ...     receiver=receiver_fault,
            ...     factor=1e-3
            ... )
            
            >>> # Compute tractions on a uniform plane
            >>> sf.setLonLatZ(lon, lat, depth)
            >>> sf.computeTriangularStressTraction(
            ...     source=source_fault,
            ...     strike=np.pi/4,  # 45 degrees
            ...     dip=np.pi/3,     # 60 degrees
            ...     slipdirection='sd'
            ... )
            >>> normal_stress = sf.Sigma  # Access normal traction
            >>> shear_stress = sf.TauStrike  # Access strike-slip traction

        Raises:
            AssertionError: If source.patchType is not 'triangle'
            AssertionError: If required parameters are not provided for specific configurations

        References:
            - Written by kfhe, 2021-10-24
            - Modified by kfhe, 2024-08-16
            - Nikkhoo, M., & Walter, T. R. (2015). Triangular dislocation: an analytical, 
            artefact-free solution. Geophysical Journal International, 201(2), 1119-1141.
        '''
        from cutde.halfspace import strain_matrix, strain_to_stress
        
        # Compute strain matrix
        strain_mat = strain_matrix(obs_pts, tris, nu)
        
        # Convert to stress
        strain_mat_reshaped = strain_mat.transpose(0, 2, 3, 1)
        stress_mat_reshaped = strain_to_stress(
            strain_mat_reshaped.reshape((-1, 6)), mu, nu
        )
        stress_mat = stress_mat_reshaped.reshape(strain_mat_reshaped.shape).transpose(0, 3, 1, 2)
        
        # Assemble stress tensor
        N_obs = obs_pts.shape[0]
        N_tri = tris.shape[0]
        stress_tensor = np.zeros((N_obs, 3, 3, N_tri, 3))
        stress_tensor[:, 0, 0, :, :] = stress_mat[:, 0, :, :]
        stress_tensor[:, 1, 1, :, :] = stress_mat[:, 1, :, :]
        stress_tensor[:, 2, 2, :, :] = stress_mat[:, 2, :, :]
        stress_tensor[:, 0, 1, :, :] = stress_tensor[:, 1, 0, :, :] = stress_mat[:, 3, :, :]
        stress_tensor[:, 0, 2, :, :] = stress_tensor[:, 2, 0, :, :] = stress_mat[:, 4, :, :]
        stress_tensor[:, 1, 2, :, :] = stress_tensor[:, 2, 1, :, :] = stress_mat[:, 5, :, :]
        
        # Compute tractions
        T = np.einsum('ik,iklmn->ilmn', n1.T, stress_tensor)
        Sigma = np.einsum('ijlm,ij->ilm', T, n1.T)
        TauStrike = np.einsum('ijlm,ij->ilm', T, n2.T)
        TauDip = np.einsum('ijlm,ij->ilm', T, n3.T)
        
        # Contract with slip
        T = np.einsum('iklm,lm->ik', T, slip)
        Sigma = np.einsum('ijk,jk->i', Sigma, slip)
        TauStrike = np.einsum('ijk,jk->i', TauStrike, slip)
        TauDip = np.einsum('ijk,jk->i', TauDip, slip)
        
        return T, Sigma, TauStrike, TauDip

    def _compute_traction_batched(self, obs_pts, tris, nu, mu, n1, n2, n3, slip,
                                   max_obs_batch, max_tri_batch, verbose):
        """Batched computation for large problems."""
        from cutde.halfspace import strain_matrix, strain_to_stress
        from tqdm import tqdm
        
        N_obs = obs_pts.shape[0]
        N_tri = tris.shape[0]
        
        # Initialize result arrays
        T = np.zeros((N_obs, 3))
        Sigma = np.zeros(N_obs)
        TauStrike = np.zeros(N_obs)
        TauDip = np.zeros(N_obs)
        
        # Create batches
        obs_batches = [(i, min(i + max_obs_batch, N_obs)) 
                       for i in range(0, N_obs, max_obs_batch)]
        tri_batches = [(j, min(j + max_tri_batch, N_tri)) 
                       for j in range(0, N_tri, max_tri_batch)]
        
        total_iterations = len(obs_batches) * len(tri_batches)
        
        if verbose:
            progress = tqdm(total=total_iterations, desc="Computing tractions")
        
        for obs_start, obs_end in obs_batches:
            obs_batch = obs_pts[obs_start:obs_end]
            n1_batch = n1[:, obs_start:obs_end]
            n2_batch = n2[:, obs_start:obs_end]
            n3_batch = n3[:, obs_start:obs_end]
            
            # Accumulate contributions from all triangle batches
            T_batch = np.zeros((obs_end - obs_start, 3))
            Sigma_batch = np.zeros(obs_end - obs_start)
            TauStrike_batch = np.zeros(obs_end - obs_start)
            TauDip_batch = np.zeros(obs_end - obs_start)
            
            for tri_start, tri_end in tri_batches:
                tri_batch = tris[tri_start:tri_end]
                slip_batch = slip[tri_start:tri_end]
                
                # Compute strain matrix for this batch
                strain_mat = strain_matrix(obs_batch, tri_batch, nu)
                
                # Convert to stress
                strain_mat_reshaped = strain_mat.transpose(0, 2, 3, 1)
                stress_mat_reshaped = strain_to_stress(
                    strain_mat_reshaped.reshape((-1, 6)), mu, nu
                )
                stress_mat = stress_mat_reshaped.reshape(strain_mat_reshaped.shape).transpose(0, 3, 1, 2)
                
                # Assemble stress tensor
                n_obs_local = obs_batch.shape[0]
                n_tri_local = tri_batch.shape[0]
                stress_tensor = np.zeros((n_obs_local, 3, 3, n_tri_local, 3))
                stress_tensor[:, 0, 0, :, :] = stress_mat[:, 0, :, :]
                stress_tensor[:, 1, 1, :, :] = stress_mat[:, 1, :, :]
                stress_tensor[:, 2, 2, :, :] = stress_mat[:, 2, :, :]
                stress_tensor[:, 0, 1, :, :] = stress_tensor[:, 1, 0, :, :] = stress_mat[:, 3, :, :]
                stress_tensor[:, 0, 2, :, :] = stress_tensor[:, 2, 0, :, :] = stress_mat[:, 4, :, :]
                stress_tensor[:, 1, 2, :, :] = stress_tensor[:, 2, 1, :, :] = stress_mat[:, 5, :, :]
                
                # Compute tractions
                T_local = np.einsum('ik,iklmn->ilmn', n1_batch.T, stress_tensor)
                Sigma_local = np.einsum('ijlm,ij->ilm', T_local, n1_batch.T)
                TauStrike_local = np.einsum('ijlm,ij->ilm', T_local, n2_batch.T)
                TauDip_local = np.einsum('ijlm,ij->ilm', T_local, n3_batch.T)
                
                # Contract with slip and accumulate
                T_batch += np.einsum('iklm,lm->ik', T_local, slip_batch)
                Sigma_batch += np.einsum('ijk,jk->i', Sigma_local, slip_batch)
                TauStrike_batch += np.einsum('ijk,jk->i', TauStrike_local, slip_batch)
                TauDip_batch += np.einsum('ijk,jk->i', TauDip_local, slip_batch)
                
                if verbose:
                    progress.update(1)
            
            # Store batch results
            T[obs_start:obs_end] = T_batch
            Sigma[obs_start:obs_end] = Sigma_batch
            TauStrike[obs_start:obs_end] = TauStrike_batch
            TauDip[obs_start:obs_end] = TauDip_batch
        
        if verbose:
            progress.close()
        
        return T, Sigma, TauStrike, TauDip
    
    def compute_stress_drop_andrews(self, slip_strike, slip_dip, dx, dy, 
                                    mu=30e9, lmbda=30e9, return_unit='MPa', verbose=False):
        """
        Compute static stress drop using Andrews (1980) FFT method for planar faults.
        
        This method assumes a planar fault with regular grid spacing and uses FFT 
        for efficient computation of stress drop from slip distribution.
        
        Args:
            slip_strike: 2D array of strike-slip distribution (unit: m)
            slip_dip: 2D array of dip-slip distribution (unit: m)
            dx: Grid spacing in strike direction (unit: km)
            dy: Grid spacing in dip direction (unit: km)
            mu: Shear modulus in Pa (default: 30 GPa)
            lmbda: Lame's first parameter in Pa (default: 30 GPa)
            return_unit: Output unit ('MPa', 'kPa', 'Pa', 'bar', default: 'MPa')
            verbose: Print progress (default: False)
        
        Returns:
            dict: {
                'stress_drop_strike': 2D array of strike-direction stress drop,
                'stress_drop_dip': 2D array of dip-direction stress drop,
                'avg_stress_drop_strike': Slip-weighted average (strike),
                'avg_stress_drop_dip': Slip-weighted average (dip),
                'avg_stress_drop_total': Total slip-weighted average
            }
        
        Notes:
            - Input slip arrays must be 2D with shape (Ny, Nx)
            - Grid spacing dx, dy must be in kilometers
            - Method assumes planar fault geometry
            - Based on Andrews (1980) and Ripperger & Mai (2004)
        
        References:
            - Andrews, D. J. (1980). A stochastic fault model: 1. Static case. 
              Journal of Geophysical Research, 85(B7), 3867-3877.
            - Ripperger, J., & Mai, P. M. (2004). Fast computation of static stress 
              changes on 2D faults from final slip distributions. Geophysical Research 
              Letters, 31(18), L18610.
        
        Examples:
            >>> # Create slip distribution
            >>> Ny, Nx = 128, 128
            >>> slip_s = np.random.rand(Ny, Nx)
            >>> slip_d = np.random.rand(Ny, Nx) * 0.5
            >>> 
            >>> # Compute stress drop
            >>> result = sf.compute_stress_drop_andrews(
            ...     slip_s, slip_d, dx=0.5, dy=0.5, return_unit='MPa', verbose=True
            ... )
            >>> 
            >>> # Plot results
            >>> import matplotlib.pyplot as plt
            >>> plt.imshow(result['stress_drop_strike'], cmap='seismic')
            >>> plt.colorbar(label='Stress Drop (MPa)')
            >>> plt.show()
        """
        from numpy.fft import fft2, ifft2, fftfreq
        
        if verbose:
            print("Computing stress drop using Andrews (1980) FFT method...")
        
        # Check input dimensions
        if slip_strike.ndim != 2 or slip_dip.ndim != 2:
            raise ValueError("Input slip arrays must be 2D")
        
        if slip_strike.shape != slip_dip.shape:
            raise ValueError("Strike-slip and dip-slip arrays must have same shape")
        
        Ny, Nx = slip_strike.shape
        
        # Convert grid spacing from km to m
        dx_m = dx * 1000.0
        dy_m = dy * 1000.0
        
        if verbose:
            print(f"  Grid size: {Ny} × {Nx}")
            print(f"  Grid spacing: {dx:.3f} km × {dy:.3f} km")
        
        # FFT of slip distribution
        D_strike_k = fft2(slip_strike)
        D_dip_k = fft2(slip_dip)
        
        # Build wavenumber grids
        kx = 2 * np.pi * fftfreq(Nx, d=dx_m)
        ky = 2 * np.pi * fftfreq(Ny, d=dy_m)
        KX, KY = np.meshgrid(kx, ky)
        
        # Calculate wavenumber magnitude
        K_sq = KX**2 + KY**2
        K_abs = np.sqrt(K_sq)
        K_abs[K_abs == 0] = 1.0e-20  # Avoid division by zero
        
        # Stiffness coefficient
        alpha = 2 * (lmbda + mu) / (lmbda + 2 * mu)
        
        if verbose:
            print(f"  Material: μ = {mu/1e9:.1f} GPa, λ = {lmbda/1e9:.1f} GPa")
            print(f"  Stiffness coefficient α = {alpha:.4f}")
        
        def get_stiffness(k_par, k_perp, k_abs):
            """Compute parallel and perpendicular stiffness."""
            # Parallel stiffness: K_par = -0.5 * μ * [α*k_par² + k_perp²] / |k|
            term1 = alpha * (k_par**2) + (k_perp**2)
            K_par = -0.5 * mu * term1 / k_abs
            
            # Perpendicular stiffness: K_perp = -0.5 * μ * (α-1) * k_par * k_perp / |k|
            term2 = (alpha - 1) * k_par * k_perp
            K_perp = -0.5 * mu * term2 / k_abs
            
            # Set k=0 components to zero
            K_par[0, 0] = 0
            K_perp[0, 0] = 0
            
            return K_par, K_perp
        
        # Compute stress from strike-slip
        K_par_S, K_perp_S = get_stiffness(KX, KY, K_abs)
        sigma_strike_from_strike_k = K_par_S * D_strike_k
        sigma_dip_from_strike_k = K_perp_S * D_strike_k
        
        # Compute stress from dip-slip (swap k_x and k_y)
        K_par_D, K_perp_D = get_stiffness(KY, KX, K_abs)
        sigma_dip_from_dip_k = K_par_D * D_dip_k
        sigma_strike_from_dip_k = K_perp_D * D_dip_k
        
        # Superpose total stress in wavenumber domain
        total_sigma_strike_k = sigma_strike_from_strike_k + sigma_strike_from_dip_k
        total_sigma_dip_k = sigma_dip_from_dip_k + sigma_dip_from_strike_k
        
        # Inverse FFT to spatial domain
        stress_drop_strike = np.real(ifft2(total_sigma_strike_k))
        stress_drop_dip = np.real(ifft2(total_sigma_dip_k))
        
        # Compute slip-weighted averages
        avg_strike, avg_dip, avg_total = self._compute_slip_weighted_average(
            slip_strike.flatten(), slip_dip.flatten(),
            stress_drop_strike.flatten(), stress_drop_dip.flatten()
        )
        
        # Convert units
        unit_factor = self._get_unit_factor(return_unit)
        stress_drop_strike /= unit_factor
        stress_drop_dip /= unit_factor
        avg_strike /= unit_factor
        avg_dip /= unit_factor
        avg_total /= unit_factor
        
        if verbose:
            print(f"\nStress Drop Results ({return_unit}):")
            print(f"  Average (strike):  {avg_strike:.3f}")
            print(f"  Average (dip):     {avg_dip:.3f}")
            print(f"  Average (total):   {avg_total:.3f}")
            print(f"  Max (strike):      {np.abs(stress_drop_strike).max():.3f}")
            print(f"  Max (dip):         {np.abs(stress_drop_dip).max():.3f}")
        
        return {
            'stress_drop_strike': stress_drop_strike,
            'stress_drop_dip': stress_drop_dip,
            'avg_stress_drop_strike': avg_strike,
            'avg_stress_drop_dip': avg_dip,
            'avg_stress_drop_total': avg_total
        }
    
    def compute_stress_drop_from_fault(self, fault, method='direct', mu=30e9, lmbda=30e9,
                                       factor=0.001, return_unit='MPa', verbose=False):
        """
        Compute static stress drop on fault patches using direct computation.
        
        This method works for arbitrary fault geometries (triangular or rectangular)
        by computing the full stress tensor at each patch center.
        
        Args:
            fault: Fault object with slip distribution
            method: Computation method (only 'direct' supported)
            mu: Shear modulus in Pa (default: 30 GPa)
            lmbda: Lame's first parameter in Pa (default: 30 GPa)
            factor: Slip unit conversion factor (default: 0.001)
            return_unit: Output unit ('MPa', 'kPa', 'Pa', 'bar')
            verbose: Print progress (default: False)
        
        Returns:
            dict: {
                'stress_drop_strike': 1D array of strike-direction stress drop,
                'stress_drop_dip': 1D array of dip-direction stress drop,
                'avg_stress_drop_strike': Slip-weighted average (strike),
                'avg_stress_drop_dip': Slip-weighted average (dip),
                'avg_stress_drop_total': Total slip-weighted average
            }
        
        Notes:
            - Works for both triangular and rectangular fault patches
            - For planar faults with regular grids, consider using 
              compute_stress_drop_andrews() for better performance
        
        Examples:
            >>> # Compute stress drop on fault
            >>> result = sf.compute_stress_drop_from_fault(
            ...     fault, method='direct', return_unit='MPa', verbose=True
            ... )
            >>> 
            >>> # Plot on fault
            >>> sf.plot_stress_drop(fault, result, return_unit='MPa')
        """
        if verbose:
            print(f"Computing stress drop on {fault.name} using {method} method...")
        
        if method != 'direct':
            raise ValueError("Only 'direct' method is supported. Use compute_stress_drop_andrews() for FFT method.")
        
        nu = lmbda / (2 * (lmbda + mu))  # Poisson's ratio
        
        if verbose:
            print(f"  Number of patches: {len(fault.patch)}")
            print(f"  Material: μ = {mu/1e9:.1f} GPa, λ = {lmbda/1e9:.1f} GPa, ν = {nu:.3f}")
        
        # Compute stress tractions at patch centers using existing method
        # This computes the stress at each patch center due to all other patches
        n1, n2, n3, T, Sigma, TauStrike, TauDip = self.computeStressTraction(
            source=fault,
            receiver=fault,
            factor=factor,
            mu=mu,
            nu=nu,
            slipdirection='sd',
            stressonpatches=True,
            verbose=False
        )
        
        # The traction components are already the stress drop components
        stress_drop_strike = TauStrike
        stress_drop_dip = TauDip
        
        # Get slip for weighted averaging
        slip_strike = fault.slip[:, 0] * factor
        slip_dip = fault.slip[:, 1] * factor
        
        # Compute slip-weighted averages
        avg_strike, avg_dip, avg_total = self._compute_slip_weighted_average(
            slip_strike, slip_dip, stress_drop_strike, stress_drop_dip
        )
        
        # Convert units
        unit_factor = self._get_unit_factor(return_unit)
        stress_drop_strike /= unit_factor
        stress_drop_dip /= unit_factor
        avg_strike /= unit_factor
        avg_dip /= unit_factor
        avg_total /= unit_factor
        
        if verbose:
            print(f"\nStress Drop Results ({return_unit}):")
            print(f"  Average (strike):  {avg_strike:.3f}")
            print(f"  Average (dip):     {avg_dip:.3f}")
            print(f"  Average (total):   {avg_total:.3f}")
            print(f"  Max (strike):      {np.abs(stress_drop_strike).max():.3f}")
            print(f"  Max (dip):         {np.abs(stress_drop_dip).max():.3f}")
        
        return {
            'stress_drop_strike': stress_drop_strike,
            'stress_drop_dip': stress_drop_dip,
            'avg_stress_drop_strike': avg_strike,
            'avg_stress_drop_dip': avg_dip,
            'avg_stress_drop_total': avg_total
        }
    
    def _compute_slip_weighted_average(self, slip_strike, slip_dip,
                                       stress_drop_strike, stress_drop_dip):
        """
        Compute slip-weighted average stress drop.
        
        Args:
            slip_strike: Strike-slip component (m)
            slip_dip: Dip-slip component (m)
            stress_drop_strike: Strike-direction stress drop (Pa)
            stress_drop_dip: Dip-direction stress drop (Pa)
        
        Returns:
            (avg_strike, avg_dip, avg_total): Weighted averages (Pa)
        """
        # Total slip magnitude
        total_slip = np.sqrt(slip_strike**2 + slip_dip**2)
        
        # Strike-weighted average
        sum_slip_strike_abs = np.sum(np.abs(slip_strike))
        if sum_slip_strike_abs > 0:
            avg_strike = -np.sum(stress_drop_strike * slip_strike) / sum_slip_strike_abs
        else:
            avg_strike = 0.0
        
        # Dip-weighted average
        sum_slip_dip_abs = np.sum(np.abs(slip_dip))
        if sum_slip_dip_abs > 0:
            avg_dip = -np.sum(stress_drop_dip * slip_dip) / sum_slip_dip_abs
        else:
            avg_dip = 0.0
        
        # Total weighted average
        sum_total_slip = np.sum(total_slip)
        if sum_total_slip > 0:
            stress_slip_product = stress_drop_strike * slip_strike + stress_drop_dip * slip_dip
            avg_total = -np.sum(stress_slip_product) / sum_total_slip
        else:
            avg_total = 0.0
        
        return avg_strike, avg_dip, avg_total
    
    def _get_unit_factor(self, return_unit):
        """Get conversion factor for stress units."""
        return_unit = return_unit.lower()
        if return_unit == 'mpa':
            return 1e6
        elif return_unit == 'kpa':
            return 1e3
        elif return_unit == 'bar':
            return 1e5
        elif return_unit == 'pa':
            return 1.0
        else:
            raise ValueError(f"Invalid unit: {return_unit}. Choose 'MPa', 'kPa', 'Pa', or 'bar'.")
    
    def plot_stress_drop(self, fault, stress_drop_result, return_unit='MPa',
                         figsize=(12, 5), cmap='seismic', vmin=None, vmax=None,
                         savefig=False, figname='stress_drop.png', dpi=300):
        """
        Plot stress drop distribution on fault.
        
        Args:
            fault: Fault object
            stress_drop_result: Dict returned by compute_stress_drop_*()
            return_unit: Unit label (default: 'MPa')
            figsize: Figure size (default: (12, 5))
            cmap: Colormap (default: 'seismic')
            vmin, vmax: Color limits (default: auto)
            savefig: Save figure (default: False)
            figname: Output filename (default: 'stress_drop.png')
            dpi: Resolution (default: 300)
        """
        import matplotlib.pyplot as plt
        from ..plottools import sci_plot_style
        
        stress_strike = stress_drop_result['stress_drop_strike']
        stress_dip = stress_drop_result['stress_drop_dip']
        avg_strike = stress_drop_result['avg_stress_drop_strike']
        avg_dip = stress_drop_result['avg_stress_drop_dip']
        avg_total = stress_drop_result['avg_stress_drop_total']
        
        # Get fault coordinates
        if fault.patchType == 'triangle':
            xc = np.mean(fault.Vertices[fault.Faces, :], axis=1)[:, 0]
            yc = np.mean(fault.Vertices[fault.Faces, :], axis=1)[:, 1]
        else:
            N = len(fault.patch)
            xc = np.zeros(N)
            yc = np.zeros(N)
            for i in range(N):
                xc[i], yc[i], *_ = fault.getpatchgeometry(fault.patch[i], center=True)
        
        with sci_plot_style():
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Plot strike-direction stress drop
            sc1 = axes[0].scatter(xc/1000, yc/1000, c=stress_strike.flatten(),
                                 cmap=cmap, vmin=vmin, vmax=vmax, s=50)
            axes[0].set_title(f'Strike Stress Drop\nAvg: {avg_strike:.2f} {return_unit}')
            axes[0].set_xlabel('X (km)')
            axes[0].set_ylabel('Y (km)')
            axes[0].set_aspect('equal')
            plt.colorbar(sc1, ax=axes[0], label=f'Stress Drop ({return_unit})')
            
            # Plot dip-direction stress drop
            sc2 = axes[1].scatter(xc/1000, yc/1000, c=stress_dip.flatten(),
                                 cmap=cmap, vmin=vmin, vmax=vmax, s=50)
            axes[1].set_title(f'Dip Stress Drop\nAvg: {avg_dip:.2f} {return_unit}')
            axes[1].set_xlabel('X (km)')
            axes[1].set_ylabel('Y (km)')
            axes[1].set_aspect('equal')
            plt.colorbar(sc2, ax=axes[1], label=f'Stress Drop ({return_unit})')
            
            plt.suptitle(f'Total Slip-Weighted Average: {avg_total:.2f} {return_unit}',
                        fontsize=12, y=1.02)
            plt.tight_layout()
            
            if savefig:
                plt.savefig(figname, dpi=dpi, bbox_inches='tight')
            plt.show()
    
    # EOF