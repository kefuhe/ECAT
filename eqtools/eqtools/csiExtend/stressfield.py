'''
A class that deals with StressField data.

Written by R. Jolivet, Feb 2014.

Modified by kfhe, at 10/24/2022
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

    def fault2Stress_cutde(self, fault, factor=0.001, mu=30e9, nu=0.25, slipdirection='sd', force_dip=None, stressonpatches=False, verbose=False):
        '''
        Takes a fault, or a list of faults, and computes the stress change associated with the slip on the fault.

        Args:   
            * fault             : Fault object (RectangularFault or TriangularFault).

        Kwargs:
            * factor            : Conversion factor between the slip units and distance units. Usually, distances are in Km. Therefore, if slip is in mm, then factor=1e-6.
            * slipdirection     : any combination of s, d, and t.
            * mu                : Shear Modulus (default is 30GPa).
            * nu                : Poisson's ratio (default is 0.25).
            * stressonpatches   : Re-sets the station locations to be where the center of the patches are.
            * force_dip         : Specify the dip angle of the patches
            * verbos            : talk to me

        Returns:
            * None
        
            * written by kfhe, at 10/24/2021
            * modified by kfhe, at 08/16/2024
        '''

        # Verbose?
        if verbose:
            print('Computing stress changes from fault {}'.format(fault.name))

        # Get a number
        nPatch = len(fault.patch)

        # Build Arrays
        xc = np.zeros((nPatch,))
        yc = np.zeros((nPatch,))
        zc = np.zeros((nPatch,))
        width = np.zeros((nPatch,))
        length = np.zeros((nPatch,)) 
        strike = np.zeros((nPatch,)) 
        dip = np.zeros((nPatch,))
        strikeslip = np.zeros((nPatch,))
        dipslip = np.zeros((nPatch,))
        tensileslip = np.zeros((nPatch,))

        # Build the arrays for okada
        for ii in range(len(fault.patch)):
            if verbose:
                sys.stdout.write('\r Patch {} / {}'.format(ii, len(fault.patch)))
                sys.stdout.flush()
            xc[ii], yc[ii], zc[ii], width[ii], length[ii], strike[ii], dip[ii] = fault.getpatchgeometry(fault.patch[ii], center=True)
            strikeslip[ii], dipslip[ii], tensileslip[ii] = fault.slip[ii,:]

        # Don't invert zc (for the patches, we give depths, so it has to be positive)
        # Apply the conversion factor
        strikeslip *= factor
        dipslip *= factor
        tensileslip *= factor

        # Set slips
        if 's' not in slipdirection:
            strikeslip[:] = 0.0
        if 'd' not in slipdirection:
            dipslip[:] = 0.0
        if 't' not in slipdirection:
            tensileslip[:] = 0.0

        # Get the stations
        if not stressonpatches:
            xs = self.x
            ys = self.y
            zs = -1.0*self.depth            # Okada wants the z of the observed stations in ENU coordinates, so we have to invert it.
        else:
            xs = xc
            ys = yc
            zs = -1.0*zc                    # Okada wants the z of the observed stations in ENU coordinates, so we have to invert it.

        # If force dip
        if force_dip is not None:
            dip[:] = force_dip

        # Get the Stress
        self.stresstype = 'total'
        if fault.patchType == 'rectangle':
            self.Stress, flag, flag2 = okada.stress(xs, ys, zs, # Observations in ENU coordinates system with z is positive upward
                                                    xc, yc, zc, # Source in END coordinates system with zc is positive downward
                                                    width, length, 
                                                    strike, dip,
                                                    strikeslip, dipslip, tensileslip, 
                                                    mu, nu, 
                                                    full=True)
        elif fault.patchType == 'triangle':
            from cutde.halfspace import strain_free, strain_to_stress
            obs_pts = np.vstack((xs, ys, zs)).T
            obs_pts = np.ascontiguousarray(obs_pts)
            tris = fault.Vertices[fault.Faces].copy()
            tris[:, :, -1] *= -1.0
            slips = np.vstack((strikeslip, dipslip, tensileslip)).T # The vertex assigned with counterclockwise order, so the dip slip is positive
            slips = np.ascontiguousarray(slips)
            strain = strain_free(obs_pts, tris, slips, nu)
            # (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)
            stress = strain_to_stress(strain, mu, nu)
            # Full
            Stress = np.zeros((3, 3, len(xs)))
            Stress[0,0,:] = stress[:,0]  # Sxx
            Stress[1,1,:] = stress[:,1]  # Syy
            Stress[2,2,:] = stress[:,2]  # Szz
            Stress[0,1,:] = stress[:,3]  # Sxy
            Stress[1,0,:] = stress[:,3]  # Sxy
            Stress[0,2,:] = stress[:,4]  # Sxz
            Stress[2,0,:] = stress[:,4]  # Sxz
            Stress[1,2,:] = stress[:,5]  # Syz
            Stress[2,1,:] = stress[:,5]  # Syz
            # self.Strain = strain
            self.Stress = Stress
            flag = np.array([True]*obs_pts.shape[0], dtype=np.bool_)
            flag2 = True
        self.flag = flag
        self.flag2 = flag2

        # All done
        return

    def computeCoulombStress(self, rake, strike=None, dip=None, cof=0.6, return_unit='MPa'):
        '''
        Computes the Coulomb Failure Stress

        Parameters:
            rake (float)        : The rake angle [Radian] of the coulomb failure stress
            strike (float)      : The strike angle [Radian] of the fault (optional)
            dip (float)         : The dip angle [Radian] of the fault (optional)
            cof (float)         : The coefficient of friction miu [0, 1.0)
            return_unit (str)   : The unit of the returned Coulomb stress ('MPa', 'bar', 'kPa', or 'Pa')

        Returns:
            coulomb (np.array)  : The values of Coulomb failure stress due to the fault in the specified unit

        Raises:
            ValueError: If an invalid return_unit is provided.

        Notes:
            - Added by kfhe, at 10/24/2021
            - Modified by kfhe, at 08/16/2024
        '''
        assert hasattr(self, 'Stress'), 'Must compute the Stress from fault at first'
        
        if not hasattr(self, 'TauStrike'):
            self.getTractions(strike, dip)
        
        b = np.cos(rake) * self.n2 + np.sin(rake) * self.n3
        Np = self.Stress.shape[2]
        coulomb = np.array([np.dot(b[:, i], self.T[i]) for i in range(Np)]) + cof * self.Sigma
        
        # Transfer the input unit of the Coulomb stress to lower case
        return_unit = return_unit.lower()
        
        # Transfer the Coulomb stress to the specified unit
        if return_unit == 'mpa':
            coulomb /= 1e6
        elif return_unit == 'bar':
            coulomb /= 1e5
        elif return_unit == 'kpa':
            coulomb /= 1e3
        elif return_unit != 'pa':
            raise ValueError("Invalid return_unit. Choose from 'MPa', 'bar', 'kPa', or 'Pa'.")
        
        return coulomb

    def computeCoulombStressFromTriangularSource(self, rake, cof=0.6, return_unit='MPa'):
        '''
        Computes the Coulomb Failure Stress from a triangular dislocation source.
    
        Parameters:
            rake (float)             : The rake angle [Radian] of the coulomb failure stress.
            cof (float)              : The coefficient of friction miu [0, 1.0).
            return_unit (str)        : The unit of the returned Coulomb stress ('MPa', 'kPa', 'Pa', or 'bar').
    
        Returns:
            coulomb (np.array)       : The values of Coulomb failure stress due to the fault in the specified unit.
    
        Raises:
            ValueError: If an invalid return_unit is provided.
    
        Notes:
            - Added by kfhe, at 10/24/2021
            - Modified by kfhe, at 08/16/2024
        '''
        # Compute the stress tractions using the triangular dislocation source
        assert hasattr(self, 'TauStrike'), 'Must compute the stress tractions from fault at first, using computeTriangularStressTraction()'
    
        # Compute the Coulomb Failure Stress
        b = np.cos(rake) * self.n2 + np.sin(rake) * self.n3
        Np = self.Sigma.shape[0]
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
    
    def computeTriangularStressTraction(self, source=None, receiver=None, strike=None, dip=None, 
                                        factor=0.001, mu=30e9, nu=0.25, slipdirection='sd', 
                                        force_dip=None, stressonpatches=False):
        '''
        Computes the stress tractions on a plane with a given strike and dip. Only for TriangularFault using cutde.

        Args:
            * strike            : Strike (radians). 
            * dip               : Dip (radians).

        If these are floats, all the tensors will be projected on that plane. Otherwise, they need to be the size ofthe number of tensors.

        Positive Normal Traction means extension. Positive Shear Traction means left-lateral. Postive Dip Traction means Reverse-thrusting.

        Returns:
            * n1, n2, n3, T, Sigma, TauStrike, TauDip
                where: 
                    n1: normal direction where positive is Open; 
                    n2: strike direction where positive is sinistral;
                    n3: dip direction where positive is reverse;
                    T: total traction tensor;
                    Sigma: normal traction tensor;
                    TauStrike: shear traction tensor along the strike direction;
                    TauDip: shear traction tensor along the dip direction.
        
        Notes: 
            * written by kfhe, at 10/24/2021
            * modified by kfhe, at 08/16/2024
        '''
        from cutde.halfspace import strain_matrix, strain_to_stress

        assert source.patchType == 'triangle', 'Only for TriangularFault using cutde!'

        if receiver is None:
            if stressonpatches:
                assert source is not None, 'Must provide source when receiver is None and stressonpatches is True!'
                xyzc = np.mean(source.Vertices[source.Faces, :], axis=1)
                obs_pts = np.vstack((xyzc[:, 0], xyzc[:, 1], -1.0*xyzc[:, 2])).T
            else:
                obs_pts = np.vstack((self.x, self.y, -1.0*self.depth)).T
                assert strike is not None and dip is not None, 'Must provide strike and dip when receiver is None and stressonpatches is False!'
        else:
            xyzc = np.mean(receiver.Vertices[receiver.Faces, :], axis=1)
            obs_pts = np.vstack((xyzc[:, 0], xyzc[:, 1], -1.0*xyzc[:, 2])).T
        # transfer obs_pts from F-order to C-order
        obs_pts = np.ascontiguousarray(obs_pts)

        # fault vertices are in ENU coordinates system with z is positive upward is needed in cutde
        tris = source.Vertices[source.Faces].copy()
        tris[:, :, -1] *= -1.0 # so we need to invert the z coordinate to make it positive upward

        # Compute the stress tensor Matrix (N_obs, 6, N_tri, 3)
        strain_mat = strain_matrix(obs_pts, tris, nu)
        N_obs = obs_pts.shape[0]
        N_tri = tris.shape[0]

        strain_mat_reshaped = strain_mat.transpose(0, 2, 3, 1)
        # Compute the stress tensor (N_obs, 6, N_tri, 3)
        stress_mat_reshaped = strain_to_stress(strain_mat_reshaped.reshape((-1, 6)), mu, nu)
        # Reshape the stress tensor to (N_obs, N_tri, 3, 6) and transpose the last two axes to get the stress tensor
        # 6 for the six components of the stress tensor: s_xx, s_yy, s_zz, s_xy, s_xz, s_yz
        # 6: s_xx, s_yy, s_zz, s_xy, s_xz, s_yz
        stress_mat = stress_mat_reshaped.reshape(strain_mat_reshaped.shape).transpose(0, 3, 1, 2)
        # Initialize the stress tensor (N_obs, 3, 3, N_tri, 3)
        stress_tensor = np.zeros((stress_mat.shape[0], 3, 3, stress_mat.shape[2], stress_mat.shape[3]))
        # Fill the diagonal elements
        stress_tensor[:, 0, 0, :, :] = stress_mat[:, 0, :, :]  # s_xx
        stress_tensor[:, 1, 1, :, :] = stress_mat[:, 1, :, :]  # s_yy
        stress_tensor[:, 2, 2, :, :] = stress_mat[:, 2, :, :]  # s_zz
        # Fill the off-diagonal elements
        stress_tensor[:, 0, 1, :, :] = stress_tensor[:, 1, 0, :, :] = stress_mat[:, 3, :, :]  # s_xy
        stress_tensor[:, 0, 2, :, :] = stress_tensor[:, 2, 0, :, :] = stress_mat[:, 4, :, :]  # s_xz
        stress_tensor[:, 1, 2, :, :] = stress_tensor[:, 2, 1, :, :] = stress_mat[:, 5, :, :]  # s_yz

        # Create the normal vectors
        if strike is not None and dip is not None:
            strike_rad = np.ones((N_obs,))*strike
            dip_rad = np.ones((N_obs,))*dip
        else:
            strike_rad, dip_rad = receiver.getStrikes(), receiver.getDips()

        # If force dip
        if force_dip is not None:
            dip_rad[:] = np.ones((N_obs,))*force_dip

        # n2: strike direction, postive is sinistral, n3: dip direction, postive is reverse;
        # shape: (3, N_obs)
        n1, n2, n3 = self.strikedip2normal(strike_rad, dip_rad)

        # Transfer the stress tensor to the receiver
        # Compute the stress vectors, shape: (N_obs, 3, N_tri, 3)
        T = np.einsum('ik,iklmn->ilmn', n1.T, stress_tensor)
        # Compute the Shear Stress and Normal Stress, shape: (N_obs, N_tri, 3)
        Sigma = np.einsum('ijlm,ij->ilm', T, n1.T)

        # Modify the calculation of the shear stress TauStrike along the strike direction
        TauStrike = np.einsum('ijlm, ij->ilm', T, n2.T)

        # Modify the calculation of the shear stress TauDip along the dip direction
        TauDip = np.einsum('ijlm, ij->ilm', T, n3.T)

        # Slip direction
        strikeslip = np.zeros((N_tri,))
        dipslip = np.zeros((N_tri,))
        tensileslip = np.zeros((N_tri,))
        if 's' in slipdirection: strikeslip = source.slip[:,0]*factor
        # reverse slip is positive in cutde, consistent with that in Okada and csi
        if 'd' in slipdirection: dipslip = source.slip[:,1]*factor
        if 't' in slipdirection: tensileslip = source.slip[:,2]*factor
        slip = np.vstack((strikeslip, dipslip, tensileslip)).T
        T = np.einsum('iklm, lm-> ik', T, slip)
        Sigma = np.einsum('ijk, jk-> i', Sigma, slip)
        TauStrike = np.einsum('ijk, jk-> i', TauStrike, slip)
        TauDip = np.einsum('ijk, jk-> i', TauDip, slip)

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.T = T
        self.Sigma = Sigma
        self.TauStrike = TauStrike
        self.TauDip = TauDip
        
        return n1, n2, n3, T, Sigma, TauStrike, TauDip
    
    def compute_coulomb_stress_on_receiver(self, sources, receiver, strike_angle=None, dip_angle=None, rake=0, cof=0.6, return_unit='MPa'):
        """
        Compute the Coulomb stress on a receiver fault from multiple sources.

        Parameters:
        sources (list): List of TriangularPatches sources.
        receiver (TriangularPatches): Receiver fault.
        strike_angle (float): Strike angle in degrees. Default is None.
        dip_angle (float): Dip angle in degrees. Default is None.
        rake (float): Rake angle in degrees. Default is 0.
        cof (float): Coefficient of friction. Default is 0.6.
        return_unit (str): Unit of the returned stress. Default is 'MPa'.

        Returns:
        np.ndarray: Computed Coulomb stress on the receiver fault.
        """
        if strike_angle is not None and dip_angle is not None:
            npatch = len(receiver.patch)
            strike = np.ones(npatch) * strike_angle / 180.0 * np.pi
            dip = np.ones(npatch) * dip_angle / 180.0 * np.pi
        else:
            strike = None
            dip = None
        coulomb_stress = 0
        rake_rad = rake / 180.0 * np.pi
        for source in sources:
            self.computeTriangularStressTraction(source, receiver=receiver, strike=strike, dip=dip)
            coulomb_stress += self.computeCoulombStressFromTriangularSource(rake=rake_rad, cof=cof, return_unit=return_unit)
        return coulomb_stress
    
    def compute_coulomb_stress_field(self, sources, lon_range, lat_range, grid_size_lon, grid_size_lat, 
                                     strike_angle, dip_angle, rake=0, cof=0.6, return_unit='MPa', 
                                     depth=10.0, plot=True, cmap='cmc.roma_r', vmin=-0.5, vmax=0.5, 
                                     add_faults=None, savefig=True, figname='coulomb_stress_field.png', dpi=600):
        """
        Compute and optionally plot the Coulomb stress field over a specified grid.
    
        Parameters:
        sources (list): List of TriangularPatches sources.
        lon_range (tuple): Longitude range as (min_lon, max_lon).
        lat_range (tuple): Latitude range as (min_lat, max_lat).
        grid_size_lon (int): Number of points in the grid for longitude.
        grid_size_lat (int): Number of points in the grid for latitude.
        strike_angle (float): Strike angle in degrees.
        dip_angle (float): Dip angle in degrees.
        rake (float): Rake angle in degrees. Default is 0.
        cof (float): Coefficient of friction. Default is 0.6.
        return_unit (str): Unit of the returned stress. Default is 'MPa'.
        depth (float): Depth at which to compute the stress field in km. Default is 10.0.
        plot (bool): If True, plot the Coulomb stress field. Default is True.
        cmap (str): Colormap to use for the plot. Default is 'cmc.roma_r'.
        vmin (float): Minimum value for the colormap. Default is -0.5.
        vmax (float): Maximum value for the colormap. Default is 0.5.
        add_faults (list): List of faults to add to the plot. Default is None.
        savefig (bool): If True, saves the figure. Default is True.
        figname (str): Name of the saved figure file. Default is 'coulomb_stress_field.png'.
        dpi (int): Dots per inch for the saved figure. Default is 600.
    
        Returns:
        np.ndarray: Computed Coulomb stress field.
        """
        lon = np.linspace(lon_range[0], lon_range[1], grid_size_lon)
        lat = np.linspace(lat_range[0], lat_range[1], grid_size_lat)
        lon, lat = np.meshgrid(lon, lat)
        lon = lon.flatten()
        lat = lat.flatten()
        strike = np.ones_like(lon) * strike_angle / 180.0 * np.pi
        dip = np.ones_like(lon) * dip_angle / 180.0 * np.pi
        self.setLonLatZ(lon, lat, np.ones_like(lon) * depth)
        coulomb_stress = 0
        rake_rad = rake / 180.0 * np.pi
        for source in sources:
            self.computeTriangularStressTraction(source, strike=strike, dip=dip)
            coulomb_stress += self.computeCoulombStressFromTriangularSource(rake=rake_rad, cof=cof, return_unit=return_unit)
        coulomb_stress = coulomb_stress.reshape((grid_size_lat, grid_size_lon))
    
        if plot:
            from ..plottools import sci_plot_style
            import cmcrameri
            import matplotlib.pyplot as plt
    
            with sci_plot_style():
                lon = np.linspace(lon_range[0], lon_range[1], grid_size_lon)
                lat = np.linspace(lat_range[0], lat_range[1], grid_size_lat)
                lon, lat = np.meshgrid(lon, lat)
                
                plt.imshow(coulomb_stress, extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
                for source in sources:
                    plt.plot(source.lon, source.lat, 'k')
                if add_faults is not None:
                    for fault in add_faults:
                        plt.plot(fault.lon, fault.lat, 'k')
                plt.colorbar(label=f'$\Delta CFS$ ({return_unit})')
                if savefig:
                    plt.savefig(figname, dpi=dpi)
                plt.show()
    
        return coulomb_stress
    
