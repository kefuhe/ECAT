'''
Added by kfhe at 5/5/2024
Object :
    * Perturbing fault for bayesian inversion
    * The perturbation methods are defined in the class BayesianAdaptiveTriangularPatches
'''

import numpy as np
import inspect
from scipy.spatial.distance import cdist
import functools
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz

# import self-defined classes
from .AdaptiveTriangularPatches import (
    AdaptiveTriangularPatches,
    transform_point,
    reverse_transform_point
)
from .make_mesh_dutta import makemesh as make_mesh_dutta
from .MeshGenerator import MeshGenerator
from .bayesian_perturbation_base import track_mesh_update, BayesianTriFaultBase
from .geom_ops import discretize_coords


class BayesianAdaptiveTriangularPatches(BayesianTriFaultBase):
    def __init__(self, name: str, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True, shared_info=None, use_shared_info=False, is_active=False):
        BayesianTriFaultBase.__init__(self, name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=verbose, 
                                      shared_info=shared_info, use_shared_info=use_shared_info, is_active=is_active)
        # self.mesh_generator = MeshGenerator()

    def is_mesh_updated(self):
        return self.mesh_updated
    
    def is_laplacian_updated(self):
        return self.laplacian_updated

    def perturb(self, method, **kwargs):
        """
        Perturb the geometry of the patches based on the given parameters.

        Parameters:
        method: The perturbation method to use. Must be a method of this class that starts with 'perturb_'.
        kwargs: A dictionary of arguments to pass to the perturbation method. Must include a 'perturbations' key.
        """
        if not method.startswith('perturb_'):
            raise ValueError("The method must start with 'perturb_'")

        if 'perturbations' not in kwargs:
            raise ValueError("The 'perturbations' argument is required")

        perturb_method = self.perturbation_methods.get(method, None)
        if perturb_method is None:
            available_methods = ', '.join(self.perturbation_methods.keys())
            raise ValueError(f"The method '{method}' does not exist. Available methods are: {available_methods}")

        return perturb_method(self, **kwargs)
    
    #--------------------------------------Perturbing Dip--------------------------------------#
    def perturb_dips(self, x_coords, y_coords, dips, 
                     perturbations, fixed_nodes=None, angle_unit='degrees', 
                     discretization_interval=None, interpolation_axis='x',
                     is_utm=False, buffer_nodes=None, buffer_radius=None, update_xydip_ref=False,
                     use_average_strike=False, average_strike_source='pca', user_direction_angle=None):
        """
        Apply random perturbations to dips.
    
        Parameters:
        x_coords (np.ndarray): x coordinates.
        y_coords (np.ndarray): y coordinates.
        dips (np.ndarray): Original dips.
        perturbations (np.ndarray): Direct perturbation values. If provided, the function will use these values directly.
        fixed_nodes (list, optional): A list of indices of fixed nodes. If provided, the function will not perturb these nodes.
        angle_unit (str, optional): The unit of the input angles, can be 'radians' or 'degrees'. Default is 'degrees'.
        discretization_interval (float, optional): The interval for discretization of the fault trace. Default is None.
        interpolation_axis (str, optional): The axis for interpolation. Default is 'x'.
        is_utm (bool, optional): Whether the coordinates are in UTM. Default is False.
        buffer_nodes (np.ndarray, optional): Coordinates of buffer nodes. If provided, the function will interpolate these nodes after perturbing the dips.
        buffer_radius (float, optional): Radius of the buffer. If provided, the function will interpolate these nodes after perturbing the dips.
        update_xydip_ref (bool, optional): Whether to update the reference x, y, and dip values. Default is False.
        use_average_strike (bool, optional): Whether to use the average strike direction. Default is False.
        average_strike_source (str, optional): The source of the average strike direction. Default is 'pca'.
        user_direction_angle (float, optional): User-specified direction angle. Default is None.
    
        Returns:
        np.ndarray: Perturbed coordinates.
        """
        if dips is None:
            raise ValueError("dips parameter is required")
    
        perturb_dips = dips.copy()
    
        if fixed_nodes is None:
            fixed_nodes = []
    
        movable_nodes = [i for i in range(len(perturb_dips)) if i not in fixed_nodes]
    
        # Add perturbation values to the dips
        perturb_dips[movable_nodes] += perturbations
    
        # Call interpolate_top_dip_from_relocated_profile and generate_bottom_from_segmented_relocated_dips methods
        xydip = np.c_[x_coords, y_coords, perturb_dips]
        self.interpolate_top_dip_from_relocated_profile(xydip, is_utm=is_utm, 
                                                        discretization_interval=discretization_interval, interpolation_axis=interpolation_axis, 
                                                        buffer_nodes=buffer_nodes, buffer_radius=buffer_radius,
                                                        update_xydip_ref=update_xydip_ref)
        perturb_coords = self.generate_bottom_from_segmented_relocated_dips(fault_depth=self.depth, update_self=True,
                                                                            use_average_strike=use_average_strike, interpolation_axis=interpolation_axis, 
                                                                            average_strike_source=average_strike_source,
                                                                            user_direction_angle=user_direction_angle)
    
        return perturb_coords
    
    @track_mesh_update()
    def perturb_dips_with_preset_params(self, perturbations, discretization_interval=None, interpolation_axis='x', 
                                        fixed_nodes=None, angle_unit='degrees', is_utm=False, 
                                        buffer_nodes=None, buffer_radius=None, update_xydip_ref=False,
                                        use_average_strike=False, average_strike_source='pca', user_direction_angle=None):
        """
        Apply random perturbations to dips using preset x_dip_ref, y_dip_ref, and dip_ref parameters.
    
        Parameters:
        perturbations (np.ndarray): Direct perturbation values. If provided, the function will use these values directly.
        fixed_nodes (list, optional): A list of indices of fixed nodes. If provided, the function will not perturb these nodes.
        angle_unit (str, optional): The unit of the input angles, can be 'radians' or 'degrees'. Default is 'degrees'.
        discretization_interval (float, optional): The interval for discretization of the fault trace. Default is None.
        interpolation_axis (str, optional): The axis for interpolation. Default is 'x'.
        is_utm (bool, optional): Whether the coordinates are in UTM. Default is False.
        buffer_nodes (np.ndarray, optional): Coordinates of buffer nodes. If provided, the function will interpolate these nodes after perturbing the dips.
        buffer_radius (float, optional): Radius of the buffer. If provided, the function will interpolate these nodes after perturbing the dips.
        update_xydip_ref (bool, optional): Whether to update the reference x, y, and dip values. Default is False.
        use_average_strike (bool, optional): Whether to use the average strike direction. Default is False.
        average_strike_source (str, optional): The source of the average strike direction. Default is 'pca'.
        user_direction_angle (float, optional): User-specified direction angle. Default is None.
    
        Returns:
        np.ndarray: Perturbed dips.
        """
        if self.x_dip_ref is None or self.y_dip_ref is None or self.dip_ref is None:
            raise ValueError("x_dip_ref, y_dip_ref, and dip_ref must be set before calling this method")
    
        return self.perturb_dips(
            self.x_dip_ref, 
            self.y_dip_ref, 
            self.dip_ref, 
            perturbations, 
            fixed_nodes, 
            angle_unit, 
            discretization_interval,
            interpolation_axis,
            is_utm,
            buffer_nodes,
            buffer_radius,
            update_xydip_ref,
            use_average_strike,
            average_strike_source,
            user_direction_angle
        )
    
    @track_mesh_update(update_mesh=True)
    def perturb_DipsPresetParams_SimpleMesh(self, perturbations, discretization_interval=None, interpolation_axis='x',
                                            fixed_nodes=None, angle_unit='degrees', is_utm=False, 
                                            buffer_nodes=None, buffer_radius=None, update_xydip_ref=False, 
                                            disct_z=None, bias=None, min_dz=None,
                                            use_average_strike=False, average_strike_source='pca', user_direction_angle=None):
        """
        Apply random perturbations to dips using preset x_dip_ref, y_dip_ref, and dip_ref parameters, and update the simple mesh.
    
        Parameters:
        perturbations (np.ndarray): Direct perturbation values. If provided, the function will use these values directly.
        fixed_nodes (list, optional): A list of indices of fixed nodes. If provided, the function will not perturb these nodes.
        angle_unit (str, optional): The unit of the input angles, can be 'radians' or 'degrees'. Default is 'degrees'.
        discretization_interval (float, optional): The interval for discretization of the fault trace. Default is None.
        interpolation_axis (str, optional): The axis for interpolation. Default is 'x'.
        is_utm (bool, optional): Whether the coordinates are in UTM. Default is False.
        buffer_nodes (np.ndarray, optional): Coordinates of buffer nodes. If provided, the function will interpolate these nodes after perturbing the dips.
        buffer_radius (float, optional): Radius of the buffer. If provided, the function will interpolate these nodes after perturbing the dips.
        update_xydip_ref (bool, optional): Whether to update the reference x, y, and dip values. Default is False.
        disct_z (float, optional): Mesh density in the z direction. Default is None.
        bias (float, optional): Mesh bias in the z direction. Default is None.
        min_dz (float, optional): Minimum mesh spacing in the z direction. Default is None.
        use_average_strike (bool, optional): Whether to use the average strike direction. Default is False.
        average_strike_source (str, optional): The source of the average strike direction. Default is 'pca'.
        user_direction_angle (float, optional): User-specified direction angle. Default is None.
    
        Returns:
        None
        """
        self.perturb_dips_with_preset_params(perturbations, 
                                             discretization_interval, 
                                             interpolation_axis, 
                                             fixed_nodes, 
                                             angle_unit, 
                                             is_utm, 
                                             buffer_nodes, 
                                             buffer_radius, 
                                             update_xydip_ref,
                                             use_average_strike,
                                             average_strike_source,
                                             user_direction_angle
                                             )
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z, bias, min_dz)
        return                      
    
    def set_xy_dip_ref(self, x_dip_ref, y_dip_ref, dip_ref):
        """
        Set the reference x, y, and dip values for perturbation.
    
        Parameters:
        x_dip_ref (np.ndarray): x coordinates for the dip reference. unit: degree
        y_dip_ref (np.ndarray): y coordinates for the dip reference. unit: degree
        dip_ref (np.ndarray): dip reference values. unit: degree
    
        Returns:
        None
        """
        self.x_dip_ref = x_dip_ref
        self.y_dip_ref = y_dip_ref
        self.dip_ref = dip_ref
        return
    
    def set_xy_dip_ref_from_coords(self, coords, dips, is_utm=False):
        """
        Set the reference x, y, and dip values for perturbation from the given coordinates and dips.
    
        Parameters:
        coords (np.ndarray): Array of coordinate points.
        dips (np.ndarray): Array of dip values.
        is_utm (bool, optional): Whether the coordinates are in UTM. Default is False.
    
        Returns:
        None
        """
        if is_utm:
            x, y = coords[:, 0], coords[:, 1]
            x_dip_ref, y_dip_ref = self.xy2ll(x, y)
        else:
            x_dip_ref, y_dip_ref = coords[:, 0], coords[:, 1]
        self.set_xy_dip_ref(x_dip_ref, y_dip_ref, dips)
        return
    
    def set_xy_dip_ref_from_file(self, filename, header=0, is_utm=False):
        """
        Set the reference x, y, and dip values for perturbation from a file.
        lon lat dip or x y dip

        Parameters:
        filename (str): Path to the file.
        is_utm (bool, optional): Whether the coordinates are in UTM. Default is False.
    
        Returns:
        None
        """
        data = np.loadtxt(filename, skiprows=header)
        coords = data[:, :2]
        dips = data[:, 2]
        self.set_xy_dip_ref_from_coords(coords, dips, is_utm)
        return
    #-------------------------------------------------------------------------------------------------------------------#

    #--------------------------------------Perturbing Coords along fixed direction--------------------------------------#
    def calculate_perturb_direction(self, coords, angle_unit='degrees', use_average_strike=False, average_direction=None):
        """
        Calculate the perturbation direction for the given coordinates.
    
        Parameters:
        coords (np.ndarray): Original coordinate points.
        angle_unit (str, optional): The unit of the input angles, can be 'radians' or 'degrees'. Default is 'degrees'.
        use_average_strike (bool, optional): Whether to use the average strike direction for all nodes. Default is False.
        average_direction (float, optional): The average direction (azimuth) in radians or degrees. If provided, this value will be used directly.
    
        Returns:
        np.ndarray: Array of azimuth directions for perturbation.
        """
        if average_direction is not None:
            if angle_unit == 'degrees':
                # Convert degrees to radians
                average_direction = np.radians(average_direction)
    
        if average_direction is None:
            # Calculate the strike direction between adjacent points in coords
            trends = np.arctan2(np.diff(coords[:, 1]), np.diff(coords[:, 0]))
            trends = np.concatenate(([trends[0]], (trends[:-1] + trends[1:]) / 2, [trends[-1]]))
            trends_azi = np.pi - trends
            if use_average_strike:
                trends_azi = np.full(len(coords), np.mean(trends_azi))
        else:
            # Use the provided average direction for all nodes
            trends_azi = np.full(len(coords), average_direction)
        
        return trends_azi
    
    def perturb_coords_along_fixed_direction(self, coords, perturbations, average_direction=None, 
                                             fixed_nodes=None, angle_unit='degrees', perturbation_direction='horizontal',
                                             use_average_strike=False):
        """
        Apply random perturbations to the coordinates along a fixed direction.
    
        Parameters:
        coords (np.ndarray): Original coordinate points.
        perturbations (np.ndarray): Direct perturbation values. If provided, the function will use these values directly.
        average_direction (float, optional): The average direction (azimuth) in radians or degrees. If provided, this value will be used directly.
                                             If None, the direction will be calculated based on adjacent points in coords.
        fixed_nodes (list, optional): A list of indices of fixed nodes. If provided, the function will not perturb these nodes.
        angle_unit (str, optional): The unit of the input angles, can be 'radians' or 'degrees'. Default is 'degrees'.
        perturbation_direction (str, optional): The direction of perturbation, can be 'horizontal' or 'vertical'. Default is 'horizontal'.
        use_average_strike (bool, optional): Whether to use the average strike direction for all nodes. Default is False.
    
        Returns:
        np.ndarray: Perturbed coordinates.
        """
        coords = coords.copy()
    
        if fixed_nodes is None:
            fixed_nodes = []
    
        movable_nodes = [i for i in range(len(coords)) if i not in fixed_nodes]
    
        # Calculate the perturbation direction
        trends_azi = self.calculate_perturb_direction(coords, angle_unit, use_average_strike, average_direction)
    
        # Convert perturbations to np.array
        perturbations = np.array(perturbations)
        
        if perturbation_direction == 'horizontal':
            # Apply perturbations to the coordinates in the horizontal direction
            # Calculate direction vectors
            trends = np.pi / 2.0 - trends_azi
            directions = np.array([np.cos(trends), np.sin(trends)]).T
    
            coords[movable_nodes, :2] += directions[movable_nodes] * perturbations[:, None]
        elif perturbation_direction == 'vertical':
            # Apply perturbations to the z-coordinate
            coords[movable_nodes, 2] += perturbations
    
        return coords

    @track_mesh_update()
    def perturb_top_coords_along_fixed_direction(self, perturbations, average_direction=None, 
                                         fixed_nodes=None, angle_unit='degrees', perturbation_direction='horizontal',
                                     use_average_strike=False):
        assert hasattr(self, 'top_coords_ref'), 'You need to set top_coords_ref before perturbing the coordinates.'
        self.top_coords = self.perturb_coords_along_fixed_direction(self.top_coords_ref, perturbations, average_direction, 
                                                            fixed_nodes, angle_unit, perturbation_direction, use_average_strike)
        self.set_coords(self.top_coords, lonlat=False, coord_type='top')
        return self.top_coords

    @track_mesh_update()
    def perturb_bottom_coords_along_fixed_direction(self, perturbations, average_direction=None, 
                                            fixed_nodes=None, angle_unit='degrees', perturbation_direction='horizontal',
                                     use_average_strike=False):
        assert hasattr(self, 'bottom_coords_ref'), 'You need to set bottom_coords_ref before perturbing the coordinates.'
        self.bottom_coords = self.perturb_coords_along_fixed_direction(self.bottom_coords_ref, perturbations, average_direction, 
                                                               fixed_nodes, angle_unit, perturbation_direction, use_average_strike)
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True)
    def perturb_BottomFixedDir_simpleMesh(self, perturbations, average_direction=None,
                                          fixed_nodes=None, angle_unit='degrees', 
                                          perturbation_direction='horizontal', 
                                          disct_z=None, bias=None, min_dz=None,
                                          use_average_strike=False):
        """
        Apply random perturbations to the bottom coordinates and update the simple mesh.
    
        Parameters:
        - perturbations (array-like): Direct perturbation values. If provided, the function will use these values directly.
        - average_direction (float, optional): Average movement direction (azimuth) in radians or degrees, ranging from 0 to 2*pi or 0 to 360.
                                               If None, the direction is calculated based on the strike direction between adjacent points in coords.
        - fixed_nodes (list, optional): A list of indices of fixed nodes. If provided, these nodes will not be perturbed.
        - angle_unit (str, optional): Unit of the input angle, can be 'radians' or 'degrees'. Default is 'degrees'.
        - perturbation_direction (str, optional): Direction of perturbation, can be 'horizontal' or 'vertical'. Default is 'horizontal'.
        - disct_z (int, optional): Grid density in the z-direction. Default is None.
        - bias (float, optional): Grid bias in the z-direction. Default is None.
        - min_dz (float, optional): Minimum grid spacing in the z-direction. Default is None.
        - use_average_strike (bool, optional): Whether to use the average strike direction. Default is False.
    
        Returns:
        - bottom_coords (array-like): Perturbed bottom coordinates.
        """
        self.perturb_bottom_coords_along_fixed_direction(perturbations, average_direction, 
                                                         fixed_nodes, angle_unit, perturbation_direction,
                                                         use_average_strike=use_average_strike)
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z, bias, min_dz)
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True)
    def perturb_BottomFixedDir_DeformMesh(self, perturbations, 
                                             top_size=2.0, bottom_size=4.0, num_segments=30, 
                                             disct_z=10, projection=None, rotation_angle=None, 
                                             show=False, verbose=0, remap=False, 
                                             average_direction=None, fixed_nodes=None, 
                                             angle_unit='degrees', perturbation_direction='horizontal', 
                                             bias=None, min_dz=None, use_average_strike=False):
        """
        Perturb bottom coordinates and generate mesh.
    
        Parameters:
        - top_coords (np.ndarray): Top coordinates for the grid.
        - bottom_coords (np.ndarray): Bottom coordinates for the grid.
        - perturbations (array-like): Direct perturbation values. If provided, the function will use these values directly.
        - top_size (float): Size of the top mesh. Default is 2.0.
        - bottom_size (float): Size of the bottom mesh. Default is 4.0.
        - num_segments (int): Number of segments for discretization. Default is 30.
        - disct_z (int): Number of divisions in the z-direction. Default is 10.
        - projection (str, optional): Projection plane to use ('xy', 'xz', 'yz'). Default is None.
        - rotation_angle (float, optional): Angle to rotate the grid and vertices. Default is None.
        - show (bool): Whether to show the mesh. Default is False.
        - verbose (int): Verbosity level. Default is 0.
        - remap (bool): Whether to remap Gmsh vertices to grid. Default is False.
        - average_direction (float, optional): Average movement direction (azimuth) in radians or degrees, ranging from 0 to 2*pi or 0 to 360.
                                               If None, the direction is calculated based on the strike direction between adjacent points in coords.
        - fixed_nodes (list, optional): A list of indices of fixed nodes. If provided, these nodes will not be perturbed.
        - angle_unit (str, optional): Unit of the input angle, can be 'radians' or 'degrees'. Default is 'degrees'.
        - perturbation_direction (str, optional): Direction of perturbation, can be 'horizontal' or 'vertical'. Default is 'horizontal'.
        - bias (float, optional): Grid bias in the z-direction. Default is None.
        - min_dz (float, optional): Minimum grid spacing in the z-direction. Default is None.
        - use_average_strike (bool, optional): Whether to use the average strike direction. Default is False.
    
        Returns:
        - new_gmsh_verts (np.ndarray): Array of new Gmsh vertices.
        - gmsh_faces (np.ndarray): Array of Gmsh faces.
        """
        # Apply perturbations to bottom coordinates
        self.perturb_bottom_coords_along_fixed_direction(perturbations, average_direction, 
                                                         fixed_nodes, angle_unit, perturbation_direction,
                                                         use_average_strike=use_average_strike)
        
        # Generate and deform mesh
        self.generate_and_deform_mesh(self.top_coords, self.bottom_coords, 
                                    top_size=top_size, bottom_size=bottom_size, 
                                    num_segments=num_segments, disct_z=disct_z, 
                                    projection=projection, rotation_angle=rotation_angle, 
                                    show=show, verbose=verbose, remap=remap, 
                                    bias=bias, min_dz=min_dz)
        return self.bottom_coords

    @track_mesh_update()
    def perturb_layer_coords_along_fixed_direction(self, perturbations, mid_layer_index=0, average_direction=None, 
                                           fixed_nodes=None, angle_unit='degrees', 
                                           perturbation_direction='horizontal',
                                           use_average_strike=False):
        assert hasattr(self, 'layers_ref') and mid_layer_index < len(self.layers_ref), 'You need to set layers_ref before perturbing the coordinates.'
        self.layers[mid_layer_index] = self.perturb_coords_along_fixed_direction(self.layers_ref[mid_layer_index], perturbations, average_direction, 
                                                                         fixed_nodes, angle_unit, perturbation_direction, use_average_strike)
        x, y, z = self.layers[mid_layer_index][:, 0], self.layers[mid_layer_index][:, 1], self.layers[mid_layer_index][:, 2]
        lon, lat = self.xy2ll(x, y)
        self.layers_ll[mid_layer_index] = np.vstack((lon, lat, z)).T
        return self.layers[mid_layer_index]
    
    @track_mesh_update(update_mesh=True, update_laplacian=True, update_area=True, expected_perturbations_count=1)
    def perturb_geometry_along_fixed_direction(self, perturbations, average_direction=None, 
                                               fixed_nodes=None, angle_unit='degrees', perturbation_direction='horizontal',
                                               use_average_strike=False):
        """
        Apply random perturbations to the coordinates along a fixed direction.
    
        Parameters:
        perturbations (np.ndarray): Direct perturbation values. If provided, the function will use these values directly.
        average_direction (float, optional): The average direction (azimuth) in radians or degrees. If provided, this value will be used directly.
                                             If None, the direction will be calculated based on adjacent points in coords.
        fixed_nodes (list, optional): A list of indices of fixed nodes. If provided, the function will not perturb these nodes.
        angle_unit (str, optional): The unit of the input angles, can be 'radians' or 'degrees'. Default is 'degrees'.
        perturbation_direction (str, optional): The direction of perturbation, can be 'horizontal' or 'vertical'. Default is 'horizontal'.
        use_average_strike (bool, optional): Whether to use the average strike direction for all nodes. Default is False.
    
        Returns:
        None
        """
        # Translate the fault geometry mesh along the fixed direction
        if not hasattr(self, 'Vertices_ref'):
            self.Vertices_ref = self.Vertices.copy()
        vertices_trans = self.perturb_coords_along_fixed_direction(self.Vertices_ref, perturbations, average_direction, 
                                                                   fixed_nodes, angle_unit, perturbation_direction, use_average_strike)
        self.VertFace2csifault(vertices_trans, self.Faces)
    
        return # top_coords, bottom_coords, layers
    #---------------------------------------------------------------------------------------------------------#

    #--------------------------------------Perturbing Coords by Rotation--------------------------------------#
    def calculate_pivot(self, pivot, coords, is_utm=True, force_pivot_in_coords=True):
        """
        Calculate the pivot point for rotation.
    
        Parameters:
        pivot (str or array-like): The coordinates of the pivot point. If a string, it can be 'start' (the first point of the coordinates), 'end' (the last point of the coordinates), or 'midpoint' (the midpoint of the coordinates). If an array, it should be an array of length 2 representing the x and y coordinates of the pivot point.
        coords (array-like): An array of coordinate points, where each point should be an array of length 2/3 representing the x and y coordinates [and z coordinate].
        is_utm (bool, optional): If True, the pivot is in latitude and longitude coordinates and needs to be converted to UTM coordinates. Default is False.
        force_pivot_in_coords (bool, optional): If True, the pivot point will be forced to be one of the points in coords. Default is False.
    
        Returns:
        array-like: The coordinates of the pivot point, which is an array of length 2 representing the x and y coordinates.
    
        Raises:
        ValueError: If pivot is a string but not 'start', 'end', or 'midpoint'.
        """
        if isinstance(pivot, str):
            if pivot == 'start':
                pivot = coords[0, :2]
            elif pivot == 'end':
                pivot = coords[-1, :2]
            elif pivot == 'midpoint':
                pivot = np.mean(coords[:, :2], axis=0)
                if force_pivot_in_coords:
                    tree = cKDTree(coords[:, :2])
                    _, idx = tree.query([pivot], k=1)
                    pivot = coords[idx[0], :2]
            else:
                raise ValueError("Invalid value for pivot. It should be 'start', 'end' or 'midpoint'.")
        else:
            if not is_utm:
                px, py = self.ll2xy(pivot[0], pivot[1])
                pivot = np.array([px, py])
            if force_pivot_in_coords:
                tree = cKDTree(coords[:, :2])
                _, idx = tree.query([pivot], k=1)
                pivot = coords[idx[0], :2]
        self.pivot = pivot
        return pivot
    
    def perturb_coords_by_rotation(self, coords, perturbations, pivot='midpoint', recalculate_pivot=False,
                                   is_utm=False, angle_unit='degrees', force_pivot_in_coords=False):
        """
        Apply perturbations to the given coordinates by rotation.
    
        Parameters:
        coords (array-like): Original coordinate points.
        perturbations (float): The angle of perturbation. Positive for counterclockwise, negative for clockwise.
        pivot (str or array-like, optional): The center point of the perturbation. Can be 'start', 'end', 'midpoint', or a coordinate point (e.g., `(x, y)`). Default is 'midpoint'.
        recalculate_pivot (bool, optional): Whether to recalculate the pivot point. If True, the pivot point will be recalculated regardless of whether self.pivot exists. Default is False.
        is_utm (bool, optional): If pivot is a coordinate point, this parameter determines whether the coordinate point is in UTM coordinates. Default is False.
        angle_unit (str, optional): The unit of the angle. Can be 'degrees' or 'radians'. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot to be one of the points in coords. If True, the pivot will be the closest point in coords. Default is False.
    
        Returns:
        array-like: The perturbed coordinates.
        """
        if recalculate_pivot or not hasattr(self, 'pivot'):
            # Determine the center point of the perturbation
            pivot = self.calculate_pivot(pivot, coords, is_utm, force_pivot_in_coords)
        else:
            pivot = self.pivot
    
        # Convert the perturbation angle from degrees to radians
        if angle_unit == 'degrees':
            perturbations = np.radians(perturbations).item()
    
        # Calculate the coordinates relative to the pivot point
        coords = coords.copy()
        relative_coords = coords[:, :2] - pivot
    
        # Use complex numbers and Euler's formula for rotation
        complex_coords = relative_coords[:, 0] + 1j * relative_coords[:, 1]
        complex_rotation = np.exp(1j * perturbations)
        perturbed_coords = complex_coords * complex_rotation
    
        # Calculate the absolute coordinates after perturbation
        coords[:, :2] = np.c_[perturbed_coords.real, perturbed_coords.imag] + pivot
    
        return coords

    @track_mesh_update()
    def perturb_top_coords_by_rotation(self, perturbations, pivot='midpoint', is_utm=False, recalculate_pivot=False,
                                        angle_unit='degrees', force_pivot_in_coords=False):
        assert hasattr(self, 'top_coords_ref'), 'You need to set top_coords_ref before perturbing the coordinates.'
        self.top_coords = self.perturb_coords_by_rotation(self.top_coords_ref, perturbations, pivot, recalculate_pivot,
                                                           is_utm, angle_unit, force_pivot_in_coords)
        self.set_coords(self.top_coords, lonlat=False, coord_type='top')
        return self.top_coords

    @track_mesh_update()
    def perturb_bottom_coords_by_rotation(self, perturbations, pivot='midpoint', is_utm=False, recalculate_pivot=False,
                                           angle_unit='degrees', force_pivot_in_coords=False):
        assert hasattr(self, 'bottom_coords_ref'), 'You need to set bottom_coords_ref before perturbing the coordinates.'
        self.bottom_coords = self.perturb_coords_by_rotation(self.bottom_coords_ref, perturbations, pivot, recalculate_pivot,
                                                              is_utm, angle_unit, force_pivot_in_coords)
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True)
    def perturb_BottomRotation_simpleMesh(self, perturbations, pivot='midpoint', is_utm=False, recalculate_pivot=False,
                                          angle_unit='degrees', force_pivot_in_coords=False,
                                          disct_z=None, bias=None, min_dz=None):
        """
        Apply perturbations to the bottom coordinates and update the simple mesh.
    
        Parameters:
        perturbations (float): The angle of perturbation. Positive for counterclockwise, negative for clockwise.
        pivot (str or tuple, optional): The center point of the perturbation. Can be 'start', 'end', 'midpoint', or a coordinate point (e.g., `(x, y)`). Default is 'midpoint'.
        is_utm (bool, optional): If pivot is a coordinate point, this parameter determines whether the coordinate point is in UTM coordinates. Default is False.
        recalculate_pivot (bool, optional): Whether to recalculate the pivot point. If True, the pivot point will be recalculated regardless of whether self.pivot exists. Default is False.
        angle_unit (str, optional): The unit of the angle. Can be 'degrees' or 'radians'. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot to be one of the points in coords. If True, the pivot will be the closest point in coords. Default is False.
        disct_z (float, optional): Mesh density in the z direction. Default is None.
        bias (float, optional): Mesh bias in the z direction. Default is None.
        min_dz (float, optional): Minimum mesh spacing in the z direction. Default is None.
    
        Returns:
        np.ndarray: The perturbed bottom coordinates.
        """
        self.perturb_bottom_coords_by_rotation(perturbations, pivot, is_utm, recalculate_pivot, angle_unit, force_pivot_in_coords)
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z, bias, min_dz)
        return self.bottom_coords

    @track_mesh_update(update_mesh=False, update_laplacian=False, update_area=False, expected_perturbations_count=4)
    def perturb_BottomFixedDir_RotateTransGeom(self, perturbations, average_direction=None, recalculate_pivot=False,
                                                          angle_unit='degrees', force_pivot_in_coords=False,
                                                          fixed_nodes=None, use_average_strike=True, pivot='midpoint'):
        assert hasattr(self, 'top_coords_ref'), 'You need to set top_coords_ref before perturbing the coordinates.'
        if recalculate_pivot or not hasattr(self, 'pivot'):
            pivot = self.calculate_pivot(pivot, self.top_coords_ref, True, force_pivot_in_coords)

        # 1. Perturb the bottom coordinates along the fixed direction
        self.bottom_coords = self.perturb_bottom_coords_along_fixed_direction(perturbations[:1], average_direction=average_direction, fixed_nodes=fixed_nodes, 
                                                         angle_unit='degrees', perturbation_direction='horizontal', use_average_strike=use_average_strike)

        # 2. Rotate the top and bottom coordinates
        self.top_coords = self.perturb_coords_by_rotation(self.top_coords_ref, perturbations[1:2], pivot, False, True, angle_unit)
        self.bottom_coords = self.perturb_coords_by_rotation(self.bottom_coords, perturbations[1:2], pivot, False, True, angle_unit)
        # layers = None
        # if hasattr(self, 'layers_ref'):
        #     layers = [self.perturb_layer_coords_by_rotation(i, perturbations[1:-1], pivot, False, True, angle_unit,) for i in range(len(self.layers_ref))]

        # 3. Translate the top and bottom coordinates
        self.top_coords = self.perturb_coords_by_translation(self.top_coords, perturbations[2:])
        self.bottom_coords = self.perturb_coords_by_translation(self.bottom_coords, perturbations[2:])
        self.set_coords(self.top_coords, lonlat=False, coord_type='top')
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        # layers = None
        # if hasattr(self, 'layers_ref'):
        #     layers = [self.perturb_layer_coords_by_translation(i, perturbations[1:-1], False, True) for i in range(len(self.layers_ref))]
        return 
    
    @track_mesh_update(update_mesh=True, update_laplacian=False, update_area=False, expected_perturbations_count=4)
    def perturb_BottomFixedDir_RotateTransGeom_simpleMesh(self, perturbations, average_direction=None, recalculate_pivot=False,
                                                          angle_unit='degrees', force_pivot_in_coords=False,
                                                          fixed_nodes=None, use_average_strike=True, pivot='midpoint',
                                                          disct_z=None, bias=None, min_dz=None):
        """
        Apply perturbations to the bottom coordinates and update the simple mesh.
    
        Approach: First rotate, then translate, and finally perturb the bottom coordinates.
        Suggestion: For efficiency, call self.calculate_pivot() to calculate the pivot in advance,
        then call self.perturb_top_coords_by_rotation() and self.perturb_bottom_coords_by_rotation() to perturb the top and bottom coordinates.
    
        Parameters:
        perturbations (float): The angle of perturbation. Positive for counterclockwise, negative for clockwise.
        average_direction (float, optional): The average direction (azimuth) in degrees. If provided, this value will be used directly.
        recalculate_pivot (bool, optional): Whether to recalculate the pivot point. Default is False.
        angle_unit (str, optional): The unit of the angle. Can be 'degrees' or 'radians'. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot to be one of the points in coords. Default is False.
        fixed_nodes (list, optional): A list of indices of fixed nodes. If provided, the function will not perturb these nodes.
        use_average_strike (bool, optional): Whether to use the average strike direction for all nodes. Default is True.
        pivot (str or tuple, optional): The center point of the perturbation. Can be 'start', 'end', 'midpoint', or a coordinate point (e.g., `(x, y)`). Default is 'midpoint'.
        disct_z (float, optional): Mesh density in the z direction. Default is None.
        bias (float, optional): Mesh bias in the z direction. Default is None.
        min_dz (float, optional): Minimum mesh spacing in the z direction. Default is None.
    
        Returns:
        np.ndarray: The perturbed bottom coordinates.
        """
        self.perturb_BottomFixedDir_RotateTransGeom(perturbations, average_direction, recalculate_pivot, angle_unit, 
                                                    force_pivot_in_coords, fixed_nodes, use_average_strike, pivot)
    
        # Generate the simple mesh
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z=disct_z, bias=bias, min_dz=min_dz)
        return
    
    @track_mesh_update(update_mesh=True, update_laplacian=False, update_area=False, expected_perturbations_count=6)
    def perturb_BottomFixedDir_RotateTransGeom_multiLayerMesh(self, perturbations, average_direction=None, recalculate_pivot=False,
                                                              angle_unit='degrees', force_pivot_in_coords=False,
                                                              fixed_nodes=None, use_average_strike=True, pivot='midpoint',
                                                              disct_z=5, bias=1.0):
        """
        Apply perturbations to the bottom coordinates and update the multi-layer mesh.
    
        Approach: First rotate, then translate, and finally perturb the bottom coordinates.
        Suggestion: For efficiency, call self.calculate_pivot() to calculate the pivot in advance,
        then call self.perturb_top_coords_by_rotation() and self.perturb_bottom_coords_by_rotation() to perturb the top and bottom coordinates.
        Assumption: One middle layer; the middle layer is perturbed in both horizontal and vertical directions, the bottom layer is perturbed only in the horizontal direction.
    
        Parameters:
        perturbations (list): The angles of perturbation. Positive for counterclockwise, negative for clockwise.
                              0: mid_layer offset along fixed direction
                              1: mid_layer offset along vertical direction
                              2: bottom offset along fixed direction
                              3: rotation of total geometry
                              4-5: dx, dy of total geometry
        average_direction (float, optional): The average direction (azimuth) in degrees. If provided, this value will be used directly.
        recalculate_pivot (bool, optional): Whether to recalculate the pivot point. Default is False.
        angle_unit (str, optional): The unit of the angle. Can be 'degrees' or 'radians'. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot to be one of the points in coords. Default is False.
        fixed_nodes (list, optional): A list of indices of fixed nodes. If provided, the function will not perturb these nodes.
        use_average_strike (bool, optional): Whether to use the average strike direction for all nodes. Default is True.
        pivot (str or tuple, optional): The center point of the perturbation. Can be 'start', 'end', 'midpoint', or a coordinate point (e.g., `(x, y)`). Default is 'midpoint'.
        disct_z (float, optional): Mesh density in the z direction. Default is 5.
        bias (float, optional): Mesh bias in the z direction. Default is 1.0.
    
        Returns:
        np.ndarray: The perturbed bottom coordinates.
        """
        assert hasattr(self, 'top_coords_ref'), 'You need to set top_coords_ref before perturbing the coordinates.'
        # Keep the pivot point in the same position
        if recalculate_pivot or not hasattr(self, 'pivot'):
            pivot = self.calculate_pivot(pivot, self.top_coords_ref, True, force_pivot_in_coords)
        # Keep the average direction in the same position
        if average_direction is None:
            # Calculate the strike direction between adjacent points in coords
            coords = self.top_coords
            trends = np.arctan2(np.diff(coords[:, 1]), np.diff(coords[:, 0]))
            trends = np.concatenate(([trends[0]], (trends[:-1] + trends[1:]) / 2, [trends[-1]]))
            trends_azi = np.pi - trends
            average_direction = np.degrees(np.mean(trends_azi))
    
        # Perturb the mid_layer and bottom coordinates along the fixed direction
        self.bottom_coords = self.perturb_bottom_coords_along_fixed_direction(perturbations[2:3], average_direction=average_direction, fixed_nodes=fixed_nodes, 
                                                                              angle_unit='degrees', perturbation_direction='horizontal', use_average_strike=use_average_strike)
        self.layers = [self.perturb_layer_coords_along_fixed_direction(perturbations[:1], i, average_direction, fixed_nodes, angle_unit, 'horizontal', use_average_strike) for i in range(len(self.layers_ref))]
        self.layers = [self.perturb_coords_along_fixed_direction(self.layers[i], perturbations[1:2], average_direction, fixed_nodes, angle_unit, 'vertical', use_average_strike) for i in range(len(self.layers))]
    
        # Rotate the top and bottom coordinates
        self.top_coords = self.perturb_coords_by_rotation(self.top_coords_ref, perturbations[3:4], pivot, False, True, angle_unit)
        self.bottom_coords = self.perturb_coords_by_rotation(self.bottom_coords, perturbations[3:4], pivot, False, True, angle_unit)
        self.layers = [self.perturb_coords_by_rotation(self.layers[i], perturbations[3:4], pivot, False, True, angle_unit) for i in range(len(self.layers))]
    
        # Translate the top and bottom coordinates
        self.top_coords = self.perturb_coords_by_translation(self.top_coords, perturbations[4:6])
        self.bottom_coords = self.perturb_coords_by_translation(self.bottom_coords, perturbations[4:6])
        self.set_coords(self.top_coords, lonlat=False, coord_type='top')
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        for i, layer in enumerate(self.layers):
            self.layers[i] = self.perturb_coords_by_translation(self.layers[i], perturbations[4:6])
    
        # Generate the multi-layer mesh
        self.generate_simple_multilayer_mesh(self.top_coords, self.layers, self.bottom_coords, disct_z=disct_z, bias=bias)
        return
    
    @track_mesh_update()
    def perturb_layer_coords_by_rotation(self, mid_layer_index, perturbations, pivot='midpoint', recalculate_pivot=False,
                                         is_utm=False, angle_unit='degrees', force_pivot_in_coords=False):
        """
        Apply perturbations to the coordinates of a specific layer by rotation.
    
        Parameters:
        mid_layer_index (int): The index of the middle layer to be perturbed.
        perturbations (float): The angle of perturbation. Positive for counterclockwise, negative for clockwise.
        pivot (str or tuple, optional): The center point of the perturbation. Can be 'start', 'end', 'midpoint', or a coordinate point (e.g., `(x, y)`). Default is 'midpoint'.
        recalculate_pivot (bool, optional): Whether to recalculate the pivot point. Default is False.
        is_utm (bool, optional): If pivot is a coordinate point, this parameter determines whether the coordinate point is in UTM coordinates. Default is False.
        angle_unit (str, optional): The unit of the angle. Can be 'degrees' or 'radians'. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot to be one of the points in coords. Default is False.
    
        Returns:
        np.ndarray: The perturbed coordinates of the specified layer.
        """
        assert hasattr(self, 'layers_ref') and mid_layer_index < len(self.layers_ref), 'You need to set layers_ref before perturbing the coordinates.'
        self.layers[mid_layer_index] = self.perturb_coords_by_rotation(self.layers_ref[mid_layer_index], perturbations, 
                                                                       pivot, recalculate_pivot, is_utm, angle_unit, force_pivot_in_coords)
        x, y, z = self.layers[mid_layer_index][:, 0], self.layers[mid_layer_index][:, 1], self.layers[mid_layer_index][:, 2]
        lon, lat = self.xy2ll(x, y)
        self.layers_ll[mid_layer_index] = np.vstack((lon, lat, z)).T
        return self.layers[mid_layer_index]
    
    @track_mesh_update(update_mesh=True, update_laplacian=True, update_area=True, expected_perturbations_count=1)
    def perturb_geometry_by_rotation(self, perturbations, pivot='midpoint', recalculate_pivot=False,
                                     angle_unit='degrees', force_pivot_in_coords=False):
        """
        Apply perturbations to all coordinates by rotation.
    
        Parameters:
        perturbations (float): The angle of perturbation. Positive for counterclockwise, negative for clockwise.
        pivot (str or tuple, optional): The center point of the perturbation. Can be 'start', 'end', 'midpoint', or a coordinate point (e.g., `(x, y)`). Default is 'midpoint'.
        recalculate_pivot (bool, optional): Whether to recalculate the pivot point. Default is False.
        angle_unit (str, optional): The unit of the angle. Can be 'degrees' or 'radians'. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot to be one of the points in coords. Default is False.
    
        Returns:
        np.ndarray: The perturbed coordinates.
        """
        assert hasattr(self, 'top_coords_ref'), 'You need to set top_coords_ref before perturbing the coordinates.'
        if recalculate_pivot or not hasattr(self, 'pivot'):
            pivot = self.calculate_pivot(pivot, self.top_coords_ref, True, force_pivot_in_coords)
    
        # Rotate the fault geometry mesh
        if not hasattr(self, 'Vertices_ref'):
            self.Vertices_ref = self.Vertices.copy()
        vertices_rot = self.perturb_coords_by_rotation(self.Vertices_ref, perturbations, pivot, False, True, angle_unit)
        self.VertFace2csifault(vertices_rot, self.Faces)
    
        return vertices_rot
    #------------------------------------------------------------------------------------------------------------#
    
    #--------------------------------------Perturbing Coords by Translation--------------------------------------#
    def perturb_coords_by_translation(self, coords, perturbations):
        """
        Apply a global translation to the given coordinates.
    
        Parameters:
        coords (np.ndarray): Original coordinate points in UTM.
        perturbations (tuple): Translation distances in the format `(dx, dy)`.
    
        Returns:
        np.ndarray: Translated coordinates.
        """
        # Perform translation
        coords = coords.copy()
        coords[:, :2] += perturbations
    
        return coords
    
    @track_mesh_update()
    def perturb_top_coords_by_translation(self, perturbations):
        """
        Apply a global translation to the top coordinates.
    
        Parameters:
        perturbations (tuple): Translation distances in the format `(dx, dy)`.
    
        Returns:
        np.ndarray: Translated top coordinates.
        """
        assert hasattr(self, 'top_coords_ref'), 'You need to set top_coords_ref before perturbing the coordinates.'
        self.top_coords = self.perturb_coords_by_translation(self.top_coords_ref, perturbations)
        self.set_coords(self.top_coords, lonlat=False, coord_type='top')
        return self.top_coords
    
    @track_mesh_update()
    def perturb_bottom_coords_by_translation(self, perturbations):
        """
        Apply a global translation to the bottom coordinates.
    
        Parameters:
        perturbations (tuple): Translation distances in the format `(dx, dy)`.
    
        Returns:
        np.ndarray: Translated bottom coordinates.
        """
        assert hasattr(self, 'bottom_coords_ref'), 'You need to set bottom_coords_ref before perturbing the coordinates.'
        self.bottom_coords = self.perturb_coords_by_translation(self.bottom_coords_ref, perturbations)
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True)
    def perturb_BottomTrans_simpleMesh(self, perturbations, disct_z=None, bias=None, min_dz=None):
        """
        Apply a global translation to the bottom coordinates and update the simple mesh.
    
        Parameters:
        perturbations (tuple): Translation distances in the format `(dx, dy)`.
    
        Returns:
        np.ndarray: Translated bottom coordinates.
        """
        self.perturb_bottom_coords_by_translation(perturbations)
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z, bias, min_dz)
        return self.bottom_coords
    
    @track_mesh_update()
    def perturb_layer_coords_by_translation(self, perturbations, mid_layer_index=0):
        """
        Apply a global translation to the coordinates of a specific layer.
    
        Parameters:
        perturbations (tuple): Translation distances in the format `(dx, dy)`.
        mid_layer_index (int): Index of the layer to be translated.
    
        Returns:
        np.ndarray: Translated coordinates of the specified layer.
        """
        assert hasattr(self, 'layers_ref') and mid_layer_index < len(self.layers_ref), 'You need to set layers_ref before perturbing the coordinates.'
        self.layers[mid_layer_index] = self.perturb_coords_by_translation(self.layers_ref[mid_layer_index], perturbations)
        x, y, z = self.layers[mid_layer_index][:, 0], self.layers[mid_layer_index][:, 1], self.layers[mid_layer_index][:, 2]
        lon, lat = self.xy2ll(x, y)
        self.layers_ll[mid_layer_index] = np.vstack((lon, lat, z)).T
        return self.layers[mid_layer_index]
    
    @track_mesh_update(update_mesh=True, update_laplacian=True, update_area=True, expected_perturbations_count=2)
    def perturb_geometry_by_translation(self, perturbations):
        """
        Apply a global translation to all coordinates.
    
        Parameters:
        perturbations (tuple): Translation distances in the format `(dx, dy)`.
    
        Returns:
        np.ndarray: Translated vertices.
        """
        # Translate the fault geometry mesh
        if not hasattr(self, 'Vertices_ref'):
            self.Vertices_ref = self.Vertices.copy()
        vertices_trans = self.perturb_coords_by_translation(self.Vertices_ref, perturbations)
        self.VertFace2csifault(vertices_trans, self.Faces)
    
        return vertices_trans
    #---------------------------------------------------------------------------------------------------------------#
    
    #-------------------------------------Perturbing geometry by translation and rotation-------------------------------------#
    @track_mesh_update(update_mesh=True, update_laplacian=True, update_area=True, expected_perturbations_count=3)
    def perturb_geometry_by_rotation_and_translation(self, perturbations, pivot='midpoint', is_utm=False,
                                                     angle_unit='degrees', force_pivot_in_coords=False):
        """
        Apply a global rotation and translation to all coordinates.
    
        Parameters:
        perturbations (list): A list containing the rotation angle and translation distances. Format: [rotation_angle, dx, dy].
        pivot (str or array-like, optional): The pivot point for rotation. Default is 'midpoint', which means the midpoint of the top coordinates.
        is_utm (bool, optional): Whether the coordinates are in UTM. Default is False.
        angle_unit (str, optional): The unit of the rotation angle. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot point to be within the coordinates. Default is False.
    
        Returns:
        np.ndarray: The rotated and translated vertices.
        """
        assert hasattr(self, 'top_coords_ref'), 'You need to set top_coords_ref before perturbing the coordinates.'
        pivot = self.calculate_pivot(pivot, self.top_coords_ref, True, force_pivot_in_coords)
    
        # Rotate the fault geometry mesh
        if not hasattr(self, 'Vertices_ref'):
            self.Vertices_ref = self.Vertices.copy()
        vertices_rot = self.perturb_coords_by_rotation(self.Vertices_ref, perturbations[0:1], pivot, False, True, angle_unit)
    
        # Translate the fault geometry mesh
        vertices_rot_trans = self.perturb_coords_by_translation(vertices_rot, perturbations[1:])
        self.VertFace2csifault(vertices_rot_trans, self.Faces)
    
        return vertices_rot_trans
    #---------------------------------------------------------------------------------------------------------------#

    #--------------------------------------Perturbing Coords by Function Fitting--------------------------------------#
    def perturb_coord_dutta(self, coords, fixed_nodes, perturbations):
        """
        Perturb coordinates between two specific nodes.
    
        Parameters:
        coords (np.ndarray): Original coordinate points.
        fixed_nodes (list): A list containing indices of two fixed nodes.
        perturbations (list): A list containing two perturbation values corresponding to amod1 and amod2.
    
        Returns:
        np.ndarray: Perturbed coordinate points.
        """
        # Copy the coordinates to avoid modifying the original array
        coords = coords.copy()
        Z = coords[:, 2]
        coords = coords[:, :2]
        p1 = coords[fixed_nodes[0]]
        p2 = coords[fixed_nodes[1]]
        new_coords = np.array([transform_point(p1, p2, p) for p in coords])
        
        # Apply perturbations in the new coordinate system
        amod1, amod2 = perturbations
        inds = np.arange(coords.shape[0])
        st, ed = inds[fixed_nodes[0]+1], inds[fixed_nodes[1]]
        for i in range(st, ed):
            xfault_newref = new_coords[i, 0]
            yfault_newref = amod2 * (xfault_newref) * (xfault_newref - amod1) * (xfault_newref - 2)
            new_coords[i, 1] = yfault_newref
    
        # Transform coordinates back to the original coordinate system
        coords[fixed_nodes[0]+1:fixed_nodes[1]] = np.array([reverse_transform_point(p1, p2, p) for p in new_coords[fixed_nodes[0]+1:fixed_nodes[1]]]).reshape(-1, 2)
        coords = np.c_[coords, Z]
        return coords
    
    @track_mesh_update(expected_perturbations_count=2)
    def perturb_bottom_coords_dutta(self, perturbations, fixed_nodes=[0, -1]):
        """
        Perturb bottom coordinates between two specific nodes.
    
        Parameters:
        perturbations (list): A list containing two perturbation values corresponding to amod1 and amod2.
    
        Returns:
        np.ndarray: Perturbed bottom coordinates.
        """
        assert hasattr(self, 'bottom_coords_ref'), 'You need to set bottom_coords_ref before perturbing the coordinates.'
        self.bottom_coords = self.perturb_coord_dutta(self.bottom_coords_ref, fixed_nodes, perturbations)
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True, update_laplacian=False, update_area=False, expected_perturbations_count=4)
    def perturb_geometry_dutta(self, perturbations, disct_z=None, bias=None, min_dx=None):
        """
        Generate a mesh for a seismic fault using the Dutta method.
    
        Parameters:
        perturbations (list): Perturbation parameters to control the shape of the seismic fault. [D1, D2, S1, S2]
        disct_z (float, optional): Discretization parameter in the z direction. If None, the default value will be used.
        bias (float, optional): Bias parameter for the mesh. If None, the default value will be used.
        min_dx (float, optional): Minimum size of the mesh. If None, the default value will be used.
    
        Returns:
        None. This function modifies self.top_coords and self.VertFace2csifault.
    
        Note:
        This function modifies the z-coordinates of self.top_coords to be negative, as the z-direction is downward in the seismic model.
        """
        top_coords = self.top_coords.copy()
        top_coords[:, 2] = -top_coords[:, 2]
        p, q, r, trired, ang, xfault, yfault, zfault = make_mesh_dutta(self.depth, 
                                                                       perturbations, 
                                                                       top_coords, 
                                                                       disct_z, 
                                                                       bias, 
                                                                       min_dx)
        vertices = np.vstack((p.flatten(), q.flatten(), -r.flatten())).T
        self.VertFace2csifault(vertices, trired)
    #-------------------------------------------------------------------------------------------------------------------#

    #-------------------------------Perturbing Bottom Coords by Translation and Rotation------------------------#
    @track_mesh_update(update_mesh=False, update_laplacian=False, expected_perturbations_count=3)
    def perturb_BottomRotateTrans(self, perturbations, pivot='midpoint', is_utm=False,
                                  angle_unit='degrees', force_pivot_in_coords=False):
        """
        Modify the bottom coordinates by rotation and translation.
    
        Parameters:
        perturbations (list): A list containing the rotation angle and translation distance. Format: `[rotation, translation]`.
        pivot (str or array-like): The pivot point for rotation. Default is 'midpoint', which means the midpoint of the bottom coordinates.
        is_utm (bool): Whether the coordinates are in UTM. Default is False.
        angle_unit (str): The unit of the rotation angle. Default is 'degrees'.
        force_pivot_in_coords (bool): Whether to force the pivot point to be within the coordinates. Default is False.
    
        Returns:
        None. This method modifies the object's bottom coordinates in place.
        """
        assert hasattr(self, 'bottom_coords_ref'), 'You need to set bottom_coords_ref before perturbing the coordinates.'
        pivot = self.calculate_pivot(pivot, self.bottom_coords_ref, True, force_pivot_in_coords)
    
        # Rotate the bottom coordinates
        self.bottom_coords = self.perturb_coords_by_rotation(self.bottom_coords_ref, perturbations[0:1], pivot, False, True, angle_unit)
        # Translate the bottom coordinates
        self.bottom_coords = self.perturb_coords_by_translation(self.bottom_coords, perturbations[1:])
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True, expected_perturbations_count=3)
    def perturb_BottomRotateTrans_simpleMesh(self, perturbations, pivot='midpoint', is_utm=False,
                                                        angle_unit='degrees', force_pivot_in_coords=False,
                                                        disct_z=None, bias=None, min_dz=None):
        self.perturb_BottomRotateTrans(perturbations, pivot, is_utm, angle_unit, force_pivot_in_coords)
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z, bias, min_dz)
    #------------------------------------------------------------------------------------------------------------#

    #--------------------------------------Simply Mesh From Top to Bottom--------------------------------------#
    def set_top_coords_ref(self, top_coords_ref=None):
        if top_coords_ref is None:
            self.top_coords_ref = self.top_coords.copy()
        else:
            self.top_coords_ref = top_coords_ref
    
    def set_bottom_coords_ref(self, bottom_coords_ref=None):
        if bottom_coords_ref is None:
            self.bottom_coords_ref = self.bottom_coords.copy()
        else:
            self.bottom_coords_ref = bottom_coords_ref
    
    def set_layers_ref(self, layers_ref=None):
        if layers_ref is None:
            self.layers_ref = [layer.copy() for layer in self.layers]

    def generate_simple_mesh(self, top_coords=None, bottom_coords=None, disct_z=None, 
                                                bias=None, min_dz=None, use_depth_only=True):
        """
        Generate a simple earthquake fault mesh from top to bottom coordinates.
    
        Parameters:
        - top_coords (ndarray): The top coordinates of the fault.
        - bottom_coords (ndarray): The bottom coordinates of the fault.
        - disct_z (int, optional): Discretization parameter in the z-direction. If provided, it overrides bias and min_dz.
        - bias (float, optional): Bias parameter for the mesh. Required if disct_z is None.
        - min_dz (float, optional): Minimum size of the mesh in the z-direction. Required if disct_z is None.
        - use_depth_only (bool, optional): If True, the full length is the mean depth; otherwise, it is the mean of the entire length. Default is True.
    
        Returns:
        - None: This function modifies self.top_coords and self.VertFace2csifault.
    
        Notes:
        - If disct_z is provided, it takes precedence over bias and min_dz.
        - If disct_z is None, both bias and min_dz must be provided.
        - This function modifies the z-coordinates of self.top_coords to be negative, as the z-direction is downward in the earthquake model.
        """
        top_coords = top_coords if top_coords is not None else self.top_coords
        bottom_coords = bottom_coords if bottom_coords is not None else self.bottom_coords
        self.mesh_generator.set_coordinates(top_coords, bottom_coords)
        vertices, faces = self.mesh_generator.generate_simple_mesh(disct_z, bias, min_dz, use_depth_only)
        self.VertFace2csifault(vertices, faces)

    def generate_simple_multilayer_mesh(self, top_coords=None, layers_coords=None, bottom_coords=None, disct_z=8, bias=1.0):
        """
        Generate a multi-layer earthquake fault mesh.
    
        Parameters:
        - top_coords (numpy.ndarray): The top coordinates, shaped (n, d), where n is the number of points and d is the dimension.
        - layers_coords (list of numpy.ndarray): The coordinates of the intermediate layers, each shaped (n, d).
        - bottom_coords (numpy.ndarray): The bottom coordinates, shaped (n, d).
        - disct_z (int): Discretization parameter in the z-direction, representing the number of segments.
        - bias (float, optional): Bias used to adjust the length of each segment. Default is 1.0.
    
        Returns:
        - None: This function modifies self.top_coords and self.VertFace2csifault.
    
        Notes:
        - The number of points in top_coords, layers_coords, and bottom_coords must be the same.
        - The function modifies the z-coordinates of self.top_coords to be negative, as the z-direction is downward in the earthquake model.
        """
        top_coords = top_coords if top_coords is not None else self.top_coords
        layers_coords = layers_coords if layers_coords is not None else self.layers
        bottom_coords = bottom_coords if bottom_coords is not None else self.bottom_coords
        self.mesh_generator.set_coordinates(top_coords, bottom_coords)
        vertices, faces = self.mesh_generator.generate_multilayer_mesh(layers_coords, disct_z, bias)
        self.VertFace2csifault(vertices, faces)

    def generate_and_deform_mesh(self, top_coords=None, bottom_coords=None, top_size=None, bottom_size=None, num_segments=30, 
                                 disct_z=10, rotation_angle: float = None, bottom_norm_offset=None, show=False, 
                                 verbose=0, remap=False, bias=None, min_dz=None, projection=None, 
                                 field_size_dict={'min_dx': 3, 'bias': 1.05}, mesh_func=None, tolerance=1e-6, debug_plot=False,
                                 use_current_mesh=False):
        """
        Generate and deform mesh.
    
        Parameters:
        - top_coords (np.ndarray): Top coordinates for the grid.
        - bottom_coords (np.ndarray): Bottom coordinates for the grid.
        
        Gmsh Generation Parameters:
        - top_size (float): Size of the top mesh. Default is 2.0.
        - bottom_size (float): Size of the bottom mesh. Default is 4.0.
        - field_size_dict (dict, optional): Dictionary containing 'min_dx' and 'bias' for mesh size progression.
        - mesh_func (callable, optional): Function to define mesh size.
        
        Grid Generation Parameters:
        - num_segments (int): Number of segments for discretization. Default is 30.
        - disct_z (int): Number of divisions in the z-direction. Default is 10.
        - bias (float, optional): Grid bias in the z-direction. Default is None.
        - min_dz (float, optional): Minimum grid spacing in the z-direction. Default is None.
        
        Mapping Parameters:
        - projection (str, optional): Projection plane to use ('xy', 'xz', 'yz'). Default is None.
        - bottom_norm_offset (float, optional): Perturbations to apply to the bottom coordinates. Default is None.
        - rotation_angle (float): Angle to rotate the grid and vertices. Default is None.
        - remap (bool): Whether to remap Gmsh vertices to grid. Default is False.
        - tolerance (float): Tolerance for remapping. Default is 1e-6.
        - debug_plot (bool): Whether to show debug plots. Default is False.
        - use_current_mesh (bool): Whether to use current vertices and faces instead of generating Gmsh mesh. Default is False.
        
        Other Parameters:
        - show (bool): Whether to show the mesh. Default is False.
        - verbose (int): Verbosity level. Default is 0.
    
        Returns:
        - new_gmsh_verts (np.ndarray): Array of new Gmsh vertices.
        - gmsh_faces (np.ndarray): Array of Gmsh faces.
        """
        top_coords = top_coords if top_coords is not None else self.top_coords
        bottom_coords = bottom_coords if bottom_coords is not None else self.bottom_coords
    
        if use_current_mesh:
            # Use existing vertices and faces instead of generating Gmsh mesh
            print("Using existing mesh (self.Vertices and self.Faces)")
            gmsh_verts = self.Vertices
            gmsh_faces = self.Faces
            
            # Apply bottom coordinate perturbation if specified
            if bottom_norm_offset is not None:
                self.perturb_bottom_coords_along_fixed_direction([bottom_norm_offset])
            top_coords, bottom_coords = self.top_coords, self.bottom_coords
            
            # Update coordinates in mesh_generator for grid generation
            self.mesh_generator.set_coordinates(top_coords, bottom_coords)
            
            # Store mesh data in mesh_generator for consistency
            self.mesh_generator.gmsh_verts = gmsh_verts
            self.mesh_generator.gmsh_faces = gmsh_faces

            # Discretize bottom and top coordinates
            sep_top_coords, _ = self.discretize_coords(top_coords, num_segments=num_segments)
            sep_bottom_coords, _ = self.discretize_coords(bottom_coords, num_segments=num_segments)
            # Update top and bottom coordinates in mesh_generator
            self.mesh_generator.set_coordinates(sep_top_coords, sep_bottom_coords)
            
            # Generate grid coordinates for deformation
            mesh_coords = self.mesh_generator.generate_grid_coordinates(top_coords=self.mesh_generator.top_coords, 
                                                                        bottom_coords=self.mesh_generator.bottom_coords, 
                                                                        disct_z=disct_z, bias=bias, min_dz=min_dz)
            
            # Map Gmsh vertices to grid
            self.mesh_generator.map_gmsh_vertices_to_grid(gmsh_verts, mesh_coords, 
                                                            rotation_angle=rotation_angle, 
                                                            projection=projection, 
                                                            tolerance=tolerance, 
                                                            debug_plot=debug_plot)
            
        else:
            # Original Gmsh mesh generation path
            # Only generate Gmsh mesh and remap if param_coords is None or remap is True
            if self.mesh_generator.param_coords is None or remap:
                if bottom_norm_offset is not None:
                    self.perturb_bottom_coords_along_fixed_direction([bottom_norm_offset])
                top_coords, bottom_coords = self.top_coords, self.bottom_coords
                # Update coordinates in mesh_generator
                self.mesh_generator.set_coordinates(top_coords, bottom_coords)
                # Generate Gmsh mesh
                gmsh_verts, gmsh_faces = self.mesh_generator.generate_gmsh_mesh(top_size=top_size, bottom_size=bottom_size, 
                                                                                show=show, verbose=verbose, save_in_self=True, 
                                                                                field_size_dict=field_size_dict, mesh_func=mesh_func)
                
                # Discretize bottom and top coordinates
                sep_top_coords, _ = self.discretize_coords(top_coords, num_segments=num_segments)
                sep_bottom_coords, _ = self.discretize_coords(bottom_coords, num_segments=num_segments)
                # Update top and bottom coordinates in mesh_generator
                self.mesh_generator.set_coordinates(sep_top_coords, sep_bottom_coords)
                
                # Generate grid coordinates
                mesh_coords = self.mesh_generator.generate_grid_coordinates(top_coords=self.mesh_generator.top_coords, 
                                                                            bottom_coords=self.mesh_generator.bottom_coords, 
                                                                            disct_z=disct_z, bias=bias, min_dz=min_dz)
                
                # Map Gmsh vertices to grid
                self.mesh_generator.map_gmsh_vertices_to_grid(gmsh_verts, mesh_coords, 
                                                              rotation_angle=rotation_angle, 
                                                              projection=projection, 
                                                              tolerance=tolerance, 
                                                              debug_plot=debug_plot)
            else:
                # Assume gmsh_verts and gmsh_faces are already available
                gmsh_verts = self.mesh_generator.gmsh_verts
                gmsh_faces = self.mesh_generator.gmsh_faces
        
        # Common deformation path for both cases
        # Discretize bottom and top coordinates again
        sep_top_coords, _ = self.discretize_coords(top_coords, num_segments=num_segments)
        sep_bottom_coords, _ = self.discretize_coords(bottom_coords, num_segments=num_segments)
        
        # Deform mesh
        new_gmsh_verts = self.mesh_generator.deform_mesh(sep_top_coords, sep_bottom_coords, disct_z=disct_z, 
                                                            bias=bias, min_dz=min_dz, projection=projection)
        
        # Generate new Gmsh vertices and faces
        self.VertFace2csifault(new_gmsh_verts, gmsh_faces)
        
        return new_gmsh_verts, gmsh_faces
    
    def rebuild_simple_mesh(self, disct_z=None, bias=None, min_dz=None, segs=5, top_tolerance=0.1, bottom_tolerance=0.1, lonlat=True, buffer_depth=0.1, sort_axis=0, sort_order='ascend', use_trace=False, discretized=False):
        """
        Rebuild the simple earthquake fault mesh for the usage of the perturbation functions.
    
        Parameters:
        -----------
        disct_z : float, optional
            Discretization parameter in the z-direction.
        bias : float, optional
            Bias parameter for the mesh.
        min_dz : float, optional
            Minimum size of the mesh in the z-direction.
        segs : int, optional
            Number of uniform segments for top and bottom coordinates. Default is 5.
        top_tolerance : float, optional
            Tolerance for finding top edge vertices. Default is 0.1.
        bottom_tolerance : float, optional
            Tolerance for finding bottom edge vertices. Default is 0.1.
        lonlat : bool, optional
            Whether to use longitude and latitude coordinates. Default is True.
        buffer_depth : float, optional
            The buffer depth to include points within the edge. Default is 0.1.
        sort_axis : int, optional
            Axis to sort the coordinates. Default is 0.
        sort_order : str, optional
            Order to sort the coordinates ('ascend' or 'descend'). Default is 'ascend'.
        use_trace : bool, optional
            Whether to use fault trace to set top coordinates. Default is False.
        discretized : bool, optional
            If True, uses the discretized fault trace. Otherwise uses the original fault trace. Default is False.
    
        Returns:
        --------
        None
        """
        if use_trace:
            self.set_top_coords_from_trace(discretized=discretized)
        else:
            self.set_top_coords_from_geometry(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, lonlat=lonlat, buffer_depth=buffer_depth, sort_axis=sort_axis, sort_order=sort_order)
        
        self.set_bottom_coords_from_geometry(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, lonlat=lonlat, buffer_depth=buffer_depth, sort_axis=sort_axis, sort_order=sort_order)
        self.discretize_bottom_coords(num_segments=segs)
        self.discretize_top_coords(num_segments=segs)
        self.set_bottom_coords_ref()
        self.set_top_coords_ref()
        # Rebuild the simple earthquake fault mesh
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z=disct_z, bias=bias, min_dz=min_dz, use_depth_only=True)
        self.initializeslip(values='depth')
        return
    
    def set_edges_for_bayesian_optimization(self, segs=None, top_tolerance=0.1, bottom_tolerance=0.1, lonlat=True, depth_tolerance=0.1, buffer_depth=0.1, sort_axis=0, sort_order='ascend', use_trace=False, discretized=False):
        """
        Setup the top and bottom edges for Bayesian geometry optimization by applying perturbations and setting up the coordinates.
    
        Parameters:
        -----------
        segs : int, optional
            Number of uniform segments for top and bottom coordinates. Default is 5.
        top_tolerance : float, optional
            Tolerance for finding top edge vertices. Default is 0.1.
        bottom_tolerance : float, optional
            Tolerance for finding bottom edge vertices. Default is 0.1.
        lonlat : bool, optional
            Whether to use longitude and latitude coordinates. Default is True.
        depth_tolerance : float, optional
            Tolerance for depth selection. Default is 0.1.
        buffer_depth : float, optional
            The buffer depth to include points within the edge. Default is 0.1.
        sort_axis : int, optional
            Axis to sort the coordinates. Default is 0.
        sort_order : str, optional
            Order to sort the coordinates ('ascend' or 'descend'). Default is 'ascend'.
        use_trace : bool, optional
            Whether to use fault trace to set top coordinates. Default is False.
        discretized : bool, optional
            If True, uses the discretized fault trace. Otherwise uses the original fault trace. Default is False.
    
        Returns:
        --------
        None
        """
        # Set up top and bottom coordinates from geometry or trace
        if use_trace:
            self.set_top_coords_from_trace(discretized=discretized)
        else:
            self.set_top_coords_from_geometry(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, lonlat=lonlat, buffer_depth=buffer_depth, sort_axis=sort_axis, sort_order=sort_order)
        
        self.set_bottom_coords_from_geometry(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, lonlat=lonlat, buffer_depth=buffer_depth, sort_axis=sort_axis, sort_order=sort_order)
        
        if segs is not None:
            # Discretize bottom and top coordinates
            self.discretize_bottom_coords(num_segments=segs)
            self.discretize_top_coords(num_segments=segs)
        
        # Set reference coordinates for bottom and top
        self.set_bottom_coords_ref()
        self.set_top_coords_ref()
        
        return
    #----------------------------------------------------------------------------------------------------------#

    #----------------------------------------Geometry Opertations----------------------------------------------#
    def calculate_bottom_endpoints(self, top_coords=None, bottom_coords=None, depth=None, mode='both'):
        from .geom_ops import PolygonIntersector
        if top_coords is None:
            top_coords = self.top_coords
        if bottom_coords is None:
            bottom_coords = self.bottom_coords.copy()
            bottom_coords[:, -1] *= -1.0
        if depth is None:
            depth = self.depth
        patch = PolygonIntersector(top_coords, bottom_coords, depth)
        start, end = patch.calculate_intersections(mode=mode)
        start[-1] *= -1.0
        end[-1] *= -1.0
        self.intersector = patch
        return start, end
    
    @track_mesh_update(update_mesh=False, expected_perturbations_count=4)
    def perturb_BottomEndpointsFixedDirAndMidpointTrans(self, perturbations):
        """
        Perturbs the bottom coordinates with limited horizontal movements for the endpoints
        and free movement for the midpoint.
    
        Parameters:
        - endpoint_perturbations: A tuple/list containing the horizontal perturbations for the endpoints.
        - midpoint_perturbation: A tuple/list containing the x and y perturbations for the midpoint.
        """
        assert hasattr(self, 'bottom_coords_ref') and self.bottom_coords_ref.shape[0] == 3, 'You need to set bottom_coords_ref before perturbing the coordinates and it must contain 3 points.'
        # Perturb the bottom coordinates
        # Perturb the endpoints
        if not hasattr(self, 'average_direction') or self.average_direction is None:
            # 计算coords中相邻两点之间的走向
            coords = self.top_coords_ref
            trends = np.arctan2(np.diff(coords[:, 1]), np.diff(coords[:, 0]))
            trends = np.concatenate(([trends[0]], (trends[:-1] + trends[1:]) / 2, [trends[-1]]))
            trends_azi = np.pi - trends
            self.average_direction = np.array([trends_azi[0], trends_azi[0], trends_azi[-1]])
        self.perturb_bottom_coords_along_fixed_direction(perturbations[:2], average_direction=self.average_direction, fixed_nodes=[1], angle_unit='radians', 
                                                         perturbation_direction='horizontal', use_average_strike=False)
        # Perturb the midpoint
        self.bottom_coords[1, 0] += perturbations[2]
        self.bottom_coords[1, 1] += perturbations[3]
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords 
    
    @track_mesh_update(update_mesh=True, expected_perturbations_count=4)
    def perturb_BottomEndpointsFixedDirAndMidpointTrans_simpleMesh(self, perturbations, disct_z=None, bias=None, min_dz=None):
        """
        Perturbs the bottom coordinates with limited horizontal movements for the endpoints
        and free movement for the midpoint.
    
        Parameters:
        - endpoint_perturbations: A tuple/list containing the horizontal perturbations for the endpoints.
        - midpoint_perturbation: A tuple/list containing the x and y perturbations for the midpoint.
        """
        self.perturb_BottomEndpointsFixedDirAndMidpointTrans(perturbations)

        # Generate geometry mesh
        self.discretize_bottom_coords(num_segments=self.top_coords.shape[0])
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z=disct_z, bias=bias, min_dz=min_dz)
        return 
    
    @track_mesh_update(update_mesh=False, expected_perturbations_count=4)
    def perturb_BottomEndpointsFixedDirandMidpointsDutta(self, perturbations):
        assert hasattr(self, 'bottom_coords_ref'), 'You need to set bottom_coords_ref before perturbing the coordinates.'
        assert self.bottom_coords_ref.shape[0] == self.top_coords_ref.shape[0], 'The number of top and bottom coordinates must be the same.'
        # Perturb middle points first using Dutta method
        self.perturb_bottom_coords_dutta(perturbations[:2])
    
        # Then perturb the endpoints along the fixed direction
        if not hasattr(self, 'average_direction') or self.average_direction is None:
            coords = self.top_coords_ref
            trends = np.arctan2(np.diff(coords[:, 1]), np.diff(coords[:, 0]))
            trends = np.concatenate(([trends[0]], (trends[:-1] + trends[1:]) / 2, [trends[-1]]))
            trends_azi = np.pi - trends
            self.average_direction = trends_azi
    
        self.bottom_coords = self.perturb_coords_along_fixed_direction(self.bottom_coords, perturbations[2:], 
                                                         average_direction=self.average_direction, fixed_nodes=np.arange(self.bottom_coords.shape[0])[1:-1], 
                                                         angle_unit='radians', perturbation_direction='horizontal', use_average_strike=False)
    
        # # Calculate the vector from start to end and its total distance
        # vector_start_end = self.bottom_coords_ref[-1] - self.bottom_coords_ref[0]
        # total_distance = np.linalg.norm(vector_start_end)

        # # Calculate the displacement for the start and end points
        # start_displacement = self.bottom_coords[0] - self.bottom_coords_ref[0]
        # end_displacement = self.bottom_coords[-1] - self.bottom_coords_ref[-1]

        # # Calculate vectors from start to each middle node
        # vectors_start_node = self.bottom_coords_ref[1:-1] - self.bottom_coords_ref[0]

        # # Compute projection lengths and displacement proportions
        # projection_lengths = np.dot(vectors_start_node, vector_start_end) / total_distance**2
        # displacements = start_displacement + projection_lengths[:, np.newaxis] * (end_displacement - start_displacement)

        # # Apply displacements to the middle nodes
        # self.bottom_coords[1:-1] += displacements
        
        # Calculate reference and current vectors
        vector_ref = self.bottom_coords_ref[-1] - self.bottom_coords_ref[0]
        vector = self.bottom_coords[-1] - self.bottom_coords[0]
        
        # Calculate angle and norm differences
        angle_diff = np.angle(vector[0] + 1j * vector[1]) - np.angle(vector_ref[0] + 1j * vector_ref[1])
        norm_diff = np.linalg.norm(vector) / np.linalg.norm(vector_ref)
        
        # Convert coordinates to complex for manipulation
        bottom_coords_complex = self.bottom_coords[:, 0] + 1.j * self.bottom_coords[:, 1]
        bottom_ref_st_complex = self.bottom_coords_ref[0, 0] + 1.j * self.bottom_coords_ref[0, 1]
        # start_diff = bottom_coords_complex[0] - bottom_ref_st_complex
        
        # Apply rotation, scaling, and translation adjustments
        bottom_coords_complex[1:-1] = bottom_coords_complex[0] + norm_diff * np.exp(1j * angle_diff) * (bottom_coords_complex[1:-1] - bottom_ref_st_complex)
        
        # Convert back to real coordinates
        self.bottom_coords[:, :2] = np.column_stack([bottom_coords_complex.real, bottom_coords_complex.imag])
        
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True, expected_perturbations_count=4)
    def perturb_BottomEndpointsFixedDirandMidpointsDutta_simpleMesh(self, perturbations, disct_z=None, bias=None, min_dz=None):
        """
        Perturbs the bottom coordinates with limited horizontal movements for the endpoints
        and free movement for the midpoint using the Dutta method, while maintaining the original shape structure.
        """
        self.perturb_BottomEndpointsFixedDirandMidpointsDutta(perturbations)
        # Generate geometry mesh
        if self.top_coords.shape[0] != self.bottom_coords.shape[0]:
            self.discretize_bottom_coords(num_segments=self.top_coords.shape[0])
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z=disct_z, bias=bias, min_dz=min_dz)
        return 
    #----------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    lon0, lat0 = 116.5, 39.5
    myfault = BayesianAdaptiveTriangularPatches('myfault', lon0=lon0, lat0=lat0)
    # Print the perturbation methods
    print(myfault.perturbation_methods.keys())