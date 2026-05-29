"""
DirectionPerturbationMixin — Fixed-direction perturbation methods.

Extracted from BayesianAdaptiveTriangularPatches for modularity.
All methods are pure move-over with zero logic changes.
"""
import numpy as np
from ..bayesian_perturbation_base import track_mesh_update
from .angle_utils import (
    circular_mean_radians,
    expand_angles_to_radians,
    normalize_angle_unit,
)


class DirectionPerturbationMixin:
    """Mixin providing fixed-direction perturbation capabilities."""

    #--------------------------------------Perturbing Coords along fixed direction--------------------------------------#
    @staticmethod
    def _compute_strike_fallback(coords):
        """Compute per-node strike when the concrete class has no helper."""
        coords = np.asarray(coords, dtype=float)
        if coords.shape[0] < 2:
            raise ValueError("At least two coordinates are required to compute strike.")

        x, y = coords[:, 0], coords[:, 1]
        segment_strike = 90 - np.degrees(np.arctan2(np.diff(y), np.diff(x)))
        segment_strike_rad = np.radians(segment_strike)
        average_strike_rad = np.arctan2(
            (np.sin(segment_strike_rad[:-1]) + np.sin(segment_strike_rad[1:])) / 2,
            (np.cos(segment_strike_rad[:-1]) + np.cos(segment_strike_rad[1:])) / 2,
        )
        average_strike = np.degrees(average_strike_rad)
        return np.concatenate(([segment_strike[0]], average_strike, [segment_strike[-1]]))

    def calculate_perturb_direction(self, coords, angle_unit='degrees', use_average_strike=False, average_direction=None):
        """Compute the **strike-normal azimuth** used as the perturbation direction.

        Convention
        ----------
        Returns ``strike_azimuth + 90°`` **in radians**, i.e. the azimuth of
        the right-hand perpendicular (dip-side) of the fault trace.

        The downstream code in :meth:`perturb_coords_along_fixed_direction`
        converts this to a Cartesian direction vector via::

            math_angle = π/2 − trends_azi
            direction  = [cos(math_angle), sin(math_angle)]

        which yields the unit vector pointing from the trace towards the
        dip side (right-hand rule relative to the strike direction).

        Parameters
        ----------
        coords : (N, 2+) ndarray
            Coordinate points (projected CRS).
        angle_unit : str
            Unit for *average_direction* when provided: ``'degrees'`` or
            ``'radians'``.  Does NOT affect the output (always radians).
        use_average_strike : bool
            If True, all nodes share the circular mean direction.
        average_direction : float or array-like, optional
            User-supplied strike-normal azimuth in the unit given by
            *angle_unit*. Scalars are broadcast to all nodes; arrays must
            match the coordinate count. When provided, overrides the
            computation from *coords*.

        Returns
        -------
        trends_azi : (N,) ndarray
            Strike-normal azimuth in **radians**.
        """
        coords = np.asarray(coords)
        n_coords = coords.shape[0]
        if n_coords < 2:
            raise ValueError("At least two coordinates are required to compute perturbation direction.")
        normalize_angle_unit(angle_unit)

        if average_direction is not None:
            return expand_angles_to_radians(
                average_direction,
                n_coords,
                angle_unit=angle_unit,
                use_average=use_average_strike,
                name='average_direction',
            )
    
        if average_direction is None:
            # compute_strike returns along-strike azimuth; perturbation uses
            # the perpendicular convention (strike_azimuth + π/2) so that the
            # downstream π/2 − trends_azi conversion yields the dip-direction vector.
            compute_strike = getattr(self, 'compute_strike', self._compute_strike_fallback)
            strike_deg = compute_strike(coords)
            trends_azi = np.radians(strike_deg) + np.pi / 2
            if use_average_strike:
                trends_azi = np.full(n_coords, circular_mean_radians(trends_azi))
        return trends_azi
    
    def perturb_coords_along_fixed_direction(self, coords, perturbations, average_direction=None, 
                                             fixed_nodes=None, angle_unit='degrees', perturbation_direction='horizontal',
                                             use_average_strike=False):
        """
        Apply random perturbations to the coordinates along a fixed direction.
    
        Parameters:
        coords (np.ndarray): Original coordinate points.
        perturbations (np.ndarray): Direct perturbation values. If provided, the function will use these values directly.
        average_direction (float or array-like, optional): Strike-normal azimuth
            in radians or degrees. Scalars are broadcast; arrays must match
            the number of coordinates. If None, the direction is calculated
            from coords.
        fixed_nodes (list, optional): A list of indices of fixed nodes. If provided, the function will not perturb these nodes.
        angle_unit (str, optional): The unit of the input angles, can be 'radians' or 'degrees'. Default is 'degrees'.
        perturbation_direction (str, optional): The direction of perturbation, can be 'horizontal' or 'vertical'. Default is 'horizontal'.
        use_average_strike (bool, optional): Whether to use the circular mean
            strike-normal direction for all nodes. Default is False.
    
        Returns:
        np.ndarray: Perturbed coordinates.
        """
        coords = coords.copy()
    
        if fixed_nodes is None:
            fixed_nodes = []
    
        movable_nodes = [i for i in range(len(coords)) if i not in fixed_nodes]
    
        # Calculate the perturbation direction
        trends_azi = self.calculate_perturb_direction(coords, angle_unit, use_average_strike, average_direction)
    
        # Convert perturbations to a per-movable-node vector. Scalar values are
        # still accepted for backward compatibility.
        perturbations = np.asarray(perturbations, dtype=float).ravel()
        if len(movable_nodes) == 0:
            return coords
        if perturbations.size == 1:
            perturbations = np.full(len(movable_nodes), perturbations.item(), dtype=float)
        elif perturbations.size != len(movable_nodes):
            raise ValueError(
                "perturbations must be scalar or match the number of movable "
                f"nodes ({len(movable_nodes)}); got {perturbations.size}."
            )
        
        if perturbation_direction == 'horizontal':
            # Apply perturbations to the coordinates in the horizontal direction
            # Calculate direction vectors
            trends = np.pi / 2.0 - trends_azi
            directions = np.array([np.cos(trends), np.sin(trends)]).T
    
            coords[movable_nodes, :2] += directions[movable_nodes] * perturbations[:, None]
        elif perturbation_direction == 'vertical':
            # Apply perturbations to the z-coordinate
            coords[movable_nodes, 2] += perturbations
        else:
            raise ValueError("perturbation_direction must be 'horizontal' or 'vertical'.")
    
        return coords

    @track_mesh_update(description="Perturb top coordinates along a fixed/average direction.",
                       params_info={"perturbations": "1D Array of shifts (km)"})
    def perturb_top_coords_along_fixed_direction(self, perturbations, average_direction=None,
                                         fixed_nodes=None, angle_unit='degrees', perturbation_direction='horizontal',
                                     use_average_strike=False):
        self._require_geometry_ref('top_coords')
        self.top_coords = self.perturb_coords_along_fixed_direction(self.geometry_ref.top_coords, perturbations, average_direction, 
                                                            fixed_nodes, angle_unit, perturbation_direction, use_average_strike)
        self.set_coords(self.top_coords, lonlat=False, coord_type='top')
        return self.top_coords

    @track_mesh_update(description="Perturb bottom coordinates along a fixed/average direction.",
                       params_info={"perturbations": "1D Array of shifts (km)"})
    def perturb_bottom_coords_along_fixed_direction(self, perturbations, average_direction=None,
                                            fixed_nodes=None, angle_unit='degrees', perturbation_direction='horizontal',
                                     use_average_strike=False):
        self._require_geometry_ref('bottom_coords')
        self.bottom_coords = self.perturb_coords_along_fixed_direction(self.geometry_ref.bottom_coords, perturbations, average_direction, 
                                                               fixed_nodes, angle_unit, perturbation_direction, use_average_strike)
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True, 
                       description="Perturb bottom coords along fixed direction and rebuild simple mesh.",
                       params_info={"perturbations": "Array of shifts", "average_direction": "Azimuth (deg/rad)"})
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
        self.densify_edges()
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z, bias, min_dz)
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True,
                       description="Perturb bottom along fixed direction, then deform existing Gmsh mesh.",
                       params_info={"perturbations": "Array of shifts"},
                       bayesian_forbidden={'remap': True})
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
        - perturbations (array-like): Direct perturbation values.
        - top_size (float): Size of the top mesh. Default is 2.0.
        - bottom_size (float): Size of the bottom mesh. Default is 4.0.
        - num_segments (int): Number of segments for discretization. Default is 30.
        - disct_z (int): Number of divisions in the z-direction. Default is 10.
        - projection (str, optional): Projection plane to use ('xy', 'xz', 'yz'). Default is None.
        - rotation_angle (float, optional): Angle to rotate the grid and vertices. Default is None.
        - show (bool): Whether to show the mesh. Default is False.
        - verbose (int): Verbosity level. Default is 0.
        - remap (bool): Whether to remap Gmsh vertices to grid. Default is False.
        - average_direction (float, optional): Average movement direction (azimuth).
        - fixed_nodes (list, optional): A list of indices of fixed nodes.
        - angle_unit (str, optional): Unit of the input angle. Default is 'degrees'.
        - perturbation_direction (str, optional): Direction of perturbation. Default is 'horizontal'.
        - bias (float, optional): Grid bias in the z-direction. Default is None.
        - min_dz (float, optional): Minimum grid spacing in the z-direction. Default is None.
        - use_average_strike (bool, optional): Whether to use the average strike direction. Default is False.
    
        Returns:
        - bottom_coords: Perturbed bottom coordinates.
        """
        # Apply perturbations to bottom coordinates
        self.perturb_bottom_coords_along_fixed_direction(perturbations, average_direction,
                                                         fixed_nodes, angle_unit, perturbation_direction,
                                                         use_average_strike=use_average_strike)

        self.densify_edges()

        # Generate and deform mesh
        self.generate_and_deform_mesh(self.top_coords, self.bottom_coords, 
                                    top_size=top_size, bottom_size=bottom_size, 
                                    num_segments=num_segments, disct_z=disct_z, 
                                    projection=projection, rotation_angle=rotation_angle, 
                                    show=show, verbose=verbose, remap=remap, 
                                    bias=bias, min_dz=min_dz)
        return self.bottom_coords

    @track_mesh_update(description="Perturb a specific middle layer along a fixed direction.",
                       params_info={"perturbations": "Array of shifts", "mid_layer_index": "Index of layer"})
    def perturb_layer_coords_along_fixed_direction(self, perturbations, mid_layer_index=0, average_direction=None, 
                                           fixed_nodes=None, angle_unit='degrees', 
                                           perturbation_direction='horizontal',
                                           use_average_strike=False):
        self._require_geometry_ref('layers')
        assert mid_layer_index < len(self.geometry_ref.layers), f'mid_layer_index {mid_layer_index} out of range for layers_ref (len={len(self.geometry_ref.layers)}).'
        self.layers[mid_layer_index] = self.perturb_coords_along_fixed_direction(self.geometry_ref.layers[mid_layer_index], perturbations, average_direction, 
                                                                         fixed_nodes, angle_unit, perturbation_direction, use_average_strike)
        x, y, z = self.layers[mid_layer_index][:, 0], self.layers[mid_layer_index][:, 1], self.layers[mid_layer_index][:, 2]
        lon, lat = self.xy2ll(x, y)
        self.layers_ll[mid_layer_index] = np.vstack((lon, lat, z)).T
        return self.layers[mid_layer_index]
    
    @track_mesh_update(update_mesh=True, update_laplacian=True, update_area=True, expected_perturbations_count=1,
                       description="Translate the entire fault geometry along a fixed direction.",
                       params_info={"perturbations": "[distance_km]"})
    def perturb_geometry_along_fixed_direction(self, perturbations, average_direction=None, 
                                               fixed_nodes=None, angle_unit='degrees', perturbation_direction='horizontal',
                                               use_average_strike=False):
        """
        Apply random perturbations to the coordinates along a fixed direction.
    
        Parameters:
        perturbations (np.ndarray): Direct perturbation values.
        average_direction (float, optional): The average direction (azimuth) in radians or degrees.
        fixed_nodes (list, optional): A list of indices of fixed nodes.
        angle_unit (str, optional): The unit of the input angles. Default is 'degrees'.
        perturbation_direction (str, optional): The direction of perturbation. Default is 'horizontal'.
        use_average_strike (bool, optional): Whether to use the average strike direction. Default is False.
        """
        # Translate the fault geometry mesh along the fixed direction
        self._ensure_vertices_ref()
        vertices_trans = self.perturb_coords_along_fixed_direction(self.geometry_ref.vertices, perturbations, average_direction, 
                                                                   fixed_nodes, angle_unit, perturbation_direction, use_average_strike)
        self.VertFace2csifault(vertices_trans, self.Faces)
    
        return vertices_trans
    #---------------------------------------------------------------------------------------------------------#
