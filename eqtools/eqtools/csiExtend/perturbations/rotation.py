"""
RotationPerturbationMixin — Rotation-based perturbation methods.

Extracted from BayesianAdaptiveTriangularPatches for modularity.
All methods are pure move-over with zero logic changes.
"""
import numpy as np
from scipy.spatial import cKDTree
from ..bayesian_perturbation_base import track_mesh_update
from .angle_utils import angles_to_radians, normalize_angle_unit


class RotationPerturbationMixin:
    """Mixin providing rotation perturbation capabilities."""

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
        recalculate_pivot (bool, optional): Whether to recalculate the pivot point. Default is False.
        is_utm (bool, optional): If pivot is a coordinate point, this parameter determines whether the coordinate point is in UTM coordinates. Default is False.
        angle_unit (str, optional): The unit of the angle. Can be 'degrees' or 'radians'. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot to be one of the points in coords. Default is False.
    
        Returns:
        array-like: The perturbed coordinates.
        """
        if recalculate_pivot or not hasattr(self, 'pivot'):
            # Determine the center point of the perturbation
            pivot = self.calculate_pivot(pivot, coords, is_utm, force_pivot_in_coords)
        else:
            pivot = self.pivot
    
        # Convert the perturbation angle to radians. The helper accepts scalar
        # and length-1 inputs and validates angle_unit consistently.
        normalize_angle_unit(angle_unit)
        perturbations = np.asarray(angles_to_radians(perturbations, angle_unit)).item()
    
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

    @track_mesh_update(expected_perturbations_count=1,
                       description="Rotate top coordinates around a pivot.",
                       params_info={"perturbations": "Rotation angle", "pivot": "'start'/'end'/'midpoint'"})
    def perturb_top_coords_by_rotation(self, perturbations, pivot='midpoint', is_utm=False, recalculate_pivot=False,
                                        angle_unit='degrees', force_pivot_in_coords=False):
        self._require_geometry_ref('top_coords')
        self.top_coords = self.perturb_coords_by_rotation(self.geometry_ref.top_coords, perturbations, pivot, recalculate_pivot,
                                                           is_utm, angle_unit, force_pivot_in_coords)
        self.set_coords(self.top_coords, lonlat=False, coord_type='top')
        return self.top_coords

    @track_mesh_update(expected_perturbations_count=1,
                       description="Rotate bottom coordinates around a pivot.",
                       params_info={"perturbations": "Rotation angle (degrees/radians)", "pivot": "'start'/'end'/'midpoint'"})
    def perturb_bottom_coords_by_rotation(self, perturbations, pivot='midpoint', is_utm=False, recalculate_pivot=False,
                                           angle_unit='degrees', force_pivot_in_coords=False):
        self._require_geometry_ref('bottom_coords')
        self.bottom_coords = self.perturb_coords_by_rotation(self.geometry_ref.bottom_coords, perturbations, pivot, recalculate_pivot,
                                                              is_utm, angle_unit, force_pivot_in_coords)
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True, expected_perturbations_count=1,
                       description="Rotate bottom coordinates and rebuild simple mesh.",
                       params_info={"perturbations": "Rotation angle (degrees/radians)", "pivot": "'start'/'end'/'midpoint'"})
    def perturb_BottomRotation_simpleMesh(self, perturbations, pivot='midpoint', is_utm=False, recalculate_pivot=False,
                                          angle_unit='degrees', force_pivot_in_coords=False,
                                          disct_z=None, bias=None, min_dz=None):
        """
        Apply perturbations to the bottom coordinates and update the simple mesh.
    
        Parameters:
        perturbations (float): The angle of perturbation. Positive for counterclockwise, negative for clockwise.
        pivot (str or tuple, optional): The center point of the perturbation. Default is 'midpoint'.
        is_utm (bool, optional): Whether the coordinate point is in UTM coordinates. Default is False.
        recalculate_pivot (bool, optional): Whether to recalculate the pivot point. Default is False.
        angle_unit (str, optional): The unit of the angle. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot to be one of the points in coords. Default is False.
        disct_z (float, optional): Mesh density in the z direction. Default is None.
        bias (float, optional): Mesh bias in the z direction. Default is None.
        min_dz (float, optional): Minimum mesh spacing in the z direction. Default is None.
    
        Returns:
        np.ndarray: The perturbed bottom coordinates.
        """
        self.perturb_bottom_coords_by_rotation(perturbations, pivot, is_utm, recalculate_pivot, angle_unit, force_pivot_in_coords)
        self.densify_edges()
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z, bias, min_dz)
        return self.bottom_coords

    @track_mesh_update(expected_perturbations_count=1,
                       description="Rotate a specific layer.",
                       params_info={"perturbations": "Angle", "mid_layer_index": "Index"})
    def perturb_layer_coords_by_rotation(self, perturbations, mid_layer_index=0, pivot='midpoint', recalculate_pivot=False,
                                         is_utm=False, angle_unit='degrees', force_pivot_in_coords=False):
        """
        Apply perturbations to the coordinates of a specific layer by rotation.
    
        Parameters:
        perturbations (float): The angle of perturbation. Positive for counterclockwise, negative for clockwise.
        mid_layer_index (int): The index of the middle layer to be perturbed. Default is 0.
        pivot (str or tuple, optional): The center point of the perturbation. Default is 'midpoint'.
        recalculate_pivot (bool, optional): Whether to recalculate the pivot point. Default is False.
        is_utm (bool, optional): Whether the coordinate point is in UTM coordinates. Default is False.
        angle_unit (str, optional): The unit of the angle. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot to be one of the points in coords. Default is False.
    
        Returns:
        np.ndarray: The perturbed coordinates of the specified layer.
        """
        self._require_geometry_ref('layers')
        assert mid_layer_index < len(self.geometry_ref.layers), f'mid_layer_index {mid_layer_index} out of range for layers_ref (len={len(self.geometry_ref.layers)}).'
        self.layers[mid_layer_index] = self.perturb_coords_by_rotation(self.geometry_ref.layers[mid_layer_index], perturbations, 
                                                                       pivot, recalculate_pivot, is_utm, angle_unit, force_pivot_in_coords)
        x, y, z = self.layers[mid_layer_index][:, 0], self.layers[mid_layer_index][:, 1], self.layers[mid_layer_index][:, 2]
        lon, lat = self.xy2ll(x, y)
        self.layers_ll[mid_layer_index] = np.vstack((lon, lat, z)).T
        return self.layers[mid_layer_index]
    
    @track_mesh_update(update_mesh=True, update_laplacian=True, update_area=True, expected_perturbations_count=1,
                       description="Rotate the entire fault geometry.",
                       params_info={"perturbations": "Angle", "pivot": "'start'/'end'/'midpoint'"})
    def perturb_geometry_by_rotation(self, perturbations, pivot='midpoint', recalculate_pivot=False,
                                     angle_unit='degrees', force_pivot_in_coords=False, is_utm=True):
        """
        Apply perturbations to all coordinates by rotation.

        Parameters:
        perturbations (float): The angle of perturbation. Positive for counterclockwise, negative for clockwise.
        pivot (str or tuple, optional): The center point of the perturbation. Default is 'midpoint'.
        recalculate_pivot (bool, optional): Whether to recalculate the pivot point. Default is False.
        angle_unit (str, optional): The unit of the angle. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot to be one of the points in coords. Default is False.
        is_utm (bool, optional): Whether the pivot coordinate (if a tuple) is in UTM. Default is True
            because this method operates on mesh vertices (UTM). Set False if providing a lon/lat pivot.

        Returns:
        np.ndarray: The perturbed coordinates.
        """
        self._require_geometry_ref('top_coords')
        if recalculate_pivot or not hasattr(self, 'pivot'):
            pivot = self.calculate_pivot(pivot, self.geometry_ref.top_coords, is_utm, force_pivot_in_coords)

        # Rotate the fault geometry mesh
        self._ensure_vertices_ref()
        vertices_rot = self.perturb_coords_by_rotation(self.geometry_ref.vertices, perturbations, pivot, False, is_utm, angle_unit)
        self.VertFace2csifault(vertices_rot, self.Faces)
    
        return vertices_rot
    #------------------------------------------------------------------------------------------------------------#
