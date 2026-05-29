"""
TranslationPerturbationMixin — Translation-based perturbation methods.

Extracted from BayesianAdaptiveTriangularPatches for modularity.
All methods are pure move-over with zero logic changes.
"""
import numpy as np
from ..bayesian_perturbation_base import track_mesh_update


class TranslationPerturbationMixin:
    """Mixin providing translation perturbation capabilities."""

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
    
    @track_mesh_update(expected_perturbations_count=2,
                       description="Translate top coordinates.",
                       params_info={"perturbations": "[dx, dy]"})
    def perturb_top_coords_by_translation(self, perturbations):
        """
        Apply a global translation to the top coordinates.
    
        Parameters:
        perturbations (tuple): Translation distances in the format `(dx, dy)`.
    
        Returns:
        np.ndarray: Translated top coordinates.
        """
        self._require_geometry_ref('top_coords')
        self.top_coords = self.perturb_coords_by_translation(self.geometry_ref.top_coords, perturbations)
        self.set_coords(self.top_coords, lonlat=False, coord_type='top')
        return self.top_coords

    @track_mesh_update(expected_perturbations_count=2,
                       description="Translate bottom coordinates.",
                       params_info={"perturbations": "[dx, dy]"})
    def perturb_bottom_coords_by_translation(self, perturbations):
        """
        Apply a global translation to the bottom coordinates.

        Parameters:
        perturbations (tuple): Translation distances in the format `(dx, dy)`.

        Returns:
        np.ndarray: Translated bottom coordinates.
        """
        self._require_geometry_ref('bottom_coords')
        self.bottom_coords = self.perturb_coords_by_translation(self.geometry_ref.bottom_coords, perturbations)
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True, expected_perturbations_count=2,
                       description="Translate bottom coordinates and rebuild simple mesh.",
                       params_info={"perturbations": "[dx, dy]", "disct_z": "Discretization in z", "bias": "Bias value", "min_dz": "Minimum dz"})
    def perturb_BottomTrans_simpleMesh(self, perturbations, disct_z=None, bias=None, min_dz=None):
        """
        Apply a global translation to the bottom coordinates and update the simple mesh.
    
        Parameters:
        perturbations (tuple): Translation distances in the format `(dx, dy)`.
    
        Returns:
        np.ndarray: Translated bottom coordinates.
        """
        self.perturb_bottom_coords_by_translation(perturbations)
        self.densify_edges()
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z, bias, min_dz)
        return self.bottom_coords
    
    @track_mesh_update(expected_perturbations_count=2,
                       description="Translate a specific layer.",
                       params_info={"perturbations": "[dx, dy]", "mid_layer_index": "int"})
    def perturb_layer_coords_by_translation(self, perturbations, mid_layer_index=0):
        """
        Apply a global translation to the coordinates of a specific layer.
    
        Parameters:
        perturbations (tuple): Translation distances in the format `(dx, dy)`.
        mid_layer_index (int): Index of the layer to be translated.
    
        Returns:
        np.ndarray: Translated coordinates of the specified layer.
        """
        self._require_geometry_ref('layers')
        assert mid_layer_index < len(self.geometry_ref.layers), f'mid_layer_index {mid_layer_index} out of range for layers_ref (len={len(self.geometry_ref.layers)}).'
        self.layers[mid_layer_index] = self.perturb_coords_by_translation(self.geometry_ref.layers[mid_layer_index], perturbations)
        x, y, z = self.layers[mid_layer_index][:, 0], self.layers[mid_layer_index][:, 1], self.layers[mid_layer_index][:, 2]
        lon, lat = self.xy2ll(x, y)
        self.layers_ll[mid_layer_index] = np.vstack((lon, lat, z)).T
        return self.layers[mid_layer_index]
    
    @track_mesh_update(update_mesh=True, update_laplacian=True, update_area=True, expected_perturbations_count=2,
                       description="Translate the entire fault geometry.",
                       params_info={"perturbations": "[dx, dy]"})
    def perturb_geometry_by_translation(self, perturbations):
        """
        Apply a global translation to all coordinates.
    
        Parameters:
        perturbations (tuple): Translation distances in the format `(dx, dy)`.
    
        Returns:
        np.ndarray: Translated vertices.
        """
        # Translate the fault geometry mesh
        self._ensure_vertices_ref()
        vertices_trans = self.perturb_coords_by_translation(self.geometry_ref.vertices, perturbations)
        self.VertFace2csifault(vertices_trans, self.Faces)
    
        return vertices_trans
    #---------------------------------------------------------------------------------------------------------------#
