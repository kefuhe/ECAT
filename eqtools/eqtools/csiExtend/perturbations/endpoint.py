"""
EndpointDuttaPerturbationMixin — Dutta fitting and endpoint perturbation methods.

Extracted from BayesianAdaptiveTriangularPatches for modularity.
All methods are pure move-over with zero logic changes.
"""
import numpy as np
from ..bayesian_perturbation_base import track_mesh_update
from ..AdaptiveTriangularPatches import transform_point, reverse_transform_point


class EndpointDuttaPerturbationMixin:
    """Mixin providing Dutta fitting and endpoint perturbation capabilities."""

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
    
    @track_mesh_update(expected_perturbations_count=2,
                       description="Perturb bottom coords using Dutta logic.",
                       params_info={"perturbations": "[amod1, amod2]"})
    def perturb_bottom_coords_dutta(self, perturbations, fixed_nodes=[0, -1]):
        """
        Perturb bottom coordinates between two specific nodes.
    
        Parameters:
        perturbations (list): A list containing two perturbation values corresponding to amod1 and amod2.
    
        Returns:
        np.ndarray: Perturbed bottom coordinates.
        """
        self._require_geometry_ref('bottom_coords')
        self.bottom_coords = self.perturb_coord_dutta(self.geometry_ref.bottom_coords, fixed_nodes, perturbations)
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True, update_laplacian=False, update_area=False, expected_perturbations_count=4,
                       description="Generate Dutta mesh (Legacy Logic).",
                       params_info={"perturbations": "[D1, D2, S1, S2]", "disct_z": "Discretization in z", "bias": "Bias value", "min_dx": "Minimum dx"})
    def perturb_geometry_dutta(self, perturbations, disct_z=None, bias=None, min_dx=None):
        """
        Generate a mesh for a seismic fault using the Dutta method.
    
        Parameters:
        perturbations (list): Perturbation parameters to control the shape of the seismic fault. [D1, D2, S1, S2]
        disct_z (float, optional): Discretization parameter in the z direction.
        bias (float, optional): Bias parameter for the mesh.
        min_dx (float, optional): Minimum size of the mesh.
        """
        from ..make_mesh_dutta import makemesh as make_mesh_dutta
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

    #----------------------------------------Geometry Operations----------------------------------------------#
    def calculate_bottom_endpoints(self, top_coords=None, bottom_coords=None, depth=None, mode='both'):
        from ..geom_ops import PolygonIntersector
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
    
    @track_mesh_update(update_mesh=False, expected_perturbations_count=4,
                       description="Perturb endpoints and midpoint (Translate only, no mesh update).",
                       params_info={"perturbations": "[dx1, dx2, mid_dx, mid_dy]"})
    def perturb_BottomEndpointsFixedDirAndMidpointTrans(self, perturbations):
        """
        Perturbs the bottom coordinates with limited horizontal movements for the endpoints
        and free movement for the midpoint.
    
        Parameters:
        - perturbations: [dx1, dx2, mid_dx, mid_dy]
        """
        self._require_geometry_ref('bottom_coords', 'top_coords')
        assert self.geometry_ref.bottom_coords.shape[0] == 3, 'bottom_coords_ref must contain 3 points.'
        # Perturb the endpoints
        trends_azi = self.calculate_perturb_direction(
            self.geometry_ref.top_coords)
        average_direction = np.array([trends_azi[0], trends_azi[0], trends_azi[-1]])
        self.perturb_bottom_coords_along_fixed_direction(perturbations[:2], average_direction=average_direction, fixed_nodes=[1], angle_unit='radians',
                                                         perturbation_direction='horizontal', use_average_strike=False)
        # Perturb the midpoint
        self.bottom_coords[1, 0] += perturbations[2]
        self.bottom_coords[1, 1] += perturbations[3]
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords 
    
    @track_mesh_update(update_mesh=True, expected_perturbations_count=4,
                       description="Perturb endpoints/midpoint and rebuild simple mesh.",
                       params_info={"perturbations": "[dx1, dx2, mid_dx, mid_dy]", "disct_z": "Discretization in z", "bias": "Bias value", "min_dz": "Minimum dz"})
    def perturb_BottomEndpointsFixedDirAndMidpointTrans_simpleMesh(self, perturbations, disct_z=None, bias=None, min_dz=None):
        """
        Perturbs the bottom coordinates with limited horizontal movements for the endpoints
        and free movement for the midpoint.
        """
        self.perturb_BottomEndpointsFixedDirAndMidpointTrans(perturbations)

        self.densify_edges()

        # Generate geometry mesh
        self.discretize_bottom_coords(num_segments=self.top_coords.shape[0])
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z=disct_z, bias=bias, min_dz=min_dz)
        return 
    
    @track_mesh_update(update_mesh=False, expected_perturbations_count=4,
                       description="Dutta Perturbation on bottom coords (No mesh update).",
                       params_info={"perturbations": "[amod1, amod2, dx_end1, dx_end2]"})
    def perturb_BottomEndpointsFixedDirandMidpointsDutta(self, perturbations):
        self._require_geometry_ref('bottom_coords', 'top_coords')
        assert self.geometry_ref.bottom_coords.shape[0] == self.geometry_ref.top_coords.shape[0], 'The number of top and bottom coordinates must be the same.'
        # Perturb middle points first using Dutta method
        self.perturb_bottom_coords_dutta(perturbations[:2])
    
        # Then perturb the endpoints along the fixed direction
        average_direction = self.calculate_perturb_direction(
            self.geometry_ref.top_coords)

        self.bottom_coords = self.perturb_coords_along_fixed_direction(self.bottom_coords, perturbations[2:],
                                                         average_direction=average_direction, fixed_nodes=np.arange(self.bottom_coords.shape[0])[1:-1], 
                                                         angle_unit='radians', perturbation_direction='horizontal', use_average_strike=False)
    
        # Calculate reference and current vectors
        vector_ref = self.geometry_ref.bottom_coords[-1] - self.geometry_ref.bottom_coords[0]
        vector = self.bottom_coords[-1] - self.bottom_coords[0]
        
        # Calculate angle and norm differences
        angle_diff = np.angle(vector[0] + 1j * vector[1]) - np.angle(vector_ref[0] + 1j * vector_ref[1])
        norm_diff = np.linalg.norm(vector) / np.linalg.norm(vector_ref)
        
        # Convert coordinates to complex for manipulation
        bottom_coords_complex = self.bottom_coords[:, 0] + 1.j * self.bottom_coords[:, 1]
        bottom_ref_st_complex = self.geometry_ref.bottom_coords[0, 0] + 1.j * self.geometry_ref.bottom_coords[0, 1]
        
        # Apply rotation, scaling, and translation adjustments
        bottom_coords_complex[1:-1] = bottom_coords_complex[0] + norm_diff * np.exp(1j * angle_diff) * (bottom_coords_complex[1:-1] - bottom_ref_st_complex)
        
        # Convert back to real coordinates
        self.bottom_coords[:, :2] = np.column_stack([bottom_coords_complex.real, bottom_coords_complex.imag])
        
        self.set_coords(self.bottom_coords, lonlat=False, coord_type='bottom')
        return self.bottom_coords
    
    @track_mesh_update(update_mesh=True, expected_perturbations_count=4,
                       description="Standard Dutta Perturbation with Simple Mesh generation.",
                       params_info={"perturbations": "[amod1, amod2, dx_end1, dx_end2]"})
    def perturb_BottomEndpointsFixedDirandMidpointsDutta_simpleMesh(self, perturbations, disct_z=None, bias=None, min_dz=None):
        """
        Perturbs the bottom coordinates with limited horizontal movements for the endpoints
        and free movement for the midpoint using the Dutta method, while maintaining the original shape structure.
        """
        self.perturb_BottomEndpointsFixedDirandMidpointsDutta(perturbations)
        self.densify_edges()
        # Generate geometry mesh
        if self.top_coords.shape[0] != self.bottom_coords.shape[0]:
            self.discretize_bottom_coords(num_segments=self.top_coords.shape[0])
        self.generate_simple_mesh(self.top_coords, self.bottom_coords, disct_z=disct_z, bias=bias, min_dz=min_dz)
        return 
    #----------------------------------------------------------------------------------------------------------#
