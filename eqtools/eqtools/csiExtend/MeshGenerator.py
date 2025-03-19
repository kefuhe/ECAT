import numpy as np
from typing import List, Optional, Tuple
from scipy.interpolate import interp1d, griddata
from scipy.integrate import cumulative_trapezoid as cumtrapz
import gmsh
import sys

# import self-defined libraries
from .make_mesh_dutta import makemesh as make_mesh_dutta
from .geom_ops import discretize_coords

import numpy as np
from scipy.optimize import fsolve
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0

def calculate_average_direction(points):
    """
    Calculate the average direction of a piecewise linear curve using PCA.

    Parameters:
    -----------
    points : np.ndarray
        Array of points representing the piecewise linear curve. Shape should be (N, 2) or (N, 3).

    Returns:
    --------
    avg_direction : np.ndarray
        The average direction vector.
    """
    # Ensure points are in 2D or 3D
    if points.shape[1] not in [2, 3]:
        raise ValueError("Points should be in 2D or 3D space.")

    # Use only the first two dimensions (x and y)
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(points[:, :2])

    # Get the direction vector from the first principal component
    direction = pca.components_[0]

    # Normalize the direction vector
    avg_direction = direction / np.linalg.norm(direction)

    return avg_direction

def calculate_optimal_projection_and_indices(grid_points, mode='total_area'):
    """
    Calculate the optimal projection plane and corresponding indices for each grid cell.

    Parameters:
    - grid_points (np.ndarray): Array of grid points.
    - mode (str): Projection mode. 'total_area' for comparing total projected area, 'cell_area' for comparing each cell's projected area. Default is 'total_area'.

    Returns:
    - projections (np.ndarray): Array of optimal projection planes for each grid cell.
    - indices (np.ndarray): Array of corresponding indices for each projection plane.
    """
    projections = np.empty((grid_points.shape[0] - 1, grid_points.shape[1] - 1), dtype=object)
    indices = np.empty((grid_points.shape[0] - 1, grid_points.shape[1] - 1, 2), dtype=int)

    if mode == 'total_area':
        # Calculate total projected area for each plane
        total_area_xy = 0
        total_area_xz = 0
        total_area_yz = 0

        for k in range(grid_points.shape[0] - 1):
            for i in range(grid_points.shape[1] - 1):
                points = [
                    (grid_points[k, i, 0], grid_points[k, i, 1], grid_points[k, i, 2]),
                    (grid_points[k, i+1, 0], grid_points[k, i+1, 1], grid_points[k, i+1, 2]),
                    (grid_points[k+1, i+1, 0], grid_points[k+1, i+1, 1], grid_points[k+1, i+1, 2]),
                    (grid_points[k+1, i, 0], grid_points[k+1, i, 1], grid_points[k+1, i, 2])
                ]
                total_area_xy += calculate_area(points, 'xy')
                total_area_xz += calculate_area(points, 'xz')
                total_area_yz += calculate_area(points, 'yz')

        # Determine the optimal projection plane based on total area
        if total_area_xy >= total_area_xz and total_area_xy >= total_area_yz:
            optimal_projection = 'xy'
            optimal_indices = [0, 1]
        elif total_area_xz >= total_area_xy and total_area_xz >= total_area_yz:
            optimal_projection = 'xz'
            optimal_indices = [0, 2]
        else:
            optimal_projection = 'yz'
            optimal_indices = [1, 2]

        # Assign the optimal projection plane and indices to all cells
        projections.fill(optimal_projection)
        indices[:] = optimal_indices

    elif mode == 'cell_area':
        # Calculate the optimal projection plane for each cell based on its projected area
        for k in range(grid_points.shape[0] - 1):
            for i in range(grid_points.shape[1] - 1):
                points = [
                    (grid_points[k, i, 0], grid_points[k, i, 1], grid_points[k, i, 2]),
                    (grid_points[k, i+1, 0], grid_points[k, i+1, 1], grid_points[k, i+1, 2]),
                    (grid_points[k+1, i+1, 0], grid_points[k+1, i+1, 1], grid_points[k+1, i+1, 2]),
                    (grid_points[k+1, i, 0], grid_points[k+1, i, 1], grid_points[k+1, i, 2])
                ]
                area_xy = calculate_area(points, 'xy')
                area_xz = calculate_area(points, 'xz')
                area_yz = calculate_area(points, 'yz')

                if area_xy >= area_xz and area_xy >= area_yz:
                    projections[k, i] = 'xy'
                    indices[k, i] = [0, 1]
                elif area_xz >= area_xy and area_xz >= area_yz:
                    projections[k, i] = 'xz'
                    indices[k, i] = [0, 2]
                else:
                    projections[k, i] = 'yz'
                    indices[k, i] = [1, 2]

    else:
        raise ValueError("Invalid mode specified. Choose from 'total_area' or 'cell_area'.")

    return projections, indices

def calculate_area(points, plane):
    """
    Calculate the area of the quadrilateral projected onto a specified plane.
    
    Parameters:
    - points: List of four tuples [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
                representing the corners of the quadrilateral.
    - plane: The plane onto which to project ('xy', 'xz', 'yz').
    
    Returns:
    - area: The area of the projected quadrilateral.
    """
    if plane not in {'xy', 'xz', 'yz'}:
        raise ValueError("Invalid plane specified. Choose from 'xy', 'xz', 'yz'.")

    indices = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}[plane]
    i1, i2 = indices

    x1, y1 = points[0][i1], points[0][i2]
    x2, y2 = points[1][i1], points[1][i2]
    x3, y3 = points[2][i1], points[2][i2]
    x4, y4 = points[3][i1], points[3][i2]

    return 0.5 * abs(x1*y2 + x2*y3 + x3*y4 + x4*y1 - y1*x2 - y2*x3 - y3*x4 - y4*x1)

def bilinear_interpolation(u, v, points):
    """
    Perform bilinear interpolation for a point (u, v) given four corner points.
    
    Parameters:
    - u, v: Parameter coordinates of the target point.
    - points: List of four tuples [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
              representing the corners of the quadrilateral.
    
    Returns:
    - Interpolated (x, y, z) value at (u, v).
    """
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4) = points
    
    one_minus_u = 1 - u
    one_minus_v = 1 - v
    
    w1 = one_minus_u * one_minus_v
    w2 = u * one_minus_v
    w3 = u * v
    w4 = one_minus_u * v
    
    x = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4
    y = w1 * y1 + w2 * y2 + w3 * y3 + w4 * y4
    z = w1 * z1 + w2 * z2 + w3 * z3 + w4 * z4
    
    return x, y, z

from scipy.optimize import fsolve, least_squares
import numpy as np
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt

def inverse_bilinear_interpolation(x, y, points, tolerance=1e-6):
    """
    Calculate the parameter coordinates (u, v) for a point (x, y) within a quadrilateral using inverse bilinear interpolation.
    
    Parameters:
    - x, y: Coordinates of the target point.
    - points: List of four tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
              representing the corners of the quadrilateral.
    - tolerance: Tolerance for point containment check. Default is 1e-6.
    
    Returns:
    - (u, v): Parameter coordinates of the target point.
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = points

    # Check if the point is within the quadrilateral
    polygon = Polygon(points)
    point = Point(x, y)
    if not polygon.contains(point) and not polygon.touches(point):
        # Find the closest point on the polygon boundary
        line = LineString(points + [points[0]])  # Close the polygon
        nearest_point = line.interpolate(line.project(point))
        # print(f"Point ({x}, {y}) is outside the quadrilateral. Using the nearest point on the boundary: ({nearest_point.x}, {nearest_point.y})")
        x, y = nearest_point.x, nearest_point.y

        # Offset the point slightly towards the center of the polygon
        # centroid = polygon.centroid
        # direction = np.array([centroid.x - x, centroid.y - y])
        # direction /= np.linalg.norm(direction)
        # x += direction[0] * tolerance
        # y += direction[1] * tolerance
        # print(f"Offsetting the point to ({x}, {y})")

    def equations(p):
        u, v = p
        x_interp = (1 - u) * (1 - v) * x1 + u * (1 - v) * x2 + u * v * x3 + (1 - u) * v * x4
        y_interp = (1 - u) * (1 - v) * y1 + u * (1 - v) * y2 + u * v * y3 + (1 - u) * v * y4
        return [x_interp - x, y_interp - y]
    
    initial_guesses = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25)]
    
    # Using fsolve first
    for initial_guess in initial_guesses:
        try:
            (u, v), infodict, ier, msg = fsolve(equations, initial_guess, full_output=True, xtol=1e-6, maxfev=1000)
            if ier == 1:
                return u, v
        except Exception as e:
            pass

    # Using least_squares as a fallback
    for initial_guess in initial_guesses:
        result = least_squares(equations, initial_guess, xtol=1e-6, ftol=1e-6, max_nfev=1000)
        if result.success:
            return result.x

    raise RuntimeError("Optimization did not converge with any initial guess.")


class MeshGenerator:
    def __init__(self, top_coords: Optional[np.ndarray] = None, bottom_coords: Optional[np.ndarray] = None):
        self.top_coords = top_coords
        self.bottom_coords = bottom_coords
        self.top_size = None
        self.bottom_size = None
        self.param_coords = None # Parameter coordinates for mapping Gmsh vertices to grid

    def set_coordinates(self, top_coords: np.ndarray, bottom_coords: np.ndarray):
        """
        Set the top and bottom coordinates for the mesh generator.

        Parameters:
        - top_coords (np.ndarray): The top coordinates of the fault.
        - bottom_coords (np.ndarray): The bottom coordinates of the fault.
        """
        self.top_coords = top_coords
        self.bottom_coords = bottom_coords

    def set_top_bottom_size(self, top_size, bottom_size=None):
        """
        Sets the size of the top and bottom.

        Parameters:
        top_size (float): The size at the top.
        bottom_size (float, optional): The size at the bottom. If not provided, the top size is used.

        Note:
        This method adjusts the mesh size at the top and bottom of the model, allowing for different resolutions at these boundaries. The sizes influence the mesh generation process, determining the granularity of the mesh at the top and bottom.
        """

        self.top_size = top_size
        self.bottom_size = bottom_size if bottom_size is not None else top_size
    
    def discretize_coords(self, coords, every=None, num_segments=None, threshold=2):
        '''
        Discretize iso-depth nodes coordinates depicting the rupture trace.
    
        Parameters:
        - coords (np.ndarray): The coordinates of the iso-depth nodes.
        - every (float, optional): The interval at which to discretize the coordinates. If provided, overrides num_segments.
        - num_segments (int, optional): The number of segments to discretize the coordinates into. Ignored if every is provided.
        - threshold (float, optional): The threshold distance to check the first and last vertex against the nearest r_new point. Default is 2.
    
        Returns:
        - xyz_new (np.ndarray): The new discretized coordinates in the original coordinate system.
        - lonlatz_new (np.ndarray): The new discretized coordinates in the longitude/latitude coordinate system.
        '''
        xyz_new = discretize_coords(coords, every, num_segments, threshold)
    
        return xyz_new
    
    def discretize_top_coords(self, every=None, num_segments=None, threshold=2):
        xyz_new = self.discretize_coords(self.top_coords, every, num_segments, threshold)
        self.top_coords = xyz_new
    
    def discretize_bottom_coords(self, every=None, num_segments=None, threshold=2):
        xyz_new = self.discretize_coords(self.bottom_coords, every, num_segments, threshold)
        self.bottom_coords = xyz_new

    def generate_simple_mesh(self, disct_z: Optional[int] = None, bias: Optional[float] = None, 
                             min_dz: Optional[float] = None, use_depth_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a simple earthquake fault mesh from top to bottom coordinates.
    
        Parameters:
        - disct_z (Optional[int]): Discretization parameter in the z-direction. If provided, it overrides bias and min_dz.
        - bias (Optional[float]): Bias parameter for the mesh. Required if disct_z is None.
        - min_dz (Optional[float]): Minimum size of the mesh in the z-direction. Required if disct_z is None.
        - use_depth_only (bool): If True, the full length is the mean depth; otherwise, it is the mean of the entire length. Default is True.
    
        Returns:
        - Tuple[np.ndarray, np.ndarray]: The vertices and faces of the generated mesh.
    
        Notes:
        - If disct_z is provided, it takes precedence over bias and min_dz.
        - If disct_z is None, both bias and min_dz must be provided.
        """
        mesh_coords = self.generate_grid_coordinates(disct_z, bias, min_dz, use_depth_only)
        vertices, faces = self._generate_mesh_faces(mesh_coords)
        return vertices, faces

    def generate_multilayer_mesh(self, layers_coords: List[np.ndarray], disct_z: int, bias: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a multi-layer earthquake fault mesh.

        Parameters:
        - layers_coords (List[np.ndarray]): The coordinates of the intermediate layers, each shaped (n, d).
        - disct_z (int): Discretization parameter in the z-direction, representing the number of segments.
        - bias (float, optional): Bias used to adjust the length of each segment. Default is 1.0.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: The vertices and faces of the generated mesh.

        Notes:
        - The number of points in top_coords, layers_coords, and bottom_coords must be the same.
        """
        assert self.top_coords is not None and self.bottom_coords is not None, 'Top and bottom coordinates must be set.'
        assert self.top_coords.shape[0] == self.bottom_coords.shape[0], 'The number of top and bottom coordinates must be the same.'
        assert len(layers_coords) > 0, 'The number of layers must be greater than 0.'
        assert all([self.top_coords.shape[0] == layer_coords.shape[0] for layer_coords in layers_coords]), 'The number of top and layer coordinates must be the same.'

        # Use split_coords_into_segments function to split all coordinates into equal segments
        mesh_coords = self._split_coords_into_segments(self.top_coords, layers_coords, self.bottom_coords, disct_z + 1, bias=bias)

        # Prepare the data for return
        vertices, faces = self._generate_mesh_faces(mesh_coords)
        return vertices, faces

    def generate_mesh_dutta(self, depth, perturbations, top_coords, disct_z=None, bias=None, min_dx=None):
        """
        Generate a mesh for a seismic fault using the Dutta method.

        Parameters:
        depth: Depth of the seismic fault.
        perturbations: Perturbation parameters to control the shape of the seismic fault. [D1, D2, S1, S2]
        top_coords: Top coordinates.
        disct_z: Discretization parameter in the z direction. If None, the default value will be used.
        bias: Bias parameter for the mesh. If None, the default value will be used.
        min_dx: Minimum size of the mesh. If None, the default value will be used.

        Returns:
        vertices: Generated vertex coordinates.
        faces: Generated face indices.
        """
        top_coords = top_coords.copy()
        top_coords[:, 2] = -top_coords[:, 2]
        p, q, r, trired, ang, xfault, yfault, zfault = make_mesh_dutta(depth, 
                                                                       perturbations, 
                                                                       top_coords, 
                                                                       disct_z, 
                                                                       bias, 
                                                                       min_dx)
        vertices = np.vstack((p.flatten(), q.flatten(), -r.flatten())).T
        return vertices, trired

    def generate_grid_coordinates(self, disct_z: Optional[int] = None, bias: Optional[float] = None, 
                                  min_dz: Optional[float] = None, use_depth_only: bool = True,
                                  top_coords: Optional[np.ndarray] = None, bottom_coords: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate grid coordinates from top to bottom coordinates.
    
        Parameters:
        - disct_z (Optional[int]): Discretization parameter in the z-direction. If provided, it overrides bias and min_dz.
        - bias (Optional[float]): Bias parameter for the mesh. Required if disct_z is None.
        - min_dz (Optional[float]): Minimum size of the mesh in the z-direction. Required if disct_z is None.
        - use_depth_only (bool): If True, the full length is the mean depth; otherwise, it is the mean of the entire length. Default is True.
        - top_coords (Optional[np.ndarray]): The coordinates of the top surface. If None, use self.top_coords.
        - bottom_coords (Optional[np.ndarray]): The coordinates of the bottom surface. If None, use self.bottom_coords.
    
        Returns:
        - np.ndarray: The mesh coordinates.
    
        Notes:
        - If disct_z is provided, it takes precedence over bias and min_dz.
        - If disct_z is None, both bias and min_dz must be provided.
        """
        if top_coords is None:
            top_coords = self.top_coords
        if bottom_coords is None:
            bottom_coords = self.bottom_coords
    
        assert top_coords is not None and bottom_coords is not None, 'Top and bottom coordinates must be set.'
        assert top_coords.shape[0] == bottom_coords.shape[0], 'The number of top and bottom coordinates must be the same.'
    
        # Calculate the difference between top and bottom coordinates
        diff = bottom_coords - top_coords
        fulllen = np.abs(diff[:, 2]).mean() if use_depth_only else np.linalg.norm(diff, axis=1).mean()
    
        # Calculate disct_z if not provided
        if disct_z is None:
            if min_dz is None or bias is None:
                raise ValueError("If disct_z is None, both min_dz and bias must be provided.")
            # Sn = min_dz * (bias**n - 1)/(bias - 1)
            disct_z = int(np.log(1 + np.abs(fulllen) / min_dz * (bias - 1)) / np.log(bias))
            
            # Initialize mesh coordinates
            mesh_coords = np.zeros((disct_z + 1, *top_coords.shape))
            for i in range(disct_z):
                Sn = min_dz * (bias**i - 1) / (bias - 1)
                mesh_coords[i] = top_coords + diff * Sn / fulllen
            mesh_coords[-1] = bottom_coords
        else:
            bias = 1.0
            # Initialize mesh coordinates
            mesh_coords = np.zeros((disct_z+1, *top_coords.shape))
            for i in range(disct_z+1):
                mesh_coords[i] = top_coords + diff * i / disct_z
    
        return mesh_coords
    
    def map_gmsh_vertices_to_grid(self, gmsh_vertices: np.ndarray, grid_points: np.ndarray, 
                                  rotation_angle: float = None, tolerance: float = 1e-6, projection: str = None, 
                                  debug_plot: bool = False, auto_rotate: bool = True) -> None:
        """
        Maps Gmsh vertices to a grid and calculates parameter coordinates.
        
        Parameters:
        - gmsh_vertices (np.ndarray): Array of Gmsh vertices.
        - grid_points (np.ndarray): Array of grid points.
        - rotation_angle (float): Angle to rotate the grid and vertices. Default is None.
        - tolerance (float): Tolerance for point containment check. Default is 1e-6.
        - projection (str): Projection plane to use ('xy', 'xz', 'yz'). Default is None, which means the plane with the largest area will be chosen.
        - debug_plot (bool): Whether to plot a 3D debug plot if a vertex is not found. Default is False.
        - auto_rotate (bool): Whether to automatically rotate using the average direction of self.top_coords. Default is True.
        
        Raises:
        - ValueError: If a vertex is not within any grid cell.
        """
        param_coords = []
    
        # Apply rotation if rotation_angle is provided or auto_rotate is True
        if rotation_angle is not None:
            rotation_angle_rad = np.radians(rotation_angle)
        elif auto_rotate and hasattr(self, 'top_coords'):
            avg_direction = calculate_average_direction(self.top_coords)
            rotation_angle_rad = -np.arctan2(avg_direction[1], avg_direction[0])
            rotation_angle = np.degrees(rotation_angle_rad)
            
            # Ensure the rotation angle is between -90 and 90 degrees
            if rotation_angle > 90:
                rotation_angle -= 180
            elif rotation_angle < -90:
                rotation_angle += 180
        
            if rank == 0:
                print(f"Auto-rotating by {rotation_angle:.2f} degrees.")
        else:
            rotation_angle = None
        
        if rotation_angle is not None:
            rotation_complex = np.exp(1j * np.radians(rotation_angle))
            
            # Rotate gmsh_vertices
            gmsh_vertices_complex = gmsh_vertices[:, 0] + 1j * gmsh_vertices[:, 1]
            gmsh_vertices_complex *= rotation_complex
            gmsh_vertices[:, 0] = gmsh_vertices_complex.real
            gmsh_vertices[:, 1] = gmsh_vertices_complex.imag
        
            # Rotate grid_points
            grid_points_complex = grid_points[:, :, 0] + 1j * grid_points[:, :, 1]
            grid_points_complex *= rotation_complex
            grid_points[:, :, 0] = grid_points_complex.real
            grid_points[:, :, 1] = grid_points_complex.imag
    
        # Determine projection and indices
        if projection in {'xy', 'xz', 'yz'}:
            if projection == 'xy':
                idx1, idx2 = 0, 1
            elif projection == 'xz':
                idx1, idx2 = 0, 2
            else:  # projection == 'yz'
                idx1, idx2 = 1, 2
            projections = np.full((grid_points.shape[0] - 1, grid_points.shape[1] - 1), projection)
            indices = np.full((grid_points.shape[0] - 1, grid_points.shape[1] - 1, 2), [idx1, idx2])
        else:
            projections, indices = calculate_optimal_projection_and_indices(grid_points, mode='total_area')

        for vertex in gmsh_vertices:
            x, y, z = vertex
    
            # Find the closest grid points in the z-direction
            k = np.searchsorted(grid_points[:, 0, 2], z) - 1
            k = max(min(k, grid_points.shape[0] - 2), 0)
    
            found = False
            for i in range(grid_points.shape[1] - 1):
                projection = projections[k, i]
                idx1, idx2 = indices[k, i]
                point = Point(vertex[idx1], vertex[idx2])
                
                polygon = Polygon([
                    (grid_points[k, i, idx1], grid_points[k, i, idx2]),
                    (grid_points[k, i+1, idx1], grid_points[k, i+1, idx2]),
                    (grid_points[k+1, i+1, idx1], grid_points[k+1, i+1, idx2]),
                    (grid_points[k+1, i, idx1], grid_points[k+1, i, idx2])
                ])
                
                if polygon.contains(point) or polygon.touches(point) or polygon.distance(point) < tolerance:
                    points = [
                        (grid_points[k, i, idx1], grid_points[k, i, idx2]),
                        (grid_points[k, i+1, idx1], grid_points[k, i+1, idx2]),
                        (grid_points[k+1, i+1, idx1], grid_points[k+1, i+1, idx2]),
                        (grid_points[k+1, i, idx1], grid_points[k+1, i, idx2])
                    ]
                    try:
                        u, v = inverse_bilinear_interpolation(vertex[idx1], vertex[idx2], points)
                    except RuntimeError as e:
                        if debug_plot:
                            print('projections:', projections)
                            print('Point:', (vertex[idx1], vertex[idx2]))
                            print('Grid Points:', points)
                            self.plot_debug(gmsh_vertices, grid_points)

                            # Plot 2D Projection
                            fig, ax = plt.subplots()
                            for k in range(grid_points.shape[0] - 1):
                                for i in range(grid_points.shape[1] - 1):
                                    ipoints = [
                                        (grid_points[k, i, idx1], grid_points[k, i, idx2]),
                                        (grid_points[k, i+1, idx1], grid_points[k, i+1, idx2]),
                                        (grid_points[k+1, i+1, idx1], grid_points[k+1, i+1, idx2]),
                                        (grid_points[k+1, i, idx1], grid_points[k+1, i, idx2])
                                    ]
                                    polygon = plt.Polygon(ipoints, fill=None, edgecolor='black')
                                    ax.add_patch(polygon)
                            ax.scatter(gmsh_vertices[:, idx1], gmsh_vertices[:, idx2], c='r', marker='o', label='Gmsh Vertices')
                            polygon = plt.Polygon(points, fill=None, edgecolor='b')
                            ax.add_patch(polygon)
                            ax.scatter(vertex[idx1], vertex[idx2], c='b', marker='^', s=100, label='Not Converged Vertex')
                            ax.set_xlabel(projections[0, 0].upper()[0])
                            ax.set_ylabel(projections[0, 0].upper()[1])
                            ax.legend()
                            plt.show()
                        raise RuntimeError(f"Optimization did not converge: {e}")
                    
                    dz = (z - grid_points[k, 0, 2]) / (grid_points[k+1, 0, 2] - grid_points[k, 0, 2])
    
                    param_coords.append((i, k, u, v, dz, projection, rotation_angle))
                    found = True
                    break
            
            if not found:
                if debug_plot:
                    self.plot_debug(gmsh_vertices, grid_points, vertex)
                raise ValueError(f"Vertex {vertex} is not within any grid cell.")
    
        self.param_coords = param_coords
    
    def plot_debug(self, gmsh_vertices, grid_points, not_found_vertex=None):
        """
        Plot a 3D debug plot showing Gmsh vertices and grid points, highlighting the not found vertex.
    
        Parameters:
        -----------
        gmsh_vertices : np.ndarray
            Array of Gmsh vertices.
        grid_points : np.ndarray
            Array of grid points.
        not_found_vertex : np.ndarray
            The vertex that was not found in any grid cell.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # Plot Gmsh vertices
        ax.scatter(gmsh_vertices[:, 0], gmsh_vertices[:, 1], gmsh_vertices[:, 2], c='b', marker='o', label='Gmsh Vertices')
    
        # Plot grid points
        ax.scatter(grid_points[:, :, 0], grid_points[:, :, 1], grid_points[:, :, 2], c='g', marker='x', label='Grid Points')
    
        # Highlight the not found vertex
        if not_found_vertex is not None:
            ax.scatter(not_found_vertex[0], not_found_vertex[1], not_found_vertex[2], c='r', marker='^', s=100, label='Not Found Vertex')
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
    
    def deform_mesh(self, new_top_coords: np.ndarray = None, new_bottom_coords: np.ndarray = None,
                    disct_z: Optional[int] = None, bias: Optional[float] = None, 
                    min_dz: Optional[float] = None, use_depth_only: bool = True, projection: Optional[str] = None) -> np.ndarray:
        """
        Deforms the mesh based on new top and bottom coordinates.
    
        Parameters:
        - new_top_coords (np.ndarray): New top coordinates for the grid. Default is None.
        - new_bottom_coords (np.ndarray): New bottom coordinates for the grid. Default is None.
        - disct_z (Optional[int]): Number of divisions in the z-direction. Default is None.
        - bias (Optional[float]): Bias for the grid spacing. Default is None.
        - min_dz (Optional[float]): Minimum spacing in the z-direction. Default is None.
        - use_depth_only (bool): Whether to use depth only for deformation. Default is True.
        - projection (Optional[str]): Projection plane to use ('xy', 'xz', 'yz'). Default is None, which means using the projection from self.param_coords.
    
        Returns:
        - np.ndarray: Array of deformed vertices.
        """
        grid_points = self.generate_grid_coordinates(top_coords=new_top_coords, bottom_coords=new_bottom_coords, 
                                                     disct_z=disct_z, bias=bias, min_dz=min_dz, use_depth_only=use_depth_only)
    
        rotation_angle = self.param_coords[0][-1]
        if rotation_angle is not None:
            rotation_angle_rad = np.radians(rotation_angle)
            rotation_complex = np.exp(1j * rotation_angle_rad)
            
            rotated_points = (grid_points[:, :, 0] + 1j * grid_points[:, :, 1]) * rotation_complex
            grid_points[:, :, 0] = rotated_points.real
            grid_points[:, :, 1] = rotated_points.imag
    
        deformed_vertices = []
        for i, k, u, v, dz, node_projection, _ in self.param_coords:
            current_projection = projection if projection is not None else node_projection
            if current_projection == 'xy':
                points = [
                    (grid_points[k, i, 0], grid_points[k, i, 1], grid_points[k, i, 2]),
                    (grid_points[k, i+1, 0], grid_points[k, i+1, 1], grid_points[k, i+1, 2]),
                    (grid_points[k+1, i+1, 0], grid_points[k+1, i+1, 1], grid_points[k+1, i+1, 2]),
                    (grid_points[k+1, i, 0], grid_points[k+1, i, 1], grid_points[k+1, i, 2])
                ]
                new_x, new_y, new_z = bilinear_interpolation(u, v, points)
            elif current_projection == 'xz':
                points = [
                    (grid_points[k, i, 0], grid_points[k, i, 2], grid_points[k, i, 1]),
                    (grid_points[k, i+1, 0], grid_points[k, i+1, 2], grid_points[k, i+1, 1]),
                    (grid_points[k+1, i+1, 0], grid_points[k+1, i+1, 2], grid_points[k+1, i+1, 1]),
                    (grid_points[k+1, i, 0], grid_points[k+1, i, 2], grid_points[k+1, i, 1])
                ]
                new_x, new_z, new_y = bilinear_interpolation(u, v, points)
            else:  # current_projection == 'yz'
                points = [
                    (grid_points[k, i, 1], grid_points[k, i, 2], grid_points[k, i, 0]),
                    (grid_points[k, i+1, 1], grid_points[k, i+1, 2], grid_points[k, i+1, 0]),
                    (grid_points[k+1, i+1, 1], grid_points[k+1, i+1, 2], grid_points[k+1, i+1, 0]),
                    (grid_points[k+1, i, 1], grid_points[k+1, i, 2], grid_points[k+1, i, 0])
                ]
                new_y, new_z, new_x = bilinear_interpolation(u, v, points)
            
            new_z = (1 - dz) * grid_points[k, 0, 2] + dz * grid_points[k+1, 0, 2]
            deformed_vertices.append([new_x, new_y, new_z])
    
        deformed_vertices = np.array(deformed_vertices)
    
        if rotation_angle is not None:
            rotation_angle_rad = np.radians(-rotation_angle)
            rotation_complex = np.exp(1j * rotation_angle_rad)
            
            rotated_points = (deformed_vertices[:, 0] + 1j * deformed_vertices[:, 1]) * rotation_complex
            deformed_vertices[:, 0] = rotated_points.real
            deformed_vertices[:, 1] = rotated_points.imag
    
        return deformed_vertices

    def generate_gmsh_mesh(self, top_size=None, bottom_size=None, mesh_func=None, out_mesh=None, 
                           write2file=False, show=True, read_mesh=True, field_size_dict={'min_dx': 3, 'bias': 1.05}, 
                           segments_dict=None, verbose=5, mesh_algorithm=2, optimize_method='Laplace2D', save_in_self=False):
        """
        Generate a mesh using Gmsh.
    
        Parameters:
        - top_size (float, optional): Mesh size for the top points. If None, mesh_func will be used.
        - bottom_size (float, optional): Mesh size for the bottom points. If None, mesh_func will be used.
        - mesh_func (callable, optional): Function to define mesh size. If None, top_size and bottom_size will be used.
        - out_mesh (str, optional): Output mesh file path. If None, a default path will be used.
        - write2file (bool, optional): Whether to write the mesh to a file.
        - show (bool, optional): Whether to show the mesh in Gmsh GUI.
        - read_mesh (bool, optional): Whether to read the mesh into the current object.
        - field_size_dict (dict, optional): Dictionary containing 'min_dx' and 'bias' for mesh size progression.
        - segments_dict (dict, optional): Dictionary containing segment information for transfinite curves.
            Keys: top_segments, bottom_segments, left_segments, right_segments, top_bottom_progression, left_right_progression
        - verbose (int, optional): Verbosity level for Gmsh, ranging from 0 (no log) to 5 (all logs).
        - mesh_algorithm (int, optional): Algorithm to use for mesh generation.
        - optimize_method (str, optional): Method to use for mesh optimization.
        - save_in_self (bool, optional): Whether to save the mesh in the current object.
    
        Returns:
        - None
    
        Notes:
        - If segments_dict is provided, top_size, bottom_size, and mesh_func should not be used.
        - If mesh_func and field_size_dict are provided, top_size and bottom_size should not be used.
        - At least one of (top_size, bottom_size), (mesh_func, field_size_dict), or segments_dict must be provided.
        """
        # Validate parameters
        if segments_dict is not None:
            if any([top_size, bottom_size, mesh_func]):
                raise ValueError("segments_dict cannot be used with top_size, bottom_size, or mesh_func.")
        elif mesh_func is not None and field_size_dict is not None:
            if any([top_size, bottom_size]):
                raise ValueError("mesh_func and field_size_dict cannot be used with top_size or bottom_size.")
        elif top_size is not None and bottom_size is not None:
            if mesh_func:
                raise ValueError("top_size and bottom_size cannot be used with mesh_func and field_size_dict.")
        else:
            raise ValueError("At least one of (top_size, bottom_size), (mesh_func, field_size_dict), or segments_dict must be provided.")
    
        assert self.top_coords is not None and self.bottom_coords is not None, 'Top and bottom coordinates must be set.'
    
        gmsh.initialize('', False)
        gmsh.option.setNumber("General.Terminal", verbose)
        gmsh.clear()
    
        # Set the 2D mesh algorithm.
        # Available options:
        # 1: MeshAdapt
        # 2: Automatic (default)
        # 3: Initial mesh only
        # 5: Delaunay
        # 6: Frontal-Delaunay
        # 7: BAMG
        # 8: Frontal-Delaunay for Quads
        # 9: Packing of Parallelograms
        # 11: Quasi-structured Quad
        gmsh.option.setNumber("Mesh.Algorithm", mesh_algorithm)
        
        # Check if top_size is a list, array, or None and has the same length as top_coords
        if isinstance(top_size, (list, np.ndarray)) and len(top_size) == len(self.top_coords):
            top_points = [gmsh.model.geo.addPoint(point[0], point[1], -point[2], size) for point, size in zip(self.top_coords, top_size)]
        else:
            top_points = [gmsh.model.geo.addPoint(point[0], point[1], -point[2], top_size or 0.0) for point in self.top_coords]
        
        # Check if bottom_size is a list, array, or None and has the same length as bottom_coords
        if isinstance(bottom_size, (list, np.ndarray)) and len(bottom_size) == len(self.bottom_coords):
            bottom_points = [gmsh.model.geo.addPoint(point[0], point[1], -point[2], size) for point, size in zip(self.bottom_coords, bottom_size)]
        else:
            bottom_points = [gmsh.model.geo.addPoint(point[0], point[1], -point[2], bottom_size or 0.0) for point in self.bottom_coords]
    
        # Create splines for top and bottom curves
        top_curve = gmsh.model.geo.addSpline(top_points)
        bottom_curve = gmsh.model.geo.addSpline(bottom_points)
        left_edge = gmsh.model.geo.addLine(bottom_points[0], top_points[0])
        right_edge = gmsh.model.geo.addLine(top_points[-1], bottom_points[-1])
    
        # Set segments if provided
        if segments_dict is not None:
            self._set_segments(top_curve, bottom_curve, left_edge, right_edge, segments_dict)
    
        # Create surface from curves
        face1 = gmsh.model.geo.addCurveLoop([top_curve, right_edge, -bottom_curve, left_edge])
        surface = gmsh.model.geo.addSurfaceFilling([face1])
    
        # Set transfinite surface if segments are provided
        if segments_dict is not None:
            gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    
        gmsh.model.geo.synchronize()
    
        # Define mesh size using mesh_func if provided
        if mesh_func is not None:
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    
            field_distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field_distance, "CurvesList", [top_curve])
    
            field_size = gmsh.model.mesh.field.add("MathEval")
            math_exp = self.get_math_progression(field_distance, min_dx=field_size_dict['min_dx'], bias=field_size_dict['bias'])
            gmsh.model.mesh.field.setString(field_size, "F", math_exp)
    
            gmsh.model.mesh.field.setAsBackgroundMesh(field_size)
    
        # Remove points and lines to clean up the model
        gmsh.model.removeEntities(gmsh.model.getEntities(0))
        gmsh.model.removeEntities(gmsh.model.getEntities(1))
    
        # Refine, generate, and optimize the mesh
        gmsh.model.mesh.refine()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize(optimize_method)
    
        # Write mesh to file if specified
        if out_mesh is not None:
            self.out_mesh = out_mesh
        if write2file:
            gmsh.write(self.out_mesh)
    
        # Show the mesh in Gmsh GUI if specified
        if show:
            if 'close' not in sys.argv:
                gmsh.fltk.run()

        # Read the mesh into the current object if specified
        if read_mesh:
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            vertices = node_coords.reshape(-1, 3)
            if np.mean(vertices[:, 2]) < 0:
                vertices[:, 2] = -vertices[:, 2]
            _, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
            faces = np.array(node_tags[0]).reshape(-1, 3) - 1
    
        gmsh.finalize()

        if save_in_self:
            self.gmsh_verts = vertices
            self.gmsh_faces = faces
        return vertices, faces
    
    def generate_multilayer_gmsh_mesh(self, layers_coords, sizes=None, mesh_func=None, 
                                      out_mesh=None, write2file=False, show=True, read_mesh=True, 
                                      field_size_dict={'min_dx': 3, 'bias': 1.05},
                                      mesh_algorithm=2, optimize_method='Laplace2D', verbose=5):
        """
        Generate a multi-layer mesh using Gmsh.
    
        Parameters:
        - layers_coords (list of np.ndarray): List of intermediate layer coordinates.
        - sizes (list of float, optional): List of mesh sizes for each layer.
        - mesh_func (callable, optional): Function to define mesh size.
        - out_mesh (str, optional): Output mesh file path.
        - write2file (bool, optional): Whether to write the mesh to a file.
        - show (bool, optional): Whether to show the mesh in Gmsh GUI.
        - read_mesh (bool, optional): Whether to read the mesh into the current object.
        - field_size_dict (dict, optional): Dictionary containing 'min_dx' and 'bias' for mesh size progression.
        - mesh_algorithm (int, optional): Algorithm to use for mesh generation.
        - optimize_method (str, optional): Method to use for mesh optimization.
        - verbose (int, optional): Verbosity level for Gmsh.
    
        Returns:
        - None
        """
        gmsh.initialize('', False)
        gmsh.option.setNumber("General.Terminal", verbose)
        gmsh.clear()
        gmsh.option.setNumber("Mesh.Algorithm", mesh_algorithm)
    
        # Add points for top and bottom coordinates
        top_points = [gmsh.model.geo.addPoint(point[0], point[1], -point[2], self.top_size or 0.0) for point in self.top_coords]
        bottom_points = [gmsh.model.geo.addPoint(point[0], point[1], -point[2], self.bottom_size or 0.0) for point in self.bottom_coords]
    
        # Add points for intermediate layers if provided
        layer_points = []
        if layers_coords is not None:
            for layer, size in zip(layers_coords, sizes or [0.0]*len(layers_coords)):
                layer_points.append([gmsh.model.geo.addPoint(point[0], point[1], -point[2], size) for point in layer])
    
        all_points = [top_points] + layer_points + [bottom_points]
    
        # Create splines for all layers
        all_curves = [gmsh.model.geo.addSpline(layer) for layer in all_points]
    
        # Create left and right edges
        left_edges = [gmsh.model.geo.addLine(all_points[i][0], all_points[i+1][0]) for i in range(len(all_points) - 1)]
        right_edges = [gmsh.model.geo.addLine(all_points[i][-1], all_points[i+1][-1]) for i in range(len(all_points) - 1)]
    
        # Create surfaces from curves
        for i in range(len(all_curves) - 1):
            face = gmsh.model.geo.addCurveLoop([all_curves[i], right_edges[i], -all_curves[i+1], -left_edges[i]])
            gmsh.model.geo.addSurfaceFilling([face])
    
        gmsh.model.geo.synchronize()
    
        # Define mesh size using mesh_func if provided
        if mesh_func is not None:
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    
            field_distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field_distance, "CurvesList", [all_curves[0]])
    
            field_size = gmsh.model.mesh.field.add("MathEval")
            math_exp = self.get_math_progression(field_distance, min_dx=field_size_dict['min_dx'], bias=field_size_dict['bias'])
            gmsh.model.mesh.field.setString(field_size, "F", math_exp)
    
            gmsh.model.mesh.field.setAsBackgroundMesh(field_size)
    
        # Remove points and lines to clean up the model
        gmsh.model.removeEntities(gmsh.model.getEntities(0))
        gmsh.model.removeEntities(gmsh.model.getEntities(1))
    
        # Refine, generate, and optimize the mesh
        gmsh.model.mesh.refine()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize(optimize_method)
    
        # Write mesh to file if specified
        if out_mesh is not None:
            self.out_mesh = out_mesh
        if write2file:
            gmsh.write(self.out_mesh)
    
        # Show the mesh in Gmsh GUI if specified
        if show:
            if 'close' not in sys.argv:
                gmsh.fltk.run()

        # Read the mesh into the current object if specified
        if read_mesh:
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            vertices = node_coords.reshape(-1, 3)
            if np.mean(vertices[:, 2]) < 0:
                vertices[:, 2] = -vertices[:, 2]
            _, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
            faces = np.array(node_tags[0]).reshape(-1, 3) - 1
    
        gmsh.finalize()
        return vertices, faces
    
    def get_math_progression(self, field_distance, min_dx, bias):
        """
        Generate the Gmsh MathEval string corresponding to the cell size as a function
        of distance, starting cell size, and bias factor.

        The expression is min_dx * bias**n, where n is the number of cells from the fault.
        n = log(1+distance/min_dx*(bias-1))/log(bias)

        In finding the expression for `n`, we make use that the sum of a geometric series with n
        terms Sn = min_dx * (1 + bias + bias**2 + ... + bias**n) = min_dx * (bias**n - 1)/(bias - 1).
    
        Parameters:
        - field_distance (int): Field distance identifier.
        - min_dx (float): Minimum mesh size.
        - bias (float): Bias for mesh size progression.
    
        Returns:
        - str: Mathematical expression for mesh size progression.
        """
        return f"{min_dx}*{bias}^(Log(1.0+F{field_distance}/{min_dx}*({bias}-1.0))/Log({bias}))"
    
    def _set_segments(self, top_curve, bottom_curve, left_edge, right_edge, segments_dict):
        """
        Set the segments for transfinite curves.
    
        Parameters:
        - top_curve (int): Top curve identifier.
        - bottom_curve (int): Bottom curve identifier.
        - left_edge (int): Left edge identifier.
        - right_edge (int): Right edge identifier.
        - segments_dict (dict): Dictionary containing segment information.
    
        Returns:
        - None
        """
        top_segments = segments_dict.get('top_segments', 1)
        bottom_segments = segments_dict.get('bottom_segments', 1)
        left_segments = segments_dict.get('left_segments', 1)
        right_segments = segments_dict.get('right_segments', 1)
    
        top_bottom_progression = segments_dict.get('top_bottom_progression', 1.0)
        left_right_progression = segments_dict.get('left_right_progression', 1.0)
    
        if top_segments != bottom_segments or left_segments != right_segments:
            raise ValueError("The number of segments for opposite sides must be equal.")
    
        gmsh.model.geo.mesh.setTransfiniteCurve(top_curve, top_segments, "Progression", top_bottom_progression)
        gmsh.model.geo.mesh.setTransfiniteCurve(bottom_curve, bottom_segments, "Progression", top_bottom_progression)
        gmsh.model.geo.mesh.setTransfiniteCurve(left_edge, left_segments, "Progression", 1/left_right_progression)
        gmsh.model.geo.mesh.setTransfiniteCurve(right_edge, right_segments, "Progression", left_right_progression)

    def _generate_mesh_faces(self, mesh_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the faces of the mesh from the mesh coordinates.

        Parameters:
        - mesh_coords (np.ndarray): The coordinates of the mesh.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: The vertices and faces of the mesh.
        """
        xfault, yfault, zfault = mesh_coords[:, :, 0], mesh_coords[:, :, 1], mesh_coords[:, :, 2]
        p, q, r = (xfault.T).reshape(-1, 1), (yfault.T).reshape(-1, 1), (zfault.T).reshape(-1, 1)
        rowp, colp = xfault.shape[0], yfault.shape[1]
        numpatch = 2 * ((rowp - 1) * (colp - 1))
        trired = np.zeros((numpatch, 3), dtype=int)

        # Initialize the counter
        triangle_index = 0
        for column in range(colp - 1):
            for row in range(rowp - 1):
                index = column * rowp + row
                trired[triangle_index, :] = [index, index + 1, index + rowp + 1]
                trired[triangle_index + 1, :] = [index, index + rowp + 1, index + rowp]
                triangle_index += 2

        vertices = np.hstack((p, q, r))
        faces = trired
        return vertices, faces

    def _split_coords_into_segments(self, top_coords: np.ndarray, layers_coords: List[np.ndarray], 
                                    bottom_coords: np.ndarray, num_segments: int, bias: float) -> np.ndarray:
        """
        Split coordinates into segments for multi-layer mesh generation.

        Parameters:
        - top_coords (np.ndarray): The top coordinates.
        - layers_coords (List[np.ndarray]): The coordinates of the intermediate layers.
        - bottom_coords (np.ndarray): The bottom coordinates.
        - num_segments (int): The number of segments.
        - bias (float): Bias used to adjust the length of each segment.

        Returns:
        - np.ndarray: The segmented coordinates.
        """
        n, d = top_coords.shape
        result = np.zeros((num_segments, n, d))

        all_coords = np.stack([top_coords] + layers_coords + [bottom_coords])
        diffs = np.diff(all_coords, axis=0)
        lengths = np.linalg.norm(diffs, axis=2)
        total_length = np.sum(lengths, axis=0)
        arc_lengths = np.cumsum(np.vstack([np.zeros((1, n)), lengths]), axis=0)

        for i in range(n):
            f = interp1d(arc_lengths[:, i], all_coords[:, i, :], axis=0)
            # If bias is 1, use linear interpolation directly
            if bias == 1:
                segment_arc_lengths = np.linspace(0, total_length[i], num_segments)
            else:
                # Sn = min_dx * (bias**n - 1)/(bias - 1)
                # Calculate min_dz
                min_dz = total_length[i] / (bias**(num_segments-1) - 1) * (bias - 1)
                # Update each layer's coordinates using the bias formula based on the number and position of depth layers
                segment_arc_lengths = min_dz * (bias**np.arange(num_segments) - 1) / (bias - 1)
                segment_arc_lengths[-1] = total_length[i]
            result[:, i, :] = f(segment_arc_lengths)

        return result

# Example usage
if __name__ == "__main__":
    top_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    bottom_coords = np.array([[0, 0, -1], [1, 0, -1], [0, 1, -1]])
    disct_z = 10

    mesh_generator = MeshGenerator(top_coords, bottom_coords)
    vertices, faces = mesh_generator.generate_simple_mesh(disct_z)
    print("Vertices:\n", vertices)
    print("Faces:\n", faces)