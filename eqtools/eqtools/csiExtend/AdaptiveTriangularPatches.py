'''
Added by kfhe at 10/5/2023
Object :
    * Adaptive build triangular fault with or without relocated aftershocks as constraint
    * Fit relocated aftershocks to iso-depth curve
    * Fit relocated aftershocks to profiles along strike
'''

# Standard library imports
import sys
import os
import json
from typing import List, Union

# Third-party imports
import gmsh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from scipy import interpolate
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from numpy import rad2deg

# Local application imports
from csi import TriangularPatches, seismiclocations
from .fitting_methods import RegressionFitter
from ..plottools import DegreeFormatter
from .MeshGenerator import MeshGenerator
from .geom_ops import discretize_coords
from ..plottools import sci_plot_style

def str2num(istr: str, dtype=int) -> List[Union[int, float]]:
    return [dtype(ix.strip()) for ix in istr.strip().split()]


from numba import jit

@jit(nopython=True)
def compute_triangle_normals(vertices, faces):
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    vector1 = v2 - v1
    vector2 = v3 - v1
    normals = np.cross(vector1, vector2)
    normals_length = np.sqrt(np.sum(normals**2, axis=1))
    normals = normals / normals_length.reshape(-1, 1)
    return normals

@jit(nopython=True)
def arrange_vertices_clockwise(vertices, faces):
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    vector1 = v2 - v1
    vector2 = v3 - v1
    cross_product = np.cross(vector1, vector2)
    faces[cross_product[:, 2] < 0, [1, 2]] = faces[cross_product[:, 2] < 0, [2, 1]]
    return faces

@jit(nopython=True)
def calculate_triangle_areas(vertices, faces):
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    a = np.sqrt(np.sum((v2 - v1)**2, axis=1))
    b = np.sqrt(np.sum((v3 - v2)**2, axis=1))
    c = np.sqrt(np.sum((v1 - v3)**2, axis=1))
    s = (a + b + c) / 2
    areas = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return areas


def calculate_progression(first_length, last_length, num_segments):
    """
    Calculates the common ratio of a geometric sequence based on the lengths of the first and last segments of a boundary.

    Parameters:
    first_length (float): The length of the first segment.
    last_length (float): The length of the last segment.
    num_segments (int): The number of segments along the boundary.

    Returns:
    float: The common ratio of the geometric sequence.

    Formula:
    The common ratio (r) is calculated using the formula: r = (last_length / first_length) ** (1 / (num_segments - 1))
    """
    return (last_length / first_length) ** (1 / (num_segments - 1))

def calculate_first_and_last_length(total_length, progression, num_segments):
    """
    Calculates the lengths of the first and last segments based on the total length, progression ratio, and number of segments.

    Parameters:
    total_length (float): The total length of all segments combined.
    progression (float): The progression ratio between consecutive segments.
    num_segments (int): The total number of segments.

    Returns:
    tuple: A tuple containing the lengths of the first and last segments (first_length, last_length).

    Formula:
    The length of the first segment (first_length) is calculated using the formula:
    first_length = total_length * (1 - progression) / (1 - progression ** num_segments)

    The length of the last segment (last_length) is calculated as:
    last_length = first_length * progression ** (num_segments - 1)
    """
    # Calculate the length of the first segment
    first_length = total_length * (1 - progression) / (1 - progression ** num_segments)

    # Calculate the length of the last segment
    last_length = first_length * progression ** (num_segments - 1)

    return first_length, last_length


def calculate_scale_and_rotation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    scale = 2 / np.sqrt((x2-x1)**2 + (y2-y1)**2)
    angrot_r = np.arctan((y2-y1) / (x2-x1))
    if x1 > x2:
        scale = -scale
    return scale, angrot_r

def create_rotation_matrix(angrot_r):
    rotA = np.zeros(4).reshape(2,2)
    rotA[0,0] = np.cos(angrot_r)
    rotA[0,1] = np.sin(angrot_r)
    rotA[1,0] = -np.sin(angrot_r)
    rotA[1,1] = np.cos(angrot_r)
    return rotA

def transform_point(point1, point2, target_point):
    scale, angrot_r = calculate_scale_and_rotation(point1, point2)
    rotA = create_rotation_matrix(angrot_r)
    trans = scale * rotA.dot(point1.reshape(2,1))
    transformed_point = scale * rotA.dot(target_point.reshape(2,1)) - trans
    return transformed_point

def reverse_transform_point(point1, point2, target_point):
    scale, angrot_r = calculate_scale_and_rotation(point1, point2)
    rotA = create_rotation_matrix(angrot_r)
    trans = scale * rotA.dot(point1.reshape(2,1))
    reversed_point = (1/scale) * np.linalg.inv(rotA).dot(target_point.reshape(2,1) + trans)
    return reversed_point

def find_buffer_points(nodes, trace, buffer_distances):
    """
    Finds the coordinates of points on either side of a given node along a fault trace at specified buffer distances.

    Parameters:
    nodes (numpy array): The given nodes, formatted as a numpy array where each row represents the (x, y) coordinates of a node.
    trace (numpy array): The fault trace, formatted as a numpy array where each row represents the (x, y) coordinates of a node.
    buffer_distances (float or numpy array): The buffer distances, which can be a single value or an array of values corresponding to the length of nodes.

    Returns:
    numpy array: The coordinates of points on either side of each node along the fault trace at the specified buffer distances, formatted as a numpy array where each row represents the (x1, y1, x2, y2) coordinates.

    Formula:
    The distance between a node and a point on the trace is calculated using the Euclidean distance formula:
    distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    where (x1, y1) are the coordinates of the node and (x2, y2) are the coordinates of a point on the trace.

    Notes:
    - If buffer_distances is a single value, it is converted into an array of the same length as nodes.
    - The function uses a KDTree for efficient nearest neighbor search.
    """
    # Convert buffer_distances to an array if it is a single value
    if np.isscalar(buffer_distances):
        buffer_distances = np.full(nodes.shape[0], buffer_distances)
    # Initialize buffer_points
    buffer_points = np.empty((nodes.shape[0], 2, 2))
    buffer_points.fill(np.nan)
    # Create a KDTree for efficient nearest neighbor search
    tree = KDTree(trace)
    # Process each node
    for i, (node, buffer_distance) in enumerate(zip(nodes, buffer_distances)):
        # Find the index of the nearest node on the trace
        nearest_index = tree.query(node)[1]
        # Find indices of nodes on either side of the given node within the buffer distance
        left_indices = np.where((trace[:nearest_index, 0] - node[0])**2 + (trace[:nearest_index, 1] - node[1])**2 <= buffer_distance**2)[0]
        right_indices = np.where((trace[nearest_index+1:, 0] - node[0])**2 + (trace[nearest_index+1:, 1] - node[1])**2 <= buffer_distance**2)[0] + nearest_index + 1
        # If multiple points are found, only return the furthest points within the buffer distance
        if len(left_indices) > 0:
            buffer_points[i, 0] = trace[left_indices[0]]
        if len(right_indices) > 0:
            buffer_points[i, 1] = trace[right_indices[-1]]
    return buffer_points


class AdaptiveTriangularPatches(TriangularPatches):
    def __init__(self, name: str, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):
        super().__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=verbose)
        self.relocated_aftershock_source = seismiclocations('relocs', utmzone=utmzone, lon0=lon0, lat0=lat0, verbose=verbose)
        self.mesh_func = None
        self.out_mesh = f'{name}.msh'
        self.profiles = {}
        self.mesh_generator = MeshGenerator()

    def set_relocated_aftershock_source(self, relocated_aftershock_source):
        """
        Sets the parameters for the relocated aftershock source.
    
        Parameters:
        relocated_aftershock_source: Parameters for the relocated aftershock source.
    
        Note:
        This method configures the relocated aftershock source by specifying its parameters. The relocated aftershock source is used to adjust the model based on aftershock data that has been relocated to more accurately reflect seismic activity.
        """
        self.relocated_aftershock_source = relocated_aftershock_source

    def set_depth_extension(self, depth_extension):
        """
        Sets the depth extension.

        Parameters:
        depth_extension (float): The depth extension value.

        Note:
        The depth extension modifies the overall depth of the mesh, impacting how deep the mesh extends into the model.
        """

        self.depth = depth_extension

    def set_output_mesh(self, output_mesh):
        """
        Sets the output mesh.

        Parameters:
        output_mesh (Mesh): The output mesh object.

        Note:
        The output mesh is the final mesh generated by the algorithm, ready for export or further manipulation.
        """

        self.out_mesh = output_mesh

    def set_mesh_function(self, mesh_function_str):
        """
        Sets the mesh function.

        Parameters:
        mesh_function_str (str): The string representation of the mesh function.

        Note:
        The mesh function defines how the mesh density varies across the model. It's specified as a string that can be evaluated to adjust mesh density dynamically.
        """

        self.mesh_func = mesh_function_str

    def set_coords(self, coords, lonlat=True, coord_type='top'):
        """
        Sets the coordinates.

        Parameters:
        coords: Coordinates, a two-dimensional array where each row represents the coordinates of a point.
        lonlat: If True, indicates that the coordinates in coords are in longitude and latitude. Otherwise, the coordinates are in UTM format.
        coord_type: The type of coordinates, which can be 'top', 'bottom', or 'layer'.

        Note:
        This method configures the coordinates for a specific boundary or layer within the model. The coordinate system (longitude/latitude or UTM) is determined by the lonlat parameter. The coord_type parameter specifies the boundary or layer these coordinates apply to.
        """
        if coords is None or len(coords) == 0:
            raise ValueError(f"{coord_type}_coords cannot be None or empty.")
        if coord_type == 'layer':
            if not isinstance(coords, list):
                raise ValueError("For 'layer', coords should be a list of 2D arrays.")
            self.layers_ll = [None] * len(coords)
            self.layers = [None] * len(coords)
            for i, layer_coords in enumerate(coords):
                if lonlat:
                    self.layers_ll[i] = layer_coords
                    lon, lat = layer_coords[:, 0], layer_coords[:, 1]
                    x, y = self.ll2xy(lon, lat)
                    self.layers[i] = np.vstack((x, y, layer_coords[:, 2])).T
                else:
                    self.layers[i] = layer_coords
                    x, y, z = layer_coords[:, 0], layer_coords[:, 1], layer_coords[:, 2] # unit: km
                    lon, lat = self.xy2ll(x, y)
                    self.layers_ll[i] = np.vstack((lon, lat, z)).T
        else:
            if lonlat:
                setattr(self, f"{coord_type}_coords_ll", coords)
                lon, lat = coords[:, 0], coords[:, 1]
                x, y = self.ll2xy(lon, lat)
                setattr(self, f"{coord_type}_coords", np.vstack((x, y, coords[:, 2])).T)
            else:
                setattr(self, f"{coord_type}_coords", coords)
                x, y, z = coords[:, 0], coords[:, 1], coords[:, 2] # unit: km
                lon, lat = self.xy2ll(x, y)
                setattr(self, f"{coord_type}_coords_ll", np.vstack((lon, lat, z)).T)

    def set_top_coords(self, top_coords, lonlat=True):
        """
        Sets the top coordinates.

        Parameters:
        top_coords: The coordinates of the top, a two-dimensional array where each row represents the coordinates of a point.
        lonlat: If True, indicates that the coordinates in top_coords are in longitude and latitude. Otherwise, the coordinates are in UTM format.

        Note:
        This method configures the top boundary of a model by specifying its coordinates. The coordinate system (longitude/latitude or UTM) is determined by the lonlat parameter.
        """
        self.set_coords(top_coords, lonlat, 'top')

    def set_bottom_coords(self, bottom_coords, lonlat=True):
        """
        Sets the bottom coordinates.

        Parameters:
        bottom_coords: The coordinates of the bottom, a two-dimensional array where each row represents the coordinates of a point.
        lonlat: If True, indicates that the coordinates in bottom_coords are in longitude and latitude. Otherwise, the coordinates are in UTM format.

        Note:
        This method configures the bottom boundary of a model by specifying its coordinates. The coordinate system (longitude/latitude or UTM) is determined by the lonlat parameter.
        """
        self.set_coords(bottom_coords, lonlat, 'bottom')

    def set_layer_coords(self, layer_coords, lonlat=True):
        """
        Sets the layer coordinates.

        Parameters:
        layer_coords: The coordinates of the layer, a two-dimensional array where each row represents the coordinates of a point.
        lonlat: If True, indicates that the coordinates in layer_coords are in longitude and latitude. Otherwise, the coordinates are in UTM format.

        Note:
        This method configures the coordinates of a specific layer within a model. The coordinate system (longitude/latitude or UTM) is determined by the lonlat parameter.
        """
        self.set_coords(layer_coords, lonlat, 'layer')
    
    def set_top_coords_from_geometry(self, top_tolerance=0.1, bottom_tolerance=0.1, lonlat=True, buffer_depth=0.1, sort_axis=0, sort_order='ascend'):
        """
        Set the top coordinates from the geometry by finding the ordered edge vertices.
    
        Parameters:
        -----------
        top_tolerance : float, optional
            The tolerance for the top edge. Default is 0.1.
        bottom_tolerance : float, optional
            The tolerance for the bottom edge. Default is 0.1.
        lonlat : bool, optional
            Whether to convert the coordinates to longitude and latitude. Default is True.
        buffer_depth : float, optional
            The buffer depth to include points within the edge. Default is 0.1.
        sort_axis : int, optional
            The axis to sort the coordinates. Default is 0.
        sort_order : str, optional
            The order to sort the coordinates ('ascend' or 'descend'). Default is 'ascend'.
    
        Returns:
        --------
        None
        """
        tcoords = self.find_ordered_edge_vertices(edge='top', depth=self.top, buffer_depth=buffer_depth, 
                                                  top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance)
        if lonlat:
            lon, lat = self.xy2ll(tcoords[:, 0], tcoords[:, 1])
            tcoords = np.vstack((lon, lat, tcoords[:, 2])).T
        
        # Sort the coordinates based on the first and last values of the specified axis
        if sort_order == 'ascend':
            if tcoords[0, sort_axis] > tcoords[-1, sort_axis]:
                tcoords = tcoords[::-1]
        elif sort_order == 'descend':
            if tcoords[0, sort_axis] < tcoords[-1, sort_axis]:
                tcoords = tcoords[::-1]
        else:
            raise ValueError("Invalid value for sort_order. It should be 'ascend' or 'descend'.")
        
        self.set_top_coords(tcoords, lonlat=lonlat)
    
    def set_bottom_coords_from_geometry(self, top_tolerance=0.1, bottom_tolerance=0.1, lonlat=True, buffer_depth=0.1, sort_axis=0, sort_order='ascend'):
        """
        Set the bottom coordinates from the geometry by finding the ordered edge vertices.
    
        Parameters:
        -----------
        top_tolerance : float, optional
            The tolerance for the top edge. Default is 0.1.
        bottom_tolerance : float, optional
            The tolerance for the bottom edge. Default is 0.1.
        lonlat : bool, optional
            Whether to convert the coordinates to longitude and latitude. Default is True.
        buffer_depth : float, optional
            The buffer depth to include points within the edge. Default is 0.1.
        sort_axis : int, optional
            The axis to sort the coordinates. Default is 0.
        sort_order : str, optional
            The order to sort the coordinates ('ascend' or 'descend'). Default is 'ascend'.
    
        Returns:
        --------
        None
        """
        bcoords = self.find_ordered_edge_vertices(edge='bottom', depth=self.depth, buffer_depth=buffer_depth, 
                                                  top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance)
        if lonlat:
            lon, lat = self.xy2ll(bcoords[:, 0], bcoords[:, 1])
            bcoords = np.vstack((lon, lat, bcoords[:, 2])).T
        
        # Sort the coordinates based on the first and last values of the specified axis
        if sort_order == 'ascend':
            if bcoords[0, sort_axis] > bcoords[-1, sort_axis]:
                bcoords = bcoords[::-1]
        elif sort_order == 'descend':
            if bcoords[0, sort_axis] < bcoords[-1, sort_axis]:
                bcoords = bcoords[::-1]
        else:
            raise ValueError("Invalid value for sort_order. It should be 'ascend' or 'descend'.")
        
        self.set_bottom_coords(bcoords, lonlat=lonlat)

    def set_top_coords_from_trace(self, discretized=False, sort_axis=0, sort_order=None):
        """
        Set the top coordinates from the fault trace, with optional sorting.
    
        Parameters:
        -----------
        discretized : bool
            If True, use the discretized fault trace; otherwise, use the original.
        sort_axis : int, optional
            Axis to sort by (0 for x, 1 for y). Default is 0.
        sort_order : str or None, optional
            'ascend' for ascending, 'descend' for descending, None for no sorting (default).
    
        Returns:
        --------
        None
        """
        if discretized:
            x, y = self.xi, self.yi
        else:
            x, y = self.xf, self.yf
        z = np.ones_like(x) * self.top
        coords = np.vstack((x, y, z)).T
    
        # Sorting logic
        if sort_order is not None:
            if sort_order == 'ascend':
                # If the first value is greater than the last, reverse to make ascending
                if coords[0, sort_axis] > coords[-1, sort_axis]:
                    coords = coords[::-1]
            elif sort_order == 'descend':
                # If the first value is less than the last, reverse to make descending
                if coords[0, sort_axis] < coords[-1, sort_axis]:
                    coords = coords[::-1]
            else:
                raise ValueError("sort_order must be 'ascend', 'descend', or None")
    
        self.set_coords(coords, lonlat=False, coord_type='top')

    #--------------------------------------Simply Mesh From Top to Bottom--------------------------------------#
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
        x_new, y_new, z = xyz_new[:, 0], xyz_new[:, 1], xyz_new[:, 2]
        # Compute the lon/lat
        loni, lati = self.xy2ll(x_new, y_new)
        lonlatz_new = np.vstack((loni, lati, z)).T
    
        return xyz_new, lonlatz_new
    
    def discretize_top_coords(self, every=None, num_segments=None, threshold=2):
        xyz_new, lonlatz_new = self.discretize_coords(self.top_coords, every, num_segments, threshold)
        self.top_coords = xyz_new
        self.top_coords_ll = lonlatz_new
    
    def discretize_bottom_coords(self, every=None, num_segments=None, threshold=2):
        xyz_new, lonlatz_new = self.discretize_coords(self.bottom_coords, every, num_segments, threshold)
        self.bottom_coords = xyz_new
        self.bottom_coords_ll = lonlatz_new
    
    def discretize_layer_coords(self, mid_layer_index=0, every=None, num_segments=None, threshold=2):
        assert hasattr(self, 'layers') and mid_layer_index < len(self.layers), 'You need to set layers before discretizing the coordinates.'
        xyz_new, lonlatz_new = self.discretize_coords(self.layers[mid_layer_index], every, num_segments, threshold)
        self.layers[mid_layer_index] = xyz_new
        self.layers_ll[mid_layer_index] = lonlatz_new

    def fit_relocated_aftershock_iso_depth_curve(
            self, 
            x_fit=None,
            methods=['ols', 'theil_sen', 'ransac', 'huber', 'lasso',
                    'ridge', 'elasticnet', 'quantile', 
                    'groupby_interpolation', 'polynomial'], 
            degree=3, fit_params=None, mindepth=0, maxdepth=20, 
            show=True, save2csi=True, 
            focusdepth=None,
            minlon=None, maxlon=None, 
            minlat=None, maxlat=None,
            trimst=None, trimed=None, 
            trimaxis='x', trimutm=False,
            save_fig=None, 
            dpi=600, 
            style=['science'],
            show_error_in_legend=False,
            use_lon_lat=False,
            fontsize=None, 
            figsize=None, 
            scatter_props=None,
            output_dir='.',  # Default to current directory
            custom_data_for_plotting=None,
            equal_aspect=False, legend_position=None,
            principal_direction_angle=None,
            show_data_in_principal_direction=False,
            align_fit_with_rupture_trace_in_principal_direction=False
        ):
        """
        Fit relocated aftershock isodepth curve using various regression methods.
    
        Parameters:
        - x_fit: X values to fit the model.
        - methods: List of regression methods to use.
        - degree: Degree of polynomial for polynomial regression.
        - fit_params: Additional parameters for fitting methods.
        - mindepth, maxdepth: Depth range for selecting aftershocks.
        - show: Whether to show the plot.
        - save2csi: Whether to save the curve to self.relocisocurve.
        - focusdepth: Depth to focus on.
        - minlon, maxlon, minlat, maxlat: Longitude and latitude range for selecting aftershocks.
        - trimst, trimed: Start and end value for trimming the curve.
        - trimaxis: Axis to trim ('x' or 'y').
        - trimutm: Whether to use UTM coordinates for trimming.
        - save_fig: Path to save the figure.
        - dpi: DPI for saving the figure.
        - style: Style for plotting.
        - show_error_in_legend: Whether to show error in legend.
        - use_lon_lat: Whether to use longitude and latitude for plotting.
        - fontsize: Font size for plotting.
        - figsize: Figure size for plotting.
        - scatter_props: Properties for scatter plot.
        - output_dir: Directory to save the results.
        - custom_data_for_plotting: Dictionary containing custom data for plotting. Keys are labels, values are (x, y) tuples.
        - equal_aspect: Whether to set equal aspect ratio for the plot.
        - legend_position: Tuple containing (bbox_to_anchor, loc) for the legend position. Default is None.
        - principal_direction_angle: Angle of the principal direction in degrees. If None, the principal direction will be calculated.
            0 degrees corresponds to the x-axis. clockwise is positive.
        - show_data_in_principal_direction: Whether to show the data in the principal direction.
        - align_fit_with_rupture_trace_in_principal_direction: Whether to align the fitted curve with the rupture trace.
    
        Returns:
        None
        """
        if type(methods) in (str,):
            methods = [methods]
        if len(methods) > 1:
            save2csi = False
    
        if fit_params is None:
            fit_params = {}
    
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # If save_fig is not an absolute path and not None, place it in output_dir
        if save_fig is not None and not os.path.isabs(save_fig):
            save_fig = os.path.join(output_dir, save_fig)
    
        # Choose the specific relocated aftershock source
        from copy import deepcopy
        seis = deepcopy(self.relocated_aftershock_source)
        if minlon is None or maxlon is None:
            minlon, maxlon = seis.lon.min(), seis.lon.max()
        if minlat is None or maxlat is None:
            minlat, maxlat = seis.lat.min(), seis.lat.max()
        seis.selectbox(minlon, maxlon, minlat, maxlat, depth=maxdepth, mindep=mindepth)
        x_values, y_values = seis.x, seis.y
    
        from sklearn.decomposition import PCA
        
        # Store selected aftershock positions
        self.selected_aftershock_positions = {
            'x': x_values,
            'y': y_values
        }
        
        if principal_direction_angle is None:
            # Perform PCA to find the principal direction of x_values and y_values
            pca = PCA(n_components=1)
            pca.fit(np.vstack((x_values, y_values)).T)
            principal_direction = pca.components_[0]
        else:
            principal_direction_rad = np.deg2rad(principal_direction_angle)
            principal_direction = np.array([np.cos(principal_direction_rad), np.sin(principal_direction_rad)])
        
        # Rotate data to align with the principal direction
        theta = -np.arctan2(principal_direction[1], principal_direction[0])
        ratation_direction = 'clockwise' if theta < 0 else 'counterclockwise'
        print(f'Rotating data by {abs(rad2deg(theta))} degrees {ratation_direction}.')
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        rotated_data = rotation_matrix @ np.vstack((x_values, y_values))
        
        if show_data_in_principal_direction:
            with sci_plot_style():
                plt.scatter(x_values, y_values, c='gray', label='Data')
                plt.scatter(rotated_data[0], rotated_data[1], c='blue', label='Rotated Data')
                plt.legend()
                plt.show()

        # Create a RegressionFitter instance
        fitter = RegressionFitter(rotated_data[0], rotated_data[1], degree=degree)
        
        if x_fit is None:
            if align_fit_with_rupture_trace_in_principal_direction:
                rotated_trace = rotation_matrix @ np.vstack((self.xf, self.yf))
                X_ = np.linspace(rotated_trace[0].min(), rotated_trace[0].max(), 100)
            else:
                X_ = np.linspace(rotated_data[0].min(), rotated_data[0].max(), 100)
        else:
            X_ = x_fit
        
        # Fit the model using the specified methods
        self.relociso_models = {}
        self.relociso_mses = {}
        self.relocisocurve_dict = {}
        self.isocurve_fitted_results = []  # Store fitted results for later plotting
        for method in methods:
            model, mse = fitter.fit_model(method, **fit_params.get(method, {}))
            self.relociso_models[method] = model
            self.relociso_mses[method] = mse
            y_plot = model.predict(X_[:, np.newaxis])
            
            # Rotate the fitted data back to the original direction
            rotated_back_data = np.linalg.inv(rotation_matrix) @ np.vstack((X_, y_plot))
            lon, lat = self.xy2ll(rotated_back_data[0], rotated_back_data[1])
            
            # Trim the curve
            flag = rotated_back_data[0] if trimaxis == 'x' else rotated_back_data[1]
            flag = lon if trimaxis == 'x' and not trimutm else flag
            flag = lat if trimaxis != 'x' and not trimutm else flag
            mask = np.ones_like(flag, np.bool_)
            mask = np.logical_and(mask, flag > trimst) if trimst is not None else mask
            mask = np.logical_and(mask, flag < trimed) if trimed is not None else mask
            lon, lat = lon[mask], lat[mask]
            X_masked, y_plot_masked = rotated_back_data[0][mask], rotated_back_data[1][mask]
            z = np.ones_like(lat) * (focusdepth if focusdepth is not None else np.quantile(seis.depth, 0.5))
            self.relocisocurve_dict[method] = np.vstack((X_masked, y_plot_masked, z)).T
            self.isocurve_fitted_results.append({
                'method': method,
                'X': X_masked,
                'y': y_plot_masked,
                'lon': lon,
                'lat': lat,
                'z': z,
                'mse': mse,
                'rotation_matrix': rotation_matrix,
                'isocurve': True  # Mark this result as an isocurve fit
            })
        
        # Perform PCA to find the principal direction of xf and yf
        pca_xf_yf = PCA(n_components=1)
        pca_xf_yf.fit(np.vstack((self.xf, self.yf)).T)
        principal_direction_xf_yf = pca_xf_yf.components_[0]

        # Ensure the principal direction is from the first point to the last point
        direction_vector = np.array([self.xf[-1] - self.xf[0], self.yf[-1] - self.yf[0]])
        if np.dot(principal_direction_xf_yf, direction_vector) < 0:
            principal_direction_xf_yf = -principal_direction_xf_yf
        
        print('Principal direction angle of the fault trace:', np.rad2deg(np.arctan2(principal_direction_xf_yf[1], principal_direction_xf_yf[0])))
        
        # Compare the principal directions with the rupture trace
        # Aim: keep the principal direction consistent with the rupture trace
        for result in self.isocurve_fitted_results:
            principal_direction_result = np.array([result['X'][-1] - result['X'][0], result['y'][-1] - result['y'][0]])
            if np.dot(principal_direction_xf_yf, principal_direction_result) < 0:
                result['X'] = result['X'][::-1]
                result['y'] = result['y'][::-1]
                result['lon'] = result['lon'][::-1]
                result['lat'] = result['lat'][::-1]

                self.relocisocurve_dict[result['method']] = self.relocisocurve_dict[result['method']][::-1, :]

        print('The focus depth to fit the isocurve is:', (focusdepth if focusdepth is not None else np.quantile(seis.depth, 0.5)))
        print('Fitting isocurve done.')
        if show:
            # plt.scatter(x_values, y_values)
            # for method in methods:
            #     plt.plot(self.isocurve_fitted_results[methods.index(method)]['X'], 
            #              self.isocurve_fitted_results[methods.index(method)]['y'], label=method)
            # plt.legend()
            # plt.show()
            self.plot_isocurve_fits(methods=methods, save_fig=save_fig, dpi=dpi, style=style,
                                    show_error_in_legend=show_error_in_legend, use_lon_lat=use_lon_lat,
                                    fontsize=fontsize, figsize=figsize, scatter_props=scatter_props, 
                                    show=show, custom_data=custom_data_for_plotting,
                                    legend_position=legend_position, equal_aspect=equal_aspect)
    
        if save2csi:
            print('Save to self.relocisocurve ...')
            y_plot = model.predict(X_[:, np.newaxis])
            x, y = X_, y_plot
            lon, lat = self.xy2ll(x, y)
            # Trim the curve
            flag = x if trimaxis == 'x' else y
            flag = lon if trimaxis == 'x' and not trimutm else flag
            flag = lat if trimaxis != 'x' and not trimutm else flag
            mask = np.ones_like(flag, np.bool_)
            mask = np.logical_and(mask, flag > trimst) if trimst is not None else mask
            mask = np.logical_and(mask, flag < trimed) if trimed is not None else mask
            x, y = x[mask], y[mask]
            z = np.ones_like(y) * (focusdepth if focusdepth is not None else np.quantile(seis.depth, 0.5))
            self.relocisocurve = np.vstack((x, y, z)).T
    
        # Save all model fitting results to iso_curve.geojson
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        for result in self.isocurve_fitted_results:
            feature = {
                "type": "Feature",
                "properties": {
                    "method": result['method'],
                    "mse": result['mse']
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[coord[0], coord[1], coord[2]] for coord in zip(result['lon'], result['lat'], result['z'])]
                }
            }
            geojson["features"].append(feature)
    
        geojson_file = os.path.join(output_dir, 'iso_curve.geojson')
        with open(geojson_file, 'w') as f:
            json.dump(geojson, f, indent=4)

    def plot_isocurve_fits(self, methods=None, save_fig=None, dpi=600, style=['science'],
                           show_error_in_legend=False, use_lon_lat=False, fontsize=None, 
                           figsize=None, scatter_props=None, show=True, custom_data=None,
                           equal_aspect=False, legend_position=None):
        """
        Plot the fitted isocurve models.
    
        Parameters:
        - methods: List of methods to plot. If None, plot all fitted models.
        - save_fig: Path to save the figure.
        - dpi: DPI for saving the figure.
        - style: Style for plotting.
        - show_error_in_legend: Whether to show error in legend.
        - use_lon_lat: Whether to use longitude and latitude for plotting.
        - fontsize: Font size for plotting.
        - figsize: Figure size for plotting.
        - scatter_props: Properties for scatter plot.
        - show: Whether to show the plot.
        - custom_data: Dictionary containing custom data for plotting. Keys are labels, values are (x, y) tuples.
        - equal_aspect: Whether to set equal aspect ratio for the plot.
        - legend_position: Tuple containing (bbox_to_anchor, loc) for the legend position. Default is None.
    
        Returns:
        None
        """
        from ..plottools import sci_plot_style, set_degree_formatter
    
        if methods is None:
            methods = [result['method'] for result in self.isocurve_fitted_results]
    
        # Use selected aftershock positions for initializing RegressionFitter
        x_values = self.selected_aftershock_positions['x']
        y_values = self.selected_aftershock_positions['y']
    
        if use_lon_lat:
            lon_values, lat_values = self.xy2ll(x_values, y_values)
            x_label = "Longitude"
            y_label = "Latitude"
        else:
            lon_values, lat_values = x_values, y_values
            x_label = "X (km)"
            y_label = "Y (km)"
    
        models = {method: self.relociso_models[method] for method in methods}
        mses = {method: self.relociso_mses[method] for method in methods}
        rotation_matrix = self.isocurve_fitted_results[0]['rotation_matrix']
        rotated_data = rotation_matrix @ np.vstack((x_values, y_values))
        X_rotated = np.linspace(rotated_data[0].min(), rotated_data[0].max(), 100)
        rotation_matrix = self.isocurve_fitted_results[0]['rotation_matrix']
        result_dict = {result['method']: result for result in self.isocurve_fitted_results}
    
        # Set default properties for plotting
        with sci_plot_style(style=style, fontsize=fontsize, figsize=figsize):
    
            line_width = 2
    
            if show:
                default_scatter_props = {'color': "black", 's': 30, 'alpha': 0.6}
                scatter_props = scatter_props if scatter_props is not None else {}
                scatter_props = {**default_scatter_props, **scatter_props}  # Update default properties with provided ones
    
                plt.scatter(lon_values, lat_values, **scatter_props)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
    
            fitter = RegressionFitter(x_values, y_values, degree=3)
            for method, model in models.items():
                # Rotate X_ to the principal direction
                y_plot_rotated = model.predict(X_rotated[:, np.newaxis])
                
                # Rotate the fitted data back to the original direction
                rotated_back_data = np.linalg.inv(rotation_matrix) @ np.vstack((X_rotated, y_plot_rotated))
                x_predict = rotated_back_data[0]
                y_predict = rotated_back_data[1]
                x_mask = (x_predict >= result_dict[method]['X'].min()) & (x_predict <= result_dict[method]['X'].max())
                y_mask = (y_predict >= result_dict[method]['y'].min()) & (y_predict <= result_dict[method]['y'].max())
                mask = x_mask & y_mask
                if use_lon_lat:
                    x_predict, y_predict = self.xy2ll(x_predict, y_predict)
                
                method_name = fitter.short_name_dict[method]
                label = method_name if not show_error_in_legend or mses is None else '%s: error = %.3f' % (method_name, mses[method])
                plt.plot(x_predict[mask], y_predict[mask], label=label, linewidth=line_width,
                         color=fitter.color_dict[method], linestyle=fitter.line_style_dict[method])
    
            # Plot custom data if provided
            if custom_data:
                for label, (x_custom, y_custom) in custom_data.items():
                    x_custom, y_custom = np.array(x_custom), np.array(y_custom)
                    if use_lon_lat:
                        x_custom, y_custom = self.xy2ll(x_custom, y_custom)
                    plt.plot(x_custom, y_custom, color='red', linestyle='-', linewidth=line_width, label=label)
            
            if use_lon_lat:
                set_degree_formatter(plt.gca(), axis='both')
    
            if equal_aspect:
                plt.gca().set_aspect('equal', adjustable='box')
    
            if show:
                if legend_position is not None:
                    bbox_to_anchor, loc = legend_position
                    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=0.)
                else:
                    plt.legend()
                if save_fig is not None:
                    plt.savefig(save_fig, dpi=dpi)
                plt.show()
    
    def combine_fitted_results(self, methods=['ols', 'theil_sen', 'ransac'], intersection_indices=None, 
                               pairwise_methods=None, split_points=None, 
                               sort_axis='x', sort_order='ascend', custom_coord_type='xy'):
        """
        Combine fitted results from multiple methods by finding the intersection points and merging the data.
    
        Parameters:
        - methods: List of fitting methods to use (default is ['ols', 'theil_sen', 'ransac']).
        - intersection_indices: List of indices to choose intersection points (default is None).
        - pairwise_methods: List of tuples specifying pairwise methods to find intersections (default is None).
        - split_points: List of precomputed split points (default is None).
        - sort_axis: Axis to sort the data ('x' or 'y'). Default is 'x'.
        - sort_order: Order to sort the data ('ascend' or 'descend'). Default is 'ascend'.
        - custom_coord_type: The type of coordinates to use for the combined data. Default is 'xy'.
    
        Returns:
        - custom_data: Dictionary containing combined data for plotting.
        """
        from shapely.geometry import LineString
    
        # Ensure the methods exist in the fitted results
        result = {res['method']: res for res in self.isocurve_fitted_results}
        for method in methods:
            if method not in result:
                raise ValueError(f"Method '{method}' not found in fitted results.")
    
        if intersection_indices is None:
            intersection_indices = [0] * (len(methods) - 1)  # Default to the first intersection point
    
        if len(intersection_indices) != len(methods) - 1:
            raise ValueError("Length of intersection_indices must be len(methods) - 1")
    
        combined_x = []
        combined_y = []
    
        sorted_results = {}
        for method in methods:
            x, y = result[method]['lon'], result[method]['lat']
            if sort_axis == 'x':
                sorted_indices = np.argsort(x) if sort_order == 'ascend' else np.argsort(x)[::-1]
            elif sort_axis == 'y':
                sorted_indices = np.argsort(y) if sort_order == 'ascend' else np.argsort(y)[::-1]
            else:
                raise ValueError("Invalid value for sort_axis. It should be 'x' or 'y'.")
            sorted_results[method] = {
                'lon': np.array(x)[sorted_indices],
                'lat': np.array(y)[sorted_indices]
            }
    
        if split_points is None:
            split_points = []
            if pairwise_methods is None:
                pairwise_methods = [(methods[i], methods[i + 1]) for i in range(len(methods) - 1)]
    
            for i, (method1, method2) in enumerate(pairwise_methods):
                x1, y1 = sorted_results[method1]['lon'], sorted_results[method1]['lat']
                x2, y2 = sorted_results[method2]['lon'], sorted_results[method2]['lat']
    
                # Create LineString objects for both sets of points
                line1 = LineString(np.column_stack((x1, y1)))
                line2 = LineString(np.column_stack((x2, y2)))
    
                # Find the intersection points
                intersection = line1.intersection(line2)
    
                # Check if the intersection is a point or MultiPoint
                if intersection.is_empty:
                    raise ValueError(f"No intersection found between the lines of methods '{method1}' and '{method2}'.")
                elif intersection.geom_type == 'Point':
                    split_points.append(intersection.x if sort_axis == 'x' else intersection.y)
                elif intersection.geom_type == 'MultiPoint':
                    intersection_points = list(intersection.geoms)
                    index = intersection_indices[i]
                    # Sort intersection points based on the specified axis and order
                    if sort_axis == 'x':
                        intersection_points.sort(key=lambda p: p.x, reverse=(sort_order == 'descend'))
                    else:
                        intersection_points.sort(key=lambda p: p.y, reverse=(sort_order == 'descend'))
                    index = intersection_indices[i]
                    if index < 0 or index >= len(intersection_points):
                        raise ValueError(f"Invalid intersection index: {index}")
                    split_points.append(intersection_points[index].x if sort_axis == 'x' else intersection_points[index].y)
                else:
                    raise ValueError(f"Unexpected intersection type: {intersection.geom_type}")
    
        # Combine the data based on split points
        for i, method in enumerate(methods):
            x, y = sorted_results[method]['lon'], sorted_results[method]['lat']
            if i == 0:
                # For the first method, take the left part
                if sort_order == 'ascend':
                    mask = np.array(x) <= split_points[0] if sort_axis == 'x' else np.array(y) <= split_points[0]
                else:
                    mask = np.array(x) >= split_points[0] if sort_axis == 'x' else np.array(y) >= split_points[0]
            elif i == len(methods) - 1:
                # For the last method, take the right part
                if sort_order == 'ascend':
                    mask = np.array(x) > split_points[-1] if sort_axis == 'x' else np.array(y) > split_points[-1]
                else:
                    mask = np.array(x) < split_points[-1] if sort_axis == 'x' else np.array(y) < split_points[-1]
            else:
                # For the middle methods, take the middle part
                if sort_order == 'ascend':
                    mask = (np.array(x) > split_points[i - 1]) & (np.array(x) <= split_points[i]) if sort_axis == 'x' else (np.array(y) > split_points[i - 1]) & (np.array(y) <= split_points[i])
                else:
                    mask = (np.array(x) < split_points[i - 1]) & (np.array(x) >= split_points[i]) if sort_axis == 'x' else (np.array(y) < split_points[i - 1]) & (np.array(y) >= split_points[i])
    
            combined_x.extend(np.array(x)[mask])
            combined_y.extend(np.array(y)[mask])
    
        # Sort the combined data by the specified axis
        combined_x = np.array(combined_x)
        combined_y = np.array(combined_y)
        if sort_axis == 'x':
            sorted_indices = np.argsort(combined_x) if sort_order == 'ascend' else np.argsort(combined_x)[::-1]
        elif sort_axis == 'y':
            sorted_indices = np.argsort(combined_y) if sort_order == 'ascend' else np.argsort(combined_y)[::-1]
        combined_x, combined_y = combined_x[sorted_indices], combined_y[sorted_indices]
    
        # Create the custom_data dictionary
        fitter = RegressionFitter(combined_x, combined_y, degree=3)
        method_names = [fitter.short_name_dict[method] for method in methods]
        if custom_coord_type == 'lonlat':
            custom_data = {
                '_'.join(method_names): (combined_x.tolist(), combined_y.tolist())
            }
        else:
            custom_x, custom_y = self.ll2xy(combined_x, combined_y)
            custom_data = {
                '_'.join(method_names): (custom_x.tolist(), custom_y.tolist())
            }
    
        # Add xyz coordinates to relocisocurve_dict
        z_combined = np.ones_like(combined_x) * result[methods[0]]['z'][0]
        combined_x, combined_y = self.ll2xy(combined_x, combined_y)
        self.relocisocurve_dict['_'.join(methods)] = np.vstack((combined_x, combined_y, z_combined)).T
    
        return custom_data

    def fit_and_plot_profiles(
        self, 
        profiles, 
        methods=['ols', 'theil_sen', 'ransac', 'huber', 'lasso',
                'ridge', 'elasticnet', 'quantile'],
        degree=1, 
        fit_params=None,
        show=True,
        nx=6, 
        style="whitegrid", 
        aspect_ratio=0.5, 
        fig_width="nature", 
        fig_height=None,
        save_fig=None,
        dpi=300,
        legend_nrows=1,  # Add a new parameter: the number of rows of the legend
        legend_height_ratio=0.2,  # Add a new parameter: the ratio of the legend height to the subplot height
        fit_indices=None,  # Add a new parameter: a dictionary of indices used for fitting the relocated aftershocks
        scatter_size=None  # Add a new parameter: the size of the scatter points
        ):

        if fit_params is None:
            fit_params = {}

        fig_width_mapping = {
            "nature": 7.48,  # Nature journal single column width
            "nature_double": 15.16,  # Nature journal double column width
            "nature_full": 22.86  # Nature journal full page width
        }
        line_width = 2
        fig_width = fig_width_mapping.get(fig_width, 7.48)  # Default to Nature journal single column width

        if fig_height is None:  # If the user did not provide a figure height
            import math
            ny = math.ceil(len(profiles) / nx)
            # Calculate the figure height based on the width, number of subplots, aspect ratio and legend height
            fig_height = fig_width * (ny / nx / aspect_ratio + legend_height_ratio)

        strike_dips = []
        if show:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial']
            plt.rcParams['font.size'] = 10
            from matplotlib import gridspec
            import seaborn as sns
            sns.set_theme(style=style)
            # Generate a color palette with the same number of colors as methods
            palette = sns.color_palette("hls", len(methods))
            # 创建一个GridSpec对象
            gs = gridspec.GridSpec(math.ceil(len(profiles) / nx) + 1, nx, height_ratios=[1]*math.ceil(len(profiles) / nx) + [legend_height_ratio])  # 额外添加一行用于放置图例
            fig = plt.figure(figsize=(fig_width, fig_height))

            # Create a legend subplot at the bottom
            ax_legend = plt.subplot(gs[-1, :])
            ax_legend.axis('off')  # Ensure the legend subplot does not have any axis
            # Create a list of subplots
            axs = [plt.subplot(gs[i]) for i in range(len(profiles))]

        for i, profiname in enumerate(profiles):
            profi = profiles[profiname]
            dist = profi['Distance']
            dep = -profi['Depth']
            
            # Check if fit_indices parameter exists and contains current profile name
            if fit_indices and profi['Name'] in fit_indices:
                indices = fit_indices[profi['Name']]
            else:
                # If not, use all indices
                indices = np.arange(len(dist))

            X, y = dep[indices], dist[indices]

            if show:
                # Plot the unselected points
                mask = np.ones(len(dist), dtype=bool)
                mask[indices] = False
                axs[i].scatter(dist[mask], dep[mask], color="#cfcfd2", alpha=0.6, label='Not used', s=scatter_size)

                # Plot the selected points
                if len(indices) > 0:
                    axs[i].scatter(dist[indices], dep[indices], color="black", alpha=0.6, label='Used', s=scatter_size)

            # Create a RegressionFitter instance
            fitter = RegressionFitter(X, y, degree=degree)
            dips = []
            mses = []
            for j, method in enumerate(methods):
                model, mse = fitter.fit_model(method, **fit_params.get(method, {}))

                # xfit = np.linspace(X.min(), X.max(), 2)
                xfit = np.linspace(dep.min(), dep.max(), 2)
                yfit = model.predict(xfit[:, np.newaxis])
                dx, dy = np.diff(xfit), np.diff(yfit)
                sign_dx = np.sign(dx)
                sign_dy = np.where(dy != 0, np.sign(dy), 1)
                # Keep the dip direction towards the right hand side of the strike
                dx, dy = sign_dy*np.abs(dx), sign_dx*np.abs(dy)
                dipi = np.degrees(np.arctan2(dx, dy))

                profi['dx'] = dx
                profi['dy'] = dy
                profi['aftershock_offset'] = yfit[0]
                profi['dip'] = dipi[0]
                profi['model'] = model
                dips.append(dipi[0])
                mses.append(mse)
                # print(f"Profile {i}, Method {method}, Dip: {dipi[0]}, MSE: {mse:.4f}")
                # print(X.min(), X.max(), dx, dy, dipi[0], mse, method)

                if show:
                    axs[i].plot(yfit, xfit, linewidth=line_width, color=palette[j], label=fitter.short_name_dict[method]) # method
                # f'Prof{i}'
                strike_dips.append([f'{profiname}', profi['Center'][0], profi['Center'][1], profi['strike'], dipi[0], mse, method])

        if show:
            # Add labels and titles
            handles, labels = axs[len(profiles)-1].get_legend_handles_labels()
            legend_ncols = math.ceil(len(handles) / legend_nrows)  # Calculate the number of columns for the legend
            ax_legend.legend(handles, labels, loc='center', ncol=legend_ncols)
            plt.tight_layout()
            # plt.legend()
            if save_fig is not None:
                plt.savefig(save_fig, dpi=dpi)
            plt.show()

        return strike_dips
    
    def fit_relocated_aftershocks_in_profiles_along_strike(
        self,  
        prof_wid=15, 
        prof_len=30, 
        methods=['ols', 'theil_sen', 'ransac', 'huber', 'lasso',
                'ridge', 'elasticnet', 'quantile'],
        degree=1,
        fit_params=None,
        show=False, 
        write2file=True,
        style="ticks", # whitegrid, darkgrid, dark, white, ticks
        aspect_ratio=0.5,
        scatter_size=None, # The size of the scatter points
        fig_width="nature_double",
        save_fig=None,
        dpi=300,
        legend_nrows=1,  # The number of rows of the legend
        legend_height_ratio=0.2,  # The ratio of the legend height to the subplot height
        min_seis_count=10,  # The minimum number of relocated aftershocks in a profile
        fit_indices=None,  # A dictionary of indices used for fitting the relocated aftershocks
        re_extract_profiles=False,  # Whether to re-extract profiles, default to False
        output_dir='.'  # The default directory to save files
    ):
        # Check if the relocated aftershock source is set
        if not hasattr(self, 'relocated_aftershock_source'):
            raise ValueError("The attribute 'relocated_aftershock_source' is not set. Please set 'relocated_aftershock_source' before calling this function.")
        seis = self.relocated_aftershock_source

        # Check if the top strike is set
        if not hasattr(self, 'top_strike'):
            raise ValueError("The attribute 'top_strike' is not set. Please calculate 'top_strike' before calling this function.")

        # Re-extract profiles if re_extract_profiles is True
        if re_extract_profiles:
            profiles = self.extract_profiles(prof_wid, prof_len, min_seis_count)
        else:
            # Otherwise, use the existing profiles
            profiles = self.profiles # list(self.profiles.values())

        # Arise an error if no profiles have more than min_seis_count relocated aftershocks
        if not profiles:
            raise Exception(f"No profiles have more than {min_seis_count} relocated aftershocks.")

        if save_fig is not None:
            # Check if save_fig contains a directory path
            if not os.path.dirname(save_fig):
                save_fig = os.path.join(output_dir, save_fig)
        strike_dips = self.fit_and_plot_profiles(profiles, methods, degree, fit_params, show, 
                                                    style=style, aspect_ratio=aspect_ratio, 
                                                    fig_width=fig_width, save_fig=save_fig, dpi=dpi,
                                                    legend_nrows=legend_nrows, legend_height_ratio=legend_height_ratio, 
                                                    fit_indices=fit_indices, scatter_size=scatter_size)

        if write2file:
            strike_dips = pd.DataFrame(strike_dips, columns='profile_index lon lat strike dip mse method'.split())
            output_file = os.path.join(output_dir, 'reloc_xydipfit.csv')
            strike_dips.to_csv(output_file, float_format='%.6f', index=False, header=True)

        # All Done
        return strike_dips
    
    def extract_profiles(self, prof_wid, prof_len, min_seis_count, center_coords=None):
        """
        Extract profiles.
    
        Parameters:
        prof_wid (int): Width of the profile.
        prof_len (int): Length of the profile.
        min_seis_count (int): Minimum number of relocated aftershocks in the profile.
        center_coords (numpy.ndarray, optional): Center coordinates. Default is None, in which case self.top_coords_ll is used as the center coordinates.
    
        Returns:
        list: A list containing profile information.
    
        Raises:
        Exception: If no profiles contain more than min_seis_count relocated aftershocks, an exception is raised.
        """
        # Check if self.top_strike exists
        if not hasattr(self, 'top_strike'):
            raise ValueError("The attribute 'top_strike' is not set. Please calculate 'top_strike' before calling this function.")
    
        # Azimuth is the length direction.
        # Keep azimuth angle pointing to the left side of the strike angle.
        azi = -(90 - self.top_strike)
        profiles = []
        
        # Check if center_coords parameter is provided
        if center_coords is None:
            center_coords = self.top_coords_ll
    
        top_lon, top_lat = center_coords[:, 0], center_coords[:, 1]
        for i, (ilonc, ilatc) in enumerate(zip(top_lon, top_lat)):
            prof_name = f'Profile_{i}'
            self.extract_profile(azi[i], ilonc, ilatc, prof_len, prof_wid, prof_name)
            
            profi = self.profiles[prof_name]
            profi['strike'] = self.top_strike[i]
            profi['Name'] = prof_name
            # Check the number of relocated aftershocks in the profile. If the number is less than min_seis_count, do not fit.
            if len(self.profiles[prof_name]['Lon']) > min_seis_count:
                profiles.append(profi)
            else:
                # If the number is less than min_seis_count, remove this profile from self.profiles
                del self.profiles[prof_name]
                # Keep consistent between self.profiles and relocated_aftershock_source.profiles
                del self.relocated_aftershock_source.profiles[prof_name]
        
        # If the profiles list is empty, raise an error warning
        if not profiles:
            raise Exception(f"No profiles have more than {min_seis_count} relocated aftershocks.")
        
        return self.profiles

    def interpolate_point_strike_along_isocurve(self, lon, lat, iso_strike, iso_lon, iso_lat):
        """
        Calculate the strike angle at a specific point for isocurve data.
    
        Parameters:
        - lon: Longitude of the point to calculate the strike angle.
        - lat: Latitude of the point to calculate the strike angle.
        - strike_data: Array of strike angles.
        - lon_data: Array of longitudes corresponding to the strike data.
        - lat_data: Array of latitudes corresponding to the strike data.
    
        Returns:
        - float: The calculated strike angle at the specified point.
        """
        from numpy import rad2deg, deg2rad
    
        # Find the nearest two points in the data
        dists = np.sqrt((iso_lon - lon)**2 + (iso_lat - lat)**2)
        nearest_indices = np.argsort(dists)[:2]
    
        # If the point is at an endpoint, return the strike of the nearest point
        if np.min(dists) == dists[0] or np.min(dists) == dists[-1]:
            return iso_strike[np.argmin(dists)]
    
        # Otherwise, return the average strike of the two nearest points
        strikes_rad = deg2rad(iso_strike[nearest_indices])
        average_strike_rad = np.arctan2(np.mean(np.sin(strikes_rad)), np.mean(np.cos(strikes_rad)))
        return rad2deg(average_strike_rad)
    
    def interpolate_point_strike_along_trace(self, lon, lat, discretized=True):
        """
        Calculate the strike angle at a specific point for trace data.
    
        Parameters:
        - lon: Longitude of the point to calculate the strike angle.
        - lat: Latitude of the point to calculate the strike angle.
        - discretized: If True, use the discretized trace data. Otherwise, use the original trace data.
    
        Returns:
        - float: The calculated strike angle at the specified point.
        """
        if discretized:
            if not hasattr(self, 'strikei') or self.strikei is None:
                raise ValueError("The discretized trace or its strikes are not calculated yet.")
            return self.interpolate_point_strike_along_isocurve(lon, lat, self.strikei, self.loni, self.lati)
        else:
            if not hasattr(self, 'strikef') or self.strikef is None:
                raise ValueError("The undiscretized trace's strikes are not calculated yet.")
            return self.interpolate_point_strike_along_isocurve(lon, lat, self.strikef, self.lon, self.lat)

    def calculate_top_strike(self, discretized=True):
            # Check if self.top_coords exists and is not None
            if not hasattr(self, 'top_coords') or self.top_coords is None:
                raise ValueError("The attribute 'top_coords' is not set. Please set 'top_coords' before calling this function.")

            # Initialize an empty list to store the strike values for each point
            top_strike = []

            # Iterate through each point in self.top_coords
            for coord in self.top_coords_ll:
                # Calculate the strike value for each point
                strike = self.interpolate_point_strike_along_trace(coord[0], coord[1], discretized=discretized)
                # Add the calculated strike value to the list
                top_strike.append(strike)

            # Convert the list to a numpy array and save it to self.top_strike
            self.top_strike = np.array(top_strike)
            return self.top_strike

    def calculate_isocurve_strike(self, x_coords, y_coords, calculate_strike_along_trace=True, is_lonlat=False):
        """
        Calculate the strike angle of the fault trace based on given coordinates.
    
        Parameters:
        - x_coords: Array of x coordinates.
        - y_coords: Array of y coordinates.
        - calculate_strike_along_trace (bool): If True, calculate the strike angle along the input order of the trace. 
            Otherwise, calculate the strike angle in the reverse order.
        - is_lonlat (bool): If True, the input coordinates are in longitude and latitude. Default is False.
    
        Returns:
        - np.ndarray: The strike angles in degrees.
        """
        from numpy import rad2deg, deg2rad
    
        # Calculate differences between consecutive coordinates
        if is_lonlat:
            x_coords, y_coords = self.ll2xy(x_coords, y_coords)
        x_diff = np.diff(x_coords)
        y_diff = np.diff(y_coords)
        
        # Calculate the strike angle for each segment and convert to azimuth
        segment_strike = 90 - rad2deg(np.arctan2(y_diff, x_diff))
        
        # Convert angles to radians
        segment_strike_rad = deg2rad(segment_strike)
        
        # Calculate the average strike angle at the midpoints
        average_strike_rad = np.arctan2(
            (np.sin(segment_strike_rad[:-1]) + np.sin(segment_strike_rad[1:])) / 2,
            (np.cos(segment_strike_rad[:-1]) + np.cos(segment_strike_rad[1:])) / 2
        )
        average_strike = rad2deg(average_strike_rad)
        
        # Concatenate the first and last segment strikes with the average strikes
        strike = np.concatenate(([segment_strike[0]], average_strike, [segment_strike[-1]]))
    
        # If calculating the strike in the reverse order, add 180 degrees
        if not calculate_strike_along_trace:
            strike += 180
    
        return strike
    
    def calculate_trace_strike(self, use_discretized_trace=True, calculate_strike_along_trace=True):
        """
        Calculate the strike angle of the fault trace. You can choose to calculate the strike angle along the input order of the trace or the reverse order.
    
        Parameters:
        - use_discretized_trace (bool): If True, use the discretized trace coordinates. Otherwise, use the original trace coordinates.
        - calculate_strike_along_trace (bool): If True, calculate the strike angle along the input order of the trace. Otherwise, calculate the strike angle in the reverse order.
    
        Returns:
        - np.ndarray: The strike angles in degrees.
        """
        if use_discretized_trace:
            if not hasattr(self, 'xi') or self.xi is None:
                raise ValueError("The discretized trace's coordinates are not calculated yet.")
            x_coords, y_coords = self.xi, self.yi
        else:
            x_coords, y_coords = self.xf, self.yf
    
        # Calculate the strike angles using the general method
        strike = self.calculate_isocurve_strike(x_coords, y_coords, calculate_strike_along_trace, is_lonlat=False)
    
        # Store the strike angles in the appropriate attribute
        if use_discretized_trace:
            self.strikei = strike
        else:
            self.strikef = strike
    
        return strike

    def extract_profile(self, azi, lonc, latc, prof_len, prof_wid, prof_name):
        # Check if the relocated aftershock source is set
        if not hasattr(self, 'relocated_aftershock_source'):
            raise ValueError("The attribute 'relocated_aftershock_source' is not set. Please set 'relocated_aftershock_source' before calling this function.")
        seis = self.relocated_aftershock_source

        seis.getprofile(prof_name, loncenter=lonc, latcenter=latc, length=prof_len, width=prof_wid, azimuth=azi)
        profile_info = seis.profiles[prof_name]
        self.profiles[prof_name] = profile_info
    
    def handle_buffer_nodes(self, xydip, buffer_nodes=None, buffer_radius=None, interpolation_axis='x', top_coords=None, update_ref=True):
        """
        Handle buffer nodes. If buffer nodes and radius are provided, find the buffer node segments and calculate the dip for each node segment.
    
        Parameters:
        xydip (DataFrame): The DataFrame containing the coordinates and dips.
        buffer_nodes (numpy.ndarray, optional): The buffer nodes. The shape is (n, 2) numpy array. Default is None.
        buffer_radius (float, optional): The buffer radius. Default is None. Unit is km.
        interpolation_axis (str, optional): The interpolation axis. It can be 'x' or 'y'. Default is 'x'.
        top_coords (numpy.ndarray, optional): The top coordinates to use. If None, self.top_coords[:, :-1] is used. Default is None.
        update_ref (bool, optional): If True, update the xydip reference. Default is True.
    
        Returns:
        DataFrame: The updated DataFrame containing the coordinates and dips. 
        If buffer nodes and radius are provided, it will contain the buffer node information.
        """
        if buffer_nodes is not None and buffer_radius is not None:
            # Convert buffer nodes' lat/lon to x, y coordinates
            buffer_nodes = np.array(buffer_nodes)
            buffer_nodes_ll = buffer_nodes
            buffer_xs, buffer_ys = self.ll2xy(buffer_nodes[:, 0], buffer_nodes[:, 1])
            buffer_nodes = np.hstack((buffer_xs[:, np.newaxis], buffer_ys[:, np.newaxis]))
    
            # Use provided top_coords or default to self.top_coords[:, :-1]
            if top_coords is None:
                top_coords = self.top_coords[:, :-1]
    
            # Find points within the buffer zone
            buffer_points = find_buffer_points(buffer_nodes, top_coords, buffer_radius)
            # print('buffer_nodes:', buffer_nodes)
            # print('buffer_points:', buffer_points)
    
            # Sort xydip based on the interpolation axis
            if interpolation_axis == 'x':
                sorted_xydip = xydip.sort_values(by='x', ascending=True).reset_index(drop=True)
                buffer_axis = 0  # x-axis
            else:
                sorted_xydip = xydip.sort_values(by='y', ascending=True).reset_index(drop=True)
                buffer_axis = 1  # y-axis
            # print('sorted_xydip:', sorted_xydip)
            # Iterate over each buffer node segment
            left_buffer_points = buffer_points[:, 0, buffer_axis]
            right_buffer_points = buffer_points[:, 1, buffer_axis]
    
            # Ensure xydip is sorted between left and right buffer points
            buffer_dfs = []
            for i, (left, right) in enumerate(zip(left_buffer_points, right_buffer_points)):
                # Filter xydip within the current buffer range
                if left < right:
                    left_candidates = sorted_xydip.loc[sorted_xydip[interpolation_axis] <= left]
                    if not left_candidates.empty:
                        left_index = left_candidates.index[-1]
                    else:
                        left_index = sorted_xydip.loc[sorted_xydip[interpolation_axis] >= left].index[0]
                    right_index = left_index + 1
                else:
                    left_index = sorted_xydip.loc[sorted_xydip[interpolation_axis] <= left].index[-1] + 1
                    if left_index >= len(sorted_xydip):
                        left_index = len(sorted_xydip) - 1
                    right_index = left_index - 1
                # print((sorted_xydip.loc[:, interpolation_axis] <= left))
                # print('left_index:', left_index, 'right_index:', right_index)
                # print('left:', left, 'right:', right)
                # Assign dips based on the sorted order
                left_dip = sorted_xydip.iloc[left_index]['dip']
                right_dip = sorted_xydip.iloc[right_index]['dip']
    
                # Create a new DataFrame containing the buffer nodes' coordinates and dips
                right_lon, right_lat = self.xy2ll(buffer_points[i, 1, 0], buffer_points[i, 1, 1])
                left_lon, left_lat = self.xy2ll(buffer_points[i, 0, 0], buffer_points[i, 0, 1])
                buffer_df = pd.DataFrame({
                    'x': [buffer_points[i, 0, 0], buffer_points[i, 1, 0]],
                    'y': [buffer_points[i, 0, 1], buffer_points[i, 1, 1]],
                    'lon': [left_lon, right_lon],
                    'lat': [left_lat, right_lat],
                    'dip': [left_dip, right_dip]
                })

                buffer_dfs.append(buffer_df)
            # Merge the new DataFrame into xydip and reset the index
            xydip = pd.concat([xydip] + buffer_dfs).drop_duplicates().reset_index(drop=True)

            # print('xydip:', xydip)
    
        if update_ref:
            self.xydip_ref = xydip
        return xydip

    def interpolate_top_dip_from_relocated_profile(
            self, 
            xydip, 
            is_utm=False, 
            discretization_interval=None,
            interpolation_axis='x', 
            save_to_file=False, 
            calculate_strike_along_trace=True,
            method='min_mse',  # optimal: min_use, ols, theil_sen, ransac, huber, lasso, ridge, elasticnet, quantile
            buffer_nodes=None,
            buffer_radius=None,
            update_xydip_ref=False,
            profiles_to_keep=None,
            profiles_to_remove=None
        ):
        """
        Interpolate the dip of the earthquake fault.
    
        Parameters:
        xydip (str, np.ndarray, pd.DataFrame): str is the path to a file containing x, y coordinates and dip angles. 
            (np.ndarray, pd.DataFrame) is the array or DataFrame containing the coordinates and dips.
        xydip_file: Path to a file containing x, y coordinates and dip angles.
        is_utm: If True, the x and y coordinates are in UTM. Otherwise, they are in geographic coordinates.
        discretization_interval: Interval for discretizing the trace.
        interpolation_axis: Axis used for interpolation, can be 'x' or 'y'.
        save_to_file: If True, save the results to a file.
        calculate_strike_along_trace: If True, calculate the strike along the fault trace.
        method: Method to select dips if multiple methods are available. Default is 'min_mse'.
        buffer_nodes: Coordinates of buffer nodes used to segment the top_coords, then adaptively interpolate the dip for each segment.
        buffer_radius: Radius of the buffer zone, used to maintain a transition radius for linear interpolation.
        update_xydip_ref: If True, update the xydip reference.
        profiles_to_keep: List of profile indices to keep. Default is None.
        profiles_to_remove: List of profile indices to remove. Default is None.
    
        Returns:
        pd.DataFrame: DataFrame containing the interpolated results.
    
        Todo: 
            * Remove the dependency on the fault trace (i.e., self.xf, self.yf or self.xi, self.yi).
        """
        # Discretize the trace for more accurate strike angle interpolation
        if discretization_interval is not None:
            self.discretize_top_coords(every=discretization_interval)
            # Calculate the x, y, lon, and lat coordinates of the discretized trace
            self.xi, self.yi = self.top_coords[:, 0], self.top_coords[:, 1]
            self.loni, self.lati = self.xy2ll(self.xi, self.yi)
        
        # Read coordinates and dips using the read_coordinates_and_dips function
        xydip = self.read_coordinates_and_dips(xydip, is_utm, method, profiles_to_keep, profiles_to_remove)
        # print('xydip:', xydip)
        # print('buffer_nodes:', buffer_nodes)
        # print('buffer_radius:', buffer_radius)
        # print('interpolation_axis:', interpolation_axis)
        # Handle buffer nodes and update self.xydip_ref
        xydip = self.handle_buffer_nodes(xydip, buffer_nodes, buffer_radius, interpolation_axis, update_ref=update_xydip_ref or not hasattr(self, 'xydip_ref'))
        
        # Save to csv file
        if save_to_file:
            xydip.to_csv(f'{self.name}_used.csv', index=False, header=True, float_format='%.6f')

        # Interpolation
        if interpolation_axis == 'x':
            x_values = xydip.x.values
            interpolated_x = self.top_coords[:, 0]
        else:
            x_values = xydip.y.values
            interpolated_x = self.top_coords[:, 1]
        
        # Sort values for interpolation
        indices = np.argsort(x_values)
        sorted_x_values = x_values[indices]
        sorted_dip_values = xydip.dip.values[indices]
        start_dip_fill = xydip.loc[indices[0], 'dip']
        end_dip_fill = xydip.loc[indices[-1], 'dip']
    
        # Create interpolation function
        interpolation_function = interp1d(sorted_x_values, sorted_dip_values, fill_value=(start_dip_fill, end_dip_fill), bounds_error=False)
        interpolated_dip = interpolation_function(interpolated_x)
    
        # Calculate strike
        if not hasattr(self, 'strikei') or self.strikei is None or self.strikei.size != self.xi.size:
            strikei = self.calculate_trace_strike(use_discretized_trace=True, calculate_strike_along_trace=calculate_strike_along_trace)
    
        top_strike = self.calculate_top_strike(discretized=True)
        lon, lat, strike, dip = self.top_coords_ll[:, 0], self.top_coords_ll[:, 1], top_strike, interpolated_dip
        interpolated_main = pd.DataFrame(np.vstack((lon, lat, strike, dip)).T, columns='lon lat strike dip'.split())
    
        # Adjust dip values greater than 90 degrees
        interpolated_main.loc[interpolated_main.dip > 90, 'dip'] = interpolated_main.loc[interpolated_main.dip > 90, 'dip'] - 180
    
        self.top_strike = top_strike
        self.top_dip = interpolated_main.loc[:, 'dip'].values  # interpolated_dip
    
        if save_to_file:
            interpolated_main.to_csv(f'{self.name}_Trace_Dip.csv', index=False, header=True, float_format='%.6f')
    
        # All Done
        return interpolated_main

    def interpolate_isocurve_dip_from_relocated_profile(
            self, 
            xydip,
            isocurve,
            discretization_interval=None,
            interpolation_axis='x', 
            calculate_strike_along_trace=True,
            method='min_mse',  # optimal: min_use, ols, theil_sen, ransac, huber, lasso, ridge, elasticnet, quantile
            buffer_nodes=None,
            buffer_radius=None,
            profiles_to_keep=None,
            profiles_to_remove=None
        ):
        """
        Interpolate the dip of the earthquake fault.
    
        Parameters:
        xydip (str, np.ndarray, pd.DataFrame): str is the path to a file containing x, y coordinates and dip angles. 
            (np.ndarray, pd.DataFrame) is the array or DataFrame containing the coordinates and dips.
        isocurve (str, np.ndarray, pd.DataFrame): str is the path to a file containing isocurve coordinates. 
            (np.ndarray, pd.DataFrame) is the array or DataFrame containing the isocurve coordinates.
        * coordinates in xydip and isocurve should be in the same coordinate system, i.e., both in lon/lat coordinates.
        discretization_interval: Interval for discretizing the isocurve. unit is km. if negative integer, it means the number of segments.
        interpolation_axis: Axis used for interpolation, can be 'x' or 'y'.
        save_to_file: If True, save the results to a file.
        calculate_strike_along_trace: If True, calculate the strike along the fault trace.
        method: Method to select dips if multiple methods are available. Default is 'min_mse'.
        buffer_nodes: Coordinates of buffer nodes used to segment the top_coords, then adaptively interpolate the dip for each segment.
        buffer_radius: Radius of the buffer zone, used to maintain a transition radius for linear interpolation.
        profiles_to_keep: List of profile indices to keep. Default is None.
        profiles_to_remove: List of profile indices to remove. Default is None.
    
        Returns:
        pd.DataFrame: interpolated_dip with lon, lat, strike, dip columns in DataFrame format.
    
        Todo: 
            * Remove the dependency on the fault trace (i.e., self.xf, self.yf or self.xi, self.yi).
        """
        # Read coordinates and dips using the read_coordinates_and_dips function
        xydip = self.read_coordinates_and_dips(xydip, False, method, profiles_to_keep, profiles_to_remove)

        # Handle buffer nodes and update self.xydip_ref
        xydip = self.handle_buffer_nodes(xydip, buffer_nodes, buffer_radius, interpolation_axis, update_ref=False)
        # Save to csv file
        xydip.to_csv(f'{self.name}_used_isocurve.csv', index=False, header=True, float_format='%.6f')

        # Discretize the isocurve for more accurate strike angle interpolation or not
        if isinstance(isocurve, str):
            isocurve = pd.read_csv(isocurve, comment='#', header=0)
        elif isinstance(isocurve, np.ndarray):
            columns = 'lon lat'.split() if isocurve.shape[1] == 2 else 'lon lat depth'.split()
            isocurve = pd.DataFrame(isocurve, columns=columns)
        if discretization_interval is not None:
            x, y = self.ll2xy(isocurve.lon.values, isocurve.lat.values)
            iso_xyz = np.vstack((x, y, isocurve.depth.values)).T
            if discretization_interval > 0:
                xyi, _ = self.discretize_coords(iso_xyz, discretization_interval)
            else:
                xyi, _ = self.discretize_coords(iso_xyz, num_segments=-discretization_interval)
        else:
            xi, yi = self.ll2xy(isocurve.lon.values, isocurve.lat.values)
            xyi = np.vstack((xi, yi)).T
        
        # Interpolation
        if interpolation_axis == 'x':
            x_values = xydip.x.values
            interpolated_x = xyi[:, 0]
        else:
            x_values = xydip.y.values
            interpolated_x = xyi[:, 1]
        
        # Sort values for interpolation
        indices = np.argsort(x_values)
        sorted_x_values = x_values[indices]
        sorted_dip_values = xydip.dip.values[indices]
        start_dip_fill = xydip.loc[indices[0], 'dip']
        end_dip_fill = xydip.loc[indices[-1], 'dip']
    
        # Create interpolation function
        interpolation_function = interp1d(sorted_x_values, sorted_dip_values, fill_value=(start_dip_fill, end_dip_fill), bounds_error=False)
        interpolated_dip = interpolation_function(interpolated_x)
    
        # Calculate strike
        iso_strike = self.calculate_isocurve_strike(x_coords=xyi[:, 0], y_coords=xyi[:, 1], calculate_strike_along_trace=calculate_strike_along_trace)

        lon, lat = self.xy2ll(xyi[:, 0], xyi[:, 1])
        strike, dip = iso_strike, interpolated_dip
        interpolated_main = pd.DataFrame(np.vstack((lon, lat, strike, dip)).T, columns='lon lat strike dip'.split())
    
        # Adjust dip values greater than 90 degrees
        interpolated_main.loc[interpolated_main.dip > 90, 'dip'] = interpolated_main.loc[interpolated_main.dip > 90, 'dip'] - 180
    
        # All Done
        return interpolated_main

    def read_coordinates_and_dips(
            self, xydip, is_utm=False, method='min_mse', 
            profiles_to_keep=None, profiles_to_remove=None):
        """
        Read coordinates and dips from provided arrays or a file, and convert between UTM and geographic coordinates if necessary.
    
        Parameters:
        xydip (str, np.ndarray, pd.DataFrame): str is the path to a file containing x, y coordinates and dip angles. 
            (np.ndarray, pd.DataFrame) is the array or DataFrame containing the coordinates and dips.
        is_utm (bool, optional): If True, coordinates are in UTM. Otherwise, they are in geographic coordinates. Default is False.
        method (str or list, optional): Method(s) to select dips if multiple methods are available. Default is 'min_mse'.
        profiles_to_keep (list, optional): List of profile indices to keep. Default is None.
        profiles_to_remove (list, optional): List of profile indices to remove. Default is None.
    
        Returns:
        pd.DataFrame: DataFrame containing coordinates and dips, with additional columns for converted coordinates.
        """
        import pandas as pd
        import numpy as np
    
        def convert_coords(df, is_utm):
            if is_utm:
                lon, lat = self.xy2ll(df.x.values, df.y.values)
                df['lon'] = lon
                df['lat'] = lat
            else:
                x, y = self.ll2xy(df.lon.values, df.lat.values)
                df['x'] = x
                df['y'] = y
            return df

        if isinstance(xydip, np.ndarray):
            columns = ['x', 'y', 'dip'] if is_utm else ['lon', 'lat', 'dip']
            xydip = pd.DataFrame(xydip, columns=columns)
            xydip = convert_coords(xydip, is_utm)
        elif isinstance(xydip, pd.DataFrame):
            xydip = convert_coords(xydip, is_utm)
        elif isinstance(xydip, str):
            xydip = pd.read_csv(xydip, comment='#', header=0)
            xydip['original_order'] = range(len(xydip))
            xydip = convert_coords(xydip, is_utm)
    
            # Ensure profiles_to_keep and profiles_to_remove are not used simultaneously
            if profiles_to_keep is not None and profiles_to_remove is not None:
                raise ValueError("Cannot use profiles_to_keep and profiles_to_remove simultaneously.")
    
            # Convert numeric profiles to string format
            if profiles_to_keep is not None:
                profiles_to_keep = [f'Profile_{p}' if isinstance(p, int) else p for p in profiles_to_keep]
            if profiles_to_remove is not None:
                profiles_to_remove = [f'Profile_{p}' if isinstance(p, int) else p for p in profiles_to_remove]
    
            # Filter profiles to keep or remove
            if profiles_to_keep is not None:
                profiles_to_remove = xydip[~xydip['profile_index'].isin(profiles_to_keep)]['profile_index'].unique()
                xydip = xydip[xydip['profile_index'].isin(profiles_to_keep)]
            if profiles_to_remove is not None:
                profiles_to_remove = xydip[xydip['profile_index'].isin(profiles_to_remove)]['profile_index'].unique()
                xydip = xydip[~xydip['profile_index'].isin(profiles_to_remove)]
    
            # Remove profiles from self.profiles and self.relocated_aftershock_source.profiles
            if profiles_to_remove is not None:
                for prof_name in profiles_to_remove:
                    if prof_name in self.profiles:
                        del self.profiles[prof_name]
                    if prof_name in self.relocated_aftershock_source.profiles:
                        del self.relocated_aftershock_source.profiles[prof_name]
    
            if 'profile_index' in xydip.columns and 'mse' in xydip.columns:
                # Validate the method
                valid_methods = xydip['method'].unique().tolist() + ['min_mse']
                if isinstance(method, str):
                    method = [method] * len(xydip['profile_index'].unique())
                if len(method) != len(xydip['profile_index'].unique()):
                    raise ValueError(f"The length of 'method' must match the number of unique profiles {len(xydip['profile_index'].unique())}.")
                if not all(m in valid_methods for m in method):
                    raise ValueError(f"All methods must be in {valid_methods}")
        
                # Create a dictionary for profile-specific methods
                profile_methods = dict(zip(xydip['profile_index'].unique(), method))
        
                # Select dips based on the specified method for each profile
                selected_rows = []
                for profile, method in profile_methods.items():
                    profile_data = xydip[xydip['profile_index'] == profile]
                    if method == 'min_mse':
                        selected_row = profile_data.loc[profile_data['mse'].idxmin()]
                    else:
                        selected_row = profile_data[profile_data['method'] == method].iloc[0]
                    selected_rows.append(selected_row)
                xydip = pd.DataFrame(selected_rows)
        
                # Sort by original order and reset index
                xydip = xydip.sort_values('original_order').reset_index(drop=True)
    
        # Ensure dip values are within the range 0 to 180 degrees
        xydip.loc[xydip.dip < 0, 'dip'] += 180
    
        return xydip
    
    def generate_top_bottom_from_nonlinear_soln(self, clon=None, clat=None, cdepth=None, 
                                                strike=None, dip=None, length=None, width=None, 
                                                top=None, depth=None, center_point_type='top_center', custom_length=None):
        """
        Generate the top and bottom coordinates of the fault trace from the nonlinear solution.
    
        Parameters:
        clon (float): The longitude of the center point of the top line.
        clat (float): The latitude of the center point of the top line.
        cdepth (float): The depth of the center point of the top line.
        strike (float): The strike angle of the fault patch. unit: degree.
        dip (float): The dip angle of the fault patch. unit: degree.
        length (float): The length of the fault patch.
        width (float): The width of the fault patch.
        top (float): The top depth of the fault patch.
        depth (float): The bottom depth of the fault patch.
        center_point_type (str): The type of the center point. Options are 'center', 'top_center', 'top_neg_end', 'top_pos_end'.
        custom_length (tuple): A tuple (neg_length, pos_length) to set the negative and positive lengths along the strike direction.
            Where 'pos' and 'neg' represent the positive and negative directions along the strike, respectively.
    
        Returns:
        top_coords: The top coordinates of the fault patch.
        bottom_coords: The bottom coordinates of the fault patch.
        """
        from numpy import deg2rad, sin, cos, tan
    
        if any(param is None for param in [clon, clat, cdepth, strike, dip]):
            raise ValueError("Please provide all the required parameters.")
        
        # Convert the strike and dip angles to radians
        str_rad = deg2rad(90 - strike)
        dip_rad = deg2rad(dip)
        
        # top or self.top, at least one of them should be provided
        if top is None:
            assert hasattr(self, 'top') and self.top is not None, "Please provide the top depth of the fault patch."
            top = self.top
    
        # width or depth/self.depth, at least one of them should be provided； if they are both provided, depth/self.depth will be used
        if width is None:
            assert depth is not None or (hasattr(self, 'depth') and self.depth is not None), "Please provide the width or depth of the fault patch."
            depth = depth if depth is not None else self.depth
            width = (depth - top) / sin(dip_rad)
        else:
            if depth is None:
                depth = self.depth if hasattr(self, 'depth') and self.depth is not None else width * sin(dip_rad) + top
    
        self.depth = depth
        
        # Calculate the top two end points of the fault patch
        cx, cy = self.ll2xy(clon, clat)
        cxy_trans = (cx + 1.j*cy) * np.exp(1.j*-str_rad)
    
        if custom_length:
            neg_length, pos_length = custom_length
            length = neg_length + pos_length
        else:
            neg_length = pos_length = length / 2.0
    
        if center_point_type == 'top_center':
            cxy_trans_neg = cxy_trans - neg_length
            cxy_trans_pos = cxy_trans + pos_length
        elif center_point_type == 'top_neg_end':
            cxy_trans_neg = cxy_trans
            cxy_trans_pos = cxy_trans_neg + length
        elif center_point_type == 'top_pos_end':
            cxy_trans_pos = cxy_trans
            cxy_trans_neg = cxy_trans_pos - length   
        elif center_point_type == 'center':
            # To be tested
            half_width = (depth - top)/sin(dip_rad)/2.0
            cxy_trans_neg = cxy_trans + (-neg_length + 1.j*half_width)
            cxy_trans_pos = cxy_trans + (pos_length + 1.j*half_width)
            cdepth = cdepth - half_width * sin(dip_rad)
    
        # top
        top_offset = (cdepth - top) / tan(dip_rad)
        cxy_top_trans_neg = cxy_trans_neg + top_offset*1.j
        cxy_top_trans_pos = cxy_trans_pos + top_offset*1.j
        cxy_top_neg = cxy_top_trans_neg * np.exp(1.j*str_rad)
        cxy_top_pos = cxy_top_trans_pos * np.exp(1.j*str_rad)
        top_coords = np.array([[cxy_top_neg.real, cxy_top_neg.imag, top], [cxy_top_pos.real, cxy_top_pos.imag, top]])
        top_coords = self.set_top_coords(top_coords, lonlat=False)
        # bottom
        bottom_offset = (cdepth - depth) / tan(dip_rad)
        cxy_bottom_trans_neg = cxy_trans_neg + bottom_offset*1.j
        cxy_bottom_trans_pos = cxy_trans_pos + bottom_offset*1.j
        cxy_bottom_neg = cxy_bottom_trans_neg * np.exp(1.j*str_rad)
        cxy_bottom_pos = cxy_bottom_trans_pos * np.exp(1.j*str_rad)
        bottom_coords = np.array([[cxy_bottom_neg.real, cxy_bottom_neg.imag, depth], [cxy_bottom_pos.real, cxy_bottom_pos.imag, depth]])
        bottom_coords = self.set_bottom_coords(bottom_coords, lonlat=False)
    
        # All Done
        return top_coords, bottom_coords

    def generate_bottom_from_segmented_relocated_dips(
            self, 
            xydip=None, 
            fault_depth=None,
            update_self=True,
            is_utm=False,
            discretization_interval=None,
            interpolation_axis='x', 
            calculate_strike_along_trace=True,
            method='min_mse',  
            buffer_nodes=None,
            buffer_radius=None,
            profiles_to_keep=None,
            profiles_to_remove=None,
            reinterpolate=True,
            use_average_strike=False,  # New parameter to use average strike direction
            average_strike_source='pca',  # New parameter to specify the source of average strike direction
            user_direction_angle=None,  # New parameter for user input direction angle
            verbose=False
        ):
        """
        Generate the bottom of the fault based on the dip angles determined from the relocated aftershock profile segments.
        This function mainly includes two steps:
        1. interpolate_dip_from_relocated_profile: Interpolate segment dip angles to trace densification points.
        2. make_bottom_from_reloc_dips: Translate to determine the bottom edge.

        Parameters:
        * xydip (str, np.ndarray, pd.DataFrame): str is the path to a file containing x, y coordinates and dip angles. 
            (np.ndarray, pd.DataFrame) is the array or DataFrame containing the coordinates and dips.
            default is None.
        * fault_depth: Depth of the fault. If None, self.depth will be used.
        * update_self: Whether to update the instance variables with the calculated coordinates. Default is True.
        * discretization_interval: Interval for discretizing the trace.
        * is_utm: If True, the x and y coordinates are in UTM. Otherwise, they are in geographic coordinates.
        * interpolation_axis: Axis used for interpolation, can be 'x' or 'y'.
        * calculate_strike_along_trace: If True, calculate the strike along the fault trace.
        * method: Method to select dips if multiple methods are available. Default is 'min_mse'.
        * buffer_nodes: Coordinates of buffer nodes used to segment the top_coords, then adaptively interpolate the dip for each segment.
        * buffer_radius: Radius of the buffer zone, used to maintain a transition radius for linear interpolation.
        * profiles_to_keep: List of profile indices to keep. Default is None.
        * profiles_to_remove: List of profile indices to remove. Default is None.
        * reinterpolate: Whether to reinterpolate the dip angles. Default is True.
        * use_average_strike: Whether to use average strike direction. Default is False.
        * average_strike_source: Source of average strike direction, can be 'pca' or 'user'. Default is 'pca'.
        * user_direction_angle: User input direction angle in degrees. Default is None.
        * verbose: Whether to print verbose output. Default is False.

        Returns:
        None. However, the function updates the following instance variables:
        * bottom_coords: UTM coordinates of the bottom.
        * bottom_coords_ll: Latitude and longitude coordinates of the bottom.
        """
        from numpy import deg2rad, sin, cos
        import os
    
        missing_depth_info = [attr for attr in ['depth', 'top'] if not hasattr(self, attr) or getattr(self, attr) is None]
        if missing_depth_info:
            raise ValueError(f"Please set the {', '.join(missing_depth_info)} attribute(s) before calling this function.")
    
        # Prepare Information
        fault_depth = fault_depth if fault_depth is not None else self.depth
    
        if xydip is not None:
            if reinterpolate:
                # Interpolate dip angles along the top coordinates
                if isinstance(xydip, np.ndarray):
                    if xydip.shape[1] == 4:
                        xydip = pd.DataFrame(xydip, columns='lon lat strike dip'.split())
                interpolated_dip_info = self.interpolate_top_dip_from_relocated_profile(
                    xydip=xydip,
                    is_utm=is_utm,
                    discretization_interval=discretization_interval,
                    interpolation_axis=interpolation_axis,
                    save_to_file=False,
                    calculate_strike_along_trace=calculate_strike_along_trace,
                    method=method,
                    buffer_nodes=buffer_nodes,
                    buffer_radius=buffer_radius,
                    update_xydip_ref=False,
                    profiles_to_keep=profiles_to_keep,
                    profiles_to_remove=profiles_to_remove
                )
                dip_info = interpolated_dip_info
            else:
                # Extract the dip angle information.
                if isinstance(xydip, str):
                    if not os.path.isfile(xydip):
                        raise ValueError(f"The file {xydip} does not exist.")
                    # Load dip angle information from file
                    dip_info = pd.read_csv(xydip, comment='#', header=0)
                elif isinstance(xydip, np.ndarray):
                    if xydip.shape[1] != 4:
                        raise ValueError("dip array must have four columns: lon, lat, strike, dip.")
                    dip_info = pd.DataFrame(xydip, columns=['lon', 'lat', 'strike', 'dip'])
                elif isinstance(xydip, pd.DataFrame):
                    if xydip.shape[1] != 4:
                        raise ValueError("dip DataFrame must have four columns: lon, lat, strike, dip.")
                    dip_info = xydip
                else:
                    raise ValueError("dip must be a file path or a numpy array with lon, lat, strike, dip.")
        else:
            # Check if self.top_dip exists and is not None
            if not hasattr(self, 'top_dip') or self.top_dip is None:
                raise ValueError("The attribute 'top_dip' is not set. Please set 'top_dip' before calling this function.")
    
            # 从self.top_coords_ll, self.top_strike和self.top_dip中提取数据
            lon, lat = self.top_coords_ll[:, 0], self.top_coords_ll[:, 1]
            strike, dip = self.top_strike, self.top_dip
            dip_info = pd.DataFrame(np.vstack((lon, lat, strike, dip)).T, columns='lon lat strike dip'.split())
    
        dip_info['strike_rad'] = deg2rad(dip_info.strike)
        dip_info['dip_rad'] = deg2rad(dip_info.dip)
    
        x, y = self.ll2xy(dip_info.lon.values, dip_info.lat.values)
        dip_info['xproj'] = x
        dip_info['yproj'] = y
    
        # Calculate the average strike direction if required
        strike_direction = np.array([x[-1] - x[0], y[-1] - y[0]])
        if use_average_strike:
            if average_strike_source == 'pca':
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                pca.fit(np.vstack((x, y)).T)
                principal_direction = pca.components_[0]
                average_strike_rad = np.arctan2(principal_direction[1], principal_direction[0])
                average_strike_rad = np.pi/2.0 - average_strike_rad
                # Ensure the strike direction is consistent with the principal direction
                if np.dot(principal_direction, strike_direction) < 0:
                    average_strike_rad += np.pi
            elif average_strike_source == 'user' and user_direction_angle is not None:
                average_strike_rad = deg2rad(user_direction_angle)
                if np.dot([cos(average_strike_rad), sin(average_strike_rad)], strike_direction) < 0:
                    raise ValueError("The user direction angle is not consistent with the strike direction.")
            else:
                raise ValueError("Invalid average_strike_source or user_direction_angle not provided.")
            strike_rad = average_strike_rad
            if verbose:
                print(f"Average strike direction: {np.rad2deg(strike_rad):.2f}")
        else:
            strike_rad = dip_info.strike_rad.values
        
        # Dip angle transfer to 0~90, and strike angle transfer in order to make dip right hand is positive
        negative_dip_flag = dip_info.dip_rad < 0
        
        # Adjust strike_rad for negative dips
        dip_info.loc[:, 'strike_rad'] = strike_rad
        dip_info.loc[negative_dip_flag, 'strike_rad'] += np.pi
        dip_info.loc[negative_dip_flag, 'dip_rad'] = -dip_info.loc[negative_dip_flag, 'dip_rad']
        
        # Ensure strike_rad is within 0-2π (0-360 degrees)
        dip_info.loc[:, 'strike_rad'] = np.mod(dip_info['strike_rad'], 2 * np.pi)
        dip_info.loc[:, 'strike'] = np.rad2deg(dip_info['strike_rad'])
        strike_rad = dip_info.strike_rad.values

        # update bottom_coords
        dip_rad = dip_info.dip_rad.values
        width = ((fault_depth - self.top)/sin(dip_rad)).reshape(-1, 1)
        old_coords = np.vstack((x, y, np.ones_like(y)*self.top)).T 
        dip_x = cos(dip_rad) * cos(-strike_rad)
        dip_y = cos(dip_rad) * sin(-strike_rad)
        dip_z = sin(dip_rad)
        dip_vector = np.vstack((dip_x, dip_y, dip_z)).T 
        bottom_coords = old_coords + dip_vector*width

        sort_order = np.argsort(bottom_coords[:, 0] if interpolation_axis == 'x' else bottom_coords[:, 1])
        bottom_coords = bottom_coords[sort_order, :]
        if np.dot([bottom_coords[-1, 0]-bottom_coords[0, 0], bottom_coords[-1, 1]-bottom_coords[0, 1]], strike_direction) < 0:
            bottom_coords = bottom_coords[::-1, :]
    
        if update_self:
            self.set_bottom_coords(bottom_coords, lonlat=False)
    
        return bottom_coords

    def generate_bottom_from_relocated_aftershock_iso_depth_curve(self, 
                                                        fault_depth=None, 
                                                        update_self=True,
                                                        method='ols'):
        """
        Generate the fault bottom coordinates from the relocated aftershock isodepth curve.
    
        Parameters:
        fault_depth: The depth of the fault. If None, self.depth will be used.
        update_self: Whether to update self.bottom_coords and self.bottom_coords_ll with the new bottom coordinates. Default is True.
        method: The fitting method to use. Default is 'ols'.
    
        Returns:
        bottom_coords_ll: The bottom coordinates of the fault in (longitude, latitude, depth) format.
        """
        # Check if the method exists in relocisocurve_dict
        if method not in self.relocisocurve_dict:
            raise ValueError(f"Method '{method}' not found in relocisocurve_dict. Available methods are: {list(self.relocisocurve_dict.keys())}")
    
        # Get the xyz coordinates for the specified method
        xyz_coords = self.relocisocurve_dict[method]
    
        # Update self.relocisocurve with the selected method's coordinates
        self.relocisocurve = xyz_coords

        # Save to csv file 
        lon, lat = self.xy2ll(xyz_coords[:, 0], xyz_coords[:, 1])
        xyz_coords_ll = np.vstack((lon, lat, xyz_coords[:, -1])).T
        pd.DataFrame(xyz_coords_ll, columns='lon lat depth'.split()).to_csv(f'{self.name}_relocisocurve_used.csv', index=False, header=True, float_format='%.6f')
    
        # Generate the bottom coordinates from the relocated iso depth curve
        bottom_coords_ll = self.skin_curve_to_bottom(depth_extend=fault_depth, use_relocisocurve=True, update_self=update_self)
    
        return bottom_coords_ll
    
    def generate_top_bottom_from_isodepth_and_dip(
            self, 
            isodepth, 
            xydip, 
            top_depth=None, 
            bottom_depth=None, 
            update_self=True,
            discretization_interval=None,
            interpolation_axis='x', 
            calculate_strike_along_trace=True,
            method='min_mse',  
            buffer_nodes=None,
            buffer_radius=None,
            profiles_to_keep=None,
            profiles_to_remove=None,
            reinterpolate=False
        ):
        """
        Generate the top and bottom coordinates based on the isodepth curve and dip angles.
    
        Parameters:
        - isodepth: Path to the file containing isodepth curve information or a numpy array with lon, lat, depth.
        - xydip: Path to the file containing dip angle information or a numpy array with lon, lat, strike, dip. angle unit: degree.
        - top_depth: Depth of the top. If None, self.top will be used.
        - bottom_depth: Depth of the bottom. If None, self.depth will be used.
        - update_self: Whether to update the instance variables with the calculated coordinates. Default is True.
        - discretization_interval: Interval for discretizing the isocurve. unit is km. if negative integer, it means the number of segments.
        - interpolation_axis: Axis used for interpolation, can be 'x' or 'y'.
        - calculate_strike_along_trace: If True, calculate the strike along the fault trace.
        - method: Method to select dips if multiple methods are available. Default is 'min_mse'.
        - buffer_nodes: Coordinates of buffer nodes used to segment the top_coords, then adaptively interpolate the dip for each segment.
        - buffer_radius: Radius of the buffer zone, used to maintain a transition radius for linear interpolation.
        - profiles_to_keep: List of profile indices to keep. Default is None.
        - profiles_to_remove: List of profile indices to remove. Default is None.
        - reinterpolate: Whether to reinterpolate the dip angles. Default is False.
    
        Returns:
        - top_coords: The top coordinates of the fault.
        - bottom_coords: The bottom coordinates of the fault.
        """
        from numpy import deg2rad, sin, cos
        import os
    
        # Extract the iso-depth curve information.
        if isinstance(isodepth, str):
            if not os.path.isfile(isodepth):
                raise ValueError(f"The file {isodepth} does not exist.")
            # Load isodepth curve information from file
            isodepth_info = pd.read_csv(isodepth, comment='#', header=0)
        elif isinstance(isodepth, np.ndarray):
            if isodepth.shape[1] != 3:
                raise ValueError("isodepth array must have three columns: lon, lat, depth.")
            isodepth_info = pd.DataFrame(isodepth, columns=['lon', 'lat', 'depth'])
        elif isinstance(isodepth, pd.DataFrame):
            if isodepth.shape[1] != 3:
                raise ValueError("isodepth DataFrame must have three columns: lon, lat, depth.")
            isodepth_info = isodepth
        else:
            raise ValueError("isodepth must be a file path or a numpy array with lon, lat, depth.")

        # Extract the dip angle information.
        if reinterpolate:
            # Interpolate dip angles along the isodepth curve
            if isinstance(xydip, np.ndarray):
                if xydip.shape[1] == 4:
                    xydip = pd.DataFrame(xydip, columns='lon lat strike dip'.split())
            interpolated_dip_info = self.interpolate_isocurve_dip_from_relocated_profile(
                xydip=xydip,
                isocurve=isodepth_info,
                discretization_interval=discretization_interval,
                interpolation_axis=interpolation_axis,
                calculate_strike_along_trace=calculate_strike_along_trace,
                method=method,
                buffer_nodes=buffer_nodes,
                buffer_radius=buffer_radius,
                profiles_to_keep=profiles_to_keep,
                profiles_to_remove=profiles_to_remove
            )
            dip_info = interpolated_dip_info
            lon, lat = dip_info.lon.values, dip_info.lat.values
            depth = np.ones_like(lon)*isodepth_info.depth.values[0]
            isodepth_info = pd.DataFrame(np.vstack((lon, lat, depth)).T, columns='lon lat depth'.split())
        else:
            # Extract the dip angle information.
            if isinstance(xydip, str):
                if not os.path.isfile(xydip):
                    raise ValueError(f"The file {xydip} does not exist.")
                # Load dip angle information from file
                dip_info = pd.read_csv(xydip, comment='#', header=0)
            elif isinstance(xydip, np.ndarray):
                if xydip.shape[1] != 4:
                    raise ValueError("dip array must have four columns: lon, lat, strike, dip.")
                dip_info = pd.DataFrame(xydip, columns=['lon', 'lat', 'strike', 'dip'])
            elif isinstance(xydip, pd.DataFrame):
                if xydip.shape[1] != 4:
                    raise ValueError("dip DataFrame must have four columns: lon, lat, strike, dip.")
                dip_info = xydip
            else:
                raise ValueError("dip must be a file path or a numpy array with lon, lat, strike, dip.")
    
        # Use self.top and self.depth if top_depth or bottom_depth is None
        top_depth = top_depth if top_depth is not None else self.top
        bottom_depth = bottom_depth if bottom_depth is not None else self.depth
    
        # Convert angles to radians
        dip_info['strike_rad'] = deg2rad(dip_info.strike)
        dip_info['dip_rad'] = deg2rad(dip_info.dip)
    
        # Convert lon/lat to x/y coordinates
        x, y = self.ll2xy(dip_info.lon.values, dip_info.lat.values)
        dip_info['xproj'] = x
        dip_info['yproj'] = y
    
        # Dip angle transfer to 0~90, and strike angle transfer in order to make dip right hand is positive
        negative_dip_flag = dip_info.dip_rad < 0
        dip_info.loc[negative_dip_flag, 'strike_rad'] += np.pi
        dip_info.loc[negative_dip_flag, 'dip_rad'] = -dip_info.loc[negative_dip_flag, 'dip_rad']

        # Calculate the unit vector along the dip direction. Positive dip is downward.
        dip_rad = dip_info.dip_rad.values
        strike_rad = dip_info.strike_rad.values
        dip_x = cos(dip_rad) * cos(-strike_rad)
        dip_y = cos(dip_rad) * sin(-strike_rad)
        dip_z = sin(dip_rad)
        dip_vector = np.vstack((dip_x, dip_y, dip_z)).T 
    
        # Calculate top coordinates
        x, y = self.ll2xy(isodepth_info.lon.values, isodepth_info.lat.values)
        ref_depth = isodepth_info.depth.values
        ref_coords = np.vstack((x, y, ref_depth)).T
        width = ((top_depth - ref_depth) / sin(dip_rad)).reshape(-1, 1)
        top_coords = ref_coords + dip_vector * width
    
        # Calculate bottom coordinates
        width = ((bottom_depth - top_depth) / sin(dip_rad)).reshape(-1, 1)
        bottom_coords = top_coords + dip_vector * width
    
        if update_self:
            self.set_top_coords(top_coords, lonlat=False)
            self.set_bottom_coords(bottom_coords, lonlat=False)
    
        return top_coords, bottom_coords
    
    def generate_bottom_from_single_dip(
            self, 
            dip_angle, 
            dip_direction, 
            update_self=True
        ):
        """
        Generate the bottom node coordinates of the fault based on dip angle and dip direction.
    
        Parameters:
        dip_angle (float): The dip angle of the fault in degrees.
        dip_direction (float): The dip direction of the fault in degrees.
        update_self (bool, optional): If True, update the instance variables with the calculated bottom coordinates. Default is True.
    
        Returns:
        np.ndarray: A 2D array containing the bottom coordinates in longitude, latitude, and depth.
        
        Raises:
        ValueError: If the 'top', 'depth', or 'top_coords' attributes are not set.
        """
        import numpy as np
    
        # Check if self.top and self.depth are set
        if not hasattr(self, 'top') or self.top is None:
            raise ValueError("Please set the 'top' attribute before calling this function.")
        if not hasattr(self, 'depth') or self.depth is None:
            raise ValueError("Please set the 'depth' attribute before calling this function.")
    
        # Check if self.top_coords exists and is not None
        if not hasattr(self, 'top_coords') or self.top_coords is None:
            raise ValueError("The attribute 'top_coords' is not set. Please set 'top_coords' before calling this function.")
    
        # Extract x_top, y_top, and z_top from self.top_coords
        x_top, y_top, z_top = self.top_coords[:, 0], self.top_coords[:, 1], self.top_coords[:, 2]
    
        # Convert degrees to radians
        dip_rad = np.deg2rad(dip_angle)
        dip_direction_rad = np.deg2rad(dip_direction)
    
        # Compute the bottom row
        depth_extend = self.depth - self.top
        width = depth_extend / np.sin(dip_rad)
        x_bottom = x_top + width * np.cos(dip_rad) * np.sin(dip_direction_rad)
        y_bottom = y_top + width * np.cos(dip_rad) * np.cos(dip_direction_rad)
        lon_bottom, lat_bottom = self.xy2ll(x_bottom, y_bottom)
        z_bottom = z_top + depth_extend
    
        # Save to self.bottom
        bottom_coords = np.vstack((x_bottom, y_bottom, z_bottom)).T
        bottom_coords_ll = np.vstack((lon_bottom, lat_bottom, z_bottom)).T
    
        if update_self:
            self.bottom_coords = bottom_coords
            self.bottom_coords_ll = bottom_coords_ll
    
        return bottom_coords_ll
    
    def generate_mesh_dutta(self, perturbations, disct_z=None, bias=None, min_dx=None):
        """
        Generate a mesh for a seismic fault using the Dutta method.

        Parameters:
        perturbations: Perturbation parameters to control the shape of the seismic fault. [D1, D2, S1, S2]
        disct_z: Discretization parameter in the z direction. If None, the default value will be used.
        bias: Bias parameter for the mesh. If None, the default value will be used.
        min_dx: Minimum size of the mesh. If None, the default value will be used.

        Returns:
        None. However, this function will modify self.top_coords and self.VertFace2csifault.

        Note:
        This function will modify the z-coordinates of self.top_coords to be negative, as the z-direction is downward in the seismic model.
        """
        vertices, faces = self.mesh_generator.generate_mesh_dutta(self.depth, perturbations, self.top_coords, disct_z, bias, min_dx)
        self.VertFace2csifault(vertices, faces)

    def read_mesh_file(self, mshfile, tag=None, save2csi=True, element_name='triangle', unit='m', proj_params=None):
        """
        Reads a mesh file, which can be in GMSH or Abaqus format.
    
        Parameters:
        mshfile (str): Path to the mesh file.
        tag (int, optional): Tag used only when reading Abaqus format files. Default is None.
        save2csi (bool, optional): If True, saves the result to a CSI format file. Default is True.
        element_name (str, optional): Name of the element. Default is 'triangle'.
        unit (str, optional): Unit of the input mesh file. Default is 'm'.
        proj_params (str, optional): Projection parameters for the input mesh file. Default is None.
    
        Returns:
        tuple: A tuple containing vertices and faces.
    
        Raises:
        ValueError: If the file cannot be read with meshio and is not in Gmsh 4.1 format.
        """
        import meshio
        try:
            # Attempt to read the file using the meshio library
            data = meshio.read(mshfile)
            vertices = data.points
            cells = data.cells
            cell_triangles = [imsh for imsh in cells if imsh.type == element_name]
    
            if tag is None:
                Faces = np.empty((0, 3), dtype=np.int32)
                for icell in cell_triangles:
                    Faces = np.append(Faces, icell.data)
            else:
                Faces = cell_triangles[tag].data
            
            Faces = Faces.reshape(-1, 3)
    
        except Exception as e:
            # Catch any exceptions and try to read the file with a different method
            with open(mshfile, 'rt') as fin:
                version_line = fin.readline()
    
            if '4.1' in version_line:
                # Read the gms file with the read_gmsh_4_1 method
                vertices, Faces = self.read_gmsh_4_1(mshfile, element_name)
            else:
                # Raise an error if the file is not Gmsh 4.1 version
                raise ValueError("Unable to read the file with meshio. The file is not Gmsh 4.1 version.") from e
    
        # Convert units, handle Vertices2, and convert to standard format
        vertices = self._process_vertices(vertices, unit, proj_params)
    
        if save2csi:
            self.VertFace2csifault(vertices, Faces)
    
        return vertices, Faces
    
    def _process_vertices(self, vertices, unit, proj_params):
        """
        Process vertices by converting units, handling Vertices2, and converting to standard format.
    
        Parameters:
        vertices (np.ndarray): Array of vertex coordinates.
        unit (str): Unit of the input mesh file.
        proj_params (str, optional): Projection parameters for the input mesh file. Default is None.
    
        Returns:
        np.ndarray: Processed vertices.
        """
        import re
        from pyproj import Proj, Transformer, CRS
    
        # Convert units to kilometers if necessary
        if unit == 'm':
            vertices /= 1000.0
        elif unit == 'cm':
            vertices /= 100000.0
        elif unit == 'mm':
            vertices /= 1000000.0
        elif unit == 'km':
            pass  # Already in kilometers
        else:
            raise ValueError(f"Unsupported unit: {unit}")
    
        # Ensure proj_params units are in kilometers
        if proj_params is not None:
            proj_params = re.sub(r'\+units=[a-z]+', '+units=km', proj_params)
            if '+units=' not in proj_params:
                proj_params += ' +units=km'
    
        # If the z-direction is negative, reverse it
        if np.mean(vertices[:, 2]) < 0:
            vertices[:, 2] = -vertices[:, 2]
    
        # Convert to lonlat using user-provided projection parameters if available
        if proj_params is not None:
            if 'utm' in proj_params:
                if 'zone' in proj_params:
                    # UTM projection with specified zone
                    proj = Proj(proj_params)
                else:
                    # UTM projection without specified zone
                    lon0_match = re.search(r'\+lon_0=(-?\d+(\.\d+)?)', proj_params)
                    lat0_match = re.search(r'\+lat_0=(-?\d+(\.\d+)?)', proj_params)
                    if lon0_match and lat0_match:
                        lon0 = float(lon0_match.group(1))
                        lat0 = float(lat0_match.group(1))
                    else:
                        raise ValueError("proj_params must include +lon_0 and +lat_0 if UTM zone is not specified.")
    
                    # Calculate the best UTM zone based on lon0 and lat0
                    from pyproj.database import query_utm_crs_info
                    from pyproj.aoi import AreaOfInterest
    
                    # Find the best UTM zone
                    utm_crs_list = query_utm_crs_info(
                        datum_name="WGS 84",
                        area_of_interest=AreaOfInterest(
                            west_lon_degree=lon0 - 2.,
                            south_lat_degree=lat0 - 2.,
                            east_lon_degree=lon0 + 2,
                            north_lat_degree=lat0 + 2
                        ),
                    )
                    utm = CRS.from_epsg(utm_crs_list[0].code)
                    proj = Proj(utm)
            else:
                # Non-UTM projection (e.g., tmerc or gauss)
                proj = Proj(proj_params)
    
            transformer = Transformer.from_proj(proj, proj.to_latlong())
            lon, lat = transformer.transform(vertices[:, 0], vertices[:, 1])
        else:
            # lon, lat = vertices[:, 0], vertices[:, 1]
            return vertices
    
        # Convert lonlat to standard format using self.ll2xy
        vertices[:, 0], vertices[:, 1] = self.ll2xy(lon, lat)
    
        return vertices
    
    def convert_mesh_file(self, mshfile, output_format=None, unit_conversion=1.0, 
                          flip_z=False, output_filename=None):
        import meshio
        import os
        try:
            data = meshio.read(mshfile)
            data.points *= unit_conversion

            if flip_z:
                data.points[:, 2] = -data.points[:, 2]

            # Fliter out the vertex cells if the output format is inp
            if output_format == 'inp':
                cells = [cell for cell in data.cells if cell.type != 'vertex']
            else:
                cells = data.cells

            # If output filename is not provided, generate it from the input filename with the new extension
            if output_filename is None:
                base_name = os.path.splitext(mshfile)[0]
                output_filename = f"{base_name}.{output_format}"

        except Exception as e:
            raise ValueError("Unable to read the file with meshio.") from e
        
        meshio.write(output_filename, meshio.Mesh(data.points, cells), file_format=output_format)
    
    def save_geometry_as_mesh(self, path, format=None, coord_type='none', 
                            output_unit='m', flip_z=False, proj_string=None):
        """
        Saves the mesh in a specified format using meshio.

        This method exports the mesh geometry to a file, supporting various formats through meshio. It allows for
        coordinate system conversion (UTM to lat-long or vice versa), unit scaling (e.g., kilometers to meters), and
        flipping the Z-axis if necessary.

        Parameters:
        - format (str): The file format to save the mesh in. Must be supported by meshio.
        - path (str): The file path where the mesh should be saved.
        - coord_type (str): The type of coordinates ('none', 'utm', 'proj'). 'none' means no projection, 'utm' means UTM coordinates, 'proj' means custom projection.
        - output_unit (str): The unit of the output coordinates. Default is 'm' (meters). Other options include 'km' (kilometers), 'ft' (feet), etc.
        - flip_z (bool): If True, the Z-axis values of the mesh vertices are negated. Useful for certain coordinate system conventions.
        - proj_string (str): The PROJ string for the target coordinate system. If None, no additional projection is applied.

        Returns:
        None
        """
        import meshio
        from pyproj import Proj
        from pyproj import Transformer
        import re

        # Define unit conversion factors from kilometers to target units
        unit_conversion = {
            'm': 1000.0,
            'km': 1.0,
            'ft': 3280.84,
            'mi': 0.621371,
            'in': 39370.1,
            'yd': 1093.61
        }
        
        # Get the conversion factor for the output unit
        unit_scale = unit_conversion.get(output_unit, 1000.0)  # Default to meters if unit not found

        # Select vertices based on the coordinate system and apply unit scaling and Z-axis flipping if needed
        if coord_type == 'utm':
            vertices = self.Vertices.copy()
            vertices *= unit_scale  # Apply unit scaling to all coordinates
        elif coord_type == 'proj' and proj_string:
            in_proj = 'epsg:4326'  # Assuming input coordinates are in WGS84
            # Use regex to replace any unit with kilometers
            if proj_string and '+units=' in proj_string:
                proj_string = re.sub(r'\+units=\w+', '+units=km', proj_string)
            else:
                proj_string += ' +units=km'
            out_proj = proj_string
            transformer = Transformer.from_crs(in_proj, out_proj, always_xy=True)
            
            vertices = self.Vertices_ll.copy()
            lon, lat, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            x, y = transformer.transform(lon, lat)
            vertices[:, 0], vertices[:, 1] = x, y

            vertices[:, 0] *= unit_scale  # Apply unit scaling to X-axis
            vertices[:, 1] *= unit_scale  # Apply unit scaling to Y-axis
            vertices[:, 2] *= unit_scale  # Apply unit scaling to Z-axis
        else:
            vertices = self.Vertices_ll.copy()
            if unit_scale != 1.0:
                vertices[:, 2] *= unit_scale  # Apply unit scaling to Z-axis only

        if flip_z:
            vertices[:, 2] = -vertices[:, 2]  # Flip Z-axis

        # Create and save the mesh object to the specified path
        mesh = meshio.Mesh(points=vertices, cells=[("triangle", self.Faces)])
        try:
            mesh.write(path, file_format=format)
        except ValueError as e:
            print(f"Error: Failed to save mesh in '{format}' format. {e}")

    def read_gmsh_4_1(self, gmsh_file, element_name=None, save2csi=True):
        """
        The purpose of this function is to read a mesh file in GMSH format.

        Parameters:
        gmsh_file: The path to the mesh file.
        element_name: The name of the element. If None, all triangular cells will be selected.
        save2csi: If True, the results will be saved to a file in CSI format.

        Returns:
        If save2csi is False, returns a tuple containing node coordinates, element vertex indices, node indices for each entity, and the number of elements for each entity.
        If save2csi is True, there is no return value, but a file in CSI format will be generated.
        """
        assert '.msh' in gmsh_file, 'Please input a file with gmsh format'

        with open(gmsh_file, 'rt') as fin:
            mesh_data = fin.readlines()
            mesh_data = [line.strip() for line in mesh_data]

        mesh_data_pd = pd.Series(mesh_data)
        # Find flags
        flags = np.argwhere((mesh_data_pd.str[:1] == '$').values).flatten()

        # Allocate space for nc and nEl arrays
        num_entities = str2num(mesh_data[flags[2]+1])[2]
        num_nodes_per_entity = np.zeros((num_entities, ), dtype=np.int_)
        num_elements_per_entity = num_nodes_per_entity.copy()

        # Nodes start at 5th flag + 1, end at 6th
        num_nodes = str2num(mesh_data[flags[4]+1])[1] # Get number of nodes
        node_coordinates = np.zeros((num_nodes, 3)) # Allocate space for node array
        nodes = mesh_data[flags[4]+2:flags[5]]

        k = 0
        # Step through node rows
        while k < len(nodes):
            # Get length of entity
            num_nodes_in_entity = str2num(nodes[k])[-1]
            if num_nodes_in_entity != 0:
                # Loops through that entity's nodes
                for i in range(k+num_nodes_in_entity+1, k+num_nodes_in_entity+num_nodes_in_entity+1):
                    node_data = str2num(nodes[i], dtype=float)
                    if len(node_data) == 3:
                        idx = str2num(nodes[i-num_nodes_in_entity])[0] - 1
                        node_coordinates[idx, :] = np.array(node_data)
                    k = i + 1
            else:
                k += 1

        # Elements start at 7th flag + 1, end at 8th flag
        num_elements = str2num(mesh_data[flags[6]+1])[1] # Get number of elements
        element_indices = np.zeros((num_elements, 3), dtype=np.int_) # Allocate space for element array
        elements = mesh_data[flags[6]+2:flags[7]] # Get all element rows

        k = 0
        # Step through element rows
        while k < len(elements):
            # Get length of entity
            num_elements_in_entity = str2num(elements[k])[-1]
            if num_elements_in_entity != 0:
                for i in range(k+1, k+num_elements_in_entity+1):
                    element_data = str2num(elements[i], dtype=int)
                    if len(element_data) == 4:  # if it's a triangle
                        idx = element_data[0] - 1
                        element_indices[idx, :] = np.array(element_data, dtype=np.int_)[1:4] - 1
                    k = i + 1
        
        # Trim empty rows
        element_indices = element_indices[np.sum(element_indices, axis=1) != 0, :]

        # Determine number of coordinates in each entity
        num_elements_per_entity = np.array([len(np.unique(element_indices[i:i+num_elements_per_entity[i], :])) for i in range(num_entities)])

        # Ensure consistent node circulation direction  
        cross_product = np.cross(node_coordinates[element_indices[:, 1], :] - node_coordinates[element_indices[:, 0], :],
                             node_coordinates[element_indices[:, 2], :] - node_coordinates[element_indices[:, 0], :])
        negative_indices = cross_product[:, 2] < 0
        element_indices[negative_indices, 1:] = element_indices[negative_indices, -1:-3:-1]

        # 如果z方向为负，则反向
        if np.mean(node_coordinates[:, 2]) < 0:
            node_coordinates[:, 2] = -node_coordinates[:, 2]
        if save2csi:
            self.VertFace2csifault(node_coordinates, element_indices)
        return node_coordinates, element_indices # , num_nodes_per_entity, num_elements_per_entity
    
    def VertFace2csifault(self, vertices, faces, check_order=True):
        """
        The purpose of this function is to convert vertex and face information into fault data in CSI format.
    
        Parameters:
        vertices: A two-dimensional array where each row represents the coordinates of a vertex.
        faces: A two-dimensional array where each row represents the vertex indices of a face.
        check_order (bool, optional): If True, check and reorder the vertices to ensure counter-clockwise order. Default is True.
    
        Return value:
        There is no return value. However, the function updates the following instance variables:
        - Vertices: The input vertex coordinates.
        - Faces: The vertex indices of the input faces.
        - patch: A list composed of the vertex coordinates of the faces.
        - numpatch: The number of faces.
        - top: The minimum z-coordinate of all vertices.
        - depth: The maximum z-coordinate of all vertices.
        - z_patches: An arithmetic sequence from 0 to depth, used for interpolation.
        - factor_depth: Depth factor, initially set to 1.
        """
        import numpy as np
    
        # Input validation
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices should be a 2D array with 3 columns.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces should be a 2D array with 3 columns.")
    
        if check_order:
            # Ensure each patch's nodes are in counter-clockwise order
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            
            # Calculate differences
            v1_v0_x = v1[:, 0] - v0[:, 0]
            v1_v0_y = v1[:, 1] - v0[:, 1]
            v2_v0_x = v2[:, 0] - v0[:, 0]
            v2_v0_y = v2[:, 1] - v0[:, 1]
            
            # Calculate cross product
            cross_product = v1_v0_x * v2_v0_y - v1_v0_y * v2_v0_x
            counter_clockwise = cross_product > 0
            
            # Swap the last two vertices to make it counter-clockwise where necessary
            faces[~counter_clockwise] = faces[~counter_clockwise][:, [0, 2, 1]]
    
        self.Vertices = vertices
        self.vertices2ll()
        self.Faces = faces
        self.patch = vertices[faces]
        self.patch2ll()
        self.numpatch = len(self.patch)
    
        self.top = np.amin(vertices[:, -1])
        self.depth = np.amax(vertices[:, -1])
        self.z_patches = self.generate_z_patches()
        self.factor_depth = 1.

    def calculate_triangle_areas(self):
        # Get the vertices of each triangle
        v1 = self.Vertices[self.Faces[:, 0]]
        v2 = self.Vertices[self.Faces[:, 1]]
        v3 = self.Vertices[self.Faces[:, 2]]

        # Calculate the lengths of the sides of each triangle
        a = np.linalg.norm(v2 - v1, axis=1)
        b = np.linalg.norm(v3 - v2, axis=1)
        c = np.linalg.norm(v1 - v3, axis=1)

        # Calculate the semiperimeter of each triangle
        s = (a + b + c) / 2

        # Calculate the area of each triangle using Heron's formula
        areas = np.sqrt(s * (s - a) * (s - b) * (s - c))
    
        self.area = areas

        return areas

    # def compute_triangle_areas(self):
    #     self.area = calculate_triangle_areas(self.Vertices, self.Faces)
    #     return self.area
    
    # def compute_patch_areas(self):
    #     self.area = calculate_triangle_areas(self.Vertices, self.Faces)
    #     return self.area

    def calculate_triangle_normals(self):
        return compute_triangle_normals(self.Vertices, self.Faces)

    def order_vertices_clockwise(self):
        self.Faces = arrange_vertices_clockwise(self.Vertices, self.Faces)

    def generate_z_patches(self, num_patches=5):
        """
        Generate an array of z values for interpolation.

        Parameters:
        num_patches: The number of patches to generate.

        Returns:
        An array of z values.
        """
        return np.linspace(0, self.depth, num_patches)
    
    def generate_mesh(self,  
                        top_size=None, 
                        bottom_size=None, 
                        mesh_func=None, 
                        out_mesh=None, 
                        write2file=False,
                        show=True,
                        read_mesh=True,
                        field_size_dict={'min_dx': 3, 'bias': 1.05},
                        segments_dict=None,
                        verbose=5,
                        mesh_algorithm=2, # 5: Delaunay, 6: Frontal-Delaunay
                        optimize_method='Laplace2D'):
        """
        Generate a slanted fault mesh.
    
        Parameters:
        top_size (float, optional): Size of the top. If None, mesh_func will be used to calculate it.
        bottom_size (float, optional): Size of the bottom. If None, mesh_func will be used to calculate it.
        mesh_func (callable, optional): Function to calculate mesh size. If None, top_size and bottom_size will be used.
        out_mesh (str, optional): Path to the output mesh file. If None, a default path will be used.
        write2file (bool, optional): If True, the mesh will be written to a file.
        show (bool, optional): If True, the graphical user interface will be displayed.
        read_mesh (bool, optional): If True, the generated mesh file will be read.
        field_size_dict (dict, optional): Dictionary containing 'min_dx' and 'bias' keys for calculating mesh size.
        segments_dict (dict, optional): Dictionary for setting mesh division parameters for each edge of the fault.
            Contains the number of segments for the top, bottom, left, and right edges, as well as the progression ratios for top-bottom and left-right.
            Keys: top_segments, bottom_segments, left_segments, right_segments, top_bottom_progression, left_right_progression
        verbose (int, optional): Gmsh log level, ranging from 0 (no log information) to 5 (print all log information).
        mesh_algorithm (int, optional): Mesh algorithm to use (2: default, 5: Delaunay, 6: Frontal-Delaunay).
        optimize_method (str, optional): Optimization method to use ('Laplace2D' by default).
    
        Returns:
        None. However, if read_mesh is True, the following instance variables will be updated:
        - Vertices: Coordinates of the vertices read from the mesh file.
        - Faces: Indices of the vertices forming the faces read from the mesh file.
        - patch: List of vertex coordinates forming the faces.
        - numpatch: Number of faces.
        - top: Minimum z-coordinate of all vertices.
        - depth: Maximum z-coordinate of all vertices.
        - z_patches: Arithmetic sequence from 0 to depth for interpolation.
        - factor_depth: Depth factor, initially set to 1.
        """
        self.mesh_generator.set_coordinates(self.top_coords, self.bottom_coords)
        vertices, faces = self.mesh_generator.generate_gmsh_mesh(top_size=top_size, bottom_size=bottom_size, mesh_func=mesh_func, 
                                                out_mesh=out_mesh, write2file=write2file, show=show, read_mesh=read_mesh, 
                                                field_size_dict=field_size_dict, segments_dict=segments_dict, 
                                                verbose=verbose, mesh_algorithm=mesh_algorithm, optimize_method=optimize_method)
        self.VertFace2csifault(vertices, faces)

    def generate_multilayer_mesh(self, layers_coords, sizes=None, mesh_func=None, 
                                 out_mesh=None, write2file=False, show=True, read_mesh=True, 
                                 field_size_dict={'min_dx': 3, 'bias': 1.05},
                                 mesh_algorithm=2, # 5: Delaunay, 6: Frontal-Delaunay
                                 optimize_method='Laplace2D', verbose=5):
        self.mesh_generator.set_coordinates(self.top_coords, self.bottom_coords)
        vertices, faces = self.mesh_generator.generate_multilayer_gmsh_mesh(layers_coords=layers_coords, sizes=sizes, mesh_func=mesh_func, 
                                                            out_mesh=out_mesh, write2file=write2file, show=show, read_mesh=read_mesh, 
                                                            field_size_dict=field_size_dict, mesh_algorithm=mesh_algorithm, 
                                                            optimize_method=optimize_method, verbose=verbose)
        self.VertFace2csifault(vertices, faces)

    def skin_curve_to_bottom(self, 
                             depth_extend=None, 
                             interval_num=20,
                             curve_top=None,
                             curve_bottom=None,
                             use_relocisocurve=False,
                             update_self=True):
        """
        Generate extended bottom coordinates to a specified depth based on the original trace and the bottom curve determined by relocated aftershocks.
        The name "skin curve" is derived from the corresponding method in the commercial software Trelis.
    
        Parameters:
        depth_extend (float, optional): The depth to extend to. If None, self.depth will be used.
        interval_num (int, optional): The number of intervals for interpolation.
        curve_top (np.ndarray, optional): The top curve. If None, self.top_coords will be used.
        curve_bottom (np.ndarray, optional): The bottom curve. If None, self.relocisocurve will be used.
        use_relocisocurve (bool, optional): Whether to use the relocated isocurve. Default is False.
        update_self (bool, optional): Whether to update self.bottom_coords and self.bottom_coords_ll with the new bottom coordinates. Default is True.
    
        Returns:
        np.ndarray: The extended bottom coordinates in longitude, latitude, and depth format.
    
        Raises:
        ValueError: If 'top' or 'depth' attributes are not set, or if 'relocisocurve' attribute is not set or is None when required.
        """
    
        if self.top is None or self.depth is None:
            raise ValueError("Please set the 'top' and 'depth' attributes before calling this function.")
    
        depth = depth_extend if depth_extend is not None else self.depth
        top = np.mean(curve_top[:, -1]) if curve_top is not None else self.top
    
        from shapely.geometry import LineString
        import numpy as np
    
        # 1. Interpolate
        param_coord_line = np.linspace(0, 1, int(interval_num))
        line1 = LineString(self.top_coords) if curve_top is None else LineString(curve_top)
        coords_in_line1 = np.array([line1.interpolate(ipara, normalized=True).coords[0] for ipara in param_coord_line])
        if curve_bottom is not None:
            line2 = LineString(curve_bottom)
        elif use_relocisocurve:
            if not hasattr(self, 'relocisocurve') or self.relocisocurve is None:
                raise ValueError("The 'relocisocurve' attribute is not set or is None.")
            line2 = LineString(self.relocisocurve)
        elif hasattr(self, 'bottom_coords') and self.bottom_coords is not None:
            line2 = LineString(self.bottom_coords)
        else:
            if not hasattr(self, 'relocisocurve') or self.relocisocurve is None:
                raise ValueError("The 'relocisocurve' attribute is not set or is None.")
            line2 = LineString(self.relocisocurve)
        coords_in_line2 = np.array([line2.interpolate(ipara, normalized=True).coords[0] for ipara in param_coord_line])
    
        # 2. Extend to specific depth
        extended_bottom_coords = []
        depth_extend = depth - top
        for i in range(coords_in_line2.shape[0]):
            vector = coords_in_line2[i] - coords_in_line1[i]
            normalized_vector = vector / np.linalg.norm(vector)
    
            # Translate the top coordinates to the specific depth along the direction of the vector
            translate_vector = depth_extend / normalized_vector[-1] * normalized_vector
            bottom = coords_in_line1[i] + translate_vector
            extended_bottom_coords.append(bottom)
        extended_bottom_coords = np.array(extended_bottom_coords)
    
        lonb, latb = self.xy2ll(extended_bottom_coords[:, 0], extended_bottom_coords[:, 1])
        extended_bottom_coords_ll = np.vstack((lonb, latb, extended_bottom_coords[:, -1])).T
    
        if update_self:
            self.bottom_coords = extended_bottom_coords
            self.bottom_coords_ll = extended_bottom_coords_ll
    
        return extended_bottom_coords_ll

    def plot_profiles(self, plot_aftershocks=False, figsize=None, 
                      style=['science'], fontsize=None, save_fig=False, 
                      file_path='profile.png', dpi=300, scatter_props=None,
                      show=True, draw_trace_arrow=True, profiles_to_keep=None, profiles_to_remove=None):
        """
        Plot the profiles with optional aftershocks.
    
        Parameters:
        plot_aftershocks (bool): Whether to plot aftershocks.
        figsize (tuple): Figure size.
        style (list): Plot style.
        fontsize (int): Font size.
        save_fig (bool): Whether to save the figure.
        file_path (str): Path to save the figure.
        dpi (int): Dots per inch for the saved figure.
        scatter_props (dict): Properties for scatter plot.
        show (bool): Whether to show the plot.
        draw_trace_arrow (bool): Whether to draw trace arrow.
        profiles_to_keep (list, optional): List of profile indices to keep. Default is None.
        profiles_to_remove (list, optional): List of profile indices to remove. Default is None.
    
        Returns:
        None
        """
        # Ensure profiles_to_keep and profiles_to_remove are not used simultaneously
        if profiles_to_keep is not None and profiles_to_remove is not None:
            raise ValueError("Cannot use profiles_to_keep and profiles_to_remove simultaneously.")
    
        # Convert numeric profiles to string format
        if profiles_to_keep is not None:
            profiles_to_keep = [f'Profile_{p}' if isinstance(p, int) else p for p in profiles_to_keep]
        if profiles_to_remove is not None:
            profiles_to_remove = [f'Profile_{p}' if isinstance(p, int) else p for p in profiles_to_remove]
    
        # Filter profiles to keep or remove
        if profiles_to_keep is not None:
            profiles_to_remove = [p for p in self.profiles.keys() if p not in profiles_to_keep]
        if profiles_to_remove is not None:
            profiles_to_remove = [p for p in self.profiles.keys() if p in profiles_to_remove]
    
        # Remove profiles from self.profiles and self.relocated_aftershock_source.profiles
        if profiles_to_remove is not None:
            for prof_name in profiles_to_remove:
                if prof_name in self.profiles:
                    del self.profiles[prof_name]
                if prof_name in self.relocated_aftershock_source.profiles:
                    del self.relocated_aftershock_source.profiles[prof_name]
    
        seis = self.relocated_aftershock_source
        seis.plot_profiles(self, plot_aftershocks=plot_aftershocks, figsize=figsize,
                           style=style, fontsize=fontsize, save_fig=save_fig,
                           file_path=file_path, dpi=dpi, scatter_props=scatter_props,
                           show=show, draw_trace_arrow=draw_trace_arrow)
        return

    def plot_3d(self, max_depth=None, scatter_props=None, fontsize=None, 
                save_fig=False, file_path='profile3D.png', dpi=300, style=['science'],
                figsize=None, show=True, elev=None, azim=None, offset_to_fault=False,
                shape=(1.0, 1.0, 1.0), z_height=None):
        """
        Plot a 3D representation of the fault surface and profiles.
    
        Parameters:
        max_depth (float, optional): Maximum depth for the plot. If None, the maximum depth of the vertices will be used.
        scatter_props (dict, optional): Properties for the scatter plot. Default is {'color': '#ff0000', 'ec': 'k', 'linewidths': 0.5}.
        fontsize (int, optional): Font size for the plot.
        save_fig (bool, optional): If True, the figure will be saved to a file. Default is False.
        file_path (str, optional): Path to save the figure. Default is 'profile3D.png'.
        dpi (int, optional): Dots per inch for the saved figure. Default is 300.
        style (list, optional): List of styles to apply to the plot. Default is ['science'].
        figsize (tuple, optional): Figure size. Default is None.
        show (bool, optional): If True, the plot will be displayed. Default is True.
        elev (float, optional): Elevation angle for the 3D plot. Default is None.
        azim (float, optional): Azimuth angle for the 3D plot. Default is None.
        offset_to_fault (bool, optional): If True, the profile center will be offset to the fault. Default is False.
        shape (tuple, optional): Shape of the plot. Default is (1.0, 1.0, 1.0).
        z_height (float, optional): Height of the z-axis. Default is None.
    
        Returns:
        None
        """
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from ..plottools import sci_plot_style
    
        with sci_plot_style(style, fontsize=fontsize, figsize=figsize):
            fig = plt.figure(facecolor='white')  # none: Set background to white
            ax = fig.add_subplot(111, projection='3d', facecolor='white')  # none: Set axis background to white
    
            # Set grid lines
            # ax.xaxis._axinfo["grid"]['color'] =  'black'
            # ax.yaxis._axinfo["grid"]['color'] =  'black'
            # ax.zaxis._axinfo["grid"]['color'] =  'black'
            # ax.xaxis._axinfo["grid"]['linestyle'] =  ':'
            # ax.yaxis._axinfo["grid"]['linestyle'] =  ':'
            # ax.zaxis._axinfo["grid"]['linestyle'] =  ':'
            # ax.xaxis._axinfo["grid"]['linewidth'] =  0.5
            # ax.yaxis._axinfo["grid"]['linewidth'] =  0.5
            # ax.zaxis._axinfo["grid"]['linewidth'] =  0.5
    
            # Set view angle
            if elev is not None and azim is not None:
                ax.view_init(elev=elev, azim=azim)
    
            # Plot fault surface
            x = self.Vertices_ll[:, 0]
            y = self.Vertices_ll[:, 1]
            z = self.Vertices_ll[:, 2]
    
            surf = ax.plot_trisurf(x, y, z, triangles=self.Faces, linewidth=0.5, edgecolor='#7291c9', zorder=1)
            surf.set_facecolor((0, 0, 0, 0))  # Set face color to transparent
    
            ax.set_zlabel('Depth')
    
            ax.invert_zaxis()  # Invert z-axis to display values from 0 to max depth downwards
    
            ax.xaxis.pane.fill = False  # Set x-axis pane to transparent
            ax.yaxis.pane.fill = False  # Set y-axis pane to transparent
            ax.zaxis.pane.fill = False  # Set z-axis pane to transparent
    
            def project_point_to_plane(points, plane_point, normal):
                # Calculate the projection of points onto a plane
                point_vectors = points - plane_point
                distances = np.dot(point_vectors, normal) / np.linalg.norm(normal)
                projections = points - np.outer(distances, normal)
                return projections
    
            # Plot scatter points
            if scatter_props is None:
                scatter_props = {'color': '#ff0000', 'ec': 'k', 'linewidths': 0.5}
    
            if max_depth is None:
                max_depth = np.max(self.Vertices_ll[:, 2])
    
            for iprofile in self.profiles:
                profile = self.profiles[iprofile]
                end_points_ll = np.array(profile['EndPointsLL'])
                end_points_ll_reversed = end_points_ll[::-1]  # Reverse end_points_ll
    
                # Create a list of 5 points: first two at 0 depth, next two at max depth, last one back to the first point
                points = np.concatenate((end_points_ll, end_points_ll_reversed, end_points_ll[0:1]), axis=0)
                depths = [0, 0, max_depth, max_depth, 0]
    
                # Plot these 5 points
                for i in range(len(points) - 1):
                    ax.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], [depths[i], depths[i+1]], 'k--', lw=1.0, zorder=3)
    
                # Calculate a point on the plane and the normal vector
                plane_point = np.append(end_points_ll[0], 0)  # Add depth value
                normal = np.cross(np.append(end_points_ll[1], 0) - plane_point, [0, 0, 1])  # Add depth value
                normal /= np.linalg.norm(normal)  # Normalize the normal vector
    
                lon = profile['Lon']
                lat = profile['Lat']
                depth = profile['Depth']
                points = np.array([lon, lat, depth]).T
    
                # Calculate projection points and plot
                if 'Projections' in profile:  # If projection coordinates already exist, use them
                    projections = profile['Projections']
                else:  # If no projection coordinates, perform projection and store in profile
                    projections = project_point_to_plane(points, plane_point, normal)
    
                # If the profile center needs to be offset to the fault
                if offset_to_fault:
                    lonc, latc = profile['Center']
                    # Calculate offset vector
                    top_offset_vector = self.compute_offset_vector(iprofile, in_lonlat=True, use_top_coords=True)
                    bottom_offset_vector = self.compute_offset_vector(iprofile, in_lonlat=True, use_top_coords=False)
                    if top_offset_vector is not None and bottom_offset_vector is not None:
                        # Offset the profile center to the fault
                        top_offset_vector.append(self.top)  # Add self.top in the z direction
                        bottom_offset_vector.append(self.depth)  # Add self.depth in the z direction
                        center = np.array([lonc, latc, self.top])
                        center_top = center + np.array(top_offset_vector)
                        center_bottom = center + np.array(bottom_offset_vector)
    
                        # Plot the intersection line
                        ax.plot([center_top[0], center_bottom[0]], 
                                [center_top[1], center_bottom[1]], 
                                [center_top[2], center_bottom[2]], color='#2e8b57', linestyle='-', zorder=4, lw=2.0)
    
                        # Save the intersection line to the profile
                        profile['Intersection'] = np.array([center_top, center_bottom])
    
                    top_offset_vector = np.array(top_offset_vector) if top_offset_vector is not None else np.array([0, 0, 0])
                    if 'Projections' not in profile:
                        if top_offset_vector.size == 2:
                            top_offset_vector = np.append(top_offset_vector, self.top)  # 0 depth?
                        profile['Projections'] = projections + top_offset_vector[None, :]
    
                # Plot all projection points
                projections = np.array(projections)
                ax.scatter(projections[:, 0], projections[:, 1], projections[:, 2], **scatter_props, zorder=2)
    
            # Get the range of x and y axes
            x_range = np.ptp(ax.get_xlim())
            y_range = np.ptp(ax.get_ylim())
    
            if z_height is not None:
                # Calculate the ratio of x and y axes
                xy_ratio = x_range / y_range
                # Set the height ratio of the plot
                ax.set_box_aspect([xy_ratio, 1, z_height])
            else:
                # Set the height ratio of the plot
                ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([shape[0], shape[1], shape[2], 1]))
            # Set the labels for x and y axes
            formatter = DegreeFormatter()
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
    
        # Show or save the figure
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Adjust the figure boundaries
        if save_fig:
            plt.savefig(file_path, dpi=dpi)  # , bbox_inches='tight', pad_inches=0.1
        # Show the plot
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_3d_plotly(self, max_depth=None, scatter_props=None, fontsize=None, 
                       save_fig=False, file_path='profile3D_plotly.png', dpi=300, style=['science'],
                       figsize=None, show=True, elev=None, azim=None, offset_to_fault=False,
                       shape=(1.0, 1.0, 1.0), z_height=None):
        import plotly.graph_objects as go
        from scipy.spatial import Delaunay
    
        # Create 3D figure
        fig = go.Figure()
    
        def project_point_to_plane(points, plane_point, normal):
            # Project points onto a plane
            point_vectors = points - plane_point
            distances = np.dot(point_vectors, normal) / np.linalg.norm(normal)
            projections = points - np.outer(distances, normal)
            return projections
    
        # Initialize scatter properties if not provided
        if scatter_props is None:
            scatter_props = {'color': '#ff0000', 'ec': 'k', 'linewidths': 0.5}
    
        # Set max depth if not provided
        if max_depth is None:
            max_depth = np.max(self.Vertices_ll[:, 2])
    
        # Collect traces
        traces = []
    
        for iprofile in self.profiles:
            profile = self.profiles[iprofile]
            end_points_ll = np.array(profile['EndPointsLL'])
            end_points_ll_reversed = end_points_ll[::-1]  # Reverse end_points_ll
    
            # Create a list of 5 points: first two at 0 depth, next two at max depth, last one back to the first point
            points = np.concatenate((end_points_ll, end_points_ll_reversed, end_points_ll[0:1]), axis=0)
            depths = [0, 0, max_depth, max_depth, 0]
    
            # Draw these 5 points
            for i in range(len(points) - 1):
                traces.append(go.Scatter3d(x=[points[i][0], points[i+1][0]], 
                                           y=[points[i][1], points[i+1][1]], 
                                           z=[depths[i], depths[i+1]], 
                                           mode='lines',
                                           line=dict(color='black', width=2),
                                           showlegend=False))
    
            # Calculate a point on the plane and the normal vector
            plane_point = np.append(end_points_ll[0], 0)  # Add depth value
            normal = np.cross(np.append(end_points_ll[1], 0) - plane_point, [0, 0, 1])  # Add depth value
            normal /= np.linalg.norm(normal)  # Normalize the normal vector
    
            lon = profile['Lon']
            lat = profile['Lat']
            depth = profile['Depth']
            points = np.array([lon, lat, depth]).T
    
            # Calculate projection points and plot
            if 'Projections' in profile:  # If projection coordinates already exist, use them directly
                projections = profile['Projections']
            else:  # If no projection coordinates, project and store them in the profile
                projections = project_point_to_plane(points, plane_point, normal)
    
            # If need to offset the profile center to the fault
            if offset_to_fault:
                lonc, latc = profile['Center']
                # Calculate offset vector
                top_offset_vector = self.compute_offset_vector(iprofile, in_lonlat=True, use_top_coords=True)
                bottom_offset_vector = self.compute_offset_vector(iprofile, in_lonlat=True, use_top_coords=False)
                if top_offset_vector is not None and bottom_offset_vector is not None:
                    # Offset the profile center to the fault
                    top_offset_vector.append(self.top)  # Add self.top in z direction
                    bottom_offset_vector.append(self.depth)  # Add self.depth in z direction
                    center = np.array([lonc, latc, self.top])
                    center_top = center + np.array(top_offset_vector)
                    center_bottom = center + np.array(bottom_offset_vector)
    
                    # Draw intersection line
                    traces.append(go.Scatter3d(x=[center_top[0], center_bottom[0]], 
                                               y=[center_top[1], center_bottom[1]], 
                                               z=[center_top[2], center_bottom[2]], 
                                               mode='lines',
                                               line=dict(color='#2e8b57', width=2),
                                               showlegend=False))
    
                    # Save the intersection line to the profile
                    profile['Intersection'] = np.array([center_top, center_bottom])
    
                    top_offset_vector = np.array(top_offset_vector) if top_offset_vector is not None else np.array([0, 0, 0])
                    if 'Projections' not in profile:
                        profile['Projections'] = projections + top_offset_vector[None, :]
    
                    # Draw all aftershock projection points
                    projections = np.array(projections)
                    traces.append(go.Scatter3d(x=projections[:, 0], y=projections[:, 1], z=projections[:, 2], 
                                               mode='markers', 
                                               marker=dict(size=2, color=scatter_props['color']),
                                               showlegend=False))
    
        # Add surface
        x = self.Vertices_ll[:, 0]
        y = self.Vertices_ll[:, 1]
        z = self.Vertices_ll[:, 2]
        # Draw 3D mesh
        traces.append(go.Mesh3d(x=x, y=y, z=z, 
                                i=self.Faces[:, 0], 
                                j=self.Faces[:, 1], 
                                k=self.Faces[:, 2], 
                                color='lightblue', 
                                opacity=0.50))
    
        # Draw mesh lines
        for face in self.Faces:
            for i in range(3):
                traces.append(go.Scatter3d(x=[x[face[i]], x[face[(i+1)%3]]], 
                                           y=[y[face[i]], y[face[(i+1)%3]]], 
                                           z=[z[face[i]], z[face[(i+1)%3]]], 
                                           mode='lines',
                                           line=dict(color='#7291c9', width=0.5),
                                           showlegend=False))
    
        # Add all traces to the figure
        fig.add_traces(traces)
    
        # Set view angle
        if elev is not None and azim is not None:
            fig.update_layout(scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                                center=dict(x=0, y=0, z=-0.5), 
                                                eye=dict(x=1.25*azim, y=1.25*azim, z=1.25*elev)))
    
        # Calculate x and y axis ranges
        x_range = np.max(self.Vertices_ll[:, 0]) - np.min(self.Vertices_ll[:, 0])
        y_range = np.max(self.Vertices_ll[:, 1]) - np.min(self.Vertices_ll[:, 1])
    
        # Calculate x and y axis ratio
        xy_ratio = x_range / y_range
    
        # Set x, y, and z axis labels and ratio
        fig.update_layout(scene=dict(xaxis=dict(title_text="Lon"),
                                     yaxis=dict(title_text="Lat"),
                                     zaxis=dict(autorange="reversed", 
                                                title_text="Depth", 
                                                tickvals=list(range(0, int(max_depth)+1, 10)), 
                                                ticktext=[str(i) for i in range(0, int(max_depth)+1, 10)]),
                                     aspectratio=dict(x=xy_ratio, y=1, z=z_height)))
    
        # Show or save the figure
        if save_fig:
            fig.write_image(file_path, scale=dpi)
        if show:
            fig.show()
    
    def compute_offset_vector(self, name, in_lonlat=False, use_top_coords=True):
        '''
        Computes the offset vector from the fault/profile intersection to the profile center.

        Args:
            * name          : name of the profile.
            * in_lonlat     : if True, return the offset in longitude/latitude coordinates;
                              if False, return the offset in projected coordinates.
            * use_top_coords: if True, use self.top_coords to get xf and yf;
                              if False, use self.bottom_coords.

        Returns:
            * Offset vector [dx, dy] or None if there is no intersection
        '''

        # Import shapely
        import shapely.geometry as geom

        # Grab the fault trace from self
        if use_top_coords:
            coords = self.top_coords
        else:
            coords = self.bottom_coords
        
        xf = coords[:, 0]
        yf = coords[:, 1]

        # Grab the profile
        prof = self.profiles[name]

        # Build a linestring with the profile center
        Lp = geom.LineString(prof['EndPoints'])

        # Build a linestring with the fault
        ff = []
        for i in range(len(xf)):
            ff.append([xf[i], yf[i]])
        Lf = geom.LineString(ff)

        # Get the intersection
        if Lp.crosses(Lf):
            Pi = Lp.intersection(Lf)
            if type(Pi) is geom.point.Point:
                p = Pi.coords[0]
            else:
                return None
        else:
            return None

        # Get the center
        lonc, latc = prof['Center']
        if in_lonlat:
            # Convert the intersection point to lon/lat
            lonp, latp = self.xy2ll(p[0], p[1])
            # Return the offset in longitude/latitude coordinates
            dx = lonp - lonc
            dy = latp - latc
        else:
            # Return the offset in projected coordinates
            xc, yc = self.ll2xy(lonc, latc)
            dx = p[0] - xc
            dy = p[1] - yc

        # All done
        return [dx, dy]
    
    def create_fault_trace_buffer(self, buffer_size, use_discrete_trace=False, return_lonlat=False):
        """
        Create a buffer around the fault trace.
    
        Parameters:
        buffer_size (float): The size of the buffer in kilometers.
        use_discrete_trace (bool, optional): Whether to use the discretized trace. Default is False, meaning the continuous trace is used.
        return_lonlat (bool, optional): Whether to return coordinates in longitude and latitude. Default is False, meaning x, y coordinates are returned.
    
        Returns:
        numpy.ndarray: The coordinates of the buffer, shaped as (n, 2). If return_lonlat is True, the coordinates are in longitude and latitude; otherwise, they are in x, y coordinates.
    
        Example:
        >>> obj = YourClass(xi, yi, xf, yf)
        >>> buffer_coords = obj.create_fault_trace_buffer(1, use_discrete_trace=True, return_lonlat=True)
        """
        from shapely.geometry import LineString, Polygon
        import numpy as np
    
        # Select the trace to use based on the value of use_discrete_trace
        x = self.xi if use_discrete_trace else self.xf
        y = self.yi if use_discrete_trace else self.yf
    
        # Create a LineString object
        line = LineString(zip(x, y))
    
        # Create the buffer
        buffer = line.buffer(buffer_size)
    
        # If the buffer is a Polygon object, get its exterior coordinates and convert to a numpy array
        if isinstance(buffer, Polygon):
            coords = np.array(buffer.exterior.coords[:])
        else:  # If the buffer is a MultiPolygon object, get the exterior coordinates of all Polygons and convert to a numpy array
            coords = np.vstack([np.array(poly.exterior.coords[:]) for poly in buffer])
    
        # If return_lonlat is True, convert x, y coordinates to longitude and latitude
        if return_lonlat:
            coords = np.array([self.xy2ll(*coord) for coord in coords])
    
        # Return the coordinates of the buffer
        return coords

    def add_index_range_to_profile(self, name, distance_range=None, normal_distance_range=None, min_depth=None, max_depth=None):
        '''
        Add index range to a profile.
    
        Args:
            * name                  : Name of the profile.
            * distance_range        : Range of distance along the profile, a tuple like (min, max).
            * normal_distance_range : Range of distance across the profile, a tuple like (min, max).
            * min_depth             : Minimum depth for selecting indices. Default is None.
            * max_depth             : Maximum depth for selecting indices. Default is None.
    
        Returns:
            * None. The index ranges are added to the profile in the attribute {profiles}
        '''
    
        # Check if the profile exists
        if name not in self.profiles:
            raise ValueError(f"No profile named {name}")
    
        # Get the profile
        profile = self.profiles[name]
    
        # Get the indices for Distance and Normal Distance within the given ranges
        if distance_range is None:
            distance_indices = np.arange(len(profile['Distance']))
        else:
            distance_indices = np.where((profile['Distance'] >= distance_range[0]) & 
                                        (profile['Distance'] <= distance_range[1]))[0]
    
        if normal_distance_range is None:
            normal_distance_indices = np.arange(len(profile['Normal Distance']))
        else:
            normal_distance_indices = np.where((profile['Normal Distance'] >= normal_distance_range[0]) & 
                                               (profile['Normal Distance'] <= normal_distance_range[1]))[0]
    
        # Get the indices for Depth within the given ranges
        if min_depth is None:
            min_depth = -np.inf
        if max_depth is None:
            max_depth = np.inf
        depth_indices = np.where((profile['Depth'] >= min_depth) & 
                                 (profile['Depth'] <= max_depth))[0]
    
        # Combine all indices
        combined_indices = np.intersect1d(distance_indices, normal_distance_indices)
        combined_indices = np.intersect1d(combined_indices, depth_indices)
    
        # Add the indices to the profile
        profile['Distance Indices'] = distance_indices
        profile['Normal Distance Indices'] = normal_distance_indices
        profile['Depth Indices'] = depth_indices
        profile['Combined Indices'] = combined_indices
    
        # All done
        return


# Alias
FaultWithSurfaceTraceAndRelocatedAftershocks = AdaptiveTriangularPatches

if __name__ == '__main__':
    pass