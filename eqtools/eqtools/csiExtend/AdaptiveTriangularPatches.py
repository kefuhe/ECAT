'''
Added by kfhe at 10/5/2023
Object :
    * Adaptive build triangular fault with or without relocated aftershocks as constraint
    * Fit relocated aftershocks to iso-depth curve
    * Fit relocated aftershocks to profiles along strike
'''

# Standard library imports
import sys
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
from .make_mesh_dutta import makemesh as make_mesh_dutta
from ..plottools import DegreeFormatter

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
        # If multiple points are found, only return the closest point on each side
        if len(left_indices) > 0:
            buffer_points[i, 0] = trace[left_indices[-1]]
        if len(right_indices) > 0:
            buffer_points[i, 1] = trace[right_indices[0]]
    return buffer_points


class AdaptiveTriangularPatches(TriangularPatches):
    def __init__(self, name: str, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):
        super().__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=verbose)
        self.top_size = None
        self.bottom_size = None
        self.relocated_aftershock_source = seismiclocations('relocs', utmzone=utmzone, lon0=lon0, lat0=lat0, verbose=verbose)
        self.mesh_func = None
        self.out_mesh = f'{name}.msh'
        self.profiles = {}

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
    
    def set_bottom_coords_from_geometry(self, top_tolerance=0.1, bottom_tolerance=0.1, lonlat=True, 
                                        sort_axis=0, sort_order='ascend', depth_tolerance=0.1):
        if not hasattr(self, 'edgepntsInVertices'):
            self.find_fault_edge_vertices(top_tolerance, bottom_tolerance)
        binds = self.edgepntsInVertices['bottom']
        if lonlat:
            bcoords = self.Vertices_ll[binds, :]
        else:
            bcoords = self.Vertices[binds, :]
        
        # Select points within depth tolerance
        # if depth_tolerance is not None:
        max_depth = np.max(bcoords[:, 2])
        min_depth = max_depth - depth_tolerance
        bcoords = bcoords[(bcoords[:, 2] >= min_depth) & (bcoords[:, 2] <= max_depth)]
        
        # Sort the coordinates
        if sort_order == 'ascend':
            bcoords = bcoords[bcoords[:, sort_axis].argsort()]
        elif sort_order == 'descend':
            bcoords = bcoords[bcoords[:, sort_axis].argsort()[::-1]]
        else:
            raise ValueError("Invalid value for sort_order. It should be 'ascend' or 'descend'.")
        
        self.set_bottom_coords(bcoords, lonlat=lonlat)
    
    def set_top_coords_from_geometry(self, top_tolerance=0.1, bottom_tolerance=0.1, lonlat=True, 
                                        sort_axis=0, sort_order='ascend', depth_tolerance=0.1):
        if not hasattr(self, 'edgepntsInVertices'):
            self.find_fault_edge_vertices(top_tolerance, bottom_tolerance)
        tinds = self.edgepntsInVertices['top']
        if lonlat:
            tcoords = self.Vertices_ll[tinds, :]
        else:
            tcoords = self.Vertices[tinds, :]
        
        # Select points within depth tolerance
        # if depth_tolerance is not None:
        min_depth = np.min(tcoords[:, 2])
        max_depth = min_depth + depth_tolerance
        tcoords = tcoords[(tcoords[:, 2] >= min_depth) & (tcoords[:, 2] <= max_depth)]

        # Sort the coordinates
        if sort_order == 'ascend':
            tcoords = tcoords[tcoords[:, sort_axis].argsort()]
        elif sort_order == 'descend':
            tcoords = tcoords[tcoords[:, sort_axis].argsort()[::-1]]
        else:
            raise ValueError("Invalid value for sort_order. It should be 'ascend' or 'descend'.")
        
        self.set_top_coords(tcoords, lonlat=lonlat)

    def set_top_coords_from_trace(self, discretized=False):
        """
        Sets the top coordinates from the fault trace.

        Parameters:
        discretized (bool): If True, uses the discretized fault trace. Otherwise
        """
        if discretized:
            x, y = self.xi, self.yi
        else:
            x, y = self.xf, self.yf 
        z = np.ones_like(x)*self.top
        self.set_coords(np.vstack((x, y, z)).T, lonlat=False, coord_type='top')

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
        scatter_props=None
    ):
        if type(methods) in (str,):
            methods = [methods]
        if len(methods) > 1:
            save2csi = False

        if fit_params is None:
            fit_params = {}

        # Choose the specific relocated aftershock source
        from copy import deepcopy
        seis = deepcopy(self.relocated_aftershock_source)
        if minlon is None or maxlon is None:
            minlon, maxlon = seis.lon.min(), seis.lon.max()
        if minlat is None or maxlat is None:
            minlat, maxlat = seis.lat.min(), seis.lat.max()
        seis.selectbox(minlon, maxlon, minlat, maxlat, depth=maxdepth, mindep=mindepth)
        x_values, y_values = seis.x, seis.y

        # Create a RegressionFitter instance
        fitter = RegressionFitter(x_values, y_values, degree=degree)

        if x_fit is None:
            X_ = np.linspace(x_values.min(), x_values.max(), 100)
        else:
            X_ = x_fit
        # Fit the model using the specified methods
        self.relociso_models = {}
        self.relociso_mses = {}
        for method in methods:
            model, mse = fitter.fit_model(method, **fit_params.get(method, {}))
            self.relociso_models[method] = model
            self.relociso_mses[method] = mse

        if show:
            fitter.plot_fit(self.relociso_models, self.relociso_mses, X_, 
                            show, save_fig, dpi, style,
                            show_error_in_legend=show_error_in_legend, fault=self, 
                            use_lon_lat=use_lon_lat, fontsize=fontsize, figsize=figsize, 
                            scatter_props=scatter_props)

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
        fit_indices=None  # Add a new parameter: a dictionary of indices used for fitting the relocated aftershocks
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
                axs[i].scatter(dist[mask], dep[mask], color="#cfcfd2", alpha=0.6, label='Not used')

                # Plot the selected points
                if len(indices) > 0:
                    axs[i].scatter(dist[indices], dep[indices], color="black", alpha=0.6, label='Used')

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
                    axs[i].plot(yfit, xfit, linewidth=line_width, color=palette[j], label=method)
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
        fig_width="nature_double",
        save_fig=None,
        dpi=300,
        legend_nrows=1,  # Add a new parameter: the number of rows of the legend
        legend_height_ratio=0.2,  # Add a new parameter: the ratio of the legend height to the subplot height
        min_seis_count=10,  # Add a new parameter: the minimum number of relocated aftershocks in a profile
        fit_indices=None,  # Add a new parameter: a dictionary of indices used for fitting the relocated aftershocks
        re_extract_profiles=False  # Add a new parameter: whether to re-extract profiles, default to False
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

        strike_dips = self.fit_and_plot_profiles(profiles, methods, degree, fit_params, show, 
                                                 style=style, aspect_ratio=aspect_ratio, 
                                                 fig_width=fig_width, save_fig=save_fig, dpi=dpi,
                                                 legend_nrows=legend_nrows, legend_height_ratio=legend_height_ratio, 
                                                 fit_indices=fit_indices)

        if write2file:
            strike_dips = pd.DataFrame(strike_dips, columns='profile_index lon lat strike dip mse method'.split())
            strike_dips.to_csv('reloc_xydipfit.csv', float_format='%.6f', index=False, header=True)

        # All Done
        return strike_dips
    
    def extract_profiles(self, prof_wid, prof_len, min_seis_count, center_coords=None):
        """
        提取剖面。

        参数:
        prof_wid (int): 剖面的宽度。
        prof_len (int): 剖面的长度。
        min_seis_count (int): 剖面中的重定位余震的最小数量。
        center_coords (numpy.ndarray, optional): 中心点坐标。默认为None，此时使用self.top_coords_ll作为中心点坐标。

        返回:
        list: 包含剖面信息的列表。

        异常:
        Exception: 如果没有任何剖面包含超过min_seis_count的重定位余震，抛出异常。

        """
        # 检查self.top_strike是否存在
        if not hasattr(self, 'top_strike'):
            raise ValueError("The attribute 'top_strike' is not set. Please calculate 'top_strike' before calling this function.")

        # azimuth is the Length direction.
        # Keep azimuth angle point to the left side of the strike angle.
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
            # 检查剖面中的重定位余震的数量, 如果数量小于min_seis_count, 则不进行拟合
            if len(self.profiles[prof_name]['Lon']) > min_seis_count:
                profiles.append(profi)
            else:
                # 如果数量小于min_seis_count, 则从self.profiles字典中删除这个剖面
                del self.profiles[prof_name]
                # Keep consistent between self.profiles and relocated_aftershock_source.profiles
                del self.relocated_aftershock_source.profiles[prof_name]
        
        # 如果profiles列表为空，抛出一个错误警告
        if not profiles:
            raise Exception(f"No profiles have more than {min_seis_count} relocated aftershocks.")
        
        return self.profiles

    def calculate_strike_at_point(self, lon, lat, discretized=True):
        from numpy import rad2deg, deg2rad
        if discretized:
            if not hasattr(self, 'strikei') or self.strikei is None:
                raise ValueError("The discretized trace or its strikes are not calculated yet.")
            # find the nearest two points in the discretized trace
            dists = np.sqrt((self.loni - lon)**2 + (self.lati - lat)**2)
            nearest_indices = np.argsort(dists)[:2]
            # if the point is at an endpoint, return the strike of the nearest point
            if np.min(dists) == dists[0] or np.min(dists) == dists[-1]:
                return self.strikei[np.argmin(dists)]
            # otherwise, return the average strike of the two nearest points
            strikes_rad = deg2rad(self.strikei[nearest_indices])
            average_strike_rad = np.arctan2(np.mean(np.sin(strikes_rad)), np.mean(np.cos(strikes_rad)))
            return rad2deg(average_strike_rad)
        else:
            if not hasattr(self, 'strikef') or self.strikef is None:
                raise ValueError("The undiscretized trace's strikes are not calculated yet.")
            # find the nearest two points in the fault trace
            dists = np.sqrt((self.lon - lon)**2 + (self.lat - lat)**2)
            nearest_indices = np.argsort(dists)[:2]
            # if the point is at an endpoint, return the strike of the nearest point
            if np.min(dists) == dists[0] or np.min(dists) == dists[-1]:
                return self.strikef[np.argmin(dists)]
            # otherwise, return the average strike of the two nearest points
            strikes_rad = deg2rad(self.strikef[nearest_indices])
            average_strike_rad = np.arctan2(np.mean(np.sin(strikes_rad)), np.mean(np.cos(strikes_rad)))
            return rad2deg(average_strike_rad)

    def calculate_top_strike(self, discretized=True):
            # Check if self.top_coords exists and is not None
            if not hasattr(self, 'top_coords') or self.top_coords is None:
                raise ValueError("The attribute 'top_coords' is not set. Please set 'top_coords' before calling this function.")

            # Initialize an empty list to store the strike values for each point
            top_strike = []

            # Iterate through each point in self.top_coords
            for coord in self.top_coords_ll:
                # Calculate the strike value for each point
                strike = self.calculate_strike_at_point(coord[0], coord[1], discretized=discretized)
                # Add the calculated strike value to the list
                top_strike.append(strike)

            # Convert the list to a numpy array and save it to self.top_strike
            self.top_strike = np.array(top_strike)
            return self.top_strike

    def extract_profile(self, azi, lonc, latc, prof_len, prof_wid, prof_name):
        # Check if the relocated aftershock source is set
        if not hasattr(self, 'relocated_aftershock_source'):
            raise ValueError("The attribute 'relocated_aftershock_source' is not set. Please set 'relocated_aftershock_source' before calling this function.")
        seis = self.relocated_aftershock_source

        seis.getprofile(prof_name, loncenter=lonc, latcenter=latc, length=prof_len, width=prof_wid, azimuth=azi)
        profile_info = seis.profiles[prof_name]
        self.profiles[prof_name] = profile_info
    
    def handle_buffer_nodes(self, xydip, buffer_nodes=None, buffer_radius=None, interpolation_axis='x'):
        """
        Handle buffer nodes. If buffer nodes and radius are provided, find the buffer node segments and calculate the dip for each node segment.

        Parameters:
        xydip (DataFrame): The DataFrame containing the coordinates and dips.
        buffer_nodes (numpy.ndarray): The buffer nodes. The shape is (n, 2) numpy array. Default is None.
        buffer_radius (float): The buffer radius. Default is None. Unit is km.
        interpolation_axis (str): The interpolation axis. It can be 'x' or 'y'. Default is 'x'.

        Returns:
        DataFrame: The updated DataFrame containing the coordinates and dips. 
        If buffer nodes and radius are provided, it will contain the buffer node information.
        """
        # 如果提供了缓冲节点和半径，找到缓冲节点段
        if buffer_nodes is not None and buffer_radius is not None:
            # 将缓冲节点的经纬度转换为x, y坐标
            buffer_nodes_ll = buffer_nodes
            buffer_xs, buffer_ys = self.ll2xy(buffer_nodes[:, 0], buffer_nodes[:, 1])
            buffer_nodes = np.hstack((buffer_xs[:, np.newaxis], buffer_ys[:, np.newaxis]))
            # 找到缓冲区内的点
            buffer_points = find_buffer_points(buffer_nodes, self.top_coords[:, :-1], buffer_radius)
            # 遍历每个缓冲节点段
            for i in range(buffer_nodes.shape[0] - 1):
                # 获取当前节点段的两个端点
                right_buffer_point, left_buffer_point = buffer_points[i][1, :], buffer_points[i+1][0, :]
                # 根据插值轴选择对应的坐标和掩码
                if interpolation_axis == 'x':
                    segment_mask = (xydip.x >= right_buffer_point[0]) & (xydip.x <= left_buffer_point[0])
                    segment_coords = xydip[segment_mask].x.values
                else:
                    segment_mask = (xydip.y >= right_buffer_point[1]) & (xydip.y <= left_buffer_point[1])
                    segment_coords = xydip[segment_mask].y.values
                # 获取当前节点段的倾角
                segment_dips = xydip[segment_mask].dip.values
                # 创建插值函数
                intp_func = interp1d(segment_coords, segment_dips, kind='nearest', 
                                    bounds_error=False, fill_value='extrapolate')
                # 计算缓冲节点的倾角
                left_dip = intp_func(left_buffer_point[0] if interpolation_axis == 'x' else left_buffer_point[1])
                right_dip = intp_func(right_buffer_point[0] if interpolation_axis == 'x' else right_buffer_point[1])
                buffer_dips = np.array([left_dip, right_dip])
                # 创建新的DataFrame，包含缓冲节点的坐标和倾角
                buffer_df = pd.DataFrame({
                    'x': [left_buffer_point[0], right_buffer_point[0]],
                    'y': [left_buffer_point[1], right_buffer_point[1]],
                    'lon': buffer_nodes_ll[[i, i+1], 0],
                    'lat': buffer_nodes_ll[[i, i+1], 1],
                    'dip': buffer_dips
                })
                # 合并新的DataFrame到xydip，并重置索引
                xydip = pd.concat([xydip, buffer_df]).drop_duplicates().reset_index(drop=True)
        self.xydip_ref = xydip
        return xydip

    def interpolate_top_dip_from_relocated_profile(
        self, 
        x_coords=None, 
        y_coords=None, 
        dips=None, 
        xydip_file=None, 
        is_utm=False, 
        discretization_interval=None,
        interpolation_axis='x', 
        save_to_file=False, 
        calculate_strike_along_trace=True,
        method='min_mse', # optimal: min_use, ols, theil_sen, ransac, huber, lasso, ridge, elasticnet, quantile
        buffer_nodes=None,
        buffer_radius=None,
        update_xydip_ref=False
    ):
        """
        对地震断层的倾向进行插值。

        参数:
        x_coords, y_coords, dips: 分别表示地震断层的x坐标、y坐标和倾角。
        xydip_file: 包含x、y坐标和倾角的文件路径。
        is_utm: 如果为True，表示x、y坐标是UTM坐标。否则，表示x、y坐标是经纬度。
        interpolation_axis: 用于插值的轴，可以是'x'或'y'。
        save_to_file: 如果为True，将结果保存到文件。
        calculate_strike_along_trace: 如果为True，沿着断层轨迹计算走向。
        buffer_nodes: 缓冲节点的坐标，用于将top_coords分段，然后每一段倾角自适应邻近插值。
        buffer_radius: 缓冲区的半径，保持分段转换区有一点的转换半径用linear插值。

        返回值:
        interpolated_main: 包含插值结果的DataFrame。
        """
        if update_xydip_ref or not hasattr(self, 'xydip_ref'):
            # 使用read_coordinates_and_dips函数读取坐标和倾角信息
            xydip = self.read_coordinates_and_dips(x_coords, y_coords, dips, xydip_file, is_utm, method)
            # 处理缓冲节点
            xydip = self.handle_buffer_nodes(xydip, buffer_nodes, buffer_radius, interpolation_axis)

        # 加密迹线，目的在于更准确的走向角插值
        if discretization_interval is not None:
            # self.discretize(every=discretization_interval)
            self.discretize_trace(every=discretization_interval)
        
        # Interpolation
        if interpolation_axis == 'x':
            x_values = xydip.x.values
            interpolated_x = self.top_coords[:, 0]
        else:
            x_values = xydip.y.values
            interpolated_x = self.top_coords[:, 1]
        indices = np.argsort(x_values)
        sorted_x_values = x_values[indices]
        sorted_dip_values = xydip.dip.values[indices]
        start_dip_fill = xydip.loc[indices[0], 'dip']
        end_dip_fill = xydip.loc[indices[-1], 'dip']

        interpolation_function = interp1d(sorted_x_values, sorted_dip_values, fill_value=(start_dip_fill, end_dip_fill), bounds_error=False)
        interpolated_dip = interpolation_function(interpolated_x)

        # 计算走向
        if not hasattr(self, 'strikei') or self.strikei is None or self.strikei.size != self.xi.size:
            strikei = self.calculate_trace_strike(use_discretized_trace=True, calculate_strike_along_trace=calculate_strike_along_trace)

        top_strike = self.calculate_top_strike(discretized=True)
        lon, lat, strike, dip = self.top_coords_ll[:, 0], self.top_coords_ll[:, 1], top_strike, interpolated_dip
        interpolated_main = pd.DataFrame(np.vstack((lon, lat, strike, dip)).T, columns='lon lat strike dip'.split())

        interpolated_main.loc[interpolated_main.dip>90, 'dip'] = interpolated_main.loc[interpolated_main.dip>90, 'dip'] - 180

        self.top_strike = top_strike
        self.top_dip = interpolated_main.loc[:, 'dip'].values # interpolated_dip

        if save_to_file:
            interpolated_main.to_csv(f'{self.name}_Trace_Dip.csv', index=False, header=True, float_format='%.6f')

        # All Done
        return interpolated_main

    def read_coordinates_and_dips(
        self, x_coords=None, y_coords=None, dips=None, 
        xydip_file=None, is_utm=False, method='min_mse'
        ):

        if x_coords is not None:
            if is_utm:
                columns = ['x', 'y', 'dip']
                xydip = pd.DataFrame(np.vstack((x_coords, y_coords, dips)).T, columns=columns)
                lon, lat = self.xy2ll(x_coords, y_coords)
                xydip['lon'] = lon
                xydip['lat'] = lat
            else:
                columns = ['lon', 'lat', 'dip']
                xydip = pd.DataFrame(np.vstack((x_coords, y_coords, dips)).T, columns=columns)
                x, y = self.ll2xy(x_coords, y_coords)
                xydip['x'] = x
                xydip['y'] = y
        elif xydip_file is not None:
            xydip = pd.read_csv(xydip_file, comment='#', header=0)
            xydip['original_order'] = range(len(xydip))  # 添加新的列
            if is_utm:
                lon, lat = self.xy2ll(xydip.lon.values, xydip.lat.values)
                xydip['lon'] = lon
                xydip['lat'] = lat
            else:
                x, y = self.ll2xy(xydip.lon.values, xydip.lat.values)
                xydip['x'] = x
                xydip['y'] = y

        if 'profile_index' in xydip.columns and 'mse' in xydip.columns:
            valid_methods = xydip['method'].unique().tolist() + ['min_mse']
            if method not in valid_methods:
                raise ValueError(f"Invalid method. Expected one of: {valid_methods}")

            if method == 'min_mse':
                xydip = xydip.loc[xydip.groupby('profile_index')['mse'].idxmin()]
            else:
                xydip = xydip[xydip['method'] == method]

            # 按照original_order的值排序，并重置索引
            xydip = xydip.sort_values('original_order').reset_index(drop=True)

        # 按照走向方向，使倾角范围改为0到180度(这保证了后续的插值是正确的)
        xydip.loc[xydip.dip < 0, 'dip'] += 180

        return xydip

    def calculate_trace_strike(self, use_discretized_trace=True, calculate_strike_along_trace=True):
        """
        计算断层迹线的走向角。可以选择是沿着迹线输入顺序的走向角还是反过来的走向角。

        参数:
        use_discretized_trace: 如果为True，将使用离散化的迹线坐标。否则，将使用原始的迹线坐标。
        calculate_strike_along_trace: 如果为True，将计算沿着迹线输入顺序的走向角。否则，将计算反过来的走向角。

        返回值:
        strike: 走向角，单位为度。

        """
        from numpy import rad2deg, deg2rad

        if use_discretized_trace:
            if not hasattr(self, 'xi') or self.xi is None:
                raise ValueError("The discretized trace's coordinates are not calculated yet.")
            x_coords, y_coords = self.xi, self.yi
        else:
            x_coords, y_coords = self.xf, self.yf

        x_diff = np.diff(x_coords)
        y_diff = np.diff(y_coords)
        # 计算每一段的走向角，并用方位角表示
        segment_strike = 90 - rad2deg(np.arctan2(y_diff, x_diff))
        # 将角度转换为弧度
        segment_strike_rad = deg2rad(segment_strike)
        # 计算中间点的走向角，这是两侧点走向角的平均值
        average_strike_rad = np.arctan2(
            (np.sin(segment_strike_rad[:-1]) + np.sin(segment_strike_rad[1:])) / 2,
            (np.cos(segment_strike_rad[:-1]) + np.cos(segment_strike_rad[1:])) / 2
        )
        average_strike = rad2deg(average_strike_rad)
        strike = np.concatenate(([segment_strike[0]], average_strike, [segment_strike[-1]]))

        if not calculate_strike_along_trace:
            strike += 180

        if use_discretized_trace:
            self.strikei = strike
        else:
            self.strikef = strike

        return strike
    
    def generate_top_bottom_from_nonlinear_soln(self, clon=None, clat=None, cdepth=None, 
                                                strike=None, dip=None, length=None, width=None, 
                                                top=None, depth=None):
        """
        Generate the top and bottom coordinates of the fault trace from the nonlinear solution.

        Parameters:
        clon (float): The longitude of the center point of the top line.
        clat (float): The latitude of the center point of the top line.
        cdepth (float): The depth of the center point of the top line.
        strike (float): The strike angle of the fault patch.
        dip (float): The dip angle of the fault patch.
        length (float): The length of the fault patch.
        width (float): The width of the fault patch.
        top (float): The top depth of the fault patch.
        depth (float): The bottom depth of the fault patch.

        Returns:
        top_coords: The top coordinates of the fault patch.
        bottom_coords: The bottom coordinates of the fault patch.
        """
        from numpy import deg2rad, sin, cos, tan

        if not all([clon, clat, cdepth, strike, dip, length]):
            raise ValueError("Please provide all the required parameters.")
        
        # Convert the strike and dip angles to radians
        str_rad = deg2rad(90 - strike)
        dip_rad = deg2rad(dip)
        half_length = length / 2.0
        # top or self.top, at least one of them should be provided
        if top is None:
            assert hasattr(self, 'top') and self.top is not None, "Please provide the top depth of the fault patch."
            top = self.top
        # depth or self.depth, at least one of them should be provided
        if depth is None:
            assert hasattr(self, 'depth') and self.depth is not None, "Please provide the depth of the fault patch."
            depth = self.depth

        if width is None:
            assert depth is not None, "Please provide the depth of the fault patch."
            width = (depth - top) / sin(dip_rad)
        
        # Calculate the top two end points of the fault patch
        cx, cy = self.ll2xy(clon, clat)
        cxy_trans = (cx + 1.j*cy) * np.exp(1.j*-str_rad)
        cxy_trans_neg = cxy_trans - half_length
        cxy_trans_pos = cxy_trans + half_length

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
        xy_dip_file=None, 
        fault_depth=None,
        update_self=True
    ):
        """
        Generate the bottom of the fault based on the dip angles determined from the relocated aftershock profile segments.
        This function mainly includes two steps:
        1. interpolate_dip_from_relocated_profile: Interpolate segment dip angles to trace densification points.
        2. make_bottom_from_reloc_dips: Translate to determine the bottom edge.

        Parameters:
        xy_dip_file: Path to the file containing dip angle information. If None, default values will be used.
        fault_depth: Depth of the fault. If None, self.depth will be used.

        Returns:
        None. However, the function updates the following instance variables:
        - bottom_coords: UTM coordinates of the bottom.
        - bottom_coords_ll: Latitude and longitude coordinates of the bottom.
        """
        from numpy import deg2rad, sin, cos, tan
        import os

        missing_depth_info = [attr for attr in ['depth', 'top'] if not hasattr(self, attr) or getattr(self, attr) is None]
        if missing_depth_info:
            raise ValueError(f"Please set the {', '.join(missing_depth_info)} attribute(s) before calling this function.")

        # Prepare Information
        fault_depth = fault_depth if fault_depth is not None else self.depth

        if xy_dip_file is not None:
            if not os.path.isfile(xy_dip_file):
                raise ValueError(f"The file {xy_dip_file} does not exist.")
            # name: lon, lat, strike, dip corrsponding to the top_coords
            dip_info = pd.read_csv(xy_dip_file, comment='#', header=0)
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
        # Dip angle transfer to 0~90, and strike angle transfer in order to make dip right hand is positive
        negative_dip_flag = dip_info.dip_rad < 0
        dip_info.loc[negative_dip_flag, 'strike_rad'] += np.pi
        dip_info.loc[negative_dip_flag, 'dip_rad'] = -dip_info.loc[negative_dip_flag, 'dip_rad']
        # update bottom_coords
        width = ((fault_depth - self.top)/sin(dip_info.dip_rad.values)).reshape(-1, 1)
        old_coords = np.vstack((x, y, np.ones_like(y)*self.top)).T 
        normal_x = cos(dip_info.dip_rad) * cos(-dip_info.strike_rad)
        normal_y = cos(dip_info.dip_rad) * sin(-dip_info.strike_rad)
        normal_z = sin(dip_info.dip_rad)
        normal_vector = np.vstack((normal_x, normal_y, normal_z)).T 
        new_coords = old_coords + normal_vector*width

        if update_self:
            self.bottom_coords = new_coords
            lon, lat = self.xy2ll(new_coords[:, 0], new_coords[:, 1])
            self.bottom_coords_ll = np.vstack((lon, lat, new_coords[:, -1])).T

        return new_coords

    def generate_bottom_from_relocated_aftershock_iso_depth_curve(self, 
                                                    fault_depth=None, 
                                                    update_self=True):
        '''
        根据重定位余震剖面的等深线生成断层的底部。

        参数:
        fault_depth: 断层的深度。如果为None，将使用self.depth。
        update_self: 是否将新生成的底部坐标赋给self.bottom_coords和self.bottom_coords_ll。默认为True。

        返回值:
        bottom_coords_ll: 断层底部的坐标（经纬深的形式）。
        '''
        # Generate the bottom coordinates from the relocated iso depth curve
        bottom_coords_ll = self.skin_curve_to_bottom(depth_extend=fault_depth, use_relocisocurve=True, update_self=update_self)

        return bottom_coords_ll
    
    def generate_bottom_from_single_dip(
        self, 
        dip_angle, 
        dip_direction, 
        update_self=True
    ):
        '''
        根据倾角和倾向生成断层的底部节点坐标。

        参数:
        dip_angle: 断层的倾斜角度，单位为度。
        dip_direction: 断层的倾斜方向，单位为度。
        discretization_interval: 用于离散化的参数。
        trace_tolerance: 用于离散化的容差。
        trace_fraction_step: 用于离散化的分数步长。
        trace_axis: 用于离散化的轴，可以是'x'或'y'。

        返回值:
        无返回值。但是，函数会更新以下实例变量：
        - bottom_coords: 底部的坐标，是一个二维数组，每一行表示一个点的坐标。
        - bottom_coords_ll: 底部的经纬度坐标，是一个二维数组，每一行表示一个点的坐标。
        '''
        # Check if self.top and self.bottom are set
        if not hasattr(self, 'top') or self.top is None:
            raise ValueError("Please set the 'top' attribute before calling this function.")
        if not hasattr(self, 'depth') or self.depth is None:
            raise ValueError("Please set the 'depth' attribute before calling this function.")

        # 检查self.top_coords是否存在且不为None
        if not hasattr(self, 'top_coords') or self.top_coords is None:
            raise ValueError("The attribute 'top_coords' is not set. Please set 'top_coords' before calling this function.")

        # 从self.top_coords中提取x_top, y_top和z_top
        x_top, y_top, z_top = self.top_coords[:, 0], self.top_coords[:, 1], self.top_coords[:, 2]

        # degree to rad
        dip_rad = dip_angle*np.pi/180.
        dip_direction_rad = dip_direction*np.pi/180.

        # Compute the bottom row
        depth_extend = self.depth - self.top
        width = depth_extend/np.sin(dip_rad)
        x_bottom = x_top + width*np.cos(dip_rad)*np.sin(dip_direction_rad)
        y_bottom = y_top + width*np.cos(dip_rad)*np.cos(dip_direction_rad)
        lon_bottom, lat_bottom = self.xy2ll(x_bottom, y_bottom)
        z_bottom = z_top + depth_extend

        # save to self.bottom
        bottom_coords = np.vstack((x_bottom, y_bottom, z_bottom)).T
        bottom_coords_ll = np.vstack((lon_bottom, lat_bottom, z_bottom)).T

        if update_self:
            self.bottom_coords = bottom_coords
            self.bottom_coords_ll = bottom_coords_ll

        return bottom_coords_ll
    
    def generate_mesh_dutta(self, perturbations, disct_z=None, bias=None, min_dx=None):
        """
        使用Dutta方法生成地震断层的网格。

        参数:
        perturbations: 扰动参数，用于控制地震断层的形状。[D1, D2, S1, S2]
        disct_z: 网格在z方向上的离散化参数。如果为None，则使用默认值。
        bias: 网格的偏差参数。如果为None，则使用默认值。
        min_dx: 网格的最小大小。如果为None，则使用默认值。

        返回:
        无返回值。但是，这个函数会修改self.top_coords和self.VertFace2csifault。

        注意:
        这个函数会修改self.top_coords的z坐标，使其变为负值。这是因为在地震模型中，z方向是向下的。
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

    def read_mesh_file(self, mshfile, tag=None, save2csi=True, element_name='triangle'):
        """
        这个函数的目的是读取一个网格文件，该文件可以是GMSH格式或Abaqus格式。

        参数:
        mshfile: 网格文件的路径。
        tag: 标签，只在读取Abaqus格式的文件时使用。
        save2csi: 如果为True，将结果保存到CSI格式的文件。
        element_name: 元素的名称，默认为'triangle'。
        """
        import meshio
        try:
            # 尝试使用meshio库读取文件
            data = meshio.read(mshfile)
            vertices = data.points

            # 如果z方向为负，则反向
            if np.mean(vertices[:, 2]) < 0:
                vertices[:, 2] = -vertices[:, 2]
            cells = data.cells
            cell_triangles = [imsh for imsh in cells if imsh.type == element_name]

            if tag is None:
                Faces = np.empty((0, 3), dtype=np.int32)
                for icell in cell_triangles:
                    Faces = np.append(Faces, icell.data)
            else:
                Faces = cell_triangles[tag].data
            
            Faces = Faces.reshape(-1, 3)

            if save2csi:
                self.VertFace2csifault(vertices, Faces)

            return vertices, Faces

        except Exception as e:
            # Catch any exceptions and try to read the file with a different method
            with open(mshfile, 'rt') as fin:
                version_line = fin.readline()

            if '4.1' in version_line:
                # Read the gms file with the read_gmsh_4_1 method
                return self.read_gmsh_4_1(mshfile, element_name)
            else:
                # Raise an error if the file is not Gmsh 4.1 version
                raise ValueError("Unable to read the file with meshio. The file is not Gmsh 4.1 version.") from e
    
    def convert_mesh_file(self, mshfile, output_format='inp', unit_conversion=1.0, 
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

            meshio.write(output_filename, meshio.Mesh(data.points, cells))

        except Exception as e:
            raise ValueError("Unable to read the file with meshio.") from e
    
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
    
    def VertFace2csifault(self, vertices, faces):
        """
        The purpose of this function is to convert vertex and face information into fault data in CSI format.

        Parameters:
        vertices: A two-dimensional array where each row represents the coordinates of a vertex.
        faces: A two-dimensional array where each row represents the vertex indices of a face.

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
        # Input validation
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices should be a 2D array with 3 columns.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces should be a 2D array with 3 columns.")

        self.Vertices = vertices
        self.vertices2ll()
        self.Faces = np.array(faces)
        self.patch = list(vertices[faces, :])
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

    def compute_triangle_areas(self):
        self.area = calculate_triangle_areas(self.Vertices, self.Faces)
        return self.area

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
        '''
        构建一个倾斜的断层。

        参数:
        topsize: 顶部的大小。如果为None，将使用meshfunc计算。
        bottomsize: 底部的大小。如果为None，将使用meshfunc计算。
        meshfunc: 用于计算网格大小的函数。如果为None，将使用topsize和bottomsize。
        outmesh: 输出的网格文件的路径。如果为None，将使用默认的路径。
        show: 如果为True，将显示图形用户界面。
        readmesh: 如果为True，将读取生成的网格文件。
        field_sizedict: 一个字典，包含'min_dx'和'bias'两个键，用于计算网格大小。
        segments_dict：一个字典，用于设置断层各边的网格划分参数，
            包含顶部、底部、左侧、右侧的段数以及顶底、左右的等比数列公比。
            keys: top_segments, bottom_segments, left_segments, right_segments, top_bottom_progression, left_right_progression
        verbose: Gmsh的日志级别，范围是0（关闭所有日志信息）到5（打印所有日志信息）。

        返回值:
        无返回值。但是，如果readmesh为True，将更新以下实例变量：
        - Vertices: 读取的网格文件的顶点坐标。
        - Faces: 读取的网格文件的面的顶点索引。
        - patch: 由面的顶点坐标组成的列表。
        - numpatch: 面的数量。
        - top: 所有顶点的最小z坐标。
        - depth: 所有顶点的最大z坐标。
        - z_patches: 从0到depth的等差数列，用于插值。
        - factor_depth: 深度因子，初始值为1。
        '''
        # 验证参数
        if segments_dict is not None:
            if any([top_size, bottom_size, mesh_func]):
                raise ValueError("segments_dict cannot be used with top_size, bottom_size, or (mesh_func, field_size_dict).")
        elif mesh_func is not None and field_size_dict is not None:
            if any([top_size, bottom_size]):
                raise ValueError("mesh_func and field_size_dict cannot be used with top_size or bottom_size.")
        elif top_size is not None and bottom_size is not None:
            if mesh_func:
                raise ValueError("top_size and bottom_size cannot be used with (mesh_func, field_size_dict).")
        else:
            raise ValueError("At least one of (top_size, bottom_size), (mesh_func, field_size_dict), or segments_dict must be provided.")
        
        # 如果提供了top_size或bottom_size，更新实例变量
        if top_size is not None:
            self.top_size = top_size
        if bottom_size is not None:
            self.bottom_size = bottom_size
        # 如果提供了mesh_func，更新实例变量
        if mesh_func is not None:
            self.mesh_func = mesh_func

        # 初始化gmsh
        # gmsh.initialize()
        gmsh.initialize('',False)
        # Set Gmsh's log level based on the verbose argument
        gmsh.option.setNumber("General.Terminal", verbose)
        gmsh.clear()
        gmsh.option.setNumber("Mesh.Algorithm", mesh_algorithm)
        # 下面是为了自己更正默认设置的用法
        # 2D mesh algorithm 
        # (1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 5: Delaunay, 6: Frontal-Delaunay, 
        #  7: BAMG, 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms, 11: Quasi-structured Quad)
        # Default value: 6

        # cube points:
        # lc = 0 # 用其他函数或其它方法时，需要将点周围的大小设置为0
        # 创建顶部和底部的点
        # 在生成网格时，将z值取反
        top_points = [gmsh.model.geo.addPoint(point[0], point[1], -point[2], self.top_size or 0.0) for point in self.top_coords]
        bottom_points = [gmsh.model.geo.addPoint(point[0], point[1], -point[2], self.bottom_size or 0.0) for point in self.bottom_coords]

        # 创建边缘
        top_curve = gmsh.model.geo.addSpline(top_points)
        bottom_curve = gmsh.model.geo.addSpline(bottom_points)
        left_edge = gmsh.model.geo.add_line(bottom_points[0], top_points[0])
        right_edge = gmsh.model.geo.add_line(top_points[-1], bottom_points[-1])

        if segments_dict is not None:
            self._set_segments(top_curve, bottom_curve, left_edge, right_edge, segments_dict)

        # 创建面
        face1 = gmsh.model.geo.add_curve_loop([top_curve, right_edge, -bottom_curve, left_edge])
        surface = gmsh.model.geo.add_surface_filling([face1])

        # 如果提供了segments_dict，设置面的分段数
        if segments_dict is not None:
            gmsh.model.geo.mesh.setTransfiniteSurface(surface)

        # 同步模型以反映最新的几何更改
        gmsh.model.geo.synchronize()

        # 如果提供了mesh_func，使用它来计算网格大小
        if mesh_func is not None:
            # field = gmsh.model.mesh.field
            # field.add("MathEval", 1)
            # field.setString(1, "F", meshfunc) # "0.1+1*(z/4)"
            # field.setAsBackgroundMesh(1)

            # Set discretization size with geometric progression from distance to the fault.
            
            # 禁用默认的网格大小计算方法
            gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)
            
            # 创建一个表示距离的字段
            field_distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field_distance, "CurvesList", [top_curve])

            # Second, we setup a field `field_size`, which is the mathematical expression
            # for the cell size as a function of the cell size on the fault, the distance from
            # the fault (as given by `field_size`, and the bias factor.
            # The `GenerateMesh` class includes a special function `get_math_progression` 
            # for creating the string with the mathematical function.
            # 创建一个表示网格大小的字段
            field_size = gmsh.model.mesh.field.add("MathEval")
            math_exp = self.get_math_progression(field_distance, min_dx=field_size_dict['min_dx'], bias=field_size_dict['bias'])
            gmsh.model.mesh.field.setString(field_size, "F", math_exp)

            # 使用field_size作为网格的大小
            gmsh.model.mesh.field.setAsBackgroundMesh(field_size)

        # 移除点和线
        gmsh.model.remove_entities(gmsh.model.getEntities(0))
        gmsh.model.remove_entities(gmsh.model.getEntities(1))

        # # 优化网格
        # gmsh.model.geo.mesh.setSmoothing(2, face1, 100)
        gmsh.model.mesh.refine()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize(optimize_method)

        # 写入网格数据
        if out_mesh is not None:
            self.out_mesh = out_mesh
        if write2file:
            gmsh.write(self.out_mesh)

        # 如果需要，显示图形用户界面
        if show:
            if 'close' not in sys.argv:
                gmsh.fltk.run()

        # 如果需要，读取生成的网格文件
        if read_mesh:
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            vertices = node_coords.reshape(-1, 3)
            # 如果z方向为负，则反向
            if np.mean(vertices[:, 2]) < 0:
                vertices[:, 2] = -vertices[:, 2]
            _, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
            faces = np.array(node_tags[0]).reshape(-1, 3) - 1
            self.VertFace2csifault(vertices, faces)

        # 结束gmsh API
        gmsh.finalize()

    def _set_segments(self, top_curve, bottom_curve, left_edge, right_edge, segments_dict):
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

    def generate_multilayer_mesh(self, coords, sizes=None, mesh_func=None, 
                                 out_mesh=None, write2file=False, show=True, read_mesh=True, 
                                 field_size_dict={'min_dx': 3, 'bias': 1.05},
                                 mesh_algorithm=2, # 5: Delaunay, 6: Frontal-Delaunay
                                 optimize_method='Laplace2D', verbose=5):
        # 初始化gmsh
        gmsh.initialize()
        gmsh.initialize('',False)
        # Set Gmsh's log level based on the verbose argument
        gmsh.option.setNumber("General.Terminal", verbose)
        gmsh.clear()
        # 设置网格算法
        gmsh.option.setNumber("Mesh.Algorithm", mesh_algorithm)

        # 创建顶部和底部的点
        # 在生成网格时，将z值取反
        top_points = [gmsh.model.geo.addPoint(point[0], point[1], -point[2], self.top_size or 0.0) for point in self.top_coords]
        bottom_points = [gmsh.model.geo.addPoint(point[0], point[1], -point[2], self.bottom_size or 0.0) for point in self.bottom_coords]

        # 创建中间层的点
        intermed_points = []
        if coords is not None:
            for layer, size in zip(coords, sizes or [0.0]*len(coords)):
                intermed_points.append([gmsh.model.geo.addPoint(point[0], point[1], -point[2], size) for point in layer])

        # 将顶部和底部的点添加到中间层的列表中
        all_points = [top_points] + intermed_points + [bottom_points]

        # 创建边缘
        all_curves = [gmsh.model.geo.addSpline(layer) for layer in all_points]
        # for icurve in all_curves:
        #     print(icurve)

        # 创建左右边
        left_edges = [gmsh.model.geo.addLine(all_points[i][0], all_points[i+1][0]) for i in range(len(all_points) - 1)]
        right_edges = [gmsh.model.geo.addLine(all_points[i][-1], all_points[i+1][-1]) for i in range(len(all_points) - 1)]

        # 同步模型
        # gmsh.model.geo.synchronize()

        # 创建面
        for i in range(len(all_curves) - 1):
            # # 获取曲线的起点和终点
            # p1, p2 = gmsh.model.getBoundary([(1, all_curves[i])], oriented=False)
            # p3, p4 = gmsh.model.getBoundary([(1, all_curves[i+1])], oriented=False)
            # p5, p6 = gmsh.model.getBoundary([(1, right_edges[i])], oriented=False)
            # p7, p8 = gmsh.model.getBoundary([(1, left_edges[i])], oriented=False)

            # # 检查曲线的起点和终点是否正确连接
            # if not (p2 == p5 and p6 == p3 and p4 == p7 and p8 == p1):
            #     print(f"Curve loop {i} is not correctly connected.")
            #     print(f"all_curves[{i}]: start point = {p1}, end point = {p2}")
            #     print(f"all_curves[{i+1}]: start point = {p3}, end point = {p4}")
            #     print(f"right_edges[{i}]: start point = {p5}, end point = {p6}")
            #     print(f"left_edges[{i}]: start point = {p7}, end point = {p8}")
            #     continue
            face = gmsh.model.geo.addCurveLoop([all_curves[i], right_edges[i], -all_curves[i+1], -left_edges[i]])
            gmsh.model.geo.addSurfaceFilling([face])

        # 同步模型以反映最新的几何更改
        gmsh.model.geo.synchronize()

        # 如果提供了mesh_func，使用它来计算网格大小
        if mesh_func is not None:
            # 禁用默认的网格大小计算方法
            gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)
            
            # 创建一个表示距离的字段
            field_distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field_distance, "CurvesList", [all_curves[0]])

            # 创建一个表示网格大小的字段
            field_size = gmsh.model.mesh.field.add("MathEval")
            math_exp = self.get_math_progression(field_distance, min_dx=field_size_dict['min_dx'], bias=field_size_dict['bias'])
            gmsh.model.mesh.field.setString(field_size, "F", math_exp)

            # 使用field_size作为网格的大小
            gmsh.model.mesh.field.setAsBackgroundMesh(field_size)

        # 移除点和线
        gmsh.model.remove_entities(gmsh.model.getEntities(0))
        gmsh.model.remove_entities(gmsh.model.getEntities(1))

        # # 优化网格
        # gmsh.model.geo.mesh.setSmoothing(2, face1, 100)
        gmsh.model.mesh.refine()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize(optimize_method)

        # 写入网格数据
        if out_mesh is not None:
            self.out_mesh = out_mesh
        if write2file:
            gmsh.write(self.out_mesh)

        # 如果需要，显示图形用户界面
        if show:
            if 'close' not in sys.argv:
                gmsh.fltk.run()

        # 如果需要，读取生成的网格文件
        if read_mesh:
            # self.read_mesh_file(self.out_mesh)
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            vertices = node_coords.reshape(-1, 3)
            # 如果z方向为负，则反向
            if np.mean(vertices[:, 2]) < 0:
                vertices[:, 2] = -vertices[:, 2]
            _, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
            faces = np.array(node_tags[0]).reshape(-1, 3) - 1
            self.VertFace2csifault(vertices, faces)

        # 结束gmsh API
        gmsh.finalize()

    def get_math_progression(self, field_distance, min_dx, bias):
        """Generate the Gmsh MathEval string corresponding to the cell size as a function
        of distance, starting cell size, and bias factor.

        The expression is min_dx * bias**n, where n is the number of cells from the fault.
        n = log(1+distance/min_dx*(bias-1))/log(bias)

        In finding the expression for `n`, we make use that the sum of a geometric series with n
        terms Sn = min_dx * (1 + bias + bias**2 + ... + bias**n) = min_dx * (bias**n - 1)/(bias - 1).
        """
        return f"{min_dx}*{bias}^(Log(1.0+F{field_distance}/{min_dx}*({bias}-1.0))/Log({bias}))"

    def skin_curve_to_bottom(self, 
                            depth_extend=None, 
                            interval_num=20,
                            curve_top=None,
                            curve_bottom=None,
                            use_relocisocurve=False,
                            update_self=True):
        '''
        根据原来的迹线和重定位余震确定的底部曲线，生成扩展到指定深度的底部坐标。
        "skin curve"这个名字来源于商业软件Trelis中的对应方法。

        参数:
        depth_extend: 扩展到的深度。如果为None，将使用self.depth。
        interval_num: 插值的间隔数量。
        curve_top: 顶部的曲线。如果为None，将使用self.top_coords。
        curve_bottom: 底部的曲线。如果为None，将使用self.relocisocurve。
        update_self: 是否将新生成的底部坐标赋给self.bottom_coords和self.bottom_coords_ll。默认为True。

        返回值:
        extended_bottom_coords_ll: 扩展到指定深度的底部坐标（经纬深的形式）。
        '''

        if self.top is None or self.depth is None:
            raise ValueError("Please set the 'top' and 'depth' attributes before calling this function.")

        depth = depth_extend if depth_extend is not None else self.depth
        top = np.mean(curve_top[:, -1]) if curve_top is not None else self.top

        from shapely.geometry import LineString
        import numpy as np

        # 1. interpolate
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
            normalized_vector = vector/np.linalg.norm(vector)

            # Translate the top coordinates to the specific depth along the direction of the vector
            translate_vector = depth_extend/normalized_vector[-1]*normalized_vector
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
                      show=True, draw_trace_arrow=True):

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
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from ..plottools import sci_plot_style
        with sci_plot_style(style, fontsize=fontsize, figsize=figsize):
            fig = plt.figure(facecolor='white')  # none: 设置背景为透明
            ax = fig.add_subplot(111, projection='3d', facecolor='white')  # none: 设置轴的背景色为透明

            # 设置格网线
            # ax.xaxis._axinfo["grid"]['color'] =  'black'
            # ax.yaxis._axinfo["grid"]['color'] =  'black'
            # ax.zaxis._axinfo["grid"]['color'] =  'black'
            # ax.xaxis._axinfo["grid"]['linestyle'] =  ':'
            # ax.yaxis._axinfo["grid"]['linestyle'] =  ':'
            # ax.zaxis._axinfo["grid"]['linestyle'] =  ':'
            # ax.xaxis._axinfo["grid"]['linewidth'] =  0.5
            # ax.yaxis._axinfo["grid"]['linewidth'] =  0.5
            # ax.zaxis._axinfo["grid"]['linewidth'] =  0.5

            # 设置视角
            if elev is not None and azim is not None:
                ax.view_init(elev=elev, azim=azim)

            # 画断层面
            x = self.Vertices_ll[:, 0]
            y = self.Vertices_ll[:, 1]
            z = self.Vertices_ll[:, 2]

            surf = ax.plot_trisurf(x, y, z, triangles=self.Faces, linewidth=0.5, edgecolor='#7291c9', zorder=1)
            surf.set_facecolor((0, 0, 0, 0))  # 设置面颜色为透明

            ax.set_zlabel('Depth')

            ax.invert_zaxis()  # 反转z轴，使值从0到20向下显示

            ax.xaxis.pane.fill = False  # 设置x轴面板为透明
            ax.yaxis.pane.fill = False  # 设置y轴面板为透明
            ax.zaxis.pane.fill = False  # 设置z轴面板为透明

            def project_point_to_plane(points, plane_point, normal):
                # 计算点到平面的投影
                point_vectors = points - plane_point
                distances = np.dot(point_vectors, normal) / np.linalg.norm(normal)
                projections = points - np.outer(distances, normal)
                return projections

            # 画散点
            if scatter_props is None:
                scatter_props = {'color': '#ff0000', 'ec': 'k', 'linewidths': 0.5}

            if max_depth is None:
                max_depth = np.max(self.Vertices_ll[:, 2])

            for iprofile in self.profiles:
                profile = self.profiles[iprofile]
                end_points_ll = np.array(profile['EndPointsLL'])
                end_points_ll_reversed = end_points_ll[::-1]  # 反转end_points_ll

                # 创建一个包含5个点的列表，前两个点在0深度，接下来两个点在最大深度，最后一个点回到第一个点
                points = np.concatenate((end_points_ll, end_points_ll_reversed, end_points_ll[0:1]), axis=0)
                depths = [0, 0, max_depth, max_depth, 0]

                # 画出这5个点
                for i in range(len(points) - 1):
                    ax.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], [depths[i], depths[i+1]], 'k--', lw=1.0, zorder=3)

                # 计算平面的一个点和法向量
                plane_point = np.append(end_points_ll[0], 0)  # 添加深度值
                normal = np.cross(np.append(end_points_ll[1], 0) - plane_point, [0, 0, 1])  # 添加深度值
                normal /= np.linalg.norm(normal)  # 归一化法向量

                lon = profile['Lon']
                lat = profile['Lat']
                depth = profile['Depth']
                points = np.array([lon, lat, depth]).T

                # 计算投影点并绘制
                if 'Projections' in profile:  # 如果已经有投影坐标，直接使用
                    projections = profile['Projections']
                else:  # 如果没有投影坐标，进行投影并存储到剖面中
                    projections = project_point_to_plane(points, plane_point, normal)

                # 如果需要将剖面中心偏移到断层上
                if offset_to_fault:
                    lonc, latc = profile['Center']
                    # 计算偏移向量
                    top_offset_vector = self.compute_offset_vector(iprofile, in_lonlat=True, use_top_coords=True)
                    bottom_offset_vector = self.compute_offset_vector(iprofile, in_lonlat=True, use_top_coords=False)
                    if top_offset_vector is not None and bottom_offset_vector is not None:
                        # 将剖面中心偏移到断层上
                        top_offset_vector.append(self.top)  # 在z方向添加self.top
                        bottom_offset_vector.append(self.depth)  # 在z方向添加self.depth
                        center = np.array([lonc, latc, self.top])
                        center_top = center + np.array(top_offset_vector)
                        center_bottom = center + np.array(bottom_offset_vector)

                        # 绘制交线
                        ax.plot([center_top[0], center_bottom[0]], 
                            [center_top[1], center_bottom[1]], 
                            [center_top[2], center_bottom[2]], color='#2e8b57', linestyle='-', zorder=4, lw=2.0)

                        # 将交线保存到profile中
                        profile['Intersection'] = np.array([center_top, center_bottom])

                    top_offset_vector = np.array(top_offset_vector) if top_offset_vector is not None else np.array([0, 0, 0])
                    if 'Projections' not in profile:
                        if top_offset_vector.size == 2:
                            top_offset_vector = np.append(top_offset_vector, self.top) # 0深度?
                        profile['Projections'] = projections + top_offset_vector[None, :]

                # 绘制所有的投影点
                projections = np.array(projections)
                ax.scatter(projections[:, 0], projections[:, 1], projections[:, 2], **scatter_props, zorder=2)

            # 获取x轴和y轴的范围
            x_range = np.ptp(ax.get_xlim())
            y_range = np.ptp(ax.get_ylim())

            if z_height is not None:
                # 计算x轴和y轴的比例
                xy_ratio = x_range / y_range
                # 设置图形的高度比例
                ax.set_box_aspect([xy_ratio, 1, z_height])
            else:
                # 设置图形的高度比例
                ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([shape[0], shape[1], shape[2], 1]))
            # 设置x轴和y轴的标签
            formatter = DegreeFormatter()
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

        # 显示或保存图像
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # 调整图像边界
        if save_fig:
            plt.savefig(file_path, dpi=dpi) # , bbox_inches='tight', pad_inches=0.1
        # 显示图形
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

        # 创建3D图形
        fig = go.Figure()

        def project_point_to_plane(points, plane_point, normal):
            # 计算点到平面的投影
            point_vectors = points - plane_point
            distances = np.dot(point_vectors, normal) / np.linalg.norm(normal)
            projections = points - np.outer(distances, normal)
            return projections

        # 添加散点图
        if scatter_props is None:
            scatter_props = {'color': '#ff0000', 'ec': 'k', 'linewidths': 0.5}

        if max_depth is None:
            max_depth = np.max(self.Vertices_ll[:, 2])

        for iprofile in self.profiles:
            profile = self.profiles[iprofile]
            end_points_ll = np.array(profile['EndPointsLL'])
            end_points_ll_reversed = end_points_ll[::-1]  # 反转end_points_ll

            # 创建一个包含5个点的列表，前两个点在0深度，接下来两个点在最大深度，最后一个点回到第一个点
            points = np.concatenate((end_points_ll, end_points_ll_reversed, end_points_ll[0:1]), axis=0)
            depths = [0, 0, max_depth, max_depth, 0]

            # 画出这5个点
            for i in range(len(points) - 1):
                fig.add_trace(go.Scatter3d(x=[points[i][0], points[i+1][0]], 
                                           y=[points[i][1], points[i+1][1]], 
                                           z=[depths[i], depths[i+1]], 
                                           mode='lines',
                                           line=dict(color='black', width=2),
                                           showlegend=False))

            # 计算平面的一个点和法向量
            plane_point = np.append(end_points_ll[0], 0)  # 添加深度值
            normal = np.cross(np.append(end_points_ll[1], 0) - plane_point, [0, 0, 1])  # 添加深度值
            normal /= np.linalg.norm(normal)  # 归一化法向量

            lon = profile['Lon']
            lat = profile['Lat']
            depth = profile['Depth']
            points = np.array([lon, lat, depth]).T

            # 计算投影点并绘制
            if 'Projections' in profile:  # 如果已经有投影坐标，直接使用
                projections = profile['Projections']
            else:  # 如果没有投影坐标，进行投影并存储到剖面中
                projections = project_point_to_plane(points, plane_point, normal)

            # 如果需要将剖面中心偏移到断层上
            if offset_to_fault:
                lonc, latc = profile['Center']
                # 计算偏移向量
                top_offset_vector = self.compute_offset_vector(iprofile, in_lonlat=True, use_top_coords=True)
                bottom_offset_vector = self.compute_offset_vector(iprofile, in_lonlat=True, use_top_coords=False)
                if top_offset_vector is not None and bottom_offset_vector is not None:
                    # 将剖面中心偏移到断层上
                    top_offset_vector.append(self.top)  # 在z方向添加self.top
                    bottom_offset_vector.append(self.depth)  # 在z方向添加self.depth
                    center = np.array([lonc, latc, self.top])
                    center_top = center + np.array(top_offset_vector)
                    center_bottom = center + np.array(bottom_offset_vector)

                    # 绘制交线
                    fig.add_trace(go.Scatter3d(x=[center_top[0], center_bottom[0]], 
                                               y=[center_top[1], center_bottom[1]], 
                                               z=[center_top[2], center_bottom[2]], 
                                               mode='lines',
                                               line=dict(color='#2e8b57', width=2),
                                               showlegend=False))

                    # 将交线保存到profile中
                    profile['Intersection'] = np.array([center_top, center_bottom])

                    top_offset_vector = np.array(top_offset_vector) if top_offset_vector is not None else np.array([0, 0, 0])
                    if 'Projections' not in profile:
                        profile['Projections'] = projections + top_offset_vector[None, :]

                    # 绘制所有的余震投影点
                    projections = np.array(projections)
                    fig.add_trace(go.Scatter3d(x=projections[:, 0], y=projections[:, 1], z=projections[:, 2], 
                                               mode='markers', 
                                               marker=dict(size=2, color=scatter_props['color']),
                                               showlegend=False))

        # 添加表面
        x = self.Vertices_ll[:, 0]
        y = self.Vertices_ll[:, 1]
        z = self.Vertices_ll[:, 2]
        # 绘制3D网格
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, 
                                i=self.Faces[:, 0], 
                                j=self.Faces[:, 1], 
                                k=self.Faces[:, 2], 
                                color='lightblue', 
                                opacity=0.50))

        # 绘制网格线
        for face in self.Faces:
            for i in range(3):
                fig.add_trace(go.Scatter3d(x=[x[face[i]], x[face[(i+1)%3]]], 
                                           y=[y[face[i]], y[face[(i+1)%3]]], 
                                           z=[z[face[i]], z[face[(i+1)%3]]], 
                                           mode='lines',
                                           line=dict(color='#7291c9', width=0.5),
                                            showlegend=False))

        # 设置视角
        if elev is not None and azim is not None:
            fig.update_layout(scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                                center=dict(x=0, y=0, z=-0.5), 
                                                eye=dict(x=1.25*azim, y=1.25*azim, z=1.25*elev)))

        # 计算x轴和y轴的范围
        x_range = np.max(self.Vertices_ll[:, 0]) - np.min(self.Vertices_ll[:, 0])
        y_range = np.max(self.Vertices_ll[:, 1]) - np.min(self.Vertices_ll[:, 1])

        # 计算x轴和y轴的比例
        xy_ratio = x_range / y_range

        # 设置x轴、y轴和z轴的标签和比例
        fig.update_layout(scene=dict(xaxis=dict(title_text="Lon"),
                                         yaxis=dict(title_text="Lat"),
                                         zaxis=dict(autorange="reversed", 
                                                    title_text="Depth", 
                                                    tickvals=list(range(0, int(max_depth)+1, 10)), 
                                                    ticktext=[str(i) for i in range(0, int(max_depth)+1, 10)]),
                                         aspectratio=dict(x=xy_ratio, y=1, z=z_height)))

        # 显示或保存图像
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
        创建断层迹线的缓冲区。

        参数:
        buffer_size (float): 缓冲区的大小，单位为km。
        use_discrete_trace (bool, 可选): 是否使用离散迹线。默认为False，表示使用连续迹线。
        return_lonlat (bool, 可选): 是否返回经纬度坐标。默认为False，表示返回x, y坐标。

        返回:
        numpy.ndarray: 缓冲区的坐标，形状为(n, 2)。如果return_lonlat为True，返回的是经纬度坐标；否则，返回的是x, y坐标。

        示例:
        >>> obj = YourClass(xi, yi, xf, yf)
        >>> buffer_coords = obj.create_fault_trace_buffer(1, use_discrete_trace=True, return_lonlat=True)
        """
        from shapely.geometry import LineString, Polygon
        import numpy as np

        # 根据use_discrete_trace的值选择使用哪个迹线
        x = self.xi if use_discrete_trace else self.xf
        y = self.yi if use_discrete_trace else self.yf

        # 创建LineString对象
        line = LineString(zip(x, y))

        # 创建缓冲区
        buffer = line.buffer(buffer_size)

        # 如果buffer是Polygon对象，获取其外部坐标并转换为numpy数组
        if isinstance(buffer, Polygon):
            coords = np.array(buffer.exterior.coords[:])
        else:  # 如果buffer是MultiPolygon对象，获取所有Polygon的外部坐标并转换为numpy数组
            coords = np.vstack([np.array(poly.exterior.coords[:]) for poly in buffer])

        # 如果return_lonlat为True，将x, y坐标转换为经纬度
        if return_lonlat:
            coords = np.array([self.xy2ll(*coord) for coord in coords])

        # 返回缓冲区的坐标
        return coords

    def add_index_range_to_profile(self, name, distance_range=None, normal_distance_range=None):
        '''
        Add index range to a profile.

        Args:
            * name                  : Name of the profile.
            * distance_range        : Range of Distance, a tuple like (min, max).
            * normal_distance_range : Range of Normal Distance, a tuple like (min, max).

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

        # Add the indices to the profile
        profile['Distance Indices'] = distance_indices
        profile['Normal Distance Indices'] = normal_distance_indices

        # All done
        return


# Alias
FaultWithSurfaceTraceAndRelocatedAftershocks = AdaptiveTriangularPatches

if __name__ == '__main__':
    pass