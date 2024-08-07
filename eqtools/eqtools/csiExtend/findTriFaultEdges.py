import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def find_adjacent_triangles(vertex_indices: np.ndarray) -> list:
    """
    Find the adjacent triangles for each triangle in a mesh.

    Args:
        vertex_indices: A numpy array of shape (n, 3) containing the vertex indices for each triangle in the mesh.

    Returns:
        A list of length n containing the indices of the adjacent triangles for each triangle in the mesh.
    """
    num_triangles = vertex_indices.shape[0]
    adjacent_triangles = [[] for i in range(num_triangles)]

    # create a boolean mask for triangles that have at least two vertices in common with each other
    common_vertices_mask = np.zeros((num_triangles, num_triangles), dtype=np.bool_)
    for i in range(num_triangles):
        # compare the vertices of triangle i with the vertices of triangles i+1 to n-1
        # using np.isin and np.logical_and.reduce to find common vertices
        common_vertices_mask[i, i+1:] = np.sum(np.isin(vertex_indices[i+1:], vertex_indices[i]), axis=1) == 2
        common_vertices_mask[i+1:, i] = common_vertices_mask[i, i+1:]

    # find adjacent triangles
    for i in range(num_triangles):
        adjacent_triangles[i] = np.where(common_vertices_mask[i])[0].tolist()

    return adjacent_triangles


def find_top_bottom_sides_triangles(vertex_indices: np.ndarray, vertex_coordinates: np.ndarray, top_tolerance: float, bottom_tolerance: float) -> tuple:
    """
    Find the triangles on the top, bottom, left and right boundaries of a mesh.

    Args:
        vertex_indices: A numpy array of shape (n, 3) containing the vertex indices for each triangle in the mesh.
        vertex_coordinates: A numpy array of shape (m, 3) containing the coordinates of each vertex in the mesh.
        top_tolerance: A float specifying the tolerance for determining the top boundary of the mesh.
        bottom_tolerance: A float specifying the tolerance for determining the bottom boundary of the mesh.

    Returns:
        A tuple containing the indices of the triangles on the top, bottom, left and right boundaries of the mesh.
    """
    num_triangles = vertex_indices.shape[0]
    top_depth = vertex_coordinates[:, 2].min()
    bottom_depth = vertex_coordinates[:, 2].max()

    # create a boolean mask for vertices that are at the top or bottom depth
    top_mask = np.isclose(vertex_coordinates[:, 2], top_depth, rtol=0, atol=top_tolerance)
    bottom_mask = np.isclose(vertex_coordinates[:, 2], bottom_depth, rtol=0, atol=bottom_tolerance)

    # create a boolean mask for triangles that have at least two vertices at the top or bottom depth
    top_triangles_mask = np.sum(top_mask[vertex_indices], axis=1) >= 2
    bottom_triangles_mask = np.sum(bottom_mask[vertex_indices], axis=1) >= 2

    # find triangles that are on the top or bottom boundary
    top_boundary_triangles = np.where(top_triangles_mask)[0].tolist()
    bottom_boundary_triangles = np.where(bottom_triangles_mask)[0].tolist()

    # find triangles that are on the left or right boundary
    left_and_right_boundary_triangles_mask = ~(top_triangles_mask | bottom_triangles_mask)
    left_and_right_boundary_triangles = []
    for i, tri_i in zip(np.arange(num_triangles)[left_and_right_boundary_triangles_mask], vertex_indices[left_and_right_boundary_triangles_mask]):
        # Count the number of triangles that share two vertices with the current triangle
        num_common_side_triangles = np.count_nonzero(np.sum(np.isin(vertex_indices, tri_i), axis=1) == 2)
        if num_common_side_triangles < 3:
            left_and_right_boundary_triangles.append(i)

    return top_boundary_triangles, bottom_boundary_triangles, left_and_right_boundary_triangles


def find_top_bottom_sides_triangles(adjacent_maps: list, vertex_indices: np.ndarray, vertex_coordinates: np.ndarray, top_tolerance: float, bottom_tolerance: float) -> tuple:
    """
    Find the triangles on the top, bottom, left and right boundaries of a mesh.

    Args:
        adjacent_maps: Adjacent map of the triangle mesh.
        vertex_indices: A numpy array of shape (n, 3) containing the vertex indices for each triangle in the mesh.
        vertex_coordinates: A numpy array of shape (m, 3) containing the coordinates of each vertex in the mesh.
        top_tolerance: A float specifying the tolerance for determining the top boundary of the mesh.
        bottom_tolerance: A float specifying the tolerance for determining the bottom boundary of the mesh.

    Returns:
        A tuple containing the indices of the triangles on the top, bottom, left and right boundaries of the mesh.
    """
    # A numpy array containing the indices of the triangles on the side edge of the mesh.
    sideedge_triangle_index = np.where(np.array([len(adjacent_maps[i]) < 3 for i in range(len(adjacent_maps))]))[0]

    sideedge_inices = vertex_indices[sideedge_triangle_index]
    sideedge_coordinates = vertex_coordinates[sideedge_inices]
    top_depth = vertex_coordinates[:, 2].min()
    bottom_depth = vertex_coordinates[:, 2].max()

    # create a boolean mask for vertices that are at the top or bottom depth
    top_mask = np.isclose(sideedge_coordinates[:, :, 2], top_depth, rtol=0, atol=top_tolerance)
    bottom_mask = np.isclose(sideedge_coordinates[:, :, 2], bottom_depth, rtol=0, atol=bottom_tolerance)

    # create a boolean mask for triangles that have at least two vertices at the top or bottom depth
    top_triangles_mask = np.sum(top_mask, axis=1) >= 2
    bottom_triangles_mask = np.sum(bottom_mask, axis=1) >= 2

    # find triangles that are on the top or bottom boundary
    sideedge_triangle_index_copy = np.array(sideedge_triangle_index)
    top_boundary_triangles = sideedge_triangle_index_copy[top_triangles_mask].tolist()
    bottom_boundary_triangles = sideedge_triangle_index_copy[bottom_triangles_mask].tolist()

    # find triangles that are on the left or right boundary
    left_and_right_boundary_triangles_mask = ~(top_triangles_mask | bottom_triangles_mask)
    left_and_right_boundary_triangles = sideedge_triangle_index_copy[left_and_right_boundary_triangles_mask].tolist()

    return top_boundary_triangles, bottom_boundary_triangles, left_and_right_boundary_triangles


def find_left_and_right_boundary_triangles(sideedge_triangle_index: np.ndarray, vertex_indices: np.ndarray, vertex_coordinates: np.ndarray) -> tuple:
    """
    Find the triangles on the left and right boundaries of a mesh.

    Args:
        sideedge_triangle_index: A numpy array containing the indices of the triangles on the side edge of the mesh.
        vertex_indices: A numpy array of shape (n, 3) containing the vertex indices for each triangle in the mesh.
        vertex_coordinates: A numpy array of shape (m, 3) containing the coordinates of each vertex in the mesh.

    Returns:
        A tuple containing the indices of the triangles on the left and right boundaries of the mesh.
    """
    left_boundary_triangles = []
    right_boundary_triangles = []

    # Define a function to check if an angle is between -pi/2 and pi/2
    @np.vectorize
    def check_angle(angle):
        return -np.pi/2.0 < angle <= np.pi/2.0

    # Define a function to find a horizontal vector perpendicular to the normal vector
    def find_horizontal_vector(normal_vector):
        # Find a vector perpendicular to the normal vector (in the plane)
        if normal_vector[0] == 0 and normal_vector[1] == 0:
            vertical_vector = np.array([1, 0, 0])
        else:
            vertical_vector = np.array([-normal_vector[1], normal_vector[0], 0])
        return np.arctan2(vertical_vector[1], vertical_vector[0])

    # Get the triangles and normal vectors for the side edge triangles
    triangles = vertex_coordinates[vertex_indices[sideedge_triangle_index]]
    normal_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])

    # Find the strike angle for each triangle
    tri_strikes = np.apply_along_axis(find_horizontal_vector, 1, normal_vectors)

    # Check if the strike angle is between -pi/2 and pi/2
    tri_strike_flags = np.apply_along_axis(check_angle, 0, tri_strikes)

    # Calculate the rotation angle for each triangle
    rotation_angles = np.where(tri_strike_flags, -tri_strikes, -tri_strikes + np.pi)

    # Rotate each triangle so that its strike is parallel to x-axis
    triangle_rotations = (triangles[:, :, 0] + triangles[:, :, 1]*1.j)*np.exp(1.j*rotation_angles[:, np.newaxis])

    # Calculate the x-coordinate of the centroid for each rotated triangle
    centroid_rotation_xs = np.sum(triangle_rotations.real, axis=1) / 3

    # Check if each triangle is on the left or right boundary
    is_left_boundary = np.sum(triangle_rotations.real < centroid_rotation_xs[:, np.newaxis], axis=1) >= 2
    is_right_boundary = ~is_left_boundary

    # Get the indices of the left and right boundary triangles
    left_boundary_triangles = sideedge_triangle_index[np.where(is_left_boundary)[0]].tolist()
    right_boundary_triangles = sideedge_triangle_index[np.where(is_right_boundary)[0]].tolist()

    return left_boundary_triangles, right_boundary_triangles


def find_four_boundary_triangles(vertex_indices: np.ndarray, vertex_coordinates: np.ndarray, top_tolerance: float, bottom_tolerance: float) -> dict:
    """
    Find the triangles on the top, bottom, left and right boundaries of a mesh.

    Args:
        vertex_indices: A numpy array of shape (n, 3) containing the vertex indices for each triangle in the mesh.
        vertex_coordinates: A numpy array of shape (m, 3) containing the coordinates of each vertex in the mesh.
        top_tolerance: A float specifying the tolerance for determining the top boundary of the mesh.
        bottom_tolerance: A float specifying the tolerance for determining the bottom boundary of the mesh.

    Returns:
        A tuple containing the indices of the triangles on the top, bottom, left and right boundaries of the mesh.
    """
    four_boundary_triangles = {'top': [], 'bottom': [], 'left': [], 'right': []}
    adjacent_maps = find_adjacent_triangles(vertex_indices)
    top_boundary_triangles, bottom_boundary_triangles, left_and_right_boundary_triangles = find_top_bottom_sides_triangles(adjacent_maps, vertex_indices, vertex_coordinates, top_tolerance, bottom_tolerance)
    left_boundary_triangles, right_boundary_triangles = find_left_and_right_boundary_triangles(np.array(left_and_right_boundary_triangles), vertex_indices, vertex_coordinates)
    four_boundary_triangles['top'] = top_boundary_triangles
    four_boundary_triangles['bottom'] = bottom_boundary_triangles
    four_boundary_triangles['left'] = left_boundary_triangles
    four_boundary_triangles['right'] = right_boundary_triangles
    return four_boundary_triangles


def find_boundary_and_corner_triangles(vertex_indices: np.ndarray, vertex_coordinates: np.ndarray, top_tolerance: float, bottom_tolerance: float, remove_corner=True) -> tuple:
    """
    Find the triangles on the top, bottom, left and right boundaries of a mesh, as well as the corner triangles.

    Args:
        vertex_indices: A numpy array of shape (n, 3) containing the vertex indices for each triangle in the mesh.
        vertex_coordinates: A numpy array of shape (m, 3) containing the coordinates of each vertex in the mesh.
        top_tolerance: A float specifying the tolerance for determining the top boundary of the mesh.
        bottom_tolerance: A float specifying the tolerance for determining the bottom boundary of the mesh.
        remove_corner: A boolean specifying whether to remove the corner triangles from the boundary triangles.

    Returns:
        A tuple containing two dictionaries: one for the indices of the triangles on the top, bottom, left and right boundaries of the mesh, and one for the indices of the corner triangles.
    """
    boundary_triangles = {'top': [], 'bottom': [], 'left': [], 'right': []}
    corner_triangles = {'top_left': None, 'top_right': None, 'bottom_left': None, 'bottom_right': None}

    # Find the adjacent triangles for each vertex
    adjacent_maps = find_adjacent_triangles(vertex_indices)

    # Find the triangles on the top, bottom, left and right boundaries of the mesh
    top_boundary_triangles, bottom_boundary_triangles, left_and_right_boundary_triangles = find_top_bottom_sides_triangles(adjacent_maps, vertex_indices, vertex_coordinates, top_tolerance, bottom_tolerance)
    # Find the triangles on the left and right boundaries of the mesh
    sideedge_triangle_index = np.where(np.array([len(adjacent_maps[i]) == 1 for i in range(len(adjacent_maps))]))[0].tolist()
    left_and_right_boundary_triangles = sideedge_triangle_index + left_and_right_boundary_triangles
    left_boundary_triangles, right_boundary_triangles = find_left_and_right_boundary_triangles(np.array(left_and_right_boundary_triangles), vertex_indices, vertex_coordinates)

    # Store the boundary triangles in the dictionary
    boundary_triangles['top'] = top_boundary_triangles
    boundary_triangles['bottom'] = bottom_boundary_triangles
    boundary_triangles['left'] = left_boundary_triangles
    boundary_triangles['right'] = right_boundary_triangles

    # Find the corner triangles
    for i in sideedge_triangle_index:
        if i in boundary_triangles['top']:
            if i in boundary_triangles['left']:
                corner_triangles['top_left'] = i
            elif i in boundary_triangles['right']:
                corner_triangles['top_right'] = i
        elif i in boundary_triangles['bottom']:
            if i in boundary_triangles['left']:
                corner_triangles['bottom_left'] = i
            elif i in boundary_triangles['right']:
                corner_triangles['bottom_right'] = i
    
    # Remove the corner triangles from the boundary triangles
    if remove_corner:
        for key in corner_triangles.keys():
            if corner_triangles[key] is not None:
                boundary_triangles[key.split('_')[0]].remove(corner_triangles[key])
                boundary_triangles[key.split('_')[1]].remove(corner_triangles[key])

    return boundary_triangles, corner_triangles


def find_left_or_right_edgeline_points(sideedge_triangle_index: np.ndarray, vertex_indices: np.ndarray, vertex_coordinates: np.ndarray, side='right') -> tuple:
    """
    Find the triangles on the left and right boundaries of a mesh.

    Args:
        sideedge_triangle_index: A numpy array containing the indices of the triangles on the side edge of the mesh.
        vertex_indices: A numpy array of shape (n, 3) containing the vertex indices for each triangle in the mesh.
        vertex_coordinates: A numpy array of shape (m, 3) containing the coordinates of each vertex in the mesh.

    Returns:
        A tuple containing the indices of the triangles on the left and right boundaries of the mesh.
    """

    # Define a function to check if an angle is between -pi/2 and pi/2
    @np.vectorize
    def check_angle(angle):
        return -np.pi/2.0 < angle <= np.pi/2.0

    # Define a function to find a horizontal vector perpendicular to the normal vector
    def find_horizontal_vector(normal_vector):
        # Find a vector perpendicular to the normal vector (in the plane)
        if normal_vector[0] == 0 and normal_vector[1] == 0:
            vertical_vector = np.array([1, 0, 0])
        else:
            vertical_vector = np.array([-normal_vector[1], normal_vector[0], 0])
        return np.arctan2(vertical_vector[1], vertical_vector[0])

    # Get the triangles and normal vectors for the side edge triangles
    triangles = vertex_coordinates[vertex_indices[sideedge_triangle_index]]
    normal_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])

    # Find the strike angle for each triangle
    tri_strikes = np.apply_along_axis(find_horizontal_vector, 1, normal_vectors)

    # Check if the strike angle is between -pi/2 and pi/2
    tri_strike_flags = np.apply_along_axis(check_angle, 0, tri_strikes)

    # Calculate the rotation angle for each triangle
    rotation_angles = np.where(tri_strike_flags, -tri_strikes, -tri_strikes + np.pi)

    # Rotate each triangle so that its strike is parallel to x-axis; (ntri, npoint)
    triangle_rotations = (triangles[:, :, 0] + triangles[:, :, 1]*1.j)*np.exp(1.j*rotation_angles[:, np.newaxis])

    # Calculate the x-coordinate of the centroid for each rotated triangle
    centroid_rotation_xs = np.sum(triangle_rotations.real, axis=1) / 3
    centroid_zs = np.sum(triangles[:, :, 2], axis=1) / 3

    # Check if each triangle is on the left or right boundary
    if side == 'left':
        is_boundary = triangle_rotations.real < centroid_rotation_xs[:, np.newaxis]
    elif side == 'right':
        is_boundary = triangle_rotations.real > centroid_rotation_xs[:, np.newaxis]

    # Get the indices of the left and right boundary points in the side edge
    edgeline_points_indexes = np.unique(vertex_indices[sideedge_triangle_index][is_boundary].flatten())
    pnt_sort_flag = np.argsort(vertex_coordinates[edgeline_points_indexes, 2])
    edgeline_points = vertex_coordinates[edgeline_points_indexes, :][pnt_sort_flag, :]
    edgeline_points_indexes = edgeline_points_indexes[pnt_sort_flag]

    # tri_strike_flags = tri_strike_flags[np.argsort(centroid_zs)]
    # print(tri_strikes)
    return edgeline_points_indexes, edgeline_points # , rotation_angles, tri_strike_flags


def plot_3d_mesh(vertices: np.ndarray, triangles: np.ndarray, boundary_triangles: list, selected_triangles: list = None) -> None:
    """
    Plot a 3D mesh with selected triangles highlighted.

    Args:
        vertices: A numpy array of shape (n, 3) containing the coordinates of each vertex in the mesh.
        triangles: A numpy array of shape (m, 3) containing the vertex indices for each triangle in the mesh.
        boundary_triangles: A list containing the indices of the triangles on the boundary of the mesh.
        selected_triangles: A list containing the indices of the triangles to be highlighted in the plot. Default is None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot all triangles in cyan
    triangle_verts = [vertices[triangle] for triangle in triangles]
    poly = Poly3DCollection(triangle_verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
    ax.add_collection3d(poly)

    # plot boundary triangles in yellow
    boundary_triangle_verts = [vertices[triangles[triangle]] for triangle in boundary_triangles]
    boundary_poly = Poly3DCollection(boundary_triangle_verts, facecolors='yellow', linewidths=1, edgecolors='b', alpha=.25)
    ax.add_collection3d(boundary_poly)

    # plot selected triangles in red
    if selected_triangles is not None:
        selected_triangle_verts = [vertices[triangles[triangle]] for triangle in selected_triangles]
        selected_poly = Poly3DCollection(selected_triangle_verts, facecolors='red', linewidths=1, edgecolors='k', alpha=.5)
        ax.add_collection3d(selected_poly)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set axis limits
    ax.set_xlim([vertices[:, 0].min(), vertices[:, 0].max()])
    ax.set_ylim([vertices[:, 1].min(), vertices[:, 1].max()])
    ax.set_zlim([vertices[:, -1].min(), vertices[:, -1].max()])

    # invert z-axis
    ax.invert_zaxis()

    plt.show()


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    from collections import OrderedDict
    import time
    # -----------------------------------Proj Information-------------------------------------#
    # center for local coordinates--M7.1 epicenter 
    lon0 = 98.25
    lat0 = 34.5

    import sys
    if len(sys.argv) > 1:
        depth = sys.argv[-1]
    else:
        depth = '60km'
    if depth == '30km':
        slipinterval = 3
    else:
        slipinterval = 4
    
    if depth == '60km':
        threshold = 1.5
    else:
        threshold = 1.0

    # CSI routines
    from csi import TriangularPatches
    from csi import SourceInv as csiSourceInv
    source = TriangularPatches('_'.join(('Maduo_main', depth)), lon0=lon0, lat0=lat0)
    slipfile = r'd:\Maduo_Postseismic\Postseismic_Inversion\output_vardep_xiong\output_{0}\slip_total_0_mudpy_{0}.gmt'.format(depth)
    source.readPatchesFromFile(slipfile)

    # 示例数据，三个三角形的顶点坐标
    vertex_coordinates = source.Vertices

    vertex_indices = source.Faces

    # 查找上下左右边界的三角形
    history = time.time()
    adjacent_maps = find_adjacent_triangles(vertex_indices)
    print(time.time() - history)
    history = time.time()
    top_boundary, bottom_boundary, left_right_boundary = find_top_bottom_sides_triangles(adjacent_maps, vertex_indices, vertex_coordinates, 
                                                                             top_tolerance=1e-6, bottom_tolerance=1e-6)
    # print(left_right_boundary)
    print(time.time() - history)
    history = time.time()
    left_boundary, right_boundary = find_left_and_right_boundary_triangles(np.array(left_right_boundary), 
                                                                           vertex_indices, vertex_coordinates)
    lapsed_time = time.time() - history
    print(lapsed_time)

    history = time.time()
    four_boundary_triangles, corner_triangles = find_boundary_and_corner_triangles(vertex_indices, vertex_coordinates, top_tolerance=1e-6, bottom_tolerance=1e-6, remove_corner=True)
    top_boundary, bottom_boundary, left_boundary, right_boundary = four_boundary_triangles['top'], four_boundary_triangles['bottom'], four_boundary_triangles['left'], four_boundary_triangles['right']
    print(time.time() - history)
    print("Top Boundary Triangles:", top_boundary)
    print("Bottom Boundary Triangles:", bottom_boundary)
    print("Left Boundary Triangles:", left_boundary)
    print("Right Boundary Triangles:", right_boundary)
    print('Corner Triangles:', corner_triangles['top_left'], corner_triangles['top_right'], corner_triangles['bottom_left'], corner_triangles['bottom_right'])

    # 可视化边界三角形
    plot_3d_mesh(vertex_coordinates, vertex_indices, top_boundary)
    plot_3d_mesh(vertex_coordinates, vertex_indices, top_boundary + bottom_boundary)
    plot_3d_mesh(vertex_coordinates, vertex_indices, top_boundary + bottom_boundary + left_boundary)
    plot_3d_mesh(vertex_coordinates, vertex_indices, top_boundary + bottom_boundary + right_boundary)