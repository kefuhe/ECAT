import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib.path as mpath
from scipy.spatial import KDTree
from tqdm import tqdm


#---------------------Downsampling 3D Disp Field----------------------#
#--------------------#
# This block is used for downsampling a 3D deformation field.
# It contains three main functions:
# 1. init_globals: Initializes global variables for parallel processing.
# 2. process_corner: Processes a single corner to calculate values of points within the corner using a specified function.
# 3. process_points_within_corners: Processes points within given corners using parallel processing.
#--------------------#

# Global variables for parallel processing
global_tree = None
global_points = None
global_data = None
global_stat_func = None

def init_globals(tree, points, data, stat_func):
    global global_tree, global_points, global_data, global_stat_func
    global_tree = tree
    global_points = points
    global_data = data
    global_stat_func = stat_func

def process_corner(args):
    i, corner = args
    if len(corner) == 6:
        # Create triangle vertices for current corner
        verts = np.array([
            [corner[0], corner[1]],  # first vertex
            [corner[2], corner[3]],  # second vertex
            [corner[4], corner[5]],  # third vertex
            [corner[0], corner[1]]   # close path
        ])
        # Calculate the radius of the circumcircle of the triangle
        a = np.linalg.norm(verts[1] - verts[0])
        b = np.linalg.norm(verts[2] - verts[1])
        c = np.linalg.norm(verts[2] - verts[0])
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        radius = (a * b * c) / (4 * area)
    elif len(corner) == 4:
        # Create rectangle vertices for current corner
        verts = np.array([
            [corner[0], corner[1]],  # left top
            [corner[2], corner[1]],  # right top
            [corner[2], corner[3]],  # right bottom
            [corner[0], corner[3]],  # left bottom
            [corner[0], corner[1]]   # close path
        ])
        # Calculate the radius for querying points
        radius = max(corner[2] - corner[0], corner[1] - corner[3])
    else:
        raise ValueError("Corner must have 4 or 6 elements.")
    
    # Create path and find points inside
    path = mpath.Path(verts)
    indices = global_tree.query_ball_point(verts.mean(axis=0), r=radius)
    mask = path.contains_points(global_points[indices])
    
    # Calculate values for points inside the corner using the specified function
    values = np.array([global_stat_func(global_data[indices][mask, k]) if np.any(~np.isnan(global_data[indices][mask, k])) else np.nan for k in range(global_data.shape[1])])
    return i, values

def process_points_within_corners(points, corners, data, stat_func=np.nanmean, n_jobs=None):
    """
    Process points within given corners using a specified function.

    Parameters:
    -----------
    points : np.ndarray
        Array of points with shape (n_points, 2).
    corners : np.ndarray
        Array of corners with shape (n_corners, 4/6).
    data : np.ndarray
        Array of data associated with points with shape (n_points, n_features).
    stat_func : function, optional (e.g., np.nanstd, np.nanmean, etc.)
        Function to apply to the data within each corner (default is np.nanmean).
    n_jobs : int, optional
        Number of parallel jobs. If None, uses all available cores.

    Returns:
    --------
    results : np.ndarray
        Array of processed values with shape (n_corners, n_features).
    """
    # Build KDTree for fast spatial queries
    tree = KDTree(points)

    # Initialize global variables in each worker
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=init_globals, initargs=(tree, points, data, stat_func)) as executor:
        results = list(tqdm(executor.map(process_corner, enumerate(corners)), total=len(corners)))

    # Collect results
    processed_data = np.zeros((len(corners), data.shape[1]))
    for i, values in results:
        processed_data[i] = values

    return processed_data
#---------------------------------------------------------------#