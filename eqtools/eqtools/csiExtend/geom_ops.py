from shapely.geometry import LineString, Point
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class InvalidFaultGeometryError(ValueError):
    """Raised when paired top/bottom edges form an invalid local cell."""


def validate_top_bottom_cells(
        top_coords,
        bottom_coords,
        *,
        raise_on_invalid=True):
    """Validate adjacent quadrilaterals formed by paired fault-edge nodes.

    Each cell [top[i], top[i+1], bottom[i+1], bottom[i]] is split along
    the same diagonal used by the simple triangular mesh. A cell is invalid
    when either triangle is degenerate or the two triangle normals point into
    opposite hemispheres. The calculation is fully three-dimensional, so a
    vertical fault surface is not mistaken for a zero-area XY polygon.

    Parameters
    ----------
    top_coords, bottom_coords : (n, 3+) array-like
        Paired boundary coordinates. Row i on the bottom must be derived
        from row i on the top.
    raise_on_invalid : bool, default True
        Raise InvalidFaultGeometryError when an invalid cell is found.
        Set to False for lightweight diagnostics.

    Returns
    -------
    ndarray of bool
        One value per adjacent cell; True means the local cell is valid.

    Notes
    -----
    This is an O(n) local check. It does not search for intersections between
    non-adjacent cells and it never reorders or repairs the input coordinates.
    One paired duplicate segment is treated as a redundant trace row and
    reported as valid without changing either array. A wholly collapsed strip
    remains invalid.
    """
    top = np.asarray(top_coords, dtype=float)
    bottom = np.asarray(bottom_coords, dtype=float)
    if (
        top.ndim != 2
        or bottom.ndim != 2
        or top.shape[1] < 3
        or bottom.shape[1] < 3
    ):
        raise ValueError("top_coords and bottom_coords must have shape (n, 3+)")
    if top.shape[0] != bottom.shape[0]:
        raise ValueError(
            "top_coords and bottom_coords must contain the same number of "
            "paired nodes"
        )
    if top.shape[0] < 2:
        raise ValueError("top/bottom geometry requires at least two paired nodes")

    top = top[:, :3]
    bottom = bottom[:, :3]
    if not np.all(np.isfinite(top)) or not np.all(np.isfinite(bottom)):
        raise ValueError("top_coords and bottom_coords must contain finite values")

    top_left = top[:-1]
    top_right = top[1:]
    bottom_left = bottom[:-1]
    bottom_right = bottom[1:]

    normal_1 = np.cross(
        top_right - top_left,
        bottom_right - top_left,
    )
    normal_2 = np.cross(
        bottom_right - top_left,
        bottom_left - top_left,
    )
    norm_1 = np.linalg.norm(normal_1, axis=1)
    norm_2 = np.linalg.norm(normal_2, axis=1)

    # A scale-relative area tolerance avoids depending on whether the caller
    # uses metres or kilometres while only classifying numerical degeneracy.
    edge_lengths = np.column_stack([
        np.linalg.norm(top_right - top_left, axis=1),
        np.linalg.norm(bottom_right - bottom_left, axis=1),
        np.linalg.norm(bottom_left - top_left, axis=1),
        np.linalg.norm(bottom_right - top_right, axis=1),
        np.linalg.norm(bottom_right - top_left, axis=1),
    ])
    cell_scale = np.max(edge_lengths, axis=1)
    area_tolerance = (
        64.0 * np.finfo(float).eps * np.square(cell_scale)
    )
    raw_degenerate = (norm_1 <= area_tolerance) | (norm_2 <= area_tolerance)

    # Real-world traces occasionally contain an adjacent duplicate node. When
    # both paired edges collapse at the same cell, that row pair is redundant
    # rather than a folded surface; later meshing may safely ignore it. Keep
    # the original arrays untouched and skip only that local diagnostic. An
    # entirely collapsed strip remains invalid.
    length_tolerance = np.sqrt(area_tolerance)
    redundant_pair = (
        (edge_lengths[:, 0] <= length_tolerance)
        & (edge_lengths[:, 1] <= length_tolerance)
    )
    if np.all(redundant_pair):
        redundant_pair[:] = False
    degenerate = raw_degenerate & ~redundant_pair
    opposite_normals = (
        np.einsum('ij,ij->i', normal_1, normal_2) <= 0.0
    ) & ~raw_degenerate
    valid = ~(degenerate | opposite_normals)

    if raise_on_invalid and not np.all(valid):
        invalid = np.flatnonzero(~valid)
        degenerate_indices = np.flatnonzero(degenerate)
        folded_indices = np.flatnonzero(opposite_normals)
        raise InvalidFaultGeometryError(
            "invalid adjacent top/bottom fault cell(s): "
            f"indices={invalid[:10].tolist()}, "
            f"degenerate={degenerate_indices[:10].tolist()}, "
            f"opposite_normals={folded_indices[:10].tolist()}. "
            "Preserve top[i] <-> bottom[i] correspondence and revise the "
            "trace/strike/dip controls; automatic reordering is not applied."
        )

    return valid


class PolygonIntersector:
    def __init__(self, top_coords, bottom_coords, depth=18.0, extension_length=1000):
        """
        初始化多边形交点计算器。
        
        :param top_coords: 顶边坐标列表。
        :param bottom_coords: 底边坐标列表。
        :param depth: 深度值，用于计算交点的Z坐标。
        :param extension_length: 线段外延长度。
        """
        self.top_coords = top_coords.tolist() if isinstance(top_coords, np.ndarray) else top_coords
        self.bottom_coords = bottom_coords.tolist() if isinstance(bottom_coords, np.ndarray) else bottom_coords
        self.depth = depth
        self.extension_length = extension_length

    def calculate_normal_vector(self, coords, index):
        """
        计算给定坐标点的法向量。
        """
        if index == 0:
            dir_vector = np.array(coords[1]) - np.array(coords[0])
        elif index == len(coords) - 1:
            dir_vector = np.array(coords[-1]) - np.array(coords[-2])
        else:
            dir_vector = np.array(coords[index + 1]) - np.array(coords[index])
        normal_vector = np.array([-dir_vector[1], dir_vector[0], 0])
        return normal_vector

    def find_intersection(self, top_point, index):
        """
        查找给定顶点和其法线与底边折线的交点。
        """
        top_normal_vector = self.calculate_normal_vector(self.top_coords, index)
        P0 = np.array(top_point[:2])
        dir = top_normal_vector[:2] / np.linalg.norm(top_normal_vector[:2])
        line_top = LineString([P0 + dir * -self.extension_length, P0, P0 + dir * self.extension_length])
        
        extended_bottom_coords = self._extend_bottom_line()
        line_bottom = LineString(extended_bottom_coords)
        
        intersection = line_top.intersection(line_bottom)
        return self._process_intersection(intersection, P0)

    def _extend_bottom_line(self):
        """
        拓展底边折线两端。
        """
        bottom_start = np.array(self.bottom_coords[0])
        bottom_end = np.array(self.bottom_coords[-1])
        bottom_dir_start = np.array(self.bottom_coords[1]) - bottom_start
        bottom_dir_end = np.array(self.bottom_coords[-2]) - bottom_end
        bottom_dir_start /= np.linalg.norm(bottom_dir_start)
        bottom_dir_end /= np.linalg.norm(bottom_dir_end)
        
        extended_start = bottom_start + bottom_dir_start * -self.extension_length
        extended_end = bottom_end + bottom_dir_end * -self.extension_length
        return [extended_start] + self.bottom_coords + [extended_end]

    def _process_intersection(self, intersection, P0):
        """
        处理交点结果。
        """
        P0_point = Point(P0)
        if not intersection.is_empty:
            if intersection.geom_type == 'MultiPoint':
                closest_point = min([p for p in intersection.geoms], key=lambda p: P0_point.distance(p))
                return np.array([closest_point.x, closest_point.y, -self.depth])
            else:
                return np.array([intersection.x, intersection.y, -self.depth])
        return None

    def calculate_intersections(self, mode='both', indices=None):
        """
        计算所有顶点与底边折线的交点。
        参数:
        - mode: 'all', 'first', 'last', 或 'specific'，默认为 'all'。
        - indices: 当 mode 为 'specific' 时，指定需要计算交点的顶点索引列表。
        """
        intersections = []
        if mode == 'all':
            indices_to_check = range(len(self.top_coords))
        elif mode == 'first':
            indices_to_check = [0]
        elif mode == 'last':
            indices_to_check = [len(self.top_coords) - 1]
        elif mode == 'both':
            indices_to_check = [0, len(self.top_coords) - 1]
        elif mode == 'specific' and indices is not None:
            indices_to_check = indices
        else:
            raise ValueError("Invalid mode or indices")
    
        for index in indices_to_check:
            top_point = self.top_coords[index]
            intersection = self.find_intersection(top_point, index)
            if intersection is not None:
                intersections.append(intersection)
        return intersections

    def plot(self, style=['notebook'], plot_on_2d=False):
        """
        绘制顶边、底边和交点。
        """
        from ..viztools import sci_plot_style
        with sci_plot_style(style=style):
            fig = plt.figure()
            if plot_on_2d:
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111, projection='3d')
            
            top_x, top_y, top_z = zip(*self.top_coords)
            bottom_x, bottom_y, bottom_z = zip(*self.bottom_coords)
            intersections = self.calculate_intersections()

            # 绘制顶边和底边
            if plot_on_2d:
                ax.plot(top_x, top_y, color='r', marker='o', label='Top Edge')
                ax.plot(bottom_x, bottom_y, color='b', marker='o', label='Bottom Edge')
            else:
                ax.plot(top_x, top_y, top_z, color='r', marker='o', label='Top Edge')
                ax.plot(bottom_x, bottom_y, bottom_z, color='b', marker='o', label='Bottom Edge')
            
            # 收集交点坐标
            intersections_x = [point[0] for point in intersections]
            intersections_y = [point[1] for point in intersections]
            if not plot_on_2d:
                intersections_z = [point[2] for point in intersections]
            
            # 使用plot命令绘制交点
            if plot_on_2d:
                ax.plot(intersections_x, intersections_y, 'g^', label='Intersection', zorder=3)
            else:
                ax.plot(intersections_x, intersections_y, intersections_z, 'g^', label='Intersection', zorder=3)
            
            ax.legend()
            plt.show()


def polyline_arclength(coords):
    """Return planar cumulative arc length for an ordered 3-D polyline.

    Fault-edge distance is measured in the projected ``x/y`` plane.  The
    input rows are *nodes*, so ``diff(coords)`` already represents the finite
    segments that must be summed; applying a quadrature rule to those segment
    lengths would shorten and shift the coordinate system.

    Parameters
    ----------
    coords : array-like, shape (n, 3+)
        Ordered fault-edge coordinates.

    Returns
    -------
    cumulative : ndarray, shape (n,)
        Distance from the first node, with ``cumulative[0] == 0``.
    segment_lengths : ndarray, shape (n - 1,)
        Planar length of each consecutive segment.
    """
    values = np.asarray(coords, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 3:
        raise ValueError("coords must have shape (n>=2, 3+)")
    if not np.all(np.isfinite(values[:, :3])):
        raise ValueError("coords must contain only finite values")

    segment_lengths = np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1)
    scale = max(1.0, float(np.max(np.abs(values[:, :2]))))
    tolerance = 64.0 * np.finfo(float).eps * scale
    duplicate = np.flatnonzero(segment_lengths <= tolerance)
    if duplicate.size:
        raise ValueError(
            "coords contains zero-length planar segment(s): "
            f"{duplicate.tolist()}"
        )
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return cumulative, segment_lengths


def interpolate_polyline_at_arclength(coords, stations):
    """Interpolate an ordered polyline at planar arc-length stations."""
    values = np.asarray(coords, dtype=float)
    cumulative, _ = polyline_arclength(values)
    requested = np.asarray(stations, dtype=float)
    if requested.ndim != 1 or not np.all(np.isfinite(requested)):
        raise ValueError("stations must be a finite one-dimensional array")
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, cumulative[-1])
    if (
        np.any(requested < -tolerance)
        or np.any(requested > cumulative[-1] + tolerance)
    ):
        raise ValueError("stations must lie within the polyline arc-length range")
    requested = np.clip(requested, 0.0, cumulative[-1])
    return np.column_stack([
        np.interp(requested, cumulative, values[:, column])
        for column in range(values.shape[1])
    ])


def _regular_arclength_stations(total_length, *, every=None, num_segments=None):
    """Resolve the established point-count contract to uniform stations."""
    if every is not None:
        if (
            isinstance(every, (bool, np.bool_))
            or not np.isfinite(every)
            or every <= 0.0
        ):
            raise ValueError("every must be a finite positive distance")
        # Preserve the established nearest-count convention while ensuring a
        # valid two-endpoint polyline for lengths shorter than one interval.
        num_points = int(np.floor(total_length / every))
        remainder = total_length - num_points * every
        if remainder >= every / 2.0:
            num_points += 1
        num_points = max(2, num_points)
    elif num_segments is not None:
        if isinstance(num_segments, (bool, np.bool_)):
            raise TypeError("num_segments must be an integer >= 2")
        num_points = int(num_segments)
        if num_points != num_segments or num_points < 2:
            raise ValueError("num_segments must be an integer >= 2")
    else:
        raise ValueError("Either 'every' or 'num_segments' must be set")
    return np.linspace(0.0, total_length, num_points)


def densify_polyline_preserving_vertices(
        coords, *, every=None, num_segments=None, required_stations=None):
    """Densify a polyline while retaining its geometric breakpoints.

    This helper is for the dip-physics working boundary.  Original trace
    vertices remain hard geometric nodes.  Optional stations (for example dip
    controls or transition endpoints) are evaluation events: they are inserted
    on the existing piecewise-linear trace but do not redefine that trace.

    ``num_segments`` retains the package's historical meaning of a target
    *point count*.  If protected nodes already exceed that target, all protected
    nodes are kept.  Interval mode subdivides each protected interval using the
    nearest sensible count; it avoids creating an extra node when an interval
    is shorter than 1.5 times the requested spacing.
    """
    values = np.asarray(coords, dtype=float)
    cumulative, _ = polyline_arclength(values)
    total_length = float(cumulative[-1])

    protected = list(cumulative)
    if required_stations is not None:
        required = np.asarray(required_stations, dtype=float).reshape(-1)
        if not np.all(np.isfinite(required)):
            raise ValueError("required_stations must contain only finite values")
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, total_length)
        if (
            np.any(required < -tolerance)
            or np.any(required > total_length + tolerance)
        ):
            raise ValueError("required_stations must lie on the input polyline")
        protected.extend(np.clip(required, 0.0, total_length))

    protected = np.sort(np.asarray(protected, dtype=float))
    # Profile positions may make a lon/lat -> projected-coordinate round trip
    # before they are projected back onto the same top edge.  A relative 1e-9
    # station tolerance (sub-millimetre on a hundreds-of-kilometres trace)
    # prevents that harmless frame roundoff from creating a duplicate node.
    merge_tolerance = max(
        64.0 * np.finfo(float).eps * max(1.0, total_length),
        1e-9 * max(1.0, total_length),
    )
    protected = protected[np.r_[True, np.diff(protected) > merge_tolerance]]

    if every is not None:
        if (
            isinstance(every, (bool, np.bool_))
            or not np.isfinite(every)
            or every <= 0.0
        ):
            raise ValueError("every must be a finite positive distance")
        stations = [protected[0]]
        for start, stop in zip(protected[:-1], protected[1:]):
            span = stop - start
            subdivisions = max(1, int(np.floor(span / every + 0.5)))
            stations.extend(np.linspace(start, stop, subdivisions + 1)[1:])
        stations = np.asarray(stations)
    elif num_segments is not None:
        if isinstance(num_segments, (bool, np.bool_)):
            raise TypeError("num_segments must be an integer >= 2")
        target = int(num_segments)
        if target != num_segments or target < 2:
            raise ValueError("num_segments must be an integer >= 2")
        if len(protected) >= target:
            stations = protected
        else:
            remaining = target - len(protected)
            spans = np.diff(protected)
            ideal = remaining * spans / np.sum(spans)
            extras = np.floor(ideal).astype(int)
            remainder = remaining - int(np.sum(extras))
            if remainder:
                order = np.argsort(-(ideal - extras), kind="stable")
                extras[order[:remainder]] += 1
            station_parts = [[protected[0]]]
            for start, stop, extra in zip(protected[:-1], protected[1:], extras):
                station_parts.append(
                    np.linspace(start, stop, int(extra) + 2)[1:].tolist()
                )
            stations = np.asarray([
                station for part in station_parts for station in part
            ])
    else:
        raise ValueError("Either 'every' or 'num_segments' must be set")

    return interpolate_polyline_at_arclength(values, stations)


def discretize_coords(coords, every=None, num_segments=None, threshold=2):
    '''
    Resample node coordinates uniformly along a piecewise-linear curve.

    Parameters:
    - coords (np.ndarray): The coordinates of the nodes, shape (N, 3).
    - every (float, optional): Finite positive distance used to discretize the
      coordinates. If provided, overrides num_segments.
    - num_segments (int, optional): Historical name for the target point count.
      Ignored if every is provided.
    - threshold (float, optional): Retained for API compatibility. Endpoints are
      now always included exactly, so this value is not used.

    Returns:
    - xyz_new (np.ndarray): The new discretized coordinates, shape (M, 3).
    '''
    del threshold  # Endpoints are always included by construction.
    r, _ = polyline_arclength(coords)
    r_new = _regular_arclength_stations(
        r[-1], every=every, num_segments=num_segments,
    )

    return interpolate_polyline_at_arclength(coords, r_new)


def resample_boundaries_by_normalized_arclength(
        top_coords, bottom_coords, *, num_segments):
    """Return the established normalized top/bottom mapping boundaries.

    Both edges receive the same point count, but each is sampled on its own
    true planar arc length.  Consequently column ``j`` means the shared
    normalized coordinate ``j / (num_segments - 1)``; it does *not* assert a
    material-point or equal-physical-distance correspondence between edges.
    This explicit helper protects the current fixed-topology deformation
    contract from accidental replacement by a different mapping model.
    """
    top = discretize_coords(top_coords, num_segments=num_segments)
    bottom = discretize_coords(bottom_coords, num_segments=num_segments)
    return top, bottom

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
    from sklearn.decomposition import PCA
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


if __name__ == '__main__':
    # 示例使用
    top_coords = np.array([(0, 0, 0), (1, 1, 0), (2, 2, 0)])  # 三个或更多节点
    bottom_coords = np.array([(0, -1, -18), (1.5, 1, -18), (2.3, 1, -18), ])
    patch = PolygonIntersector(top_coords, bottom_coords)
    patch.plot()
