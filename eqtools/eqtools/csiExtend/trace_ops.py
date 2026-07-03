"""Pure helpers for 2-D fault-trace and polyline operations.

The functions in this module operate in a metric coordinate system, usually
CSI/eqtools projected ``x/y`` coordinates in kilometers.  Convert lon/lat to
``x/y`` before using them, then convert the result back if needed.
"""
from __future__ import annotations

from heapq import heappop, heappush

import numpy as np

__all__ = (
    "buffer_trace",
    "clean_trace",
    "cumulative_distance",
    "extend_trace",
    "orient_trace",
    "resample_trace",
    "reverse_trace",
    "simplify_trace",
    "smooth_trace",
    "trace_length",
    "trim_trace",
)


def _as_trace_array(coords, *, min_points=2) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("coords must be a 2-D array with at least x and y columns.")
    if arr.shape[0] < min_points:
        raise ValueError(f"coords must contain at least {min_points} points.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("coords contains NaN or infinite values.")
    return arr.copy()


def clean_trace(coords, *, atol=0.0) -> np.ndarray:
    """Remove consecutive duplicate vertices from a trace.

    Only consecutive duplicates are removed, so a line that legitimately returns
    to an earlier location keeps that later point.
    """
    arr = _as_trace_array(coords)
    if atol < 0:
        raise ValueError("atol must be non-negative.")

    step = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
    keep = np.r_[True, step > atol]
    out = arr[keep]
    if out.shape[0] < 2:
        raise ValueError("trace has fewer than two distinct vertices.")
    return out


def cumulative_distance(coords) -> np.ndarray:
    """Return cumulative along-trace distance in the x/y plane."""
    arr = clean_trace(coords)
    step = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
    return np.r_[0.0, np.cumsum(step)]


def trace_length(coords) -> float:
    """Return total trace length in the x/y plane."""
    return float(cumulative_distance(coords)[-1])


def _interp_at_distances(coords: np.ndarray, distances: np.ndarray) -> np.ndarray:
    s = cumulative_distance(coords)
    arr = clean_trace(coords)
    distances = np.asarray(distances, dtype=float)
    if np.any(distances < -1e-10) or np.any(distances > s[-1] + 1e-10):
        raise ValueError("interpolation distances must lie within the trace length.")
    distances = np.clip(distances, 0.0, s[-1])
    return np.column_stack([np.interp(distances, s, arr[:, col]) for col in range(arr.shape[1])])


def resample_trace(coords, *, every=None, num_points=None, keep_endpoints=True) -> np.ndarray:
    """Resample a trace by arc length.

    Provide exactly one of ``every`` or ``num_points``.  ``every`` gives the
    target spacing in the trace coordinate units.  ``num_points`` gives the
    number of returned vertices.
    """
    arr = clean_trace(coords)
    length = trace_length(arr)

    if (every is None) == (num_points is None):
        raise ValueError("Provide exactly one of every or num_points.")

    if every is not None:
        every = float(every)
        if every <= 0:
            raise ValueError("every must be positive.")
        distances = np.arange(0.0, length, every)
        if keep_endpoints and (distances.size == 0 or not np.isclose(distances[-1], length)):
            distances = np.r_[distances, length]
        elif not keep_endpoints:
            distances = distances[(distances > 0.0) & (distances < length)]
    else:
        num_points = int(num_points)
        min_count = 2 if keep_endpoints else 1
        if num_points < min_count:
            raise ValueError(f"num_points must be >= {min_count}.")
        if keep_endpoints:
            distances = np.linspace(0.0, length, num_points)
        else:
            distances = np.linspace(0.0, length, num_points + 2)[1:-1]

    return _interp_at_distances(arr, distances)


def _endpoint_vector(coords: np.ndarray, *, at_start: bool, tangent_window: int) -> np.ndarray:
    if tangent_window < 1:
        raise ValueError("tangent_window must be >= 1.")
    arr = clean_trace(coords)
    window = min(int(tangent_window), arr.shape[0] - 1)
    if at_start:
        vec = arr[window] - arr[0]
    else:
        vec = arr[-1] - arr[-1 - window]
    norm_xy = np.linalg.norm(vec[:2])
    if norm_xy <= 0:
        raise ValueError("Cannot determine endpoint tangent from zero-length segment.")
    return vec / norm_xy


def extend_trace(coords, *, start=0.0, end=0.0, mode="endpoint_tangent", tangent_window=1) -> np.ndarray:
    """Extend the first and/or last trace endpoint along local tangents.

    ``start`` extends backward from the first point and ``end`` extends forward
    from the last point.  The default ``endpoint_tangent`` mode uses the local
    end segments; increase ``tangent_window`` to use a longer endpoint chord for
    noisy traces.
    """
    if mode != "endpoint_tangent":
        raise ValueError("Only mode='endpoint_tangent' is currently supported.")
    start = float(start)
    end = float(end)
    if start < 0 or end < 0:
        raise ValueError("start and end extensions must be non-negative.")

    arr = clean_trace(coords)
    parts = []
    if start > 0:
        parts.append(arr[0] - _endpoint_vector(arr, at_start=True, tangent_window=tangent_window) * start)
    parts.extend(arr)
    if end > 0:
        parts.append(arr[-1] + _endpoint_vector(arr, at_start=False, tangent_window=tangent_window) * end)
    return np.vstack(parts)


def trim_trace(coords, *, start=None, end=None, mode="distance") -> np.ndarray:
    """Trim a trace between two along-trace distances.

    Distances are measured from the first vertex along cumulative arc length.
    The returned trace includes interpolated endpoints at ``start`` and ``end``.
    """
    if mode != "distance":
        raise ValueError("Only mode='distance' is currently supported.")
    arr = clean_trace(coords)
    s = cumulative_distance(arr)
    length = s[-1]
    start = 0.0 if start is None else float(start)
    end = length if end is None else float(end)
    if start < 0 or end > length or start >= end:
        raise ValueError("Expected 0 <= start < end <= trace length.")

    distances = [start]
    distances.extend(s[(s > start + 1e-10) & (s < end - 1e-10)])
    distances.append(end)
    return _interp_at_distances(arr, np.asarray(distances))


def reverse_trace(coords) -> np.ndarray:
    """Return a copy of the trace with vertex order reversed."""
    return clean_trace(coords)[::-1].copy()


def orient_trace(coords, *, start="west") -> np.ndarray:
    """Orient a trace so the first point is on a requested map side.

    Supported values are ``west/east/south/north`` and the aliases
    ``min_x/max_x/min_y/max_y``.
    """
    arr = clean_trace(coords)
    key = str(start).lower()
    aliases = {
        "west": (0, "min"),
        "min_x": (0, "min"),
        "east": (0, "max"),
        "max_x": (0, "max"),
        "south": (1, "min"),
        "min_y": (1, "min"),
        "north": (1, "max"),
        "max_y": (1, "max"),
    }
    if key not in aliases:
        raise ValueError("start must be west/east/south/north or min_x/max_x/min_y/max_y.")
    axis, rule = aliases[key]
    first, last = arr[0, axis], arr[-1, axis]
    should_reverse = (rule == "min" and first > last) or (rule == "max" and first < last)
    return arr[::-1].copy() if should_reverse else arr


def _rdp_indices(xy: np.ndarray, first: int, last: int, epsilon: float, keep: np.ndarray) -> None:
    if last <= first + 1:
        return
    start = xy[first]
    end = xy[last]
    line = end - start
    denom = np.linalg.norm(line)
    if denom == 0:
        distances = np.linalg.norm(xy[first + 1:last] - start, axis=1)
    else:
        vec = xy[first + 1:last] - start
        distances = np.abs(line[0] * vec[:, 1] - line[1] * vec[:, 0]) / denom
    rel = int(np.argmax(distances))
    if distances[rel] > epsilon:
        idx = first + 1 + rel
        keep[idx] = True
        _rdp_indices(xy, first, idx, epsilon, keep)
        _rdp_indices(xy, idx, last, epsilon, keep)


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _vw_simplify(coords: np.ndarray, area_threshold: float) -> np.ndarray:
    n = coords.shape[0]
    if n <= 2:
        return coords

    xy = coords[:, :2]
    prev_idx = np.arange(n) - 1
    next_idx = np.arange(n) + 1
    prev_idx[0] = -1
    next_idx[-1] = -1
    removed = np.zeros(n, dtype=bool)
    heap = []

    def current_area(i: int) -> float:
        return _triangle_area(xy[prev_idx[i]], xy[i], xy[next_idx[i]])

    for i in range(1, n - 1):
        heappush(heap, (current_area(i), i))

    while heap:
        area, i = heappop(heap)
        if removed[i] or prev_idx[i] < 0 or next_idx[i] < 0:
            continue
        actual = current_area(i)
        if not np.isclose(area, actual, rtol=0.0, atol=1e-14):
            heappush(heap, (actual, i))
            continue
        if actual > area_threshold:
            break

        # Remove the least informative interior vertex, then refresh the two
        # neighboring effective areas in the linked list.
        removed[i] = True
        p = prev_idx[i]
        q = next_idx[i]
        next_idx[p] = q
        prev_idx[q] = p
        for j in (p, q):
            if 0 < j < n - 1 and not removed[j]:
                heappush(heap, (current_area(j), j))

    return coords[~removed]


def simplify_trace(coords, *, method="rdp", tolerance=1.0) -> np.ndarray:
    """Simplify a trace using RDP distance or VW effective-area filtering.

    ``method='rdp'`` uses the Ramer-Douglas-Peucker perpendicular-distance
    threshold.  ``method='vw'`` uses Visvalingam-Whyatt effective triangle area.
    Both methods preserve the first and last vertex.
    """
    arr = clean_trace(coords)
    tolerance = float(tolerance)
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    method = str(method).lower()

    if method == "rdp":
        keep = np.zeros(arr.shape[0], dtype=bool)
        keep[[0, -1]] = True
        _rdp_indices(arr[:, :2], 0, arr.shape[0] - 1, tolerance, keep)
        return arr[keep]
    if method in ("vw", "visvalingam", "visvalingam_whyatt"):
        return _vw_simplify(arr, tolerance)
    raise ValueError("method must be 'rdp' or 'vw'.")


def smooth_trace(
    coords,
    *,
    method="bspline",
    smoothing=1.0,
    num_points=None,
    window=5,
    polyorder=2,
    preserve_endpoints=True,
) -> np.ndarray:
    """Smooth trace x/y coordinates.

    ``bspline`` uses SciPy's parametric B-spline fit.  ``savgol`` applies a
    Savitzky-Golay filter and is useful when you want modest denoising while
    keeping the original sampling pattern.
    """
    arr = clean_trace(coords)
    n_out = arr.shape[0] if num_points is None else int(num_points)
    if n_out < 2:
        raise ValueError("num_points must be >= 2.")

    method = str(method).lower()
    base = arr if n_out == arr.shape[0] else resample_trace(arr, num_points=n_out)
    result = base.copy()

    if method == "bspline":
        from scipy.interpolate import splev, splprep

        degree = min(3, arr.shape[0] - 1)
        if degree < 1:
            return result
        tck, _ = splprep([arr[:, 0], arr[:, 1]], s=float(smoothing), k=degree)
        u_new = np.linspace(0.0, 1.0, n_out)
        result[:, :2] = np.column_stack(splev(u_new, tck))
    elif method == "savgol":
        from scipy.signal import savgol_filter

        if window < 3:
            raise ValueError("window must be >= 3 for savgol smoothing.")
        if window % 2 == 0:
            window += 1
        window = min(window, n_out if n_out % 2 == 1 else n_out - 1)
        if window <= polyorder:
            raise ValueError("window must be greater than polyorder.")
        result[:, 0] = savgol_filter(result[:, 0], window, polyorder, mode="interp")
        result[:, 1] = savgol_filter(result[:, 1], window, polyorder, mode="interp")
    else:
        raise ValueError("method must be 'bspline' or 'savgol'.")

    if preserve_endpoints:
        result[0, :2] = arr[0, :2]
        result[-1, :2] = arr[-1, :2]
    return result


def buffer_trace(coords, distance, *, cap_style="round", join_style="round", return_geometry=False):
    """Buffer a trace and return polygon exterior coordinates.

    The return value is a list of ``(N, 2)`` arrays because buffering complex
    traces may produce multiple polygons.  Set ``return_geometry=True`` to get
    the Shapely geometry instead.
    """
    from shapely.geometry import LineString, Polygon

    arr = clean_trace(coords)
    distance = float(distance)
    if distance <= 0:
        raise ValueError("distance must be positive.")

    cap_map = {
        "round": 1,
        "flat": 2,
        "square": 3,
    }
    join_map = {
        "round": 1,
        "mitre": 2,
        "miter": 2,
        "bevel": 3,
    }
    if cap_style not in cap_map:
        raise ValueError("cap_style must be round, flat, or square.")
    if join_style not in join_map:
        raise ValueError("join_style must be round, mitre/miter, or bevel.")

    geom = LineString(arr[:, :2]).buffer(
        distance,
        cap_style=cap_map[cap_style],
        join_style=join_map[join_style],
    )
    if return_geometry:
        return geom
    if isinstance(geom, Polygon):
        return [np.asarray(geom.exterior.coords, dtype=float)]
    return [np.asarray(poly.exterior.coords, dtype=float) for poly in geom.geoms]
