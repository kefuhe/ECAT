"""Shared continuous-trace queries for observation preprocessing.

CSI owns observation mutation. These helpers only return distances or indices;
they never change fault traces, observation arrays, or mesh/GF state. All x/y
coordinates and thresholds use the same projected kilometre frame.
"""

import numpy as np
import shapely


_QUERY_CHUNK_SIZE = 100_000


def _observation_xy(data):
    """Validate representative-point coordinates before any data are removed."""
    x = np.asarray(data.x, dtype=float)
    y = np.asarray(data.y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("Observation x/y must be matching one-dimensional arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Observation x/y must be finite")
    return x, y


def _fault_trace_geometry(data, faults):
    """Build independent original traces, without connecting separate faults."""
    if not isinstance(faults, (list, tuple)):
        faults = [faults]
    if not faults:
        raise ValueError("At least one fault trace is required")

    traces = []
    for fault in faults:
        data_crs = getattr(data, "utm", None)
        fault_crs = getattr(fault, "utm", None)
        if data_crs is not None and fault_crs is not None and data_crs != fault_crs:
            raise ValueError("Observations and faults must use the same projected CRS")

        # Never switch implicitly to xi/yi, top_coords, or a mesh boundary:
        # the original surface trace defines this preprocessing selection.
        x = np.asarray(getattr(fault, "xf", None), dtype=float)
        y = np.asarray(getattr(fault, "yf", None), dtype=float)
        if (
            x.ndim != 1
            or y.shape != x.shape
            or len(x) < 2
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
        ):
            raise ValueError("Each fault needs at least two finite original x/y nodes")
        coords = np.column_stack((x, y))
        keep = np.r_[True, np.any(np.diff(coords, axis=0) != 0.0, axis=1)]
        coords = coords[keep]
        if len(coords) < 2:
            raise ValueError("Each fault trace needs at least two distinct nodes")
        traces.append(coords)

    return shapely.MultiLineString(traces)


def distances_to_faults(data, faults):
    """Return representative-point distances to continuous original traces.

    The result is a one-dimensional array in km. Querying points in bounded
    chunks avoids the old observations-by-trace-vertices distance matrix.
    """
    x, y = _observation_xy(data)
    traces = _fault_trace_geometry(data, faults)
    distances = np.empty(len(x), dtype=float)
    for start in range(0, len(x), _QUERY_CHUNK_SIZE):
        stop = min(start + _QUERY_CHUNK_SIZE, len(x))
        points = shapely.points(x[start:stop], y[start:stop])
        distances[start:stop] = shapely.distance(points, traces)
    return distances


def fault_rejection_indices(data, dis, faults):
    """Return ascending pre-deletion indices selected by a trace buffer.

    Positive dis removes representative points on/inside the buffer; negative
    dis removes points on/outside it. Zero retains the historical remove-all
    behavior. Round caps/joins approximate the continuous distance threshold:
    16 segments per quadrant give about 0.6 m arc error at a 500 m radius.

    Build and prepare one geometry per call, then query x/y arrays in chunks.
    Keep polygon holes and multipart topology intact, and retain no cache that
    could become stale after a caller changes the trace.
    """
    threshold = float(dis)
    if not np.isfinite(threshold):
        raise ValueError("Fault distance threshold must be finite")
    x, y = _observation_xy(data)
    traces = _fault_trace_geometry(data, faults)
    if threshold == 0.0:
        return np.arange(len(x), dtype=np.intp)
    if len(x) == 0:
        return np.empty(0, dtype=np.intp)

    buffer = traces.buffer(
        abs(threshold), quad_segs=16, cap_style=1, join_style=1
    )
    if buffer.is_empty or not buffer.is_valid:
        raise ValueError("Fault traces did not produce a valid non-empty buffer")
    # prepare() modifies the geometry in place and returns None.
    shapely.prepare(buffer)

    reject = np.empty(len(x), dtype=bool)
    for start in range(0, len(x), _QUERY_CHUNK_SIZE):
        stop = min(start + _QUERY_CHUNK_SIZE, len(x))
        if threshold > 0.0:
            reject[start:stop] = shapely.intersects_xy(
                buffer, x[start:stop], y[start:stop]
            )
        else:
            # Exclude the strict interior so boundary points are also removed,
            # preserving the historical >= abs(dis) rule for negative values.
            reject[start:stop] = ~shapely.contains_xy(
                buffer, x[start:stop], y[start:stop]
            )
    return np.flatnonzero(reject)
