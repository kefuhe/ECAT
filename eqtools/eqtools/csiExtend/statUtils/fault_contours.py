"""Fault-surface contour extraction utilities.

This module separates two common operations that are easy to confuse:

* isodepth contours are geometric curves where a fault surface intersects
  the horizontal plane ``depth = constant``;
* scalar contours are isolines of a field such as slip or coupling on a
  surface and are handled by :mod:`contour3D_extraction`.

The public helper here implements the geometric isodepth case.  The default
backend converts the fault object into a triangle mesh and intersects every
triangle with the requested depth plane.  It therefore avoids any x-y, x-z,
or y-z projection choice at the user level.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


MESH_PLANE_METHODS = {"mesh_plane_intersection", "plane_intersection", "geometric"}


def extract_isodepth_contours(
    fault_obj,
    depths,
    *,
    engine=None,
    method="mesh_plane_intersection",
    rect_subdivision=1,
    merge_tol=0.0,
    stitch_tol=None,
    intersection_eps=None,
    min_len=2,
    largest_only=True,
    sort_by=None,
    reverse=False,
    return_diagnostics=False,
):
    """Extract depth contours from a CSI/eqtools fault object.

    Parameters
    ----------
    fault_obj : object
        Fault object to query.  Triangular faults are read from ``Vertices``
        and ``Faces``.  Rectangular faults are read from ``patch`` and are
        converted to a triangle soup internally.
    depths : float or sequence of float
        Positive target depths in kilometers.  Negative values are accepted
        and converted with ``abs`` for compatibility with CSI z conventions.
    engine : object, optional
        Optional object with an ``xy2ll(x, y)`` method.  When supplied, output
        lines have five columns: ``x, y, depth, lon, lat``.  Without a
        converter, output lines have three columns: ``x, y, depth``.
    method : str, default ``"mesh_plane_intersection"``
        Geometric backend name.  Accepted aliases are
        ``"mesh_plane_intersection"``, ``"plane_intersection"`` and
        ``"geometric"``.  Projection-based scalar contouring is intentionally
        not implemented here.
    rect_subdivision : int, default 1
        Rectangular-patch preprocessing level.  ``1`` means each quadrilateral
        is represented by two triangles.  Values greater than ``1`` create a
        bilinear grid with ``rect_subdivision`` points per side before
        triangulation.  For non-planar quadrilaterals, strictness applies to
        the resulting piecewise-linear mesh, not to an analytic bilinear
        surface unless it has been sufficiently subdivided.
    merge_tol : float, default 0.0
        Optional vertex welding tolerance in kilometers when rectangular
        patches are converted to triangles.  This is mainly useful for meshes
        whose adjacent patch corners differ by small numerical noise.
    stitch_tol : float, optional
        Endpoint tolerance used to stitch triangle-plane segments into ordered
        polylines.  If omitted, a scale-aware tolerance is derived from the
        mesh extent.
    intersection_eps : float, optional
        Tolerance for deciding whether a vertex lies on the target depth.
        If omitted, a scale-aware value is used.
    min_len : int, default 2
        Minimum number of points required for a returned polyline.
    largest_only : bool, default True
        If true, return the longest polyline for each depth to match the
        historical ``FaultGeometryEngine.extract_contours_from_fault`` return
        shape.  If false, return all stitched polylines for each depth.
    sort_by : {None, "x", "y", "lon", "lat"}, optional
        Orient each polyline by comparing its two endpoints along the selected
        coordinate.  This preserves curve connectivity.  It does not sort all
        points independently.
    reverse : bool, default False
        Reverse the selected endpoint orientation, or simply reverse each line
        when ``sort_by`` is ``None``.
    return_diagnostics : bool, default False
        If true, return ``(contours, diagnostics)``.

    Returns
    -------
    dict or tuple
        With ``largest_only=True``, returns ``{depth: ndarray}``.  Each array
        has columns ``x, y, depth`` or ``x, y, depth, lon, lat`` depending on
        whether a converter is available.  With ``largest_only=False``, returns
        ``{depth: [ndarray, ...]}``.  When ``return_diagnostics=True``, a
        second dictionary reports raw segment counts, stitched line counts,
        tolerances and residuals.

    Notes
    -----
    For triangular patches this is exact for the discrete planar triangle
    surface.  For rectangular patches it is exact for the triangle mesh built
    from the rectangles; use ``rect_subdivision`` when a non-planar bilinear
    quadrilateral representation is desired.
    """
    method = _normalize_method(method)
    if method not in MESH_PLANE_METHODS:
        raise ValueError(
            "extract_isodepth_contours only implements the geometric "
            "mesh-plane backend. Use FaultGeometryEngine with "
            "method='legacy_map_contour' for the historical tricontour path."
        )

    target_depths = _normalize_depths(depths)
    vertices, faces, mesh_info = fault_to_triangle_mesh(
        fault_obj,
        rect_subdivision=rect_subdivision,
        merge_tol=merge_tol,
    )
    converter = engine if engine is not None else fault_obj
    if not hasattr(converter, "xy2ll"):
        converter = None

    scale = _mesh_scale(vertices)
    local_stitch_tol = stitch_tol
    if local_stitch_tol is None:
        local_stitch_tol = max(scale * 1.0e-10, 1.0e-7)
    local_intersection_eps = intersection_eps
    if local_intersection_eps is None:
        local_intersection_eps = max(scale * 1.0e-12, 1.0e-10)

    contours = {}
    diagnostics = {
        "method": method,
        "mesh": mesh_info,
        "rect_subdivision": rect_subdivision,
        "merge_tol": merge_tol,
        "stitch_tol": local_stitch_tol,
        "intersection_eps": local_intersection_eps,
        "depths": {},
    }

    for depth in target_depths:
        segments, segment_info = mesh_plane_segments(
            vertices,
            faces,
            depth,
            eps=local_intersection_eps,
            dedupe_tol=local_stitch_tol,
        )
        polylines, stitch_info = stitch_segments_to_polylines(
            segments,
            tol=local_stitch_tol,
        )
        min_points = max(int(min_len or 0), 2)
        polylines = [line for line in polylines if len(line) >= min_points]
        polylines.sort(key=_polyline_length, reverse=True)

        formatted = [
            _format_polyline(line, depth, converter=converter)
            for line in polylines
        ]
        formatted = [
            _orient_polyline(line, sort_by=sort_by, reverse=reverse)
            for line in formatted
        ]

        if largest_only:
            contours[depth] = formatted[0] if formatted else _empty_contour(converter)
        else:
            contours[depth] = formatted

        diagnostics["depths"][depth] = {
            "raw_segments": int(len(segments)),
            "stitched_polylines": int(stitch_info["polyline_count"]),
            "retained_polylines": int(len(formatted)),
            "branch_nodes": int(stitch_info["branch_nodes"]),
            "degenerate_triangles": int(segment_info["degenerate_triangles"]),
            "coplanar_edges": int(segment_info["coplanar_edges"]),
            "max_depth_residual": _max_depth_residual(formatted, depth),
            "largest_length": _largest_formatted_length(formatted),
        }

    if return_diagnostics:
        return contours, diagnostics
    return contours


def fault_to_triangle_mesh(fault_obj, *, rect_subdivision=1, merge_tol=0.0):
    """Convert a triangular or rectangular fault object into triangle mesh arrays.

    Parameters
    ----------
    fault_obj : object
        Object with either ``Vertices``/``Faces`` or rectangular ``patch`` data.
    rect_subdivision : int, default 1
        Rectangular preprocessing level.  ``1`` uses two triangles per patch;
        values greater than ``1`` use bilinear interpolation with that many
        points along each patch side before triangulation.
    merge_tol : float, default 0.0
        Optional tolerance for welding duplicated rectangle vertices.

    Returns
    -------
    vertices : ndarray, shape (n_vertices, 3)
        Mesh coordinates in kilometers.  The third column is positive depth.
    faces : ndarray, shape (n_faces, 3)
        Triangle vertex indices.
    info : dict
        Metadata describing the source mesh type and generated mesh size.
    """
    if hasattr(fault_obj, "Vertices") and hasattr(fault_obj, "Faces"):
        vertices = np.asarray(fault_obj.Vertices, dtype=float).copy()
        faces = np.asarray(fault_obj.Faces, dtype=int).copy()
        vertices[:, 2] = np.abs(vertices[:, 2])
        faces = _drop_degenerate_faces(faces)
        return vertices, faces, {
            "source": "triangular",
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
        }

    if not hasattr(fault_obj, "patch"):
        raise ValueError("fault_obj must provide either Vertices/Faces or patch data")

    patches = [np.asarray(patch, dtype=float) for patch in fault_obj.patch]
    if not patches:
        raise ValueError("fault_obj.patch is empty")

    vertices_list = []
    faces_list = []
    for patch in patches:
        patch_vertices, patch_faces = _patch_to_triangles(
            patch,
            rect_subdivision=rect_subdivision,
        )
        offset = len(vertices_list)
        vertices_list.extend(patch_vertices)
        faces_list.extend((patch_faces + offset).tolist())

    vertices = np.asarray(vertices_list, dtype=float)
    faces = np.asarray(faces_list, dtype=int)
    vertices[:, 2] = np.abs(vertices[:, 2])
    if merge_tol and merge_tol > 0:
        vertices, faces = weld_vertices(vertices, faces, tol=merge_tol)
    faces = _drop_degenerate_faces(faces)
    return vertices, faces, {
        "source": "rectangular_or_patch",
        "patches": int(len(patches)),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
    }


def mesh_plane_segments(vertices, faces, depth, *, eps=None, dedupe_tol=None):
    """Intersect triangle mesh elements with a constant-depth plane.

    The returned segments are exact for the input triangle mesh.  Each segment
    is represented by two 3-D endpoints whose third coordinate is the positive
    target depth.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    depth = abs(float(depth))
    if eps is None:
        eps = max(_mesh_scale(vertices) * 1.0e-12, 1.0e-10)
    if dedupe_tol is None:
        dedupe_tol = max(_mesh_scale(vertices) * 1.0e-10, 1.0e-7)

    segments = []
    degenerate_triangles = 0
    coplanar_edges = 0
    for face in faces:
        tri = vertices[face].astype(float, copy=True)
        tri[:, 2] = np.abs(tri[:, 2])
        dz = tri[:, 2] - depth
        points = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p0, p1 = tri[i], tri[j]
            d0, d1 = dz[i], dz[j]
            if abs(d0) <= eps and abs(d1) <= eps:
                if np.linalg.norm(p0 - p1) > eps:
                    q0 = p0.copy()
                    q1 = p1.copy()
                    q0[2] = depth
                    q1[2] = depth
                    segments.append((q0, q1))
                    coplanar_edges += 1
            elif abs(d0) <= eps:
                q = p0.copy()
                q[2] = depth
                points.append(q)
            elif abs(d1) <= eps:
                q = p1.copy()
                q[2] = depth
                points.append(q)
            elif d0 * d1 < 0:
                alpha = -d0 / (d1 - d0)
                q = p0 + alpha * (p1 - p0)
                q[2] = depth
                points.append(q)

        unique_points = _unique_points(points, tol=eps)
        if len(unique_points) == 2:
            if np.linalg.norm(unique_points[0] - unique_points[1]) > eps:
                segments.append((unique_points[0], unique_points[1]))
        elif len(unique_points) > 2:
            p0, p1 = _farthest_pair(unique_points)
            if np.linalg.norm(p0 - p1) > eps:
                segments.append((p0, p1))
            degenerate_triangles += 1

    segments = _deduplicate_segments(np.asarray(segments, dtype=float), tol=dedupe_tol)
    info = {
        "degenerate_triangles": degenerate_triangles,
        "coplanar_edges": coplanar_edges,
    }
    return segments, info


def stitch_segments_to_polylines(segments, *, tol):
    """Stitch unordered 3-D line segments into deterministic polylines.

    Parameters
    ----------
    segments : ndarray, shape (n_segments, 2, 3)
        Raw triangle-plane intersection segments.
    tol : float
        Endpoint equality tolerance in the same distance unit as ``segments``.

    Returns
    -------
    polylines : list of ndarray
        Ordered polylines.  Closed loops repeat the first point as the last
        point when the final segment closes the chain.
    info : dict
        Diagnostic information including the number of branch nodes.  Branch
        nodes are not expected for a regular fault isodepth line and should be
        inspected if nonzero.
    """
    segments = np.asarray(segments, dtype=float)
    if segments.size == 0:
        return [], {"polyline_count": 0, "branch_nodes": 0}

    endpoint_map = defaultdict(list)
    for index, segment in enumerate(segments):
        for side in (0, 1):
            endpoint_map[_point_key(segment[side], tol)].append((index, side))

    branch_nodes = sum(1 for values in endpoint_map.values() if len(values) > 2)
    unused = set(range(len(segments)))
    polylines = []

    while unused:
        start_index = min(unused, key=lambda idx: _segment_sort_key(segments[idx], tol))
        unused.remove(start_index)
        line = [segments[start_index, 0], segments[start_index, 1]]

        changed = True
        while changed:
            changed = False
            for at_start in (True, False):
                endpoint = line[0] if at_start else line[-1]
                key = _point_key(endpoint, tol)
                candidates = [
                    (idx, side)
                    for idx, side in endpoint_map.get(key, [])
                    if idx in unused
                ]
                if not candidates:
                    continue
                idx, side = min(
                    candidates,
                    key=lambda item: _point_key(segments[item[0], 1 - item[1]], tol),
                )
                unused.remove(idx)
                other = segments[idx, 1 - side]
                if at_start:
                    line.insert(0, other)
                else:
                    line.append(other)
                changed = True

        polylines.append(np.asarray(line, dtype=float))

    polylines.sort(key=lambda line: (-_polyline_length(line), line[0, 0], line[0, 1]))
    return polylines, {
        "polyline_count": len(polylines),
        "branch_nodes": branch_nodes,
    }


def weld_vertices(vertices, faces, *, tol):
    """Weld vertices whose coordinates fall in the same tolerance cell."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if tol is None or tol <= 0:
        return vertices, faces
    keys = np.round(vertices / tol).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_unique = int(inverse.max()) + 1
    welded = np.zeros((n_unique, 3), dtype=float)
    counts = np.bincount(inverse)
    np.add.at(welded, inverse, vertices)
    welded /= counts[:, None]
    welded_faces = inverse[faces]
    return welded, _drop_degenerate_faces(welded_faces)


def _normalize_method(method):
    return "mesh_plane_intersection" if method is None else str(method).lower()


def _normalize_depths(depths):
    if np.isscalar(depths):
        return [abs(float(depths))]
    return [abs(float(depth)) for depth in depths]


def _patch_to_triangles(patch, *, rect_subdivision):
    patch = np.asarray(patch, dtype=float)
    if patch.shape[0] == 3:
        return patch.copy(), np.asarray([[0, 1, 2]], dtype=int)
    if patch.shape[0] != 4:
        raise ValueError("Only triangular or quadrilateral patches can be triangulated")

    n = int(rect_subdivision or 1)
    if n <= 1:
        return patch.copy(), np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)

    n = max(n, 2)
    u = np.linspace(0.0, 1.0, n)
    v = np.linspace(0.0, 1.0, n)
    points = []

    for vv in v:
        for uu in u:
            p = (
                (1.0 - uu) * (1.0 - vv) * patch[0]
                + uu * (1.0 - vv) * patch[1]
                + uu * vv * patch[2]
                + (1.0 - uu) * vv * patch[3]
            )
            points.append(p)

    faces = []
    for row in range(n - 1):
        for col in range(n - 1):
            i0 = row * n + col
            i1 = row * n + col + 1
            i2 = (row + 1) * n + col + 1
            i3 = (row + 1) * n + col
            faces.append([i0, i1, i2])
            faces.append([i0, i2, i3])
    return np.asarray(points, dtype=float), np.asarray(faces, dtype=int)


def _drop_degenerate_faces(faces):
    faces = np.asarray(faces, dtype=int)
    if faces.size == 0:
        return faces.reshape(0, 3)
    keep = [len(set(map(int, face))) == 3 for face in faces]
    return faces[np.asarray(keep, dtype=bool)]


def _mesh_scale(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.size == 0:
        return 1.0
    extent = np.ptp(vertices, axis=0)
    scale = float(np.max(extent))
    return scale if scale > 0 else 1.0


def _unique_points(points, *, tol):
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - existing) <= tol for existing in unique):
            unique.append(np.asarray(point, dtype=float))
    return unique


def _farthest_pair(points):
    best = (points[0], points[1])
    best_distance = -1.0
    for i, p0 in enumerate(points):
        for p1 in points[i + 1:]:
            distance = float(np.linalg.norm(p0 - p1))
            if distance > best_distance:
                best = (p0, p1)
                best_distance = distance
    return best


def _deduplicate_segments(segments, *, tol):
    segments = np.asarray(segments, dtype=float)
    if segments.size == 0:
        return segments.reshape(0, 2, 3)
    seen = set()
    deduped = []
    for segment in segments:
        key0 = _point_key(segment[0], tol)
        key1 = _point_key(segment[1], tol)
        key = tuple(sorted((key0, key1)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(segment)
    return np.asarray(deduped, dtype=float).reshape(-1, 2, 3)


def _point_key(point, tol):
    return tuple(np.round(np.asarray(point, dtype=float) / tol).astype(np.int64))


def _segment_sort_key(segment, tol):
    keys = [_point_key(segment[0], tol), _point_key(segment[1], tol)]
    return tuple(sorted(keys))


def _polyline_length(line):
    line = np.asarray(line, dtype=float)
    if len(line) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(line[:, :3], axis=0), axis=1)))


def _format_polyline(line, depth, *, converter):
    line = np.asarray(line, dtype=float).copy()
    line[:, 2] = abs(float(depth))
    if converter is None:
        return line
    lon, lat = converter.xy2ll(line[:, 0], line[:, 1])
    return np.column_stack([line[:, 0], line[:, 1], line[:, 2], lon, lat])


def _empty_contour(converter):
    return np.empty((0, 5 if converter is not None else 3), dtype=float)


def _orient_polyline(line, *, sort_by=None, reverse=False):
    if len(line) == 0:
        return line
    if sort_by is None:
        return line[::-1].copy() if reverse else line
    column_map = {"x": 0, "y": 1, "lon": 3, "lat": 4}
    if sort_by not in column_map:
        raise ValueError("sort_by must be one of None, 'x', 'y', 'lon', or 'lat'")
    column = column_map[sort_by]
    if column >= line.shape[1]:
        raise ValueError(f"sort_by='{sort_by}' requires lon/lat columns")
    start_value = line[0, column]
    end_value = line[-1, column]
    should_reverse = end_value < start_value
    if reverse:
        should_reverse = not should_reverse
    return line[::-1].copy() if should_reverse else line


def _max_depth_residual(lines, depth):
    if not lines:
        return np.nan
    residuals = []
    for line in lines:
        if len(line) == 0:
            continue
        residuals.append(float(np.max(np.abs(line[:, 2] - depth))))
    return max(residuals) if residuals else np.nan


def _largest_formatted_length(lines):
    if not lines:
        return 0.0
    return max(_polyline_length(line) for line in lines)
