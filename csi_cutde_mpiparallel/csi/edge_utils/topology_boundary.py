"""Topology-based boundary extraction for triangular fault meshes.

This module keeps the low-level mesh logic independent from ``TriangularPatches``.
It assumes one fault segment per mesh object, a single connected manifold
boundary loop, and no interior holes.
"""

from collections import Counter, defaultdict, deque

import numpy as np


EDGE_NAMES = ("top", "bottom", "left", "right")


def extract_topology_boundary_loop(vertices, faces):
    """Return an ordered boundary loop and boundary-edge metadata.

    Boundary edges are undirected triangle edges that occur exactly once. The
    current implementation expects the largest boundary component to be a
    closed manifold loop where every boundary node has degree 2.
    """
    edge_count = Counter()
    edge_faces = defaultdict(list)
    for itri, tri in enumerate(np.asarray(faces, dtype=int)):
        for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(i), int(j))))
            edge_count[edge] += 1
            edge_faces[edge].append(int(itri))

    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    graph = defaultdict(list)
    for i, j in boundary_edges:
        graph[i].append(j)
        graph[j].append(i)

    if not graph:
        raise ValueError("No boundary edges found in triangular mesh.")

    components = []
    visited = set()
    for node in graph:
        if node in visited:
            continue
        queue = deque([node])
        visited.add(node)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    components = sorted(components, key=len, reverse=True)
    boundary_nodes = set(components[0])
    bad_degrees = {
        node: len(graph[node])
        for node in boundary_nodes
        if len(graph[node]) != 2
    }
    if bad_degrees:
        raise ValueError(f"Boundary is not a closed manifold loop. Bad node degrees: {bad_degrees}")

    start = min(boundary_nodes, key=lambda node: (vertices[node, 2], vertices[node, 0], vertices[node, 1]))
    neighbors = sorted(graph[start], key=lambda node: (vertices[node, 2], vertices[node, 0], vertices[node, 1]))
    previous = None
    current = start
    next_node = neighbors[0]
    loop = [start]

    while True:
        previous, current = current, next_node
        if current == start:
            break
        loop.append(current)
        candidates = graph[current]
        next_node = candidates[0] if candidates[0] != previous else candidates[1]
        if len(loop) > len(boundary_nodes) + 1:
            raise ValueError("Boundary loop ordering did not close as expected.")

    info = {
        "boundary_edge_count": len(boundary_edges),
        "boundary_node_count": len(graph),
        "component_node_counts": [len(component) for component in components],
    }
    return np.array(loop, dtype=int), boundary_edges, edge_faces, info


def cyclic_label_runs(labels):
    """Return cyclic runs as ``(label, positions)`` pairs."""
    labels = np.asarray(labels, dtype=object)
    nlabels = len(labels)
    breaks = [i for i in range(nlabels) if labels[i] != labels[i - 1]]
    if not breaks:
        return [(labels[0], list(range(nlabels)))]

    start = breaks[0]
    runs = []
    current_label = labels[start]
    current_positions = []
    for offset in range(nlabels):
        pos = (start + offset) % nlabels
        if labels[pos] != current_label and current_positions:
            runs.append((current_label, current_positions))
            current_label = labels[pos]
            current_positions = []
        current_positions.append(pos)
    runs.append((current_label, current_positions))
    return runs


def bridge_short_side_gaps(labels, max_gap_points=0):
    """Relabel short side gaps between matching top/bottom labels.

    This is only used by ``gap_policy='standardize'``. The default topology
    backend does not bridge gaps.
    """
    labels = np.asarray(labels, dtype=object).copy()
    if max_gap_points is None or max_gap_points <= 0:
        return labels

    changed = True
    while changed:
        changed = False
        runs = cyclic_label_runs(labels)
        for irun, (label, positions) in enumerate(runs):
            if label != "side" or len(positions) > max_gap_points:
                continue
            prev_label = runs[irun - 1][0]
            next_label = runs[(irun + 1) % len(runs)][0]
            if prev_label == next_label and prev_label != "side":
                labels[positions] = prev_label
                changed = True
                break
    return labels


def classify_side_runs(runs):
    """Classify side-labeled runs by neighboring labels."""
    side_info = []
    for irun, (label, positions) in enumerate(runs):
        if label != "side":
            continue
        prev_label = runs[irun - 1][0]
        next_label = runs[(irun + 1) % len(runs)][0]
        neighbor_labels = {prev_label, next_label}
        if neighbor_labels == {"top", "bottom"}:
            kind = "true_side"
        elif prev_label == next_label and prev_label in {"top", "bottom"}:
            kind = f"{prev_label}_gap"
        else:
            kind = "ambiguous_side"
        side_info.append(
            {
                "kind": kind,
                "prev_label": prev_label,
                "next_label": next_label,
                "positions": positions,
                "point_count": len(positions),
            }
        )
    return side_info


def find_short_side_gap_candidates(labels, loop_points, top_depth, bottom_depth, max_gap_points=2):
    """Collect diagnostics for short side runs between matching edge labels."""
    labels = np.asarray(labels, dtype=object)
    if max_gap_points is None or max_gap_points <= 0:
        return []

    candidates = []
    runs = cyclic_label_runs(labels)
    for irun, (label, positions) in enumerate(runs):
        if label != "side" or len(positions) > max_gap_points:
            continue
        prev_label = runs[irun - 1][0]
        next_label = runs[(irun + 1) % len(runs)][0]
        if prev_label != next_label or prev_label == "side":
            continue

        points = loop_points[positions]
        reference_depth = top_depth if prev_label == "top" else bottom_depth
        if len(points) > 1:
            segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            length = float(np.sum(segment_lengths))
        else:
            length = 0.0
        candidates.append(
            {
                "target_edge": prev_label,
                "point_count": len(positions),
                "loop_positions": [int(pos) for pos in positions],
                "max_depth_offset": float(np.max(np.abs(points[:, 2] - reference_depth))),
                "approx_length_km": length,
                "first_point_xyz": points[0].tolist(),
                "last_point_xyz": points[-1].tolist(),
            }
        )
    return candidates


def principal_xy_direction(points):
    """Estimate the main horizontal direction using PCA."""
    xy = np.asarray(points, dtype=float)[:, :2]
    centered = xy - np.mean(xy, axis=0)
    if len(xy) < 2 or np.allclose(centered, 0.0):
        return np.array([1.0, 0.0])
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.array([1.0, 0.0])
    return direction / norm


def orient_strike_for_left_right(direction):
    """Orient the projection vector and return the geographic naming rule."""
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    if abs(direction[0]) >= abs(direction[1]):
        if direction[0] < 0:
            direction = -direction
        naming_rule = "west_east"
    else:
        if direction[1] > 0:
            direction = -direction
        naming_rule = "north_south"
    return direction, naming_rule


def _edge_run_boundary_edges(loop, positions):
    if len(positions) == 0:
        return []
    position_set = set(positions)
    edges = []
    nloop = len(loop)
    for pos in positions:
        nxt = (pos + 1) % nloop
        if nxt in position_set:
            edges.append(tuple(sorted((int(loop[pos]), int(loop[nxt])))))
    return edges


def _side_run_boundary_edges(loop, positions):
    if len(positions) == 0:
        return []
    nloop = len(loop)
    edge_positions = [(positions[0] - 1) % nloop] + list(positions)
    return [
        tuple(sorted((int(loop[pos]), int(loop[(pos + 1) % nloop]))))
        for pos in edge_positions
    ]


def _unique_edge_faces(edges, edge_faces):
    faces = []
    seen = set()
    for edge in edges:
        for face in edge_faces.get(tuple(sorted(edge)), []):
            if face not in seen:
                seen.add(face)
                faces.append(face)
    return np.array(faces, dtype=int)


def _unique_vertices_from_faces(faces, face_indices):
    if len(face_indices) == 0:
        return np.array([], dtype=int)
    return np.unique(np.asarray(faces, dtype=int)[face_indices].ravel())


def extract_four_edges_topology(
    vertices,
    faces,
    top_tolerance=0.1,
    bottom_tolerance=0.1,
    gap_policy="clean",
    bridge_gap_points=0,
    short_gap_points=2,
    side_axis="strike",
):
    """Extract four fault edges from triangular mesh topology.

    Edge orientation convention:
    - top: left -> right
    - bottom: left -> right
    - left: top -> bottom
    - right: top -> bottom

    ``gap_policy`` controls short top/top or bottom/bottom gaps:
    - ``strict``: reject non-standard boundary runs.
    - ``clean``: omit short gap points from canonical edge curves and warn.
    - ``diagnostic``: keep top/bottom gaps as separate segments in diagnostics.
    - ``standardize``: bridge short gaps before splitting.
    """
    if gap_policy not in {"strict", "clean", "diagnostic", "standardize"}:
        raise ValueError("gap_policy must be 'strict', 'clean', 'diagnostic', or 'standardize'.")
    if side_axis not in {"strike", "auto", "x", "y"}:
        raise ValueError("side_axis must be 'strike', 'auto', 'x', or 'y'.")

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    loop, boundary_edges, edge_faces, loop_info = extract_topology_boundary_loop(vertices, faces)
    loop_points = vertices[loop]
    top_depth = np.min(vertices[:, 2])
    bottom_depth = np.max(vertices[:, 2])

    labels = np.full(len(loop), "side", dtype=object)
    labels[np.abs(loop_points[:, 2] - top_depth) <= top_tolerance] = "top"
    labels[np.abs(loop_points[:, 2] - bottom_depth) <= bottom_tolerance] = "bottom"
    short_gap_candidates = find_short_side_gap_candidates(
        labels, loop_points, top_depth, bottom_depth, max_gap_points=short_gap_points
    )
    raw_runs = cyclic_label_runs(labels)
    raw_side_run_info = classify_side_runs(raw_runs)
    if gap_policy == "standardize":
        labels = bridge_short_side_gaps(labels, max_gap_points=bridge_gap_points)
    runs = cyclic_label_runs(labels)
    side_run_info = classify_side_runs(runs)

    top_runs = [positions for label, positions in runs if label == "top"]
    bottom_runs = [positions for label, positions in runs if label == "bottom"]
    true_side_runs = [item["positions"] for item in side_run_info if item["kind"] == "true_side"]
    nontrue_side_runs = [item for item in side_run_info if item["kind"] != "true_side"]

    if gap_policy == "strict":
        strict_errors = []
        if len(top_runs) != 1:
            strict_errors.append(f"top run count is {len(top_runs)}, expected 1")
        if len(bottom_runs) != 1:
            strict_errors.append(f"bottom run count is {len(bottom_runs)}, expected 1")
        if nontrue_side_runs:
            strict_errors.append(
                "non-true side runs exist: "
                + str([(item["kind"], item["point_count"]) for item in nontrue_side_runs])
            )
        if len(true_side_runs) != 2:
            strict_errors.append(f"true side run count is {len(true_side_runs)}, expected 2")
        if strict_errors:
            raise ValueError("Strict boundary extraction failed: " + "; ".join(strict_errors))

    if gap_policy == "clean":
        clean_errors = []
        for item in nontrue_side_runs:
            if item["kind"] not in {"top_gap", "bottom_gap"}:
                clean_errors.append(f"{item['kind']} cannot be cleaned automatically")
            if item["point_count"] > short_gap_points:
                clean_errors.append(
                    f"{item['kind']} has {item['point_count']} points, "
                    f"larger than short_gap_points={short_gap_points}"
                )
        if clean_errors:
            raise ValueError("Clean boundary extraction failed: " + "; ".join(clean_errors))

    if not top_runs or not bottom_runs or len(true_side_runs) != 2:
        raise ValueError(
            f"Cannot split boundary loop into four edges: "
            f"top={len(top_runs)}, bottom={len(bottom_runs)}, true_side={len(true_side_runs)}, "
            f"other_side={[(item['kind'], item['point_count']) for item in nontrue_side_runs]}"
        )

    top_positions = [position for positions in top_runs for position in positions]
    bottom_positions = [position for positions in bottom_runs for position in positions]
    side_runs = true_side_runs

    def positions_to_indices(positions, include_neighbors=False):
        edge_positions = list(positions)
        if include_neighbors:
            edge_positions = [(positions[0] - 1) % len(loop)] + edge_positions + [(positions[-1] + 1) % len(loop)]
        return loop[edge_positions]

    def positions_to_points(positions, include_neighbors=False):
        return vertices[positions_to_indices(positions, include_neighbors=include_neighbors)]

    top_points = positions_to_points(top_positions)
    bottom_points = positions_to_points(bottom_positions)
    side_points = [positions_to_points(positions, include_neighbors=True) for positions in side_runs]

    strike_direction = principal_xy_direction(np.vstack([top_points, bottom_points]))
    if side_axis == "strike":
        projection_direction, naming_rule = orient_strike_for_left_right(strike_direction)
    elif side_axis == "auto":
        mean0 = [np.mean(points[:, 0]) for points in side_points]
        mean1 = [np.mean(points[:, 1]) for points in side_points]
        axis_index = 0 if abs(mean0[0] - mean0[1]) >= abs(mean1[0] - mean1[1]) else 1
        projection_direction = np.array([1.0, 0.0]) if axis_index == 0 else np.array([0.0, -1.0])
        naming_rule = "west_east" if axis_index == 0 else "north_south"
    elif side_axis == "x":
        projection_direction = np.array([1.0, 0.0])
        naming_rule = "west_east"
    else:
        projection_direction = np.array([0.0, -1.0])
        naming_rule = "north_south"

    side_projections = [float(np.mean(points[:, :2] @ projection_direction)) for points in side_points]
    left_index = int(np.argmin(side_projections))
    right_index = int(np.argmax(side_projections))

    def orient_along_strike(points, indices):
        projections = points[:, :2] @ projection_direction
        if projections[0] > projections[-1]:
            return points[::-1], indices[::-1]
        return points, indices

    def orient_downward(points, indices):
        if points[0, 2] > points[-1, 2]:
            return points[::-1], indices[::-1]
        return points, indices

    def make_horizontal(edge_runs):
        segments = []
        index_segments = []
        for positions in edge_runs:
            indices = positions_to_indices(positions)
            points, indices = orient_along_strike(vertices[indices], indices)
            segments.append(points)
            index_segments.append(indices)
        order = np.argsort([np.mean(points[:, :2] @ projection_direction) for points in segments])
        segments = [segments[i] for i in order]
        index_segments = [index_segments[i] for i in order]
        if gap_policy == "clean" and len(segments) > 1:
            segments = [np.vstack(segments)]
            index_segments = [np.concatenate(index_segments)]
        return segments, index_segments

    top_segments, top_index_segments = make_horizontal(top_runs)
    bottom_segments, bottom_index_segments = make_horizontal(bottom_runs)

    left_indices = positions_to_indices(side_runs[left_index], include_neighbors=True)
    right_indices = positions_to_indices(side_runs[right_index], include_neighbors=True)
    left_segment, left_indices = orient_downward(vertices[left_indices], left_indices)
    right_segment, right_indices = orient_downward(vertices[right_indices], right_indices)

    edge_segments = {
        "top": top_segments,
        "bottom": bottom_segments,
        "left": [left_segment],
        "right": [right_segment],
    }
    edge_index_segments = {
        "top": top_index_segments,
        "bottom": bottom_index_segments,
        "left": [left_indices],
        "right": [right_indices],
    }
    edge_vertices = {edge_name: np.vstack(edge_segments[edge_name]) for edge_name in EDGE_NAMES}
    edge_vertex_indices = {edge_name: np.concatenate(edge_index_segments[edge_name]).astype(int) for edge_name in EDGE_NAMES}

    side_edge_runs = {
        "left": side_runs[left_index],
        "right": side_runs[right_index],
    }
    edge_boundary_edges = {
        "top": [edge for positions in top_runs for edge in _edge_run_boundary_edges(loop, positions)],
        "bottom": [edge for positions in bottom_runs for edge in _edge_run_boundary_edges(loop, positions)],
        "left": _side_run_boundary_edges(loop, side_edge_runs["left"]),
        "right": _side_run_boundary_edges(loop, side_edge_runs["right"]),
    }
    edge_triangles_indices = {
        edge_name: _unique_edge_faces(edge_boundary_edges[edge_name], edge_faces)
        for edge_name in EDGE_NAMES
    }
    edge_triangle_vertex_indices = {
        edge_name: _unique_vertices_from_faces(faces, edge_triangles_indices[edge_name])
        for edge_name in EDGE_NAMES
    }
    edge_dict = {edge_name: edge_triangles_indices[edge_name].tolist() for edge_name in EDGE_NAMES}
    corner_dict = {
        "left_top": None,
        "right_top": None,
        "left_bottom": None,
        "right_bottom": None,
    }

    info = {
        **loop_info,
        "top_depth": float(top_depth),
        "bottom_depth": float(bottom_depth),
        "gap_policy": gap_policy,
        "cleaned_gap_summary": [
            {
                "kind": item["kind"],
                "point_count": item["point_count"],
                "prev_label": item["prev_label"],
                "next_label": item["next_label"],
            }
            for item in nontrue_side_runs
        ] if gap_policy == "clean" else [],
        "raw_run_summary": [(label, len(positions)) for label, positions in raw_runs],
        "raw_side_run_summary": [
            {
                "kind": item["kind"],
                "prev_label": item["prev_label"],
                "next_label": item["next_label"],
                "point_count": item["point_count"],
            }
            for item in raw_side_run_info
        ],
        "run_summary": [(label, len(positions)) for label, positions in runs],
        "top_run_lengths": [len(positions) for positions in top_runs],
        "bottom_run_lengths": [len(positions) for positions in bottom_runs],
        "side_run_lengths_used": [len(positions) for positions in side_runs],
        "nontrue_side_run_summary": [
            {
                "kind": item["kind"],
                "prev_label": item["prev_label"],
                "next_label": item["next_label"],
                "point_count": item["point_count"],
            }
            for item in nontrue_side_runs
        ],
        "side_axis": side_axis,
        "left_right_naming_rule": naming_rule,
        "strike_vector_xy": strike_direction.tolist(),
        "projection_vector_xy": projection_direction.tolist(),
        "side_projection_means": side_projections,
        "short_side_gap_candidates": short_gap_candidates,
        "bridge_gap_points": bridge_gap_points,
        "edge_segment_counts": {edge_name: len(edge_segments[edge_name]) for edge_name in EDGE_NAMES},
    }

    return {
        "edge_vertices": edge_vertices,
        "edge_vertex_indices": edge_vertex_indices,
        "edge_triangles_indices": edge_triangles_indices,
        "edge_triangle_vertex_indices": edge_triangle_vertex_indices,
        "edge_dict": edge_dict,
        "corner_dict": corner_dict,
        "edge_segments": edge_segments,
        "edge_index_segments": edge_index_segments,
        "info": info,
    }
