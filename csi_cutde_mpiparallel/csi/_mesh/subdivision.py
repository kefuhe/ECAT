"""Shared-edge triangular subdivision without coordinate-based welding."""

import numpy as np


def subdivide_triangles(vertices, faces, levels):
    """Return independent vertices, faces and input-face IDs after ``levels``.

    A midpoint belongs to one unordered pair of vertex IDs. Coincident vertices
    with different IDs remain distinct, preserving cracks in the input mesh.
    Children remain face-major in the order used by ``splitPatch``. Their
    winding follows their parent; this kernel never reorients triangles.
    """
    if isinstance(levels, (bool, np.bool_)) or not isinstance(levels, (int, np.integer)):
        raise TypeError("levels must be a non-negative integer.")
    if levels < 0:
        raise ValueError("levels must be a non-negative integer.")

    vertices = np.array(vertices, dtype=float, copy=True)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("vertices must be a non-empty array with shape (n, 3).")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must contain only finite coordinates.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces must be a non-empty array with shape (n, 3).")
    if not np.issubdtype(faces.dtype, np.integer):
        raise TypeError("faces must contain integer vertex IDs.")
    faces = np.array(faces, dtype=np.intp, copy=True)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces contain a vertex ID outside the vertices array.")
    if np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2])
              | (faces[:, 2] == faces[:, 0])):
        raise ValueError("each face must contain three distinct vertex IDs.")

    parent_face_ids = np.arange(len(faces), dtype=np.intp)
    for _ in range(levels):
        # All faces resolve the same unordered edge to the same midpoint ID.
        edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        edges.sort(axis=1)
        unique_edges, inverse = np.unique(edges, axis=0, return_inverse=True)
        midpoint_ids = (len(vertices) + inverse).reshape(-1, 3)
        a, b, c = faces.T
        ab, bc, ca = midpoint_ids.T

        # Preserve both historical child order and each parent's 3D winding.
        children = np.empty((len(faces), 4, 3), dtype=np.intp)
        children[:, 0] = np.column_stack((a, ab, ca))
        children[:, 1] = np.column_stack((ab, b, bc))
        children[:, 2] = np.column_stack((ca, bc, c))
        children[:, 3] = np.column_stack((ca, ab, bc))
        first = vertices[unique_edges[:, 0]]
        midpoints = first + (vertices[unique_edges[:, 1]] - first) / 2.0
        vertices = np.concatenate((vertices, midpoints), axis=0)
        faces = children.reshape(-1, 3)
        parent_face_ids = np.repeat(parent_face_ids, 4)

    return vertices, faces, parent_face_ids
