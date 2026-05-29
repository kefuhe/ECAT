"""
ParametricMeshBackend — arc-length parametric domain mesh generation and deformation.

Generates a Gmsh triangular mesh in a (s, t) parametric domain where s is the
normalized arc length along strike and t is the normalized dip coordinate.
Deformation uses the stored (s, t) coordinates with ruled-surface interpolation,
eliminating the need for projection, rotation, polygon containment, and inverse
bilinear solves.

Standalone module — does not modify any existing code.
"""

import numpy as np
from scipy.interpolate import interp1d

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False


# ---------------------------------------------------------------------------
# Arc-length utilities
# ---------------------------------------------------------------------------

def _cumulative_arc_length(coords):
    """Cumulative chord-length arc length for an (N, 3) polyline."""
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate(([0.0], np.cumsum(seg_lengths)))


def _arc_length_parameterize(coords):
    """
    Compute normalized arc-length parameters for each vertex of a polyline.

    Parameters
    ----------
    coords : (N, 3) array

    Returns
    -------
    s_values : (N,) array — normalized to [0, 1]
    total_length : float
    """
    cum = _cumulative_arc_length(coords)
    total = cum[-1]
    if total < 1e-15:
        return np.zeros(len(coords)), 0.0
    return cum / total, total

def _make_arc_interpolator(coords):
    """
    Create a linear interpolator C(s) -> (x, y, z) parameterized by
    normalized arc length s in [0, 1].

    Parameters
    ----------
    coords : (N, 3) array — polyline vertices

    Returns
    -------
    interp_func : callable — accepts scalar or array of s values,
                  returns (M, 3) array of interpolated points
    total_length : float
    """
    s_values, total_length = _arc_length_parameterize(coords)
    fx = interp1d(s_values, coords[:, 0], kind='linear', assume_sorted=True)
    fy = interp1d(s_values, coords[:, 1], kind='linear', assume_sorted=True)
    fz = interp1d(s_values, coords[:, 2], kind='linear', assume_sorted=True)

    def interp_func(s):
        s = np.asarray(s, dtype=float)
        s_clamped = np.clip(s, 0.0, 1.0)
        x = fx(s_clamped)
        y = fy(s_clamped)
        z = fz(s_clamped)
        return np.column_stack([x, y, z]) if s.ndim >= 1 else np.array([x, y, z])

    return interp_func, total_length


# ---------------------------------------------------------------------------
# Parametric domain Gmsh mesh generation
# ---------------------------------------------------------------------------

def _generate_parametric_gmsh(L_top, D_mean, *,
                              top_size=None, bottom_size=None,
                              mesh_func=None, field_size_dict=None,
                              segments_dict=None,
                              mesh_algorithm=2, optimize_method='Laplace2D',
                              verbose=0, show=False):
    """
    Generate a Gmsh mesh in the rectangular parametric domain [0, L_top] x [0, D_mean].

    Parameters
    ----------
    L_top : float — total arc length of the top curve (km)
    D_mean : float — mean depth extent (km)
    top_size, bottom_size : float or None — mesh element sizes at top/bottom
    mesh_func : bool or None — if True, use Distance+MathEval field sizing
    field_size_dict : dict — {'min_dx': float, 'bias': float}
    segments_dict : dict or None — Transfinite meshing parameters
    mesh_algorithm : int — Gmsh meshing algorithm
    optimize_method : str — Gmsh optimization method
    verbose : int — Gmsh verbosity
    show : bool — open Gmsh GUI

    Returns
    -------
    param_vertices : (N, 2) array — (s_physical, t_physical) in parametric domain
    faces : (M, 3) int array — triangle connectivity (0-based)
    """
    if not HAS_GMSH:
        raise ImportError("gmsh is required for parametric mesh generation")

    gmsh.initialize('', False)
    gmsh.option.setNumber("General.Terminal", verbose)
    gmsh.clear()
    gmsh.option.setNumber("Mesh.Algorithm", mesh_algorithm)

    # Four corners of the parametric rectangle
    ts = top_size if top_size is not None else 0.0
    bs = bottom_size if bottom_size is not None else 0.0

    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, ts)        # top-left
    p2 = gmsh.model.geo.addPoint(L_top, 0.0, 0.0, ts)      # top-right
    p3 = gmsh.model.geo.addPoint(L_top, D_mean, 0.0, bs)    # bottom-right
    p4 = gmsh.model.geo.addPoint(0.0, D_mean, 0.0, bs)      # bottom-left

    # Four edges
    top_curve = gmsh.model.geo.addLine(p1, p2)
    right_edge = gmsh.model.geo.addLine(p2, p3)
    bottom_curve = gmsh.model.geo.addLine(p3, p4)
    left_edge = gmsh.model.geo.addLine(p4, p1)

    # Transfinite segments
    if segments_dict is not None:
        _set_transfinite(top_curve, bottom_curve, left_edge, right_edge, segments_dict)

    # Surface
    loop = gmsh.model.geo.addCurveLoop([top_curve, right_edge, bottom_curve, left_edge])
    surface = gmsh.model.geo.addPlaneSurface([loop])

    if segments_dict is not None:
        gmsh.model.geo.mesh.setTransfiniteSurface(surface)

    gmsh.model.geo.synchronize()

    # Mesh sizing via Distance + MathEval field
    if mesh_func:
        if field_size_dict is None:
            field_size_dict = {'min_dx': 3.0, 'bias': 1.05}

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        field_distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_distance, "CurvesList", [top_curve])

        min_dx = field_size_dict['min_dx']
        bias = field_size_dict['bias']
        math_exp = (f"{min_dx}*{bias}^(Log(1.0+F{field_distance}"
                    f"/{min_dx}*({bias}-1.0))/Log({bias}))")

        field_size = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(field_size, "F", math_exp)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_size)

    # Generate mesh
    gmsh.model.mesh.generate(2)
    if optimize_method:
        gmsh.model.mesh.optimize(optimize_method)

    if show:
        import sys
        if 'close' not in sys.argv:
            gmsh.fltk.run()

    # Extract vertices and faces
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    vertices_3d = node_coords.reshape(-1, 3)
    param_vertices = vertices_3d[:, :2]  # (x, y) in parametric domain = (s_phys, t_phys)

    _, _, node_tags_elem = gmsh.model.mesh.getElements(dim=2)
    faces_raw = np.array(node_tags_elem[0], dtype=int).reshape(-1, 3)

    # Remap node tags to 0-based contiguous indices
    tag_to_idx = {int(tag): idx for idx, tag in enumerate(node_tags)}
    faces_remapped = np.vectorize(tag_to_idx.get)(faces_raw)

    gmsh.finalize()

    return param_vertices, faces_remapped


def _set_transfinite(top_curve, bottom_curve, left_edge, right_edge, segments_dict):
    """Apply Transfinite curve settings from segments_dict."""
    top_seg = segments_dict.get('top_segments', 1)
    bot_seg = segments_dict.get('bottom_segments', 1)
    left_seg = segments_dict.get('left_segments', 1)
    right_seg = segments_dict.get('right_segments', 1)
    tb_prog = segments_dict.get('top_bottom_progression', 1.0)
    lr_prog = segments_dict.get('left_right_progression', 1.0)

    if top_seg != bot_seg or left_seg != right_seg:
        raise ValueError("Opposite sides must have equal segment counts.")

    gmsh.model.geo.mesh.setTransfiniteCurve(top_curve, top_seg, "Progression", tb_prog)
    gmsh.model.geo.mesh.setTransfiniteCurve(bottom_curve, bot_seg, "Progression", tb_prog)
    gmsh.model.geo.mesh.setTransfiniteCurve(left_edge, left_seg, "Progression", 1 / lr_prog)
    gmsh.model.geo.mesh.setTransfiniteCurve(right_edge, right_seg, "Progression", lr_prog)


# ---------------------------------------------------------------------------
# ParametricMeshBackend
# ---------------------------------------------------------------------------

class ParametricMeshBackend:
    """
    Arc-length parametric domain mesh generation and deformation.

    Workflow:
        1. generate(top_coords, bottom_coords, ...) — creates a Gmsh mesh in
           the (s, t) parametric domain and maps it to physical space.
        2. deform(new_top, new_bottom) — rebuilds physical vertices from stored
           (s, t) using ruled-surface interpolation on the new geometry.
    """

    def __init__(self):
        self.s_coords = None       # (N,) normalized arc-length [0, 1]
        self.t_coords = None       # (N,) normalized dip coordinate [0, 1]
        self.faces = None          # (M, 3) triangle connectivity
        self._L_top = None         # top curve total arc length
        self._D_mean = None        # mean depth extent

    def generate(self, top_coords, bottom_coords, *,
                 top_size=None, bottom_size=None,
                 mesh_func=None, field_size_dict=None,
                 segments_dict=None,
                 mesh_algorithm=2, optimize_method='Laplace2D',
                 verbose=0, show=False):
        """
        Generate a mesh in the parametric domain and map to physical space.

        Parameters
        ----------
        top_coords : (N, 3) array — top curve vertices (x, y, z)
        bottom_coords : (N, 3) array — bottom curve vertices (x, y, z)
        top_size, bottom_size : float or None — element sizes
        mesh_func : bool or None — use Distance+MathEval sizing
        field_size_dict : dict — {'min_dx': float, 'bias': float}
        segments_dict : dict or None — Transfinite meshing
        mesh_algorithm : int
        optimize_method : str
        verbose : int
        show : bool

        Returns
        -------
        vertices : (N, 3) array — physical space coordinates
        faces : (M, 3) int array — triangle connectivity
        """
        top_coords = np.asarray(top_coords, dtype=float)
        bottom_coords = np.asarray(bottom_coords, dtype=float)

        # Arc-length parameterization
        top_interp, L_top = _make_arc_interpolator(top_coords)
        bot_interp, L_bot = _make_arc_interpolator(bottom_coords)

        # Mean depth extent
        diffs = bottom_coords - top_coords
        D_mean = np.mean(np.linalg.norm(diffs, axis=1))

        self._L_top = L_top
        self._D_mean = D_mean

        # Generate mesh in parametric domain
        param_verts, faces = _generate_parametric_gmsh(
            L_top, D_mean,
            top_size=top_size, bottom_size=bottom_size,
            mesh_func=mesh_func, field_size_dict=field_size_dict,
            segments_dict=segments_dict,
            mesh_algorithm=mesh_algorithm, optimize_method=optimize_method,
            verbose=verbose, show=show,
        )

        # Convert parametric coords to normalized [0, 1]
        s_norm = param_verts[:, 0] / L_top if L_top > 1e-15 else np.zeros(len(param_verts))
        t_norm = param_verts[:, 1] / D_mean if D_mean > 1e-15 else np.zeros(len(param_verts))

        s_norm = np.clip(s_norm, 0.0, 1.0)
        t_norm = np.clip(t_norm, 0.0, 1.0)

        self.s_coords = s_norm
        self.t_coords = t_norm
        self.faces = faces

        # Map to physical space via ruled surface
        vertices = self._ruled_surface(top_interp, bot_interp)
        return vertices, faces

    def deform(self, new_top_coords, new_bottom_coords):
        """
        Rebuild physical vertices from stored (s, t) on new geometry.

        Parameters
        ----------
        new_top_coords : (N, 3) array
        new_bottom_coords : (N, 3) array

        Returns
        -------
        vertices : (N, 3) array — new physical coordinates
        """
        if self.s_coords is None:
            raise RuntimeError("Must call generate() before deform()")

        new_top_coords = np.asarray(new_top_coords, dtype=float)
        new_bottom_coords = np.asarray(new_bottom_coords, dtype=float)

        top_interp, _ = _make_arc_interpolator(new_top_coords)
        bot_interp, _ = _make_arc_interpolator(new_bottom_coords)

        return self._ruled_surface(top_interp, bot_interp)

    def get_param_coords(self):
        """Return stored (s_coords, t_coords) for debugging."""
        return self.s_coords, self.t_coords

    def _ruled_surface(self, top_interp, bot_interp):
        """X(s, t) = (1 - t) * C_top(s) + t * C_bot(s)"""
        top_pts = top_interp(self.s_coords)   # (N, 3)
        bot_pts = bot_interp(self.s_coords)   # (N, 3)
        t = self.t_coords[:, np.newaxis]      # (N, 1)
        return (1.0 - t) * top_pts + t * bot_pts
