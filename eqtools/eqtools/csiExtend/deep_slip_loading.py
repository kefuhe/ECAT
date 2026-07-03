"""Deep-slip loading proxy helpers.

This module implements the geometry mapping and linear constraint matrices for
the deep-slip loading proxy model.  The helpers are intentionally stateless and
do not interpret Euler/block loading; both shallow and deep variables are normal
fault slip parameters.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .interseismic_fields import (
    _get_source_param_names,
    _get_source_start,
    normalize_slip_component,
)
from .patch_indices import (
    get_patch_centers,
    normalize_patch_indices,
    select_patch_indices,
)


DEEP_SLIP_EQUALITY_STATES = {
    "bottom_continuity": "bottom_continuity",
    "continuity": "bottom_continuity",
    "full_creep": "full_creep",
    "creep": "full_creep",
    "free_slip": "full_creep",
    "full_locking": "full_locking",
    "locked": "full_locking",
    "zero_shallow_slip": "full_locking",
    "prescribed_creep_ratio": "prescribed_creep_ratio",
    "creep_ratio": "prescribed_creep_ratio",
    "prescribed_ratio": "prescribed_creep_ratio",
    "prescribed_locking": "prescribed_locking",
    "locking": "prescribed_locking",
    "fixed_shallow_slip": "fixed_shallow_slip",
    "fixed_slip": "fixed_shallow_slip",
    "cap": "cap",
    "same_direction_cap": "cap",
    "free": "free",
}


def normalize_deep_slip_state(state: str) -> str:
    """Return the canonical deep-slip proxy constraint state.

    Parameters
    ----------
    state : str
        Public state name.  Common values are ``bottom_continuity``,
        ``full_creep``, ``full_locking``, ``prescribed_creep_ratio``,
        ``prescribed_locking``, ``fixed_shallow_slip`` and ``cap``.

    Returns
    -------
    str
        Canonical state name.
    """
    key = str(state).lower().replace("-", "_").replace(" ", "_")
    try:
        return DEEP_SLIP_EQUALITY_STATES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(DEEP_SLIP_EQUALITY_STATES))
        raise ValueError(f"Unknown deep-slip loading constraint state '{state}'. Accepted: {allowed}") from exc


def _motion_sign(motion_sense: str | int | float | None) -> float:
    """Return +1 for expected positive slip and -1 for expected negative slip."""
    if motion_sense is None:
        raise ValueError("motion_sense or motion_sign is required for cap constraints")
    if isinstance(motion_sense, (int, float, np.integer, np.floating)):
        value = float(motion_sense)
        if value == 0.0:
            raise ValueError("motion_sign cannot be zero")
        return 1.0 if value > 0.0 else -1.0

    key = str(motion_sense).lower().replace("-", "_").replace(" ", "_")
    if key in ("positive", "pos", "+", "sinistral", "left", "left_lateral"):
        return 1.0
    if key in ("negative", "neg", "-", "dextral", "right", "right_lateral"):
        return -1.0
    raise ValueError(
        "motion_sense must be positive/sinistral/left or negative/dextral/right "
        "for cap constraints"
    )


def _ensure_edge_selector_ready(fault: Any, selector: Any) -> None:
    """Best-effort edge detection before a selector such as ``{"edge": "bottom"}``.

    The function is deliberately conservative: it only tries established CSI /
    eqtools edge methods when the selector asks for an edge and
    ``edge_triangles_indices`` is missing.  All failures are ignored so the
    normal selector error can report the missing edge metadata.
    """
    if not isinstance(selector, Mapping):
        return
    if "edge" not in selector and "edges" not in selector:
        return
    if hasattr(fault, "edge_triangles_indices"):
        return

    for method_name in (
        "find_fault_fouredge_vertices",
        "find_fault_edge_vertices",
        "find_ordered_edge_vertices",
    ):
        method = getattr(fault, method_name, None)
        if method is None:
            continue
        try:
            if method_name == "find_ordered_edge_vertices":
                edges = selector.get("edge", selector.get("edges"))
                edge_list = [edges] if isinstance(edges, str) else list(edges)
                for edge in edge_list:
                    try:
                        method(edge=edge)
                    except TypeError:
                        method(edge)
            else:
                try:
                    method(refind=False)
                except TypeError:
                    method()
        except Exception:
            continue
        if hasattr(fault, "edge_triangles_indices"):
            return


def _patch_vertices(
    fault: Any,
    patch_index: int,
    *,
    coord_frame: str = "same_xy",
    reference_fault: Any | None = None,
) -> np.ndarray:
    """Return one patch's vertices as ``x, y, depth`` rows."""
    key = str(coord_frame).lower().replace("-", "_")
    if key in ("same_xy", "xy", "local", "utm"):
        patch = np.asarray(fault.patch[int(patch_index)], dtype=float)
        if patch.ndim == 2 and patch.shape[1] >= 3:
            return patch[:, :3].copy()
        if hasattr(fault, "Vertices") and hasattr(fault, "Faces"):
            face = np.asarray(fault.Faces[int(patch_index)], dtype=int)
            return np.asarray(fault.Vertices[face], dtype=float)[:, :3].copy()
        raise ValueError(
            f"Cannot read xyz vertices for patch {patch_index} on fault "
            f"'{getattr(fault, 'name', '<unnamed>')}'."
        )

    if key in ("shallow_xy_from_lonlat", "lonlat_to_reference_xy"):
        if reference_fault is None or not hasattr(reference_fault, "ll2xy"):
            raise AttributeError("reference_fault with ll2xy() is required for lonlat fallback")
        if not hasattr(fault, "patchll") or getattr(fault, "patchll") is None:
            if not hasattr(fault, "patch2ll"):
                raise AttributeError("fault has no patchll and no patch2ll() method")
            fault.patch2ll()
        patchll = np.asarray(fault.patchll[int(patch_index)], dtype=float)
        if patchll.ndim != 2 or patchll.shape[1] < 3:
            raise ValueError("fault.patchll entries must contain lon, lat, depth columns")
        x, y = reference_fault.ll2xy(patchll[:, 0], patchll[:, 1])
        return np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float), patchll[:, 2]))

    raise ValueError("coord_frame must be 'same_xy' or 'shallow_xy_from_lonlat'")


def get_patch_top_segments(
    fault: Any,
    patch_indices: Sequence[int] | int | None = None,
    *,
    coord_frame: str = "same_xy",
    reference_fault: Any | None = None,
    top_edge_policy: str = "infer",
    top_edge_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Extract inferred top-edge segments for selected patches.

    The top edge is the polygon edge with the smallest mean depth.  This avoids
    hard-coding a rectangular patch vertex order and works for both rectangular
    and triangular patches.  For ambiguous triangular edges, ``top_edge_policy``
    can be set to ``"strict"`` to raise instead of silently choosing one edge.

    Parameters
    ----------
    fault : object
        CSI fault-like object with ``patch`` vertices.
    patch_indices : sequence of int, int, optional
        Candidate patch ids.  ``None`` means all patches.
    coord_frame : {"same_xy", "shallow_xy_from_lonlat"}, default "same_xy"
        Coordinate frame used for returned vertices.
    reference_fault : object, optional
        Fault whose ``ll2xy`` is used only for ``coord_frame`` lon/lat fallback.
    top_edge_policy : {"infer", "strict"}, default "infer"
        Ambiguity handling for patches whose shallowest edges are tied.
    top_edge_tolerance : float, default 1e-8
        Depth tolerance for ambiguity detection, in the same depth unit as
        patch vertices.

    Returns
    -------
    dict
        ``segment_start`` and ``segment_end`` arrays, selected patch ids, and
        ambiguity flags.
    """
    indices = normalize_patch_indices(fault, patch_indices, allow_none_all=True, unique=True)
    policy = str(top_edge_policy).lower()
    if policy not in ("infer", "strict"):
        raise ValueError("top_edge_policy must be 'infer' or 'strict'")

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    ambiguous: list[bool] = []
    chosen_edges: list[tuple[int, int]] = []

    for patch_id in indices:
        vertices = _patch_vertices(
            fault,
            int(patch_id),
            coord_frame=coord_frame,
            reference_fault=reference_fault,
        )
        if vertices.shape[0] < 3:
            raise ValueError(f"Patch {patch_id} has fewer than three vertices")
        if vertices.shape[1] < 3:
            raise ValueError(f"Patch {patch_id} vertices must contain x/y/depth")

        edge_pairs = [(i, (i + 1) % vertices.shape[0]) for i in range(vertices.shape[0])]
        edge_depths = np.asarray(
            [(vertices[i, 2] + vertices[j, 2]) / 2.0 for i, j in edge_pairs],
            dtype=float,
        )
        order = np.argsort(edge_depths)
        tied = bool(
            len(order) > 1 and abs(edge_depths[order[1]] - edge_depths[order[0]]) <= top_edge_tolerance
        )
        if tied and policy == "strict":
            raise ValueError(
                f"Patch {patch_id} has ambiguous top-edge candidates within "
                f"tolerance {top_edge_tolerance}"
            )
        i, j = edge_pairs[int(order[0])]
        segment = (vertices[i, :3].astype(float), vertices[j, :3].astype(float))
        length = np.linalg.norm(segment[1] - segment[0])
        if length <= 0.0:
            raise ValueError(f"Patch {patch_id} inferred top edge has zero length")
        starts.append(segment[0])
        ends.append(segment[1])
        ambiguous.append(tied)
        chosen_edges.append((int(i), int(j)))

    return {
        "fault_name": getattr(fault, "name", None),
        "patch_indices": indices,
        "segment_start": np.vstack(starts) if starts else np.zeros((0, 3), dtype=float),
        "segment_end": np.vstack(ends) if ends else np.zeros((0, 3), dtype=float),
        "ambiguous_top_edge": np.asarray(ambiguous, dtype=bool),
        "chosen_edge_vertices": chosen_edges,
        "coord_frame": coord_frame,
        "top_edge_policy": policy,
    }


def point_to_segment_distance_3d(
    points: Sequence[Sequence[float]],
    segment_start: Sequence[Sequence[float]],
    segment_end: Sequence[Sequence[float]],
    *,
    chunk_size: int = 4096,
) -> dict[str, np.ndarray]:
    """Return nearest 3-D line segment for each point.

    Parameters
    ----------
    points : array-like, shape (n, 3)
        Query points.
    segment_start, segment_end : array-like, shape (m, 3)
        Segment endpoints.
    chunk_size : int, default 4096
        Number of points processed per vectorized chunk.

    Returns
    -------
    dict
        ``nearest_segment``, ``distance`` and ``closest_point`` arrays.
    """
    pts = np.asarray(points, dtype=float)
    a = np.asarray(segment_start, dtype=float)
    b = np.asarray(segment_end, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (n, 3)")
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[1] < 3:
        raise ValueError("segment_start and segment_end must both have shape (m, 3)")
    if a.shape[0] == 0:
        raise ValueError("At least one segment is required")

    pts = pts[:, :3]
    a = a[:, :3]
    b = b[:, :3]
    u = b - a
    denom = np.sum(u * u, axis=1)
    if np.any(denom <= 0.0):
        bad = np.nonzero(denom <= 0.0)[0].tolist()
        raise ValueError(f"Zero-length segment(s): {bad[:10]}")

    nearest = np.empty(pts.shape[0], dtype=int)
    distances = np.empty(pts.shape[0], dtype=float)
    closest = np.empty((pts.shape[0], 3), dtype=float)
    step = max(1, int(chunk_size))

    for start in range(0, pts.shape[0], step):
        stop = min(start + step, pts.shape[0])
        p = pts[start:stop]
        rel = p[:, None, :] - a[None, :, :]
        t = np.sum(rel * u[None, :, :], axis=2) / denom[None, :]
        t = np.clip(t, 0.0, 1.0)
        c = a[None, :, :] + t[:, :, None] * u[None, :, :]
        d2 = np.sum((p[:, None, :] - c) ** 2, axis=2)
        idx = np.argmin(d2, axis=1)
        row = np.arange(stop - start)
        nearest[start:stop] = idx
        distances[start:stop] = np.sqrt(d2[row, idx])
        closest[start:stop] = c[row, idx]

    return {
        "nearest_segment": nearest,
        "distance": distances,
        "closest_point": closest,
    }


def _selector_for_deep_fault(
    deep_selectors: Any,
    deep_fault: Any,
    *,
    single_deep_fault: bool,
) -> Any:
    if deep_selectors is None:
        return None
    if isinstance(deep_selectors, Mapping):
        name = getattr(deep_fault, "name", None)
        if name in deep_selectors:
            return deep_selectors[name]
        selector_keys = {
            "patches",
            "patch_indices",
            "edge",
            "edges",
            "depth_range",
            "trace_range",
            "box",
            "lon_range",
            "lat_range",
            "x_range",
            "y_range",
        }
        if single_deep_fault and any(key in deep_selectors for key in selector_keys):
            return deep_selectors
        return None
    if single_deep_fault:
        return deep_selectors
    raise ValueError("deep_selectors must be a mapping by deep fault name when multiple deep faults are used")


def map_shallow_patches_to_deep_top_trace(
    shallow_fault: Any,
    deep_faults: Sequence[Any] | Any,
    *,
    shallow_selector: Any = None,
    deep_selectors: Any = None,
    coord_frame: str = "same_xy",
    component: str = "strikeslip",
    top_edge_policy: str = "infer",
    top_edge_tolerance: float = 1.0e-8,
    max_distance: float | None = None,
    chunk_size: int = 4096,
) -> dict[str, Any]:
    """Map selected shallow patches to nearest deep patch top-edge segments.

    Parameters
    ----------
    shallow_fault : object
        Shallow fault whose selected patch centers are mapped.
    deep_faults : object or sequence of objects
        Deep fault(s) providing loading-proxy slip parameters.
    shallow_selector : selector, optional
        Patch selector for shallow fault.  Defaults to ``{"edge": "bottom"}``.
        Use ``"all"`` to map every shallow patch, which is useful for field
        evaluation rather than bottom-edge constraints.
    deep_selectors : selector or mapping, optional
        Deep candidate patch selectors.  For multiple deep faults, pass a
        mapping by fault name.
    coord_frame : {"same_xy", "shallow_xy_from_lonlat"}, default "same_xy"
        Coordinate frame used for distance calculation.
    component : str, default "strikeslip"
        Slip component later used by constraints and field extraction.
    max_distance : float, optional
        Raise if any nearest mapping exceeds this 3-D distance.

    Returns
    -------
    dict
        Mapping table with shallow patch ids, matched deep fault names, matched
        deep patch ids and nearest distances.
    """
    if shallow_selector is None:
        shallow_selector = {"edge": "bottom"}
        allow_none_all = False
    elif isinstance(shallow_selector, str) and shallow_selector.lower().replace("-", "_") in (
        "all",
        "all_patches",
        "full_fault",
    ):
        shallow_selector = None
        allow_none_all = True
    else:
        allow_none_all = False
    component = normalize_slip_component(component)
    _ensure_edge_selector_ready(shallow_fault, shallow_selector)
    shallow_indices = select_patch_indices(
        shallow_fault,
        shallow_selector,
        allow_none_all=allow_none_all,
        unique=True,
        name="shallow_selector",
    )
    if shallow_indices.size == 0:
        raise ValueError("shallow_selector selected no patches")

    if isinstance(deep_faults, (str, bytes)):
        raise TypeError("deep_faults must be fault objects here; names are resolved by DeepSlipLoadingMixin")
    if not isinstance(deep_faults, Sequence):
        deep_fault_list = [deep_faults]
    else:
        deep_fault_list = list(deep_faults)
    if not deep_fault_list:
        raise ValueError("deep_faults cannot be empty")

    centers = get_patch_centers(shallow_fault, coord="xy")[shallow_indices]
    all_start: list[np.ndarray] = []
    all_end: list[np.ndarray] = []
    all_fault_names: list[str] = []
    all_patch_indices: list[int] = []
    all_ambiguous: list[bool] = []
    deep_candidate_counts: dict[str, int] = {}

    for deep_fault in deep_fault_list:
        selector = _selector_for_deep_fault(
            deep_selectors,
            deep_fault,
            single_deep_fault=(len(deep_fault_list) == 1),
        )
        _ensure_edge_selector_ready(deep_fault, selector)
        deep_indices = select_patch_indices(
            deep_fault,
            selector,
            allow_none_all=True,
            unique=True,
            name=f"deep_selector[{getattr(deep_fault, 'name', '<unnamed>')}]",
        )
        if deep_indices.size == 0:
            continue
        segments = get_patch_top_segments(
            deep_fault,
            deep_indices,
            coord_frame=coord_frame,
            reference_fault=shallow_fault,
            top_edge_policy=top_edge_policy,
            top_edge_tolerance=top_edge_tolerance,
        )
        name = str(getattr(deep_fault, "name", f"deep_{len(all_fault_names)}"))
        count = len(segments["patch_indices"])
        deep_candidate_counts[name] = count
        all_start.append(segments["segment_start"])
        all_end.append(segments["segment_end"])
        all_fault_names.extend([name] * count)
        all_patch_indices.extend(segments["patch_indices"].astype(int).tolist())
        all_ambiguous.extend(segments["ambiguous_top_edge"].astype(bool).tolist())

    if not all_start:
        raise ValueError("deep_selectors selected no candidate deep patches")

    seg_start = np.vstack(all_start)
    seg_end = np.vstack(all_end)
    nearest = point_to_segment_distance_3d(centers, seg_start, seg_end, chunk_size=chunk_size)
    idx = nearest["nearest_segment"]
    distances = nearest["distance"]
    if max_distance is not None:
        too_far = np.nonzero(distances > float(max_distance))[0]
        if too_far.size:
            examples = [
                {
                    "shallow_patch": int(shallow_indices[i]),
                    "distance": float(distances[i]),
                }
                for i in too_far[:10]
            ]
            raise ValueError(
                f"{too_far.size} shallow patches exceed max_distance={max_distance}: "
                f"{examples}"
            )

    deep_fault_names = np.asarray(all_fault_names, dtype=object)[idx]
    deep_patch_indices = np.asarray(all_patch_indices, dtype=int)[idx]
    ambiguous = np.asarray(all_ambiguous, dtype=bool)[idx]

    return {
        "model": "deep_slip_loading_proxy",
        "shallow_fault": getattr(shallow_fault, "name", None),
        "shallow_patch_indices": shallow_indices.astype(int),
        "deep_fault_names": deep_fault_names,
        "deep_patch_indices": deep_patch_indices,
        "mapping_distance": distances,
        "closest_points": nearest["closest_point"],
        "deep_segment_indices": idx.astype(int),
        "deep_candidate_counts": deep_candidate_counts,
        "component": component,
        "coord_frame": coord_frame,
        "shallow_selector": shallow_selector,
        "deep_selectors": deep_selectors,
        "top_edge_policy": top_edge_policy,
        "top_edge_tolerance": top_edge_tolerance,
        "ambiguous_deep_top_edge": ambiguous,
    }


def _linear_parameter_count(inversion: Any, n_total: int | None = None) -> int:
    if n_total is not None:
        return int(n_total)
    if hasattr(inversion, "lsq_parameters"):
        return int(getattr(inversion, "lsq_parameters"))
    if hasattr(inversion, "mpost") and getattr(inversion, "mpost") is not None:
        return int(len(inversion.mpost))
    if hasattr(inversion, "slip_positions") and inversion.slip_positions:
        offset = int(getattr(inversion, "linear_sample_start_position", 0) or 0)
        return max(int(end) for _, end in inversion.slip_positions.values()) - offset
    raise AttributeError("Cannot determine linear parameter count; pass n_total explicitly")


def _component_column(
    inversion: Any,
    fault: Any,
    patch_index: int,
    component: str,
    *,
    n_total: int,
) -> int:
    dummy_solution = np.zeros(n_total, dtype=float)
    start = _get_source_start(inversion, fault.name, dummy_solution)
    param_names = list(_get_source_param_names(inversion, fault))
    if component not in param_names:
        raise ValueError(f"Fault '{fault.name}' has no '{component}' component")
    comp_index = param_names.index(component)
    n_patches = len(fault.patch)
    column = start + comp_index * n_patches + int(patch_index)
    if column < 0 or column >= n_total:
        raise ValueError(
            f"Fault '{fault.name}' patch {patch_index} component '{component}' maps to "
            f"column {column}, outside [0, {n_total - 1}]"
        )
    return int(column)


def build_deep_slip_proxy_constraints(
    inversion: Any,
    mapping: Mapping[str, Any],
    *,
    state: str = "bottom_continuity",
    component: str | None = None,
    creep_ratio: float | None = None,
    locking: float | None = None,
    value: float | None = None,
    cap_ratio: float | None = None,
    motion_sense: str | int | float | None = None,
    motion_sign: str | int | float | None = None,
    n_total: int | None = None,
) -> dict[str, Any]:
    """Build linear constraints for a deep-slip loading proxy mapping.

    Parameters
    ----------
    inversion : object
        BLSE/Bayesian-like object with faults and source parameter positions.
    mapping : mapping
        Output from ``map_shallow_patches_to_deep_top_trace``.
    state : str, default "bottom_continuity"
        Constraint state.  ``bottom_continuity`` and ``full_creep`` build
        ``s_i - b_j = 0``.  ``full_locking`` builds ``s_i = 0``.
        ``cap`` builds same-direction cap inequalities.
    component : str, optional
        Slip component.  Defaults to ``mapping["component"]``.
    creep_ratio : float, optional
        Ratio ``c`` for ``s_i - c*b_j = 0``.
    locking : float, optional
        Locking ratio ``K`` for ``s_i - (1-K)*b_j = 0``.
    value : float, optional
        Fixed shallow slip value for ``fixed_shallow_slip``.
    cap_ratio : float, optional
        Upper ratio for cap inequalities.  Defaults to 1.0 when
        ``state="cap"``.
    motion_sense, motion_sign : str or number, optional
        Expected sign convention for cap constraints.  Dextral/right means -1;
        sinistral/left means +1.
    n_total : int, optional
        Linear parameter count.  Usually inferred from the inversion object.

    Returns
    -------
    dict
        ``equality`` and ``inequality`` sections.  Each section contains
        ``A``/``b`` arrays or is ``None``.
    """
    state_key = normalize_deep_slip_state(state)
    component_key = normalize_slip_component(component or mapping.get("component", "strikeslip"))
    n_params = _linear_parameter_count(inversion, n_total)

    from .interseismic_fields import get_fault_by_name

    shallow_fault = get_fault_by_name(inversion, mapping["shallow_fault"])
    shallow_indices = np.asarray(mapping["shallow_patch_indices"], dtype=int)
    deep_fault_names = np.asarray(mapping["deep_fault_names"], dtype=object)
    deep_patch_indices = np.asarray(mapping["deep_patch_indices"], dtype=int)

    if not (
        shallow_indices.shape[0] == deep_fault_names.shape[0] == deep_patch_indices.shape[0]
    ):
        raise ValueError("mapping arrays must have the same length")
    n_rows = int(shallow_indices.size)
    shallow_cols = np.asarray(
        [
            _component_column(inversion, shallow_fault, idx, component_key, n_total=n_params)
            for idx in shallow_indices
        ],
        dtype=int,
    )
    deep_fault_cache: dict[str, Any] = {}
    deep_cols: list[int] = []
    for fault_name, patch_idx in zip(deep_fault_names.tolist(), deep_patch_indices.tolist()):
        fault_name = str(fault_name)
        if fault_name not in deep_fault_cache:
            deep_fault_cache[fault_name] = get_fault_by_name(inversion, fault_name)
        deep_cols.append(
            _component_column(
                inversion,
                deep_fault_cache[fault_name],
                int(patch_idx),
                component_key,
                n_total=n_params,
            )
        )
    deep_cols_arr = np.asarray(deep_cols, dtype=int)

    equality = None
    formula = None
    if state_key != "cap" and state_key != "free":
        Aeq = np.zeros((n_rows, n_params), dtype=float)
        beq = np.zeros(n_rows, dtype=float)
        row = np.arange(n_rows, dtype=int)
        Aeq[row, shallow_cols] = 1.0

        if state_key in ("bottom_continuity", "full_creep"):
            Aeq[row, deep_cols_arr] -= 1.0
            formula = "s - b = 0"
        elif state_key == "full_locking":
            formula = "s = 0"
        elif state_key == "prescribed_creep_ratio":
            if creep_ratio is None:
                raise ValueError("state='prescribed_creep_ratio' requires creep_ratio=...")
            Aeq[row, deep_cols_arr] -= float(creep_ratio)
            formula = "s - creep_ratio * b = 0"
        elif state_key == "prescribed_locking":
            if locking is None:
                raise ValueError("state='prescribed_locking' requires locking=...")
            Aeq[row, deep_cols_arr] -= 1.0 - float(locking)
            formula = "s - (1 - locking) * b = 0"
        elif state_key == "fixed_shallow_slip":
            if value is None:
                raise ValueError("state='fixed_shallow_slip' requires value=...")
            beq[:] = float(value)
            formula = "s = value"
        else:
            raise ValueError(f"Unsupported deep-slip state '{state}'")
        equality = {"A": Aeq, "b": beq, "formula": formula}

    inequality = None
    if state_key == "cap" or cap_ratio is not None:
        ratio = 1.0 if cap_ratio is None else float(cap_ratio)
        if ratio < 0.0:
            raise ValueError("cap_ratio must be non-negative")
        sign = _motion_sign(motion_sign if motion_sign is not None else motion_sense)
        A = np.zeros((3 * n_rows, n_params), dtype=float)
        rhs = np.zeros(3 * n_rows, dtype=float)
        row = np.arange(n_rows, dtype=int)
        A[row, deep_cols_arr] = -sign
        A[row + n_rows, shallow_cols] = -sign
        A[row + 2 * n_rows, shallow_cols] = sign
        A[row + 2 * n_rows, deep_cols_arr] = -ratio * sign
        inequality = {
            "A": A,
            "b": rhs,
            "formula": "-sigma*b <= 0; -sigma*s <= 0; sigma*s - cap_ratio*sigma*b <= 0",
            "motion_sign": sign,
            "cap_ratio": ratio,
        }

    return {
        "state": state_key,
        "component": component_key,
        "n_pairs": n_rows,
        "n_total": n_params,
        "shallow_columns": shallow_cols,
        "deep_columns": deep_cols_arr,
        "equality": equality,
        "inequality": inequality,
        "metadata": {
            "model": "deep_slip_loading_proxy",
            "shallow_fault": mapping["shallow_fault"],
            "deep_faults": sorted({str(name) for name in deep_fault_names.tolist()}),
            "state": state_key,
            "component": component_key,
            "formula": formula,
        },
    }
