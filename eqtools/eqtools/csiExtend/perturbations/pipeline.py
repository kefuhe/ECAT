"""Composable, non-cumulative geometry perturbation pipeline.

Internal module. All existing public method names, @track_mesh_update decorator,
PerturbationRegistry, and Bayesian config YAML remain unchanged.

Each invocation starts from a new mutable copy of the frozen reference.  Stages
never use the previous proposal as their baseline, which keeps sample meaning
independent of evaluation order.  Only ``materialize`` writes the finished
candidate back to the fault.

Data flow::

    GeometryReference (frozen)
        -> GeometryState.from_ref()   (mutable working copy)
        -> [coordinate stages]        (OffsetStage / RotateStage / TranslateStage)
        -> MeshPolicy.apply()         (densify + mesh generation)
        -> materialize()              (write back to fault)

The pipeline classifies coordinate changes as ``rigid`` or ``deform``.  CSI's
mesh publisher remains the authority for topology comparison and derived-cache
invalidation; stages do not edit cache flags or infer remeshing themselves.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..bayesian_perturbation_base import GeometryReference, DensificationConfig, DipControlPoints
from .angle_utils import angles_to_radians, normalize_angle_unit


# ============================================================================
# 1. Data Structures
# ============================================================================

@dataclass
class GeometryState:
    """Mutable candidate state passed between pipeline stages.

    Coordinate fields are copies of ``GeometryReference`` arrays.  ``dirty``
    records which fields must be published; untouched fields are not written
    back to the fault.  ``mesh_change_kind`` records the strongest metric
    change seen by whole-mesh stages: rigid transforms preserve pairwise
    distances and areas, while deformation does not.  ``meta`` carries
    per-candidate generator outputs such as ``top_strike`` and ``top_dip``;
    those values are not part of the frozen reference.
    """

    top: np.ndarray | None = None
    bottom: np.ndarray | None = None
    layers: list[np.ndarray] | None = None
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    dip_control_points: DipControlPoints | None = None
    densification: DensificationConfig | None = None
    meta: dict = field(default_factory=dict)
    dirty: set = field(default_factory=set)
    mesh_change_kind: str = 'none'

    @classmethod
    def from_ref(cls, ref: GeometryReference) -> GeometryState:
        """Create mutable state from frozen reference. Copies all arrays."""
        return cls(
            top=ref.top_coords.copy() if ref.top_coords is not None else None,
            bottom=ref.bottom_coords.copy() if ref.bottom_coords is not None else None,
            layers=[l.copy() for l in ref.layers] if ref.layers else None,
            vertices=ref.vertices.copy() if ref.vertices is not None else None,
            faces=ref.faces.copy() if ref.faces is not None else None,
            dip_control_points=ref.dip_control_points,
            densification=ref.densification,
        )

    def mark_dirty(self, *fields):
        """Mark fields as modified by a stage."""
        self.dirty.update(fields)

    def mark_mesh_change(self, change_kind):
        """Accumulate the strongest metric change made by pipeline stages.

        ``rigid`` preserves distances and areas; ``deform`` does not.  Mesh
        topology replacement is detected centrally by ``VertFace2csifault``
        from the final faces rather than guessed by individual stages.
        """
        precedence = {'none': 0, 'rigid': 1, 'deform': 2}
        if change_kind not in precedence:
            raise ValueError(
                "change_kind must be 'none', 'rigid', or 'deform'; "
                f"got {change_kind!r}"
            )
        if precedence[change_kind] > precedence[self.mesh_change_kind]:
            self.mesh_change_kind = change_kind


@dataclass(frozen=True)
class Target:
    """Declarative address of a coordinate field in ``GeometryState``.

    ``kind`` is one of ``top``, ``bottom``, ``vertices``, ``layer`` or
    ``layers``.  A single ``layer`` requires ``index``; ``layers`` optionally
    accepts ``indices`` and otherwise resolves every intermediate layer.
    """

    kind: str
    index: int | None = None
    indices: tuple | None = None

    def resolve(self, state: GeometryState) -> list[tuple[str, int | None, np.ndarray]]:
        """Resolve to mutable array references in deterministic layer order.

        Returns tuples of ``(label, layer_index_or_none, array)``.  The arrays
        belong to the candidate state, not to ``GeometryReference``; modifying
        them is therefore safe inside a stage.
        """
        if self.kind == 'top':
            if state.top is None:
                raise ValueError("Target('top') but state.top is None.")
            return [('top', None, state.top)]

        if self.kind == 'bottom':
            if state.bottom is None:
                raise ValueError("Target('bottom') but state.bottom is None.")
            return [('bottom', None, state.bottom)]

        if self.kind == 'vertices':
            if state.vertices is None:
                raise ValueError("Target('vertices') but state.vertices is None.")
            return [('vertices', None, state.vertices)]

        if self.kind == 'layer':
            if state.layers is None:
                raise ValueError("Target('layer') but state.layers is None.")
            if self.index is None:
                raise ValueError("Target('layer') requires index.")
            if self.index >= len(state.layers):
                raise IndexError(
                    f"Target('layer', index={self.index}) out of range "
                    f"(state has {len(state.layers)} layers)."
                )
            return [('layer', self.index, state.layers[self.index])]

        if self.kind == 'layers':
            if state.layers is None:
                raise ValueError("Target('layers') but state.layers is None.")
            if self.indices is not None:
                return [
                    ('layer', i, state.layers[i]) for i in self.indices
                ]
            return [
                ('layer', i, state.layers[i]) for i in range(len(state.layers))
            ]

        raise ValueError(f"Unknown target kind: {self.kind!r}")


# ============================================================================
# 2. NodeSelector
# ============================================================================

class NodeSelector(ABC):
    """Strategy that selects coordinate rows affected by an offset stage."""

    @abstractmethod
    def select(self, coords: np.ndarray) -> np.ndarray:
        """Return boolean mask of shape (n,) for nodes to perturb."""
        ...


class AllNodes(NodeSelector):
    """Select every coordinate row."""

    def select(self, coords):
        return np.ones(len(coords), dtype=bool)


class IndexNodes(NodeSelector):
    """Select only the supplied NumPy-style row indices."""

    def __init__(self, indices):
        self.indices = indices

    def select(self, coords):
        mask = np.zeros(len(coords), dtype=bool)
        mask[list(self.indices)] = True
        return mask


class ExcludeNodes(NodeSelector):
    """Select all rows except ``indices`` (the public ``fixed_nodes`` rule)."""

    def __init__(self, indices):
        self.indices = indices

    def select(self, coords):
        mask = np.ones(len(coords), dtype=bool)
        mask[list(self.indices)] = False
        return mask


class MaskNodes(NodeSelector):
    """Select rows using a caller-supplied boolean mask."""

    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=bool)

    def select(self, coords):
        return self.mask


# ============================================================================
# 3. DirectionProvider
# ============================================================================

class DirectionProvider(ABC):
    """Strategy that returns per-node Cartesian unit direction vectors."""

    @abstractmethod
    def compute(self, coords: np.ndarray, ctx: PipelineContext) -> np.ndarray:
        """Return (n, 2) or (n, 3) direction unit vectors."""
        ...


class StrikeNormalDirection(DirectionProvider):
    """Right-hand strike-normal direction in the local projected frame.

    By default the direction is derived from the frozen top edge so a bottom
    or layer stage does not silently change direction after an earlier stage.
    ``source`` may select the target coordinates for methods that explicitly
    need candidate-local directions.
    """

    def __init__(self, use_average: bool = True,
                 average_direction=None, angle_unit: str = 'degrees',
                 source: str = 'reference_top'):
        self.use_average = use_average
        self.average_direction = average_direction
        self.angle_unit = angle_unit
        self.source = source

    def compute(self, coords, ctx):
        ref_coords = ctx.ref.top_coords if self.source == 'reference_top' else coords
        azimuths = ctx.fault.calculate_perturb_direction(
            ref_coords,
            angle_unit=self.angle_unit,
            use_average_strike=self.use_average,
            average_direction=self.average_direction,
        )
        trends = np.pi / 2.0 - azimuths
        return np.column_stack([np.cos(trends), np.sin(trends)])


class VerticalDirection(DirectionProvider):
    """Positive local z direction; depth sign interpretation stays with caller."""

    def compute(self, coords, ctx):
        n = len(coords)
        dirs = np.zeros((n, 3))
        dirs[:, 2] = 1.0
        return dirs


class FixedAzimuthDirection(DirectionProvider):
    """One geographic azimuth, broadcast as a local horizontal unit vector."""

    def __init__(self, azimuth_deg: float):
        self.azimuth_rad = np.radians(azimuth_deg)

    def compute(self, coords, ctx):
        trend = np.pi / 2.0 - self.azimuth_rad
        d = np.array([np.cos(trend), np.sin(trend)])
        return np.tile(d, (len(coords), 1))


class CustomVectors(DirectionProvider):
    """Return caller-provided direction vectors without normalization."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=float)

    def compute(self, coords, ctx):
        return self.vectors


# ============================================================================
# 4. PipelineContext
# ============================================================================

@dataclass
class PipelineContext:
    """Shared stage context: target fault, frozen reference, and angle unit.

    The dataclass is treated as read-only by convention.  Mutable candidate
    geometry belongs exclusively to ``GeometryState``.
    """
    fault: object
    ref: GeometryReference
    angle_unit: str = 'degrees'


# ============================================================================
# 5. Stage Protocol
# ============================================================================

class Stage(ABC):
    """One ordered candidate transformation with no direct fault write-back."""

    @abstractmethod
    def apply(self, state: GeometryState, ctx: PipelineContext) -> GeometryState:
        """Transform and return ``state``, recording dirty fields as needed."""
        ...


def _collect_dirty_labels(targets, state):
    """Collect dirty-field names from resolved targets."""
    labels = set()
    for t in targets:
        for label, idx, _ in t.resolve(state):
            labels.add('layers' if idx is not None else label)
    return labels


# ---- OffsetStage -----------------------------------------------------------

@dataclass
class OffsetStage(Stage):
    """Offset selected rows along per-node direction vectors.

    A scalar value broadcasts to all selected rows; otherwise the value count
    must equal the selected-row count.  Offsets are conservatively classified
    as deformation because node-dependent directions or fixed nodes can change
    distances even when a single scalar was supplied.
    """

    target: Target
    nodes: NodeSelector
    direction: DirectionProvider
    values: np.ndarray

    def apply(self, state, ctx):
        """Apply the offset to every array resolved by ``target``."""
        resolved = self.target.resolve(state)
        for label, idx, coords in resolved:
            mask = self.nodes.select(coords)
            n_movable = int(mask.sum())
            if n_movable == 0:
                continue

            dirs = self.direction.compute(coords, ctx)

            vals = np.asarray(self.values, dtype=float).ravel()
            if vals.size == 1:
                vals = np.full(n_movable, vals.item(), dtype=float)
            elif vals.size != n_movable:
                raise ValueError(
                    f"OffsetStage values must be scalar or match movable node "
                    f"count ({n_movable}); got {vals.size}."
                )

            if dirs.shape[1] == 2:
                coords[mask, :2] += dirs[mask] * vals[:, None]
            elif dirs.shape[1] == 3:
                coords[mask] += dirs[mask] * vals[:, None]

        state.mark_dirty(*_collect_dirty_labels([self.target], state))
        state.mark_mesh_change('deform')
        return state


# ---- RotateStage -----------------------------------------------------------

@dataclass
class RotateStage(Stage):
    """Apply one horizontal rigid rotation to one or more targets.

    The pivot can come from the current candidate or frozen reference.  A
    ``pivot_key`` reuses one resolved pivot across stages in the same pipeline;
    it never persists the pivot into the fault or across proposals.
    """

    targets: list
    angle: float
    pivot: object = 'midpoint'
    pivot_source: Target = field(default_factory=lambda: Target("top"))
    pivot_is_utm: bool = True
    force_pivot_in_coords: bool = False
    pivot_key: str | None = None
    pivot_frame: str = "current"

    def apply(self, state, ctx):
        """Rotate target x/y coordinates and mark a rigid metric change."""
        pivot = self._resolve_pivot(state, ctx)
        angle_rad = float(np.asarray(
            angles_to_radians(self.angle, ctx.angle_unit)
        ).ravel()[0])
        rotation = np.exp(1j * angle_rad)

        for target in self.targets:
            for label, idx, coords in target.resolve(state):
                rel = coords[:, :2] - pivot
                c = rel[:, 0] + 1j * rel[:, 1]
                c_rot = c * rotation
                coords[:, :2] = np.column_stack([c_rot.real, c_rot.imag]) + pivot

        state.mark_dirty(*_collect_dirty_labels(self.targets, state))
        state.mark_mesh_change('rigid')
        return state

    def _resolve_pivot(self, state, ctx):
        """Resolve named or explicit pivot coordinates in local projected km."""
        if self.pivot_frame not in ("current", "reference"):
            raise ValueError(
                f"pivot_frame must be 'current' or 'reference'; got {self.pivot_frame!r}"
            )

        if self.pivot_key is not None and self.pivot_key in state.meta:
            return state.meta[self.pivot_key]

        if isinstance(self.pivot, str):
            source_coords = self._get_source_coords(state, ctx)
            if self.pivot == 'start':
                pivot = source_coords[0, :2].copy()
            elif self.pivot == 'end':
                pivot = source_coords[-1, :2].copy()
            elif self.pivot == 'midpoint':
                pivot = np.mean(source_coords[:, :2], axis=0)
                if self.force_pivot_in_coords:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(source_coords[:, :2])
                    _, nearest = tree.query(pivot)
                    pivot = source_coords[nearest, :2].copy()
            else:
                raise ValueError(f"Unknown pivot type: {self.pivot}")
        else:
            pivot = np.asarray(self.pivot, dtype=float)[:2].copy()
            if not self.pivot_is_utm:
                x, y = ctx.fault.ll2xy(pivot[0], pivot[1])
                pivot = np.array([x, y], dtype=float)
            if self.force_pivot_in_coords:
                source_coords = self._get_source_coords(state, ctx)
                from scipy.spatial import cKDTree
                tree = cKDTree(source_coords[:, :2])
                _, nearest = tree.query(pivot)
                pivot = source_coords[nearest, :2].copy()

        if self.pivot_key is not None:
            state.meta[self.pivot_key] = pivot

        return pivot

    def _get_source_coords(self, state, ctx):
        """Return the current or reference coordinates used to define pivot."""
        if self.pivot_frame == "reference":
            return _resolve_ref_coords(self.pivot_source, ctx.ref)
        return self.pivot_source.resolve(state)[0][2]


def _resolve_ref_coords(target: Target, ref: GeometryReference) -> np.ndarray:
    """Resolve a Target to coordinates from the frozen GeometryReference (zero-copy)."""
    if target.kind == "top":
        return ref.top_coords
    if target.kind == "bottom":
        return ref.bottom_coords
    if target.kind == "layer":
        if target.index is None:
            raise ValueError("Target('layer') requires index to resolve from reference.")
        return ref.layers[target.index]
    if target.kind == "layers":
        raise ValueError(
            "Target('layers') cannot be used as pivot_source; "
            "use Target('layer', index=i) for a specific layer."
        )
    raise ValueError(f"Cannot resolve pivot_source {target.kind!r} from reference.")


# ---- TranslateStage --------------------------------------------------------

@dataclass
class TranslateStage(Stage):
    """Apply one rigid horizontal translation to one or more targets."""

    targets: list
    dx: float
    dy: float

    def apply(self, state, ctx):
        """Add ``(dx, dy)`` in km and mark a rigid metric change."""
        delta = np.array([self.dx, self.dy])
        for target in self.targets:
            for label, idx, coords in target.resolve(state):
                coords[:, :2] += delta

        state.mark_dirty(*_collect_dirty_labels(self.targets, state))
        state.mark_mesh_change('rigid')
        return state


# ---- DipGeneratorStage -----------------------------------------------------

@dataclass
class DipGeneratorStage(Stage):
    """Generate bottom coordinates from dip perturbation.

    Unlike OffsetStage/RotateStage/TranslateStage which *transform* existing
    coordinates, this stage *generates* new bottom from physics
    (top + dip + depth + strike).

    Workflow:
        1. Convert equivalent reference dips to continuous 0--180 coordinates
           and add perturbations there
        2. Densify top in-state (DensificationConfig and/or discretization_interval)
        3. Interpolate dip onto densified top nodes
        4. Compute bottom = top + dip_vector * width

    ``interpolation_axis`` accepts ``'auto'``, ``'x'``, ``'y'``, or
    ``'arc_length'``. The first three operate on fault-local projected x/y;
    ``'auto'`` PCA-selects one projected axis. ``'arc_length'`` projects control
    points onto the current top-edge polyline and interpolates along cumulative
    distance, which avoids x/y ordering ambiguity on curved traces. Buffer
    augmentation requires a resolved x or y axis and is rejected for
    ``'arc_length'`` before the candidate state is materialized back onto the
    fault object.

    ``top_strike`` and ``top_dip`` written to ``state.meta`` are reference-node
    values used to generate the bottom edge.  In particular, ``top_dip`` is the
    signed side representation returned by interpolation.  They are not the
    canonical strike/dip of a generated CSI patch; obtain final scientific
    geometry from patch vertices and ``getpatchgeometry()``.
    """

    dip_control_points: DipControlPoints
    perturbations: np.ndarray
    fixed_nodes: list | None = None
    angle_unit: str = 'degrees'
    densify_top: bool = True
    discretization_interval: float | None = None
    interpolation_axis: str = 'auto'
    buffer_nodes: np.ndarray | None = None
    buffer_radius: float | None = None
    use_average_strike: bool = False
    average_strike_source: str = 'pca'
    user_direction_angle: float | None = None

    def _densify_top_only(self, state):
        """Densify top using DensificationConfig (mirrors densify_edges(top_only=True))."""
        cfg = state.densification
        if cfg is None or not cfg.enabled:
            return

        from ..geom_ops import discretize_coords

        if state.top is None:
            return

        n_before = state.top.shape[0]
        if cfg.num_segments is not None and n_before >= cfg.num_segments:
            return
        if cfg.interval is not None:
            dx = np.diff(state.top[:, 0])
            dy = np.diff(state.top[:, 1])
            arc_length = np.sum(np.sqrt(dx * dx + dy * dy))
            target_n = max(2, int(np.floor(arc_length / cfg.interval)))
            if n_before >= target_n:
                return

        kw = {}
        if cfg.num_segments is not None:
            kw['num_segments'] = cfg.num_segments
        elif cfg.interval is not None:
            kw['every'] = cfg.interval
        state.top = discretize_coords(state.top, **kw)
        state.mark_dirty('top')

    def apply(self, state, ctx):
        """Perturb controls, interpolate dip, and regenerate candidate bottom.

        The calculation operates entirely on the mutable candidate state.
        Control-point values remain immutable, and generated strike/dip arrays
        are stored in ``state.meta`` for publication after all stages succeed.
        """
        from .dip_ops import (
            perturb_dip_values,
            interpolate_dip_onto_coords,
            generate_bottom_from_dips,
            augment_control_points_with_buffers,
            determine_interpolation_axis,
        )
        from ..geom_ops import discretize_coords

        dcp = self.dip_control_points
        perturbed_dips = perturb_dip_values(
            dcp.dip, self.perturbations,
            fixed_nodes=self.fixed_nodes,
            angle_unit=self.angle_unit,
        )

        if self.densify_top:
            self._densify_top_only(state)

        if self.discretization_interval is not None:
            state.top = discretize_coords(state.top, every=self.discretization_interval)
            state.mark_dirty('top')

        control_xy_dip = np.column_stack([dcp.x, dcp.y, perturbed_dips])
        valid_axes = {'auto', 'x', 'y', 'arc_length'}
        if self.interpolation_axis not in valid_axes:
            raise ValueError(
                "interpolation_axis must be one of 'auto', 'x', 'y', or "
                f"'arc_length'; got {self.interpolation_axis!r}"
            )
        resolved_axis = self.interpolation_axis
        if resolved_axis == 'auto':
            resolved_axis = determine_interpolation_axis(
                state.top[:, 0],
                state.top[:, 1],
            )
        if (
            self.buffer_nodes is not None
            and self.buffer_radius is not None
            and resolved_axis == 'arc_length'
        ):
            raise ValueError("buffer augmentation does not support arc_length")

        if self.buffer_nodes is not None and self.buffer_radius is not None:
            control_xy_dip = augment_control_points_with_buffers(
                control_xy_dip,
                buffer_nodes_lonlat=self.buffer_nodes,
                buffer_radius=self.buffer_radius,
                interpolation_axis=resolved_axis,
                top_coords_2d=state.top[:, :2],
                ll2xy=ctx.fault.ll2xy,
                xy2ll=ctx.fault.xy2ll,
            )

        interpolated_dip, strike = interpolate_dip_onto_coords(
            control_xy_dip, state.top,
            interpolation_axis=resolved_axis,
        )

        state.bottom = generate_bottom_from_dips(
            state.top, interpolated_dip, strike,
            fault_depth=ctx.fault.depth,
            fault_top=ctx.fault.top,
            use_average_strike=self.use_average_strike,
            average_strike_source=self.average_strike_source,
            user_direction_angle=self.user_direction_angle,
            interpolation_axis=resolved_axis,
        )

        # Preserve the generator's reference-node convention.  Negative dip is
        # converted to strike+180/abs(dip) only in the local bottom-generation
        # calculation; final patch geometry is derived later from mesh vertices.
        state.meta['top_strike'] = strike
        state.meta['top_dip'] = interpolated_dip
        state.mark_dirty('top', 'bottom')
        state.mark_mesh_change('deform')
        return state


# ============================================================================
# 6. MeshPolicy
# ============================================================================

class MeshPolicy(ABC):
    """Final policy that either skips or materializes candidate mesh arrays."""

    @abstractmethod
    def apply(self, state: GeometryState, ctx: PipelineContext) -> GeometryState:
        """Return state after the policy-specific mesh action."""
        ...

    def _densify(self, state, ctx):
        """Densify edges. Top determines point count; bottom/layers aligned.

        Replicates PerturbationBase.densify_edges() semantics:
        - geom_ops.discretize_coords() returns a single ndarray.
        - Top densified first using config (num_segments or interval).
        - Bottom and layers aligned to num_segments=n_dense.
        """
        cfg = state.densification
        if cfg is None or not cfg.enabled:
            return

        from ..geom_ops import discretize_coords

        if state.top is None:
            return

        n_before = state.top.shape[0]
        if cfg.num_segments is not None and n_before >= cfg.num_segments:
            return
        if cfg.interval is not None:
            dx = np.diff(state.top[:, 0])
            dy = np.diff(state.top[:, 1])
            arc_length = np.sum(np.sqrt(dx * dx + dy * dy))
            target_n = max(2, int(np.floor(arc_length / cfg.interval)))
            if n_before >= target_n:
                return

        kw = {}
        if cfg.num_segments is not None:
            kw['num_segments'] = cfg.num_segments
        elif cfg.interval is not None:
            kw['every'] = cfg.interval
        state.top = discretize_coords(state.top, **kw)
        n_dense = state.top.shape[0]

        if state.bottom is not None and state.bottom.shape[0] != n_dense:
            state.bottom = discretize_coords(state.bottom, num_segments=n_dense)
        if state.layers:
            for i, layer in enumerate(state.layers):
                if layer is not None and layer.shape[0] != n_dense:
                    state.layers[i] = discretize_coords(layer, num_segments=n_dense)

        state.mark_dirty('top', 'bottom', 'layers')

    def _record_mesh_params(self, ctx, method, **params):
        """Record mesh params via the unified record_mesh_call pathway."""
        if not hasattr(ctx.fault, 'record_mesh_call'):
            raise TypeError(
                f"{ctx.fault.__class__.__name__} missing record_mesh_call; "
                "pipeline requires PerturbationBase"
            )
        ctx.fault.record_mesh_call(method, params, source='pipeline')


class NoMeshPolicy(MeshPolicy):
    """Publish coordinate changes only; leave mesh generation to the caller."""

    def apply(self, state, ctx):
        """Return the coordinate candidate unchanged."""
        return state


class SimpleMeshPolicy(MeshPolicy):
    """Build a simple triangular mesh from candidate top and bottom edges."""

    def __init__(self, disct_z=None, bias=None, min_dz=None, use_depth_only=True):
        self.disct_z = disct_z
        self.bias = bias
        self.min_dz = min_dz
        self.use_depth_only = use_depth_only

    def apply(self, state, ctx):
        """Densify aligned edges, generate vertices/faces, and mark them dirty."""
        self._densify(state, ctx)
        self._record_mesh_params(
            ctx, 'generate_simple_mesh',
            disct_z=self.disct_z, bias=self.bias,
            min_dz=self.min_dz, use_depth_only=self.use_depth_only,
        )
        mg = ctx.fault.mesh_generator
        mg.set_coordinates(state.top, state.bottom)
        vertices, faces = mg.generate_simple_mesh(
            self.disct_z, self.bias, self.min_dz, self.use_depth_only,
        )
        state.vertices = vertices
        state.faces = faces
        state.mark_dirty('vertices', 'faces')
        state.mark_mesh_change('deform')
        return state


class MultiLayerMeshPolicy(MeshPolicy):
    """Build a triangular mesh through candidate intermediate layers."""

    def __init__(self, disct_z=None, bias=None):
        self.disct_z = disct_z
        self.bias = bias

    def apply(self, state, ctx):
        """Densify aligned layers and generate the multilayer mesh pair."""
        self._densify(state, ctx)
        self._record_mesh_params(
            ctx, 'generate_simple_multilayer_mesh',
            disct_z=self.disct_z, bias=self.bias,
        )
        mg = ctx.fault.mesh_generator
        mg.set_coordinates(state.top, state.bottom)
        vertices, faces = mg.generate_multilayer_mesh(
            state.layers, self.disct_z, self.bias,
        )
        state.vertices = vertices
        state.faces = faces
        state.mark_dirty('vertices', 'faces')
        state.mark_mesh_change('deform')
        return state


# ============================================================================
# 7. Materialize
# ============================================================================

def materialize(state: GeometryState, ctx: PipelineContext):
    """Write pipeline results back to the fault object (dirty fields only).

    ``top_strike``/``top_dip`` remain reference-node generator metadata.  This
    function deliberately does not relabel them as canonical patch geometry.

    Coordinate setters keep projected and lon/lat views synchronized.  A dirty
    whole mesh must provide vertices and faces together and is published only
    through ``VertFace2csifault`` so CSI owns patch reconstruction, topology
    comparison, and cache invalidation.
    """
    fault = ctx.fault

    if 'top' in state.dirty and state.top is not None:
        fault.set_coords(state.top, lonlat=False, coord_type='top')

    if 'bottom' in state.dirty and state.bottom is not None:
        fault.set_coords(state.bottom, lonlat=False, coord_type='bottom')

    if 'layers' in state.dirty and state.layers is not None:
        fault.set_coords(state.layers, lonlat=False, coord_type='layer')

    if 'vertices' in state.dirty:
        if state.vertices is None or state.faces is None:
            raise ValueError(
                "A pipeline modified whole-mesh vertices without a complete "
                "candidate vertices/faces pair. Capture the final mesh in "
                "GeometryReference before using whole-mesh stages."
            )
        # Coordinate stages describe metric change; the mesh publisher owns
        # topology comparison and all cache invalidation.  Unknown/custom
        # vertex stages fall back to deform, which is scientifically safe.
        change_kind = (
            state.mesh_change_kind
            if state.mesh_change_kind != 'none'
            else 'deform'
        )
        fault.VertFace2csifault(
            state.vertices, state.faces, change_kind=change_kind,
        )

    if 'top_strike' in state.meta:
        fault.top_strike = state.meta['top_strike']
    if 'top_dip' in state.meta:
        fault.top_dip = state.meta['top_dip']


# ============================================================================
# 8. run_pipeline
# ============================================================================

def run_pipeline(
    fault,
    stages: list[Stage],
    mesh_policy: MeshPolicy | None = None,
    angle_unit: str = 'degrees',
) -> GeometryState:
    """Execute a perturbation pipeline.

    Parameters
    ----------
    fault : BayesianAdaptiveTriangularPatches
        The fault object (provides geometry_ref, coordinate transforms, mesh).
    stages : list[Stage]
        Ordered list of coordinate perturbation stages.
    mesh_policy : MeshPolicy or None
        If None, uses NoMeshPolicy (no mesh generation).
    angle_unit : str
        Angle unit for stages that need it ('degrees' or 'radians').

    Returns
    -------
    GeometryState
        Final state after all stages + mesh + materialize.

    Notes
    -----
    A new state is copied from ``fault.geometry_ref`` on every call.  Stage
    order matters within this call, but calls do not accumulate on the
    previously materialized candidate.  The frozen reference is never updated
    by this function.
    """
    ref = fault.geometry_ref
    if ref is None:
        raise ValueError("fault.geometry_ref is not set. Call snapshot() first.")

    ctx = PipelineContext(fault=fault, ref=ref, angle_unit=angle_unit)
    state = GeometryState.from_ref(ref)

    for stage in stages:
        state = stage.apply(state, ctx)

    if mesh_policy is None:
        mesh_policy = NoMeshPolicy()
    state = mesh_policy.apply(state, ctx)

    materialize(state, ctx)

    return state
