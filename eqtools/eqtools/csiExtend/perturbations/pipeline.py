"""
Perturbation Pipeline — composable stage-based geometry perturbation engine.

Internal module. All existing public method names, @track_mesh_update decorator,
PerturbationRegistry, and Bayesian config YAML remain unchanged.

Architecture:
    GeometryReference (frozen)
        -> GeometryState.from_ref()   (mutable working copy)
        -> [coordinate stages]        (OffsetStage / RotateStage / TranslateStage)
        -> MeshPolicy.apply()         (densify + mesh generation)
        -> materialize()              (write back to fault)
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
    """Mutable working copy of geometry, flowing through pipeline stages."""

    top: np.ndarray | None = None
    bottom: np.ndarray | None = None
    layers: list[np.ndarray] | None = None
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    dip_control_points: DipControlPoints | None = None
    densification: DensificationConfig | None = None
    meta: dict = field(default_factory=dict)
    dirty: set = field(default_factory=set)

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


@dataclass(frozen=True)
class Target:
    """Identifies which geometry component a stage operates on."""

    kind: str
    index: int | None = None
    indices: tuple | None = None

    def resolve(self, state: GeometryState) -> list[tuple[str, int | None, np.ndarray]]:
        """Return list of (label, index_or_none, array_ref) for this target."""
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
    @abstractmethod
    def select(self, coords: np.ndarray) -> np.ndarray:
        """Return boolean mask of shape (n,) for nodes to perturb."""
        ...


class AllNodes(NodeSelector):
    def select(self, coords):
        return np.ones(len(coords), dtype=bool)


class IndexNodes(NodeSelector):
    def __init__(self, indices):
        self.indices = indices

    def select(self, coords):
        mask = np.zeros(len(coords), dtype=bool)
        mask[list(self.indices)] = True
        return mask


class ExcludeNodes(NodeSelector):
    """Equivalent to current fixed_nodes semantics."""
    def __init__(self, indices):
        self.indices = indices

    def select(self, coords):
        mask = np.ones(len(coords), dtype=bool)
        mask[list(self.indices)] = False
        return mask


class MaskNodes(NodeSelector):
    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=bool)

    def select(self, coords):
        return self.mask


# ============================================================================
# 3. DirectionProvider
# ============================================================================

class DirectionProvider(ABC):
    @abstractmethod
    def compute(self, coords: np.ndarray, ctx: PipelineContext) -> np.ndarray:
        """Return (n, 2) or (n, 3) direction unit vectors."""
        ...


class StrikeNormalDirection(DirectionProvider):
    """Strike-normal direction. Delegates to calculate_perturb_direction()."""

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
    def compute(self, coords, ctx):
        n = len(coords)
        dirs = np.zeros((n, 3))
        dirs[:, 2] = 1.0
        return dirs


class FixedAzimuthDirection(DirectionProvider):
    def __init__(self, azimuth_deg: float):
        self.azimuth_rad = np.radians(azimuth_deg)

    def compute(self, coords, ctx):
        trend = np.pi / 2.0 - self.azimuth_rad
        d = np.array([np.cos(trend), np.sin(trend)])
        return np.tile(d, (len(coords), 1))


class CustomVectors(DirectionProvider):
    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=float)

    def compute(self, coords, ctx):
        return self.vectors


# ============================================================================
# 4. PipelineContext
# ============================================================================

@dataclass
class PipelineContext:
    """Read-only context passed to all stages."""
    fault: object
    ref: GeometryReference
    angle_unit: str = 'degrees'


# ============================================================================
# 5. Stage Protocol
# ============================================================================

class Stage(ABC):
    @abstractmethod
    def apply(self, state: GeometryState, ctx: PipelineContext) -> GeometryState:
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
    """Offset coordinates along a direction. Replaces perturb_coords_along_fixed_direction()."""

    target: Target
    nodes: NodeSelector
    direction: DirectionProvider
    values: np.ndarray

    def apply(self, state, ctx):
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
        return state


# ---- RotateStage -----------------------------------------------------------

@dataclass
class RotateStage(Stage):
    """Rotate coordinates around a pivot. Replaces perturb_coords_by_rotation()."""

    targets: list
    angle: float
    pivot: object = 'midpoint'
    pivot_source: Target = field(default_factory=lambda: Target("top"))
    pivot_is_utm: bool = True
    force_pivot_in_coords: bool = False
    pivot_key: str | None = None
    pivot_frame: str = "current"

    def apply(self, state, ctx):
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
        return state

    def _resolve_pivot(self, state, ctx):
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
    """Translate coordinates. Replaces perturb_coords_by_translation()."""

    targets: list
    dx: float
    dy: float

    def apply(self, state, ctx):
        delta = np.array([self.dx, self.dy])
        for target in self.targets:
            for label, idx, coords in target.resolve(state):
                coords[:, :2] += delta

        state.mark_dirty(*_collect_dirty_labels(self.targets, state))
        return state


# ---- DipGeneratorStage -----------------------------------------------------

@dataclass
class DipGeneratorStage(Stage):
    """Generate bottom coordinates from dip perturbation.

    Unlike OffsetStage/RotateStage/TranslateStage which *transform* existing
    coordinates, this stage *generates* new bottom from physics
    (top + dip + depth + strike).

    Workflow:
        1. Perturb dip control-point values
        2. Densify top in-state (DensificationConfig and/or discretization_interval)
        3. Interpolate dip onto densified top nodes
        4. Compute bottom = top + dip_vector * width
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
        from .dip_ops import (
            perturb_dip_values,
            interpolate_dip_onto_coords,
            generate_bottom_from_dips,
            augment_control_points_with_buffers,
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

        if self.buffer_nodes is not None and self.buffer_radius is not None:
            control_xy_dip = augment_control_points_with_buffers(
                control_xy_dip,
                buffer_nodes_lonlat=self.buffer_nodes,
                buffer_radius=self.buffer_radius,
                interpolation_axis=(
                    self.interpolation_axis if self.interpolation_axis != 'auto'
                    else 'x'
                ),
                top_coords_2d=state.top[:, :2],
                ll2xy=ctx.fault.ll2xy,
                xy2ll=ctx.fault.xy2ll,
            )

        interpolated_dip, strike = interpolate_dip_onto_coords(
            control_xy_dip, state.top,
            interpolation_axis=self.interpolation_axis,
        )

        state.bottom = generate_bottom_from_dips(
            state.top, interpolated_dip, strike,
            fault_depth=ctx.fault.depth,
            fault_top=ctx.fault.top,
            use_average_strike=self.use_average_strike,
            average_strike_source=self.average_strike_source,
            user_direction_angle=self.user_direction_angle,
            interpolation_axis=self.interpolation_axis,
        )

        state.meta['top_strike'] = strike
        state.meta['top_dip'] = interpolated_dip
        state.mark_dirty('top', 'bottom')
        return state


# ============================================================================
# 6. MeshPolicy
# ============================================================================

class MeshPolicy(ABC):
    @abstractmethod
    def apply(self, state: GeometryState, ctx: PipelineContext) -> GeometryState:
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
    """Skip mesh generation."""

    def apply(self, state, ctx):
        return state


class SimpleMeshPolicy(MeshPolicy):
    def __init__(self, disct_z=None, bias=None, min_dz=None, use_depth_only=True):
        self.disct_z = disct_z
        self.bias = bias
        self.min_dz = min_dz
        self.use_depth_only = use_depth_only

    def apply(self, state, ctx):
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
        return state


class MultiLayerMeshPolicy(MeshPolicy):
    def __init__(self, disct_z=None, bias=None):
        self.disct_z = disct_z
        self.bias = bias

    def apply(self, state, ctx):
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
        return state


# ============================================================================
# 7. Materialize
# ============================================================================

def materialize(state: GeometryState, ctx: PipelineContext):
    """Write pipeline results back to the fault object (dirty fields only)."""
    fault = ctx.fault

    if 'top' in state.dirty and state.top is not None:
        fault.set_coords(state.top, lonlat=False, coord_type='top')

    if 'bottom' in state.dirty and state.bottom is not None:
        fault.set_coords(state.bottom, lonlat=False, coord_type='bottom')

    if 'layers' in state.dirty and state.layers is not None:
        fault.set_coords(state.layers, lonlat=False, coord_type='layer')

    if 'vertices' in state.dirty and state.vertices is not None and state.faces is not None:
        fault.VertFace2csifault(state.vertices, state.faces)

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
