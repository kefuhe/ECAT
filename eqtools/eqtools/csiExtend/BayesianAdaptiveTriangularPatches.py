'''
Added by kfhe at 5/5/2024
Object :
    * Perturbing fault for bayesian inversion
    * The perturbation methods are defined via Mixin classes and assembled here.

Architecture (v2 - Mixin):
    BayesianAdaptiveTriangularPatches inherits from Mixins + BayesianTriFaultBase.
    BayesianTriFaultBase inherits AdaptiveLayeredDipTriangularPatches + PerturbationBase.
    Each Mixin provides one category of perturbation methods.
'''

import inspect
import warnings

import numpy as np

# import self-defined Mixin classes
from .perturbations import (
    DipPerturbationMixin,
    DirectionPerturbationMixin,
    RotationPerturbationMixin,
    TranslationPerturbationMixin,
    CompositePerturbationMixin,
    EndpointDuttaPerturbationMixin,
)

from .bayesian_perturbation_base import (
    BayesianTriFaultBase,
    GeometryReference,
    DensificationConfig,
)
from .dip_profile import (
    DipProfileSpec,
    build_dip_profile_spec,
    transform_dip_profile_coordinates,
)
from .MeshGenerator import MeshGenerator
from . import mesh_registry
from .geom_ops import resample_boundaries_by_normalized_arclength


class BayesianAdaptiveTriangularPatches(
    DipPerturbationMixin,
    DirectionPerturbationMixin,
    RotationPerturbationMixin,
    TranslationPerturbationMixin,
    CompositePerturbationMixin,
    EndpointDuttaPerturbationMixin,
    BayesianTriFaultBase,
):
    """Triangular fault with immutable-reference Bayesian perturbations.

    The class is the public assembly point for four distinct responsibilities:

    1. build or import the current top/bottom/layer geometry;
    2. freeze the zero-perturbation state in :class:`GeometryReference`;
    3. apply one registered perturbation from that frozen state; and
    4. materialize or deform a mesh when the selected method requires it.

    Candidate perturbations are intentionally non-cumulative: each call reads
    ``geometry_ref`` rather than the previous candidate.  A new ``snapshot``
    therefore starts a new baseline and must not be taken inside an SMC/MCMC
    proposal loop.
    """

    def __init__(self, name: str, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True, role='standalone', shared_info=None, **kwargs):
        BayesianTriFaultBase.__init__(self, name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=verbose, 
                                      role=role, shared_info=shared_info, **kwargs)

    #--------------------------------------Snapshot & Reference Coordinate Setup--------------------------------------#
    def snapshot(self, capture_vertices=True, capture_layers=True):
        """Create an immutable GeometryReference from the current fault state.

        Parameters:
            capture_vertices (bool): Include current Vertices/Faces if available.
            capture_layers (bool): Include current layers if available.

        After this call, ``self.geometry_ref`` holds a frozen baseline that all
        perturbation methods will read from.

        Note:
            If ``geometry_ref`` already exists, its ``dip_profile`` and
            ``densification`` config are **preserved** into the new snapshot.
            Re-calling ``snapshot()`` (e.g. via ``rebuild_simple_mesh``) will
            not lose previously configured dip or densification settings.
        """
        if not hasattr(self, 'top_coords') or self.top_coords is None:
            raise ValueError("top_coords not set. Build geometry first.")
        if not hasattr(self, 'bottom_coords') or self.bottom_coords is None:
            raise ValueError("bottom_coords not set. Build geometry first.")

        layers = None
        if capture_layers and hasattr(self, 'layers') and self.layers is not None:
            layers = [layer.copy() for layer in self.layers]

        vertices = None
        faces = None
        if capture_vertices:
            has_vertices = hasattr(self, 'Vertices') and self.Vertices is not None
            has_faces = hasattr(self, 'Faces') and self.Faces is not None
            if has_vertices != has_faces:
                raise ValueError(
                    "snapshot(capture_vertices=True) requires current "
                    "Vertices and Faces to be present together. The fault "
                    "currently contains only one half of the mesh pair."
                )
            if has_vertices:
                vertices = self.Vertices.copy()
                faces = self.Faces.copy()
            else:
                warnings.warn(
                    "snapshot(capture_vertices=True) called without a current "
                    "mesh. The reference was created without vertices/faces; "
                    "build the mesh first if a whole-mesh perturbation will be "
                    "used.",
                    stacklevel=2,
                )

        # Preserve the entire scientific profile as one unit.  Splitting its
        # controls, roles, interpolation axis and transitions across snapshots
        # would permit an internally inconsistent candidate baseline.
        dip_profile = None
        if self.geometry_ref is not None:
            dip_profile = self.geometry_ref.dip_profile

        # Preserve existing densification config if already attached
        densification = None
        if self.geometry_ref is not None and self.geometry_ref.densification is not None:
            densification = self.geometry_ref.densification

        self.geometry_ref = GeometryReference(
            top_coords=self.top_coords.copy(),
            bottom_coords=self.bottom_coords.copy(),
            layers=layers,
            vertices=vertices,
            faces=faces,
            dip_profile=dip_profile,
            densification=densification,
        )
        self._mesh_reference_validation_key = None
        # The captured current boundaries are, by definition, the materialized
        # zero-perturbation state for this new immutable reference.
        self._materialized_geometry_ref = self.geometry_ref
        self._materialized_trace_reference = self.geometry_ref
        self._materialized_trace_top_coords = self.top_coords.copy()

        # Keep legacy attributes in sync (read-only views into the frozen ref)

        if getattr(self, 'verbose', False) and not getattr(self, '_geometry_summary_printed', False):
            self.geometry_summary()
            self._geometry_summary_printed = True

        return self.geometry_ref

    # --- Legacy setters (delegate to snapshot) --------------------------------
    def set_top_coords_ref(self, top_coords_ref=None):
        """Legacy entry point — delegates to ``snapshot(capture_vertices=False)``.

        If *top_coords_ref* is provided, ``self.top_coords`` is replaced
        before snapshotting so the ref captures the caller's value.
        """
        if top_coords_ref is not None:
            self.top_coords = np.asarray(top_coords_ref)
        self.snapshot(capture_vertices=False)

    def set_bottom_coords_ref(self, bottom_coords_ref=None):
        """Legacy entry point — delegates to ``snapshot(capture_vertices=False)``.

        If *bottom_coords_ref* is provided, ``self.bottom_coords`` is replaced
        before snapshotting so the ref captures the caller's value.
        """
        if bottom_coords_ref is not None:
            self.bottom_coords = np.asarray(bottom_coords_ref)
        self.snapshot(capture_vertices=False)

    # --- Along-strike dip-profile setup (unified via geometry_ref) ------------
    def _ensure_geometry_ref_for_dip_profile(self):
        """Create the minimal reference needed to attach a dip profile.

        When no reference exists, the current top is captured without requiring
        a generated bottom or mesh.  When a reference already exists, its top
        remains authoritative; changing the profile never promotes a current
        candidate top implicitly.  Use :meth:`snapshot` only when deliberately
        promoting a complete independent geometry baseline.
        """
        if self.geometry_ref is not None:
            return
        top = getattr(self, "top_coords", None)
        bottom = getattr(self, "bottom_coords", None)
        self.geometry_ref = GeometryReference(
            top_coords=top.copy() if top is not None else None,
            bottom_coords=bottom.copy() if bottom is not None else None,
        )

    def set_dip_profile(
        self,
        sampled_controls,
        fixed_controls=None,
        *,
        interpolation_axis="auto",
        transition_zones=None,
        perturbation_groups=None,
        is_utm=False,
    ):
        """Freeze an along-strike dip profile for candidate generation.

        If no :attr:`geometry_ref` exists, the current top is captured as the
        authoritative generator reference.  If one already exists, its frozen
        top is retained and only the profile contract is replaced.  The setter
        never promotes a previously materialized candidate top implicitly.
        Each candidate regenerates bottom from the frozen top, profile, and
        density policy before any later rigid transform.  Do not snapshot the
        zero-perturbation output back into this generator reference.

        Parameters
        ----------
        sampled_controls, fixed_controls : sequence
            Rows are ``[lon, lat, reference_dip]`` by default or
            ``[x_km, y_km, reference_dip]`` when ``is_utm=True``.  Only the
            sampled group consumes candidate values.  Individual controls may
            instead use ``{'s_km': d, 'dip': angle}`` from the ordered
            reference-top start or ``{'s_from_end_km': d, 'dip': angle}``
            from its end.  Declaration order defines sample-vector order;
            spatial ordering is resolved later.
        interpolation_axis : {'auto', 'x', 'y', 'arc_length'}
            One-dimensional coordinate evaluated *after* every position is
            projected onto the top edge. ``arc_length`` is recommended for a
            curved or x/y-nonmonotonic trace.
        transition_zones : sequence of mappings, optional
            Use ``{'center': [c1, c2], 'half_width': value}`` (symmetric), an
            asymmetric ``half_width`` mapping with ``lower``/``upper``, or
             ``{'endpoints': [[c1, c2], [c1, c2]]}``.  Centre form accepts
             ``metric='axis'`` (default) or ``metric='euclidean'``.  Each
             centre or endpoint may use ``{'s_km': d}`` or
             ``{'s_from_end_km': d}`` instead of a coordinate pair.  Each zone
             accepts ``shape='linear'`` (default) or the opt-in cubic Hermite
             blend ``shape='smoothstep'``.
        perturbation_groups : sequence of str, optional
            Labels aligned with ``sampled_controls`` declaration order. Equal
            labels make those controls share one additive perturbation. The
            independent parameter order is the first occurrence of each label.
            Omission preserves the established scalar-broadcast or one-value-
            per-sampled-control behavior.
        is_utm : bool, default False
            Whether all declared position coordinates are fault-local x/y in
            kilometres.  The frozen profile itself is stored canonically in
            longitude/latitude.

        Returns
        -------
        DipProfileSpec
            Immutable profile attached to :attr:`geometry_ref`.
        """
        self._ensure_geometry_ref_for_dip_profile()
        profile = build_dip_profile_spec(
            sampled_controls,
            fixed_controls,
            interpolation_axis=interpolation_axis,
            transition_zones=transition_zones,
            perturbation_groups=perturbation_groups,
            reference_top_xy=self.geometry_ref.top_coords,
            xy_to_declaration_frame=None if is_utm else self.xy2ll,
        )
        if is_utm:
            profile = transform_dip_profile_coordinates(profile, self.xy2ll)
        elif (
            np.any(np.abs(profile.controls.x) > 360.0)
            or np.any(np.abs(profile.controls.y) > 90.0)
        ):
            raise ValueError(
                "dip-profile coordinates exceed lon/lat bounds; set "
                "is_utm=True for fault-local projected coordinates"
            )
        new_reference = self.geometry_ref.with_dip_profile(profile)
        # The top is unchanged, but the current bottom has not yet been
        # regenerated from this newly frozen profile contract.
        self._adopt_geometry_reference(
            new_reference,
            boundary_contract_changed=True,
        )
        return profile

    def analyze_reference_top_curvature(
        self,
        *,
        spacing_km=0.5,
        smoothing_method="savgol",
        smoothing_km=2.0,
        polyorder=3,
        min_prominence_ratio=0.05,
    ):
        """Run a read-only curvature preflight on the frozen reference top.

        This setup-time helper does not update ``top_coords``, ``bottom_coords``,
        mesh state, or caches.  Its result can suggest explicit along-top
        endpoints for the existing ``transition_zones`` protocol.  A dip
        profile may first be declared with ``transition_zones=None`` to create
        the minimal reference needed before bottom generation.

        Parameters
        ----------
        spacing_km : float, default 0.5
            Target uniform arc-length spacing used only by the diagnostic.
        smoothing_method : {'savgol', 'none'}, default 'savgol'
            Coordinate smoothing used before curvature differentiation.
        smoothing_km : float or None, default 2.0
            Physical Savitzky-Golay window length in kilometres.  It is
            ignored when ``smoothing_method='none'``.
        polyorder : int, default 3
            Savitzky-Golay polynomial order; it must be at least two.
        min_prominence_ratio : float, default 0.05
            Candidate-peak prominence divided by the largest absolute
            curvature.  This filters diagnostic noise and does not define the
            final transition width.

        Returns
        -------
        TopCurvatureAnalysis
            Immutable analysis tied to the current ordered reference top.
        """
        if self.geometry_ref is None or self.geometry_ref.top_coords is None:
            raise ValueError(
                "reference top not set. Declare the dip profile or snapshot "
                "the geometry before curvature analysis."
            )
        from .dip_transition_analysis import analyze_top_curvature

        return analyze_top_curvature(
            self.geometry_ref.top_coords,
            spacing_km=spacing_km,
            smoothing_method=smoothing_method,
            smoothing_km=smoothing_km,
            polyorder=polyorder,
            min_prominence_ratio=min_prominence_ratio,
            xy_to_lonlat=self.xy2ll,
        )

    def refresh_geometry_baseline(self):
        """Deliberately re-snapshot current coordinates as a new baseline.

        This advanced lifecycle operation preserves the declared dip profile
        and density policy but replaces their frozen top/bottom reference with
        the current geometry.  It is not part of ordinary dip-profile setup and
        must never be called inside a candidate loop.
        """
        if self.geometry_ref is None or self.geometry_ref.dip_profile is None:
            raise ValueError(
                "No existing geometry_ref with a dip profile to preserve. "
                "Declare the dip profile before deliberately refreshing its "
                "complete top/bottom baseline."
            )
        self.snapshot()

    # --- Densification configuration -------------------------------------------
    def set_densification(self, num_segments=None, interval=None, enabled=True):
        """Configure automatic coordinate densification for mesh/physics consumers.

        Call after either ``set_dip_profile()`` for a generator-owned profile
        or ``snapshot()`` for an independent-boundary geometry.  The immutable
        policy is stored in ``geometry_ref`` and survives deliberate snapshots.

        Parameters
        ----------
        num_segments : int, optional
            Historical name for the target boundary-node count. The dip path
            will not discard original trace vertices or required profile-event
            nodes merely to force this count.
        interval : float, optional
            Target spacing (km) between densified points.
        enabled : bool
            Set False to disable without removing config.
        """
        if self.geometry_ref is None:
            raise ValueError(
                "geometry_ref not set. Call set_dip_profile() for a generated "
                "dip profile, or snapshot() for independent boundaries."
            )
        cfg = DensificationConfig(num_segments=num_segments, interval=interval, enabled=enabled)
        previous_reference = self.geometry_ref
        previous_cfg = previous_reference.densification
        if previous_cfg == cfg:
            # Reapplying the same immutable policy is a true no-op.  In
            # particular, config normalization must not stale a mapping that
            # was already prepared from this exact contract.
            return

        new_reference = previous_reference.with_densification(cfg)
        dormant_change = (
            not cfg.enabled
            and (previous_cfg is None or not previous_cfg.enabled)
        )
        self._adopt_geometry_reference(
            new_reference,
            boundary_contract_changed=not dormant_change,
        )
        if dormant_change:
            # Replacing one disabled/dormant declaration cannot change the
            # materialized boundaries; no replay or user-facing message is
            # necessary.
            return

        if getattr(self, 'verbose', False):
            n_ctrl = self.geometry_ref.top_coords.shape[0] if self.geometry_ref.top_coords is not None else '?'
            mode = f"num_segments={num_segments}" if num_segments else f"interval={interval} km"
            print(f"[{self.name}] DensificationConfig: {mode}")
            print(f"  Perturbation: {n_ctrl} sparse control points -> Physics/mesh: dense points")

    #--------------------------------------Mesh Generation--------------------------------------#
    def _build_simple_mesh(self, top_coords, bottom_coords, disct_z, bias, min_dz, use_depth_only):
        """Core mesh generation — no parameter recording."""
        self.mesh_generator.set_coordinates(top_coords, bottom_coords)
        vertices, faces = self.mesh_generator.generate_simple_mesh(disct_z, bias, min_dz, use_depth_only)
        self.VertFace2csifault(vertices, faces)

    def generate_simple_mesh(self, top_coords=None, bottom_coords=None, disct_z=None,
                                                bias=None, min_dz=None, use_depth_only=True):
        """
        Generate a simple earthquake fault mesh from top to bottom coordinates.

        Parameters:
        - top_coords (ndarray): The top coordinates of the fault.
        - bottom_coords (ndarray): The bottom coordinates of the fault.
        - disct_z (int, optional): Discretization parameter in the z-direction.
        - bias (float, optional): Bias parameter for the mesh.
        - min_dz (float, optional): Minimum size of the mesh in the z-direction.
        - use_depth_only (bool, optional): If True, the full length is the mean depth. Default is True.
        """
        self.record_mesh_call('generate_simple_mesh', {
            'disct_z': disct_z, 'bias': bias, 'min_dz': min_dz,
            'use_depth_only': use_depth_only,
        })
        top_coords = top_coords if top_coords is not None else self.top_coords
        bottom_coords = bottom_coords if bottom_coords is not None else self.bottom_coords
        if top_coords.shape[0] <= 10 and (self.geometry_ref is None or self.geometry_ref.densification is None):
            import warnings
            warnings.warn(
                f"generate_simple_mesh called with only {top_coords.shape[0]} top_coords points "
                f"and no DensificationConfig set. Sparse control points may produce inaccurate "
                "meshes. Densify explicit boundaries before this direct "
                "mesh call.",
                stacklevel=2,
            )
        self._build_simple_mesh(top_coords, bottom_coords, disct_z, bias, min_dz, use_depth_only)

    def generate_simple_multilayer_mesh(self, top_coords=None, layers_coords=None, bottom_coords=None, disct_z=8, bias=1.0):
        """
        Generate a multi-layer earthquake fault mesh.
    
        Parameters:
        - top_coords (numpy.ndarray): The top coordinates, shaped (n, d).
        - layers_coords (list of numpy.ndarray): The coordinates of the intermediate layers.
        - bottom_coords (numpy.ndarray): The bottom coordinates, shaped (n, d).
        - disct_z (int): Discretization parameter in the z-direction.
        - bias (float, optional): Bias used to adjust the length of each segment. Default is 1.0.
        """
        # Record geometry-affecting parameters for config sync
        self.record_mesh_call('generate_simple_multilayer_mesh', {
            'disct_z': disct_z, 'bias': bias,
        })
        top_coords = top_coords if top_coords is not None else self.top_coords
        layers_coords = layers_coords if layers_coords is not None else self.layers
        bottom_coords = bottom_coords if bottom_coords is not None else self.bottom_coords
        self.mesh_generator.set_coordinates(top_coords, bottom_coords)
        vertices, faces = self.mesh_generator.generate_multilayer_mesh(layers_coords, disct_z, bias)
        self.VertFace2csifault(vertices, faces)

    def validate_bayesian_mesh_replay(self, method_name, replay_params=None):
        """Validate prepared state required by one Bayesian mesh replay.

        The check is deliberately separate from ordinary mesh generation.  It
        runs once when a Bayesian target is constructed and never creates,
        remaps, or deforms a mesh.  Mesh methods without a registered replay
        contract need no extra prepared state and return immediately.

        Parameters
        ----------
        method_name : str
            Registered mesh method selected by the effective geometry update.
        replay_params : mapping, optional
            Parameters that the candidate path will pass to ``method_name``.

        Raises
        ------
        ValueError
            If the fixed-topology parametric mapping is absent, no longer
            matches the published mesh, or was prepared with incompatible
            mapping parameters.
        """
        contract = mesh_registry.get_bayesian_replay_contract(method_name)
        if contract is None:
            return
        if contract.get('state') != 'prepared_parametric_mapping':
            raise ValueError(
                f"Mesh method '{method_name}' declares an unsupported "
                f"Bayesian replay state {contract.get('state')!r}."
            )

        replay_params = dict(replay_params or {})
        prepare_hint = (
            "Prepare the baseline with generate_and_deform_mesh(..., "
            "remap=True, bottom_norm_offset=None) before constructing the "
            "Bayesian sampling target."
        )
        prefix = f"Fault '{self.name}': "

        generator = getattr(self, 'mesh_generator', None)
        if generator is None:
            raise ValueError(prefix + "mesh_generator is unavailable. " + prepare_hint)

        mapping_reference = getattr(generator, 'param_mapping_reference', None)
        if (
            mapping_reference is not None
            and mapping_reference is not self.geometry_ref
        ):
            raise ValueError(
                prefix + "the fixed-topology mapping belongs to an older "
                "GeometryReference. Replay the zero-perturbation geometry "
                "after changing the dip profile or densification rule, then "
                + prepare_hint
            )

        param_coords = getattr(generator, 'param_coords', None)
        gmsh_verts = getattr(generator, 'gmsh_verts', None)
        gmsh_faces = getattr(generator, 'gmsh_faces', None)
        current_verts = getattr(self, 'Vertices', None)
        current_faces = getattr(self, 'Faces', None)
        if param_coords is None or len(param_coords) == 0:
            raise ValueError(
                prefix + "the fixed-topology parametric mapping is missing. "
                + prepare_hint
            )
        if any(value is None for value in (
                gmsh_verts, gmsh_faces, current_verts, current_faces)):
            raise ValueError(
                prefix + "the mapped mesh does not contain a complete "
                "vertices/faces pair. " + prepare_hint
            )

        gmsh_verts = np.asarray(gmsh_verts)
        gmsh_faces = np.asarray(gmsh_faces)
        current_verts = np.asarray(current_verts)
        current_faces = np.asarray(current_faces)
        if (
            gmsh_verts.ndim != 2
            or current_verts.ndim != 2
            or gmsh_faces.ndim != 2
            or current_faces.ndim != 2
            or gmsh_faces.shape[1:] != (3,)
            or current_faces.shape[1:] != (3,)
        ):
            raise ValueError(
                prefix + "the prepared mapping or published triangular mesh "
                "has an invalid array shape. " + prepare_hint
            )
        if len(param_coords) != gmsh_verts.shape[0]:
            raise ValueError(
                prefix + "the parametric-coordinate count does not match the "
                "stored Gmsh vertex count. " + prepare_hint
            )
        if current_verts.shape[0] != gmsh_verts.shape[0]:
            raise ValueError(
                prefix + "the published mesh vertex count no longer matches "
                "the prepared mapping. " + prepare_hint
            )
        if (
            current_faces.shape != gmsh_faces.shape
            or not np.array_equal(
                np.sort(current_faces, axis=1),
                np.sort(gmsh_faces, axis=1),
            )
        ):
            raise ValueError(
                prefix + "the published face topology no longer matches the "
                "prepared fixed-topology mapping. " + prepare_hint
            )

        last_call = self.get_last_mesh_call()
        if not last_call or last_call.get('method') != method_name:
            raise ValueError(
                prefix + f"the prepared mapping is not attributable to "
                f"'{method_name}'. " + prepare_hint
            )

        try:
            signature = inspect.signature(getattr(self, method_name))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                prefix + f"cannot resolve the registered mesh method "
                f"'{method_name}'."
            ) from exc

        effective = {}
        for key in contract.get('mapping_parameter_keys', ()):
            if key in replay_params:
                effective[key] = replay_params[key]
                continue
            parameter = signature.parameters.get(key)
            if parameter is None or parameter.default is inspect.Parameter.empty:
                raise ValueError(
                    prefix + f"cannot resolve replay parameter '{key}' for "
                    f"'{method_name}'."
                )
            effective[key] = parameter.default

        prepared = getattr(generator, 'param_mapping_spec', None)
        if not isinstance(prepared, dict) or prepared.get('method') != method_name:
            raise ValueError(
                prefix + "the parametric mapping has no completed preparation "
                "record. " + prepare_hint
            )

        def values_equal(left, right):
            try:
                return bool(np.array_equal(left, right, equal_nan=True))
            except TypeError:
                return bool(np.array_equal(left, right))

        # disct_z takes precedence in MeshGenerator.generate_grid_coordinates;
        # bias/min_dz define the mapping only when disct_z is intentionally None.
        comparable_keys = ['num_segments', 'disct_z']
        if effective.get('disct_z') is None:
            comparable_keys.extend(['bias', 'min_dz'])
        incompatible = [
            key for key in comparable_keys
            if key not in prepared
            or not values_equal(effective.get(key), prepared.get(key))
        ]
        if incompatible:
            details = ', '.join(
                f"{key}: prepared={prepared.get(key)!r}, "
                f"replay={effective.get(key)!r}"
                for key in incompatible
            )
            raise ValueError(
                prefix + "fixed-topology replay parameters do not match the "
                f"prepared mapping ({details}). " + prepare_hint
            )

        projection = effective.get('projection')
        if projection is not None and projection not in {'xy', 'xz', 'yz'}:
            raise ValueError(
                prefix + f"projection must be one of 'xy', 'xz', 'yz', or "
                f"None; got {projection!r}."
            )

        num_segments = effective.get('num_segments')
        disct_z = effective.get('disct_z')
        for index, item in enumerate(param_coords):
            if not isinstance(item, (tuple, list)) or len(item) < 7:
                raise ValueError(
                    prefix + f"parametric coordinate {index} does not follow "
                    "the expected (i, k, u, v, dz, projection, rotation) "
                    "contract. " + prepare_hint
                )
            i, k, _, _, _, node_projection, _ = item[:7]
            if (
                not isinstance(i, (int, np.integer))
                or not isinstance(k, (int, np.integer))
                or i < 0
                or k < 0
            ):
                raise ValueError(
                    prefix + f"parametric coordinate {index} has invalid "
                    f"cell indices ({i!r}, {k!r}). " + prepare_hint
                )
            if num_segments is not None and i + 1 >= int(num_segments):
                raise ValueError(
                    prefix + f"parametric coordinate {index} references "
                    f"along-strike cell {i}, outside num_segments="
                    f"{num_segments}. " + prepare_hint
                )
            if disct_z is not None and k >= int(disct_z):
                raise ValueError(
                    prefix + f"parametric coordinate {index} references "
                    f"down-dip cell {k}, outside disct_z={disct_z}. "
                    + prepare_hint
                )
            if node_projection not in {'xy', 'xz', 'yz'}:
                raise ValueError(
                    prefix + f"parametric coordinate {index} has invalid "
                    f"stored projection {node_projection!r}. " + prepare_hint
                )
            if projection is not None and node_projection != projection:
                raise ValueError(
                    prefix + f"replay projection {projection!r} overrides a "
                    f"stored {node_projection!r} mapping at vertex {index}. "
                    "Use projection=None to preserve each stored projection, "
                    "or prepare the mapping with one explicit projection."
                )

    def generate_and_deform_mesh(self, top_coords=None, bottom_coords=None, top_size=None, bottom_size=None, num_segments=30, 
                                 disct_z=10, rotation_angle: float = None, bottom_norm_offset=None, show=False, 
                                 verbose=0, remap=False, bias=None, min_dz=None, projection=None, 
                                 field_size_dict={'min_dx': 3, 'bias': 1.05}, mesh_func=None, tolerance=1e-6, debug_plot=False,
                                 use_current_mesh=False):
        """Create or reuse a Gmsh topology, then deform it to current edges.

        This method separates *topology setup* from *candidate deformation*.
        The first call normally uses ``remap=True`` to create the Gmsh
        vertices/faces and their parametric mapping.  Later fixed-topology
        candidates use ``remap=False`` and reuse that mapping while changing
        only vertex coordinates.  ``VertFace2csifault`` publishes the final
        pair and invalidates geometry-dependent derived products.

        Parameters
        ----------
        top_coords, bottom_coords : ndarray, optional
            Projected ``(x, y, depth)`` boundary coordinates.  Defaults to the
            current fault edges.  They are ignored when
            ``use_current_mesh=True`` because that mode is anchored to the
            fault's current mesh state.
        top_size, bottom_size : float, optional
            Gmsh target sizes used only when a topology is generated.
        num_segments, disct_z, bias, min_dz : optional
            Resolution of the intermediate structured coordinates used to map
            and deform the Gmsh vertices.
        rotation_angle, projection : optional
            Mapping controls forwarded to ``MeshGenerator``.
        bottom_norm_offset : float, optional
            Pre-mesh bottom-edge perturbation in km.  Any non-``None`` value
            invokes the registered bottom-direction perturbation and therefore
            reads ``geometry_ref``; it is not a mesh-only translation or a
            hidden constant added to every Bayesian sample.
        remap : bool, default False
            Rebuild Gmsh topology and its parametric mapping.  Bayesian
            candidate replay normally forbids this because parameter and slip
            indexing must remain fixed.
        use_current_mesh : bool, default False
            Rebuild the mapping from current ``Vertices/Faces`` instead of
            generating a new Gmsh topology.

        Returns
        -------
        vertices, faces : tuple of ndarray
            Published deformed vertices and their triangular connectivity.

        Notes
        -----
        ``Faces`` row order, vertex numbering, and triangle winding are passed
        through unchanged on the fixed-topology path.  Scientific consumers
        such as slip vectors, Green's functions, and Laplacians rely on this
        stable indexing.
        """
        _user_passed_coords = top_coords is not None or bottom_coords is not None
        # Record geometry-affecting parameters for config sync
        self.record_mesh_call('generate_and_deform_mesh', {
            'top_size': top_size, 'bottom_size': bottom_size,
            'num_segments': num_segments, 'disct_z': disct_z,
            'rotation_angle': rotation_angle, 'bottom_norm_offset': bottom_norm_offset,
            'remap': remap, 'bias': bias, 'min_dz': min_dz,
            'projection': projection, 'field_size_dict': field_size_dict,
            'mesh_func': mesh_func, 'tolerance': tolerance,
            'use_current_mesh': use_current_mesh,
        })
        top_coords = top_coords if top_coords is not None else self.top_coords
        bottom_coords = bottom_coords if bottom_coords is not None else self.bottom_coords
        reference = self.geometry_ref
        if (
            reference is not None
            and reference.dip_profile is not None
            and self._materialized_geometry_ref is not reference
        ):
            raise ValueError(
                f"Fault '{self.name}': the dip profile or densification rule "
                "changed the frozen candidate contract, but current top/bottom "
                "still belong to the previous materialization. Replay the "
                "zero-perturbation "
                "dip method before generate_and_deform_mesh(), for example "
                "profile = fault.geometry_ref.dip_profile; "
                "fault.perturb_dips_with_preset_params("
                "np.zeros(profile.perturbation_parameter_count), ...)."
            )
        if top_coords.shape[0] <= 10 and (self.geometry_ref is None or self.geometry_ref.densification is None):
            import warnings
            warnings.warn(
                f"generate_and_deform_mesh called with only {top_coords.shape[0]} top_coords points "
                f"and no DensificationConfig set. Sparse control points may produce inaccurate "
                "meshes. Configure reference-owned densification and replay "
                "the candidate before creating the mapping.",
                stacklevel=2,
            )

        topology_changed = False
        if use_current_mesh:
            if _user_passed_coords:
                import warnings
                warnings.warn(
                    "top_coords/bottom_coords are ignored when use_current_mesh=True. "
                    "Using self.top_coords and self.bottom_coords instead.",
                    stacklevel=2,
                )
            print("Using existing mesh (self.Vertices and self.Faces)")
            gmsh_verts = self.Vertices
            gmsh_faces = self.Faces
            
            if bottom_norm_offset is not None:
                self.perturb_bottom_coords_along_fixed_direction([bottom_norm_offset])
            top_coords, bottom_coords = self.top_coords, self.bottom_coords
            
            self.mesh_generator.set_coordinates(top_coords, bottom_coords)
            self.mesh_generator.gmsh_verts = gmsh_verts
            self.mesh_generator.gmsh_faces = gmsh_faces

            sep_top_coords, sep_bottom_coords = (
                resample_boundaries_by_normalized_arclength(
                    top_coords,
                    bottom_coords,
                    num_segments=num_segments,
                )
            )
            self.mesh_generator.set_coordinates(sep_top_coords, sep_bottom_coords)
            
            mesh_coords = self.mesh_generator.generate_grid_coordinates(top_coords=self.mesh_generator.top_coords, 
                                                                        bottom_coords=self.mesh_generator.bottom_coords, 
                                                                        disct_z=disct_z, bias=bias, min_dz=min_dz)
            
            self.mesh_generator.map_gmsh_vertices_to_grid(gmsh_verts, mesh_coords, 
                                                            rotation_angle=rotation_angle, 
                                                            projection=projection, 
                                                            tolerance=tolerance, 
                                                            debug_plot=debug_plot)
            self.mesh_generator.param_mapping_spec = {
                'method': 'generate_and_deform_mesh',
                'num_segments': num_segments,
                'disct_z': disct_z,
                'bias': bias,
                'min_dz': min_dz,
                'projection': projection,
            }
            self.mesh_generator.param_mapping_reference = self.geometry_ref
            
        else:
            if self.mesh_generator.param_coords is None or remap:
                topology_changed = True
                if bottom_norm_offset is not None:
                    self.perturb_bottom_coords_along_fixed_direction([bottom_norm_offset])
                top_coords, bottom_coords = self.top_coords, self.bottom_coords
                # Dip controls and transition endpoints may be present only in
                # the physics working boundary.  The Gmsh top spline remains
                # defined by the trace-node layer so evaluation events cannot
                # silently change the reference trace curve.
                gmsh_top_coords = top_coords
                if (
                    self._materialized_trace_reference is self.geometry_ref
                    and self._materialized_trace_top_coords is not None
                ):
                    gmsh_top_coords = self._materialized_trace_top_coords
                self.mesh_generator.set_coordinates(gmsh_top_coords, bottom_coords)
                gmsh_verts, gmsh_faces = self.mesh_generator.generate_gmsh_mesh(top_size=top_size, bottom_size=bottom_size, 
                                                                                show=show, verbose=verbose, save_in_self=True, 
                                                                                field_size_dict=field_size_dict, mesh_func=mesh_func)
                
                sep_top_coords, sep_bottom_coords = (
                    resample_boundaries_by_normalized_arclength(
                        top_coords,
                        bottom_coords,
                        num_segments=num_segments,
                    )
                )
                self.mesh_generator.set_coordinates(sep_top_coords, sep_bottom_coords)
                
                mesh_coords = self.mesh_generator.generate_grid_coordinates(top_coords=self.mesh_generator.top_coords, 
                                                                            bottom_coords=self.mesh_generator.bottom_coords, 
                                                                            disct_z=disct_z, bias=bias, min_dz=min_dz)
                
                self.mesh_generator.map_gmsh_vertices_to_grid(gmsh_verts, mesh_coords, 
                                                              rotation_angle=rotation_angle, 
                                                              projection=projection, 
                                                              tolerance=tolerance, 
                                                              debug_plot=debug_plot)
                # This is the successful mapping transaction.  Candidate
                # replays below reuse this state and therefore do not replace
                # its provenance with each sample's call arguments.
                self.mesh_generator.param_mapping_spec = {
                    'method': 'generate_and_deform_mesh',
                    'num_segments': num_segments,
                    'disct_z': disct_z,
                    'bias': bias,
                    'min_dz': min_dz,
                    'projection': projection,
                }
                self.mesh_generator.param_mapping_reference = self.geometry_ref
            else:
                gmsh_verts = self.mesh_generator.gmsh_verts
                gmsh_faces = self.mesh_generator.gmsh_faces
        
        # Mapping columns intentionally retain the historical normalized-edge
        # contract: top and bottom are resampled independently on their own
        # true arc lengths, then paired by shared xi=j/(N-1).  This correction
        # does not introduce material-point station pairing.
        sep_top_coords, sep_bottom_coords = (
            resample_boundaries_by_normalized_arclength(
                top_coords,
                bottom_coords,
                num_segments=num_segments,
            )
        )
        
        new_gmsh_verts = self.mesh_generator.deform_mesh(sep_top_coords, sep_bottom_coords, disct_z=disct_z, 
                                                            bias=bias, min_dz=min_dz, projection=projection)
        
        self.VertFace2csifault(
            new_gmsh_verts, gmsh_faces,
            topology_changed=topology_changed,
            change_kind='deform',
        )
        
        return new_gmsh_verts, gmsh_faces
    
    #--------------------------------------Mesh Rebuild & Edge Setup--------------------------------------#
    def rebuild_simple_mesh(self, disct_z=None, bias=None, min_dz=None, segs=5, top_tolerance=0.1, bottom_tolerance=0.1, lonlat=True, buffer_depth=0.1, sort_axis=0, sort_order='ascend', use_trace=False, discretized=False):
        """
        Rebuild the simple earthquake fault mesh for the usage of the perturbation functions.
        """
        # Record geometry-affecting parameters for config sync
        self.record_mesh_call('rebuild_simple_mesh', {
            'disct_z': disct_z, 'bias': bias, 'min_dz': min_dz,
            'segs': segs, 'top_tolerance': top_tolerance,
            'bottom_tolerance': bottom_tolerance, 'lonlat': lonlat,
            'buffer_depth': buffer_depth, 'sort_axis': sort_axis,
            'sort_order': sort_order, 'use_trace': use_trace,
            'discretized': discretized,
        })
        if use_trace:
            self.set_top_coords_from_trace(discretized=discretized)
        else:
            self.set_top_coords_from_geometry(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, lonlat=lonlat, buffer_depth=buffer_depth, sort_axis=sort_axis, sort_order=sort_order)

        self.set_bottom_coords_from_geometry(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, lonlat=lonlat, buffer_depth=buffer_depth, sort_axis=sort_axis, sort_order=sort_order)
        self.discretize_bottom_coords(num_segments=segs)
        self.discretize_top_coords(num_segments=segs)
        self._build_simple_mesh(self.top_coords, self.bottom_coords, disct_z=disct_z, bias=bias, min_dz=min_dz, use_depth_only=True)
        self.initializeslip(values='depth')
        self.snapshot()
        return
    
    def set_edges_for_bayesian_optimization(self, segs=None, top_tolerance=0.1, bottom_tolerance=0.1, lonlat=True, depth_tolerance=0.1, buffer_depth=0.1, sort_axis=0, sort_order='ascend', use_trace=False, discretized=False,
                                               densify_num_segments=None, densify_interval=None):
        """Extract reference edges from current geometry and freeze them.

        ``use_trace=True`` takes the top edge from the trace; otherwise both
        edges are identified from current geometry.  The bottom edge is always
        obtained from geometry.  Optional ``segs`` discretization happens
        before ``snapshot(capture_vertices=False)`` so the frozen coordinates
        have the same point order used by later perturbations.

        ``sort_axis`` and ``sort_order`` define the positive boundary sequence
        used by strike/dip operations.  They do not describe observation-track
        direction.  ``densify_*`` stores a later physics/mesh densification
        rule without changing the frozen sparse control coordinates.

        ``depth_tolerance`` is retained in the historical signature but is not
        consumed by the current implementation.  Edge selection uses
        ``top_tolerance``, ``bottom_tolerance`` and ``buffer_depth``.

        Use this convenience method only when the reference edges must be
        extracted from existing geometry.  If authoritative top/bottom arrays
        are already available, call ``snapshot(...)`` directly.
        """
        if use_trace:
            self.set_top_coords_from_trace(discretized=discretized)
        else:
            self.set_top_coords_from_geometry(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, lonlat=lonlat, buffer_depth=buffer_depth, sort_axis=sort_axis, sort_order=sort_order)

        self.set_bottom_coords_from_geometry(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, lonlat=lonlat, buffer_depth=buffer_depth, sort_axis=sort_axis, sort_order=sort_order)

        if segs is not None:
            self.discretize_bottom_coords(num_segments=segs)
            self.discretize_top_coords(num_segments=segs)

        self.snapshot(capture_vertices=False)

        if densify_num_segments is not None or densify_interval is not None:
            self.set_densification(num_segments=densify_num_segments, interval=densify_interval)

        return
    #----------------------------------------------------------------------------------------------------------#

    #--------------------------------------Convenience Setup--------------------------------------#
    def prepare_for_inversion(self, segs=None, sort_axis=0, sort_order='ascend',
                              top_tolerance=0.1, bottom_tolerance=0.1, lonlat=True,
                              buffer_depth=0.1, use_trace=False, discretized=False,
                              dip_sampled_controls=None,
                              dip_fixed_controls=None,
                              dip_interpolation_axis='auto',
                              dip_transition_zones=None,
                              dip_controls_are_utm=False,
                              densify_num_segments=None, densify_interval=None,
                              dip_perturbation_groups=None):
        """One-call convenience setup for Bayesian geometry optimization.

        Delegates to ``set_edges_for_bayesian_optimization()`` for geometry
        setup (all geometry parameters are forwarded), then optionally freezes
        one explicit sampled/fixed dip profile. See :meth:`set_dip_profile`
        for the profile and transition-zone contract.
        """
        self.set_edges_for_bayesian_optimization(
            segs=segs, top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance,
            lonlat=lonlat, buffer_depth=buffer_depth, sort_axis=sort_axis,
            sort_order=sort_order, use_trace=use_trace, discretized=discretized,
            densify_num_segments=densify_num_segments, densify_interval=densify_interval,
        )

        if dip_sampled_controls is not None:
            self.set_dip_profile(
                sampled_controls=dip_sampled_controls,
                fixed_controls=dip_fixed_controls,
                interpolation_axis=dip_interpolation_axis,
                transition_zones=dip_transition_zones,
                perturbation_groups=dip_perturbation_groups,
                is_utm=dip_controls_are_utm,
            )
    #----------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    lon0, lat0 = 116.5, 39.5
    myfault = BayesianAdaptiveTriangularPatches('myfault', lon0=lon0, lat0=lat0)
    # Print the perturbation methods
    print(myfault.perturbation_methods.keys())
