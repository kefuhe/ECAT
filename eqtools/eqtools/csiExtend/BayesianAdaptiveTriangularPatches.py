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
    DipControlPoints,
    DensificationConfig,
)
from .MeshGenerator import MeshGenerator
from . import mesh_registry


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
            If ``geometry_ref`` already exists, its ``dip_control_points`` and
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

        # Preserve existing dip_control_points if already attached
        dip_cp = None
        if self.geometry_ref is not None and self.geometry_ref.dip_control_points is not None:
            dip_cp = self.geometry_ref.dip_control_points

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
            dip_control_points=dip_cp,
            densification=densification,
        )
        self._mesh_reference_validation_key = None

        # Keep legacy attributes in sync (read-only views into the frozen ref)

        # Apply deferred densification config from YAML (set before geometry_ref existed)
        pending = getattr(self, '_pending_densification', None)
        if pending is not None:
            self.set_densification(**pending)
            del self._pending_densification

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

    # --- Dip-control-point setters (unified: always via geometry_ref) ---------
    def _ensure_geometry_ref_for_dip(self):
        """Auto-create a minimal geometry_ref if it doesn't exist yet.

        This allows ``set_dip_control_points*()`` to be called before
        ``snapshot()`` — a common workflow where dip control points are
        needed to generate bottom_coords before the mesh exists.

        The minimal ref only captures whatever top/bottom coords are
        currently available (may be None).  A later ``snapshot()`` call
        will replace this with a complete ref while preserving the
        attached ``dip_control_points``.
        """
        if self.geometry_ref is not None:
            return
        top = getattr(self, 'top_coords', None)
        bot = getattr(self, 'bottom_coords', None)
        self.geometry_ref = GeometryReference(
            top_coords=top.copy() if top is not None else None,
            bottom_coords=bot.copy() if bot is not None else None,
        )

    def set_dip_control_points(self, x, y, dip, is_utm=False):
        """Set dip control points and attach to geometry_ref.

        Parameters
        ----------
        x, y : array-like
            Control point coordinates. Longitude/latitude by default;
            set ``is_utm=True`` to pass UTM-x/y in km.
        dip : array-like
            Oriented reference dip in degrees. Values in
            ``[-90, 0) U (0, 180)`` are accepted; for example, ``-80`` and
            ``100`` are equivalent. The immutable control-point container
            stores only continuous ``(0, 180)`` values, so later Bayesian
            proposal increments never need to infer the caller's convention.
        is_utm : bool, default False
            If True, ``x``/``y`` are interpreted as UTM (km) and converted
            via ``xy2ll`` before storage. Stored form is always lon/lat.
        """
        self._ensure_geometry_ref_for_dip()
        if is_utm:
            lon, lat = self.xy2ll(np.asarray(x, dtype=np.float64),
                                  np.asarray(y, dtype=np.float64))
        else:
            lon, lat = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
            if np.any(np.abs(lon) > 360) or np.any(np.abs(lat) > 90):
                warnings.warn(
                    "Coordinate values exceed lon/lat range (|x|>360 or |y|>90). "
                    "If passing UTM coordinates, set is_utm=True.",
                    stacklevel=2,
                )
        dcp = DipControlPoints(x=lon, y=lat, dip=np.asarray(dip, dtype=np.float64))
        self.geometry_ref = self.geometry_ref.with_dip(dcp)

    def set_dip_control_points_from_coords(self, coords, dips, is_utm=False):
        """Set dip control points from a coordinate array."""
        self._ensure_geometry_ref_for_dip()
        dcp = DipControlPoints.from_coords(coords, dips, is_utm=is_utm,
                                            xy2ll_func=self.xy2ll if is_utm else None)
        self.geometry_ref = self.geometry_ref.with_dip(dcp)

    def set_dip_control_points_from_file(self, filename, header=0, is_utm=False):
        """Set dip control points from a text file (lon lat dip)."""
        self._ensure_geometry_ref_for_dip()
        dcp = DipControlPoints.from_file(filename, header=header, is_utm=is_utm,
                                          xy2ll_func=self.xy2ll if is_utm else None)
        self.geometry_ref = self.geometry_ref.with_dip(dcp)

    def refresh_geometry_baseline(self):
        """Re-snapshot top/bottom coords while preserving dip and densification.

        Use after external modifications to ``self.top_coords`` or
        ``self.bottom_coords`` that should be reflected in the baseline,
        without changing dip control points or densification config.
        """
        if self.geometry_ref is None or self.geometry_ref.dip_control_points is None:
            raise ValueError(
                "No existing geometry_ref with dip control points to preserve. "
                "Call set_dip_control_points() and snapshot() first."
            )
        self.snapshot()

    def update_dip_baseline(self, x=None, y=None, dip=None):
        """Update the dip control points baseline in geometry_ref.

        Parameters:
            x, y, dip: If all three are provided, replace the current dip
                baseline. If all are None, equivalent to
                ``refresh_geometry_baseline()`` — re-snapshot coords while
                preserving existing dip settings. Partial arguments raise
                ValueError.
        """
        if x is not None and y is not None and dip is not None:
            self.set_dip_control_points(x, y, dip)
        elif x is None and y is None and dip is None:
            self.refresh_geometry_baseline()
        else:
            raise ValueError("Provide all of x, y, dip or none of them.")

    # --- Densification configuration -------------------------------------------
    def set_densification(self, num_segments=None, interval=None, enabled=True):
        """Configure automatic coordinate densification for mesh/physics consumers.

        Call after ``snapshot()`` (or ``set_edges_for_bayesian_optimization``).
        The config is stored in ``geometry_ref`` and survives re-snapshots.

        Parameters
        ----------
        num_segments : int, optional
            Target number of segments after densification.
        interval : float, optional
            Target spacing (km) between densified points.
        enabled : bool
            Set False to disable without removing config.
        """
        if self.geometry_ref is None:
            raise ValueError("geometry_ref not set. Call snapshot() first.")
        cfg = DensificationConfig(num_segments=num_segments, interval=interval, enabled=enabled)
        self.geometry_ref = self.geometry_ref.with_densification(cfg)

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
                f"meshes. Consider calling set_densification(num_segments=N) after snapshot().",
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
        if top_coords.shape[0] <= 10 and (self.geometry_ref is None or self.geometry_ref.densification is None):
            import warnings
            warnings.warn(
                f"generate_and_deform_mesh called with only {top_coords.shape[0]} top_coords points "
                f"and no DensificationConfig set. Sparse control points may produce inaccurate "
                f"meshes. Consider calling set_densification(num_segments=N) after snapshot().",
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

            sep_top_coords, _ = self.discretize_coords(top_coords, num_segments=num_segments)
            sep_bottom_coords, _ = self.discretize_coords(bottom_coords, num_segments=num_segments)
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
            
        else:
            if self.mesh_generator.param_coords is None or remap:
                topology_changed = True
                if bottom_norm_offset is not None:
                    self.perturb_bottom_coords_along_fixed_direction([bottom_norm_offset])
                top_coords, bottom_coords = self.top_coords, self.bottom_coords
                self.mesh_generator.set_coordinates(top_coords, bottom_coords)
                gmsh_verts, gmsh_faces = self.mesh_generator.generate_gmsh_mesh(top_size=top_size, bottom_size=bottom_size, 
                                                                                show=show, verbose=verbose, save_in_self=True, 
                                                                                field_size_dict=field_size_dict, mesh_func=mesh_func)
                
                sep_top_coords, _ = self.discretize_coords(top_coords, num_segments=num_segments)
                sep_bottom_coords, _ = self.discretize_coords(bottom_coords, num_segments=num_segments)
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
            else:
                gmsh_verts = self.mesh_generator.gmsh_verts
                gmsh_faces = self.mesh_generator.gmsh_faces
        
        sep_top_coords, _ = self.discretize_coords(top_coords, num_segments=num_segments)
        sep_bottom_coords, _ = self.discretize_coords(bottom_coords, num_segments=num_segments)
        
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
                              dip_control_coords=None, dip_control_dips=None,
                              dip_control_file=None, is_utm=False,
                              densify_num_segments=None, densify_interval=None):
        """One-call convenience setup for Bayesian geometry optimization.

        Delegates to ``set_edges_for_bayesian_optimization()`` for geometry
        setup (all geometry parameters are forwarded), then optionally
        configures dip control points.  See the individual methods for
        parameter details.
        """
        self.set_edges_for_bayesian_optimization(
            segs=segs, top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance,
            lonlat=lonlat, buffer_depth=buffer_depth, sort_axis=sort_axis,
            sort_order=sort_order, use_trace=use_trace, discretized=discretized,
            densify_num_segments=densify_num_segments, densify_interval=densify_interval,
        )

        if dip_control_file is not None:
            self.set_dip_control_points_from_file(dip_control_file, is_utm=is_utm)
        elif dip_control_coords is not None and dip_control_dips is not None:
            self.set_dip_control_points_from_coords(dip_control_coords, dip_control_dips, is_utm=is_utm)
    #----------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    lon0, lat0 = 116.5, 39.5
    myfault = BayesianAdaptiveTriangularPatches('myfault', lon0=lon0, lat0=lat0)
    # Print the perturbation methods
    print(myfault.perturbation_methods.keys())
