import copy
import functools
import inspect
import warnings
from dataclasses import dataclass, fields, replace as _dataclass_replace

import numpy as np

# Import geometry base classes
# Ensure these files exist in the same directory
from .AdaptiveLayeredDipTriangularPatches import AdaptiveLayeredDipTriangularPatches
from . import mesh_registry


# =============================================================================
# 0. Geometry Reference — Immutable Snapshot of Baseline Geometry
# =============================================================================
def _freeze_array(arr):
    """Return a read-only copy of a numpy array. None passes through."""
    if arr is None:
        return None
    arr = np.array(arr)
    arr.flags.writeable = False
    return arr


def _freeze_tuple_of_arrays(seq):
    """Freeze a sequence of arrays into a tuple of read-only arrays."""
    if seq is None:
        return None
    return tuple(_freeze_array(a) for a in seq)


@dataclass(frozen=True)
class DipControlPoints:
    """Immutable container for dip control-point reference data.

    Attributes:
        x:   longitude (or x-UTM) coordinates of dip control points.
        y:   latitude  (or y-UTM) coordinates of dip control points.
        dip: dip angle values (degrees).
    """
    x: np.ndarray
    y: np.ndarray
    dip: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'x', _freeze_array(self.x))
        object.__setattr__(self, 'y', _freeze_array(self.y))
        object.__setattr__(self, 'dip', _freeze_array(self.dip))

    # --- convenience constructors -------------------------------------------
    @classmethod
    def from_coords(cls, coords, dips, is_utm=False, xy2ll_func=None):
        """Create from an (N,2) coordinate array and a dip array.

        Parameters:
            coords:     (N, 2) array — lon/lat or x/y.
            dips:       (N,) array of dip angles.
            is_utm:     if *True*, convert via *xy2ll_func* first.
            xy2ll_func: callable(x, y) → (lon, lat). Required when *is_utm*.
        """
        if is_utm:
            if xy2ll_func is None:
                raise ValueError("xy2ll_func is required when is_utm=True")
            x, y = coords[:, 0], coords[:, 1]
            lon, lat = xy2ll_func(x, y)
            return cls(x=lon, y=lat, dip=np.asarray(dips))
        return cls(x=coords[:, 0], y=coords[:, 1], dip=np.asarray(dips))

    @classmethod
    def from_file(cls, filename, header=0, is_utm=False, xy2ll_func=None):
        """Load from a text file (columns: lon lat dip  or  x y dip)."""
        data = np.loadtxt(filename, skiprows=header)
        coords = data[:, :2]
        dips = data[:, 2]
        return cls.from_coords(coords, dips, is_utm=is_utm, xy2ll_func=xy2ll_func)


@dataclass(frozen=True)
class DensificationConfig:
    """Configuration for sparse-to-dense coordinate densification.

    Stored in ``GeometryReference`` and applied automatically at the boundary
    between perturbation (which works on sparse control points) and physics
    operations (which need dense coordinates for accurate strike, dip
    interpolation, and meshing).

    Specify exactly one of *num_segments* or *interval*.

    Attributes:
        num_segments: Fixed number of equal-arc-length segments.
        interval:     Arc-length interval in km for discretization.
        enabled:      Master switch.  When False, densification is skipped.
    """
    num_segments: int = None
    interval: float = None
    enabled: bool = True

    def __post_init__(self):
        if self.enabled:
            if self.num_segments is not None and self.interval is not None:
                raise ValueError(
                    "DensificationConfig: specify num_segments OR interval, not both."
                )
            if self.num_segments is None and self.interval is None:
                raise ValueError(
                    "DensificationConfig: when enabled, provide num_segments or interval."
                )
        if self.num_segments is not None and self.num_segments < 2:
            raise ValueError("num_segments must be >= 2")
        if self.interval is not None and self.interval <= 0:
            raise ValueError("interval must be > 0")


@dataclass(frozen=True)
class GeometryReference:
    """Immutable snapshot of baseline fault geometry.

    Created once via ``snapshot()`` (or ``set_edges_for_bayesian_optimization``),
    and consumed by all perturbation mixins as the fixed reference for
    generating perturbed geometry each SMC/MCMC iteration.

    All numpy arrays stored here have ``writeable=False``; attempting to
    modify elements raises ``ValueError``.

    To *evolve* a snapshot (e.g. add vertices later, or attach dip data),
    use the ``with_*`` helpers which return a **new** frozen instance via
    ``dataclasses.replace``.
    """
    top_coords: np.ndarray = None                  # (n, 3) | None
    bottom_coords: np.ndarray = None              # (n, 3) | None
    layers: object = None                         # tuple[np.ndarray, ...] | None
    vertices: np.ndarray = None                   # (V, 3) | None
    faces: np.ndarray = None                      # (F, 3) | None
    dip_control_points: DipControlPoints = None   # optional
    densification: DensificationConfig = None     # optional

    def __post_init__(self):
        object.__setattr__(self, 'top_coords', _freeze_array(self.top_coords))
        object.__setattr__(self, 'bottom_coords', _freeze_array(self.bottom_coords))
        object.__setattr__(self, 'layers', _freeze_tuple_of_arrays(self.layers))
        object.__setattr__(self, 'vertices', _freeze_array(self.vertices))
        object.__setattr__(self, 'faces', _freeze_array(self.faces))

    # --- evolution helpers (return NEW frozen instances) ----------------------
    def with_vertices(self, vertices, faces):
        """Return a new GeometryReference with updated vertices/faces."""
        return _dataclass_replace(self, vertices=vertices, faces=faces)

    def with_layers(self, layers):
        """Return a new GeometryReference with updated layers."""
        return _dataclass_replace(self, layers=layers)

    def with_dip(self, dip_control_points):
        """Return a new GeometryReference with updated dip control points."""
        return _dataclass_replace(self, dip_control_points=dip_control_points)

    def with_densification(self, densification):
        """Return a new GeometryReference with updated densification config."""
        return _dataclass_replace(self, densification=densification)

# =============================================================================
# 1. Registry System (For Help & Documentation & Config Validation)
# =============================================================================
class PerturbationRegistry:
    """
    Global Registry: Stores the method names, descriptions, and parameters 
    for all perturbation methods across different fault classes.
    """
    _methods = {}

    @classmethod
    def register(cls, class_name, method_name, meta_info):
        """Register a method with its metadata under a specific class name."""
        if class_name not in cls._methods:
            cls._methods[class_name] = {}
        cls._methods[class_name][method_name] = meta_info

    @classmethod
    def get_meta(cls, instance_or_class, method_name):
        """Look up a single method's meta_info via MRO (child → parent), or None."""
        if isinstance(instance_or_class, type):
            target_cls = instance_or_class
        else:
            target_cls = instance_or_class.__class__
        for base in inspect.getmro(target_cls):
            base_name = base.__name__
            if base_name in cls._methods and method_name in cls._methods[base_name]:
                return cls._methods[base_name][method_name]
        return None

    @classmethod
    def get_help(cls, instance_or_name=None):
        """
        Retrieve the help dictionary.
        
        Args:
            instance_or_name: Can be an instance object, a class object, or a string class name.
            
        Returns:
            dict: A dictionary of available methods and their metadata.
            If an instance/class is provided, it returns methods from the entire 
            inheritance chain (MRO).
        """
        # If no argument, return the entire registry
        if instance_or_name is None:
            return cls._methods

        # Handling String Input (e.g., from Config file)
        # Returns exactly what is registered under this string name.
        if isinstance(instance_or_name, str):
            return cls._methods.get(instance_or_name, {})

        # Handling Object/Class Input (e.g., fault.help())
        # Uses MRO to collect methods from all parent classes.
        target_cls = instance_or_name if isinstance(instance_or_name, type) else instance_or_name.__class__
        
        combined_methods = {}
        # Iterate MRO in reverse (Parent -> Child) so children can overwrite parents
        for base in reversed(inspect.getmro(target_cls)):
            base_name = base.__name__
            if base_name in cls._methods:
                combined_methods.update(cls._methods[base_name])
                
        return combined_methods

# =============================================================================
# 2. Enhanced Decorator (Tracks State & Records Documentation)
# =============================================================================
def track_mesh_update(update_mesh=False, update_laplacian=False, update_area=False,
                      description="", params_info=None, expected_perturbations_count=None,
                      bayesian_forbidden=None):
    """
    Decorator: Tracks the update state of the mesh/laplacian while recording
    documentation information for the registry.

    Args:
        update_mesh (bool): If True, sets self.mesh_updated = True after execution.
        update_laplacian (bool): If True, sets self.laplacian_updated = True.
        update_area (bool): If True, sets self.area_updated = True.
        description (str): A brief description of what the perturbation does.
        params_info (dict/str): Information about expected parameters.
        expected_perturbations_count (int, optional): Validates the length of the 'perturbations' array.
        bayesian_forbidden (dict, optional): Parameters forbidden in Bayesian mode.
            Keys are parameter names, values are rules: True (exact match),
            'not_none', or 'truthy'.
    """
    def decorator(func):
        # Mount metadata for the Metaclass/Registry to read
        func._bayesian_meta = {
            "description": description,
            "params": params_info,
            "flags": {"mesh": update_mesh, "lap": update_laplacian},
            "forbidden": bayesian_forbidden or {},
        }
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            perturbations = args[0] if args else kwargs.get('perturbations', None)
            if perturbations is None:
                raise ValueError(f"Method '{func.__name__}': The 'perturbations' parameter is required.")
            
            # Validate parameter count if specified
            if expected_perturbations_count is not None and len(perturbations) != expected_perturbations_count:
                raise ValueError(f"Method '{func.__name__}': Expected {expected_perturbations_count} elements in 'perturbations', but got {len(perturbations)}.")

            # Reset flags before execution
            self.mesh_updated = False
            self.laplacian_updated = False
            self.area_updated = False
            
            # Execute the actual perturbation function
            result = func(self, *args, **kwargs)
            
            # Update flags based on decorator arguments
            if update_mesh: self.mesh_updated = True
            if update_laplacian: self.laplacian_updated = True
            if update_area: self.area_updated = True
            
            return result
        
        wrapper._is_decorated = True
        # Ensure metadata is accessible via the wrapper
        wrapper._bayesian_meta = func._bayesian_meta
        return wrapper
    return decorator

# =============================================================================
# 3. Perturbation Base Logic (uses __init_subclass__ instead of Metaclass)
# =============================================================================
class PerturbationBase:
    """
    Base class providing logic for state flags and dynamic method dispatch.
    """
    def __init_subclass__(cls, **kwargs):
        """
        Automatically called when a subclass is defined.
        Scans the entire MRO for perturb_* methods, categorizes them,
        enforces decorator usage, and registers to PerturbationRegistry.
        """
        super().__init_subclass__(**kwargs)
        cls.perturbation_methods = {}
        cls.bayesian_perturbation_methods = {}

        # Collect all perturb_* methods from the full MRO (including Mixins)
        for key in dir(cls):
            if not key.startswith('perturb_'):
                continue
            value = getattr(cls, key, None)
            if value is None or not callable(value):
                continue

            # Heuristic to identify Bayesian methods: (self, perturbations, ...)
            try:
                parameters = list(inspect.signature(value).parameters.values())
                is_bayesian = (len(parameters) >= 2 and
                               parameters[0].name == 'self' and
                               parameters[1].name == 'perturbations')
            except (ValueError, TypeError):
                is_bayesian = False

            if is_bayesian:
                cls.bayesian_perturbation_methods[key] = value

                # 1. Enforce Decorator Usage
                if not getattr(value, '_is_decorated', False):
                    raise TypeError(
                        f"The method '{key}' in class '{cls.__name__}' is not "
                        f"decorated with '@track_mesh_update'.")

                # 2. Auto-infer family from the Mixin class that defines this method
                family = "Other"
                for base in cls.__mro__:
                    if key in base.__dict__:
                        fname = base.__name__
                        for suffix in ('PerturbationMixin', 'Mixin'):
                            if fname.endswith(suffix):
                                fname = fname[:-len(suffix)]
                                break
                        family = fname
                        break

                # 3. Extract configuration kwargs from signature
                #    (skip 'self' and 'perturbations' — those are handled by the SMC sampler)
                try:
                    sig = inspect.signature(value)
                    kwargs_info = {}
                    for pname, param in sig.parameters.items():
                        if pname in ('self', 'perturbations'):
                            continue
                        kwargs_info[pname] = param.default
                except (ValueError, TypeError):
                    kwargs_info = {}

                # 4. Register to Global Registry (copy to avoid mutating the original)
                meta_info = {**getattr(value, '_bayesian_meta',
                                       {"description": "No description", "params": "N/A"}),
                             "family": family,
                             "kwargs": kwargs_info}
                PerturbationRegistry.register(cls.__name__, key, meta_info)
            else:
                # General perturbation methods
                cls.perturbation_methods[key] = value

    def __init__(self):
        self.geometry_updated = False
        self.mesh_updated = False
        self.laplacian_updated = False
        self.area_updated = False
        self.geometry_ref = None
        self._last_mesh_params = None
        self._last_mesh_raw_call = None

    # --- Geometry-reference helpers ------------------------------------------
    def _require_geometry_ref(self, *field_names):
        """Ensure ``self.geometry_ref`` exists and requested fields are not None.

        Raises:
            ValueError: if geometry_ref is None or any requested field is None.
        """
        if self.geometry_ref is None:
            raise ValueError(
                "geometry_ref is not set. Call snapshot() or "
                "set_edges_for_bayesian_optimization() first."
            )
        for name in field_names:
            if getattr(self.geometry_ref, name, None) is None:
                raise ValueError(
                    f"geometry_ref.{name} is None. "
                    f"Ensure it was captured during snapshot()."
                )

    def _ensure_vertices_ref(self):
        """Lazily capture current Vertices into the geometry_ref if missing."""
        self._require_geometry_ref()
        if self.geometry_ref.vertices is None:
            if not hasattr(self, 'Vertices') or self.Vertices is None:
                raise ValueError(
                    "Cannot capture vertices reference: self.Vertices is None. "
                    "Build the mesh first (e.g. build_triangular_mesh or "
                    "rebuild_simple_mesh) before calling perturbations that "
                    "require vertices (direction, rotation, translation)."
                )
            self.geometry_ref = self.geometry_ref.with_vertices(
                self.Vertices.copy(), self.Faces.copy()
            )

    def is_mesh_updated(self): return self.mesh_updated
    def is_laplacian_updated(self): return self.laplacian_updated

    # --- Edge densification ---------------------------------------------------
    def densify_edges(self, top_only=False):
        """Densify sparse control coords to working resolution before physics ops.

        Reads ``DensificationConfig`` from ``self.geometry_ref.densification``.
        No-op when the config is absent, disabled, or coords are already dense.

        Call **after** perturbation and **before** the first consumer operation
        (dip interpolation, bottom generation, or mesh generation).

        Parameters
        ----------
        top_only : bool
            If True, only densify ``self.top_coords``.  Use this in the dip
            perturbation path where bottom is regenerated from physics
            (dense top + dip) and pre-densifying bottom would be wasted.

        Returns
        -------
        bool
            True if densification was actually performed.
        """
        if self.geometry_ref is None or self.geometry_ref.densification is None:
            return False
        config = self.geometry_ref.densification
        if not config.enabled:
            return False

        from .geom_ops import discretize_coords

        n_before = self.top_coords.shape[0]
        if config.num_segments is not None and n_before >= config.num_segments:
            return False
        if config.interval is not None:
            dx = np.diff(self.top_coords[:, 0])
            dy = np.diff(self.top_coords[:, 1])
            arc_length = np.sum(np.sqrt(dx*dx + dy*dy))
            target_n = max(2, int(np.floor(arc_length / config.interval)))
            if n_before >= target_n:
                return False

        dense_top = discretize_coords(
            self.top_coords,
            every=config.interval,
            num_segments=config.num_segments,
        )
        self.top_coords = dense_top
        x_new, y_new = dense_top[:, 0], dense_top[:, 1]
        loni, lati = self.xy2ll(x_new, y_new)
        self.top_coords_ll = np.vstack((loni, lati, dense_top[:, 2])).T

        n_dense = dense_top.shape[0]

        if not top_only:
            if self.bottom_coords is not None and self.bottom_coords.shape[0] != n_dense:
                dense_bot = discretize_coords(
                    self.bottom_coords,
                    num_segments=n_dense,
                )
                self.bottom_coords = dense_bot
                bx, by = dense_bot[:, 0], dense_bot[:, 1]
                blon, blat = self.xy2ll(bx, by)
                self.bottom_coords_ll = np.vstack((blon, blat, dense_bot[:, 2])).T

            if hasattr(self, 'layers') and self.layers is not None:
                for idx, layer in enumerate(self.layers):
                    if layer is not None and layer.shape[0] != n_dense:
                        dense_layer = discretize_coords(layer, num_segments=n_dense)
                        self.layers[idx] = dense_layer
                        lx, ly = dense_layer[:, 0], dense_layer[:, 1]
                        llon, llat = self.xy2ll(lx, ly)
                        self.layers_ll[idx] = np.vstack((llon, llat, dense_layer[:, 2])).T

        import logging
        logging.getLogger(__name__).debug(
            "densify_edges: top_coords %d -> %d points%s",
            n_before, n_dense, " (top_only)" if top_only else "",
        )

        if getattr(self, 'verbose', False) and not getattr(self, '_densify_first_run_logged', False):
            print(f"[{self.name}] densify_edges: {n_before} -> {n_dense} points (first iteration confirmed)")
            self._densify_first_run_logged = True

        return True

    # --- Geometry status reporting --------------------------------------------
    @staticmethod
    def _format_azimuth(deg):
        """Format azimuth degrees as quadrant bearing + numeric value."""
        deg = deg % 360
        if deg <= 90:
            return f"N{deg:.1f}E ({deg:.1f})"
        elif deg <= 180:
            return f"S{180 - deg:.1f}E ({deg:.1f})"
        elif deg <= 270:
            return f"S{deg - 180:.1f}W ({deg:.1f})"
        else:
            return f"N{360 - deg:.1f}W ({deg:.1f})"

    def geometry_summary(self):
        """Print a structured summary of the current fault geometry state.

        Shows control-point counts, average strike azimuth, densification
        config, dip control points, and mesh info.  Useful for verifying
        setup before running inversion.

        The strike is computed from ``geometry_ref.top_coords`` (the frozen
        baseline) when available, otherwise from ``self.top_coords``.
        """
        name = getattr(self, 'name', '?')
        sep = f"{'=' * 16} Geometry Summary: {name} {'=' * 16}"
        print(sep)

        # --- Control points ---
        ref = self.geometry_ref
        if ref is not None and ref.top_coords is not None:
            tc = ref.top_coords
        else:
            tc = getattr(self, 'top_coords', None)
        if ref is not None and ref.bottom_coords is not None:
            bc = ref.bottom_coords
        else:
            bc = getattr(self, 'bottom_coords', None)

        n_top = tc.shape[0] if tc is not None else 0
        n_bot = bc.shape[0] if bc is not None else 0
        print(f"  Control points:    {n_top} (top) / {n_bot} (bottom)")

        # --- Strike ---
        if tc is not None and n_top >= 2:
            strike_deg = self.compute_strike(tc)
            sin_mean = np.mean(np.sin(np.radians(strike_deg)))
            cos_mean = np.mean(np.cos(np.radians(strike_deg)))
            avg_strike = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
            smin, smax = float(strike_deg.min() % 360), float(strike_deg.max() % 360)
            print(f"  Average strike:    {self._format_azimuth(avg_strike)}")
            print(f"  Strike range:      [{smin:.1f}, {smax:.1f}]")
        else:
            print(f"  Average strike:    N/A (< 2 points)")

        # --- Dip control points ---
        dcp = ref.dip_control_points if ref is not None else None
        if dcp is not None:
            print(f"  Dip ctrl points:   {len(dcp.x)} attached")
        else:
            print(f"  Dip ctrl points:   None")

        # --- Densification ---
        dcfg = ref.densification if ref is not None else None
        if dcfg is not None and dcfg.enabled:
            mode = f"num_segments={dcfg.num_segments}" if dcfg.num_segments else f"interval={dcfg.interval} km"
            print(f"  Densification:     ACTIVE  {mode}")
            print(f"    -> Physics/mesh will use dense points per iteration")
        elif dcfg is not None and not dcfg.enabled:
            print(f"  Densification:     DISABLED")
        else:
            print(f"  Densification:     None")

        # --- Layers ---
        layers = ref.layers if ref is not None else getattr(self, 'layers', None)
        if layers is not None:
            print(f"  Layers:            {len(layers)}")
        else:
            print(f"  Layers:            None")

        # --- Mesh ---
        verts = getattr(self, 'Vertices', None)
        faces = getattr(self, 'Faces', None)
        if verts is not None and faces is not None:
            print(f"  Mesh:              {verts.shape[0]} vertices, {faces.shape[0]} faces")
        else:
            print(f"  Mesh:              not built")

        print("=" * len(sep))

    def get_mesh_params(self):
        """Return parameters from the last mesh generation call as a dict.

        The returned dict is suitable for passing directly to
        ``inv.set_method_parameters(fault_name, 'update_mesh', fault.get_mesh_params())``
        to keep the config in sync with the script-level rebuild.

        Returns ``None`` if no mesh generation method has been called yet.
        Debug-only parameters (``show``, ``verbose``, ``debug_plot``) are excluded.

        Bayesian-forbidden keys (e.g. ``remap``, ``use_current_mesh``,
        ``bottom_norm_offset``) are stripped so the returned dict is safe
        for iterative replay during Bayesian sampling.
        """
        if self._last_mesh_params is None:
            return None
        params = copy.deepcopy(self._last_mesh_params)
        method = params.get('method')
        if method:
            forbidden = mesh_registry.get_bayesian_forbidden(method)
            for key in forbidden:
                params.pop(key, None)
        return params

    def record_mesh_call(self, method_name, params, *, source='method'):
        """Record a mesh generation call with unified schema.

        Parameters
        ----------
        method_name : str
            Name of the mesh method (e.g. 'generate_simple_mesh').
        params : dict
            All parameters passed to the method (excluding debug params).
        source : str
            Origin of the call: 'method' (direct) or 'pipeline'.
        """
        meta = mesh_registry.get_meta(method_name)
        replayable_keys = mesh_registry.get_replayable_keys(method_name)

        if replayable_keys is not None:
            replayable = {k: params[k] for k in replayable_keys if k in params}
        else:
            replayable = {k: v for k, v in params.items()}

        replayable['method'] = method_name

        if 'mesh_func' in replayable and callable(replayable['mesh_func']):
            import warnings
            warnings.warn(
                f"mesh_func is callable in '{method_name}', "
                "cannot serialize to YAML config; degrading to True.",
                stacklevel=3,
            )
            replayable['mesh_func'] = True

        self._last_mesh_params = copy.deepcopy(replayable)
        self._last_mesh_raw_call = copy.deepcopy({
            'method': method_name,
            'source': source,
            'params': params,
        })

    def get_last_mesh_call(self):
        """Return the recorded mesh call parameters including source, or None."""
        if self._last_mesh_raw_call is None:
            return None
        return copy.deepcopy(self._last_mesh_raw_call)

    def help(self, method_name=None):
        """
        Print available perturbation methods and their configuration kwargs.

        Args:
            method_name: If provided, prints detailed help for this method
                         including all kwargs with defaults and a YAML config snippet.
                         The ``perturb_`` prefix is optional.
                         If None, prints an overview of all methods.
        """
        info = PerturbationRegistry.get_help(self)

        if method_name is not None:
            self._help_method_detail(method_name, info)
            return

        print(f"\n{'='*20} Fault Class: {self.__class__.__name__} {'='*20}")

        if not info:
            print("  (No registered methods found. Please check decorators.)")
            print("=" * 60)
            return

        # Group methods by family
        from collections import OrderedDict
        families = OrderedDict()
        for method, meta in info.items():
            family = meta.get('family', 'Other')
            families.setdefault(family, []).append((method, meta))

        total = len(info)
        print(f"Available Perturbation Methods ({total} total):\n")

        for family, methods in families.items():
            print(f"  [{family}] ({len(methods)} methods)")
            for method, meta in methods:
                desc = meta.get('description', 'N/A')
                params = meta.get('params', {})
                kwargs = meta.get('kwargs', {})
                flags = meta.get('flags', {})

                print(f"    * {method}")
                print(f"        Description: {desc}")
                # Perturbation params
                if isinstance(params, dict) and 'perturbations' in params:
                    print(f"        Perturbations: {params['perturbations']}")
                elif params:
                    print(f"        Parameters: {params}")
                # Update flags
                active_flags = [k for k, v in flags.items() if v] if flags else []
                if active_flags:
                    print(f"        Triggers update: {', '.join(active_flags)}")
                # Config kwargs summary
                if kwargs:
                    parts = []
                    for k, v in kwargs.items():
                        if v is inspect.Parameter.empty:
                            parts.append(k)
                        else:
                            parts.append(f"{k}={v!r}")
                    print(f"        Config kwargs: {', '.join(parts)}")
            print()
        print(f"  Tip: Call help('method_name') for detailed kwargs and YAML config example.")
        print("=" * 60)

    def _help_method_detail(self, method_name, info):
        """Print detailed help for a single perturbation method."""
        # Allow omitting the perturb_ prefix
        if not method_name.startswith('perturb_'):
            method_name_full = 'perturb_' + method_name
        else:
            method_name_full = method_name

        if method_name_full not in info:
            print(f"Method '{method_name}' not found.")
            close = [m for m in info if method_name.lower() in m.lower()]
            if close:
                print(f"Did you mean: {', '.join(close)}")
            return

        meta = info[method_name_full]
        desc = meta.get('description', 'N/A')
        params = meta.get('params', {})
        kwargs = meta.get('kwargs', {})
        family = meta.get('family', 'Other')
        flags = meta.get('flags', {})

        print(f"\n{'='*60}")
        print(f"Method: {method_name_full}")
        print(f"Family: [{family}]")
        print(f"Description: {desc}")

        # Perturbation info
        if isinstance(params, dict) and 'perturbations' in params:
            print(f"Perturbations: {params['perturbations']}")
        elif params:
            print(f"Parameters: {params}")

        # Update flags
        active_flags = [k for k, v in flags.items() if v] if flags else []
        if active_flags:
            print(f"Triggers update: {', '.join(active_flags)}")

        # Full kwargs listing
        if kwargs:
            print(f"\nConfiguration kwargs ({len(kwargs)}):")
            max_len = max(len(k) for k in kwargs)
            for k, v in kwargs.items():
                if v is inspect.Parameter.empty:
                    print(f"  {k:<{max_len}}  (required)")
                else:
                    print(f"  {k:<{max_len}} = {v!r}")

        # YAML config snippet
        print(f"\nYAML config snippet (method_parameters.update_fault_geometry):")
        print(f"  method_parameters:")
        print(f"    update_fault_geometry:")
        print(f"      method: {method_name_full}")
        for k, v in kwargs.items():
            if v is inspect.Parameter.empty:
                print(f"      {k}:   # <required>")
            elif v is None:
                print(f"      # {k}: null")
            elif isinstance(v, bool):
                print(f"      # {k}: {str(v).lower()}")
            elif isinstance(v, (int, float)):
                print(f"      # {k}: {v}")
            elif isinstance(v, str):
                print(f"      # {k}: '{v}'")
            else:
                print(f"      # {k}: {v!r}")
        print(f"\n  (Lines starting with '#' use default values; uncomment to override.)")

        # Current geometry status
        ref = getattr(self, 'geometry_ref', None)
        if ref is not None and ref.top_coords is not None:
            tc = ref.top_coords
            n_top = tc.shape[0]
            print(f"\nCurrent geometry status:")
            print(f"  Control points:  {n_top} (top)")
            dcfg = ref.densification
            if dcfg is not None and dcfg.enabled:
                mode = f"num_segments={dcfg.num_segments}" if dcfg.num_segments else f"interval={dcfg.interval} km"
                print(f"  Densification:   ACTIVE ({mode})")
            else:
                print(f"  Densification:   NOT CONFIGURED")
            if n_top >= 2:
                strike_deg = self.compute_strike(tc)
                sin_m = np.mean(np.sin(np.radians(strike_deg)))
                cos_m = np.mean(np.cos(np.radians(strike_deg)))
                avg = np.degrees(np.arctan2(sin_m, cos_m)) % 360
                print(f"  Average strike:  {self._format_azimuth(avg)}")
        else:
            print(f"\nCurrent geometry status: geometry_ref not set -- call snapshot() first")

        print(f"{'='*60}")

    def perturb(self, method, **kwargs):
        """
        Dynamic dispatcher for perturbation methods.
        """
        if not method.startswith('perturb_'):
            raise ValueError("The method name must start with 'perturb_'")
        if 'perturbations' not in kwargs:
            raise ValueError("The 'perturbations' argument is required")

        # Search in general methods first
        perturb_method = self.perturbation_methods.get(method, None)
        
        # Fallback: Search in Bayesian methods
        if perturb_method is None:
            perturb_method = getattr(self, method, None)

        if perturb_method is None:
            # Gather available methods for error message
            available = list(self.perturbation_methods.keys()) + list(getattr(self, 'bayesian_perturbation_methods', {}).keys())
            raise ValueError(f"Method '{method}' not found. Available methods: {available}")

        return perturb_method(**kwargs)

# =============================================================================
# 5. Shared Memory Mechanism (MPI Support)
# =============================================================================
SHARED_ATTRIBUTES = [
    'xf', 'yf', 'lon', 'lat', 'xi', 'yi', 'loni', 'lati', 'top', 'depth', 'z_patches',
    'top_coords', 'top_coords_ll', 'layers', 'layers_ll',
    'bottom_coords', 'bottom_coords_ll', 'Faces', 'Vertices',
    'Vertices_ll', 'patch', 'patchll', 'area', 'GL'
]

class SharedFaultInfo:
    def __init__(self):
        for attr in SHARED_ATTRIBUTES:
            setattr(self, f"_{attr}", None)

def shared_property(attr_name):
    def getter(self): return getattr(self.shared_info, f"_{attr_name}")
    def setter(self, value): setattr(self.shared_info, f"_{attr_name}", value)
    return property(getter, setter)

# =============================================================================
# 6. The Core Base Class: BayesianTriFaultBase
# =============================================================================
class BayesianTriFaultBase(AdaptiveLayeredDipTriangularPatches, PerturbationBase):
    """
    Base class for Bayesian Triangular Faults.

    Inherits from:
      - AdaptiveLayeredDipTriangularPatches: For geometry, multi-layer dip, and mesh generation.
      - PerturbationBase: For state tracking and method dispatch logic.

    Features:
      - Shared memory attributes for MPI efficiency (see *MPI Shared Memory* below).
      - Integrated help system via ``help()`` / ``help('method_name')``.

    MPI Shared Memory
    -----------------
    In MPI-parallel Bayesian inversion, the fault's heavy array attributes
    (vertices, faces, patches, laplacian, …) are stored in a single
    ``SharedFaultInfo`` object.  All MPI processes hold a reference to the
    **same** ``SharedFaultInfo``, so memory is shared rather than duplicated.

    The ``role`` parameter controls this mechanism:

    +-------------------+--------------------------------------------+
    | role              | Behaviour                                  |
    +===================+============================================+
    | ``'standalone'``  | Each instance owns a private               |
    | (default)         | SharedFaultInfo — no sharing, simplest     |
    |                   | usage.                                     |
    +-------------------+--------------------------------------------+
    | ``'master'``      | Creates (or reuses) a SharedFaultInfo and  |
    |                   | writes geometry/mesh into it.  Shares the  |
    |                   | same object with follower copies.           |
    +-------------------+--------------------------------------------+
    | ``'follower'``    | Receives an existing SharedFaultInfo (via  |
    |                   | ``shared_info``); reads arrays from it and |
    |                   | skips all geometry/mesh recomputation.      |
    +-------------------+--------------------------------------------+

    Typical MPI workflow::

        # --- rank 0 (master) ---
        fault = BayesianTriFaultBase('main', role='master', ...)
        fault.set_edges_for_bayesian_optimization(...)
        fault.build_triangular_meshes(...)

        # --- rank 1..N (followers) ---
        worker = fault.copy_with_shared_info('worker')
        # worker already sees the master's arrays; no rebuild needed.

    Args:
        name:            Fault name identifier.
        utmzone:         UTM zone string (e.g. ``'11S'``). ``None`` = auto-detect.
        ellps:           Reference ellipsoid for pyproj (default ``'WGS84'``).
        lon0:            Reference longitude for local UTM projection.
        lat0:            Reference latitude  for local UTM projection.
        verbose:         If ``True``, print progress messages during init.
        role:            One of ``'standalone'``, ``'master'``, ``'follower'``.

                         - ``'standalone'`` (default): private SharedFaultInfo,
                           no geometry sharing.
                         - ``'master'``: owns the SharedFaultInfo and is
                           responsible for computing geometry/mesh/laplacian.
                         - ``'follower'``: reads geometry from the master's
                           SharedFaultInfo; update flags are pre-set to ``True``
                           so downstream code skips redundant rebuilds.
        shared_info:     An existing ``SharedFaultInfo`` instance to attach.

                         - For *master*: optional — a new one is created if
                           ``None``.
                         - For *follower*: **required** — must point to the
                           master's ``SharedFaultInfo``.

    .. deprecated::
        The old ``use_shared_info`` / ``is_active`` keyword arguments are
        still accepted but will emit a ``DeprecationWarning``.  Use ``role``
        instead.
    """
    def __init__(self, name: str, utmzone=None, ellps='WGS84', lon0=None, lat0=None,
                 verbose=True, role='standalone', shared_info=None, **kwargs):

        # Backward compatibility: translate old use_shared_info/is_active to role
        if 'use_shared_info' in kwargs or 'is_active' in kwargs:
            warnings.warn(
                "use_shared_info/is_active are deprecated. "
                "Use role='standalone'|'master'|'follower' instead.",
                DeprecationWarning, stacklevel=2,
            )
            _usi = kwargs.pop('use_shared_info', False)
            _ia = kwargs.pop('is_active', False)
            if _usi and _ia:
                role = 'master'
            elif _usi:
                role = 'follower'
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        _valid_roles = ('standalone', 'master', 'follower')
        if role not in _valid_roles:
            raise ValueError(f"role must be one of {_valid_roles}, got {role!r}")

        # 1. Bind shared properties to the class.
        #    This replaces normal instance attributes (Vertices, patch, area, …)
        #    with property descriptors that read/write through self.shared_info,
        #    enabling transparent memory sharing across MPI ranks.
        self._bind_shared_properties()

        # 2. Setup the SharedFaultInfo storage backend.
        #    - Standalone: private SharedFaultInfo.
        #    - Master: create a new SharedFaultInfo (or reuse the provided one).
        #    - Follower: will be overwritten in step 4 below with the master's.
        if role == 'master':
            self.shared_info = shared_info or SharedFaultInfo()
        else:
            self.shared_info = SharedFaultInfo()

        self.role = role
        self.geometry_master = None  # set by copy_with_shared_info for followers

        # 3. Initialize parent classes (geometry + perturbation state).
        AdaptiveLayeredDipTriangularPatches.__init__(self, name, utmzone=utmzone, ellps=ellps,
                                                     lon0=lon0, lat0=lat0, verbose=verbose)
        PerturbationBase.__init__(self)

        # 4. Follower-only post-init: attach the master's SharedFaultInfo and mark
        #    all update flags as True so that downstream code (e.g. GF assembly)
        #    does NOT attempt to rebuild geometry/mesh/laplacian on this rank.
        if role == 'follower':
            self.shared_info = shared_info
            assert self.shared_info is not None, (
                "shared_info is required for follower instances. "
                "Pass the master's SharedFaultInfo object."
            )
            self.geometry_updated = True
            self.mesh_updated = True
            self.area_updated = True
            self.laplacian_updated = True

    def _bind_shared_properties(self):
        """Dynamically bind shared property descriptors to the **class**.

        For every attribute name in ``SHARED_ATTRIBUTES``, a Python ``property``
        is installed on ``self.__class__`` (not the instance).  The property
        delegates reads/writes to ``self.shared_info._<attr>``, so all instances
        that share the same ``SharedFaultInfo`` transparently share the same data.

        This is called once per ``__init__``; a guard prevents double-binding
        when the property already exists in the MRO (e.g. after the first
        instantiation or in a subclass).
        """
        for attr in SHARED_ATTRIBUTES:
            # Check if property already exists to avoid double binding in MRO
            if not isinstance(getattr(self.__class__, attr, None), property):
                setattr(self.__class__, attr, shared_property(attr))

    # -- backward-compat properties (deprecated) --------------------------------
    @property
    def use_shared_info(self):
        """Deprecated — use ``role`` instead."""
        warnings.warn(
            "use_shared_info is deprecated, check role instead.",
            DeprecationWarning, stacklevel=2,
        )
        return self.role in ('master', 'follower')

    @use_shared_info.setter
    def use_shared_info(self, value):
        warnings.warn(
            "use_shared_info is deprecated, set role instead.",
            DeprecationWarning, stacklevel=2,
        )

    @property
    def is_active(self):
        """Deprecated — use ``role`` instead."""
        warnings.warn(
            "is_active is deprecated, check role instead.",
            DeprecationWarning, stacklevel=2,
        )
        return self.role != 'follower'

    @is_active.setter
    def is_active(self, value):
        warnings.warn(
            "is_active is deprecated, set role instead.",
            DeprecationWarning, stacklevel=2,
        )

    def copy_with_shared_info(self, name):
        """Create a lightweight follower copy that shares the master's heavy arrays.

        The caller is auto-promoted from ``'standalone'`` to ``'master'`` if
        needed.  The returned instance has ``role='follower'`` and points to the
        **same** ``SharedFaultInfo`` as ``self``.  This means all large arrays
        (Vertices, Faces, patch, area, GL, …) are shared — not duplicated —
        across MPI ranks.

        The frozen ``geometry_ref`` is also shallow-copied (safe because it is
        immutable).

        Args:
            name: A unique name for the follower copy (e.g. ``'worker_rank1'``).

        Returns:
            A new ``BayesianTriFaultBase`` (or subclass) instance configured as
            a follower.

        Example::

            worker = master_fault.copy_with_shared_info('worker_0')
            # worker.Vertices is master_fault.Vertices  → True (same object)
        """
        # Auto-promote standalone → master
        if self.role == 'standalone':
            self.role = 'master'

        copy = self.__class__(name=name, shared_info=self.shared_info, role='follower',
                              lon0=self.lon0, lat0=self.lat0, utmzone=self.utmzone,
                              ellps=self.ellps, verbose=False)
        # geometry_ref is frozen/immutable, so shallow copy is safe
        copy.geometry_ref = self.geometry_ref
        copy.geometry_master = self.name
        return copy