"""
Base Constraint Manager Module

Shared storage, constraint CRUD, caching, and properties for both
BLSE and SMC constraint managers.
"""

import warnings
from types import MappingProxyType

import numpy as np
import yaml
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path


class ConstraintManagerBase:
    """
    Base class for constraint and bounds management.

    Provides shared infrastructure used by both BLSE and SMC managers:
    - Constraint storage (inequality / equality dicts)
    - Combined constraint cache with lazy rebuild
    - CRUD operations: add / remove constraints
    - Properties: lb, ub, inequality_constraints, equality_constraints
    - YAML normalisation helper

    Subclasses must set ``self.verbose`` and ``self._bounds`` /
    ``self._bounds_config`` / ``self._inequality_constraints`` /
    ``self._equality_constraints`` / ``self._combined_cache`` in their
    own ``__init__`` (usually via ``_init_shared_storage``).
    """

    # ------------------------------------------------------------------
    # Rank helper (override in subclasses with MPI context)
    # ------------------------------------------------------------------

    def _get_parallel_rank(self) -> Optional[int]:
        """Return the MPI rank, or *None* when running single-process."""
        return None

    def _should_warn(self) -> bool:
        """True when this process should emit user-facing warnings."""
        rank = self._get_parallel_rank()
        return rank is None or rank == 0

    # ------------------------------------------------------------------
    # Shared initialiser helper (called by subclass __init__)
    # ------------------------------------------------------------------

    def _init_shared_storage(self):
        """Initialise constraint dicts, cache, and bounds-config holder.

        Subclasses are responsible for extending ``self._bounds`` with any
        backend-specific keys *after* calling this method.
        """
        self._bounds_config = None

        self._inequality_constraints = {}   # name -> constraint_dict
        self._equality_constraints = {}     # name -> constraint_dict

        self._combined_cache = {
            'inequality': {'A': None, 'b': None, 'valid': False},
            'equality':   {'A': None, 'b': None, 'valid': False},
        }

        # Bounds storage – common keys only.  Subclasses extend this dict.
        self._bounds = {
            'lb': None,
            'ub': None,
            'global': {'lb': None, 'ub': None},
            'strikeslip': {},
            'dipslip': {},
            'poly': {},
            'source': None,
            'config_file': None,
            'applied_time': None,
        }

    @staticmethod
    def _validate_rake_interval(fault_name: str, rake_limits):
        """Return a validated ``(min_rake, max_rake)`` pair in degrees.

        Linear rake inequalities represent one convex sector in ``(ss, ds)``
        space.  That sector must have a positive aperture no larger than
        180 degrees.  Wider ranges are non-convex and endpoints separated by
        360 degrees collapse to a line in the current half-plane formula.
        """
        values = np.asarray(rake_limits, dtype=float)
        if values.shape != (2,):
            raise ValueError(
                f"rake_angle for '{fault_name}' must be a two-value "
                f"[min_rake, max_rake] interval, got shape {values.shape}"
            )

        rake_start, rake_end = float(values[0]), float(values[1])
        aperture = (rake_end - rake_start) % 360.0
        tol = 1.0e-10
        if aperture <= tol:
            raise ValueError(
                f"rake_angle for '{fault_name}' has zero aperture after "
                "360-degree wrapping. Use fixed_rake for a single rake, or "
                "omit rake_angle if the rake should be unconstrained."
            )
        if aperture > 180.0 + tol:
            raise ValueError(
                f"rake_angle for '{fault_name}' spans {aperture:g} degrees. "
                "Linear rake constraints can only represent one convex "
                "sector with aperture <= 180 degrees; split the range or use "
                "direct strikeslip/dipslip bounds instead."
            )
        return rake_start, rake_end

    @staticmethod
    def _source_component_slices(fault, source_start: int, adapter=None):
        """Return global slices for each ordered source component block."""
        if adapter is not None:
            param_names = list(adapter.get_param_names())
            params_per_component = dict(adapter.get_n_params_per_component())
        else:
            from .source_adapters import FaultAdapter
            slipdir = FaultAdapter._canonicalize_slipdir(getattr(fault, 'slipdir', 'sd'))
            char_to_name = {
                's': 'strikeslip',
                'd': 'dipslip',
                't': 'tensile',
                'c': 'coupling',
            }
            param_names = [char_to_name[char] for char in slipdir if char in char_to_name]
            n_spatial = len(getattr(fault, 'patch', []))
            params_per_component = {name: n_spatial for name in param_names}

        offset = 0
        slices = {}
        for name in param_names:
            n_component = int(params_per_component[name])
            start = int(source_start) + offset
            slices[name] = slice(start, start + n_component)
            offset += n_component
        return slices

    @staticmethod
    def _rake_component_starts(fault, source_start: int, n_patch: int, adapter=None):
        """Return global starts for strike-slip and dip-slip component blocks."""
        component_slices = ConstraintManagerBase._source_component_slices(
            fault, source_start, adapter=adapter
        )
        param_names = list(component_slices.keys())
        required = {'strikeslip', 'dipslip'}
        missing = required.difference(param_names)
        if missing:
            missing_str = ', '.join(sorted(missing))
            raise ValueError(
                f"rake constraints for '{fault.name}' require strikeslip and "
                f"dipslip components; missing {missing_str}"
            )

        for name in required:
            n_component = component_slices[name].stop - component_slices[name].start
            if int(n_component) != int(n_patch):
                raise ValueError(
                    f"rake constraints for '{fault.name}' expected {n_patch} "
                    f"{name} parameters, got {n_component}"
                )

        return component_slices['strikeslip'].start, component_slices['dipslip'].start

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _invalidate_constraint_cache(self):
        """Invalidate combined-constraint caches after mutation."""
        self._combined_cache['inequality']['valid'] = False
        self._combined_cache['equality']['valid'] = False

    # ------------------------------------------------------------------
    # Read-only diagnostics
    # ------------------------------------------------------------------

    @classmethod
    def _readonly_snapshot(cls, value):
        """Return a recursive read-only copy for diagnostic views."""
        if isinstance(value, np.ndarray):
            array = value.copy()
            array.setflags(write=False)
            return array
        if isinstance(value, dict):
            return MappingProxyType({
                key: cls._readonly_snapshot(item)
                for key, item in value.items()
            })
        if isinstance(value, list):
            return tuple(cls._readonly_snapshot(item) for item in value)
        if isinstance(value, tuple):
            return tuple(cls._readonly_snapshot(item) for item in value)
        return value

    @staticmethod
    def _constraint_group_summary(groups: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
        """Summarise constraint groups without exposing dense matrices."""
        return {
            name: {
                'rows': int(group['A'].shape[0]),
                'cols': int(group['A'].shape[1]),
                'source': group.get('source'),
                'shape': tuple(group.get('shape', group['A'].shape)),
            }
            for name, group in groups.items()
        }

    def get_constraint_snapshot(self, include_matrices: bool = False) -> Dict[str, Any]:
        """Return a compact diagnostic snapshot of bounds and constraints.

        The constraint manager is the single writable source of truth.  Use
        ``set_*``, ``add_*constraint`` and ``remove_constraint`` methods to
        update state.  This method is intended for inspection and logging.
        """
        n_parameters = None
        if self._bounds.get('lb') is not None:
            n_parameters = int(len(self._bounds['lb']))
        elif self._bounds.get('ub') is not None:
            n_parameters = int(len(self._bounds['ub']))

        snapshot = {
            'bounds': {
                'has_lb': self._bounds.get('lb') is not None,
                'has_ub': self._bounds.get('ub') is not None,
                'n_parameters': n_parameters,
                'global': dict(self._bounds.get('global', {})),
                'source': self._bounds.get('source'),
                'config_file': self._bounds.get('config_file'),
            },
            'inequality_constraints': self._constraint_group_summary(
                self._inequality_constraints
            ),
            'equality_constraints': self._constraint_group_summary(
                self._equality_constraints
            ),
        }
        if include_matrices:
            snapshot['inequality_constraints'] = self.inequality_constraints
            snapshot['equality_constraints'] = self.equality_constraints
        return snapshot

    # ------------------------------------------------------------------
    # YAML normalisation (static, 100 % identical in both subclasses)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_constraint_list(src_cfg):
        """Convert list-of-dicts or dict-of-dicts from YAML into ``{name: params}``.

        Accepts:
        - ``[{name: 'foo', type: 'inequality', rule: '...'}, ...]``  (list form)
        - ``{foo: {type: 'inequality', rule: '...'}, ...}``           (dict form)
        """
        if isinstance(src_cfg, list):
            result = {}
            for item in src_cfg:
                cname = item.get('name', f'constraint_{len(result)}')
                result[cname] = {k: v for k, v in item.items() if k != 'name'}
            return result
        elif isinstance(src_cfg, dict):
            result = {}
            for key, val in src_cfg.items():
                if key == 'type':
                    continue
                if key == 'constraints' and isinstance(val, list):
                    return ConstraintManagerBase._normalise_constraint_list(val)
                if isinstance(val, dict):
                    result[key] = val
            return result
        return {}

    # ------------------------------------------------------------------
    # Constraint CRUD  (default implementation – no mode guard)
    # ------------------------------------------------------------------

    def add_inequality_constraint(self, A: np.ndarray, b: np.ndarray, name: str,
                                  source: str = "manual", overwrite: bool = False):
        """Add inequality constraint *A @ x ≤ b*.

        Parameters
        ----------
        A : array (m, n)
        b : array (m,)
        name : str   – unique identifier
        source : str – provenance tag
        overwrite : bool – replace existing constraint with the same *name*
        """
        if name in self._inequality_constraints and not overwrite:
            raise ValueError(
                f"Inequality constraint '{name}' already exists. "
                "Use overwrite=True to replace."
            )

        A = np.asarray(A)
        b = np.asarray(b)

        if A.ndim != 2:
            raise ValueError(f"Constraint matrix A must be 2D, got shape {A.shape}")
        if A.shape[0] != len(b):
            raise ValueError(f"A.shape[0] ({A.shape[0]}) != len(b) ({len(b)})")

        self._inequality_constraints[name] = {
            'A': A.copy(),
            'b': b.copy(),
            'source': source,
            'shape': A.shape,
            'added_time': datetime.now(),
        }
        self._invalidate_constraint_cache()

        if self.verbose:
            print(f"[INQ] Added inequality constraint '{name}': "
                  f"{A.shape[0]} constraints (source: {source})")

    def add_equality_constraint(self, A: np.ndarray, b: np.ndarray, name: str,
                                source: str = "manual", overwrite: bool = False):
        """Add equality constraint *A @ x = b*."""
        if name in self._equality_constraints and not overwrite:
            raise ValueError(
                f"Equality constraint '{name}' already exists. "
                "Use overwrite=True to replace."
            )

        A = np.asarray(A)
        b = np.asarray(b)

        if A.ndim != 2:
            raise ValueError(f"Constraint matrix A must be 2D, got shape {A.shape}")
        if A.shape[0] != len(b):
            raise ValueError(f"A.shape[0] ({A.shape[0]}) != len(b) ({len(b)})")

        self._equality_constraints[name] = {
            'A': A.copy(),
            'b': b.copy(),
            'source': source,
            'shape': A.shape,
            'added_time': datetime.now(),
        }
        self._invalidate_constraint_cache()

        if self.verbose:
            print(f"[EQ] Added equality constraint '{name}': "
                  f"{A.shape[0]} constraints (source: {source})")

    def remove_constraint(self, name: str, constraint_type: Optional[str] = None):
        """Remove a named constraint.

        Parameters
        ----------
        name : str
        constraint_type : ``'inequality'``, ``'equality'``, or ``None`` (try both)
        """
        removed = False

        if constraint_type in (None, 'inequality') and name in self._inequality_constraints:
            del self._inequality_constraints[name]
            self._invalidate_constraint_cache()
            removed = True
            constraint_type = 'inequality'

        if constraint_type in (None, 'equality') and name in self._equality_constraints:
            del self._equality_constraints[name]
            self._invalidate_constraint_cache()
            removed = True
            constraint_type = 'equality'

        if removed:
            if self.verbose:
                print(f"[X] Removed {constraint_type} constraint: '{name}'")
        else:
            raise ValueError(f"Constraint '{name}' not found")

    def set_parameter_bounds_by_indices(self, indices, lower, upper, source: str = "manual"):
        """Set bounds for explicit model-vector indices.

        This is the low-level write path for helpers that resolve semantic
        parameters, such as data-correction transform components, to concrete
        solver columns.  ``indices`` are always in the full parameter vector
        used by the owning constraint manager.
        """
        initialize_bounds = getattr(self, '_initialize_bounds_arrays', None)
        if callable(initialize_bounds):
            initialize_bounds()

        lb_array = self._bounds.get('lb')
        ub_array = self._bounds.get('ub')
        if lb_array is None or ub_array is None:
            raise ValueError(
                "Bounds arrays are not initialized. Build parameter positions "
                "before setting index-based bounds."
            )

        index_array = np.asarray(indices, dtype=int).reshape(-1)
        if index_array.size == 0:
            raise ValueError("At least one parameter index is required")
        if np.unique(index_array).size != index_array.size:
            raise ValueError(f"Duplicate parameter indices are not allowed: {index_array.tolist()}")
        if np.any(index_array < 0) or np.any(index_array >= len(lb_array)):
            raise ValueError(
                f"Parameter index out of range for bounds vector of length {len(lb_array)}: "
                f"{index_array.tolist()}"
            )

        def _broadcast(values, name):
            array = np.asarray(values, dtype=float)
            if array.ndim == 0:
                return np.full(index_array.size, float(array), dtype=float)
            array = array.reshape(-1)
            if array.size == 1:
                return np.full(index_array.size, float(array[0]), dtype=float)
            if array.size != index_array.size:
                raise ValueError(
                    f"{name} must be a scalar or have {index_array.size} value(s), "
                    f"got {array.size}"
                )
            return array.astype(float, copy=False)

        lower_array = _broadcast(lower, "lower bounds")
        upper_array = _broadcast(upper, "upper bounds")
        if np.any(np.isnan(lower_array)) or np.any(np.isnan(upper_array)):
            raise ValueError("Bounds cannot contain NaN")
        if np.any(lower_array > upper_array):
            bad = np.where(lower_array > upper_array)[0]
            raise ValueError(
                "Lower bound is greater than upper bound for parameter "
                f"indices {index_array[bad].tolist()}"
            )

        self._bounds['lb'][index_array] = lower_array
        self._bounds['ub'][index_array] = upper_array
        self._bounds['source'] = source
        self._bounds['applied_time'] = datetime.now()

    # ------------------------------------------------------------------
    # Combined constraints (DRY helper from SMC pattern)
    # ------------------------------------------------------------------

    def _deduplicate_equality_constraints(
        self, A: np.ndarray, b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Deduplicate equality rows; raise on conflicting b for same A-row."""
        if A.shape[0] <= 1:
            return A, b

        b_col = b.reshape(-1, 1)
        augmented = np.hstack([A, b_col])
        rounded = np.round(augmented, decimals=12)

        _, unique_indices = np.unique(rounded, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)

        n_removed = A.shape[0] - len(unique_indices)
        if n_removed > 0:
            A_rounded = np.round(A, decimals=12)
            _, A_unique_idx, A_inverse = np.unique(
                A_rounded, axis=0, return_index=True, return_inverse=True
            )
            for group_id in range(len(A_unique_idx)):
                member_mask = (A_inverse == group_id)
                if np.sum(member_mask) <= 1:
                    continue
                b_values = np.round(b[member_mask], decimals=12)
                if not np.all(b_values == b_values[0]):
                    conflict_indices = np.where(member_mask)[0]
                    raise ValueError(
                        f"Conflicting equality constraints detected: "
                        f"rows {conflict_indices.tolist()} have identical A-row "
                        f"but different b values: {b[member_mask].tolist()}. "
                        f"Remove or reconcile these constraints before solving."
                    )

            if self._should_warn():
                warnings.warn(
                    f"Removed {n_removed} duplicate equality constraint row(s) "
                    f"from {A.shape[0]} total. "
                    f"{len(unique_indices)} unique rows remain.",
                    RuntimeWarning,
                    stacklevel=3,
                )

        return A[unique_indices], b[unique_indices]

    def _get_combined_constraints(self, constraint_type: str):
        """Return ``(A, b)`` for the given *constraint_type* with caching."""
        constraints = getattr(self, f'_{constraint_type}_constraints')
        if not constraints:
            return None, None

        cache = self._combined_cache[constraint_type]
        if cache['valid']:
            return (cache['A'].copy() if cache['A'] is not None else None,
                    cache['b'].copy() if cache['b'] is not None else None)

        A_list = [c['A'] for c in constraints.values()]
        b_list = [c['b'] for c in constraints.values()]

        A_combined = np.vstack(A_list)
        b_combined = np.hstack(b_list)

        if constraint_type == 'equality':
            A_combined, b_combined = self._deduplicate_equality_constraints(
                A_combined, b_combined
            )

        cache['A'] = A_combined.copy()
        cache['b'] = b_combined.copy()
        cache['valid'] = True

        return A_combined, b_combined

    def get_combined_inequality_constraints(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get combined inequality constraints as ``(A, b)``."""
        return self._get_combined_constraints('inequality')

    def get_combined_equality_constraints(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get combined equality constraints as ``(A, b)``."""
        return self._get_combined_constraints('equality')

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    def load_bounds_config(self, config_file: str, encoding: str = 'utf-8'):
        """Load bounds configuration from a YAML file.

        Subclasses may override ``_on_bounds_config_loaded`` to propagate
        the parsed config to their backend object.
        """
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Bounds config file not found: {config_file}")

            with open(config_path, 'r', encoding=encoding) as f:
                if config_path.suffix.lower() in ('.yml', '.yaml'):
                    self._bounds_config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")

            self._bounds['config_file'] = str(config_path)

            # Subclass hook
            self._on_bounds_config_loaded()

            if self.verbose:
                print(f"[DIR] Loaded bounds config from: {config_file}")
                self._print_config_summary()

        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to load bounds config: {e}")
            raise

    def _on_bounds_config_loaded(self):
        """Hook called after ``_bounds_config`` is set.  Override in subclasses."""

    # ------------------------------------------------------------------
    # Config summary (template method)
    # ------------------------------------------------------------------

    def _print_config_summary(self):
        """Print summary of loaded configuration.

        Collects common keys, then calls ``_get_extra_config_summary_items()``
        so subclasses can append backend-specific keys without copy-pasting
        the entire method.
        """
        if not self._bounds_config:
            return

        config_items = []
        if 'lb' in self._bounds_config or 'ub' in self._bounds_config:
            config_items.append("global bounds")
        if 'strikeslip' in self._bounds_config:
            config_items.append(f"strike-slip bounds for {len(self._bounds_config['strikeslip'])} fault(s)")
        if 'dipslip' in self._bounds_config:
            config_items.append(f"dip-slip bounds for {len(self._bounds_config['dipslip'])} fault(s)")
        if 'poly' in self._bounds_config:
            config_items.append(f"polynomial bounds for {len(self._bounds_config['poly'])} fault(s)")
        if 'rake_angle' in self._bounds_config:
            config_items.append(f"rake angle constraints for {len(self._bounds_config['rake_angle'])} fault(s)")
        if 'source_bounds' in self._bounds_config:
            config_items.append(f"source component bounds for {len(self._bounds_config['source_bounds'])} source(s)")
        if 'source_constraints' in self._bounds_config:
            config_items.append(f"source constraints for {len(self._bounds_config['source_constraints'])} source(s)")

        # Subclass hook for extra keys (geometry, sigmas, alpha, etc.)
        config_items.extend(self._get_extra_config_summary_items())

        if config_items:
            print(f"   - Configuration contains: {', '.join(config_items)}")

    def _get_extra_config_summary_items(self) -> list:
        """Return extra config summary strings.  Override in subclasses."""
        return []

    # ------------------------------------------------------------------
    # Validation (common structure, subclasses extend)
    # ------------------------------------------------------------------

    def validate(self) -> Dict:
        """Validate current bounds and constraints.

        Returns ``{valid: bool, errors: [...], warnings: [...]}``.
        Subclasses should call ``super().validate()`` and extend the result.
        """
        result: Dict = {'valid': True, 'errors': [], 'warnings': []}

        # Bounds consistency: lb <= ub
        if self._bounds['lb'] is not None and self._bounds['ub'] is not None:
            inconsistent = np.where(self._bounds['lb'] > self._bounds['ub'])[0]
            if len(inconsistent) > 0:
                result['errors'].append(
                    f"Inconsistent bounds at {len(inconsistent)} parameter(s): "
                    f"indices {inconsistent[:10].tolist()}"
                    + ("..." if len(inconsistent) > 10 else "")
                )
                result['valid'] = False

        # Constraint dimension sanity
        for name, c in self._inequality_constraints.items():
            if c['A'].shape[0] != len(c['b']):
                result['errors'].append(
                    f"Inequality '{name}': A rows ({c['A'].shape[0]}) != len(b) ({len(c['b'])})")
                result['valid'] = False

        for name, c in self._equality_constraints.items():
            if c['A'].shape[0] != len(c['b']):
                result['errors'].append(
                    f"Equality '{name}': A rows ({c['A'].shape[0]}) != len(b) ({len(c['b'])})")
                result['valid'] = False

        # Check combined equality constraint rank (CVXOPT requires full row rank)
        if self._equality_constraints:
            try:
                Aeq, beq = self.get_combined_equality_constraints()
                if Aeq is not None:
                    rank = np.linalg.matrix_rank(Aeq)
                    if rank < Aeq.shape[0]:
                        result['valid'] = False
                        result['errors'].append(
                            f"Combined equality constraint matrix is rank-deficient: "
                            f"rank={rank}, rows={Aeq.shape[0]}. "
                            f"CVXOPT requires full row rank."
                        )
            except ValueError as e:
                result['valid'] = False
                result['errors'].append(str(e))

        return result

    # ------------------------------------------------------------------
    # Properties (backward compatibility)
    # ------------------------------------------------------------------

    @property
    def lb(self) -> Optional[np.ndarray]:
        """Lower bounds array."""
        return self._bounds['lb'].copy() if self._bounds['lb'] is not None else None

    @property
    def ub(self) -> Optional[np.ndarray]:
        """Upper bounds array."""
        return self._bounds['ub'].copy() if self._bounds['ub'] is not None else None

    @property
    def bounds(self):
        """Read-only diagnostic view of bounds.

        Update bounds through manager methods such as ``set_global_bounds`` or
        inversion-level helpers.  The returned mapping is not a write API.
        """
        return self._readonly_snapshot(self._bounds)

    @property
    def inequality_constraints(self) -> Dict[str, Dict]:
        """Read-only diagnostic view of inequality constraints."""
        return self._readonly_snapshot({
            name: {'A': c['A'], 'b': c['b'], 'source': c.get('source')}
            for name, c in self._inequality_constraints.items()
        })

    @property
    def equality_constraints(self) -> Dict[str, Dict]:
        """Read-only diagnostic view of equality constraints."""
        return self._readonly_snapshot({
            name: {'A': c['A'], 'b': c['b'], 'source': c.get('source')}
            for name, c in self._equality_constraints.items()
        })
