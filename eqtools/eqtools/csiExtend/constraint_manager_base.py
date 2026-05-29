"""
Base Constraint Manager Module

Shared storage, constraint CRUD, caching, and properties for both
BLSE and SMC constraint managers.
"""

import warnings

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

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _invalidate_constraint_cache(self):
        """Invalidate combined-constraint caches after mutation."""
        self._combined_cache['inequality']['valid'] = False
        self._combined_cache['equality']['valid'] = False

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

        if constraint_type == 'equality' and len(constraints) > 1:
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
        if len(self._equality_constraints) > 1:
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
    def inequality_constraints(self) -> Dict[str, Dict]:
        """Inequality constraints dict (name → {A, b})."""
        return {name: {'A': c['A'], 'b': c['b']}
                for name, c in self._inequality_constraints.items()}

    @property
    def equality_constraints(self) -> Dict[str, Dict]:
        """Equality constraints dict (name → {A, b})."""
        return {name: {'A': c['A'], 'b': c['b']}
                for name, c in self._equality_constraints.items()}
