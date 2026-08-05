"""Shared constraint-management infrastructure for BLSE and SMC backends.

The base manager owns declaration storage, resolved bounds, named linear
constraint groups, transaction snapshots, validation, and read-only
diagnostics. Backend subclasses provide parameter positions and decide which
constraint families are active for their numerical mode.

Two index spaces occur in Bayesian workflows: the full sampling vector and the
active linear subspace. Methods that create linear ``A`` matrices operate in
the active linear space exposed by the concrete manager; methods that write
``lb``/``ub`` use the bounds-vector space owned by that manager.
"""

import copy
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from types import MappingProxyType

import numpy as np
import yaml
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path


class ConstraintManagerBase:
    """Base class for constraint and bounds management.

    Provides shared infrastructure used by both BLSE and SMC managers:

    - constraint storage for named inequality and equality groups;
    - lazily combined ``A``/``b`` matrices;
    - global, source, component, index, and patch bounds declarations;
    - config-owned versus runtime-owned declaration reconciliation;
    - atomic update/rollback helpers and read-only diagnostic snapshots.

    Subclasses must set ``self.verbose`` and ``self._bounds`` /
    ``self._bounds_config`` / ``self._inequality_constraints`` /
    ``self._equality_constraints`` / ``self._combined_cache`` in their
    own ``__init__`` (usually via ``_init_shared_storage``).

    Backend hooks fall into two categories:

    - full-linear defaults that BLSE/VCE inherit, such as
      :meth:`_get_linear_matrix_source_start`;
    - required backend implementations that deliberately raise
      ``NotImplementedError`` in this class, such as
      :meth:`_fault_exists`, :meth:`_get_source_type`,
      :meth:`_initialize_bounds_arrays`, and
      :meth:`_rebuild_rake_constraints`.

    A sampled backend whose linear matrices address only a subspace must
    override both active-linear coordinate hooks as one pair. It must not rely
    on the full-linear defaults.
    """

    _RESERVED_MANAGED_GROUP_NAMES = frozenset({
        'rake_sector',
        'fixed_rake',
    })

    # These bounds-config sections are always source-name mappings.  YAML
    # represents a bare key such as ``source_bounds:`` as None; treating that
    # spelling as an empty mapping keeps optional sections equivalent whether
    # users omit them, write ``{}``, or leave a generated example commented.
    _BOUNDS_CONFIG_MAPPING_FIELDS = (
        'geometry',
        'slip_magnitude',
        'rake_angle',
        'strikeslip',
        'dipslip',
        'poly',
        'source_bounds',
        'source_constraints',
    )

    # ------------------------------------------------------------------
    # Rank helper (override in subclasses with MPI context)
    # ------------------------------------------------------------------

    def _get_parallel_rank(self) -> Optional[int]:
        """Return the MPI rank used for user-facing diagnostics.

        Returns
        -------
        int or None
            MPI rank, or ``None`` for a single-process manager.
        """
        return None

    def _should_warn(self) -> bool:
        """Return whether this process should emit user-facing warnings.

        Returns
        -------
        bool
            ``True`` for a serial process or MPI rank zero.
        """
        rank = self._get_parallel_rank()
        return rank is None or rank == 0

    # ------------------------------------------------------------------
    # Shared initialiser helper (called by subclass __init__)
    # ------------------------------------------------------------------

    def _init_shared_storage(self):
        """Initialise constraint dicts, cache, and bounds-config holder.

        Subclasses are responsible for extending ``self._bounds`` with any
        backend-specific keys *after* calling this method.

        Returns
        -------
        None

        Notes
        -----
        This initializes declaration and resolved-state containers only; it
        does not allocate backend-specific bounds arrays or load a config file.
        ``_state_revision`` advances only after resolved state changes.
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
            'source_bounds': {},
            'parameter_bounds': {},
            'patch_constraints': {},
            'source': None,
            'config_file': None,
            'applied_time': None,
        }
        self._runtime_rake_limits = {}
        self._runtime_patch_constraints = []
        self._last_source_constraint_report = {
            'scope': 'source_constraints',
            'applied': [],
            'inactive': [],
            'ignored': [],
            'generated_rows': {'inequality': 0, 'equality': 0},
        }
        self._last_reconciled_flags = None
        self._state_revision = 0
        self._bounds_rebuild_depth = 0
        self._constraint_transaction_depth = 0
        self._constraint_transaction_sync_pending = False

    @staticmethod
    def _validate_rake_interval(fault_name: str, rake_limits):
        """Return a validated ``(min_rake, max_rake)`` pair in degrees.

        Linear rake inequalities represent one convex sector in ``(ss, ds)``
        space.  That sector must have a positive aperture no larger than
        180 degrees.  Wider ranges are non-convex and endpoints separated by
        360 degrees collapse to a line in the current half-plane formula.

        Parameters
        ----------
        fault_name : str
            Fault or rule label used in validation messages.
        rake_limits : array-like of float
            Two endpoints ``[rake_min, rake_max]`` in degrees.

        Returns
        -------
        tuple of float
            Validated endpoints, preserving the user's angular representation.

        Raises
        ------
        ValueError
            If the input is not a finite two-value interval, has zero aperture
            after 360-degree wrapping, or spans more than 180 degrees.

        Notes
        -----
        The interval represents one convex sector. A single exact rake belongs
        in the fixed-rake equality interface rather than this helper.
        """
        values = np.asarray(rake_limits, dtype=float)
        if values.shape != (2,):
            raise ValueError(
                f"rake_angle for '{fault_name}' must be a two-value "
                f"[min_rake, max_rake] interval, got shape {values.shape}"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"rake_angle for '{fault_name}' must contain finite values"
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
    def _require_finite(values, label):
        """Convert a declaration to finite floating-point values.

        Parameters
        ----------
        values : array-like
            Scalar or array declaration to validate.
        label : str
            Human-readable field label used in exceptions.

        Returns
        -------
        numpy.ndarray
            Floating-point view or array produced by ``numpy.asarray``.

        Raises
        ------
        TypeError
            If ``values`` cannot be converted to numeric data.
        ValueError
            If any converted value is NaN or infinite.
        """
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} must be numeric") from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain finite values")
        return array

    @staticmethod
    def _source_component_slices(fault, source_start: int, adapter=None):
        """Resolve ordered component slices for one source parameter block.

        Parameters
        ----------
        fault : object
            Fault-like source. Its patch count and optional ``slipdir`` are
            used when no adapter is supplied.
        source_start : int
            Start column of the source in the manager's active parameter space.
        adapter : SourceAdapter, optional
            Authoritative component names and per-component parameter counts.

        Returns
        -------
        dict of str to slice
            Ordered component names mapped to half-open parameter slices.

        Notes
        -----
        Adapter order is authoritative. The fallback is restricted to
        Fault-like sources and canonicalizes ``slipdir`` through
        :class:`FaultAdapter`.
        """
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

    def _get_component_columns_for_patches(
        self,
        source_name,
        component,
        patch_indices,
        *,
        space,
    ):
        """Map Fault-local patch ids to one authoritative parameter space.

        Parameters
        ----------
        source_name : str
            Existing Fault source name.
        component : str
            Strike- or dip-slip component name/alias.
        patch_indices : array-like of int
            Fault-local patch ids.
        space : {"bounds", "active_linear"}
            Complete owning bounds vector or active equality/inequality matrix.

        Returns
        -------
        numpy.ndarray
            Columns in the requested space.
        """
        if not self._fault_exists(source_name):
            raise ValueError(f"Unknown source '{source_name}'")
        source_type = self._get_source_type(source_name)
        if source_type != 'Fault':
            raise TypeError(
                "Slip-component patch columns require a Fault source; "
                f"'{source_name}' is {source_type}"
            )
        if space == 'bounds':
            source_start = self._get_source_start(source_name)
        elif space == 'active_linear':
            source_start = self._get_linear_matrix_source_start(source_name)
        else:
            raise ValueError("space must be 'bounds' or 'active_linear'")

        fault = self._get_fault_by_name(source_name)
        adapter = self._get_adapter_for_source(source_name)
        component = self._normalize_fault_slip_component(component)
        component_slices = self._source_component_slices(
            fault,
            source_start,
            adapter=adapter,
        )
        if component not in component_slices:
            raise ValueError(
                f"Fault '{source_name}' has no {component} component"
            )

        indices = np.asarray(patch_indices, dtype=int).reshape(-1)
        n_component = (
            component_slices[component].stop
            - component_slices[component].start
        )
        if np.any(indices < 0) or np.any(indices >= n_component):
            raise ValueError(
                f"Invalid patch indices for fault '{source_name}'; expected "
                f"values in [0, {n_component - 1}]"
            )
        return component_slices[component].start + indices

    @staticmethod
    def _rake_component_starts(fault, source_start: int, n_patch: int, adapter=None):
        """Return strike-slip and dip-slip starts used by rake matrices.

        Parameters
        ----------
        fault : object
            Fault-like source being constrained.
        source_start : int
            Start column of the source in the active matrix space.
        n_patch : int
            Expected number of parameters in each slip component.
        adapter : SourceAdapter, optional
            Authoritative source-component description.

        Returns
        -------
        tuple of int
            ``(strikeslip_start, dipslip_start)``.

        Raises
        ------
        ValueError
            If either required component is absent or its parameter count does
            not equal ``n_patch``.
        """
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

    @staticmethod
    def _normalize_fault_slip_component(component):
        """Normalize a public strike/dip alias to its canonical component name.

        Parameters
        ----------
        component : object
            Component name or alias such as ``"ss"`` or ``"dip_slip"``.

        Returns
        -------
        {"strikeslip", "dipslip"}
            Canonical fault-slip component name.

        Raises
        ------
        ValueError
            If the alias does not identify strike slip or dip slip.
        """
        comp = str(component).lower().replace(' ', '').replace('_', '').replace('-', '')
        if comp in ('strikeslip', 'ss', 's', 'strike'):
            return 'strikeslip'
        if comp in ('dipslip', 'ds', 'd', 'dip'):
            return 'dipslip'
        raise ValueError(
            f"Unknown slip component '{component}'. Use 'strikeslip'/'ss' "
            "or 'dipslip'/'ds'."
        )

    def _get_fault_sequence(self):
        """Return the ordered source sequence owned by the concrete backend.

        Returns
        -------
        list
            A new list from ``solver.faults``, ``config.faults_list``, or
            ``multifaults.faults``. Returns an empty list if none is present.
        """
        if hasattr(self, 'solver') and hasattr(self.solver, 'faults'):
            return list(self.solver.faults)
        if hasattr(self, 'config') and hasattr(self.config, 'faults_list'):
            return list(self.config.faults_list)
        if hasattr(self, 'multifaults') and hasattr(self.multifaults, 'faults'):
            return list(self.multifaults.faults)
        return []

    def _get_fault_by_name(self, fault_name):
        """Look up one source in the backend's ordered source sequence.

        Parameters
        ----------
        fault_name : str
            Exact source name.

        Returns
        -------
        object
            Matching source object.

        Raises
        ------
        ValueError
            If no source has the requested name.
        """
        for fault in self._get_fault_sequence():
            if getattr(fault, 'name', None) == fault_name:
                return fault
        raise ValueError(
            f"Fault '{fault_name}' not found. Available: "
            f"{[getattr(f, 'name', None) for f in self._get_fault_sequence()]}"
        )

    def _fault_exists(self, fault_name):
        """Return whether a source exists in the concrete backend.

        Parameters
        ----------
        fault_name : str
            Source name to test.

        Returns
        -------
        bool
            Whether the backend owns the source.

        Raises
        ------
        NotImplementedError
            The concrete backend must define its authoritative source
            registry.

        Notes
        -----
        BLSE/VCE resolves names against ``solver.faults``. SMC resolves them
        against ``config.faults_list``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _fault_exists()"
        )

    def _get_source_type(self, source_name):
        """Return the semantic source type used by constraint compilers.

        Parameters
        ----------
        source_name : str
            Existing source name.

        Returns
        -------
        str
            Adapter/source type such as ``"Fault"``.

        Raises
        ------
        NotImplementedError
            The concrete backend must define how source metadata is resolved.

        Notes
        -----
        Callers that accept user-provided names must check
        :meth:`_fault_exists` first. Rake and patch constraints are restricted
        to ``Fault`` sources.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _get_source_type()"
        )

    def _get_adapter_for_source(self, source_name):
        """Return the registered adapter for a source.

        Parameters
        ----------
        source_name : str
            Source name used as the adapter mapping key.

        Returns
        -------
        SourceAdapter or None
            Backend adapter, or ``None`` when no adapter mapping is available.
        """
        if hasattr(self, 'solver') and hasattr(self.solver, 'adapters'):
            return self.solver.adapters.get(source_name)
        if hasattr(self, 'multifaults') and hasattr(self.multifaults, 'adapters'):
            return self.multifaults.adapters.get(source_name)
        return None

    def _get_source_start(self, source_name):
        """Return a source start in the manager-owned bounds-vector space.

        Parameters
        ----------
        source_name : str
            Source whose first parameter is requested.

        Returns
        -------
        int
            First source position in the complete BLSE linear vector or the
            complete SMC sampling vector. BLSE may fall back to
            ``fault_indexes`` for inherited CSI compatibility.

        Raises
        ------
        AttributeError
            If the manager exposes no supported position mapping.
        KeyError
            If a position mapping exists but omits ``source_name``.

        Notes
        -----
        This method does not return an ``A``-matrix-local column for sampled
        backends. Bounds and patch selectors use this owning-vector position.
        Linear matrix compilers must use
        :meth:`_get_linear_matrix_source_start`, which SMC overrides to remove
        the nonlinear prefix.
        """
        if hasattr(self, 'solver') and hasattr(self.solver, 'slip_positions'):
            return int(self.solver.slip_positions[source_name][0])
        if hasattr(self, 'solver') and hasattr(self.solver, 'fault_indexes'):
            return int(self.solver.fault_indexes[source_name][0])
        if hasattr(self, 'slip_positions'):
            return int(self.slip_positions[source_name][0])
        raise AttributeError("constraint manager has no slip_positions information")

    def _get_linear_matrix_source_start(self, source_name):
        """Return a source start in the active linear-matrix space.

        Parameters
        ----------
        source_name : str
            Source whose first active linear column is requested.

        Returns
        -------
        int
            First source column in every active equality/inequality matrix.

        Notes
        -----
        This is the full-linear default inherited by BLSE/VCE, where bounds
        and matrices share one coordinate space. A sampled backend with a
        nonlinear prefix must override this method together with
        :meth:`_get_linear_matrix_n_parameters`.
        """
        return self._get_source_start(source_name)

    def _get_linear_matrix_n_parameters(self):
        """Return the number of columns in every active linear matrix.

        Returns
        -------
        int
            BLSE/VCE ``solver.lsq_parameters`` in the base implementation.

        Raises
        ------
        NotImplementedError
            If the manager does not expose a full-linear solver width. Sampled
            backends must override this hook rather than use their full sample
            count.

        Notes
        -----
        ``mcmc_samples`` is intentionally not a fallback: it includes the
        nonlinear prefix in SMC_FJ and would silently produce a wrong matrix
        width.
        """
        if hasattr(self, 'solver') and hasattr(self.solver, 'lsq_parameters'):
            return int(self.solver.lsq_parameters)
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "_get_linear_matrix_n_parameters()"
        )

    def _initialize_bounds_arrays(self):
        """Allocate backend-owned ``lb``/``ub`` arrays when absent.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            The concrete backend must obtain the authoritative vector width
            from its own parameter layout.

        Notes
        -----
        BLSE/VCE allocates the complete linear vector. SMC allocates the
        complete sampled vector, including nonlinear and linear parameters.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "_initialize_bounds_arrays()"
        )

    def _rebuild_resolved_bounds(self, source="resolved_declarations"):
        """Compile stored declarations into backend-owned ``lb``/``ub``.

        Concrete managers must implement this hook because bounds vector
        width, coordinate space, and precedence are backend-specific.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "_rebuild_resolved_bounds()"
        )

    def _rebuild_rake_constraints(self, additional_rake_limits=None):
        """Compile current rake declarations into the managed sector group.

        Parameters
        ----------
        additional_rake_limits : mapping, optional
            Call-specific fault-level declarations merged according to the
            concrete backend's documented precedence.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Matrix construction and lifecycle reconciliation are
            backend-specific.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "_rebuild_rake_constraints()"
        )

    @staticmethod
    def _empty_source_constraint_report():
        """Create an empty source-constraint audit report.

        Returns
        -------
        dict
            Fresh report with ``applied``, ``inactive``, ``ignored``, and row
            count containers. No nested mutable object is shared between calls.
        """
        return {
            'scope': 'source_constraints',
            'applied': [],
            'inactive': [],
            'ignored': [],
            'generated_rows': {'inequality': 0, 'equality': 0},
        }

    @staticmethod
    def _describe_unconsumed_source_rules(source_name, raw_constraints):
        """Describe rules that cannot be consumed by the current backend.

        Parameters
        ----------
        source_name : str
            Source owning the declarations.
        raw_constraints : object
            Raw list/mapping from ``source_constraints``.

        Returns
        -------
        list of dict
            Best-effort descriptors containing source, name, type, and rule
            text. Malformed entries are represented instead of raised.

        Notes
        -----
        Adapter grammar is deliberately not invoked here so unavailable or
        inactive source declarations remain visible in diagnostic reports.
        """
        if (
            isinstance(raw_constraints, Mapping)
            and isinstance(raw_constraints.get('constraints'), list)
        ):
            raw_constraints = raw_constraints['constraints']

        entries = []
        if isinstance(raw_constraints, list):
            iterable = enumerate(raw_constraints)
            for index, raw in iterable:
                if isinstance(raw, Mapping):
                    entries.append({
                        'source': str(source_name),
                        'name': str(raw.get('name', f'constraint_{index}')),
                        'type': str(raw.get('type', 'inequality')).lower(),
                        'rule': str(raw.get('rule', '')),
                    })
                else:
                    entries.append({
                        'source': str(source_name),
                        'name': f'constraint_{index}',
                        'type': None,
                        'rule': None,
                    })
        elif isinstance(raw_constraints, Mapping):
            for name, raw in raw_constraints.items():
                if name == 'type':
                    continue
                if isinstance(raw, Mapping):
                    entries.append({
                        'source': str(source_name),
                        'name': str(name),
                        'type': str(raw.get('type', 'inequality')).lower(),
                        'rule': str(raw.get('rule', '')),
                    })

        if not entries:
            entries.append({
                'source': str(source_name),
                'name': None,
                'type': None,
                'rule': None,
            })
        return entries

    def _apply_source_constraint_declarations(self):
        """Apply configured source rules one declaration at a time.

        The adapter remains the owner of source-specific rule grammar.  This
        orchestration layer provides uniform error handling and an audit report:
        valid rules are applied, declarations for unavailable sources/adapters
        are reported as ignored, and linear rules are reported as inactive in
        modes that do not consume linear constraints.

        Returns
        -------
        dict
            Audit report containing applied, inactive, and ignored rules plus
            generated inequality/equality row counts. A deep copy is retained
            as ``_last_source_constraint_report``.

        Raises
        ------
        ValueError
            If a declared type is unsupported or an adapter reports that the
            rule belongs to the opposite constraint family.

        Notes
        -----
        Generated groups are named ``src_<source>_<rule>`` and replace prior
        config-owned groups with the same name. Config reload callers remove
        stale config groups before invoking this method.
        """
        report = self._empty_source_constraint_report()
        configured = (
            (self._bounds_config or {}).get('source_constraints') or {}
        )
        if not configured:
            self._last_source_constraint_report = report
            return report

        active = self._linear_constraints_active()
        n_total = (
            self._get_linear_matrix_n_parameters() if active else None
        )
        for source_name, raw_constraints in configured.items():
            if not active:
                report['inactive'].extend({
                    **item,
                    'reason': 'linear_constraints_inactive',
                } for item in self._describe_unconsumed_source_rules(
                    source_name, raw_constraints
                ))
                continue
            if not self._fault_exists(source_name):
                report['ignored'].extend({
                    **item,
                    'reason': 'unknown_source',
                } for item in self._describe_unconsumed_source_rules(
                    source_name, raw_constraints
                ))
                continue
            adapter = self._get_adapter_for_source(source_name)
            if adapter is None:
                report['ignored'].extend({
                    **item,
                    'reason': 'missing_adapter',
                } for item in self._describe_unconsumed_source_rules(
                    source_name, raw_constraints
                ))
                continue

            constraints = self._normalise_constraint_list(raw_constraints)
            for constraint_name, params in constraints.items():
                params = copy.deepcopy(params)
                constraint_type = str(
                    params.get('type', 'inequality')
                ).lower()
                rule = str(params.get('rule', ''))
                item = {
                    'source': str(source_name),
                    'name': str(constraint_name),
                    'type': constraint_type,
                    'rule': rule,
                }

                if constraint_type not in ('inequality', 'equality'):
                    raise ValueError(
                        f"source_constraints.{source_name}.{constraint_name}.type "
                        "must be 'inequality' or 'equality'"
                    )

                param_start = self._get_linear_matrix_source_start(source_name)
                one = {constraint_name: params}
                if constraint_type == 'inequality':
                    generated = adapter.generate_source_inequality_constraints(
                        one, param_start, n_total
                    )
                    opposite_params = copy.deepcopy(params)
                    opposite_params['type'] = 'equality'
                    opposite = adapter.generate_source_equality_constraints(
                        {constraint_name: opposite_params},
                        param_start,
                        n_total,
                    )
                else:
                    generated = adapter.generate_source_equality_constraints(
                        one, param_start, n_total
                    )
                    opposite_params = copy.deepcopy(params)
                    opposite_params['type'] = 'inequality'
                    opposite = adapter.generate_source_inequality_constraints(
                        {constraint_name: opposite_params},
                        param_start,
                        n_total,
                    )

                if not generated:
                    if opposite:
                        expected = (
                            'equality'
                            if constraint_type == 'inequality'
                            else 'inequality'
                        )
                        raise ValueError(
                            f"source constraint rule '{rule}' for "
                            f"'{source_name}' is a {expected} rule, not "
                            f"{constraint_type}"
                        )
                    report['ignored'].append({
                        **item,
                        'reason': 'adapter_not_consumed',
                    })
                    continue

                groups = []
                n_rows = 0
                for generated_name, A, b in generated:
                    full_name = f"src_{source_name}_{generated_name}"
                    if constraint_type == 'inequality':
                        self._register_inequality_group(
                            A,
                            b,
                            name=full_name,
                            source=f"source_constraints/{source_name}",
                            replace=True,
                            owner='config',
                            family='source_constraints',
                        )
                    else:
                        self._register_equality_group(
                            A,
                            b,
                            name=full_name,
                            source=f"source_constraints/{source_name}",
                            replace=True,
                            owner='config',
                            family='source_constraints',
                        )
                    groups.append(full_name)
                    n_rows += int(np.asarray(A).shape[0])

                report['generated_rows'][constraint_type] += n_rows
                report['applied'].append({
                    **item,
                    'groups': groups,
                    'rows': n_rows,
                })

        self._last_source_constraint_report = copy.deepcopy(report)
        return report

    def _supports_patch_component_bounds(self):
        """Return whether the active parameterization exposes patch ss/ds bounds.

        Returns
        -------
        bool
            ``True`` in the base full-linear implementation.

        Notes
        -----
        BLSE/VCE inherits this default. Sampled backends override it when
        their parameterization does not expose independent strike-slip and
        dip-slip components.
        """
        return True

    def _get_patch_selector_fault_name(self, spec, index):
        """Read and normalize the source name from one patch rule.

        Parameters
        ----------
        spec : mapping
            Patch rule using ``fault`` or a compatibility alias
            ``fault_name``/``source``.
        index : int
            Rule position used in error messages.

        Returns
        -------
        str
            Selected source name.

        Raises
        ------
        ValueError
            If none of the supported source-name fields is present or truthy.
        """
        fault_name = spec.get('fault') or spec.get('fault_name') or spec.get('source')
        if not fault_name:
            raise ValueError(f"patch_constraints[{index}] must define 'fault'")
        return str(fault_name)

    @staticmethod
    def _normalise_patch_constraint_entries(raw_specs):
        """Normalize supported patch-constraint declarations to a rule list.

        Parameters
        ----------
        raw_specs : None, list of mapping, or mapping
            Declarations from ``bounds_config.yml`` or a runtime API. Mapping
            forms may be a ``constraints`` list, ``fault -> single rule``,
            ``fault -> rule list``, or ``fault -> rule name -> rule``.

        Returns
        -------
        list of dict
            List-form declarations. Fault-grouped forms receive a ``fault``
            field here; generated names and ``overwrite=False`` are added later
            by :meth:`_iter_patch_constraint_specs`.

        Raises
        ------
        TypeError
            If a list entry cannot be converted to a dictionary.
        ValueError
            If the top-level object, per-fault value, or named rule has an
            unsupported structure.

        Notes
        -----
        This method normalizes syntax only. It does not validate source names,
        selectors, patch ranges, component names, overlaps, or numeric bounds.
        The reserved keys ``defaults`` and ``enabled`` are currently skipped;
        they do not apply defaults or disable the group.

        Returned dictionaries are shallow normalization objects. In the direct
        ``fault -> single rule`` compatibility form, the rule mapping is reused
        and may receive a missing ``fault`` key through ``setdefault``; runtime
        commit methods deep-copy declarations before retaining them.
        """
        if raw_specs is None:
            return []
        # Direct list/API form: every item must carry its own source identity.
        if isinstance(raw_specs, list):
            return [dict(item) for item in raw_specs]
        if isinstance(raw_specs, Mapping):
            # Generic wrapper form shared by some config/API call sites.
            if 'constraints' in raw_specs and isinstance(raw_specs['constraints'], list):
                return [dict(item) for item in raw_specs['constraints']]
            entries = []
            # Fault-grouped forms avoid repeating ``fault`` in every rule.
            for fault_name, fault_specs in raw_specs.items():
                if fault_name in ('defaults', 'enabled'):
                    continue
                if fault_specs is None:
                    continue
                if isinstance(fault_specs, Mapping):
                    # Semantic rule keys identify the single-rule form;
                    # otherwise mapping keys are interpreted as rule names.
                    if any(key in fault_specs for key in ('selector', 'bounds', 'rake_angle')):
                        fault_entries = [fault_specs]
                    else:
                        fault_entries = []
                        for name, item in fault_specs.items():
                            if not isinstance(item, Mapping):
                                raise ValueError(
                                    f"patch_constraints.{fault_name}.{name} must be a mapping"
                                )
                            entry = dict(item)
                            entry.setdefault('name', str(name))
                            fault_entries.append(entry)
                elif isinstance(fault_specs, list):
                    fault_entries = [dict(item) for item in fault_specs]
                else:
                    raise ValueError(
                        f"patch_constraints.{fault_name} must be a mapping or list"
                    )
                for entry in fault_entries:
                    entry.setdefault('fault', str(fault_name))
                    entries.append(entry)
            return entries
        raise ValueError("patch_constraints must be a list or mapping")

    @staticmethod
    def _extract_selector_from_patch_spec(spec):
        """Extract the selector declaration from one normalized patch rule.

        Parameters
        ----------
        spec : mapping
            Rule containing either ``selector`` or supported selector keys at
            the rule's top level.

        Returns
        -------
        object
            Explicit ``selector`` value, or a new mapping of flattened selector
            keys. Validation and conversion to patch ids happen later.

        Raises
        ------
        ValueError
            If the rule declares no supported selector field.
        """
        if 'selector' in spec:
            return spec['selector']
        selector_keys = {
            'patches', 'patch_indices', 'edge', 'edges', 'depth_range',
            'trace_range', 'trace_segment', 'box', 'lon_range', 'lat_range',
            'x_range', 'y_range',
        }
        selector = {key: spec[key] for key in selector_keys if key in spec}
        if selector:
            return selector
        raise ValueError("patch constraint must define selector or patch selector keys")

    @classmethod
    def _extract_bounds_from_patch_spec(cls, spec):
        """Extract canonical strike/dip bounds from one patch rule.

        Parameters
        ----------
        spec : mapping
            Rule with an optional ``bounds`` mapping and/or top-level
            strike/dip aliases.

        Returns
        -------
        dict
            Canonical ``strikeslip``/``dipslip`` keys mapped to their original
            bound declarations. Top-level aliases replace the same component
            from the nested ``bounds`` mapping.

        Raises
        ------
        TypeError
            If ``bounds`` cannot be converted to a dictionary.
        ValueError
            If a component name is not a supported strike/dip alias.
        """
        bounds = dict(spec.get('bounds') or {})
        for alias in ('strikeslip', 'strike_slip', 'ss', 'dipslip', 'dip_slip', 'ds'):
            if alias in spec:
                component = cls._normalize_fault_slip_component(alias)
                bounds[component] = spec[alias]
        return {
            cls._normalize_fault_slip_component(component): value
            for component, value in bounds.items()
        }

    @staticmethod
    def _parse_patch_bound_values(values, n_selected, label):
        """Broadcast one component's bounds to the selected patch count.

        Parameters
        ----------
        values : array-like or mapping
            ``[lb, ub]``, ``[[lb...], [ub...]]``, or
            ``{"lb": ..., "ub": ...}``.
        n_selected : int
            Number of selected patches.
        label : str
            Rule/component label used in exceptions.

        Returns
        -------
        tuple of numpy.ndarray
            One-dimensional ``(lower, upper)`` arrays, each of length
            ``n_selected``.

        Raises
        ------
        TypeError, ValueError
            If values are nonnumeric, have an unsupported shape, cannot be
            broadcast to ``n_selected``, contain NaN/Inf, or have ``lb > ub``.
        """
        if isinstance(values, Mapping) and 'lb' in values and 'ub' in values:
            lower = np.asarray(values['lb'], dtype=float).reshape(-1)
            upper = np.asarray(values['ub'], dtype=float).reshape(-1)
            if lower.size == 1:
                lower = np.full(n_selected, float(lower[0]), dtype=float)
            if upper.size == 1:
                upper = np.full(n_selected, float(upper[0]), dtype=float)
            if lower.size != n_selected or upper.size != n_selected:
                raise ValueError(
                    f"{label} expects scalar bounds or {n_selected} per-patch "
                    f"values, got lower={lower.size}, upper={upper.size}"
                )
        else:
            array = np.asarray(values, dtype=float)
            if array.ndim == 1 and array.size == 2:
                lower = np.full(n_selected, float(array[0]), dtype=float)
                upper = np.full(n_selected, float(array[1]), dtype=float)
            elif array.ndim == 2 and array.shape[0] == 2:
                lower = np.asarray(array[0], dtype=float).reshape(-1)
                upper = np.asarray(array[1], dtype=float).reshape(-1)
                if lower.size == 1:
                    lower = np.full(n_selected, float(lower[0]), dtype=float)
                if upper.size == 1:
                    upper = np.full(n_selected, float(upper[0]), dtype=float)
                if lower.size != n_selected or upper.size != n_selected:
                    raise ValueError(
                        f"{label} expects scalar bounds or {n_selected} per-patch "
                        f"values, got lower={lower.size}, upper={upper.size}"
                    )
            else:
                raise ValueError(
                    f"{label} must be [lb, ub], [[lb...], [ub...]], or {{lb, ub}}"
                )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError(f"{label} must contain finite bounds")
        if np.any(lower > upper):
            raise ValueError(f"{label} has lower bounds greater than upper bounds")
        return lower, upper

    def _select_patch_constraint_indices(self, fault, spec, spec_index):
        """Resolve and validate the patch ids selected by one rule.

        Parameters
        ----------
        fault : object
            Fault whose local patch ids are being selected.
        spec : mapping
            Normalized patch rule.
        spec_index : int
            Rule position used to generate a fallback diagnostic name.

        Returns
        -------
        numpy.ndarray
            Unique, validated, zero-based local patch ids.

        Raises
        ------
        ValueError
            If the selector is missing/invalid, contains out-of-range ids, or
            selects no patches while ``allow_empty`` is false.

        Notes
        -----
        ``None`` never means "all patches" in ``patch_constraints`` because
        this call sets ``allow_none_all=False``.
        """
        from .patch_indices import select_patch_indices

        selector = self._extract_selector_from_patch_spec(spec)
        name = spec.get('name', f"patch_constraint_{spec_index}")
        selected = select_patch_indices(
            fault,
            selector,
            allow_none_all=False,
            unique=True,
            name=f"patch_constraints.{name}.selector",
        )
        if selected.size == 0 and not spec.get('allow_empty', False):
            raise ValueError(f"patch constraint '{name}' selected no patches")
        return selected

    def _iter_patch_constraint_specs(
        self,
        raw_specs=None,
        *,
        include_config=True,
        include_runtime=True,
    ):
        """Collect patch declarations and add common rule defaults.

        Parameters
        ----------
        raw_specs : object, optional
            Explicit declarations to normalize. When provided, config/runtime
            stores are not read.
        include_config : bool, default True
            Include ``bounds_config.yml`` declarations when ``raw_specs`` is
            omitted.
        include_runtime : bool, default True
            Include declarations added through runtime patch APIs when
            ``raw_specs`` is omitted.

        Returns
        -------
        list of dict
            Fresh rule dictionaries containing canonical ``fault``, a stable
            ``name``, and an explicit ``overwrite`` flag.

        Raises
        ------
        TypeError, ValueError
            If declaration normalization fails or a rule omits its source name.

        Notes
        -----
        Config declarations are ordered before runtime declarations. This order
        defines overlap detection and the meaning of ``overwrite=True``.
        """
        specs = []
        if raw_specs is None:
            # Config declarations precede runtime overlays so later runtime
            # rules can explicitly replace overlaps with ``overwrite=True``.
            if include_config and self._bounds_config:
                specs.extend(self._normalise_patch_constraint_entries(
                    self._bounds_config.get('patch_constraints')
                ))
            if include_runtime:
                specs.extend(self._normalise_patch_constraint_entries(
                    self._runtime_patch_constraints
                ))
        else:
            specs.extend(self._normalise_patch_constraint_entries(raw_specs))

        normalised = []
        for index, spec in enumerate(specs):
            entry = dict(spec)
            entry['fault'] = self._get_patch_selector_fault_name(entry, index)
            entry.setdefault('name', f"{entry['fault']}_patch_constraint_{index}")
            entry.setdefault('overwrite', False)
            normalised.append(entry)
        return normalised

    def apply_patch_bounds(self, patch_constraints=None, *, source="patch_constraints"):
        """Resolve patch rules and write component bounds into ``lb``/``ub``.

        Parameters
        ----------
        patch_constraints : object, optional
            Explicit declarations. If omitted, active config-owned declarations
            and runtime declarations are combined.
        source : str, default "patch_constraints"
            Provenance label stored with resolved bound diagnostics.

        Returns
        -------
        list of dict
            Applied entries with ``name``, ``fault``, ``component``, selected
            ``patches`` as an array, and ``source``. A rule containing only
            ``rake_angle`` produces no item here.

        Raises
        ------
        ValueError
            If a source/selector/component/bound is invalid, patch selection is
            empty, or overlapping patch bounds lack ``overwrite=True``.

        Notes
        -----
        Selected ids are fault-local patch ids. They are converted to model
        columns through the adapter-defined component slice. Patch bounds are
        resolved after global and fault/component bounds, so they override the
        broader declaration for only those columns.
        """
        specs = self._iter_patch_constraint_specs(
            patch_constraints,
            include_config=(
                True
                if patch_constraints is not None
                else self._config_flag_enabled('use_bounds_constraints')
            ),
        )
        if not specs:
            return []

        self._initialize_bounds_arrays()
        # Track ownership per fault/component/patch. Different components may
        # overlap legitimately; duplicate writes to one component are strict.
        seen = {}
        applied = []
        for spec_index, spec in enumerate(specs):
            fault_name = spec['fault']
            if not self._fault_exists(fault_name):
                raise ValueError(f"patch constraint '{spec['name']}' references unknown fault '{fault_name}'")
            if self._get_source_type(fault_name) != 'Fault':
                raise ValueError(
                    f"patch constraint '{spec['name']}' only applies to Fault sources; "
                    f"'{fault_name}' is {self._get_source_type(fault_name)}"
                )
            bounds = self._extract_bounds_from_patch_spec(spec)
            if not bounds:
                continue
            if not self._supports_patch_component_bounds():
                raise ValueError(
                    f"patch constraint '{spec['name']}' defines strike/dip bounds, "
                    "but the current slip parameterization does not expose "
                    "independent strikeslip/dipslip parameters."
                )

            fault = self._get_fault_by_name(fault_name)
            selected = self._select_patch_constraint_indices(fault, spec, spec_index)
            overwrite = bool(spec.get('overwrite', False))

            for component, values in bounds.items():
                key = (fault_name, component)
                seen.setdefault(key, {})
                overlap = [
                    int(patch)
                    for patch in selected.tolist()
                    if int(patch) in seen[key]
                ]
                if overlap and not overwrite:
                    previous = sorted({seen[key][int(patch)] for patch in overlap})
                    raise ValueError(
                        f"patch constraint '{spec['name']}' overlaps previous "
                        f"{fault_name}.{component} patch constraint(s) {previous} "
                        f"at patches {overlap}. Set overwrite: true to replace."
                    )

                lower, upper = self._parse_patch_bound_values(
                    values,
                    selected.size,
                    f"patch constraint '{spec['name']}' {component}",
                )
                columns = self._get_component_columns_for_patches(
                    fault_name,
                    component,
                    selected,
                    space='bounds',
                )
                self.set_parameter_bounds_by_indices(
                    columns, lower, upper, source=source, persist=False
                )
                for patch in selected.tolist():
                    seen[key][int(patch)] = spec['name']
                applied.append({
                    'name': spec['name'],
                    'fault': fault_name,
                    'component': component,
                    'patches': selected.copy(),
                    'source': source,
                })

        if applied:
            self._bounds['patch_constraints'] = {
                f"{item['name']}.{item['component']}": {
                    'fault': item['fault'],
                    'component': item['component'],
                    'n_patches': int(len(item['patches'])),
                    'source': item['source'],
                }
                for item in applied
            }
        return applied

    def add_patch_constraints(self, patch_constraints, *, source="manual", sync=True):
        """Add patch-level bounds/rake override specs and apply them.

        This is intended for script-side updates before solving or sampling.
        Existing specs are kept; use explicit ``overwrite: true`` on a spec
        when it should replace a previous patch-level bound/rake on the same
        patch and component.

        Parameters
        ----------
        patch_constraints : object
            Declarations accepted by
            :meth:`_normalise_patch_constraint_entries`.
        source : str, default "manual"
            Provenance assigned while rebuilding resolved state.
        sync : bool, default True
            Call a backend ``sync_to_solver`` hook after successful validation.

        Returns
        -------
        list of dict
            Deep copy of normalized declarations committed to the runtime
            declaration store. This is not the ``apply_patch_bounds`` report.

        Raises
        ------
        TypeError, ValueError, RuntimeError
            Propagates parsing, overlap, bounds, rake, validation, or backend
            synchronization errors. State is restored if commit validation
            fails before synchronization.
        """
        entries = self._normalise_patch_constraint_entries(patch_constraints)
        if not entries:
            return []

        return self._set_runtime_patch_constraints(
            entries,
            replace=False,
            source=source,
            sync=sync,
        )

    def replace_patch_constraints(self, patch_constraints, *, source="manual", sync=True):
        """Replace all script-side patch overrides and rebuild resolved state.

        Config-file ``patch_constraints`` remain active.  This method replaces
        only declarations previously added through the runtime API, making it
        suitable for parameter sweeps that reuse one inversion object.

        Parameters
        ----------
        patch_constraints : object
            Complete replacement for runtime-owned patch declarations.
        source : str, default "manual"
            Provenance assigned while rebuilding resolved state.
        sync : bool, default True
            Call a backend ``sync_to_solver`` hook after successful validation.

        Returns
        -------
        list of dict
            Deep copy of the normalized replacement declarations.

        Raises
        ------
        TypeError, ValueError, RuntimeError
            Propagates normalization, rebuild, validation, or synchronization
            errors. The pre-update manager state is restored on rebuild or
            validation failure.
        """
        entries = self._normalise_patch_constraint_entries(patch_constraints)
        return self._set_runtime_patch_constraints(
            entries,
            replace=True,
            source=source,
            sync=sync,
        )

    def clear_patch_constraints(self, *, source="manual", sync=True):
        """Remove all runtime patch declarations and rebuild resolved state.

        Parameters
        ----------
        source : str, default "manual"
            Provenance assigned to the rebuilt state.
        sync : bool, default True
            Call a backend ``sync_to_solver`` hook after successful validation.

        Returns
        -------
        list
            Empty list. Config-file patch declarations remain active.
        """
        return self.replace_patch_constraints([], source=source, sync=sync)

    def _set_runtime_patch_constraints(self, entries, *, replace, source, sync):
        """Commit runtime patch declarations as one validated transaction.

        Parameters
        ----------
        entries : iterable of mapping
            Already-normalized runtime declarations.
        replace : bool
            Replace the runtime store when true; append when false.
        source : str
            Provenance passed to the bounds rebuild.
        sync : bool
            Invoke ``sync_to_solver`` after a successful commit when available.

        Returns
        -------
        list of dict
            Deep copy of ``entries`` after commit.

        Raises
        ------
        Exception
            Any rebuild, rake-generation, validation, or synchronization error
            is propagated. Manager-owned state is restored for errors raised
            inside the guarded rebuild/validation transaction.

        Notes
        -----
        Bounds and rake are both rebuilt because replacement may remove the
        final runtime rule of either kind.
        """

        snapshot = self._snapshot_mutable_state()
        try:
            # Validate config-level and all runtime declarations together.  This
            # makes overlap handling independent of how many API calls were used.
            if replace:
                self._runtime_patch_constraints = copy.deepcopy(list(entries))
            else:
                self._runtime_patch_constraints.extend(copy.deepcopy(list(entries)))
            self._rebuild_resolved_bounds(source=source)
            # Always rebuild rake: replacement may have removed the final
            # runtime rake declaration and must not leave its old matrix live.
            self._rebuild_rake_constraints()
            self._validate_or_raise()
        except Exception:
            self._restore_mutable_state(snapshot)
            raise

        sync_to_solver = getattr(self, 'sync_to_solver', None)
        if sync and callable(sync_to_solver):
            sync_to_solver()
        return copy.deepcopy(entries)

    def _linear_constraints_active(self) -> bool:
        """Return whether this backend currently consumes linear constraints.

        Returns
        -------
        bool
            ``True`` in the base full-linear implementation.

        Notes
        -----
        BLSE/VCE inherits this default. FULLSMC-like modes override it to
        report source/rake matrices as inactive.
        """
        return True

    def _require_linear_constraints_active(self, action):
        """Raise when a backend cannot consume linear constraint matrices."""
        if not self._linear_constraints_active():
            raise RuntimeError(
                f"{action} requires an active linear-constraint mode"
            )

    def _config_flag_enabled(self, name):
        """Read one config-owned constraint activation flag.

        Parameters
        ----------
        name : str
            Attribute name on the backend config object.

        Returns
        -------
        bool
            Configured value, or ``True`` when the config/attribute is absent.

        Notes
        -----
        These flags gate declarations loaded from a config file. Runtime-owned
        declarations remain independently active.
        """
        config = getattr(self, 'config', None)
        if config is None or not hasattr(config, name):
            return True
        return bool(getattr(config, name))

    def _current_activation_flags(self):
        """Return live switches that gate config-owned declarations.

        Returns
        -------
        dict of str to bool
            Current bounds and rake activation flags.
        """
        return {
            'use_bounds_constraints': self._config_flag_enabled(
                'use_bounds_constraints'
            ),
            'use_rake_angle_constraints': self._config_flag_enabled(
                'use_rake_angle_constraints'
            ),
        }

    def _mark_activation_flags_reconciled(self):
        """Record that resolved state represents the current config switches.

        Returns
        -------
        None
        """
        self._last_reconciled_flags = self._current_activation_flags()

    def _activation_flags_are_stale(self):
        """Return whether live switches changed after the last rebuild.

        Returns
        -------
        bool
            ``True`` only after an initial reconciliation and a later flag
            change.
        """
        return (
            self._last_reconciled_flags is not None
            and self._last_reconciled_flags != self._current_activation_flags()
        )

    def _require_activation_flags_reconciled(self, context):
        """Reject solver/target construction from stale activation switches.

        Changing a config flag mutates the declaration policy, not the already
        resolved bounds or matrices.  Callers must therefore reconcile the
        configuration before a numerical consumer reads manager state.

        Parameters
        ----------
        context : str
            Solver/target operation included in the error message.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If activation flags changed after the last reconciliation.
        """
        if not self._activation_flags_are_stale():
            return
        raise RuntimeError(
            f"{context}: constraint activation flags changed after the last "
            "reconciliation. Call apply_constraints_from_config() before "
            "solving or constructing a sampling target."
        )

    def _validate_runtime_rake_limits(self, rake_limits):
        """Validate script-side fault rake declarations before commit.

        Parameters
        ----------
        rake_limits : mapping
            Fault name to two-value rake interval in degrees.

        Returns
        -------
        dict
            Canonical string fault names mapped to validated endpoint tuples.

        Raises
        ------
        TypeError
            If ``rake_limits`` is not a mapping.
        ValueError
            If a source is unknown/non-Fault, lacks independent strike/dip
            components, or declares an invalid convex rake sector.
        """
        if not isinstance(rake_limits, Mapping):
            raise TypeError("rake_limits must be a mapping of fault name to [min, max]")
        validated = {}
        for fault_name, limits in rake_limits.items():
            fault_name = str(fault_name)
            if not self._fault_exists(fault_name):
                raise ValueError(
                    f"rake_limits references unknown fault '{fault_name}'"
                )
            source_type = self._get_source_type(fault_name)
            if source_type != 'Fault':
                raise ValueError(
                    f"rake_limits only applies to Fault sources; "
                    f"'{fault_name}' is {source_type}"
                )
            fault = self._get_fault_by_name(fault_name)
            adapter = self._get_adapter_for_source(fault_name)
            self._rake_component_starts(
                fault,
                self._get_linear_matrix_source_start(fault_name),
                len(getattr(fault, 'patch', [])),
                adapter=adapter,
            )
            validated[fault_name] = self._validate_rake_interval(
                fault_name, limits
            )
        return validated

    def update_fault_rake_limits(self, rake_limits, *, replace=False, source="manual", sync=True):
        """Update fault-level rake limits and rebuild the final rake matrix.

        The update changes the declaration layer only; the final inequality
        matrix is rebuilt from config-level rake defaults, runtime rake
        updates, and patch-level rake overrides.  Existing patch overrides are
        preserved.

        Parameters
        ----------
        rake_limits : mapping or None
            Runtime fault-level intervals. ``None`` returns current state
            without mutation.
        replace : bool, default False
            Replace all runtime fault intervals instead of updating named
            faults. An empty mapping clears only when ``replace=True``.
        source : str, default "manual"
            Provenance used by the generated rake group.
        sync : bool, default True
            Call a backend ``sync_to_solver`` hook after successful validation.

        Returns
        -------
        dict
            Deep copy of the complete runtime fault-rake declaration mapping.

        Raises
        ------
        TypeError, ValueError
            If declarations or the resolved constraint state are invalid.
        RuntimeError
            If the current backend does not consume linear constraints.

        Notes
        -----
        The declaration update, rake rebuild, and validation are transactional.
        Backend synchronization occurs only after that transaction succeeds.
        """
        if rake_limits is None:
            return copy.deepcopy(self._runtime_rake_limits)
        if not rake_limits and not replace:
            return copy.deepcopy(self._runtime_rake_limits)
        if not self._linear_constraints_active():
            raise RuntimeError(
                "Fault rake sectors are linear ss/ds constraints and are only "
                "available in a mode with active linear slip parameters"
            )
        self._validate_runtime_rake_limits(rake_limits)
        declarations = copy.deepcopy(dict(rake_limits))
        snapshot = self._snapshot_mutable_state()
        try:
            if replace:
                self._runtime_rake_limits = declarations
            else:
                self._runtime_rake_limits.update(declarations)
            self._rebuild_rake_constraints()
            self._validate_or_raise()
        except Exception:
            self._restore_mutable_state(snapshot)
            raise
        sync_to_solver = getattr(self, 'sync_to_solver', None)
        if sync and callable(sync_to_solver):
            sync_to_solver()
        return copy.deepcopy(self._runtime_rake_limits)

    def clear_fault_rake_limits(self, *, source="manual", sync=True):
        """Clear script-side fault rake sectors and rebuild remaining rake state.

        Configuration rake sectors and runtime patch rake declarations are not
        removed. Repeated calls are safe.

        Parameters
        ----------
        source : str, default "manual"
            Provenance label retained for API symmetry.
        sync : bool, default True
            Call a backend ``sync_to_solver`` hook after successful validation.

        Returns
        -------
        dict
            Empty mapping. Repeated calls return the same logical result.

        Raises
        ------
        ValueError
            If rebuilding the remaining rake state produces invalid constraints.
        """
        if not self._runtime_rake_limits:
            return {}
        snapshot = self._snapshot_mutable_state()
        try:
            self._runtime_rake_limits = {}
            self._rebuild_rake_constraints()
            self._validate_or_raise()
        except Exception:
            self._restore_mutable_state(snapshot)
            raise
        sync_to_solver = getattr(self, 'sync_to_solver', None)
        if sync and callable(sync_to_solver):
            sync_to_solver()
        return {}

    def _resolve_rake_intervals_by_patch(self, rake_limits=None):
        """Resolve fault-level rake defaults and patch-level replacements.

        Parameters
        ----------
        rake_limits : mapping, optional
            Fault-level default intervals. Each default expands to all patches
            of that fault.

        Returns
        -------
        dict
            ``fault_name -> patch_id -> (rake_min, rake_max)``. Patch rules
            replace the default interval for selected ids.

        Raises
        ------
        ValueError
            If a patch rule references an invalid source/selector/rake interval
            or overlaps another patch-rake rule without ``overwrite=True``.

        Notes
        -----
        Config patch-rake declarations are included only when
        ``use_rake_angle_constraints`` is active. Runtime patch declarations
        remain included. Fault-level entries for unavailable/non-Fault sources
        are skipped for compatibility; patch-level references are strict.
        """
        rake_limits = dict(rake_limits or {})
        intervals = {}
        for fault_name, limits in rake_limits.items():
            if not self._fault_exists(fault_name):
                continue
            if self._get_source_type(fault_name) != 'Fault':
                continue
            fault = self._get_fault_by_name(fault_name)
            n_patch = len(getattr(fault, 'patch', []))
            validated = self._validate_rake_interval(fault_name, limits)
            intervals[fault_name] = {int(i): validated for i in range(n_patch)}

        seen_patch_specs = {}
        for spec_index, spec in enumerate(self._iter_patch_constraint_specs(
            include_config=self._config_flag_enabled(
                'use_rake_angle_constraints'
            )
        )):
            if 'rake_angle' not in spec:
                continue
            fault_name = spec['fault']
            if not self._fault_exists(fault_name):
                raise ValueError(f"patch constraint '{spec['name']}' references unknown fault '{fault_name}'")
            if self._get_source_type(fault_name) != 'Fault':
                raise ValueError(
                    f"patch constraint '{spec['name']}' only applies to Fault sources; "
                    f"'{fault_name}' is {self._get_source_type(fault_name)}"
                )
            fault = self._get_fault_by_name(fault_name)
            selected = self._select_patch_constraint_indices(fault, spec, spec_index)
            interval = self._validate_rake_interval(
                f"{fault_name}.{spec['name']}",
                spec['rake_angle'],
            )
            overwrite = bool(spec.get('overwrite', False))
            intervals.setdefault(fault_name, {})
            seen_patch_specs.setdefault(fault_name, {})
            overlap = [
                int(patch)
                for patch in selected.tolist()
                if int(patch) in seen_patch_specs[fault_name]
            ]
            if overlap and not overwrite:
                previous = sorted({seen_patch_specs[fault_name][int(patch)] for patch in overlap})
                raise ValueError(
                    f"patch rake constraint '{spec['name']}' overlaps previous "
                    f"patch rake constraint(s) {previous} at {fault_name} patches "
                    f"{overlap}. Set overwrite: true to replace."
                )
            for patch in selected.tolist():
                # Patch rake replaces the fault-level interval for this id.
                intervals[fault_name][int(patch)] = interval
                seen_patch_specs[fault_name][int(patch)] = spec['name']

        return {
            fault_name: patch_map
            for fault_name, patch_map in intervals.items()
            if patch_map
        }

    def _generate_rake_inequality_constraints_from_intervals(self, intervals_by_fault):
        """Build the final linear rake-sector inequality matrix.

        Parameters
        ----------
        intervals_by_fault : mapping
            Resolved ``fault -> patch -> (rake_min, rake_max)`` intervals.

        Returns
        -------
        A : numpy.ndarray
            Matrix with two rows per constrained patch and active linear-model
            width.
        b : numpy.ndarray
            Zero right-hand side for ``A @ m <= b``.
        constrained_faults : list of str
            Faults contributing at least one interval.

        Raises
        ------
        ValueError
            If required strike/dip components are absent, component sizes are
            inconsistent, or a patch id is outside the fault-local range.

        Notes
        -----
        For each fault, rows for all minimum-rake boundaries precede rows for
        all maximum-rake boundaries. Matrix columns are resolved through the
        adapter-defined strike/dip blocks.
        """
        if not intervals_by_fault:
            return np.zeros((0, self._get_linear_matrix_n_parameters())), np.zeros(0), []

        n_params = self._get_linear_matrix_n_parameters()
        n_rows = 2 * sum(len(patch_map) for patch_map in intervals_by_fault.values())
        A = np.zeros((n_rows, n_params))
        b = np.zeros(n_rows)
        constrained_faults = []

        row_offset = 0
        for fault_name, patch_map in intervals_by_fault.items():
            fault = self._get_fault_by_name(fault_name)
            n_patch = len(getattr(fault, 'patch', []))
            adapter = self._get_adapter_for_source(fault_name)
            source_start = self._get_linear_matrix_source_start(fault_name)
            ss_start, ds_start = self._rake_component_starts(
                fault,
                source_start,
                n_patch,
                adapter=adapter,
            )
            patch_ids = sorted(int(patch) for patch in patch_map)
            for local_row, patch in enumerate(patch_ids):
                if patch < 0 or patch >= n_patch:
                    raise ValueError(
                        f"rake patch index {patch} is out of range for "
                        f"fault '{fault_name}' with {n_patch} patches"
                    )
                rake_start, rake_end = patch_map[patch]
                # Store all lower-boundary rows first, followed by the matching
                # upper-boundary rows for this fault.
                lower_row = row_offset + local_row
                upper_row = row_offset + len(patch_ids) + local_row
                A[lower_row, ss_start + patch] = np.sin(np.deg2rad(rake_start))
                A[lower_row, ds_start + patch] = -np.cos(np.deg2rad(rake_start))
                A[upper_row, ss_start + patch] = -np.sin(np.deg2rad(rake_end))
                A[upper_row, ds_start + patch] = np.cos(np.deg2rad(rake_end))
            row_offset += 2 * len(patch_ids)
            constrained_faults.append(fault_name)

        return A, b, constrained_faults

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _invalidate_constraint_cache(self):
        """Invalidate both combined-matrix caches after a group mutation.

        Returns
        -------
        None

        Notes
        -----
        A single call invalidates equality and inequality caches and advances
        ``state_revision`` once, even if only one constraint family changed.
        """
        self._combined_cache['inequality']['valid'] = False
        self._combined_cache['equality']['valid'] = False
        self._state_revision += 1

    def _mark_bounds_changed(self):
        """Record a resolved-bounds change for target lifecycle checks.

        Returns
        -------
        None
        """
        self._state_revision += 1

    @contextmanager
    def batch_bounds_update(self):
        """Defer bounds rebuilding until a declaration batch completes.

        Yields
        ------
        ConstraintManagerBase
            This manager, so callers can group several declaration writes.

        Notes
        -----
        Contexts may be nested. Only the outermost successful context invokes
        ``_rebuild_resolved_bounds``. This context batches work but does not
        provide rollback; use :meth:`atomic_bounds_update` for transactions.
        """
        self._bounds_rebuild_depth += 1
        succeeded = False
        try:
            yield self
            succeeded = True
        finally:
            self._bounds_rebuild_depth -= 1
            if succeeded and self._bounds_rebuild_depth == 0:
                self._rebuild_resolved_bounds()

    @contextmanager
    def atomic_bounds_update(self):
        """Apply a public bounds update as one validated transaction.

        Yields
        ------
        ConstraintManagerBase
            This manager for declaration mutations.

        Raises
        ------
        Exception
            Re-raises any mutation, rebuild, or validation error after
            restoring the pre-update manager state.

        Notes
        -----
        The context combines deferred rebuilding, final validation, and
        rollback. It does not automatically call a backend synchronization
        hook.
        """
        snapshot = self._snapshot_mutable_state()
        try:
            with self.batch_bounds_update():
                yield self
            self._validate_or_raise()
        except Exception:
            self._restore_mutable_state(snapshot)
            raise

    @property
    def in_constraint_transaction(self):
        """Return whether a facade-level constraint transaction is active."""
        return self._constraint_transaction_depth > 0

    @contextmanager
    def constraint_transaction(self):
        """Group several facade mutations into one validated transaction.

        The outermost context snapshots manager state and the two
        configuration fields that participate in interseismic reconciliation.
        Nested contexts join the outer transaction. Validation and BLSE
        compatibility synchronization are deferred until the outer commit.

        Yields
        ------
        ConstraintManagerBase
            This manager.

        Raises
        ------
        Exception
            Re-raises the original mutation/validation error after restoring
            manager and configuration state.
        """
        outermost = self._constraint_transaction_depth == 0
        if outermost:
            manager_snapshot = self._snapshot_mutable_state()
            config_snapshot = None
            if self.config is not None:
                config_snapshot = {
                    'interseismic_config': copy.deepcopy(
                        getattr(self.config, 'interseismic_config', None)
                    ),
                    'interseismic_config_file': copy.deepcopy(
                        getattr(self.config, 'interseismic_config_file', None)
                    ),
                }
            self._constraint_transaction_sync_pending = False

        self._constraint_transaction_depth += 1
        try:
            yield self
            if outermost:
                self._validate_or_raise(force=True)
        except Exception:
            if outermost:
                self._restore_mutable_state(manager_snapshot)
                if self.config is not None and config_snapshot is not None:
                    self.config.interseismic_config = config_snapshot[
                        'interseismic_config'
                    ]
                    self.config.interseismic_config_file = config_snapshot[
                        'interseismic_config_file'
                    ]
                # A rollback refreshes compatibility mirrors immediately.
                # Clear any deferred request so the outer ``finally`` block
                # does not perform a duplicate second synchronization.
                self._constraint_transaction_sync_pending = False
                sync = getattr(self, 'sync_to_solver', None)
                if callable(sync):
                    sync(force=True)
            raise
        finally:
            self._constraint_transaction_depth -= 1
            if outermost:
                pending_sync = self._constraint_transaction_sync_pending
                self._constraint_transaction_sync_pending = False
                if pending_sync:
                    sync = getattr(self, 'sync_to_solver', None)
                    if callable(sync):
                        sync(force=True)

    def _request_bounds_rebuild(self):
        """Request resolution of stored bounds declarations.

        Returns
        -------
        None

        Notes
        -----
        Rebuilding occurs immediately outside ``batch_bounds_update`` and is
        deferred while a declaration batch is open.
        """
        if self._bounds_rebuild_depth == 0:
            self._rebuild_resolved_bounds()

    def _snapshot_mutable_state(self):
        """Capture manager-owned mutable state for transaction rollback.

        Returns
        -------
        dict
            Snapshot of declarations, resolved bounds, reports, caches,
            constraint groups, activation state, and revision number.

        Notes
        -----
        Bounds/declarations are deep-copied. Constraint groups and cached
        matrices are copied shallowly because manager updates replace their
        arrays rather than mutating them in place. This helper is intended for
        infrequent control-plane updates, not numerical inner loops.
        """
        return {
            'bounds_config': copy.deepcopy(self._bounds_config),
            'bounds': copy.deepcopy(self._bounds),
            'runtime_rake_limits': copy.deepcopy(self._runtime_rake_limits),
            'runtime_patch_constraints': copy.deepcopy(self._runtime_patch_constraints),
            'last_source_constraint_report': copy.deepcopy(
                self._last_source_constraint_report
            ),
            'last_reconciled_flags': copy.deepcopy(self._last_reconciled_flags),
            # Constraint arrays are replaced, never mutated in place by manager
            # updates, so shallow group copies avoid duplicating dense matrices.
            'inequality_constraints': dict(self._inequality_constraints),
            'equality_constraints': dict(self._equality_constraints),
            'combined_cache': {
                kind: {
                    'A': cache['A'],
                    'b': cache['b'],
                    'valid': cache['valid'],
                }
                for kind, cache in self._combined_cache.items()
            },
            'state_revision': self._state_revision,
        }

    def _restore_mutable_state(self, snapshot):
        """Restore a snapshot after a failed declaration transaction.

        Parameters
        ----------
        snapshot : mapping
            Object returned by :meth:`_snapshot_mutable_state`.

        Returns
        -------
        None

        Notes
        -----
        After restoring manager-owned fields, ``_on_bounds_config_loaded`` is
        called so subclass compatibility mirrors receive the restored config.
        """
        self._bounds_config = snapshot['bounds_config']
        self._bounds = snapshot['bounds']
        self._runtime_rake_limits = snapshot['runtime_rake_limits']
        self._runtime_patch_constraints = snapshot['runtime_patch_constraints']
        self._last_source_constraint_report = snapshot[
            'last_source_constraint_report'
        ]
        self._last_reconciled_flags = snapshot['last_reconciled_flags']
        self._inequality_constraints = snapshot['inequality_constraints']
        self._equality_constraints = snapshot['equality_constraints']
        self._combined_cache = snapshot['combined_cache']
        self._state_revision = snapshot['state_revision']
        self._on_bounds_config_loaded()

    def _validate_or_raise(self, *, force=False):
        """Validate resolved state and raise one aggregated error if invalid.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If :meth:`validate` reports one or more errors.
        """
        if self.in_constraint_transaction and not force:
            return
        result = self.validate()
        if not result['valid']:
            raise ValueError("Invalid constraint state: " + "; ".join(result['errors']))

    def _expected_linear_constraint_columns(self):
        """Return the active linear-constraint width when it can be inferred.

        Returns
        -------
        int or None
            Expected column count, or ``None`` when backend state is not yet
            sufficient to infer it.

        Notes
        -----
        This helper intentionally converts backend lookup failures to ``None``
        so early construction-time validation can remain conditional.
        """
        try:
            layout = self.get_linear_parameter_layout()
            if not layout.get('active', False):
                return None
            return int(layout['width'])
        except NotImplementedError:
            try:
                return int(self._get_linear_matrix_n_parameters())
            except (AttributeError, NotImplementedError, TypeError, ValueError):
                return None

    def _build_linear_parameter_layout(
        self,
        *,
        space,
        width,
        global_offset,
        source_positions,
        poly_positions,
        active=True,
        inactive_reason=None,
    ):
        """Build and validate one backend's linear parameter-layout descriptor.

        Parameters
        ----------
        space : str
            Public parameter-space identifier.
        width : int
            Number of columns consumed by active linear constraints.
        global_offset : int
            Offset of column zero in the owning full/sample vector.
        source_positions, poly_positions : mapping
            Per-source half-open ranges in the owning full/sample vector.
        active : bool, default True
            Whether the current backend mode consumes linear matrices.
        inactive_reason : str, optional
            Human-readable explanation for an inactive layout.

        Returns
        -------
        dict
            Validated descriptor containing ``space``, ``width``,
            ``global_offset``, and ordered ``blocks``.

        Raises
        ------
        ValueError
            If source/component/poly ranges are non-contiguous, disagree with
            adapter counts or assembled CSI matrices, or fail to cover the
            declared width exactly.

        Notes
        -----
        CSI/adapter assembly remains authoritative. This descriptor reports
        that existing order; it never creates a second order. Current ECAT
        layouts intentionally reject CSI custom-GF columns because those
        columns are not represented by the source/poly position mappings.
        """
        width = int(width)
        global_offset = int(global_offset)
        if not active:
            return {
                'space': str(space),
                'active': False,
                'width': 0,
                'global_offset': global_offset,
                'blocks': [],
                'inactive_reason': inactive_reason,
            }

        blocks = []
        cursor = 0
        for source in self._get_fault_sequence():
            source_name = source.name
            if source_name not in source_positions:
                raise ValueError(
                    f"Linear layout has no source position for '{source_name}'"
                )
            source_start_abs, source_stop_abs = source_positions[source_name]
            if source_name in poly_positions:
                poly_start_abs, poly_stop_abs = poly_positions[source_name]
            else:
                # A source without a declared polynomial range has zero
                # nuisance columns. Exact final coverage below still rejects
                # any genuinely undeclared columns.
                poly_start_abs = poly_stop_abs = source_stop_abs
            source_start = int(source_start_abs) - global_offset
            source_stop = int(source_stop_abs) - global_offset
            poly_start = int(poly_start_abs) - global_offset
            poly_stop = int(poly_stop_abs) - global_offset

            if source_start != cursor:
                raise ValueError(
                    f"Linear layout gap/overlap before '{source_name}': "
                    f"expected start {cursor}, got {source_start}"
                )
            if source_stop < source_start or poly_stop < poly_start:
                raise ValueError(
                    f"Linear layout for '{source_name}' has a reversed range"
                )
            if poly_start != source_stop:
                raise ValueError(
                    f"Linear layout for '{source_name}' has a gap/overlap "
                    f"between source ({source_start}:{source_stop}) and "
                    f"polynomial ({poly_start}:{poly_stop}) blocks"
                )

            adapter = self._get_adapter_for_source(source_name)
            component_slices = self._source_component_slices(
                source,
                source_start,
                adapter=adapter,
            )
            component_cursor = source_start
            for component, component_slice in component_slices.items():
                start = int(component_slice.start)
                stop = int(component_slice.stop)
                if start != component_cursor:
                    raise ValueError(
                        f"Linear component layout for "
                        f"'{source_name}.{component}' starts at {start}; "
                        f"expected {component_cursor}"
                    )
                blocks.append({
                    'source': source_name,
                    'component': component,
                    'start': start,
                    'stop': stop,
                    'role': 'source_parameter',
                })
                component_cursor = stop
            if component_cursor != source_stop:
                raise ValueError(
                    f"Adapter components for '{source_name}' end at "
                    f"{component_cursor}; source position ends at {source_stop}"
                )

            if poly_stop > poly_start:
                blocks.append({
                    'source': source_name,
                    'component': 'polynomial',
                    'start': poly_start,
                    'stop': poly_stop,
                    'role': 'data_correction',
                })

            expected_local_width = poly_stop - source_start
            assembled = getattr(source, 'Gassembled', None)
            if assembled is not None:
                assembled_array = np.asarray(assembled)
                if assembled_array.ndim != 2:
                    raise ValueError(
                        f"CSI Gassembled for '{source_name}' must be 2D, "
                        f"got shape {assembled_array.shape}"
                    )
                if int(assembled_array.shape[1]) != expected_local_width:
                    raise ValueError(
                        f"CSI/ECAT parameter layout mismatch for "
                        f"'{source_name}': Gassembled has "
                        f"{assembled_array.shape[1]} columns, while ECAT "
                        f"source/poly positions describe "
                        f"{expected_local_width}. CSI custom=True Green-"
                        "function columns are not yet supported; assemble "
                        "with custom=False."
                    )
            cursor = poly_stop

        if cursor != width:
            raise ValueError(
                f"Linear layout covers {cursor} columns; declared width is "
                f"{width}"
            )
        for index, block in enumerate(blocks):
            if block['start'] < 0 or block['stop'] > width:
                raise ValueError(
                    f"Linear layout block {index} is outside [0, {width})"
                )
            if index and block['start'] != blocks[index - 1]['stop']:
                raise ValueError(
                    f"Linear layout blocks {index - 1} and {index} are not "
                    "contiguous"
                )

        return {
            'space': str(space),
            'active': True,
            'width': width,
            'global_offset': global_offset,
            'blocks': blocks,
            'inactive_reason': None,
        }

    def get_linear_parameter_layout(self):
        """Return the validated linear parameter layout for this backend.

        Concrete managers must report their own full-vector or linear-suffix
        coordinate system through :meth:`_build_linear_parameter_layout`.

        Returns
        -------
        dict
            Active or inactive layout descriptor.

        Raises
        ------
        NotImplementedError
            The concrete backend must define its authoritative layout.
        """
        raise NotImplementedError

    def sync_to_solver(self, *, force=False):
        """Synchronize backend compatibility mirrors.

        SMC has no solver-side mirror and therefore uses this no-op base
        implementation. BLSE overrides it.

        Parameters
        ----------
        force : bool, default False
            Reserved for concrete backends that defer synchronization inside
            transactions.

        Returns
        -------
        None
        """
        return None

    # ------------------------------------------------------------------
    # Read-only diagnostics
    # ------------------------------------------------------------------

    @classmethod
    def _readonly_snapshot(cls, value):
        """Create a recursive read-only diagnostic snapshot.

        Parameters
        ----------
        value : object
            Nested arrays, dictionaries, lists, tuples, and scalar metadata.

        Returns
        -------
        object
            Arrays are copied and marked non-writeable, dictionaries become
            ``MappingProxyType`` objects, and lists become tuples. Scalars and
            unrecognized immutable objects are returned unchanged.

        Notes
        -----
        The snapshot prevents ordinary mutation of manager state through public
        diagnostic properties; it is not a serialization format.
        """
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
        """Summarize named constraint groups without exposing dense matrices.

        Parameters
        ----------
        groups : mapping
            Constraint group storage containing at least an ``A`` matrix.

        Returns
        -------
        dict
            Per-group row/column counts, provenance, and declared shape.
        """
        return {
            name: {
                'rows': int(group['A'].shape[0]),
                'cols': int(group['A'].shape[1]),
                'source': group.get('source'),
                'owner': group.get('owner', 'user'),
                'family': group.get('family'),
                'shape': tuple(group.get('shape', group['A'].shape)),
            }
            for name, group in groups.items()
        }

    def get_constraint_snapshot(
        self,
        include_matrices: bool = False,
        validate: bool = False,
    ) -> Dict[str, Any]:
        """Return a compact diagnostic snapshot of bounds and constraints.

        The constraint manager is the single writable source of truth.
        Application code should mutate it through the inversion object's
        public constraint facade; manager registration helpers are internal.
        Set ``validate=True`` for an explicit consistency and equality-rank
        check before solving. It is optional because matrix-rank validation
        can be expensive for large systems.

        Parameters
        ----------
        include_matrices : bool, default False
            Include read-only ``A``/``b`` snapshots instead of lightweight
            per-group summaries.
        validate : bool, default False
            Run :meth:`validate` and attach its report.

        Returns
        -------
        dict
            Bounds counts, runtime declarations, source-rule report,
            activation state, cache state, constraint totals, and optionally
            matrices/validation results.

        Notes
        -----
        This method is diagnostic only and does not reconcile stale activation
        flags or rebuild bounds/constraint matrices.
        """
        n_parameters = None
        lb = self._bounds.get('lb')
        ub = self._bounds.get('ub')
        if lb is not None:
            n_parameters = int(len(lb))
        elif ub is not None:
            n_parameters = int(len(ub))

        finite_lb = np.isfinite(lb) if lb is not None else np.zeros(0, dtype=bool)
        finite_ub = np.isfinite(ub) if ub is not None else np.zeros(0, dtype=bool)
        if lb is not None and ub is not None:
            fixed = finite_lb & finite_ub & np.isclose(lb, ub)
        else:
            fixed = np.zeros(0, dtype=bool)

        inequality_summary = self._constraint_group_summary(
            self._inequality_constraints
        )
        equality_summary = self._constraint_group_summary(
            self._equality_constraints
        )

        snapshot = {
            'state_revision': self.state_revision,
            'bounds': {
                'has_lb': lb is not None,
                'has_ub': ub is not None,
                'n_parameters': n_parameters,
                'n_finite_lower': int(np.count_nonzero(finite_lb)),
                'n_finite_upper': int(np.count_nonzero(finite_ub)),
                'n_fixed': int(np.count_nonzero(fixed)),
                'global': dict(self._bounds.get('global', {})),
                'source': self._bounds.get('source'),
                'config_file': self._bounds.get('config_file'),
            },
            'runtime_overrides': {
                'patch_constraints': int(len(self._runtime_patch_constraints)),
                'rake_faults': tuple(sorted(self._runtime_rake_limits)),
            },
            'source_constraint_report': copy.deepcopy(
                self._last_source_constraint_report
            ),
            'activation_flags': {
                'live': self._current_activation_flags(),
                'reconciled': copy.deepcopy(self._last_reconciled_flags),
                'stale': self._activation_flags_are_stale(),
            },
            'cache': {
                'inequality_valid': bool(self._combined_cache['inequality']['valid']),
                'equality_valid': bool(self._combined_cache['equality']['valid']),
            },
            'constraint_totals': {
                'inequality_groups': int(len(inequality_summary)),
                'inequality_rows': int(sum(item['rows'] for item in inequality_summary.values())),
                'equality_groups': int(len(equality_summary)),
                'equality_rows': int(sum(item['rows'] for item in equality_summary.values())),
            },
            'inequality_constraints': inequality_summary,
            'equality_constraints': equality_summary,
        }
        if include_matrices:
            snapshot['inequality_constraints'] = self.inequality_constraints
            snapshot['equality_constraints'] = self.equality_constraints
        if validate:
            snapshot['validation'] = self.validate()
        return snapshot

    # ------------------------------------------------------------------
    # YAML normalisation (static, 100 % identical in both subclasses)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_constraint_list(src_cfg):
        """Convert list-of-dicts or dict-of-dicts from YAML into ``{name: params}``.

        Parameters
        ----------
        src_cfg : None, list of mapping, or mapping
            One source's ``source_constraints`` declarations. Accepted forms
            are ``[{name: ..., type: ..., rule: ...}, ...]``,
            ``{name: {type: ..., rule: ...}}``, and a mapping containing a
            ``constraints`` list.

        Returns
        -------
        dict
            Constraint name mapped to a parameter dictionary. List-form
            ``name`` fields are removed from their parameter dictionaries.

        Raises
        ------
        ValueError
            If the structure is unsupported, a list entry is not a mapping, or
            list-form declarations contain duplicate names.

        Notes
        -----
        This helper normalizes container syntax only. Adapter-specific rule
        grammar and equality/inequality semantics are validated later.
        """
        if isinstance(src_cfg, list):
            result = {}
            for item in src_cfg:
                if not isinstance(item, Mapping):
                    raise ValueError(
                        "Each source constraint list entry must be a mapping"
                    )
                cname = item.get('name', f'constraint_{len(result)}')
                if cname in result:
                    raise ValueError(
                        f"Duplicate source constraint name '{cname}'"
                    )
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
        if src_cfg is None:
            return {}
        raise ValueError(
            "source_constraints entries must be a list or mapping"
        )

    # ------------------------------------------------------------------
    # Constraint CRUD  (default implementation – no mode guard)
    # ------------------------------------------------------------------

    def _normalise_linear_constraint_pair(self, A, b, *, kind):
        """Normalize and validate one linear ``A``/``b`` constraint pair.

        Parameters
        ----------
        A : array-like
            Two-dimensional real constraint matrix.
        b : array-like
            Right-hand side as a vector or a single row/column matrix.
        kind : str
            Human-readable constraint family used in error messages.

        Returns
        -------
        A_array, b_array : tuple of numpy.ndarray
            Independent finite floating-point copies with shapes ``(m, n)``
            and ``(m,)``.

        Raises
        ------
        TypeError
            If either input is nonnumeric or complex.
        ValueError
            If dimensions, row counts, active linear width, or finite-value
            requirements are violated.
        """
        raw_A = np.asarray(A)
        raw_b = np.asarray(b)
        if (
            not np.issubdtype(raw_A.dtype, np.number)
            or not np.issubdtype(raw_b.dtype, np.number)
            or np.iscomplexobj(raw_A)
            or np.iscomplexobj(raw_b)
        ):
            raise TypeError(f"{kind} constraint A and b must be real numeric arrays")

        A_array = np.asarray(A, dtype=float)
        b_array = np.asarray(b, dtype=float)
        if A_array.ndim != 2:
            raise ValueError(
                f"Constraint matrix A must be 2D, got shape {A_array.shape}"
            )
        if b_array.ndim == 1:
            pass
        elif b_array.ndim == 2 and 1 in b_array.shape:
            b_array = b_array.reshape(-1)
        else:
            raise ValueError(
                f"Constraint vector b must be 1D or a row/column vector, "
                f"got shape {b_array.shape}"
            )
        if A_array.shape[0] != b_array.size:
            raise ValueError(
                f"A.shape[0] ({A_array.shape[0]}) != len(b) ({b_array.size})"
            )
        expected_cols = self._expected_linear_constraint_columns()
        if expected_cols is not None and A_array.shape[1] != expected_cols:
            raise ValueError(
                f"Constraint matrix A has {A_array.shape[1]} columns; expected "
                f"{expected_cols} active linear parameters"
            )
        if not np.all(np.isfinite(A_array)) or not np.all(np.isfinite(b_array)):
            raise ValueError(f"{kind} constraint A and b must contain finite values")
        return A_array.copy(), b_array.copy()

    def _normalise_active_bounds_pair(self, lb, ub, *, expected_length, label):
        """Normalize a complete bounds pair for a numerical consumer.

        Parameters
        ----------
        lb, ub : array-like
            Lower and upper bounds.
        expected_length : int
            Required number of active parameters.
        label : str
            Consumer label used in exceptions.

        Returns
        -------
        lower, upper : tuple of numpy.ndarray
            Independent one-dimensional floating-point copies.

        Raises
        ------
        ValueError
            If either side is absent, lengths differ from
            ``expected_length``, values are nonfinite, or any lower bound
            exceeds its upper bound.
        """
        if lb is None or ub is None:
            raise ValueError(f"{label} requires both lower and upper bounds")
        lower = np.asarray(lb, dtype=float).reshape(-1)
        upper = np.asarray(ub, dtype=float).reshape(-1)
        if lower.size != expected_length or upper.size != expected_length:
            raise ValueError(
                f"{label} expected {expected_length} lower/upper values, got "
                f"{lower.size}/{upper.size}"
            )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError(f"{label} must contain finite values")
        if np.any(lower > upper):
            bad = np.where(lower > upper)[0]
            raise ValueError(
                f"{label} has lower bounds greater than upper bounds at "
                f"indices {bad[:10].tolist()}"
            )
        return lower.copy(), upper.copy()

    def _register_inequality_group(
        self,
        A: np.ndarray,
        b: np.ndarray,
        name: str,
        source: str = "manual",
        *,
        replace: bool = False,
        owner: str = "user",
        family: Optional[str] = None,
    ):
        """Register a named inequality group representing ``A @ x <= b``.

        Parameters
        ----------
        A : array-like, shape (m, n)
            Matrix in the manager's active linear parameter order.
        b : array-like, shape (m,)
            Inequality right-hand side.
        name : str
            Unique group identifier.
        source : str, default "manual"
            Provenance tag used by diagnostics and config reconciliation.
        replace : bool, default False
            Replace an existing inequality group with the same name.
        owner : {"user", "managed", "config"}
            State owner used for removal/reconciliation policy.
        family : str, optional
            Logical constraint family.

        Returns
        -------
        None

        Raises
        ------
        TypeError, ValueError
            If the name already exists without explicit replacement, or
            ``A``/``b`` fail
            numeric, shape, width, or finite-value validation.

        Notes
        -----
        Stored arrays are copies. A successful update invalidates both combined
        caches and advances ``state_revision``.
        """
        self._require_linear_constraints_active(
            "Adding a linear inequality"
        )
        if owner not in {'user', 'managed', 'config'}:
            raise ValueError(
                "constraint owner must be 'user', 'managed', or 'config'"
            )
        if owner == 'user' and name in self._RESERVED_MANAGED_GROUP_NAMES:
            raise ValueError(
                f"Constraint name '{name}' is reserved for a managed family"
            )
        if name in self._equality_constraints:
            raise ValueError(
                f"Constraint name '{name}' is already used by an equality "
                "group; names are globally unique"
            )
        existing = self._inequality_constraints.get(name)
        if existing is not None and not replace:
            raise ValueError(
                f"Inequality constraint '{name}' already exists. "
                "Use the explicit replace operation."
            )
        if (
            existing is not None
            and existing.get('owner', 'user') != owner
        ):
            raise ValueError(
                f"Constraint '{name}' is "
                f"{existing.get('owner', 'user')}-owned and cannot be "
                f"replaced by {owner}-owned state"
            )

        A, b = self._normalise_linear_constraint_pair(
            A, b, kind="Inequality"
        )

        self._inequality_constraints[name] = {
            'A': A.copy(),
            'b': b.copy(),
            'source': source,
            'owner': owner,
            'family': family,
            'shape': A.shape,
            'added_time': datetime.now(),
        }
        self._invalidate_constraint_cache()

        if self.verbose:
            print(f"[INQ] Added inequality constraint '{name}': "
                  f"{A.shape[0]} constraints (source: {source})")

    def _register_equality_group(
        self,
        A: np.ndarray,
        b: np.ndarray,
        name: str,
        source: str = "manual",
        *,
        replace: bool = False,
        owner: str = "user",
        family: Optional[str] = None,
    ):
        """Register a named equality group representing ``A @ x = b``.

        Parameters
        ----------
        A : array-like, shape (m, n)
            Matrix in the manager's active linear parameter order.
        b : array-like, shape (m,)
            Equality right-hand side.
        name : str
            Unique group identifier.
        source : str, default "manual"
            Provenance tag used by diagnostics and config reconciliation.
        replace : bool, default False
            Replace an existing equality group with the same name.
        owner : {"user", "managed", "config"}
            State owner used for removal/reconciliation policy.
        family : str, optional
            Logical constraint family.

        Returns
        -------
        None

        Raises
        ------
        TypeError, ValueError
            If the name already exists without explicit replacement, or
            ``A``/``b`` fail
            numeric, shape, width, or finite-value validation.

        Notes
        -----
        Stored arrays are copies. Duplicate equality rows are reconciled only
        when groups are combined.
        """
        self._require_linear_constraints_active(
            "Adding a linear equality"
        )
        if owner not in {'user', 'managed', 'config'}:
            raise ValueError(
                "constraint owner must be 'user', 'managed', or 'config'"
            )
        if owner == 'user' and name in self._RESERVED_MANAGED_GROUP_NAMES:
            raise ValueError(
                f"Constraint name '{name}' is reserved for a managed family"
            )
        if name in self._inequality_constraints:
            raise ValueError(
                f"Constraint name '{name}' is already used by an inequality "
                "group; names are globally unique"
            )
        existing = self._equality_constraints.get(name)
        if existing is not None and not replace:
            raise ValueError(
                f"Equality constraint '{name}' already exists. "
                "Use the explicit replace operation."
            )
        if (
            existing is not None
            and existing.get('owner', 'user') != owner
        ):
            raise ValueError(
                f"Constraint '{name}' is "
                f"{existing.get('owner', 'user')}-owned and cannot be "
                f"replaced by {owner}-owned state"
            )

        A, b = self._normalise_linear_constraint_pair(
            A, b, kind="Equality"
        )

        self._equality_constraints[name] = {
            'A': A.copy(),
            'b': b.copy(),
            'source': source,
            'owner': owner,
            'family': family,
            'shape': A.shape,
            'added_time': datetime.now(),
        }
        self._invalidate_constraint_cache()

        if self.verbose:
            print(f"[EQ] Added equality constraint '{name}': "
                  f"{A.shape[0]} constraints (source: {source})")

    def _replace_user_group(self, kind, A, b, *, name, source='user'):
        """Replace one existing user-owned raw linear group.

        Public ``replace_*`` calls require an existing group. Compiler code
        may still use idempotent ``replace=True`` registration when rebuilding
        a managed family that is absent on its first application.
        """
        if kind == 'inequality':
            groups = self._inequality_constraints
            other = self._equality_constraints
            register = self._register_inequality_group
        elif kind == 'equality':
            groups = self._equality_constraints
            other = self._inequality_constraints
            register = self._register_equality_group
        else:
            raise ValueError("kind must be 'inequality' or 'equality'")

        if name in other:
            raise ValueError(
                f"Constraint '{name}' exists as the other linear kind"
            )
        if name not in groups:
            raise ValueError(
                f"Constraint '{name}' does not exist; use the add operation"
            )
        if groups[name].get('owner', 'user') != 'user':
            raise ValueError(
                f"Constraint '{name}' is {groups[name].get('owner')}-owned "
                "and cannot be replaced through the raw user API"
            )
        register(
            A,
            b,
            name=name,
            source=source,
            replace=True,
            owner='user',
        )

    def _remove_group(
        self,
        name: str,
        *,
        expected_kind: Optional[str] = None,
        allow_managed: bool = False,
    ):
        """Remove one globally named resolved constraint group.

        Parameters
        ----------
        name : str
            Group name to remove.
        expected_kind : ``'inequality'``, ``'equality'``, or ``None``
            Optional assertion about the group kind.
        allow_managed : bool, default False
            Permit removal of config/managed groups from internal rebuild code.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``expected_kind`` is unsupported, no matching group exists, the
            group kind differs from ``expected_kind``, or a public caller
            attempts to remove a managed/config-owned group.

        Notes
        -----
        Equality and inequality groups share one global name space, so exactly
        one store can contain ``name``.
        """
        if expected_kind not in (None, 'inequality', 'equality'):
            raise ValueError(
                "expected_kind must be 'inequality', 'equality', or None"
            )
        if name in self._inequality_constraints:
            kind = 'inequality'
            groups = self._inequality_constraints
        elif name in self._equality_constraints:
            kind = 'equality'
            groups = self._equality_constraints
        else:
            raise ValueError(f"Constraint '{name}' not found")
        if expected_kind is not None and kind != expected_kind:
            raise ValueError(
                f"Constraint '{name}' is {kind}, not {expected_kind}"
            )
        owner = groups[name].get('owner', 'user')
        if owner != 'user' and not allow_managed:
            raise ValueError(
                f"Constraint '{name}' is {owner}-owned; use its family-specific "
                "update/clear operation"
            )
        del groups[name]
        self._invalidate_constraint_cache()
        if self.verbose:
            print(f"[X] Removed {kind} constraint: '{name}'")

    def _remove_groups_by_owner(self, owner, *, families=None):
        """Remove resolved groups by explicit ownership metadata.

        Parameters
        ----------
        owner : {"user", "managed", "config"}
            Group owner to remove.
        families : iterable of str, optional
            Restrict removal to logical families.

        Returns
        -------
        tuple of tuple
            Removed ``(constraint_type, group_name)`` pairs in traversal order.

        Notes
        -----
        Display ``source`` strings are intentionally ignored.
        """
        family_set = None if families is None else {
            str(family) for family in families
        }
        removed = []
        for constraint_type, groups in (
            ('inequality', self._inequality_constraints),
            ('equality', self._equality_constraints),
        ):
            for name, group in list(groups.items()):
                if (
                    group.get('owner', 'user') == owner
                    and (
                        family_set is None
                        or group.get('family') in family_set
                    )
                ):
                    del groups[name]
                    removed.append((constraint_type, name))

        if removed:
            self._invalidate_constraint_cache()
        return tuple(removed)

    def set_parameter_bounds_by_indices(
        self, indices, lower, upper, source: str = "manual", persist: bool = True
    ):
        """Set bounds for explicit model-vector indices.

        This is the low-level write path for helpers that resolve semantic
        parameters, such as data-correction transform components, to concrete
        solver columns.  ``indices`` are always in the full parameter vector
        used by the owning constraint manager.  Persistent entries participate
        in later full bounds rebuilds; selector-derived patch bounds pass
        ``persist=False`` because their declaration is stored separately.

        Parameters
        ----------
        indices : array-like of int
            Unique positions in the manager-owned bounds vector.
        lower, upper : scalar or array-like
            Values broadcast to ``indices`` or supplied one per index.
        source : str, default "manual"
            Provenance stored with persistent index declarations.
        persist : bool, default True
            Retain the index declaration for future full bounds rebuilds.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If bounds arrays are unavailable; indices are empty, duplicated, or
            out of range; values cannot broadcast; values are nonfinite; or
            any lower bound exceeds its upper bound.

        Notes
        -----
        This method writes resolved ``lb``/``ub`` immediately and advances
        ``state_revision``. With ``persist=False`` only the resolved arrays are
        changed; the owning higher-level declaration must recreate them later.
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
        if not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
            raise ValueError("Explicit bounds must contain finite values")
        if np.any(lower_array > upper_array):
            bad = np.where(lower_array > upper_array)[0]
            raise ValueError(
                "Lower bound is greater than upper bound for parameter "
                f"indices {index_array[bad].tolist()}"
            )

        self._bounds['lb'][index_array] = lower_array
        self._bounds['ub'][index_array] = upper_array
        if persist:
            for index, lb_value, ub_value in zip(index_array, lower_array, upper_array):
                self._bounds['parameter_bounds'][int(index)] = (
                    float(lb_value), float(ub_value), source
                )
        self._bounds['source'] = source
        self._bounds['applied_time'] = datetime.now()
        self._mark_bounds_changed()

    # ------------------------------------------------------------------
    # Combined constraints (DRY helper from SMC pattern)
    # ------------------------------------------------------------------

    def _deduplicate_equality_constraints(
        self, A: np.ndarray, b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Deduplicate equality rows and reject conflicting right-hand sides.

        Parameters
        ----------
        A : numpy.ndarray
            Combined equality matrix.
        b : numpy.ndarray
            Combined equality right-hand side.

        Returns
        -------
        A_unique, b_unique : tuple of numpy.ndarray
            Arrays containing the first occurrence of each row, using
            12-decimal comparison.

        Raises
        ------
        ValueError
            If numerically identical rows of ``A`` have different ``b`` values.

        Notes
        -----
        Exact duplicate ``[A | b]`` rows are removed with a warning emitted
        only by the designated warning process.
        """
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
        """Combine all named groups for one constraint family with caching.

        Parameters
        ----------
        constraint_type : {"inequality", "equality"}
            Internal group-store and cache key.

        Returns
        -------
        A, b : tuple of numpy.ndarray or tuple of None
            Vertically/horizontally combined matrices, or ``(None, None)`` when
            no groups exist. Cached results are returned as copies.

        Raises
        ------
        AttributeError, KeyError
            If an unsupported internal constraint type is requested.
        ValueError
            If equality-row deduplication finds conflicting right-hand sides.

        Notes
        -----
        Equality groups are deduplicated after stacking. Newly combined arrays
        are copied into the cache, so mutating the returned arrays cannot alter
        cached manager state.
        """
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
        """Return all inequality groups as one ``(A, b)`` pair.

        Returns
        -------
        tuple of numpy.ndarray or tuple of None
            Combined ``A @ m <= b``, or ``(None, None)`` when absent.
        """
        return self._get_combined_constraints('inequality')

    def get_combined_equality_constraints(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return all equality groups as one ``(A, b)`` pair.

        Returns
        -------
        tuple of numpy.ndarray or tuple of None
            Combined ``A @ m = b``, or ``(None, None)`` when absent.
        """
        return self._get_combined_constraints('equality')

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    @classmethod
    def _normalise_bounds_config_root(cls, loaded):
        """Return the canonical in-memory form of a bounds YAML mapping.

        Parameters
        ----------
        loaded : mapping
            Root mapping returned by the YAML parser.

        Returns
        -------
        dict
            Deep-copied configuration in which optional source-name mapping
            sections use ``{}`` instead of ``None``.

        Raises
        ------
        ValueError
            If a known mapping section contains a non-mapping value.

        Notes
        -----
        ``patch_constraints`` is intentionally excluded because its public
        grammar accepts either a mapping or a list. Scalar bounds such as
        ``lb``/``ub`` and array-like ``sigmas``/``alpha`` are also preserved.
        """
        config = copy.deepcopy(dict(loaded))
        for field in cls._BOUNDS_CONFIG_MAPPING_FIELDS:
            if field not in config:
                continue
            value = config[field]
            if value is None:
                config[field] = {}
            elif not isinstance(value, Mapping):
                raise ValueError(
                    f"Bounds config field '{field}' must be a mapping; "
                    f"got {type(value).__name__}"
                )
        return config

    def load_bounds_config(self, config_file: str, encoding: str = 'utf-8'):
        """Load bounds configuration from a YAML file.

        Subclasses may override ``_on_bounds_config_loaded`` to propagate
        the parsed config to their backend object.

        Parameters
        ----------
        config_file : str or path-like
            YAML bounds/constraint configuration.
        encoding : str, default "utf-8"
            Text encoding used to open the file.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If ``config_file`` does not exist.
        ValueError
            If the suffix is unsupported, the YAML root is not a mapping, or
            a known source-name mapping section has an invalid type.
        yaml.YAMLError
            If YAML parsing fails.

        Notes
        -----
        Loading replaces ``_bounds_config`` and records its path, but does not
        itself resolve bounds or generate matrices. Concrete initialization or
        ``apply_constraints_from_config`` performs that second step. Optional
        mapping sections written as a bare YAML key or ``null`` are normalized
        to an empty mapping before backend hooks run.
        """
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Bounds config file not found: {config_file}")

            with open(config_path, 'r', encoding=encoding) as f:
                if config_path.suffix.lower() in ('.yml', '.yaml'):
                    loaded = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")

            if loaded is None:
                loaded = {}
            if not isinstance(loaded, Mapping):
                raise ValueError(
                    "Bounds config root must be a YAML mapping; "
                    f"got {type(loaded).__name__}"
                )
            self._bounds_config = self._normalise_bounds_config_root(loaded)

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
        """Run a backend hook after ``_bounds_config`` is set or restored.

        Returns
        -------
        None

        Notes
        -----
        The base hook is a no-op. BLSE/SMC subclasses use it to refresh
        compatibility mirrors without sharing the manager's writable mapping.
        """

    # ------------------------------------------------------------------
    # Config summary (template method)
    # ------------------------------------------------------------------

    def _print_config_summary(self):
        """Print summary of loaded configuration.

        Collects common keys, then calls ``_get_extra_config_summary_items()``
        so subclasses can append backend-specific keys without copy-pasting
        the entire method.

        Returns
        -------
        None

        Notes
        -----
        This is diagnostic output only. It normalizes patch declarations to
        count them but does not apply selectors, bounds, or matrices.
        """
        if not self._bounds_config:
            return

        config_items = []
        if 'lb' in self._bounds_config or 'ub' in self._bounds_config:
            config_items.append("global bounds")
        mapping_summaries = (
            ('strikeslip', 'strike-slip bounds', 'fault'),
            ('dipslip', 'dip-slip bounds', 'fault'),
            ('poly', 'polynomial bounds', 'fault'),
            ('rake_angle', 'rake angle constraints', 'fault'),
            ('source_bounds', 'source component bounds', 'source'),
            ('source_constraints', 'source constraints', 'source'),
        )
        for key, label, item_name in mapping_summaries:
            declarations = self._bounds_config.get(key) or {}
            if declarations:
                config_items.append(
                    f"{label} for {len(declarations)} {item_name}(s)"
                )
        if 'patch_constraints' in self._bounds_config:
            patch_specs = self._normalise_patch_constraint_entries(
                self._bounds_config.get('patch_constraints')
            )
            config_items.append(f"patch-level bounds/rake overrides for {len(patch_specs)} segment(s)")

        # Subclass hook for extra keys (geometry, sigmas, alpha, etc.)
        config_items.extend(self._get_extra_config_summary_items())

        if config_items:
            print(f"   - Configuration contains: {', '.join(config_items)}")

    def _get_extra_config_summary_items(self) -> list:
        """Return backend-specific phrases for the config summary.

        Returns
        -------
        list of str
            Empty in the base implementation.
        """
        return []

    # ------------------------------------------------------------------
    # Validation (common structure, subclasses extend)
    # ------------------------------------------------------------------

    def validate(self) -> Dict:
        """Validate current bounds and constraints.

        Returns
        -------
        dict
            ``{"valid": bool, "errors": list, "warnings": list}``.
            Subclasses should call ``super().validate()`` and extend it.

        Notes
        -----
        Validation checks resolved bound ordering, finite matrix/vector values,
        row and active-column dimensions, and combined equality rank. It does
        not currently prove semantic agreement among adapter component order,
        position mappings, and assembled Green's-function columns. Equality
        rank evaluation may be expensive for large systems.
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
            if not np.all(np.isfinite(c['A'])) or not np.all(np.isfinite(c['b'])):
                result['errors'].append(
                    f"Inequality '{name}' contains NaN or Inf"
                )
                result['valid'] = False
            if c['A'].shape[0] != len(c['b']):
                result['errors'].append(
                    f"Inequality '{name}': A rows ({c['A'].shape[0]}) != len(b) ({len(c['b'])})")
                result['valid'] = False
            expected_cols = self._expected_linear_constraint_columns()
            if expected_cols is not None and c['A'].shape[1] != expected_cols:
                result['errors'].append(
                    f"Inequality '{name}': A columns ({c['A'].shape[1]}) != "
                    f"active linear parameters ({expected_cols})"
                )
                result['valid'] = False

        for name, c in self._equality_constraints.items():
            if not np.all(np.isfinite(c['A'])) or not np.all(np.isfinite(c['b'])):
                result['errors'].append(
                    f"Equality '{name}' contains NaN or Inf"
                )
                result['valid'] = False
            if c['A'].shape[0] != len(c['b']):
                result['errors'].append(
                    f"Equality '{name}': A rows ({c['A'].shape[0]}) != len(b) ({len(c['b'])})")
                result['valid'] = False
            expected_cols = self._expected_linear_constraint_columns()
            if expected_cols is not None and c['A'].shape[1] != expected_cols:
                result['errors'].append(
                    f"Equality '{name}': A columns ({c['A'].shape[1]}) != "
                    f"active linear parameters ({expected_cols})"
                )
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
    def state_revision(self) -> int:
        """Return the monotonic revision of resolved numerical state.

        Returns
        -------
        int
            Revision advanced by resolved bounds or constraint-group changes.
        """
        return int(self._state_revision)

    @property
    def lb(self) -> Optional[np.ndarray]:
        """Return a writable copy of the resolved lower bounds.

        Returns
        -------
        numpy.ndarray or None
            Copy of ``lb``. Mutating it does not update manager state.
        """
        return self._bounds['lb'].copy() if self._bounds['lb'] is not None else None

    @property
    def ub(self) -> Optional[np.ndarray]:
        """Return a writable copy of the resolved upper bounds.

        Returns
        -------
        numpy.ndarray or None
            Copy of ``ub``. Mutating it does not update manager state.
        """
        return self._bounds['ub'].copy() if self._bounds['ub'] is not None else None

    @property
    def bounds(self):
        """Read-only diagnostic view of bounds.

        Update bounds through manager methods such as ``set_global_bounds`` or
        inversion-level helpers.  The returned mapping is not a write API.

        Returns
        -------
        mapping
            Recursive read-only snapshot of declarations, resolved arrays, and
            provenance metadata.
        """
        return self._readonly_snapshot(self._bounds)

    @property
    def bounds_config(self):
        """Return a read-only snapshot of the loaded configuration.

        Returns
        -------
        mapping or None
            Recursive read-only config snapshot, or ``None`` before loading.
        """
        return (
            self._readonly_snapshot(self._bounds_config)
            if self._bounds_config is not None
            else None
        )

    @property
    def inequality_constraints(self) -> Dict[str, Dict]:
        """Return read-only copies of named inequality groups.

        Returns
        -------
        mapping
            Group name to read-only ``A``, ``b``, and provenance. Internal
            timestamps/cache metadata are intentionally omitted.
        """
        return self._readonly_snapshot({
            name: {
                'A': c['A'],
                'b': c['b'],
                'source': c.get('source'),
                'owner': c.get('owner', 'user'),
                'family': c.get('family'),
            }
            for name, c in self._inequality_constraints.items()
        })

    @property
    def equality_constraints(self) -> Dict[str, Dict]:
        """Return read-only copies of named equality groups.

        Returns
        -------
        mapping
            Group name to read-only ``A``, ``b``, and provenance. Internal
            timestamps/cache metadata are intentionally omitted.
        """
        return self._readonly_snapshot({
            name: {
                'A': c['A'],
                'b': c['b'],
                'source': c.get('source'),
                'owner': c.get('owner', 'user'),
                'family': c.get('family'),
            }
            for name, c in self._equality_constraints.items()
        })
