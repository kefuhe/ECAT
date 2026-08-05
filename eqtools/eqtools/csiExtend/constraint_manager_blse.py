import numpy as np
import copy
import pandas as pd
from collections.abc import Mapping
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from datetime import datetime
from pathlib import Path

from .constraint_manager_base import ConstraintManagerBase
from .source_adapters import FaultAdapter


class ConstraintManagerBLSE(ConstraintManagerBase):
    """Manage bounds and linear constraints in the BLSE parameter space.

    All resolved bound vectors and constraint-matrix columns use the same
    ordering as ``solver.lsq_parameters``.  Source component locations are
    obtained from ``solver.slip_positions`` and source adapters; polynomial
    locations come from ``solver.poly_positions``.

    The manager is the writable source of truth.  :meth:`sync_to_solver`
    refreshes legacy solver attributes for compatibility, but those mirrored
    attributes do not define an independent constraint state.
    """
    
    def __init__(self, solver, config=None, verbose: bool = True):
        """Initialize a BLSE constraint manager.

        Parameters
        ----------
        solver : object
            BLSE/VCE solver whose linear model layout defines every bound and
            constraint index.
        config : object, optional
            Configuration object providing activation flags and optional
            interseismic declarations.
        verbose : bool
            Whether to print application and validation summaries.
        """
        self.solver = solver
        self.config = config
        self.verbose = verbose
        
        # Shared storage (constraints, cache, common bounds keys)
        self._init_shared_storage()
        
        if self.verbose:
            print("[OK] ConstraintManagerBLSE initialized")

    def _on_bounds_config_loaded(self):
        """Mirror the loaded bounds declaration onto the legacy solver API.

        Returns
        -------
        None

        Notes
        -----
        This overrides the base no-op hook.
        A deep copy prevents later mutation of the manager's declaration
        through ``solver.bounds_config``.
        """
        self.solver.bounds_config = copy.deepcopy(self._bounds_config)

    def _build_rake_sector_matrix(
        self,
        rake_limits: Dict[str, Tuple[float, float]],
    ):
        """Build the resolved BLSE rake-sector matrix without registering it.

        Parameters
        ----------
        rake_limits : dict
            Fault names mapped to ``(minimum_rake, maximum_rake)`` in degrees.
            Patch-level rake declarations maintained by the base manager are
            resolved at the same time.
        Raises
        ------
        TypeError
            If a rake declaration has an unsupported type.
        ValueError
            If a fault, selector, angle, interval, or source component layout
            is invalid.

        Notes
        -----
        The generated rows have the form ``A @ x <= b`` and the matrix columns
        follow the complete BLSE linear parameter vector.
        """
        intervals_by_fault = self._resolve_rake_intervals_by_patch(rake_limits)
        return self._generate_rake_inequality_constraints_from_intervals(
            intervals_by_fault
        )

    def _rebuild_rake_constraints(self, additional_rake_limits: Dict = None):
        """Resolve configured and runtime rake sectors into one matrix group.

        Parameters
        ----------
        additional_rake_limits : dict, optional
            Call-specific fault rake sectors.  These override file-level
            fault declarations for matching names.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If a resolved rake declaration has an unsupported type.
        ValueError
            If a resolved fault, selector, interval, or parameter layout is
            invalid.

        Notes
        -----
        Fault-level precedence is bounds config, then this argument, then
        persistent runtime declarations.  Patch-level declarations are
        subsequently resolved by the shared base implementation.
        """
        try:
            # Collect rake limits from multiple sources
            final_rake_limits = {}
            
            # From bounds config
            if (
                self._config_flag_enabled('use_rake_angle_constraints')
                and self._bounds_config
                and 'rake_angle' in self._bounds_config
            ):
                final_rake_limits.update(self._bounds_config['rake_angle'])
            
            # From additional parameters
            if additional_rake_limits:
                final_rake_limits.update(additional_rake_limits)

            # From script/API updates. These are declaration-layer updates
            # and are resolved together with any patch-level overrides below.
            if self._runtime_rake_limits:
                final_rake_limits.update(self._runtime_rake_limits)
            
            has_patch_rake = any(
                'rake_angle' in spec for spec in self._iter_patch_constraint_specs()
            )
            if not final_rake_limits and not has_patch_rake:
                if 'rake_sector' in self._inequality_constraints:
                    self._remove_group(
                        'rake_sector',
                        expected_kind='inequality',
                        allow_managed=True,
                    )
                if self.verbose:
                    print("[i]  No rake angle limits specified")
                return
            
            source = 'bounds_config'
            if additional_rake_limits:
                source += ' + additional_limits'
            A, b, constrained_fault_names = self._build_rake_sector_matrix(
                final_rake_limits
            )
            if A.size == 0:
                if 'rake_sector' in self._inequality_constraints:
                    self._remove_group(
                        'rake_sector',
                        expected_kind='inequality',
                        allow_managed=True,
                    )
                return
            self._register_inequality_group(
                A,
                b,
                name='rake_sector',
                source=source,
                replace=True,
                owner='managed',
                family='rake_sector',
            )
            if self.verbose:
                print(
                    f"[INQ] Applied rake angle constraints: {A.shape[0]} "
                    f"constraints for {len(constrained_fault_names)} fault(s)"
                )

        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply rake constraints: {e}")
            raise

    def set_fixed_rake_constraints(self, fixed_rake: Dict[str, float], source: str = "manual"):
        """Replace the fixed-rake equality group for selected faults.

        Parameters
        ----------
        fixed_rake : dict
            Fault names mapped to finite rake angles in degrees.  An empty
            mapping is a no-op; use :meth:`clear_fixed_rake_constraints` to
            remove an existing group.
        source : str
            Provenance label stored with the generated constraint group.

        Raises
        ------
        TypeError
            If ``fixed_rake`` is not a mapping or an angle is not numeric.
        ValueError
            If a source is unknown, is not a Fault, has a non-finite angle, or
            lacks the strike-slip/dip-slip layout required by fixed rake.

        Notes
        -----
        Each patch contributes
        ``ss * sin(rake) - ds * cos(rake) = 0``.  Matrix columns use the BLSE
        linear parameter order.
        """
        try:
            if not isinstance(fixed_rake, Mapping):
                raise TypeError(
                    "fixed_rake must be a mapping of fault name to angle"
                )
            # Empty input is an explicit no-op.  Deletion is handled only by
            # clear_fixed_rake_constraints(), keeping set and clear distinct.
            if not fixed_rake:
                return

            fault_names = {fault.name for fault in self.solver.faults}
            unknown = sorted(set(fixed_rake) - fault_names)
            if unknown:
                raise ValueError(
                    f"fixed_rake references unknown fault(s): {unknown}"
                )
            non_fault = sorted(
                name for name in fixed_rake
                if self._get_source_type(name) != 'Fault'
            )
            if non_fault:
                raise ValueError(
                    "fixed_rake only applies to Fault sources; invalid "
                    f"source(s): {non_fault}"
                )
            for fault_name, rake in fixed_rake.items():
                try:
                    rake_value = float(rake)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"fixed_rake for '{fault_name}' must be numeric"
                    ) from exc
                if not np.isfinite(rake_value):
                    raise ValueError(
                        f"fixed_rake for '{fault_name}' must be finite"
                    )

            constrained_fault_names = list(fixed_rake)
            
            # Calculate total patches for constrained faults
            npatch = 0
            Np = self._get_linear_matrix_n_parameters()
            
            # Get constrained fault objects. Only Fault-type sources have rake.
            constrained_faults = [
                fault for fault in self.solver.faults
                if fault.name in constrained_fault_names
            ]
            constrained_fault_names = [f.name for f in constrained_faults]
            
            for ifault in constrained_faults:
                npatch += len(ifault.patch)
            
            # Create constraint matrices
            Aeq = np.zeros((npatch, Np))
            beq = np.zeros((npatch,))
            
            patch_count = 0
            for ifault in constrained_faults:
                irake = fixed_rake[ifault.name]
                inpatch = len(ifault.patch)
                start = self._get_linear_matrix_source_start(ifault.name)
                adapter = getattr(self.solver, 'adapters', {}).get(ifault.name)
                ss_start, ds_start = self._rake_component_starts(
                    ifault, start, inpatch, adapter=adapter
                )
                rake_angle = np.deg2rad(float(irake))
                
                # Generate equality constraints for each patch
                for i in range(inpatch):
                    # Fixed rake constraint: ss*sin(rake) - ds*cos(rake) = 0
                    Aeq[patch_count + i, ss_start + i] = np.sin(rake_angle)
                    Aeq[patch_count + i, ds_start + i] = -np.cos(rake_angle)
                
                patch_count += inpatch
            
            self._register_equality_group(
                Aeq,
                beq,
                name='fixed_rake',
                source=source,
                replace=True,
                owner='managed',
                family='fixed_rake',
            )
            
            if self.verbose:
                print(f"[EQ] Applied fixed rake constraints: {Aeq.shape[0]} constraints for {len(constrained_fault_names)} fault(s)")
                for fault_name, rake_angle in fixed_rake.items():
                    if fault_name in constrained_fault_names:
                        print(f"   - {fault_name}: rake = {rake_angle} deg")
                        
        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply fixed rake constraints: {e}")
            raise

    def clear_fixed_rake_constraints(self, *, sync=True):
        """Remove the fixed-rake equality group if it exists.

        Parameters
        ----------
        sync : bool
            Whether to refresh the solver's legacy constraint attributes after
            removal.

        Returns
        -------
        bool
            ``True`` when a group was removed, otherwise ``False``.
        """
        if 'fixed_rake' not in self._equality_constraints:
            return False
        self._remove_group(
            'fixed_rake',
            expected_kind='equality',
            allow_managed=True,
        )
        if sync:
            self.sync_to_solver()
        return True

    def apply_euler_cap_constraints(self):
        """Generate and replace configured Euler-cap inequalities.

        Non-Fault entries are excluded before matrix generation.  The method
        is a no-op when the feature is disabled or no rows are generated.

        Raises
        ------
        Exception
            Propagates configuration, dataset, and matrix-generation errors.
        """
        try:
            interseismic_config = getattr(self.config, 'interseismic_config', {})
            if not interseismic_config.get('cap_constraints', {}).get('enabled', False):
                if self.verbose:
                    print("[i]  Interseismic Euler-cap constraints not enabled")
                return

            from .euler_inequality_constraints import generate_euler_cap_constraints

            active_config = copy.deepcopy(interseismic_config)
            cap_faults = active_config.get('cap_constraints', {}).get('faults', {})
            non_fault_names = [
                fn for fn in list(cap_faults)
                if self._get_source_type(fn) != 'Fault'
            ]
            for fn in non_fault_names:
                if self.verbose:
                    print(f"[!]  Warning: Euler-cap constraint skipping non-Fault source '{fn}'")
                del cap_faults[fn]

            all_datasets = self.config.geodata['data']
            A_ineq, b_ineq = generate_euler_cap_constraints(self.solver, active_config, all_datasets)

            if A_ineq is not None and A_ineq.size > 0:
                self._register_inequality_group(
                    A_ineq,
                    b_ineq,
                    name='euler_cap_constraints',
                    source='interseismic_config.cap_constraints',
                    replace=True,
                    owner='config',
                    family='interseismic_cap',
                )

                if self.verbose:
                    print(f"[INQ] Applied Euler-cap constraints: {A_ineq.shape[0]} constraints")
                    configured_faults = active_config.get('cap_constraints', {}).get('configured_faults', [])
                    print(f"   - Constrained faults: {configured_faults}")
            elif self.verbose:
                print("[i]  No Euler-cap constraints generated")

        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply Euler-cap constraints: {e}")
            raise

    def apply_interseismic_backslip_constraints(self):
        """Delegate configured hard backslip constraints to the solver.

        Non-Fault sources are skipped. Each remaining declaration is compiled
        through the inversion's private config-owned backslip path without
        exposing registry ownership on the public facade.

        Raises
        ------
        Exception
            Propagates malformed declarations and solver-side validation
            errors.
        """
        constraints = getattr(self.config, 'interseismic_config', {}).get('backslip_constraints', [])
        for index, spec in enumerate(constraints):
            if self._get_source_type(spec['fault']) != 'Fault':
                if self.verbose:
                    print(f"[!]  Warning: Interseismic backslip constraint skipping non-Fault source '{spec['fault']}'")
                continue
            self.solver._apply_interseismic_backslip_constraint(
                spec['fault'],
                spec['state'],
                selector=spec.get('selector'),
                component=spec.get('component', 'strikeslip'),
                coupling=spec.get('coupling'),
                value=spec.get('value'),
                name=spec.get('name', f"interseismic_backslip_{index}"),
                source='interseismic_config.backslip_constraints',
                replace=True,
                owner='config',
                require_existing=False,
            )

    def apply_interseismic_block_constraints(self):
        """Generate and replace block-level Euler-sharing equalities.

        The generated matrix spans ``solver.lsq_parameters``.  No group is
        installed when the generator returns no rows.

        Raises
        ------
        Exception
            Propagates configuration and matrix-generation errors.
        """
        interseismic_config = getattr(self.config, 'interseismic_config', {})

        try:
            from .interseismic_parameter_model import generate_block_euler_equality_constraints

            A_eq, b_eq = generate_block_euler_equality_constraints(
                self.solver,
                interseismic_config,
                n_total=self._get_linear_matrix_n_parameters(),
            )
            if A_eq is None or A_eq.size == 0:
                if self.verbose:
                    print("[i]  No interseismic block Euler-sharing constraints generated")
                return
            self._register_equality_group(
                A_eq,
                b_eq,
                name='interseismic_block_euler_constraints',
                source='interseismic_config.blocks',
                replace=True,
                owner='config',
                family='interseismic_blocks',
            )
        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply interseismic block constraints: {e}")
            raise

    def apply_source_constraints_from_config(self):
        """Apply adapter-defined source constraints from the loaded config.

        Returns
        -------
        list of str
            Names of the inequality or equality groups installed by the base
            declaration resolver.

        Raises
        ------
        TypeError
            If the ``source_constraints`` declaration has an invalid shape.
        ValueError
            If a source, rule, matrix shape, or constraint type is invalid.

        Notes
        -----
        Adapters translate source-specific rules into matrices whose columns
        occupy the complete BLSE parameter space.
        """
        return self._apply_source_constraint_declarations()

    def _apply_constraint_config(
        self,
        bounds_config_file: str = None,
        rake_limits: Dict = None,
        encoding: str = 'utf-8',
    ):
        """Reconcile all file-owned declarations as one transaction.

        Parameters
        ----------
        bounds_config_file : str, optional
            Bounds YAML to load before reconciliation.  If omitted, the
            already loaded declaration is used.
        rake_limits : dict, optional
            Call-specific fault rake sectors layered over the file values.
        encoding : str
            Text encoding used when loading ``bounds_config_file``.

        Raises
        ------
        OSError
            If the requested configuration file cannot be read.
        TypeError
            If any declaration has an unsupported structure.
        ValueError
            If any bound, selector, source, matrix, or activation state fails
            validation.

        Notes
        -----
        Previous groups owned by configuration sources are replaced, while
        script-created groups with other provenance are retained.  On failure,
        mutable manager state is restored to the pre-call snapshot.
        """
        if self.verbose:
            print("\n[RUN] Applying all constraints from configuration...")

        snapshot = self._snapshot_mutable_state()
        try:
            if bounds_config_file is not None:
                self.load_bounds_config(bounds_config_file, encoding)

            # Reconcile file-owned groups before rebuilding the current
            # declaration.  Script-added groups have different provenance and
            # are intentionally preserved.
            self._remove_groups_by_owner(
                'config',
                families={
                    'source_constraints',
                    'interseismic_blocks',
                    'interseismic_cap',
                    'interseismic_backslip',
                },
            )

            if self._bounds_config is not None:
                self._rebuild_config_bounds()

            self._rebuild_rake_constraints(rake_limits)

            interseismic_config = getattr(
                self.config, 'interseismic_config', {}
            ) or {}
            if interseismic_config.get('blocks', {}).get('enabled', False):
                self.apply_interseismic_block_constraints()
            if interseismic_config.get('cap_constraints', {}).get('enabled', False):
                self.apply_euler_cap_constraints()
            if interseismic_config.get('backslip_constraints'):
                self.apply_interseismic_backslip_constraints()

            self.apply_source_constraints_from_config()

            self._validate_or_raise()
            self._mark_activation_flags_reconciled()
        except Exception:
            self._restore_mutable_state(snapshot)
            raise

        if self.verbose:
            print("[OK] All constraints applied successfully")
            self.print_summary()

    def set_global_bounds(self, lb: float = None, ub: float = None, source: str = "manual"):
        """Set default bounds for otherwise unassigned BLSE parameters.

        Parameters
        ----------
        lb : float, optional
            Finite global lower bound.
        ub : float, optional
            Finite global upper bound.
        source : str
            Human-readable provenance used in verbose output.

        Raises
        ------
        ValueError
            If either value is non-finite or ``lb > ub``.

        Notes
        -----
        Global bounds have the lowest precedence.  Fault/source, explicit
        parameter-index, and patch declarations overwrite them during rebuild.
        """
        if lb is not None:
            self._require_finite(lb, "global lower bound")
            lb = float(lb)
        if ub is not None:
            self._require_finite(ub, "global upper bound")
            ub = float(ub)
        if lb is not None and ub is not None and lb > ub:
            raise ValueError("Global lower bound should be less than upper bound")
        
        if lb is not None:
            self._bounds['global']['lb'] = lb
        if ub is not None:
            self._bounds['global']['ub'] = ub
        
        self._request_bounds_rebuild()
        
        if self.verbose:
            print(f"[GLB] Set global bounds: lb={lb}, ub={ub} (source: {source})")

    def set_fault_slip_bounds(self, fault_name: str, strikeslip: Tuple[float, float] = None, 
                            dipslip: Tuple[float, float] = None, source: str = "manual"):
        """Set fault-level strike-slip and/or dip-slip declarations.

        Parameters
        ----------
        fault_name : str
            Existing source name in ``solver.faults``.
        strikeslip : tuple, optional
            Finite ``(lower, upper)`` bounds for all strike-slip parameters.
        dipslip : tuple, optional
            Finite ``(lower, upper)`` bounds for all dip-slip parameters.
        source : str
            Human-readable provenance used in verbose output.

        Raises
        ------
        ValueError
            If the source is unknown, either pair is malformed or non-finite,
            or a lower bound exceeds its upper bound.

        Notes
        -----
        Declarations are stored by component and expanded using adapter-aware
        slices on the next bounds rebuild.
        """
        if not self._fault_exists(fault_name):
            raise ValueError(f"Fault '{fault_name}' not found in solver")
        
        if strikeslip is not None:
            values = self._require_finite(
                strikeslip, f"{fault_name} strike-slip bounds"
            )
            if values.shape != (2,):
                raise ValueError("Strike-slip bounds must be [lb, ub]")
            strikeslip = (float(values[0]), float(values[1]))
        if dipslip is not None:
            values = self._require_finite(
                dipslip, f"{fault_name} dip-slip bounds"
            )
            if values.shape != (2,):
                raise ValueError("Dip-slip bounds must be [lb, ub]")
            dipslip = (float(values[0]), float(values[1]))

        if strikeslip is not None and strikeslip[0] > strikeslip[1]:
            raise ValueError(
                f"Strike-slip lower bound ({strikeslip[0]}) > upper bound ({strikeslip[1]})"
            )
        if dipslip is not None and dipslip[0] > dipslip[1]:
            raise ValueError(
                f"Dip-slip lower bound ({dipslip[0]}) > upper bound ({dipslip[1]})"
            )
        if strikeslip is not None:
            self._bounds['strikeslip'][fault_name] = strikeslip
        if dipslip is not None:
            self._bounds['dipslip'][fault_name] = dipslip

        self._request_bounds_rebuild()
        
        if self.verbose:
            print(f"[*] Set slip bounds for '{fault_name}': ss={strikeslip}, ds={dipslip} (source: {source})")

    def set_fault_poly_bounds(self, fault_name: str, poly_bounds: Tuple[float, float], source: str = "manual"):
        """Set one uniform bound pair for a source's polynomial block.

        Parameters
        ----------
        fault_name : str
            Existing source name in ``solver.faults``.
        poly_bounds : tuple
            Finite ``(lower, upper)`` pair applied to all polynomial
            coefficients of the source.
        source : str
            Human-readable provenance used in verbose output.

        Raises
        ------
        ValueError
            If the source is unknown, the pair is malformed or non-finite, or
            its lower bound exceeds its upper bound.
        """
        if not self._fault_exists(fault_name):
            raise ValueError(f"Fault '{fault_name}' not found in solver")
        
        values = self._require_finite(
            poly_bounds, f"{fault_name} polynomial bounds"
        )
        if values.shape != (2,):
            raise ValueError("Polynomial bounds must be [lb, ub]")
        plb, pub = float(values[0]), float(values[1])
        if plb > pub:
            raise ValueError(f"Polynomial lower bound ({plb}) > upper bound ({pub})")
        
        self._bounds['poly'][fault_name] = (plb, pub)
        self._request_bounds_rebuild()
        
        if self.verbose:
            print(f"[GEO] Set poly bounds for '{fault_name}': {poly_bounds} (source: {source})")

    def set_source_component_bounds(self, source_name: str, comp_bounds: Dict[str, Tuple[float, float]], 
                                     source: str = "manual"):
        """
        Set per-component bounds for any source type using source adapters.
        
        Parameters
        ----------
        source_name : str
            Name of the source (fault/pressure/sbarbot).
        comp_bounds : dict
            {component_name: (lb, ub)}; component names come from
            adapter.get_param_names(), e.g. {'eps12': (-1e-4, 1e-4)}.
        source : str
            Source description for audit trail.

        Raises
        ------
        ValueError
            If the source or adapter is missing, a supplied component pair is
            malformed or non-finite, or a lower bound exceeds its upper bound.

        Notes
        -----
        Only component names declared by the adapter are resolved.  Their
        slices follow adapter order within ``solver.slip_positions``.
        """
        if not self._fault_exists(source_name):
            raise ValueError(f"Source '{source_name}' not found in solver")
        
        if not hasattr(self.solver, 'adapters') or source_name not in self.solver.adapters:
            raise ValueError(f"No adapter found for source '{source_name}'")
        
        adapter = self.solver.adapters[source_name]
        params_per_comp = adapter.get_n_params_per_component()
        slip_st, _ = self.solver.slip_positions[source_name]
        
        resolved = {}
        offset = slip_st
        for comp_name in adapter.get_param_names():
            n = params_per_comp[comp_name]
            if comp_name in comp_bounds:
                values = self._require_finite(
                    comp_bounds[comp_name],
                    f"{source_name}.{comp_name} bounds",
                )
                if values.shape != (2,):
                    raise ValueError(
                        f"Bounds for {source_name}.{comp_name} must be [lb, ub]"
                    )
                clb, cub = float(values[0]), float(values[1])
                if clb > cub:
                    raise ValueError(f"Lower bound ({clb}) > upper bound ({cub}) for {comp_name}")
                resolved[comp_name] = (clb, cub)
            offset += n

        self._bounds['source_bounds'][source_name] = resolved
        self._request_bounds_rebuild()
        
        if self.verbose:
            print(f"[SRC] Set component bounds for '{source_name}': {comp_bounds} (source: {source})")

    def _rebuild_config_bounds(self):
        """Replace file-owned bound declarations and rebuild resolved arrays.

        File-level global, fault component, polynomial, and source-component
        declarations are reset before applying the current config.  Persistent
        runtime index and patch declarations remain available and therefore
        retain their higher precedence.

        Raises
        ------
        TypeError
            If a bound declaration has an unsupported structure.
        ValueError
            If a bound, source, component, selector, or resolved state is
            invalid.

        Notes
        -----
        The operation is transactional: a failure restores the prior manager
        state.  If bounds are disabled by config, only runtime declarations are
        rebuilt.
        """
        if self._bounds_config is None:
            if self.verbose:
                print("[!]  No bounds config loaded")
            return
        
        snapshot = self._snapshot_mutable_state()
        try:
            # Loading a bounds file replaces the previous coarse declarations;
            # runtime patch declarations remain explicit script-side overrides.
            self._bounds['global'] = {'lb': None, 'ub': None}
            for key in ('strikeslip', 'dipslip', 'poly', 'source_bounds', 'patch_constraints'):
                self._bounds[key] = {}

            if not self._config_flag_enabled('use_bounds_constraints'):
                self._rebuild_resolved_bounds(source='runtime_bounds_only')
                self._validate_or_raise()
                return

            with self.batch_bounds_update():
                lb = self._bounds_config.get('lb', None)
                ub = self._bounds_config.get('ub', None)
                if lb is not None or ub is not None:
                    self.set_global_bounds(lb, ub, source="config_file")

                strikeslip_config = self._bounds_config.get('strikeslip', {})
                dipslip_config = self._bounds_config.get('dipslip', {})
                all_slip_faults = set(strikeslip_config) | set(dipslip_config)
                for fault_name in all_slip_faults:
                    if self._fault_exists(fault_name):
                        if self._get_source_type(fault_name) != 'Fault':
                            if self.verbose:
                                print(f"[!]  Warning: '{fault_name}' is not a Fault source, "
                                      f"skipping strikeslip/dipslip bounds. Use 'source_bounds' instead.")
                            continue
                        self.set_fault_slip_bounds(
                            fault_name,
                            strikeslip_config.get(fault_name),
                            dipslip_config.get(fault_name),
                            source="config_file",
                        )

                for fault_name, poly_bounds in self._bounds_config.get('poly', {}).items():
                    if self._fault_exists(fault_name):
                        self.set_fault_poly_bounds(fault_name, poly_bounds, source="config_file")

                for source_name, comp_bounds in self._bounds_config.get('source_bounds', {}).items():
                    if self._fault_exists(source_name):
                        self.set_source_component_bounds(source_name, comp_bounds, source="config_file")
            self._bounds['source'] = "config_file"
            self._bounds['applied_time'] = datetime.now()
            self._validate_or_raise()
        except Exception:
            self._restore_mutable_state(snapshot)
            raise

    def _rebuild_resolved_bounds(self, source="resolved_declarations"):
        """Resolve every stored declaration into fresh BLSE bound vectors.

        Parameters
        ----------
        source : str
            Provenance label recorded on the resolved state.

        Notes
        -----
        Precedence is global, fault slip, polynomial, source component,
        explicit parameter index, then patch.  Later layers overwrite earlier
        values only at their selected indices.
        """
        self._bounds['lb'] = None
        self._bounds['ub'] = None
        self._initialize_bounds_arrays()

        global_bounds = self._bounds['global']
        self._apply_global_bounds_to_arrays(global_bounds.get('lb'), global_bounds.get('ub'))

        fault_names = set(self._bounds['strikeslip']) | set(self._bounds['dipslip'])
        for fault_name in fault_names:
            if fault_name in self._bounds['strikeslip']:
                self._apply_strikeslip_bounds(fault_name, self._bounds['strikeslip'][fault_name])
            if fault_name in self._bounds['dipslip']:
                self._apply_dipslip_bounds(fault_name, self._bounds['dipslip'][fault_name])

        for fault_name, bounds in self._bounds['poly'].items():
            self._apply_poly_bounds(fault_name, bounds)
        for source_name, comp_bounds in self._bounds['source_bounds'].items():
            self._apply_source_component_bounds(source_name, comp_bounds)
        for index, (lower, upper, _) in self._bounds['parameter_bounds'].items():
            self._bounds['lb'][index] = lower
            self._bounds['ub'][index] = upper

        self._bounds['patch_constraints'] = {}
        self.apply_patch_bounds(source=source)
        self._bounds['source'] = source
        self._bounds['applied_time'] = datetime.now()
        self._mark_bounds_changed()

    def _initialize_bounds_arrays(self):
        """Allocate unresolved BLSE bound vectors when needed.

        ``solver.lsq_parameters`` is authoritative.  The adapter/source
        fallback exists only for compatible solver objects that do not expose
        that aggregate count.

        Returns
        -------
        None
        """
        if hasattr(self.solver, 'lsq_parameters'):
            n_params = self.solver.lsq_parameters
        else:
            # Fallback: calculate from adapters if available, else from source attributes
            n_params = 0
            for fault in self.solver.faults:
                if hasattr(self.solver, 'adapters') and fault.name in self.solver.adapters:
                    n_params += self.solver.adapters[fault.name].get_n_source_params()
                elif hasattr(fault, 'patch') and hasattr(fault, 'slipdir'):
                    n_params += len(fault.patch) * len(FaultAdapter._canonicalize_slipdir(fault.slipdir))
                elif hasattr(fault, 'volumes') and hasattr(fault, 'strain_components'):
                    n_params += len(fault.volumes) * len(fault.strain_components)
                else:
                    n_params += 1  # Point source (Pressure)
                if hasattr(fault, 'numberofpolys'):
                    n_params += sum(fault.numberofpolys.values())
        
        if self._bounds['lb'] is None:
            self._bounds['lb'] = np.ones(n_params) * np.nan
        if self._bounds['ub'] is None:
            self._bounds['ub'] = np.ones(n_params) * np.nan

    def _apply_global_bounds_to_arrays(self, lb: float = None, ub: float = None):
        """Fill currently undefined entries with global defaults.

        Parameters
        ----------
        lb, ub : float, optional
            Lower and upper defaults.  Existing finite declarations are not
            overwritten.
        """
        self._initialize_bounds_arrays()
        
        if lb is not None:
            self._bounds['lb'][np.isnan(self._bounds['lb'])] = lb
        if ub is not None:
            self._bounds['ub'][np.isnan(self._bounds['ub'])] = ub

    def _apply_strikeslip_bounds(self, fault_name: str, bounds: Tuple[float, float]):
        """Write a fault's strike-slip pair to its adapter-resolved slice.

        Raises
        ------
        ValueError
            If the source layout has no strike-slip component.
        """
        if hasattr(self.solver, 'slip_positions'):
            st, _ = self.solver.slip_positions[fault_name]
            fault = next(f for f in self.solver.faults if f.name == fault_name)
            adapter = getattr(self.solver, 'adapters', {}).get(fault_name)
            component_slices = self._source_component_slices(
                fault, st, adapter=adapter
            )
            if 'strikeslip' not in component_slices:
                raise ValueError(
                    f"Fault '{fault_name}' has no strikeslip component for bounds"
                )
            slb, sub = bounds
            ss_slice = component_slices['strikeslip']
            self._bounds['lb'][ss_slice] = slb
            self._bounds['ub'][ss_slice] = sub

    def _apply_dipslip_bounds(self, fault_name: str, bounds: Tuple[float, float]):
        """Write a fault's dip-slip pair to its adapter-resolved slice.

        Raises
        ------
        ValueError
            If the source layout has no dip-slip component.
        """
        if hasattr(self.solver, 'slip_positions'):
            st, _ = self.solver.slip_positions[fault_name]
            fault = next(f for f in self.solver.faults if f.name == fault_name)
            adapter = getattr(self.solver, 'adapters', {}).get(fault_name)
            component_slices = self._source_component_slices(
                fault, st, adapter=adapter
            )
            if 'dipslip' not in component_slices:
                raise ValueError(
                    f"Fault '{fault_name}' has no dipslip component for bounds"
                )
            dlb, dub = bounds
            ds_slice = component_slices['dipslip']
            self._bounds['lb'][ds_slice] = dlb
            self._bounds['ub'][ds_slice] = dub

    def _apply_poly_bounds(self, fault_name: str, bounds: Tuple[float, float]):
        """Write a uniform pair to the source polynomial slice, if exposed."""
        if hasattr(self.solver, 'poly_positions'):
            st, se = self.solver.poly_positions[fault_name]
            plb, pub = bounds
            self._bounds['lb'][st:se] = plb
            self._bounds['ub'][st:se] = pub

    def _apply_source_component_bounds(self, source_name, comp_bounds):
        """Write validated component pairs in adapter parameter order.

        Parameters
        ----------
        source_name : str
            Source whose slip/source block receives the values.
        comp_bounds : mapping
            Adapter component names mapped to resolved ``(lower, upper)``
            pairs.
        """
        adapter = self.solver.adapters[source_name]
        params_per_comp = adapter.get_n_params_per_component()
        offset = self.solver.slip_positions[source_name][0]
        for comp_name in adapter.get_param_names():
            n_component = int(params_per_comp[comp_name])
            if comp_name in comp_bounds:
                lower, upper = comp_bounds[comp_name]
                self._bounds['lb'][offset:offset + n_component] = lower
                self._bounds['ub'][offset:offset + n_component] = upper
            offset += n_component

    def _fault_exists(self, fault_name: str) -> bool:
        """Return whether ``fault_name`` is present in ``solver.faults``.

        Parameters
        ----------
        fault_name : str
            Source name to test.

        Returns
        -------
        bool
            ``True`` for a known source name.
        """
        return any(fault.name == fault_name for fault in self.solver.faults)

    def _get_source_type(self, fault_name):
        """Return the adapter/source type used for semantic validation.

        Parameters
        ----------
        fault_name : str
            Existing source name.

        Returns
        -------
        str
            Adapter ``source_type`` when available, otherwise the source
            object's ``type`` with ``"Fault"`` as the compatibility fallback.
        """
        if hasattr(self.solver, 'adapters') and fault_name in self.solver.adapters:
            return self.solver.adapters[fault_name].source_type
        fault_obj = next((f for f in self.solver.faults if f.name == fault_name), None)
        return getattr(fault_obj, 'type', 'Fault') if fault_obj else 'Fault'

    def get_linear_parameter_layout(self):
        """Return and validate the complete BLSE/VCE linear-vector layout.

        Returns
        -------
        dict
            Active ``blse_full_linear`` descriptor with zero global offset.

        Raises
        ------
        ValueError
            If source/component/poly ranges or assembled matrix widths are
            inconsistent.

        Notes
        -----
        BLSE/VCE inherits the base active-linear coordinate hooks because its
        bounds vector and every ``A`` matrix use this same full vector.
        """
        width = self._get_linear_matrix_n_parameters()
        layout = self._build_linear_parameter_layout(
            space='blse_full_linear',
            width=width,
            global_offset=0,
            source_positions=getattr(
                self.solver,
                'slip_positions',
                getattr(self.solver, 'fault_indexes', {}),
            ),
            poly_positions=getattr(self.solver, 'poly_positions', {}),
        )
        assembled = getattr(self.solver, 'G', None)
        if assembled is not None and np.asarray(assembled).ndim == 2:
            assembled_width = int(np.asarray(assembled).shape[1])
            if assembled_width != width:
                raise ValueError(
                    f"BLSE G has {assembled_width} columns but "
                    f"lsq_parameters/layout has {width}"
                )
        return layout

    def sync_to_solver(self, *, force=False):
        """Refresh legacy solver-side caches from the constraint manager.

        BLSE/VCE solve paths read bounds and linear constraints directly from
        ``constraint_manager``.  This method keeps older inspection attributes
        such as ``solver.lb`` and ``solver.A_ueq`` current without creating a
        second writable constraint state on the solver.

        Parameters
        ----------
        force : bool, default False
            Synchronize immediately even inside a transaction.

        Returns
        -------
        None

        Notes
        -----
        Arrays are copied where stored as legacy bounds.  Combined constraint
        matrices are regenerated from the manager's named groups.
        """
        if self.in_constraint_transaction and not force:
            self._constraint_transaction_sync_pending = True
            return

        # Sync legacy bounds arrays
        self.solver._lb = (
            self._bounds['lb'].copy()
            if self._bounds['lb'] is not None
            else None
        )
        self.solver._ub = (
            self._bounds['ub'].copy()
            if self._bounds['ub'] is not None
            else None
        )

        # Sync combined constraints for backward-compatible readers.
        A_ineq, b_ineq = self.get_combined_inequality_constraints()
        A_eq, b_eq = self.get_combined_equality_constraints()

        self.solver._A_ueq = A_ineq
        self.solver._b_ueq = b_ineq
        self.solver._Aeq = A_eq
        self.solver._beq = b_eq

        if self.verbose:
            print("[SYNC] Refreshed legacy solver constraint cache")

    def validate(self) -> Dict[str, Any]:
        """Validate the resolved BLSE bounds and constraint groups.

        Returns
        -------
        dict
            Base validation report plus a ``summary`` mapping containing bound
            presence and constraint group/row counts.
        """
        result = super().validate()
        result['summary'] = {
            'bounds_set': self._bounds['lb'] is not None or self._bounds['ub'] is not None,
            'inequality_groups': len(self._inequality_constraints),
            'equality_groups': len(self._equality_constraints),
            'total_inequality_constraints': sum(c['A'].shape[0] for c in self._inequality_constraints.values()),
            'total_equality_constraints': sum(c['A'].shape[0] for c in self._equality_constraints.values())
        }
        return result

    def print_summary(self):
        """Print bounds, constraint provenance, and validation diagnostics."""
        print("\n" + "="*70)
        print("COMPLETE CONSTRAINT MANAGER SUMMARY")
        print("="*70)
        
        # Configuration info
        print(f"[OK] CONFIGURATION")
        if self._bounds['config_file']:
            print(f"   Config file: {self._bounds['config_file']}")
        
        if self.config:
            bounds_enabled = getattr(self.config, 'use_bounds_constraints', False)
            rake_enabled = getattr(self.config, 'use_rake_angle_constraints', False)
            interseismic_config = getattr(self.config, 'interseismic_config', {})
            cap_enabled = interseismic_config.get('cap_constraints', {}).get('enabled', False)
            
            print(f"   Bounds constraints: {'[OK] Enabled' if bounds_enabled else '[X] Disabled'}")
            print(f"   Rake constraints: {'[OK] Enabled' if rake_enabled else '[X] Disabled'}")
            print(f"   Euler-cap constraints: {'[OK] Enabled' if cap_enabled else '[X] Disabled'}")
        
        # Bounds info
        print(f"\n[STAT] BOUNDS MANAGEMENT")
        if self._bounds['lb'] is not None or self._bounds['ub'] is not None:
            n_params = len(self._bounds['lb']) if self._bounds['lb'] is not None else len(self._bounds['ub'])
            n_lb = np.sum(~np.isnan(self._bounds['lb'])) if self._bounds['lb'] is not None else 0
            n_ub = np.sum(~np.isnan(self._bounds['ub'])) if self._bounds['ub'] is not None else 0
            n_both = 0
            if self._bounds['lb'] is not None and self._bounds['ub'] is not None:
                n_both = np.sum((~np.isnan(self._bounds['lb'])) & (~np.isnan(self._bounds['ub'])))
            
            print(f"   Total parameters: {n_params}")
            print(f"   Lower bounded: {n_lb}")
            print(f"   Upper bounded: {n_ub}")
            print(f"   Fully bounded: {n_both}")
            
            # Global bounds
            global_lb = self._bounds['global']['lb']
            global_ub = self._bounds['global']['ub']
            if global_lb is not None or global_ub is not None:
                print(f"   Global defaults: lb={global_lb}, ub={global_ub}")
            
            # Per-fault bounds
            if self._bounds['strikeslip']:
                print(f"   Strike-slip bounds: {len(self._bounds['strikeslip'])} fault(s)")
                for fault, bounds in self._bounds['strikeslip'].items():
                    print(f"     - {fault}: {bounds}")
            
            if self._bounds['dipslip']:
                print(f"   Dip-slip bounds: {len(self._bounds['dipslip'])} fault(s)")
                for fault, bounds in self._bounds['dipslip'].items():
                    print(f"     - {fault}: {bounds}")
            
            if self._bounds['poly']:
                print(f"   Polynomial bounds: {len(self._bounds['poly'])} fault(s)")
                for fault, bounds in self._bounds['poly'].items():
                    print(f"     - {fault}: {bounds}")
                    
            print(f"   Source: {self._bounds['source']}")
        else:
            print("   No bounds set")
        
        # Inequality constraints
        print(f"\n[INQ] INEQUALITY CONSTRAINTS")
        print(f"   Groups: {len(self._inequality_constraints)}")
        total_ineq = sum(c['A'].shape[0] for c in self._inequality_constraints.values())
        print(f"   Total constraints: {total_ineq}")
        
        for name, constraint in self._inequality_constraints.items():
            print(f"   - {name}: {constraint['A'].shape[0]} constraints (source: {constraint['source']})")
        
        # Equality constraints
        print(f"\n[EQ] EQUALITY CONSTRAINTS")
        print(f"   Groups: {len(self._equality_constraints)}")
        total_eq = sum(c['A'].shape[0] for c in self._equality_constraints.values())
        print(f"   Total constraints: {total_eq}")
        
        for name, constraint in self._equality_constraints.items():
            print(f"   - {name}: {constraint['A'].shape[0]} constraints (source: {constraint['source']})")
        
        # Validation status
        validation = self.validate()
        print(f"\n[OK] VALIDATION: {'PASSED' if validation['valid'] else 'FAILED'}")
        
        if validation['errors']:
            print("   [X] Errors:")
            for error in validation['errors']:
                print(f"      {error}")
        
        if validation['warnings']:
            print("   [!]  Warnings:")
            for warning in validation['warnings']:
                print(f"      {warning}")
        
        print("="*70)

    # BLSE-specific property
    @property
    def bounds_config_file(self) -> Optional[str]:
        """Return the loaded bounds-config path, if any.

        Returns
        -------
        str or None
            Path stored by the shared configuration loader.
        """
        return self._bounds['config_file']
