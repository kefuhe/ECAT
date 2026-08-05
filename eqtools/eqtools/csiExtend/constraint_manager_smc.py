"""
Bounds Manager Module

This module provides the BoundsManager class for managing parameter bounds
and constraints in Bayesian fault inversion processes.
"""

import numpy as np
import copy
from collections.abc import Mapping
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime

from .constraint_manager_base import ConstraintManagerBase
from .source_adapters import FaultAdapter


class ConstraintManagerSMC(ConstraintManagerBase):
    """Manage bounds and active linear constraints for Bayesian/SMC inversion.

    Resolved bound arrays always follow the vector sampled by SMC.  That vector
    equals the full physical model for ``ss_ds`` and ``magnitude_rake``; in
    ``rake_fixed`` it is compacted to one slip magnitude per Fault patch.

    Linear equality and inequality matrices are consumed only by
    ``SMC_FJ + ss_ds``.  Their columns address the linear suffix beginning at
    ``inversion_instance.linear_sample_start_position``, not the full sampled
    vector.  FULLSMC retains bounds/priors while linear-only declarations are
    reported as inactive.
    """
    
    def __init__(self, inversion_instance, verbose: bool = True):
        """Initialize an SMC constraint manager.

        Parameters
        ----------
        inversion_instance : object
            Bayesian inversion object defining configuration, source adapters,
            and sampled-parameter positions.
        verbose : bool
            Whether to print application and validation summaries.

        Raises
        ------
        ValueError
            If the selected FULLSMC slip parameterization is incompatible with
            the configured source types or Fault components.
        """
        self.inversion_instance = inversion_instance
        self.config = inversion_instance.config
        self.multifaults = inversion_instance.multifaults
        self.verbose = verbose
        
        # Basic parameters
        self.mcmc_samples = inversion_instance.mcmc_samples
        self.geometry_positions = inversion_instance.geometry_positions
        self.sigmas_position = inversion_instance.sigmas_position
        self.alpha_position = inversion_instance.alpha_position
        # The inversion owns the full physical model layout.  The manager owns
        # the vector actually sampled by SMC, which is compact only for
        # ``rake_fixed``.  Keep these names distinct at every cross-layer use.
        full_slip_positions = getattr(
            inversion_instance, 'full_slip_positions', inversion_instance.slip_positions
        )
        full_poly_positions = getattr(
            inversion_instance, 'full_poly_positions', inversion_instance.poly_positions
        )
        self.sample_slip_positions = full_slip_positions.copy()
        self.sample_poly_positions = full_poly_positions.copy()
        
        # Sampling modes
        self.slip_sampling_mode = self.config.slip_sampling_mode
        self.bayesian_sampling_mode = getattr(self.config, 'bayesian_sampling_mode', 'SMC_FJ')

        self._validate_sampling_source_compatibility()

        # Adjust positions for rake_fixed mode
        self._adjust_positions_for_rake_fixed()
        
        # Shared storage (constraints, cache, common bounds keys)
        self._init_shared_storage()
        
        # Extend bounds with SMC-specific keys
        self._bounds.update({
            'geometry': {},          # Per-fault geometry bounds
            'slip_magnitude': {},    # Per-fault slip magnitude bounds
            'rake_angle': {},        # Per-fault rake angle bounds
            'sigmas': None,          # Global sigmas bounds
            'alpha': None,           # Global alpha bounds
        })
        self._inactive_constraints = ()
        
        if self.verbose:
            print(f"[OK] BoundsManager initialized (mode: {self.bayesian_sampling_mode}/{self.slip_sampling_mode})")

    def _adjust_positions_for_rake_fixed(self):
        """Compact sampled Fault ``sd`` blocks to one magnitude per patch.

        Notes
        -----
        ``full_*_positions`` on the inversion remain unchanged.  This manager
        adjusts only ``sample_*_positions`` and shifts downstream polynomial
        blocks by the accumulated number of removed dip-slip parameters.
        """
        if self.slip_sampling_mode == 'rake_fixed':
            total_half = 0
            for ifault in self.config.faults_list:
                lb_slip, ub_slip = self.sample_slip_positions[ifault.name]
                lb_poly, ub_poly = self.sample_poly_positions[ifault.name]
                reduction = 0
                if self._get_source_type(ifault.name) == 'Fault':
                    reduction = self._get_n_spatial_elements(ifault)
                self.sample_slip_positions[ifault.name] = [
                    lb_slip - total_half,
                    ub_slip - total_half - reduction,
                ]
                total_half += reduction
                self.sample_poly_positions[ifault.name] = [
                    lb_poly - total_half,
                    ub_poly - total_half,
                ]

    def _validate_sampling_source_compatibility(self):
        """Reject unsupported FULLSMC parameterizations before indexing.

        Raises
        ------
        ValueError
            If magnitude/rake sampling includes a non-Fault source or a Fault
            whose canonical component order is not exactly strike-slip then
            dip-slip.
        """
        if self.bayesian_sampling_mode != 'FULLSMC':
            return
        if self.slip_sampling_mode not in {'magnitude_rake', 'rake_fixed'}:
            return
        non_faults = [
            source.name
            for source in self.config.faults_list
            if self._get_source_type(source.name) != 'Fault'
        ]
        if non_faults:
            raise ValueError(
                f"FULLSMC + {self.slip_sampling_mode} currently supports Fault "
                f"sources only; non-Fault sources: {non_faults}. Use ss_ds for "
                "mixed-source FULLSMC inversion."
            )

        incompatible = []
        for source in self.config.faults_list:
            adapter = getattr(self.multifaults, 'adapters', {}).get(source.name)
            if adapter is not None:
                slipdir = getattr(adapter, 'slipdir', None)
                components = tuple(adapter.get_param_names())
            else:
                slipdir = FaultAdapter._canonicalize_slipdir(
                    getattr(source, 'slipdir', None)
                )
                components = tuple(
                    FaultAdapter._CHAR_TO_NAME[char] for char in slipdir
                )
            if components != ('strikeslip', 'dipslip'):
                incompatible.append(f"{source.name} (slipdir={slipdir!r})")
        if incompatible:
            raise ValueError(
                f"FULLSMC + {self.slip_sampling_mode} requires every Fault "
                "source to use exactly strike-slip and dip-slip components "
                "(canonical slipdir='sd'); incompatible sources: "
                + ", ".join(incompatible)
            )

    @property
    def slip_positions(self):
        """Return sampled-vector slip/source positions.

        Returns
        -------
        dict
            Compatibility alias for :attr:`sample_slip_positions`.
        """
        return self.sample_slip_positions

    @property
    def poly_positions(self):
        """Return sampled-vector polynomial positions.

        Returns
        -------
        dict
            Compatibility alias for :attr:`sample_poly_positions`.
        """
        return self.sample_poly_positions

    def _snapshot_mutable_state(self):
        """Capture base state plus the SMC inactive-feature registry.

        Returns
        -------
        dict
            Deep snapshot suitable for transactional rollback.
        """
        snapshot = super()._snapshot_mutable_state()
        snapshot['inactive_constraints'] = tuple(self._inactive_constraints)
        return snapshot

    def _restore_mutable_state(self, snapshot):
        """Restore a snapshot created by :meth:`_snapshot_mutable_state`.

        Parameters
        ----------
        snapshot : dict
            Previously captured mutable manager state.

        Returns
        -------
        None
        """
        super()._restore_mutable_state(snapshot)
        self._inactive_constraints = tuple(snapshot.get('inactive_constraints', ()))

    def _get_parallel_rank(self):
        """Return the configured parallel rank used to suppress duplicate logs.

        Returns
        -------
        int or None
            Configured rank, or ``None`` outside ranked execution.
        """
        return getattr(self.config, 'parallel_rank', None)

    def _get_source_type(self, fault_name):
        """Return the adapter/source type used for semantic validation.

        Parameters
        ----------
        fault_name : str
            Existing source name.

        Returns
        -------
        str
            Adapter ``source_type`` when available, otherwise the configured
            object's ``type`` with ``"Fault"`` as the compatibility fallback.
        """
        if hasattr(self.multifaults, 'adapters') and fault_name in self.multifaults.adapters:
            return self.multifaults.adapters[fault_name].source_type
        fault_obj = next((f for f in self.config.faults_list if f.name == fault_name), None)
        return getattr(fault_obj, 'type', 'Fault') if fault_obj else 'Fault'

    def _get_linear_matrix_source_start(self, source_name):
        """Return a source start relative to the SMC_FJ linear suffix.

        Parameters
        ----------
        source_name : str
            Source whose first active linear column is requested.

        Returns
        -------
        int
            Zero-based constraint-matrix column offset for ``source_name``.

        Notes
        -----
        This overrides the base full-linear default. The owning
        ``sample_slip_positions`` remain absolute in the complete sampled
        vector; only linear matrix columns subtract the nonlinear prefix.
        """
        return int(self.sample_slip_positions[source_name][0]) - int(
            self.inversion_instance.linear_sample_start_position
        )

    def _get_linear_matrix_n_parameters(self):
        """Return the number of SMC_FJ linear constraint columns.

        Returns
        -------
        int
            Sampled-vector length minus the linear suffix start.

        Notes
        -----
        This overrides the base full-linear default and is the shared width
        for rake, source-adapter, interseismic, and raw linear matrices.
        """
        return int(self.mcmc_samples) - int(
            self.inversion_instance.linear_sample_start_position
        )

    def _get_n_spatial_elements(self, fault_obj):
        """Return the number of patches, volumes, or point-source elements.

        Returns
        -------
        int
            Adapter-provided spatial count when available, otherwise a
            compatible source-attribute fallback.
        """
        if hasattr(self.multifaults, 'adapters') and fault_obj.name in self.multifaults.adapters:
            return self.multifaults.adapters[fault_obj.name].get_n_spatial_elements()
        if hasattr(fault_obj, 'patch'):
            return len(fault_obj.patch)
        if hasattr(fault_obj, 'volumes'):
            return len(fault_obj.volumes)
        return 1  # Point source (Pressure)

    def _is_smc_fj_mode(self) -> bool:
        """Return whether the mode actively consumes linear constraints.

        Returns
        -------
        bool
            ``True`` only for ``SMC_FJ + ss_ds``.
        """
        return (self.bayesian_sampling_mode == 'SMC_FJ' and
                self.slip_sampling_mode == 'ss_ds')

    def _linear_constraints_active(self) -> bool:
        """Return whether the active SMC mode consumes linear ``ss_ds`` state.

        Returns
        -------
        bool
            ``True`` only for ``SMC_FJ + ss_ds``.

        Notes
        -----
        This overrides the base full-linear default. FULLSMC keeps full-vector
        bounds but does not register or consume linear ``A`` matrices.
        """
        return self._is_smc_fj_mode()

    def _require_linear_constraints_active(self, action):
        """Require the only SMC mode that consumes linear matrices."""
        if not self._is_smc_fj_mode():
            raise RuntimeError(
                f"{action} requires SMC_FJ with ss_ds sampling"
            )

    def _resolve_inactive_constraints(self):
        """Identify configured linear-only features ignored by this SMC mode.

        Returns
        -------
        tuple of str
            Stable feature identifiers.  The tuple is empty in active
            ``SMC_FJ + ss_ds`` mode.
        """
        if self._is_smc_fj_mode():
            return ()

        inactive = []
        bounds_config = self._bounds_config or {}
        config_patch_rake = any(
            'rake_angle' in spec
            for spec in self._iter_patch_constraint_specs(
                include_config=True, include_runtime=False
            )
        )
        runtime_patch_rake = any(
            'rake_angle' in spec
            for spec in self._iter_patch_constraint_specs(
                include_config=False, include_runtime=True
            )
        )
        has_config_rake = (
            self._config_flag_enabled('use_rake_angle_constraints')
            and (bounds_config.get('rake_angle') or config_patch_rake)
        )
        has_runtime_rake = bool(
            self._runtime_rake_limits or runtime_patch_rake
        )
        if (
            self.slip_sampling_mode == 'ss_ds'
            and (has_config_rake or has_runtime_rake)
        ):
            inactive.append('rake_linear_sector')
        patch_rake_supported = (
            self._is_smc_fj_mode()
            or (
                self.bayesian_sampling_mode == 'FULLSMC'
                and self.slip_sampling_mode == 'magnitude_rake'
            )
        )
        config_patch_enabled = (
            config_patch_rake
            and (
                self._config_flag_enabled('use_bounds_constraints')
                if (
                    self.bayesian_sampling_mode == 'FULLSMC'
                    and self.slip_sampling_mode == 'magnitude_rake'
                )
                else self._config_flag_enabled(
                    'use_rake_angle_constraints'
                )
            )
        )
        if (
            not patch_rake_supported
            and (config_patch_enabled or runtime_patch_rake)
        ):
            inactive.append('patch_rake_angle')
        if bounds_config.get('source_constraints'):
            inactive.append('source_constraints')

        interseismic = getattr(self.config, 'interseismic_config', {}) or {}
        if interseismic.get('blocks', {}).get('enabled', False):
            inactive.append('block_euler_constraints')
        if interseismic.get('cap_constraints', {}).get('enabled', False):
            inactive.append('euler_cap_constraints')
        if interseismic.get('backslip_constraints'):
            inactive.append('backslip_constraints')
        return tuple(inactive)

    def _supports_patch_component_bounds(self):
        """Return whether patch strike/dip bounds match the sampled layout.

        Returns
        -------
        bool
            ``True`` only for explicit ``ss_ds`` sampling.

        Notes
        -----
        This overrides the base full-linear default because magnitude/rake
        sampling does not expose independent strike-slip and dip-slip bounds.
        """
        return self.slip_sampling_mode == 'ss_ds'

    def _validate_runtime_patch_rake_mode(self, patch_constraints):
        """Reject explicit patch-rake updates that have no active consumer.

        Runtime calls are imperative: silently retaining a declaration that
        the selected sampler cannot consume would make the caller believe the
        update took effect. Config-file declarations remain diagnosable as
        inactive so shared/copyable configuration files stay convenient.
        """
        has_rake = any(
            'rake_angle' in spec
            for spec in self._iter_patch_constraint_specs(
                patch_constraints,
                include_config=False,
                include_runtime=False,
            )
        )
        if not has_rake:
            return
        supported = (
            self._is_smc_fj_mode()
            or (
                self.bayesian_sampling_mode == 'FULLSMC'
                and self.slip_sampling_mode == 'magnitude_rake'
            )
        )
        if not supported:
            raise RuntimeError(
                "Runtime patch rake_angle requires SMC_FJ with ss_ds "
                "sampling (linear rake sector), or FULLSMC with "
                "magnitude_rake sampling (sampled rake bounds)."
            )

    def add_patch_constraints(self, patch_constraints, *, source="manual", sync=True):
        """Add runtime patch declarations after validating rake mode support."""
        self._validate_runtime_patch_rake_mode(patch_constraints)
        return super().add_patch_constraints(
            patch_constraints,
            source=source,
            sync=sync,
        )

    def replace_patch_constraints(
        self,
        patch_constraints,
        *,
        source="manual",
        sync=True,
    ):
        """Replace runtime patch declarations after validating rake support."""
        self._validate_runtime_patch_rake_mode(patch_constraints)
        return super().replace_patch_constraints(
            patch_constraints,
            source=source,
            sync=sync,
        )

    def get_linear_parameter_layout(self):
        """Return and validate the SMC_FJ linear-suffix layout.

        FULLSMC and non-``ss_ds`` modes return an explicit inactive descriptor
        because they do not consume linear constraint matrices.

        Returns
        -------
        dict
            Active ``smc_fj_linear_suffix`` descriptor, or an inactive
            descriptor for modes that do not consume linear matrices.

        Raises
        ------
        ValueError
            If source/component/poly ranges, assembled matrices, or the
            inversion's declared linear width disagree.
        """
        global_offset = int(
            self.inversion_instance.linear_sample_start_position
        )
        if not self._is_smc_fj_mode():
            return self._build_linear_parameter_layout(
                space='inactive',
                width=0,
                global_offset=global_offset,
                source_positions={},
                poly_positions={},
                active=False,
                inactive_reason=(
                    f"{self.bayesian_sampling_mode}/"
                    f"{self.slip_sampling_mode} does not consume linear "
                    "constraint matrices"
                ),
            )

        width = self._get_linear_matrix_n_parameters()
        layout = self._build_linear_parameter_layout(
            space='smc_fj_linear_suffix',
            width=width,
            global_offset=global_offset,
            source_positions=self.sample_slip_positions,
            poly_positions=self.sample_poly_positions,
        )
        declared_width = getattr(
            self.inversion_instance, 'lsq_parameters', None
        )
        if declared_width is not None and int(declared_width) != width:
            raise ValueError(
                f"SMC_FJ linear suffix has width {width}, but "
                f"lsq_parameters reports {int(declared_width)}"
            )
        return layout

    def _fault_exists(self, fault_name: str) -> bool:
        """Return whether a source name exists in ``config.faults_list``.

        Parameters
        ----------
        fault_name : str
            Source name to test.

        Returns
        -------
        bool
            ``True`` for a configured source name.
        """
        return any(fault.name == fault_name for fault in self.config.faults_list)

    def _on_bounds_config_loaded(self):
        """Mirror the loaded declaration onto the legacy inversion API.

        Returns
        -------
        None

        Notes
        -----
        This overrides the base no-op hook.
        A deep copy prevents mutation of manager state through
        ``inversion_instance.bounds_config``.
        """
        self.inversion_instance.bounds_config = copy.deepcopy(
            self._bounds_config
        )

    # ==================== Configuration Loading ====================

    def _get_extra_config_summary_items(self) -> list:
        """Return human-readable SMC sections present in the loaded config.

        Returns
        -------
        list of str
            Geometry, slip-magnitude, sigma, and alpha summary labels.
        """
        items = []
        if not self._bounds_config:
            return items
        geometry = self._bounds_config.get('geometry') or {}
        if geometry:
            items.append(f"geometry bounds for {len(geometry)} fault(s)")
        slip_magnitude = self._bounds_config.get('slip_magnitude') or {}
        if slip_magnitude:
            items.append(
                f"slip magnitude bounds for {len(slip_magnitude)} fault(s)"
            )
        if 'sigmas' in self._bounds_config:
            items.append("sigmas bounds")
        if 'alpha' in self._bounds_config:
            items.append("alpha bounds")
        return items

    # ==================== Global Bounds Management ====================
    
    def set_global_bounds(self, lb: float = None, ub: float = None, source: str = "manual"):
        """Set default bounds for otherwise unassigned sampled parameters.

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
        Global bounds have the lowest precedence during a rebuild.
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

    def _initialize_bounds_arrays(self):
        """Allocate unresolved bounds for the complete sampled vector.

        Returns
        -------
        None

        Notes
        -----
        Unlike active linear matrices, these arrays include the nonlinear
        prefix in both SMC_FJ and FULLSMC.
        """
        if self._bounds['lb'] is None:
            self._bounds['lb'] = np.ones(self.mcmc_samples) * np.nan
        if self._bounds['ub'] is None:
            self._bounds['ub'] = np.ones(self.mcmc_samples) * np.nan

    # ==================== Hyperparameter Bounds ====================
    
    def set_hyperparameter_bounds(self, geometry=None, sigmas=None, alpha=None, source: str = "manual"):
        """Set geometry, sigma, and alpha declarations as one batch.

        Parameters
        ----------
        geometry, sigmas, alpha : object, optional
            Bounds accepted by their corresponding specialized setter.
        source : str
            Human-readable provenance used in verbose output.

        Raises
        ------
        ValueError
            If any supplied bound declaration is malformed, non-finite, or
            internally inconsistent.
        """
        with self.batch_bounds_update():
            if geometry is not None:
                self.set_geometry_bounds(geometry, source)
            if sigmas is not None:
                self.set_sigmas_bounds(sigmas, source)
            if alpha is not None:
                self.set_alpha_bounds(alpha, source)

    def set_geometry_bounds(self, geometry_bounds, source: str = "manual"):
        """Set per-source geometry bounds in sampled-vector coordinates.

        Parameters
        ----------
        geometry_bounds : mapping or None
            Source names mapped to uniform or per-parameter bound pairs.
            Unknown sources and geometry blocks not marked ``update`` are
            skipped.
        source : str
            Human-readable provenance used in verbose output.

        Raises
        ------
        ValueError
            If a selected source declaration cannot be converted to the exact
            geometry-block length.
        """
        if geometry_bounds is None:
            return
        
        self._initialize_bounds_arrays()
        
        for fault_name, bounds in geometry_bounds.items():
            if not self._fault_exists(fault_name):
                if self.verbose:
                    print(f"[!]  Warning: Fault '{fault_name}' not found in faults_list, skipping geometry bounds")
                continue
                
            if (fault_name in self.config.faults and 
                self.config.faults[fault_name]['geometry']['update']):
                start, end = self.config.faults[fault_name]['geometry']['sample_positions']
                expected_params = end - start
                
                lb_vals, ub_vals = self._process_parameter_bounds(
                    bounds, expected_params, f"geometry for {fault_name}"
                )
                
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['geometry'][fault_name] = (lb_vals, ub_vals)
                    
                    if self.verbose:
                        print(f"[*]  Set geometry bounds for '{fault_name}': {len(lb_vals)} parameters (source: {source})")
        self._request_bounds_rebuild()

    def set_sigmas_bounds(self, sigmas_bounds, source: str = "manual"):
        """Set bounds for sampled sigma hyperparameters.

        Parameters
        ----------
        sigmas_bounds : object
            Uniform or per-sigma lower/upper declaration.  The method is a
            no-op when no sigma is sampled.
        source : str
            Human-readable provenance used in verbose output.

        Raises
        ------
        ValueError
            If the declaration is malformed, non-finite, inconsistent, or has
            the wrong length.
        """
        if not any(self.config.sigmas['update']) or sigmas_bounds is None:
            return
        
        self._initialize_bounds_arrays()
        start, end = self.sigmas_position
        expected_sigmas = end - start
        
        lb_vals, ub_vals = self._process_parameter_bounds(
            sigmas_bounds, expected_sigmas, "sigmas"
        )
        
        if lb_vals is not None and ub_vals is not None:
            self._bounds['sigmas'] = (lb_vals, ub_vals)
            
            if self.verbose:
                print(f"[STAT] Set sigmas bounds: {len(lb_vals)} sigmas (source: {source})")
        self._request_bounds_rebuild()

    def set_alpha_bounds(self, alpha_bounds, source: str = "manual"):
        """Set bounds for sampled alpha hyperparameters.

        Parameters
        ----------
        alpha_bounds : object
            Uniform or per-alpha lower/upper declaration.  The method is a
            no-op when no alpha parameter is sampled.
        source : str
            Human-readable provenance used in verbose output.

        Raises
        ------
        ValueError
            If the declaration is malformed, non-finite, inconsistent, or has
            the wrong length.
        """
        if not any(self.config.alpha['update']) or alpha_bounds is None:
            return
        
        self._initialize_bounds_arrays()
        start, end = self.alpha_position
        expected_alphas = end - start
        
        lb_vals, ub_vals = self._process_parameter_bounds(
            alpha_bounds, expected_alphas, "alpha"
        )
        
        if lb_vals is not None and ub_vals is not None:
            self._bounds['alpha'] = (lb_vals, ub_vals)
            
            if self.verbose:
                print(f"[*] Set alpha bounds: {len(lb_vals)} alphas (source: {source})")
        self._request_bounds_rebuild()

    # ==================== Linear Parameter Bounds ====================
    
    def set_linear_parameter_bounds(self, slip_magnitude=None, rake_angle=None, 
                                   strikeslip=None, dipslip=None, poly=None, source: str = "manual"):
        """Set slip/source and polynomial declarations as one batch.

        Parameters
        ----------
        slip_magnitude, rake_angle, strikeslip, dipslip : mapping, optional
            Per-source declarations interpreted according to
            ``slip_sampling_mode``.
        poly : mapping, optional
            Per-source polynomial bound declarations.
        source : str
            Human-readable provenance used in verbose output.

        Raises
        ------
        ValueError
            If a selected declaration is malformed or incompatible with the
            sampled source layout.
        """
        with self.batch_bounds_update():
            self.set_slip_bounds_for_all_faults(slip_magnitude, rake_angle, strikeslip, dipslip, source)
            self.set_poly_bounds_for_all_faults(poly, source)

    def set_slip_bounds_for_all_faults(self, slip_magnitude=None, rake_angle=None, 
                                      strikeslip=None, dipslip=None, source: str = "manual"):
        """Dispatch Fault slip bounds according to the sampling mode.

        Parameters
        ----------
        slip_magnitude, rake_angle, strikeslip, dipslip : mapping, optional
            Fault names mapped to bound declarations for the named physical
            quantity.
        source : str
            Human-readable provenance used in verbose output.

        Notes
        -----
        Non-Fault sources are intentionally skipped; use
        :meth:`set_source_component_bounds` for their adapter components.
        """
        self._initialize_bounds_arrays()
        
        for fault_name in self.config.faultnames:
            if not self._fault_exists(fault_name):
                if self.verbose:
                    print(f"[!]  Warning: Fault '{fault_name}' not found in faults_list, skipping slip bounds")
                continue
            
            # Non-Fault sources do not use Fault strike/dip/rake semantics.
            if self._get_source_type(fault_name) != 'Fault':
                continue

            if self.slip_sampling_mode == "rake_fixed":
                selected = (
                    slip_magnitude.get(fault_name)
                    if slip_magnitude
                    else None
                )
                if selected is not None:
                    self._store_fault_slip_bounds(
                        fault_name,
                        slip_magnitude=selected,
                        source=source,
                    )
            elif self.slip_sampling_mode == "ss_ds":
                ss = strikeslip.get(fault_name) if strikeslip else None
                ds = dipslip.get(fault_name) if dipslip else None
                if ss is not None or ds is not None:
                    self._store_fault_slip_bounds(
                        fault_name,
                        strikeslip=ss,
                        dipslip=ds,
                        source=source,
                    )
            elif self.slip_sampling_mode == "magnitude_rake":
                magnitude = (
                    slip_magnitude.get(fault_name)
                    if slip_magnitude
                    else None
                )
                rake = (
                    rake_angle.get(fault_name)
                    if rake_angle
                    else None
                )
                if magnitude is not None or rake is not None:
                    self._store_fault_slip_bounds(
                        fault_name,
                        slip_magnitude=magnitude,
                        rake_angle=rake,
                        source=source,
                    )

    def _store_fault_slip_bounds(
        self,
        fault_name,
        slip_magnitude=None,
        rake_angle=None,
        strikeslip=None,
        dipslip=None,
        source: str = "manual",
    ):
        """Validate and store one Fault's mode-specific bound declarations.

        Parameters
        ----------
        fault_name : str
            Configured Fault source name.
        slip_magnitude, rake_angle, strikeslip, dipslip : object, optional
            Uniform or per-patch lower/upper declarations.  Only quantities
            represented by the active ``slip_sampling_mode`` are used.
        source : str
            Human-readable provenance used in verbose output.

        Raises
        ------
        ValueError
            If a declaration has the wrong shape or bounds order, or if
            ``ss_ds`` requests a component absent from the source layout.

        Notes
        -----
        ``rake_fixed`` stores one magnitude per patch;
        ``magnitude_rake`` stores magnitudes followed by rake angles; ``ss_ds``
        uses adapter-resolved component slices. This compiler does not write
        final ``lb``/``ub`` arrays; :meth:`_rebuild_resolved_bounds` is the
        single resolver/writer.
        """
        slip_start, slip_end = self.sample_slip_positions[fault_name]
        n_patches = slip_end - slip_start
        
        # Get fault object for patch count validation
        fault_obj = next((f for f in self.config.faults_list if f.name == fault_name), None)
        if fault_obj is None:
            if self.verbose:
                print(f"[!]  Warning: Fault '{fault_name}' not found in faults_list")
            return
        
        # Use adapter if available for spatial element count, fallback to len(patch)
        if hasattr(self.config, 'multifaults') and hasattr(self.config.multifaults, 'adapters') \
                and fault_name in self.config.multifaults.adapters:
            expected_patches = self.config.multifaults.adapters[fault_name].get_n_spatial_elements()
        elif hasattr(fault_obj, 'patch'):
            expected_patches = len(fault_obj.patch)
        elif hasattr(fault_obj, 'volumes'):
            expected_patches = len(fault_obj.volumes)
        else:
            expected_patches = n_patches

        if self.slip_sampling_mode == 'rake_fixed':
            # Only magnitude bounds (n_patches values)
            if slip_magnitude is not None:
                lb_vals, ub_vals = self._process_parameter_bounds(
                    slip_magnitude, expected_patches, f"slip_magnitude for {fault_name}"
                )
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['slip_magnitude'][fault_name] = (lb_vals, ub_vals)
                    
                    if self.verbose:
                        print(f"[*] Set slip magnitude bounds for '{fault_name}': {len(lb_vals)} patches (source: {source})")
                
        elif self.slip_sampling_mode == 'magnitude_rake':
            # Magnitude and rake bounds (each n_patches values)
            if slip_magnitude is not None:
                lb_vals, ub_vals = self._process_parameter_bounds(
                    slip_magnitude, expected_patches, f"slip_magnitude for {fault_name}"
                )
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['slip_magnitude'][fault_name] = (lb_vals, ub_vals)
                    
                    if self.verbose:
                        print(f"[*] Set slip magnitude bounds for '{fault_name}': {len(lb_vals)} patches (source: {source})")
                
            if rake_angle is not None:
                lb_vals, ub_vals = self._process_parameter_bounds(
                    rake_angle, expected_patches, f"rake_angle for {fault_name}"
                )
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['rake_angle'][fault_name] = (lb_vals, ub_vals)
                    
                    if self.verbose:
                        print(f"[SYNC] Set rake angle bounds for '{fault_name}': {len(lb_vals)} patches (source: {source})")
                
        elif self.slip_sampling_mode == 'ss_ds':
            # Strike-slip and dip-slip bounds (each n_patches values)
            adapter = None
            if hasattr(self.multifaults, 'adapters'):
                adapter = self.multifaults.adapters.get(fault_name)
            component_slices = self._source_component_slices(
                fault_obj, slip_start, adapter=adapter
            )

            if strikeslip is not None:
                if 'strikeslip' not in component_slices:
                    raise ValueError(
                        f"Fault '{fault_name}' has no strikeslip component for bounds"
                    )
                ss_slice = component_slices['strikeslip']
                n_ss = ss_slice.stop - ss_slice.start
                lb_vals, ub_vals = self._process_parameter_bounds(
                    strikeslip, n_ss, f"strikeslip for {fault_name}"
                )
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['strikeslip'][fault_name] = (lb_vals, ub_vals)
                    
                    if self.verbose:
                        print(f"[<>]  Set strike-slip bounds for '{fault_name}': {len(lb_vals)} patches (source: {source})")
                
            if dipslip is not None:
                if 'dipslip' not in component_slices:
                    raise ValueError(
                        f"Fault '{fault_name}' has no dipslip component for bounds"
                    )
                ds_slice = component_slices['dipslip']
                n_ds = ds_slice.stop - ds_slice.start
                lb_vals, ub_vals = self._process_parameter_bounds(
                    dipslip, n_ds, f"dipslip for {fault_name}"
                )
                if lb_vals is not None and ub_vals is not None:
                    self._bounds['dipslip'][fault_name] = (lb_vals, ub_vals)
                    
                    if self.verbose:
                        print(f"[UD]  Set dip-slip bounds for '{fault_name}': {len(lb_vals)} patches (source: {source})")

        self._request_bounds_rebuild()


    def set_poly_bounds_for_all_faults(self, poly_bounds, source: str = "manual"):
        """Set polynomial declarations for all known source names.

        Unknown source names are skipped with an optional warning.
        """
        if poly_bounds is None:
            return
        
        self._initialize_bounds_arrays()
        
        for fault_name, bounds in poly_bounds.items():
            if not self._fault_exists(fault_name):
                if self.verbose:
                    print(f"[!]  Warning: Fault '{fault_name}' not found in faults_list, skipping poly bounds")
                continue
            
            self.set_poly_bounds(fault_name, bounds, source)

    def set_poly_bounds(self, fault_name: str, poly_bounds, source: str = "manual"):
        """Set uniform or per-coefficient bounds for one polynomial block.

        Raises
        ------
        ValueError
            If the declaration is malformed, non-finite, inconsistent, or has
            a length different from the sampled polynomial block.
        """
        start, end = self.sample_poly_positions[fault_name]
        expected_coeffs = end - start
        
        lb_vals, ub_vals = self._process_parameter_bounds(
            poly_bounds, expected_coeffs, f"poly for {fault_name}"
        )
        
        if lb_vals is not None and ub_vals is not None:
            self._bounds['poly'][fault_name] = (lb_vals, ub_vals)
            
            if self.verbose:
                print(f"[GEO] Set poly bounds for '{fault_name}': {len(lb_vals)} coefficients (source: {source})")
        self._request_bounds_rebuild()

    def set_source_component_bounds(self, source_name: str, comp_bounds: Dict[str, Any],
                                     source: str = "manual"):
        """
        Set per-component bounds for any source type using source adapters.
        
        Parameters
        ----------
        source_name : str
            Name of the source (fault/pressure/sbarbot).
        comp_bounds : dict
            {component_name: bounds_input} 鈥?component names from adapter.get_param_names().
            Each bounds_input is processed by _process_parameter_bounds (supports
            [lb, ub] uniform or per-element formats).
        source : str
            Source description for audit trail.

        Raises
        ------
        ValueError
            If a selected component declaration is malformed, non-finite,
            inconsistent, or has the wrong adapter-defined length.

        Notes
        -----
        Unknown sources or missing adapters are skipped with an optional
        warning.  Component order and widths come exclusively from the source
        adapter.
        """
        if not self._fault_exists(source_name):
            if self.verbose:
                print(f"[!]  Warning: Source '{source_name}' not found in faults_list, skipping source_bounds")
            return
        
        if not hasattr(self.multifaults, 'adapters') or source_name not in self.multifaults.adapters:
            if self.verbose:
                print(f"[!]  Warning: No adapter found for source '{source_name}', skipping source_bounds")
            return
        
        self._initialize_bounds_arrays()
        
        adapter = self.multifaults.adapters[source_name]
        params_per_comp = adapter.get_n_params_per_component()
        slip_st, _ = self.sample_slip_positions[source_name]
        
        resolved = {}
        offset = slip_st
        for comp_name in adapter.get_param_names():
            n = params_per_comp[comp_name]
            if comp_name in comp_bounds:
                lb_vals, ub_vals = self._process_parameter_bounds(
                    comp_bounds[comp_name], n, f"{comp_name} for {source_name}"
                )
                if lb_vals is not None and ub_vals is not None:
                    resolved[comp_name] = (lb_vals, ub_vals)
            offset += n

        self._bounds['source_bounds'][source_name] = resolved
        self._request_bounds_rebuild()
        
        if self.verbose:
            print(f"[SRC] Set component bounds for '{source_name}': {comp_bounds} (source: {source})")

    def _process_parameter_bounds(self, bounds_input, expected_length: int, param_name: str):
        """Normalize uniform or per-element lower/upper bounds.

        Parameters
        ----------
        bounds_input : various types
            A uniform ``[lb, ub]`` pair, a ``[lower_array, upper_array]``
            pair, or a mapping with explicit ``lb`` and ``ub`` entries.
        expected_length : int
            Required length of each returned array.
        param_name : str
            Quantity label used in validation messages.

        Returns
        --------
        tuple of numpy.ndarray or tuple of None
            ``(lower, upper)`` arrays, or ``(None, None)`` when input is
            ``None``.

        Raises
        ------
        ValueError
            If the representation is ambiguous, non-finite, inconsistent, or
            cannot be expanded to ``expected_length``.
        """
        if bounds_input is None:
            return None, None
        
        try:
            # Case 1: Dictionary format
            if isinstance(bounds_input, dict):
                if 'lb' in bounds_input and 'ub' in bounds_input:
                    lb_input = bounds_input['lb']
                    ub_input = bounds_input['ub']
                    
                    # Convert to arrays
                    lb_array = self._convert_to_array(lb_input, expected_length, f"{param_name} lower bounds")
                    ub_array = self._convert_to_array(ub_input, expected_length, f"{param_name} upper bounds")
                    
                    if lb_array is not None and ub_array is not None:
                        # Validate bounds consistency
                        if np.any(lb_array > ub_array):
                            raise ValueError(f"Lower bounds > upper bounds for {param_name}")
                        return lb_array, ub_array
                else:
                    raise ValueError(f"Dictionary bounds for {param_name} must contain 'lb' and 'ub' keys")
            
            # Case 2: List/array format
            elif isinstance(bounds_input, (list, tuple, np.ndarray)):
                bounds_array = np.asarray(bounds_input, dtype=float)
                
                # Case 2a: Simple [lb, ub] format (uniform bounds)
                if bounds_array.ndim == 1 and len(bounds_array) == 2:
                    self._require_finite(bounds_array, f"{param_name} bounds")
                    lb_val, ub_val = bounds_array
                    if lb_val > ub_val:
                        raise ValueError(f"Lower bound > upper bound for {param_name}: {lb_val} > {ub_val}")
                    
                    lb_array = np.full(expected_length, lb_val)
                    ub_array = np.full(expected_length, ub_val)
                    return lb_array, ub_array
                
                # Case 2b: [[lb1, lb2, ...], [ub1, ub2, ...]] format (per-element bounds)
                elif bounds_array.ndim == 2 and bounds_array.shape[0] == 2:
                    lb_input, ub_input = bounds_array[0], bounds_array[1]
                    
                    lb_array = self._convert_to_array(lb_input, expected_length, f"{param_name} lower bounds")
                    ub_array = self._convert_to_array(ub_input, expected_length, f"{param_name} upper bounds")
                    
                    if lb_array is not None and ub_array is not None:
                        if np.any(lb_array > ub_array):
                            raise ValueError(f"Lower bounds > upper bounds for {param_name}")
                        return lb_array, ub_array
                
                # Case 2c: Single array interpreted as uniform bounds
                elif bounds_array.ndim == 1 and len(bounds_array) == expected_length:
                    raise ValueError(
                        f"Ambiguous one-sided bounds for {param_name}. "
                        "Provide both 'lb' and 'ub' explicitly."
                    )
                
                else:
                    raise ValueError(f"Invalid bounds array shape for {param_name}: {bounds_array.shape}")
            
            else:
                raise ValueError(f"Unsupported bounds format for {param_name}: {type(bounds_input)}")
        
        except Exception as e:
            if self.verbose:
                print(f"[X] Error processing bounds for {param_name}: {e}")
            raise

    def _convert_to_array(self, input_val, expected_length: int, param_desc: str):
        """Convert one bound side to a finite one-dimensional array.

        Parameters
        ----------
        input_val : scalar, list, or array
            Scalar to broadcast, length-one value to broadcast, or exact-length
            vector.
        expected_length : int
            Required output length.
        param_desc : str
            Quantity label used in validation messages.

        Returns
        --------
        numpy.ndarray
            Finite vector of length ``expected_length``.

        Raises
        ------
        ValueError
            If the input is non-finite, not one-dimensional, or has an
            unsupported length.
        """
        try:
            # Scalar case - broadcast to expected length
            if np.isscalar(input_val):
                self._require_finite(input_val, param_desc)
                return np.full(expected_length, float(input_val))
            
            # Array case - validate length
            array_val = np.asarray(input_val, dtype=float)
            self._require_finite(array_val, param_desc)
            
            if array_val.ndim != 1:
                raise ValueError(f"Expected 1D array for {param_desc}")
            
            if len(array_val) == 1:
                # Single element - broadcast
                return np.full(expected_length, array_val[0])
            elif len(array_val) == expected_length:
                # Correct length
                return array_val
            else:
                raise ValueError(f"Length mismatch for {param_desc}: expected {expected_length}, got {len(array_val)}")
        
        except Exception as e:
            if self.verbose:
                print(f"[X] Error converting {param_desc}: {e}")
            raise

    # ==================== Apply Bounds from Config ====================
    
    def _rebuild_config_bounds(self):
        """Replace file-owned bound declarations and rebuild sampled bounds.

        Geometry, sigma, alpha, slip, rake, polynomial, and source-component
        declarations from the previous file are cleared before the current
        config is applied.  Persistent runtime index and patch declarations
        remain available and retain their higher precedence.

        Raises
        ------
        TypeError
            If a bound declaration has an unsupported structure.
        ValueError
            If a bound, source, component, selector, or resolved state is
            invalid.

        Notes
        -----
        The operation is transactional.  If bounds are disabled by config,
        only runtime declarations are rebuilt.
        """
        if self._bounds_config is None:
            if self.verbose:
                print("[!]  No bounds config loaded")
            return
        
        snapshot = self._snapshot_mutable_state()
        try:
            self._bounds['global'] = {'lb': None, 'ub': None}
            for key in (
                'geometry', 'slip_magnitude', 'rake_angle', 'strikeslip',
                'dipslip', 'poly', 'source_bounds', 'patch_constraints',
            ):
                self._bounds[key] = {}
            self._bounds['sigmas'] = None
            self._bounds['alpha'] = None

            if not self._config_flag_enabled('use_bounds_constraints'):
                self._rebuild_resolved_bounds(source='runtime_bounds_only')
                self._validate_or_raise()
                return

            with self.batch_bounds_update():
                lb = self._bounds_config.get('lb', None)
                ub = self._bounds_config.get('ub', None)
                if lb is not None or ub is not None:
                    self.set_global_bounds(lb, ub, source="config_file")

                self.set_hyperparameter_bounds(
                    geometry=self._bounds_config.get('geometry', None),
                    sigmas=self._bounds_config.get('sigmas', None),
                    alpha=self._bounds_config.get('alpha', None),
                    source="config_file",
                )
                self.set_linear_parameter_bounds(
                    slip_magnitude=self._bounds_config.get('slip_magnitude', None),
                    rake_angle=self._bounds_config.get('rake_angle', None),
                    strikeslip=self._bounds_config.get('strikeslip', None),
                    dipslip=self._bounds_config.get('dipslip', None),
                    poly=self._bounds_config.get('poly', None),
                    source="config_file",
                )
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
        """Resolve all declarations into fresh sampled-vector bounds.

        Parameters
        ----------
        source : str
            Provenance label recorded on the resolved state.

        Notes
        -----
        Precedence is global, hyperparameter, mode-specific Fault slip,
        polynomial, source component, explicit parameter index, then patch.
        Later layers overwrite earlier values only at their selected indices.
        """
        self._bounds['lb'] = np.full(self.mcmc_samples, np.nan, dtype=float)
        self._bounds['ub'] = np.full(self.mcmc_samples, np.nan, dtype=float)

        global_bounds = self._bounds['global']
        if global_bounds.get('lb') is not None:
            self._bounds['lb'][:] = float(global_bounds['lb'])
        if global_bounds.get('ub') is not None:
            self._bounds['ub'][:] = float(global_bounds['ub'])

        for fault_name, (lower, upper) in self._bounds['geometry'].items():
            start, end = self.config.faults[fault_name]['geometry']['sample_positions']
            self._bounds['lb'][start:end] = lower
            self._bounds['ub'][start:end] = upper
        if self._bounds['sigmas'] is not None:
            start, end = self.sigmas_position
            self._bounds['lb'][start:end], self._bounds['ub'][start:end] = self._bounds['sigmas']
        if self._bounds['alpha'] is not None:
            start, end = self.alpha_position
            self._bounds['lb'][start:end], self._bounds['ub'][start:end] = self._bounds['alpha']

        for fault in self.config.faults_list:
            fault_name = fault.name
            if self._get_source_type(fault_name) != 'Fault':
                continue
            start, end = self.sample_slip_positions[fault_name]
            if self.slip_sampling_mode == 'rake_fixed':
                values = self._bounds['slip_magnitude'].get(fault_name)
                if values is not None:
                    self._bounds['lb'][start:end], self._bounds['ub'][start:end] = values
            elif self.slip_sampling_mode == 'magnitude_rake':
                midpoint = (start + end) // 2
                magnitude = self._bounds['slip_magnitude'].get(fault_name)
                rake = self._bounds['rake_angle'].get(fault_name)
                if magnitude is not None:
                    self._bounds['lb'][start:midpoint], self._bounds['ub'][start:midpoint] = magnitude
                if rake is not None:
                    self._bounds['lb'][midpoint:end], self._bounds['ub'][midpoint:end] = rake
            elif self.slip_sampling_mode == 'ss_ds':
                adapter = getattr(self.multifaults, 'adapters', {}).get(fault_name)
                slices = self._source_component_slices(fault, start, adapter=adapter)
                strikeslip = self._bounds['strikeslip'].get(fault_name)
                dipslip = self._bounds['dipslip'].get(fault_name)
                if strikeslip is not None and 'strikeslip' in slices:
                    self._bounds['lb'][slices['strikeslip']], self._bounds['ub'][slices['strikeslip']] = strikeslip
                if dipslip is not None and 'dipslip' in slices:
                    self._bounds['lb'][slices['dipslip']], self._bounds['ub'][slices['dipslip']] = dipslip

        for fault_name, values in self._bounds['poly'].items():
            start, end = self.sample_poly_positions[fault_name]
            self._bounds['lb'][start:end], self._bounds['ub'][start:end] = values

        for source_name, components in self._bounds['source_bounds'].items():
            adapter = self.multifaults.adapters[source_name]
            offset = self.sample_slip_positions[source_name][0]
            for comp_name in adapter.get_param_names():
                n_component = int(adapter.get_n_params_per_component()[comp_name])
                if comp_name in components:
                    self._bounds['lb'][offset:offset + n_component], self._bounds['ub'][offset:offset + n_component] = components[comp_name]
                offset += n_component

        for index, (lower, upper, _) in self._bounds['parameter_bounds'].items():
            self._bounds['lb'][index] = lower
            self._bounds['ub'][index] = upper

        self._bounds['patch_constraints'] = {}
        self.apply_patch_bounds(source=source)
        self._apply_sampled_patch_rake_bounds(source=source)
        self._bounds['source'] = source
        self._bounds['applied_time'] = datetime.now()
        self._mark_bounds_changed()

    # ==================== Bounds Retrieval Methods ====================

    def _apply_sampled_patch_rake_bounds(self, *, source):
        """Apply patch ``rake_angle`` as FULLSMC sampled-rake bounds.

        The same public field has two mode-specific numerical realizations:
        SMC_FJ ``ss_ds`` compiles it into a convex linear sector, whereas
        FULLSMC ``magnitude_rake`` writes ordinary lower/upper bounds on the
        sampled rake block. No modulo-angle interpretation is used here;
        sampled bounds therefore require finite ``lower <= upper``.
        """
        if not (
            self.bayesian_sampling_mode == 'FULLSMC'
            and self.slip_sampling_mode == 'magnitude_rake'
        ):
            return []

        specs = self._iter_patch_constraint_specs(
            include_config=self._config_flag_enabled('use_bounds_constraints'),
            include_runtime=True,
        )
        seen = {}
        applied = []
        for spec_index, spec in enumerate(specs):
            if 'rake_angle' not in spec:
                continue
            fault_name = spec['fault']
            if not self._fault_exists(fault_name):
                raise ValueError(
                    f"patch constraint '{spec['name']}' references unknown "
                    f"fault '{fault_name}'"
                )
            source_type = self._get_source_type(fault_name)
            if source_type != 'Fault':
                raise ValueError(
                    f"patch constraint '{spec['name']}' only applies to Fault "
                    f"sources; '{fault_name}' is {source_type}"
                )

            fault = self._get_fault_by_name(fault_name)
            selected = self._select_patch_constraint_indices(
                fault,
                spec,
                spec_index,
            )
            start, end = self.sample_slip_positions[fault_name]
            midpoint = start + (end - start) // 2
            n_rake = end - midpoint
            n_spatial = self._get_n_spatial_elements(fault)
            if n_rake != n_spatial:
                raise ValueError(
                    f"Fault '{fault_name}' sampled rake block has {n_rake} "
                    f"parameters but the source has {n_spatial} patches"
                )

            overlap = [
                int(patch)
                for patch in selected.tolist()
                if int(patch) in seen.setdefault(fault_name, {})
            ]
            overwrite = bool(spec.get('overwrite', False))
            if overlap and not overwrite:
                previous = sorted({
                    seen[fault_name][int(patch)]
                    for patch in overlap
                })
                raise ValueError(
                    f"patch constraint '{spec['name']}' overlaps previous "
                    f"{fault_name}.rake_angle patch constraint(s) {previous} "
                    f"at patches {overlap}. Set overwrite: true to replace."
                )

            lower, upper = self._parse_patch_bound_values(
                spec['rake_angle'],
                selected.size,
                f"patch constraint '{spec['name']}' rake_angle",
            )
            columns = midpoint + selected
            self.set_parameter_bounds_by_indices(
                columns,
                lower,
                upper,
                source=source,
                persist=False,
            )
            for patch in selected.tolist():
                seen[fault_name][int(patch)] = spec['name']
            applied.append({
                'name': spec['name'],
                'fault': fault_name,
                'component': 'rake_angle',
                'patches': selected.copy(),
                'source': source,
            })

        for item in applied:
            self._bounds['patch_constraints'][
                f"{item['name']}.rake_angle"
            ] = {
                'fault': item['fault'],
                'component': 'rake_angle',
                'n_patches': int(len(item['patches'])),
                'source': item['source'],
            }
        return applied

    def _iter_fault_magnitude_slices(self):
        """Yield sampled slices that represent non-negative magnitudes.

        Yields
        ------
        tuple
            ``(fault_name, slice)`` for each Fault magnitude block in
            ``magnitude_rake`` or ``rake_fixed`` mode.
        """
        if self.slip_sampling_mode not in {'magnitude_rake', 'rake_fixed'}:
            return
        for fault in self.config.faults_list:
            if self._get_source_type(fault.name) != 'Fault':
                continue
            start, end = self.sample_slip_positions[fault.name]
            if self.slip_sampling_mode == 'magnitude_rake':
                end = start + (end - start) // 2
            yield fault.name, slice(start, end)
    
    def _resolve_fullsmc_effective_bounds(self):
        """Resolve finite FULLSMC bounds and describe applied defaults.

        Raw manager arrays keep ``NaN`` for unspecified endpoints. Effective
        arrays are produced only at the sampler boundary, so diagnostic state
        continues to distinguish a user declaration from a convenience
        default.
        """
        self._initialize_bounds_arrays()

        lb_full = self._bounds['lb'].copy()
        ub_full = self._bounds['ub'].copy()
        undefined_lower = np.isnan(lb_full)
        undefined_upper = np.isnan(ub_full)
        magnitude_default_mask = np.zeros(self.mcmc_samples, dtype=bool)
        for _, magnitude_slice in self._iter_fault_magnitude_slices() or ():
            magnitude_default_mask[magnitude_slice] = undefined_lower[
                magnitude_slice
            ]

        lb_full[undefined_lower] = -10.0
        ub_full[undefined_upper] = 10.0
        lb_full[magnitude_default_mask] = 0.0

        metadata = {
            'ordinary_default': {'lower': -10.0, 'upper': 10.0},
            'magnitude_lower_default': 0.0,
            'n_lower_defaulted': int(np.count_nonzero(undefined_lower)),
            'n_upper_defaulted': int(np.count_nonzero(undefined_upper)),
            'n_magnitude_lower_defaulted': int(
                np.count_nonzero(magnitude_default_mask)
            ),
            'raw_state_preserves_nan': True,
        }
        return lb_full, ub_full, metadata

    def get_bounds_for_fullsmc(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return finite bounds for the complete FULLSMC sampled vector.

        Returns
        -------
        tuple of numpy.ndarray
            Copies of lower and upper arrays.  Undefined entries default to
            ``[-10, 10]`` except undefined magnitude lower bounds, which
            default to zero.
        """
        lb_full, ub_full, _ = self._resolve_fullsmc_effective_bounds()
        return lb_full, ub_full

    def get_bounds_for_hyperparameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return finite bounds for the SMC_FJ hyperparameter prefix.

        Returns
        -------
        tuple of numpy.ndarray
            Lower and upper copies before ``linear_sample_start_position``;
            undefined entries default to ``[-10, 10]``.
        """
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        
        self._initialize_bounds_arrays()
        lb_hyper = self._bounds['lb'][:linear_sample_start].copy()
        ub_hyper = self._bounds['ub'][:linear_sample_start].copy()
        
        # Handle NaN values
        lb_hyper[np.isnan(lb_hyper)] = -10.0
        ub_hyper[np.isnan(ub_hyper)] = 10.0
        
        return lb_hyper, ub_hyper

    def get_bounds_for_linear_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return finite bounds for the SMC_FJ linear suffix.

        Returns
        -------
        tuple of numpy.ndarray
            Lower and upper copies beginning at
            ``linear_sample_start_position``; undefined entries default to
            ``[-10, 10]``.
        """
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        
        self._initialize_bounds_arrays()
        lb_linear = self._bounds['lb'][linear_sample_start:].copy()
        ub_linear = self._bounds['ub'][linear_sample_start:].copy()
        
        # Handle NaN values
        lb_linear[np.isnan(lb_linear)] = -10.0
        ub_linear[np.isnan(ub_linear)] = 10.0
        
        return lb_linear, ub_linear

    def has_active_linear_bounds(self) -> bool:
        """Return whether the linear suffix contains any explicit bound.

        Returns
        -------
        bool
            ``True`` if either bound array has a finite entry in the suffix.
        """
        self._initialize_bounds_arrays()
        linear_start = self.inversion_instance.linear_sample_start_position
        lower = self._bounds['lb'][linear_start:]
        upper = self._bounds['ub'][linear_start:]
        return bool(
            np.any(np.isfinite(lower)) or np.any(np.isfinite(upper))
        )

    # ==================== SMC_FJ Linear Constraints Generation ====================
    
    def _build_rake_sector_matrix(self, rake_angle=None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate rake-sector inequalities in the SMC_FJ linear space.

        Parameters
        ----------
        rake_angle : mapping, optional
            Fault-level rake sectors in degrees.  When omitted, the loaded
            bounds config is consulted.

        Returns
        -------
        tuple of numpy.ndarray
            ``(A, b)`` for ``A @ x <= b``.  Unsupported modes or absent
            declarations return empty arrays.

        Raises
        ------
        TypeError
            If a rake declaration has an unsupported type.
        ValueError
            If a source, selector, interval, or component layout is invalid.
        """
        if not self._is_smc_fj_mode():
            return np.zeros((0, 0)), np.zeros(0)
        
        # Get rake angle constraints
        if rake_angle is None:
            if (not hasattr(self, '_bounds_config') or self._bounds_config is None or 
                'rake_angle' not in self._bounds_config):
                return np.zeros((0, 0)), np.zeros(0)
            rake_angle = self._bounds_config['rake_angle']
        
        return self._generate_rake_inequality_constraints(rake_angle)

    def _build_fixed_rake_matrix(self, fixed_rake) -> Tuple[np.ndarray, np.ndarray]:
        """Generate fixed-rake equalities in the SMC_FJ linear space.

        Parameters
        ----------
        fixed_rake : mapping
            Fault names mapped to rake angles in degrees.

        Returns
        -------
        tuple of numpy.ndarray
            ``(A, b)`` for ``A @ x = b``.  Unsupported modes return empty
            arrays.
        """
        if not self._is_smc_fj_mode():
            return np.zeros((0, 0)), np.zeros(0)
        
        return self._generate_rake_equality_constraints(fixed_rake)

    def _generate_rake_inequality_constraints(self, rake_angle):
        """Resolve Fault rake sectors into ``A @ x <= b`` rows.

        Returns
        -------
        tuple of numpy.ndarray
            Inequality matrix and right-hand side in linear-suffix coordinates.

        Notes
        -----
        Non-Fault sources in ``rake_angle`` are skipped.
        """
        intervals_by_fault = self._resolve_rake_intervals_by_patch(rake_angle)
        A, b, _ = self._generate_rake_inequality_constraints_from_intervals(
            intervals_by_fault
        )
        return A, b

    def _generate_rake_equality_constraints(self, fixed_rake):
        """Build one fixed-rake equality per selected Fault patch.

        Returns
        -------
        tuple of numpy.ndarray
            Equality matrix and zero right-hand side in linear-suffix
            coordinates.

        Notes
        -----
        Every row encodes
        ``ss * sin(rake) - ds * cos(rake) = 0``.  Non-Fault sources are
        skipped.
        """
        nlinear = self._get_linear_matrix_n_parameters()
        
        # Filter faults with fixed rake constraints that exist AND are Fault type
        constrained_faults = [fault for fault in self.config.faults_list 
                             if fault.name in fixed_rake
                             and self._get_source_type(fault.name) == 'Fault']
        
        if not constrained_faults:
            return np.zeros((0, nlinear)), np.zeros(0)
        
        npatch = sum(self._get_n_spatial_elements(fault) for fault in constrained_faults)
        A_eq = np.zeros((npatch, nlinear))
        b_eq = np.zeros(npatch)

        patch_count = 0
        for fault in constrained_faults:
            start = self._get_linear_matrix_source_start(fault.name)
            
            rake = fixed_rake[fault.name]
            inpatch = self._get_n_spatial_elements(fault)
            adapter = getattr(self.multifaults, 'adapters', {}).get(fault.name)
            ss_start, ds_start = self._rake_component_starts(
                fault, start, inpatch, adapter=adapter
            )
            
            for i in range(inpatch):
                # Fixed rake: ss*sin(rake) - ds*cos(rake) = 0
                A_eq[patch_count + i, ss_start + i] = np.sin(np.deg2rad(rake))
                A_eq[patch_count + i, ds_start + i] = -np.cos(np.deg2rad(rake))
                
            patch_count += inpatch
        
        return A_eq, b_eq

    # ==================== Constraint Management API ====================
    
    def set_fixed_rake_constraints(self, fixed_rake, *, source='manual'):
        """Replace the active fixed-rake equality group.

        Parameters
        ----------
        fixed_rake : mapping
            Fault names mapped to finite rake angles in degrees.  An empty
            mapping is a no-op; use :meth:`clear_fixed_rake_constraints` to
            remove an existing group.
        source : str
            Provenance label stored with the generated group.

        Raises
        ------
        RuntimeError
            If linear constraints are inactive for the sampling mode.
        TypeError
            If the declaration is not a mapping or an angle is not numeric.
        ValueError
            If a source is unknown, is not a Fault, has a non-finite angle, or
            resolves to no linear parameters.
        """
        if not self._is_smc_fj_mode():
            raise RuntimeError(
                "Fixed-rake equalities require SMC_FJ with ss_ds sampling"
            )
        if not isinstance(fixed_rake, Mapping):
            raise TypeError("fixed_rake must be a mapping of fault name to angle")
        # Empty input is a no-op on every backend.  Use the dedicated clear
        # operation when the existing fixed-rake group should be removed.
        if not fixed_rake:
            return
        validated_fixed_rake = {}
        for fault_name, rake in fixed_rake.items():
            if not self._fault_exists(fault_name):
                raise ValueError(
                    f"fixed_rake references unknown fault '{fault_name}'"
                )
            if self._get_source_type(fault_name) != 'Fault':
                raise ValueError(
                    f"fixed_rake only applies to Fault sources; "
                    f"'{fault_name}' is {self._get_source_type(fault_name)}"
                )
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
            validated_fixed_rake[fault_name] = rake_value

        A_eq, b_eq = self._build_fixed_rake_matrix(
            validated_fixed_rake
        )
        if A_eq.size == 0:
            raise ValueError("fixed_rake did not resolve to any Fault parameters")
        self._register_equality_group(
            A_eq,
            b_eq,
            name='fixed_rake',
            source=source,
            replace=True,
            owner='managed',
            family='fixed_rake',
        )

    def clear_fixed_rake_constraints(self):
        """Remove the fixed-rake equality group if it exists.

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
        return True

    def add_euler_cap_constraints(self):
        """Generate and replace configured Euler-cap inequalities.

        Non-Fault entries are excluded before matrix generation.  The method
        is a no-op outside active linear mode, when disabled, or when no rows
        are generated.

        Raises
        ------
        Exception
            Propagates configuration, dataset, and matrix-generation errors.
        """
        if not self._is_smc_fj_mode():
            return

        try:
            interseismic_config = getattr(self.config, 'interseismic_config', {})
            if not interseismic_config.get('cap_constraints', {}).get('enabled', False):
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
            A_ineq, b_ineq = generate_euler_cap_constraints(
                self.inversion_instance,
                active_config,
                all_datasets,
            )
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
        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply Euler-cap constraints: {e}")
            raise

    def apply_interseismic_backslip_constraints(self):
        """Delegate configured hard backslip constraints to the inversion.

        The method is a no-op outside active linear mode.  Non-Fault sources
        are skipped; remaining declarations preserve selector, component,
        coupling, and fixed-value semantics.

        Raises
        ------
        Exception
            Propagates malformed declarations and inversion-side validation
            errors.
        """
        if not self._is_smc_fj_mode():
            return
        constraints = getattr(self.config, 'interseismic_config', {}).get('backslip_constraints', [])
        for index, spec in enumerate(constraints):
            if self._get_source_type(spec['fault']) != 'Fault':
                if self.verbose:
                    print(f"[!]  Warning: Interseismic backslip constraint skipping non-Fault source '{spec['fault']}'")
                continue
            self.inversion_instance._apply_interseismic_backslip_constraint(
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

        The matrix width is the SMC_FJ linear suffix.  The method is a no-op
        outside active linear mode or when no rows are generated.

        Raises
        ------
        Exception
            Propagates configuration and matrix-generation errors.
        """
        if not self._is_smc_fj_mode():
            return
        try:
            from .interseismic_parameter_model import generate_block_euler_equality_constraints

            interseismic_config = getattr(self.config, 'interseismic_config', {})
            n_linear = self._get_linear_matrix_n_parameters()
            A_eq, b_eq = generate_block_euler_equality_constraints(
                self.inversion_instance,
                interseismic_config,
                n_total=n_linear,
            )
            if A_eq is not None and A_eq.size > 0:
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

    def get_combined_inequality_constraints(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return all active inequalities in deterministic group order.

        Returns
        -------
        tuple
            Combined ``(A, b)`` in the SMC_FJ linear suffix, or
            ``(None, None)`` when the mode is inactive or no groups exist.
        """
        if not self._is_smc_fj_mode():
            return None, None
        return super().get_combined_inequality_constraints()

    def get_combined_equality_constraints(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return all active equalities in deterministic group order.

        Returns
        -------
        tuple
            Combined ``(A, b)`` in the SMC_FJ linear suffix, or
            ``(None, None)`` when the mode is inactive or no groups exist.
        """
        if not self._is_smc_fj_mode():
            return None, None
        return super().get_combined_equality_constraints()

    # ==================== Complete Constraint Application ====================
    
    def apply_source_constraints_from_config(self):
        """Apply adapter-defined source constraints from the loaded config.

        Returns
        -------
        list of str
            Names installed by the shared resolver.  The list is empty when
            linear constraints are inactive.

        Raises
        ------
        TypeError
            If the ``source_constraints`` declaration has an invalid shape.
        ValueError
            If a source, rule, matrix shape, or constraint type is invalid.

        Notes
        -----
        Active matrices address only the SMC_FJ linear suffix.
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
        In active ``SMC_FJ + ss_ds`` mode, linear declarations are generated
        after bounds.  Other modes retain applicable bounds and record
        linear-only declarations as inactive.  Failure restores the complete
        pre-call manager state.
        """
        if self.verbose:
            print("\n[RUN] Applying all constraints from configuration...")

        snapshot = self._snapshot_mutable_state()
        try:
            if bounds_config_file is not None:
                self.load_bounds_config(bounds_config_file, encoding)

            # A config reload is declarative replacement.  Remove only groups
            # owned by the previous files; manual/runtime groups remain live.
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

            if self._is_smc_fj_mode():
                self._rebuild_rake_constraints(rake_limits)

                interseismic_config = getattr(
                    self.config, 'interseismic_config', {}
                ) or {}
                if interseismic_config.get('blocks', {}).get('enabled', False):
                    self.apply_interseismic_block_constraints()
                if interseismic_config.get('cap_constraints', {}).get('enabled', False):
                    self.add_euler_cap_constraints()
                if interseismic_config.get('backslip_constraints'):
                    self.apply_interseismic_backslip_constraints()

                self._inactive_constraints = ()
            else:
                if 'rake_sector' in self._inequality_constraints:
                    self._remove_group(
                        'rake_sector',
                        expected_kind='inequality',
                        allow_managed=True,
                    )
                inactive = list(self._resolve_inactive_constraints())
                if (
                    rake_limits
                    and self.slip_sampling_mode == 'ss_ds'
                    and 'rake_linear_sector' not in inactive
                ):
                    inactive.append('rake_linear_sector')
                self._inactive_constraints = tuple(inactive)

            self.apply_source_constraints_from_config()
            self._validate_or_raise()
            self._mark_activation_flags_reconciled()
        except Exception:
            self._restore_mutable_state(snapshot)
            raise

        if self.verbose and self._should_warn():
            if self._inactive_constraints:
                joined = ', '.join(self._inactive_constraints)
                print(
                    "[SMC] FULLSMC uses bounds/priors; inactive linear "
                    f"constraints: {joined}"
                )
            print("[OK] All active constraints applied successfully")
            self.print_summary()

    def _rebuild_rake_constraints(self, additional_rake_limits: Dict = None):
        """Resolve configured and runtime rake sectors into one matrix group.

        Parameters
        ----------
        additional_rake_limits : dict, optional
            Call-specific fault rake sectors.  These override file-level
            declarations for matching names.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If called while linear constraints are inactive.
        TypeError
            If a resolved rake declaration has an unsupported type.
        ValueError
            If a source, selector, interval, or matrix layout is invalid.

        Notes
        -----
        Fault-level precedence is bounds config, then this argument, then
        persistent runtime declarations.  Patch-level declarations are
        resolved by the shared base implementation.
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

            # From script/API updates.
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
            A, b = self._build_rake_sector_matrix(final_rake_limits)
            if A.size > 0:
                self._register_inequality_group(
                    A,
                    b,
                    name='rake_sector',
                    source=source,
                    replace=True,
                    owner='managed',
                    family='rake_sector',
                )
            elif 'rake_sector' in self._inequality_constraints:
                self._remove_group(
                    'rake_sector',
                    expected_kind='inequality',
                    allow_managed=True,
                )

        except Exception as e:
            if self.verbose:
                print(f"[X] Failed to apply rake constraints: {e}")
            raise

    # ==================== Summary and Diagnostics ====================

    def get_constraint_snapshot(self, include_matrices=False, validate=False):
        """Return a read-only diagnostic snapshot with SMC mode metadata.

        Parameters
        ----------
        include_matrices : bool
            Include copies of named constraint matrices when ``True``.
        validate : bool
            Include a fresh validation report when ``True``.

        Returns
        -------
        dict
            Base snapshot augmented with sampling-mode identifiers and the
            currently inactive feature names. FULLSMC snapshots also include
            counts and values for effective sampler-bound defaults.
        """
        snapshot = super().get_constraint_snapshot(
            include_matrices=include_matrices,
            validate=validate,
        )
        snapshot['sampling_mode'] = {
            'bayesian': self.bayesian_sampling_mode,
            'slip': self.slip_sampling_mode,
        }
        snapshot['inactive_constraints'] = tuple(
            self._resolve_inactive_constraints()
        )
        if self.bayesian_sampling_mode == 'FULLSMC':
            _, _, default_metadata = (
                self._resolve_fullsmc_effective_bounds()
            )
            snapshot['effective_bounds_defaults'] = default_metadata
        return snapshot
    
    def print_summary(self):
        """Print sampled bounds, active matrices, modes, and provenance."""
        print("\n" + "="*70)
        print("[*] BOUNDS MANAGER SUMMARY")
        print("="*70)
        
        print(f"[OK] Configuration:")
        print(f"   Slip sampling mode: {self.slip_sampling_mode}")
        print(f"   Bayesian sampling mode: {self.bayesian_sampling_mode}")
        print(f"   Linear constraints supported: {'[OK] Yes' if self._is_smc_fj_mode() else '[X] No'}")
        
        # Configuration file info
        if self._bounds['config_file']:
            print(f"   Config file: {self._bounds['config_file']}")
        
        # Bounds summary with detailed parameter breakdown
        print(f"\n[STAT] Bounds Status:")
        if self._bounds['lb'] is not None or self._bounds['ub'] is not None:
            n_params = self.mcmc_samples
            n_lb = np.sum(~np.isnan(self._bounds['lb'])) if self._bounds['lb'] is not None else 0
            n_ub = np.sum(~np.isnan(self._bounds['ub'])) if self._bounds['ub'] is not None else 0
            n_both = 0
            if self._bounds['lb'] is not None and self._bounds['ub'] is not None:
                n_both = np.sum((~np.isnan(self._bounds['lb'])) & (~np.isnan(self._bounds['ub'])))
            
            print(f"   Total parameters: {n_params}")
            print(f"   Lower bounded: {n_lb}")
            print(f"   Upper bounded: {n_ub}")
            print(f"   Fully bounded: {n_both}")
            if self.bayesian_sampling_mode == 'FULLSMC':
                _, _, defaults = self._resolve_fullsmc_effective_bounds()
                print(
                    "   Effective FULLSMC defaults: "
                    f"lower={defaults['n_lower_defaulted']}, "
                    f"upper={defaults['n_upper_defaulted']}, "
                    "magnitude lower="
                    f"{defaults['n_magnitude_lower_defaulted']}"
                )
            
            # Global bounds
            global_lb = self._bounds['global']['lb']
            global_ub = self._bounds['global']['ub']
            if global_lb is not None or global_ub is not None:
                print(f"   Global defaults: lb={global_lb}, ub={global_ub}")
            
            # Parameter-specific bounds with detailed breakdown
            if hasattr(self.inversion_instance, 'linear_sample_start_position'):
                linear_start = self.inversion_instance.linear_sample_start_position
                print(f"   Hyperparameters: [0:{linear_start}] (geometry, sigma, alpha)")
                print(f"   Linear parameters: [{linear_start}:{self.mcmc_samples}] (slip, poly)")
            
            # Per-parameter bounds summary with element count
            bound_types = ['geometry', 'strikeslip', 'dipslip', 'slip_magnitude', 'rake_angle', 'poly']
            for bound_type in bound_types:
                bounds_dict = self._bounds[bound_type]
                if bounds_dict:
                    print(f"\n   {bound_type.capitalize()} bounds: {len(bounds_dict)} fault(s)")
                    for fault, bounds in bounds_dict.items():
                        if isinstance(bounds[0], np.ndarray):
                            # Per-element bounds
                            lb_array, ub_array = bounds
                            if len(lb_array) > 0 and len(ub_array) > 0:  # Check for non-empty arrays
                                print(f"     - {fault}: {len(lb_array)} elements")
                                print(f"       lb: [{lb_array.min():.3f} ... {lb_array.max():.3f}]")
                                print(f"       ub: [{ub_array.min():.3f} ... {ub_array.max():.3f}]")
                            else:
                                print(f"     - {fault}: {len(lb_array)} elements")

                        else:
                            # Uniform bounds (legacy)
                            print(f"     - {fault}: {bounds} (uniform)")
            
            # Global parameter bounds with element details
            if self._bounds['sigmas']:
                sigmas_bounds = self._bounds['sigmas']
                if isinstance(sigmas_bounds[0], np.ndarray):
                    lb_array, ub_array = sigmas_bounds
                    print(f"\n   Sigmas bounds: {len(lb_array)} elements")
                    print(f"     lb: [{lb_array.min():.3f} ... {lb_array.max():.3f}]")
                    print(f"     ub: [{ub_array.min():.3f} ... {ub_array.max():.3f}]")
                else:
                    print(f"\n   Sigmas bounds: {sigmas_bounds} (uniform)")
            
            if self._bounds['alpha']:
                alpha_bounds = self._bounds['alpha']
                if isinstance(alpha_bounds[0], np.ndarray):
                    lb_array, ub_array = alpha_bounds
                    print(f"\n   Alpha bounds: {len(lb_array)} elements")
                    print(f"     lb: [{lb_array.min():.3f} ... {lb_array.max():.3f}]")
                    print(f"     ub: [{ub_array.min():.3f} ... {ub_array.max():.3f}]")
                else:
                    print(f"\n   Alpha bounds: {alpha_bounds} (uniform)")
                    
            print(f"\n   Source: {self._bounds['source']}")
        else:
            print("   No bounds set")
        
        # Constraints summary (SMC_FJ only)
        if self._is_smc_fj_mode():
            print(f"\n[INQ] Inequality Constraints: {len(self._inequality_constraints)} groups")
            total_ineq = sum(c['A'].shape[0] for c in self._inequality_constraints.values())
            print(f"   Total constraints: {total_ineq}")
            
            for name, constraint in self._inequality_constraints.items():
                print(f"   - {name}: {constraint['A'].shape[0]} constraints (source: {constraint['source']})")
            
            print(f"\n[EQ] Equality Constraints: {len(self._equality_constraints)} groups")
            total_eq = sum(c['A'].shape[0] for c in self._equality_constraints.values())
            print(f"   Total constraints: {total_eq}")
            
            for name, constraint in self._equality_constraints.items():
                print(f"   - {name}: {constraint['A'].shape[0]} constraints (source: {constraint['source']})")
        
        print("="*70)

    def validate(self) -> Dict[str, Any]:
        """Validate bounds, matrices, and SMC-mode compatibility.

        Returns
        -------
        dict
            Base validation report extended with warnings for inactive linear
            semantics and undefined bounds, plus errors for negative explicit
            slip-magnitude lower bounds.
        """
        result = super().validate()

        # SMC-specific: mode compatibility
        if self.bayesian_sampling_mode == 'SMC_FJ' and self.slip_sampling_mode != 'ss_ds':
            result['warnings'].append("Linear constraints only supported in SMC_FJ mode with ss_ds sampling")

        # SMC-specific: undefined bounds warning
        undefined_count = 0
        if self._bounds['lb'] is not None:
            undefined_count += int(np.sum(np.isnan(self._bounds['lb'])))
        if self._bounds['ub'] is not None:
            undefined_count += int(np.sum(np.isnan(self._bounds['ub'])))
        if undefined_count > 0:
            result['warnings'].append(f"{undefined_count} bounds are undefined (will use defaults)")

        if self._bounds['lb'] is not None:
            for fault_name, magnitude_slice in self._iter_fault_magnitude_slices() or ():
                lower = self._bounds['lb'][magnitude_slice]
                if np.any(np.isfinite(lower) & (lower < 0.0)):
                    result['errors'].append(
                        f"slip_magnitude lower bounds for '{fault_name}' must "
                        "be non-negative"
                    )
                    result['valid'] = False

        return result
