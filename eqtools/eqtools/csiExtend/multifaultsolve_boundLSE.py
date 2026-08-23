# import the necessary libraries
import warnings
from contextlib import contextmanager

from csi import multifaultsolve
import copy
import yaml
import numpy as np
import pyproj as pp
from .fnnls import fnnls
from scipy.linalg import block_diag as blkdiag
# import self-written library
from . import lsqlin
from ..viztools import sci_plot_style, DegreeFormatter
from .fault_analysis_mixin import FaultAnalysisMixin
from .constraint_manager_blse import ConstraintManagerBLSE
from .covariance_utils import prepare_block_covariance_metrics
from .source_adapters import make_adapter

# Plot
from eqtools.getcpt import get_cpt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import cmcrameri # cmc.devon_r cmc.lajolla_r cmc.batlow


_LSQLIN_ACCEPTED_STATUS = {"optimal", "optimal_inaccurate"}
_LINEAR_CONSTRAINT_TOL = 1.0e-6


def _validate_lsqlin_status(ret, context="lsqlin"):
    """Raise a clear error when the underlying QP solver did not converge."""
    status = str(ret.get("status", "")).lower()
    if status and status not in _LSQLIN_ACCEPTED_STATUS:
        objective = ret.get("primal objective", None)
        suffix = "" if objective is None else f"; primal objective={objective}"
        raise ValueError(f"{context} failed: solver status is '{status}'{suffix}")


def _constraint_vector(value):
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(-1)


def _constraint_matrix(value):
    if value is None:
        return None
    return value


def _max_violation(values):
    if values is None:
        return 0.0
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.max(values))


def _validate_linear_solution_constraints(
    x,
    lb,
    ub,
    A_ineq,
    b_ineq,
    A_eq,
    b_eq,
    *,
    context="linear solve",
    tol=_LINEAR_CONSTRAINT_TOL,
):
    """Validate the recovered solution against original linear constraints."""
    x = np.asarray(x, dtype=float).reshape(-1)
    lb = _constraint_vector(lb)
    ub = _constraint_vector(ub)
    b_ineq = _constraint_vector(b_ineq)
    b_eq = _constraint_vector(b_eq)
    A_ineq = _constraint_matrix(A_ineq)
    A_eq = _constraint_matrix(A_eq)

    violations = {}
    if lb is not None:
        violations["lower_bound"] = _max_violation(lb - x)
    if ub is not None:
        violations["upper_bound"] = _max_violation(x - ub)
    if A_ineq is not None and b_ineq is not None:
        violations["inequality"] = _max_violation(A_ineq.dot(x) - b_ineq)
    if A_eq is not None and b_eq is not None:
        eq_residual = np.asarray(A_eq.dot(x), dtype=float).reshape(-1) - b_eq
        violations["equality"] = _max_violation(np.abs(eq_residual))

    failed = {name: value for name, value in violations.items() if value > tol}
    if failed:
        details = ", ".join(f"{name}={value:.3e}" for name, value in failed.items())
        raise ValueError(f"{context} produced a solution that violates constraints: {details}")


class multifaultsolve_boundLSE(multifaultsolve, FaultAnalysisMixin):
    '''
    Enhanced multi-fault solver with unified constraint management.
    
    Features:
    - Unified constraint and bounds management via ConstraintManager
    - Laplace smoothing constraints
    - Depth-Equalized Smoothing (DES) support
    - Simple Variance Component Estimation (VCE)
    - Multiple solver backends (lsqlin, fnnls)
    '''
    
    def __init__(self, name, faults, verbose=True, extra_parameters=None, des_enabled=False, des_config=None):
        super(multifaultsolve_boundLSE, self).__init__(name,
                                                faults,
                                                verbose=verbose)
        
        # Build source adapters
        self.adapters = {fault.name: make_adapter(fault) for fault in faults}

        # Prepare one covariance metric per data set. This is intentionally
        # done once; solver iterations only reuse the resulting whiteners.
        self.prepare_data_covariance_metrics()
        self.calculate_slip_and_poly_positions()

        # Configure DES Logger
        from .des_utils import setup_des_logging
        import logging
        
        # Smart logging level detection
        # 1. If user explicitly configured logging (e.g. to DEBUG), we want to respect that.
        # 2. If user didn't configure (default WARNING), but verbose=True, we want INFO.
        root_level = logging.getLogger().getEffectiveLevel()
        
        if verbose:
            # If verbose=True, ensure at least INFO. But if root is DEBUG (10 < 20), allow DEBUG.
            level = min(logging.INFO, root_level)
        else:
            # If verbose=False, ensure at least WARNING.
            level = max(logging.WARNING, root_level)
            
        setup_des_logging(level)

        # Initialize storage for bounds and constraints
        self._lb = None
        self._ub = None
        self._A_ueq = None
        self._b_ueq = None
        self._Aeq = None
        self._beq = None

        # Initialize unified constraint manager
        self.constraint_manager = ConstraintManagerBLSE(
            solver=self,
            verbose=verbose,
        )
        
        # Initialize parameter count
        self.lsq_parameters = self._calculate_total_parameters()
        
        if extra_parameters is not None:
            self.ramp_switch = len(extra_parameters)
        else:
            self.ramp_switch = 0
        
        # DES (Depth-Equalized Smoothing) parameters
        self.des_enabled = des_enabled
        self.des_config = des_config if des_config is not None else {
            'mode': 'per_patch',
            'G_norm': 'l2',
            'depth_grouping': {
                'strategy': 'uniform',
                'interval': 1.0
                }
        }
        
        if verbose:
            print(f"[OK] Enhanced multifaultsolve_boundLSE initialized with unified constraint management")
        
        return
    
    def _calculate_total_parameters(self):
        """Calculate total number of parameters for all faults."""
        total = 0
        for fault in self.faults:
            adapter = self.adapters[fault.name]
            num_source_params = adapter.get_n_source_params()
            num_poly_samples = np.sum([fault.numberofpolys[ikey] for ikey in fault.numberofpolys], dtype=int)
            total += num_source_params + num_poly_samples
        return total
    
    def _build_data_ranges(self):
        """Return the canonical observation-row range for each data set."""
        ranges = {}
        start = 0
        for data_name in self.faults[0].datanames:
            end = start + np.asarray(self.faults[0].d[data_name]).shape[0]
            ranges[data_name] = (start, end)
            start = end
        return ranges

    def prepare_data_covariance_metrics(self):
        """Factor each covariance block directly and cache its metric.

        ``self.Cd`` is assembled block-diagonally by CSI. Cross-data terms are
        rejected because BLSE/VCE assign independent weights or variance
        components to the named data sets and cannot represent such terms.
        """
        self.data_ranges = self._build_data_ranges()
        self.data_covariance_metrics = prepare_block_covariance_metrics(
            self.Cd,
            self.data_ranges,
        )
        return self.data_covariance_metrics

    def _normalize_data_weights(self, data_weight):
        """Return one multiplicative whitening weight per data set."""
        n_data = len(self.data_ranges)
        if isinstance(data_weight, (int, float)):
            return np.full(n_data, float(data_weight))
        if isinstance(data_weight, (list, np.ndarray)):
            weights = np.asarray(data_weight, dtype=float).reshape(-1)
            if weights.size != n_data:
                raise ValueError(
                    "The length of data_weight must equal the number of data sets."
                )
            return weights
        raise ValueError("data_weight must be a scalar or a list of scalars.")

    def _whiten_data_system(self, G, d, data_weight=1.0):
        """Whiten ``G`` and ``d`` blockwise without a global dense operator."""
        if not hasattr(self, "data_covariance_metrics"):
            self.prepare_data_covariance_metrics()
        weights = self._normalize_data_weights(data_weight)
        G_blocks = []
        d_blocks = []
        for weight, (data_name, (start, end)) in zip(
            weights, self.data_ranges.items()
        ):
            metric = self.data_covariance_metrics[data_name]
            G_blocks.append(metric.whiten(G[start:end, :]) * weight)
            d_blocks.append(metric.whiten(d[start:end]) * weight)
        return np.vstack(G_blocks), np.concatenate(d_blocks)

    @contextmanager
    def _blse_quadratic_scan_context(self):
        """Provide one call-local cache for a fixed-geometry BLSE scan.

        The cache is intentionally scoped to :meth:`simple_run_loop`.  It is
        never retained across independent ``run()`` calls, where users may
        legitimately have changed geometry, Green's functions, covariance, or
        smoothing.  Nested diagnostic transactions restore the outer context.
        """
        sentinel = object()
        previous = getattr(self, '_blse_quadratic_scan_cache', sentinel)
        self._blse_quadratic_scan_cache = {}
        try:
            yield self._blse_quadratic_scan_cache
        finally:
            if previous is sentinel:
                delattr(self, '_blse_quadratic_scan_cache')
            else:
                self._blse_quadratic_scan_cache = previous

    def _prepare_data_quadratic(self, G, d, data_weight=1.0):
        """Accumulate the exact BLSE data contribution without stacking rows.

        ``data_weight`` is the multiplicative residual-row weight used by the
        public BLSE API.  It therefore enters the normal matrix and right-hand
        side quadratically.  Scaling both ``WG`` and ``Wd`` before their Gram
        and cross products preserves that convention for signed or zero legacy
        weights as well as the normal positive ``1 / sigma`` case.
        """
        if not hasattr(self, 'data_covariance_metrics'):
            self.prepare_data_covariance_metrics()
        weights = self._normalize_data_weights(data_weight)
        G = np.asarray(G, dtype=float)
        d = np.asarray(d, dtype=float).reshape(-1)
        n_parameters = G.shape[1]
        H_data = np.zeros((n_parameters, n_parameters), dtype=float)
        q_data = np.zeros(n_parameters, dtype=float)
        for weight, (data_name, (start, end)) in zip(
            weights, self.data_ranges.items()
        ):
            metric = self.data_covariance_metrics[data_name]
            WG = metric.whiten(G[start:end, :]) * weight
            Wd = metric.whiten(d[start:end]) * weight
            H_data += WG.T @ WG
            q_data -= WG.T @ Wd
        return H_data, q_data

    def _data_quadratic_for_solve(self, G, d, data_weight, *, cache=False):
        """Return current data ``H, q``, optionally reusing a scan-local copy."""
        scan_cache = (
            getattr(self, '_blse_quadratic_scan_cache', None)
            if cache else None
        )
        weights = self._normalize_data_weights(data_weight)
        cache_matches = (
            scan_cache is not None
            and scan_cache.get('G_object') is G
            and scan_cache.get('d_object') is d
            and np.array_equal(scan_cache.get('data_weights'), weights)
        )
        if cache_matches:
            return (
                scan_cache['H_data'].copy(),
                scan_cache['q_data'].copy(),
            )

        H_data, q_data = self._prepare_data_quadratic(G, d, weights)
        if scan_cache is not None:
            scan_cache.update({
                'G_object': G,
                'd_object': d,
                'data_weights': weights.copy(),
                'H_data': H_data.copy(),
                'q_data': q_data.copy(),
            })
        return H_data, q_data

    def _assemble_blse_quadratic(
            self, G, d, smoothing_matrix, data_weight,
            *, use_scan_cache=False, penalty_weight=None):
        """Build the same ``H, q`` as the materialized BLSE residual system."""
        H, q = self._data_quadratic_for_solve(
            G, d, data_weight, cache=use_scan_cache,
        )
        smoothing_matrix = np.asarray(smoothing_matrix, dtype=float)
        if smoothing_matrix.ndim != 2 or smoothing_matrix.shape[1] != H.shape[0]:
            raise ValueError(
                "smoothing_matrix must be two-dimensional and match the "
                "assembled model-column layout"
            )
        if smoothing_matrix.shape[0] == 0:
            return H, q

        scan_cache = (
            getattr(self, '_blse_quadratic_scan_cache', None)
            if use_scan_cache else None
        )
        base = None if scan_cache is None else scan_cache.get(
            'base_smoothing_matrix'
        )
        weights = np.asarray(penalty_weight, dtype=float).reshape(-1)
        uniform_weight = (
            weights.size > 0
            and np.all(weights == weights[0])
            and np.isfinite(weights[0])
        )
        can_reuse_smoothing = (
            base is not None
            and uniform_weight
            and np.asarray(base).shape == smoothing_matrix.shape
        )
        if can_reuse_smoothing:
            if 'S_base' not in scan_cache:
                base = np.asarray(base, dtype=float)
                scan_cache['S_base'] = base.T @ base
            H += float(weights[0]) ** 2 * scan_cache['S_base']
        else:
            H += smoothing_matrix.T @ smoothing_matrix
        return H, q

    def _assemble_blse_residual_system(
            self, G, d, smoothing_matrix, data_weight):
        """Materialize the exact BLSE residual form for the rare fallback."""
        G_data, d_data = self._whiten_data_system(G, d, data_weight)
        smoothing_matrix = np.asarray(smoothing_matrix, dtype=float)
        return (
            np.vstack((G_data, smoothing_matrix)),
            np.hstack((d_data, np.zeros(smoothing_matrix.shape[0]))),
        )

    def calculate_slip_and_poly_positions(self):
        """
        Calculates indices for source params and poly for all source types.
        Uses source adapters for type-safe parameter counting.
        DES configuration is built only for sources that support it.
        """
        self.slip_positions = {}
        self.poly_positions = {}
        self.des_indices_config = []  # Only for sources that support DES

        current_idx = 0

        for fault in self.faults:
            adapter = self.adapters[fault.name]
            params_per_comp = adapter.get_n_params_per_component()
            comp_names = adapter.get_param_names()

            # --- Build per-component index ranges ---
            source_start = current_idx
            comp_indices = {}  # {comp_name: list of indices}
            for comp_name in comp_names:
                n = params_per_comp[comp_name]
                comp_indices[comp_name] = list(range(current_idx, current_idx + n))
                current_idx += n

            slip_end_idx = current_idx
            self.slip_positions[fault.name] = (source_start, slip_end_idx)

            # --- Poly indices ---
            num_poly = 0
            if hasattr(fault, 'numberofpolys'):
                num_poly = np.sum([fault.numberofpolys.get(ikey, 0)
                                   for ikey in fault.numberofpolys], dtype=int)
            if num_poly > 0:
                poly_indices = list(range(current_idx, current_idx + num_poly))
                current_idx += num_poly
            else:
                poly_indices = []
            self.poly_positions[fault.name] = (slip_end_idx, current_idx)

            # --- DES configuration (only for sources that support it) ---
            ss_indices = comp_indices.get('strikeslip', [])
            ds_indices = comp_indices.get('dipslip', [])
            des_cfg = adapter.get_des_config(ss_indices, ds_indices, poly_indices)
            if des_cfg is not None:
                self.des_indices_config.append(des_cfg)

        self.lsq_parameters = current_idx

    def update_bounds(
        self,
        *,
        lb=None,
        ub=None,
        strikeslip_bounds=None,
        dipslip_bounds=None,
        poly_bounds=None,
        pressure_bounds=None,
        source_bounds=None,
    ):
        """
        Set bounds using constraint manager with type-safe handling.
        
        Parameters
        ----------
        lb, ub : float or array-like, optional
            Global lower/upper bounds
        strikeslip_bounds : dict, optional
            Strike-slip bounds per fault (only for Fault type)
            Format: {fault_name: (lb, ub)}
        dipslip_bounds : dict, optional
            Dip-slip bounds per fault (only for Fault type)
            Format: {fault_name: (lb, ub)}
        poly_bounds : dict, optional
            Polynomial parameter bounds per fault
            Format: {fault_name: (lb, ub)}
        pressure_bounds : dict, optional
            Pressure parameter bounds (only for Pressure type)
            Format: {fault_name: (lb, ub)}
        source_bounds : dict, optional
            Generic per-source, per-component bounds.
            Format: {source_name: {component_name: (lb, ub)}}
            Works for any source type (Fault, Pressure, Sbarbot).
        """
        with self.constraint_manager.atomic_bounds_update():
            if lb is not None or ub is not None:
                self.constraint_manager.set_global_bounds(lb, ub, source="manual")

            for fault in self.faults:
                adapter = self.adapters[fault.name]
                if adapter.source_type == 'Fault':
                    ss_bounds = strikeslip_bounds.get(fault.name) if strikeslip_bounds else None
                    ds_bounds = dipslip_bounds.get(fault.name) if dipslip_bounds else None
                    if ss_bounds is not None or ds_bounds is not None:
                        self.constraint_manager.set_fault_slip_bounds(
                            fault.name, ss_bounds, ds_bounds, source="manual"
                        )

                elif adapter.source_type == 'Pressure':
                    p_bounds = pressure_bounds.get(fault.name) if pressure_bounds else None
                    if p_bounds is not None:
                        self.constraint_manager.set_source_component_bounds(
                            fault.name, {name: p_bounds for name in adapter.get_param_names()}, source="manual"
                        )

                if poly_bounds and fault.name in poly_bounds:
                    self.constraint_manager.set_fault_poly_bounds(
                        fault.name, poly_bounds[fault.name], source="manual"
                    )

            if source_bounds:
                for source_name, comp_bounds in source_bounds.items():
                    self.constraint_manager.set_source_component_bounds(
                        source_name, comp_bounds, source="manual"
                    )
        
        # Sync to solver
        self.constraint_manager.sync_to_solver()

    # Read-only diagnostic mirrors. The manager is the only writable source.
    @property
    def lb(self):
        """Return the latest manager-synchronized lower bounds."""
        return self._lb

    @property 
    def ub(self):
        """Return the latest manager-synchronized upper bounds."""
        return self._ub

    @property
    def A_ueq(self):
        """Return the manager-owned combined inequality matrix."""
        if self._A_ueq is not None:
            return self._A_ueq
        A, _ = self.constraint_manager.get_combined_inequality_constraints()
        return A

    @property
    def b_ueq(self):
        """Return the manager-owned combined inequality vector."""
        if self._b_ueq is not None:
            return self._b_ueq
        _, b = self.constraint_manager.get_combined_inequality_constraints()
        return b

    @property
    def Aeq(self):
        """Return the manager-owned combined equality matrix."""
        if self._Aeq is not None:
            return self._Aeq
        A, _ = self.constraint_manager.get_combined_equality_constraints()
        return A

    @property
    def beq(self):
        """Return the manager-owned combined equality vector."""
        if self._beq is not None:
            return self._beq
        _, b = self.constraint_manager.get_combined_equality_constraints()
        return b

    @property
    def inequality_constraints(self):
        """Read-only inequality-constraint view for diagnostics."""
        return self.constraint_manager.inequality_constraints

    @property
    def equality_constraints(self):
        """Read-only equality-constraint view for diagnostics."""
        return self.constraint_manager.equality_constraints

    def add_linear_inequality_constraint(
        self, A, b, *, name, source="user"
    ):
        """Add one user-owned ``A @ x <= b`` group."""
        with self.constraint_transaction():
            self.constraint_manager._register_inequality_group(
                A, b, name, source, owner='user'
            )
            self.constraint_manager.sync_to_solver()

    def replace_linear_inequality_constraint(
        self, A, b, *, name, source="user"
    ):
        """Replace one existing user-owned inequality group."""
        with self.constraint_transaction():
            self.constraint_manager._replace_user_group(
                'inequality', A, b, name=name, source=source
            )
            self.constraint_manager.sync_to_solver()

    def add_linear_equality_constraint(
        self, Aeq, beq, *, name, source="user"
    ):
        """Add one user-owned ``Aeq @ x = beq`` group."""
        with self.constraint_transaction():
            self.constraint_manager._register_equality_group(
                Aeq, beq, name, source, owner='user'
            )
            self.constraint_manager.sync_to_solver()

    def replace_linear_equality_constraint(
        self, Aeq, beq, *, name, source="user"
    ):
        """Replace one existing user-owned equality group."""
        with self.constraint_transaction():
            self.constraint_manager._replace_user_group(
                'equality', Aeq, beq, name=name, source=source
            )
            self.constraint_manager.sync_to_solver()

    def remove_linear_constraint(self, name):
        """Remove one user-owned raw linear group."""
        with self.constraint_transaction():
            self.constraint_manager._remove_group(name)
            self.constraint_manager.sync_to_solver()

    def get_linear_parameter_layout(self):
        """Return the validated BLSE/VCE linear parameter layout."""
        return self.constraint_manager.get_linear_parameter_layout()

    def constraint_transaction(self):
        """Return an advanced all-or-nothing constraint-update context."""
        return self.constraint_manager.constraint_transaction()

    def apply_constraints_from_config(
        self,
        bounds_config_file,
        *,
        encoding='utf-8',
    ):
        """Transactionally replace configuration-owned declarations."""
        result = self.constraint_manager._apply_constraint_config(
            bounds_config_file=bounds_config_file,
            encoding=encoding,
        )
        self.constraint_manager.sync_to_solver()
        return result

    def update_fault_rake_limits(self, rake_limits, *, source="manual"):
        """Merge fault-level runtime rake sectors and rebuild the matrix.

        ``rake_limits`` is a mapping ``{fault_name: [min_rake, max_rake]}``.
        Patch-level rake overrides already declared in ``patch_constraints``
        are preserved and resolved after these fault-level defaults.
        """
        return self.constraint_manager.update_fault_rake_limits(
            rake_limits,
            replace=False,
            source=source,
            sync=True,
        )

    def replace_fault_rake_limits(self, rake_limits, *, source="manual"):
        """Replace all fault-level runtime rake sectors."""
        return self.constraint_manager.update_fault_rake_limits(
            rake_limits,
            replace=True,
            source=source,
            sync=True,
        )

    def clear_fault_rake_limits(self):
        """Clear script-side fault rake sectors, preserving config/patch rake."""
        return self.constraint_manager.clear_fault_rake_limits(sync=True)

    def add_patch_constraints(self, patch_constraints, *, source="manual"):
        """Add patch-level bounds/rake override specs before solving.

        The specs update the constraint manager declaration layer and then
        refresh resolved bounds/rake constraints.  Use this helper before
        ``ConstrainedLeastSquareSoln``/``run``.
        """
        return self.constraint_manager.add_patch_constraints(
            patch_constraints,
            source=source,
            sync=True,
        )

    def replace_patch_constraints(self, patch_constraints, *, source="manual"):
        """Replace script-side patch overrides while keeping config rules."""
        return self.constraint_manager.replace_patch_constraints(
            patch_constraints,
            source=source,
            sync=True,
        )

    def clear_patch_constraints(self, *, source="manual"):
        """Remove script-side patch overrides while keeping config rules."""
        return self.constraint_manager.clear_patch_constraints(
            source=source,
            sync=True,
        )
        
    def set_fixed_rake_constraints(self, fixed_rake):
        """Replace the managed fixed-rake equality family."""
        with self.constraint_transaction():
            result = self.constraint_manager.set_fixed_rake_constraints(
                fixed_rake,
                source="manual",
            )
            self.constraint_manager.sync_to_solver()
            return result

    def clear_fixed_rake_constraints(self):
        """Remove fixed-rake equalities; repeated calls are safe."""
        return self.constraint_manager.clear_fixed_rake_constraints(sync=True)

    def set_incompressibility_constraints(self, source_names=None):
        """Set incompressibility equality constraints for Sbarbot sources.

        For each volume element: eps11 + eps22 + eps33 = 0.

        Parameters
        ----------
        source_names : str or list of str, optional
            Sbarbot source name(s) to constrain. ``None`` applies to all
            Sbarbot sources in the solver.
        """
        if source_names is None:
            source_names = [f.name for f in self.faults
                            if self.adapters[f.name].source_type == 'Sbarbot']
        elif isinstance(source_names, str):
            source_names = [source_names]

        n_total = self.lsq_parameters
        with self.constraint_transaction():
            self.constraint_manager._remove_groups_by_owner(
                'managed',
                families={'incompressibility'},
            )
            for sname in source_names:
                adapter = self.adapters[sname]
                if adapter.source_type != 'Sbarbot':
                    raise TypeError(f"'{sname}' is not a Sbarbot source")
                param_start = self.slip_positions[sname][0]
                cfg = {
                    'incompressible': {
                        'type': 'equality',
                        'rule': 'incompressible',
                    }
                }
                for cname, A, b in adapter.generate_source_equality_constraints(
                    cfg,
                    param_start,
                    n_total,
                ):
                    full_name = f"src_{sname}_{cname}"
                    self.constraint_manager._register_equality_group(
                        A,
                        b,
                        name=full_name,
                        source=f"incompressibility/{sname}",
                        replace=True,
                        owner='managed',
                        family='incompressibility',
                    )
            self.constraint_manager.sync_to_solver()

    def clear_incompressibility_constraints(self):
        """Remove the complete managed incompressibility family."""
        removed = self.constraint_manager._remove_groups_by_owner(
            'managed',
            families={'incompressibility'},
        )
        self.constraint_manager.sync_to_solver()
        return tuple(name for _, name in removed)

    def print_constraint_summary(self):
        """Print constraint summary using constraint manager."""
        self.constraint_manager.print_summary()

    def get_constraint_snapshot(self, include_matrices=False, validate=False):
        """Return a diagnostic snapshot of manager-owned constraints.

        The returned object is for inspection. Use the inversion facade to
        mutate constraints.
        """
        return self.constraint_manager.get_constraint_snapshot(
            include_matrices=include_matrices,
            validate=validate,
        )

    @property
    def bounds(self):
        """Read-only bounds view for diagnostics."""
        return self.constraint_manager.bounds

    @staticmethod
    def _normalize_fault_slip_component(component):
        comp = str(component).lower().replace(' ', '').replace('_', '')
        if comp in ('strikeslip', 'ss', 's', 'strike'):
            return 'strikeslip'
        if comp in ('dipslip', 'ds', 'd', 'dip'):
            return 'dipslip'
        raise ValueError(
            f"Unknown slip component '{component}'. Please use 'strikeslip' or 'dipslip'."
        )

    def add_zero_edge_slip_constraint(self, fault_names, edges, slip_modes):
        """
        Add zero-slip equality constraints for triangles on specified fault edges.

        Builds a constraint matrix per (fault, edge, slip_mode) combination and
        registers one equality group per combination instead of looping
        triangle by triangle.

        Parameters
        ----------
        fault_names : str or list of str
            Fault name(s) to constrain.
        edges : str or list of str
            Edge name(s), e.g. 'top', 'bottom', 'left', 'right'.
        slip_modes : str or list of str
            Slip mode(s) to zero out (case-insensitive, spaces/underscores ignored).
            Strike-slip aliases: 'strikeslip', 'strike_slip', 'strike slip', 'ss'.
            Dip-slip   aliases: 'dipslip',    'dip_slip',    'dip slip',    'ds'.

        Examples
        --------
        # Zero both slip components on the top edge of one fault
        inversion.add_zero_edge_slip_constraint(
            'Aheqi_2025', 'top', ['strikeslip', 'dipslip'])

        # Zero dip-slip on top and bottom edges of two faults
        inversion.add_zero_edge_slip_constraint(
            ['FaultA', 'FaultB'], ['top', 'bottom'], 'dip slip')
        """
        if isinstance(fault_names, str):
            fault_names = [fault_names]
        if isinstance(edges, str):
            edges = [edges]
        if isinstance(slip_modes, str):
            slip_modes = [slip_modes]

        slip_modes = list(dict.fromkeys(self._normalize_fault_slip_component(m) for m in slip_modes))

        groups = []
        for fault_name in fault_names:
            if fault_name not in self.faults_dict:
                raise ValueError(
                    f"Fault '{fault_name}' not found. Available: {list(self.faults_dict.keys())}"
                )
            fault = self.faults_dict[fault_name]

            if not hasattr(fault, 'edge_triangles_indices'):
                raise AttributeError(
                    f"Fault '{fault_name}' has no 'edge_triangles_indices'. "
                    "Run edge detection first."
                )

            for edge in edges:
                if edge not in fault.edge_triangles_indices:
                    available = list(fault.edge_triangles_indices.keys())
                    raise KeyError(
                        f"Edge '{edge}' not found in fault '{fault_name}'. "
                        f"Available: {available}"
                    )
                tri_indices = np.asarray(fault.edge_triangles_indices[edge])

                for slip_mode in slip_modes:
                    global_indices = (
                        self.constraint_manager
                        ._get_component_columns_for_patches(
                            fault_name,
                            slip_mode,
                            tri_indices,
                            space='active_linear',
                        )
                    )
                    n_constrained = len(global_indices)

                    A = np.zeros((n_constrained, self.lsq_parameters))
                    A[np.arange(n_constrained), global_indices] = 1.0
                    b = np.zeros(n_constrained)

                    name = f"zero_edge_{fault_name}_{edge}_{slip_mode}"
                    groups.append((name, A, b))

        with self.constraint_transaction():
            for name, A, b in groups:
                self.constraint_manager._register_equality_group(
                    A,
                    b,
                    name=name,
                    source="zero_edge_slip",
                    owner="user",
                )
            self.constraint_manager.sync_to_solver()
        return [name for name, _, _ in groups]

    def add_patch_slip_constraint(self, fault_patches, slip_component, value=0.0, constraint_type='equality', operator='=='):
        """
        Set slip constraints for specific sub-fault patches.

        This method allows setting equality (e.g., slip = 0) or inequality 
        (e.g., slip >= 0) constraints for the strike-slip or dip-slip 
        components of a given set of patches.

        Parameters
        ----------
        fault_patches : dict
            Dictionary mapping fault names to lists of patch indices.
            Format: {'fault_name': [patch_idx1, patch_idx2, ...]}
        slip_component : str or list of str
            Slip component(s) to constrain. Can be 'strikeslip' or 'dipslip'.
            Aliases such as 'ss' and 'ds' are also accepted.
        value : float, optional
            The constraint value. Default is 0.0.
        constraint_type : str, optional
            Type of constraint: 'equality' or 'inequality'. Default is 'equality'.
        operator : str, optional
            Operator used for inequality constraints ('<=' or '>=').
            Ignored for equality constraints. Default is '=='.
        """

        if isinstance(slip_component, str):
            slip_components = [slip_component]
        else:
            slip_components = list(slip_component)

        all_global_indices = []

        for f_name, patch_indices in fault_patches.items():
            if f_name not in self.faults_dict:
                raise ValueError(f"Fault '{f_name}' not found. Available faults: {list(self.faults_dict.keys())}")

            for s_comp in slip_components:
                columns = (
                    self.constraint_manager
                    ._get_component_columns_for_patches(
                        f_name,
                        s_comp,
                        patch_indices,
                        space='active_linear',
                    )
                )
                all_global_indices.extend(columns.tolist())

        n_constrained = len(all_global_indices)
        if n_constrained == 0:
            return

        A = np.zeros((n_constrained, self.lsq_parameters))
        A[np.arange(n_constrained), all_global_indices] = 1.0
        b = np.full(n_constrained, value)

        f_name_str = "_".join(fault_patches.keys())[:20]
        c_name_str = "_".join(slip_components)[:15]
        name = f"patch_slip_constraint_{f_name_str}_{c_name_str}"

        if constraint_type == 'equality':
            self.add_linear_equality_constraint(
                A,
                b,
                name=name,
                source='manual',
            )
        elif constraint_type == 'inequality':
            if operator in ('<=', '<'):
                # A*x <= b is the standard form
                pass
            elif operator in ('>=', '>'):
                # A*x >= b  => -A*x <= -b
                A = -A
                b = -b
            else:
                raise ValueError(f"Unsupported inequality operator '{operator}'. Please use '<=' or '>='.")
            self.add_linear_inequality_constraint(
                A,
                b,
                name=name,
                source='manual',
            )
        else:
            raise ValueError(f"Invalid constraint type '{constraint_type}'. Please use 'equality' or 'inequality'.")

    def ConstrainedLeastSquareSoln(self, penalty_weight=1., smoothing_matrix=None, data_weight=1.,
                                smoothing_constraints=None, method='mudpy',
                                verbose=False, extra_parameters=None,
                                iterations=1000, tolerance=None, maxfun=100000, des_enabled=None,
                                validate_constraints=True):
        """Solve the constrained, covariance-whitened BLSE system.

        ``smoothing_matrix`` is a complete matrix in assembled model-column
        coordinates.  It is mutually exclusive with ``smoothing_constraints``.
        When ECAT constructs the Laplacian, every source retains its model
        columns, but only adapters that support smoothing contribute rows.
        ``self.G_lap_base`` records the unweighted rows and ``self.G_lap`` the
        rows multiplied by ``penalty_weight``; neither is reconstructed during
        result reporting.
        """
        self.constraint_manager._require_activation_flags_reconciled(
            "BLSE constrained least squares"
        )

        # Validate constraints using constraint manager
        if validate_constraints:
            validation = self.constraint_manager.validate()
            if not validation['valid']:
                print("[X] Constraint validation failed:")
                for error in validation['errors']:
                    print(f"   Error: {error}")
                raise ValueError("Invalid constraints detected")
            
            if validation['warnings'] and verbose:
                print("[!]  Constraint validation warnings:")
                for warning in validation['warnings']:
                    print(f"   Warning: {warning}")

        if verbose:
            print("\n[RUN] Starting constrained least squares with unified constraint management:")
            if not self.constraint_manager.verbose:
                self.constraint_manager.print_summary()

        # Ensure constraints are synced
        self.constraint_manager.sync_to_solver()

        # Import DES utilities
        from .des_utils import apply_des_transformation, recover_sf_with_poly

        # Determine if DES should be used
        use_des = des_enabled if des_enabled is not None else self.des_enabled

        # Get the faults
        faults = self.faults

        # Get the matrixes and vectors
        G = self.G
        Cd = self.Cd
        d = self.d

        # Nd = d.shape[0]
        Np = G.shape[1]

        # ----------------------------Smoothing matrix-----------------------------#
        if smoothing_matrix is not None and smoothing_constraints is not None:
            raise ValueError(
                "Provide either smoothing_matrix or smoothing_constraints, "
                "not both. Their precedence would otherwise be ambiguous."
            )

        if smoothing_matrix is None:
            if isinstance(penalty_weight, (int, float)):
                penalty_weight = np.ones(len(faults)) * penalty_weight
            elif isinstance(penalty_weight, (list, np.ndarray)):
                assert len(penalty_weight) == len(faults), "The length of penalty_weight should be equal to the number of faults."
            else:
                raise ValueError("penalty_weight should be a scalar or a list of scalars.")

            # Only smoothing-capable sources own Laplacian rows.  All sources
            # still retain their columns in the full model layout.  Avoiding
            # zero-row placeholders is mathematically neutral for the solve,
            # but keeps roughness diagnostics and matrix dimensions truthful.
            smoothing_faults = [
                fault.name
                for fault in faults
                if self.adapters[fault.name].supports_smoothing()
            ]
            if smoothing_constraints is not None:
                if isinstance(smoothing_constraints, tuple) and len(smoothing_constraints) == 4:
                    smoothing_constraints = {
                        fault_name: smoothing_constraints
                        for fault_name in smoothing_faults
                    }
                elif isinstance(smoothing_constraints, dict):
                    missing = set(smoothing_faults) - set(smoothing_constraints)
                    if missing:
                        raise ValueError(
                            "smoothing_constraints is missing smoothing-capable "
                            "sources: " + ", ".join(sorted(missing))
                        )
                    smoothing_constraints = dict(smoothing_constraints)
                else:
                    raise ValueError("smoothing_constraints should be a 4-tuple or a dictionary with fault names as keys and 4-tuples as values.")
            else:
                smoothing_constraints = {
                    fault_name: (None, None, None, None)
                    for fault_name in smoothing_faults
                }

            base_blocks = []
            weighted_blocks = []
            for fault, ipenalty_weight in zip(faults, penalty_weight):
                st = self.fault_indexes[fault.name][0]
                adapter = self.adapters[fault.name]
                if not adapter.supports_smoothing():
                    continue

                lap = fault.buildLaplacian(
                    method=method,
                    bounds=smoothing_constraints[fault.name],
                )
                n_components = len(adapter.get_param_names())
                if n_components == 1:
                    lap_sd = lap
                else:
                    lap_sd = blkdiag(*([lap] * n_components))
                lap_sd = np.asarray(lap_sd, dtype=float)
                se = st + lap_sd.shape[1]
                if se > Np:
                    raise ValueError(
                        f"Laplacian columns for '{fault.name}' exceed the "
                        "assembled model layout"
                    )
                block = np.zeros((lap_sd.shape[0], Np), dtype=float)
                block[:, st:se] = lap_sd
                base_blocks.append(block)
                weighted_blocks.append(block * float(ipenalty_weight))

            G_lap_base = (
                np.vstack(base_blocks)
                if base_blocks
                else np.zeros((0, Np), dtype=float)
            )
            G_lap = (
                np.vstack(weighted_blocks)
                if weighted_blocks
                else np.zeros((0, Np), dtype=float)
            )
        else:
            G_lap = smoothing_matrix
            G_lap_base = None
        self.G_lap = G_lap
        # The base matrix is available only when this method constructed the
        # Laplacian. A caller-provided matrix may already contain arbitrary row
        # scaling, so inferring an unscaled matrix here would be unsafe.
        self.G_lap_base = G_lap_base

        # ----------------------------Data weight-----------------------------#
        # Whitening is applied after any DES column transformation, so each
        # selected design matrix is multiplied only once.

        # ----------------------------Set constraints using constraint manager-----------------------------#
        # Get constraints from constraint manager
        A_ueq_cm, b_ueq_cm = self.constraint_manager.get_combined_inequality_constraints()
        Aeq_cm, beq_cm = self.constraint_manager.get_combined_equality_constraints()
        
        A_ueq = A_ueq_cm
        b_ueq = b_ueq_cm
        Aeq_final = Aeq_cm
        beq_final = beq_cm

        # Get bounds from constraint manager
        lb = self.constraint_manager.lb
        ub = self.constraint_manager.ub
        if lb is None or ub is None or any(np.isnan(lb)) or any(np.isnan(ub)):
            raise ValueError("Set bounds first with update_bounds()")

        # ----------------------------Apply DES transformation if enabled-----------------------------#
        if use_des:
            if verbose:
                print("Applying Depth-Equalized Smoothing (DES) transformation...")
            
            # Get polynomial positions
            # poly_positions = get_poly_positions_from_multifaults(self)
            
            # Apply DES transformation to the original Green's function matrix G
            des_result = apply_des_transformation(
                G=G,  # Use original G matrix for DES parameter calculation
                D=G_lap,
                A_ineq=A_ueq,
                b_ineq=b_ueq,
                A_eq=Aeq_final,
                b_eq=beq_final,
                lb=lb,
                ub=ub,
                fault_indices_config=self.des_indices_config,
                mode=self.des_config.get('mode', 'per_column'),
                # groups=self.des_config.get('groups', None),
                G_norm=self.des_config.get('G_norm', 'l2'),
                depth_grouping_config=self.des_config.get('depth_grouping', None)
            )
            
            # Apply DES scaling to get G_prime
            G_prime = des_result['G_prime']
            
            # DES changes parameter columns only.  The exact transformed
            # matrices feed the same quadratic objective that the historical
            # augmented residual system would have produced.
            G_quadratic = G_prime
            D_quadratic = des_result['D_prime']
            
            # Update constraints with DES-transformed versions
            A_ueq_prime = des_result.get('A_ineq_prime', A_ueq)
            b_ueq_prime = des_result.get('b_ineq', b_ueq)
            Aeq_prime = des_result.get('A_eq_prime', Aeq_final)
            beq_prime = des_result.get('b_eq', beq_final)
            lb_prime = des_result['lb_prime']
            ub_prime = des_result['ub_prime']
            
            # Store DES information for recovery
            self.des_result = des_result
            
            if verbose:
                print(f"DES applied: {len(des_result['fault_indices'])} fault parameters scaled")
                print(f"Scaling factor range: [{des_result['scale_factors'].min():.3f}, {des_result['scale_factors'].max():.3f}]")
        else:
            # No DES transformation - retain the original parameter columns.
            G_quadratic = G
            D_quadratic = G_lap
            
            A_ueq_prime, b_ueq_prime = A_ueq, b_ueq
            Aeq_prime, beq_prime = Aeq_final, beq_final
            lb_prime, ub_prime = lb, ub

        # ----------------------------Inverse using lsqlin-----------------------------#
        # CVXOPT's historical lsqlin path immediately converted the augmented
        # residual matrix into H=A.T@A and q=-A.T@b.  Accumulate those exact
        # terms blockwise so the normal path does not materialize or copy A.
        # A scan-local cache is used only by simple_run_loop and only without
        # DES; independent run() calls always rebuild from current state.
        use_scan_cache = (
            not use_des
            and getattr(self, '_blse_quadratic_scan_cache', None) is not None
        )
        H, q = self._assemble_blse_quadratic(
            G_quadratic,
            d,
            D_quadratic,
            data_weight,
            use_scan_cache=use_scan_cache,
            penalty_weight=penalty_weight,
        )

        def residual_factory():
            return self._assemble_blse_residual_system(
                G_quadratic, d, D_quadratic, data_weight,
            )

        opts = {'show_progress': False}
        ret = lsqlin.lsqlin_quadratic_auto(
            H, q, residual_factory, 0,
            A_ueq_prime, b_ueq_prime,
            Aeq_prime, beq_prime,
            lb_prime, ub_prime,
            None, opts,
        )
        if verbose and ret.get('solver') != 'cvxopt_qp':
            print(
                "[SOLVER] CVXOPT did not converge; solved the unchanged "
                "constraint system with the robust Clarabel fallback"
            )
        _validate_lsqlin_status(ret, context="BLSE constrained least squares")
        mpost_prime = lsqlin.cvxopt_to_numpy_matrix(ret['x'])
        
        # ----------------------------Recover solution if DES was used-----------------------------#
        if use_des:
            # Recover the final solution
            mpost = recover_sf_with_poly(
                mpost_prime, 
                des_result['alpha'], 
                des_result['norm2_fault'], 
                des_result['fault_indices']
            )
        else:
            mpost = mpost_prime
        
        # Store mpost
        self.mpost = mpost
        _validate_linear_solution_constraints(
            self.mpost,
            lb,
            ub,
            A_ueq,
            b_ueq,
            Aeq_final,
            beq_final,
            context="BLSE constrained least squares",
        )

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def simple_vce(self, smoothing_matrix=None, smoothing_constraints=None, method='mudpy',
                   verbose=False, max_iter=10, tol=1e-4, des_enabled=None,
                   sigma_mode='individual', sigma_groups=None, sigma_update=None, sigma_values=None,
                   smooth_mode='single', smooth_groups=None, smooth_update=None, smooth_values=None,
                   validate_constraints=True):
        """
        Perform Simple Variance Component Estimation (VCE) for multi-fault inversion.

        This method iteratively estimates optimal weights between data fitting and
        regularization components using a simplified VCE approach with lsqlin solver.
        The penalty weights are automatically determined through VCE iterations.

        Parameters
        ----------
        smoothing_matrix : array, optional
            Pre-computed smoothing matrix. Mutually exclusive with
            ``smoothing_constraints``.
        smoothing_constraints : tuple or dict, optional
            Smoothing constraints for Laplacian construction. Mutually
            exclusive with ``smoothing_matrix``.
        method : str
            Method for building Laplacian ('mudpy')
        verbose : bool
            Enable verbose output
        max_iter : int
            Maximum VCE iterations
        tol : float
            Convergence tolerance
        des_enabled : bool, optional
            Whether to use DES (if None, uses self.des_enabled)
        sigma_mode : str
            'single', 'individual', or 'grouped' for data variance components
        sigma_groups : dict, optional
            Custom grouping for data variance components
        sigma_update : list of bool, optional
            Whether to update each sigma group (same order as sigma groups)
        sigma_values : list of float, optional
            Initial/fixed values for each sigma group (same order as sigma groups)
        smooth_mode : str
            'single', 'individual', or 'grouped' for smoothing variance components
        smooth_groups : dict, optional
            Custom grouping for smoothing variance components
        smooth_update : list of bool, optional
            Whether to update each smoothing group (same order as smoothing groups)
        smooth_values : list of float, optional
            Initial/fixed values for each smoothing group (same order as smoothing groups)

        Returns
        -------
        dict with keys:
            - 'm': estimated parameters
            - 'solved_sigma2_by_group': data variances used by ``m``
            - 'solved_alpha2_by_group': smoothing variances used by ``m``
            - 'proposed_sigma2_by_group': possible next-iteration values
            - 'proposed_alpha2_by_group': possible next-iteration values
            - 'converged': convergence flag
            - 'iterations': number of iterations
            - 'smoothing_matrix': unscaled smoothing matrix in original model coordinates
            - 'model_smoothing_matrix': the same rows with the returned model's 1/alpha multipliers
            - 'smoothing_provenance': construction method, constraints and DES context
              (``provided_matrix``, ``built_from_constraints``, or
              ``fault_laplacian``)
        """
        self.constraint_manager._require_activation_flags_reconciled(
            "BLSE/VCE"
        )

        matrix_was_provided = smoothing_matrix is not None
        constraints_were_provided = smoothing_constraints is not None
        if matrix_was_provided and constraints_were_provided:
            raise ValueError(
                "Provide either smoothing_matrix or smoothing_constraints, "
                "not both. Their precedence would otherwise be ambiguous."
            )

        # Validate constraints using constraint manager
        if validate_constraints:
            validation = self.constraint_manager.validate()
            if not validation['valid']:
                print("[X] Constraint validation failed:")
                for error in validation['errors']:
                    print(f"   Error: {error}")
                raise ValueError("Invalid constraints detected")

        if verbose:
            print("\n[RUN] Starting VCE with unified constraint management:")
            self.constraint_manager.print_summary()

        # Ensure constraints are synced
        self.constraint_manager.sync_to_solver()

        from .simple_vce import (
            assemble_weighted_smoothing_matrix,
            simplified_vce,
        )
        from .des_utils import apply_des_transformation, recover_sf_with_poly

        use_des = des_enabled if des_enabled is not None else self.des_enabled

        if verbose:
            print("="*60)
            print("Starting Simple VCE for Multi-Fault Inversion")
            print(f"Number of faults: {len(self.faults)}")
            print(f"DES enabled: {use_des}")
            print(f"Sigma mode: {sigma_mode}")
            print(f"Smooth mode: {smooth_mode}")
            print("="*60)

        # Get basic matrices
        G = self.G
        d = self.d
        if not hasattr(self, "data_covariance_metrics"):
            self.prepare_data_covariance_metrics()

        # Set bounds
        lb = self.constraint_manager.lb
        ub = self.constraint_manager.ub
        if lb is None or ub is None or any(np.isnan(lb)) or any(np.isnan(ub)):
            raise ValueError("Set bounds first with update_bounds()")

        # Reuse the same row-layout contract that prepared the covariance
        # metrics; VCE must not reconstruct a second interpretation of G/d.
        data_ranges = dict(self.data_ranges)

        # Setup fault ranges
        fault_ranges = {}
        for fault in self.faults:
            start, end = self.slip_positions[fault.name]
            fault_ranges[fault.name] = (start, end)
        smoothing_faults = [
            fault.name
            for fault in self.faults
            if (
                fault.name not in self.adapters
                or self.adapters[fault.name].supports_smoothing()
            )
        ]

        # Build smoothing matrix if not provided
        if smoothing_matrix is None:
            if verbose:
                print("Building smoothing matrix...")

            Np = G.shape[1]
            smoothing_blocks = []
            if smoothing_constraints is not None:
                if isinstance(smoothing_constraints, (tuple, list)) and len(smoothing_constraints) == 4:
                    smoothing_constraints = {
                        fault_name: tuple(smoothing_constraints)
                        for fault_name in smoothing_faults
                    }
                elif isinstance(smoothing_constraints, dict):
                    missing = set(smoothing_faults) - set(smoothing_constraints)
                    if missing:
                        raise ValueError(
                            "smoothing_constraints is missing smoothing-capable "
                            "sources: " + ", ".join(sorted(missing))
                        )
                    smoothing_constraints = dict(smoothing_constraints)
                else:
                    raise ValueError(
                        "smoothing_constraints must be a 4-tuple or a mapping"
                    )
            else:
                smoothing_constraints = {
                    fault_name: (None, None, None, None)
                    for fault_name in smoothing_faults
                }

            for fault in self.faults:
                adapter = self.adapters[fault.name]
                if not adapter.supports_smoothing():
                    continue
                ismoothing_constraints = smoothing_constraints[fault.name]
                lap = fault.buildLaplacian(
                    method=method,
                    bounds=ismoothing_constraints,
                )
                from scipy.linalg import block_diag as blkdiag
                lap_sd = blkdiag(lap, lap)
                n_components = len(adapter.get_param_names())
                if n_components == 1:
                    lap_sd = lap
                elif n_components > 2:
                    lap_sd = blkdiag(*([lap] * n_components))

                lap_sd = np.asarray(lap_sd, dtype=float)
                st = self.fault_indexes[fault.name][0]
                se = st + lap_sd.shape[1]
                if se > Np:
                    raise ValueError(
                        f"Laplacian columns for '{fault.name}' exceed the "
                        "assembled model layout"
                    )
                block = np.zeros((lap_sd.shape[0], Np), dtype=float)
                block[:, st:se] = lap_sd
                smoothing_blocks.append(block)

            smoothing_matrix = (
                np.vstack(smoothing_blocks)
                if smoothing_blocks
                else np.zeros((0, Np), dtype=float)
            )

            if verbose:
                print(f"Smoothing matrix built: {smoothing_matrix.shape}")

        # Get constraints from constraint manager
        A_ueq, b_ueq = self.constraint_manager.get_combined_inequality_constraints()
        Aeq, beq = self.constraint_manager.get_combined_equality_constraints()

        # Prepare for DES transformation if enabled
        if use_des:
            des_result = apply_des_transformation(
                G=G,  # Use original G matrix for DES parameter calculation
                D=smoothing_matrix,
                A_ineq=A_ueq,
                b_ineq=b_ueq,
                A_eq=Aeq,
                b_eq=beq,
                lb=lb,
                ub=ub,
                fault_indices_config=self.des_indices_config,
                mode=self.des_config.get('mode', 'per_column'),
                G_norm=self.des_config.get('G_norm', 'l2'),
                depth_grouping_config=self.des_config.get('depth_grouping', None)
            )

            G_vce = des_result['G_prime']
            L_vce = des_result['D_prime']
            lb_vce = des_result['lb_prime']
            ub_vce = des_result['ub_prime']
            A_ueq_vce = des_result.get('A_ineq_prime', A_ueq)
            b_ueq_vce = des_result.get('b_ineq', b_ueq)
            Aeq_vce = des_result.get('A_eq_prime', Aeq)
            beq_vce = des_result.get('b_eq', beq)
            fault_ranges_vce = fault_ranges  # Keep original for now

            self.des_result = des_result
        else:
            G_vce = G
            L_vce = smoothing_matrix
            lb_vce = lb
            ub_vce = ub
            A_ueq_vce = A_ueq
            b_ueq_vce = b_ueq
            Aeq_vce = Aeq
            beq_vce = beq
            fault_ranges_vce = fault_ranges

        # Run Simple VCE with lsqlin solver
        if verbose:
            print(f"Running Simple VCE with lsqlin solver (max_iter={max_iter}, tol={tol})...")

        vce_result = simplified_vce(
            data_metrics=self.data_covariance_metrics,
            d=d,
            G=G_vce,
            L=L_vce,
            bounds=(lb_vce, ub_vce),
            data_ranges=data_ranges,
            fault_ranges=fault_ranges_vce,
            smoothing_faults=smoothing_faults,
            sigma_mode=sigma_mode,
            sigma_groups=sigma_groups,
            sigma_update=sigma_update,
            sigma_values=sigma_values,
            smooth_mode=smooth_mode,
            smooth_groups=smooth_groups,
            smooth_update=smooth_update,
            smooth_values=smooth_values,
            A_ueq=A_ueq_vce,
            b_ueq=b_ueq_vce,
            Aeq=Aeq_vce,
            beq=beq_vce,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose
        )

        # Recover solution if DES was used
        if use_des:
            m_prime = vce_result['m']
            m_recovered = recover_sf_with_poly(
                m_prime,
                des_result['alpha'],
                des_result['norm2_fault'],
                des_result['fault_indices']
            )

            vce_result['m'] = m_recovered

        # Publish the exact regularization state associated with the returned
        # model.  ``smoothing_matrix`` is the unscaled matrix in original model
        # coordinates; ``model_smoothing_matrix`` includes the final 1/alpha
        # row multipliers.  High-level result code must consume this state
        # instead of rebuilding it from mutable fault.GL caches.
        vce_result['smoothing_matrix'] = smoothing_matrix
        vce_result['model_smoothing_matrix'] = assemble_weighted_smoothing_matrix(
            smoothing_matrix,
            fault_ranges,
            vce_result['smooth_groups'],
            vce_result['solved_alpha2_by_group'],
        )
        vce_result['smoothing_provenance'] = {
            'source': (
                'provided_matrix'
                if matrix_was_provided
                else (
                    'built_from_constraints'
                    if constraints_were_provided
                    else 'fault_laplacian'
                )
            ),
            'method': None if matrix_was_provided else method,
            'constraints': (
                smoothing_constraints if constraints_were_provided else None
            ),
            'smoothing_sources': list(smoothing_faults),
            'des_enabled': bool(use_des),
            'coordinate_space': 'original_model',
        }

        # Store results
        self.mpost = vce_result['m']
        self.vce_result = vce_result
        _validate_linear_solution_constraints(
            self.mpost,
            lb,
            ub,
            A_ueq,
            b_ueq,
            Aeq,
            beq,
            context="VCE constrained least squares",
        )

        if verbose:
            print(f"VCE completed in {vce_result['iterations']} iterations")
            print(f"Converged: {vce_result['converged']}")

        return vce_result
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def solve_with_fnnls(self, penalty_weight=1., smoothing_matrix=None, data_weight=1.,
                         smoothing_constraints=None, method='mudpy', verbose=False,
                         validate_constraints=True):
        """
        Solve the constrained least squares problem using fnnls.
    
        Parameters:
        - penalty_weight: Weight for the smoothing matrix penalty.
        - smoothing_matrix: Matrix used for smoothing (Laplacian if provided).
        - data_weight: Weight for the data.
        - smoothing_constraints: Smoothing constraints to apply during the least squares process.
        - method: Solver method to use.
        - verbose: Enable verbose output.
    
        Returns:
        - mpost: The solution vector.
        """
        # Validate constraints
        if validate_constraints:
            validation = self.constraint_manager.validate()
            if not validation['valid']:
                print("[X] Constraint validation failed:")
                for error in validation['errors']:
                    print(f"   Error: {error}")
                raise ValueError("Invalid constraints detected")

        if verbose:
            print("\n[RUN] Starting fnnls solver with unified constraint management:")
            self.constraint_manager.print_summary()

        # Ensure constraints are synced
        self.constraint_manager.sync_to_solver()

        # Get the faults
        faults = self.faults
    
        # Get the matrixes and vectors
        G = self.G
        Cd = self.Cd
        d = self.d
    
        # Nd = d.shape[0]
        Np = G.shape[1]
        Ns = 0
        Ns_st = []
        Ns_se = []
        # build Laplace
        for fault in faults:
            Ns_st.append(Ns)
            adapter = self.adapters[fault.name]
            Ns += adapter.get_n_source_params()
            Ns_se.append(Ns)
        G_lap = np.zeros((Ns, Np))
        d_lap = np.zeros((Ns, ))
    
        # ----------------------------Smoothing matrix-----------------------------#
        if smoothing_matrix is None:
            if isinstance(penalty_weight, (int, float)):
                penalty_weight = np.ones(len(faults)) * penalty_weight
            elif isinstance(penalty_weight, (list, np.ndarray)):
                assert len(penalty_weight) == len(faults), "The length of penalty_weight should be equal to the number of faults."
            else:
                raise ValueError("penalty_weight should be a scalar or a list of scalars.")
    
            # Handle smoothing constraints
            faultnames = [ifault.name for ifault in faults]
            if smoothing_constraints is not None:
                if isinstance(smoothing_constraints, tuple) and len(smoothing_constraints) == 4:
                    smoothing_constraints = {fault_name: smoothing_constraints for fault_name in faultnames}
                elif isinstance(smoothing_constraints, dict):
                    assert all(fault_name in smoothing_constraints for fault_name in faultnames), "All fault names must be in smoothing_constraints."
                else:
                    raise ValueError("smoothing_constraints should be a 4-tuple or a dictionary with fault names as keys and 4-tuples as values.")
    
            smoothing_constraints = [smoothing_constraints[ifaultname] for ifaultname in faultnames]
            for ii, (fault, ipenalty_weight, ismoothing_constraints) in enumerate(zip(faults, penalty_weight, smoothing_constraints)):
                st = self.fault_indexes[fault.name][0]
                adapter = self.adapters[fault.name]
                if adapter.supports_smoothing():
                    if fault.patchType in ('rectangle'):
                        lap = fault.buildLaplacian(method=method, bounds=ismoothing_constraints)
                    else:
                        lap = fault.buildLaplacian(method=method, bounds=ismoothing_constraints)
                    lap_sd = blkdiag(lap, lap)
                    Nsd = len(adapter.get_param_names())
                    # TODO: The following code is not clear, need to be modified
                    if Nsd == 1:
                        lap_sd = lap
                    se = st + Nsd * lap.shape[0]
                    G_lap[Ns_st[ii]:Ns_se[ii], st:se] = lap_sd * ipenalty_weight
        else:
            G_lap = np.zeros((smoothing_matrix.shape[0], Np))
            G_lap[:, :Ns] = smoothing_matrix
            d_lap = np.zeros((G_lap.shape[0], ))
        self.G_lap = G_lap
    
        G_lap2I = G_lap
    
        # ----------------------------Data weight-----------------------------#
        G_data_whitened, d_data_whitened = self._whiten_data_system(
            G,
            d,
            data_weight,
        )
        d2I = np.hstack((d_data_whitened, d_lap))
        G2I = np.vstack((G_data_whitened, G_lap2I))
    
        # ----------------------------Inverse using fnnls-----------------------------#
        # Get bounds from constraint manager
        lb = self.constraint_manager.lb
        ub = self.constraint_manager.ub
        if lb is None or ub is None or any(np.isnan(lb)) or any(np.isnan(ub)):
            raise ValueError("Set bounds first with update_bounds()")
        
        # Ensure lb and ub are numpy arrays
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        
        # Initialize masks for variables that need to be flipped
        flip_mask = (lb < 0) | (ub < 0)
        
        # Flip the necessary parts of G2I
        G2I_flipped = G2I.copy()
        G2I_flipped[:, flip_mask] = -G2I[:, flip_mask]
        
        # Use fnnls to solve the problem
        mpost_flipped, res = fnnls(G2I_flipped, d2I)
        
        # Flip the necessary parts of the solution back
        mpost = mpost_flipped.copy()
        mpost[flip_mask] = -mpost_flipped[flip_mask]
        
        # Store mpost
        self.mpost = mpost
        
        # All done
        return

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def distributem(self, verbose=False):
        '''
        After computing the m_post model, this routine distributes the m parameters to the faults.
        Uses source adapters for type-safe parameter distribution.

        Kwargs:
            * verbose   : talk to me

        Returns:
            * None
        '''

        # Get the faults
        faults = self.faults

        # Loop over the faults
        for fault in faults:

            if verbose:
                print ("---------------------------------")
                print ("---------------------------------")
                print("Distribute the slip values to fault {}".format(fault.name))

            # Store the mpost
            st = self.fault_indexes[fault.name][0]
            se = self.fault_indexes[fault.name][1]
            fault.mpost = self.mpost[st:se]

            # Transformation object
            if fault.type=='transformation':
                
                # Distribute simply
                fault.distributem()

            # Use adapter for Fault/Pressure/Sbarbot
            elif fault.name in self.adapters:
                adapter = self.adapters[fault.name]

                # Affect the indexes (only for Fault type from CSI parent)
                if adapter.source_type == 'Fault':
                    self.affectIndexParameters(fault)

                # Distribute source parameters using adapter
                n_source = adapter.get_n_source_params()
                adapter.distribute_results(fault.mpost[:n_source])
                st = n_source

                # Handle custom parameters (any source with NumberCustom)
                if hasattr(fault, 'NumberCustom') and fault.NumberCustom > 0:
                    fault.custom = {}
                    for dset in fault.datanames:
                        if 'custom' in fault.G[dset].keys():
                            nc = fault.G[dset]['custom'].shape[1]
                            se = st + nc
                            fault.custom[dset] = fault.mpost[st:se]
                            st += nc

                # Get the polynomial/orbital/helmert values if they exist
                if hasattr(fault, 'poly'):
                    fault.polysol = {}
                    fault.polysolindex = {}
                    for dset in fault.datanames:
                        if dset in fault.poly.keys():
                            if (fault.poly[dset] is None):
                                fault.polysol[dset] = None
                            else:

                                if (fault.poly[dset].__class__ is not str) and (fault.poly[dset].__class__ is not list):
                                    if (fault.poly[dset] > 0):
                                        se = st + fault.poly[dset]
                                        fault.polysol[dset] = fault.mpost[st:se]
                                        fault.polysolindex[dset] = range(st,se)
                                        st += fault.poly[dset]
                                elif (fault.poly[dset].__class__ is str):
                                    if fault.poly[dset] == 'full':
                                        nh = fault.helmert[dset]
                                        se = st + nh
                                        fault.polysol[dset] = fault.mpost[st:se]
                                        fault.polysolindex[dset] = range(st,se)
                                        st += nh
                                    if fault.poly[dset] in ('strain', 'strainnorotation', 'strainonly', 'strainnotranslation', 'translation', 'translationrotation'):
                                        nh = fault.strain[dset]
                                        se = st + nh
                                        fault.polysol[dset] = fault.mpost[st:se]
                                        fault.polysolindex[dset] = range(st,se)
                                        st += nh
                                    # Added by kfhe, at 10/12/2021
                                    if fault.poly[dset] == 'eulerrotation':
                                        nh = fault.eulerrot[dset]
                                        se = st + nh
                                        fault.polysol[dset] = fault.mpost[st:se]
                                        fault.polysolindex[dset] = range(st,se)
                                        st += nh
                                    if fault.poly[dset] == 'internalstrain':
                                        nh = fault.intstrain[dset]
                                        se = st + nh
                                        fault.polysol[dset] = fault.mpost[st:se]
                                        fault.polysolindex[dset] = range(st,se)
                                        st += nh
                                elif (fault.poly[dset].__class__ is list):
                                    nh = fault.transformation[dset]
                                    se = st + nh
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += nh

        # All done
        return
    # ----------------------------------------------------------------------

if __name__ == "__main__":
    solver = multifaultsolve_boundLSE()
    # 设置边界约束
    solver.update_bounds(lb=-10, ub=10)
    solver.update_bounds(strikeslip_bounds={'main_fault': (-5, 5)})
    
    # 运行VCE - 每个数据集和断层都有独立的方差分量
    result = solver.simple_vce(
        sigma_mode='individual',    # 每个数据集独立sigma
        smooth_mode='individual',   # 每个断层独立alpha
        verbose=True
    )
    
    # 或者使用分组模式
    result = solver.simple_vce(
        sigma_mode='grouped',
        sigma_groups={'sar': ['insar1', 'insar2'], 'gnss': ['gps']},
        smooth_mode='grouped', 
        smooth_groups={'main': ['main_fault'], 'secondary': ['branch_fault', 'background']},
        verbose=True
    )
    
    # 分发结果
    solver.distributem()

# EOF
