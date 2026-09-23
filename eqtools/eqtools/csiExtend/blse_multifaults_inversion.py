import scipy
import numpy as np
import copy
import json
import os
import matplotlib.pyplot as plt
import pandas as pd

from .multifaults_base import MyMultiFaultsInversion
from .config.blse_config import BoundLSEInversionConfig
from .config.parameter_groups import attach_group_parameters, resolve_group_layout
from .data_correction_constraints import DataCorrectionConstraintMixin
from .data_correction_report_mixin import DataCorrectionReportMixin
from .deep_slip_loading_mixin import DeepSlipLoadingMixin
from .data_prediction import get_prediction_result_state_attributes
from .blse_diagnostics import plot_blse_roughness_rms
from .interseismic_mixin import InterseismicKinematicsMixin
from .plot_product_mixin import FigureProductMixin
from .patch_indices import normalize_patch_indices
from .fit_statistics import build_vce_component_rows
from .hyperparameter_reporting import (
    build_fixed_scale_parameter_rows,
    format_scale_parameter_report,
)
from .parameter_layout_reporting import (
    build_active_scale_layout_rows,
    build_linear_layout_rows,
    build_scale_layout_rows,
    describe_data_correction_blocks,
    format_parameter_layout_report,
    make_parameter_layout_report,
)
from .config.config_utils import get_observation_unit_info, parse_observation_unit
from ..viztools import normalize_image_format


_PENALTY_SCAN_SUMMARY_COLUMNS = (
    'candidate_index',
    'penalty_weight',
    'equivalent_log10_alpha',
    'resolved_penalty_order',
    'resolved_penalty_weights',
    'roughness',
    'global_rms',
    'global_vr_percent',
    'observation_unit',
)

_PENALTY_SCAN_FIT_COLUMNS = (
    'candidate_index',
    'penalty_weight',
    'observation_unit',
    'scope',
    'model',
    'dataset',
    'data_type',
    'vertical',
    'poly',
    'sigma_group',
    'rms',
    'vr',
    'ss_res',
    'ss_obs',
    'n_observations',
)

_PENALTY_SCAN_WEIGHTED_COLUMNS = (
    'sigma_scale',
    'base_marginal_std',
    'effective_marginal_std',
    'weighted_quadratic',
    'weighted_rms',
    'weighted_effective_dof',
    'reduced_weighted_misfit',
)

class BoundLSEMultiFaultsInversion(
    DataCorrectionReportMixin,
    DataCorrectionConstraintMixin,
    DeepSlipLoadingMixin,
    InterseismicKinematicsMixin,
    FigureProductMixin,
    MyMultiFaultsInversion,
):
    # A penalty scan is a diagnostic transaction: candidate solves may
    # temporarily replace these active-result attributes, but the object must
    # leave the loop in exactly the state in which it entered.  Matrix entries
    # are assigned, not mutated, by ``run()``, so retaining their references is
    # sufficient; source result arrays are copied separately below because
    # ``distributem()`` updates them in place.
    _LINEAR_RESULT_STATE_ATTRIBUTES = (
        'mpost',
        'G_lap',
        'G_lap_base',
        'des_result',
        'GL_combined_poly',
        'current_smoothing_matrix',
        'current_model_smoothing_matrix',
        'current_smoothing_provenance',
        'current_penalty_weight',
        'current_data_sigmas',
        'current_data_weights',
        'current_data_sigma_groups',
        'current_data_sigma_group_members',
        'current_data_effective_dof',
        'current_scale_parameter_rows',
        'current_scale_parameter_method',
        'current_scale_parameter_title',
    )
    def __init__(self, name, faults_list, geodata=None, config='default_config_BLSE.yml', encoding='utf-8',
                 gfmethods=None, bounds_config='bounds_config.yml', interseismic_config=None,
                 rake_limits=None, extra_parameters=None, verbose=True, des_enabled=False, des_config=None):
        """
        Initialize BoundLSEMultiFaultsInversion with DES support.
        
        Parameters:
        -----------
        name : str
            Name of the inversion
        faults_list : list
            List of fault objects
        geodata : object, optional
            Geodetic data object
        config : str or object, optional
            Configuration file path or config object (default: 'default_config_BLSE.yml')
        encoding : str, optional
            File encoding (default: 'utf-8')
        gfmethods : sequence of str, optional
            One Green's-function method per entry in ``faults_list``. This is
            a runtime method override; method-specific options remain in the
            configuration's ``method_parameters.update_GFs`` block.
        bounds_config : str, optional
            Bounds configuration file (default: 'bounds_config.yml')
        interseismic_config : str or dict, optional
            Interseismic block-motion and optional cap/backslip constraint
            configuration.  If omitted, ``config.interseismic_config_file`` is
            used when present.
        rake_limits : dict, optional
            Rake angle limits
        extra_parameters : dict, optional
            Additional parameters for the solver
        verbose : bool, optional
            Enable verbose output (default: True)
        des_enabled : bool, optional
            Whether to enable Depth-Equalized Smoothing (DES) (default: False)
        des_config : dict, optional
            DES configuration parameters (default: None)
        """
        # Initialize the faults ahead of the configuration
        self.faults = faults_list
        self.faults_dict = {fault.name: fault for fault in self.faults}
        # To order the G matrix based on the order of the faults
        self.faultnames = [fault.name for fault in self.faults]
        
        # Initialize BoundLSEInversionConfig first
        if isinstance(config, str):
            assert geodata is not None, "geodata must be provided when config is a file"
            self.config = BoundLSEInversionConfig(config, multifaults=None, 
                                                  geodata=geodata, 
                                                  faults_list=faults_list, 
                                                  gfmethods=gfmethods, 
                                                  encoding=encoding,
                                                  verbose=verbose)
        else:
            self.config = config

        reporter = getattr(self.config, 'report_config_diagnostics', None)
        if reporter is not None:
            reporter(enabled=verbose)

        if interseismic_config is None:
            interseismic_config = getattr(self.config, 'interseismic_config_file', None)
        if interseismic_config is not None:
            self.config.load_interseismic_config(interseismic_config, encoding=encoding)

        # Initialize MyMultiFaultsInversion with DES support
        super(BoundLSEMultiFaultsInversion, self).__init__(name, 
                                                           faults_list, 
                                                           extra_parameters=extra_parameters, 
                                                           verbose=verbose,
                                                           des_enabled=des_enabled,
                                                           des_config=des_config)

        self.assembleGFs()
        
        self.update_config(self.config)

        # DES (Depth-Equalized Smoothing) parameters
        des_from_config = getattr(self.config, 'des', {'enabled': False})
        des_config = des_config if des_config is not None else des_from_config
        self.des_enabled = des_enabled or des_config.get('enabled', False)
        self.des_config = des_config if des_config is not None else {
            'mode': 'per_patch',
            'G_norm': 'l2',
            'depth_grouping': {
                'strategy': 'uniform',
                'interval': 1.0
                }
        }
        
        # Constructor rake limits are persistent runtime coarse declarations,
        # not a transient fifth precedence layer.
        if rake_limits:
            self.constraint_manager.update_fault_rake_limits(
                rake_limits,
                replace=False,
                source='constructor',
                sync=False,
            )

        # Apply all constraints using the constraint manager.
        self.constraint_manager._apply_constraint_config(
            bounds_config_file=bounds_config,
            encoding=encoding
        )
        
        # Sync constraints to solver for backward compatibility
        self.constraint_manager.sync_to_solver()

    def update_config(self, config):
        self.config = config
        if hasattr(self, 'constraint_manager'):
            self.constraint_manager.config = config
        self._update_faults()

    def update_interseismic_config(self, interseismic_config, reapply=True):
        """Load a new interseismic config and optionally rebuild its constraints."""
        with self.constraint_transaction():
            parsed = self.config.load_interseismic_config(interseismic_config)
            if reapply:
                self.constraint_manager._remove_groups_by_owner(
                    'config',
                    families={
                        'interseismic_blocks',
                        'interseismic_cap',
                        'interseismic_backslip',
                    },
                )
                if parsed.get('blocks', {}).get('enabled', False):
                    self.constraint_manager.apply_interseismic_block_constraints()
                self.constraint_manager.apply_euler_cap_constraints()
                self.constraint_manager.apply_interseismic_backslip_constraints()
                self.constraint_manager.sync_to_solver()
            return parsed

    def update_euler_cap_constraint(
        self,
        fault_name,
        *,
        selector=None,
        max_coupling=None,
        mode=None,
        min_loading_abs=None,
        enabled=None,
        reapply=True,
    ):
        """Update optional cap-constraint selector for one fault.

        Parameters
        ----------
        fault_name : str
            Fault whose cap constraint should be updated.
        selector : dict or iterable of int, optional
            Patch selector for cap rows only.  It does not affect tectonic
            loading-rate calculation.
        max_coupling : float, optional
            Upper multiplier ``k`` in ``|backslip| <= k * |loading|``.
            Defaults to the value already stored in config, or 1.0.
        mode : {"motion_sense", "loading_sign"}, optional
            Cap construction mode.  ``motion_sense`` is the default and works
            with estimated Euler loading.  ``loading_sign`` requires fixed
            loading and constrains ``0 <= -q / b <= k`` from the projected sign.
        min_loading_abs : float, optional
            Minimum absolute loading accepted by ``mode="loading_sign"``.
        enabled : bool, optional
            If provided, update ``cap_constraints.enabled``.
        reapply : bool, default True
            If True, rebuild the cap constraint matrix after updating config.

        Returns
        -------
        dict
            Current parsed ``config.interseismic_config`` dictionary.
        """
        if fault_name not in self.faults_dict:
            raise ValueError(
                f"Fault '{fault_name}' not found. Available: {list(self.faults_dict.keys())}"
            )

        interseismic = copy.deepcopy(getattr(self.config, 'interseismic_config', {}))
        cap = interseismic.setdefault('cap_constraints', {})
        if enabled is not None:
            cap['enabled'] = bool(enabled)
        cap.setdefault('faults', {})
        cap['faults'].setdefault(fault_name, {})
        if selector is not None:
            if isinstance(selector, (list, tuple, np.ndarray)):
                indices = normalize_patch_indices(
                    self.faults_dict[fault_name],
                    selector,
                    allow_none_all=False,
                    unique=True,
                    name=f"cap selector for fault '{fault_name}'",
                )
                selector = {'patches': indices.tolist()}
            cap['faults'][fault_name]['selector'] = selector
        if max_coupling is not None:
            max_coupling = float(max_coupling)
            if max_coupling < 0.0:
                raise ValueError("max_coupling must be non-negative")
            cap['faults'][fault_name]['max_coupling'] = max_coupling
        if mode is not None:
            cap['faults'][fault_name]['mode'] = str(mode)
        if min_loading_abs is not None:
            min_loading_abs = float(min_loading_abs)
            if min_loading_abs < 0.0:
                raise ValueError("min_loading_abs must be non-negative")
            cap['faults'][fault_name]['min_loading_abs'] = min_loading_abs
        return self.update_interseismic_config(interseismic, reapply=reapply)

    def _update_faults(self):
        # Update the faults based on the configuration parameters and method parameters for each fault 
        datanames = [d.name for d in self.config.geodata.get('data', [])]
        Nd = len(datanames)
        faultnames = self.faultnames
        for fault_name, fault_config in self.config.faults.items():
            if fault_name != 'defaults':
                # Update Green's functions
                dataFaults = self.config.dataFaults
                # Check if dataFaults is a list of lists, each equal to faultnames
                if not (isinstance(dataFaults, list) and len(dataFaults) == Nd and all(fault_name in flist for flist in dataFaults)):
                    self.update_GFs(fault_names=[fault_name], **fault_config['method_parameters']['update_GFs'])
                # Update Laplacian
                self.update_Laplacian(fault_names=[fault_name], **fault_config['method_parameters']['update_Laplacian'])

    @staticmethod
    def _resolve_result_report(report, *, verbose):
        """Resolve the shared BLSE/VCE final-report policy."""

        resolved = ('compact' if verbose else 'none') if report is None else str(report).lower()
        if resolved not in {'none', 'compact', 'full'}:
            raise ValueError("report must be one of: None, 'none', 'compact', 'full'")
        return resolved

    def _clear_scale_parameter_context(self):
        """Invalidate scale rows whenever no successful active solve exists."""

        self.current_scale_parameter_rows = []
        self.current_scale_parameter_method = None
        self.current_scale_parameter_title = None

    def _publish_scale_parameter_context(self, *, rows, method, title):
        """Freeze report-only rows already bound to a successful active solve."""

        self.current_scale_parameter_rows = copy.deepcopy(list(rows))
        self.current_scale_parameter_method = str(method)
        self.current_scale_parameter_title = str(title)

    def collect_scale_parameters(self):
        """Return a detached copy of the active BLSE/VCE scale rows."""

        return copy.deepcopy(getattr(self, 'current_scale_parameter_rows', []))

    def format_scale_parameters(self):
        """Format scales from the active result without re-reading config."""

        rows = self.collect_scale_parameters()
        method = getattr(self, 'current_scale_parameter_method', None)
        title = getattr(self, 'current_scale_parameter_title', None)
        if method == 'VCE':
            return format_scale_parameter_report(
                rows,
                title=title or 'VCE variance components',
                show_index=False,
                show_value_source=True,
                show_sampling_space=False,
                show_posterior_uncertainty=False,
                show_variance=True,
                show_diagnostics=True,
            )
        return format_scale_parameter_report(
            rows,
            title=title or 'BLSE scale parameters',
            show_index=False,
            show_value_source=True,
            show_sampling_space=False,
            show_posterior_uncertainty=False,
        )

    def print_scale_parameters(self):
        """Print the active BLSE/VCE scale table."""

        print(self.format_scale_parameters())

    def collect_parameter_layout(self):
        """Return the validated BLSE/VCE parameter layout as detached rows.

        The report is structural.  Linear rows come from the active constraint
        manager.  Before a solve, sigma/alpha rows come from canonical config
        layouts; after a solve, their group membership and estimated/fixed
        state come from the frozen active result.  Numerical scale values
        remain the responsibility of :meth:`collect_scale_parameters`.
        """

        linear_layout = self.get_linear_parameter_layout()
        sources = {source.name: source for source in self.faults}
        poly_descriptions = {}
        for block in linear_layout.get('blocks', ()):
            if block.get('role') != 'data_correction':
                continue
            source = sources[str(block['source'])]
            poly_descriptions[source.name] = describe_data_correction_blocks(
                self,
                source,
                start=int(block['start']),
                stop=int(block['stop']),
            )

        observation_unit = get_observation_unit_info(self.config)['observation']
        rows = build_linear_layout_rows(
            layout=linear_layout,
            adapters=self.adapters,
            observation_unit=observation_unit,
            use='solve',
            space='L',
            poly_descriptions=poly_descriptions,
        )

        result_method = getattr(self, 'current_scale_parameter_method', None)
        if result_method == 'VCE':
            mode = 'VCE'
            active_use = 'estimate'
        elif result_method == 'BLSE':
            mode = 'BLSE'
            active_use = 'fixed'
        else:
            mode = 'BLSE/VCE'
            active_use = 'configured'

        if result_method is None:
            rows.extend(
                build_scale_layout_rows(
                    kind='sigma',
                    layout=self.config.sigmas.get('group_layout', {}),
                    sample_slice=None,
                    sampled_use=active_use,
                    fixed_use='fixed',
                    log_scaled=bool(self.config.sigmas.get('log_scaled', False)),
                )
            )
            if self.config.alpha_enabled:
                rows.extend(
                    build_scale_layout_rows(
                        kind='alpha',
                        layout=self.config.alpha.get('group_layout', {}),
                        sample_slice=None,
                        sampled_use=active_use,
                        fixed_use='fixed',
                        log_scaled=bool(self.config.alpha.get('log_scaled', False)),
                    )
                )
        else:
            active_scale_rows = self.collect_scale_parameters()
            rows.extend(
                build_active_scale_layout_rows(
                    kind='sigma',
                    result_rows=active_scale_rows,
                    log_scaled=bool(self.config.sigmas.get('log_scaled', False)),
                )
            )
            rows.extend(
                build_active_scale_layout_rows(
                    kind='alpha',
                    result_rows=active_scale_rows,
                    log_scaled=bool(self.config.alpha.get('log_scaled', False)),
                )
            )

        return make_parameter_layout_report(
            mode=mode,
            spaces=[
                {
                    'key': 'L',
                    'name': 'linear model vector',
                    'width': int(linear_layout['width']),
                    'global_offset': 0,
                }
            ],
            rows=rows,
            notes=[
                'sigma/alpha are scale controls, not columns of the linear model vector',
                (
                    'pre-solve scale rows show configured update eligibility; '
                    'the solved BLSE/VCE scale report is authoritative for active values'
                    if result_method is None else
                    'active scale values are reported separately by print_scale_parameters()'
                ),
            ],
        )

    def format_parameter_layout(self, *, summary_only=False):
        """Format the current BLSE/VCE structural parameter layout."""

        return format_parameter_layout_report(
            self.collect_parameter_layout(), summary_only=summary_only
        )

    def print_parameter_positions(self):
        """Print the compact BLSE/VCE parameter layout."""

        print(self.format_parameter_layout())

    @staticmethod
    def _build_vce_scale_rows(result):
        """Adapt a solved VCE result to the shared read-only row contract."""

        return build_vce_component_rows(result)

    def run(self, penalty_weight=None, smoothing_constraints=None, data_weight=None, data_log_scaled=None,
            penalty_log_scaled=None, sigma=None, alpha=None, verbose=True, des_enabled=None,
            report=None):
        """
        Start the boundary-constrained least squares process.
    
        Parameters:
        -----------
        penalty_weight : int, float, list, or np.ndarray, optional
            Direct smoothing-row multipliers. They use ``1/s`` coordinates
            and therefore imply physical alpha ``s = 1 / penalty_weight``.
        smoothing_constraints : tuple or dict, optional
            Smoothing constraints to apply during the least squares process. If None, the function will use the combined Green's functions matrix.
            If a tuple, it should be a 4-tuple. If a dict, the keys should be fault names and the values should be 4-tuples.
            (top, bottom, left, right) for the smoothing constraints.
        data_weight : np.ndarray, optional
            Direct data-row multipliers in ``1/s`` coordinates. They imply
            physical sigma ``s = 1 / data_weight``.
        data_log_scaled : bool, optional
            Coordinate of ``sigma`` only: ``False`` means physical ``s`` and
            ``True`` means ``log10(s)``. It does not transform ``data_weight``.
        penalty_log_scaled : bool, optional
            Coordinate of ``alpha`` only: ``False`` means physical ``s`` and
            ``True`` means ``log10(s)``. It does not transform
            ``penalty_weight``.
        sigma : np.ndarray, optional
            Data standard-deviation scales in the coordinate selected by
            ``data_log_scaled`` (or ``sigmas.log_scaled`` when omitted).
        alpha : np.ndarray, optional
            Smoothing scales in the coordinate selected by
            ``penalty_log_scaled`` (or ``alpha.log_scaled`` when omitted).
        verbose : bool, optional
            Whether to print the results of the inversion. Default is True.
        des_enabled : bool, optional
            Whether to use Depth-Equalized Smoothing (DES). If None, uses self.des_enabled.
        report : {None, 'none', 'compact', 'full'}, optional
            Final reporting policy. ``None`` selects ``'compact'`` when
            ``verbose=True`` and ``'none'`` otherwise. ``'compact'`` prints
            the exact active sigma/alpha scales and row multipliers;
            ``'full'`` additionally prints fit statistics and, when an active
            unweighted smoothing matrix exists, model regularization.
    
        Returns:
        --------
        None
        """
        from .config.config_utils import parse_initial_values

        self._clear_fit_weight_context()
        self._clear_scale_parameter_context()
        resolved_report = self._resolve_result_report(report, verbose=verbose)

        # Ensure data_weight and sigma are either both None or only one is provided
        if (data_weight is not None) and (sigma is not None):
            raise ValueError("data_weight and sigma must either both be None or only one is provided.")
    
        # Ensure penalty_weight and alpha are either both None or only one is provided
        if (penalty_weight is not None) and (alpha is not None):
            raise ValueError("penalty_weight and alpha must either both be None or only one is provided.")
    
        # Handle data weights
        sigma_layout = self.config.sigmas['group_layout']
        n_datasets = int(sigma_layout['total_params'])
        if self.config.sigmas['mode'] == 'single':
            data_names = ['All_data']
        elif self.config.sigmas['mode'] == 'individual':
            data_names = list(sigma_layout['member_names'])
        elif self.config.sigmas['mode'] == 'grouped':
            data_names = list(sigma_layout['group_names'])
        data_indices = np.asarray(sigma_layout['member_param_indices'], dtype=int)
        legacy_data_indices = np.asarray(
            self.config.sigmas['dataset_param_indices'], dtype=int
        )
        if not np.array_equal(data_indices, legacy_data_indices):
            raise ValueError(
                "sigmas.dataset_param_indices is inconsistent with the canonical "
                "group_layout; rebuild the inversion configuration"
            )
        if data_weight is None:
            if sigma is None:
                sigma = self.config.sigmas['initial_value']
                sigma_source = 'config sigmas.initial_value'
            else:
                sigma = parse_initial_values({'initial_value': sigma},
                                                n_datasets=n_datasets,
                                                param_name='initial_value',  # initial_value or 'values'
                                                dataset_names=data_names,
                                                print_name='sigma',
                                                reject_unknown_keys=True)
                sigma_source = 'run sigma'
            sigma = np.array(sigma)
            if data_log_scaled is None:
                data_log_scaled = self.config.sigmas['log_scaled']
            sigma_source += ' [log10(s)]' if data_log_scaled else ' [s]'
            if data_log_scaled:
                sigma = np.power(10, sigma)
            data_weight = 1.0 / sigma
        else:
            wgt_dict = {'initial_value': data_weight}
            data_weight = parse_initial_values(wgt_dict, n_datasets=n_datasets,
                                                param_name='initial_value',  # initial_value or 'values'
                                                dataset_names=data_names,
                                                print_name='data_weight',
                                                reject_unknown_keys=True)
            data_weight = np.array(data_weight)
            sigma_source = 'run data_weight [1/s]'
        data_group_weights = np.asarray(data_weight, dtype=float).reshape(-1)
        sigma_group_names = list(sigma_layout['group_names'])
        if data_group_weights.size != len(sigma_group_names):
            raise ValueError(
                "Resolved data weights do not match the canonical sigma group count"
            )
        sigma_group_members = sigma_layout['members_by_group']
        sigma_rows = build_fixed_scale_parameter_rows(
            kind='sigma',
            layout=sigma_layout,
            row_multipliers_by_group=dict(zip(sigma_group_names, data_group_weights)),
            value_source_by_group={name: sigma_source for name in sigma_group_names},
        )
        data_weight = data_group_weights[data_indices]

        # Handle penalty weights
        # If alpha smoothing is disabled, use uniform weight (no regularization penalty)
        if not self.config.alpha_enabled:
            penalty_weight = np.ones(len(self.faults))
            self.current_penalty_weight = penalty_weight
            # Alpha disabled: use empty smoothing matrix, ignore smoothing_constraints
            self.combine_GL_poly(penalty_weight=penalty_weight)
            base_smoothing_matrix = np.asarray(
                self.GL_combined_poly, dtype=float
            ).copy()
            scan_cache = getattr(self, '_blse_quadratic_scan_cache', None)
            if scan_cache is not None:
                scan_cache['base_smoothing_matrix'] = (
                    base_smoothing_matrix.copy()
                )
            self.ConstrainedLeastSquareSoln(penalty_weight=penalty_weight,
                                            smoothing_matrix=self.GL_combined_poly,
                                            data_weight=data_weight,
                                            des_enabled=des_enabled,
                                            verbose=verbose)
            self.current_model_smoothing_matrix = np.asarray(
                self.G_lap, dtype=float
            )
            self.current_smoothing_matrix = base_smoothing_matrix
            self.current_smoothing_provenance = {
                'source': 'disabled',
                'coordinate_space': 'original_model',
            }
            self.distributem()
            self._publish_data_weight_context(
                data_weights=data_weight,
                group_members=sigma_group_members,
            )
            self._publish_scale_parameter_context(
                rows=sigma_rows,
                method='BLSE',
                title='BLSE scale parameters',
            )
            if resolved_report in {'compact', 'full'}:
                print('\n' + self.format_scale_parameters())
            if resolved_report == 'full':
                self.calculate_and_print_fit_statistics()
                self._print_model_regularization()
            return
        else:
            alpha_layout = self.config.alpha['group_layout']
            n_faults = int(alpha_layout['total_params'])
            if self.config.alpha['mode'] == 'single':
                fault_names = ['All_faults']
            elif self.config.alpha['mode'] == 'individual':
                fault_names = list(alpha_layout['member_names'])
            elif self.config.alpha['mode'] == 'grouped':
                fault_names = list(alpha_layout['group_names'])
            fault_indices = np.asarray(
                self.config.alpha['fault_param_indices'], dtype=int
            )
            smoothing_names = list(alpha_layout['member_names'])
            smoothing_positions = [self.faultnames.index(name) for name in smoothing_names]
            if not np.array_equal(
                fault_indices[smoothing_positions],
                np.asarray(alpha_layout['member_param_indices'], dtype=int),
            ):
                raise ValueError(
                    "alpha.fault_param_indices is inconsistent with the canonical "
                    "group_layout; rebuild the inversion configuration"
                )

            if penalty_weight is None:
                if alpha is None:
                    alpha = self.config.alpha['initial_value']
                    alpha_source = 'config alpha.initial_value'
                else:
                    alpha = parse_initial_values({'initial_value': alpha},
                                                    n_datasets=n_faults,
                                                    param_name='initial_value',  # initial_value or 'values'
                                                    dataset_names=fault_names,
                                                    print_name='alpha',
                                                    reject_unknown_keys=True)
                    alpha_source = 'run alpha'
                alpha = np.array(alpha)
                if penalty_log_scaled is None:
                    penalty_log_scaled = self.config.alpha['log_scaled']
                alpha_source += ' [log10(s)]' if penalty_log_scaled else ' [s]'
                if penalty_log_scaled:
                    alpha = np.power(10, alpha)
                penalty_weight = 1.0 / alpha
            else:
                penalty_weight = parse_initial_values({'initial_value': penalty_weight},
                                                      n_datasets=n_faults,
                                                      param_name='initial_value',  # initial_value or 'values'
                                                      dataset_names=fault_names,
                                                      print_name='penalty_weight',
                                                      reject_unknown_keys=True)
                penalty_weight = np.array(penalty_weight)
                alpha_source = 'run penalty_weight [1/s]'
            alpha_group_weights = np.asarray(penalty_weight, dtype=float).reshape(-1)
            alpha_group_names = list(alpha_layout['group_names'])
            if alpha_group_weights.size != len(alpha_group_names):
                raise ValueError(
                    "Resolved penalty weights do not match the canonical alpha group count"
                )
            alpha_rows = build_fixed_scale_parameter_rows(
                kind='alpha',
                layout=alpha_layout,
                row_multipliers_by_group=dict(
                    zip(alpha_group_names, alpha_group_weights)
                ),
                value_source_by_group={name: alpha_source for name in alpha_group_names},
            )
            penalty_weight = alpha_group_weights[fault_indices]

            self.current_penalty_weight = penalty_weight
        # Handle smoothing constraints
        if smoothing_constraints is not None:
            if isinstance(smoothing_constraints, (tuple, list)) and len(smoothing_constraints) == 4:
                smoothing_constraints = {fault_name: smoothing_constraints for fault_name in self.faultnames}
            elif isinstance(smoothing_constraints, dict):
                assert all(fault_name in smoothing_constraints for fault_name in self.faultnames), "All fault names must be in smoothing_constraints."
            else:
                raise ValueError("smoothing_constraints should be a 4-tuple or a dictionary with fault names as keys and 4-tuples as values.")
    
        if smoothing_constraints is not None:
            self.ConstrainedLeastSquareSoln(penalty_weight=penalty_weight, 
                                            smoothing_constraints=smoothing_constraints, 
                                            data_weight=data_weight,
                                            des_enabled=des_enabled,
                                            verbose=verbose)
            base_smoothing_matrix = np.asarray(
                self.G_lap_base, dtype=float
            )
        else:
            self.combine_GL_poly(penalty_weight=1.0)
            base_smoothing_matrix = np.asarray(
                self.GL_combined_poly, dtype=float
            ).copy()
            scan_cache = getattr(self, '_blse_quadratic_scan_cache', None)
            if scan_cache is not None:
                cached_base = scan_cache.get('base_smoothing_matrix')
                if cached_base is None:
                    scan_cache['base_smoothing_matrix'] = (
                        base_smoothing_matrix.copy()
                    )
                elif not np.array_equal(cached_base, base_smoothing_matrix):
                    # A fixed-geometry scan must keep one unweighted L0.  If a
                    # caller changes it during the transaction, invalidate the
                    # smoothing Gram and continue from the current exact base.
                    scan_cache['base_smoothing_matrix'] = (
                        base_smoothing_matrix.copy()
                    )
                    scan_cache.pop('S_base', None)
            self.combine_GL_poly(penalty_weight=penalty_weight)
            self.ConstrainedLeastSquareSoln(penalty_weight=penalty_weight, 
                                            smoothing_matrix=self.GL_combined_poly,
                                            data_weight=data_weight,
                                            des_enabled=des_enabled,
                                            verbose=verbose)
        self.current_model_smoothing_matrix = np.asarray(
            self.G_lap, dtype=float
        )
        self.current_smoothing_matrix = base_smoothing_matrix
        self.current_smoothing_provenance = {
            'source': (
                'built_from_constraints'
                if smoothing_constraints is not None
                else 'fault_laplacian'
            ),
            'coordinate_space': 'original_model',
        }
        self.distributem()
        self._publish_data_weight_context(
            data_weights=data_weight,
            group_members=sigma_group_members,
        )
        self._publish_scale_parameter_context(
            rows=[*sigma_rows, *alpha_rows],
            method='BLSE',
            title='BLSE scale parameters',
        )
        if resolved_report in {'compact', 'full'}:
            print('\n' + self.format_scale_parameters())
        if resolved_report == 'full':
            self.calculate_and_print_fit_statistics()
            self._print_model_regularization()

    @staticmethod
    def _resolve_vce_component_contract(
        *,
        component_name,
        configured,
        member_names,
        mode,
        groups,
        update,
        values,
        member_label,
        individual_prefix,
    ):
        """Resolve one coherent VCE component override or reuse its config.

        ``mode``, membership, values, and update flags jointly define a
        parameter layout.  Mixing only part of a runtime override with the
        loaded layout is rejected because equal-length arrays can otherwise be
        assigned to the wrong scientific group without a shape error.
        """

        runtime_values = (mode, groups, update, values)
        if all(value is None for value in runtime_values):
            contract = configured.get("group_layout")
            if contract is None:
                # Compatibility for externally constructed config-like objects
                # that predate group_layout.  Normal package configs always
                # take the branch above and therefore normalize only once.
                configured_mode = configured.get("mode", "single")
                configured_groups = (
                    configured.get("groups")
                    if configured_mode == "grouped" else None
                )
                if (
                    configured_groups is None
                    and configured_mode == "grouped"
                    and component_name == "smooth"
                    and configured.get("faults") is not None
                ):
                    raise ValueError(
                        "Grouped smoothing no longer accepts the anonymous "
                        "alpha.faults list. Define named alpha.groups and "
                        "rebuild the inversion configuration."
                    )
                layout = resolve_group_layout(
                    member_names,
                    configured_mode,
                    configured_groups,
                    member_label=member_label,
                    single_group_name="all",
                    individual_prefix=individual_prefix,
                )
                contract = attach_group_parameters(
                    layout,
                    values=configured.get("initial_value", 0.0),
                    update=configured.get("update", True),
                    value_name=f"{component_name}_values",
                    default_value=0.0,
                )
            return contract

        missing = []
        if mode is None:
            missing.append(f"{component_name}_mode")
        if update is None:
            missing.append(f"{component_name}_update")
        if values is None:
            missing.append(f"{component_name}_values")
        if missing:
            raise ValueError(
                f"Runtime {component_name} override is incomplete; provide "
                + ", ".join(missing)
                + " together with the other component fields."
            )
        mode = str(mode).lower()
        if mode == "grouped" and groups is None:
            raise ValueError(
                f"{component_name}_groups is required when "
                f"{component_name}_mode='grouped'"
            )
        if mode != "grouped" and groups is not None:
            raise ValueError(
                f"{component_name}_groups is only valid when "
                f"{component_name}_mode='grouped'"
            )

        layout = resolve_group_layout(
            member_names,
            mode,
            groups,
            member_label=member_label,
            single_group_name="all",
            individual_prefix=individual_prefix,
        )
        return attach_group_parameters(
            layout,
            values=values,
            update=update,
            value_name=f"{component_name}_values",
            default_value=0.0,
        )

    def run_simple_vce(self, smoothing_constraints=None, verbose=True, max_iter=20, tol=1e-4, 
                       des_enabled=None, sigma_mode=None, sigma_groups=None, sigma_update=None, sigma_values=None,
                       smooth_mode=None, smooth_groups=None, smooth_update=None, smooth_values=None,
                       report=None, qp_acceleration='off'):
        """
        Run Simple Variance Component Estimation (VCE) for multi-fault inversion.
    
        This method automatically determines optimal weights between data fitting and
        regularization components using iterative VCE approach with lsqlin solver.
        No manual penalty weights are needed - they are estimated through VCE iterations.
    
        Parameters
        ----------
        smoothing_constraints : tuple or dict, optional
            Smoothing constraints to apply during the least squares process. If None,
            the function will use the combined Green's functions matrix.
            If a tuple, it should be a 4-tuple. If a dict, the keys should be fault
            names and the values should be 4-tuples.
            (top, bottom, left, right) for the smoothing constraints.
        verbose : bool, optional
            Whether to print detailed progress information. Default is True.
        max_iter : int, optional
            Maximum number of VCE iterations. Default is 20.
        tol : float, optional
            Dimensionless tolerance for the largest absolute logarithmic
            variance-update factor.  VCE converges when every effective
            updated component satisfies ``abs(log(u)) < tol``.  Default is
            ``1e-4``.
        des_enabled : bool, optional
            Whether to use Depth-Equalized Smoothing (DES). If None, uses self.des_enabled.
        sigma_mode : str, optional
            Mode for data variance components: 'single', 'individual', or 'grouped'.
        sigma_groups : dict, optional
            Custom grouping for data variance components when sigma_mode='grouped'.
        sigma_update : list of bool, optional
            Whether to update each sigma group (same order as sigma groups)
        sigma_values : list of float, optional
            Initial/fixed values for each sigma group (same order as sigma groups)
        smooth_mode : str, optional
            Mode for smoothing variance components: 'single', 'individual', or 'grouped'.
        smooth_groups : dict, optional
            Custom grouping for smoothing variance components when smooth_mode='grouped'.
        smooth_update : list of bool, optional
            Whether VCE updates each smoothing group. ``False`` keeps that
            group's alpha fixed at ``smooth_values`` (or at
            ``alpha.initial_value`` when no runtime value is supplied).
        smooth_values : list of float, optional
            Initial or fixed alpha values for each smoothing group. Values use
            the representation selected by ``alpha.log_scaled`` in the loaded
            configuration.
        report : {None, 'none', 'compact', 'full'}, optional
            Final reporting policy. ``None`` selects ``'compact'`` when
            ``verbose=True`` and ``'none'`` otherwise. ``'compact'`` prints
            one variance-component table; ``'full'`` additionally prints the
            current model's fit table and available model regularization.
            Iteration progress remains controlled only by ``verbose``.
        qp_acceleration : {'off', 'certified_kkt'}, optional
            Optional VCE-local active-set acceleration. The central solver
            always routes the problem automatically. The default ``'off'``
            adds no sequential fast path; ``'certified_kkt'`` attempts reuse
            of a certified active set and falls back to the central route on
            any rejection.
    
        Returns
        -------
        dict
            VCE results containing:
            - 'm': estimated parameters
            - 'solved_sigma2_by_group'/'solved_alpha2_by_group': variance
              scales used by the returned model
            - 'proposed_sigma2_by_group'/'proposed_alpha2_by_group': values
              proposed for a possible next iteration
            - 'sigma_groups'/'smooth_groups': resolved member mappings
            - 'component_diagnostics': group Qw and approximate reduced Q
            - 'qp_diagnostics': report-only acceleration, solver-route and
              fallback counters
            - 'convergence_metric': ``'max_abs_log_update_factor'``
            - 'smoothing_matrix'/'model_smoothing_matrix': exact unscaled
              and active weighted regularization rows for the returned model
            - 'smoothing_provenance': source, method, bounds and DES context
            - 'converged': convergence flag
            - 'iterations': number of iterations

        Notes
        -----
        With no runtime component arguments, the normalized configuration is
        used as one coherent group contract.  A runtime override must provide
        ``mode``, ``update``, and ``values`` together, plus ``groups`` for
        ``grouped`` mode.  This prevents a new grouping from being paired with
        stale values or update flags from the loaded configuration.
        """
        self._clear_fit_weight_context()
        self._clear_scale_parameter_context()
        resolved_report = self._resolve_result_report(report, verbose=verbose)
        sigma_uses_config = all(
            value is None
            for value in (sigma_mode, sigma_groups, sigma_update, sigma_values)
        )
        smooth_uses_config = all(
            value is None
            for value in (smooth_mode, smooth_groups, smooth_update, smooth_values)
        )

        sigma_contract = self._resolve_vce_component_contract(
            component_name="sigma",
            configured=self.config.sigmas,
            member_names=list(self.data_ranges),
            mode=sigma_mode,
            groups=sigma_groups,
            update=sigma_update,
            values=sigma_values,
            member_label="dataset",
            individual_prefix="group_",
        )
        sigma_mode = sigma_contract['mode']
        sigma_groups = sigma_contract['members_by_group']
        sigma_update = sigma_contract['update_by_group']
        sigma_values = sigma_contract['values_by_group']
        if self.config.sigmas['log_scaled']:
            sigma_values = np.power(10, sigma_values)**2
        else:
            sigma_values = np.array(sigma_values)**2
        # print(sigma_mode, sigma_groups, sigma_update, sigma_values)

        # Check if alpha (smoothing) is disabled
        alpha_disabled = not self.config.alpha_enabled
        smoothing_faultnames = [
            name
            for name in self.faultnames
            if self.adapters[name].supports_smoothing()
        ]

        if alpha_disabled:
            # Sigma-only VCE: no smoothing estimation, only data weighting
            smooth_mode = 'single'
            smooth_groups = {'no_smooth': smoothing_faultnames}
            smooth_values = [1.0]
            smooth_update = [False]  # No smoothing rows, so alpha is fixed.
        else:
            smooth_contract = self._resolve_vce_component_contract(
                component_name="smooth",
                configured=self.config.alpha,
                member_names=smoothing_faultnames,
                mode=smooth_mode,
                groups=smooth_groups,
                update=smooth_update,
                values=smooth_values,
                member_label="smoothing source",
                individual_prefix="smooth_",
            )
            smooth_mode = smooth_contract['mode']
            smooth_groups = smooth_contract['members_by_group']
            smooth_values = smooth_contract['values_by_group']
            smooth_update = smooth_contract['update_by_group']
            if self.config.alpha['log_scaled']:
                smooth_values = np.power(10, smooth_values)**2
            else:
                smooth_values = np.array(smooth_values)**2
        # print(smooth_mode, smooth_groups, smooth_update, smooth_values)
        if des_enabled is None:
            des_enabled = getattr(self, 'des_enabled', False)

        if verbose:
            print("="*70)
            print("Starting Simple VCE for Multi-Fault Inversion")
            if alpha_disabled:
                print("Alpha smoothing is DISABLED; running sigma-only VCE (data weighting only).")
            else:
                print("Automatically determining optimal regularization weights...")
            print(f"Number of faults: {len(self.faults)}")
            print(f"Data variance mode: {sigma_mode}")
            print(f"Smoothing variance mode: {smooth_mode}")
            print(f"DES enabled: {des_enabled}")
            print("="*70)
    
        # Ensure bounds are set through the constraint manager, which is the
        # canonical source for BLSE/VCE constraints.
        lb = self.constraint_manager.lb
        ub = self.constraint_manager.ub
        if lb is None or ub is None:
            raise ValueError(
                "Bounds must be set before running VCE. Use "
                "apply_constraints_from_config() or update_bounds()."
            )
        if np.any(np.isnan(lb)) or np.any(np.isnan(ub)):
            raise ValueError("Some bounds are not set (NaN values found). Please set all bounds first.")
    
        # Handle smoothing constraints (ignored when alpha is disabled)
        if alpha_disabled:
            smoothing_constraints = None
            if verbose:
                print("Alpha disabled; smoothing_constraints ignored, using empty smoothing matrix.")
        elif smoothing_constraints is not None:
            if isinstance(smoothing_constraints, (tuple, list)) and len(smoothing_constraints) == 4:
                smoothing_constraints = {fault_name: smoothing_constraints for fault_name in self.faultnames}
            elif isinstance(smoothing_constraints, dict):
                missing_faults = set(self.faultnames) - set(smoothing_constraints.keys())
                if missing_faults:
                    if verbose:
                        print(f"Warning: Smoothing constraints not specified for faults: {missing_faults}")
                        print("Using default constraints (None, None, None, None) for these faults.")
                    for fault_name in missing_faults:
                        smoothing_constraints[fault_name] = (None, None, None, None)
            else:
                raise ValueError("smoothing_constraints should be a 4-tuple or a dictionary with fault names as keys and 4-tuples as values.")
    
        # Prepare smoothing matrix if using custom constraints
        if smoothing_constraints is not None:
            if verbose:
                print("Using custom smoothing constraints...")
                for fault_name, constraints in smoothing_constraints.items():
                    if all(c is not None for c in constraints):
                        print(f"  {fault_name}: top={constraints[0]}, bottom={constraints[1]}, left={constraints[2]}, right={constraints[3]} km")
                    else:
                        print(f"  {fault_name}: using default constraints")
    
            vce_result = self.simple_vce(
                smoothing_matrix=None,
                smoothing_constraints=smoothing_constraints,
                method='mudpy',
                verbose=verbose,
                max_iter=max_iter,
                tol=tol,
                des_enabled=des_enabled,
                sigma_mode=sigma_mode,
                sigma_groups=sigma_groups,
                sigma_update=sigma_update,
                sigma_values=sigma_values,
                smooth_mode=smooth_mode,
                smooth_groups=smooth_groups,
                smooth_update=smooth_update,
                smooth_values=smooth_values,
                qp_acceleration=qp_acceleration,
            )
        else:
            if verbose:
                print("Using default Laplacian smoothing matrix...")
    
            self.combine_GL_poly(penalty_weight=1.0)
    
            vce_result = self.simple_vce(
                smoothing_matrix=self.GL_combined_poly,
                smoothing_constraints=None,
                method='mudpy',
                verbose=verbose,
                max_iter=max_iter,
                tol=tol,
                des_enabled=des_enabled,
                sigma_mode=sigma_mode,
                sigma_groups=sigma_groups,
                sigma_update=sigma_update,
                sigma_values=sigma_values,
                smooth_mode=smooth_mode,
                smooth_groups=smooth_groups,
                smooth_update=smooth_update,
                smooth_values=smooth_values,
                qp_acceleration=qp_acceleration,
            )
    
        self.distributem()

        # Post-process penalty weights (only meaningful when smoothing is enabled)
        if alpha_disabled:
            self.current_penalty_weight = np.zeros(len(self.faults))
        else:
            self.current_penalty_weight = self._vce_penalty_weights_by_fault(
                vce_result
            )

        # Publish the exact final scales used by this VCE result. Reporting
        # reads this state later but never feeds it back into the solver.
        result_sigma_groups = vce_result['sigma_groups']
        active_var_d = vce_result['solved_sigma2_by_group']
        dataset_sigmas = {}
        for group, members in result_sigma_groups.items():
            sigma = float(np.sqrt(active_var_d[group]))
            dataset_sigmas.update({name: sigma for name in members})
        component_diagnostics = vce_result.get('component_diagnostics', {})
        group_dofs = {
            group: values.get('effective_dof')
            for group, values in component_diagnostics.get('data', {}).items()
            if values.get('effective_dof') is not None
        }
        self._publish_data_sigma_context(
            dataset_sigmas,
            group_members=result_sigma_groups,
            group_dofs=group_dofs,
        )

        # Publish the exact weighted smoothing rows used by the returned VCE
        # model.  In particular, custom boundary constraints create a local L
        # that need not equal mutable ``fault.GL``.  Rebuilding here would make
        # reporting state depend on a different matrix than the solve.
        self.current_smoothing_matrix = vce_result['smoothing_matrix']
        self.current_smoothing_provenance = dict(
            vce_result['smoothing_provenance']
        )
        self.current_model_smoothing_matrix = vce_result[
            'model_smoothing_matrix'
        ]
        self.GL_combined_poly = self.current_model_smoothing_matrix

        sigma_fixed_source = (
            'config sigmas.initial_value'
            if sigma_uses_config else 'run_simple_vce sigma_values'
        )
        sigma_fixed_source += (
            ' [log10(s)]' if self.config.sigmas['log_scaled'] else ' [s]'
        )
        vce_result['sigma_value_source_by_group'] = {
            group: ('-' if bool(should_update) else sigma_fixed_source)
            for group, should_update in vce_result.get(
                'sigma_update_by_group', {}
            ).items()
        }
        if not alpha_disabled:
            smooth_fixed_source = (
                'config alpha.initial_value'
                if smooth_uses_config else 'run_simple_vce smooth_values'
            )
            smooth_fixed_source += (
                ' [log10(s)]' if self.config.alpha['log_scaled'] else ' [s]'
            )
            vce_result['smooth_value_source_by_group'] = {
                group: ('-' if bool(should_update) else smooth_fixed_source)
                for group, should_update in vce_result.get(
                    'smooth_update_by_group', {}
                ).items()
            }

        status = 'converged' if vce_result.get('converged') else 'not converged'
        self._publish_scale_parameter_context(
            rows=self._build_vce_scale_rows(vce_result),
            method='VCE',
            title=(
                'VCE variance components '
                f"({status}, {vce_result.get('iterations', 0)} iterations)"
            ),
        )

        if resolved_report in {'compact', 'full'}:
            print("\n" + self.format_scale_parameters())
        if resolved_report == 'full':
            self.calculate_and_print_fit_statistics()
            self._print_model_regularization()

        return vce_result

    def _resolved_sigma_group_members(self, mode, groups=None):
        """Return the same dataset grouping contract used by BLSE/VCE."""
        return resolve_group_layout(
            list(self.data_ranges),
            mode,
            groups,
            member_label="dataset",
            single_group_name="all",
            individual_prefix="group_",
        )["members_by_group"]

    def _vce_penalty_weights_by_fault(self, result):
        """Expand active VCE alpha scales to the current fault order.

        The result's explicit group membership is authoritative so runtime
        ``smooth_mode``/``smooth_groups`` overrides cannot be remapped through
        stale configuration indices.
        """
        active = result.get('solved_alpha2_by_group')
        groups = result.get('smooth_groups')
        if isinstance(active, dict) and isinstance(groups, dict):
            by_fault = {}
            known_faults = set(self.faultnames)
            smoothing_faults = {
                name
                for name in self.faultnames
                if self.adapters[name].supports_smoothing()
            }
            for group, members in groups.items():
                if group not in active:
                    raise ValueError(
                        f"VCE smooth group '{group}' has no active variance."
                    )
                weight = 1.0 / np.sqrt(float(active[group]))
                for fault_name in members:
                    if fault_name not in known_faults:
                        raise ValueError(
                            f"Unknown fault '{fault_name}' in VCE smooth group "
                            f"'{group}'."
                        )
                    if fault_name not in smoothing_faults:
                        raise ValueError(
                            f"Source '{fault_name}' in VCE smooth group "
                            f"'{group}' does not support smoothing."
                        )
                    if fault_name in by_fault:
                        raise ValueError(
                            f"Fault '{fault_name}' belongs to multiple VCE "
                            "smooth groups."
                        )
                    by_fault[fault_name] = weight
            missing = [
                name for name in self.faultnames
                if name in smoothing_faults and name not in by_fault
            ]
            if missing:
                raise ValueError(
                    "VCE smooth groups do not cover faults: "
                    + ", ".join(missing)
                )
            return np.asarray(
                [by_fault.get(name, 1.0) for name in self.faultnames],
                dtype=float,
            )

        raise ValueError(
            "VCE result must provide solved_alpha2_by_group and "
            "smooth_groups as mappings."
        )

    def _publish_data_sigma_context(
        self,
        dataset_sigmas,
        *,
        group_members,
        group_dofs=None,
    ):
        """Publish report-only sigma metadata for the active linear model."""
        self.current_data_sigmas = {
            str(name): float(value) for name, value in dataset_sigmas.items()
        }
        self.current_data_weights = {
            name: 1.0 / sigma for name, sigma in self.current_data_sigmas.items()
        }
        self.current_data_sigma_group_members = {
            str(group): list(members) for group, members in group_members.items()
        }
        self.current_data_sigma_groups = {
            str(member): str(group)
            for group, members in self.current_data_sigma_group_members.items()
            for member in members
        }
        self.current_data_effective_dof = dict(group_dofs or {})

    def _publish_data_weight_context(self, *, data_weights, group_members):
        """Publish fixed-BLSE weights when they have a sigma interpretation."""
        data_names = list(self.data_ranges)
        weights = np.asarray(data_weights, dtype=float).reshape(-1)
        self.current_data_weights = dict(zip(data_names, weights))
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            # A zero/negative direct multiplier can be passed to the solver,
            # but it is not a positive standard-deviation scale. Keep raw
            # RMS/VR available without manufacturing weighted diagnostics.
            self.current_data_sigmas = {}
            return
        self._publish_data_sigma_context(
            dict(zip(data_names, 1.0 / weights)),
            group_members=group_members,
        )

    def _snapshot_linear_result_state(self):
        """Capture the active linear result without copying invariant matrices.

        The snapshot is intentionally limited to state changed by a BLSE solve
        or by result distribution.  Geometry, Green's functions, covariance,
        constraints, Laplacians owned by the sources, and mesh caches are not
        part of this transaction and are therefore neither copied nor rebuilt.
        """
        solver_state = {
            name: (hasattr(self, name), getattr(self, name, None))
            for name in self._LINEAR_RESULT_STATE_ATTRIBUTES
        }
        source_states = []
        for source in self.faults:
            result_attributes = self.adapters[
                source.name
            ].get_result_state_attributes()
            state = {
                name: (
                    hasattr(source, name),
                    copy.deepcopy(getattr(source, name, None)),
                )
                for name in result_attributes
            }
            source_states.append((source, state))
        return solver_state, source_states

    def _restore_linear_result_state(self, snapshot):
        """Restore a snapshot created by :meth:`_snapshot_linear_result_state`."""
        solver_state, source_states = snapshot
        for name, (existed, value) in solver_state.items():
            if existed:
                setattr(self, name, value)
            elif hasattr(self, name):
                delattr(self, name)
        for source, state in source_states:
            for name, (existed, value) in state.items():
                if existed:
                    setattr(source, name, value)
                elif hasattr(source, name):
                    delattr(source, name)

    def _snapshot_prediction_result_state(self):
        """Capture only geodata fields rebuilt by fit-statistics collection."""
        data_objects, _, _ = self._get_fit_geodata_config()
        states = []
        for data in data_objects:
            state = {
                name: (
                    hasattr(data, name),
                    copy.deepcopy(getattr(data, name, None)),
                )
                for name in get_prediction_result_state_attributes(data)
            }
            states.append((data, state))
        return states

    @staticmethod
    def _restore_prediction_result_state(states):
        """Restore a snapshot created by ``_snapshot_prediction_result_state``."""
        for data, state in states:
            for name, (existed, value) in state.items():
                if existed:
                    setattr(data, name, value)
                elif hasattr(data, name):
                    delattr(data, name)

    def scan_penalty_weights(
        self,
        penalty_weights,
        *,
        include_fit_statistics=False,
        include_weighted=False,
        verbose=True,
    ):
        """Scan scalar BLSE penalty weights on one fixed geometry.

        The scan owns candidate execution, scan-local quadratic reuse and
        restoration of the active result.  It performs no plotting or file
        output.  The returned summary always contains global solver-vector
        RMS/VR and unweighted model roughness.  Per-dataset rows are optional
        because rebuilding their synthetic fields has a measurable cost.

        Returns
        -------
        tuple of pandas.DataFrame
            ``(summary, fit_stats)``.  ``fit_stats`` contains dataset rows
            only and has a stable empty schema when collection is disabled.
        """
        if not self.config.alpha_enabled:
            raise ValueError(
                "penalty-weight scanning requires alpha.enabled: true"
            )
        if include_weighted and not include_fit_statistics:
            raise ValueError(
                "include_weighted=True requires include_fit_statistics=True"
            )

        candidates = np.asarray(penalty_weights, dtype=float)
        if candidates.ndim == 0:
            candidates = candidates.reshape(1)
        elif candidates.ndim != 1:
            raise ValueError("penalty_weights must be a scalar or one-dimensional")
        if candidates.size == 0:
            raise ValueError("penalty_weights must contain at least one value")
        if np.any(~np.isfinite(candidates)) or np.any(candidates <= 0.0):
            raise ValueError(
                "penalty_weights must be finite and strictly positive"
            )

        observation_unit = get_observation_unit_info(self)['observation']
        summary_records = []
        fit_records = []
        entry_state = self._snapshot_linear_result_state()
        prediction_state = (
            self._snapshot_prediction_result_state()
            if include_fit_statistics else None
        )
        try:
            with self._blse_quadratic_scan_context():
                for candidate_index, candidate in enumerate(candidates):
                    weight = float(candidate)
                    self.run(
                        penalty_weight=weight,
                        alpha=None,
                        verbose=verbose,
                        report='none',
                    )

                    rows = self.collect_fit_statistics(
                        model=f'penalty_{weight:g}',
                        data_poly='config',
                        include_dataset=include_fit_statistics,
                        include_global=True,
                        include_weighted=include_weighted,
                        rebuild_synth=include_fit_statistics,
                    )
                    global_rows = [
                        row for row in rows
                        if row.get('scope') == 'global_solver_vector'
                    ]
                    if len(global_rows) != 1:
                        raise RuntimeError(
                            "penalty scan requires exactly one global solver fit row"
                        )
                    global_fit = global_rows[0]

                    base_smoothing = np.asarray(
                        self.current_smoothing_matrix, dtype=float
                    )
                    roughness_vector = base_smoothing.dot(self.mpost)
                    roughness = (
                        float(np.sqrt(np.mean(roughness_vector ** 2)))
                        if roughness_vector.size else 0.0
                    )

                    resolved_weights = np.asarray(
                        self.current_penalty_weight, dtype=float
                    ).reshape(-1)
                    resolved_order = list(self.faultnames)
                    if resolved_weights.size != len(resolved_order):
                        raise RuntimeError(
                            "resolved penalty weights do not align with faultnames"
                        )
                    summary_records.append({
                        'candidate_index': int(candidate_index),
                        'penalty_weight': weight,
                        'equivalent_log10_alpha': float(-np.log10(weight)),
                        'resolved_penalty_order': json.dumps(
                            resolved_order, separators=(',', ':')
                        ),
                        'resolved_penalty_weights': json.dumps(
                            resolved_weights.tolist(), separators=(',', ':')
                        ),
                        'roughness': roughness,
                        'global_rms': float(global_fit['rms']),
                        'global_vr_percent': float(global_fit['vr']),
                        'observation_unit': observation_unit,
                    })

                    if include_fit_statistics:
                        for row in rows:
                            if row.get('scope') != 'dataset':
                                continue
                            fit_records.append({
                                'candidate_index': int(candidate_index),
                                'penalty_weight': weight,
                                'observation_unit': observation_unit,
                                **row,
                            })

                    if verbose:
                        print(
                            f'Penalty_weight: {weight:g}, '
                            f'Roughness: {roughness:.4f}, '
                            f'RMS: {float(global_fit["rms"]):.4f}, '
                            f'VR: {float(global_fit["vr"]):.2f}%'
                        )
        finally:
            try:
                if prediction_state is not None:
                    self._restore_prediction_result_state(prediction_state)
            finally:
                self._restore_linear_result_state(entry_state)

        summary = pd.DataFrame(
            summary_records, columns=_PENALTY_SCAN_SUMMARY_COLUMNS
        )
        fit_columns = list(_PENALTY_SCAN_FIT_COLUMNS)
        if include_weighted:
            fit_columns.extend(_PENALTY_SCAN_WEIGHTED_COLUMNS)
        fit_stats = pd.DataFrame(fit_records).reindex(columns=fit_columns)
        return summary, fit_stats

    def simple_run_loop(self, penalty_weights=None, output_file='run_loop.dat', preferred_penalty_weight=None, rms_unit='m', verbose=True, equal_aspect=False):
        """
        Diagnose a range of BLSE penalty weights on one fixed geometry.

        This compatibility wrapper delegates candidate execution to
        :meth:`scan_penalty_weights`, writes the historical four-column table,
        and produces the historical single-panel figure.  New scripts should
        call :meth:`scan_penalty_weights` directly.

        Every candidate is solved independently from the source Laplacians.
        Its reported roughness uses the exact unweighted matrix ``L0``
        published by that solve as ``current_smoothing_matrix``; the active
        objective matrix remains available as
        ``current_model_smoothing_matrix``.  The method restores the complete
        active result that existed on entry, both after success and after an
        exception.  Use :meth:`run` explicitly with the selected penalty to
        make a candidate the active model.
    
        Parameters:
        -----------
        penalty_weights : list or np.ndarray
            Finite positive penalty weights to scan.  Configuration defaults
            do not define this candidate list or the unweighted ``L0``.
        output_file : str, optional
            Path to the output file. If None, the results will only be printed to the screen. Default is 'run_loop.dat'.
        preferred_penalty_weight : float, optional
            The preferred penalty weight to highlight in the plot. If None, no preferred point will be highlighted.
        rms_unit : str, optional
            Target display unit for the plot.  The returned legacy table and
            CSV retain RMS in the configured observation unit.
        verbose : bool, optional
            Whether to print one summary line per candidate. Default is True.
        equal_aspect : bool, optional
            If True, set equal aspect ratio for the plot. Default is False.

        Returns
        -------
        pandas.DataFrame
            Candidate penalty, unweighted roughness, RMS, and variance
            reduction.  Returning does not activate any candidate model.
        """
        summary, _ = self.scan_penalty_weights(
            penalty_weights,
            include_fit_statistics=False,
            verbose=verbose,
        )
        self.plot_roughness_vs_rms(
            summary,
            output_file='Roughness_vs_RMS.png',
            show=True,
            preferred_penalty_weight=preferred_penalty_weight,
            rms_unit=rms_unit,
            equal_aspect=equal_aspect,
        )
        legacy = summary.rename(columns={
            'penalty_weight': 'Penalty_weight',
            'roughness': 'Roughness',
            'global_rms': 'RMS',
            'global_vr_percent': 'VR',
        })[['Penalty_weight', 'Roughness', 'RMS', 'VR']]
        legacy.attrs['observation_unit'] = summary['observation_unit'].iloc[0]
        if output_file:
            legacy.to_csv(output_file, index=False)
        return legacy
    
    def plot_roughness_vs_rms(self, df, output_file='Roughness_vs_RMS.png', show=True, preferred_penalty_weight=None, rms_unit='m', equal_aspect=False):
        """
        Plot Roughness vs RMS and save the plot to a file.
    
        Parameters:
        -----------
        df : pd.DataFrame
            Canonical penalty-scan summary or a historical table containing
            ``Penalty_weight``, ``Roughness`` and ``RMS``.
        output_file : str, optional
            Path to the output file. Default is 'Roughness_vs_RMS.png'.
        show : bool, optional
            Whether to display the plot. Default is True.
        preferred_penalty_weight : float, optional
            The preferred penalty weight to highlight in the plot. If None, no preferred point will be highlighted.
        rms_unit : str, optional
            Target display unit.  Canonical summaries carry their source unit;
            historical tables without unit metadata retain the legacy
            assumption that their RMS is in metres.
        equal_aspect : bool, optional
            If True, set equal aspect ratio for the plot. Default is False.
        """
        canonical_columns = {
            'candidate_index', 'penalty_weight', 'roughness', 'global_rms',
            'global_vr_percent', 'observation_unit',
        }
        if canonical_columns.issubset(df.columns):
            summary = df.copy()
            source_units = summary['observation_unit'].dropna().astype(str).unique()
            if source_units.size != 1:
                raise ValueError(
                    "canonical BLSE summary must contain one observation_unit"
                )
            source_unit = source_units[0]
        else:
            required = {'Penalty_weight', 'Roughness', 'RMS'}
            missing = sorted(required.difference(df.columns))
            if missing:
                raise ValueError(
                    "legacy BLSE loop table is missing column(s): "
                    + ", ".join(missing)
                )
            source_unit = df.attrs.get('observation_unit', 'm')
            summary = pd.DataFrame({
                'candidate_index': np.arange(len(df), dtype=int),
                'penalty_weight': df['Penalty_weight'].to_numpy(dtype=float),
                'roughness': df['Roughness'].to_numpy(dtype=float),
                'global_rms': df['RMS'].to_numpy(dtype=float),
                'global_vr_percent': (
                    df['VR'].to_numpy(dtype=float)
                    if 'VR' in df else np.full(len(df), np.nan)
                ),
                'observation_unit': source_unit,
            })

        source_info = parse_observation_unit(source_unit)
        target_info = parse_observation_unit(rms_unit)
        if source_info['kind'] != target_info['kind']:
            raise ValueError(
                f"Cannot convert RMS from {source_unit!r} to {rms_unit!r}: "
                "displacement and rate units are not interchangeable"
            )
        display_summary = summary.copy()
        display_summary['global_rms'] = (
            display_summary['global_rms'].to_numpy(dtype=float)
            * source_info['to_si'] / target_info['to_si']
        )
        display_summary['observation_unit'] = target_info['observation']
        fig, _ = plot_blse_roughness_rms(
            display_summary,
            preferred_penalty_weight=preferred_penalty_weight,
            equal_aspect=equal_aspect,
        )
        if output_file:
            fig.savefig(output_file, dpi=600)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def reassemble_data(self, geodata=None, trifaults_list=None, verticals=None):
        """
        Assemble data for inversion.

        Parameters:
        - geodata: list
            List of geodetic data objects (e.g., InSAR, GPS, Optical).
        - trifaults_list: list
            List of fault objects.
        - verticals: list
            List of vertical data objects.
        """
        self._clear_fit_weight_context()
        self._clear_scale_parameter_context()
        faults = self.faults if trifaults_list is None else trifaults_list
        geodata = self.config.geodata['data'] if geodata is None else geodata
        vertical = self.config.geodata['verticals'] if verticals is None else verticals
        for data, vert in zip(geodata, vertical):
            for ifault in faults:
                if data.dtype in ('insar', 'tsunami'):
                    ifault.d[data.name] = data.vel if data.dtype == 'insar' else data.d
                elif data.dtype in ('gps', 'multigps'):
                    ifault.d[data.name] = data.vel_enu[:, :data.obs_per_station].T.flatten()
                    ifault.d[data.name] = ifault.d[data.name][np.isfinite(ifault.d[data.name])]
                elif data.dtype == 'opticorr':
                    ifault.d[data.name] = np.hstack((data.east.T.flatten(), data.north.T.flatten()))
                    if vert:
                        ifault.d[data.name] = np.hstack((ifault.d[data.name], np.zeros_like(data.east.T.ravel())))
                elif data.dtype == 'leveling':
                    ifault.d[data.name] = data.vel
                elif data.dtype == 'crossfaultoffset':
                    ifault.d[data.name] = data.data_vector

        for ifault in faults:
            ifault.assembled(geodata)

        self.d = faults[0].dassembled

    def returnModel(
        self,
        mpost=None,
        print_stat=True,
        *,
        print_fit_statistics=None,
    ):
        """Activate one BLSE/VCE model and optionally print its fit table.

        ``print_fit_statistics`` is the canonical result-API spelling;
        ``print_stat`` remains accepted for existing scripts.

        If ``mpost`` is supplied, it becomes the active solution. The method
        keeps ``self.mpost``, distributed source parameters, fit statistics,
        and returned RMS/VR on that same vector. Roughness is computed from
        the unscaled smoothing matrix published by the successful solve;
        solver weighting remains available in
        ``current_model_smoothing_matrix``.
        """
        if print_fit_statistics is not None:
            print_stat = bool(print_fit_statistics)
        if mpost is not None:
            # A supplied vector becomes the active model, matching the result
            # APIs of the Bayesian solvers.  Keeping self.mpost, distributed
            # source parameters, predictions, and reported statistics on one
            # vector avoids the former half-temporary state.
            self.mpost = np.asarray(mpost, dtype=float).copy()
        self.distributem()
        
        # Calculate and print fit statistics
        if print_stat:
            self.calculate_and_print_fit_statistics()

        # Caluculate RMS and VR for the solution and print the results
        rms = np.sqrt(np.mean((np.dot(self.G, self.mpost) - self.d)**2))
        vr = (1 - np.sum((np.dot(self.G, self.mpost) - self.d)**2) / np.sum(self.d**2)) * 100
        active_smoothing = getattr(
            self,
            'current_smoothing_matrix',
            self.GL_combined_poly,
        )
        regularization = self._collect_model_regularization(
            model_vector=self.mpost,
            smoothing_matrix=active_smoothing,
        )
        roughness = (
            0.0 if regularization is None else regularization['roughness']
        )
        if print_stat:
            self._print_model_regularization(
                model_vector=self.mpost,
                smoothing_matrix=active_smoothing,
            )
        return roughness, rms, vr
    
    def calculate_and_print_fit_statistics(self):
        """
        Calculate and print fit statistics for all datasets.
        """
        # Call parent class method with 'BLSE' model
        super().calculate_and_print_fit_statistics(model='BLSE')
        

    def combine_GL_poly(self, GL_combined=None, penalty_weight=None):
        """Build the weighted smoothing matrix in global model-column order.

        ``GL_combined`` is the optional, unweighted block-diagonal matrix for
        smoothing-capable sources only.  Its blocks follow ``self.faults``
        order and do not yet contain polynomial columns.  When omitted, each
        source's current ``fault.GL`` supplies the corresponding block.  This
        method inserts zero polynomial/non-smoothing columns and then applies
        each source's penalty weight, so the result aligns with the global
        linear model vector.

        Parameters
        ----------
        GL_combined : array-like or sparse matrix, optional
            Explicit source-only smoothing blocks.  The full matrix must be
            consumed by the current smoothing-source layout.
        penalty_weight : int, float, list, or numpy.ndarray, optional
            Per-source smoothing weights.  A scalar or one-element sequence
            is broadcast to all sources.  If omitted, values are derived from
            the configured alpha state.

        Returns
        -------
        numpy.ndarray
            Weighted smoothing rows with columns matching the global model
            parameter layout.
        """
        if penalty_weight is not None:
            if isinstance(penalty_weight, (int, float)):
                penalty_weight = np.ones(len(self.faults)) * penalty_weight
            elif isinstance(penalty_weight, (list, np.ndarray)):
                if len(penalty_weight) == 1:
                    # Single value in list, expand it
                    penalty_weight = np.ones(len(self.faults)) * penalty_weight[0]
                assert len(penalty_weight) == len(self.faults) or len(penalty_weight) == 1, "The length of penalty_weight should be equal to the number of faults or a single value."
            else:
                raise ValueError("penalty_weight should be a scalar or a list of scalars.")
        else:
            # When alpha is disabled, use uniform weight=1.0 (no regularization penalty)
            if not self.config.alpha_enabled:
                penalty_weight = np.ones(len(self.faults))
            else:
                alpha = np.array(self.config.alpha['initial_value'])
                fault_index = self.config.alpha['fault_param_indices']
                alpha = alpha[fault_index]
                assert len(alpha) == len(self.faults), "The length of alpha should be equal to the number of faults."
                if self.config.alpha['log_scaled']:
                    penalty_weight = 1.0 / np.power(10, alpha)
                else:
                    penalty_weight = 1.0 / alpha

        # When alpha is disabled, skip all GL blocks to prevent smoothing
        alpha_disabled = not self.config.alpha_enabled

        explicit_gl = None
        if GL_combined is not None:
            explicit_gl = (
                GL_combined.toarray()
                if hasattr(GL_combined, 'toarray')
                else GL_combined
            )
            explicit_gl = np.asarray(explicit_gl, dtype=float)
            if explicit_gl.ndim != 2:
                raise ValueError("GL_combined must be a two-dimensional matrix.")

        expanded_blocks = []
        row_start = 0
        col_start = 0
        for fault, ipenalty_weight in zip(self.faults, penalty_weight):
            has_gl = hasattr(fault, 'GL') and fault.GL is not None
            slip_start, slip_end = self.slip_positions.get(
                fault.name, (0, 0)
            )
            poly_start, poly_end = self.poly_positions.get(
                fault.name, (0, 0)
            )
            n_source = slip_end - slip_start
            n_poly = poly_end - poly_start

            source_gl = None
            if has_gl:
                n_rows, n_cols = fault.GL.shape
                if n_cols != n_source:
                    raise ValueError(
                        "Smoothing columns are inconsistent with the linear "
                        f"parameter layout for source '{fault.name}': "
                        f"GL has {n_cols}, layout has {n_source}."
                    )
                if explicit_gl is None:
                    source_gl = (
                        fault.GL.toarray()
                        if hasattr(fault.GL, 'toarray')
                        else np.asarray(fault.GL, dtype=float)
                    )
                else:
                    row_end = row_start + n_rows
                    col_end = col_start + n_cols
                    if (
                        row_end > explicit_gl.shape[0]
                        or col_end > explicit_gl.shape[1]
                    ):
                        raise ValueError(
                            "GL_combined shape is inconsistent with current "
                            f"smoothing source '{fault.name}'."
                        )
                    source_gl = explicit_gl[
                        row_start:row_end, col_start:col_end
                    ]
                    row_start = row_end
                    col_start = col_end

            if has_gl and not alpha_disabled:
                combined = np.zeros((n_rows, n_source + n_poly))
                combined[:, :n_source] = source_gl * ipenalty_weight
                expanded_blocks.append(combined)
            else:
                n_params = n_source + n_poly
                if n_params > 0:
                    expanded_blocks.append(np.zeros((0, n_params)))

        if explicit_gl is not None and (
            row_start != explicit_gl.shape[0]
            or col_start != explicit_gl.shape[1]
        ):
            raise ValueError(
                "GL_combined contains rows or columns not described by the "
                "current smoothing sources."
            )

        if expanded_blocks:
            self.GL_combined_poly = scipy.linalg.block_diag(*expanded_blocks)
        else:
            self.GL_combined_poly = np.zeros((0, 0))

        return self.GL_combined_poly

    def extract_and_plot_blse_results(self, rank=0, 
                                          plot_faults=True, plot_data=True,
                                          antisymmetric=True, res_use_data_norm=True, cmap='RdBu_r', azimuth=None, elevation=None,
                                          slip_cmap='cmc.roma_r', depth_range=None, z_ticks=None, 
                                          axis_shape=(1.0, 1.0, 0.6), 
                                          zratio=None,
                                          gps_title=True, sar_title=True, sar_cbaxis=[0.1, 0.15, 0.35, 0.04], # [0.15, 0.25, 0.25, 0.02],
                                          gps_figsize=None, sar_figsize='double', gps_scale=0.05, gps_legendscale=0.2,
                                          file_type='png',
                                          remove_direction_labels=False,
                                          fault_cbaxis=[0.15, 0.22, 0.15, 0.02], 
                                          data_poly="config",
                                          print_fit_statistics=True,
                                          print_fault_statistics=True,
                                          fault_outdir='output',
                                          data_outdir='Modeling',
                                          show=True,
                                          ):
        """
        Extract and plot the Bayesian results.
    
        args:
        rank: process rank (default is 0)
        filename: name of the HDF5 file to save the samples (default is 'samples_mag_rake_multifaults.h5')
        plot_faults: whether to plot faults (default is True)
        plot_data: whether to plot data (default is True)
        antisymmetric: whether to set the colormap to be antisymmetric (default is True)
        res_use_data_norm: whether to make the norm of 'res' consistent with 'data' and 'synth' (default is True)
        cmap: colormap to use (default is 'RdBu_r')
        slip_cmap: colormap for slip (default is 'precip3_16lev_change.cpt')
        depth_range: depth range for the plot (default is None)
        z_ticks: z-axis ticks for the plot (default is None)
        zratio: optional z-axis compression ratio passed to plot_multifaults_slip
        gps_title: whether to show title for GPS data plots (default is True)
        sar_title: whether to show title for SAR data plots (default is True)
        sar_cbaxis: colorbar axis position for SAR data plots (default is [0.1, 0.15, 0.35, 0.04])
        gps_figsize: figure size for GPS data plots (default is None)
        sar_figsize: figure size for SAR data plots (default is (3.5, 2.7))
        gps_scale: scale for GPS data plots (default is 0.05)
        gps_legendscale: legend scale for GPS data plots (default is 0.2)
        file_type: file type to save the figures (default is 'png')
        remove_direction_labels : If True, remove E, N, S, W from axis labels (default is False)
        fault_cbaxis: colorbar axis position for fault plots (default is [0.15, 0.22, 0.15, 0.02])
        data_poly: prediction correction mode. "config" (default) follows
            each dataset's parsed geodata.polys value; "include" includes
            solved corrections; None explicitly plots the source/slip-only
            prediction.
        print_fit_statistics: whether to print fit statistics (default is True)
        print_fault_statistics: whether to print fault statistics (default is True)
        fault_outdir: directory for fault-field figures (default is 'output')
        data_outdir: directory for GPS/InSAR/leveling/cross-fault figures
            (default is 'Modeling')
        show: whether underlying plotting methods call their interactive show
            path (default is True)
        """
        if rank == 0:
            import cmcrameri
            from ..getcpt import get_cpt 

            file_type = normalize_image_format(file_type)
    
            if slip_cmap is not None and slip_cmap.endswith('.cpt'):
                # 'precip3_16lev_change.cpt'
                cmap_slip = get_cpt.get_cmap(slip_cmap, method='list', N=15)
            else:
                cmap_slip = slip_cmap
            if slip_cmap is None:
                cmap_slip = get_cpt.get_cmap('precip3_16lev_change.cpt', method='list', N=15)
            self.returnModel(print_fit_statistics=print_fit_statistics)
            if print_fault_statistics:
                self._print_fault_statistics()
    
            if plot_faults:
                self.plot_fault_fields(
                    fields=('total',),
                    outdir=fault_outdir,
                    file_type=file_type,
                    slip_cmap=cmap_slip,
                    show=show,
                    drawCoastlines=False,
                    cblabel='Slip (m)',
                    style=['notebook'],
                    cbaxis=fault_cbaxis,
                    xtickpad=5,
                    ytickpad=5,
                    ztickpad=5,
                    xlabelpad=15,
                    ylabelpad=15,
                    zlabelpad=15,
                    shape=axis_shape,
                    elevation=elevation,
                    azimuth=azimuth,
                    zratio=zratio,
                    depth=depth_range,
                    zticks=z_ticks,
                    fault_expand=0.0,
                    plot_faultEdges=False,
                    suffix='_slip',
                    remove_direction_labels=remove_direction_labels,
                )

            # The product layer preserves the established buildsynth contract:
            # GPS(vertical, poly), InSAR(vertical=True, poly), leveling
            # (vertical=True, poly), and cross-fault offsets(poly).
            self.plot_data_fits(
                data_types=("gps", "insar", "leveling", "crossfaultoffset"),
                outdir=data_outdir,
                file_type=file_type,
                plot_data=plot_data,
                data_poly=data_poly,
                antisymmetric=antisymmetric,
                res_use_data_norm=res_use_data_norm,
                cmap=cmap,
                gps_title=gps_title,
                sar_title=sar_title,
                gps_figsize=gps_figsize,
                sar_figsize=sar_figsize,
                gps_scale=gps_scale,
                gps_legendscale=gps_legendscale,
                sar_cbaxis=sar_cbaxis,
                remove_direction_labels=remove_direction_labels,
                gps_fault_color='k',
                sar_fault_color='k',
                fault_linewidth=2.0,
                show=show,
            )


#EOF
