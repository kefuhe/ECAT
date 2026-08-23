"""
Fault Analysis Mixin Module

This module provides a mixin class that contains common fault analysis methods
for seismic moment and moment magnitude calculations, fault statistics, and plotting.
This mixin can be used by different fault inversion classes to avoid code duplication.

Author: Kefeng He
Date: 2025-08-01
Version: 1.0.0
"""

from collections.abc import Mapping, Sequence

import numpy as np
from tabulate import tabulate

from .config.config_utils import get_observation_unit_info
from .data_prediction import get_geodata_prediction_specs, resolve_data_poly
from .fault_summary import print_faults_summary, summarize_faults
from .fit_statistics import (
    data_fit_vectors,
    fit_metrics_from_vectors,
    fit_statistics_rows_to_dataframe,
    format_fit_statistics_report,
    format_fit_statistics_table,
    solver_fit_metrics,
    weighted_fit_metrics_from_residual,
    write_fit_statistics_report_files,
)


class FaultAnalysisMixin:
    """
    A mixin class that provides fault analysis functionality including seismic moment
    and moment magnitude calculations, fault statistics printing, and slip distribution plotting.
    
    This mixin is designed to be inherited by classes that have fault objects,
    regardless of whether they are accessed via self.faults or self.multifaults.faults.
    
    Methods:
        _get_faults(): Abstract method to get fault list, implemented by subclasses
        calculate_moment_magnitude(): Calculate seismic moment and magnitude
        _calculate_moment_magnitude_csi(): CSI method implementation
        _calculate_moment_magnitude_hankel(): Hankel formula implementation
        print_moment_magnitude(): Print moment magnitude results
        _print_fault_statistics(): Print comprehensive fault statistics
        plot_multifaults_slip(): Plot slip distribution of multiple faults
    """
    
    def _get_faults(self):
        """
        Get the list of fault objects. This method should be overridden by subclasses
        to return the appropriate fault list based on their internal structure.
        
        Returns:
            list: List of fault objects
            
        Raises:
            AttributeError: If no fault list can be found in the object
        """
        if hasattr(self, 'multifaults') and hasattr(self.multifaults, 'faults'):
            return self.multifaults.faults
        elif hasattr(self, 'faults'):
            return self.faults
        else:
            raise AttributeError("Cannot find faults in this object. "
                               "Please ensure the class has either 'faults' or 'multifaults.faults' attribute.")

    def _select_faults(self, faults=None):
        """
        Select fault objects from this inversion object.

        ``faults`` may be ``None`` for all faults, a list of fault objects, or
        a list of fault names. The helper keeps summary and moment APIs aligned.
        """
        if faults is None:
            return self._get_faults()

        if isinstance(faults, list):
            if len(faults) == 0:
                return self._get_faults()
            if isinstance(faults[0], str):
                all_faults = self._get_faults()
                fault_dict = {fault.name: fault for fault in all_faults}
                target_faults = []
                for fault_name in faults:
                    if fault_name in fault_dict:
                        target_faults.append(fault_dict[fault_name])
                    else:
                        print(f"Warning: Fault '{fault_name}' not found in available faults")
                return target_faults
            return faults

        print("Warning: Invalid faults parameter, using available faults")
        return self._get_faults()

    def _default_fault_groups(self, target_faults):
        """
        Return configured fault groups when available.
        """
        if (hasattr(self, 'config') and hasattr(self.config, 'alpha')
            and isinstance(self.config.alpha, dict)
            and 'faults' in self.config.alpha):
            return self.config.alpha['faults']
        return None

    def _resolve_slip_factor(self, slip_factor=None):
        """
        Return the factor converting stored slip values to meters.

        ECAT assumes observations, Green's functions, slip variables and
        constraint right-hand sides are already in one numerical unit. Moment
        magnitude is different: it requires physical slip in meters.
        """
        if slip_factor is not None:
            return float(slip_factor), {
                "observation": None,
                "kind": None,
                "assumed": False,
                "explicit_slip_factor": True,
            }

        unit_info = get_observation_unit_info(self, default="m")
        if unit_info["kind"] != "displacement":
            raise ValueError(
                "Moment magnitude requires displacement slip. "
                f"units.observation is '{unit_info['observation']}', which is a rate unit. "
                "Pass slip_factor explicitly if you intentionally converted rates to cumulative slip."
            )
        unit_info = dict(unit_info)
        unit_info["explicit_slip_factor"] = False
        return float(unit_info["to_si"]), unit_info

    def _resolve_summary_slip_context(self, slip_factor=None, include_moment=True):
        """
        Resolve slip and moment scaling for fault summaries.

        Direct moment APIs must still reject rate units unless the caller passes
        ``slip_factor`` explicitly. Interactive summaries are diagnostic: for
        rate-unit inversions they keep the slip table in the configured rate
        unit and report moment rate in ``N*m/yr``.
        """
        if slip_factor is not None:
            factor, unit_info = self._resolve_slip_factor(slip_factor)
            return {
                "slip_factor": factor,
                "slip_unit_label": "m",
                "moment_slip_factor": factor,
                "moment_kind": "moment",
                "moment_unit_label": "N*m",
                "equivalent_duration_years": None,
                "include_moment": include_moment,
                "unit_info": unit_info,
            }

        unit_info = get_observation_unit_info(self, default="m")
        unit_info = dict(unit_info)
        unit_info["explicit_slip_factor"] = False

        if unit_info["kind"] == "displacement":
            factor = float(unit_info["to_si"])
            return {
                "slip_factor": factor,
                "slip_unit_label": "m",
                "moment_slip_factor": factor,
                "moment_kind": "moment",
                "moment_unit_label": "N*m",
                "equivalent_duration_years": None,
                "include_moment": include_moment,
                "unit_info": unit_info,
            }

        return {
            "slip_factor": 1.0,
            "slip_unit_label": unit_info.get("observation") or "stored",
            "moment_slip_factor": float(unit_info["to_si"]),
            "moment_kind": "moment_rate",
            "moment_unit_label": "N*m/yr",
            "equivalent_duration_years": 1.0,
            "include_moment": include_moment,
            "unit_info": unit_info,
        }
    
    def calculate_moment_magnitude(self, faults=None, mu=3.e10, slip_factor=None, mode='hankel'):
        """
        Calculate seismic moment and moment magnitude for specified faults.
        
        This method provides two calculation modes:
        - 'csi': Uses CSI faultpostproc method for combined fault processing
        - 'hankel': Uses standard Hankel formula (Mw = 2/3 * (log10(Mo) - 9.1))
        
        Args:
            faults (list, optional): List of fault objects or fault names to analyze.
                                   If None, uses all available faults. Defaults to None.
            mu (float, optional): Shear modulus in Pa. Defaults to 3.e10.
            slip_factor (float, optional): Factor to convert stored slip values to meters.
                If omitted, inferred from units.observation for displacement units.
            mode (str, optional): Calculation mode ('csi' or 'hankel'). Defaults to 'hankel'.
        
        Returns:
            dict: Dictionary containing moment and magnitude information with keys:
                - For 'csi' mode: 'total_moment', 'total_magnitude', 'mode', 'processor', 'faults'
                - For 'hankel' mode: 'fault_moments', 'total_moment', 'total_magnitude', 
                  'mode', 'mu', 'slip_factor', 'faults'
        
        Raises:
            ValueError: If mode is not 'csi' or 'hankel'
        """
        # Handle faults parameter
        if faults is None:
            target_faults = self._get_faults()
        elif isinstance(faults, list):
            if len(faults) > 0:
                if isinstance(faults[0], str):
                    # List of fault names
                    all_faults = self._get_faults()
                    fault_dict = {fault.name: fault for fault in all_faults}
                    target_faults = []
                    for fault_name in faults:
                        if fault_name in fault_dict:
                            target_faults.append(fault_dict[fault_name])
                        else:
                            print(f"Warning: Fault '{fault_name}' not found in available faults")
                else:
                    # List of fault objects
                    target_faults = faults
            else:
                target_faults = self._get_faults()
        else:
            target_faults = self._get_faults()
        
        resolved_slip_factor, unit_info = self._resolve_slip_factor(slip_factor)

        if mode == 'csi':
            return self._calculate_moment_magnitude_csi(target_faults, mu, resolved_slip_factor, unit_info)
        elif mode == 'hankel':
            return self._calculate_moment_magnitude_hankel(target_faults, mu, resolved_slip_factor, unit_info)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'csi' or 'hankel'.")
    
    def _calculate_moment_magnitude_csi(self, target_faults, mu=3.e10, slip_factor=1.0, unit_info=None):
        """
        Calculate moment magnitude using CSI faultpostproc method.
        
        This method combines multiple faults into a single fault object and uses
        the CSI library's faultpostproc class to compute seismic moments, moment
        tensor, and magnitude.
        
        Args:
            target_faults (list): List of fault objects to process
            mu (float, optional): Shear modulus in Pa. Defaults to 3.e10.
            slip_factor (float, optional): Factor to scale slip values. Defaults to 1.0.
        
        Returns:
            dict: Dictionary containing:
                - 'total_moment': Total seismic moment in N·m
                - 'total_magnitude': Moment magnitude
                - 'mode': 'csi'
                - 'processor': CSI faultpostproc object for further analysis
                - 'faults': List of fault names processed
        """
        import csi.faultpostproc as faultpp
        
        # Get the first fault for reference parameters
        first_fault = target_faults[0]
        lon0, lat0, utmzone = first_fault.lon0, first_fault.lat0, first_fault.utmzone

        # Combine the faults by duplicating the first fault
        combined_fault = first_fault.duplicateFault()

        # Add patches and slip from remaining faults
        if len(target_faults) > 1:
            for ifault in target_faults[1:]:
                for patch, slip in zip(ifault.patch, ifault.slip):
                    combined_fault.N_slip = combined_fault.slip.shape[0] + 1
                    combined_fault.addpatch(patch, slip)

        # Create combined fault name
        fault_names = [fault.name for fault in target_faults]
        combined_name = '_'.join(fault_names)

        # Scale the slip values
        combined_fault.slip *= slip_factor

        # Convert patches to vertices
        combined_fault.setVerticesFromPatches()
        combined_fault.numpatch = combined_fault.Faces.shape[0]
        
        # Compute triangle areas, moments, moment tensor and magnitude
        combined_fault.compute_patch_areas()
        fault_processor = faultpp(combined_name, combined_fault, mu, 
                                lon0=lon0, lat0=lat0, utmzone=utmzone)
        fault_processor.computeMoments()
        fault_processor.computeMomentTensor()
        fault_processor.computeMagnitude()

        # Store processor for later use
        self.tripproc = fault_processor
        
        return {
            'total_moment': fault_processor.Mo,
            'total_magnitude': fault_processor.Mw,
            'mode': 'csi',
            'processor': fault_processor,
            'faults': [fault.name for fault in target_faults],
            'slip_factor': slip_factor,
            'unit_context': unit_info or {},
        }
    
    def _calculate_moment_magnitude_hankel(self, target_faults, mu=3.e10, slip_factor=1.0, unit_info=None):
        """
        Calculate moment magnitude using the standard Hankel formula.
        
        Uses the relationship: Mw = 2/3 * (log10(Mo) - 9.1)
        where Mo is the seismic moment in N·m.
        
        Args:
            target_faults (list): List of fault objects to process
            mu (float, optional): Shear modulus in Pa. Defaults to 3.e10.
            slip_factor (float, optional): Factor to scale slip values. Defaults to 1.0.
        
        Returns:
            dict: Dictionary containing:
                - 'fault_moments': Dictionary with individual fault moment data
                - 'total_moment': Total seismic moment in N·m
                - 'total_magnitude': Total moment magnitude
                - 'mode': 'hankel'
                - 'mu': Shear modulus used
                - 'slip_factor': Slip scaling factor used
                - 'faults': List of fault names processed
        """
        fault_moments = {}
        total_moment = 0
        
        for ifault in target_faults:
            if hasattr(ifault, 'slip') and ifault.slip is not None:
                # Calculate total slip magnitude
                if ifault.slip.shape[1] >= 2:
                    # For strike-slip and dip-slip components
                    total_slip = np.sqrt(ifault.slip[:, 0]**2 + ifault.slip[:, 1]**2) * slip_factor
                else:
                    # For single slip component
                    total_slip = np.abs(ifault.slip[:, 0]) * slip_factor
                
                # Calculate seismic moment: Mo = μ * A * D
                area_getter = getattr(ifault, 'get_patch_areas', None)
                if callable(area_getter):
                    patch_areas = area_getter()
                else:
                    # Compatibility with CSI versions predating the
                    # current-value getter. Keep the branch explicit so
                    # ``compute_patch_areas`` is not evaluated eagerly.
                    patch_areas = getattr(ifault, 'area', None)
                    if patch_areas is None or np.asarray(patch_areas).size == 0:
                        patch_areas = ifault.compute_patch_areas()
                areas = np.array(patch_areas) * 1e6  # Convert km^2 to m^2
                moment = np.sum(mu * areas * total_slip)
                
                # Calculate moment magnitude using Hankel formula
                moment_magnitude = 2.0 / 3.0 * (np.log10(moment) - 9.1)
                
                # Store individual fault data
                fault_moments[ifault.name] = {
                    'moment': moment,
                    'magnitude': moment_magnitude,
                    'mean_slip': np.mean(total_slip),
                    'max_slip': np.max(total_slip)
                }
                
                total_moment += moment
        
        # Calculate total magnitude
        total_magnitude = 2.0 / 3.0 * (np.log10(total_moment) - 9.1) if total_moment > 0 else 0.0
        
        return {
            'fault_moments': fault_moments,
            'total_moment': total_moment,
            'total_magnitude': total_magnitude,
            'mode': 'hankel',
            'mu': mu,
            'slip_factor': slip_factor,
            'unit_context': unit_info or {},
            'faults': [fault.name for fault in target_faults]
        }
    
    def print_moment_magnitude(self, faults=None, mu=3.e10, slip_factor=None, mode='hankel'):
        """
        Print seismic moment and moment magnitude information in a formatted table.
        
        This method calculates and displays the seismic moment and magnitude information
        for the specified faults using either CSI or Hankel calculation methods.
        
        Args:
            faults (list, optional): List of fault objects or fault names to analyze.
                                   If None, uses all available faults. Defaults to None.
            mu (float, optional): Shear modulus in Pa. Defaults to 3.e10.
            slip_factor (float, optional): Factor to convert stored slip values to meters.
                If omitted, inferred from units.observation for displacement units.
            mode (str, optional): Calculation mode ('csi' or 'hankel'). Defaults to 'hankel'.
        
        Returns:
            dict: Results dictionary from calculate_moment_magnitude method
        """
        # Calculate moment and magnitude
        results = self.calculate_moment_magnitude(faults=faults, mu=mu, 
                                                slip_factor=slip_factor, mode=mode)
        
        fault_list_str = ', '.join(results['faults']) if 'faults' in results else 'All faults'
        
        print("\n" + "="*60)
        print(f"Seismic Moment and Magnitude ({mode.upper()} mode)")
        print(f"Analyzed faults: {fault_list_str}")
        print("="*60)
        unit_context = results.get('unit_context') or {}
        if unit_context.get("observation"):
            assumed = " (assumed)" if unit_context.get("assumed") else ""
            print(f"Slip unit: {unit_context['observation']}{assumed}; converted to meters for Mo")
        
        if mode == 'csi':
            # Simple output for CSI mode
            print(f"Mo is: {results['total_moment']:.8e} N·m")
            print(f"Mw is: {results['total_magnitude']:.2f}")
            print(f"Shear modulus: {mu:.2e} Pa")
            print(f"Slip scaling factor: {results['slip_factor']:.6g}")
            
        elif mode == 'hankel':
            # Detailed table output for Hankel mode
            if 'fault_moments' in results:
                table_data = []
                for fault_name, fault_data in results['fault_moments'].items():
                    table_data.append([
                        fault_name,
                        f"{fault_data['moment']:.3e}",
                        f"{fault_data['magnitude']:.2f}",
                        f"{fault_data['mean_slip']:.4f}",
                        f"{fault_data['max_slip']:.4f}"
                    ])
                
                # Add total row
                table_data.append([
                    'TOTAL',
                    f"{results['total_moment']:.3e}",
                    f"{results['total_magnitude']:.2f}",
                    '-',
                    '-'
                ])
                
                headers = ['Fault Name', 'Moment (N·m)', 'Magnitude', 'Mean Slip (m)', 'Max Slip (m)']
                print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='left'))
            
            print(f"\nParameters:")
            print(f"Shear modulus: {results['mu']:.2e} Pa")
            print(f"Slip scaling factor: {results['slip_factor']:.6g}")
            print(f"Formula: Mw = 2/3 * (log10(Mo) - 9.1)")
        
        print("="*60)
        
        return results

    def get_faults_summary(self, faults=None, fault_groups=None, mu=None,
                           slip_factor=None, include_slip=True, include_moment=True):
        """
        Return structured geometry, slip, and moment summaries for selected faults.

        This is the public summary API for inversion objects. Use
        ``print_faults_summary`` or ``show_faults_summary`` for formatted
        interactive output.
        """
        target_faults = self._select_faults(faults)
        if mu is None:
            mu = getattr(self, 'shear_modulus', 3.e10)
        if fault_groups is None:
            fault_groups = self._default_fault_groups(target_faults)
        summary_context = self._resolve_summary_slip_context(slip_factor, include_moment)

        summary = summarize_faults(
            target_faults,
            fault_groups=fault_groups,
            mu=mu,
            slip_factor=summary_context["slip_factor"],
            slip_unit_label=summary_context["slip_unit_label"],
            moment_slip_factor=summary_context["moment_slip_factor"],
            moment_kind=summary_context["moment_kind"],
            moment_unit_label=summary_context["moment_unit_label"],
            equivalent_duration_years=summary_context["equivalent_duration_years"],
            include_slip=include_slip,
            include_moment=summary_context["include_moment"],
        )
        summary["unit_context"] = summary_context["unit_info"]
        return summary

    def get_fault_summary(self, *args, **kwargs):
        """Alias for ``get_faults_summary`` for interactive use."""
        return self.get_faults_summary(*args, **kwargs)

    def print_faults_summary(self, faults=None, fault_groups=None, mu=None,
                             slip_factor=None, include_slip=True,
                             include_moment=True, file=None, tablefmt='grid'):
        """
        Print and return fault summaries for selected faults.
        """
        target_faults = self._select_faults(faults)
        if mu is None:
            mu = getattr(self, 'shear_modulus', 3.e10)
        if fault_groups is None:
            fault_groups = self._default_fault_groups(target_faults)
        summary_context = self._resolve_summary_slip_context(slip_factor, include_moment)

        summary = print_faults_summary(
            target_faults,
            fault_groups=fault_groups,
            mu=mu,
            slip_factor=summary_context["slip_factor"],
            slip_unit_label=summary_context["slip_unit_label"],
            moment_slip_factor=summary_context["moment_slip_factor"],
            moment_kind=summary_context["moment_kind"],
            moment_unit_label=summary_context["moment_unit_label"],
            equivalent_duration_years=summary_context["equivalent_duration_years"],
            include_slip=include_slip,
            include_moment=summary_context["include_moment"],
            file=file,
            tablefmt=tablefmt,
        )
        summary["unit_context"] = summary_context["unit_info"]
        return summary

    def print_fault_summary(self, *args, **kwargs):
        """Alias for ``print_faults_summary`` for interactive use."""
        return self.print_faults_summary(*args, **kwargs)

    def show_faults_summary(self, *args, **kwargs):
        """Alias for ``print_faults_summary`` for interactive use."""
        return self.print_faults_summary(*args, **kwargs)

    def _print_fault_statistics(self, faults=None, fault_groups=None):
        """
        Print comprehensive fault statistics in formatted tables.
        
        This method displays:
        1. Fault geometry statistics (strike, dip, patches, length)
        2. Slip statistics (strike-slip, dip-slip, total slip)
        3. Seismic moment and magnitude calculations with grouping support
        
        Args:
            faults (list, optional): List of fault objects or fault names to analyze.
                                   If None, uses all available faults. Defaults to None.
            fault_groups (list, optional): List of fault groups for moment calculation.
                                         Each group should be a list of fault names.
                                         If None, treats each fault as its own group.
                                         Defaults to None.
        """
        return self.print_faults_summary(faults=faults, fault_groups=fault_groups)
        
    def calculate_data_fit_metrics(self, data, vertical=True):
        """
        Calculate RMS and VR for different data types.
        
        Parameters:
        -----------
        data : csi data object
            GPS, InSAR, or optical correlation data object
        vertical : bool
            Whether to include vertical component (for GPS data)
            
        Returns:
        --------
        tuple : (rms, vr)
            Root Mean Square error and Variance Reduction percentage
        """
        observed, synthetic = data_fit_vectors(data, vertical=vertical)
        metrics = fit_metrics_from_vectors(observed, synthetic)
        return metrics["rms"], metrics["vr"]

    def _get_fit_geodata_config(self):
        """Return configured data, vertical flags and poly settings."""
        specs = get_geodata_prediction_specs(self)
        return (
            [spec.data for spec in specs],
            [spec.vertical for spec in specs],
            [spec.configured_poly for spec in specs],
        )

    def collect_fit_statistics(
        self,
        *,
        model="median",
        data_poly="config",
        include_dataset=True,
        include_global=True,
        include_dataset_average=False,
        include_weighted=False,
        rebuild_synth=True,
        faults=None,
    ):
        """Collect RMS/VR diagnostics without changing the numerical formula.

        Dataset rows reuse the legacy ``calculate_data_fit_metrics`` vector
        definitions.  The global solver row, when available, is computed from
        the assembled solver vector, e.g. ``G @ mpost - d`` for BLSE.

        When ``include_weighted`` is true and the active inversion exposes a
        prepared covariance metric plus the sigma scale used for each data
        set, dataset rows also contain ``Qw`` and whitened RMS diagnostics.
        Collection never refactorizes covariance or changes the active model.
        """
        rows = []
        data_objects, verticals, polys = self._get_fit_geodata_config()
        target_faults = self._select_faults(faults)
        weight_context = self._fit_weight_context() if include_weighted else None

        if include_dataset:
            for data, vertical, config_poly in zip(data_objects, verticals, polys):
                resolved_poly = resolve_data_poly(config_poly, requested=data_poly)
                if rebuild_synth:
                    data.buildsynth(target_faults, direction="sd", poly=resolved_poly, vertical=vertical)
                observed, synthetic = data_fit_vectors(data, vertical=vertical)
                metrics = fit_metrics_from_vectors(observed, synthetic)
                weighted = {}
                sigma_group = None
                if weight_context is not None:
                    data_name = getattr(data, "name", None)
                    metric = weight_context["metrics"].get(data_name)
                    sigma = weight_context["sigmas"].get(data_name)
                    sigma_group = weight_context["groups"].get(data_name)
                    if metric is not None and sigma is not None:
                        group_members = weight_context["group_members"].get(
                            sigma_group, []
                        )
                        effective_dof = None
                        # VCE effective degrees of freedom belong to a group.
                        # They are attached to a dataset row only when the
                        # group contains that dataset alone.
                        if len(group_members) == 1:
                            effective_dof = weight_context["group_dofs"].get(
                                sigma_group
                            )
                        weighted = weighted_fit_metrics_from_residual(
                            synthetic - observed,
                            metric,
                            sigma=sigma,
                            effective_dof=effective_dof,
                        )
                rows.append(
                    {
                        "scope": "dataset",
                        "model": model,
                        "dataset": getattr(data, "name", None),
                        "data_type": getattr(data, "dtype", None),
                        "vertical": bool(vertical),
                        "poly": resolved_poly,
                        "sigma_group": sigma_group,
                        **metrics,
                        **weighted,
                    }
                )

        if include_dataset_average and rows:
            dataset_rows = [row for row in rows if row.get("scope") == "dataset"]
            if dataset_rows:
                rows.append(
                    {
                        "scope": "dataset_average",
                        "model": model,
                        "dataset": None,
                        "data_type": None,
                        "vertical": None,
                        "poly": None,
                        "rms": float(np.mean([row["rms"] for row in dataset_rows])),
                        "vr": float(np.mean([row["vr"] for row in dataset_rows])),
                        "ss_res": float(np.sum([row["ss_res"] for row in dataset_rows])),
                        "ss_obs": float(np.sum([row["ss_obs"] for row in dataset_rows])),
                        "n_observations": int(np.sum([row["n_observations"] for row in dataset_rows])),
                    }
                )

        if include_global:
            global_row = self._collect_global_solver_fit_statistics(model=model)
            if global_row is not None:
                rows.append(global_row)

        return rows

    def _fit_weight_context(self):
        """Return the active, report-only covariance/sigma mapping.

        Solvers publish this context only after a concrete model has been
        activated.  Absence means that raw RMS/VR remain available but a
        covariance-aware report would risk using stale or guessed weights.
        """
        metrics = getattr(self, "data_covariance_metrics", None)
        sigmas = getattr(self, "current_data_sigmas", None)
        if not isinstance(metrics, Mapping) or not isinstance(sigmas, Mapping):
            return None
        groups = getattr(self, "current_data_sigma_groups", {})
        group_members = getattr(self, "current_data_sigma_group_members", {})
        group_dofs = getattr(self, "current_data_effective_dof", {})
        return {
            "metrics": metrics,
            "sigmas": sigmas,
            "groups": groups if isinstance(groups, Mapping) else {},
            "group_members": (
                group_members if isinstance(group_members, Mapping) else {}
            ),
            "group_dofs": group_dofs if isinstance(group_dofs, Mapping) else {},
        }

    def _clear_fit_weight_context(self):
        """Invalidate report-only weights before activating another result.

        Covariance factors remain reusable scientific state.  Sigma/group/dof
        mappings, however, describe one concrete active model and must not
        survive a new solve, a data reassembly, or activation of a nonphysical
        posterior ``std`` summary.
        """
        self.current_data_sigmas = {}
        self.current_data_weights = {}
        self.current_data_sigma_groups = {}
        self.current_data_sigma_group_members = {}
        self.current_data_effective_dof = {}

    def _collect_global_solver_fit_statistics(self, *, model="median"):
        """Return the assembled solver-vector fit row when dimensions match."""
        candidates = []
        if all(hasattr(self, name) for name in ("G", "mpost", "d")):
            candidates.append(("global_solver_vector", getattr(self, "G"), getattr(self, "mpost"), getattr(self, "d")))
        if all(hasattr(self, name) for name in ("G_combined", "mpost", "observations")):
            candidates.append(
                (
                    "global_solver_vector",
                    getattr(self, "G_combined"),
                    getattr(self, "mpost"),
                    getattr(self, "observations"),
                )
            )
        for scope, G, mpost, data_vector in candidates:
            if G is None or mpost is None or data_vector is None:
                continue
            m_array = np.asarray(mpost, dtype=float).reshape(-1)
            d_array = np.asarray(data_vector, dtype=float).reshape(-1)
            G_shape = getattr(G, "shape", None)
            if G_shape is None or len(G_shape) != 2:
                continue
            if int(G_shape[1]) != m_array.size or int(G_shape[0]) != d_array.size:
                continue
            metrics = solver_fit_metrics(G, m_array, d_array)
            return {
                "scope": scope,
                "model": model,
                "dataset": None,
                "data_type": "assembled",
                "vertical": None,
                "poly": None,
                **metrics,
            }
        return None

    @staticmethod
    def fit_statistics_to_dataframe(rows: Sequence[Mapping]):
        """Return a compact pandas DataFrame for fit-statistics rows."""
        return fit_statistics_rows_to_dataframe(rows)

    @staticmethod
    def format_fit_statistics_report(rows: Sequence[Mapping], *, model=None):
        """Return a compact text report for fit-statistics rows."""
        return format_fit_statistics_report(rows, model=model)

    def write_fit_statistics_report(
        self,
        outdir="output",
        rows: Sequence[Mapping] | None = None,
        *,
        model="median",
        data_poly="config",
        include_dataset=True,
        include_global=True,
        include_dataset_average=False,
        include_weighted=False,
        rebuild_synth=True,
        basename="fit_statistics",
        formats=("txt", "tsv"),
    ):
        """Write RMS/VR diagnostics to text/table files."""
        if rows is None:
            rows = self.collect_fit_statistics(
                model=model,
                data_poly=data_poly,
                include_dataset=include_dataset,
                include_global=include_global,
                include_dataset_average=include_dataset_average,
                include_weighted=include_weighted,
                rebuild_synth=rebuild_synth,
            )
        return write_fit_statistics_report_files(rows, outdir, basename=basename, formats=formats, model=model)
    
    def calculate_and_print_fit_statistics(self, model='median'):
        """
        Calculate and print fit statistics for all datasets.
        
        Parameters:
        -----------
        model : str
            Model type to use ('median', 'mean', 'MAP', etc.) or BLSE for BLSE model.
        """
        rows = self.collect_fit_statistics(
            model=model,
            data_poly="config",
            include_dataset=True,
            include_global=False,
            include_weighted=True,
            rebuild_synth=True,
        )
        print("\n" + format_fit_statistics_table(rows, model=model))

    def plot_multifaults_slip(self, faults=None, figsize=(None, None), slip='total', 
                             cmap='precip3_16lev_change.cpt', norm=None, show=True, savefig=False, 
                             ftype='pdf', dpi=600, bbox_inches=None, method='cdict', N=None, 
                             drawCoastlines=False, plot_on_2d=True, style=['notebook'], 
                             cbaxis=[0.1, 0.2, 0.1, 0.02], cblabel='', xlabelpad=None, 
                             ylabelpad=None, zlabelpad=None, xtickpad=None, ytickpad=None, 
                             ztickpad=None, elevation=None, azimuth=None, shape=(1.0, 1.0, 1.0), 
                             zratio=None, plotTrace=True, depth=None, zticks=None, map_expand=0.2, 
                             fault_expand=0.1, plot_faultEdges=False, faultEdges_color='k', 
                             faultEdges_linewidth=1.0, suffix='', outdir=None, remove_direction_labels=False, 
                             zaxis_position='bottom-left', show_grid=True, grid_color='#bebebe',
                             background_color='white', axis_color=None, cbticks=None, 
                             cblinewidth=None, cbfontsize=None, cb_label_side='opposite',
                             map_cbaxis=None):
        """
        Plot the slip distribution of multiple faults.
        
        This method creates a comprehensive visualization of slip distribution across
        multiple fault surfaces, combining them if necessary and providing both 2D
        map and 3D fault plane visualizations.
        
        Args:
            faults (list, optional): List of fault objects or fault names to plot.
                                   If None, uses all available faults. Defaults to None.
            figsize (tuple, optional): Size of the figure and map. Defaults to (None, None).
            slip (str, optional): Type of slip to plot ('total', 'strike', 'dip'). 
                                Defaults to 'total'.
            cmap (str, optional): Colormap to use. Defaults to 'precip3_16lev_change.cpt'.
            norm (optional): Normalization for the colormap. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to True.
            savefig (bool, optional): Whether to save the figure. Defaults to False.
            ftype (str, optional): File type for saving the figure. Defaults to 'pdf'.
            dpi (int, optional): Dots per inch for the saved figure. Defaults to 600.
            bbox_inches (optional): Bounding box in inches for saving the figure. 
                                  Defaults to None.
            method (str, optional): Method for getting the colormap. Defaults to 'cdict'.
            N (int, optional): Number of colors in the colormap. Defaults to None.
            drawCoastlines (bool, optional): Whether to draw coastlines. Defaults to False.
            plot_on_2d (bool, optional): Whether to plot on a 2D map. Defaults to True.
            style (list, optional): Style for the plot. Defaults to ['notebook'].
            cbaxis (list, optional): Colorbar axis position. Defaults to [0.1, 0.2, 0.1, 0.02].
            cblabel (str, optional): Label for the colorbar. Defaults to ''.
            xlabelpad, ylabelpad, zlabelpad (float, optional): Padding for the axis labels. 
                                                             Defaults to None.
            xtickpad, ytickpad, ztickpad (float, optional): Padding for the axis ticks. 
                                                          Defaults to None.
            elevation, azimuth (float, optional): Elevation and azimuth angles for the 3D plot. 
                                                Defaults to None.
            shape (tuple, optional): Shape of the 3D plot. Defaults to (1.0, 1.0, 1.0).
            zratio (float, optional): Ratio for the z-axis. Defaults to None.
            plotTrace (bool, optional): Whether to plot the fault trace. Defaults to True.
            depth (float, optional): Depth for the z-axis. Defaults to None.
            zticks (list, optional): Ticks for the z-axis. Defaults to None.
            map_expand (float, optional): Expansion factor for the map. Defaults to 0.2.
            fault_expand (float, optional): Expansion factor for the fault. Defaults to 0.1.
            plot_faultEdges (bool, optional): Whether to plot the fault edges. Defaults to False.
            faultEdges_color (str, optional): Color for the fault edges. Defaults to 'k'.
            faultEdges_linewidth (float, optional): Line width for the fault edges. 
                                                  Defaults to 1.0.
            suffix (str, optional): Suffix for the saved figure filename. Defaults to ''.
            outdir (str, optional): Output directory for saving the figure. Defaults to None.
            remove_direction_labels (bool, optional): If True, remove E, N, S, W from axis labels. 
                                                    Defaults to False.
            zaxis_position (str, optional): Position of the z-axis ('bottom-left', 'top-right'). 
                                          Defaults to 'bottom-left'.
            show_grid (bool, optional): Whether to show grid lines. Defaults to True.
            grid_color (str, optional): Color of the grid lines. Defaults to '#bebebe'.
            background_color (str, optional): Background color of the plot. Defaults to 'white'.
            axis_color (str, optional): Color of the axes. Defaults to None.
            cbticks (list, optional): List of ticks to set on the colorbar. Defaults to None.
            cblinewidth (int, optional): Width of the colorbar label border and tick lines. 
                                       Defaults to None.
            cbfontsize (int, optional): Font size of the colorbar label. Defaults to None.
            cb_label_side (str, optional): Position of the label relative to the ticks 
                                         ('opposite' or 'same'). Defaults to 'opposite'.
            map_cbaxis (optional): Axis for the colorbar on the map plot. Defaults to None.
        
        Returns:
            None
        """
        from ..viztools import plot_slip_distribution
        
        # Handle faults parameter
        if faults is None:
            target_faults = self._get_faults()
        elif isinstance(faults, list):
            if len(faults) > 0:
                if isinstance(faults[0], str):
                    # List of fault names
                    all_faults = self._get_faults()
                    fault_dict = {fault.name: fault for fault in all_faults}
                    target_faults = []
                    for fault_name in faults:
                        if fault_name in fault_dict:
                            target_faults.append(fault_dict[fault_name])
                        else:
                            print(f"Warning: Fault '{fault_name}' not found in available faults")
                else:
                    # List of fault objects
                    target_faults = faults
            else:
                target_faults = self._get_faults()
        else:
            print("Warning: Invalid faults parameter, using all available faults")
            target_faults = self._get_faults()

        if len(target_faults) > 1:
            # 是要判断所有fault的patchType是一样的，可选triangle或rectangle
            if all(fault.patchType == target_faults[0].patchType for fault in target_faults):
                # Combine faults if there are more than one
                combined_fault = target_faults[0].duplicateFault()
                combined_fault.name = 'Combined Fault'
                for fault in target_faults[1:]:
                    for ipatch, islip in zip(fault.patch, fault.slip):
                        combined_fault.N_slip = combined_fault.slip.shape[0] + 1
                        combined_fault.addpatch(ipatch, islip)
                combined_fault.setTrace(0.1)
                mfault = combined_fault
                add_faults = target_faults
            else:
                mfault = target_faults
                add_faults = target_faults
        else:
            # Directly plot if there is only one fault
            mfault = target_faults[0]
            add_faults = [mfault]

        # Plot the slip distribution using the external plotting function
        plot_slip_distribution(mfault, add_faults=add_faults, slip=slip, cmap=cmap, norm=norm, 
                             figsize=figsize, method=method, N=N, drawCoastlines=drawCoastlines, 
                             plot_on_2d=plot_on_2d, cbaxis=cbaxis, cblabel=cblabel, show=show, 
                             savefig=savefig, ftype=ftype, dpi=dpi, bbox_inches=bbox_inches,
                             remove_direction_labels=remove_direction_labels, cbticks=cbticks, 
                             cblinewidth=cblinewidth, cbfontsize=cbfontsize, 
                             cb_label_side=cb_label_side, map_cbaxis=map_cbaxis, style=style,
                             xlabelpad=xlabelpad, ylabelpad=ylabelpad, zlabelpad=zlabelpad, 
                             xtickpad=xtickpad, ytickpad=ytickpad, ztickpad=ztickpad, 
                             elevation=elevation, azimuth=azimuth, shape=shape, zratio=zratio, 
                             plotTrace=plotTrace, depth=depth, zticks=zticks, 
                             map_expand=map_expand, fault_expand=fault_expand, 
                             plot_faultEdges=plot_faultEdges, faultEdges_color=faultEdges_color,
                             faultEdges_linewidth=faultEdges_linewidth, suffix=suffix, outdir=outdir,
                             show_grid=show_grid, grid_color=grid_color, 
                             background_color=background_color, axis_color=axis_color,
                             zaxis_position=zaxis_position)

        # All Done
        return
    
# EOF
