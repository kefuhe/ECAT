"""
Public interseismic-field methods for BLSE and Bayesian inversion classes.

The mixin exposes calculation, plotting, and GMT export helpers while keeping
the numerical work in :mod:`eqtools.csiExtend.interseismic_fields`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .config.config_utils import get_observation_unit_info
from .interseismic_fields import (
    calculate_interseismic_fields as _calculate_interseismic_fields,
    calculate_tectonic_loading_rate as _calculate_tectonic_loading_rate,
    get_fault_by_name,
    get_interseismic_field_values,
    normalize_interseismic_field,
    normalize_slip_component,
    resolve_patch_indices,
    summarize_values,
    _get_transform_indices,
    _get_solution_vector,
    _get_source_param_names,
    _get_source_start,
)
from .interseismic_parameter_model import (
    build_loading_linear_terms,
    get_blocks_config,
    get_fault_loading_config,
    resolve_transform_columns,
)
from .euler_inequality_constraints import (
    calculate_euler_matrix_for_points,
    convert_euler_pole_to_vector,
    filter_cap_patch_indices_for_hard_constraints,
    normalize_cap_hard_overlap_policy,
    normalize_interseismic_backslip_state,
    project_euler_to_strike,
)
from .patch_indices import select_patch_indices


class InterseismicKinematicsMixin:
    """Mixin adding interseismic loading, backslip, and coupling utilities.

    Host classes are expected to provide fault objects through ``_get_faults()``
    or ``faults``/``multifaults.faults``, plus a current linear solution in
    ``mpost``.  Bayesian host classes can pass ``model=...`` to first call
    ``returnModel(model=..., print_stat=False)``.
    """

    def calculate_interseismic_fields(
        self,
        fault_name,
        euler_params1=None,
        euler_params2=None,
        solution=None,
        slip_component="strikeslip",
        model=None,
        store=True,
    ):
        """Calculate tectonic loading, backslip, slip deficit, and coupling.

        Parameters
        ----------
        fault_name : str
            Name of the target fault.
        euler_params1, euler_params2 : array-like, optional
            Explicit Cartesian Euler vectors ``[wx, wy, wz]`` in radians/year
            for the two blocks.  If omitted, the vectors are resolved from the
            parsed ``interseismic_config.yml:fault_loading`` and current solution.
        solution : array-like, optional
            Linear solution vector.  Defaults to ``self.mpost``.
        slip_component : {"strikeslip", "dipslip", "total"}, default "strikeslip"
            Slip component used to compute backslip and coupling.  Aliases such
            as ``ss`` and ``ds`` are accepted.
        model : str or array-like, optional
            For Bayesian inversion objects, call ``returnModel(model=model,
            print_stat=False)`` before extracting the current solution.  Typical
            values are ``"median"``, ``"mean"`` and ``"MAP"``.
        store : bool, default True
            Store the result in ``self.interseismic_results`` and attach common
            compatibility attributes to the fault object.

        Returns
        -------
        dict
            ``{"fields": ..., "stats": ..., "metadata": ...}``.  Field arrays
            are per-patch and the method does not overwrite ``fault.slip``.
        """
        if model is not None:
            if not hasattr(self, "returnModel"):
                raise ValueError("model=... is only supported on objects with returnModel()")
            self.returnModel(model=model, print_stat=False)

        result = _calculate_interseismic_fields(
            self,
            fault_name,
            euler_params1=euler_params1,
            euler_params2=euler_params2,
            solution=solution,
            slip_component=slip_component,
        )

        if store:
            if not hasattr(self, "interseismic_results"):
                self.interseismic_results = {}
            store_key = (fault_name, result["metadata"]["slip_component"])
            self.interseismic_results[store_key] = result
            self.interseismic_results[fault_name] = result

            fault = get_fault_by_name(self, fault_name)
            fault.interseismic_fields = result["fields"]
            fault.interseismic_stats = result["stats"]
            fault.tectonic_loading_rate = result["fields"]["tectonic_loading_rate"]
            fault.backslip_rate = result["fields"]["backslip_rate"]
            fault.slip_deficit_signed = result["fields"]["slip_deficit_signed"]
            fault.slip_deficit_magnitude = result["fields"]["slip_deficit_magnitude"]
            fault.coupling_ratio = result["fields"]["coupling_ratio"]
            fault.coupling_magnitude = result["fields"]["coupling_magnitude"]
            fault.creep_rate_signed = result["fields"]["creep_rate_signed"]
            fault.creep_ratio = result["fields"]["creep_ratio"]

        return result

    def get_interseismic_field(
        self,
        fault_name,
        field,
        result=None,
        model=None,
        euler_params1=None,
        euler_params2=None,
        solution=None,
        slip_component="strikeslip",
    ):
        """Return one interseismic field array using public field aliases.

        Parameters
        ----------
        fault_name : str
            Target fault name.
        field : str
            Field name or alias, for example ``backslip_rate``,
            ``slip_deficit_magnitude``, ``coupling_ratio``,
            ``creep_rate_signed`` or ``loading``.
        result : dict, optional
            Precomputed result from ``calculate_interseismic_fields()``.
        model : str or array-like, optional
            Bayesian representative model passed to ``returnModel()`` if a
            result must be calculated.
        euler_params1, euler_params2 : array-like, optional
            Explicit Euler vectors ``[wx, wy, wz]`` in radians/year.
        solution : array-like, optional
            Linear solution vector.  Defaults to ``self.mpost``.
        slip_component : {"strikeslip", "dipslip", "total"}, default "strikeslip"
            Slip component used if the result must be calculated.
        Returns
        -------
        numpy.ndarray
            One value per patch.
        """
        result = self._get_or_calculate_interseismic_result(
            fault_name,
            result=result,
            model=model,
            euler_params1=euler_params1,
            euler_params2=euler_params2,
            solution=solution,
            slip_component=slip_component,
        )
        return get_interseismic_field_values(result, field)

    def calculate_tectonic_loading_rate(
        self,
        fault_name,
        euler_params1=None,
        euler_params2=None,
        solution=None,
        store=True,
    ):
        """Calculate long-term strike-parallel tectonic loading rate.

        Parameters
        ----------
        fault_name : str
            Target fault name.
        euler_params1, euler_params2 : array-like, optional
            Explicit Euler vectors ``[wx, wy, wz]`` in radians/year.  If omitted,
            block vectors are resolved from ``interseismic_config.fault_loading``.
        solution : array-like, optional
            Linear solution vector.  Defaults to ``self.mpost``.
        store : bool, default True
            Store the field as ``fault.tectonic_loading_rate``.

        Returns
        -------
        numpy.ndarray
            One loading-rate value per patch.
        """
        values = _calculate_tectonic_loading_rate(
            self,
            fault_name,
            euler_params1=euler_params1,
            euler_params2=euler_params2,
            solution=solution,
        )
        if store:
            get_fault_by_name(self, fault_name).tectonic_loading_rate = values
        return values

    def calculate_locking_degree(
        self,
        fault_name,
        euler_params1=None,
        euler_params2=None,
        method="absolute",
        slip_component="strikeslip",
        solution=None,
        store=True,
    ):
        """Deprecated legacy locking interface.

        The old ``locking`` names were ambiguous for direct-backslip
        interseismic inversions.  Use ``calculate_interseismic_fields()`` and
        read ``slip_deficit_magnitude`` for a rate magnitude or
        ``coupling_ratio`` for signed coupling.
        """
        raise RuntimeError(
            "calculate_locking_degree() has been removed from the interseismic "
            "workflow because the legacy locking definitions were ambiguous. "
            "Use calculate_interseismic_fields() and read "
            "'slip_deficit_magnitude' or 'coupling_ratio' instead."
        )

    def get_interseismic_constraint_report(
        self,
        fault_name,
        solution=None,
        slip_component="strikeslip",
        selector=None,
        model=None,
        zero_tolerance=1.0e-12,
    ):
        """Return a diagnostic report for backslip bounds and Euler cap rows.

        The report does not modify the solver.  It is intended for checking the
        sign convention and whether the currently built inequality matrix still
        matches the configured patch selection after dynamic script updates.

        Parameters
        ----------
        fault_name : str
            Target fault name.
        solution : array-like, optional
            Linear solution vector.  Defaults to ``self.mpost``.
        slip_component : {"strikeslip", "dipslip"}, default "strikeslip"
            Slip component whose bounds are summarized.
        selector : dict or iterable of int, optional
            Patch selector for the diagnostic summary.  ``None`` uses the
            configured cap selector when available, otherwise all patches.
        model : str or array-like, optional
            Bayesian representative model passed to ``returnModel()`` before
            extracting the current solution.
        zero_tolerance : float, default 1e-12
            Threshold used to count near-zero loading values.

        Returns
        -------
        dict
            Diagnostic values and short interpretation strings.
        """
        if model is not None:
            if not hasattr(self, "returnModel"):
                raise ValueError("model=... is only supported on objects with returnModel()")
            self.returnModel(model=model, print_stat=False)

        component = normalize_slip_component(slip_component)
        if component == "total":
            raise ValueError("Constraint diagnostics require a signed component, not 'total'.")

        interseismic_config = getattr(getattr(self, "config", None), "interseismic_config", {})
        fault_loading = get_fault_loading_config(interseismic_config)
        params = fault_loading.get("faults", {}).get(fault_name)
        if not fault_loading.get("enabled", False) or params is None:
            raise ValueError(f"Interseismic fault_loading is not enabled for fault '{fault_name}'.")

        fault = get_fault_by_name(self, fault_name)
        if selector is None:
            fault_cap = interseismic_config.get("cap_constraints", {}).get("faults", {}).get(fault_name, {})
            selector = fault_cap.get("selector")
        else:
            fault_cap = interseismic_config.get("cap_constraints", {}).get("faults", {}).get(fault_name, {})
        max_coupling = float(fault_cap.get("max_coupling", 1.0))
        patch_indices = select_patch_indices(fault, selector, allow_none_all=True, unique=True, name="selector")
        result = self.calculate_interseismic_fields(
            fault_name,
            solution=solution,
            slip_component=component,
            store=False,
        )
        loading = result["fields"]["tectonic_loading_rate"][patch_indices]
        backslip = result["fields"]["backslip_rate"][patch_indices]

        negative = int(np.sum(loading < -zero_tolerance))
        positive = int(np.sum(loading > zero_tolerance))
        near_zero = int(loading.size - negative - positive)
        motion_sense = str(params.get("motion_sense", "dextral")).lower()
        cap_term = "b" if max_coupling == 1.0 else f"{max_coupling:g}*b"
        if motion_sense in ("dextral", "right_lateral", "right"):
            euler_cap_formula = f"q + {cap_term} <= 0"
            usual_loading_sign = "negative"
            physical_range = f"bounds q >= 0 plus Euler cap gives 0 <= q <= {-max_coupling:g}*b"
        elif motion_sense in ("sinistral", "left_lateral", "left"):
            euler_cap_formula = f"q + {cap_term} >= 0"
            usual_loading_sign = "positive"
            physical_range = f"bounds q <= 0 plus Euler cap gives {-max_coupling:g}*b <= q <= 0"
        else:
            euler_cap_formula = "unknown"
            usual_loading_sign = "unknown"
            physical_range = "motion_sense is not recognized"

        bounds_report = self._summarize_interseismic_component_bounds(
            fault_name,
            component,
            patch_indices,
            solution=solution,
        )
        matrix_report = self._summarize_euler_constraint_matrix()
        warnings = []
        expected_rows = int(patch_indices.size)
        euler_rows = matrix_report.get("euler_rows")
        if euler_rows is not None and euler_rows != expected_rows:
            warnings.append(
                "Euler-cap row count differs from the diagnostic patch count; "
                "rebuild cap constraints with update_euler_cap_constraint(..., reapply=True)."
            )
        if motion_sense in ("dextral", "right_lateral", "right") and positive > negative:
            warnings.append("Dextral/right-lateral loading is usually negative in ECAT's left-lateral-positive convention.")
        if motion_sense in ("sinistral", "left_lateral", "left") and negative > positive:
            warnings.append("Sinistral/left-lateral loading is usually positive in ECAT's left-lateral-positive convention.")

        return {
            "fault_name": fault_name,
            "component": component,
            "convention": "direct_backslip",
            "motion_sense": params.get("motion_sense", "dextral"),
            "reference_strike": params.get("reference_strike"),
            "block_types": list(params.get("block_types", [])),
            "blocks": list(params.get("blocks", params.get("blocks_standard", []))),
            "selected_patch_count": expected_rows,
            "selected_patches": patch_indices.tolist(),
            "max_coupling": max_coupling,
            "loading_stats": summarize_values(loading),
            "backslip_stats": summarize_values(backslip),
            "loading_sign_counts": {
                "negative": negative,
                "positive": positive,
                "near_zero": near_zero,
            },
            "usual_loading_sign": usual_loading_sign,
            "bounds": bounds_report,
            "euler_cap_formula": euler_cap_formula,
            "physical_range_hint": physical_range,
            "matrix": matrix_report,
            "warnings": warnings,
        }

    def print_interseismic_constraint_report(self, fault_name, **kwargs):
        """Print and return ``get_interseismic_constraint_report()`` output.

        Parameters
        ----------
        fault_name : str
            Target fault name.
        **kwargs
            Forwarded to ``get_interseismic_constraint_report()``.

        Returns
        -------
        dict
            The diagnostic report that was printed.
        """
        report = self.get_interseismic_constraint_report(fault_name, **kwargs)
        print(self._format_interseismic_constraint_report(report))
        return report

    def get_interseismic_preflight_report(
        self,
        solution=None,
        slip_component="strikeslip",
        model=None,
        zero_tolerance=1.0e-12,
    ):
        """Return a compact pre-inversion report for interseismic settings.

        The report does not modify the solver.  It summarizes the fault-loading
        convention, loading sign and magnitude, configured cap rows, and hard
        backslip/coupling rows so scripts can catch block-order or selector
        mistakes before a solve.

        Parameters
        ----------
        solution : array-like, optional
            Current or trial linear solution used when loading depends on
            estimated block parameters.  Defaults to ``self.mpost`` when
            available.
        slip_component : {"strikeslip", "dipslip"}, default "strikeslip"
            Slip component used for signed backslip diagnostics.
        model : str or array-like, optional
            Bayesian representative model passed to ``returnModel()`` before
            extracting the current solution.
        zero_tolerance : float, default 1e-12
            Threshold used to count near-zero loading values.

        Returns
        -------
        dict
            A compact per-fault diagnostic report.  Use
            ``print_interseismic_preflight_report()`` for readable text.
        """
        if model is not None:
            if not hasattr(self, "returnModel"):
                raise ValueError("model=... is only supported on objects with returnModel()")
            self.returnModel(model=model, print_stat=False)

        component = normalize_slip_component(slip_component)
        if component == "total":
            raise ValueError("Preflight diagnostics require a signed component, not 'total'.")

        interseismic_config = getattr(getattr(self, "config", None), "interseismic_config", {}) or {}
        fault_loading = get_fault_loading_config(interseismic_config)
        cap_config = interseismic_config.get("cap_constraints", {}) or {}
        configured_faults = list(
            fault_loading.get("configured_faults")
            or fault_loading.get("faults", {}).keys()
        )

        report = {
            "enabled": bool(fault_loading.get("enabled", False)),
            "component": component,
            "units": get_observation_unit_info(self, default="m/yr"),
            "convention": {
                "loading": "b = first configured block - second configured block, projected to local patch strike",
                "direct_backslip": "q",
                "slip_deficit_signed": "-q",
                "coupling_ratio": "-q / b",
                "creep_rate_signed": "b + q",
            },
            "faults": [],
            "blocks": [],
            "matrix": self._summarize_euler_constraint_matrix(),
            "warnings": [],
        }

        if not report["enabled"]:
            report["warnings"].append("interseismic fault_loading is not enabled")
            return report

        block_report = self._summarize_interseismic_blocks(interseismic_config)
        report["blocks"] = block_report["blocks"]
        report["warnings"].extend(block_report["warnings"])

        total_configured_cap_patch_count = 0
        total_active_cap_patch_count = 0
        for fault_name in configured_faults:
            params = fault_loading.get("faults", {}).get(fault_name)
            if params is None:
                continue
            fault = get_fault_by_name(self, fault_name)
            all_patch_indices = resolve_patch_indices(fault)
            loading_report = self._summarize_interseismic_preflight_loading(
                fault_name,
                all_patch_indices,
                solution=solution,
                zero_tolerance=zero_tolerance,
            )
            loading_regions_report = self._summarize_interseismic_loading_regions(
                fault_name,
                fault,
                params,
                solution=solution,
            )
            backslip_report = self._summarize_interseismic_backslip_selectors(
                fault_name,
                fault,
                interseismic_config.get("backslip_constraints", []),
            )
            cap_report = self._summarize_interseismic_cap_selector(
                fault_name,
                fault,
                cap_config,
                interseismic_config,
            )
            overlap_report = self._summarize_interseismic_selector_overlap(
                cap_report,
                backslip_report,
            )
            total_configured_cap_patch_count += int(cap_report.get("configured_patch_count", 0))
            total_active_cap_patch_count += int(cap_report["patch_count"])

            fault_warnings = []
            loading_warnings = self._interseismic_loading_sign_warnings(
                params.get("motion_sense", "dextral"),
                loading_report,
            )
            fault_warnings.extend(loading_warnings)
            fault_warnings.extend(loading_regions_report.get("warnings", []))
            if cap_report.get("selector_error"):
                fault_warnings.append(cap_report["selector_error"])
            elif (
                overlap_report["cap_with_hard_backslip"] > 0
                and cap_report.get("hard_overlap") != "skip"
            ):
                fault_warnings.append(
                    "cap and hard backslip/coupling constraints overlap; verify the duplicated selection is intentional"
                )
            if overlap_report["cap_with_free"] > 0:
                fault_warnings.append(
                    "state='free' does not disable Euler cap; remove these patches from cap if they should be unconstrained"
                )
            if overlap_report["duplicate_hard_backslip_patches"] > 0:
                fault_warnings.append(
                    "multiple hard backslip/coupling constraints target the same patch/component"
                )

            row = {
                "fault_name": fault_name,
                "patch_count": int(len(fault.patch)),
                "component": component,
                "block_types": list(params.get("block_types", [])),
                "blocks": list(params.get("blocks", params.get("blocks_standard", []))),
                "block_names": list(params.get("block_names", [])),
                "loading_expression": self._format_interseismic_loading_expression(params),
                "reference_strike": params.get("reference_strike"),
                "motion_sense": params.get("motion_sense", "dextral"),
                "loading": loading_report,
                "loading_regions": loading_regions_report,
                "cap_constraints": cap_report,
                "backslip_constraints": backslip_report,
                "overlap": overlap_report,
                "warnings": fault_warnings,
            }
            report["faults"].append(row)
            report["warnings"].extend(f"{fault_name}: {warning}" for warning in fault_warnings)

        euler_rows = report["matrix"].get("euler_rows")
        expected_cap_rows = sum(
            int(fault_report["cap_constraints"].get("expected_row_count", 0))
            for fault_report in report["faults"]
        )
        if cap_config.get("enabled", False) and euler_rows is not None and euler_rows != expected_cap_rows:
            report["warnings"].append(
                "Euler-cap matrix row count differs from active cap patch count; rebuild constraints after selector updates"
            )
        report["configured_cap_patch_count"] = int(total_configured_cap_patch_count)
        report["active_cap_patch_count"] = int(total_active_cap_patch_count)
        return report

    def print_interseismic_preflight_report(self, **kwargs):
        """Print and return ``get_interseismic_preflight_report()`` output."""
        report = self.get_interseismic_preflight_report(**kwargs)
        print(self._format_interseismic_preflight_report(report))
        return report

    def _summarize_interseismic_blocks(self, interseismic_config):
        """Summarize block-to-dataset Euler parameter binding."""
        blocks_config = get_blocks_config(interseismic_config)
        rows = []
        warnings = []
        dataset_to_block = {}
        mpost = getattr(self, "mpost", None)
        fallback_n = len(mpost) if mpost is not None else 0
        n_total = int(getattr(self, "lsq_parameters", fallback_n) or fallback_n)

        for block_name, block in (blocks_config.get("items", {}) or {}).items():
            euler = block.get("euler", {}) or {}
            mode = euler.get("mode")
            if not mode:
                source = euler.get("source")
                mode = {
                    "dataset": "estimate",
                    "euler_pole": "fixed_pole",
                    "euler_vector": "fixed_vector",
                }.get(source, str(source))
            datasets = [str(name) for name in list(block.get("datasets", euler.get("datasets", [])) or [])]
            anchor = euler.get("anchor_dataset")
            resolved = []
            missing = []

            for dataset in datasets:
                previous = dataset_to_block.setdefault(dataset, block_name)
                if previous != block_name:
                    warnings.append(
                        f"dataset '{dataset}' is assigned to multiple blocks: {previous}, {block_name}"
                    )
                try:
                    columns, owner = resolve_transform_columns(
                        self,
                        dataset,
                        owner_source=euler.get("owner_source"),
                        n_total=n_total or None,
                    )
                    resolved.append({
                        "dataset": dataset,
                        "owner_source": owner,
                        "columns": columns.tolist(),
                    })
                except Exception as exc:
                    missing.append({
                        "dataset": dataset,
                        "reason": str(exc),
                    })

            if mode == "estimate":
                if not datasets:
                    warnings.append(f"block '{block_name}' estimates Euler but has no datasets")
                for item in missing:
                    warnings.append(
                        f"block '{block_name}' estimates Euler but dataset '{item['dataset']}' has no eulerrotation transform"
                    )
                status = "estimate_shared" if len(datasets) > 1 else "estimate"
            elif mode in {"fixed_pole", "fixed_vector"}:
                if resolved:
                    status = "fixed_to_block"
                elif datasets:
                    status = "fixed_block_no_transform"
                    warnings.append(
                        f"block '{block_name}' is fixed and its datasets have no eulerrotation transform; "
                        "assuming their rigid block motion was removed before inversion"
                    )
                else:
                    status = "fixed_loading_only"
            else:
                status = str(mode)

            rows.append({
                "block": str(block_name),
                "datasets": datasets,
                "euler_mode": str(mode),
                "anchor_dataset": str(anchor) if anchor else None,
                "resolved_eulerrotation": resolved,
                "missing_eulerrotation": missing,
                "status": status,
            })

        return {"blocks": rows, "warnings": warnings}

    def _summarize_interseismic_preflight_loading(
        self,
        fault_name,
        patch_indices,
        solution=None,
        zero_tolerance=1.0e-12,
    ):
        """Summarize signed loading for all selected patches."""
        try:
            loading_all = self.calculate_tectonic_loading_rate(
                fault_name,
                solution=solution,
                store=False,
            )
            loading = np.asarray(loading_all, dtype=float)[np.asarray(patch_indices, dtype=int)]
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

        negative = int(np.sum(loading < -zero_tolerance))
        positive = int(np.sum(loading > zero_tolerance))
        near_zero = int(loading.size - negative - positive)
        return {
            "available": True,
            "stats": summarize_values(loading),
            "sign_counts": {
                "negative": negative,
                "positive": positive,
                "near_zero": near_zero,
            },
        }

    def _summarize_interseismic_loading_regions(self, fault_name, fault, params, solution=None):
        """Summarize optional manual loading-region overrides for one fault."""
        regions = list(params.get("loading_regions", []) or [])
        if not regions:
            return {
                "enabled": False,
                "regions": [],
                "default": {
                    "patch_count": int(len(fault.patch)),
                    "patches": list(range(len(fault.patch))),
                    "loading_expression": self._format_interseismic_loading_expression(params),
                    "reference_strike": params.get("reference_strike"),
                    "motion_sense": params.get("motion_sense"),
                },
                "overlap_patches": [],
                "warnings": [],
            }

        try:
            loading_all = np.asarray(
                self.calculate_tectonic_loading_rate(
                    fault_name,
                    solution=solution,
                    store=False,
                ),
                dtype=float,
            )
            loading_available = True
            loading_reason = None
        except Exception as exc:
            loading_all = None
            loading_available = False
            loading_reason = str(exc)

        assigned = {}
        overlap_patches = set()
        region_rows = []
        warnings = []
        for index, region in enumerate(regions):
            name = str(region.get("name", f"region_{index}"))
            try:
                patch_indices = select_patch_indices(
                    fault,
                    region.get("selector"),
                    allow_none_all=False,
                    unique=True,
                    name=f"loading region '{name}' selector for fault '{fault_name}'",
                )
                patches = [int(i) for i in patch_indices.tolist()]
            except Exception as exc:
                region_rows.append({
                    "name": name,
                    "available": False,
                    "reason": str(exc),
                    "patch_count": 0,
                    "patches": [],
                    "loading_expression": self._format_interseismic_loading_expression(region),
                    "reference_strike": region.get("reference_strike"),
                    "motion_sense": region.get("motion_sense"),
                })
                warnings.append(f"loading region '{name}' selector failed: {exc}")
                continue

            overlaps = [patch for patch in patches if patch in assigned]
            if overlaps:
                overlap_patches.update(overlaps)
                previous = sorted({assigned[patch] for patch in overlaps})
                warnings.append(
                    f"loading region '{name}' overlaps previous region(s) {previous} on patches {overlaps}"
                )
            for patch in patches:
                assigned.setdefault(patch, name)

            row = {
                "name": name,
                "available": True,
                "patch_count": int(len(patches)),
                "patches": patches,
                "loading_expression": self._format_interseismic_loading_expression(region),
                "reference_strike": region.get("reference_strike"),
                "motion_sense": region.get("motion_sense"),
                "block_types": list(region.get("block_types", [])),
                "blocks": list(region.get("blocks", region.get("blocks_standard", []))),
                "block_names": list(region.get("block_names", [])),
            }
            if loading_available:
                row["loading"] = {
                    "available": True,
                    "stats": summarize_values(loading_all[patch_indices]),
                }
            else:
                row["loading"] = {
                    "available": False,
                    "reason": loading_reason,
                }
            region_rows.append(row)

        all_patches = set(range(len(fault.patch)))
        default_patches = sorted(all_patches - set(assigned.keys()))
        default = {
            "patch_count": int(len(default_patches)),
            "patches": default_patches,
            "loading_expression": self._format_interseismic_loading_expression(params),
            "reference_strike": params.get("reference_strike"),
            "motion_sense": params.get("motion_sense"),
        }
        if loading_available:
            default["loading"] = {
                "available": True,
                "stats": summarize_values(loading_all[np.asarray(default_patches, dtype=int)]),
            }
        else:
            default["loading"] = {
                "available": False,
                "reason": loading_reason,
            }
        if overlap_patches:
            warnings.append("loading_regions overlap; build_loading_linear_terms will reject overlapping patches")

        return {
            "enabled": True,
            "regions": region_rows,
            "default": default,
            "overlap_patches": sorted(overlap_patches),
            "warnings": warnings,
        }

    def _summarize_interseismic_cap_selector(self, fault_name, fault, cap_config, interseismic_config):
        """Summarize configured Euler-cap selector for one fault."""
        enabled = bool(cap_config.get("enabled", False))
        fault_cap = (cap_config.get("faults", {}) or {}).get(fault_name)
        if not enabled or fault_cap is None:
            return {
                "enabled": False,
                "patch_count": 0,
                "configured_patch_count": 0,
                "active_patch_count": 0,
                "skipped_hard_patch_count": 0,
                "expected_row_count": 0,
                "patches": [],
                "configured_patches": [],
                "skipped_hard_patches": [],
                "selector": None,
                "max_coupling": None,
                "mode": None,
                "hard_overlap": None,
            }
        configured_indices = select_patch_indices(
            fault,
            fault_cap.get("selector"),
            allow_none_all=True,
            unique=True,
            name=f"cap selector for fault '{fault_name}'",
        )
        hard_overlap = normalize_cap_hard_overlap_policy(fault_cap.get("hard_overlap", "skip"))
        selector_error = None
        try:
            active_indices, overlap = filter_cap_patch_indices_for_hard_constraints(
                fault,
                interseismic_config,
                fault_name,
                configured_indices,
                component="strikeslip",
                hard_overlap=hard_overlap,
            )
        except ValueError as exc:
            selector_error = str(exc)
            active_indices = configured_indices
            hard_set = set(
                self._summarize_interseismic_backslip_selectors(
                    fault_name,
                    fault,
                    interseismic_config.get("backslip_constraints", []),
                )
                .get("hard_component_patches", {})
                .get("strikeslip", [])
            )
            skipped = [int(i) for i in configured_indices.tolist() if int(i) in hard_set]
            overlap = {
                "policy": hard_overlap,
                "configured_patch_count": int(configured_indices.size),
                "active_patch_count": int(active_indices.size),
                "hard_overlap_patch_count": int(len(skipped)),
                "skipped_hard_patch_count": 0,
                "hard_patches": sorted(hard_set),
                "overlap_patches": skipped,
                "skipped_hard_patches": [],
            }
        mode = str(fault_cap.get("mode", "motion_sense"))
        expected_row_count = int(active_indices.size) * (2 if mode == "loading_sign" else 1)
        return {
            "enabled": True,
            "patch_count": int(active_indices.size),
            "configured_patch_count": int(configured_indices.size),
            "active_patch_count": int(active_indices.size),
            "skipped_hard_patch_count": int(overlap.get("skipped_hard_patch_count", 0)),
            "hard_overlap_patch_count": int(overlap.get("hard_overlap_patch_count", 0)),
            "expected_row_count": expected_row_count,
            "patches": active_indices.tolist(),
            "configured_patches": configured_indices.tolist(),
            "skipped_hard_patches": overlap.get("skipped_hard_patches", []),
            "hard_overlap_patches": overlap.get("overlap_patches", []),
            "selector": fault_cap.get("selector"),
            "max_coupling": float(fault_cap.get("max_coupling", 1.0)),
            "mode": mode,
            "hard_overlap": hard_overlap,
            "selector_error": selector_error,
        }

    def _summarize_interseismic_backslip_selectors(self, fault_name, fault, constraints):
        """Summarize configured hard backslip/coupling selectors for one fault."""
        states = {}
        hard_component_hits = []
        hard_patch_set = set()
        free_patch_set = set()
        spec_count = 0
        row_count = 0

        for index, spec in enumerate(constraints or []):
            if spec.get("fault") != fault_name:
                continue
            spec_count += 1
            state = self._normalize_interseismic_backslip_state(spec.get("state"))
            patch_indices = select_patch_indices(
                fault,
                spec.get("selector"),
                allow_none_all=True,
                unique=True,
                name=f"backslip selector {index} for fault '{fault_name}'",
            )
            entry = states.setdefault(
                state,
                {
                    "spec_count": 0,
                    "row_count": 0,
                    "patches": set(),
                },
            )
            entry["spec_count"] += 1
            entry["patches"].update(int(i) for i in patch_indices)
            if state == "free":
                free_patch_set.update(int(i) for i in patch_indices)
                continue
            rows = int(patch_indices.size)
            entry["row_count"] += rows
            row_count += rows
            component = normalize_slip_component(spec.get("component", "strikeslip"))
            hard_component_hits.extend((int(i), component) for i in patch_indices)
            hard_patch_set.update(int(i) for i in patch_indices)

        normalized_states = {}
        for state, entry in states.items():
            normalized_states[state] = {
                "spec_count": int(entry["spec_count"]),
                "row_count": int(entry["row_count"]),
                "patch_count": int(len(entry["patches"])),
                "patches": sorted(entry["patches"]),
            }
        duplicate_count = int(len(hard_component_hits) - len(set(hard_component_hits)))
        hard_component_patches = {}
        for patch, component in hard_component_hits:
            hard_component_patches.setdefault(component, set()).add(patch)
        return {
            "spec_count": int(spec_count),
            "row_count": int(row_count),
            "states": normalized_states,
            "hard_patches": sorted(hard_patch_set),
            "hard_component_patches": {
                component: sorted(patches)
                for component, patches in hard_component_patches.items()
            },
            "free_patches": sorted(free_patch_set),
            "duplicate_hard_backslip_patches": duplicate_count,
        }

    def _summarize_interseismic_selector_overlap(self, cap_report, backslip_report):
        """Summarize overlap between cap and hard/free backslip selections."""
        active_cap_patches = set(cap_report.get("patches", []))
        configured_cap_patches = set(cap_report.get("configured_patches", cap_report.get("patches", [])))
        hard_patches = set(
            backslip_report.get("hard_component_patches", {}).get(
                "strikeslip",
                backslip_report.get("hard_patches", []),
            )
        )
        free_patches = set(backslip_report.get("free_patches", []))
        return {
            "cap_with_hard_backslip": int(len(configured_cap_patches & hard_patches)),
            "active_cap_with_hard_backslip": int(len(active_cap_patches & hard_patches)),
            "cap_with_free": int(len(active_cap_patches & free_patches)),
            "hard_backslip_with_free": int(len(hard_patches & free_patches)),
            "duplicate_hard_backslip_patches": int(
                backslip_report.get("duplicate_hard_backslip_patches", 0)
            ),
        }

    def _interseismic_loading_sign_warnings(self, motion_sense, loading_report):
        """Return compact warnings for loading sign mismatches."""
        if not loading_report.get("available"):
            return [f"loading stats unavailable: {loading_report.get('reason', 'unknown reason')}"]
        signs = loading_report.get("sign_counts", {})
        negative = int(signs.get("negative", 0))
        positive = int(signs.get("positive", 0))
        near_zero = int(signs.get("near_zero", 0))
        total = negative + positive + near_zero
        if total > 0 and near_zero == total:
            return ["loading is near zero on all checked patches"]

        key = str(motion_sense).lower()
        if key in ("dextral", "right_lateral", "right") and positive > negative:
            return ["dextral/right-lateral loading is usually negative in ECAT's left-lateral-positive convention"]
        if key in ("sinistral", "left_lateral", "left") and negative > positive:
            return ["sinistral/left-lateral loading is usually positive in ECAT's left-lateral-positive convention"]
        return []

    def _format_interseismic_loading_expression(self, params):
        """Format the ordered block expression used for loading."""
        blocks = list(params.get("block_names") or params.get("blocks") or params.get("blocks_standard", []))
        if len(blocks) == 2:
            return f"{blocks[0]} - {blocks[1]}"
        return "first configured block - second configured block"

    def _format_interseismic_preflight_report(self, report):
        """Format a concise human-readable interseismic preflight report."""
        lines = [
            "Interseismic preflight report",
            "  convention: b=first block - second block; q=backslip; slip_deficit=-q; coupling=-q/b",
        ]
        units = report.get("units") or {}
        unit = units.get("observation")
        if unit:
            assumed = " (assumed)" if units.get("assumed") else ""
            lines.append(
                f"  units: observation={unit}{assumed}; fixed Euler loading m/yr -> {unit}"
            )
        if not report.get("enabled"):
            lines.append("  fault_loading: disabled")
            for warning in report.get("warnings", []):
                lines.append(f"  warning: {warning}")
            return "\n".join(lines)

        matrix = report.get("matrix", {})
        if matrix.get("available") and matrix.get("euler_rows") is not None:
            lines.append(
                f"  Euler-cap rows: configured={report.get('configured_cap_patch_count', 0)}, matrix={matrix['euler_rows']}"
            )
        else:
            lines.append(
                f"  Euler-cap rows: configured={report.get('configured_cap_patch_count', 0)}, matrix=unavailable"
            )

        block_rows = report.get("blocks", [])
        if block_rows:
            lines.append("  blocks:")
            for block in block_rows:
                dataset_count = len(block.get("datasets") or [])
                resolved = len(block.get("resolved_eulerrotation", []) or [])
                missing = len(block.get("missing_eulerrotation", []) or [])
                lines.append(
                    (
                        f"    - {block['block']}: mode={block['euler_mode']}, "
                        f"datasets={dataset_count}, status={block['status']}, "
                        f"eulerrotation resolved/missing={resolved}/{missing}"
                    )
                )

        for fault_report in report.get("faults", []):
            lines.extend(self._format_one_interseismic_preflight_fault(fault_report))
        for warning in report.get("warnings", []):
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)

    def _format_one_interseismic_preflight_fault(self, fault_report):
        """Format one fault row in the preflight report."""
        lines = [
            (
                f"  {fault_report['fault_name']}: blocks={fault_report['loading_expression']}, "
                f"ref={fault_report['reference_strike']}, sense={fault_report['motion_sense']}, "
                f"patches={fault_report['patch_count']}"
            )
        ]
        loading = fault_report["loading"]
        if loading.get("available"):
            stats = loading["stats"]
            signs = loading["sign_counts"]
            lines.append(
                "    loading b: min={min:.6g}, median={median:.6g}, max={max:.6g}, signs neg/pos/zero={negative}/{positive}/{near_zero}".format(
                    negative=signs["negative"],
                    positive=signs["positive"],
                    near_zero=signs["near_zero"],
                    **stats,
                )
            )
        else:
            lines.append(f"    loading b: unavailable ({loading.get('reason', 'unknown reason')})")

        loading_regions = fault_report.get("loading_regions", {})
        if loading_regions.get("enabled"):
            default = loading_regions.get("default", {})
            lines.append(
                "    loading regions: {count}, default patches={default_count}".format(
                    count=len(loading_regions.get("regions", [])),
                    default_count=default.get("patch_count", 0),
                )
            )
            for region in loading_regions.get("regions", []):
                if not region.get("available", True):
                    lines.append(f"      - {region.get('name')}: unavailable ({region.get('reason')})")
                    continue
                region_loading = region.get("loading", {})
                if region_loading.get("available"):
                    median = region_loading["stats"]["median"]
                    median_text = f"{median:.6g}"
                else:
                    median_text = "unavailable"
                lines.append(
                    (
                        f"      - {region['name']}: patches={region['patch_count']}, "
                        f"blocks={region['loading_expression']}, ref={region['reference_strike']}, "
                        f"median={median_text}"
                    )
                )

        cap = fault_report["cap_constraints"]
        backslip = fault_report["backslip_constraints"]
        state_bits = []
        for state in ("full_coupling", "prescribed_coupling", "zero_backslip", "prescribed_backslip", "free"):
            info = backslip["states"].get(state)
            if info:
                state_bits.append(f"{state}={info['patch_count']}")
        state_text = ", ".join(state_bits) if state_bits else "none"
        overlap = fault_report["overlap"]
        if cap.get("enabled"):
            cap_text = (
                f"{cap['patch_count']}/{cap.get('configured_patch_count', cap['patch_count'])} "
                f"(k={cap.get('max_coupling') or 1:g}, skipped_hard={cap.get('skipped_hard_patch_count', 0)})"
            )
        else:
            cap_text = "0/0"
        lines.append(
            (
                f"    constraints: cap={cap_text}, "
                f"hard_rows={backslip['row_count']}, "
                f"states({state_text}), cap&hard={overlap['cap_with_hard_backslip']}, "
                f"cap&free={overlap['cap_with_free']}"
            )
        )
        return lines

    def _summarize_interseismic_component_bounds(self, fault_name, component, patch_indices, solution=None):
        """Summarize lb/ub for one signed slip component on selected patches."""
        try:
            sol = _get_solution_vector(self, solution)
            fault = get_fault_by_name(self, fault_name)
            param_names = list(_get_source_param_names(self, fault))
            comp_index = param_names.index(component)
            source_start = _get_source_start(self, fault_name, sol)
            n_patches = len(fault.patch)
            indices = source_start + comp_index * n_patches + np.asarray(patch_indices, dtype=int)
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

        lb, ub = None, None
        manager = getattr(self, "constraint_manager", None)
        for holder in (manager, self):
            if holder is None:
                continue
            try:
                lb = getattr(holder, "lb")
                ub = getattr(holder, "ub")
            except Exception:
                continue
            if lb is not None and ub is not None:
                break
        if lb is None or ub is None:
            return {"available": False, "reason": "No lb/ub arrays found on constraint_manager or inversion object."}

        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        if indices.size == 0:
            return {"available": True, "count": 0}
        if np.any(indices < 0) or np.any(indices >= len(lb)) or np.any(indices >= len(ub)):
            return {"available": False, "reason": "Selected component indices fall outside lb/ub arrays."}
        return {
            "available": True,
            "count": int(indices.size),
            "solution_indices_min": int(np.min(indices)),
            "solution_indices_max": int(np.max(indices)),
            "lb_stats": summarize_values(lb[indices]),
            "ub_stats": summarize_values(ub[indices]),
        }

    def _summarize_euler_constraint_matrix(self):
        """Return shape information for the current Euler-cap inequality group."""
        manager = getattr(self, "constraint_manager", None)
        group = None
        if manager is not None:
            group = getattr(manager, "_inequality_constraints", {}).get("euler_cap_constraints")
        if isinstance(group, Mapping):
            A = group.get("A")
            shape = tuple(group.get("shape", getattr(A, "shape", (None, None))))
            return {
                "available": True,
                "euler_rows": int(shape[0]) if shape[0] is not None else None,
                "euler_cols": int(shape[1]) if shape[1] is not None else None,
                "group_name": "euler_cap_constraints",
            }

        try:
            A = getattr(self, "A_ueq")
        except Exception:
            A = None
        if A is not None:
            return {
                "available": True,
                "euler_rows": None,
                "combined_inequality_shape": tuple(A.shape),
                "reason": "Only combined inequality matrix is available; Euler-cap rows cannot be isolated.",
            }
        return {"available": False, "reason": "No Euler-cap inequality group found."}

    def _format_interseismic_constraint_report(self, report):
        """Format a concise human-readable interseismic diagnostic report."""
        lines = [
            f"Interseismic constraint report: {report['fault_name']}",
            f"  component: {report['component']}",
            f"  convention: {report['convention']} (q=backslip, b=block/loading rate)",
            f"  motion_sense: {report['motion_sense']}",
            f"  reference_strike: {report['reference_strike']}",
            f"  selected patches: {report['selected_patch_count']}",
            f"  Euler cap: {report['euler_cap_formula']}",
            f"  usual loading sign: {report['usual_loading_sign']}",
            f"  physical range hint: {report['physical_range_hint']}",
        ]
        loading = report["loading_stats"]
        signs = report["loading_sign_counts"]
        lines.append(
            "  loading b: min={min:.6g}, max={max:.6g}, mean={mean:.6g}, signs neg/pos/zero={negative}/{positive}/{near_zero}".format(
                negative=signs["negative"],
                positive=signs["positive"],
                near_zero=signs["near_zero"],
                **loading,
            )
        )
        backslip = report["backslip_stats"]
        lines.append(
            "  backslip q: min={min:.6g}, max={max:.6g}, mean={mean:.6g}".format(**backslip)
        )
        bounds = report["bounds"]
        if bounds.get("available"):
            lines.append(
                "  bounds q: lb=[{lb_min:.6g}, {lb_max:.6g}], ub=[{ub_min:.6g}, {ub_max:.6g}]".format(
                    lb_min=bounds["lb_stats"]["min"],
                    lb_max=bounds["lb_stats"]["max"],
                    ub_min=bounds["ub_stats"]["min"],
                    ub_max=bounds["ub_stats"]["max"],
                )
            )
        else:
            lines.append(f"  bounds q: unavailable ({bounds.get('reason', 'unknown reason')})")
        matrix = report["matrix"]
        if matrix.get("available") and matrix.get("euler_rows") is not None:
            lines.append(f"  Euler-cap matrix group: rows={matrix['euler_rows']}, cols={matrix['euler_cols']}")
        elif matrix.get("available"):
            lines.append(f"  Euler-cap matrix group: {matrix.get('reason', 'available')}")
        else:
            lines.append(f"  Euler-cap matrix group: unavailable ({matrix.get('reason', 'unknown reason')})")
        for warning in report["warnings"]:
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)

    def add_interseismic_backslip_constraint(
        self,
        fault_name,
        state,
        selector=None,
        component="strikeslip",
        coupling=None,
        value=None,
        name=None,
        overwrite=True,
        source=None,
    ):
        """Add a hard equality constraint on direct backslip ``q``.

        Parameters
        ----------
        fault_name : str
            Target fault name.
        state : {"creep", "zero_backslip", "full_coupling", "prescribed_coupling", "prescribed_backslip", "free"}
            Constraint state.  ``creep``/``zero_backslip`` adds ``q = 0``.
            ``full_coupling`` adds ``q + b = 0``.  ``prescribed_coupling`` adds
            ``q + coupling * b = 0``.  ``prescribed_backslip`` adds ``q = value``.
            ``free`` is accepted as a no-op for script logic.
        selector : dict or iterable of int, optional
            Patch selector.  Supported mapping forms include ``{"edge": "top"}``,
            ``{"patches": [...]}``, ``{"depth_range": [...]}`` and
            ``{"trace_range": {...}}``.  ``None`` selects all patches.
        component : {"strikeslip", "dipslip"}, default "strikeslip"
            Slip component to constrain.  Coupling states require ``strikeslip``
            because Euler loading is projected along strike.
        coupling : float, optional
            Coupling fraction for ``state="prescribed_coupling"``.
        value : float, optional
            Backslip value for ``state="prescribed_backslip"``.
        name : str, optional
            Constraint group name.  A stable name is generated when omitted.
        overwrite : bool, default True
            Replace an existing equality constraint with the same name.
        source : str, optional
            Provenance label stored in the constraint manager.

        Returns
        -------
        dict
            Metadata describing the generated constraint.  The returned matrix
            is useful for tests and auditing.
        """
        state_key = self._normalize_interseismic_backslip_state(state)
        component_key = normalize_slip_component(component)
        if component_key == "total":
            raise ValueError("Backslip constraints require a signed component, not 'total'.")

        fault = get_fault_by_name(self, fault_name)
        patch_indices = select_patch_indices(
            fault,
            selector,
            allow_none_all=True,
            unique=True,
            name="selector",
        )
        if state_key == "free":
            return {
                "added": False,
                "reason": "state='free' does not generate constraint rows",
                "fault_name": fault_name,
                "state": state_key,
                "component": component_key,
                "patch_indices": patch_indices.tolist(),
            }
        if patch_indices.size == 0:
            return {
                "added": False,
                "reason": "selector matched zero patches",
                "fault_name": fault_name,
                "state": state_key,
                "component": component_key,
                "patch_indices": [],
            }

        Aeq, beq, formula = self._build_interseismic_backslip_equality(
            fault_name,
            state_key,
            patch_indices,
            component_key,
            coupling=coupling,
            value=value,
        )
        if name is None:
            safe_fault = str(fault_name).replace(" ", "_")
            safe_state = state_key.replace(" ", "_")
            name = f"interseismic_{safe_fault}_{safe_state}_{component_key}"
        if source is None:
            source = f"interseismic_backslip/{fault_name}"

        self._add_interseismic_equality_constraint(Aeq, beq, name, source=source, overwrite=overwrite)
        return {
            "added": True,
            "name": name,
            "source": source,
            "fault_name": fault_name,
            "state": state_key,
            "component": component_key,
            "patch_indices": patch_indices.tolist(),
            "formula": formula,
            "A": Aeq,
            "b": beq,
        }

    def _normalize_interseismic_backslip_state(self, state):
        """Normalize public interseismic backslip state aliases."""
        return normalize_interseismic_backslip_state(state)

    def _build_interseismic_backslip_equality(
        self,
        fault_name,
        state,
        patch_indices,
        component,
        coupling=None,
        value=None,
    ):
        """Build Aeq/beq rows for one interseismic backslip state."""
        n_total = int(getattr(self, "lsq_parameters"))
        fault = get_fault_by_name(self, fault_name)
        n_patches = len(fault.patch)
        dummy_solution = np.zeros(n_total, dtype=float)
        source_start = _get_source_start(self, fault_name, dummy_solution)
        param_names = list(_get_source_param_names(self, fault))
        if component not in param_names:
            raise ValueError(f"Fault '{fault_name}' has no '{component}' component in the current model")

        rows = np.arange(len(patch_indices), dtype=int)
        comp_index = param_names.index(component)
        slip_indices = source_start + comp_index * n_patches + np.asarray(patch_indices, dtype=int)
        Aeq = np.zeros((len(patch_indices), n_total), dtype=float)
        beq = np.zeros(len(patch_indices), dtype=float)
        Aeq[rows, slip_indices] = 1.0

        if state == "zero_backslip":
            return Aeq, beq, "q = 0"

        if state == "prescribed_backslip":
            if value is None:
                raise ValueError("state='prescribed_backslip' requires value=...")
            beq[:] = float(value)
            return Aeq, beq, "q = value"

        if component != "strikeslip":
            raise ValueError(f"state='{state}' requires component='strikeslip' because it depends on Euler loading")

        if state == "full_coupling":
            coupling_value = 1.0
            formula = "q + b = 0"
        elif state == "prescribed_coupling":
            if coupling is None:
                raise ValueError("state='prescribed_coupling' requires coupling=...")
            coupling_value = float(coupling)
            formula = "q + coupling * b = 0"
        else:
            raise ValueError(f"Unsupported interseismic backslip state '{state}'")

        A_loading, fixed_loading = self._build_euler_loading_linear_terms(
            fault_name,
            patch_indices,
            n_total=n_total,
            source_start=source_start,
        )
        Aeq += coupling_value * A_loading
        beq -= coupling_value * fixed_loading
        return Aeq, beq, formula

    def _build_euler_loading_linear_terms(self, fault_name, patch_indices, n_total, source_start):
        """Return A/fixed terms for ``b = block1 - block2`` loading."""
        return build_loading_linear_terms(self, fault_name, patch_indices, n_total=n_total)

    def _add_interseismic_equality_constraint(self, A, b, name, source, overwrite):
        """Store an interseismic equality row group through the active manager."""
        manager = getattr(self, "constraint_manager", None)
        if manager is not None and hasattr(manager, "add_equality_constraint"):
            manager.add_equality_constraint(A, b, name=name, source=source, overwrite=overwrite)
            if hasattr(manager, "_equality_constraints") and name not in manager._equality_constraints:
                raise RuntimeError(
                    f"Equality constraint '{name}' was not added. "
                    "Check that the current inversion mode supports linear equality constraints."
                )
            if hasattr(manager, "sync_to_solver"):
                manager.sync_to_solver()
            elif hasattr(manager, "get_combined_equality_constraints"):
                manager.get_combined_equality_constraints()
            return
        if hasattr(self, "add_custom_equality_constraint"):
            if overwrite and manager is not None and hasattr(manager, "remove_constraint"):
                manager.remove_constraint(name, "equality")
            self.add_custom_equality_constraint(A, b, name=name, source=source)
            return
        if hasattr(self, "add_equality_constraint"):
            self.add_equality_constraint(A=A, b=b, name=name, source=source)
            return
        raise AttributeError("No equality-constraint API found on this inversion object")

    def analyze_fault_kinematics(
        self,
        fault_name,
        euler_params1=None,
        euler_params2=None,
        slip_component="strikeslip",
        save_results=True,
        model=None,
    ):
        """Return a compatibility analysis dictionary for one fault.

        This wrapper preserves the historical BLSE method name while delegating
        to ``calculate_interseismic_fields()``.  It returns explicit
        backslip/slip-deficit/coupling/creep fields and avoids the removed
        legacy ``locking`` names.
        """
        result = self.calculate_interseismic_fields(
            fault_name,
            euler_params1=euler_params1,
            euler_params2=euler_params2,
            slip_component=slip_component,
            model=model,
            store=save_results,
        )
        fields = result["fields"]
        stats = result["stats"]
        return {
            "fault_name": fault_name,
            "slip_component": result["metadata"]["slip_component"],
            "num_patches": result["metadata"]["n_patches"],
            "loading_rate": {
                "values": fields["tectonic_loading_rate"],
                "stats": stats["tectonic_loading_rate"],
            },
            "inverted_slip": {
                "values": fields["inverted_slip"],
                "stats": stats["inverted_slip"],
            },
            "backslip_rate": {
                "values": fields["backslip_rate"],
                "stats": stats["backslip_rate"],
            },
            "slip_deficit_signed": {
                "values": fields["slip_deficit_signed"],
                "stats": stats["slip_deficit_signed"],
            },
            "slip_deficit_magnitude": {
                "values": fields["slip_deficit_magnitude"],
                "stats": stats["slip_deficit_magnitude"],
            },
            "coupling_ratio": {
                "values": fields["coupling_ratio"],
                "stats": stats["coupling_ratio"],
            },
            "coupling_magnitude": {
                "values": fields["coupling_magnitude"],
                "stats": stats["coupling_magnitude"],
            },
            "creep_rate_signed": {
                "values": fields["creep_rate_signed"],
                "stats": stats["creep_rate_signed"],
            },
            "creep_ratio": {
                "values": fields["creep_ratio"],
                "stats": stats["creep_ratio"],
            },
        }

    def _get_or_calculate_interseismic_result(
        self,
        fault_name,
        result=None,
        model=None,
        **kwargs,
    ):
        if result is not None:
            return result
        calc_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value is not None and not (key == "slip_component" and value == "strikeslip")
        }
        if model is None and not calc_kwargs and hasattr(self, "interseismic_results"):
            stored = self.interseismic_results.get(fault_name)
            if stored is not None:
                return stored
        return self.calculate_interseismic_fields(fault_name, model=model, **kwargs)

    def plot_interseismic_field(
        self,
        fault_name,
        field="coupling_ratio",
        result=None,
        model=None,
        euler_params1=None,
        euler_params2=None,
        solution=None,
        slip_component="strikeslip",
        cblabel=None,
        show=True,
        savefig=False,
        **plot_kwargs,
    ):
        """Plot one interseismic scalar field without modifying ``fault.slip``.

        Parameters
        ----------
        fault_name : str
            Target fault name.
        field : str, default "coupling_ratio"
            Field or alias to plot, for example ``loading``, ``backslip_rate``,
            ``slip_deficit_signed``, ``slip_deficit_magnitude`` or
            ``coupling_ratio``.
        result : dict, optional
            Precomputed result from ``calculate_interseismic_fields()``.
        model : str or array-like, optional
            Bayesian representative model passed to ``returnModel()`` before
            calculation.
        euler_params1, euler_params2 : array-like, optional
            Explicit Euler vectors ``[wx, wy, wz]`` in radians/year.
        solution : array-like, optional
            Linear solution vector.  Defaults to ``self.mpost``.
        slip_component : {"strikeslip", "dipslip", "total"}, default "strikeslip"
            Slip component used if the result must be calculated.
        cblabel : str, optional
            Colorbar label.  Defaults to the canonical field name.
        show : bool, default True
            Forwarded to CSI ``fault.plot``.
        savefig : bool, default False
            Forwarded to CSI ``fault.plot``.
        **plot_kwargs
            Additional CSI ``fault.plot`` keyword arguments.

        Returns
        -------
        object
            Whatever the underlying CSI ``fault.plot`` returns.
        """
        result = self._get_or_calculate_interseismic_result(
            fault_name,
            result=result,
            model=model,
            euler_params1=euler_params1,
            euler_params2=euler_params2,
            solution=solution,
            slip_component=slip_component,
        )
        values = get_interseismic_field_values(result, field)
        canonical = normalize_interseismic_field(field)
        fault = get_fault_by_name(self, fault_name)
        if cblabel is None:
            cblabel = canonical.replace("_", " ")
        return fault.plot(
            slip=values,
            cblabel=cblabel,
            show=show,
            savefig=savefig,
            **plot_kwargs,
        )

    def plot_fault_kinematics(
        self,
        fault_name,
        euler_params1=None,
        euler_params2=None,
        slip_component="strikeslip",
        plot_type="all",
        save_path=None,
        show=True,
        **plot_kwargs,
    ):
        """Compatibility plotting wrapper for interseismic fields.

        Parameters
        ----------
        plot_type : {"loading", "slip", "backslip", "deficit", "deficit_magnitude", "coupling", "creep", "all"}
            ``all`` creates one CSI figure per field because CSI ``fault.plot``
            does not accept an external axes object consistently across patch
            classes.

        Returns
        -------
        dict or object
            A dictionary of plot return values for ``plot_type="all"``, otherwise
            the single underlying ``fault.plot`` return value.
        """
        result = self.calculate_interseismic_fields(
            fault_name,
            euler_params1=euler_params1,
            euler_params2=euler_params2,
            slip_component=slip_component,
            store=True,
        )
        mapping = {
            "loading": "tectonic_loading_rate",
            "slip": "backslip_rate",
            "backslip": "backslip_rate",
            "deficit": "slip_deficit_signed",
            "deficit_magnitude": "slip_deficit_magnitude",
            "coupling": "coupling_ratio",
            "creep": "creep_rate_signed",
        }
        if plot_type == "all":
            return {
                key: self.plot_interseismic_field(
                    fault_name,
                    field=field,
                    result=result,
                    show=show,
                    savefig=False,
                    **plot_kwargs,
                )
                for key, field in mapping.items()
            }
        if plot_type not in mapping:
            raise ValueError(f"Invalid plot_type '{plot_type}'. Use one of {sorted(mapping)} or 'all'.")
        ret = self.plot_interseismic_field(
            fault_name,
            field=mapping[plot_type],
            result=result,
            show=show,
            savefig=False,
            **plot_kwargs,
        )
        if save_path:
            import matplotlib.pyplot as plt

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return ret

    def write_interseismic_field_gmt(
        self,
        fault_name,
        field,
        filename,
        result=None,
        model=None,
        scale=1.0,
        **kwargs,
    ):
        """Write one interseismic field to a CSI patch GMT file.

        Parameters
        ----------
        fault_name : str
            Target fault name.
        field : str
            Field or alias, for example ``loading``, ``coupling_ratio`` or
            ``slip_deficit_magnitude``.
        filename : str or path-like
            Output patch GMT path.
        result : dict, optional
            Precomputed result from ``calculate_interseismic_fields()``.
        model : str or array-like, optional
            Bayesian representative model passed to ``returnModel()``.
        scale : float, default 1.0
            Forwarded to CSI ``writePatches2File``.
        **kwargs
            Additional CSI ``writePatches2File`` keyword arguments.

        Returns
        -------
        pathlib.Path
            Written file path.
        """
        result = self._get_or_calculate_interseismic_result(fault_name, result=result, model=model)
        values = get_interseismic_field_values(result, field)
        fault = get_fault_by_name(self, fault_name)
        path = Path(filename)
        if path.parent and str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)
        fault.writePatches2File(str(path), add_slip=values, scale=scale, **kwargs)
        return path

    def write_interseismic_field_centers(
        self,
        fault_name,
        field,
        filename,
        result=None,
        model=None,
        scale=1.0,
        neg_depth=False,
        header=True,
    ):
        """Write patch-center lon/lat/depth/value records for one field.

        Parameters
        ----------
        fault_name : str
            Target fault name.
        field : str
            Field or alias to write.
        filename : str or path-like
            Output text file path.
        result : dict, optional
            Precomputed result from ``calculate_interseismic_fields()``.
        model : str or array-like, optional
            Bayesian representative model passed to ``returnModel()``.
        scale : float, default 1.0
            Multiplicative factor applied to the written values.
        neg_depth : bool, default False
            Write depth as negative values for plotting packages that use
            positive-up coordinates.
        header : bool, default True
            Write a one-line comment header.

        Returns
        -------
        pathlib.Path
            Written file path.
        """
        result = self._get_or_calculate_interseismic_result(fault_name, result=result, model=model)
        values = get_interseismic_field_values(result, field) * scale
        fault = get_fault_by_name(self, fault_name)
        path = Path(filename)
        if path.parent and str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as fout:
            if header:
                fout.write("# lon lat depth value\n")
            centers = np.asarray(fault.getcenters())
            for center, value in zip(centers, values):
                lon, lat = fault.xy2ll(center[0], center[1])
                depth = -float(center[2]) if neg_depth else float(center[2])
                fout.write(f"{float(lon):.12g} {float(lat):.12g} {depth:.12g} {float(value):.12g}\n")
        return path

    def export_interseismic_results(
        self,
        fault_name,
        outdir,
        fields=(
            "tectonic_loading_rate",
            "backslip_rate",
            "slip_deficit_signed",
            "coupling_ratio",
            "creep_rate_signed",
        ),
        result=None,
        model=None,
        write_gmt=True,
        write_centers=True,
        metadata_name="interseismic_metadata.json",
    ):
        """Export a standard interseismic result bundle for one fault.

        Parameters
        ----------
        fault_name : str
            Target fault name.
        outdir : str or path-like
            Output directory.
        fields : iterable of str
            Fields to export.  Aliases are accepted.
        result : dict, optional
            Precomputed result from ``calculate_interseismic_fields()``.
        model : str or array-like, optional
            Bayesian representative model passed to ``returnModel()``.
        write_gmt, write_centers : bool, default True
            Whether to write patch GMT and/or patch-center text files.
        metadata_name : str, default "interseismic_metadata.json"
            Metadata and statistics JSON filename.

        Returns
        -------
        dict
            Paths written under keys ``gmt``, ``centers`` and ``metadata``.
        """
        result = self._get_or_calculate_interseismic_result(fault_name, result=result, model=model)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        written = {"gmt": {}, "centers": {}, "metadata": None}

        for field in fields:
            canonical = normalize_interseismic_field(field)
            safe = canonical.replace("_", "-")
            if write_gmt:
                written["gmt"][canonical] = self.write_interseismic_field_gmt(
                    fault_name,
                    canonical,
                    outdir / f"{fault_name}_{safe}.gmt",
                    result=result,
                )
            if write_centers:
                written["centers"][canonical] = self.write_interseismic_field_centers(
                    fault_name,
                    canonical,
                    outdir / f"{fault_name}_{safe}_centers.txt",
                    result=result,
                )

        metadata = {
            "fault_name": result["fault_name"],
            "metadata": result["metadata"],
            "stats": result["stats"],
            "fields": list(result["fields"].keys()),
        }
        metadata_path = outdir / metadata_name
        with metadata_path.open("w", encoding="utf-8") as fout:
            json.dump(metadata, fout, indent=2, sort_keys=True)
        written["metadata"] = metadata_path
        return written
