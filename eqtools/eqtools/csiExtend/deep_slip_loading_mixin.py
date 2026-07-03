"""Public methods for the deep-slip loading proxy model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deep_slip_loading import (
    build_deep_slip_proxy_constraints,
    map_shallow_patches_to_deep_top_trace,
    normalize_deep_slip_state,
)
from .deep_slip_loading_fields import (
    calculate_deep_slip_loading_fields as _calculate_deep_slip_loading_fields,
    expand_deep_slip_loading_field,
    get_deep_slip_loading_field_values,
    normalize_deep_slip_loading_field,
)
from .interseismic_fields import get_fault_by_name, summarize_values


class DeepSlipLoadingMixin:
    """Mixin for shallow/deep slip-rate proxy constraints and fields.

    The deep-slip proxy is separate from the Euler/block direct-backslip
    workflow.  It relates ordinary shallow slip parameters ``s`` to ordinary
    deep slip parameters ``b`` and can then report ``1 - s / b`` style coupling
    fields.  It does not modify ``fault.slip``.
    """

    def _resolve_deep_proxy_fault(self, fault):
        if isinstance(fault, str):
            return get_fault_by_name(self, fault)
        return fault

    def _resolve_deep_proxy_faults(self, faults):
        if isinstance(faults, (str, bytes)):
            return [get_fault_by_name(self, str(faults))]
        if not isinstance(faults, Sequence):
            return [faults]
        return [self._resolve_deep_proxy_fault(fault) for fault in faults]

    def _resolve_deep_proxy_field_mapping(
        self,
        shallow_fault=None,
        deep_faults=None,
        *,
        mapping=None,
        field_mapping=None,
        field_shallow_selector="all",
        deep_selectors=None,
        component="strikeslip",
        mapping_kwargs=None,
    ):
        """Return the mapping used for derived fields.

        Deep-slip constraints often use only the shallow bottom edge, while
        derived fields such as coupling are normally interpreted on the whole
        shallow fault.  This resolver keeps those two selections separate.
        """
        if field_mapping is not None:
            return field_mapping

        selector_key = field_shallow_selector
        if isinstance(selector_key, str):
            selector_key = selector_key.lower().replace("-", "_").replace(" ", "_")
        if isinstance(selector_key, str) and selector_key in (
            "mapping",
            "constraint",
            "selected",
            "as_mapping",
            "same",
        ):
            if mapping is None:
                raise ValueError("field_shallow_selector='mapping' requires mapping=...")
            return mapping

        options = dict(mapping_kwargs or {})
        if mapping is not None:
            shallow = get_fault_by_name(self, mapping["shallow_fault"])
            if deep_faults is None:
                deep_names = list(mapping.get("deep_candidate_counts", {}).keys())
                if not deep_names:
                    deep_names = sorted({str(name) for name in mapping.get("deep_fault_names", [])})
                deep = [get_fault_by_name(self, name) for name in deep_names]
            else:
                deep = self._resolve_deep_proxy_faults(deep_faults)
            if deep_selectors is None:
                deep_selectors = mapping.get("deep_selectors")
            options.setdefault("coord_frame", mapping.get("coord_frame", "same_xy"))
            options.setdefault("top_edge_policy", mapping.get("top_edge_policy", "infer"))
            options.setdefault("top_edge_tolerance", mapping.get("top_edge_tolerance", 1.0e-8))
        else:
            if shallow_fault is None or deep_faults is None:
                raise ValueError("Provide mapping=... or both shallow_fault=... and deep_faults=...")
            shallow = self._resolve_deep_proxy_fault(shallow_fault)
            deep = self._resolve_deep_proxy_faults(deep_faults)

        options.pop("component", None)
        if selector_key is None or (isinstance(selector_key, str) and selector_key == "all"):
            shallow_selector = "all"
        else:
            shallow_selector = field_shallow_selector

        return map_shallow_patches_to_deep_top_trace(
            shallow,
            deep,
            shallow_selector=shallow_selector,
            deep_selectors=deep_selectors,
            component=component,
            **options,
        )

    def preview_deep_slip_loading_mapping(
        self,
        shallow_fault,
        deep_faults,
        *,
        shallow_selector=None,
        deep_selectors=None,
        coord_frame="same_xy",
        component="strikeslip",
        top_edge_policy="infer",
        top_edge_tolerance=1.0e-8,
        max_distance=None,
        chunk_size=4096,
    ):
        """Build and return the shallow-to-deep mapping without adding constraints.

        Parameters
        ----------
        shallow_fault : str or fault object
            Shallow fault whose selected patches are related to deep loading.
        deep_faults : str, fault object, or sequence
            Deep fault(s) whose slip rates provide the loading proxy.
        shallow_selector : selector, optional
            Patch selector for shallow patches.  Defaults to ``{"edge":
            "bottom"}`` inside the helper.  Use ``"all"`` to preview the
            full-fault field-evaluation mapping.
        deep_selectors : selector or mapping, optional
            Candidate patch selector(s) for deep fault(s).
        coord_frame : {"same_xy", "shallow_xy_from_lonlat"}, default "same_xy"
            Coordinate frame for 3-D distance mapping.
        component : {"strikeslip", "dipslip"}, default "strikeslip"
            Slip component used later by constraints and fields.
        max_distance : float, optional
            Raise if any mapped pair exceeds this distance.

        Returns
        -------
        dict
            Mapping table with shallow patch ids, matched deep fault names,
            matched deep patch ids, and distances.
        """
        shallow = self._resolve_deep_proxy_fault(shallow_fault)
        deep = self._resolve_deep_proxy_faults(deep_faults)
        return map_shallow_patches_to_deep_top_trace(
            shallow,
            deep,
            shallow_selector=shallow_selector,
            deep_selectors=deep_selectors,
            coord_frame=coord_frame,
            component=component,
            top_edge_policy=top_edge_policy,
            top_edge_tolerance=top_edge_tolerance,
            max_distance=max_distance,
            chunk_size=chunk_size,
        )

    def add_deep_slip_loading_constraint(
        self,
        shallow_fault=None,
        deep_faults=None,
        *,
        mapping=None,
        shallow_selector=None,
        deep_selectors=None,
        state="bottom_continuity",
        component="strikeslip",
        creep_ratio=None,
        locking=None,
        value=None,
        cap_ratio=None,
        motion_sense=None,
        motion_sign=None,
        coord_frame="same_xy",
        top_edge_policy="infer",
        top_edge_tolerance=1.0e-8,
        max_distance=None,
        name=None,
        source="deep_slip_loading_proxy",
        overwrite=True,
        sync=True,
    ):
        """Add deep-slip proxy equality and/or inequality constraints.

        Parameters
        ----------
        shallow_fault, deep_faults : str or fault object(s), optional
            Faults used to build a mapping when ``mapping`` is not supplied.
        mapping : dict, optional
            Precomputed output from ``preview_deep_slip_loading_mapping``.
        state : str, default "bottom_continuity"
            Constraint state.  ``bottom_continuity`` and ``full_creep`` add
            ``s - b = 0``; ``full_locking`` adds ``s = 0``; ``cap`` adds
            same-direction upper-bound inequalities.
        creep_ratio, locking, value : float, optional
            State-specific parameters.
        cap_ratio : float, optional
            Cap upper ratio.  Defaults to 1 when ``state="cap"``.
        motion_sense, motion_sign : str or number, optional
            Required for cap constraints.  Dextral/right maps to expected
            negative slip; sinistral/left maps to expected positive slip.
        name : str, optional
            Base constraint name.  Suffixes ``_equality`` and ``_inequality``
            are added when both row types exist.
        overwrite : bool, default True
            Replace an existing constraint group with the same name.

        Returns
        -------
        dict
            Added constraint names, mapping, matrices, and metadata.
        """
        state_key = normalize_deep_slip_state(state)
        if mapping is None:
            if shallow_fault is None or deep_faults is None:
                raise ValueError("Provide mapping=... or both shallow_fault=... and deep_faults=...")
            mapping = self.preview_deep_slip_loading_mapping(
                shallow_fault,
                deep_faults,
                shallow_selector=shallow_selector,
                deep_selectors=deep_selectors,
                coord_frame=coord_frame,
                component=component,
                top_edge_policy=top_edge_policy,
                top_edge_tolerance=top_edge_tolerance,
                max_distance=max_distance,
            )
        constraints = build_deep_slip_proxy_constraints(
            self,
            mapping,
            state=state_key,
            component=component,
            creep_ratio=creep_ratio,
            locking=locking,
            value=value,
            cap_ratio=cap_ratio,
            motion_sense=motion_sense,
            motion_sign=motion_sign,
        )

        base_name = name or (
            f"deep_slip_loading_{mapping['shallow_fault']}_{state_key}_{constraints['component']}"
        )
        manager = getattr(self, "constraint_manager", None)
        if manager is None:
            raise AttributeError("No constraint_manager found on this inversion object")

        added: dict[str, str] = {}
        equality = constraints.get("equality")
        inequality = constraints.get("inequality")
        if equality is not None:
            eq_name = base_name if inequality is None else f"{base_name}_equality"
            manager.add_equality_constraint(
                equality["A"],
                equality["b"],
                name=eq_name,
                source=source,
                overwrite=overwrite,
            )
            added["equality"] = eq_name
        if inequality is not None:
            ineq_name = base_name if equality is None else f"{base_name}_inequality"
            manager.add_inequality_constraint(
                inequality["A"],
                inequality["b"],
                name=ineq_name,
                source=source,
                overwrite=overwrite,
            )
            added["inequality"] = ineq_name

        if sync:
            if hasattr(manager, "sync_to_solver"):
                manager.sync_to_solver()
            else:
                if hasattr(manager, "get_combined_equality_constraints"):
                    manager.get_combined_equality_constraints()
                if hasattr(manager, "get_combined_inequality_constraints"):
                    manager.get_combined_inequality_constraints()

        return {
            "added": bool(added),
            "names": added,
            "source": source,
            "mapping": mapping,
            "constraints": constraints,
        }

    def calculate_deep_slip_loading_fields(
        self,
        shallow_fault=None,
        deep_faults=None,
        *,
        mapping=None,
        field_mapping=None,
        field_shallow_selector="all",
        shallow_selector=None,
        deep_selectors=None,
        solution=None,
        component="strikeslip",
        zero_tolerance=1.0e-12,
        model=None,
        store=True,
        **mapping_kwargs,
    ):
        """Calculate deep-proxy loading, shallow slip, deficit and coupling.

        Parameters
        ----------
        shallow_fault, deep_faults : str or fault object(s), optional
            Faults used to build a mapping when ``mapping`` is not supplied.
        mapping : dict, optional
            Precomputed constraint mapping from
            ``preview_deep_slip_loading_mapping``.  It may select only the
            shallow bottom edge.
        field_mapping : dict, optional
            Precomputed mapping used specifically for field evaluation.
        field_shallow_selector : selector or {"all", "mapping"}, default "all"
            Shallow patches used for field evaluation.  ``"all"`` evaluates
            every shallow patch and maps each patch center to the nearest deep
            top segment.  ``"mapping"`` uses ``mapping`` exactly, which is
            useful only when a subset result is intended.
        solution : array-like, optional
            Linear solution vector.  Defaults to ``self.mpost``.
        component : {"strikeslip", "dipslip"}, default "strikeslip"
            Slip component used for shallow and deep rates.
        zero_tolerance : float, default 1e-12
            Denominator tolerance for ratio fields such as
            ``coupling_to_deep = (b - s) / b`` and
            ``creep_fraction_to_deep = s / b``.  Rows with
            ``abs(deep_loading_proxy_rate) < zero_tolerance`` are counted in
            ``metadata["near_zero_deep_loading_count"]`` and their ratio
            fields are set to zero to avoid unstable divisions by nearly zero
            deep loading.
        model : str or array-like, optional
            For Bayesian inversion objects, call ``returnModel(model=model,
            print_stat=False)`` before extracting fields.
        store : bool, default True
            Store results on ``self.deep_slip_loading_results`` and attach
            namespaced fields to the shallow fault.

        Returns
        -------
        dict
            Result dictionary.  Field arrays are aligned with the selected
            shallow patch ids.  The returned ``evaluation_mapping`` records
            the mapping actually used for field evaluation; it can be passed
            to ``print_deep_slip_loading_report(...)``.
        """
        if model is not None:
            if not hasattr(self, "returnModel"):
                raise ValueError("model=... is only supported on objects with returnModel()")
            self.returnModel(model=model, print_stat=False)

        if shallow_selector is not None and field_shallow_selector == "all":
            field_shallow_selector = shallow_selector
        evaluation_mapping = self._resolve_deep_proxy_field_mapping(
            shallow_fault,
            deep_faults,
            mapping=mapping,
            field_mapping=field_mapping,
            field_shallow_selector=field_shallow_selector,
            deep_selectors=deep_selectors,
            component=component,
            mapping_kwargs=mapping_kwargs,
        )

        result = _calculate_deep_slip_loading_fields(
            self,
            evaluation_mapping,
            solution=solution,
            component=component,
            zero_tolerance=zero_tolerance,
        )
        result["evaluation_mapping"] = evaluation_mapping
        if store:
            if not hasattr(self, "deep_slip_loading_results"):
                self.deep_slip_loading_results = {}
            store_key = (result["shallow_fault"], result["metadata"]["slip_component"])
            self.deep_slip_loading_results[store_key] = result
            self.deep_slip_loading_results[result["shallow_fault"]] = result

            fault = get_fault_by_name(self, result["shallow_fault"])
            fault.deep_slip_loading_fields = result["fields"]
            fault.deep_slip_loading_stats = result["stats"]
            fault.deep_loading_proxy_rate = result["fields"]["deep_loading_proxy_rate"]
            fault.shallow_slip_rate = result["fields"]["shallow_slip_rate"]
            fault.slip_deficit_to_deep_signed = result["fields"]["slip_deficit_to_deep_signed"]
            fault.coupling_to_deep = result["fields"]["coupling_to_deep"]

        return result

    def get_deep_slip_loading_field(
        self,
        field,
        shallow_fault=None,
        deep_faults=None,
        *,
        result=None,
        mapping=None,
        model=None,
        **kwargs,
    ):
        """Return one deep-proxy field array using public aliases."""
        if result is None:
            result = self.calculate_deep_slip_loading_fields(
                shallow_fault,
                deep_faults,
                mapping=mapping,
                model=model,
                **kwargs,
            )
        return get_deep_slip_loading_field_values(result, field)

    def plot_deep_slip_loading_field(
        self,
        field="coupling_to_deep",
        shallow_fault=None,
        deep_faults=None,
        *,
        result=None,
        mapping=None,
        fill_value=0.0,
        cblabel=None,
        show=True,
        savefig=False,
        **plot_kwargs,
    ):
        """Plot one deep-proxy field on the shallow fault mesh.

        Values are expanded to the full shallow fault with ``fill_value`` for
        patches outside the mapping.  CSI's fault plotting path cannot use a
        slip array containing NaNs because color normalization is computed from
        the whole array, so ``fill_value`` must be finite.  GMT export keeps
        ``np.nan`` as its default mask value; this finite default is only for
        display.
        """
        if result is None:
            result = self.calculate_deep_slip_loading_fields(
                shallow_fault,
                deep_faults,
                mapping=mapping,
            )
        fault = get_fault_by_name(self, result["shallow_fault"])
        values = expand_deep_slip_loading_field(result, fault, field, fill_value=fill_value)
        if not np.all(np.isfinite(values)):
            selected_values = get_deep_slip_loading_field_values(result, field)
            if not np.all(np.isfinite(selected_values)):
                raise ValueError(
                    f"Deep-slip loading field '{field}' contains non-finite values on selected "
                    "patches. Check near-zero deep loading proxy rates or choose another field."
                )
            raise ValueError(
                "plot_deep_slip_loading_field() requires a finite fill_value because CSI "
                "fault plotting cannot normalize arrays containing NaN/inf. Use the default "
                "fill_value=0.0 or pass another finite display value."
            )
        canonical = normalize_deep_slip_loading_field(field)
        if cblabel is None:
            cblabel = canonical.replace("_", " ")
        return fault.plot(
            slip=values,
            cblabel=cblabel,
            show=show,
            savefig=savefig,
            **plot_kwargs,
        )

    def write_deep_slip_loading_field_gmt(
        self,
        field,
        filename,
        shallow_fault=None,
        deep_faults=None,
        *,
        result=None,
        mapping=None,
        fill_value=np.nan,
        scale=1.0,
        include_proxy_comment_columns=True,
        **kwargs,
    ):
        """Write one deep-proxy field to a CSI patch GMT file.

        The field is expanded to the full shallow fault.  Unselected patches use
        ``fill_value``.  By default the segment comment columns are written as
        ``field_value, shallow_slip_rate, deep_loading_proxy_rate`` so a single
        GMT file carries the plotted field and the two rates used to derive it.
        Set ``include_proxy_comment_columns=False`` to use CSI's default custom
        value writer, which comments ``field_value, 0, 0``.
        """
        if result is None:
            result = self.calculate_deep_slip_loading_fields(
                shallow_fault,
                deep_faults,
                mapping=mapping,
            )
        fault = get_fault_by_name(self, result["shallow_fault"])
        values = expand_deep_slip_loading_field(result, fault, field, fill_value=fill_value)
        path = Path(filename)
        if path.parent and str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)
        if include_proxy_comment_columns:
            if kwargs:
                names = ", ".join(sorted(kwargs))
                raise ValueError(
                    "Custom deep proxy GMT comments do not support extra writePatches2File "
                    f"arguments: {names}. Set include_proxy_comment_columns=False to use "
                    "fault.writePatches2File directly."
                )
            shallow_values = expand_deep_slip_loading_field(
                result,
                fault,
                "shallow_slip_rate",
                fill_value=fill_value,
            )
            deep_values = expand_deep_slip_loading_field(
                result,
                fault,
                "deep_loading_proxy_rate",
                fill_value=fill_value,
            )
            self._write_deep_slip_loading_proxy_gmt(
                fault,
                path,
                values,
                shallow_values,
                deep_values,
                scale=scale,
            )
        else:
            fault.writePatches2File(str(path), add_slip=values, scale=scale, **kwargs)
        return path

    def _write_deep_slip_loading_proxy_gmt(
        self,
        fault,
        path,
        values,
        shallow_values,
        deep_values,
        *,
        scale=1.0,
    ):
        """Write a CSI-style patch GMT with deep-proxy comment columns."""
        if not hasattr(fault, "patchll") or getattr(fault, "patchll") is None:
            if not hasattr(fault, "patch2ll"):
                raise AttributeError(
                    f"Fault '{getattr(fault, 'name', '<unnamed>')}' has no patchll "
                    "and no patch2ll() method"
                )
            fault.patch2ll()
        patchll = getattr(fault, "patchll")
        n_patches = len(fault.patch)
        arrays = {
            "values": np.asarray(values, dtype=float),
            "shallow_values": np.asarray(shallow_values, dtype=float),
            "deep_values": np.asarray(deep_values, dtype=float),
        }
        for name, arr in arrays.items():
            if arr.shape[0] != n_patches:
                raise ValueError(
                    f"{name} length ({arr.shape[0]}) must match number of patches ({n_patches})"
                )
        with path.open("w", encoding="utf-8") as fout:
            fout.write("# ECAT deep_slip_loading_proxy patch GMT\n")
            fout.write("# -Z: plotted field value\n")
            fout.write("# header comment columns: field_value shallow_slip_rate deep_loading_proxy_rate\n")
            for patch_id in range(n_patches):
                value = float(arrays["values"][patch_id]) * scale
                shallow = float(arrays["shallow_values"][patch_id]) * scale
                deep = float(arrays["deep_values"][patch_id]) * scale
                fout.write(f"> -Z{value:.12g} # {value:.12g} {shallow:.12g} {deep:.12g}\n")
                for point in np.asarray(patchll[patch_id], dtype=float):
                    fout.write(
                        f"{round(float(point[0]), 4)} "
                        f"{round(float(point[1]), 4)} "
                        f"{round(float(point[2]), 4)}\n"
                    )

    def write_deep_slip_loading_field_centers(
        self,
        field,
        filename,
        shallow_fault=None,
        deep_faults=None,
        *,
        result=None,
        mapping=None,
        scale=1.0,
        neg_depth=False,
        header=True,
        full_fault=False,
        fill_value=np.nan,
    ):
        """Write patch-center records for one deep-proxy field.

        By default only selected shallow patches are written.  Set
        ``full_fault=True`` to write all patches with ``fill_value`` outside the
        mapping.
        """
        if result is None:
            result = self.calculate_deep_slip_loading_fields(
                shallow_fault,
                deep_faults,
                mapping=mapping,
            )
        fault = get_fault_by_name(self, result["shallow_fault"])
        if full_fault:
            values = expand_deep_slip_loading_field(result, fault, field, fill_value=fill_value) * scale
            patch_indices = np.arange(len(fault.patch), dtype=int)
        else:
            values = get_deep_slip_loading_field_values(result, field) * scale
            patch_indices = np.asarray(result["metadata"]["shallow_patch_indices"], dtype=int)

        path = Path(filename)
        if path.parent and str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)
        centers = np.asarray(fault.getcenters(), dtype=float)
        with path.open("w", encoding="utf-8") as fout:
            if header:
                fout.write("# lon lat depth value\n")
            for patch_id, value in zip(patch_indices, values):
                center = centers[int(patch_id)]
                lon, lat = fault.xy2ll(center[0], center[1])
                depth = -float(center[2]) if neg_depth else float(center[2])
                fout.write(f"{float(lon):.12g} {float(lat):.12g} {depth:.12g} {float(value):.12g}\n")
        return path

    def export_deep_slip_loading_results(
        self,
        outdir,
        fields=(
            "deep_loading_proxy_rate",
            "shallow_slip_rate",
            "slip_deficit_to_deep_signed",
            "coupling_to_deep",
        ),
        shallow_fault=None,
        deep_faults=None,
        *,
        result=None,
        mapping=None,
        write_gmt=True,
        write_centers=True,
        metadata_name="deep_slip_loading_metadata.json",
    ):
        """Export a standard deep-slip loading result bundle."""
        if result is None:
            result = self.calculate_deep_slip_loading_fields(
                shallow_fault,
                deep_faults,
                mapping=mapping,
            )
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        written = {"gmt": {}, "centers": {}, "metadata": None}
        for field in fields:
            canonical = normalize_deep_slip_loading_field(field)
            safe = canonical.replace("_", "-")
            if write_gmt:
                written["gmt"][canonical] = self.write_deep_slip_loading_field_gmt(
                    canonical,
                    outdir / f"{result['shallow_fault']}_{safe}.gmt",
                    result=result,
                )
            if write_centers:
                written["centers"][canonical] = self.write_deep_slip_loading_field_centers(
                    canonical,
                    outdir / f"{result['shallow_fault']}_{safe}_centers.txt",
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

    def get_deep_slip_loading_report(self, mapping, result=None):
        """Return a compact diagnostic report for a deep-proxy mapping/result."""
        distances = np.asarray(mapping.get("mapping_distance", []), dtype=float)
        shallow_indices = np.asarray(mapping.get("shallow_patch_indices", []), dtype=int)
        deep_patch_indices = np.asarray(mapping.get("deep_patch_indices", []), dtype=int)
        deep_fault_names = [str(name) for name in np.asarray(mapping.get("deep_fault_names", []), dtype=object)]
        report = {
            "model": "deep_slip_loading_proxy",
            "shallow_fault": mapping.get("shallow_fault"),
            "component": mapping.get("component"),
            "selected_shallow_patch_count": int(shallow_indices.size),
            "deep_faults": sorted(set(deep_fault_names)),
            "deep_candidate_counts": dict(mapping.get("deep_candidate_counts", {})),
            "unique_matched_deep_pairs": int(len(set(zip(deep_fault_names, deep_patch_indices.tolist())))),
            "distance_stats": summarize_values(distances),
            "ambiguous_deep_top_edges": int(np.sum(mapping.get("ambiguous_deep_top_edge", []))),
            "warnings": [],
        }
        if distances.size == 0:
            report["warnings"].append("mapping contains no shallow-deep pairs")
        if report["ambiguous_deep_top_edges"] > 0:
            report["warnings"].append("some matched deep patches have ambiguous inferred top edges")
        if result is not None:
            meta = result.get("metadata", {})
            report["near_zero_deep_loading_count"] = meta.get("near_zero_deep_loading_count", 0)
            if meta.get("near_zero_deep_loading_count", 0):
                report["warnings"].append("some matched deep loading proxy rates are near zero")
            report["field_stats"] = result.get("stats", {})
        return report

    def print_deep_slip_loading_report(self, mapping, result=None):
        """Print and return ``get_deep_slip_loading_report`` output."""
        report = self.get_deep_slip_loading_report(mapping, result=result)
        stats = report["distance_stats"]
        print("Deep-slip loading proxy report")
        print(f"  shallow fault: {report['shallow_fault']}")
        print(f"  component: {report['component']}")
        print(f"  selected shallow patches: {report['selected_shallow_patch_count']}")
        print(f"  deep faults: {', '.join(report['deep_faults'])}")
        print(
            "  mapping distance: "
            f"min={stats['min']:.4g}, median={stats['median']:.4g}, max={stats['max']:.4g}"
        )
        print(f"  unique matched deep pairs: {report['unique_matched_deep_pairs']}")
        if report["warnings"]:
            print("  warnings:")
            for warning in report["warnings"]:
                print(f"    - {warning}")
        return report
