"""Mixin for reporting estimated data-correction parameters and predictions."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config.config_utils import get_observation_unit_info
from .data_correction_parameters import interpret_data_correction_parameters
from .interseismic_parameter_model import get_faults_from_inversion


def _as_name_set(values: Iterable[str] | str | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


def _finite_values(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _finite_value_pairs(left: Any, right: Any) -> tuple[np.ndarray, np.ndarray]:
    left_array = np.asarray(left, dtype=float).reshape(-1)
    right_array = np.asarray(right, dtype=float).reshape(-1)
    if left_array.shape != right_array.shape:
        raise ValueError(f"Prediction shape mismatch: {left_array.shape} != {right_array.shape}")
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    return left_array[mask], right_array[mask]


def _rms(values: Any) -> float | None:
    finite = _finite_values(values)
    if finite.size == 0:
        return None
    return float(np.sqrt(np.mean(finite ** 2)))


def _max_abs(values: Any) -> float | None:
    finite = _finite_values(values)
    if finite.size == 0:
        return None
    return float(np.max(np.abs(finite)))


def _corrcoef(left: Any, right: Any) -> float | None:
    left_values, right_values = _finite_value_pairs(left, right)
    if left_values.size < 2:
        return None
    if np.std(left_values) == 0.0 or np.std(right_values) == 0.0:
        return None
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _format_optional(value: float | None, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}g}"


class DataCorrectionReportMixin:
    """Report data-correction coefficients without modifying the solver state."""

    def _get_data_correction_data_objects(self) -> list[Any]:
        config = getattr(self, "config", None)
        geodata = getattr(config, "geodata", None)
        if isinstance(geodata, Mapping):
            return list(geodata.get("data", []) or [])
        return list(getattr(self, "geodata", []) or [])

    def _get_data_correction_data_map(self) -> dict[str, Any]:
        data_map = {}
        for data in self._get_data_correction_data_objects():
            name = getattr(data, "name", None)
            if name is not None:
                data_map[str(name)] = data
        return data_map

    def _get_data_correction_fault_map(self) -> dict[str, Any]:
        fault_map = {}
        for fault in get_faults_from_inversion(self):
            name = getattr(fault, "name", None)
            if name is not None:
                fault_map[str(name)] = fault
        return fault_map

    def _get_data_correction_unit_info(self) -> dict[str, Any]:
        config = getattr(self, "config", None)
        default = "m"
        interseismic = getattr(config, "interseismic_config", None)
        if isinstance(interseismic, Mapping) and interseismic:
            fault_loading = interseismic.get("fault_loading", {}) or {}
            if isinstance(fault_loading, Mapping) and fault_loading.get("enabled", False):
                default = "m/yr"
        return get_observation_unit_info(self, default=default)

    def _resolve_data_correction_data(self, dataset: str | Any) -> Any:
        if not isinstance(dataset, str):
            return dataset
        data_map = self._get_data_correction_data_map()
        try:
            return data_map[dataset]
        except KeyError as exc:
            raise ValueError(f"Dataset '{dataset}' was not found in this inversion") from exc

    def _resolve_data_correction_source(self, source: str | Any | None, dataset_name: str) -> Any:
        if source is not None and not isinstance(source, str):
            return source
        fault_map = self._get_data_correction_fault_map()
        if isinstance(source, str):
            try:
                return fault_map[source]
            except KeyError as exc:
                raise ValueError(f"Source '{source}' was not found in this inversion") from exc

        candidates = [
            fault for fault in fault_map.values()
            if dataset_name in (getattr(fault, "poly", {}) or {})
            and (getattr(fault, "poly", {}) or {}).get(dataset_name) is not None
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"No data-correction source found for dataset '{dataset_name}'")
        names = ", ".join(str(getattr(fault, "name", "")) for fault in candidates)
        raise ValueError(
            f"Multiple data-correction sources found for dataset '{dataset_name}': {names}. "
            "Pass source=... explicitly."
        )

    @staticmethod
    def _copy_prediction_array(data: Any, attr: str) -> np.ndarray:
        if not hasattr(data, attr):
            raise AttributeError(f"Data object has no '{attr}' attribute after prediction")
        return np.asarray(getattr(data, attr), dtype=float).copy()

    @staticmethod
    def _copy_observed_array(data: Any) -> np.ndarray:
        if hasattr(data, "vel_enu"):
            return np.asarray(data.vel_enu, dtype=float).copy()
        if hasattr(data, "vel"):
            return np.asarray(data.vel, dtype=float).copy()
        raise AttributeError("Data object must expose 'vel_enu' or 'vel' to report observed values")

    @staticmethod
    def _prediction_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if left.shape != right.shape:
            raise ValueError(f"Prediction shape mismatch: {left.shape} != {right.shape}")
        return left - right

    @staticmethod
    def _make_single_transform_fault(fault: Any, dataset_name: str, transform: Any, params: np.ndarray) -> Any:
        fault_copy = copy.copy(fault)
        fault_copy.poly = dict(getattr(fault, "poly", {}) or {})
        fault_copy.poly[dataset_name] = transform
        fault_copy.polysol = dict(getattr(fault, "polysol", {}) or {})
        fault_copy.polysol[dataset_name] = np.asarray(params, dtype=float).reshape(-1)
        return fault_copy

    @staticmethod
    def _extract_polysol_parameters(fault: Any, dataset_name: str) -> tuple[np.ndarray | None, list[int] | None]:
        polysol = getattr(fault, "polysol", {}) or {}
        params = polysol.get(dataset_name)
        if params is None:
            return None, None
        values = np.asarray(params, dtype=float).reshape(-1)
        index_map = getattr(fault, "polysolindex", {}) or {}
        indices = index_map.get(dataset_name)
        if indices is None:
            columns = None
        else:
            columns = [int(index) for index in list(indices)]
        return values, columns

    @staticmethod
    def _get_transform_index_info(fault: Any, dataset_name: str, transform: Any) -> list[int] | None:
        transform_indices = getattr(fault, "transform_indices", {}) or {}
        if not isinstance(transform, str):
            return None
        local = transform_indices.get(dataset_name, {}).get(str(transform))
        if local is None:
            return None
        start, end = map(int, local)
        return list(range(start, end))

    @staticmethod
    def _get_transform_parameter_count(data: Any, transform: Any, remaining: int) -> int:
        if isinstance(transform, (int, np.integer)):
            return int(transform)
        if data is not None and hasattr(data, "getNumberOfTransformParameters"):
            return int(data.getNumberOfTransformParameters(transform))
        expected = {
            "full": 4,
            "eulerrotation": 3,
            "internalstrain": 3,
            "translation": 2,
            "translationrotation": 3,
            "strainonly": 3,
            "strainnorotation": 5,
            "strainnotranslation": 4,
            "strain": 6,
        }
        return int(expected.get(str(transform).lower(), remaining))

    def _iter_transform_parameter_blocks(
        self,
        transform: Any,
        params: np.ndarray,
        columns: list[int] | None,
        data: Any,
    ):
        if not isinstance(transform, list):
            yield transform, params, columns, None
            return

        start = 0
        for item in transform:
            n_params = self._get_transform_parameter_count(data, item, params.size - start)
            end = start + n_params
            item_columns = columns[start:end] if columns is not None else None
            yield item, params[start:end], item_columns, transform
            start = end

    def collect_data_correction_parameters(
        self,
        *,
        datasets: Sequence[str] | str | None = None,
        sources: Sequence[str] | str | None = None,
        include_missing: bool = False,
        euler_output_units: Sequence[str] = ("degrees", "degrees", "degrees_per_myr"),
    ) -> list[dict[str, Any]]:
        """Collect interpreted data-correction parameters.

        Parameters
        ----------
        datasets, sources : sequence of str or str, optional
            Optional filters for dataset names and source/fault names.
        include_missing : bool, default False
            If True, include entries for configured transforms whose estimated
            parameters are not available yet.
        euler_output_units : sequence of str, default degrees/degrees/Myr
            Units used for the reported Euler pole converted from estimated
            Cartesian ``eulerrotation`` parameters.

        Returns
        -------
        list of dict
            One entry per estimated data-correction transform.
        """
        dataset_filter = _as_name_set(datasets)
        source_filter = _as_name_set(sources)
        data_map = self._get_data_correction_data_map()
        unit_info = self._get_data_correction_unit_info()
        entries: list[dict[str, Any]] = []

        for fault in get_faults_from_inversion(self):
            source_name = str(getattr(fault, "name", ""))
            if source_filter is not None and source_name not in source_filter:
                continue
            poly = getattr(fault, "poly", {}) or {}
            if not isinstance(poly, Mapping):
                continue

            for dataset_name, transform in poly.items():
                dataset_name = str(dataset_name)
                if dataset_filter is not None and dataset_name not in dataset_filter:
                    continue
                if transform is None:
                    continue
                params, columns = self._extract_polysol_parameters(fault, dataset_name)
                data = data_map.get(dataset_name)
                warnings: list[str] = []
                if data is None:
                    warnings.append(f"Dataset object '{dataset_name}' was not found; normalization metadata may be missing.")
                if params is None:
                    if not include_missing:
                        continue
                    warnings.append(
                        "Estimated parameters are not available. Run the inversion and distribute/return the model first."
                    )
                    entries.append(
                        {
                            "dataset": dataset_name,
                            "source": source_name,
                            "transform": transform,
                            "parameters_available": False,
                            "warnings": warnings,
                        }
                    )
                    continue

                for item_transform, item_params, item_columns, transform_group in self._iter_transform_parameter_blocks(
                    transform,
                    params,
                    columns,
                    data,
                ):
                    try:
                        interpreted = interpret_data_correction_parameters(
                            item_transform,
                            item_params,
                            data=data,
                            euler_output_units=euler_output_units,
                            observation_unit=None if unit_info.get("assumed") else unit_info["observation"],
                            default_observation_unit=unit_info["observation"],
                        )
                    except Exception as exc:  # keep reporting robust for mixed data types
                        entries.append(
                            {
                                "dataset": dataset_name,
                                "source": source_name,
                                "transform": item_transform,
                                "transform_group": transform_group,
                                "parameters_available": True,
                                "raw_parameters": item_params.tolist(),
                                "columns": item_columns,
                                "warnings": warnings + [f"Failed to interpret transform: {exc}"],
                            }
                        )
                        continue

                    entries.append(
                        {
                            "dataset": dataset_name,
                            "source": source_name,
                            "data_type": getattr(data, "dtype", data.__class__.__name__ if data is not None else None),
                            "transform": item_transform,
                            "transform_group": transform_group,
                            "parameters_available": True,
                            "columns": item_columns,
                            "transform_columns": self._get_transform_index_info(fault, dataset_name, item_transform),
                            **interpreted,
                            "warnings": warnings + interpreted.get("warnings", []),
                        }
                    )
        return entries

    @staticmethod
    def data_correction_parameters_to_dataframe(entries: Sequence[Mapping[str, Any]]):
        """Return a compact pandas DataFrame for report entries."""
        import pandas as pd

        rows = []
        for entry in entries:
            physical = dict(entry.get("physical", {}) or {})
            raw_value = entry.get("raw_parameters", {}) or {}
            raw = dict(raw_value) if isinstance(raw_value, Mapping) else {}
            pole = physical.get("euler_pole", {}) or {}
            row = {
                "source": entry.get("source"),
                "dataset": entry.get("dataset"),
                "data_type": entry.get("data_type"),
                "transform": entry.get("transform"),
                "transform_group": entry.get("transform_group"),
                "kind": entry.get("kind"),
                "columns": entry.get("columns"),
                "warnings": "; ".join(entry.get("warnings", []) or []),
            }
            for key, value in raw.items():
                if np.isscalar(value):
                    row[f"raw_{key}"] = value
            if pole:
                row["euler_pole"] = pole.get("value")
                row["euler_pole_units"] = pole.get("units")
            for key in ("rotation_cw_per_coord", "scale_per_coord"):
                if key in physical:
                    row[key] = physical.get(key)
            rows.append(row)
        return pd.DataFrame(rows)

    def print_data_correction_report(
        self,
        entries: Sequence[Mapping[str, Any]] | None = None,
        *,
        datasets: Sequence[str] | str | None = None,
        sources: Sequence[str] | str | None = None,
    ) -> list[dict[str, Any]]:
        """Print a concise report and return the underlying entries."""
        if entries is None:
            entries = self.collect_data_correction_parameters(datasets=datasets, sources=sources)
        entries = [dict(entry) for entry in entries]
        print("Data-correction parameter report")
        print(f"  entries: {len(entries)}")
        unit_info = self._get_data_correction_unit_info()
        if unit_info.get("observation"):
            assumed = " (assumed)" if unit_info.get("assumed") else ""
            print(f"  units: observation={unit_info['observation']}{assumed}")
        for entry in entries:
            header = f"  - {entry.get('source')} / {entry.get('dataset')}: {entry.get('transform')}"
            if entry.get("transform_group") is not None:
                header += f" (from {entry.get('transform_group')})"
            print(header)
            physical = entry.get("physical", {}) or {}
            raw = entry.get("raw_parameters", {}) or {}
            if raw:
                if isinstance(raw, Mapping):
                    raw_text = ", ".join(
                        f"{key}={value:.6g}" for key, value in raw.items() if isinstance(value, (int, float, np.floating))
                    )
                else:
                    raw_text = ", ".join(
                        f"{value:.6g}" for value in raw if isinstance(value, (int, float, np.floating))
                    )
                if raw_text:
                    print(f"      raw: {raw_text}")
            pole = physical.get("euler_pole")
            if pole:
                vals = pole.get("value", [])
                units = pole.get("units", [])
                print(f"      euler pole: {vals} {units}")
            if physical.get("rotation_cw_per_coord") is not None:
                print(f"      rotation_cw_per_coord: {physical['rotation_cw_per_coord']:.6g}")
            if physical.get("scale_per_coord") is not None:
                print(f"      scale_per_coord: {physical['scale_per_coord']:.6g}")
            for warning in entry.get("warnings", []) or []:
                print(f"      warning: {warning}")
        return entries

    def calculate_data_correction_prediction_parts(
        self,
        dataset: str | Any,
        *,
        source: str | Any | None = None,
        faults: Sequence[Any] | None = None,
        direction: str = "sd",
        vertical: bool | None = None,
        transforms: Sequence[Any] | Any | None = None,
        compute_norm_fact: bool = False,
        compute_int_strain_norm_fact: bool = False,
    ) -> dict[str, Any]:
        """Return read-only prediction pieces for one data set.

        The method decomposes the current model prediction into observed data,
        slip-only prediction, data-correction prediction, total prediction and
        optional single-transform predictions.  It calls CSI prediction methods
        on shallow copies of the data object, so the original data object is not
        modified.

        Parameters
        ----------
        dataset : str or data object
            Dataset name or data object to diagnose.
        source : str or object, optional
            Fault/source that owns the correction parameters.  Required when
            more than one source has correction terms for the same dataset.
        faults : sequence, optional
            Faults used for the slip-only and total predictions.  Defaults to
            all faults in the inversion.
        direction : str, default "sd"
            Slip components passed to ``buildsynth``.
        vertical : bool, optional
            Vertical flag passed to ``buildsynth``.  If omitted, the value is
            taken from ``config.geodata['verticals']`` when available; otherwise
            CSI's default behavior is used.
        transforms : sequence or str, optional
            Subset of correction transforms to evaluate individually.  By
            default all transforms configured on ``source.poly[dataset]`` are
            reported.
        compute_norm_fact, compute_int_strain_norm_fact : bool, default False
            Passed through to CSI.  Defaults use the normalization metadata
            saved when the design matrix was built.

        Returns
        -------
        dict
            Arrays and metadata for ``observed``, ``slip_only``,
            ``transformation``, ``total``, residuals and ``single_transforms``.
        """
        data = self._resolve_data_correction_data(dataset)
        dataset_name = str(getattr(data, "name", dataset))
        source_fault = self._resolve_data_correction_source(source, dataset_name)
        model_faults = list(faults) if faults is not None else list(get_faults_from_inversion(self))

        vertical_value = vertical
        if vertical_value is None:
            config = getattr(self, "config", None)
            geodata = getattr(config, "geodata", None)
            if isinstance(geodata, Mapping):
                data_names = [str(getattr(item, "name", "")) for item in geodata.get("data", []) or []]
                if dataset_name in data_names:
                    verticals = geodata.get("verticals", []) or []
                    idx = data_names.index(dataset_name)
                    if idx < len(verticals):
                        vertical_value = bool(verticals[idx])

        build_kwargs = {
            "direction": direction,
            "poly": None,
        }
        if vertical_value is not None:
            build_kwargs["vertical"] = vertical_value

        observed = self._copy_observed_array(data)

        slip_data = copy.copy(data)
        slip_data.buildsynth(model_faults, **build_kwargs)
        slip_only = self._copy_prediction_array(slip_data, "synth")

        transform_data = copy.copy(data)
        transform_data.computeTransformation(
            source_fault,
            computeNormFact=compute_norm_fact,
            computeIntStrainNormFact=compute_int_strain_norm_fact,
        )
        transformation = self._copy_prediction_array(transform_data, "transformation")

        total_data = copy.copy(data)
        total_kwargs = dict(build_kwargs)
        total_kwargs["poly"] = "include"
        total_kwargs["computeNormFact"] = compute_norm_fact
        total_kwargs["computeIntStrainNormFact"] = compute_int_strain_norm_fact
        total_data.buildsynth(model_faults, **total_kwargs)
        total = self._copy_prediction_array(total_data, "synth")

        params, columns = self._extract_polysol_parameters(source_fault, dataset_name)
        transform_config = (getattr(source_fault, "poly", {}) or {}).get(dataset_name)
        if transform_config is None:
            raise ValueError(f"Source '{getattr(source_fault, 'name', source_fault)}' has no transform for '{dataset_name}'")
        if params is None:
            raise ValueError(
                f"Source '{getattr(source_fault, 'name', source_fault)}' has no estimated polysol for '{dataset_name}'"
            )

        selected = transforms
        if selected is None:
            selected_set = None
        elif isinstance(selected, (str, int, np.integer)):
            selected_set = {selected}
        else:
            selected_set = set(selected)

        single_transforms = {}
        single_columns = {}
        for item_transform, item_params, item_columns, _group in self._iter_transform_parameter_blocks(
            transform_config,
            params,
            columns,
            data,
        ):
            if selected_set is not None and item_transform not in selected_set:
                continue
            item_fault = self._make_single_transform_fault(source_fault, dataset_name, item_transform, item_params)
            item_data = copy.copy(data)
            item_data.computeTransformation(
                item_fault,
                computeNormFact=compute_norm_fact,
                computeIntStrainNormFact=compute_int_strain_norm_fact,
            )
            key = str(item_transform)
            single_transforms[key] = self._copy_prediction_array(item_data, "transformation")
            single_columns[key] = item_columns

        return {
            "dataset": dataset_name,
            "source": str(getattr(source_fault, "name", "")),
            "direction": direction,
            "vertical": vertical_value,
            "transform": transform_config,
            "observed": observed,
            "slip_only": slip_only,
            "transformation": transformation,
            "total": total,
            "residual_total": self._prediction_difference(observed, total),
            "residual_slip_only": self._prediction_difference(observed, slip_only),
            "residual_after_transform": self._prediction_difference(
                self._prediction_difference(observed, transformation),
                slip_only,
            ),
            "total_consistency": self._prediction_difference(total, slip_only + transformation),
            "single_transforms": single_transforms,
            "single_transform_columns": single_columns,
        }

    def collect_data_correction_diagnostics(
        self,
        dataset: str | Any,
        *,
        source: str | Any | None = None,
        faults: Sequence[Any] | None = None,
        direction: str = "sd",
        vertical: bool | None = None,
        transforms: Sequence[Any] | Any | None = None,
        compute_norm_fact: bool = False,
        compute_int_strain_norm_fact: bool = False,
        cancellation_ratio_threshold: float = 0.2,
        cancellation_correlation_threshold: float = -0.95,
        consistency_tolerance: float = 1.0e-8,
    ) -> dict[str, Any]:
        """Summarize slip/correction prediction balance for one data set.

        This is a read-only diagnostic helper.  It evaluates the current
        solved model with ``calculate_data_correction_prediction_parts`` and
        returns compact RMS, consistency, and pairwise-cancellation statistics.
        It does not change the data object, fault object, solver matrices, or
        saved model parameters.

        Parameters
        ----------
        dataset, source, faults, direction, vertical, transforms
            Passed through to ``calculate_data_correction_prediction_parts``.
        compute_norm_fact, compute_int_strain_norm_fact
            Passed through to CSI prediction methods.  Defaults reuse the
            normalization metadata saved when the design matrix was built.
        cancellation_ratio_threshold : float, default 0.2
            Pairwise transform sums with ``rms(a + b) / max(rms(a), rms(b))``
            below this value are treated as strong cancellation candidates.
        cancellation_correlation_threshold : float, default -0.95
            Pairwise transform correlations below this value are treated as
            near-opposite prediction patterns.
        consistency_tolerance : float, default 1e-8
            Tolerance for checking ``total ~= slip_only + transformation``.

        Returns
        -------
        dict
            Compact numerical diagnostics.  Large prediction arrays are not
            included; use ``calculate_data_correction_prediction_parts`` when
            the full arrays are needed.
        """
        parts = self.calculate_data_correction_prediction_parts(
            dataset,
            source=source,
            faults=faults,
            direction=direction,
            vertical=vertical,
            transforms=transforms,
            compute_norm_fact=compute_norm_fact,
            compute_int_strain_norm_fact=compute_int_strain_norm_fact,
        )

        residual_slip_only = _rms(parts["residual_slip_only"])
        residual_total = _rms(parts["residual_total"])
        improvement_absolute = None
        improvement_relative = None
        if residual_slip_only is not None and residual_total is not None:
            improvement_absolute = residual_slip_only - residual_total
            if residual_slip_only != 0.0:
                improvement_relative = improvement_absolute / residual_slip_only

        consistency_max_abs = _max_abs(parts["total_consistency"])
        warnings: list[str] = []
        if consistency_max_abs is not None and consistency_max_abs > consistency_tolerance:
            warnings.append(
                "Total prediction is not numerically equal to slip-only plus data-correction prediction; "
                "check CSI prediction settings and normalization flags."
            )

        single_summary: dict[str, dict[str, Any]] = {}
        for name, values in parts["single_transforms"].items():
            single_summary[name] = {
                "rms": _rms(values),
                "max_abs": _max_abs(values),
                "columns": parts["single_transform_columns"].get(name),
            }

        pairwise: list[dict[str, Any]] = []
        names = list(parts["single_transforms"].keys())
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                first_values = parts["single_transforms"][first]
                second_values = parts["single_transforms"][second]
                first_rms = _rms(first_values)
                second_rms = _rms(second_values)
                sum_rms = _rms(first_values + second_values)
                corr = _corrcoef(first_values, second_values)
                cancellation_ratio = None
                if first_rms is not None and second_rms is not None and sum_rms is not None:
                    denom = max(first_rms, second_rms)
                    if denom != 0.0:
                        cancellation_ratio = sum_rms / denom

                pair_warning = None
                if (
                    corr is not None
                    and cancellation_ratio is not None
                    and corr <= cancellation_correlation_threshold
                    and cancellation_ratio <= cancellation_ratio_threshold
                ):
                    pair_warning = (
                        f"{first} and {second} strongly cancel each other in the prediction field; "
                        "interpret their individual physical parameters with caution."
                    )
                    warnings.append(pair_warning)

                pairwise.append(
                    {
                        "transforms": [first, second],
                        "correlation": corr,
                        "sum_rms": sum_rms,
                        "cancellation_ratio": cancellation_ratio,
                        "warning": pair_warning,
                    }
                )

        return {
            "dataset": parts["dataset"],
            "source": parts["source"],
            "direction": parts["direction"],
            "vertical": parts["vertical"],
            "transform": parts["transform"],
            "rms": {
                "observed": _rms(parts["observed"]),
                "slip_only_prediction": _rms(parts["slip_only"]),
                "data_correction_prediction": _rms(parts["transformation"]),
                "total_prediction": _rms(parts["total"]),
                "residual_slip_only": residual_slip_only,
                "residual_total": residual_total,
                "residual_after_transform": _rms(parts["residual_after_transform"]),
            },
            "improvement": {
                "absolute": improvement_absolute,
                "relative": improvement_relative,
            },
            "consistency": {
                "max_abs_total_minus_slip_plus_correction": consistency_max_abs,
                "tolerance": consistency_tolerance,
            },
            "single_transforms": single_summary,
            "pairwise_transforms": pairwise,
            "warnings": warnings,
        }

    def print_data_correction_diagnostics(
        self,
        dataset: str | Any,
        *,
        source: str | Any | None = None,
        faults: Sequence[Any] | None = None,
        direction: str = "sd",
        vertical: bool | None = None,
        transforms: Sequence[Any] | Any | None = None,
        compute_norm_fact: bool = False,
        compute_int_strain_norm_fact: bool = False,
        cancellation_ratio_threshold: float = 0.2,
        cancellation_correlation_threshold: float = -0.95,
        consistency_tolerance: float = 1.0e-8,
        file: Any | None = None,
    ) -> dict[str, Any]:
        """Print a compact data-correction prediction diagnostic report."""
        diagnostics = self.collect_data_correction_diagnostics(
            dataset,
            source=source,
            faults=faults,
            direction=direction,
            vertical=vertical,
            transforms=transforms,
            compute_norm_fact=compute_norm_fact,
            compute_int_strain_norm_fact=compute_int_strain_norm_fact,
            cancellation_ratio_threshold=cancellation_ratio_threshold,
            cancellation_correlation_threshold=cancellation_correlation_threshold,
            consistency_tolerance=consistency_tolerance,
        )

        def emit(line: str = "") -> None:
            if file is None:
                print(line)
            else:
                print(line, file=file)

        emit("Data-correction prediction diagnostics")
        emit(f"  source/dataset: {diagnostics['source']} / {diagnostics['dataset']}")
        emit(f"  transform: {diagnostics['transform']}")
        emit(f"  direction: {diagnostics['direction']}, vertical: {diagnostics['vertical']}")

        rms = diagnostics["rms"]
        emit("  RMS")
        emit(f"    observed: {_format_optional(rms['observed'])}")
        emit(f"    slip-only prediction: {_format_optional(rms['slip_only_prediction'])}")
        emit(f"    data-correction prediction: {_format_optional(rms['data_correction_prediction'])}")
        emit(f"    total prediction: {_format_optional(rms['total_prediction'])}")
        emit(f"    residual, slip only: {_format_optional(rms['residual_slip_only'])}")
        emit(f"    residual, slip + correction: {_format_optional(rms['residual_total'])}")

        improvement = diagnostics["improvement"]
        emit("  Improvement")
        emit(f"    absolute RMS reduction: {_format_optional(improvement['absolute'])}")
        relative = improvement["relative"]
        relative_text = "n/a" if relative is None else f"{100.0 * relative:.3g}%"
        emit(f"    relative RMS reduction: {relative_text}")

        single = diagnostics["single_transforms"]
        if single:
            emit("  Single transforms")
            for name, item in single.items():
                columns = item.get("columns")
                column_text = "" if columns is None else f", columns={columns}"
                emit(
                    f"    - {name}: rms={_format_optional(item.get('rms'))}, "
                    f"max_abs={_format_optional(item.get('max_abs'))}{column_text}"
                )

        pairwise = diagnostics["pairwise_transforms"]
        if pairwise:
            emit("  Pairwise transform balance")
            for item in pairwise:
                first, second = item["transforms"]
                emit(
                    f"    - {first} + {second}: corr={_format_optional(item.get('correlation'))}, "
                    f"sum_rms={_format_optional(item.get('sum_rms'))}, "
                    f"cancellation_ratio={_format_optional(item.get('cancellation_ratio'))}"
                )

        consistency = diagnostics["consistency"]
        emit("  Consistency")
        emit(
            "    max_abs(total - slip_only - correction): "
            f"{_format_optional(consistency['max_abs_total_minus_slip_plus_correction'])}"
        )

        if diagnostics["warnings"]:
            emit("  Warnings")
            for warning in diagnostics["warnings"]:
                emit(f"    - {warning}")

        return diagnostics
