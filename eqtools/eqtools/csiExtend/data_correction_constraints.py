"""Linear constraints between data-correction parameters.

This module resolves configured ``geodata.polys`` parameters before an
inversion is run and builds ordinary linear constraint matrices.  It does not
modify CSI data objects, recompute Green's functions, or interpret solved
parameters.  The public mixin is intentionally thin: it resolves named
data-correction parameters, optionally converts raw coefficients to simple
physical units, and updates the existing constraint manager with either hard
equalities or explicit parameter bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .data_correction_parameters import extract_normalization
from .interseismic_parameter_model import get_fault_by_name, get_faults_from_inversion


@dataclass(frozen=True)
class DataCorrectionParameterRef:
    """Reference to one configured data-correction transform.

    Parameters
    ----------
    dataset : str
        Data-set name, such as ``"gps_campaign"``.
    transform : str or int, optional
        Transform inside ``fault.poly[dataset]``.  Required when the dataset
        uses a list of transforms.
    owner : str, optional
        Source/fault that owns the poly columns.  If omitted, ECAT searches for
        a unique source containing the dataset transform.
    components : sequence of str, optional
        Parameter components to select.  If omitted, all components of the
        selected transform are used.
    """

    dataset: str
    transform: Any | None = None
    owner: str | None = None
    components: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ResolvedDataCorrectionParameter:
    """Resolved model-vector columns for one data-correction reference."""

    owner: str
    dataset: str
    transform: Any
    components: tuple[str, ...]
    columns: np.ndarray
    scales: np.ndarray
    space: str


def _as_ref(ref: Mapping[str, Any] | DataCorrectionParameterRef) -> DataCorrectionParameterRef:
    if isinstance(ref, DataCorrectionParameterRef):
        return ref
    components = ref.get("components")
    if isinstance(components, str):
        components = (components,)
    elif components is not None:
        components = tuple(str(component) for component in components)
    return DataCorrectionParameterRef(
        dataset=str(ref["dataset"]),
        transform=ref.get("transform"),
        owner=ref.get("owner") or ref.get("source"),
        components=components,
    )


def _get_data_objects(inversion: Any) -> list[Any]:
    config = getattr(inversion, "config", None)
    geodata = getattr(config, "geodata", None)
    if isinstance(geodata, Mapping):
        return list(geodata.get("data", []) or [])
    return list(getattr(inversion, "geodata", []) or [])


def _get_data_by_name(inversion: Any, dataset: str) -> Any:
    for data in _get_data_objects(inversion):
        if str(getattr(data, "name", "")) == str(dataset):
            return data
    raise ValueError(f"Dataset '{dataset}' was not found in this inversion")


def _resolve_owner(inversion: Any, owner: str | None, dataset: str) -> Any:
    if owner is not None:
        return get_fault_by_name(inversion, owner)

    candidates = []
    for fault in get_faults_from_inversion(inversion):
        poly = getattr(fault, "poly", {}) or {}
        if isinstance(poly, Mapping) and dataset in poly and poly[dataset] is not None:
            candidates.append(fault)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"No data-correction owner found for dataset '{dataset}'")
    names = ", ".join(str(getattr(fault, "name", "")) for fault in candidates)
    raise ValueError(
        f"Multiple data-correction owners found for dataset '{dataset}': {names}. "
        "Pass owner=... explicitly."
    )


def _get_transform_parameter_count(data: Any, transform: Any, remaining: int | None = None) -> int:
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
    key = str(transform).lower()
    if key in expected:
        return expected[key]
    if remaining is not None:
        return int(remaining)
    raise ValueError(f"Cannot infer parameter count for transform {transform!r}")


def _configured_dataset_order(owner: Any) -> list[str]:
    datanames = getattr(owner, "datanames", None)
    if datanames is not None:
        return [str(name) for name in datanames]
    poly = getattr(owner, "poly", {}) or {}
    return [str(name) for name in poly.keys()]


def _get_dataset_poly_start(inversion: Any, owner: Any, dataset: str) -> int:
    owner_name = str(getattr(owner, "name", ""))
    try:
        poly_start, _poly_end = getattr(inversion, "poly_positions")[owner_name]
    except Exception as exc:
        raise ValueError(
            f"Cannot find poly_positions for owner '{owner_name}'. "
            "Build/assemble the inversion before adding data-correction constraints."
        ) from exc

    numberofpolys = getattr(owner, "numberofpolys", {}) or {}
    offset = 0
    for dname in _configured_dataset_order(owner):
        if dname == dataset:
            return int(poly_start + offset)
        offset += int(numberofpolys.get(dname, 0))
    raise ValueError(f"Dataset '{dataset}' is not configured on owner '{owner_name}'")


def _iter_transform_blocks(transform_config: Any, data: Any, total_count: int):
    if not isinstance(transform_config, list):
        yield transform_config, 0, int(total_count)
        return

    start = 0
    for item in transform_config:
        n_params = _get_transform_parameter_count(data, item, total_count - start)
        yield item, start, n_params
        start += n_params


def _select_transform_block(owner: Any, data: Any, dataset: str, transform: Any | None) -> tuple[Any, int, int]:
    poly = getattr(owner, "poly", {}) or {}
    if dataset not in poly or poly[dataset] is None:
        raise ValueError(f"Owner '{getattr(owner, 'name', owner)}' has no transform for dataset '{dataset}'")

    transform_config = poly[dataset]
    numberofpolys = getattr(owner, "numberofpolys", {}) or {}
    total_count = int(numberofpolys.get(dataset, 0))
    if total_count <= 0:
        total_count = _get_transform_parameter_count(data, transform_config)

    if transform is None:
        if isinstance(transform_config, list):
            raise ValueError(
                f"Dataset '{dataset}' uses multiple transforms {transform_config}; pass transform=..."
            )
        return transform_config, 0, total_count

    for item, start, n_params in _iter_transform_blocks(transform_config, data, total_count):
        if item == transform or str(item).lower() == str(transform).lower():
            return item, start, n_params
    raise ValueError(f"Transform {transform!r} was not found for dataset '{dataset}'")


def _same_transform(left: Any, right: Any) -> bool:
    return left == right or str(left).lower() == str(right).lower()


def _transform_config_has_unique_match(owner: Any, data: Any, dataset: str, transform: Any) -> bool:
    poly = getattr(owner, "poly", {}) or {}
    if dataset not in poly or poly[dataset] is None:
        return False

    transform_config = poly[dataset]
    numberofpolys = getattr(owner, "numberofpolys", {}) or {}
    total_count = int(numberofpolys.get(dataset, 0))
    if total_count <= 0:
        total_count = _get_transform_parameter_count(data, transform_config)

    matches = [
        item
        for item, _start, _n_params in _iter_transform_blocks(transform_config, data, total_count)
        if _same_transform(item, transform)
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Dataset '{dataset}' has repeated transform {transform!r} in owner "
            f"'{getattr(owner, 'name', owner)}'. Pass a custom column selector instead."
        )
    return len(matches) == 1


def _normalise_components(components: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if components is None:
        return None
    if isinstance(components, str):
        return (components,)
    return tuple(str(component) for component in components)


def _normalise_dataset_names(
    inversion: Any,
    *,
    dataset: str | None,
    datasets: str | Sequence[str] | None,
) -> tuple[list[str], bool]:
    if dataset is not None and datasets is not None:
        raise ValueError("Pass either dataset=... or datasets=..., not both")
    if dataset is not None:
        return [str(dataset)], False
    if datasets is None:
        raise ValueError("Pass dataset=... or datasets=...")
    if isinstance(datasets, str):
        if datasets.lower() == "all":
            return [str(getattr(data, "name", "")) for data in _get_data_objects(inversion)], True
        return [str(datasets)], False
    return [str(name) for name in datasets], False


def _matching_data_correction_refs(
    inversion: Any,
    dataset_names: Sequence[str],
    *,
    transform: Any | None,
    owner: str | None,
    components: tuple[str, ...] | None,
    skip_unmatched: bool,
) -> list[DataCorrectionParameterRef]:
    owners = [get_fault_by_name(inversion, owner)] if owner is not None else get_faults_from_inversion(inversion)
    refs: list[DataCorrectionParameterRef] = []

    for dataset in dataset_names:
        data = _get_data_by_name(inversion, dataset)
        candidates = []
        for fault in owners:
            poly = getattr(fault, "poly", {}) or {}
            if dataset not in poly or poly[dataset] is None:
                continue
            if transform is not None and not _transform_config_has_unique_match(
                fault, data, dataset, transform
            ):
                continue
            candidates.append(fault)

        if len(candidates) > 1:
            names = ", ".join(str(getattr(fault, "name", "")) for fault in candidates)
            raise ValueError(
                f"Multiple data-correction owners found for dataset '{dataset}': {names}. "
                "Pass owner=... explicitly."
            )
        if not candidates:
            if skip_unmatched:
                continue
            target = f" transform {transform!r}" if transform is not None else ""
            raise ValueError(f"No data-correction owner found for dataset '{dataset}'{target}")

        refs.append(
            DataCorrectionParameterRef(
                dataset=dataset,
                transform=transform,
                owner=str(getattr(candidates[0], "name", "")),
                components=components,
            )
        )

    return refs


def _scalar_components(n_params: int) -> list[str]:
    if n_params == 1:
        return ["offset"]
    if n_params == 3:
        return ["offset", "x_ramp", "y_ramp"]
    if n_params == 4:
        return ["offset", "x_ramp", "y_ramp", "xy_cross"]
    return [f"p{i}" for i in range(n_params)]


def _transform_components(transform: Any, n_params: int, data: Any = None) -> list[str]:
    if isinstance(transform, (int, np.integer)):
        return _scalar_components(n_params)

    key = str(transform).lower()
    if key == "translation":
        return ["east", "north"] + (["up"] if n_params >= 3 else [])
    if key == "translationrotation":
        return ["east", "north", "omega"] + (["up"] if n_params >= 4 else [])
    if key == "strainonly":
        return ["exx", "exy", "eyy"] + (["up"] if n_params >= 4 else [])
    if key == "strainnorotation":
        return ["east", "north", "exx", "exy", "eyy"] + (["up"] if n_params >= 6 else [])
    if key == "strainnotranslation":
        return ["exx", "exy", "eyy", "omega"] + (["up"] if n_params >= 5 else [])
    if key == "strain":
        if n_params == 3:
            return ["exx", "exy", "eyy"]
        return ["east", "north", "exx", "exy", "eyy", "omega"] + (["up"] if n_params >= 7 else [])
    if key == "full":
        if n_params == 7:
            return ["east", "north", "up", "rx", "ry", "rz", "scale"]
        return ["east", "north", "theta", "scale"]
    if key == "eulerrotation":
        return ["wx", "wy", "wz"]
    if key == "internalstrain":
        return ["sxx", "syy", "sxy"]
    return [f"p{i}" for i in range(n_params)]


_ALIASES = {
    "tx": "east",
    "ty": "north",
    "tz": "up",
    "e": "east",
    "n": "north",
    "u": "up",
    "rotation": "omega",
    "rot": "omega",
    "theta_cw": "rotation_cw",
    "rotation_cw_per_coord": "rotation_cw",
    "rz_cw": "rz",
    "xy": "xy_cross",
    "xy_ramp": "xy_cross",
    "xy_cross_term": "xy_cross",
}


def _canonical_component(component: str) -> str:
    key = str(component).lower()
    return _ALIASES.get(key, key)


def _component_index(components: Sequence[str], requested: str) -> tuple[int, str]:
    requested_key = _canonical_component(requested)
    lookup = {_canonical_component(component): i for i, component in enumerate(components)}

    # Physical rotation aliases resolve to the raw rotation component.
    if requested_key == "rotation_cw" and "omega" in lookup:
        return lookup["omega"], "rotation_cw"
    if requested_key == "rotation_cw" and "theta" in lookup:
        return lookup["theta"], "rotation_cw"
    if requested_key == "rotation_cw" and "rz" in lookup:
        return lookup["rz"], "rotation_cw"
    if requested_key == "omega" and "theta" in lookup:
        return lookup["theta"], "theta"
    if requested_key == "omega" and "rz" in lookup:
        return lookup["rz"], "rz"
    if requested_key == "theta" and "omega" in lookup:
        return lookup["omega"], "omega"
    if requested_key == "strain_xy" and "exy" in lookup:
        return lookup["exy"], "strain_xy"
    if requested_key == "strain_xy" and "sxy" in lookup:
        return lookup["sxy"], "strain_xy"
    if requested_key in lookup:
        return lookup[requested_key], requested_key
    raise ValueError(f"Component '{requested}' not found. Available components: {list(components)}")


def _finite_scale(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing normalization value '{name}'")
    scale = float(value)
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError(f"Invalid normalization value '{name}': {value}")
    return scale


def _physical_scale(data: Any, transform: Any, component: str) -> float:
    component = _canonical_component(component)
    norm = extract_normalization(data, transform)

    if isinstance(transform, (int, np.integer)):
        if component == "offset":
            return 1.0
        if component == "x_ramp":
            return 1.0 / _finite_scale(norm.get("x"), "x")
        if component == "y_ramp":
            return 1.0 / _finite_scale(norm.get("y"), "y")
        if component == "xy_cross":
            x = _finite_scale(norm.get("x"), "x")
            y = _finite_scale(norm.get("y"), "y")
            return 1.0 / (x * y)
        return 1.0

    key = str(transform).lower()
    if key in {
        "translation",
        "translationrotation",
        "strainonly",
        "strainnorotation",
        "strainnotranslation",
        "strain",
    }:
        if component in {"east", "north", "up"}:
            return 1.0
        base = _finite_scale(norm.get("base"), "base")
        if component == "rotation_cw":
            return 1.0 / (2.0 * base)
        if component == "strain_xy":
            return 0.5 / base
        if component in {"exx", "exy", "eyy", "omega"}:
            return 1.0 / base
        return 1.0

    if key == "full":
        if component in {"east", "north", "up"}:
            return 1.0
        base = _finite_scale(norm.get("base"), "base")
        if component in {"theta", "rx", "ry", "rz", "scale", "rotation_cw"}:
            return 1.0 / base
        return 1.0

    if key == "internalstrain":
        if component in {"strain_xy", "sxy"}:
            return 0.5
        return 1.0

    if key == "eulerrotation":
        return 1.0

    return 1.0


def _parse_bounds_pair(value: Any, label: str) -> tuple[float, float]:
    if isinstance(value, str):
        raise ValueError(f"{label} must be [lower, upper], not a string")
    items = list(value) if isinstance(value, Sequence) else [value]
    if len(items) == 3 and isinstance(items[0], str):
        items = items[1:]
    if len(items) != 2:
        raise ValueError(f"{label} must be [lower, upper] or [Distribution, lower, upper]")
    lower, upper = float(items[0]), float(items[1])
    if np.isnan(lower) or np.isnan(upper):
        raise ValueError(f"{label} cannot contain NaN")
    if lower > upper:
        raise ValueError(f"{label} has lower bound greater than upper bound: {lower} > {upper}")
    return lower, upper


def _bounds_for_components(bounds: Any, components: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(bounds, Mapping):
        bounds_lookup = {_canonical_component(key): value for key, value in bounds.items()}
        lower = []
        upper = []
        for component in components:
            key = _canonical_component(component)
            if key not in bounds_lookup:
                raise ValueError(f"Missing bounds for component '{component}'")
            lb, ub = _parse_bounds_pair(bounds_lookup[key], f"bounds for component '{component}'")
            lower.append(lb)
            upper.append(ub)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    lb, ub = _parse_bounds_pair(bounds, "bounds")
    return (
        np.full(len(components), lb, dtype=float),
        np.full(len(components), ub, dtype=float),
    )


def _convert_bounds_to_raw(
    lower: np.ndarray,
    upper: np.ndarray,
    scales: np.ndarray,
    *,
    space: str,
) -> tuple[np.ndarray, np.ndarray]:
    if space == "raw":
        return lower, upper

    scale_array = np.asarray(scales, dtype=float)
    if np.any(~np.isfinite(scale_array)) or np.any(scale_array == 0.0):
        raise ValueError(f"Cannot convert physical bounds with invalid scales: {scale_array.tolist()}")

    raw_lower = lower / scale_array
    raw_upper = upper / scale_array
    swap = raw_lower > raw_upper
    if np.any(swap):
        swapped_lower = raw_upper.copy()
        raw_upper[swap] = raw_lower[swap]
        raw_lower[swap] = swapped_lower[swap]
    return raw_lower, raw_upper


def resolve_data_correction_parameters(
    inversion: Any,
    ref: Mapping[str, Any] | DataCorrectionParameterRef,
    *,
    space: str = "raw",
) -> ResolvedDataCorrectionParameter:
    """Resolve data-correction parameter columns and scaling factors.

    ``space="raw"`` returns unit scaling.  ``space="physical"`` applies the
    simple per-parameter conversion used by the reporting helpers, for example
    ramp coefficients divided by their coordinate normalization lengths.
    """
    if space not in {"raw", "physical"}:
        raise ValueError("space must be 'raw' or 'physical'")

    ref_obj = _as_ref(ref)
    data = _get_data_by_name(inversion, ref_obj.dataset)
    owner = _resolve_owner(inversion, ref_obj.owner, ref_obj.dataset)
    transform, local_start, n_params = _select_transform_block(
        owner,
        data,
        ref_obj.dataset,
        ref_obj.transform,
    )
    dataset_start = _get_dataset_poly_start(inversion, owner, ref_obj.dataset)
    layout = _transform_components(transform, n_params, data=data)
    requested = ref_obj.components if ref_obj.components is not None else tuple(layout)

    columns = []
    scales = []
    resolved_components = []
    for component in requested:
        local_index, canonical = _component_index(layout, component)
        columns.append(dataset_start + local_start + local_index)
        resolved_components.append(canonical)
        if space == "raw":
            scales.append(1.0)
        else:
            scales.append(_physical_scale(data, transform, canonical))

    return ResolvedDataCorrectionParameter(
        owner=str(getattr(owner, "name", "")),
        dataset=ref_obj.dataset,
        transform=transform,
        components=tuple(resolved_components),
        columns=np.asarray(columns, dtype=int),
        scales=np.asarray(scales, dtype=float),
        space=space,
    )


def build_data_correction_equality_matrix(
    inversion: Any,
    refs: Sequence[Mapping[str, Any] | DataCorrectionParameterRef],
    *,
    space: str = "raw",
    n_parameters: int | None = None,
    linear_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build ``Aeq, beq`` tying two or more data-correction refs together.

    The first reference is the anchor.  Each subsequent reference is constrained
    component-by-component to match it in the requested parameter space.
    """
    if len(refs) < 2:
        raise ValueError("At least two references are required for an equality constraint")

    resolved = [resolve_data_correction_parameters(inversion, ref, space=space) for ref in refs]
    anchor = resolved[0]
    for item in resolved[1:]:
        if item.components != anchor.components:
            raise ValueError(
                "All references must resolve to the same components. "
                f"Got {anchor.components} and {item.components}."
            )

    if n_parameters is None:
        n_parameters = int(getattr(inversion, "lsq_parameters"))

    n_components = len(anchor.components)
    n_rows = n_components * (len(resolved) - 1)
    Aeq = np.zeros((n_rows, int(n_parameters)), dtype=float)
    beq = np.zeros(n_rows, dtype=float)

    row = 0
    for item in resolved[1:]:
        for i in range(n_components):
            anchor_col = int(anchor.columns[i] - linear_offset)
            item_col = int(item.columns[i] - linear_offset)
            if anchor_col < 0 or anchor_col >= n_parameters or item_col < 0 or item_col >= n_parameters:
                raise ValueError(
                    "Resolved data-correction column is outside the target parameter vector. "
                    f"Columns {anchor.columns[i]}, {item.columns[i]}, linear_offset={linear_offset}, "
                    f"n_parameters={n_parameters}."
                )
            Aeq[row, anchor_col] = float(anchor.scales[i])
            Aeq[row, item_col] = -float(item.scales[i])
            row += 1

    metadata = {
        "space": space,
        "components": list(anchor.components),
        "references": [
            {
                "owner": item.owner,
                "dataset": item.dataset,
                "transform": item.transform,
                "columns": item.columns.tolist(),
                "scales": item.scales.tolist(),
            }
            for item in resolved
        ],
    }
    return Aeq, beq, metadata


class DataCorrectionConstraintMixin:
    """Mixin for adding linear relations between data-correction parameters."""

    def resolve_data_correction_parameters(
        self,
        ref: Mapping[str, Any] | DataCorrectionParameterRef,
        *,
        space: str = "raw",
    ) -> ResolvedDataCorrectionParameter:
        """Resolve one named data-correction reference to model-vector columns."""
        return resolve_data_correction_parameters(self, ref, space=space)

    def set_data_correction_bounds(
        self,
        *,
        bounds: Mapping[str, Sequence[float]] | Sequence[float],
        dataset: str | None = None,
        datasets: str | Sequence[str] | None = None,
        transform: Any | None = None,
        owner: str | None = None,
        components: Sequence[str] | str | None = None,
        space: str = "raw",
        source: str = "data_correction_bounds",
    ) -> list[dict[str, Any]]:
        """Set bounds for selected data-correction transform components.

        Parameters
        ----------
        bounds : mapping or sequence
            Either one ``[lower, upper]`` pair applied to all selected
            components, or a component-to-bounds mapping such as
            ``{"east": [-5, 5], "north": [-5, 5]}``.  Values may also use the
            existing config style ``[Distribution, lower, upper]``; only the
            numeric lower/upper limits are used here.
        dataset, datasets : str or sequence
            Target one dataset, a list of datasets, or ``datasets="all"``.
            ``datasets="all"`` requires ``transform=...`` and updates only
            datasets that actually configure that transform.
        transform : str or int, optional
            Transform to select.  Required for composite ``geodata.polys``
            lists such as ``["eulerrotation", "translation"]``.
        owner : str, optional
            Source/fault that owns the data-correction columns.  Required only
            when more than one source configures the same dataset transform.
        components : sequence of str, optional
            Components to bound.  If omitted and ``bounds`` is a mapping, the
            mapping keys define the selected components.  If omitted and
            ``bounds`` is one pair, all components of the selected transform
            are bounded.
        space : {"raw", "physical"}
            ``raw`` writes solver coefficients directly.  ``physical`` accepts
            simple physical-gradient limits and converts them back to raw
            solver coefficients using the stored normalization factors.

        Returns
        -------
        list of dict
            One metadata dictionary per resolved dataset/owner pair.
        """
        if space not in {"raw", "physical"}:
            raise ValueError("space must be 'raw' or 'physical'")

        component_list = _normalise_components(components)
        if component_list is None and isinstance(bounds, Mapping):
            component_list = tuple(str(component) for component in bounds.keys())

        dataset_names, skip_unmatched = _normalise_dataset_names(
            self,
            dataset=dataset,
            datasets=datasets,
        )
        if skip_unmatched and transform is None:
            raise ValueError("datasets='all' requires transform=... to avoid ambiguous bulk updates")

        refs = _matching_data_correction_refs(
            self,
            dataset_names,
            transform=transform,
            owner=owner,
            components=component_list,
            skip_unmatched=skip_unmatched,
        )
        if not refs:
            raise ValueError("No configured data-correction parameters matched the requested selector")

        manager = getattr(self, "constraint_manager", None)
        if manager is None or not hasattr(manager, "set_parameter_bounds_by_indices"):
            raise AttributeError(
                "This inversion object has no constraint manager with "
                "set_parameter_bounds_by_indices()"
            )

        metadata = []
        for ref in refs:
            resolved = resolve_data_correction_parameters(self, ref, space=space)
            input_lb, input_ub = _bounds_for_components(bounds, resolved.components)
            raw_lb, raw_ub = _convert_bounds_to_raw(
                input_lb,
                input_ub,
                resolved.scales,
                space=space,
            )
            manager.set_parameter_bounds_by_indices(
                resolved.columns,
                raw_lb,
                raw_ub,
                source=source,
            )
            metadata.append({
                "owner": resolved.owner,
                "dataset": resolved.dataset,
                "transform": resolved.transform,
                "components": list(resolved.components),
                "columns": resolved.columns.tolist(),
                "space": space,
                "scales": resolved.scales.tolist(),
                "input_bounds": np.column_stack([input_lb, input_ub]).tolist(),
                "raw_bounds": np.column_stack([raw_lb, raw_ub]).tolist(),
            })

        sync_to_solver = getattr(manager, "sync_to_solver", None)
        if callable(sync_to_solver):
            sync_to_solver()
        return metadata

    def build_data_correction_equality(
        self,
        refs: Sequence[Mapping[str, Any] | DataCorrectionParameterRef],
        *,
        space: str = "raw",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Build but do not register a data-correction equality constraint."""
        n_parameters, linear_offset = self._data_correction_constraint_shape()
        return build_data_correction_equality_matrix(
            self,
            refs,
            space=space,
            n_parameters=n_parameters,
            linear_offset=linear_offset,
        )

    def add_data_correction_equality(
        self,
        refs: Sequence[Mapping[str, Any] | DataCorrectionParameterRef],
        *,
        space: str = "raw",
        name: str = "data_correction_equality",
        source: str = "data_correction_constraints",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Add a hard equality relation between data-correction parameters.

        The method returns the generated ``Aeq``, ``beq`` and metadata so users
        can inspect the exact columns that were tied together.
        """
        Aeq, beq, metadata = self.build_data_correction_equality(refs, space=space)
        add_equality = getattr(self, "add_equality_constraint", None)
        add_custom_equality = getattr(self, "add_custom_equality_constraint", None)
        if callable(add_equality):
            self.add_equality_constraint(Aeq, beq, name=name, source=source)
        elif callable(add_custom_equality):
            self.add_custom_equality_constraint(Aeq, beq, name=name, source=source)
        else:
            raise AttributeError("This inversion object has no equality-constraint registration method")
        return Aeq, beq, metadata

    def _data_correction_constraint_shape(self) -> tuple[int, int]:
        """Return target parameter count and global-to-target column offset."""
        if hasattr(self, "linear_sample_start_position") and callable(getattr(self, "add_custom_equality_constraint", None)):
            linear_offset = int(getattr(self, "linear_sample_start_position"))
            return int(getattr(self, "lsq_parameters")), linear_offset
        return int(getattr(self, "lsq_parameters")), 0
