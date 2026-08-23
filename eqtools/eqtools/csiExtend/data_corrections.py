"""Data-correction specifications for nonlinear geometry SMC.

For each dataset, the nonlinear prediction is assembled as

``synthetic_total = synthetic_fault + H_correction @ coefficients``.

This module owns ``H_correction`` and the corresponding coefficient metadata;
it does not change the fault-geometry parameters.  Every design-matrix row must
therefore follow the same observation layout as the synthetic vector and the
dataset covariance.  In particular, CSI GPS rows are component-major:
``E(all stations), N(all stations), [U(all stations)]``.

The correction names and transform semantics remain consistent with the
linear BLSE/Bayesian ``geodata.polys`` setting, while the nonlinear layer keeps
their priors, fixed values, and sampled-parameter slices explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


NO_TRANSFORM_VALUES = {None, "none", "None", "null", "Null", "NULL"}
SAR_TRANSFORMS = {1, 3, 4}
LEVELING_TRANSFORMS = {1, 3, 4}
GPS_TRANSFORMS = {"translation"}
SUPPORTED_MODES = {"sampled", "fixed"}
DEFAULT_POLY_BOUNDS = ["Uniform", -1000.0, 2000.0]


@dataclass(frozen=True)
class DataCorrectionSpec:
    """Normalized correction configuration for one geodetic dataset."""

    data_name: str
    data_type: str
    transform: Any
    mode: str
    parameter_names: List[str]
    priors: Dict[str, Any] = field(default_factory=dict)
    values: Dict[str, float] = field(default_factory=dict)
    display_names: Dict[str, str] = field(default_factory=dict)
    parameter_slice: Optional[slice] = None

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)

    @property
    def sampled(self) -> bool:
        return self.mode == "sampled" and self.n_parameters > 0

    def full_parameter_names(self) -> List[str]:
        return [
            f"data_corrections.{self.data_name}.{name}"
            for name in self.parameter_names
        ]

    def display_parameter_names(self) -> List[str]:
        return [
            self.display_names.get(name, name)
            for name in self.parameter_names
        ]


def is_transform_enabled(transform: Any) -> bool:
    """Return False for null/no-correction transform values."""
    try:
        return transform not in NO_TRANSFORM_VALUES
    except TypeError:
        return True


def _data_type(data: Any) -> str:
    dtype = getattr(data, "dtype", None)
    if dtype is None:
        raise ValueError(f"Dataset {getattr(data, 'name', '<unnamed>')} has no dtype")
    return str(dtype).lower()


def _coerce_int_transform(transform: Any, data_name: str) -> int:
    if isinstance(transform, bool):
        raise ValueError(f"{data_name}: boolean transform is not valid")
    try:
        value = int(transform)
    except (TypeError, ValueError):
        raise ValueError(f"{data_name}: transform must be an integer, got {transform!r}")
    return value


def canonical_correction_parameter_names(
    data: Any,
    transform: Any,
    *,
    vertical: bool = True,
) -> List[str]:
    """Return canonical correction parameter names for one dataset."""
    if not is_transform_enabled(transform):
        return []

    dtype = _data_type(data)
    data_name = getattr(data, "name", "<unnamed>")

    if dtype in {"insar", "sar"}:
        ptype = _coerce_int_transform(transform, data_name)
        if ptype == 1:
            return ["offset"]
        if ptype == 3:
            return ["offset", "x_ramp", "y_ramp"]
        if ptype == 4:
            return ["offset", "x_ramp", "y_ramp", "xy_ramp"]
        if ptype == 5:
            raise ValueError(
                f"{data_name}: SAR/InSAR transform=5 is not supported by the "
                "current CSI InSAR basis. Use 1, 3, or 4."
            )
        raise ValueError(
            f"{data_name}: SAR/InSAR transform must be one of 1, 3, 4; got {transform!r}"
        )

    if dtype == "leveling":
        ptype = _coerce_int_transform(transform, data_name)
        if ptype == 1:
            return ["offset"]
        if ptype == 3:
            return ["offset", "x_ramp", "y_ramp"]
        if ptype == 4:
            return ["offset", "x_ramp", "y_ramp", "xy_ramp"]
        raise ValueError(
            f"{data_name}: leveling transform must be one of 1, 3, 4; got {transform!r}"
        )

    if dtype == "gps":
        transform_name = str(transform).lower()
        if transform_name != "translation":
            raise ValueError(
                f"{data_name}: nonlinear GPS data corrections currently support "
                "only transform='translation'. More complex CSI transforms should "
                "be added only after their basis order is mirrored exactly."
            )
        names = ["east_offset", "north_offset"]
        if vertical and _gps_has_vertical(data):
            names.append("up_offset")
        return names

    raise ValueError(
        f"{data_name}: data corrections are not supported for dtype={dtype!r}"
    )


def _gps_has_vertical(data: Any) -> bool:
    vel = getattr(data, "vel_enu", None)
    if vel is None:
        return True
    arr = np.asarray(vel)
    return arr.ndim == 2 and arr.shape[1] >= 3


def _coordinate_arrays(data: Any) -> tuple[np.ndarray, np.ndarray]:
    data_name = getattr(data, "name", "<unnamed>")
    if not hasattr(data, "x") or not hasattr(data, "y"):
        raise ValueError(f"{data_name}: correction transform requires x/y coordinates")
    x = np.asarray(data.x, dtype=float).reshape(-1)
    y = np.asarray(data.y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"{data_name}: x and y coordinate arrays must have the same shape")
    return x, y


def build_correction_design_matrix(
    data: Any,
    transform: Any,
    *,
    vertical: bool = True,
) -> np.ndarray:
    """Build ``H_correction`` for one dataset and active component set.

    Parameters
    ----------
    data : object
        CSI geodetic dataset providing ``dtype`` and
        ``getTransformEstimator()``.  GPS datasets also provide station data
        through ``vel_enu`` or ``x`` and must already have the CSI
        ``obs_per_station`` state established by the Green-function setup.
    transform : object
        CSI transform selector, such as ``1``, ``3``, or ``4`` for scalar
        datasets and ``"translation"`` for nonlinear GPS corrections.
    vertical : bool, default True
        Whether the inversion data vector includes the GPS Up component.
        This option does not reorder rows: GPS output remains
        ``E(all), N(all), [U(all)]``.

    Returns
    -------
    numpy.ndarray
        Design matrix whose rows match the active observation, synthetic, and
        covariance rows, and whose columns match
        :func:`canonical_correction_parameter_names` exactly.

    Notes
    -----
    CSI may return an ENU GPS translation estimator even when the current
    inversion uses only EN.  The GPS alignment helper removes the inactive Up
    rows and the structurally zero Up-translation column; it never converts
    CSI's component-major rows to station-major order.
    """
    names = canonical_correction_parameter_names(data, transform, vertical=vertical)
    if not names:
        return np.zeros((0, 0), dtype=float)

    dtype = _data_type(data)
    estimator = _call_data_transform_estimator(data, transform)
    if estimator is None:
        raise ValueError(
            f"{getattr(data, 'name', '<unnamed>')}: getTransformEstimator returned None "
            f"for transform={transform!r}"
        )
    design_matrix = np.asarray(estimator, dtype=float)
    if design_matrix.ndim != 2:
        raise ValueError(
            f"{getattr(data, 'name', '<unnamed>')}: correction estimator must be 2D, "
            f"got shape {design_matrix.shape}"
        )

    if dtype == "gps":
        design_matrix = _align_gps_estimator_to_active_components(
            data,
            design_matrix,
            vertical=vertical,
            n_active_parameters=len(names),
        )

    if design_matrix.shape[1] != len(names):
        raise ValueError(
            f"{getattr(data, 'name', '<unnamed>')}: transform={transform!r} produced "
            f"{design_matrix.shape[1]} columns, but canonical names are {names}"
        )
    return design_matrix


def _call_data_transform_estimator(data: Any, transform: Any) -> Any:
    """Call a CSI transform estimator across supported keyword signatures.

    Current CSI estimators accept both normalization flags, while older or
    data-type-specific implementations may accept only one of them or only the
    transform selector.  The returned object is the raw estimator; row and
    column alignment is handled by :func:`build_correction_design_matrix`.
    """
    data_name = getattr(data, "name", "<unnamed>")
    if not hasattr(data, "getTransformEstimator"):
        raise ValueError(
            f"{data_name}: data object does not provide getTransformEstimator(); "
            "cannot build a scientifically consistent correction matrix"
        )
    try:
        return data.getTransformEstimator(
            transform,
            computeNormFact=True,
            computeIntStrainNormFact=True,
            verbose=False,
        )
    except TypeError:
        try:
            return data.getTransformEstimator(
                transform,
                computeNormFact=True,
                computeIntStrainNormFact=True,
            )
        except TypeError:
            return data.getTransformEstimator(transform)


def _align_gps_estimator_to_active_components(
    data: Any,
    raw_estimator: np.ndarray,
    *,
    vertical: bool,
    n_active_parameters: int,
) -> np.ndarray:
    """Align a CSI GPS estimator with the active data rows and parameters.

    Parameters
    ----------
    data : object
        CSI GPS dataset used to determine ``n_stations``.
    raw_estimator : numpy.ndarray
        Matrix returned by ``data.getTransformEstimator()``.  Its rows already
        follow CSI component-major order: ``E(all)``, ``N(all)``, then
        optionally ``U(all)``.  Columns represent transform parameters.
    vertical : bool
        Retain ENU observation rows when true; retain only EN rows when false.
    n_active_parameters : int
        Number of normalized correction parameters used by this inversion.
        GPS translation uses three parameters for ENU and two for EN.

    Returns
    -------
    numpy.ndarray
        ``H_correction`` aligned with the active observation vector.  For
        ``n_stations = N``, GPS translation returns shape ``(3N, 3)`` for ENU
        or ``(2N, 2)`` for EN.

    Notes
    -----
    This function selects active component rows; it never reorders them.  An
    inactive parameter column may be removed only when it is zero on every
    retained row, preventing a future CSI estimator change from being silently
    discarded.
    """
    n_stations = _gps_station_count(data)
    if raw_estimator.shape[0] not in {
        n_stations * 2,
        n_stations * 3,
        n_stations,
    }:
        raise ValueError(
            f"{getattr(data, 'name', '<unnamed>')}: unexpected GPS estimator row count "
            f"{raw_estimator.shape[0]} for {n_stations} stations"
        )

    if vertical and raw_estimator.shape[0] == n_stations * 2:
        raise ValueError(
            f"{getattr(data, 'name', '<unnamed>')}: vertical GPS correction "
            "requires an estimator with ENU rows"
        )

    active_design_matrix = raw_estimator
    if not vertical and raw_estimator.shape[0] == n_stations * 3:
        # Horizontal inversion retains the first two component blocks.  CSI
        # already orders these rows as E(all) followed by N(all).
        active_design_matrix = raw_estimator[: 2 * n_stations, :]

    if active_design_matrix.shape[1] > n_active_parameters:
        inactive_parameter_columns = active_design_matrix[:, n_active_parameters:]
        if np.any(np.abs(inactive_parameter_columns) > 0):
            raise ValueError(
                f"{getattr(data, 'name', '<unnamed>')}: cannot drop non-zero GPS "
                "correction estimator columns"
            )
        # For horizontal translation this removes the Up-offset column only
        # after proving that it contributes nothing to the retained EN rows.
        active_design_matrix = active_design_matrix[:, :n_active_parameters]
    return active_design_matrix


def _gps_station_count(data: Any) -> int:
    vel = getattr(data, "vel_enu", None)
    if vel is not None:
        arr = np.asarray(vel)
        if arr.ndim == 2:
            return arr.shape[0]
    if hasattr(data, "x"):
        return np.asarray(data.x).reshape(-1).size
    raise ValueError(f"{getattr(data, 'name', '<unnamed>')}: cannot infer GPS station count")


def expand_polys_transforms(geodata: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve basic ``geodata.polys`` into per-dataset transforms."""
    datas = list(geodata.get("data", []))
    names = [getattr(data, "name", None) for data in datas]
    polys = geodata.get("polys", None)
    transforms = {name: None for name in names}

    if isinstance(polys, dict):
        if polys.get("enabled", False):
            estimate = polys.get("estimate") or names
            for name in estimate:
                if name in transforms:
                    transforms[name] = 1
        return transforms

    if isinstance(polys, list):
        if len(polys) != len(datas):
            raise ValueError(
                f"Length of geodata.polys ({len(polys)}) does not match "
                f"number of datasets ({len(datas)})"
            )
        return dict(zip(names, polys))

    if is_transform_enabled(polys):
        return {name: polys for name in names}

    return transforms


def normalize_data_correction_specs(
    geodata: Mapping[str, Any],
    *,
    default_prior: Any = None,
    verticals: Optional[Sequence[bool]] = None,
) -> List[DataCorrectionSpec]:
    """Normalize basic and advanced correction config into specs."""
    datas = list(geodata.get("data", []))
    if verticals is None:
        verticals = geodata.get("verticals", [True] * len(datas))
    if isinstance(verticals, bool):
        verticals = [verticals] * len(datas)
    if len(verticals) != len(datas):
        raise ValueError("Length of geodata.verticals must match geodata.data")

    transforms = expand_polys_transforms(geodata)
    corrections = geodata.get("data_corrections") or {}
    enabled = corrections.get("enabled", True)
    if enabled is False:
        return []

    defaults = corrections.get("defaults") or {}
    dataset_configs = corrections.get("datasets") or {}
    default_mode = defaults.get("mode", "sampled")
    default_transform = defaults.get("transform", None)
    if default_prior is None:
        default_prior = (
            geodata.get("poly_bounds")
            or _single_value_alias(
                defaults,
                primary="bounds",
                aliases=("prior",),
                default=DEFAULT_POLY_BOUNDS,
                context="geodata.data_corrections.defaults",
            )
        )

    specs = []
    used_display_names = set()
    for data, vertical in zip(datas, verticals):
        name = getattr(data, "name", None)
        if name is None:
            raise ValueError("All geodata objects must have a name")

        dcfg = dataset_configs.get(name, {}) or {}
        transform = dcfg.get("transform", transforms.get(name, default_transform))
        if "transform" not in dcfg and transforms.get(name) is None:
            transform = default_transform
        if not is_transform_enabled(transform):
            continue

        mode = str(dcfg.get("mode", default_mode)).lower()
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"{name}: unsupported data correction mode {mode!r}; "
                f"expected one of {sorted(SUPPORTED_MODES)}"
            )

        param_names = canonical_correction_parameter_names(data, transform, vertical=vertical)
        dataset_prior = _single_value_alias(
            dcfg,
            primary="bounds",
            aliases=("prior",),
            default=default_prior,
            context=f"geodata.data_corrections.datasets.{name}",
        )
        parameter_bounds = _mapping_alias(
            dcfg,
            primary="parameter_bounds",
            aliases=("priors",),
            default={},
            context=f"geodata.data_corrections.datasets.{name}",
        )
        priors = _resolve_parameter_mapping(
            parameter_bounds,
            param_names,
            default_value=dataset_prior,
            dataset_name=name,
            field_name="parameter_bounds",
        )
        values = _resolve_values(dcfg.get("values", {}), param_names, name, mode)
        display_names = _resolve_display_names(
            dcfg.get("display_names", {}),
            param_names,
            name,
            used_display_names,
        )

        specs.append(
            DataCorrectionSpec(
                data_name=name,
                data_type=_data_type(data),
                transform=transform,
                mode=mode,
                parameter_names=param_names,
                priors=priors,
                values=values,
                display_names=display_names,
            )
        )

    return specs


def _single_value_alias(
    config: Mapping[str, Any],
    *,
    primary: str,
    aliases: Sequence[str],
    default: Any,
    context: str,
) -> Any:
    keys = [key for key in (primary, *aliases) if key in config]
    if len(keys) > 1:
        raise ValueError(
            f"{context}: use only one of {keys}; prefer '{primary}'"
        )
    if keys:
        return config[keys[0]]
    return default


def _mapping_alias(
    config: Mapping[str, Any],
    *,
    primary: str,
    aliases: Sequence[str],
    default: Mapping[str, Any],
    context: str,
) -> Mapping[str, Any]:
    value = _single_value_alias(
        config,
        primary=primary,
        aliases=aliases,
        default=default,
        context=context,
    )
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.{primary} must be a mapping")
    return value


def _resolve_parameter_mapping(
    user_mapping: Mapping[str, Any],
    param_names: Sequence[str],
    *,
    default_value: Any,
    dataset_name: str,
    field_name: str,
) -> Dict[str, Any]:
    unknown = set(user_mapping) - set(param_names)
    if unknown:
        raise ValueError(
            f"{dataset_name}: unknown {field_name} keys {sorted(unknown)}; "
            f"expected canonical names {list(param_names)}"
        )
    return {
        name: user_mapping.get(name, default_value)
        for name in param_names
    }


def _resolve_values(
    values: Any,
    param_names: Sequence[str],
    dataset_name: str,
    mode: str,
) -> Dict[str, float]:
    if mode != "fixed":
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{dataset_name}: fixed data corrections require values mapping")
    unknown = set(values) - set(param_names)
    missing = set(param_names) - set(values)
    if unknown or missing:
        raise ValueError(
            f"{dataset_name}: fixed values must match canonical names exactly; "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    return {name: float(values[name]) for name in param_names}


def _resolve_display_names(
    display_names: Any,
    param_names: Sequence[str],
    dataset_name: str,
    used_display_names: set[str],
) -> Dict[str, str]:
    if display_names in (None, {}):
        return {}
    if isinstance(display_names, (list, tuple)):
        if len(display_names) != len(param_names):
            raise ValueError(
                f"{dataset_name}: display_names list length ({len(display_names)}) "
                f"must match transform parameters ({len(param_names)}): {list(param_names)}"
            )
        resolved = dict(zip(param_names, display_names))
    elif isinstance(display_names, Mapping):
        unknown = set(display_names) - set(param_names)
        if unknown:
            raise ValueError(
                f"{dataset_name}: unknown display_names keys {sorted(unknown)}; "
                f"expected canonical names {list(param_names)}"
            )
        resolved = dict(display_names)
    else:
        raise ValueError(
            f"{dataset_name}: display_names must be a mapping or a list in "
            f"parameter order {list(param_names)}"
        )

    resolved = {
        name: _coerce_display_name(label, dataset_name, name)
        for name, label in resolved.items()
    }
    duplicates = set(resolved.values()) & used_display_names
    if duplicates:
        raise ValueError(
            f"{dataset_name}: duplicate display_names across data corrections: "
            f"{sorted(duplicates)}"
        )
    used_display_names.update(resolved.values())
    return resolved


def _coerce_display_name(label: Any, dataset_name: str, parameter_name: str) -> str:
    text = str(label)
    if not text.strip():
        raise ValueError(
            f"{dataset_name}: display name for {parameter_name} cannot be empty"
        )
    return text


def assign_parameter_slices(
    specs: Iterable[DataCorrectionSpec],
    *,
    start_index: int,
) -> List[DataCorrectionSpec]:
    """Return copies of specs with sampled-parameter slices assigned."""
    next_index = start_index
    resolved = []
    for spec in specs:
        if spec.sampled:
            param_slice = slice(next_index, next_index + spec.n_parameters)
            next_index += spec.n_parameters
        else:
            param_slice = None
        resolved.append(replace(spec, parameter_slice=param_slice))
    return resolved


def correction_coefficients_from_theta(
    spec: DataCorrectionSpec,
    theta: Sequence[float],
) -> np.ndarray:
    """Extract sampled or fixed coefficients for one correction spec."""
    if spec.mode == "fixed":
        return np.array([spec.values[name] for name in spec.parameter_names], dtype=float)
    if spec.parameter_slice is None:
        raise ValueError(f"{spec.data_name}: sampled correction has no parameter_slice")
    return np.asarray(theta[spec.parameter_slice], dtype=float)
