"""Interpret estimated data-correction parameters in physical terms.

The functions in this module do not build Green's functions and do not modify
an inversion result.  They translate CSI/ECAT correction coefficients together
with the normalization metadata stored on data objects into quantities that are
easier to inspect: velocity gradients, strain tensors, Helmert-like rotation
and Euler poles.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .config.config_utils import parse_euler_units, parse_observation_unit, standardize_euler_pole


ANGLE_UNITS = {
    "radians": 1.0,
    "degrees": np.pi / 180.0,
}

ANGULAR_VELOCITY_UNITS = {
    "radians_per_year": 1.0,
    "radians_per_myr": 1.0e-6,
    "degrees_per_year": np.pi / 180.0,
    "degrees_per_myr": np.pi / 180.0 * 1.0e-6,
}

GPS_STRAIN_TRANSFORMS = {
    "translation",
    "translationrotation",
    "strainonly",
    "strainnorotation",
    "strainnotranslation",
    "strain",
}


def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value == 0.0:
        return None
    return value


def _matrix_or_none(matrix: np.ndarray, scale: float | None) -> list[list[float]] | None:
    if scale is None:
        return None
    return (np.asarray(matrix, dtype=float) / scale).tolist()


def _observation_unit_info(observation_unit: str | None, default_observation_unit: str = "m") -> dict[str, Any]:
    return parse_observation_unit(observation_unit, default=default_observation_unit)


def _physical_gradient_matrix(
    gradient_per_km: np.ndarray | None,
    unit_info: Mapping[str, Any],
) -> list[list[float]] | None:
    if gradient_per_km is None or unit_info.get("to_si") is None:
        return None
    # CSI x/y are kilometers; convert observation length to meters and km to m.
    return (np.asarray(gradient_per_km, dtype=float) * float(unit_info["to_si"]) / 1000.0).tolist()


def _rotation_axis(vector: np.ndarray) -> dict[str, Any]:
    vec = np.asarray(vector, dtype=float).reshape(-1)
    magnitude = float(np.linalg.norm(vec))
    if magnitude == 0.0:
        lon_rad = np.nan
        lat_rad = np.nan
    else:
        lon_rad = float(np.arctan2(vec[1], vec[0]))
        lat_rad = float(np.arcsin(np.clip(vec[2] / magnitude, -1.0, 1.0)))
    return {
        "type": "rotation_axis",
        "value": [np.degrees(lon_rad), np.degrees(lat_rad), magnitude],
        "units": ["degrees", "degrees", "radians"],
    }


def euler_pole_to_vector(
    lon: float,
    lat: float,
    omega: float,
    units: Sequence[str] = ("degrees", "degrees", "degrees_per_myr"),
) -> np.ndarray:
    """Convert an Euler pole to a Cartesian vector in radians/year.

    Parameters
    ----------
    lon, lat, omega : float
        Euler pole longitude, latitude and angular velocity.
    units : sequence of str, default ("degrees", "degrees", "degrees_per_myr")
        Input units for ``lon``, ``lat`` and ``omega``.
    """
    _, vector = standardize_euler_pole([lon, lat, omega], list(units))
    return vector


def euler_vector_to_pole(
    vector: Sequence[float],
    output_units: Sequence[str] = ("degrees", "degrees", "degrees_per_myr"),
) -> dict[str, Any]:
    """Convert a Cartesian Euler vector in radians/year to pole form.

    Returns a dictionary with ``value = [lon, lat, omega]`` and the requested
    units.  A zero vector has undefined pole longitude/latitude and returns
    ``nan`` for those two values.
    """
    vec = _as_array(vector)
    if vec.size != 3:
        raise ValueError("Euler vector must contain exactly three values")
    factors = parse_euler_units(list(output_units), "euler_pole")["conversion_factors"]
    omega = float(np.linalg.norm(vec))
    if omega == 0.0:
        lon_rad = np.nan
        lat_rad = np.nan
    else:
        lon_rad = float(np.arctan2(vec[1], vec[0]))
        lat_rad = float(np.arcsin(np.clip(vec[2] / omega, -1.0, 1.0)))
    return {
        "type": "euler_pole",
        "value": [lon_rad / factors[0], lat_rad / factors[1], omega / factors[2]],
        "units": list(output_units),
        "vector_radians_per_year": vec.tolist(),
    }


def interpret_euler_vector_parameters(
    params: Sequence[float],
    output_units: Sequence[str] = ("degrees", "degrees", "degrees_per_myr"),
    observation_unit: str | None = None,
    default_observation_unit: str = "m/yr",
) -> dict[str, Any]:
    """Interpret ``eulerrotation`` coefficients using the observation unit."""
    vec = _as_array(params)
    if vec.size != 3:
        raise ValueError("eulerrotation expects exactly three parameters")
    unit_info = _observation_unit_info(observation_unit, default_observation_unit)
    physical_vec = vec * float(unit_info["to_si"])
    warnings = []
    if unit_info.get("assumed"):
        warnings.append(
            f"units.observation was not set; interpreting eulerrotation with assumed {unit_info['observation']}."
        )
    if unit_info["kind"] == "rate":
        pole = euler_vector_to_pole(physical_vec, output_units=output_units)
        physical = {
            "euler_vector_radians_per_year": physical_vec.tolist(),
            "euler_pole": pole,
        }
        physical_units = "radians_per_year"
    else:
        physical = {
            "rotation_vector_radians": physical_vec.tolist(),
            "rotation_axis": _rotation_axis(physical_vec),
        }
        physical_units = "radians"
    return {
        "kind": "eulerrotation",
        "raw_parameters": {
            "wx": float(vec[0]),
            "wy": float(vec[1]),
            "wz": float(vec[2]),
            "units": f"{unit_info['observation']}_per_meter",
        },
        "unit_context": unit_info,
        "physical": {
            **physical,
            "vector_units": physical_units,
        },
        "warnings": warnings,
    }


def _gps_strain_components(transform: str, params: Sequence[float]) -> tuple[dict[str, float], list[str]]:
    key = str(transform).lower()
    vals = _as_array(params)
    warnings: list[str] = []
    comp = {
        "tx": 0.0,
        "ty": 0.0,
        "exx": 0.0,
        "exy": 0.0,
        "eyy": 0.0,
        "omega": 0.0,
        "tz": np.nan,
    }
    expected = {
        "translation": {2, 3},
        "translationrotation": {3, 4},
        "strainonly": {3, 4},
        "strainnorotation": {5, 6},
        "strainnotranslation": {4, 5},
        "strain": {6, 7},
    }[key]
    if vals.size not in expected:
        warnings.append(f"Unexpected parameter count {vals.size} for {transform}; interpreting available columns.")

    if key == "translation":
        if vals.size >= 1:
            comp["tx"] = vals[0]
        if vals.size >= 2:
            comp["ty"] = vals[1]
        if vals.size >= 3:
            comp["tz"] = vals[2]
    elif key == "translationrotation":
        if vals.size >= 1:
            comp["tx"] = vals[0]
        if vals.size >= 2:
            comp["ty"] = vals[1]
        if vals.size >= 3:
            comp["omega"] = vals[2]
        if vals.size >= 4:
            comp["tz"] = vals[3]
    elif key == "strainonly":
        if vals.size >= 1:
            comp["exx"] = vals[0]
        if vals.size >= 2:
            comp["exy"] = vals[1]
        if vals.size >= 3:
            comp["eyy"] = vals[2]
        if vals.size >= 4:
            comp["tz"] = vals[3]
    elif key == "strainnorotation":
        if vals.size >= 1:
            comp["tx"] = vals[0]
        if vals.size >= 2:
            comp["ty"] = vals[1]
        if vals.size >= 3:
            comp["exx"] = vals[2]
        if vals.size >= 4:
            comp["exy"] = vals[3]
        if vals.size >= 5:
            comp["eyy"] = vals[4]
        if vals.size >= 6:
            comp["tz"] = vals[5]
    elif key == "strainnotranslation":
        if vals.size >= 1:
            comp["exx"] = vals[0]
        if vals.size >= 2:
            comp["exy"] = vals[1]
        if vals.size >= 3:
            comp["eyy"] = vals[2]
        if vals.size >= 4:
            comp["omega"] = vals[3]
        if vals.size >= 5:
            comp["tz"] = vals[4]
    elif key == "strain":
        if vals.size >= 1:
            comp["tx"] = vals[0]
        if vals.size >= 2:
            comp["ty"] = vals[1]
        if vals.size >= 3:
            comp["exx"] = vals[2]
        if vals.size >= 4:
            comp["exy"] = vals[3]
        if vals.size >= 5:
            comp["eyy"] = vals[4]
        if vals.size >= 6:
            comp["omega"] = vals[5]
        if vals.size >= 7:
            comp["tz"] = vals[6]
    return comp, warnings


def interpret_gps_strain_parameters(
    transform: str,
    params: Sequence[float],
    normalization: Mapping[str, Any] | None = None,
    observation_unit: str | None = None,
    default_observation_unit: str = "m",
) -> dict[str, Any]:
    """Interpret CSI ``get2DstrainEst``-family GPS parameters.

    Physical gradients are obtained by dividing the normalized velocity-gradient
    matrix by ``base``.  If the base is unavailable, the normalized matrix is
    still reported and physical gradients are set to ``None``.
    """
    key = str(transform).lower()
    if key not in GPS_STRAIN_TRANSFORMS:
        raise ValueError(f"{transform!r} is not a GPS strain-family transform")
    comp, warnings = _gps_strain_components(key, params)
    norm = dict(normalization or {})
    unit_info = _observation_unit_info(observation_unit, default_observation_unit)
    base = _finite_or_none(norm.get("base"))
    if base is None and key != "translation":
        warnings.append("Missing or zero strain normalization base; physical gradients are not reported.")

    a_norm = np.array(
        [
            [comp["exx"], 0.5 * (comp["exy"] + comp["omega"])],
            [0.5 * (comp["exy"] - comp["omega"]), comp["eyy"]],
        ],
        dtype=float,
    )
    strain_norm = np.array(
        [[comp["exx"], 0.5 * comp["exy"]], [0.5 * comp["exy"], comp["eyy"]]],
        dtype=float,
    )
    rotation_norm = np.array(
        [[0.0, 0.5 * comp["omega"]], [-0.5 * comp["omega"], 0.0]],
        dtype=float,
    )
    gradient_per_km = None if base is None else np.asarray(a_norm, dtype=float) / base
    strain_per_km = None if base is None else np.asarray(strain_norm, dtype=float) / base
    rotation_per_km = None if base is None else np.asarray(rotation_norm, dtype=float) / base
    return {
        "kind": key,
        "raw_parameters": {name: float(value) for name, value in comp.items() if np.isfinite(value)},
        "normalization": norm,
        "unit_context": unit_info,
        "physical": {
            "translation": {"east": comp["tx"], "north": comp["ty"], **({"up": comp["tz"]} if np.isfinite(comp["tz"]) else {})},
            "gradient_matrix_normalized": a_norm.tolist(),
            "gradient_matrix_per_coord": _matrix_or_none(a_norm, base),
            "gradient_matrix_physical": _physical_gradient_matrix(gradient_per_km, unit_info),
            "strain_tensor_normalized": strain_norm.tolist(),
            "strain_tensor_per_coord": _matrix_or_none(strain_norm, base),
            "strain_tensor_physical": _physical_gradient_matrix(strain_per_km, unit_info),
            "rotation_matrix_normalized": rotation_norm.tolist(),
            "rotation_matrix_per_coord": _matrix_or_none(rotation_norm, base),
            "rotation_matrix_physical": _physical_gradient_matrix(rotation_per_km, unit_info),
            "rotation_cw_per_coord": None if base is None else float(comp["omega"] / (2.0 * base)),
            "rotation_cw_physical": None if base is None else float(comp["omega"] * unit_info["to_si"] / (2.0 * base * 1000.0)),
        },
        "warnings": warnings,
    }


def interpret_gps_helmert_parameters(
    params: Sequence[float],
    normalization: Mapping[str, Any] | None = None,
    observation_unit: str | None = None,
    default_observation_unit: str = "m",
) -> dict[str, Any]:
    """Interpret CSI GPS ``full`` Helmert-like parameters."""
    vals = _as_array(params)
    norm = dict(normalization or {})
    unit_info = _observation_unit_info(observation_unit, default_observation_unit)
    base = _finite_or_none(norm.get("base"))
    warnings: list[str] = []
    if vals.size == 4:
        tx, ty, theta, scale = vals
        raw = {"tx": tx, "ty": ty, "theta": theta, "scale": scale}
    elif vals.size == 7:
        tx, ty, tz, rx, ry, rz, scale = vals
        theta = rz
        raw = {"tx": tx, "ty": ty, "tz": tz, "rx": rx, "ry": ry, "rz": rz, "scale": scale}
    else:
        warnings.append(f"Unexpected Helmert parameter count {vals.size}; expected 4 or 7.")
        tx = vals[0] if vals.size > 0 else 0.0
        ty = vals[1] if vals.size > 1 else 0.0
        theta = vals[2] if vals.size > 2 else 0.0
        scale = vals[3] if vals.size > 3 else 0.0
        raw = {f"p{i}": float(value) for i, value in enumerate(vals)}
    if base is None:
        warnings.append("Missing or zero Helmert normalization base; physical gradients are not reported.")

    a_norm = np.array([[scale, theta], [-theta, scale]], dtype=float)
    gradient_per_km = None if base is None else np.asarray(a_norm, dtype=float) / base
    return {
        "kind": "full",
        "raw_parameters": {key: float(value) for key, value in raw.items()},
        "normalization": norm,
        "unit_context": unit_info,
        "physical": {
            "translation": {"east": float(tx), "north": float(ty), **({"up": float(vals[2])} if vals.size == 7 else {})},
            "gradient_matrix_normalized": a_norm.tolist(),
            "gradient_matrix_per_coord": _matrix_or_none(a_norm, base),
            "gradient_matrix_physical": _physical_gradient_matrix(gradient_per_km, unit_info),
            "scale_per_coord": None if base is None else float(scale / base),
            "scale_physical": None if base is None else float(scale * unit_info["to_si"] / (base * 1000.0)),
            "rotation_cw_per_coord": None if base is None else float(theta / base),
            "rotation_cw_physical": None if base is None else float(theta * unit_info["to_si"] / (base * 1000.0)),
        },
        "warnings": warnings,
    }


def interpret_internal_strain_parameters(
    params: Sequence[float],
    normalization: Mapping[str, Any] | None = None,
    observation_unit: str | None = None,
    default_observation_unit: str = "m",
) -> dict[str, Any]:
    """Interpret CSI ``internalstrain`` parameters.

    CSI builds this basis from spherical arc-length coordinates around a
    stored lon/lat center.  The coefficients are already gradients in the
    velocity unit per arc-length unit used by CSI.
    """
    vals = _as_array(params)
    warnings: list[str] = []
    if vals.size != 3:
        warnings.append(f"Unexpected internalstrain parameter count {vals.size}; expected 3.")
    sxx = vals[0] if vals.size > 0 else 0.0
    syy = vals[1] if vals.size > 1 else 0.0
    sxy = vals[2] if vals.size > 2 else 0.0
    tensor = np.array([[sxx, 0.5 * sxy], [0.5 * sxy, syy]], dtype=float)
    norm = dict(normalization or {})
    unit_info = _observation_unit_info(observation_unit, default_observation_unit)
    tensor_physical = tensor * float(unit_info["to_si"])
    if "ref" not in norm:
        warnings.append("Missing internal-strain center; center-dependent interpretation should be checked.")
    return {
        "kind": "internalstrain",
        "raw_parameters": {"sxx": float(sxx), "syy": float(syy), "sxy": float(sxy)},
        "normalization": norm,
        "unit_context": unit_info,
        "physical": {
            "strain_tensor_per_arc_length": tensor.tolist(),
            "strain_tensor_physical": tensor_physical.tolist(),
            "strain_tensor_physical_units": "1/year" if unit_info["kind"] == "rate" else "strain",
            "center": norm.get("ref"),
        },
        "warnings": warnings,
    }


def interpret_scalar_poly_parameters(
    ptype: int,
    params: Sequence[float],
    normalization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Interpret scalar polynomial corrections such as InSAR ``1/3/4``."""
    vals = _as_array(params)
    norm = dict(normalization or {})
    warnings: list[str] = []
    names = ["offset"]
    if int(ptype) >= 3:
        names.extend(["x_ramp", "y_ramp"])
    if int(ptype) == 4:
        names.append("xy_cross")
    if vals.size != len(names):
        warnings.append(f"Parameter count {vals.size} does not match scalar poly {ptype}.")
    raw = {name: float(vals[i]) for i, name in enumerate(names) if i < vals.size}
    norm_x = _finite_or_none(norm.get("x"))
    norm_y = _finite_or_none(norm.get("y"))
    gradient = {}
    if "x_ramp" in raw:
        gradient["x"] = None if norm_x is None else raw["x_ramp"] / norm_x
    if "y_ramp" in raw:
        gradient["y"] = None if norm_y is None else raw["y_ramp"] / norm_y
    if "xy_cross" in raw:
        gradient["xy"] = None if norm_x is None or norm_y is None else raw["xy_cross"] / (norm_x * norm_y)
    if any(value is None for value in gradient.values()):
        warnings.append("Missing scalar-poly normalization; some physical gradients are not reported.")
    return {
        "kind": f"scalar_poly_{int(ptype)}",
        "raw_parameters": raw,
        "normalization": norm,
        "physical": {
            "offset": raw.get("offset"),
            "gradient_per_coord": gradient,
        },
        "warnings": warnings,
    }


def extract_normalization(data: Any, transform: Any) -> dict[str, Any]:
    """Return known CSI normalization metadata for one data object/transform."""
    key = str(transform).lower() if isinstance(transform, str) else transform
    if isinstance(transform, (int, np.integer)):
        norm = getattr(data, "TransformNormalizingFactor", {}) or {}
        return {
            "x": norm.get("x"),
            "y": norm.get("y"),
            "ref": norm.get("ref"),
        }
    if key in GPS_STRAIN_TRANSFORMS:
        norm = getattr(data, "TransformNormalizingFactor", {}) or {}
        base = getattr(data, "StrainNormalizingFactor", None)
        if base is None:
            base = norm.get("base")
        return {"base": base, "ref": norm.get("ref")}
    if key == "full":
        return {
            "base": getattr(data, "HelmertNormalizingFactor", None),
            "ref": getattr(data, "HelmertCenter", None),
        }
    if key == "internalstrain":
        return dict(getattr(data, "InternalStrainNormalizingFactor", {}) or {})
    if key == "eulerrotation":
        return {"earth_radius": 6378137.0}
    return {}


def interpret_data_correction_parameters(
    transform: Any,
    params: Sequence[float],
    *,
    data: Any = None,
    normalization: Mapping[str, Any] | None = None,
    euler_output_units: Sequence[str] = ("degrees", "degrees", "degrees_per_myr"),
    observation_unit: str | None = None,
    default_observation_unit: str = "m",
) -> dict[str, Any]:
    """Dispatch to the appropriate interpretation function for one transform."""
    norm = dict(normalization or (extract_normalization(data, transform) if data is not None else {}))
    if transform is None:
        raise ValueError("transform must not be None")
    if isinstance(transform, (int, np.integer)):
        return interpret_scalar_poly_parameters(int(transform), params, norm)
    key = str(transform).lower()
    if key in GPS_STRAIN_TRANSFORMS:
        return interpret_gps_strain_parameters(
            key,
            params,
            norm,
            observation_unit=observation_unit,
            default_observation_unit=default_observation_unit,
        )
    if key == "full":
        return interpret_gps_helmert_parameters(
            params,
            norm,
            observation_unit=observation_unit,
            default_observation_unit=default_observation_unit,
        )
    if key == "eulerrotation":
        result = interpret_euler_vector_parameters(
            params,
            output_units=euler_output_units,
            observation_unit=observation_unit,
            default_observation_unit=default_observation_unit,
        )
        result["normalization"] = norm
        return result
    if key == "internalstrain":
        return interpret_internal_strain_parameters(
            params,
            norm,
            observation_unit=observation_unit,
            default_observation_unit=default_observation_unit,
        )
    raise ValueError(f"Unsupported data-correction transform: {transform!r}")
