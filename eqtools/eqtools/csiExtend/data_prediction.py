"""Shared prediction-composition policy for configured geodetic data.

Linear inversion configuration is authoritative: by the time an inversion
object is constructed, ``config.geodata['verticals']`` and
``config.geodata['polys']`` must already be aligned with
``config.geodata['data']``. Helpers here consume that normalized state; they
do not reinterpret raw YAML or silently repair length mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_MISSING = object()
_VALID_DATA_POLY_REQUESTS = (None, "config", "include")


@dataclass(frozen=True)
class GeodataPredictionSpec:
    """Normalized prediction inputs for one configured dataset."""

    data: Any
    vertical: bool
    configured_poly: Any


def _aligned_values(raw, *, count: int, field: str, missing_default):
    """Return an aligned list without re-normalizing parsed configuration."""
    if raw is _MISSING:
        return [missing_default] * count
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"{field} must be a parsed list aligned with config.geodata['data']; "
            f"got {type(raw).__name__}. Construct the inversion through the "
            "standard configuration parser so scalar values are normalized first."
        )
    if len(raw) != count:
        raise ValueError(
            f"{field} has {len(raw)} item(s), but config.geodata['data'] has "
            f"{count}; prediction composition cannot be aligned safely"
        )
    return list(raw)


def get_geodata_prediction_specs(inversion: Any) -> list[GeodataPredictionSpec]:
    """Return data, vertical flags and corrections from normalized config.

    ``config.geodata`` is the preferred and authoritative source. The
    attribute fallback supports legacy analysis objects that have no config,
    while applying the same strict alignment contract.
    """
    config = getattr(inversion, "config", None)
    geodata = getattr(config, "geodata", None)
    if isinstance(geodata, Mapping):
        data = list(geodata.get("data", []) or [])
        verticals_raw = geodata.get("verticals", _MISSING)
        polys_raw = geodata.get("polys", _MISSING)
        prefix = "config.geodata"
    else:
        legacy_data = getattr(inversion, "datas", _MISSING)
        if legacy_data is _MISSING or legacy_data is None:
            legacy_data = getattr(inversion, "geodata", [])
        data = list(legacy_data or [])
        verticals_raw = getattr(inversion, "verticals", _MISSING)
        polys_raw = getattr(inversion, "polys", _MISSING)
        prefix = "inversion"

    verticals = _aligned_values(
        verticals_raw,
        count=len(data),
        field=f"{prefix}.verticals",
        missing_default=True,
    )
    polys = _aligned_values(
        polys_raw,
        count=len(data),
        field=f"{prefix}.polys",
        missing_default=None,
    )
    return [
        GeodataPredictionSpec(item, bool(vertical), poly)
        for item, vertical, poly in zip(data, verticals, polys)
    ]


def resolve_data_poly(configured_poly: Any, requested: str | None = "config"):
    """Resolve ECAT prediction policy to the value expected by CSI.

    ``"config"`` follows each dataset's parsed correction configuration,
    ``"include"`` requests the solved total prediction, and ``None`` is the
    explicit source/slip-only diagnostic mode.
    """
    if requested == "config":
        return None if configured_poly is None else "include"
    if requested in (None, "include"):
        return requested
    allowed = ", ".join(repr(value) for value in _VALID_DATA_POLY_REQUESTS)
    raise ValueError(f"data_poly must be one of {allowed}; got {requested!r}")
