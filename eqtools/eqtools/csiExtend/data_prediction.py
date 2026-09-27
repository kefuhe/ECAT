"""Shared prediction-composition policy for configured geodetic data.

Linear inversion configuration is authoritative: by the time an inversion
object is constructed, ``config.geodata['verticals']`` and
``config.geodata['polys']`` must already be aligned with
``config.geodata['data']``. Helpers here consume that normalized state; they
do not reinterpret raw YAML or silently repair length mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Mapping, Sequence

import numpy as np

from .data_vector_layout import (
    data_observation_vector,
    data_synthetic_vector,
    prepare_data_synthetic_fields,
)


_MISSING = object()
_VALID_DATA_POLY_REQUESTS = (None, "config", "include")


@dataclass(frozen=True)
class GeodataPredictionSpec:
    """Normalized prediction inputs for one configured dataset."""

    data: Any
    vertical: bool
    configured_poly: Any


@dataclass(frozen=True)
class FitPredictionBlock:
    """One configured dataset's observation-space prediction."""

    spec: GeodataPredictionSpec
    observed: np.ndarray
    predicted: np.ndarray


def require_source_prediction_protocol() -> None:
    """Reject an outdated CSI before source-aware diagnostic prediction."""
    if find_spec("csi._slip_direction") is None:
        raise RuntimeError(
            "source-aware prediction requires the matching CSI update; "
            "install the current csi_cutde_mpiparallel package before "
            "running diagnostic fit statistics or figure products"
        )


def build_linear_prediction_blocks(
    specs: Sequence[GeodataPredictionSpec],
    data_ranges: Mapping[str, tuple[int, int]],
    observed,
    predicted,
) -> list[FitPredictionBlock]:
    """Split one assembled linear prediction using its authoritative rows."""
    specs = list(specs)
    observed = np.asarray(observed, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if observed.shape != predicted.shape:
        raise ValueError(
            "assembled observed and predicted vectors must have the same shape"
        )

    names = [str(getattr(spec.data, "name", "")) for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("configured geodetic dataset names must be unique")
    if names != list(data_ranges):
        raise ValueError(
            "configured geodetic datasets do not match the assembled data-row order"
        )

    blocks = []
    cursor = 0
    for spec, name in zip(specs, names):
        start, end = (int(value) for value in data_ranges[name])
        if start != cursor or end <= start or end > observed.size:
            raise ValueError(
                "data_ranges must be contiguous, non-empty, and cover the "
                "assembled observation vector in configured dataset order"
            )
        expected_size = data_observation_vector(
            spec.data,
            vertical=spec.vertical,
        ).size
        if end - start != expected_size:
            raise ValueError(
                f"{name} occupies {end - start} assembled rows, but its active "
                f"observation layout has {expected_size} rows"
            )
        blocks.append(
            FitPredictionBlock(
                spec=spec,
                observed=observed[start:end].copy(),
                predicted=predicted[start:end].copy(),
            )
        )
        cursor = end
    if cursor != observed.size:
        raise ValueError(
            f"data_ranges cover {cursor} rows but the assembled vector has "
            f"{observed.size}"
        )
    return blocks


def publish_prediction_blocks(blocks: Sequence[FitPredictionBlock]) -> None:
    """Publish prediction blocks after validating every CSI field layout.

    All fields are prepared before assignment, so ordinary shape/layout errors
    cannot leave a partially published result. This is a prevalidated batch
    update, not a general transaction for arbitrary custom attribute setters.
    """
    prepared = []
    for block in blocks:
        fields = prepare_data_synthetic_fields(
            block.spec.data,
            block.predicted,
            vertical=block.spec.vertical,
        )
        prepared.append((block.spec.data, fields))
    for data, fields in prepared:
        for field, value in fields.items():
            setattr(data, field, value)


def build_diagnostic_prediction_blocks(
    specs: Sequence[GeodataPredictionSpec],
    sources: Sequence[Any],
    *,
    requested_poly: str | None = "config",
    rebuild_synth: bool = True,
) -> list[FitPredictionBlock]:
    """Build source-selected diagnostic predictions through CSI forward APIs.

    This function owns the adapter boundary between ECAT's high-level
    correction policy and CSI's ``buildsynth`` call. Formal BLSE/VCE results do
    not use this path: they are split directly from the assembled ``G @ m``
    vector and therefore never reinterpret correction configuration.

    When ``rebuild_synth`` is false, the function reads the already active CSI
    synthetic fields without changing them or requiring the unused forward
    protocol. The caller remains responsible for activating the intended
    model first.
    """
    specs = list(specs)
    sources = list(sources)
    if rebuild_synth:
        rebuild_diagnostic_synthetics(
            specs,
            sources,
            requested_poly=requested_poly,
        )
    return [
        FitPredictionBlock(
            spec=spec,
            observed=data_observation_vector(
                spec.data,
                vertical=spec.vertical,
            ),
            predicted=data_synthetic_vector(
                spec.data,
                vertical=spec.vertical,
            ),
        )
        for spec in specs
    ]


def rebuild_diagnostic_synthetics(
    specs: Sequence[GeodataPredictionSpec],
    sources: Sequence[Any],
    *,
    requested_poly: str | None = "config",
) -> None:
    """Publish source-selected CSI synthetics using one adapter contract."""
    require_source_prediction_protocol()
    sources = list(sources)
    for spec in specs:
        resolved_poly = resolve_data_poly(
            spec.configured_poly,
            requested=requested_poly,
            data_type=getattr(spec.data, "dtype", None),
        )
        spec.data.buildsynth(
            sources,
            direction="source",
            poly=resolved_poly,
            vertical=spec.vertical,
        )


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


def resolve_data_poly(
    configured_poly: Any,
    requested: str | None = "config",
    *,
    data_type: str | None = None,
):
    """Resolve ECAT prediction policy to the value expected by CSI.

    ``"config"`` follows each dataset's parsed correction configuration,
    ``"include"`` requests the solved total prediction, and ``None`` is the
    explicit source/slip-only diagnostic mode. Most CSI data classes consume
    the policy token ``"include"`` directly. ``crossfaultoffset`` instead
    needs the concrete estimator specification already accepted during matrix
    assembly, including a list-valued specification. This adapter passes that
    parsed value through rather than imposing a second, narrower schema in the
    result layer.
    """
    if requested == "config":
        resolved = None if configured_poly is None else "include"
    elif requested in (None, "include"):
        resolved = requested
    else:
        allowed = ", ".join(repr(value) for value in _VALID_DATA_POLY_REQUESTS)
        raise ValueError(f"data_poly must be one of {allowed}; got {requested!r}")

    if str(data_type or "").lower() != "crossfaultoffset" or resolved != "include":
        return resolved
    return configured_poly
