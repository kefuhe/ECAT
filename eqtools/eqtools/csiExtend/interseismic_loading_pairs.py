"""Read-only helpers for interseismic loading-pair inspection.

The functions in this module mirror the existing ``fault_loading`` and
``loading_overrides`` definitions.  They do not build constraints, modify solver
matrices, or overwrite ``fault.slip``.  Their purpose is to make manual
fault/segment block-pair assignments easy to audit before interpreting
coupling fields.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from .interseismic_fields import summarize_values
from .interseismic_parameter_model import (
    calculate_loading_from_terms,
    get_fault_by_name,
    get_fault_loading_params,
)
from .interseismic_overrides import get_loading_overrides
from .patch_indices import normalize_patch_indices, select_patch_indices


PAIR_TABLE_COLUMNS = [
    "fault",
    "patch_index",
    "pair_id",
    "source",
    "region_name",
    "block_positive",
    "block_negative",
    "block_type_positive",
    "block_type_negative",
    "reference_strike",
    "motion_sense",
    "loading",
    "center_x",
    "center_y",
    "center_depth",
    "center_lon",
    "center_lat",
]


def resolve_interseismic_loading_pairs(
    inversion: Any,
    fault_name: str,
    patch_indices: Optional[Iterable[int]] = None,
    *,
    solution: Optional[Sequence[float]] = None,
    include_centers: bool = True,
) -> Dict[str, Any]:
    """Resolve the effective fault-loading block pair for selected patches.

    Parameters
    ----------
    inversion : object
        Inversion object with parsed ``interseismic_config`` and fault objects.
    fault_name : str
        Name of the target fault.
    patch_indices : iterable of int, optional
        Patches to inspect.  ``None`` inspects all patches.
    solution : array-like, optional
        Linear solution vector used to evaluate loading values.  Defaults to
        ``inversion.mpost`` when available.  If no solution exists, pair
        assignment still succeeds and loading values are ``nan``.
    include_centers : bool, default True
        Include local and lon/lat patch centers in exported records when the
        fault object provides ``getcenters()`` and ``xy2ll()``.

    Returns
    -------
    dict
        A read-only result with per-patch ``records``, numeric ``fields`` for
        plotting, compact ``pairs`` metadata, and ``warnings``.
    """
    fault = get_fault_by_name(inversion, fault_name)
    params = get_fault_loading_params(inversion, fault_name)
    selected = normalize_patch_indices(
        fault,
        patch_indices,
        allow_none_all=True,
        unique=True,
        name=f"interseismic loading pair patches for fault '{fault_name}'",
    )
    n_patches = len(fault.patch)

    rows = _resolve_pair_rows_for_patches(fault, fault_name, params, selected)
    pair_catalog = _build_pair_catalog(rows)

    loading = np.full(selected.size, np.nan, dtype=float)
    warnings: list[str] = []
    try:
        loading = np.asarray(
            calculate_loading_from_terms(
                inversion,
                fault_name,
                selected.tolist(),
                solution=solution,
            ),
            dtype=float,
        )
    except ValueError as exc:  # pragma: no cover - exercised by callers without a solution
        if "No current linear solution" not in str(exc):
            raise
        warnings.append(f"Loading values were not evaluated: {exc}")

    centers_xy, centers_ll = _get_centers(fault, include_centers=include_centers)
    records = []
    pair_id_field = np.full(n_patches, np.nan, dtype=float)
    loading_field = np.full(n_patches, np.nan, dtype=float)
    selected_mask = np.zeros(n_patches, dtype=bool)

    for row, value in zip(rows, loading):
        patch_idx = int(row["patch_index"])
        pair_id = int(row["pair_id"])
        pair_id_field[patch_idx] = float(pair_id)
        loading_field[patch_idx] = float(value)
        selected_mask[patch_idx] = True
        record = _record_from_row(fault_name, row, value, centers_xy, centers_ll)
        records.append(record)

    return {
        "fault_name": fault_name,
        "records": records,
        "pairs": pair_catalog,
        "fields": {
            "pair_id": pair_id_field,
            "loading": loading_field,
            "selected": selected_mask,
        },
        "patch_indices": selected,
        "warnings": warnings,
        "metadata": {
            "source": "fault_loading",
            "has_loading_overrides": bool(get_loading_overrides(params)),
            "has_loading_regions": bool(get_loading_overrides(params)),
            "record_count": len(records),
            "pair_count": len(pair_catalog),
            "configured_pair_count": 1 + len(get_loading_overrides(params)),
        },
    }


def summarize_interseismic_pair_diagnostics(
    result: Mapping[str, Any],
    *,
    zero_tolerance: float = 1.0e-12,
) -> Dict[str, Any]:
    """Return compact diagnostics for a resolved loading-pair result."""
    records = list(result.get("records", []))
    by_pair: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        by_pair.setdefault(int(record["pair_id"]), []).append(record)

    pair_reports = []
    warnings = list(result.get("warnings", []))
    for pair_id in sorted(by_pair):
        pair_records = by_pair[pair_id]
        loading = np.asarray([rec.get("loading", np.nan) for rec in pair_records], dtype=float)
        finite = loading[np.isfinite(loading)]
        pair = dict(result.get("pairs", {}).get(pair_id, {}))
        report = {
            "pair_id": pair_id,
            "source": pair.get("source", pair_records[0].get("source")),
            "region_name": pair.get("region_name", pair_records[0].get("region_name")),
            "blocks": pair.get("block_names", [
                pair_records[0].get("block_positive"),
                pair_records[0].get("block_negative"),
            ]),
            "block_types": pair.get("block_types", [
                pair_records[0].get("block_type_positive"),
                pair_records[0].get("block_type_negative"),
            ]),
            "reference_strike": pair.get("reference_strike", pair_records[0].get("reference_strike")),
            "motion_sense": pair.get("motion_sense", pair_records[0].get("motion_sense")),
            "patch_count": len(pair_records),
            "loading": summarize_values(finite) if finite.size else summarize_values([]),
            "positive_loading_count": int(np.sum(finite > zero_tolerance)),
            "negative_loading_count": int(np.sum(finite < -zero_tolerance)),
            "near_zero_loading_count": int(np.sum(np.abs(finite) <= zero_tolerance)),
            "missing_loading_count": int(loading.size - finite.size),
            "patch_indices": [int(rec["patch_index"]) for rec in pair_records],
        }
        if report["missing_loading_count"]:
            warnings.append(
                f"{result.get('fault_name', 'fault')} pair {pair_id} has "
                f"{report['missing_loading_count']} patch(es) without evaluated loading."
            )
        pair_reports.append(report)

    return {
        "fault_name": result.get("fault_name"),
        "patch_count": len(records),
        "pair_count": len(pair_reports),
        "pairs": pair_reports,
        "warnings": warnings,
    }


def export_interseismic_loading_pair_table(
    result: Mapping[str, Any],
    filename: str | Path,
    *,
    include_header: bool = True,
) -> Path:
    """Write a resolved loading-pair table as CSV.

    Parameters
    ----------
    result : mapping
        Output from ``resolve_interseismic_loading_pairs()``.
    filename : str or path-like
        Output CSV path.
    include_header : bool, default True
        Write the standard header row.

    Returns
    -------
    pathlib.Path
        Written path.
    """
    path = Path(filename)
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=PAIR_TABLE_COLUMNS, extrasaction="ignore")
        if include_header:
            writer.writeheader()
        for record in result.get("records", []):
            writer.writerow({key: _csv_value(record.get(key)) for key in PAIR_TABLE_COLUMNS})
    return path


def _resolve_pair_rows_for_patches(
    fault: Any,
    fault_name: str,
    default_params: Mapping[str, Any],
    patch_indices: np.ndarray,
) -> list[Dict[str, Any]]:
    selected_list = [int(idx) for idx in patch_indices.tolist()]
    rows_by_patch = {
        int(idx): {
            "patch_index": int(idx),
            "params": default_params,
            "source": "default",
            "region_name": "",
            "region_index": None,
            "pair_id": 0,
        }
        for idx in selected_list
    }
    assigned_by_region: dict[int, str] = {}

    for region_index, region in enumerate(get_loading_overrides(default_params)):
        region_name = str(region.get("name", f"region_{region_index}"))
        region_indices = select_patch_indices(
            fault,
            region.get("selector"),
            allow_none_all=False,
            unique=True,
            name=f"loading override '{region_name}' selector for fault '{fault_name}'",
        )
        selected_region = [int(idx) for idx in region_indices.tolist() if int(idx) in rows_by_patch]
        for patch_idx in selected_region:
            if patch_idx in assigned_by_region:
                raise ValueError(
                    f"fault_loading.faults.{fault_name}.loading_overrides overlap on patch "
                    f"{patch_idx}; region '{region_name}' overlaps with "
                    f"'{assigned_by_region[patch_idx]}'."
                )
            rows_by_patch[patch_idx] = {
                "patch_index": patch_idx,
                "params": region,
                "source": "loading_override",
                "region_name": region_name,
                "region_index": region_index,
                "pair_id": region_index + 1,
            }
            assigned_by_region[patch_idx] = region_name

    rows = []
    for patch_idx in selected_list:
        row = rows_by_patch[patch_idx]
        params = row["params"]
        row.update(_pair_descriptor(params))
        rows.append(row)
    return rows


def _build_pair_catalog(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    catalog: dict[int, Dict[str, Any]] = {}
    signatures: dict[int, tuple[Any, ...]] = {}
    for row in rows:
        pair_id = int(row["pair_id"])
        signature = _pair_signature(row)
        if pair_id not in catalog:
            signatures[pair_id] = signature
            catalog[pair_id] = {
                "pair_id": pair_id,
                "source": row.get("source"),
                "region_name": row.get("region_name"),
                "region_index": row.get("region_index"),
                "block_types": list(row.get("block_types", [])),
                "blocks": list(row.get("blocks", [])),
                "block_names": list(row.get("block_names", [])),
                "pair_signature": signature,
                "reference_strike": float(row.get("reference_strike", 0.0)),
                "motion_sense": row.get("motion_sense"),
                "patch_count": 0,
            }
        elif signatures[pair_id] != signature:
            raise ValueError(
                f"Internal loading pair id {pair_id} maps to more than one "
                "block definition. Check loading_overrides parsing."
            )
        catalog[pair_id]["patch_count"] += 1
    return catalog


def _pair_descriptor(params: Mapping[str, Any]) -> Dict[str, Any]:
    block_types = [str(value) for value in params.get("block_types", [])]
    blocks = list(params.get("blocks_original", params.get("blocks", params.get("blocks_standard", []))))
    block_names = list(params.get("block_names", blocks))
    return {
        "block_types": block_types,
        "blocks": blocks,
        "block_names": block_names,
        "reference_strike": float(params.get("reference_strike", 0.0)),
        "motion_sense": str(params.get("motion_sense", "")),
    }


def _record_from_row(
    fault_name: str,
    row: Mapping[str, Any],
    loading: float,
    centers_xy: Optional[np.ndarray],
    centers_ll: Optional[np.ndarray],
) -> Dict[str, Any]:
    patch_idx = int(row["patch_index"])
    block_names = list(row.get("block_names", ["", ""]))
    block_types = list(row.get("block_types", ["", ""]))
    record = {
        "fault": fault_name,
        "patch_index": patch_idx,
        "pair_id": int(row["pair_id"]),
        "source": row.get("source", ""),
        "region_name": row.get("region_name", ""),
        "block_positive": block_names[0] if len(block_names) > 0 else "",
        "block_negative": block_names[1] if len(block_names) > 1 else "",
        "block_type_positive": block_types[0] if len(block_types) > 0 else "",
        "block_type_negative": block_types[1] if len(block_types) > 1 else "",
        "reference_strike": float(row.get("reference_strike", np.nan)),
        "motion_sense": row.get("motion_sense", ""),
        "loading": float(loading),
    }
    if centers_xy is not None:
        record.update({
            "center_x": float(centers_xy[patch_idx, 0]),
            "center_y": float(centers_xy[patch_idx, 1]),
            "center_depth": float(centers_xy[patch_idx, 2]),
        })
    if centers_ll is not None:
        record.update({
            "center_lon": float(centers_ll[patch_idx, 0]),
            "center_lat": float(centers_ll[patch_idx, 1]),
        })
    return record


def _get_centers(fault: Any, *, include_centers: bool) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not include_centers or not hasattr(fault, "getcenters"):
        return None, None
    centers = np.asarray(fault.getcenters(), dtype=float)
    if centers.ndim != 2 or centers.shape[1] < 3:
        return None, None
    centers_xy = centers[:, :3].copy()
    centers_ll = None
    if hasattr(fault, "xy2ll"):
        lon, lat = fault.xy2ll(centers_xy[:, 0], centers_xy[:, 1])
        centers_ll = np.column_stack((np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)))
    return centers_xy, centers_ll


def _hashable_block(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, list):
        return tuple(_hashable_block(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable_block(val)) for key, val in value.items()))
    return value


def _pair_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source"),
        row.get("region_name"),
        tuple(row.get("block_types", [])),
        tuple(_hashable_block(value) for value in row.get("blocks", [])),
        float(row.get("reference_strike", 0.0)),
        row.get("motion_sense"),
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(value.tolist() if isinstance(value, np.ndarray) else value, ensure_ascii=False)
    return value
