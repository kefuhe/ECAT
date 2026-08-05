"""Read-only helpers for interseismic loading and motion-sense overrides."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from .interseismic_parameter_model import get_fault_by_name, get_fault_loading_params
from .patch_indices import normalize_patch_indices, select_patch_indices


MOTION_SENSE_TABLE_COLUMNS = [
    "fault",
    "patch_index",
    "motion_sense",
    "motion_sign",
    "source",
    "override_name",
    "override_index",
    "center_x",
    "center_y",
    "center_depth",
    "center_lon",
    "center_lat",
]


def get_loading_overrides(params: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return normalized local loading overrides for one fault-loading config."""
    overrides = params.get("loading_overrides")
    if overrides is None:
        overrides = params.get("loading_regions", [])
    return list(overrides or [])


def get_motion_sense_overrides(params: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return local motion-sense overrides for one fault-loading config."""
    return list(params.get("motion_sense_overrides", []) or [])


def normalize_motion_sense(motion_sense: Any) -> str:
    """Normalize public right-/left-lateral aliases."""
    key = str(motion_sense).lower()
    if key in {"dextral", "right_lateral", "right"}:
        return "dextral"
    if key in {"sinistral", "left_lateral", "left"}:
        return "sinistral"
    raise ValueError(f"Invalid motion_sense: {motion_sense}")


def motion_sense_to_sign(motion_sense: Any) -> float:
    """Return ``+1`` for dextral and ``-1`` for sinistral cap rows."""
    return 1.0 if normalize_motion_sense(motion_sense) == "dextral" else -1.0


def resolve_motion_sense_rows_for_patches(
    fault: Any,
    fault_name: str,
    params: Mapping[str, Any],
    patch_indices: Iterable[int],
) -> list[Dict[str, Any]]:
    """Resolve the effective motion-sense expectation for selected patches.

    Priority is deliberately simple and mirrors the configuration hierarchy:
    fault default, then ``loading_overrides`` motion_sense, then
    ``motion_sense_overrides``.  The last layer is useful for search tests where
    a candidate left-/right-lateral transition should not redefine block pairs.
    """
    selected = normalize_patch_indices(
        fault,
        patch_indices,
        allow_none_all=True,
        unique=True,
        name=f"motion-sense patches for fault '{fault_name}'",
    )
    default_sense = normalize_motion_sense(params.get("motion_sense", "dextral"))
    rows_by_patch: dict[int, Dict[str, Any]] = {
        int(idx): {
            "patch_index": int(idx),
            "motion_sense": default_sense,
            "motion_sign": motion_sense_to_sign(default_sense),
            "source": "default",
            "override_name": "",
            "override_index": None,
        }
        for idx in selected.tolist()
    }

    _apply_motion_sense_layer(
        fault,
        fault_name,
        rows_by_patch,
        get_loading_overrides(params),
        layer_name="loading_overrides",
        source="loading_override",
        require_motion_sense=False,
        inherited_motion_sense=default_sense,
    )
    _apply_motion_sense_layer(
        fault,
        fault_name,
        rows_by_patch,
        get_motion_sense_overrides(params),
        layer_name="motion_sense_overrides",
        source="motion_sense_override",
        require_motion_sense=True,
        inherited_motion_sense=None,
    )

    return [rows_by_patch[int(idx)] for idx in selected.tolist()]


def resolve_interseismic_motion_sense(
    inversion: Any,
    fault_name: str,
    patch_indices: Optional[Iterable[int]] = None,
    *,
    include_centers: bool = True,
) -> Dict[str, Any]:
    """Resolve effective motion-sense expectations for plotting or export."""
    fault = get_fault_by_name(inversion, fault_name)
    params = get_fault_loading_params(inversion, fault_name)
    selected = normalize_patch_indices(
        fault,
        patch_indices,
        allow_none_all=True,
        unique=True,
        name=f"interseismic motion-sense patches for fault '{fault_name}'",
    )
    rows = resolve_motion_sense_rows_for_patches(fault, fault_name, params, selected)
    centers_xy, centers_ll = _get_centers(fault, include_centers=include_centers)

    n_patches = len(fault.patch)
    motion_sign = np.full(n_patches, np.nan, dtype=float)
    source_id = np.full(n_patches, np.nan, dtype=float)
    selected_mask = np.zeros(n_patches, dtype=bool)
    records = []

    for row in rows:
        patch_idx = int(row["patch_index"])
        motion_sign[patch_idx] = float(row["motion_sign"])
        source_id[patch_idx] = _motion_source_id(row)
        selected_mask[patch_idx] = True
        records.append(_motion_record_from_row(fault_name, row, centers_xy, centers_ll))

    return {
        "fault_name": fault_name,
        "records": records,
        "fields": {
            "motion_sign": motion_sign,
            "source_id": source_id,
            "selected": selected_mask,
        },
        "patch_indices": selected,
        "metadata": {
            "source": "fault_loading",
            "loading_override_count": len(get_loading_overrides(params)),
            "motion_sense_override_count": len(get_motion_sense_overrides(params)),
            "record_count": len(records),
        },
    }


def summarize_interseismic_motion_sense_diagnostics(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return compact counts for resolved motion-sense assignments."""
    records = list(result.get("records", []))
    by_source: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (str(record.get("source", "")), str(record.get("override_name", "")))
        by_source.setdefault(key, []).append(record)

    sources = []
    for (source, name), source_records in sorted(by_source.items()):
        signs = np.asarray([rec["motion_sign"] for rec in source_records], dtype=float)
        sources.append({
            "source": source,
            "override_name": name,
            "patch_count": len(source_records),
            "dextral_count": int(np.sum(signs > 0)),
            "sinistral_count": int(np.sum(signs < 0)),
            "patch_indices": [int(rec["patch_index"]) for rec in source_records],
        })

    signs = np.asarray([rec["motion_sign"] for rec in records], dtype=float)
    return {
        "fault_name": result.get("fault_name"),
        "patch_count": len(records),
        "dextral_count": int(np.sum(signs > 0)),
        "sinistral_count": int(np.sum(signs < 0)),
        "sources": sources,
    }


def export_interseismic_motion_sense_table(
    result: Mapping[str, Any],
    filename: str | Path,
    *,
    include_header: bool = True,
) -> Path:
    """Write resolved motion-sense assignments as CSV."""
    path = Path(filename)
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=MOTION_SENSE_TABLE_COLUMNS, extrasaction="ignore")
        if include_header:
            writer.writeheader()
        for record in result.get("records", []):
            writer.writerow({key: record.get(key, "") for key in MOTION_SENSE_TABLE_COLUMNS})
    return path


def _apply_motion_sense_layer(
    fault: Any,
    fault_name: str,
    rows_by_patch: dict[int, Dict[str, Any]],
    overrides: list[Mapping[str, Any]],
    *,
    layer_name: str,
    source: str,
    require_motion_sense: bool,
    inherited_motion_sense: str | None,
) -> None:
    seen: dict[int, str] = {}
    for index, override in enumerate(overrides):
        name = str(override.get("name", f"override_{index}"))
        if "selector" not in override:
            raise ValueError(f"fault_loading.faults.{fault_name}.{layer_name}[{index}] is missing required 'selector'")
        if require_motion_sense and "motion_sense" not in override:
            raise ValueError(
                f"fault_loading.faults.{fault_name}.{layer_name}[{index}] "
                "is missing required 'motion_sense'"
            )

        selected = select_patch_indices(
            fault,
            override.get("selector"),
            allow_none_all=False,
            unique=True,
            name=f"{layer_name} '{name}' selector for fault '{fault_name}'",
        )
        motion_sense = normalize_motion_sense(override.get("motion_sense", inherited_motion_sense))
        for patch_idx in np.asarray(selected, dtype=int).tolist():
            patch_idx = int(patch_idx)
            if patch_idx not in rows_by_patch:
                continue
            if patch_idx in seen:
                raise ValueError(
                    f"fault_loading.faults.{fault_name}.{layer_name} overlap on patch "
                    f"{patch_idx}; override '{name}' overlaps with '{seen[patch_idx]}'."
                )
            rows_by_patch[patch_idx] = {
                "patch_index": patch_idx,
                "motion_sense": motion_sense,
                "motion_sign": motion_sense_to_sign(motion_sense),
                "source": source,
                "override_name": name,
                "override_index": index,
            }
            seen[patch_idx] = name


def _motion_source_id(row: Mapping[str, Any]) -> float:
    source = row.get("source")
    if source == "default":
        return 0.0
    index = row.get("override_index")
    if index is None:
        index = 0
    if source == "loading_override":
        return float(int(index) + 1)
    if source == "motion_sense_override":
        return -float(int(index) + 1)
    return np.nan


def _motion_record_from_row(
    fault_name: str,
    row: Mapping[str, Any],
    centers_xy: np.ndarray | None,
    centers_ll: np.ndarray | None,
) -> Dict[str, Any]:
    patch_idx = int(row["patch_index"])
    record = {
        "fault": fault_name,
        "patch_index": patch_idx,
        "motion_sense": row.get("motion_sense", ""),
        "motion_sign": float(row.get("motion_sign", np.nan)),
        "source": row.get("source", ""),
        "override_name": row.get("override_name", ""),
        "override_index": "" if row.get("override_index") is None else int(row.get("override_index")),
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


def _get_centers(fault: Any, *, include_centers: bool) -> tuple[np.ndarray | None, np.ndarray | None]:
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
