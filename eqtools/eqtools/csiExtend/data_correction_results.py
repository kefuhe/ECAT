"""Structured report helpers for estimated data-correction parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _format_matrix(matrix: Any) -> str:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        return str(matrix)
    rows = []
    for row in array:
        rows.append("[" + ", ".join(f"{value:.6g}" for value in row) + "]")
    return "[" + ", ".join(rows) + "]"


def data_correction_entries_to_dataframe(entries: Sequence[Mapping[str, Any]]):
    """Return a compact pandas DataFrame for data-correction entries."""
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
        for key in (
            "rotation_cw_per_coord",
            "rotation_cw_physical",
            "scale_per_coord",
            "scale_physical",
        ):
            if key in physical:
                row[key] = physical.get(key)
        if physical.get("strain_tensor_display") is not None:
            row["strain_tensor_display"] = physical.get("strain_tensor_display")
            row["strain_tensor_display_units"] = physical.get("strain_tensor_display_units")
        if physical.get("strain_tensor_physical") is not None:
            row["strain_tensor_physical"] = physical.get("strain_tensor_physical")
            row["strain_tensor_physical_units"] = physical.get("strain_tensor_physical_units")
        rows.append(row)
    return pd.DataFrame(rows)


def format_data_correction_report(
    entries: Sequence[Mapping[str, Any]],
    *,
    unit_info: Mapping[str, Any] | None = None,
) -> str:
    """Format a human-readable data-correction parameter report."""
    lines = ["Data-correction parameter report", f"  entries: {len(entries)}"]
    if unit_info and unit_info.get("observation"):
        assumed = " (assumed)" if unit_info.get("assumed") else ""
        lines.append(f"  units: observation={unit_info['observation']}{assumed}")
    for entry in entries:
        header = f"  - {entry.get('source')} / {entry.get('dataset')}: {entry.get('transform')}"
        if entry.get("transform_group") is not None:
            header += f" (from {entry.get('transform_group')})"
        lines.append(header)
        physical = entry.get("physical", {}) or {}
        raw = entry.get("raw_parameters", {}) or {}
        if raw:
            if isinstance(raw, Mapping):
                raw_text = ", ".join(
                    f"{key}={value:.6g}"
                    for key, value in raw.items()
                    if isinstance(value, (int, float, np.floating))
                )
            else:
                raw_text = ", ".join(
                    f"{value:.6g}"
                    for value in raw
                    if isinstance(value, (int, float, np.floating))
                )
            if raw_text:
                lines.append(f"      raw: {raw_text}")
        pole = physical.get("euler_pole")
        if pole:
            lines.append(f"      euler pole: {pole.get('value', [])} {pole.get('units', [])}")
        strain_display = physical.get("strain_tensor_display")
        if strain_display is not None:
            units = physical.get("strain_tensor_display_units", "")
            lines.append(f"      strain tensor: {_format_matrix(strain_display)} {units}")
        for key in (
            "rotation_cw_per_coord",
            "rotation_cw_physical",
            "scale_per_coord",
            "scale_physical",
        ):
            value = physical.get(key)
            if value is not None:
                lines.append(f"      {key}: {value:.6g}")
        for warning in entry.get("warnings", []) or []:
            lines.append(f"      warning: {warning}")
    return "\n".join(lines) + "\n"


def write_data_correction_report_files(
    entries: Sequence[Mapping[str, Any]],
    outdir,
    *,
    basename: str = "data_correction_parameters",
    formats: Sequence[str] = ("txt", "tsv"),
    unit_info: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write data-correction report files and return their paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    requested = {str(fmt).lower().lstrip(".") for fmt in formats}
    unsupported = requested - {"txt", "tsv"}
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported data-correction report format(s): {names}")
    written: dict[str, Path] = {}

    if "txt" in requested:
        path = outdir / f"{basename}.txt"
        path.write_text(
            format_data_correction_report(entries, unit_info=unit_info),
            encoding="utf-8",
        )
        written["txt"] = path

    if "tsv" in requested:
        path = outdir / f"{basename}.tsv"
        dataframe = data_correction_entries_to_dataframe(entries)
        dataframe.to_csv(path, sep="\t", index=False)
        written["tsv"] = path

    return written
