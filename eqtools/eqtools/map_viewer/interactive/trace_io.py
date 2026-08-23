"""Adapt shared trace I/O to interactive-editor models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...csiExtend.trace_io import read_trace_segments, write_trace
from .models import PathDraft, ReferencePath


def _path_id(stem, index, count):
    return stem if count == 1 else f"{stem}.{index + 1}"


def read_reference_paths(path):
    """Return immutable reference paths from TXT/GMT or GeoJSON line data."""

    path = Path(path).resolve()
    segments = read_trace_segments(path)
    result = []
    for index, segment in enumerate(segments):
        result.append(
            ReferencePath(
                id=_path_id(path.stem, index, len(segments)),
                name=segment.name or _path_id(path.stem, index, len(segments)),
                coordinates=segment.coordinates,
                source_file=path,
            )
        )
    return tuple(result)


def reference_path_from_arrays(
    path_id,
    name,
    longitude,
    latitude,
    *,
    source_file=None,
    metadata=None,
):
    """Build one immutable reference from aligned in-memory lon/lat arrays."""

    longitude = np.asarray(longitude, dtype=float).reshape(-1)
    latitude = np.asarray(latitude, dtype=float).reshape(-1)
    if longitude.shape != latitude.shape:
        raise ValueError("Reference longitude and latitude must be aligned.")
    return ReferencePath(
        id=path_id,
        name=name,
        coordinates=np.column_stack((longitude, latitude)),
        source_file=source_file,
        metadata=metadata or {},
    )


def save_trace(path, draft, *, overwrite=False):
    """Save one validated working trace without silently replacing a source."""

    if not isinstance(draft, PathDraft):
        raise TypeError("save_trace expects a PathDraft.")
    if len(draft.coordinates) < 2:
        raise ValueError("A saved trace requires at least two vertices.")
    return write_trace(
        path,
        draft.coordinates,
        name=draft.name,
        role="adjusted_trace",
        properties={"source_layer_id": draft.source_layer_id},
        overwrite=overwrite,
        text_title="ECAT adjusted trace",
    )


__all__ = [
    "read_reference_paths",
    "reference_path_from_arrays",
    "save_trace",
]
