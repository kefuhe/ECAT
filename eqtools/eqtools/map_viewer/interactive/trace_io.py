"""Read reference polylines and safely save one adjusted ECAT trace."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from .models import PathDraft, ReferencePath


_TEXT_SUFFIXES = {".txt", ".dat", ".trace", ".gmt"}
_GEOJSON_SUFFIXES = {".json", ".geojson"}


def _path_id(stem, index, count):
    return stem if count == 1 else f"{stem}.{index + 1}"


def _text_segments(path):
    segments = []
    current = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if current:
                segments.append(np.asarray(current, dtype=float))
                current = []
            continue
        fields = [item for item in re.split(r"[\s,]+", line) if item]
        if len(fields) < 2:
            raise ValueError(
                f"Trace line {line_number} in {path} requires lon and lat."
            )
        try:
            current.append((float(fields[0]), float(fields[1])))
        except ValueError as exc:
            raise ValueError(
                f"Trace line {line_number} in {path} contains non-numeric "
                "longitude/latitude."
            ) from exc
    if current:
        segments.append(np.asarray(current, dtype=float))
    return segments


def _geojson_segments(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        document = json.load(stream)
    if document.get("type") == "FeatureCollection":
        features = document.get("features") or []
    elif document.get("type") == "Feature":
        features = [document]
    elif document.get("type") in {"LineString", "MultiLineString"}:
        features = [{"type": "Feature", "geometry": document, "properties": {}}]
    else:
        raise ValueError(
            "Trace GeoJSON must be a FeatureCollection, Feature, LineString, "
            "or MultiLineString."
        )
    segments = []
    names = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates") or []
        properties = feature.get("properties") or {}
        feature_name = properties.get("name")
        if geometry_type == "LineString":
            segments.append(np.asarray(coordinates, dtype=float))
            names.append(feature_name)
        elif geometry_type == "MultiLineString":
            for part_index, part in enumerate(coordinates):
                segments.append(np.asarray(part, dtype=float))
                names.append(
                    f"{feature_name} {part_index + 1}"
                    if feature_name
                    else None
                )
        else:
            raise ValueError(
                "Trace GeoJSON features must use LineString or MultiLineString."
            )
    return segments, names


def read_reference_paths(path):
    """Return immutable reference paths from TXT/GMT or GeoJSON line data."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference trace does not exist: {path}.")
    suffix = path.suffix.lower()
    if suffix in _GEOJSON_SUFFIXES:
        segments, names = _geojson_segments(path)
    elif suffix in _TEXT_SUFFIXES or not suffix:
        segments = _text_segments(path)
        names = [None] * len(segments)
    else:
        raise ValueError(
            "Reference trace format must be TXT/DAT/TRACE/GMT or GeoJSON."
        )
    if not segments:
        raise ValueError(f"Reference trace contains no line coordinates: {path}.")
    result = []
    for index, coordinates in enumerate(segments):
        result.append(
            ReferencePath(
                id=_path_id(path.stem, index, len(segments)),
                name=names[index] or _path_id(path.stem, index, len(segments)),
                coordinates=coordinates[:, :2],
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


def _open_output(path, overwrite):
    path = Path(path)
    if not path.parent.exists():
        raise FileNotFoundError(
            f"Trace output directory does not exist: {path.parent}."
        )
    return path.open("w" if overwrite else "x", encoding="utf-8", newline="\n")


def _save_text(path, draft, overwrite):
    with _open_output(path, overwrite) as stream:
        stream.write("# ECAT adjusted trace\n")
        stream.write(f"# name: {draft.name}\n")
        stream.write("# longitude latitude\n")
        for longitude, latitude in draft.coordinates:
            stream.write(f"{longitude:.8f} {latitude:.8f}\n")


def _save_geojson(path, draft, overwrite):
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(longitude), float(latitude)]
                        for longitude, latitude in draft.coordinates
                    ],
                },
                "properties": {
                    "name": draft.name,
                    "ecat_role": "adjusted_trace",
                    "source_layer_id": draft.source_layer_id,
                },
            }
        ],
    }
    with _open_output(path, overwrite) as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def save_trace(path, draft, *, overwrite=False):
    """Save one validated working trace without silently replacing a source."""

    if not isinstance(draft, PathDraft):
        raise TypeError("save_trace expects a PathDraft.")
    if len(draft.coordinates) < 2:
        raise ValueError("A saved trace requires at least two vertices.")
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _GEOJSON_SUFFIXES:
        _save_geojson(path, draft, overwrite)
    elif suffix in _TEXT_SUFFIXES or not suffix:
        _save_text(path, draft, overwrite)
    else:
        raise ValueError(
            "Trace output must end in .txt, .dat, .trace, .gmt, .json, or "
            ".geojson."
        )
    return path.resolve()


__all__ = [
    "read_reference_paths",
    "reference_path_from_arrays",
    "save_trace",
]
