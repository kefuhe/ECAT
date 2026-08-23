"""Shared readers and safe writers for ECAT fault-trace polylines."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


TEXT_SUFFIXES = {".txt", ".dat", ".trace", ".gmt"}
GEOJSON_SUFFIXES = {".json", ".geojson"}


@dataclass(frozen=True)
class TraceSegment:
    """One named longitude/latitude polyline read from a trace file."""

    coordinates: np.ndarray
    name: str | None = None
    source_file: Path | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.coordinates, dtype=float)
        if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] < 2:
            raise ValueError("a trace segment requires at least two lon/lat points.")
        values = np.array(values[:, :2], dtype=float, copy=True)
        if not np.all(np.isfinite(values)):
            raise ValueError("trace coordinates contain NaN or infinite values.")
        values.setflags(write=False)
        object.__setattr__(self, "coordinates", values)
        if self.source_file is not None:
            object.__setattr__(self, "source_file", Path(self.source_file).resolve())


def _coordinate_column_indices(columns: Sequence[str] | None) -> tuple[int, int]:
    if columns is None:
        return 0, 1
    if isinstance(columns, (str, bytes)):
        raise ValueError("trace columns must be a sequence of column names.")
    names = [str(name).strip().lower() for name in columns]
    if len(names) < 2:
        raise ValueError("trace columns must contain at least longitude and latitude.")
    if "lon" in names and "lat" in names:
        return names.index("lon"), names.index("lat")
    if "longitude" in names and "latitude" in names:
        return names.index("longitude"), names.index("latitude")
    return 0, 1


def _text_segments(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
    sep: str | None = None,
    comment: str | None = "#",
) -> list[TraceSegment]:
    segments: list[TraceSegment] = []
    current: list[tuple[float, float]] = []
    longitude_index, latitude_index = _coordinate_column_indices(columns)
    required_index = max(longitude_index, latitude_index)
    separator = r"[\s,]+" if sep is None else str(sep)
    if comment is not None and not str(comment):
        raise ValueError("trace comment marker must be non-empty or null.")
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line
        if comment is not None:
            line = line.split(str(comment), 1)[0]
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                segments.append(TraceSegment(current, source_file=path))
                current = []
            continue
        fields = [item for item in re.split(separator, line) if item]
        if len(fields) <= required_index:
            raise ValueError(
                f"trace line {line_number} in {path} does not contain the configured "
                "longitude/latitude columns."
            )
        try:
            current.append(
                (float(fields[longitude_index]), float(fields[latitude_index]))
            )
        except ValueError as exc:
            raise ValueError(
                f"trace line {line_number} in {path} contains non-numeric longitude/latitude."
            ) from exc
    if current:
        segments.append(TraceSegment(current, source_file=path))
    return segments


def _geojson_features(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    kind = document.get("type")
    if kind == "FeatureCollection":
        return list(document.get("features") or [])
    if kind == "Feature":
        return [document]
    if kind in {"LineString", "MultiLineString"}:
        return [{"type": "Feature", "geometry": document, "properties": {}}]
    raise ValueError(
        "trace GeoJSON must be a FeatureCollection, Feature, LineString, or MultiLineString."
    )


def _geojson_segments(path: Path) -> list[TraceSegment]:
    with path.open("r", encoding="utf-8-sig") as stream:
        document = json.load(stream)
    segments: list[TraceSegment] = []
    for feature in _geojson_features(document):
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates") or []
        properties = feature.get("properties") or {}
        feature_name = properties.get("name")
        if geometry_type == "LineString":
            segments.append(
                TraceSegment(coordinates, name=feature_name, source_file=path)
            )
        elif geometry_type == "MultiLineString":
            for index, part in enumerate(coordinates):
                name = f"{feature_name} {index + 1}" if feature_name else None
                segments.append(TraceSegment(part, name=name, source_file=path))
        else:
            raise ValueError("trace GeoJSON features must use LineString or MultiLineString.")
    return segments


def read_trace_segments(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    sep: str | None = None,
    comment: str | None = "#",
) -> tuple[TraceSegment, ...]:
    """Read TXT/DAT/TRACE/GMT or GeoJSON line data without merging parts.

    Text inputs consume only the configured longitude/latitude columns.  Extra
    columns may be present and are intentionally ignored by this 2-D surface
    trace protocol.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"trace does not exist: {source}.")
    suffix = source.suffix.lower()
    if suffix in GEOJSON_SUFFIXES:
        segments = _geojson_segments(source)
    elif suffix in TEXT_SUFFIXES or not suffix:
        segments = _text_segments(
            source,
            columns=columns,
            sep=sep,
            comment=comment,
        )
    else:
        raise ValueError("trace format must be TXT/DAT/TRACE/GMT or GeoJSON.")
    if not segments:
        raise ValueError(f"trace contains no line coordinates: {source}.")
    return tuple(segments)


def read_trace(
    path: str | Path,
    *,
    segment: int | None = None,
    columns: Sequence[str] | None = None,
    sep: str | None = None,
    comment: str | None = "#",
) -> TraceSegment:
    """Read exactly one trace segment, requiring an index for multipart input."""
    segments = read_trace_segments(
        path,
        columns=columns,
        sep=sep,
        comment=comment,
    )
    if segment is None:
        if len(segments) != 1:
            raise ValueError(
                f"trace contains {len(segments)} segments; choose one with segment=<index>."
            )
        return segments[0]
    if isinstance(segment, bool) or not isinstance(segment, int) or segment < 0:
        raise ValueError("trace segment must be a zero-based non-negative integer.")
    try:
        return segments[segment]
    except IndexError as exc:
        raise IndexError(f"trace segment index {segment} is out of range.") from exc


def _open_output(path: Path, overwrite: bool):
    if not path.parent.exists():
        raise FileNotFoundError(f"trace output directory does not exist: {path.parent}.")
    return path.open("w" if overwrite else "x", encoding="utf-8", newline="\n")


def write_trace(
    path: str | Path,
    coordinates: Any,
    *,
    name: str = "Processed trace",
    role: str = "processed_trace",
    properties: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    text_title: str = "ECAT processed trace",
) -> Path:
    """Safely write one lon/lat trace without silently replacing a source."""
    target = Path(path)
    segment = TraceSegment(coordinates, name=name)
    suffix = target.suffix.lower()
    extra_properties = dict(properties or {})
    if suffix in GEOJSON_SUFFIXES:
        feature_properties = {
            "name": name,
            "ecat_role": role,
            **extra_properties,
        }
        document = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": segment.coordinates.tolist(),
                    },
                    "properties": feature_properties,
                }
            ],
        }
        with _open_output(target, overwrite) as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    elif suffix in TEXT_SUFFIXES or not suffix:
        with _open_output(target, overwrite) as stream:
            stream.write(f"# {text_title}\n")
            stream.write(f"# name: {name}\n")
            stream.write("# longitude latitude\n")
            for longitude, latitude in segment.coordinates:
                stream.write(f"{longitude:.8f} {latitude:.8f}\n")
    else:
        raise ValueError(
            "trace output must end in .txt, .dat, .trace, .gmt, .json, or .geojson."
        )
    return target.resolve()


__all__ = [
    "GEOJSON_SUFFIXES",
    "TEXT_SUFFIXES",
    "TraceSegment",
    "read_trace",
    "read_trace_segments",
    "write_trace",
]
