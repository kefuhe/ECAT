"""Renderer-independent state for interactive ECAT polyline editing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

import numpy as np


def canonical_longitude(values):
    """Return finite longitudes in the ECAT display interval ``[-180, 180)``."""

    values = np.asarray(values, dtype=float)
    return ((values + 180.0) % 360.0) - 180.0


def _readonly_array(values, *, dtype=float):
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _coordinates(values, *, allow_empty=True):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        if not allow_empty:
            raise ValueError("A trace requires at least one coordinate.")
        array = np.empty((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Trace coordinates must have shape (n, 2) as lon/lat.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Trace longitude/latitude must be finite.")
    if np.any((array[:, 1] < -90.0) | (array[:, 1] > 90.0)):
        raise ValueError("Trace latitude must be within [-90, 90] degrees.")
    array = np.array(array, dtype=float, copy=True)
    if array.size:
        array[:, 0] = canonical_longitude(array[:, 0])
    array.setflags(write=False)
    return array


def _frozen_metadata(values):
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True)
class ReferencePath:
    """One immutable published or user-supplied reference polyline."""

    id: str
    name: str
    coordinates: np.ndarray
    source_file: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        path_id = str(self.id).strip()
        name = str(self.name).strip()
        if not path_id or not name:
            raise ValueError("Reference path id and name must not be empty.")
        coordinates = _coordinates(self.coordinates, allow_empty=False)
        if coordinates.shape[0] < 2:
            raise ValueError("A reference path requires at least two vertices.")
        source_file = self.source_file
        if source_file is not None:
            source_file = Path(source_file).resolve()
        object.__setattr__(self, "id", path_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "source_file", source_file)
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True)
class PathState:
    """Immutable history snapshot for one active working path."""

    id: str
    name: str
    purpose: str
    coordinates: tuple[tuple[float, float], ...]
    source_layer_id: str | None
    source_file: Path | None
    saved_path: Path | None
    dirty: bool
    metadata: Mapping[str, Any]


@dataclass
class PathDraft:
    """Mutable working copy; source/reference geometry is never changed."""

    id: str = field(default_factory=lambda: f"draft-{uuid4().hex[:12]}")
    name: str = "Adjusted trace"
    purpose: str = "trace"
    coordinates: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=float)
    )
    source_layer_id: str | None = None
    source_file: Path | None = None
    saved_path: Path | None = None
    dirty: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.id = str(self.id).strip()
        self.name = str(self.name).strip()
        self.purpose = str(self.purpose).strip().lower()
        if not self.id or not self.name:
            raise ValueError("Working path id and name must not be empty.")
        if self.purpose not in {"trace", "profile"}:
            raise ValueError("Path purpose must be trace or profile.")
        self.coordinates = _coordinates(self.coordinates)
        self.source_layer_id = (
            str(self.source_layer_id).strip()
            if self.source_layer_id is not None
            else None
        )
        self.source_file = (
            Path(self.source_file).resolve()
            if self.source_file is not None
            else None
        )
        self.saved_path = (
            Path(self.saved_path).resolve()
            if self.saved_path is not None
            else None
        )
        self.metadata = _frozen_metadata(self.metadata)
        self.dirty = bool(self.dirty)

    def with_coordinates(self, values, *, dirty=True):
        """Return a detached draft with validated replacement coordinates."""

        return replace(
            self,
            coordinates=_coordinates(values),
            dirty=bool(dirty),
        )

    def snapshot(self):
        """Return an immutable, array-free history representation."""

        return PathState(
            id=self.id,
            name=self.name,
            purpose=self.purpose,
            coordinates=tuple(
                (float(longitude), float(latitude))
                for longitude, latitude in self.coordinates
            ),
            source_layer_id=self.source_layer_id,
            source_file=self.source_file,
            saved_path=self.saved_path,
            dirty=self.dirty,
            metadata=_frozen_metadata(self.metadata),
        )

    @classmethod
    def from_state(cls, state):
        """Restore a detached working copy from :class:`PathState`."""

        if not isinstance(state, PathState):
            raise TypeError("PathDraft.from_state expects a PathState.")
        return cls(
            id=state.id,
            name=state.name,
            purpose=state.purpose,
            coordinates=state.coordinates,
            source_layer_id=state.source_layer_id,
            source_file=state.source_file,
            saved_path=state.saved_path,
            dirty=state.dirty,
            metadata=state.metadata,
        )


class EditHistory:
    """Bounded undo/redo history for one optional working path."""

    def __init__(self, initial=None, *, max_entries=50):
        max_entries = int(max_entries)
        if max_entries < 1:
            raise ValueError("Edit history max_entries must be positive.")
        self.max_entries = max_entries
        self._past = []
        self._current = initial
        self._future = []

    @property
    def current(self):
        return self._current

    @property
    def can_undo(self):
        return bool(self._past)

    @property
    def can_redo(self):
        return bool(self._future)

    def record(self, state):
        """Record one completed edit and discard a stale redo branch."""

        if state == self._current:
            return state
        self._past.append(self._current)
        self._past = self._past[-self.max_entries :]
        self._current = state
        self._future.clear()
        return state

    def replace_current(self, state):
        """Update non-geometric state, such as a successful save checkpoint."""

        self._current = state
        return state

    def undo(self):
        if not self._past:
            return self._current
        self._future.append(self._current)
        self._current = self._past.pop()
        return self._current

    def redo(self):
        if not self._future:
            return self._current
        self._past.append(self._current)
        self._past = self._past[-self.max_entries :]
        self._current = self._future.pop()
        return self._current


class InteractiveWorkspace:
    """Small session state for reference paths and one active working copy."""

    def __init__(self, references=(), *, max_history=50):
        self.references = tuple(references)
        if not all(isinstance(item, ReferencePath) for item in self.references):
            raise TypeError("Workspace references must be ReferencePath objects.")
        ids = [item.id for item in self.references]
        if len(set(ids)) != len(ids):
            raise ValueError("Reference path ids must be unique.")
        self.active = None
        self.history = EditHistory(None, max_entries=max_history)

    def _state(self):
        return self.active.snapshot() if self.active is not None else None

    def _restore(self, state):
        self.active = PathDraft.from_state(state) if state is not None else None
        return self.active

    def _commit(self, draft):
        self.active = draft
        self.history.record(self._state())
        return self.active

    def new_path(self, *, name="Adjusted trace"):
        return self._commit(PathDraft(name=name, purpose="trace"))

    def copy_reference(self, reference_id, *, name=None):
        reference = next(
            (item for item in self.references if item.id == reference_id),
            None,
        )
        if reference is None:
            raise KeyError(f"Unknown reference path id: {reference_id!r}.")
        draft = PathDraft(
            name=name or f"{reference.name} adjusted",
            coordinates=reference.coordinates,
            source_layer_id=reference.id,
            source_file=reference.source_file,
            metadata={"created_from": "reference"},
            dirty=True,
        )
        return self._commit(draft)

    def replace_coordinates(self, coordinates):
        if self.active is None:
            self.new_path()
        return self._commit(self.active.with_coordinates(coordinates))

    def add_vertex(self, longitude, latitude):
        coordinates = np.vstack(
            (self.active.coordinates if self.active is not None else np.empty((0, 2)),
             [[longitude, latitude]])
        )
        return self.replace_coordinates(coordinates)

    def insert_vertex(self, index, longitude, latitude):
        if self.active is None:
            raise ValueError("Create or copy a working path before inserting vertices.")
        index = int(index)
        if not 0 <= index <= len(self.active.coordinates):
            raise IndexError("Vertex insertion index is out of range.")
        coordinates = np.insert(
            self.active.coordinates,
            index,
            [longitude, latitude],
            axis=0,
        )
        return self.replace_coordinates(coordinates)

    def move_vertex(self, index, longitude, latitude):
        if self.active is None:
            raise ValueError("There is no active working path.")
        index = int(index)
        if not 0 <= index < len(self.active.coordinates):
            raise IndexError("Vertex index is out of range.")
        coordinates = np.array(self.active.coordinates, copy=True)
        coordinates[index] = [longitude, latitude]
        return self.replace_coordinates(coordinates)

    def delete_vertex(self, index):
        if self.active is None:
            raise ValueError("There is no active working path.")
        index = int(index)
        if not 0 <= index < len(self.active.coordinates):
            raise IndexError("Vertex index is out of range.")
        return self.replace_coordinates(
            np.delete(self.active.coordinates, index, axis=0)
        )

    def clear_path(self):
        if self.active is None:
            return None
        return self.replace_coordinates(np.empty((0, 2)))

    def delete_path(self):
        return self._commit(None)

    def undo(self):
        return self._restore(self.history.undo())

    def redo(self):
        return self._restore(self.history.redo())

    def validate_for_save(self):
        if self.active is None:
            raise ValueError("There is no working path to save.")
        if len(self.active.coordinates) < 2:
            raise ValueError("A saved trace requires at least two vertices.")
        return self.active

    def mark_saved(self, path):
        draft = self.validate_for_save()
        self.active = replace(
            draft,
            saved_path=Path(path).resolve(),
            dirty=False,
        )
        self.history.replace_current(self._state())
        return self.active


@dataclass(frozen=True)
class EditorBackground:
    """Detached quantitative background shown behind editable geometry."""

    name: str
    longitude: np.ndarray
    latitude: np.ndarray
    values: np.ndarray
    valid_mask: np.ndarray
    units: str | None = None
    variable: str | None = None
    source_identity: str | None = None
    style: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        longitude = np.asarray(self.longitude, dtype=float)
        latitude = np.asarray(self.latitude, dtype=float)
        values = np.asarray(self.values, dtype=float)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if not (
            longitude.shape == latitude.shape == values.shape == valid.shape
        ):
            raise ValueError(
                "Editor background longitude, latitude, values, and mask "
                "must have identical shapes."
            )
        valid = (
            valid
            & np.isfinite(longitude)
            & np.isfinite(latitude)
            & np.isfinite(values)
        )
        if not np.any(valid):
            raise ValueError("Editor background contains no finite valid values.")
        object.__setattr__(self, "name", str(self.name).strip() or "Observation")
        object.__setattr__(self, "longitude", _readonly_array(longitude))
        object.__setattr__(self, "latitude", _readonly_array(latitude))
        object.__setattr__(self, "values", _readonly_array(values))
        object.__setattr__(self, "valid_mask", _readonly_array(valid, dtype=bool))
        object.__setattr__(self, "style", _frozen_metadata(self.style))


@dataclass
class TraceEditorSession:
    """Backend-neutral input for one local trace-editor server."""

    background: EditorBackground
    workspace: InteractiveWorkspace
    output_path: Path
    title: str = "ECAT trace editor"

    def __post_init__(self):
        if not isinstance(self.background, EditorBackground):
            raise TypeError("TraceEditorSession requires an EditorBackground.")
        if not isinstance(self.workspace, InteractiveWorkspace):
            raise TypeError("TraceEditorSession requires an InteractiveWorkspace.")
        self.output_path = Path(self.output_path)
        self.title = str(self.title).strip() or "ECAT trace editor"


__all__ = [
    "EditHistory",
    "EditorBackground",
    "InteractiveWorkspace",
    "PathDraft",
    "PathState",
    "ReferencePath",
    "TraceEditorSession",
    "canonical_longitude",
]
