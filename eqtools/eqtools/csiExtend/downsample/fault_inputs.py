from pathlib import Path

import numpy as np
import pandas as pd

from ..trace_io import read_trace_segments


def _entries(section):
    if section is None:
        return []
    if isinstance(section, list):
        return section
    if isinstance(section, dict):
        enabled = section.get("enabled", True)
        return [
            {**entry, "enabled": entry.get("enabled", enabled)}
            for entry in section.get("sources", [])
        ]
    raise ValueError("fault_traces/fault_models must be a list or a mapping with sources.")


def _enabled_entries(section):
    return [entry for entry in _entries(section) if entry.get("enabled", False)]


def _as_list(value, default=None):
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [value]
    return list(value)


def _stage_enabled(entry, stage, *, default=("raw", "decim")):
    stages = _as_list(entry.get("stages"), default=default)
    stages = {str(item).replace("-", "_").lower() for item in stages}
    return "all" in stages or str(stage).replace("-", "_").lower() in stages


def _resolve_path(path, base_dir=None):
    path = Path(path)
    if path.is_absolute() or base_dir is None:
        return path
    return Path(base_dir) / path


def _trace_segments_for_entry(entry, base_dir=None):
    file_path = _resolve_path(entry["file"], base_dir=base_dir)
    return file_path, read_trace_segments(
        file_path,
        columns=entry.get("columns"),
        sep=entry.get("sep"),
        comment=entry.get("comment", "#"),
    )


def _segment_indices(selection, segment_count, *, label):
    if selection is None or selection == "all":
        return list(range(segment_count))
    if isinstance(selection, bool):
        raise ValueError(f"{label} must be 'all', an integer, or a list of integers.")
    if isinstance(selection, int):
        indices = [selection]
    elif isinstance(selection, (list, tuple)):
        indices = list(selection)
        if not indices:
            raise ValueError(f"{label} must not be empty.")
    else:
        raise ValueError(f"{label} must be 'all', an integer, or a list of integers.")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError(f"{label} must contain only integer segment indices.")
    if any(index < 0 or index >= segment_count for index in indices):
        raise IndexError(
            f"{label} selects a segment outside the available range "
            f"0..{segment_count - 1}."
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label} must not contain duplicate segment indices.")
    return indices


def _trace_frame(entry, file_path, segment, *, segment_index, segment_count):
    data = pd.DataFrame(segment.coordinates, columns=["lon", "lat"])
    base_id = str(entry.get("id") or file_path.stem)
    data.attrs["id"] = (
        base_id if segment_count == 1 else f"{base_id}.{segment_index + 1}"
    )
    data.attrs["source_file"] = str(file_path)
    data.attrs["source_trace_id"] = base_id
    data.attrs["source_segment_index"] = int(segment_index)
    data.attrs["source_segment_count"] = int(segment_count)
    if segment.name:
        data.attrs["source_segment_name"] = segment.name
    marker = entry.get("marker")
    if marker is not None:
        markersize = entry.get("markersize")
        data.attrs["plot_marker"] = marker
        data.attrs["plot_markersize"] = 3.0 if markersize is None else float(markersize)
    return data


def read_trace_file(entry, base_dir=None):
    """Read one trace for computation, requiring a selector for multipart input."""
    file_path, segments = _trace_segments_for_entry(entry, base_dir=base_dir)
    selection = entry.get("segment")
    if selection is None:
        if len(segments) != 1:
            raise ValueError(
                f"trace {file_path} contains {len(segments)} segments; set segment "
                "to the zero-based segment index required for computation."
            )
        segment_index = 0
    else:
        if isinstance(selection, bool) or not isinstance(selection, int):
            raise ValueError("segment must be a zero-based non-negative integer.")
        segment_index = _segment_indices(
            selection,
            len(segments),
            label="segment",
        )[0]
    return _trace_frame(
        entry,
        file_path,
        segments[segment_index],
        segment_index=segment_index,
        segment_count=len(segments),
    )


def load_fault_traces(config, base_dir=None, stage=None):
    traces = []
    for entry in _enabled_entries(config.get("fault_traces")):
        if stage is not None and not _stage_enabled(entry, stage):
            continue
        file_path, segments = _trace_segments_for_entry(entry, base_dir=base_dir)
        indices = _segment_indices(
            entry.get("segments", "all"),
            len(segments),
            label=f"fault_traces[{entry.get('id', file_path.stem)!r}].segments",
        )
        traces.extend(
            _trace_frame(
                entry,
                file_path,
                segments[index],
                segment_index=index,
                segment_count=len(segments),
            )
            for index in indices
        )
    return traces


def _fault_model_type(entry):
    if entry.get("type") is not None:
        return str(entry["type"]).replace("-", "_").lower()
    if entry.get("file") is not None:
        return "csi_gmt"
    return "generated_from_trace"


def _fault_geometry(entry):
    return str(entry.get("geometry", "triangular")).replace("-", "_").lower()


def build_generated_fault_model(entry, lon0, lat0, triangular_cls, base_dir=None):
    if triangular_cls is None:
        raise RuntimeError("TriangularPatches is required to build generated fault models.")
    trace_entry = {
        "file": entry.get("trace_file"),
        "columns": entry.get("columns", ["lon", "lat"]),
        "comment": entry.get("comment", "#"),
        "sep": entry.get("sep", r"\s+"),
        "id": entry.get("id"),
        "segment": entry.get("segment"),
    }
    trace = read_trace_file(trace_entry, base_dir=base_dir)
    fault = triangular_cls(entry.get("id", "Triangular Fault"), lon0=lon0, lat0=lat0, verbose=True)
    fault.trace(trace.lon.values, trace.lat.values)
    fault.top = entry.get("top_depth", 3.0)
    fault.depth = entry.get("bottom_depth", 15.0)
    fault.set_top_coords_from_trace()
    fault.generate_bottom_from_single_dip(
        dip_angle=entry["dip_angle"],
        dip_direction=entry["dip_direction"],
    )
    fault.generate_mesh(
        top_size=entry["top_size"],
        bottom_size=entry["bottom_size"],
        verbose=0,
        show=False,
    )
    fault.initializeslip(values="depth")
    fault._eqtools_fault_id = entry.get("id")
    return fault


def read_csi_gmt_fault_model(entry, lon0, lat0, triangular_cls, rectangular_cls, base_dir=None):
    geometry = _fault_geometry(entry)
    file_path = _resolve_path(entry["file"], base_dir=base_dir)
    common = {
        "readpatchindex": entry.get("readpatchindex", True),
        "donotreadslip": entry.get("donotreadslip", True),
        "inputCoordinates": entry.get("input_coordinates", entry.get("inputCoordinates", "lonlat")),
    }
    if geometry == "triangular":
        if triangular_cls is None:
            raise RuntimeError("TriangularPatches is required to read triangular CSI GMT fault models.")
        fault = triangular_cls(entry.get("id", file_path.stem), lon0=lon0, lat0=lat0, verbose=True)
        fault.readPatchesFromFile(
            str(file_path),
            gmtslip=entry.get("gmtslip", True),
            **common,
        )
    elif geometry == "rectangular":
        if rectangular_cls is None:
            raise RuntimeError("RectangularPatches is required to read rectangular CSI GMT fault models.")
        fault = rectangular_cls(entry.get("id", file_path.stem), lon0=lon0, lat0=lat0, verbose=True)
        fault.readPatchesFromFile(
            str(file_path),
            increasingy=entry.get("increasingy", True),
            **common,
        )
    else:
        raise ValueError("fault_models.geometry must be 'triangular' or 'rectangular'.")
    fault._eqtools_fault_id = entry.get("id", file_path.stem)
    fault._eqtools_fault_geometry = geometry
    return fault


def load_fault_model(entry, lon0, lat0, triangular_cls, rectangular_cls=None, base_dir=None):
    model_type = _fault_model_type(entry)
    geometry = _fault_geometry(entry)
    if model_type == "generated_from_trace":
        if geometry != "triangular":
            raise ValueError("generated_from_trace fault_models currently require geometry: triangular.")
        return build_generated_fault_model(entry, lon0, lat0, triangular_cls, base_dir=base_dir)
    if model_type == "csi_gmt":
        return read_csi_gmt_fault_model(
            entry,
            lon0,
            lat0,
            triangular_cls,
            rectangular_cls,
            base_dir=base_dir,
        )
    raise ValueError("fault_models.type must be 'generated_from_trace' or 'csi_gmt'.")


def select_fault_model_entries_for_compute(config, method):
    """Select enabled compute entries and enforce the required TriRB role."""
    method = str(method).replace("-", "_").lower()
    enabled_entries = _enabled_entries(config.get("fault_models"))
    selected = []
    for entry in enabled_entries:
        use_for = {
            str(item).replace("-", "_").lower()
            for item in _as_list(entry.get("use_for"))
        }
        if method not in use_for:
            continue
        if method == "trirb" and _fault_geometry(entry) != "triangular":
            raise ValueError(
                "downsample.method='trirb' supports only triangular fault_models."
            )
        selected.append(entry)

    if method == "trirb" and not selected:
        candidates = [
            str(
                entry.get("id")
                or entry.get("file")
                or entry.get("trace_file")
                or "<unnamed>"
            )
            for entry in enabled_entries
            if _fault_geometry(entry) == "triangular"
        ]
        if candidates:
            raise ValueError(
                "downsample.method='trirb' found enabled triangular fault_models "
                "but none selects the TriRB compute role: "
                f"{', '.join(candidates)}. Add use_for: [trirb] to each model "
                "that should participate."
            )
        raise ValueError(
            "downsample.method='trirb' requires at least one enabled triangular "
            "fault_model with use_for: [trirb]."
        )
    return selected


def load_fault_models_for_compute(config, method, lon0, lat0, triangular_cls, rectangular_cls=None, base_dir=None):
    models = []
    for entry in select_fault_model_entries_for_compute(config, method):
        models.append(
            load_fault_model(
                entry,
                lon0,
                lat0,
                triangular_cls,
                rectangular_cls=rectangular_cls,
                base_dir=base_dir,
            )
        )
    return models


def _patch_edge_overlays(fault):
    overlays = []
    patchll = getattr(fault, "patchll", None)
    if patchll is None:
        return overlays
    for index, patch in enumerate(patchll):
        patch = np.asarray(patch, dtype=float)
        if patch.ndim != 2 or patch.shape[1] < 2:
            continue
        lon = np.r_[patch[:, 0], patch[0, 0]]
        lat = np.r_[patch[:, 1], patch[0, 1]]
        data = pd.DataFrame({"lon": lon, "lat": lat})
        data.attrs["id"] = f"{getattr(fault, '_eqtools_fault_id', 'fault')}_patch_{index}"
        overlays.append(data)
    return overlays


def _fault_trace_overlay(fault):
    if hasattr(fault, "lon") and hasattr(fault, "lat"):
        return [pd.DataFrame({"lon": np.asarray(fault.lon), "lat": np.asarray(fault.lat)})]
    return []


def fault_model_overlays(fault, mode="edges"):
    mode = str(mode or "edges").replace("-", "_").lower()
    overlays = []
    if mode in ("trace", "outline", "both"):
        overlays.extend(_fault_trace_overlay(fault))
    if mode in ("edges", "patch_edges", "both") or not overlays:
        overlays.extend(_patch_edge_overlays(fault))
    return overlays


def load_fault_model_overlays(config, stage, lon0, lat0, triangular_cls, rectangular_cls=None, base_dir=None):
    overlays = []
    for entry in _enabled_entries(config.get("fault_models")):
        plot_config = entry.get("plot")
        if plot_config in (None, False):
            continue
        if plot_config is True:
            plot_config = {"stages": ["raw", "decim"]}
        if not _stage_enabled(plot_config, stage, default=()):
            continue
        fault = load_fault_model(
            entry,
            lon0,
            lat0,
            triangular_cls,
            rectangular_cls=rectangular_cls,
            base_dir=base_dir,
        )
        overlays.extend(fault_model_overlays(fault, mode=plot_config.get("mode", "edges")))
    return overlays


def load_plot_fault_overlays(config, stage, lon0, lat0, triangular_cls=None, rectangular_cls=None, base_dir=None):
    overlays = load_fault_traces(config, base_dir=base_dir, stage=stage)
    overlays.extend(
        load_fault_model_overlays(
            config,
            stage,
            lon0,
            lat0,
            triangular_cls,
            rectangular_cls=rectangular_cls,
            base_dir=base_dir,
        )
    )
    return overlays
