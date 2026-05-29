from pathlib import Path

import numpy as np
import pandas as pd


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


def read_trace_file(entry, base_dir=None):
    file_path = _resolve_path(entry["file"], base_dir=base_dir)
    columns = entry.get("columns", ["lon", "lat"])
    if len(columns) < 2:
        raise ValueError("fault_traces columns must contain at least lon and lat.")
    data = pd.read_csv(
        file_path,
        names=list(columns),
        sep=entry.get("sep", r"\s+"),
        comment=entry.get("comment", "#"),
        engine="python",
    )
    if "lon" not in data or "lat" not in data:
        data = data.rename(columns={columns[0]: "lon", columns[1]: "lat"})
    data = data[["lon", "lat"]].astype(float)
    data.attrs["id"] = entry.get("id", file_path.stem)
    data.attrs["source_file"] = str(file_path)
    return data


def load_fault_traces(config, base_dir=None, stage=None):
    traces = []
    for entry in _enabled_entries(config.get("fault_traces")):
        if stage is not None and not _stage_enabled(entry, stage):
            continue
        traces.append(read_trace_file(entry, base_dir=base_dir))
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


def load_fault_models_for_compute(config, method, lon0, lat0, triangular_cls, rectangular_cls=None, base_dir=None):
    method = str(method).replace("-", "_").lower()
    models = []
    for entry in _enabled_entries(config.get("fault_models")):
        use_for = {str(item).replace("-", "_").lower() for item in _as_list(entry.get("use_for"))}
        if method not in use_for:
            continue
        geometry = _fault_geometry(entry)
        if method == "trirb" and geometry != "triangular":
            raise ValueError("downsample.method='trirb' supports only triangular fault_models.")
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
