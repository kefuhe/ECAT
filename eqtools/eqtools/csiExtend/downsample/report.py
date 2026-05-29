import numpy as np
import yaml
from matplotlib import path as mpath


def downsample_report_file(config, out_name):
    report_file = (config or {}).get("report_file", "auto")
    if report_file in (None, False):
        return None
    if str(report_file).lower() == "auto":
        return f"{out_name}_downsample_report.yml"
    return str(report_file)


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_percentile(values, q):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.nanpercentile(values, q))


def _value_summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    summary = {
        "finite_count": int(finite.size),
        "total_count": int(values.size),
    }
    if finite.size:
        summary.update(
            {
                "nanmean": float(np.nanmean(finite)),
                "nanmedian": float(np.nanmedian(finite)),
                "nanstd": float(np.nanstd(finite)),
                "robust_99": [_safe_percentile(finite, 0.5), _safe_percentile(finite, 99.5)],
                "full_range": [float(np.nanmin(finite)), float(np.nanmax(finite))],
            }
        )
    return summary


def _cell_weight_summary(newimage):
    weights = np.asarray(getattr(newimage, "wgt", []), dtype=float)
    finite = weights[np.isfinite(weights)]
    if finite.size == 0:
        return {"output_cells": 0}
    return {
        "output_cells": int(finite.size),
        "min_cell_pixels": int(np.nanmin(finite)),
        "median_cell_pixels": float(np.nanmedian(finite)),
        "max_cell_pixels": int(np.nanmax(finite)),
    }


def _cell_indices(downsampler, block):
    block = np.asarray(block, dtype=float)
    block_center = np.mean(block, axis=0)
    block_radius = np.max(np.linalg.norm(block - block_center, axis=1))
    indices = downsampler.PIXXY_Tree.query_ball_point(block_center, r=block_radius)
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return indices
    mask = mpath.Path(block, closed=False).contains_points(downsampler.PIXXY[indices])
    return indices[mask]


def _reconstruction_quality(data, data_type, downsampler):
    newimage = getattr(downsampler, "newimage", None)
    blocks = getattr(downsampler, "blocks", None)
    if newimage is None or blocks is None or not hasattr(downsampler, "PIXXY_Tree"):
        return {}

    sum_sq = 0.0
    count = 0
    if data_type == "sar" and hasattr(data, "vel") and hasattr(newimage, "vel"):
        cell_values = np.asarray(newimage.vel, dtype=float)
        source_values = np.asarray(data.vel, dtype=float)
        for block, cell_value in zip(blocks, cell_values):
            indices = _cell_indices(downsampler, block)
            residual = source_values[indices] - cell_value
            residual = residual[np.isfinite(residual)]
            sum_sq += float(np.sum(residual**2))
            count += int(residual.size)
    elif data_type == "optical" and hasattr(data, "east") and hasattr(newimage, "east"):
        source_east = np.asarray(data.east, dtype=float)
        source_north = np.asarray(data.north, dtype=float)
        cell_east = np.asarray(newimage.east, dtype=float)
        cell_north = np.asarray(newimage.north, dtype=float)
        for block, east_value, north_value in zip(blocks, cell_east, cell_north):
            indices = _cell_indices(downsampler, block)
            residual2 = (source_east[indices] - east_value) ** 2 + (source_north[indices] - north_value) ** 2
            residual2 = residual2[np.isfinite(residual2)]
            sum_sq += float(np.sum(residual2))
            count += int(residual2.size)
    else:
        return {}

    if count == 0:
        return {}
    return {
        "reconstruction_rms": float(np.sqrt(sum_sq / count)),
        "reconstruction_points": count,
    }


def _observation_summary(data, data_type):
    if data_type == "sar" and hasattr(data, "vel"):
        return {"vel": _value_summary(data.vel)}
    if data_type == "optical" and hasattr(data, "east") and hasattr(data, "north"):
        east = np.asarray(data.east, dtype=float)
        north = np.asarray(data.north, dtype=float)
        return {
            "east": _value_summary(east),
            "north": _value_summary(north),
            "magnitude": _value_summary(np.sqrt(east**2 + north**2)),
        }
    return {}


def build_downsample_report(data, data_type, config, downsampler, out_name):
    downsample = config.get("downsample", {})
    newimage = getattr(downsampler, "newimage", None)
    input_count = int(np.asarray(getattr(data, "x", [])).size)
    cell_summary = _cell_weight_summary(newimage) if newimage is not None else {"output_cells": 0}
    output_cells = cell_summary.get("output_cells", 0)
    report = {
        "out_name": out_name,
        "data_type": data_type,
        "method": downsample.get("method"),
        "input": {
            "processing_points": input_count,
            "processing_region": getattr(data, "processing_region_report", None),
        },
        "grid": {
            **cell_summary,
            "reduction_ratio": float(input_count / output_cells) if output_cells else None,
        },
        "extraction": downsample.get("extraction", {}),
        "guide_grid": getattr(downsampler, "guide_grid_report", {"enabled": False}),
        "observation": _observation_summary(data, data_type),
    }

    method = downsample.get("method")
    if method == "std":
        report["std_config"] = {
            key: downsample.get("std_config", {}).get(key)
            for key in (
                "startingsize",
                "minimumsize",
                "min_valid_fraction",
                "split_std_threshold",
                "split_metric_correction",
                "split_metric_smoothing",
                "use_variance",
                "amplitude_stat",
            )
        }
    elif method == "data":
        report["data_config"] = {
            key: downsample.get("data_config", {}).get(key)
            for key in (
                "startingsize",
                "minimumsize",
                "min_valid_fraction",
                "split_metric_threshold",
                "split_metric",
                "split_metric_smoothing",
            )
        }
    elif method == "trirb":
        report["trirb_config"] = {
            key: downsample.get("trirb_config", {}).get(key)
            for key in (
                "minimumsize",
                "min_valid_fraction",
                "max_samples",
                "change_threshold",
                "smooth_factor",
                "slipdirection",
            )
        }

    if (downsample.get("report") or {}).get("quality", True):
        quality = _reconstruction_quality(data, data_type, downsampler)
        if quality:
            report["quality"] = quality
    return _to_builtin(report)


def write_downsample_report(report, config, out_name):
    report_config = (config.get("downsample", {}) or {}).get("report", {})
    if not report_config.get("enabled", True):
        return None
    path = downsample_report_file(report_config, out_name)
    if path is None:
        return None
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(report, file, allow_unicode=True, sort_keys=False)
    return path


def format_downsample_report(report, report_file=None):
    grid = report.get("grid", {})
    observation = report.get("observation", {})
    observation_key = "vel" if "vel" in observation else "magnitude"
    observation_std = (observation.get(observation_key) or {}).get("nanstd")
    lines = [
        "Downsample report:",
        f"  method       : {report.get('method')}",
        f"  input points : {report.get('input', {}).get('processing_points')}",
        f"  input nanstd : {observation_std}",
        f"  output cells : {grid.get('output_cells')}",
        f"  reduction    : {grid.get('reduction_ratio')}",
    ]
    guide = report.get("guide_grid", {})
    if guide.get("enabled"):
        lines.append(f"  guide backend: {guide.get('backend')}")
    if report_file:
        lines.append(f"  report file  : {report_file}")
    return "\n".join(lines)
