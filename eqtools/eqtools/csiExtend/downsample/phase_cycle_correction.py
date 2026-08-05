"""Explicit regional phase-cycle correction for unwrapped-phase SAR data."""

import numpy as np
import yaml

from .region_utils import lonlat_region_mask


def phase_cycle_correction_report_file(config, out_name):
    """Resolve the optional YAML report path."""

    report_file = (config or {}).get("report_file", "auto")
    if report_file in (None, False):
        return None
    if str(report_file).lower() == "auto":
        return f"{out_name}_phase_cycle_correction.yml"
    return str(report_file)


def _resolved_phase_wavelength(data):
    spec = getattr(data, "observation_spec", None)
    if spec is None and hasattr(data, "build_observation_spec"):
        spec = data.build_observation_spec()
    if spec is None:
        raise ValueError(
            "phase_cycle_correction requires a resolved SAR observation spec."
        )

    observation_type = getattr(spec, "observation_type", None)
    if hasattr(observation_type, "value"):
        observation_type = observation_type.value
    observation_type = str(observation_type).replace("-", "_").lower()
    if observation_type != "unwrapped_phase":
        raise ValueError(
            "phase_cycle_correction is valid only for "
            "sar_config.mode='unwrapped_phase'."
        )

    wavelength = getattr(spec, "wavelength", None)
    try:
        wavelength = float(wavelength)
    except (TypeError, ValueError):
        wavelength = np.nan
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError(
            "phase_cycle_correction requires a positive resolved wavelength."
        )
    return wavelength


def _finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "count": 0,
            "median": None,
            "mad": None,
            "min": None,
            "max": None,
        }
    median = float(np.median(values))
    return {
        "count": int(values.size),
        "median": median,
        "mad": float(np.median(np.abs(values - median))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _apply_to_csi_data(data, grid, corrected_grid):
    raw_indices = grid.raw_flat_indices()
    corrected_values = np.asarray(corrected_grid).reshape(-1)[raw_indices]
    current = np.asarray(data.vel)
    if corrected_values.size != current.size:
        raise ValueError(
            "phase_cycle_correction cannot align the full observation grid "
            f"with current CSI values: {corrected_values.size} versus "
            f"{current.size}."
        )
    data.observation_before_phase_cycle_correction = np.array(
        current,
        copy=True,
    )
    data.vel = np.array(corrected_values, dtype=float, copy=True)


def apply_phase_cycle_correction(
    data,
    grid,
    config,
    out_name="observation",
    base_dir=None,
    write_report=True,
):
    """Apply user-declared integer phase cycles to non-overlapping regions.

    ``cycles_to_remove=n`` means ``phase_corrected = phase_observed - 2*pi*n``.
    Under ECAT's toward-sensor convention this is the additive LOS correction
    ``+n*wavelength/2``.
    """

    config = config or {}
    enabled = bool(config.get("enabled", False))
    report = {
        "enabled": enabled,
        "formula_phase": (
            "phase_corrected = phase_observed - 2*pi*cycles_to_remove"
        ),
        "formula_los": (
            "los_corrected = los_observed + cycles_to_remove*wavelength/2"
        ),
        "corrections": [],
    }
    if not enabled:
        return report

    wavelength = _resolved_phase_wavelength(data)
    entries = config.get("corrections", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "phase_cycle_correction.corrections must be a non-empty list."
        )
    if "observation" not in grid.components:
        raise ValueError(
            "phase_cycle_correction requires a scalar SAR observation grid."
        )
    if grid.correction_surfaces:
        raise RuntimeError(
            "phase_cycle_correction must run before observation_correction."
        )

    original = np.asarray(grid.components["observation"], dtype=float)
    total_delta = np.zeros(grid.shape, dtype=float)
    already_selected = np.zeros(grid.shape, dtype=bool)
    applied_selected = np.zeros(grid.shape, dtype=bool)
    report["wavelength_m"] = wavelength

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"phase_cycle_correction.corrections[{index}] must be a mapping."
            )
        name = str(entry.get("name", f"region_{index + 1}"))
        cycles = entry.get("cycles_to_remove")
        if isinstance(cycles, bool) or not isinstance(cycles, (int, np.integer)):
            raise ValueError(
                f"phase_cycle_correction.corrections[{index}]."
                "cycles_to_remove must be an integer."
            )
        cycles = int(cycles)
        selector = entry.get("selector")
        mask = lonlat_region_mask(
            data,
            grid.longitude,
            grid.latitude,
            selector,
            base_dir=base_dir,
            label=(
                f"phase_cycle_correction.corrections[{index}].selector"
            ),
        )
        source_mask = mask & grid.source_valid_mask
        source_count = int(np.count_nonzero(source_mask))
        if source_count == 0:
            raise ValueError(
                f"phase_cycle_correction correction {name!r} selects no "
                "finite source observations."
            )
        analysis_mask = source_mask & grid.analysis_valid_mask
        analysis_count = int(np.count_nonzero(analysis_mask))
        if analysis_count == 0:
            raise ValueError(
                f"phase_cycle_correction correction {name!r} selects no "
                "active analysis observations."
            )
        overlap = source_mask & already_selected
        if np.any(overlap):
            raise ValueError(
                f"phase_cycle_correction correction {name!r} overlaps a "
                "previous correction region; correction regions must not overlap."
            )

        delta_m = cycles * wavelength / 2.0
        # The canonical observation grid represents the full finite source,
        # whereas ``data.vel`` contains only the active analysis subset.
        # Correct the full source region for lossless export, then map the
        # active entries back to the CSI object below.
        total_delta[source_mask] = delta_m
        already_selected |= source_mask
        applied_selected |= analysis_mask
        before = original[analysis_mask]
        report["corrections"].append(
            {
                "name": name,
                "cycles_to_remove": cycles,
                "phase_to_remove_rad": float(2.0 * np.pi * cycles),
                "los_delta_m": float(delta_m),
                "source_pixel_count": source_count,
                "analysis_pixel_count": analysis_count,
                "selector": selector,
                "before": _finite_stats(before),
                "after": _finite_stats(before + delta_m),
            }
        )

    corrected = original + total_delta
    corrected[~np.isfinite(original)] = np.nan
    grid.set_phase_cycle_delta("observation", total_delta, corrected)
    _apply_to_csi_data(data, grid, corrected)

    report["total_source_pixel_count"] = int(np.count_nonzero(already_selected))
    report["total_analysis_pixel_count"] = int(
        np.count_nonzero(applied_selected)
    )
    report_file = phase_cycle_correction_report_file(config, out_name)
    if write_report and config.get("report", True) and report_file:
        with open(report_file, "w", encoding="utf-8") as stream:
            yaml.safe_dump(report, stream, allow_unicode=True, sort_keys=False)
        report["report_file"] = report_file
    return report


def format_phase_cycle_correction_report(report):
    """Format a concise console summary."""

    if not report.get("enabled"):
        return ""
    lines = [
        "Regional phase-cycle correction:",
        f"  wavelength : {report['wavelength_m']:.8g} m",
    ]
    for item in report.get("corrections", []):
        lines.append(
            "  {name}: cycles_to_remove={cycles}, LOS delta={delta:.8g} m, "
            "source pixels={source}, analysis pixels={analysis}".format(
                name=item["name"],
                cycles=item["cycles_to_remove"],
                delta=item["los_delta_m"],
                source=item["source_pixel_count"],
                analysis=item["analysis_pixel_count"],
            )
        )
    if report.get("report_file"):
        lines.append(f"  report file: {report['report_file']}")
    return "\n".join(lines)
