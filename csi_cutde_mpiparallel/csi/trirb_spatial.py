"""Optional distance constraints on TriRB's interpolated mesh-size field.

This module owns the formula and a single run's immutable trace geometry.
It changes sampling priorities, not Green functions or observation conventions.
"""
import numpy as np
import shapely

from .fault_trace_selection import _fault_trace_geometry


def spatial_target_sizes(base, distance, *, minimumsize, near_distance,
                         far_distance, near_max_size, far_scale, max_target_size=None):
    """Blend near-field caps and far-field scaling, then bound target sizes (km).

    w = smoothstep((distance - near_distance) / (far_distance - near_distance))
    target = clip((1-w)*min(base, near_max_size) + w*far_scale*base, min, max)
    These are mesher targets, not guaranteed edge lengths or sample spacings.
    """
    t = np.clip((np.asarray(distance) - near_distance) / (far_distance - near_distance), 0., 1.)
    w = t * t * (3. - 2. * t)
    target = (1. - w) * np.minimum(base, near_max_size) + w * far_scale * np.asarray(base)
    return np.clip(target, minimumsize, np.inf if max_target_size is None else max_target_size)


class SpatialSizeControl:
    """Prepare distance geometry once; bind a fresh size field each iteration."""

    def __init__(self, data, faults, options, minimumsize, distance_traces=None):
        settings = dict(options)
        allowed = {"enabled", "near_distance", "far_distance", "near_max_size", "far_scale",
                   "max_target_size", "max_iterations"}
        unknown = sorted(set(settings) - allowed)
        if unknown:
            raise ValueError(f"spatial_size has unsupported keys: {', '.join(unknown)}")
        self.formula = {"minimumsize": float(minimumsize)}
        for key in ("near_distance", "far_distance", "near_max_size", "far_scale"):
            value = settings.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"spatial_size.{key} must be a finite number")
            self.formula[key] = float(value)
        maximum = settings.get("max_target_size")
        if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not np.isfinite(maximum)):
            raise ValueError("spatial_size.max_target_size must be a finite number or None")
        self.formula["max_target_size"] = maximum
        if not 0 <= self.formula["near_distance"] < self.formula["far_distance"]:
            raise ValueError("spatial_size requires 0 <= near_distance < far_distance")
        if not 0 < minimumsize <= self.formula["near_max_size"] or self.formula["far_scale"] < 1:
            raise ValueError("spatial_size requires near_max_size >= minimumsize > 0 and far_scale >= 1")
        if maximum is not None and maximum < self.formula["near_max_size"]:
            raise ValueError("spatial_size.max_target_size must be >= near_max_size")
        self.max_iterations = settings.get("max_iterations", 20)
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("spatial_size.max_iterations must be a positive integer")
        traces = faults if distance_traces is None else distance_traces
        try:
            self.geometry = _fault_trace_geometry(data, traces)
        except ValueError as exc:
            raise ValueError(f"spatial_size needs valid original traces in the data CRS; "
                             f"provide distance_traces (YAML trace_ids) for a mesh without traces: {exc}") from exc
        self.probes = np.column_stack((data.x, data.y)).copy()
        self.previous_probe_sizes = None
        self.report = {
            "enabled": True, "units": "km", "settings": dict(self.formula),
            "max_iterations": self.max_iterations,
            "distance_reference": "minimum distance to continuous original traces",
            "trace_ids": [str(getattr(f, "_eqtools_fault_id", getattr(f, "name", i)))
                          for i, f in enumerate(traces if isinstance(traces, (list, tuple)) else [traces])],
            "fixed_probe_count": len(self.probes), "iterations": [],
        }

    def bind(self, base_field):
        """Apply the one canonical formula at Gmsh query positions, after interpolation."""
        def target_field(x, y):
            distance = shapely.distance(self.geometry, shapely.Point(x, y))
            return float(spatial_target_sizes(base_field(x, y), distance, **self.formula))
        return target_field

    def observe(self, x, y, fitted_widths, target_field):
        """Record diagnostics; small count changes do not certify spatial convergence."""
        targets = np.array([target_field(a, b) for a, b in zip(x, y)])
        probes = np.array([target_field(a, b) for a, b in self.probes])
        change = None if self.previous_probe_sizes is None else float(
            np.max(np.abs(probes / self.previous_probe_sizes - 1.)))
        self.previous_probe_sizes = probes
        distances = shapely.distance(self.geometry, shapely.points(x, y))
        near, far = self.formula["near_distance"], self.formula["far_distance"]
        row = {
            "iteration": len(self.report["iterations"]) + 1, "cells": len(x),
            "fitted_width_min_median_max": np.quantile(fitted_widths, [0., .5, 1.]).tolist(),
            "target_at_samples_min_median_max": np.quantile(targets, [0., .5, 1.]).tolist(),
            "near_transition_far_cells": [int((distances <= near).sum()),
                                          int(((distances > near) & (distances < far)).sum()),
                                          int((distances >= far).sum())],
            "fixed_probe_max_relative_change": change,
            "used_for_remeshing": False,
        }
        self.report["iterations"].append(row)
