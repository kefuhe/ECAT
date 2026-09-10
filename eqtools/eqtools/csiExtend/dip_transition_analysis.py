"""Read-only curvature preflight for non-layered dip transition zones.

This module helps users choose transition endpoints before a Bayesian run.  It
reads one frozen reference top, evaluates planar curvature on an arc-length
grid, and exports suggestions through the existing explicit ``s_km`` endpoint
protocol.  It never mutates fault geometry, builds a mesh, or participates in
the candidate-evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Callable, Mapping, Sequence

import numpy as np

from .trace_ops import clean_trace, cumulative_distance, point_at_trace_distance, resample_trace


_LonLatConverter = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def reference_top_fingerprint(reference_top_xy) -> str:
    """Return a stable fingerprint for the ordered planar reference top."""
    values = np.asarray(reference_top_xy, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("reference_top_xy must be a 2-D array with x/y columns")
    top = np.ascontiguousarray(values[:, :2])
    digest = hashlib.sha256()
    digest.update(np.asarray(top.shape, dtype=np.int64).tobytes())
    digest.update(top.tobytes())
    return digest.hexdigest()


def _readonly_array(values, *, dtype=float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class AlongTopLocation:
    """One reported position on the ordered reference top."""

    s_km: float
    s_from_end_km: float
    fraction: float
    xy: np.ndarray
    lonlat: np.ndarray | None
    segment_index: int
    signed_curvature_per_km: float
    abs_curvature_per_km: float
    normalized_abs_curvature: float
    local_strike_deg: float

    def __post_init__(self):
        object.__setattr__(self, "xy", _readonly_array(self.xy))
        if self.lonlat is not None:
            object.__setattr__(self, "lonlat", _readonly_array(self.lonlat))

    def to_dict(self) -> dict[str, object]:
        """Return a YAML/JSON-friendly representation."""
        return {
            "s_km": float(self.s_km),
            "s_from_end_km": float(self.s_from_end_km),
            "fraction": float(self.fraction),
            "xy_km": self.xy.tolist(),
            "lonlat": None if self.lonlat is None else self.lonlat.tolist(),
            "segment_index": int(self.segment_index),
            "signed_curvature_per_km": float(self.signed_curvature_per_km),
            "abs_curvature_per_km": float(self.abs_curvature_per_km),
            "normalized_abs_curvature": float(self.normalized_abs_curvature),
            "local_strike_deg": float(self.local_strike_deg),
        }


@dataclass(frozen=True)
class DipTransitionSuggestion:
    """A curvature-derived interval expressed on one frozen reference top."""

    center: AlongTopLocation
    lower: AlongTopLocation
    upper: AlongTopLocation
    peak_prominence_per_km: float
    curvature_fraction: float | None
    curvature_threshold_per_km: float | None
    selection_mode: str
    anchor_s_km: float | None
    reference_fingerprint: str
    warnings: tuple[str, ...] = ()
    extent_method: str = "peak_fraction"
    turning_coverage: float | None = None
    turning_interval_s_km: tuple[float, float] | None = None
    total_abs_turning_rad: float | None = None

    @property
    def lower_width_km(self) -> float:
        return float(self.center.s_km - self.lower.s_km)

    @property
    def upper_width_km(self) -> float:
        return float(self.upper.s_km - self.center.s_km)

    @property
    def total_width_km(self) -> float:
        return float(self.upper.s_km - self.lower.s_km)

    @property
    def lower_chord_km(self) -> float:
        return float(np.linalg.norm(self.center.xy - self.lower.xy))

    @property
    def upper_chord_km(self) -> float:
        return float(np.linalg.norm(self.upper.xy - self.center.xy))

    def validate_reference(self, reference_top_xy) -> None:
        """Reject export when the result no longer matches the reference top."""
        if reference_top_fingerprint(reference_top_xy) != self.reference_fingerprint:
            raise ValueError(
                "transition suggestion belongs to a different reference top; "
                "rerun curvature analysis after changing or reordering top_coords"
            )

    def as_transition_zone(self, reference_top_xy) -> dict[str, object]:
        """Export through the authoritative explicit-endpoint profile protocol."""
        self.validate_reference(reference_top_xy)
        return {
            "endpoints": [
                {"s_km": float(self.lower.s_km)},
                {"s_km": float(self.upper.s_km)},
            ]
        }

    def to_dict(self) -> dict[str, object]:
        """Return a YAML/JSON-friendly diagnostic record."""
        center = self.center.to_dict()
        lower = self.lower.to_dict()
        upper = self.upper.to_dict()
        center["delta_s_from_center_km"] = 0.0
        lower["delta_s_from_center_km"] = -self.lower_width_km
        upper["delta_s_from_center_km"] = self.upper_width_km
        return {
            "selection_mode": self.selection_mode,
            "anchor_s_km": self.anchor_s_km,
            "extent_method": self.extent_method,
            "curvature_fraction": (
                None if self.curvature_fraction is None
                else float(self.curvature_fraction)
            ),
            "curvature_threshold_per_km": (
                None if self.curvature_threshold_per_km is None
                else float(self.curvature_threshold_per_km)
            ),
            "turning_coverage": (
                None if self.turning_coverage is None
                else float(self.turning_coverage)
            ),
            "turning_interval_s_km": (
                None if self.turning_interval_s_km is None
                else [float(value) for value in self.turning_interval_s_km]
            ),
            "total_abs_turning_rad": (
                None if self.total_abs_turning_rad is None
                else float(self.total_abs_turning_rad)
            ),
            "peak_prominence_per_km": float(self.peak_prominence_per_km),
            "arc_width_km": {
                "lower": self.lower_width_km,
                "upper": self.upper_width_km,
                "total": self.total_width_km,
            },
            "chord_width_km": {
                "lower": self.lower_chord_km,
                "upper": self.upper_chord_km,
            },
            "center": center,
            "lower": lower,
            "upper": upper,
            "transition_zone": {
                "endpoints": [
                    {"s_km": float(self.lower.s_km)},
                    {"s_km": float(self.upper.s_km)},
                ]
            },
            "reference_fingerprint": self.reference_fingerprint,
            "warnings": list(self.warnings),
        }

    def format_report(self) -> str:
        """Format the decision-relevant values as a compact text report."""
        rows = [
            "Dip transition suggestion",
            f"selection={self.selection_mode}  extent={self.extent_method}  "
            f"anchor_s={_format_optional(self.anchor_s_km)} km",
        ]
        if self.extent_method == "peak_fraction":
            rows.append(
                f"curvature_fraction={self.curvature_fraction:.4g}  "
                f"threshold={self.curvature_threshold_per_km:.6g}  "
                f"peak_abs_curvature={self.center.abs_curvature_per_km:.6g}  "
                f"prominence={self.peak_prominence_per_km:.6g}  [1/km]"
            )
        else:
            lower, upper = self.turning_interval_s_km
            rows.append(
                f"turning_coverage={self.turning_coverage:.4g}  "
                f"integration_interval=[{lower:.3f}, {upper:.3f}] km  "
                f"total_abs_turning={self.total_abs_turning_rad:.6g} rad"
            )
            rows.append(
                f"selected_peak_abs_curvature={self.center.abs_curvature_per_km:.6g}  "
                f"prominence={self.peak_prominence_per_km:.6g}  [1/km]"
            )
        rows.append(
            "role     s_from_START s_from_END delta_s fraction segment |kappa| strike"
        )
        for role, location in (
            ("lower", self.lower),
            ("center", self.center),
            ("upper", self.upper),
        ):
            delta_s = location.s_km - self.center.s_km
            rows.append(
                f"{role:<8} {location.s_km:8.3f} {location.s_from_end_km:8.3f} "
                f"{delta_s:8.3f} {location.fraction:9.5f} "
                f"{location.segment_index:8d} {location.abs_curvature_per_km:9.5g} "
                f"{location.local_strike_deg:8.2f}"
            )
        rows.append(
            "projected coordinates (lower / center / upper): "
            + " ; ".join(_format_location_coordinates(location) for location in (
                self.lower, self.center, self.upper,
            ))
        )
        rows.append(
            f"arc widths: lower={self.lower_width_km:.3f} km, "
            f"upper={self.upper_width_km:.3f} km, total={self.total_width_km:.3f} km"
        )
        rows.append(
            f"chord distances: lower={self.lower_chord_km:.3f} km, "
            f"upper={self.upper_chord_km:.3f} km"
        )
        rows.append(
            "transition_zones entry: "
            f"{{'endpoints': [{{'s_km': {self.lower.s_km:.6g}}}, "
            f"{{'s_km': {self.upper.s_km:.6g}}}]}}"
        )
        rows.extend(f"warning: {message}" for message in self.warnings)
        return "\n".join(rows)


@dataclass(frozen=True)
class TopCurvatureAnalysis:
    """Immutable curvature analysis tied to one ordered reference top."""

    reference_top_xy: np.ndarray
    sample_xy: np.ndarray
    sample_s_km: np.ndarray
    signed_curvature_per_km: np.ndarray
    abs_curvature_per_km: np.ndarray
    normalized_abs_curvature: np.ndarray
    local_strike_deg: np.ndarray
    peak_indices: np.ndarray
    peak_prominences_per_km: np.ndarray
    requested_spacing_km: float
    actual_spacing_km: float
    smoothing_method: str
    requested_smoothing_km: float | None
    actual_smoothing_km: float | None
    savgol_window: int | None
    polyorder: int
    min_prominence_ratio: float
    reference_fingerprint: str
    warnings: tuple[str, ...] = ()
    _xy_to_lonlat: _LonLatConverter | None = field(
        default=None, repr=False, compare=False,
    )

    def __post_init__(self):
        for name in (
            "reference_top_xy",
            "sample_xy",
            "sample_s_km",
            "signed_curvature_per_km",
            "abs_curvature_per_km",
            "normalized_abs_curvature",
            "local_strike_deg",
            "peak_indices",
            "peak_prominences_per_km",
        ):
            dtype = int if name == "peak_indices" else float
            object.__setattr__(self, name, _readonly_array(getattr(self, name), dtype=dtype))

    @property
    def length_km(self) -> float:
        return float(self.sample_s_km[-1])

    def validate_reference(self, reference_top_xy) -> None:
        """Verify that this analysis still belongs to ``reference_top_xy``."""
        if reference_top_fingerprint(reference_top_xy) != self.reference_fingerprint:
            raise ValueError(
                "curvature analysis belongs to a different reference top; "
                "rerun it after changing or reordering top_coords"
            )

    def peak_locations(self) -> tuple[AlongTopLocation, ...]:
        """Return all retained curvature peaks in along-top order."""
        return tuple(self._location(float(self.sample_s_km[index])) for index in self.peak_indices)

    def suggest_transition(
        self,
        *,
        anchor: float | Mapping[str, float] | None = None,
        center_mode: str = "nearest_peak",
        curvature_fraction: float = 0.2,
        search_radius_km: float | None = None,
        extent_method: str = "peak_fraction",
        turning_coverage: float = 0.9,
        turning_interval: Sequence[float | Mapping[str, float]] | None = None,
    ) -> DipTransitionSuggestion:
        """Suggest endpoints around one retained curvature peak.

        ``extent_method='peak_fraction'`` preserves the established rule:
        ``curvature_fraction`` is relative to the selected peak and each side
        is searched independently for the first threshold crossing.

        ``extent_method='turning_coverage'`` returns the central
        ``turning_coverage`` fraction of ``integral(abs(curvature), ds)`` over
        one explicit ``turning_interval``.  That interval must contain two
        along-top positions (numeric ``s_km`` values or ``s_km`` /
        ``s_from_end_km`` mappings).  Requiring the interval keeps adjacent
        bends from being merged by an implicit whole-trace integration.

        The two extent methods only produce read-only endpoint suggestions.
        Neither method enters candidate evaluation, and no silent fallback is
        applied when the selected rule cannot define a valid interval.
        """
        mode = str(center_mode).lower()
        if mode not in {"nearest_peak", "largest_peak"}:
            raise ValueError("center_mode must be 'nearest_peak' or 'largest_peak'")
        extent = str(extent_method).lower()
        if extent not in {"peak_fraction", "turning_coverage"}:
            raise ValueError(
                "extent_method must be 'peak_fraction' or 'turning_coverage'"
            )
        fraction = None
        coverage = None
        turning_bounds = None
        if extent == "peak_fraction":
            fraction = float(curvature_fraction)
            if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
                raise ValueError("curvature_fraction must lie strictly between 0 and 1")
            if turning_interval is not None:
                raise ValueError(
                    "turning_interval is only valid with "
                    "extent_method='turning_coverage'"
                )
        else:
            coverage = float(turning_coverage)
            if not np.isfinite(coverage) or not 0.0 < coverage < 1.0:
                raise ValueError("turning_coverage must lie strictly between 0 and 1")
            turning_bounds = _resolve_turning_interval(
                turning_interval,
                self.length_km,
            )
        if self.peak_indices.size == 0:
            raise ValueError(
                "no curvature peak passed the prominence filter; inspect the "
                "diagnostic or lower min_prominence_ratio"
            )

        anchor_s = _resolve_anchor_s(anchor, self.length_km)
        candidates = self.peak_indices
        if turning_bounds is not None:
            lower_bound, upper_bound = turning_bounds
            inside = (
                (self.sample_s_km[candidates] >= lower_bound)
                & (self.sample_s_km[candidates] <= upper_bound)
            )
            candidates = candidates[inside]
            if candidates.size == 0:
                raise ValueError(
                    "no retained curvature peak lies within turning_interval"
                )
        if search_radius_km is not None:
            if anchor_s is None:
                raise ValueError("search_radius_km requires an anchor")
            radius = float(search_radius_km)
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError("search_radius_km must be positive")
            inside = np.abs(self.sample_s_km[candidates] - anchor_s) <= radius
            candidates = candidates[inside]
            if candidates.size == 0:
                raise ValueError("no retained curvature peak lies within search_radius_km")

        if mode == "nearest_peak":
            if anchor_s is None:
                raise ValueError("center_mode='nearest_peak' requires anchor")
            peak_index = int(candidates[np.argmin(np.abs(self.sample_s_km[candidates] - anchor_s))])
        else:
            peak_index = int(candidates[np.argmax(self.abs_curvature_per_km[candidates])])

        peak_value = float(self.abs_curvature_per_km[peak_index])
        threshold = None
        total_turning = None
        if extent == "peak_fraction":
            threshold = fraction * peak_value
            lower_s = _threshold_crossing(
                self.sample_s_km,
                self.abs_curvature_per_km,
                peak_index,
                threshold,
                side="lower",
            )
            upper_s = _threshold_crossing(
                self.sample_s_km,
                self.abs_curvature_per_km,
                peak_index,
                threshold,
                side="upper",
            )
        else:
            lower_s, upper_s, total_turning = _turning_coverage_bounds(
                self.sample_s_km,
                self.abs_curvature_per_km,
                turning_bounds,
                coverage,
            )
            peak_s = float(self.sample_s_km[peak_index])
            if not lower_s < peak_s < upper_s:
                raise ValueError(
                    "the selected curvature peak is outside the central "
                    "turning-coverage interval; narrow turning_interval or "
                    "select the largest peak"
                )

        peak_position = int(np.flatnonzero(self.peak_indices == peak_index)[0])
        messages = list(self.warnings)
        if turning_bounds is not None:
            peaks_in_interval = np.count_nonzero(
                (self.sample_s_km[self.peak_indices] >= turning_bounds[0])
                & (self.sample_s_km[self.peak_indices] <= turning_bounds[1])
            )
            if peaks_in_interval > 1:
                messages.append(
                    "turning_interval contains multiple retained curvature "
                    "peaks; the coverage interval may combine distinct bends"
                )
        if lower_s <= self.actual_spacing_km or self.length_km - upper_s <= self.actual_spacing_km:
            messages.append("suggested interval approaches a reference-top endpoint")
        return DipTransitionSuggestion(
            center=self._location(float(self.sample_s_km[peak_index])),
            lower=self._location(lower_s),
            upper=self._location(upper_s),
            peak_prominence_per_km=float(self.peak_prominences_per_km[peak_position]),
            curvature_fraction=fraction,
            curvature_threshold_per_km=threshold,
            selection_mode=mode,
            anchor_s_km=anchor_s,
            reference_fingerprint=self.reference_fingerprint,
            warnings=tuple(dict.fromkeys(messages)),
            extent_method=extent,
            turning_coverage=coverage,
            turning_interval_s_km=turning_bounds,
            total_abs_turning_rad=total_turning,
        )

    def _location(self, s_km: float) -> AlongTopLocation:
        record = point_at_trace_distance(self.reference_top_xy, s_km)
        s = float(record["trace_distance_km"])
        lonlat = None
        if self._xy_to_lonlat is not None:
            xy = np.asarray(record["xy"], dtype=float)
            lon, lat = self._xy_to_lonlat(
                np.asarray([xy[0]], dtype=float),
                np.asarray([xy[1]], dtype=float),
            )
            lonlat = np.asarray([np.asarray(lon).ravel()[0], np.asarray(lat).ravel()[0]])
        return AlongTopLocation(
            s_km=s,
            s_from_end_km=self.length_km - s,
            fraction=0.0 if self.length_km == 0.0 else s / self.length_km,
            xy=np.asarray(record["xy"], dtype=float),
            lonlat=lonlat,
            segment_index=_reference_segment_index(self.reference_top_xy, s),
            signed_curvature_per_km=float(np.interp(s, self.sample_s_km, self.signed_curvature_per_km)),
            abs_curvature_per_km=float(np.interp(s, self.sample_s_km, self.abs_curvature_per_km)),
            normalized_abs_curvature=float(np.interp(s, self.sample_s_km, self.normalized_abs_curvature)),
            local_strike_deg=float(_interpolate_angle(s, self.sample_s_km, self.local_strike_deg)),
        )

    def to_dict(self, *, include_samples: bool = False) -> dict[str, object]:
        """Return settings and peak diagnostics; samples are optional."""
        peaks = []
        for index, prominence, location in zip(
            self.peak_indices,
            self.peak_prominences_per_km,
            self.peak_locations(),
        ):
            item = location.to_dict()
            item["sample_index"] = int(index)
            item["prominence_per_km"] = float(prominence)
            peaks.append(item)
        result: dict[str, object] = {
            "reference_length_km": self.length_km,
            "reference_fingerprint": self.reference_fingerprint,
            "settings": {
                "requested_spacing_km": self.requested_spacing_km,
                "actual_spacing_km": self.actual_spacing_km,
                "smoothing_method": self.smoothing_method,
                "requested_smoothing_km": self.requested_smoothing_km,
                "actual_smoothing_km": self.actual_smoothing_km,
                "savgol_window": self.savgol_window,
                "polyorder": self.polyorder,
                "min_prominence_ratio": self.min_prominence_ratio,
            },
            "peaks": peaks,
            "warnings": list(self.warnings),
        }
        if include_samples:
            result["samples"] = {
                "s_km": self.sample_s_km.tolist(),
                "xy_km": self.sample_xy.tolist(),
                "signed_curvature_per_km": self.signed_curvature_per_km.tolist(),
                "abs_curvature_per_km": self.abs_curvature_per_km.tolist(),
                "normalized_abs_curvature": self.normalized_abs_curvature.tolist(),
                "local_strike_deg": self.local_strike_deg.tolist(),
            }
        return result

    def format_report(self) -> str:
        """Format analysis settings and candidate peaks for setup review."""
        rows = [
            "Reference-top curvature preflight",
            (
                f"length={self.length_km:.3f} km  samples={self.sample_s_km.size}  "
                f"spacing(requested/actual)={self.requested_spacing_km:.3g}/"
                f"{self.actual_spacing_km:.3g} km"
            ),
            (
                f"smoothing={self.smoothing_method}  "
                f"window={_format_optional(self.actual_smoothing_km)} km  "
                f"polyorder={self.polyorder}  "
                f"min_prominence_ratio={self.min_prominence_ratio:.4g}"
            ),
            f"retained peaks={self.peak_indices.size}",
            "rank  s_start   s_end    abs_kappa   prominence  lon         lat",
        ]
        order = np.argsort(self.abs_curvature_per_km[self.peak_indices])[::-1]
        locations = self.peak_locations()
        for rank, position in enumerate(order, start=1):
            location = locations[int(position)]
            prominence = self.peak_prominences_per_km[int(position)]
            if location.lonlat is None:
                lon_text = lat_text = "--"
            else:
                lon_text = f"{location.lonlat[0]:.6f}"
                lat_text = f"{location.lonlat[1]:.6f}"
            rows.append(
                f"{rank:>4} {location.s_km:8.3f} {location.s_from_end_km:8.3f} "
                f"{location.abs_curvature_per_km:10.6g} {prominence:11.6g} "
                f"{lon_text:>11} {lat_text:>11}"
            )
        rows.extend(f"warning: {message}" for message in self.warnings)
        return "\n".join(rows)


def analyze_top_curvature(
    reference_top_xy,
    *,
    spacing_km: float = 0.5,
    smoothing_method: str = "savgol",
    smoothing_km: float | None = 2.0,
    polyorder: int = 3,
    min_prominence_ratio: float = 0.05,
    xy_to_lonlat: _LonLatConverter | None = None,
) -> TopCurvatureAnalysis:
    """Analyze signed planar curvature on a uniform reference-top arc grid.

    Curvature uses derivatives of smoothed ``x(s)`` and ``y(s)``:

    ``kappa = (x' * y'' - y' * x'') / (x'^2 + y'^2)^(3/2)``.

    The Savitzky-Golay window is specified in kilometres and converted once to
    an odd sample count.  ``min_prominence_ratio`` only filters candidate peak
    noise; transition endpoints use an independently reported fraction of the
    selected peak.
    """
    raw_top = np.asarray(reference_top_xy, dtype=float)
    if raw_top.ndim != 2 or raw_top.shape[1] < 2:
        raise ValueError("reference_top_xy must be a 2-D array with x/y columns")
    if not np.all(np.isfinite(raw_top[:, :2])):
        raise ValueError("reference_top_xy contains NaN or infinite values")
    fingerprint = reference_top_fingerprint(raw_top)
    top = clean_trace(raw_top[:, :2])
    length = float(cumulative_distance(top)[-1])
    spacing = float(spacing_km)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing_km must be positive")
    sample_count = max(5, int(np.ceil(length / spacing)) + 1)
    sample_xy = resample_trace(top, num_points=sample_count)[:, :2]
    sample_s = np.linspace(0.0, length, sample_count)
    actual_spacing = float(sample_s[1] - sample_s[0])

    method = str(smoothing_method).lower()
    order = int(polyorder)
    if order < 2:
        raise ValueError("polyorder must be at least 2 for curvature")
    warnings_list: list[str] = []
    window = None
    actual_smoothing = None
    if method == "savgol":
        if smoothing_km is None:
            raise ValueError("smoothing_km is required for savgol smoothing")
        requested_smoothing = float(smoothing_km)
        if not np.isfinite(requested_smoothing) or requested_smoothing <= 0.0:
            raise ValueError("smoothing_km must be positive")
        window = int(np.rint(requested_smoothing / actual_spacing)) + 1
        minimum_window = order + 2
        if minimum_window % 2 == 0:
            minimum_window += 1
        window = max(window, minimum_window)
        if window % 2 == 0:
            window += 1
        largest_window = sample_count if sample_count % 2 == 1 else sample_count - 1
        if largest_window <= order:
            raise ValueError("reference top is too short for the requested Savitzky-Golay order")
        if window > largest_window:
            window = largest_window
            warnings_list.append("smoothing window was clipped by the reference-top sample count")
        actual_smoothing = float((window - 1) * actual_spacing)
        from scipy.signal import savgol_filter

        smooth_xy = np.column_stack([
            savgol_filter(sample_xy[:, column], window, order, mode="interp")
            for column in range(2)
        ])
        first = np.column_stack([
            savgol_filter(
                sample_xy[:, column], window, order, deriv=1,
                delta=actual_spacing, mode="interp",
            )
            for column in range(2)
        ])
        second = np.column_stack([
            savgol_filter(
                sample_xy[:, column], window, order, deriv=2,
                delta=actual_spacing, mode="interp",
            )
            for column in range(2)
        ])
    elif method == "none":
        if smoothing_km is not None:
            warnings_list.append("smoothing_km is ignored when smoothing_method='none'")
        requested_smoothing = smoothing_km
        smooth_xy = sample_xy.copy()
        first = np.column_stack([
            np.gradient(smooth_xy[:, column], sample_s, edge_order=2)
            for column in range(2)
        ])
        second = np.column_stack([
            np.gradient(first[:, column], sample_s, edge_order=2)
            for column in range(2)
        ])
    else:
        raise ValueError("smoothing_method must be 'savgol' or 'none'")

    speed2 = np.sum(first**2, axis=1)
    speed_tolerance = 128.0 * np.finfo(float).eps * max(1.0, float(np.max(speed2)))
    if np.any(speed2 <= speed_tolerance):
        raise ValueError(
            "smoothed reference top has a zero or numerically unstable tangent; "
            "adjust spacing/smoothing or repair the top"
        )
    signed_curvature = (
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    ) / np.power(speed2, 1.5)
    abs_curvature = np.abs(signed_curvature)
    maximum = float(np.max(abs_curvature))
    normalized = np.zeros_like(abs_curvature) if maximum == 0.0 else abs_curvature / maximum
    strike = np.mod(np.degrees(np.arctan2(first[:, 0], first[:, 1])), 360.0)

    prominence_ratio = float(min_prominence_ratio)
    if not np.isfinite(prominence_ratio) or not 0.0 <= prominence_ratio < 1.0:
        raise ValueError("min_prominence_ratio must lie in [0, 1)")
    from scipy.signal import find_peaks, peak_prominences

    if maximum == 0.0:
        peak_indices = np.empty(0, dtype=int)
        prominences = np.empty(0, dtype=float)
        warnings_list.append("reference top is straight at the analyzed scale")
    else:
        peak_indices, properties = find_peaks(
            abs_curvature,
            prominence=maximum * prominence_ratio,
        )
        if "prominences" in properties:
            prominences = np.asarray(properties["prominences"], dtype=float)
        else:
            prominences = np.asarray(
                peak_prominences(abs_curvature, peak_indices)[0], dtype=float,
            )
        if peak_indices.size == 0:
            warnings_list.append("no interior curvature peak passed the prominence filter")

    return TopCurvatureAnalysis(
        reference_top_xy=raw_top[:, :2],
        sample_xy=sample_xy,
        sample_s_km=sample_s,
        signed_curvature_per_km=signed_curvature,
        abs_curvature_per_km=abs_curvature,
        normalized_abs_curvature=normalized,
        local_strike_deg=strike,
        peak_indices=peak_indices,
        peak_prominences_per_km=prominences,
        requested_spacing_km=spacing,
        actual_spacing_km=actual_spacing,
        smoothing_method=method,
        requested_smoothing_km=None if smoothing_km is None else float(smoothing_km),
        actual_smoothing_km=actual_smoothing,
        savgol_window=window,
        polyorder=order,
        min_prominence_ratio=prominence_ratio,
        reference_fingerprint=fingerprint,
        warnings=tuple(warnings_list),
        _xy_to_lonlat=xy_to_lonlat,
    )


def _resolve_anchor_s(anchor, length_km: float) -> float | None:
    if anchor is None:
        return None
    if isinstance(anchor, Mapping):
        present = [key for key in ("s_km", "s_from_end_km") if key in anchor]
        if len(present) != 1:
            raise ValueError("anchor mapping must contain exactly one of s_km or s_from_end_km")
        value = float(anchor[present[0]])
        s = value if present[0] == "s_km" else length_km - value
    else:
        s = float(anchor)
    if not np.isfinite(s) or s < 0.0 or s > length_km:
        raise ValueError("anchor must lie within the reference-top arc length")
    return s


def _resolve_turning_interval(
    interval: Sequence[float | Mapping[str, float]] | None,
    length_km: float,
) -> tuple[float, float]:
    """Resolve the explicit integration domain for turning coverage."""
    if interval is None:
        raise ValueError(
            "extent_method='turning_coverage' requires turning_interval with "
            "two along-top positions"
        )
    if isinstance(interval, Mapping):
        raise ValueError("turning_interval must contain two along-top positions")
    try:
        endpoints = list(interval)
    except TypeError as exc:
        raise ValueError(
            "turning_interval must contain two along-top positions"
        ) from exc
    if len(endpoints) != 2:
        raise ValueError("turning_interval must contain two along-top positions")
    first = _resolve_anchor_s(endpoints[0], length_km)
    second = _resolve_anchor_s(endpoints[1], length_km)
    if first is None or second is None:
        raise ValueError("turning_interval positions cannot be None")
    lower, upper = sorted((first, second))
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, length_km)
    if upper - lower <= tolerance:
        raise ValueError("turning_interval must have positive along-top length")
    return float(lower), float(upper)


def _turning_coverage_bounds(s, values, interval, coverage):
    """Return central absolute-turning quantiles on an explicit interval.

    Absolute curvature is linearly interpolated at interval boundaries and
    integrated by the trapezoidal rule.  Quantile locations invert the same
    piecewise-linear curvature model, so the reported interval captures the
    requested fraction without replacing the analysis grid or its smoothing.
    """
    lower, upper = interval
    interior = (s > lower) & (s < upper)
    domain_s = np.concatenate(([lower], np.asarray(s)[interior], [upper]))
    domain_values = np.interp(domain_s, s, values)
    widths = np.diff(domain_s)
    increments = 0.5 * (
        domain_values[:-1] + domain_values[1:]
    ) * widths
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    total = float(cumulative[-1])
    tolerance = 128.0 * np.finfo(float).eps * max(
        1.0,
        float(np.max(domain_values)) * float(upper - lower),
    )
    if not np.isfinite(total) or total <= tolerance:
        raise ValueError(
            "turning_interval contains no resolvable absolute curvature; "
            "adjust the interval or curvature-analysis scale"
        )
    tail = 0.5 * (1.0 - coverage)
    lower_s = _piecewise_linear_density_quantile(
        domain_s,
        domain_values,
        cumulative,
        tail * total,
    )
    upper_s = _piecewise_linear_density_quantile(
        domain_s,
        domain_values,
        cumulative,
        (1.0 - tail) * total,
    )
    return lower_s, upper_s, total


def _piecewise_linear_density_quantile(s, values, cumulative, target):
    """Invert an integrated non-negative piecewise-linear density."""
    index = int(np.searchsorted(cumulative, target, side="right") - 1)
    index = max(0, min(index, len(s) - 2))
    segment_target = float(target - cumulative[index])
    width = float(s[index + 1] - s[index])
    start_value = float(values[index])
    delta = float(values[index + 1] - start_value)
    segment_total = width * (start_value + 0.5 * delta)
    if segment_total <= 0.0:
        return float(s[index])

    # Bisection is deterministic and avoids cancellation when the density is
    # nearly constant. The polynomial is monotone because curvature magnitude
    # is non-negative at both segment ends.
    lower_fraction, upper_fraction = 0.0, 1.0
    for _ in range(53):
        fraction = 0.5 * (lower_fraction + upper_fraction)
        area = width * (
            start_value * fraction + 0.5 * delta * fraction**2
        )
        if area < segment_target:
            lower_fraction = fraction
        else:
            upper_fraction = fraction
    fraction = 0.5 * (lower_fraction + upper_fraction)
    return float(s[index] + width * fraction)


def _threshold_crossing(s, values, peak_index: int, threshold: float, *, side: str) -> float:
    if side == "lower":
        for lower in range(peak_index - 1, -1, -1):
            if values[lower] <= threshold:
                return _linear_crossing(
                    s[lower], values[lower], s[lower + 1], values[lower + 1], threshold,
                )
    else:
        for upper in range(peak_index + 1, len(values)):
            if values[upper] <= threshold:
                return _linear_crossing(
                    s[upper - 1], values[upper - 1], s[upper], values[upper], threshold,
                )
    raise ValueError(
        f"absolute curvature does not cross the requested threshold on the {side} side; "
        "use a higher curvature_fraction, a different peak, or inspect the reference-top endpoint"
    )


def _linear_crossing(s0, value0, s1, value1, threshold) -> float:
    delta = float(value1 - value0)
    if abs(delta) <= np.finfo(float).eps * max(1.0, abs(value0), abs(value1)):
        return float(0.5 * (s0 + s1))
    fraction = float(np.clip((threshold - value0) / delta, 0.0, 1.0))
    return float(s0 + fraction * (s1 - s0))


def _reference_segment_index(reference_top_xy, s_km: float) -> int:
    """Locate ``s_km`` in the original ordered top, retaining its indices."""
    xy = np.asarray(reference_top_xy, dtype=float)[:, :2]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(step)]
    if np.isclose(s_km, cumulative[-1], rtol=0.0, atol=1e-10):
        index = len(step) - 1
    else:
        index = int(np.searchsorted(cumulative, s_km, side="right") - 1)
        index = max(0, min(index, len(step) - 1))
    if step[index] > 0.0:
        return index
    nonzero = np.flatnonzero(step > 0.0)
    if nonzero.size == 0:
        raise ValueError("reference top contains no non-zero segment")
    return int(nonzero[np.argmin(np.abs(cumulative[nonzero] - s_km))])


def _interpolate_angle(value, xp, angles_deg) -> float:
    radians = np.unwrap(np.radians(angles_deg))
    return float(np.mod(np.degrees(np.interp(value, xp, radians)), 360.0))


def _format_optional(value) -> str:
    return "--" if value is None else f"{float(value):.4g}"


def _format_location_coordinates(location: AlongTopLocation) -> str:
    xy = f"xy=({location.xy[0]:.3f}, {location.xy[1]:.3f}) km"
    if location.lonlat is None:
        return xy
    return (
        f"{xy}, lonlat=({location.lonlat[0]:.6f}, "
        f"{location.lonlat[1]:.6f})"
    )


__all__ = [
    "AlongTopLocation",
    "DipTransitionSuggestion",
    "TopCurvatureAnalysis",
    "analyze_top_curvature",
    "reference_top_fingerprint",
]
