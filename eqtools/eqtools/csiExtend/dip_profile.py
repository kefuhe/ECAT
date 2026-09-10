"""Immutable along-strike dip profiles and their geometric resolution.

This module owns the scientific contract between user-declared dip controls
and the one-dimensional profile consumed by the perturbation pipeline.  All
position-like inputs are first projected onto the reference top-edge polyline.
Only then are they represented by one resolved coordinate ``u``:

``x``
    Projected easting in the fault-local kilometre frame.
``y``
    Projected northing in the fault-local kilometre frame.
``arc_length``
    Cumulative distance in kilometres along the ordered top edge.

The frozen specification contains no candidate values.  A resolved profile is
created independently for every candidate, so Bayesian proposals remain
non-cumulative and sample-to-control indexing never depends on spatial sort
order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _freeze_array(values, *, dtype=None):
    array = np.asarray(values, dtype=dtype).copy()
    array.flags.writeable = False
    return array


def _as_control_matrix(values, *, name):
    matrix = np.asarray(values, dtype=float)
    if matrix.size == 0:
        return np.empty((0, 3), dtype=float)
    matrix = np.atleast_2d(matrix)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3): coordinate-1, coordinate-2, dip")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


_ARC_POSITION_KEYS = frozenset({"s_km", "s_from_end_km"})


def _transform_xy_point(point, transform):
    """Transform one fault-local x/y point into the declaration frame."""
    point = np.asarray(point, dtype=float)
    if transform is None:
        return point
    first, second = transform(float(point[0]), float(point[1]))
    transformed = np.asarray([
        np.asarray(first, dtype=float).reshape(-1)[0],
        np.asarray(second, dtype=float).reshape(-1)[0],
    ])
    if not np.all(np.isfinite(transformed)):
        raise ValueError("dip-profile coordinate transform returned non-finite values")
    return transformed


def _materialize_profile_position(
    position,
    *,
    reference_top_xy,
    xy_to_declaration_frame,
    name,
):
    """Return one position as two coordinates in the declaration frame.

    Coordinate pairs pass through unchanged.  A mapping with exactly one of
    ``s_km`` or ``s_from_end_km`` is first located on the frozen reference top
    edge, whose ordering defines start and end, and is then converted to the
    same frame as coordinate-pair declarations.  Keeping this conversion at
    the setup boundary lets controls and transitions share the existing
    projection/interpolation implementation without a second candidate path.
    """
    if not isinstance(position, Mapping):
        coordinates = np.asarray(position, dtype=float)
        if coordinates.shape != (2,) or not np.all(np.isfinite(coordinates)):
            raise ValueError(f"{name} must be a finite coordinate pair")
        return coordinates

    unknown = set(position) - _ARC_POSITION_KEYS
    selected = set(position) & _ARC_POSITION_KEYS
    if unknown or len(selected) != 1:
        raise ValueError(
            f"{name} must contain exactly one of 's_km' or "
            "'s_from_end_km'"
        )
    key = next(iter(selected))
    value = position[key]
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name}.{key} must be a non-negative distance in km")
    distance = float(value)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError(f"{name}.{key} must be a finite non-negative distance in km")

    if reference_top_xy is None:
        raise ValueError(
            f"{name} uses along-top distance but no frozen reference top is available"
        )
    top = np.asarray(reference_top_xy, dtype=float)
    if top.ndim != 2 or top.shape[0] < 2 or top.shape[1] < 2:
        raise ValueError("reference_top_xy must have shape (n>=2, 2+)")
    if not np.all(np.isfinite(top[:, :2])):
        raise ValueError("reference_top_xy must contain only finite coordinates")
    top_xy = top[:, :2]
    top_s, _ = _polyline_arclength(top_xy)
    total_length = float(top_s[-1])
    tolerance = _coordinate_tolerance(top_s)
    if distance > total_length + tolerance:
        raise ValueError(
            f"{name}.{key}={distance:g} km exceeds reference top length "
            f"{total_length:g} km"
        )
    distance = float(np.clip(distance, 0.0, total_length))
    absolute_s = distance if key == "s_km" else total_length - distance
    point_xy = _point_at_s(absolute_s, top_xy, top_s)
    return _transform_xy_point(point_xy, xy_to_declaration_frame)


def _materialize_control_group(
    values,
    *,
    name,
    reference_top_xy,
    xy_to_declaration_frame,
):
    """Normalize coordinate rows and along-top control mappings alike."""
    if values is None:
        return np.empty((0, 3), dtype=float)
    if isinstance(values, Mapping):
        items = [values]
    else:
        try:
            matrix = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            matrix = None
        if matrix is not None:
            if matrix.size == 0:
                return np.empty((0, 3), dtype=float)
            matrix = np.atleast_2d(matrix)
            if matrix.ndim == 2 and matrix.shape[1] == 3:
                if not np.all(np.isfinite(matrix)):
                    raise ValueError(f"{name} must contain only finite values")
                return matrix
        try:
            items = list(values)
        except TypeError as exc:
            raise TypeError(f"{name} must be a control row or sequence of rows") from exc

    rows = []
    for index, item in enumerate(items):
        item_name = f"{name}[{index}]"
        if isinstance(item, Mapping):
            unknown = set(item) - (_ARC_POSITION_KEYS | {"dip"})
            selected = set(item) & _ARC_POSITION_KEYS
            if unknown or set(item) != selected | {"dip"} or len(selected) != 1:
                raise ValueError(
                    f"{item_name} must contain 'dip' and exactly one of "
                    "'s_km' or 's_from_end_km'"
                )
            key = next(iter(selected))
            coordinates = _materialize_profile_position(
                {key: item[key]},
                reference_top_xy=reference_top_xy,
                xy_to_declaration_frame=xy_to_declaration_frame,
                name=item_name,
            )
            dip = item["dip"]
        else:
            row = np.asarray(item, dtype=float)
            if row.shape != (3,) or not np.all(np.isfinite(row)):
                raise ValueError(
                    f"{item_name} must be [coordinate_1, coordinate_2, dip] "
                    "or an along-top control mapping"
                )
            coordinates = row[:2]
            dip = row[2]
        if isinstance(dip, (bool, np.bool_)):
            raise TypeError(f"{item_name}.dip must be a finite angle")
        dip = float(dip)
        if not np.isfinite(dip):
            raise ValueError(f"{item_name}.dip must be a finite angle")
        rows.append([coordinates[0], coordinates[1], dip])
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def _materialize_transition_zone(
    zone,
    *,
    reference_top_xy,
    xy_to_declaration_frame,
    index,
):
    """Resolve all transition position fields through the shared protocol."""
    if isinstance(zone, DipTransitionZone):
        return zone
    if not isinstance(zone, Mapping):
        raise TypeError("each transition_zones entry must be a mapping")
    materialized = dict(zone)
    if "center" in zone:
        materialized["center"] = _materialize_profile_position(
            zone["center"],
            reference_top_xy=reference_top_xy,
            xy_to_declaration_frame=xy_to_declaration_frame,
            name=f"transition_zones[{index}].center",
        )
    if "endpoints" in zone:
        endpoints = zone["endpoints"]
        if isinstance(endpoints, Mapping):
            raise ValueError(
                f"transition_zones[{index}].endpoints must contain two positions"
            )
        try:
            endpoints = list(endpoints)
        except TypeError as exc:
            raise ValueError(
                f"transition_zones[{index}].endpoints must contain two positions"
            ) from exc
        if len(endpoints) != 2:
            raise ValueError(
                f"transition_zones[{index}].endpoints must contain two positions"
            )
        materialized["endpoints"] = [
            _materialize_profile_position(
                endpoint,
                reference_top_xy=reference_top_xy,
                xy_to_declaration_frame=xy_to_declaration_frame,
                name=f"transition_zones[{index}].endpoints[{endpoint_index}]",
            )
            for endpoint_index, endpoint in enumerate(endpoints)
        ]
    return DipTransitionZone.from_mapping(materialized)


@dataclass(frozen=True)
class DipControlPoints:
    """Immutable dip controls with explicit sampled/fixed roles.

    ``sampled`` is a Boolean array aligned with ``x``, ``y`` and ``dip``.
    Bayesian values map to ``np.flatnonzero(sampled)`` in declaration order;
    projecting or sorting the controls never changes that mapping.
    """

    x: np.ndarray
    y: np.ndarray
    dip: np.ndarray
    sampled: np.ndarray | None = None

    def __post_init__(self):
        from .fault_angle_conventions import normalize_oriented_reference_dip

        x = np.atleast_1d(np.asarray(self.x, dtype=float))
        y = np.atleast_1d(np.asarray(self.y, dtype=float))
        dip = np.atleast_1d(np.asarray(self.dip, dtype=float))
        if x.ndim != 1 or y.ndim != 1 or dip.ndim != 1:
            raise ValueError("DipControlPoints x, y and dip must be 1-D arrays")
        if len(x) == 0 or len(x) != len(y) or len(x) != len(dip):
            raise ValueError(
                "DipControlPoints x, y and dip must have the same non-zero length"
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("DipControlPoints coordinates must be finite")
        dip = normalize_oriented_reference_dip(
            dip,
            name="DipControlPoints.dip",
        )

        if self.sampled is None:
            sampled = np.ones(len(x), dtype=bool)
        else:
            sampled = np.asarray(self.sampled)
            if sampled.ndim != 1 or len(sampled) != len(x):
                raise ValueError(
                    "DipControlPoints.sampled must be a 1-D Boolean array "
                    "aligned with x, y and dip"
                )
            if not np.issubdtype(sampled.dtype, np.bool_):
                raise TypeError("DipControlPoints.sampled must contain Boolean values")
            sampled = sampled.astype(bool, copy=False)

        object.__setattr__(self, "x", _freeze_array(x, dtype=float))
        object.__setattr__(self, "y", _freeze_array(y, dtype=float))
        object.__setattr__(self, "dip", _freeze_array(dip, dtype=float))
        object.__setattr__(self, "sampled", _freeze_array(sampled, dtype=bool))

    @property
    def sample_to_control(self):
        """Read-only control indices addressed by the candidate sample vector."""
        return _freeze_array(np.flatnonzero(self.sampled), dtype=int)

    @property
    def sampled_count(self):
        return int(np.count_nonzero(self.sampled))

    @classmethod
    def from_groups(
        cls,
        sampled_controls,
        fixed_controls=None,
        *,
        is_utm=False,
        xy2ll_func=None,
    ):
        """Build one control set from explicit sampled and fixed groups.

        Each group has columns ``[lon, lat, dip]`` by default or
        ``[x_km, y_km, dip]`` when ``is_utm=True``.  Sampled controls are kept
        in their declared order and occupy the first sample mapping entries;
        spatial ordering is resolved later from projected positions.
        """
        sampled_matrix = _as_control_matrix(sampled_controls, name="sampled_controls")
        fixed_matrix = _as_control_matrix(
            [] if fixed_controls is None else fixed_controls,
            name="fixed_controls",
        )
        if len(sampled_matrix) + len(fixed_matrix) == 0:
            raise ValueError("at least one sampled or fixed dip control is required")

        controls = np.vstack([sampled_matrix, fixed_matrix])
        if is_utm:
            if xy2ll_func is None:
                raise ValueError("xy2ll_func is required when is_utm=True")
            lon, lat = xy2ll_func(controls[:, 0], controls[:, 1])
        else:
            lon, lat = controls[:, 0], controls[:, 1]
        sampled = np.concatenate([
            np.ones(len(sampled_matrix), dtype=bool),
            np.zeros(len(fixed_matrix), dtype=bool),
        ])
        return cls(x=lon, y=lat, dip=controls[:, 2], sampled=sampled)


@dataclass(frozen=True)
class DipTransitionZone:
    """One plateau-to-plateau transition declared in profile coordinates.

    Exactly one form is accepted:

    - ``center`` plus ``lower_width``/``upper_width``;
    - two explicit ``endpoints``.

    Coordinates use the same frame as the containing :class:`DipProfileSpec`.
    ``metric='axis'`` measures the widths in the resolved profile coordinate;
    ``metric='euclidean'`` finds the first circle intersection on either side
    of the projected centre while remaining on the same top-edge branch.
    ``shape='linear'`` preserves the established piecewise-linear transition;
    ``shape='smoothstep'`` uses the cubic Hermite blend ``3t**2 - 2t**3`` so
    the transition joins both plateaus with zero endpoint slope.
    """

    center: tuple[float, float] | None = None
    lower_width: float | None = None
    upper_width: float | None = None
    endpoints: tuple[tuple[float, float], tuple[float, float]] | None = None
    metric: str = "axis"
    shape: str = "linear"

    def __post_init__(self):
        has_center = self.center is not None
        has_endpoints = self.endpoints is not None
        if has_center == has_endpoints:
            raise ValueError(
                "a transition zone requires exactly one of center or endpoints"
            )
        if self.metric not in {"axis", "euclidean"}:
            raise ValueError("transition metric must be 'axis' or 'euclidean'")
        if not isinstance(self.shape, str):
            raise TypeError("transition shape must be 'linear' or 'smoothstep'")
        shape = self.shape.lower()
        if shape not in {"linear", "smoothstep"}:
            raise ValueError("transition shape must be 'linear' or 'smoothstep'")
        object.__setattr__(self, "shape", shape)

        if has_center:
            center = tuple(np.asarray(self.center, dtype=float).reshape(-1))
            if len(center) != 2 or not np.all(np.isfinite(center)):
                raise ValueError("transition center must contain two finite coordinates")
            lower = float(self.lower_width)
            upper = float(self.upper_width)
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError("transition half-widths must be finite")
            if lower <= 0.0 or upper <= 0.0:
                raise ValueError("transition half-widths must be positive")
            object.__setattr__(self, "center", center)
        else:
            endpoints = np.asarray(self.endpoints, dtype=float)
            if endpoints.shape != (2, 2) or not np.all(np.isfinite(endpoints)):
                raise ValueError("transition endpoints must have shape (2, 2)")
            if self.lower_width is not None or self.upper_width is not None:
                raise ValueError("explicit transition endpoints cannot also define widths")
            if self.metric != "axis":
                raise ValueError("explicit transition endpoints use the resolved axis directly")
            object.__setattr__(
                self,
                "endpoints",
                (tuple(endpoints[0]), tuple(endpoints[1])),
            )

    @classmethod
    def from_mapping(cls, config: Mapping):
        """Parse the concise public transition-zone dictionary."""
        if not isinstance(config, Mapping):
            raise TypeError("each transition_zones entry must be a mapping")
        unknown = set(config) - {
            "center", "half_width", "endpoints", "metric", "shape",
        }
        if unknown:
            raise ValueError(
                "unknown transition-zone field(s): " + ", ".join(sorted(unknown))
            )
        if "endpoints" in config:
            if "center" in config or "half_width" in config:
                raise ValueError(
                    "transition endpoints are mutually exclusive with center/half_width"
                )
            return cls(
                endpoints=config["endpoints"],
                metric=config.get("metric", "axis"),
                shape=config.get("shape", "linear"),
            )

        if "center" not in config or "half_width" not in config:
            raise ValueError(
                "a center transition requires both center and half_width"
            )
        width = config["half_width"]
        if isinstance(width, Mapping):
            unknown_width = set(width) - {"lower", "upper"}
            if unknown_width or set(width) != {"lower", "upper"}:
                raise ValueError(
                    "asymmetric half_width must contain exactly lower and upper"
                )
            lower, upper = width["lower"], width["upper"]
        else:
            lower = upper = width
        return cls(
            center=config["center"],
            lower_width=lower,
            upper_width=upper,
            metric=config.get("metric", "axis"),
            shape=config.get("shape", "linear"),
        )


@dataclass(frozen=True)
class DipProfileSpec:
    """Frozen along-strike dip-profile declaration.

    The reference object stores this entire specification as one source of
    truth.  Candidate methods therefore do not receive a second copy of the
    interpolation axis, fixed-control selection, or transition geometry.
    """

    controls: DipControlPoints
    interpolation_axis: str = "auto"
    transition_zones: tuple[DipTransitionZone, ...] = ()

    def __post_init__(self):
        if not isinstance(self.controls, DipControlPoints):
            raise TypeError("DipProfileSpec.controls must be DipControlPoints")
        if self.interpolation_axis not in {"auto", "x", "y", "arc_length"}:
            raise ValueError(
                "interpolation_axis must be 'auto', 'x', 'y', or 'arc_length'"
            )
        zones = tuple(
            zone if isinstance(zone, DipTransitionZone)
            else DipTransitionZone.from_mapping(zone)
            for zone in self.transition_zones
        )
        object.__setattr__(self, "transition_zones", zones)


def build_dip_profile_spec(
    sampled_controls,
    fixed_controls=None,
    *,
    interpolation_axis="auto",
    transition_zones=None,
    reference_top_xy=None,
    xy_to_declaration_frame=None,
):
    """Build one profile from coordinate and/or along-top declarations.

    Existing control rows remain ``[coordinate_1, coordinate_2, dip]``.
    Individual controls may instead be mappings with ``dip`` plus exactly one
    of ``s_km`` (distance from the ordered reference-top start) or
    ``s_from_end_km`` (distance back from its end).  Transition centres and
    endpoints accept the same position mappings without ``dip``.

    Along-top declarations are materialized on ``reference_top_xy`` exactly
    once at this setup boundary.  ``xy_to_declaration_frame`` converts that
    fault-local x/y anchor to the coordinate frame used by ordinary rows.  The
    returned :class:`DipProfileSpec` therefore enters the existing candidate
    resolver without a special interpolation, sampling, or cache path.
    """
    sampled_matrix = _materialize_control_group(
        sampled_controls,
        name="sampled_controls",
        reference_top_xy=reference_top_xy,
        xy_to_declaration_frame=xy_to_declaration_frame,
    )
    fixed_matrix = _materialize_control_group(
        fixed_controls,
        name="fixed_controls",
        reference_top_xy=reference_top_xy,
        xy_to_declaration_frame=xy_to_declaration_frame,
    )
    controls = DipControlPoints.from_groups(sampled_matrix, fixed_matrix)
    zone_declarations = () if transition_zones is None else transition_zones
    zones = tuple(
        _materialize_transition_zone(
            zone,
            reference_top_xy=reference_top_xy,
            xy_to_declaration_frame=xy_to_declaration_frame,
            index=index,
        )
        for index, zone in enumerate(zone_declarations)
    )
    return DipProfileSpec(
        controls=controls,
        interpolation_axis=interpolation_axis,
        transition_zones=zones,
    )


@dataclass(frozen=True)
class ResolvedTransitionZone:
    """A transition interval after projection onto the candidate top edge."""

    lower_u: float
    upper_u: float
    lower_s: float
    upper_s: float
    lower_xy: np.ndarray
    upper_xy: np.ndarray
    lower_control: int
    upper_control: int
    metric: str
    shape: str
    source: str
    raw_anchor_xy: np.ndarray
    projected_anchor_xy: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "lower_xy", _freeze_array(self.lower_xy, dtype=float))
        object.__setattr__(self, "upper_xy", _freeze_array(self.upper_xy, dtype=float))
        object.__setattr__(
            self,
            "raw_anchor_xy",
            _freeze_array(self.raw_anchor_xy, dtype=float),
        )
        object.__setattr__(
            self,
            "projected_anchor_xy",
            _freeze_array(self.projected_anchor_xy, dtype=float),
        )


@dataclass(frozen=True)
class ResolvedDipProfile:
    """Read-only result shared by the candidate stage and diagnostics."""

    interpolation_axis: str
    raw_control_xy: np.ndarray
    projected_control_xy: np.ndarray
    projection_distance: np.ndarray
    control_s: np.ndarray
    control_u: np.ndarray
    control_dip: np.ndarray
    sampled: np.ndarray
    sample_to_control: np.ndarray
    top_xy: np.ndarray
    top_s: np.ndarray
    top_u: np.ndarray
    top_dip_continuous: np.ndarray
    interpolation_u: np.ndarray
    interpolation_dip: np.ndarray
    transition_zones: tuple[ResolvedTransitionZone, ...]

    def __post_init__(self):
        for name in (
            "raw_control_xy", "projected_control_xy", "projection_distance",
            "control_s", "control_u", "control_dip", "sampled",
            "sample_to_control", "top_xy", "top_s", "top_u",
            "top_dip_continuous", "interpolation_u", "interpolation_dip",
        ):
            values = getattr(self, name)
            dtype = bool if name == "sampled" else int if name == "sample_to_control" else float
            object.__setattr__(self, name, _freeze_array(values, dtype=dtype))


def transform_dip_profile_coordinates(profile, transform):
    """Return a copy transformed between lon/lat and fault-local x/y.

    This is a coordinate-frame conversion only. Projection onto the top edge
    happens later and exclusively inside :func:`resolve_dip_profile`.
    """
    controls = profile.controls
    x, y = transform(controls.x, controls.y)
    projected_controls = DipControlPoints(
        x=x,
        y=y,
        dip=controls.dip,
        sampled=controls.sampled,
    )

    zones = []
    for zone in profile.transition_zones:
        if zone.center is not None:
            transformed_center = _transform_xy_point(zone.center, transform)
            zones.append(DipTransitionZone(
                center=tuple(transformed_center),
                lower_width=zone.lower_width,
                upper_width=zone.upper_width,
                metric=zone.metric,
                shape=zone.shape,
            ))
        else:
            endpoints = np.asarray(zone.endpoints, dtype=float)
            zx, zy = transform(endpoints[:, 0], endpoints[:, 1])
            zones.append(DipTransitionZone(
                endpoints=np.column_stack([zx, zy]),
                shape=zone.shape,
            ))
    return DipProfileSpec(
        controls=projected_controls,
        interpolation_axis=profile.interpolation_axis,
        transition_zones=tuple(zones),
    )


def apply_dip_perturbations(controls, perturbations, *, angle_unit="degrees"):
    """Apply candidate values to sampled controls without positional sentinels."""
    from .DipInterpolation import (
        normalize_dip_to_0_180,
        validate_dip_angles_for_depth_projection,
    )
    from .perturbations.angle_utils import angles_to_degrees

    reference = validate_dip_angles_for_depth_projection(
        np.asarray(controls.dip, dtype=float),
        name="reference dip",
    )
    result = np.asarray(normalize_dip_to_0_180(reference), dtype=float).copy()
    sample_to_control = controls.sample_to_control
    values = np.asarray(
        angles_to_degrees(perturbations, angle_unit),
        dtype=float,
    ).ravel()

    if len(sample_to_control) == 0:
        if values.size != 0:
            raise ValueError("a profile with no sampled controls requires an empty perturbation vector")
        return result
    if values.size == 1:
        values = np.full(len(sample_to_control), values.item(), dtype=float)
    elif values.size != len(sample_to_control):
        raise ValueError(
            "perturbations must be scalar or match sampled control count "
            f"({len(sample_to_control)}); got {values.size}."
        )
    result[sample_to_control] += values

    invalid = ~np.isfinite(result) | (result <= 0.0) | (result >= 180.0)
    if np.any(invalid):
        indices = np.flatnonzero(invalid)[:5].tolist()
        raise ValueError(
            "perturbed dip must remain finite and in (0, 180) degrees; "
            f"invalid control indices: {indices}"
        )
    return result


def resolve_dip_profile(
    profile,
    top_coords,
    perturbations,
    *,
    angle_unit="degrees",
    resolved_axis=None,
):
    """Project, validate and interpolate one candidate dip profile.

    The returned object is the only resolved representation used by bottom
    generation and plotting.  Raw declaration order is retained for parameter
    mapping, while independent spatial sorting is used only to build the
    one-dimensional interpolant.
    """
    from .DipInterpolation import normalize_dip_to_0_180

    if not isinstance(profile, DipProfileSpec):
        raise TypeError("profile must be a DipProfileSpec")
    top = np.asarray(top_coords, dtype=float)
    if top.ndim != 2 or top.shape[0] < 2 or top.shape[1] < 2:
        raise ValueError("top_coords must have shape (n>=2, 2+)")
    if not np.all(np.isfinite(top[:, :2])):
        raise ValueError("top_coords must be finite")
    top_xy = top[:, :2]
    top_s, seg_len = _polyline_arclength(top_xy)

    axis = resolved_axis or profile.interpolation_axis
    if axis == "auto":
        axis = _dominant_axis(top_xy)
    if axis not in {"x", "y", "arc_length"}:
        raise ValueError("resolved interpolation axis must be 'x', 'y', or 'arc_length'")

    controls = profile.controls
    raw_xy = np.column_stack([controls.x, controls.y])
    projected_xy, control_s, distances, ambiguous = _project_points(
        raw_xy,
        top_xy,
        top_s,
        seg_len,
    )
    if np.any(ambiguous):
        indices = np.flatnonzero(ambiguous).tolist()
        raise ValueError(
            "dip control projection is ambiguous between non-adjacent top-edge "
            f"segments at control indices {indices}"
        )

    top_u = _coordinate_from_projection(axis, top_xy, top_s)
    control_u = _coordinate_from_projection(axis, projected_xy, control_s)
    tolerance = _coordinate_tolerance(top_u)
    order = np.argsort(control_u, kind="stable")
    sorted_u = control_u[order]
    if np.any(np.diff(sorted_u) <= tolerance):
        raise ValueError(
            "dip controls must project to distinct positions on the resolved "
            f"{axis!r} coordinate"
        )

    if axis in {"x", "y"}:
        _validate_monotonic_axis(top_u, tolerance, required=bool(profile.transition_zones))

    control_dip = apply_dip_perturbations(
        controls,
        perturbations,
        angle_unit=angle_unit,
    )
    sorted_dip = control_dip[order]

    resolved_zones = _resolve_transition_zones(
        profile.transition_zones,
        axis=axis,
        top_xy=top_xy,
        top_s=top_s,
        top_u=top_u,
        seg_len=seg_len,
        control_u=control_u,
        control_order=order,
        tolerance=tolerance,
    )

    interpolation_u = list(sorted_u)
    interpolation_dip = list(sorted_dip)
    for zone in resolved_zones:
        lower_pos = int(np.where(order == zone.lower_control)[0][0])
        upper_pos = int(np.where(order == zone.upper_control)[0][0])
        interpolation_u.extend([zone.lower_u, zone.upper_u])
        interpolation_dip.extend([sorted_dip[lower_pos], sorted_dip[upper_pos]])

    interpolation_u = np.asarray(interpolation_u, dtype=float)
    interpolation_dip = np.asarray(interpolation_dip, dtype=float)
    interp_order = np.argsort(interpolation_u, kind="stable")
    interpolation_u = interpolation_u[interp_order]
    interpolation_dip = interpolation_dip[interp_order]
    interpolation_u, interpolation_dip = _deduplicate_interpolant(
        interpolation_u,
        interpolation_dip,
        tolerance,
    )
    top_dip = np.interp(
        top_u,
        interpolation_u,
        interpolation_dip,
        left=interpolation_dip[0],
        right=interpolation_dip[-1],
    )
    top_dip = normalize_dip_to_0_180(top_dip)
    if any(zone.shape == "smoothstep" for zone in resolved_zones):
        top_dip = _apply_transition_shapes(
            top_u,
            top_dip,
            resolved_zones,
            control_dip,
        )
        top_dip = normalize_dip_to_0_180(top_dip)

    return ResolvedDipProfile(
        interpolation_axis=axis,
        raw_control_xy=raw_xy,
        projected_control_xy=projected_xy,
        projection_distance=distances,
        control_s=control_s,
        control_u=control_u,
        control_dip=control_dip,
        sampled=controls.sampled,
        sample_to_control=controls.sample_to_control,
        top_xy=top_xy,
        top_s=top_s,
        top_u=top_u,
        top_dip_continuous=top_dip,
        interpolation_u=interpolation_u,
        interpolation_dip=interpolation_dip,
        transition_zones=resolved_zones,
    )


def _dominant_axis(top_xy):
    centred = top_xy - np.mean(top_xy, axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    return "x" if abs(vh[0, 0]) > abs(vh[0, 1]) else "y"


def _polyline_arclength(polyline):
    differences = np.diff(polyline, axis=0)
    seg_len = np.linalg.norm(differences, axis=1)
    if np.any(seg_len <= np.finfo(float).eps):
        indices = np.flatnonzero(seg_len <= np.finfo(float).eps).tolist()
        raise ValueError(f"top edge contains zero-length segment(s): {indices}")
    return np.concatenate([[0.0], np.cumsum(seg_len)]), seg_len


def _project_points(points, polyline, cumlen, seg_len):
    projected = np.empty_like(points, dtype=float)
    arclength = np.empty(len(points), dtype=float)
    distances = np.empty(len(points), dtype=float)
    ambiguous = np.zeros(len(points), dtype=bool)
    segment_start = polyline[:-1]
    segment_vector = np.diff(polyline, axis=0)

    for index, point in enumerate(points):
        rel = point - segment_start
        t = np.sum(rel * segment_vector, axis=1) / (seg_len ** 2)
        t = np.clip(t, 0.0, 1.0)
        candidates = segment_start + t[:, None] * segment_vector
        squared_distance = np.sum((candidates - point) ** 2, axis=1)
        best = int(np.argmin(squared_distance))
        projected[index] = candidates[best]
        arclength[index] = cumlen[best] + t[best] * seg_len[best]
        distances[index] = np.sqrt(squared_distance[best])

        threshold = max(1e-20, squared_distance[best] + 1e-12)
        ties = np.flatnonzero(squared_distance <= threshold)
        ambiguous[index] = any(abs(int(other) - best) > 1 for other in ties)
    return projected, arclength, distances, ambiguous


def _coordinate_from_projection(axis, xy, arclength):
    if axis == "x":
        return np.asarray(xy[:, 0], dtype=float)
    if axis == "y":
        return np.asarray(xy[:, 1], dtype=float)
    return np.asarray(arclength, dtype=float)


def _coordinate_tolerance(values):
    span = float(np.ptp(values)) if len(values) else 0.0
    return 64.0 * np.finfo(float).eps * max(1.0, span)


def _validate_monotonic_axis(values, tolerance, *, required):
    differences = np.diff(values)
    increasing = np.all(differences >= -tolerance)
    decreasing = np.all(differences <= tolerance)
    if not (increasing or decreasing):
        raise ValueError(
            "top edge is not monotonic on the resolved x/y interpolation axis; "
            "use interpolation_axis='arc_length'"
        )
    if required and np.any(np.abs(differences) <= tolerance):
        raise ValueError(
            "transition zones require a strictly monotonic x/y top-edge axis; "
            "use interpolation_axis='arc_length'"
        )


def _resolve_transition_zones(
    zones,
    *,
    axis,
    top_xy,
    top_s,
    top_u,
    seg_len,
    control_u,
    control_order,
    tolerance,
):
    resolved = []
    sorted_control_u = control_u[control_order]
    for zone in zones:
        if zone.center is not None:
            points = np.asarray([zone.center], dtype=float)
            projected, centre_s, _, ambiguous = _project_points(
                points, top_xy, top_s, seg_len,
            )
            if ambiguous[0]:
                raise ValueError("transition center projection is ambiguous")
            centre_xy = projected[0]
            centre_s = float(centre_s[0])
            centre_u = float(_coordinate_from_projection(
                axis, projected, np.array([centre_s]),
            )[0])
            if zone.metric == "axis":
                lower_u = centre_u - zone.lower_width
                upper_u = centre_u + zone.upper_width
                lower_xy, lower_s = _point_at_u(
                    lower_u, axis, top_xy, top_s, top_u, tolerance,
                )
                upper_xy, upper_s = _point_at_u(
                    upper_u, axis, top_xy, top_s, top_u, tolerance,
                )
            else:
                lower_xy, lower_s = _circle_intersection_on_side(
                    centre_xy, centre_s, zone.lower_width,
                    side="lower", axis=axis, top_xy=top_xy,
                    top_s=top_s, top_u=top_u, seg_len=seg_len,
                )
                upper_xy, upper_s = _circle_intersection_on_side(
                    centre_xy, centre_s, zone.upper_width,
                    side="upper", axis=axis, top_xy=top_xy,
                    top_s=top_s, top_u=top_u, seg_len=seg_len,
                )
                lower_u = float(_coordinate_from_projection(
                    axis, lower_xy[None, :], np.array([lower_s]),
                )[0])
                upper_u = float(_coordinate_from_projection(
                    axis, upper_xy[None, :], np.array([upper_s]),
                )[0])
            source = "center"
            raw_anchor_xy = points
            projected_anchor_xy = projected
        else:
            points = np.asarray(zone.endpoints, dtype=float)
            projected, endpoint_s, _, ambiguous = _project_points(
                points, top_xy, top_s, seg_len,
            )
            if np.any(ambiguous):
                raise ValueError("transition endpoint projection is ambiguous")
            endpoint_u = _coordinate_from_projection(axis, projected, endpoint_s)
            endpoint_order = np.argsort(endpoint_u)
            lower_u, upper_u = endpoint_u[endpoint_order]
            lower_xy, upper_xy = projected[endpoint_order]
            lower_s, upper_s = endpoint_s[endpoint_order]
            source = "endpoints"
            raw_anchor_xy = points
            projected_anchor_xy = projected

        if lower_u > upper_u:
            lower_u, upper_u = upper_u, lower_u
            lower_xy, upper_xy = upper_xy, lower_xy
            lower_s, upper_s = upper_s, lower_s
        if upper_u - lower_u <= tolerance:
            raise ValueError("transition zone collapses on the resolved coordinate")

        insertion = np.searchsorted(sorted_control_u, [lower_u, upper_u])
        lower_position = int(insertion[0] - 1)
        upper_position = int(insertion[1])
        if lower_position < 0 or upper_position >= len(sorted_control_u):
            raise ValueError("transition zone must lie between two dip controls")
        if upper_position != lower_position + 1:
            raise ValueError(
                "transition zone may span exactly one adjacent dip-control pair"
            )
        if np.any(
            (sorted_control_u > lower_u + tolerance)
            & (sorted_control_u < upper_u - tolerance)
        ):
            raise ValueError("transition zone cannot contain a dip control")

        resolved.append(ResolvedTransitionZone(
            lower_u=float(lower_u),
            upper_u=float(upper_u),
            lower_s=float(lower_s),
            upper_s=float(upper_s),
            lower_xy=lower_xy,
            upper_xy=upper_xy,
            lower_control=int(control_order[lower_position]),
            upper_control=int(control_order[upper_position]),
            metric=zone.metric,
            shape=zone.shape,
            source=source,
            raw_anchor_xy=raw_anchor_xy,
            projected_anchor_xy=projected_anchor_xy,
        ))

    resolved.sort(key=lambda item: item.lower_u)
    used_control_pairs = set()
    for zone in resolved:
        control_pair = (zone.lower_control, zone.upper_control)
        if control_pair in used_control_pairs:
            raise ValueError(
                "each adjacent dip-control pair may define at most one "
                "transition zone"
            )
        used_control_pairs.add(control_pair)
    for previous, current in zip(resolved, resolved[1:]):
        if current.lower_u <= previous.upper_u + tolerance:
            raise ValueError("transition zones must not overlap or touch")
    return tuple(resolved)


def _apply_transition_shapes(top_u, top_dip, zones, control_dip):
    """Apply opt-in transition blends without changing the linear baseline.

    The established interpolation above remains authoritative for plateaus,
    ordinary control-to-control segments, and every ``shape='linear'`` zone.
    Only samples inside an explicitly requested ``smoothstep`` interval are
    replaced.  For ``t=(u-u_L)/(u_R-u_L)``, the Hermite blend
    ``h(t)=3t**2-2t**3`` is monotone on ``[0, 1]``, stays between the adjacent
    control dips, and has zero derivative at both transition endpoints.
    """
    shaped = np.asarray(top_dip, dtype=float).copy()
    for zone in zones:
        if zone.shape == "linear":
            continue
        inside = (top_u >= zone.lower_u) & (top_u <= zone.upper_u)
        if not np.any(inside):
            continue
        t = np.clip(
            (top_u[inside] - zone.lower_u) / (zone.upper_u - zone.lower_u),
            0.0,
            1.0,
        )
        blend = 3.0 * t**2 - 2.0 * t**3
        lower_dip = control_dip[zone.lower_control]
        upper_dip = control_dip[zone.upper_control]
        shaped[inside] = lower_dip + blend * (upper_dip - lower_dip)
    return shaped


def _point_at_u(target_u, axis, top_xy, top_s, top_u, tolerance):
    minimum, maximum = float(np.min(top_u)), float(np.max(top_u))
    if target_u < minimum - tolerance or target_u > maximum + tolerance:
        raise ValueError("transition half-width extends beyond the top edge")
    if axis == "arc_length":
        s = float(np.clip(target_u, top_s[0], top_s[-1]))
    else:
        order = np.argsort(top_u, kind="stable")
        s = float(np.interp(target_u, top_u[order], top_s[order]))
    return _point_at_s(s, top_xy, top_s), s


def _point_at_s(target_s, top_xy, top_s):
    target_s = float(np.clip(target_s, top_s[0], top_s[-1]))
    segment = min(int(np.searchsorted(top_s, target_s, side="right") - 1), len(top_s) - 2)
    length = top_s[segment + 1] - top_s[segment]
    fraction = (target_s - top_s[segment]) / length
    return top_xy[segment] + fraction * (top_xy[segment + 1] - top_xy[segment])


def _circle_intersection_on_side(
    centre_xy,
    centre_s,
    radius,
    *,
    side,
    axis,
    top_xy,
    top_s,
    top_u,
    seg_len,
):
    """Find the closest on-branch circle intersection on one u side."""
    candidates = []
    vectors = np.diff(top_xy, axis=0)
    for index, vector in enumerate(vectors):
        start = top_xy[index] - centre_xy
        a = float(np.dot(vector, vector))
        b = 2.0 * float(np.dot(start, vector))
        c = float(np.dot(start, start) - radius ** 2)
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            continue
        root = np.sqrt(max(0.0, discriminant))
        for fraction in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
            if -1e-12 <= fraction <= 1.0 + 1e-12:
                fraction = float(np.clip(fraction, 0.0, 1.0))
                point = top_xy[index] + fraction * vector
                s = float(top_s[index] + fraction * seg_len[index])
                u = float(_coordinate_from_projection(
                    axis, point[None, :], np.array([s]),
                )[0])
                candidates.append((point, s, u))

    centre_u = float(_coordinate_from_projection(
        axis, centre_xy[None, :], np.array([centre_s]),
    )[0])
    if side == "lower":
        candidates = [item for item in candidates if item[2] < centre_u]
    else:
        candidates = [item for item in candidates if item[2] > centre_u]
    if not candidates:
        raise ValueError(
            f"euclidean transition radius has no {side}-side top-edge intersection"
        )
    point, s, _ = min(candidates, key=lambda item: abs(item[1] - centre_s))
    return point, s


def _deduplicate_interpolant(u, dip, tolerance):
    keep_u = [u[0]]
    keep_dip = [dip[0]]
    for current_u, current_dip in zip(u[1:], dip[1:]):
        if abs(current_u - keep_u[-1]) <= tolerance:
            if abs(current_dip - keep_dip[-1]) > 1e-10:
                raise ValueError(
                    "two profile constraints occupy the same resolved coordinate "
                    "with different dip values"
                )
            continue
        keep_u.append(current_u)
        keep_dip.append(current_dip)
    return np.asarray(keep_u), np.asarray(keep_dip)


__all__ = [
    "DipControlPoints",
    "DipTransitionZone",
    "DipProfileSpec",
    "ResolvedTransitionZone",
    "ResolvedDipProfile",
    "build_dip_profile_spec",
    "transform_dip_profile_coordinates",
    "apply_dip_perturbations",
    "resolve_dip_profile",
]
