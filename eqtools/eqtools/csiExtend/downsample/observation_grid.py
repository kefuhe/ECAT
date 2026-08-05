"""Lossless observation-grid representation and export helpers.

The downsampling runtime works with flattened CSI observations, while raw SAR
and optical products are rasters.  This module keeps those two representations
connected by an explicit row/column topology.  It never interpolates values or
reduces two-dimensional geographic coordinates to approximate one-dimensional
axes.
"""

from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import yaml


GRID_TOPOLOGIES = (
    "geographic_rectilinear",
    "projected_rectilinear",
    "affine_rotated",
    "geographic_curvilinear",
)
OBSERVATION_GRID_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ObservationVariable:
    """Resolved export variable from an :class:`ObservationGrid`.

    Parameters
    ----------
    name : str
        Public variable name used by the standard NetCDF/HDF5 contract.
    values : numpy.ndarray
        Two-dimensional values in the source grid row/column order.
    component : str
        Base observation component, such as ``observation`` or ``east``.
    role : str
        One of ``component``, ``phase_cycle_delta``, ``correction_surface``,
        or ``corrected_component``.
    units : str or None
        Stored scientific unit.
    positive_convention : str or None
        Positive-direction convention inherited from the base component.
    """

    name: str
    values: np.ndarray
    component: str
    role: str
    units: str | None
    positive_convention: str | None


def _as_grid(values, shape, name):
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(
            f"Observation grid expected {name} to have shape {shape}; "
            f"got {array.shape}."
        )
    return np.array(array, copy=True)


def _coordinate_mesh(longitude, latitude, shape):
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    ny, nx = shape
    if longitude.ndim == 1 and latitude.ndim == 1:
        if longitude.size != nx or latitude.size != ny:
            raise ValueError(
                "One-dimensional longitude/latitude axes must match the "
                f"observation shape {shape}; got {longitude.size} and "
                f"{latitude.size}."
            )
        return np.meshgrid(longitude, latitude)
    if longitude.shape != shape or latitude.shape != shape:
        raise ValueError(
            "Two-dimensional longitude/latitude coordinates must match the "
            f"observation shape {shape}; got {longitude.shape} and "
            f"{latitude.shape}."
        )
    return longitude, latitude


def _coordinate_tolerance(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    scale = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    return max(1.0e-10, np.finfo(float).eps * max(scale, 1.0) * 100.0)


def _rectilinear_geographic_axes(longitude, latitude):
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    if longitude.ndim == 1 and latitude.ndim == 1:
        return longitude.copy(), latitude.copy()
    if longitude.ndim != 2 or latitude.ndim != 2:
        return None
    lon_axis = longitude[0, :]
    lat_axis = latitude[:, 0]
    lon_expected = np.broadcast_to(lon_axis, longitude.shape)
    lat_expected = np.broadcast_to(lat_axis[:, None], latitude.shape)
    if not np.allclose(
        longitude,
        lon_expected,
        rtol=0.0,
        atol=_coordinate_tolerance(longitude),
        equal_nan=True,
    ):
        return None
    if not np.allclose(
        latitude,
        lat_expected,
        rtol=0.0,
        atol=_coordinate_tolerance(latitude),
        equal_nan=True,
    ):
        return None
    return lon_axis.copy(), lat_axis.copy()


def _rectilinear_native_axes(native_x, native_y, shape):
    if native_x is None or native_y is None:
        return None
    native_x = np.asarray(native_x, dtype=float)
    native_y = np.asarray(native_y, dtype=float)
    ny, nx = shape
    if native_x.ndim == 1 and native_y.ndim == 1:
        if native_x.size == nx and native_y.size == ny:
            return native_x.copy(), native_y.copy()
        return None
    if native_x.shape != shape or native_y.shape != shape:
        return None
    x_axis = native_x[0, :]
    y_axis = native_y[:, 0]
    if not np.allclose(
        native_x,
        np.broadcast_to(x_axis, shape),
        rtol=0.0,
        atol=_coordinate_tolerance(native_x),
        equal_nan=True,
    ):
        return None
    if not np.allclose(
        native_y,
        np.broadcast_to(y_axis[:, None], shape),
        rtol=0.0,
        atol=_coordinate_tolerance(native_y),
        equal_nan=True,
    ):
        return None
    return x_axis.copy(), y_axis.copy()


def _affine_is_rotated(geotransform):
    if geotransform is None:
        return False
    if len(geotransform) != 6:
        raise ValueError("A raster geotransform must contain six coefficients.")
    return not (
        np.isclose(float(geotransform[2]), 0.0)
        and np.isclose(float(geotransform[4]), 0.0)
    )


def _analysis_mask(data, shape, source_valid_mask):
    size = int(np.prod(shape))
    for name in (
        "data_filter_raw_valid_index",
        "projection_raw_valid_index",
        "raw_valid_index",
    ):
        indices = getattr(data, name, None)
        if indices is None:
            continue
        indices = np.asarray(indices, dtype=int).reshape(-1)
        if np.any((indices < 0) | (indices >= size)):
            raise ValueError(f"{name} contains indices outside the raw grid.")
        if indices.size > 1 and np.any(np.diff(indices) <= 0):
            raise ValueError(
                f"{name} must contain unique, strictly increasing raw-grid "
                "indices in current CSI point order."
            )
        mask = np.zeros(size, dtype=bool)
        mask[indices] = True
        return mask.reshape(shape)

    point_values = getattr(data, "vel", None)
    if point_values is None:
        point_values = getattr(data, "east", None)
    if point_values is not None:
        point_count = np.asarray(point_values).size
        source_indices = np.flatnonzero(source_valid_mask.reshape(-1))
        if point_count == source_indices.size:
            mask = np.zeros(size, dtype=bool)
            mask[source_indices] = True
            return mask.reshape(shape)
    return np.array(source_valid_mask, dtype=bool, copy=True)


@dataclass
class ObservationGrid:
    """A raster observation with explicit coordinate topology.

    ``components`` always contains analysis-ready values (meters for current
    SAR and optical readers), before optional reference correction.
    """

    components: dict
    longitude: np.ndarray
    latitude: np.ndarray
    native_x: np.ndarray | None = None
    native_y: np.ndarray | None = None
    projection: np.ndarray | None = None
    crs_wkt: str | None = None
    geotransform: tuple | None = None
    component_units: dict = field(default_factory=dict)
    component_conventions: dict = field(default_factory=dict)
    source_valid_mask: np.ndarray | None = None
    analysis_valid_mask: np.ndarray | None = None
    phase_cycle_deltas: dict = field(default_factory=dict)
    correction_surfaces: dict = field(default_factory=dict)
    corrected_components: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.components:
            raise ValueError("ObservationGrid requires at least one component.")
        first_name, first_values = next(iter(self.components.items()))
        first_values = np.asarray(first_values)
        if first_values.ndim != 2:
            raise ValueError(
                f"Observation component {first_name!r} must be two-dimensional."
            )
        self.shape = first_values.shape
        self.components = {
            name: _as_grid(values, self.shape, name)
            for name, values in self.components.items()
        }
        self.longitude, self.latitude = _coordinate_mesh(
            self.longitude,
            self.latitude,
            self.shape,
        )
        if self.native_x is not None:
            self.native_x = np.asarray(self.native_x, dtype=float)
        if self.native_y is not None:
            self.native_y = np.asarray(self.native_y, dtype=float)
        if self.projection is not None:
            projection = np.asarray(self.projection, dtype=float)
            expected = self.shape + (3,)
            if projection.shape != expected:
                raise ValueError(
                    "Observation projection must have shape "
                    f"{expected}; got {projection.shape}."
                )
            self.projection = np.array(projection, copy=True)
        if self.geotransform is not None:
            self.geotransform = tuple(float(value) for value in self.geotransform)

        coordinate_valid = np.isfinite(self.longitude) & np.isfinite(self.latitude)
        component_valid = np.ones(self.shape, dtype=bool)
        for values in self.components.values():
            component_valid &= np.isfinite(values)
        inferred_source_valid = coordinate_valid & component_valid
        if self.source_valid_mask is None:
            self.source_valid_mask = inferred_source_valid
        else:
            self.source_valid_mask = _as_grid(
                self.source_valid_mask,
                self.shape,
                "source_valid_mask",
            ).astype(bool)
        if self.analysis_valid_mask is None:
            self.analysis_valid_mask = np.array(
                self.source_valid_mask,
                dtype=bool,
                copy=True,
            )
        else:
            self.analysis_valid_mask = _as_grid(
                self.analysis_valid_mask,
                self.shape,
                "analysis_valid_mask",
            ).astype(bool)
        if np.any(self.analysis_valid_mask & ~coordinate_valid):
            raise ValueError(
                "analysis_valid_mask selects pixels without finite longitude/latitude."
            )

        self.phase_cycle_deltas = {
            name: _as_grid(values, self.shape, f"{name} phase-cycle delta")
            for name, values in self.phase_cycle_deltas.items()
        }
        self.correction_surfaces = {
            name: _as_grid(values, self.shape, f"{name} correction surface")
            for name, values in self.correction_surfaces.items()
        }
        self.corrected_components = {
            name: _as_grid(values, self.shape, f"corrected {name}")
            for name, values in self.corrected_components.items()
        }
        self.topology = self._detect_topology()

    def _detect_topology(self):
        if _rectilinear_geographic_axes(self.longitude, self.latitude) is not None:
            return "geographic_rectilinear"
        if _affine_is_rotated(self.geotransform):
            return "affine_rotated"
        if _rectilinear_native_axes(self.native_x, self.native_y, self.shape) is not None:
            return "projected_rectilinear"
        return "geographic_curvilinear"

    def raw_flat_indices(self):
        """Return raw-grid indices represented by current CSI analysis points."""

        return np.flatnonzero(self.analysis_valid_mask.reshape(-1))

    def set_correction(self, component, surface, corrected):
        if component not in self.components:
            raise KeyError(f"Unknown observation component: {component!r}.")
        self.correction_surfaces[component] = _as_grid(
            surface,
            self.shape,
            f"{component} correction surface",
        )
        self.corrected_components[component] = _as_grid(
            corrected,
            self.shape,
            f"corrected {component}",
        )

    def set_phase_cycle_delta(self, component, delta, corrected):
        """Store a regional phase-cycle delta and its current result."""

        if component not in self.components:
            raise KeyError(f"Unknown observation component: {component!r}.")
        self.phase_cycle_deltas[component] = _as_grid(
            delta,
            self.shape,
            f"{component} phase-cycle delta",
        )
        self.corrected_components[component] = _as_grid(
            corrected,
            self.shape,
            f"phase-cycle corrected {component}",
        )

    def display_component(self, component):
        """Return the values that feed processing, corrected when available."""

        return self.corrected_components.get(component, self.components[component])

    def _dataset_coordinates(self):
        rectilinear = _rectilinear_geographic_axes(
            self.longitude,
            self.latitude,
        )
        if self.topology == "geographic_rectilinear" and rectilinear is not None:
            lon_axis, lat_axis = rectilinear
            return (
                ("latitude", "longitude"),
                {
                    "longitude": (
                        "longitude",
                        lon_axis,
                        {
                            "standard_name": "longitude",
                            "units": "degrees_east",
                            "axis": "X",
                        },
                    ),
                    "latitude": (
                        "latitude",
                        lat_axis,
                        {
                            "standard_name": "latitude",
                            "units": "degrees_north",
                            "axis": "Y",
                        },
                    ),
                },
            )

        native_axes = _rectilinear_native_axes(
            self.native_x,
            self.native_y,
            self.shape,
        )
        if self.topology == "projected_rectilinear" and native_axes is not None:
            x_axis, y_axis = native_axes
            return (
                ("y", "x"),
                {
                    "x": ("x", x_axis, {"axis": "X", "units": "m"}),
                    "y": ("y", y_axis, {"axis": "Y", "units": "m"}),
                    "longitude": (
                        ("y", "x"),
                        self.longitude,
                        {
                            "standard_name": "longitude",
                            "units": "degrees_east",
                        },
                    ),
                    "latitude": (
                        ("y", "x"),
                        self.latitude,
                        {
                            "standard_name": "latitude",
                            "units": "degrees_north",
                        },
                    ),
                },
            )

        coords = {
            "row": ("row", np.arange(self.shape[0], dtype=np.int64)),
            "column": ("column", np.arange(self.shape[1], dtype=np.int64)),
            "longitude": (
                ("row", "column"),
                self.longitude,
                {
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            ),
            "latitude": (
                ("row", "column"),
                self.latitude,
                {
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            ),
        }
        if self.native_x is not None and self.native_y is not None:
            native_x = np.asarray(self.native_x, dtype=float)
            native_y = np.asarray(self.native_y, dtype=float)
            if native_x.ndim == 1 and native_y.ndim == 1:
                native_x, native_y = np.meshgrid(native_x, native_y)
            coords["projected_x"] = (
                ("row", "column"),
                native_x,
                {"standard_name": "projection_x_coordinate", "units": "m"},
            )
            coords["projected_y"] = (
                ("row", "column"),
                native_y,
                {"standard_name": "projection_y_coordinate", "units": "m"},
            )
        return ("row", "column"), coords

    def to_xarray_dataset(self):
        """Build a CF-oriented xarray Dataset without interpolation."""

        try:
            import xarray as xr
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Observation-grid NetCDF export requires xarray."
            ) from exc

        dims, coords = self._dataset_coordinates()
        data_vars = {}
        for component, values in self.components.items():
            attrs = {
                "long_name": (
                    f"reader-normalized source {component} observation "
                    "before deterministic correction"
                ),
                "units": self.component_units.get(component, "m"),
                "coordinates": "longitude latitude",
                "ecat_data_role": "component",
                "ecat_base_component": component,
            }
            convention = self.component_conventions.get(component)
            if convention:
                attrs["positive_convention"] = str(convention)
            data_vars[component] = (dims, values, attrs)
            if component in self.phase_cycle_deltas:
                delta_name = (
                    "phase_cycle_delta"
                    if component == "observation"
                    else f"{component}_phase_cycle_delta"
                )
                data_vars[delta_name] = (
                    dims,
                    self.phase_cycle_deltas[component],
                    {
                        "long_name": (
                            f"additive regional phase-cycle correction for {component}"
                        ),
                        "units": self.component_units.get(component, "m"),
                        "coordinates": "longitude latitude",
                        "ecat_data_role": "phase_cycle_delta",
                        "ecat_base_component": component,
                    },
                )
                if convention:
                    data_vars[delta_name][2]["positive_convention"] = str(convention)
            if component in self.correction_surfaces:
                correction_name = (
                    "correction_surface"
                    if component == "observation"
                    else f"{component}_correction_surface"
                )
                data_vars[correction_name] = (
                    dims,
                    self.correction_surfaces[component],
                    {
                        "long_name": f"deterministic correction surface for {component}",
                        "units": self.component_units.get(component, "m"),
                        "coordinates": "longitude latitude",
                        "ecat_data_role": "correction_surface",
                        "ecat_base_component": component,
                    },
                )
                if convention:
                    data_vars[correction_name][2]["positive_convention"] = str(convention)
            if component in self.corrected_components:
                corrected_name = (
                    "corrected_observation"
                    if component == "observation"
                    else f"corrected_{component}"
                )
                has_phase_cycle = component in self.phase_cycle_deltas
                has_reference = component in self.correction_surfaces
                if has_phase_cycle and has_reference:
                    formula = (
                        "corrected = observation + phase_cycle_delta "
                        "- correction_surface"
                    )
                elif has_phase_cycle:
                    formula = "corrected = observation + phase_cycle_delta"
                else:
                    formula = "corrected = observation - correction_surface"
                data_vars[corrected_name] = (
                    dims,
                    self.corrected_components[component],
                    {
                        "long_name": (
                            f"{component} observation after deterministic correction"
                        ),
                        "units": self.component_units.get(component, "m"),
                        "coordinates": "longitude latitude",
                        "correction_formula": formula,
                        "ecat_data_role": "corrected_component",
                        "ecat_base_component": component,
                    },
                )
                if convention:
                    data_vars[corrected_name][2]["positive_convention"] = str(convention)
        if self.projection is not None:
            for index, direction in enumerate(("east", "north", "up")):
                data_vars[f"projection_{direction}"] = (
                    dims,
                    self.projection[..., index],
                    {
                        "long_name": (
                            f"ENU observation projection {direction} component"
                        ),
                        "units": "1",
                        "coordinates": "longitude latitude",
                        "ecat_data_role": "projection",
                        "ecat_base_component": direction,
                    },
                )
        data_vars["source_valid_mask"] = (
            dims,
            self.source_valid_mask.astype(np.uint8),
            {
                "long_name": "finite source observation and coordinate mask",
                "ecat_data_role": "mask",
            },
        )
        data_vars["analysis_valid_mask"] = (
            dims,
            self.analysis_valid_mask.astype(np.uint8),
            {
                "long_name": "pixels retained after reader stride and data filters",
                "ecat_data_role": "mask",
            },
        )

        reserved_attrs = {
            "Conventions",
            "grid_topology",
            "pixel_registration",
            "resampling",
            "ecat_observation_grid_version",
            "ecat_observation_components",
        }
        custom_attrs = {
            str(key): value
            for key, value in self.attrs.items()
            if key not in reserved_attrs
            and value is not None
            and np.isscalar(value)
        }
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                **custom_attrs,
                "Conventions": "CF-1.8",
                "grid_topology": self.topology,
                "pixel_registration": "pixel_center",
                "resampling": "none",
                "ecat_observation_grid_version": OBSERVATION_GRID_SCHEMA_VERSION,
                "ecat_observation_components": ",".join(self.components),
            },
        )
        if self.crs_wkt:
            spatial_ref_attrs = {
                "spatial_ref": str(self.crs_wkt),
                "crs_wkt": str(self.crs_wkt),
            }
            if self.geotransform is not None:
                spatial_ref_attrs["GeoTransform"] = " ".join(
                    f"{value:.17g}" for value in self.geotransform
                )
            dataset["spatial_ref"] = xr.DataArray(0, attrs=spatial_ref_attrs)
            for variable in data_vars:
                dataset[variable].attrs["grid_mapping"] = "spatial_ref"
        return dataset

    def export_variables(self):
        names = list(self.components)
        for component in self.components:
            if component in self.phase_cycle_deltas:
                names.append(
                    "phase_cycle_delta"
                    if component == "observation"
                    else f"{component}_phase_cycle_delta"
                )
            if component in self.correction_surfaces:
                names.append(
                    "correction_surface"
                    if component == "observation"
                    else f"{component}_correction_surface"
                )
            if component in self.corrected_components:
                names.append(
                    "corrected_observation"
                    if component == "observation"
                    else f"corrected_{component}"
                )
        if self.projection is not None:
            names.extend(
                ("projection_east", "projection_north", "projection_up")
            )
        names.extend(("source_valid_mask", "analysis_valid_mask"))
        return names


def _derived_variable_names(component):
    """Return standard derived variable names for one base component."""

    return {
        "phase_cycle_delta": (
            "phase_cycle_delta"
            if component == "observation"
            else f"{component}_phase_cycle_delta"
        ),
        "correction_surface": (
            "correction_surface"
            if component == "observation"
            else f"{component}_correction_surface"
        ),
        "corrected_component": (
            "corrected_observation"
            if component == "observation"
            else f"corrected_{component}"
        ),
    }


def resolve_observation_variable(grid, variable):
    """Resolve one public standard-file variable from an observation grid.

    Parameters
    ----------
    grid : ObservationGrid
        Source grid.
    variable : str
        Name returned by :meth:`ObservationGrid.export_variables`, such as
        ``observation`` or ``corrected_observation``.

    Returns
    -------
    ObservationVariable
        Values and scientific metadata. The returned array is a read-only view
        of the grid state; callers that retain it should copy it.

    Raises
    ------
    KeyError
        If ``variable`` is not an observation or correction variable.
    """

    variable = str(variable)
    for component, values in grid.components.items():
        units = grid.component_units.get(component)
        convention = grid.component_conventions.get(component)
        if variable == component:
            return ObservationVariable(
                variable,
                np.asarray(values),
                component,
                "component",
                units,
                convention,
            )
        names = _derived_variable_names(component)
        if (
            variable == names["phase_cycle_delta"]
            and component in grid.phase_cycle_deltas
        ):
            return ObservationVariable(
                variable,
                np.asarray(grid.phase_cycle_deltas[component]),
                component,
                "phase_cycle_delta",
                units,
                convention,
            )
        if (
            variable == names["correction_surface"]
            and component in grid.correction_surfaces
        ):
            return ObservationVariable(
                variable,
                np.asarray(grid.correction_surfaces[component]),
                component,
                "correction_surface",
                units,
                convention,
            )
        if (
            variable == names["corrected_component"]
            and component in grid.corrected_components
        ):
            return ObservationVariable(
                variable,
                np.asarray(grid.corrected_components[component]),
                component,
                "corrected_component",
                units,
                convention,
            )
    raise KeyError(
        f"Unknown observation variable {variable!r}; available variables are "
        f"{grid.export_variables()}."
    )


def _dataset_component_names(dataset):
    declared = dataset.attrs.get("ecat_observation_components")
    if declared:
        names = [name.strip() for name in str(declared).split(",") if name.strip()]
        missing = [name for name in names if name not in dataset.data_vars]
        if missing:
            raise ValueError(
                "Observation file declares missing components: "
                f"{missing}."
            )
        return names

    role_names = [
        name
        for name, data_array in dataset.data_vars.items()
        if data_array.attrs.get("ecat_data_role") == "component"
    ]
    if role_names:
        return role_names

    reserved = {
        "source_valid_mask",
        "analysis_valid_mask",
        "projection_east",
        "projection_north",
        "projection_up",
        "spatial_ref",
        "phase_cycle_delta",
        "correction_surface",
        "corrected_observation",
    }
    candidates = []
    for name, data_array in dataset.data_vars.items():
        if name in reserved or name.startswith("corrected_"):
            continue
        if name.endswith("_phase_cycle_delta") or name.endswith(
            "_correction_surface"
        ):
            continue
        if data_array.ndim == 2:
            candidates.append(name)
    if not candidates:
        raise ValueError("Observation file contains no base observation component.")
    return candidates


def _dataset_georeference(dataset):
    crs_wkt = None
    geotransform = None
    if "spatial_ref" in dataset:
        attrs = dataset["spatial_ref"].attrs
        crs_wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
        raw_transform = attrs.get("GeoTransform")
        if raw_transform:
            values = str(raw_transform).split()
            if len(values) != 6:
                raise ValueError(
                    "Observation file GeoTransform must contain six values."
                )
            geotransform = tuple(float(value) for value in values)
    return crs_wkt, geotransform


def read_observation_grid(path):
    """Read an ECAT standard CF-NetCDF/HDF5 observation grid.

    Parameters
    ----------
    path : path-like
        File previously written by :func:`write_observation_netcdf`.

    Returns
    -------
    ObservationGrid
        Fully loaded, detached grid. No interpolation or coordinate reduction
        is performed.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ModuleNotFoundError
        If xarray or a required NetCDF engine is unavailable.
    ValueError
        If the file does not satisfy the ECAT observation-grid contract.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Observation grid file does not exist: {path}.")
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading an ECAT observation grid requires xarray."
        ) from exc

    with xr.open_dataset(path, engine=_netcdf_engine(path)) as opened:
        dataset = opened.load()

    if dataset.attrs.get("pixel_registration") != "pixel_center":
        raise ValueError(
            "ECAT observation grids require pixel_registration='pixel_center'."
        )
    stored_topology = dataset.attrs.get("grid_topology")
    if stored_topology not in GRID_TOPOLOGIES:
        raise ValueError(
            "Observation file has missing or unsupported grid_topology: "
            f"{stored_topology!r}."
        )
    if "longitude" not in dataset or "latitude" not in dataset:
        raise ValueError(
            "Observation file requires longitude and latitude coordinates."
        )

    component_names = _dataset_component_names(dataset)
    first = dataset[component_names[0]]
    if first.ndim != 2:
        raise ValueError("Observation components must be two-dimensional.")
    shape = tuple(first.shape)

    components = {}
    component_units = {}
    component_conventions = {}
    phase_cycle_deltas = {}
    correction_surfaces = {}
    corrected_components = {}
    for component in component_names:
        data_array = dataset[component]
        if tuple(data_array.shape) != shape:
            raise ValueError(
                f"Observation component {component!r} has shape "
                f"{data_array.shape}; expected {shape}."
            )
        components[component] = np.asarray(data_array.values)
        component_units[component] = data_array.attrs.get("units")
        convention = data_array.attrs.get("positive_convention")
        if convention is not None:
            component_conventions[component] = str(convention)
        names = _derived_variable_names(component)
        for role, target in (
            ("phase_cycle_delta", phase_cycle_deltas),
            ("correction_surface", correction_surfaces),
            ("corrected_component", corrected_components),
        ):
            name = names[role]
            if name in dataset:
                values = np.asarray(dataset[name].values)
                if values.shape != shape:
                    raise ValueError(
                        f"Observation variable {name!r} has shape "
                        f"{values.shape}; expected {shape}."
                    )
                target[component] = values

    longitude = np.asarray(dataset["longitude"].values, dtype=float)
    latitude = np.asarray(dataset["latitude"].values, dtype=float)
    if longitude.ndim == 1 and latitude.ndim == 1:
        longitude, latitude = np.meshgrid(longitude, latitude)

    native_x = native_y = None
    if "x" in dataset.coords and "y" in dataset.coords:
        native_x = np.asarray(dataset["x"].values, dtype=float)
        native_y = np.asarray(dataset["y"].values, dtype=float)
    elif "projected_x" in dataset and "projected_y" in dataset:
        native_x = np.asarray(dataset["projected_x"].values, dtype=float)
        native_y = np.asarray(dataset["projected_y"].values, dtype=float)

    projection_names = (
        "projection_east",
        "projection_north",
        "projection_up",
    )
    present_projection = [name in dataset for name in projection_names]
    if any(present_projection) and not all(present_projection):
        raise ValueError(
            "Observation file must contain all three ENU projection variables."
        )
    projection = None
    if all(present_projection):
        projection = np.stack(
            [np.asarray(dataset[name].values) for name in projection_names],
            axis=-1,
        )

    source_valid_mask = (
        np.asarray(dataset["source_valid_mask"].values, dtype=bool)
        if "source_valid_mask" in dataset
        else None
    )
    analysis_valid_mask = (
        np.asarray(dataset["analysis_valid_mask"].values, dtype=bool)
        if "analysis_valid_mask" in dataset
        else None
    )
    crs_wkt, geotransform = _dataset_georeference(dataset)
    reserved_attrs = {
        "Conventions",
        "grid_topology",
        "pixel_registration",
        "resampling",
        "ecat_observation_grid_version",
        "ecat_observation_components",
    }
    attrs = {
        str(key): value
        for key, value in dataset.attrs.items()
        if key not in reserved_attrs
    }
    attrs["source_file"] = str(path)
    attrs["ecat_observation_grid_version"] = int(
        dataset.attrs.get("ecat_observation_grid_version", 0)
    )

    grid = ObservationGrid(
        components=components,
        longitude=longitude,
        latitude=latitude,
        native_x=native_x,
        native_y=native_y,
        projection=projection,
        crs_wkt=crs_wkt,
        geotransform=geotransform,
        component_units=component_units,
        component_conventions=component_conventions,
        source_valid_mask=source_valid_mask,
        analysis_valid_mask=analysis_valid_mask,
        phase_cycle_deltas=phase_cycle_deltas,
        correction_surfaces=correction_surfaces,
        corrected_components=corrected_components,
        attrs=attrs,
    )
    if grid.topology != stored_topology:
        raise ValueError(
            "Observation file topology metadata does not match its coordinates: "
            f"stored={stored_topology!r}, resolved={grid.topology!r}."
        )
    return grid


def _enum_value(value):
    """Return a stable text value from an enum-like metadata field."""

    if value is None:
        return None
    return str(value.value if hasattr(value, "value") else value)


def _sar_standard_metadata(spec):
    """Describe source encoding and the normalized scalar stored by ECAT."""

    if spec is None:
        return {}, "ECAT reader target convention"

    source_type = _enum_value(getattr(spec, "observation_type", None))
    source_convention = _enum_value(
        getattr(spec, "raw_value_convention", None)
    )
    if source_type == "azimuth_offset":
        stored_quantity = "azimuth_offset"
        positive_convention = "positive along heading"
    elif source_type in {"unwrapped_phase", "los_displacement"}:
        stored_quantity = "los_displacement"
        positive_convention = "positive toward satellite"
    else:
        stored_quantity = None
        positive_convention = "ECAT reader target convention"

    metadata = {
        # Retain the established attribute while making its source role
        # explicit for standalone NetCDF/HDF5 users.
        "observation_type": source_type,
        "source_observation_type": source_type,
        "source_value_convention": source_convention,
        "stored_observation_quantity": stored_quantity,
        "wavelength_m": getattr(spec, "wavelength", None),
    }
    return metadata, positive_convention


def build_observation_grid(data, data_type):
    """Build an :class:`ObservationGrid` from an existing ECAT reader object."""

    longitude = getattr(data, "raw_mesh_lon", None)
    latitude = getattr(data, "raw_mesh_lat", None)
    if longitude is None or latitude is None:
        raise ValueError(
            "Standard observation export requires raw_mesh_lon/raw_mesh_lat."
        )

    projection = None
    observation_attrs = {}
    if data_type == "sar":
        raw_values = getattr(data, "raw_vel", None)
        if raw_values is None:
            raise ValueError("SAR observation export requires raw_vel.")
        spec = getattr(data, "observation_spec", None)
        if spec is None and hasattr(data, "build_observation_spec"):
            spec = data.build_observation_spec()
        if spec is not None and hasattr(data, "convert_observation_values"):
            observation = data.convert_observation_values(raw_values, spec)
        else:
            observation = np.asarray(raw_values)
        raw_projection = getattr(data, "raw_projection_full", None)
        if raw_projection is not None:
            raw_projection = np.asarray(raw_projection, dtype=float)
            if raw_projection.size != observation.size * 3:
                raise ValueError(
                    "SAR raw projection size is not aligned with the "
                    "observation grid."
                )
            projection = raw_projection.reshape(observation.shape + (3,))
        observation_attrs, positive_convention = _sar_standard_metadata(spec)
        components = {"observation": np.asarray(observation)}
        conventions = {"observation": positive_convention}
    elif data_type == "optical":
        components = {
            "east": np.asarray(getattr(data, "raw_east")),
            "north": np.asarray(getattr(data, "raw_north")),
        }
        conventions = {
            "east": "positive east",
            "north": "positive north",
        }
    else:
        raise ValueError(f"Unsupported observation-grid data type: {data_type!r}.")

    shape = next(iter(components.values())).shape
    lon_mesh, lat_mesh = _coordinate_mesh(longitude, latitude, shape)
    source_valid = np.isfinite(lon_mesh) & np.isfinite(lat_mesh)
    for values in components.values():
        source_valid &= np.isfinite(values)

    grid = ObservationGrid(
        components=components,
        longitude=lon_mesh,
        latitude=lat_mesh,
        native_x=getattr(data, "raw_mesh_x", getattr(data, "raw_x", None)),
        native_y=getattr(data, "raw_mesh_y", getattr(data, "raw_y", None)),
        projection=projection,
        crs_wkt=getattr(data, "im_proj", None),
        geotransform=getattr(
            data,
            "raw_geotransform",
            getattr(data, "im_geotrans", None),
        ),
        component_units={name: "m" for name in components},
        component_conventions=conventions,
        source_valid_mask=source_valid,
        analysis_valid_mask=_analysis_mask(data, shape, source_valid),
        attrs={
            "data_type": data_type,
            "reader_class": type(data).__name__,
            "source_stride": int(getattr(data, "downsample", 1) or 1),
            **observation_attrs,
        },
    )
    return grid


def _netcdf_engine(output_file=None):
    suffix = (
        Path(output_file).suffix.lower()
        if output_file is not None
        else ""
    )
    if find_spec("h5netcdf") is not None:
        return "h5netcdf"
    if find_spec("netCDF4") is not None:
        return "netcdf4"
    if suffix in (".h5", ".hdf5"):
        raise ModuleNotFoundError(
            "Writing .h5/.hdf5 observation grids requires h5netcdf or "
            "netCDF4. Use a .nc output or install one HDF5-backed engine."
        )
    return "scipy"


def _verify_netcdf_roundtrip(grid, output_file, engine):
    import xarray as xr

    expected = grid.to_xarray_dataset()
    with xr.open_dataset(output_file, engine=engine) as actual:
        if actual.attrs.get("grid_topology") != grid.topology:
            raise RuntimeError("NetCDF round-trip changed grid_topology metadata.")
        if grid.crs_wkt:
            if "spatial_ref" not in actual:
                raise RuntimeError("NetCDF round-trip lost spatial_ref.")
            if actual["spatial_ref"].attrs.get("crs_wkt") != str(grid.crs_wkt):
                raise RuntimeError("NetCDF round-trip changed CRS metadata.")
            if grid.geotransform is not None:
                actual_transform = tuple(
                    float(value)
                    for value in actual["spatial_ref"].attrs[
                        "GeoTransform"
                    ].split()
                )
                if actual_transform != tuple(grid.geotransform):
                    raise RuntimeError(
                        "NetCDF round-trip changed affine geotransform metadata."
                    )
        for name in expected.data_vars:
            if name == "spatial_ref":
                continue
            if name not in actual:
                raise RuntimeError(f"NetCDF round-trip lost variable {name!r}.")
            left = np.asarray(expected[name].values)
            right = np.asarray(actual[name].values)
            if left.shape != right.shape or not np.allclose(
                left,
                right,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"NetCDF round-trip changed values for {name!r}."
                )
        for name in ("longitude", "latitude"):
            left = np.asarray(expected[name].values)
            right = np.asarray(actual[name].values)
            if left.shape != right.shape or not np.allclose(
                left,
                right,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"NetCDF round-trip changed {name} coordinates."
                )


def write_observation_netcdf(grid, output_file, verify=True):
    """Write a lossless CF-NetCDF grid to ``.nc`` or HDF5-backed output."""

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset = grid.to_xarray_dataset()
    engine = _netcdf_engine(output_file)
    encoding = {}
    if engine in ("netcdf4", "h5netcdf"):
        for name in dataset.data_vars:
            if name != "spatial_ref":
                encoding[name] = {"zlib": True, "complevel": 4}
    dataset.to_netcdf(output_file, engine=engine, encoding=encoding)
    if verify:
        _verify_netcdf_roundtrip(grid, output_file, engine)
    return str(output_file)


def _gdal_dtype(array, gdal):
    dtype = np.asarray(array).dtype
    if dtype == np.float64:
        return gdal.GDT_Float64
    if np.issubdtype(dtype, np.floating):
        return gdal.GDT_Float32
    if dtype == np.int16:
        return gdal.GDT_Int16
    if dtype == np.uint16:
        return gdal.GDT_UInt16
    if dtype == np.uint8:
        return gdal.GDT_Byte
    return gdal.GDT_Float64


def _geotiff_arrays(grid):
    arrays = dict(grid.components)
    for component in grid.components:
        if component in grid.phase_cycle_deltas:
            delta_name = (
                "phase_cycle_delta"
                if component == "observation"
                else f"{component}_phase_cycle_delta"
            )
            arrays[delta_name] = grid.phase_cycle_deltas[component]
        if component in grid.correction_surfaces:
            correction_name = (
                "correction_surface"
                if component == "observation"
                else f"{component}_correction_surface"
            )
            arrays[correction_name] = grid.correction_surfaces[component]
        if component in grid.corrected_components:
            corrected_name = (
                "corrected_observation"
                if component == "observation"
                else f"corrected_{component}"
            )
            arrays[corrected_name] = grid.corrected_components[component]
    return arrays


def write_observation_geotiffs(grid, netcdf_path, verify=True):
    """Write same-grid GeoTIFF sidecars when a reliable affine CRS exists."""

    if grid.geotransform is None or not grid.crs_wkt:
        return []
    try:
        from osgeo import gdal, osr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "GeoTIFF sidecar export requires GDAL/osgeo."
        ) from exc

    netcdf_path = Path(netcdf_path)
    netcdf_stem = netcdf_path.stem
    sidecar_stem = netcdf_stem
    canonical_suffix = "_observation"
    canonical_observation_name = (
        netcdf_stem == "observation"
        or netcdf_stem.endswith(canonical_suffix)
    )
    if netcdf_stem.endswith(canonical_suffix):
        candidate = sidecar_stem[: -len(canonical_suffix)]
        if candidate:
            sidecar_stem = candidate
    outputs = []
    for name, values in _geotiff_arrays(grid).items():
        output_name = (
            f"{netcdf_stem}.tif"
            if name == "observation" and canonical_observation_name
            else f"{sidecar_stem}_{name}.tif"
        )
        output = netcdf_path.with_name(output_name)
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            str(output),
            grid.shape[1],
            grid.shape[0],
            1,
            _gdal_dtype(values, gdal),
            options=["COMPRESS=DEFLATE", "PREDICTOR=3"],
        )
        if dataset is None:
            raise RuntimeError(f"GDAL could not create {output}.")
        dataset.SetGeoTransform(grid.geotransform)
        dataset.SetProjection(str(grid.crs_wkt))
        band = dataset.GetRasterBand(1)
        band.SetDescription(name)
        array = np.asarray(values)
        band.WriteArray(array)
        if np.issubdtype(array.dtype, np.floating):
            band.SetNoDataValue(np.nan)
        band.FlushCache()
        dataset.FlushCache()
        dataset = None

        if verify:
            reopened = gdal.Open(str(output), gdal.GA_ReadOnly)
            if reopened is None:
                raise RuntimeError(f"Could not reopen GeoTIFF sidecar {output}.")
            actual = reopened.GetRasterBand(1).ReadAsArray()
            if not np.allclose(
                np.asarray(array, dtype=float),
                np.asarray(actual, dtype=float),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"GeoTIFF round-trip changed values for {name!r}."
                )
            if not np.allclose(
                reopened.GetGeoTransform(),
                grid.geotransform,
                rtol=0.0,
                atol=0.0,
            ):
                raise RuntimeError(
                    f"GeoTIFF round-trip changed geotransform for {name!r}."
                )
            expected_crs = osr.SpatialReference()
            actual_crs = osr.SpatialReference()
            expected_crs.ImportFromWkt(str(grid.crs_wkt))
            actual_crs.ImportFromWkt(str(reopened.GetProjection()))
            if not bool(expected_crs.IsSame(actual_crs)):
                raise RuntimeError(
                    f"GeoTIFF round-trip changed CRS for {name!r}."
                )
            reopened = None
        outputs.append(str(output))
    return outputs


def observation_export_report_file(config, out_name):
    report_file = config.get("report_file", "auto")
    if report_file in (None, False):
        return None
    if str(report_file).lower() == "auto":
        return f"{out_name}_observation_grid.yml"
    return str(report_file)


def export_observation_grid(grid, config, out_name):
    """Export the canonical NetCDF and optional same-grid GeoTIFF sidecars."""

    config = config or {}
    enabled = bool(config.get("enabled", False))
    result = {
        "enabled": enabled,
        "format": "netcdf",
        "topology": grid.topology,
        "shape": [int(value) for value in grid.shape],
        "resampling": "none",
        "files": [],
        "variables": grid.export_variables(),
    }
    if not enabled:
        return result

    output_file = config.get("file", "auto")
    if output_file in (None, "auto"):
        output_file = f"{out_name}_observation.nc"
    output_file = write_observation_netcdf(
        grid,
        output_file,
        verify=bool(config.get("verify", True)),
    )
    result["files"].append(output_file)

    sidecar = config.get("geotiff_sidecar", "auto")
    if sidecar is True and (
        grid.geotransform is None or not bool(grid.crs_wkt)
    ):
        raise ValueError(
            "export.observation_grid.geotiff_sidecar=true requires a source "
            "grid with a reliable affine geotransform and CRS. Use auto or "
            "false for GAMMA/curvilinear grids."
        )
    write_sidecar = sidecar is True or (
        str(sidecar).replace("-", "_").lower() == "auto"
        and grid.geotransform is not None
        and bool(grid.crs_wkt)
    )
    if write_sidecar:
        result["files"].extend(
            write_observation_geotiffs(
                grid,
                output_file,
                verify=bool(config.get("verify", True)),
            )
        )

    report_file = observation_export_report_file(config, out_name)
    if config.get("report", True) and report_file:
        with open(report_file, "w", encoding="utf-8") as stream:
            yaml.safe_dump(result, stream, allow_unicode=True, sort_keys=False)
        result["report_file"] = report_file
    return result


def format_observation_export_report(report):
    if not report.get("enabled"):
        return ""
    lines = [
        "Observation-grid export:",
        f"  topology  : {report['topology']}",
        f"  shape     : {tuple(report['shape'])}",
        "  resampling: none",
    ]
    for path in report.get("files", []):
        lines.append(f"  file      : {path}")
    if report.get("report_file"):
        lines.append(f"  report    : {report['report_file']}")
    return "\n".join(lines)
