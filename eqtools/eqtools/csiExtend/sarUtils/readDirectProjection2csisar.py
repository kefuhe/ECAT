"""Base reader for SAR products that provide ENU projection components."""

import os

import numpy as np

from .grid_io import read_lonlat_grid
from .readBase2csisar import ReadBase2csisar
from .sar_conventions import (
    AcquisitionLookSide,
    DirectProjectionSarConfig,
    ObservationType,
    ProjectionAxis,
    ProjectionDirection,
    coerce_enum,
)
from .sar_geometry import rotate_ccw90, rotate_cw90


class DirectProjectionSarReader(ReadBase2csisar):
    """Read values and explicit ENU projections, then canonicalize both."""

    config_cls = DirectProjectionSarConfig
    mode_presets = {}

    @staticmethod
    def _resolve_path(directory_name, filename):
        if filename is None:
            return None
        if os.path.isabs(filename):
            return filename
        return os.path.join(directory_name, filename)

    @classmethod
    def read_grid(cls, filename, variable=None, lon_name=None, lat_name=None,
                  engine=None, coord_is_lonlat=None):
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"GMTSAR/direct-projection grid file not found: {filename}"
            )
        return read_lonlat_grid(
            filename,
            variable=variable,
            lon_name=lon_name,
            lat_name=lat_name,
            engine=engine,
            coord_is_lonlat=coord_is_lonlat,
        )

    @staticmethod
    def _check_matching_grid(reference_shape, values, label):
        if np.asarray(values).shape != tuple(reference_shape):
            raise ValueError(
                f"{label} grid shape {np.asarray(values).shape} does not match "
                f"value grid shape {tuple(reference_shape)}."
            )

    @staticmethod
    def _projection_from_components(east, north, up):
        return np.stack((east, north, up), axis=-1)

    @staticmethod
    def _target_projection(observation_spec):
        if observation_spec.observation_type in (
            ObservationType.UNWRAPPED_PHASE,
            ObservationType.LOS_DISPLACEMENT,
        ):
            return ProjectionAxis.LOS, ProjectionDirection.GROUND_TO_SENSOR
        if observation_spec.observation_type == ObservationType.AZIMUTH_OFFSET:
            return ProjectionAxis.AZIMUTH, ProjectionDirection.ALONG_HEADING
        raise ValueError(
            f"Unsupported observation_type: {observation_spec.observation_type}."
        )

    @staticmethod
    def _validate_axis_direction(axis, direction):
        valid = {
            ProjectionAxis.LOS: {
                ProjectionDirection.GROUND_TO_SENSOR,
                ProjectionDirection.SENSOR_TO_GROUND,
            },
            ProjectionAxis.AZIMUTH: {
                ProjectionDirection.ALONG_HEADING,
                ProjectionDirection.OPPOSITE_HEADING,
            },
        }
        if direction not in valid[axis]:
            allowed = ", ".join(item.value for item in valid[axis])
            raise ValueError(
                f"input_projection_axis={axis.value!r} does not accept "
                f"input_projection_direction={direction.value!r}; "
                f"expected one of: {allowed}."
            )

    @staticmethod
    def _canonicalize_input_direction(projection, axis, direction):
        DirectProjectionSarReader._validate_axis_direction(axis, direction)
        canonical = {
            ProjectionAxis.LOS: ProjectionDirection.GROUND_TO_SENSOR,
            ProjectionAxis.AZIMUTH: ProjectionDirection.ALONG_HEADING,
        }[axis]
        return np.asarray(projection, dtype=float) * (
            1.0 if direction == canonical else -1.0
        )

    @staticmethod
    def _derive_heading_from_ground_to_sensor(projection, acquisition_look_side):
        projection = np.asarray(projection, dtype=float)
        horizontal = projection[..., :2].reshape((-1, 2))
        norms = np.linalg.norm(horizontal, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("LOS projection contains zero horizontal vectors.")
        horizontal = horizontal / norms[:, np.newaxis]

        side = coerce_enum(
            AcquisitionLookSide,
            acquisition_look_side,
            "acquisition_look_side",
        )
        if side == AcquisitionLookSide.RIGHT:
            heading = rotate_cw90(horizontal)
        else:
            heading = rotate_ccw90(horizontal)
        return np.column_stack((heading, np.zeros(heading.shape[0]))).reshape(
            projection.shape
        )

    def _projection_for_observation(
            self, projection, input_projection_axis=None,
            input_projection_direction=None, acquisition_look_side=None,
            spec=None):
        spec = spec if spec is not None else self.build_observation_spec()
        axis = coerce_enum(
            ProjectionAxis,
            input_projection_axis
            if input_projection_axis is not None
            else self.config.input_projection_axis,
            "input_projection_axis",
        )
        direction = coerce_enum(
            ProjectionDirection,
            input_projection_direction
            if input_projection_direction is not None
            else self.config.input_projection_direction,
            "input_projection_direction",
        )
        canonical_input = self._canonicalize_input_direction(
            projection, axis, direction
        )
        target_axis, _ = self._target_projection(spec)
        if axis == target_axis:
            return canonical_input
        if axis == ProjectionAxis.LOS and target_axis == ProjectionAxis.AZIMUTH:
            side = (
                acquisition_look_side
                if acquisition_look_side is not None
                else self.config.acquisition_look_side
            )
            return self._derive_heading_from_ground_to_sensor(
                canonical_input, side
            )
        raise ValueError(
            "Cannot derive a LOS projection from an azimuth-only projection "
            "because incidence information is unavailable."
        )

    def extract_raw_grd(
            self, directory_name=None, prefix=None, phsname=None,
            valuefile=None, eastfile=None, northfile=None, upfile=None,
            variable=None, value_variable=None, projection_variable=None,
            east_variable=None, north_variable=None, up_variable=None,
            lon_name=None, lat_name=None, grid_engine=None,
            coord_is_lonlat=None, zero2nan=None, factor_to_m=1.0,
            input_projection_axis=None, input_projection_direction=None,
            acquisition_look_side=None, verbose=None):
        """Read value/projection grids and normalize the projection direction."""

        if directory_name is not None:
            self.directory_name = directory_name
        else:
            directory_name = self.directory_name

        if valuefile is None:
            valuefile = phsname
        if valuefile is None and prefix is not None:
            valuefile = prefix
        if valuefile is None:
            raise ValueError(
                "Set valuefile, phsname, or prefix for direct-projection SAR input."
            )
        if eastfile is None or northfile is None:
            raise ValueError(
                "eastfile and northfile are required for direct-projection SAR input."
            )

        zero2nan = zero2nan if zero2nan is not None else self.config.zero2nan
        axis = coerce_enum(
            ProjectionAxis,
            input_projection_axis
            if input_projection_axis is not None
            else self.config.input_projection_axis,
            "input_projection_axis",
        )
        direction = coerce_enum(
            ProjectionDirection,
            input_projection_direction
            if input_projection_direction is not None
            else self.config.input_projection_direction,
            "input_projection_direction",
        )
        side = coerce_enum(
            AcquisitionLookSide,
            acquisition_look_side
            if acquisition_look_side is not None
            else self.config.acquisition_look_side,
            "acquisition_look_side",
        )
        self._validate_axis_direction(axis, direction)
        value_variable = value_variable if value_variable is not None else variable
        projection_variable = (
            projection_variable if projection_variable is not None else variable
        )
        east_variable = east_variable if east_variable is not None else projection_variable
        north_variable = north_variable if north_variable is not None else projection_variable
        up_variable = up_variable if up_variable is not None else projection_variable

        value_path = self._resolve_path(directory_name, valuefile)
        east_path = self._resolve_path(directory_name, eastfile)
        north_path = self._resolve_path(directory_name, northfile)
        up_path = self._resolve_path(directory_name, upfile)

        values, lon, lat, mesh_lon, mesh_lat = self.read_grid(
            value_path,
            variable=value_variable,
            lon_name=lon_name,
            lat_name=lat_name,
            engine=grid_engine,
            coord_is_lonlat=coord_is_lonlat,
        )
        values = np.asarray(values, dtype=float) * float(factor_to_m)
        east = self.read_grid(
            east_path,
            variable=east_variable,
            lon_name=lon_name,
            lat_name=lat_name,
            engine=grid_engine,
            coord_is_lonlat=coord_is_lonlat,
        )[0]
        north = self.read_grid(
            north_path,
            variable=north_variable,
            lon_name=lon_name,
            lat_name=lat_name,
            engine=grid_engine,
            coord_is_lonlat=coord_is_lonlat,
        )[0]
        if up_path is None:
            up = np.zeros_like(east, dtype=float)
        else:
            up = self.read_grid(
                up_path,
                variable=up_variable,
                lon_name=lon_name,
                lat_name=lat_name,
                engine=grid_engine,
                coord_is_lonlat=coord_is_lonlat,
            )[0]

        for label, grid in (("east", east), ("north", north), ("up", up)):
            self._check_matching_grid(values.shape, grid, label)
        if zero2nan:
            values = np.array(values, copy=True)
            values[values == 0] = np.nan

        spec = self.build_observation_spec()
        input_projection = self._projection_from_components(east, north, up)
        projection = self._projection_for_observation(
            input_projection,
            input_projection_axis=axis,
            input_projection_direction=direction,
            acquisition_look_side=side,
            spec=spec,
        )
        invalid_projection = ~np.all(np.isfinite(projection), axis=-1)
        if np.any(invalid_projection):
            values = np.array(values, copy=True)
            values[invalid_projection] = np.nan

        self.value_file = value_path
        self.east_file = east_path
        self.north_file = north_path
        self.up_file = up_path
        self.raw_vel = values
        self.raw_lon = lon if lon.ndim == 1 else np.nanmean(mesh_lon, axis=0)
        self.raw_lat = lat if lat.ndim == 1 else np.nanmean(mesh_lat, axis=1)
        self.raw_mesh_lon = mesh_lon
        self.raw_mesh_lat = mesh_lat
        self.raw_input_projection_grid = input_projection
        self.raw_projection_grid = projection
        self.raw_projection_full = projection.reshape((-1, 3))
        self.raw_input_projection_axis = axis
        self.raw_input_projection_direction = direction
        self.raw_projection_look_side = side
        self.raw_projection_axis, self.raw_projection_direction = (
            self._target_projection(spec)
        )
        if self._is_verbose(verbose):
            self.print_input_summary()

    def _projection_convention_summary(self):
        target_axis, target_direction = self._target_projection(
            self.observation_spec
            if self.observation_spec is not None
            else self.build_observation_spec()
        )
        input_axis = getattr(
            self,
            "raw_input_projection_axis",
            self.config.input_projection_axis,
        )
        input_axis = coerce_enum(
            ProjectionAxis, input_axis, "input_projection_axis"
        )
        return {
            "input_projection_axis": self._summary_value(input_axis),
            "input_projection_direction": self._summary_value(
                getattr(
                    self,
                    "raw_input_projection_direction",
                    self.config.input_projection_direction,
                )
            ),
            "target_projection_axis": self._summary_value(target_axis),
            "target_projection_direction": self._summary_value(target_direction),
            "acquisition_look_side_used": (
                input_axis == ProjectionAxis.LOS
                and target_axis == ProjectionAxis.AZIMUTH
            ),
        }

    def read_observation(
            self, downsample=1, zero2nan=True, wavelength=None,
            observation_type=None, raw_value_convention=None, verbose=None):
        self._require_raw_grid(
            "read_observation()",
            fields=(
                "raw_vel",
                "raw_mesh_lon",
                "raw_mesh_lat",
                "raw_input_projection_grid",
            ),
        )
        spec = self.build_observation_spec(
            observation_type=observation_type,
            raw_value_convention=raw_value_convention,
            wavelength=wavelength,
        )
        projection = self._projection_for_observation(
            self.raw_input_projection_grid,
            input_projection_axis=self.raw_input_projection_axis,
            input_projection_direction=self.raw_input_projection_direction,
            acquisition_look_side=self.raw_projection_look_side,
            spec=spec,
        )
        self.raw_projection_grid = projection
        self.raw_projection_axis, self.raw_projection_direction = (
            self._target_projection(spec)
        )
        return self.read_observation_with_projection_to_csi(
            self.raw_vel,
            lon=self.raw_mesh_lon,
            lat=self.raw_mesh_lat,
            projection=projection,
            downsample=downsample,
            zero2nan=zero2nan,
            spec=spec,
            verbose=verbose,
        )
