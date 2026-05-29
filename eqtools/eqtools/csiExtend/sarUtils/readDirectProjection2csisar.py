import os

import numpy as np

from .readBase2csisar import ReadBase2csisar
from .grid_io import read_lonlat_grid
from .sar_conventions import (
    InputProjectionConvention,
    InputProjectionRole,
    ObservationType,
    DirectProjectionSarConfig,
    coerce_enum,
)
from .sar_geometry import rotate_ccw90, rotate_cw90


class DirectProjectionSarReader(ReadBase2csisar):
    """
    Reader base for products that provide ENU projection vectors directly.

    File-format subclasses should read product-specific value/projection grids
    and call the common direct-projection CSI path instead of reconstructing
    azimuth/incidence.
    """

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
            raise FileNotFoundError(f"GMTSAR/direct-projection grid file not found: {filename}")
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

    def _target_observation_type(self):
        return coerce_enum(
            ObservationType,
            self.config.observation_type,
            "observation_type",
        )

    @staticmethod
    def _axis_for_projection_convention(convention):
        convention = coerce_enum(
            InputProjectionConvention,
            convention,
            "input_projection_convention",
        )
        if convention in (
            InputProjectionConvention.TOWARD_SATELLITE,
            InputProjectionConvention.AWAY_FROM_SATELLITE,
        ):
            return "los"
        if convention in (
            InputProjectionConvention.ALONG_HEADING,
            InputProjectionConvention.OPPOSITE_HEADING,
        ):
            return "azimuth"
        raise ValueError(
            "input_projection_convention must resolve to a physical direction "
            "before axis checks."
        )

    @staticmethod
    def _projection_convention_from_value(value_convention):
        value_convention = str(
            value_convention.value if hasattr(value_convention, "value") else value_convention
        ).replace("-", "_").lower()
        if value_convention == "toward_satellite":
            return InputProjectionConvention.TOWARD_SATELLITE
        if value_convention == "away_from_satellite":
            return InputProjectionConvention.AWAY_FROM_SATELLITE
        if value_convention == "along_heading":
            return InputProjectionConvention.ALONG_HEADING
        if value_convention == "opposite_heading":
            return InputProjectionConvention.OPPOSITE_HEADING
        raise ValueError(
            "input_projection_convention='same_as_value' is not defined for "
            f"input_value_convention={value_convention!r}; use an explicit "
            "projection convention such as 'toward_satellite'."
        )

    @staticmethod
    def _target_projection_convention(spec):
        observation_type = coerce_enum(
            ObservationType,
            spec.observation_type,
            "observation_type",
        )
        if observation_type == ObservationType.PHASE_LOS:
            return InputProjectionConvention.TOWARD_SATELLITE
        if observation_type == ObservationType.LOS_DISPLACEMENT:
            return InputProjectionConvention.TOWARD_SATELLITE
        if observation_type == ObservationType.AZIMUTH_OFFSET:
            return InputProjectionConvention.ALONG_HEADING
        raise ValueError(f"Unsupported observation_type: {observation_type}.")

    def _resolve_input_projection_convention(self, convention, target_convention, spec):
        convention = coerce_enum(
            InputProjectionConvention,
            convention,
            "input_projection_convention",
        )
        if convention == InputProjectionConvention.CANONICAL:
            return target_convention
        if convention == InputProjectionConvention.SAME_AS_VALUE:
            return self._projection_convention_from_value(spec.input_value_convention)
        return convention

    @staticmethod
    def _projection_sign_between(input_convention, target_convention):
        input_convention = coerce_enum(
            InputProjectionConvention,
            input_convention,
            "input_projection_convention",
        )
        target_convention = coerce_enum(
            InputProjectionConvention,
            target_convention,
            "target_projection_convention",
        )
        if input_convention == target_convention:
            return 1.0
        opposite_pairs = {
            frozenset((
                InputProjectionConvention.TOWARD_SATELLITE,
                InputProjectionConvention.AWAY_FROM_SATELLITE,
            )),
            frozenset((
                InputProjectionConvention.ALONG_HEADING,
                InputProjectionConvention.OPPOSITE_HEADING,
            )),
        }
        if frozenset((input_convention, target_convention)) in opposite_pairs:
            return -1.0
        raise ValueError(
            "Cannot convert supplied projection direction "
            f"{input_convention.value!r} to {target_convention.value!r}; "
            "the projection axis does not match the observation axis."
        )

    def _derive_azimuth_projection_from_los(self, projection, look_side=None,
                                            target_convention=None):
        projection = np.asarray(projection, dtype=float)
        horizontal = projection[..., :2].reshape((-1, 2))
        norms = np.linalg.norm(horizontal, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("LOS projection contains zero horizontal vectors.")
        horizontal = horizontal / norms[:, np.newaxis]

        look_side = look_side or self.config.look_side
        look_side = str(look_side.value if hasattr(look_side, "value") else look_side).lower()
        if look_side == "right":
            heading = rotate_cw90(horizontal)
        elif look_side == "left":
            heading = rotate_ccw90(horizontal)
        else:
            raise ValueError("look_side must be 'right' or 'left'.")

        if target_convention is None:
            target_convention = InputProjectionConvention.ALONG_HEADING
        target_convention = coerce_enum(
            InputProjectionConvention,
            target_convention,
            "target_projection_convention",
        )
        if target_convention == InputProjectionConvention.OPPOSITE_HEADING:
            heading *= -1.0
        elif target_convention != InputProjectionConvention.ALONG_HEADING:
            raise ValueError("target azimuth projection must be 'along_heading' or 'opposite_heading'.")

        return np.column_stack((heading, np.zeros(heading.shape[0]))).reshape(projection.shape)

    def _projection_for_observation(self, projection, input_projection_role=None,
                                    input_projection_convention=None, look_side=None,
                                    spec=None):
        spec = spec if spec is not None else self.build_observation_spec()
        target_convention = self._target_projection_convention(spec)
        role = coerce_enum(
            InputProjectionRole,
            input_projection_role,
            "input_projection_role",
        )
        raw_input_convention = coerce_enum(
            InputProjectionConvention,
            input_projection_convention,
            "input_projection_convention",
        )
        input_convention = self._resolve_input_projection_convention(
            raw_input_convention,
            target_convention,
            spec,
        )
        target_axis = self._axis_for_projection_convention(target_convention)
        input_axis = self._axis_for_projection_convention(input_convention)

        if role == InputProjectionRole.SAME_AS_OBSERVATION:
            if input_axis != target_axis:
                raise ValueError(
                    "input_projection_role='same_as_observation' requires the "
                    "input projection direction to resolve to the same axis as "
                    f"the target observation; got {input_convention.value!r} "
                    f"for target {target_convention.value!r}."
                )
            return projection * self._projection_sign_between(input_convention, target_convention)

        if role == InputProjectionRole.LOS:
            if input_axis != "los":
                extra = ""
                if raw_input_convention == InputProjectionConvention.SAME_AS_VALUE:
                    extra = (
                        " For azimuth observations, 'same_as_value' resolves "
                        "from the azimuth value convention, not from the LOS "
                        "projection direction."
                    )
                raise ValueError(
                    "input_projection_role='los' requires "
                    "input_projection_convention to resolve to "
                    "'toward_satellite' or 'away_from_satellite'; got "
                    f"{input_convention.value!r}.{extra}"
                )
            projection = projection * self._projection_sign_between(
                input_convention,
                InputProjectionConvention.TOWARD_SATELLITE,
            )
            if target_axis == "los":
                return projection * self._projection_sign_between(
                    InputProjectionConvention.TOWARD_SATELLITE,
                    target_convention,
                )
            return self._derive_azimuth_projection_from_los(
                projection,
                look_side=look_side,
                target_convention=target_convention,
            )

        if role == InputProjectionRole.AZIMUTH:
            if input_axis != "azimuth":
                raise ValueError(
                    "input_projection_role='azimuth' requires "
                    "input_projection_convention to resolve to 'along_heading' "
                    f"or 'opposite_heading'; got {input_convention.value!r}."
                )
            if target_axis != "azimuth":
                raise ValueError("Cannot derive LOS projection from an azimuth-only projection vector.")
            return projection * self._projection_sign_between(input_convention, target_convention)

        raise ValueError(f"Unsupported input_projection_role: {role}.")

    def extract_raw_grd(self, directory_name=None, prefix=None, phsname=None,
                        valuefile=None, eastfile=None, northfile=None, upfile=None,
                        variable=None, value_variable=None, projection_variable=None,
                        east_variable=None, north_variable=None, up_variable=None,
                        lon_name=None, lat_name=None, grid_engine=None, coord_is_lonlat=None,
                        zero2nan=None, factor_to_m=1.0,
                        input_projection_role=None, input_projection_convention=None,
                        look_side=None, verbose=None):
        if directory_name is not None:
            self.directory_name = directory_name
        else:
            directory_name = self.directory_name

        if valuefile is None:
            valuefile = phsname
        if valuefile is None and prefix is not None:
            valuefile = prefix
        if valuefile is None:
            raise ValueError("Set valuefile, phsname, or prefix for direct-projection SAR input.")
        if eastfile is None or northfile is None:
            raise ValueError("eastfile and northfile are required for direct-projection SAR input.")

        zero2nan = zero2nan if zero2nan is not None else self.config.zero2nan
        input_projection_role = (
            input_projection_role
            if input_projection_role is not None
            else self.config.input_projection_role
        )
        input_projection_convention = (
            input_projection_convention
            if input_projection_convention is not None
            else self.config.input_projection_convention
        )
        look_side = look_side if look_side is not None else self.config.look_side
        value_variable = value_variable if value_variable is not None else variable
        projection_variable = projection_variable if projection_variable is not None else variable
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
        target_projection_convention = self._target_projection_convention(spec)
        resolved_input_projection_convention = self._resolve_input_projection_convention(
            input_projection_convention,
            target_projection_convention,
            spec,
        )
        input_projection = self._projection_from_components(east, north, up)
        projection = self._projection_for_observation(
            input_projection,
            input_projection_role=input_projection_role,
            input_projection_convention=input_projection_convention,
            look_side=look_side,
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
        self.raw_input_projection_role = input_projection_role
        self.raw_input_projection_convention = input_projection_convention
        self.raw_resolved_input_projection_convention = resolved_input_projection_convention
        self.raw_projection_look_side = look_side
        self.raw_projection_convention = target_projection_convention
        if self._is_verbose(verbose):
            self.print_input_summary()

    def _observation_spec_summary(self, spec):
        if spec is None:
            return None
        return {
            "observation_type": self._summary_value(spec.observation_type),
            "input_value_convention": self._summary_value(spec.input_value_convention),
            "wavelength": spec.wavelength,
        }

    def _projection_convention_summary(self):
        return {
            "input_projection_role": self._summary_value(
                getattr(self, "raw_input_projection_role", self.config.input_projection_role)
            ),
            "input_projection_convention": self._summary_value(
                getattr(self, "raw_input_projection_convention", self.config.input_projection_convention)
            ),
            "resolved_input_projection_convention": self._summary_value(
                getattr(self, "raw_resolved_input_projection_convention", None)
            ),
            "target_projection_convention": self._summary_value(
                getattr(
                    self,
                    "raw_projection_convention",
                    self._target_projection_convention(self.build_observation_spec()),
                )
            ),
        }

    def read_observation(self, downsample=1, zero2nan=True, wavelength=None,
                         observation_type=None, input_azimuth_role=None,
                         look_side=None, input_value_convention=None, verbose=None):
        self._require_raw_grid(
            "read_observation()",
            fields=("raw_vel", "raw_mesh_lon", "raw_mesh_lat", "raw_projection_grid"),
        )
        has_semantic_override = any(
            value is not None
            for value in (wavelength, observation_type, input_azimuth_role, look_side, input_value_convention)
        )
        spec = self.build_observation_spec(
            observation_type=observation_type,
            input_azimuth_role=input_azimuth_role,
            look_side=look_side,
            input_value_convention=input_value_convention,
            wavelength=wavelength,
        )
        projection = self.raw_projection_grid
        if has_semantic_override:
            input_projection = getattr(self, "raw_input_projection_grid", None)
            if input_projection is None:
                raise RuntimeError(
                    "Direct-projection semantic overrides require raw_input_projection_grid. "
                    "Call extract_raw_grd() with a current direct-projection reader."
                )
            raw_input_projection_convention = getattr(
                self,
                "raw_input_projection_convention",
                self.config.input_projection_convention,
            )
            self.raw_resolved_input_projection_convention = self._resolve_input_projection_convention(
                raw_input_projection_convention,
                self._target_projection_convention(spec),
                spec,
            )
            projection = self._projection_for_observation(
                input_projection,
                input_projection_role=getattr(
                    self,
                    "raw_input_projection_role",
                    self.config.input_projection_role,
                ),
                input_projection_convention=raw_input_projection_convention,
                look_side=look_side or getattr(
                    self,
                    "raw_projection_look_side",
                    self.config.look_side,
                ),
                spec=spec,
            )
        self.raw_projection_convention = self._target_projection_convention(spec)
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
