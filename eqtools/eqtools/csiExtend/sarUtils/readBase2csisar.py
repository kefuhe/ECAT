import xarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from abc import abstractmethod

# import csi modules and csiExtend modules
from csi.insar import insar
from eqtools.viztools import set_degree_formatter, sci_plot_style
from .sar_conventions import (
    AngleProjectionSarConfig,
    GammaTiffConfig,
    GammasarConfig,
    Hyp3TiffConfig,
    InputAzimuthRole,
    InputValueConvention,
    LookSide,
    ObservationType,
    SarObservationSpec,
    SarReaderConfig,
    coerce_enum,
    config_from_preset,
)
from .sar_geometry import (
    infer_look_side,
    normalize_azimuth as _normalize_azimuth,
    normalize_incidence as _normalize_incidence,
)
from .sar_observation import (
    convert_observation_values as _convert_observation_values,
    prepare_observation_for_csi,
    unwrapped_phase_to_los,
)


# -------------------------Define the reader class-------------------------#
class ReadBase2csisar(insar):
    config_cls = AngleProjectionSarConfig
    mode_presets = {
        "unwrapped_phase": "generic_phase_los",
        "phase_los": "generic_phase_los",
        "los": "generic_los_displacement",
        "los_displacement": "generic_los_displacement",
        "range": "generic_range_offset",
        "range_offset": "generic_range_offset",
        "az": "generic_azimuth_offset",
        "azimuth": "generic_azimuth_offset",
        "azimuth_offset": "generic_azimuth_offset",
    }

    def __init__(self, name=None, utmzone=None, lon0=None, lat0=None,
                 directory_name='.', config=None, preset=None, mode=None,
                 verbose=False):
        super().__init__(name, utmzone=utmzone, lon0=lon0, lat0=lat0, verbose=verbose)
        self.directory_name = directory_name
        self.config = self._resolve_config(config=config, preset=preset, mode=mode)
        self.verbose = verbose

        # Save in self object
        self.wavelength = None
        self.raw_vel = None
        self.raw_azimuth_input = None
        self.raw_incidence_input = None
        self.raw_azimuth_enu = None
        self.raw_azimuth_role = None
        self.raw_incidence = None
        self.raw_lon = None
        self.raw_lat = None
        self.raw_mesh_lon = None
        self.raw_mesh_lat = None
        self.raw_projection_full = None
        self.projection_valid = None
        self.projection_downsampled_valid_index = None
        self.observation_spec = None
        self.raw_angle_convention = None

    @classmethod
    def _mode_key(cls, mode):
        return str(mode).replace("-", "_").lower()

    @classmethod
    def preset_from_mode(cls, mode):
        """
        Return this reader's full preset name for a short user-facing mode.
        """
        key = cls._mode_key(mode)
        try:
            return cls.mode_presets[key]
        except KeyError:
            modes = ", ".join(cls.available_modes()) or "none"
            raise ValueError(
                f"Unsupported mode {mode!r} for {cls.__name__}. "
                f"Available modes: {modes}."
            )

    @classmethod
    def available_modes(cls):
        """Return short modes supported by this reader class."""
        return tuple(sorted(cls.mode_presets))

    @classmethod
    def available_presets(cls):
        """Return full preset names reachable from this reader's short modes."""
        return tuple(sorted(set(cls.mode_presets.values())))

    @classmethod
    def _preset_value(cls, preset):
        return preset.value if hasattr(preset, "value") else str(preset).replace("-", "_").lower()

    @classmethod
    def _check_supported_preset(cls, preset):
        preset_value = cls._preset_value(preset)
        allowed = cls.available_presets()
        if preset_value not in allowed:
            allowed_text = ", ".join(allowed) or "none"
            raise ValueError(
                f"Preset {preset_value!r} is not supported for {cls.__name__}. "
                f"Available presets: {allowed_text}. Use config=... for a custom convention."
            )
        return preset_value

    def _config_from_reader_preset(self, preset):
        preset_value = self._check_supported_preset(preset)
        return config_from_preset(preset_value)

    def _resolve_config(self, config=None, preset=None, mode=None):
        explicit = [item is not None for item in (config, preset, mode)]
        if sum(explicit) > 1:
            raise ValueError("Use only one of config, preset, or mode.")
        if mode is not None:
            return self._config_from_reader_preset(self.preset_from_mode(mode))
        if preset is not None:
            return self._config_from_reader_preset(preset)
        if config is not None:
            return config
        return self.config_cls()

    def _has_raw_grid(self):
        return any(
            getattr(self, name, None) is not None
            for name in (
                "raw_vel",
                "raw_azimuth_enu",
                "raw_incidence",
                "raw_lon",
                "raw_lat",
                "raw_mesh_lon",
                "raw_mesh_lat",
            )
        )

    def _ensure_config_mutable(self, action):
        if self._has_raw_grid():
            raise RuntimeError(
                f"{action} must be called before extract_raw_grd(); raw SAR grids "
                "have already been loaded. Pass read_observation() overrides for "
                "one-off semantic changes, or create a new reader."
            )

    def apply_preset(self, preset):
        """
        Replace the reader configuration with a common product preset.

        Call this before `extract_raw_grd()` because presets define angle
        conventions as well as observation-value conventions.
        """
        self._ensure_config_mutable("apply_preset()")
        self.config = self._config_from_reader_preset(preset)
        return self

    def apply_mode(self, mode):
        """
        Replace the reader configuration with one of this reader's short modes.

        Call this before `extract_raw_grd()`, just like `apply_preset()`.
        """
        return self.apply_preset(self.preset_from_mode(mode))

    def _require_raw_grid(self, action, fields=None):
        if fields is None:
            fields = (
                "raw_vel",
                "raw_mesh_lon",
                "raw_mesh_lat",
                "raw_azimuth_enu",
                "raw_incidence",
            )
        missing = [name for name in fields if getattr(self, name, None) is None]
        if missing:
            raise RuntimeError(
                f"{action} requires extracted raw SAR grids. Call extract_raw_grd() "
                f"first. Missing: {', '.join(missing)}."
            )

    @staticmethod
    def _single_file_match(pattern, label, exclude_suffixes=()):
        from glob import glob

        matches = sorted(glob(pattern))
        if exclude_suffixes:
            suffixes = tuple(suffix.lower() for suffix in exclude_suffixes)
            matches = [
                path for path in matches
                if not path.replace("\\", "/").lower().endswith(suffixes)
            ]

        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"Could not find {label} matching {pattern!r}.")

        match_text = ", ".join(matches)
        raise ValueError(f"Found multiple {label} files matching {pattern!r}: {match_text}.")

    def _is_verbose(self, verbose=None):
        return self.verbose if verbose is None else verbose

    @staticmethod
    def _summary_value(value):
        if hasattr(value, "value"):
            return value.value
        return value

    def _set_raw_angle_convention(self, azimuth_reference=None, azimuth_unit=None,
                                  azimuth_direction=None, incidence_reference=None,
                                  incidence_unit=None):
        self.raw_angle_convention = {
            "raw_azimuth_input": {
                "unit": self._summary_value(azimuth_unit),
                "reference": self._summary_value(azimuth_reference),
                "direction": self._summary_value(azimuth_direction),
            },
            "raw_incidence_input": {
                "unit": self._summary_value(incidence_unit),
                "reference": self._summary_value(incidence_reference),
            },
        }

    def _angle_convention_summary(self):
        if not isinstance(self.config, AngleProjectionSarConfig):
            return {}
        if self.raw_angle_convention is not None:
            return self.raw_angle_convention
        return {
            "raw_azimuth_input": {
                "unit": self._summary_value(self.config.azimuth_unit),
                "reference": self._summary_value(self.config.azimuth_reference),
                "direction": self._summary_value(self.config.azimuth_direction),
            },
            "raw_incidence_input": {
                "unit": self._summary_value(self.config.incidence_unit),
                "reference": self._summary_value(self.config.incidence_reference),
            },
        }

    def _observation_spec_summary(self, spec):
        if spec is None:
            return None
        return {
            "observation_type": self._summary_value(spec.observation_type),
            "input_azimuth_role": self._summary_value(spec.input_azimuth_role),
            "look_side": self._summary_value(spec.look_side),
            "input_value_convention": self._summary_value(spec.input_value_convention),
            "wavelength": spec.wavelength,
        }

    def _array_stats(self, values):
        if values is None:
            return None
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return None
        finite = array[np.isfinite(array)]
        return {
            "total_count": int(array.size),
            "valid_count": int(finite.size),
            "finite": finite,
        }

    def _nanmean_stat(self, values):
        stats = self._array_stats(values)
        if stats is None or stats["valid_count"] == 0:
            return None
        return {
            "nanmean": float(np.nanmean(stats["finite"])),
            "valid_count": stats["valid_count"],
            "total_count": stats["total_count"],
        }

    def _percentile_range_stat(self, values, central_percentile=99.0):
        if not 0.0 < central_percentile <= 100.0:
            raise ValueError("central_percentile must be in the interval (0, 100].")
        stats = self._array_stats(values)
        if stats is None or stats["valid_count"] == 0:
            return None
        tail_percent = (100.0 - central_percentile) / 2.0
        range_min, range_max = np.nanpercentile(
            stats["finite"],
            [tail_percent, 100.0 - tail_percent],
        )
        return {
            "central_percentile": float(central_percentile),
            "min": float(range_min),
            "max": float(range_max),
            "robust_min": float(range_min),
            "robust_max": float(range_max),
            "full_min": float(np.nanmin(stats["finite"])),
            "full_max": float(np.nanmax(stats["finite"])),
            "valid_count": stats["valid_count"],
            "total_count": stats["total_count"],
            "invalid_count": int(stats["total_count"] - stats["valid_count"]),
        }

    def _raw_angle_nanmeans(self):
        if self.raw_azimuth_input is None and self.raw_incidence_input is None:
            return {}
        angle_convention = self._angle_convention_summary()
        angle_sources = (
            ("raw_azimuth_input", self.raw_azimuth_input),
            ("raw_incidence_input", self.raw_incidence_input),
        )
        stats = {}
        for name, values in angle_sources:
            stat = self._nanmean_stat(values)
            if stat is not None:
                stat.update(angle_convention.get(name, {}))
                stats[name] = stat
        return stats

    def _observation_value_range(self, central_percentile=99.0):
        vel_stat = self._percentile_range_stat(
            getattr(self, "vel", None),
            central_percentile,
        )
        if vel_stat is not None:
            return {"vel": vel_stat}

        raw_values = self.raw_vel
        raw_value_source = "raw_vel"
        if raw_values is not None:
            try:
                spec = self.observation_spec if self.observation_spec is not None else self.build_observation_spec()
                raw_values = self.convert_observation_values(raw_values, spec)
                raw_value_source = "raw_vel converted to observation"
            except Exception:
                raw_values = self.raw_vel
        raw_vel_stat = self._percentile_range_stat(raw_values, central_percentile)
        if raw_vel_stat is not None:
            raw_vel_stat["source"] = raw_value_source
            return {"raw_vel": raw_vel_stat}
        return {}

    def _projection_nanmeans(self):
        projection = getattr(self, "projection_valid", None)
        source = "projection_valid"
        if projection is None:
            projection = getattr(self, "raw_projection_full", None)
            source = "raw_projection_full"
        if projection is None:
            projection_grid = getattr(self, "raw_projection_grid", None)
            if projection_grid is not None:
                projection = np.asarray(projection_grid).reshape((-1, 3))
                source = "raw_projection_grid"
        if projection is None:
            return {}

        projection = np.asarray(projection, dtype=float)
        if projection.ndim != 2 or projection.shape[1] != 3:
            return {}

        stats = {}
        labels = ("east", "north", "up")
        finite_rows = np.all(np.isfinite(projection), axis=1)
        for index, label in enumerate(labels):
            component = projection[:, index]
            stat = self._nanmean_stat(component)
            if stat is not None:
                stat["source"] = source
                stat["finite_vector_count"] = int(finite_rows.sum())
                stats[f"projection_{label}"] = stat
        return stats

    def _projection_convention_summary(self):
        return {}

    def get_input_summary(self, central_percentile=99.0):
        """
        Return a compact summary of the SAR observation convention and values.

        The user-facing summary is intentionally small: final observation
        convention, raw angle means, and the robust range of the observation
        values. Observation values (`raw_vel`/`vel`) do not report a mean.
        """
        spec = self.observation_spec
        if spec is None:
            spec = self.build_observation_spec()

        summary = {
            "observation_spec": self._observation_spec_summary(spec),
            "projection_convention": self._projection_convention_summary(),
            "raw_angle_nanmean": self._raw_angle_nanmeans(),
            "projection_nanmean": self._projection_nanmeans(),
            "value_range": self._observation_value_range(central_percentile),
        }

        return summary

    @staticmethod
    def _format_stat_value(value):
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _format_angle_mean(self, stat):
        mean_text = self._format_stat_value(stat["nanmean"])
        unit = stat.get("unit")
        if unit:
            mean_text = f"{mean_text} {unit}"
        if unit == "radian":
            degree_value = np.rad2deg(stat["nanmean"])
            mean_text = f"{mean_text} ({self._format_stat_value(float(degree_value))} degree)"
        return mean_text

    def _print_rows(self, title, rows, file=None):
        if not rows:
            return
        print(f"  {title}:", file=file)
        key_width = max(len(key) for key, _ in rows)
        for key, value in rows:
            print(f"    {key:<{key_width}} : {value}", file=file)

    def _spec_rows(self, spec_summary):
        if not spec_summary:
            return []
        preferred_order = (
            "observation_type",
            "input_value_convention",
            "input_azimuth_role",
            "look_side",
            "wavelength",
        )
        return [
            (key, self._format_stat_value(spec_summary[key]))
            for key in preferred_order
            if key in spec_summary
        ]

    def _projection_convention_rows(self, projection_summary):
        if not projection_summary:
            return []
        preferred_order = (
            "input_projection_role",
            "input_projection_convention",
            "resolved_input_projection_convention",
            "target_projection_convention",
        )
        return [
            (key, self._format_stat_value(projection_summary[key]))
            for key in preferred_order
            if key in projection_summary and projection_summary[key] is not None
        ]

    def _nanmean_rows(self, stats):
        rows = []
        for name, stat in stats.items():
            mean_text = self._format_angle_mean(stat)

            details = []
            if stat.get("reference"):
                details.append(f"reference={stat['reference']}")
            if stat.get("direction"):
                details.append(f"direction={stat['direction']}")
            details.append(f"valid {stat['valid_count']}/{stat['total_count']}")
            rows.append((name, f"{mean_text} ({', '.join(details)})"))
        return rows

    def _range_rows(self, stats):
        rows = []
        for name, stat in stats.items():
            details = [
                f"finite {stat['valid_count']}/{stat['total_count']}",
                (
                    f"robust {stat['central_percentile']:g}% "
                    f"[{self._format_stat_value(stat['robust_min'])}, "
                    f"{self._format_stat_value(stat['robust_max'])}]"
                ),
                (
                    f"full [{self._format_stat_value(stat['full_min'])}, "
                    f"{self._format_stat_value(stat['full_max'])}]"
                ),
            ]
            if stat.get("source"):
                details.append(f"source={stat['source']}")
            rows.append((name, ", ".join(details)))
        return rows

    def _projection_mean_rows(self, stats):
        rows = []
        for name, stat in stats.items():
            details = [f"valid {stat['valid_count']}/{stat['total_count']}"]
            source = stat.get("source")
            if source:
                details.append(f"source={source}")
            if "finite_vector_count" in stat:
                details.append(f"finite vectors {stat['finite_vector_count']}/{stat['total_count']}")
            rows.append((
                name,
                f"{self._format_stat_value(stat['nanmean'])} ({', '.join(details)})",
            ))
        return rows

    def print_input_summary(self, central_percentile=99.0, file=None):
        """
        Print the SAR observation convention and compact value diagnostics.

        This method is safe to call explicitly after `extract_raw_grd()` or
        after `read_observation()`. When CSI-level `vel` is not available yet,
        the range is reported from `raw_vel`.
        """
        summary = self.get_input_summary(central_percentile=central_percentile)
        name = getattr(self, "name", None)
        name_suffix = f" for {name}" if name else ""
        print(f"SAR observation summary{name_suffix}:", file=file)
        self._print_rows("Final observation spec", self._spec_rows(summary["observation_spec"]), file=file)
        self._print_rows(
            "Direct projection convention",
            self._projection_convention_rows(summary["projection_convention"]),
            file=file,
        )
        self._print_rows(
            "Raw angle inputs (NaN ignored)",
            self._nanmean_rows(summary["raw_angle_nanmean"]),
            file=file,
        )
        self._print_rows(
            "Projection component means (NaN ignored)",
            self._projection_mean_rows(summary["projection_nanmean"]),
            file=file,
        )
        self._print_rows(
            "Observation values",
            self._range_rows(summary["value_range"]),
            file=file,
        )
        return summary

    def show_input_summary(self, central_percentile=99.0, file=None):
        """Alias for `print_input_summary()` for interactive use."""
        return self.print_input_summary(central_percentile=central_percentile, file=file)
    
    @abstractmethod
    def extract_raw_grd(self, directory_name=None, prefix=None, phsname=None, rscname=None,
                        azifile=None, incfile=None, zero2nan=True, wavelength=None,
                        azimuth_reference=None, azimuth_unit=None,
                        azimuth_direction=None, incidence_reference=None,
                        incidence_unit=None, verbose=None, *args, **kwargs):
        pass

    def set_directory_name(self, directory_name):
        self.directory_name = directory_name

    def normalize_azimuth(self, azimuth, reference=None, unit=None, direction=None):
        reference = reference if reference is not None else self.config.azimuth_reference
        unit = unit if unit is not None else self.config.azimuth_unit
        direction = direction if direction is not None else self.config.azimuth_direction
        return _normalize_azimuth(azimuth, reference, unit, direction)

    def normalize_incidence(self, incidence, reference=None, unit=None):
        reference = reference if reference is not None else self.config.incidence_reference
        unit = unit if unit is not None else self.config.incidence_unit
        return _normalize_incidence(incidence, reference, unit)
    
    def phase_to_los(self, vel, wavelength=None):
        if wavelength is None:
            wavelength = self.wavelength
        else:
            self.wavelength = wavelength

        return unwrapped_phase_to_los(vel, wavelength)

    def build_observation_spec(self, spec=None, observation_type=None,
                               input_azimuth_role=None, look_side=None,
                               input_value_convention=None, wavelength=None):
        if spec is not None:
            if not isinstance(spec, SarObservationSpec):
                raise TypeError("spec must be a SarObservationSpec.")
            return spec

        observation_type = coerce_enum(
            ObservationType,
            observation_type if observation_type is not None else self.config.observation_type,
            "observation_type",
        )
        role = coerce_enum(
            InputAzimuthRole,
            input_azimuth_role
            if input_azimuth_role is not None
            else getattr(self.config, "input_azimuth_role", InputAzimuthRole.RIGHT_LOOK_AWAY),
            "input_azimuth_role",
        )
        if look_side is None:
            look_side = infer_look_side(role, getattr(self.config, "look_side", LookSide.RIGHT))
        else:
            look_side = coerce_enum(LookSide, look_side, "look_side")
        if input_value_convention is None:
            input_value_convention = self.config.input_value_convention
        else:
            input_value_convention = coerce_enum(
                InputValueConvention,
                input_value_convention,
                "input_value_convention",
            )
        if wavelength is None:
            wavelength = self.wavelength if self.wavelength is not None else self.config.wavelength
        return SarObservationSpec(
            observation_type=observation_type,
            input_azimuth_role=role,
            look_side=look_side,
            input_value_convention=input_value_convention,
            wavelength=wavelength,
        )

    def convert_observation_values(self, values, spec):
        return _convert_observation_values(values, spec)

    def _coerce_projection_array(self, projection, values_size):
        projection = np.asarray(projection, dtype=float)
        if projection.shape == (3,):
            projection = np.broadcast_to(projection, (values_size, 3)).copy()
        elif projection.ndim >= 1 and projection.shape[-1] == 3 and projection.size == values_size * 3:
            projection = projection.reshape((values_size, 3))
        elif projection.ndim == 1 and projection.size == values_size * 3:
            projection = projection.reshape((values_size, 3))
        else:
            raise ValueError(
                "projection must have shape (3,), (N, 3), or values.shape + (3,); "
                f"got projection shape {projection.shape} for {values_size} values."
            )
        return projection

    def _read_converted_observation_with_projection_to_csi(
            self, data, lon, lat, projection, spec=None, downsample=1,
            zero2nan=True, verbose=None):
        if downsample <= 0:
            raise ValueError("downsample must be a positive integer.")
        data = np.asarray(data)
        values_size = data.size
        projection = self._coerce_projection_array(projection, values_size)

        if zero2nan:
            data = np.array(data, dtype=float, copy=True)
            data[data == 0] = np.nan

        data_flat = np.asarray(data).flatten()[::downsample]
        lon_flat = np.asarray(lon).flatten()[::downsample]
        lat_flat = np.asarray(lat).flatten()[::downsample]
        if lon_flat.size != data_flat.size or lat_flat.size != data_flat.size:
            raise ValueError("lon, lat, and values must have matching flattened sizes after downsampling.")
        i_zeros = np.flatnonzero(np.logical_or(data_flat != 0., np.logical_or(lon_flat != 0., lat_flat != 0.)))
        i_finite = np.flatnonzero(np.isfinite(data_flat))

        self.raw_projection_full = projection.reshape(values_size, 3)
        self.projection_downsampled_valid_index = np.intersect1d(i_zeros, i_finite)
        self.projection_raw_valid_index = (
            np.arange(values_size)[::downsample][self.projection_downsampled_valid_index]
        )
        self.observation_spec = spec
        self.read_from_binary(
            data,
            lon=lon,
            lat=lat,
            projection=projection,
            downsample=downsample,
        )
        self.projection_valid = self.los
        if self._is_verbose(verbose):
            self.print_input_summary()
        return data, projection

    def read_observation_to_csi(self, values, lon, lat, azimuth, incidence,
                                spec=None, downsample=1, zero2nan=True,
                                verbose=None, **spec_kwargs):
        spec = self.build_observation_spec(spec=spec, **spec_kwargs)
        data, projection = prepare_observation_for_csi(values, azimuth, incidence, spec)
        return self._read_converted_observation_with_projection_to_csi(
            data,
            lon=lon,
            lat=lat,
            projection=projection,
            spec=spec,
            downsample=downsample,
            zero2nan=zero2nan,
            verbose=verbose,
        )

    def read_observation_with_projection_to_csi(
            self, values, lon, lat, projection, spec=None, downsample=1,
            zero2nan=True, verbose=None, **spec_kwargs):
        """
        Convert scalar values and load them with a supplied ENU projection.

        This is the extension path for products that already provide ENU
        projection vectors. `projection` must be a 3-component ENU vector in
        the target positive-observation direction used after
        `SarObservationSpec` value conversion: toward satellite for LOS/range
        observations and along heading for azimuth observations. It may be a single `(3,)` vector, a flat
        `(N, 3)` array, or `values.shape + (3,)`.

        Value conversion follows `SarObservationSpec`: raw LOS/range values
        declared as `away_from_satellite` are sign-flipped into the
        toward-satellite target convention before loading into CSI.
        """
        spec = self.build_observation_spec(spec=spec, **spec_kwargs)
        data = self.convert_observation_values(values, spec)
        return self._read_converted_observation_with_projection_to_csi(
            data,
            lon=lon,
            lat=lat,
            projection=projection,
            spec=spec,
            downsample=downsample,
            zero2nan=zero2nan,
            verbose=verbose,
        )
    
    def to_xarray_dataarray(self):
        '''
        Also for GMT plot
        '''
        import xarray

        # Convert velocity data to an xarray DataArray with proper coordinates
        data_array = xarray.DataArray(self.vel, coords=[('lat', self.lat), ('lon', self.lon)], dims=['lat', 'lon'])
    
        return data_array
    
    def cut_raw_sar(self, lon_range, lat_range, inplace=False):
        """
        Cut the raw SAR data based on the given longitude and latitude ranges.

        Parameters:
        lon_range (list): The longitude range [min, max].
        lat_range (list): The latitude range [min, max].

        Returns:
        numpy.ndarray: The meshgrid of longitude.
        numpy.ndarray: The meshgrid of latitude.
        list: The coordinate range [lon_min, lon_max, lat_min, lat_max].
        """
        self._require_raw_grid(
            "cut_raw_sar()",
            fields=("raw_vel", "raw_lon", "raw_lat", "raw_mesh_lon", "raw_mesh_lat"),
        )
        # Cut the raw SAR data based on the given longitude and latitude ranges
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range
        lon_idx = np.where((self.raw_lon >= lon_min) & (self.raw_lon <= lon_max))[0]
        lat_idx = np.where((self.raw_lat >= lat_min) & (self.raw_lat <= lat_max))[0]
        if lon_idx.size == 0 or lat_idx.size == 0:
            raw_lon_min, raw_lon_max = np.nanmin(self.raw_lon), np.nanmax(self.raw_lon)
            raw_lat_min, raw_lat_max = np.nanmin(self.raw_lat), np.nanmax(self.raw_lat)
            raise ValueError(
                "Requested SAR cut range does not overlap the raw grid. "
                f"Requested lon={lon_range}, lat={lat_range}; "
                f"raw lon=[{raw_lon_min:g}, {raw_lon_max:g}], "
                f"raw lat=[{raw_lat_min:g}, {raw_lat_max:g}]."
            )
        mesh_lon = self.raw_mesh_lon[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
        mesh_lat = self.raw_mesh_lat[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
        rawsar = self.raw_vel[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
        coordrange = [mesh_lon.min(), mesh_lon.max(), mesh_lat.min(), mesh_lat.max()]
    
        if inplace:
            self.raw_vel = rawsar
            self.raw_mesh_lon = mesh_lon
            self.raw_mesh_lat = mesh_lat
            self.raw_lon = mesh_lon[0, :]
            self.raw_lat = mesh_lat[:, 0]
        return rawsar, mesh_lon, mesh_lat, coordrange
    
    def select_pixels(self, minlon, maxlon, minlat, maxlat):
        self.cut_raw_sar([minlon, maxlon], [minlat, maxlat], inplace=True)
        return super().select_pixels(minlon, maxlon, minlat, maxlat)

    def _coerce_plot_value_space(self, value_space):
        key = str(value_space).replace("-", "_").lower()
        aliases = {
            "raw": "raw",
            "input": "raw",
            "observation": "observation",
            "converted": "observation",
            "csi": "observation",
        }
        if key not in aliases:
            raise ValueError(
                "value_space must be 'observation' or 'raw' "
                f"(got {value_space!r})."
            )
        return aliases[key]

    def _plot_observation_spec(self, observation_type=None, input_azimuth_role=None,
                               look_side=None, input_value_convention=None,
                               wavelength=None):
        has_override = any(
            value is not None
            for value in (
                observation_type,
                input_azimuth_role,
                look_side,
                input_value_convention,
                wavelength,
            )
        )
        if not has_override and self.observation_spec is not None:
            return self.observation_spec
        return self.build_observation_spec(
            observation_type=observation_type,
            input_azimuth_role=input_azimuth_role,
            look_side=look_side,
            input_value_convention=input_value_convention,
            wavelength=wavelength,
        )

    def _raw_sar_values_for_plot(self, values, value_space="observation",
                                 observation_type=None, input_azimuth_role=None,
                                 look_side=None, input_value_convention=None,
                                 wavelength=None):
        value_space = self._coerce_plot_value_space(value_space)
        if value_space == "raw":
            return np.asarray(values), None

        spec = self._plot_observation_spec(
            observation_type=observation_type,
            input_azimuth_role=input_azimuth_role,
            look_side=look_side,
            input_value_convention=input_value_convention,
            wavelength=wavelength,
        )
        return self.convert_observation_values(values, spec), spec

    def _default_raw_sar_plot_label(self, value_space, spec, factor4plot):
        value_space = self._coerce_plot_value_space(value_space)
        if value_space == "raw":
            return "Raw value"

        if factor4plot == 100:
            unit = " (cm)"
        elif factor4plot == 1:
            unit = " (m)"
        else:
            unit = f" (x{factor4plot:g})"

        if spec is not None and spec.observation_type == ObservationType.AZIMUTH_OFFSET:
            return f"Azimuth offset{unit}"
        return f"LOS displacement{unit}"

    def _normalize_colorbar_orientation(self, orientation):
        key = str(orientation).replace("-", "_").lower()
        aliases = {
            "h": "horizontal",
            "horizontal": "horizontal",
            "v": "vertical",
            "vertical": "vertical",
        }
        if key not in aliases:
            raise ValueError(
                "colorbar_orientation must be 'horizontal' or 'vertical' "
                f"(got {orientation!r})."
            )
        return aliases[key]

    @staticmethod
    def _legacy_manual_colorbar_requested(colorbar_x, colorbar_y,
                                          colorbar_length, colorbar_height):
        defaults = {
            "colorbar_x": 0.1,
            "colorbar_y": 0.1,
            "colorbar_length": 0.4,
            "colorbar_height": 0.02,
        }
        values = {
            "colorbar_x": colorbar_x,
            "colorbar_y": colorbar_y,
            "colorbar_length": colorbar_length,
            "colorbar_height": colorbar_height,
        }
        return any(
            not np.isclose(float(values[name]), default)
            for name, default in defaults.items()
        )

    def _default_colorbar_pad(self, orientation, mode, loc):
        if mode == "inside" and orientation == "horizontal":
            if "lower" in loc or "bottom" in loc:
                return 0.10
            return 0.04
        if mode == "inside":
            return 0.04
        return 0.02

    def _resolve_colorbar_layout(self, orientation, mode="auto", loc=None,
                                 size=None, thickness=None, pad=None):
        orientation = self._normalize_colorbar_orientation(orientation)
        mode_key = str(mode).replace("-", "_").lower()
        aliases = {
            "auto": "auto",
            "inside": "inside",
            "inset": "inside",
            "outside": "outside",
            "external": "outside",
            "manual": "manual",
            "figure": "manual",
        }
        if mode_key not in aliases:
            raise ValueError(
                "colorbar_mode must be 'auto', 'inside', 'outside', or 'manual' "
                f"(got {mode!r})."
            )
        mode = aliases[mode_key]
        if mode == "auto":
            mode = "inside" if orientation == "horizontal" else "outside"

        if loc is None:
            if mode == "inside":
                loc = "lower left" if orientation == "horizontal" else "lower right"
            elif mode == "outside":
                loc = "bottom" if orientation == "horizontal" else "lower right"
            else:
                loc = "manual"
        loc = str(loc).replace("_", " ").replace("-", " ").lower()

        size = 0.4 if size is None else float(size)
        thickness = 0.025 if thickness is None else float(thickness)
        if pad is None:
            pad = self._default_colorbar_pad(orientation, mode, loc)
        else:
            pad = float(pad)
        if size <= 0.0 or thickness <= 0.0:
            raise ValueError("colorbar_size and colorbar_thickness must be positive.")
        if mode != "manual" and (size > 1.0 or thickness > 1.0):
            raise ValueError(
                "colorbar_size and colorbar_thickness are axes-relative fractions "
                "and must be <= 1 outside manual mode."
            )
        if pad < 0.0:
            raise ValueError("colorbar_pad must be non-negative.")

        return {
            "orientation": orientation,
            "mode": mode,
            "loc": loc,
            "size": size,
            "thickness": thickness,
            "pad": pad,
        }

    def _inside_colorbar_bounds(self, orientation, loc, size, thickness, pad):
        if orientation == "horizontal":
            width, height = size, thickness
            if "right" in loc:
                x0 = 1.0 - pad - width
            elif "center" in loc:
                x0 = (1.0 - width) / 2.0
            else:
                x0 = pad

            if "upper" in loc or "top" in loc:
                y0 = 1.0 - pad - height
            else:
                y0 = pad
        else:
            width, height = thickness, size
            if "left" in loc:
                x0 = pad
            else:
                x0 = 1.0 - pad - width

            if "upper" in loc or "top" in loc:
                y0 = 1.0 - pad - height
            elif "center" in loc:
                y0 = (1.0 - height) / 2.0
            else:
                y0 = pad

        return [x0, y0, width, height]

    def _outside_colorbar_bounds(self, ax, orientation, loc, size, thickness, pad):
        bbox = ax.get_position()
        if orientation == "vertical":
            width = thickness * bbox.width
            height = size * bbox.height
            if "left" in loc:
                x0 = bbox.x0 - pad * bbox.width - width
            else:
                x0 = bbox.x1 + pad * bbox.width

            if "upper" in loc or "top" in loc:
                y0 = bbox.y1 - height
            elif "center" in loc or "middle" in loc:
                y0 = bbox.y0 + (bbox.height - height) / 2.0
            else:
                y0 = bbox.y0
        else:
            width = size * bbox.width
            height = thickness * bbox.height
            if "right" in loc:
                x0 = bbox.x1 - width
            elif "left" in loc:
                x0 = bbox.x0
            else:
                x0 = bbox.x0 + (bbox.width - width) / 2.0

            if "upper" in loc or "top" in loc:
                y0 = bbox.y1 + pad * bbox.height
            else:
                y0 = bbox.y0 - pad * bbox.height - height

        return [x0, y0, width, height]

    def _make_colorbar_axes(self, fig, ax, layout,
                            colorbar_x=None, colorbar_y=None,
                            colorbar_length=None, colorbar_height=None):
        if layout["mode"] == "manual":
            return fig.add_axes([
                colorbar_x,
                colorbar_y,
                colorbar_length,
                colorbar_height,
            ])

        if layout["mode"] == "inside":
            bounds = self._inside_colorbar_bounds(
                layout["orientation"],
                layout["loc"],
                layout["size"],
                layout["thickness"],
                layout["pad"],
            )
            return ax.inset_axes(bounds, transform=ax.transAxes)

        bounds = self._outside_colorbar_bounds(
            ax,
            layout["orientation"],
            layout["loc"],
            layout["size"],
            layout["thickness"],
            layout["pad"],
        )
        return fig.add_axes(bounds)

    def _set_colorbar_label_position(self, cb, orientation, layout, cb_label_loc=None):
        if orientation == "vertical":
            if cb_label_loc is None:
                cb_label_loc = "left" if layout["mode"] == "inside" else "right"
            cb.ax.yaxis.set_label_position(cb_label_loc)
            cb.ax.yaxis.set_ticks_position(cb_label_loc)
            return

        if cb_label_loc is None:
            cb_label_loc = "bottom"
        cb.ax.xaxis.set_label_position(cb_label_loc)
        cb.ax.xaxis.set_ticks_position(cb_label_loc)
    
    # ------------------------Plotting-------------------------------#
    def plot_sar_values(self, coordrange=None, faults=None, rawdownsample4plot=100, factor4plot=100,
                        vmin=None, vmax=None, symmetry=True, cax=None, tickfontsize=10, labelfontsize=10,
                        style=['science'], fontsize=None, figsize=None, save_fig=False,
                        file_path='sar_values.png', dpi=300, show=True, cmap='cmc.roma_r',
                        trace_color='black', trace_linewidth=0.5, add_colorbar=True,
                        colorbar_length=0.4, colorbar_height=0.02, cb_label_loc=None,
                        colorbar_x=0.1, colorbar_y=0.1, colorbar_orientation='vertical',
                        colorbar_mode='auto', colorbar_loc=None, colorbar_pad=None,
                        colorbar_size=None, colorbar_thickness=None,
                        cb_label=None,
                        text=None, text_position=(0.05, 0.95), text_fontsize=12, text_color='black',
                        value_space='observation', observation_type=None,
                        input_azimuth_role=None, look_side=None,
                        input_value_convention=None, wavelength=None):
        """
        Plot SAR grid values for quick visual QC.

        By default this plots values after conversion to the same scalar
        observation convention used by CSI (`value_space="observation"`), with
        `factor4plot=100` so meter-based LOS values are shown in centimeters.
        Use `value_space="raw"` when you want to inspect product values exactly
        as they were read from the file.

        Common examples
        ---------------
        Plot converted observation values:

            sar.plot_sar_values()

        Plot raw product values before phase conversion or target-direction
        interpretation:

            sar.plot_sar_values(value_space="raw", factor4plot=1,
                                cb_label="Raw value")

        Use a vertical colorbar outside the right edge of the map axes:

            sar.plot_sar_values(colorbar_orientation="vertical")

        Save a quick-look image:

            sar.plot_sar_values(save_fig=True, file_path="sar_values.png",
                                dpi=300, show=False)

        Parameters
        ----------
        Value semantics:
            value_space : {"observation", "raw"}
                "observation" applies phase conversion and the reader's
                target-direction convention. "raw" plots product values
                before semantic conversion.
            observation_type, input_azimuth_role, look_side,
            input_value_convention, wavelength : optional
                One-call semantic overrides. Prefer mode/preset/config for
                repeated use; use these only when a single plot differs from
                the reader's configured convention.

        Display scaling:
            factor4plot : float
                Display multiplier only. Use 100 for meters-to-centimeters and
                1 for raw phase radians or already-scaled values.
            vmin, vmax, symmetry, cmap : optional
                Color scale controls. With symmetry=True, the color range is
                symmetric around zero.

        Sampling and extent:
            rawdownsample4plot : int
                Plot every Nth raw-grid pixel.
            coordrange : sequence, optional
                [lon_min, lon_max, lat_min, lat_max] subset to display.

        Colorbar:
            colorbar_orientation : {"horizontal", "vertical"}
                Horizontal defaults to an inside lower-left colorbar with tick
                labels below. Vertical defaults to an outside lower-right
                colorbar aligned with the map axis bottom edge, with labels on
                the right.
            colorbar_mode : {"auto", "inside", "outside", "manual"}
                Use "auto" for quick QC. Use "manual" only for final layout
                tuning with figure-coordinate colorbar_x/y/length/height.
            colorbar_loc, colorbar_size, colorbar_thickness, colorbar_pad :
                Axes-relative layout controls. If the default horizontal
                colorbar is crowded, first try colorbar_loc="upper left".
            cb_label, cb_label_loc, tickfontsize, labelfontsize : optional
                Colorbar text controls.

        Output and styling:
            save_fig, file_path, dpi, show : optional
                Output controls.
            cax, figsize, style, fontsize, faults, trace_color,
            trace_linewidth, text, text_position, text_fontsize, text_color :
                Advanced matplotlib/styling controls.
            colorbar_x, colorbar_y, colorbar_length, colorbar_height :
                Manual-mode figure-coordinate placement kept for compatibility.

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        # Check if raw SAR data exists
        if rawdownsample4plot <= 0:
            raise ValueError("rawdownsample4plot must be a positive integer.")
        self._require_raw_grid(
            "plot_sar_values()",
            fields=("raw_vel", "raw_mesh_lon", "raw_mesh_lat"),
        )
        if coordrange is not None:
            if len(coordrange) != 4:
                raise ValueError("coordrange must be [lon_min, lon_max, lat_min, lat_max].")
            lon_range, lat_range = coordrange[:2], coordrange[2:]
            rawsar, mesh_lon, mesh_lat, coordrange = self.cut_raw_sar(lon_range, lat_range)
        else:
            mesh_lon, mesh_lat = self.raw_mesh_lon, self.raw_mesh_lat
            rawsar = self.raw_vel
        rawsar, spec = self._raw_sar_values_for_plot(
            rawsar,
            value_space=value_space,
            observation_type=observation_type,
            input_azimuth_role=input_azimuth_role,
            look_side=look_side,
            input_value_convention=input_value_convention,
            wavelength=wavelength,
        )
        if coordrange is None and str(value_space).replace("-", "_").lower() != "raw":
            filter_index = getattr(self, "data_filter_raw_valid_index", None)
            if filter_index is not None:
                rawsar = np.array(rawsar, dtype=float, copy=True)
                flat = rawsar.reshape(-1)
                keep = np.zeros(flat.size, dtype=bool)
                filter_index = np.asarray(filter_index, dtype=int)
                filter_index = filter_index[(filter_index >= 0) & (filter_index < flat.size)]
                keep[filter_index] = True
                flat[~keep] = np.nan
        if cb_label is None:
            cb_label = self._default_raw_sar_plot_label(value_space, spec, factor4plot)
        rawsar = rawsar[::rawdownsample4plot, ::rawdownsample4plot] * factor4plot
        mesh_lon = mesh_lon[::rawdownsample4plot, ::rawdownsample4plot]
        mesh_lat = mesh_lat[::rawdownsample4plot, ::rawdownsample4plot]
        extent = coordrange if coordrange is not None else [mesh_lon.min(), mesh_lon.max(), mesh_lat.min(), mesh_lat.max()]
        rvmax = vmax if vmax is not None else np.nanmax(rawsar)
        rvmin = vmin if vmin is not None else np.nanmin(rawsar)
    
        # Set the color scaling
        if symmetry:
            vmax = max(abs(rvmin), rvmax)
            vmin = -vmax
        else:
            vmax, vmin = rvmax, rvmin
    
        # Determine the origin of the plot
        if mesh_lat[0, 0] > mesh_lat[0, -1]:
            origin = 'lower'
        else:
            origin = 'upper'
    
        # Set the plotting style
        with sci_plot_style(style=style, fontsize=fontsize, figsize=figsize):
    
            # Create the figure and axes
            if cax is None:
                fig, ax = plt.subplots(1, 1) # , tight_layout=True
            else:
                fig = plt.gcf()
                ax = cax
    
            # Plot the SAR values
            # im = ax.imshow(rawsar, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, extent=extent)
            im = ax.pcolormesh(mesh_lon, mesh_lat, rawsar, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
            set_degree_formatter(ax, axis='both')
    
            # Plot faults
            if faults is not None:
                for fault in faults:
                    if isinstance(fault, pd.DataFrame):
                        ax.plot(fault.lon.values, fault.lat.values, color=trace_color, lw=trace_linewidth)
                    else:
                        ax.plot(fault.lon, fault.lat, color=trace_color, lw=trace_linewidth)
            if coordrange is not None:
                ax.set_xlim(*lon_range)
                ax.set_ylim(*lat_range)

            # Set aspect before placing axes-relative colorbars.
            ax.set_aspect('equal', adjustable='box')
    
            # Add colorbar
            if add_colorbar:
                colorbar_layout_mode = colorbar_mode
                if (
                    str(colorbar_mode).replace("-", "_").lower() == "auto"
                    and self._legacy_manual_colorbar_requested(
                        colorbar_x,
                        colorbar_y,
                        colorbar_length,
                        colorbar_height,
                    )
                ):
                    warnings.warn(
                        "colorbar_x/colorbar_y/colorbar_length/colorbar_height "
                        "now belong to colorbar_mode='manual'. Using manual "
                        "colorbar placement for this call because those legacy "
                        "parameters differ from their defaults. For axes-relative "
                        "layout, use colorbar_size/colorbar_thickness/colorbar_pad.",
                        UserWarning,
                        stacklevel=2,
                    )
                    colorbar_layout_mode = "manual"

                colorbar_layout = self._resolve_colorbar_layout(
                    colorbar_orientation,
                    mode=colorbar_layout_mode,
                    loc=colorbar_loc,
                    size=colorbar_size,
                    thickness=colorbar_thickness,
                    pad=colorbar_pad,
                )
                colorbar_orientation = colorbar_layout["orientation"]
                fig.canvas.draw()
                cbar_ax = self._make_colorbar_axes(
                    fig,
                    ax,
                    colorbar_layout,
                    colorbar_x=colorbar_x,
                    colorbar_y=colorbar_y,
                    colorbar_length=colorbar_length,
                    colorbar_height=colorbar_height,
                )
                cb = fig.colorbar(im, cax=cbar_ax, orientation=colorbar_orientation)
                cb.ax.tick_params(labelsize=tickfontsize)
                cb.set_label(cb_label, fontdict={'size': labelfontsize})
                self._set_colorbar_label_position(
                    cb,
                    colorbar_orientation,
                    colorbar_layout,
                    cb_label_loc=cb_label_loc,
                )
    
            # Add text
            if text is not None:
                ax.text(text_position[0], text_position[1], text, transform=ax.transAxes,
                        fontsize=text_fontsize, color=text_color, verticalalignment='top')
    
            # Save or show the figure
            if save_fig:
                plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
            if show:
                plt.show()
            elif cax is None:
                plt.close()
    
        return fig, ax

    def remove_orbit_error(self, order=1, exclude_range=None, use_raw=False):
        """
        Remove orbital error from velocity data.
        
        Parameters:
        order (int): The order of the polynomial to fit (1 for linear, 2 for quadratic).
        exclude_range (tuple): Optional. A tuple of (lon_min, lon_max, lat_min, lat_max) to exclude from fitting.
        use_raw (bool): Whether to use raw velocity data (raw_vel) or processed velocity data (vel).
        
        Returns:
        np.ndarray: The velocity data with orbital error removed.
        """
        if use_raw and hasattr(self, 'raw_vel'):
            vel = self.raw_vel.flatten()
            x = self.raw_lon.flatten()
            y = self.raw_lat.flatten()
            x, y = self.ll2xy(x, y)
        else:
            vel = self.vel
            x = self.x
            y = self.y
    
        if exclude_range:
            lon_min, lon_max, lat_min, lat_max = exclude_range
            xmin, ymin = self.ll2xy(lon_min, lat_min)
            xmax, ymax = self.ll2xy(lon_max, lat_max)
            mask = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)
            x_fit = x[mask]
            y_fit = y[mask]
            vel_fit = vel[mask]
        else:
            x_fit = x
            y_fit = y
            vel_fit = vel
    
        # Fit a polynomial surface
        if order == 1:
            # Linear fit
            A = np.c_[x_fit, y_fit, np.ones(x_fit.shape)]
        elif order == 2:
            # Quadratic fit
            A = np.c_[x_fit**2, y_fit**2, x_fit*y_fit, x_fit, y_fit, np.ones(x_fit.shape)]
        else:
            raise ValueError("Order must be 1 (linear) or 2 (quadratic)")
    
        # Solve for the coefficients
        coeff, _, _, _ = np.linalg.lstsq(A, vel_fit, rcond=None)
    
        # Create the fitted surface
        if order == 1:
            fitted_surface = coeff[0]*x + coeff[1]*y + coeff[2]
        elif order == 2:
            fitted_surface = coeff[0]*x**2 + coeff[1]*y**2 + coeff[2]*x*y + coeff[3]*x + coeff[4]*y + coeff[5]
    
        # Subtract the fitted surface from the original velocity data
        vel_corrected = vel - fitted_surface
        if use_raw and hasattr(self, 'raw_vel'):
            self.raw_vel = vel_corrected.reshape(self.raw_vel.shape)
        else:
            self.vel = vel_corrected
    
        if order == 1:
            return coeff[0], coeff[1], coeff[2]
        elif order == 2:
            return coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5]
