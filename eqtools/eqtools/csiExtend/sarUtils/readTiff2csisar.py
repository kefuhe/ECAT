import numpy as np
import os
import warnings

from .readBase2csisar import ReadBase2csisar, Hyp3TiffConfig, GammaTiffConfig
from .readTiffUtils import read_tiff, read_tiff_info, utm_to_latlon
from .sar_conventions import ObservationType, coerce_enum

class TiffsarReader(ReadBase2csisar):
    """
    Base reader for SAR GeoTIFF-style products.

    Users normally instantiate `GammaTiffReader` or `Hyp3TiffReader` with a
    short mode such as `unwrapped_phase`, `los_displacement`, or
    `range_offset`. Full preset names supported by the concrete reader remain
    available for reproducible configs; use `config=...` for custom products.
    """
    config_cls = Hyp3TiffConfig
    mode_presets = {
        "unwrapped_phase": "hyp3_unwrapped_phase",
        "phase_los": "hyp3_unwrapped_phase",
        "los": "hyp3_los_displacement",
        "los_displacement": "hyp3_los_displacement",
    }

    def __init__(self, name=None, utmzone=None, lon0=None, lat0=None,
                 directory_name='.', config=None, preset=None, mode=None,
                 verbose=False):
        super().__init__(
            name,
            utmzone=utmzone,
            lon0=lon0,
            lat0=lat0,
            directory_name=directory_name,
            config=config,
            preset=preset,
            mode=mode,
            verbose=verbose,
        )

    def extract_raw_grd(self, directory_name=None, prefix=None, phsname=None,
                        azifile=None, incfile=None, phase_band=1, azi_band=1, inc_band=1, factor_to_m=1.0,
                        zero2nan=None, wavelength=None, azimuth_reference=None,
                        azimuth_unit=None, azimuth_direction=None,
                        incidence_reference=None, incidence_unit=None, is_lonlat=None, verbose=None):
        """
        Extract SAR value, azimuth, incidence, and coordinate grids.

        Presets or config values should be set before calling this method
        because angle conventions are applied here. The value raster itself is
        only scaled by `factor_to_m`; semantic conversion to CSI observation
        values happens later in `read_observation()`.
    
        Parameters:
        directory_name (str, optional): The directory name where the raw images are stored. Defaults to None.
        prefix (str, optional): The prefix for the output files. Defaults to None.
        phsname (str, optional): The name of the phase file. Defaults to None.
        azifile (str, optional): The name of the azimuth file. Defaults to None.
        incfile (str, optional): The name of the incidence file. Defaults to None.
        phase_band (int, optional): Band index to read from the phase file. Defaults to 1.
        azi_band (int, optional): Band index to read from the azimuth file. Defaults to 1.
        inc_band (int, optional): Band index to read from the incidence file. Defaults to 1.
        factor_to_m (float, optional): Multiplier applied to the value raster
            immediately after reading. For unwrapped phase rasters, usually
            keep 1.0 so values remain radians. For displacement rasters, use it
            to convert the file unit to meters, for example 0.01 for cm.
        zero2nan (bool, optional): Whether to convert zero values to NaN. Defaults to config value.
        wavelength (float, optional): The wavelength of the SAR signal. Defaults to config value.
        azimuth_reference (str, optional): The reference direction for azimuth. Defaults to config value.
        azimuth_unit (str, optional): The unit of the azimuth angle. Defaults to config value.
        azimuth_direction (str, optional): The rotation direction for azimuth. Defaults to config value.
        incidence_reference (str, optional): The reference direction for incidence. Defaults to config value.
        incidence_unit (str, optional): The unit of the incidence angle. Defaults to config value.
        is_lonlat (bool, optional): Whether the data is in longitude and latitude. Defaults to config value.
    
        Returns:
        None
        """
        # Use config values if parameters are not provided
        zero2nan = zero2nan if zero2nan is not None else self.config.zero2nan
        wavelength = wavelength if wavelength is not None else self.config.wavelength
        azimuth_reference = azimuth_reference if azimuth_reference is not None else self.config.azimuth_reference
        azimuth_unit = azimuth_unit if azimuth_unit is not None else self.config.azimuth_unit
        azimuth_direction = azimuth_direction if azimuth_direction is not None else self.config.azimuth_direction
        incidence_reference = incidence_reference if incidence_reference is not None else self.config.incidence_reference
        incidence_unit = incidence_unit if incidence_unit is not None else self.config.incidence_unit
        is_lonlat = is_lonlat if is_lonlat is not None else self.config.is_lonlat
        self._set_raw_angle_convention(
            azimuth_reference=azimuth_reference,
            azimuth_unit=azimuth_unit,
            azimuth_direction=azimuth_direction,
            incidence_reference=incidence_reference,
            incidence_unit=incidence_unit,
        )
    
        # Extract SAR raw images
        if directory_name is not None:
            self.directory_name = directory_name
        else:
            directory_name = self.directory_name

        observation_type = coerce_enum(
            ObservationType,
            self.config.observation_type,
            "observation_type",
        )
        if observation_type == ObservationType.PHASE_LOS and not np.isclose(float(factor_to_m), 1.0):
            warnings.warn(
                "factor_to_m is applied before phase_los conversion. Keep "
                "factor_to_m=1.0 for unwrapped phase rasters unless the file "
                "values are intentionally pre-scaled phase units.",
                UserWarning,
                stacklevel=2,
            )
    
        # Construct file paths
        if prefix:
            phase_file, azi_file, inc_file = self._construct_file_paths(directory_name, prefix)
        else:
            if not (phsname and azifile and incfile):
                raise ValueError(
                    "phsname, azifile, and incfile are required when prefix is not provided."
                )
            phase_file = os.path.join(directory_name, phsname)
            azi_file = os.path.join(directory_name, azifile)
            inc_file = os.path.join(directory_name, incfile)
    
        # Read SAR data from specified bands
        vel, azi, inc, x_lon, x_lat, mesh_lon_x, mesh_lat_y, im_geotrans, im_proj = self._read_sar_data(
            phase_file, azi_file, inc_file, phase_band=phase_band, azi_band=azi_band, inc_band=inc_band, factor_to_m=factor_to_m
        )
    
        self.raw_azimuth_input = azi
        self.raw_incidence_input = inc
    
        # Process azimuth and incidence angles
        azi = self.normalize_azimuth(azi, azimuth_reference, azimuth_unit, azimuth_direction)
        inc = self.normalize_incidence(inc, incidence_reference, incidence_unit)
    
        # Transfer coordinate system from UTM to latitude and longitude
        if not is_lonlat:
            im_height, im_width = mesh_lon_x.shape
            mesh_lat, mesh_lon = utm_to_latlon(mesh_lon_x.flatten(), mesh_lat_y.flatten(), im_proj)
            mesh_lat = mesh_lat.reshape(im_height, im_width)
            mesh_lon = mesh_lon.reshape(im_height, im_width)
        else:
            mesh_lon = mesh_lon_x
            mesh_lat = mesh_lat_y
    
        if zero2nan:
            vel[vel == 0] = np.nan
    
        # Save in self
        self.wavelength = wavelength
        self.raw_vel = vel
        self.raw_azimuth_enu = azi
        self.raw_azimuth_role = self.config.input_azimuth_role
        self.raw_incidence = inc
        self.raw_lon = mesh_lon.mean(axis=0)
        self.raw_lat = mesh_lat.mean(axis=1)
        self.raw_mesh_lon = mesh_lon
        self.raw_mesh_lat = mesh_lat
        self.im_geotrans = im_geotrans
        self.im_proj = im_proj
        if self._is_verbose(verbose):
            self.print_input_summary()

    def _construct_file_paths(self, directory_name, prefix):
        phase_file = self._single_file_match(
            os.path.join(directory_name, f'{prefix}*.tif'),
            "SAR value TIFF",
            exclude_suffixes=(".azi.tif", ".inc.tif"),
        )
        azi_file = self._single_file_match(
            os.path.join(directory_name, f'{prefix}*.azi.tif'),
            "SAR azimuth TIFF",
        )
        inc_file = self._single_file_match(
            os.path.join(directory_name, f'{prefix}*.inc.tif'),
            "SAR incidence TIFF",
        )
        return phase_file, azi_file, inc_file

    def _read_sar_data(self, phase_file, azi_file, inc_file, phase_band=1, azi_band=1, inc_band=1, factor_to_m=1.0):
        """
        Read SAR value, azimuth, and incidence rasters from TIFF files.
    
        Parameters:
            phase_file (str): Path to the value raster. It may contain
                unwrapped phase, LOS/range displacement, or azimuth offset
                depending on the reader preset/config.
            azi_file (str): Path to the azimuth file (TIFF format).
            inc_file (str): Path to the incidence file (TIFF format).
            phase_band (int, optional): Band index to read from the phase file. Defaults to 1.
            azi_band (int, optional): Band index to read from the azimuth file. Defaults to 1.
            inc_band (int, optional): Band index to read from the incidence file. Defaults to 1.
            factor_to_m (float, optional): Raw multiplier applied to the value
                raster. Leave phase rasters in radians; scale displacement
                rasters to meters.
    
        Returns:
            tuple:
                - vel (numpy.ndarray): Raw value matrix after `factor_to_m`.
                - azi (numpy.ndarray): Azimuth data matrix.
                - inc (numpy.ndarray): Incidence data matrix.
                - x_lon (numpy.ndarray): Longitude coordinates.
                - x_lat (numpy.ndarray): Latitude coordinates.
                - mesh_lon_x (numpy.ndarray): Projected x mesh grid along Longitude direction.
                - mesh_lat_y (numpy.ndarray): Projected y mesh grid along Latitude direction.
                - im_geotrans (tuple): Geotransform (origin and pixel size).
                - im_proj (str): Projection parameters.
        """
        # Read phase data
        vel, im_geotrans, im_proj, im_width, im_height = read_tiff(phase_file, band_index=phase_band, factor=factor_to_m)
        # Read azimuth data
        azi, _, _, _, _ = read_tiff(azi_file, band_index=azi_band)
        # Read incidence data
        inc, _, _, _, _ = read_tiff(inc_file, band_index=inc_band)
    
        # Read metadata information
        x_lon, _, x_lat, _, mesh_lon_x, mesh_lat_y = read_tiff_info(phase_file, im_width, im_height)
    
        return vel, azi, inc, x_lon, x_lat, mesh_lon_x, mesh_lat_y, im_geotrans, im_proj

    def read_observation(self, downsample=1,
                         zero2nan=True, wavelength=None, observation_type=None,
                         input_azimuth_role=None, look_side=None,
                         input_value_convention=None, verbose=None):
        """
        Convert the extracted raw TIFF grids into CSI observation arrays.

        If no overrides are supplied, the reader config or preset determines
        whether `raw_vel` is treated as unwrapped phase, LOS/range
        displacement, or azimuth offset. Use `wavelength` for
        `phase_los`; use `input_value_convention` only when a product's sign
        convention differs from the selected preset.
        """
        self._require_raw_grid("read_observation()")
        vel = self.raw_vel
        mesh_lon = self.raw_mesh_lon
        mesh_lat = self.raw_mesh_lat

        self.read_observation_to_csi(
            vel,
            lon=mesh_lon.flatten(),
            lat=mesh_lat.flatten(),
            azimuth=self.raw_azimuth_enu.flatten(),
            incidence=self.raw_incidence.flatten(),
            downsample=downsample,
            zero2nan=zero2nan,
            observation_type=observation_type,
            input_azimuth_role=input_azimuth_role,
            look_side=look_side,
            input_value_convention=input_value_convention,
            wavelength=wavelength,
            verbose=verbose,
        )


class Hyp3TiffReader(TiffsarReader):
    """
    Reader for HyP3 GeoTIFF products.

    The default config matches `hyp3_los_displacement`. Use
    `mode="unwrapped_phase"` or preset `hyp3_unwrapped_phase` when the HyP3
    geometry rasters are paired with an unwrapped phase value raster. Use an
    explicit config if a HyP3-derived product changes angle units or value sign
    convention.
    """
    config_cls = Hyp3TiffConfig

class GammaTiffReader(TiffsarReader):
    """
    Reader for GAMMA GeoTIFF products.

    Use `mode="unwrapped_phase"` for unwrapped phase rasters,
    `mode="los_displacement"` for LOS displacement rasters, and
    `mode="range_offset"` for range-offset rasters. Full Gamma TIFF preset
    names remain available when exact reproducibility in config files is
    preferred. Angle convention changes must be configured before
    `extract_raw_grd()`.
    """
    config_cls = GammaTiffConfig
    mode_presets = {
        "unwrapped_phase": "gamma_tiff_unwrapped_phase",
        "phase_los": "gamma_tiff_unwrapped_phase",
        "los": "gamma_tiff_los_displacement",
        "los_displacement": "gamma_tiff_los_displacement",
        "range": "gamma_tiff_range_offset",
        "range_offset": "gamma_tiff_range_offset",
        "az": "gamma_tiff_azimuth_offset",
        "azimuth": "gamma_tiff_azimuth_offset",
        "azimuth_offset": "gamma_tiff_azimuth_offset",
    }


if __name__ == '__main__':
    pass
    # plt.imshow(los, interpolation='none', # cmap=cmap, norm=norm,
    #          origin='upper', extent=[np.nanmin(x_lon), np.nanmax(x_lon), np.nanmin(x_lat), np.nanmax(x_lat)], aspect='auto', vmax=1, vmin=-1)
