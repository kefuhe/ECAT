from osgeo import gdal, osr
import xarray
import numpy as np
import os
import pandas as pd
import warnings
from csi.insar import insar
from .readBase2csisar import ReadBase2csisar, GammasarConfig
from .readTiffUtils import save_to_tiff


class GammasarReader(ReadBase2csisar):
    """
    Reader for GAMMA binary SAR products with `.phs`, `.azi`, `.inc`, and
    `.rsc` files.

    Use short modes such as `unwrapped_phase`, `los_displacement`,
    `range_offset`, or `azimuth_offset` for normal use. Full presets such as
    `gamma_unwrapped_phase` remain available for explicit configuration files.
    Presets from other reader families are rejected; use `config=...` for
    custom conventions.
    """
    config_cls = GammasarConfig
    mode_presets = {
        "unwrapped_phase": "gamma_unwrapped_phase",
        "phase_los": "gamma_unwrapped_phase",
        "los": "gamma_los_displacement",
        "los_displacement": "gamma_los_displacement",
        "range": "gamma_range_offset",
        "range_offset": "gamma_range_offset",
        "az": "gamma_azimuth_offset",
        "azimuth": "gamma_azimuth_offset",
        "azimuth_offset": "gamma_azimuth_offset",
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
    
    def extract_raw_grd(self, directory_name=None, prefix=None, phsname=None, rscname=None,
                        azifile=None, incfile=None, zero2nan=None, wavelength=None,
                        azimuth_reference=None, azimuth_unit=None,
                        azimuth_direction=None, incidence_reference=None,
                        incidence_unit=None, verbose=None):
        """
        Extract GAMMA value, azimuth, incidence, and coordinate grids.

        Presets or config values should be set before calling this method
        because angle conventions are applied here. Raw value conversion to the
        CSI scalar-observation convention happens later in `read_observation()`.

        Parameters:
        directory_name (str, optional): The directory name where the raw images are stored. Defaults to None.
        prefix (str, optional): The prefix for the output files. Defaults to None.
        phsname (str, optional): The name of the phase file. Defaults to None.
        rscname (str, optional): The name of the resource file. Defaults to None.
        azifile (str, optional): The name of the azimuth file. Defaults to None.
        incfile (str, optional): The name of the incidence file. Defaults to None.
        zero2nan (bool, optional): Whether to convert zero values to NaN. Defaults to config value.
        wavelength (float, optional): Expected wavelength of the SAR signal. GAMMA
            `.rsc` WAVELENGTH is used for the loaded data; a supplied value is
            checked against it and warned on mismatch.
        azimuth_reference (str, optional): The reference direction for azimuth. Defaults to config value.
        azimuth_unit (str, optional): The unit of the azimuth angle. Defaults to config value.
        azimuth_direction (str, optional): The rotation direction for azimuth. Defaults to config value.
        incidence_reference (str, optional): The reference direction for incidence. Defaults to config value.
        incidence_unit (str, optional): The unit of the incidence angle. Defaults to config value.
        Returns:
        None
        """
        # Use config values if parameters are not provided
        zero2nan = zero2nan if zero2nan is not None else self.config.zero2nan
        requested_wavelength = wavelength
        wavelength = wavelength if wavelength is not None else self.config.wavelength
        azimuth_reference = azimuth_reference if azimuth_reference is not None else self.config.azimuth_reference
        azimuth_unit = azimuth_unit if azimuth_unit is not None else self.config.azimuth_unit
        azimuth_direction = azimuth_direction if azimuth_direction is not None else self.config.azimuth_direction
        incidence_reference = incidence_reference if incidence_reference is not None else self.config.incidence_reference
        incidence_unit = incidence_unit if incidence_unit is not None else self.config.incidence_unit
        self._set_raw_angle_convention(
            azimuth_reference=azimuth_reference,
            azimuth_unit=azimuth_unit,
            azimuth_direction=azimuth_direction,
            incidence_reference=incidence_reference,
            incidence_unit=incidence_unit,
        )

        # extract sar raw images 
        if directory_name is not None:
            self.directory_name = directory_name
        else:
            directory_name = self.directory_name
        # phase file
        if prefix is not None:
            phase_file, rsc_file, azi_file, inc_file = self._construct_file_paths(directory_name, prefix)
        else:
            if not (phsname and rscname and azifile and incfile):
                raise ValueError(
                    "phsname, rscname, azifile, and incfile are required when prefix is not provided."
                )
            phase_file = os.path.join(directory_name, phsname)
            rsc_file = os.path.join(directory_name, rscname)
            azi_file = os.path.join(directory_name, azifile)
            inc_file = os.path.join(directory_name, incfile)

        rsc = pd.read_csv(rsc_file, sep=r'\s+', names=['name', 'value'])
        rsc.set_index('name', inplace=True)

        # Raster dimensions from GAMMA resource metadata.
        nrow, ncol = [int(rsc.loc['WIDTH', 'value']), int(rsc.loc['FILE_LENGTH', 'value'])]
        rsc_wavelength = float(rsc.loc['WAVELENGTH', 'value'])
        if requested_wavelength is not None and not np.isclose(float(requested_wavelength), rsc_wavelength):
            warnings.warn(
                "The supplied wavelength differs from the GAMMA .rsc WAVELENGTH; "
                f"using the .rsc value {rsc_wavelength:g} m.",
                UserWarning,
                stacklevel=2,
            )
        wavelength = rsc_wavelength
        dtype = np.float32
        # Raw value raster. Its physical meaning is defined by the preset/config.
        phs = np.fromfile(phase_file, dtype=dtype).reshape(ncol, nrow)
        los = phs

        # Raw azimuth and incidence rasters.
        azi = np.fromfile(azi_file, dtype=dtype).reshape(ncol, nrow)
        inc = np.fromfile(inc_file, dtype=dtype).reshape(ncol, nrow)

        # Geographic coordinates from GAMMA resource metadata.
        x_first = np.float32(rsc.loc['X_FIRST', 'value'])
        y_first = np.float32(rsc.loc['Y_FIRST', 'value'])

        x_step = np.float32(rsc.loc['X_STEP', 'value'])
        y_step = np.float32(rsc.loc['Y_STEP', 'value'])
        lon = np.arange(x_first, x_first + x_step*nrow, x_step)[:nrow]
        lat = np.arange(y_first, y_first + y_step*ncol, y_step)[:ncol]

        self.raw_azimuth_input = azi
        self.raw_incidence_input = inc
        # Process azimuth and incidence angles
        azi = self.normalize_azimuth(azi, azimuth_reference, azimuth_unit, azimuth_direction)
        inc = self.normalize_incidence(inc, incidence_reference, incidence_unit)

        if zero2nan:
            los[los == 0] = np.nan

        # Save in self object
        self.phase_file = phase_file
        self.rsc_file = rsc_file
        self.azi_file = azi_file
        self.inc_file = inc_file
        self.wavelength = wavelength
        self.raw_vel = los
        self.raw_azimuth_enu = azi
        self.raw_azimuth_role = self.config.input_azimuth_role
        self.raw_incidence = inc
        self.raw_lon = lon
        self.raw_lat = lat
        mesh_lon, mesh_lat = np.meshgrid(lon, lat)
        self.raw_mesh_lon = mesh_lon
        self.raw_mesh_lat = mesh_lat
        if self._is_verbose(verbose):
            self.print_input_summary()
    
    def _construct_file_paths(self, directory_name, prefix):
        phase_file = self._single_file_match(
            os.path.join(directory_name, prefix + '*.phs'),
            "GAMMA value file",
        )
        rsc_file = self._single_file_match(
            os.path.join(directory_name, prefix + '*.phs.rsc'),
            "GAMMA resource file",
        )
        azi_file = self._single_file_match(
            os.path.join(directory_name, prefix + '*.azi'),
            "GAMMA azimuth file",
        )
        inc_file = self._single_file_match(
            os.path.join(directory_name, prefix + '*.inc'),
            "GAMMA incidence file",
        )
        return phase_file, rsc_file, azi_file, inc_file

    def read_observation(self, downsample=1, zero2nan=True, wavelength=None,
                         observation_type=None, input_azimuth_role=None,
                         look_side=None, input_value_convention=None, verbose=None):
        """
        Convert extracted GAMMA grids into CSI observation arrays.

        The selected preset/config determines whether `raw_vel` is interpreted
        as unwrapped phase, LOS/range displacement, or azimuth offset. Override
        the semantic fields only when the product documentation differs from
        the selected preset.
        
        Parameters:
        -----------
        downsample : int, optional
            Downsample factor for the data. Default is 1 (no downsampling).
        zero2nan : bool, optional
            Whether to convert zero values to NaN. Default is True.
        wavelength : float, optional
            The wavelength to use for conversion. Default is None (use self.wavelength).
        observation_type : str or ObservationType, optional
            Explicit observation type: "phase_los", "los_displacement", or
            "azimuth_offset". If omitted, config.observation_type is used.
        input_azimuth_role : str or InputAzimuthRole, optional
            Meaning of the input azimuth raster after angle normalization.
        look_side : str or LookSide, optional
            Right/left looking geometry for phase_los and los_displacement
            when the azimuth role does not already imply it.
        input_value_convention : str or InputValueConvention, optional
            Raw value sign convention. phase_los accepts only
            "unwrapped_phase"; los_displacement accepts "toward_satellite" or
            "away_from_satellite"; azimuth_offset accepts "along_heading" or
            "opposite_heading".
        """
        self._require_raw_grid(
            "read_observation()",
            fields=("raw_vel", "raw_lon", "raw_lat", "raw_azimuth_enu", "raw_incidence"),
        )
        # Generate meshgrid for longitude and latitude
        Lon, Lat = np.meshgrid(self.raw_lon, self.raw_lat)

        self.read_observation_to_csi(
            self.raw_vel,
            lon=Lon.flatten(),
            lat=Lat.flatten(),
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

    def save_outputs_as_tiff(self, directory_name, save_azi=False, save_inc=False,
                             save_vel=True, observation_type=None,
                             input_value_convention=None, wavelength=None,
                             extent=None, grid_resolution=None):
        """
        Save azi, inc, and vel as GeoTIFF files, with optional resampling to a regular grid and user-defined extent.
    
        Parameters:
        - directory_name: str, directory to save the TIFF files.
        - save_azi: bool, whether to save azimuth data as TIFF.
        - save_inc: bool, whether to save incidence data as TIFF.
        - save_vel: bool, whether to save velocity data as TIFF.
        - observation_type: SAR observation type used for value conversion.
        - input_value_convention: Sign convention of the raw input values.
        - wavelength: Wavelength used for phase conversion.
        - extent: tuple, optional, geographic extent to define the regular grid (min_lon, max_lon, min_lat, max_lat).
        - grid_resolution: float, optional, resolution of the grid in degrees (e.g., 0.01 for ~1 km grid spacing).
        """
        def resample_to_regular_grid(data, lon, lat, extent, resolution):
            """
            Resample data to a regular grid.
    
            Parameters:
            - data: numpy.ndarray, 2D array to resample.
            - lon: numpy.ndarray, longitude array.
            - lat: numpy.ndarray, latitude array.
            - extent: tuple, geographic extent (min_lon, max_lon, min_lat, max_lat).
            - resolution: float, resolution of the grid in degrees.
    
            Returns:
            - resampled_data: numpy.ndarray, resampled 2D array.
            - grid_lon: numpy.ndarray, longitude of the regular grid.
            - grid_lat: numpy.ndarray, latitude of the regular grid.
            """
            from scipy.interpolate import griddata
    
            # Flatten the input data
            lon = np.asarray(lon)
            lat = np.asarray(lat)
            if lon.ndim == 1 and lat.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
            points = np.array([lon.flatten(), lat.flatten()]).T
            values = data.flatten()
    
            # Define the regular grid based on the extent and resolution
            min_lon, max_lon, min_lat, max_lat = extent
            grid_lon = np.arange(min_lon, max_lon + resolution, resolution)
            grid_lat = np.arange(max_lat, min_lat + resolution, -resolution)
            grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
            # Interpolate data to the regular grid
            resampled_data = griddata(points, values, (grid_lon, grid_lat), method='nearest')
    
            return resampled_data, grid_lon, grid_lat
    
        # Save azimuth data
        if save_azi:
            azi_data = np.array(self.raw_azimuth_input, copy=True)
            azi_lon, azi_lat = self.raw_lon, self.raw_lat
            if grid_resolution and extent:
                azi_data, azi_lon, azi_lat = resample_to_regular_grid(azi_data, azi_lon, azi_lat, extent, grid_resolution)
            azi_data -= 90.0
            azi_file = os.path.join(directory_name, 'azi.tif')
            save_to_tiff(azi_data, azi_lon, azi_lat, azi_file)
    
        # Save incidence data
        if save_inc:
            inc_data = np.array(self.raw_incidence_input, copy=True)
            inc_lon, inc_lat = self.raw_lon, self.raw_lat
            if grid_resolution and extent:
                inc_data, inc_lon, inc_lat = resample_to_regular_grid(inc_data, inc_lon, inc_lat, extent, grid_resolution)
            inc_file = os.path.join(directory_name, 'inc.tif')
            save_to_tiff(inc_data, inc_lon, inc_lat, inc_file)
    
        # Save velocity data
        if save_vel:
            vel_data = np.array(self.raw_vel, copy=True)
            vel_lon, vel_lat = self.raw_lon, self.raw_lat
            spec = self.build_observation_spec(
                observation_type=observation_type,
                input_value_convention=input_value_convention,
                wavelength=wavelength,
            )
            vel_data = self.convert_observation_values(vel_data, spec)
            if grid_resolution and extent:
                vel_data, vel_lon, vel_lat = resample_to_regular_grid(vel_data, vel_lon, vel_lat, extent, grid_resolution)
            vel_file = os.path.join(directory_name, 'disp.tif')
            save_to_tiff(vel_data, vel_lon, vel_lat, vel_file)


if __name__ == '__main__':
    pass
