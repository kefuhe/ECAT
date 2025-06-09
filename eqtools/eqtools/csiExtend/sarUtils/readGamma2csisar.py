from osgeo import gdal, osr
import xarray
import numpy as np
from glob import glob
import os
import pandas as pd
from csi.insar import insar
from .readBase2csisar import ReadBase2csisar, GammasarConfig
from .readTiffUtils import save_to_tiff


class GammasarReader(ReadBase2csisar):
    def __init__(self, name=None, utmzone=None, lon0=None, lat0=None, directory_name='.', config=None):
        super().__init__(name, utmzone=utmzone, lon0=lon0, lat0=lat0)
        self.directory_name = directory_name
        self.config = config if config else GammasarConfig()
    
    def extract_raw_grd(self, directory_name=None, prefix=None, phsname=None, rscname=None,
                        azifile=None, incfile=None, zero2nan=True, wavelength=None,
                        azi_reference=None, azi_unit=None, azi_direction=None,
                        inc_reference=None, inc_unit=None, mode=None):
        """
        Extract SAR raw images and process azimuth and incidence angles.

        Parameters:
        directory_name (str, optional): The directory name where the raw images are stored. Defaults to None.
        prefix (str, optional): The prefix for the output files. Defaults to None.
        phsname (str, optional): The name of the phase file. Defaults to None.
        rscname (str, optional): The name of the resource file. Defaults to None.
        azifile (str, optional): The name of the azimuth file. Defaults to None.
        incfile (str, optional): The name of the incidence file. Defaults to None.
        zero2nan (bool, optional): Whether to convert zero values to NaN. Defaults to config value.
        wavelength (float, optional): The wavelength of the SAR signal. Defaults to config value.
        azi_reference (str, optional): The reference direction for azimuth. Defaults to config value.
        azi_unit (str, optional): The unit of the azimuth angle. Defaults to config value.
        azi_direction (str, optional): The rotation direction for azimuth. Defaults to config value.
        inc_reference (str, optional): The reference direction for incidence. Defaults to config value.
        inc_unit (str, optional): The unit of the incidence angle. Defaults to config value.
        mode (str, optional): The mode of processing. Defaults to config value.

        Returns:
        None
        """
        # Use config values if parameters are not provided
        wavelength = wavelength if wavelength is not None else self.config.wavelength
        azi_reference = azi_reference if azi_reference is not None else self.config.azi_reference
        azi_unit = azi_unit if azi_unit is not None else self.config.azi_unit
        azi_direction = azi_direction if azi_direction is not None else self.config.azi_direction
        inc_reference = inc_reference if inc_reference is not None else self.config.inc_reference
        inc_unit = inc_unit if inc_unit is not None else self.config.inc_unit
        mode = mode if mode is not None else self.config.mode

        # extract sar raw images 
        if directory_name is not None:
            self.directory_name = directory_name
        else:
            directory_name = self.directory_name
        # phase file
        if prefix is not None:
            phase_file, rsc_file, azi_file, inc_file = self._construct_file_paths(directory_name, prefix)
        else:
            assert phsname is not None and rscname is not None, 'phase and rsc file must to be not None.'
            phase_file = os.path.join(directory_name, phsname)
            rsc_file = os.path.join(directory_name, rscname)
            azi_file = os.path.join(directory_name, azifile)
            inc_file = os.path.join(directory_name, incfile)

        rsc = pd.read_csv(rsc_file, sep=r'\s+', names=['name', 'value'])
        rsc.set_index('name', inplace=True)

        # 读取los文件信息
        # 数据存储信息
        nrow, ncol = [int(rsc.loc['WIDTH', 'value']), int(rsc.loc['FILE_LENGTH', 'value'])]
        wavelength = float(rsc.loc['WAVELENGTH', 'value'])
        dtype = np.float32
        # 数据读入和格式转存
        phs = np.fromfile(phase_file, dtype=dtype).reshape(ncol, nrow)
        los = phs

        # 方位角信息
        azi = np.fromfile(azi_file, dtype=dtype).reshape(ncol, nrow)
        # 入射角信息
        inc = np.fromfile(inc_file, dtype=dtype).reshape(ncol, nrow)

        # 位置信息
        x_first = np.float32(rsc.loc['X_FIRST', 'value'])
        y_first = np.float32(rsc.loc['Y_FIRST', 'value'])

        x_step = np.float32(rsc.loc['X_STEP', 'value'])
        y_step = np.float32(rsc.loc['Y_STEP', 'value'])
        lon = np.arange(x_first, x_first + x_step*nrow, x_step)[:nrow]
        lat = np.arange(y_first, y_first + y_step*ncol, y_step)[:ncol]

        self.raw_azi_input = azi
        self.raw_inc_input = inc
        # Process azimuth and incidence angles
        azi = self._process_azimuth(azi, azi_reference, azi_unit, azi_direction, mode=mode)
        inc = self._process_incidence(inc, inc_reference, inc_unit)

        if zero2nan:
            los[los == 0] = np.nan

        # Save in self object
        self.phase_file = phase_file
        self.rsc_file = rsc_file
        self.azi_file = azi_file
        self.inc_file = inc_file
        self.wavelength = wavelength
        self.raw_vel = los
        self.raw_azimuth = azi
        self.raw_incidence = inc
        self.raw_lon = lon
        self.raw_lat = lat
        mesh_lon, mesh_lat = np.meshgrid(lon, lat)
        self.raw_mesh_lon = mesh_lon
        self.raw_mesh_lat = mesh_lat
    
    def _construct_file_paths(self, directory_name, prefix):
        phase_file = prefix + '*.phs'
        rsc_file = prefix + '*.phs.rsc'
        azi_file = prefix + '*.azi'
        inc_file = prefix + '*.inc'
        phase_file = glob(os.path.join(directory_name, phase_file))[0]
        rsc_file = glob(os.path.join(directory_name, rsc_file))[0]
        azi_file = glob(os.path.join(directory_name, azi_file))[0]
        inc_file = glob(os.path.join(directory_name, inc_file))[0]
        return phase_file, rsc_file, azi_file, inc_file

    def read_from_gamma(self, downsample=1, apply_wavelength_conversion=True, zero2nan=True, wavelength=None):
        # Generate meshgrid for longitude and latitude
        Lon, Lat = np.meshgrid(self.raw_lon, self.raw_lat)

        # Read SAR data from binary files using inherited method
        # Desc: azimuth = ~-170; Asc: azimuth = ~-10
        self.read_from_binary(self.raw_vel, lon=Lon.flatten(), lat=Lat.flatten(), 
                              azimuth=self.raw_azimuth.flatten(), incidence=self.raw_incidence.flatten(), downsample=downsample)

        # Apply wavelength conversion to velocity if enabled
        if apply_wavelength_conversion:
            self.vel = self.unw_phase_to_los(self.vel, wavelength)
            self.raw_vel = self.unw_phase_to_los(self.raw_vel, wavelength)
        else:
            self.vel = -self.vel
            self.raw_vel = -self.raw_vel

        # Convert zeros to NaN in velocity data if enabled
        if zero2nan:
            self.vel[self.vel == 0] = np.nan

    def read_from_gamma(self, downsample=1, apply_wavelength_conversion=True, zero2nan=True, wavelength=None, sartype='unwrapPhase'):
        """
        Read SAR data from GAMMA format files.
        
        Parameters:
        -----------
        downsample : int, optional
            Downsample factor for the data. Default is 1 (no downsampling).
        apply_wavelength_conversion : bool, optional
            Whether to apply wavelength conversion to velocity. Default is True.
        zero2nan : bool, optional
            Whether to convert zero values to NaN. Default is True.
        wavelength : float, optional
            The wavelength to use for conversion. Default is None (use self.wavelength).
        sartype : str, optional
            Type of SAR data being processed. Options are 'unwrapPhase', 'rangeOffset', or 'azimuthOffset'.
            Default is 'unwrapPhase'.
        """
        # Generate meshgrid for longitude and latitude
        Lon, Lat = np.meshgrid(self.raw_lon, self.raw_lat)

        # Read SAR data from binary files using inherited method
        # Desc: azimuth = ~-100; Asc: azimuth = ~100
        if sartype in ('unwrapphase', 'UnwrapPhase', 'Unwrapphase', 'unwrapPhase'):
            incidence_flatten = self.raw_incidence.flatten()
            azimuth_flatten = self.raw_azimuth.flatten()
        elif sartype in ('rangeoffset', 'RangeOffset', 'Rangeoffset', 'rangeOffset'):
            # incidence_flatten = np.ones_like(self.raw_incidence.flatten()) * 90.0
            incidence_flatten = self.raw_incidence.flatten()
            azimuth_flatten = self.raw_azimuth.flatten()
        elif sartype in ('azimuthoffset', 'AzimuthOffset', 'Azimuthoffset', 'azimuthOffset'):
            incidence_flatten = np.ones_like(self.raw_incidence.flatten()) * 90.0
            azimuth_flatten = self.raw_azimuth.flatten() + 90.0

        self.read_from_binary(self.raw_vel, lon=Lon.flatten(), lat=Lat.flatten(), 
                                azimuth=azimuth_flatten, incidence=incidence_flatten, downsample=downsample)
    
        # Apply wavelength conversion to velocity if enabled
        if apply_wavelength_conversion:
            self.vel = self.unw_phase_to_los(self.vel, wavelength)
            self.raw_vel = self.unw_phase_to_los(self.raw_vel, wavelength)
        else:
            self.vel = -self.vel
            self.raw_vel = -self.raw_vel

        # Convert zeros to NaN in velocity data if enabled
        if zero2nan:
            self.vel[self.vel == 0] = np.nan

    def save_outputs_as_tiff(self, directory_name, save_azi=False, save_inc=False, save_vel=True, apply_wavelength_conversion=True, extent=None, grid_resolution=None):
        """
        Save azi, inc, and vel as GeoTIFF files, with optional resampling to a regular grid and user-defined extent.
    
        Parameters:
        - directory_name: str, directory to save the TIFF files.
        - save_azi: bool, whether to save azimuth data as TIFF.
        - save_inc: bool, whether to save incidence data as TIFF.
        - save_vel: bool, whether to save velocity data as TIFF.
        - apply_wavelength_conversion: bool, whether to apply wavelength conversion to velocity.
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
            azi_data = self.raw_azi_input
            azi_lon, azi_lat = self.raw_lon, self.raw_lat
            if grid_resolution and extent:
                azi_data, azi_lon, azi_lat = resample_to_regular_grid(azi_data, azi_lon, azi_lat, extent, grid_resolution)
            azi_data -= 90.0
            azi_file = os.path.join(directory_name, 'azi.tif')
            save_to_tiff(azi_data, azi_lon, azi_lat, azi_file)
    
        # Save incidence data
        if save_inc:
            inc_data = self.raw_inc_input
            inc_lon, inc_lat = self.raw_lon, self.raw_lat
            if grid_resolution and extent:
                inc_data, inc_lon, inc_lat = resample_to_regular_grid(inc_data, inc_lon, inc_lat, extent, grid_resolution)
            inc_file = os.path.join(directory_name, 'inc.tif')
            save_to_tiff(inc_data, inc_lon, inc_lat, inc_file)
    
        # Save velocity data
        if save_vel:
            vel_data = self.raw_vel
            vel_lon, vel_lat = self.raw_lon, self.raw_lat
            if apply_wavelength_conversion:
                vel_data = self.unw_phase_to_los(vel_data, self.wavelength)
            else:
                vel_data = -vel_data
            if grid_resolution and extent:
                vel_data, vel_lon, vel_lat = resample_to_regular_grid(vel_data, vel_lon, vel_lat, extent, grid_resolution)
            vel_file = os.path.join(directory_name, 'disp.tif')
            save_to_tiff(vel_data, vel_lon, vel_lat, vel_file)


if __name__ == '__main__':
    pass