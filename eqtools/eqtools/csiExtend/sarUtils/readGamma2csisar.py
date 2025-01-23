import xarray
import numpy as np
from glob import glob
import os
import pandas as pd
from csi.insar import insar
from .readBase2csisar import ReadBase2csisar, GammasarConfig


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
        

if __name__ == '__main__':
    pass