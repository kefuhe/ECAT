import xarray
import numpy as np
from glob import glob
import os
import pandas as pd
from csi.insar import insar


class GammasarReader(insar):
    def __init__(self, name=None, utmzone=None, lon0=None, lat0=None, directory_name='.'):
        super().__init__(name, utmzone=utmzone, lon0=lon0, lat0=lat0)
        self.directory_name = directory_name

    def set_directory_name(self, directory_name):
        self.directory_name = directory_name
    
    def extract_raw_grd(self, directory_name=None, prefix=None, phsname=None, rscname=None,
                        azifile=None, incfile=None, zero2nan=True):
        # extract sar raw images 
        if directory_name is not None:
            self.directory_name = directory_name
        else:
            directory_name = self.directory_name
        # phase file
        if prefix is not None:
            phase_file = prefix + '*.phs'
            rsc_file = prefix + '*.phs.rsc'
            azi_file = prefix + '*.azi'
            inc_file = prefix + '*.inc'
            phase_file = glob(os.path.join(directory_name, phase_file))[0]
            rsc_file = glob(os.path.join(directory_name, rsc_file))[0]
            azi_file = glob(os.path.join(directory_name, azi_file))[0]
            inc_file = glob(os.path.join(directory_name, inc_file))[0]
        else:
            assert phsname is not None and rscname is not None, 'phase and rsc file must to be not None.'
            phase_file = os.path.join(directory_name, phsname)
            rsc_file = os.path.join(directory_name, rsc_file)
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
        los = phs*wavelength/(-4*np.pi)

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
        self.raw_inidence = inc
        self.raw_lon = lon
        self.raw_lat = lat

    def read_from_gamma(self, downsample=1, apply_wavelength_conversion=True, zero2nan=True):
        # Generate meshgrid for longitude and latitude
        Lon, Lat = np.meshgrid(self.raw_lon, self.raw_lat)

        # Read SAR data from binary files using inherited method
        self.read_from_binary(self.phase_file, lon=Lon.flatten(), lat=Lat.flatten(), 
                              azimuth=self.azi_file, incidence=self.inc_file, downsample=downsample)

        # Apply wavelength conversion to velocity if enabled
        if apply_wavelength_conversion:
            self.vel *= self.wavelength / (-4 * np.pi)

        # Convert zeros to NaN in velocity data if enabled
        if zero2nan:
            self.vel[self.vel == 0] = np.nan

    def to_xarray_dataarray(self):
        '''
        Also for GMT plot
        '''
        # Convert velocity data to an xarray DataArray with proper coordinates
        data_array = xarray.DataArray(self.vel, coords=[('lat', self.lat), ('lon', self.lon)], dims=['lat', 'lon'])
    
        return data_array
        

if __name__ == '__main__':
    pass