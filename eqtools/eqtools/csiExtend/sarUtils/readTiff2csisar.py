# insar导入需要在gdal前面可能存在cartopy，shapely库冲突放在其后
from osgeo import gdal
import numpy as np
from pyproj import Proj
import matplotlib.pyplot as plt
import os
from glob import glob

from csi.insar import insar


class TiffsarReader(insar):
    def __init__(self, name=None, utmzone=None, lon0=None, lat0=None, directory_name='.'):
        super().__init__(name, utmzone=utmzone, lon0=lon0, lat0=lat0)
        self.directory_name = directory_name

    def extract_raw_grd(self, directory_name=None, prefix=None, phsname=None,
                        azifile=None, incfile=None, zero2nan=True, wavelength=0.056):
                # extract sar raw images 
        if directory_name is not None:
            self.directory_name = directory_name
        else:
            directory_name = self.directory_name
        # phase file
        if prefix:
            phase_file, azi_file, inc_file = self._construct_file_paths(directory_name, prefix)
        else:
            assert phsname and azifile and incfile, 'Phase, azimuth, and incidence files must not be None.'
            phase_file = os.path.join(directory_name, phsname)
            azi_file = os.path.join(directory_name, azifile)
            inc_file = os.path.join(directory_name, incfile)
        
        vel, azi, inc, x_lon, x_lat, mesh_lon, mesh_lat = self._read_sar_data(phase_file, azi_file, inc_file)
        if zero2nan:
            vel[vel == 0] = np.nan

        # Save in self
        self.wavelength = wavelength
        self.raw_vel = vel
        self.raw_azimuth = azi
        self.raw_indience = inc
        self.raw_lon = x_lon
        self.raw_lat = x_lat
        self.mesh_lon = mesh_lon
        self.mesh_lat = mesh_lat

    def _construct_file_paths(self, directory_name, prefix):
        phase_file = glob(os.path.join(directory_name, f'{prefix}*.tif'))[0]
        azi_file = glob(os.path.join(directory_name, f'{prefix}*.azi.tif'))[0]
        inc_file = glob(os.path.join(directory_name, f'{prefix}*.inc.tif'))[0]
        return phase_file, azi_file, inc_file

    def _read_sar_data(self, phase_file, azi_file, inc_file):
        vel, _, _, im_width, im_height = read_tiff(phase_file)
        # 方位角信息
        azi, _, _, _, _ = read_tiff(azi_file)
        # 入射角信息
        inc, _, _, _, _ = read_tiff(inc_file)
        azi += 90  # Adjust azimuth for specific use case
        # 读取元数据
        x_lon, _, x_lat, _, mesh_lon, mesh_lat = read_tiff_info(phase_file, im_width, im_height)
        return vel, azi, inc, x_lon, x_lat, mesh_lon, mesh_lat

    def read_from_tiff(self, downsample=1, apply_wavelength_conversion=False, 
                       zero2nan=True, wavelength=None):
        vel = self.raw_vel
        azi = self.raw_azimuth
        inc = self.raw_indience
        mesh_lon = self.mesh_lon
        mesh_lat = self.mesh_lat
        # Read SAR data from binary files using inherited method
        self.read_from_binary(vel, lon=mesh_lon.flatten(), lat=mesh_lat.flatten(), 
                             azimuth=azi.flatten(), incidence=inc.flatten(), downsample=downsample)
        
        # Apply wavelength conversion to velocity if enabled
        if apply_wavelength_conversion:
            self.wavelegnth_conversion(wavelength)

        # Convert zeros to NaN in velocity data if enabled
        if zero2nan:
            self.vel[self.vel == 0] = np.nan
    
    def wavelegnth_conversion(self, wavelength=None):
        if wavelength is None:
            wavelength = self.wavelength
        else:
            self.wavelength = wavelength
        self.vel *= wavelength / (-4 * np.pi)
    
    def to_xarray_dataarray(self):
        import xarray as xr
        return xr.DataArray(self.vel, coords={'lat': self.lat, 'lon': self.lon}, dims=['lat', 'lon'])


def read_tiff(unwfile):
    '''
    Input:
        * unwfile     : tiff格式影像文件
    Output:
        * im_data     : 数据矩阵
        * im_geotrans : 横纵坐标起始点和步长
        * im_proj     : 投影参数
        * im_width    : 像元的行数
        * im_height   : 像元的列数
    '''
    dataset = gdal.Open(unwfile, gdal.GA_ReadOnly)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率/步长
    im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
    im_band = dataset.GetRasterBand(1)
    # im_data = np.asarray(imread(filename))
    # 下面这个一直进行数据读取有问题
    im_data = im_band.ReadAsArray(0, 0, im_width, im_height) # 将栅格图像值存为数据矩阵
    del dataset
    return im_data, im_geotrans, im_proj, im_width, im_height


def read_tiff_info(unwfile, im_width, im_height, meshout=True):
    '''
    Object:
        * 获取tiff文件坐标系统和横轴、纵轴范围
    Input:
        * tiff文件文件名
    Output:
        * x_lon, y_step, x_lat, y_step
    '''

    metadata = gdal.Info(unwfile, format='json', deserialize=True)
    upperLeft = metadata['cornerCoordinates']['upperLeft']
    lowerRight = metadata['cornerCoordinates']['lowerRight']
    x_upperleft, y_upperleft = upperLeft
    x_lowerright, y_lowerright = lowerRight
    x_step = (x_upperleft - x_lowerright)/im_width
    y_step = -(y_upperleft - y_lowerright)/im_height
    x_lon = np.linspace(x_upperleft, x_lowerright, im_width)
    x_lat = np.linspace(y_upperleft, y_lowerright, im_height)

    if meshout:
        mesh_lon, mesh_lat = np.meshgrid(x_lon, x_lat)
        return x_lon, y_step, x_lat, y_step, mesh_lon, mesh_lat
    else:
        return x_lon, x_step, x_lat, y_step


def write_tiff(filename, im_proj, im_geotrans, im_data=None):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
        
    # 判读数组维数
    # im_data.shape可能有三个元素，对应多维矩阵，即多个波段；
    # 也可能有两个元素，对应二维矩阵，即一个波段
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands = 1
        (im_height, im_width) = im_data.shape


if __name__ == '__main__':
    unwfile = 'sarfile.tif'
    im_data, im_geotrans, im_proj, im_width, im_height = read_tiff(unwfile)
    x_lon, y_step, x_lat, y_step, mesh_lon, mesh_lat = read_tiff_info(unwfile)
    
    # %%
    # 数据存储信息
    im_data, im_geotrans, im_proj, im_width, im_height = read_tiff(phase_file)
    # im_data = im_data*0.056/(-4*np.pi) # 存储为相位则需要这一项，如果存储为直接形变则不用
    los = im_data
    # 读取元数据
    x_lon, y_step, x_lat, y_step, mesh_lon, mesh_lat = read_tiff_info(phase_file, im_width, im_height)
    x_lowerright = np.nanmin(x_lon)
    x_upperleft = np.nanmax(x_lon)
    y_lowerright = np.nanmin(x_lat)
    y_upperleft = np.nanmax(x_lat)

    coordrange = [x_lowerright, x_upperleft, y_lowerright, y_upperleft]
    # %%
    plt.imshow(los, interpolation='none', # cmap=cmap, norm=norm,
             origin='upper', extent=[np.nanmin(x_lon), np.nanmax(x_lon), np.nanmin(x_lat), np.nanmax(x_lat)], aspect='auto', vmax=1, vmin=-1)

    # %% [markdown]
    # 读取方位角和入射角文件信息

    # %%
    # 方位角信息
    azi, _, _, _, _ = read_tiff(azi_file)
    # 入射角信息
    inc, _, _, _, _ = read_tiff(inc_file)

    from csi import insar
    sar = insar('{}_coseismic'.format(outName), lon0=98.25, lat0=34.5)
    sar.read_from_binary(sardata, lon, lat, los=los, downsample=1)