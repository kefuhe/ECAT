# insar导入需要在gdal前面可能存在cartopy，shapely库冲突放在其后
from osgeo import gdal
from osgeo import osr
import numpy as np
from pyproj import Proj
import matplotlib.pyplot as plt
import os
from glob import glob

from csi.insar import insar
from .readBase2csisar import ReadBase2csisar, Hyp3TiffConfig, GammaTiffConfig


class TiffsarReader(ReadBase2csisar):
    def __init__(self, name=None, utmzone=None, lon0=None, lat0=None, directory_name='.', config=None):
        super().__init__(name, utmzone=utmzone, lon0=lon0, lat0=lat0)
        self.directory_name = directory_name
        self.config = config if config else Hyp3TiffConfig()

    def extract_raw_grd(self, directory_name=None, prefix=None, phsname=None,
                        azifile=None, incfile=None, zero2nan=True, wavelength=None,
                        azi_reference=None, azi_unit=None, azi_direction=None,
                        inc_reference=None, inc_unit=None, mode=None, is_lonlat=None):
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
        is_lonlat (bool, optional): Whether the data is in longitude and latitude. Defaults to config value.

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
        is_lonlat = is_lonlat if is_lonlat is not None else self.config.is_lonlat

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
        
        vel, azi, inc, x_lon, x_lat, mesh_lon, mesh_lat, im_geotrans, im_proj = self._read_sar_data(phase_file, azi_file, inc_file)

        self.raw_azi_input = azi
        self.raw_inc_input = inc
        # Process azimuth and incidence angles
        azi = self._process_azimuth(azi, azi_reference, azi_unit, azi_direction, mode=mode)
        inc = self._process_incidence(inc, inc_reference, inc_unit)

        # Transfer coordinate system from UTM to latitude and longitude
        if not is_lonlat:
            im_height, im_width = mesh_lon.shape
            mesh_lat, mesh_lon = utm_to_latlon(mesh_lon.flatten(), mesh_lat.flatten(), im_proj)
            mesh_lat = mesh_lat.reshape(im_height, im_width)
            mesh_lon = mesh_lon.reshape(im_height, im_width)
        
        if zero2nan:
            vel[vel == 0] = np.nan

        # Save in self
        self.wavelength = wavelength
        self.raw_vel = vel
        self.raw_azimuth = azi
        self.raw_incidence = inc
        self.raw_lon = x_lon
        self.raw_lat = x_lat
        self.raw_mesh_lon = mesh_lon
        self.raw_mesh_lat = mesh_lat
        self.im_geotrans = im_geotrans
        self.im_proj = im_proj

    def _construct_file_paths(self, directory_name, prefix):
        phase_file = glob(os.path.join(directory_name, f'{prefix}*.tif'))[0]
        azi_file = glob(os.path.join(directory_name, f'{prefix}*.azi.tif'))[0]
        inc_file = glob(os.path.join(directory_name, f'{prefix}*.inc.tif'))[0]
        return phase_file, azi_file, inc_file

    def _read_sar_data(self, phase_file, azi_file, inc_file):
        vel, im_geotrans, im_proj, im_width, im_height = read_tiff(phase_file)
        # Azimuth information
        azi, _, _, _, _ = read_tiff(azi_file)
        # Incidence information
        inc, _, _, _, _ = read_tiff(inc_file)

        # Read Metadata information
        x_lon, _, x_lat, _, mesh_lon, mesh_lat = read_tiff_info(phase_file, im_width, im_height)

        return vel, azi, inc, x_lon, x_lat, mesh_lon, mesh_lat, im_geotrans, im_proj

    def read_from_tiff(self, downsample=1, apply_wavelength_conversion=False, 
                       zero2nan=True, wavelength=None):
        vel = self.raw_vel
        azi = self.raw_azimuth
        inc = self.raw_incidence
        mesh_lon = self.raw_mesh_lon
        mesh_lat = self.raw_mesh_lat
        # Read SAR data from binary files using inherited method
        self.read_from_binary(vel, lon=mesh_lon.flatten(), lat=mesh_lat.flatten(), 
                             azimuth=azi.flatten(), incidence=inc.flatten(), downsample=downsample)
        
        # Apply wavelength conversion to velocity if enabled
        if apply_wavelength_conversion:
            self.vel = self.unw_phase_to_los(self.vel, wavelength)
            self.raw_vel = self.unw_phase_to_los(self.raw_vel, wavelength)

        # Convert zeros to NaN in velocity data if enabled
        if zero2nan:
            self.vel[self.vel == 0] = np.nan


class Hyp3TiffReader(TiffsarReader):
    def __init__(self, name=None, utmzone=None, lon0=None, lat0=None, directory_name='.', config=None):
        super().__init__(name, utmzone=utmzone, lon0=lon0, lat0=lat0)
        self.directory_name = directory_name
        self.config = config if config else Hyp3TiffConfig()

class GammaTiffReader(TiffsarReader):
    def __init__(self, name=None, utmzone=None, lon0=None, lat0=None, directory_name='.', config=None):
        super().__init__(name, utmzone=utmzone, lon0=lon0, lat0=lat0)
        self.directory_name = directory_name
        self.config = config if config else GammaTiffConfig()


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


def utm_to_latlon(easting, northing, proj_info):
    """
    Convert UTM coordinates to latitude and longitude.

    Parameters:
    easting (numpy.ndarray): The easting values (X coordinates).
    northing (numpy.ndarray): The northing values (Y coordinates).
    proj_info (str): The projection information string.

    Returns:
    tuple: Two numpy arrays containing the latitudes and longitudes.
    """
    # Ensure easting and northing are numpy arrays
    easting = np.asarray(easting)
    northing = np.asarray(northing)

    # Check if easting and northing arrays have the same size
    if easting.shape != northing.shape:
        raise ValueError("Easting and northing arrays must have the same size")

    # Create a spatial reference object for the UTM projection
    utm_srs = osr.SpatialReference()
    utm_srs.ImportFromWkt(proj_info)

    # Create a spatial reference object for the WGS84 geographic coordinate system
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)  # EPSG code for WGS84

    # Create a coordinate transformation object
    transform = osr.CoordinateTransformation(utm_srs, wgs84_srs)

    # Initialize arrays for latitudes and longitudes
    latitudes = np.zeros_like(easting)
    longitudes = np.zeros_like(northing)

    # Perform the transformation for each point
    for i in range(len(easting)):
        # TODO: Check Use of TransformPoint() method
        lat, lon, _ = transform.TransformPoint(easting[i], northing[i])
        latitudes[i] = lat
        longitudes[i] = lon

    return latitudes, longitudes


if __name__ == '__main__':
    pass
    # plt.imshow(los, interpolation='none', # cmap=cmap, norm=norm,
    #          origin='upper', extent=[np.nanmin(x_lon), np.nanmax(x_lon), np.nanmin(x_lat), np.nanmax(x_lat)], aspect='auto', vmax=1, vmin=-1)