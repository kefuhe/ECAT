# insar导入需要在gdal前面可能存在cartopy，shapely库冲突放在其后
from osgeo import gdal
import xarray
import numpy as np
import pandas as pd
from pyproj import Proj
import matplotlib.pyplot as plt
import cmcrameri as cmc
import os
from glob import glob
from abc import ABC, abstractmethod

# import csi modules and csiExtend modules
from csi.insar import insar
from ...plottools import set_degree_formatter, sci_plot_style

# -------------------------Define the configuration class-------------------------#
from enum import Enum

class AngleUnit(Enum):
    DEGREE = 'degree'
    RADIAN = 'radian'

class Direction(Enum):
    CLOCKWISE = 'clockwise'
    COUNTERCLOCKWISE = 'counterclockwise'

class Mode(Enum):
    RIGHT_LOS = 'right_los'
    HEADING = 'heading'
    LEFT_LOS = 'left_los'

class IncReference(Enum):
    ELEVATION = 'elevation'
    ZENITH = 'zenith'

class AziReference(Enum):
    NORTH = 'north'
    EAST = 'east'

class BaseConfig:
    def __init__(self):
        self.zero2nan = True
        self.wavelength = 0.056
        self.azi_reference = AziReference.NORTH
        self.azi_unit = AngleUnit.DEGREE
        self.azi_direction = Direction.CLOCKWISE
        self.inc_reference = IncReference.ELEVATION
        self.inc_unit = AngleUnit.DEGREE
        self.mode = Mode.RIGHT_LOS
        self.is_lonlat = True

class GammasarConfig(BaseConfig):
    def __init__(self):
        super().__init__()

class GammaTiffConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.mode = Mode.HEADING
    
class Hyp3TiffConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.azi_reference = AziReference.EAST
        self.azi_unit = AngleUnit.RADIAN
        self.azi_direction = Direction.COUNTERCLOCKWISE
        self.inc_unit = AngleUnit.RADIAN
        self.mode = Mode.LEFT_LOS
        self.is_lonlat = False


# -------------------------Define the reader class-------------------------#
class ReadBase2csisar(insar):
    def __init__(self, name=None, utmzone=None, lon0=None, lat0=None, directory_name='.', config=None):
        super().__init__(name, utmzone=utmzone, lon0=lon0, lat0=lat0)
        self.directory_name = directory_name
        self.config = config if config else BaseConfig()

        # Save in self object
        self.wavelength = None
        self.raw_vel = None
        self.raw_azimuth = None
        self.raw_incidence = None
        self.raw_lon = None
        self.raw_lat = None
        self.raw_mesh_lon = None
        self.raw_mesh_lat = None
    
    @abstractmethod
    def extract_raw_grd(self, directory_name=None, prefix=None, phsname=None, rscname=None,
                        azifile=None, incfile=None, zero2nan=True, wavelength=None,
                        azi_reference=None, azi_unit=None, azi_direction=None,
                        inc_reference=None, inc_unit=None, mode=None, *args, **kwargs):
        pass

    def set_directory_name(self, directory_name):
        self.directory_name = directory_name

    def _process_azimuth(self, azi, reference='east', unit='degree', direction='counterclockwise', mode='right_los'):
        """
        Process the azimuth angle based on the given reference direction, unit, rotation direction, and mode.

        Parameters:
        azi (numpy.ndarray): The azimuth angle array.
        reference (str): The reference direction for azimuth ('north' or 'east'). Default is 'east'.
        unit (str): The unit of the azimuth angle ('degree' or 'radian'). Default is 'degree'.
        direction (str): The rotation direction for azimuth ('clockwise' or 'counterclockwise'). Default is 'counterclockwise'.
        mode (str): The mode of processing ('heading', 'right_los', 'left_los'). Default is 'right_los'.

        Returns:
        heading (numpy.ndarray): The heading angle array in degrees with East as 0 degree and counterclockwise as positive.
        """
        unit = AngleUnit(unit)
        direction = Direction(direction)
        mode = Mode(mode)
        reference = AziReference(reference)

        # Convert azimuth angle if necessary
        # azi in azimuth direction with North as 0 degree and clockwise as positive
        if unit == AngleUnit.RADIAN:
            azi = np.rad2deg(azi)
        if direction == Direction.COUNTERCLOCKWISE:
            azi = -azi
        if reference == AziReference.EAST:
            azi = 90 + azi

        # Process based on the mode
        # Heading with East as 0 degree and counterclockwise as positive
        if mode == Mode.HEADING:
            return 90 - azi
        elif mode == Mode.RIGHT_LOS:
            heading = 90 - azi + 90
        elif mode == Mode.LEFT_LOS:
            heading = 90 - azi - 90
        else:
            raise ValueError("Invalid mode. Choose from 'heading', 'right_los', or 'left_los'.")

        return heading

    def _process_incidence(self, inc, reference='elevation', unit='degree'):
        """
        Process the incidence angle based on the given reference direction and unit.

        Parameters:
        inc (numpy.ndarray): The incidence angle array.
        reference (str): The reference direction for incidence ('zenith' or 'elevation'). Default is 'elevation'.
        unit (str): The unit of the incidence angle ('degree' or 'radian'). Default is 'degree'.

        Returns:
        numpy.ndarray: The processed incidence angle array in radians.
        """
        unit = AngleUnit(unit)
        reference = IncReference(reference)

        # Convert incidence angle if necessary
        if unit == AngleUnit.RADIAN:
            inc = np.rad2deg(inc)
        if reference == IncReference.ZENITH:
            inc = 90 - inc
        return inc
    
    def unw_phase_to_los(self, vel, wavelength=None):
        if wavelength is None:
            wavelength = self.wavelength
        else:
            self.wavelength = wavelength
        vel *= wavelength / (-4 * np.pi)

        return vel
    
    def to_xarray_dataarray(self):
        '''
        Also for GMT plot
        '''
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
        # Cut the raw SAR data based on the given longitude and latitude ranges
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range
        lon_idx = np.where((self.raw_lon >= lon_min) & (self.raw_lon <= lon_max))[0]
        lat_idx = np.where((self.raw_lat >= lat_min) & (self.raw_lat <= lat_max))[0]
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
    
    # ------------------------Plotting-------------------------------#
    def plot_raw_sar(self, coordrange=None, faults=None, rawdownsample4plot=100, factor4plot=100, 
                  vmin=None, vmax=None, symmetry=True, ax=None, tickfontsize=10, labelfontsize=10,
                  style=['science'], fontsize=None, figsize=None, save_fig=False, 
                  file_path='raw_sar.png', dpi=300, show=True, cmap='cmc.roma_r', 
                  trace_color='black', trace_linewidth=0.5, colorbar_length=0.4, colorbar_height=0.02,
                  colorbar_x=0.1, colorbar_y=0.1, colorbar_orientation='horizontal'):
        """
        Plot the raw SAR data.
        """
        # 计算绘图数据
        if coordrange is not None:
            lon_range, lat_range = coordrange[:2], coordrange[2:]
            rawsar, mesh_lon, mesh_lat, coordrange = self.cut_raw_sar(lon_range, lat_range)
        else:
            mesh_lon, mesh_lat = self.raw_mesh_lon, self.raw_mesh_lat
            rawsar = self.raw_vel
        rawsar = rawsar[::rawdownsample4plot, ::rawdownsample4plot] * factor4plot
        mesh_lon = mesh_lon[::rawdownsample4plot, ::rawdownsample4plot]
        mesh_lat = mesh_lat[::rawdownsample4plot, ::rawdownsample4plot]
        extent = coordrange if coordrange is not None else [mesh_lon.min(), mesh_lon.max(), mesh_lat.min(), mesh_lat.max()]
        rvmax = vmax if vmax is not None else np.nanmax(rawsar)
        rvmin = vmin if vmin is not None else np.nanmin(rawsar)
    
        # 设置颜色条的对称性
        if symmetry:
            vmax = max(abs(rvmin), rvmax)
            vmin = -vmax
        else:
            vmax, vmin = rvmax, rvmin

        # 确定图像的原点
        if mesh_lat[0, 0] > mesh_lat[0, -1]:
            origin = 'lower'
        else:
            origin = 'upper'
    
        # 设置绘图风格
        with sci_plot_style(style=style, fontsize=fontsize, figsize=figsize):
    
            # 创建或使用现有的Axes对象
            if ax is None:
                fig, ax = plt.subplots(1, 1) # , tight_layout=True
            else:
                fig = plt.gcf()
    
            # 绘制图像
            im = ax.imshow(rawsar, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, extent=extent)
            set_degree_formatter(ax, axis='both')
    
            # 绘制断层
            if faults is not None:
                for fault in faults:
                    if isinstance(fault, pd.DataFrame):
                        ax.plot(fault.lon.values, fault.lat.values, color=trace_color, lw=trace_linewidth)
                    else:
                        ax.plot(fault.lon, fault.lat, color=trace_color, lw=trace_linewidth)
    
            # 添加颜色条
            cbar_ax = fig.add_axes([colorbar_x, colorbar_y, colorbar_length, colorbar_height])  # [left, bottom, width, height]
            cb = fig.colorbar(im, cax=cbar_ax, orientation=colorbar_orientation)
            cb.ax.tick_params(labelsize=tickfontsize)
            cb.set_label('Disp. (cm)', fontdict={'size': labelfontsize})

            # 根据颜色条的方向设置标签位置
            if colorbar_orientation == 'vertical':
                cb.ax.yaxis.set_label_position("left")
            else:  # colorbar_orientation == 'horizontal'
                cb.ax.xaxis.set_label_position("top")

            # 保存或显示图像
            if save_fig:
                plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
    
        return fig, ax