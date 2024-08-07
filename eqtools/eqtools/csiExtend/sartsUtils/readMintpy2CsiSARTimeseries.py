import os
import numpy as np
import pandas as pd
import h5py
from math import floor, ceil
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import cmcrameri as cmc
import string

from csi import insartimeseries as csiinsartimeseries
from ...plottools import set_degree_formatter, sci_plot_style

class MintpyInSARTimeseriesReader(csiinsartimeseries):
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, 
                 hdf5_file_dict=None, directory_name=None, downsample=1):
        # Base class init with verbose set to False as default
        super().__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=False) 
        
        default_hdf5_file_dict = {
            'avgSpatialCoh': 'geo_avgSpatialCoh.h5',
            'geoRadar': 'geo_geometryRadar.h5',
            'maskTempCoh': 'geo_maskTempCoh.h5',
            'temporalCoherence': 'geo_temporalCoherence.h5',
            'velocity': 'geo_velocity.h5',
            'timeseries': 'geo_timeseries_demErr.h5'
        }
        self.hdf5_file_dict = default_hdf5_file_dict if hdf5_file_dict is None else {**default_hdf5_file_dict, **hdf5_file_dict}
        
        self.directory_name = directory_name
        
        self.downsample = downsample
    
    def set_directory_name(self, directory_name):
        self.directory_name = directory_name
    
    def update_hdf5_file_dict(self, hdf5_file_dict):
        self.hdf5_file_dict.update(hdf5_file_dict)
    
    def set_geometry(self, shadow_mask=False, downsample=1, keep_row_time_series=True):

        dirname = self.directory_name
        filenames = self.hdf5_file_dict
        self.downsample = downsample

        # keys: ['azimuthAngle', 'height', 'incidenceAngle', 'latitude', 
        #        'longitude', 'shadowMask', 'slantRangeDistance']
        # 使用with语句安全地打开和操作HDF5文件
        with h5py.File(os.path.join(dirname, filenames['geoRadar']), 'r') as geo_radar:
            # 直接从文件中提取所需数据
            azimuth_angle = geo_radar['azimuthAngle'][:]
            incidence_angle = geo_radar['incidenceAngle'][:]
            longitude = geo_radar['longitude'][:]
            latitude = geo_radar['latitude'][:]
            if shadow_mask:
                self.shadow_mask = geo_radar['shadowMask'][:]

        # 计算经纬度范围
        min_lon, max_lon = np.nanmin(longitude), np.nanmax(longitude)
        min_lat, max_lat = np.nanmin(latitude), np.nanmax(latitude)
        self.raw_coord_range = [min_lon, max_lon, min_lat, max_lat]
        
        # 生成经纬度网格
        y_size, x_size = longitude.shape
        lon = np.linspace(min_lon, max_lon, x_size)
        lat = np.linspace(max_lat, min_lat, y_size)
        mesh_lon, mesh_lat = np.meshgrid(lon, lat)

        # 根据keep_row_time_series标志决定是否保留原始网格数据
        if keep_row_time_series:
            self.raw_mesh_lon = mesh_lon
            self.raw_mesh_lat = mesh_lat

        # 设置经纬度和其他属性
        self.setLonLat(mesh_lon.flatten()[::downsample], mesh_lat.flatten()[::downsample], 
                        incidence=incidence_angle.flatten()[::downsample], 
                        heading=-azimuth_angle.flatten()[::downsample]+90.0, dtype=np.float32)
        self.raw_incidence = incidence_angle.flatten()[::downsample]
        self.raw_azimuth = azimuth_angle.flatten()[::downsample]
        # All Done
        return

    def extractTimeSeries(self, maskTempCoh=False, downsample=1, factor=1, keep_row_time_series=True):

        dirname = self.directory_name
        filenames = self.hdf5_file_dict
        self.factor = factor
        # Extract SAR time series
        # keys: ['bperp', 'date', 'timeseries']
        with h5py.File(os.path.join(dirname, filenames['timeseries']), 'r+') as ts:
            timeseries = ts['timeseries'][:]

            # 修复后的代码
            dateseries = pd.DatetimeIndex(pd.to_datetime([date.decode("utf-8") for date in ts['date'][:]], 
                                                         format='%Y%m%d'))

            # bperp = ts['bperp'][:]
            pydates = [ts.to_pydatetime() for ts in dateseries]

        if maskTempCoh:
            with h5py.File(os.path.join(dirname, filenames['maskTempCoh']), 'r+') as maskTempCoh:
                mask = maskTempCoh['mask'][:]
            # mask sar image
            sar_ts = []
            for ts in timeseries:
                ts[~mask] = np.nan
                sar_ts.append(ts.flatten()[::downsample]*factor)
        else:
            # 如果mask为False，直接处理时间序列数据而不应用掩码
            sar_ts = [ts.flatten()[::downsample]*factor for ts in timeseries]
        
        if keep_row_time_series:
            self.raw_time_series = timeseries

        self.initializeTimeSeries(time=pydates, dtype=np.float32)

        return self.setTimeSeries(sar_ts)

    def read_from_h5file(self, directory_name=None, hdf5_file_dict=None, factor=1.0, maskTempCoh=False, 
                         downsample=1, shadow_mask=False, keep_row_time_series=True):
        '''
        
        Args      :
            * dirname         :
            * sarfile_pattern : dict

        Kwargs   :
            * factor          : 形变缩放单位

        Return
            * None
        '''
        if directory_name is not None:
            self.set_directory_name(directory_name)
        if hdf5_file_dict is not None:
            self.update_hdf5_file_dict(hdf5_file_dict)
        self.set_geometry(shadow_mask=shadow_mask, downsample=downsample)
        self.extractTimeSeries(maskTempCoh=maskTempCoh, downsample=downsample, factor=factor, 
                               keep_row_time_series=keep_row_time_series)
    
        # All Done
        return
    
    def set_other_mask(self, mask):
        """
        Apply a mask to the time series data, setting values to NaN where mask is False.
        Downsamples the time series by the object's downsample factor after masking.
        """
        sar_ts = []
        for ts in self.raw_time_series:
            ts[~mask] = np.nan  # Use ~mask for clearer expression
            sar_ts.append(ts.flatten()[::self.downsample] * self.factor)

        return self.setTimeSeries(sar_ts)
    
    def cut_raw_timeseries(self, lon_range, lat_range, inplace=False):
        """
        Cuts the raw time series data to the specified longitude and latitude ranges.
        If inplace is True, updates the object's attributes. Otherwise, returns the cut data and new coordinate range.
        """
        corner_lon, corner_lat = self.raw_coord_range[0], self.raw_coord_range[-1]
        nx, ny = self.raw_mesh_lon.shape
        min_lon, max_lon, min_lat, max_lat = self.raw_coord_range
        dx = (max_lon - min_lon) / (ny - 1)
        dy = (-max_lat + min_lat) / (nx - 1)
    
        l_low = max(floor((lon_range[0] - corner_lon) / dx), 0)
        r_low = ceil((lon_range[1] - corner_lon) / dx)
        t_row = max(floor((lat_range[0] - corner_lat) / dy), 0)
        b_row = ceil((lat_range[1] - corner_lat) / dy)
    
        # Extract the relevant slices from the time series and coordinate grids
        tsts = self.raw_time_series[:, b_row:t_row:1, l_low:r_low:1]
        lon = self.raw_mesh_lon[b_row:t_row:1, l_low:r_low:1]
        lat = self.raw_mesh_lat[b_row:t_row:1, l_low:r_low:1]
    
        # Calculate the new coordinate range
        # 计算新的经度和纬度范围
        new_min_lon = corner_lon + l_low * dx
        new_max_lon = corner_lon + (r_low - 1) * dx
        new_min_lat = corner_lat + b_row * dy
        new_max_lat = corner_lat + (t_row - 1) * dy
        
        # 确保新的最小值小于等于最大值
        new_min_lon, new_max_lon = min(new_min_lon, new_max_lon), max(new_min_lon, new_max_lon)
        new_min_lat, new_max_lat = min(new_min_lat, new_max_lat), max(new_min_lat, new_max_lat)
        new_coord_range = (new_min_lon, new_max_lon, new_min_lat, new_max_lat)
    
        if inplace:
            # Update the object's attributes with the cut data
            self.raw_mesh_lon = lon
            self.raw_mesh_lat = lat
            self.raw_time_series = tsts
            self.raw_coord_range = new_coord_range
        else:
            return tsts, lon, lat, new_coord_range

    def plot_raw_sar_at_time(self, time_index=-1, coordrange=None, faults=None, rawdownsample4plot=100, factor4plot=100, 
                  vmin=None, vmax=None, symmetry=True, ax=None, tickfontsize=10, labelfontsize=10,
                  style=['science'], fontsize=None, figsize=None, save_fig=False, 
                  file_path='raw_sar_at_time.png', dpi=300, show=True, cmap='cmc.roma_r', 
                  trace_color='black', trace_linewidth=0.5, colorbar_length=0.4, colorbar_height=0.02,
                  colorbar_x=0.1, colorbar_y=0.1, colorbar_orientation='horizontal'):
        """
        在matplotlib中绘制SAR时间序列的指定帧。
        """
        # 计算绘图数据
        if coordrange is not None:
            lon_range, lat_range = coordrange[:2], coordrange[2:]
            time_series, mesh_lon, mesh_lat, coordrange = self.cut_raw_timeseries(lon_range, lat_range)
        else:
            time_series, mesh_lon, mesh_lat = self.raw_time_series, self.raw_mesh_lon, self.raw_mesh_lat
        endsar = time_series[time_index, ::rawdownsample4plot, ::rawdownsample4plot] * factor4plot
        extent = coordrange if coordrange is not None else self.raw_coord_range
        rvmax = vmax if vmax is not None else np.nanmax(endsar)
        rvmin = vmin if vmin is not None else np.nanmin(endsar)
    
        # 设置颜色条的对称性
        if symmetry:
            vmax = max(abs(rvmin), rvmax)
            vmin = -vmax
        else:
            vmax, vmin = rvmax, rvmin
    
        # 设置绘图风格
        with sci_plot_style(style=style, fontsize=fontsize, figsize=figsize):
    
            # 创建或使用现有的Axes对象
            if ax is None:
                fig, ax = plt.subplots(1, 1, tight_layout=True)
            else:
                fig = plt.gcf()
    
            # 绘制图像
            im = ax.imshow(endsar, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', extent=extent)
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

    def plot_raw_sar_timeseries(self, time_indices=None, coordrange=None, rawdownsample4plot=100, factor4plot=100, 
                                vmin=None, vmax=None, symmetry=True, columns=3, figsize=None, 
                                save_fig=False, file_path='timeseries_plot.png', dpi=300, 
                                show=True, cmap='cmc.roma_r', trace_color='black', trace_linewidth=0.5, 
                                style='science', fontsize=None, faults=None, colorbar_length=0.4, colorbar_height=0.02,
                                reference_time=None, colorbar_x=0.1, colorbar_y=0.1, colorbar_orientation='horizontal'):
        """
        以多子图形式绘制SAR时间序列的指定帧，列数可调，默认为3列。如果未指定time_indices，则绘制所有时间序列帧。
        """
        # 在绘图循环之前，创建一个字母序列
        alphabet = list(string.ascii_lowercase)

        # 如果未指定time_indices，则绘制所有时间序列帧
        if time_indices is None:
            time_indices = range(self.raw_time_series.shape[0])

        # 计算绘图数据
        if coordrange is not None:
            lon_range, lat_range = coordrange[:2], coordrange[2:]
            time_series, mesh_lon, mesh_lat, coordrange = self.cut_raw_timeseries(lon_range, lat_range)
        else:
            time_series, mesh_lon, mesh_lat = self.raw_time_series, self.raw_mesh_lon, self.raw_mesh_lat
            coordrange = self.raw_coord_range
        time_series = time_series[:, ::rawdownsample4plot, ::rawdownsample4plot] * factor4plot
    
        # 计算全局vmin和vmax
        if vmin is None or vmax is None:
            if vmin is None:
                vmin = np.nanmin(time_series)
            if vmax is None:
                vmax = np.nanmax(time_series)
            if symmetry:
                vmax = max(abs(vmin), vmax)
                vmin = -vmax
    
        # 设置绘图风格
        with sci_plot_style(style=style, fontsize=fontsize):
            # 计算子图布局
            total_frames = len(time_indices)
            rows = np.ceil(total_frames / columns).astype(int)
            if figsize is None:
                figsize = (8.3, 2.9 * rows)  # 默认全幅宽度，高度根据行数调整
    
            fig, axs = plt.subplots(rows, columns, figsize=figsize, tight_layout=True, sharex=True, sharey=True)
            if rows * columns > 1:
                axs = axs.flatten()
            else:
                axs = [axs]
    
            for i, time_index in enumerate(range(total_frames)):
                ax = axs[i]
                endsar = time_series[time_index]
                itime = self.time[time_index]
                if reference_time is not None:
                    itime_str = itime - pd.to_datetime(reference_time)
                    itime_str = '{:d} days'.format(itime_str.days)
                else:
                    itime_str = itime.strftime('%Y-%m-%d')
                extent = coordrange
    
                # 绘制图像
                im = ax.imshow(endsar, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', extent=extent)
                set_degree_formatter(ax, axis='both')
    
                # 在左上角添加字母标注
                ax.text(0.05, 0.95, f'({alphabet[i]}) {itime_str}', transform=ax.transAxes, fontsize=12, 
                        verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5, facecolor='white'))
    
                # 绘制断层
                if faults is not None:
                    for fault in faults:
                        if isinstance(fault, pd.DataFrame):
                            ax.plot(fault.lon.values, fault.lat.values, color=trace_color, lw=trace_linewidth)
                        else:
                            ax.plot(fault.lon, fault.lat, color=trace_color, lw=trace_linewidth)
    
            # 隐藏空白子图
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')
    
            # 添加颜色条
            cbar_ax = fig.add_axes([colorbar_x, colorbar_y, colorbar_length, colorbar_height])  # [left, bottom, width, height]
            fig.colorbar(im, cax=cbar_ax, orientation=colorbar_orientation)
    
            # 保存或显示图像
            if save_fig:
                plt.savefig(file_path, dpi=dpi) # , bbox_inches='tight'
            if show:
                plt.show()
            else:
                plt.close()
    
            return fig, axs


if __name__ == '__main__':

    ## 选取数据范围
    lonrange = [99.46, 102.762]
    latrange = [36.59, 38.942]

    # %% Building a SAR Timeseries Object
    # 5 m(距离向) * 15 m
    downsample = 50 # 10

    # center for local coordinates--M7.4 epicenter
    lon0 = 101.31
    lat0 = 37.80
    sarts_menyuan = MySarTsReader('Menyuan', utmzone=None, lon0=lon0, lat0=lat0)
    sarts_menyuan.setdirname(os.path.join('..', '..', 'geo'))
    sarts_menyuan.read_from_h5file(factor=1.0, mask=True, downssample=100)

    # Image Display
    # Fault Trace
    main_rupture = pd.read_csv('../Main.txt', names=['lon', 'lat'], sep=r'\s+')
    tip_rupture = pd.read_csv('../Second.txt', names=['lon', 'lat'], sep=r'\s+')

    from csi import RectangularPatches as csiRect
    main_fault = csiRect('main', lon0=lon0, lat0=lat0)
    main_fault.trace(main_rupture.lon.values, main_rupture.lat.values)
    main_fault.discretize()
    sec_fault = csiRect('sec', lon0=lon0, lat0=lat0)
    sec_fault.trace(tip_rupture.lon.values, tip_rupture.lat.values)
    sec_fault.discretize()
    faults = [main_fault, sec_fault]


    # hypocenter: 98.38 34.86 Strike: 285
    hypolon = 101.428
    hypolat = 37.736
    strike = 110
    sarts_menyuan.getProfiles('hypo', loncenter=hypolon, latcenter=hypolat, length=60, azimuth=strike-90, width=5, verbose=True)
    sarts_menyuan.smoothProfiles('hypo', window=0.25, method='mean')
    sarts_menyuan.plotProfiles('Smoothed hypo', color='b')

    sarts_menyuan.plotinmpl(vmin=-2, vmax=2, rawdownsample4plot=1, faults=faults)

    sartmp = sarts_menyuan.timeseries[-1]
    name = 'hypo {}'.format(sartmp.name)
    # sartmp.smoothProfile(name, window=0.25, method='mean')
    # sartmp.plotprofile(name, norm=[-0.02, 0.02]) # Smoothed {}
    sartmp.plotprofile('Smoothed {}'.format(name), norm=[-0.02, 0.02])


    # %% 提取特定点位附近的insar点信号
    from csi import gps as csigps, gpstimeseries as csigpstimeseries

    samp_gps = csigps('Sampling_Points', utmzone=None, lon0=lon0, lat0=lat0)
    pnt_stat_filename = os.path.join('..', 'sampling_points_lonlat.dat')
    samp_gps.setStatFromFile(pnt_stat_filename, initVel=False, header=1)
    # samp_gps.initializeTimeSeries(time=sarts_menyuan.time, los=True, verbose=True)
    # 采样点大小需要斟酌， distance： x Km
    samp_pnts = sarts_menyuan.extractAroundGPS(samp_gps, distance=2, doprojection=False, reference=False)

    # %% 将采样剖面存储下来
    # sarts_menyuan.writeProfiles2Files('hypo', 'samp', )
