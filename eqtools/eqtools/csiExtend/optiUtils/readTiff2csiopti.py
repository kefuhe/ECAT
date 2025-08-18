from ..sarUtils.readTiffUtils import read_tiff_with_metadata
from ...plottools import sci_plot_style, set_degree_formatter
from ..sarUtils.readTiffUtils import save_to_tiff
from csi.opticorr import opticorr as csiopticorr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray
import os


class TiffoptiReader(csiopticorr):
    def __init__(self, name=None, utmzone=None, lon0=None, lat0=None, directory_name='.', verbose=True):
        super().__init__(name, utmzone=utmzone, lon0=lon0, lat0=lat0, verbose=verbose)
        self.directory_name = directory_name

    def extract_raw_grd(self, directory_name=None, filename=None, ew_band=1, sn_band=2, vert_band=None,
                        factor_to_m=1.0, zero2nan=True, wavelength=None):
        """
        Extract SAR raw images and process azimuth and incidence angles.
    
        Parameters:
            * directory_name (str, optional): The directory name where the raw images are stored. Defaults to None.
            * filename (str, optional): The name of the raw image file. Defaults to None.
            * ew_band (int, optional): The band number for the east-west component. Defaults to 1.
            * sn_band (int, optional): The band number for the north-south component. Defaults to 2.
            * vert_band (int, optional): The band number for the vertical component. Defaults to None.
            * factor_to_m (float, optional): Factor to convert to meters. Defaults to 1.0.
            * zero2nan (bool, optional): Whether to convert zeros to NaN. Defaults to True.
            * wavelength (float, optional): Wavelength of the radar. Defaults to None.
            
        """
        if directory_name is None:
            directory_name = self.directory_name
        assert filename is not None, "Filename must be provided."
        assert ew_band is not None, "East-West band must be provided."
        assert sn_band is not None, "North-South band must be provided."

        file_path = os.path.join(directory_name, filename)
        data_ew = read_tiff_with_metadata(file_path, band_index=ew_band, factor=factor_to_m, meshout=True)
        data_sn = read_tiff_with_metadata(file_path, band_index=sn_band, factor=factor_to_m, meshout=False)

        # 提取经纬度网格和数据
        mesh_lon, mesh_lat = data_ew['mesh_lon'], data_ew['mesh_lat']

        if zero2nan:
            data_ew['data'][data_ew['data'] == 0] = np.nan
            data_sn['data'][data_sn['data'] == 0] = np.nan

        # Save in self
        self.factor_to_m = factor_to_m
        self.wavelength = wavelength
        self.raw_east = data_ew['data']
        self.raw_north = data_sn['data']
        self.raw_lon = mesh_lon.mean(axis=0)
        self.raw_lat = mesh_lat.mean(axis=1)
        self.raw_mesh_lon = mesh_lon
        self.raw_mesh_lat = mesh_lat
        self.im_geotrans = data_ew['geotrans']
        self.im_proj = data_ew['proj']
    
    def read_from_tiff(self, remove_nan=True):
        east = self.raw_east
        north = self.raw_north
        mesh_lon = self.raw_mesh_lon
        mesh_lat = self.raw_mesh_lat
        # Read SAR data from binary files using inherited method
        self.read_from_binary(east, north, lon=mesh_lon, lat=mesh_lat, 
                             remove_nan=remove_nan)

    def to_xarray_dataarray(self, data='east'):
        '''
        Also for GMT plot

        Convert the velocity data to an xarray DataArray for easier manipulation and plotting.

        Parameters:
            * data (str): The type of data to convert ('east', 'north', or 'vertical').

        Returns:
            * xarray.DataArray: The converted data array.
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
        raw_east = self.raw_east[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
        raw_north = self.raw_north[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
        coordrange = [mesh_lon.min(), mesh_lon.max(), mesh_lat.min(), mesh_lat.max()]
    
        if inplace:
            self.raw_east = raw_east
            self.raw_north = raw_north
            self.raw_mesh_lon = mesh_lon
            self.raw_mesh_lat = mesh_lat
            self.raw_lon = mesh_lon[0, :]
            self.raw_lat = mesh_lat[:, 0]
        return raw_east, raw_north, mesh_lon, mesh_lat, coordrange
    
    def select_pixels(self, minlon, maxlon, minlat, maxlat):
        self.cut_raw_sar([minlon, maxlon], [minlat, maxlat], inplace=True)
        return super().select_pixels(minlon, maxlon, minlat, maxlat)
    
    def plot_raw_sar(self, data=['east'], coordrange=None, cmap='jet', vmin=None, vmax=None, 
                     title=None, savefig=None, figsize=(7, 4), dpi=300, show=True, 
                     add_colorbar=True, cb_label='Amplitude', faults=None, 
                     trace_color='red', trace_linewidth=0.5, style=['science'], fontsize=None,
                     colorbar_length=0.4, colorbar_height=0.02, colorbar_x=0.1, colorbar_y=0.1,
                     colorbar_orientation='vertical', cb_label_loc=None, tickfontsize=10, labelfontsize=10,
                     unified_colorbar=False, sharey=True, equal_aspect=False):
        """
        Plot the raw optical or SAR data.
    
        Parameters:
            data (str or list): The type(s) of data to plot ('east', 'north', or 'vertical').
                                Can be a single string or a list of strings.
            coordrange (list, optional): Coordinate range for plotting [lon_min, lon_max, lat_min, lat_max].
            cmap (str): The colormap to use for the plot.
            vmin (float or list, optional): The minimum value(s) for the color scale. If a single value is provided, all subplots share it.
            vmax (float or list, optional): The maximum value(s) for the color scale. If a single value is provided, all subplots share it.
            title (str or list, optional): The title(s) of the plot(s). If multiple data are plotted, provide a list of titles.
            savefig (str, optional): The filename to save the plot. If None, the plot will not be saved.
            figsize (tuple): The size of the figure.
            dpi (int): The resolution of the figure.
            show (bool): Whether to display the plot.
            add_colorbar (bool): Whether to add a colorbar to the plot.
            cb_label (str or list): Label(s) for the colorbar. If multiple data are plotted, provide a list of labels.
            faults (list, optional): Fault lines to plot (list of DataFrames or objects with lon/lat attributes).
            trace_color (str): Color for fault lines.
            trace_linewidth (float): Line width for fault lines.
            style (list): Plotting style (e.g., ['science']).
            fontsize (int, optional): Font size for the plot.
            colorbar_length (float): Length of the colorbar.
            colorbar_height (float): Height of the colorbar.
            colorbar_x (float): X position of the colorbar.
            colorbar_y (float): Y position of the colorbar.
            colorbar_orientation (str): Orientation of the colorbar ('horizontal' or 'vertical').
            cb_label_loc (str, optional): Location of the colorbar label.
            tickfontsize (int): Font size for colorbar ticks.
            labelfontsize (int): Font size for colorbar label.
            unified_colorbar (bool): Whether to use a unified colorbar for all subplots.
            sharey (bool): Whether to share the y-axis across subplots.
            equal_aspect (bool): Whether to set equal aspect ratio for the plots.
    
        Returns:
            matplotlib.figure.Figure: The figure object of the plot.
        """
        # Ensure data is a list for consistent handling
        if isinstance(data, str):
            data = [data]
        if isinstance(title, str):
            title = [title] * len(data)
        if isinstance(cb_label, str):
            cb_label = [cb_label] * len(data)
        if isinstance(vmin, (int, float)):
            vmin = [vmin] * len(data)
        if isinstance(vmax, (int, float)):
            vmax = [vmax] * len(data)
    
        # Validate data types
        valid_data_types = {'east': self.raw_east, 'north': self.raw_north}
        for d in data:
            if d not in valid_data_types:
                raise ValueError(f"Invalid data type '{d}'. Choose from {list(valid_data_types.keys())}.")
    
        # Handle coordinate range
        if coordrange is not None:
            lon_range, lat_range = coordrange[:2], coordrange[2:]
            coordrange = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
            east, north, mesh_lon, mesh_lat, coordrange = self.cut_raw_sar(lon_range, lat_range)
            data_dict = {'east': east, 'north': north}
            cut_data = {d: data_dict[d] for d in data}
        else:
            coordrange = [self.raw_mesh_lon.min(), self.raw_mesh_lon.max(), 
                          self.raw_mesh_lat.min(), self.raw_mesh_lat.max()]
            mesh_lon = self.raw_mesh_lon
            mesh_lat = self.raw_mesh_lat
            cut_data = {d: valid_data_types[d] for d in data}
    
        # Determine unified vmin and vmax if unified_colorbar is True
        if unified_colorbar:
            unified_vmin = min([np.nanmin(cut_data[d]) for d in data])
            unified_vmax = max([np.nanmax(cut_data[d]) for d in data])
            vmin = [unified_vmin] * len(data)
            vmax = [unified_vmax] * len(data)
    
        # Create subplots for multiple data
        n_plots = len(data)
        with sci_plot_style(style=style, fontsize=fontsize, figsize=(figsize[0] * n_plots, figsize[1])):
            fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0] * n_plots, figsize[1]), 
                                     dpi=dpi, squeeze=False, sharey=sharey)
            axes = axes.flatten()
    
            for i, d in enumerate(data):
                data_to_plot = cut_data[d]
    
                # Plot the data
                im = axes[i].pcolormesh(mesh_lon, mesh_lat, data_to_plot, cmap=cmap, shading='auto', vmin=vmin[i], vmax=vmax[i])
                axes[i].set_title(title[i] if title else f"Raw {d.capitalize()} Data")
                set_degree_formatter(axes[i], axis='both')
                # axes[i].set_xlabel("Longitude")
                # axes[i].set_ylabel("Latitude")

                if equal_aspect:
                    axes[i].set_aspect('equal')
    
                # Add fault lines if provided
                if faults is not None:
                    for fault in faults:
                        if isinstance(fault, pd.DataFrame):
                            axes[i].plot(fault.lon.values, fault.lat.values, color=trace_color, lw=trace_linewidth)
                        else:
                            axes[i].plot(fault.lon, fault.lat, color=trace_color, lw=trace_linewidth)
    
                # Add colorbar
                if add_colorbar and not unified_colorbar:
                    cbar = fig.colorbar(im, ax=axes[i], orientation=colorbar_orientation)
                    cbar.set_label(cb_label[i] if cb_label else "Displacement")
                    cbar.ax.tick_params(labelsize=tickfontsize)
    
            # Add unified colorbar if requested
            if add_colorbar and unified_colorbar:
                cbar_ax = fig.add_axes([colorbar_x, colorbar_y, colorbar_length, colorbar_height])  # [left, bottom, width, height]
                cbar = fig.colorbar(im, cax=cbar_ax, orientation=colorbar_orientation)
                cbar.set_label(cb_label[0] if cb_label else "Amplitude", fontdict={'size': labelfontsize})
                cbar.ax.tick_params(labelsize=tickfontsize)
    
            # Save the figure if requested
            if savefig:
                plt.savefig(savefig, bbox_inches='tight')
    
            # Show the plot
            if show:
                plt.show()
    
        return fig

    def save_outputs_as_tiff(self, directory_name, save_east=True, save_north=True, extent=None, grid_resolution=None):
        """
        Save azi, inc, and vel as GeoTIFF files, with optional resampling to a regular grid and user-defined extent.
    
        Parameters:
        - directory_name: str, directory to save the TIFF files.
        - save_east: bool, whether to save east component.
        - save_north: bool, whether to save north component.
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
    
        # Save velocity data
        if save_east:
            vel_data = self.raw_east
            vel_lon, vel_lat = self.raw_mesh_lon, self.raw_mesh_lat
            if grid_resolution and extent:
                vel_data, vel_lon, vel_lat = resample_to_regular_grid(vel_data, vel_lon, vel_lat, extent, grid_resolution)
            vel_file = os.path.join(directory_name, 'EWdisp.tif')
            save_to_tiff(vel_data, vel_lon, vel_lat, vel_file)
        
        if save_north:
            vel_data = self.raw_north
            vel_lon, vel_lat = self.raw_mesh_lon, self.raw_mesh_lat
            if grid_resolution and extent:
                vel_data, vel_lon, vel_lat = resample_to_regular_grid(vel_data, vel_lon, vel_lat, extent, grid_resolution)
            vel_file = os.path.join(directory_name, 'NSdisp.tif')
            save_to_tiff(vel_data, vel_lon, vel_lat, vel_file)