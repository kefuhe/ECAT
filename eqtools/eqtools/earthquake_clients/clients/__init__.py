import os
import json
import logging
import warnings
from datetime import datetime

from .usgs_client import USGSClient
from .gcmt_client import GCMTClient
from .iris_client import IRISClient
from .logging_config import logger

class EarthquakeClientFactory:
    @staticmethod
    def create_client(client_type, **kwargs):
        """
        Factory method to create earthquake client instances.

        Args:
            client_type (str): The type of the client to create ('fdsn', 'usgs', 'gcmt', 'iris').
            **kwargs: Additional keyword arguments to pass to the client constructor.

        Returns:
            An instance of the requested earthquake client.
        """
        if client_type == 'usgs':
            return USGSClient(**kwargs)
        elif client_type == 'gcmt':
            return GCMTClient(**kwargs)
        elif client_type == 'iris':
            return IRISClient(**kwargs)
        else:
            logger.error(f"Unknown client type: {client_type}")
            raise ValueError(f"Unknown client type: {client_type}")

    @staticmethod
    def download_china_earthquakes(client_type='usgs', start_time="1990-01-01", end_time=None, 
                                   min_magnitude=0.0, max_magnitude=9.0, output_file="china_earthquake_catalog.csv", **kwargs):
        """
        Download earthquake data for China region (including Taiwan).

        Args:
            client_type (str): The type of the client to create ('fdsn', 'usgs', 'gcmt', 'iris').
            start_time (str): The start time for the earthquake data.
            end_time (str): The end time for the earthquake data.
            min_magnitude (float): The minimum magnitude of the earthquakes.
            max_magnitude (float): The maximum magnitude of the earthquakes.
            output_file (str): The output file to save the earthquake data.
            **kwargs: Additional keyword arguments to pass to the client constructor.
        """
        client = EarthquakeClientFactory.create_client(client_type, **kwargs)
        safe_output_file = os.path.join(os.path.dirname(output_file), f"{client_type}_{os.path.basename(output_file)}")
        end_time = end_time if end_time else datetime.now().strftime("%Y-%m-%d")
        client.get_events(start_time=start_time, end_time=end_time, min_magnitude=min_magnitude, max_magnitude=max_magnitude, output_file=safe_output_file,
                          min_longitude=73, max_longitude=135.1, min_latitude=3.5, max_latitude=53.5)

    @staticmethod
    def download_global_earthquakes(client_type='usgs', start_time="1900-01-01", end_time=None, 
                                    min_magnitude=6.0, max_magnitude=9.0, output_file="global_earthquake_catalog.csv", **kwargs):
        """
        Download global earthquake data.

        Args:
            client_type (str): The type of the client to create ('fdsn', 'usgs', 'gcmt', 'iris').
            start_time (str): The start time for the earthquake data.
            end_time (str): The end time for the earthquake data.
            min_magnitude (float): The minimum magnitude of the earthquakes.
            max_magnitude (float): The maximum magnitude of the earthquakes.
            output_file (str): The output file to save the earthquake data.
            **kwargs: Additional keyword arguments to pass to the client constructor.
        """
        client = EarthquakeClientFactory.create_client(client_type, **kwargs)
        safe_output_file = os.path.join(os.path.dirname(output_file), f"{client_type}_{os.path.basename(output_file)}")
        end_time = end_time if end_time else datetime.now().strftime("%Y-%m-%d")
        client.get_events(start_time=start_time, end_time=end_time, min_magnitude=min_magnitude, max_magnitude=max_magnitude, output_file=safe_output_file)
    
    @staticmethod
    def plot_earthquakes_on_map(csv_file, output_file=None, dpi=600, plot_faults=False, selected_faults=None, 
                                plot_blocks=True, selected_blocks=None, plot_beachballs=False):
        """
        Plot earthquakes on a map using data from a CSV file.
    
        Args:
            csv_file (str): The path to the CSV file containing earthquake data.
            output_file (str): The path to save the output plot image. If None, use the CSV file name with .png extension.
            dpi (int): The resolution of the output plot image.
            plot_faults (bool): Whether to plot fault data. Default is True.
            selected_faults (list): List of selected fault data file names to plot. Default is None (all faults).
            plot_blocks (bool): Whether to plot blocks data. Default is True.
            selected_blocks (list): List of selected blocks data file names to plot. Default is None (all blocks).
            plot_beachballs (bool): Whether to plot beachballs instead of scatter. Default is False.
        """
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cmcrameri  # noqa: F401 - registers the cmc colormaps
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import matplotlib.dates as mdates
        from obspy.imaging.beachball import beach
        from ...gmttools import read_gmt_lines
        from ..utils.file_utils import (
            find_and_prioritize_files,
            load_geojson_data,
        )
    
        # Load earthquake data from CSV
        df = pd.read_csv(csv_file)
    
        # Parse nodal_plane1 and nodal_plane2 columns as JSON
        df['nodal_plane1'] = df['nodal_plane1'].apply(lambda x: json.loads(x.replace("'", "\"")) if pd.notnull(x) else None)
        df['nodal_plane2'] = df['nodal_plane2'].apply(lambda x: json.loads(x.replace("'", "\"")) if pd.notnull(x) else None)
    
        # Extract time range
        df['time'] = pd.to_datetime(df['time'])
        start_time = df['time'].min().strftime('%Y-%m-%d')
        end_time = df['time'].max().strftime('%Y-%m-%d')
    
        # Convert time to float years for color mapping
        df['year_float'] = df['time'].dt.year + df['time'].dt.dayofyear / 365.25
        norm = mcolors.Normalize(vmin=df['year_float'].min(), vmax=df['year_float'].max())
        cmap = cm.get_cmap('plasma')  # Use a more distinct colormap
        cmap = cm.get_cmap('cmc.hawaii')
    
        # Create a map
        fig = plt.figure(figsize=(11.7, 8.3), dpi=300)  # Set A4 size with 300 dpi
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, edgecolor='black', zorder=2)
        ax.add_feature(cfeature.OCEAN, zorder=0)
        ax.add_feature(cfeature.COASTLINE, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=2)

        # Add gridlines with labels
        gl = ax.gridlines(draw_labels=True, color='#fcfdfe', linestyle='-', zorder=1)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}
    
        # Set map extent based on earthquake data
        min_lon, max_lon = df['longitude'].min() - 1, df['longitude'].max() + 1
        min_lat, max_lat = df['latitude'].min() - 1, df['latitude'].max() + 1
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        def plot_faults_and_blocks(data_dir, selected_files, color, linewidth):
            selected_files = find_and_prioritize_files(data_dir, ['.gmt', '.json'], selected_files)
            for file in selected_files:
                if file.endswith('.gmt'):
                    fault_segments = read_gmt_lines(file)
                    for segment in fault_segments:
                        ax.plot(segment['X'], segment['Y'], transform=ccrs.PlateCarree(), color=color, linewidth=linewidth, zorder=3)
                elif file.endswith('.json'):
                    fault_data = load_geojson_data(file)
                    if fault_data:
                        for feature in fault_data['features']:
                            geometry_type = feature['geometry']['type']
                            if geometry_type == 'LineString':
                                coordinates = feature['geometry']['coordinates']
                                filtered_coordinates = [(lon, lat) for lon, lat, *_ in coordinates]
                                lon, lat = zip(*filtered_coordinates)
                                ax.plot(lon, lat, transform=ccrs.PlateCarree(), color=color, linewidth=linewidth, zorder=3)
                            elif geometry_type == 'MultiLineString':
                                for line in feature['geometry']['coordinates']:
                                    filtered_coordinates = [(lon, lat) for lon, lat, *_ in line]
                                    lon, lat = zip(*filtered_coordinates)
                                    ax.plot(lon, lat, transform=ccrs.PlateCarree(), color=color, linewidth=linewidth, zorder=3)

        if plot_faults:
            fault_dir = 'data/Faults'
            plot_faults_and_blocks(fault_dir, selected_faults, 'red', 0.5)

        if plot_blocks:
            blocks_dir = 'data/Blocks'
            plot_faults_and_blocks(blocks_dir, selected_blocks, 'blue', 1)
    
        # Plot earthquakes with beachballs or scatter
        if plot_beachballs:
            for _, row in df.iterrows():
                if row['nodal_plane1'] is not None:
                    nodal_plane1 = row['nodal_plane1']
                    mt = [float(nodal_plane1['strike']), float(nodal_plane1['dip']), float(nodal_plane1['rake'])]
                    magnitude = row['magnitude']
                    beachball_size = magnitude * 0.25  # Adjust the size factor as needed
                    beachball = beach(mt, xy=(row['longitude'], row['latitude']), width=beachball_size, linewidth=0.5,
                                      facecolor=cmap(norm(row['year_float'])), alpha=0.8, zorder=4)
                    ax.add_collection(beachball)
        else:
            ax.scatter(
                df['longitude'], df['latitude'], 
                s=(df['magnitude'] - df['magnitude'].min() + 1) * 100,  # Increase size range
                c=df['year_float'], 
                cmap=cmap,
                norm=norm,
                alpha=0.6, edgecolors='k', transform=ccrs.PlateCarree(), zorder=4,
            )
        
        # Add color bar for time with date format
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, aspect=30)
        cbar.set_label('Date')
        
        # Set color bar ticks and labels
        tick_locs = np.linspace(df['year_float'].min(), df['year_float'].max(), num=5)
        cbar.set_ticks(tick_locs)
        tick_labels = [pd.Timestamp(year=int(tick), month=1, day=1) + pd.Timedelta(days=(tick - int(tick)) * 365.25) for tick in tick_locs]
        cbar.set_ticklabels([tick.strftime('%Y-%m-%d') for tick in tick_labels])
    
        # Create legend for magnitudes
        min_magnitude = np.floor(df['magnitude'].min() * 2) / 2
        max_magnitude = np.ceil(df['magnitude'].max() * 2) / 2
        legend_magnitudes = np.arange(min_magnitude, max_magnitude + 0.5, 0.5)
        for mag in legend_magnitudes:
            plt.scatter([], [], s=(mag - df['magnitude'].min() + 1) * 100, c='k', alpha=0.6,
                        label=str(mag))
        legend = plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title='Magnitude', loc='lower right')
        frame = legend.get_frame()
        frame.set_edgecolor('black')
    
        # Set title with date range
        plt.title(f'Global Earthquakes\n{start_time} to {end_time}')
    
        # Save the plot
        if output_file is None:
            output_file = os.path.splitext(csv_file)[0] + '.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
        # Show the plot
        plt.show()

if __name__ == '__main__':
    # Set logging level to INFO if debugging information is needed
    logger.setLevel(logging.INFO)

    # Suppress cartopy download warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")

    # Download earthquake data for the China region
    EarthquakeClientFactory.download_china_earthquakes(client_type='usgs', min_magnitude=6.7, include_focal_mechanism=True)

    # Download global earthquakes with magnitude 6.0 and above
    EarthquakeClientFactory.download_global_earthquakes(client_type='usgs', min_magnitude=6.0, output_file="global_earthquake_catalog_6.csv")

    # Download global earthquakes with magnitude 7.0 and above
    EarthquakeClientFactory.download_global_earthquakes(client_type='usgs', min_magnitude=7.0, output_file="global_earthquake_catalog_7.csv")

    # Plot global earthquakes with magnitude 6.0 and above on the map
    EarthquakeClientFactory.plot_earthquakes_on_map("global_earthquake_catalog_6.csv")

    # Plot global earthquakes with magnitude 7.0 and above on the map
    EarthquakeClientFactory.plot_earthquakes_on_map("global_earthquake_catalog_7.csv")
