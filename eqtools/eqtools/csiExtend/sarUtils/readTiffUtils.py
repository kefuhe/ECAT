# insar导入需要在gdal前面可能存在cartopy，shapely库冲突放在其后
from osgeo import gdal
from osgeo import osr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ...plottools import set_degree_formatter, sci_plot_style


# -------------------------For 3D SAR Displacements------------------------------#
def read_tiff_group(file_paths, keys=None):
    """
    Read multiple TIFF files and organize them into a dictionary with shared metadata.

    Parameters:
    file_paths (list of str or str): List of file paths or a single file path to read the data from.
    keys (list of str or str, optional): List of keys or a single key to use for the data dictionary. 
                                       If None, the file name prefixes will be used as keys.

    Returns:
    dict: A dictionary containing the TIFF data and shared metadata.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    if isinstance(keys, str):
        keys = [keys]

    if keys is not None and len(keys) != len(file_paths):
        raise ValueError("The length of keys must match the length of file_paths.")

    data_dict = {}
    metadata = {}

    for i, file_path in enumerate(file_paths):
        data, im_geotrans, im_proj, im_width, im_height = read_tiff(file_path)
        x_lon, y_step, x_lat, y_step, mesh_lon, mesh_lat = read_tiff_info(file_path, im_width, im_height)

        key = keys[i] if keys else file_path.split('.')[0]
        data_dict[key] = data

        if not metadata:
            metadata = {
                'geotrans': im_geotrans,
                'proj': im_proj,
                'width': im_width,
                'height': im_height,
                'x_lon': x_lon,
                'y_step': y_step,
                'x_lat': x_lat,
                'mesh_lon': mesh_lon,
                'mesh_lat': mesh_lat
            }

    data_dict['metadata'] = metadata
    return data_dict

def plot_displacement_data(displacement_data=None, sigma_data=None,
                           disp_keys=['E', 'N', 'U'],
                           sigma_keys=['E Sigma', 'N Sigma', 'U Sigma'],
                           disp_range=(-4, 4),
                           sigma_range=(0, 1),
                           disp_cmap=None,
                           sigma_cmap=None,
                           lon_range=None,
                           lat_range=None,
                           font_size=8,
                           figsize=None,
                           output_file='displacement.pdf',
                           faults=None,
                           fault_traces=None,
                           trace_color='black',
                           start_label='a'):
    """
    Plot displacement data with optional sigma (uncertainty) data and fault traces.
    
    Parameters:
    * displacement_data : dict, optional - Dictionary containing displacement data and metadata
    * sigma_data : dict, optional - Dictionary containing sigma data and metadata
    * disp_keys : list - Keys for displacement data plots
    * sigma_keys : list - Keys for sigma data plots (used only if sigma_data is provided)
    * disp_range : tuple or dict - (min, max) range for displacement colormap or a dictionary with keys 'E', 'N', 'U'
    * sigma_range : tuple - (min, max) range for sigma colormap
    * disp_cmap : str, optional - Colormap for displacement data
    * sigma_cmap : str, optional - Colormap for sigma data
    * lon_range : tuple, optional - (min, max) range for longitude. If None, uses metadata range
    * lat_range : tuple, optional - (min, max) range for latitude. If None, uses metadata range
    * font_size : int - Font size for labels and ticks
    * figsize : tuple, optional - Figure size in inches
    * output_file : str - Output file path
    * faults : list, optional - List of fault objects with lon and lat attributes
    * fault_traces : list, optional - List of arrays with fault trace coordinates
    * trace_color : str - Color for fault traces (default is 'black')
    * start_label : str - Starting label for subplots (default is 'a')
    """
    if displacement_data is None and sigma_data is None:
        raise ValueError("Both displacement_data and sigma_data cannot be None.")

    # Extract metadata and set default ranges
    if displacement_data is not None:
        metadata = displacement_data['metadata']
        x_lon = metadata['x_lon']
        x_lat = metadata['x_lat']
    elif sigma_data is not None:
        metadata = sigma_data['metadata']
        x_lon = metadata['x_lon']
        x_lat = metadata['x_lat']
    
    if lon_range is None:
        lon_range = (np.min(x_lon), np.max(x_lon))
    if lat_range is None:
        lat_range = (np.min(x_lat), np.max(x_lat))

    # Generate subplot labels
    labels = [chr(ord(start_label) + i) for i in range(6)]

    with sci_plot_style(pdf_fonttype=42):
        # Adjust figure layout based on whether sigma data is provided
        if sigma_data is None:
            fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5) if figsize is None else figsize)
            axes = [axes]  # Make it 2D array for consistent indexing
        else:
            fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2) if figsize is None else figsize)

        # Plot displacement data
        if displacement_data is not None:
            for i, (ax, key) in enumerate(zip(axes[0], disp_keys)):
                im_data = displacement_data[key]
                if isinstance(disp_range, dict):
                    vmin, vmax = disp_range.get(key, (-4, 4))
                else:
                    vmin, vmax = disp_range
                cax = ax.imshow(im_data, extent=[x_lon[0], x_lon[-1], x_lat[0], x_lat[-1]], 
                              cmap='cmc.roma_r' if disp_cmap is None else disp_cmap, origin='lower', vmin=vmin, vmax=vmax)
                ax.set_xlim(lon_range)
                ax.set_ylim(lat_range)
                
                ax.text(0.05, 0.95, f'({labels[i]}) {key} Displacement', 
                       transform=ax.transAxes,
                       fontsize=font_size,
                       verticalalignment='top',
                       horizontalalignment='left')
                
                set_degree_formatter(ax, axis='both')
        
                cax_bounds = [0.65, 0.15, 0.3, 0.03]
                cax_pos = ax.inset_axes(cax_bounds)
                
                cbar = fig.colorbar(cax, cax=cax_pos, orientation='horizontal', extend='both')
                cbar.ax.xaxis.set_label_position('top')
                cbar.set_label('m', size=font_size)
                cbar.ax.tick_params(labelsize=font_size)
                cbar.set_ticks([vmin, 0, vmax])

                # Plot fault traces if provided
                if faults is not None:
                    for fault in faults:
                        color = fault.color if hasattr(fault, 'color') and fault.color is not None else trace_color
                        ax.plot(fault.lon, fault.lat, color=color, linewidth=1.5)

                if fault_traces is not None:
                    for trace in fault_traces:
                        ax.plot(trace[:, 0], trace[:, 1], color=trace_color, linewidth=1.5)

        # Plot sigma data if provided
        if sigma_data is not None:
            if sigma_cmap is None:
                colors = ['#ab8d68', 'white']
                custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
            else:
                custom_cmap = sigma_cmap

            for i, (ax, key) in enumerate(zip(axes[1], sigma_keys)):
                im_data = sigma_data[key]
                cax = ax.imshow(im_data, extent=[x_lon[0], x_lon[-1], x_lat[0], x_lat[-1]], 
                              cmap=custom_cmap, origin='lower', vmin=sigma_range[0], vmax=sigma_range[1])
                ax.set_xlim(lon_range)
                ax.set_ylim(lat_range)
                
                ax.text(0.05, 0.95, f'({labels[i + 3]}) {key}', 
                       transform=ax.transAxes,
                       fontsize=font_size,
                       verticalalignment='top',
                       horizontalalignment='left')
                
                set_degree_formatter(ax, axis='both')
        
                cax_bounds = [0.65, 0.15, 0.3, 0.03]
                cax_pos = ax.inset_axes(cax_bounds)
                
                cbar = fig.colorbar(cax, cax=cax_pos, orientation='horizontal', extend='max')
                cbar.ax.xaxis.set_label_position('top')
                cbar.set_label('m', size=font_size)
                cbar.ax.tick_params(labelsize=font_size)
                cbar.set_ticks([sigma_range[0], sigma_range[1]/2, sigma_range[1]])

                # Plot fault traces if provided
                if faults is not None:
                    for fault in faults:
                        color = fault.color if hasattr(fault, 'color') and fault.color is not None else trace_color
                        ax.plot(fault.lon, fault.lat, color=color, linewidth=1.5)

                if fault_traces is not None:
                    for trace in fault_traces:
                        ax.plot(trace[:, 0], trace[:, 1], color=trace_color, linewidth=1.5)
    
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.show()
# -------------------------For 3D SAR Displacements------------------------------#
def read_tiff_with_metadata(unwfile, band_index=1, factor=1.0, meshout=True):
    """
    Read data from a specific band of a TIFF file and retrieve metadata including geotransform,
    projection, and coordinate mesh grids.

    Parameters:
        unwfile (str): Path to the TIFF file.
        band_index (int, optional): Band index to read (1-based). Defaults to 1.
        factor (float, optional): Factor to multiply the data values. Defaults to 1.0.
        meshout (bool, optional): Whether to output mesh grids for longitude and latitude. Defaults to True.

    Returns:
        dict: A dictionary containing the following keys:
            - 'data' (numpy.ndarray): Data matrix of the specified band.
            - 'geotrans' (tuple): Geotransform (origin and pixel size).
            - 'proj' (str): Projection parameters.
            - 'width' (int): Number of columns (pixels).
            - 'height' (int): Number of rows (pixels).
            - 'x_lon' (numpy.ndarray): Longitude values along the x-axis.
            - 'y_lat' (numpy.ndarray): Latitude values along the y-axis.
            - 'mesh_lon' (numpy.ndarray, optional): Longitude mesh grid (if meshout=True).
            - 'mesh_lat' (numpy.ndarray, optional): Latitude mesh grid (if meshout=True).
    """
    # Read the specified band data and metadata
    im_data, im_geotrans, im_proj, im_width, im_height = read_tiff(unwfile, band_index=band_index, factor=factor)

    # Get coordinate system and range information
    if meshout:
        x_lon, y_step, y_lat, y_step, mesh_lon_x, mesh_lat_y = read_tiff_info(unwfile, im_width, im_height, meshout=meshout)
    else:
        x_lon, x_step, y_lat, y_step = read_tiff_info(unwfile, im_width, im_height, meshout=meshout)

    # Convert UTM coordinates to latitude and longitude if necessary
    if meshout:
        im_height, im_width = mesh_lon_x.shape
        mesh_lat, mesh_lon = utm_to_latlon(mesh_lon_x.flatten(), mesh_lat_y.flatten(), im_proj)
        mesh_lat = mesh_lat.reshape(im_height, im_width)
        mesh_lon = mesh_lon.reshape(im_height, im_width)

    # Construct the result dictionary
    result = {
        'data': im_data,
        'geotrans': im_geotrans,
        'proj': im_proj,
        'width': im_width,
        'height': im_height,
        'x_lon': x_lon,
        'y_lat': y_lat
    }

    if meshout:
        result['mesh_lon'] = mesh_lon
        result['mesh_lat'] = mesh_lat

    return result


def read_tiff(unwfile, band_index=1, factor=1.0):
    """
    Read a specific band from a TIFF file.

    Parameters:
        unwfile (str): TIFF format image file.
        band_index (int, optional): The band index to read (1-based). Defaults to 1.
        factor (float, optional): Factor to multiply the data values. Defaults to 1.0.

    Returns:
        tuple: 
            - im_data (numpy.ndarray): Data matrix of the specified band.
            - im_geotrans (tuple): Geotransform (origin and pixel size).
            - im_proj (str): Projection parameters.
            - im_width (int): Number of columns (pixels).
            - im_height (int): Number of rows (pixels).
    """
    dataset = gdal.Open(unwfile, gdal.GA_ReadOnly)  # Open file
    if dataset is None:
        raise FileNotFoundError(f"Unable to open file: {unwfile}")

    im_width = dataset.RasterXSize  # Number of columns in the raster matrix
    im_height = dataset.RasterYSize  # Number of rows in the raster matrix
    im_bands = dataset.RasterCount  # Number of bands

    if band_index < 1 or band_index > im_bands:
        raise ValueError(f"Invalid band_index {band_index}. The file has {im_bands} band(s).")

    im_geotrans = dataset.GetGeoTransform()  # Affine transform (origin and pixel size)
    im_proj = dataset.GetProjection()  # Map projection information (as a string)
    im_band = dataset.GetRasterBand(band_index)  # Get the specified band
    im_data = im_band.ReadAsArray(0, 0, im_width, im_height)  # Store raster image values as a data matrix
    im_data *= factor  # Apply the factor to the data

    del dataset  # Close the dataset
    return im_data, im_geotrans, im_proj, im_width, im_height


def read_tiff_info(unwfile, im_width, im_height, meshout=True):
    '''
    Objective:
        * Get the coordinate system and range of the x and y axes from a TIFF file
    Input:
        * unwfile     : TIFF file name
    Output:
        * x_lon, y_step, x_lat, y_step
    '''

    metadata = gdal.Info(unwfile, format='json', deserialize=True)
    upperLeft = metadata['cornerCoordinates']['upperLeft']
    lowerRight = metadata['cornerCoordinates']['lowerRight']
    x_upperleft, y_upperleft = upperLeft
    x_lowerright, y_lowerright = lowerRight
    x_step = (x_upperleft - x_lowerright) / im_width
    y_step = -(y_upperleft - y_lowerright) / im_height
    x_lon = np.linspace(x_upperleft, x_lowerright, im_width)
    y_lat = np.linspace(y_upperleft, y_lowerright, im_height)

    if meshout:
        mesh_lon_x, mesh_lat_y = np.meshgrid(x_lon, y_lat)
        return x_lon, x_step, y_lat, y_step, mesh_lon_x, mesh_lat_y
    else:
        return x_lon, x_step, y_lat, y_step


def write_tiff(filename, im_proj, im_geotrans, im_data=None):
    # Determine the data type of the raster data
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
        
    # Determine the number of dimensions of the array
    # im_data.shape may have three elements, corresponding to a multi-dimensional matrix (i.e., multiple bands);
    # or it may have two elements, corresponding to a two-dimensional matrix (i.e., a single band)
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


def save_to_tiff(data, lon, lat, output_file, dtype=gdal.GDT_Float32):
    """
    Save a 2D array as a GeoTIFF file.

    Parameters:
    - data: numpy.ndarray, 2D array to save.
    - lon: numpy.ndarray, longitude array.
    - lat: numpy.ndarray, latitude array.
    - output_file: str, path to the output GeoTIFF file.
    - dtype: gdal data type, default is gdal.GDT_Float32.
    """
    # Ensure lon and lat are numpy arrays
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # Check for NaN or invalid values in lon and lat
    if np.any(np.isnan(lon)) or np.any(np.isnan(lat)):
        raise ValueError("Longitude or latitude contains NaN values. Please check the input data.")

    # Ensure lon and lat are 1D arrays
    if lon.ndim > 1:
        lon = lon[0, :]  # Take the first row if it's a 2D array
    if lat.ndim > 1:
        lat = lat[:, 0]  # Take the first column if it's a 2D array

    # Ensure data is a 2D array
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")

    nrows, ncols = data.shape
    if len(lon) != ncols or len(lat) != nrows:
        raise ValueError("Longitude and latitude dimensions do not match the data dimensions.")

    # Create GeoTIFF file
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_file, ncols, nrows, 1, dtype)

    # Set geotransform (top-left corner and pixel size)
    lon_step = lon[1] - lon[0] if len(lon) > 1 else 0
    lat_step = lat[1] - lat[0] if len(lat) > 1 else 0
    geotransform = (lon.min(), lon_step, 0, lat.max(), 0, -lat_step)
    dataset.SetGeoTransform(geotransform)

    # Set projection (WGS84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # EPSG:4326 corresponds to WGS84
    dataset.SetProjection(srs.ExportToWkt())

    # Write data to the GeoTIFF
    dataset.GetRasterBand(1).WriteArray(data)
    dataset.FlushCache()
    print(f"Saved GeoTIFF: {output_file}")