import numpy as np
import os
from glob import glob

from .readBase2csisar import ReadBase2csisar, Hyp3TiffConfig, GammaTiffConfig
from .readTiffUtils import read_tiff, read_tiff_info, utm_to_latlon

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
        self.raw_lon = mesh_lon.mean(axis=0)
        self.raw_lat = mesh_lat.mean(axis=1)
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
        else:
            self.vel = -self.vel
            self.raw_vel = -self.raw_vel

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


if __name__ == '__main__':
    pass
    # plt.imshow(los, interpolation='none', # cmap=cmap, norm=norm,
    #          origin='upper', extent=[np.nanmin(x_lon), np.nanmax(x_lon), np.nanmin(x_lat), np.nanmax(x_lat)], aspect='auto', vmax=1, vmin=-1)