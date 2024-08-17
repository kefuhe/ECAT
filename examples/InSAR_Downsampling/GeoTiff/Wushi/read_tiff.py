# External libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

# CSI routines and ECAT routines
import csi.insar as insar
import csi.imagecovariance as imcov
from eqtools.csiExtend.sarUtils.readTiff2csisar import TiffsarReader, GammaTiffReader

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    # set some flags
    input_check = True
    do_downsample = False
    output_check = False

    # CSI does most computations in a local Cartesian coordinate system. 
    # It presently supports only UTM projections, so we need to set the zone or centrial lon0 and lat0 to use.
    lon0, lat0 = -68.5, -23.0

    # Prepare InSAR data for Sentinel-1
    outName = 'S1_T128A'
    dirname = os.path.join('.', 'insar')
    unwphasename = 'Wushi_asc_unw.tif'
    aziname = 'Wushi_asc_azi.tif'
    incname = 'Wushi_asc_inc.tif'
    mysar = GammaTiffReader(name='Wushi', lon0=lon0, lat0=lat0, directory_name=dirname)
    mysar.extract_raw_grd(phsname=unwphasename, azifile=aziname, incfile=incname)
                        #   azi_reference='north', azi_unit='degree', azi_direction='clockwise',
                        #   inc_reference='elevation', inc_unit='degree', mode='heading', is_lonlat=True
    mysar.read_from_tiff(downsample=1, apply_wavelength_conversion=True, zero2nan=True)

    # Remove Zero and NaN values and value where LOS equals 1
    mysar.checkZeros()
    mysar.checkNaNs()
    mysar.checkLosEqualsOne()

    # Plot Raw InSAR data
    mysar.plot_raw_sar(rawdownsample4plot=1, save_fig=True, file_path='raw_insar.png', dpi=300)