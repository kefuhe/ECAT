# External libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

# CSI routines and ECAT routines
import csi.insar as insar
import csi.imagecovariance as imcov
from eqtools.csiExtend.sarUtils.readTiff2csisar import TiffsarReader, Hyp3TiffReader

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
    losdispname = 'S1AA_20240707T100141_20240719T100140_VVR012_INT80_G_ueF_C070_los_disp.tif'
    phiname = 'S1AA_20240707T100141_20240719T100140_VVR012_INT80_G_ueF_C070_lv_phi.tif'
    thetaname = 'S1AA_20240707T100141_20240719T100140_VVR012_INT80_G_ueF_C070_lv_theta.tif'
    mysar = Hyp3TiffReader(name='Chile', lon0=lon0, lat0=lat0, directory_name=dirname)
    mysar.extract_raw_grd(phsname=losdispname, azifile=phiname, incfile=thetaname)
    #                       azi_reference='east', azi_unit='radian', azi_direction='counterclockwise',
    #                       inc_reference='elevation', inc_unit='radian', mode='left_los', is_lonlat=False
    mysar.read_from_tiff(downsample=1, apply_wavelength_conversion=False, zero2nan=True)

    # Remove Zero and NaN values and value where LOS equals 1
    mysar.checkZeros()
    mysar.checkNaNs()
    mysar.checkLosEqualsOne()

    # Plot Raw InSAR data
    mysar.plot_raw_sar(rawdownsample4plot=1, save_fig=True, file_path='raw_insar.png', dpi=300)