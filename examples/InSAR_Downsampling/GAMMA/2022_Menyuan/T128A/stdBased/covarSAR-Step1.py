# External libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

# CSI routines and ECAT routines
import csi.insar as insar
import csi.imagecovariance as imcov
from eqtools.csiExtend.sarUtils.readGamma2csisar import GammasarReader

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    # set some flags
    input_check = True
    do_downsample = False
    output_check = False

    # CSI does most computations in a local Cartesian coordinate system. It presently supports only UTM projections, so we need to set the zone to use.
    lon0, lat0 = 101.31, 37.80
    # UTM zone 47 for 2022 Mw 6.6 Menyuan earthquake
    utmzone = 47

    # Prepare InSAR data for Sentinel-1
    outName = 'S1_T128A'
    prefix = r'geo_20220105_20220117'
    mysar = GammasarReader(name='Menyuan', lon0=lon0, lat0=lat0, directory_name='..')
    mysar.extract_raw_grd(prefix=prefix)
    mysar.read_from_gamma(downsample=1, apply_wavelength_conversion=True, zero2nan=True)

    # Remove Zero and NaN values and value where LOS equals 1
    mysar.checkZeros()
    mysar.checkNaNs()
    mysar.checkLosEqualsOne()

    # Covariance first estimate on original data
    covar = imcov('Covariance estimator',mysar, verbose=True)

    # mask out high deformation above earthquake rupture
    maskOut = [100.5, 101.75, 37.35, 38.1]
    covar.maskOut([maskOut])

    # We use the computeCovariance method to sample the dataset with a random set on 0.002 of the pixels, 
    # estimate and remove a ramp, calculate the semivariogram at distances of every 2 km out to 100 km, 
    # convert the semivariogram to covariance, and estimate a fit of an exponential function vs. distance.
    covar.computeCovariance(function='exp', frac=0.002, every=2.0, distmax=100., rampEst=True)
    covar.plot(data='all')
    covar.write2file(savedir='./')
