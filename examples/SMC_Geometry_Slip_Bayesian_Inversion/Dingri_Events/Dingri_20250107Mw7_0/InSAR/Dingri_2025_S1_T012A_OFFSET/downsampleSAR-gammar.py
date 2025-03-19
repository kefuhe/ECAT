# External libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import argparse
from config import config
matplotlib.rcParams['figure.figsize'] = (30,30) # make the plots bigger

# CSI routines and ECAT routines
import csi.insar as insar
import csi.geodeticplot as geoplt
import csi.imagedownsampling as imagedownsampling
import csi.imagecovariance as imcov
from csi.imagedownsamplingTriangular import imagedownsamplingTriangular as imagedownsampling
# from csi import faultwithdip
from csi.csiutils import utm_zone_epsg
from eqtools.csiExtend.sarUtils.readGamma2csisar import GammasarReader
from eqtools.csiExtend.AdaptiveTriangularPatches import AdaptiveTriangularPatches as TriangularPatches


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process SAR data for downsampling.')
    parser.add_argument('--do_covar', '-c', action='store_true', help='Perform covariance estimation')
    parser.add_argument('--do_downsample', '-d', action='store_true', help='Perform downsampling')
    parser.add_argument('--show_raw_sar', '-s', action='store_true', help='Show raw SAR plot')
    args = parser.parse_args()

    # set some flags
    do_covar = args.do_covar if args.do_covar else config["do_covar"]
    do_downsample = args.do_downsample if args.do_downsample else config["do_downsample"]
    show_raw_sar = args.show_raw_sar if args.show_raw_sar else config["show_raw_sar"]
    output_check = config["output_check"]

    if show_raw_sar:
        do_covar = False
        do_downsample = False
    elif do_covar or do_downsample:
        show_raw_sar = False

    print(f'do_covar: {do_covar}, do_downsample: {do_downsample}, show_raw_sar: {show_raw_sar}')

    # UTM zone 45 for Dingri earthquake
    lon0, lat0 = config["lon0"], config["lat0"]
    utmzone, epsg = utm_zone_epsg(lon0, lat0)
    print(f'UTM zone: {utmzone}, EPSG: {epsg}')

    # set the output name root
    outName = config["outName"]
    prefix = config["prefix"]
    mysar = GammasarReader(name='Dingri', lon0=lon0, lat0=lat0, directory_name='..')

    if config["use_offset_sar"]:
        sar_dict = config["sar_dict"]
        mysar.extract_raw_grd(**sar_dict)
    else:
        mysar.extract_raw_grd(prefix=prefix)

    mysar.read_from_gamma(downsample=1, apply_wavelength_conversion=True, zero2nan=True)

    # Remove Zero and NaN values and value where LOS equals 1
    mysar.checkZeros()
    mysar.checkNaNs()
    mysar.checkLosEqualsOne()

    if config["use_offset_sar"]:
        mysar.remove_significant_outliers(4.0)

    # Covariance object
    covar = imcov('Covariance estimator', mysar, verbose=True)

    # see covarSAR notebook for details
    if do_covar:
        # Covariance first estimate on original data
        covar = imcov('Covariance estimator', mysar, verbose=True)
        # mask out high deformation above earthquake rupture
        maskOut = config["maskOut"]
        covar.maskOut([maskOut])
        # run the semi-variogram covariance calculation
        # We use the computeCovariance method to sample the dataset with a random set on 0.002 of the pixels, 
        # estimate and remove a ramp, calculate the semivariogram at distances of every 2 km out to 100 km, 
        # convert the semivariogram to covariance, and estimate a fit of an exponential function vs. distance.
        covar.computeCovariance(function=config["covar"]["function"], frac=config["covar"]["frac"], 
                                every=config["covar"]["every"], distmax=config["covar"]["distmax"], 
                                rampEst=config["covar"]["rampEst"])
        # plot results
        covar.plot(data='all')
        # write estimated function to file
        covar.write2file(savedir='./')

    else:  # read in previous calculation
        covar.read_from_covfile('Covariance estimator', 'Covariance_estimator.cov')
        print('read previously calculated covariance estimates')

    # Construct fault patches
    selected_faults = []
    for fault_config in config["faults"]:
        trace_file = fault_config["trace_file"]
        trace = pd.read_csv(trace_file, names=['lon', 'lat'], sep=r'\s+', comment='#')
        trifault = TriangularPatches('Triangular Fault', lon0=lon0, lat0=lat0, verbose=True)
        trifault.trace(trace.lon.values, trace.lat.values)
        trifault.top = fault_config.get("top_depth", config["default_top_depth"])
        trifault.depth = fault_config.get("bottom_depth", config["default_bottom_depth"])
        trifault.set_top_coords_from_trace()
        trifault.generate_bottom_from_single_dip(dip_angle=fault_config["dip_angle"], dip_direction=fault_config["dip_direction"])
        trifault.generate_mesh(top_size=fault_config["top_size"], bottom_size=fault_config["bottom_size"], verbose=0, show=False)
        trifault.initializeslip(values='depth')
        # trifault.plot()
        selected_faults.append(trifault)

    if show_raw_sar:
        plot_raw_sar_config = config["plot_raw_sar"]
        mysar.plot_raw_sar(save_fig=plot_raw_sar_config["save_fig"], faults=selected_faults, 
                           rawdownsample4plot=plot_raw_sar_config["rawdownsample4plot"], 
                        colorbar_x=plot_raw_sar_config["colorbar_x"], colorbar_y=plot_raw_sar_config["colorbar_y"], 
                        colorbar_length=plot_raw_sar_config["colorbar_length"], 
                        vmin=plot_raw_sar_config["vmin"], vmax=plot_raw_sar_config["vmax"])

    minlat, maxlat, minlon, maxlon = np.min(mysar.lat), np.max(mysar.lat), np.min(mysar.lon), np.max(mysar.lon)
    minlon = minlon if config["downsample_box"]["minlon"] is None else config["downsample_box"]["minlon"]
    maxlon = maxlon if config["downsample_box"]["maxlon"] is None else config["downsample_box"]["maxlon"]
    minlat = minlat if config["downsample_box"]["minlat"] is None else config["downsample_box"]["minlat"]
    maxlat = maxlat if config["downsample_box"]["maxlat"] is None else config["downsample_box"]["maxlat"]
    plot_box = config["plot_box"]
    plot_box['plotMinLat'] = minlat if plot_box["plotMinLat"] is None else plot_box["plotMinLat"]
    plot_box['plotMaxLat'] = maxlat if plot_box["plotMaxLat"] is None else plot_box["plotMaxLat"]
    plot_box['plotMinLon'] = minlon if plot_box["plotMinLon"] is None else plot_box["plotMinLon"]
    plot_box['plotMaxLon'] = maxlon if plot_box["plotMaxLon"] is None else plot_box["plotMaxLon"]

    mysar.select_pixels(minlon, maxlon, minlat, maxlat)
    print('cropped image area longitude {} to {}'.format(mysar.lon.min(), mysar.lon.max()))

    if (not show_raw_sar) and do_downsample:
        downsampler = imagedownsampling('Downsampler', mysar, faults=selected_faults)
        downsampler.initialstate(minimumsize=config['downsample']['minimumsize'], tolerance=config['downsample']['tolerance'], 
                                 plot=False, decimorig=10)
        downsampler.resolutionBased(max_samples=config['downsample']['max_samples'], 
                                    change_threshold=config['downsample']['change_threshold'], 
                                    smooth_factor=config['downsample']['smooth_factor'], 
                                    slipdirection=config['downsample']['slipdirection'], 
                                    plot=False, 
                                    verboseLevel='maximum', 
                                    decimorig=10, 
                                    vertical=False)
        downsampler.writeDownsampled2File(prefix=outName+'_ifg', rsp=True)
    
    if (not show_raw_sar) and config["plot_downsample"]:
        sardecim = insar('Decimated DataSet', lon0=lon0, lat0=lat0)
        sardecim.read_from_varres(outName+'_ifg', factor=1.0, step=0.0, triangular=True)
        sardecim.checkLosEqualsOne()
        sardecim.checkLOS()
        sardecim.Cd = covar.buildCovarianceMatrix(sardecim, 'Covariance estimator', write2file=outName+'_ifg.cov')

        plot_faults = selected_faults
        if output_check:
            plot_box = config["plot_box"]
            gp = geoplt(figure=1, latmin=plot_box["plotMinLat"], lonmin=plot_box["plotMinLon"], latmax=plot_box["plotMaxLat"], lonmax=plot_box["plotMaxLon"])
            gp.drawCoastlines(parallels=0.2, meridians=0.2, drawOnFault=True, resolution='10m')
            for f in plot_faults:
                gp.faulttrace(f)
            gp.insar(mysar, decim=10, plotType='scatter')
            gp.titlemap('{} original'.format(outName))
            gp.savefig(prefix='raw_', ftype='jpg', dpi=300, saveFig=['map'])

            gp2 = geoplt(figure=3, latmin=plot_box["plotMinLat"], lonmin=plot_box["plotMinLon"], latmax=plot_box["plotMaxLat"], lonmax=plot_box["plotMaxLon"])
            gp2.drawCoastlines(parallels=0.2, meridians=0.2, drawOnFault=True, resolution='10m')
            for f in plot_faults:
                gp2.faulttrace(f)
            gp2.insar(sardecim, data='data', plotType='decimate')
            gp2.titlemap('{} decimated'.format(outName))
            gp2.savefig(prefix='downsample_', ftype='jpg', dpi=300, saveFig=['map'])