import numpy as np
import pandas as pd
from eqtools.csiExtend.exploremultifaults_smc import explorefault
from csi.gps import gps
from csi.insar import insar
from mpi4py import MPI
import matplotlib.pyplot as plt

import os
import argparse
# using the C++ backend
# os.environ['CUTDE_USE_BACKEND'] = 'cpp'

"""
This script demonstrates how to perform a Bayesian inversion for a multifault model using the CSI package.
The script reads GPS and InSAR data, sets up the Bayesian inversion, and runs the inversion.
The results are then plotted using the explorefault class.
Steps:
1. Read GPS and InSAR data. [Where you need to modify]
2. run ecat-generate-nonlinear to generate the default configuration file. [Where you need to modify]
3. run ecat-generate-boundary to generate the default boundary configuration file. [Where you need to modify]
4. Set up the explorefault object with the data and configuration file.
5. Run the Bayesian inversion using the walk method.
6. Extract and plot the results using the extract_and_plot_bayesian_results method. [Where you need to modify]

usage: test_nonlinear_bayesian.py [-h] [-r] [--no-plot]

Perform Bayesian inversion or plot results. Use mpiexec for parallel execution when running the inversion.

optional arguments:
  -h, --help     Show this help message and exit.
  -r, --run      Run Bayesian inversion (requires mpiexec for parallel execution).
  --no-plot      Disable plotting results after running the script.

Run Bayesian inversion or plot results.
1. Run Bayesian inversion. You can change 4 to the number of processes you want to use.
        mpiexec -n 4 python test_nonlinear_bayesian.py -r
2. Only plot results after running the script.
    python test_nonlinear_bayesian.py
"""


def remove_orbit_error(sar, order=1, exclude_range=None):
    """
    Remove orbital error from velocity data.
    
    Parameters:
    sar (object): The SAR object containing velocity data and coordinates.
    order (int): The order of the polynomial to fit (1 for linear, 2 for quadratic).
    exclude_range (tuple): Optional. A tuple of (lon_min, lon_max, lat_min, lat_max) to exclude from fitting.
    
    Returns:
    np.ndarray: The velocity data with orbital error removed.
    """
    vel = sar.vel
    x = sar.x
    y = sar.y
    
    if exclude_range:
        lon_min, lon_max, lat_min, lat_max = exclude_range
        xmin, ymin = sar.ll2xy(lon_min, lat_min)
        xmax, ymax = sar.ll2xy(lon_max, lat_max)
        mask = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)
        x_fit = x[mask]
        y_fit = y[mask]
        vel_fit = vel[mask]
    else:
        x_fit = x
        y_fit = y
        vel_fit = vel
    
    # Fit a polynomial surface
    if order == 1:
        # Linear fit
        A = np.c_[x_fit, y_fit, np.ones(x_fit.shape)]
    elif order == 2:
        # Quadratic fit
        A = np.c_[x_fit**2, y_fit**2, x_fit*y_fit, x_fit, y_fit, np.ones(x_fit.shape)]
    else:
        raise ValueError("Order must be 1 (linear) or 2 (quadratic)")
    
    # Solve for the coefficients
    coeff, _, _, _ = np.linalg.lstsq(A, vel_fit, rcond=None)
    
    # Create the fitted surface
    if order == 1:
        fitted_surface = coeff[0]*x + coeff[1]*y + coeff[2]
    elif order == 2:
        fitted_surface = coeff[0]*x**2 + coeff[1]*y**2 + coeff[2]*x*y + coeff[3]*x + coeff[4]*y + coeff[5]
    
    # Subtract the fitted surface from the original velocity data
    vel_corrected = vel - fitted_surface
    
    return vel_corrected


if __name__ == '__main__':
    # -----------------------------------Parse Arguments--------------------------------------#
    parser = argparse.ArgumentParser(
        description="Perform Bayesian inversion or plot results. Use mpiexec for parallel execution when running the inversion."
    )
    parser.add_argument(
        '-r', '--run', action='store_true',
        help="Run Bayesian inversion (requires mpiexec for parallel execution)."
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help="Disable plotting results after running the script."
    )
    args = parser.parse_args()

    # -----------------------------------MPI Init---------------------------------------------#
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # -----------------------------------Read Data---------------------------------------------#
    lon0 = 87.5
    lat0 = 28.5

    # ---------------------------------Generate GPS Object------------------------------------#
    verbose = False
    # # 获取同震GPS数据
    # gpsfile_6_4 = os.path.join('..', 'GPS', 'GPS_ENU6_4NoEW_CSI.dat')
    # gpsfile_7_1 = os.path.join('..', 'GPS', 'GPS_ENU7_1NoEW_CSI.dat')

    # cogps6_4 = gps(name='co6_4', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    # cogps6_4.read_from_enu(gpsfile_6_4, factor=1., minerr=1., header=1, checkNaNs=True)
    # cogps6_4.buildCd(direction='enu')

    # cogps7_1 = gps(name='co7_1', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    # cogps7_1.read_from_enu(gpsfile_7_1, factor=1., minerr=1., header=1, checkNaNs=True)
    # cogps7_1.buildCd(direction='enu')
    # ----------------------------------Generate SAR Object-----------------------------------#

    # Extract SAR data
    sar_t012a_file = os.path.join('..', 'InSAR', 'RawInSAR', 'Dingri_2020_T012A', 'stdBased', 'S1_T012A_ifg')
    sar_t121d_file = os.path.join('..', 'InSAR', 'RawInSAR', 'Dingri_2020_T121D', 'stdBased', 'S1_T121D_ifg')

    sar_t012a = insar(name='T012A', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t012a.read_from_varres(sar_t012a_file, triangular=False, cov=True)
    # Optional: build the diagonal covariance matrix
    # sar_t012a.buildDiagCd()

    sar_t121d = insar(name='T121D', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t121d.read_from_varres(sar_t121d_file, triangular=False, cov=True)

    gpsdata = []
    sardata = [sar_t012a, sar_t121d]
    geodata = gpsdata + sardata
    #------------------------------Set ExploreFault Object-------------------------------------#
    expfault = explorefault('invrc', lat0=lat0, lon0=lon0, config_file='default_config.yml', geodata=geodata, verbose=verbose)
    nchains = expfault.nchains
    chain_length = expfault.chain_length
    expfault.setPriors(bounds=None, initialSample=None, datas=None) # datas for Sar reference
    expfault.setLikelihood(datas=None, verticals=None) 
    # -----------------------------------Run Bayesian Inversion--------------------------------#
    if args.run:
        # Run Bayesian inversion
        expfault.walk(
            nchains=nchains, chain_length=chain_length, comm=comm,
            filename='samples_mag_rake_multifaults.h5', save_every=2,
            save_at_interval=False, covariance_epsilon=1e-9, amh_a=1.0/9.0, amh_b=8.0/9.0
        )

    # -----------------------------------Plot Results------------------------------------------#
    if not args.no_plot:
        expfault.extract_and_plot_bayesian_results(
            rank=rank, filename='samples_mag_rake_multifaults.h5',
            fault_figsize=None, sigmas_figsize=None, plot_faults=False,
            plot_sigmas=True, plot_data=False, save_data=True, sar_corner='quad'
        )

        if rank == 0:
            # Plot the KDE matrix
            axis_labels = [r'$log(\sigma_{T012A}^2)$', r'$log(\sigma_{T121D}^2)$']
            expfault.plot_kde_matrix(
                plot_sigmas=True, plot_faults=False, fill=True, save=True,
                scatter=False, filename='kde_matrix_sigmas.png', axis_labels=axis_labels,
                figsize=(5.0, 5.0), hspace=0.1, wspace=0.1
            )
            axis_labels = ['Lon', 'Lat', 'Depth', 'Dip', 'Width', 'Length', 'Strike', 'Slip', 'Rake']
            expfault.plot_kde_matrix(
                plot_sigmas=False, plot_faults=True, fill=True, save=True,
                scatter=False, filename='kde_matrix_faults.png', axis_labels=axis_labels,
                hspace=0.1, wspace=0.2, center_lon_lat=False, figsize=(7.5, 6.5), xtick_rotation=45
            )