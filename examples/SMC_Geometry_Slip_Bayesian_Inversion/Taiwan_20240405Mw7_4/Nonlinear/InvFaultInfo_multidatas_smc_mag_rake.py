import numpy as np
import pandas as pd
from csi.explorefault_smc import explorefault
from csi.gps import gps
from csi.insar import insar
from mpi4py import MPI
import matplotlib.pyplot as plt

import os
# using the C++ backend
# os.environ['CUTDE_USE_BACKEND'] = 'cpp'


if __name__ == '__main__':
    # -----------------------------------MPI Init---------------------------------------------#
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # -----------------------------------Read Data---------------------------------------------#
    # Read the data
    lon0 = 122
    lat0 = 24

    # 
    verbose = False
    vertical = False

    # Read GNSS Data
    gpsfile = os.path.join('..', r'GPS', 'us7000m9g4_forweb_NGL.txt')
    mygps = gps('gps', lon0=lon0, lat0=lat0, verbose=verbose)
    mygps.read_from_enu(gpsfile, header=2, factor=1)
    mygps.vel_enu[:, 2] = np.nan
    mygps.buildCd(direction='en')
    err_mean = mygps.err_enu[:, :2].mean()

    #------------------------------Set ExploreFault Object-------------------------------------#
    expfault = explorefault('invrc', mode='mag_rake', lat0=lat0, lon0=lon0, verbose=verbose)
    expfault.theta = []
    bounds = {
        'lon': ['Uniform', 119, 5.0],
        'lat': ['Uniform', 21, 5.0],
        'depth': ['Uniform', 0, 50],
        'dip': ['Uniform', 45, 44.9],
        'width': ['Uniform', 5.1, 39.9],
        'length': ['Uniform', 5.1, 74.9],
        'strike': ['Uniform', 0, 360],
        'magnitude': ['Uniform', 0, 10],
        'rake': ['Uniform', 60, 60] # -180,180 or 0, 360
    }

    initial = {
        'lon': 121,
        'lat': 24,
        'depth': 5,
        'dip': 60,
        'width': 10,
        'length': 25,
        'strike': 125,
        'magnitude': 3,
        'rake': 0}
    expfault.setPriors(bounds=bounds, initialSample=initial)
    expfault.setLikelihood([mygps], vertical=vertical) 
    # expfault.walk(nchains=500, chain_length=200, comm=comm, filename='samples_mag_rake.h5', save_every=2, save_at_interval=False)
    expfault.load_samples_from_h5(filename='samples_mag_rake.h5')

    # expfault.plot_smc_statistics()
    if rank == 0:
        expfault.plot_kde_matrix(save=True, fill=True, scatter=False)
        fault = expfault.returnModel()
        for data in [mygps]:
            # Build the green's functions
            fault.buildGFs(data, slipdir='sd', verbose=False, vertical=vertical)
            # Build the synthetics
            data.buildsynth(fault)
        mygps.plot(drawCoastlines=True, data=['data', 'synth'], scale=0.2, legendscale=0.05, color=['k', 'r'])

