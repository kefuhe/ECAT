import numpy as np
import pandas as pd
# from myexplorefault_smc import MyExploreFault as explorefault
from eqtools.csiExtend.exploremultifaults_smc import explorefault
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
    verbose = False
    #------------------------------Set ExploreFault Object-------------------------------------#
    expfault = explorefault('invrc', config_file='default_config.yml', geodata=None, verbose=verbose)
    nchains = expfault.nchains
    chain_length = expfault.chain_length
    expfault.setPriors(bounds=None, initialSample=None, datas=None) # datas for Sar reference
    expfault.setLikelihood(datas=None, verticals=None) 
    # expfault.walk(nchains=nchains, chain_length=chain_length, comm=comm, filename='samples_mag_rake_multifaults.h5', 
    #               save_every=2, save_at_interval=False, covariance_epsilon = 1e-9, amh_a=1.0/9.0, amh_b=8.0/9.0)

    # ---------------------------------Plot Results---------------------------------------------#
    expfault.extract_and_plot_bayesian_results(rank=rank, filename='samples_mag_rake_multifaults.h5',
                                               plot_faults=True, plot_sigmas=True, plot_data=True)