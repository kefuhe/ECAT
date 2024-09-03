# vim: set filetype=cfg:
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
    lon0 = 60.44
    lat0 = 35.78

    # ---------------------------------Generate GPS Object------------------------------------#
    verbose = False
    # 获取同震GPS数据
    #gpsfile_6_4 = os.path.join('..', '..', '..', 'GPS', 'GPS_ENU6_4NoEW_CSI.dat')
    #gpsfile_7_1 = os.path.join('..', '..', '..', 'GPS', 'GPS_ENU7_1NoEW_CSI.dat')

    #cogps6_4 = gps(name='co6_4', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    #cogps6_4.read_from_enu(gpsfile_6_4, factor=1., minerr=1., header=1, checkNaNs=True)
    #cogps6_4.buildCd(direction='enu')

    #cogps7_1 = gps(name='co7_1', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    #cogps7_1.read_from_enu(gpsfile_7_1, factor=1., minerr=1., header=1, checkNaNs=True)
    #cogps7_1.buildCd(direction='enu')

    # gpsdata = [cogps6_4, cogps7_1]
    # ----------------------------------Generate SAR Object-----------------------------------#

    basedir = '..'
    # 获取同震SAR数据
    varres_t034d = os.path.join(basedir, r'InSAR', 'downsample', 'S1_T020D_ifg')
    varres_t056a = os.path.join(basedir, r'InSAR', 'downsample', 'S1_T013A_ifg')

    coDscsar = insar('S1_T020D_ifg', lon0=lon0, lat0=lat0, verbose=verbose)
    coDscsar.read_from_varres(varres_t034d, cov=True)

    coAscsar = insar('S1_T013A_ifg', lon0=lon0, lat0=lat0, verbose=verbose)
    coAscsar.read_from_varres(varres_t056a, cov=True)

    insardata = [coAscsar, coDscsar]
    geodata = insardata 
    #------------------------------Set ExploreFault Object-------------------------------------#
    expfault = explorefault('invrc', lat0=lat0, lon0=lon0, config_file='default_config.yml', geodata=geodata, verbose=verbose)
    nchains = expfault.nchains
    chain_length = expfault.chain_length
    expfault.setPriors(bounds=None, initialSample=None, datas=None) # datas for Sar reference
    expfault.setLikelihood(datas=None, verticals=None) 
    expfault.walk(nchains=nchains, chain_length=chain_length, comm=comm, filename='samples_mag_rake_multifaults.h5', 
                   save_every=2, save_at_interval=False, covariance_epsilon = 1e-9, amh_a=1.0/9.0, amh_b=8.0/9.0)

    # ---------------------------------Plot Results---------------------------------------------#
    expfault.extract_and_plot_bayesian_results(rank=rank, filename='samples_mag_rake_multifaults.h5',
                                               plot_faults=True, plot_sigmas=True, plot_data=True)
