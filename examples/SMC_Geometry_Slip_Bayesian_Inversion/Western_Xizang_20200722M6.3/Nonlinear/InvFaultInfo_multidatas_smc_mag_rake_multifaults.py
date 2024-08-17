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
    lon0 = 86.83
    lat0 = 33.16

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
    # ----------------------------------Generate SAR Object-----------------------------------#

    basedir = '..'
    # 获取同震SAR数据
    varres_t034d = os.path.join(basedir, r'InSAR', 'downsample', 'S1_T121D_ifg')
    varres_t056a = os.path.join('..', r'InSAR', 'downsample', 'S1_T012A_ifg')

    coDscsar = insar('S1_T121D_ifg', lon0=lon0, lat0=lat0, verbose=verbose)
    coDscsar.read_from_varres(varres_t034d, cov=True)
    #coDscsar.err *= 1
    #coDscsar.buildDiagCd()

    coAscsar = insar('S1_T012A_ifg', lon0=lon0, lat0=lat0, verbose=verbose)
    coAscsar.read_from_varres(varres_t056a, cov=True)
    #coAscsar.err *= 1
    #coAscsar.buildDiagCd()

    #------------------------------Set ExploreFault Object-------------------------------------#
    expfault = explorefault('invrc', mode='mag_rake', num_faults=1, lat0=lat0, lon0=lon0, verbose=verbose,
                            config_file='default_config.yml')

    expfault.setPriors(bounds=None, initialSample=None)
    expfault.setLikelihood([coAscsar, coDscsar], vertical=True) # coAscsar, coDscsar,  cogps7_1
    expfault.walk(nchains=100, chain_length=50, comm=comm, filename='samples_mag_rake_multifaults.h5', 
                   save_every=2, save_at_interval=False, covariance_epsilon = 1e-9, amh_a=1.0/9.0, amh_b=8.0/9.0)

    # ---------------------------------Plot Results---------------------------------------------#
    if rank == 0:
        expfault.load_samples_from_h5(filename='samples_mag_rake_multifaults.h5')
        expfault.print_mcmc_parameter_positions()
        expfault.plot_kde_matrix(save=True, plot_faults=True, faults='fault_0', fill=True, 
                                  scatter=False, filename='kde_matrix_F1.png')
        # expfault.plot_kde_matrix(save=True, plot_faults=True, faults='fault_1', fill=True, 
        #                          scatter=False, filename='kde_matrix_F2.png')
        expfault.plot_kde_matrix(save=True, plot_faults=False, plot_sigmas=True, fill=True, 
                                  scatter=False, filename='kde_matrix_sigmas.png')
        faults = expfault.returnModels()
        
        # Plot GPS data
        for fault in faults:
            fault.color = 'b'
        #for cogps in [cogps7_1]:
            #cogps.buildsynth(faults, vertical=True)
            #cogps.plot(faults=faults, drawCoastlines=True, data=['data', 'synth'], scale=0.2, legendscale=0.05, color=['k', 'r'],
                    #seacolor='lightblue', box=[-119.5, -116, 34.3, 37], titleyoffset=1.02)
            #cogps.fig.savefig(f'gps_{cogps.name}', ftype='png', dpi=600, 
                              #bbox_inches='tight', mapaxis=None, saveFig=['map'])
        #for cogps in [cogps6_4]:
            #cogps.buildsynth(faults, vertical=True)
            #cogps.plot(faults=faults, drawCoastlines=True, data=['data', 'synth'], scale=0.2, legendscale=0.05, color=['k', 'r'],
                    #seacolor='lightblue', titleyoffset=1.02)
            # cogps.fig.figCarte.savefig(f'gps_{cogps.name}_map.png', dpi=600, bbox_inches='tight')
            #cogps.fig.savefig(f'gps_{cogps.name}', ftype='png', dpi=600, 
                             #bbox_inches='tight', mapaxis=None, saveFig=['map'])
        # Plot SAR data
        for fault in faults:
            fault.color = 'k'
        for cosar in [coAscsar, coDscsar]:
            cosar.buildsynth(faults, vertical=True)
            for data in ['data', 'synth', 'res']:
                cosar.plot(faults=faults, data=data, seacolor='lightblue', figsize=(3.5, 2.7),
                           cbaxis=[0.15, 0.25, 0.25, 0.02], drawCoastlines=True, titleyoffset=1.02) # faults=faults
                cosar.fig.savefig(f'sar_{cosar.name}_{data}', ftype='png', dpi=600, saveFig=['map'], 
                                  bbox_inches='tight', mapaxis=None)
                # cosar.fig.figCarte.savefig(f'sar_{cosar.name}_{data}_map.png', dpi=600, bbox_inches='tight')
        print(expfault.model_dict)

