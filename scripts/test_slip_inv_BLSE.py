import pandas as pd
import numpy as np
from collections import OrderedDict
import argparse

from csi import RectangularPatches as RectFault
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault
)
from csi import gps, insar

# Bayesian inversion
# from eqtools.csiExtend.bayesian_multifaults_inversion import (
        # BayesianMultiFaultsInversionConfig,
        # BayesianMultiFaultsInversion,
        # MyMultiFaultsInversion,
        # NT1, NT2
        # )

import os
# using the C++ backend
os.environ['CUTDE_USE_BACKEND'] = 'cpp'

import h5py
from mpi4py import MPI
from collections import namedtuple
import matplotlib.pyplot as plt

"""
This script demonstrates how to perform a BLSE inversion for a multifault model using the CSI package.
The script reads GPS and InSAR data, sets up the BLSE inversion, and runs the inversion.
The results are then plotted using the explorefault class.
Steps:
1. Read GPS and InSAR data. [Where you need to modify]
2. run ecat-generate-config to generate the default configuration file. [Where you need to modify]
3. run ecat-generate-boundary to generate the default boundary configuration file. [Where you need to modify]
4. Set up the BoundLSEMultiFaultsInversion object with the data and configuration file.
5. Run the BLSE inversion using the run method.
6. Extract and plot the results using the extract_and_plot_bayesian_results method. [Where you need to modify]
"""


if __name__ == '__main__':
    # -----------------------------------Parse Arguments---------------------------------------------#
    parser = argparse.ArgumentParser(description='BLSE Inversion Script')
    
    # Main execution mode
    parser.add_argument('--mode', choices=['single', 'loop'], default='single',
                       help='Execution mode: single run or penalty weight loop (default: single)')
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
    # # Obtain GPS data
    # gpsfile_6_4 = os.path.join('..', 'GPS', 'GPS_ENU6_4NoEW_CSI.dat')

    # cogps6_4 = gps(name='co6_4', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    # cogps6_4.read_from_enu(gpsfile_6_4, factor=1., minerr=1., header=1, checkNaNs=True)
    # cogps6_4.buildCd(direction='enu')
    # ----------------------------------Generate SAR Object-----------------------------------#
    sar_t012a_file = os.path.join('..', 'InSAR', 'RawInSAR', 'Dingri_2020_T012A', 'stdBased', 'S1_T012A_ifg')
    sar_t121d_file = os.path.join('..', 'InSAR', 'RawInSAR', 'Dingri_2020_T121D', 'stdBased', 'S1_T121D_ifg')

    sar_t012a = insar(name='T012A', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t012a.read_from_varres(sar_t012a_file, triangular=False, cov=True)

    sar_t121d = insar(name='T121D', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t121d.read_from_varres(sar_t121d_file, triangular=False, cov=True)

    gpsdata=[]
    insardata = [sar_t012a, sar_t121d]
    geodata = gpsdata + insardata

    # ----------------------------------Generate Fault Object----------------------------------#
    fault_em1 = TriFault(name='Dingri_2020', lon0=lon0, lat0=lat0, verbose=verbose)
    fault_em1.top = 0.0
    fault_em1.depth = 8.0
    fault_em1.generate_top_bottom_from_nonlinear_soln(clon=87.39976, clat=28.66787, cdepth=1.7692, strike=332.2241, dip=52.0271, length=12)
    fault_em1.generate_mesh(top_size=1.0, bottom_size=1.5, show=False, verbose=0)
    fault_em1.initializeslip(values='depth')
    fault_em1.find_fault_fouredge_vertices()
    top_coords = fault_em1.edge_vertices['top']
    fault_em1.trace(top_coords[:, 0], top_coords[:, 1], utm=True)
    # fault_em1.plot()

    trifaults = OrderedDict()
    trifaults['Dingri_2020'] = fault_em1
    trifaults_list = [trifaults[faultname] for faultname in trifaults]

    # Remove some pixels close to the faults
    # for sardata in insardata:
    #     sardata.reject_pixels_fault(1.0, trifaults_list)

    # --------------------------Build MultiFaults BLSE Inversion-------------------#
    # Way 1
    # from eqtools.csiExtend.bayesian_config import BoundLSEInversionConfig as BoundLSEConfig
    # inversion_blse = BoundLSEConfig(config_file='default_config_BLSE.yml', faults_list=trifaults_list, 
    #                                          geodata=geodata, verbose=True)
    # from eqtools.csiExtend.multifaultsolve_boundLSE import multifaultsolve_boundLSE as multifaultsolve
    # inversion = multifaultsolve('inv', trifaults_list, verbose=verbose)
    # inversion.assembleGFs()
    # inversion.set_bounds_from_config('bounds_config.yml')
    # inversion.set_inequality_constraints_for_rake_angle(rake_limits=inversion.bounds_config['rake_angle'])
    # inversion.ConstrainedLeastSquareSoln(penalty_weight=8.0, smoothing_constraints=('free', 'free', 'free', 'free'), verbose=True)
    # inversion.distributem()

    # Way 2
    from eqtools.csiExtend.blse_multifaults_inversion import BoundLSEMultiFaultsInversion
    inversion = BoundLSEMultiFaultsInversion('inv', trifaults_list, geodata, verbose=True,
                                             config='default_config_BLSE_CovDiag.yml', bounds_config='bounds_config.yml')
    if args.mode == 'single':
        inversion.run(penalty_weight=None, alpha=[np.log10(1/100.0)])
        inversion.returnModel(print_stat=False)

    elif args.mode == 'loop':
        # ---------------------------------Loop Penalty Weight---------------------------------------------#
        penalty_weight = [1.0, 5.0, 10.0, 30.0, 50.0, 80.0, 100.0, 125.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0]
        # for ipenalty in penalty_weight:
        #     alpha = [np.log10(1.0/ipenalty)]
        #     print(f'penalty_weight: {ipenalty:.1f},', end=' ')
        #     inversion.run(penalty_weight=None, alpha=alpha)
        inversion.simple_run_loop(penalty_weight, preferred_penalty_weight=10.0, output_file='run_loop_covdiag.dat', verbose=True)

    if args.mode == 'single':
        # ---------------------------------Plot Data and Slip Distribution---------------------------------------------#
        inversion.extract_and_plot_blse_results(rank=rank, plot_faults=True, plot_data=True, 
                                                gps_figsize=(3.5, 2.7), gps_scale=0.05, gps_legendscale=0.2, file_type='pdf',
                                                axis_shape=(1.0, 1.0, 0.25), elevation=56, azimuth=-70, gps_title=False,
                                                depth_range=25, z_ticks=[-20, -10, 0], remove_direction_labels=True,
                                                fault_cbaxis=[0.45, 0.32, 0.15, 0.02])
        # ---------------------------------Plot Results---------------------------------------------#
        # if rank == 0:
        #     import cmcrameri
        #     from eqtools.getcpt import get_cpt 
        #     slip_cmap = None
        #     if slip_cmap is not None and slip_cmap.endswith('.cpt'):
        #         cmap_slip = get_cpt.get_cmap(slip_cmap, method='list', N=15)
        #     else:
        #         cmap_slip = slip_cmap
        #     if slip_cmap is None: # precip3_16lev_change.cpt
        #         cmap_slip = get_cpt.get_cmap('precip3_16lev_change.cpt', method='list', N=8) #, method='list', N=15 cmap_slip
        #     inversion.plot_multifaults_slip(slip='total', cmap='cmc.roma_r', # norm=[0, 7.6],
        #                                 drawCoastlines=False, cblabel='Slip (m)',
        #                                 savefig=True, style=['notebook'], cbaxis=[0.15, 0.22, 0.15, 0.02],
        #                                 xtickpad=5, ytickpad=10, ztickpad=5,
        #                                 xlabelpad=15, ylabelpad=25, zlabelpad=15,
        #                                 shape=(1.0, 2.0, 0.8), elevation=54, azimuth=24,
        #                                 depth=18, zticks=[-12, -6, 0], fault_expand=0.0,
        #                                 plot_faultEdges=False, suffix='mean')

        # ---------------------------------Write Results to File---------------------------------------------#
        if rank == 0:
            for i, trifault in enumerate(trifaults_list):
                four = trifault.writeFourEdges2File(dirname=r'output/stat_infos')
                trifault.writePatches2File(f'output/slip_{trifault.name}.gmt', add_slip='total')
                trifault.writeSlipDirection2File(filename=f'output/slipdir_{trifault.name}.txt', 
                                                scale='total', factor=0.4) # reference_strike=None, threshold=0.0

        # ---------------------------------Write Data to File---------------------------------------------#
        if rank == 0:
            verticals = inversion.config.geodata['verticals']
            geodata = inversion.config.geodata['data']
            # Recalculate the synthetic data
            for idata, ivert in zip(geodata, verticals):
                idata.buildsynth(trifaults_list, direction='sd', poly='include', vertical=ivert)
            # Write the InSAR data to file
            if not os.path.exists('Modeling'):
                os.mkdir('Modeling')
            for i, sardata in enumerate(insardata):
                if sardata.dtype == 'opticorr':
                    for itype in ['data', 'synth', 'res']:
                        for idir in ['east', 'north']:
                            itypedir = f'{itype}{idir}'
                            sardata.writeDecim2file(f'{sardata.name}_{itypedir}.txt', itypedir, outDir='Modeling', triangular=True)
                else:
                    for itype in ['data', 'synth', 'resid']:
                        sardata.writeDecim2file(f'{sardata.name}_{itype}.txt', itype, outDir='Modeling', triangular=True)
            
            # Write the GPS data to file
            for i, gpsdata in enumerate(gpsdata):
                for itype in ['data', 'synth', 'resid']:
                    gpsdata.write2file(f'{gpsdata.name}_{itype}.txt', itype, outDir='Modeling')