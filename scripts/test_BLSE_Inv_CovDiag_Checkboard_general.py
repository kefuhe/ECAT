import pandas as pd
import numpy as np
from collections import OrderedDict

from csi import RectangularPatches as RectFault
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault
)
from csi import gps, insar
from csi.opticorr import opticorr

# Bayesian inversion
from eqtools.csiExtend.bayesian_multifaults_inversion import (
        BayesianMultiFaultsInversionConfig,
        BayesianMultiFaultsInversion,
        MyMultiFaultsInversion,
        NT1, NT2
        )

import os
# using the C++ backend
os.environ['CUTDE_USE_BACKEND'] = 'cpp'

import h5py
from mpi4py import MPI
from collections import namedtuple
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # -----------------------------------MPI Init---------------------------------------------#
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # -----------------------------------Read Data---------------------------------------------#
    lon0 = 96.0
    lat0 = 21.5

    # ---------------------------------Generate GPS Object------------------------------------#
    verbose = False
    # # 获取同震GPS数据
    # gpsfile_6_4 = os.path.join('..', 'GPS', 'GPS_ENU6_4NoEW_CSI.dat')

    # cogps6_4 = gps(name='co6_4', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    # cogps6_4.read_from_enu(gpsfile_6_4, factor=1., minerr=1., header=1, checkNaNs=True)
    # cogps6_4.buildCd(direction='enu')
    # ----------------------------------Generate SAR Object-----------------------------------#

    # 获取同震SAR数据 LOS
    sar_t033d_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S1_T033D', 'resolutionBased', 'S1_T033D_ifg')
    sar_t106d_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S1_T106D', 'resolutionBased', 'S1_T106D_ifg')
    sar_t070a_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S1_T070A', 'resolutionBased', 'S1_T070A_ifg')
    sar_t143a_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S1_T143A', 'resolutionBased', 'S1_T143A_ifg')

    # AziOff
    sar_t033d_azi_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S1_T033D_OFFSET', 'resolutionBased_Azi', 'S1_T033D_AziOff_ifg')
    sar_t106d_azi_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S1_T106D_OFFSET', 'resolutionBased_Azi', 'S1_T106D_AziOff_ifg')
    sar_t070a_azi_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S1_T070A_OFFSET', 'resolutionBased_Azi', 'S1_T070A_AziOff_ifg')
    sar_t143a_azi_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S1_T143A_OFFSET', 'resolutionBased_Azi', 'S1_T143A_AziOff_ifg')

    # RngOff
    sar_t033d_rng_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S1_T033D_OFFSET', 'resolutionBased_Rng', 'S1_T033D_RngOff_ifg')
    sar_t106d_rng_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S1_T106D_OFFSET', 'resolutionBased_Rng', 'S1_T106D_RngOff_ifg')
    sar_t070a_rng_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S1_T070A_OFFSET', 'resolutionBased_Rng', 'S1_T070A_RngOff_ifg')
    sar_t143a_rng_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S1_T143A_OFFSET', 'resolutionBased_Rng', 'S1_T143A_RngOff_ifg')

    # Optical data
    opt_p1_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S2_Part1', 'resolutionBased', 'Optical_S2_part1_ifg')
    opt_p2_file = os.path.join('..', 'InSAR', 'Downsampled', 'Sagaing_S2_Part2', 'resolutionBased', 'Optical_S2_part2_ifg')


    sar_t033d = insar(name='T033D', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t033d.read_from_varres(sar_t033d_file, triangular=True)
    sar_t033d.buildDiagCd()
    # sar_t033d.buildCd(sigma=0.0020271867958938984, lam=11.257362012304698)

    sar_t106d = insar(name='T106D', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t106d.read_from_varres(sar_t106d_file, triangular=True)
    sar_t106d.buildDiagCd()
    # sar_t106d.buildCd(sigma=0.1678878393615418, lam=12.314091138409898)

    sar_t070a = insar(name='T070A', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t070a.read_from_varres(sar_t070a_file, triangular=True)
    sar_t070a.buildDiagCd()
    # sar_t070a.buildCd(sigma=0.0106758121257053, lam=29.944900558321407)

    sar_t143a = insar(name='T143A', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t143a.read_from_varres(sar_t143a_file, triangular=True)
    sar_t143a.buildDiagCd()
    # sar_t143a.buildCd(sigma=0.004627301835096236, lam=21.383990889288842)

    # AziOff SAR
    sar_t033d_azi = insar(name='T033D_AziOff', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t033d_azi.read_from_varres(sar_t033d_azi_file, triangular=True)
    sar_t033d_azi.buildDiagCd()

    sar_t106d_azi = insar(name='T106D_AziOff', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t106d_azi.read_from_varres(sar_t106d_azi_file, triangular=True)
    sar_t106d_azi.buildDiagCd()

    sar_t070a_azi = insar(name='T070A_AziOff', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t070a_azi.read_from_varres(sar_t070a_azi_file, triangular=True)
    sar_t070a_azi.buildDiagCd()

    sar_t143a_azi = insar(name='T143A_AziOff', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t143a_azi.read_from_varres(sar_t143a_azi_file, triangular=True)
    sar_t143a_azi.buildDiagCd()

    # RngOff SAR
    sar_t033d_rng = insar(name='T033D_RngOff', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t033d_rng.read_from_varres(sar_t033d_rng_file, triangular=True)
    sar_t033d_rng.buildDiagCd()

    sar_t106d_rng = insar(name='T106D_RngOff', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t106d_rng.read_from_varres(sar_t106d_rng_file, triangular=True)
    sar_t106d_rng.buildDiagCd()

    sar_t070a_rng = insar(name='T070A_RngOff', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t070a_rng.read_from_varres(sar_t070a_rng_file, triangular=True)
    sar_t070a_rng.buildDiagCd()

    sar_t143a_rng = insar(name='T143A_RngOff', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    sar_t143a_rng.read_from_varres(sar_t143a_rng_file, triangular=True)
    sar_t143a_rng.buildDiagCd()

    # Optical data
    opt_p1 = opticorr(name='Optical_Part1', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    opt_p1.read_from_varres(opt_p1_file, triangular=True, cov=True)
    # 让Cd等于原来Cd的对角矩阵
    # opt_p1.Cd = np.diag(np.diag(opt_p1.Cd))
    # opt_p1.buildDiagCd()
    opt_p1.err_east =np.ones_like(opt_p1.east) * 0.3
    opt_p1.err_north =np.ones_like(opt_p1.north) * 0.5
    opt_p1.buildDiagCd()

    opt_p2 = opticorr(name='Optical_Part2', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    opt_p2.read_from_varres(opt_p2_file, triangular=True, cov=True)
    # opt_p2.Cd = np.diag(np.diag(opt_p2.Cd))
    opt_p2.err_east =np.ones_like(opt_p2.east) * 0.3
    opt_p2.err_north =np.ones_like(opt_p2.north) * 0.5
    opt_p2.buildDiagCd()

    # Combine InSAR data and GPS data into geodata
    gpsdata = []
    insar_los_data = [sar_t033d, sar_t106d, sar_t070a, sar_t143a]
    insar_azi_data = [sar_t033d_azi, sar_t106d_azi, sar_t070a_azi, sar_t143a_azi]
    insar_rng_data = [sar_t033d_rng, sar_t106d_rng, sar_t070a_rng, sar_t143a_rng]
    opt_data = [opt_p1, opt_p2]
    insardata = insar_los_data + insar_azi_data
    geodata = insardata + opt_data

    # ----------------------------------Generate Fault Object----------------------------------#
    main_trace_file = os.path.join('..', 'Faults', 'Sagaing_rupture_trace_depicted.dat')
    main_trace = pd.read_csv(main_trace_file, header=None, sep=r'\s+', names=['lon', 'lat'])
    fault_em1 = TriFault(name='Sagaing_Main', lon0=lon0, lat0=lat0, verbose=verbose)
    fault_em1.top = 0.0
    fault_em1.depth = 20.0
    fault_em1.trace(main_trace['lon'].values, main_trace['lat'].values, utm=False)
    fault_em1.set_top_coords_from_trace()
    
    fault_em1.readPatchesFromFile('slip_Sagaing_Main.gmt')
    # fault_em1.plot()

    trifaults = OrderedDict()
    trifaults['Sagaing_Main'] = fault_em1
    trifaults_list = [trifaults[faultname] for faultname in trifaults]

    # Remove some pixels close to the faults
    for idata in geodata:
        idata.reject_pixels_fault(0.25, trifaults_list)

    # -----------------------------Generate Checkerboard----------------------------------#
    fault_em1.generate_checkboard_slip(horizontal_discretization=40, depth_ranges=[0, 10, 20], normalize=True, rake_angle=180.0)
    # fault_em1.plot()
    fault_em1.writePatches2File('output/checkerboard_Sagaing_Main_Input.gmt', add_slip='total')
    print('Checkerboard slip generated, with maximum slip of %.2f m'.format(np.linalg.norm(fault_em1.slip, axis=1).max()))
    # --------------------------------Build GreenFns-----------------------------------------#
    verticals = [True]*len(insardata) + [False]*len(opt_data) # True for InSAR and Opticorr, False for GPS if vertical component is not used
    polys = [None] * len(geodata)  # polys for InSAR and Opticorr, None for GPS
    nonpolys = [None] * len(geodata)

    for ifault in trifaults_list:
        faultname = ifault.name
        for obsdata, vertical in zip(geodata, verticals):
            ifault.buildGFs(obsdata, vertical=vertical, slipdir='sd', verbose=False)
        # ifault.initializeslip()

    # ----------------------Assemble data and GreenFns for Inversion------------------------#
    poly_assembled = False  # flag to check if the polynomial is assembled
    for ifault in trifaults_list:
        # assemble data
        ifault.assembled(geodata, verbose=False)
        # assemble GreensFns
        if not poly_assembled:
            ifault.assembleGFs(geodata, polys=polys, slipdir='sd', verbose=False, custom=False)
            poly_assembled = True
        else:
            ifault.assembleGFs(geodata, polys=nonpolys, slipdir='sd', verbose=False, custom=False)

    # --------------------------Build Covariance Matrix for GPS/InSAR data-------------------#
    # assemble data Covariance matrices, You should assemble the Green's function matrix first
    for ifault in trifaults_list:
        # Bug: the verbose may lead to the error if it is set to True
        ifault.assembleCd(geodata, verbose=False, add_prediction=None)

    # add noise to data
    for isar in insar_los_data:
        isar.buildsynth(faults=trifaults_list, direction='sd', poly=None, vertical=True,)
        err = 0.005 # isar.err.mean()
        isar.add_random_noise(sigma=err, data='synth')
        isar.vel = isar.synth.copy()
    
    for isar in insar_azi_data:
        isar.buildsynth(faults=trifaults_list, direction='sd', poly=None, vertical=True,)
        err = 0.05 # isar.err.mean()
        isar.add_random_noise(sigma=err, data='synth')
        isar.vel = isar.synth.copy()
    
    for iopt in opt_data:
        iopt.buildsynth(faults=trifaults_list, direction='sd', poly=None, vertical=False,)
        err_east = 0.1 # iopt.err_east.mean()
        err_north = 0.1 # iopt.err_north.mean()
        iopt.add_random_noise(sigma_east=err_east, sigma_north=err_north, data='synth')
        iopt.east = iopt.east_synth.copy()
        iopt.north = iopt.north_synth.copy()

    # Save the synthetic data
    if not os.path.exists('Modeling'):
        os.makedirs('Modeling')
    for i, idata in enumerate(geodata):
        if idata.dtype == 'opticorr':
            for itype in ['data']:
                for idir in ['east', 'north']:
                    itypedir = f'{itype}{idir}'
                    idata.writeDecim2file(f'{idata.name}_{itypedir}.txt', itypedir, outDir='Modeling', triangular=True)
        else:
            for itype in ['data']:
                idata.writeDecim2file(f'{idata.name}_{itype}.txt', itype, outDir='Modeling', triangular=True)

    # fault_em1.plot()
    # --------------------------Build BayesianMultiFaultsInversion-------------------#
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
                                             config='default_config.yml', bounds_config='bounds_config.yml')
    inversion.run(penalty_weight=None, alpha=[np.log10(1/100.0)]) # 5.75 = 1/(10**-0.76) -2.1
    inversion.returnModel()

    if rank == 0:
        from csi.faultpostproc import faultpostproc
        myfaultpostproc = faultpostproc('postproc', fault_em1, Mu=3.0e10, lon0=lon0, lat0=lat0, verbose=True)
        myfaultpostproc.computeMomentTensor()
        mw = myfaultpostproc.computeMagnitude()
        print(f'Magnitude of the fault is Mw {mw:.2f}')
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
    #                                 depth=18, zticks=[-20, -10, 0], fault_expand=0.0,
    #                                 plot_faultEdges=False, suffix='mean')
    
    # # inversion.extract_and_plot_blse_results(rank=rank, plot_faults=False, plot_data=True, 
    # #                                         gps_figsize=(3.5, 2.7), gps_scale=0.05, gps_legendscale=0.2, file_type='pdf',
    # #                                         axis_shape=(1.0, 1.0, 0.25), elevation=56, azimuth=-70, gps_title=False,
    # #                                         depth_range=25, z_ticks=[-20, -10, 0], remove_direction_labels=True,
    # #                                         fault_cbaxis=[0.45, 0.32, 0.15, 0.02])

    # Print Statistics Information
    for i, trifault in enumerate(trifaults_list):
        four = trifault.writeFourEdges2File(dirname=r'output/stat_infos')
        trifault.writePatches2File(f'output/slip_{trifault.name}.gmt', add_slip='total')
        trifault.writeSlipCenter2File(f'output/slip_{trifault.name}_center.gmt', add_slip='total', scale=1.0, neg_depth=False)
        trifault.writeSlipDirection2File(filename=f'output/slipdir_{trifault.name}.txt', scale='total', factor=0.6, threshold=0)
    
    for i, sardata in enumerate(insardata):
        if sardata.dtype == 'opticorr':
            for itype in ['data', 'synth', 'res']:
                for idir in ['east', 'north']:
                    itypedir = f'{itype}{idir}'
                    sardata.writeDecim2file(f'{sardata.name}_{itypedir}.txt', itypedir, outDir='Modeling', triangular=True)
        else:
            for itype in ['data', 'synth', 'resid']:
                sardata.writeDecim2file(f'{sardata.name}_{itype}.txt', itype, outDir='Modeling', triangular=True)