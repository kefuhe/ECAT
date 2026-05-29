import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
from mpi4py import MPI

# CSI / EqTools Imports
from eqtools.csiExtend.AdaptiveRectangularPatches import AdaptiveRectangularPatches as RectFault
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault
)
from csi import insar
from eqtools.csiExtend.InvTools.CheckerboardInversion import CheckerboardInversion

# Set backend
os.environ['CUTDE_USE_BACKEND'] = 'cpp'

def main():
    # -----------------------------------MPI Init---------------------------------------------#
    from eqtools.csiExtend.logging_utils.mpi_logging import setup_parallel_logging
    setup_parallel_logging(log_filename='checkerboard_run.log', console_output=False)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    verbose = (rank == 0)

    # -----------------------------------Read Data---------------------------------------------#
    lon0, lat0 = 86.802, 33.172
    
    # Modify the path according to your actual situation
    # Only read data structure (lon/lat positions) here, values will be overwritten by synthetic data
    sar_t012a_file = pathlib.Path('..') / 'InSAR' / 'downsample' / 'T012A' / 'S1_T012A_ifg'
    sar_t121d_file = pathlib.Path('..') / 'InSAR' / 'downsample' / 'T121D' / 'S1_T121D_ifg'

    if verbose: print("Loading Geodata structures...")

    # Read InSAR structure
    sar_t012a = insar(name='T012A', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=False)
    sar_t012a.read_from_varres(sar_t012a_file, triangular=False, cov=True)

    sar_t121d = insar(name='T121D', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=False)
    sar_t121d.read_from_varres(sar_t121d_file, triangular=False, cov=True)

    # Put into list
    insardata = [sar_t012a, sar_t121d]
    gpsdata = []
    geodata = insardata
    # ----------------------------------Generate Fault-----------------------------------------#
    fault_em1 = TriFault(name='Nima_2020', lon0=lon0, lat0=lat0, verbose=verbose)
    fault_em1.top = 0.0
    fault_em1.depth = 20.0
    fault_em1.generate_top_bottom_from_nonlinear_soln(clon=86.871102, clat=33.191409, cdepth=5.971379, strike=31.504098, dip=53.327562, length=20)
    fault_em1.generate_mesh(top_size=1.0, bottom_size=1.5, show=False, verbose=0)
    fault_em1.initializeslip(values='depth')
    fault_em1.find_fault_fouredge_vertices()
    top_coords = fault_em1.edge_vertices['top']
    fault_em1.trace(top_coords[:, 0], top_coords[:, 1], utm=True)
    # fault_em1.plot()

    from collections import OrderedDict
    trifaults = OrderedDict()
    trifaults['Nima_2020'] = fault_em1
    trifaults_list = [trifaults[faultname] for faultname in trifaults]

    # Remove some pixels close to the faults
    # for sardata in insardata:
    #     sardata.reject_pixels_fault(1.0, trifaults_list)

    # --------------------------------Init Inversion-----------------------------------------#
    inv = CheckerboardInversion(
        name='checkboard_test', 
        faults=trifaults_list, 
        data=geodata, 
        verbose=verbose,
        config='default_config_BLSE_CovDiag.yml', 
        bounds_config='bounds_config.yml'
    )

    # --------------------------------Set Checkerboard---------------------------------------#
    # Add checkerboard pattern to the fault
    # horizontal_discretization: three modes are supported
    #   1) float: the size of each checkerboard square in km
    #   2) int: the number of checkerboard squares along the fault strike direction
    #   3) list of float: the size of each checkerboard square in km for each depth range
    inv.add_checkerboard_pattern(
        fault_name='Nima_2020',
        horizontal_discretization=6.7,   
        depth_ranges=[0.0, 5.0, 10.0],   
        normalize=True,                  
        rake_angle=-70.0,                
        target_magnitude=6.3,            
        start_with_slip=True             
    )

    if rank == 0:
        # 4. Save true model (Ground Truth)
        inv.save_true_model('output/checkerboard_truth')

        # 5. [Upgrade] Set differential noise and generate data
        # Use dictionary to set different noise levels for each dataset
        noise_config = {
            'T012A': 0.003,  # 3mm noise
            'T121D': 0.005   # 5mm noise
        }
        
        # apply_synthetics will automatically:
        # 1. Forward modeling 2. Add noise 3. Replace vel 4. Update data weights (Cd) 5. Update inv.d
        inv.apply_synthetics(noise_sigma=noise_config, update_weight=True, save_dir='Modeling')

        # 6. [Upgrade] Plot input status (view before inversion starts)
        # This will plot the True Slip Model and Synthetic Noisy Data
        inv.plot_inputs(
            plot_faults=True, 
            plot_data=False, 
            cmap='cmc.roma_r', 
            save_dir='output/inputs_visualization', 
            figsize=(3.5, 3.5),
            show=False # Set to False if running on a server
        )

    # --------------------------------Run Inversion------------------------------------------#
    if verbose: print("Running Inversion...")
    
    # Run inversion
    # penalty_weight can be set to None (use config) or specified manually
    inv.run(penalty_weight=None, alpha=[np.log10(1/30.0)]) 
    
    # Get result model
    inv.returnModel(print_stat=False)

    # --------------------------------Plot Results-------------------------------------------#
    if rank == 0:
        print("Plotting posterior results...")
        inv.extract_and_plot_blse_results(
            rank=rank, 
            plot_faults=True, 
            plot_data=False, 
            cmap='RdBu_r', 
            slip_cmap='cmc.roma_r',
            file_type='pdf', 
            gps_title=False,
            depth_range=25, 
            z_ticks=[-20, -10, 0], 
            remove_direction_labels=True
        )

        # 1. Model comparison (Truth vs Result)
        inv.plot_model_comparison(
            cmap='cmc.roma_r',   
            slip_type='totalslip',   
            save_path='output/inputs_visualization',
            show=True              
        )

        # 2. Data comparison (Synthetic Data vs Recovered Data)
        for sardata in insardata:
            sardata.plot_fit_comparison(
                faults=trifaults_list,
                cmap='RdBu_r',
                share_colorbar=True,
                save_path=f'output/inputs_visualization/{sardata.name}_fit_comparison.pdf',
                show=True
            )

        # ---------------------------------Write Results to File---------------------------------------------#
        if rank == 0:
            for i, trifault in enumerate(trifaults_list):
                four = trifault.writeFourEdges2File(dirname=r'output/stat_infos')
                trifault.writePatches2File(f'output/slip_{trifault.name}.gmt', add_slip='total')
                trifault.writeSlipDirection2File(filename=f'output/slipdir_{trifault.name}.txt', 
                                                scale='total')
        # ---------------------------------Write Data to File---------------------------------------------#
        if rank == 0:
            # Create Modeling directory (if not exists)
            outDir = 'Modeling'
            if not os.path.exists(outDir):
                os.makedirs(outDir)

            # Write the InSAR data to file
            for i, sardata in enumerate(insardata):
                if sardata.dtype == 'opticorr':
                    for itype in ['data', 'synth', 'res']:
                        for idir in ['east', 'north']:
                            itypedir = f'{itype}{idir}'
                            sardata.writeDecim2file(f'{sardata.name}_{itypedir}.txt', itypedir, outDir='Modeling', triangular=True)
                else:
                    for itype in ['data', 'synth', 'resid']:
                        #sardata.writeDecim2file(f'{sardata.name}_{itype}.txt', itype, outDir='Modeling', triangular=True)
                        sardata.write2file(f'{sardata.name}_{itype}.txt', itype, outDir='Modeling') 
            # Write the GPS data to file
            for i, gpsdata in enumerate(gpsdata):
                for itype in ['data', 'synth', 'resid']:
                    gpsdata.write2file(f'{gpsdata.name}_{itype}.txt', itype, outDir='Modeling')

if __name__ == '__main__':
    main()