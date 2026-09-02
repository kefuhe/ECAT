"""
Fixed-geometry BLSE/VCE slip-inversion template.

Edit the data paths, shared projection, fault geometry, YAML filenames and
penalty weight in their existing sections. The script intentionally keeps a
top-to-bottom research workflow instead of hiding case choices in helpers.

Typical commands:

    python test_slip_inv_BLSE.py --mode single
    python test_slip_inv_BLSE.py --mode single --export-point-values
    python test_slip_inv_BLSE.py --mode loop
"""

import argparse
import os
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("CUTDE_USE_BACKEND", "cpp")

import numpy as np
from csi import gps, insar
from mpi4py import MPI

from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)
from eqtools.csiExtend.blse_multifaults_inversion import (
    BoundLSEMultiFaultsInversion,
)


if __name__ == '__main__':
    # -------------------------------- Parse Arguments -------------------------------
    parser = argparse.ArgumentParser(
        description='Run a fixed-geometry BLSE inversion or a smoothing loop.'
    )
    parser.add_argument('--mode', choices=['single', 'loop'], default='single',
                       help='Execution mode: single run or penalty weight loop (default: single)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip standard slip and data-fit figures.')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--modeling-dir', default='Modeling')
    parser.add_argument(
        '--export-point-values', action='store_true',
        help='Also export InSAR/optical point tables under Modeling/points.',
    )
    args = parser.parse_args()

    # --------------------------- MPI and Output Directories --------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    output_dir = Path(args.output_dir)
    modeling_dir = Path(args.modeling_dir)
    point_values_dir = modeling_dir / 'points'
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        modeling_dir.mkdir(parents=True, exist_ok=True)
        if args.export_point_values:
            point_values_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # ----------------------------- Shared Geographic Reference ----------------------
    # Edit this once; every data set and fault below must use the same origin.
    lon0 = 87.5
    lat0 = 28.5

    # ---------------------------------- Read Data ------------------------------------
    verbose = False
    # Optional GPS example; uncomment all three lines together.
    # gpsfile_6_4 = os.path.join('..', 'GPS', 'GPS_ENU6_4NoEW_CSI.dat')
    # cogps6_4 = gps(name='co6_4', utmzone=None, ellps='WGS84', lon0=lon0, lat0=lat0, verbose=verbose)
    # cogps6_4.read_from_enu(gpsfile_6_4, factor=1., minerr=1., header=1, checkNaNs=True)
    # cogps6_4.buildCd(direction='enu')

    # Replace these two paths with the fixed-geometry inversion inputs.
    sar_t012a_file = os.path.join(
        '..', 'InSAR', 'RawInSAR', 'Dingri_2020_T012A', 'stdBased', 'S1_T012A_ifg'
    )
    sar_t121d_file = os.path.join(
        '..', 'InSAR', 'RawInSAR', 'Dingri_2020_T121D', 'stdBased', 'S1_T121D_ifg'
    )

    sar_t012a = insar(
        name='T012A', utmzone=None, ellps='WGS84',
        lon0=lon0, lat0=lat0, verbose=verbose,
    )
    sar_t012a.read_from_varres(sar_t012a_file, triangular=False, cov=True)

    sar_t121d = insar(
        name='T121D', utmzone=None, ellps='WGS84',
        lon0=lon0, lat0=lat0, verbose=verbose,
    )
    sar_t121d.read_from_varres(sar_t121d_file, triangular=False, cov=True)

    gpsdata = []
    insardata = [sar_t012a, sar_t121d]
    geodata = gpsdata + insardata

    # ------------------------------ Set Fault Geometry -------------------------------
    fault_em1 = TriFault(name='Dingri_2020', lon0=lon0, lat0=lat0, verbose=verbose)
    fault_em1.top = 0.0
    fault_em1.depth = 8.0
    fault_em1.generate_top_bottom_from_nonlinear_soln(
        clon=87.39976, clat=28.66787, cdepth=1.7692,
        strike=332.2241, dip=52.0271, length=12,
    )
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

    # ------------------------ Optional Boundary Diagnostics -------------------------
    if verbose:
        # fault_em1.plot()

        from eqtools.viztools import plot_fault_boundary_diagnostics

        for ifault in trifaults_list:
            ifault.find_fault_fouredge_vertices(
                top_tolerance=0.1,
                bottom_tolerance=0.1,
                edge_method="topology",
                gap_policy="clean",
            )

            plot_fault_boundary_diagnostics(
                ifault,
                coordinates="lonlat",
                save=f"fault_{ifault.name}_boundary_diagnostics.pdf",
                show=True,
            )

    # --------------------------- Build and Run BLSE/VCE -----------------------------
    inversion = BoundLSEMultiFaultsInversion(
        'inv', trifaults_list, geodata, verbose=True,
        config='default_config_BLSE_CovDiag.yml', bounds_config='bounds_config.yml',
    )
    if args.mode == 'single':
        inversion.run(penalty_weight=None, alpha=[np.log10(1/100.0)])
        inversion.returnModel(print_fit_statistics=False)

    elif args.mode == 'loop':
        penalty_weight = [
            1.0, 5.0, 10.0, 30.0, 50.0, 80.0, 100.0, 125.0, 150.0,
            200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0,
        ]
        # Diagnostic only: roughness uses unweighted L0 and the pre-loop model
        # is restored. Re-run --mode single with the selected weight to export.
        inversion.simple_run_loop(
            penalty_weight, preferred_penalty_weight=10.0,
            output_file='run_loop_covdiag.dat', verbose=True,
        )

    if args.mode == 'single':
        # -------------------------- Plot the Final BLSE/VCE Result --------------------------
        # This high-level entry redistributes the solved model and rebuilds synthetic
        # data using the parsed vertical/poly settings. BLSE/VCE has one final model,
        # so a posterior slip-standard-deviation field is not defined here.
        inversion.extract_and_plot_blse_results(
            rank=rank, plot_faults=not args.no_plot, plot_data=not args.no_plot,
            gps_figsize=(3.5, 2.7), gps_scale=0.05, gps_legendscale=0.2,
            file_type='pdf', axis_shape=(1.0, 1.0, 0.25),
            elevation=56, azimuth=-70, gps_title=False,
            depth_range=25, z_ticks=[-20, -10, 0],
            remove_direction_labels=True,
            fault_cbaxis=[0.45, 0.32, 0.15, 0.02],
            data_poly='config', fault_outdir=str(output_dir),
            data_outdir=str(modeling_dir), show=False,
        )

        # ---------------- Optional Custom Slip Figure -----------------------
        # Uncomment and edit this compact block for a publication-specific view.
        # inversion.plot_multifaults_slip(
        #     faults=None, slip='total', cmap='cmc.roma_r', norm=None,
        #     savefig=True, show=False, outdir=str(output_dir), ftype='pdf',
        #     style=['notebook'], shape=(1.0, 1.0, 0.4),
        #     elevation=54, azimuth=24, depth=18, zticks=[-12, -6, 0],
        #     plot_faultEdges=False, suffix='custom',
        # )

        # ------------------------ Fault and Slip Text Products -----------------------
        if rank == 0:
            for trifault in trifaults_list:
                trifault.writeFourEdges2File(dirname=str(output_dir / 'stat_infos'))
                trifault.writePatches2File(
                    str(output_dir / f'slip_{trifault.name}.gmt'), add_slip='total',
                )
                trifault.writeSlipCenter2File(
                    str(output_dir / f'slip_{trifault.name}_center.gmt'),
                    add_slip='total', scale=1.0, neg_depth=False,
                )
                trifault.writeSlipDirection2File(
                    filename=str(output_dir / f'slipdir_{trifault.name}.txt'),
                    scale='total', factor=0.4, threshold=0.0,
                )

            # ------------------------- Modeled Data Text Products -----------------------
            # Synthetic values already correspond to the final model. Polygon files keep
            # the usual Modeling/ layout; optional point tables use Modeling/points/.
            for result_data in inversion.config.geodata['data']:
                if result_data.dtype == 'gps':
                    for data_type in ('data', 'synth', 'resid'):
                        result_data.write2file(
                            f'{result_data.name}_{data_type}.txt', data=data_type,
                            outDir=str(modeling_dir), write_header=True,
                        )
                elif result_data.dtype == 'insar':
                    for data_type in ('data', 'synth', 'resid'):
                        result_data.writeDecim2file(
                            f'{result_data.name}_{data_type}.txt', data_type,
                            outDir=str(modeling_dir), triangular=None,
                        )
                        if args.export_point_values:
                            result_data.write2file(
                                f'{result_data.name}_{data_type}.txt', data=data_type,
                                outDir=str(point_values_dir), write_los=True,
                                write_err=False, write_header=True,
                                precision=None,
                            )
                elif result_data.dtype == 'opticorr':
                    optical_fields = (
                        ('data', 'dataEast', 'dataNorth'),
                        ('synth', 'synthEast', 'synthNorth'),
                        ('resid', 'resEast', 'resNorth'),
                    )
                    for data_type, east_type, north_type in optical_fields:
                        result_data.writeDecim2file(
                            f'{result_data.name}_{east_type}.txt', east_type,
                            outDir=str(modeling_dir), triangular=None,
                        )
                        result_data.writeDecim2file(
                            f'{result_data.name}_{north_type}.txt', north_type,
                            outDir=str(modeling_dir), triangular=None,
                        )
                        if args.export_point_values:
                            result_data.write2file(
                                f'{result_data.name}_{data_type}.txt', data=data_type,
                                outDir=str(point_values_dir), component=None,
                                write_err=False, write_header=True,
                                precision=None,
                            )
