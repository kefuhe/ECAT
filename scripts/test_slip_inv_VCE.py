"""
Fixed-geometry VCE slip-inversion template.

Edit the data paths, shared projection, fault geometry, YAML filenames and
VCE component settings in their existing sections. The script intentionally
keeps a top-to-bottom research workflow instead of hiding case choices in
helpers.

Typical commands:

    python test_slip_inv_VCE.py
    python test_slip_inv_VCE.py --export-point-values

This template runs one VCE solve. Use test_BLSE_L_Curve.py for a
fixed-geometry BLSE L-curve search.
"""

# Editing guide
# Edit data paths, shared projection, fault geometry, and solver settings.
# Relative paths in this template start from the current working directory.
# Customize figures in plotting calls or local figure settings; keep execution order.
# Template selection and setup: docs/examples/script_templates.md

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUTDE_USE_BACKEND", "cpp")

from csi import gps, insar

from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)
from eqtools.csiExtend.blse_multifaults_inversion import (
    BoundLSEMultiFaultsInversion,
)


if __name__ == '__main__':
    # ========================= Runtime options ==========================
    parser = argparse.ArgumentParser(
        description='Run one fixed-geometry VCE slip inversion.'
    )
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip standard slip and data-fit figures.')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--modeling-dir', default='Modeling')
    parser.add_argument(
        '--export-point-values', action='store_true',
        help=(
            'For raster data with corners, also export point tables under '
            'Modeling/points.'
        ),
    )
    args = parser.parse_args()

    # Output directories
    output_dir = Path(args.output_dir)
    modeling_dir = Path(args.modeling_dir)
    point_values_dir = modeling_dir / 'points'
    output_dir.mkdir(parents=True, exist_ok=True)
    modeling_dir.mkdir(parents=True, exist_ok=True)
    if args.export_point_values:
        point_values_dir.mkdir(parents=True, exist_ok=True)

    # ========================= Shared settings ==========================
    # Edit these once; every data set and fault below uses the same projection origin.
    verbose = False
    lon0, lat0 = 87.5, 28.5

    # =============================== Data ===============================
    # Optional: GPS observations. Enable the complete block below.
    # Also update gpsdata below and the matching YAML data settings.
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
        'T012A', lon0=lon0, lat0=lat0, utmzone=None, ellps='WGS84', verbose=verbose
    )
    sar_t012a.read_from_varres(sar_t012a_file, triangular=False, cov=True)

    sar_t121d = insar(
        'T121D', lon0=lon0, lat0=lat0, utmzone=None, ellps='WGS84', verbose=verbose
    )
    sar_t121d.read_from_varres(sar_t121d_file, triangular=False, cov=True)

    gpsdata = []  # Use [cogps6_4] after enabling the GPS block.
    insardata = [sar_t012a, sar_t121d]
    # Keep this order consistent with the YAML data settings.
    geodata = gpsdata + insardata

    # ===================== Fault geometry and mesh ======================
    fault_em1 = TriFault(name='Dingri_2020', lon0=lon0, lat0=lat0, verbose=verbose)
    fault_em1.top = 0.0
    fault_em1.depth = 8.0
    # clon/clat/cdepth specify the top-edge midpoint (degrees/degrees/km).
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

    # List order is the source/parameter-block order used by the inversion.
    faults_list = [fault_em1]

    # Optional: exclude near-fault pixels before building the inversion.
    # This changes the observations used by the solve or scan.
    # for sardata in insardata:
    #     sardata.reject_pixels_fault(1.0, faults_list)

    # Optional: boundary diagnostics
    # This template uses verbose to enable both logging and these figures.
    if verbose:
        # fault_em1.plot()

        from eqtools.viztools import plot_fault_boundary_diagnostics

        for ifault in faults_list:
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

    # ========================== VCE inversion ===========================
    inversion = BoundLSEMultiFaultsInversion(
        'inv', faults_list, geodata, verbose=verbose,
        config='default_config_VCE.yml', bounds_config='bounds_config.yml',
    )
    inversion.print_parameter_positions()

    vce_result = inversion.run_simple_vce(
        max_iter=20,
        tol=1e-4,
        verbose=verbose,
        # qp_acceleration='certified_kkt',  # Optional certified fast path.
    )
    inversion.returnModel(print_fit_statistics=False)

    # ============ Results: final model and standard figures =============
    # This high-level entry redistributes the solved model and rebuilds synthetic
    # data using the parsed vertical/poly settings. VCE has one final model,
    # so a posterior slip-standard-deviation field is not defined here.
    inversion.extract_and_plot_blse_results(
        plot_faults=not args.no_plot, plot_data=not args.no_plot,
        gps_figsize=(3.5, 2.7), gps_scale=0.05, gps_legendscale=0.2,
        file_type='pdf', axis_shape=(1.0, 1.0, 0.25),
        elevation=56, azimuth=-70, gps_title=False,
        depth_range=25, z_ticks=[-20, -10, 0],
        remove_direction_labels=True,
        fault_cbaxis=[0.45, 0.32, 0.15, 0.02],
        data_poly='config', fault_outdir=str(output_dir),
        data_outdir=str(modeling_dir), show=False,
    )

    # Optional: custom slip figure
    # Uncomment and edit this compact block for a publication-specific view.
    # inversion.plot_multifaults_slip(
    #     faults=None, slip='total', cmap='cmc.roma_r', norm=None,
    #     savefig=True, show=False, outdir=str(output_dir), ftype='pdf',
    #     style=['notebook'], shape=(1.0, 1.0, 0.4),
    #     elevation=54, azimuth=24, depth=18, zticks=[-12, -6, 0],
    #     plot_faultEdges=False, suffix='custom',
    # )

    # Fault and slip text products
    for trifault in faults_list:
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

    # Modeled data text products
    # Synthetic values already correspond to the final model. Raster data with
    # corners keep polygon files; point inputs stay as point tables in Modeling/.
    for result_data in inversion.config.geodata['data']:
        if result_data.dtype == 'gps':
            # GPS: active components
            for data_type in ('data', 'synth', 'resid'):
                result_data.write2file(
                    f'{result_data.name}_{data_type}.txt', data=data_type,
                    outDir=str(modeling_dir), write_header=True,
                )
        elif result_data.dtype == 'insar':
            # InSAR: polygons or point tables
            has_corner = result_data.corner_mode is not None
            point_outdir = point_values_dir if has_corner else modeling_dir
            for data_type in ('data', 'synth', 'resid'):
                if has_corner:
                    result_data.writeDecim2file(
                        f'{result_data.name}_{data_type}.txt', data_type,
                        outDir=str(modeling_dir), triangular=None,
                    )
                if not has_corner or args.export_point_values:
                    result_data.write2file(
                        f'{result_data.name}_{data_type}.txt', data=data_type,
                        outDir=str(point_outdir), write_los=True,
                        write_err=False, write_header=True,
                        precision=None,
                    )
        elif result_data.dtype == 'opticorr':
            # Optical: east/north components
            has_corner = result_data.corner_mode is not None
            point_outdir = point_values_dir if has_corner else modeling_dir
            optical_fields = (
                ('data', 'dataEast', 'dataNorth'),
                ('synth', 'synthEast', 'synthNorth'),
                ('resid', 'resEast', 'resNorth'),
            )
            for data_type, east_type, north_type in optical_fields:
                if has_corner:
                    result_data.writeDecim2file(
                        f'{result_data.name}_{east_type}.txt', east_type,
                        outDir=str(modeling_dir), triangular=None,
                    )
                    result_data.writeDecim2file(
                        f'{result_data.name}_{north_type}.txt', north_type,
                        outDir=str(modeling_dir), triangular=None,
                    )
                if not has_corner or args.export_point_values:
                    result_data.write2file(
                        f'{result_data.name}_{data_type}.txt', data=data_type,
                        outDir=str(point_outdir), component=None,
                        write_err=False, write_header=True,
                        precision=None,
                    )
