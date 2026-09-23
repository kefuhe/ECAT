"""Run a serial InSAR/GPS/optical checkerboard resolution test with BLSE.

Edit the data paths, fixed geometry, checkerboard pattern, noise levels, and
configuration files below. Optional GPS and optical blocks use the same
synthetic-injection entry as InSAR.
"""

# Editing guide
# Edit data, geometry, YAML files, checkerboard pattern, and noise levels.
# Relative paths in this template start from the current working directory.
# Customize figures in plotting calls or local figure settings; keep execution order.
# Template selection and setup: docs/examples/script_templates.md

import os
from pathlib import Path

os.environ.setdefault("CUTDE_USE_BACKEND", "cpp")

import numpy as np
from csi import gps, insar, opticorr

from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)
from eqtools.csiExtend.InvTools.CheckerboardInversion import (
    CheckerboardInversion,
)


def main():
    # ========================= Shared settings ==========================
    verbose = True
    lon0, lat0 = 86.802, 33.172

    # =============================== Data ===============================
    # Only observation locations, projections, component availability, and
    # uncertainties are needed here; measured values are replaced below.
    # Optional: GPS observations. Enable the complete block below.
    # Also update gpsdata, noise_config, and matching YAML data settings.
    # gps_file = Path("..") / "GPS" / "gps_enu.txt"
    # gps_network = gps(
    #     "GNSS", lon0=lon0, lat0=lat0, utmzone=None, ellps="WGS84", verbose=False
    # )
    # gps_network.read_from_enu(
    #     gps_file, factor=1.0, minerr=0.001, header=1, checkNaNs=True
    # )
    # gps_network.buildCd(direction="enu")  # Use "en" when U is absent/NaN.

    # Optional two-component optical example; uncomment the block and replace
    # the prefix; update opticaldata, noise_config, and YAML data settings.
    # Its matching YAML verticals entry must be false.
    # optical_file = Path("..") / "Optical" / "downsample" / "Optical_ifg"
    # optical_offsets = opticorr(
    #     "Optical", lon0=lon0, lat0=lat0, utmzone=None,
    #     ellps="WGS84", verbose=False,
    # )
    # optical_offsets.read_from_varres(optical_file, triangular=True, cov=True)

    sar_t012a_file = (
        Path("..") / "InSAR" / "downsample" / "T012A" / "S1_T012A_ifg"
    )
    sar_t121d_file = (
        Path("..") / "InSAR" / "downsample" / "T121D" / "S1_T121D_ifg"
    )

    print("Loading InSAR observation geometry...")

    sar_t012a = insar(
        "T012A", lon0=lon0, lat0=lat0, utmzone=None, ellps="WGS84", verbose=False
    )
    sar_t012a.read_from_varres(sar_t012a_file, triangular=False, cov=True)

    sar_t121d = insar(
        "T121D", lon0=lon0, lat0=lat0, utmzone=None, ellps="WGS84", verbose=False
    )
    sar_t121d.read_from_varres(sar_t121d_file, triangular=False, cov=True)

    gpsdata = []  # Replace with [gps_network] after enabling the block above.
    opticaldata = []  # Replace with [optical_offsets] after enabling its block.
    insardata = [sar_t012a, sar_t121d]
    # Keep this order consistent with the YAML data settings.
    geodata = gpsdata + insardata + opticaldata

    # ===================== Fault geometry and mesh ======================
    fault_em1 = TriFault("Nima_2020", lon0=lon0, lat0=lat0, verbose=verbose)
    fault_em1.top = 0.0
    fault_em1.depth = 20.0
    # clon/clat/cdepth specify the top-edge midpoint (degrees/degrees/km).
    fault_em1.generate_top_bottom_from_nonlinear_soln(
        clon=86.871102,
        clat=33.191409,
        cdepth=5.971379,
        strike=31.504098,
        dip=53.327562,
        length=20,
    )
    fault_em1.generate_mesh(top_size=1.0, bottom_size=1.5, show=False, verbose=0)
    fault_em1.initializeslip(values="depth")
    fault_em1.find_fault_fouredge_vertices()
    top_coords = fault_em1.edge_vertices["top"]
    fault_em1.trace(top_coords[:, 0], top_coords[:, 1], utm=True)

    # List order is the source/parameter-block order used by the inversion.
    faults_list = [fault_em1]

    # Optional scientific choice: remove documented decorrelation, unwrapping,
    # or unresolved rupture-zone pixels before constructing the inversion.
    # This changes observational coverage and the resolution test itself.
    # for sardata in insardata:
    #     sardata.reject_pixels_fault(1.0, faults_list)

    # ========================== BLSE inversion ==========================
    inversion = CheckerboardInversion(
        name="checkerboard_test",
        faults=faults_list,
        data=geodata,
        verbose=verbose,
        config="default_config_BLSE.yml",
        bounds_config="bounds_config.yml",
    )

    # ============= Checkerboard and synthetic observations ==============
    # horizontal_discretization supports a physical size, a number of cells
    # along strike, or one physical size per depth range.
    inversion.add_checkerboard_pattern(
        fault_name="Nima_2020",
        horizontal_discretization=6.7,
        depth_ranges=[0.0, 5.0, 10.0],
        normalize=True,
        rake_angle=-70.0,
        target_magnitude=6.3,
        start_with_slip=True,
    )
    inversion.save_true_model("output/checkerboard_truth")

    # Generate the noisy observations reproducibly. Dataset names must match
    # the objects above. A scalar GPS value gives every active component the
    # same standard deviation; mappings allow component-specific GPS/optical
    # noise. update_weight=True rebuilds each corresponding diagonal Cd.
    random_seed = 2026
    np.random.seed(random_seed)
    noise_config = {
        # "GNSS": {"east": 0.002, "north": 0.002, "up": 0.006},
        # "Optical": {"east": 0.05, "north": 0.08},
        "T012A": 0.003,
        "T121D": 0.005,
    }
    inversion.apply_synthetics(
        noise_sigma=noise_config,
        update_weight=True,
        save_dir="Modeling",
    )
    inversion.plot_inputs(
        plot_faults=True,
        plot_data=False,
        cmap="cmc.roma_r",
        save_dir="output/inputs_visualization",
        figsize=(3.5, 3.5),
        show=False,
    )

    # ============================ Solve BLSE ============================
    print("Running checkerboard inversion...")
    inversion.print_parameter_positions()
    inversion.run(
        penalty_weight=None,
        alpha=[np.log10(1 / 30.0)],
    )
    inversion.returnModel(print_fit_statistics=False)

    # ========================= Results: figures =========================
    inversion.extract_and_plot_blse_results(
        plot_faults=True,
        plot_data=False,
        cmap="RdBu_r",
        slip_cmap="cmc.roma_r",
        file_type="pdf",
        gps_title=False,
        depth_range=25,
        z_ticks=[-20, -10, 0],
        remove_direction_labels=True,
    )

    inversion.plot_model_comparison(
        cmap="cmc.roma_r",
        slip_type="totalslip",
        save_path="output/inputs_visualization",
        show=True,
    )

    for sardata in insardata:
        sardata.plot_fit_comparison(
            faults=faults_list,
            cmap="RdBu_r",
            share_colorbar=True,
            save_path=(
                f"output/inputs_visualization/"
                f"{sardata.name}_fit_comparison.pdf"
            ),
            show=True,
        )

    # ====================== Results: text products ======================
    output_dir = Path("output")
    stat_dir = output_dir / "stat_infos"
    modeling_dir = Path("Modeling")
    output_dir.mkdir(parents=True, exist_ok=True)
    stat_dir.mkdir(parents=True, exist_ok=True)
    modeling_dir.mkdir(parents=True, exist_ok=True)

    for fault in faults_list:
        fault.writeFourEdges2File(dirname=str(stat_dir))
        fault.writePatches2File(
            str(output_dir / f"slip_{fault.name}.gmt"), add_slip="total"
        )
        fault.writeSlipDirection2File(
            filename=str(output_dir / f"slipdir_{fault.name}.txt"),
            scale="total",
        )

    # Preserve decimation polygons when corner geometry exists; point-mode
    # InSAR and optical inputs are exported directly as point tables.
    for sardata in insardata:
        has_corner = sardata.corner_mode is not None
        for data_type in ("data", "synth", "resid"):
            if has_corner:
                sardata.writeDecim2file(
                    f"{sardata.name}_{data_type}.txt",
                    data_type,
                    outDir=str(modeling_dir),
                    triangular=None,
                )
            else:
                sardata.write2file(
                    f"{sardata.name}_{data_type}.txt",
                    data=data_type,
                    outDir=str(modeling_dir),
                    write_los=True,
                    write_err=False,
                    write_header=True,
                    precision=None,
                )

    for optical_offsets in opticaldata:
        has_corner = optical_offsets.corner_mode is not None
        for data_type in ("data", "synth", "resid"):
            if has_corner:
                mode = {"data": "data", "synth": "synth", "resid": "res"}[
                    data_type
                ]
                for component in ("East", "North"):
                    optical_offsets.writeDecim2file(
                        f"{optical_offsets.name}_{data_type}_{component.lower()}.txt",
                        data=f"{mode}{component}",
                        outDir=str(modeling_dir),
                        triangular=None,
                    )
            else:
                optical_offsets.write2file(
                    f"{optical_offsets.name}_{data_type}.txt",
                    data=data_type,
                    outDir=str(modeling_dir),
                    component=None,
                    write_err=False,
                    write_header=True,
                )

    for gps_network in gpsdata:
        for data_type in ("data", "synth", "resid"):
            gps_network.write2file(
                f"{gps_network.name}_{data_type}.txt",
                data=data_type,
                outDir=str(modeling_dir),
                write_header=True,
            )


if __name__ == "__main__":
    main()
