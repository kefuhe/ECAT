"""Build a BLSE L-curve for a fixed fault geometry.

Edit the data paths, fixed geometry, configuration files, and penalty-weight
candidates below. The script reuses one assembled BLSE inversion and records
the fit and unweighted model roughness after every independent solve.
"""

# Editing guide
# Edit data, geometry, YAML files, and candidates in Search settings.
# Relative paths in this template start from the current working directory.
# Customize figures in plotting calls or local figure settings; keep execution order.
# Template selection and setup: docs/examples/script_templates.md

import os

os.environ.setdefault("CUTDE_USE_BACKEND", "cpp")

import matplotlib.pyplot as plt
import numpy as np
from csi import gps, insar
from eqtools.viztools import save_fig

from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)
from eqtools.csiExtend.blse_multifaults_inversion import (
    BoundLSEMultiFaultsInversion,
)
from eqtools.csiExtend.blse_diagnostics import (
    plot_blse_lcurve_summary,
    plot_blse_roughness_rms,
)


if __name__ == "__main__":
    # ========================= Shared settings ==========================
    verbose = False
    lon0, lat0 = 87.5, 28.5

    # =============================== Data ===============================
    sar_ascending_file = os.path.join(
        "..", "InSAR", "Ascending", "std_app", "ascending_ifg"
    )
    sar_descending_file = os.path.join(
        "..", "InSAR", "Descending", "std_app", "descending_ifg"
    )

    sar_ascending = insar(
        "Ascending", lon0=lon0, lat0=lat0, utmzone=None, ellps="WGS84", verbose=verbose
    )
    sar_ascending.read_from_varres(sar_ascending_file, triangular=False, cov=True)

    sar_descending = insar(
        "Descending", lon0=lon0, lat0=lat0, utmzone=None, ellps="WGS84", verbose=verbose
    )
    sar_descending.read_from_varres(sar_descending_file, triangular=False, cov=True)

    # Optional: GPS observations. Enable the complete block below. GPS and InSAR
    # then participate in every candidate through the same fixed inversion.
    # Also update gpsdata below and the matching YAML data settings.
    # gps_file = os.path.join("..", "GPS", "gps_enu.txt")
    # gps_network = gps("GNSS", lon0=lon0, lat0=lat0, verbose=verbose)
    # gps_network.read_from_enu(gps_file, factor=1.0, minerr=0.001, header=1)
    # gps_network.buildCd(direction="enu")
    gpsdata = []  # Replace with [gps_network] after enabling the block above.
    insardata = [sar_ascending, sar_descending]
    # Keep this order consistent with the YAML data settings.
    geodata = gpsdata + insardata

    # ========================= Search settings ==========================
    output_dir = "blse_l_curve_results"
    config_file = "default_config_BLSE.yml"
    bounds_file = "bounds_config.yml"

    # Same broad range used by the established BLSE loop template. Larger
    # penalty weight means stronger smoothing. Narrow it after the first scan.
    penalty_weight_candidates = [
        1.0, 5.0, 10.0, 30.0, 50.0, 80.0, 100.0, 125.0, 150.0,
        200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0,
    ]
    preferred_penalty_weight = 100.0  # Visual reference, not an automatic optimum.

    # Plot settings for the entire three-panel figure. The default is the
    # publication double-column width; use a numeric size with unit="cm" when needed.
    plot_options = dict(
        figsize="double",
        figsize_unit="inch",
        label_fontsize=9,
        tick_fontsize=8,
        style="science",  # Use "science-serif" for serif text.
    )
    single_lcurve_options = {
        **plot_options,
        "figsize": "single",
        "legend_loc": "best",
    }
    figure_formats = ("png",)  # For example: ("png", "pdf").
    figure_dpi = 300  # Applies primarily to raster formats such as PNG.

    # Sorting is only for a readable curve; the scanner itself validates the
    # candidates and preserves the sequence it receives.
    penalties = np.sort(np.asarray(penalty_weight_candidates, dtype=float))

    # ===================== Fault geometry and mesh ======================
    fault_name = "MainFault"  # Must match config and bounds source names.
    fault_trace_file = os.path.join("..", "Faults", "main_fault_trace.txt")
    fault_top = 0.0
    fault_depth = 20.0
    fault_dip = 65.0
    dip_direction = 180.0
    top_size = 1.0
    bottom_size = 2.0

    trace_lonlat = np.loadtxt(fault_trace_file, ndmin=2)
    if trace_lonlat.shape[1] < 2:
        raise ValueError(f"{fault_trace_file} must contain lon and lat columns")
    trace_lonlat = np.asarray(trace_lonlat[:, :2], dtype=float)

    fault = TriFault(fault_name, lon0=lon0, lat0=lat0, verbose=verbose)
    fault.top = fault_top
    fault.depth = fault_depth
    fault.trace(trace_lonlat[:, 0], trace_lonlat[:, 1], utm=False)
    fault.set_top_coords_from_trace()
    fault.generate_bottom_from_single_dip(
        dip_angle=fault_dip,
        dip_direction=dip_direction,
    )
    fault.generate_mesh(
        top_size=top_size,
        bottom_size=bottom_size,
        show=False,
        verbose=0,
    )
    fault.initializeslip(values="depth")
    fault.find_fault_fouredge_vertices()
    top_coords = fault.edge_vertices["top"]
    fault.trace(top_coords[:, 0], top_coords[:, 1], utm=True)

    # List order is the source/parameter-block order used by the inversion.
    faults_list = [fault]

    # Optional scientific choice: remove documented decorrelation, unwrapping,
    # or unresolved rupture-zone InSAR pixels before constructing inversion.
    # This changes the observations used by every smoothing candidate.
    # for sardata in insardata:
    #     sardata.reject_pixels_fault(2.0, faults_list)

    # ========================== BLSE inversion ==========================
    inversion = BoundLSEMultiFaultsInversion(
        "smoothing_search",
        faults_list,
        geodata,
        verbose=verbose,
        config=config_file,
        bounds_config=bounds_file,
    )
    inversion.print_parameter_positions()

    # ========================= Smoothing search =========================
    summary, fit_stats = inversion.scan_penalty_weights(
        penalties,
        include_fit_statistics=True,
        verbose=verbose,
    )

    # Legacy simple_run_loop example: docs/reference/blse_vce.md

    # ========================= Results: tables ==========================
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "blse_l_curve.csv")
    fit_csv_file = os.path.join(
        output_dir, "blse_l_curve_fit_statistics.csv"
    )
    summary.to_csv(csv_file, index=False)
    fit_stats.to_csv(fit_csv_file, index=False)

    # ========================= Results: figures =========================
    figure_stem = os.path.join(output_dir, "blse_l_curve")
    fig, _ = plot_blse_lcurve_summary(
        summary,
        preferred_penalty_weight=preferred_penalty_weight,
        **plot_options,
    )
    save_fig(fig, figure_stem, fmts=figure_formats, dpi=figure_dpi)
    plt.close(fig)

    # Save the classic roughness--RMS L-curve as a separate single-panel figure.
    lcurve_figure_stem = os.path.join(
        output_dir, "blse_l_curve_roughness_rms"
    )
    fig, _ = plot_blse_roughness_rms(
        summary,
        preferred_penalty_weight=preferred_penalty_weight,
        **single_lcurve_options,
    )
    save_fig(
        fig, lcurve_figure_stem, fmts=figure_formats, dpi=figure_dpi
    )
    plt.close(fig)

    figure_stems = (figure_stem, lcurve_figure_stem)
    figure_files = [
        f'{stem}.{fmt.lstrip(".")}'
        for stem in figure_stems
        for fmt in figure_formats
    ]

    # Summary
    print(
        "\nNo automatic optimum is selected. Inspect the L-curve, residuals, "
        "and slip models before choosing a penalty weight."
    )
    print(f"Results: {csv_file}")
    print(f"Per-dataset fit statistics: {fit_csv_file}")
    print(f"Figures: {', '.join(figure_files)}")
