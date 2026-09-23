"""Explore dip and BLSE smoothing-weight sensitivity on a fixed topology.

This is an advanced sensitivity template, not the default first search. For
each dip it rebuilds geometry, Green's functions, Laplacian, and constraints
once, then reuses that inversion while scanning penalty weights.
"""

# Editing guide
# Edit data, geometry, YAML files, and candidates in Search settings.
# Relative paths in this template start from the current working directory.
# Customize figures in plotting calls or local figure settings; keep execution order.
# Template selection and setup: docs/examples/script_templates.md

import csv
import os

os.environ.setdefault("CUTDE_USE_BACKEND", "cpp")

import matplotlib.pyplot as plt
import numpy as np
from csi import insar

from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)
from eqtools.csiExtend.blse_multifaults_inversion import (
    BoundLSEMultiFaultsInversion,
)


def _centers_to_edges(values):
    """Return plotting-cell edges for ordered one-dimensional centers."""
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        (
            [values[0] - (midpoints[0] - values[0])],
            midpoints,
            [values[-1] + (values[-1] - midpoints[-1])],
        )
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

    gpsdata = []
    insardata = [sar_ascending, sar_descending]
    # Keep this order consistent with the YAML data settings.
    geodata = gpsdata + insardata

    # ========================= Search settings ==========================
    output_dir = "dip_smoothing_search_results"
    config_file = "default_config_BLSE.yml"
    bounds_file = "bounds_config.yml"

    # Start from the established broad BLSE loop range, then narrow it after
    # inspecting the first fixed-dip smoothing scan.
    penalty_weight_candidates = [
        1.0, 5.0, 10.0, 30.0, 50.0, 80.0, 100.0, 125.0, 150.0,
        200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0,
    ]

    # Candidate dips use the physical CSI/Okada convention: (0, 90] degrees.
    mainfault_reference_dip = 65.0
    mainfault_dips = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0]

    penalties = np.sort(np.asarray(penalty_weight_candidates, dtype=float))
    if (
        penalties.ndim != 1
        or penalties.size == 0
        or not np.all(np.isfinite(penalties))
        or np.any(penalties <= 0.0)
    ):
        raise ValueError(
            "penalty_weight_candidates must contain finite positive values"
        )
    if np.unique(penalties).size != penalties.size:
        raise ValueError(
            "penalty_weight_candidates must not contain duplicate values"
        )

    dips = np.sort(np.asarray(mainfault_dips, dtype=float))
    if dips.ndim != 1 or dips.size == 0 or not np.all(np.isfinite(dips)):
        raise ValueError("mainfault_dips must contain at least one finite value")
    if np.any((dips <= 0.0) | (dips > 90.0)):
        raise ValueError("mainfault_dips must use physical dips in (0, 90]")
    if np.unique(dips).size != dips.size:
        raise ValueError("mainfault_dips must not contain duplicate values")
    if (
        not np.isfinite(mainfault_reference_dip)
        or mainfault_reference_dip <= 0.0
        or mainfault_reference_dip > 90.0
    ):
        raise ValueError("mainfault_reference_dip must be in (0, 90]")

    # ===================== Fault geometry and mesh ======================
    fault_name = "MainFault"  # Must match config and bounds source names.
    fault_trace_file = os.path.join("..", "Faults", "main_fault_trace.txt")
    fault_top = 0.0
    fault_depth = 20.0
    dip_direction = 180.0
    top_size = 1.0
    bottom_size = 2.0
    mapping_num_segments = 30
    mapping_disct_z = 10

    trace_lonlat = np.loadtxt(fault_trace_file, ndmin=2)
    if trace_lonlat.shape[1] < 2:
        raise ValueError(f"{fault_trace_file} must contain lon and lat columns")
    trace_lonlat = np.asarray(trace_lonlat[:, :2], dtype=float)

    fault = TriFault(
        name=fault_name, lon0=lon0, lat0=lat0, verbose=verbose
    )
    fault.top = fault_top
    fault.depth = fault_depth
    fault.trace(trace_lonlat[:, 0], trace_lonlat[:, 1], utm=False)
    fault.set_top_coords_from_trace()
    fault.generate_bottom_from_single_dip(
        dip_angle=mainfault_reference_dip,
        dip_direction=dip_direction,
    )
    fault.generate_and_deform_mesh(
        fault.top_coords,
        fault.bottom_coords,
        top_size=top_size,
        bottom_size=bottom_size,
        num_segments=mapping_num_segments,
        disct_z=mapping_disct_z,
        show=False,
        verbose=0,
        remap=True,
    )
    reference_npatch = fault.numpatch

    # ===================== Dip and smoothing search =====================
    results = []
    total_runs = len(dips) * len(penalties)
    run_index = 0
    print(
        f"Dip-smoothing grid: {len(dips)} dips x {len(penalties)} "
        f"penalty weights = {total_runs} BLSE solves"
    )

    for dip in dips:
        fault.generate_bottom_from_single_dip(
            dip_angle=float(dip),
            dip_direction=dip_direction,
        )
        fault.generate_and_deform_mesh(
            fault.top_coords,
            fault.bottom_coords,
            top_size=top_size,
            bottom_size=bottom_size,
            num_segments=mapping_num_segments,
            disct_z=mapping_disct_z,
            show=False,
            verbose=0,
            remap=False,
        )
        if fault.numpatch != reference_npatch:
            raise RuntimeError(
                "fixed-topology dip-smoothing search changed patch count"
            )

        fault.initializeslip(values="depth")
        fault.find_fault_fouredge_vertices()
        top_coords = fault.edge_vertices["top"]
        fault.trace(top_coords[:, 0], top_coords[:, 1], utm=True)

        faults_list = [fault]

        # Rebuild geometry-dependent state once per dip. The inner penalty loop
        # reuses this G/L/constraint layout and only resolves a new BLSE model.
        inversion = BoundLSEMultiFaultsInversion(
            f"dip_{dip:g}",
            faults_list,
            geodata,
            verbose=verbose,
            config=config_file,
            bounds_config=bounds_file,
        )
        if not inversion.config.alpha_enabled:
            raise ValueError(
                "dip-smoothing search requires alpha.enabled: true in the "
                "BLSE config"
            )

        for penalty_weight in penalties:
            run_index += 1
            print(
                f"[{run_index}/{total_runs}] dip={dip:g} degree, "
                f"penalty weight={penalty_weight:g}"
            )
            inversion.run(
                penalty_weight=float(penalty_weight),
                alpha=None,
                verbose=verbose,
            )
            roughness, solver_rms, solver_vr = inversion.returnModel(
                print_fit_statistics=False
            )

            fit_rows = inversion.collect_fit_statistics(
                model=f"dip_{dip:g}_penalty_{penalty_weight:g}",
                data_poly="config",
                include_dataset=True,
                include_global=True,
            )
            global_fit = next(
                row
                for row in fit_rows
                if row["scope"] == "global_solver_vector"
            )
            if (
                not np.isclose(
                    solver_rms,
                    global_fit["rms"],
                    rtol=1e-10,
                    atol=1e-12,
                )
                or not np.isclose(
                    solver_vr,
                    global_fit["vr"],
                    rtol=1e-8,
                    atol=1e-6,
                )
            ):
                raise RuntimeError(
                    "returnModel and structured fit statistics disagree"
                )

            resolved_penalties = np.asarray(
                inversion.current_penalty_weight, dtype=float
            ).reshape(-1)
            result = {
                "dip_deg": float(dip),
                "penalty_weight": float(penalty_weight),
                "equivalent_log10_alpha": float(
                    np.log10(1.0 / penalty_weight)
                ),
                "resolved_penalty_weights": ";".join(
                    f"{value:.12g}" for value in resolved_penalties
                ),
                "n_patches": int(fault.numpatch),
                "roughness": float(roughness),
                "global_rms": float(global_fit["rms"]),
                "global_vr_percent": float(global_fit["vr"]),
            }
            for fit_row in fit_rows:
                if fit_row["scope"] == "dataset":
                    data_name = str(fit_row["dataset"])
                    result[f"{data_name}_rms"] = float(fit_row["rms"])
                    result[f"{data_name}_vr_percent"] = float(
                        fit_row["vr"]
                    )
            results.append(result)

    # ========================= Results: tables ==========================
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "dip_smoothing_search.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    rms_grid = np.full((len(penalties), len(dips)), np.nan, dtype=float)
    vr_grid = np.full_like(rms_grid, np.nan)
    for result in results:
        i = int(np.where(penalties == result["penalty_weight"])[0][0])
        j = int(np.where(dips == result["dip_deg"])[0][0])
        rms_grid[i, j] = result["global_rms"]
        vr_grid[i, j] = result["global_vr_percent"]

    # ==================== Results: sensitivity grid =====================
    log_penalties = np.log10(penalties)
    dip_edges = _centers_to_edges(dips)
    penalty_edges = _centers_to_edges(log_penalties)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    rms_image = axes[0].pcolormesh(
        dip_edges,
        penalty_edges,
        rms_grid,
        shading="flat",
        cmap="viridis",
    )
    axes[0].set_title("Global RMS")
    axes[0].set_xlabel("Dip (degree)")
    axes[0].set_ylabel("log10(penalty weight)")
    fig.colorbar(rms_image, ax=axes[0], shrink=0.85)

    vr_image = axes[1].pcolormesh(
        dip_edges,
        penalty_edges,
        vr_grid,
        shading="flat",
        cmap="viridis_r",
    )
    axes[1].set_title("Global VR (%)")
    axes[1].set_xlabel("Dip (degree)")
    axes[1].set_ylabel("log10(penalty weight)")
    fig.colorbar(vr_image, ax=axes[1], shrink=0.85)

    grid_figure_file = os.path.join(
        output_dir, "dip_smoothing_grid.png"
    )
    fig.savefig(grid_figure_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ======================== Results: L-curves =========================
    fig, axis = plt.subplots(figsize=(6.2, 4.6))
    cmap = plt.get_cmap("viridis")
    dip_norm = plt.Normalize(vmin=float(dips.min()), vmax=float(dips.max()))
    for dip in dips:
        dip_rows = [row for row in results if row["dip_deg"] == float(dip)]
        roughness_values = np.array(
            [row["roughness"] for row in dip_rows], dtype=float
        )
        rms_values = np.array(
            [row["global_rms"] for row in dip_rows], dtype=float
        )
        axis.plot(
            roughness_values,
            rms_values,
            "o-",
            color=cmap(dip_norm(float(dip))),
            label=f"{dip:g} deg",
        )
    axis.set_xlabel("Roughness")
    axis.set_ylabel("Global RMS")
    axis.grid(alpha=0.25)
    axis.legend(title="Dip", ncol=2, fontsize="small")
    lcurve_figure_file = os.path.join(
        output_dir, "roughness_vs_rms_by_dip.png"
    )
    fig.tight_layout()
    fig.savefig(lcurve_figure_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Summary
    print(
        "\nNo global minimum-RMS model is selected: weak smoothing normally "
        "wins that criterion. Inspect L-curves, residuals, roughness, and slip "
        "models before choosing a dip and penalty weight."
    )
    print(f"Results : {csv_file}")
    print(f"Grid    : {grid_figure_file}")
    print(f"L-curves: {lcurve_figure_file}")
