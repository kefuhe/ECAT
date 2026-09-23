"""Search fault dip with BLSE while preserving triangular patch topology.

Edit the paths, fault geometry, dip candidates, and BLSE configuration below.
The reference mesh is generated once; every candidate deforms that same mesh,
then rebuilds Green's functions and solves a fresh BLSE problem.
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
    output_dir = "dip_search_results"
    config_file = "default_config_BLSE.yml"
    bounds_file = "bounds_config.yml"
    alpha = [-2.0]  # Follow alpha.mode and alpha.log_scaled in the config.

    # Candidate dips use the physical CSI/Okada convention: (0, 90] degrees.
    # The reference dip controls only the mesh used to preserve patch identity.
    mainfault_reference_dip = 65.0
    mainfault_dips = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0]

    # ===================== Fault geometry and mesh ======================
    fault_name = "MainFault"  # Must match config and bounds source names.
    fault_trace_file = os.path.join("..", "Faults", "main_fault_trace.txt")
    fault_top = 0.0
    fault_depth = 20.0
    dip_direction = 180.0
    top_size = 1.0
    bottom_size = 2.0

    # These control the reference-coordinate mapping, not the patch count.
    mapping_num_segments = 30
    mapping_disct_z = 10

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

    # remap=True creates the reference mesh and vertex mapping exactly once.
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

    # ============================ Dip search ============================
    results = []
    for index, dip in enumerate(dips, start=1):
        print(f"[{index}/{len(dips)}] BLSE dip = {dip:g} degree")

        # remap=False updates physical coordinates while preserving patch rows.
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
            raise RuntimeError("fixed-topology dip search changed patch count")

        fault.initializeslip(values="depth")
        fault.find_fault_fouredge_vertices()
        top_coords = fault.edge_vertices["top"]
        fault.trace(top_coords[:, 0], top_coords[:, 1], utm=True)

        faults_list = [fault]

        # Geometry, Green's functions, Laplacian, and constraints all depend on
        # the current dip, so each candidate uses a fresh inversion object.
        inversion = BoundLSEMultiFaultsInversion(
            f"dip_{dip:g}",
            faults_list,
            geodata,
            verbose=verbose,
            config=config_file,
            bounds_config=bounds_file,
        )
        inversion.run(
            penalty_weight=None,
            alpha=alpha,
            verbose=verbose,
        )
        roughness, solver_rms, solver_vr = inversion.returnModel(
            print_fit_statistics=False
        )

        # data_poly="config" includes the polynomial/ramp terms configured for
        # each dataset, matching the model used by the BLSE solution.
        fit_rows = inversion.collect_fit_statistics(
            model=f"dip_{dip:g}",
            data_poly="config",
            include_dataset=True,
            include_global=True,
        )
        global_fit = next(
            row for row in fit_rows if row["scope"] == "global_solver_vector"
        )
        if (
            not np.isclose(
                solver_rms, global_fit["rms"], rtol=1e-10, atol=1e-12
            )
            or not np.isclose(
                solver_vr, global_fit["vr"], rtol=1e-8, atol=1e-6
            )
        ):
            raise RuntimeError(
                "returnModel and structured fit statistics disagree"
            )

        result = {
            "dip_deg": float(dip),
            "n_patches": int(fault.numpatch),
            "roughness": float(roughness),
            "global_rms": float(global_fit["rms"]),
            "global_vr_percent": float(global_fit["vr"]),
        }
        for fit_row in fit_rows:
            if fit_row["scope"] == "dataset":
                data_name = str(fit_row["dataset"])
                result[f"{data_name}_rms"] = float(fit_row["rms"])
                result[f"{data_name}_vr_percent"] = float(fit_row["vr"])
        results.append(result)

    # ========================= Results: tables ==========================
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "dip_search.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    # ========================= Results: figures =========================
    dip_values = np.array([result["dip_deg"] for result in results])
    rms_values = np.array([result["global_rms"] for result in results])
    vr_values = np.array([result["global_vr_percent"] for result in results])
    best_dip = dip_values[int(np.argmin(rms_values))]

    fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.0), sharex=True)
    axes[0].plot(dip_values, rms_values, "o-", color="tab:blue")
    axes[0].axvline(best_dip, color="0.5", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Global RMS")
    axes[0].grid(alpha=0.25)

    axes[1].plot(dip_values, vr_values, "o-", color="tab:orange")
    axes[1].axvline(best_dip, color="0.5", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Dip (degree)")
    axes[1].set_ylabel("Global VR (%)")
    axes[1].grid(alpha=0.25)

    figure_file = os.path.join(output_dir, "dip_search.png")
    fig.tight_layout()
    fig.savefig(figure_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Summary
    best = min(results, key=lambda row: row["global_rms"])
    print("\nMinimum-RMS candidate (inspect before adopting):")
    print(
        f"  dip={best['dip_deg']:g} degree, "
        f"RMS={best['global_rms']:.6g}, "
        f"VR={best['global_vr_percent']:.3f}%, "
        f"patches={best['n_patches']}"
    )
    print(f"Results: {csv_file}")
    print(f"Figure : {figure_file}")
