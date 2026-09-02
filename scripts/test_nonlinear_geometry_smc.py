"""Template script for the new nonlinear geometry SMC inversion.

This case-editable template follows the legacy
``scripts/test_nonlinear_bayesian.py`` workflow, but uses
``NonlinearGeometrySMCInversion`` and the new ``nonlinear_geometry.yml``
configuration.

Typical commands
----------------
Generate the new YAML template in the current run directory first:

    ecat-generate-nonlinear-geometry -o nonlinear_geometry.yml

Run sampling with MPI:

    mpiexec -n 25 python test_nonlinear_geometry_smc.py -r

Rebuild summaries and plots from an existing HDF5 sample file:

    python test_nonlinear_geometry_smc.py
"""

import argparse
import os

from csi.gps import gps
from csi.insar import insar
from eqtools.csiExtend import NonlinearGeometrySMCInversion
from mpi4py import MPI


if __name__ == "__main__":
    # -------------------------------- Parse Arguments -------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Run or post-process the new nonlinear geometry SMC inversion. "
            "Use mpiexec when running with -r."
        )
    )
    parser.add_argument(
        "-r", "--run", action="store_true",
        help="Run SMC sampling before post-processing.",
    )
    parser.add_argument(
        "-c", "--config", default="nonlinear_geometry.yml",
        help="New nonlinear geometry YAML file.",
    )
    parser.add_argument(
        "-s", "--samples", default="samples_mag_rake_multifaults.h5",
        help="HDF5 sample file to write or read.",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting and model-summary post-processing.",
    )
    parser.add_argument(
        "--no-trends", action="store_true",
        help="Skip the fault-parameter trend figure.",
    )
    parser.add_argument(
        "--diagnose-detail", action="store_true",
        help="Print the detailed convergence table in addition to the summary.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show figures interactively. Saved figures are always written.",
    )
    args = parser.parse_args()

    # -------------------------------- MPI Init --------------------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # -------------------------------- Read Data -------------------------------------
    # Modify these for each case. They are the CSI local projection origin, not
    # inverted model parameters.
    lon0 = 87.5
    lat0 = 28.5
    verbose = rank == 0

    # ------------------------------ Generate GPS Object -----------------------------#
    # gps_file = os.path.join("..", "GPS", "GPS_ENU_CSI.dat")
    # cogps = gps(
    #     name="GPS",
    #     utmzone=None,
    #     ellps="WGS84",
    #     lon0=lon0,
    #     lat0=lat0,
    #     verbose=False,
    # )
    # cogps.read_from_enu(
    #     gps_file,
    #     factor=1.0,
    #     minerr=1.0,
    #     header=1,
    #     checkNaNs=True,
    # )
    # cogps.buildCd(direction="enu")

    # ------------------------------ Generate SAR Object -----------------------------#
    # Replace these paths and reader options with the actual case inputs.
    sar_t012a_file = os.path.join(
        "..", "InSAR", "RawInSAR", "Dingri_2018-12-23_T012A",
        "stdBased", "S1_T012A_ifg",
    )
    sar_t121d_file = os.path.join(
        "..", "InSAR", "RawInSAR", "Dingri_2018-12-23_T121D",
        "stdBased", "S1_T121D_ifg",
    )

    sar_t012a = insar(
        name="T012A", utmzone=None, ellps="WGS84",
        lon0=lon0, lat0=lat0, verbose=False,
    )
    sar_t012a.read_from_varres(sar_t012a_file, triangular=True)
    sar_t012a.err *= 1.0
    sar_t012a.buildDiagCd()

    sar_t121d = insar(
        name="T121D", utmzone=None, ellps="WGS84",
        lon0=lon0, lat0=lat0, verbose=False,
    )
    sar_t121d.read_from_varres(sar_t121d_file, triangular=True)
    sar_t121d.err *= 1.0
    sar_t121d.buildDiagCd()

    # The geodata order must match geodata.verticals, geodata.faults,
    # geodata.polys and geodata.sigmas in nonlinear_geometry.yml.
    geodata = [sar_t012a, sar_t121d]

    # ------------------------------ Set Inversion Object ----------------------------#
    inv = NonlinearGeometrySMCInversion(
        "invrc", lat0=lat0, lon0=lon0,
        config_file=args.config, geodata=geodata, verbose=verbose,
    )
    nchains = inv.nchains
    chain_length = inv.chain_length

    inv.setPriors(bounds=None, initialSample=None, datas=None)
    inv.setLikelihood(datas=None, verticals=None)

    # ------------------------------ Run SMC Inversion -------------------------------#
    if args.run:
        inv.walk(
            nchains=nchains, chain_length=chain_length, comm=comm,
            filename=args.samples, save_every=2, save_at_interval=False,
            covariance_epsilon=1e-9, amh_a=1.0 / 9.0, amh_b=8.0 / 9.0,
            diagnose=True, diagnose_detail=args.diagnose_detail,
        )

    # ------------------------------ Plot and Summarize ------------------------------#
    if not args.no_plot:
        inv.extract_and_plot_bayesian_results(
            rank=rank, filename=args.samples,
            plot_faults=True, plot_sigmas=True, plot_data=True,
            plot_data_corrections=True, save_data=True, show=args.show,
            diagnose=True, diagnose_detail=args.diagnose_detail,
        )

        if rank == 0 and not args.no_trends:
            inv.load_samples_from_h5(args.samples)
            inv.plot_fault_parameter_trends(
                save_path="fault_parameter_trends.png", show=args.show,
            )
