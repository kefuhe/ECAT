"""Template: project multi-fault ENU forward displacement to SAR LOS.

This script demonstrates the common SAR pattern:
1. Read one or more fault slip models.
2. Read a SAR product and use its valid pixels as computation points.
3. Compute ENU displacement with ``compute_surface_displacement``.
4. Project ENU to LOS using ``scalar_observation = ENU dot projection``.
5. Save per-fault and total LOS fields as GeoTIFF and PNG quicklook files.

Default execution only redraws existing GeoTIFF outputs in ``OUTDIR``:

    python test_sar_los_surface_forward.py

Run the forward calculation and then plot:

    python test_sar_los_surface_forward.py -r

The SAR reader block below uses ``Hyp3TiffReader`` as an example.  For other
SAR products, replace only that reader block as long as the resulting object
provides ``lon``, ``lat``, ``los``, and flat valid-pixel indices.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUTDE_USE_BACKEND", "cpp")

from eqtools.viztools import plot_geotiff


# =========================
# User-editable parameters
# =========================

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "surface_forward_sar_los"

LON0, LAT0 = 78.63, 41.141

NU = 0.25
METHOD = "cutde"
TARGET_MEM_GB = 1.0
MAX_OBS_BATCH = 20000

FAULT_FILES = [
    {
        "name": "f1",
        "type": "triangular",
        "path": "slip/slip_20240130aftershock_f1.gmt",
    },
    {
        "name": "f2",
        "type": "triangular",
        "path": "slip/slip_20240130aftershock_f2.gmt",
    },
]

# Example HyP3 images.  Replace these paths for your own case.
HYP3_IMAGES = {
    "as": {
        "phase": "20240130/as/unw_phase_mask_orb_demcorr.tif",
        "phi": "20240130/as/lv_phi.tif",
        "theta": "20240130/as/lv_theta.tif",
    },
    "de": {
        "phase": "20240130/de/unw_phase_orb_demcorr.tif",
        "phi": "20240130/de/lv_phi.tif",
        "theta": "20240130/de/lv_theta.tif",
    },
}

PNG_DPI = 180
PLOT_EXISTING_PATTERN = "*_hyp3_los_m.tif"
PLOT_CMAP = "RdBu_r"
PLOT_PERCENTILE = 99.0
PLOT_SYMMETRIC = True
PLOT_FIGSIZE = (8.0, 6.0)
PLOT_COLORBAR_LABEL = "LOS displacement (m)"
PLOT_AXIS = "geo"  # use "off" for clean image-only quicklooks
PLOT_AXIS_MAX_MAJOR_TICKS = 5
PLOT_COLORBAR_MAX_MAJOR_TICKS = 4
PLOT_TICK_FONTSIZE = 8
PLOT_LABEL_FONTSIZE = 9

# "auto" tries a regular lon/lat GeoTIFF from sar.raw_mesh_lon/raw_mesh_lat
# first, then falls back to copying the reference raster metadata.
# Use "reference" to force exact reference-raster georeferencing inheritance,
# or "lonlat_regular" to require a regular lon/lat GeoTIFF.
SAVE_GEOREFERENCE_MODE = "auto"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot existing SAR LOS forward GeoTIFFs by default. Use -r to "
            "rerun the forward calculation and regenerate GeoTIFFs first."
        )
    )
    parser.add_argument(
        "-r",
        "--run-forward",
        action="store_true",
        help="Run forward modeling before plotting. Default only plots existing GeoTIFFs.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="With -r, only generate GeoTIFFs and skip PNG quicklook plotting.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving PNG quicklooks.",
    )
    parser.add_argument(
        "--pattern",
        default=PLOT_EXISTING_PATTERN,
        help=(
            "GeoTIFF glob pattern used in default plotting mode "
            f"(default: {PLOT_EXISTING_PATTERN!r})."
        ),
    )
    return parser.parse_args()


def plot_los_geotiff(tif_path, *, show=False):
    fig, ax, _ = plot_geotiff(
        tif_path,
        cmap=PLOT_CMAP,
        symmetric=PLOT_SYMMETRIC,
        percentile=PLOT_PERCENTILE,
        colorbar_label=PLOT_COLORBAR_LABEL,
        title=Path(tif_path).stem,
        save=Path(tif_path).with_suffix(".png"),
        show=show,
        axis=PLOT_AXIS,
        axis_max_major_ticks=PLOT_AXIS_MAX_MAJOR_TICKS,
        colorbar_max_major_ticks=PLOT_COLORBAR_MAX_MAJOR_TICKS,
        tickfontsize=PLOT_TICK_FONTSIZE,
        labelfontsize=PLOT_LABEL_FONTSIZE,
        figsize=PLOT_FIGSIZE,
        dpi=PNG_DPI,
        close=True,
    )
    return fig, ax


def plot_existing_geotiffs(*, pattern=PLOT_EXISTING_PATTERN, show=False):
    tif_files = sorted(OUTDIR.glob(pattern))
    if not tif_files:
        print(
            f"No GeoTIFF files found in {OUTDIR} matching {pattern!r}. "
            "Run with -r to generate forward-model outputs first."
        )
        return 0

    print(f"Plotting {len(tif_files)} existing GeoTIFF file(s) from {OUTDIR}")
    for tif_file in tif_files:
        plot_los_geotiff(tif_file, show=show)
    return len(tif_files)


def run_forward_and_save(*, plot=True, show=False):
    import numpy as np
    from csi import RectangularPatches, TriangularPatches
    from eqtools.csiExtend.sarUtils.readTiff2csisar import Hyp3TiffReader
    from eqtools.csiExtend.surface_forward import (
        compute_multifault_surface_displacement,
        project_enu_to_los,
        save_lonlat_regular_geotiff,
        save_raster_like_geotiff,
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # --------------------- Read fault models --------------------- #
    faults = {}
    for info in FAULT_FILES:
        fault_type = info["type"].lower()
        fault_path = ROOT / info["path"]

        if fault_type == "triangular":
            fault = TriangularPatches(info["name"], lon0=LON0, lat0=LAT0, verbose=False)
            fault.readPatchesFromFile(str(fault_path), gmtslip=True)
        elif fault_type == "rectangular":
            fault = RectangularPatches(info["name"], lon0=LON0, lat0=LAT0, verbose=False)
            fault.readPatchesFromFile(str(fault_path), readpatchindex=True)
        else:
            raise ValueError(f"Unsupported fault type: {info['type']}")

        faults[info["name"]] = fault
        print(f"Read fault {info['name']}: {fault_path}, patches={fault.numpatch}")

    # --------------------- Process each SAR image --------------------- #
    for image_name, files in HYP3_IMAGES.items():
        print(f"\nImage {image_name}")

        # Reader-specific block.  Replace this block when using another SAR
        # product, but keep the required outputs: lon, lat, los, valid index.
        sar = Hyp3TiffReader(
            image_name,
            lon0=LON0,
            lat0=LAT0,
            directory_name=str(ROOT),
            mode="unwrapped_phase",
        )
        sar.extract_raw_grd(
            phsname=files["phase"],
            azifile=files["phi"],
            incfile=files["theta"],
            zero2nan=False,
        )
        sar.read_observation(zero2nan=False)
        print(f"  valid pixels: {sar.x.size}")

        # Compute ENU displacement at valid SAR pixels.
        result = compute_multifault_surface_displacement(
            faults,
            lonlat=(sar.lon, sar.lat),
            nu=NU,
            method=METHOD,
            target_mem_gb=TARGET_MEM_GB,
            max_obs_batch=MAX_OBS_BATCH,
            output_coords="xy",
            return_each_fault=True,
            verbose=False,
        )

        los_outputs = {}
        for fault_name, disp_enu in result.disp_by_fault_enu.items():
            los_outputs[fault_name] = project_enu_to_los(disp_enu, sar)
        los_outputs["total"] = project_enu_to_los(result.disp_total_enu, sar)

        for name, los in los_outputs.items():
            out_tif = OUTDIR / f"{image_name}_{name}_hyp3_los_m.tif"
            saved_mode = _save_sar_los_geotiff(
                out_tif,
                los,
                sar=sar,
                reference_raster=ROOT / files["phase"],
                mode=SAVE_GEOREFERENCE_MODE,
                save_lonlat_regular_geotiff=save_lonlat_regular_geotiff,
                save_raster_like_geotiff=save_raster_like_geotiff,
            )

            if plot:
                plot_los_geotiff(out_tif, show=show)

            print(
                f"  {name}: georef={saved_mode}; min/mean/max = "
                f"{np.nanmin(los):.5g} / {np.nanmean(los):.5g} / "
                f"{np.nanmax(los):.5g} m"
            )


def _save_sar_los_geotiff(
    out_tif,
    los,
    *,
    sar,
    reference_raster,
    mode,
    save_lonlat_regular_geotiff,
    save_raster_like_geotiff,
):
    mode = mode.lower()
    if mode not in {"auto", "reference", "lonlat_regular"}:
        raise ValueError(
            "SAVE_GEOREFERENCE_MODE must be 'auto', 'reference', or 'lonlat_regular'."
        )

    if mode in {"auto", "lonlat_regular"}:
        try:
            save_lonlat_regular_geotiff(
                out_tif,
                los,
                sar.raw_mesh_lon,
                sar.raw_mesh_lat,
                valid_index=sar.projection_raw_valid_index,
            )
            return "lonlat_regular"
        except ValueError as exc:
            if mode == "lonlat_regular":
                raise
            print(
                "  lonlat_regular GeoTIFF skipped: "
                f"{exc} Falling back to reference raster metadata."
            )

    save_raster_like_geotiff(
        out_tif,
        los,
        reference_raster,
        valid_index=sar.projection_raw_valid_index,
    )
    return "reference"


if __name__ == "__main__":
    args = parse_args()
    if args.no_plot and not args.run_forward:
        raise SystemExit("--no-plot is only useful with -r/--run-forward.")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.run_forward:
        run_forward_and_save(plot=not args.no_plot, show=args.show)
    else:
        plot_existing_geotiffs(pattern=args.pattern, show=args.show)
