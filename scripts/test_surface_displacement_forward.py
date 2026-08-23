"""Template: compute dense ENU surface displacement from one or more faults.

This is a case-editable template. For a new case, usually only the
"User-editable parameters" block needs changes.

Typical uses:
1. Compute a regular lon/lat box grid from one or more fault slip models.
2. Compute ENU displacement at custom lon/lat points loaded from a text file.
3. Save total and per-fault displacement as HDF5/TXT for plotting or export.
"""

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("CUTDE_USE_BACKEND", "cpp")

from csi import RectangularPatches, TriangularPatches
from eqtools.csiExtend.surface_forward import (
    compute_multifault_surface_displacement,
    save_surface_forward_h5,
    save_surface_forward_txt,
)


# =========================
# User-editable parameters
# =========================

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "surface_forward_output"

# Local projection center shared by the fault objects.
LON0, LAT0 = 87.5, 28.5

# Fault slip models.  type can be "triangular" or "rectangular".
FAULT_FILES = [
    {
        "name": "F1",
        "type": "triangular",
        "path": "output/slip_F1.gmt",
    },
    # {
    #     "name": "F2",
    #     "type": "rectangular",
    #     "path": "output/slip_F2_rect.gmt",
    # },
]

# Observation-point mode:
# - "box": regular lon/lat grid
# - "lonlat_file": custom points from a text file
POINT_MODE = "box"

BOX = [87.2, 87.8, 28.2, 28.8]  # [minlon, maxlon, minlat, maxlat]
NPOINTS = 300

LONLAT_FILE = "points_lonlat.txt"
LON_COLUMN = 0
LAT_COLUMN = 1

NU = 0.25
METHOD = "cutde"
TARGET_MEM_GB = 4.0
MAX_OBS_BATCH = 50000

SAVE_H5 = True
SAVE_TXT = True
INCLUDE_BY_FAULT = True


if __name__ == "__main__":
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

    # --------------------- Choose observation points --------------------- #
    if POINT_MODE == "box":
        sample_kwargs = {"box": BOX, "npoints": NPOINTS}
        print(f"Observation points: regular box grid, npoints={NPOINTS}")
    elif POINT_MODE == "lonlat_file":
        points = np.loadtxt(ROOT / LONLAT_FILE)
        lon = points[:, LON_COLUMN]
        lat = points[:, LAT_COLUMN]
        sample_kwargs = {"lonlat": (lon, lat)}
        print(f"Observation points: {len(lon)} custom lon/lat points")
    else:
        raise ValueError(f"Unsupported POINT_MODE: {POINT_MODE}")

    # --------------------- Compute ENU displacement --------------------- #
    result = compute_multifault_surface_displacement(
        faults,
        nu=NU,
        method=METHOD,
        target_mem_gb=TARGET_MEM_GB,
        max_obs_batch=MAX_OBS_BATCH,
        output_coords="lonlat",
        return_each_fault=INCLUDE_BY_FAULT,
        verbose=False,
        **sample_kwargs,
    )

    total = result.disp_total_enu
    print(
        "Total ENU displacement (m): "
        f"E[{total[:, 0].min():.5g}, {total[:, 0].max():.5g}], "
        f"N[{total[:, 1].min():.5g}, {total[:, 1].max():.5g}], "
        f"U[{total[:, 2].min():.5g}, {total[:, 2].max():.5g}]"
    )

    for fault_name, disp in result.disp_by_fault_enu.items():
        print(
            f"{fault_name} ENU displacement (m): "
            f"E[{disp[:, 0].min():.5g}, {disp[:, 0].max():.5g}], "
            f"N[{disp[:, 1].min():.5g}, {disp[:, 1].max():.5g}], "
            f"U[{disp[:, 2].min():.5g}, {disp[:, 2].max():.5g}]"
        )

    # --------------------- Save outputs --------------------- #
    if SAVE_H5:
        out_h5 = save_surface_forward_h5(
            OUTDIR / "surface_displacement_enu.h5",
            result,
            include_by_fault=INCLUDE_BY_FAULT,
        )
        print(f"Saved: {out_h5}")

    if SAVE_TXT:
        out_txt = save_surface_forward_txt(
            OUTDIR / "surface_displacement_enu.txt",
            result,
            include_by_fault=INCLUDE_BY_FAULT,
        )
        print(f"Saved: {out_txt}")
