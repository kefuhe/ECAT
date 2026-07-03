"""
Helpers for reusing CSI downsampling grids.

The functions here intentionally stay lightweight. They parse CSI ``.rsp``
files in lon/lat space, then project the cell vertices through the active CSI
data object. This avoids relying on the original ``.rsp`` x/y columns, which
may have been generated with a different projection origin. Supported inputs
are legacy 10-column rectangles, 18-column full-corner rectangles, and
8-column triangles.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


RSP_GEOMETRY_CHOICES = ("auto", "rectangle", "triangle")


@dataclass(frozen=True)
class RspGridTemplate:
    """Parsed CSI downsampling grid in geographic coordinates."""

    path: Path
    geometry: str
    blocksll: list
    source_columns: int

    @property
    def cell_count(self):
        return len(self.blocksll)


def normalize_rsp_geometry(geometry):
    key = str(geometry or "auto").replace("-", "_").lower()
    aliases = {
        "rect": "rectangle",
        "rectangular": "rectangle",
        "std": "rectangle",
        "tri": "triangle",
        "triangular": "triangle",
        "trirb": "triangle",
    }
    key = aliases.get(key, key)
    if key not in RSP_GEOMETRY_CHOICES:
        raise ValueError(
            "rsp geometry must be one of "
            f"{RSP_GEOMETRY_CHOICES}; got {geometry!r}."
        )
    return key


def resolve_rsp_path(path):
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if candidate.suffix.lower() != ".rsp":
        with_suffix = candidate.with_suffix(candidate.suffix + ".rsp") if candidate.suffix else candidate.with_suffix(".rsp")
        if with_suffix.exists():
            return with_suffix
    raise FileNotFoundError(f"Cannot find rsp grid file: {path!r}.")


def _numeric_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            try:
                rows.append([float(part) for part in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric grid rows found in rsp file: {path}.")
    return rows


def _infer_row_geometry(row, requested):
    column_count = len(row)
    if column_count in (10, 18):
        inferred = "rectangle"
    elif column_count == 8:
        inferred = "triangle"
    else:
        raise ValueError(
            "Unsupported rsp row format: expected 10 columns for legacy "
            "rectangles, 18 columns for full-corner rectangles, "
            f"or 8 columns for triangles, got {column_count}."
        )

    if requested != "auto" and requested != inferred:
        raise ValueError(
            f"rsp geometry={requested!r} does not match {column_count}-column "
            f"rows inferred as {inferred!r}."
        )
    return inferred


def _legacy_rectangle_vertices(row):
    ul_lon, ul_lat = row[6], row[7]
    lr_lon, lr_lat = row[8], row[9]
    return [
        [ul_lon, ul_lat],
        [lr_lon, ul_lat],
        [lr_lon, lr_lat],
        [ul_lon, lr_lat],
    ]


def _full_rectangle_vertices(row):
    return [
        [row[10], row[11]],
        [row[12], row[13]],
        [row[14], row[15]],
        [row[16], row[17]],
    ]


def _triangle_vertices(row):
    return [
        [row[2], row[3]],
        [row[4], row[5]],
        [row[6], row[7]],
    ]


def read_rsp_grid_template(path, geometry="auto"):
    """Read a CSI ``.rsp`` file as a reusable lon/lat grid template."""

    requested = normalize_rsp_geometry(geometry)
    rsp_path = resolve_rsp_path(path)
    rows = _numeric_rows(rsp_path)
    inferred = _infer_row_geometry(rows[0], requested)
    expected_columns = len(rows[0])

    blocksll = []
    for row_index, row in enumerate(rows, start=1):
        if len(row) != expected_columns:
            raise ValueError(
                f"Inconsistent rsp row length at data row {row_index}: "
                f"expected {expected_columns}, got {len(row)}."
            )
        row_geometry = _infer_row_geometry(row, inferred)
        if row_geometry == "rectangle":
            if expected_columns == 18:
                blocksll.append(_full_rectangle_vertices(row))
            else:
                blocksll.append(_legacy_rectangle_vertices(row))
        else:
            blocksll.append(_triangle_vertices(row))

    return RspGridTemplate(
        path=rsp_path,
        geometry=inferred,
        blocksll=blocksll,
        source_columns=expected_columns,
    )


def _project_vertices(sampler, vertices):
    lon = np.asarray([vertex[0] for vertex in vertices], dtype=float)
    lat = np.asarray([vertex[1] for vertex in vertices], dtype=float)
    x, y = sampler.ll2xy(lon, lat)
    return np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float))).tolist()


def apply_rsp_grid_template(sampler, template, *, tolerance=0.0):
    """
    Apply a parsed ``.rsp`` template to a CSI downsampler object.

    Parameters
    ----------
    sampler : CSI imagedownsampling object
        Must provide ``ll2xy`` and expose ``blocks``/``blocksll`` attributes.
    template : RspGridTemplate
        Parsed template returned by :func:`read_rsp_grid_template`.
    tolerance : float, default 0.0
        Minimum valid-pixel fraction used by CSI ``downsample()``.
    """

    if not isinstance(template, RspGridTemplate):
        template = read_rsp_grid_template(template)

    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError("from_rsp tolerance must be non-negative.")

    sampler.blocksll = [[list(vertex) for vertex in block] for block in template.blocksll]
    sampler.blocks = [_project_vertices(sampler, vertices) for vertices in template.blocksll]
    sampler.tolerance = tolerance
    return template
