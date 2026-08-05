"""Pure, read-only access to paired CSI varres result files.

This module owns the ``<prefix>.txt`` plus ``<prefix>.rsp`` file contract used
by ECAT downsampling. It does not construct CSI data objects, choose a
projection, read covariance matrices, or modify source files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from .grid_template import RspGridTemplate, read_rsp_grid_template


VARRES_DATA_TYPES = ("sar", "optical")


@dataclass(frozen=True)
class CsiVarresResult:
    """Detached values and cell geometry from a CSI varres result.

    Parameters
    ----------
    prefix : pathlib.Path
        Common path prefix without ``.txt`` or ``.rsp``.
    data_type : {"sar", "optical"}
        Scientific row contract used by the text file.
    geometry : {"rectangle", "triangle"}
        Cell geometry parsed from the RSP file.
    row_ids, indices : numpy.ndarray
        Result row identifiers and CSI x/y pixel indices.
    longitude, latitude : numpy.ndarray
        Cell-center coordinates in degrees.
    components, errors : mapping of str to numpy.ndarray
        Stored observations and their reported errors.
    vertices : tuple of numpy.ndarray
        Geographic cell vertices in the same order as the text rows.
    projection : numpy.ndarray or None
        ENU projection vectors for SAR, shape ``(n, 3)``.
    weights : numpy.ndarray or None
        SAR downsampling weights.
    metadata : mapping
        Format diagnostics that do not change scientific values.
    """

    prefix: Path
    data_type: str
    geometry: str
    row_ids: np.ndarray
    indices: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    components: Mapping[str, np.ndarray]
    errors: Mapping[str, np.ndarray]
    vertices: tuple
    projection: np.ndarray | None = None
    weights: np.ndarray | None = None
    metadata: Mapping = field(default_factory=dict)

    @property
    def cell_count(self):
        """Number of downsampled cells."""

        return int(self.longitude.size)

    @property
    def available_components(self):
        """Stored and well-defined display components."""

        names = list(self.components)
        if self.data_type == "optical":
            names.append("magnitude")
        return tuple(names)

    def component(self, name):
        """Return one stored or explicitly derived observation component.

        ``magnitude`` is defined only for optical results as
        ``hypot(east, north)``. No error-propagation assumption is made for
        that derived display value.
        """

        key = str(name).strip().lower()
        if key in self.components:
            return np.asarray(self.components[key])
        if key == "magnitude" and self.data_type == "optical":
            return np.hypot(
                np.asarray(self.components["east"]),
                np.asarray(self.components["north"]),
            )
        raise KeyError(
            f"Unknown varres component {name!r}; available components are "
            f"{self.available_components}."
        )


def _resolve_prefix(path):
    text = str(path)
    lower = text.lower()
    if lower.endswith(".txt") or lower.endswith(".rsp"):
        text = text[:-4]
    prefix = Path(text)
    txt_path = Path(f"{prefix}.txt")
    rsp_path = Path(f"{prefix}.rsp")
    if not txt_path.is_file():
        raise FileNotFoundError(f"Cannot find CSI varres text file: {txt_path}.")
    if not rsp_path.is_file():
        raise FileNotFoundError(f"Cannot find CSI varres RSP file: {rsp_path}.")
    return prefix, txt_path, rsp_path


def _numeric_rows(path):
    rows = []
    started = False
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            try:
                row = [float(part) for part in parts]
            except ValueError as exc:
                if started:
                    raise ValueError(
                        f"Malformed numeric row in {path} at line "
                        f"{line_number}."
                    ) from exc
                continue
            rows.append(row)
            started = True
    if not rows:
        raise ValueError(f"No numeric varres rows found in {path}.")
    width = len(rows[0])
    for index, row in enumerate(rows, start=1):
        if len(row) != width:
            raise ValueError(
                f"Inconsistent varres row width at data row {index}: "
                f"expected {width}, got {len(row)}."
            )
    return np.asarray(rows, dtype=float)


def _integer_columns(values, label):
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1.0e-9):
        raise ValueError(f"{label} must contain integer values.")
    return rounded.astype(int)


def read_csi_varres_result(prefix, *, data_type="sar", geometry="auto"):
    """Read paired CSI varres values and geographic cell geometry.

    Parameters
    ----------
    prefix : path-like
        Common prefix or either member of the ``.txt/.rsp`` pair.
    data_type : {"sar", "optical"}, default "sar"
        Text-row schema. SAR rows contain one scalar observation and ENU
        projection; optical rows contain east and north observations.
    geometry : {"auto", "rectangle", "triangle"}, default "auto"
        Optional RSP geometry assertion.

    Returns
    -------
    CsiVarresResult
        Fully detached arrays in source row order.

    Raises
    ------
    FileNotFoundError
        If either paired file is missing.
    ValueError
        If row width, row order, indices, coordinates, or geometry are
        inconsistent.

    Notes
    -----
    ``.cov`` is intentionally not read because covariance is not part of the
    Google Earth display contract.
    """

    data_type = str(data_type).strip().lower().replace("-", "_")
    aliases = {"insar": "sar", "opticorr": "optical"}
    data_type = aliases.get(data_type, data_type)
    if data_type not in VARRES_DATA_TYPES:
        raise ValueError(
            f"data_type must be one of {VARRES_DATA_TYPES}; got "
            f"{data_type!r}."
        )

    resolved_prefix, txt_path, rsp_path = _resolve_prefix(prefix)
    rows = _numeric_rows(txt_path)
    expected_width = 11 if data_type == "sar" else 7
    if rows.shape[1] != expected_width:
        raise ValueError(
            f"CSI {data_type} varres rows require {expected_width} columns; "
            f"got {rows.shape[1]}. Check data_type."
        )
    template: RspGridTemplate = read_rsp_grid_template(
        rsp_path,
        geometry=geometry,
    )
    if template.cell_count != rows.shape[0]:
        raise ValueError(
            "CSI varres .txt/.rsp row counts differ: "
            f"{rows.shape[0]} versus {template.cell_count}."
        )

    row_ids = _integer_columns(rows[:, 0], "varres row ids")
    if np.unique(row_ids).size != row_ids.size:
        raise ValueError("CSI varres row ids must be unique.")
    if data_type == "sar":
        indices = _integer_columns(rows[:, 1:3], "SAR xind/yind")
        longitude = rows[:, 3]
        latitude = rows[:, 4]
        components = {"observation": rows[:, 5].copy()}
        errors = {"observation": rows[:, 6].copy()}
        weights = rows[:, 7].copy()
        projection = rows[:, 8:11].copy()
    else:
        indices = np.asarray(template.indices, dtype=int)
        longitude = rows[:, 1]
        latitude = rows[:, 2]
        components = {
            "east": rows[:, 3].copy(),
            "north": rows[:, 4].copy(),
        }
        errors = {
            "east": rows[:, 5].copy(),
            "north": rows[:, 6].copy(),
        }
        weights = None
        projection = None

    rsp_indices = np.asarray(template.indices, dtype=int)
    if indices.shape != rsp_indices.shape or not np.array_equal(
        indices,
        rsp_indices,
    ):
        raise ValueError(
            "CSI varres .txt xind/yind columns are not aligned with .rsp "
            "row order."
        )
    if not np.all(np.isfinite(longitude)) or not np.all(np.isfinite(latitude)):
        raise ValueError("CSI varres center longitude/latitude must be finite.")
    if np.any((longitude < -360.0) | (longitude > 360.0)):
        raise ValueError("CSI varres longitude is outside [-360, 360].")
    if np.any((latitude < -90.0) | (latitude > 90.0)):
        raise ValueError("CSI varres latitude is outside [-90, 90].")

    vertices = tuple(
        np.asarray(cell, dtype=float).copy() for cell in template.blocksll
    )
    if any(
        vertex.ndim != 2
        or vertex.shape[1] != 2
        or not np.all(np.isfinite(vertex))
        for vertex in vertices
    ):
        raise ValueError("CSI varres cell vertices must be finite lon/lat pairs.")

    return CsiVarresResult(
        prefix=resolved_prefix,
        data_type=data_type,
        geometry=template.geometry,
        row_ids=row_ids,
        indices=indices,
        longitude=np.asarray(longitude, dtype=float).copy(),
        latitude=np.asarray(latitude, dtype=float).copy(),
        components={
            name: np.asarray(values, dtype=float).copy()
            for name, values in components.items()
        },
        errors={
            name: np.asarray(values, dtype=float).copy()
            for name, values in errors.items()
        },
        vertices=vertices,
        projection=projection,
        weights=weights,
        metadata={
            "txt_columns": int(rows.shape[1]),
            "rsp_columns": int(template.source_columns),
            "covariance_read": False,
        },
    )


__all__ = [
    "CsiVarresResult",
    "VARRES_DATA_TYPES",
    "read_csi_varres_result",
]
