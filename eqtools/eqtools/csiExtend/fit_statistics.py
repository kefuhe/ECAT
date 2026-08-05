"""Shared RMS/VR helpers for ECAT inversion diagnostics.

This module keeps the numerical definitions used by existing BLSE/Bayesian
reporting in one place.  It is intentionally read-only: helpers inspect
observed and synthetic vectors or solver matrices, but never rebuild Green's
functions or alter model parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def data_fit_vectors(data: Any, vertical: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return observed and synthetic vectors using existing CSI data fields.

    The vectorization matches the legacy ``calculate_data_fit_metrics`` paths:
    GPS can include ENU or only horizontal EN components; scalar datasets use
    their native flat arrays; optical offsets concatenate east and north.
    """
    dtype = getattr(data, "dtype", None)
    if dtype == "insar":
        observed = data.vel
        synthetic = data.synth
    elif dtype == "gps":
        if vertical:
            observed = data.vel_enu.flatten()
            synthetic = data.synth.flatten()
        else:
            observed = data.vel_enu[:, :-1].flatten()
            synthetic = data.synth[:, :-1].flatten()
    elif dtype in ("opticorr", "optical"):
        observed = np.hstack((data.east, data.north))
        synthetic = np.hstack((data.east_synth, data.north_synth))
    elif dtype == "leveling":
        observed = data.vel
        synthetic = data.synth
    elif dtype == "crossfaultoffset":
        observed = data.data_vector
        synthetic = data.synth_vector
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

    return np.asarray(observed, dtype=float).reshape(-1), np.asarray(synthetic, dtype=float).reshape(-1)


def fit_metrics_from_vectors(observed: Any, synthetic: Any) -> dict[str, float | int]:
    """Compute RMS and VR with the established ECAT formula."""
    observed = np.asarray(observed, dtype=float).reshape(-1)
    synthetic = np.asarray(synthetic, dtype=float).reshape(-1)
    if observed.shape != synthetic.shape:
        raise ValueError(f"Observed/synthetic shape mismatch: {observed.shape} != {synthetic.shape}")
    residuals = synthetic - observed
    ss_res = float(np.sum(residuals ** 2))
    ss_obs = float(np.sum(observed ** 2))
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    vr = float((1.0 - ss_res / ss_obs) * 100.0) if ss_obs != 0.0 else 0.0
    return {
        "rms": rms,
        "vr": vr,
        "ss_res": ss_res,
        "ss_obs": ss_obs,
        "n_observations": int(residuals.size),
    }


def solver_fit_metrics(G: Any, m: Any, d: Any) -> dict[str, float | int]:
    """Compute RMS/VR from a solver matrix and vector, matching BLSE output."""
    m_array = np.asarray(m, dtype=float).reshape(-1)
    predicted = G.dot(m_array) if hasattr(G, "dot") else np.dot(G, m_array)
    return fit_metrics_from_vectors(d, predicted)


def fit_statistics_rows_to_dataframe(rows: Sequence[Mapping[str, Any]]):
    """Return a pandas DataFrame from fit-statistics rows."""
    import pandas as pd

    return pd.DataFrame([dict(row) for row in rows])


def format_fit_statistics_report(rows: Sequence[Mapping[str, Any]], *, model: str | None = None) -> str:
    """Format fit statistics as a compact human-readable report."""
    title = "Fit statistics report"
    if model is not None:
        title += f" ({str(model).upper()} model)"
    lines = [title, f"  rows: {len(rows)}"]
    for row in rows:
        scope = row.get("scope")
        label = row.get("dataset") if scope == "dataset" else scope
        details = []
        if row.get("data_type") is not None:
            details.append(f"type={row.get('data_type')}")
        if row.get("poly") is not None:
            details.append(f"poly={row.get('poly')}")
        details.append(f"n={row.get('n_observations')}")
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(
            f"  - {label}{suffix}: RMS={float(row['rms']):.6g}, VR={float(row['vr']):.4g}%"
        )
    return "\n".join(lines) + "\n"


def write_fit_statistics_report_files(
    rows: Sequence[Mapping[str, Any]],
    outdir,
    *,
    basename: str = "fit_statistics",
    formats: Sequence[str] = ("txt", "tsv"),
    model: str | None = None,
) -> dict[str, Path]:
    """Write fit-statistics text/table files and return paths by format."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    requested = {str(fmt).lower().lstrip(".") for fmt in formats}
    unsupported = requested - {"txt", "tsv"}
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported fit-statistics report format(s): {names}")

    written: dict[str, Path] = {}
    if "txt" in requested:
        path = outdir / f"{basename}.txt"
        path.write_text(format_fit_statistics_report(rows, model=model), encoding="utf-8")
        written["txt"] = path

    if "tsv" in requested:
        path = outdir / f"{basename}.tsv"
        dataframe = fit_statistics_rows_to_dataframe(rows)
        dataframe.to_csv(path, sep="\t", index=False)
        written["tsv"] = path

    return written
