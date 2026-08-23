"""Canonical covariance metrics shared by ECAT inversion solvers.

The covariance matrix ``C`` is the only statistical source of truth.  Every
metric implements the left-whitening action ``W`` with
``W.T @ W = C**-1`` without forming the precision matrix.  Exact identity and
diagonal covariances retain scalar/vector factors; a general SPD covariance
stores the dense ``W = L**-1`` prepared from ``C = L @ L.T``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.linalg import solve_triangular


def _validated_symmetric_matrix(matrix, *, name):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")

    if np.array_equal(matrix, matrix.T):
        return matrix

    # Judge symmetry relative to the whole matrix scale. This accepts harmless
    # roundoff while still rejecting a statistically different metric.
    scale = max(
        abs(float(np.min(matrix))),
        abs(float(np.max(matrix))),
        np.finfo(float).tiny,
    )
    skew = float(np.max(np.abs(matrix - matrix.T)))
    if skew > 1.0e-12 + 1.0e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * (matrix + matrix.T)


@dataclass(frozen=True)
class DataCovarianceMetric:
    """Prepared metric for one statistically independent data set.

    ``marginal_rms_std`` is the root-mean-square marginal standard deviation,
    ``sqrt(mean(diag(C)))``.  It is descriptive metadata for reports only;
    whitening and likelihood calculations continue to use the exact
    covariance through :meth:`whiten`.  ``kind`` selects an internal
    identity, diagonal, or dense representation; it is not a user-visible
    approximation mode.
    """

    _factor: np.ndarray | float
    logdet: float
    name: str = "data"
    marginal_rms_std: float | None = None
    kind: str = "dense"
    _size: int | None = None

    @property
    def size(self):
        if self._size is not None:
            return self._size
        return np.asarray(self._factor).shape[0]

    @property
    def whitener(self):
        """Return a dense whitener compatibility view.

        Solver code should call :meth:`whiten` so identity and diagonal
        metrics retain their linear-time action.  This property remains for
        diagnostics and older callers that explicitly require the matrix.
        """
        if self.kind == "dense":
            return self._factor
        if self.kind == "identity":
            return float(self._factor) * np.eye(self.size)
        if self.kind == "diagonal":
            return np.diag(np.asarray(self._factor, dtype=float))
        raise RuntimeError(f"unknown covariance metric kind {self.kind!r}")

    def whiten(self, values):
        """Whiten a vector or row-aligned matrix with this data metric."""
        values = np.asarray(values, dtype=float)
        if values.ndim not in (1, 2) or values.shape[0] != self.size:
            raise ValueError(
                f"{self.name}: expected {self.size} data rows, got "
                f"shape {values.shape}"
            )
        if self.kind == "dense":
            return self._factor @ values
        if self.kind == "identity":
            return float(self._factor) * values
        if self.kind == "diagonal":
            inverse_std = np.asarray(self._factor, dtype=float)
            return (
                inverse_std * values
                if values.ndim == 1
                else inverse_std[:, None] * values
            )
        raise RuntimeError(f"unknown covariance metric kind {self.kind!r}")

    def __matmul__(self, values):
        """Apply the metric like the historical dense whitener matrix."""
        return self.whiten(values)

    def quadratic_form(self, residual):
        """Return ``residual.T @ C**-1 @ residual`` without forming ``C**-1``."""
        whitened = self.whiten(residual)
        return float(np.dot(whitened, whitened))


def prepare_covariance_metric(covariance, *, name="data covariance"):
    """Validate one SPD covariance and return its reusable metric.

    Exact identity/diagonal structure is preserved as scalar or row factors.
    Otherwise the factorization follows ``C = L @ L.T``,
    ``W = solve(L, I)`` and ``W.T @ W = C**-1``. Preparation belongs outside
    solver iterations and Bayesian candidate loops.
    """
    covariance = _validated_symmetric_matrix(covariance, name=name)
    diagonal = np.diag(covariance).copy()
    is_diagonal = True
    # Scan rows without allocating another n-by-n mask or diagonal matrix.
    for index, row in enumerate(covariance):
        if np.any(row[:index] != 0.0) or np.any(row[index + 1:] != 0.0):
            is_diagonal = False
            break

    marginal_rms_std = float(np.sqrt(np.mean(diagonal)))
    if is_diagonal:
        if np.any(diagonal <= 0.0):
            raise ValueError(f"{name} must be positive definite")
        logdet = float(np.sum(np.log(diagonal)))
        if np.all(diagonal == diagonal[0]):
            return DataCovarianceMetric(
                _factor=float(1.0 / np.sqrt(diagonal[0])),
                logdet=logdet,
                name=name,
                marginal_rms_std=marginal_rms_std,
                kind="identity",
                _size=covariance.shape[0],
            )
        return DataCovarianceMetric(
            _factor=1.0 / np.sqrt(diagonal),
            logdet=logdet,
            name=name,
            marginal_rms_std=marginal_rms_std,
            kind="diagonal",
            _size=covariance.shape[0],
        )

    try:
        lower = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc

    identity = np.eye(covariance.shape[0], dtype=float)
    whitener = solve_triangular(
        lower,
        identity,
        lower=True,
        check_finite=False,
        overwrite_b=True,
    )
    logdet = float(2.0 * np.sum(np.log(np.diag(lower))))
    return DataCovarianceMetric(
        _factor=whitener,
        logdet=logdet,
        name=name,
        marginal_rms_std=marginal_rms_std,
        kind="dense",
        _size=covariance.shape[0],
    )


def gaussian_log_likelihood(
        residual, covariance_metric, sigma, *, include_base_logdet=True):
    """Return the Gaussian log likelihood under ``sigma**2 * C``.

    Fixed constants such as ``n * log(2*pi)`` remain omitted.  Some legacy
    joint solvers also omit the sample-independent ``log|C|`` term; callers
    preserve that convention with ``include_base_logdet=False``.
    """
    residual = np.asarray(residual, dtype=float).reshape(-1)
    sigma = float(sigma)
    if residual.size != covariance_metric.size:
        raise ValueError(
            f"{covariance_metric.name}: residual length {residual.size} does "
            f"not match covariance size {covariance_metric.size}"
        )
    if not np.isfinite(sigma) or sigma <= 0.0:
        return -np.inf
    value = covariance_metric.quadratic_form(residual) / sigma**2
    value += residual.size * np.log(sigma**2)
    if include_base_logdet:
        value += covariance_metric.logdet
    return -0.5 * float(value)


def prepare_block_covariance_metrics(covariance, data_ranges: Mapping[str, tuple]):
    """Prepare independent metrics from a block-diagonal covariance.

    ``data_ranges`` must partition every covariance row exactly once. ECAT's
    multi-data solvers assign one variance component per named data set, so a
    nonzero cross-data covariance cannot be represented by that model and is
    rejected instead of being silently discarded.
    """
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "data covariance must be a square two-dimensional matrix"
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError("data covariance must contain only finite values")
    n_data = covariance.shape[0]
    ordered = list(data_ranges.items())
    cursor = 0
    for data_name, (start, end) in ordered:
        if start != cursor or end <= start or end > n_data:
            raise ValueError(
                "data_ranges must be contiguous, non-empty, and cover the "
                f"covariance in order; invalid range for '{data_name}'"
            )
        cursor = end
    if cursor != n_data:
        raise ValueError(
            f"data_ranges cover {cursor} rows but covariance has {n_data}"
        )

    scale = max(
        abs(float(np.min(covariance))),
        abs(float(np.max(covariance))),
        np.finfo(float).tiny,
    )
    cross_tolerance = 1.0e-12 + 1.0e-10 * scale
    for i, (left_name, (left_start, left_end)) in enumerate(ordered):
        for right_name, (right_start, right_end) in ordered[i + 1:]:
            cross_block = covariance[left_start:left_end, right_start:right_end]
            reverse_block = covariance[right_start:right_end, left_start:left_end]
            if (
                np.any(np.abs(cross_block) > cross_tolerance)
                or np.any(np.abs(reverse_block) > cross_tolerance)
            ):
                raise ValueError(
                    "data covariance contains cross-data terms between "
                    f"'{left_name}' and '{right_name}', but the current "
                    "variance-component model requires independent blocks"
                )

    return {
        data_name: prepare_covariance_metric(
            covariance[start:end, start:end],
            name=f"covariance for data set '{data_name}'",
        )
        for data_name, (start, end) in ordered
    }


def whiten_data_blocks(data_metrics, data_ranges, G, d):
    """Validate layout and whiten each data block once.

    Returns a mapping ``name -> (W_k G_k, W_k d_k)``. VCE iterations may
    rescale these arrays by variance components without refactorizing or
    remultiplying the base covariance metric.
    """
    if set(data_metrics) != set(data_ranges):
        missing = sorted(set(data_ranges) - set(data_metrics))
        extra = sorted(set(data_metrics) - set(data_ranges))
        raise ValueError(
            "data metric names do not match data_ranges: "
            f"missing={missing}, extra={extra}"
        )
    G = np.asarray(G, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)
    if G.ndim != 2 or G.shape[0] != d.size:
        raise ValueError("G and d must have the same number of data rows")

    blocks = {}
    cursor = 0
    for data_name, (start, end) in data_ranges.items():
        if start != cursor or end <= start or end > d.size:
            raise ValueError(
                "data_ranges must be contiguous, non-empty, and ordered; "
                f"invalid range for '{data_name}': {(start, end)}"
            )
        metric = data_metrics[data_name]
        if metric.size != end - start:
            raise ValueError(
                f"{data_name}: metric size {metric.size} does not match "
                f"data range length {end - start}"
            )
        blocks[data_name] = (
            metric.whiten(G[start:end, :]),
            metric.whiten(d[start:end]),
        )
        cursor = end
    if cursor != d.size:
        raise ValueError(f"data_ranges cover {cursor} rows but G/d have {d.size}")
    return blocks
