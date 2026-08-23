"""Reusable quadratic blocks for weighted least-squares problems.

The scientific problem remains a residual least-squares problem.  This module
only prepares the exact CVXOPT objective

``0.5 * x.T @ H @ x + q.T @ x``

with ``H = A.T @ A`` and ``q = -A.T @ b``.  Keeping the residual block beside
its Gram/cross products lets repeated solvers reuse the expensive products and
still reconstruct the original residual system for diagnostics or the robust
Clarabel fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LeastSquaresBlock:
    """One immutable residual block and its exact quadratic contribution.

    ``rhs=None`` represents a zero right-hand side, as used by smoothing
    blocks.  ``gram`` and ``cross`` are prepared once from the unscaled block;
    candidate/iteration variance components only rescale these arrays.
    """

    matrix: np.ndarray
    rhs: np.ndarray | None
    gram: np.ndarray
    cross: np.ndarray | None
    name: str = "block"

    @classmethod
    def prepare(cls, matrix, rhs=None, *, name="block"):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name}: matrix must be a finite 2-D array")

        rhs_array = None
        cross = None
        if rhs is not None:
            rhs_array = np.asarray(rhs, dtype=float).reshape(-1)
            if rhs_array.size != matrix.shape[0]:
                raise ValueError(
                    f"{name}: rhs length {rhs_array.size} does not match "
                    f"matrix rows {matrix.shape[0]}"
                )
            if not np.all(np.isfinite(rhs_array)):
                raise ValueError(f"{name}: rhs must contain only finite values")
            cross = matrix.T @ rhs_array

        return cls(
            matrix=matrix,
            rhs=rhs_array,
            gram=matrix.T @ matrix,
            cross=cross,
            name=name,
        )


def assemble_quadratic_objective(
    weighted_blocks: Iterable[tuple[LeastSquaresBlock, float]],
    *,
    n_parameters: int | None = None,
):
    """Combine prepared blocks using inverse-variance weights.

    For a block ``(A_i, b_i)`` with inverse variance ``w_i``, add
    ``w_i * A_i.T @ A_i`` to ``H`` and ``-w_i * A_i.T @ b_i`` to ``q``.
    The function consumes the iterable once and never builds the augmented
    residual matrix.
    """

    blocks = list(weighted_blocks)
    if not blocks and n_parameters is None:
        raise ValueError("n_parameters is required when no blocks are supplied")

    if n_parameters is None:
        n_parameters = blocks[0][0].matrix.shape[1]
    n_parameters = int(n_parameters)
    if n_parameters < 0:
        raise ValueError("n_parameters must be non-negative")

    hessian = np.zeros((n_parameters, n_parameters), dtype=float)
    linear = np.zeros(n_parameters, dtype=float)
    for block, inverse_variance in blocks:
        if block.matrix.shape[1] != n_parameters:
            raise ValueError(
                f"{block.name}: expected {n_parameters} parameter columns, "
                f"got {block.matrix.shape[1]}"
            )
        inverse_variance = float(inverse_variance)
        if not np.isfinite(inverse_variance) or inverse_variance <= 0.0:
            raise ValueError(
                f"{block.name}: inverse variance must be finite and positive"
            )
        hessian += inverse_variance * block.gram
        if block.cross is not None:
            linear -= inverse_variance * block.cross
    return hessian, linear


def assemble_residual_system(
    weighted_blocks: Iterable[tuple[LeastSquaresBlock, float]],
    *,
    n_parameters: int | None = None,
):
    """Materialize the residual form corresponding to prepared blocks.

    This is intentionally separate from normal execution.  It is used only
    when a residual-form solver or diagnostic genuinely needs ``A`` and ``b``.
    """

    blocks = list(weighted_blocks)
    if not blocks:
        if n_parameters is None:
            raise ValueError("n_parameters is required when no blocks are supplied")
        return np.zeros((0, int(n_parameters))), np.zeros(0)

    matrices = []
    right_sides = []
    for block, inverse_variance in blocks:
        inverse_variance = float(inverse_variance)
        if not np.isfinite(inverse_variance) or inverse_variance <= 0.0:
            raise ValueError(
                f"{block.name}: inverse variance must be finite and positive"
            )
        scale = np.sqrt(inverse_variance)
        matrices.append(block.matrix * scale)
        if block.rhs is None:
            right_sides.append(np.zeros(block.matrix.shape[0], dtype=float))
        else:
            right_sides.append(block.rhs * scale)
    return np.vstack(matrices), np.concatenate(right_sides)


def weighted_residual_quadratic(block, model, inverse_variance):
    """Return ``w * ||A @ model - b||**2`` for one prepared data block."""

    if block.rhs is None:
        residual = block.matrix @ model
    else:
        residual = block.matrix @ model - block.rhs
    return float(inverse_variance) * float(np.dot(residual, residual))


def gaussian_curvature_log_term(hessian, *, name="quadratic Hessian"):
    """Return the Gaussian elimination term ``-0.5 * log(det(H))``.

    Eliminating a full-rank linear parameter vector from an unconstrained
    Gaussian density contributes ``det(H)**(-1/2)`` to the marginal density,
    where ``H`` is the conditional quadratic Hessian.  This function keeps
    that statistical contract separate from optional smoothing blocks: a
    data-only Hessian and a data-plus-smoothing Hessian are treated identically.

    The determinant is evaluated from a Cholesky factor instead of accepting
    the absolute determinant returned by ``slogdet``.  A non-symmetric or
    non-positive-definite matrix has no valid full-dimensional Gaussian
    curvature under this contract and therefore raises a clear error rather
    than producing a finite but scientifically ambiguous score.
    """

    hessian = np.asarray(hessian, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"{name} must be a square two-dimensional matrix")
    if hessian.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one parameter")
    if not np.all(np.isfinite(hessian)):
        raise ValueError(f"{name} must contain only finite values")

    scale = max(1.0, float(np.max(np.abs(hessian))))
    symmetry_error = float(np.max(np.abs(hessian - hessian.T)))
    symmetry_tolerance = (
        100.0 * np.finfo(float).eps * max(1, hessian.shape[0]) * scale
    )
    if symmetry_error > symmetry_tolerance:
        raise ValueError(
            f"{name} must be symmetric; maximum asymmetry "
            f"{symmetry_error:.3e} exceeds {symmetry_tolerance:.3e}"
        )

    # Gram blocks are symmetric by construction.  Averaging removes only
    # round-off-level assembly asymmetry admitted by the check above.
    symmetric_hessian = 0.5 * (hessian + hessian.T)
    try:
        chol = np.linalg.cholesky(symmetric_hessian)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"{name} must be positive definite for Gaussian marginalization; "
            "the linear parameters are rank-deficient or the quadratic "
            "metric is invalid"
        ) from exc

    diagonal = np.diag(chol)
    # An algebraically singular Gram matrix can occasionally pass an
    # unpivoted Cholesky because round-off creates a tiny positive pivot.  Such
    # a factor does not define a numerically resolved p-dimensional Gaussian
    # volume.  The scale-free threshold rejects only pivots at the effective
    # double-precision rank boundary.
    if np.min(diagonal) <= (
        10.0 * np.sqrt(np.finfo(float).eps) * np.max(diagonal)
    ):
        raise ValueError(
            f"{name} must be positive definite for Gaussian marginalization; "
            "its Cholesky factor is numerically rank-deficient"
        )
    return -float(np.sum(np.log(diagonal)))
