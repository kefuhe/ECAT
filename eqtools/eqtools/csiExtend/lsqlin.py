#!/usr/bin/python

# See http://maggotroot.blogspot.ch/2013/11/constrained-linear-least-squares-in.html for more info
'''
    A simple library to solve constrained linear least squares problems
    with sparse and dense matrices. Uses cvxopt library for
    optimization
'''

__author__ = 'Valeriy Vishnevskiy'
__email__ = 'valera.vishnevskiy@yandex.ru'
__version__ = '1.0'
__date__ = '22.11.2013'
__license__ = 'WTFPL'

import itertools
from dataclasses import dataclass

import numpy as np
from cvxopt import solvers, matrix, spmatrix, mul
from scipy import linalg, sparse
from scipy.linalg.lapack import dpocon


_DIRECT_CERTIFICATE_TOL = 1.0e-8
_TRUSTED_CERTIFICATE_TOL = 1.0e-6
_DIRECT_RCOND_THRESHOLD = 100.0 * np.finfo(float).eps


def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = spmatrix(coo.data, coo.row.tolist(), coo.col.tolist())
    return SP

def spmatrix_sparse_to_scipy(A):
    data = np.array(A.V).squeeze()
    rows = np.array(A.I).squeeze()
    cols = np.array(A.J).squeeze()
    return sparse.coo_matrix( (data, (rows, cols)) )

def sparse_None_vstack(A1, A2):
    if A1 is None:
        return A2
    else:
        return sparse.vstack([A1, A2])

def numpy_None_vstack(A1, A2):
    if A1 is None:
        return A2
    else:
        # print(A1.shape, A2.shape)
        return np.vstack([A1, A2])

def numpy_None_concatenate(A1, A2):
    if A1 is None:
        return A2
    else:
        return np.concatenate([A1, A2])

def get_shape(A):
    if isinstance(C, spmatrix):
        return C.size
    else:
        return C.shape

def numpy_to_cvxopt_matrix(A):
    if A is None:
        return A
    if sparse.issparse(A):
        if isinstance(A, sparse.spmatrix):
            return scipy_sparse_to_spmatrix(A)
        else:
            return A
    else:
        if isinstance(A, np.ndarray):
            if A.ndim == 1:
                return matrix(A, (A.shape[0], 1), 'd')
            else:
                return matrix(A, A.shape, 'd')
        else:
            return A

def cvxopt_to_numpy_matrix(A):
    """Convert CVXOPT/dense values without collapsing vectors to scalars.

    CVXOPT represents a one-parameter solution as a ``(1, 1)`` matrix.  A
    plain ``squeeze()`` turns that solution into a zero-dimensional array,
    which no longer satisfies the linear-model vector contract used by the
    BLSE, VCE, and Bayesian solvers.  Dense vector-shaped inputs therefore
    remain at least one-dimensional; genuine two-dimensional matrices keep
    their matrix shape.
    """
    if A is None:
        return A
    if isinstance(A, spmatrix):
        return spmatrix_sparse_to_scipy(A)
    return np.atleast_1d(np.asarray(A).squeeze())


@dataclass(frozen=True)
class PreparedQPConstraints:
    """Canonical linear constraints shared by every quadratic solve path.

    Inequalities always use ``G @ x <= h`` and equalities use
    ``Aeq @ x == beq``.  Bounds are appended after caller-supplied
    inequalities in the historical CVXOPT order: lower bounds first, then
    upper bounds.  Keeping this conversion in one place prevents a fast path
    from assigning different signs or row positions to the same constraint.
    """

    G: object
    h: object
    Aeq: object
    beq: object


@dataclass(frozen=True)
class QuadraticConstraintProfile:
    """Small routing view of one quadratic problem's constraints.

    The inversion and constraint-manager layers remain authoritative for the
    scientific meaning and parameter layout.  This profile records only the
    numerical structure needed to choose an exact solver path.
    """

    kind: str
    lower: np.ndarray
    upper: np.ndarray
    finite_lower: np.ndarray
    finite_upper: np.ndarray


def _prepare_qp_constraints(
        nvars, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None):
    """Return the exact linear-constraint layout consumed by CVXOPT.

    This helper intentionally preserves the dense/sparse choice and bound-row
    order of the longstanding ``_solve_qp`` implementation.  It is also the
    single source of truth for certified sequential-QP fast paths.
    """
    sparse_case = sparse.issparse(A) or isinstance(A, spmatrix)
    if isinstance(A, spmatrix):
        A = spmatrix_sparse_to_scipy(A)
    elif isinstance(A, matrix):
        A = np.asarray(A, dtype=float)
    if isinstance(Aeq, spmatrix):
        Aeq = spmatrix_sparse_to_scipy(Aeq)
    elif isinstance(Aeq, matrix):
        Aeq = np.asarray(Aeq, dtype=float)

    lb = cvxopt_to_numpy_matrix(lb)
    ub = cvxopt_to_numpy_matrix(ub)
    b = cvxopt_to_numpy_matrix(b)
    beq = cvxopt_to_numpy_matrix(beq)
    if b is not None and b.size == 1:
        b = np.array([b.item(0)])
    if beq is not None and np.asarray(beq).size == 1:
        beq = np.asarray(beq, dtype=float).reshape(1)

    if lb is not None:
        if lb.size == 1:
            lb = np.repeat(lb, nvars)
        if lb.size != nvars or np.any(np.isnan(lb)):
            raise ValueError(f"lb must contain one or {nvars} non-NaN entries")
        finite = np.isfinite(lb)
        if np.any(finite):
            if sparse_case:
                lb_A = -sparse.eye(nvars, nvars, format='csr')[finite]
                A = sparse_None_vstack(A, lb_A)
            else:
                lb_A = -np.eye(nvars)[finite]
                A = numpy_None_vstack(A, lb_A)
            b = numpy_None_concatenate(b, -lb[finite])
    if ub is not None:
        if ub.size == 1:
            ub = np.repeat(ub, nvars)
        if ub.size != nvars or np.any(np.isnan(ub)):
            raise ValueError(f"ub must contain one or {nvars} non-NaN entries")
        finite = np.isfinite(ub)
        if np.any(finite):
            if sparse_case:
                ub_A = sparse.eye(nvars, nvars, format='csr')[finite]
                A = sparse_None_vstack(A, ub_A)
            else:
                ub_A = np.eye(nvars)[finite]
                A = numpy_None_vstack(A, ub_A)
            b = numpy_None_concatenate(b, ub[finite])

    return PreparedQPConstraints(G=A, h=b, Aeq=Aeq, beq=beq)


def _as_vector(value, size=None, default=None):
    if value is None:
        if size is None or default is None:
            return None
        return np.full(size, default, dtype=float)
    out = np.asarray(cvxopt_to_numpy_matrix(value), dtype=float).reshape(-1)
    if size is not None and out.size == 1:
        out = np.full(size, out.item(), dtype=float)
    return out


def _matrix_row_count(value):
    if value is None:
        return 0
    if isinstance(value, (matrix, spmatrix)):
        return int(value.size[0])
    return int(value.shape[0])


def _rhs_has_entries(value):
    if value is None:
        return False
    return np.asarray(cvxopt_to_numpy_matrix(value)).size > 0


def _quadratic_constraint_profile(
        nvars, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None):
    """Classify constraints without changing their scientific semantics."""
    lower = _as_vector(lb, nvars, -np.inf)
    upper = _as_vector(ub, nvars, np.inf)
    if lower.size != nvars or upper.size != nvars:
        raise ValueError(
            f"bounds must contain one or {nvars} entries per side"
        )
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("bounds must not contain NaN")
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")

    general = (
        _matrix_row_count(A) > 0
        or _matrix_row_count(Aeq) > 0
        or _rhs_has_entries(b)
        or _rhs_has_entries(beq)
    )
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    if general:
        kind = "general_linear"
    elif np.any(finite_lower) or np.any(finite_upper):
        kind = "box_only"
    else:
        kind = "unconstrained"
    return QuadraticConstraintProfile(
        kind=kind,
        lower=lower,
        upper=upper,
        finite_lower=finite_lower,
        finite_upper=finite_upper,
    )


def certify_box_quadratic_solution(
        H, q, x, profile, *, tolerance=_DIRECT_CERTIFICATE_TOL):
    """Certify primal feasibility and projected KKT stationarity.

    For a lower-active variable the objective gradient must be non-negative;
    for an upper-active variable it must be non-positive; for a free variable
    it must vanish.  Fixed variables need no projected-gradient condition.
    This avoids constructing dense ``2p x p`` identity constraint matrices
    merely to certify an unconstrained optimum lying inside its bounds.
    """
    H = np.asarray(H, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    if (
        H.ndim != 2
        or H.shape[0] != H.shape[1]
        or x.size != q.size
        or x.size != H.shape[0]
        or not np.all(np.isfinite(H))
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(x))
    ):
        return {
            "passed": False,
            "primal": np.inf,
            "stationarity": np.inf,
            "active_lower": 0,
            "active_upper": 0,
        }

    lower_scale = 1.0 + np.abs(profile.lower)
    upper_scale = 1.0 + np.abs(profile.upper)
    lower_violation = np.zeros_like(x)
    upper_violation = np.zeros_like(x)
    lower_violation[profile.finite_lower] = np.maximum(
        profile.lower[profile.finite_lower] - x[profile.finite_lower], 0.0
    ) / lower_scale[profile.finite_lower]
    upper_violation[profile.finite_upper] = np.maximum(
        x[profile.finite_upper] - profile.upper[profile.finite_upper], 0.0
    ) / upper_scale[profile.finite_upper]
    primal = float(max(np.max(lower_violation), np.max(upper_violation)))

    activity_tol = 10.0 * tolerance
    at_lower = profile.finite_lower & (
        x - profile.lower <= activity_tol * lower_scale
    )
    at_upper = profile.finite_upper & (
        profile.upper - x <= activity_tol * upper_scale
    )
    fixed = at_lower & at_upper
    gradient = H @ x + q
    projected = gradient.copy()
    projected[at_lower & ~at_upper] = np.minimum(
        gradient[at_lower & ~at_upper], 0.0
    )
    projected[at_upper & ~at_lower] = np.maximum(
        gradient[at_upper & ~at_lower], 0.0
    )
    projected[fixed] = 0.0
    stationarity = float(
        np.linalg.norm(projected, ord=np.inf)
        / (1.0 + np.linalg.norm(q, ord=np.inf)
           + np.linalg.norm(H @ x, ord=np.inf))
    )
    return {
        "passed": bool(primal <= tolerance and stationarity <= tolerance),
        "primal": primal,
        "stationarity": stationarity,
        "active_lower": int(np.count_nonzero(at_lower)),
        "active_upper": int(np.count_nonzero(at_upper)),
    }


def _try_direct_quadratic_solution(
        H, q, reg, profile, *, symmetry_tolerance=1.0e-10):
    """Try the exact free optimum for unconstrained or box-only problems.

    Rejection changes only the numerical route: the caller proceeds to the
    trusted constrained QP with the original objective and constraints.
    """
    diagnostics = {
        "attempted": False,
        "accepted": False,
        "constraint_class": profile.kind,
        "reason": None,
    }
    if profile.kind == "general_linear":
        diagnostics["reason"] = "general_linear_constraints"
        return None, diagnostics
    if sparse.issparse(H) or isinstance(H, (matrix, spmatrix)):
        diagnostics["reason"] = "non_dense_hessian"
        return None, diagnostics

    H = np.asarray(H, dtype=float)
    q = np.asarray(cvxopt_to_numpy_matrix(q), dtype=float).reshape(-1)
    if (
        H.ndim != 2
        or H.shape[0] != H.shape[1]
        or q.size != H.shape[0]
        or not np.all(np.isfinite(H))
        or not np.all(np.isfinite(q))
    ):
        diagnostics["reason"] = "invalid_quadratic"
        return None, diagnostics
    diagnostics["attempted"] = True

    h_scale = max(1.0, np.linalg.norm(H, ord=np.inf))
    relative_asymmetry = float(
        np.linalg.norm(H - H.T, ord=np.inf) / h_scale
    )
    diagnostics["relative_asymmetry"] = relative_asymmetry
    if relative_asymmetry > symmetry_tolerance:
        diagnostics["reason"] = "nonsymmetric_hessian"
        return None, diagnostics

    H_effective = 0.5 * (H + H.T)
    if reg > 0:
        H_effective = H_effective + reg * np.eye(H.shape[0])
    try:
        factor = linalg.cholesky(
            H_effective, lower=True, check_finite=False
        )
    except linalg.LinAlgError:
        diagnostics["reason"] = "non_spd_hessian"
        return None, diagnostics

    rcond, info = dpocon(
        factor, np.linalg.norm(H_effective, ord=1), uplo='L'
    )
    rcond = float(rcond)
    diagnostics["hessian_rcond"] = rcond
    if info != 0 or not np.isfinite(rcond):
        diagnostics["reason"] = "condition_estimate_failed"
        return None, diagnostics
    if rcond <= _DIRECT_RCOND_THRESHOLD:
        diagnostics["reason"] = "numerically_rank_deficient_hessian"
        return None, diagnostics

    x_free = linalg.cho_solve(
        (factor, True), -q, check_finite=False
    )
    lower_scale = 1.0 + np.abs(profile.lower)
    upper_scale = 1.0 + np.abs(profile.upper)
    lower_violation = np.zeros_like(x_free)
    upper_violation = np.zeros_like(x_free)
    lower_violation[profile.finite_lower] = np.maximum(
        profile.lower[profile.finite_lower]
        - x_free[profile.finite_lower],
        0.0,
    ) / lower_scale[profile.finite_lower]
    upper_violation[profile.finite_upper] = np.maximum(
        x_free[profile.finite_upper]
        - profile.upper[profile.finite_upper],
        0.0,
    ) / upper_scale[profile.finite_upper]
    max_violation = float(max(
        np.max(lower_violation), np.max(upper_violation)
    ))
    diagnostics["free_solution_bound_violation"] = max_violation
    if max_violation > _DIRECT_CERTIFICATE_TOL:
        diagnostics["reason"] = "free_solution_outside_bounds"
        return None, diagnostics

    candidate = np.minimum(np.maximum(x_free, profile.lower), profile.upper)
    certificate = certify_box_quadratic_solution(
        H_effective, q, candidate, profile,
        tolerance=_DIRECT_CERTIFICATE_TOL,
    )
    diagnostics["certificate"] = certificate
    if not certificate["passed"]:
        diagnostics["reason"] = "direct_certificate_failed"
        return None, diagnostics
    if (
        profile.kind == "box_only"
        and (certificate["active_lower"] or certificate["active_upper"])
    ):
        # A free optimum numerically touching a bound is still mathematically
        # feasible, but the active-bound route is deliberately left to the
        # trusted box QP.  This conservative boundary guard preserves the
        # established constrained-solver tolerance semantics and keeps the
        # fast route limited to genuinely inactive bounds.
        diagnostics["reason"] = "free_solution_near_bound"
        return None, diagnostics

    diagnostics["accepted"] = True
    diagnostics["reason"] = "certified"
    route = (
        "direct_unconstrained"
        if profile.kind == "unconstrained"
        else "direct_box_inactive"
    )
    return {
        "status": "optimal",
        "x": matrix(candidate),
        "solver": "scipy_cholesky",
        "solve_route": route,
        "constraint_class": profile.kind,
        "qp_certificate": certificate,
        "route_diagnostics": diagnostics,
    }, diagnostics


def _as_sparse_rows(value, rhs, nvars):
    if value is None:
        return sparse.csr_matrix((0, nvars)), np.empty(0, dtype=float)
    if isinstance(value, spmatrix):
        value = spmatrix_sparse_to_scipy(value)
    rows = sparse.csr_matrix(value, dtype=float)
    return rows, _as_vector(rhs)


def _qp_constraint_scales(rows, rhs, x):
    if _matrix_row_count(rows) == 0:
        return np.empty(0, dtype=float)
    if sparse.issparse(rows):
        row_norms = np.asarray(np.abs(rows).sum(axis=1)).reshape(-1)
    elif isinstance(rows, spmatrix):
        rows = spmatrix_sparse_to_scipy(rows)
        row_norms = np.asarray(np.abs(rows).sum(axis=1)).reshape(-1)
    else:
        row_norms = np.sum(np.abs(np.asarray(rows, dtype=float)), axis=1)
    return 1.0 + np.abs(rhs) + row_norms * max(
        1.0, np.linalg.norm(x, ord=np.inf)
    )


def _qp_matvec(rows, vector):
    if isinstance(rows, spmatrix):
        rows = spmatrix_sparse_to_scipy(rows)
    return np.asarray(rows @ vector, dtype=float).reshape(-1)


def _qp_transpose_matvec(rows, vector):
    if isinstance(rows, spmatrix):
        rows = spmatrix_sparse_to_scipy(rows)
    return np.asarray(rows.T @ vector, dtype=float).reshape(-1)


def certify_qp_solution(
        H, q, constraints, x, lambda_ineq=None, lambda_eq=None,
        *, tolerance=_TRUSTED_CERTIFICATE_TOL):
    """Evaluate one canonical QP with scaled KKT residuals.

    The same certificate is consumed by the central trusted route and the
    optional VCE active-set path.  It is diagnostic for CVXOPT/Clarabel but a
    mandatory acceptance condition for alternative fast paths.
    """
    H = np.asarray(H, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    n_ineq = _matrix_row_count(constraints.G)
    n_eq = _matrix_row_count(constraints.Aeq)
    lambda_ineq = (
        np.zeros(n_ineq, dtype=float)
        if lambda_ineq is None
        else np.asarray(lambda_ineq, dtype=float).reshape(-1)
    )
    lambda_eq = (
        np.zeros(n_eq, dtype=float)
        if lambda_eq is None
        else np.asarray(lambda_eq, dtype=float).reshape(-1)
    )
    finite = (
        H.ndim == 2
        and H.shape[0] == H.shape[1]
        and x.size == q.size == H.shape[0]
        and lambda_ineq.size == n_ineq
        and lambda_eq.size == n_eq
        and np.all(np.isfinite(H))
        and np.all(np.isfinite(q))
        and np.all(np.isfinite(x))
        and np.all(np.isfinite(lambda_ineq))
        and np.all(np.isfinite(lambda_eq))
    )
    if not finite:
        return {
            "passed": False,
            "primal": np.inf,
            "stationarity": np.inf,
            "dual": np.inf,
            "complementarity": np.inf,
        }

    Hx = H @ x
    gradient = Hx + q
    primal = 0.0
    complementarity = 0.0
    if n_ineq:
        h = _as_vector(constraints.h)
        violation = _qp_matvec(constraints.G, x) - h
        scales = _qp_constraint_scales(constraints.G, h, x)
        primal = max(
            primal,
            float(np.max(np.maximum(violation, 0.0) / scales)),
        )
        gradient = gradient + _qp_transpose_matvec(
            constraints.G, lambda_ineq
        )
        objective_scale = 1.0 + abs(
            0.5 * float(x @ Hx) + float(q @ x)
        )
        complementarity = float(np.max(
            np.abs(lambda_ineq * violation) / objective_scale
        ))
    if n_eq:
        equality_rhs = _as_vector(constraints.beq)
        equality_residual = (
            _qp_matvec(constraints.Aeq, x) - equality_rhs
        )
        equality_scales = _qp_constraint_scales(
            constraints.Aeq, equality_rhs, x
        )
        primal = max(
            primal,
            float(np.max(np.abs(equality_residual) / equality_scales)),
        )
        gradient = gradient + _qp_transpose_matvec(
            constraints.Aeq, lambda_eq
        )

    stationarity = float(
        np.linalg.norm(gradient, ord=np.inf)
        / (1.0 + np.linalg.norm(q, ord=np.inf)
           + np.linalg.norm(Hx, ord=np.inf))
    )
    dual = (
        float(np.max(np.maximum(-lambda_ineq, 0.0)))
        / (1.0 + np.linalg.norm(lambda_ineq, ord=np.inf))
        if n_ineq else 0.0
    )
    return {
        "passed": bool(
            primal <= tolerance
            and stationarity <= tolerance
            and dual <= tolerance
            and complementarity <= tolerance
        ),
        "primal": primal,
        "stationarity": stationarity,
        "dual": dual,
        "complementarity": complementarity,
    }


def _normalize_rows(rows, rhs, include_rhs=False):
    """Scale each linear constraint by a positive factor."""
    if rows.shape[0] == 0:
        return rows, rhs
    norms = np.sqrt(np.asarray(rows.multiply(rows).sum(axis=1)).reshape(-1))
    magnitudes = np.maximum(norms, np.abs(rhs)) if include_rhs else norms
    factors = np.ones_like(magnitudes)
    active = magnitudes > 0.0
    factors[active] = 1.0 / magnitudes[active]
    return sparse.diags(factors) @ rows, rhs * factors


def _eliminate_separable_equalities(C, d, A, b, Aeq, beq, lb, ub):
    """Eliminate equalities having one private pivot column per row.

    This is exact algebraic substitution.  If the equality structure is not
    separable, the original problem is returned unchanged.
    """
    nvars = C.shape[1]
    if Aeq.shape[0] == 0:
        return C, d, A, b, Aeq, beq, lb, ub, None

    Aeq = Aeq.tocsr()
    column_counts = np.asarray(Aeq.getnnz(axis=0)).reshape(-1)
    pivot_columns = []
    pivot_values = []
    for row in range(Aeq.shape[0]):
        start, stop = Aeq.indptr[row:row + 2]
        columns = Aeq.indices[start:stop]
        values = Aeq.data[start:stop]
        candidates = np.flatnonzero(column_counts[columns] == 1)
        if candidates.size == 0:
            return C, d, A, b, Aeq, beq, lb, ub, None
        choice = candidates[np.argmax(np.abs(values[candidates]))]
        pivot_columns.append(int(columns[choice]))
        pivot_values.append(float(values[choice]))

    pivot_columns = np.asarray(pivot_columns, dtype=int)
    pivot_values = np.asarray(pivot_values, dtype=float)
    if np.any(pivot_values == 0.0):
        return C, d, A, b, Aeq, beq, lb, ub, None

    free_mask = np.ones(nvars, dtype=bool)
    free_mask[pivot_columns] = False
    free_columns = np.flatnonzero(free_mask)
    transform = sparse.lil_matrix((nvars, free_columns.size), dtype=float)
    transform[free_columns, np.arange(free_columns.size)] = 1.0
    transform[pivot_columns, :] = (
        -sparse.diags(1.0 / pivot_values) @ Aeq[:, free_columns]
    )
    transform = transform.tocsr()
    offset = np.zeros(nvars, dtype=float)
    offset[pivot_columns] = beq / pivot_values

    C_original = C
    A_original = A
    C = C_original @ transform
    d = d - np.asarray(C_original @ offset).reshape(-1)
    A = A_original @ transform
    b = b - np.asarray(A_original @ offset).reshape(-1)

    lower = _as_vector(lb, nvars, -np.inf)
    upper = _as_vector(ub, nvars, np.inf)
    blocks = [A]
    right_sides = [b]
    finite = np.isfinite(upper)
    if np.any(finite):
        blocks.append(transform[finite])
        right_sides.append(upper[finite] - offset[finite])
    finite = np.isfinite(lower)
    if np.any(finite):
        blocks.append(-transform[finite])
        right_sides.append(-lower[finite] + offset[finite])

    A = sparse.vstack(blocks, format='csr')
    b = np.concatenate(right_sides)
    empty_eq = sparse.csr_matrix((0, free_columns.size))
    recovery = {'offset': offset, 'transform': transform}
    return C, d, A, b, empty_eq, np.empty(0), None, None, recovery


def lsqlin_clarabel(C, d, reg=0, A=None, b=None, Aeq=None, beq=None,
                    lb=None, ub=None, x0=None, opts=None):
    """Solve constrained least squares without forming ``C.T @ C``.

    The least-squares norm is represented by a second-order cone.  Separable
    hard equalities are eliminated exactly before solving and restored before
    return.  This path is intended as a robust fallback for ill-scaled linear
    inversions, not as the normal fast path.
    """
    try:
        import clarabel
    except ImportError as exc:
        raise ImportError(
            "Clarabel is required for the robust constrained least-squares "
            "fallback. Install ECAT dependencies or run 'pip install clarabel'."
        ) from exc

    C = sparse.csr_matrix(C, dtype=float)
    d = _as_vector(d)
    original_nvars = C.shape[1]
    if reg > 0:
        C = sparse.vstack(
            (C, np.sqrt(reg) * sparse.eye(original_nvars)), format='csr'
        )
        d = np.concatenate((d, np.zeros(original_nvars)))

    A, b = _as_sparse_rows(A, b, original_nvars)
    Aeq, beq = _as_sparse_rows(Aeq, beq, original_nvars)
    C, d, A, b, Aeq, beq, lb, ub, recovery = (
        _eliminate_separable_equalities(C, d, A, b, Aeq, beq, lb, ub)
    )

    nvars = C.shape[1]
    column_norms = np.sqrt(np.asarray(C.multiply(C).sum(axis=0)).reshape(-1))
    scales = np.ones(nvars, dtype=float)
    active = column_norms > 1.0
    scales[active] = 1.0 / column_norms[active]
    scale_matrix = sparse.diags(scales)
    C_scaled = C @ scale_matrix
    A, b = _normalize_rows(A @ scale_matrix, b)
    Aeq, beq = _normalize_rows(Aeq @ scale_matrix, beq)

    lower = _as_vector(lb, nvars, -np.inf) / scales
    upper = _as_vector(ub, nvars, np.inf) / scales
    linear_blocks = [
        sparse.hstack((A, sparse.csr_matrix((A.shape[0], 1))))
    ]
    linear_rhs = [b]
    finite = np.isfinite(lower)
    if np.any(finite):
        linear_blocks.append(sparse.hstack((
            -sparse.eye(nvars, format='csr')[finite],
            sparse.csr_matrix((np.count_nonzero(finite), 1)),
        )))
        linear_rhs.append(-lower[finite])
    finite = np.isfinite(upper)
    if np.any(finite):
        linear_blocks.append(sparse.hstack((
            sparse.eye(nvars, format='csr')[finite],
            sparse.csr_matrix((np.count_nonzero(finite), 1)),
        )))
        linear_rhs.append(upper[finite])
    linear_matrix = sparse.vstack(linear_blocks, format='csr')
    linear_rhs = np.concatenate(linear_rhs)
    linear_matrix, linear_rhs = _normalize_rows(
        linear_matrix, linear_rhs, include_rhs=True
    )

    soc_matrix = sparse.vstack((
        sparse.csr_matrix(([-1.0], ([0], [nvars])), shape=(1, nvars + 1)),
        sparse.hstack((C_scaled, sparse.csr_matrix((C_scaled.shape[0], 1)))),
    ), format='csr')
    soc_rhs = np.concatenate(([0.0], d))

    cone_blocks = []
    cone_rhs = []
    cones = []
    if Aeq.shape[0]:
        cone_blocks.append(sparse.hstack((
            Aeq, sparse.csr_matrix((Aeq.shape[0], 1))
        )))
        cone_rhs.append(beq)
        cones.append(clarabel.ZeroConeT(Aeq.shape[0]))
    cone_blocks.append(linear_matrix)
    cone_rhs.append(linear_rhs)
    cones.append(clarabel.NonnegativeConeT(linear_matrix.shape[0]))
    cone_blocks.append(soc_matrix)
    cone_rhs.append(soc_rhs)
    cones.append(clarabel.SecondOrderConeT(soc_matrix.shape[0]))

    constraint_matrix = sparse.vstack(cone_blocks, format='csc')
    constraint_rhs = np.concatenate(cone_rhs)
    objective = np.zeros(nvars + 1, dtype=float)
    objective[-1] = 1.0
    settings = clarabel.DefaultSettings()
    settings.verbose = bool((opts or {}).get('show_progress', False))
    settings.max_iter = int((opts or {}).get('maxiters', 200))
    settings.direct_solve_method = 'qdldl'
    settings.max_threads = 1
    settings.max_step_fraction = 0.95

    result = clarabel.DefaultSolver(
        sparse.csc_matrix((nvars + 1, nvars + 1)),
        objective,
        constraint_matrix,
        constraint_rhs,
        cones,
        settings,
    ).solve()
    clarabel_status = str(result.status)
    status_map = {'Solved': 'optimal', 'AlmostSolved': 'optimal_inaccurate'}
    status = status_map.get(clarabel_status, clarabel_status.lower())
    solution = None
    if result.x is not None:
        solution = np.asarray(result.x, dtype=float).reshape(-1)[:nvars] * scales
        if recovery is not None:
            solution = recovery['offset'] + np.asarray(
                recovery['transform'] @ solution
            ).reshape(-1)
        solution = matrix(solution)
    return {
        'status': status,
        'x': solution,
        'solver': 'clarabel_socp',
        'clarabel_status': clarabel_status,
        'iterations': getattr(result, 'iterations', None),
    }


def lsqlin_auto(C, d, reg=0, A=None, b=None, Aeq=None, beq=None,
                lb=None, ub=None, x0=None, opts=None):
    """Use the legacy QP first, then retry the same problem robustly."""
    primary_error = None
    primary = None
    try:
        primary = lsqlin(C, d, reg, A, b, Aeq, beq, lb, ub, x0, opts)
    except Exception as exc:
        primary_error = exc
    if primary is not None and str(primary.get('status', '')).lower() == 'optimal':
        primary['solver'] = 'cvxopt_qp'
        return primary

    primary_status = (
        f"exception {type(primary_error).__name__}: {primary_error}"
        if primary_error is not None
        else f"status {primary.get('status', 'unknown')!r}"
    )
    robust = lsqlin_clarabel(
        C, d, reg, A, b, Aeq, beq, lb, ub, x0, opts
    )
    robust['fallback_reason'] = primary_status
    if str(robust.get('status', '')).lower() not in {
        'optimal', 'optimal_inaccurate'
    }:
        raise RuntimeError(
            "Constrained least squares failed in both backends: "
            f"CVXOPT {primary_status}; Clarabel status "
            f"{robust.get('clarabel_status', robust.get('status'))!r}."
        )
    return robust


def lsqlin_quadratic_auto(
        H, q, residual_factory, reg=0, A=None, b=None, Aeq=None, beq=None,
        lb=None, ub=None, x0=None, opts=None):
    """Solve a prepared quadratic objective with a lazy residual fallback.

    ``H`` and ``q`` define the same objective as an unmaterialized residual
    system, ``H = C.T @ C`` and ``q = -C.T @ d``.  The central quadratic
    route first accepts a certified Cholesky solution only when the problem is
    unconstrained or box-only and the free optimum is feasible.  All other
    cases retain the trusted CVXOPT QP.  ``residual_factory`` is evaluated
    only if that trusted solve fails, preserving the residual-form Clarabel
    fallback without paying its assembly cost during a normal solve.
    """
    primary_error = None
    primary = None
    try:
        primary = lsqlin_quadratic(
            H, q, reg, A, b, Aeq, beq, lb, ub, x0, opts
        )
    except Exception as exc:
        primary_error = exc
    if primary is not None and str(primary.get('status', '')).lower() == 'optimal':
        primary.setdefault('solver', 'cvxopt_qp')
        return primary

    primary_status = (
        f"exception {type(primary_error).__name__}: {primary_error}"
        if primary_error is not None
        else f"status {primary.get('status', 'unknown')!r}"
    )
    C, d = residual_factory()
    robust = lsqlin_clarabel(
        C, d, reg, A, b, Aeq, beq, lb, ub, x0, opts
    )
    robust['fallback_reason'] = primary_status
    robust['solve_route'] = 'clarabel_fallback'
    if str(robust.get('status', '')).lower() not in {
        'optimal', 'optimal_inaccurate'
    }:
        raise RuntimeError(
            "Constrained least squares failed in both backends: "
            f"CVXOPT {primary_status}; Clarabel status "
            f"{robust.get('clarabel_status', robust.get('status'))!r}."
        )
    return robust


def _solve_qp(H, q, A=None, b=None, Aeq=None, beq=None,
              lb=None, ub=None, x0=None, opts=None,
              prepared_constraints=None):
    """Send an already assembled quadratic objective to CVXOPT."""
    H = numpy_to_cvxopt_matrix(H)
    q = numpy_to_cvxopt_matrix(q)
    nvars = H.size[1]
    if H.size[0] != nvars:
        raise ValueError("H must be square")
    if q.size == (1, nvars):
        q = q.T
    if q.size != (nvars, 1):
        raise ValueError(f"q must contain {nvars} entries")

    constraints = prepared_constraints
    if constraints is None:
        constraints = _prepare_qp_constraints(
            nvars, A, b, Aeq, beq, lb, ub
        )
    A = numpy_to_cvxopt_matrix(constraints.G)
    Aeq = numpy_to_cvxopt_matrix(constraints.Aeq)
    b = numpy_to_cvxopt_matrix(constraints.h)
    beq = numpy_to_cvxopt_matrix(constraints.beq)

    if opts is not None:
        for key, value in opts.items():
            solvers.options[key] = value
    return solvers.qp(H, q, A, b, Aeq, beq, None, x0)


def lsqlin_quadratic(H, q, reg=0, A=None, b=None, Aeq=None, beq=None,
                     lb=None, ub=None, x0=None, opts=None):
    """Solve one exact quadratic through the shared automatic route.

    The objective follows CVXOPT's convention,
    ``0.5 * x.T @ H @ x + q.T @ x``.  ``reg`` preserves ``lsqlin`` semantics
    by adding ``reg * I`` to ``H``.  Unconstrained and box-only problems first
    try the free SPD optimum; that solution is accepted only after bounds and
    projected-KKT certification.  General linear constraints, active bounds,
    non-SPD metrics, and failed certificates use the unchanged CVXOPT QP.
    """
    if sparse.issparse(H):
        nvars = H.shape[0]
        if H.ndim != 2 or H.shape[1] != nvars:
            raise ValueError("H must be square")
    elif isinstance(H, (matrix, spmatrix)):
        nvars = H.size[0]
        if H.size[1] != nvars:
            raise ValueError("H must be square")
    else:
        H = np.asarray(H, dtype=float)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("H must be a square two-dimensional matrix")
        if not np.all(np.isfinite(H)):
            raise ValueError("H must contain only finite values")
        nvars = H.shape[0]
    q_array = np.asarray(cvxopt_to_numpy_matrix(q), dtype=float).reshape(-1)
    if q_array.size != nvars or not np.all(np.isfinite(q_array)):
        raise ValueError(f"q must contain {nvars} finite entries")

    profile = _quadratic_constraint_profile(
        nvars, A, b, Aeq, beq, lb, ub
    )
    direct, route_diagnostics = _try_direct_quadratic_solution(
        H, q_array, reg, profile
    )
    if direct is not None:
        return direct

    if sparse.issparse(H):
        H_effective = H.copy()
        if reg > 0:
            H_effective = H_effective + reg * sparse.eye(
                nvars, format=H.format
            )
    elif isinstance(H, (matrix, spmatrix)):
        H_effective = H
        if reg > 0:
            identity = (
                spmatrix(1.0, range(nvars), range(nvars))
                if isinstance(H, spmatrix)
                else matrix(np.eye(nvars), (nvars, nvars), 'd')
            )
            H_effective = H_effective + reg * identity
    else:
        H_effective = H + reg * np.eye(nvars) if reg > 0 else H

    constraints = _prepare_qp_constraints(
        nvars, A, b, Aeq, beq, lb, ub
    )
    result = _solve_qp(
        H_effective,
        q_array,
        A,
        b,
        Aeq,
        beq,
        lb,
        ub,
        x0,
        opts,
        prepared_constraints=constraints,
    )
    result['solver'] = 'cvxopt_qp'
    result['constraint_class'] = profile.kind
    result['solve_route'] = (
        'general_qp'
        if profile.kind == 'general_linear'
        else 'box_qp'
        if profile.kind == 'box_only'
        else 'unconstrained_qp'
    )
    result['route_diagnostics'] = route_diagnostics

    # CVXOPT remains authoritative when selected.  Its KKT certificate is
    # recorded centrally for diagnostics and VCE active-set seeding; unlike a
    # fast-path certificate it does not silently replace backend status.
    if (
        isinstance(H_effective, np.ndarray)
        and str(result.get('status', '')).lower() == 'optimal'
        and result.get('x') is not None
        and result.get('z') is not None
    ):
        x = np.asarray(
            cvxopt_to_numpy_matrix(result['x']), dtype=float
        ).reshape(-1)
        lambda_ineq = np.asarray(
            cvxopt_to_numpy_matrix(result['z']), dtype=float
        ).reshape(-1)
        lambda_eq = (
            np.empty(0, dtype=float)
            if result.get('y') is None
            else np.asarray(
                cvxopt_to_numpy_matrix(result['y']), dtype=float
            ).reshape(-1)
        )
        result['qp_certificate'] = certify_qp_solution(
            H_effective,
            q_array,
            constraints,
            x,
            lambda_ineq,
            lambda_eq,
            tolerance=_TRUSTED_CERTIFICATE_TOL,
        )
    return result


def lsqlin(C, d, reg=0, A=None, b=None, Aeq=None, beq=None, \
        lb=None, ub=None, x0=None, opts=None):
    '''
        Solve linear constrained l2-regularized least squares. Can
        handle both dense and sparse matrices. Matlab's lsqlin
        equivalent. It is actually wrapper around CVXOPT QP solver.
            min_x ||C*x  - d||^2_2 + reg * ||x||^2_2
            s.t.    A * x <= b
                    Aeq * x = beq
                    lb <= x <= ub
        Input arguments:
            C   is m x n dense or sparse matrix
            d   is n x 1 dense matrix
            reg is regularization parameter
            A   is p x n dense or sparse matrix
            b   is p x 1 dense matrix
            Aeq is q x n dense or sparse matrix
            beq is q x 1 dense matrix
            lb  is n x 1 matrix or scalar
            ub  is n x 1 matrix or scalar
        Output arguments:
            Return dictionary, the output of CVXOPT QP.
        Dont pass matlab-like empty lists to avoid setting parameters,
        just use None:
            lsqlin(C, d, 0.05, None, None, Aeq, beq) #Correct
            lsqlin(C, d, 0.05, [], [], Aeq, beq) #Wrong!
    '''
    C =   numpy_to_cvxopt_matrix(C)
    d =   numpy_to_cvxopt_matrix(d)
    Q = C.T * C
    q = - C.T * d
    nvars = C.size[1]

    if reg > 0:
        if isinstance(Q, spmatrix):
            I = scipy_sparse_to_spmatrix(sparse.eye(nvars, nvars,\
                                        format='coo'))
        else:
            I = matrix(np.eye(nvars), (nvars, nvars), 'd')
        Q = Q + reg * I
    return _solve_qp(Q, q, A, b, Aeq, beq, lb, ub, x0, opts)

def lsqnonneg(C, d, opts):
    '''
    Solves nonnegative linear least-squares problem:
    min_x ||C*x - d||_2^2,  where x >= 0
    '''
    return lsqlin(C, d, reg = 0, A = None, b = None, Aeq = None, \
                beq = None, lb = 0, ub = None, x0 = None, opts = opts)


if __name__ == '__main__':
    # simple Testing routines
    C = np.array(np.mat('''0.9501,0.7620,0.6153,0.4057;
    0.2311,0.4564,0.7919,0.9354;
    0.6068,0.0185,0.9218,0.9169;
    0.4859,0.8214,0.7382,0.4102;
    0.8912,0.4447,0.1762,0.8936'''))
    sC = sparse.coo_matrix(C)
    csC = scipy_sparse_to_spmatrix(sC)

    A = np.array(np.mat('''0.2027,0.2721,0.7467,0.4659;
    0.1987,0.1988,0.4450,0.4186;
    0.6037,0.0152,0.9318,0.8462'''))
    sA = sparse.coo_matrix(A)
    csA = scipy_sparse_to_spmatrix(sA)

    d = np.array([0.0578, 0.3528, 0.8131, 0.0098, 0.1388])
    md = matrix(d)

    b =  np.array([0.5251, 0.2026, 0.6721])
    mb = matrix(b)

    lb = np.array([-0.1] * 4)
    mlb = matrix(lb)
    mmlb = -0.1

    ub = np.array([2] * 4)
    mub = matrix(ub)
    mmub = 2

    #solvers.options[show_progress'] = False
    opts = {'show_progress': False}

    for iC in [C, sC, csC]:
        for iA in [A, sA, csA]:
            for iD in [d, md]:
                for ilb in [lb, mlb, mmlb]:
                    for iub in [ub, mub, mmub]:
                        for ib in [b, mb]:
                            ret = lsqlin(iC, iD, 0, iA, ib, None, None, ilb, iub, None, opts)
                            print(ret['x'].T)
    print('Should be [-1.00e-01 -1.00e-01  2.15e-01  3.50e-01]')

    #test lsqnonneg
    C = np.array([[0.0372, 0.2869], [0.6861, 0.7071], [0.6233, 0.6245], [0.6344, 0.6170]]);
    d = np.array([0.8587, 0.1781, 0.0747, 0.8405]);
    ret = lsqnonneg(C, d, {'show_progress': False})
    print(ret['x'].T)
    print('Should be [2.5e-07; 6.93e-01]')
