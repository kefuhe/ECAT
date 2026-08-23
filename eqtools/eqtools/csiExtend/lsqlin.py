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

import numpy as np
from cvxopt import solvers, matrix, spmatrix, mul
import itertools
from scipy import sparse


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


def _as_vector(value, size=None, default=None):
    if value is None:
        if size is None or default is None:
            return None
        return np.full(size, default, dtype=float)
    out = np.asarray(cvxopt_to_numpy_matrix(value), dtype=float).reshape(-1)
    if size is not None and out.size == 1:
        out = np.full(size, out.item(), dtype=float)
    return out


def _as_sparse_rows(value, rhs, nvars):
    if value is None:
        return sparse.csr_matrix((0, nvars)), np.empty(0, dtype=float)
    if isinstance(value, spmatrix):
        value = spmatrix_sparse_to_scipy(value)
    rows = sparse.csr_matrix(value, dtype=float)
    return rows, _as_vector(rhs)


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
    system, ``H = C.T @ C`` and ``q = -C.T @ d``.  The callable
    ``residual_factory`` is evaluated only when CVXOPT fails, preserving the
    residual-form Clarabel fallback without paying its assembly cost during a
    normal solve.
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
        primary['solver'] = 'cvxopt_qp'
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
              lb=None, ub=None, x0=None, opts=None):
    """Send an already assembled quadratic objective to CVXOPT."""
    sparse_case = sparse.issparse(A) or isinstance(A, spmatrix)
    if isinstance(A, spmatrix):
        A = spmatrix_sparse_to_scipy(A)

    H = numpy_to_cvxopt_matrix(H)
    q = numpy_to_cvxopt_matrix(q)
    nvars = H.size[1]
    if H.size[0] != nvars:
        raise ValueError("H must be square")
    if q.size == (1, nvars):
        q = q.T
    if q.size != (nvars, 1):
        raise ValueError(f"q must contain {nvars} entries")

    lb = cvxopt_to_numpy_matrix(lb)
    ub = cvxopt_to_numpy_matrix(ub)
    b = cvxopt_to_numpy_matrix(b)
    if b is not None and b.size == 1:
        b = np.array([b.item(0)])

    if lb is not None:
        if lb.size == 1:
            lb = np.repeat(lb, nvars)
        if sparse_case:
            lb_A = -sparse.eye(nvars, nvars, format='coo')
            A = sparse_None_vstack(A, lb_A)
        else:
            lb_A = -np.eye(nvars)
            A = numpy_None_vstack(A, lb_A)
        b = numpy_None_concatenate(b, -lb)
    if ub is not None:
        if ub.size == 1:
            ub = np.repeat(ub, nvars)
        if sparse_case:
            ub_A = sparse.eye(nvars, nvars, format='coo')
            A = sparse_None_vstack(A, ub_A)
        else:
            ub_A = np.eye(nvars)
            A = numpy_None_vstack(A, ub_A)
        b = numpy_None_concatenate(b, ub)

    A = numpy_to_cvxopt_matrix(A)
    Aeq = numpy_to_cvxopt_matrix(Aeq)
    b = numpy_to_cvxopt_matrix(b)
    beq = numpy_to_cvxopt_matrix(beq)

    if opts is not None:
        for key, value in opts.items():
            solvers.options[key] = value
    return solvers.qp(H, q, A, b, Aeq, beq, None, x0)


def lsqlin_quadratic(H, q, reg=0, A=None, b=None, Aeq=None, beq=None,
                     lb=None, ub=None, x0=None, opts=None):
    """Solve the exact quadratic form of a constrained least-squares problem.

    The objective follows CVXOPT's convention,
    ``0.5 * x.T @ H @ x + q.T @ x``.  ``reg`` preserves ``lsqlin`` semantics
    by adding ``reg * I`` to ``H``.
    """
    if sparse.issparse(H):
        H = H.copy()
        nvars = H.shape[0]
        if H.ndim != 2 or H.shape[1] != nvars:
            raise ValueError("H must be square")
        if reg > 0:
            H = H + reg * sparse.eye(nvars, format=H.format)
    elif isinstance(H, (matrix, spmatrix)):
        nvars = H.size[0]
        if H.size[1] != nvars:
            raise ValueError("H must be square")
        if reg > 0:
            identity = (
                spmatrix(1.0, range(nvars), range(nvars))
                if isinstance(H, spmatrix)
                else matrix(np.eye(nvars), (nvars, nvars), 'd')
            )
            H = H + reg * identity
    else:
        H = np.asarray(H, dtype=float)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("H must be a square two-dimensional matrix")
        if not np.all(np.isfinite(H)):
            raise ValueError("H must contain only finite values")
        nvars = H.shape[0]
        if reg > 0:
            H = H + reg * np.eye(nvars)
    q_array = np.asarray(cvxopt_to_numpy_matrix(q), dtype=float).reshape(-1)
    if q_array.size != nvars or not np.all(np.isfinite(q_array)):
        raise ValueError(f"q must contain {nvars} finite entries")
    return _solve_qp(
        H, q_array, A, b, Aeq, beq, lb, ub, x0, opts
    )


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
