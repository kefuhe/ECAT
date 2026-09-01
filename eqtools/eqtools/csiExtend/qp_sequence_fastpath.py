"""Certified active-set fast path for one local sequence of convex QPs.

The module deliberately does not own a solver backend registry or any VCE
state.  It only reuses a certified working set inside one caller-owned
sequence.  Every failed prediction is discarded and the caller's trusted
solver remains authoritative.
"""

from collections import Counter
from dataclasses import dataclass, field
import time
import warnings

import numpy as np
from scipy import linalg, sparse

from .lsqlin import (
    PreparedQPConstraints,
    certify_qp_solution,
    cvxopt_to_numpy_matrix,
)


_MAX_ACTIVE_RANK_RATIO = 0.5
_MAX_REPAIRS = 10
_REPAIR_BATCH_SIZE = 8
_FAST_CERTIFICATE_TOL = 1.0e-8
_SEED_CERTIFICATE_TOL = 1.0e-6
_ACTIVE_SLACK_TOL = 1.0e-6
_ACTIVE_DUAL_TOL = 1.0e-9


@dataclass
class VCEKKTSession:
    """Candidate-local state for consecutive QPs in one VCE call."""

    constraints: PreparedQPConstraints
    x: np.ndarray | None = None
    active_rows: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )
    inequality_multipliers: np.ndarray | None = None
    certified: bool = False
    retry_measure: float | None = None
    disabled_reason: str | None = None
    attempts: int = 0
    successes: int = 0
    repairs: int = 0
    fallbacks: int = 0
    skips: int = 0
    kkt_seconds: float = 0.0
    fallback_seconds: float = 0.0
    failure_reasons: Counter = field(default_factory=Counter)
    last_failure_certificate: dict | None = None
    last_seed_active_count: int = 0
    repair_histogram: Counter = field(default_factory=Counter)

    def diagnostics(self):
        """Return report-only counters; none of them affect VCE updates."""
        return {
            "acceleration": "certified_kkt",
            "attempts": self.attempts,
            "successes": self.successes,
            "repairs": self.repairs,
            "fallbacks": self.fallbacks,
            "skips": self.skips,
            "kkt_seconds": self.kkt_seconds,
            "fallback_seconds": self.fallback_seconds,
            "disabled_reason": self.disabled_reason,
            "failure_reasons": dict(self.failure_reasons),
            "last_failure_certificate": self.last_failure_certificate,
            "last_seed_active_count": self.last_seed_active_count,
            "repair_histogram": dict(self.repair_histogram),
        }


def solve_vce_qp_candidate(
        H, q, *, session, fallback, change_measure=None):
    """Solve one VCE QP with a certified KKT attempt and trusted fallback.

    ``fallback`` must solve the same ``H, q`` and canonical constraints.  A
    KKT trial mutates no accepted state until its central certificate passes;
    otherwise the fallback result replaces the session state atomically.
    """
    if not isinstance(session, VCEKKTSession):
        raise TypeError("session must be a VCEKKTSession")

    attempted = False
    if _eligible_for_kkt(session, H.shape[0], change_measure):
        attempted = True
        session.attempts += 1
        started = time.perf_counter()
        trial = _try_active_set_kkt(
            H,
            q,
            session.constraints,
            session.active_rows,
        )
        session.kkt_seconds += time.perf_counter() - started
        session.repairs += trial.get("repairs", 0)
        outcome = "success" if trial["success"] else "failure"
        session.repair_histogram[
            f"{outcome}:{trial.get('repairs', 0)}"
        ] += 1
        if trial["success"]:
            session.successes += 1
            session.x = trial["x"]
            session.active_rows = trial["active_rows"]
            session.inequality_multipliers = trial[
                "inequality_multipliers"
            ]
            session.certified = True
            session.retry_measure = None
            return {
                "status": "optimal",
                "x": trial["x"],
                "solver": "kkt_fastpath",
                "solve_route": "vce_certified_kkt",
                "qp_certificate": trial["certificate"],
            }

        session.failure_reasons[trial["reason"]] += 1
        session.last_failure_certificate = trial.get("certificate")
        if trial.get("fatal", False):
            session.disabled_reason = trial["reason"]
        elif change_measure is not None and np.isfinite(change_measure):
            session.retry_measure = 0.5 * float(change_measure)

    if not attempted:
        session.skips += 1
    session.fallbacks += 1
    started = time.perf_counter()
    result = fallback()
    session.fallback_seconds += time.perf_counter() - started
    _replace_state_from_trusted_result(session, H, q, result)
    return result


def _eligible_for_kkt(session, n_parameters, change_measure):
    if session.disabled_reason is not None or not session.certified:
        return False
    if session.x is None or session.inequality_multipliers is None:
        return False
    if session.retry_measure is not None:
        if change_measure is None or not np.isfinite(change_measure):
            return False
        if float(change_measure) > session.retry_measure:
            return False
    equality_count = _row_count(session.constraints.Aeq)
    active_count = equality_count + session.active_rows.size
    if active_count / max(1, n_parameters) > _MAX_ACTIVE_RANK_RATIO:
        return False
    return True


def _try_active_set_kkt(H, q, constraints, initial_active):
    H = np.asarray(H, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    if (
        H.ndim != 2
        or H.shape[0] != H.shape[1]
        or q.size != H.shape[0]
        or not np.all(np.isfinite(H))
        or not np.all(np.isfinite(q))
    ):
        return _failed("invalid_quadratic", fatal=True)

    asymmetry = np.linalg.norm(H - H.T, ord=np.inf)
    h_scale = max(1.0, np.linalg.norm(H, ord=np.inf))
    if asymmetry > 1.0e-10 * h_scale:
        return _failed("nonsymmetric_hessian", fatal=True)
    H = 0.5 * (H + H.T)
    try:
        h_factor = linalg.cho_factor(H, lower=True, check_finite=False)
    except linalg.LinAlgError:
        return _failed("non_spd_hessian", fatal=True)

    G = constraints.G
    h = _vector(constraints.h)
    E = constraints.Aeq
    f = _vector(constraints.beq)
    n_ineq = _row_count(G)
    active = np.unique(np.asarray(initial_active, dtype=int))
    if np.any(active < 0) or np.any(active >= n_ineq):
        return _failed("invalid_active_rows", fatal=True)

    for repair in range(_MAX_REPAIRS + 1):
        solved = _solve_working_set(
            h_factor, q, G, h, E, f, active
        )
        if not solved["success"]:
            return _failed(solved["reason"], fatal=True, repairs=repair)

        x = solved["x"]
        lambda_ineq = np.zeros(n_ineq, dtype=float)
        if active.size:
            lambda_ineq[active] = solved["active_multipliers"]
        certificate = certify_qp_solution(
            H,
            q,
            constraints,
            x,
            lambda_ineq,
            solved["equality_multipliers"],
            tolerance=_FAST_CERTIFICATE_TOL,
        )
        if certificate["passed"]:
            return {
                "success": True,
                "x": x,
                "active_rows": active,
                "inequality_multipliers": lambda_ineq,
                "certificate": certificate,
                "repairs": repair,
            }
        if repair == _MAX_REPAIRS:
            break

        next_active = _repair_active_set(
            G, h, x, active, lambda_ineq
        )
        if np.array_equal(next_active, active):
            break
        active = next_active

    return _failed(
        "certificate_failed",
        fatal=False,
        repairs=_MAX_REPAIRS,
        certificate=certificate,
    )


def _solve_working_set(h_factor, q, G, h, E, f, active):
    equality_count = _row_count(E)
    blocks = []
    rhs_blocks = []
    if equality_count:
        blocks.append(_dense_rows(E))
        rhs_blocks.append(f)
    if active.size:
        blocks.append(_dense_rows(G, active))
        rhs_blocks.append(h[active])

    hq = linalg.cho_solve(h_factor, q, check_finite=False)
    if not blocks:
        return {
            "success": True,
            "x": -hq,
            "equality_multipliers": np.empty(0),
            "active_multipliers": np.empty(0),
        }

    C = np.vstack(blocks)
    target = np.concatenate(rhs_blocks)
    Hinv_Ct = linalg.cho_solve(
        h_factor, C.T, check_finite=False
    )
    schur = C @ Hinv_Ct
    schur = 0.5 * (schur + schur.T)
    dual_rhs = -target - C @ hq
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", linalg.LinAlgWarning)
            schur_factor = linalg.cho_factor(
                schur, lower=True, check_finite=False
            )
            multipliers = linalg.cho_solve(
                schur_factor, dual_rhs, check_finite=False
            )
    except (linalg.LinAlgError, linalg.LinAlgWarning):
        return {"success": False, "reason": "rank_deficient_working_set"}

    x = -hq - Hinv_Ct @ multipliers
    return {
        "success": True,
        "x": x,
        "equality_multipliers": multipliers[:equality_count],
        "active_multipliers": multipliers[equality_count:],
    }


def _repair_active_set(G, h, x, active, multipliers):
    n_ineq = _row_count(G)
    if n_ineq == 0:
        return active
    values = _matvec(G, x) - h
    scales = _constraint_scales(G, h, x)
    normalized_violation = values / scales

    active_mask = np.zeros(n_ineq, dtype=bool)
    active_mask[active] = True
    inactive = np.flatnonzero(~active_mask)
    violated = inactive[normalized_violation[inactive] > _FAST_CERTIFICATE_TOL]
    if violated.size:
        order = np.argsort(normalized_violation[violated])[::-1]
        violated = violated[order[:_REPAIR_BATCH_SIZE]]

    bad_dual = active[
        multipliers[active]
        < -_FAST_CERTIFICATE_TOL
        * (1.0 + np.linalg.norm(multipliers, ord=np.inf))
    ]
    if bad_dual.size > _REPAIR_BATCH_SIZE:
        bad_dual = bad_dual[:_REPAIR_BATCH_SIZE]

    kept = np.setdiff1d(active, bad_dual, assume_unique=True)
    return np.union1d(kept, violated).astype(int, copy=False)


def _replace_state_from_trusted_result(session, H, q, result):
    session.x = None
    session.active_rows = np.empty(0, dtype=int)
    session.inequality_multipliers = None
    session.certified = False
    if str(result.get("solver", "")).lower() != "cvxopt_qp":
        return
    if str(result.get("status", "")).lower() != "optimal":
        return
    if result.get("x") is None or result.get("z") is None:
        return

    x = np.asarray(cvxopt_to_numpy_matrix(result["x"]), dtype=float).reshape(-1)
    lambda_ineq = np.asarray(
        cvxopt_to_numpy_matrix(result["z"]), dtype=float
    ).reshape(-1)
    lambda_eq = (
        np.empty(0, dtype=float)
        if result.get("y") is None
        else np.asarray(
            cvxopt_to_numpy_matrix(result["y"]), dtype=float
        ).reshape(-1)
    )
    certificate = certify_qp_solution(
        H,
        q,
        session.constraints,
        x,
        lambda_ineq,
        lambda_eq,
        tolerance=_SEED_CERTIFICATE_TOL,
    )
    if not certificate["passed"]:
        return

    h = _vector(session.constraints.h)
    slack = h - _matvec(session.constraints.G, x)
    scales = _constraint_scales(session.constraints.G, h, x)
    dual_scale = 1.0 + np.linalg.norm(lambda_ineq, ord=np.inf)
    active = np.flatnonzero(
        (slack / scales <= _ACTIVE_SLACK_TOL)
        & (lambda_ineq > _ACTIVE_DUAL_TOL * dual_scale)
    )
    session.x = x
    session.active_rows = active.astype(int, copy=False)
    session.inequality_multipliers = lambda_ineq
    session.certified = True
    session.last_seed_active_count = int(active.size)


def _constraint_scales(rows, rhs, x):
    if _row_count(rows) == 0:
        return np.empty(0, dtype=float)
    if sparse.issparse(rows):
        row_norms = np.asarray(np.abs(rows).sum(axis=1)).reshape(-1)
    else:
        row_norms = np.sum(np.abs(np.asarray(rows, dtype=float)), axis=1)
    return 1.0 + np.abs(rhs) + row_norms * max(
        1.0, np.linalg.norm(x, ord=np.inf)
    )


def _row_count(rows):
    return 0 if rows is None else int(rows.shape[0])


def _vector(value):
    if value is None:
        return np.empty(0, dtype=float)
    return np.asarray(cvxopt_to_numpy_matrix(value), dtype=float).reshape(-1)


def _dense_rows(rows, indices=None):
    if sparse.issparse(rows) and indices is not None:
        # COO matrices do not implement row slicing.  Canonical constraints
        # may retain COO storage from the historical bound assembly, so use a
        # row-addressable view only for the small active working set.
        selected = rows.tocsr()[indices]
    else:
        selected = rows if indices is None else rows[indices]
    if sparse.issparse(selected):
        selected = selected.toarray()
    return np.asarray(selected, dtype=float)


def _matvec(rows, vector):
    return np.asarray(rows @ vector, dtype=float).reshape(-1)


def _failed(reason, *, fatal, repairs=0, certificate=None):
    result = {
        "success": False,
        "reason": reason,
        "fatal": fatal,
        "repairs": repairs,
    }
    if certificate is not None:
        result["certificate"] = certificate
    return result
