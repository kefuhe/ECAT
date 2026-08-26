"""Shared likelihood-tempering utilities for the SMC backends.

The current particle population represents ``pi_beta_old``.  A candidate
temperature therefore has incremental weights

``w_i(beta) = exp((beta - beta_old) * log_likelihood_i)``.

``beta_old`` is immutable during the search.  Keeping that reference separate
from the mutable bisection bounds prevents the bracket width from being used as
the likelihood exponent.
"""

from __future__ import annotations

import numpy as np


def select_next_beta_by_cov(
    log_likelihoods,
    beta_old,
    *,
    target_cov=1.0,
    max_delta_beta=0.5,
    tolerance=1.0e-6,
    ddof=1,
):
    """Select the next SMC temperature from an incremental-weight COV target.

    Parameters
    ----------
    log_likelihoods : array-like
        Log likelihood values of the particles representing ``pi_beta_old``.
        Any common additive constant is immaterial to the normalized weights.
    beta_old : float
        Fixed temperature of the current particle population.
    target_cov : float, optional
        Target coefficient of variation of the incremental weights.
    max_delta_beta : float, optional
        Maximum temperature increment considered in one stage.  The historical
        ECAT/Dutta policy limits this to 0.5.
    tolerance : float, optional
        Absolute tolerance of the beta bisection bracket.
    ddof : int, optional
        Delta degrees of freedom used by ``numpy.std``.  Active ECAT paths use
        ``ddof=1``; the deprecated original path retains ``ddof=0``.

    Returns
    -------
    float
        ``beta_new`` in ``(beta_old, 1]`` unless ``beta_old`` is already 1.

    Notes
    -----
    If the largest permitted step already has COV no greater than the target,
    it is accepted directly.  Otherwise bisection solves

    ``COV(exp((beta - beta_old) * centered_log_likelihood)) = target_cov``.
    """
    beta_old = float(beta_old)
    target_cov = float(target_cov)
    max_delta_beta = float(max_delta_beta)
    tolerance = float(tolerance)

    if not 0.0 <= beta_old <= 1.0:
        raise ValueError("beta_old must lie in [0, 1].")
    if beta_old == 1.0:
        return 1.0
    if target_cov < 0.0:
        raise ValueError("target_cov must be non-negative.")
    if max_delta_beta <= 0.0:
        raise ValueError("max_delta_beta must be positive.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    values = np.asarray(log_likelihoods, dtype=float).reshape(-1)
    if values.size <= ddof:
        raise ValueError(
            "log_likelihoods must contain more values than the requested ddof."
        )
    if (
        np.isnan(values).any()
        or np.isposinf(values).any()
        or not np.isfinite(values).any()
    ):
        raise ValueError(
            "log_likelihoods must contain at least one finite value and no "
            "NaNs or positive infinity."
        )

    centered = values - np.max(values)

    def weight_cov(beta_candidate):
        delta_beta = beta_candidate - beta_old
        weights = np.exp(delta_beta * centered)
        return np.std(weights, ddof=ddof) / np.mean(weights)

    beta_low = beta_old
    beta_high = min(beta_old + max_delta_beta, 1.0)

    # Do not add a needless intermediate stage when the full permitted step is
    # already at or below the particle-degeneracy target.
    if weight_cov(beta_high) <= target_cov:
        return beta_high

    while beta_high - beta_low > tolerance:
        beta_candidate = 0.5 * (beta_low + beta_high)
        if weight_cov(beta_candidate) > target_cov:
            beta_high = beta_candidate
        else:
            beta_low = beta_candidate

    return 0.5 * (beta_low + beta_high)
