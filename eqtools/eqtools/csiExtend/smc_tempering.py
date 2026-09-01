"""Shared likelihood-tempering utilities for the SMC backends.

The current particle population represents ``pi_beta_old``.  A candidate
temperature therefore has incremental weights

``w_i(beta) = exp((beta - beta_old) * log_likelihood_i)``.

``beta_old`` is immutable during the search.  Keeping that reference separate
from the mutable bisection bounds prevents the bracket width from being used as
the likelihood exponent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


_PUBLIC_TEMPERING_FIELDS = frozenset(
    {"target_cov", "max_delta_beta"}
)
_TEMPERING_METADATA_FIELDS = {
    "smc_tempering_target_cov": "target_cov",
    "smc_tempering_max_delta_beta": "max_delta_beta",
    "smc_tempering_tolerance": "tolerance",
    "smc_tempering_ddof": "ddof",
}


def _finite_float(value, *, name):
    """Normalize one real policy value while rejecting bools and NaNs."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number, not bool.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number.") from error
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class SMCTemperingPolicy:
    """Immutable run-level policy for COV-controlled SMC temperatures.

    Only ``target_cov`` and ``max_delta_beta`` belong to the public config.
    ``tolerance`` and ``ddof`` are recorded for reproducibility but remain
    internal numerical/statistical conventions.  Keeping all four values in
    one frozen object prevents generic and nonlinear MPI backends from
    silently drifting to different schedules.
    """

    target_cov: float = 1.0
    max_delta_beta: float = 0.5
    tolerance: float = 1.0e-6
    ddof: int = 1

    def __post_init__(self):
        target_cov = _finite_float(self.target_cov, name="target_cov")
        max_delta_beta = _finite_float(
            self.max_delta_beta,
            name="max_delta_beta",
        )
        tolerance = _finite_float(self.tolerance, name="tolerance")
        if isinstance(self.ddof, (bool, np.bool_)) or not isinstance(
            self.ddof, (int, np.integer)
        ):
            raise ValueError("ddof must be an integer.")
        ddof = int(self.ddof)

        if target_cov <= 0.0:
            raise ValueError("target_cov must be positive.")
        if not 0.0 < max_delta_beta <= 1.0:
            raise ValueError("max_delta_beta must lie in (0, 1].")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        if max_delta_beta <= tolerance:
            raise ValueError(
                "max_delta_beta must be greater than tolerance."
            )
        if ddof not in (0, 1):
            raise ValueError("ddof must be 0 or 1.")

        object.__setattr__(self, "target_cov", target_cov)
        object.__setattr__(self, "max_delta_beta", max_delta_beta)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "ddof", ddof)

    @classmethod
    def from_public_config(cls, value=None):
        """Resolve the public mapping without exposing internal conventions."""
        if value is None:
            return DEFAULT_SMC_TEMPERING_POLICY
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(
                "smc_tempering must be a mapping containing target_cov and/or "
                "max_delta_beta."
            )
        unknown = set(value) - _PUBLIC_TEMPERING_FIELDS
        if unknown:
            fields = ", ".join(sorted(unknown))
            raise ValueError(
                "Unknown public smc_tempering field(s): "
                f"{fields}. Supported fields are target_cov and "
                "max_delta_beta."
            )
        return cls(
            target_cov=value.get("target_cov", 1.0),
            max_delta_beta=value.get("max_delta_beta", 0.5),
        )

    def as_public_config(self):
        """Return the stable user-facing YAML/JSON representation."""
        return {
            "target_cov": self.target_cov,
            "max_delta_beta": self.max_delta_beta,
        }

    def as_metadata(self):
        """Return complete scalar metadata for checkpoints and final results."""
        return {
            metadata_name: getattr(self, field_name)
            for metadata_name, field_name in _TEMPERING_METADATA_FIELDS.items()
        }

    def select_next_beta(self, log_likelihoods, beta_old, *, ddof=None):
        """Apply this policy through the shared pure beta selector."""
        return select_next_beta_by_cov(
            log_likelihoods,
            beta_old,
            target_cov=self.target_cov,
            max_delta_beta=self.max_delta_beta,
            tolerance=self.tolerance,
            ddof=self.ddof if ddof is None else ddof,
        )


DEFAULT_SMC_TEMPERING_POLICY = SMCTemperingPolicy()


def resolve_smc_tempering_policy(value=None):
    """Resolve a policy object or its public mapping to one frozen policy."""
    return SMCTemperingPolicy.from_public_config(value)


def write_smc_tempering_metadata(attributes, policy):
    """Write policy scalars to an HDF5-like attribute mapping."""
    resolved = resolve_smc_tempering_policy(policy)
    for key, value in resolved.as_metadata().items():
        attributes[key] = value


def read_smc_tempering_metadata(attributes):
    """Read complete policy metadata, returning ``None`` for legacy files."""
    present = [key in attributes for key in _TEMPERING_METADATA_FIELDS]
    if not any(present):
        return None
    if not all(present):
        missing = [
            key
            for key, exists in zip(_TEMPERING_METADATA_FIELDS, present)
            if not exists
        ]
        raise ValueError(
            "Incomplete SMC tempering metadata; missing: "
            + ", ".join(missing)
        )
    values = {
        field_name: attributes[metadata_name]
        for metadata_name, field_name in _TEMPERING_METADATA_FIELDS.items()
    }
    return SMCTemperingPolicy(**values)


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
