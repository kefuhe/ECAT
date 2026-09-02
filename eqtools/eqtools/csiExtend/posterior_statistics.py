"""Small result objects for posterior linear-field statistics.

The Bayesian inversion owns geometry reconstruction and conditional solves.
This module deliberately owns only streaming vector moments and the immutable
summary returned to reporting code, keeping solver and plotting concerns out
of the statistical accumulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PosteriorSlipStatistics:
    """One component-wise posterior statistic in canonical linear layout.

    ``values`` contains the complete linear result layout (source parameters
    followed by polynomial parameters), not geometry or scale parameters.
    ``definition`` distinguishes sampled FULLSMC slip from the dispersion of
    SMC-FJ conditional optimizers.
    """

    statistic: str
    values: np.ndarray
    sample_count: int
    ddof: int
    definition: str
    elapsed_seconds: float
    solver_diagnostics: dict = field(default_factory=dict)


class OnlineVectorMoments:
    """Accumulate vector mean and variance without storing all solutions."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, values):
        values = np.asarray(values, dtype=float).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("posterior linear solution must be finite")
        if self.mean is None:
            self.mean = np.zeros_like(values)
            self.m2 = np.zeros_like(values)
        elif values.size != self.mean.size:
            raise ValueError(
                "posterior linear-solution width changed during accumulation"
            )

        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (values - self.mean)

    def standard_deviation(self, *, ddof=0):
        ddof = int(ddof)
        if ddof < 0:
            raise ValueError("ddof must be non-negative")
        if self.count <= ddof:
            raise ValueError(
                f"cannot compute standard deviation from {self.count} "
                f"samples with ddof={ddof}"
            )
        return np.sqrt(np.maximum(self.m2 / (self.count - ddof), 0.0))
