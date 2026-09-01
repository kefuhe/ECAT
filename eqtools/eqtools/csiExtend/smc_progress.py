"""Rank-zero progress reporting for long-running SMC jobs.

The reporter observes stage boundaries only.  It deliberately has no access
to sampler state, random-number generators, or MPI communicators, so changing
the presentation cannot change the numerical or collective execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import sys
import time


def _format_duration(seconds):
    """Format wall time using the compact Slurm-style ``[DD-]HH:MM:SS`` form."""
    total_seconds = max(0, int(round(float(seconds))))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}-{clock}" if days else clock


@dataclass
class _ActiveOperation:
    """One timed operation whose completion will be reported by the next row."""

    label: str
    beta_text: str
    started_at: datetime
    started_tick: float


class SMCProgressReporter:
    """Append one compact rank-zero state table to ``stdout``.

    Every row is an immutable state transition: it says what ATMIP is doing
    now and how long the immediately preceding operation took.  Completion of
    an operation is therefore recorded by the next row, rather than by a
    second line for the same stage.  This keeps long MPI output readable while
    still leaving the current operation as the last visible line.

    The reporter uses one stream, writes no carriage returns or ANSI controls,
    and flushes every record.  Consequently the same output is suitable for a
    normal terminal, MPI-forwarded stdout, redirected files, and batch logs.
    The class performs no MPI calls and owns no sampler or random state.
    """

    _HEADER = (
        "TIME      ELAPSED    STATUS  CURRENT   "
        "BETA (FROM -> TO)       PREVIOUS / DETAIL"
    )

    def __init__(
        self,
        *,
        chains,
        chain_length,
        mpi_ranks,
        target_cov=1.0,
        max_delta_beta=0.5,
        stream=None,
        timer=None,
        now=None,
    ):
        self.chains = int(chains)
        self.chain_length = int(chain_length)
        self.mpi_ranks = int(mpi_ranks)
        self.target_cov = float(target_cov)
        self.max_delta_beta = float(max_delta_beta)
        self.stream = stream if stream is not None else sys.stdout
        self._timer = timer if timer is not None else time.perf_counter
        self._now = now if now is not None else datetime.now
        self._run_started_tick = None
        self._run_started_at = None
        self._active = None
        self._last_completed_label = None
        self._last_completed_duration = None

    def start(self, *, resumed=False):
        """Start session timing and print the single table header."""
        if self._run_started_tick is not None:
            raise RuntimeError("SMC progress reporting has already started")
        self._run_started_tick = self._timer()
        self._run_started_at = self._now()
        if resumed:
            self._last_completed_label = "CHECKPOINT"

        # Flush verbose setup messages already written to the same stdout
        # before introducing the progress table.  This is important when an
        # MPI launcher turns stdout into a block-buffered pipe; users should
        # not need ``python -u`` merely to see initialization output.
        self.stream.flush()
        mode = "resume" if resumed else "fresh"
        self._write_line(
            "ATMIP  "
            f"mode={mode}  chains={self.chains}x{self.chain_length}  "
            f"mpi_ranks={self.mpi_ranks}  "
            f"target_cov={self.target_cov:g}  "
            f"max_delta_beta={self.max_delta_beta:g}  "
            f"started={self._run_started_at:%Y-%m-%d %H:%M:%S}"
        )
        self._write_line(self._HEADER)

    def begin_prior(self):
        """Record evaluation of the initial beta-zero population as current."""
        self._begin_operation(
            label="PRIOR",
            beta_text="initial population",
        )

    def complete_prior(self):
        """Store prior-evaluation duration for the next state row."""
        self._complete_operation(expected_label="PRIOR")

    def begin_stage(self, *, stage, beta_previous, beta_current):
        """Record a stage immediately before its existing MCMC mutation."""
        self._begin_operation(
            label=f"STAGE {int(stage):02d}",
            beta_text=(
                f"{float(beta_previous):.6f} -> "
                f"{float(beta_current):.6f}"
            ),
        )

    def complete_stage(self):
        """Store active-stage duration after existing MPI synchronization."""
        if self._active is None or not self._active.label.startswith("STAGE "):
            raise RuntimeError("no active SMC progress stage")
        self._complete_operation(expected_label=self._active.label)

    def complete(self, *, stage, beta):
        """Print the final state after final-stage persistence."""
        self._require_started()
        if self._active is not None:
            raise RuntimeError("cannot complete SMC progress with an active stage")
        self._write_state(
            status="DONE",
            current="ATMIP",
            beta_text=f"stage={int(stage)} beta={float(beta):.6f}",
        )

    def fail(self, message):
        """Print a final failure state without altering the sampler exception."""
        self._require_started()
        detail = " ".join(str(message).split())
        if self._active is None:
            current = "ATMIP"
            beta_text = "-"
            failure_detail = detail
        else:
            ended_tick = self._timer()
            current = self._active.label
            beta_text = self._active.beta_text
            failure_duration = ended_tick - self._active.started_tick
            failure_detail = (
                f"after {_format_duration(failure_duration)}; {detail}"
            )
            self._active = None
        self._write_state(
            status="FAILED",
            current=current,
            beta_text=beta_text,
            previous_text=failure_detail,
        )

    def _begin_operation(self, *, label, beta_text):
        self._require_started()
        if self._active is not None:
            raise RuntimeError("an SMC progress operation is already active")
        self._active = _ActiveOperation(
            label=label,
            beta_text=beta_text,
            started_at=self._now(),
            started_tick=self._timer(),
        )
        self._write_state(
            status="RUN",
            current=label,
            beta_text=beta_text,
            at=self._active.started_at,
        )

    def _complete_operation(self, *, expected_label):
        if self._active is None or self._active.label != expected_label:
            if expected_label == "PRIOR":
                raise RuntimeError("no active SMC prior progress operation")
            raise RuntimeError("no matching active SMC progress operation")
        ended_tick = self._timer()
        self._last_completed_label = self._active.label
        self._last_completed_duration = ended_tick - self._active.started_tick
        self._active = None

    def _write_state(
        self,
        *,
        status,
        current,
        beta_text,
        at=None,
        previous_text=None,
    ):
        at = self._now() if at is None else at
        elapsed = _format_duration(self._timer() - self._run_started_tick)
        self._write_line(
            f"{at:%H:%M:%S}  {elapsed:>10}  {status:<6}  "
            f"{current:<9}  {beta_text:<22}  "
            f"{self._previous_text() if previous_text is None else previous_text}"
        )

    def _previous_text(self):
        if self._last_completed_label is None:
            return "-"
        if self._last_completed_duration is None:
            return self._last_completed_label
        return (
            f"{self._last_completed_label} "
            f"{_format_duration(self._last_completed_duration)}"
        )

    def _require_started(self):
        if self._run_started_tick is None:
            raise RuntimeError("SMC progress reporting has not started")

    def _write_line(self, text):
        self.stream.write(text + "\n")
        self.stream.flush()
