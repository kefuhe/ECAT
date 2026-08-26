"""Rank-zero progress reporting for long-running SMC jobs.

The reporter observes stage boundaries only.  It deliberately has no access
to sampler state, random-number generators, or MPI communicators, so changing
the presentation cannot change the numerical or collective execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import sys
import time
import warnings


_AUTO_TERMINAL = object()


def _format_duration(seconds):
    """Format wall time using the compact Slurm-style ``[DD-]HH:MM:SS`` form."""
    total_seconds = max(0, int(round(float(seconds))))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}-{clock}" if days else clock


def _windows_forwarded_stdout_is_interactive(environ=None):
    """Return whether a Windows MPI pipe is attached to a known terminal host.

    Microsoft MPI forwards child stdout through a pipe, so ``isatty()`` is
    false even in Windows Terminal or the VS Code integrated terminal.  In
    those hosts carriage returns are forwarded correctly and stdout is the
    only reliable visible sink.  ``CONOUT$`` is deliberately not used: an MPI
    child can open it successfully without the handle being visible in the
    parent ConPTY session, which would silently discard the entire progress
    table.

    This is a presentation capability check only.  An uncertain environment
    returns ``False`` and therefore uses completion-only stdout records.
    """
    environ = os.environ if environ is None else environ
    if environ.get("CI") or environ.get("TERM", "").lower() == "dumb":
        return False

    if any(
        environ.get(name)
        for name in (
            "WT_SESSION",
            "VSCODE_PID",
            "PYCHARM_HOSTED",
            "ConEmuANSI",
            "ANSICON",
        )
    ):
        return True

    return environ.get("TERM_PROGRAM", "").lower() in {
        "vscode",
        "wezterm",
        "hyper",
    }


def _open_control_terminal(stream):
    """Return a verified live progress sink or ``None`` for durable stdout.

    MPI launchers normally replace rank stdout with a pipe.  POSIX ranks may
    still have a controlling ``/dev/tty``.  Windows MPI ranks instead keep
    using forwarded stdout only when a known interactive host is present;
    otherwise they safely fall back to one completed stdout record per stage.
    """
    if stream is not sys.stdout:
        return None, False

    isatty = getattr(stream, "isatty", None)
    if isatty is not None and isatty():
        return stream, False

    if os.name == "nt":
        if _windows_forwarded_stdout_is_interactive():
            return stream, False
        return None, False

    try:
        terminal = open(  # noqa: SIM115 - owned for the reporter lifetime
            "/dev/tty",
            mode="w",
            encoding=getattr(stream, "encoding", None) or "utf-8",
            errors="replace",
            buffering=1,
        )
    except OSError:
        return None, False
    return terminal, True


@dataclass
class _ActiveStage:
    stage: int
    beta_previous: float
    beta_current: float
    started_at: datetime
    started_tick: float


class SMCProgressReporter:
    """Render one compact rank-zero table without touching sampler state.

    When a verified live terminal exists, the header, transient row, completed
    rows, and final summary all use that one stream.  Windows MPI uses its
    forwarded stdout for known ConPTY hosts instead of opening ``CONOUT$``.
    Headless or uncertain jobs use stdout and emit completion-only records, so
    progress can never disappear and redirected logs contain no carriage-return
    controls.  No user-facing display option is required.

    The class performs no MPI calls and owns no sampler or random state.
    """

    _HEADER = (
        "STAGE  BETA (FROM -> TO)        START     END       "
        "MCMC TIME   TOTAL"
    )

    def __init__(
        self,
        *,
        chains,
        chain_length,
        mpi_ranks,
        stream=None,
        terminal_stream=_AUTO_TERMINAL,
        timer=None,
        now=None,
    ):
        self.chains = int(chains)
        self.chain_length = int(chain_length)
        self.mpi_ranks = int(mpi_ranks)
        self.stream = stream if stream is not None else sys.stdout
        self._timer = timer if timer is not None else time.perf_counter
        self._now = now if now is not None else datetime.now
        if terminal_stream is _AUTO_TERMINAL:
            self._terminal_stream, self._owns_terminal_stream = (
                _open_control_terminal(self.stream)
            )
        else:
            self._terminal_stream = terminal_stream
            self._owns_terminal_stream = False
        self._run_started_tick = None
        self._run_started_at = None
        self._active = None
        self._prior_started_at = None
        self._prior_started_tick = None
        self._live_row = None
        self._live_row_width = 0
        self._warning_renderer = None
        self._previous_showwarning = None

    def start(self, *, resumed=False):
        """Start session timing and print the single table header."""
        self._run_started_tick = self._timer()
        self._run_started_at = self._now()
        mode = "resume" if resumed else "fresh"
        self._write_line(
            "ATMIP  "
            f"chains={self.chains}  chain_length={self.chain_length}  "
            f"mpi_ranks={self.mpi_ranks}  mode={mode}  "
            f"started={self._run_started_at:%Y-%m-%d %H:%M:%S}"
        )
        self._write_line(self._HEADER)

    def begin_prior(self):
        """Display evaluation of the initial beta-zero population."""
        self._prior_started_at = self._now()
        self._prior_started_tick = self._timer()
        self._draw_live_row(
            self._format_prior_row(end="--", elapsed="--") + "  PRIOR"
        )

    def complete_prior(self):
        """Complete the initial-population accounting row."""
        if self._prior_started_tick is None:
            raise RuntimeError("no active SMC prior progress row")
        ended_at = self._now()
        ended_tick = self._timer()
        self._record_completed_row(
            self._format_prior_row(
                end=f"{ended_at:%H:%M:%S}",
                elapsed=_format_duration(ended_tick - self._prior_started_tick),
            )
        )
        self._prior_started_at = None
        self._prior_started_tick = None

    def begin_stage(self, *, stage, beta_previous, beta_current):
        """Record and display a stage immediately before its MCMC mutation."""
        if self._active is not None:
            raise RuntimeError("an SMC progress stage is already active")
        self._active = _ActiveStage(
            stage=int(stage),
            beta_previous=float(beta_previous),
            beta_current=float(beta_current),
            started_at=self._now(),
            started_tick=self._timer(),
        )
        total = self._timer() - self._run_started_tick
        state = "FINAL MCMC" if self._active.beta_current == 1.0 else "MCMC"
        row = self._format_stage_row(
            self._active,
            end="--",
            stage_time="--",
            total=_format_duration(total),
        )
        self._draw_live_row(row + f"  {state}")

    def complete_stage(self):
        """Complete the active row after MCMC and existing MPI synchronization."""
        if self._active is None:
            raise RuntimeError("no active SMC progress stage")
        ended_at = self._now()
        ended_tick = self._timer()
        row = self._format_stage_row(
            self._active,
            end=f"{ended_at:%H:%M:%S}",
            stage_time=_format_duration(ended_tick - self._active.started_tick),
            total=_format_duration(ended_tick - self._run_started_tick),
        )
        self._record_completed_row(row)
        self._active = None

    def complete(self, *, stage, beta):
        """Print a final session summary after final-stage persistence."""
        if self._active is not None:
            raise RuntimeError("cannot complete SMC progress with an active stage")
        ended_at = self._now()
        elapsed = self._timer() - self._run_started_tick
        self._write_line(
            "ATMIP completed  "
            f"ended={ended_at:%Y-%m-%d %H:%M:%S}  "
            f"stage={int(stage)}  beta={float(beta):.6f}  "
            f"total={_format_duration(elapsed)}"
        )
        self._close_terminal_stream()

    def fail(self, message):
        """Close a possible live row and report a rank-zero initialization failure."""
        self._clear_live_row()
        self._write_line(f"ATMIP failed  time={self._now():%Y-%m-%d %H:%M:%S}  {message}")
        self._close_terminal_stream()

    def _format_prior_row(self, *, end, elapsed):
        total = _format_duration(self._timer() - self._run_started_tick)
        return (
            f"{1:>5d}  {'PRIOR -> 0.000000':<20}  "
            f"{self._prior_started_at:%H:%M:%S}  {end:>8}  "
            f"{elapsed:>10}  {total:>10}"
        )

    def _format_stage_row(self, active, *, end, stage_time, total):
        return (
            f"{active.stage:>5d}  "
            f"{active.beta_previous:>8.6f} -> {active.beta_current:<8.6f}  "
            f"{active.started_at:%H:%M:%S}  {end:>8}  "
            f"{stage_time:>10}  {total:>10}"
        )

    def _write_line(self, text):
        target = (
            self._terminal_stream
            if self._terminal_stream is not None
            else self.stream
        )
        target.write(text + "\n")
        target.flush()

    def _draw_live_row(self, row):
        """Draw one transient row directly on the controlling terminal."""
        self._live_row = row
        self._live_row_width = len(row)
        if self._terminal_stream is None:
            return
        self._install_warning_renderer()
        self._terminal_stream.write("\r" + row)
        self._terminal_stream.flush()

    def _clear_live_row(self):
        """Erase the transient row without emitting a durable log record."""
        self._erase_live_row()
        self._live_row = None
        self._live_row_width = 0
        self._restore_warning_renderer()

    def _erase_live_row(self):
        if self._terminal_stream is None or not self._live_row_width:
            return
        self._terminal_stream.write(
            "\r" + " " * self._live_row_width + "\r"
        )
        self._terminal_stream.flush()

    def _install_warning_renderer(self):
        """Keep Python warnings readable without abandoning the live row."""
        if self._warning_renderer is not None:
            return
        previous_showwarning = warnings.showwarning
        self._previous_showwarning = previous_showwarning

        def showwarning(message, category, filename, lineno, file=None, line=None):
            # A surrounding warnings.catch_warnings() may restore this wrapper
            # after the stage has already closed.  Delegate safely instead of
            # writing through a stale terminal handle.
            if (
                self._terminal_stream is None
                or self._warning_renderer is not showwarning
            ):
                return previous_showwarning(
                    message, category, filename, lineno, file=file, line=line
                )
            live_row = self._live_row
            self._erase_live_row()
            warning_text = warnings.formatwarning(
                message, category, filename, lineno, line=line
            )
            self._terminal_stream.write(warning_text)
            self._terminal_stream.flush()
            if live_row is not None:
                self._terminal_stream.write("\r" + live_row)
                self._terminal_stream.flush()

        self._warning_renderer = showwarning
        warnings.showwarning = showwarning

    def _restore_warning_renderer(self):
        if self._warning_renderer is None:
            return
        if warnings.showwarning is self._warning_renderer:
            warnings.showwarning = self._previous_showwarning
        self._warning_renderer = None
        self._previous_showwarning = None

    def _record_completed_row(self, row):
        """Replace the live display with one immutable record on its sink."""
        self._clear_live_row()
        self._write_line(row)

    def _close_terminal_stream(self):
        self._restore_warning_renderer()
        if self._owns_terminal_stream and self._terminal_stream is not None:
            self._terminal_stream.close()
        self._terminal_stream = None
        self._owns_terminal_stream = False
