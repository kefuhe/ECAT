"""Structured, rank-neutral diagnostics produced while resolving config.

Configuration parsing is a library operation: it records deterministic
normalization decisions here instead of writing through ``logging`` or
``warnings`` at the point of discovery. The owning config object renders
pending records only at explicit lifecycle boundaries. Rendering uses one
plain-text ``stdout`` block so MPI-forwarded output, normal terminals, and
redirected logs preserve the same startup order.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class ConfigDiagnostic:
    """One deterministic, non-fatal configuration decision."""

    severity: str
    code: str
    message: str
    field: str | None = None

    def format(self) -> str:
        """Return the compact, stable console representation."""
        location = f"{self.field}: " if self.field else ""
        return (
            f"CONFIG  {self.severity:<4}  {self.code:<30}  "
            f"{location}{self.message}"
        ).rstrip()


class ConfigDiagnostics:
    """Collect diagnostics and render each record at most once.

    Records remain available after reporting for inspection and tests. A
    cursor, rather than a single boolean, lets later preflight stages append
    new diagnostics without repeating messages already shown.
    """

    _SEVERITY_LABELS = {
        "warning": "WARN",
        "notice": "NOTE",
    }

    def __init__(self):
        self._records = []
        self._reported_count = 0

    @property
    def records(self):
        """Return an immutable view of all recorded diagnostics."""
        return tuple(self._records)

    @property
    def pending(self):
        """Return diagnostics not yet emitted by :meth:`report`."""
        return tuple(self._records[self._reported_count :])

    def add(self, code, message, *, field=None, severity="warning"):
        """Record one diagnostic, suppressing exact duplicates."""
        try:
            label = self._SEVERITY_LABELS[severity]
        except KeyError as exc:
            raise ValueError(
                "config diagnostic severity must be 'warning' or 'notice'"
            ) from exc

        diagnostic = ConfigDiagnostic(
            severity=label,
            code=str(code),
            message=" ".join(str(message).split()),
            field=None if field is None else str(field),
        )
        if diagnostic not in self._records:
            self._records.append(diagnostic)
        return diagnostic

    def report(self, *, enabled, stream=None):
        """Write pending diagnostics as one flushed static stdout block."""
        if not enabled:
            return ()
        pending = self.pending
        if not pending:
            return ()

        output = stream if stream is not None else sys.stdout
        output.write("\n".join(item.format() for item in pending) + "\n")
        output.flush()
        self._reported_count = len(self._records)
        return pending
