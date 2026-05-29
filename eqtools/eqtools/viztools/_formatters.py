"""
_formatters.py — Matplotlib tick formatters for geographic and degree axes.

Public API
----------
DegreeFormatter      : Tick formatter appending the degree symbol
set_degree_formatter : Convenience function to apply DegreeFormatter to an axis
LatFormatter         : Tick formatter with degrees N/S suffix (latitude)
LonFormatter         : Tick formatter with degrees E/W suffix (longitude)
DMSFormatter         : Tick formatter for degrees-minutes-seconds format
"""

import matplotlib as mpl
from typing import Literal


class DegreeFormatter(mpl.ticker.ScalarFormatter):
    """Tick formatter that appends the degree symbol (°) to each label."""

    def __call__(self, x, pos=None):
        label = super().__call__(x, pos)
        return label + '\u00b0'


def set_degree_formatter(ax, axis='both'):
    """Apply :class:`DegreeFormatter` to one or both axes of *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    axis : {'x', 'y', 'both'}
        Which axis to format (default ``'both'``).

    Example
    -------
    >>> set_degree_formatter(ax, axis='both')   # both axes show 45°
    >>> set_degree_formatter(ax, axis='x')      # only x-axis
    """
    formatter = DegreeFormatter()
    if axis in ['x', 'both']:
        ax.xaxis.set_major_formatter(formatter)
    if axis in ['y', 'both']:
        ax.yaxis.set_major_formatter(formatter)


def _decimal_to_dms(decimal_deg: float, precision: int = 0) -> tuple:
    """Convert decimal degrees to degrees, minutes, seconds.

    Parameters
    ----------
    decimal_deg : float
        Decimal degrees value.
    precision : int
        Number of decimal places for seconds (default 0).

    Returns
    -------
    tuple
        (degrees, minutes, seconds) as integers/floats.
    """
    abs_deg = abs(decimal_deg)
    degrees = int(abs_deg)
    minutes_float = (abs_deg - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60

    if precision == 0:
        seconds = int(round(seconds))
        # Handle rounding overflow
        if seconds == 60:
            seconds = 0
            minutes += 1
        if minutes == 60:
            minutes = 0
            degrees += 1
    else:
        seconds = round(seconds, precision)
        if seconds >= 60:
            seconds = 0
            minutes += 1
        if minutes >= 60:
            minutes = 0
            degrees += 1

    return degrees, minutes, seconds


class _GeoFormatterBase(mpl.ticker.FuncFormatter):
    """Base class for geographic coordinate formatters.

    Provides shared logic for DMS and decimal formatting with hemisphere labels.
    Subclasses should define:
    - _positive_suffix: string shown for positive values
    - _negative_suffix: string shown for negative values
    - _is_zero_neutral(x): whether the value should be shown without suffix
    """

    _positive_suffix: str = ''
    _negative_suffix: str = ''

    def __init__(
        self,
        decimal_places: int = 0,
        format: Literal['decimal', 'dms'] = 'decimal',
        dms_precision: int = 0
    ):
        self.format = format
        self.dms_precision = dms_precision
        self.decimal_places = decimal_places

        if format == 'dms':
            super().__init__(self._create_dms_formatter(dms_precision))
        else:
            super().__init__(self._create_decimal_formatter(decimal_places))

    def _is_zero_neutral(self, x: float) -> bool:
        """Check if value should be shown without hemisphere suffix."""
        return x == 0

    def _create_dms_formatter(self, precision: int):
        """Create DMS format function."""
        def formatter(x, _):
            if self._is_zero_neutral(x):
                d, m, s = _decimal_to_dms(abs(x), precision)
                if precision > 0:
                    return f'{d}\u00b0{m}\'{s:.{precision}f}"'
                return f'{d}\u00b0{m}\'{s}"'

            d, m, s = _decimal_to_dms(abs(x), precision)
            suffix = self._positive_suffix if x > 0 else self._negative_suffix
            if precision > 0:
                return f'{d}\u00b0{m}\'{s:.{precision}f}"{suffix}'
            return f'{d}\u00b0{m}\'{s}"{suffix}'
        return formatter

    def _create_decimal_formatter(self, decimal_places: int):
        """Create decimal format function."""
        fmt = f'{{:.{decimal_places}f}}'
        def formatter(x, _):
            abs_val = fmt.format(abs(x))
            if x > 0:
                return f'{abs_val}\u00b0{self._positive_suffix}'
            elif x < 0:
                return f'{abs_val}\u00b0{self._negative_suffix}'
            else:
                return f'{abs_val}\u00b0'
        return formatter


class LatFormatter(_GeoFormatterBase):
    """Tick formatter that appends °N / °S to latitude values.

    Parameters
    ----------
    decimal_places : int
        Number of decimal places for decimal degrees (default 0 = integer).
    format : {'decimal', 'dms'}
        Output format: 'decimal' for decimal degrees, 'dms' for degrees-minutes-seconds.
    dms_precision : int
        Number of decimal places for seconds in DMS format (default 0).

    Example
    -------
    >>> ax.yaxis.set_major_formatter(LatFormatter())
    # -> 21°N, 20°S, 0°
    >>> ax.yaxis.set_major_formatter(LatFormatter(format='dms'))
    # -> 21°30'15"N, 20°45'30"S
    """

    _positive_suffix = 'N'
    _negative_suffix = 'S'


class LonFormatter(_GeoFormatterBase):
    """Tick formatter that appends °E / °W to longitude values.

    Parameters
    ----------
    decimal_places : int
        Number of decimal places for decimal degrees (default 0 = integer).
    format : {'decimal', 'dms'}
        Output format: 'decimal' for decimal degrees, 'dms' for degrees-minutes-seconds.
    dms_precision : int
        Number of decimal places for seconds in DMS format (default 0).

    Example
    -------
    >>> ax.xaxis.set_major_formatter(LonFormatter())
    # -> 96°E, 180°, 181°W (same as -179°E)
    >>> ax.xaxis.set_major_formatter(LonFormatter(format='dms'))
    # -> 96°30'15"E, 180°0'0", 179°30'0"W
    """

    _positive_suffix = 'E'
    _negative_suffix = 'W'

    def _is_zero_neutral(self, x: float) -> bool:
        """0 and 180/-180 are shown without E/W suffix."""
        return x == 0 or abs(x) == 180


class DMSFormatter(mpl.ticker.FuncFormatter):
    """Tick formatter for degrees-minutes-seconds format (no hemisphere suffix).

    Parameters
    ----------
    precision : int
        Number of decimal places for seconds (default 0).
    show_sign : bool
        If True, show + or - sign for positive/negative values (default False).

    Example
    -------
    >>> ax.xaxis.set_major_formatter(DMSFormatter())
    # -> 45°30'15", -12°15'30"
    >>> ax.xaxis.set_major_formatter(DMSFormatter(precision=2, show_sign=True))
    # -> +45°30'15.50", -12°15'30.25"
    """

    def __init__(self, precision: int = 0, show_sign: bool = False):
        self.precision = precision
        self.show_sign = show_sign

        def formatter(x, _):
            d, m, s = _decimal_to_dms(abs(x), precision)
            sign = ''
            if show_sign:
                sign = '+' if x >= 0 else '-'
            elif x < 0:
                sign = '-'

            if precision > 0:
                return f'{sign}{d}\u00b0{m}\'{s:.{precision}f}"'
            return f'{sign}{d}\u00b0{m}\'{s}"'

        super().__init__(formatter)
