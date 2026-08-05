"""Thin public facade for reusable ECAT figure products."""

from __future__ import annotations

from .figure_products import (
    plot_data_fits_product,
    plot_deep_slip_loading_summary_product,
    plot_fault_fields_product,
    plot_interseismic_summary_product,
)


class FigureProductMixin:
    """High-level plotting entry points that delegate to existing plot methods.

    The mixin is intentionally thin: it does not build Green's functions, alter
    solver matrices, or replace CSI's native ``fault.plot`` / ``data.plot``
    methods.  It only organizes common figure products used in inversion
    scripts.
    """

    def plot_data_fits(self, **kwargs):
        """Plot observed/synthetic fit products for configured geodetic data."""
        return plot_data_fits_product(self, **kwargs)

    def plot_fault_fields(self, **kwargs):
        """Plot standard slip fields on one or more faults."""
        return plot_fault_fields_product(self, **kwargs)

    def plot_interseismic_summary(self, **kwargs):
        """Plot a standard bundle of Euler/block interseismic fields."""
        return plot_interseismic_summary_product(self, **kwargs)

    def plot_deep_slip_loading_summary(self, **kwargs):
        """Plot a standard bundle of deep-slip loading proxy fields."""
        return plot_deep_slip_loading_summary_product(self, **kwargs)
