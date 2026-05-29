"""
Utilities for geodetic data downsampling workflows.

This package is intentionally lightweight: configuration normalization and
validation can be imported without a working CSI installation. Processing
functions that need CSI live in the CLI layer for now.
"""

from .config import (
    AMPLITUDE_STAT_CHOICES,
    DEFAULT_OPTICAL_CONFIG,
    DEFAULT_SAR_CONFIG,
    COVAR_MISSING_POLICY_CHOICES,
    DEFAULT_PROCESSING_REGION,
    DOWNSAMPLE_METHOD_CHOICES,
    FOCUS_REGION_COORD_CHOICES,
    FAULT_MODEL_GEOMETRY_CHOICES,
    FAULT_MODEL_TYPE_CHOICES,
    FAULT_MODEL_USE_CHOICES,
    FAULT_PLOT_STAGE_CHOICES,
    ORIGIN_MODE_CHOICES,
    PROCESSING_REGION_GEOMETRY_CHOICES,
    SAR_MODE_CHOICES,
    SAR_READER_ALIASES,
    SAR_OUTPUT_SUFFIX_BY_MODE,
    SAR_READER_CHOICES,
    SPLIT_METRIC_CHOICES,
    compact_kwargs,
    deep_update,
    get_sar_output_name,
    normalize_downsample_config,
    normalize_optical_config,
    normalize_sar_config,
    validate_optical_config,
    validate_downsample_config,
    validate_sar_config,
)
from .grid_template import (
    RSP_GEOMETRY_CHOICES,
    RspGridTemplate,
    apply_rsp_grid_template,
    read_rsp_grid_template,
)
from .plotting import plot_decimated_geodata
from .processing_region import (
    apply_processing_region,
    format_processing_region_report,
    processing_region_report_file,
)

__all__ = [
    "DEFAULT_SAR_CONFIG",
    "DEFAULT_OPTICAL_CONFIG",
    "AMPLITUDE_STAT_CHOICES",
    "COVAR_MISSING_POLICY_CHOICES",
    "DEFAULT_PROCESSING_REGION",
    "DOWNSAMPLE_METHOD_CHOICES",
    "FOCUS_REGION_COORD_CHOICES",
    "FAULT_MODEL_GEOMETRY_CHOICES",
    "FAULT_MODEL_TYPE_CHOICES",
    "FAULT_MODEL_USE_CHOICES",
    "FAULT_PLOT_STAGE_CHOICES",
    "ORIGIN_MODE_CHOICES",
    "PROCESSING_REGION_GEOMETRY_CHOICES",
    "RSP_GEOMETRY_CHOICES",
    "SAR_MODE_CHOICES",
    "SAR_READER_ALIASES",
    "SAR_OUTPUT_SUFFIX_BY_MODE",
    "SAR_READER_CHOICES",
    "SPLIT_METRIC_CHOICES",
    "RspGridTemplate",
    "apply_processing_region",
    "apply_rsp_grid_template",
    "compact_kwargs",
    "deep_update",
    "format_processing_region_report",
    "get_sar_output_name",
    "normalize_downsample_config",
    "normalize_optical_config",
    "normalize_sar_config",
    "processing_region_report_file",
    "read_rsp_grid_template",
    "validate_downsample_config",
    "validate_optical_config",
    "validate_sar_config",
    "plot_decimated_geodata",
]
