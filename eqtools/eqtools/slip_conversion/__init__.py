"""
Slip model format conversion utilities.

This package provides tools for converting between different slip model formats
commonly used in earthquake modeling and inversion.

Supported formats:
- JinInv (MATLAB .mat files)
- SDM (ASCII .dat files)
- GMT (output format)

Example usage:
    >>> from slip_conversion import JinInvSlipConverter, SDMSlipConverter
    >>> 
    >>> # Convert JinInv format
    >>> jininv_converter = JinInvSlipConverter('slip.mat')
    >>> jininv_converter.convert_all()
    >>> 
    >>> # Convert SDM format
    >>> sdm_converter = SDMSlipConverter('slip.dat')
    >>> sdm_converter.convert_all()
"""

from .base_converter import BaseSlipConverter
from .jininv_converter import JinInvSlipConverter
from .tga_converter import TGASlipConverter
from .sdm_converter import SDMSlipConverter
from .yueinv_converter import YueinvSlipConverter
from .jiinv_converter import JiInvSlipConverter
from .mudpy_converter import MudpySlipConverter
from .usgs_converter import USGSGeoJSONConverter
from .geometry_utils import FaultGeometry, ReferencePoint, CoordinateConverter

__version__ = '1.0.0'
__author__ = 'Kefeng He'

__all__ = [
    'BaseSlipConverter',
    'JinInvSlipConverter', 
    'SDMSlipConverter',
    'TGASlipConverter',
    'YueinvSlipConverter',
    'MudpySlipConverter',
    'JiInvSlipConverter',
    'USGSGeoJSONConverter',
    'FaultGeometry',
    'ReferencePoint',
    'CoordinateConverter'
]