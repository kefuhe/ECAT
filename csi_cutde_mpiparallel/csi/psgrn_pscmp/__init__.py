"""
PSGRN/PSCMP Interface Module

This module provides tools to generate configuration files for PSGRN/PSCMP
calculations and interface with the external PSGRN/PSCMP programs for 
computing layered Green's functions and forward modeling deformation.

Classes:
    PSGRNConfig: Configuration generator for PSGRN input files
    PSCMPConfig: Configuration generator for PSCMP input files  
    LayeredGreenFunction: Main interface for PSGRN/PSCMP calculations

Functions:
    run_psgrn: Execute PSGRN calculation
    run_pscmp: Execute PSCMP calculation
"""

from .psgrn_config import PSGRNConfig, LayerModel, PSGRNParameters
from .pscmp_config import PSCMPConfig, FaultSource, PSCMPParameters
# from .green_function import LayeredGreenFunction
# from .runner import run_psgrn, run_pscmp

__all__ = [
    'PSGRNConfig', 'LayerModel', 'PSGRNParameters',
    'PSCMPConfig', 'FaultSource', 'PSCMPParameters', 
    # 'LayeredGreenFunction',
    # 'run_psgrn', 'run_pscmp'
]