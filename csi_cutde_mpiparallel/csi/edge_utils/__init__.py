"""
Edge utilities for geological fault processing.

This package provides functionality for:
- Fault edge detection and identification  
- Edge vertex ordering and reconstruction
- Curve reconstruction from unordered points
- Edge visualization and analysis
"""

from .patch_edge_finder import RectangularPatchAnalyzer
from .mesh_edge_finder import find_adjacent_triangles, find_four_boundary_triangles
from .fault_edge_reconstruction import CurveReconstructor

__all__ = [
    'RectangularPatchAnalyzer',
    'find_adjacent_triangles', 
    'find_four_boundary_triangles',
    'CurveReconstructor'
]

__version__ = '1.0.0'