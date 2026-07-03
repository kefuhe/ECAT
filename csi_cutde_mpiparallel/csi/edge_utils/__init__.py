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
from .topology_boundary import extract_four_edges_topology, extract_topology_boundary_loop

__all__ = [
    'RectangularPatchAnalyzer',
    'find_adjacent_triangles', 
    'find_four_boundary_triangles',
    'CurveReconstructor',
    'extract_four_edges_topology',
    'extract_topology_boundary_loop',
]

__version__ = '1.0.0'
