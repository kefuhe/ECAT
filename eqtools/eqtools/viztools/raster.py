"""Compatibility alias for ecat_viz.raster."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("ecat_viz.raster")
