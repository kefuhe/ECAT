"""Compatibility alias; CPT implementation and resources belong to ecat_viz."""
import sys
from ecat_viz import cpt as _implementation
sys.modules[__name__] = _implementation
