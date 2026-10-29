"""Fault-geometry support with explicit per-module scientific contracts.

Trace coordinates, dip profiles, boundary operations, patch queries and mesh
backends live here. Stateful fault objects and the perturbation pipeline own
reference freezing, candidate materialization and CSI mesh publication.

Import the specific implementation module. This entrance deliberately does
not initialize CSI, plotting, meshing or inversion libraries. Planar trace
length and three-dimensional parametric-mesh length remain distinct contracts.
"""
