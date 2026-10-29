"""Private rendering implementations for CSI data objects.

User scripts call data-object plot methods. Renderers consume prepared fields
and display options; they do not activate inversion models or own observation
geometry, prediction composition, Green functions or solver state.
"""
