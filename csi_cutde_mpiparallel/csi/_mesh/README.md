# Internal mesh kernels

`subdivision.py` owns shared-edge triangular subdivision and input-face lineage.
It operates only on arrays; it has no fault, event, inversion or plotting state.

Public callers use `TriangularPatches.subdivided_copy()`. That method constructs
an independent CSI fault and publishes its complete geometry through `set_mesh`.
New reusable mesh array operations belong here; object state publication stays
on the fault class. Event assignment and statistical aggregation belong in
higher-level analysis code, not in these kernels.
