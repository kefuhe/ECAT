# Statistical and post-analysis utilities

Import concrete modules when following a workflow. See the public manual's
[seismic projection reference](https://github.com/kefuhe/ECAT/blob/main/docs/reference/seismic_projection.md).

| Module | Responsibility |
| --- | --- |
| `earthquakesProj2Fault.py` | Event projection, input-face lineage and counts; consumes CSI mesh and boundary methods |
| `contour3D_extraction.py` | Scalar slip contours and its own source/receiver interpolation contract |
| `statisticsInFaultEdge.py` | Fault-edge statistical profiles and geometry diagnostics |
| `stat_utils.py` | Depth-based statistics |
| `profile_analyzer.py`, `multi_profile_plotter.py` | Profile preparation and presentation |
| `regional_ramp_removal.py` | Regional data correction, rather than inversion-result analysis |
| `fault_contours.py` | Established compatibility entry for isodepth operations in `geometry.fault_contours` |

CSI owns mesh connectivity, subdivision and boundary caches. Analysis owns event
assignment and aggregation, not another mesh implementation. Counts are arrays,
not physical slip. Names such as "total" or "contour" do not imply equal formulas.
Generic figure style belongs to ecat_viz; domain rendering belongs to viztools.

New analysis helpers stay near their scientific consumer. Reusable mesh kernels
belong in CSI. Do not import inversion sample machinery solely to analyze a fault.
Directory-wide relocation is a separate API migration, not part of this fix.
