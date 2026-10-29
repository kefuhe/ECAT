# Observation support

This package owns the numerical observation contract used by several inversion
routes. CSI retains data objects, coordinates, source response and assembled GF.
Import the named module; the package entrance loads no numerical or solver library.

| File | Responsibility |
| --- | --- |
| `data_vector_layout.py` | Observation/synthetic row order and validated synthetic field assignment. GPS uses E(all), N(all), optionally U(all). |
| `data_prediction.py` | Formal linear prediction blocks, source-aware diagnostic predictions and explicit synthetic publication. |
| `data_corrections.py` | Nonlinear correction specifications, H matrices and coefficient slices. |
| `data_correction_parameters.py` | Raw correction coefficient interpretation, normalization and physical-unit conversion. Also consumed before solving constraints. |
| `covariance_utils.py` | Exact covariance metrics, whitening and Gaussian data likelihood helpers. No covariance estimation or approximation. |

Read row layout first, then prediction or correction composition, then covariance
weighting. Readers and deterministic preprocessing stay in their existing owners;
estimating a nuisance parameter is a different operation from altering observations.

Scientific contracts and import migration:
[observation and constraint support](../../../docs/developer/observation_constraint_support.md),
[row layout](../../../docs/concepts/observation_matrix_layout.md),
[corrections](../../../docs/reference/data_corrections.md).
