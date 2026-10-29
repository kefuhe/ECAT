# Result reporting

Reporting consumes prepared model, observation and constraint information. User
methods remain on inversion objects; numerical/state preparation stays with its
existing owner, including explicit diagnostic-prediction delegation.

| Files | Responsibility |
| --- | --- |
| `figure_products.py`, `plot_product_mixin.py`, `data_plot_utils.py` | Standard figure products and data-specific presentation. |
| `fit_statistics.py`, `hyperparameter_reporting.py` | Fit metrics, sigma/alpha interpretation and result tables. |
| `parameter_layout_reporting.py` | Read-only sampled/linear parameter layout descriptions. |
| `nonlinear_geometry_plots.py`, `nonlinear_fit_statistics_mixin.py` | Nonlinear posterior plots and fit reporting. |
| `data_correction_report_mixin.py` | Collect and expose estimated nuisance-parameter results and diagnostics. |
| `data_correction_results.py` | Correction result DataFrames, text tables and file formatting. |

Start at the inversion method, then the product/collection module, then its
formatter. Coefficient normalization/physical interpretation belongs to
`observations/data_correction_parameters.py`; tying those parameters belongs to
`constraints/data_correction_constraints.py`.

All reporting implementations use their current `reporting` paths. The eight
former root-level forwarding modules have been removed; stable inversion methods
and existing package-level public exports remain. Old script imports and pickled
class paths require the documented migration.
See [reporting import migration](../../../docs/developer/architecture.md#reporting-import-migration) and
[observation/constraint support](../../../docs/developer/observation_constraint_support.md).
