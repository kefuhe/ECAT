# Constraint support

This package owns declarations, resolved bounds/rows and compilation into the
active parameter space. Solvers consume that state; this package does not own
observations, physical loading definitions, GF generation or inversion results.
The package entrance is light; import the named implementation module.

| File | Responsibility |
| --- | --- |
| `constraint_manager_base.py` | Shared declarations, registry, transaction/revision state and validation. |
| `constraint_manager_blse.py` | Compile constraints in the full BLSE/VCE linear vector. |
| `constraint_manager_smc.py` | Compile Bayesian bounds and supported constraints; distinguish sampled space and the SMC_FJ linear suffix. |
| `data_correction_constraints.py` | Resolve named nuisance parameters to columns/scales; compile ties and explicit bounds. |
| `euler_inequality_constraints.py` | Euler cap rows, hard-state overlap policy and cap preflight checks. |
| `block_euler_constraints.py` | Dataset/block Euler equality rows. Read physical vectors and column resolution from the loading model. |

Start with the relevant inversion object's public method, then its manager, then
shared storage. Rake, patch/source rules and interseismic declarations retain their
distinct meaning. FULLSMC does not acquire linear matrix constraints merely because
its manager shares this directory. Physical Euler projection stays in
`interseismic_parameter_model.py` and never imports constraint compilation.

Scientific contracts and import migration:
[observation and constraint support](../../../docs/developer/observation_constraint_support.md),
[constraint API](../../../docs/reference/constraint_manager.md).
