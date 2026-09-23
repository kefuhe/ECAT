# Joint Bayesian template configurations

These files pair with the editable scripts in the parent `scripts/` directory.
Each pair keeps the geometry reference, sampled-vector slice,
perturbation method, mesh policy, and bounds in one documented scenario.

| Scenario | Script | Main config | Bounds |
| --- | --- | --- | --- |
| One bottom-edge offset | `test_joint_bayesian_bottom_offset.py` | `bottom_offset.yml` | `bottom_offset_bounds.yml` |
| Multiple dip controls (the template uses three) | `test_joint_bayesian_three_dip_controls.py` | `three_dip_controls.yml` | `three_dip_controls_bounds.yml` |
| Composite perturbation (the current method uses four parameters) | `test_joint_bayesian_custom_perturbation.py` | `custom_perturbation.yml` | `custom_perturbation_bounds.yml` |

Copy one complete set into a case directory and rename the YAML files to the
script defaults, or pass their paths with `--config` and `--bounds`. Paths in
the Python templates are resolved with `pathlib` and do not embed Windows or
POSIX absolute paths. See the installation guide for currently supported
platforms and environment requirements.

For a new configuration based on the installed ECAT version, generate the
full files first and use these scenario files as an editing reference:

```text
ecat-generate-config -o default_config.yml --gf-method cutde
ecat-generate-boundary -o bounds_config.yml -f MainFault
```

The main configs keep the standard SMC schedule in one block:

```yaml
smc_tempering:
  target_cov: 1.0
  max_delta_beta: 0.5
```

Keep these defaults for a baseline run. `max_delta_beta: 0.75` is an advanced
schedule test that may remove a short final beta increment when the COV target
also permits the larger step; it is not an unconditional stage merge. See
[SMC tempering](../../../docs/reference/smc_tempering.md) for formulas and
checkpoint rules.

Per-parameter geometry bounds use explicit lower and upper arrays:

```yaml
geometry:
  MainFault:
    lb: [-10.0, -15.0, -5.0, -5.0]
    ub: [10.0, 15.0, 5.0, 5.0]
```

An `N x 2` list containing one `[lower, upper]` row per parameter is not a
supported ECAT bounds representation.

The three-control template keeps independent increments by default. For an
optional shared west/middle increment, set
`dip_perturbation_groups = ["wm", "wm", "east"]` in the Python script, use
`sample_positions: [0, 2]`, and provide two geometry bounds in `[wm, east]`
order. Group labels live only in `set_dip_profile()`; the YAML slice remains a
global half-open parameter interval.

All three scripts use the same post-processing contract. The representative
model is activated first; `output/` then receives geometry/KDE/statistics and
fault products, while `Modeling/` receives InSAR data, synthetic, residual,
and fit figures. `--no-plot` disables figures only, so it does not skip model
activation or the text/GMT exports.
