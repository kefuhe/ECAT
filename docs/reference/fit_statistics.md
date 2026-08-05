# Fit Statistics

This page defines ECAT RMS/VR diagnostics used after BLSE/VCE and Bayesian
linear slip inversion.  The goal is to keep screen output, loop tables and
report files numerically consistent.

## Core Formula

For one vectorized dataset, ECAT uses the existing CSI-style residual

```python
r = synthetic - observed
rms = np.sqrt(np.mean(r**2))
vr = (1 - np.sum(r**2) / np.sum(observed**2)) * 100
```

For an assembled solver vector, the same definition is applied to the linear
system:

```python
r = np.dot(G, mpost) - d
rms = np.sqrt(np.mean(r**2))
vr = (1 - np.sum(r**2) / np.sum(d**2)) * 100
```

This is the same quantity printed by BLSE `returnModel()` when the assembled
matrix is available.  ECAT does not define the global RMS/VR as the arithmetic
mean of per-dataset RMS/VR values.

## Dataset Rows

Dataset-level statistics use the same vectorization as the legacy
`calculate_data_fit_metrics()` method:

| Data type | Observed vector | Synthetic vector |
| --- | --- | --- |
| GPS, `vertical=True` | `vel_enu.flatten()` | `synth.flatten()` |
| GPS, `vertical=False` | horizontal EN columns from `vel_enu` | horizontal EN columns from `synth` |
| InSAR/SAR | `vel` | `synth` |
| optical offset | `east` + `north` | `east_synth` + `north_synth` |
| leveling | `vel` | `synth` |
| cross-fault offset | `data_vector` | `synth_vector` |

When `data_poly="config"`, plotting and statistics use the same parsed,
per-dataset prediction policy:

| Configured poly | Synthetic used for statistics |
| --- | --- |
| `None` | slip-only prediction |
| any configured correction | `poly="include"` |

Use `data_poly=None` only when intentionally checking slip-only fits, and use
`data_poly="include"` when all configured data-correction terms should be
included regardless of the original config entry.

## Global Row

`collect_fit_statistics(include_global=True)` attempts to add a
`global_solver_vector` row from the assembled linear system:

```python
# BLSE
np.dot(self.G, self.mpost) - self.d

# Bayesian linear vector, when dimensions match
np.dot(self.G_combined, self.mpost) - self.observations
```

If the assembled matrix or vector is not available, or if dimensions do not
match the current model object, ECAT omits the global row instead of computing a
different surrogate statistic.  This avoids accidentally mixing a dataset-level
synthetic convention with the solver-vector convention.

## Current-model contract

Fit statistics describe the model that is currently distributed to the fault
and data objects.  The `model=` argument records a label in the returned rows;
it does not select a Bayesian posterior summary by itself.

The required order therefore depends on the inversion route:

| Route | Activate the model first | Then collect statistics |
| --- | --- | --- |
| BLSE | `inv.run(...)` already solves and distributes the latest `mpost` | call `collect_fit_statistics()` directly |
| VCE | `inv.run_simple_vce(...)` already distributes the final VCE solution | call `collect_fit_statistics()` directly |
| Bayesian linear/joint inversion | call `inv.returnModel(model="mean" | "median" | "MAP", print_stat=False)` | use the same name only as the report label |

For example, Bayesian MAP and median fits must be activated and collected
separately:

```python
inv.returnModel(model="MAP", print_stat=False)
map_rows = inv.collect_fit_statistics(model="MAP")

inv.returnModel(model="median", print_stat=False)
median_rows = inv.collect_fit_statistics(model="median")
```

Calling only `collect_fit_statistics(model="MAP")` after another model has
been activated does not switch the fault slip, polynomial solution, or
`mpost` to MAP.

For BLSE, use the solution produced by `run()` as the synchronized statistics
state.  The optional temporary-vector form `returnModel(mpost=...)` restores
the previous solver vector after distributing the temporary values and should
not be used to compare dataset and global rows for an arbitrary external
vector.

## Public API

The two main structured interfaces have separate responsibilities:

| Method | Responsibility |
| --- | --- |
| `collect_fit_statistics(...)` | rebuild configured synthetics when requested and calculate dataset/global rows |
| `fit_statistics_to_dataframe(rows)` | convert existing rows to a DataFrame; it does not recalculate a model or fit |
| `write_fit_statistics_report(...)` | write existing or newly collected rows as text/table files |

After a BLSE solve, collect and write reports as follows:

```python
inv.run(alpha=[-2.0])

rows = inv.collect_fit_statistics(
    model="BLSE",
    data_poly="config",
    include_dataset=True,
    include_global=True,
)

df = inv.fit_statistics_to_dataframe(rows)
inv.write_fit_statistics_report("output", rows=rows)
```

`collect_fit_statistics()` rebuilds dataset synthetics by default.  With
`data_poly="config"`, each dataset follows its configured correction state.
The global row is calculated independently from the current assembled
solver vector.

The default screen output from `calculate_and_print_fit_statistics()` remains
dataset-oriented for compatibility with existing scripts.  Use
`collect_fit_statistics()` when tuning geometry, dip angle, smoothing weight or
data-correction choices and you need a structured table.

Structured rows preserve the numerical unit of the observation vector; unit
labels do not rescale RMS automatically.  They also preserve the mathematical
VR value when it is negative.  Some legacy Bayesian screen summaries floor a
negative global VR at zero, so use the structured row when comparing runs.

## BLSE loop pattern

For a general geometry, smoothing, or constraint experiment, keep the loop in
the scientific script and append both dataset and global rows after every
successful solve:

```python
all_rows = []

for value in test_values:
    # Update the geometry/configuration/constraints for this case.
    inv.run(...)

    rows = inv.collect_fit_statistics(
        model=f"case_{value}",
        data_poly="config",
        include_dataset=True,
        include_global=True,
    )
    all_rows.extend({"test_value": value, **row} for row in rows)

df = inv.fit_statistics_to_dataframe(all_rows)
```

The call to `run()` must complete before collection.  It updates `mpost` and
distributes the same solution to fault slip and polynomial fields, so the
dataset synthetics and global solver row refer to the same loop iteration.

## Interpretation

- Per-dataset RMS/VR answers how well a specific dataset is matched under its
  vectorization and vertical/poly settings.
- The global solver-vector row answers how well the assembled inversion system
  is matched as one linear vector.
- A dataset average can be useful as a quick diagnostic, but it is not the same
  as the global solver-vector RMS/VR and should not be reported as the total
  fit.
