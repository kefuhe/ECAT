<p align="center">
  <img src="image/logo.png" alt="ECAT logo">
</p>

# ECAT / eqtools

ECAT is a Python toolkit for earthquake-cycle geodetic modeling and inversion.
The public package centers on `eqtools` and `csi` workflows for InSAR/GPS data
preparation, nonlinear fault-geometry search, and distributed slip inversion.

## Documentation

Start from the public user manual:

- [ECAT User Manual](docs/index.md)
- [Installation and environment checks](docs/getting_started/installation.md)
- [Standard two-step quickstart](docs/getting_started/quickstart_two_step.md)
- [Task-oriented examples](docs/examples/index.md)
- [Reference map](docs/reference/index.md)

The manual is organized by scientific workflow rather than source-code layout:

1. Read and prepare InSAR/GPS data.
2. Estimate compact fault geometry with Bayesian nonlinear inversion.
3. Build a fixed fault mesh and solve distributed slip with BLSE/VCE.
4. Use advanced joint Bayesian geometry-slip inversion only after the standard
   two-step workflow is understood and checked.

## Installation

From a local checkout:

```bash
python -m pip install .
```

For editable development installs:

```bash
python -m pip install -e .
```

If pip cannot download isolated build dependencies because of network or SSL
issues, use the Python environment that already contains `setuptools` and
`wheel`:

```bash
python -m pip install -e . --no-build-isolation
```

Full environment notes are in
[Installation and environment checks](docs/getting_started/installation.md).

## Cases

Runnable case scripts, input data, and reference outputs are maintained in
[ECAT-Cases](https://github.com/kefuhe/ECAT-Cases).  Use the
[casebook](docs/casebook/index.md) to choose a case that matches the workflow
you want to learn.

## Main Entry Points

- `ecat-generate-downsample` and `ecat-downsample` for SAR/offset downsampling.
- `ecat-generate-nonlinear-geometry` for the newer nonlinear geometry template.
- `ecat-generate-config` and `ecat-generate-boundary` for BLSE/VCE linear slip.
- `ecat-fault-trace-tool` for trace simplification and smoothing.

See the [CLI reference](docs/reference/cli.md) for command details.

## Related Projects

- [CSI](https://github.com/jolivetr/csi), the classic slip inversion framework
  that ECAT extends.
- [Gmsh](https://gmsh.info/), used by several mesh-generation workflows.

## Selected References

If ECAT or eqtools supports your research, cite the scientific studies and
software components used in your workflow.  Common background references include:

- He, K., Y. Wen, C. Xu, and Y. Zhao (2021), fault geometry and slip
  distribution of the 2021 Mw 7.4 Maduo, China, earthquake inferred from InSAR
  measurements and relocated aftershocks.
- Diao, F., R. Wang, Y. Wang, X. Xiong, and T. R. Walter (2018), fault behavior
  and lower-crust rheology inferred from postseismic GPS data after the 2008
  Wenchuan earthquake.
- Diao, F., R. Wang, X. Xiong, and C. Liu (2021), overlapped postseismic
  deformation caused by afterslip and viscoelastic relaxation following the 2015
  Gorkha earthquake.
- Wang, K., Y. Zhu, E. Nissen, and Z. K. Shen (2021), relevance of geodetic
  deformation rates to earthquake potential.

Case-specific references are listed in the documentation and case materials.
