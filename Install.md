# Installation

ECAT supports CPython 3.10, 3.11, and 3.12 on 64-bit Windows and Linux. Use
the ECAT repository to create a complete environment. Use the individual
package directories only when updating an existing installation or developing
eqtools/CSI.

## 1. Clone ECAT and create the environment

```bash
git clone https://github.com/kefuhe/ECAT.git
cd ECAT

conda create -n ecat -c conda-forge --file requirements/ecat-requirements.txt
conda activate ecat
```

`requirements/ecat-requirements.txt` is the only supported user environment
file. It contains the direct runtime dependencies of CSI and eqtools with
compatibility ranges. It is not a full export of a maintainer's environment
and does not contain operating-system build strings, PyMC, PyTensor, or Theano.

The file permits Python 3.10--3.12. To choose a version explicitly, use for
example:

```bash
conda create -n ecat -c conda-forge python=3.11 --file requirements/ecat-requirements.txt
```

## 2. Install the required okada4py wheel

CSI imports `okada4py` during package import, so it is required. Download a
wheel matching the active CPython version, ABI, operating system, and CPU
architecture from [okada4py Releases](https://github.com/kefuhe/okada4py/releases),
then install and check it:

```bash
python -m pip install path/to/okada4py-<version>-cp<major><minor>-cp<major><minor>-<platform>.whl
python -c "import okada4py; print(okada4py.__file__)"
```

For example, CPython 3.12 requires a `cp312-cp312` wheel. `win_amd64` denotes
64-bit Windows and `linux_x86_64` denotes 64-bit Linux. If no matching release
wheel is available, build the pinned public source instead; this requires a C++
compiler:

```bash
python -m pip install "git+https://github.com/kefuhe/okada4py.git@v12.0.2"
python -c "import okada4py; print(okada4py.__file__)"
```

## 3. Install CSI and eqtools

Run the platform script from the ECAT repository root:

```bash
# Linux
chmod +x install.sh
./install.sh

# Windows
.\install.bat
```

The scripts use the active interpreter through `python -m pip`, install CSI
before eqtools, and verify that both packages import. They stop if the Python
version is outside 3.10--3.12 or if `okada4py` is missing.

## 4. Update an existing source installation

A normal eqtools source update does not require recreating the Conda
environment. Pull the new ECAT checkout, enter the eqtools project, and repeat
the editable install:

```bash
# Run from the ECAT repository root
git pull
conda activate ecat
cd eqtools
python -m pip install -e .
```

Pip reconciles the direct requirements declared by `eqtools/setup.py`; it does
not reproduce every package installed on a maintainer's machine.

Update CSI separately only when its source or dependency metadata changed:

```bash
# Run from the ECAT repository root
cd csi_cutde_mpiparallel
python -m pip install -e .
```

In an independent eqtools or CSI development checkout, activate the existing
ECAT environment and run `python -m pip install -e .` directly from that
repository root. This is a package-level incremental install, not a bootstrap
procedure for an empty environment.

## 5. Install optional eqtools features

Mesh generation and commonly used SAR/InSAR and GeoTIFF readers are base ECAT
features; no mesh or SAR extra is needed. Optional user-interface/export
features are installed from the eqtools project directory:

```bash
# Run this line first when starting from the ECAT repository root
cd eqtools

python -m pip install -e ".[geoexport]"   # Google Earth/KMZ export
python -m pip install -e ".[viewer]"      # Dash/Plotly map viewer
python -m pip install -e ".[interaction]" # Bokeh trace editor
```

The base environment includes the supported Bayesian SMC geometry inversion,
BLSE/VCE linear slip inversion, mesh and SAR dependencies. PyMC, PyTensor, and
Theano are not installed or supported.

## 6. MPI, mpi4py, and oneAPI

`mpi4py` is a Python binding used by the current SMC implementation. `mpiexec`
is supplied by an MPI runtime and starts multiple Python processes.
Conda/conda-forge normally resolves a compatible runtime with `mpi4py`, so a
separate full oneAPI installation is not required for ordinary Windows or Linux
installations.

Use oneAPI/Intel MPI only when a cluster explicitly requires Intel MPI, and
ensure that `mpi4py` is built or installed against the same implementation. Do
not mix an unrelated system `mpiexec` with the active environment's `mpi4py`.

Before a large MPI SMC run, check the launcher/runtime pair:

```bash
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

An MPI failure does not prevent non-MPI data preparation, mesh, CVXOPT/Clarabel,
or BLSE/VCE workflows from running.

## 7. Verify the installation

```bash
python -c "import csi, eqtools, okada4py; print('ECAT imports succeeded')"
ecat-generate-downsample --help
ecat-generate-nonlinear --help
ecat-downsample --help
```

## Dependency maintenance

Package metadata is the source of truth for direct dependencies. The ECAT
integration repository aggregates the two package declarations into the one
user environment file:

```bash
python scripts/generate_requirements.py
python scripts/generate_requirements.py --check
```

Run these commands only from the ECAT repository root after changing package
imports or metadata. Do not generate the public environment file with
`conda list`, `pip freeze`, or a personal development environment. See the
[maintainer integration guide](docs/developer/release_sync.md) for the
standalone-to-ECAT synchronization boundary.

## Common installation problems

- `No module named 'okada4py'`: install a wheel matching the active CPython and
  platform before running the ECAT installation script.
- Conda cannot solve the environment: create a fresh Python 3.10, 3.11, or 3.12
  environment with conda-forge; do not delete arbitrary requirements or reuse
  old platform build strings.
- A source update is not visible: activate the intended environment and rerun
  `python -m pip install -e .` from the changed package directory.
- An extra installs the wrong project or fails: run `.[viewer]`,
  `.[interaction]`, or `.[geoexport]` from the directory containing the eqtools
  `setup.py`.
- `mpiexec` is unavailable: diagnose the MPI runtime and `mpi4py` together;
  installing full oneAPI is not a default remedy.
