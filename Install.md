# Installation

ECAT supports CPython 3.10, 3.11, and 3.12 on 64-bit Windows and Linux. The
supported installation is a direct-dependency environment, not a copy of a
maintainer's entire Python installation.

## 1. Create the ECAT environment

Install the one supported dependency list with conda-forge:

```bash
conda create -n ecat -c conda-forge --file requirements/ecat-requirements.txt
conda activate ecat
```

The file permits Python 3.10--3.12. To choose a particular supported version,
add it explicitly; for example, use `python=3.11` in the create command.

`requirements/ecat-requirements.txt` contains only direct ECAT runtime
dependencies with tested compatibility ranges. It intentionally does not pin
operating-system build strings or list transitive packages such as Flask,
Werkzeug, or build tools.

## 2. Install the required okada4py wheel

CSI imports `okada4py` at package import time, so it is required for the
supported ECAT installation. Obtain a wheel matching the active CPython version
and platform from [okada4py Releases](https://github.com/kefuhe/okada4py/releases), then install it in the active environment:

```bash
python -m pip install path/to/okada4py-<version>-cp<major><minor>-cp<major><minor>-<platform>.whl
python -c "import okada4py; print(okada4py.__file__)"
```

For example, `win_amd64` is 64-bit Windows and `linux_x86_64` is 64-bit Linux.
If pip reports that the wheel is unsupported, choose the wheel matching the
current Python version, ABI, operating system, and CPU architecture. For
example, CPython 3.12 requires a `cp312-cp312` wheel. If no matching release
wheel is available, build the public source at the pinned tag instead (a C++
compiler is required):

```bash
python -m pip install "git+https://github.com/kefuhe/okada4py.git@v12.0.2"
python -c "import okada4py; print(okada4py.__file__)"
```

## 3. Install ECAT

Clone the repository and run the platform script:

```bash
git clone https://github.com/kefuhe/ECAT.git
cd ECAT

# Linux
chmod +x install.sh
./install.sh

# Windows
.\install.bat
```

The scripts use `python -m pip`, install `csi` before `eqtools`, and verify that
both packages import. They stop early if the interpreter is outside Python
3.10--3.12 or `okada4py` is missing.

For an editable installation:

```bash
python -m pip install -e csi_cutde_mpiparallel
python -m pip install -e eqtools
```

## 4. Standard mesh and SAR support

Mesh generation and commonly used SAR/InSAR and GeoTIFF readers are part of
the base ECAT environment. The required Gmsh, meshio, GeoPandas, GDAL,
Rasterio, and Xarray dependencies are installed from
`requirements/ecat-requirements.txt`; no mesh or SAR extra is needed.

## 5. Optional feature groups

Install optional dependencies only for the workflows you use:

```bash
python -m pip install -e "eqtools[geoexport]"   # Google Earth export
python -m pip install -e "eqtools[viewer]"      # Dash/Plotly map viewer
python -m pip install -e "eqtools[interaction]" # Bokeh trace editor
```

The standard SMC geometry inversion and BLSE/VCE linear slip inversion are in
the base environment. `pymc`, `pytensor`, and `theano` are deliberately not
installed or supported. The current SMC implementation uses ECAT's MPI SMC
sampler with `mpi4py` and `numba` instead.

## 6. Verify the installation

```bash
python -c "import csi, eqtools, okada4py; print('ECAT imports succeeded')"
ecat-generate-downsample --help
ecat-generate-nonlinear --help
ecat-downsample --help
```

For MPI runs, also verify the runtime before launching a large SMC job:

```bash
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

## Dependency maintenance

There is one dependency directory, `requirements/`, and one user-facing
environment file, `requirements/ecat-requirements.txt`. Do not recreate it
with `conda list`, `pip freeze`, or a personal `cutde` environment: those
commands mix unrelated applications and transitive dependencies into ECAT.

Use the repository dependency tool after changing imports or package metadata:

```bash
python scripts/generate_requirements.py --check
```

It derives the conda file from the two package `install_requires` lists and
reports source imports not covered by package metadata. Exact, platform-specific
reproducibility locks should be generated separately from a clean test
environment; they are not a normal installation instruction.

## Common installation problems

- `No module named 'okada4py'`: install a matching release wheel before running
  `install.sh` or `install.bat`.
- Conda cannot solve the environment: create a fresh CPython 3.10, 3.11, or
  3.12 environment using conda-forge. Do not remove arbitrary lines from the
  supplied file.
- MPI fails while imports succeed: diagnose the system MPI runtime and
  `mpi4py` separately; non-MPI workflows remain available.
