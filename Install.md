# Installation

This page describes the recommended installation workflow for ECAT. ECAT is a research toolkit, so the exact dependency set can vary by operating system, Python version, and the Green's function backends used in a given case. The practical goal is to create a working environment first, then install any missing optional packages as needed.

## 1. Create a Python Environment

We recommend using Anaconda or Miniconda:

```bash
conda create -n myecat python=3.10
conda activate myecat
python -m pip install -U pip setuptools wheel build numpy
```

Python 3.10 is currently a conservative choice for the released examples and compiled dependencies. Python 3.11 or 3.12 may also work, but make sure compiled packages such as `okada4py` provide wheels for your Python version and platform, or that you have a working compiler toolchain.

## 2. Optional: Use the Provided Requirements Files

ECAT provides platform-specific requirements files:

```text
requirements/conda-requirements-win-64.txt
requirements/pip-requirements-win-64.txt
requirements/conda-requirements-linux-64.txt
requirements/pip-requirements-linux-64.txt
```

These files are snapshots of the maintainer-tested Windows/Linux environments. They are useful references, but they are not universal lock files that every platform must install exactly.

If you want to start from the tested snapshot, use the file matching your platform:

```bash
# Windows
conda create -n myecat --file requirements/conda-requirements-win-64.txt
conda activate myecat
python -m pip install -r requirements/pip-requirements-win-64.txt

# Linux
conda create -n myecat --file requirements/conda-requirements-linux-64.txt
conda activate myecat
python -m pip install -r requirements/pip-requirements-linux-64.txt
```

If conda cannot solve or download one of the pinned packages, copy the requirements file, remove the failing pinned line, and continue. On a different platform or Python build, it is normal for some exact build strings to be unavailable. After ECAT is installed, run the examples and install missing packages only when they are actually needed:

```bash
conda install -c conda-forge <package>
# or
python -m pip install <package>
```

The `*-full.txt` files are closer to complete environment exports. They are useful for debugging or reproducing the maintainer's machine, but they are not recommended as the first installation path for new users.

If you need `conda-forge`, add it explicitly:

```bash
conda config --add channels conda-forge
conda config --set channel_priority flexible
```

## 3. Install ECAT

Clone the repository:

```bash
git clone https://github.com/kefuhe/ECAT.git
cd ECAT
```

Install the ECAT Python packages from the repository root:

```bash
# Linux / macOS
chmod +x install.sh
./install.sh

# Windows
.\install.bat
```

The install scripts install the two main Python subpackages:

- `eqtools`
- `csi_cutde_mpiparallel`, which provides the `csi` package used by the examples

For development installs, you can also install the subpackages manually:

```bash
python -m pip install -e eqtools
python -m pip install -e csi_cutde_mpiparallel
```

## 4. Install okada4py

ECAT uses [okada4py](https://github.com/kefuhe/okada4py) for some Okada Green's function workflows. This package contains a compiled extension, so installation can fail if a compiler is missing. This is especially common on Windows.

The recommended path is to install a prebuilt wheel from GitHub Releases:

[https://github.com/kefuhe/okada4py/releases](https://github.com/kefuhe/okada4py/releases)

Download the wheel matching your Python version and platform, then install it in the activated `myecat` environment:

```bash
python -m pip install path/to/okada4py-<version>-<python-tag>-<abi-tag>-<platform-tag>.whl
```

Example wheel names:

```text
okada4py-12.0.2-cp310-cp310-win_amd64.whl
okada4py-12.0.2-cp310-cp310-linux_x86_64.whl
```

Here, `cp310` means CPython 3.10 and `win_amd64` means 64-bit Windows. A wheel built for one Python version or platform will not install on another. If pip reports `not a supported wheel on this platform`, download a matching wheel or install from source.

Only build from source when no matching wheel is available:

```bash
git clone https://github.com/kefuhe/okada4py.git
cd okada4py
python -m pip install -U pip setuptools wheel build numpy
python -m pip install .
```

Source builds require a compiler:

- Windows: Microsoft C++ Build Tools with MSVC and a Windows SDK.
- Linux: `build-essential` and Python development headers, or equivalent packages for your distribution.
- macOS: Xcode Command Line Tools.

For most Windows users, the release wheel is the best option.

Verify the installation:

```bash
python -c "import okada4py; print(okada4py.__file__)"
```

## 5. Optional Parallel and Performance Dependencies

Small examples can be run without configuring a full MPI or oneAPI environment. Configure these only when you need large Bayesian sampling jobs or production-scale parallel runs.

Optional packages:

```bash
conda install -c conda-forge mpi4py
conda install scikit-learn-intelex
```

On Linux, if you use Intel MPI / oneAPI, install oneAPI following Intel's official instructions and load the environment before running MPI jobs:

```bash
source ~/intel/oneapi/setvars.sh intel64
```

If MPI is not working yet, first verify ECAT with small non-MPI examples, then debug `mpi4py`, the MPI runtime, or oneAPI separately.

## 6. Case Library

The case library is hosted separately:

[https://github.com/kefuhe/ECAT-Cases](https://github.com/kefuhe/ECAT-Cases)

Clone it when you need to run the tutorial cases:

```bash
git clone https://github.com/kefuhe/ECAT-Cases.git
```

The ECAT repository contains the code, templates, and method documentation. `ECAT-Cases` contains data, runnable scripts, reference outputs, and figures.

## 7. Verify the Installation

Check the Python packages:

```bash
python -c "import eqtools; print('eqtools import ok')"
python -c "import csi; print('csi import ok')"
python -c "import okada4py; print('okada4py import ok')"
```

Check the ECAT command line tools:

```bash
ecat-generate-downsample --help
ecat-downsample --help
ecat-generate-nonlinear --help
ecat-generate-config --help
ecat-generate-boundary --help
```

If the command line entry points are not available, try the module form:

```bash
python -m eqtools.cli_tools.generate_downsample_config --help
python -m eqtools.cli_tools.process_data_downsampling --help
python -m eqtools.cli_tools.generate_nonlinear_config --help
```

If you plan to use MPI:

```bash
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

## 8. Common Installation Problems

### `ModuleNotFoundError`

Install the missing package in the active environment:

```bash
conda install -c conda-forge <package>
# or
python -m pip install <package>
```

### Conda cannot solve the requirements file

The requirements files are tested environment snapshots. Remove the failing pinned package line from a copy of the file and continue. Different platforms do not always provide the same build strings.

### okada4py fails to build on Windows

Use a matching release wheel from [okada4py Releases](https://github.com/kefuhe/okada4py/releases). Building from source on Windows requires Microsoft C++ Build Tools and is not the recommended first attempt for most users.

### `No module named 'okada4py._okada92'`

The compiled extension was not installed correctly. Reinstall a matching wheel, or rebuild from source in an environment with a working compiler and NumPy installed.

### MPI fails but normal Python imports work

This is usually an MPI runtime or `mpi4py` configuration issue. You can still run non-MPI examples. Fix MPI separately before launching large Bayesian sampling jobs.

## Tested Platforms

The installation has been tested on Windows 11 and Ubuntu 20.04.6. Other platforms may work, but may require small dependency adjustments.
