# Installation

ECAT supports CPython 3.10, 3.11, and 3.12 on 64-bit Windows and Linux.
CPython 3.10 is the recommended and most extensively validated version;
3.11 and 3.12 remain supported installation targets. Use the ECAT repository
to create a complete environment. Use the individual package directories only
when updating an existing installation or developing eqtools/CSI.

## 1. Clone ECAT and create the environment

```bash
git clone https://github.com/kefuhe/ECAT.git
cd ECAT

conda create -n ecat -c conda-forge python=3.10 --file requirements/ecat-requirements.txt
conda activate ecat
```

This is the normal user command; Conda selects platform-appropriate BLAS and
MPI implementations. If you already want MKL, Intel MPI, or Windows MS-MPI,
select it while creating the environment rather than replacing binary
runtimes after ECAT has been installed. The default command does not promise
Intel MPI or require any `I_MPI_*` variables; verify the selected
implementation after installation with `MPI.get_vendor()`.

### 1.1 Choose the numerical and MPI runtime before installation (optional)

Choose exactly one command below instead of the default `conda create`
command. All later steps still use `conda activate ecat`, install the matching
`okada4py` wheel, and run the ECAT installation script.

Linux/WSL with MKL and Open MPI:

```bash
conda create -n ecat --override-channels -c conda-forge --strict-channel-priority python=3.10 "libblas=*=*_mkl" openmpi --file requirements/ecat-requirements.txt
```

Windows with MKL and MS-MPI:

```powershell
conda create -n ecat --override-channels -c conda-forge --strict-channel-priority python=3.10 "libblas=*=*_mkl" msmpi --file requirements/ecat-requirements.txt
```

Linux/WSL or Windows with MKL and Intel MPI:

```bash
conda create -n ecat --override-channels -c conda-forge --strict-channel-priority python=3.10 "libblas=*=*_mkl" impi_rt --file requirements/ecat-requirements.txt
```

These profiles let Conda solve NumPy/SciPy, BLAS, mpi4py, and the MPI runtime
together. `impi_rt` supplies the Intel MPI runtime without requiring the full
oneAPI toolkit. Install full oneAPI only when Intel compilers, profiling tools,
or a source build against an external Intel MPI are required. MKL or Intel MPI
is not universally faster; benchmark the same ECAT case after installation.
See the [compute runtime stack](docs/concepts/compute_runtime_stack.md) for the
component hierarchy and trade-offs.

The explicit-provider commands use command-scoped `--override-channels` and
`--strict-channel-priority` so compiled BLAS/MPI packages cannot be split by a
pre-existing channel configuration. These flags do not modify `.condarc`.

### 1.2 Direct alternatives for two common create failures

If a VPN/proxy has been turned off but the Linux/WSL shell still contains stale
proxy variables, run the default create command without those variables:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  conda create -n ecat -c conda-forge python=3.10 \
  --file requirements/ecat-requirements.txt
```

Windows PowerShell, after confirming that the proxy is no longer needed:

```powershell
Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY, Env:ALL_PROXY -ErrorAction SilentlyContinue
conda create -n ecat -c conda-forge python=3.10 --file requirements/ecat-requirements.txt
```

If metadata download completes but channel mixing or the solver prevents a
solution, use the command-scoped isolated form:

```bash
conda create -n ecat --override-channels -c conda-forge \
  --strict-channel-priority --solver libmamba python=3.10 \
  --file requirements/ecat-requirements.txt
```

These alternatives do not persistently change `.condarc`. See
[installation troubleshooting](docs/getting_started/troubleshooting.md#2-conda-创建环境失败或很慢)
for certificates, existing pins, Windows proxy restoration, and combinations
with an explicit BLAS/MPI profile.

`requirements/ecat-requirements.txt` is the only supported user environment
file. It contains the direct runtime dependencies of CSI and eqtools with
compatibility ranges.

The stable numerical compatibility window remains:

```text
numpy>=1.23,<2
scipy>=1.10,<1.12
numba>=0.58,<0.60
```

The generated file has three ownership sections: dependencies imported by both
CSI and eqtools, CSI-only dependencies, and eqtools-only dependencies. Shared
packages remain declared in both package `setup.py` files so either standalone
checkout installs correctly; the ECAT environment file deduplicates them only
after ownership has been verified.

The file permits Python 3.10--3.12, while the main command selects the
recommended 3.10 version explicitly. To test a newer interpreter, use:

```bash
conda create -n ecat -c conda-forge python=3.11 --file requirements/ecat-requirements.txt
# or
conda create -n ecat -c conda-forge python=3.12 --file requirements/ecat-requirements.txt
```

All three versions require wheels matching the selected CPython ABI and
platform. If a required 3.11 or 3.12 wheel is unavailable, prefer the
recommended 3.10 environment instead of removing core dependencies.

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
environment. Pull the new ECAT checkout, enter the eqtools project, and run a
normal package installation:

```bash
# Run from the ECAT repository root
git pull
conda activate ecat
cd eqtools
python -m pip install .
```

Pip reconciles the direct requirements declared by `eqtools/setup.py`; it does
not require recreating the complete ECAT environment.

Only maintainers who need source edits to take effect without reinstalling
should use the editable form:

```bash
python -m pip install -e .
```

Update CSI separately only when its source or dependency metadata changed:

```bash
# Run from the ECAT repository root
cd csi_cutde_mpiparallel
python -m pip install .
```

## 5. Install optional eqtools features

Mesh generation and commonly used SAR/InSAR and GeoTIFF readers are base ECAT
features; no mesh or SAR extra is needed. Optional user-interface/export
features are installed from the eqtools project directory:

```bash
# Run this line first when starting from the ECAT repository root
cd eqtools

python -m pip install ".[geoexport]"   # Google Earth/KMZ export
python -m pip install ".[viewer]"      # Dash/Plotly map viewer
python -m pip install ".[interaction]" # Bokeh trace editor
```

The base environment includes the supported Bayesian SMC geometry inversion,
BLSE/VCE linear slip inversion, mesh, and SAR dependencies.

## 6. MPI, mpi4py, and oneAPI

MPI is a standard. Open MPI, MPICH, Intel MPI, and MS-MPI are implementations
that provide a runtime library and `mpiexec`; `mpi4py` is the Python binding.
The launcher and the library loaded by `mpi4py` must belong to the same MPI
implementation or a documented compatible ABI family.

oneAPI is a toolkit rather than a single acceleration switch. oneMKL supplies
BLAS/LAPACK routines, Intel MPI supplies process launch and communication, and
Intel compilers build native code. Installing oneAPI does not relink an
existing NumPy/SciPy installation, and an Intel MPI launcher cannot be mixed
with an Open MPI build of mpi4py. A local workstation does not require oneAPI
for multiprocessing: all supported MPI implementations can launch local
processes, while MKL/OpenBLAS can use threads inside each process.

Conda normally resolves a platform-appropriate pair. Users who want to compare
oneMKL or Intel MPI should use a separate test environment and benchmark the
same ECAT case rather than modify a working installation. See the
[compute runtime stack](docs/concepts/compute_runtime_stack.md) for the
component hierarchy, platform table, common package pairs, and performance
guidance.

Before a large MPI SMC run, check the launcher/runtime pair:

```bash
python -c "from mpi4py import MPI; print(MPI.get_vendor()); print(MPI.Get_library_version())"
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())"
```

The two-process result must contain `0 2` and `1 2`. Two rank-zero/singleton
results indicate a launcher/runtime mismatch; follow the MPI path checks in
the troubleshooting guide. For a vendor-neutral first-run command and a
controlled comparison of rank and thread settings, see
[parallel execution basics](docs/concepts/parallel_process_rank_thread.md#1-首次运行的通用复制模板).

An MPI failure does not prevent non-MPI data preparation, mesh, CVXOPT/Clarabel,
or BLSE/VCE workflows from running.

## 7. Verify the installation

```bash
python -c "import csi, eqtools, okada4py; print('ECAT imports succeeded')"
ecat-generate-downsample --help
ecat-generate-nonlinear --help
ecat-downsample --help
```

## Installation and runtime problems

For VPN or stale-proxy failures, Conda solver/channel problems, pip dependency
adjustments, missing `okada4py`, editor import warnings, MPI runtime errors,
GDAL/PROJ ABI errors, or BLSE runs that remain at `Initializing solver object`, use
[Installation and runtime troubleshooting](docs/getting_started/troubleshooting.md).
It keeps the default command short and supplies symptom-specific fallback
commands, path checks, and temporary MKL/OpenBLAS tests without silently
changing persistent settings.
