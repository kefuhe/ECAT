# Sbarbot Fortran Library — Installation Guide

## Overview

The `sbarbot_src/` directory contains the essential Fortran source files for
computing surface displacement from vertical strain volumes
(Barbot et al., 2017, BSSA).

**Most users do NOT need to compile anything.**  CSI ships with:

1. **Pre-compiled binaries** in `csi/bin/windows/` (and `ubuntu20.04/` for Linux)
2. **Pure-NumPy fallback** (`sbarbot_python.py`) that works automatically when
   no compiled library is found

The Fortran library is only needed for **production-scale** computations
(~2× faster for large arrays).

---

## Dependencies (for compilation only)

| Dependency | Version | Purpose |
|------------|---------|---------|
| `gfortran` | ≥ 4.8 | GNU Fortran compiler (part of GCC) |
| `python`   | ≥ 3.6 | Build script driver |

### Installing gfortran

**Windows** (choose one):
```bash
# Option 1: MSYS2 (recommended)
pacman -S mingw-w64-x86_64-gcc-fortran

# Option 2: conda
conda install -c conda-forge gfortran
```

**Ubuntu / Debian:**
```bash
sudo apt-get install gfortran
```

**macOS:**
```bash
brew install gcc   # includes gfortran
```

---

## Compilation

```bash
cd csi/sbarbot_src
python build_sbarbot.py
```

This produces:
- **Windows:** `sbarbot.dll`
- **Linux:**   `libsbarbot.so`
- **macOS:**   `libsbarbot.dylib`

The library is created in the `sbarbot_src/` directory.

### Install to `bin/` (recommended)

```bash
python build_sbarbot.py --install
```

This additionally copies the library to `csi/bin/<platform>/`, making it
available system-wide for all scripts using CSI.

---

## Verification

After compilation, verify with:

```python
from csi import sbarbotfull
import numpy as np

u1, u2, u3 = sbarbotfull.displacement(
    np.array([30.0]), np.array([30.0]), np.array([0.0]),
    0, 0, 30, 60, 60, 40, 0.0,
    eps12p=1e-3, G=30e9, nu=0.25)
print(f"u1={u1[0]:.4e}, u2={u2[0]:.4e}, u3={u3[0]:.4e}")
# Expected: u1=2.0493e-03, u2=0.0000e+00, u3≈0
```

---

## Library search order

When `sbarbotfull.displacement()` is called, it searches for the compiled
library in this order:

1. `csi/bin/<platform>/` — pre-compiled shipped binaries
2. `csi/sbarbot_src/` — user-compiled in source directory
3. `csi/` — library placed next to `sbarbotfull.py`

If none is found, the pure-NumPy fallback is used automatically (with a
one-time warning).

---

## Source files

| File | Purpose |
|------|---------|
| `computeDisplacementVerticalStrainVolume.f90` | Core analytic solution (Barbot 2017) |
| `sbarbot_array_wrapper.f90` | Array-level wrapper for ctypes interface |
| `xlogy.f90` | Helper: x·log(y) with 0·log(0)=0 |
| `atan3.f90` | Helper: safe arctan for boundary cases |
| `build_sbarbot.py` | Build script (compile + optional install) |
| `sbarbot_python.py` | Pure-NumPy fallback (~640 lines) |

---

## Troubleshooting

**"gfortran: command not found"**
→ Install gfortran (see Dependencies above)

**DLL load error on Windows**
→ Ensure the gfortran runtime DLLs are on PATH, or use the same conda
environment where gfortran was installed

**Library found but wrong architecture**
→ Ensure gfortran and Python are both 64-bit (or both 32-bit)

**Performance identical to fallback**
→ Check that `sbarbotfull._lib` is not `"PYTHON_FALLBACK"`:
```python
from csi import sbarbotfull
print(sbarbotfull._load_library())  # should show a ctypes handle
```
