# csi

Classic Slip Inversion (CSI)  
_Pythonic Version with Layered and Triangular Green's Function Support_

---

> ⚡️ **Based on [CSI](https://github.com/jolivetr/csi), with further compatibility improvements, optimizations, and feature extensions by Kefeng He.**  
> **Note:** This version is **not fully compatible** with the original CSI. Please be aware that the two versions are not direct drop-in replacements.

---

## ✨ Features

- **🟩 Green's Function Calculation**
  - 🚀 Parallel computation of triangular dislocation element Green's functions using [cutde](https://github.com/cutde-org/cutde).
  - 🏗️ Layered Green's function calculation via EDCMP ([original repo](https://github.com/RongjiangWang/EDGRN_EDCMP_2.0)) and PSCMP ([original repo](https://github.com/RongjiangWang/PSGRN-PSCMP_2020)).
  - 📦 EDCMP and PSCMP binaries are pre-packaged in `csi/bin`:
    - 🪟 Windows: `.exe` files.
    - 🐧 Linux: binaries compiled under Ubuntu 20.04 (other Linux distributions may require manual compilation; see below).

---

## 🚦 Installation and Usage Notes

- **If you only need homogeneous (non-layered) Green's function calculation:**  
  👉 Simply run  
  ```bash
  pip install .
  ```
  No additional binaries or compilation are required.

- **If you need layered Green's function calculation (EDCMP/PSCMP):**
  - On **Windows**:  
    Pre-built `.exe` binaries are included and will be used automatically.
  - On **Linux**:  
    - Binaries for Ubuntu 20.04 are included by default.
    - If you are on another Linux distribution, you may need to compile EDCMP/PSCMP yourself:
      1. Compile the binaries on your platform (see below for source and patch instructions).
      2. Replace the binaries in the corresponding `csi/bin` subfolder.
      3. Then run `pip install .` to install the package.

---

## 🛠️ Compiling and Binary Notes

- **PSCMP Dependency Notice**
  - PSCMP binaries are compiled from [pyrocko/fomosto-psgrn-pscmp](https://github.com/pyrocko/fomosto-psgrn-pscmp).
  - **Before compiling**, you must modify the input file reading section in the Fortran source code as follows (not present in the original code, you need to update manually):

    <details>
    <summary>psgmain.f</summary>

    ```fortran
    write(*,'(a,$)') ' Please type the file name of input data: '
    c      read(*,'(a)')inputfile
    call getarg(1, inputfile)
    write(*,*) inputfile
    runtime=time()
    open(10, file=inputfile, status='old')
    ```

    </details>

    <details>
    <summary>pscmain.f</summary>

    ```fortran
    write(*,'(a,$)') ' Please type the file name of input data: '
    call getarg(1, infile)
    write(*,*) infile
    open(10, file=infile, status='old')
    ```

    </details>

    Similarly, update `edgmain.f` and `edcmain.f` to use `getarg` for input file arguments before compiling.

- **Compiling EDCMP/EDGRN**
  - For EDCMP/EDGRN, simply use `gfortran` to compile the source code.

- **🐧 Linux Library Dependencies**
  - For PSCMP binaries on Ubuntu 20.04, you may need to install `libgfortran3` and `gcc-6-base`.  
    See this [gist for details](https://gist.github.com/sakethramanujam/faf5b677b6505437dbdd82170ac55322#installing-libgfortran3-on-ubuntu-2004).
    - Download:
      - [`libgfortran3`](http://archive.ubuntu.com/ubuntu/pool/universe/g/gcc-6/libgfortran3_6.4.0-17ubuntu1_amd64.deb)
      - [`gcc-6-base`](http://archive.ubuntu.com/ubuntu/pool/universe/g/gcc-6/gcc-6-base_6.4.0-17ubuntu1_amd64.deb)
    - Install in order:
      ```bash
      sudo dpkg -i gcc-6-base_6.4.0-17ubuntu1_amd64.deb
      sudo dpkg -i libgfortran3_6.4.0-17ubuntu1_amd64.deb
      ```
    - ⚠️ _This may affect your existing GCC installation. Proceed with caution._

---

## 🖥️ Command Line Tools via ECAT

After installing the full [ECAT package](https://github.com/kefuhe/ECAT), the following commands are available for direct use in the terminal:

| 🛠️ Command                        | 📄 Description                        |
|------------------------------------|---------------------------------------|
| `ecat-psgrn`                       | Run PSGRN                             |
| `ecat-pscmp`                       | Run PSCMP                             |
| `ecat-edgrn`                       | Run EDGRN                             |
| `ecat-edcmp`                       | Run EDCMP                             |
| `ecat-generate-psgrn-template`     | Generate PSGRN input file template    |
| `ecat-generate-pscmp-template`     | Generate PSCMP input file template    |
| `ecat-generate-edgrn-template`     | Generate EDGRN input file template    |
| `ecat-generate-edcmp-template`     | Generate EDCMP input file template    |

These commands correspond to running the respective programs or generating input file templates for each module.

---

## 📚 References

- [CSI (original)](https://github.com/jolivetr/csi)
- [cutde](https://github.com/cutde-org/cutde)
- [EDGRN/EDCMP (original)](https://github.com/RongjiangWang/EDGRN_EDCMP_2.0)
- [PSGRN/PSCMP (original)](https://github.com/RongjiangWang/PSGRN-PSCMP_2020)
- [Fomosto PSCMP (repackaged)](https://github.com/pyrocko/fomosto-psgrn-pscmp)

---

For usage instructions and examples, see the documentation and code comments.