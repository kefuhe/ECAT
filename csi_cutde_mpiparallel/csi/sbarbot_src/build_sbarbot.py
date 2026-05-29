'''
Build script for the sbarbot vertical strain volume shared library.
Requires gfortran (e.g. via MinGW-w64 on Windows, or system gfortran
on Linux/Mac).

Usage:
    python build_sbarbot.py          # build in this directory
    python build_sbarbot.py --install  # build and copy to csi/bin/

Alternatively, from Python:
    from csi.sbarbot_src.build_sbarbot import build
    build()
'''

import subprocess
import platform
import os
import sys
import shutil


def build(install=False):
    """Compile the Fortran sources into a shared library.

    Parameters
    ----------
    install : bool
        If True, also copy the resulting library into the appropriate
        ``csi/bin/<platform>/`` directory.

    Returns
    -------
    str
        Path to the compiled library.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))

    src_files = [
        os.path.join(src_dir, 'xlogy.f90'),
        os.path.join(src_dir, 'atan3.f90'),
        os.path.join(src_dir, 'computeDisplacementVerticalStrainVolume.f90'),
        os.path.join(src_dir, 'sbarbot_array_wrapper.f90'),
    ]

    # Check source files exist
    for f in src_files:
        if not os.path.isfile(f):
            print(f"Error: source file not found: {f}")
            sys.exit(1)

    system = platform.system()
    if system == 'Windows':
        out_name = 'sbarbot.dll'
        cmd = ['gfortran', '-shared', '-cpp', '-O2'] + src_files + [
            '-o', os.path.join(src_dir, out_name)]
    elif system == 'Darwin':
        out_name = 'libsbarbot.dylib'
        cmd = ['gfortran', '-shared', '-fPIC', '-cpp', '-O2'] + src_files + [
            '-o', os.path.join(src_dir, out_name)]
    else:
        out_name = 'libsbarbot.so'
        cmd = ['gfortran', '-shared', '-fPIC', '-cpp', '-O2'] + src_files + [
            '-o', os.path.join(src_dir, out_name)]

    print("Compiling sbarbot shared library...")
    print(f"  Command: {' '.join(cmd)}")

    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        print("Error: gfortran not found. Please install gfortran:")
        if system == 'Windows':
            print("  conda install -c conda-forge m2w64-gcc-fortran")
            print("  or install MinGW-w64 and add gfortran to PATH")
        elif system == 'Darwin':
            print("  brew install gcc")
        else:
            print("  sudo apt install gfortran")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        sys.exit(1)

    out_path = os.path.join(src_dir, out_name)
    print(f"  Output: {out_path}")

    if install:
        csi_dir = os.path.dirname(src_dir)  # csi/
        if system == 'Windows':
            dst_dir = os.path.join(csi_dir, 'bin', 'windows')
        elif system == 'Darwin':
            dst_dir = os.path.join(csi_dir, 'bin', 'macos')
        else:
            dst_dir = os.path.join(csi_dir, 'bin', 'ubuntu20.04')
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, out_name)
        shutil.copy2(out_path, dst_path)
        print(f"  Installed to: {dst_path}")

    print("Done.")
    return out_path


if __name__ == '__main__':
    do_install = '--install' in sys.argv
    build(install=do_install)
