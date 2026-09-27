#!/usr/bin/env bash
set -euo pipefail

# git submodule update --init --recursive  # Optional ECAT-Cases download.
repo_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
python_bin="${PYTHON:-python}"
"$python_bin" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "ECAT supports CPython 3.10, 3.11, and 3.12 only."'
if ! "$python_bin" -c 'import okada4py' >/dev/null 2>&1; then
    echo "Missing required dependency: okada4py. Install a matching wheel; see Install.md." >&2
    exit 1
fi
# Resolve all local components together, including ecat-viz before its release on PyPI.
"$python_bin" -m pip install "$repo_dir/ecat-viz" "$repo_dir/csi_cutde_mpiparallel" "$repo_dir/eqtools"
# Restore CSI scripts after legacy shared command ownership during upgrades.
"$python_bin" -m pip install --no-deps --force-reinstall "$repo_dir/csi_cutde_mpiparallel"
"$python_bin" -c 'import ecat_viz, csi, eqtools; print("ECAT package imports succeeded.")'
echo "Installation complete. See Install.md for component updates and optional extras."
