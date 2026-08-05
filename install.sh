#!/usr/bin/env bash
set -euo pipefail

# Uncomment the following line to download the full ECAT case library.
# git submodule update --init --recursive

repo_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
python_bin="${PYTHON:-python}"

"$python_bin" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "ECAT supports CPython 3.10, 3.11, and 3.12 only."'

if ! "$python_bin" -c 'import okada4py' >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Missing required dependency: okada4py.
Install a wheel matching the active CPython version and your platform before
running this script. See Install.md for the wheel and source-build instructions.
EOF
    exit 1
fi

# Install CSI first because eqtools imports it for its main workflows.
"$python_bin" -m pip install "$repo_dir/csi_cutde_mpiparallel"
"$python_bin" -m pip install "$repo_dir/eqtools"

"$python_bin" -c 'import csi, eqtools; print("ECAT package imports succeeded.")'
echo "Installation complete. See Install.md for eqtools incremental updates and optional extras."
