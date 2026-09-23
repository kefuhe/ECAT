#!/usr/bin/env bash
#
# Run one ECAT MPI script in WSL/Linux with controlled numerical threads.
# Activate the intended Conda environment before invoking this launcher.
#
# Examples:
#   conda activate ecat
#   PYTHON_SCRIPT=test_smc.py ./run_ecat_mpi_wsl.sh
#   RANKS=20 THREADS_PER_RANK=2 PYTHON_SCRIPT=test_smc.py \
#       ./run_ecat_mpi_wsl.sh -r
#
# Positional arguments are forwarded unchanged to the Python script. Exported
# variables exist only in this launcher process and its MPI children.

set -Eeuo pipefail

# -------------------------- User settings --------------------------

RANKS="${RANKS:-4}"
THREADS_PER_RANK="${THREADS_PER_RANK:-1}"
CASE_DIR="${CASE_DIR:-$PWD}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-test_smc.py}"
PYTHON_EXE="${PYTHON_EXE:-python}"
MPIEXEC="${MPIEXEC:-mpiexec}"
INTERACTIVE_PLOTS="${INTERACTIVE_PLOTS:-0}"

# ----------------------- Numerical threading -----------------------

export OMP_NUM_THREADS="$THREADS_PER_RANK"
export MKL_NUM_THREADS="$THREADS_PER_RANK"
export OPENBLAS_NUM_THREADS="$THREADS_PER_RANK"
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE

# Keep independent helper pools serial unless the scientific script explicitly
# owns their parallelism.
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

if [[ "$INTERACTIVE_PLOTS" != "1" ]]; then
    export MPLBACKEND=Agg
fi

# ----------------------------- Checks ------------------------------

if ! [[ "$RANKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "RANKS must be a positive integer: $RANKS" >&2
    exit 2
fi
if ! [[ "$THREADS_PER_RANK" =~ ^[1-9][0-9]*$ ]]; then
    echo "THREADS_PER_RANK must be a positive integer: $THREADS_PER_RANK" >&2
    exit 2
fi
if [[ ! -d "$CASE_DIR" ]]; then
    echo "Case directory does not exist: $CASE_DIR" >&2
    exit 2
fi
if [[ ! -f "$CASE_DIR/$PYTHON_SCRIPT" ]]; then
    echo "Python script does not exist: $CASE_DIR/$PYTHON_SCRIPT" >&2
    exit 2
fi
if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
    echo "Python executable was not found: $PYTHON_EXE" >&2
    exit 2
fi
if ! command -v "$MPIEXEC" >/dev/null 2>&1; then
    echo "MPI launcher was not found: $MPIEXEC" >&2
    exit 2
fi

# ------------------------------ Run --------------------------------

cd "$CASE_DIR"

echo "ECAT MPI launcher (WSL/Linux)"
echo "  case directory : $CASE_DIR"
echo "  Python script  : $PYTHON_SCRIPT"
echo "  MPI ranks      : $RANKS"
echo "  threads/rank   : $THREADS_PER_RANK"
echo "  thread budget  : $((RANKS * THREADS_PER_RANK))"

started_at=$(date +%s)
if "$MPIEXEC" -n "$RANKS" "$PYTHON_EXE" "./$PYTHON_SCRIPT" "$@"; then
    status=0
else
    status=$?
fi
elapsed=$(( $(date +%s) - started_at ))

echo "ECAT MPI launcher finished; wall time: ${elapsed}s; exit code: $status"
exit "$status"
