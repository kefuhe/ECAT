@echo off
setlocal

REM Uncomment the following line to download the full ECAT case library.
REM git submodule update --init --recursive

set "PYTHON_BIN=python"
set "REPO_DIR=%~dp0"

%PYTHON_BIN% -c "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), 'ECAT supports CPython 3.10, 3.11, and 3.12 only.'"
if errorlevel 1 exit /b 1

%PYTHON_BIN% -c "import okada4py" >nul 2>&1
if errorlevel 1 (
    echo Missing required dependency: okada4py.
    echo Install a wheel matching the active CPython version and your platform before running this script.
    echo See Install.md for the wheel and source-build instructions.
    exit /b 1
)

REM Install CSI first because eqtools imports it for its main workflows.
%PYTHON_BIN% -m pip install "%REPO_DIR%csi_cutde_mpiparallel"
if errorlevel 1 exit /b 1
%PYTHON_BIN% -m pip install "%REPO_DIR%eqtools"
if errorlevel 1 exit /b 1

%PYTHON_BIN% -c "import csi, eqtools; print('ECAT package imports succeeded.')"
if errorlevel 1 exit /b 1

echo Installation complete. See Install.md for eqtools incremental updates and optional extras.
endlocal
