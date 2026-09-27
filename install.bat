@echo off
setlocal
REM git submodule update --init --recursive  (Optional ECAT-Cases download.)
set "PYTHON_BIN=python"
if defined PYTHON set "PYTHON_BIN=%PYTHON%"
set "REPO_DIR=%~dp0"
"%PYTHON_BIN%" -c "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), 'ECAT supports CPython 3.10, 3.11, and 3.12 only.'"
if errorlevel 1 exit /b 1
"%PYTHON_BIN%" -c "import okada4py" >nul 2>&1
if errorlevel 1 (
    echo Missing required dependency: okada4py. Install a matching wheel; see Install.md.
    exit /b 1
)
REM Resolve the three local components together; ecat-viz need not be on PyPI.
"%PYTHON_BIN%" -m pip install "%REPO_DIR%ecat-viz" "%REPO_DIR%csi_cutde_mpiparallel" "%REPO_DIR%eqtools"
if errorlevel 1 exit /b 1
REM Restore CSI-owned scripts after old eqtools uninstall records may remove them.
"%PYTHON_BIN%" -m pip install --no-deps --force-reinstall "%REPO_DIR%csi_cutde_mpiparallel"
if errorlevel 1 exit /b 1
"%PYTHON_BIN%" -c "import ecat_viz, csi, eqtools; print('ECAT package imports succeeded.')"
if errorlevel 1 exit /b 1
echo Installation complete. See Install.md for component updates and optional extras.
endlocal
