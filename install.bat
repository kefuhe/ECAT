@echo off

REM Uncomment the following line to download the full ECAT case library
REM git submodule update --init --recursive

REM install eqtools
cd eqtools
pip install .
cd ..

REM install csi_cutde_mpiparallel
cd csi_cutde_mpiparallel
pip install .
cd ..

echo Installation complete. If you need the full ECAT case library, uncomment the submodule command in this script or download it manually.
pause