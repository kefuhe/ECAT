@echo off

REM install eqtools
cd eqtools
pip install .
cd ..

REM install csi_cutde_mpiparallel
cd csi_cutde_mpiparallel
pip install .
cd ..