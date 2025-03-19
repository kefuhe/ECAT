@echo off

REM The full ECAT case library, including research cases and advanced examples, 
REM is stored in a separate repository. 
REM To download the full case library, run:
REM git submodule update --init --recursive

REM install eqtools
cd eqtools
pip install .
cd ..

REM install csi_cutde_mpiparallel
cd csi_cutde_mpiparallel
pip install .
cd ..