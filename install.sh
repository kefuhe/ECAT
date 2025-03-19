#!/bin/bash

# The full ECAT case library, including research cases and advanced examples, 
# is stored in a separate repository. 
# To download the full case library, run:
# git submodule update --init --recursive

# install eqtools
cd eqtools
pip install .
cd ..

# install csi_cutde_mpiparallel
cd csi_cutde_mpiparallel
pip install .
cd ..