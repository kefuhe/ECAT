#!/bin/bash

# 安装eqtools
cd eqtools
pip install .
cd ..

# 安装csi_cutde_mpiparallel
cd csi_cutde_mpiparallel
pip install .
cd ..