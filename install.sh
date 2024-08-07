#!/bin/bash

# 初始化和更新子模块 --remote: 保持和子模块的更新同步，如果不更新则移除
git submodule update --init --recursive --remote

# 安装eqtools
cd eqtools
pip install .
cd ..

# 安装csi_cutde_mpiparallel
cd csi_cutde_mpiparallel
pip install .
cd ..