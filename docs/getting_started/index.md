# 快速开始

本部分负责完整 ECAT 安装、环境检查和最短可运行路线。

- [安装与环境检查](installation.md)：区分首次完整安装、创建前 BLAS/MPI配置、
  eqtools/CSI增量更新、optional extras和 MPI runtime检查。
- [安装与运行故障排查](troubleshooting.md)：按症状检查版本、wheel、导入路径、
  MPI runtime、BLAS 后端和线程性能，不永久修改用户环境。
- [计算运行栈](../concepts/compute_runtime_stack.md)：理解 oneAPI、oneMKL、
  OpenBLAS、mpi4py、Open MPI、MPICH、Intel MPI 和 MS-MPI 的层级与配套关系。
- [标准两阶段最短流程](quickstart_two_step.md)：从数据准备进入 Bayesian 非线性
  几何反演，再固定几何完成 BLSE/VCE 线性滑动反演。

第一次使用应先完成安装页的基础导入和 CLI 检查，再进入两阶段流程。
