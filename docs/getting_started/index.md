# 快速开始

本部分负责完整 ECAT 安装、环境检查和最短可运行路线。

- [安装与环境检查](installation.md)：区分 Windows PowerShell 与 Linux/WSL，选择
  官方源或中国大陆命令级镜像，并说明首次完整安装、创建前 BLAS/MPI 配置、
  eqtools/CSI 增量更新、optional extras 和 MPI runtime 检查。
- [标准两阶段最短流程](quickstart_two_step.md)：从数据准备进入 Bayesian 非线性
  几何反演，再固定几何完成 BLSE/VCE 线性滑动反演。
- [并行运行基础](../concepts/parallel_process_rank_thread.md)：第一次遇到
  `mpiexec -n N` 时，理解进程、rank、线程、CPU affinity 和 Windows/WSL/Linux
  的一般差异，并复制不绑定 MPI 实现的检查模板。
- [安装与运行故障排查](troubleshooting.md)：默认命令失败或速度异常时，按症状检查
  版本、wheel、导入路径、MPI runtime、BLAS 后端和线程性能。
- [计算运行栈](../concepts/compute_runtime_stack.md)：进阶理解 oneAPI、oneMKL、
  OpenBLAS、mpi4py、Open MPI、MPICH、Intel MPI 和 MS-MPI 的层级与配套关系。

第一次使用应先完成安装页的基础导入和 CLI 检查，再进入两阶段流程。普通 BLSE
直接运行脚本，Bayesian SMC 先按案例的 `-n 4` 命令跑通；只有运行较慢、内存紧张
或准备扩大进程数时，才进入并行基础和故障排查页调优；只有需要选择或更换底层
实现时，再阅读计算运行栈。
