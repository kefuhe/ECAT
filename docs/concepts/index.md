# Concepts / 核心概念

本目录解释跨多个工作流反复出现的概念。它回答“为什么这样组织”和“这些对象之间是什么关系”，不替代具体命令和字段参考。

## 概念地图

| 问题 | 先读 |
| --- | --- |
| 为什么推荐先非线性几何、再线性滑动 | [标准两阶段反演逻辑](two_step_inversion.md) |
| trace、top/bottom、layers、mesh 和 patch 有什么区别 | [断层几何状态](fault_geometry_states.md) |
| 联合 Bayesian 中参考几何、样本状态和候选 mesh 是什么关系 | [Bayesian 联合反演中的几何参考](bayesian_geometry_reference.md) |
| `strike/dip/rake` 如何配套解释，跨 `90°` 时怎样进入 CSI | [断层走向、倾角与滑动基底约定](fault_angle_conventions.md) |
| SAR/offset 数据的正负号和 LOS projection 怎么理解 | [SAR 投影和观测约定](sar_projection_conventions.md) |
| `d/G/Cd/H` 的行、列和 GPS/optical 分量顺序怎样对齐 | [观测向量、协方差与设计矩阵排列合同](observation_matrix_layout.md) |
| Python 进程、MPI rank、线程、CPU affinity 和前置环境变量有什么区别 | [进程、MPI Rank、线程与 CPU 亲和性](parallel_process_rank_thread.md) |
| oneAPI、MKL、OpenBLAS、mpi4py 和不同 MPI 实现是什么关系 | [Python 数值计算、BLAS、MPI 与 oneAPI 的层级](compute_runtime_stack.md) |

## 和其他文档层的关系

- `concepts/` 解释概念和判断逻辑。
- `workflows/` 给可执行步骤。
- `examples/` 给短小可复制代码。
- `reference/` 给完整字段、参数、API 和误区。
- `casebook/` 对应真实事件脚本和数据。
