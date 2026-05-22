# 文档架构说明

用户手册按科研工作流组织，源码按可复用组件组织。两者不要混在一起。

## 主要实现位置

| 功能 | 代码位置 |
| --- | --- |
| SAR 读取 | [eqtools/csiExtend/sarUtils](https://github.com/kefuhe/ECAT/tree/main/eqtools/csiExtend/sarUtils) |
| 降采样 CLI | [eqtools/cli_tools/process_data_downsampling.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/cli_tools/process_data_downsampling.py) |
| 降采样配置校验 | [eqtools/csiExtend/downsample/config.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/downsample/config.py) |
| 非线性几何反演 | [eqtools/csiExtend/exploremultifaults_smc.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/exploremultifaults_smc.py) |
| BLSE/VCE | [eqtools/csiExtend/blse_multifaults_inversion.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/blse_multifaults_inversion.py) |
| 线性求解器 | [eqtools/csiExtend/multifaultsolve_boundLSE.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/multifaultsolve_boundLSE.py) |
| VCE 算法 | [simple_vce.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/simple_vce.py), [rigorous_vce.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/rigorous_vce.py) |

## 文档边界

[ECAT 仓库 `docs/`](https://github.com/kefuhe/ECAT/tree/main/docs) 是用户手册，目标读者是科研用户和新学生。它应该稳定、有顺序、可学习。

[ECAT 仓库 `eqtools/csiExtend/docs/`](https://github.com/kefuhe/ECAT/tree/main/eqtools/csiExtend/docs) 是技术材料目录。可以作为用户手册的素材来源，但不要把设计笔记原样堆到手册中。

## 手册当前覆盖

- InSAR/GPS 数据读取
- InSAR 降采样
- Bayesian 非线性几何反演
- BLSE/VCE 线性滑动分布反演
