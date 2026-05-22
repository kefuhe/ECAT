# ECAT 用户手册

[ECAT](https://github.com/kefuhe/ECAT) 是面向地震大地测量建模与反演的科研工具集。对外发布的 ECAT 代码包包含 `eqtools` 和 `csi` 两个 Python 子包；本手册按实际同震形变研究流程组织，而不是按源码目录组织。

本手册的反演主线按两步组织，数据读取和 InSAR 降采样是前置准备：

1. **Bayesian 非线性几何反演**：估计断层顶边中点经纬度和深度、走向、倾角、长度、宽度等几何参数。
2. **BLSE/VCE 线性滑动分布反演**：固定优选几何后，反演分布式滑动。

## 如何使用

本手册按“快速上手 -> 工作流 -> 案例 -> 参考”的关系组织：

- **入门页**给最短路径，回答“先装什么、先跑什么”。
- **工作流页**讲每一步的目的、输入输出、关键代码和下一步。
- **案例页**把工作流对应到 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases) 中的真实脚本、数据和输出。
- **参考页**解释 CLI、配置字段、reader、sigma/alpha、约束管理器和 BLSE/VCE 细节。

## 检索地图

| 目标 | 先读 | 参考案例 | 细节参考 |
| --- | --- | --- | --- |
| 安装并跑通最小路线 | [安装与环境检查](getting_started/installation.md), [标准两步走路线](getting_started/quickstart_two_step.md) | [Wushi](casebook/wushi_nonlinear_geometry.md), [Dingri](casebook/dingri_blse_vce.md) | [CLI 命令参考](reference/cli.md) |
| 读入 InSAR/GPS 数据 | [InSAR 与 GPS 数据读取](workflows/01_data_reading_insar_gps.md) | [Ridgecrest](casebook/ridgecrest_gps_insar.md), [Wushi](casebook/wushi_nonlinear_geometry.md) | [SAR Reader 参考](reference/sar_reader.md) |
| 从原始 SAR/offset 生成反演输入 | [InSAR 降采样](workflows/02_insar_downsampling.md), [降采样两步走](workflows/02a_insar_downsampling_two_step.md) | [InSAR 降采样案例](casebook/insar_downsampling_gamma_geotiff.md) | [CLI 命令参考](reference/cli.md), [SAR Reader 参考](reference/sar_reader.md) |
| 做 Bayesian 非线性几何反演 | [Bayesian 非线性几何反演](workflows/03_nonlinear_geometry_bayesian.md) | [Wushi](casebook/wushi_nonlinear_geometry.md), [Ridgecrest](casebook/ridgecrest_gps_insar.md) | [非线性几何反演配置](reference/config_nonlinear_geometry.md), [Sigmas 与 Alpha 配置模式](reference/sigmas_alpha.md) |
| 做 BLSE/VCE 线性滑动反演 | [BLSE/VCE 线性滑动分布反演](workflows/04_linear_slip_blse_vce.md) | [Dingri 2020](casebook/dingri_blse_vce.md) | [线性滑动反演配置](reference/config_linear_slip.md), [ECAT 约束管理器](reference/constraint_manager.md), [BLSE/VCE 参考](reference/blse_vce.md) |

## 入门

- [安装与环境检查](getting_started/installation.md)
- [标准两步走路线](getting_started/quickstart_two_step.md)

## 核心工作流

- [InSAR 与 GPS 数据读取](workflows/01_data_reading_insar_gps.md)
- [InSAR 降采样](workflows/02_insar_downsampling.md)
- [InSAR 降采样两步走](workflows/02a_insar_downsampling_two_step.md)
- [Bayesian 非线性几何反演](workflows/03_nonlinear_geometry_bayesian.md)
- [BLSE/VCE 线性滑动分布反演](workflows/04_linear_slip_blse_vce.md)

## 案例导读

案例放在 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases) 仓库中，主仓库 [ECAT](https://github.com/kefuhe/ECAT) 维护代码、方法手册与接口参考。

- [Wushi：InSAR-only 非线性几何反演](casebook/wushi_nonlinear_geometry.md)
- [Ridgecrest：GPS+InSAR 非线性几何反演](casebook/ridgecrest_gps_insar.md)
- [Dingri 2020：BLSE/VCE 线性滑动反演](casebook/dingri_blse_vce.md)
- [InSAR 降采样案例](casebook/insar_downsampling_gamma_geotiff.md)

## 参考手册

- [CLI 命令参考](reference/cli.md)
- [SAR Reader 参考](reference/sar_reader.md)
- [非线性几何反演配置](reference/config_nonlinear_geometry.md)
- [线性滑动反演配置](reference/config_linear_slip.md)
- [Sigmas 与 Alpha 配置模式](reference/sigmas_alpha.md)
- [ECAT 约束管理器](reference/constraint_manager.md)
- [BLSE/VCE 参考](reference/blse_vce.md)

## 维护说明

- [文档架构说明](developer/architecture.md)
- [文档维护规范](developer/contributing_docs.md)

## 技术材料

[ECAT 仓库的 `eqtools/csiExtend/docs/`](https://github.com/kefuhe/ECAT/tree/main/eqtools/csiExtend/docs) 中已有大量技术材料，可作为实现细节和开发参考。面向用户的手册放在 [ECAT 仓库 `docs/`](https://github.com/kefuhe/ECAT/tree/main/docs)，并按学习路线重新组织。
