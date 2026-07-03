# ECAT 用户手册

[ECAT](https://github.com/kefuhe/ECAT) 是面向地震大地测量建模与反演的科研工具集。公开代码包含 `eqtools` 和 `csi` 相关扩展；本手册按实际科研流程组织，而不是按源码目录组织。

标准入门路线是两步走：

1. **Bayesian 非线性几何反演**：估计断层顶边中点经纬度和深度、走向、倾角、长度、宽度等几何参数。
2. **BLSE/VCE 线性滑动分布反演**：固定优选几何后，反演分布式滑动。

数据读取、SAR/InSAR 降采样和断层几何构建是这两步之前的准备工作。高级用户可在完成标准两步走检查后继续阅读 Bayesian 联合几何-滑动分布反演。

## 如何导航

| 你想做什么 | 先读 | 说明 |
| --- | --- | --- |
| 第一次安装或跑通最短路线 | [安装与环境检查](getting_started/installation.md), [标准两步走路线](getting_started/quickstart_two_step.md) | 从环境、数据读取、非线性几何到线性滑动建立主线 |
| 理解为什么这样组织 | [核心概念入口](concepts/index.md) | 解释两步走、SAR 投影和断层几何状态 |
| 复制一个小任务的代码 | [任务短例入口](examples/index.md) | trace 预处理、GAMMA quick-look、BLSE 最小脚本和地表正演 |
| 对照真实事件脚本和输出 | [案例选择表](casebook/index.md) | 根据数据类型和反演阶段选择 ECAT-Cases 中的案例 |
| 查命令、字段、API 或误区 | [Reference Map](reference/index.md) | 确认 CLI、配置、reader、约束、几何和输出细节 |

## 任务地图

| 科研任务 | 主线页面 | 参考与案例 |
| --- | --- | --- |
| 读取 InSAR/GPS 数据 | [InSAR 和 GPS 数据读取](workflows/01_data_reading_insar_gps.md) | [SAR Reader](reference/sar_reader.md), [SAR 投影约定](concepts/sar_projection_conventions.md) |
| 从 SAR/offset 生成反演输入 | [InSAR 降采样](workflows/02_insar_downsampling.md) | [Downsampling App](reference/downsampling_app.md), [GAMMA quick-look](examples/gamma_sar_quicklook.md) |
| 自定义读取或时序 InSAR 复用网格 | [Adapter 降采样](workflows/02b_adapter_downsampling.md) | [Downsampling App: `input_adapter`](reference/downsampling_app.md#input_adapter) |
| Bayesian 非线性几何反演 | [非线性几何反演](workflows/03_nonlinear_geometry_bayesian.md) | [非线性几何配置](reference/config_nonlinear_geometry.md), [数据改正项](reference/data_corrections.md), [两步走概念](concepts/two_step_inversion.md) |
| 构建断层几何和 mesh | [Fault Geometry Construction](reference/fault_geometry_construction.md) | [Fault Summary](reference/fault_summary.md), [Fault Edges](reference/fault_edges.md), [Fault Patch Indices](reference/fault_patch_indices.md) |
| BLSE/VCE 线性滑动反演 | [BLSE/VCE 线性滑动反演](workflows/04_linear_slip_blse_vce.md) | [线性滑动配置](reference/config_linear_slip.md), [数据改正项](reference/data_corrections.md), [约束管理器](reference/constraint_manager.md), [Dingri 2020](casebook/dingri_blse_vce.md) |
| 震间 loading/backslip/coupling 解释 | [Interseismic Kinematics](reference/interseismic_kinematics.md) | [Fault Patch Indices](reference/fault_patch_indices.md), [Constraint Manager](reference/constraint_manager.md) |
| 高级联合 Bayesian 几何-滑动反演 | [Joint Bayesian Geometry-Slip](workflows/05_joint_bayesian_geometry_slip.md) | [Bayesian Joint Inversion](reference/bayesian_joint_inversion.md), [Perturbable Fault Geometry](reference/geometry_perturbation.md) |
| 生成地表位移或统一图件格式 | [Surface Forward Example](examples/surface_forward_grid.md) | [Surface Displacement Forward](reference/surface_displacement_forward.md), [Viztools](reference/viztools.md) |

## 文档目录

### Getting Started

- [安装与环境检查](getting_started/installation.md)
- [标准两步走路线](getting_started/quickstart_two_step.md)

### Concepts

- [核心概念入口](concepts/index.md)
- [标准两步走反演逻辑](concepts/two_step_inversion.md)
- [断层几何状态](concepts/fault_geometry_states.md)
- [SAR 投影和观测约定](concepts/sar_projection_conventions.md)

### Workflows

- [InSAR 和 GPS 数据读取](workflows/01_data_reading_insar_gps.md)
- [InSAR 降采样](workflows/02_insar_downsampling.md)
- [InSAR 降采样 Step1/Step2 调参](workflows/02a_insar_downsampling_two_step.md)
- [自定义读取 Adapter 降采样](workflows/02b_adapter_downsampling.md)
- [Bayesian 非线性几何反演](workflows/03_nonlinear_geometry_bayesian.md)
- [BLSE/VCE 线性滑动分布反演](workflows/04_linear_slip_blse_vce.md)
- [Bayesian 联合几何-滑动分布反演](workflows/05_joint_bayesian_geometry_slip.md)

### Examples

- [任务短例入口](examples/index.md)
- [Trace 预处理与断层顶部边界](examples/fault_trace_preprocessing.md)
- [非线性几何结果到 fault object](examples/fault_from_nonlinear_geometry.md)
- [GAMMA SAR quick-look 与配置生成](examples/gamma_sar_quicklook.md)
- [BLSE/VCE 最小脚本骨架](examples/blse_minimal_run.md)
- [地表形变正演最小例子](examples/surface_forward_grid.md)

### Casebook

案例放在 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases) 仓库中，主仓库 [ECAT](https://github.com/kefuhe/ECAT) 维护代码、方法手册与接口参考。

- [案例选择表](casebook/index.md)
- [Wushi: InSAR-only 非线性几何反演](casebook/wushi_nonlinear_geometry.md)
- [Ridgecrest: GPS+InSAR 非线性几何反演](casebook/ridgecrest_gps_insar.md)
- [Dingri 2020: BLSE/VCE 线性滑动反演](casebook/dingri_blse_vce.md)
- [InSAR/Offset 降采样案例](casebook/insar_downsampling_gamma_geotiff.md)

### Reference

- [Reference Map](reference/index.md)
- [CLI Reference](reference/cli.md)
- [SAR Reader](reference/sar_reader.md)
- [Data Corrections](reference/data_corrections.md)
- [Downsampling App](reference/downsampling_app.md)
- [Nonlinear Config](reference/config_nonlinear_geometry.md)
- [Linear Slip Config](reference/config_linear_slip.md)
- [Fault Geometry Construction](reference/fault_geometry_construction.md)
- [Fault Summary](reference/fault_summary.md)
- [Fault Edges](reference/fault_edges.md)
- [Fault Patch Indices](reference/fault_patch_indices.md)
- [Fault Contours](reference/fault_contours.md)
- [Sigmas and Alpha](reference/sigmas_alpha.md)
- [Constraint Manager](reference/constraint_manager.md)
- [Interseismic Kinematics](reference/interseismic_kinematics.md)
- [BLSE/VCE](reference/blse_vce.md)
- [Surface Displacement Forward](reference/surface_displacement_forward.md)
- [Bayesian Joint Inversion](reference/bayesian_joint_inversion.md)
- [Perturbable Fault Geometry](reference/geometry_perturbation.md)
- [Viztools](reference/viztools.md)

### Developer

- [Documentation Architecture](developer/architecture.md)
- [Documentation Guidelines](developer/contributing_docs.md)

## 手册范围

本手册放在 [ECAT 仓库 `docs/`](https://github.com/kefuhe/ECAT/tree/main/docs)，按学习路线组织常用工作流、案例导读和字段参考。真实数据和可运行脚本放在 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases)。
