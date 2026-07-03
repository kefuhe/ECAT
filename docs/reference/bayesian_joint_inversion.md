# Bayesian 联合反演参考 / Bayesian Joint Inversion

本文说明 `BayesianMultiFaultsInversion` 的联合 Bayesian 反演语义。它面向已经理解两步走路线、需要处理几何-滑动耦合、多源数据和高级约束的用户。

普通同震研究优先阅读 [Bayesian 非线性几何反演 / Nonlinear Geometry](../workflows/03_nonlinear_geometry_bayesian.md) 和 [BLSE/VCE 线性滑动分布反演 / Linear Slip](../workflows/04_linear_slip_blse_vce.md)。联合路线见 [Bayesian 联合几何-滑动分布反演 / Joint Bayesian Geometry-Slip](../workflows/05_joint_bayesian_geometry_slip.md)。

## 阅读路径

- 第一次做 ECAT 反演：不要从本页开始，先完成标准两步走 workflow。
- 已经完成两步走，想传播几何不确定性到滑动分布：先看 [入口定位](#入口定位) 和 [采样模式](#采样模式)。
- 不确定哪些参数被采样、哪些由线性求解得到：看 [参数分层](#参数分层)。
- 需要让断层几何随样本更新：看 [几何更新](#几何更新)，再进入 [可扰动断层几何](geometry_perturbation.md)。
- 正在检查案例配置：按 [配置检查顺序](#配置检查顺序) 逐项核对。

## 入口定位

| 入口 | 用途 |
| --- | --- |
| `BayesianMultiFaultsInversion` | 联合 Bayesian 主入口；管理多断层、多数据、SMC 采样和样本内求解 |
| `BayesianMultiFaultsInversionConfig` | 解析 `bayesian_sampling_mode`、`slip_sampling_mode`、sigma/alpha、GF、几何更新和约束 |
| `BayesianAdaptiveTriangularPatches` | 常用可扰动三角断层对象；提供几何基线、扰动方法和网格更新标志 |
| `ConstraintManagerSMC` | 管理 SMC 边界；在 `SMC_F_J + ss_ds` 中管理线性约束矩阵 |

联合 Bayesian 不是单独的“非线性几何搜索”。它可以让几何、滑动、sigma、alpha、poly 和多源约束在同一后验框架中耦合。

## 采样模式

| 字段 | 可选值 | 说明 |
| --- | --- | --- |
| `bayesian_sampling_mode` | `SMC_FJ`, `SMC_F_J`, `FULLSMC` | `SMC_FJ` 是配置别名，会被归一化为 `SMC_F_J` |
| `slip_sampling_mode` | `ss_ds`, `magnitude_rake`, `mag_rake`, `rake_fixed` | `mag_rake` 会归一化为 `magnitude_rake`；`SMC_F_J` 使用 `ss_ds` |

两种 Bayesian 模式的核心差异：

| 模式 | SMC 样本向量包含 | 滑动如何得到 | 约束特点 |
| --- | --- | --- | --- |
| `SMC_F_J` | 几何、sigma、alpha 等超参数 | 样本内通过线性求解得到 `strikeslip/dipslip` | 支持 rake、Euler、零滑、边界零滑和自定义 `A @ x` 线性约束 |
| `FULLSMC` | 几何、sigma、alpha 和滑动参数 | 直接从样本向量得到 | 主要通过参数边界和先验控制，不合并线性约束矩阵 |

`SMC_F_J` 维度通常较低，是联合几何-滑动反演的推荐高级路线。`FULLSMC` 适合小规模模型或需要显式研究滑动先验的场景，但计算成本和收敛诊断难度都更高。

## 参数分层

| 层级 | 参数 | 在 `SMC_F_J` 中 | 在 `FULLSMC` 中 |
| --- | --- | --- | --- |
| 几何 | 断层位置、走向、倾角、网格扰动参数 | SMC 采样 | SMC 采样 |
| 数据权重 | `sigmas` | SMC 采样 | SMC 采样 |
| 平滑尺度 | `alpha` | SMC 采样 | SMC 采样 |
| 滑动 | `strikeslip/dipslip` 或 `magnitude/rake` | 线性求解 | SMC 采样 |
| 数据修正 | `poly`、ramp 或 Euler rotation 等 | 线性求解 | SMC 采样或按配置处理 |

在 `SMC_F_J` 中，约束矩阵 `A` 的列只对应线性参数块，即滑动和 poly 参数；不包含 geometry、sigma 或 alpha。这个边界很重要，否则容易把几何先验和线性约束混写。

## 几何更新

几何更新通常写在每个断层的 `method_parameters.update_fault_geometry` 中：

```yaml
faults:
  FaultA:
    method_parameters:
      update_fault_geometry:
        method: perturb_BottomFixedDir_RotateTransGeom_simpleMesh
        disct_z: 10
        bias: 1.0
        angle_unit: degrees
```

运行时，采样器会把当前样本中的几何扰动参数传给 `update_fault_geometry`。配置里只写方法名和固定 kwargs，不写 `perturbations`。扰动方法、冻结基线和网格更新规则见 [Bayesian 联合反演中的可扰动断层几何 / Perturbable Fault Geometry](geometry_perturbation.md)。

## 约束适用性

| 约束 | `SMC_F_J + ss_ds` | `FULLSMC` |
| --- | :---: | :---: |
| `strikeslip/dipslip` 边界 | 是 | 是 |
| `rake_angle` 线性角度约束 | 是 | 否 |
| `slip_magnitude/rake_angle` 采样边界 | 否 | 视 `slip_sampling_mode` 而定 |
| Euler 约束 | 是 | 否 |
| 零滑、边界零滑 | 是 | 否 |
| 自定义 `A @ x <= b` / `A @ x = b` | 是 | 否 |
| 震级先验 | 否 | 是 |

完整写法和模式守卫见 [ECAT 约束管理器 / Constraint Manager](constraint_manager.md)。

## 配置检查顺序

配置联合 Bayesian 案例时，建议按这个顺序检查：

1. 数据对象顺序和 `geodata` 配置一致。
2. `bayesian_sampling_mode` 和 `slip_sampling_mode` 与科学问题一致。
3. 每个可扰动断层已经建立几何基线和网格初值。
4. `update_fault_geometry.method` 是该断层对象可发现的方法。
5. 几何扰动参数边界和方法期望参数个数一致。
6. GF 后端和网格更新成本在计算预算内。
7. sigma/alpha 的 `mode`、初值、边界和数据组顺序一致。
8. 约束类型与当前采样模式匹配。

## 结果报告

报告联合 Bayesian 结果时，至少说明：

- `bayesian_sampling_mode` 和 `slip_sampling_mode`。
- 使用的断层对象类型和几何扰动方法。
- 几何扰动参数、边界、单位和是否使用坐标加密。
- 网格策略、GF 方法和是否随样本重建 GF。
- 数据协方差、sigma/alpha 配置和先验范围。
- 约束类型，以及哪些约束只在 `SMC_F_J` 中生效。
- 几何后验、滑动后验和数据残差诊断。

## 相关页面

- [Bayesian 联合几何-滑动分布反演 / Joint Bayesian Geometry-Slip](../workflows/05_joint_bayesian_geometry_slip.md)
- [Bayesian 联合反演中的可扰动断层几何 / Perturbable Fault Geometry](geometry_perturbation.md)
- [ECAT 约束管理器 / Constraint Manager](constraint_manager.md)
- [Sigmas 与 Alpha 配置模式 / Sigmas and Alpha](sigmas_alpha.md)
