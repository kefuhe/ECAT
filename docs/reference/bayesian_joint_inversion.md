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
| `ConstraintManagerSMC` | 管理 SMC 边界；在 `SMC_FJ + ss_ds` 中管理线性约束矩阵 |

联合 Bayesian 不是单独的“非线性几何搜索”。它可以让几何、滑动、sigma、alpha、poly 和多源约束在同一后验框架中耦合。

## 采样模式

| 字段 | 可选值 | 说明 |
| --- | --- | --- |
| `bayesian_sampling_mode` | `SMC_FJ`, `FULLSMC` | 解析后只保存这两个规范值 |
| `slip_sampling_mode` | `ss_ds`, `magnitude_rake`, `mag_rake`, `rake_fixed` | `mag_rake` 会归一化为 `magnitude_rake`；`SMC_FJ` 使用 `ss_ds` |

两种 Bayesian 模式的核心差异：

| 模式 | SMC 样本向量包含 | 滑动如何得到 | 约束特点 |
| --- | --- | --- | --- |
| `SMC_FJ` | 几何、sigma、alpha 等超参数 | 样本内通过线性求解得到 `strikeslip/dipslip` | 支持 rake、Euler、零滑、边界零滑和自定义 `A @ x` 线性约束 |
| `FULLSMC` | 几何、sigma、alpha 和滑动参数 | 直接从样本向量得到 | 主要通过参数边界和先验控制，不合并线性约束矩阵 |

`SMC_FJ` 维度通常较低，是联合几何-滑动反演的推荐高级路线。`FULLSMC` 适合小规模模型或需要显式研究滑动先验的场景，但计算成本和收敛诊断难度都更高。

旧配置可能使用 `SMC_F_J`。该拼写只作为兼容输入保留，读取时立即归一化为
`SMC_FJ`；新配置、CLI 模板、配置导出、日志和运行时状态统一使用
`SMC_FJ`。

`FULLSMC + magnitude_rake` 和 `FULLSMC + rake_fixed` 当前只支持纯
`Fault` 源，并要求每个 Fault 恰好包含规范顺序 `[ss, ds]`。输入 `ds` 会被
规范化为 `sd`；只有 `s`/`d`、或包含 tensile/coupling 的 Fault 不适用于这两种
参数化。Fault、Pressure、Sbarbot 等源混合进入 `FULLSMC` 时使用 `ss_ds`，
避免把断层专属的 magnitude/rake 语义套到其他源参数上。

## 参数分层

| 层级 | 参数 | 在 `SMC_FJ` 中 | 在 `FULLSMC` 中 |
| --- | --- | --- | --- |
| 几何 | 断层位置、走向、倾角、网格扰动参数 | SMC 采样 | SMC 采样 |
| 数据权重 | `sigmas` | SMC 采样 | SMC 采样 |
| 平滑尺度 | `alpha` | SMC 采样 | SMC 采样 |
| 滑动 | `strikeslip/dipslip` 或 `magnitude/rake` | 线性求解 | SMC 采样 |
| 数据修正 | `poly`、ramp 或 Euler rotation 等 | 线性求解 | SMC 采样或按配置处理 |

在 `SMC_FJ` 中，约束矩阵 `A` 的列只对应线性参数块，即滑动和 poly 参数；不包含 geometry、sigma 或 alpha。这个边界很重要，否则容易把几何先验和线性约束混写。

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

| 约束 | `SMC_FJ + ss_ds` | `FULLSMC` |
| --- | :---: | :---: |
| `strikeslip/dipslip` 边界 | 是 | 是 |
| `rake_angle` 线性角度约束 | 是 | 否 |
| `slip_magnitude/rake_angle` 采样边界 | 否 | 视 `slip_sampling_mode` 而定 |
| Euler 约束 | 是 | 否 |
| 零滑、边界零滑 | 是 | 否 |
| 自定义 `A @ x <= b` / `A @ x = b` | 是 | 否 |
| 震级先验 | 否 | 是 |

完整写法和模式守卫见 [ECAT 约束管理器 / Constraint Manager](constraint_manager.md)。

同一份 `bounds_config.yml` 和可选的 `interseismic_config.yml` 可以用于两种
Bayesian 模式，不需要为 `FULLSMC` 复制第二份配置。管理器会始终解析并应用
当前参数化可用的 bounds；只属于线性子问题的 rake sector、Euler、zero-slip、
source matrix 约束在 `FULLSMC` 中记为 inactive，不会悄悄转换为另一种
约束。可用以下轻量检查确认实际模式和 inactive 项：

```python
snapshot = inversion.get_constraint_snapshot()
print(snapshot["sampling_mode"])
print(snapshot["inactive_constraints"])
```

`slip_magnitude` 是物理幅值，lower bound 必须非负。若 `rake_fixed` 或
`magnitude_rake` 未给出 magnitude lower bound，`FULLSMC` 默认使用 `0`；显式
给出负下界会在初始化时被拒绝。

`walk(..., magprior=False)` 默认从 bounds 生成普通初始样本，保持常规 FULLSMC
入口简洁。`magprior=True` 只启用 magnitude-aware **初始样本**；`ss_ds` 模式下
还需显式提供 `rake_angle`、`rake_sigma` 和 `rake_range`。是否把震级项加入后验由
`magposteriors` 单独控制，不应把初始化策略当作后验约束。

## 初始化和失败语义

约束初始化是原子操作：bounds、rake、震间约束和 source constraints 全部解析、
组装并验证后才提交。任一步失败都会恢复初始化前状态，不会留下“部分约束已生效”
的对象。

| 情况 | 行为 |
| --- | --- |
| 未提供 bounds 文件，或默认可选文件不存在 | 保持无该文件的原有用法，继续初始化 |
| 文件存在但 YAML、selector、bounds 或矩阵无效 | 立即报错，不开始采样 |
| 配置了当前模式不消费的线性约束 | 保留配置并在 `inactive_constraints` 中列出 |

`SMC_FJ` 的每个样本都必须在完整的 bounds、等式和不等式约束下完成线性求解。
若求解器报错或返回非成功状态，该样本会清空临时线性解并获得低似然；ECAT 不会
删除等式约束后重试，也不会复用上一样本的 `mpost`。首次失败只在 rank 0 给出一条
紧凑告警。反复出现时应在采样前检查：

```python
snapshot = inversion.get_constraint_snapshot(validate=True)
print(snapshot["validation"])
```

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

### 激活代表模型后再统计

联合 Bayesian 采样保留的是后验样本集合。`mean`、`median` 和 `MAP` 是不同的
代表模型；统计之前必须先用 `returnModel()` 把目标代表模型写入当前
`mpost`、断层滑动和 poly 字段：

```python
inversion.returnModel(model="MAP", print_stat=False)
rows = inversion.collect_fit_statistics(
    model="MAP",
    data_poly="config",
    include_dataset=True,
    include_global=True,
)
df = inversion.fit_statistics_to_dataframe(rows)
```

`collect_fit_statistics(model="MAP")` 中的 `model` 只作为结果表标签，不会自行
从后验样本中选择 MAP。若要比较 MAP、median 和 mean，应对每一种模型分别调用
`returnModel()`，随后立即收集对应统计。完整公式和输出字段见
[Fit Statistics](fit_statistics.md)。

报告联合 Bayesian 结果时，至少说明：

- `bayesian_sampling_mode` 和 `slip_sampling_mode`。
- 使用的断层对象类型和几何扰动方法。
- 几何扰动参数、边界、单位和是否使用坐标加密。
- 网格策略、GF 方法和是否随样本重建 GF。
- 数据协方差、sigma/alpha 配置和先验范围。
- 约束类型，以及哪些约束只在 `SMC_FJ` 中生效。
- 几何后验、滑动后验和数据残差诊断。

## 相关页面

- [Bayesian 联合几何-滑动分布反演 / Joint Bayesian Geometry-Slip](../workflows/05_joint_bayesian_geometry_slip.md)
- [Bayesian 联合反演中的可扰动断层几何 / Perturbable Fault Geometry](geometry_perturbation.md)
- [ECAT 约束管理器 / Constraint Manager](constraint_manager.md)
- [Sigmas 与 Alpha 配置模式 / Sigmas and Alpha](sigmas_alpha.md)
- [Fit Statistics](fit_statistics.md)
