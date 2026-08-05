# Bayesian 联合几何-滑动分布反演 / Joint Bayesian Geometry-Slip Inversion

本页说明 ECAT 中的高级 Bayesian 联合反演路线。它不是入门两步走的替代品，而是在几何不确定性会显著影响滑动分布、数据覆盖足够、计算预算允许时使用。

入门路线通常是：

```text
Bayesian 非线性几何反演 -> BLSE/VCE 线性滑动分布反演
```

联合路线则把可扰动断层几何、滑动分布、数据权重和多源约束放进同一个 Bayesian 采样框架：

```text
数据对象 + 可扰动断层对象 + Bayesian 配置
  -> BayesianMultiFaultsInversion
  -> SMC_FJ 或 FULLSMC
  -> 几何后验、滑动后验、sigma/alpha 后验和拟合诊断
```

## 适用场景

优先使用两步走路线，除非至少满足以下条件之一：

- 几何扰动会明显改变滑动分布，固定单一优选几何会低估不确定性。
- 需要比较多源、多断层或多事件场景下的几何-滑动耦合。
- 希望在同一后验中报告几何、滑动、sigma/alpha 和约束影响。
- 已经有稳定的两步走结果，可作为联合采样的初值、边界和质量检查基准。

不建议把联合 Bayesian 当作第一个案例。它对几何参数化、网格更新、GF 重建、约束模式和计算成本都更敏感。

## 模式选择

| 模式 | 采样内容 | 滑动处理 | 适合场景 |
| --- | --- | --- | --- |
| `SMC_FJ` | 几何、sigma、alpha 等超参数 | 每个样本内用约束线性求解滑动 | 推荐的高级主线；维度较低，可复用 BLSE 约束体系 |
| `FULLSMC` | 几何、sigma、alpha 和滑动参数 | 滑动也作为采样参数 | 研究滑动先验或小规模模型；维度高，计算成本大 |

`SMC_FJ` 是规范配置值，读入后也作为唯一内部名称保存。旧配置中的
`SMC_F_J` 仍会在配置入口自动归一化为 `SMC_FJ`，但新配置、导出结果和
运行状态不再使用旧拼写。`SMC_FJ` 会使用 `ss_ds` 滑动参数化；线性 rake、
Euler、零滑和自定义矩阵约束只在 `SMC_FJ + ss_ds` 或 BLSE 中形成线性约束
矩阵。完整约束差异见
[ECAT 约束管理器 / Constraint Manager](../reference/constraint_manager.md)。

两种模式复用同一份 bounds 和震间配置。切换到 `FULLSMC` 时，不必维护第二份
YAML；线性子问题专属约束会被标为 inactive，bounds 和适用先验仍按当前参数化
生效。纯断层小模型可使用 `magnitude_rake` 或 `rake_fixed`；包含 Pressure、
Sbarbot 等非断层源时，`FULLSMC` 使用 `ss_ds`。

## 数据和断层对象

联合反演仍然使用 CSI/ECAT 的标准数据对象。InSAR、GPS、光学或其他数据进入反演前，应先完成读取、单位、正负号、协方差和降采样检查。相关流程见：

- [InSAR 与 GPS 数据读取 / Data Reading](01_data_reading_insar_gps.md)
- [InSAR 降采样 / InSAR Downsampling](02_insar_downsampling.md)
- [降采样超级入口参考 / Downsampling App](../reference/downsampling_app.md)

可扰动断层几何通常使用 `BayesianAdaptiveTriangularPatches`。它不是普通两步走中的固定三角断层网格，而是带有冻结基线、扰动方法、网格更新标志和自动发现机制的断层对象。几何扰动细节见 [Bayesian 联合反演中的可扰动断层几何 / Perturbable Fault Geometry](../reference/geometry_perturbation.md)。

## 配置骨架

联合 Bayesian 配置仍由主配置和边界配置控制。下面只展示和联合路线相关的关键字段，正式案例需要继续设置数据、断层、GF、sigma/alpha 和约束。

```yaml
bayesian_sampling_mode: SMC_FJ    # SMC_FJ | FULLSMC
slip_sampling_mode: ss_ds         # SMC_FJ 会使用 ss_ds；FULLSMC 可按问题选择
nchains: 100
chain_length: 50

faults:
  FaultA:
    method_parameters:
      update_fault_geometry:
        method: perturb_BottomFixedDir_RotateTransGeom_simpleMesh
        disct_z: 10
        bias: 1.0
        angle_unit: degrees
```

`update_fault_geometry` 中的 `method` 必须是断层对象可发现的扰动方法。`perturbations` 由采样器从当前样本自动传入，不要写进 YAML。可用方法可通过 CLI 或断层对象帮助系统查看：

```bash
ecat-list-fault-perturb-methods
```

或在交互环境中：

```python
fault.help()
```

## 执行逻辑

每个 SMC 样本大致执行以下步骤：

1. 从样本向量取出当前几何、sigma、alpha 等超参数。
2. 对启用 `update_fault_geometry` 的断层调用指定扰动方法。
3. 根据扰动结果更新网格、patch 面积、Laplacian 和必要的 GF。
4. 在 `SMC_FJ` 中，用当前超参数和约束求解线性滑动；在 `FULLSMC` 中，滑动来自样本本身。
5. 计算似然、权重和后验诊断。

因此，联合路线的关键不是单独“画一个扰动几何”，而是保证每个样本中的几何、网格、GF、约束矩阵和数据协方差是一致的。

开始采样前建议做一次轻量检查：

```python
snapshot = inversion.get_constraint_snapshot(validate=True)
print(snapshot["sampling_mode"])
print(snapshot["inactive_constraints"])
print(snapshot["validation"])
```

初始化阶段的无效 active constraint 会直接报错，并回滚整次约束更新。`SMC_FJ`
运行中若某个样本的受约束线性求解失败，该样本只获得低似然；程序不会移除等式
约束后重算，因此不会把约束外解混入后验。

## 与两步走的关系

两步走仍然是推荐公开入门路线：

- 非线性几何反演负责寻找合理的几何参数范围和优选模型。
- BLSE/VCE 负责在固定几何上做分布式滑动和权重诊断。
- 联合 Bayesian 用于进一步传播几何不确定性到滑动分布，而不是替代前两步的基础检查。

实际研究中，建议先用两步走获得稳定结果，再用这些结果设置联合 Bayesian 的几何先验、扰动尺度、网格尺度和计算预算。

## 输出和检查

后验采样完成后，先激活要报告的代表模型，再计算拟合统计：

```python
inversion.returnModel(model="median", print_stat=False)
rows = inversion.collect_fit_statistics(model="median")
df = inversion.fit_statistics_to_dataframe(rows)
```

这里 `returnModel()` 负责选择并分发 median；`collect_fit_statistics()` 中的
`model="median"` 只是统计表标签。比较 MAP、mean 或 median 时，每个模型都要
分别执行这两步。详细说明见 [Fit Statistics](../reference/fit_statistics.md)。

联合反演结果至少应检查：

- 几何参数后验是否收敛，是否贴边。
- 滑动后验均值、中位数和可信区间是否受几何扰动主导。
- 每条数据的残差、sigma 后验和权重是否合理。
- `SMC_FJ` 中线性约束是否按预期生效。
- `inactive_constraints` 是否只包含当前模式预期不消费的配置项。
- 是否出现重复的 constrained linear solve 失败告警；若有，应先检查约束可行性。
- 网格更新和 GF 重建是否与扰动方法一致。

报告联合 Bayesian 结果时，应明确写出 `bayesian_sampling_mode`、`slip_sampling_mode`、几何扰动方法、扰动参数边界、网格策略、GF 后端、数据协方差处理和约束类型。

## 相关页面

- [Bayesian 联合反演参考 / Bayesian Joint Inversion](../reference/bayesian_joint_inversion.md)
- [Bayesian 联合反演中的可扰动断层几何 / Perturbable Fault Geometry](../reference/geometry_perturbation.md)
- [ECAT 约束管理器 / Constraint Manager](../reference/constraint_manager.md)
- [Sigmas 与 Alpha 配置模式 / Sigmas and Alpha](../reference/sigmas_alpha.md)
- [BLSE/VCE 参考 / BLSE/VCE](../reference/blse_vce.md)
