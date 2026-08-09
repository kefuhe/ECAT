# Sigmas 与 Alpha 配置模式

`sigmas` 和 `alpha` 是反演中贯穿数据权重和平滑权重的两类超参数。建议显式写 `mode`，不要依赖省略字段后的默认行为。

## 阅读路径

- 只想确认基本物理含义：先看 [基本含义](#基本含义)。
- 遇到 `values` 和 `initial_value` 混淆：看 [字段差异](#字段差异)。
- 不确定 `single / individual / grouped` 怎么组织：看 [Mode](#mode)。
- 正在配置非线性几何：看 [Sigmas：非线性几何反演](#非线性几何反演)。
- 正在配置 BLSE/VCE 或滑动 Bayesian：看 [Sigmas：线性滑动与滑动 Bayesian](#线性滑动与滑动-bayesian) 和 [Alpha](#alpha)。

## 基本含义

| 参数 | 控制对象 | 主要出现位置 | 实际作用 |
| --- | --- | --- | --- |
| `sigmas` | 观测数据标准差或单位权重标准差 | 非线性几何、BLSE/VCE、滑动 Bayesian | 调整各数据集在似然或线性目标函数中的权重。 |
| `alpha` | 拉普拉斯平滑尺度 | BLSE/VCE、滑动 Bayesian | 调整分布式滑动模型的平滑强度。 |

两步走路线中，Bayesian 非线性反演只做几何搜索，因此只需要解释 `geodata.sigmas`。`alpha` 属于分布式滑动反演的平滑项，主要用于 BLSE/VCE 和滑动 Bayesian 配置。

## 字段差异

当前各反演入口使用的 sigma 初值字段名不同：

| 入口 | 配置类 | sigma 初值字段 | alpha 初值字段 |
| --- | --- | --- | --- |
| 非线性几何 `explorefault` | `ExploreFaultConfig` | `geodata.sigmas.values` | 几何工作流不使用 |
| BLSE/VCE 线性滑动 | `BLSEConfig` | `geodata.sigmas.initial_value` | `alpha.initial_value` |
| 滑动 Bayesian | `BayesianMultiFaultsInversionConfig` | `geodata.sigmas.initial_value` | `alpha.initial_value` |

因此，非线性几何配置不要把 `values` 改写成 `initial_value`；线性滑动和滑动 Bayesian 配置也不要把 `initial_value` 写成 `values`。

## Mode

| `mode` | 含义 | 参数数量 |
| --- | --- | --- |
| `single` | 所有数据集或所有断层共享一个参数 | 1 |
| `individual` | 每个数据集或每个断层独立参数 | N |
| `grouped` | 按自定义分组共享参数 | M |

`mode` 与 `update`、`initial_value` 或 `values` 的组织必须一致：

- `single`：`update` 和值字段必须是单个值，或长度为 1 的列表。
- `individual`：布尔列表和数值列表长度必须等于数据集数量或断层数量；值字段也可使用名称字典，未指定项默认为 `0.0`。
- `grouped`：必须提供分组；布尔列表和数值列表长度必须等于分组数量，每个数据集或断层必须且只能出现在一个组里。Sigma 的值字段也可使用组名字典。

Sigma 的 `update` 应写成布尔标量或与 dataset/group 对齐的布尔列表；不要把索引列表、数据集名列表或 `true_indices` 字典当成这里的 `update` 格式。

当前解析器的默认行为是：`sigmas` 未指定 `mode` 时按 `individual`，`alpha` 未指定 `mode` 时按 `single`。模板和手册应显式写出 `mode`，避免后来增删数据集或断层时产生歧义。

## Sigmas

### 非线性几何反演

新版非线性几何 SMC 的 `sigmas` 写在 `geodata` 下，初值字段是 `values`。若 `update: true`，还需要给出 sigma 参数的先验边界：

```yaml
prior_bounds_format: lower_upper

geodata:
  sigmas:
    mode: individual
    update: true
    bounds:
      defaults: [Uniform, -3.0, 3.0]
    values: [0.0, 0.0]
    log_scaled: true
```

边界含义由配置顶层的 `prior_bounds_format` 决定。新版模板使用 `lower_upper`，所以上例表示下界 `-3.0`、上界 `3.0`；legacy 模板使用 `lower_range` 时，同一范围应写成 `[Uniform, -3.0, 6.0]`。不要在两种模板之间直接复制第三个数。

非线性几何中按数据组共享 sigma 的完整写法是：

```yaml
prior_bounds_format: lower_upper

geodata:
  sigmas:
    mode: grouped
    groups:
      sentinel1: [S1_T134D_ifg]
      alos2: [A2_P126A_ifg, A2_P025D_ifg]
    update: true
    bounds:
      defaults: [Uniform, -3.0, 3.0]
    values:
      sentinel1: 0.0
      alos2: 0.0
    log_scaled: true
```

`groups` 中使用的是实际 `data.name`，不是 fault 名；每个输入数据集必须且只能出现一次。组名决定模型摘要中的 sigma 名称。若不同数据集的噪声尺度不应共享，继续使用 `individual`。

### 线性滑动与滑动 Bayesian

BLSE/VCE 和滑动 Bayesian 使用 `initial_value`：

```yaml
geodata:
  sigmas:
    mode: individual
    update: true
    initial_value: [0.0, 0.0]
    log_scaled: true
```

若多个数据集共用一个 sigma：

```yaml
geodata:
  sigmas:
    mode: single
    update: true
    initial_value: 0.0
    log_scaled: true
```

按数据类型分组时，`groups` 是分组名到数据集名列表的字典：

```yaml
geodata:
  sigmas:
    mode: grouped
    groups:
      InSAR_group: ["S1T056A_ifg", "S1T034D_ifg"]
      GPS_horizontal: ["GPS_E", "GPS_N"]
      GPS_vertical: ["GPS_U"]
    update: [true, false, true]
    initial_value: [0.0, -0.3, -0.1]
    log_scaled: true
```

线性滑动与滑动 Bayesian 的 `grouped` 结构相同，但值字段使用 `initial_value`。非线性几何使用前一节给出的 `values` 写法。

## Alpha

`alpha` 控制分布式滑动的拉普拉斯平滑尺度。在线性求解中，代码通常使用 `penalty_weight = 1 / alpha`，所以 `alpha` 越小，平滑惩罚权重越大。

单个平滑参数：

```yaml
alpha:
  enabled: true
  mode: single
  update: true
  initial_value: -2.0
  log_scaled: true
  faults: null
```

每个断层独立平滑参数：

```yaml
alpha:
  enabled: true
  mode: individual
  update: [true, false, true]
  initial_value:
    HH_Main: -2.0
    HH_Deep: -1.5
    XJ_Fault: -1.8
  log_scaled: true
```

按断层组共享平滑参数时，推荐使用 `faults` 的列表分组：

```yaml
alpha:
  enabled: true
  mode: grouped
  faults:
    - ["HH_Main", "HH_Deep"]
    - ["HH_North", "HH_South"]
    - ["XJ_Fault"]
  update: [true, false, true]
  initial_value: [-2.0, -1.5, -1.8]
  log_scaled: true
```

旧材料里可能能看到 `groups` 字典格式。低层解析器可把它按字典顺序转换成组列表，但案例建议统一迁移成上面的 `faults` 列表，避免与配置初始化阶段的 `faults` 预处理混淆：

```yaml
alpha:
  enabled: true
  mode: grouped
  faults:
    - ["HH_Main", "HH_Deep"]
    - ["HH_North", "HH_South"]
    - ["XJ_Fault"]
  update: [true, false, true]
  initial_value: [-2.0, -1.5, -1.8]
  log_scaled: true
```

## Log Scale

当前案例普遍使用 `log_scaled: true`。此时配置值是 `log10` 尺度：

```text
actual_sigma = 10 ** config_sigma
actual_alpha = 10 ** config_alpha
```

非线性几何反演的模型摘要会把这两个尺度分开显示：带参数索引的 `Sigma parameters` 是采样尺度，和 KDE、HDF5 样本列一致；`Physical sigma values used in likelihood` 才是似然实际使用的 `10 ** sampled_sigma`。因此看到 `0.110743` 和 `1.290455` 同时出现时，它们不是两套结果，而是同一个 sigma 的采样尺度和物理尺度。

例如：

| 配置值 | 实际值 |
| --- | --- |
| `0.0` | `1.0` |
| `-1.0` | `0.1` |
| `-2.0` | `0.01` |

因此，`alpha.initial_value: -2.0` 且 `log_scaled: true` 时，实际 `alpha = 0.01`，线性求解中的平滑惩罚权重约为 `100`。若在脚本里直接调用 `inv.run(alpha=[...])`，也应与 `penalty_log_scaled` 或配置中的 `alpha.log_scaled` 保持一致。

## 与 Bounds 的关系

非线性几何 `explorefault` 把 sigma 先验边界放在 `geodata.sigmas.bounds` 中，采用 `[Uniform, start, range]` 分布格式。

线性滑动和滑动 Bayesian 的 `bounds_config.yml` 通常包含：

```yaml
sigmas: [-3, 3]
alpha: [-3, 3]
```

这类边界用于可更新或可采样的 sigma/alpha 超参数。固定平滑的 BLSE 运行也可以在脚本中直接传入：

```python
inv.run(alpha=[-2.0])
```

或传入实际惩罚权重：

```python
inv.run(penalty_weight=[100.0])
```

二者不要同时给。`alpha` 是平滑尺度，`penalty_weight` 是求解器中直接使用的惩罚权重。

## 建议策略

入门案例建议：

- InSAR-only 或少量数据集：`sigmas.mode: individual`，每个数据集一个 sigma。
- GPS 三分量：可先把 E/N/U 拆成独立数据集；若需要减少超参数，再用 `grouped`。
- 单断层滑动反演：`alpha.mode: single` 足够清晰。
- 多断层或多段断层：若构造上需要不同平滑强度，再使用 `alpha.mode: grouped`。
- VCE 案例：明确写出哪些 sigma/alpha `update: true`，并保存每轮权重诊断。

## 相关页面

- [非线性几何反演配置 / Nonlinear Config](config_nonlinear_geometry.md)
- [线性滑动反演配置 / Linear Slip Config](config_linear_slip.md)
- [BLSE/VCE 参考 / BLSE/VCE](blse_vce.md)
- [Bayesian 联合反演参考 / Bayesian Joint Inversion](bayesian_joint_inversion.md)
