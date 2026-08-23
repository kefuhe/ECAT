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

配置和求解器在三个不同空间之间转换：

| 空间 | 记号 | 含义 | 例子 |
| --- | --- | --- | --- |
| 成员空间 | N | 实际数据集，或支持 Laplacian 的 source | 3 个 InSAR 数据集 |
| 参数组空间 | M | 共享同一个 sigma/alpha 的组 | 前两个共享、第三个独立，所以 M=2 |
| 更新/采样空间 | U | `update: true` 的参数组 | 仅第一组更新，所以 U=1 |

数值流转是 `U -> M -> N`：求解器先把更新/采样值放回完整组空间，再按成员到组的映射展开到数据集或平滑 source。不要用 N 推断 M，也不要用 M 推断 U。特别是多断层下的 `alpha.mode: single` 仍只有一个 alpha；固定组仍属于 M，但不占 U。

| `mode` | 含义 | 参数数量 |
| --- | --- | --- |
| `single` | 所有数据集或所有断层共享一个参数 | 1 |
| `individual` | 每个数据集或每个断层独立参数 | N |
| `grouped` | 按自定义分组共享参数 | M |

`mode` 与 `update`、`initial_value` 或 `values` 的组织必须一致：

- `single`：`update` 和值字段必须是单个值，或长度为 1 的列表。
- `individual`：布尔列表和数值列表长度必须等于数据集数量或支持平滑的 source 数量；值字段也可使用名称字典。
- `grouped`：必须提供分组；布尔列表和数值列表长度必须等于分组数量，每个数据集或支持平滑的 source 必须且只能出现在一个组里。Sigma 的值字段也可使用组名字典。

只有标量允许自动广播到全部参数组。列表、元组和数组必须与 M 严格等长；字典必须覆盖所有合法名称，而且不能包含未知键。缺值、拼错名称、重复成员、遗漏成员和不存在的成员都会在反演前报错，不会再静默补 `0.0`。

Sigma 的 `update` 应写成布尔标量或与 dataset/group 对齐的布尔列表；不要把索引列表、数据集名列表或 `true_indices` 字典当成这里的 `update` 格式。

当前解析器的默认行为是：`sigmas` 未指定 `mode` 时按 `individual`，`alpha` 未指定 `mode` 时按 `single`。底层 `simplified_vce()` 和 `rigorous_vce()` 的默认值与此一致。模板和手册仍应显式写出 `mode`，避免后来增删数据集或断层时产生歧义。

在联合 `SMC_FJ` 中，`alpha.enabled: false` 只关闭平滑块和 alpha 参数，不关闭线性参数
消元产生的 Gaussian 曲率项。无平滑时该项由数据 Hessian \(H_d\) 计算；这与 BLSE/VCE
中“关闭平滑后只解数据最小二乘”的职责不同。公式和秩条件见
[Bayesian 联合反演参考](bayesian_joint_inversion.md#为什么消去线性参数会产生-log-determinant)。

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

联合 `SMC_FJ` 模板还可在运行时传入 `smooth_prior_weight`。两者作用层级不同：`alpha`
进入当前样本的线性滑动求解，而 `smooth_prior_weight` 只在求解之后乘到平滑对数似然项，
因此不会为同一个样本改变线性解。默认 `1.0` 保持原始评分；大于 `1.0` 会让粗糙模型获得
更低的后验分数。它只用于 `alpha.enabled: true` 的 `SMC_FJ` 后验敏感性试验，不能替代
`alpha` 的初值、更新开关或先验边界。

在 BLSE/VCE 中，`enabled` 和 `update` 控制不同层次：

- `enabled: false`：不使用平滑项。
- `enabled: true` 且 `update: true`：使用平滑，并由 VCE 更新该 alpha。
- `enabled: true` 且 `update: false`：使用平滑，但将该 alpha 固定在
  `initial_value`。

`update` 可按组混合设置。例如 `[true, false, true]` 表示 VCE 只更新第一、
第三组，第二组始终使用它自己的 `initial_value`。直接调用
`run_simple_vce()` 时，省略全部 `smooth_*` 参数表示完整使用已解析的配置；若要运行期覆盖，必须一起提供 `smooth_mode`、`smooth_update`、`smooth_values`，且 `grouped` 还必须提供 `smooth_groups`。Sigma 的 `sigma_*` 参数遵循同一规则。整套替换避免把新分组与旧值或旧更新标志错配。

alpha 分组只覆盖支持 Laplacian 平滑的 fault 源。Pressure、Sbarbot 等非平滑源仍占据
完整线性参数列并参与数据拟合，但不需要、也不允许为了“覆盖所有源”而加入
`alpha.faults`。混合源结果中的 `smooth_groups` 因此只列平滑源；按完整 source 顺序保存的
内部 penalty 数组对非平滑源使用中性占位值，该值不乘入任何平滑行。

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

需要稳定组名时，也可直接使用 `groups` 字典；组名会进入解析结果和诊断输出：

```yaml
alpha:
  enabled: true
  mode: grouped
  groups:
    shallow: ["HH_Main", "HH_Deep"]
    branches: ["HH_North", "HH_South"]
    auxiliary: ["XJ_Fault"]
  update: [true, false, true]
  initial_value:
    shallow: -2.0
    branches: -1.5
    auxiliary: -1.8
  log_scaled: true
```

`faults` 列表和 `groups` 字典二选一；同时出现时 `faults` 是既有兼容入口。新配置若需要可读的稳定组名，优先使用 `groups`。

## 求解器中的统一流转

| 入口 | N 的含义 | M 的含义 | U 的含义 |
| --- | --- | --- | --- |
| BLSE | 数据集 / 可平滑 source | 固定 sigma/alpha 组 | 不采样；运行时展开为求解行权 |
| VCE | 数据集 / 可平滑 source | 方差分量组 | 参与迭代更新的组 |
| 滑动 Bayesian、SMC-FJ | 数据集 / 可平滑 source | sigma/alpha 参数组 | 进入采样向量的组 |
| 非线性几何 SMC | 数据集 | sigma 参数组 | 进入几何采样向量的 sigma 组 |

非平滑 source 仍保留在完整模型列和 `Gm` 中，但不属于 alpha 的 N、M 或 U。它不会虚构一个 alpha，也不会改变真实平滑组的参数位置。

### 配置初值与激活模型结果

联合滑动 Bayesian 明确区分配置与已激活结果。组空间初值位于
`inversion.config.sigmas` 和 `inversion.config.alpha`，采用采样尺度；通用的
`inversion.sigmas` / `inversion.alpha` 快捷属性已停用，因为同一名称无法说明读取的是
配置初值、采样切片还是物理结果。`returnModel()` 不会用 posterior 切片覆盖配置。

```python
sigma_initial = inversion.config.sigmas["initial_value"]
alpha_initial = inversion.config.alpha["initial_value"]
```

激活 `mean`、`median`、`MAP` 或显式模型向量后，完整物理结果从以下具名映射读取：

```python
inversion.returnModel(model="MAP", print_fit_statistics=False)

# 每个数据集的物理 sigma，包括固定组和采样组。
data_sigmas = inversion.current_data_sigmas

# 每个支持 Laplacian 的 source 的物理 alpha。
smoothing_alphas = inversion.current_smoothing_alphas

# 参数组层面的物理 alpha；共享组只出现一次。
alpha_groups = inversion.current_alpha_group_values
```

这些值已经按照各自的 `log_scaled` 设置完成转换。不要再对它们应用 `10**`。
`returnModel(model="std")` 得到的是 sampler/模型分量的标准差，并不构成一个物理模型，
因此不会发布上述活动模型权重。

## Log Scale

当前案例普遍使用 `log_scaled: true`。此时配置值是 `log10` 尺度：

```text
actual_sigma = 10 ** config_sigma
actual_alpha = 10 ** config_alpha
```

若明确设置 `log_scaled: false`，sigma 初值和整个采样边界必须严格大于零，因为 sigma 是
似然中的物理标准差。不要把 `sigmas: [-3, 3]` 与非对数 sigma 配合；这会允许负标准差和
零附近的无效数值。非对数模式应使用与观测单位一致的正边界。

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

### VCE 结果中的名称

VCE 最终表使用统一符号，避免把方差、标准差尺度和矩阵乘数混为一谈：

| 表中字段 | 数据组 | 平滑组 |
| --- | --- | --- |
| `Variance (v)` | `sigma²` | `alpha²` |
| `Std scale (s)` | `sigma` | `alpha` |
| `log10(s)` | 与 log-scaled sigma 配置对照 | 与 log-scaled alpha 配置对照 |
| `Row mult. (1/s)` | 数据白化行乘数 | Laplacian 行乘数 |

simple VCE 结果使用 `solved_sigma2_by_group` 与 `solved_alpha2_by_group` 保存返回模型
实际使用的尺度，并使用对应的 `proposed_*_by_group` 保存可能的下一轮更新。四个字段
始终为按组命名的字典；实际行乘数分别为
`1/sqrt(solved_sigma2_by_group[group])` 和
`1/sqrt(solved_alpha2_by_group[group])`，也就是表中的 `1/s`。旧的无明确阶段含义的
`weights`、`var_d` 和 `var_alpha` 结果字段不再返回。

## 相关页面

- [非线性几何反演配置 / Nonlinear Config](config_nonlinear_geometry.md)
- [线性滑动反演配置 / Linear Slip Config](config_linear_slip.md)
- [BLSE/VCE 参考 / BLSE/VCE](blse_vce.md)
- [Bayesian 联合反演参考 / Bayesian Joint Inversion](bayesian_joint_inversion.md)
