# 非线性几何反演配置

本文说明 ECAT 非线性几何反演配置。完整工作流见 [Bayesian 非线性几何反演](../workflows/03_nonlinear_geometry_bayesian.md)。

ECAT 目前保留两套入口：

| 入口 | 类 | 配置文件 | 生成命令 | 默认边界语义 |
| --- | --- | --- | --- | --- |
| 旧版 legacy | `explorefault` | `default_config.yml` | `ecat-generate-nonlinear` | `lower_range` |
| 新版 nonlinear geometry SMC | `NonlinearGeometrySMCInversion` | `nonlinear_geometry.yml` | `ecat-generate-nonlinear-geometry` | `lower_upper` |

新项目建议优先使用新版 nonlinear geometry SMC。旧版继续用于复现已有案例或维持旧脚本。

## 阅读路径

- 第一次运行：先看 [生成模板](#生成模板)，再复制或参考 `scripts/test_nonlinear_geometry_smc.py`。
- 设置几何搜索范围：读 [几何参数边界](#几何参数边界)。
- 设置跨直立倾角或解释 `strike/dip/rake`：先读 [断层角度约定](../concepts/fault_angle_conventions.md)。
- 设置数据顺序、断层参与关系和改正项：读 [Geodata 段](#geodata-段)。
- 设置 sigma：本页只说明字段位置，完整模式见 [Sigmas 与 Alpha 配置模式](sigmas_alpha.md)。

## 生成模板

新版模板：

```bash
ecat-generate-nonlinear-geometry -o nonlinear_geometry.yml
```

等价模块形式：

```bash
python -m eqtools.cli_tools.generate_nonlinear_geometry_config -o nonlinear_geometry.yml
```

不传 `-o` 时，新版默认输出当前目录下的 `nonlinear_geometry.yml`。对应运行脚本模板为：

```text
scripts/test_nonlinear_geometry_smc.py
```

旧版模板：

```bash
ecat-generate-nonlinear -o default_config.yml
```

旧版不传 `-o` 时默认输出 `default_config.yml`。不要把旧版 `default_config.yml` 的边界数值直接复制到新版 `nonlinear_geometry.yml`，除非先确认 `prior_bounds_format`。

## 核心字段

```yaml
nchains: 100
chain_length: 50
nfaults: 1
fault_aliasnames: [DR]
lon_lat_0: null
prior_bounds_format: lower_upper
slip_sampling_mode: mag_rake
```

| 字段 | 含义 |
| --- | --- |
| `nchains` | SMC 粒子数。 |
| `chain_length` | 每个 stage 的 mutation 链长度。 |
| `nfaults` | 紧凑几何源数量。 |
| `fault_aliasnames` | 可选断层别名，用于屏幕输出和绘图标签。 |
| `lon_lat_0` | 可选 CSI 投影原点；也可在脚本构造对象时传入。 |
| `prior_bounds_format` | 用户 YAML 中 `Uniform` 的解释方式。新版默认 `lower_upper`。 |
| `slip_sampling_mode` | 常用 `mag_rake`；也可使用 `ss_ds`。 |

## 几何参数边界

`lon`、`lat`、`depth` 表示**断层顶边中点**的经度、纬度和深度，不是断层面几何中心，也不是线性滑动反演中扩展断层面的 `top/depth`。

新版模板默认：

```yaml
prior_bounds_format: lower_upper

bounds:
  defaults:
    lon: [Uniform, 87.3, 87.6]
    lat: [Uniform, 28.6, 28.8]
    depth: [Uniform, 0.0, 10.0]
    dip: [Uniform, 10.0, 80.0]
    width: [Uniform, 1.0, 40.0]
    length: [Uniform, 1.0, 200.0]
    strike: [Uniform, 270.0, 360.0]
    slip: [Uniform, 0.0, 10.0]
    rake: [Uniform, -150.0, -30.0]
```

这里 `[Uniform, lower, upper]` 直接表示下界和上界。解析后内部会转换成底层采样需要的 lower/range 形式。

### Strike、dip 和 rake 的输入协议

- `strike` 是从北顺时针增加的地理方位角。有限值均可输入，建立 CSI 几何前会取模到
  `[0, 360)`；配置中仍建议使用常见 `[0, 360)` 表达。
- 新版紧凑几何的 `dip` 推荐使用 `[0, 180]`。只搜索一个确定下倾侧时，优先限制在
  `(0, 90]`；需要让几何连续跨过直立面时，可让先验跨过 `90°`。
- 历史带符号 `dip in [-90, 0)` 继续兼容，且与 `180 + dip` 的原生表达等价；新配置不建议
  用它跨直立采样。超出 `[-90, 180]`、非有限的固定值或先验边界会在参数注册时拒绝。
- 新版 `NonlinearGeometrySMCInversion` 与旧版 `explorefault` 现在经过同一个几何规范化函数和
  同一类早期角度边界校验；二者的差别仍是配置结构和 `lower_upper/lower_range` 表达，不是
  strike/dip 的物理约定。共享紧凑源正演可代数表示 `0°/180°`，但标准两阶段 bridge 需要
  用深度差反推 top/bottom edge，会明确拒绝这两个水平退化端点。面向完整工作流时请使用
  非退化的开区间，并避免把固定 dip 设为 `0°` 或 `180°`。
- `rake` 或 `strikeslip/dipslip` 不随几何规范化自动转换。它们属于完整样本中的滑动基底
  坐标，不是一个独立于 strike/dip 的全球方向角。

实际传给 CSI 的角度为：

```text
0 <= dip <= 90 : solver = (strike, dip)
90 < dip <= 180: solver = (strike + 180, 180 - dip)
-90 <= dip < 0 : solver = (strike + 180, -dip)
```

模型详细报告同时保留 input/sample 角度和 `CSI solver geometry`；屏幕表格只在二者不同时
增加两行 `Solver geometry`。完整原理、rake 语义和两阶段衔接见
[断层走向、倾角与滑动基底约定](../concepts/fault_angle_conventions.md)。

旧版默认使用：

```yaml
prior_bounds_format: lower_range
```

这时 `[Uniform, lower, range]` 的实际上界为 `lower + range`。例如旧版 `dip: [Uniform, 45.0, 44.9]` 表示 `45.0 <= dip <= 89.9`。

断层专属配置会覆盖 `defaults`。键可以使用 `fault_0`，也可以使用 `fault_aliasnames` 中的别名：

```yaml
fault_aliasnames: [DR]

bounds:
  defaults:
    strike: [Uniform, 270.0, 360.0]
  DR:
    rake: [Uniform, -160.0, -100.0]
```

## 固定参数

```yaml
fixed_params:
  DR:
    strike: 323.0
```

固定参数不进入 SMC 采样向量，但会在模型摘要中以 fixed 标记并放回对应断层的顺序列表，便于后续线性滑动反演复制几何。

若固定 `dip` 使用另一侧表达，例如 `110°` 或历史 `-70°`，摘要中的原始固定值保持不变，
同时会列出实际用于建 patch 的 canonical `strike/dip`。不要根据看到 `dip=70°` 就手工改变
`rake`；标准两阶段几何入口会自动使用相同的几何规范化协议。

## Geodata 段

新版普通用户优先使用 `polys` 和 `poly_bounds`：

```yaml
geodata:
  verticals: [true, true]
  faults: null
  polys: [3, 1]
  poly_bounds: [Uniform, -1000.0, 1000.0]
  sigmas:
    mode: individual
    update: true
    bounds:
      defaults: [Uniform, -3.0, 3.0]
    values: [0.0, 0.0]
    log_scaled: true
```

这些列表的顺序必须和 Python 脚本中的 `geodata = [...]` 顺序一致。

`geodata.faults` 用来说明每个数据集参与哪些断层源的预测：

- `null`：该数据集使用全部断层源。
- `["FaultA"]`：该数据集只使用 `FaultA`。
- `["FaultA", "FaultB"]`：该数据集只使用列出的断层子集。

多事件案例尤其需要显式写这个字段。例如：

```yaml
fault_aliasnames: ["RCM", "RCP"]

geodata:
  # Python geodata = [coAscsar, coDscsar, cogps7_1, cogps6_4]
  faults: [null, null, [RCM], [RCP]]
```

`geodata.polys` 是数据改正项开关。这里沿用历史字段名 `polys`，但含义是
offset/ramp 或 GPS frame transform。新版 nonlinear geometry SMC 当前只开放受控子集：

| 数据类型 | 可用设置 | 参数含义 |
| --- | --- | --- |
| SAR/InSAR, leveling | `null` | 不估计改正项 |
| SAR/InSAR, leveling | `1` | offset |
| SAR/InSAR, leveling | `3` | offset, x ramp, y ramp |
| SAR/InSAR, leveling | `4` | offset, x ramp, y ramp, xy cross term |
| GPS | `translation` | east/north/up 平移；是否包含 up 由 `verticals` 决定 |

仅包含同类且支持同一 transform 的数据集时，可以使用标量简写，例如
`polys: 3`。混合数据必须使用与 Python `geodata` 顺序一致的列表，例如
`polys: [3, null, translation]`；不要把整数 `1/3/4` 用于 GPS。

`poly_bounds` 是所有启用改正项的默认边界。正式案例中建议根据数据单位和物理量级收紧，不要长期依赖默认的 `[-1000, 1000]`。

线性 BLSE/VCE 可以直接使用 CSI GPS 的更多字符串 transform；非线性几何 SMC 目前只开放
GPS `translation`，以保证参数命名、先验和绘图输出可控。完整说明见
[数据改正项与 Frame Transform](data_corrections.md)。

## 高级 data_corrections

只有需要逐数据集或逐参数覆盖时，才写 `data_corrections`：

```yaml
geodata:
  polys: [3, 1]
  poly_bounds: [Uniform, -1000.0, 1000.0]
  data_corrections:
    enabled: true
    datasets:
      T012A:
        bounds: [Uniform, -1.0, 1.0]
        parameter_bounds:
          offset: [Uniform, -0.05, 0.05]
        display_names: ["$b_A$", "$r^x_A$", "$r^y_A$"]
```

优先级为：

```text
data_corrections.datasets.<data>.parameter_bounds.<parameter>
> data_corrections.datasets.<data>.bounds
> geodata.poly_bounds
> 内部默认 poly_bounds
```

`display_names` 只影响屏幕输出和绘图标签，不改变 canonical 参数名或采样向量顺序。它可以是参数名到显示名的字典，也可以是按 transform 参数顺序排列的列表。

## Sigma 参数

`geodata.sigmas` 控制各数据集的标准差超参数。非线性几何入口使用 `values` 作为初值；当 `log_scaled: true` 时，采样值为 `log10(sigma)`。`mode` 支持 `single`、`individual` 和 `grouped`。切换到 `grouped` 时不能只修改 `mode`，还必须增加 `groups`，并让每个实际 `data.name` 恰好属于一个组；组外遗漏、重复或未知名称都会在采样前报错。`values/update` 按参数组而非数据集数量填写，只有标量允许广播。完整组织方式见 [Sigmas 与 Alpha 配置模式](sigmas_alpha.md)。

非线性几何反演不设置 `alpha`。`alpha` 是后续分布式滑动反演中的平滑尺度，放在线性滑动或滑动 Bayesian 配置中说明。

## 脚本需要同步检查

YAML 配置不负责读取数据。用户应在脚本中显式构造 CSI 数据对象，再传给 `NonlinearGeometrySMCInversion`：

```python
from eqtools.csiExtend import NonlinearGeometrySMCInversion

geodata = [sar_t012a, sar_t121d]

inv = NonlinearGeometrySMCInversion(
    "invrc",
    lat0=lat0,
    lon0=lon0,
    config_file="nonlinear_geometry.yml",
    geodata=geodata,
)
```

每个案例至少检查：

- `lon0/lat0` 是否与数据和断层对象一致。
- `geodata` 列表顺序是否与 YAML 中所有 geodata 列表一致。
- 每个数据对象是否已经构建 covariance，例如 InSAR 的 `buildDiagCd()` 或读取 `.cov`；
  新版入口会在 `setLikelihood()` 时检查它是有限、对称正定方阵，不合法时在采样前报错。
- GPS/GNSS 的观测、模拟、frame transform 和 `Cd` 统一按 CSI 分量优先顺序排列：
  `E(all stations), N(all stations), [U(all stations)]`。用户仍以常规
  `(n_stations, 3)` 的 `vel_enu`/`synth` 数组工作，不需要自行 reshape 或重排。完整的
  `d/G/Cd/H` 行列约定见
  [观测向量、协方差与设计矩阵排列合同](../concepts/observation_matrix_layout.md)。
- `geodata.polys` 是否符合数据类型；不支持的 transform 应直接报错，而不是回退。

新版非线性几何入口在 `setLikelihood()` 时直接分解每个数据集的协方差：
\(C=L L^\mathsf{T}\)、\(W=L^{-1}\)。候选几何的残差通过
\(\lVert Wr/\sigma\rVert^2=r^\mathsf{T}C^{-1}r/\sigma^2\) 评分；它不进入
BLSE/SMC-FJ 的条件线性求解，也不显式保存 \(C^{-1}\)。单位阵、对角阵和完整非对角
协方差代表不同的观测误差模型；若为诊断而替换 `Cd`，应在结果中明确记录，而不要把
结果差异解释为求解器自动修复了原协方差。

## Data Sources

部分模板含有 `data_sources` 提示：

```yaml
data_sources:
  gps:
    directory: ../gps
    file_pattern: cogps*
  insar:
    directory: ../insar
    file_pattern: "*.rsp"
```

当前标准脚本仍推荐在 Python 中显式读取数据，再传入 `geodata`。只有实际实现自动读取时，才需要维护和解释该段。

## 相关页面

- [Bayesian 非线性几何反演](../workflows/03_nonlinear_geometry_bayesian.md)
- [CLI 命令参考](cli.md)
- [数据改正项与 Frame Transform](data_corrections.md)
- [Sigmas 与 Alpha 配置模式](sigmas_alpha.md)
