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

`geodata.sigmas` 控制各数据集的标准差超参数。非线性几何入口使用 `values` 作为初值；当 `log_scaled: true` 时，采样值为 `log10(sigma)`。`mode` 支持 `single`、`individual` 和 `grouped`，完整组织方式见 [Sigmas 与 Alpha 配置模式](sigmas_alpha.md)。

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
- 每个数据对象是否已经构建 covariance，例如 InSAR 的 `buildDiagCd()` 或读取 `.cov`。
- `geodata.polys` 是否符合数据类型；不支持的 transform 应直接报错，而不是回退。

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
