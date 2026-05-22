# 非线性几何反演配置

本文说明 `explorefault` 系列案例使用的非线性几何配置。

## 生成模板

在一个新的非线性反演目录中，可以先生成默认配置模板：

```bash
ecat-generate-nonlinear -o default_config.yml
```

等价模块形式：

```bash
python -m eqtools.cli_tools.generate_nonlinear_config -o default_config.yml
```

如果不传 `-o`，默认输出文件也是当前目录下的 `default_config.yml`。模板只提供字段结构和默认示例值，正式案例需要参照已有案例设置几何参数范围、固定参数、数据顺序和 sigma 策略。

## 核心字段

```yaml
nchains: 100
chain_length: 50
nfaults: 1
fault_aliasnames: [F1]
slip_sampling_mode: mag_rake
```

| 字段 | 含义 |
| --- | --- |
| `nchains` | SMC 链数量。 |
| `chain_length` | 每条链的 MCMC 步数。 |
| `nfaults` | 紧凑几何源数量。 |
| `fault_aliasnames` | 可选断层别名。 |
| `slip_sampling_mode` | 几何搜索通常使用 `mag_rake`。 |

## 几何参数边界

`lon`、`lat`、`depth` 表示断层顶边中点的经度、纬度和深度，不是断层面几何中心，也不是线性滑动反演中扩展断层面的 `top/depth`。

```yaml
bounds:
  defaults:
    lon: [Uniform, 78.56, 2.0]
    lat: [Uniform, 41.19, 2.0]
    depth: [Uniform, 5.0, 20.0]
    dip: [Uniform, 45.0, 44.9]
    width: [Uniform, 0.1, 29.9]
    length: [Uniform, 5.0, 45.0]
    strike: [Uniform, 180.0, 180.0]
    magnitude: [Uniform, 0.0, 10.0]
    rake: [Uniform, -90.0, 180.0]
```

`Uniform` 当前沿用 `scipy.stats.uniform` 的 loc/scale 输入方案，不是直接的 `[下界, 上界]`：

```text
[Uniform, start, range]
upper = start + range
```

其中第二个数是起点或 loc，第三个数是范围或 scale。例如：

| 配置 | 实际约束 |
| --- | --- |
| `lon: [Uniform, 78.56, 2.0]` | `78.56 <= lon <= 80.56` |
| `dip: [Uniform, 45.0, 44.9]` | `45.0 <= dip <= 89.9` |
| `rake: [Uniform, -90.0, 180.0]` | `-90.0 <= rake <= 90.0` |

这里的 `start` 是分布起点，不是 `initialSample` 初始样本。未来配置格式可能改成显式下界和上界；当前文档先维持现有格式。

断层专属配置覆盖 `defaults`：

```yaml
bounds:
  defaults:
    strike: [Uniform, 180.0, 180.0]
  RCP:
    strike: [Uniform, 0.0, 270.0]
    rake: [Uniform, -30.0, 60.0]
```

## 固定参数

```yaml
fixed_params:
  RCP:
    lon: -117.541
    lat: 35.6431
    depth: 0.0
    strike: 227.0
```

当某些几何参数来自野外制图或已发表模型时，可以固定。

## Geodata 段

```yaml
geodata:
  verticals: [true, true]
  faults: null
  polys:
    enabled: true
    boundaries:
      defaults: [Uniform, -200.0, 400.0]
  sigmas:
    mode: individual
    update: true
    bounds:
      defaults: [Uniform, -3.0, 6.0]
      sigma_0: [Uniform, -3.0, 6.0]
    values: [0.0, 0.0]
    log_scaled: true
```

`verticals`、`faults`、`sigmas.values` 的顺序必须与 Python 脚本中的 `geodata` 顺序一致。

`geodata.faults` 用来说明每个数据集参与哪些断层源的预测：

- `null`：该数据集使用全部断层源。
- `["FaultA"]`：该数据集只使用 `FaultA`。
- `["FaultA", "FaultB"]`：该数据集只使用列出的断层子集。

多事件案例尤其需要显式写这个字段。例如 Ridgecrest 中，InSAR 覆盖 Mw6.4 前震和 Mw7.1 主震的累计形变，而两组 GPS 分别只覆盖其中一个事件：

```yaml
fault_aliasnames: ["RCM", "RCP"]

geodata:
  # Python geodata = [coAscsar, coDscsar, cogps7_1, cogps6_4]
  faults: [null, null, [RCM], [RCP]]
```

这表示两条 InSAR 轨道使用全部断层源，`cogps7_1` 只约束 `RCM`，`cogps6_4` 只约束 `RCP`。该关系必须和脚本中的 `geodata` 列表顺序一一对应。

非线性几何入口 `explorefault` 使用 `geodata.sigmas.values` 作为 sigma 初值字段；BLSE/VCE 和滑动 Bayesian 配置则使用 `geodata.sigmas.initial_value`。不要在两类配置之间直接替换字段名。`sigmas.mode` 支持 `single`、`individual` 和 `grouped`，完整组织方式见 [Sigmas 与 Alpha 配置模式](sigmas_alpha.md)。

非线性几何反演不设置 `alpha`。`alpha` 是分布式滑动反演中的拉普拉斯平滑尺度，放在线性滑动或滑动 Bayesian 配置中说明。

## Data Sources

部分配置含有数据源提示：

```yaml
data_sources:
  gps:
    directory: ../gps
    file_pattern: cogps*
  insar:
    directory: ../insar/downsample
    file_pattern: "*.rsp"
```

当前案例多在 Python 脚本中显式读取数据，再传入 `geodata`。只有实际使用自动读取时，才保留并讲解该段。
