# InSAR 降采样

降采样把密集 InSAR 或 offset 产品转换为较小的大地测量数据集，并配套估计协方差。推荐流程固定为：

1. 生成降采样配置。
2. 读取原始数据并做 quick-look 检查。
3. 掩膜震源形变区域后估计协方差。
4. 降采样并写出 CSI 风格的 `.txt`、`.rsp`、`.cov` 文件。

## 对应案例与参考

| 你要确认的问题 | 推荐案例 | 相关参考 |
| --- | --- | --- |
| GAMMA/GeoTIFF 数据如何组织成降采样配置 | [InSAR 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md) | [SAR Reader 参考](../reference/sar_reader.md), [CLI 命令参考](../reference/cli.md#降采样配置) |
| `-s/-c/-d` 每一步输出什么 | [InSAR 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md#案例运行顺序) | [CLI 命令参考](../reference/cli.md#执行降采样) |
| 想按案例脚本 Step1/Step2 手动调参 | [InSAR 降采样两步走](02a_insar_downsampling_two_step.md) | [InSAR 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md#旧脚本到新-cli-的映射) |
| 降采样结果如何进入反演 | [Wushi：InSAR-only 非线性几何反演](../casebook/wushi_nonlinear_geometry.md), [Dingri 2020：BLSE/VCE 线性滑动反演](../casebook/dingri_blse_vce.md) | [InSAR 与 GPS 数据读取](01_data_reading_insar_gps.md) |

## 生成配置

GAMMA range offset：

```bash
ecat-generate-downsample \
  --mode sar \
  --sar-reader gamma \
  --sar-mode range_offset \
  --downsample-method std \
  -o downsample_range.yml \
  --copy-script
```

GAMMA GeoTIFF 解缠相位：

```bash
ecat-generate-downsample \
  --mode sar \
  --sar-reader gamma_tiff \
  --sar-mode unwrapped_phase \
  -o downsample_gamma_tiff.yml
```

光学 offset：

```bash
ecat-generate-downsample --mode optical -o downsample_optical.yml
```

## 三步运行

同一个配置文件建议分三次跑。三条命令不是三种算法，而是一次降采样处理的三个阶段：

```bash
# 1. show raw data: 只读入和画 quick-look
ecat-downsample -f downsample_range.yml -s

# 2. covariance: 估计经验协方差
ecat-downsample -f downsample_range.yml -c

# 3. downsample: 正式降采样并写出反演输入
ecat-downsample -f downsample_range.yml -d
```

等价模块形式：

```bash
python -m eqtools.cli_tools.process_data_downsampling -f downsample_range.yml -s
python -m eqtools.cli_tools.process_data_downsampling -f downsample_range.yml -c
python -m eqtools.cli_tools.process_data_downsampling -f downsample_range.yml -d
```

### 每一步做什么

| 步骤 | 参数 | 主要输入 | 程序动作 | 主要输出 | 需要检查 |
| --- | --- | --- | --- | --- | --- |
| 1. 原始数据检查 | `-s`, `--show_raw_data` | `sar_config` 或 `optical_config` 中指向的原始产品 | 读取原始 SAR/optical 产品，应用单位和符号约定，处理 zero/NaN/异常值，解析投影原点，画 quick-look；不估计协方差，不降采样 | SAR 通常有 `sar_output.txt`、`sar_values.png` 和 metadata；optical 通常有原始形变图 | 文件是否读对，形变是否在预期位置，正负号、单位、色标、经纬度范围是否合理 |
| 2. 协方差估计 | `-c`, `--do_covar` | 第 1 步确认过的原始数据，`covar.mask_out`，`covar.function/frac/every/distmax/rampEst` | 先用 `covar.mask_out` 排除主震源形变区，再抽样估计经验协方差或半变异函数；不写最终降采样 `.txt/.rsp` | SAR 输出 `Covariance_estimator.cov`；optical 输出 `Covariance_estimator_East.cov`、`Covariance_estimator_North.cov` | `mask_out` 是否避开主形变而保留背景噪声，协方差估计是否稳定 |
| 3. 正式降采样 | `-d`, `--do_downsample` | 原始数据，`downsample.method` 及对应配置，已有 `Covariance_estimator*.cov` 或 `covar.missing_policy` | 按 `std/data/trirb/from_rsp` 之一生成降采样单元，写 CSI varres 文件；再把协方差估计器投影到降采样观测上 | `<outName>_ifg.txt`、`<outName>_ifg.rsp`、`<outName>_ifg.cov`、`<outName>_decim.png`、`<outName>_run_metadata.yml` | 降采样点是否保留近场梯度，远场是否足够稀疏，`.cov` 维度是否等于观测数 |

`-s` 是 `show_raw_data`，不是保存开关；`-d` 是 `do_downsample`，不是删除。正式处理时不要直接从 `-d` 开始，除非已经确认读入、符号、投影和协方差策略都正确。

### 为什么要拆成三步

- `-s` 用来提前发现最容易出错的问题：文件选错、`reader/mode` 不匹配、相位到位移转换因子不对、LOS/offset 正负号反了、投影范围异常。
- `-c` 是给反演噪声权重用的。估计协方差前必须设置 `covar.mask_out`，否则真实震源形变会被当成噪声相关性，后续权重会偏。
- `-d` 才生成后续非线性几何反演和 BLSE/VCE 线性滑动反演需要的 CSI 输入文件。如果没有先生成 `Covariance_estimator*.cov`，程序会按 `covar.missing_policy` 读取已有协方差、写单位阵，或直接报错。

如果配置文件中已经设置 `covar.do_covar: true` 或 `downsample.enabled: true`，也可以不传 `-c/-d`。命令行参数优先级更高；`-s` 会强制只做原始数据 quick-look。

面向教学或手动调参时，可以把这三条命令理解为“预检查 + 两步走”：`-s` 做 quick-look，`-c` 是 Step 1 协方差准备，`-d` 是 Step 2 正式降采样。完整的 Step1/Step2 对照见 [InSAR 降采样两步走](02a_insar_downsampling_two_step.md)。

## 降采样方法选择

| 方法 | 建议用途 |
| --- | --- |
| `std` | 入门教程首选；不依赖断层模型。 |
| `data` | 按振幅、梯度或曲率等数据特征细化。 |
| `trirb` | 有断层模型时做三角分辨率控制降采样。 |
| `from_rsp` | 复用已有 CSI `.rsp` 格网。 |

新用户先用 `std` 跑通流程。`trirb` 应在几何和掩膜稳定后再引入。

## 核心配置段

```yaml
general:
  origin: auto

sar_config:
  outName: S1_example
  reader: gamma
  mode: range_offset
  directory: ..
  files:
    phsname: roff_20250101_20250113.phs
    rscname: roff_20250101_20250113.phs.rsc
    azifile: off_20250101_20250113.azi
    incfile: off_20250101_20250113.inc

covar:
  do_covar: false
  mask_out: [lon_min, lon_max, lat_min, lat_max]
  missing_policy: existing_or_identity
  function: exp
  frac: 0.002
  every: 2.0
  distmax: 100.0
  rampEst: true

downsample:
  enabled: false
  method: std
```

只有当案例已有固定 CSI 投影原点时，才把 `general.origin` 改成 `manual` 并填写 `lon0/lat0`。

`covar.mask_out` 是协方差估计的关键框。它应该罩住主形变源区，让协方差估计主要看到背景噪声；不要把整个有效数据区都罩掉。`covar.missing_policy` 只影响直接运行 `-d` 但当前目录没有 `Covariance_estimator*.cov` 的情况：

| 值 | 行为 |
| --- | --- |
| `existing_or_identity` | 默认值。有已有协方差就读取；没有则警告并写单位阵 `<outName>_ifg.cov` |
| `identity` | 不读取已有协方差，直接写单位阵，适合快速流程测试 |
| `error` | 找不到已有协方差时直接报错，适合正式处理时强制检查 |

## 输出文件

成功降采样后应保留：

```text
<outName>_ifg.txt
<outName>_ifg.rsp
<outName>_ifg.cov
<outName>_decim.png
<outName>_run_metadata.yml
```

这些文件前缀可以直接进入后续非线性几何反演或 BLSE/VCE 线性反演。

## 后续反演中读取

标准降采样输出使用同一个前缀读入，不要把 `.txt`、`.rsp`、`.cov` 分开传入。假设输出为：

```text
S1T056A_ifg.txt
S1T056A_ifg.rsp
S1T056A_ifg.cov
```

那么后续脚本中使用前缀 `S1T056A_ifg`。

非线性几何反演中通常读取协方差：

```python
from csi.insar import insar

asc = insar("S1T056A_ifg", lon0=lon0, lat0=lat0, verbose=False)
asc.read_from_varres("../InSAR/downsample/S1T056A_ifg", cov=True)

dsc = insar("S1T034D_ifg", lon0=lon0, lat0=lat0, verbose=False)
dsc.read_from_varres("../InSAR/downsample/S1T034D_ifg", cov=True)

geodata = [asc, dsc]
```

BLSE/VCE 线性滑动反演中也读同一个前缀。若不使用完整 `.cov`，常见案例会读入后构造对角协方差：

```python
from csi import insar

sar_t012a = insar("T012A", lon0=lon0, lat0=lat0, verbose=False)
sar_t012a.read_from_varres("../InSAR/Dingri_2020_T012A/downsampled/S1_T012A_ifg")
sar_t012a.buildDiagCd()

sar_t121d = insar("T121D", lon0=lon0, lat0=lat0, verbose=False)
sar_t121d.read_from_varres("../InSAR/Dingri_2020_T121D/downsampled/S1_T121D_ifg")
sar_t121d.buildDiagCd()

geodata = [sar_t012a, sar_t121d]
```

如果降采样阶段已经写出了可信的 `<outName>_ifg.cov`，非线性阶段优先用 `cov=True`；线性阶段是否使用完整协方差或对角阵，要在案例说明和配置中保持一致。

如果降采样结果来自三角单元，例如 `trirb` 或复用三角 `.rsp`，读入时还需要按案例设置 `triangular=True`：

```python
sar.read_from_varres("../InSAR/downsample/S1_tri_ifg", triangular=True, cov=True)
```

## 质量检查

- quick-look 图显示预期形变信号。
- 协方差掩膜排除了主要破裂形变区域。
- 降采样点保留近场梯度。
- 最终协方差矩阵维度与降采样观测数量一致。
- 每次运行的 metadata 与输出一起保存。

## 下一步

- 几何未知时，把降采样前缀读入 [Bayesian 非线性几何反演](03_nonlinear_geometry_bayesian.md)。
- 几何已固定时，把降采样前缀读入 [BLSE/VCE 线性滑动分布反演](04_linear_slip_blse_vce.md)。
