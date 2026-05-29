# InSAR 降采样

降采样把密集 InSAR 或 offset 产品转换为较小的大地测量数据集，并配套估计协方差。推荐流程固定为：

1. 生成降采样配置。
2. 读取原始数据并做 quick-look 检查。
3. 掩膜震源形变区域后估计协方差。
4. 降采样并写出 CSI 风格的 `.txt`、`.rsp`、`.cov` 文件。

## 对应案例与参考

| 你要确认的问题 | 推荐案例 | 相关参考 |
| --- | --- | --- |
| GAMMA/GeoTIFF/GMTSAR 数据如何组织成降采样配置 | [InSAR/Offset 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md) | [SAR Reader 参考](../reference/sar_reader.md), [CLI 命令参考](../reference/cli.md#降采样配置) |
| GMTSAR-style phase/LOS/range/azimuth direct-projection GRD/NetCDF 如何读入 | [InSAR/Offset 降采样案例：GMTSAR](../casebook/insar_downsampling_gamma_geotiff.md#按数据格式选择案例) | [SAR Reader 参考](../reference/sar_reader.md#gmtsar-direct-projection-grd), [CLI 命令参考](../reference/cli.md#降采样配置) |
| `-s/-c/-d` 每一步输出什么 | [InSAR/Offset 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md#三步运行顺序) | [CLI 命令参考](../reference/cli.md#执行降采样) |
| 想按案例脚本 Step1/Step2 手动调参 | [InSAR 降采样两步走](02a_insar_downsampling_two_step.md) | [InSAR/Offset 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md#旧脚本到新-cli-的映射) |
| 读入阶段需要自己写代码或处理时序 InSAR | [自定义读入 Adapter 降采样](02b_adapter_downsampling.md) | [降采样超级入口参考：input_adapter](../reference/downsampling_app.md#input_adapter) |
| 降采样入口全部字段如何查 | [降采样超级入口参考](../reference/downsampling_app.md) | [CLI 命令参考](../reference/cli.md), [SAR Reader 参考](../reference/sar_reader.md) |
| 降采样结果如何进入反演 | [Wushi：InSAR-only 非线性几何反演](../casebook/wushi_nonlinear_geometry.md), [Dingri 2020：BLSE/VCE 线性滑动反演](../casebook/dingri_blse_vce.md) | [InSAR 与 GPS 数据读取](01_data_reading_insar_gps.md) |

## 选择数据入口

先按数据格式选择 reader 和 mode。只有原始 SAR、offset 或 optical 产品需要进入本页的降采样流程；已经是
`.txt/.rsp/.cov` 的 CSI varres 前缀时，直接在反演脚本中 `read_from_varres(...)`。

| 输入数据 | 推荐 reader / 入口 | 常用 mode | 说明 |
| --- | --- | --- | --- |
| GAMMA 二进制 `.phs/.azi/.inc/.rsc` | `gamma` | `unwrapped_phase`, `los_displacement`, `range_offset`, `azimuth_offset` | 相位/offset 产品可用 `prefix` 或显式文件名 |
| GeoTIFF value + azimuth/incidence | `gamma_tiff` | `unwrapped_phase`, `los_displacement`, `range_offset`, `azimuth_offset` | 只适合可明确表达 azimuth/incidence 的产品；检查 band、角度约定、`factor_to_m` 和 wavelength |
| HyP3 GeoTIFF | `hyp3` | `unwrapped_phase`, `los_displacement` | 不要用 `gamma_tiff` 误读 HyP3 角度约定 |
| GMTSAR-style GRD/NetCDF + ENU projection | `gmtsar` | `phase_los`, `los_displacement`, `range_offset`, `azimuth_offset` | 显式写 `files.value` 和 `files.projection.east/north/up` |
| 非标准文本、外部数组、时序 InSAR | adapter | 用户自行返回 CSI 对象 | 只替换读入层，后续仍用标准 runtime |
| 已有 CSI `.txt/.rsp/.cov` | `read_from_varres(...)` | 不走 reader | 见本页“后续反演中读取” |

对应真实案例见 [InSAR/Offset 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md)。

## 生成配置

GAMMA range offset：

```bash
ecat-generate-downsample \
  --mode sar \
  --sar-reader gamma \
  --sar-mode range_offset \
  --downsample-method std \
  -o downsample_range.yml
```

GAMMA GeoTIFF 解缠相位：

```bash
ecat-generate-downsample \
  --mode sar \
  --sar-reader gamma_tiff \
  --sar-mode unwrapped_phase \
  -o downsample_gamma_tiff.yml
```

GMTSAR range offset：

```bash
ecat-generate-downsample \
  --mode sar \
  --sar-reader gmtsar \
  --sar-mode range_offset \
  -o downsample_gmtsar_range.yml
```

光学 offset：

```bash
ecat-generate-downsample --mode optical -o downsample_optical.yml
```

标准 reader 不能覆盖读入阶段时，不要改降采样大脚本；使用
[自定义读入 Adapter 降采样](02b_adapter_downsampling.md)，只替换数据进入 CSI 对象之前的部分。

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
| 1. 原始数据检查 | `-s`, `--show_raw_data` | `sar_config` 或 `optical_config` 中指向的原始产品 | 读取原始 SAR/optical 产品，应用单位和符号约定，处理 zero/NaN/异常值，应用对应 `data_filters`，解析投影原点，画 quick-look；不估计协方差，不降采样 | SAR 通常有 `sar_output.txt`、`sar_values.png` 和 metadata；optical 通常有 `optical_output.txt`、`<outName>_deformation_map.jpg`；启用过滤时还有过滤报告 | 文件是否读对，形变是否在预期位置，正负号、单位、色标、经纬度范围是否合理；若启用过滤，检查删除点数是否合理 |
| 2. 协方差估计 | `-c`, `--do_covar` | 第 1 步确认过的原始数据，顶层 `processing_region`，`covar.mask_out`，`covar.function/frac/every/distmax/rampEst` | 在过滤后的数据上按 `processing_region` 可选保留关注区域，再用 `covar.mask_out` 临时排除主震源形变区；eqtools 调用 CSI `imagecovariance` 在剩余背景点上抽样拟合经验协方差模型；不写最终降采样 `.txt/.rsp` | SAR 输出 `Covariance_estimator.cov`；若启用 `processing_region` 还有 `<effective_outName>_processing_region_report.yml`；optical 输出 `Covariance_estimator_East.cov`、`Covariance_estimator_North.cov` | `processing_region` 是否覆盖要处理的数据区，`mask_out` 是否避开主形变而保留背景噪声，协方差估计是否稳定 |
| 3. 正式降采样 | `-d`, `--do_downsample` | 原始数据，顶层 `processing_region`，`downsample.method` 及对应配置，已有 `Covariance_estimator*.cov` 或 `covar.missing_policy` | 按 `processing_region` 可选保留关注区域，再按 `std/data/trirb/from_rsp` 之一生成降采样单元，写 CSI varres 文件；若有协方差估计器，再调用 CSI `buildCovarianceMatrix()` 构建降采样观测的 `.cov` 矩阵 | `<effective_outName>_ifg.txt`、`<effective_outName>_ifg.rsp`、`<effective_outName>_ifg.cov`、`<effective_outName>_decim.png`、`<effective_outName>_run_metadata.yml`、`<effective_outName>_downsample_report.yml`；若启用 `processing_region` 还有区域报告 | 降采样点是否位于目标关注区，是否保留近场梯度，远场是否足够稀疏，`.cov` 维度是否等于观测数 |

`-s` 是 `show_raw_data`，不是保存开关；`-d` 是 `do_downsample`，不是删除。正式处理时不要直接从 `-d` 开始，除非已经确认读入、符号、投影和协方差策略都正确。

`-s` 阶段的 reader summary、`sar_output.txt` 或 `optical_output.txt` 默认报告转换后的观测/形变值统计。`plot_robust_99_range` 用来判断常规显示范围，`plot_full_range` 用来发现极端噪声尾部，`plot_clipped` 用来检查当前色标上下限会截掉多少有效点。range/azimuth/optical offset 的 full range 经常被孤立坏点拉大，不能单独作为 quick-look 色标依据。

SAR/optical 的内部处理顺序是一致的；差异在读入对象和过滤规则：

```text
extract_raw_grd()
read_observation() / read_from_tiff()
checkZeros/checkNaNs/checkLosEqualsOne()  # SAR only
apply_data_filters()
print_input_summary()
if -s: quick-look plot
if -c or -d:
  build_processing_image()
  apply_processing_region()
  if -c:
    create CSI imagecovariance from processing data
    apply covar.mask_out to exclude deformation-source area
    sample background pixels and fit exp/gauss covariance model
    write Covariance_estimator*.cov
  if -d and guide_grid is enabled for std/data:
    build filtered guide image after processing_region
    generate the grid on the guide image
    switch back to unfiltered processing data
    extract final cell values by downsample.extraction
  if -d without guide_grid:
    downsample processing data directly
```

`data_filters` 会真实删除读入后的坏点或粗差点；SAR 使用 `sar_config.data_filters`，optical 使用 `optical_config.data_filters`。`processing_region` 是正式处理范围，只影响 `-c/-d`，不影响 `-s`；`guide_grid` 在 `processing_region` 之后才生效，只控制 `std/data` 的网格生成；`covar.mask_out` 沿用 CSI 的 `maskOut()` 语义，只在协方差估计时排除震源形变区，不改变最终降采样数据。

责任边界上，eqtools 负责 reader、单位/符号转换、过滤、处理区域、命令行三步流程和输出组织；CSI 负责 `imagecovariance` 的经验协方差模型估计、`std/data/trirb/from_rsp` 降采样核心和最终 `.cov` 矩阵构建。这样用户调 YAML 时只需要看 ECAT 字段，但输出仍保持 CSI varres 文件约定。

SAR 和 optical 共享 `processing_region`、`covar`、`downsample` 和三步运行方式。两者主要差异是观测结构：SAR 进入 CSI 是单标量 `vel` 加投影向量，过滤规则使用 `value_*` 和 `projection_norm`；optical 是 `east/north` 两个水平分量，过滤规则使用 `component_*` 和 `vector_norm_range`。

### 为什么要拆成三步

- `-s` 用来提前发现最容易出错的问题：文件选错、`reader/mode` 不匹配、相位到位移转换因子不对、LOS/offset 正负号反了、投影范围异常。
- `-c` 是给反演噪声权重用的。`covar.mask_out` 不是最终数据掩膜，而是在 `processing_region` 内排除主震源形变区；剩余背景点才是 CSI 协方差估计的抽样池。若该框过小，真实形变会混入噪声模型；若过大，背景点不足，估计会不稳定。
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

如果需要按字段查完整配置含义，见 [降采样超级入口参考](../reference/downsampling_app.md)。

## 核心配置段

```yaml
general:
  origin: auto

sar_config:
  outName: S1_example
  output_suffix: auto
  reader: gamma
  mode: range_offset
  directory: ..
  files:
    prefix:
    value: roff_20250101_20250113.phs
    metadata: roff_20250101_20250113.phs.rsc
    geometry:
      azimuth: off_20250101_20250113.azi
      incidence: off_20250101_20250113.inc
    projection:
      east:
      north:
      up:

  data_filters:
    enabled: false
    report: true
    report_file: auto
    rules:
      - name: valid_observation_range
        enabled: false
        kind: value_range
        value_space: observation
        min:
        max:

processing_region:
  enabled: false
  coord_type: lonlat
  geometry: box
  box:

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
  compute:
    cutde_backend: cpp
  method: std
```

如果原始图存在明显闪烁噪声、局部 offset 或离群点，`std/data` 可能会在噪声区过度细分。
这时可以启用 `downsample.guide_grid`，即 Guided Quadtree Downsampling（引导式四叉树降采样）：
用 Gaussian 低通后的引导图做基于滤波/平滑干涉图的四叉树划分，再回到原始未滤波数据提取最终观测值。
最终 cell 值的提取方式由 `downsample.extraction` 控制，默认仍是块内均值和标准差；
需要稳健代表值时可改为 `value_statistic: median` 或 `trimmed_mean`。完整字段见
[降采样超级入口参考](../reference/downsampling_app.md)。

若使用 `method: std`，`std_config.split_metric_correction` 还可以控制分裂判据的修正方式。默认 `std`
保持 CSI 原行为；`bilinear` 会在候选块内先去掉局部平面趋势，再用残差标准差判断是否继续分裂。
它只影响网格生成，不影响最终 cell 值提取。新数据调参时建议优先尝试 `median`，当前模板默认
保留 `std` 是为了兼容既有 CSI/ECAT 结果。

GMTSAR 使用同一套三步运行流程，但 `sar_config.files` 写法不同。它直接读取标量观测和 ENU 投影系数，不再需要 `files.geometry.azimuth/incidence`：

```yaml
sar_config:
  outName: S1_T033D
  output_suffix: auto
  reader: gmtsar
  mode: range_offset
  directory: ..
  files:
    prefix:
    value: azimuth_range/T33D_range.grd
    metadata:
    geometry:
      azimuth:
      incidence:
    projection:
      east: enu_range/e_sample.grd
      north: enu_range/n_sample.grd
      up: enu_range/u_sample.grd
```

这里的 `files.projection.east/north/up` 必须是 `files.value` 原始正值方向对应的 ENU 投影系数。GMTSAR `mode: range_offset` 默认认为原始 value 和 projection 都以“朝向卫星”为正方向，进入 CSI 后也保持朝向卫星为正；不要只修改数值正负号而不同步投影方向。本示例描述的是 GMTSAR-style direct-projection GRD/NetCDF，其他来源的 `.grd` 不能只按扩展名套用，需要先确认变量、坐标和 projection 正方向。

只有当案例已有固定 CSI 投影原点时，才把 `general.origin` 改成 `manual` 并填写 `lon0/lat0`。

`data_filters` 默认关闭。若 offset 数据存在明显粗差，可启用规则，例如：

```yaml
sar_config:
  data_filters:
    enabled: true
    rules:
      - name: gross_observation_abs
        kind: value_abs
        value_space: observation
        threshold: 0.5
```

SAR 常用 `kind` 包括 `value_abs`、`value_range`、`lonlat_box`、`lonlat_polygon` 和 `projection_norm`。Optical offset 使用双分量规则，例如：

```yaml
optical_config:
  data_filters:
    enabled: true
    rules:
      - name: valid_horizontal_component_range
        kind: component_range
        components: [east, north]
        min: -1.0
        max: 1.0
      - name: valid_horizontal_norm
        kind: vector_norm_range
        min: 0.0
        max: 1.5
```

启用后会先自动执行 `finite` 隐式规则删除 NaN/inf。SAR 的 `value_space: observation` 表示按转换后的反演观测值过滤，不受 `factor4plot` 影响；optical 的阈值按读入后的 `east/north` 单位设置。

如果只想让协方差和正式降采样处理一个关注区域，不要把它混写成坏点过滤规则，优先使用 `processing_region`：

```yaml
processing_region:
  enabled: true
  coord_type: lonlat
  geometry: box
  box: [lon_min, lon_max, lat_min, lat_max]
```

`processing_region` 不影响 `-s` quick-look。原始图只想放大某个范围时，SAR 用 `sar_config.qc.plot.coordrange`，optical 用 `optical_config.qc.plot.coordrange`；降采样检查图只想显示某个范围时，用 `downsample.plot_decim.coordrange`。这些都只是显示范围，不会裁剪数据。

`processing_region` 是顶层配置，SAR/InSAR/offset 和 optical offset 共用；它不是 `sar_config` 或 `optical_config` 的 reader 参数。

`downsample.compute.cutde_backend` 默认是 `cpp`。它主要影响 `trirb` 这类需要断层 Green 函数的降采样；`std` 和 `data` 通常不触发 cutde GF。Windows 或普通工作站建议保持 `cpp`，只有确认 CUDA/OpenCL 环境可用时再显式改成 `cuda` 或 `opencl`。

`covar.mask_out` 是协方差估计的关键框。它应该罩住主形变源区，让 CSI `imagecovariance` 主要看到背景噪声；不要把整个有效数据区都罩掉。`Covariance_estimator*.cov` 保存的是拟合后的协方差模型参数和曲线信息，不是最终降采样矩阵；最终矩阵在 `-d` 阶段写成 `<effective_outName>_ifg.cov`。`covar.missing_policy` 只影响直接运行 `-d` 但当前目录没有 `Covariance_estimator*.cov` 的情况：

| 值 | 行为 |
| --- | --- |
| `existing_or_identity` | 默认值。有已有协方差就读取；没有则警告并写单位阵 `<outName>_ifg.cov` |
| `identity` | 不读取已有协方差，直接写单位阵，适合快速流程测试 |
| `error` | 找不到已有协方差时直接报错，适合正式处理时强制检查 |

## 输出文件

SAR 输出文件使用有效输出前缀，而不是简单地逐字使用 `sar_config.outName`。`output_suffix: auto`
会给 `mode: range_offset` 追加 `_RngOff`，给 `mode: azimuth_offset` 追加 `_AziOff`；如果
`outName` 已经带有同名后缀，程序不会重复追加。希望完全手动命名时可写 `output_suffix: none`。

例如 `outName: S1_T033D`、`mode: range_offset`、`output_suffix: auto` 的有效输出前缀是
`S1_T033D_RngOff`。

成功降采样后应保留：

```text
<effective_outName>_ifg.txt
<effective_outName>_ifg.rsp
<effective_outName>_ifg.cov
<effective_outName>_decim.png
<effective_outName>_run_metadata.yml
<effective_outName>_downsample_report.yml
```

SAR 检查图通常是 `<effective_outName>_decim.png`；optical 会输出 `<outName>_East_decim.png` 和 `<outName>_North_decim.png`。

若启用了顶层 `processing_region`，还会写 `<effective_outName>_processing_region_report.yml`，记录进入协方差/降采样前保留和删除的点数。
`<effective_outName>_downsample_report.yml` 记录输入点数、输出 cell 数、降采样比例、guide-grid 后端、
最终提取规则、整幅观测 `nanstd` 和可选 RMS 诊断，是调参时优先查看的文件。

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
- 若启用 `processing_region`，报告中的保留点数和空间范围合理。
- 协方差掩膜排除了主要破裂形变区域。
- 降采样点保留近场梯度。
- 最终协方差矩阵维度与降采样观测数量一致。
- 每次运行的 metadata 与输出一起保存。

## 下一步

- 几何未知时，把降采样前缀读入 [Bayesian 非线性几何反演](03_nonlinear_geometry_bayesian.md)。
- 几何已固定时，把降采样前缀读入 [BLSE/VCE 线性滑动分布反演](04_linear_slip_blse_vce.md)。
