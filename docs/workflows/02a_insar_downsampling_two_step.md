# InSAR 降采样两步走

这一页讲面向教学和手动调参的 InSAR 降采样模式。它把预处理分成两个核心阶段：

1. **Step 1：协方差准备**，确认读入后估计 CSI `Covariance_estimator*.cov`。
2. **Step 2：正式降采样**，在已有协方差估计器基础上写出 `<effective_outName>_ifg.txt/.rsp/.cov`。

`ecat-downsample -s` 是随时可做的 quick-look 预检查，不算核心两步之一。第一次处理新数据、修改 `reader/mode/files`、改单位或正负号后，都建议先跑 `-s`。

## 对应案例

这个模式对应 [ECAT-Cases / InSAR_Downsampling](https://github.com/kefuhe/ECAT-Cases/tree/main/InSAR_Downsampling) 中的旧脚本组织：

| 旧案例脚本 | 新 CLI 阶段 | 作用 |
| --- | --- | --- |
| `read_tiff.py` 或脚本中的 `input_check` | `ecat-downsample -f downsample.yml -s` | 读取原始数据并画 quick-look，检查文件、单位、正负号和范围 |
| `covarSAR-Step1.py` | `ecat-downsample -f downsample.yml -c` | 排除主形变区，在背景点上估计 CSI 协方差估计器，写 `Covariance_estimator.cov` |
| `downsampleSAR-Step2.py` | `ecat-downsample -f downsample.yml -d` | 按配置降采样，写反演需要的 CSI varres 文件 |
| `downsampleSAR-Step2_NoFault.py` | `method: std`，不启用 `fault_models.use_for: [trirb]` 或 `focus_region` | 不依赖断层的基础降采样 |
| `downsampleSAR-Step2_WithFault.py` | `fault_traces` 用于叠加断层迹线；`fault_models` 用于 `trirb` 或网格叠加；`std` 近场调参用 `focus_region` | 旧脚本还手动调用了 `reject_pixels_fault(...)`，当前 CLI 的基础 `std` 路线不自动复现这一句 |

Menyuan GAMMA 示例可从 [GAMMA/2022_Menyuan](https://github.com/kefuhe/ECAT-Cases/tree/main/InSAR_Downsampling/GAMMA/2022_Menyuan) 开始看；GeoTIFF 数据读取示例见 [GeoTiff](https://github.com/kefuhe/ECAT-Cases/tree/main/InSAR_Downsampling/GeoTiff)。GMTSAR-style phase/LOS/range/azimuth direct-projection GRD/NetCDF 读入仍走同一套 `-s/-c/-d` 流程，只是 `sar_config.files` 使用 `value` 和 `projection.east/north/up`，详见 [SAR Reader 参考：GMTSAR direct-projection GRD](../reference/sar_reader.md#gmtsar-direct-projection-grd)。

## 0. 生成模板并预检查

先在当前处理目录生成模板：

```bash
ecat-generate-downsample \
  --mode sar \
  --sar-reader gamma \
  --sar-mode unwrapped_phase \
  --downsample-method std \
  -o downsample.yml
```

常规用户只改 YAML 配置。当前不再建议复制并修改整份处理脚本，因为降采样 runtime 已经包含
reader、过滤、区域裁剪、协方差、`std/data/trirb/from_rsp`、绘图和报告等多层逻辑，手工改大脚本很容易
造成后续版本难以同步。若读入阶段确实需要自定义，使用
[自定义读入 Adapter 降采样](02b_adapter_downsampling.md)，只替换数据进入 CSI 对象之前的部分。

常见输入组合见 [SAR Reader 参考](../reference/sar_reader.md)。如果需要完整字段字典，见 [降采样超级入口参考](../reference/downsampling_app.md)、[CLI 命令参考](../reference/cli.md#降采样配置) 和 [InSAR 降采样](02_insar_downsampling.md)。

修改模板时先看这些字段：

```yaml
general:
  origin: manual
  lon0: 101.31
  lat0: 37.80

sar_config:
  outName: S1_T128A
  reader: gamma
  mode: unwrapped_phase
  directory: ..
  files:
    prefix: geo_20220105_20220117
  read:
    downsample: 1
    downsample_for_covar: 1
    zero2nan: true
```

单幅数据初测可以保留 `general.origin: auto`；多轨道数据需要保持同一个 CSI 局部坐标系，或要和已有案例脚本完全对齐时，再改成 `manual` 并填写 `lon0/lat0`。

如果是 `range_offset` 或 `azimuth_offset`，`sar_config.output_suffix: auto` 会把有效输出前缀解析为
`<outName>_RngOff` 或 `<outName>_AziOff`；若 `outName` 已经带同名后缀，不会重复追加。后续文中的
`<effective_outName>` 指这个解析后的前缀。

然后做 quick-look：

```bash
ecat-downsample -f downsample.yml -s
```

检查重点是原始形变是否读到预期位置，色标和单位是否合理，LOS/offset 正负号是否和科学约定一致，`lon0/lat0` 或自动投影原点是否稳定。

如果只想放大查看原始图，在 `sar_config.qc.plot.coordrange` 中设置 `[minlon, maxlon, minlat, maxlat]`。如果正式 Step 1/Step 2 只处理一个关注区域，用顶层 `processing_region`，它会在协方差和降采样前保留该区域内的点，但不影响 `-s`：

```yaml
processing_region:
  enabled: true
  coord_type: lonlat
  geometry: box
  box: [lon_min, lon_max, lat_min, lat_max]
```

启用 `processing_region` 后，Step 1 和 Step 2 应使用同一个配置重新运行；生成的 `<effective_outName>_processing_region_report.yml` 用来检查保留点数是否合理。

## 1. Step 1 协方差准备

协方差估计只关心背景噪声结构。核心是用 `covar.mask_out` 在当前 `processing_region` 内排除主震源形变区，让 CSI `imagecovariance` 只从剩余背景点抽样拟合协方差模型：

```yaml
covar:
  do_covar: false
  mask_out: [100.5, 101.75, 37.35, 38.1]
  function: exp
  frac: 0.002
  every: 2.0
  distmax: 100.0
  rampEst: true
```

运行：

```bash
ecat-downsample -f downsample.yml -c
```

这一步对应旧脚本中的：

```python
covar.maskOut([maskOut])
covar.computeCovariance(function="exp", frac=0.002, every=2.0, distmax=100.0, rampEst=True)
covar.write2file(savedir="./")
```

输出通常是：

```text
Covariance_estimator.cov
```

光学 offset 会按分量写成 `Covariance_estimator_East.cov`、`Covariance_estimator_North.cov`。这些文件保存的是 CSI 协方差估计器，不是最终降采样矩阵；最终矩阵会在 Step 2 按降采样后的观测点写成 `<effective_outName>_ifg.cov`。

如果只是在快速测试流程，可以让 Step 2 使用单位阵；正式反演前应重新检查 `processing_region`、`mask_out` 和协方差曲线。`mask_out` 不会从 Step 2 的降采样结果中删除点，也不能替代 `data_filters` 的粗差剔除。

## 2. Step 2 正式降采样

基础两步走建议先用 `method: std`，因为它不需要预设断层模型，参数含义也最直观：

```yaml
downsample:
  enabled: false
  compute:
    cutde_backend: cpp
  method: std
  std_config:
    startingsize: 5.0
    minimumsize: 0.25
    min_valid_fraction: 0.1
    split_std_threshold: 0.005
    split_metric_smoothing:
    focus_region:
      enabled: false
      polygon_file:
      max_splits_outside: 5
    high_value_refinement:
      enabled: false
    low_amplitude_cap:
      enabled: false
    decimorig: 10
    itmax: 100
```

运行：

```bash
ecat-downsample -f downsample.yml -d
```

这一步对应旧脚本中的：

```python
downsampler.initialstate(10, 0.5, tolerance=0.25, plot=False, decimorig=10)
downsampler.stdBased(0.03, plot=False, verboseLevel="minimum", decimorig=10, smooth=2, itmax=100)
downsampler.writeDownsampled2File(prefix=outName + "_ifg", rsp=True)
sardecim.Cd = covar.buildCovarianceMatrix(sardecim, "Covariance estimator", write2file=outName + "_ifg.cov")
```

新模板里这些旧参数的主要对应关系是：

| 旧脚本参数 | YAML 字段 | 说明 |
| --- | --- | --- |
| `initialstate(10, 0.5, tolerance=0.25)` | `startingsize`, `minimumsize`, `min_valid_fraction` | 初始块大小、最小块大小、有效像素比例阈值 |
| `stdBased(0.03, ...)` | `std_config.split_std_threshold` | 块内标准差阈值，越小采样越密 |
| `smooth=2` | `std_config.split_metric_smoothing` | 对块内标准差判据做平滑 |
| `itmax=100` | `std_config.itmax` | 最大分裂迭代次数 |
| `writeDownsampled2File(prefix=outName + "_ifg", rsp=True)` | `sar_config.outName` + `sar_config.output_suffix` | 输出 `<effective_outName>_ifg.txt/.rsp/.cov` |

## 手动调参顺序

推荐保留 Step 1 的协方差结果，然后反复改 Step 2：

```bash
ecat-downsample -f downsample.yml -s
ecat-downsample -f downsample.yml -c
ecat-downsample -f downsample.yml -d
```

只有在 raw 数据、`reader/mode/files`、单位/正负号、投影原点、`data_filters`、`processing_region`、`mask_out` 和 `covar.function/frac/every/distmax/rampEst` 都不变时，Step 1 结果才适合复用。改了这些输入中的任意一项，应重新运行 `-c`。

如果降采样太密，优先增大 `std_config.split_std_threshold`、增大 `minimumsize`，或启用 `low_amplitude_cap`。如果近场梯度被采得太粗，优先减小 `split_std_threshold`、减小 `minimumsize`，或启用 `focus_region` / `high_value_refinement`。

每次试验建议改 `sar_config.outName`，例如 `S1_T128A_std_v1`、`S1_T128A_std_v2`，避免覆盖上一组 `.txt/.rsp/.cov`。

## 有无断层辅助的区别

基础 `std` 降采样不需要断层，适合先跑通。案例中的 `NoFault` 版本就是这种思路。

如果已有断层迹线，只想在 quick-look 和降采样检查图中叠加，放到 `fault_traces`：

```yaml
fault_traces:
  - enabled: true
    id: surface_trace
    file: Fault_Trace_Menyuan.txt
    stages: [raw, decim]
```

如果要使用 `trirb` 这类需要断层几何的高级降采样，需要在 `fault_models` 中启用三角断层模型。模型可以从迹线生成，也可以读取已有 CSI GMT 三角网格：

```yaml
fault_models:
  - enabled: true
    id: generated_triangular_fault
    type: generated_from_trace
    geometry: triangular
    trace_file: Fault_Trace_Menyuan.txt
    dip_angle: 82
    dip_direction: 194
    top_size: 2.0
    bottom_size: 3.0
    top_depth: 0.0
    bottom_depth: 21.0
    use_for: [trirb]
    plot:
      stages: [decim]
      mode: edges
```

如果使用基础 `std` 方法，并且希望主要破裂附近保留更细采样，应使用面状 `focus_region`。这里需要的是一个包围关注区的 polygon，不是单条断层迹线：

```yaml
downsample:
  method: std
  std_config:
    focus_region:
      enabled: true
      coord_type: lonlat
      polygon_file: focus_region_polygon.txt
      max_splits_outside: 5
```

`focus_region` 是控制细分层级的区域，不是删除数据的掩膜。它和 `covar.mask_out` 含义不同：`mask_out` 只用于协方差估计时排除主形变区；`focus_region` 只用于正式降采样时控制哪些区域允许更细。

如果目标是“只在某个关注区域内估计协方差并降采样”，使用顶层 `processing_region`；如果目标是“数据都保留，但关注区域内采得更细”，使用 `downsample.std_config.focus_region`。

如果已有 CSI GMT 三角网格，可用 `type: csi_gmt`：

```yaml
fault_models:
  - enabled: true
    id: existing_mesh
    type: csi_gmt
    geometry: triangular
    file: fault_mesh.gmt
    use_for: [trirb]
    plot:
      stages: [raw, decim]
      mode: edges
```

如果改用 `method: trirb`，程序会根据 `fault_models` 中 `use_for: [trirb]` 的三角断层模型构建 cutde Green 函数来做分辨率控制。默认 `downsample.compute.cutde_backend: cpp` 是跨平台稳定设置；只有确认 CUDA/OpenCL 环境可用时，再显式改成 `cuda` 或 `opencl`。运行后的 `<effective_outName>_run_metadata.yml` 会记录 requested 和 active backend。

旧 `WithFault` 脚本中的 `reject_pixels_fault(1, jcrect)` 是额外的手工处理，用来剔除断层附近一定距离内的像素。当前 YAML 的基础 `std` 路线不自动做这个动作，也不再建议通过复制整份处理脚本来复现。通常应先判断真实需求：如果是删除粗差或排除空间区域，用 `data_filters` 或顶层 `processing_region`；如果是让近断层区采样更细，用 `focus_region` 或 `trirb`。只有读入阶段本身不标准时，才使用 adapter 模板。

## 输出如何进入反演

Step 2 结束后保留同一前缀的三个文件：

```text
S1_T128A_ifg.txt
S1_T128A_ifg.rsp
S1_T128A_ifg.cov
```

非线性几何反演和 BLSE/VCE 线性滑动反演都用前缀读入：

```python
from csi import insar

sar = insar("S1_T128A", lon0=lon0, lat0=lat0, verbose=False)
sar.read_from_varres("../InSAR/downsample/S1_T128A_ifg", cov=True)
```

如果线性反演脚本选择重新构造对角协方差，也可以读入后调用：

```python
sar.buildDiagCd()
```

具体反演阶段见 [Bayesian 非线性几何反演](03_nonlinear_geometry_bayesian.md) 和 [BLSE/VCE 线性滑动分布反演](04_linear_slip_blse_vce.md)。

## 最小检查清单

- `-s` 的 raw quick-look 图位置、单位、正负号正确。
- 如果启用 `processing_region`，Step 1 和 Step 2 使用同一个区域配置，报告中的保留点数合理。
- `covar.mask_out` 只排除主形变源区，没有把背景噪声区域全部罩掉。
- `Covariance_estimator*.cov` 已生成，并与当前数据、处理区域、`mask_out` 和投影原点匹配。
- `-d` 生成的点在近场足够密、远场不过密。
- `<effective_outName>_ifg.cov` 维度与 `<effective_outName>_ifg.txt` 观测数一致。
- `<effective_outName>_run_metadata.yml` 中记录的 `steps` 和 `effective_config` 与本次试验一致。
