# InSAR/Offset 降采样案例

本文把 [ECAT-Cases / InSAR_Downsampling](https://github.com/kefuhe/ECAT-Cases/tree/main/InSAR_Downsampling)
中的原始 SAR、offset 和 GeoTIFF 数据连接到 ECAT 降采样工作流。文件名保留
`insar_downsampling_gamma_geotiff.md` 是为了兼容已有链接；本页范围已经扩展到
GAMMA、GeoTIFF、GMTSAR direct-projection 和 adapter 模板。

## GitHub 位置

- [ECAT-Cases / InSAR_Downsampling](https://github.com/kefuhe/ECAT-Cases/tree/main/InSAR_Downsampling)
- [GAMMA / 2022_Menyuan](https://github.com/kefuhe/ECAT-Cases/tree/main/InSAR_Downsampling/GAMMA/2022_Menyuan)
- [GeoTiff](https://github.com/kefuhe/ECAT-Cases/tree/main/InSAR_Downsampling/GeoTiff)
- [GMTSAR](https://github.com/kefuhe/ECAT-Cases/tree/main/InSAR_Downsampling/GMTSAR)
- [GMTSAR / Myanmar](https://github.com/kefuhe/ECAT-Cases/tree/main/InSAR_Downsampling/GMTSAR/Myanmar)

## 本页范围

这些案例用于讲解：

- GAMMA 二进制 `.phs/.azi/.inc/.rsc` 产品读取。
- GeoTIFF value + azimuth/incidence 几何栅格读取。
- GMTSAR-style GRD/NetCDF 标量观测 + ENU projection 栅格读取。
- phase/LOS、range offset 和 azimuth offset 的 reader/mode 选择。
- `-s/-c/-d` 三步运行、协方差估计、`std` 降采样、`processing_region` 和 `guide_grid`。
- 输出 CSI `.txt/.rsp/.cov` 并进入后续非线性几何反演或 BLSE/VCE 线性反演。

<a id="choose-by-format"></a>

## 按数据格式选择案例

| 手里数据 | 推荐案例目录 | reader / mode | 重点检查 |
| --- | --- | --- | --- |
| GAMMA 二进制解缠相位 | `GAMMA/2022_Menyuan/T128A/std_downsample_superapp` | `reader: gamma`, `mode: unwrapped_phase` | `files.prefix`、`.rsc` 中波长、`factor_to_m: 1.0` |
| GAMMA 旧两步脚本 | `GAMMA/2022_Menyuan/T033D/stdBased` 和 `resolutionBased` | 旧脚本对照 `ecat-downsample -c/-d` | `covarSAR-Step1.py`、`downsampleSAR-Step2.py` 与 YAML 字段映射 |
| GeoTIFF 解缠相位 | `GeoTiff/Wushi/insar/std_superapp` | `reader: gamma_tiff`, `mode: unwrapped_phase` | `files.value`、`geometry.azimuth/incidence`、band 和 wavelength |
| GeoTIFF 位移或外部几何产品 | `GeoTiff/Chile` | 优先确认产品语义；标准 reader 不匹配时用 adapter | `gamma_tiff` 只适合 value + azimuth/incidence 可明确表达的产品 |
| GMTSAR LOS disp. | `GMTSAR/Myanmar/.../Phase_Los/T33D` | `reader: gmtsar`, `mode: los_displacement` | `files.value` 与 `projection.east/north/up` 是否成套对应 |
| GMTSAR range offset | `GMTSAR/Myanmar/.../Pixel Offset Tracking/T33D` | `reader: gmtsar`, `mode: range_offset` | value 和 projection 默认同为朝向卫星正方向；不要只翻转 value |
| GMTSAR azimuth offset | `GMTSAR/Myanmar/.../Pixel Offset Tracking/T33D` | `reader: gmtsar`, `mode: azimuth_offset` | 使用沿 heading 的 east/north projection，`projection.up` 可为空 |
| 自定义读入或时序复用网格 | `GAMMA/2022_Menyuan/T128A/std_adapter_downsampling_workflow` | `input_adapter.enabled: true` | 只改 `input_adapter.py` 和 YAML，后续复用标准 runtime |

如果数据已经是降采样后的 CSI varres 前缀 `.txt/.rsp/.cov`，不要再走原始 SAR reader，直接在反演脚本中使用
`read_from_varres(...)`。如果只是外部整理好的普通 ASCII 点位，使用 CSI 的
`read_from_ascii(...)`。这些入口见 [InSAR 与 GPS 数据读取](../workflows/01_data_reading_insar_gps.md)。

## 新建降采样目录

新建案例目录时，优先用 ECAT CLI 在当前目录生成 `downsample.yml` 模板，然后按数据产品修改
`sar_config.files`、`general.origin`、`covar.mask_out`、`processing_region`、`downsample.method`
和输出前缀。

GAMMA 二进制相位：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase --downsample-method std -o downsample.yml
```

GeoTIFF 相位：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma_tiff --sar-mode unwrapped_phase --downsample-method std -o downsample_los.yml
```

GMTSAR range offset：

```bash
ecat-generate-downsample --mode sar --sar-reader gmtsar --sar-mode range_offset --downsample-method std -o downsample_rng.yml
```

GMTSAR azimuth offset：

```bash
ecat-generate-downsample --mode sar --sar-reader gmtsar --sar-mode azimuth_offset --downsample-method std -o downsample_az.yml
```

命令细节见 [CLI 命令参考](../reference/cli.md)，reader 和 mode 的含义见
[SAR Reader 参考](../reference/sar_reader.md)，完整字段字典见
[降采样超级入口参考](../reference/downsampling_app.md)。

<a id="three-step-run"></a>

## 三步运行顺序

同一个配置文件建议分三次运行。三条命令不是三种算法，而是一次降采样处理的三个阶段：

```bash
# 1. -s: show raw data
# 读入原始产品，检查文件、单位、正负号、projection 和 quick-look 图。
ecat-downsample -f downsample.yml -s

# 2. -c: do covariance
# 根据 covar.mask_out 排除主形变区，估计 CSI Covariance_estimator*.cov。
ecat-downsample -f downsample.yml -c

# 3. -d: do downsample
# 按 downsample.method 正式降采样，写出 CSI .txt/.rsp/.cov。
ecat-downsample -f downsample.yml -d
```

第一次处理新数据时不要直接从 `-d` 开始。先用 `-s` 检查 raw quick-look、summary 中的
robust/full range、projection 均值和正负号；再用 `-c` 检查协方差估计；最后看
`<effective_outName>_decim.png`、`<effective_outName>_downsample_report.yml` 和 `.cov` 维度。

## 配置要点

GAMMA 和 GeoTIFF 这类角度型 reader 使用 value + azimuth/incidence 几何文件：

```yaml
sar_config:
  reader: gamma_tiff
  mode: unwrapped_phase
  files:
    value: Wushi_asc_unw.tif
    geometry:
      azimuth: Wushi_asc_azi.tif
      incidence: Wushi_asc_inc.tif
  read:
    factor_to_m: 1.0
```

GMTSAR direct-projection reader 不读取 azimuth/incidence，而是直接读取观测值和 ENU projection：

```yaml
sar_config:
  reader: gmtsar
  mode: range_offset
  files:
    value: azimuth_range/T33D_range.grd
    projection:
      east: enu_range/e_sample.grd
      north: enu_range/n_sample.grd
      up: enu_range/u_sample.grd
```

`files.value` 与 `files.projection.east/north/up` 必须描述同一个原始正方向。GMTSAR
`mode: range_offset` 默认认为 value 和 projection 都以朝向卫星为正；`mode: azimuth_offset`
默认认为 value 和 projection 都以沿 heading 为正。不要只手工翻转 value 而不同时表达 projection
正方向，确实需要覆盖时应使用 `mode/preset/convention`。

### 输出前缀和 `output_suffix`

`sar_config.outName` 建议只写轨道或数据集基础名，例如 `S1_T033D`。当
`sar_config.output_suffix: auto` 时，程序会把 `range_offset` 的有效输出前缀解析为
`S1_T033D_RngOff`，把 `azimuth_offset` 解析为 `S1_T033D_AziOff`。如果 `outName` 已经带有
同名后缀，`auto` 不会再重复追加；若希望完全手动控制命名，可写 `output_suffix: none`。

因此推荐写法是：

```yaml
sar_config:
  outName: S1_T033D
  mode: range_offset
  output_suffix: auto
```

正式输出会是 `S1_T033D_RngOff_ifg.txt/.rsp/.cov`。相位或 LOS disp. 没有自动后缀，
仍输出 `<outName>_ifg.*`。

如果只希望协方差估计和正式降采样处理一个关注区域，使用顶层 `processing_region`：

```yaml
processing_region:
  enabled: true
  coord_type: lonlat
  geometry: box
  box: [lon_min, lon_max, lat_min, lat_max]
```

`processing_region` 不影响 `-s` quick-look。原始图只想放大局部时用
`check_plots.raw.coordrange`；降采样结果图只想显示局部时用 `check_plots.decim.coordrange`。
这两个 `coordrange` 都只是绘图范围，不裁剪正式处理数据。

range/azimuth offset 常有局部粗差或闪烁噪声。若希望用滤波后的图控制 quadtree 网格，但最终观测仍从原始数据提取，可启用
`downsample.guide_grid`：

```yaml
downsample:
  method: std
  guide_grid:
    enabled: true
    filter:
      kind: gaussian
      sigma: 1.5
      unit: km
```

若要真实删除粗差点，使用 `sar_config.data_filters`；不要把粗差剔除、协方差掩膜、正式处理区域和
std 近场细化混在一个字段里。

<a id="legacy-script-mapping"></a>

## 旧脚本到新 CLI 的映射

旧脚本核心逻辑：

```python
downsampler.stdBased(threshold=0.02, plot=False, verboseLevel="minimum")
downsampler.writeDownsampled2File(prefix=outName + "_ifg", rsp=True)
```

这段代码是旧脚本的最小示意。当前 `ecat-downsample -d` 写矩形 `.rsp` 时默认使用 18 列 full-corner cell；`trirb` 或三角格网复用正式使用 8 列三角 `.rsp`。旧 10 列矩形 `.rsp` 仍可通过 `from_rsp` 读取，但新案例里的矩形输出应以 full-corner 语义为准。

CLI 路线：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase -o downsample.yml
ecat-downsample -f downsample.yml -s
ecat-downsample -f downsample.yml -c
ecat-downsample -f downsample.yml -d
```

| 旧脚本或人工步骤 | 新 CLI 阶段 | 作用 |
| --- | --- | --- |
| 读取 GAMMA/GeoTIFF/GMTSAR 并画原始图 | `ecat-downsample -f downsample.yml -s` | 先确认读入没有问题，不产生最终反演文件 |
| `covarSAR-Step1.py` | `ecat-downsample -f downsample.yml -c` | 输出 CSI `Covariance_estimator*.cov` 估计器，给后续降采样协方差矩阵使用 |
| `downsampleSAR-Step2.py` | `ecat-downsample -f downsample.yml -d` | 生成 `<effective_outName>_ifg.txt/.rsp/.cov`，作为后续反演输入 |
| `downsampleSAR-Step2_NoFault.py` | `downsample.method: std` | 不依赖断层的基础降采样 |
| `downsampleSAR-Step2_WithFault.py` | `fault_traces` / `fault_models` / `focus_region` | 迹线叠加、`trirb` 或近场细化分别用不同 YAML 字段表达 |

旧脚本里的 `reject_pixels_fault(...)` 不再作为推荐路径维护。通常应判断真实需求：删除粗差用
`data_filters`，只处理科学关注区用 `processing_region`，让近场更细用 `focus_region` 或
`trirb`。完整字段对照见 [InSAR 降采样 Step1/Step2 调参](../workflows/02a_insar_downsampling_two_step.md)。

## Adapter 与时序数据

标准 GAMMA、GMTSAR、HyP3、GeoTIFF 或 optical 产品优先走 `ecat-downsample`。只有读入阶段不适合标准 reader，
或用户已经自行构造好 `csi.insar` / `csi.opticorr` 对象时，才使用 adapter 模板：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase -o downsample.yml --copy-adapter-template
```

通常只修改 `input_adapter.py` 和 `downsample.yml`。完全绕过标准 reader 时，应设置
`general.origin: manual` 并填写 `lon0/lat0`。时序 InSAR 需要复用同一 `.rsp` 网格时，使用 adapter
模板中的 `run_timeseries_downsampling.py` 和 `timeseries.mode: reference_grid`。具体见
[自定义读入 Adapter 降采样](../workflows/02b_adapter_downsampling.md)。

## 应保留输出

```text
raw quick-look figure
sar_output.txt
Covariance_estimator.cov
<effective_outName>_ifg.txt
<effective_outName>_ifg.rsp
<effective_outName>_ifg.cov
<effective_outName>_decim.png
<effective_outName>_run_metadata.yml
<effective_outName>_downsample_report.yml
```

这里的 `<effective_outName>` 是 `outName` 经过 `output_suffix` 解析后的有效输出前缀。若启用了
`processing_region`，还应保留 `<effective_outName>_processing_region_report.yml`。这些文件和对应 YAML
应一起保存，后续反演只读取同一前缀，例如 `S1_T033D_RngOff_ifg`。
