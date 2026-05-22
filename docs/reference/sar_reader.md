# SAR Reader 参考

SAR reader 的任务是把不同产品中的相位、LOS 位移、range offset 或 azimuth offset 统一成 CSI 可用的标量观测：

```text
scalar_observation = ENU_displacement dot projection
```

代码里历史上常用 `self.los` 保存这个三分量向量；在本手册中应把它理解为 `projection`，即 east/north/up 投影向量。

## 快速选择

先判断数据处于哪一层，再选择入口：

| 输入数据 | 应用入口 | 典型用途 |
| --- | --- | --- |
| 已经降采样的 CSI varres 前缀：`.txt/.rsp/.cov` | `csi.insar.read_from_varres(...)` | Wushi、Dingri 等反演脚本直接读入 |
| 外部整理好的 ASCII 点位：`lon lat data err Elos Nlos Ulos` | `csi.insar.read_from_ascii(...)` | Ridgecrest 的普通文本 InSAR 点位 |
| GAMMA 二进制产品：`.phs/.azi/.inc/.rsc` | `GammasarReader` | 原始或栅格化 GAMMA 产品进入降采样 |
| GAMMA GeoTIFF 产品 | `GammaTiffReader` | GeoTIFF 相位、LOS/range 位移或 offset |
| HyP3 GeoTIFF 产品 | `Hyp3TiffReader` | HyP3 解缠相位或 LOS displacement |

如果数据已经是 varres 或 ASCII 点位，不要再走 `GammasarReader` / `GammaTiffReader`。原始 SAR/offset 产品才需要 SAR reader，然后通常进入 [InSAR 降采样](../workflows/02_insar_downsampling.md)。

## Reader 和 Mode

交互脚本和配置模板优先写 `reader + mode`：

| 产品 | reader key | Python 类 | 推荐 mode |
| --- | --- | --- | --- |
| GAMMA 二进制解缠相位 | `gamma` | `GammasarReader` | `unwrapped_phase` |
| GAMMA 二进制 LOS 位移 | `gamma` | `GammasarReader` | `los_displacement` |
| GAMMA 二进制 range offset | `gamma` | `GammasarReader` | `range_offset` |
| GAMMA 二进制 azimuth offset | `gamma` | `GammasarReader` | `azimuth_offset` |
| GAMMA GeoTIFF 解缠相位 | `gamma_tiff` | `GammaTiffReader` | `unwrapped_phase` |
| GAMMA GeoTIFF LOS 位移 | `gamma_tiff` | `GammaTiffReader` | `los_displacement` |
| GAMMA GeoTIFF range offset | `gamma_tiff` | `GammaTiffReader` | `range_offset` |
| GAMMA GeoTIFF azimuth offset | `gamma_tiff` | `GammaTiffReader` | `azimuth_offset` |
| HyP3 解缠相位 TIFF | `hyp3` | `Hyp3TiffReader` | `unwrapped_phase` |
| HyP3 LOS displacement TIFF | `hyp3` | `Hyp3TiffReader` | `los_displacement` |

`mode` 是短名称，reader 会把它映射到完整 preset。常规用户优先用 `mode`；需要写日志或复现实验时，可用完整 `preset`；产品约定不在内置 preset 中时，再显式传 `config`。

`mode`、`preset`、`config` 三者只选一个，并且必须在 `extract_raw_grd()` 前确定，因为它们会影响方位角、入射角和原始值正号的解释。

```python
GammaTiffReader.available_modes()
GammaTiffReader.available_presets()
```

## 文件匹配

配置文件里的 `sar_config.files.prefix` 和显式文件名二选一。

| reader | `prefix` 自动匹配 | 显式文件字段 | 说明 |
| --- | --- | --- | --- |
| `gamma` / `GammasarReader` | `{prefix}*.phs`, `{prefix}*.phs.rsc`, `{prefix}*.azi`, `{prefix}*.inc` | `phsname`, `rscname`, `azifile`, `incfile` | `.rsc` 提供尺寸、地理参考和可选 `WAVELENGTH` |
| `gamma_tiff` / `GammaTiffReader` | `{prefix}*.tif`, `{prefix}*.azi.tif`, `{prefix}*.inc.tif` | `phsname`, `azifile`, `incfile` | value TIFF 可为相位、LOS/range 位移或 azimuth offset；自动匹配会排除 `.azi.tif` 和 `.inc.tif` |
| `hyp3` / `Hyp3TiffReader` | `{prefix}*.tif`, `{prefix}*.azi.tif`, `{prefix}*.inc.tif` | `phsname`, `azifile`, `incfile` | HyP3 派生 TIFF；角度和坐标约定与 GAMMA GeoTIFF 不同 |

offset 产品或同一目录下候选文件较多时，优先写显式文件名，避免 `prefix` 匹配到多个 value raster。

## 最小示例

### GAMMA 二进制解缠相位

```python
from eqtools.csiExtend.sarUtils.readGamma2csisar import GammasarReader

sar = GammasarReader(
    name="asc_phase",
    lon0=lon0,
    lat0=lat0,
    directory_name=".",
    mode="unwrapped_phase",
)
sar.extract_raw_grd(prefix="geo_20250101_20250113")
sar.read_observation(downsample=10)
sar.print_input_summary()
```

### GAMMA 二进制 range offset

```python
sar = GammasarReader(
    name="range_offset",
    lon0=lon0,
    lat0=lat0,
    directory_name=".",
    mode="range_offset",
)
sar.extract_raw_grd(
    phsname="roff_20250101_20250113.phs",
    rscname="roff_20250101_20250113.phs.rsc",
    azifile="off_20250101_20250113.azi",
    incfile="off_20250101_20250113.inc",
)
sar.read_observation(downsample=1)
```

### GAMMA GeoTIFF 解缠相位

```python
from eqtools.csiExtend.sarUtils.readTiff2csisar import GammaTiffReader

sar = GammaTiffReader(
    name="gamma_phase",
    lon0=lon0,
    lat0=lat0,
    directory_name=".",
    mode="unwrapped_phase",
)
sar.extract_raw_grd(
    phsname="phase.tif",
    azifile="azi.tif",
    incfile="inc.tif",
    factor_to_m=1.0,
)
sar.read_observation(wavelength=0.056)
```

### GAMMA GeoTIFF LOS 位移或 range offset

```python
sar = GammaTiffReader(
    name="gamma_los",
    lon0=lon0,
    lat0=lat0,
    directory_name=".",
    mode="los_displacement",
)
sar.extract_raw_grd(prefix="los_product", factor_to_m=1.0)
sar.read_observation()
```

```python
sar = GammaTiffReader(
    name="gamma_range",
    lon0=lon0,
    lat0=lat0,
    directory_name=".",
    mode="range_offset",
)
sar.extract_raw_grd(prefix="range_product", factor_to_m=1.0)
sar.read_observation()
```

`los_displacement` 默认把原始值理解为正值朝向卫星；`range_offset` 默认把原始值理解为正值远离卫星。

### HyP3 GeoTIFF

```python
from eqtools.csiExtend.sarUtils.readTiff2csisar import Hyp3TiffReader

sar = Hyp3TiffReader(
    name="hyp3_los",
    lon0=lon0,
    lat0=lat0,
    directory_name=".",
    mode="los_displacement",
)
sar.extract_raw_grd(prefix="hyp3_product", factor_to_m=1.0)
sar.read_observation()
```

HyP3 reader 的角度、坐标和投影约定已由 `hyp3` preset 处理。不要把 HyP3 TIFF 误用 `gamma_tiff` reader 读入。

### 外部 ASCII 点位数据

如果 InSAR 数据已经由外部流程整理为点位文本，直接使用 CSI 的 ASCII 入口：

```python
from csi.insar import insar

sar = insar("coAscsar", lon0=lon0, lat0=lat0, verbose=False)
sar.read_from_ascii("coAscending_los_CSI.dat", factor=1.0, header=1)
sar.buildDiagCd()
```

列顺序为：

```text
lon lat data err Elos Nlos Ulos
```

`data` 是 LOS 位移或 LOS 速率等标量观测，取决于输入文件和 `factor`；`err` 会进入 `sar.err` 并用于 `buildDiagCd()`；`Elos/Nlos/Ulos` 是 LOS 投影向量分量，不是 GPS 的 east/north/up 位移。外部文件若把第 4 列写成 `wt`，应先确认它是否真的是误差；如果是权重，需要先转换为误差。

## Mode 和 Preset 字典

| reader | mode | full preset | 常见用途 | 方位角含义 | 原始值正号 |
| --- | --- | --- | --- | --- | --- |
| `GammasarReader` | `unwrapped_phase` | `gamma_unwrapped_phase` | GAMMA `.phs` 解缠相位 | right look-away | unwrapped phase |
| `GammasarReader` | `los_displacement` | `gamma_los_displacement` | GAMMA LOS 位移 | right look-away | toward satellite |
| `GammasarReader` | `range_offset` | `gamma_range_offset` | GAMMA range offset | right look-away | away from satellite |
| `GammasarReader` | `azimuth_offset` | `gamma_azimuth_offset` | GAMMA azimuth offset | right look-away | along heading |
| `GammaTiffReader` | `unwrapped_phase` | `gamma_tiff_unwrapped_phase` | GAMMA GeoTIFF 解缠相位 | heading | unwrapped phase |
| `GammaTiffReader` | `los_displacement` | `gamma_tiff_los_displacement` | GAMMA GeoTIFF LOS 位移 | heading | toward satellite |
| `GammaTiffReader` | `range_offset` | `gamma_tiff_range_offset` | GAMMA GeoTIFF range offset | heading | away from satellite |
| `GammaTiffReader` | `azimuth_offset` | `gamma_tiff_azimuth_offset` | GAMMA GeoTIFF azimuth offset | heading | along heading |
| `Hyp3TiffReader` | `unwrapped_phase` | `hyp3_unwrapped_phase` | HyP3 解缠相位 TIFF | right LOS toward | unwrapped phase |
| `Hyp3TiffReader` | `los_displacement` | `hyp3_los_displacement` | HyP3 LOS displacement TIFF | right LOS toward | toward satellite |

`range_offset` 不是底层 `observation_type`。它是 preset 层的产品语义：底层仍按 `los_displacement` 类型处理，但默认 `input_value_convention="away_from_satellite"`。

## 单位和正负号

| 输入数据 | 推荐设置 | 说明 |
| --- | --- | --- |
| 解缠相位，单位 rad | `mode="unwrapped_phase"`, `factor_to_m=1.0` | `read_observation()` 用 `phase * wavelength / (-4*pi)` 转 LOS 位移 |
| LOS/range 位移，单位 m | `mode="los_displacement"` 或 `range_offset`, `factor_to_m=1.0` | 只处理正号约定，不做相位转换 |
| LOS/range 位移，单位 cm | `factor_to_m=0.01` | 先转 m，再按 preset 处理正号 |
| LOS/range 位移，单位 mm | `factor_to_m=0.001` | 先转 m，再按 preset 处理正号 |
| azimuth offset | `mode="azimuth_offset"` | 进入 along-track 投影，不是 LOS 投影 |

`factor_to_m` 只做数值缩放，不表达物理语义。解缠相位通常保持 `factor_to_m=1.0`；如果在 `mode="unwrapped_phase"` 时设置了非 1.0 的 `factor_to_m`，reader 会给出 warning。

GAMMA 二进制 reader 会读取 `.rsc` 中的 `WAVELENGTH`。如果显式传入的 `wavelength` 与 `.rsc` 不一致，会 warning 并使用 `.rsc` 值。

如果产品说明与内置正号约定不同，不要在脚本里手工翻转数组。优先用更合适的 `mode` / `preset`，或显式覆盖：

```python
sar.read_observation(
    observation_type="los_displacement",
    input_value_convention="toward_satellite",
)
```

常用高级字段：

| 字段 | 含义 |
| --- | --- |
| `observation_type` | 原始观测类型：`phase_los`, `los_displacement`, `azimuth_offset` |
| `input_azimuth_role` | 输入方位角物理含义：航向、look-away、LOS-toward |
| `look_side` | 右视或左视；只在 phase/LOS displacement 投影中需要 |
| `input_value_convention` | 原始观测值正号约定 |
| `wavelength` | 解缠相位转 LOS 位移时使用的波长 |

## 诊断和 Quick-Look

读取后先检查 reader 诊断，再进入降采样。

```python
sar.print_input_summary()
sar.show_input_summary()
```

诊断会列出最终生效的观测类型、原始方位角/入射角统计和观测值范围。若原始角度单位是 radian，也会同时给出 degree 值，方便人工检查。

`plot_sar_values()` 默认绘制转换后的观测值，也就是将进入 CSI 的 `self.vel`：

```python
sar.plot_sar_values(value_space="observation", factor4plot=100)
```

如果需要检查产品原始读入值，使用 `raw`：

```python
sar.plot_sar_values(value_space="raw", factor4plot=1, cb_label="Raw value")
```

`raw` 不做相位到位移转换，也不做朝向/远离卫星的符号转换。旧入口 `plot_raw_sar()` 只是兼容别名；新脚本使用 `plot_sar_values()`。

在 CLI 降采样流程中，`ecat-downsample -f downsample.yml -s` 就是用于这类 quick-look 检查。后续 `-c` 做协方差估计，`-d` 正式降采样；完整流程见 [InSAR 降采样](../workflows/02_insar_downsampling.md)，按案例 Step1/Step2 手动调参见 [InSAR 降采样两步走](../workflows/02a_insar_downsampling_two_step.md)。

## 降采样配置入口

新建 SAR 降采样目录时，先生成配置模板：

```bash
ecat-generate-downsample \
  --mode sar \
  --sar-reader gamma \
  --sar-mode unwrapped_phase \
  --downsample-method std \
  -o downsample.yml
```

GeoTIFF 示例：

```bash
ecat-generate-downsample \
  --mode sar \
  --sar-reader gamma_tiff \
  --sar-mode unwrapped_phase \
  -o downsample_gamma_tiff.yml
```

生成后重点检查：

- `sar_config.reader` 是否匹配产品格式。
- `sar_config.mode` 是否匹配观测物理量。
- `sar_config.files` 是 `prefix` 还是显式文件名。
- `factor_to_m` 是否只用于位移/offset 单位转换。
- `wavelength` 是否与产品元数据一致。
- `covar.mask_out` 是否排除主形变区。

CLI 参数详见 [CLI 命令参考](cli.md#降采样配置)，案例见 [InSAR 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md)。

## 输出进入反演

SAR reader 读入的是原始或栅格产品。完成降采样后，反演脚本通常读取降采样前缀：

```python
sar = insar("S1T056A_ifg", lon0=lon0, lat0=lat0, verbose=False)
sar.read_from_varres("../InSAR/downsample/S1T056A_ifg", cov=True)
```

如果没有完整协方差，线性案例可能读入后构造对角阵：

```python
sar.read_from_varres("../InSAR/Dingri_2020_T012A/downsampled/S1_T012A_ifg")
sar.buildDiagCd()
```

三角单元降采样结果需要按案例设置：

```python
sar.read_from_varres("../InSAR/downsample/S1_tri_ifg", triangular=True, cov=True)
```

## 常见错误检查

- 把外部 ASCII 点位误当成原始 SAR reader 输入。
- 解缠相位用了 `factor_to_m=0.01` 或 `0.001`，导致相位在转位移前被错误缩放。
- range offset 用了 `los_displacement`，但产品正号实际是远离卫星。
- HyP3 TIFF 用 `gamma_tiff` reader 读取，角度和坐标约定不匹配。
- `prefix` 匹配到多个 value TIFF，或把 `.azi.tif/.inc.tif` 混成观测值。
- 手动翻转数组正负号，但没有在配置、图件和案例说明中记录。
- `geodata` 顺序与配置中的 `verticals`、`sigmas`、`faults` 顺序不一致。

## 相关页面

- [InSAR 与 GPS 数据读取](../workflows/01_data_reading_insar_gps.md)
- [InSAR 降采样](../workflows/02_insar_downsampling.md)
- [InSAR 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md)
- [CLI 命令参考](cli.md)
