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
| GMTSAR-style direct-projection NetCDF/GRD + ENU 投影系数 | `GmtsarReader` | GMTSAR phase/LOS/range/azimuth 栅格产品 |

如果数据已经是 varres 或 ASCII 点位，不要再走原始 SAR reader。原始 SAR/offset 产品才需要 SAR reader，然后通常进入 [InSAR 降采样](../workflows/02_insar_downsampling.md)。降采样入口的完整字段字典见 [降采样超级入口参考](downsampling_app.md)。

## 目标正方向

SAR reader 的内部合同是：原始产品可以有自己的正号约定，但进入 CSI 的 value 和 projection 必须使用同一个目标正方向。默认规则如下：

| mode / 产品语义 | 默认目标正方向 |
| --- | --- |
| `phase_los` / `unwrapped_phase` | 相位转换后朝向卫星为正 |
| `los_displacement` | 朝向卫星为正 |
| `range_offset` | 朝向卫星为正 |
| `azimuth_offset` | 沿 heading 为正 |

后续反演使用的是 `scalar_observation = ENU_displacement dot projection`。因此检查正负号时，要同时看观测值和 projection 三分量；`print_input_summary()` 会输出 projection 均值辅助判断。`input_value_convention` 描述的是原始文件正号，不等同于最终目标正方向。

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
| GMTSAR-style 解缠相位 GRD/NetCDF | `gmtsar` | `GmtsarReader` | `phase_los` 或 `unwrapped_phase` |
| GMTSAR-style LOS displacement GRD/NetCDF | `gmtsar` | `GmtsarReader` | `los_displacement` |
| GMTSAR-style range offset GRD/NetCDF | `gmtsar` | `GmtsarReader` | `range_offset` |
| GMTSAR-style azimuth offset GRD/NetCDF | `gmtsar` | `GmtsarReader` | `azimuth_offset` |

`mode` 是短名称，reader 会把它映射到完整 preset。常规用户优先用 `mode`；需要写日志或复现实验时，可用完整 `preset`；产品约定不在内置 preset 中时，再显式传 `config` 或 YAML 中的 `convention`。

`mode`、`preset`、`config/convention` 三者只选一个，并且必须在 `extract_raw_grd()` 前确定。角度型 reader 会用它们解释方位角、入射角和原始值正号；GMTSAR 这类 direct-projection reader 会用它们解释原始值正号和输入 ENU 投影系数的正方向。

```python
GammaTiffReader.available_modes()
GammaTiffReader.available_presets()
```

## 文件匹配

配置文件里的 `sar_config.files.prefix` 和显式文件名二选一。

| reader | `prefix` 自动匹配 | 显式文件字段 | 说明 |
| --- | --- | --- | --- |
| `gamma` / `GammasarReader` | `{prefix}*.phs`, `{prefix}*.phs.rsc`, `{prefix}*.azi`, `{prefix}*.inc` | `files.value`, `files.metadata`, `files.geometry.azimuth`, `files.geometry.incidence` | `.rsc` 提供尺寸、地理参考和可选 `WAVELENGTH` |
| `gamma_tiff` / `GammaTiffReader` | `{prefix}*.tif`, `{prefix}*.azi.tif`, `{prefix}*.inc.tif` | `files.value`, `files.geometry.azimuth`, `files.geometry.incidence` | value TIFF 可为相位、LOS/range 位移或 azimuth offset；自动匹配会排除 `.azi.tif` 和 `.inc.tif` |
| `hyp3` / `Hyp3TiffReader` | `{prefix}*.tif`, `{prefix}*.azi.tif`, `{prefix}*.inc.tif` | `files.value`, `files.geometry.azimuth`, `files.geometry.incidence` | HyP3 派生 TIFF；角度和坐标约定与 GAMMA GeoTIFF 不同 |
| `gmtsar` / `GmtsarReader` | 不建议依赖 prefix | `files.value`, `files.projection.east/north/up` | `files.value` 是标量观测；`projection.east/north/up` 是该观测正方向的 ENU 投影系数。direct azimuth 可省略 `projection.up`，由 reader 填 0 |

offset 产品或同一目录下候选文件较多时，优先写显式文件名，避免 `prefix` 匹配到多个 value raster。GMTSAR 建议始终显式写 value 和 ENU 投影文件，让观测值与投影系数的配对关系清楚可查。

上表是 YAML 配置字段。直接调用 Python reader 的 `extract_raw_grd(...)` 时仍使用各 reader 的底层参数名，例如 GAMMA 的 `phsname/rscname/azifile/incfile` 和 GMTSAR 的 `valuefile/eastfile/northfile/upfile`；CLI 会把 YAML 的 `files` 结构映射到这些底层参数。

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

`los_displacement` 默认把原始值理解为正值朝向卫星。`range_offset` 进入 CSI 后同样统一为朝向卫星为正；若某个产品的原始 range offset 是远离卫星为正，reader 会按 preset 或显式 `input_value_convention` 自动翻转数值。

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

### GMTSAR direct-projection GRD

GMTSAR reader 读取的不是方位角和入射角，而是产品已经给出的 ENU 三分量投影系数：

这里的 GRD 指 GMTSAR-style direct-projection NetCDF/GRD 栅格；不要只凭 `.grd` 扩展名套用该入口。其他来源的栅格必须先确认 value 变量、坐标变量、ENU projection 变量和正方向语义，再按本节字段显式配置。

```python
from eqtools.csiExtend.sarUtils.readGmtsar2csisar import GmtsarReader

sar = GmtsarReader(
    name="T33D_range",
    lon0=lon0,
    lat0=lat0,
    directory_name=".",
    mode="range_offset",
)
sar.extract_raw_grd(
    valuefile="azimuth_range/T33D_range.grd",
    eastfile="enu_range/e_sample.grd",
    northfile="enu_range/n_sample.grd",
    upfile="enu_range/u_sample.grd",
)
sar.read_observation()
sar.print_input_summary()
```

YAML 配置中对应写法是：

```yaml
sar_config:
  reader: gmtsar
  mode: range_offset
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
  grid:
    engine:
    value_variable:
    projection_variable:
    lon_name:
    lat_name:
    coord_is_lonlat:
```

`files.value` 与 `files.projection.east/north/up` 必须描述同一个原始正方向。GMTSAR 内置 `range_offset` preset 把原始值理解为朝向卫星为正，并默认把输入投影系数也理解为这个原始值正方向；最终进入 CSI 的 value 和 projection 仍为朝向卫星为正。不要只手工翻转 value 而不翻转投影系数。

azimuth offset 使用沿航向的 direct projection；如果产品只给 east/north 分量，YAML 中可省略 `files.projection.up`，Python API 中可传 `upfile=None`：

```python
sar = GmtsarReader(name="T33D_az", lon0=lon0, lat0=lat0, mode="azimuth_offset")
sar.extract_raw_grd(
    valuefile="azimuth_range/T33D_az.grd",
    eastfile="enu_az/e.grd",
    northfile="enu_az/n.grd",
    upfile=None,
)
sar.read_observation()
```

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
| `GmtsarReader` | `phase_los` / `unwrapped_phase` | `gmtsar_unwrapped_phase` | GMTSAR-style 解缠相位 GRD/NetCDF | direct ENU projection | unwrapped phase |
| `GmtsarReader` | `los_displacement` | `gmtsar_los_displacement` | GMTSAR-style LOS displacement GRD/NetCDF | direct ENU projection | toward satellite |
| `GmtsarReader` | `range_offset` | `gmtsar_range_offset` | GMTSAR-style range offset GRD/NetCDF | direct ENU projection | toward satellite |
| `GmtsarReader` | `azimuth_offset` | `gmtsar_azimuth_offset` | GMTSAR-style azimuth offset GRD/NetCDF | direct ENU projection | along heading |

`range_offset` 不是底层 `observation_type`。它是 preset 层的产品语义：底层仍按 `los_displacement` 类型处理，最终目标正方向与 LOS displacement 一样是朝向卫星。不同产品的原始正号由 `input_value_convention` 表达；例如 GAMMA range preset 可声明原始值远离卫星为正，而 GMTSAR range preset 默认声明原始值朝向卫星为正。

GMTSAR 的 `direct ENU projection` 表示 reader 不再用方位角/入射角构造 projection，而是直接读取 `files.projection.east/north/up`。这些系数的正方向由 direct projection convention 解释；内置 GMTSAR preset 默认认为系数和原始 value 正方向配对。

### GMTSAR Direct-Projection Preset

| mode | full preset | value 正号 | projection role | input projection convention | target projection convention |
| --- | --- | --- | --- | --- | --- |
| `phase_los` / `unwrapped_phase` | `gmtsar_unwrapped_phase` | `unwrapped_phase` | `same_as_observation` | `toward_satellite` | `toward_satellite` |
| `los_displacement` | `gmtsar_los_displacement` | `toward_satellite` | `same_as_observation` | `same_as_value` | `toward_satellite` |
| `range_offset` | `gmtsar_range_offset` | `toward_satellite` | `same_as_observation` | `same_as_value` | `toward_satellite` |
| `azimuth_offset` | `gmtsar_azimuth_offset` | `along_heading` | `same_as_observation` | `same_as_value` | `along_heading` |

`same_as_value` 是 GMTSAR 最常见、也最安全的写法：输入投影系数表达的就是 `files.value` 中正值对应的物理方向。对默认 GMTSAR range offset，这个方向就是朝向卫星；若用户显式声明其他输入正号，reader 会把 value 和 projection 一起转换到朝向卫星目标方向。

### Direct-Projection 转换矩阵

LOS/range 类观测的目标方向统一为朝向卫星。direct-projection reader 会分别根据原始 value 正号和原始 projection 正方向做转换：

| 原始 value 正号 | 原始 projection 正方向 | 目标方向 | value 转换 | projection 转换 |
| --- | --- | --- | --- | --- |
| `toward_satellite` | `toward_satellite` | `toward_satellite` | 不变 | 不变 |
| `away_from_satellite` | `away_from_satellite` | `toward_satellite` | 乘 `-1` | 乘 `-1` |
| `toward_satellite` | `away_from_satellite` | `toward_satellite` | 不变 | 乘 `-1` |
| `away_from_satellite` | `toward_satellite` | `toward_satellite` | 乘 `-1` | 不变 |

`same_as_value` 会先解析为原始 value 正号对应的 projection 方向。调试时看 `print_input_summary()` 中的 `resolved_input_projection_convention`，它会显示 `same_as_value` 最终解析成了 `toward_satellite`、`away_from_satellite`、`along_heading` 还是 `opposite_heading`。

如果 GMTSAR 的 east/north/up 网格变量名不同，可以分别指定：

```yaml
grid:
  value_variable: value
  east_variable: e_var
  north_variable: n_var
  up_variable: u_var
```

### 从 LOS Projection 推导 Azimuth Projection

常规 GMTSAR azimuth offset 应直接使用 `enu_az/e.grd` 和 `enu_az/n.grd`。只有当产品没有提供 azimuth projection、但提供了 LOS/range projection 时，才使用这个高级入口：

```yaml
convention:
  observation_type: azimuth_offset
  input_value_convention: along_heading
  input_projection_role: los
  input_projection_convention: toward_satellite
  look_side: right
```

转换过程是：先把输入 LOS projection 转成朝向卫星方向，取其水平分量并归一化，再按左右视旋转得到 heading：

```text
right look: heading = rotate_cw90(los_toward_horizontal)
left look : heading = rotate_ccw90(los_toward_horizontal)
azimuth projection = [heading_east, heading_north, 0]
```

这里不能把 `input_projection_convention` 写成 `same_as_value`，因为 azimuth value 的 `same_as_value` 只会解析成沿 heading，不会说明输入 LOS projection 是朝向卫星还是远离卫星。

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

角度型 reader 的高级字段：

| 字段 | 含义 |
| --- | --- |
| `observation_type` | 原始观测类型：`phase_los`, `los_displacement`, `azimuth_offset` |
| `input_azimuth_role` | 输入方位角物理含义：航向、look-away、LOS-toward |
| `look_side` | 右视或左视；只在 phase/LOS displacement 投影中需要 |
| `input_value_convention` | 原始观测值正号约定 |
| `wavelength` | 解缠相位转 LOS 位移时使用的波长 |

GMTSAR/direct-projection reader 不使用 `input_azimuth_role`、`azimuth_reference`、`azimuth_unit`、`azimuth_direction`、`incidence_reference` 或 `incidence_unit`。它的高级字段是：

| 字段 | 含义 |
| --- | --- |
| `observation_type` | 原始观测类型：`phase_los`, `los_displacement`, `azimuth_offset` |
| `input_value_convention` | 原始观测值正号约定 |
| `input_projection_role` | 输入 ENU projection 对应观测方向：`same_as_observation`, `los`, `azimuth` |
| `input_projection_convention` | 输入 ENU projection 的正方向：`same_as_value`, `canonical`, `toward_satellite`, `away_from_satellite`, `along_heading`, `opposite_heading` |
| `look_side` | 仅当需要从 LOS projection 推导 azimuth projection 时使用 |
| `wavelength` | 解缠相位转 LOS 位移时使用的波长 |

文件变量名、xarray engine 和坐标名属于 `sar_config.grid`，不属于 convention。

## 诊断和 Quick-Look

读取后先检查 reader 诊断，再进入降采样。

```python
sar.print_input_summary()
sar.show_input_summary()
```

角度型 reader 的诊断会列出最终生效的观测类型、原始方位角/入射角统计、projection 三分量均值和观测值范围。若原始角度单位是 radian，也会同时给出 degree 值，方便人工检查。GMTSAR/direct-projection reader 的诊断会改为列出 `Direct projection convention`，包括输入 projection 的 role、输入正方向、`same_as_value` 解析后的正方向和最终目标正方向；它不会输出无关的方位角/入射角字段。

观测值统计默认面向转换后的观测/形变值：已经调用 `read_observation()` 时统计 `self.vel`；还没进入 CSI 时，会先按当前 `mode/preset` 把 `raw_vel` 转成同一目标正方向后再统计。输出同时给出 robust 区间和 full 区间：

```text
Observation values:
  vel : finite 9800/10000, robust 99% [-0.12, 0.09], full [-10.6, 12.3]
```

`robust 99%` 适合判断常规形变量级和默认显示范围；`full` 用来发现 offset 产品或低质量相位中的极端噪声尾部。如果想看产品文件原始值，不要用默认 summary 代替，改用 `plot_sar_values(value_space="raw")`。

`plot_sar_values()` 默认绘制转换后的观测值，也就是将进入 CSI 的 `self.vel`：

```python
sar.plot_sar_values(value_space="observation", factor4plot=100)
```

如果需要检查产品原始读入值，使用 `raw`：

```python
sar.plot_sar_values(value_space="raw", factor4plot=1, cb_label="Raw value")
```

`raw` 不做相位到位移转换，也不按 `mode/preset` 解释目标正方向。SAR reader 的绘图入口统一使用 `plot_sar_values()`。

在 CLI 降采样流程中，`ecat-downsample -f downsample.yml -s` 就是用于这类 quick-look 检查。若启用了 `sar_config.data_filters`，reader 会先按规则删除粗差点，再输出 summary 和 quick-look；`sar_config.qc.plot.coordrange` 只控制 `-s` 图的显示范围，不裁剪正式处理数据。`sar_output.txt` 会记录 `plot_full_range`、`plot_robust_99_range` 和 `plot_clipped`；前两个分别对应完整极值和稳健显示范围，`plot_clipped` 表示当前色标上下限会截掉多少有效点。后续 `-c` 做协方差估计，`-d` 正式降采样；如果只希望 `-c/-d` 处理关注区域，使用顶层 `processing_region`。完整流程见 [InSAR 降采样](../workflows/02_insar_downsampling.md)，字段字典见 [降采样超级入口参考](downsampling_app.md)，按案例 Step1/Step2 手动调参见 [InSAR 降采样两步走](../workflows/02a_insar_downsampling_two_step.md)。

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

GMTSAR 示例：

```bash
ecat-generate-downsample \
  --mode sar \
  --sar-reader gmtsar \
  --sar-mode range_offset \
  -o downsample_gmtsar_range.yml
```

生成后重点检查：

- `sar_config.reader` 是否匹配产品格式。
- `sar_config.mode` 是否匹配观测物理量。
- `sar_config.files` 是 `prefix` 还是显式文件名；GMTSAR 应重点检查 `files.value` 和 `files.projection.east/north/up` 是否成套对应。
- `factor_to_m` 是否只用于位移/offset 单位转换。
- `wavelength` 是否与产品元数据一致。
- `covar.mask_out` 是否排除主形变区。

CLI 参数详见 [CLI 命令参考](cli.md#降采样配置)，案例见 [InSAR/Offset 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md)。

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
- range offset 用了 `los_displacement`，但产品正号实际不是朝向卫星，且没有用 `range_offset` preset 或显式 `input_value_convention` 表达。
- HyP3 TIFF 用 `gamma_tiff` reader 读取，角度和坐标约定不匹配。
- GMTSAR 手工翻转了 `files.value` 的正负号，却没有同步投影系数方向；应通过 `mode/preset/convention` 表达正号。
- GMTSAR 配置中混入 `azimuth_reference`、`input_azimuth_role` 等角度型字段；direct-projection reader 应改用 `input_projection_role` 和 `input_projection_convention`。
- `prefix` 匹配到多个 value TIFF，或把 `.azi.tif/.inc.tif` 混成观测值。
- 手动翻转数组正负号，但没有在配置、图件和案例说明中记录。
- `geodata` 顺序与配置中的 `verticals`、`sigmas`、`faults` 顺序不一致。

## 相关页面

- [InSAR 与 GPS 数据读取](../workflows/01_data_reading_insar_gps.md)
- [InSAR 降采样](../workflows/02_insar_downsampling.md)
- [InSAR/Offset 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md)
- [CLI 命令参考](cli.md)
