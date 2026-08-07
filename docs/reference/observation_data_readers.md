# 观测数据读入参考

本页集中给出可复制的 Python 读入骨架，覆盖 ECAT 当前支持的原始 SAR、offset、
光学 GeoTIFF、标准全分辨率网格、CSI 降采样文件、外部 ASCII SAR 和 GNSS ENU。第一次只想完成降采样时，优先使用
[InSAR 降采样工作流](../workflows/02_insar_downsampling.md) 和生成的 YAML；只有需要
手动检查 reader、组合其他功能或编写自定义流程时，才复制本页脚本。

## 阅读路径

| 手头数据 | 直接阅读 |
| --- | --- |
| GAMMA `.phs/.rsc/.azi/.inc` | [GAMMA binary](#gamma-binary) |
| GAMMA GeoTIFF | [GAMMA GeoTIFF](#gamma-geotiff) |
| HyP3 GeoTIFF | [HyP3 GeoTIFF](#hyp3-geotiff) |
| GMTSAR/NetCDF value + ENU projection grids | [GMTSAR direct projection](#gmtsar-direct-projection) |
| 双分量 optical offset GeoTIFF | [光学 GeoTIFF](#optical-geotiff) |
| ECAT `.nc/.h5` 标准网格 | [标准全分辨率格式](#standard-observation-grid) |
| CSI `.txt/.rsp/.cov` | [CSI 降采样格式](#csi-varres) |
| 外部七列 SAR 点 | [外部 ASCII SAR](#external-ascii-sar) |
| GNSS ENU 点 | [GNSS ENU](#gnss-enu) |
| 外部自定义 reader | [自定义 reader 边界](#custom-reader) |

字段的完整物理含义、正号和左/右视关系仍以
[SAR Reader](sar_reader.md) 为准。本页不重复列出全部 convention 字典。

## 共同输出契约

当前 reader 先保存二维原始网格，再建立 CSI 分析对象：

| 数据 | reader 二维属性 | CSI 分析属性 |
| --- | --- | --- |
| SAR | `raw_vel`, `raw_mesh_lon`, `raw_mesh_lat`, `raw_projection_full` | `vel`, `lon`, `lat`, `los` |
| optical | `raw_east`, `raw_north`, `raw_mesh_lon`, `raw_mesh_lat` | `east`, `north`, `lon`, `lat` |

SAR 标量满足：

```text
scalar_observation = ENU_displacement dot projection
```

ECAT 的目标 convention 是：

- LOS、range：正值朝向卫星；
- azimuth：正值沿卫星飞行方向；
- optical east/north：分别向东、向北为正；
- 当前 reader 的分析值以米为单位。

因此不要在读入脚本外再手工翻转 `vel` 或 projection。`factor_to_m` 只把文件单位换算
成米；解缠相位保持弧度输入，reader 再按波长转换。

<a id="gamma-binary"></a>
## GAMMA binary

最常见的 prefix 方式会在同一目录中匹配唯一的 `.phs/.rsc/.azi/.inc`：

```python
from eqtools.csiExtend.sarUtils.readGamma2csisar import GammasarReader

sar = GammasarReader(
    name="track_a",
    lon0=100.0,
    lat0=30.0,
    directory_name="raw",
    mode="unwrapped_phase",
    verbose=False,
)
sar.extract_raw_grd(
    prefix="geo_20250101_20250113",
    factor_to_m=1.0,       # phase 保持 rad
    byte_order="native",   # native | little | big
)
sar.read_observation(
    downsample=1,
    acquisition_look_side="right",  # NISAR 左视 GAMMA 产品改为 left
)
sar.print_input_summary()
```

不使用 prefix 时显式写四个文件：

```python
sar.extract_raw_grd(
    phsname="pair.phs",
    rscname="pair.phs.rsc",
    azifile="pair.azi",
    incfile="pair.inc",
    byte_order="native",
)
```

`mode` 可取 `unwrapped_phase`、`los_displacement`、`range_offset`、
`azimuth_offset`。位移或 offset 文件若以 cm 保存，使用 `factor_to_m=0.01`；
不要对 `unwrapped_phase` 使用该倍率。`native` 是历史默认，不会改变已有 GAMMA 案例。

<a id="gamma-geotiff"></a>
## GAMMA GeoTIFF

```python
from eqtools.csiExtend.sarUtils.readTiff2csisar import GammaTiffReader

sar = GammaTiffReader(
    name="track_a",
    lon0=100.0,
    lat0=30.0,
    directory_name="raw",
    mode="los_displacement",
    verbose=False,
)
sar.extract_raw_grd(
    phsname="los.tif",
    azifile="azimuth.tif",
    incfile="incidence.tif",
    phase_band=1,
    azi_band=1,
    inc_band=1,
    factor_to_m=0.01,  # 本例文件单位为 cm
    is_lonlat=True,
)
sar.read_observation(
    downsample=1,
    acquisition_look_side="right",
)
sar.print_input_summary()
```

`is_lonlat=False` 表示 TIFF 坐标由文件 CRS 转换到经纬度。只有产品说明与内置
GAMMA convention 不同，才在 reader config 中显式覆盖 angle/value convention。

<a id="hyp3-geotiff"></a>
## HyP3 GeoTIFF

HyP3 使用独立 preset；不要因为文件同为 TIFF 而改用 `gamma_tiff`：

```python
from eqtools.csiExtend.sarUtils.readTiff2csisar import Hyp3TiffReader

sar = Hyp3TiffReader(
    name="hyp3_track",
    lon0=100.0,
    lat0=30.0,
    directory_name="raw",
    mode="los_displacement",
    verbose=False,
)
sar.extract_raw_grd(
    phsname="displacement.tif",
    azifile="azimuth.tif",
    incfile="incidence.tif",
    factor_to_m=1.0,
    is_lonlat=False,
)
sar.read_observation(downsample=1)
sar.print_input_summary()
```

当前 HyP3 短模式支持 `los_displacement` 和 `unwrapped_phase`。若衍生产品改变了
angle unit、angle reference 或正号，必须依据产品元数据显式配置，不能只按文件名猜测。

<a id="gmtsar-direct-projection"></a>
## GMTSAR direct projection

GMTSAR 入口直接读取 scalar value 和 ENU projection grids，不再要求 azimuth/incidence
角度栅格。east、north 必需，up 可选：

```python
from eqtools.csiExtend.sarUtils.readGmtsar2csisar import GmtsarReader

sar = GmtsarReader(
    name="gmtsar_track",
    lon0=100.0,
    lat0=30.0,
    directory_name="raw",
    mode="range_offset",
    verbose=False,
)
sar.extract_raw_grd(
    valuefile="range_offset.grd",
    eastfile="look_e.grd",
    northfile="look_n.grd",
    upfile="look_u.grd",
    factor_to_m=1.0,
    input_projection_axis="los",
    input_projection_direction="ground_to_sensor",
    coord_is_lonlat=True,
)
sar.read_observation(downsample=1)
sar.print_input_summary()
```

NetCDF 中变量名不唯一时，再补 `value_variable`、`east_variable`、`north_variable`、
`up_variable`、`lon_name` 和 `lat_name`。合法投影语义是：

| axis | direction |
| --- | --- |
| `los` | `ground_to_sensor` 或 `sensor_to_ground` |
| `azimuth` | `along_heading` 或 `opposite_heading` |

`acquisition_look_side` 只在需要从 LOS 投影推导 azimuth heading 时参与几何关系，不是
scalar 正负号字段。

<a id="optical-geotiff"></a>
## 光学 GeoTIFF

当前光学 reader 面向同一 GeoTIFF 内的 east-west 和 south-north 两个 band：

```python
from eqtools.csiExtend.optiUtils.readTiff2csiopti import TiffoptiReader

optical = TiffoptiReader(
    name="optical_pair",
    lon0=100.0,
    lat0=30.0,
    directory_name="raw",
    verbose=False,
)
optical.extract_raw_grd(
    filename="offsets.tif",
    ew_band=1,
    sn_band=2,
    factor_to_m=0.1,  # 按产品单位改成 m
    zero2nan=True,
    downsample=1,
)
optical.read_from_tiff(remove_nan=True)
optical.print_input_summary()
```

`downsample` 是一致作用于 east、north 和坐标的像元 stride；正式降采样通常保持 `1`。
`remove_nan=True` 只把 east/north/坐标均有限的像元送入 CSI，对应二维原始网格仍保留
在 reader 中。

<a id="standard-observation-grid"></a>
## 标准全分辨率格式

ECAT 标准 `.nc` 或显式 `.h5/.hdf5` 保存二维值、逐像元经纬度、projection、mask、
单位、正号、原始/改正值和拓扑；写出时不插值、不重投影。

SAR 中的 `projection_east/north/up` 是 ENU 投影向量的三个系数，并不是三方向位移。
标准化后的 SAR `observation` 仍是一个标量；optical 才直接保存 `east/north` 两个
水平观测分量。自定义 reader 没有提供完整逐像元 projection 时，writer 不会猜测补齐。

从已读 reader 构建并写出：

```python
from eqtools.csiExtend.downsample.observation_grid import (
    build_observation_grid,
    write_observation_netcdf,
)

grid = build_observation_grid(sar, "sar")       # optical 时传 "optical"
write_observation_netcdf(grid, "track_a.nc", verify=True)
```

重新读取并明确选择变量：

```python
from eqtools.csiExtend.downsample.observation_grid import (
    read_observation_grid,
    resolve_observation_variable,
)

grid = read_observation_grid("track_a.nc")
selected = resolve_observation_variable(grid, "corrected_observation")

values = selected.values
longitude = grid.longitude
latitude = grid.latitude
valid = grid.analysis_valid_mask
print(selected.units, selected.positive_convention, grid.topology)
```

`.h5/.hdf5` 必须是 ECAT writer 生成的标准文件，不能把任意 HDF5 当作同一协议读取。
该格式适合无损复用、PyGMT/自定义绘图、Viewer 和 Google Earth adapter；它不是对任意
CSI 子类的隐藏反序列化协议。

<a id="csi-varres"></a>
## CSI 降采样格式

反演脚本读取共同前缀，不写单个成员后缀：

```python
from csi.insar import insar

sar = insar("track_a", lon0=100.0, lat0=30.0, verbose=False)
sar.read_from_varres(
    "downsampled/track_a_ifg",
    triangular=False,
    cov=True,
)
```

对应：

```text
track_a_ifg.txt
track_a_ifg.rsp
track_a_ifg.cov
```

`triangular` 的行为是：

| `.rsp` 类型 | 参数 | 自动行为 |
| --- | --- | --- |
| 四叉树/矩形 | `triangular=False` | 会在 legacy 10 列和 full-corner 18 列矩形布局之间判断 |
| trirb 或其他三角形 `.rsp` | `triangular=True` | 按三角形顶点读取 |

`insar.read_from_varres()` 的默认值是 `triangular=False`，不支持用 `None` 自动区分三角形与
矩形。`cov=True` 会读取完整 `.cov`；此后不要再调用 `buildDiagCd()` 覆盖完整协方差。若没有
`.cov`，使用：

```python
sar.read_from_varres(
    "downsampled/track_a_ifg",
    triangular=False,
    cov=False,
)
sar.buildDiagCd()
```

只需检查文件而不构建 CSI 对象时：

```python
from eqtools.csiExtend.downsample import read_csi_varres_result

result = read_csi_varres_result(
    "downsampled/track_a_ifg",
    data_type="sar",       # optical 时改为 optical
    geometry="auto",
)
print(result.geometry, result.available_components, result.cell_count)
```

这个纯读取接口会从 `.rsp` 自动识别 `rectangle` 或 `triangle`，保持 `.txt/.rsp` 行序和
polygon 顶点，但不读取 `.cov`，适合导出与检查。若还要建立 CSI 反演对象，可显式传回：

```python
sar.read_from_varres(
    "downsampled/track_a_ifg",
    triangular=(result.geometry == "triangle"),
    cov=True,
)
```

<a id="external-ascii-sar"></a>

## 外部 ASCII SAR

CSI `insar.read_from_ascii(...)` 的七列契约是：

```text
lon lat data err Elos Nlos Ulos
100.10 30.10 0.012 0.004 -0.42 -0.10 0.90
```

```python
from csi.insar import insar

sar = insar("track_ascii", lon0=100.0, lat0=30.0, verbose=False)
sar.read_from_ascii(
    "track_ascii.txt",
    factor=1.0,
    header=1,
)
sar.buildDiagCd()
```

`data` 是标量观测，`err` 是标准差或不确定度，后三列是 ENU projection。第 4 列始终读入
`sar.err`；若文件保存的是权重，必须先按其定义转换为标准差。`factor` 同时缩放 `data` 和
`err`，不缩放 projection。

<a id="gnss-enu"></a>

## GNSS ENU

CSI `gps.read_from_enu(...)` 的九列契约是：

```text
station lon lat east north up sigma_e sigma_n sigma_u
STA001 100.10 30.10 0.012 -0.004 0.001 0.002 0.002 0.005
```

```python
from csi.gps import gps

gnss = gps("gnss", lon0=100.0, lat0=30.0, verbose=False)
gnss.read_from_enu(
    "gnss_enu.txt",
    factor=1.0,
    minerr=0.001,
    header=1,
    checkNaNs=True,
)
gnss.buildCd(direction="enu")
```

`factor` 同时缩放 ENU 观测和误差。`minerr` 只替换输入文件中等于零的误差，并在缩放之前
按输入单位解释；例如文件单位为 mm、目标为 m 时，使用 `factor=1e-3`，同时以 mm 写
`minerr`。若反演只使用部分方向，`buildCd(direction=...)` 和后续配置必须采用一致分量。

<a id="custom-reader"></a>
## 自定义 reader 边界

外部格式没有稳定公共协议时，不应继续向主配置增加平台猜测。先生成 adapter 模板：

```bash
ecat-generate-downsample --mode sar --copy-adapter-template -o downsample.yml
```

adapter 只负责把外部文件变成合法 CSI `insar` 或 `opticorr` 对象，再调用
`run_downsample_from_data(...)` 进入同一改正、协方差和降采样运行时。完整契约见
[自定义读入 Adapter 降采样](../workflows/02b_adapter_downsampling.md)。

## 与导出联动

从同一 reader 配置生成 Google Earth 全分辨率显示副本，只需在 YAML 启用：

```yaml
export:
  google_earth:
    enabled: true
    file: auto
```

不加 quick-look、协方差或降采样阶段选项：

```bash
ecat-downsample -f downsample.yml
```

输出默认是 `<outName>_google_earth.kmz`，只包含全分辨率最终观测，不自动包含
降采样单元。网格不能被 Google Earth 精确表示时明确报错，绝不插值。单独转换标准
文件、varres、断层或地震目录时，使用
[Google Earth Export](google_earth_export.md)。

## 最小核对清单

- reader 与产品来源匹配，不按扩展名混用 preset；
- `mode` 与文件物理量匹配；
- phase 是 rad，位移/offset 已通过 `factor_to_m` 转为 m；
- GAMMA binary 的 byte order 已核实，未知时保留历史默认 `native`；
- 左/右视只描述 acquisition geometry，不代替 raw value convention；
- `print_input_summary()` 的 value range、projection 和有效点数合理；
- varres 的 `triangular` 与 `.rsp` 几何一致，完整 `.cov` 没有被对角阵覆盖；
- 外部 SAR 第 4 列是标准差而不是未经转换的权重；
- GNSS 的 ENU 和误差列顺序、单位、`factor/minerr` 已核实；
- 标准网格或 KMZ 导出没有被当作反演权威输入反向使用。

## 相关页面

- [InSAR 降采样工作流](../workflows/02_insar_downsampling.md)
- [反演前读取 InSAR 与 GNSS 数据](../examples/inversion_data_loading.md)
- [SAR projection conventions](../concepts/sar_projection_conventions.md)
- [SAR Reader 完整语义](sar_reader.md)
- [Downsampling App 配置字段](downsampling_app.md)
- [Observation Correction and Grid Export](observation_correction_export.md)
- [Google Earth Export](google_earth_export.md)
