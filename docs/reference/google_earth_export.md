# Google Earth Export Reference

本页完整说明 `eqtools.geoexport` 的 CLI、project YAML、Python API、数据映射和错误边界。
第一次导出先读[工作流](../workflows/06_google_earth_export.md)；只需复制常用片段看
[短例](../examples/google_earth_export.md)。

## 阅读路径

- 已有 downsample reader 配置，只导出全分辨率观测：看
  [Reader 配置联动](#downsample-integration)。
- 一幅全分辨率观测：看 [CLI](#cli-reference) 的 `observation-grid`。
- CSI 降采样单元：看 `varres` 和 [CSI varres](#csi-varres)。
- 多图层：看 [Project YAML](#project-yaml)。
- 内存 fault/滑动/地震：看 [Python API](#python-api)。
- 导出失败或担心扭曲：看 [坐标与数值保证](#scientific-invariants) 和
  [错误行为](#error-behavior)。

<a id="scientific-invariants"></a>
## 坐标与数值保证

数据流固定为：

```text
权威科学文件或对象
  -> 显式只读 adapter
  -> RasterLayer / VectorLayer
  -> KML/KMZ 显示副本
```

导出器保证：

- 不修改输入数组、fault、slip 或 `seismiclocations`；
- 不重新解释 reader、LOS、look side、offset、单位或正号；
- 不调用 BLSE/VCE/SMC/FULLSMC，不读取 solver 参数向量；
- display factor、colormap、`vmin/vmax`、`symmetry` 和 `alpha` 只影响显示；
- raster 使用像元中心推导真实外边界，不把中心点极值直接当边界；
- GroundOverlay 只接受经纬两轴等间距的规则网格，不把不等间距轴压进矩形；
- `0..360` longitude 会等价规范为 KML 的 `-180..180` 表示；跨日界线几何仍拒绝；
- CSI `.txt/.rsp` 的 row id、cell 顶点和值严格同序；
- 输出替换先完成临时 KMZ，再原子替换指定文件。

`vmin/vmax` 总是在 `display_factor` 之后的显示数值中定义。例如米转厘米时：

```yaml
style:
  display_factor: 100.0
  display_unit: cm
  vmin: -10.0
  vmax: 10.0
```

颜色范围是 `[-10, 10] cm`。源科学文件保持不变，但 raster KMZ 只保存颜色化 PNG 和
显示元数据，不保存逐像元数值，不能从 KMZ 反向恢复科学栅格。Vector feature 的属性
才会写入 KML `ExtendedData`。`vmin/vmax` 必须同时设置或同时留空；显式范围优先于
`symmetry`。

<a id="downsample-integration"></a>
## Reader 配置联动

该入口复用 `downsample.yml` 中已经明确的 reader、mode、单位、正号、look side、
数据改正和二维坐标，只写全分辨率观测 KMZ：

```yaml
export:
  google_earth:
    enabled: true
    file: auto
    mask: source_valid
    visible: true
    style:
      cmap: RdBu_r
      display_factor: 100.0
      display_unit: cm
      vmin: null
      vmax: null
      symmetry: true
```

运行时不加阶段选项：

```bash
ecat-downsample -f downsample.yml
```

`-s` 只做 quick-look，`-c` 只估计协方差，`-d` 只做正式降采样；它们不触发这个
KMZ。因此反复调协方差或降采样参数不会重复覆盖显示文件。集成入口不读取或自动加入
CSI `.txt/.rsp`；确实需要显示降采样单元时，显式使用 `varres` 子命令。

字段如下：

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `enabled` | `false` | 是否在无阶段选项运行中导出 |
| `file` | `auto` | `auto` 写 `<outName>_google_earth.kmz`，或给显式 `.kmz` |
| `variables` | `auto` | 每个 component 自动取最终改正值；高级用户可给明确变量列表 |
| `mask` | `source_valid` | 默认保留全部有效源像元；`analysis_valid` 只显示进入分析的子集 |
| `visible` | `true` | Google Earth 中图层初始是否勾选；属于图层状态，不属于 `style` |
| `style` | 模板值 | 与下文通用样式同义；只影响显示 |
| `overwrite` | `true` | 允许同一次数据准备流程替换这一确切输出 |
| `document_name` | `null` | Google Earth 顶层名称 |

联动入口和 project YAML 使用同一字段归属：`visible` 与 `style` 同级。把
`visible` 写进 `style` 会在配置校验阶段明确报错。

`variables: auto` 对 SAR 选择 `observation`，存在确定性改正时选择
`corrected_observation`；对 optical 分别选择 east/north 及其可能的改正值。若要比较
原始和改正结果，可显式写：

```yaml
variables: [observation, corrected_observation]
```

该入口只接受 Google Earth 可精确表示的规则、等间距经纬网格。projected、rotated、
curvilinear、不等间距或跨日界线网格明确报错，不插值、不重投影、不近似成矩形。
需要保留这些网格时使用 ECAT 标准 `.nc/.h5` 和本地 Viewer/PyGMT。

<a id="cli-reference"></a>
## CLI Reference

总入口：

```bash
ecat-export-google-earth --help
python -m eqtools.cli_tools.export_google_earth --help
```

子命令选项分别查看：

```bash
ecat-export-google-earth observation-grid --help
ecat-export-google-earth varres --help
ecat-export-google-earth catalog --help
ecat-export-google-earth project --help
```

### `observation-grid`

```bash
ecat-export-google-earth observation-grid SOURCE -o OUTPUT.kmz [options]
```

| 选项 | 含义 |
| --- | --- |
| `--variable NAME` | `observation`、`corrected_observation`、`east`、`north` 等标准文件变量 |
| `--mask source_valid` | reader 有效像元；默认 |
| `--mask analysis_valid` | 实际进入分析的数据像元 |
| `--mask finite` | 只按所选变量有限值显示 |
| `--layer-id ID` | KMZ 内稳定图层 id |
| `--name NAME` | Google Earth 显示名称 |

文件只有一个未改正 component 时可省略 `--variable`。文件有多个 component，或同时
含原始和改正值时必须明确选择。

### `varres`

```bash
ecat-export-google-earth varres PREFIX -o OUTPUT.kmz [options]
```

| 选项 | 默认 | 含义 |
| --- | --- | --- |
| `--data-type` | `sar` | `sar` 或 `optical` 文本列协议 |
| `--geometry` | `auto` | `auto`、`rectangle` 或 `triangle` |
| `--component` | SAR=`observation`；optical=`magnitude` | 显示分量 |
| `--units` | `m` | `.txt` 中观测值的存储单位 |
| `--convention` | 空 | 正方向说明 |

`PREFIX` 可以是公共前缀，也可以写 `.txt` 或 `.rsp` 成员；两份文件都必须存在。

### `catalog`

```bash
ecat-export-google-earth catalog EVENTS.csv -o events.kmz
```

至少需要 `longitude`/`latitude`，也接受 `lon`/`lat`。`magnitude`/`mag`、`depth`、
`time` 和其他列作为属性保留。

### `project`

```bash
ecat-export-google-earth project google_earth.yml [--force]
```

project 的输出路径写在 YAML 中。所有子命令的 `--force` 只允许替换确切的目标 KMZ，
不会清理目录或删除其他结果。

### 通用显示选项

| 选项 | 含义 |
| --- | --- |
| `--cmap NAME` | Matplotlib colormap |
| `--vmin MIN`、`--vmax MAX` | display factor 后的颜色范围；必须同时使用 |
| `--symmetry`、`--no-symmetry` | 自动范围是否关于 0 对称；显式上下限优先 |
| `--alpha VALUE` | 图层 alpha；`0` 完全透明，`1` 完全不透明 |
| `--display-factor VALUE` | 仅显示用倍率 |
| `--display-unit UNIT` | 倍率后的图例单位 |
| `--normalization linear\|cyclic` | 线性或周期颜色映射 |
| `--cyclic-period VALUE` | cyclic 模式的正周期，例如 wrapped phase 的 `2*pi` 数值 |
| `--document-name NAME` | Google Earth 顶层名称 |
| `--force` | 替换确切输出文件 |

## Project YAML

project 只有一个版本和一层字段，不使用 preset、include、alias 或 environment
override。未知字段立即报错。

```yaml
version: 2

output:
  path: results/context.kmz
  document_name: Research context

layers:
  - id: track_a
    name: Track A
    kind: observation_grid
    source: data/track_a.nc
    variable: observation
    mask: source_valid
    visible: true
    style:
      cmap: RdBu_r
      vmin: -10.0
      vmax: 10.0
      symmetry: true
      alpha: 0.8
      display_factor: 100.0
      display_unit: cm
      normalization: linear

  - id: cells
    name: Downsampled cells
    kind: csi_varres
    source: downsampled/track_a
    data_type: sar
    geometry: auto
    component: observation
    units: m
    convention: positive toward satellite

  - id: events
    name: Earthquakes
    kind: earthquake_catalog
    source: data/events.csv

  - id: fault
    name: Fault trace
    kind: vector
    format: gmt
    source: data/fault_trace.gmt

  - id: boundary
    name: Study boundary
    kind: vector
    format: geojson
    source: data/boundary.geojson
```

### 顶层字段

| 字段 | 必需 | 含义 |
| --- | --- | --- |
| `version` | 是 | 当前只能为 `2` |
| `output.path` | 是 | `.kmz`；相对 YAML 定位 |
| `output.document_name` | 否 | Google Earth 顶层名称 |
| `layers` | 是 | 非空图层列表；`id` 必须唯一 |

### Layer kinds

| kind | 必需字段 | 可选字段 |
| --- | --- | --- |
| `observation_grid` | `id`, `source` | `name`, `variable`, `mask`, `visible`, `style` |
| `csi_varres` | `id`, `source` | `name`, `data_type`, `geometry`, `component`, `units`, `convention`, `visible`, `style` |
| `earthquake_catalog` | `id`, `source` | `name`, `visible`, `style` |
| `vector` | `id`, `source` | `name`, `format: geojson\|gmt`, `visible`, `style` |

### `style` 字段

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `cmap` | `viridis` | 定量颜色表 |
| `vmin/vmax` | 空 | display factor 后的颜色范围；必须同时设置或同时为空 |
| `symmetry` | `false` | 自动范围是否关于 0 对称；显式 `vmin/vmax` 优先 |
| `alpha` | `0.8` | 图层 alpha；`0` 完全透明，`1` 完全不透明 |
| `display_factor` | `1.0` | 仅用于显示值和颜色 |
| `display_unit` | 存储单位 | 图例单位 |
| `normalization` | `linear` | `linear` 或 `cyclic` |
| `cyclic_period` | 空 | cyclic 时必须为正 |
| `line_color` | `#ffffff` | 无定量值 vector 的 `#RRGGBB` 颜色 |
| `line_width` | `1.5` | vector 线宽 |
| `point_scale` | `0.8` | point icon scale |

通用 `LayerStyle` 默认 `symmetry: false`，因为地震震级、深度等定量 vector 不应自动
关于零对称。`downsample.yml` 的全分辨率 SAR/optical 联动模板显式使用
`symmetry: true`，适合有正负的形变和 offset。多轨对比若要求相同颜色尺度，应为每个
图层写相同的 `vmin/vmax`；自动 symmetry 只负责零值居中。

YAML 不创建 CSI fault 或 `seismiclocations`；这些是内存科研对象，使用 Python API。
图层 `id` 必须以字母开头，只能包含字母、数字、点、下划线和连字符。

`visible` 是图层状态字段，不属于 `style`，默认 `true`。linear 模式的解析顺序固定为：

1. `vmin/vmax` 都是数值时严格使用显式范围，`symmetry` 不再改变它；
2. 两者都为空时，先从完整有限显示值求 2–98 百分位；
3. `symmetry: true` 时再以百分位范围的最大绝对值构造零中心范围；
4. `symmetry: false` 时直接使用百分位范围；
5. 只写 `vmin` 或只写 `vmax` 会报错。

cyclic 模式不接受 `symmetry: true`。显式 `vmin/vmax` 必须恰好跨一个
`cyclic_period`。例如 wrapped phase 可使用：

```yaml
style:
  normalization: cyclic
  cyclic_period: 6.283185307179586
  vmin: -3.141592653589793
  vmax: 3.141592653589793
  symmetry: false
```

这些字段只改变颜色映射，不改写相位值。

## Python API

公共模型和 writer：

```python
from eqtools.geoexport import (
    LayerStyle,
    RasterLayer,
    VectorLayer,
    write_kmz,
)
```

### Raster adapters

```text
raster_from_observation_grid(
    grid, *, variable=None, layer_id="observation",
    name=None, mask="source_valid", style=None, visible=True
)

raster_from_observation_file(path, **same_options)

raster_from_arrays(
    values, longitude, latitude, *,
    layer_id="raster", name="Raster", mask=None,
    topology="geographic_rectilinear",
    units=None, convention=None, metadata=None, style=None, visible=True
)
```

`raster_from_arrays` 不重投影、不插值、不转换单位和正号。

<a id="csi-varres"></a>
### CSI varres

```python
from eqtools.csiExtend.downsample import read_csi_varres_result
from eqtools.geoexport import cells_from_varres, cells_from_varres_file

result = read_csi_varres_result(
    "result", data_type="sar", geometry="auto"
)
layer = cells_from_varres(
    result, component="observation", units="m"
)
```

`read_csi_varres_result` 是格式所有者中的纯读取接口；不构造 CSI data object，不读取
`.cov`，不需要投影原点。optical 的 `magnitude` 只定义为
`hypot(east, north)`，不暗含误差传播公式。

### Fault trace 和 patch/slip

```text
trace_from_fault(fault, *, trace="original", ...)
patches_from_fault(
    fault, *, component="total", altitude_mode="surface",
    units=None, convention=None, ...
)
```

`trace` 只能是 `original` 或 `discretized`。patch component：

| 名称 | 来源/公式 |
| --- | --- |
| `strikeslip` | `fault.slip[:, 0]` |
| `dipslip` | `fault.slip[:, 1]` |
| `tensile` | `fault.slip[:, 2]` |
| `total` | `hypot(strikeslip, dipslip)` |
| `rake` | `degrees(atan2(dipslip, strikeslip))` |
| `coupling` | `fault.coupling` |

fault 本身不保证 slip 单位，因此 `units` 默认不猜测。`surface` 默认把 polygon
贴地并保留深度属性；`depth_3d` 显式采用
`altitude_m = -depth_km * 1000`。

adapter 优先读取当前 `fault.patch` 并调用该对象自己的 `xy2ll()`，只在缺少这条路径时
回退到 `patchll`。每个 polygon 保留 patch id、当前显示值、可用 slip components、
rake、深度范围和单位。

### 地震目录与 `seismiclocations`

```text
earthquakes_from_client_catalog(csv_or_dataframe_or_records, ...)
earthquakes_from_seismiclocations(seismic, ...)
```

`seismiclocations` 是 CSI 的权威地震科研对象，geoexport 不建立替代类。adapter
duck-type 读取：

- 必需：`lon`, `lat`；
- 可选：`depth`（km，正向向下）、`mag`/`magnitude`、`time`、
  `event_name`/`event_names`；
- 忽略对象的投影 `x/y`，直接使用经纬度；
- 不调用 `Cmt2Dislocation()`，不解释 `CMTinfo`，不改变对象。

earthquake-client CSV 的 nodal-plane 角度和 CSI `CMTinfo` 可能使用不同单位约定，因此
当前只保留明确的目录属性，不尝试统一 beachball。

### 通用 vector

```text
vector_from_geojson(path, ...)
vector_from_gmt(path, ...)
cells_from_arrays(vertices, values, ...)
```

GeoJSON 接受 Point、LineString、Polygon，并把 MultiPoint、MultiLineString、
MultiPolygon 和 GeometryCollection 确定性展开为简单 feature；GMT 作为 lon/lat
多段线读取。两者不做 CRS 猜测或 sidecar 转换。

### Writer

```python
result = write_kmz(
    [layer_a, layer_b],
    "context.kmz",
    overwrite=False,
    document_name="Research context",
)
```

`ExportResult.output_files`、`layer_ids`、`warnings` 和 `package_mode` 可用于脚本检查。
KMZ 内固定包含 `doc.kml`、`manifest.json` 和必要的 `images/`。
manifest 记录解析后的色限、colormap、alpha、symmetry、normalization、显示倍率和
初始显隐；大 raster 预计临时数组内存超过约 128 MiB 时，`warnings` 会给出提示，但
不会自动降采样或改变源数据。

<a id="error-behavior"></a>
## 错误行为

以下情况不会降级猜测，而是报错：

- standard observation file 缺少 ECAT grid metadata；
- 多变量未选 `variable`，或原始/改正值同时存在但未选择；
- raster 不是精确 geographic rectilinear、轴不等间距/不单调、少于 `2 x 2`
  或跨日界线；
- `.txt/.rsp` 缺失、行数/索引/列协议不一致；
- patch 数量和 slip 行数不一致；
- `seismiclocations` 字段长度不同；
- project YAML 版本、kind、字段或路径错误；
- 图层 id 重复、输出不是 `.kmz`、同名输出未授权覆盖。

## 当前不支持的功能

- 任意 GeoTIFF/NetCDF 自动识别；
- 原始 GAMMA/GMTSAR/HyP3 再解析；
- raster 重投影、曲线网格近似、super-overlay/LOD；
- NetworkLink project bundle；
- beachball、COLLADA 或通用 3D 模型；
- KML/KMZ 反向导入与定量 round-trip。

需要上述功能时，当前导出器不适用；应使用外部 GIS/KML 工具，并保留原始科学文件
作为权威数据源。
