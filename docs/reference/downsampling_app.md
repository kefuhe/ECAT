# 降采样超级入口参考

本页是 `ecat-downsample` 的字段字典和执行逻辑参考。若只是想跑通流程，先读 [InSAR 降采样](../workflows/02_insar_downsampling.md)；若要理解 SAR 观测方向、GMTSAR direct-projection 或 `mode/preset/convention`，读 [SAR Reader 参考](sar_reader.md)。

## 入口定位

降采样超级入口负责把原始 SAR、offset 或 optical 产品整理为 CSI 风格的反演输入：

```text
raw product -> reader -> optional data filters -> optional processing region -> covariance/downsampling -> <effective_outName>_ifg.txt/.rsp/.cov
```

常用命令：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode range_offset -o downsample.yml
ecat-downsample -f downsample.yml -s
ecat-downsample -f downsample.yml -c
ecat-downsample -f downsample.yml -d
```

模块形式：

```bash
python -m eqtools.cli_tools.process_data_downsampling -f downsample.yml -s
```

## 执行顺序

SAR 数据的核心执行顺序固定为：

```text
load config
normalize/validate config
configure downsample.compute.cutde_backend
resolve projection origin
build SAR reader
extract_raw_grd()
read_observation()
checkZeros() / checkNaNs() / checkLosEqualsOne()  # SAR only
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
  if -d and downsample.guide_grid.enabled and method in [std, data]:
    build filtered guide image from the processing data
    construct std/data grid on the guide image
    restore the unfiltered processing data
    extract final cell values by downsample.extraction
  if -d without guide_grid:
    construct grid and extract final cell values from processing data
write run metadata
```

`data_filters` 会真实删除读入后的坏点或粗差点；SAR 使用 `sar_config.data_filters`，optical 使用 `optical_config.data_filters`。`processing_region` 只在协方差和正式降采样前保留科学关注区域；`-s` quick-look 不受它影响，应使用数据类型对应的 quick-look 绘图设置控制显示范围。`guide_grid` 在 `processing_region` 之后生效，只影响 `std/data` 的网格生成，不改变最终取值来源。`covar.mask_out` 沿用 CSI 的 `maskOut()` 语义，只在协方差估计阶段排除震源形变区，不改变最终降采样数据。

eqtools 在这里承担流程编排：读入、单位/符号转换、过滤、区域裁剪、YAML 字段校验、报告和输出命名。CSI 承担核心数值对象和算法：`imagecovariance`、`std/data/trirb/from_rsp` 以及最终 varres `.txt/.rsp/.cov` 约定。文档中的配置字段按 eqtools 入口解释；涉及 `imagecovariance` 和 varres 输出时，保持 CSI 命名。

配置文件使用严格字段集：同一含义只保留当前接口，未知字段会直接报错。SAR 绘图配置统一放在 `sar_config.qc.plot`，optical 绘图配置统一放在 `optical_config.qc.plot`；协方差读入抽稀使用 `sar_config.read.downsample_for_covar`，不要在 reader 顶层放快捷字段。

## 顶层配置块

| 字段 | 作用 | 常见值 |
| --- | --- | --- |
| `data_type` | 选择主数据类型 | `sar`, `optical` |
| `general` | 投影原点和局部坐标设置 | `origin/lon0/lat0` |
| `sar_config` | SAR/InSAR/offset 读入、过滤、quick-look 设置 | 见下文 |
| `optical_config` | optical offset 读入和显示设置 | `filename/factor_to_m/bands` |
| `input_adapter` | 可选自定义读入开关；只在 adapter 模板中使用 | `enabled` |
| `processing_region` | SAR 或 optical 的协方差和正式降采样处理区域 | `enabled/coord_type/geometry` |
| `covar` | 协方差估计设置 | `mask_out/function/frac/every/distmax` |
| `downsample` | 降采样方法、计算后端和参数 | `compute`, `std`, `data`, `trirb`, `from_rsp` |
| `fault_traces` | 可选断层迹线叠加，只用于 raw/decim 检查图 | lon/lat 文本文件 |
| `fault_models` | 可选断层模型，用于 `trirb` 计算或 GMT 网格叠加 | `generated_from_trace`, `csi_gmt` |

## `general`

| 字段 | 作用 |
| --- | --- |
| `origin` | `auto` 从输入数据中心推断；`manual` 使用 `lon0/lat0` |
| `lon0` / `lat0` | `origin: manual` 时必填 |

`origin` 只控制 CSI 局部 x/y 坐标原点，不改变原始经纬度观测值。

## `input_adapter`

标准 `ecat-downsample` 直接使用 `sar_config` 或 `optical_config` 读入数据。若用户需要先用自己的脚本读取非标准格式、外部时序 InSAR 或已经构造好的 CSI 对象，可用源码维护的 adapter 模板：

```bash
ecat-generate-downsample -m sar -o downsample.yml --copy-adapter-template
```

生成的 YAML 会包含：

```yaml
input_adapter:
  enabled: true
```

这表示配置允许跳过标准 reader 文件校验，由复制出的 `input_adapter.py` 返回标准 CSI 数据对象。SAR adapter 必须返回包含 `lon/lat/x/y/vel/los` 的 `csi.insar` 对象；optical adapter 必须返回包含 `lon/lat/x/y/east/north/err_east/err_north` 的 `csi.opticorr` 对象。进入该对象之后，`processing_region`、`covar`、`std/data/trirb/from_rsp`、`guide_grid`、`extraction`、报告和检查图全部复用标准 runtime。

完全绕过标准 reader 时必须显式指定投影原点：

```yaml
general:
  origin: manual
  lon0: <project_lon0>
  lat0: <project_lat0>
```

因为程序会在调用 `input_adapter.py` 前解析投影原点，非标准文件没有统一的信息源可供自动推断。

完整操作流程见 [自定义读入 Adapter 降采样](../workflows/02b_adapter_downsampling.md)。本节只作为字段字典。

## SAR 与 optical 的共用和差异

两类数据共用 `general`、顶层 `processing_region`、`covar`、`downsample`、`fault_traces`、`fault_models` 和三步运行方式。差异主要在读入对象和观测分量：

| 项目 | SAR/InSAR/offset | Optical offset |
| --- | --- | --- |
| 数据配置 | `sar_config` | `optical_config` |
| 观测量 | 单个标量 `vel`，配套 ENU projection/LOS | `east` 和 `north` 两个水平分量 |
| 粗差过滤 | `sar_config.data_filters`，按转换后的单标量观测和 projection 过滤 | `optical_config.data_filters`，按 `east/north` 分量或水平模长过滤 |
| quick-look 范围 | `sar_config.qc.plot.coordrange` | `optical_config.qc.plot.coordrange` |
| 正式处理范围 | 顶层 `processing_region` | 顶层 `processing_region` |
| 协方差输出 | `Covariance_estimator.cov` | `Covariance_estimator_East.cov` 和 `Covariance_estimator_North.cov` |
| 降采样检查图 | `<outName>_decim.png` | `<outName>_East_decim.png` 和 `<outName>_North_decim.png` |

## `sar_config`

| 字段 | 作用 |
| --- | --- |
| `outName` | 基础输出前缀；最终 SAR 输出前缀还会经过 `output_suffix` 解析 |
| `output_suffix` | 默认 `auto`；`range_offset` 自动追加 `_RngOff`，`azimuth_offset` 自动追加 `_AziOff`，若 `outName` 已带同名后缀则不重复追加；`none`、`false` 或空值表示不追加，自定义字符串会直接追加 |
| `reader` | `gamma`, `gamma_tiff`, `gmtsar`, `hyp3` |
| `mode` | `unwrapped_phase`, `phase_los`, `los_displacement`, `range_offset`, `azimuth_offset` |
| `preset` | 完整产品 preset；与 `mode`、`convention` 三选一 |
| `convention` | 显式产品语义；高级用户用于非内置产品 |
| `directory` | 数据文件所在目录 |
| `files` | value、角度或 ENU projection 文件 |
| `read` | 读入抽稀、单位缩放和波长 |
| `grid` | raster/grid 技术读入细节，例如 band、变量名、engine 和坐标名 |
| `data_filters` | 真实删除数据点的过滤规则；默认关闭 |
| `qc` | summary 和 quick-look 绘图设置 |

`reader/mode/preset/convention` 的语义见 [SAR Reader 参考](sar_reader.md)。

## `sar_config.files`

| reader | 常用字段 | 说明 |
| --- | --- | --- |
| `gamma` | `prefix` 或 `value + metadata + geometry.azimuth/incidence` | 二进制 GAMMA 产品 |
| `gamma_tiff` | `prefix` 或 `value + geometry.azimuth/incidence` | GeoTIFF value + angle grids |
| `hyp3` | `prefix` 或 `value + geometry.azimuth/incidence` | HyP3 GeoTIFF |
| `gmtsar` | `value + projection.east/north/up` | GMTSAR-style direct-projection GRD/NetCDF |

`prefix` 和显式文件名不要混用。GMTSAR 建议始终显式写 `files.value` 和 `files.projection.*`，让标量观测和 ENU projection 文件成套可查。这里的 GRD/NetCDF 指 GMTSAR-style direct-projection 栅格，不表示任意 `.grd` 文件都可直接套用；其他来源必须先确认变量名、坐标和 projection 正方向。

标准结构如下：

```yaml
sar_config:
  files:
    prefix:
    value:
    metadata:
    geometry:
      azimuth:
      incidence:
    projection:
      east:
      north:
      up:
```

角度型 reader 使用 `prefix` 或 `value/metadata/geometry`；direct-projection reader 使用 `value/projection`，不使用 `geometry`。

## `sar_config.read`

| 字段 | 作用 |
| --- | --- |
| `downsample` | quick-look 和正式降采样读入时的抽稀 |
| `downsample_for_covar` | 协方差估计读入时的抽稀 |
| `zero2nan` | 读入时将 0 值视为无效值 |
| `wavelength` | phase 转 LOS displacement 的波长 |
| `factor_to_m` | 位移/offset 产品单位缩放到米；相位通常保持 `1.0` |

## `sar_config.grid`

`grid` 只放 raster/grid 读入细节，不放文件名和物理正号约定。

| 字段 | 作用 |
| --- | --- |
| `phase_band/azi_band/inc_band` | GeoTIFF band 选择 |
| `engine` | NetCDF/GRD 的 xarray engine；为空时自动选择并按 `netcdf4/h5netcdf/scipy/rasterio` 回退 |
| `value_variable` | GMTSAR/direct-projection 标量观测变量名；为空时优先 `z`，再尝试唯一变量 |
| `projection_variable` | east/north/up 三个 projection grid 共用变量名 |
| `east_variable/north_variable/up_variable` | east/north/up 变量名不同时分别指定 |
| `lon_name/lat_name` | 坐标变量名；为空时尝试 `lon/lat`、`longitude/latitude`、`x/y` |
| `coord_is_lonlat` | `null` 时检查 `x/y` 数值是否像经纬度；`true` 表示用户确认是经纬度；`false` 会拒绝 direct-projection SAR 读入 |

## `data_filters`

`data_filters` 是真实数据过滤层。启用后会先自动执行 `finite` 规则，然后按 `rules` 顺序继续过滤。

```yaml
sar_config:
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
```

默认模板保留一条禁用的 `value_range` 示例，方便用户理解常规写法。它不会删除任何点；启用时需要同时打开全局开关和该规则，并填写 `min/max`。

常用绝对值粗差剔除：

```yaml
data_filters:
  enabled: true
  rules:
    - name: gross_observation_abs
      enabled: true
      kind: value_abs
      value_space: observation
      threshold: 0.5
```

常用范围保留：

```yaml
data_filters:
  enabled: true
  rules:
    - name: valid_observation_range
      enabled: true
      kind: value_range
      value_space: observation
      min: -0.5
      max: 0.5
```

支持的 `kind`：

| `kind` | 作用 | 常用字段 |
| --- | --- | --- |
| `finite` | 内置隐式规则，删除 `vel/lon/lat/projection` 中的 NaN/inf | 自动执行 |
| `value_abs` | 删除 `abs(value) > threshold` 的点 | `threshold`, `value_space: observation` |
| `value_range` | 保留 `[min, max]` 内的点 | `min`, `max` |
| `lonlat_box` | 删除或保留经纬度框 | `box`/`boxes`, `action` |
| `lonlat_polygon` | 删除或保留多边形 | `polygon`/`polygons`/`file`, `action` |
| `projection_norm` | 删除 projection norm 异常点 | `min/max` 或 `target/tolerance` |

区域类规则的 `action`：

| action | 作用 |
| --- | --- |
| `remove_inside` | 删除区域内点，默认值 |
| `keep_inside` | 只保留区域内点 |
| `remove_outside` | 删除区域外点 |
| `keep_outside` | 只保留区域外点 |

`value_space: observation` 表示阈值作用于转换后的 `vel`，即反演实际使用的观测值；它不受 `qc.plot.factor4plot` 影响。过滤报告默认写入 `<outName>_filter_report.yml`。

## `optical_config`

`optical_config` 用于 optical offset 产品。它和 SAR 共用 `general`、顶层 `processing_region`、`covar` 和 `downsample`，但读入后保存的是两个水平分量：

```yaml
optical_config:
  outName: Optical_S2_part1
  directory: ..
  filename: Sagaing_S2_Part1.tif
  vel_type: north
  factor_to_m: 10.0
  ew_band: 1
  sn_band: 2
  remove_nan: true
  output_check: true
  data_filters:
    enabled: false
    report: true
    report_file: auto
    rules:
      - name: valid_horizontal_component_range
        enabled: false
        kind: component_range
        components: [east, north]
        min:
        max:
  qc:
    summary_percentile: 99.0
    plot:
      save_fig: true
      file_path:
      show: true
      rawdownsample4plot: 1
      coordrange:
      data: [east, north]
      vmin: [-1.0, -3.0]
      vmax: [1.0, 3.0]
      factor4plot: 1.0
      symmetry: true
      cmap: cmc.roma_r
```

| 字段 | 作用 |
| --- | --- |
| `outName` | 输出前缀 |
| `directory` / `filename` | optical offset GeoTIFF 所在目录和文件名 |
| `ew_band` / `sn_band` | 东西向和南北向分量所在 band |
| `factor_to_m` | 产品单位转米的比例 |
| `remove_nan` | 读入时删除 NaN 像素 |
| `vel_type` | `trirb` 使用的 optical 分量，`north` 或 `east` |
| `output_check` | 是否输出降采样检查图 |
| `data_filters` | 真实删除 optical 坏点/粗差的规则；默认关闭 |
| `qc.summary_percentile` | summary 和 metadata 使用的稳健统计中心百分比 |
| `qc.plot` | `-s` 原始 optical quick-look 显示设置 |

optical 的 `data_filters` 与 SAR 在同一时机执行，但规则含义不同：SAR 的 `value_*` 规则作用于单标量 `vel`，optical 的 `component_*` 和 `vector_norm_range` 作用于 `east/north` 双分量。不要把 SAR 的 `projection_norm` 或 `value_space` 用到 optical。

### `data_filters` 顶层字段

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | 是否启用过滤层。关闭时所有规则只作为模板/说明，不删除点 |
| `report` | bool | `true` | 是否写过滤报告 |
| `report_file` | string/null | `auto` | `auto` 写为 `<outName>_filter_report.yml`；也可给固定文件名 |
| `rules` | list | 禁用的示例规则 | 规则按列表顺序执行；启用过滤后会先自动执行内置 `finite` |

### 规则通用字段

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `name` | string | 自动名 | 报告中显示的规则名称 |
| `enabled` | bool | `true` | 单条规则开关；默认模板中的示例规则为 `false` |
| `kind` | string | 必填 | 规则类型 |

### `kind: value_abs`（SAR）

删除转换后观测值绝对值过大的点。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `threshold` | number | 是 | 删除 `abs(value) > threshold` 的点 |
| `value_space` | string | 否 | 当前支持 `observation`；默认 `observation` |

示例：

```yaml
- name: gross_observation_abs
  enabled: true
  kind: value_abs
  value_space: observation
  threshold: 0.5
```

### `kind: value_range`（SAR）

只保留转换后观测值位于指定范围内的点。它使用 `min/max`，不是 `threshold`。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `min` | number/null | 至少一个 | 下界；空值表示不限制下界 |
| `max` | number/null | 至少一个 | 上界；空值表示不限制上界 |
| `value_space` | string | 否 | 当前支持 `observation`；默认 `observation` |

示例：

```yaml
- name: valid_observation_range
  enabled: true
  kind: value_range
  value_space: observation
  min: -0.5
  max: 0.5
```

### `kind: lonlat_box`

按经纬度矩形框删除或保留点。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `box` | mapping/list | `box` 或 `boxes` 二选一 | 一个矩形框 |
| `boxes` | list | `box` 或 `boxes` 二选一 | 多个矩形框，多个框取并集 |
| `action` | string | 否 | `remove_inside` 默认；也可用 `keep_inside`, `remove_outside`, `keep_outside` |

`box` 可写成：

```yaml
box:
  lon_min: 96.0
  lon_max: 96.5
  lat_min: 20.0
  lat_max: 20.5
```

也可写成 `[lon_min, lon_max, lat_min, lat_max]`。

示例：

```yaml
- name: remove_noisy_corner
  enabled: true
  kind: lonlat_box
  action: remove_inside
  box:
    lon_min: 96.0
    lon_max: 96.5
    lat_min: 20.0
    lat_max: 20.5
```

### `kind: lonlat_polygon`

按经纬度多边形删除或保留点。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `polygon` | list/mapping | 与 `polygons/file/path/points` 之一 | 单个多边形 |
| `polygons` | list | 可选 | 多个多边形，多个区域取并集 |
| `points` | list | 可选 | 直接给点列 |
| `file` / `path` | string | 可选 | 外部文本文件，至少两列 `lon lat` |
| `action` | string | 否 | `remove_inside` 默认；也可用 `keep_inside`, `remove_outside`, `keep_outside` |

示例：

```yaml
- name: remove_bad_polygon
  enabled: true
  kind: lonlat_polygon
  action: remove_inside
  polygon:
    - [96.0, 20.0]
    - [96.5, 20.0]
    - [96.5, 20.5]
    - [96.0, 20.5]
```

外部文件示例：

```yaml
- name: keep_manual_area
  enabled: true
  kind: lonlat_polygon
  action: keep_inside
  file: keep_area.xy
```

### `kind: projection_norm`（SAR）

按 projection 向量模长过滤。常用于发现 ENU projection 栅格异常。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `min` / `max` | number/null | 与 `target/tolerance` 二选一 | 保留 `min <= norm <= max` |
| `target` | number | 与 `tolerance` 配合 | 目标模长，常用 `1.0` |
| `tolerance` | number | 与 `target` 配合 | 保留 `target ± tolerance` |

示例：

```yaml
- name: projection_unit_norm
  enabled: true
  kind: projection_norm
  target: 1.0
  tolerance: 0.2
```

等价范围写法：

```yaml
- name: projection_norm_range
  enabled: true
  kind: projection_norm
  min: 0.8
  max: 1.2
```

### `kind: component_abs`（optical）

删除指定 optical 分量绝对值过大的点。默认同时检查 `east` 和 `north`，也可以只检查一个分量。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `threshold` | number | 是 | 删除任一指定分量满足 `abs(component) > threshold` 的点 |
| `component` | string | 否 | 单个分量，`east` 或 `north` |
| `components` | list | 否 | 多个分量，默认 `[east, north]` |

示例：

```yaml
- name: gross_east_component
  enabled: true
  kind: component_abs
  component: east
  threshold: 1.0
```

### `kind: component_range`（optical）

只保留指定 optical 分量位于 `min/max` 范围内的点。多个分量同时给出时，所有分量都必须落在范围内。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `min` | number/null | 至少一个 | 下界；空值表示不限制下界 |
| `max` | number/null | 至少一个 | 上界；空值表示不限制上界 |
| `component` / `components` | string/list | 否 | 默认 `[east, north]` |

示例：

```yaml
- name: valid_horizontal_component_range
  enabled: true
  kind: component_range
  components: [east, north]
  min: -1.0
  max: 1.0
```

### `kind: vector_norm_range`（optical）

按水平位移模长 `sqrt(east^2 + north^2)` 过滤，适合删除双分量共同表现为异常大的 optical offset 粗差点。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `min` / `max` | number/null | 与 `target/tolerance` 二选一 | 保留 `min <= norm <= max` |
| `target` | number | 与 `tolerance` 配合 | 目标模长 |
| `tolerance` | number | 与 `target` 配合 | 保留 `target ± tolerance` |

示例：

```yaml
- name: valid_horizontal_norm
  enabled: true
  kind: vector_norm_range
  min: 0.0
  max: 1.5
```

## `processing_region`

`processing_region` 是正式处理区域，不是坏点过滤。它是顶层配置，SAR/InSAR/offset 和 optical offset 共用。启用后，程序先完整读取原始数据并执行对应的 `data_filters`，然后在构造协方差/降采样使用的 CSI 数据对象时，只保留该区域内的点。它会影响 `-c` 和 `-d`，不会影响 `-s` quick-look；quick-look 只看局部时用对应数据类型的绘图范围配置。

```yaml
processing_region:
  enabled: false
  report: true
  report_file: auto
  coord_type: lonlat      # lonlat | xy
  geometry: box           # box | polygon | polygon_file
  box: [95.5, 97.5, 20.5, 22.5]
  polygon:
  polygon_file:
```

字段说明：

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | 是否启用处理区域；关闭时完全不改变当前行为 |
| `report` | bool | `true` | 是否写处理区域报告 |
| `report_file` | string/null | `auto` | `auto` 写为 `<outName>_processing_region_report.yml` |
| `coord_type` | string | `lonlat` | `lonlat` 使用 `lon/lat`；`xy` 使用 CSI 局部 `x/y` |
| `geometry` | string | `box` | `box`, `polygon`, `polygon_file` |
| `box` | list/mapping/null | `null` | `geometry: box` 时使用；列表格式为 `[minlon, maxlon, minlat, maxlat]` 或 `[minx, maxx, miny, maxy]` |
| `polygon` | list/null | `null` | `geometry: polygon` 时使用，至少三个点 |
| `polygon_file` | string/null | `null` | `geometry: polygon_file` 时使用，相对配置文件路径 |

三种范围配置不要混用：

| 配置 | 改变数据点 | 作用阶段 |
| --- | --- | --- |
| `sar_config.qc.plot.coordrange` | 否 | 只控制 `-s` quick-look 显示范围 |
| `optical_config.qc.plot.coordrange` | 否 | 只控制 `-s` optical quick-look 显示范围 |
| `processing_region` | 是 | 控制 `-c` 和 `-d` 的实际处理区域 |
| `downsample.plot_decim.coordrange` | 否 | 只控制降采样检查图显示范围 |

如果开启了 `processing_region`，协方差估计和正式降采样应使用同一个配置重新运行。`covar.mask_out` 仍然只表示在协方差估计中排除震源形变区；它应位于处理区域之内或与处理区域有足够交集。

## `sar_config.qc`

| 字段 | 作用 |
| --- | --- |
| `summary_percentile` | summary 和 metadata 中 robust range 的中心百分比 |
| `plot.value_space` | `observation` 画转换后的观测值；`raw` 画产品原始值 |
| `plot.factor4plot` | 仅显示用比例，例如米转厘米用 `100` |
| `plot.vmin/vmax` | quick-look 色标；命令行 `--vmin/--vmax` 可覆盖 |
| `plot.symmetry` | 是否让色标关于 0 对称 |
| `plot.rawdownsample4plot` | 绘图抽稀 |
| `plot.coordrange` | `-s` quick-look 的显示范围 `[minlon, maxlon, minlat, maxlat]`；不裁剪数据 |
| `plot.file_path/save_fig/show` | quick-look 输出控制 |

`sar_output.txt` 记录：

| 字段 | 作用 |
| --- | --- |
| `plot_full_range` | 所有有限显示值的完整极值 |
| `plot_robust_99_range` | 稳健中心范围，用于判断显示量级 |
| `plot_clipped` | 当前 `vmin/vmax` 截掉的有效点比例 |

## `optical_config.qc`

| 字段 | 作用 |
| --- | --- |
| `summary_percentile` | optical summary 和 metadata 中 robust range 的中心百分比 |
| `plot.data` | quick-look 绘制的分量，常用 `[east, north]` |
| `plot.factor4plot` | 仅显示用比例，例如米转厘米用 `100` |
| `plot.vmin/vmax` | quick-look 色标；可为单个值或 `[east, north]` 两个值，命令行 `--vmin/--vmax` 会同时覆盖两个分量 |
| `plot.symmetry` | 是否让色标关于 0 对称；主要用于 decimated plot 继承 |
| `plot.rawdownsample4plot` | 绘图抽稀；当前主要作为配置记录 |
| `plot.coordrange` | `-s` optical quick-look 的显示范围 `[minlon, maxlon, minlat, maxlat]`；不裁剪数据 |
| `plot.file_path/save_fig/show` | quick-look 输出控制 |

`optical_output.txt` 记录 `east/north` 的完整范围、稳健中心范围、当前色标裁剪比例和水平模长稳健范围。

## `covar`

| 字段 | 作用 |
| --- | --- |
| `do_covar` | 配置中启用协方差估计；也可用 `-c` 临时启用 |
| `mask_out` | 协方差估计时排除震源形变区；不删除最终数据点 |
| `missing_policy` | 直接 `-d` 且缺少已有协方差时的处理方式 |
| `function` | 协方差模型，常用 `exp` 或 `gauss` |
| `frac` | CSI `imagecovariance` 在剩余背景点中的抽样比例 |
| `every` | 经验协方差距离分箱间隔，单位为 CSI 局部 `x/y` 坐标单位，通常是 km |
| `distmax` | 参与协方差拟合的最大距离，单位同 `every` |
| `rampEst` | 调用 CSI 估计协方差前是否估计/移除 ramp |

`mask_out` 可以是一个框，也可以是多个框。它用于背景噪声估计，不应理解为坏点删除或降采样范围控制。

运行 `ecat-downsample -c` 时，eqtools 先按当前配置读入数据、执行 `data_filters`，再在 `processing_region` 内构造轻量 CSI 处理对象。之后程序创建 CSI `imagecovariance`，用 `mask_out` 排除主形变源区，并在剩余背景点上按 `frac/every/distmax/rampEst/function` 拟合经验协方差模型。SAR 写 `Covariance_estimator.cov`；optical offset 分别写 `Covariance_estimator_East.cov` 和 `Covariance_estimator_North.cov`。

`Covariance_estimator*.cov` 是协方差估计器文件，不是最终反演矩阵。运行 `-d` 时，若当前目录存在对应估计器，eqtools 会把降采样后的 CSI 对象交给 CSI `buildCovarianceMatrix()`，写出 `<effective_outName>_ifg.cov`。若没有估计器，则按 `missing_policy` 读取已有矩阵、写单位阵或报错。

对于 SAR，`.cov` 是单标量观测的矩阵；对于 optical offset，East 和 North 会分别估计，并在最终输出中组成分量块矩阵，当前配置不引入 East-North 交叉协方差。CSI 的抽样包含随机抽样步骤，重复运行 `-c` 可能有小差异；正式案例应把 `Covariance_estimator*.cov` 和 YAML 一起保留。

## `downsample`

```yaml
downsample:
  compute:
    cutde_backend: cpp
```

`downsample.compute.cutde_backend` 控制降采样入口中 cutde 使用的后端，当前主要影响
`method: trirb` 的断层 Green 函数和分辨率判据计算。`std` 和 `data` 通常不触发
cutde GF，但仍保留同一配置入口，避免同一降采样命令在不同机器上隐式选择不同后端。

| 字段 | 可选值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `compute.cutde_backend` | `cpp`, `cuda`, `opencl`, `auto` | `cpp` | `cpp` 是跨平台稳定默认；`cuda/opencl` 需要用户显式选择并保证本机环境可用；`auto` 交给 cutde 和环境变量选择 |

| 方法 | 配置块 | 适用场景 |
| --- | --- | --- |
| `std` | `std_config` | 入门首选；按块内统计逐级细化 |
| `data` | `data_config` | 按振幅、梯度或曲率等数据特征细化 |
| `trirb` | `trirb_config` | 有断层模型时按三角分辨率控制降采样，会构建 cutde GF |
| `from_rsp` | `from_rsp_config` | 复用已有 CSI `.rsp` 格网 |

通用提取统计由 `downsample.extraction` 控制，对 `std`、`data`、`trirb` 和
`from_rsp` 都生效。默认值完全等价于 CSI 旧行为：块内观测取均值，误差取标准差，
坐标取块内像素均值，InSAR 投影向量取均值后归一化。

```yaml
downsample:
  extraction:
    value_statistic: mean        # mean | median | center_nearest | trimmed_mean
    error_statistic: std         # std | mad | sem | none
    coordinate_statistic: mean   # mean | block_center | center_nearest
    projection_statistic: mean_normalized # mean_normalized | center_nearest
    trim_fraction: 0.1           # value_statistic: trimmed_mean 时使用
```

`value_statistic` 控制 `<outName>_ifg.txt` 中的降采样观测值。`median` 和
`trimmed_mean` 更抗离群点；`center_nearest` 使用最靠近 cell 中心的原始像素。
`error_statistic` 控制输出误差列，`mad` 是按 median absolute deviation 缩放的稳健误差，
`sem` 是标准误，`none` 写 0。`coordinate_statistic: block_center` 会把输出点放在
cell 几何中心；默认 `mean` 保持旧行为。

`downsample.guide_grid` 是可选的 Guided Quadtree Downsampling（引导式四叉树降采样）入口，
也可理解为 Two-stage Quadtree（两阶段四叉树）。它只允许用于 `method: std` 和 `method: data`。
启用后，程序先执行 quadtree partitioning based on filtered/smoothed interferograms
（基于滤波/平滑干涉图的四叉树划分），再切回原始未滤波数据，按 `downsample.extraction`
提取最终值。它不删除数据，也不改变协方差输入；若要真实剔除粗差，仍使用 `data_filters`。

对规则 SAR/optical 栅格，Gaussian 滤波会优先使用 lon/lat 网格重建后的 NaN-aware 二维滤波；
这适合大幅 InSAR 栅格。只有无法识别为规则栅格时才退回散点邻域滤波。

```yaml
downsample:
  guide_grid:
    enabled: false
    source: filtered_observation
    component: auto        # SAR 为 observation；optical 可用 magnitude/east/north/both
    filter:
      kind: gaussian
      sigma: null          # enabled: true 时必须设置
      unit: km             # km | pixel
      radius_sigma: 3.0
```

`guide_grid` 不用于 `trirb`，因为 `trirb` 的网格由断层 Green 函数、样本权重和分辨率判据控制，
不是由观测图像局部标准差或曲率直接驱动。`trirb` 和 `from_rsp` 仍会使用
`downsample.extraction` 控制最终 cell 值提取。

每次 `-d` 默认会写一个降采样诊断报告，方便检查实际点数、降采样比例、处理区域、
guide-grid 后端和最终取值统计：

```yaml
downsample:
  report:
    enabled: true
    report_file: auto       # auto -> <outName>_downsample_report.yml
    quality: true           # 计算 cell 代表值相对原始像素的 RMS 诊断
```

`quality: true` 只写诊断信息，不改变降采样结果。若数据量很大且只想快速运行，可以设为
`false`，报告仍会保留点数、格网和配置摘要。

`cutde_backend: cuda` 或 `opencl` 是显式高级选项，失败时应修复对应计算环境或改回
`cpp`，程序不会静默把显式 GPU 请求降级为 CPU。每次运行的 `<outName>_run_metadata.yml`
会记录 requested、environment 和 active backend，便于复现和排错；如果当前步骤没有真正触发 cutde 计算，active backend 可能为空，以 requested 和 environment 为准。

常用 `std_config` 字段：

| 字段 | 作用 |
| --- | --- |
| `startingsize` | 初始块大小 |
| `minimumsize` | 最小块大小 |
| `min_valid_fraction` | 候选块内最小有效像素比例；传给 CSI `initialstate(..., tolerance=...)` |
| `split_std_threshold` | 块内标准差分裂阈值，单位与观测值一致 |
| `split_metric_correction` | std split metric 的修正方式：`std`、`mean`、`median` 或 `bilinear` |
| `split_metric_smoothing` | 可选的 split metric 平滑长度；`null` 表示关闭 |
| `focus_region` | 控制重点区域细分，不删除数据 |
| `high_value_refinement` | 高值区域额外细分 |
| `low_amplitude_cap` | 低振幅区域限制过度细分 |

`std_config.split_metric_correction` 只控制 std-based quadtree 的“是否继续分裂”判据，不控制最终
`<outName>_ifg.txt` 的 cell 值。默认 `std` 保持 CSI 原行为；`bilinear` 会在每个候选块内
先拟合并去掉一个局部平面趋势，再用残差标准差判断是否分裂，适合存在长波坡度但不希望坡度本身
导致过度细分的引导图。最终输出值仍由 `downsample.extraction.value_statistic` 决定。
新数据调参时建议优先尝试 `median`，它更容易解释为“围绕块内中位数评估离散程度”，对局部离群点也更稳健；当前模板仍保留 `std` 是为了让不主动修改配置的旧流程结果保持一致。

`<outName>_downsample_report.yml` 中的 `observation.*.nanstd` 是处理区域内整幅观测的标准差，
可作为设置 `split_std_threshold` 或比较不同场景噪声/形变量级的第一参考，但不应直接机械等同于最终阈值。

`data_config` 使用同一套语义化命名，但底层仍映射到 CSI `dataBased()`：

| 字段 | 作用 |
| --- | --- |
| `startingsize` / `minimumsize` | 初始块大小和最小块大小 |
| `min_valid_fraction` | 候选块内最小有效像素比例；传给 CSI `initialstate(..., tolerance=...)` |
| `split_metric_threshold` | 数据特征分裂阈值；传给 CSI `dataBased(threshold=...)` |
| `split_metric` | 分裂判据类型：`curvature` 或 `gradient`；传给 CSI `dataBased(quantity=...)` |
| `split_metric_smoothing` | 可选的分裂判据平滑长度；传给 CSI `dataBased(smooth=...)` |

`trirb_config.min_valid_fraction` 和 `from_rsp_config.min_valid_fraction` 也是有效像素比例阈值；
它们保留同一含义，但分别作用于三角初始块和复用 `.rsp` cell。

`downsample.plot_decim.coordrange` 只控制降采样检查图显示范围，不控制数据范围。

## `fault_traces` 与 `fault_models`

断层相关配置分成两个入口，避免把“画图叠加”和“降采样计算”混在一起：

| 字段 | 作用 | 是否参与降采样计算 |
| --- | --- | --- |
| `fault_traces` | 读取 lon/lat 文本迹线，叠加到 `-s` raw quick-look 或 `-d` decim 检查图 | 否 |
| `fault_models` | 读取或生成 CSI 断层模型；可用于 `trirb`，也可把 patch edges 叠加到检查图 | 仅当 `use_for` 包含当前方法时 |

常用迹线叠加：

```yaml
fault_traces:
  - enabled: true
    id: surface_trace
    file: Fault_Trace_Menyuan.txt
    stages: [raw, decim]   # raw | decim | all
```

`fault_traces.file` 是至少两列的文本文件，默认按 `lon lat` 读取。若列名或分隔符不同，可加：

```yaml
columns: [lon, lat]
sep: "\\s+"
comment: "#"
```

从迹线生成三角断层模型，供 `trirb` 使用：

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

读取已有 CSI GMT patch 网格，只支持必要的 `csi_gmt` 格式：

```yaml
fault_models:
  - enabled: true
    id: published_mesh
    type: csi_gmt
    geometry: triangular    # triangular | rectangular；trirb 只支持 triangular
    file: fault_mesh.gmt
    readpatchindex: true
    donotreadslip: true
    gmtslip: true           # triangular GMT 常用；rectangular 不用该项
    use_for: []             # 若要用于 trirb，设为 [trirb]
    plot:
      stages: [raw, decim]
      mode: edges           # edges | outline | both
```

要点：

- `std`、`data`、`from_rsp` 不需要断层模型；入门两步走建议先用这些方法跑通。
- `trirb` 必须至少启用一个 `geometry: triangular` 且 `use_for: [trirb]` 的 `fault_models` 条目。
- `fault_traces` 只画线，不会自动生成 `trirb` 所需模型。
- `fault_models.plot: true` 表示 raw 和 decim 两类检查图都叠加；若要精确控制，用 `plot.stages`。
- `csi_gmt` 只作为已构建 CSI patch 网格的轻量入口；不在降采样配置里扩展更多网格格式，避免维护和理解负担。

## 输出文件

下表中的 `<outputName>` 表示实际写文件的前缀：SAR 使用 `outName` 经过 `output_suffix`
解析后的 `<effective_outName>`，optical 直接使用 `optical_config.outName`。

| 文件 | 来源 | 作用 |
| --- | --- | --- |
| `sar_output.txt` | `-s` quick-look | 记录显示统计、色标、飞行方向等 |
| `sar_values.png` | `-s` quick-look | 原始/转换观测值图 |
| `optical_output.txt` | `-s` optical quick-look | 记录 east/north 统计、色标和水平模长 |
| `<outName>_deformation_map.jpg` | `-s` optical quick-look | 原始 optical east/north 形变图 |
| `<outputName>_filter_report.yml` | 启用 `data_filters` | 记录每条过滤规则删除点数 |
| `<outputName>_processing_region_report.yml` | 启用 `processing_region` 且运行 `-c` 或 `-d` | 记录正式处理区域保留/删除点数 |
| `<outputName>_run_metadata.yml` | 每次运行 | 有效配置、执行步骤和预期输出 |
| `<outputName>_downsample_report.yml` | `-d` 且 `downsample.report.enabled: true` | 记录降采样点数、格网、guide-grid、提取规则和质量诊断 |
| `Covariance_estimator.cov` | `-c` | SAR CSI 协方差估计器 |
| `Covariance_estimator_East.cov` / `Covariance_estimator_North.cov` | `-c` | optical east/north CSI 协方差估计器 |
| `<outputName>_ifg.txt` | SAR `-d` | 降采样观测值 |
| `<outputName>_ifg.rsp` | SAR `-d` | 降采样单元几何 |
| `<outputName>_ifg.cov` | SAR `-d` | 降采样协方差矩阵 |
| `<outputName>_decim.png` | SAR `-d` | 降采样结果检查图 |
| `<outName>_East_decim.png` / `<outName>_North_decim.png` | `-d` optical | optical 降采样结果检查图 |

## 常见歧义

| 容易混淆 | 正确理解 |
| --- | --- |
| `data_filters` vs `covar.mask_out` | 前者真实删除点；后者只在协方差估计时排除震源形变区 |
| `data_filters` vs `processing_region` | 前者用于数据质量过滤；后者用于科学关注区域，会影响 `-c/-d` |
| `qc.plot.coordrange` vs `processing_region` | 前者只裁剪 quick-look 视野；后者裁剪正式处理数据 |
| `processing_region` vs `std_config.focus_region` | 前者保留处理区域；后者只控制 std-based 细分层级 |
| `guide_grid` vs `extraction` | 前者只控制 `std/data` 的网格怎么生成；后者控制所有方法最终如何从原始数据提取 cell 值 |
| `std_config.split_metric_correction` vs `extraction.value_statistic` | 前者只影响 std quadtree 分裂判据；后者决定最终输出 cell 观测值 |
| `read.downsample` vs `downsample.method` | 前者是读入抽稀；后者是正式降采样算法 |
| `qc.plot.value_space` vs `data_filters.value_space` | 前者控制画什么；后者控制过滤阈值作用在哪个数值空间 |
| SAR `value_*` vs optical `component_*` | `value_*` 只用于 SAR 单标量 `vel`；`component_*` 只用于 optical `east/north` |
| `factor4plot` vs 真实单位 | `factor4plot` 只影响显示；过滤阈值按读入后的真实观测单位设置 |
| `outName` vs `output_suffix` | `outName` 是基础名；SAR range/azimuth offset 的最终前缀由 `output_suffix` 决定 |
| `-s/-c/-d` | 分别是 quick-look、协方差估计、正式降采样 |
| `prefix` vs 显式文件名 | 二选一；offset/GMTSAR 建议显式文件名 |
| `range_offset` vs `los_displacement` | `range_offset` 是产品 mode；底层观测目标方向仍按 LOS/range 标量处理 |
| `trirb` CUDA 报错 | 多数是 cutde/PyCUDA/nvcc 环境问题；默认 `downsample.compute.cutde_backend: cpp` 更适合跨平台运行 |

## 相关页面

- 跑通流程：[InSAR 降采样](../workflows/02_insar_downsampling.md)
- 手动调参：[InSAR 降采样两步走](../workflows/02a_insar_downsampling_two_step.md)
- 自定义读入和时序网格复用：[自定义读入 Adapter 降采样](../workflows/02b_adapter_downsampling.md)
- Reader 和符号约定：[SAR Reader 参考](sar_reader.md)
- 命令行入口：[CLI 命令参考](cli.md)
