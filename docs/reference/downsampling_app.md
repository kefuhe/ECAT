# 降采样超级入口参考

本页是 `ecat-downsample` 的字段字典和执行逻辑参考。若只是想跑通流程，先读 [InSAR 降采样](../workflows/02_insar_downsampling.md)；若要理解 SAR 观测方向、GMTSAR direct-projection 或 `mode/preset/convention`，读 [SAR Reader 参考](sar_reader.md)。

## 阅读路径

- 入门运行：先看 [InSAR 降采样](../workflows/02_insar_downsampling.md)，按 `-s/-c/-d` 跑通一个标准 reader 案例。
- 数据格式对照：看 [InSAR/Offset 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md)，按 GAMMA、GeoTIFF、GMTSAR 或 adapter 选择模板。
- 字段查阅：回到本页确认 YAML 字段语义、默认值、兼容记录和输出文件。
- 维护扩展：先读 [文档架构说明](../developer/architecture.md) 和 [文档维护规范](../developer/contributing_docs.md)；用户字段语义以本页为准。

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

配置文件使用严格字段集：同一含义只保留当前接口，未知字段会直接报错，并在报错中提示已改名字段。当前模板使用 `min_valid_fraction`、`split_std_threshold` 和 `split_metric_smoothing`；不要再使用旧脚本里的 `tolerance`、`std_threshold` 或 `smooth`。raw quick-look 和 decim 检查图统一由顶层 `check_plots.raw` / `check_plots.decim` 控制。为保护已有案例，`sar_config.qc.plot`、`optical_config.qc.plot` 和 `downsample.plot_decim` 仍会被读入并映射到 `check_plots`，但它们是 deprecated compatibility 字段；新 YAML 和 ECAT-Cases 案例应直接使用 `check_plots`，运行 metadata 会记录这些旧字段的使用情况。协方差读入抽稀使用 `sar_config.read.downsample_for_covar` 或 `optical_config.read.downsample_for_covar`，不要和纯绘图抽稀混用。

## 顶层配置块

| 字段 | 作用 | 常见值 |
| --- | --- | --- |
| `config_version` | 配置语义版本；当前只支持 `1`，旧 YAML 不写时按 `1` 处理 | `1` |
| `data_type` | 选择主数据类型 | `sar`, `optical` |
| `general` | 投影原点和局部坐标设置 | `origin/lon0/lat0` |
| `sar_config` | SAR/InSAR/offset 读入、过滤和 summary 设置 | 见下文 |
| `optical_config` | optical offset 读入、过滤和 summary 设置 | `filename/read/grid` |
| `input_adapter` | 可选自定义读入开关；只在 adapter 模板中使用 | `enabled` |
| `check_plots` | raw quick-look 和 decim 检查图显示/保存设置 | `raw`, `decim` |
| `processing_region` | SAR 或 optical 的协方差和正式降采样处理区域 | `enabled/coord_type/geometry` |
| `covar` | 协方差估计设置 | `mask_out/function/frac/every/distmax` |
| `downsample` | 降采样方法、计算后端和参数 | `compute`, `std`, `data`, `trirb`, `from_rsp` |
| `fault_traces` | 可选断层迹线叠加，只用于 raw/decim 检查图 | lon/lat 文本文件 |
| `fault_models` | 可选断层模型，用于 `trirb` 计算或 GMT 网格叠加 | `generated_from_trace`, `csi_gmt` |

`config_version` 用于固定当前 YAML 语义。当前版本为 `1`；已有旧配置未写该字段时会按 `1` 归一化。运行时写出的 `<outputName>_run_metadata.yml` 会记录 `config_version` 和 `compatibility.deprecated_fields`，方便后续判断案例是否仍依赖旧字段。

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
| quick-look 范围 | `check_plots.raw.coordrange` | `check_plots.raw.coordrange` |
| 正式处理范围 | 顶层 `processing_region` | 顶层 `processing_region` |
| 协方差输出 | `Covariance_estimator.cov` | `Covariance_estimator_East.cov` 和 `Covariance_estimator_North.cov` |
| 降采样结果文件 | `<outputName>_ifg.txt/.rsp/.cov` | `<outputName>_ifg.txt/.rsp/.cov` |
| 降采样检查图 | `<outName>_decim.png` | `<outName>_decim.png`，默认两列显示 east/north |

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
| `qc` | summary 百分位等诊断设置；绘图统一放在顶层 `check_plots` |

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
| `wavelength` | phase 转 LOS disp. 的波长 |
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

`value_space: observation` 表示阈值作用于转换后的 `vel`，即反演实际使用的观测值；它不受 `check_plots.raw.factor4plot` 影响。过滤报告默认写入 `<outName>_filter_report.yml`。

## `optical_config`

`optical_config` 用于 optical offset 产品。它和 SAR 共用 `general`、顶层 `processing_region`、`covar` 和 `downsample`，但读入后保存的是两个水平分量：

```yaml
optical_config:
  outName: Optical_S2_part1
  directory: ..
  filename: Sagaing_S2_Part1.tif
  vel_type: north
  read:
    downsample: 1
    downsample_for_covar: 1
    zero2nan: true
    remove_nan: true
    factor_to_m: 10.0
  grid:
    ew_band: 1
    sn_band: 2
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
```

| 字段 | 作用 |
| --- | --- |
| `outName` | 输出前缀 |
| `directory` / `filename` | optical offset GeoTIFF 所在目录和文件名 |
| `read.downsample` | `-s` quick-look 和 `-d` 降采样读入时的像素步长 |
| `read.downsample_for_covar` | `-c` 协方差估计读入时的像素步长；大 optical 数据可设为 `2/4/8` 试算 |
| `read.zero2nan` | 读 GeoTIFF 后先把 0 值转为 NaN |
| `read.remove_nan` | 构造 CSI `opticorr` 时删除 east 或 north 为 NaN 的像素 |
| `read.factor_to_m` | 产品单位转米的比例 |
| `grid.ew_band` / `grid.sn_band` | 东西向和南北向分量所在 band |
| `vel_type` | `trirb` 使用的 optical 分量，`north` 或 `east` |
| `output_check` | 是否输出降采样检查图 |
| `data_filters` | 真实删除 optical 坏点/粗差的规则；默认关闭 |
| `qc.summary_percentile` | summary 和 metadata 使用的稳健统计中心百分比 |

optical quick-look 和 decim 检查图统一使用顶层 `check_plots`。旧配置里的 `optical_config.qc.plot` 仍会被映射到 `check_plots.raw` 并记录到 metadata，但新配置不要再写旧入口。`check_plots.raw.components: auto` 表示同时画 east/north；`check_plots.decim.components: auto` 表示在一个 `<outName>_decim.png` 中用两列显示 east/north。`check_plots.raw.plot_stride` 只减少绘图点数，不减少读入、summary、协方差或降采样点数；需要真实降密度时使用 `optical_config.read.downsample` 或 `downsample_for_covar`。

optical 的 `data_filters` 与 SAR 在同一时机执行，但规则含义不同：SAR 的 `value_*` 规则作用于单标量 `vel`，optical 的 `component_*` 和 `vector_norm_range` 作用于 `east/north` 双分量。不要把 SAR 的 `projection_norm` 或 `value_space` 用到 optical。

当前标准 optical 入口面向一个 GeoTIFF 中的 EW/SN 两个 band。`read.factor_to_m` 只做单位缩放；`read.downsample*` 会同步抽稀 east、north 和投影坐标轴，并在转换为 lon/lat mesh 后继续使用真实像素坐标绘制，不退化为 `imshow(extent=...)` 的规则经纬度框。`vel_type` 只决定 `trirb` 这类单分量分辨率判据使用 east 还是 north，不会丢弃另一个分量。正式 `-d` 输出仍使用 CSI varres 前缀 `<outputName>_ifg.txt/.rsp/.cov`，其中 `.cov` 由 East 和 North 两个分量块组成；当前运行时不加入 East-North 交叉协方差。

<a id="data-filters-top-level"></a>

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
| `check_plots.raw.coordrange` | 否 | 只控制 `-s` quick-look 显示范围 |
| `processing_region` | 是 | 控制 `-c` 和 `-d` 的实际处理区域 |
| `check_plots.decim.coordrange` | 否 | 只控制降采样检查图显示范围 |

如果开启了 `processing_region`，协方差估计和正式降采样应使用同一个配置重新运行。`covar.mask_out` 仍然只表示在协方差估计中排除震源形变区；它应位于处理区域之内或与处理区域有足够交集。

## `sar_config.qc` 和 `optical_config.qc`

| 字段 | 作用 |
| --- | --- |
| `summary_percentile` | summary、metadata 和默认稳健色标范围使用的中心百分比 |

`sar_output.txt` 记录：

| 字段 | 作用 |
| --- | --- |
| `plot_full_range` | 所有有限显示值的完整极值 |
| `plot_robust_99_range` | 稳健中心范围，用于判断显示量级 |
| `plot_clipped` | 当前 `vmin/vmax` 截掉的有效点比例 |

`optical_output.txt` 记录 `east/north` 的完整范围、稳健中心范围、当前色标裁剪比例和水平模长稳健范围。

## `check_plots`

`check_plots` 是降采样超级入口唯一的绘图配置入口，只控制 raw quick-look 和 decim 检查图，不改变读入、过滤、协方差、降采样或输出 `.txt/.rsp/.cov` 的数值逻辑。

```yaml
check_plots:
  raw:
    show: true
    save_fig: true
    file_path: auto       # SAR: sar_values.png; optical: <outName>_deformation_map.jpg
    coordrange:           # [minlon, maxlon, minlat, maxlat]; only display extent
    plot_stride: 1
    figsize: single       # SAR 常用 single；optical 双列常用 double；高瘦图可试 [4, 5] 或 [7, 5]
    dpi: 300              # 保存图 dpi；屏显交互窗口会内部限制到不超过 200 dpi
    fontsize:             # 空值 = 按 figsize 自动，约 6-10 pt
    factor4plot: auto     # SAR auto=100; optical auto=1
    vmin:
    vmax:
    symmetry: true
    cmap: cmc.roma_r
    axis_tick_direction: out
    colorbar_orientation: auto
    colorbar_pad:
    colorbar_size:
    colorbar_thickness:
    panel_pad:
    colorbar_tick_direction: out
    colorbar_max_major_ticks: 3

  decim:
    show: false
    save_fig: true
    file_path: auto       # <outName>_decim.png
    coordrange:
    cell_style: cells     # cells | points
    figsize: double       # 常规检查图起步值；紧凑 SAR 可用 single，optical 双列可试 [7, 5]
    dpi: 300              # 保存图 dpi；屏显交互窗口会内部限制到不超过 200 dpi
    fontsize:             # 空值 = 按 figsize 自动，约 6-10 pt
    factor4plot: inherit_raw
    vmin:
    vmax:
    symmetry: true
    cmap: cmc.roma_r
    axis_tick_direction: out
    colorbar_orientation: auto
    colorbar_pad:
    colorbar_size:
    colorbar_thickness:
    panel_pad:
    colorbar_tick_direction: out
    colorbar_max_major_ticks: 3
    edgewidth: 0.1
    edgecolor: black
    alpha: 1.0
    markersize: 10
```

`minimal` 模板只写出上面这些高频字段；它们足够完成读入检查、降采样检查图和常规论文图微调。需要更细的双列布局、分量选择、色标位置、字体、次刻度或 trace 样式时，使用 `--template full` 生成包含高级绘图字段的模板，或按下表手动补充字段。

| 字段 | 作用 |
| --- | --- |
| `raw` | `-s` 原始数据 quick-look；输出 summary metadata 后绘图 |
| `decim` | `-d` 结束后读回降采样结果并绘图 |
| `show` | 是否屏显真实 Matplotlib 坐标窗口；`decim.show: true` 可让降采样完成后弹出检查图 |
| `save_fig/file_path` | 是否保存图；`auto` 使用标准文件名 |
| `coordrange` | 只控制显示范围，不裁剪数据 |
| `components` | SAR 固定为 observation；optical 可用 `east`、`north`、`both` 或 `auto` |
| `layout` | `auto` 自动选择单图或双列；也可写 `single`、`columns` |
| `figsize` | 支持 `[width, height]` 或 viztools 注册宽度字符串；常用 `single`、`double`、`full`，也可用期刊预设如 `nature`、`science`、`pnas` |
| `dpi` | 保存图 dpi；屏显交互窗口会内部限制到不超过 200 dpi，避免 `dpi: 600` 导致窗口过大 |
| `fontsize` | 主图基础字号；空值按 `figsize` 宽度自动映射到约 6-10 pt，显式数字则固定 |
| `tickfontsize` | colorbar tick label 字号；空值默认取 `max(fontsize - 1, 6)` |
| `labelfontsize` | colorbar label 字号；空值默认等于 `fontsize` |
| `factor4plot` | 仅显示缩放；`inherit_raw` 只用于 decim |
| `vmin/vmax` | 显式色标范围；optical 支持 `[east, north]` |
| `auto_percentile` | 不写 `vmin/vmax` 时的稳健中心百分比；空值继承对应 `qc.summary_percentile` |
| `symmetry` | 自动色标是否关于 0 对称 |
| `cell_style` | decim 图绘制 cell polygon 或采样中心点 |
| `axis_tick_direction` | 主图经纬度坐标轴刻度线方向，`out`、`in` 或 `inout`；默认 `out`，避免刻度线被形变图遮住 |
| `axis_max_major_ticks` | 主图每个坐标轴最多显示的主刻度数量；默认 `5`，设为空则交给 Matplotlib/viztools 自动决定 |
| `axis_minor_ticks` | 是否启用主图次刻度；默认 `false`，避免高分辨率图中次刻度过密 |
| `axis_minor_subdivisions` | 主图次刻度分段数；仅在 `axis_minor_ticks: true` 时使用 |
| `colorbar_orientation` | `auto`、`vertical` 或 `horizontal`；`auto` 对单图用竖向色标，对 optical 双列图用横向色标 |
| `colorbar_loc` | 色标相对对应主图的位置；空值使用方向默认值，横向外置居中，竖向外置在右侧 |
| `colorbar_pad` | 组内距离：色标与对应主图或 tick label 区域的距离 |
| `colorbar_size` | 色标长边长度，相对对应主图轴尺寸；横向控制宽度，竖向控制高度 |
| `colorbar_thickness` | 色标短边厚度，相对对应主图轴尺寸；横向控制高度，竖向控制宽度 |
| `panel_pad` | 组间最小距离：相邻 map+colorbar panel group 的间距，单位为整张 figure 宽度比例；空值表示自动紧凑且不重叠 |
| `colorbar_tick_direction` | colorbar 刻度线方向，`out`、`in` 或 `inout`；默认 `out`，避免刻度线被色带吞掉 |
| `colorbar_max_major_ticks` | colorbar 最多显示的主刻度数量；默认 `3` |
| `colorbar_minor_ticks` | 是否启用 colorbar 次刻度；默认 `false` |
| `colorbar_minor_subdivisions` | colorbar 次刻度分段数；仅在 `colorbar_minor_ticks: true` 时使用 |

当 `vmin/vmax` 为空时，程序使用中心百分位的稳健范围计算色标；`symmetry: true` 时取正负对称范围。命令行 `--vmin/--vmax` 仍会覆盖当前图的上下限。optical raw 和 decim 默认都在一张图中用两列显示 east/north，各分量有独立 colorbar，避免 east/north 色标范围互相遮蔽；默认 `colorbar_orientation: auto` 会为这种双列图使用横向色标，减少分量标签与另一列地图互相遮挡。主图和 colorbar 默认只显示受 `*_max_major_ticks` 限制的主刻度，次刻度默认关闭；如果论文图需要更细读数，再显式启用 `axis_minor_ticks` 或 `colorbar_minor_ticks`。

典型起步值：SAR 单图用 `figsize: single`，需要更高的经纬度图幅时试 `[4, 5]`；optical east/north 双列图用 `figsize: double`，高瘦图幅或横向 colorbar 较拥挤时试 `[7, 5]` 或 `[8, 5]`。`single/double/full` 会由 viztools 转成出版列宽和默认高宽比；显式 `[width, height]` 的单位是 inch，适合最终微调。`fontsize/tickfontsize/labelfontsize` 留空时，程序按最终 `figsize` 宽度自动给字号：`single` 及以下约 6 pt，`double` 及以上约 10 pt，中间线性过渡。正式论文图如果要求统一字号，可显式写定这些字段。

`show: true` 使用真实 Matplotlib figure，保留坐标读数、缩放、圈选等交互能力。保存图仍使用配置中的 `dpi`，例如 300 或 600；屏显前程序会把交互窗口的 figure dpi 限制到不超过 200，避免高保存 dpi 直接生成超大窗口。由于保存图使用 `bbox_inches="tight"`，屏显窗口和保存图不保证逐像素完全相同，最终排版以保存文件为准。

外置 colorbar 会绑定到对应主图的当前 active box；因此 `equal aspect` 图件在后端重绘后，色标仍会跟随主图重新定位。`colorbar_pad` 只调同一 panel 内主图和色标的距离；多列图中左右 panel group 之间的间距用 `panel_pad` 控制。程序会先把 tick label 和 axis label 保持在 figure canvas 内，`save_fig` 写图时再使用 tight bbox 防止边缘文字被裁切。如果最终排版需要把色标放入图内，可在 `--template full` 中使用：

```yaml
check_plots:
  raw:
    colorbar_mode: inside
    colorbar_orientation: horizontal
    colorbar_loc: lower right
    colorbar_size: 0.35
    colorbar_thickness: 0.04
    colorbar_pad: 0.03
```

图内色标可能遮挡形变场，建议只用于最终排版；常规检查和批处理保持默认外置色标更稳妥。

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
| `from_rsp` | `from_rsp_config` | 复用已有 CSI `.rsp` 格网；支持 10 列 legacy 矩形、18 列 full-corner 矩形和 8 列正式三角形 |

通用提取统计由 `downsample.extraction` 控制，对 `std`、`data`、`trirb` 和
`from_rsp` 都生效。新模板默认块内观测取中位数，误差取标准差，坐标取块内像素均值，
InSAR 投影向量取均值后归一化；需要复现 CSI 旧行为时，把 `value_statistic` 显式改为 `mean`。

```yaml
downsample:
  extraction:
    value_statistic: median      # mean | median | center_nearest | trimmed_mean
    error_statistic: std         # std | mad | sem | none
    coordinate_statistic: mean   # mean | block_center | center_nearest
    projection_statistic: mean_normalized # mean_normalized | center_nearest
    trim_fraction: 0.1           # value_statistic: trimmed_mean 时使用
```

`value_statistic` 控制 `<outName>_ifg.txt` 中的降采样观测值。默认 `median`
更抗离群点；`trimmed_mean` 也是稳健选项；`center_nearest` 使用最靠近 cell 中心的原始像素。
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
`<outName>_ifg.txt` 的 cell 值。新模板默认 `median`，需要复现 CSI 原行为时可显式设置为 `std`；`bilinear` 会在每个候选块内
先拟合并去掉一个局部平面趋势，再用残差标准差判断是否分裂，适合存在长波坡度但不希望坡度本身
导致过度细分的引导图。最终输出值仍由 `downsample.extraction.value_statistic` 决定。
`median` 更容易解释为“围绕块内中位数评估离散程度”，对局部离群点也更稳健。

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
它们保留同一含义，但分别作用于三角初始块和复用 `.rsp` cell。`from_rsp` 读取 `.rsp` 中的
lon/lat 顶点并投影到当前数据对象的局部坐标；10 列 legacy 矩形只有左上和右下角，18 列
full-corner 矩形保存四个真实角点，8 列是 `trirb` 和三角 `.rsp` 复用的正式三角 cell。

降采样检查图由 `check_plots.decim` 控制；`coordrange` 只裁剪显示视野，不控制数据范围。`vmin/vmax`
为空时会使用稳健自动色标，`factor4plot: inherit_raw` 会继承 raw quick-look 的显示比例。optical 可写
二元列表 `[east, north]` 分别控制双列图中两个分量的色标；命令行 `--vmin/--vmax` 仍会用同一个标量覆盖两个分量。

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
| `<outputName>_run_metadata.yml` | 每次运行 | 有效配置、配置版本、deprecated compatibility 字段、执行步骤和预期输出 |
| `<outputName>_downsample_report.yml` | `-d` 且 `downsample.report.enabled: true` | 记录降采样点数、格网、guide-grid、提取规则和质量诊断 |
| `Covariance_estimator.cov` | `-c` | SAR CSI 协方差估计器 |
| `Covariance_estimator_East.cov` / `Covariance_estimator_North.cov` | `-c` | optical east/north CSI 协方差估计器 |
| `<outputName>_ifg.txt` | SAR/optical `-d` | 降采样观测值；SAR 为单标量，optical 为 east/north 双分量 |
| `<outputName>_ifg.rsp` | SAR/optical `-d` | 降采样单元几何；矩形输出默认 18 列 full-corner，三角输出为 8 列 |
| `<outputName>_ifg.cov` | SAR/optical `-d` | 降采样协方差矩阵；optical 为 East/North 分量块矩阵 |
| `<outputName>_decim.png` | SAR/optical `-d` | 降采样结果检查图；optical 默认双列显示 east/north |

## 常见歧义

| 容易混淆 | 正确理解 |
| --- | --- |
| `data_filters` vs `covar.mask_out` | 前者真实删除点；后者只在协方差估计时排除震源形变区 |
| `data_filters` vs `processing_region` | 前者用于数据质量过滤；后者用于科学关注区域，会影响 `-c/-d` |
| `check_plots.*.coordrange` vs `processing_region` | 前者只裁剪图件视野；后者裁剪正式处理数据 |
| `processing_region` vs `std_config.focus_region` | 前者保留处理区域；后者只控制 std-based 细分层级 |
| `guide_grid` vs `extraction` | 前者只控制 `std/data` 的网格怎么生成；后者控制所有方法最终如何从原始数据提取 cell 值 |
| `std_config.split_metric_correction` vs `extraction.value_statistic` | 前者只影响 std quadtree 分裂判据；后者决定最终输出 cell 观测值 |
| `read.downsample` vs `downsample.method` | 前者是读入抽稀；后者是正式降采样算法 |
| `check_plots.raw.value_space` vs `data_filters.value_space` | 前者控制 raw SAR 图画什么；后者控制过滤阈值作用在哪个数值空间 |
| `sar_config.qc.plot` / `optical_config.qc.plot` / `downsample.plot_decim` vs `check_plots` | 前三者只作为旧配置兼容入口读取并记录；当前推荐入口是顶层 `check_plots.raw/decim` |
| SAR `value_*` vs optical `component_*` | `value_*` 只用于 SAR 单标量 `vel`；`component_*` 只用于 optical `east/north` |
| `factor4plot` vs 真实单位 | `factor4plot` 只影响显示；过滤阈值按读入后的真实观测单位设置 |
| `outName` vs `output_suffix` | `outName` 是基础名；SAR range/azimuth offset 的最终前缀由 `output_suffix` 决定 |
| `-s/-c/-d` | 分别是 quick-look、协方差估计、正式降采样 |
| `prefix` vs 显式文件名 | 二选一；offset/GMTSAR 建议显式文件名 |
| `range_offset` vs `los_displacement` | `range_offset` 是产品 mode；底层观测目标方向仍按 LOS/range 标量处理 |
| `trirb` CUDA 报错 | 多数是 cutde/PyCUDA/nvcc 环境问题；默认 `downsample.compute.cutde_backend: cpp` 更适合跨平台运行 |

## 相关页面

- 跑通流程：[InSAR 降采样](../workflows/02_insar_downsampling.md)
- 手动调参：[InSAR 降采样 Step1/Step2 调参](../workflows/02a_insar_downsampling_two_step.md)
- 自定义读入和时序网格复用：[自定义读入 Adapter 降采样](../workflows/02b_adapter_downsampling.md)
- Reader 和符号约定：[SAR Reader 参考](sar_reader.md)
- 命令行入口：[CLI 命令参考](cli.md)
- 图件样式和出版尺寸：[ECAT 图件样式参考 / Viztools](viztools.md)
