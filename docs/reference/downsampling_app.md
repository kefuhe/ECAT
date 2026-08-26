# 降采样超级入口参考

本页是 `ecat-downsample` 的字段字典和执行逻辑参考。若只是想跑通流程，先读 [InSAR 降采样](../workflows/02_insar_downsampling.md)；若要理解 SAR 观测方向、左右视或 GMTSAR direct projection，读 [SAR Reader 参考](sar_reader.md)。

## 阅读路径

- 入门运行：先看 [InSAR 降采样](../workflows/02_insar_downsampling.md)，按 `-s/-c/-d` 跑通一个标准 reader 案例。
- 数据格式对照：看 [InSAR/Offset 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md)，按 GAMMA、GeoTIFF、GMTSAR 或 adapter 选择模板。
- 字段查阅：回到本页确认 YAML 字段语义、默认值、兼容记录和输出文件。
- 维护扩展：先读 [文档架构说明](../developer/architecture.md) 和 [文档维护规范](../developer/contributing_docs.md)；用户字段语义以本页为准。

## 入口定位

降采样超级入口负责把原始 SAR、offset 或 optical 产品整理为 CSI 风格的反演输入：

```text
raw product
  -> reader
  -> optional data filters
  -> optional observation correction
  -> optional processing region
  -> covariance/downsampling
  -> <effective_outName>_ifg.txt/.rsp/.cov
```

常用命令：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode range_offset -o downsample.yml
ecat-downsample -f downsample.yml
ecat-downsample -f downsample.yml -s
ecat-downsample -f downsample.yml -c
ecat-downsample -f downsample.yml -d
ecat-downsample -f downsample.yml --edit-trace
```

### 四种处理模式与可选编辑入口

| 调用 | 有效步骤 |
| --- | --- |
| 不加 `-s/-c/-d` | 使用 YAML 的 `covar.do_covar` 和 `downsample.enabled`；生成模板两者均为 `false`，所以通常只执行已启用的改正与导出 |
| `-s` | quick-look，并强制关闭本次 covariance/downsample |
| `-c` | 强制启用 covariance；downsample 是否同时运行仍由 `downsample.enabled` 决定 |
| `-d` | 强制启用 downsample；covariance 是否同时运行仍由 `covar.do_covar` 决定 |
| `--edit-trace` | 打开可选迹线编辑器，并强制关闭本次 covariance/downsample；不属于四种数值处理模式 |

`-c -d` 可以组合。标准 `observation_grid` 在启用后随任一模式写出；集成
`google_earth` 只在三个有效步骤全部为 `false` 时写出，避免 quick-look、协方差调参
或降采样时反复覆盖 KMZ。

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
print_input_summary()  # reader/filter summary before reference correction
build ObservationGrid only when correction/export is enabled
apply phase_cycle_correction()  # optional; user-declared integer cycles
apply observation_correction()  # optional; keeps the original observation
export full-resolution observation grid  # optional; no resampling
if effective edit_trace: open the optional editor, skip covariance/downsampling
if effective show_raw_data: quick-look plot
if effective do_covar or do_downsample:
  build_processing_image()
  apply_processing_region()
  if effective do_covar:
    create CSI imagecovariance from processing data
    apply covar.mask_out to exclude deformation-source area
    sample background pixels and fit exp/gauss covariance model
    write Covariance_estimator*.cov
  if effective do_downsample and downsample.guide_grid.enabled and method in [std, data]:
    build filtered guide image from the processing data
    construct std/data grid on the guide image
    restore the unfiltered processing data
    extract final cell values by downsample.extraction
  if effective do_downsample without guide_grid:
    construct grid and extract final cell values from processing data
write run metadata
```

`data_filters` 会真实删除读入后的坏点或粗差点；SAR 使用 `sar_config.data_filters`，optical 使用 `optical_config.data_filters`。少见的 `phase_cycle_correction` 只给用户明确指定的解缠相位区域增加整数周对应的 LOS delta；随后 `observation_correction` 在过滤后、正式处理区域之前估计并应用 offset/plane。原观测仍保留，启用任一改正时 quick-look、协方差和降采样使用最终改正后观测。`processing_region` 只在协方差和正式降采样前保留科学关注区域；`-s` quick-look 不受它影响，应使用数据类型对应的 quick-look 绘图设置控制显示范围。`guide_grid` 在 `processing_region` 之后生效，只影响 `std/data` 的网格生成，不改变最终取值来源。`covar.mask_out` 沿用 CSI 的 `maskOut()` 语义，只在协方差估计阶段排除震源形变区，不改变最终降采样数据。

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
| `phase_cycle_correction` | 高级：给已确认的不连通解缠分量移除整数周 | `corrections[].cycles_to_remove/selector` |
| `observation_correction` | 过滤后、降采样前的 offset/一阶 plane 参考改正 | `model/coefficient_mode/fit` |
| `export` | 无重采样导出标准网格或全分辨率 Google Earth 显示副本 | `observation_grid`, `google_earth` |
| `processing_region` | SAR 或 optical 的协方差和正式降采样处理区域 | `enabled/coord_type/geometry` |
| `covar` | 协方差估计设置 | `mask_out/function/frac/every/distmax` |
| `downsample` | 降采样方法、计算后端和参数 | `compute`, `std`, `data`, `trirb`, `from_rsp` |
| `fault_traces` | raw/decim 检查图叠加；`--edit-trace` 时 raw-stage 项作为只读参考 | TXT/DAT/GMT 或 GeoJSON 折线 |
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
| 参考改正 | `observation` 单分量系数 | east/north 共用区域、分别估计系数 |
| 标准导出 | `observation/corrected_observation` | `east/north/corrected_east/corrected_north` |
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
| `mode` | `unwrapped_phase`, `los_displacement`, `range_offset`, `azimuth_offset` |
| `acquisition_look_side` | `right` 或 `left`；表示地面条带位于平台航向哪一侧 |
| `geometry_convention` | angle reader 的高级角度协议；普通产品省略 |
| `projection_convention` | GMTSAR direct-projection reader 的高级投影协议；普通产品省略 |
| `directory` | 数据文件所在目录 |
| `files` | value、角度或 ENU projection 文件 |
| `read` | 读入抽稀、单位缩放和波长 |
| `grid` | raster/grid 技术读入细节，例如 band、变量名、engine 和坐标名 |
| `data_filters` | 真实删除数据点的过滤规则；默认关闭 |
| `qc` | summary 百分位等诊断设置；绘图统一放在顶层 `check_plots` |

`reader/mode`、左右视和高级协议的语义见 [SAR Reader 参考](sar_reader.md)。

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
| `byte_order` | 仅 GAMMA 二进制：`native`, `little`, `big`；`native` 保持历史默认 |

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

## `phase_cycle_correction`、`observation_correction` 与 `export`

这些顶层块位于 reader 配置之外。`phase_cycle_correction` 仅用于
`sar_config.mode: unwrapped_phase`，而 `observation_correction` 和 `export`
服务 SAR 与 optical：

```yaml
observation_correction:
  enabled: false
  model: offset
  coefficient_mode: estimate
  fit:
    coord_type: lonlat
    regions: []
    exclude_regions: []

export:
  observation_grid:
    enabled: false
    format: netcdf
    file: auto
    geotiff_sidecar: auto
    verify: true
  google_earth:
    enabled: false
    file: auto
    style:
      cmap: RdBu_r
      display_factor: 100.0
      display_unit: cm
      vmin:               # 显示单位中的色标下限；不是经纬度范围
      vmax:               # vmin/vmax 同时设置或同时留空
      symmetry: true      # 只作用于自动范围；显式 vmin/vmax 优先
```

`phase_cycle_correction` 不出现在默认模板；完整配置见下方链接。
`observation_correction` 只支持 `offset/plane` 和 `estimate/fixed`。
生成模板保持 `enabled: false` 且不填虚构参考坐标。`offset + estimate` 必须在
`fit.regions` 中填写稳定零参考区；`plane + estimate` 的空 `regions` 表示使用全部有效
观测，通常再用 `exclude_regions` 排除形变或噪声 box。命令行 `-c` 只请求协方差估计，
不会自动启用观测改正，也不会复用 `covar.mask_out`。
`export.observation_grid` 当前只支持规范化的 CF-NetCDF；具有可靠 affine/CRS
的 TIFF reader 可自动附加同网格 GeoTIFF。导出不插值、不重投影，也不把二维
经纬度压成一维轴。显式文件可使用 `.nc/.h5/.hdf5`；HDF5 扩展名要求
`netCDF4` 或 `h5netcdf` 引擎。完整周跳公式、circle/box/polygon、固定 plane
原点、输出变量和 PyGMT 限制见
[观测参考改正与无重采样网格导出](observation_correction_export.md)。

`export.google_earth` 复用同一个 reader 和改正结果，只导出全分辨率观测，不自动
加入 CSI `.txt/.rsp`。启用后使用无阶段选项命令
`ecat-downsample -f downsample.yml`；`-s/-c/-d` 不触发 KMZ。自动变量选择、
mask、`vmin/vmax/symmetry` 样式和严格坐标限制见
[Google Earth Export Reference](google_earth_export.md#downsample-integration)。

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

经度区域按 360° 周期匹配。西经 `-118°` 与 `242°` 等价，配置无需迁就 CSI
对象内部使用的经度分支；程序不会因此改写 CSI 经度、局部 `x/y` 或输出文件。
跨日界线 box 使用 `[179, 181, minlat, maxlat]` 这类连续展开写法。完整规则、
polygon 行为和诊断字段见 [经度约定与区域配置](longitude_regions.md)。

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
    contours:
      enabled: false      # 仅 structured raw 2-D 网格；默认关闭
      levels: auto        # auto | 等值线条数 | 显式显示单位数值列表
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
| `contours` | 仅 raw structured 2-D 网格的等值线诊断层；不修改数据、不用于 decim |
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

`coordrange` 的经度同样按 360° 周期匹配。程序会把观测中心、cell corners 和断层
叠加线临时显示到 `coordrange` 所在分支，因此负经度范围可以直接显示 CSI 中保存为
`0–360°` 的降采样结果；数据和输出坐标不被改写。

当 `vmin/vmax` 为空时，程序使用中心百分位的稳健范围计算色标；`symmetry: true` 时取正负对称范围。命令行 `--vmin/--vmax` 仍会覆盖当前图的上下限。optical raw 和 decim 默认都在一张图中用两列显示 east/north，各分量有独立 colorbar，避免 east/north 色标范围互相遮蔽；默认 `colorbar_orientation: auto` 会为这种双列图使用横向色标，减少分量标签与另一列地图互相遮挡。主图和 colorbar 默认只显示受 `*_max_major_ticks` 限制的主刻度，次刻度默认关闭；如果论文图需要更细读数，再显式启用 `axis_minor_ticks` 或 `colorbar_minor_ticks`。

### Raw 等值线诊断层

`check_plots.raw.contours` 在同一幅彩色 raw quick-look 上叠加等值线，适合辅助辨认闭合形变瓣、
空间梯度和相干结构。它消费的正是当前 raw 图已经完成观测转换、改正、有效像元屏蔽、
`plot_stride` 和 `factor4plot` 后的二维显示数据，不会写回观测，也不会进入协方差或降采样。

最短配置：

```yaml
check_plots:
  raw:
    contours:
      enabled: true
      levels: auto
```

`levels` 的含义：

- `auto`：在当前 `coordrange` 内按 `auto_percentile` 和 `symmetry` 的稳健数据范围生成 7 条内部等值线；
- 正整数，例如 `9`：按同一稳健数据范围生成指定数量的等值线；
- 递增列表，例如 `[-5, -2, 0, 2, 5]`：使用显式等值线值，单位是 `factor4plot` 后的显示单位。

自动等值线范围不读取显式 `vmin/vmax`，因此即使用户为彩色背景设置了较窄色标，等值线仍可
提供独立的稳健结构提示。需要在最终图中显示数值或调整样式时使用：

```yaml
check_plots:
  raw:
    contours:
      enabled: true
      levels: [-5, -2, 0, 2, 5]
      labels: true
      color: "0.20"
      linewidth: 0.5
      alpha: 0.8
```

SAR 等值线表示当前 `value_space` 下的标量观测；默认是转换后的 observation，不是三维位移。
optical east/north 双列图分别从各自分量生成自动 levels；显式列表则共同用于所选分量。
等值线与彩色背景使用同一个 `plot_stride`，不会另行平滑或插值。输入不是至少 `2 x 2` 的
structured 二维网格时会明确报错；不支持给散点或 decim cells 人为插值出连续等值线。
破碎或密集的线条可能来自真实梯度，也可能来自噪声、无效区或解缠问题，需要结合原图和
数据质量诊断判断。

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
| `frac` | CSI `imagecovariance` 的抽样规模：正整数为固定点数，`(0, 1]` 浮点数为剩余背景点的抽样比例 |
| `every` | 经验协方差距离分箱间隔，单位为 CSI 局部 `x/y` 坐标单位，通常是 km |
| `distmax` | 参与协方差拟合的最大距离，单位同 `every` |
| `rampEst` | 调用 CSI 估计协方差前是否估计/移除 ramp |

`mask_out` 可以是一个框，也可以是多个框。它用于背景噪声估计，不应理解为坏点删除或降采样范围控制。

生成模板默认使用 `frac: 5000`，即从掩膜后的剩余背景点中随机抽取最多 5000 点；若
可用点不足 5000，CSI 使用全部可用点。比例模式应使用带小数点的 `(0, 1]` 浮点数，例如
`frac: 0.002` 表示抽取 0.2%。类型本身参与语义判断：`frac: 1` 表示 1 个点，
`frac: 1.0` 表示全部点，因此固定点数不能写成 `5000.0`。经验协方差会构造采样点对，
时间和临时内存开销随采样点数近似按平方增长，不宜盲目增大整数点数。

调用 CSI `maskOut()` 前，eqtools 会把每个经度区间解析到当前处理数据的数值分支。
例如负经度 box 可以正确掩膜 CSI 中保存为 `0–360°` 的点；跨日界线时可能解析为
两个数值 box。解析只作用于本次协方差调用，原始 YAML 保持不变，并记录在运行
metadata 的 `_runtime.resolved_covariance_mask_out` 中。

运行 `ecat-downsample -c` 时，eqtools 先按当前配置读入数据、执行 `data_filters`，再在 `processing_region` 内构造轻量 CSI 处理对象。之后程序创建 CSI `imagecovariance`，用 `mask_out` 排除主形变源区，并在剩余背景点上按 `frac/every/distmax/rampEst/function` 拟合经验协方差模型。SAR 写 `Covariance_estimator.cov`；optical offset 分别写 `Covariance_estimator_East.cov` 和 `Covariance_estimator_North.cov`。

`Covariance_estimator*.cov` 是协方差估计器文件，不是最终反演矩阵。运行 `-d` 时，若当前目录存在对应估计器，eqtools 会把降采样后的 CSI 对象交给 CSI `buildCovarianceMatrix()`，写出 `<effective_outName>_ifg.cov`。若没有估计器，则按 `missing_policy` 读取已有矩阵、写单位阵或报错。

CSI 当前 `imagecovariance` 最终矩阵采用相关项

\[
C_{ij}=\sigma^2\exp(-d_{ij}/\lambda)
\]

（`gauss` 使用对应高斯距离函数），没有额外的显式 nugget
\(\tau^2\delta_{ij}\)。拟合器输出中的 `Sill` 用于经验曲线处理，但
`buildCovarianceMatrix()` 只把拟合的 `Sigma`、`Lambda` 和函数类型传给矩阵构造。因此，长
相关长度且采样点间距较小时，矩阵可能高度相关；应结合 Cholesky、条件数、白化残差和不同
协方差假设的敏感性判断，而不能把 `Sill` 当作已经加入对角线的 nugget。

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

<a id="guide-grid"></a>

`downsample.guide_grid` 是可选的 Guided Quadtree Downsampling（引导式四叉树降采样）入口，
也可理解为 Two-stage Quadtree（两阶段四叉树）。它只允许用于 `method: std` 和 `method: data`。
启用后，程序先执行 quadtree partitioning based on filtered/smoothed interferograms
（基于滤波/平滑干涉图的四叉树划分），再切回原始未滤波数据，按 `downsample.extraction`
提取最终值。它不删除数据，也不改变协方差输入；若要真实剔除粗差，仍使用 `data_filters`。

`component: auto` 在 SAR 中等价于 `observation`，在 optical 中等价于 `both`。
optical 的 `both` 会分别过滤 east/north，再由 CSI 以两个分量的联合离散程度决定是否细分；
这是推荐入口，因为只过滤 `magnitude` 可能漏掉“水平位移幅值近似不变、方向发生变化”的边界。
`magnitude`、`east` 和 `north` 保留为进阶诊断选项，使用时应明确接受只让所选量控制网格。

```yaml
downsample:
  guide_grid:
    enabled: false
    source: filtered_observation
    component: auto        # SAR: observation；optical: both
    filter:
      kind: gaussian
      sigma: null          # enabled: true 时必须设置
      unit: km             # km | pixel
      radius_sigma: 3.0
```

`filter.kind` 目前只提供两个职责清楚的选项：

| `kind` | 主要适用问题 | 参数 | 不适合替代的处理 |
| --- | --- | --- | --- |
| `gaussian` | 连续、局部的高频随机噪声使 quadtree 过度细分 | `sigma`、`unit`、`radius_sigma` | 连续大气/电离层条带、解缠边界、轨道坡度 |
| `median` | SAR offset 或 optical offset 中少量孤立错配、小尺度亮暗斑点 | `window_size`，默认 `3`，必须是大于等于 3 的奇数像素窗口 | 长距离相干异常带、有效数据边界、需要真正删除的坏点 |

最短 median 配置只需：

```yaml
downsample:
  guide_grid:
    enabled: true
    filter:
      kind: median          # 默认使用 3 x 3 像素窗口
```

若需显式调整窗口：

```yaml
downsample:
  guide_grid:
    enabled: true
    filter:
      kind: median
      window_size: 5
```

Gaussian 和 median 都保留输入的原始 NaN 掩膜，并只改变网格生成阶段的引导值。
median 仅对可识别的完整或稀疏规则 lon/lat 栅格开放，不为散点输入猜测“像素邻域”。
Gaussian 对这类规则栅格使用分块 NaN-aware 二维计算；稀疏栅格只重建 value 网格，
不再同时重建完整 `x/y` 网格。小型非规则点集仍可使用散点邻域实现。若大数据无法识别为规则栅格，
程序会明确停止并提示保留规则 lon/lat 网格拓扑或在上游滤波，不会静默进入耗时的逐点散点循环。

生成模板同时列出两种 `kind`。把 `kind` 改为 `median` 时，未删除的 Gaussian 默认字段
不会进入运行时 filter；`window_size` 未写时使用 3。运行报告中的 `guide_grid.filter`
记录实际采用的、按 `kind` 归一化后的字段；`components.*` 还记录 backend、grid layout、
grid shape 和有效点覆盖率。未知或拼错的 filter 字段会在运行前报错。

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

`focus_region` 启用时，`box`、`polygon` 和 `polygon_file` 必须且只能设置一个。
最常用的矩形写法是：

```yaml
focus_region:
  enabled: true
  coord_type: lonlat
  box: [94.5, 97.0, 20.0, 23.0]  # [minlon, maxlon, minlat, maxlat]
  max_splits_outside: 5
```

`focus_region.box` 当前只支持经纬度顺序
`[minlon, maxlon, minlat, maxlat]`，且两个最小值必须分别小于最大值。它不会删除
box 外的数据；CSI 根据降采样块中心是否位于该区域内，限制区域外的最大细分层级。
需要真正移除区域外数据时使用顶层 `processing_region.box`。

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

`trirb_config` 的参数对本次参与计算的全部断层模型统一生效：

| 字段 | 作用 |
| --- | --- |
| `minimumsize` | 三角初始网格允许的最小尺寸 |
| `min_valid_fraction` | 三角候选块内最小有效像素比例；传给 CSI `initialstate(..., tolerance=...)` |
| `max_samples` | 目标最大降采样观测数 |
| `change_threshold` | 相邻迭代样本数变化的停止阈值，单位为百分比 |
| `smooth_factor` | 分辨率矩阵中断层 Laplacian 平滑项的统一权重 |
| `slipdirection` | 构建 Green 函数的分量组合：`s`、`d`、`t` 或其组合；对全部活动断层统一生效 |
| `plot` | 是否显示 CSI 内部迭代图；不同于顶层检查图叠加设置 |
| `decimorig` | 仅在 `plot: true` 时使用的内部绘图抽稀 |
| `verboseLevel` | CSI 迭代输出级别 |

`trirb_config.min_valid_fraction` 和 `from_rsp_config.min_valid_fraction` 都是有效像素比例阈值；
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
| `fault_traces` | 读取单段或多段 lon/lat 折线，叠加到 `-s` raw quick-look 或 `-d` decim 检查图 | 否 |
| `fault_models` | 读取或生成 CSI 断层模型；可用于 `trirb`，也可把 patch edges 叠加到检查图 | 仅当 `use_for` 包含当前方法时 |

`use_for` 是显式的计算角色列表，不由 `enabled` 或 `plot.stages` 推断。生成
`--downsample-method trirb` 模板时，三角模型示例预设为 `use_for: [trirb]`；其他方法的
模板保持 `use_for: []`。所有示例默认仍是 `enabled: false`，因此预设角色本身不会触发
模型读取或计算。

两类条目都使用列表形式。`type` 表示模型从哪里来，`geometry` 表示 patch 形状，
两者不是同一个维度：

| 字段 | 可选值 | 含义 |
| --- | --- | --- |
| `type` | `generated_from_trace`、`csi_gmt` | 从迹线构建模型，或读取已有 CSI GMT patch 文件 |
| `geometry` | `triangular`、`rectangular` | 模型的 patch 形状；`trirb` 只接受 `triangular` |

常用迹线叠加：

```yaml
fault_traces:
  - enabled: true
    id: surface_trace
    file: Fault_Trace_Menyuan.gmt
    stages: [raw, decim]   # raw | decim | all
    marker: null           # null=只画线；改为 "x" 可标出输入迹线点
    markersize: 3.0        # marker 非 null 时生效，单位 pt
```

`marker` 省略或设为 YAML `null` 时保持原有纯线条绘图；临时改为 `"x"` 会在每个输入迹线点上
叠加小叉号，适合对照检查并调整迹线节点。符号继承当前 raw/decim 图的 `trace_color`，
`markersize` 只控制符号大小，不参与读取、断层构建或降采样计算，也不会作用到
`fault_models` 的 patch edges。

`fault_traces.file` 支持以下轻量折线输入：

- TXT/DAT/TRACE：每个数据行至少含经度和纬度两列；
- GMT/OGR 风格文本：`>` 开始新段，`#` 元数据行和注释忽略；
- GeoJSON：`LineString`、`MultiLineString`、Feature 或 FeatureCollection。

文本默认取前两列作为 `lon lat`。第三列及后续列可以存在，也可以逐行缺省；二维地表迹线只消费
经纬度列，其余列明确忽略，不会把 `lon lat z` 错读成 `lat z`。若列顺序、分隔符或注释符不同，
可加：

```yaml
columns: [lat, lon, z]   # 本例表示第 1 列 lat、第 2 列 lon；z 不参与二维迹线
sep: "\\s+"
comment: "#"
```

显示语义默认最省配置：单段照常显示；多段文件的每个 `>` 段分别成为一条 overlay/reference，
不会跨段连线。显式 `id: surface_trace` 的多段显示标识依次为 `surface_trace.1`、
`surface_trace.2` 等。只想显示部分段时使用零基索引：

```yaml
segments: [0, 2]   # 也可写一个整数；省略或写 all 表示全部段
```

`segments` 只属于 `fault_traces` 的显示选择，不裁剪、不重排源文件，也不参与降采样计算。

从迹线生成三角断层模型，供 `trirb` 使用：

```yaml
fault_models:
  - enabled: true
    id: generated_triangular_fault
    type: generated_from_trace
    geometry: triangular
    trace_file: Fault_Trace_Menyuan.txt
    # segment: 0             # 多段 GMT/GeoJSON 时必填；零基索引
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

`generated_from_trace` 必须得到一条连续迹线：单段文件无需 `segment`；多段输入必须显式写
`segment: INDEX`，否则在构建模型前报错，不会把不连续的多段静默拼接。这里使用单数
`segment`，与显示入口可选择多段的 `segments` 有意区分。

读取已有 CSI GMT patch 网格，只支持必要的 `csi_gmt` 格式：

```yaml
fault_models:
  - enabled: true
    id: published_mesh
    type: csi_gmt
    geometry: triangular    # triangular | rectangular；trirb 只支持 triangular
    file: fault_mesh.gmt
    readpatchindex: true     # 段头含 3 个拓扑索引
    donotreadslip: true      # 只读几何并忽略滑动量
    gmtslip: true            # 仅 triangular；段头含 -Z... 时为 true
    use_for: [trirb]        # 参与 TriRB；仅绘图时改为 []
    plot:
      stages: [raw, decim]
      mode: edges           # edges | trace | both
```

CSI GMT 读取字段必须与文件段头一致：

| 字段 | 何时设为 `true` | 适用范围 |
| --- | --- | --- |
| `readpatchindex` | 每个 `>` 段头包含三个 patch/拓扑索引 | 三角和矩形 CSI GMT |
| `donotreadslip` | 只需要几何，或希望忽略段头中的滑动量；`trirb` 通常保持 `true` | 三角和矩形 CSI GMT |
| `gmtslip` | 三角 GMT 段头包含 `-Z...` token；它会改变后续索引/滑动列的解析位置 | 仅三角 CSI GMT；矩形条目必须省略 |
| `increasingy` | 希望 CSI 按递增 y 方向整理矩形 patch 角点 | 仅矩形 CSI GMT；默认 `true` |

三角 GMT 没有 `-Z...` 时写 `gmtslip: false`；没有三个拓扑索引时写
`readpatchindex: false`。只想叠加三角模型而不参与计算时写 `use_for: []`；矩形模型不参与
`trirb`，也必须保持 `use_for: []`，但两者仍可通过 `plot.stages` 叠加到 raw/decim 检查图。

多断层或混合来源不需要额外的 `group`/`role` 层。将条目依次放入同一个列表：

```yaml
fault_models:
  - enabled: true
    id: trace_built_fault
    type: generated_from_trace
    geometry: triangular
    trace_file: fault_a_trace.txt
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

  - enabled: true
    id: gmt_fault
    type: csi_gmt
    geometry: triangular
    file: fault_b_mesh.gmt
    readpatchindex: true
    donotreadslip: true
    gmtslip: true
    use_for: [trirb]
    plot:
      stages: [raw, decim]
      mode: edges
```

活动计算条目的精确条件是：`enabled: true`、`geometry: triangular`，并且
`use_for` 包含当前的 `trirb` 方法。多个活动模型按列表顺序共同传给同一个 CSI
downsampler；底层构造 `G_total = [G_1 G_2 ...]`，并以
`D_total = block_diag(D_1, D_2, ...)` 组合各断层平滑矩阵。因此它是一次联合的
分辨率计算，不是逐断层依次重新降采样。`trirb_config.slipdirection` 和
`smooth_factor` 对这些活动模型统一生效。

计算和绘图是两条独立选择路径：

| `enabled` | `use_for` | `plot.stages` | 实际作用 |
| --- | --- | --- | --- |
| `false` | 任意 | 任意 | 不读取、不计算、不绘图 |
| `true` | `[trirb]` | `[]` 或未设置 | 只参与 `trirb` 计算 |
| `true` | `[]` | `[raw]`、`[decim]` 或两者 | 只叠加到指定检查图 |
| `true` | `[trirb]` | `[raw, decim]` | 同时参与计算和两类检查图 |

要点：

- `std`、`data`、`from_rsp` 不需要断层模型；入门两步走建议先用这些方法跑通。
- `trirb` 必须至少启用一个 `geometry: triangular` 且 `use_for: [trirb]` 的 `fault_models` 条目。
- 实际请求降采样时，TriRB 角色检查发生在读取观测数据之前；若已有启用的三角模型但
  `use_for` 为空，错误会列出相应模型 ID，并提示补充 `use_for: [trirb]`。
- `fault_traces` 只用于绘图，不会自动生成 `trirb` 所需模型；`marker` 只标记所选迹线段的输入节点。
- `fault_models.plot: true` 表示 raw 和 decim 两类检查图都叠加；若要精确控制，用 `plot.stages`。
- `fault_traces` 的绘图阶段直接写在 `stages`；`fault_models` 必须写在 `plot.stages`。配置校验会拒绝放错层级的字段。
- 同一 `fault_traces` 或 `fault_models` 列表中的显式 `id` 必须唯一，便于报告、绘图和排错。
- 普通多段 GMT/OGR 折线属于 `fault_traces` 或 `generated_from_trace.trace_file`；CSI patch GMT 属于
  `fault_models.type: csi_gmt`。两者虽然都使用 `>`，几何语义和 reader 不同，不互相猜测。
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
| `<outputName>_phase_cycle_correction.yml` | 启用 `phase_cycle_correction` | 记录 selector、整数周、波长、LOS delta 和像元数 |
| `<outputName>_observation_correction.yml` | 启用 `observation_correction` | 记录参考区、系数、公式和改正前后统计 |
| `<outputName>_observation.nc` 或显式 `.h5/.hdf5` | 启用 `export.observation_grid` | 无重采样全分辨率原观测、周跳 delta、改正面、最终观测、projection 和坐标 |
| `<outputName>_observation_grid.yml` | 启用 `export.observation_grid` | 记录 topology、变量、shape 和实际导出文件 |
| `<outputName>_<variable>.tif` | affine TIFF 且 `geotiff_sidecar: auto/true` | 保持原 geotransform/CRS 的逐变量绘图副本 |
| `<outputName>_google_earth.kmz` | 启用 `export.google_earth` 且无 `-s/-c/-d` 阶段 | 全分辨率最终观测的显示副本；不含降采样单元 |
| `<outputName>_processing_region_report.yml` | 启用 `processing_region` 且运行 `-c` 或 `-d` | 记录正式处理区域保留/删除点数 |
| `<outputName>_run_metadata.yml` | 每次运行 | 有效配置、配置版本、deprecated compatibility 字段、执行步骤和预期输出 |
| `<outputName>_downsample_report.yml` | `-d` 且 `downsample.report.enabled: true` | 记录降采样点数、格网、guide-grid、提取规则和质量诊断 |
| `Covariance_estimator.cov` | `-c` | SAR CSI 协方差估计器 |
| `Covariance_estimator_East.cov` / `Covariance_estimator_North.cov` | `-c` | optical east/north CSI 协方差估计器 |
| `<outputName>_ifg.txt` | SAR/optical `-d` | 降采样观测值；SAR 为单标量，optical 为 east/north 双分量 |
| `<outputName>_ifg.rsp` | SAR/optical `-d` | 降采样单元几何；矩形输出默认 18 列 full-corner，三角输出为 8 列 |
| `<outputName>_ifg.cov` | SAR/optical `-d` | 降采样协方差矩阵；optical 为 East/North 分量块矩阵 |
| `<outputName>_decim.png` | SAR/optical `-d` | 降采样结果检查图；optical 默认双列显示 east/north |

### 将 SAR 降采样结果读回反演

`std/data` 或矩形 `from_rsp` 使用 `triangular=False`；`trirb` 或三角 `from_rsp` 必须使用
`triangular=True`：

```python
from csi.insar import insar

sar = insar("TrackA", lon0=lon0, lat0=lat0, verbose=False)
sar.read_from_varres(
    "Downsample/track_ifg",
    triangular=False,  # trirb 或三角 from_rsp 改为 True
    cov=True,
)
```

共同前缀不带 `.txt/.rsp/.cov`。CSI reader 不用 `None` 自动区分三角与矩形；需要自动检查时，
先调用 `read_csi_varres_result(prefix, geometry="auto")`，再把识别结果显式传给
`triangular`。`cov=True` 后不要调用 `buildDiagCd()` 覆盖完整协方差；没有 `.cov` 时才使用
`cov=False` 并建立对角阵。完整示例见
[反演前读取 InSAR 与 GNSS 数据](../examples/inversion_data_loading.md)，字段契约见
[观测数据读入参考](observation_data_readers.md#csi-varres)。

## 常见歧义

| 容易混淆 | 正确理解 |
| --- | --- |
| `data_filters` vs `covar.mask_out` | 前者真实删除点；后者只在协方差估计时排除震源形变区 |
| `data_filters` vs `processing_region` | 前者用于数据质量过滤；后者用于科学关注区域，会影响 `-c/-d` |
| `phase_cycle_correction.selector` vs `observation_correction.fit.regions` | 前者是实际施加整数周改正的目标；后者只是估计全局 offset/plane 的参考样本 |
| `observation_correction` vs `covar.rampEst` | 前者改正降采样输入；后者只服务协方差拟合 |
| `observation_correction` vs 反演 `geodata.polys` | 前者是确定性预处理；后者在反演中联合估计 nuisance 参数 |
| 标准 NetCDF vs PyGMT 派生网格 | 标准文件不重采样；地图投影显示若需重采样，必须另存派生文件 |
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
- 参考改正和全网格导出：[观测参考改正与无重采样网格导出](observation_correction_export.md)
- 命令行入口：[CLI 命令参考](cli.md)
- 图件样式和出版尺寸：[ECAT 图件样式参考 / Viztools](viztools.md)
