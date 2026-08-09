# Research Map Viewer Reference

本页是项目字段、图层契约和 CLI 的完整查阅层。第一次使用先读
[本地科研地图查看](../workflows/07_research_map_viewer.md)。

## 阅读路径

- 只想查看随包背景：看 [CLI](#cli) 和 [内置背景目录](#内置背景目录)。
- 要加入自己的数据：看 [最小项目](#最小项目)。
- 不确定 `kind/format/variable`：看 [图层字段](#图层字段) 和
  [支持的科学格式](#支持的科学格式)。
- 担心数值被修改：看 [只读与缓存边界](#只读与缓存边界)。
- 图层无法显示：看 [错误与当前限制](#错误与当前限制)。

## CLI

```text
ecat-map [PROJECT] [--catalog EVENTS.csv]
         [--region MIN_LON MAX_LON MIN_LAT MAX_LAT] [--basemap STYLE]
         [--host HOST] [--port PORT] [--debug BOOL]
```

安装后可直接运行 `ecat-map --help` 查看当前入口选项。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `PROJECT` | 空 | 可选的 version 2 项目 YAML；省略时只打开内置背景目录 |
| `--catalog` | 空 | 不写 YAML，直接加入一个 earthquake-client CSV；不能与 PROJECT 同时使用 |
| `--region` | project 的 `view.region` 或全球 | 临时初始视角；不写回 YAML |
| `--basemap` | project 或 `open-street-map` | 临时覆盖初始底图 |
| `--host` | `127.0.0.1` | 本地 Dash host |
| `--port` | `8050` | 本地端口 |
| `--debug` | `false` | Dash debug/reloader；接受 `true/false` 等明确布尔值 |

快速查看与项目模式：

```bash
ecat-map
ecat-map --region -72 -62 8 12
ecat-map --catalog data/events.csv
ecat-map research_map.yml
```

等价模块入口：

```bash
python -m eqtools.map_viewer
python -m eqtools.map_viewer research_map.yml
```

<a id="内置背景目录"></a>
## 内置背景目录

无 YAML 模式会注册随安装包发布的三类资源，并在页面中按科学用途分组：

| 页面分组 | 代表资料 | quick mode 初始状态 |
| --- | --- | --- |
| Global tectonic context | PB2002、GEM Global Active Faults | 只显示并读取 PB2002 boundaries |
| Regional fault data | AFEAD、CAFD、中国区域断层 | 隐藏、未读取 |
| Regional blocks | 中国区域块体 | 隐藏、未读取 |
| GNSS velocity fields | Wang et al. (2020) stable-Eurasia / ITRF2008 | 隐藏、未读取 |

同一 stem 同时有 JSON/GeoJSON 和 GMT 时只注册首选 JSON/GeoJSON，不重复显示。资源
查找只发生在安装包目录，不扫描当前工作目录；用户自己的同名文件不会覆盖内置背景。

项目模式先保留 YAML 中图层的顺序，再附加同一套内置背景。项目模式中的背景图层全部
保持 `visible: false`，不会因 quick mode 的轻量默认值而自动覆盖用户项目。背景图层
使用 `background.*` 稳定 ID；第一次勾选后才通过现有 loader 读取。

内置资源另有一份加载前元数据：显示范围、用途、引用、单位和参考框架。指定
`view.region` 或 CLI `--region` 后，相交的区域资料会带 `study region` 标记并优先
排列；这只是目录提示，不自动读取、不裁剪，也不隐藏其他资源。

## 最小项目

```yaml
version: 2
project:
  name: regional_review
view:
  region: [-72.0, -62.0, 8.0, 12.0]
  basemap: open-street-map
layers:
  - id: events
    name: Earthquakes
    kind: earthquake_catalog
    source: data/events.csv
    visible: true
```

根层只接受 `version/project/view/layers`。当前不接受 `discover`，不会静默猜测标准
观测文件中的变量。YAML 只声明用户项目图层；内置背景由应用目录提供。

## 顶层字段

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `version` | 是 | 当前必须为 `2` |
| `project.name` | 否 | 页面标题；缺省取 YAML 文件名 |
| `view.region` | 否 | `[min_lon, max_lon, min_lat, max_lat]` |
| `view.basemap` | 否 | 初始底图，默认 `open-street-map` |
| `layers` | 否 | 显式图层列表 |

本地 `source` 相对项目 YAML 解析。当前不从项目文件读取任意远程 URL。

## 图层字段

| 字段 | 必填 | 职责 |
| --- | --- | --- |
| `id` | 是 | 项目内稳定唯一 ID；不是显示名 |
| `name` | 否 | UI 显示名，缺省使用 `id` |
| `kind` | 是 | 科学/业务语义，决定 loader |
| `source` | 是 | 文件或 CSI varres 公共 prefix |
| `variable` | 依 kind | 多变量文件中的明确变量 |
| `mask` | 依 kind | 仅用于 `observation_grid` 的有效像元选择 |
| `visible` | 否 | 初始显隐，默认 `false` |
| `style` | 否 | 少量显示覆盖，不改变科学值 |
| `format` | 否 | 仅在扩展名无法消歧时声明编码/子契约 |
| `data_type` | 依 kind | `csi_varres` 行契约：`sar` 或 `optical` |

三者不可混用：

- `kind` 回答“这是什么科学对象”；
- `format` 回答“这个对象如何编码”；
- `variable` 回答“多变量文件里显示哪一个量”。

## 支持的科学格式

| `kind` | 当前文件契约 | `variable/format` |
| --- | --- | --- |
| `earthquake_catalog` | ECAT/USGS/GCMT 规范 CSV；必须有 `longitude/latitude` | 使用稳定列名，不猜任意别名 |
| `vector` | GeoJSON FeatureCollection 或 GMT line | `format` 只能是 `geojson` 或 `gmt`；扩展名明确时省略 |
| `gnss_velocity` | ECAT velocity GeoJSON；Point feature 的 `velocity=[east,north,...]` | `style.display_scale` 只改变箭头长度 |
| `observation_grid` | `write_observation_netcdf()` 生成的 ECAT CF-NetCDF/HDF5 | `variable` 必填 |
| `raster` | 具有 CRS 和 affine transform 的 GeoTIFF | 当前只读 `band_1`；不要把其他 band 名写入 `variable` |
| `csi_varres` | 成对 `<prefix>.txt/.rsp` | `data_type: sar`（默认）或 `optical`；`variable` 选 component |

`observation_grid` 复用
`eqtools.csiExtend.downsample.observation_grid.read_observation_grid()`；`csi_varres`
复用 `read_csi_varres_result()`。viewer 不再实现平台 reader、projection、正号或单位
转换。

### observation grid 可用变量

标准文件可包含：

- 基础量：`observation`、`east`、`north` 等；
- `corrected_observation` 或 `corrected_<component>`；
- `phase_cycle_delta`；
- `correction_surface`；
- 相应 optical component 的派生变量。

配置必须明确选择。projection 仍保留在标准文件中，但当前不作为独立彩色图层。

`observation_grid.mask` 可取：

- `source_valid`：reader 判定有效的源像元，默认，也与 Google Earth 导出默认一致；
- `analysis_valid`：实际进入分析范围的像元；
- `finite`：仅要求所选变量和坐标为有限值。

mask 只控制显示哪些像元，不改写标准文件中的值。`visible` 必须写 YAML 布尔值
`true/false`，不要写成带引号的字符串。

### style 字段

| 字段 | 含义 |
| --- | --- |
| `color` | 线、边界或 GNSS 颜色 |
| `cmap` | 定量图层连续色标名 |
| `vmin/vmax` | `display_factor` 后的成对显式色限 |
| `symmetry` | 自动色限是否关于 0 对称；显式上下限优先 |
| `alpha` | `[0, 1]`，0 完全透明，1 完全不透明 |
| `display_factor/display_unit` | 仅改变显示值、色标和标签，不改源值 |
| `display_scale` | GNSS 显示箭头比例，不改变物理速度 |
| `line_width` | 线宽 |
| `marker_size` | 点大小 |

style 会按 `kind` 校验；例如 `vector.style.cmap` 会报错，而不是被 renderer 静默忽略。
自动色限统一使用完整有限显示值的 2–98 百分位；`symmetry: true` 再取其中最大绝对值
构造零中心范围。抽稀仅发生在色限确定以后。

## 页面图层操作

页面区分三个概念：

- available：图层已进入目录，可能仍未读取；
- visible：当前勾选并显示，可同时有多个；
- active：当前接收 alpha、定量色限、范围定位和声明检查，并拥有 colorbar 的一个图层。

`Hide all` 只清空 visible 状态；`Show active only` 令活动层成为唯一可见层。二者都不
删除目录、缓存或源文件。`Apply alpha`、`Apply color limits` 和 `Reset to auto` 只
修改当前页面的显示覆盖，不写回 YAML；`Fit active layer` 显式读取该层元数据并缩放
到其 bbox。

定量图层可以同时显示，但只有活动层显示 colorbar，避免多个色标相互覆盖。绘制顺序按
科学对象固定为：

```text
observation grid / raster / CSI varres
  -> vector
  -> GNSS
  -> earthquake catalog
```

同类图层再按项目/目录声明顺序排列，因此首次勾选顺序不会改变覆盖关系。
地图本身不再重复显示 Plotly trace 图例；图层名称、分组与显隐以左侧图层面板为准，
避免图例挤占地图宽度。

## 配置与运行时层级

```text
随包背景图层（始终注册；quick mode 仅 PB2002 boundaries 默认显示）
  + 可选 project YAML（用户图层与初始 basemap / region / visible）
  + 可选 CLI --region（仅覆盖本次初始视角）
  -> 当前页面覆盖 basemap / viewport / visible / active / alpha / color limits
```

无 YAML 时直接由背景资源工厂建立内存 `ViewerProject`；有 YAML 时解析用户项目后
合并同一背景目录。两条入口进入相同的 catalog、loader、renderer 和 callback。

运行时状态不写回项目。用户状态和 `layer id -> trace index` 只保存在当前页面内存；
F5 后从项目重新建立 figure 和索引，避免旧索引与新 figure 错配。数组、GeoJSON、
栅格和完整 figure 不放入浏览器 Store；同一个服务进程中的页面共享 detached parsed
cache。

## 只读与缓存边界

- 隐藏图层第一次勾选时才解析；
- 再次显隐只修改 trace visibility；
- 切换底图只 Patch `layout.map.style`，不返回 overlay payload；
- callback 只传用户状态、renderer 索引、click/viewport 小事件，不把完整 figure
  作为服务器 State；
- 缓存键包含 resolved source、大小、`mtime_ns`、kind、variable、mask、format 和
  loader version；
- 已加载源文件变化后，重启 viewer 才会读取新版本；
- 缓存淘汰只释放内存，不删除或覆盖源文件；
- overview、marker 抽稀、色标和透明度只影响显示；
- viewer 不编辑要素；观测上的迹线调整由独立的
  [交互迹线编辑器](interactive_trace_editor.md) 使用 working copy 完成；
- 内置 GNSS 的 `0..360` 经度只在 detached display payload 中规范为
  `[-180, 180)`；源经度另行保留，源 JSON 不因查看而改写。

标准观测网格的二维 lon/lat 按原 row/column 对应抽取显示点，不压成一维轴、不插值。
GeoTIFF 大图使用 nearest display overview，并在 metadata 标记为显示派生；科研计算仍
应读取权威源文件。标准网格色限从完整有效值解析后，最多选择 60,000 个空间分布的
显示点。

## Renderer

查看器要求 `plotly>=5.24`，并只使用 `Scattermap/layout.map` 的 MapLibre 路径；不再
保留 Scattermapbox backend 或 Mapbox token。底图包括：

- `open-street-map`
- `carto-positron`
- `carto-darkmatter`
- `streets`
- `outdoors`
- `satellite`
- `satellite-streets`
- `white-bg`

除 `white-bg` 外均依赖在线 tile。外部 tile 的许可和 attribution 由使用者按 provider
要求核对。

## 地震目录快捷入口

```bash
ecat-map --catalog events.csv
```

该入口生成临时 `ViewerProject` 并附加内置背景目录。旧脚本中的
`plot_earthquakes_on_map_interactive()` 不再提供；应改用上述 `ecat-map --catalog`
命令。下载 API、CSV 字段和 Matplotlib 静态绘图 API 不受影响。

## 错误与当前限制

- 一个图层加载失败只在状态面板报告，不清空其他已显示图层；
- 内置背景目录缺少某个可选分类时以空目录处理，不扫描其他位置补齐；
- 标准观测不写 `variable` 会在项目解析阶段报错；
- `.txt/.rsp` 缺一、行数或索引错位会 fail-fast；
- GNSS hover 无单位/参考框架时显示 `unknown`；活动图层卡片只为内置背景显示
  随包维护的单位、参考框架与引用，不猜测普通项目文件未声明的元数据；
- GeoTIFF 无 CRS/affine 时拒绝显示；
- `vector` 的 GeoJSON Point/MultiPoint 显示为 marker，LineString/Polygon 显示为线；
- 当前未实现自动 `discover`、beachball sprite、COG/XYZ tiles、视域服务端查询、
  项目状态写回和要素编辑。

这些限制位于显示层，不改变 downsampling、BLSE、VCE、SMC_FJ、FULLSMC 或约束管理器。
