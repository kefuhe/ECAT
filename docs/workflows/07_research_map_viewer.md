# 07 本地科研地图查看

本工作流用于把地震目录、断层/块体、GNSS、ECAT 标准观测网格、GeoTIFF 和 CSI
降采样单元放在一张本地交互地图上，快速检查空间关系。查看器只读数据，不执行改正、
降采样或反演，也不替代 Google Earth Pro。

## 1. 安装查看器依赖

在 ECAT 源码目录执行：

```bash
python -m pip install -e ".[viewer]"
```

正式渲染路径使用 Plotly MapLibre，并要求 `plotly>=5.24`。`viewer` extra 同时安装
标准 NetCDF/HDF5 网格和 GeoTIFF 图层依赖。

## 2. 直接查看内置背景

只想随手查看 ECAT 随包发布的公开断层、块体和 GNSS 数据时，不需要 YAML：

```bash
ecat-map
```

页面左侧按“全球构造背景、区域断层、区域块体、GNSS”分组。quick mode 只默认读取并
显示体量很小的 `PB2002 — Plate boundaries`；GEM、AFEAD、区域断层、块体和 GNSS
仍保持隐藏，勾选后才首次加载。需要从指定研究区视角启动时：

```bash
ecat-map --region -72 -62 8 12
```

这里的顺序是 `[min_lon, max_lon, min_lat, max_lat]`。有研究区时，与其相交的区域
资料会带 `study region` 标记并排在相应分组前面；这只是推荐，不会自动加载或排除
其他资料。未安装 console script 时，等价入口是：

```bash
python -m eqtools.map_viewer
```

终端会显示本地地址，默认在浏览器打开 `http://127.0.0.1:8050`。查看期间保持终端
运行，结束时在终端按 `Ctrl+C`。

内置背景来自安装包中的受控资源目录，不扫描当前工作目录，也不会猜测用户文件。

只查看一个 earthquake-client CSV 时也不需要 YAML：

```bash
ecat-map --catalog earthquake/events.csv
```

地震目录会立即显示，内置断层、块体和 GNSS 仍在侧栏按需勾选。

## 3. 加入自己的数据

当需要加入自己的地震目录、观测网格、GeoTIFF 或 CSI varres，或者希望保存并复现
当前项目时，再写 `research_map.yml`。内置背景仍会自动出现在页面中，不需要重复写
进 YAML。

把下面内容保存为 `research_map.yml`。这就是可复制的基础模板，路径相对这个 YAML
解析。第一次通常只改 `view.region`、各层 `source`、`variable/mask` 和 `visible`：

```yaml
version: 2

project:
  name: regional_review  # 页面标题

view:
  region: [-72.0, -62.0, 8.0, 12.0]  # [min_lon, max_lon, min_lat, max_lat]
  basemap: open-street-map             # 最通用的初始底图

layers:
  - id: earthquakes                    # 项目内唯一且保持稳定
    name: Earthquakes                  # 页面中显示的名称
    kind: earthquake_catalog           # 决定怎样读取数据
    source: earthquake/events.csv      # 相对本 YAML
    visible: true                      # true 表示启动时加载并显示

  - id: corrected_los
    name: Corrected LOS
    kind: observation_grid
    source: InSAR/track_observation.nc
    variable: corrected_observation    # 必须明确原始量或改正量
    mask: source_valid                 # 默认；分析像元可用 analysis_valid
    visible: false
    style:                              # 可整段省略
      cmap: RdBu
      alpha: 0.8
      display_factor: 100.0
      display_unit: cm
      vmin: -20.0                      # display_factor 后的显示单位
      vmax: 20.0                       # 只改色标，不裁剪或改写源值
      symmetry: true                   # 仅在自动范围时生效

  - id: downsampled_los
    name: Downsampled LOS
    kind: csi_varres
    source: InSAR/downsample/track_ifg
    data_type: sar                     # sar | optical
    variable: observation
    visible: false
```

`observation_grid` 必须明确写 `variable`。这可以避免将 `observation`、
`corrected_observation`、`phase_cycle_delta` 或 `correction_surface` 静默混用。
`mask` 默认是 `source_valid`；只有明确想显示实际进入分析的像元时才改为
`analysis_valid`。

基础字段各司其职：

| 字段 | 什么时候改 |
| --- | --- |
| `view.region` | 建议按研究区设置；省略时从全球视图启动 |
| `view.basemap` | 通常保留 `open-street-map`；地形用 `outdoors`，影像用 `satellite` 或 `satellite-streets` |
| `id` | 新增图层时给一个项目内唯一、稳定的标识 |
| `kind` | 按数据对象选择，不是随意显示标签 |
| `source` | 每个案例都要改成真实文件或 varres 前缀 |
| `variable/mask` | 标准观测必须选变量；mask 通常保留默认 |
| `visible` | `true` 启动时加载；大图层通常先写 `false`，需要时再勾选 |
| `style` | 可选且只影响显示；不确定时整段省略 |

启动这个项目：

```bash
ecat-map research_map.yml
```

未安装 console script 时等价运行：

```bash
python -m eqtools.map_viewer research_map.yml
```

临时改变初始视角而不修改 YAML：

```bash
ecat-map research_map.yml --region -72 -62 8 12
```

需要端口、host 或 debug 选项时先看：

```bash
ecat-map --help
```

浏览器中的底图、图层显隐、活动图层、运行时样式和视角属于当前页面，不会写回 YAML
或科学源文件；F5 刷新会从 YAML/默认值重新建立一致状态。隐藏图层在第一次勾选前
不会解析；隐藏后再次显示使用进程内只读缓存。

页面中最常用的联合查看操作是：

- `Hide all`：清空当前显示，但不清空目录或缓存；
- `Show active only`：只显示活动图层，避免影像、断层和点数据互相遮盖；
- `Apply alpha`：只修改活动图层的本次显示透明度；
- `Apply color limits`：给活动定量图层设置成对的 `vmin/vmax`；
- `Reset to auto`：恢复 2–98 百分位自动范围，可选择关于零对称；
- `Fit active layer`：缩放到该层范围；
- `Active layer metadata`：项目图层显示声明的来源、变量、格式和数据类型；内置背景
  还显示随包提供的单位、参考框架和引用。已加载观测的数值、单位等继续在图上
  hover/click 信息中检查。

活动图层与可见图层不是同一个概念：可以同时显示多层，但只有活动层显示连续色标并
接受 alpha、色限和范围定位操作。

## 4. 使用逻辑

Viewer 始终按三层组织：

```text
随包发布的背景目录
  + 可选的 project YAML 用户图层
  -> 页面中的可选图层目录
  -> 当前页面的可见图层与一个活动图层
  -> 底图、alpha、色限和视角
```

“进入目录”和“显示”是两回事：目录决定页面里有哪些图层，勾选状态决定当前加载和
显示哪些图层。YAML 只声明用户项目，不是打开 Viewer 的硬要求；浏览器状态也不会
反写 YAML。

## 5. 核对

至少检查：

- 地震、断层和观测的经纬度区域一致；
- 标准观测图层的变量名、单位和正号约定符合预期；
- GNSS hover 未提供单位或参考框架时会显示 `unknown`，不会猜测；
- 内置 Wang et al. (2020) GNSS 明确显示 `mm/yr` 和相应参考框架；
- CSI varres 的中心、值和 cell 边界与降采样 QC 图一致；
- 切换底图后图层、zoom 和 center 不丢失；
- 同一组图层按数据语义保持稳定叠放，不随首次勾选顺序改变；
- 多个定量图层同时显示时只有活动层显示 colorbar；
- 大栅格显示的是只读 overview，科研计算仍使用原标准文件。

## 6. 直接查看地震目录

```bash
ecat-map --catalog earthquake/events.csv
```

该入口直接建立一个临时 ViewerProject。旧 Scattermapbox manager/strategy 网页实现
及 `plot_earthquakes_on_map_interactive()` 已移除；下载 API 和 Matplotlib 静态绘图
仍是独立功能。

## 7. 当前范围

首版支持六个明确的 `kind`：

- `earthquake_catalog`
- `vector`
- `gnss_velocity`
- `observation_grid`
- `raster`
- `csi_varres`

原始 GAMMA、GMTSAR、HyP3 和 optical 平台文件先通过现有降采样/标准网格导出流程
统一语义，再进入 viewer。当前不在网页中执行 beachball sprite、自动目录发现、
瓦片化或数据编辑；这些限制不会影响地震目录普通点显示。需要在观测上移动断层迹线
节点时，使用职责独立的
[交互调整断层迹线](02c_interactive_trace_editing.md)，而不是让 `ecat-map` 修改图层。

完整字段、格式契约和错误说明见
[Research Map Viewer Reference](../reference/research_map_viewer.md)。只想复制一个更短
的项目时见 [科研地图项目短例](../examples/research_map_viewer.md)。

## 8. 预期结果与下一步

运行后应得到一个只读的本地交互页面：可以按需加载图层、勾选显示状态、切换底图，并检查地震、断层、GNSS 和观测数据的空间关系。页面交互不会修改输入文件。

- 发现坐标、单位或正号异常：返回对应 reader 或标准网格导出流程修正，不在 viewer 中改数值。
- 需要调整断层迹线：进入 [交互调整断层迹线](02c_interactive_trace_editing.md)。
- 需要把原始观测、断层或地震目录带到 Google Earth Pro：进入 [Google Earth 科研导出](06_google_earth_export.md)。
- 需要重复打开同一批用户数据：把临时命令整理成 `research_map.yml`，并保留相对路径。
