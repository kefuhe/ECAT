# ECAT Research Map Viewer 内部维护指南

> 当前实现基线：2026-08-04。
>
> 本文解释代码职责、只读科学边界和扩展步骤，不是用户配置教程。

## 1. 架构结论

viewer 是科研处理之后的本地只读显示层：

```text
平台 reader / observation correction / downsampling
  -> ECAT 标准观测网格或 CSI varres
  -> 可选 project YAML 用户层
随包 Faults / Blocks / GNSS
  -> packaged background project factory
两者 -> 同一个 LayerCatalog / loaders
  -> detached LayerPayload
  -> Plotly renderer
  -> Dash page-memory state
```

反演、约束和降采样代码不 import `eqtools.map_viewer`。`eqtools.map_viewer` 顶层也使用
lazy import；未安装 Dash/Plotly 时，数值模块仍可独立导入。

## 2. 包职责

```text
eqtools/map_viewer/
  models.py             LayerSpec/Metadata/Payload/ViewerState/Project
  project.py            严格 version-2 YAML 与相对路径解析
  backgrounds.py        随包背景目录元数据、LayerSpec 与 quick/project 工厂
  loaders.py            kind -> read-only semantic loader
  cache.py              有上限的进程内 parsed LRU
  catalog.py            稳定 ID、lazy load 与 fingerprint cache key
  renderer_plotly.py    payload -> stable child traces
  ui.py                 分组图层面板与只读元数据卡片
  app.py                Dash callback、Patch 和 page-memory state
  cli.py/__main__.py    ecat-map 与 python -m 入口
  interactive/
    models.py            working copy、有限历史和只读背景合同
    coordinates.py       canonical lon/lat 与 Web Mercator 唯一转换
    trace_io.py          reference 读取与安全 Save As
    adapters.py          LayerPayload/ObservationGrid 到编辑背景
    display.py           显示倍率、全场色限和权威色表方向
    raster.py            可选 Datashader 曲线网格与视窗帧
    bokeh_trace_editor.py Bokeh 视图、显示控制和本地 server
    cli.py/__main__.py   ecat-trace-edit 与 python -m 入口

eqtools/earthquake_clients/
  resources.py          只列随包 Faults/Blocks/GNSS，不扫描工作目录
```

旧 `earthquake_clients.app`、`MapLayerManager`、strategy 交互栈和
`plot_earthquakes_on_map_interactive()` 已删除。下载 client 和 Matplotlib 静态绘图
不依赖新 viewer；直接目录快捷入口统一为 `ecat-map --catalog EVENTS.csv`。

### P0 交互边界

`ecat-map` 继续只读；`eqtools.map_viewer.interactive` 是独立的 working-copy
任务边界，不把可变节点塞进 `LayerPayload`、`ViewerState` 或 parsed cache：

```text
authoritative reader/correction
  -> detached EditorBackground
immutable ReferencePath(s)
  -> InteractiveWorkspace
  -> one mutable PathDraft + EditHistory
  -> explicit Save As
```

`InteractiveWorkspace.active` / `PathDraft.coordinates` 是 working geometry 的唯一权威状态。
Bokeh working `MultiLine` source 是浏览器几何编辑的唯一提交通道；其完成事件经过统一
commit 写回 workspace，再投影到坐标表。PolyEdit 节点 source 由 Bokeh 管理，只表示当前
选中工作线的临时编辑手柄：不得从 workspace 预填，不得注册 Python data commit callback，
也不得把清空临时节点解释为清空 working trace。workspace 的 New/Copy/Undo/Redo/Delete
等命令只投影到 working line 与坐标表；浏览器手势提交时不得立即回写 working line 或
节点 source，以免打断 Bokeh 对共享顶点数组的更新生命周期。不要调用 `_show_vertices`、
`_selected_renderer` 等 Bokeh 私有接口。节点/表格选择只同步 selection，严禁在 selection
callback 中修改 plot range；平移和缩放只属于 Browse/地图工具。

`interactive.__init__` 不导入 Bokeh。`bokeh_trace_editor.py` 才延迟检查
`bokeh>=3.6,<4`；反演、约束、标准降采样和 `ecat-map` 不依赖 interaction extra。
降采样交接只在 `cli_tools.process_data_downsampling` 调度层延迟导入，复用已经解释和
改正的 `ObservationGrid`。严禁让 `csiExtend.downsample` 反向 import viewer。

结构化二维背景的显示链固定为：`EditorBackground` 完整只读数组 → `display.py` 解析完整场色限和色表方向 → `raster.py` 逐点转 Web Mercator 并缓存二维 DataArray → Bokeh range callback 防抖生成当前视窗 RGBA。视窗聚合只属于显示；不得把 Datashader aggregate 传给保存、导出、剖面采样或降采样。非二维或坐标含非有限值时明确走有效点回退。

扩展 P1 profile 时沿用下面的最小模式，不在 renderer 中读取科学格式：

```python
payload = authoritative_loader(spec)          # detached, read-only
background = background_from_payload(payload) # display adapter
workspace = InteractiveWorkspace(references)  # only path state is mutable
```

新增路径 I/O 时必须先定义坐标、分段、覆盖和 provenance 规则，再扩展
`trace_io.py`；新增科学 source 时先扩展既有 loader/adapter，不能在 Bokeh callback
里复制平台 reader。

## 3. 入口和目录层

公开入口分两种，但进入同一个对象模型：

```text
ecat-map
  -> create_background_project()

ecat-map research_map.yml
  -> load_viewer_project()
  -> with_packaged_backgrounds()

ecat-map --catalog events.csv
  -> project_from_catalog()
  -> with_packaged_backgrounds()

两者 -> create_app(ViewerProject)
```

背景资源是应用级、受控的图层目录，不是用户目录 discovery：

- 只访问安装包内 `earthquake_clients/data/{Faults,Blocks,GNSS}`；
- 同 stem 优先 GeoJSON/JSON，避免与 GMT 重复注册；
- ID 使用 `background.<category>.<stem>` 命名空间；
- `packaged_background_layers()` 全部 `visible=False`；
- `create_background_project()` 只把轻量 PB2002 boundaries 设为可见；
- `with_packaged_backgrounds()` 仍向用户项目附加全隐藏背景；
- 用户项目层保持 YAML 顺序并排在背景层之前；
- `--region` 只替换本次内存 project 的初始视角，不写回 YAML。

不要把 quick mode 扩展成任意工作目录扫描。用户文件仍必须通过 project 层明确
`kind/source/variable/mask`，避免观测变量和编码猜测。

## 4. 四层对象

### `LayerSpec`

用户/项目层稳定声明。`id` 是父图层身份，`name` 只用于显示。`kind`、`format`、
`data_type` 和 `variable` 分别表示科学对象、文件编码、varres 行协议和文件内变量，
不能混用。`mask` 只属于
`observation_grid`，默认 `source_valid`；`visible` 必须是 YAML boolean。GeoTIFF
首版只读 `band_1`，其他 `variable` 必须在解析阶段拒绝，不能只改显示标签。

定量图层的公共用户词汇与 geoexport 对齐：

```text
cmap / vmin / vmax / symmetry / alpha
display_factor / display_unit
```

`display_scale` 只属于 GNSS 箭头几何，不能与数值 `display_factor` 合并。style 必须按
kind 校验；renderer 不得静默忽略合法模型中已经接受的字段。

### `LayerMetadata`

loader 产生的小型信息：bbox、shape/count、变量、单位、CRS、topology 和 fingerprint。
它不写回 YAML。

### `LayerPayload`

科学 loader 的 detached 结果。数组复制后设为 read-only；共享缓存中的 payload
不得被 callback 或 renderer 修改。

### `ViewerState`

每个页面内存中的小型 JSON：basemap、viewport、visible IDs、active layer ID
和 runtime style overrides。严禁放 DataFrame、大数组、GeoJSON 或 raster。不要为
尚未实现的 filter 或 selected-feature 行为预留无调用字段。

Dash 另有一个小型 `viewer-render-state`，只保存 `trace_count` 和
`layer_id -> child trace indices`、`layer_id -> colorbar trace indices`。它是 renderer
bookkeeping，不进入公开项目模型；其作用是让显隐、活动色标和 alpha callback 不需要
把完整 figure 作为 `State` 传回服务器。

两个 Store 都使用 `storage_type="memory"`。F5 后必须同时重建 figure、trace index 和
用户状态；不要只持久化 index。只有出现明确的跨刷新恢复需求时，才可设计完整
rehydration，不得重新引入半持久化 session index。

## 5. 加载和缓存调用链

```text
LayerCatalog.load(layer_id)
  -> source_fingerprint()
       local path + size + mtime_ns
       varres 同时包含 .txt 和 .rsp
  -> (id, fingerprint, kind, variable, mask, format, data_type, loader_version)
  -> ParsedLayerCache.get()
       hit: 返回同一只读 payload
       miss: load_layer(spec) -> cache.put()
```

当前不提供 UI refresh 或文件监视。源文件变化后重启 viewer；缓存淘汰只能释放内存，
不能执行文件删除。render cache/tiles 当前尚未实施。

## 6. 科学 loader 的权威边界

| kind | 权威读取入口 | viewer 允许做什么 |
| --- | --- | --- |
| earthquake_catalog | 规范 CSV 列 | 数值列校验、时间解析 |
| vector | GeoJSON 或 `read_gmt_lines()` | GMT 在内存转 geometry，不写 JSON sidecar |
| gnss_velocity | ECAT velocity GeoJSON | 保存物理 east/north 和源经度；display lon 规范为 `[-180,180)`，scale 延迟到 renderer |
| observation_grid | `read_observation_grid()` + `resolve_observation_variable()` | 按明确 variable 与 source/analysis/finite mask 拷贝值和二维坐标 |
| raster | rasterio | nearest display overview、真实 affine center 和 CRS 转换 |
| csi_varres | `read_csi_varres_result()` | 保持 row、projection、value 和 vertices 顺序 |

不得在 loader 中增加：

- GAMMA/GMTSAR/HyP3 单位和正号解释；
- 相位到 LOS 转换；
- reference/ramp/cycle correction；
- 插值后写回源文件；
- CSI/csiExtend 对象重建；
- BLSE/VCE/SMC 参数索引。

## 7. Renderer 与 callback

`renderer_plotly.py` 为每个父图层生成一条或多条 child trace；所有 child trace 的
`meta.layer_id` 相同。GNSS 的 station/vector、varres 的 cell/value 因此在 UI 中仍是
一个图层。通用 vector 的 Point/MultiPoint 生成 marker child trace，
LineString/Polygon 生成 line child trace；loader 接受的 geometry 不得在 renderer 中
无声丢失。

正式后端是 Plotly MapLibre：

```text
Scattermap + layout.map
```

项目要求 Plotly 5.24+；不保留 Scattermapbox compatibility backend。`uirevision`
使用稳定 project key，不能再写当前时间。

callback 职责固定：

```text
basemap
  -> Patch layout.<map>.style

visibility
  -> 查小型 viewer-render-state
  -> 已存在 child traces: Patch visible
  -> 第一次显示: catalog.load() + 按 kind/order Insert traces
  -> 插入后同步平移所有 trace/colorbar indices

active layer
  -> 只 Patch colorbar-capable trace 的 marker.showscale

alpha
  -> 只 Patch active layer child trace 的 Plotly opacity

color limits
  -> 从完整有限显示值解析 vmin/vmax
  -> 再做 overview / spatial point sampling
  -> 只 Patch active quantitative child trace 的 cmin/cmax

fit
  -> 显式加载 active payload metadata.bbox
  -> Patch center/zoom，不修改数据

viewport
  -> 只更新当前 page-memory ViewerState

inspector
  -> 读取 clickData 的显示属性
```

稳定绘制优先级是 grid/raster/varres、vector、GNSS、earthquake；同类保持 catalog
顺序。不能退回“首次勾选顺序就是覆盖顺序”。callback 不得修改闭包外 manager，也
不得调用旧 `clear_layers()`。图层名称、分组与显隐由 `ui.py` 的左侧面板统一管理；
renderer 保持 `layout.showlegend=False`，避免重复 trace legend 挤占地图宽度。

CLI 使用 `app.run()`，兼容声明支持的 Dash 2/3；不要重新使用 Dash 3 已移除的
`app.run_server()`。

## 8. 新增 loader 的模板

只有现有六种 `kind` 无法表达真实科研对象时才新增 kind；新文件编码优先扩展已有
kind。步骤：

1. 在 `models.SUPPORTED_LAYER_KINDS` 明确语义；
2. 在 `loaders.py` 新增 `_load_<kind>(spec, fingerprint)`；
3. 从权威 reader 复制 detached 值，不复制科学解释；
4. 建立 `LayerMetadata`，写明 units/CRS/topology/derived display；
5. 在 `_LOADERS` 注册；
6. 在 renderer 增加一种稳定 parent-ID 映射；
7. 测试源文件 hash、shape、坐标、值、mask 和顺序不变；
8. 更新公开 workflow/example/reference。

最小结构：

```python
def _load_new_kind(spec, fingerprint):
    source = authoritative_read_only_reader(spec.source)
    values = _readonly(source.values)
    metadata = LayerMetadata(
        layer_id=spec.id,
        bbox=...,
        fingerprint=fingerprint,
        units=source.units,
        crs=source.crs,
    )
    return LayerPayload(spec, metadata, {"values": values})
```

若无法明确单位、坐标、正号、mask 或变量职责，loader 不得注册为正式支持。

## 9. 测试要求

每次修改至少覆盖：

- quick mode 无 YAML 启动、随包资源稳定去重和不扫描 cwd；
- quick mode 只解析 PB2002 boundaries，项目模式背景仍全部隐藏；
- 研究区相交只影响目录提示和排序，不自动加载；
- project mode 保留用户层顺序并附加隐藏背景；
- 首页响应不能触发隐藏背景解析；
- project 未知字段、重复 ID、相对路径和 explicit variable；
- loader source hash 不变；
- raw/corrected/delta/surface 不混淆；
- varres value/projection/vertices 顺序；
- cache hit 和 source fingerprint 变化；
- basemap Patch 操作中不存在 `data`；
- callback State 中不存在完整 `viewer-map.figure`；
- hidden layer 不在初始响应中解析；
- 延迟加载的低层级 trace 使用 Insert，并正确平移既有 trace/colorbar 索引；
- 同时只给 active layer 打开 colorbar，alpha Patch 只作用于该父层；
- 自动色限使用完整值，空间显示抽样不能反向改变色限；
- F5 后两个 memory Store 与新 figure 一致重建；
- 卫星、卫星街道、地形底图在 MapLibre 下可选；
- 地图关闭重复 trace legend，图层面板仍完整列出可用图层；
- GNSS source longitude 保留，display longitude 规范化且 bbox 对齐；
- GNSS 中高纬显示端点保持真实 EN 方位，源 east/north 不变；
- stable `uirevision` 和 viewport parser；
- `ecat-map --catalog` 直接目录入口；
- `tests`、`csiExtend/tests` 回归，确认数值核心零回退；
- interaction 顶层导入不加载 Bokeh，缺依赖时只有交互入口报可操作错误；
- reference/working 分离，源文件 hash 与 observation 数组不变；
- add/move/insert/delete、clear/delete path、undo/redo 和 redo 分支；
- canonical longitude、Web Mercator 往返和 UTF-8 路径；
- TXT/GeoJSON 往返、默认拒绝覆盖、输出兼容 `read_trace_file()`；
- Bokeh 文档验证、Browse/Draw/Edit、节点表、可见按钮和快捷动作；
- PolyEdit 临时节点 source 的独立变化不修改 workspace、工作线、坐标表或历史；
- 完成的 working-line 变化写回 workspace 和坐标表且只产生一次历史；
- 进入 Edit 后先选择黄色工作线才显示临时节点，离开 Edit 后节点隐藏；
- 删除节点保持剩余顺序并重连，选择和编辑均不隐式修改 plot range；
- 降采样交接复用 corrected component，未启用交互时步骤与数值路径不变；
- 二维曲线网格视窗重绘、固定全场色限、色表方向、无效坐标回退和源数组不变；
- 灰白/街道/地形/卫星/无底图切换，观测与 colorbar 显隐及透明度只改 renderer。

MapLibre 版本升级后还需运行浏览器点击序列：切底图、显隐、再次显示、F5 后再显隐、
保持视角以及两个独立页面互不污染。

## 10. 克制的待办边界

当前有意未实施：

- 用户目录自动 `discover` 和 variable 子项 UI；
- focal-mechanism sprite；
- server-side viewport/LOD、COG/XYZ/vector tile；
- render disk cache；
- filter 面板和项目状态写回；
- `ecat-map` 内原位要素编辑和任意 HDF5/KML 导入。

增加这些能力前必须有真实案例和性能证据。首先扩展纯 loader/state/renderer 边界，
不能让 viewer 反向进入数值核心。

交互迹线调整 P0 已按独立 working copy 和可选 Bokeh 后端落地；后续多源剖面仍处于
计划阶段，详见
[ECAT 交互迹线调整与多源剖面分析实现计划](INTERACTIVE_TRACE_PROFILE_PLAN.md)。
P0 不改变本指南定义的只读 LayerPayload、缓存和数值核心边界。
