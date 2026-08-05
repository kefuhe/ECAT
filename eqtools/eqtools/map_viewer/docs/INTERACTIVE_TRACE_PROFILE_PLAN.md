# ECAT 交互迹线调整与多源剖面分析实现计划

> 文档性质：内部维护计划，不是已经发布的用户接口说明。
>
> 当前基线：2026-08-04。
>
> 实施状态：P0-A/P0-B/P0-C 与 P0-D 连续观测显示已完成；P1/P2 保持规划，尚未实现观测、GNSS 或地震目录剖面。

## 1. 目标和边界

### 1.1 当前首要需求

用户查看降采样前的原始或改正后观测时，通常已有发表迹线或粗略描绘的迹线，但它未必
与当前影像中的形变梯度、断层不连续带或局部构造位置完全一致。首阶段支持：

1. 以连续二维栅格显示原始观测，并叠加一条或多条只读参考迹线；
2. 从参考迹线复制工作副本，或新建连续折线；
3. 添加、移动、插入、删除节点，并支持撤销和重做；
4. 同步查看、选择和复制节点经纬度；
5. 以新文件保存调整后的迹线，供现有 fault_traces.file 直接读取；
6. 不修改原始观测、发表迹线、降采样数值或项目 YAML。

### 1.2 面向未来的剖面需求

后续复用同一套路径编辑能力绘制剖面线，并对不同科学对象执行与其语义匹配的提取：

- SAR、光学等规则或曲线观测网格；
- CSI varres 或其他离散观测；
- GNSS 位移或速度点及其平行、垂直剖面的分量；
- 小震目录的沿剖面距离—深度分布和走廊统计；
- 确有案例时再接入其他点、线或网格数据。

P0 只保留可复用的 PathDraft 和坐标协议，不提前实现采样器、剖面图、自动拟合或插件
注册系统。

### 1.3 明确不做

- 不改变 GAMMA、GMTSAR、HyP3、光学等科学 reader；
- 不在交互层解释单位、正号、LOS 投影或观测改正协议；
- 不用编辑结果自动覆盖发表迹线或输入配置；
- 不默认自动吸附到影像梯度，不自动平滑或简化迹线；
- 不把自由手绘作为默认方式，避免大量难以审阅的噪声节点；
- 不把剖面插值写回原始观测；
- 不替代 Google Earth Pro、GIS 或完整桌面制图软件；
- 不把 Bokeh、Dash 或浏览器依赖引入反演、约束和降采样数值核心。

## 2. 架构结论

### 2.1 保留只读 viewer，新增独立交互工作区

现有 ecat-map 继续负责快速查看多源图层。可变节点不能直接放入 LayerPayload 或
ViewerState，也不能让 callback 修改缓存中的科学数组。

~~~text
reader / correction / downsampling
  -> 标准观测网格、CSI varres、GNSS、地震目录或 vector
  -> LayerCatalog / loader
  -> immutable LayerPayload ----------------------+
                                                    |
                                                    v
                                     InteractiveWorkspace
                                       reference layers 只读
                                       PathDraft 工作副本可写
                                       EditHistory 小型历史
                                                    |
                         +--------------------------+------------------+
                         |                                             |
                         v                                             v
                P0 Bokeh trace editor                       P1/P2 ProfileSampler
                         |                                             |
                         v                                             v
             lon/lat TXT 或 GeoJSON                         ProfileResult / CSV
~~~

InteractiveWorkspace 是交互任务边界，不是新的科学数据容器。背景数据仍来自现有 loader；
只有用户正在编辑的少量坐标和操作历史可以修改。

### 2.2 后端选择

当前 Plotly/Dash viewer 适合图层显隐、底图切换和属性查看，但 Plotly 的标准 shape
drawing 主要面向笛卡尔坐标轴，不能未经验证就当作 Scattermap 上的完整节点编辑器。
直接把 MapLibre 绘图库嵌入 Plotly 又需要自定义前端组件，P0 维护成本过高。

P0 推荐使用可选 Bokeh 后端：

- PolyDrawTool 可逐点建立连通折线；
- PolyEditTool 可添加、移动和删除节点；
- ColumnDataSource 可把坐标与表格同步；
- Bokeh Server 可在 Python 侧完成保存、校验和状态管理；
- 仍是本地浏览器工具，不要求新增桌面 GUI 框架。

结构化二维观测由 Datashader 根据当前视窗生成连续 RGBA；固定色限来自完整有效场，视窗聚合只服务显示。非结构化输入继续显示有效点。Bokeh 和 Datashader 只属于交互扩展，不替换现有 Plotly/Dash viewer。只有真实案例证明需要更强地图
编辑能力时，才重新评估原生 MapLibre 加 Terra Draw；P0 不引入 Panel、HoloViews、
napari 或自定义 Dash 组件。

### 2.3 两种入口，共用一个编辑核心

推荐最终提供两个入口，但分阶段落地。

独立快捷入口，P0 必做：

~~~text
ecat-trace-edit OBSERVATION.nc --trace published_trace.txt
~~~

不强制 YAML；需要组合多图层时才接受 viewer project YAML。

降采样交接入口，P0 稳定后实施：

~~~text
ecat-downsample -f downsample.yml --edit-trace
~~~

只在 CLI orchestration 层延迟导入交互模块，把当前已经解析、改正并复制出的显示数组交给
编辑器；csiExtend.downsample 数值模块不 import viewer。关闭编辑器后仍由用户明确选择
保存位置，首版不自动改写 fault_traces 配置，也不自动继续生成断层网格。

交接时直接复用同一配置中已启用且适用于 raw stage 的 fault_traces 作为只读 reference，
不再要求用户写一组 editor 专用迹线配置。多条 reference 都可显示，但仍只产生一个
active working path。

GAMMA 等平台原始文件继续由现有 downsampling reader 和配置解析；交互包不再实现 prefix、
incidence、azimuth 或单位协议。标准 H5/NC 等 viewer source 才走独立快捷入口。

独立入口先解决“随手调整迹线”的需求；降采样入口只是减少第二次读取，不形成第二套
状态、格式和编辑逻辑。

## 3. 科学与状态协议

### 3.1 源数据不可变

- LayerPayload 及其数组继续保持 read-only；
- reference trace 只用于显示和复制；
- “从参考迹线开始”必须先产生新的 PathDraft；
- 保存默认采用 Save As，目标存在时拒绝；
- 只有用户显式使用 --overwrite 或确认覆盖时才允许替换目标文件；
- 编辑历史只保存坐标快照，不保存栅格、DataFrame 或完整 figure。

### 3.2 坐标协议

内部权威编辑坐标统一为：

~~~text
CRS: EPSG:4326
order: longitude, latitude
longitude storage: canonical [-180, 180)
~~~

Web Mercator 只属于 Bokeh 显示适配器。每次节点变更都转换回经纬度后写入 PathDraft；
保存文件不得写入 Web Mercator 米坐标。坐标往返使用一个受测入口，不允许 renderer、
表格和导出器各自实现不同公式。

跨日期变更线的显示对齐与 canonical storage 分开。P0 典型区域较小，但测试仍覆盖
0–360 输入规范化和经纬度往返精度。

### 3.3 显示数据与科学采样分离

大幅观测在地图上可以使用 overview 或空间抽样，但：

- overview 只影响显示速度；
- P0 的迹线坐标不吸附到 overview 像元；
- P1 剖面采样重新访问权威源值或完整 detached payload；
- 色限、mask、单位和 corrected/raw 身份沿用 loader metadata；
- 不允许对显示图做截图取值后当作科学剖面。

### 3.4 迹线角色

| 角色 | 是否可编辑 | 建议样式 | 含义 |
| --- | --- | --- | --- |
| reference | 否 | 灰色或浅色虚线，无活动节点 | 发表、已有或粗略迹线 |
| working | 是 | 高对比实线，小型空心节点 | 本次调整中的工作副本 |
| saved | 否，除非重新载入为 working | 实线、无活动节点 | 已显式保存的输出 |

不能只靠颜色表达角色；图层名称、只读标记和工具启用状态同时明确。

## 4. 最小内部对象

### 4.1 PathDraft

建议字段：

~~~text
id                 稳定的本次会话 ID
name               用户可见名称
purpose            trace；预留 profile，但 P0 不实现 profile sampler
coordinates        N x 2 的 canonical lon/lat
source_layer_id    从哪条 reference 复制；新建时为 null
source_file        仅作 provenance，不用于原位写回
dirty              相对最后保存状态是否变化
metadata           少量 JSON-safe 信息
~~~

约束：

- coordinates 必须为有限二维数；
- trace 保存至少需要两个节点；
- 不保存 Bokeh ColumnDataSource、figure 或科学栅格；
- P0 不增加 polygon、point set、multi-part geometry。

### 4.2 EditHistory

历史记录使用有限长度的坐标快照：

- 默认最多 50 个完成操作；
- add、move、delete 在一次手势完成后记一条，不按鼠标移动事件逐帧记录；
- undo 后执行新操作时清空 redo 分支；
- new、copy、clear、delete path 同样进入历史；
- 历史对象不负责文件 I/O。

Bokeh 自带 Undo/Redo 可以作为视图辅助，但科研编辑的权威撤销语义由 EditHistory
测试保证，不能依赖浏览器工具栏内部实现。

`PathDraft.coordinates` 是唯一权威 working geometry。Bokeh working `MultiLine` source
是浏览器几何编辑的唯一提交通道；一次完成的 line event 经统一 commit 转回规范经纬度，
真实变化才写入 workspace、历史和坐标表。PolyEdit 节点 source 只表示当前选中工作线的
临时编辑手柄，不预填为第二份几何，也不注册 Python data commit callback。浏览器手势
提交时不立刻从 Python 回写 working line 或节点 source，避免打断 Bokeh 对共享顶点数组的
更新。New/Copy/Undo/Redo/Delete 等 workspace 命令才把权威状态投影到 working line 和
坐标表。PolyEdit 为取消选择而清空节点 source 时不得清空迹线；selection callback 只
联动节点与表格高亮，不能改变 plot range。

### 4.3 InteractiveWorkspace

只保存 reference 元数据和只读坐标、一个 active PathDraft、EditHistory、当前模式、
输出路径和少量会话消息。

P0 只允许一个 active working path，避免多条线同时编辑造成选择和保存歧义。可以同时
显示多条 reference；用户需要第二条输出时先保存，再新建工作路径。

## 5. P0 交互迹线调整

### 5.1 用户流程

~~~text
打开观测
  -> 加载 reference trace，可为多条
  -> 调整底图、色限、显示范围
  -> New trace 自动进入 Draw，或 Copy as working 自动进入 Edit
  -> Edit 中点击黄色 working trace，显示临时编辑圆点
  -> Draw / Edit
  -> 节点表与地图联动核对
  -> Undo / Redo / Delete / Clear
  -> Validate
  -> Save As lon/lat TXT，可选 GeoJSON
  -> 在 downsample YAML 的 fault_traces.file 中引用输出
~~~

浏览和编辑模式分开。默认进入 Browse，避免平移地图时误加节点；`New trace` 自动进入
Draw，`Copy as working` 自动进入 Edit。进入 Edit 后点击黄色工作线才显示其临时节点；
离开 Edit 时节点隐藏，再次进入后重新选择工作线。节点或表格选择只高亮，不改变当前视窗。

### 5.2 首版必需操作

- 单击增加节点并立即显示相邻连线；
- 完成当前绘制；
- 选中并移动节点；
- 在线段上插入节点；
- 删除选中节点；
- 删除或清空 working path；
- 撤销、重做；
- 从 reference 复制；
- 表格显示节点序号、longitude、latitude；
- 点击表格定位地图节点，点击节点高亮表格行；
- 复制单点或全部坐标；
- Save As 和保存后 dirty 状态提示。

| 操作 | 快捷键 |
| --- | --- |
| Undo | Ctrl+Z |
| Redo | Ctrl+Y 或 Ctrl+Shift+Z |
| 删除选中节点 | Delete 或 Backspace |
| 退出当前绘制 | Esc |
| 完成当前绘制 | Enter |

快捷键只是按钮补充，所有操作都有可见按钮和当前模式提示。

### 5.3 输出协议

默认输出是现有降采样可直接读取的文本：

~~~text
# longitude latitude
-69.10000000 10.20000000
-69.05000000 10.18000000
~~~

要求：

- UTF-8 文本；
- 两列 lon lat，有限数字；
- 默认保留 8 位小数；
- 不自动闭合、重排、平滑或等距加密；
- 注释可记录生成工具和源迹线，但不写本地绝对路径；
- 输出可被 read_trace_file() 按默认 columns: [lon, lat] 读取。

GeoJSON LineString 是可选 companion 输出，便于 viewer、GIS 或 Google Earth 后续转换；
P0 不把 GeoJSON 设为 downsampling 的必需格式，也不同时生成大量格式。

### 5.4 与静态降采样绘图的关系

现有 fault_traces.marker 静态绘制继续保留，适合快速查看原始节点和人工修改文本。
交互编辑器是 opt-in 工具，不替换 Matplotlib 原始或降采样图，也不改变默认结果。

P0 独立入口稳定后，降采样 CLI 的可选交接只负责构造只读背景和打开同一个 editor；
不得复制节点操作、保存或坐标转换代码。默认模板首屏不增加交互块，完整 reference 再
说明高级入口，避免普通降采样用户承担新概念。

## 6. 建议代码组织

~~~text
eqtools/map_viewer/interactive/
  __init__.py            lazy public exports，不导入 Bokeh
  models.py              PathDraft、EditHistory、InteractiveWorkspace
  coordinates.py         EPSG:4326 与 Web Mercator 的唯一转换入口
  trace_io.py            TXT/GeoJSON read-copy-save-as
  display.py             显示倍率、完整场色限、Matplotlib 色表方向
  raster.py              Datashader 曲线网格、视窗栅格帧
  bokeh_trace_editor.py  视图、tool adapter 和 server callback
  cli.py                 ecat-trace-edit
~~~

职责约束：

- models.py 和 trace_io.py 不 import Bokeh、Dash 或 Plotly；
- bokeh_trace_editor.py 只把模型映射到 Bokeh，不重新读取科学格式；
- 背景观测继续通过 LayerCatalog/loaders.py 获得；
- 两列迹线协议与 csiExtend.downsample.fault_inputs.read_trace_file() 对齐；
- 不让 map_viewer.loaders import csiExtend.downsample；通过纯 I/O helper 或交叉测试
  维持共享协议；
- downsampling 的可选交接只允许出现在 CLI orchestration，且使用 lazy import；
- P1 到来前不创建空的 samplers 插件树。

P1 实现时再增加 profile_models.py、profile_sampling.py 和 bokeh_profile_view.py。
先使用显式 source kind 映射，不建立 plugin registry；只有外部 kind 真正需要独立扩展
时才评估注册机制。

## 7. 可选依赖和环境策略

~~~text
.[viewer]       Dash + Plotly，只读地图
.[interaction]  Bokeh + 交互所需的明确数据和坐标依赖
~~~

实际约束：

~~~text
bokeh>=3.6,<4
datashader>=0.19,<0.20
cmcrameri>=1.8
matplotlib>=3.6
~~~

P0 的 Web Mercator 只在一个受测的轻量公式适配器中使用，因此没有额外引入 pyproj。
NumPy、Pandas、Xarray、Rasterio、h5netcdf 按实际 import 明确列入，不能依赖偶然的
传递安装。两个 extras 重复声明共同依赖可接受，pip 会合并；暂不增加 viewer-all。

实施前先在 cutde 环境副本中验证依赖求解和最小启动；确认不升级现有 NumPy/Pandas 等
数值栈后，再在日常环境安装。没有安装 interaction 时：

- import eqtools、反演和 ecat-downsample 行为不变；
- ecat-map 仍可按 viewer extra 工作；
- 只有运行 ecat-trace-edit 才给出清楚的可选依赖安装提示。

## 8. P1/P2 多源剖面框架

### 8.1 共享剖面几何

同一个 PathDraft 以后以 purpose=profile 使用。剖面可为两点直线或多段折线，统一产生：

~~~text
s            沿剖面累计距离，km
n            点到剖面的有符号垂距，km
longitude
latitude
segment_id
~~~

距离与垂距使用明确的 geodesic 或局部投影计算；不能直接把经纬度差当作平面距离。路径
编辑器只提供几何，采样器根据 source kind 决定如何使用 s/n。

### 8.2 ProfileRequest

~~~text
path                    canonical lon/lat polyline
source_layer_ids        明确选择的数据层
sampling_mode           由 source kind 校验
spacing_km              网格沿线采样间距
corridor_half_width_km  点数据或统计剖面走廊半宽
statistic               显式统计量；不提供时不猜测
component               GNSS parallel/perpendicular/up 等
~~~

不能用一个含混的 method 同时表示栅格插值、GNSS 分量和地震统计。

### 8.3 各数据类型的科学语义

| 数据 | 首选剖面语义 | 首版默认 | 不能静默做什么 |
| --- | --- | --- | --- |
| 规则观测网格 | 沿线标量值 | nearest；bilinear 显式可选 | 用 overview 代替原值 |
| 二维曲线网格 | 基于二维 lon/lat mesh 查询 | nearest 或 KD-tree | 分别对一维 lon、lat 插值 |
| CSI varres 或散点 | 距离或半径内观测 | nearest 或明确统计 | 伪装成规则栅格 |
| GNSS | 走廊内站点、平行或垂直分量 | 原站点散点 | 默认栅格化或丢失 ENU |
| 地震目录 | 距离—深度散点及走廊筛选 | 原事件散点 | 插值成连续曲线 |

GNSS 保存原始 east/north/up，再按剖面局部方向派生 parallel/perpendicular；派生列标注
方向和单位。地震剖面至少保留 event id、time、magnitude、depth、s/n，便于回到地图。

### 8.4 结果与联动

ProfileResult 是 detached、只读且带 provenance 的结果，记录 source_layer_id、变量、
单位、mask、采样方式、路径和数据表，不修改 source payload。

以后支持地图和剖面图联动。只在完成移动节点后重新计算，不能随每个鼠标移动事件重采样
大网格。首选导出 CSV，并另存 path 的 TXT 或 GeoJSON；只有多变量规则剖面确有需求时
才增加 NetCDF。

## 9. 分阶段实施

### P0-A：模型、坐标和 I/O（已实施）

- 建立 PathDraft、EditHistory、InteractiveWorkspace；
- 建立唯一的经纬度和 Web Mercator 转换；
- 读取参考 TXT/GeoJSON 并创建 working copy；
- Save As 默认两列 TXT，可选 GeoJSON；
- 建立纯 Python 单元测试。

完成标准：无需浏览器即可验证节点操作、历史、坐标往返和输出兼容性。

### P0-B：Bokeh 迹线编辑器（已实施）

- 显示一个 observation layer 和多条 reference；
- Draw、Edit、Browse 三种明确模式；
- 节点表、按钮、快捷键和 dirty 状态；
- reference 和 working 样式区分；
- 独立 ecat-trace-edit 入口和可选依赖提示。

完成标准：真实 SAR 或光学观测上可连续画线、移动节点、回撤并得到现有 downsampling
可读输出，原文件 hash 不变。

### P0-C：降采样可选交接（已实施）

当前实现：

- CLI 参数显式启用，不改变默认模板和默认执行；
- 复用已加载的只读显示数据，不重新解释平台格式；
- editor 关闭后不自动改配置、不自动覆盖迹线；
- 验证启用与不启用时的降采样数值结果完全一致。

### P0-D：连续观测与显示控制（已实施）

- 二维有限坐标网格保留完整 row/column 对应，以 Datashader quadmesh 动态绘制当前视窗；
- 缩放期间色表和色限保持固定，不以视窗数据重新定标；
- 灰白、街道、地形、卫星和无底图可切换，观测、底图、colorbar 可独立显隐；
- 透明度只作用显示 renderer；源数值、坐标、mask、输出和后续计算不变；
- 非二维或含非有限坐标的输入回退到有效点，不猜测缺失的网格坐标；
- 当前仍是一幅观测、多条 reference 和一条 working path，不借显示优化引入多影像状态。

### P1：观测剖面

先支持标准 observation grid 和 CSI varres；实现原值采样与 overview 分离、地图联动和
CSV 导出；用合成梯度、曲线网格和 NaN/mask 校验位置与数值。

### P2：GNSS 与地震目录剖面

实现 GNSS 走廊筛选及平行、垂直分量，地震距离—深度散点、走廊过滤和事件回查。多个
source 可以同图显示，但保持各自单位和采样语义。

### P3：真实需求驱动

以后再评估原生 MapLibre 或 Terra Draw、自动辅助吸附、显式平滑、等距重采样、NetCDF
结果和多工作路径管理。不能因为框架可扩展就提前实现。

### P0 实际代码映射

| 职责 | 实现位置 |
| --- | --- |
| 状态、历史、reference/working 分离 | `interactive/models.py` |
| 坐标转换 | `interactive/coordinates.py` |
| TXT/GMT/GeoJSON I/O 与 Save As | `interactive/trace_io.py` |
| 既有 viewer/downsample 数据适配 | `interactive/adapters.py` |
| 显示倍率、固定全场色限和色表方向 | `interactive/display.py` |
| 二维曲线网格和当前视窗 RGBA | `interactive/raster.py` |
| Bokeh UI、显示控制、按钮、快捷键和本地 server | `interactive/bokeh_trace_editor.py` |
| 独立入口 | `interactive/cli.py`、`ecat-trace-edit` |
| 降采样交接 | `cli_tools/process_data_downsampling.py --edit-trace` |

实现仍只允许一条 active working polyline；P1/P2 没有借 P0 提前加入。

### P0/P0-D 校验记录（2026-08-04）

- 仓库固定收集范围（`tests` 与 `eqtools/csiExtend/tests`）：1696 passed；warning 主要来自既有 `numpy.matrix`、稀疏 mesh 提示和测试函数返回值，不是本轮交互失败；
- 公开测试全集：536 passed；两个 warning 分别来自无地理参考测试栅格和既有 Matplotlib tick 计算，不是交互失败；
- 交互模型、CLI、Bokeh、Datashader、降采样交接聚焦回归：隔离依赖环境 41 passed；
- 文档、模板、降采样 CLI、无可选依赖回退交叉回归：174 passed、1 skipped；
- 二维曲线网格完整场色限、CMCrAmeri 色表方向、当前视窗重绘、南北空间方向、无效坐标回退和源数组不变均有独立测试；
- 本地 Bokeh Server 在临时端口启动、连续栅格页面 HTTP 200 响应和关闭通过；
- `pip install --dry-run ".[interaction]"` 只新增 Datashader、CMCrAmeri 及其轻量依赖，未要求升级当前 NumPy、Pandas、Xarray、SciPy、Numba、Rasterio 或 h5netcdf；
- 公开 Markdown 相对链接、可复制 YAML 和 `git diff --check` 通过。

公开全集包含聚焦回归的大部分用例，各组计数不能相加解释为独立测试总数。本轮没有修改 `eqtools/csiExtend` 数值实现，因此未重复运行其全包科学慢测。

## 10. 测试与验收

### 10.1 P0 必测

- 新建迹线、复制 reference、清空和删除；
- 节点 add、move、insert、delete 的坐标与顺序；
- PolyEdit 临时节点 source 独立变化不修改 workspace、工作线、坐标表或历史；
- working line 是唯一浏览器几何提交通道，完成的变化只增加一条历史；
- workspace 命令只向 working line 和坐标表投影权威状态；
- Edit 模式中选择工作线才显示节点，离开 Edit 后节点 source 可清空而不删除迹线；
- 删除中间节点后按原顺序重连；
- 每类操作的 undo/redo 和 redo 分支清除；
- 浏览模式平移地图不会增加节点，选择节点或表格行不会改变 plot range；
- TXT/GeoJSON 读取—保存—重读一致；
- 默认输出能被 read_trace_file() 直接读取；
- 已存在输出默认拒绝覆盖；
- reference 源文件和 observation fingerprint 不变；
- LayerPayload 数组仍不可写；
- Web Mercator 往返误差满足编辑精度；
- 0–360 经度规范化不改变空间位置；
- 缺少 Bokeh 时核心模块可导入、CLI 提示准确；
- 大网格 overview 不影响保存坐标；
- 二维规则/曲线网格连续显示，缩放后固定全场色限且源数组不变；
- 非结构化或非有限坐标回退为有效点，不静默猜测网格；
- 底图、观测、colorbar 显隐与透明度只改变 renderer；
- Windows 路径、中文文件名和 UTF-8 注释可用。

### 10.2 P0-C 数值零回退

对相同配置比较不启用 editor 与启用但不修改迹线两种运行。逐项比较载入后的观测值、
projection、mask、降采样中心、输出值和协方差结果；交互层不能改变数组、随机种子或
计算顺序。只验证“图能打开”不足以接受合并。

### 10.3 P1/P2 后续科学测试

- 规则网格解析解剖面；
- 二维曲线坐标网格无一维经纬度错配；
- NaN/mask、边界和走廊空数据；
- 折线跨 segment 的累计距离连续性；
- GNSS 已知 EN 向量投影到 parallel/perpendicular；
- 地震目录 s/n/depth 与独立几何计算对照；
- display factor、color limits 或 overview 不改变科学剖面。

浏览器测试只覆盖关键点击序列；几何、I/O 和采样准确性主要由无头纯 Python 测试保证。

## 11. 文档分层

P0 已建立对应公开文档：

1. workflow：`docs/workflows/02c_interactive_trace_editing.md`；
2. example：`docs/examples/interactive_trace_editing.md`；
3. reference：`docs/reference/interactive_trace_editor.md`；
4. developer：本计划与 `DEVELOPER_GUIDE.md` 的状态、适配器和测试边界。

公开 workflow 直接展示最常用命令，不强迫普通用户先写 project YAML；高级多图层项目再
链接到 viewer reference。可选依赖说明明确 viewer 与 interaction 的职责。

## 12. 实施决策检查表

- [x] P0 只编辑 polyline，没有顺手加入 polygon、freehand 或 profile sampling；
- [x] reference 和 working 在模型、样式和保存策略上都分离；
- [x] 默认输出直接兼容现有 fault_traces.file；
- [x] ecat-map 和 downsampling 数值核心仍不依赖 Bokeh；
- [x] 坐标转换只有一个权威入口并有往返测试；
- [x] overview 与未来科学采样明确分离；
- [x] 默认 Save As，没有静默覆盖或配置写回；
- [x] 先完成独立入口，再增加 downsampling 交接；
- [x] 连续栅格是显示缓存，不进入保存、导出或科学计算；
- [x] 一般用户可在界面切换底图、图层显隐和透明度，无需新增 YAML；
- [ ] P1 按数据类型定义采样语义，而不是一个万能 method；
- [ ] 新增高级能力都有真实案例证据。

## 13. 最终落地顺序

1. [已完成] 用独立、可选的 Bokeh editor 完成迹线 working-copy 编辑；
2. [已完成] 输出保持简单的 lon lat 文本，立即服务现有降采样；
3. [已完成] 保留 Plotly/Dash viewer 的只读多图层职责；
4. [已完成] 以显式 `--edit-trace` 接到 downsampling CLI；
5. [已完成] 以固定全场色标提供连续二维观测、底图切换和显示控制；
6. [后续] 复用 PathDraft 建立剖面路径，但分别实现观测、GNSS 和地震目录采样。

这条路线先解决当前最高频的问题，同时不会把首版扩展成难以维护的通用 GIS 平台。
