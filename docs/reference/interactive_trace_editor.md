# 交互迹线编辑器参考

本页是 `ecat-trace-edit`、降采样交接入口、输入输出格式和交互契约的完整查阅层。第一次使用先读
[交互调整断层迹线](../workflows/02c_interactive_trace_editing.md)。

## 阅读路径

- 只想复制命令：看 [交互迹线调整短例](../examples/interactive_trace_editing.md)。
- 原始 GAMMA/GMTSAR/HyP3/光学输入：看 [降采样交接](#降采样交接)。
- 已有标准 `.nc/.h5` 或 GeoTIFF：看 [独立 CLI](#独立-cli)。
- 不确定什么会被修改：看 [状态与科学边界](#状态与科学边界)。
- 不确定坐标或保存格式：看 [坐标协议](#坐标协议) 和 [输出协议](#输出协议)。
- 已知经度、纬度、最近点或沿线距离，只需确定性裁剪/延长：改用 [断层迹线处理](fault_trace_processing.md)。

## 安装与职责

```bash
cd eqtools
python -m pip install -e ".[interaction]"
```

第一行假定当前位于 ECAT 仓库根目录；独立 eqtools 开发仓库中已经位于项目根目录，
只执行第二行。

`interaction` extra 使用 `bokeh>=3.6,<4` 和 `datashader>=0.19,<0.20`，并提供 Matplotlib/CMCrAmeri 色表；这些依赖只在启动编辑器时延迟导入。`ecat-map` 仍是 Plotly/Dash 只读多图层查看器；`ecat-trace-edit` 只负责一条 working polyline 的调整。当前不实现剖面采样、自由手绘、polygon、自动吸附、自动平滑或多观测联合编辑。

## 独立 CLI

```text
ecat-trace-edit OBSERVATION
  [--kind observation_grid|raster|csi_varres]
  [--variable NAME]
  [--mask source_valid|analysis_valid|finite]
  [--data-type sar|optical]
  [--trace PATH ...]
  [--output PATH]
  [--title TEXT]
  [--cmap NAME]
  [--vmin VALUE --vmax VALUE]
  [--auto-percentile PERCENT]
  [--display-factor VALUE]
  [--display-unit UNIT]
  [--no-symmetry]
  [--basemap gray|street|terrain|satellite|none]
  [--opacity VALUE]
  [--host HOST] [--port PORT] [--no-browser]
```

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `OBSERVATION` | 必填 | 标准 `.nc/.h5/.hdf5`、有地理参考的 `.tif/.tiff`，或显式 `csi_varres` 输入 |
| `--kind` | 按扩展名推断 | 文本 varres 必须显式指定，避免误判普通文本 |
| `--variable` | 网格 `observation`；栅格 `band_1` | 显示变量；改正后网格常用 `corrected_observation` |
| `--mask` | `source_valid` | 标准观测网格的显示 mask |
| `--data-type` | `sar` | 只解释 CSI varres 行合同 |
| `--trace` | 空 | 只读 TXT/GMT/GeoJSON 参考迹线；可重复 |
| `--output` | `adjusted_trace.txt` | 界面初始 Save As 目标，不会在启动时自动写文件 |
| `--cmap` | `RdBu_r` | Matplotlib/CMCrAmeri 色表名；数值由低到高严格按色表方向映射 |
| `--vmin/--vmax` | 自动 | 必须成对提供；显式值优先且只影响显示 |
| `--auto-percentile` | `99` | 未显式给色限时，使用完整有效场的中间 99% 求范围 |
| `--display-factor` | `1` | 显示值倍率，不改变观测和保存迹线 |
| `--no-symmetry` | false | 自动色限不强制围绕 0 对称 |
| `--basemap` | `gray` | 初始灰白、街道、地形、卫星或无底图；界面中仍可切换 |
| `--opacity` | `0.82` | 初始观测透明度，范围 `[0, 1]`；界面中仍可调整 |
| `--host/--port` | `127.0.0.1:5006` | 本地 Bokeh 服务；默认不暴露到网络 |
| `--no-browser` | false | 只启动服务，不自动打开默认浏览器 |

模块入口等价：

```bash
python -m eqtools.map_viewer.interactive scene_observation.nc --trace trace.txt
```

## 降采样交接

```text
ecat-downsample -f CONFIG --edit-trace
  [--trace-output PATH]
  [--trace-component NAME]
  [--vmin VALUE --vmax VALUE]
```

交接发生在 CLI orchestration 层：先由现有 reader 解释平台文件，再应用配置中已启用的整周、参考区或 ramp 改正，最后把分离的只读 `ObservationGrid` 交给同一个编辑器。它不会让 downsampling 数值模块 import Bokeh。

规则：

- `--edit-trace` 是检查模式，本次关闭 `-c/-d`，不自动继续协方差或降采样；
- SAR 默认 component 为 `observation`，光学默认 `east`；
- `fault_traces` 中启用且 `stages` 包含 `raw` 的项作为只读 reference；多段文件按段展开，
  可用 `segments` 选择显示段；
- `--trace-output` 只设置 Save As 初值，不代表文件一定已经保存；
- 不写回 YAML，不自动把输出替换为活动迹线；
- `--sar-prefix` 可与 `--edit-trace` 一起用于 GAMMA 快捷读取，但不支持 `-c/-d`。

## 显示合同和图层控制

结构化二维观测使用完整的二维 `longitude/latitude/value/mask` 建立曲线网格。显示时逐点转换为 Web Mercator，并由 Datashader 针对当前视窗生成连续 RGBA 图像；缩放较远时，同一显示像素内的多个源像元只做显示用 mean 聚合。该聚合不会回写源数组，也不用于保存迹线、导出观测或后续科学计算。

色表和色限在启动时由完整有效场计算一次，缩放不会重新定标。默认 `RdBu_r` 与降采样常用 `cmc.roma_r` 都保持低值/负值为蓝、高值/正值为红；降采样交接继承 `check_plots.raw` 的 `cmap`、`factor4plot`、`symmetry`、`auto_percentile` 和 `vmin/vmax`，命令行显式色限优先。

当输入不是二维结构化网格，或二维坐标本身包含非有限值时，编辑器回退为有效点显示，不猜测缺失网格坐标。两种路径都只使用有限、有效观测；最多 80,000 个精确点用于 hover 或散点回退，不影响连续栅格的完整数据范围。

`Display` 面板提供：

- 灰白、街道、地形、卫星或无底图；在线底图需要网络；
- 独立显隐底图、观测和 colorbar；
- 观测透明度调整，便于同时辨认影像梯度、地形和参考迹线。

当前一次会话只显示一幅观测和多条只读 reference，不提供多影像叠加编辑。

## 参考迹线格式

支持：

- TXT/DAT/TRACE：每行至少含经纬度列，默认前两列为 `lon lat`，额外列忽略；
- CSI/GMT 风格分段文本：`>` 开始新线段；
- GeoJSON：`LineString` 或 `MultiLineString`，也可放在 `FeatureCollection` 中。

GMT/OGR 文件可保留 `#` 元数据头和可选第三列，例如 `lon lat z`；二维迹线只读取配置指定的
经纬度列。通过降采样配置进入时，省略 `segments` 会把所有段分别列为 reference；写
`segments: [0, 2]` 可只加载零基索引 0 和 2。独立 CLI 的每个 `--trace` 文件同样按段展开。

每条 reference 在模型中不可变。`Copy reference as working` 创建所选段的坐标副本，随后编辑不会改变参考数组或源文件。

## 交互与快捷键

| 模式/操作 | 含义 |
| --- | --- |
| `Browse` | 平移、缩放和检查，不增加或移动节点 |
| `New trace` | 新建空 working trace，并自动进入 `Draw` |
| `Copy reference as working` | 复制所选 reference，并自动进入 `Edit`；再点击黄色 working trace 显示编辑圆点，reference 本身不变 |
| `Draw` | 建立一条连续 working trace；只允许一条活动线 |
| `Edit` | 先点击黄色 working trace 显示临时圆点，再拖动、在线段中插点或选择待删除节点 |
| `Delete selected vertex` | 删除地图或坐标表中选中的节点；剩余节点按原顺序重新连线 |
| `Finish drawing / Browse` | 明确结束当前绘制并回到浏览模式 |
| `Clear vertices` | 清空 working 坐标，可 Undo |
| `Delete working trace` | 删除本次 working copy，不影响 reference |
| `Validate` | 检查至少两个有限经纬度节点 |
| `Save As` | 写入新目标；默认拒绝覆盖 |

| 快捷键 | 操作 |
| --- | --- |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` 或 `Ctrl+Shift+Z` | Redo |
| `Delete` | 删除选中节点，与 `Delete selected vertex` 使用同一操作 |
| `Esc` 或 `Enter` | 完成当前绘制并回到 Browse |

圆点只属于 `Edit` 中当前选中的黄色 working trace：离开 `Edit` 时会隐藏，再次进入后重新点击黄色线即可。节点表和临时圆点共享选择。点击节点或表格行只同步高亮，不自动平移或缩放视图；需要查看局部时由用户在 `Browse` 中自行缩放。一次拖动、插入或删除完成后，工作线的坐标事件写回 working state，坐标表和 Undo 历史随之更新。`Copy selected coordinate` 和 `Copy all coordinates` 写入系统剪贴板；浏览器拒绝剪贴板权限时，界面会显示错误，坐标仍可从表格读取。

## 坐标协议

权威编辑和保存坐标为：

```text
CRS: EPSG:4326
order: longitude, latitude
longitude storage: [-180, 180)
```

Web Mercator 米坐标只用于 Bokeh 底图与观测显示，每次浏览器编辑后立即转换回经纬度
并更新 working trace。纬度超出 Web Mercator 可显示范围时明确报错。连续栅格重绘和
精确点抽样都不吸附、不重投影保存结果，也不改变迹线坐标。

## 输出协议

默认文本：

```text
# ECAT adjusted trace
# name: Adjusted trace
# longitude latitude
-69.10000000 10.20000000
-69.05000000 10.18000000
```

该文件可由 `fault_traces.file` 和默认 `columns: [lon, lat]` 直接读取。输出不自动闭合、重排、平滑、等距加密，也不记录输入文件绝对路径。扩展名为 `.geojson/.json` 时写一个 `LineString`；`.gmt` 当前仍写单段两列文本。

覆盖规则：

- 默认使用独占创建，目标存在即报错；
- 只有界面中显式启用 `Allow explicit overwrite` 才允许覆盖目标；
- reference 的原路径从不作为隐式写回目标。

## 状态与科学边界

```text
观测 reader / correction
  -> 只读观测背景和参考迹线
  -> 一条可编辑 working trace
  -> 显式 Save As
```

编辑器只允许一条可变的 working trace；参考迹线和观测背景始终只读。节点圆点和
坐标表只是当前 working trace 的编辑视图，不会修改参考文件。Undo 历史只保存少量
working 坐标快照，不复制观测网格。结构化二维背景按当前视窗生成显示帧，80,000 点
上限只用于 hover 或非结构化显示回退；色限仍由完整有效值计算。保存坐标与显示栅格
或抽样无关，编辑器也不会重新解释 LOS、offset、projection、单位或正号。

## 常见错误

- “requires the optional interaction dependencies”：进入 ECAT 的 `eqtools` 子目录，
  安装 `.[interaction]`。
- “requires Bokeh >=3.6,<4”：当前 Bokeh 不在已测试范围，使用 interaction extra 的版本约束。
- “Continuous trace-editor imagery requires Datashader”：在 eqtools 项目根目录重新
  安装 `.[interaction]`；非结构化输入仍可回退为点显示，但结构化连续图需要受测版本。
- “Set both --vmin and --vmax”：同时提供上下限，或两者都省略。
- 输出已存在：更换 Save As 文件名；确认确需替换时再显式允许覆盖。
- reference 不显示：检查每个所选段至少有两个有限经纬度节点；降采样交接还要检查
  `enabled`、`stages: [raw]` 和零基 `segments` 是否在范围内。
- 端口占用：独立 CLI 用 `--port` 更换端口；降采样交接关闭占用 `5006` 的旧编辑器后重试。

相关页面：

- [InSAR 降采样](../workflows/02_insar_downsampling.md)
- [本地科研地图查看](../workflows/07_research_map_viewer.md)
- [SAR 与光学观测读入脚本](observation_data_readers.md)
- [Downsampling App](downsampling_app.md)
