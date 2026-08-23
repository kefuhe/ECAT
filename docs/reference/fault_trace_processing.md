# Fault Trace Processing Reference / 断层迹线处理参考

本页是 `ecat-fault-trace-tool`、`TracePath`、TraceMarker、纯函数和迹线文件协议的完整查阅层。
第一次使用先看[预处理短例](../examples/fault_trace_processing.md)，需要按步骤检查输入和输出时看
[预处理 workflow](../workflows/02d_fault_trace_preprocessing.md)。

## 阅读路径

- 只想裁到指定经度、纬度或最近点：看 [trim 与 marker](#trim-markers)。
- 想连续执行多步：看 [apply 配置](#apply-config)。
- 想在 Python 中处理：看 [Python API](#python-api)。
- 不确定该不该修改 fault object：看 [状态边界](#fault-boundary)。
- 遇到多交点、跨日界线或最近点过远：看 [定位语义](#marker-contract)。
- 维护 CLI 或增加算法：看 [实现分层](#architecture)。

## 输入输出协议

输入支持：

| 格式 | 内容 |
| --- | --- |
| `.txt/.dat/.trace` | 至少含经度和纬度列；默认前两列，额外列忽略；空行和 `#` 注释忽略 |
| `.gmt` | `>` 分隔的一段或多段迹线；支持 GMT/OGR 元数据头及可选第三列 |
| `.json/.geojson` | `LineString`、`MultiLineString`、Feature 或 FeatureCollection |

多段输入不自动拼接；用 `--segment INDEX` 选一段。输出为单段 TXT/GMT 或 GeoJSON
`LineString`。所有新子命令默认使用独占创建，目标存在时报错；只有显式 `--overwrite` 才覆盖。

文本输出示例：

```text
# ECAT processed trace
# name: fault_trace_trimmed
# longitude latitude
101.10000000 24.10000000
101.25000000 24.18000000
```

## CLI 总览

```text
ecat-fault-trace-tool COMMAND INPUT [projection options] [command options]
```

| 命令 | 作用 | 是否写迹线 |
| --- | --- | --- |
| `inspect` | 点数、长度、端点、投影中心 | 否 |
| `locate` | 解析并预览 TraceMarker | 否 |
| `clean` | 删除连续重复或过近节点 | 是 |
| `orient` | 让首点位于 west/east/south/north 一侧 | 是 |
| `reverse` | 反转点序 | 是 |
| `trim` | 按两个 marker 的沿迹线距离保留区间 | 是 |
| `extend` | 沿端部切线延长 | 是 |
| `resample` | 按弧长间距或点数重采样 | 是 |
| `simplify` | RDP 或 VW 减点 | 是 |
| `smooth` | B-Spline 或 Savitzky–Golay 平滑 | 是 |
| `apply` | 按 YAML 顺序执行多步 | 是 |

所有命令都可用 `--lon0/--lat0/--utmzone/--ellps` 控制投影。未指定中心时按输入经度的环形
平均值和纬度平均值自动选择，避免简单算术平均在 ±180° 附近得到错误中心。

写输出的命令共同支持：

```text
-o/--output PATH
--overwrite
--plot [PNG]
--report REPORT.json
```

`--plot` 不给路径时使用输出文件同名 PNG；`--report` 记录投影、处理前后点数/长度和操作参数。

<a id="trim-markers"></a>

## trim 与 marker

`trim` 的底层数值操作只接受两个沿迹线距离；CLI 和 `TracePath` 会先把以下 marker 解析成
`trace_distance_km`：

| CLI | Python marker | 含义 |
| --- | --- | --- |
| `--start-km 20` | `{"trace_distance_km": 20}` | 从输入首点沿折线累计 20 km |
| `--end-fraction 0.8` | `{"fraction": 0.8}` | 总长度的 80% |
| `--end-lon 101.5` | `{"longitude": 101.5}` | 与目标经度线的交点 |
| `--end-lat 24.0` | `{"latitude": 24.0}` | 与目标纬度线的交点 |
| `--end-nearest 101.5 24.0` | `{"nearest": [101.5, 24.0]}` | 指定点在折线上的最近投影 |

`start` 留空表示原始首点，`end` 留空表示原始末点。至少应给出一端：

```bash
ecat-fault-trace-tool trim trace.txt \
  --start-fraction 0.1 \
  --end-lon 101.5 \
  --end-which last \
  -o segment.txt
```

<a id="marker-contract"></a>

## TraceMarker 定位语义

解析后的 `TraceMarker` 包含：

```text
lon, lat
x, y
trace_distance_km
segment_index
segment_fraction
distance_to_trace_km
method
candidate_count
candidate_index
```

重要规则：

- 所有 `trace_distance_km` 都沿局部投影折线累计，不是单独的经差、纬差或测站直线距离。
- 经纬度交点会落到输入折线段内部，不吸附到最近节点。
- 最近点查询投影到整条折线，`distance_to_trace_km` 表示查询点到迹线的距离。
- 最近点可设置 `max_distance_km`；超出后报错而不是静默吸附。
- 同一经纬线可能有多个交点；默认 `first`，也可用 `last` 或零基整数索引。
- 交点正好位于两个相邻线段的公共节点时只返回一个候选。
- 一段迹线与目标经纬线重合时，重合段两端都是候选；用 `first/last` 明确取端点。
- 经度比较会对齐到目标经度分支，支持跨 ±180° 的局部迹线。

预览所有候选：

```bash
ecat-fault-trace-tool locate trace.txt --lon 101.5 --json
```

最近点距离阈值：

```bash
ecat-fault-trace-tool locate trace.txt \
  --nearest 101.5 24.0 \
  --max-snap-km 5
```

## 其他单步命令

### clean

```bash
ecat-fault-trace-tool clean trace.txt --atol-km 0.001 -o cleaned.txt
```

只删除连续重复或间距不超过阈值的节点；不会删除较远位置出现的同坐标点。

### orient 和 reverse

```bash
ecat-fault-trace-tool orient trace.txt --start west -o oriented.txt
ecat-fault-trace-tool reverse trace.txt -o reversed.txt
```

`orient` 只比较首尾点对应坐标，必要时整体反转。点序定义 strike 正方向，改变点序后应重新核对
下倾方向约定。

### extend

```bash
ecat-fault-trace-tool extend trace.txt \
  --start-km 2 --end-km 5 \
  --tangent-window 3 \
  -o extended.txt
```

`start` 从首点反向延长，`end` 从末点正向延长。`tangent_window=1` 使用端部第一/最后一段；更大值
使用跨多个节点的端部弦方向。长度必须非负，且至少一端大于零。

### resample

```bash
ecat-fault-trace-tool resample trace.txt --every-km 1 -o sampled.txt
ecat-fault-trace-tool resample trace.txt --num-points 51 -o sampled.txt
```

二者只能选一个，默认保留两个端点。

### simplify

```bash
ecat-fault-trace-tool simplify trace.txt \
  --method rdp --tolerance 0.2 \
  -o simplified.txt
```

| 方法 | `--tolerance` 单位 | 特点 |
| --- | --- | --- |
| `rdp` | km | 控制垂直偏离，适合保留明显折角 |
| `vw` | km² | 控制有效三角面积，适合自然折线减点 |

两者都保留首尾点。

### smooth

```bash
ecat-fault-trace-tool smooth trace.txt \
  --method bspline --smoothing 2 \
  --num-points 100 \
  -o smoothed.txt

ecat-fault-trace-tool smooth trace.txt \
  --method savgol --window 7 --polyorder 2 \
  -o smoothed.txt
```

默认保留首尾点；只有显式 `--move-endpoints` 才允许平滑改变端点。失败会返回非零状态，不会伪装成
成功并写回原迹线。

<a id="apply-config"></a>

## apply 配置

```yaml
operations:
  - op: orient
    start: west
  - op: trim
    start: {fraction: 0.05}
    end: {longitude: 101.5, which: last}
  - op: extend
    end_km: 5.0
    tangent_window: 3
  - op: simplify
    method: vw
    tolerance: 0.2
  - op: resample
    every_km: 1.0
```

```bash
ecat-fault-trace-tool apply trace.txt \
  --config trace_operations.yml \
  -o prepared.txt \
  --report prepared.json
```

操作严格按列表顺序执行。marker 总是相对于到达该步骤时的当前迹线解析；例如先 `orient` 后
`trim` 会改变 `first/last` 和沿迹线距离的起点。

<a id="python-api"></a>

## Python API

### 纯函数层

```python
from eqtools.csiExtend import (
    clean_trace,
    extend_trace,
    orient_trace,
    resample_trace,
    simplify_trace,
    smooth_trace,
    trace_length,
    trim_trace,
)
```

这些函数接受投影后的 `x/y` 数组，不读文件、不投影、不修改输入数组。它们是 CLI、`TracePath`
和其他 eqtools 模块共用的唯一数值内核。

### TracePath 高层接口

```python
from eqtools.csiExtend import TracePath

trace = TracePath.from_lonlat(
    trace_lonlat,
    lon0=101.5,
    lat0=24.0,
)

prepared = (
    trace
    .orient(start="west")
    .trim(
        start={"fraction": 0.05},
        end={"nearest": [101.8, 24.2], "max_distance_km": 5.0},
    )
    .extend(end_km=5.0, tangent_window=3)
    .resample(every_km=1.0)
)
```

每个方法返回新对象。常用属性：

```python
prepared.xy
prepared.lonlat
prepared.point_count
prepared.length_km
prepared.history
prepared.report()
```

已有 CSI fault 时可只借用其投影：

```python
trace = TracePath.from_lonlat(trace_lonlat, projection=fault)
```

这不会读取或修改 fault 的 `xf/yf`、`xi/yi`、top/bottom、mesh 或 patch。

### 有序函数入口

```python
from eqtools.csiExtend import process_trace

prepared = process_trace(
    trace,
    [
        {"op": "orient", "start": "west"},
        {"op": "trim", "end": {"longitude": 101.5}},
        {"op": "resample", "every_km": 1.0},
    ],
)
```

### 共享文件 I/O

```python
from eqtools.csiExtend import read_trace, read_trace_segments, write_trace

segment = read_trace("fault_trace.txt")
write_trace("prepared.txt", prepared.lonlat)
```

`read_trace()` 要求恰好一段；多段文件用 `read_trace_segments()` 或传 `segment=INDEX`。文本默认
读取前两列，也可用 `columns=["lat", "lon", "z"]` 指定列位置；未被指定为经纬度的额外列不进入
二维迹线计算。

<a id="fault-boundary"></a>

## 与 fault object 的状态边界

迹线预处理发生在 fault 几何构建之前。处理完成后再显式写入：

```python
fault.trace(prepared.lonlat[:, 0], prepared.lonlat[:, 1])
fault.discretize_trace(every=1.0)
fault.set_top_coords_from_trace(discretized=True)
```

不提供 `fault.trim_trace()`、`fault.extend_trace()` 之类的隐式修改方法。原因是更改 `xf/yf` 后，
已有 `xi/yi/loni/lati`、top/bottom/layers、mesh/patch、GF、Laplacian 和约束可能全部失效。
如果必须替换已有 fault 的 trace，应把后续派生状态全部重新构建。

<a id="architecture"></a>

## 实现分层

```text
trace_ops.py -> trace_markers.py -> trace_processing.py
纯 x/y 算法    定位语义             TracePath、投影和有序操作

trace_io.py
TXT/GMT/GeoJSON 的唯一分段折线协议
       ↓
fault_trace_tool.py / interactive editor / downsample fault_inputs.py
CLI 与编辑器模型适配 / 显示多段或为计算显式选择单段

patch_indices.py -> trace_markers.py
fault 与 patch 选择适配
```

增加新操作时先实现纯函数和单元测试，再决定是否加入 `TracePath.apply()` 和 CLI。patch selector
继续通过兼容 `resolve_trace_marker(fault, ...)` 入口调用同一个通用 marker 实现。普通折线 GMT 与
CSI patch GMT 使用不同 reader：前者是本页的分段表面迹线，后者属于断层 patch 模型协议。

## 旧命令兼容

旧调用仍可运行：

```bash
ecat-fault-trace-tool trace.txt \
  --algo vw --param 0.5 \
  --output simplified
```

它保留原有输出：`simplified_trace.txt`、`simplified_fixed_params.txt` 和
`simplified_plot.png`。新脚本建议使用 `simplify` 或 `smooth` 子命令；新子命令只默认写用户明确指定
的 trace，绘图和报告均为可选，不再自动生成 `fixed_params`。
