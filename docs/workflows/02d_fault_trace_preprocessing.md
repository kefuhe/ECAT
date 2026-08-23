# 02d 断层迹线预处理

本流程用于在构建 fault object 之前，对一条地表迹线做可检查、可复现的方向统一、裁剪、延长、
重采样、简化或平滑。只需要复制最短命令时先看
[断层迹线预处理短例](../examples/fault_trace_processing.md)；查完整参数时打开
[Fault Trace Processing Reference](../reference/fault_trace_processing.md)。

## 什么时候使用哪个入口

| 任务 | 入口 |
| --- | --- |
| 已知距离、经度、纬度或最近点，想得到可复现的新文件 | `ecat-fault-trace-tool` |
| 在科研脚本中循环比较多套处理参数 | `TracePath` 或 `trace_ops` |
| 需要在观测图上拖动、增加或删除节点 | [交互迹线编辑器](02c_interactive_trace_editing.md) |
| 已经进入 top/bottom、mesh 或 patch 构建 | [Fault Geometry Construction](../reference/fault_geometry_construction.md) |

预处理不会自动生成倾角、底边、mesh、patch 或反演配置，也不会修改输入文件。

## 输入

支持以下迹线文件：

- 两列 `lon lat` 的 TXT、DAT 或 TRACE；
- GMT 多段文本；
- GeoJSON `LineString` 或 `MultiLineString`。

文件包含多段时，用 `--segment INDEX` 明确选择一段。所有长度、间距和最近点距离在统一局部投影
`x/y` 中计算，单位为 km；若不指定 `--lon0/--lat0`，工具按输入迹线自动选择投影中心。

## 第一步：检查输入

```bash
ecat-fault-trace-tool inspect fault_trace.txt
```

至少核对：

- 点数和总长度是否符合迹线尺度；
- 首尾点是否定义了期望的走向正方向；
- 自动投影中心是否位于研究区附近；
- 输入是否意外包含多个 GMT/GeoJSON 分段。

机器读取时加 `--json`。

## 第二步：预览裁剪端点

按经度或纬度查交点：

```bash
ecat-fault-trace-tool locate fault_trace.txt --lon 101.8
ecat-fault-trace-tool locate fault_trace.txt --lat 24.2
```

按经纬度点查迹线最近位置：

```bash
ecat-fault-trace-tool locate fault_trace.txt \
  --nearest 101.8 24.2 \
  --max-snap-km 5
```

输出中的 `trace_distance_km` 始终从输入首点沿折线累计。弯曲迹线可能与同一经纬线多次相交；
默认使用第一个交点，`--which last` 或整数索引可选择其他交点。最近点不是最近输入顶点，而是折线段
上的正交投影点。

## 第三步：执行一个明确操作

### 统一方向

```bash
ecat-fault-trace-tool orient fault_trace.txt \
  --start west \
  -o fault_trace_oriented.txt
```

或只反转输入点序：

```bash
ecat-fault-trace-tool reverse fault_trace.txt -o fault_trace_reversed.txt
```

### 裁剪

保留首点到经度交点：

```bash
ecat-fault-trace-tool trim fault_trace.txt \
  --end-lon 101.8 \
  -o fault_trace_trimmed.txt
```

保留两个 marker 之间的区间：

```bash
ecat-fault-trace-tool trim fault_trace.txt \
  --start-lat 24.0 \
  --end-nearest 101.8 24.2 \
  --max-snap-km 5 \
  -o fault_trace_segment.txt
```

还可使用 `--start-km/--end-km` 和 `--start-fraction/--end-fraction`。

### 延长和重采样

```bash
ecat-fault-trace-tool extend fault_trace.txt \
  --start-km 2 --end-km 5 \
  --tangent-window 3 \
  -o fault_trace_extended.txt

ecat-fault-trace-tool resample fault_trace_extended.txt \
  --every-km 1 \
  -o fault_trace_resampled.txt
```

延长方向由输入首尾点和端部切线决定。噪声较强时增加 `--tangent-window`，但仍需画图检查。

### 简化或平滑

```bash
ecat-fault-trace-tool simplify fault_trace.txt \
  --method vw --tolerance 0.2 \
  -o fault_trace_simplified.txt

ecat-fault-trace-tool smooth fault_trace.txt \
  --method savgol --window 5 --polyorder 2 \
  -o fault_trace_smoothed.txt
```

VW 的阈值是 km²，RDP 的阈值是 km。平滑会改变中间节点的位置，应比较处理前后局部走向；默认保留
两个端点。

## 多步处理

操作顺序会改变结果。需要稳定复现时写一份短 YAML：

```yaml
operations:
  - op: orient
    start: west
  - op: trim
    end: {longitude: 101.8, which: first}
  - op: extend
    end_km: 5.0
    tangent_window: 3
  - op: resample
    every_km: 1.0
```

然后运行：

```bash
ecat-fault-trace-tool apply fault_trace.txt \
  --config trace_operations.yml \
  -o fault_trace_prepared.txt \
  --plot \
  --report fault_trace_prepared.json
```

`--report` 记录投影、点数、长度和每一步参数。输出文件默认不覆盖；确认替换目标时显式加
`--overwrite`。

## Python 批量处理

高层脚本优先使用不可变 `TracePath`：

```python
import numpy as np
from eqtools.csiExtend import TracePath

trace_lonlat = np.loadtxt("fault_trace.txt", ndmin=2)[:, :2]
trace = TracePath.from_lonlat(trace_lonlat, lon0=101.5, lat0=24.1)

candidate = trace.trim(
    start={"fraction": 0.1},
    end={"nearest": [101.8, 24.2], "max_distance_km": 5.0},
)
```

已经持有投影 `x/y` 数组且需要最小开销时，直接使用 `trim_trace()`、`extend_trace()`、
`resample_trace()` 等纯函数。两层接口最终调用同一数值内核。

## 检查与下一步

处理后至少检查：

1. 首尾点和点序；
2. 处理前后总长度和点数；
3. 经纬度 marker 的实际交点或最近点距离；
4. 延长段是否符合端部走向；
5. 简化/平滑后是否丢失重要弯折；
6. 输出与原始迹线叠加是否合理。

确认后，把新迹线交给
[由地表迹线和倾角构建三角断层](../examples/fault_trace_preprocessing.md)，再进入
[BLSE/VCE 线性滑动反演](04_linear_slip_blse_vce.md)。不要在已经生成 mesh、patch 或 GF 的 fault
对象上直接替换 trace。
