# 断层迹线预处理短例

本页只演示如何在写入 fault object 前检查和修改一条地表迹线。完整步骤见
[断层迹线预处理 workflow](../workflows/02d_fault_trace_preprocessing.md)，全部命令和 API 参数见
[Fault Trace Processing Reference](../reference/fault_trace_processing.md)。

输入 `fault_trace.txt` 至少有两列：

```text
# longitude latitude
101.10 24.10
101.20 24.15
101.30 24.21
```

## 命令行：检查并裁到指定经度

```bash
ecat-fault-trace-tool inspect fault_trace.txt

ecat-fault-trace-tool trim fault_trace.txt \
  --end-lon 101.25 \
  -o fault_trace_trimmed.txt \
  --plot \
  --report fault_trace_trimmed.json
```

`--end-lon` 保留输入首点到经度交点。若弯曲迹线多次穿过同一经度，可先查看：

```bash
ecat-fault-trace-tool locate fault_trace.txt --lon 101.25
```

再用 `--end-which last` 或交点索引明确选择。

## 命令行：裁到某个经纬度点的最近迹线位置

```bash
ecat-fault-trace-tool trim fault_trace.txt \
  --end-nearest 101.25 24.18 \
  --max-snap-km 5 \
  -o fault_trace_nearest.txt
```

最近点落在真实折线段上，不要求它是原始输入顶点。`--max-snap-km` 防止坐标写错后吸附到很远的位置。

## Python：链式处理但不修改原数组

```python
import numpy as np

from eqtools.csiExtend import TracePath, write_trace

lon0, lat0 = 101.2, 24.2
trace_lonlat = np.loadtxt("fault_trace.txt", ndmin=2)[:, :2]

trace = TracePath.from_lonlat(
    trace_lonlat,
    lon0=lon0,
    lat0=lat0,
)

prepared = (
    trace
    .orient(start="west")
    .trim(end={"longitude": 101.25})
    .extend(end_km=3.0, tangent_window=3)
    .resample(every_km=1.0)
)

write_trace(
    "fault_trace_prepared.txt",
    prepared.lonlat,
)
```

`TracePath` 的每个方法都返回新对象；`trace` 和 `trace_lonlat` 不会被原位修改。已经有 CSI fault
作为投影提供者时，也可以写：

```python
trace = TracePath.from_lonlat(
    trace_lonlat,
    projection=fault,
)
```

处理完成后再显式交给 fault：

```python
fault.trace(
    prepared.lonlat[:, 0],
    prepared.lonlat[:, 1],
)
fault.discretize_trace(every=1.0)
fault.set_top_coords_from_trace(discretized=True)
```

不要在 fault 已经生成 top/bottom、mesh、patch 或 GF 后替换地表迹线；这些派生状态需要重新构建。
