# Trace 预处理与断层顶部边界

这个例子演示如何在写入 fault object 前，用数组级工具统一 trace 方向、裁剪、端点延伸和重采样。

## 输入

- `fault_trace.txt`：两列 `lon lat`。
- `lon0/lat0`：CSI/eqtools 局部投影原点。
- 目标：得到干净、等间距的地表 trace，并把它作为三维顶部边界。

## 最小代码

```python
import numpy as np
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)
from eqtools.csiExtend.trace_ops import (
    extend_trace,
    orient_trace,
    resample_trace,
    simplify_trace,
    trim_trace,
)

lon0, lat0 = 87.5, 28.5
trace = np.loadtxt("fault_trace.txt")

fault = TriFault("TraceFault", lon0=lon0, lat0=lat0, verbose=False)

# Project lon/lat to local x/y in km before length-based operations.
x, y = fault.ll2xy(trace[:, 0], trace[:, 1])
trace_xy = np.column_stack((x, y))

trace_xy = orient_trace(trace_xy, start="west")
trace_xy = simplify_trace(trace_xy, method="vw", tolerance=0.2)
trace_xy = trim_trace(trace_xy, start=2.0, end=45.0)
trace_xy = extend_trace(trace_xy, start=5.0, end=8.0, tangent_window=3)
trace_xy = resample_trace(trace_xy, every=1.0)

lon_new, lat_new = fault.xy2ll(trace_xy[:, 0], trace_xy[:, 1])
fault.trace(lon_new, lat_new)
fault.set_top_coords_from_trace()
```

## 继续生成断层面

如果使用单一倾角生成底边：

```python
fault.top = 0.0
fault.depth = 20.0
fault.generate_bottom_from_single_dip(
    dip_angle=70.0,
    dip_direction=180.0,
)
fault.generate_mesh(top_size=1.0, bottom_size=2.0, show=False, verbose=0)
fault.initializeslip(values="depth")
```

## 检查

```python
from eqtools.csiExtend import print_fault_summary

print_fault_summary(fault)
```

重点看 trace 长度、顶部/底部深度范围、mesh 数量和平均走向/倾角。

## 何时不用这个例子

- 如果只是要把 CSI fault trace 离散成 `xi/yi`，直接用 `fault.discretize_trace(every=...)`。
- 如果已有 top/bottom 三维曲线，优先用 `discretize_top_coords(...)` 和 `discretize_bottom_coords(...)` 统一点数。
- 如果是多条等深线或 slab 几何，使用 `FaultGeometryEngine` 管理 layers。

相关参考：
[Fault Geometry Construction](../reference/fault_geometry_construction.md),
[Fault Summary](../reference/fault_summary.md),
[Fault Contours](../reference/fault_contours.md)。
