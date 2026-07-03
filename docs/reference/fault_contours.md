# 断层等深线与等值线参考 / Fault Contours

本页区分两个在图件上相似、但在代码和几何意义上不同的操作。

- **等深线 / isodepth contour**：断层几何面与 `depth = constant` 水平面的交线，例如 25 km 深度线。
- **等值线 / scalar contour**：断层面上某个标量场的等值线，例如 slip、coupling、应变或其他节点/单元属性。

当目标是断层几何本身时，使用等深线接口。当目标是定义在断层面上的数值场时，使用等值线接口。

## 推荐接口

### 等深线

推荐入口是 `FaultGeometryEngine.extract_contours_from_fault`：

```python
from eqtools.csiExtend.FaultGeometryEngine import FaultGeometryEngine

engine = FaultGeometryEngine("geom", lon0=lon0, lat0=lat0, utmzone=utmzone)
contours = engine.extract_contours_from_fault(
    fault,
    target_depths=[25.0],
    method="mesh_plane_intersection",
)

line25 = contours[25.0]  # columns: x, y, depth, lon, lat
```

`method="mesh_plane_intersection"` 是默认值。脚本里显式写出 method 更便于审查和复现。

需要检查线段拼接、不连续段或深度残差时，可以打开诊断信息：

```python
contours, diagnostics = engine.extract_contours_from_fault(
    fault,
    [25.0],
    return_diagnostics=True,
)

print(diagnostics["depths"][25.0]["raw_segments"])
print(diagnostics["depths"][25.0]["stitched_polylines"])
print(diagnostics["depths"][25.0]["max_depth_residual"])
```

默认每个深度返回最长的一条拼接线。如果一个深度可能穿过多个不连续几何片，可以保留所有线：

```python
contours = engine.extract_contours_from_fault(
    fault,
    [25.0],
    largest_only=False,
)

for line in contours[25.0]:
    # each line has columns x, y, depth, lon, lat
    pass
```

### 低层等深线函数

纯几何 helper 位于 `statUtils`：

```python
from eqtools.csiExtend.statUtils.fault_contours import extract_isodepth_contours

contours = extract_isodepth_contours(fault, [25.0])
```

没有传入 `engine` 或其他带 `xy2ll(x, y)` 方法的对象时，输出列为：

```text
x, y, depth
```

传入 `engine=engine` 后，输出列为：

```text
x, y, depth, lon, lat
```

### 等值线

等值线仍由 `contour3D_extraction.py` 负责：

```python
from eqtools.csiExtend.statUtils.contour3D_extraction import Contour3DExtraction
from eqtools.csiExtend.statUtils.contour3D_extraction import plot_contourinsurf
```

这些工具适用于 slip、coupling 或其他定义在断层面上的标量场。它们在计算参数面上对标量场做 contour，然后恢复三维坐标；这和等深线的几何交线问题不同。

## 实现原理

### `mesh_plane_intersection` 后端

默认等深线后端按以下流程执行：

1. 把 fault object 转换为三角网格。
2. 对每个目标深度 `d`，逐个三角形求它与水平面 `z = d` 的交线段。
3. 对重合线段和端点做容差去重。
4. 根据共享端点把无序线段拼接为确定顺序的 polyline。
5. 默认返回每个深度最长的一条 polyline；设置 `largest_only=False` 时返回全部 polyline。
6. 如果存在 `xy2ll` 坐标转换器，补充经纬度列。

这个方法不需要用户选择 `x-y`、`x-z` 或 `y-z` 投影面。对垂直或近垂直断层，这一点很重要，因为 map-view contour 可能退化，或者点集正确但线序出现长距离跳接。

## 三角元与矩形元

三角元：

- 使用 `fault.Vertices` 和 `fault.Faces`。
- 对离散的 planar triangular surface 是严格几何交线。
- 深度残差应接近浮点精度。

矩形元：

- 使用 `fault.patch`。
- `subdivision=1` 时，每个四边形拆成两个三角形。
- `subdivision>1` 时，先在四边形上做 bilinear subdivision，再把细分网格三角化。

如果矩形四角共面，两三角化就是该 patch 的严格平面表示。如果矩形四角不共面，结果严格对应生成后的 piecewise-linear mesh。需要更接近 bilinear 四边形曲面时，可增大 `subdivision`，例如 `3` 或 `5`。

## 旧方法

历史 map-view contour 路径仍然保留：

```python
contours = engine.extract_contours_from_fault(
    fault,
    [25.0],
    method="legacy_map_contour",
)
```

这个方法在 map view 中三角化断层面，并用 `matplotlib.tricontour` 对 depth 做 contour。它适合复现旧结果和做回归对照，但不建议作为新的等深线默认方法，原因是：

- 垂直或近垂直断层在 map view 中可能投影退化。
- 点集可能正确，但路径顺序可能包含长距离跳接。
- 它把几何平面交线问题转化成了投影 contour 问题。

## 返回结构

`FaultGeometryEngine.extract_contours_from_fault` 默认返回：

```python
{
    depth: ndarray,  # columns: x, y, depth, lon, lat
}
```

`depth` 为正值，单位 km。`x` 和 `y` 使用当前 CSI/eqtools 对象的投影坐标系，通常单位也是 km。

设置 `largest_only=False` 时，返回：

```python
{
    depth: [ndarray, ndarray, ...]
}
```

多条线通常表示目标深度穿过多个不连续断层片、对象包含多个几何组件，或 mesh 本身存在不连续区域。

## 方法选择

等深线提取建议按以下顺序选择：

1. 常规工作使用 `mesh_plane_intersection`。
2. 复杂几何验证时使用 `largest_only=False` 和 `return_diagnostics=True`。
3. 只有需要复现历史输出时，才使用 `legacy_map_contour`。

对 slip、coupling 等标量场做等值线时，使用 `Contour3DExtraction` 或 `plot_contourinsurf`，不要使用等深线接口。
