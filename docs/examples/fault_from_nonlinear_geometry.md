# 非线性几何结果到 fault object

这个例子把 Bayesian 非线性几何反演得到的紧凑几何参数转换成后续线性滑动反演可用的矩形元或三角元 fault object。

## 输入

非线性几何结果至少需要：

```python
geom = {
    "clon": 87.40,      # top-edge midpoint longitude
    "clat": 28.67,     # top-edge midpoint latitude
    "cdepth": 1.8,     # top-edge midpoint depth, km
    "strike": 332.0,   # degree
    "dip": 52.0,       # degree
    "length": 12.0,    # km
}
lon0, lat0 = 87.5, 28.5
top, depth = 0.0, 8.0
```

这里的 `clon/clat/cdepth` 表示**断层顶边中点**，不是断层面中心。`top/depth` 是后续分布式滑动面网格的顶部和底部深度。

## 矩形元

矩形元适合快速、规则、可控的线性滑动反演。

```python
from eqtools.csiExtend.AdaptiveRectangularPatches import (
    AdaptiveRectangularPatches as RectFault,
)

rect = RectFault("RectFault", lon0=lon0, lat0=lat0, verbose=False)
rect.buildPatches_from_nonlinear_soln(
    clon=geom["clon"],
    clat=geom["clat"],
    cdepth=geom["cdepth"],
    strike=geom["strike"],
    dip=geom["dip"],
    length=geom["length"],
    width=None,
    top=top,
    depth=depth,
    n_strike=20,
    n_dip=8,
    verbose=False,
)
rect.initializeslip(values="depth")
```

## 三角元

三角元适合复杂边界、自适应网格、cutde Green's functions 或后续几何扰动。

```python
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)

tri = TriFault("TriFault", lon0=lon0, lat0=lat0, verbose=False)
tri.top = top
tri.depth = depth
tri.generate_top_bottom_from_nonlinear_soln(
    clon=geom["clon"],
    clat=geom["clat"],
    cdepth=geom["cdepth"],
    strike=geom["strike"],
    dip=geom["dip"],
    length=geom["length"],
    width=None,
    top=tri.top,
    depth=tri.depth,
    center_point_type="top_center",
)
tri.generate_mesh(top_size=1.0, bottom_size=1.5, show=False, verbose=0)
tri.initializeslip(values="depth")
```

如果非线性几何只约束一侧长度，或希望沿走向正负方向使用不同长度：

```python
tri.generate_top_bottom_from_nonlinear_soln(
    clon=geom["clon"],
    clat=geom["clat"],
    cdepth=geom["cdepth"],
    strike=geom["strike"],
    dip=geom["dip"],
    custom_length=(8.0, 14.0),
    top=tri.top,
    depth=tri.depth,
    center_point_type="top_center",
)
```

## 检查和导出

```python
from eqtools.csiExtend import print_fault_summary

print_fault_summary(tri)
tri.find_fault_fouredge_vertices()
tri.writePatches2File("tri_fault.gmt", add_slip="total")
```

线性滑动反演前，至少检查 trace 长度、patch 数量、面积、深度范围、顶部/底部边界和平均走向/倾角。

相关参考：
[Bayesian 非线性几何反演](../workflows/03_nonlinear_geometry_bayesian.md),
[BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md),
[Fault Geometry Construction](../reference/fault_geometry_construction.md),
[Fault Summary](../reference/fault_summary.md)。
