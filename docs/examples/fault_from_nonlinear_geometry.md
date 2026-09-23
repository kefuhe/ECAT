# 非线性几何结果到 fault object

这个例子把 Bayesian 非线性几何反演得到的紧凑几何参数转换成后续线性滑动反演可用的矩形元或三角元 fault object。

本页两个构网分支共用下面的输入定义，选择矩形元或三角元之一即可。
示例坐标和网格尺度是占位值；它们需要与本项目的观测、投影原点和配置名称一起替换。

<a id="geometry-handoff"></a>

## 输入

从同一个已选代表模型的摘要取几何，不把不同候选的参数拼接使用：

| 来源 | 本页变量或参数 | 交接时检查 |
| --- | --- | --- |
| 非线性摘要的 `lon/lat/depth` | `geom` 中的 `clon/clat/cdepth` | 顶边中点，经纬度为度、深度为 km |
| 所选模型的 solver geometry 角度及长度 | `strike/dip/length` | 角度与该模型一致，长度为 km |
| 数据读取时的投影原点 | `lon0/lat0` | 与所有观测对象一致，不用断层位置替换它 |
| 线性滑动模型的覆盖范围 | `top/depth` | 根据研究问题确定，区别于紧凑源的顶边中点深度 |
| 网格分辨率选择 | `n_strike/n_dip` 或 `top_size/bottom_size` | 构网后检查 patch 数、面积、深度与边界 |

这里通过显式数值交接，不直接把 HDF5 样本文件当成滑动网格读取。代表几何的选择见
[非线性结果判读](../workflows/03_nonlinear_geometry_bayesian.md#result-checks)。

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

优先从非线性模型报告的 `CSI solver geometry` 读取交接角度。标准矩形/三角入口也会再次执行
同一规范化，因此把历史负 `dip` 原样传入不会改变 SMC 实际使用的几何；这一步只处理
`strike/dip`，不会搬运或转换紧凑源 `rake/slip`。详见
[断层角度约定](../concepts/fault_angle_conventions.md)。

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

如果非线性几何只约束一侧长度，或希望沿走向正负方向使用不同长度，下面片段替换上面的
边界生成调用，放在 `generate_mesh()` 之前。若已经执行过构网，应在更改边界后重新执行
`generate_mesh()` 和滑动初始化，再检查或导出，不能继续使用旧网格：

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
如果非线性 input `dip` 曾超过 `90°` 或为负值，还应确认 fault summary/getpatchgeometry
给出的 canonical strike/dip 与非线性报告中的 `CSI solver geometry` 一致。

上面的检查与导出代码使用 `tri`；选择矩形分支时，应使用已建立的 `rect` 并按
[几何构建参考](../reference/fault_geometry_construction.md)检查对应网格。
此时尚未求解分布式滑动，初始化后的 slip 和 GMT 文件不能当作反演结果。

下一步将选定对象放入 `faults_list`，与[已准备的 geodata](inversion_data_loading.md#assemble-geodata)
和匹配的配置一起交给 [BLSE 输入装配](../workflows/04_linear_slip_blse_vce.md#input-handoff)。
本页 `RectFault`/`TriFault` 是对象名示例，应与配置中的断层名一致；不要直接套用另一个短例的 `MainFault` 配置。

相关参考：
[Bayesian 非线性几何反演](../workflows/03_nonlinear_geometry_bayesian.md),
[反演前读取 InSAR 与 GNSS 数据](inversion_data_loading.md),
[BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md),
[Fault Geometry Construction](../reference/fault_geometry_construction.md),
[Fault Summary](../reference/fault_summary.md)。
