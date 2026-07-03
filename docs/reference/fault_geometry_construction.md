# Fault Geometry Construction Reference / 断层几何构建参考

本页汇总 ECAT/eqtools 中常用的断层几何构建路径。它关注“如何从已有几何信息构建可用于正演或反演的 fault object”，不替代 nonlinear geometry、BLSE/VCE、constraint、edge 或 contour 的专门页面。

如果只需要最小代码，先看 [Trace 预处理与断层顶部边界](../examples/fault_trace_preprocessing.md) 或 [非线性几何结果到 fault object](../examples/fault_from_nonlinear_geometry.md)。如果还不清楚 trace、top/bottom、layers、mesh 和 patch GMT 的区别，先读 [断层几何状态](../concepts/fault_geometry_states.md)。

## 选择路径

先按已有输入选择路径，再进入对应小节复制最小代码片段。

| 如果你已有 | 跳到 | 先准备 |
| --- | --- | --- |
| 非线性几何反演结果 | [矩形元](#nonlinear-rect) 或 [三角元](#nonlinear-tri) | `geom`、`lon0/lat0`、`top/depth` |
| 地表迹线和固定倾角 | [地表迹线和倾角](#trace-dip) | trace 文件、`dip_angle`、`dip_direction` |
| 地表迹线和沿走向变化倾角 | [地表迹线和倾角](#trace-dip) | trace 文件、`xydip` 控制点或剖面倾角 |
| 倾角随深度变化 | [倾角随深度变化](#layered-dip) | trace、深度-倾角剖面或深度函数 |
| 多条等深线或 slab 几何 | [多条等深线和 Slab 几何](#slab-contours) | 等深线或 Slab2 grid、`lon0/lat0`、裁剪范围 |
| 外部 Gmsh/PyLith mesh | [外部 Mesh 和 PyLith](#external-mesh) | mesh 文件、坐标系、单位和 `z` 正负号 |
| 已有 GMT fault/slip model | [GMT 读取和保存](#gmt-io) | GMT 类型、patch 类型、slip header 约定 |
| 需要简化、离散、加密、缓冲或外推已有几何 | [常用几何辅助操作](#geometry-helpers) | trace、top/bottom/layer、等深线或已有 fault object |

## 参数约定

非线性几何反演结果进入分布式滑动模型时，最容易混淆的是 `cdepth`、`top` 和 `depth`：

- `clon/clat/cdepth` 表示非线性几何结果中的**断层顶边中点**经度、纬度和深度。
- `top` 是后续滑动面网格的顶部深度，通常比 `cdepth` 更浅。
- `depth` 是后续滑动面网格的底部深度，通常比 `cdepth` 更深。
- 深度在 CSI/eqtools fault object 中通常取正值，单位为 km。
- `x/y/z` 通常是局部投影坐标，单位 km；外部 mesh 常用 m，需要读入或保存时显式转换。
- `width=None` 时，三角元和矩形元构建函数会优先用 `top/depth/dip` 推断下倾宽度；如果传入 `width`，需要确认它和目标 `depth` 是否一致。

## Trace 和边界离散化约定

ECAT/eqtools 中有两套常见几何状态，使用时不要混淆：

- **地表迹线 trace**：CSI 原生状态，`fault.trace(...)` 写入 `xf/yf` 和 `lon/lat`。需要把地表迹线离散为 `xi/yi` 时，推荐使用 `fault.discretize_trace(every=...)`。
- **三维边界坐标**：csiExtend 状态，`top_coords`、`bottom_coords` 和 `layers` 表示顶部、底部和中间层等深线。需要离散这些三维边界时，推荐使用 `discretize_top_coords(...)`、`discretize_bottom_coords(...)` 或 `discretize_layer_coords(...)`。

`fault.set_top_coords_from_trace(discretized=False)` 默认使用原始 trace；如果传入 `discretized=True`，需要先调用 `fault.discretize_trace(...)` 生成 `xi/yi`。对于需要 top/bottom 点数一一对应的 mesh，优先使用相同的 `num_segments` 离散 top 和 bottom，避免分别按 `every` 离散后点数不同。

`fault.discretize(...)` 是 CSI 的 legacy trace 离散化接口，依赖 `xaxis/tol/fracstep` 等旧参数。新代码和新文档不再推荐使用它；需要地表迹线离散化时使用 `discretize_trace(...)`，需要三维边界离散化时使用 `discretize_*_coords(...)`。

<a id="geometry-helpers"></a>

## 常用几何辅助操作

这一节只列通用操作入口。完整建模仍按后面的场景小节选择；不要把这些 helper 当成独立反演流程。

| 需求 | 推荐入口 | 说明 |
| --- | --- | --- |
| 简化或平滑过密 trace | `ecat-fault-trace-tool` | 支持 `vw`、`rdp` 和 `bspline`；输出简化 trace、分段参数和对比图。 |
| 对任意 `x/y` trace 数组做长度、重采样、延伸、裁剪或方向统一 | `trace_ops.clean_trace(...)`、`trace_length(...)`、`resample_trace(...)`、`extend_trace(...)`、`trim_trace(...)`、`orient_trace(...)` | 纯函数入口，适合在写入 fault object 前预处理。输入应是投影后的 `x/y`，单位通常为 km。 |
| 对任意 `x/y` trace 数组简化、平滑或缓冲 | `trace_ops.simplify_trace(...)`、`smooth_trace(...)`、`buffer_trace(...)` | 适合脚本化批处理；`ecat-fault-trace-tool` 也复用同一套底层算法。 |
| 设置、读取或保存 CSI trace | `fault.trace(...)`、`fault.file2trace(...)`、`fault.writeTrace2File(...)` | 处理两列 `lon lat` 或局部 `x y` 迹线，不表示 patch。 |
| 地表 trace 等距离散 | `fault.discretize_trace(every=...)` | 生成 `xi/yi/loni/lati`；新代码优先用它，不再用 legacy `discretize(...)`。 |
| 从 trace 生成三维顶部边界 | `fault.set_top_coords_from_trace(discretized=...)` | `discretized=True` 时先运行 `discretize_trace(...)`。 |
| top/bottom/layer 曲线加密或统一点数 | `discretize_top_coords(...)`、`discretize_bottom_coords(...)`、`discretize_layer_coords(...)` | 适合手动构建 mesh 前统一三维边界点数。 |
| Bayesian 几何扰动前自动加密稀疏控制点 | `set_densification(...)`、`densify_edges(...)` | 由 `DensificationConfig` 控制，通常放在扰动和物理建底边之间。 |
| 非线性结果沿走向正负方向使用不同长度 | `custom_length=(neg_length, pos_length)` | 目前三角元 `generate_top_bottom_from_nonlinear_soln(...)` 支持；矩形元主流程仍是对称 `length`。 |
| 由两条等深线外推目标深度或地表迹线 | `FaultGeometryEngine.extrapolate_layer(...)`、`generate_surface_trace(...)` | 这是基于两条等深线的深度外推，不是简单把 trace 端点沿切线延长。 |
| 从已有 fault object 反提 trace 或等深线 | `fault.setTrace(...)`、`FaultGeometryEngine.extract_contours_from_fault(...)` | `setTrace(...)` 从浅部 patch 顶点反推 trace；等深线详见 [Fault Contours](fault_contours.md)。 |
| 生成 trace 周边缓冲多边形 | `fault.create_fault_trace_buffer(...)` | 适合筛选、遮罩或检查近断层区域；不是改变 fault 几何本身。 |
| 读写 CSI patch GMT | `readPatchesFromFile(...)`、`writePatches2File(...)` | 只用于每段表示一个 patch 的 CSI GMT；普通 polyline GMT 用 `eqtools.gmttools`。 |

常见 trace 预处理命令：

```bash
ecat-fault-trace-tool input_trace.txt --algo vw --param 0.5 --output trace_simplified
```

它会写出 `trace_simplified_trace.txt`，可直接作为后续 `fault.trace(...)` 或 `fault.file2trace(...)` 的输入。`vw` 通常适合自然断层迹线减点；`rdp` 适合保留折线拐角；`bspline` 适合平滑噪声较强的 trace。

如果要在脚本中批量控制 trace 的方向、裁剪、端点延伸和采样间隔，优先用 `trace_ops` 纯函数。注意不要直接在经纬度上做长度和距离操作，先转成局部投影 `x/y`：

```python
import numpy as np
from eqtools.csiExtend.trace_ops import (
    extend_trace,
    orient_trace,
    resample_trace,
    trim_trace,
)

x, y = fault.ll2xy(lon, lat)
trace_xy = np.column_stack((x, y))
trace_xy = orient_trace(trace_xy, start="west")
trace_xy = trim_trace(trace_xy, start=2.0, end=45.0)
trace_xy = extend_trace(trace_xy, start=5.0, end=8.0, tangent_window=3)
trace_xy = resample_trace(trace_xy, every=1.0)
lon_new, lat_new = fault.xy2ll(trace_xy[:, 0], trace_xy[:, 1])
fault.trace(lon_new, lat_new)
```

如果只是要把已有 trace 加密到固定间隔：

```python
fault.trace(lon, lat)
fault.discretize_trace(every=2.0, threshold=0.5)
fault.set_top_coords_from_trace(discretized=True)
```

如果已经有 top/bottom 三维曲线，并且后续 mesh 要求两者点数一致：

```python
fault.discretize_top_coords(num_segments=60)
fault.discretize_bottom_coords(num_segments=60)
```

<a id="nonlinear-rect"></a>

## 非线性结果到矩形元

规则平面矩形元适合先做稳定、可控的线性滑动反演，或需要和传统矩形 patch 工作流兼容时使用。

需要准备：

- `geom`：至少包含 `clon`、`clat`、`cdepth`、`strike`、`dip` 和 `length`。
- `lon0/lat0`：CSI/eqtools 局部投影原点，不是反演参数。
- `top/depth`：扩展后的滑动面顶部和底部深度。
- `n_strike/n_dip`：沿走向和下倾方向的矩形元数量。

```python
from eqtools.csiExtend.AdaptiveRectangularPatches import (
    AdaptiveRectangularPatches as RectFault,
)

rect = RectFault("MainFault", lon0=lon0, lat0=lat0, verbose=False)
rect.buildPatches_from_nonlinear_soln(
    clon=geom["clon"],
    clat=geom["clat"],
    cdepth=geom["cdepth"],
    strike=geom["strike"],
    dip=geom["dip"],
    length=geom["length"],
    width=None,
    top=0.0,
    depth=25.0,
    n_strike=20,
    n_dip=10,
    verbose=False,
)
rect.initializeslip(values="depth")
```

如果后续需要输出 patch 面积，可以调用：

```python
rect.compute_patch_areas()
```

建完几何后，建议先打印一次断层概览，检查迹线长度、patch 数、面积和深度范围：

```python
from eqtools.csiExtend.fault_summary import print_fault_summary

print_fault_summary(rect)
```

输出字段说明见 [Fault Summary / 断层概览和统计](fault_summary.md)。

<a id="nonlinear-tri"></a>

## 非线性结果到三角元

三角元适合需要自适应网格、复杂边界、cutde Green's functions、几何扰动或分层几何时使用。

需要准备：

- `geom`：至少包含 `clon`、`clat`、`cdepth`、`strike`、`dip` 和 `length`。
- `lon0/lat0`：CSI/eqtools 局部投影原点。
- `top/depth`：扩展后的滑动面顶部和底部深度。
- `top_size/bottom_size`：顶部和底部附近的目标网格尺度。

```python
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)

tri = TriFault("MainFault", lon0=lon0, lat0=lat0, verbose=False)
tri.top = 0.0
tri.depth = 25.0

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
tri.generate_mesh(
    top_size=1.0,
    bottom_size=2.0,
    show=False,
    verbose=0,
)
tri.initializeslip(values="depth")
```

`center_point_type` 用来说明输入点代表哪一个几何位置。常用值包括 `top_center`、`top_neg_end`、`top_pos_end` 和 `center`。

当非线性结果只约束一侧长度，或希望向正负走向方向使用不同长度时，可用 `custom_length=(neg_length, pos_length)`。

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

`neg_length` 和 `pos_length` 分别表示沿该方法内部正负走向方向的长度，二者之和是最终顶边长度。使用前建议画出 trace 或打印 [Fault Summary](fault_summary.md)，确认左右端点与预期一致。

<a id="trace-dip"></a>

## 地表迹线和倾角

已有地表迹线时，先把 trace 读入 fault object，再生成顶部坐标、底部坐标和 mesh。

如果 trace 点过密、噪声较强，或只是希望先得到较干净的建模迹线，可先用 [常用几何辅助操作](#geometry-helpers) 中的 `ecat-fault-trace-tool` 预处理，再把输出的 `_trace.txt` 文件交给下面流程。

需要准备：

- `fault_trace.txt`：默认是两列 `lon lat`，点顺序定义 trace 方向。
- `dip_angle`：倾角，单位 degree。
- `dip_direction`：下倾方向方位角，单位 degree。
- `top/depth`：最终 fault surface 顶部和底部深度。

如果 trace 已经是局部 `x y` 坐标，先转换成经纬度再调用 `fault.trace(...)`，或直接构造 `[x, y, top]` 后使用 `set_top_coords(..., lonlat=False)`。

```python
import numpy as np
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)

trace = np.loadtxt("fault_trace.txt")

fault = TriFault("TraceFault", lon0=lon0, lat0=lat0, verbose=False)
fault.top = 0.0
fault.depth = 20.0
fault.trace(trace[:, 0], trace[:, 1])
fault.set_top_coords_from_trace()
fault.generate_bottom_from_single_dip(
    dip_angle=70.0,
    dip_direction=180.0,
)
fault.generate_mesh(top_size=1.0, bottom_size=2.0, show=False, verbose=0)
fault.initializeslip(values="depth")
```

如果倾角沿走向变化，推荐把 trace 离散化后，在顶部节点上插值倾角，再按分段倾角生成底边：

`xydip` 可以是数组、`DataFrame` 或 CSV 文件。经纬度输入通常包含 `lon, lat, dip`；局部坐标输入通常包含 `x, y, dip` 并设置 `is_utm=True`。

```python
fault.interpolate_top_dip_from_relocated_profile(
    xydip=dip_points,
    is_utm=False,
    discretization_interval=2.0,
    interpolation_axis="auto",
    method="min_mse",
)
fault.generate_bottom_from_segmented_relocated_dips(
    fault_depth=25.0,
    use_average_strike=True,
)
fault.generate_mesh(top_size=1.0, bottom_size=2.0, show=False, verbose=0)
```

<a id="layered-dip"></a>

## 倾角随深度变化

当断层不是单一平面，而是随深度变缓、变陡或具有多层结构时，使用 layered dip 类。

需要准备：

- trace：两列 `lon lat`，或已转换好的顶部坐标。
- `reference_nodes`：沿 trace 或剖面上的控制节点。
- `depth_dip_profiles`：每个控制节点对应的深度-倾角数组，深度为正值，单位 km。
- `num_layers` 或 `layer_depths`：控制中间层数量或深度。

```python
from eqtools.csiExtend.AdaptiveLayeredDipTriangularPatches import (
    AdaptiveLayeredDipTriangularPatches as LayeredTriFault,
)

fault = LayeredTriFault("LayeredFault", lon0=lon0, lat0=lat0, verbose=False)
fault.top = 0.0
fault.depth = 30.0
fault.trace(trace[:, 0], trace[:, 1])
fault.set_top_coords_from_trace()

fault.set_depth_dip_from_profiles(
    profiles_data={
        "reference_nodes": reference_nodes,
        "depth_dip_profiles": depth_dip_profiles,
    },
    interpolation_method="linear",
)
fault.setup_interpolation(discretization_interval=2.0, interpolation_axis="auto")
fault.generate_layer_coords(num_layers=4)
fault.generate_bottom_coords()
fault.generate_layered_mesh(
    num_layers=4,
    nodes_on_layers=True,
    mesh_func=True,
    field_size_dict={"min_dx": 1.0, "bias": 1.1},
    show=False,
)
```

可选的倾角来源包括：

- `set_depth_dip_from_constant(...)`：固定倾角的分层模型。
- `set_depth_dip_from_function(...)`：倾角由深度函数控制。
- `set_depth_dip_from_profiles(...)`：每个参考节点给出离散的 `depth, dip` 剖面。

如果需要结构化矩形元，可使用 `AdaptiveLayeredDipRectangularPatches`，通过深度-倾角剖面构建沿走向和下倾方向规则分块。

<a id="slab-contours"></a>

## 多条等深线和 Slab 几何

多条等深线、Slab2 网格或已有三维 slab surface 推荐统一交给 `FaultGeometryEngine` 管理。它负责把不同深度的曲线组织成 layers，再生成矩形或三角 fault model。

需要准备：

- `lon0/lat0/utmzone`：统一投影参数。
- `target_levels`：目标深度列表，使用正值，单位 km。
- `bbox_ll`：裁剪范围，顺序为 `[lon_min, lon_max, lat_min, lat_max]`。
- `buffer_km`：围绕参考等深线或地表迹线保留的垂向缓冲距离。

```python
from eqtools.csiExtend.FaultGeometryEngine import FaultGeometryEngine

engine = FaultGeometryEngine(
    "SlabGeometry",
    lon0=lon0,
    lat0=lat0,
    utmzone=utmzone,
    verbose=True,
)

engine.load_from_slab2(
    grd_file="slab_depth.grd",
    target_levels=[20, 40, 60, 80, 100],
    min_points=50,
    stitch_mode="lat",
)
engine.generate_surface_trace(
    shallow_depth=20.0,
    deep_depth=40.0,
)
engine.apply_spatial_filter(
    bbox_ll=[lon_min, lon_max, lat_min, lat_max],
    buffer_km=100.0,
)
```

如果等深线已经由其他工具提取好，不需要 `load_from_slab2`，可以逐层加入：

```python
engine.add_layer(coords20, depth=20.0, coords_type="ll", sort_by="lon")
engine.add_layer(coords40, depth=40.0, coords_type="ll", sort_by="lon")
engine.add_layer(coords60, depth=60.0, coords_type="ll", sort_by="lon")
```

构建矩形模型：

```python
rect = engine.build_rectangular_model(
    "RectSlab",
    total_width=120.0,
    numz=12,
    mesh_len=20.0,
    num_profiles=8,
)
rect.initializeslip(values="depth")
```

构建三角模型：

```python
tri = engine.build_triangular_model(
    "TriSlab",
    field_size_dict={"min_dx": 10.0, "bias": 1.1},
    top_size=10.0,
    bottom_size=30.0,
    sparse_factor=0.5,
)
tri.initializeslip(values="depth")
```

从已有 fault object 反提等深线时，使用 [Fault Contours](fault_contours.md) 中的 `extract_contours_from_fault`。

<a id="external-mesh"></a>

## 外部 Mesh 和 PyLith

已有 Gmsh mesh 时，可以直接读入三角网格并保存为 CSI fault object：

需要准备：

- mesh 文件中代表断层面的 triangle cells。
- 输入 mesh 的坐标系：局部投影坐标、UTM 或经纬度。
- 输入 mesh 的长度单位：`unit="m"` 表示读入时从 m 转为 km。
- `z` 符号约定：eqtools fault object 使用正深度；外部 mesh 若为负向下，需要读入后检查或导出时使用 `flip_z`。

```python
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)

proj_string = None  # 如果 mesh x/y 需要投影回 lon/lat，在这里提供 PROJ string。

fault = TriFault("MeshFault", lon0=lon0, lat0=lat0, verbose=False)
fault.read_mesh_file(
    "fault_mesh.msh",
    tag=None,
    save2csi=True,
    element_name="triangle",
    unit="m",
    proj_params=proj_string,
)
fault.initializeslip(values="depth")
```

常用 mesh 输出和转换接口：

`convert_mesh_file(..., unit_conversion=1000.0)` 表示把读入文件中的坐标整体乘以 1000。

只有当待转换 mesh 文件的坐标以 km 存储、目标格式需要 m 时才这样设置；不要和 `read_mesh_file(unit="m")` 的读入转换重复使用。

```python
fault.convert_mesh_file(
    "fault_mesh.msh",
    output_format="abaqus",
    unit_conversion=1000.0,
    flip_z=False,
)

fault.save_geometry_as_mesh(
    "fault_geometry.vtk",
    coord_type="utm",
    output_unit="m",
    flip_z=False,
)
```

PyLith 相关接口偏向有限元 Green's functions 或位移场提取：

- `eqtools.pylithtools.TriangularTents_kfh.extractfromPylith(...)`：从 PyLith HDF5 Green's functions 中提取 vertices、cells 和响应矩阵。
- `eqtools.dispExtract.PylithDisp.PylithDisp.readdispts(...)`：读取 PyLith 位移时序并转换坐标。

这些接口通常属于有限元耦合流程，不建议和普通 GMT fault 构建流程混用。

<a id="gmt-io"></a>

## GMT 读取和保存

需要区分两类 GMT 文件：

- **普通线段 GMT**：表示 trace、等深线或其他 polyline。
- **CSI patch GMT**：每个 GMT segment 表示一个三角形或四边形 patch，可在 header 中保存 slip 信息。

普通线段 GMT 不表示 patch，它只保存一段或多段 polyline。CSI patch GMT 才能直接读回 fault object。

普通线段 GMT 使用：

```python
from eqtools.gmttools import read_gmt_lines, write_lines_to_gmt

segments = read_gmt_lines("contours.gmt", read_z=True)
write_lines_to_gmt(segments, z_values=[20, 40, 60], gmt_file="contours_out.gmt")
```

CSI patch GMT 读写使用 fault object 自带接口：

```python
tri.readPatchesFromFile(
    "tri_fault.gmt",
    gmtslip=True,
    readpatchindex=True,
)

rect.readPatchesFromFile(
    "rect_fault.gmt",
    increasingy=True,
    readpatchindex=True,
)

tri.writePatches2File("tri_fault_out.gmt", add_slip="total")
rect.writePatches2File("rect_fault_out.gmt", add_slip="strikeslip")
```

如果输入是外部 slip model，优先使用 `eqtools.slip_conversion` 中的 converter 标准化为 CSI patch GMT，再用 `readPatchesFromFile` 检查。

不要把外部 header 格式假定为等价的 CSI patch GMT。

边界和 patch 中心输出：

```python
fault.find_fault_fouredge_vertices(
    top_tolerance=0.1,
    bottom_tolerance=0.1,
    edge_method="topology",
    gap_policy="clean",
)
fault.writeFourEdges2File(dirname="output/stat_infos")
fault.writeSlipCenter2File("output/slip_centers.dat")
```

边界识别细节见 [Fault Edges](fault_edges.md)。

## 质量检查

完成几何构建后，至少检查以下内容：

1. 调用 `print_fault_summary(fault)`，检查 trace 长度、patch/mesh 数、面积、深度范围、平均走向和平均倾角。
2. 绘制 trace、top edge、bottom edge 和 mesh，确认倾向和深度方向正确。
3. 检查 `fault.top`、`fault.depth`、`Vertices[:, 2]` 或 patch 顶点深度范围。
4. 对三角元运行 `find_fault_fouredge_vertices(...)`，确认 `top/bottom/left/right` 合理。
5. 对复杂几何用 `FaultGeometryEngine.extract_contours_from_fault(...)` 反提关键等深线。
6. 写出 GMT 后重新读入一次，检查 patch 数量、深度范围和 slip header 是否符合预期。
7. 如果用于 Green's functions，确认 mesh 单位、投影、`flip_z` 和 `unit_conversion` 没有重复转换。

## 相关页面

- [Bayesian Nonlinear Geometry](../workflows/03_nonlinear_geometry_bayesian.md)
- [BLSE/VCE Linear Slip](../workflows/04_linear_slip_blse_vce.md)
- [Fault Summary](fault_summary.md)
- [Fault Edges](fault_edges.md)
- [Fault Contours](fault_contours.md)
- [Perturbable Fault Geometry](geometry_perturbation.md)
- [CLI 命令参考](cli.md)
