# Fault Geometry Construction Reference / 断层几何构建参考

本页汇总 ECAT/eqtools 中常用的断层几何构建路径。它关注“如何从已有几何信息构建可用于正演或反演的 fault object”，不替代 nonlinear geometry、BLSE/VCE、constraint、edge 或 contour 的专门页面。

如果输入迹线还需要裁剪、延长、统一方向或重采样，先看
[断层迹线预处理](../workflows/02d_fault_trace_preprocessing.md)。构建 fault object 的最小代码见
[地表迹线和倾角构建示例](../examples/fault_trace_preprocessing.md) 或
[非线性几何结果到 fault object](../examples/fault_from_nonlinear_geometry.md)。如果还不清楚
trace、top/bottom、layers、mesh 和 patch GMT 的区别，先读
[断层几何状态](../concepts/fault_geometry_states.md)；如果要区分紧凑非线性、固定 trace dip 和
多倾角入口的不同角度协议，先读 [断层角度约定](../concepts/fault_angle_conventions.md)。

## 构建路径与阅读入口

当前按输入来源分成六类构建路径：非线性几何结果、地表 trace + dip、倾角随深度变化、
多等深线/slab、外部 mesh、已有 CSI GMT。矩形元或三角元是输出几何形式，不另设一套
输入协议。先按已有输入选择路径，再进入对应小节复制最小代码片段；辅助操作只负责清理或
转换已有几何，不算第七种构建模式。

| 如果你已有 | 跳到 | 先准备 |
| --- | --- | --- |
| 非线性几何反演结果 | [矩形元](#nonlinear-rect) 或 [三角元](#nonlinear-tri) | `geom`、`lon0/lat0`、`top/depth` |
| 地表迹线和固定倾角 | [固定倾角和显式下倾方向](#trace-dip-single) | trace 文件、`dip_angle`、`dip_direction` |
| 地表迹线和沿走向变化倾角 | [沿走向变化倾角](#trace-dip-varying) | trace 文件、`xydip` 控制点或剖面倾角 |
| 一组候选倾角且要求 patch 一一对应 | [参考网格与固定拓扑](#fixed-topology-dip-search) | 参考倾角、候选倾角、统一的映射和网格参数 |
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
- `strike` 和 `dip_direction` 都是地理方位角：`0°` 指北、`90°` 指东，角度从北顺时针增加。
- `strike` 的正方向跟随有序 trace/top 点序；反转点序会把正走向反转约 `180°`。
- `dip_angle` 是断层面相对水平面的倾角。常规输入使用 `0 < dip_angle <= 90°`；`0°` 会使按深度推断宽度的公式退化。

这里的 `dip_angle` 规则专指“trace + 显式 dip direction”入口。紧凑非线性 SMC 原生支持
`dip in [0, 180]`，并在建 patch 前与 strike 联动规范化；标准
`*_from_nonlinear_soln(...)` 桥接方法采用同一协议。不要把这条协议直接推广到 layered dip、
多倾角控制点或任意外部 mesh。

## Trace 和边界离散化约定

ECAT/eqtools 中有两套常见几何状态，使用时不要混淆：

- **地表迹线 trace**：CSI 原生状态，`fault.trace(...)` 写入 `xf/yf` 和 `lon/lat`。需要把地表迹线离散为 `xi/yi` 时，推荐使用 `fault.discretize_trace(every=...)`。
- **三维边界坐标**：csiExtend 状态，`top_coords`、`bottom_coords` 和 `layers` 表示顶部、底部和中间层等深线。需要离散这些三维边界时，推荐使用 `discretize_top_coords(...)`、`discretize_bottom_coords(...)` 或 `discretize_layer_coords(...)`。

`fault.set_top_coords_from_trace(discretized=False)` 默认使用原始 trace；如果传入 `discretized=True`，需要先调用 `fault.discretize_trace(...)` 生成 `xi/yi`。对于需要 top/bottom 点数一一对应的 mesh，优先使用相同的 `num_segments` 离散 top 和 bottom，避免分别按 `every` 离散后点数不同。

`fault.discretize(...)` 是 CSI 的 legacy trace 离散化接口，依赖 `xaxis/tol/fracstep`
等旧参数。新项目不建议使用它；需要地表迹线离散化时使用
`discretize_trace(...)`，需要三维边界离散化时使用 `discretize_*_coords(...)`。

<a id="geometry-helpers"></a>

## 常用几何辅助操作

这一节只列通用操作入口。完整建模仍按后面的场景小节选择；不要把这些 helper 当成独立反演流程。

| 需求 | 推荐入口 | 说明 |
| --- | --- | --- |
| 检查、定位、裁剪、延长、重采样、简化或平滑 trace 文件 | `ecat-fault-trace-tool` | 新子命令输出一条明确的新迹线；完整参数见 [断层迹线处理参考](fault_trace_processing.md)。 |
| 在 Python 中有序处理经纬度 trace | `TracePath` | 不可变高层接口，统一投影、marker、操作历史和坐标转换。 |
| 对任意 `x/y` trace 数组做长度、重采样、延伸、裁剪或方向统一 | `trace_ops.clean_trace(...)`、`trace_length(...)`、`resample_trace(...)`、`extend_trace(...)`、`trim_trace(...)`、`orient_trace(...)` | 纯函数数值内核，输入应是投影后的 `x/y`，单位通常为 km。 |
| 对任意 `x/y` trace 数组简化、平滑或缓冲 | `trace_ops.simplify_trace(...)`、`smooth_trace(...)`、`buffer_trace(...)` | 适合脚本化批处理；`ecat-fault-trace-tool` 也复用同一套底层算法。 |
| 设置、读取或保存 CSI trace | `fault.trace(...)`、`fault.file2trace(...)`、`fault.writeTrace2File(...)` | 处理两列 `lon lat` 或局部 `x y` 迹线，不表示 patch。 |
| 地表 trace 等距离散 | `fault.discretize_trace(every=...)` | 生成 `xi/yi/loni/lati`；新项目优先使用，不再用 legacy `discretize(...)`。 |
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
ecat-fault-trace-tool trim input_trace.txt \
  --end-lon 101.8 \
  -o trace_trimmed.txt
```

如果要在脚本中批量控制方向、裁剪、端点延伸和采样间隔，优先使用 `TracePath`。它负责投影和
marker 解析，底层仍复用 `trace_ops` 纯函数：

```python
from eqtools.csiExtend import TracePath

trace = TracePath.from_lonlat(trace_lonlat, projection=fault)
prepared = (
    trace
    .orient(start="west")
    .trim(end={"longitude": 101.8})
    .extend(end_km=8.0, tangent_window=3)
    .resample(every_km=1.0)
)
fault.trace(prepared.lonlat[:, 0], prepared.lonlat[:, 1])
```

`TracePath` 只借用 fault 的投影，不修改 fault。处理必须发生在 top/bottom、mesh、patch 和 GF 构建前；
完整 marker、I/O、YAML 和底层函数语义见 [断层迹线处理参考](fault_trace_processing.md)。

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

<a id="nonlinear-result"></a>
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

这一输入路线分成两个明确模式：

| 模式 | 倾角 | 下倾方向 | 适用场景 |
| --- | --- | --- | --- |
| 固定倾角 | 一个 `dip_angle` | 一个显式 `dip_direction` | 近似平面、走向较直、需要最简单可控几何 |
| 沿走向变化倾角 | 多个 `xydip` 控制点插值到顶部节点 | 可使用一个代表性 strike，或使用每个顶部节点的局部 strike | 倾角沿走向有可靠变化；下倾方向复杂度按 trace 形态选择 |

“沿走向变化”与“随深度变化”不是一回事。若同一走向位置的倾角还随深度改变，使用
[倾角随深度变化](#layered-dip) 的 layered-dip 路线。

两种模式共同需要：

- `fault_trace.txt`：默认是两列 `lon lat`，点顺序定义 trace 方向。
- `top/depth`：最终 fault surface 顶部和底部深度。

如果 trace 已经是局部 `x y` 坐标，先转换成经纬度再调用 `fault.trace(...)`，或直接构造 `[x, y, top]` 后使用 `set_top_coords(..., lonlat=False)`。

<a id="trace-dip-single"></a>

### 模式 A：固定倾角和显式下倾方向

`dip_angle` 是相对水平面的倾角；`dip_direction` 是底边相对顶边移动的地理方位角。
`generate_bottom_from_single_dip(...)` 直接使用这个方向，不会根据 trace 自动判断倾向，
也不会强制它与 strike 垂直。因为倾向已经由 `dip_direction` 显式给出，这一模式只接受物理倾角
`0 < dip_angle <= 90°`；不要再用正负号或 `90°–180°` 表示另一侧。

若采用右手规则，并令 `strike` 沿 trace 正点序，则通常使用：

```text
dip_direction = (strike + 90°) mod 360°
```

例如正走向为 `90°`（向东）时，右手侧下倾方向为 `180°`（向南）。左侧下倾则使用
`(strike - 90°) mod 360°`。曲线 trace 使用一个固定 `dip_direction` 时，整个底边会沿同一
水平参考方向平移；因此应选择并记录一个代表性 strike，并在建网格前画出 top/bottom 检查。

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

# Example: representative strike is eastward (90°). Replace after checking
# the ordered trace. Right-hand-rule dip direction is strike + 90°.
reference_strike = 90.0
dip_direction = (reference_strike + 90.0) % 360.0
fault.generate_bottom_from_single_dip(
    dip_angle=70.0,
    dip_direction=dip_direction,
)
fault.generate_mesh(top_size=1.0, bottom_size=2.0, show=False, verbose=0)
fault.initializeslip(values="depth")
```

对 `top = fault.top`、`depth = fault.depth`、倾角 `δ` 和下倾方位角 `α`，底边水平位移为：

```text
Δz = depth - top
Δx = Δz / tan(δ) * sin(α)    # x 向东
Δy = Δz / tan(δ) * cos(α)    # y 向北
```

这也说明 `dip_direction=0°/90°/180°/270°` 分别向北/东/南/西移动。

<a id="trace-dip-varying"></a>

### 模式 B：沿走向变化倾角

推荐先把 trace 转成顶部边界并适度离散，再把多个倾角控制点插值到顶部节点。
走向处理分成三种可执行模式：

| 模式 | 设置 | 实际采用的逐节点 strike | 推荐场景 |
| --- | --- | --- | --- |
| 单一代表性走向 | `use_average_strike=True` | 所有节点使用同一个 strike；来源为显式 `user` 或自动 `pca` | 初次使用、近直线、走向变化不大 |
| top edge 自动局部走向 | 三列 `xydip`，并设置 `use_average_strike=False` | 由有序 `top_coords` 的相邻线段自动计算 | 曲线 trace，且几何切线就是所需走向 |
| 控制点走向插值 | 四列 `xydip` 含 `strike`，并设置 `use_average_strike=False` | 控制点 strike 经过圆周插值后写入每个顶部节点 | 已有分段地质走向或独立走向约束 |

第一种模式中的 `user` 和 `pca` 是同一“单一代表性走向”模式的两个来源，不另算两套几何。
第二、三种模式都必须保持 `use_average_strike=False`；若设为 `True`，单一代表性走向会覆盖
逐节点 strike。

三种模式共同以 `top_coords[0] -> top_coords[-1]` 作为正走向基准，所以初始 trace/top edge
点序非常重要：它决定 PCA 主轴的定向、自动局部 strike 的正方向，以及控制点 strike 的
一致性校验。若倾向侧整体相反，应先确认并反转 trace 点序，再重新插值和生成底边。

`xydip` 有三种等价容器形式：

1. `numpy.ndarray`：自动局部走向使用三列 `[lon, lat, dip]` 或 `[x, y, dip]`；控制点走向
   插值使用四列 `[lon, lat, strike, dip]` 或 `[x, y, strike, dip]`。
2. `pandas.DataFrame`：至少含坐标列和 `dip`；需要第三种模式时再增加 `strike`，也可以带
   其他说明列。
3. CSV 路径：第一行必须有同名表头，后续通过 `is_utm` 指明使用
   `lon/lat` 还是 `x/y`。

这里 `is_utm=True` 沿用历史参数名，实际要求的是与当前 fault object 相同投影下的
CSI `x/y`，单位 km。最稳妥的来源是 `x, y = fault.ll2xy(lon, lat)`；不要直接传入
单位为 m 的 UTM easting/northing。

最短的经纬度数组示例：

```python
dip_points = np.array([
    [96.00, 21.00, 55.0],
    [96.20, 21.10, 65.0],
    [96.45, 21.20, 72.0],
])

fault.depth = 25.0
interpolated = fault.interpolate_top_dip_from_relocated_profile(
    xydip=dip_points,
    is_utm=False,
    discretization_interval=2.0,  # km; also rediscretizes top_coords
    interpolation_axis="auto",   # PCA chooses x or y
)

# Recommended for a nearly straight trace: state the representative strike
# explicitly so the geometry is reproducible. This is strike, not dip direction.
representative_strike = 65.0
fault.generate_bottom_from_segmented_relocated_dips(
    fault_depth=fault.depth,
    use_average_strike=True,
    average_strike_source="user",
    user_direction_angle=representative_strike,
    verbose=True,  # prints the representative strike actually applied
)
fault.generate_mesh(top_size=1.0, bottom_size=2.0, show=False, verbose=0)
fault.initializeslip(values="depth")
```

`interpolated` 是带 `lon, lat, strike, dip` 的 `DataFrame`。其中 strike 是插值阶段得到的
逐节点候选值；若后续选择 `use_average_strike=True`，最终底边计算会再用 `user`/`pca` 的单一
代表性 strike 覆盖这些候选值，`verbose=True` 会打印实际采用值。无论最终选择单一还是逐点
strike，正倾角都向对应 strike 的右手侧，即 `dip_direction = strike + 90°`。

三列控制点、不使用统一走向时，strike 自动来自 top edge：

```python
fault.interpolate_top_dip_from_relocated_profile(
    dip_points,                 # [lon, lat, dip]
    is_utm=False,
    interpolation_axis="auto",
)
fault.generate_bottom_from_segmented_relocated_dips(
    fault_depth=fault.depth,
    use_average_strike=False,   # use local strike computed from top_coords
)
```

四列控制点可以同时约束逐点走向。下面的 `350° -> 10°` 会沿短路径经过 `0°` 插值，不会错误地
穿过 `180°`：

```python
strike_dip_points = np.array([
    [96.00, 21.00, 350.0, 55.0],  # lon, lat, strike, dip
    [96.20, 21.10,   0.0, 65.0],
    [96.45, 21.20,  10.0, 72.0],
])

controlled = fault.interpolate_top_dip_from_relocated_profile(
    strike_dip_points,
    is_utm=False,
    interpolation_axis="auto",
)
fault.generate_bottom_from_segmented_relocated_dips(
    fault_depth=fault.depth,
    use_average_strike=False,   # preserve interpolated control-point strikes
)
```

第三种模式会把插值后的 strike 保存到 `fault.top_strike` 并实质用于底边计算。每个插值节点的
strike 必须与该处 top edge 正点序处于同一方向半平面；反向或垂直会报错。这里仍然是
**strike 控制点**，不是 `dip_direction` 控制点。

#### 倾角的取值和正负号

多倾角模式需要用 strike 与 dip 的组合表达倾向，公开推荐使用带符号形式：

| 输入 dip | 几何含义 |
| --- | --- |
| `(0°, 90°)` | 向正 strike 的右手侧下倾 |
| `-90°` 或 `90°` | 垂直；左右侧没有水平差别 |
| `(-90°, 0°)` | 向正 strike 的左手侧下倾；计算时等价于 `strike + 180°` 和 `abs(dip)` |
| `(90°, 180°)` | 兼容形式，规范化为 `dip - 180°`；例如 `150° == -30°` |
| `0°` 或 `180°` | 不允许；水平面无法从 top 投影到不同的 `fault_depth` |

因此有效输入域是 `[-90°, 0°) ∪ (0°, 180°)`，但新脚本优先写成
`[-90°, 0°) ∪ (0°, 90°]`。内部插值把负倾角转换到 `(90°, 180°)`，使倾向换侧时经过
垂直 `90°`，而不是经过会造成除零的水平 `0°`；插值完成后再恢复为带符号倾角。

Bayesian 高级入口 `set_dip_control_points(...)` +
`perturb_dips_with_preset_params(...)` 使用同一个连续 `(0°, 180°)` proposal 坐标。参考倾角先
规范化，再施加扰动；例如 `77° + [-30°, 30°]` 的搜索范围是 `[47°, 107°]`，而等价输入
`-80°` 和 `100°` 加同一个 `-20°` 扰动都得到 `80°`。扰动后的候选若到达 `0°/180°` 或超出
该开区间会明确报错，不会回绕或静默换侧。

第三种模式中，控制点 `strike` 始终表示 top edge 的正走向并先接受方向校验；dip 的负号只负责
选择其左/右下倾侧。不要同时把 strike 加 `180°` 又把 dip 改成负值，否则会发生两次翻转。

#### 节点索引契约

控制点可以按插值轴临时排序，但输出数组始终回到有序 `top_coords` 的节点顺序：

```text
xydip controls
  -> dip/optional strike interpolation evaluated at top_coords[i]
  -> top_dip[i] + top_strike[i]
  -> bottom_coords[i] generated only from top_coords[i]
  -> paired top/bottom boundaries enter mesh generation
```

这里的 `top_strike/top_dip` 是生成底边时的 reference-node metadata；负 dip 对应的
`strike + 180°` 只在底边计算的局部副本中应用。最终 mesh/patch 的 canonical 走向和倾角必须
由实际顶点及 `getpatchgeometry()` 确定，不能直接把这两个元数据字段当作最终 patch 几何。

底边生成完成后，ECAT 按相同索引构造相邻四边形
`[top[i], top[i+1], bottom[i+1], bottom[i]]`，并沿固定对角线拆成两片三角形。对应法向为：

\[
\mathbf n_1=(T_{i+1}-T_i)\times(B_{i+1}-T_i),\qquad
\mathbf n_2=(B_{i+1}-T_i)\times(B_i-T_i).
\]

任一三角形真实退化，或 `n1 · n2 <= 0`，表示该局部单元已经折返/翻转，不能静默重排底边
来掩盖。计算使用完整三维坐标，因此垂直断层不会因其平面投影面积为零而误报。真实迹线中偶尔
出现的成对重复节点只作为冗余局部段跳过，原始节点顺序和 `top[i] <-> bottom[i]` 对应关系均不
改变；若整条边退化仍会失败。

直接几何构建遇到无效单元会给出单元索引并报错；FULLSMC、magnitude FULLSMC 和 SMC_FJ
在采样目标中只把这一类明确的几何错误解释为该候选 `-inf`。配置、索引和其他数值异常仍会
原样抛出，不会伪装成普通拒绝。该检查为 O(N) 的相邻单元检查，不搜索非相邻曲面全局相交，
也不自动排序或修复边界。

本节的等深线组合入口按目标等深线点序返回 top/bottom，不会猜测或静默重排输入行。
`reinterpolate=False` 要求 `xydip` 与 `isodepth` 具有相同逐行点数、经纬度和顺序；
若提供 `xydip.strike`，它必须与等深线点序的正方向位于同一方向半平面，并会实际进入
底边投影。负 dip 的 `strike + 180°` 只由底边生成器执行一次；不要在输入中预先反转 strike。

| 模式 | `xydip` | dip | strike |
| --- | --- | --- | --- |
| `reinterpolate=False` | `lon, lat, strike, dip` | 逐行直接使用 | **使用用户逐节点 strike** |
| `reinterpolate=False` | `lon, lat, dip` | 逐行直接使用 | 从有序等深线局部切线推导 |
| `reinterpolate=True` | `lon, lat, dip` | 插值到目标等深线 | 从目标等深线点序推导 |
| `reinterpolate=True` | 含 `strike` 或 profile-fit 元数据 | 仍只插值 dip | 输入 strike 明确忽略 |

因此 `reinterpolate` 只回答“dip 是否重新采样”，不等于“strike 是否使用”。若需要把用户
strike 控制点也插值到另一组 top 节点，应使用
`interpolate_top_dip_from_relocated_profile(...)`，不能使用这个等深线组合入口。

高级用户可直接复制下面的逐节点形式：

```python
isodepth = np.array([
    [100.00, 30.00, 5.0],
    [100.08, 30.00, 5.0],
    [100.16, 30.00, 5.0],
])
xydip = np.array([
    [100.00, 30.00, 90.0, -80.0],  # lon, lat, reference strike, dip
    [100.08, 30.00, 90.0, -80.0],
    [100.16, 30.00, 90.0, -80.0],
])

top_coords, bottom_coords = fault.generate_top_bottom_from_isodepth_and_dip(
    isodepth,
    xydip,
    top_depth=0.0,
    bottom_depth=10.0,
    reinterpolate=False,
    isodepth_tolerance=0.1,
)
```

如果没有逐节点 strike，可直接使用同顺序三列输入：

```python
direct_xydip = xydip[:, [0, 1, 3]]  # lon, lat, dip
```

此时不会插值 dip，局部 strike 由有序 `isodepth` 切线推导；若提供四列，用户 strike
仍会实际参与投影。因此三列是自动局部走向，四列是用户控制逐节点走向。

`isodepth` 必须表示一条近似等深曲线。接口在任何倾角插值之前检查
`max(depth) - min(depth) <= isodepth_tolerance`；默认容差是 `0.1 km`，也可显式调整。
直接逐节点分支保留容差内各节点的实际参考深度。若启用重插值，由于输出节点可能已经重采样，
接口使用原等深曲线深度的中位数作为该层代表深度。

稀疏倾角控制点不需要与等深曲线点数一致，推荐提供 `lon, lat, dip` 三列；局部 strike
由目标等深线点序计算：

```python
sparse_xydip = np.array([
    [100.00, 30.00, -80.0],  # lon, lat, dip
    [100.16, 30.00, -70.0],
])

top_coords, bottom_coords = fault.generate_top_bottom_from_isodepth_and_dip(
    isodepth,
    sparse_xydip,
    top_depth=0.0,
    bottom_depth=10.0,
    reinterpolate=True,
    interpolation_axis="auto",  # PCA on the target isodepth; "x"/"y" are explicit
    isodepth_tolerance=0.1,
)
```

`reinterpolate=True` 的职责是把 dip 重新采样到目标节点。输入表即使含有 strike 或
`profile_index/mse/method` 等拟合元数据，也只选择和插值 dip；输入 strike 会被明确忽略。
如需插值用户 strike 控制点，请改用
`interpolate_top_dip_from_relocated_profile(...)`。

`interpolation_axis` 只接受 `auto | x | y`，拼写错误会立即报错。`auto` 在目标等深线上
执行一次 PCA，同一个解析轴同时用于 buffer 和最终插值。若目标点在该轴上全部落在控制范围
之外，程序会保持既有“最近端点填充”数值行为，但发出 `RuntimeWarning`，列出实际参与判断的
CSI x/y 范围（km）、控制点和目标曲线的 lon/lat 范围，以及被采用的端点 dip。

`dip=-80°` 与 `dip=100°` 在这里等价；`0°/180°`、非有限值和越界值会在 mesh 生成前
失败。direct 四列的反向 strike 或 direct 行错位也会提前失败。稀疏控制点应使用
`reinterpolate=True`，不要伪造逐行对应。

DataFrame 形式：

```python
import pandas as pd

dip_points = pd.DataFrame(
    {
        "lon": [96.00, 96.20, 96.45],
        "lat": [21.00, 21.10, 21.20],
        "dip": [55.0, 65.0, 72.0],
    }
)
fault.interpolate_top_dip_from_relocated_profile(dip_points, is_utm=False)
```

CSV 形式：

```csv
lon,lat,dip
96.00,21.00,55.0
96.20,21.10,65.0
96.45,21.20,72.0
```

若要使用控制点走向插值，只需增加 `strike` 列：

```csv
lon,lat,strike,dip
96.00,21.00,350.0,55.0
96.20,21.10,0.0,65.0
96.45,21.20,10.0,72.0
```

```python
fault.interpolate_top_dip_from_relocated_profile(
    "dip_controls.csv",
    is_utm=False,
    interpolation_axis="auto",
)
```

投影坐标数组形式：

```python
x, y = fault.ll2xy(control_lon, control_lat)  # km, same projection as fault
dip_points_xy = np.column_stack((x, y, control_dip))
fault.interpolate_top_dip_from_relocated_profile(
    dip_points_xy,
    is_utm=True,
)
```

如果 CSV 来自重定位余震剖面拟合，还可以带 `profile_index`、`method` 和 `mse` 列。
此时 `method="min_mse"` 会对每个 profile 选择最小 MSE 的倾角；也可以通过
`profiles_to_keep` 或 `profiles_to_remove` 筛选 profile。普通三列控制点没有多套拟合结果，
不需要调整 `method`。

参数选择要点：

- `interpolation_axis="auto"` 只是在投影 `x/y` 中用 PCA 选择主轴，再做一维插值；它不是
  沿曲线弧长插值。强弯曲、回折或在 x/y 上不单调的 trace 应分段处理并检查结果。
- 对初次使用、近直线或走向变化不大的 trace，优先使用 `use_average_strike=True`。它让所有
  顶部节点采用同一个代表性 strike，生成的下倾方向更容易检查和复现。
- 已知地质走向或已经从 trace 统计出代表性走向时，推荐同时显式设置
  `average_strike_source="user"` 和 `user_direction_angle=<strike>`。这里设置的是 **strike**，
  不是 `dip_direction`；底层会在本次底边计算中用它覆盖所有逐节点 strike。
- `average_strike_source="pca"` 是无需人工给角度的快捷模式。它对当前顶部节点的投影 `x/y`
  做 PCA，取第一主轴作为统一 strike，并用首末点方向消除 `180°` 二义性。它是 PCA 主轴，
  不是逐点 strike 的算术平均；建议显式写出 source，避免读配置或脚本时误判来源。
- `average_strike_source="user"` 时，`user_direction_angle` 按北为 `0°`、顺时针为正解释，且必须
  与 trace 首点到末点的正方向一致。反向或垂直方向会报错；若实际点序相反，应先反转 trace，
  而不是只把走向角加 `180°`。
- 当 trace 明显弯曲，并且确实希望下倾方向随走向变化时，使用 `use_average_strike=False`。
  三列 `xydip` 保留由有序 `top_coords` 计算的局部 strike；四列 `xydip` 保留由控制点圆周插值
  得到的逐节点 strike。此时 `average_strike_source` 和 `user_direction_angle` 不参与计算。
- `calculate_strike_along_trace=False` 已弃用且会被忽略；需要反向 strike 时直接反转
  `top_coords`/trace 点序。
- 常规输入优先使用 `(0°, 90°]`，并通过 trace 点序控制右手侧；确需在同一正走向下表示左侧时
  使用负倾角。`(90°, 180°)` 只作兼容输入，不建议与带符号形式在同一文件中混写。

<a id="fixed-topology-dip-search"></a>

## 倾角搜索的参考网格与固定拓扑

倾角搜索比较的是不同物理几何。如果每个候选都重新剖分，patch 数量、位置和编号也可能
改变，拟合差异会混入离散化差异。`generate_and_deform_mesh(...)` 用一套参考参数坐标解决
这个问题：

```python
reference_dip = 65.0
candidate_dips = [50.0, 60.0, 70.0, 80.0]
dip_direction = 180.0

mapping_num_segments = 30
mapping_disct_z = 10

fault.generate_bottom_from_single_dip(reference_dip, dip_direction)
fault.generate_and_deform_mesh(
    fault.top_coords,
    fault.bottom_coords,
    top_size=1.0,
    bottom_size=2.0,
    num_segments=mapping_num_segments,
    disct_z=mapping_disct_z,
    remap=True,
    show=False,
    verbose=0,
)
reference_npatch = fault.numpatch

for dip in candidate_dips:
    fault.generate_bottom_from_single_dip(dip, dip_direction)
    fault.generate_and_deform_mesh(
        fault.top_coords,
        fault.bottom_coords,
        top_size=1.0,
        bottom_size=2.0,
        num_segments=mapping_num_segments,
        disct_z=mapping_disct_z,
        remap=False,
        show=False,
        verbose=0,
    )
    if fault.numpatch != reference_npatch:
        raise RuntimeError("fixed-topology dip search changed patch count")
```

参数契约：

- `remap=True` 在参考倾角上生成一次 Gmsh 网格，并建立网格顶点到规则参数坐标的映射。
- `remap=False` 使用既有映射更新物理坐标，不重新定义 patch 身份。
- `num_segments` 和 `disct_z` 控制参考坐标映射；它们不直接等于最终 patch 数，但候选间必须一致。
- `top_size`、`bottom_size`、`field_size_dict`、`mesh_func`、迹线点序和顶部/底部控制点数量也应保持一致。
- 每个候选都必须重新初始化 slip，并新建 inversion 对象；GF、Laplacian、边界与 rake 约束不能跨几何复用。
- 至少检查 `numpatch`、`Faces` 和 patch 行顺序保持一致，再比较逐数据集 RMS/VR、粗糙度和滑动分布。

可直接运行的循环结构见[固定拓扑倾角搜索工作流](../workflows/04b_blse_dip_search.md)。

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

这三个高层入口都把 `dip` 解释为**带侧别的参考倾角**：推荐直接写
`[-90, 0) U (0, 90]`，也接受等价的 `(90, 180)` 连续表达。例如 `-80°` 与 `100°`
表示同一参考侧，入口会在插值前统一到内部 `(0, 180)`；`0°/180°`、非有限值和超范围
输入会立即报错。

`set_depth_dip_from_function(...)` 的回调接收 signed base dip，因此即使参考节点写
`100°`，回调收到的仍是等价的 `-80°`。回调只在 `num_depth_samples` 个存储深度上各
计算一次，生成的离散 profile 是之后插值、导出和读回的权威数据，不会在查询时重复调用：

```python
def dip_by_depth(base_dip, depth):
    return base_dip + 0.25 * (depth - fault.top)

fault.set_depth_dip_from_function(
    reference_nodes=np.array([[lon0, lat0, -60.0]]),
    depth_function=dip_by_depth,
    depth_range=(fault.top, fault.depth),
    num_depth_samples=8,
)
```

`num_depth_samples` 必须是至少为 2 的整数，`depth_range` 必须有限且递增。离散 profile
要求每个节点至少两个不重复深度；`cubic` 至少需要四个深度点。低层
`DepthDipProfile(..., input_dip_range=...)` 只接受明确的 `neg90_90` 或 `0_180` 协议，
一般用户应优先调用上面的 `set_depth_dip_*` 高层入口，避免自行管理内部数值域。

每个 `depth, dip` 表示该深度处相对统一参考走向的**绝对倾角**，不是相对上一层的角度增量。
正 dip 向参考走向右手侧下倾，负 dip 向另一侧下倾；连续多个负 dip 会持续位于同一侧，不会
随网格层数交替翻转。节点位置逐层递推，但每段实际倾向都由该段 dip 的符号重新确定。

如果需要结构化矩形元，可使用 `AdaptiveLayeredDipRectangularPatches`，通过深度-倾角剖面构建
沿走向和下倾方向规则分块：

```python
from eqtools.csiExtend.AdaptiveLayeredDipRectangularPatches import (
    AdaptiveLayeredDipRectangularPatches as LayeredRectFault,
)

fault = LayeredRectFault("LayeredRect", lon0=lon0, lat0=lat0, verbose=False)
fault.top = 0.0
fault.trace(trace[:, 0], trace[:, 1])
fault.set_depth_dip_from_profiles(
    {
        "reference_nodes": reference_nodes,
        "depth_dip_profiles": depth_dip_profiles,
    },
    interpolation_method="linear",
)
fault.buildPatches(
    width=30.0,
    numz=6,
    every=5.0,
    dipdirection=None,  # positive-dip reference: local strike + 90 degrees
)
```

显式 `dipdirection` 是正 dip 对应的参考下倾方位角，按北为 `0°`、顺时针为正解释。当前段
dip 为负时，该段实际倾向为 `dipdirection + 180°`。`dip_at_nodes` 保留带符号参考倾角；
出现负参考值的 patch 会从最终几何派生正的 physical `patchdip`，全正输入则保留既有 corner
mean。最终 patch 的 canonical strike/dip 仍应读取 `getpatchgeometry()`。改变 `numz` 只能改变
离散精度，不能改变固定倾角曲面的最终位置。

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
