# 由地表迹线和倾角构建三角断层

这个示例从两列 `lon lat` 地表迹线出发，给出线性滑动反演最常用的三条可复制路线：

| 已有信息 | 使用路线 |
| --- | --- |
| 一条地表迹线和一个倾角 | [单倾角平面](#single-dip) |
| 一条地表迹线和多个倾角参考点 | [沿走向变化倾角](#multiple-dips) |
| 需要逐个比较候选倾角 | [固定参考拓扑的倾角搜索](#fixed-topology) |

如果已有非线性几何反演结果，改用
[由非线性几何结果构建断层](fault_from_nonlinear_geometry.md)。如果倾角还随深度变化，使用
[Fault Geometry Construction：倾角随深度变化](../reference/fault_geometry_construction.md#layered-dip)。

## 公共初始化

`fault_trace.txt` 至少包含两列 `lon lat`。点序决定 strike 的正方向，也会影响右手规则下的
下倾侧，因此先画出迹线并确认首尾顺序。

```python
import numpy as np
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)

lon0, lat0 = 96.2, 21.1
trace_lonlat = np.loadtxt("fault_trace.txt", ndmin=2)[:, :2]

fault = TriFault("MainFault", lon0=lon0, lat0=lat0, verbose=False)
fault.top = 0.0
fault.depth = 20.0
fault.trace(trace_lonlat[:, 0], trace_lonlat[:, 1], utm=False)
fault.set_top_coords_from_trace()
```

本页角度均为地理方位角：北为 `0°`、东为 `90°`，从北顺时针增加。

<a id="single-dip"></a>

## A. 地表迹线 + 单倾角

给定代表性走向和物理倾角后，用右手规则计算下倾方向：

```python
representative_strike = 65.0
dip_direction = (representative_strike + 90.0) % 360.0

fault.generate_bottom_from_single_dip(
    dip_angle=70.0,
    dip_direction=dip_direction,
)
fault.generate_mesh(top_size=1.0, bottom_size=2.0, show=False, verbose=0)
fault.initializeslip(values="depth")
```

`dip_direction` 不会自动从迹线推导。使用前应确认迹线点序、`representative_strike` 和实际
下倾侧一致；若需要向 strike 左侧下倾，应直接给出正确的 `dip_direction`。

<a id="multiple-dips"></a>

## B. 地表迹线 + 多个倾角参考点

控制点使用与 fault 一致的坐标约定。下面的三列数组是 `[lon, lat, dip]`，因此必须设置
`is_utm=False`：

```python
dip_points = np.array([
    [96.00, 21.00, 55.0],
    [96.20, 21.10, 65.0],
    [96.45, 21.20, 72.0],
])

dip_table = fault.interpolate_top_dip_from_relocated_profile(
    dip_points,
    is_utm=False,
    discretization_interval=2.0,
    interpolation_axis="auto",
)

representative_strike = 65.0
fault.generate_bottom_from_segmented_relocated_dips(
    fault_depth=fault.depth,
    use_average_strike=True,
    average_strike_source="user",
    user_direction_angle=representative_strike,
    verbose=True,
)
fault.generate_mesh(top_size=1.0, bottom_size=2.0, show=False, verbose=0)
fault.initializeslip(values="depth")
```

`dip_table` 包含插值后的 `lon/lat/strike/dip`，建议先检查再生成 mesh。这里的
`user_direction_angle` 是 **strike**，不是下倾方向。近直线但没有可靠走向角时，可改为
`average_strike_source="pca"`；只有明显弯曲且确实希望下倾方向沿迹线变化时，才使用
`use_average_strike=False`。

若控制点已经是 fault 局部投影下的 `x y dip`（单位 km），应改用 `is_utm=True`。不要把
经纬度数组和投影坐标开关混用。四列 `[lon, lat, strike, dip]`、CSV、DataFrame 和带符号倾角
的完整约定见
[Fault Geometry Construction：沿走向变化倾角](../reference/fault_geometry_construction.md#trace-dip-varying)。

<a id="fixed-topology"></a>

## C. 倾角搜索时保持参考坐标和拓扑一致

若要用 BLSE 比较多个倾角，不能让每个候选角重新生成一套不同的三角形。先在参考倾角上
建立一次网格与参数坐标映射，后续候选只变形现有拓扑：

```python
reference_dip = 70.0
candidate_dips = [50.0, 60.0, 70.0, 80.0]
dip_direction = 155.0

mapping_num_segments = 30
mapping_disct_z = 10

fault.generate_bottom_from_single_dip(
    dip_angle=reference_dip,
    dip_direction=dip_direction,
)
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
    fault.generate_bottom_from_single_dip(
        dip_angle=dip,
        dip_direction=dip_direction,
    )
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

    fault.initializeslip(values="depth")
    # 在这里为当前 dip 新建 inversion，并重新组装 GF、Laplacian 和约束。
```

`remap=True` 只在参考几何上调用一次。各候选必须保持相同的迹线点序、`num_segments`、
`disct_z` 和网格参数；每个候选都应新建 inversion 对象，因为 GF、Laplacian 和约束依赖当前
几何。完整搜索流程见[固定拓扑倾角搜索](../workflows/04b_blse_dip_search.md)，精确参数约定见
[Fault Geometry Construction：固定拓扑](../reference/fault_geometry_construction.md#fixed-topology-dip-search)。

## 可选：先清理和重采样迹线

只有迹线需要统一方向、裁剪、端点延伸或重采样时，才在“公共初始化”之前加入：

```python
from eqtools.csiExtend import TracePath

trace = TracePath.from_lonlat(trace_lonlat, lon0=lon0, lat0=lat0)
prepared = (
    trace
    .orient(start="west")
    .simplify(method="vw", tolerance=0.2)
    .trim(start={"trace_distance_km": 2.0}, end={"longitude": 101.8})
    .extend(end_km=8.0, tangent_window=3)
    .resample(every_km=1.0)
)
trace_lonlat = prepared.lonlat
```

这些长度参数都在局部投影坐标中解释，单位为 km；示例值只是占位值，应按迹线尺度调整。经度、
纬度、最近点、多交点选择和命令行用法见专门的
[断层迹线预处理短例](fault_trace_processing.md)；不要在已经生成 mesh、patch 或 GF 后替换 trace。

## 检查

```python
from eqtools.csiExtend import print_fault_summary

print_fault_summary(fault)
```

至少检查迹线首尾方向、顶部和底部深度、下倾侧、patch 数量及平均走向/倾角，并画出
top、bottom 和 mesh。随后可按
[反演前读取 InSAR 与 GNSS 数据](inversion_data_loading.md)准备 `geodata`，再进入
[BLSE/VCE 线性滑动反演](../workflows/04_linear_slip_blse_vce.md)。

## 何时不用本例

- 已有 top/bottom 三维曲线时，优先统一两条曲线的点数后直接建 mesh。
- 多条等深线或 slab 几何应交给 `FaultGeometryEngine` 管理 layers。
- 倾角随深度变化时，使用 `AdaptiveLayeredDipTriangularPatches`。

相关参考：
[Fault Geometry Construction](../reference/fault_geometry_construction.md)、
[Fault Summary](../reference/fault_summary.md)、
[Fault Contours](../reference/fault_contours.md)。
