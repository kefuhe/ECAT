# 密集三维地表形变正演 / Surface Displacement Forward

本文说明如何用 ECAT 依赖栈中的 fault object 计算密集地表三维形变场。典型用途是：已经有断层几何和滑动量后，生成规则网格或自定义点上的 `east, north, up` 位移。

推荐入口是：

```python
obs_pts, disp = fault.compute_surface_displacement(...)
```

这个接口不是 InSAR 降采样接口，也不要求先建立 InSAR/GPS 数据对象。它直接对给定地表点做三维位移正演，并可保存 HDF5 或 TXT 结果。

如果需要多断层求和、LOS 投影或统一保存格式，可以使用 ECAT 的薄封装函数：

```python
from eqtools.csiExtend.surface_forward import (
    compute_multifault_surface_displacement,
    project_enu_to_los,
    save_surface_forward_h5,
    save_surface_forward_txt,
    save_lonlat_regular_geotiff,
    save_raster_like_geotiff,
)
```

这些函数不替代底层 `fault.compute_surface_displacement()`，只是把常见的多断层、SAR LOS 和输出组织逻辑集中起来，避免用户脚本重复写相同代码。

可直接参考两个脚本模板：

| 模板脚本 | 用途 |
| --- | --- |
| `scripts/test_surface_displacement_forward.py` | 通用 ENU 三分量正演：多断层、`box` 网格或自定义 lon/lat 点、HDF5/TXT 输出 |
| `scripts/test_sar_los_surface_forward.py` | SAR LOS 正演：以 HyP3 为例读取有效像元和投影向量，计算 ENU 后投影为 LOS，并保存 GeoTIFF/PNG |

`test_sar_los_surface_forward.py` 默认只重绘已有 GeoTIFF，不重新正演：

```powershell
python scripts/test_sar_los_surface_forward.py
```

第一次运行或断层/SAR 输入改变后，用 `-r` 重新计算并生成 GeoTIFF，然后自动绘制 PNG：

```powershell
python scripts/test_sar_los_surface_forward.py -r
```

如果只想重新生成 GeoTIFF、不绘制 PNG：

```powershell
python scripts/test_sar_los_surface_forward.py -r --no-plot
```

`--show` 可在保存 PNG 后同时屏显；`--pattern` 可指定默认重绘模式下要匹配的 GeoTIFF 文件名模式。这个组织方式让慢速正演和快速重绘分开，便于反复调整色标、尺寸和输出图件。

SAR LOS 模板的 `SAVE_GEOREFERENCE_MODE="auto"` 会优先尝试用 reader 中的 `sar.raw_mesh_lon/raw_mesh_lat` 写规则经纬度 GeoTIFF；如果网格不是普通 affine lon/lat 栅格，则回退到复制参考 phase tif 的 georeferencing metadata。正演计算始终使用 reader 生成的观测点坐标，GeoTIFF 轴是否显示经纬度取决于输出文件本身是否带有可靠的 `crs/transform/bounds`。

## 何时使用

适合使用本接口的场景：

- 线性滑动反演完成后，想生成密集 `east/north/up` 形变场。
- 想在规则 lon/lat 网格上画三分量地表形变。
- 想对一批自定义地表点正演三维位移。
- 想比较三角元和矩形元断层模型的地表位移。

如果目标是对已有 InSAR/GPS 数据点生成合成观测值，应优先看对应 geodata 的 Green's functions 和 synthetic workflow；如果目标是密集地表三分量形变场，则使用本文接口。

## 支持对象

三角元和常规矩形元 fault object 都可以使用：

```python
obs_pts, disp = fault.compute_surface_displacement(...)
```

常见对象包括：

| 对象类型 | 说明 |
| --- | --- |
| `csi.TriangularPatches` | CSI 三角元断层对象 |
| `eqtools.csiExtend.AdaptiveTriangularPatches` | ECAT 三角元拓展对象 |
| `eqtools.csiExtend.BayesianAdaptiveTriangularPatches` | ECAT Bayesian 三角元对象，常用于非线性几何和线性滑动工作流 |
| `csi.RectangularPatches` | CSI 矩形 patch 基类 |
| `csi.planarfault.planarfault` | 常用平面矩形断层类 |
| `csi.faultwithlistric.faultwithlistric` | listric 矩形断层类 |
| `eqtools.csiExtend.AdaptiveRectangularPatches` | ECAT 矩形元拓展类 |
| `eqtools.csiExtend.AdaptiveLayeredDipRectangularPatches` | ECAT 分层倾角矩形元拓展类 |

关键不是类名本身，而是对象是否具备可正演的 fault geometry 和 slip。对于矩形元对象，只要继承自 `RectangularPatches` 系列，通常就同时具备直接计算和显式转三角元两种路径。

返回量含义：

| 返回量 | 形状 | 含义 |
| --- | --- | --- |
| `obs_pts` | `(N, 3)` | 观测点坐标。默认 `output_coords="lonlat"` 时为 `lon, lat, depth/z` |
| `disp` | `(N, 3)` | 三维位移，列顺序为 `east, north, up` |

位移单位为米。坐标中 `lon/lat` 为度，`depth/z` 为 km；地表点通常为 `0`。

## 三种计算路径

### 三角元对象直接计算

如果 fault object 本身是三角元对象，直接调用：

```python
obs_pts, disp = tri_fault.compute_surface_displacement(
    box=[lon_min, lon_max, lat_min, lat_max],
    npoints=300,
    method="cutde",
    output_file="dense_surface_disp.h5",
)
```

这会直接使用对象已有的三角网格、`Faces`、`Vertices` 和 `slip`。

### 矩形元对象直接计算

如果 fault object 是常规矩形元对象，也可以直接调用同一个接口：

```python
obs_pts, disp = rect_fault.compute_surface_displacement(
    box=[lon_min, lon_max, lat_min, lat_max],
    npoints=300,
    method="cutde",
    output_file="dense_surface_disp.h5",
)
```

这不是另一套不同的物理计算逻辑。矩形元接口内部会把每个矩形 patch 拆成两个三角形，把该矩形 patch 的 slip 复制给两个三角形，然后调用三角元正演核心。普通用户优先使用这条路径。

### 矩形元显式转三角元后计算

如果需要检查三角化结果、保存三角元对象，或后续统一使用三角元工具，可以显式转换：

```python
tri_fault = rect_fault._rect2triangular(return_tri_fault=True)

obs_pts, disp = tri_fault.compute_surface_displacement(
    box=[lon_min, lon_max, lat_min, lat_max],
    npoints=300,
    method="cutde",
    output_file="dense_surface_disp_tri.h5",
)
```

这一路径更适合开发者或高级用户。普通正演任务不需要先做这一步。

如果脚本里 fault object 来源不固定，可以用能力判断：

```python
if getattr(fault, "patchType", None) == "triangle":
    tri_fault = fault
elif hasattr(fault, "_rect2triangular"):
    tri_fault = fault._rect2triangular(return_tri_fault=True)
else:
    raise TypeError(
        "This fault object is neither triangular nor convertible rectangular."
    )
```

## 最小示例

规则网格正演：

```python
obs_pts, disp = fault.compute_surface_displacement(
    box=[lon_min, lon_max, lat_min, lat_max],
    npoints=300,
    method="cutde",
    output_file="dense_surface_disp.h5",
    output_format="h5",
    output_coords="lonlat",
)
```

取出结果：

```python
lon = obs_pts[:, 0]
lat = obs_pts[:, 1]
east = disp[:, 0]
north = disp[:, 1]
up = disp[:, 2]
```

如果需要恢复规则网格：

```python
east_grid = east.reshape((300, 300))
north_grid = north.reshape((300, 300))
up_grid = up.reshape((300, 300))
```

## 多断层通用接口

如果有多个断层对象，推荐使用 `compute_multifault_surface_displacement()` 组织求和：

```python
from eqtools.csiExtend.surface_forward import (
    compute_multifault_surface_displacement,
    save_surface_forward_h5,
)

faults = {
    "F1": fault_1,
    "F2": fault_2,
}

result = compute_multifault_surface_displacement(
    faults,
    box=[lon_min, lon_max, lat_min, lat_max],
    npoints=300,
    method="cutde",
    target_mem_gb=4.0,
    max_obs_batch=50000,
)

save_surface_forward_h5(
    "surface_displacement_enu.h5",
    result,
    include_by_fault=True,
)
```

返回的 `result` 包含：

| 属性 | 含义 |
| --- | --- |
| `result.obs_pts` | 观测点坐标 |
| `result.disp_total_enu` | 所有断层叠加后的 ENU 位移 |
| `result.disp_by_fault_enu` | 每个断层单独贡献的 ENU 位移 |
| `result.fault_names` | 断层名顺序 |
| `result.metadata` | 计算方法、泊松比、采样方式等轻量元信息 |

如果只计算一个断层，直接使用 `fault.compute_surface_displacement()` 仍然是最短路径；如果需要多断层叠加、统一保存或后续投影，则使用这个薄封装更方便。

## 读入矩形元结果后正演

如果已有矩形元 slip 结果文件，可以先读入矩形元对象，再选择直接计算或显式转三角元：

```python
from csi import RectangularPatches

rect_fault = RectangularPatches("rect_fault", lon0=lon0, lat0=lat0)
rect_fault.readPatchesFromFile("slip_rect.gmt", readpatchindex=True)

# 路径 1：矩形元对象直接计算
obs_pts, disp = rect_fault.compute_surface_displacement(
    box=[lon_min, lon_max, lat_min, lat_max],
    npoints=300,
    method="cutde",
    output_file="dense_surface_disp.h5",
)

# 路径 2：显式转成三角元对象后计算
tri_fault = rect_fault._rect2triangular(return_tri_fault=True)
obs_pts, disp = tri_fault.compute_surface_displacement(
    box=[lon_min, lon_max, lat_min, lat_max],
    npoints=300,
    method="cutde",
    output_file="dense_surface_disp_tri.h5",
)
```

上面的 `RectangularPatches` 可以替换为具体的 ECAT/CSI 矩形类，例如 `planarfault`、`AdaptiveRectangularPatches` 或 `AdaptiveLayeredDipRectangularPatches`。

如果矩形元 GMT 文件由常规 `writePatches2File(add_slip=...)` 写出，文件头通常会保留 `strike-slip, dip-slip, tensile` 三个 slip 分量，`readPatchesFromFile()` 可以恢复 `rect_fault.slip`。如果文件只包含几何而没有 slip 分量，需要在正演前显式设置 `rect_fault.slip`，或在调用时传入三列 `slipVec`。

## 自定义密集点

如果接收点不是规则网格，可以传入 `lonlat=(lon_array, lat_array)`：

```python
obs_pts, disp = fault.compute_surface_displacement(
    lonlat=(lon_array, lat_array),
    method="cutde",
    output_file="dense_surface_disp.txt",
    output_format="txt",
    output_coords="lonlat",
)
```

`lon_array` 和 `lat_array` 应为长度相同的一维数组。接口内部会把 lon/lat 转换到 fault object 使用的局部投影坐标，再做正演。

## SAR LOS 投影

SAR reader 一般会提供有效像元的 `lon/lat` 和 ENU 投影向量。ECAT 约定：

```text
scalar_observation = ENU_displacement dot projection
```

计算流程是先算 ENU，再投影到 LOS：

```python
from eqtools.csiExtend.surface_forward import (
    compute_multifault_surface_displacement,
    project_enu_to_los,
    save_lonlat_regular_geotiff,
    save_raster_like_geotiff,
)

result = compute_multifault_surface_displacement(
    faults,
    lonlat=(sar.lon, sar.lat),
    method="cutde",
    output_coords="xy",
)

total_los = project_enu_to_los(result.disp_total_enu, sar)

save_raster_like_geotiff(
    "total_los_m.tif",
    total_los,
    reference_raster="unwrapped_phase.tif",
    valid_index=sar.projection_raw_valid_index,
)
```

这里 `sar` 可以是 HyP3、GMTSAR、GAMMA 或其他 reader 结果，只要它提供有效点的 `lon`、`lat`、`los` 和用于写回原始栅格的有效像元索引。不同 SAR 数据格式的读取逻辑应留在脚本中，正演和投影逻辑保持一致。

如果参考 tif 的 georeferencing metadata 不可靠，`save_raster_like_geotiff()` 会保留这种限制并给出提示。对于规则经纬度网格，可以改用：

```python
save_lonlat_regular_geotiff(
    "total_los_m.tif",
    total_los,
    sar.raw_mesh_lon,
    sar.raw_mesh_lat,
    valid_index=sar.projection_raw_valid_index,
)
```

这个写法只适合 affine lon/lat 栅格；曲线网格应保存为 NetCDF/xarray 这类能携带二维坐标的格式。

如果已经保存了 GeoTIFF，只想重绘或调整 quick-look 图，不需要重新计算正演。可以直接使用 `viztools` 的通用栅格入口：

```python
from eqtools.viztools import plot_geotiff

fig, ax, im = plot_geotiff(
    "total_los_m.tif",
    symmetric=True,
    percentile=99,
    axis="geo",
    axis_max_major_ticks=5,
    colorbar_label="LOS displacement (m)",
    colorbar_max_major_ticks=4,
    save="total_los_m.png",
    show=True,
)
```

这个绘图入口只负责显示已经准备好的二维栅格，不解释 SAR 正负号，也不重新计算 LOS 投影。

## Slip 设置

如果 fault object 已经有正确的 `fault.slip`，可以不传 `slipVec`。更显式的写法是传入三列：

```python
slipVec = np.column_stack([
    strike_slip,
    dip_slip,
    np.zeros_like(strike_slip),
])

obs_pts, disp = fault.compute_surface_displacement(
    box=[lon_min, lon_max, lat_min, lat_max],
    npoints=300,
    slipVec=slipVec,
    method="cutde",
)
```

`slipVec` 列顺序为：

```text
strike_slip, dip_slip, tensile
```

对矩形元对象，`slipVec` 形状应为 `(n_patch, 3)`。接口内部会把每个矩形 patch 的三分量 slip 复制到拆分后的两个三角形上。

## 输出文件

HDF5 适合大规模网格：

```python
fault.compute_surface_displacement(
    box=[lon_min, lon_max, lat_min, lat_max],
    npoints=600,
    method="cutde",
    output_file="dense_surface_disp.h5",
    output_format="h5",
)
```

常见 HDF5 内容包括：

```text
coordinates/longitude
coordinates/latitude
coordinates/depth
displacement/east
displacement/north
displacement/up
source_slip/strike_slip
source_slip/dip_slip
source_slip/tensile
```

TXT 适合小规模快速检查：

```python
fault.compute_surface_displacement(
    box=[lon_min, lon_max, lat_min, lat_max],
    npoints=100,
    output_file="dense_surface_disp.txt",
    output_format="txt",
)
```

文本列含义为：

```text
longitude latitude depth disp_east disp_north disp_up
```

也可以只接收返回值后自行保存：

```python
df = pd.DataFrame({
    "lon": obs_pts[:, 0],
    "lat": obs_pts[:, 1],
    "east": disp[:, 0],
    "north": disp[:, 1],
    "up": disp[:, 2],
})
df.to_csv("dense_surface_disp.xyz", sep=" ", index=False, float_format="%.8e")
```

## 加速和内存控制

推荐默认使用：

```python
method="cutde"
```

密集网格可能很大，内存需求大致随以下规模增长：

```text
观测点数 × 三角形数
```

接口内部支持批处理和内存控制。常用参数：

| 参数 | 作用 |
| --- | --- |
| `target_mem_gb` | 目标内存上限；默认自动估计 |
| `max_obs_batch` | 每批最多观测点数 |
| `max_tri_batch` | 每批最多三角形数 |
| `min_batch_count` | 最少分批数，避免单批过大 |

大模型可这样限制内存：

```python
obs_pts, disp = fault.compute_surface_displacement(
    box=[lon_min, lon_max, lat_min, lat_max],
    npoints=600,
    method="cutde",
    target_mem_gb=4.0,
    max_obs_batch=50000,
    output_file="dense_surface_disp.h5",
)
```

实用建议：

- 先用 `npoints=100` 或 `200` 检查范围、单位和正负号。
- 确认结果合理后，再增大到最终分辨率。
- 大结果优先保存为 HDF5，不建议先写超大 TXT。
- 如果只用于画图，优先保存规则网格，再按需要转换成 GMT/NetCDF。

## 与 geodata 正演的区别

`buildGFs()` + `buildsynth()` 面向已有 geodata 对象，例如 GPS 点、InSAR LOS 点或降采样点，适合生成与观测数据对应的 synthetic。

`compute_surface_displacement()` 面向任意密集地表点，直接生成三维位移场，适合输出：

- `lon lat east north up` 密集点表
- 规则网格三分量形变场
- 剖面三分量形变
- 大范围可视化背景场

如果目标是密集三维地表形变场，应优先使用 `compute_surface_displacement()`。
