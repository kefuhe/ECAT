# 观测参考改正与无重采样网格导出

本页说明降采样前的确定性参考改正、少见的用户指定区域整周修正，以及供
PyGMT/GMT、xarray 和 GIS 工具使用的全分辨率观测网格导出。第一次使用可直接复制
[InSAR 降采样 workflow](../workflows/02_insar_downsampling.md#5-可选参考改正与全分辨率导出)
中的圆形参考区示例。

## 阅读路径

| 只需要一个结果 | 从这里开始 |
| --- | --- |
| 参考区归零 | [最常用配置：圆形参考区归零](#最常用配置圆形参考区归零) |
| 已确认局部整周跳变 | [高级：给指定不连通区域修正整周](#高级给指定不连通区域修正整周) |
| 估计或指定长波 ramp | [一阶平面](#一阶平面) / [固定系数](#固定系数) |
| 导出 HDF5/NetCDF 原始与改正网格 | [全分辨率标准导出](#全分辨率标准导出) |
| 用 PyGMT 保留真实网格拓扑绘图 | [网格拓扑和 PyGMT](#网格拓扑和-pygmt) |

## 适用范围

该功能解决两类常见问题：

- SAR/InSAR 全图带有近似常数 datum offset；
- SAR 或 optical offset 存在可由一阶平面描述的稳定长波趋势。

另外提供一个高级的 `phase_cycle_correction`，用于用户已经确认某个不连通
解缠分量需要移除整数个 \(2\pi\) 的情况。ECAT 不会从单幅影像自动猜测整周数。
它不是通用解缠修复、大气改正或高阶轨道模型。若一阶平面仍不能解释数据，应检查
处理链、解缠、大气和轨道误差，而不是提高多项式阶数或强行指定周数。

执行层级固定为：

```text
reader 单位/符号转换
  -> data_filters
  -> phase_cycle_correction
  -> observation_correction
  -> optional full-grid export
  -> processing_region
  -> covariance / downsampling
```

它与以下设置不同：

| 设置 | 作用 |
| --- | --- |
| `phase_cycle_correction` | 只在用户指定区域移除明确的整数个 \(2\pi\) |
| `observation_correction` | 降采样前直接改正观测 |
| `covar.rampEst` | 仅服务经验协方差拟合 |
| 反演 `geodata.polys` | 在反演中联合估计 nuisance 参数 |
| `downsample.guide_grid` | 只控制采样网格，不负责归零或去 ramp |

## 高级：给指定不连通区域修正整周

这个块不会出现在默认生成模板中。只有已经通过重叠影像、稳定区、GNSS 或其他可靠
证据确定周数后，才手工加入配置。它是顶层块，缩进与 `sar_config`、
`observation_correction` 相同；推荐放在两者之间，不要放进 `sar_config` 内。

最短可复制配置使用圆心和半径指定目标区域：

```yaml
phase_cycle_correction:
  enabled: true
  report: true
  report_file: auto
  corrections:
    - name: western_unwrap_component
      cycles_to_remove: 1
      selector:
        kind: circle
        center: [-68.30, 10.50]  # [lon, lat]
        radius_km: 15.0
```

直接复制到已有 `unwrapped_phase` 模板后，只需修改 `name`、`cycles_to_remove`、
`center` 和 `radius_km`。`corrections` 是列表；需要修正多个互不重叠的分量时，
在列表中继续增加结构相同的 `- name: ...` 条目。

`cycles_to_remove: n` 的唯一含义是：

\[
\phi_{\rm corrected}=\phi_{\rm observed}-2\pi n
\]

当前 ECAT 使用 toward-sensor LOS，因此运行时在已经转换为米的观测上执行完全等价的：

\[
d_{\rm corrected}=d_{\rm observed}+n\frac{\lambda}{2}
\]

波长只从 reader 最终解析的 `observation_spec` 获取，不在该块重复配置。该功能只接受
`data_type: sar` 和 `sar_config.mode: unwrapped_phase`；不能用于已经失去相位周期
语义的 `los_displacement`、range/azimuth offset 或 optical 数据。

若目标区域更适合其他几何，只替换上面 correction 内的 `selector`，其余字段不变。

经纬度 box：

```yaml
selector:
  kind: box
  bounds: [-68.6, -68.2, 10.2, 10.7]
  # [min_lon, max_lon, min_lat, max_lat]
```

两列 `lon lat` 的 polygon 文件：

```yaml
selector:
  kind: polygon_file
  polygon_file: western_component.lonlat
```

内联 polygon：

```yaml
selector:
  kind: polygon
  polygon:
    - [-68.60, 10.20]
    - [-68.20, 10.20]
    - [-68.20, 10.70]
    - [-68.60, 10.70]
```

第一版只支持 lon/lat selector。每个 correction 必须有唯一 `name`，
`cycles_to_remove` 必须是整数；多个目标区域不能重叠，空区域或无法映射到当前
分析像元时直接报错。

默认报告为：

```text
<outName>_phase_cycle_correction.yml
```

它记录解析波长、每个区域的 selector、像元数、移除的相位、施加的 LOS 增量和改正
前后统计。全局 offset/plane 会在整周修正之后估计，因此两种改正可以同时使用，但
职责不会混合。

区域 delta 会写入区域内全部有限源像元，以保证全分辨率 NetCDF/HDF5 输出连续；
映射回 CSI 计算数组时只取 `analysis_valid_mask` 对应的活动索引。两层使用同一个
原网格索引映射，不会用降采样后的顺序反推源像元。

## 最常用配置：圆形参考区归零

```yaml
observation_correction:
  enabled: true
  model: offset                 # offset | plane
  coefficient_mode: estimate   # estimate | fixed
  report: true
  report_file: auto
  fit:
    coord_type: lonlat
    regions:
      - kind: circle
        center: [-68.30, 9.40]  # [lon, lat]
        radius_km: 20.0
    exclude_regions: []
  fixed_coefficients:
```

参考圆按 ECAT 当前局部投影中的 km 距离选择，不使用经纬度差近似半径。offset
固定使用参考区中位数：

\[
b_0=\operatorname{median}(d_i),\qquad
d_{\mathrm{corrected}}=d-b_0
\]

原观测不会被覆盖。运行时保留原观测、改正面和改正后观测；协方差和降采样使用
改正后观测。

## box、多个区域和排除区

box：

```yaml
fit:
  coord_type: lonlat
  regions:
    - kind: box
      bounds: [-68.6, -68.2, 9.1, 9.5]
      # [min_lon, max_lon, min_lat, max_lat]
  exclude_regions: []
```

`regions` 中多个区域取并集，`exclude_regions` 再从并集中扣除：

```yaml
fit:
  coord_type: lonlat
  regions:
    - kind: circle
      center: [-68.30, 9.40]
      radius_km: 20.0
    - kind: box
      bounds: [-67.2, -66.9, 9.0, 9.3]
  exclude_regions:
    - kind: polygon
      polygon:
        - [-67.80, 9.20]
        - [-67.60, 9.20]
        - [-67.60, 9.35]
        - [-67.80, 9.35]
```

也可使用：

```yaml
- kind: polygon_file
  polygon_file: reference_region.txt
```

polygon 文件为两列 `lon lat`，相对路径按配置文件目录解析。首版只支持 lon/lat
改正区域，不提供同义 shorthand 或 xy 区域。

## 一阶平面

确认 offset 后仍存在稳定长波趋势时：

```yaml
observation_correction:
  enabled: true
  model: plane
  coefficient_mode: estimate
  fit:
    coord_type: lonlat
    regions:
      - kind: box
        bounds: [-69.1, -65.6, 8.9, 10.8]
    exclude_regions:
      - kind: circle
        center: [-67.8, 10.2]
        radius_km: 50.0
```

计算公式为：

\[
c(x,y)=b_0+b_E(x-x_0)+b_N(y-y_0)
\]

\[
d_{\mathrm{corrected}}(x,y)=d(x,y)-c(x,y)
\]

其中：

- \(b_0\) 与观测单位相同；
- \(b_E,b_N\) 的单位是“观测单位/km”；
- \(x_0,y_0\) 是拟合样本中心，用于降低数值相关性；
- `plane` 已包含 offset，不应再叠加一个 offset operation。

plane 使用固定的稳健线性估计流程。报告包含系数、参考点数、rank、条件数、
迭代次数和改正前后统计。空间退化或秩不足会报错，不会静默回退到 offset。

## 固定系数

固定 offset：

```yaml
observation_correction:
  enabled: true
  model: offset
  coefficient_mode: fixed
  fit:
    coord_type: lonlat
    regions: []
    exclude_regions: []
  fixed_coefficients:
    observation:
      offset: 0.55
      east_gradient: 0.0
      north_gradient: 0.0
```

固定 plane 必须给出系数原点，避免同一梯度因截距参考位置不同而产生歧义：

```yaml
observation_correction:
  enabled: true
  model: plane
  coefficient_mode: fixed
  fit:
    coord_type: lonlat
    regions: []
    exclude_regions: []
  fixed_coefficients:
    origin: [-68.30, 9.40]  # [lon, lat]，offset 在此位置定义
    observation:
      offset: 0.55
      east_gradient: 0.00012
      north_gradient: 0.00037
```

无论估计还是固定系数，都只使用：

```text
corrected = observation - correction_surface
```

不存在方向相反的 `add_offset` 接口。

## optical east/north

optical offset 共用参考区，但 east 和 north 分量分别估计系数。固定系数写成：

```yaml
fixed_coefficients:
  origin: [100.0, 20.0]  # 仅 plane 需要
  east:
    offset: 0.0
    east_gradient: 0.0
    north_gradient: 0.0
  north:
    offset: 0.0
    east_gradient: 0.0
    north_gradient: 0.0
```

任一分量的 NaN 只排除该分量的拟合样本，east 系数不会用于 north。

## 改正报告

默认写出：

```text
<outName>_observation_correction.yml
```

报告包含：

- 使用 `offset` 还是 `plane`；
- 系数来自 `estimate` 还是 `fixed`；
- 每个分量的 offset、东西/南北梯度；
- 参考区 count、mean、median、MAD、std 和范围；
- 改正前后统计；
- plane 的原点、rank、条件数和迭代次数。

参考区为空、固定 plane 没有原点、region 字段混用或数据与原网格索引无法对齐时
会直接报错。

## 全分辨率标准导出

最小配置：

```yaml
export:
  observation_grid:
    enabled: true
    format: netcdf
    file: auto
    geotiff_sidecar: auto
    verify: true
    report: true
    report_file: auto
```

配置好后直接运行：

```bash
ecat-downsample -f downsample_phase.yml
```

该命令可以只生成参考改正报告和标准观测网格；不要求同时使用 `-c` 或 `-d`。
加上 `-c`/`-d` 时，协方差和降采样使用同一份改正后观测。

当前规范格式只有 `netcdf`，避免出现多个含义相近的 format 选择。底层优先使用项目
可选依赖中声明的 `h5netcdf`，不可用时再使用 `netCDF4`；两者都写
HDF5-backed NetCDF。文件名可以显式使用
`.nc`、`.h5` 或 `.hdf5`。例如：

```yaml
export:
  observation_grid:
    enabled: true
    format: netcdf
    file: S1_pair_observation.h5
```

如果环境提示 `numpy.ndarray size changed ... binary incompatibility`，它表示某个
已编译 I/O 扩展（常见为旧 `netCDF4`）与当前 NumPy ABI 不匹配，不表示观测值已经
改变。ECAT 的标准网格路径会优先使用 `h5netcdf`；在其他代码中显式使用
`netCDF4` 时，应在同一包管理渠道中重装与当前 NumPy 匹配的版本。

`.h5/.hdf5` 要求 HDF5-backed engine；不可用时会明确报错，不会写一个只有扩展名
像 HDF5 的文件。`file: auto`
写：

```text
<outName>_observation.nc
```

导出不插值、不重投影、不改变分辨率。写完会重新读取并逐变量验证 shape、数值、
NaN mask 和经纬度坐标。

如果源数据具有可靠的 affine geotransform 和 CRS，
`geotiff_sidecar: auto` 还会按变量写同网格 GeoTIFF。GAMMA 二进制或任意二维
曲线经纬网格不会被强制伪装成 GeoTIFF。显式设置 `true` 但输入缺少可靠 affine
时会报错。默认 NetCDF 名称对应的 sidecar 使用
`<outName>_<variable>.tif`，例如 `<outName>_observation.tif` 和
`<outName>_corrected_observation.tif`；实际路径同时记录在导出报告中。

## NetCDF 变量

SAR：

```text
observation
phase_cycle_delta          # 启用区域整周修正时，单位 m，加到 observation
correction_surface          # 启用改正时
corrected_observation       # 启用改正时
projection_east             # reader 可提供完整 ENU projection 时
projection_north
projection_up
source_valid_mask
analysis_valid_mask
```

optical：

```text
east
north
east_correction_surface     # 启用改正时
north_correction_surface
corrected_east
corrected_north
source_valid_mask
analysis_valid_mask
```

`observation` 表示 reader 已完成单位、phase 和符号约定转换后的分析观测，不是原始
文件字节。SAR 当前单位为 m；optical 由 `factor_to_m` 转为 m。
`projection_east/north/up` 是满足
`scalar_observation = ENU_displacement dot projection` 的三个投影系数，不是三个
方向的位移观测。

SAR 数据集 attributes 明确区分：

- `source_observation_type`：源 reader mode 的物理类型；
- `source_value_convention`：源标量进入 reader 前的正号/编码；
- `stored_observation_quantity`：标准文件实际保存的规范物理量；
- `observation_type`：为已有读取者保留的 source type 字段；
- `wavelength_m`：源产品可用波长。

变量自身的 `positive_convention` 明确写为 `positive toward satellite` 或
`positive along heading`。原始 phase 文件仍应作为权威源数据保留。

## 网格拓扑和 PyGMT

导出报告：

```text
<outName>_observation_grid.yml
```

其中会标明 topology：

| topology | 坐标 | 推荐绘图 |
| --- | --- | --- |
| `geographic_rectilinear` | 精确一维 longitude/latitude | PyGMT/GMT 可直接使用 NetCDF variable |
| `projected_rectilinear` | 一维原生 x/y + 二维 lon/lat | 保持原投影绘制；GeoTIFF sidecar 最直接 |
| `affine_rotated` | 完整 affine + 二维 projected/geographic 坐标 | 使用保持 affine 的 GeoTIFF |
| `geographic_curvilinear` | 二维 longitude/latitude | 用 coordinate-aware `pcolormesh` 核验 |

规则 geographic 网格：

```python
import pygmt

fig = pygmt.Figure()
fig.grdimage(
    grid="S1_T012A_observation.nc?corrected_observation",
    projection="M12c",
    frame=True,
    cmap="vik",
)
fig.colorbar(frame='af+l"LOS displacement (m)"')
fig.show()
```

没有启用改正时使用 `?observation`。

二维曲线网格可以在 CF-NetCDF 中无损保存，但不能假定所有 GMT
`grdimage` 路径都能把二维辅助经纬度当作规则网格。精确核验可用：

```python
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset("observation.nc")
plt.pcolormesh(
    ds.longitude,
    ds.latitude,
    ds.corrected_observation,
    shading="nearest",
)
plt.show()
```

若必须把曲线网格放到另一个 PyGMT 地理投影，应另建明确命名的派生绘图网格。
该过程会重采样，不能覆盖标准观测文件，也不能作为反演或降采样输入。当前
ECAT 不提供隐式通用重采样入口。

## 与原 reader 的对应

| reader | 常见 topology | 导出行为 |
| --- | --- | --- |
| GAMMA `.rsc` | `geographic_rectilinear` | CF-NetCDF 可直接用于 GMT grid |
| GMTSAR/direct projection | rectilinear 或 curvilinear | 按实际 lon/lat shape 判定 |
| GAMMA TIFF / HyP3 TIFF | 多为 `projected_rectilinear` | CF-NetCDF + 同 affine GeoTIFF |
| optical TIFF | 多为 `projected_rectilinear` | east/north 同索引导出 |

TIFF 像元中心使用完整 GDAL 六参数计算，包含半像元偏移和 rotation/skew。二维
经纬度不会取第一行/列或行列均值代替。

## 相关页面

- [InSAR 降采样 workflow](../workflows/02_insar_downsampling.md)
- [降采样超级入口参考](downsampling_app.md)
- [SAR Reader 参考](sar_reader.md)
- [反演数据改正](data_corrections.md)
