# InSAR 降采样

本流程把密集 SAR/offset 栅格转换成 CSI `.txt/.rsp/.cov` 输入。推荐顺序是：

1. 生成与产品匹配的短配置。
2. 运行 quick-look，核对文件、单位、正负号和 projection。
3. 可选：用稳定参考区归零/去一阶平面，并导出全分辨率标准网格。
4. 遮蔽主要形变区后估计经验协方差。
5. 正式降采样并检查输出。

SAR 标量与 projection 的合同见
[SAR 投影与方向约定](../concepts/sar_projection_conventions.md)，所有字段见
[Downsampling App](../reference/downsampling_app.md)。需要绕过 YAML 手写 GAMMA、
HyP3、GMTSAR 或 optical reader 时，直接查
[SAR 与光学观测读入脚本](../reference/observation_data_readers.md)。

## 1. 选择 reader 和 mode

| 输入 | reader | 常用 mode |
| --- | --- | --- |
| GAMMA `.phs/.rsc/.azi/.inc` | `gamma` | 四种 mode 均支持 |
| GAMMA GeoTIFF + angle rasters | `gamma_tiff` | 四种 mode 均支持 |
| HyP3 GeoTIFF | `hyp3` | `unwrapped_phase`, `los_displacement` |
| GMTSAR-style value + ENU projection | `gmtsar` | 四种 mode 均支持 |

四种 mode 是：

```text
unwrapped_phase
los_displacement
range_offset
azimuth_offset
```

已经得到 CSI `.txt/.rsp/.cov` 时，不再经过原始 SAR reader，直接在反演脚本中
调用 `read_from_varres(...)`。

## 2. 生成配置

以下命令可以直接复制。

默认右视 GAMMA 解缠相位：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase -o downsample_phase.yml
```

默认右视 GAMMA LOS、range offset 和 azimuth offset：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode los_displacement -o downsample_los.yml
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode range_offset -o downsample_range.yml
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode azimuth_offset -o downsample_azimuth.yml
```

左视、由 GAMMA 导出的 NISAR 解缠相位：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase --sar-look-side left -o downsample_nisar_left.yml
```

如果二进制明确为大端：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase --sar-byte-order big -o downsample_phase_big_endian.yml
```

其他 reader：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma_tiff --sar-mode unwrapped_phase -o downsample_gamma_tiff.yml
ecat-generate-downsample --mode sar --sar-reader gmtsar --sar-mode range_offset -o downsample_gmtsar_range.yml
```

### 按降采样方法生成短模板

以下命令使用同一种 GAMMA 解缠相位输入，只改变降采样方法。`std`
是默认值；这里仍显式写出，便于脚本和配置文件名自说明。

```bash
# 标准差四叉树：常用默认方法
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase --downsample-method std -o downsample_std.yml

# 数据特征四叉树：按曲率或梯度细化
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase --downsample-method data -o downsample_data.yml

# 三角形断层模型引导
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase --downsample-method trirb -o downsample_trirb.yml

# 复用已有 CSI .rsp 采样单元
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase --downsample-method from_rsp -o downsample_from_rsp.yml
```

默认的 `--template minimal` 只写出所选方法的 `*_config`，适合直接编辑。
`--template full` 会列出所有方法配置供进阶查阅，但运行时仍只有
`downsample.method` 指定的方法生效。

生成 `trirb` 模板时，两个三角断层示例会预先写好 `use_for: [trirb]`，但仍保持
`enabled: false`。选择一种模型来源，填写真实文件和几何参数后再将它启用即可；生成
`std`、`data` 或 `from_rsp` 模板时，示例模型保持 `use_for: []`，不会被隐式加入计算。

`trirb` 还要求配置一个启用的三角形断层模型；只有 `fault_traces`
不能满足算法输入。已有 CSI GMT 三角网格时，可直接复制下面这段：

```yaml
fault_models:
  - enabled: true
    id: fault_mesh
    type: csi_gmt              # 模型输入来源：读取 CSI GMT
    geometry: triangular
    file: fault_mesh.gmt
    readpatchindex: true       # GMT 段头含 3 个拓扑索引时设 true
    donotreadslip: true        # trirb 只需几何；忽略文件中的滑动量
    gmtslip: true              # 三角网格专用；段头含 -Z... 时设 true
    use_for: [trirb]
    plot:
      stages: [decim]
      mode: edges
```

三角 GMT 段头没有 `-Z...` 时改为 `gmtslip: false`；没有 3 个拓扑索引时改为
`readpatchindex: false`。矩形 CSI GMT 不支持 `trirb`，只能令 `use_for: []` 用于
检查图叠加，并省略 `gmtslip`。若没有现成网格，可改用模板中的
`type: generated_from_trace` 从迹线生成三角模型。`fault_models` 是列表，可以重复或
混合两种来源；所有启用且写有 `use_for: [trirb]` 的三角模型会共同参与一次分辨率计算。

`from_rsp` 需要在 `from_rsp_config.rsp_file` 中指定已有的 CSI `.rsp`
文件。各方法的参数选择和运行前检查见
[Downsampling App 参考](../reference/downsampling_app.md)。

## 3. 编辑最小配置

### GAMMA prefix

```yaml
general:
  origin: auto

sar_config:
  outName: S1_T012A
  reader: gamma
  mode: unwrapped_phase
  acquisition_look_side: right
  directory: InSAR/raw
  files:
    prefix: geo_20250101_20250113
  read:
    downsample: 1
    downsample_for_covar: 1
    zero2nan: true
    wavelength:
    factor_to_m: 1.0
    byte_order: native
```

`native` 是历史默认，不改变已有案例。只有产品文档明确说明，或数值诊断确认
字节序不匹配时，才改为 `little` 或 `big`。

### GAMMA offset 显式文件

```yaml
sar_config:
  outName: S1_T012A_range
  reader: gamma
  mode: range_offset
  acquisition_look_side: right
  directory: InSAR/raw
  files:
    value: roff_20250101_20250113.phs
    metadata: roff_20250101_20250113.phs.rsc
    geometry:
      azimuth: off_20250101_20250113.azi
      incidence: off_20250101_20250113.inc
```

同一目录中有多个候选 value 时，offset 应优先显式写文件，避免 prefix 匹配歧义。

### 左视 GAMMA/NISAR

```yaml
sar_config:
  reader: gamma
  mode: unwrapped_phase
  acquisition_look_side: left
```

这只改变采集几何，不改变 `.azi` 默认的 `sensor_to_ground` 语义，也不改变相位或
offset 原始正号。如果 angle 文件本身代表 heading 或 `ground_to_sensor`，再增加：

```yaml
sar_config:
  geometry_convention:
    azimuth_angle_role: heading
```

### GMTSAR direct projection

```yaml
sar_config:
  outName: T33D_range
  reader: gmtsar
  mode: range_offset
  directory: InSAR/gmtsar
  files:
    value: range.grd
    projection:
      east: enu/e.grd
      north: enu/n.grd
      up: enu/u.grd
```

不要在 GMTSAR 块中添加 azimuth/incidence angle 字段。产品 projection 方向与
内置规则不同时，使用 `projection_convention`，详见
[SAR Reader 参考](../reference/sar_reader.md#gmtsar-direct-projection)。

## 4. 四种运行模式

生成模板中的 `covar.do_covar` 和 `downsample.enabled` 默认都是 `false`。在这个默认
前提下，最常用的四种运行方式是：

| 模式 | 命令 | 主要用途 |
| --- | --- | --- |
| 无阶段选项 | `ecat-downsample -f downsample_phase.yml` | 读入数据，并执行配置中启用的整周修正、参考改正、标准网格导出和 Google Earth 全分辨率导出 |
| `-s` | `ecat-downsample -f downsample_phase.yml -s` | 原始/最终改正观测 quick-look；强制关闭本次协方差和降采样 |
| `-c` | `ecat-downsample -f downsample_phase.yml -c` | 估计经验协方差，不自动执行正式降采样 |
| `-d` | `ecat-downsample -f downsample_phase.yml -d` | 按所选方法正式降采样并写 `.txt/.rsp/.cov` |

需要在观测上调整断层迹线时，使用独立的可选检查入口：

```bash
ecat-downsample -f downsample_phase.yml --edit-trace
```

它复用同一次 reader 和观测改正结果，但本次不执行协方差或降采样，也不改写
`fault_traces` 或 YAML。完整步骤见
[交互调整断层迹线](02c_interactive_trace_editing.md)。

命令行 `-c/-d` 是对配置开关的临时启用：如果 YAML 中已经把另一个阶段设为 `true`，
它仍会同时运行；`-c -d` 也可以显式组合。`-s` 是例外，它优先作为检查模式并关闭
协方差和降采样。标准 NC/H5 导出只要 `observation_grid.enabled: true` 就会随本次读入
写出；Google Earth 联动只在有效阶段全部关闭时执行，因此不随 `-s/-c/-d` 重复写 KMZ。

### 4.1 Quick-look

配置文件方式：

```bash
ecat-downsample -f downsample_phase.yml -s
```

只检查一个 GAMMA prefix 时，可不写 YAML：

```bash
ecat-downsample -s --sar-prefix geo_20250101_20250113 --sar-mode unwrapped_phase
ecat-downsample -s --sar-prefix nisar_pair --sar-mode unwrapped_phase --sar-look-side left
```

这个快捷入口固定使用 GAMMA reader，只用于 `-s` 或 `--edit-trace`，不能执行
`-c/-d`。可用 `--sar-byte-order native|little|big` 指定二进制字节序。

检查：

- 经纬度范围和栅格方向；
- `Final observation spec` 的 mode/原始标量语义；
- `Angle geometry` 的 angle role 与左右视；
- projection East/North/Up 均值是否符合轨道方向；
- 形变量级、正负号和异常尾部；
- incidence 是否在合理范围，是否出现字节序警告。

如果彩色图受少量极值影响，或希望更清楚地辨认闭合形变瓣和梯度，可在配置中临时启用
raw 等值线：

```yaml
check_plots:
  raw:
    contours:
      enabled: true
      levels: auto
```

等值线只叠加到 `-s` 的 structured 二维 raw quick-look，不平滑、不插值，也不影响随后
`-c/-d` 的数值。密集或破碎线条不一定是离散形变，也可能来自噪声、无效区或解缠问题；
完整 levels、单位和样式语义见[降采样应用参考](../reference/downsampling_app.md#raw-等值线诊断层)。

## 5. 可选：参考改正与全分辨率导出

如果远场整体不在零附近，先用一个明确的稳定参考区估计常数 offset。最方便的是
圆心加半径：

```yaml
observation_correction:
  enabled: true
  model: offset
  coefficient_mode: estimate
  fit:
    coord_type: lonlat
    regions:
      - kind: circle
        center: [-68.30, 9.40]  # [lon, lat]
        radius_km: 20.0
    exclude_regions: []
```

也可以换成 box：

```yaml
    regions:
      - kind: box
        bounds: [-68.6, -68.2, 9.1, 9.5]
        # [min_lon, max_lon, min_lat, max_lat]
```

先使用 `offset`。只有确认归零后仍存在稳定长波趋势，才改用 `plane`。plane 估计默认
使用全部有效观测，不需要手工填写截距、梯度或覆盖全图的 region；主要形变区可用 box
排除。可直接复制：

```yaml
observation_correction:
  enabled: true
  model: plane
  coefficient_mode: estimate
  fit:
    coord_type: lonlat
    regions: []                 # 全部有效观测；非空时只拟合指定区域
    exclude_regions:
      - kind: box
        bounds: [-68.2, -67.2, 9.7, 10.7]
        # 排除主要形变或噪声区域
  fixed_coefficients: null
```

最终拟合样本应在东西、南北两个方向都有足够展布；排除过多，或把 plane 限制在过窄、
近似共线的显式 regions 中，都无法稳定确定两个梯度，并会明确报秩不足。若使用
`coefficient_mode: fixed`，则必须清空
`fit.regions/exclude_regions`，并提供系数原点以及每个观测分量的 `offset`、
`east_gradient`、`north_gradient`。完整 SAR/optical 写法见下方 reference。

`-c` 只启用经验协方差阶段，不会自动启用参考改正，也不会把 `covar.mask_out` 当成
观测改正的 regions/exclusions。只想估计协方差、暂不进行观测改正时，应保持
`observation_correction.enabled: false`；`covar.rampEst: true` 只处理协方差拟合中的 ramp，
不会改写随后降采样使用的观测。

改正发生在 `data_filters` 后、`processing_region` 前；原观测保留，协方差和降采样使用
改正后观测。运行会写 `<outName>_observation_correction.yml`。

少数情况下，不连通解缠分量存在已确认的 \(2n\pi\) jump。不要用全局 offset/plane
拟合这种离散阶跃；在可靠确定整数周数后，使用高级顶层
`phase_cycle_correction` 给目标区域明确设置 `cycles_to_remove`。该块不在默认
模板中，完整可复制配置和符号公式见
[指定区域整周修正](../reference/observation_correction_export.md#高级给指定不连通区域修正整周)。

需要绘制全分辨率原观测和改正后观测时：

```yaml
export:
  observation_grid:
    enabled: true
    format: netcdf
    file: auto
    geotiff_sidecar: auto
    verify: true
```

它写 `<outName>_observation.nc`，不插值、不重投影。规则经纬网格可以直接给
PyGMT/GMT；二维曲线经纬网格会原样保存在 NetCDF 中，不会被压成近似的一维轴。
若后续需要用 xarray/HDF5 工具读取，可把 `file` 显式设置为
`<outName>_observation.h5`；变量和坐标结构保持相同。
详细变量、fixed plane 和 PyGMT 路径见
[观测参考改正与无重采样网格导出](../reference/observation_correction_export.md)。

如果想从同一 reader 配置生成 Google Earth 全分辨率显示副本，在同一个 `export`
下加入并列块，不需要第二份配置：

```yaml
export:
  observation_grid:
    enabled: false        # 只有需要标准 .nc/.h5 时才打开
  google_earth:
    enabled: true
    file: auto            # <outName>_google_earth.kmz
    mask: source_valid    # 全部有效源像元；analysis_valid 仅显示分析子集
    style:
      vmin: null          # 显示色标范围，不是经纬度范围
      vmax: null          # 两者同时留空时自动计算
      symmetry: true      # 自动范围关于 0 对称；显式 vmin/vmax 优先
```

KMZ 自动读取当前 reader 和最终改正值，只导出全分辨率 raster，不自动加入
`.txt/.rsp`。规则、等间距的经纬网格可精确写出；其他拓扑明确报错，绝不插值。
不加 `-s/-c/-d`、只运行 `ecat-downsample -f downsample_phase.yml` 时执行该导出；
quick-look、协方差和降采样阶段不重复写 KMZ。完整高级字段见
[Google Earth Export Reference](../reference/google_earth_export.md#downsample-integration)。

只执行参考改正/导出而不估计协方差或降采样时，保持
`covar.do_covar: false`、`downsample.enabled: false` 并直接运行：

```bash
ecat-downsample -f downsample_phase.yml
```

预计输出：

```text
<outName>_observation.nc
<outName>_phase_cycle_correction.yml  # 启用区域整周修正时
<outName>_observation_correction.yml   # 启用参考改正时
<outName>_observation_grid.yml
<outName>_run_metadata.yml
```

具有可靠 affine/CRS 的 TIFF 输入还可能生成
`<outName>_<variable>.tif`；纯 GAMMA 二进制通常只生成标准 NetCDF，
不会因缺少 GeoTIFF sidecar 而视为失败。

若同一次运行还需要 quick-look、协方差或降采样，照常加 `-s`、`-c` 或 `-d`；
这些选项不改变标准 NC/H5 导出的网格分辨率，但不会触发 Google Earth 联动。

这两个块均默认关闭；不需要改正或导出时，现有读取和降采样数值路径不变。

## 6. 协方差和正式降采样

初次处理建议继续使用同一个配置文件，依次完成 `-c` 和 `-d`。命令行选项会临时启用
对应阶段，因此不需要把模板中的 `covar.do_covar: false` 或
`downsample.enabled: false` 改成 `true`。

### 6.1 `-c`：先设置主形变区掩膜

```yaml
covar:
  do_covar: false
  mask_out: [100.5, 101.75, 37.35, 38.1]  # [minlon, maxlon, minlat, maxlat]
  function: exp
  frac: 0.002
  every: 2.0
  distmax: 100.0
  rampEst: true
```

第一次通常只需要修改 `mask_out`。它应遮住主要震源形变，同时保留足够背景点估计
空间相关噪声；它不会删除最终降采样点，也不能替代 `data_filters` 或
`processing_region`。其余参数可先保留模板值。

西半球 box 可以直接使用负经度；区域选择和协方差掩膜会与 CSI 的等价 `0–360°`
经度自动匹配。跨日界线写法和运行诊断见
[经度约定与区域配置](../reference/longitude_regions.md)。

```bash
ecat-downsample -f downsample_phase.yml -c
```

确认生成 `Covariance_estimator.cov`，且 `mask_out` 没有覆盖全部背景。光学 offset
会分别生成 East/North 协方差估计器。正式处理区域仍使用顶层
`processing_region`。

### 6.2 `-d`：首次只关注四个 `std` 参数

```yaml
downsample:
  enabled: false
  method: std
  std_config:
    startingsize: 5.0
    minimumsize: 0.25
    min_valid_fraction: 0.1
    split_std_threshold: 0.005
```

| 参数 | 首次理解 | 常用调整方向 |
| --- | --- | --- |
| `startingsize` | 初始方块大小，CSI 投影单位通常为 km | 增大可减少初始块数量；它不直接规定最终分辨率 |
| `minimumsize` | 最小允许块大小 | 减小允许近场继续细分；过小可能造成过密采样 |
| `min_valid_fraction` | 块内最小有效像素比例 | 初次通常保留 `0.1`；主要用于含 NaN/空洞的区域 |
| `split_std_threshold` | 是否继续分裂的主要阈值 | 减小会更密，增大会更稀；单位与进入降采样的观测值一致 |

对于 `unwrapped_phase`，相位会先按波长转换为目标 LOS 位移再进入降采样，因此
`split_std_threshold` 通常按米理解，而不是按原始弧度设置。

建议先调 `split_std_threshold`；只有近场达到当前最小块后仍过粗，再减小
`minimumsize`。`startingsize` 更多影响初始划分和运行效率，通常后调；
没有明显空洞问题时先不改 `min_valid_fraction`。

```bash
ecat-downsample -f downsample_phase.yml -d
```

检查 `<outName>_decim.png`、输出点数、近场和远场密度，以及 `.cov` 维度是否等于
观测数。仅修改 `std_config` 时，可以复用已有协方差估计器，只重新运行 `-d`。

### 6.3 可选：用引导图稳定 quadtree 网格

普通 `std` 因高频噪声、局部闪烁或 offset 粗差而过度细分时，可以启用
`guide_grid`。它不是新的 `method`，而是 `std`/`data` 的网格生成辅助层：

连续、局部高频噪声先用 Gaussian：

```yaml
downsample:
  method: std
  guide_grid:
    enabled: true
    filter:
      kind: gaussian
      sigma: 1.5
      unit: km
```

若问题是 SAR/optical offset 中少量孤立错配或小斑点，可改用最短 median 配置：

```yaml
downsample:
  method: std
  guide_grid:
    enabled: true
    filter:
      kind: median          # 默认 3 x 3；可显式设置 window_size: 5
```

程序先用平滑后的引导图划分网格，再回到原始未滤波数据提取最终 cell 值。
`guide_grid` 不删除粗差，也不改变 `-c` 的协方差输入；只修改它时通常只需重跑
`-d`。`sigma` 或 median 窗口过大会平滑真实近场梯度，必须结合降采样检查图判断。
连续大气/电离层条带、解缠边界、轨道坡度或有效数据边界不是 median 的目标；
这类问题应优先做上游改正、掩膜或调整 `min_valid_fraction/focus_region`。完整字段和
optical 双分量语义见[降采样配置参考](../reference/downsampling_app.md#guide-grid)。

常用方法：

| method | 适用情况 |
| --- | --- |
| `std` | 首次跑通；不依赖断层几何 |
| `data` | 按数据振幅、梯度或曲率细化 |
| `trirb` | 已有可靠断层网格时按分辨率细化 |
| `from_rsp` | 复用已有 CSI `.rsp` 网格 |

ECAT-Cases 中以 `covarSAR-Step1.py`、`downsampleSAR-Step2*.py` 组织的既有案例仍可
按原两步走代码复现；代码参数与当前 YAML/CLI 的逐项对应见
[InSAR 降采样 Step1/Step2 调参](02a_insar_downsampling_two_step.md)。

<a id="downsampled-output-files"></a>

## 7. 输出检查

启用参考改正或标准网格导出时，先检查：

```text
<outName>_observation.nc
<outName>_observation_correction.yml   # 启用参考改正时
<outName>_observation_grid.yml
```

确认 NetCDF 同时保留改正前观测、改正面和改正后观测，报告中的公式为
`corrected = observation - correction_surface`，且 topology、shape 和实际文件
符合输入 reader。具有可靠 affine/CRS 的 TIFF 输入还可检查
`<outName>_<variable>.tif`。

正式降采样通常生成：

```text
<outName>_ifg.txt
<outName>_ifg.rsp
<outName>_ifg.cov
<outName>_decim.png
<outName>_run_metadata.yml
<outName>_downsample_report.yml
```

`<outName>_google_earth.kmz` 属于上面的无阶段选项导出，不属于正式 `-d` 输出。

确认：

- `.txt` 的数据列与 `Elos/Nlos/Ulos` 符号合理；
- `.cov` 维度等于观测数；
- 近场梯度保留、远场足够稀疏；
- run metadata 中 reader、mode、采集侧和字节序与配置一致。

<a id="read-downsampled-output"></a>
<a id="read-downsampled-output-for-inversion"></a>

## 8. 将降采样输出读回反演

`read_from_varres()` 使用共同前缀，不带扩展名。`std/data` 四叉树或矩形 `from_rsp` 输出：

```python
from csi.insar import insar

sar = insar("TrackA", lon0=lon0, lat0=lat0, verbose=False)
sar.read_from_varres(
    "Downsample/track_ifg",
    triangular=False,
    cov=True,
)
```

`trirb` 或三角 `from_rsp` 输出必须显式改为：

```python
sar.read_from_varres(
    "Downsample/track_ifg",
    triangular=True,
    cov=True,
)
```

这里不能用 `triangular=None` 让 CSI reader 自动区分三角形与矩形。若调用方不知道 `.rsp`
类型，可先用 `read_csi_varres_result(prefix, geometry="auto")` 检查，再把
`checked.geometry == "triangle"` 的结果传给 `triangular`。该检查接口不读取 `.cov`；完整
协方差仍由 `read_from_varres(..., cov=True)` 载入。没有 `.cov` 时改用 `cov=False`，然后
调用 `buildDiagCd()`；`cov=True` 后不要再用对角阵覆盖它。

完整可复制代码见[反演前读取 InSAR 与 GNSS 数据](../examples/inversion_data_loading.md)，精确
列格式和自动识别边界见[观测数据读入参考](../reference/observation_data_readers.md#csi-varres)。

下一步：

- 非线性几何反演读取降采样数据估计紧凑断层几何；
- 固定几何后，BLSE/VCE 线性反演分布式滑动。
