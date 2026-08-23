# SAR Reader 参考

SAR reader 把相位、LOS/range 位移或 azimuth offset 规范成 CSI 可用的标量观测：

```text
scalar_observation = ENU_displacement dot projection
```

进入 CSI 前，ECAT 固定使用：

- LOS/range：标量与 projection 均以地面指向传感器为正；
- azimuth：标量与 projection 均以沿平台航向为正。

精确计算公式和左右视旋转见
[SAR 投影与方向约定](../concepts/sar_projection_conventions.md#角度左右视与投影公式)。

## 阅读路径

| 当前问题 | 建议阅读顺序 |
| --- | --- |
| 需要直接复制 Python reader 脚本 | [SAR 与光学观测读入参考](observation_data_readers.md)；本页继续核对 mode、正号和 projection |
| 第一次选 reader 和观测模式 | [选择 reader 和 mode](#选择-reader-和-mode) → [配置层级](#配置层级) → [常用可复制命令](#常用可复制命令) |
| 使用 GAMMA 二进制或左视 GAMMA/NISAR 导出 | [GAMMA 二进制](#gamma-二进制) → [左视 GAMMA/NISAR](#左视-gammanisar) → [`byte_order`](#byte_order) |
| 核对形变正负号和投影方向 | [标量语义](#标量语义) → [SAR 投影与方向约定](../concepts/sar_projection_conventions.md) |
| 使用 GMTSAR ENU projection grid | [GMTSAR direct projection](#gmtsar-direct-projection) → [`print_input_summary()`](#print_input_summary) |
| 排查读入结果 | [`print_input_summary()`](#print_input_summary) → [eqtools 到 CSI 的边界](#eqtools-到-csi-的边界) → [常见错误](#常见错误) |

## 选择 reader 和 mode

| 输入 | reader | mode |
| --- | --- | --- |
| GAMMA `.phs/.phs.rsc/.azi/.inc` 解缠相位 | `gamma` | `unwrapped_phase` |
| GAMMA LOS 位移 | `gamma` | `los_displacement` |
| GAMMA range offset | `gamma` | `range_offset` |
| GAMMA azimuth offset | `gamma` | `azimuth_offset` |
| GAMMA GeoTIFF + angle rasters | `gamma_tiff` | 上述四种之一 |
| HyP3 GeoTIFF | `hyp3` | `unwrapped_phase` 或 `los_displacement` |
| GMTSAR-style value + ENU projection GRD/NetCDF | `gmtsar` | 上述四种之一 |

公开 mode 只有：

```text
unwrapped_phase
los_displacement
range_offset
azimuth_offset
```

`phase_los` 不再接受，因为它既可能被理解为相位，也可能被理解为已经转换后的
LOS 位移。简写 `los`、`range`、`az` 同样不接受。

## 配置层级

按以下顺序选择，通常只需第一层：

1. `reader + mode`：确定产品家族和标量语义。
2. `acquisition_look_side`：角度型 reader 的采集左右视。
3. `geometry_convention` 或 `projection_convention`：仅在产品偏离 reader
   内置协议时使用，二者按 reader 类型互斥。
4. Python `config` 对象：用于开发新产品协议，不属于普通 YAML 工作流。

YAML 不再提供 `preset`、通用 `convention` 或 runtime `overrides`。这些层级会把
产品语义、角度几何和文件读取混在一起，难以判断最终生效值。

## 常用可复制命令

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

显式大端 GAMMA：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase --sar-byte-order big -o downsample_phase_big_endian.yml
```

生成模板只保留当前 reader/mode 必要的短字段。`--template full` 额外写出高级
协议提示和技术性 grid 选项。

## GAMMA 二进制

### 最小配置

```yaml
sar_config:
  outName: S1_pair
  reader: gamma
  mode: unwrapped_phase
  acquisition_look_side: right
  directory: .
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

`prefix` 匹配：

```text
{prefix}*.phs
{prefix}*.phs.rsc
{prefix}*.azi
{prefix}*.inc
```

候选文件不唯一时 reader 会报错。offset 产品建议显式写文件名：

```yaml
sar_config:
  reader: gamma
  mode: range_offset
  acquisition_look_side: right
  files:
    value: roff_20250101_20250113.phs
    metadata: roff_20250101_20250113.phs.rsc
    geometry:
      azimuth: off_20250101_20250113.azi
      incidence: off_20250101_20250113.inc
```

<a id="byte_order"></a>

### `byte_order`

| 值 | 含义 |
| --- | --- |
| `native` | 主机原生 float32；历史默认，现有案例结果不变 |
| `little` | little-endian float32 |
| `big` | big-endian float32 |

同一个值同时用于 `.phs`、`.azi` 和 `.inc`。reader 会核对文件字节数是否等于
`.rsc` 尺寸乘以 4，并对异常 incidence 范围给出诊断。不要通过改变 byte order
来修正几何正负号。

### 左视 GAMMA/NISAR

这里的 `acquisition_look_side` 是“地面条带在平台航向的左侧还是右侧”，不是
LOS 向量的正方向：

```yaml
sar_config:
  reader: gamma
  mode: unwrapped_phase
  acquisition_look_side: left
  files:
    prefix: nisar_pair
```

GAMMA 默认把归一化后的 `.azi` 理解为 `sensor_to_ground`。如果实际导出文件
表达的是航向或相反 LOS 方向，才增加高级字段：

```yaml
sar_config:
  geometry_convention:
    azimuth_angle_role: heading  # heading | sensor_to_ground | ground_to_sensor
```

完整角度协议字段如下：

| 字段 | 值 | 职责 |
| --- | --- | --- |
| `azimuth_angle_role` | `heading`, `sensor_to_ground`, `ground_to_sensor` | angle 数值代表的有向水平轴 |
| `azimuth_reference` | `north`, `east` | azimuth 零点 |
| `azimuth_unit` | `degree`, `radian` | azimuth 单位 |
| `azimuth_direction` | `clockwise`, `counterclockwise` | azimuth 增长方向 |
| `incidence_reference` | `zenith`, `elevation` | incidence 参考轴 |
| `incidence_unit` | `degree`, `radian` | incidence 单位 |

`acquisition_look_side` 位于 `sar_config` 顶层，不重复放入
`geometry_convention`。当 `.azi` 已经直接给出 `sensor_to_ground` 时，LOS
projection 不需要借助左右视旋转，所以左右视不会改变
`unwrapped_phase`/`los_displacement`/`range_offset` 的 LOS 数值；它仍是必要
的采集元数据，并在由 cross-track 推导 `azimuth_offset` 的 along-heading
projection 时起作用。`print_input_summary()` 会以
`acquisition_look_side_used: true/false` 明示当前路径是否实际使用了该字段。

## 标量语义

mode 对原始值的内置解释如下：

| mode | GAMMA 原始值 | 进入 CSI 后 |
| --- | --- | --- |
| `unwrapped_phase` | radian；按 `phase * wavelength / (-4*pi)` 转换 | 地面指向传感器 |
| `los_displacement` | `toward_sensor` | 地面指向传感器 |
| `range_offset` | `away_from_sensor`，因此 value 乘 `-1` | 地面指向传感器 |
| `azimuth_offset` | `along_heading` | 沿航向 |

公式为：

$$
d_{\rm phase}=-\frac{\lambda}{4\pi}\phi
$$

$$
d_{\rm los}=
\begin{cases}
x, & \text{toward-sensor input}\\
-x, & \text{away-from-sensor input}
\end{cases}
$$

$$
d_{\rm az}=
\begin{cases}
x, & \text{along-heading input}\\
-x, & \text{opposite-heading input}
\end{cases}
$$

原始标量方向和角度几何相互独立。例如 angle 是 `ground_to_sensor`，而 range
value 是 `away_from_sensor` 时，只翻转 value，不翻转 projection。

Python 高级调用可用 `raw_value_convention` 做一次性标量覆盖：

```python
sar.read_observation(raw_value_convention="toward_sensor")
```

普通配置应优先选正确 mode，不要手工翻转数组。

### 历史 GAMMA azimuth 输出

早期 reader 对 GAMMA azimuth offset 同时使用反航向标量和反航向 projection：

```text
d_legacy = -d_canonical
p_legacy = -p_canonical
```

因此：

```text
d_legacy - predicted_ENU dot p_legacy
    = -(d_canonical - predicted_ENU dot p_canonical)
```

两种表示具有相同的平方残差和物理解，但逐列比较 `.txt` 时，`data`、`Elos`
和 `Nlos` 会同时反号。ECAT 不提供 legacy 计算模式：新结果统一写 canonical
along-heading 表示；读取已有 `.txt` 时则使用文件中已经配对的 data/projection，
不要只翻转其中一项。

## Python 用法

右视 GAMMA 解缠相位：

```python
from eqtools.csiExtend.sarUtils.readGamma2csisar import GammasarReader

sar = GammasarReader(
    name="S1_phase",
    lon0=lon0,
    lat0=lat0,
    directory_name=".",
    mode="unwrapped_phase",
)
sar.extract_raw_grd(prefix="geo_20250101_20250113", byte_order="native")
sar.read_observation()
sar.print_input_summary()
```

左视数据：

```python
sar = GammasarReader(
    name="nisar_left",
    lon0=lon0,
    lat0=lat0,
    directory_name=".",
    mode="unwrapped_phase",
)
sar.config.acquisition_look_side = "left"
sar.extract_raw_grd(prefix="nisar_pair")
sar.read_observation()
```

必须在 `extract_raw_grd()` 前确定采集侧和角度协议，因为角度栅格在读取阶段就会
归一化。

<a id="gmtsar-direct-projection-grd"></a>

## GMTSAR direct projection

GMTSAR-style reader 不读取 azimuth/incidence，而是读取 value 和 ENU projection：

```yaml
sar_config:
  reader: gmtsar
  mode: range_offset
  files:
    value: range.grd
    projection:
      east: enu/e.grd
      north: enu/n.grd
      up: enu/u.grd
```

内置 mode 使用明确协议：

| mode | input projection axis | input projection direction |
| --- | --- | --- |
| `unwrapped_phase` | `los` | `ground_to_sensor` |
| `los_displacement` | `los` | `ground_to_sensor` |
| `range_offset` | `los` | `ground_to_sensor` |
| `azimuth_offset` | `azimuth` | `along_heading` |

产品不同时才写：

```yaml
sar_config:
  projection_convention:
    input_projection_axis: los
    input_projection_direction: sensor_to_ground
```

projection direction 必须和 axis 匹配：

- `los`：`ground_to_sensor` 或 `sensor_to_ground`；
- `azimuth`：`along_heading` 或 `opposite_heading`。

不再支持 `same_as_value`、`same_as_observation` 或 `canonical`。输入 projection
与原始 value 的正方向是两个独立事实。

若 azimuth observation 只提供 LOS projection，reader 可以结合
`acquisition_look_side` 推导 heading；反向从纯 azimuth projection 推导 LOS
不可能，因为缺少 incidence。

<a id="print_input_summary"></a>

## `print_input_summary()`

summary 分开报告：

- `Final observation spec`：`observation_type`、`raw_value_convention`、波长；
- `Angle geometry`：`azimuth_angle_role`、`acquisition_look_side`；
- `Direct projection convention`：输入/目标 projection axis 和 direction；
- 原始角度均值、projection 三分量均值、观测值 robust/full 范围。

这三个块分别回答“标量是什么”“角度代表什么”“投影指向哪里”，不会用一个字段
混合表达。

## eqtools 到 CSI 的边界

reader 在调用 CSI `read_from_binary()` 前已经完成：

1. 原始标量转成目标正方向；
2. 角度或直接 ENU 分量转成目标 projection；
3. 删除 value 非有限或 projection 无效的像元；
4. 保证 `vel` 与 `los/projection` 一一对应。

因此后续 CSI、非线性几何反演和 BLSE/VCE 直接使用：

```text
predicted_scalar = predicted_ENU dot data.los
```

若一组观测整体进行符号换基，令 $S$ 为对角元素为 $\pm1$ 的符号矩阵，则：

$$
\mathbf d'=S\mathbf d,\qquad G'=SG,\qquad C'=SCS^T
$$

只要 value、Green 函数投影和 covariance 成对变换，weighted least squares 与
Bayesian likelihood 不变。对整组 azimuth 数据统一双反号时 $S=-I$，因此
$C'=C$。

下游不应再次根据卫星左右视或产品平台翻转数据。

## 常见错误

- 把 `acquisition_look_side` 当成原始 value 正号。
- 把 `.azi` 文件名自动理解为 heading；应以产品协议和
  `azimuth_angle_role` 为准。
- 为左视数据同时修改 angle role 和 look side，却没有证据表明 angle 文件语义改变。
- 用 `range_offset` 读取实际 LOS displacement，或反过来。
- GAMMA 字节序不匹配却把异常数值归因于符号。
- 在 GMTSAR 配置中放 angle 字段，或在 GAMMA 配置中放 direct projection 字段。

相关页面：

- [SAR 投影与方向约定](../concepts/sar_projection_conventions.md)
- [InSAR 降采样](../workflows/02_insar_downsampling.md)
- [Downsampling App](downsampling_app.md)
