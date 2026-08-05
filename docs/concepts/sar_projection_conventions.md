# SAR 投影与方向约定

ECAT 对所有 SAR 标量观测使用同一个合同：

```text
scalar_observation = ENU_displacement dot projection
```

`projection` 是 East、North、Up 三分量向量。CSI 中历史上常把它存为
`data.los`，但它也可以表示 range 或 azimuth offset 的观测方向。

## 进入 CSI 后的固定方向

| mode | 标量与 projection 的目标正方向 |
| --- | --- |
| `unwrapped_phase` | 相位转位移后为地面指向传感器 |
| `los_displacement` | 地面指向传感器 |
| `range_offset` | 地面指向传感器 |
| `azimuth_offset` | 沿平台航向 |

原始文件可以使用不同正号。reader 会先分别规范化标量和 projection，再传给
CSI；二者不应通过某个“same as”占位字段互相推断。

## 标量与 projection 必须成对

设地表 ENU 位移为 $\mathbf{u}=(u_E,u_N,u_U)$，进入 CSI 的 projection 为
$\mathbf{p}=(p_E,p_N,p_U)$，则：

$$
d = u_Ep_E + u_Np_N + u_Up_U
$$

如果把同一观测改写成相反坐标基：

$$
d'=-d,\qquad \mathbf{p}'=-\mathbf{p}
$$

物理方程保持不变，因为：

$$
d'-\mathbf{u}\cdot\mathbf{p}'
=-\left(d-\mathbf{u}\cdot\mathbf{p}\right)
$$

因此，同时翻转 scalar 和 projection 只是等价表示；只翻转其中一个则会改变物理
形变正负号。不要为了匹配某个历史文件而在脚本中单独对 value 或 projection
乘 `-1`。

## 三类参数各司其职

### `acquisition_look_side`

表示成像地面条带位于平台航向的哪一侧：

```yaml
acquisition_look_side: right  # right | left
```

它描述采集几何，不描述标量正号，也不表示 LOS 向量自身朝向。以平台向北为例：

| 采集侧 | 传感器指向地面 | 地面指向传感器 |
| --- | --- | --- |
| `right` | 东 | 西 |
| `left` | 西 | 东 |

因此，左视卫星并不意味着“输入方位向必然指向右侧”。必须先看角度文件代表的物理轴。

### `azimuth_angle_role`

只说明归一化后的 azimuth angle 数值代表什么方向：

- `heading`：平台沿轨航向；
- `sensor_to_ground`：传感器指向成像地面；
- `ground_to_sensor`：成像地面指向传感器。

若角度已经是 `sensor_to_ground` 或 `ground_to_sensor`，构造 LOS projection 时
不需要 `acquisition_look_side`。只有在 heading 与 LOS/azimuth 两类轴之间转换时，
左右视才参与旋转。

## 角度、左右视与投影公式

ECAT 内部把 azimuth 统一成“从 East 起、逆时针为正”的 ENU 角度
$\alpha$。若输入角度 $\beta$ 是“从 North 起、顺时针为正”，则：

$$
\alpha=(90^\circ-\beta)\bmod 360^\circ
$$

对应水平单位向量：

$$
\mathbf{q}=(\cos\alpha,\sin\alpha)
$$

定义顺时针和逆时针 $90^\circ$ 旋转：

$$
R_{\rm cw}(E,N)=(N,-E),\qquad
R_{\rm ccw}(E,N)=(-N,E)
$$

GAMMA 默认把 `.azi` 解释为
$\mathbf{q}_{sg}$，即 `sensor_to_ground`。地面指向传感器为：

$$
\mathbf{g}=-\mathbf{q}_{sg}
$$

平台 heading 为：

$$
\mathbf{h}=
\begin{cases}
R_{\rm ccw}(\mathbf{q}_{sg}), & \text{right-looking}\\
R_{\rm cw}(\mathbf{q}_{sg}), & \text{left-looking}
\end{cases}
$$

若 incidence $\theta$ 从天顶量起，则 LOS/range projection 为：

$$
\mathbf{p}_{los}
=
(g_E\sin\theta,\;g_N\sin\theta,\;\cos\theta)
$$

azimuth-offset projection 为：

$$
\mathbf{p}_{az}=(h_E,\;h_N,\;0)
$$

这解释了为什么 GAMMA `.azi` 已是 `sensor_to_ground` 时，LOS/range 不使用
`acquisition_look_side`，而从该 cross-track 方向推导 azimuth heading 时必须使用。

### `raw_value_convention`

只说明原始标量的编码或正号，例如：

- `unwrapped_phase`；
- `toward_sensor` / `away_from_sensor`；
- `along_heading` / `opposite_heading`。

它不改变角度文件的物理含义。比如“角度文件为地面指向传感器，但 range offset
原值以远离传感器为正”是完全合法的组合：projection 保持朝向传感器，标量乘
`-1` 后进入 CSI。

GAMMA 默认标量转换为：

$$
d_{\rm phase}=-\frac{\lambda}{4\pi}\phi,\qquad
d_{\rm range}=-x_{\rm range},\qquad
d_{\rm az}=x_{\rm az}
$$

这里 phase 和 range 的目标方向都是 ground-to-sensor，azimuth 的目标方向是
along-heading。

早期 ECAT GAMMA azimuth 输出曾使用
$(d_{\rm old},\mathbf{p}_{\rm old})=(-d_{\rm az},-\mathbf{p}_{az})$。
它与当前 canonical 表示物理等价，但 `.txt` 中的 `data/Elos/Nlos` 会同时反号。
比较历史结果时必须比较配对后的观测方程，不能只比较 `data` 一列。

## GAMMA 左视数据

纯 GAMMA `.phs/.rsc/.azi/.inc` 产品仍使用 `reader: gamma`。如果 `.azi`
沿用 GAMMA 默认的 `sensor_to_ground` 含义，NISAR 左视数据只需把采集侧改成
`left`：

```yaml
sar_config:
  reader: gamma
  mode: unwrapped_phase
  acquisition_look_side: left
  files:
    prefix: nisar_pair
  read:
    byte_order: native
```

这里没有新增 NISAR 文件格式 reader；卫星平台只影响采集几何。若实际 `.azi`
文件不是 GAMMA 默认含义，再使用高级 `geometry_convention.azimuth_angle_role`
明确说明。对 `unwrapped_phase`、`los_displacement` 和 `range_offset`，若
`.azi` 已直接表示 `sensor_to_ground`，LOS projection 可由角度本身确定，
`acquisition_look_side` 不再参与数值旋转；该字段仍应如实记录采集侧，并会在
需要由 cross-track 方向推导 along-heading（例如 `azimuth_offset`）时参与计算。

## 检查顺序

1. 确认 `reader` 与实际处理平台格式一致。
2. 确认 `mode` 与 value raster 的物理量一致。
3. 对角度型 reader，确认 `acquisition_look_side` 和 `azimuth_angle_role`。
4. 对 GAMMA 二进制，确认 `read.byte_order`；`native` 保持历史默认。
5. 运行 `ecat-downsample -f downsample.yml -s`。
6. 检查 `print_input_summary()` 中分离显示的 scalar、angle geometry 和 projection。
   其中 `acquisition_look_side_used` 明示本次转换是否实际使用左右视。
7. 在 data、synthetic、residual 三类图中复核最终符号。

## 继续阅读

- [SAR Reader 参考](../reference/sar_reader.md)
- [InSAR 降采样](../workflows/02_insar_downsampling.md)
- [GAMMA SAR quick-look 与配置生成](../examples/gamma_sar_quicklook.md)
