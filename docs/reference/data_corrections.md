# 数据改正项与 Frame Transform

本页说明 `geodata.polys`、`poly_bounds` 和 `data_corrections` 的统一语义。这里的
`poly` 是历史字段名，更准确的概念是数据改正项或 nuisance transform：

```text
d = G_slip m_slip + H_corr c + noise
```

其中 `G_slip m_slip` 是断层滑动或其他源的物理响应，`H_corr c` 是每个数据集自己的
offset、ramp、参考框架平移或框架形变项。数据改正项会和滑动一起估计，但不应解释为断层滑动。

所有改正项参数都使用进入反演矩阵后的观测数值单位。主配置的 `units.observation` 用来说明这个单位：累计位移反演通常写 `m`，震间速度反演通常写 `m/yr` 或 `mm/yr`。ECAT 假设观测、Green 函数预测、滑动变量和约束右端项已经统一到同一数值单位；该字段不会自动重标定原始数据，只用于固定 Euler loading、矩震级换算和改正项物理解释报告。

## 什么时候使用

| 场景 | 建议 |
| --- | --- |
| InSAR/LOS 有整体零点偏移 | 用 `1` |
| InSAR/LOS 有长波线性 ramp | 用 `3` |
| InSAR/LOS 需要交叉项 | 用 `4`，先确认不会吸收断层长波信号 |
| GPS 参考框架有整体平移 | 用 `translation` |
| GPS 需要平移加旋转、Helmert 或 strain | 只在有明确框架误差或块体模型需要时使用，优先做敏感性测试 |
| 震间块体运动的 Euler/内部应变模型 | 不要当作普通 `polys` 使用；应放在震间配置或专门块体模型流程中 |

过强的数据改正项会吸收真实的长波形变。正式反演中应报告启用了哪些改正项、参数边界和结果量级。

## 配置位置

线性 BLSE/VCE 主配置直接使用 `geodata.polys`：

```yaml
geodata:
  # Python geodata = [asc, desc, gps]
  verticals: [true, true, true]
  polys: [3, 3, translation]
```

非线性几何 SMC 使用同一个入口，但当前只开放受控子集；需要逐数据集或逐参数设置先验时，再写
`data_corrections`：

```yaml
geodata:
  polys: [3, translation]
  poly_bounds: [Uniform, -1000.0, 1000.0]
  data_corrections:
    enabled: true
    datasets:
      asc:
        parameter_bounds:
          offset: [Uniform, -0.05, 0.05]
          x_ramp: [Uniform, -0.5, 0.5]
          y_ramp: [Uniform, -0.5, 0.5]
      gps_campaign:
        bounds: [Uniform, -10.0, 10.0]
```

`geodata.polys` 的顺序必须和 Python 脚本里的 `geodata = [...]` 顺序一致。

## 线性 BLSE/VCE 支持范围

线性 BLSE/VCE 会把 `geodata.polys` 传给 CSI 数据对象的 `getTransformEstimator()`。因此支持范围由实际数据类型决定。

| 数据类型 | 常用设置 | 参数含义 |
| --- | --- | --- |
| InSAR/SAR LOS | `null` | 不估计改正项 |
| InSAR/SAR LOS | `1` | 标量 offset |
| InSAR/SAR LOS | `3` | 标量 offset, x ramp, y ramp |
| InSAR/SAR LOS | `4` | 标量 offset, x ramp, y ramp, xy cross term |
| InSAR/SAR LOS | `strain`, `eulerrotation`, `internalstrain` | CSI 支持的高级投影项；常规 ECAT 配置不建议默认使用 |
| leveling | `1`, `3`, `4` | 与单分量 InSAR 类似 |
| optical/cross-fault offset | `1`, `3`, `4` | 对每个观测分量分别估计 offset/ramp |
| GPS | `translation` | east/north 平移；若使用完整垂直分量则再加 up 平移 |
| GPS | `translationrotation`, `full`, `strain*`, `eulerrotation`, `internalstrain` | 高级框架或形变项，见下节 |

当前 CSI GPS 实现不支持给 GPS 使用整数 `1/3/4`。GPS 需要使用字符串 transform，例如
`translation`。如果把 GPS 写成 `polys: 3`，可能在组装 transform 时失败，或造成参数语义错误。

多断层线性反演中，数据改正项只应组装一次；ECAT 当前会把 poly columns 放在第一个参与组装的 source
后面，后续 source 不再重复组装同一组数据改正项。`bounds_config.yml` 中的 `poly` 边界应设置给实际持有
poly columns 的 source。

## 实现入口和坐标约定

以下公式描述的是 CSI/ECAT 组装进反演矩阵的 `H_corr` 列。实现入口主要是：

| 数据对象 | 实现入口 | 常用 transform |
| --- | --- | --- |
| SAR/InSAR scalar | `csi.insar.insar.getPolyEstimator()` | `1`, `3`, `4` |
| SAR/InSAR scalar | `csi.insar.insar.get2DstrainEst()`, `getEulerMatrix()`, `getInternalStrain()` | `strain`, `eulerrotation`, `internalstrain` |
| GPS | `csi.gps.gps.get2DstrainEst()` | `translation`, `translationrotation`, `strain*` |
| GPS | `csi.gps.gps.getHelmertMatrix()` | `full` |
| GPS | `csi.gps.gps.getEulerMatrix()`, `getInternalStrain()` | `eulerrotation`, `internalstrain` |
| Optical/cross-fault offset | `csi.opticorr.opticorr.getPolyEstimator()` | per-component `1`, `3`, `4` |

`x, y` 是 CSI/ECAT 由经纬度投影得到的局部平面坐标。下面的归一化坐标都是无量纲量；因此 ramp
参数的单位仍是数据单位，而不是“数据单位/km”。如果需要物理梯度，需要再除以对应的归一化长度。

## 数学来源：低阶空间场

数据改正项不是任意给反演增加自由度，而是用低阶空间场近似那些比断层近场信号更长波的误差或参考框架差异。
对一个缓慢变化的标量误差场 `f(x, y)`，在数据集中心附近做 Taylor 展开：

```text
f(x, y) = f0
        + fx (x - x0)
        + fy (y - y0)
        + fxy (x - x0)(y - y0)
        + higher-order terms
```

保留常数项得到 offset；保留两个一阶项得到线性 ramp；再保留交叉项得到 `xy_cross`。SAR/InSAR 的
`polys: 1/3/4` 就是这个标量场近似。

对 GPS 这类水平矢量速度，低阶场的一阶近似是：

```text
u(r) = t + A r
u = [east, north]^T
r = [x, y]^T
A = [[dE/dx, dE/dy],
     [dN/dx, dN/dy]]
```

`t` 是整体平移；`A` 是二维速度梯度。地球物理上通常把 `A` 分解为：

```text
A = symmetric strain + antisymmetric rotation
```

CSI 的 2D strain 参数化等价于：

```text
A = [[exx,              0.5(exy + omega)],
     [0.5(exy - omega), eyy             ]]
```

因此：

```text
epsilon_xx = exx
epsilon_yy = eyy
epsilon_xy = 0.5 exy
theta_cw   = 0.5 omega
```

上式是在归一化坐标中的梯度。如果 `x' = (x - x0) / B`、`y' = (y - y0) / B`，则物理空间中的梯度还要除以
`B`。例如 `epsilon_xy = exy / (2B)`，顺时针旋转梯度 `theta_cw = omega / (2B)`。

Helmert/similarity transform 是同一个一阶矢量场的受限版本。它要求速度梯度只能由等比例尺度和刚体旋转组成：

```text
A = scale I + theta J_cw
I     = [[1, 0], [0, 1]]
J_cw  = [[0, 1], [-1, 0]]
```

也就是说 Helmert 允许整体平移、整体旋转和等比例放大/缩小，但不允许各向异性伸缩或剪切。`strain`
比 Helmert 更自由，因此更容易吸收真实构造长波信号；只有有明确框架误差或区域内部形变假设时才应使用。

## 模式关系和使用场景

GPS 水平 transform 可以按二维速度场的自由度理解。令：

```text
u(r) = t + A r
```

不同字符串 transform 的差别就是对 `t` 和 `A` 施加不同限制：

| 模式 | 数学限制 | 物理含义 | 常见使用场景 | 主要风险 |
| --- | --- | --- | --- | --- |
| `translation` | `A = 0` | 整个 GPS 网络平移 | 参考框架零点不一致；GPS 与 InSAR 联合时只允许整体 offset | 不能表示旋转、尺度或区域内部形变 |
| `translationrotation` | `A = theta J`，CSI 中 `theta = omega / 2` | 平移 + 刚体旋转；无尺度项 | 小范围 GPS 网络有残余框架旋转，但不想估计尺度 | 不是 Helmert 路径；参数不能直接和 `full` 的 `theta` 比较 |
| `full` | `A = scale I + theta J` | 2D Helmert/similarity：平移、旋转、等比例尺度 | 明确需要参考框架旋转或尺度修正 | 仍不能表示剪切或各向异性伸缩 |
| `strainonly` | `t = 0`, `A = E` | 均匀内部应变，无整体平移和旋转 | 外部已固定框架，只想测试区域拉张、压缩或剪切 | 很少单独适用；容易和断层深部长波信号混淆 |
| `strainnorotation` | `A = E` | 平移 + 均匀内部应变，无刚体旋转 | 允许 frame offset 和区域 strain，但不允许整体旋转 | 比 Helmert 更容易吸收构造长波 |
| `strainnotranslation` | `t = 0`, `A = E + theta J` | 内部应变 + 刚体旋转，无整体平移 | 平移已由其他方式固定，但仍需测试旋转/strain | 对参考框架零点敏感 |
| `strain` | `A` 为任意 2D 一阶速度梯度 | 平移 + 内部应变 + 刚体旋转；等价于完整水平 affine field | 研究区域长波形变或做敏感性测试 | 自由度最大，最容易吸收真实断层或块体运动信号 |
| `eulerrotation` | `v = w cross r` | 球面刚体旋转；参数是 Cartesian Euler vector | 震间块体运动或大范围框架旋转 | 不应作为普通 GPS ramp；块体归属和符号约定必须明确 |
| `internalstrain` | 球面局部 strain basis | 块体或区域内部连续应变 | 震间模型中给 block 增加内部应变 | 会和 Euler motion、断层 coupling 和普通 poly trade off |

层级关系可以概括为：

```text
translation
  ⊂ translationrotation
  ⊂ full Helmert-like field
  ⊂ strain / full 2D affine field
```

这个包含关系只描述“能表示的水平速度场集合”。在 CSI 实现中，`translationrotation` 来自
`get2DstrainEst(strain=False)`，`full` 来自 `getHelmertMatrix()`；二者不是同一个函数的不同参数开关。
因此它们的旋转列虽然形状相似，但归一化尺度和参数含义不同。

### 相对关系、刚体旋转和内部应变

选择 GPS frame transform 时，最容易混淆的是“坐标转换是否改变几何形状”和“速度场改正是否改变站点间速度差”。
七参数 Helmert/similarity 坐标转换可写为：

```text
X_new = T + (1 + s) R X
```

作为坐标系转换，它不包含剪切或各向异性拉张；平移和旋转保持点间距离，统一尺度只让所有距离同比例改变。
因此它不是一般意义上的内部变形模型。

但在 GPS 速度场中估计的是这些转换参数的速率或等效速度项：

```text
v_corr(r) = Tdot + Omegadot x r + sdot r
```

这时只有纯 `translation` 不改变站点间速度差：

```text
(v_i + Tdot) - (v_j + Tdot) = v_i - v_j
```

`translationrotation` 不产生对称应变张量，但它会给不同位置的站点添加不同速度：

```text
v_i - v_j -> (v_i - v_j) + theta J (r_i - r_j)
```

这个附加速度差垂直于基线方向，代表刚体旋转；它不表示拉张、压缩或剪切，但会改变图上看到的相对速度矢量。
`full` 的 rotation 部分也是刚体旋转，scale 部分则是各向同性尺度率；它不含剪切或各向异性伸缩，但会改变基线长度变化率。
`strain` 和 `internalstrain` 才允许一般对称应变，包括方向性伸缩和剪切。

可以用下面的表判断风险：

| 模式 | 改变共同零点 | 改变站点间速度差 | 对称应变含义 |
| --- | --- | --- | --- |
| `translation` | 是 | 否 | 无 |
| `translationrotation` | 是 | 是 | 无；刚体旋转 |
| `full` | 是 | 是 | 只有各向同性尺度率，无剪切和各向异性伸缩 |
| `strain` | 是 | 是 | 一般二维均匀应变，含剪切和各向异性伸缩 |
| `eulerrotation` | 是 | 是 | 球面刚体旋转；不是 block 内部应变 |
| `internalstrain` | 依赖中心和是否同时估计平移 | 是 | block 或区域内部连续应变 |

因此，“不改变内部形状”不等于“不改变速度场相对关系”。如果科学目标是断层 loading 或 coupling，
应优先确认 frame transform 改变的是参考框架表达，而不是吸收了真实构造长波场。

### 等价条件和物理量换算

CSI 在矩阵中使用归一化坐标，反演得到的 `poly` 参数通常不是可以直接解释的物理梯度。若数据单位是
`velocity_unit`，局部投影坐标单位是 `coord_unit`，则下面的物理梯度单位是
`velocity_unit / coord_unit`。如果速度是 `mm/yr`、坐标是 `km`，再乘 `1e-6` 才是近似的
dimensionless strain-rate `yr^-1`。

`strain` family 使用：

```text
x' = (x - x0) / B
y' = (y - y0) / B
```

因此反演参数对应的真实一阶速度梯度为：

```text
A_strain =
[[exx / B,               0.5 (exy + omega) / B],
 [0.5 (exy - omega) / B, eyy / B              ]]
```

分解成应变和旋转：

```text
epsilon_xx = exx / B
epsilon_yy = eyy / B
epsilon_xy = exy / (2B)
theta_cw   = omega / (2B)
```

`full` Helmert-like transform 使用另一套归一化：

```text
x'' = (x - x0) / M
y'' = (y - y0) / M
```

它的真实一阶速度梯度为：

```text
A_full =
[[scale / M,  theta / M],
 [-theta / M, scale / M]]
```

因此 `full` 的 `theta` 和 `strain` 的 `omega` 只有在换回真实梯度后才可比较。两者产生相同旋转梯度的条件是：

```text
omega / (2B) = theta / M
```

`full` 是 `strain` 一阶场的受限子集。二者在水平速度场上完全等价时，需要同时满足：

```text
exx / B = scale / M
eyy / B = scale / M
exy     = 0
omega / (2B) = theta / M
```

也就是说，`strain` 中没有剪切，东西向和南北向伸缩相等，并且旋转梯度与 Helmert 旋转梯度一致。
`translationrotation` 与无尺度 `full` 的旋转场等价时，则需要：

```text
scale = 0
omega / (2B) = theta / M
```

这也是为什么不能直接比较 `omega` 和 `theta` 的数值：它们来自不同矩阵、不同归一化长度和不同参数化。

`eulerrotation` 与上述局部平面 rotation 也不能直接等同。`eulerrotation` 先在球面上定义刚体旋转：

```text
v = w cross r
```

再投影到每个站点的 east/north 方向。小范围内，它可以被 Taylor 展开成近似的局部平移加一阶速度梯度：

```text
v(lon, lat) ~= v0 + A_euler delta r
```

所以在很小区域里，Euler rotation 的预测可能看起来像 `translation + rotation/strain`。但它只有
`[wx, wy, wz]` 三个球面刚体参数，约束来自整个地球几何；`strain/full` 的一阶场则是局部经验 basis。
因此震间块体刚体运动应优先用 `eulerrotation` 或专门的 block 配置，而不是用 `full` 或 `strain`
间接吸收。

`internalstrain` 与 `strainonly` 都表示对称水平应变，但坐标和语义不同。`strainonly` 使用投影平面归一化坐标：

```text
v_E = exx x' + 0.5 exy y'
v_N = eyy y' + 0.5 exy x'
```

`internalstrain` 使用相对中心的球面弧长坐标：

```text
x_s = R Delta_lambda cos(phi)
y_s = R Delta_phi
v_E = sxx x_s + 0.5 sxy y_s
v_N = syy y_s + 0.5 sxy x_s
```

所以在小区域、坐标近似一致时，它们可以表达相似的对称速度梯度；但 `internalstrain` 更适合和
`eulerrotation` 一起表示 block 内部连续应变，而 `strainonly` 是普通 GPS frame transform 里的局部
归一化 basis。两者的参数单位和缩放不同，不能直接比较。

CSI/ECAT 代码实现中，常用归一化量保存在数据对象上，便于后处理换算：

| transform | 归一化量 | 代码中常见属性 |
| --- | --- | --- |
| `strain*`, `translationrotation` | `B` | `gps.StrainNormalizingFactor` 或 `gps.TransformNormalizingFactor["base"]` |
| `full` | `M` | `gps.HelmertNormalizingFactor` |
| `internalstrain` | 球面弧长，`R` 为 Earth radius | `gps.InternalStrainNormalizingFactor` 记录中心点 |
| `eulerrotation` | Earth-centered radius `R` | 参数直接进入 Euler vector basis |

线性反演后，参数通常在 `fault.polysol[data.name]` 中，列顺序应按上面的参数表读取。换算时必须使用同一次
构建设计矩阵时记录的 `B`、`M` 和中心点；重新读取数据或改变投影中心后，旧的归一化量不应混用。

### 反演后参数报告

BLSE/VCE 和 Bayesian inversion 对象提供一个只读后处理接口，用来把已经估计出的
`fault.polysol[data.name]` 转成更容易检查的物理量：

```python
report = inversion.collect_data_correction_parameters()
inversion.print_data_correction_report(report)

df = inversion.data_correction_parameters_to_dataframe(report)
```

该接口不重新组装矩阵、不修改 `mpost`、`fault.slip` 或 `fault.polysol`。它只读取：

- `fault.poly` 和 `fault.polysol` 中的 raw correction 参数；
- `fault.polysolindex` 或 `transform_indices` 中的参数列号；
- 数据对象上的 `TransformNormalizingFactor`、`StrainNormalizingFactor`、
  `HelmertNormalizingFactor`、`InternalStrainNormalizingFactor` 等归一化 metadata。

输出会按 dataset/source 列出 raw 参数、归一化信息和可解释物理量。例如：

| transform | 报告内容 |
| --- | --- |
| `1/3/4` | offset、`x/y/xy` 归一化后的标量梯度 |
| `translationrotation`, `strain*` | 平移、真实一阶速度梯度、应变张量、顺时针旋转梯度 |
| `full` | local Helmert-like 平移、旋转梯度和尺度梯度 |
| `eulerrotation` | Cartesian Euler vector 和 Euler pole |
| `internalstrain` | 以内应变中心为参考的内部应变张量 |

如果归一化 metadata 缺失，报告会给 warning，而不会静默重新计算。科研解释时应优先修复
metadata 来源，因为重新计算中心或尺度可能和原设计矩阵不一致。

### 数据改正参数关系约束

有些应用不是要给每个数据集各自估计一套完全独立的改正项，而是要让几个数据集共享某些参数。常见例子包括：

- 两个 GPS 数据集使用共同的 frame translation，表示同一整体参考框架偏移。
- 两个 SAR 数据集共享物理 ramp 梯度，而不是共享归一化坐标下的 raw 系数。
- 组合 transform 中只约束某个子项，例如 `["eulerrotation", "translation"]` 里的 `translation`。

BLSE/VCE 和 Bayesian inversion 对象提供一个薄的只读解析 + 线性等式构造层：

```python
Aeq, beq, meta = inversion.add_data_correction_equality(
    [
        {
            "owner": "MainFault",
            "dataset": "gps_a",
            "transform": "translation",
            "components": ["east", "north"],
        },
        {
            "owner": "MainFault",
            "dataset": "gps_b",
            "transform": "translation",
            "components": ["east", "north"],
        },
    ],
    space="raw",
    name="common_gps_translation",
)
```

该接口只生成并注册 `Aeq @ x = beq`，不改变数据对象、不重新组装 Green's functions、不修改
`fault.slip` 或 `fault.polysol`。调用时机应在反演对象已经构建、`poly_positions` 已知之后，求解
`run()` 之前；MPI 脚本中应在所有 rank 上调用，不要只放在 `rank == 0` 分支。

`owner` 是持有 poly columns 的 source/fault 名称。若只有一个 source 配置了该数据集的 poly，可以省略；
多源反演中建议显式填写，避免约束落到错误的参数块。`dataset` 必须是数据对象名；`transform` 在组合
poly 中用于定位子项；`components` 省略时表示该 transform 的全部参数。

`space` 决定等式约束比较的量：

| `space` | 约束含义 | 适用场景 |
| --- | --- | --- |
| `raw` | 直接比较求解器参数列 | `translation`、Euler vector，或两个数据集使用完全相同归一化时 |
| `physical` | 先把 raw 参数乘以简单物理缩放再比较 | SAR ramp 梯度、GPS strain/rotation/Helmert 梯度等 |

例如两个 SAR 数据集的 `x_ramp` 和 `y_ramp` 若使用不同空间范围，raw 系数不能直接相等。此时应比较物理梯度：

```python
inversion.add_data_correction_equality(
    [
        {"owner": "MainFault", "dataset": "asc", "components": ["x_ramp", "y_ramp"]},
        {"owner": "MainFault", "dataset": "desc", "components": ["x_ramp", "y_ramp"]},
    ],
    space="physical",
    name="common_sar_ramp_gradient",
)
```

`space="physical"` 当前执行的是逐参数缩放，例如 `x_ramp/Nx`、`y_ramp/Ny`、`exy/(2B)`、
`omega/(2B)`、`theta/M`。它适合约束“同名物理梯度相等”，但不是完整的场等价转换。若要强制两个不同中心
的 affine field 在物理空间完全相同，需要同时处理

```text
A1 = A2
t1 - A1 x01 = t2 - A2 x02
```

这比逐参数缩放更强，当前不自动生成。对于这种情况，建议先用 `build_data_correction_equality()` 检查矩阵，
再根据科学问题手工写自定义约束。

若只想检查即将注册的矩阵，不写入 constraint manager，可使用：

```python
Aeq, beq, meta = inversion.build_data_correction_equality(
    refs,
    space="physical",
)
print(meta)
```

`meta` 会列出每个 ref 解析到的 owner、dataset、transform、全局列号和缩放因子。正式反演前建议打印一次，
确认列号和 `components` 符合预期。

### 数据改正参数边界

`poly_bounds` 和 `bounds_config.yml:poly` 适合给一个 source 持有的全部 data-correction 参数设置同一组宽边界。
如果只想限制某个数据集、某个 transform 或某几个分量，可以在脚本中使用：

```python
meta = inversion.set_data_correction_bounds(
    dataset="gps_campaign",
    transform="translation",
    bounds={"east": [-5.0, 5.0], "north": [-5.0, 5.0]},
)
print(meta)
```

组合 transform 需要显式指定子项：

```python
inversion.set_data_correction_bounds(
    dataset="gps_campaign",
    transform="translation",
    bounds=[-5.0, 5.0],
)
```

当多个数据集都使用同名 transform 时，可以批量更新：

```python
inversion.set_data_correction_bounds(
    datasets="all",
    transform="translation",
    bounds={"east": [-5.0, 5.0], "north": [-5.0, 5.0]},
)
```

`datasets="all"` 只会更新实际配置了该 `transform` 的数据集；为了避免组合 `polys` 中的歧义，它必须和
`transform=...` 一起使用。若同一数据集的 data-correction columns 被多个 source 持有，应显式传入
`owner="FaultName"`。

`bounds` 可以是一个 `[lower, upper]` 数值对，也可以是分量字典。字典键同时接受常用别名，例如 `tx/ty`
对应 `east/north`，`xy_ramp` 对应整数 polynomial 的 `xy_cross`。`space` 的含义和 equality helper 一致：

| `space` | 边界解释 | 典型用法 |
| --- | --- | --- |
| `raw` | 直接限制求解器系数 | `translation`、Euler vector、同一归一化下的 transform |
| `physical` | 先按归一化因子换算到简单物理梯度，再反算回 raw bounds 写入 solver | SAR/GPS ramp 或 strain 梯度的量级控制 |

例如 SAR 的 `polys: 3` 中，`space="physical"` 下的 `x_ramp` 边界会除以数据对象保存的 x 方向归一化长度后写回 raw
系数。`set_data_correction_bounds()` 返回的 `meta` 会同时列出 `input_bounds`、`raw_bounds`、列号和缩放因子，
建议在正式反演前打印一次核对。

调用时机和 equality helper 相同：必须在反演对象已经构建、`poly_positions` 已知之后，求解或构建 SMC target
之前。BLSE/VCE 在 `run()` 时读取最新 constraint manager 状态；`SMC_F_J` 会在
`make_F_J_target_for_parallel()` / `walk_F_J()` 构建 target 时冻结当前 bounds，因此之后若再改 data-correction
bounds，需要重新构建 target。

### 预测、移除与单项改正检查

CSI/ECAT 的线性预测可写成：

```text
d = G_slip m_slip + H_corr c + residual
```

其中 `G_slip m_slip` 是断层滑动预测，`H_corr c` 是数据改正项。常用调用对应关系如下：

| 操作 | 比较对象 | 是否改变数据对象 |
| --- | --- | --- |
| `data.buildsynth(faults, poly="include")` | `data` vs `G_slip m_slip + H_corr c` | 不改变 `data.vel_enu` / `data.vel` |
| `data.buildsynth(faults, poly=None)` | `data` vs `G_slip m_slip` | 不改变 `data.vel_enu` / `data.vel` |
| `data.removeTransformation(fault)` 后再 `buildsynth(poly=None)` | `data - H_corr c` vs `G_slip m_slip` | **会原地修改观测数据** |

因此下面两个残差在数学上应一致：

```text
data - buildsynth(poly="include")
    == (data - transformation) - buildsynth(poly=None)
```

但 `removeTransformation()` 是原地操作，反复执行会重复扣除同一组改正项；如果之后又用
`buildsynth(poly="include")` 写 residual，就会把“已扣除改正项的数据”和“包含改正项的预测”混在一起。
做诊断图时建议使用拷贝对象：

```python
import copy

data_corr = copy.deepcopy(data)
data_corr.removeTransformation(fault, computeNormFact=False, computeIntStrainNormFact=False)
data_corr.buildsynth(faults, poly=None, vertical=vertical)
data_corr.plot(faults=faults, data=["data", "synth"])
```

若只想看某一个 transform 的影响，例如组合设置 `["eulerrotation", "translation"]` 中的
`eulerrotation`，不要直接用 `removeTransformation()`，因为它会计算整组 transform。
反演对象提供只读 helper：

```python
parts = inversion.calculate_data_correction_prediction_parts(
    dataset="gps_campaign",
    source="MainFault",
    faults=faults,
    direction="sd",
    vertical=False,
)

slip_only = parts["slip_only"]
transform = parts["transformation"]
total = parts["total"]
euler_part = parts["single_transforms"]["eulerrotation"]
translation_part = parts["single_transforms"]["translation"]
```

该 helper 在数据对象的临时拷贝上调用 CSI 的 `buildsynth()` 和 `computeTransformation()`，
不会修改原始 `data.vel_enu`、`data.vel`、`data.synth` 或 `data.transformation`。返回结果中：

| 字段 | 含义 |
| --- | --- |
| `observed` | 原始观测数组 |
| `slip_only` | `G_slip m_slip` |
| `transformation` | 整组 `H_corr c` |
| `total` | `G_slip m_slip + H_corr c` |
| `single_transforms` | 每个子 transform 的单项预测 |
| `residual_total` | `observed - total` |
| `residual_after_transform` | `(observed - transformation) - slip_only` |
| `total_consistency` | `total - (slip_only + transformation)`，用于检查分解是否闭合 |

默认 `compute_norm_fact=False` 和 `compute_int_strain_norm_fact=False`，即使用组装设计矩阵时保存的归一化信息。
这更适合反演后诊断；只有明确要重算归一化尺度时才改为 `True`。

若只需要快速判断当前模型中多项式是否合理，可直接打印紧凑诊断：

```python
diag = inversion.print_data_correction_diagnostics(
    dataset="gps_campaign",
    source="MainFault",
    faults=faults,
    direction="sd",
    vertical=False,
)
```

该接口同样只读，返回的 `diag` 不包含大数组，只给出：

- `residual_slip_only` 与 `residual_total` 的 RMS，用于判断 correction 是否显著降低残差。
- 每个子 transform 的 RMS 和最大绝对预测量，用于判断单项量级。
- 子 transform 两两之间的相关系数、`rms(a + b) / max(rms(a), rms(b))` 抵消比。
- `total - slip_only - transformation` 的闭合误差，用于检查分解是否和 CSI 预测一致。

如果报告提示两个 transform `strongly cancel each other`，说明它们在观测空间内几乎相反。
这种情况下总预测可能更好，但单独解释某个 Euler pole、translation 或 strain 参数会很危险；
应优先做敏感性测试、收紧先验，或改用更明确的块体/框架模型。

选择改正项时建议按最小充分原则：

| 现象或需求 | 首选设置 | 不建议直接跳到 |
| --- | --- | --- |
| SAR/InSAR 只有整体零点问题 | `1` | `4` 或 GPS-style strain |
| SAR/InSAR 残差有清楚长波线性趋势 | `3` | `strain`, `eulerrotation` |
| SAR/InSAR 线性 ramp 不足且有简单扭曲 | `4`，并做敏感性测试 | 高阶自由项 |
| GPS 与其他数据集参考零点不一致 | `translation` | `full` 或 `strain` |
| GPS 网络存在明确框架旋转但不需要尺度 | `translationrotation`，并和 `translation` 对比 | `strain` |
| GPS 网络存在明确尺度差或 Helmert 框架差 | `full`，并检查 scale 量级 | `strain` |
| GPS 区域内部确有均匀应变假设 | `strain` 或 `internalstrain`，并收紧先验 | 无边界宽泛估计 |
| 震间块体刚体运动 | 专门的 Euler/block 配置 | 普通 `polys` |

不要同时把同一个物理过程放进多个改正项。例如震间块体旋转已经由 Euler/block 模型解释时，再给同一 GPS
数据集打开宽泛的 `full` 或 `strain`，可能把真实 loading 或 coupling 信号吸收到 nuisance 参数中。

## 文献依据和风险边界

SAR/InSAR 的低阶 ramp 不是对所有误差的物理建模，而是对长波误差的经验近似。InSAR 文献中反复讨论过大气传播延迟、
轨道误差和参考零点/基准差异等问题：大气传播延迟会在干涉图中形成空间相干相位误差，轨道误差也会表现为长波相位坡度或缓变项
（见 Zebker et al., 1997; Bürgmann et al., 2000; Hanssen, 2001; Shirzaei & Walter, 2011;
Fattahi & Amelung, 2015）。这些现象是使用 offset/ramp 作为 nuisance basis 的主要依据之一，
而不是说每一景数据都必须估计 ramp。

在 ECAT 中可以按下面的边界理解：

| 来源或机制 | 常见数据表现 | 可考虑的改正项 | 需要检查 |
| --- | --- | --- | --- |
| 参考零点或统一基准差异 | 整体偏移 | SAR `1`；GPS `translation` | 改正量是否只是数据零点量级 |
| 残余轨道或长波处理误差 | 单轨道内缓变坡度 | SAR `3` 或必要时 `4` | ramp 是否削弱远场构造信号 |
| 对流层延迟或大气长波结构 | 空间相干长波残差，可能和地形相关 | SAR `3/4`，或先做外部大气改正 | 残差是否仍与地形、时间或轨道相关 |
| GPS 参考框架差异 | 网络整体平移、旋转或尺度差 | `translation`，必要时 `full` | 是否有独立框架理由支持旋转/尺度 |
| 区域连续形变或块体内部应变 | 多站点呈一阶水平速度梯度 | `strain` 或 `internalstrain` | 是否与断层深部滑动、块体 Euler 或 coupling trade off |

这些风险不是某个特定案例的经验断言，而是线性反演的矩阵相关性问题：

```text
d = G_slip m_slip + H_corr c
```

如果 `G_slip` 的某些列和 `H_corr` 的长波列在观测空间中相似，反演就可以在 `m_slip` 与 `c` 之间重新分配信号。
因此打开高自由度 correction 后，应至少比较：

- 断层滑动或 coupling 的空间分布是否系统性变弱。
- 远场残差是否确实减少，而不是只把构造长波吸收到 nuisance 参数。
- 改正项参数是否落在可解释的框架、轨道或大气误差量级内。
- 不同 `polys` 设置下的主要科学结论是否稳定。

没有独立证据时，不应把 `full`、`strain`、`eulerrotation` 或 `internalstrain` 当作“提高拟合”的默认开关。

## SAR/InSAR 标量 Polynomial

SAR/InSAR 进入 CSI 后是一个标量观测。ECAT 的 SAR reader 约定见
[SAR 投影和观测约定](../concepts/sar_projection_conventions.md)：

```text
scalar_observation = ENU_displacement dot projection
```

`polys: 1/3/4` 的 polynomial 是直接加到这个标量观测上的长波改正项，不是 ENU 三分量位移，也不再乘
LOS/projection。CSI 的实现先取数据集中心和范围：

```text
X = (x - x0) / Nx
Y = (y - y0) / Ny
Nx = max(abs(x - x0))
Ny = max(abs(y - y0))
```

对应参数顺序和公式为：

| `geodata.polys` | 参数顺序 | 标量改正项 `delta d` |
| --- | --- | --- |
| `1` | `[offset]` | `offset` |
| `3` | `[offset, x_ramp, y_ramp]` | `offset + x_ramp X + y_ramp Y` |
| `4` | `[offset, x_ramp, y_ramp, xy_cross]` | `offset + x_ramp X + y_ramp Y + xy_cross X Y` |

这些项可以理解为标量误差场的 Taylor basis：

- `offset` 是整幅 SAR 标量观测的零点误差。
- `x_ramp` 和 `y_ramp` 是沿局部投影坐标的长波线性趋势，常用于近似轨道、大气或参考面残差。
- `xy_cross` 是一个双线性交叉项，可描述简单扭曲形态，但也更容易吸收真实长波形变。

CSI 会把设计矩阵整体乘以数据对象的 `factor`。大多数 ECAT reader 已经在读入阶段完成单位转换；如果脚本手动设置
`factor`，需要把它一起计入参数边界。

SAR/InSAR 也有高级字符串 transform，但它们不是普通 orbit/ramp：

| transform | 参数顺序 | 标量预测 |
| --- | --- | --- |
| `strain` | `[exx, exy, eyy]` | `los_E (exx x' + 0.5 exy y') + los_N (0.5 exy x' + eyy y')` |
| `eulerrotation` | `[wx, wy, wz]` | `los_E v_E(w) + los_N v_N(w)` |
| `internalstrain` | `[sxx, syy, sxy]` | `los_E u_E(s) + los_N u_N(s)` |

这里 `x', y'` 使用 CSI 的 strain 归一化；`v_E, v_N` 是 Euler vector 产生的水平速度；`u_E, u_N`
是球面近似内部应变产生的水平速度。它们都会投影到 SAR 标量观测方向。除非有明确参考框架或块体运动假设，常规
SAR/InSAR 反演建议优先使用 `1/3/4`。

注意 SAR 的字符串 `strain` 只有 3 个参数，不包含 GPS `strain` 里的 `translation` 和 `omega`；
SAR 也不支持 GPS `full` Helmert 族。若需要同时估计 SAR 标量 ramp 和高级投影项，可在 `geodata.polys`
中写组合列表，例如 `[3, strain]`，但应在报告中单独检查每个子项的量级和 trade-off。

## GPS Frame Transform

GPS 的 frame transform 用于描述数据集参考框架误差或长波背景项，不是断层滑动。推荐默认只用
`translation`。`full`、`strain` 和 `internalstrain` 可能强烈吸收区域长波信号，只有在有清楚的参考框架或
块体运动假设时才应使用。

常用参数数如下：

| `geodata.polys` 值 | 水平 GPS | 含垂直 GPS | 参数顺序 |
| --- | ---: | ---: | --- |
| `translation` | 2 | 3 | `[tx, ty, (tz)]` |
| `translationrotation` | 3 | 4 | `[tx, ty, omega, (tz)]` |
| `full` | 4 | 7 | 水平 `[tx, ty, theta, scale]`；ENU `[tx, ty, tz, rx, ry, rz, scale]` |
| `strainonly` | 3 | 4 | `[exx, exy, eyy, (tz)]` |
| `strainnorotation` | 5 | 6 | `[tx, ty, exx, exy, eyy, (tz)]` |
| `strainnotranslation` | 4 | 5 | `[exx, exy, eyy, omega, (tz)]` |
| `strain` | 6 | 7 | `[tx, ty, exx, exy, eyy, omega, (tz)]` |
| `eulerrotation` | 3 | 3 | `[wx, wy, wz]` |
| `internalstrain` | 3 | 3 | `[sxx, syy, sxy]` |

### GPS 2D Strain 和 Rotation

`translation`、`translationrotation` 和 `strain*` 都来自 CSI 的 2D strain estimator，而不是 Helmert
matrix。CSI 先定义：

```text
x' = (x - x0) / B
y' = (y - y0) / B
B  = max(max(abs(x - x0)), max(abs(y - y0)))
```

完整 `strain` 水平项的列顺序为：

```text
east  = tx + exx x' + 0.5 exy y' + 0.5 omega y'
north = ty + 0.5 exy x' + eyy y' - 0.5 omega x'
```

这个公式来自二维速度梯度的应变-旋转分解。把水平速度写成矩阵形式：

```text
[east ]   [tx]   [exx               0.5(exy + omega)] [x']
[north] = [ty] + [0.5(exy - omega)  eyy             ] [y']
```

其中对称部分是应变：

```text
strain =
[[exx,     0.5 exy],
 [0.5 exy, eyy    ]]
```

反对称部分是顺时针旋转：

```text
rotation =
[[0,           0.5 omega],
 [-0.5 omega, 0          ]]
```

所以 `exy` 是 engineering shear 参数，物理张量分量是 `0.5 exy`；`omega` 是两倍的顺时针旋转梯度参数。
如果要和标准应变率或旋转率比较，还需要除以归一化长度 `B`。

子模式只是从这组列中选择一部分：

| transform | 水平公式 |
| --- | --- |
| `translation` | `east = tx`, `north = ty` |
| `translationrotation` | `east = tx + 0.5 omega y'`, `north = ty - 0.5 omega x'` |
| `strainonly` | `east = exx x' + 0.5 exy y'`, `north = 0.5 exy x' + eyy y'` |
| `strainnorotation` | `translation + strainonly` |
| `strainnotranslation` | `strainonly + rotation` |
| `strain` | `translation + strainonly + rotation` |

实现上，CSI 的 `getTransformEstimator()` 会把这些字符串映射为 `get2DstrainEst()` 的三个布尔开关：

| transform | CSI 调用逻辑 | 选中的水平列 | 对站点间速度差的影响 | 内部应变含义 |
| --- | --- | --- | --- | --- |
| `translation` | `strain=False, rotation=False` | `tx, ty` | 不改变 | 无 |
| `translationrotation` | `strain=False` | `tx, ty, omega` | 改变；刚体旋转项随位置变化 | 无对称应变 |
| `strainonly` | `rotation=False, translation=False` | `exx, exy, eyy` | 改变 | 只有对称应变；无整体零点和平移 |
| `strainnorotation` | `rotation=False` | `tx, ty, exx, exy, eyy` | 改变 | 平移 + 对称应变；不允许刚体旋转 |
| `strainnotranslation` | `translation=False` | `exx, exy, eyy, omega` | 改变 | 对称应变 + 刚体旋转；无整体平移 |
| `strain` | 默认全部打开 | `tx, ty, exx, exy, eyy, omega` | 改变 | 平移 + 对称应变 + 刚体旋转 |

这几个 `strain*` 的公式族相同，区别不是坐标定义不同，而是允许哪些自由度进入反演。可以按下面理解：

- `translation` 只修正 GPS 数据集共同速度零点，是最不容易吸收构造信号的选择。
- `translationrotation` 仍是刚体场，不引入内部应变；但会改变站点间速度矢量，所以会和 Euler rotation 或局部框架旋转竞争。
- `strainonly` 不含平移。如果数据参考零点没有被外部固定，单独使用它通常不稳，因为零点误差不能被吸收。
- `strainnorotation` 适合“允许区域均匀应变，但不希望整体刚体旋转抢占 Euler/block rotation”的测试。
- `strainnotranslation` 适合“平移已由其他方式固定，但仍要测试应变和旋转”的受控场景。
- `strain` 是完整二维 affine field，自由度最大；它最容易提高拟合，也最容易和断层深部滑动、块体 loading 或 InSAR ramp trade off。

CSI 注释中 rotation 正号为 clockwise positive。由于旋转列带 `0.5`，`omega` 不应直接等同于 Helmert
里的旋转角参数。若需要解释为物理梯度，应结合 `B` 和 CSI 的 `0.5` 参数化一起换算。

当 CSI 判断 GPS 数据包含完整垂直分量时，上述 2D estimator 会在最后追加一个 `tz` 垂直平移列：

```text
up = tz
```

这只是垂直 offset，不是三维 strain。使用 `verticals: false` 时应确保 GPS 数据对象的 `obs_per_station`
和垂直速度 NaN 情况一致，否则可能出现矩阵行数不匹配。

### GPS `full` Helmert-Like Transform

`full` 不走 2D strain estimator，而是调用 CSI 的 Helmert matrix。水平时，CSI 使用另一套归一化：

```text
x'' = (x - x0) / M
y'' = (y - y0) / M
M   = (max(abs(x - x0)) + max(abs(y - y0))) / 2
```

水平 `full` 的公式为：

```text
east  = tx + theta y'' + scale x''
north = ty - theta x'' + scale y''
```

这个旋转项来自小角度旋转矩阵。若顺时针旋转角为 `theta`：

```text
R_cw(theta) =
[[ cos(theta), sin(theta)],
 [-sin(theta), cos(theta)]]
```

小角度时 `cos(theta) ~= 1`，`sin(theta) ~= theta`，因此：

```text
R_cw(theta) r - r
= [[0, theta], [-theta, 0]] [x'', y'']^T
= [theta y'', -theta x'']^T
```

尺度项来自相似变换 `(1 + scale) r - r = scale r`。把平移、旋转和尺度相加，就得到：

```text
[east ]   [tx]   [0  theta] [x'']   [scale 0    ] [x'']
[north] = [ty] + [-theta 0] [y''] + [0     scale] [y'']
```

因此 `full` 是局部 2D Helmert/similarity transform：平移、旋转和等比例尺度。它和
`translationrotation` 都能产生水平旋转场，但参数尺度不同：

```text
translationrotation rotation gradient = omega / (2B)
full Helmert rotation gradient        = theta / M
```

二者的共同点是 rotation 部分为反对称速度梯度，不产生二维对称应变。区别是：

- `translationrotation` 只含平移和刚体旋转；它会改变站点间速度差，但不改变基线长度变化率。
- `full` 额外包含 `scale`；作为 frame scale rate，它在速度场中表现为各向同性尺度率，会统一改变基线长度变化率。
- `strain` / `internalstrain` 则允许一般对称应变，可能表示真实区域内部形变，也可能和深部滑动、Euler loading trade off。

含垂直 GPS 时，CSI 使用 7 参数局部 Helmert-like 矩阵：

```text
[east ]   [1 0 0   0  -z''  y''  x''] [tx   ]
[north] = [0 1 0  z''   0  -x''  y''] [ty   ]
[up   ]   [0 0 1 -y''  x''   0   z''] [tz   ]
                                             [rx   ]
                                             [ry   ]
                                             [rz   ]
                                             [scale]
```

实现上 CSI 将站点高度 `z''` 固定为 0，于是实际进入矩阵的是：

```text
east  = tx + rz y'' + scale x''
north = ty - rz x'' + scale y''
up    = tz - rx y'' + ry x''
```

因此这不是完整地心坐标意义上的 3D Helmert；它只是在 ENU 数据向量中加入 `tz, rx, ry, rz, scale`
列。正式科研解释时应把它称为 local Helmert-like correction，而不是完整三维坐标框架转换。

### GPS Euler 和 Internal Strain

`eulerrotation` 估计的是 Cartesian Euler vector：

```text
w = [wx, wy, wz]
```

这里的 raw `wx/wy/wz` 是进入反演矩阵的系数，不一定已经是物理 radians/year。报告函数会根据主配置
`units.observation` 把它转回物理量：若 `units.observation: m/yr`，raw 数值就是 radians/year；若
`units.observation: mm/yr`，raw 数值约为物理 `w` 的 1000 倍，报告时会乘以 `1e-3` 再转换为 Euler pole。固定
`fixed_pole`/`fixed_vector` 的输入则始终按物理角速度写，不需要用户按观测单位手动缩放。

其原理是刚体球面旋转：

```text
v = w cross r
r = R [cos(phi) cos(lambda), cos(phi) sin(lambda), sin(phi)]^T
```

再把 `v` 投影到站点的 east 和 north 单位向量上，就得到 CSI 的水平速度 basis。在每个站点
`lon = lambda`, `lat = phi`：

```text
v_E = -R cos(lambda) sin(phi) wx
      -R sin(lambda) sin(phi) wy
      +R cos(phi) wz

v_N =  R sin(lambda) wx
      -R cos(lambda) wy
```

GPS 三分量时垂向行为为 0，因为刚体 Euler rotation 在球面切平面内给出水平速度。

`internalstrain` 使用相对数据集中心的球面近似内部应变。先把经纬度差转换为局部弧长：

```text
x_s = R Delta_lambda cos(phi)
y_s = R Delta_phi
```

然后使用不含平移和旋转的水平 strain basis：

```text
Delta_lambda = lambda - lambda0
Delta_phi    = phi - phi0

v_E = sxx x_s + 0.5 sxy y_s
v_N = syy y_s + 0.5 sxy x_s
```

这两个 transform 更适合震间块体或区域框架问题；不要把它们当作普通 GPS offset/ramp 的替代品。

## Optical 和水平 offset 数据

`opticorr` 等双分量水平数据的整数 polynomial 是 per-component block diagonal 形式。CSI 先定义：

```text
X = (x - x0) / Nx
Y = (y - y0) / Ny
```

每个水平分量各自一套参数：

```text
east_corr  = ae0 + aeX X + aeY Y + aeXY X Y + ...
north_corr = an0 + anX X + anY Y + anXY X Y + ...
```

因此同样写 `polys: 3` 时，单分量 SAR 是 3 个参数，而双分量 optical 是 6 个参数。不要把不同数据类型的
整数 `polys` 参数个数混用。

## 非线性几何 SMC 支持范围

新版 nonlinear geometry SMC 有自己的 `data_corrections` 归一化层。为了保持参数命名、先验和绘图输出可控，
当前只开放：

| 数据类型 | 支持设置 |
| --- | --- |
| SAR/InSAR/leveling | `1`, `3`, `4` |
| GPS | `translation` |

如果需要在非线性几何采样中加入 GPS `full`、`strain` 或 `eulerrotation`，应先在代码中完整镜像 CSI 的矩阵列顺序、
参数命名、垂直分量行为和测试，再开放配置字段；不要直接把线性 BLSE 的所有字符串 transform 无差别搬进
SMC 配置。

## 边界和输出检查

`poly_bounds` 或 `bounds_config.yml` 中的 `poly` 边界应按数据单位设置。建议至少检查：

- 改正项参数是否明显大于数据本身的量级。
- 打开 ramp 后断层滑动是否发生系统性长波削弱。
- GPS `translation` 是否只是整体平移，而不是替代块体运动模型。
- `geodata.verticals` 是否与 GPS 数据对象的 `obs_per_station` 和垂直速度 NaN 情况一致。
- 多数据集时，`polys`、`verticals`、`sigmas` 和脚本中的 `geodata` 顺序是否完全一致。

## 参考文献

- Zebker, H. A., Rosen, P. A., & Hensley, S. (1997). Atmospheric effects in interferometric synthetic aperture radar surface deformation and topographic maps. https://doi.org/10.1029/97JB01189
- Bürgmann, R., Rosen, P. A., & Fielding, E. J. (2000). Synthetic aperture radar interferometry to measure Earth's surface topography and its deformation. https://doi.org/10.1146/annurev.earth.28.1.169
- Hanssen, R. F. (2001). Radar Interferometry: Data Interpretation and Error Analysis. https://doi.org/10.1007/0-306-47633-9
- Shirzaei, M., & Walter, T. R. (2011). Estimating the effect of satellite orbital error using wavelet-based robust regression applied to InSAR deformation data. https://doi.org/10.1109/TGRS.2010.2098354
- Fattahi, H., & Amelung, F. (2015). InSAR bias and uncertainty due to the systematic and stochastic tropospheric delay. https://doi.org/10.1002/2015JB012419
- PROJ contributors. Helmert transform documentation. https://proj.org/en/stable/operations/transformations/helmert.html

## 相关页面

- [非线性几何配置](config_nonlinear_geometry.md)
- [线性滑动配置](config_linear_slip.md)
- [Sigmas 和 Alpha](sigmas_alpha.md)
- [BLSE/VCE](blse_vce.md)
