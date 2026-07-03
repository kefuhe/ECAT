# Rake Constraints

本页面解释 ECAT/csiExtend 在线性反演中如何把 `rake_angle`
转换为线性不等式约束。它面向需要检查约束矩阵、符号约定和多断层参数排列的高级用户。

## 适用范围

`rake_angle` 在不同反演模式下含义不同：

| 模式 | 参数化 | `rake_angle` 的作用 |
| --- | --- | --- |
| `BLSE/VCE + ss_ds` | 直接反演 `strikeslip` 和 `dipslip` | 转成线性不等式 `A @ m <= b` |
| `SMC_F_J + ss_ds` | 线性子问题反演 `strikeslip` 和 `dipslip` | 转成线性不等式 `A @ m <= b` |
| `FULLSMC + magnitude_rake` | 直接采样 magnitude 和 rake | 作为 rake 参数边界 |
| `rake_fixed` | 固定 rake，只估计滑动量 | 使用固定 rake 关系，不生成 rake 区间不等式 |

BLSE/VCE 中，如果主配置 `use_rake_angle_constraints: true`，并且
`bounds_config.yml` 含有 `rake_angle`，约束会在初始化时自动加入。

## 符号约定

对普通断层源，ECAT 的滑动分量约定为：

```text
ss : left-lateral strike slip
+ds : reverse dip slip
```

其中 `ss` 是走滑分量，`ds` 是倾滑分量。rake 使用：

```text
rake = atan2(ds, ss)
```

因此：

| rake 范围 | 近似含义 |
| --- | --- |
| `[-60, 60]` | 正走滑为主，允许一定正负倾滑 |
| `[120, 240]` | 负走滑为主，允许一定正负倾滑 |
| `[-90, 90]` | 要求 `ss >= 0` |
| `[90, 270]` | 要求 `ss <= 0` |

不要用 `[-180, 180]` 表示“不限制 rake”。在线性半平面公式中，
这会退化为一条线性约束，而不是全角度自由。如果不想限制 rake，
应关闭 `use_rake_angle_constraints`，或从 `bounds_config.yml` 中移除该断层的
`rake_angle`。

## 线性公式

给定一个连续 rake 扇区：

```yaml
rake_angle:
  MyFault: [rake_min, rake_max]
```

`rake_min` 和 `rake_max` 不需要预先归一化到 `0-360` 度。ECAT 只按
`(rake_max - rake_min) mod 360` 计算从下限到上限的逆时针开角；例如
`[-90, 90]`、`[90, 270]`、`[300, 60]` 都是合法表达，只要该开角不超过
180 度。

在 `ss_ds` 参数化下，每个 patch 生成两行不等式：

```text
ss * sin(rake_min) - ds * cos(rake_min) <= 0
-ss * sin(rake_max) + ds * cos(rake_max) <= 0
```

这两条半平面共同限定 `(ss, ds)` 落在从 `rake_min` 到 `rake_max`
的凸扇区内。固定 rake 则使用等式：

```text
ss * sin(rake0) - ds * cos(rake0) = 0
```

## 为什么范围不能大于 180 度

上述两条线性不等式只能表示一个凸扇区。以原点为顶点的 rake 扇区如果
张角大于 180 度，就是非凸集合，不能由这一对线性半平面准确表示。

ECAT 因此要求线性 rake 区间满足：

```text
0 < (rake_max - rake_min) mod 360 <= 180 degrees
```

示例：

| 配置 | 是否有效 | 说明 |
| --- | --- | --- |
| `[-60, 60]` | 有效 | 张角 120 度 |
| `[120, 240]` | 有效 | 张角 120 度 |
| `[90, -90]` | 有效 | 张角 180 度，约束 `ss <= 0` |
| `[-120, 120]` | 无效 | 张角 240 度，非凸 |
| `[-180, 180]` | 无效 | 端点相差 360 度，在线性公式中退化 |

如果需要表达大于 180 度的物理可能性，建议改用更直接的
`strikeslip` / `dipslip` bounds，或拆成多个断层段分别设置。

## 线性未知参数排列（高级）

这一小节只在检查约束矩阵列、调试 `A @ m <= b` / `A @ m = b`、
或编写自定义线性约束时需要阅读。普通 BLSE/VCE 用户只需要在配置中按
`strikeslip`、`dipslip` 等分量名设置 bounds 和 constraints，不需要手工维护
列号。

BLSE/VCE 的全局未知参数向量按 source 分块排列。对普通断层，`slipdir`
只表示启用哪些滑动分量，字符顺序不会改变参数列顺序。ECAT/CSI 内部统一按
canonical `sdtc` 顺序排列断层参数，因此 `slipdir: ds` 与 `slipdir: sd`
等价；启用走滑和倾滑时，每条断层内部先排全部走滑，再排全部倾滑，之后是该
断层的数据改正参数：

```text
model vector m =

Fault_1:
  ss_0, ss_1, ..., ss_n
  ds_0, ds_1, ..., ds_n
  poly / data-correction parameters

Fault_2:
  ss_0, ss_1, ..., ss_m
  ds_0, ds_1, ..., ds_m
  poly / data-correction parameters

...
```

如果启用 tensile 或 coupling 分量，它们继续按 canonical `sdtc` 中的顺序跟在
`dipslip` 后面；普通线性滑动反演和 rake 约束通常只使用 `strikeslip` 与
`dipslip`。关键原则是：先按 source 分块，再在每个 Fault source 内按分量块
排列；不是把所有断层的 `ss` 放在全局最前、再把所有断层的 `ds` 放在后面。

因此，多断层 rake 矩阵不能假定“所有断层的 ss 在前、所有断层的 ds 在后”。
它必须逐 source、按分量名查找该断层内部的 `strikeslip` 和 `dipslip` 列。

## 矩阵行排列

每个 patch 有两条 rake 不等式，所以每条断层需要 `2 * n_patch` 行。
多断层时，行排列按断层连续分块：

```text
Fault_1 lower rows
Fault_1 upper rows
Fault_2 lower rows
Fault_2 upper rows
...
```

每一行通常只应该有两个非零系数：

```text
A[row, ss_i] != 0
A[row, ds_i] != 0
```

如果同一行出现 4、6、8 个非零系数，通常说明多个 patch 或多条断层的
rake 半平面被写到了同一行；如果出现大量全零行，则说明约束行数和写入行偏移
不一致。正常矩阵结构应满足：

```text
row_count = 2 * sum(n_patch for constrained faults)
nonzero_count_per_row = 2
```

## 使用建议

- rake 约束不是 BLSE/VCE 求解的必需项；它是物理先验。
- 对单一运动学机制明确的断层，可使用相对窄的 rake 区间。
- 对震间 backslip、loading 符号沿走向改变、或断层不同段机制不同的情况，
  不建议整条断层使用同一个硬 rake 区间。
- 如果只是想限定走滑正负或倾滑正负，优先考虑直接设置
  `strikeslip` / `dipslip` bounds。
- 固定 rake 比 rake 区间更强，应只在机制非常明确时使用。
