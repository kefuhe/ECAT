# Rake Constraints

本页说明 ECAT 如何在 BLSE/VCE 和 `SMC_FJ + ss_ds` 中把 rake 扇区
转换为线性不等式，以及 `FULLSMC + magnitude_rake` 如何把同一物理字段
用作 sampled-rake bounds。

普通使用只需要配置 `rake_angle` 或调用 fault/patch 公开方法；只有检查
矩阵列和符号时才需要阅读后半页。

## 阅读路径

| 当前问题 | 阅读位置 |
| --- | --- |
| 不知道该用 fault sector、patch sector 还是 fixed rake | 三种不同语义 → 选择建议 |
| 需要确认正负滑动与角度方向 | 符号约定 → 线性公式 |
| 需要设置配置或 runtime 约束 | 配置与运行时入口 → Patch-level override |
| 担心多断层或 `SMC_FJ` 列错位 | 参数列顺序（高级） → 矩阵结构核验 |

## 三种不同语义

| 类型 | 数学形式 | 推荐入口 |
| --- | --- | --- |
| fault-level rake sector | 不等式扇区 | `rake_angle` / `update_fault_rake_limits()` |
| patch-level rake | 在线性模式覆盖 fault 扇区；在 magnitude/rake 模式覆盖 sampled bounds | `patch_constraints` |
| fixed rake | 等式直线 | `set_fixed_rake_constraints()` |

三者不能混为一个 `update_rake` 操作。在线性模式中 fault 和 patch sector
最终编译成一个 `rake_sector` 不等式组；fixed rake 独立保存为
`fixed_rake` 等式组。FULLSMC magnitude/rake 不生成这两个矩阵组。

## 适用模式

| 模式 | `rake_angle` 的作用 |
| --- | --- |
| BLSE/VCE | 约束求解得到的 `strikeslip`/`dipslip` |
| `SMC_FJ + ss_ds` | 约束每个样本内部的线性滑动解 |
| `FULLSMC + magnitude_rake` | fault/patch `rake_angle` 都是 sampled-rake 边界，不生成线性矩阵 |
| `FULLSMC + ss_ds` | 无线性 rake consumer；配置标 inactive，runtime patch-rake 报错 |
| `FULLSMC + rake_fixed` | 无 sampled-rake 参数；配置标 inactive，runtime patch-rake 报错 |

## 符号约定

ECAT/CSI 断层滑动分量按以下约定解释：

```text
ss > 0 : left-lateral strike slip
ds > 0 : reverse dip slip
rake = atan2(ds, ss)
```

因此：

| rake sector | 对应的主要滑动方向 |
| --- | --- |
| `[-60, 60]` | 正 strike-slip 为主 |
| `[120, 240]` | 负 strike-slip 为主 |
| `[-90, 90]` | `ss >= 0` |
| `[90, 270]` | `ss <= 0` |

这里的约定与卫星 look side、LOS 正方向无关；rake 描述断层面内的
strike-slip/dip-slip 组合，不描述 SAR 观测投影。

## 线性公式

对每个 patch，连续 sector `[rake_min, rake_max]` 生成两行：

```text
ss * sin(rake_min) - ds * cos(rake_min) <= 0
-ss * sin(rake_max) + ds * cos(rake_max) <= 0
```

两条半平面的交集把 `(ss, ds)` 限定在从 `rake_min` 到 `rake_max`
逆时针张开的凸扇区内。

固定 rake 使用等式：

```text
ss * sin(rake0) - ds * cos(rake0) = 0
```

该等式只限定一条穿过原点的直线。若还需要限定沿直线的正负方向，应另外使用
slip bounds 或符号不等式。

### 避免用 component bounds 重复表达 rake 方向

`rake_angle` 与 `strikeslip`/`dipslip` 可以同时使用，但应分别承担不同职责：

- rake sector 表达断层面内允许的**方向或分量耦合关系**；
- component bounds 表达各分量独立的**物理幅值范围**或宽泛数值保护。

不要在没有独立科学理由时，用 component bound 再重复 rake 已经表达的符号。例如
`rake_angle: [150, 210]` 对每个 patch 生成

\[
0.5\,ss+0.866\,ds\le 0,
\qquad
0.5\,ss-0.866\,ds\le 0.
\]

两式相加可得 \(ss\le 0\)。这时再设置 `strikeslip: [-10, 0]`，其上界行
\(ss\le0\) 与 rake 两行线性相关；零滑 patch 处三行还可能同时活动。约束可行域没有
因此变得更严格，但连续 VCE 的活动集 KKT 快速路径可能因工作集秩亏而回退到标准 QP。
回退只影响耗时，不改变原问题或最终证书。

若 rake sector 已经表达方向，通常把 component bounds 保留为宽泛幅值保护：

```yaml
rake_angle:
  FaultA: [150.0, 210.0]

strikeslip:
  FaultA: [-10.0, 10.0]
dipslip:
  FaultA: [-10.0, 10.0]
```

这里 `strikeslip` 的正值区间并不会被线性解实际采用，因为 rake sector 仍要求
\(ss\le0\)。如果只知道某一分量的符号、并不知道可信的 rake 范围，则反过来：不声明
rake sector，只用 `strikeslip: [-10, 0]` 或相应的 `dipslip` bound。

若 rake 与 component bound 分别来自独立物理先验，应保留两者，即使快速路径因此回退；
不能为了提速删除有意义的约束。也不要用 `-epsilon` 或很小的正上界规避诊断，这会改变
约束语义或让结果依赖数值容差。component box 约束各分量，不等价于总滑动幅值上界。
VCE 路线和回退原因的查看方法见
[连续 QP 快速路径](blse_vce.md#连续-qp-快速路径可选)。

## 两种 angle 校验

### 线性 sector

ECAT 按以下张角解释 sector：

```text
aperture = (rake_max - rake_min) mod 360
0 < aperture <= 180 degrees
```

角度不必预先归一化到 `[0, 360)`。例如 `[300, 60]` 与 `[-60, 60]`
等价。

| 输入 | 结果 |
| --- | --- |
| `[-60, 60]` | 合法，张角 120° |
| `[120, 240]` | 合法，张角 120° |
| `[90, -90]` | 合法，张角 180° |
| `[-120, 120]` | 非法，张角 240°，非凸 |
| `[-180, 180]` | 非法，零/整周退化 |

不要用 `[-180, 180]` 表示“不约束”。不需要 rake sector 时，应删除对应声明、
关闭配置中的 rake 消费，或清除 runtime fault/patch 声明。

### FULLSMC sampled-rake bounds

`FULLSMC + magnitude_rake` 的 rake 参数已经是显式采样变量，因此这里只做普通
bounds 校验：

```text
lower 和 upper 必须有限
lower <= upper
不做 modulo
```

所以 `[-180, 180]` 在这里合法，表示 sampled rake 可取该闭区间；它与上面的
360° sector 退化不是一回事。不要把线性 sector 校验函数复用于 sampled bounds。

## 配置与运行时入口

fault-level 配置：

```yaml
rake_angle:
  FaultA: [-30.0, 60.0]
```

runtime fault 更新：

```python
# 合并指定 fault；未写出的 runtime fault 保留
inversion.update_fault_rake_limits({
    "FaultA": [-30.0, 60.0],
})

# 替换完整 runtime fault-rake 映射
inversion.replace_fault_rake_limits({
    "FaultA": [-20.0, 50.0],
    "FaultB": [120.0, 240.0],
})

# 只清除 runtime fault rake；config 和 patch rake 保留
inversion.clear_fault_rake_limits()
```

fixed rake：

```python
inversion.set_fixed_rake_constraints({
    "FaultB": -90.0,
})
inversion.clear_fixed_rake_constraints()
```

`set_fixed_rake_constraints({})` 是 no-op，不等价于 clear。

## Patch-level override

当一条断层不同段具有不同运动机制时，使用 patch 规则：

```yaml
rake_angle:
  FaultA: [120.0, 240.0]

patch_constraints:
  FaultA:
    - name: local_left_lateral
      selector: {patches: [10, 11, 12]}
      rake_angle: [-60.0, 60.0]
```

解析顺序：

```text
config fault rake
  -> runtime fault rake
      -> config patch rake
          -> runtime patch rake
              -> one rake_sector matrix
```

在线性模式中，patch 规则替换选中 patch 的 fault 默认 sector。它不是
`fault matrix + patch matrix`，否则选中 patch 会被错误地限制为两个 sector 的交集。

在 `FULLSMC + magnitude_rake` 中，同一规则直接覆盖选中 sampled-rake 参数的
lower/upper bounds：

```yaml
patch_constraints:
  FaultA:
    - name: sampled_rake_window
      selector: {patches: [10, 11, 12]}
      rake_angle: [-180.0, 180.0]
```

该模式的 config patch-rake 属于 bounds，由 `use_bounds_constraints` 控制；
线性 sector 属于 rake matrix，由 `use_rake_angle_constraints` 控制。

运行时：

```python
inversion.add_patch_constraints({
    "FaultA": [{
        "name": "runtime_local_segment",
        "selector": {"patches": [10, 11, 12]},
        "rake_angle": [-60.0, 60.0],
    }],
})
```

重叠 patch rake 默认报错；后面的规则只有显式 `overwrite: true` 才能覆盖。
`replace_patch_constraints()` 和 `clear_patch_constraints()` 只改变 runtime
patch 声明，不删除配置中的规则。

runtime patch-rake 只接受确实有 consumer 的两种组合：

- BLSE/VCE 或 `SMC_FJ + ss_ds`；
- `FULLSMC + magnitude_rake`。

其他 FULLSMC 参数化会立即报错，避免用户误以为显式更新已经生效。

## 参数列顺序（高级）

不要从 `slipdir` 字符串书写顺序推断列号。Fault 参数按 canonical `sdtc`
分量顺序排列，因此 `slipdir: ds` 和 `slipdir: sd` 都是：

```text
source 1:
  all strikeslip patches
  all dipslip patches
  optional tensile/coupling
  source 1 data-correction parameters

source 2:
  all strikeslip patches
  all dipslip patches
  optional tensile/coupling
  source 2 data-correction parameters
```

多断层 rake matrix 也是逐 source 定位，不是先排列所有断层的 `ss`、再排列
所有断层的 `ds`。

使用公开布局检查真实列：

```python
layout = inversion.get_linear_parameter_layout()
assert layout["active"]
print(layout["space"], layout["width"], layout["global_offset"])
for block in layout["blocks"]:
    print(block["source"], block["component"],
          block["start"], block["stop"])
```

BLSE/VCE 中 `start/stop` 对应完整线性向量。`SMC_FJ` 中矩阵对应线性后缀，
`global_offset` 给出该后缀在完整采样向量中的起点；矩阵列号本身仍从 0 开始。

## 矩阵结构核验

每个受约束 patch 有两行 sector 不等式：

```text
row_count = 2 * constrained_patch_count
nonzero_count_per_row = 2
```

每一行的非零列应恰好对应同一 patch 的 `strikeslip` 和 `dipslip`。若一行出现
四个以上非零项，通常是多个 patch 被误写到同一行；若出现全零行，通常是行偏移
或 source block 计算错误。

可用：

```python
snapshot = inversion.get_constraint_snapshot(
    include_matrices=True,
    validate=True,
)
A = snapshot["inequality_constraints"]["rake_sector"]["A"]
```

矩阵快照只读。修改 rake 应回到 fault/patch 公开方法。

## 选择建议

- 只限制某一滑动分量正负时，优先使用 `strikeslip`/`dipslip` bounds。
- 已知一个允许范围时使用 rake sector。
- 同时使用 rake 与 component bounds 时，让前者表达方向、后者表达独立的宽泛幅值范围；
  不要无意中重复同一符号约束。
- 机制非常明确且确实需要零宽角度时才使用 fixed rake。
- 不同断层段机制不同时使用 patch selector，不要把整条 fault 强制为同一 sector。

完整的 config/runtime 层级、raw matrix API 和事务说明见
[Constraint Manager](constraint_manager.md)。
