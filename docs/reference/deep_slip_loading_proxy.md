# 深部滑动加载代理

本页说明 ECAT 中的 deep-slip loading proxy。它用于固定几何线性滑动反演：把一个或多个深部断层元的自由滑动速率解释为浅部孕震层的长期加载代理，再对浅部断层元施加底边连续、闭锁、蠕滑或 cap 约束，并导出对应的 coupling 字段。

它不是完整块体模型反演。若目标是闭合 block polygon、自动 segment/block topology、同一长断层不同段自动继承不同 block pair，建议使用 Blocks/celeri 这类专门震间块体模型程序，或先用这些程序确定块体拓扑后再把需要的几何和先验转入 ECAT。

## 先选模型

ECAT 当前有两套震间解释路径，字段名和公式不要混用：

| 路径 | loading 来源 | 反演滑动变量解释 | 主要公式 | 入口 |
| --- | --- | --- | --- | --- |
| Euler/block direct-backslip | `fault_loading.blocks[0] - blocks[1]` 投影到断层走向 | `q = backslip_rate` | `coupling_ratio = -q / b`, `creep_rate = b + q` | `interseismic_config.yml`, `calculate_interseismic_fields()` |
| Deep-slip loading proxy | 匹配到的深部 fault slip `b` | `s = shallow_slip_rate` | `coupling_to_deep = (b - s) / b`, `creep_fraction_to_deep = s / b` | `preview_deep_slip_loading_mapping()`, `calculate_deep_slip_loading_fields()` |

如果长期加载来自 Euler pole 或数据集估计的块体旋转，用 [Interseismic Kinematics](interseismic_kinematics.md)。如果长期加载由深部滑块直接作为普通 slip 参数参与同一个线性反演，用本页。

## 变量和公式

对浅部 patch `i` 和匹配的深部 patch `j(i)`：

```text
b_i = matched deep slip rate
s_i = shallow slip or creep rate
D_i = b_i - s_i
K_i = (b_i - s_i) / b_i = 1 - s_i / b_i
C_i = s_i / b_i
```

其中：

- `b_i` 是深部滑动加载代理，不是 Euler/block loading。
- `s_i` 是浅部实际滑动或蠕滑速率，不是 direct backslip `q`。
- `D_i` 是相对深部加载的滑动亏损。
- `K_i` 是相对深部加载的 coupling ratio。
- `C_i` 是相对深部加载的 creep fraction。

典型状态：

| 状态 | 方程 | `K_i` 解释 |
| --- | --- | --- |
| 完全闭锁 | `s_i = 0` | `K_i = 1` |
| 完全跟随深部蠕滑 | `s_i = b_i` | `K_i = 0` |
| 部分闭锁 | `0 < abs(s_i) < abs(b_i)` 且同号 | `0 < K_i < 1` |
| 同向浅部滑动超过深部 | `abs(s_i) > abs(b_i)` | `K_i < 0` |
| 浅部与深部反向 | `s_i * b_i < 0` | `K_i > 1`，需要检查物理解释 |

当 `abs(b_i)` 接近 0 时，比例字段没有稳定意义；结果 metadata 会记录 near-zero deep loading 数量。

## 基本流程

第一版使用脚本 API，不把 deep proxy 配置塞进 `interseismic_config.yml`。这样做是为了先显式检查浅部到深部的几何映射，再决定是否添加硬约束。

### 只计算 coupling，不加深部约束

如果深部 fault 已作为普通滑动源参与反演，即使不调用
`add_deep_slip_loading_constraint()`，也可以在反演后计算相对深部滑动的
`coupling_to_deep`。这时 `b` 是反演得到的深部滑动速率，`s` 是反演得到的浅部滑动速率：

```python
inversion.run(...)
inversion.returnModel(print_stat=False)

result = inversion.calculate_deep_slip_loading_fields(
    shallow_fault="ShallowFault",
    deep_faults=["DeepFault"],
    component="strikeslip",
    zero_tolerance=1.0e-12,
)

inversion.print_deep_slip_loading_report(
    result["evaluation_mapping"],
    result=result,
)

inversion.plot_deep_slip_loading_field(
    "coupling_to_deep",
    result=result,
    cblabel="Coupling to deep slip",
)
```

这种结果是“相对估计深部滑动的 coupling 诊断”。它没有强制浅部底边与深部连续，
也没有强制 `0 <= K <= 1`；若深部滑动被数据、平滑、bounds 或端部效应影响很大，
`coupling_to_deep` 也会随之变化。科研解释时应同时检查
`deep_loading_proxy_rate`、`shallow_slip_rate`、`mapping_distance` 和 near-zero
deep loading 警告。

如果同一反演还估计了 GPS/SAR 的 `eulerrotation`、`internalstrain`、`full` 或
`strain` 等长波数据改正项，可先用
`inversion.print_data_correction_report()` 检查这些参数量级。它们不进入 deep proxy 的
loading 公式，但可能通过数据拟合 trade-off 间接影响深部滑动 `b`。

如果只希望把特定深度范围内的深部 patch 作为加载代理，字段计算时也要传入同一
`deep_selectors`；否则默认会在全部深部 patch 中寻找最近顶部边段：

```python
result = inversion.calculate_deep_slip_loading_fields(
    shallow_fault="ShallowFault",
    deep_faults=["DeepFault"],
    deep_selectors={"DeepFault": {"depth_range": [20.0, 5000.0]}},
    component="strikeslip",
)
```

若要先预览全断层字段评估 mapping，可显式使用：

```python
field_mapping = inversion.preview_deep_slip_loading_mapping(
    shallow_fault="ShallowFault",
    deep_faults=["DeepFault"],
    shallow_selector="all",
    component="strikeslip",
)
```

### 添加底边连续约束

若希望浅部孕震层底边与深部加载代理连续，再建立底边 mapping 并注册约束：

```python
mapping = inversion.preview_deep_slip_loading_mapping(
    shallow_fault="ShallowFault",
    deep_faults=["DeepFault"],
    shallow_selector={"edge": "bottom"},
    deep_selectors={"DeepFault": {"depth_range": [20.0, 5000.0]}},
    component="strikeslip",
)

inversion.print_deep_slip_loading_report(mapping)

inversion.add_deep_slip_loading_constraint(
    mapping=mapping,
    state="bottom_continuity",
)
```

默认推荐先约束浅部孕震层底边：

```text
s_i - b_i = 0
```

这表示浅部孕震层底界及以下自由滑动，和“深部滑块代表长期加载”的物理假设一致。不要默认把浅部所有 patch 都强制等于深部速率；加载、蠕滑和弹性亏损沿深度可以变化。

## 浅深映射

映射规则是：

1. 用 `shallow_selector` 选择浅部 patch，默认建议 `{"edge": "bottom"}`。
2. 用 `deep_selectors` 选择深部候选 patch。
3. 取浅部 patch 中心点 `P_i = (x, y, z)`。
4. 对每个深部候选 patch 推断顶部边段 `A_j B_j`。
5. 选择三维点到线段距离最小的深部 patch。

点到线段距离为：

```text
u_j = B_j - A_j
t_ij = clip(((P_i - A_j) dot u_j) / (u_j dot u_j), 0, 1)
C_ij = A_j + t_ij u_j
d_ij = norm(P_i - C_ij)
j(i) = argmin_j d_ij
```

默认 `coord_frame="same_xy"`，即假设浅部和深部 fault 已在同一套 UTM/本地 `x/y/depth` 坐标中进入反演。这符合 CSI/ECAT 线性反演的常规组织方式。

如果 fault 没有 `edge_triangles_indices`，优先先运行已有边界识别方法生成 top/bottom/left/right；确实无法稳定识别边界时，再用深度带近似，例如：

```python
shallow_selector={"depth_range": [18.0, 22.0]}
```

深度带是 fallback，不应在文档或论文中等同于严格 bottom edge。

## 约束状态

`add_deep_slip_loading_constraint()` 支持以下状态：

| `state` | 矩阵公式 | 用途 |
| --- | --- | --- |
| `bottom_continuity` | `s_i - b_i = 0` | 浅部底边与深部加载代理连续，推荐默认 |
| `full_creep` | `s_i - b_i = 0` | 显式选择的 patch 完全跟随深部滑动 |
| `full_locking` | `s_i = 0` | 显式选择 patch 完全闭锁 |
| `prescribed_creep_ratio` | `s_i - c*b_i = 0` | 指定蠕滑比例 `c` |
| `prescribed_locking` | `s_i - (1-K)*b_i = 0` | 指定闭锁比例 `K` |
| `fixed_shallow_slip` | `s_i = value` | 指定浅部滑动速率 |
| `cap` | 见下节 | 约束浅部同向滑动不超过深部量级 |

`bottom_continuity` 和 `full_creep` 矩阵相同，但语义不同。前者用于浅部底界边界条件，后者用于任意显式 patch group 的蠕滑状态。

Cap 约束是符号敏感的不等式。给定 `sigma=+1` 表示预期正滑动、`sigma=-1` 表示预期负滑动：

```text
-sigma * b_i <= 0
-sigma * s_i <= 0
 sigma * s_i - kmax * sigma * b_i <= 0
```

`motion_sense="dextral"` 对应 `sigma=-1`，`motion_sense="sinistral"` 对应 `sigma=+1`。Cap 表达的是同向且 `abs(s_i) <= kmax * abs(b_i)`；如果研究上允许 over-coupling 或反向浅部滑动，不要对这些 patch 启用 cap。

## 字段和导出

反演完成后计算字段。这里要区分两个选择：

- `mapping` 可以是底边连续约束使用的 mapping，例如只包含浅部底边 patch。
- 字段评估默认使用 `field_shallow_selector="all"`，会把浅部断层所有 patch
  都映射到最近的深部顶部边段并计算 `coupling_to_deep` 等字段。

因此，常用脚本可以继续把底边约束 mapping 传给字段计算；结果图仍覆盖完整浅部断层：

```python
result = inversion.calculate_deep_slip_loading_fields(
    mapping=mapping,
    component="strikeslip",
)
```

如果确实只想检查约束涉及的底边 patch，可显式使用：

```python
result_bottom = inversion.calculate_deep_slip_loading_fields(
    mapping=mapping,
    field_shallow_selector="mapping",
)
```

Canonical 字段为：

| 字段 | 公式 | 说明 |
| --- | --- | --- |
| `deep_loading_proxy_rate` | `b` | 匹配深部 patch 的滑动速率 |
| `shallow_slip_rate` | `s` | 选中浅部 patch 的实际滑动或蠕滑速率 |
| `slip_deficit_to_deep_signed` | `b - s` | 相对深部加载的有符号亏损 |
| `slip_deficit_to_deep_magnitude` | `abs(b - s)` | 亏损量级 |
| `coupling_to_deep` | `(b - s) / b` | 相对深部加载的 coupling |
| `coupling_to_deep_magnitude` | `abs(b - s) / abs(b)` | coupling 量级 |
| `creep_fraction_to_deep` | `s / b` | 浅部蠕滑比例 |
| `mapping_distance` | `d_ij` | 浅部中心到深部顶部边段距离 |

这些数组和字段评估 mapping 的浅部 patch 对齐。公共 mixin 默认评估全浅部断层；
写 center text 时默认只写已评估 patch：

```python
inversion.write_deep_slip_loading_field_centers(
    "coupling_to_deep",
    "output/ShallowFault_coupling_to_deep_centers.txt",
    result=result,
)
```

写 patch GMT 或绘图时会扩展到完整浅部 fault。默认全断层评估时不会产生未选中
patch；如果用户显式做子集评估，patch GMT 默认用 `nan` 填子集外 patch，
而 3-D 绘图默认用有限背景值填充，因为 CSI 的 fault plot 不能用含 `nan` 的
自定义 slip 数组做颜色归一化：

```python
inversion.write_deep_slip_loading_field_gmt(
    "coupling_to_deep",
    "output/ShallowFault_coupling_to_deep.gmt",
    result=result,
)

inversion.plot_deep_slip_loading_field(
    "coupling_to_deep",
    result=result,
    cblabel="Coupling to deep slip",
)
```

### 比值图异常时的诊断

`coupling_to_deep` 是比值字段。若三维图几乎全是同一种颜色，且 colorbar
出现类似 `-5000` 到 `0` 的极端范围，通常不是绘图函数把断层画坏了，而是
`b = deep_loading_proxy_rate` 在部分 patch 上接近 0，或浅部滑动 `s`
的量级明显大于深部 proxy：

```text
coupling_to_deep = (b - s) / b
```

此时应先检查 GMT 注释列或 center text 中的 `shallow_slip_rate` 和
`deep_loading_proxy_rate`，再决定如何展示或解释。常用处理是：

```python
result = inversion.calculate_deep_slip_loading_fields(
    shallow_fault="ShallowFault",
    deep_faults=["DeepFault"],
    component="strikeslip",
    zero_tolerance=0.01,  # 与滑动速率同单位；按数据量级调整
)

inversion.print_deep_slip_loading_report(
    result["evaluation_mapping"],
    result=result,
)

inversion.plot_deep_slip_loading_field(
    "coupling_to_deep",
    result=result,
    cblabel="Coupling to deep slip",
    norm=[0, 1],
)
```

`zero_tolerance` 只控制比值分母过小时的数值稳定性；它不会改变深部或浅部反演
滑动本身。`norm=[0, 1]` 只用于展示 0--1 coupling 区间；正式解释前仍应报告
是否存在 `K < 0`、`K > 1` 或 near-zero deep loading。若没有添加 deep proxy
约束，这个字段只是相对自由深部滑动的诊断，不应被解读为已经强制满足闭锁率边界。

Deep proxy 的 patch GMT 默认会让一个文件携带更多信息：

```text
> -Z<field_value> # <field_value> <shallow_slip_rate> <deep_loading_proxy_rate>
```

其中 `-Z` 仍是 GMT 着色使用的字段，例如 `coupling_to_deep`。`#` 后三列复用
CSI patch GMT 的 slip 注释位置，但在 deep proxy 文件中它们不是物理
`strikeslip/dipslip/tensile` 分量，而是：

| 注释列 | deep proxy 含义 |
| --- | --- |
| 第 1 列 | 当前写出的字段值，通常和 `-Z` 相同 |
| 第 2 列 | `shallow_slip_rate = s` |
| 第 3 列 | `deep_loading_proxy_rate = b` |

这样一个闭锁率 GMT 同时保留了计算闭锁率所需的浅部滑动和深部加载值。若需要
完全沿用 CSI 对自定义 `add_slip` 的旧行为，即 `# field_value 0 0`，可设置：

```python
inversion.write_deep_slip_loading_field_gmt(
    "coupling_to_deep",
    "output/ShallowFault_coupling_to_deep.gmt",
    result=result,
    include_proxy_comment_columns=False,
)
```

批量导出：

```python
inversion.export_deep_slip_loading_results(
    "output/deep_proxy_ShallowFault",
    result=result,
)
```

## 与分段场景的关系

如果只需要按地表迹线粗略分段，不必引入完整 Blocks/celeri 拓扑。优先复用 ECAT selector：

```python
mapping = inversion.preview_deep_slip_loading_mapping(
    shallow_fault="ShallowFault",
    deep_faults=["DeepNorth"],
    shallow_selector={
        "trace_range": {
            "point1": [100.0, 25.0],
            "point2": [101.0, 24.5],
            "buffer_distance": 20.0,
        },
        "depth_range": [15.0, 25.0],
    },
    component="strikeslip",
)
```

如果这种分段关系成为常用需求，再考虑增加轻量 `link_regions` 配置层。不要把它命名成 `blocks` 或 `segment_pair_table`，以免和完整块体拓扑模型混淆。

## 检查清单

- 先确认 deep fault 的滑动变量确实代表长期加载代理，而不是普通同震破裂 slip。
- 优先用 `{"edge": "bottom"}` 约束浅部底边；只有边界识别不可用时才用底部深度带。
- 运行 `print_deep_slip_loading_report(mapping)`，检查映射距离和 unique deep patch 数。
- 明确 `strikeslip` 或 `dipslip` 分量；浅部和深部必须使用同一分量和同一符号约定。
- 若启用 `cap`，必须明确 `motion_sense` 或 `motion_sign`，并确认这确实是研究假设。
- 输出时使用 `coupling_to_deep`，不要把它和 Euler/block direct-backslip 的 `coupling_ratio` 混写。

## 相关页面

- [震间加载、Backslip 与 Coupling](interseismic_kinematics.md)
- [ECAT 约束管理器](constraint_manager.md)
- [Fault Patch Indices](fault_patch_indices.md)
- [BLSE/VCE 线性滑动反演](../workflows/04_linear_slip_blse_vce.md)
