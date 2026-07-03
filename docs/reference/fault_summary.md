# Fault Summary / 断层概览和统计

本页说明如何在构建断层、读入 GMT/mesh 或完成滑动反演后，快速查看 fault object 的一般信息。它适合回答：

- 迹线长度是多少？
- patch/mesh 数量是否符合预期？
- 断层深度范围、面积、平均走向和平均倾角是否合理？
- 如果已有滑动量，最大滑动、地震矩和 Mw 是多少？

如果还没有构建 fault object，先读 [Fault Geometry Construction](fault_geometry_construction.md)。如果正在检查边界识别，读 [Fault Edges](fault_edges.md)。如果要整理 BLSE/VCE 报告字段，结合 [BLSE/VCE 参考](blse_vce.md#recommended-reporting) 使用本页。

## 最小用法

单个断层：

```python
from eqtools.csiExtend.fault_summary import summarize_fault, print_fault_summary

summary = summarize_fault(fault)
print_fault_summary(fault)
```

多个断层：

```python
from eqtools.csiExtend.fault_summary import summarize_faults, print_faults_summary

summary = summarize_faults([fault1, fault2])
print_faults_summary([fault1, fault2])
```

反演对象：

```python
summary = inv.get_faults_summary()
inv.print_faults_summary()
```

`BoundLSEMultiFaultsInversion` 和 `BayesianMultiFaultsInversion` 的结果提取流程中，`print_fault_statistics=True` 会继续打印断层概览；底层调用同一套 summary 逻辑。

## 返回内容

`summarize_fault(fault)` 返回一个 dict。主要字段如下：

| 字段 | 内容 |
| --- | --- |
| `name` | 断层名 |
| `class_name` | fault object 类名 |
| `patch_type` | `triangular`、`rectangular` 或原始类型名 |
| `basic` | trace/slip 是否存在、`lon0/lat0/utmzone` 等基本信息 |
| `trace` | 原始 trace 点数、迹线长度、离散 trace 点数和长度 |
| `bounds` | 已有 vertices 或 patches 的 `x/y/depth` 范围，若有 lon/lat 顶点也会报告 |
| `mesh` | patch 数、vertex 数、face 数、面积统计；矩形元还会报告 patch length/width 统计 |
| `orientation` | strike/dip 的 mean、median、min、max 和 std |
| `slip` | strike-slip、dip-slip、total slip 等统计；无 slip 时为 `None` |
| `moment` | seismic moment、Mw、剪切模量和使用的 patch 数；无面积或 slip 时为 `None` |
| `warnings` | 缺失字段、无法计算面积或数量不一致等提示 |

多个断层时，`summarize_faults(...)` 返回：

```python
{
    "faults": [...],   # 每个断层的 summarize_fault 结果
    "groups": [...],   # fault_groups 对应的事件组 moment/Mw
    "total": {...},    # 总 patch 数、总面积、总 moment/Mw
}
```

## 输出示例

```text
Fault Summary
=============

Geometry
+-----------+-------------+---------+--------------+------------+-------------+-------------+----------+
| Fault     | Patch Type  | Patches | Trace Length | Area       | Depth Range | Mean Strike | Mean Dip |
+-----------+-------------+---------+--------------+------------+-------------+-------------+----------+
| MainFault | triangular  | 2846    | 186.42 km    | 4217.35 km^2 | 0.00 - 25.00 km | 101.40 deg | 42.70 deg |
+-----------+-------------+---------+--------------+------------+-------------+-------------+----------+

Slip
...

Moment
...
```

实际列宽由 `tabulate` 控制；如果环境中没有 `tabulate`，会退化成简单文本表。

## 矩形元和三角元的区别

矩形元的 `getpatchgeometry()` 中，patch `length` 和 `width` 有直接几何意义，因此 summary 会报告矩形 patch 的 length/width 统计。

三角元的局部边长可以用于内部几何计算，但不建议把每个三角形的某条边解释为物理“断层长度”或“断层宽度”。三角断层概览中优先使用：

- 迹线长度表示走向长度；
- 三角面片面积求和表示总面积；
- `Vertices/Faces` 表示 mesh 规模；
- `getStrikes()` 和 `getDips()` 表示面片方向统计。

## Moment 和分组

默认剪切模量为 `3.0e10 Pa`：

```python
summary = summarize_fault(fault, mu=3.2e10)
```

如果滑动量需要缩放：

```python
summary = summarize_fault(fault, slip_factor=0.01)
```

`summarize_fault()` 这类纯函数不读取反演配置，`slip_factor` 仍由用户显式给出。反演对象上的 `inv.get_faults_summary()` 和 `inv.print_faults_summary()` 会优先读取主配置 `units.observation`：若为 `m`、`cm` 或 `mm`，会自动换算到米后计算 `Mo` 和 `Mw`；若为 `m/yr`、`cm/yr` 或 `mm/yr`，slip 表保留为 slip-rate 统计，moment 表改为矩率 `N*m/yr`，并给出 1 年等效的 `Mw equiv. (1 yr)` 作为量级参考。直接调用 `inv.calculate_moment_magnitude()` 时，速率单位仍会报错，除非用户显式传入 `slip_factor` 或先把速率转换为累计滑动。

注意：`Mw equiv. (1 yr)` 不是一次地震事件的矩震级，只是把当前矩率乘以 1 年后代入标准 `Mw` 公式得到的量级标签。震间速率反演、长期深部蠕滑或负位错模型中，应优先报告矩率和 slip-rate；同震或累计余滑模型中才报告普通 `Mo/Mw`。

多个断层可以按事件分组计算 moment/Mw：

```python
summary = summarize_faults(
    [fault1, fault2, fault3],
    fault_groups=[
        ["fault1", "fault2"],
        ["fault3"],
    ],
)
```

反演对象若配置中已有 `alpha.faults` 分组，`inv.get_faults_summary()` 会优先使用该分组；否则只报告每条断层的 moment 或 moment-rate 以及所有断层的 TOTAL，不额外生成事件组。

## 常见检查顺序

建完断层后建议先检查：

```python
print_fault_summary(fault)
```

确认以下内容：

1. `Trace Length` 是否和地表迹线或预期长度同量级。
2. `Patches`、`Vertices`、`Faces` 是否和网格设置一致。
3. `Depth Range` 是否符合 `top/depth`，没有出现符号反转。
4. `Area` 是否非零且数量级合理。
5. `Mean Strike` 和 `Mean Dip` 是否符合构建参数或输入等深线趋势。

完成线性滑动反演后，再检查：

1. total slip 的最大值和均值是否合理。
2. Moment/Mw 是否和目标事件量级一致。
3. 多断层时 `fault_groups` 是否代表真实事件分组。

## 相关页面

- [Fault Geometry Construction](fault_geometry_construction.md)
- [Fault Edges](fault_edges.md)
- [Fault Contours](fault_contours.md)
- [BLSE/VCE 参考](blse_vce.md)
- [Bayesian 联合反演参考](bayesian_joint_inversion.md)
