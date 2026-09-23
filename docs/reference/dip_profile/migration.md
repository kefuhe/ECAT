# 旧倾角剖面调用迁移

当前非分层 Bayesian 倾角扰动以 `DipProfileSpec` 为单一事实源。旧 setter、索引式固定节点、
buffer 参数和 preset 中重复的轴设置不保留兼容字段；迁移应在 setup 层一次完成。

本页的“等价替换”首先表示角色、参数顺序和候选生成职责等价。只有满足下文列出的条件时，
才可以预期浮点精度范围内的逐点几何等价；涉及离断层控制点或旧 buffer 的脚本必须用诊断图
和零扰动结果重新验收，不能只按名称机械替换。

## 逐项替换

| 旧写法 | 当前写法 | 迁移要点 |
| --- | --- | --- |
| `set_xy_dip_ref(x, y, dip)` | `set_dip_profile(sampled_controls=...)` | 把三列合成 controls |
| `set_xy_dip_ref_from_coords(coords, dips)` | `set_dip_profile(sampled_controls=np.column_stack(...))` | 保持原行顺序 |
| `set_xy_dip_ref_from_file(path)` | 先读表，再调用 `set_dip_profile(...)` | 当前 setter 不隐式读取文件 |
| `set_dip_control_points(x, y, dip)` | `set_dip_profile(sampled_controls=np.column_stack(...))` | 原 controls 默认全为 sampled |
| `set_dip_control_points_from_coords(coords, dips)` | `set_dip_profile(sampled_controls=np.column_stack([coords, dips]))` | 保持原行顺序 |
| `set_dip_control_points_from_file(path)` | 先读表，再调用 `set_dip_profile(...)` | 当前 setter 不隐式读取文件 |
| `fixed_nodes=[...]` | 把对应行移入 `fixed_controls` | fixed 不再用整数哨兵或索引解释 |
| `buffer_nodes + buffer_radius` | `transition_zones` | 明确选择 `axis` 或 `euclidean` 距离语义 |
| preset 中的 `interpolation_axis` | `set_dip_profile(interpolation_axis=...)` | profile 冻结后不在 YAML 重复 |
| preset 中的 `is_utm` | `set_dip_profile(is_utm=...)` | 坐标系只属于 setup 输入边界 |
| `update_xydip_ref=True` | 显式 `set_dip_profile()`；生成型 profile 不 snapshot 派生 bottom | 候选过程不改写 top/profile 基线 |
| `update_dip_baseline()` | `set_dip_profile()` 或高级 `refresh_geometry_baseline()` | setter 只换 profile 并保留已有 frozen top；后者才有意重设完整 current top/bottom 基线 |
| `prepare_for_inversion(..., dip_control_coords=..., dip_control_dips=...)` | `prepare_for_inversion(..., dip_sampled_controls=...)` | controls 改为统一三列表 |

## 哪些替换可以逐点对齐

以下条件同时成立时，旧 x/y 插值与新 profile 的零扰动结果应当只剩浮点舍入差异：

- controls 本来就在 top 上，或投影后所选 x/y 轴坐标不变；
- 仍使用同一个显式 `interpolation_axis="x"` 或 `"y"`；
- 没有 `fixed_nodes`、`buffer_nodes` 或其他会改变控制序列的旧参数；
- top、深度、走向模式、加密和 mesh policy 没有变化；
- sampled controls 的声明顺序与旧可动参数顺序一致。

新协议会先把所有位置投影到 top，再求统一一维坐标。这是有意收紧的空间语义。因此，离 top
较远的旧控制点、`auto` 轴选择以及任何 buffer 迁移都只能称为科学意图等价，不能预先承诺
逐点数值相等。迁移后应比较 resolved dip、bottom、mesh/patch 点序和一个非零候选，而不只
比较最终图形。

## 全部控制点都参与采样

旧写法：

```python
fault.set_xy_dip_ref(x, y, dip)
```

当前写法：

```python
import numpy as np

controls = np.column_stack([x, y, dip])
fault.set_dip_profile(
    sampled_controls=controls,
    fixed_controls=None,
    interpolation_axis="arc_length",
    is_utm=False,
)
```

如果旧坐标是 fault-local x/y km，改为 `is_utm=True`。

## 有固定节点

假设旧 controls 为 `controls`，旧 `fixed_nodes=[0, 3]`：

```python
fixed_index = np.array([0, 3], dtype=int)
sampled_mask = np.ones(len(controls), dtype=bool)
sampled_mask[fixed_index] = False

fault.set_dip_profile(
    sampled_controls=controls[sampled_mask],
    fixed_controls=controls[~sampled_mask],
    interpolation_axis="arc_length",
    is_utm=False,
)
```

迁移时必须保持 `sampled_controls` 中原可移动节点的相对顺序，因为它就是候选向量和 bounds
顺序。fixed controls 的空间位置会在 resolver 中重新排序，不要求和 sampled 交错输入。

相应地，`sample_positions` 长度从“旧控制点总数或可移动索引规则”改为：

```text
1（广播）或 sampled_controls 的数量
```

全部 fixed 时必须使用空切片。

## 旧 buffer 到 transition

旧的单个中心和对称半径通常迁移为：

```python
transition_zones=[
    {
        "center": [lon_buffer, lat_buffer],
        "half_width": buffer_radius,
        "metric": "axis",
    },
]
```

若原研究意图明确是平面圆形半径，使用 `metric="euclidean"`。两者不是可以随意互换的
数值选项：新协议会先把中心投影到 top，`axis` 再沿 resolved u 量宽度，`euclidean` 则寻找
圆与同一 top 分支的交点。

旧 buffer 不能机械批量替换，至少检查：

- 它应落在哪一对相邻 controls 之间；
- 多个 buffer 是否落入同一 control pair；
- 旧 x/y 半径究竟表示轴向距离还是平面距离；
- 迁移后的投影点、transition endpoints 和最终 dip 曲线是否符合原科学意图。

旧实现只允许 buffer 配合最终解析为 x/y 的轴，并按旧的平面搜索规则补控制点；它不支持
`arc_length`。所以旧 `buffer_nodes + buffer_radius` 与新 `transition_zones` 不是严格数值别名。
如果旧脚本本来依赖 x/y 行为，可先用 `interpolation_axis="x"|"y"` 和
`metric="euclidean"` 做对照；确认研究意图其实是“沿断层过渡宽度”后，再改为
`interpolation_axis="arc_length"`、`metric="axis"`。

## 旧直接调用

旧直接入口把 x、y、dip 拆成三个位置参数：

```python
fault.perturb_dips(
    x_coords,
    y_coords,
    dips,
    perturbations,
    fixed_nodes=fixed_nodes,
    interpolation_axis="x",
    is_utm=False,
)
```

当前入口接收与 setup 相同的三列分组：

```python
import numpy as np

controls = np.column_stack([x_coords, y_coords, dips])
fixed_index = np.asarray(
    [] if fixed_nodes is None else fixed_nodes,
    dtype=int,
)
sampled_mask = np.ones(len(controls), dtype=bool)
sampled_mask[fixed_index] = False

fault.perturb_dips(
    sampled_controls=controls[sampled_mask],
    fixed_controls=controls[~sampled_mask],
    perturbations=perturbations,
    interpolation_axis="x",
    is_utm=False,
)
```

该直接入口只构造临时 profile，不改写冻结参考；Bayesian 候选应优先采用一次
`set_dip_profile()` 加多次 `perturb_dips_with_preset_params()`。

## preset 与 SimpleMesh 调用

`perturb_dips_with_preset_params()` 和 `perturb_DipsPresetParams_SimpleMesh()` 的方法名保留，
但 `interpolation_axis`、`fixed_nodes`、`buffer_nodes`、`buffer_radius` 与 `is_utm` 已从候选
调用层移除。这些字段都应在前置 `set_dip_profile()` 中声明。候选调用只保留：

- `perturbations` 与其 `angle_unit`；
- bottom 走向策略；
- 可选加密间隔；
- SimpleMesh 入口自身的 `disct_z/bias/min_dz`。

这不是删除能力，而是避免 setup 和每个候选各保存一份可能互相矛盾的 profile。

## 文件输入

```python
import pandas as pd

table = pd.read_csv("dip_controls.csv")
sampled_controls = table.loc[:, ["lon", "lat", "dip"]].to_numpy()

fault.set_dip_profile(
    sampled_controls=sampled_controls,
    interpolation_axis="arc_length",
    is_utm=False,
)
```

若文件还用一列标识 fixed/sample，应先按该列拆成两个数组；不要把角色列传入三列 controls。

## preset 配置前后

旧 YAML 可能同时保存 profile 信息：

```yaml
update_fault_geometry:
  method: perturb_dips_with_preset_params
  interpolation_axis: x
  fixed_nodes: [0]
  buffer_nodes: [[100.2, 30.1]]
  buffer_radius: 5.0
  is_utm: false
```

当前 YAML 只保存候选生成选项：

```yaml
update_fault_geometry:
  method: perturb_dips_with_preset_params
  angle_unit: degrees
  use_average_strike: false
```

controls、fixed 角色、axis、transition 和可选 `perturbation_groups` 已由 Python 中的
`set_dip_profile()` 冻结。这样采样准备阶段只有一个 profile 定义，不会出现 Python 与 YAML
各保存一套但彼此错位。

## 迁移验收

1. 零扰动 bottom 的点数、点序、深度和倾向侧符合预期。
2. 未分组时 `sample_positions` 为一个广播值或与 sampled controls 数量一致；显式分组时与
   唯一标签数一致。bounds 按 sampled 声明顺序或 group 首次出现顺序逐项核对。
3. 用 `plot_dip_profile_diagnostics()` 核对原始位置、投影位置、S/F 角色和 transition。
4. 对一个手工非零候选逐控制点核对绝对 dip。
5. 固定拓扑流程再核对 patch 数量、Faces 和参数位置；重网格流程则重新计算依赖几何的
   GF、Laplacian 和约束。

分层倾角类及普通非 Bayesian 构模接口的同名 buffer 参数不属于这次迁移，不能按本页直接
替换。
