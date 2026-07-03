# Fault Patch Indices / 断层 Patch 子集

本页说明如何在脚本中生成、校验和复用断层 patch id。Patch id 是运行时的中间结果，常用于边界零滑、局部零滑、震间 Euler cap、震间 backslip/coupling 约束、分段统计和质量检查。

## 使用原则

| 原则 | 含义 |
| --- | --- |
| helper 只负责选 patch | 不生成约束矩阵，不修改 `fault.slip`，不改变 mesh |
| 约束接口只消费 selector 或 id | 零滑、震间约束和自定义矩阵只关心最终 patch 子集 |
| 动态选择留在脚本 | 按 trace 段、深度、边界或数据质量选择时，脚本更容易审查和复现 |
| YAML 保持克制 | 不把高度可变的大量 patch id 过早固定进主配置 |

常用 helper：

```python
from eqtools.csiExtend import (
    get_edge_patch_indices,
    get_patches_by_depth,
    get_patches_in_box,
    get_patches_in_trace_range,
    normalize_patch_indices,
    select_patch_indices,
)
```

## 基本校验

`normalize_patch_indices(...)` 将 list、NumPy array 或单个整数转换为一维整数数组，并检查负数和越界：

```python
patch_ids = normalize_patch_indices(fault, [0, 2, 4], allow_none_all=False)
```

传入 `None` 且 `allow_none_all=True` 时，返回全部 patch：

```python
all_ids = normalize_patch_indices(fault, None)
```

这个函数适合在自定义脚本或高级接口内部使用，避免每个地方手写 `[int(i) for i in patch_ids]` 和越界检查。

## 统一 Selector

`select_patch_indices(fault, selector)` 是常用选择方式的统一入口。它返回最终 patch id，不生成任何物理约束。

```python
top_ids = select_patch_indices(fault, {"edge": "top"})
shallow_ids = select_patch_indices(fault, {"depth_range": [0.0, 10.0]})
manual_ids = select_patch_indices(fault, {"patches": [0, 1, 2]})
```

支持的 selector：

| Selector | 含义 |
| --- | --- |
| `None` | 全部 patch，前提是调用方允许 |
| `[0, 1, 2]` | 显式 patch id |
| `{"patches": [...]}` | 显式 patch id |
| `{"patch_indices": [...]}` | 显式 patch id |
| `{"edge": "top"}` | 一个边界 |
| `{"edges": ["top", "bottom"]}` | 多个边界 |
| `{"depth_range": [zmin, zmax]}` | 按 patch center 深度 |
| `{"box": {"lon_range": [...], "lat_range": [...]}}` | 按经纬度框 |
| `{"box": {"x_range": [...], "y_range": [...]}}` | 按本地坐标框 |
| `{"trace_range": {...}}` | 按 trace 段 |

`depth_range` 可与 `edge`、`trace_range` 或 `box` 组合，用于进一步限制 patch center 深度。

## 按边界取 Patch

边界零滑、顶部 full coupling 和底部 creep 通常依赖 `edge_triangles_indices`。先完成边界识别：

```python
fault.find_fault_fouredge_vertices(
    edge_method="topology",
    gap_policy="clean",
)
```

再提取边界 patch：

```python
top_ids = get_edge_patch_indices(fault, "top")
side_ids = get_edge_patch_indices(fault, ["left", "right"])
```

统一 selector 写法：

```python
top_ids = select_patch_indices(fault, {"edge": "top"})
```

这些 id 可传给普通零滑约束：

```python
inversion.add_patch_slip_constraint(
    {"MyFault": top_ids},
    slip_component=["ss", "ds"],
    value=0.0,
    constraint_type="equality",
)
```

或传给震间 backslip/coupling 约束：

```python
inversion.add_interseismic_backslip_constraint(
    "MyFault",
    state="full_coupling",
    selector={"edge": "top"},
)
```

若只是固定整条边界为零滑，公开配置仍推荐使用更清晰的 `source_constraints`：

```yaml
source_constraints:
  MyFault:
    - {name: zero_top_ss, type: equality, rule: "zero_edge_slip(top, ss)"}
```

## 按深度或空间范围取 Patch

按中心深度：

```python
shallow_ids = get_patches_by_depth(fault, (0.0, 15.0))
```

按本地 `x/y` 范围和深度：

```python
box_ids = get_patches_in_box(
    fault,
    x_range=(0.0, 40.0),
    y_range=(-10.0, 20.0),
    depth_range=(0.0, 20.0),
)
```

按经纬度范围：

```python
box_ids = get_patches_in_box(
    fault,
    lon_range=(100.0, 101.0),
    lat_range=(24.0, 25.0),
)
```

统一 selector：

```python
box_ids = select_patch_indices(
    fault,
    {"box": {"lon_range": [100.0, 101.0], "lat_range": [24.0, 25.0]}},
)
```

这些选择都基于 patch center。若科学问题要求 patch polygon 与区域精确相交，应在脚本中显式实现并记录算法。

## 按 Trace 段取 Patch

沿断层迹线两点之间选择 patch，可用于局部 cap、分辨率测试或分段统计：

```python
patch_ids = get_patches_in_trace_range(
    fault,
    point1=(100.25, 25.57),
    point2=(101.80, 23.80),
    buffer_distance=30.0,
    depth_range=(0.0, 25.0),
    coord_system="lonlat",
)
```

统一 selector：

```python
patch_ids = select_patch_indices(
    fault,
    {
        "trace_range": {
            "point1": (100.25, 25.57),
            "point2": (101.80, 23.80),
            "buffer_distance": 30.0,
            "coord_system": "lonlat",
        },
        "depth_range": (0.0, 25.0),
    },
)
```

算法步骤：

```text
point1/point2 投影到 fault trace
patch center 投影到 fault trace
按 along-trace 位置、可选 buffer_distance 和 depth_range 过滤
```

这个方法适合快速、可解释的段落选择，但不是正式的断层分段模型。若后续形成稳定分段参数化，再设计更高层接口。

## 与震间约束配合

`fault_loading` 会在所有 patch 上计算 loading。若只想对局部 patch 使用 coupling cap，使用 cap selector；默认 `motion_sense` 模式还需要普通 bounds 给 `q` 设置基础符号，固定 loading 场景也可用 `mode="loading_sign"` 直接约束区间：

```python
cap_ids = get_patches_in_trace_range(
    fault,
    point1=(100.25, 25.57),
    point2=(101.80, 23.80),
    buffer_distance=30.0,
    depth_range=(0.0, 25.0),
)

inversion.update_euler_cap_constraint(
    "MyFault",
    selector={"patches": cap_ids},
    mode="motion_sense",
    enabled=True,
)
```

如果只是让浅部自由估计，不要把浅部从 `fault_loading` 中移除；保持 cap disabled 或不选这些 patch 即可。构造加载率仍应由两个块体照常投影到所有 patch。

`add_interseismic_backslip_constraint(...)` 可直接接收 selector：

```python
inversion.add_interseismic_backslip_constraint(
    "MyFault",
    state="creep",
    selector={"edge": "bottom"},
)

inversion.add_interseismic_backslip_constraint(
    "MyFault",
    state="prescribed_coupling",
    selector={"depth_range": [0.0, 8.0]},
    coupling=1.0,
)
```

## 复查清单

- 选中的 patch 数是否符合预期。
- 深度范围和 along-trace 范围是否覆盖目标段。
- 边界 id 是否来自当前 mesh 的 `edge_triangles_indices`。
- 如果 mesh 拓扑变化，旧 patch id 不应继续复用。
- 若只是调整震间 cap 范围，使用 `update_euler_cap_constraint(..., reapply=True)`；不要修改 `blocks` 或 `fault_loading`。

## 相关页面

- [ECAT 约束管理器](constraint_manager.md)
- [震间加载、Backslip 与 Coupling](interseismic_kinematics.md)
- [断层边界识别](fault_edges.md)
- [Fault Geometry Construction](fault_geometry_construction.md)
