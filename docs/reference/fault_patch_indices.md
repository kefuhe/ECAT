# Fault Patch Indices / 断层 Patch 子集

本页说明如何在脚本中生成、校验和复用断层 patch id。Patch id 是运行时的中间结果，常用于边界零滑、局部零滑、震间 Euler cap、震间 backslip/coupling 约束、分段统计和质量检查。

## 阅读路径

| 当前问题 | 建议阅读顺序 |
| --- | --- |
| 第一次选择 patch 子集 | [使用原则](#使用原则) → [基本校验](#基本校验) → [统一 Selector](#统一-selector) |
| 按边界、深度或矩形范围选择 | [按边界取 Patch](#按边界取-patch) → [按深度或空间范围取 Patch](#按深度或空间范围取-patch) |
| 按 trace 分段选择 | [按 Trace 段取 Patch](#按-trace-段取-patch) → [Trace marker](#trace-marker用经度纬度或真实沿迹线距离定义端点) |
| 把选择结果交给震间或普通约束 | [与震间约束配合](#与震间约束配合) → [约束管理器](constraint_manager.md) |
| 选择完成后复查 | [复查清单](#复查清单) |

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
    get_patches_in_trace_segment,
    get_patches_in_trace_range,
    normalize_patch_indices,
    resolve_trace_marker,
    sample_trace_markers,
    select_patch_indices,
    trace_range_selector_from_markers,
    update_fault_loading_override_from_trace_segment,
    update_fault_motion_sense_override_from_trace_segment,
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

`select_patch_indices(fault, selector)` 是常用选择方式的统一入口。它返回最终 patch id，不生成任何物理约束。selector 可直接传给零滑、震间 cap、震间 backslip/coupling、统计和绘图接口。

```python
top_ids = select_patch_indices(fault, {"edge": "top"})
shallow_ids = select_patch_indices(fault, {"depth_range": [0.0, 10.0]})
manual_ids = select_patch_indices(fault, {"patches": [0, 1, 2]})
```

### Selector Cookbook

下面这些写法都可以直接套用。除显式 patch id 外，选择均基于 patch center。

#### 1. 全部 Patch

```yaml
selector: null
```

`null` 只有在调用方允许“全部 patch”时有效，例如 cap selector 或某些统计接口。

#### 2. 显式 Patch ID

```yaml
selector:
  patches: [0, 1, 2]
```

等价低层写法：

```yaml
selector:
  patch_indices: [0, 1, 2]
```

也可以在脚本中直接传 `[0, 1, 2]`。如果 mesh 重新生成，旧 patch id 不应继续复用。

#### 3. 单个边界

```yaml
selector:
  edge: top
```

常用边界名为 `top`、`bottom`、`left`、`right`。边界选择依赖当前 fault 的 `edge_triangles_indices`，通常需要先运行边界识别。

#### 4. 多个边界

```yaml
selector:
  edges: [top, bottom]
```

适合边界零滑、顶部/底部 hard constraint 或边界统计。

#### 5. 深度范围

```yaml
selector:
  depth_range: [0, 25]
```

深度单位通常为 km，按 patch center depth 判断。若 `zmin > zmax`，内部会自动排序。

#### 6. 经纬度范围

```yaml
selector:
  box:
    lon_range: [101.0, 102.0]
    lat_range: [23.0, 24.0]
```

#### 7. 本地 XY 范围

```yaml
selector:
  box:
    x_range: [0, 100]
    y_range: [20, 80]
```

本地 `x/y` 使用 fault 对象中的投影坐标，单位通常为 km。

#### 8. 沿 Trace 选段：推荐写法

对同一条 fault 做震间 `loading_overrides` 或 `motion_sense_overrides` 分段时，
推荐使用 `trace_segment`。这个名字直接表达“用 start/end 定义一段 trace”：

```yaml
selector:
  trace_segment:
    start: {longitude: 101.8}
    end: {longitude: 102.4}
  depth_range: [0, 30]
```

`start` 和 `end` 支持灵活 trace marker：

```yaml
selector:
  trace_segment:
    start: {point: [101.8, 23.7], coord_system: lonlat}
    end: {trace_distance_km: 180.0}
  depth_range: [0, 30]
```

所有 marker 都会先解析成 fault trace 上的真实点，再按 along-trace 区间选择 patch。

#### 9. 沿 Trace 选段：低层/兼容写法

`trace_range` 是较低层的 selector 名称，表示“两个端点投影到 trace 后形成的 along-trace 范围”。它适合已经有明确 `point1/point2` 的场景：

```yaml
selector:
  trace_range:
    point1: [101.8, 23.7]
    point2: [102.4, 23.2]
    coord_system: lonlat
  depth_range: [0, 30]
```

若端点写成 marker mapping，`trace_range` 内部也会调用 `trace_segment` 逻辑：

```yaml
selector:
  trace_range:
    start: {longitude: 101.8}
    end: {longitude: 102.4}
  depth_range: [0, 30]
```

新项目优先使用 `trace_segment`；`trace_range` 主要用于兼容旧配置或明确提供两个端点的
低层调用。

### 组合规则

`depth_range` 可与 `edge`、`trace_range`、`trace_segment` 或 `box` 组合，用于进一步限制 patch center 深度。推荐把 `depth_range` 写在 selector 顶层，便于所有 selector 形式保持一致：

```yaml
selector:
  trace_segment:
    start: {longitude: 101.8}
    end: {longitude: 102.4}
  depth_range: [0, 30]
```

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

沿断层迹线两端点之间选择 patch，可用于震间分段、局部 cap、分辨率测试或分段统计。
默认逻辑是把 patch center 投影到 fault trace，然后按真实 along-trace 距离判断是否落在端点之间：

```python
patch_ids = get_patches_in_trace_segment(
    fault,
    {"point": [100.25, 25.57], "coord_system": "lonlat"},
    {"point": [101.80, 23.80], "coord_system": "lonlat"},
    depth_range=(0.0, 25.0),
)
```

推荐 selector：

```python
patch_ids = select_patch_indices(
    fault,
    {
        "trace_segment": {
            "start": {"point": [100.25, 25.57], "coord_system": "lonlat"},
            "end": {"point": [101.80, 23.80], "coord_system": "lonlat"},
        },
        "depth_range": (0.0, 25.0),
    },
)
```

如果端点已经是普通坐标点，也可以使用低层 `trace_range`：

```python
patch_ids = select_patch_indices(
    fault,
    {
        "trace_range": {
            "point1": (100.25, 25.57),
            "point2": (101.80, 23.80),
            "coord_system": "lonlat",
        },
        "depth_range": (0.0, 25.0),
    },
)
```

算法步骤：

```text
start/end 或 point1/point2 投影到 fault trace
patch center 投影到 fault trace
按 along-trace 位置和 depth_range 过滤
```

对同一条 fault 的 `fault_loading.loading_overrides` 和
`fault_loading.motion_sense_overrides`，通常不需要 `buffer_distance`：
patch 本来就属于该 fault，分段归属应由 along-trace 区间决定。`buffer_distance`
仍可用于更一般的空间带状筛选，例如从多个对象中只保留离某条 trace 足够近的 patch。
这个方法适合快速、可解释的段落选择，但不是正式的断层分段模型。若后续形成稳定分段参数化，再设计更高层接口。

### Trace marker：用经度、纬度或真实沿迹线距离定义端点

当需要循环测试某个分段边界位置时，端点不一定正好落在已有 trace 顶点上。此时使用 trace marker，让 ECAT 先把用户给定的经度、纬度、最近点或沿迹线距离解析成真实 trace 上的投影点：

```python
start = resolve_trace_marker(fault, {"longitude": 101.5})
end = resolve_trace_marker(fault, {"longitude": 102.5})

print(start.trace_distance_km, start.lon, start.lat)
```

支持的常用 marker：

| Marker | 含义 |
| --- | --- |
| `{"longitude": 101.5}` | trace 与指定经度线的交点 |
| `{"latitude": 24.0}` | trace 与指定纬度线的交点 |
| `{"point": [lon, lat], "coord_system": "lonlat"}` | 将点投影到最近的 trace 位置，不吸附到最近顶点 |
| `{"xy": [x, y]}` | 将本地坐标点投影到最近 trace 位置 |
| `{"trace_distance_km": 80.0}` | 从 trace 第一个点起算的真实沿迹线距离 |
| `{"fraction": 0.5}` | trace 总长度的比例位置 |

如果一条弯曲 trace 与同一经度或纬度有多个交点，默认使用沿迹线方向的第一个交点；可加 `which: "last"` 或整数索引显式指定。所有 `trace_distance_km` 都是本地投影 `x/y` 平面中的真实折线弧长，单位通常为 km；不要理解为 GPS station distance，也不要理解为单独沿经度或纬度的距离。

按真实迹线距离采样：

```python
markers = sample_trace_markers(
    fault,
    {"longitude": 101.5},
    {"longitude": 102.5},
    step_km=5.0,
)

for marker in markers:
    print(marker.trace_distance_km, marker.lon, marker.lat)
```

用 marker 直接选 patch：

```python
patch_ids = get_patches_in_trace_segment(
    fault,
    {"longitude": 101.5},
    {"longitude": 102.5},
    depth_range=(0.0, 25.0),
)
```

也可以生成标准 selector，继续交给统一入口：

```python
selector = trace_range_selector_from_markers(
    fault,
    {"longitude": 101.5},
    {"longitude": 102.5},
    depth_range=(0.0, 25.0),
)

patch_ids = select_patch_indices(fault, selector)
```

### 在循环脚本中更新震间 override

对于搜索走滑方向转换点、块体对分界点或局部 cap 范围这类问题，推荐在脚本中循环
marker，然后更新对应 override 的 selector。不要把循环维度写进
`interseismic_config.yml`。

如果测试的是 block pair 分界或局部 reference strike，更新 `loading_overrides`：

```python
base_config = inversion.config.interseismic_config

markers = sample_trace_markers(
    hh_main,
    {"longitude": 101.5},
    {"longitude": 102.5},
    step_km=5.0,
)

for marker in markers:
    trial_config, meta = update_fault_loading_override_from_trace_segment(
        base_config,
        hh_main,
        "HH_Main",
        "north_segment",
        marker,
        {"longitude": 102.5},
        depth_range=(0.0, 25.0),
        return_metadata=True,
    )
    inversion.update_interseismic_config(trial_config, reapply=True)
```

如果测试的是局部 dextral/sinistral 转换，而 block pair 不变，更新
`motion_sense_overrides`：

```python
trial_config, meta = update_fault_motion_sense_override_from_trace_segment(
    base_config,
    hh_main,
    "HH_Main",
    "trial_sinistral_zone",
    marker,
    {"longitude": 102.5},
    depth_range=(0.0, 25.0),
    return_metadata=True,
)
```

这一步只更新 override 的 selector。`blocks`、`fault_loading`、`cap_constraints`
和 `backslip_constraints` 的物理含义不变；求解器仍只看到标准配置。

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
