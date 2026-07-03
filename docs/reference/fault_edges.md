# 断层边界识别参考 / Fault Edge Reference

本页说明三角断层对象中的 `top`、`bottom`、`left`、`right` 四条边界，以及 `edge_vertices` 和 `edge_triangles_indices` 等字段的用途。边界识别通常在写出边界 GMT、做边界零滑约束、统计边界 patch 或绘制三维断层时使用。

## 四条边的约定

对一个单独的三角断层段，边界识别假定断层面是单连通、无孔洞的三角网格。四条边的含义是：

| 边名 | 含义 |
| --- | --- |
| `top` | 浅部边界，通常接近自由面或给定 top 深度。 |
| `bottom` | 深部边界，通常接近给定 bottom 深度。 |
| `left` | 沿走向一端。优先按 top/bottom 主方向投影的低端命名；对常规东西向断层通常对应西端。 |
| `right` | 沿走向另一端。优先按 top/bottom 主方向投影的高端命名；对常规东西向断层通常对应东端。 |

边界点排序约定为：

```text
top:    left -> right
bottom: left -> right
left:   top -> bottom
right:  top -> bottom
```

这个排序约定让边界曲线、边界约束和可视化输出更容易对齐。

## 常用字段

运行边界识别后，断层对象会维护下列字段：

| 字段 | 说明 |
| --- | --- |
| `edge_vertices` | 每条边上的顶点坐标，键通常为 `top/bottom/left/right`。 |
| `edge_vertex_indices` | 每条边上的顶点索引。 |
| `edge_triangles_indices` | 每条边对应的三角单元索引。边界零滑约束通常依赖这个字段。 |
| `edge_triangle_vertex_indices` | 每条边对应三角单元涉及的顶点索引。 |
| `edge_dict` | 底层边界三角形字典，保留给兼容接口和内部工具使用。 |
| `corner_dict` | 角点三角形字典，保留给兼容接口和内部工具使用。 |
| `edge_extraction_info` | topology 后端的诊断信息，包括边界 component、run summary、gap summary、left/right 投影方向等。 |

如果线性滑动反演配置中使用 `zero_edge_slip(...)`，断层对象必须先完成边界识别并具有 `edge_triangles_indices`。

## 拓扑保持几何更新中的缓存使用

边界识别分成两件事：先确定哪些顶点和三角形属于 `top/bottom/left/right`，再从当前 `Vertices` 中取出这些边界的坐标。对于拓扑保持的几何更新，第二步可以重复执行，第一步不需要每次重算。

典型场景是 Bayesian 几何采样中的 `generate_and_deform_mesh` 类流程：如果每个样本只更新顶点坐标，而 `Faces`、顶点编号和三角连接关系保持不变，那么第一次识别得到的 `edge_vertex_indices`、`edge_triangles_indices` 和边界排序仍然有效。此时应复用这些 indices，用新的 `Vertices` 刷新 `edge_vertices` 坐标；不建议在每个样本中 `refind=True` 重新发现四边。

推荐流程：

1. 生成初始网格后，运行一次 topology 边界识别，并记录 `edge_extraction_info`。
2. 如果后续只是变形同一张网格，保留 `edge_vertex_indices` 和 `edge_triangles_indices`。
3. 每次顶点坐标更新后，用缓存的 indices 更新 `edge_vertices`，约束和统计继续使用同一套边界拓扑。
4. 只有网格拓扑改变时，才重新运行 `find_fault_fouredge_vertices(..., refind=True)`。

需要重新发现边界的情况包括：重新 gmsh 剖分、`remap=True`、三角形 `Faces` 改变、顶点编号重排、patch 增删、裁剪/合并/拆分断层段、重采样或修补网格。自动流程中可以用 `Faces.shape`、顶点数和 `Faces` 内容签名做缓存有效性检查；一旦签名变化，就应丢弃旧边界缓存并重新识别。

这个约定对边界零滑很重要：`zero_edge_slip(...)` 实际使用的是 `edge_triangles_indices`。在拓扑保持的几何反演中，边界三角形集合不变，因此不需要因为每个样本的坐标变化而重新计算边界三角形。反复重算不仅浪费时间，还可能因为浅部离群点、gap 清理或容差变化，让不同样本使用了不一致的边界定义。

## 识别方法

当前推荐并默认使用 topology 后端：

```python
edge_method="topology"
gap_policy="clean"
```

topology 方法基于三角网格外边界环：

1. 找出只属于一个三角形的 boundary edge。
2. 将 boundary edge 连接成闭合边界 loop。
3. 按 `top_tolerance` 和 `bottom_tolerance` 标记 `top/bottom/side`。
4. 只把连接 `top` 和 `bottom` 的 `side` run 作为真实左右边。
5. 用 top/bottom 的主方向投影命名 `left/right`。
6. 写入与旧接口兼容的 `edge_vertices`、`edge_triangles_indices` 等字段。

示例：

```python
fault.find_fault_fouredge_vertices(
    top_tolerance=0.1,
    bottom_tolerance=0.1,
    edge_method="topology",
    gap_policy="clean",
)
```

写出 GMT：

```python
fault.writeFourEdges2File(
    dirname="output/stat_infos",
    edge_method="topology",
    gap_policy="clean",
)
```

历史方法仍然保留，用于复现旧结果或排查差异：

```python
fault.find_fault_fouredge_vertices(edge_method="legacy")
```

过渡期如果需要 topology 失败后自动回退旧方法，可以使用：

```python
fault.find_fault_fouredge_vertices(edge_method="auto")
```

`auto` 适合兼容旧案例，但正式反演前更建议显式使用 `topology` 或显式使用 `legacy`，避免 silent fallback 让结果来源不清楚。

## gap_policy

真实网格的 top 或 bottom 边可能有一两个浅部离群点，导致边界 loop 出现：

```text
top -> side -> top
```

这类短片段不是左右边，而是 top gap。`gap_policy` 控制如何处理：

| 策略 | 用途 |
| --- | --- |
| `clean` | 默认推荐。省略短 gap 点，把 canonical top/bottom 写成标准单段，同时保留诊断信息。 |
| `strict` | 自动反演前质量检查。发现 top gap、bottom gap、ambiguous side 或边界 run 数量异常时直接报错。 |
| `diagnostic` | 人工排查。保留多段 top/bottom 结构，便于检查网格边界。 |
| `standardize` | 显式桥接短 gap。只应在人工确认后使用。 |

对于自动化 Bayesian 几何或几何-滑动联合反演，建议先用 `strict` 检查输入网格。如果确认短 gap 可忽略，再用固定、可复现的 `clean` 规则预处理，不要在采样过程中交互决定。

## 可能失败或需要人工检查的场景

topology 后端更稳健，但它依赖一个清楚的几何假设：一个 fault object 表示一个单连通、无孔洞、外边界可分为 `top/bottom/left/right` 的断层段。以下场景需要检查：

| 场景 | 影响 | 建议 |
| --- | --- | --- |
| 多个不连通断层段放在一个 object 中 | 外边界会有多个 component，四边界语义不唯一。 | 拆成多个 fault object，或用 `strict` 先报错。 |
| 网格有孔洞或内部孤岛 | 会出现多个 boundary loop。 | 检查 mesh，必要时先修网格。 |
| 非流形边界、T-junction、重复边或裂口 | 边界节点度数不为 2，loop 无法可靠排序。 | 让 topology 显式失败，修正网格后再提取。 |
| top/bottom 本身不是近似等深边 | 深度阈值会把 top/bottom 切碎。 | 调整 `top_tolerance/bottom_tolerance`，或确认该几何是否适合四边界模型。 |
| 强弯曲、U 形或分叉断层 | left/right 的主方向投影可能不符合直觉。 | 检查 `edge_extraction_info` 中的投影方向和 side projection。 |
| 短 gap 代表真实几何变化 | `clean` 会把它标准化为单段边界。 | 用 `diagnostic` 查看；确认后再选择 `clean` 或 `strict`。 |

显式失败通常比静默给出错误边界更适合反演流程。旧方法可能在问题网格上返回一组看似完整的边界，但 `top/right` 或 `left/bottom` 已经混叠；topology 后端会尽量把这类问题暴露在诊断信息或异常中。

## 与约束管理器的关系

`zero_edge_slip(...)` 固定的是指定边界上的三角单元滑动分量，因此依赖 `edge_triangles_indices`。如果没有先识别边界，约束管理器会报错并提示需要先完成边界识别。

常见边界名：

```text
top
bottom
left
right
```

实际可用边界以断层对象中的 `edge_triangles_indices.keys()` 为准。使用 topology 后端时，建议在批量反演前记录 `fault.edge_extraction_info`，便于之后复查边界 run、gap 处理和 left/right 命名。

如果需要把边界 patch 用于统计、指定 patch 零滑或自定义约束，可用统一 helper 提取：

```python
from eqtools.csiExtend import get_edge_patch_indices

top_ids = get_edge_patch_indices(fault, "top")
side_ids = get_edge_patch_indices(fault, ["left", "right"])
```

更多 patch 子集生成方式见 [Fault Patch Indices](fault_patch_indices.md)。
