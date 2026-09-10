# MeshPolicy、拓扑与派生状态

本页说明几何 pipeline 完成后怎样处理 mesh，以及 GF、Laplacian、面积和拓扑缓存何时失效。

## 方法和 MeshPolicy 的责任

| 方法形式 | mesh 责任 |
| --- | --- |
| 无 mesh 后缀 | 只更新几何，YAML 必须提供适用的 `update_mesh` |
| `_simpleMesh` | 方法内部重建简单 mesh |
| `_DeformMesh` | 方法内部变形已准备的参数化 mesh |
| `_multiLayerMesh` | 方法内部处理多层 mesh |

一个候选只能选择一条 mesh 路由。自带 mesh 的方法不能再让 YAML 执行第二次重建。

```text
sample slice
  -> update_fault_geometry
  -> MeshPolicy（方法内或独立 update_mesh，二选一）
  -> materialize
  -> update GF / GL / area as required
  -> likelihood
```

## Whole-mesh pair

直接作用完整 mesh 的方法要求：

1. reference vertices 和 faces 同时存在并来自同一次最终 mesh；
2. 当前 mesh 保持 vertex shape、face rows、编号、连接关系和绕序；
3. fixed-topology deform 只改 vertices；
4. remesh 后重新建立 reference，再开始新的 run。

第一轮完成完整 pair 检查后，同一 reference 和正式 mesh 发布下的后续候选可复用验证状态，
不会逐候选 hash 整个 Faces。原地写 `fault.Faces[:]` 会绕过发布协议，不是支持用法。

## 参数化固定拓扑

`generate_and_deform_mesh()` 的准备阶段使用 `remap=True` 建立 vertices/faces 与逐顶点参数
映射；Bayesian replay 复用映射，不能在首个候选静默 remap。`num_segments/disct_z` 以及定义
竖向网格的 `bias/min_dz` 必须和准备阶段相容。

标准准备让 `bottom_norm_offset=None`。任何非 `None` 值（包括 `0.0`）都会进入底边扰动调用，
它不是采样器初值。若 offset 属于物理基线，应先修改 current bottom，再 snapshot 和 remap。

## 中央变化映射

| 变化 | GF | Laplacian | 面积 | 邻接/边界 |
| --- | --- | --- | --- | --- |
| `none` | 保留 | 保留 | 保留 | 保留 |
| `rigid` | 失效 | 保留 | 保留 | 保留 |
| `deform` | 失效 | 失效 | 失效 | 保留拓扑索引关系 |
| `remesh` | 失效 | 失效 | 失效 | 全部失效 |

`none` 表示没有发布新候选 mesh，不承诺“数值增量恰为零”自动走零成本比较。`deform` 虽可
保留 Faces 和 patch 顺序，但 Laplacian 数值依赖 patch 距离；面积也随形状变化。二者只有
在当前目标真正需要时才物化，不因每次几何调用无条件重算。

rigid 保持内部距离和面积，因此 Laplacian/area 可复用；GF 因绝对空间位置改变仍需更新。
remesh 还必须重新检查 slip/patch 参数映射，不能只刷新绘图边界。

## 方法场景

`SMC_FJ` 与 `FULLSMC` 共享候选刷新和变化映射。BLSE/VCE 则要求先完成固定几何，再创建
新的 inversion；同一 BLSE/VCE run 的平滑或方差分量循环不会隐式重建 GF、Laplacian、
参数布局和约束。

## 采样前检查

- 极值候选仍生成有效 mesh；
- 固定拓扑场景 patch 数量、Faces 和参数顺序不变；
- remesh 场景的 slip layout 与约束同步重建；
- rigid/deform/remesh 的报告符合真实操作；
- GF、Laplacian 和面积只在对应状态失效且消费者需要时更新。
