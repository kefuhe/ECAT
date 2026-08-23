# Bayesian 联合反演中的几何参考

联合 Bayesian 反演不会在上一粒子的断层上继续累加扰动。每个样本都从同一份冻结的
`GeometryReference` 出发，生成本样本的候选几何，再更新网格、Green's functions
和线性滑动子问题。

核心不变量是：

```text
candidate_i = transform(reference, delta_i)
```

而不是：

```text
candidate_i = transform(candidate_(i-1), delta_i)
```

这保证了样本顺序不会改变几何含义，也避免长链中出现累计漂移。

## 一个样本从配置到似然的完整流转

普通用户只选择公开几何扰动方法、该方法在样本向量中的切片、参数 bounds 和公开
kwargs。方法内部怎样拆分几何操作、选择节点和安排顺序，由已注册方法本身定义，不需要
也不应在 YAML 中拼装内部 stage。

```mermaid
flowchart TD
    A["YAML / Python<br/>选择公开几何方法和样本切片"] --> B["target 构造前预检<br/>方法、kwargs、reference、参数个数"]
    B --> B1["按 mesh 方法契约检查<br/>所需 fixed-topology replay 状态"]
    B1 --> C["从冻结 GeometryReference<br/>复制本样本候选状态"]
    C --> D["按该方法声明的顺序<br/>施加几何操作"]
    D --> E{"方法是否自行生成 mesh？"}
    E -- "否" --> F["由配置的 update_mesh<br/>生成或更新候选 mesh"]
    E -- "是" --> G["方法内部的 mesh policy<br/>生成候选 mesh"]
    F --> H["统一发布当前 fault 几何与 mesh"]
    G --> H
    H --> I["为候选几何更新 Green's functions"]
    I --> J["仅在失效且确有消费者时<br/>更新 Laplacian / 面积"]
    J --> K["求解滑动并计算 likelihood"]
```

这里的“发布”表示把候选 `top/bottom/layers` 或 `Vertices/Faces` 变成当前 fault 的权威
状态，并同步其 patches 与必要缓存。用户不手工设置 GF、Laplacian 或面积的有效性标志；
反演层依据几何变化性质和本次计算实际需要统一调度。`SMC_FJ` 与 `FULLSMC` 使用同一套
候选几何刷新链，差别在采样和似然组织，不在 reference 或 mesh 发布契约。MPI 下每个
rank 维护自己的候选和缓存，pipeline 不为每个候选增加额外的跨 rank 通信。

mesh replay 检查按注册方法的明确契约触发，不从方法后缀或已有数组名称猜测。
`generate_and_deform_mesh` 需要采样前准备的逐顶点参数映射，因此 target 构造会只读核对
映射与当前固定拓扑及重放参数；其他 mesh 路径没有声明该依赖时不会参与。该核对每次
target 构造执行一次，不放入单个样本的 pipeline，也不会替用户自动 remap。

## 四个容易混淆的状态

| 状态 | 典型字段或对象 | 生命周期 | 是否应由用户直接修改 |
| --- | --- | --- | --- |
| 构建中的当前几何 | `top_coords`、`bottom_coords`、`layers`、`Vertices/Faces` | 采样前构建和检查 | 可以，通过公开建模接口修改 |
| 冻结参考 | `fault.geometry_ref` / `GeometryReference` | 一次 inversion run 的零扰动基线 | 不直接改字段；用 fault 的公开接口生成新参考 |
| 样本状态 | `GeometryState` | 单个样本内部 | 不需要；由 pipeline 创建 |
| 候选当前几何 | fault 上物化后的坐标、mesh 和 patches | 单个样本内部 | 不需要；由 pipeline 更新 |

`GeometryReference` 是不可变值对象。`with_dip()`、`with_layers()`、
`with_vertices()` 和 `with_densification()` 会返回新对象；普通用户优先调用断层对象上的
`snapshot()`、`set_dip_control_points*()` 和 `set_densification()`，不要直接给
`geometry_ref` 的字段赋值。

## `snapshot()` 到底保存了什么

`snapshot()` 是一次**状态捕获**，不是一次 mesh 重建。调用时，它会把当前断层对象上已
完成检查的 top/bottom 等字段复制到新的 `GeometryReference`，把其中的数组设为只读，
再同时赋给 `fault.geometry_ref` 并作为返回值返回：

```python
ref = fault.snapshot(
    capture_vertices=False,
    capture_layers=False,
)

assert ref is fault.geometry_ref
assert not ref.top_coords.flags.writeable
```

参考数组和当前建模数组相互独立。之后改变 `fault.top_coords`、`fault.bottom_coords` 或
mesh，不会反向改变已冻结的 reference。`snapshot()` 本身也不会修改当前边界、生成 mesh、
改写先验或写入 YAML/结果文件；构建脚本需要在每次新运行中重新建立同一份参考。
在 MPI 运行中，每个 rank 都有自己的进程内 reference；它们应由相同输入和相同调用顺序
确定性建立，而不是假定一个普通 Python 对象会自动跨进程共享。

采样时，扰动 pipeline 从 `fault.geometry_ref` 复制出单样本 `GeometryState`，施加当前
样本的增量，再把候选边界或 mesh 物化到 fault 上。候选状态只服务于当前样本，不会回写
reference。因此 reference 服务的是“为所有样本定义共同零点”，而不是保存每个样本的
历史。

## 参考不是先验，也不是参数位置

以下四项共同定义一次几何采样，但作用不同：

| 项目 | 回答的问题 |
| --- | --- |
| `geometry_ref` | 零扰动时的断层是什么样？ |
| `geometry.sample_positions` | 全局样本向量的哪一段属于该断层？ |
| `update_fault_geometry.method` | 怎样把这段样本增量施加到参考几何？ |
| bounds 文件中的 `geometry` | 这些增量允许落在什么范围？ |

例如 `sample_positions: [0, 1]` 表示使用半开区间 `[0, 1)`，也就是一个几何采样量；
它不表示第 0 和第 1 个断层节点。一个标量是否广播到多个底边节点，由所选扰动方法定义。

## 参考应包含什么

`GeometryReference` 至少服务于所选扰动方法。它可以包含：

- `top_coords` 和 `bottom_coords`：边界坐标类扰动的基础。
- `layers`：多层断层和 layered mesh 的基础。
- `vertices` 和 `faces`：整体平移、旋转等直接变换现有 mesh 的方法所需。
- `dip_control_points`：沿走向变化倾角的参考控制点。
- `densification`：稀疏采样控制点到密集网格边界的加密策略。

不是所有场景都要捕获全部字段。只移动底边坐标、随后单独重建 mesh 时，
`snapshot(capture_vertices=False, capture_layers=False)` 已足够；直接变换整个 mesh 时，
必须在最终参考 mesh 建好后使用
`snapshot(capture_vertices=True, capture_layers=False)`。多层方法才需要捕获 `layers`。

对直接操作整张三角网格的方法，`vertices` 和 `faces` 是一个不可拆分的参考 pair：它们必须
来自同一次最终 mesh 捕获。只有其中一个字段、或者冻结后又改变了当前 Faces 的 row、编号、
绕序或连接关系，都不能再解释为同一基线。此时应先完成最终 remesh 和必要检查，再重新
`snapshot(capture_vertices=True, ...)`；系统不会把 frozen vertices 与当前 Faces 拼在一起。
这个约束只属于 whole-mesh 方法，不会把只需 top/bottom、layers 或 dip controls 的最小
reference 变成非法状态。

## 一个样本内部怎样组合多个操作

每个候选只有一份临时 `GeometryState`。同一候选中的坐标 stage 按声明顺序连续作用在这份
状态上，最后再由 mesh policy 生成或更新 mesh，并一次性写回 fault：

```text
GeometryReference
  -> candidate GeometryState
  -> stage 1 -> stage 2 -> ...
  -> mesh policy
  -> 一次 materialize
```

因此组合旋转和平移的含义是 `translate(rotate(reference))`，而不是每一步都重新读取
reference，也不是从上一样本继续累加。当前公开 pipeline 支持“坐标 stages → mesh policy”
和直接 whole-mesh 变换；“先由控制点生成新 mesh，再对该候选 mesh 做额外顶点变换”尚不是
通用公开配置。不要把这两种基线来源手工混在一个样本里；需要这种 mixed 流程时，应使用
有明确阶段契约的专用 composite 方法。

## 入口由权威状态决定

“曲线断层”不是一个 reference 入口。应先回答：哪一份状态才是经过科学检查、需要定义
零扰动几何的权威输入？

```text
最终 top/bottom 已由迹线、倾角或外部坐标明确确定？
├─ 是：直接 snapshot；不先生成临时 mesh
└─ 否：权威边界是否只存在于已导入或修整后的 mesh？
   ├─ 是：从 mesh 提取边界并建立 reference
   └─ 否：先完成几何构建，不能让 snapshot 猜测基线
```

只有直接变换 mesh 顶点的方法才要求“先建最终 mesh，再捕获 vertices/faces”。边界坐标类
扰动通常应先冻结 top/bottom，再由同一组边界生成一次参数化 mesh。

## 何时建立或重新定义参考

推荐在以下时机建立参考：

1. 准备冻结的 top/bottom，以及所需的 layer、dip control 或 mesh 字段已完成科学检查。
2. 点序、下倾侧、深度和坐标单位已经确定；若方法直接变换 mesh，拓扑和网格尺度也已确定。
3. 即将创建 `BayesianMultiFaultsInversion` 或开始一次新的独立采样运行。

只在以下场景重新定义参考：

- 修正了输入迹线、边界、倾角控制点或初始 mesh，并准备重新开始一次 inversion run。
- 上一次反演得到新的代表几何，明确要把它作为下一次独立反演的零扰动中心。
- 做阶段化敏感性分析，每个阶段有清楚记录的不同基线。

不要在 SMC 目标函数、样本循环、单个 stage 内或恢复同一次 run 时调用
`snapshot()`。否则参数边界仍表示旧基线的增量，而样本实际围绕新基线解释，后验会失去
一致含义。进程重启后恢复同一次 run 时，应从原输入重建语义相同的 reference，不能把
中断时最后一个候选几何冻结成新零点。

## 两种 snapshot 不同

```python
fault.snapshot(
    capture_vertices=False,
    capture_layers=False,
)
```

冻结的是断层几何参考。

```python
snapshot = inversion.get_constraint_snapshot(validate=True)
```

返回的是 bounds、线性约束及其验证状态的诊断副本。它不会建立或刷新断层几何参考。

## 继续阅读

- [联合 Bayesian workflow](../workflows/05_joint_bayesian_geometry_slip.md)
- [联合几何设置短例](../examples/joint_bayesian_geometry_setup.md)
- [可扰动断层几何参考](../reference/geometry_perturbation.md)
- [断层几何状态](fault_geometry_states.md)
